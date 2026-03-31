"""
Append-only event log for all documents ingested by the pipeline.

Each line in cache/event_log.jsonl is one JSON record describing a single
document fetch: source, ticker, URL, publication_ts, content_hash, and the
path to its Layer-2 processed cache entry.

Guarantees:
  - Append-only: existing lines are never modified or deleted.
  - Deduplication: (url, publication_ts) pair and content_hash are both checked
    before writing; duplicates are silently skipped.
  - Concurrent-write safety: writes are protected by fcntl.flock (exclusive
    lock on the file descriptor before writing, released immediately after).
  - Fast startup: on load, two in-memory sets are built so dedup checks are O(1).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import CACHE_DIR

logger = logging.getLogger(__name__)

EVENT_LOG_PATH = CACHE_DIR / "event_log.jsonl"

GDELT_FILTER_VERSION = "gdelt_v1"   # bump when GDELT processing logic changes


# ── EventLog class ────────────────────────────────────────────────────────────

class EventLog:
    """
    Thin wrapper around an append-only JSONL file.

    Usage (module-level singleton):
        from event_log import event_log
        event_log.append(source="edgar_8k", ticker="AAPL", ...)
        records = event_log.query("AAPL", source="edgar_8k", as_of="2024-03-15")
    """

    def __init__(self, path: Path = EVENT_LOG_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory dedup sets — populated at startup from any existing log
        self._seen_url_ts: set[tuple[str, str]] = set()
        self._seen_hashes: set[str] = set()
        self._load()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Build in-memory dedup sets from the existing log file (if any)."""
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                url    = rec.get("url", "")
                pub_ts = rec.get("publication_ts") or ""
                ch     = rec.get("content_hash", "")
                if url and pub_ts:
                    self._seen_url_ts.add((url, pub_ts))
                if ch:
                    self._seen_hashes.add(ch)

    # ── Public API ────────────────────────────────────────────────────────────

    def append(
        self,
        *,
        source: str,
        ticker: str,
        url: str,
        publication_ts: Optional[str],
        fetched_at: str,
        content_hash: str,
        filter_version: str,
        extracted_text_path: str,
    ) -> bool:
        """
        Append one event record to the log.

        Returns True if the record was written, False if it was a duplicate.

        Dedup order:
          1. (url, publication_ts) match → skip
          2. content_hash match          → skip
          3. Otherwise                   → write
        """
        pub_ts = publication_ts or ""

        # Dedup check 1: url + publication_ts
        if pub_ts and (url, pub_ts) in self._seen_url_ts:
            logger.debug("EventLog skip dup url+ts  %s | %s", url[:60], pub_ts)
            return False

        # Dedup check 2: content_hash
        if content_hash and content_hash in self._seen_hashes:
            logger.debug("EventLog skip dup hash %s", content_hash[:16])
            return False

        record = {
            "event_id":            str(uuid.uuid4()),
            "source":              source,
            "ticker":              ticker,
            "url":                 url,
            "publication_ts":      publication_ts,   # None stored as JSON null
            "fetched_at":          fetched_at,
            "content_hash":        content_hash,
            "filter_version":      filter_version,
            "extracted_text_path": extracted_text_path,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        # Exclusive file lock so concurrent processes don't interleave bytes
        with open(self._path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

        # Update in-memory sets *after* successful write
        if pub_ts:
            self._seen_url_ts.add((url, pub_ts))
        if content_hash:
            self._seen_hashes.add(content_hash)

        return True

    def query(
        self,
        ticker: str,
        source: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> list[dict]:
        """
        Return matching event records, sorted by publication_ts ascending.

        Filters:
          - ticker   : exact match
          - source   : exact match if provided ("edgar_8k" | "edgar_10q" | "gdelt")
          - as_of    : publication_ts <= as_of (point-in-time filter, ISO 8601 date)

        Each returned record gains an 'extracted_text' convenience field:
          - For SEC L2 files (JSON with 'extracted_text' key): the plain text value.
          - For GDELT batch files (JSON with 'records' key): the raw JSON string.
          - None if the file is missing or unreadable.
        """
        results: list[dict] = []

        if not self._path.exists():
            return results

        # Parse as_of once
        as_of_dt: Optional[datetime] = None
        if as_of:
            try:
                as_of_dt = datetime.fromisoformat(as_of[:10])
            except ValueError:
                pass

        with open(self._path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Ticker filter
                if rec.get("ticker") != ticker:
                    continue

                # Source filter
                if source and rec.get("source") != source:
                    continue

                # Point-in-time filter: publication_ts <= as_of
                if as_of_dt is not None:
                    pub_ts = rec.get("publication_ts")
                    if pub_ts:
                        try:
                            rec_dt = datetime.fromisoformat(pub_ts[:10])
                            if rec_dt > as_of_dt:
                                continue
                        except ValueError:
                            pass
                    # Records with null publication_ts pass through (unknown date)

                # Attach extracted_text
                rec = dict(rec)
                ext_path = rec.get("extracted_text_path", "")
                rec["extracted_text"] = _read_extracted_text(ext_path)

                results.append(rec)

        results.sort(key=lambda r: r.get("publication_ts") or "")
        return results


# ── Extracted-text reader ─────────────────────────────────────────────────────

def _read_extracted_text(path: str) -> Optional[str]:
    """
    Read the processed text from a Layer-2 cache file path.

    SEC L2 files are JSON: {"extracted_text": "...", ...}  → return the text value.
    GDELT batch files are JSON: {"ts": ..., "records": [...]} → return raw JSON.
    Missing or unreadable files → return None.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = p.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        # SEC L2: has 'extracted_text' key → return the plain text
        if "extracted_text" in parsed:
            return parsed["extracted_text"]
        # GDELT batch: has 'records' key → return the raw JSON for caller to parse
        return raw
    except Exception:
        return None


# ── Module-level singleton ────────────────────────────────────────────────────

event_log = EventLog()


# ── CLI utility ───────────────────────────────────────────────────────────────

def _cli_summary(ticker: str) -> None:
    """Print a quality summary for one ticker."""
    if not EVENT_LOG_PATH.exists():
        print(f"No event log found at {EVENT_LOG_PATH}")
        return

    # Load all records for this ticker (no as_of filter)
    records = event_log.query(ticker, source=None, as_of=None)
    if not records:
        print(f"No records found for ticker '{ticker}'")
        return

    # Group by source
    by_source: dict[str, list[dict]] = {}
    null_pub_ts_total = 0

    for rec in records:
        src = rec.get("source", "unknown")
        by_source.setdefault(src, []).append(rec)
        if rec.get("publication_ts") is None:
            null_pub_ts_total += 1

    print(f"\n{'─'*60}")
    print(f"  Event Log Summary — {ticker}")
    print(f"{'─'*60}")
    print(f"  Total records : {len(records)}")
    print(f"  Null pub_ts   : {null_pub_ts_total}  (data quality flag)\n")

    for src, recs in sorted(by_source.items()):
        ts_values = sorted(
            r["publication_ts"] for r in recs if r.get("publication_ts")
        )
        earliest = ts_values[0][:10]  if ts_values else "N/A"
        latest   = ts_values[-1][:10] if ts_values else "N/A"
        null_cnt = sum(1 for r in recs if not r.get("publication_ts"))

        print(f"  Source: {src}")
        print(f"    Count    : {len(recs)}")
        print(f"    Earliest : {earliest}")
        print(f"    Latest   : {latest}")
        if null_cnt:
            print(f"    Null ts  : {null_cnt}")

        # Gap analysis — find consecutive gaps > 45 days
        if len(ts_values) >= 2:
            gaps = []
            for i in range(1, len(ts_values)):
                try:
                    d0 = datetime.fromisoformat(ts_values[i - 1][:10])
                    d1 = datetime.fromisoformat(ts_values[i][:10])
                    delta = (d1 - d0).days
                    if delta > 45:
                        gaps.append((ts_values[i - 1][:10], ts_values[i][:10], delta))
                except ValueError:
                    continue
            if gaps:
                print(f"    Gaps > 45d:")
                for g_start, g_end, g_days in gaps:
                    print(f"      {g_start} → {g_end}  ({g_days} days)")
        print()

    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Revision Radar event log utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    summary_parser = sub.add_parser("summary", help="Print quality summary for a ticker")
    summary_parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")

    args = parser.parse_args()

    if args.command == "summary":
        _cli_summary(args.ticker.upper())
    else:
        parser.print_help()
