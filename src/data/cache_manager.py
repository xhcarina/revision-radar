"""
Cache lifecycle management for Revision Radar.

Responsibilities:
  1. Auto-purge stale HTTP + LLM cache files on app startup (TTL = CACHE_TTL_HOURS).
  2. Save / load named analysis snapshots to disk (saved_reports/).
     Snapshots use pickle so all Pydantic models, DataFrames, and dataclasses
     round-trip without any manual serialization.

Usage:
    from src.data.cache_manager import purge_stale_cache, save_report, load_report, list_reports
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CACHE_DIR, CACHE_TTL_HOURS

REPORTS_DIR = Path("saved_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Auto-purge ────────────────────────────────────────────────────────────────

def purge_stale_cache() -> int:
    """
    Delete any cache file whose timestamp is older than CACHE_TTL_HOURS.
    Returns the number of files removed.

    Safe to call at every app startup — fast (one directory scan, JSON header read).
    Corrupted or unrecognized files are left untouched.
    """
    if not CACHE_DIR.exists():
        return 0

    cutoff = datetime.now() - timedelta(hours=CACHE_TTL_HOURS)
    removed = 0
    for f in CACHE_DIR.iterdir():
        if not f.is_file() or f.suffix != ".json":
            continue
        try:
            data = json.loads(f.read_text())
            ts = datetime.fromisoformat(data["ts"])
            if ts < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            # Corrupt / unknown format — skip, don't delete
            pass
    return removed


# ── Saved reports ─────────────────────────────────────────────────────────────

def _safe_stem(name: str) -> str:
    """Sanitize a report name to a filesystem-safe string."""
    return re.sub(r"[^\w\-]", "_", name.strip())[:80]


def save_report(results: dict, name: str) -> Path:
    """
    Pickle a full analysis result dict to saved_reports/<name>.pkl.
    Overwrites any existing report with the same name.
    Returns the path written.
    """
    stem = _safe_stem(name)
    if not stem:
        raise ValueError("Report name cannot be empty.")
    path = REPORTS_DIR / f"{stem}.pkl"
    meta = {
        "name":      name,
        "ticker":    results.get("ticker", ""),
        "company":   results.get("company_name", ""),
        "saved_at":  datetime.now().isoformat(),
        "results":   results,
    }
    path.write_bytes(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def load_report(stem: str) -> Optional[dict]:
    """
    Load a saved report by its stem (filename without .pkl).
    Returns the full results dict, or None if the file is missing / corrupt.
    """
    path = REPORTS_DIR / f"{stem}.pkl"
    if not path.exists():
        return None
    try:
        meta = pickle.loads(path.read_bytes())
        return meta["results"]
    except Exception:
        return None


def list_reports() -> list[dict]:
    """
    Return metadata for all saved reports, newest first.
    Each entry: {stem, name, ticker, company, saved_at}
    """
    entries = []
    for f in sorted(REPORTS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = pickle.loads(f.read_bytes())
            entries.append({
                "stem":     f.stem,
                "name":     meta.get("name", f.stem),
                "ticker":   meta.get("ticker", ""),
                "company":  meta.get("company", ""),
                "saved_at": meta.get("saved_at", ""),
            })
        except Exception:
            entries.append({"stem": f.stem, "name": f.stem,
                            "ticker": "", "company": "", "saved_at": ""})
    return entries


def delete_report(stem: str) -> bool:
    """Delete a saved report. Returns True if deleted, False if not found."""
    path = REPORTS_DIR / f"{stem}.pkl"
    if path.exists():
        path.unlink()
        return True
    return False
