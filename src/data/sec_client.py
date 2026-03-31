"""
SEC EDGAR client — fetches 8-K and 10-Q filings for a given ticker.

Strategy:
  1. Resolve ticker → CIK via the EDGAR company_tickers.json map.
  2. Fetch the submission metadata JSON to list recent filings.
  3. Download filing index pages and pull the largest text exhibit
     (EX-99.1 for press releases, or raw 8-K text for transcripts).
  4. For 10-Q, extract the MD&A section via simple heuristic splitting.

Caching is split into two layers:
  Layer 1 — raw/edgar/      : raw HTML exactly as received from EDGAR.
  Layer 2 — processed/edgar/: clean plaintext after BeautifulSoup stripping.
             Keyed on (url, FILTER_VERSION) so filter logic changes invalidate
             processed output without re-hitting EDGAR.
JSON API responses (submission metadata, company tickers) use the legacy
single-layer cache — they are structured data, not HTML, so the split
does not apply.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CACHE_DIR, CACHE_TTL_HOURS, EDGAR_BASE, EDGAR_HEADERS
from event_log import event_log as _event_log

# Increment when BeautifulSoup stripping or regex cleanup logic changes.
# This invalidates all Layer 2 entries without touching Layer 1 raw HTML.
FILTER_VERSION = "v1.0"

# ── Cache directory layout ────────────────────────────────────────────────────
_L1_DIR = CACHE_DIR / "raw"       / "edgar"                   # raw HTML
_L2_DIR = CACHE_DIR / "processed" / "edgar" / FILTER_VERSION  # clean text
_L1_DIR.mkdir(parents=True, exist_ok=True)
_L2_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CIK ticker map is fetched once and cached
_CIK_CACHE: dict[str, str] = {}

# ─────────────────────────── JSON API cache (legacy single-layer) ─────────────
# Used for structured EDGAR endpoints (submission metadata, company_tickers).
# These are not HTML so the raw/processed split does not apply.

def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^\w_-]", "_", key)
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str) -> Optional[dict | list | str]:
    p = _cache_path(key)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if datetime.fromisoformat(data["ts"]) < datetime.now() - timedelta(hours=CACHE_TTL_HOURS):
        return None
    return data["payload"]


def _save_cache(key: str, payload: dict | list | str) -> None:
    p = _cache_path(key)
    p.write_text(json.dumps({"ts": datetime.now().isoformat(), "payload": payload}))


def _get(url: str, cache_key: Optional[str] = None, retries: int = 3) -> dict | str:
    if cache_key:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            result = resp.json() if "json" in content_type else resp.text
            if cache_key:
                _save_cache(cache_key, result)
            return result
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ─────────────────────────── Two-layer HTML cache ────────────────────────────

def _url_key(url: str) -> str:
    """Stable filename key for a URL — SHA-256 truncated to 40 hex chars."""
    return hashlib.sha256(url.encode()).hexdigest()[:40]


def _l1_path(url: str) -> Path:
    return _L1_DIR / f"{_url_key(url)}.json"


def _l2_path(url: str) -> Path:
    return _L2_DIR / f"{_url_key(url)}.json"


def _normalize_pub_ts(date_str: Optional[str]) -> Optional[str]:
    """
    Normalize an EDGAR filingDate (YYYY-MM-DD) to ISO 8601 with UTC suffix.
    Returns None — never raises — if the input is missing or unparseable.
    """
    if not date_str:
        return None
    try:
        datetime.strptime(date_str[:10], "%Y-%m-%d")   # validate format
        return f"{date_str[:10]}T00:00:00Z"
    except (ValueError, TypeError):
        return None


def _l1_read(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Return (raw_html, publication_ts) from Layer 1.
    Both are None if the entry does not exist.
    """
    p = _l1_path(url)
    if not p.exists():
        return None, None
    entry = json.loads(p.read_text())
    return entry.get("raw_html"), entry.get("publication_ts", None)


def _l1_write(url: str, raw_html: str, publication_ts: Optional[str]) -> None:
    """
    Persist raw HTML to Layer 1.

    Rules:
    - If the entry does not exist: write all fields.
    - If it exists with a non-null publication_ts: preserve it (never overwrite
      with a potentially-null value from a later refresh). Update fetched_at.
    - If it exists with a null publication_ts and we now have one: write it.
    """
    p = _l1_path(url)
    now_ts = datetime.now().isoformat()

    if not p.exists():
        p.write_text(json.dumps({
            "raw_html":       raw_html,
            "fetched_at":     now_ts,
            "publication_ts": publication_ts,   # may be null — stored explicitly
        }))
        return

    # Entry exists — preserve non-null publication_ts
    existing = json.loads(p.read_text())
    existing_pub_ts = existing.get("publication_ts")
    p.write_text(json.dumps({
        "raw_html":       raw_html,
        "fetched_at":     now_ts,
        "publication_ts": existing_pub_ts if existing_pub_ts is not None else publication_ts,
    }))


def _l2_read(url: str) -> Optional[str]:
    """Return processed text from Layer 2, or None if not cached."""
    p = _l2_path(url)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("extracted_text")


def _l2_write(url: str, extracted_text: str, publication_ts: Optional[str]) -> None:
    """Write processed text and publication_ts to Layer 2."""
    _l2_path(url).write_text(json.dumps({
        "extracted_text": extracted_text,
        "processed_at":   datetime.now().isoformat(),
        "filter_version": FILTER_VERSION,
        "publication_ts": publication_ts,   # copied from L1 — never re-derived
    }))


def _fetch_raw(
    url: str,
    publication_ts: Optional[str],
    retries: int = 3,
    ticker: str = "",
    source: str = "edgar_8k",
) -> str:
    """Fetch raw HTML from network with retry. Writes to Layer 1 and event log."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
            resp.raise_for_status()
            raw = resp.text if isinstance(resp.text, str) else json.dumps(resp.json())
            _l1_write(url, raw, publication_ts)
            # Log to event log after every successful L1 write
            _event_log.append(
                source=source,
                ticker=ticker,
                url=url,
                publication_ts=publication_ts,
                fetched_at=datetime.now().isoformat(),
                content_hash=hashlib.sha256(raw.encode()).hexdigest(),
                filter_version=FILTER_VERSION,
                extracted_text_path=str(_l2_path(url)),
            )
            return raw
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ─────────────────────────── CIK resolution ──────────────────────────────────

def get_cik(ticker: str) -> str:
    """Return zero-padded 10-digit CIK for *ticker*."""
    global _CIK_CACHE
    ticker = ticker.upper().strip()
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    data = _get(
        "https://www.sec.gov/files/company_tickers.json",
        cache_key="company_tickers_map",
    )
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected response from SEC company tickers endpoint.")
    for entry in data.values():
        try:
            if str(entry.get("ticker", "")).upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker] = cik
                return cik
        except (KeyError, TypeError):
            continue
    raise ValueError(f"Ticker '{ticker}' not found on SEC EDGAR. Check the symbol is correct.")


# ─────────────────────────── Filing index ────────────────────────────────────

def _submission_data(cik: str) -> dict:
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    return _get(url, cache_key=f"submissions_{cik}")


def list_filings(
    ticker: str,
    form: str,
    limit: int = 10,
    earnings_only: bool = False,
) -> list[dict]:
    """
    Return a list of filing metadata dicts.

    If earnings_only=True and form='8-K', returns only filings with
    item 2.02 (Results of Operations) — i.e. actual earnings releases.
    """
    cik = get_cik(ticker)
    data = _submission_data(cik)
    recent = data.get("filings", {}).get("recent", {})

    forms        = recent.get("form", [])
    acc_nums     = recent.get("accessionNumber", [])
    dates        = recent.get("filingDate", [])
    doc_names    = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])
    items_list   = recent.get("items", [""] * len(forms))

    results = []
    for form_type, acc, date, doc, rdate, items in zip(
        forms, acc_nums, dates, doc_names, report_dates, items_list
    ):
        if not form_type.startswith(form):
            continue
        # For 8-K, optionally filter to earnings releases only (item 2.02)
        if earnings_only and form == "8-K" and "2.02" not in str(items):
            continue
        results.append({
            "accessionNumber": acc,
            "filingDate":      date,
            "reportDate":      rdate,
            "primaryDocument": doc,
            "items":           items,
            "cik":             cik,
        })
        if len(results) >= limit:
            break
    return results


# ─────────────────────────── Document retrieval ───────────────────────────────

def _filing_index_url(cik: str, accession: str) -> str:
    acc_no_dash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik or 0)}/{acc_no_dash}/{accession}-index.htm"


def _filing_document_urls(cik: str, accession: str) -> list[dict]:
    """
    Return list of {url, name, type, size} for all documents in a filing.
    Uses the EDGAR filing index HTML page (the -index.htm endpoint is reliable).
    """
    acc_no_dash = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik or 0)}/{acc_no_dash}"
    index_url = f"{base}/{accession}-index.htm"
    cache_key = f"idx_{accession}"

    try:
        raw = _get(index_url, cache_key=cache_key)
        if not isinstance(raw, str):
            return []
        soup = BeautifulSoup(raw, "html.parser")
        docs = []
        # The filing index has a table with document entries
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            # Columns: Seq, Description, Document, Type, Size
            links = row.find_all("a", href=True)
            if not links:
                continue
            href = links[0]["href"]
            if not href.startswith("/Archives/"):
                continue
            name = href.split("/")[-1]
            # Determine type from the row cells
            doc_type = cells[-2].get_text(strip=True) if len(cells) >= 4 else ""
            size_str = cells[-1].get_text(strip=True).replace(",", "").replace(" KB", "000") if len(cells) >= 5 else "0"
            try:
                size = int(size_str)
            except ValueError:
                size = 0
            docs.append({
                "url":  f"https://www.sec.gov{href}",
                "name": name,
                "type": doc_type,
                "size": size,
            })
        return docs
    except Exception:
        return []


def _process_html(raw_html: str) -> str:
    """
    Pure function: raw HTML → clean plaintext.
    No I/O — safe to call independently of the cache layer.
    Increment FILTER_VERSION when this logic changes.
    """
    if not isinstance(raw_html, str):
        raw_html = json.dumps(raw_html)
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "table", "ix:header", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Strip EDGAR inline file header before real content
    text = re.sub(
        r"^(?:EX-\d+\.\d+\s+\d+\s+\S+\.(?:htm|txt|html)\s+EX-\d+\.\d+\s+(?:Exhibit\s+\d+\.\d+\s+)?(?:Document\s+)?)",
        "", text, flags=re.S,
    )
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def _fetch_text(
    url: str,
    publication_ts: Optional[str] = None,
    ticker: str = "",
    source: str = "edgar_8k",
) -> str:
    """
    Return clean plaintext for an EDGAR document URL.

    Cache lookup order:
      1. Layer 2 (processed) — return immediately if filter_version matches.
      2. Layer 1 (raw HTML)  — reprocess through _process_html, write to L2.
      3. Network             — fetch, write to L1, process, write to L2.

    publication_ts is the source-reported publish date (from filingDate).
    It is written into L2; when reprocessing from L1, it is copied from the
    L1 entry rather than re-derived.
    ticker/source are passed through to the event log on network fetches.
    """
    # Layer 2 hit — processed text already exists for this filter version
    text = _l2_read(url)
    if text is not None:
        return text

    # Layer 1 hit — raw HTML cached; reprocess without network call
    raw_html, l1_pub_ts = _l1_read(url)
    if raw_html is None:
        # Layer 1 miss — fetch from network, writing publication_ts to L1
        raw_html = _fetch_raw(url, publication_ts, ticker=ticker, source=source)
        l1_pub_ts = publication_ts  # L1 now holds it

    # Copy publication_ts from L1 entry (preferred) over caller-supplied value
    effective_pub_ts = l1_pub_ts if l1_pub_ts is not None else publication_ts

    text = _process_html(raw_html)
    _l2_write(url, text, effective_pub_ts)
    return text


# ─────────────────────────── 8-K (Earnings release / transcript) ─────────────

def _replay_from_log(
    ticker: str,
    source: str,
    as_of: str,
    mda: bool = False,
) -> list[dict]:
    """
    Reconstruct fetch results from the event log for point-in-time analysis.
    Used by fetch_eight_k / fetch_ten_q when as_of is provided.
    """
    records = _event_log.query(ticker, source=source, as_of=as_of)
    results = []
    for r in records:
        text = r.get("extracted_text") or ""
        pub_ts = r.get("publication_ts") or ""
        filing_date = pub_ts[:10] if pub_ts else ""
        if mda:
            mda_text = _extract_mda(text)
            results.append({
                "filingDate": filing_date,
                "reportDate": filing_date,
                "mda_text":   mda_text,
                "full_text":  text[:3000],
                "source_url": r.get("url", ""),
                "form":       "10-Q",
                "quarter":    _quarter_label(filing_date),
            })
        else:
            results.append({
                "filingDate": filing_date,
                "reportDate": filing_date,
                "text":       text,
                "source_url": r.get("url", ""),
                "form":       "8-K",
                "quarter":    _quarter_label(filing_date),
            })
    return results


def _best_exhibit(docs: list[dict]) -> Optional[str]:
    """Pick the best text exhibit URL from a filing's document list."""
    # Prefer EX-99.1 > EX-99.2 > primary .htm/.txt with largest size
    for target_type in ("EX-99.1", "EX-99.2"):
        for doc in docs:
            if doc["type"] == target_type and doc["name"].endswith((".htm", ".txt", ".html")):
                return doc["url"]
    # Fallback: largest .htm document
    htm_docs = [d for d in docs if d["name"].endswith((".htm", ".html")) and d["size"] > 1000]
    if htm_docs:
        return max(htm_docs, key=lambda d: d["size"])["url"]
    return None


def fetch_eight_k(ticker: str, limit: int = 6, as_of: Optional[str] = None) -> list[dict]:
    """
    Fetch recent earnings 8-K filings (item 2.02 only) with full text.
    Falls back to 6-K for foreign private issuers (e.g. TSM, ASML) that
    file earnings on EDGAR under form 6-K instead of 8-K.
    Returns list of dicts with: filingDate, reportDate, text, source_url.

    When as_of is provided, returns only event-log records with
    publication_ts <= as_of (point-in-time replay; no network calls).
    """
    if as_of is not None:
        return _replay_from_log(ticker, "edgar_8k", as_of)

    # earnings_only filters to item 2.02 (Results of Operations) — the true earnings releases
    filings = list_filings(ticker, "8-K", limit=limit, earnings_only=True)

    # Foreign filers use 6-K for earnings releases; fall back when no 8-K found
    if not filings:
        filings = list_filings(ticker, "6-K", limit=limit, earnings_only=False)

    results = []
    for f in filings:
        try:
            docs = _filing_document_urls(f["cik"], f["accessionNumber"])
            url = _best_exhibit(docs)
            if not url and f.get("primaryDocument"):
                # Fall back to primary document
                acc_no_dash = f["accessionNumber"].replace("-", "")
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(f.get('cik', 0) or 0)}/"
                    f"{acc_no_dash}/{f['primaryDocument']}"
                )
            text = _fetch_text(
                url,
                publication_ts=_normalize_pub_ts(f["filingDate"]),
                ticker=ticker,
                source="edgar_8k",
            )
            # Use reportDate when available; fall back to filingDate shifted back
            # 60 days because 8-Ks are filed ~3-6 weeks after quarter end,
            # meaning filingDate alone would label Q1 earnings as Q2.
            _rd = f["reportDate"]
            _quarter = (_quarter_label(_rd) if _rd
                        else _quarter_label(f["filingDate"], shift_back=True))
            results.append({
                "filingDate":  f["filingDate"],
                "reportDate":  _rd,
                "text":        text,
                "source_url":  url,
                "form":        "8-K",
                "quarter":     _quarter,
            })
        except Exception as exc:
            _rd = f["reportDate"]
            _quarter = (_quarter_label(_rd) if _rd
                        else _quarter_label(f["filingDate"], shift_back=True))
            results.append({
                "filingDate": f["filingDate"],
                "reportDate": _rd,
                "text":       "",
                "source_url": "",
                "form":       "8-K",
                "quarter":    _quarter,
                "error":      str(exc),
            })
    return results


# ─────────────────────────── 10-Q (MD&A extraction) ──────────────────────────

_MDA_PATTERNS = [
    r"(?i)item\s+2[\.\s]+management.{0,40}discussion",
    r"(?i)management.{0,20}discussion\s+and\s+analysis",
    r"(?i)MD&A",
]

_MDA_END_PATTERNS = [
    r"(?i)item\s+3[\.\s]+quantitative",
    r"(?i)item\s+4[\.\s]+controls",
    r"(?i)item\s+1[^a-z]",   # Item 1 of Part II
]


def _extract_mda(text: str) -> str:
    """Heuristically extract the MD&A section from a 10-Q text."""
    start_idx = None
    for pat in _MDA_PATTERNS:
        m = re.search(pat, text)
        if m:
            start_idx = m.start()
            break
    if start_idx is None:
        return text[:8000]  # Return first 8k chars if no header found

    end_idx = len(text)
    for pat in _MDA_END_PATTERNS:
        m = re.search(pat, text[start_idx + 200:])
        if m:
            end_idx = start_idx + 200 + m.start()
            break

    section = text[start_idx:end_idx]
    # Cap at ~12k characters to keep Claude context manageable
    return section[:12000]


def fetch_ten_q(ticker: str, limit: int = 6, as_of: Optional[str] = None) -> list[dict]:
    """
    Fetch recent 10-Q filings and extract the MD&A section.
    Uses the primaryDocument field directly — the most reliable source.
    Returns list of dicts with: filingDate, reportDate, mda_text, source_url.

    When as_of is provided, returns only event-log records with
    publication_ts <= as_of (point-in-time replay; no network calls).
    """
    if as_of is not None:
        return _replay_from_log(ticker, "edgar_10q", as_of, mda=True)

    filings = list_filings(ticker, "10-Q", limit=limit)
    results = []
    for f in filings:
        try:
            # Always use the primary document for 10-Q — it's the full filing.
            # The document index finds exhibits (EX-31, EX-32) which are wrong targets.
            acc_no_dash = f["accessionNumber"].replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(f.get('cik', 0) or 0)}/"
                f"{acc_no_dash}/{f['primaryDocument']}"
            )

            full_text = _fetch_text(
                url,
                publication_ts=_normalize_pub_ts(f["filingDate"]),
                ticker=ticker,
                source="edgar_10q",
            )
            mda = _extract_mda(full_text)
            results.append({
                "filingDate":  f["filingDate"],
                "reportDate":  f["reportDate"],
                "mda_text":    mda,
                "full_text":   full_text[:3000],  # brief snippet of full text
                "source_url":  url,
                "form":        "10-Q",
                "quarter":     _quarter_label(f["reportDate"] or f["filingDate"]),
            })
        except Exception as exc:
            results.append({
                "filingDate": f["filingDate"],
                "reportDate": f["reportDate"],
                "mda_text":   "",
                "full_text":  "",
                "source_url": "",
                "form":       "10-Q",
                "quarter":    _quarter_label(f["reportDate"] or f["filingDate"]),
                "error":      str(exc),
            })
    return results


def fetch_ten_k(ticker: str, limit: int = 2, as_of: Optional[str] = None) -> list[dict]:
    """
    Fetch recent 10-K (annual) filings and extract the MD&A section.
    Covers Q4 periods which are never in 10-Q.
    """
    if as_of is not None:
        return _replay_from_log(ticker, "edgar_10k", as_of, mda=True)

    filings = list_filings(ticker, "10-K", limit=limit)
    results = []
    for f in filings:
        try:
            acc_no_dash = f["accessionNumber"].replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(f.get('cik', 0) or 0)}/"
                f"{acc_no_dash}/{f['primaryDocument']}"
            )
            full_text = _fetch_text(
                url,
                publication_ts=_normalize_pub_ts(f["filingDate"]),
                ticker=ticker,
                source="edgar_10k",
            )
            mda = _extract_mda(full_text)
            results.append({
                "filingDate":  f["filingDate"],
                "reportDate":  f["reportDate"],
                "mda_text":    mda,
                "full_text":   full_text[:3000],
                "source_url":  url,
                "form":        "10-K",
                "quarter":     _quarter_label(f["reportDate"] or f["filingDate"]),
            })
        except Exception as exc:
            results.append({
                "filingDate": f["filingDate"],
                "reportDate": f["reportDate"],
                "mda_text":   "",
                "full_text":  "",
                "source_url": "",
                "form":       "10-K",
                "quarter":    _quarter_label(f["reportDate"] or f["filingDate"]),
                "error":      str(exc),
            })
    return results


# ─────────────────────────── MD&A QoQ diff ───────────────────────────────────

def _tokenize(sentence: str) -> frozenset[str]:
    """Lowercase word tokens from a sentence, ignoring punctuation."""
    return frozenset(re.sub(r"[^\w\s]", "", sentence.lower()).split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences on '. ' boundaries."""
    raw = re.split(r"\.\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


def diff_mda(current_text: str, prior_text: str) -> dict:
    """
    Compute a quarter-over-quarter diff of two MD&A sections.

    Classifies each current sentence as:
      - new:      max Jaccard against all prior sentences < 0.5
      - retained: max Jaccard >= 0.5 with at least one prior sentence

    Classifies each prior sentence as:
      - removed:  max Jaccard against all current sentences < 0.5

    Returns:
      {
        "new_language":     str,   joined new sentences
        "removed_language": str,   joined removed sentences
        "unchanged_ratio":  float, retained / total current sentences
        "current_quarter":  None,  filled by caller
        "prior_quarter":    None,  filled by caller
      }
    """
    current_sentences = _split_sentences(current_text or "")
    prior_sentences   = _split_sentences(prior_text   or "")

    if not current_sentences:
        return {
            "new_language":     "",
            "removed_language": "\n".join(prior_sentences),
            "unchanged_ratio":  0.0,
            "current_quarter":  None,
            "prior_quarter":    None,
        }

    # Tokenize once; reuse for both passes to avoid O(n²) re-tokenization
    prior_tokens   = [_tokenize(s) for s in prior_sentences]
    current_tokens = [_tokenize(s) for s in current_sentences]

    new_sentences      = []
    retained_sentences = []

    for sent, tok in zip(current_sentences, current_tokens):
        best = max((_jaccard(tok, pt) for pt in prior_tokens), default=0.0)
        if best >= 0.5:
            retained_sentences.append(sent)
        else:
            new_sentences.append(sent)

    # removed: prior sentences with no close match in current
    removed_sentences = []
    for sent, pt in zip(prior_sentences, prior_tokens):
        best = max((_jaccard(pt, ct) for ct in current_tokens), default=0.0)
        if best < 0.5:
            removed_sentences.append(sent)

    unchanged_ratio = (
        len(retained_sentences) / len(current_sentences)
        if current_sentences else 0.0
    )

    return {
        "new_language":     "\n".join(new_sentences),
        "removed_language": "\n".join(removed_sentences),
        "unchanged_ratio":  round(unchanged_ratio, 4),
        "current_quarter":  None,
        "prior_quarter":    None,
    }


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _quarter_label(date_str: str, shift_back: bool = False) -> str:
    """Convert 'YYYY-MM-DD' to 'Q1 2024' label.

    shift_back=True is used when date_str is a filingDate fallback for an 8-K:
    earnings releases are filed ~3-6 weeks after quarter end, so the filingDate
    typically falls in the *next* calendar quarter. Shifting back 60 days recovers
    the reported quarter label.
    """
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        if shift_back:
            d = d - timedelta(days=60)
        q = (d.month - 1) // 3 + 1
        return f"Q{q} {d.year}"
    except Exception:
        return date_str[:10] if date_str else "Unknown"


def get_company_name(ticker: str) -> str:
    """Return the company's full name from EDGAR submission data."""
    try:
        cik = get_cik(ticker)
        data = _submission_data(cik)
        return data.get("name", ticker.upper())
    except Exception:
        return ticker.upper()
