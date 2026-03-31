"""
News intelligence client using GDELT (free, no API key required).

GDELT provides global news coverage with tone/sentiment scoring.
We query it for company-specific coverage and return:
  - Article lists with tone scores
  - Timeline of average sentiment per day
  - Industry-level news aggregation

GDELT tone scale:  < 0 = negative, > 0 = positive, range roughly -10 to +10
We normalize to [-1, +1] for consistency with other signals.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

try:
    from gdeltdoc import GdeltDoc, Filters as GdeltFilters
    GDELT_LIB = True
except ImportError:
    GDELT_LIB = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CACHE_DIR, CACHE_TTL_HOURS
from event_log import event_log as _event_log, GDELT_FILTER_VERSION

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── GDELT article cache ─────────────────────────────

_GDELT_CACHE_DIR = CACHE_DIR / "raw" / "gdelt"
_GDELT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _gdelt_cache_path(query: str, start: str, end: str) -> Path:
    key = hashlib.sha256(f"{query}|{start}|{end}".encode()).hexdigest()[:40]
    return _GDELT_CACHE_DIR / f"{key}.json"


def _normalize_gdelt_ts(seendate: Optional[str]) -> Optional[str]:
    """
    Normalize a GDELT seendate string to ISO 8601.

    GDELT seendate formats seen in practice:
      '20240315T120000Z'  → '2024-03-15T12:00:00Z'
      '20240315120000'    → '2024-03-15T12:00:00Z'
      '2024-03-15 12:00:00' → '2024-03-15T12:00:00Z'
    Returns None — never raises — if the input is missing or unparseable.
    """
    if not seendate:
        return None
    try:
        s = str(seendate).strip()
        # Strip Z suffix temporarily
        s_clean = s.rstrip("Z").replace("T", "").replace("-", "").replace(":", "").replace(" ", "")
        if len(s_clean) >= 14:
            dt = datetime.strptime(s_clean[:14], "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if len(s_clean) >= 8:
            dt = datetime.strptime(s_clean[:8], "%Y%m%d")
            return dt.strftime("%Y-%m-%dT00:00:00Z")
    except (ValueError, TypeError):
        pass
    return None


def _gdelt_cache_read(query: str, start: str, end: str) -> Optional[pd.DataFrame]:
    p = _gdelt_cache_path(query, start, end)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        if datetime.fromisoformat(raw["ts"]) < datetime.now() - timedelta(hours=CACHE_TTL_HOURS):
            return None
        records = raw.get("records", [])
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return None


def _gdelt_cache_write(query: str, start: str, end: str, df: pd.DataFrame) -> None:
    p = _gdelt_cache_path(query, start, end)
    records = []
    for rec in df.to_dict(orient="records"):
        seendate_raw = rec.get("seendate")
        rec["publication_ts"] = _normalize_gdelt_ts(
            str(seendate_raw) if seendate_raw is not None else None
        )
        records.append(rec)
    p.write_text(json.dumps({
        "ts":      datetime.now().isoformat(),
        "records": records,
    }, default=str))

# ─────────────────────────── GDELT helpers ───────────────────────────────────

def _gdelt_artlist(query: str, start: str, end: str, max_records: int = 100) -> pd.DataFrame:
    """
    Raw GDELT v2 Doc API call returning a DataFrame of articles.
    Results are cached to disk (CACHE_TTL_HOURS) with publication_ts per article.
    Falls back gracefully if gdeltdoc library is missing or rate-limited.
    """
    cached = _gdelt_cache_read(query, start, end)
    if cached is not None:
        return cached

    df = pd.DataFrame()

    if GDELT_LIB:
        try:
            f = GdeltFilters(
                keyword=query,
                start_date=start,
                end_date=end,
            )
            gd = GdeltDoc()
            df = gd.article_search(f)
        except Exception:
            # Includes RateLimitError — fall through to direct HTTP
            time.sleep(1)

    if df.empty:
        # Direct API fallback
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={requests.utils.quote(query)}"
            f"&mode=artlist&maxrecords={max_records}&format=json"
            f"&startdatetime={start.replace('-','')}000000"
            f"&enddatetime={end.replace('-','')}235959"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            if articles:
                df = pd.DataFrame(articles)
        except Exception:
            pass

    if not df.empty:
        _gdelt_cache_write(query, start, end, df)

    return df


def _gdelt_timeline(query: str, start: str, end: str) -> pd.DataFrame:
    """Return daily average tone timeline from GDELT."""
    if GDELT_LIB:
        try:
            f = GdeltFilters(keyword=query, start_date=start, end_date=end)
            gd = GdeltDoc()
            tl = gd.timeline_search("timelinevol", f)
            return tl
        except Exception:
            pass

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query)}"
        f"&mode=timelinevol&format=json"
        f"&startdatetime={start.replace('-','')}000000"
        f"&enddatetime={end.replace('-','')}235959"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # GDELT v2 timeline response: {"timeline": [{"series": "...", "data": [{...}]}]}
        # Flatten all series' data points into one list
        records = []
        for series in data.get("timeline", []):
            for item in series.get("data", []):
                d = str(item.get("date", ""))[:8]
                if d:
                    records.append({"date": d, "value": item.get("value", 0)})
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────── Public interface ─────────────────────────────────

def get_company_news(
    ticker: str,
    company_name: str,
    days: int = 180,
    max_articles: int = 250,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch recent news articles about *company_name / ticker*.

    GDELT article_search caps at 250 records per request. For date ranges longer
    than 90 days, the range is split into 90-day windows so each quarter gets up
    to 250 articles (rather than the entire span competing for 250 slots).

    GDELT article_search does not return tone per-article; tone comes from the
    timeline API. We annotate articles with the daily tone from the timeline.

    Returns a DataFrame with columns:
        title, url, seendate, domain, language, tone (normalized -1..1)

    When as_of is provided, returns only event-log records with
    publication_ts <= as_of reconstructed from the GDELT batch cache files.
    """
    if as_of is not None:
        return _replay_gdelt_from_log(ticker, as_of)

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    _short_name = company_name.split()[0] if company_name else ticker

    # ── Chunked fetch: one request per 90-day window ─────────────────────────
    # GDELT caps at 250 records per call; chunking ensures each quarter gets
    # its own allocation rather than recent dates crowding out older ones.
    CHUNK_DAYS = 90
    chunks: list[tuple[str, str]] = []
    chunk_end = end_dt
    while chunk_end > start_dt:
        chunk_start = max(chunk_end - timedelta(days=CHUNK_DAYS), start_dt)
        chunks.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        chunk_end = chunk_start - timedelta(days=1)

    frames = []
    query = ticker
    for c_start, c_end in chunks:
        for query in [company_name, _short_name, ticker]:
            chunk_df = _gdelt_artlist(query, c_start, c_end, max_records=min(max_articles, 250))
            if not chunk_df.empty:
                frames.append(chunk_df)
                break

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"]) \
         if frames else pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=["title", "url", "seendate", "domain", "tone_raw", "tone"])

    # Ensure date column
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    else:
        df["seendate"] = pd.NaT

    # GDELT article search does NOT return per-article tone scores.
    # Fetch tone timeline and join on date for approximate per-article sentiment.
    _start_str = start_dt.strftime("%Y-%m-%d")
    _end_str   = end_dt.strftime("%Y-%m-%d")
    try:
        tone_tl = _get_tone_timeline(query, _start_str, _end_str)
        if not tone_tl.empty:
            tone_tl["date_key"] = pd.to_datetime(tone_tl["date"]).dt.date
            df["date_key"] = df["seendate"].dt.date
            df = df.merge(tone_tl[["date_key", "tone_raw"]], on="date_key", how="left")
            df["tone_raw"] = df["tone_raw"].fillna(0.0)
        else:
            df["tone_raw"] = 0.0
    except Exception:
        df["tone_raw"] = 0.0

    # Normalize tone from GDELT scale (~-10..+10) to -1..+1
    df["tone"] = pd.to_numeric(df["tone_raw"], errors="coerce").fillna(0.0).clip(-10, 10) / 10.0

    keep = ["title", "url", "seendate", "domain", "tone_raw", "tone"]
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    # Log each article to the event log (dedup prevents duplicate entries)
    _log_gdelt_articles(df, ticker, query, _start_str, _end_str)

    return df[[c for c in keep if c in df.columns]]


def _log_gdelt_articles(
    df: pd.DataFrame,
    ticker: str,
    query: str,
    start: str,
    end: str,
) -> None:
    """Append one event log record per article in df."""
    batch_path = str(_gdelt_cache_path(query, start, end))
    now_ts = datetime.now().isoformat()
    for _, row in df.iterrows():
        art_url = str(row.get("url", ""))
        seendate_raw = row.get("seendate")
        # seendate may already be a parsed datetime at this point
        if hasattr(seendate_raw, "strftime"):
            pub_ts = seendate_raw.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            pub_ts = _normalize_gdelt_ts(str(seendate_raw) if seendate_raw is not None else None)
        art_dict = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in row.to_dict().items()}
        art_hash = hashlib.sha256(
            json.dumps(art_dict, sort_keys=True, default=str).encode()
        ).hexdigest()
        _event_log.append(
            source="gdelt",
            ticker=ticker,
            url=art_url,
            publication_ts=pub_ts,
            fetched_at=now_ts,
            content_hash=art_hash,
            filter_version=GDELT_FILTER_VERSION,
            extracted_text_path=batch_path,
        )


def _replay_gdelt_from_log(ticker: str, as_of: str) -> pd.DataFrame:
    """
    Reconstruct get_company_news result from the event log for point-in-time analysis.
    Reads GDELT batch cache files and filters articles by publication_ts <= as_of.
    """
    records = _event_log.query(ticker, source="gdelt", as_of=as_of)
    if not records:
        return pd.DataFrame(columns=["title", "url", "seendate", "domain", "tone_raw", "tone"])

    # Collect unique batch cache files; read each once
    article_url_set = {r["url"] for r in records if r.get("url")}
    batch_paths: dict[str, list[dict]] = {}
    for r in records:
        bp = r.get("extracted_text_path", "")
        if bp and bp not in batch_paths:
            raw = r.get("extracted_text")  # already loaded by query()
            if raw:
                try:
                    data = json.loads(raw)
                    batch_paths[bp] = data.get("records", [])
                except Exception:
                    batch_paths[bp] = []
            else:
                batch_paths[bp] = []

    rows = []
    seen_urls: set[str] = set()
    for batch_articles in batch_paths.values():
        for art in batch_articles:
            art_url = art.get("url", "")
            if art_url in article_url_set and art_url not in seen_urls:
                seen_urls.add(art_url)
                rows.append(art)

    if not rows:
        return pd.DataFrame(columns=["title", "url", "seendate", "domain", "tone_raw", "tone"])

    df = pd.DataFrame(rows)
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    if "tone" not in df.columns:
        df["tone"] = 0.0
    if "tone_raw" not in df.columns:
        df["tone_raw"] = df["tone"]
    keep = ["title", "url", "seendate", "domain", "tone_raw", "tone"]
    return df[[c for c in keep if c in df.columns]]


def _get_tone_timeline(query: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily tone timeline from GDELT (returns tone_raw in GDELT scale)."""
    if GDELT_LIB:
        try:
            f = GdeltFilters(keyword=query, start_date=start, end_date=end)
            gd = GdeltDoc()
            tl = gd.timeline_search("timelinetone", f)  # type: ignore
            if tl is not None and not tl.empty:
                # gdeltdoc returns columns like 'datetime', 'series', 'value'
                if "value" in tl.columns:
                    tl = tl.rename(columns={"value": "tone_raw"})
                elif tl.columns[-1] != "tone_raw":
                    tl = tl.rename(columns={tl.columns[-1]: "tone_raw"})
                if "datetime" in tl.columns:
                    tl["date"] = pd.to_datetime(tl["datetime"]).dt.date
                elif "date" not in tl.columns:
                    tl["date"] = pd.to_datetime(tl.index).date
                return tl[["date", "tone_raw"]]
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "tone_raw"])


def _naive_date(series) -> "pd.Series":
    """Convert a datetime series to tz-naive date (strips UTC or any tz)."""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s.dt.normalize()


def get_news_sentiment_timeline(
    ticker: str,
    company_name: str,
    days: int = 180,
    articles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return a daily time series of average news tone for *company_name*.
    Uses GDELT tone timeline directly (more reliable than per-article tone).
    Columns: date (datetime), sentiment (float, -1..+1), volume (int), sentiment_ma7

    articles_df: pre-fetched articles (from get_company_news) — used to compute
    volume counts when the GDELT volume timeline endpoint fails, avoiding a second
    network call.
    """
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    _short_name = company_name.split()[0] if company_name else ticker
    tone_tl = pd.DataFrame()
    vol_tl  = pd.DataFrame()
    for query in [company_name, _short_name, ticker]:
        tone_tl = _get_tone_timeline(query, start, end)
        vol_tl  = _gdelt_volume_timeline(query, start, end)
        if not tone_tl.empty:
            break

    if not tone_tl.empty:
        daily = tone_tl.copy()
        daily["date"] = _naive_date(daily["date"])
        daily["sentiment"] = daily["tone_raw"].astype(float).clip(-10, 10) / 10.0
        daily = daily[["date", "sentiment"]].sort_values("date")
        if not vol_tl.empty:
            vol_tl["date"] = _naive_date(vol_tl["date"])
            daily = daily.merge(vol_tl, on="date", how="left")
            daily["volume"] = daily["volume"].fillna(0).astype(int)
        else:
            daily["volume"] = 0
        # If GDELT volume failed, count from pre-fetched articles
        if daily["volume"].sum() == 0 and articles_df is not None and not articles_df.empty:
            _arts = articles_df.dropna(subset=["seendate"]).copy()
            _arts["_date"] = _naive_date(_arts["seendate"])
            _vol = _arts.groupby("_date").size().reset_index(name="_vol")
            _vol = _vol.rename(columns={"_date": "date"})
            daily = daily.merge(_vol, on="date", how="left")
            daily["volume"] = daily["_vol"].fillna(0).astype(int)
            daily = daily.drop(columns=["_vol"], errors="ignore")
        daily["sentiment_ma7"] = daily["sentiment"].rolling(7, min_periods=1).mean()
        return daily

    # Fallback: aggregate fully from article-level data
    articles = articles_df if (articles_df is not None and not articles_df.empty) \
               else get_company_news(ticker, company_name, days=days, max_articles=200)
    if articles.empty:
        return pd.DataFrame(columns=["date", "sentiment", "volume", "sentiment_ma7"])
    articles = articles.dropna(subset=["seendate"])
    articles["date"] = _naive_date(articles["seendate"])
    daily = (
        articles.groupby("date")
        .agg(sentiment=("tone", "mean"), volume=("tone", "count"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["sentiment_ma7"] = daily["sentiment"].rolling(7, min_periods=1).mean()
    return daily


def _gdelt_volume_timeline(query: str, start: str, end: str) -> pd.DataFrame:
    """Fetch article volume timeline from GDELT.
    Tries gdeltdoc library first, then falls back to GDELT REST API so that
    the full requested date range is always covered (same approach as _get_tone_timeline).
    """
    # ── 1. gdeltdoc library ───────────────────────────────────────────────────
    if GDELT_LIB:
        try:
            f = GdeltFilters(keyword=query, start_date=start, end_date=end)
            gd = GdeltDoc()
            tl = gd.timeline_search("timelinevol", f)
            if tl is not None and not tl.empty:
                if "datetime" in tl.columns:
                    tl["date"] = _naive_date(tl["datetime"])
                elif "date" not in tl.columns:
                    tl["date"] = _naive_date(pd.Series(tl.index))
                else:
                    tl["date"] = _naive_date(tl["date"])
                if "value" in tl.columns:
                    vol_col = "value"
                else:
                    candidates = [c for c in tl.columns if c not in ("date", "datetime")]
                    if not candidates:
                        pass  # fall through to REST fallback
                    else:
                        vol_col = candidates[-1]
                        tl = tl.rename(columns={vol_col: "volume"})
                        return tl[["date", "volume"]]
        except Exception:
            pass

    # ── 2. GDELT REST API fallback — covers full date range reliably ──────────
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query)}"
        f"&mode=timelinevol&format=json"
        f"&startdatetime={start.replace('-', '')}000000"
        f"&enddatetime={end.replace('-', '')}235959"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # GDELT v2 structure: {"timeline": [{"series": "...", "data": [{...}]}]}
        records = []
        for series in data.get("timeline", []):
            for item in series.get("data", []):
                d = str(item.get("date", ""))[:8]
                if d:
                    records.append({"date": d, "volume": item.get("value", 0)})
        if not records:
            return pd.DataFrame(columns=["date", "volume"])
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "volume"])


def get_industry_news(
    sector_keywords: list[str],
    days: int = 90,
    max_articles: int = 50,
) -> pd.DataFrame:
    """
    Fetch industry-level news for context (e.g., ['semiconductor', 'AI chips']).
    Returns same schema as get_company_news.
    """
    if not sector_keywords:
        return pd.DataFrame()
    query = sector_keywords[0]  # use primary keyword; gdeltdoc works best with single plain terms
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = _gdelt_artlist(query, start, end, max_records=max_articles)
    if df.empty:
        return pd.DataFrame(columns=["title", "url", "seendate", "domain", "tone_raw", "tone"])
    if "tone" in df.columns:
        df["tone_raw"] = df["tone"]
        df["tone"] = df["tone_raw"].astype(float).clip(-10, 10) / 10.0
    else:
        df["tone_raw"] = 0
        df["tone"] = 0.0
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    return df.dropna(subset=["title"]).reset_index(drop=True)


# ─────────────────────────── Source-tone baseline correction ─────────────────

# Empirical baseline offsets per source-domain pattern.
# Subtract from tone *before* aggregation so cross-source comparisons are fair.
# Keys are substrings matched against article url or domain.
SOURCE_TONE_BASELINE: dict[str, float] = {
    "reuters":       -0.05,   # wire services run structurally negative
    "bloomberg":     -0.05,
    "wsj":           -0.03,
    "ft.com":        -0.03,
    "businesswire":  +0.04,   # press-release wires run structurally positive
    "prnewswire":    +0.04,
    "globenewswire": +0.04,
}


def _match_source_baseline(url: str, domain: str) -> float:
    """
    Return the baseline offset for one article.
    Checks url first, then domain, using substring matching.
    Returns 0.0 if no pattern matches.
    """
    haystack = f"{url} {domain}".lower()
    for pattern, offset in SOURCE_TONE_BASELINE.items():
        if pattern in haystack:
            return offset
    return 0.0


def apply_source_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function: add df["source_baseline"] and subtract it from df["tone"].
    Input df must have a "tone" column; "url" and "domain" are used if present.
    Returns a new DataFrame — does not mutate the input.
    """
    df = df.copy()
    url_col    = df["url"].astype(str)    if "url"    in df.columns else pd.Series("", index=df.index)
    domain_col = df["domain"].astype(str) if "domain" in df.columns else pd.Series("", index=df.index)
    df["source_baseline"] = [
        _match_source_baseline(u, d) for u, d in zip(url_col, domain_col)
    ]
    df["tone"] = df["tone"] - df["source_baseline"]
    return df


# ─────────────────────────── Article deduplication ───────────────────────────

import re as _re
import string as _string


def _normalize_title(title: str) -> frozenset[str]:
    """Lowercase, strip punctuation, split on whitespace → frozenset of tokens."""
    t = title.lower()
    t = t.translate(str.maketrans("", "", _string.punctuation))
    return frozenset(t.split())


def dedup_articles(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Pure function: remove duplicate and near-duplicate articles.

    Pass 1 — URL dedup: keep first occurrence (earliest seendate) per url.
    Pass 2 — Title similarity dedup within each calendar day:
              token-overlap (Jaccard) >= 0.75 → keep the earlier seendate article.

    Returns a new DataFrame with a reset index.
    Logs drop counts when debug=True.
    """
    if df.empty:
        return df.copy()

    original_len = len(df)

    # ── Pass 1: URL dedup ─────────────────────────────────────────────────────
    if "url" in df.columns:
        df = (
            df.sort_values("seendate", na_position="last")
              .drop_duplicates(subset=["url"], keep="first")
              .reset_index(drop=True)
        )
    after_url = len(df)

    # ── Pass 2: title similarity within each day-bucket ───────────────────────
    if "title" in df.columns and "seendate" in df.columns:
        df = df.copy()
        df["_day"] = pd.to_datetime(df["seendate"], errors="coerce").dt.date
        df["_tokens"] = df["title"].fillna("").apply(_normalize_title)

        keep_mask = [True] * len(df)
        # Group by calendar day for O(n²) within small buckets
        for day, grp in df.groupby("_day", dropna=False):
            idxs = grp.index.tolist()
            for i, idx_i in enumerate(idxs):
                if not keep_mask[idx_i]:
                    continue
                tok_i = df.at[idx_i, "_tokens"]
                for idx_j in idxs[i + 1:]:
                    if not keep_mask[idx_j]:
                        continue
                    tok_j = df.at[idx_j, "_tokens"]
                    union = tok_i | tok_j
                    if not union:
                        continue
                    overlap = len(tok_i & tok_j) / len(union)
                    if overlap >= 0.75:
                        keep_mask[idx_j] = False   # drop later article

        df = df[keep_mask].drop(columns=["_day", "_tokens"]).reset_index(drop=True)

    after_title = len(df)

    if debug:
        url_dropped   = original_len - after_url
        title_dropped = after_url    - after_title
        print(
            f"[dedup] {original_len} → {after_url} (url: -{url_dropped})"
            f" → {after_title} (title sim: -{title_dropped})"
        )

    return df


# ─────────────────────────── Recency weighting ───────────────────────────────

def compute_recency_weights(seendate_series: pd.Series, half_life_days: int = 30) -> pd.Series:
    """
    Pure function: exponential decay with given half-life.

    weight = 0.5 ** (age_days / half_life_days)

    age_days=0  → weight 1.0
    age_days=30 → weight 0.5
    age_days=90 → weight 0.125
    age_days=180→ weight ~0.016

    Returns a Series aligned with the input index.
    """
    now = pd.Timestamp.now()
    dates = pd.to_datetime(seendate_series, errors="coerce").dt.tz_localize(None)
    age_days = ((now - dates).dt.total_seconds() / 86400).fillna(180).clip(lower=0)
    return (0.5 ** (age_days / half_life_days)).astype(float)


# ─────────────────────────── Shared article prep ─────────────────────────────

def prepare_articles(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Apply the full pre-processing pipeline to a raw articles DataFrame:
      1. URL + title-similarity deduplication
      2. Source-tone baseline correction
      3. Recency weight column

    Returns a new DataFrame ready for both compute_news_signal and
    LLMExtractor.extract_news. Does not mutate the input.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = dedup_articles(df, debug=debug)
    if df.empty:
        return df

    df = apply_source_baseline(df)

    if "seendate" in df.columns:
        df["recency_weight"] = compute_recency_weights(df["seendate"])
    else:
        df["recency_weight"] = 1.0

    return df


# ─────────────────────────── Signal aggregation ──────────────────────────────

def compute_news_signal(articles: pd.DataFrame, debug: bool = False) -> float:
    """
    Aggregate article sentiments into a single signal in [-1, +1].

    Pipeline (applied inside this function so it is always consistent):
      1. Dedup by URL and title similarity
      2. Source-tone baseline correction
      3. Exponential recency weighting (half-life 30 days)
      4. Weighted mean of corrected tone

    df["recency_weight"] is added as a column before aggregation.

    Returns 0.0 on empty or unscoreable input.
    """
    if articles is None or articles.empty or "tone" not in articles.columns:
        return 0.0

    df = prepare_articles(articles, debug=debug)
    df = df.dropna(subset=["tone", "recency_weight"])
    if df.empty:
        return 0.0

    total_weight = df["recency_weight"].sum()
    if total_weight == 0:
        return 0.0

    signal = (df["tone"] * df["recency_weight"]).sum() / total_weight

    if debug:
        print(
            f"[compute_news_signal] n={len(df)}"
            f"  weighted_signal={signal:.4f}"
            f"  total_weight={total_weight:.2f}"
        )

    return float(signal)
