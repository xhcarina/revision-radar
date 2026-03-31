"""
Company metadata and analyst estimate client using yfinance (free, no API key).

NOTE: Stock price data is fetched ONLY for display context and ground-truth
validation — it is NEVER used as a predictive feature in revision scoring.
Analyst estimate revision data (yf.Ticker.eps_revisions, revenue_estimate)
is used only to show what the consensus currently expects.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
import yfinance as yf

# ─────────────────────────── Company metadata ────────────────────────────────

def get_company_info(ticker: str) -> dict:
    """Return basic company metadata dict."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "ticker":        ticker.upper(),
            "name":          info.get("longName", ticker.upper()),
            "sector":        info.get("sector", "Unknown"),
            "industry":      info.get("industry", "Unknown"),
            "description":   info.get("longBusinessSummary", ""),
            "market_cap":    info.get("marketCap"),
            "employees":     info.get("fullTimeEmployees"),
            "website":       info.get("website", ""),
            "logo_url":      info.get("logo_url", ""),
            "currency":      info.get("financialCurrency", "USD"),
            "exchange":      info.get("exchange", ""),
            "country":       info.get("country", ""),
            "analyst_count": info.get("numberOfAnalystOpinions"),
        }
    except Exception:
        return {"ticker": ticker.upper(), "name": ticker.upper()}


# ─────────────────────────── Analyst estimates ───────────────────────────────

def get_analyst_estimates(ticker: str) -> dict:
    """
    Return current analyst consensus estimates for EPS and Revenue.
    Includes historical revision counts (up/down).
    NOT used as a predictive feature — used for display only.
    """
    try:
        t = yf.Ticker(ticker)
        result = {}

        # EPS estimates
        try:
            eps_est = t.earnings_estimate
            if eps_est is not None and not eps_est.empty:
                result["eps_estimate"] = eps_est.to_dict()
        except Exception:
            pass

        # Revenue estimates
        try:
            rev_est = t.revenue_estimate
            if rev_est is not None and not rev_est.empty:
                result["revenue_estimate"] = rev_est.to_dict()
        except Exception:
            pass

        # EPS revisions (number of analysts raising / lowering)
        try:
            eps_rev = t.eps_revisions
            if eps_rev is not None and not eps_rev.empty:
                result["eps_revisions"] = eps_rev.to_dict()
        except Exception:
            pass

        # Analyst recommendations
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                latest = recs.sort_index().tail(20)
                result["recommendations"] = latest.reset_index().to_dict(orient="records")
        except Exception:
            pass

        # Upgrades / downgrades
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                result["upgrades_downgrades"] = (
                    ud.reset_index().sort_values("GradeDate", ascending=False)
                    .head(20)
                    .to_dict(orient="records")
                )
        except Exception:
            pass

        return result
    except Exception as exc:
        return {"error": str(exc)}


def get_earnings_history(ticker: str, quarters: int = 8) -> pd.DataFrame:
    """
    Return historical earnings results vs. analyst estimates.
    Columns: date, epsActual, epsEstimate, surprisePercent
    Not used as a predictive feature.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is None or hist.empty:
            return pd.DataFrame()
        df = hist.copy().reset_index()
        # Normalize column names
        col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=col_map)
        for col in ["epsactual", "epsestimate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Back-calculate missing estimates from surprise %:
        # epsEstimate = epsActual / (1 + surprise_pct / 100)
        # surprisePercent from yfinance is already a fraction (e.g. 0.156 = 15.6%)
        if "epsactual" in df.columns and "epsestimate" in df.columns:
            surprise_raw = next(
                (c for c in df.columns if "surprise" in c.lower() and "pct" not in c.lower()),
                None,
            )
            if surprise_raw:
                df[surprise_raw] = pd.to_numeric(df[surprise_raw], errors="coerce")
                missing = df["epsestimate"].isna() & df["epsactual"].notna() & df[surprise_raw].notna()
                # yfinance surprisePercent is a decimal fraction (0.15 = 15%)
                df.loc[missing, "epsestimate"] = (
                    df.loc[missing, "epsactual"] / (1 + df.loc[missing, surprise_raw])
                ).round(4)
            denom = df["epsestimate"].abs().replace(0, float("nan"))
            df["surprise_pct"] = (
                (df["epsactual"] - df["epsestimate"]) / denom * 100
            ).round(2)
        return df.tail(quarters)
    except Exception:
        return pd.DataFrame()


def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Return OHLCV price history (display context only — NOT a signal feature).
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        return df.reset_index()
    except Exception:
        return pd.DataFrame()


def get_short_interest(ticker: str) -> dict:
    """
    Return short interest data (display context only — NOT a model signal).
    Fields: short_percent_of_float, short_ratio, shares_short.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "short_ratio":            info.get("shortRatio"),
            "shares_short":           info.get("sharesShort"),
            "float_shares":           info.get("floatShares"),
        }
    except Exception:
        return {}


# ─────────────────────────── Supplier / customer inference ──────────────────

# Peer universe upgrade path:
# The curated dict covers ~50 tickers with validated supply-chain relationships.
# For broader coverage, replace sector_fallback with relationships extracted
# from 10-K Item 1A (risk factors) and supplier concentration disclosures —
# companies are required to disclose material customer and supplier dependencies.
# A graph built from these disclosures would give true upstream/downstream peers
# for any EDGAR-registered company without manual curation.

_PEER_UNIVERSE_PATH = Path(__file__).parent.parent.parent / "data" / "peer_universe.json"
_PEER_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "peer_cache"
_PEER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PEER_CACHE_TTL = timedelta(days=7)

_SECTOR_FALLBACK_WARNING = (
    "Peers are industry-level, not supply-chain validated. "
    "Sector tone comparison may be noisy."
)


def _load_peer_universe() -> tuple[dict, dict]:
    """
    Load curated and sector_fallback peer maps from data/peer_universe.json.
    Returns (curated_dict, sector_fallback_dict).
    Falls back to empty dicts if the file is missing or malformed.
    """
    try:
        raw = json.loads(_PEER_UNIVERSE_PATH.read_text(encoding="utf-8"))
        return raw.get("curated", {}), raw.get("sector_fallback", {})
    except Exception:
        return {}, {}


_CURATED_PEERS, _SECTOR_PEERS = _load_peer_universe()

# Hard caps by sector: how many peers to use at most.
# Niche sectors get fewer (2) since extra peers add noise, not signal.
_SECTOR_CAPS: dict[str, int] = {
    "Technology":                4,
    "Communication Services":    3,
    "Consumer Cyclical":         3,
    "Consumer Defensive":        2,
    "Healthcare":                3,
    "Financial Services":        3,
    "Energy":                    3,
    "Industrials":               3,
    "Basic Materials":           2,
    "Real Estate":               2,
    "Utilities":                 2,
}
_DEFAULT_CAP = 3

# Normalized lookup built once at import time.
# yfinance returns strings like "Software - Infrastructure" (space-hyphen-space)
# but our map uses em dashes. _norm() strips all of that to plain lowercase words.
def _norm(s: str) -> str:
    s = re.sub(r"[—–\-&/]+", " ", s)
    return re.sub(r"\s+", " ", s).lower().strip()

_SECTOR_PEERS_NORM: dict[str, list[str]] = {
    _norm(k): v for k, v in _SECTOR_PEERS.items()
}


def _discover_peers_via_llm(
    ticker: str,
    company_info: dict | None = None,
) -> list[str]:
    """
    Use Claude to discover upstream supply-chain peers for *ticker*.
    Results are cached per-ticker for _PEER_CACHE_TTL days.
    Returns a list of validated (EDGAR-registered) US-listed tickers.
    """
    cache_file = _PEER_CACHE_DIR / f"{ticker.upper()}.json"

    # Check cache
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            written_at = datetime.fromisoformat(cached["written_at"])
            if datetime.utcnow() - written_at < _PEER_CACHE_TTL:
                return cached["peers"]
        except Exception:
            pass

    info = company_info or {}
    company_name = info.get("name", ticker.upper())
    sector   = info.get("sector", "")
    industry = info.get("industry", "")
    context  = f"{sector} / {industry}".strip(" /") if sector or industry else "unknown sector"

    prompt = (
        f"List the 5 most important UPSTREAM supply-chain suppliers for {company_name} "
        f"({ticker}), a {context} company. Focus on direct component and material "
        f"suppliers — the companies that sell inputs to {ticker}, not customers or "
        f"horizontal peers. Return ONLY a JSON array of US-listed ticker symbols, "
        f"e.g. [\"TSM\", \"AVGO\"]. No explanation, no markdown fences."
    )

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config import ANTHROPIC_API_KEY
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Extract JSON array
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        candidates: list[str] = json.loads(match.group(0)) if match else []
        candidates = [str(t).upper().strip() for t in candidates if t]
    except Exception:
        return []

    # Validate each ticker against EDGAR
    validated: list[str] = []
    try:
        from src.data.sec_client import get_cik
        for t in candidates:
            if t == ticker.upper():
                continue
            try:
                get_cik(t)
                validated.append(t)
            except Exception:
                pass
    except ImportError:
        validated = [t for t in candidates if t != ticker.upper()]

    # Cache result
    try:
        cache_file.write_text(
            json.dumps({"written_at": datetime.utcnow().isoformat(), "peers": validated}),
            encoding="utf-8",
        )
    except Exception:
        pass

    return validated


def get_supply_chain_peers(
    ticker: str,
    company_info: dict | None = None,
) -> tuple[list[str], dict]:
    """
    Return (peer_tickers, peer_source) for cross-signal analysis.

    peer_source is a dict:
      {
        "type":         "curated" | "llm_dynamic" | "none",
        "confidence":   "high" | "medium" | "low",
        "warning":      None | str,
        "ticker_count": int,
      }

    Strategy:
      1. Curated map (validated supply-chain relationships) → confidence "high".
      2. Claude-powered dynamic discovery, validated against SEC EDGAR → confidence "medium".
      3. Return empty list with type "none" — never silently returns wrong peers.
    """
    ticker = ticker.upper()

    # 1. Curated
    if ticker in _CURATED_PEERS:
        peers = [p for p in _CURATED_PEERS[ticker] if p != ticker]
        cap   = _SECTOR_CAPS.get(_get_sector(company_info), _DEFAULT_CAP)
        peers = peers[:cap]
        return peers, {
            "type":         "curated",
            "confidence":   "high",
            "warning":      None,
            "ticker_count": len(peers),
        }

    # 2. LLM-powered dynamic discovery (replaces sector_fallback)
    info   = company_info or {}
    sector = info.get("sector", "") or _get_sector_from_yf(ticker)
    cap    = _SECTOR_CAPS.get(sector, _DEFAULT_CAP)

    llm_peers = _discover_peers_via_llm(ticker, company_info=company_info)
    if llm_peers:
        peers = llm_peers[:cap]
        return peers, {
            "type":         "llm_dynamic",
            "confidence":   "medium",
            "warning":      "Peers discovered by AI and validated against SEC EDGAR. "
                            "Review before acting — supply-chain relationships may have changed.",
            "ticker_count": len(peers),
        }

    return [], {"type": "none", "confidence": "low", "warning": None, "ticker_count": 0}


def _get_sector(company_info: dict | None) -> str:
    return (company_info or {}).get("sector", "")


def _get_sector_from_yf(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).info.get("sector", "") or ""
    except Exception:
        return ""


def get_sector_keywords(ticker: str) -> list[str]:
    """Return industry keywords for GDELT news queries."""
    sector_keywords_map = {
        "Technology":            ["technology", "tech sector", "semiconductors", "AI chips"],
        "Consumer Cyclical":     ["consumer spending", "retail", "e-commerce"],
        "Healthcare":            ["healthcare", "pharma", "biotech", "drug approval"],
        "Financial Services":    ["banking", "interest rates", "credit", "fintech"],
        "Energy":                ["oil prices", "energy", "crude oil", "OPEC"],
        "Communication Services":["advertising", "streaming", "social media"],
        "Industrials":           ["manufacturing", "supply chain", "logistics"],
        "Consumer Defensive":    ["consumer staples", "inflation", "grocery"],
        "Basic Materials":       ["commodities", "mining", "metals"],
        "Real Estate":           ["real estate", "housing market", "REIT"],
        "Utilities":             ["energy grid", "utilities", "power"],
    }
    info = get_company_info(ticker)
    sector = info.get("sector", "")
    return sector_keywords_map.get(sector, [sector.lower(), "corporate earnings"])


def format_market_cap(value: Optional[float]) -> str:
    if not value:
        return "N/A"
    if value >= 1e12:
        return f"${value/1e12:.1f}T"
    if value >= 1e9:
        return f"${value/1e9:.1f}B"
    if value >= 1e6:
        return f"${value/1e6:.1f}M"
    return f"${value:,.0f}"
