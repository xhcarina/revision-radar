"""
Check — Predictive Analyst Revision Engine
Streamlit application entry point.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_NAME, APP_SUBTITLE, APP_VERSION,
    ANTHROPIC_API_KEY, SP500_SAMPLE, TICKER_COMPANY_MAP, FORECAST_ITEMS,
    COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_WARNING,
    COLOR_BG, COLOR_CARD, COLOR_BORDER, COLOR_TEXT, COLOR_SUBTEXT,
    DEFAULT_QUARTERS,
)

# ── Cache housekeeping (runs once per process startup) ────────────────────────
from src.data.cache_manager import purge_stale_cache, save_report, load_report, list_reports, delete_report
if "cache_purged" not in st.session_state:
    _purged = purge_stale_cache()
    st.session_state.cache_purged = True

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Check",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — theme-adaptive via CSS variables (works in both light & dark mode) ──
st.markdown(f"""
<style>
/* ── Hide Streamlit footer branding only ── */
footer {{visibility: hidden;}}

/* ── Block container — top padding clears the Streamlit toolbar ── */
.block-container {{
    padding: 3.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: var(--secondary-background-color);
    border-right: 1px solid rgba(128,128,128,0.2);
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: rgba(128,128,128,0.75);
    font-size: 12px;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(128,128,128,0.2);
    margin: 12px 0;
}}

/* ── Metric cards (native st.metric) ── */
[data-testid="metric-container"] {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] {{
    font-size: 11px !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(128,128,128,0.75) !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 2rem !important;
    font-weight: 700;
    line-height: 1.1;
}}
[data-testid="stMetricDelta"] {{
    font-size: 12px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--secondary-background-color);
    border-radius: 8px;
    padding: 4px 6px;
    gap: 2px;
    border: 1px solid rgba(128,128,128,0.2);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    color: rgba(128,128,128,0.75);
    background: transparent;
    border: none;
    transition: all 0.15s ease;
}}
.stTabs [aria-selected="true"] {{
    background: var(--background-color) !important;
    color: var(--text-color) !important;
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 20px;
}}

/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, {COLOR_POSITIVE} 0%, #00a886 100%);
    border: none;
    color: #0e1117;
    font-weight: 700;
    font-size: 13px;
    border-radius: 8px;
    padding: 10px 20px;
    transition: opacity 0.15s ease;
}}
[data-testid="stButton"] > button[kind="primary"]:hover {{
    opacity: 0.88;
}}
[data-testid="stButton"] > button:not([kind="primary"]) {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    color: var(--text-color);
    border-radius: 6px;
    font-size: 11px;
    padding: 4px 8px;
}}
[data-testid="stButton"] > button:not([kind="primary"]):hover {{
    border-color: {COLOR_POSITIVE};
    color: {COLOR_POSITIVE};
}}

/* ── Inputs ── */
[data-testid="stTextInput"] input {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px;
    color: var(--text-color);
    font-size: 14px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {COLOR_POSITIVE};
    box-shadow: 0 0 0 2px {COLOR_POSITIVE}22;
}}

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] {{
    background: {COLOR_POSITIVE};
}}

/* ── Selectbox / multiselect ── */
[data-testid="stMultiSelect"] [data-baseweb="select"] {{
    background: var(--secondary-background-color);
    border-color: rgba(128,128,128,0.2);
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px;
}}

/* ── Status / spinner ── */
[data-testid="stStatusWidget"] {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2) !important;
    border-radius: 8px;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    border-radius: 8px;
    font-size: 13px;
}}

/* ── Custom utility classes ── */
.rr-section-header {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: rgba(128,128,128,0.75);
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 4px;
}}
.rr-card {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
}}
.rr-quote {{
    border-left: 3px solid {COLOR_NEUTRAL};
    background: var(--secondary-background-color);
    padding: 10px 14px;
    margin-bottom: 8px;
    border-radius: 0 6px 6px 0;
    font-size: 13px;
    line-height: 1.55;
}}
.rr-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    border: 1px solid;
}}
.rr-pill {{
    display: inline-block;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px;
    font-weight: 700;
    color: var(--text-color);
    margin-right: 4px;
}}
.rr-divider {{
    border: none;
    border-top: 1px solid rgba(128,128,128,0.2);
    margin: 16px 0;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_md(text: str) -> str:
    """Escape dollar signs in Claude-generated text so Streamlit doesn't
    misinterpret them as LaTeX math delimiters ($...$)."""
    return text.replace("$", r"\$") if text else ""


# ── Header ────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;padding:4px 0 12px 0;'>"
        f""
        f"<div>"
        f"<div style='font-size:20px;font-weight:800;letter-spacing:-0.02em;line-height:1;'>{APP_NAME}</div>"
        f"<div style='font-size:12px;color:rgba(128,128,128,0.75);margin-top:2px;'>"
        f"{APP_SUBTITLE} &nbsp;·&nbsp; v{APP_VERSION}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr class='rr-divider'>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            f"<div style='font-size:13px;font-weight:700;color:inherit;"
            f"margin-bottom:12px;'>Analysis Parameters</div>",
            unsafe_allow_html=True,
        )

        if not ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY not set.\nCreate `.env` with your key.")

        # ── Ticker ──
        st.markdown("<div class='rr-section-header'>Company</div>", unsafe_allow_html=True)

        from streamlit_searchbox import st_searchbox

        # Build a flat lookup: "AAPL — Apple Inc." → "AAPL"
        _ticker_lookup: dict[str, str] = {
            f"{t} — {TICKER_COMPANY_MAP[t]}": t
            for t in sorted(TICKER_COMPANY_MAP)
        }
        _all_labels = list(_ticker_lookup.keys())

        def _search_tickers(query: str) -> list[str]:
            q = query.strip().upper()
            if not q:
                return _all_labels[:20]
            matches = [label for label in _all_labels if q in label.upper()][:20]
            # Always offer the raw query as the first option for unlisted tickers
            if q and q not in [m.split(" — ")[0] for m in matches]:
                matches = [q] + matches
            return matches

        _result = st_searchbox(
            _search_tickers,
            placeholder="Search ticker or company name…",
            key="ticker_searchbox",
        )

        if _result and " — " in str(_result):
            ticker_input = _result.split(" — ")[0].strip()
        elif _result:
            ticker_input = str(_result).upper().strip()
        else:
            ticker_input = ""

        if not ticker_input:
            st.caption("Select a ticker above to enable analysis.")

        # ── Lookback ──
        st.markdown("<div class='rr-section-header' style='margin-top:16px;'>Lookback Window</div>",
                    unsafe_allow_html=True)
        n_quarters = st.slider(
            "Quarters", min_value=1, max_value=20, value=DEFAULT_QUARTERS,
            label_visibility="collapsed",
        )
        years = n_quarters / 4
        years_label = f"{years:.1f}".rstrip("0").rstrip(".") + " yr"
        st.caption(f"{n_quarters} quarters ({years_label}) · ~{n_quarters * 2} API calls")

        # ── Data Sources ──
        st.markdown("<div class='rr-section-header' style='margin-top:16px;'>Data Sources</div>",
                    unsafe_allow_html=True)

        # Always-on
        st.markdown("**Always included**")
        st.caption("SEC 8-K — earnings releases and press releases")
        st.caption("Analyst consensus — for display context only")

        # Optional sources
        st.markdown("**Optional**")
        include_mda = st.checkbox(
            "SEC 10-Q MD&A",
            value=True,
            help="MD&A section — risk factor drift, cost language, liquidity signals.",
        )
        include_news = st.checkbox(
            "GDELT News",
            value=True,
            help="Global news tone and headlines over 180 days.",
        )
        include_peers = st.checkbox(
            "Supply-chain Peers",
            value=True,
            help="Up to 4 peer earnings releases for read-through demand signals.",
        )

        # ── Forecast Focus ──
        st.markdown("<div class='rr-section-header' style='margin-top:18px;'>Forecast Focus</div>",
                    unsafe_allow_html=True)

        from config import FORECAST_ITEMS as _ALL_ITEMS
        focus_items = st.multiselect(
            "Focus dimensions",
            options=_ALL_ITEMS,
            default=[],
            placeholder="All dimensions (no filter)",
            label_visibility="collapsed",
        )

        st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)
        run = st.button(
            "Run Analysis", type="primary",
            use_container_width=True,
            disabled=not ANTHROPIC_API_KEY or not ticker_input,
        )

        # Force refresh — clears disk cache for the current ticker so the next
        # run fetches live data from EDGAR, GDELT, and the Claude API.
        if st.button("Force Refresh Cache", use_container_width=True):
            import shutil
            from config import CACHE_DIR
            from src.extraction.llm_extractor import LLMExtractor
            removed = 0
            ticker_upper = ticker_input.upper()
            # Clear HTTP cache files related to this ticker
            if CACHE_DIR.exists():
                for f in CACHE_DIR.iterdir():
                    if f.is_file() and ticker_upper.lower() in f.name.lower():
                        f.unlink()
                        removed += 1
            # Clear LLM extraction cache
            llm_cache = LLMExtractor.CACHE_DIR
            if llm_cache.exists():
                # LLM cache keys embed the first 500 chars of content — clear all
                for f in llm_cache.iterdir():
                    if f.is_file():
                        f.unlink()
                        removed += 1
            st.success(f"Cache cleared ({removed} files). Re-run analysis for fresh data.")

        st.caption(
            "Stock price data is fetched for display context only — "
            "never used as a predictive signal. API cache auto-expires in 48h."
        )

        # ── API Cost ──
        _res = st.session_state.get("results") or {}
        _api = _res.get("api_usage") if _res else None
        if _api:
            st.markdown("<div class='rr-section-header' style='margin-top:16px;'>Last Analysis Cost</div>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:22px;font-weight:700;'>"
                f"${_api['cost_usd']:.4f}</div>"
                f"<div style='font-size:12px;opacity:0.65;margin-top:2px;'>"
                f"{_api['total_tokens']:,} tokens &nbsp;·&nbsp; "
                f"{_api['calls']} API calls &nbsp;·&nbsp; "
                f"{_api['cache_hits']} cached</div>",
                unsafe_allow_html=True,
            )

        # ── Saved Reports ──────────────────────────────────────────────────
        st.markdown("<div class='rr-section-header' style='margin-top:20px;'>Saved Reports</div>",
                    unsafe_allow_html=True)

        saved = list_reports()

        # Save current analysis
        if st.session_state.get("results"):
            default_name = (
                f"{st.session_state.results.get('ticker','?')}_"
                f"{datetime.now().strftime('%Y-%m-%d')}"
            )
            report_name = st.text_input(
                "Report name", value=default_name,
                label_visibility="collapsed", key="report_name_input",
            )
            if st.button("Save Analysis", use_container_width=True):
                try:
                    path = save_report(st.session_state.results, report_name)
                    st.success(f"Saved: {path.name}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("Run an analysis first to enable saving.")

        # Load / delete saved reports
        if saved:
            options = {f"{r['ticker']} — {r['name']} ({r['saved_at'][:10]})": r["stem"]
                       for r in saved}
            chosen_label = st.selectbox(
                "Load report", list(options.keys()),
                label_visibility="collapsed", key="report_load_select",
            )
            chosen_stem = options[chosen_label]
            lc, dc = st.columns(2)
            if lc.button("Load", use_container_width=True, key="load_btn"):
                loaded = load_report(chosen_stem)
                if loaded:
                    st.session_state.results = loaded
                    st.success("Report loaded.")
                    st.rerun()
                else:
                    st.error("Could not load — file may be corrupt.")
            if dc.button("Delete", use_container_width=True, key="del_btn"):
                delete_report(chosen_stem)
                st.rerun()
        else:
            st.caption("No saved reports yet.")

    return dict(
        ticker=ticker_input, n_quarters=n_quarters,
        focus_items=focus_items, include_mda=include_mda,
        include_news=include_news, include_peers=include_peers, run=run,
    )


# ── Analysis pipeline ─────────────────────────────────────────────────────────
def run_analysis(params: dict) -> dict:
    from src.data.sec_client       import fetch_eight_k, fetch_ten_q, fetch_ten_k, get_company_name, diff_mda
    from src.data.news_client      import (get_company_news, get_news_sentiment_timeline,
                                            prepare_articles)
    from src.data.company_client   import (get_company_info, get_analyst_estimates,
                                           get_earnings_history, get_supply_chain_peers,
                                           get_price_history, get_short_interest)
    from src.extraction.llm_extractor import LLMExtractor, reset_usage, get_usage
    from src.scoring.scorer        import RevisionScorer

    reset_usage()

    ticker     = params["ticker"]
    n_quarters = params["n_quarters"]
    results: dict = {"ticker": ticker, "error": None}

    with st.status("Running analysis…", expanded=True) as status:

        st.write("📋  Fetching company information…")
        company_info = get_company_info(ticker)
        company_name = company_info.get("name", ticker)
        results.update(company_info=company_info, company_name=company_name)

        if not ticker or len(ticker) < 1 or len(ticker) > 10:
            raise ValueError(f"Invalid ticker symbol: '{ticker}'. Please enter a valid ticker.")

        st.write(f"📂  Fetching {n_quarters} quarters of SEC 8-K earnings releases…")
        try:
            eight_k = fetch_eight_k(ticker, limit=n_quarters)
        except Exception as e:
            err = str(e)
            if "not found" in err.lower() or "cik" in err.lower():
                raise ValueError(
                    f"Ticker '{ticker}' was not found on SEC EDGAR. "
                    f"Check the symbol is correct and the company is publicly traded in the US."
                ) from e
            raise
        results["eight_k_filings"] = eight_k

        if not eight_k:
            raise ValueError(
                f"No SEC 8-K earnings filings found for '{ticker}' in the last {n_quarters} quarters. "
                f"The company may not have filed earnings releases, or the ticker may be incorrect."
            )

        # ── Staleness check ───────────────────────────────────────────────────
        try:
            _last_date = pd.to_datetime(eight_k[0].get("filingDate", ""), errors="coerce")
            _days_stale = (pd.Timestamp.now() - _last_date).days if not pd.isnull(_last_date) else None
        except Exception:
            _days_stale = None
        results["days_since_last_filing"] = _days_stale
        results["last_filing_date"]       = eight_k[0].get("filingDate", "")

        ten_q = []
        if params["include_mda"]:
            st.write("📄  Fetching 10-Q MD&A sections…")
            ten_q = fetch_ten_q(ticker, limit=n_quarters + 4)
            # Also fetch 10-K (annual report) — covers Q4 which never has a 10-Q
            st.write("📄  Fetching 10-K annual MD&A…")
            ten_k = fetch_ten_k(ticker, limit=2)
            # Merge: 10-K fills gaps where no 10-Q exists for that quarter
            _tq_quarters = {f["quarter"] for f in ten_q}
            for _k in ten_k:
                if _k["quarter"] not in _tq_quarters and _k.get("mda_text"):
                    ten_q.append(_k)
            # Re-sort newest first by filingDate
            ten_q.sort(key=lambda f: f.get("filingDate", ""), reverse=True)
        results["ten_q_filings"] = ten_q

        news_articles = pd.DataFrame()
        news_timeline = pd.DataFrame()
        if params["include_news"]:
            st.write("📰  Fetching GDELT news sentiment…")
            # Match the exact date span of the fetched 8-K filings
            try:
                _dates = [
                    pd.to_datetime(f.get("filingDate", ""), errors="coerce")
                    for f in eight_k
                ]
                _valid = [d for d in _dates if pd.notna(d)]
                _oldest = min(_valid) if _valid else None
                _news_days = (
                    max((pd.Timestamp.now() - _oldest).days + 7, 30)
                    if _oldest else n_quarters * 91
                )
            except Exception:
                _news_days = n_quarters * 91
            _raw_articles = get_company_news(ticker, company_name, days=_news_days)
            news_articles = prepare_articles(_raw_articles)
            news_timeline = get_news_sentiment_timeline(
                ticker, company_name, days=_news_days, articles_df=_raw_articles
            )
        results.update(news_articles=news_articles, news_timeline=news_timeline)

        st.write("📊  Fetching analyst consensus + market context…")
        results["analyst_data"]  = get_analyst_estimates(ticker)
        results["earnings_hist"] = get_earnings_history(ticker, quarters=8)
        results["price_history"] = get_price_history(ticker, period="2y")
        results["short_interest"] = get_short_interest(ticker)

        # ── Pre-compute transcript diffs (QoQ) ────────────────────────────────
        # Transcripts are filed 4-6 weeks after quarter end — more timely than
        # 10-Q MD&A (45+ days). Diff consecutive transcripts to get actionable
        # language-change signal even when 10-Q is missing.
        st.write("📝  Computing transcript language diffs…")
        transcript_diffs: dict[str, dict] = {}
        for i, filing in enumerate(eight_k):
            if i + 1 >= len(eight_k):
                break
            curr_text = filing.get("text", "")
            prev_text = eight_k[i + 1].get("text", "")
            curr_q    = filing.get("quarter", "")
            prev_q    = eight_k[i + 1].get("quarter", "")
            if curr_text and prev_text and curr_q:
                try:
                    td = diff_mda(curr_text, prev_text)
                    td["current_quarter"] = curr_q
                    td["prior_quarter"]   = prev_q
                    transcript_diffs[curr_q] = td
                except Exception:
                    pass
        results["transcript_diffs"] = transcript_diffs

        st.write(f"🤖  Running Claude extraction across {len(eight_k)} filings…")
        extractor    = LLMExtractor()
        quarter_data = []

        for i, filing in enumerate(eight_k):
            quarter = filing.get("quarter", f"Q{i+1}")
            st.write(f"    · {quarter} 8-K ({i+1}/{len(eight_k)})…")
            transcript_signals = None
            if filing.get("text"):
                # Pass the prior quarter's transcript (next index, since list is newest-first)
                prior_filing = eight_k[i + 1] if i + 1 < len(eight_k) else None
                prior_text = prior_filing.get("text") if prior_filing else None
                transcript_signals = extractor.extract_transcript(
                    text=filing["text"], ticker=ticker,
                    company_name=company_name, quarter=quarter,
                    prior_transcript_text=prior_text,
                )
            mda_signals = None
            mda_is_delta = False
            if params["include_mda"]:
                matched = next((f for f in ten_q if f.get("quarter") == quarter), None)
                if matched and matched.get("mda_text"):
                    st.write(f"    · {quarter} 10-Q MD&A…")
                    matched_idx = next(
                        (j for j, f in enumerate(ten_q) if f.get("quarter") == quarter), None
                    )
                    prior = (
                        ten_q[matched_idx + 1]
                        if matched_idx is not None and matched_idx + 1 < len(ten_q)
                        else None
                    )
                    if prior and prior.get("mda_text"):
                        mda_diff = diff_mda(matched["mda_text"], prior["mda_text"])
                        mda_diff["current_quarter"] = quarter
                        mda_diff["prior_quarter"]   = prior.get("quarter", "prior quarter")
                        mda_signals = extractor.extract_mda_delta(
                            diff=mda_diff, ticker=ticker, quarter=quarter,
                        )
                        mda_is_delta = True
                        # Store diff metadata keyed by quarter so Tab 2 can look up
                        # the correct diff info for any selected quarter
                        if "mda_diff_meta" not in results:
                            results["mda_diff_meta"] = {}
                        results["mda_diff_meta"][quarter] = {
                            "unchanged_ratio": mda_diff["unchanged_ratio"],
                            "current_quarter": mda_diff["current_quarter"],
                            "prior_quarter":   mda_diff["prior_quarter"],
                            "new_language":    mda_diff["new_language"],
                            "removed_language": mda_diff["removed_language"],
                        }
                    else:
                        mda_signals = extractor.extract_mda(
                            text=matched["mda_text"], ticker=ticker,
                            company_name=company_name, quarter=quarter,
                        )
            quarter_data.append(dict(
                quarter=quarter, filing=filing,
                transcript=transcript_signals, mda=mda_signals,
                mda_is_delta=mda_is_delta,
            ))
        results["quarter_data"] = quarter_data

        news_signals = None
        if params["include_news"] and not news_articles.empty:
            st.write("🌐  Synthesising news signals…")
            news_signals = extractor.extract_news(news_articles, ticker, company_name)
        results["news_signals"] = news_signals

        peer_signals: list[dict] = []
        peer_tone_scores: list[dict] = []
        # status dict: ticker → "ok" | "no_filing" | "no_text" | "no_cik" | "error:<msg>"
        peer_status: dict[str, str] = {}
        peer_source: dict = {"type": "none", "confidence": "low", "warning": None, "ticker_count": 0}

        if params["include_peers"]:
            peers, peer_source = get_supply_chain_peers(ticker, company_info=company_info)
            if peers:
                st.write(f"🔗  Analysing {len(peers)} supply-chain peers ({peer_source['type']} list)…")
                from src.data.company_client import get_company_info as gci
                for peer in peers:
                    try:
                        peer_filings = fetch_eight_k(peer, limit=1)
                        if not peer_filings:
                            peer_status[peer] = "no_filing"
                            st.write(f"    · {peer} — no SEC filing found (may be a foreign filer)")
                            continue
                        if not peer_filings[0].get("text"):
                            peer_status[peer] = "no_text"
                            st.write(f"    · {peer} — filing found but text could not be extracted")
                            continue
                        peer_info    = gci(peer)
                        peer_text    = peer_filings[0]["text"]
                        peer_quarter = peer_filings[0].get("quarter", "")
                        st.write(f"    · {peer} ({peer_info.get('name', peer)}) {peer_quarter}…")
                        # Read-through signal for the target company
                        sig = extractor.extract_peer_signal(
                            peer_text=peer_text,
                            peer_ticker=peer, peer_name=peer_info.get("name", peer),
                            target_ticker=ticker, target_name=company_name,
                        )
                        if sig:
                            sig.update(ticker=peer, name=peer_info.get("name", peer))
                            peer_signals.append(sig)
                        # Raw tone extraction for sector tone comparison
                        peer_ts = extractor.extract_transcript(
                            text=peer_text, ticker=peer,
                            company_name=peer_info.get("name", peer),
                            quarter=peer_quarter,
                        )
                        if peer_ts:
                            peer_tone_scores.append({
                                "ticker":                    peer,
                                "overall_tone":              peer_ts.overall_tone,
                                "hedging_intensity":         peer_ts.hedging_intensity,
                                "forward_guidance_strength": peer_ts.forward_guidance_strength,
                                "risk_language_escalation":  peer_ts.risk_language_escalation,
                                "demand_language_tone":      peer_ts.demand_language_tone,
                                "margin_language_tone":      peer_ts.margin_language_tone,
                            })
                        peer_status[peer] = "ok"
                    except Exception as exc:
                        err = str(exc)
                        if "not found" in err.lower():
                            peer_status[peer] = "no_cik"
                            st.write(f"    · {peer} — not found on SEC EDGAR")
                        else:
                            peer_status[peer] = f"error: {err[:60]}"
                            st.write(f"    · {peer} — fetch error")
            else:
                st.write("🔗  No supply-chain peers identified for this ticker.")

        results["peer_signals"]     = peer_signals
        results["peer_tone_scores"] = peer_tone_scores
        results["peer_status"]      = peer_status
        results["peer_source"]      = peer_source

        st.write("⚡  Computing revision scores…")
        analysis = RevisionScorer.build_full_analysis(
            ticker=ticker, company_name=company_name,
            quarter_data=quarter_data, news=news_signals,
            peer_signals=results.get("peer_signals", []),
        )
        results["analysis"] = analysis

        st.write("✍️  Generating analyst revision narrative…")
        signals_dict = {
            "overall": {
                "probability": analysis.overall_probability,
                "direction":   analysis.overall_direction,
                "magnitude":   analysis.overall_magnitude,
                "trend":       analysis.trend_direction,
            },
            "quarterly": [
                {"quarter": qd["quarter"],
                 "score": next((q.weighted_score for q in analysis.quarters
                                if q.quarter == qd["quarter"]), 0.0),
                 "transcript": qd["transcript"].model_dump() if qd.get("transcript") else {},
                 "mda":        qd["mda"].model_dump() if qd.get("mda") else {}}
                for qd in quarter_data
            ],
            "news": news_signals.model_dump() if news_signals else {},
        }
        consensus_lines = []
        ad = results["analyst_data"]
        if ad.get("eps_estimate"):
            consensus_lines.append("EPS estimates available.")
        if ad.get("eps_revisions"):
            consensus_lines.append("Recent analyst revision counts available.")
        narrative = extractor.synthesize_narrative(
            all_signals=signals_dict, ticker=ticker,
            company_name=company_name,
            consensus_summary=" ".join(consensus_lines) or "No consensus data.",
            focus_items=params.get("focus_items", []),
        )
        analysis.narrative = narrative
        results["analysis"] = analysis

        # ── Falsifiability: signal vs realized EPS surprises ──────────────
        from src.scoring.falsifiability import compute_falsifiability
        _fals_eh = results.get("earnings_hist", pd.DataFrame())
        results["falsifiability"] = compute_falsifiability(
            analysis.quarters, _fals_eh
        )

        usage = get_usage()
        results["api_usage"] = usage
        status.update(label="✅  Analysis complete", state="complete", expanded=False)

    # Show cost immediately after the status widget closes
    _api = results.get("api_usage", {})
    if _api:
        st.caption(
            f"API cost: **${_api['cost_usd']:.4f}** "
            f"· {_api['total_tokens']:,} tokens "
            f"· {_api['calls']} calls "
            f"· {_api['cache_hits']} cached"
        )

    return results


# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard(results: dict):
    import plotly.graph_objects as go
    import math as _math
    from src.scoring.scorer import magnitude_display
    from src.visualization.charts import (
        waterfall_chart, news_sentiment_timeline, analyst_estimate_chart,
        peer_heatmap, price_history_chart, sector_tone_comparison,
    )

    # Suppress Streamlit's scroll-to-top and tab-reset on widget reruns.
    st.markdown("""
<script>
(function() {
    window.scrollTo = () => {};

    if (window._rrTab) return;
    window._rrTab = { savedIdx: 0, restoring: false };

    // Track which tab the user clicked
    document.addEventListener('click', function(e) {
        if (window._rrTab.restoring) return;
        const tab = e.target.closest('[data-baseweb="tab"]');
        if (!tab) return;
        const all = document.querySelectorAll('[data-baseweb="tab"]');
        window._rrTab.savedIdx = Array.from(all).indexOf(tab);
    }, true);

    // Watch for Streamlit resetting the active tab back to 0 after rerun.
    // Observe documentElement — it is never replaced by Streamlit's partial
    // re-renders, so this observer stays alive for the lifetime of the page.
    new MutationObserver(function() {
        const s = window._rrTab.savedIdx;
        if (window._rrTab.restoring || s === 0) return;
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        if (!tabs.length || !tabs[s]) return;
        const active = Array.from(tabs).findIndex(
            t => t.getAttribute('aria-selected') === 'true'
        );
        if (active === 0) {
            window._rrTab.restoring = true;
            tabs[s].click();
            setTimeout(() => { window._rrTab.restoring = false; }, 500);
        }
    }).observe(document.documentElement, {
        subtree: true,
        attributeFilter: ['aria-selected']
    });
})();
</script>
""", unsafe_allow_html=True)

    # Guard: if results are incomplete (e.g. analysis failed mid-run), bail out cleanly
    if not results.get("analysis") or not results.get("company_info"):
        st.error("Analysis results are incomplete. Please run the analysis again.")
        return

    # ── Staleness warning ─────────────────────────────────────────────────────
    _days_stale = results.get("days_since_last_filing")
    _last_dt    = results.get("last_filing_date", "")
    if _days_stale is not None:
        if _days_stale > 180:
            st.error(
                f"**Stale data warning:** The most recent 8-K filing is **{_days_stale} days old** "
                f"(filed {_last_dt}). This company may be a late filer, have a non-standard fiscal "
                f"year, or have not yet filed its most recent earnings release. Signals are based on "
                f"old disclosures and may not reflect current conditions."
            )
        elif _days_stale > 100:
            st.warning(
                f"**Filing lag notice:** The most recent 8-K is {_days_stale} days old (filed {_last_dt}). "
                f"A new earnings release may be due soon — consider re-running after the next filing."
            )

    analysis     = results["analysis"]
    company_info = results.get("company_info", {})
    company_name = results.get("company_name", results.get("ticker", ""))
    ticker       = results.get("ticker", "")

    # ── Company strip ─────────────────────────────────────────────────────────
    from src.data.company_client import format_market_cap
    mcap   = format_market_cap(company_info.get("market_cap"))
    dir_c  = (COLOR_POSITIVE if analysis.overall_direction == "upward"
              else COLOR_NEGATIVE if analysis.overall_direction == "downward"
              else COLOR_NEUTRAL)
    pct    = int(analysis.overall_probability * 100)
    mag_label, mag_dots = magnitude_display(analysis.overall_magnitude)
    conf_pct = int(analysis.overall_confidence * 100)
    trend_icon = {"improving":"↗","deteriorating":"↘","stable":"→"}.get(analysis.trend_direction,"→")

    c1, c2, c3, c4 = st.columns([0.38, 0.20, 0.20, 0.20], gap="small")
    with c1:
        st.markdown(
            f"<div class='rr-card' style='height:80px;display:flex;flex-direction:column;"
            f"justify-content:center;'>"
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span class='rr-pill'>{ticker}</span>"
            f"<span style='font-size:16px;font-weight:700;'>{company_name}</span>"
            f"</div>"
            f"<div style='font-size:11px;color:rgba(128,128,128,0.75);margin-top:5px;'>"
            f"{company_info.get('sector','')} &middot; {company_info.get('industry','')} &middot; {mcap}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Revision Conviction", f"{pct}%",
                  delta=f"↑ upward" if analysis.overall_direction=="upward"
                        else f"↓ downward" if analysis.overall_direction=="downward"
                        else "→ neutral")
    with c3:
        st.metric("Expected Magnitude", mag_label,
                  delta=f"{mag_dots}" if mag_dots else None)
    with c4:
        st.metric("Signal Confidence", f"{conf_pct}%",
                  delta=f"{trend_icon} {analysis.trend_direction}")

    # ── Peer adjustment badge ─────────────────────────────────────────────────
    _top_peer_adj = analysis.signal_contributions.get("peer_signals", 0.0)
    if _top_peer_adj != 0.0:
        _pa_color = "#2ecc71" if _top_peer_adj > 0 else "#ff4d6d"
        _pa_label = "upward" if _top_peer_adj > 0 else "downward"
        _n_peers  = len(results.get("peer_signals", []))
        st.markdown(
            f"<div style='display:inline-flex;align-items:center;gap:6px;"
            f"background:rgba(255,255,255,0.04);border:1px solid {_pa_color}44;"
            f"border-radius:6px;padding:4px 10px;font-size:12px;margin-bottom:4px;'>"
            f"<span style='color:{_pa_color};font-weight:700;'>"
            f"{'↑' if _top_peer_adj > 0 else '↓'} {_top_peer_adj:+.3f}</span>"
            f"<span style='color:rgba(200,200,200,0.7);'>"
            f"peer signal adjustment ({_n_peers} peer{'s' if _n_peers != 1 else ''} · "
            f"{_pa_label} pressure on score)</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reusable helpers ──────────────────────────────────────────────────────
    def render_quote_card(text: str, color: str) -> None:
        """Render a single quoted phrase card with a colored left border."""
        if not text:
            return
        border_color = {
            "negative": COLOR_NEGATIVE,
            "positive": COLOR_POSITIVE,
            "neutral":  COLOR_NEUTRAL,
            "blue":     "#4A90D9",
        }.get(color, COLOR_NEUTRAL)
        st.markdown(
            f"<div style='border-left:3px solid {border_color};"
            f"background:var(--secondary-background-color);"
            f"padding:9px 14px;margin-bottom:7px;border-radius:0 6px 6px 0;"
            f"font-size:13px;line-height:1.55;font-style:italic;'>"
            f"&ldquo;{text}&rdquo;</div>",
            unsafe_allow_html=True,
        )

    def render_bar(label: str, value: float,
                   min_val: float = -1.0, max_val: float = 1.0,
                   label_width: int = 165) -> None:
        """Render a horizontal signal bar on [min_val, max_val]."""
        bc = (COLOR_POSITIVE if value > 0.05 else
              COLOR_NEGATIVE if value < -0.05 else COLOR_NEUTRAL)
        span    = max_val - min_val if max_val != min_val else 1.0
        pct_bar = max(0, min(100, int(((value - min_val) / span) * 100)))
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:7px;'>"
            f"<div style='width:{label_width}px;font-size:12px;"
            f"color:rgba(128,128,128,0.75);flex-shrink:0;'>{label}</div>"
            f"<div style='flex:1;background:rgba(128,128,128,0.2);border-radius:4px;height:7px;'>"
            f"<div style='width:{pct_bar}%;background:{bc};height:100%;"
            f"border-radius:4px;'></div></div>"
            f"<div style='width:42px;font-size:12px;color:{bc};"
            f"text-align:right;font-family:monospace;'>{value:+.2f}</div></div>",
            unsafe_allow_html=True,
        )

    def _net_revision(eps_rev) -> float:
        """Net up-revisions minus down-revisions (handles DataFrame or dict)."""
        if eps_rev is None:
            return 0.0
        try:
            if isinstance(eps_rev, pd.DataFrame):
                up   = float(eps_rev["Up"].sum())   if "Up"   in eps_rev.columns else 0.0
                down = float(eps_rev["Down"].sum()) if "Down" in eps_rev.columns else 0.0
                return up - down
            elif isinstance(eps_rev, dict):
                up_d   = eps_rev.get("Up",   {}) or {}
                down_d = eps_rev.get("Down", {}) or {}
                def _sum_d(d: dict) -> float:
                    return sum(
                        v for v in d.values()
                        if v is not None and not (isinstance(v, float) and _math.isnan(v))
                    )
                return _sum_d(up_d) - _sum_d(down_d)
        except Exception:
            pass
        return 0.0

    def _revision_counts(eps_rev) -> tuple:
        """Return (up_30d, down_30d) for the most recent period."""
        if eps_rev is None:
            return None, None
        try:
            if isinstance(eps_rev, pd.DataFrame):
                up_c   = eps_rev["Up"]   if "Up"   in eps_rev.columns else None
                down_c = eps_rev["Down"] if "Down" in eps_rev.columns else None
                if up_c is not None and down_c is not None:
                    return float(up_c.iloc[0]), float(down_c.iloc[0])
            elif isinstance(eps_rev, dict):
                up_d   = eps_rev.get("Up",   {}) or {}
                down_d = eps_rev.get("Down", {}) or {}
                # Use is-None check so that a count of 0 is preserved (not treated as falsy)
                up_30   = up_d.get("0M")   if "0M"   in up_d   else (list(up_d.values())[0]   if up_d   else None)
                down_30 = down_d.get("0M") if "0M"   in down_d else (list(down_d.values())[0] if down_d else None)
                return up_30, down_30
        except Exception:
            pass
        return None, None

    def _gap_chip(arrow: str, color: str, text: str) -> None:
        """Render a colored directional gap chip."""
        st.markdown(
            f"<div style='background:{color}15;border:1px solid {color}33;"
            f"border-radius:6px;padding:8px 10px;margin-bottom:8px;'>"
            f"<span style='font-size:16px;font-weight:700;color:{color};'>{arrow}</span>"
            f"<span style='font-size:12px;margin-left:6px;'>{text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📋  The Case",
        "🔍  The Evidence",
        "📊  Peers & Context",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — The Case
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        # Convenience aliases
        n   = analysis.narrative
        qs0 = analysis.quarters[0] if analysis.quarters else None
        t0  = qs0.transcript if qs0 else None
        m0  = qs0.mda        if qs0 else None
        ad  = results.get("analyst_data", {}) or {}
        si  = results.get("short_interest", {}) or {}
        eps_rev          = ad.get("eps_revisions")
        net_rev          = _net_revision(eps_rev)
        has_rev_data     = eps_rev is not None  # distinguish "no data" from "zero net"
        si_pct      = si.get("short_percent_of_float")
        si_pct_num  = (si_pct * 100) if si_pct is not None else None
        _n = analysis.narrative

        # ── SECTION A — Signal Header ──────────────────────────────────────
        dir_arrow = {"upward": "↑", "downward": "↓", "neutral": "→"}.get(
            analysis.overall_direction, "→")
        dir_text  = {"upward": "UPWARD REVISION SIGNAL",
                     "downward": "DOWNWARD REVISION SIGNAL",
                     "neutral": "NEUTRAL"}.get(analysis.overall_direction, "NEUTRAL")
        _horizon_raw = (analysis.narrative.revision_horizon
                        if analysis.narrative and
                        getattr(analysis.narrative, "revision_horizon", None)
                        else None)
        _horizon_label = {
            "30_days":    "~30 days",
            "60_days":    "~60 days",
            "90_days":    "~90 days",
            "next_quarter": "next quarter",
        }.get(_horizon_raw, "")
        _horizon_badge = (
            f"<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
            f"font-size:13px;font-weight:700;border:1px solid rgba(128,128,128,0.3);"
            f"color:rgba(128,128,128,0.85);'>⏱ {_horizon_label}</span>"
            if _horizon_label else ""
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:14px;padding:16px 0 8px 0;'>"
            f"<div style='font-size:30px;font-weight:900;color:{dir_c};"
            f"letter-spacing:-0.01em;'>{dir_arrow} {dir_text}</div>"
            f"<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
            f"font-size:13px;font-weight:700;border:1px solid {dir_c}44;"
            f"color:{dir_c};background:{dir_c}11;'>{mag_label.title()}</span>"
            f"<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
            f"font-size:13px;font-weight:700;border:1px solid rgba(128,128,128,0.3);"
            f"color:rgba(128,128,128,0.85);'>{trend_icon} {analysis.trend_direction.title()}</span>"
            f"{_horizon_badge}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── BLOCK 2 — THE THESIS ───────────────────────────────────────────
        st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

        # Narrative thesis (executive summary)
        _thesis_text = getattr(_n, "executive_summary", "") if _n else ""
        _thesis_failed = (
            not _thesis_text or
            _thesis_text.startswith("Narrative synthesis failed") or
            _thesis_text == "Analysis could not be completed due to an API error."
        )
        if not _thesis_failed:
            st.markdown(
                f"<div class='rr-card' style='font-size:14px;line-height:1.75;"
                f"border-color:{dir_c}33;margin:4px 0 12px 0;'>"
                f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.10em;color:rgba(128,128,128,0.6);margin-bottom:8px;'>"
                f"AI Revision Thesis</div>"
                f"{_safe_md(_thesis_text)}</div>",
                unsafe_allow_html=True,
            )
        elif _n is not None:
            st.caption("⚠ Narrative synthesis failed — re-run to regenerate.")

        # Headline claim — what analysts are missing
        headline_text = ""
        if n and getattr(n, "what_analysts_are_missing", ""):
            headline_text = n.what_analysts_are_missing
        elif t0 and t0.analyst_revision_rationale:
            headline_text = t0.analyst_revision_rationale
        if headline_text:
            st.markdown(
                f"<div style='background:{dir_c}0d;border-left:4px solid {dir_c};"
                f"border-radius:0 8px 8px 0;padding:14px 18px;margin:8px 0 18px 0;"
                f"font-size:14px;line-height:1.7;'>{_safe_md(headline_text)}</div>",
                unsafe_allow_html=True,
            )

        # ── SECTION C2 — Bull / Bear / Watch (narrative) ──────────────────
        if _n is not None and (_n.bull_case or _n.bear_case or _n.watch_items
                               or _n.key_risk_to_thesis or _n.primary_drivers):
            b1, b2, b3, b4 = st.columns(4, gap="small")
            with b1:
                st.markdown(
                    "<div class='rr-section-header' style='color:#2ecc71;'>"
                    "🐂 Bull Case</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rr-card' style='border-color:#2ecc7133;"
                    f"font-size:13px;line-height:1.65;'>"
                    f"{_safe_md(_n.bull_case or '—')}</div>",
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    "<div class='rr-section-header' style='color:#e74c3c;'>"
                    "🐻 Bear Case</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rr-card' style='border-color:#e74c3c33;"
                    f"font-size:13px;line-height:1.65;'>"
                    f"{_safe_md(_n.bear_case or '—')}</div>",
                    unsafe_allow_html=True,
                )
            with b3:
                st.markdown(
                    "<div class='rr-section-header'>👀 Watch Items</div>",
                    unsafe_allow_html=True,
                )
                if _n.watch_items:
                    items_html = "".join(
                        f"<li style='font-size:13px;line-height:1.65;"
                        f"margin-bottom:4px;'>{_safe_md(w)}</li>"
                        for w in _n.watch_items
                    )
                    st.markdown(
                        f"<div class='rr-card' style='border-color:#f39c1233;'>"
                        f"<ul style='margin:0;padding-left:16px;'>{items_html}</ul></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='rr-card' style='border-color:#f39c1233;"
                        "font-size:13px;'>—</div>",
                        unsafe_allow_html=True,
                    )
            with b4:
                st.markdown(
                    "<div class='rr-section-header' style='color:#e67e22;'>"
                    "⚠️ Key Risk</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='rr-card' style='border-color:#e67e2233;"
                    f"font-size:13px;line-height:1.65;'>"
                    f"{_safe_md(_n.key_risk_to_thesis or '—')}</div>",
                    unsafe_allow_html=True,
                )

            # ── Primary Drivers (always shown when available) ──────────────
            _drivers = [d for d in (_n.primary_drivers or []) if isinstance(d, dict)]
            if _drivers:
                st.markdown(
                    "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.5);margin:14px 0 6px;'>"
                    "Primary Drivers</div>",
                    unsafe_allow_html=True,
                )
                _drv_cols = st.columns(min(len(_drivers), 3), gap="small")
                for _dc, driver in zip(_drv_cols, _drivers[:3]):
                    d_c = (COLOR_POSITIVE if driver.get("direction") == "upward" else
                           COLOR_NEGATIVE if driver.get("direction") == "downward" else
                           COLOR_NEUTRAL)
                    d_arr = "↑" if driver.get("direction") == "upward" else "↓" if driver.get("direction") == "downward" else "→"
                    _dc.markdown(
                        f"<div class='rr-card' style='border-color:{d_c}33;'>"
                        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:6px;'>"
                        f"<span style='color:{d_c};font-weight:800;font-size:16px;'>{d_arr}</span>"
                        f"<span style='font-weight:700;font-size:13px;'>{driver.get('factor','')}</span>"
                        f"</div>"
                        f"<div style='font-size:11px;color:rgba(128,128,128,0.75);line-height:1.55;'>"
                        f"{driver.get('evidence','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Language diff variables (displayed inside left column) ───────────
        _transcript_diffs = results.get("transcript_diffs") or {}
        _diff_meta_t1     = (results.get("mda_diff_meta") or {})
        _all_diffs = [
            (qs.quarter, _transcript_diffs[qs.quarter], "transcript")
            for qs in analysis.quarters
            if qs.quarter in _transcript_diffs
        ]
        # ── SECTION C — Three columns ──────────────────────────────────────
        left_col, mid_col, right_col = st.columns([0.38, 0.26, 0.36], gap="medium")

        with left_col:
            st.markdown("<div class='rr-section-header'>What the filings show</div>",
                        unsafe_allow_html=True)
            if not qs0:
                st.info("No filing data available.")
            else:
                # ── Three Core Signals (compact inline row) ───────────────────
                _tvpq_val  = t0.tone_vs_prior_quarter if t0 else None
                _guid_val  = (qs0.normalized.get("guidance_quantification")
                              if qs0 and qs0.normalized else None)
                if _guid_val is None and t0:
                    _guid_val = (t0.guidance_quantification_rate - 0.5) * 2
                _removed_count = 0
                if _all_diffs:
                    _rem_text = _all_diffs[0][1].get("removed_language", "") or ""
                    _removed_count = len([l for l in _rem_text.split("\n") if l.strip()])

                def _chip_color(val, fmt):
                    if val is None: return "rgba(128,128,128,0.4)"
                    if fmt == "count":
                        return COLOR_NEGATIVE if val > 3 else COLOR_WARNING if val > 0 else COLOR_POSITIVE
                    return COLOR_NEGATIVE if val < -0.1 else COLOR_POSITIVE if val > 0.1 else "rgba(200,200,200,0.5)"

                def _chip_val(val, fmt):
                    if val is None: return "—"
                    return str(int(val)) if fmt == "count" else f"{val:+.2f}"

                _cs = [
                    ("① Tone Shift",        "QoQ",          _tvpq_val,     "signed"),
                    ("② Guidance",          "specificity Δ",_guid_val,     "signed"),
                    ("③ Disappeared",       "lines",         _removed_count if _all_diffs else None, "count"),
                ]
                _chips_html = "".join(
                    f"<div style='flex:1;border:1px solid {_chip_color(v,f)}44;"
                    f"border-radius:6px;padding:6px 8px;'>"
                    f"<div style='font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
                    f"color:rgba(128,128,128,0.55);margin-bottom:2px;'>{lbl}</div>"
                    f"<span style='font-size:17px;font-weight:800;color:{_chip_color(v,f)};'>"
                    f"{_chip_val(v,f)}</span>"
                    f"<span style='font-size:10px;color:rgba(128,128,128,0.5);margin-left:4px;'>{sub}</span>"
                    f"</div>"
                    for lbl, sub, v, f in _cs
                )
                st.markdown(
                    f"<div style='display:flex;gap:6px;margin-bottom:10px;'>{_chips_html}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

                # Cluster 1 — Management Posture
                st.markdown(
                    "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.55);margin:10px 0 6px;'>"
                    "Management Posture</div>",
                    unsafe_allow_html=True,
                )
                if t0:
                    _tvpq_case_label = (
                        "Tone vs Prior Qtr (actual)"
                        if len(analysis.quarters) > 1 and analysis.quarters[1].transcript is not None
                        else "Tone vs Prior Qtr (est.)"
                    )
                    render_bar(_tvpq_case_label, t0.tone_vs_prior_quarter)
                    if t0.tone_vs_prior_quarter < 0 and t0.key_hedging_phrases:
                        render_quote_card(t0.key_hedging_phrases[0], "negative")
                    elif t0.tone_vs_prior_quarter >= 0 and t0.key_bullish_phrases:
                        render_quote_card(t0.key_bullish_phrases[0], "positive")
                    render_bar("Hedging (inv.)",      -t0.hedging_intensity)
                    if t0.hedging_intensity > 0.5 and len(t0.key_hedging_phrases) > 1:
                        render_quote_card(t0.key_hedging_phrases[1], "negative")
                    render_bar("Q&A Openness",        -t0.qa_deflection_score)
                else:
                    st.info("No transcript data available.")

                st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

                # Cluster 2 — Forward Signals
                st.markdown(
                    "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.55);margin:8px 0 6px;'>"
                    "Forward Signals</div>",
                    unsafe_allow_html=True,
                )
                if t0:
                    render_bar("Forward Guidance",    t0.forward_guidance_strength)
                    # Use QoQ delta from scorer when two quarters are available
                    _qs0_guid = qs0.normalized.get("guidance_quantification") if qs0 and qs0.normalized else None
                    _has_prior_q = len(analysis.quarters) > 1 and analysis.quarters[1].transcript is not None
                    if _qs0_guid is not None and _has_prior_q:
                        render_bar("Δ Guidance Specificity", _qs0_guid)
                    else:
                        render_bar("Guidance Specificity",
                                   (t0.guidance_quantification_rate - 0.5) * 2)
                if m0:
                    if m0.key_quotes:
                        render_quote_card(m0.key_quotes[0], "blue")
                elif t0 and len(t0.key_bullish_phrases) > 1:
                    render_quote_card(t0.key_bullish_phrases[1], "positive")
                if not t0 and not m0:
                    st.info("No forward signals data available.")

                st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

                # Cluster 3 — Risk Profile
                st.markdown(
                    "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.55);margin:8px 0 6px;'>"
                    "Risk Profile</div>",
                    unsafe_allow_html=True,
                )
                if t0:
                    render_bar("Risk Lang. Escalation", -t0.risk_language_escalation)
                if m0 and m0.new_risk_factors:
                    st.markdown(
                        "<div style='font-size:11px;color:rgba(128,128,128,0.6);"
                        "margin:6px 0 3px;'>New risk factors:</div>",
                        unsafe_allow_html=True,
                    )
                    for rf in m0.new_risk_factors[:3]:
                        st.markdown(
                            f"<div style='font-size:12px;color:{COLOR_NEGATIVE};"
                            f"padding:1px 0 1px 10px;'>• {_safe_md(rf)}</div>",
                            unsafe_allow_html=True,
                        )
                if not t0 and not m0:
                    st.info("No risk profile data available.")

        with mid_col:
            st.markdown("<div class='rr-section-header'>The Gap</div>",
                        unsafe_allow_html=True)
            if not qs0:
                st.info("—")
            else:
                hed  = t0.hedging_intensity         if t0 else 0.0
                tone = t0.overall_tone              if t0 else 0.0
                fgs  = t0.forward_guidance_strength if t0 else 0.0
                risk_esc = (m0.risk_escalation if m0 else
                            (t0.risk_language_escalation if t0 else 0.0))

                st.markdown(
                    "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.5);margin:10px 0 4px;'>"
                    "Posture vs Consensus</div>",
                    unsafe_allow_html=True,
                )
                if not has_rev_data:
                    # No consensus revision data — surface strong standalone signals
                    if hed > 0.5:
                        _gap_chip("←", COLOR_NEGATIVE,
                                  "High hedging intensity — watch for downward revision")
                    elif tone > 0.4:
                        _gap_chip("→", COLOR_POSITIVE,
                                  "Strong positive tone — potential upward revision catalyst")
                    elif tone < -0.2:
                        _gap_chip("←", COLOR_NEGATIVE,
                                  "Cautious management tone — potential downward revision risk")
                    else:
                        _gap_chip("—", "#888888",
                                  "No analyst revision data available to compare")
                elif hed > 0.6 and net_rev > 0:
                    _gap_chip("←", COLOR_NEGATIVE,
                              "Analysts pricing improvement; filings show hedging")
                elif tone < -0.2 and net_rev > 0:
                    _gap_chip("←", COLOR_NEGATIVE,
                              "Weak tone contradicts positive consensus")
                elif tone > 0.3 and net_rev < 0:
                    _gap_chip("→", COLOR_POSITIVE,
                              "Positive tone vs net analyst downgrades — potential upside miss")
                elif hed < 0.2 and net_rev < 0:
                    _gap_chip("→", COLOR_POSITIVE,
                              "Low hedging contradicts negative consensus")
                else:
                    _gap_chip("—", COLOR_NEUTRAL,
                              "Signals in line with consensus direction")

                st.markdown(
                    "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.5);margin:14px 0 4px;'>"
                    "Guidance vs Consensus</div>",
                    unsafe_allow_html=True,
                )
                # Use guidance delta (QoQ specificity change) as primary signal where available
                _guid_delta_gap = (
                    qs0.normalized.get("guidance_quantification")
                    if qs0 and qs0.normalized else None
                )
                _guid_sig = _guid_delta_gap if _guid_delta_gap is not None else fgs
                if not has_rev_data:
                    if _guid_sig < -0.25:
                        _gap_chip("←", COLOR_NEGATIVE,
                                  "Guidance specificity declining vs prior quarter")
                    elif _guid_sig > 0.25:
                        _gap_chip("→", COLOR_POSITIVE,
                                  "Guidance becoming more specific vs prior quarter")
                    else:
                        _gap_chip("—", "#888888",
                                  "No analyst revision data available to compare")
                elif _guid_sig < -0.25 and net_rev > 0:
                    _gap_chip("←", COLOR_NEGATIVE,
                              "Consensus stable; management giving less specific guidance than prior quarter")
                elif _guid_sig < -0.25:
                    _gap_chip("←", COLOR_NEGATIVE,
                              "Guidance specificity declining — aligns with analyst caution")
                elif _guid_sig > 0.25 and net_rev < 0:
                    _gap_chip("→", COLOR_POSITIVE,
                              "More specific guidance than prior quarter despite net negative revisions")
                elif _guid_sig > 0.25:
                    _gap_chip("→", COLOR_POSITIVE,
                              "Guidance becoming more specific — bullish signal")
                else:
                    _gap_chip("—", COLOR_NEUTRAL,
                              "Guidance in line with consensus direction")

                st.markdown(
                    "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.5);margin:14px 0 4px;'>"
                    "Risk vs Positioning</div>",
                    unsafe_allow_html=True,
                )
                if risk_esc > 0.4 and si_pct_num is not None and si_pct_num < 7:
                    _gap_chip("←", COLOR_NEGATIVE,
                              "Low short interest despite escalating risk language")
                elif risk_esc > 0.4:
                    _gap_chip("←", COLOR_NEGATIVE,
                              "Escalating risk language — potential downward revision risk")
                elif risk_esc < -0.2 and si_pct_num is not None and si_pct_num > 10:
                    _gap_chip("→", COLOR_POSITIVE,
                              "Risk language easing while short interest remains elevated — squeeze setup")
                elif risk_esc < -0.2:
                    _gap_chip("→", COLOR_POSITIVE,
                              "Risk language declining — filing less cautious than prior quarter")
                elif not has_rev_data:
                    _gap_chip("—", "#888888",
                              "No analyst revision data available to compare")
                else:
                    _gap_chip("—", COLOR_NEUTRAL,
                              "Risk language in line with consensus positioning")

                # ── Lead-time & coverage context ──────────────────────────
                _days_lag  = results.get("days_since_last_filing")
                _n_analysts = company_info.get("analyst_count")

                st.markdown(
                    "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.5);margin:14px 0 4px;'>"
                    "Signal Context</div>",
                    unsafe_allow_html=True,
                )

                if _days_lag is not None:
                    _lag_color = (COLOR_POSITIVE if _days_lag <= 7
                                  else COLOR_WARNING if _days_lag <= 21
                                  else "rgba(128,128,128,0.5)")
                    _lag_note  = ("filed recently — likely ahead of model updates"
                                  if _days_lag <= 7
                                  else "analysts may still be updating models"
                                  if _days_lag <= 21
                                  else "likely reflected in consensus by now")
                    st.markdown(
                        f"<div style='font-size:12px;margin-bottom:6px;'>"
                        f"<span style='color:{_lag_color};font-weight:700;'>"
                        f"8-K filed {_days_lag}d ago</span>"
                        f"<span style='color:rgba(128,128,128,0.55);'> — {_lag_note}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if _n_analysts is not None:
                    _cov_color = (COLOR_POSITIVE if _n_analysts <= 5
                                  else COLOR_WARNING if _n_analysts <= 15
                                  else "rgba(128,128,128,0.5)")
                    _cov_note  = ("thin coverage — revision lag largest here"
                                  if _n_analysts <= 5
                                  else "moderate coverage"
                                  if _n_analysts <= 15
                                  else "well-covered — faster consensus updates")
                    st.markdown(
                        f"<div style='font-size:12px;'>"
                        f"<span style='color:{_cov_color};font-weight:700;'>"
                        f"{_n_analysts} analyst{'s' if _n_analysts != 1 else ''}</span>"
                        f"<span style='color:rgba(128,128,128,0.55);'> — {_cov_note}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                _mktcap = company_info.get("market_cap")
                _is_largecap = (
                    (_mktcap and _mktcap > 10_000_000_000) or
                    (_n_analysts and _n_analysts > 20)
                )
                if _is_largecap:
                    st.markdown(
                        f"<div style='margin-top:8px;font-size:11px;line-height:1.5;"
                        f"color:rgba(180,140,60,0.9);background:rgba(255,200,0,0.06);"
                        f"border:1px solid rgba(255,200,0,0.2);border-radius:5px;"
                        f"padding:6px 10px;'>"
                        f"⚠ Large-cap — MD&A language tends to be heavily reviewed by legal. "
                        f"Signal reliability is lower; mid-cap names show larger revision lag.</div>",
                        unsafe_allow_html=True,
                    )

        with right_col:
            st.markdown("<div class='rr-section-header'>What consensus shows</div>",
                        unsafe_allow_html=True)
            with st.container(border=True):
                # EPS estimate
                eps_est = ad.get("eps_estimate")
                if eps_est:
                    try:
                        avg_d  = (eps_est.get("avg", {}) or {}) if isinstance(eps_est, dict) else {}
                        # Use is-in check so that a zero EPS estimate is preserved
                        cq_avg = avg_d.get("0q") if "0q" in avg_d else (list(avg_d.values())[0] if avg_d else None)
                        if cq_avg is not None:
                            st.metric("EPS Estimate (Current Qtr)", f"${float(cq_avg):.2f}")
                        else:
                            st.caption("EPS estimate available (multiple periods)")
                    except Exception:
                        st.caption("EPS estimate available")
                else:
                    st.caption("EPS estimate: N/A")

                # Revenue estimate
                rev_est = ad.get("revenue_estimate")
                if rev_est:
                    try:
                        avg_d  = (rev_est.get("avg", {}) or {}) if isinstance(rev_est, dict) else {}
                        cq_avg = avg_d.get("0q") if "0q" in avg_d else (list(avg_d.values())[0] if avg_d else None)
                        if cq_avg is not None:
                            rev_b = float(cq_avg)
                            # Dynamic suffix: T / B / M
                            if rev_b >= 1e12:
                                rev_str = f"${rev_b/1e12:.2f}T"
                            elif rev_b >= 1e9:
                                rev_str = f"${rev_b/1e9:.1f}B"
                            elif rev_b >= 1e6:
                                rev_str = f"${rev_b/1e6:.0f}M"
                            else:
                                rev_str = f"${rev_b:,.0f}"
                            st.metric("Revenue Estimate (Current Qtr)", rev_str)
                        else:
                            st.caption("Revenue estimate available (multiple periods)")
                    except Exception:
                        st.caption("Revenue estimate available")
                else:
                    st.caption("Revenue estimate: N/A")

                st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

                # Revision balance
                up_30, dn_30 = _revision_counts(eps_rev)
                if up_30 is not None or dn_30 is not None:
                    up_str = str(int(up_30)) if up_30 is not None else "?"
                    dn_str = str(int(dn_30)) if dn_30 is not None else "?"
                    net_c  = (COLOR_POSITIVE if net_rev > 0 else
                              COLOR_NEGATIVE if net_rev < 0 else COLOR_NEUTRAL)
                    st.markdown(
                        f"<div style='margin:8px 0;'>"
                        f"<div style='font-size:11px;color:rgba(128,128,128,0.65);"
                        f"margin-bottom:3px;'>EPS Revisions (last 30d)</div>"
                        f"<div style='font-size:16px;font-weight:700;color:{net_c};'>"
                        f"{up_str} up / {dn_str} down</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("EPS revision counts: N/A")

                # Short interest
                if si_pct is not None:
                    si_color  = (COLOR_NEGATIVE if si_pct > 0.15 else
                                 COLOR_WARNING  if si_pct > 0.07 else COLOR_POSITIVE)
                    si_ratio  = si.get("short_ratio")
                    ratio_str = f" · Days-to-cover: {si_ratio:.1f}" if si_ratio else ""
                    st.markdown(
                        f"<div style='margin:8px 0;'>"
                        f"<div style='font-size:11px;color:rgba(128,128,128,0.65);"
                        f"margin-bottom:3px;'>Short Interest</div>"
                        f"<div style='font-size:16px;font-weight:700;color:{si_color};'>"
                        f"{si_pct*100:.1f}% of float{ratio_str}</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Short interest: N/A")

                st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

                # Last EPS surprise
                eh = results.get("earnings_hist", pd.DataFrame())
                last_surprise = None
                if not eh.empty:
                    sc = next((c for c in eh.columns if "surprise" in c.lower()), None)
                    if sc:
                        valid = eh.dropna(subset=[sc])
                        if not valid.empty:
                            last_surprise = valid.iloc[-1][sc]
                if last_surprise is not None:
                    sc_c = COLOR_POSITIVE if last_surprise >= 0 else COLOR_NEGATIVE
                    st.markdown(
                        f"<div style='margin:8px 0;'>"
                        f"<div style='font-size:11px;color:rgba(128,128,128,0.65);"
                        f"margin-bottom:3px;'>Last EPS Surprise</div>"
                        f"<div style='font-size:16px;font-weight:700;color:{sc_c};'>"
                        f"{last_surprise:+.1f}% "
                        f"{'beat' if last_surprise >= 0 else 'miss'}</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Last EPS surprise: N/A")

                # Most recent analyst rating change
                ud = ad.get("upgrades_downgrades") or []
                if ud:
                    try:
                        first      = ud[0] if isinstance(ud, list) else {}
                        grade_date = str(first.get("GradeDate", "")).split("T")[0]
                        firm       = first.get("Firm", "")
                        to_grade   = first.get("ToGrade", "")
                        action     = first.get("Action", "")
                        if firm or to_grade:
                            st.markdown(
                                f"<div style='margin:8px 0;'>"
                                f"<div style='font-size:11px;color:rgba(128,128,128,0.65);"
                                f"margin-bottom:3px;'>Latest Rating Change</div>"
                                f"<div style='font-size:13px;font-weight:600;'>"
                                f"{firm} → {to_grade}</div>"
                                f"<div style='font-size:11px;color:rgba(128,128,128,0.65);'>"
                                f"{action} · {grade_date}</div></div>",
                                unsafe_allow_html=True,
                            )
                    except Exception:
                        pass

        # ── BLOCK 4 — EXTERNAL SIGNALS ────────────────────────────────────
        st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

        # ── SECTION D — External Signals (News + Peers) ───────────────────
        ns_tab1    = results.get("news_signals")
        ps_tab1    = results.get("peer_signals", [])
        _peer_adj1 = analysis.signal_contributions.get("peer_signals", 0.0)
        _has_ext   = ns_tab1 is not None or ps_tab1

        if _has_ext:
            st.markdown("<div class='rr-section-header'>External Signals</div>",
                        unsafe_allow_html=True)

            # ── News dims ────────────────────────────────────────────────────
            if ns_tab1 is not None:
                st.markdown(
                    "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.45);margin:4px 0 8px;'>"
                    "GDELT News Sentiment</div>",
                    unsafe_allow_html=True,
                )
                n_dims = [
                    ("Overall Sentiment",  ns_tab1.news_sentiment_score),
                    ("Supply Chain",       ns_tab1.supply_chain_signal),
                    ("Competitive",        ns_tab1.competitive_signal),
                    ("Macro",              ns_tab1.macro_signal),
                    ("Regulatory",         ns_tab1.regulatory_signal),
                ]
                ncols = st.columns(len(n_dims))
                for ncol, (label, val) in zip(ncols, n_dims):
                    bc      = (COLOR_POSITIVE if val > 0.05 else
                               COLOR_NEGATIVE if val < -0.05 else COLOR_NEUTRAL)
                    pct_bar = max(0, min(100, int((val + 1) / 2 * 100)))
                    ncol.markdown(
                        f"<div style='text-align:center;'>"
                        f"<div style='font-size:10px;color:rgba(128,128,128,0.65);"
                        f"margin-bottom:4px;'>{label}</div>"
                        f"<div style='background:rgba(128,128,128,0.2);border-radius:4px;"
                        f"height:6px;margin-bottom:4px;'>"
                        f"<div style='width:{pct_bar}%;background:{bc};height:100%;"
                        f"border-radius:4px;'></div></div>"
                        f"<div style='font-size:13px;font-weight:700;color:{bc};"
                        f"font-family:monospace;'>{val:+.2f}</div></div>",
                        unsafe_allow_html=True,
                    )
                if ns_tab1.underweighted_factors:
                    st.markdown(
                        "<div style='margin-top:8px;font-size:12px;font-weight:600;'>"
                        "Analysts may be underweighting:</div>",
                        unsafe_allow_html=True,
                    )
                    for uf in ns_tab1.underweighted_factors:
                        st.markdown(
                            f"<div style='font-size:12px;padding:2px 0 2px 10px;'>"
                            f"• {_safe_md(uf)}</div>",
                            unsafe_allow_html=True,
                        )

            # ── Peer signals ─────────────────────────────────────────────────
            if ps_tab1 and _peer_adj1 != 0.0:
                _pa_col   = COLOR_POSITIVE if _peer_adj1 > 0 else COLOR_NEGATIVE
                _pa_arrow = "↑" if _peer_adj1 > 0 else "↓"
                st.markdown(
                    f"<div style='font-size:11px;margin-top:10px;'>"
                    f"<span style='font-size:10px;text-transform:uppercase;"
                    f"letter-spacing:0.07em;color:rgba(128,128,128,0.45);'>"
                    f"Peer read-through: </span>"
                    f"<span style='color:{_pa_col};font-weight:700;'>"
                    f"{_pa_arrow} {_peer_adj1:+.3f}</span>"
                    f"<span style='color:rgba(128,128,128,0.5);'> applied to score"
                    f" · full detail in Peers & Context tab</span></div>",
                    unsafe_allow_html=True,
                )

        # ── BLOCK 5 — VALIDATION ──────────────────────────────────────────
        st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

        # ── SECTION E — Signal Validation (Falsifiability) ────────────────
        _fals = results.get("falsifiability")
        if _fals and _fals.n_fired > 0:
            st.markdown(
                "<div class='rr-section-header'>Signal Validation</div>",
                unsafe_allow_html=True,
            )

            # Plain-English explanation
            st.markdown(
                "<div style='font-size:12px;line-height:1.65;color:rgba(160,160,160,0.85);"
                "margin-bottom:12px;'>"
                "Each quarter the signal fired (predicted up or down), we check whether the "
                "eventual EPS result matched that direction. "
                "The bar for validity: <b>does firing beat random chance?</b> "
                "Formally, P(correct | signal fired) > P(correct at random)."
                "</div>",
                unsafe_allow_html=True,
            )

            # ── Per-quarter visual verdict row ──────────────────────────────
            _verdict_chips = []
            for _v in _fals.validations:
                if _v.signal_direction == "neutral" or _v.surprise_pct is None:
                    continue  # only show fired quarters with data
                _sig_arrow = "↑" if _v.signal_direction == "upward" else "↓"
                _surp_str  = f"{_v.surprise_pct:+.1f}%"
                if _v.match is True:
                    _chip_bg, _chip_border, _chip_icon = (
                        f"rgba(46,204,113,0.12)", f"rgba(46,204,113,0.4)", "✓"
                    )
                    _outcome_label = "beat" if _v.surprise_direction == "beat" else "miss"
                else:
                    _chip_bg, _chip_border, _chip_icon = (
                        f"rgba(255,77,109,0.12)", f"rgba(255,77,109,0.4)", "✗"
                    )
                    _outcome_label = _v.surprise_direction or "inline"
                _verdict_chips.append(
                    f"<div style='display:inline-flex;flex-direction:column;"
                    f"align-items:center;background:{_chip_bg};"
                    f"border:1px solid {_chip_border};border-radius:8px;"
                    f"padding:8px 12px;margin:4px;min-width:90px;'>"
                    f"<div style='font-size:10px;color:rgba(128,128,128,0.6);"
                    f"margin-bottom:3px;'>{_v.quarter}</div>"
                    f"<div style='font-size:16px;font-weight:800;"
                    f"color:{'#2ecc71' if _v.match else '#ff4d6d'};'>"
                    f"{_sig_arrow} {_chip_icon}</div>"
                    f"<div style='font-size:10px;color:rgba(160,160,160,0.7);"
                    f"margin-top:2px;'>EPS {_surp_str}</div>"
                    f"<div style='font-size:9px;color:rgba(128,128,128,0.5);'>"
                    f"{_outcome_label}</div>"
                    f"</div>"
                )
            if _verdict_chips:
                st.markdown(
                    "<div style='display:flex;flex-wrap:wrap;gap:4px;margin-bottom:14px;'>"
                    + "".join(_verdict_chips) + "</div>",
                    unsafe_allow_html=True,
                )

            # ── Aggregate stats (4 cols) ─────────────────────────────────────
            _fals_c1, _fals_c2, _fals_c3, _fals_c4 = st.columns(4, gap="small")
            _da_pct  = int(_fals.directional_accuracy * 100)
            _br_pct  = int(_fals.base_rate * 100)
            _da_col  = COLOR_POSITIVE if _fals.passes_test else COLOR_NEGATIVE
            _lift_pct  = int(_fals.lift * 100)
            _lift_col  = COLOR_POSITIVE if _fals.lift > 0 else COLOR_NEGATIVE
            _lift_sign = "+" if _lift_pct >= 0 else ""
            _test_col  = COLOR_POSITIVE if _fals.passes_test else COLOR_NEGATIVE
            _p_str = f"p={_fals.p_value:.2f}" if _fals.p_value < 0.99 else "p≈1.0"

            _fals_c1.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.65);"
                f"margin-bottom:4px;'>When fired, correct</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_da_col};'>{_da_pct}%</div>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.5);'>"
                f"{_fals.n_fired_correct}/{_fals.n_fired} quarters</div></div>",
                unsafe_allow_html=True,
            )
            _fals_c2.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.65);"
                f"margin-bottom:4px;'>Random baseline</div>"
                f"<div style='font-size:22px;font-weight:800;"
                f"color:rgba(180,180,180,0.7);'>{_br_pct}%</div>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.5);'>"
                f"across {_fals.n_total} qtrs</div></div>",
                unsafe_allow_html=True,
            )
            _fals_c3.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.65);"
                f"margin-bottom:4px;'>Edge over random</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_lift_col};'>"
                f"{_lift_sign}{_lift_pct}%</div>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.5);'>"
                f"lift = accuracy / baseline − 1</div></div>",
                unsafe_allow_html=True,
            )
            _fals_c4.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.65);"
                f"margin-bottom:4px;'>Verdict</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_test_col};'>"
                f"{'✓ Beats' if _fals.passes_test else '✗ Fails'}</div>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.5);'>"
                f"z={_fals.z_score:.2f} · {_p_str}</div></div>",
                unsafe_allow_html=True,
            )

            # ── Limitations ─────────────────────────────────────────────────
            _caveat_parts = [
                "Proxy: EPS surprise direction ≈ revision direction, not identical.",
                "Lead time (what matters most) is unmeasurable without timestamped I/B/E/S.",
            ]
            if _fals.sample_size_warning:
                _caveat_parts.insert(0,
                    f"Small sample (n={_fals.n_fired}) — indicative only, not statistically conclusive."
                )
            st.caption("  ·  ".join(_caveat_parts))

        # ── Signal Trend ───────────────────────────────────────────────────────
        if analysis.quarters:
            quarters_chron = list(reversed(analysis.quarters))
            q_labels  = [qs.quarter for qs in quarters_chron]
            probs     = [qs.revision_probability for qs in quarters_chron]
            scores    = [qs.weighted_score for qs in quarters_chron]
            pt_colors = [
                COLOR_POSITIVE if qs.direction == "upward" else
                COLOR_NEGATIVE if qs.direction == "downward" else
                COLOR_NEUTRAL
                for qs in quarters_chron
            ]
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=q_labels, y=probs,
                mode="lines+markers",
                name="Revision Probability",
                line=dict(color=dir_c, width=2),
                marker=dict(color=pt_colors, size=9,
                            line=dict(color="white", width=1.5)),
                yaxis="y1",
            ))
            fig_trend.add_trace(go.Scatter(
                x=q_labels, y=scores,
                mode="lines+markers",
                name="Weighted Score",
                line=dict(color=COLOR_NEUTRAL, width=1.5, dash="dot"),
                marker=dict(color=COLOR_NEUTRAL, size=6),
                yaxis="y2",
                opacity=0.7,
            ))
            fig_trend.add_hline(y=0.5, line_dash="dash",
                                line_color="rgba(128,128,128,0.3)",
                                annotation_text="neutral", yref="y1")
            fig_trend.update_layout(
                title=f"Signal Trend — {len(analysis.quarters)} Quarters",
                xaxis=dict(title=None),
                yaxis=dict(title="Revision Probability", range=[0, 1],
                           tickformat=".0%", side="left"),
                yaxis2=dict(title="Weighted Score", range=[-1.5, 1.5],
                            side="right", overlaying="y",
                            showgrid=False, zeroline=True,
                            zerolinecolor="rgba(128,128,128,0.2)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                height=260,
                margin=dict(t=50, b=30, l=0, r=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")
            if analysis.trend_direction == "deteriorating":
                st.warning("Signal has deteriorated across consecutive quarters")
            elif analysis.trend_direction == "improving":
                st.info("Signal has improved across consecutive quarters")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — The Evidence
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        # ── Global section — signal scoring across all quarters ───────────────
        st.markdown("<div class='rr-section-header' style='font-size:15px;margin-bottom:8px;'>Score Breakdown</div>", unsafe_allow_html=True)
        if analysis.signal_contributions:
            st.plotly_chart(
                waterfall_chart(analysis.signal_contributions),
                use_container_width=True, theme="streamlit",
            )
            st.caption("Aggregated across all quarters — most recent quarter carries highest weight. Older quarters provide trend signal only.")
        st.markdown("<div class='rr-section-header'>Quarter Scores</div>",
                        unsafe_allow_html=True)
        score_rows = []
        for i_qs, qs in enumerate(analysis.quarters):
            sym = ("↑" if qs.direction == "upward" else
                   "↓" if qs.direction == "downward" else "→")
            guid_val = qs.normalized.get("guidance_quantification", 0.0) if qs.normalized else 0.0
            has_prior = (i_qs + 1 < len(analysis.quarters)
                         and analysis.quarters[i_qs + 1].transcript is not None)
            guid_label = f"Δ {guid_val:+.2f}" if has_prior else f"{guid_val:+.2f} (abs)"
            prior_avail_icon = "✓ actual" if (
                i_qs + 1 < len(analysis.quarters)
                and analysis.quarters[i_qs + 1].transcript is not None
            ) else "est."
            status = "▶ New Signal" if i_qs == 0 else "Historical"
            score_rows.append({
                "Quarter":          qs.quarter,
                "Status":           status,
                "Direction":        f"{sym} {qs.direction.title()}",
                "Probability":      f"{int(qs.revision_probability*100)}%",
                "Score":            f"{qs.weighted_score:+.3f}",
                "Guid. Δ":          guid_label,
                "Prior Qtr Tone":   prior_avail_icon,
                "Magnitude":        qs.magnitude.title(),
                "Confidence":       f"{int(qs.confidence*100)}%",
            })
        st.dataframe(pd.DataFrame(score_rows),
                     use_container_width=True, hide_index=True)
        st.caption("▶ New Signal = most recent filing, may not yet be priced into analyst models. Historical rows provide trend context only.")

        # ── Peer adjustment summary ────────────────────────────────────────
        _pa_disp = analysis.signal_contributions.get("peer_signals", 0.0)
        if _pa_disp != 0.0:
            _pa_peers = results.get("peer_signals", [])
            _pa_dir   = "upward" if _pa_disp > 0 else "downward"
            st.markdown(
                f"<div style='font-size:12px;margin-top:6px;"
                f"color:rgba(200,200,200,0.7);'>"
                f"Peer signal adjustment applied to final score: "
                f"<b style='color:{'#2ecc71' if _pa_disp > 0 else '#ff4d6d'};'>"
                f"{_pa_disp:+.3f}</b> ({_pa_dir}, {len(_pa_peers)} peer{'s' if len(_pa_peers)!=1 else ''})"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Global: News panel (not per-quarter) ──────────────────────────────
        ns_ev     = results.get("news_signals")
        news_tl   = results.get("news_timeline",  pd.DataFrame())
        news_arts = results.get("news_articles",  pd.DataFrame())
        st.markdown("<div class='rr-section-header' style='font-size:15px;margin-bottom:8px;margin-top:16px;'>News & External Signals</div>", unsafe_allow_html=True)
        if news_tl.empty and ns_ev is None and news_arts.empty:
            st.info("No news data available. "
                    "Enable News in the sidebar to run news analysis.")
        else:
            # ── Row 1: signal bars + key themes side by side ──────────────
            if ns_ev is not None:
                ns_left, ns_right = st.columns([0.35, 0.65], gap="medium")
                with ns_left:
                    for label, val in [
                        ("Overall Sentiment",  ns_ev.news_sentiment_score),
                        ("Supply Chain",       ns_ev.supply_chain_signal),
                        ("Competitive",        ns_ev.competitive_signal),
                        ("Macro",              ns_ev.macro_signal),
                        ("Regulatory",         ns_ev.regulatory_signal),
                    ]:
                        render_bar(label, val)
                with ns_right:
                    if ns_ev.key_themes:
                        st.markdown(
                            "<div class='rr-section-header' style='margin-top:0;margin-bottom:6px;'>"
                            "Key Themes</div>", unsafe_allow_html=True)
                        for theme in ns_ev.key_themes:
                            st.markdown(
                                f"<div style='font-size:13px;line-height:1.5;"
                                f"word-break:break-word;margin-bottom:3px;'>"
                                f"• {_safe_md(theme)}</div>",
                                unsafe_allow_html=True,
                            )
                    if ns_ev.news_summary:
                        st.caption(_safe_md(ns_ev.news_summary))
            # ── Row 2: sentiment timeline chart ───────────────────────────
            # Click a bar to filter articles to that date; scroll is suppressed above
            _art_dates = None
            if not news_tl.empty:
                _chart_ev = st.plotly_chart(
                    news_sentiment_timeline(news_tl),
                    use_container_width=True, theme="streamlit",
                    on_select="rerun", key="news_timeline_chart",
                )
                # Extract clicked date from selection
                try:
                    _pts = _chart_ev.selection.get("points", [])
                    if _pts:
                        import pandas as _pd2
                        _art_dates = _pd2.to_datetime(_pts[0]["x"]).date()
                except Exception:
                    pass

            if not news_arts.empty:
                from datetime import timedelta as _td

                # ── Period selector ───────────────────────────────────────
                # Build quarter-aligned options from the articles themselves,
                # plus rolling windows for when no bar is clicked.
                _period_col, _per_col2 = st.columns([0.55, 0.45], gap="small")
                with _period_col:
                    # Quarter buckets derived from article dates
                    if "seendate" in news_arts.columns:
                        _art_qtrs = (
                            news_arts["seendate"]
                            .dropna()
                            .dt.to_period("Q")
                            .drop_duplicates()
                            .sort_values(ascending=False)
                        )
                        _qtr_labels = ["All periods"] + [str(q) for q in _art_qtrs]
                    else:
                        _qtr_labels = ["All periods"]
                    _sel_period = st.selectbox(
                        "Period",
                        options=_qtr_labels,
                        index=0,
                        key="news_period_filter",
                        label_visibility="collapsed",
                    )

                # Apply period filter first, then optional single-day click filter
                if _sel_period != "All periods" and "seendate" in news_arts.columns:
                    try:
                        _pq = pd.Period(_sel_period, freq="Q")
                        _p_start = _pq.start_time.date()
                        _p_end   = _pq.end_time.date()
                        _arts_period = news_arts[
                            news_arts["seendate"].dt.date.between(_p_start, _p_end)
                        ]
                    except Exception:
                        _arts_period = news_arts
                else:
                    _arts_period = news_arts

                # Single-day click narrows within the period
                if _art_dates is not None and "seendate" in news_arts.columns:
                    _window = {_art_dates - _td(days=1), _art_dates, _art_dates + _td(days=1)}
                    _arts_filtered = _arts_period[_arts_period["seendate"].dt.date.isin(_window)]
                else:
                    _arts_filtered = _arts_period

                _header = (
                    f"Articles for {_art_dates.strftime('%b %d, %Y')} · {len(_arts_filtered)} found"
                    if _art_dates
                    else f"{_sel_period} · {len(_arts_filtered)} articles"
                )
                st.markdown(
                    f"<div class='rr-section-header' style='margin-top:12px;'>"
                    f"{_header}</div>", unsafe_allow_html=True)
                _show_all_news = st.checkbox(
                    f"Show all {len(_arts_filtered)} articles",
                    key="news_show_all"
                )
                _arts_to_show = _arts_filtered if _show_all_news else _arts_filtered.head(5)
                rows_html = []
                for _, row in _arts_to_show.iterrows():
                    title    = str(row.get("title", "—"))
                    url      = str(row.get("url", ""))
                    domain   = str(row.get("domain", ""))
                    try:
                        _dt  = pd.to_datetime(row.get("seendate"))
                        date = _dt.strftime("%b %d") if not pd.isnull(_dt) else "—"
                    except Exception:
                        date = "—"
                    tone_val = 0.0
                    try:
                        tone_val = float(row.get("tone", 0.0))
                    except Exception:
                        pass
                    tc = (COLOR_POSITIVE if tone_val > 0.5 else
                          COLOR_NEGATIVE if tone_val < -0.5 else "inherit")
                    tc_html = (
                        f'<a href="{url}" target="_blank" rel="noopener" '
                        f'style="color:inherit;text-decoration:none;'
                        f'font-weight:500;">{title}</a>'
                    ) if url and url != "nan" else title
                    rows_html.append(
                        f"<tr><td style='padding:6px 10px;font-size:12px;'>{tc_html}</td>"
                        f"<td style='padding:6px 10px;font-size:11px;opacity:0.6;"
                        f"white-space:nowrap;'>{date}</td>"
                        f"<td style='padding:6px 10px;font-size:11px;opacity:0.6;'>"
                        f"{domain}</td>"
                        f"<td style='padding:6px 10px;font-size:11px;color:{tc};"
                        f"text-align:right;font-family:monospace;'>"
                        f"{tone_val:+.1f}</td></tr>"
                    )
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;'>"
                    f"<thead><tr style='border-bottom:1px solid rgba(128,128,128,0.2);'>"
                    f"<th style='text-align:left;font-size:11px;padding:4px 10px;"
                    f"opacity:0.6;font-weight:600;'>Title</th>"
                    f"<th style='text-align:left;font-size:11px;padding:4px 10px;"
                    f"opacity:0.6;font-weight:600;'>Date</th>"
                    f"<th style='text-align:left;font-size:11px;padding:4px 10px;"
                    f"opacity:0.6;font-weight:600;'>Source</th>"
                    f"<th style='text-align:right;font-size:11px;padding:4px 10px;"
                    f"opacity:0.6;font-weight:600;'>Tone</th>"
                    f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr class='rr-divider'>", unsafe_allow_html=True)

        # ── Per-quarter drill-down ─────────────────────────────────────────────
        qd_list = results.get("quarter_data", [])
        if not qd_list:
            st.info("No filing data available.")
        else:
            quarters_list = [qd["quarter"] for qd in qd_list]
            sel_q  = st.selectbox("Quarter", quarters_list, index=0,
                                  key="evidence_quarter")
            sel_qd = next((qd for qd in qd_list
                           if qd["quarter"] == sel_q), None)
            # QuarterSignals object for the selected quarter (holds .normalized deltas etc.)
            sel_qs = next((qs for qs in analysis.quarters if qs.quarter == sel_q), None)

            # Warn if selected quarter's MD&A data is stale (by absolute age, not just position)
            _sel_q_idx = quarters_list.index(sel_q) if sel_q in quarters_list else 0
            from src.scoring.falsifiability import _quarter_end_date as _qed2
            _sel_qend = _qed2(sel_q)
            import datetime as _dt_mod2
            _sel_age  = (_dt_mod2.date.today() - _sel_qend).days if _sel_qend else 0
            if _sel_age > 75:
                _sel_months = round(_sel_age / 30)
                st.caption(f"⏳ ~{_sel_months}mo old — likely priced in. Use for trend context only.")

            t = sel_qd.get("transcript") if sel_qd else None
            m = sel_qd.get("mda")        if sel_qd else None
            filing     = (sel_qd.get("filing") or {}) if sel_qd else {}
            filing_url = filing.get("source_url", "")
            ten_q_url  = next(
                (f.get("source_url", "") for f in results.get("ten_q_filings", [])
                 if f.get("quarter") == sel_q), ""
            ) if sel_qd else ""
            src_links = []
            if filing_url:
                src_links.append(f"[8-K]({filing_url})")
            if ten_q_url:
                src_links.append(f"[10-Q]({ten_q_url})")
            if src_links:
                st.caption("EDGAR: " + " · ".join(src_links))

            # ── EXPANDER 1: Language Diff (PRIMARY SIGNAL) ────────────────────
            # Prefer transcript diff (always available) over 10-Q diff
            _tdiffs     = results.get("transcript_diffs") or {}
            _mda_diffs  = results.get("mda_diff_meta") or {}
            mda_meta = _tdiffs.get(sel_q) or _mda_diffs.get(sel_q) or None
            _diff_src   = "transcript" if sel_q in _tdiffs else ("10-Q" if sel_q in _mda_diffs else None)
            _diff_label = (
                f"Language Diff  ·  {sel_q} vs {mda_meta['prior_quarter']}"
                if mda_meta else "Language Diff"
            )
            with st.expander(_diff_label, expanded=True):
                if not mda_meta:
                    st.caption("No prior quarter to compare — oldest filing in window.")
                else:
                    unchanged_pct = round(mda_meta["unchanged_ratio"] * 100, 1)
                    _src_label = "transcript" if _diff_src == "transcript" else "10-Q"
                    st.caption(
                        f"{100-unchanged_pct:.1f}% changed vs {mda_meta['prior_quarter']} ({_src_label})"
                    )
                    new_lang     = mda_meta.get("new_language", "").strip()
                    removed_lang = mda_meta.get("removed_language", "").strip()
                    if not new_lang and not removed_lang:
                        st.info("No significant language changes detected.")
                    else:
                        diff_cols = st.columns(2, gap="medium")
                        with diff_cols[0]:
                            st.markdown(
                                "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                                "letter-spacing:0.08em;color:#2ecc71;margin-bottom:8px;'>"
                                "+ Added this quarter</div>", unsafe_allow_html=True)
                            if new_lang:
                                _new_lines = [l.strip() for l in new_lang.split("\n") if l.strip()]
                                for s in _new_lines[:3]:
                                    st.markdown(
                                        f"<div style='padding:5px 0 5px 12px;border-left:3px solid "
                                        f"#2ecc7177;font-size:12px;line-height:1.65;"
                                        f"margin-bottom:4px;'>{_safe_md(s)}</div>",
                                        unsafe_allow_html=True)
                                if len(_new_lines) > 3:
                                    if st.checkbox(f"Show all {len(_new_lines)} added lines",
                                                   key=f"diff_new_{sel_q}"):
                                        for s in _new_lines[3:]:
                                            st.markdown(
                                                f"<div style='padding:5px 0 5px 12px;"
                                                f"border-left:3px solid #2ecc7177;"
                                                f"font-size:12px;line-height:1.65;"
                                                f"margin-bottom:4px;'>{_safe_md(s)}</div>",
                                                unsafe_allow_html=True)
                            else:
                                st.caption("No new language.")
                        with diff_cols[1]:
                            st.markdown(
                                "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                                "letter-spacing:0.08em;color:#ff4d6d;margin-bottom:2px;'>"
                                "⬛ Disappeared Language</div>"
                                "<div style='font-size:10px;color:rgba(128,128,128,0.5);"
                                "margin-bottom:8px;font-style:italic;'>"
                                "Absent vs. prior quarter — no interpretation needed</div>",
                                unsafe_allow_html=True)
                            if removed_lang:
                                _rem_lines = [l.strip() for l in removed_lang.split("\n") if l.strip()]
                                st.caption(f"{len(_rem_lines)} sentence{'s' if len(_rem_lines) != 1 else ''} no longer appear in this filing.")
                                for s in _rem_lines[:3]:
                                    st.markdown(
                                        f"<div style='padding:5px 0 5px 12px;border-left:3px solid "
                                        f"#ff4d6d55;font-size:12px;line-height:1.65;margin-bottom:4px;"
                                        f"background:rgba(255,77,109,0.04);border-radius:0 4px 4px 0;"
                                        f"color:rgba(180,100,100,0.8);'>{_safe_md(s)}</div>",
                                        unsafe_allow_html=True)
                                if len(_rem_lines) > 3:
                                    if st.checkbox(f"Show all {len(_rem_lines)} disappeared lines",
                                                   key=f"diff_rem_{sel_q}"):
                                        for s in _rem_lines[3:]:
                                            st.markdown(
                                                f"<div style='padding:5px 0 5px 12px;"
                                                f"border-left:3px solid #ff4d6d55;"
                                                f"font-size:12px;line-height:1.65;margin-bottom:4px;"
                                                f"background:rgba(255,77,109,0.04);"
                                                f"border-radius:0 4px 4px 0;"
                                                f"color:rgba(180,100,100,0.8);'>"
                                                f"{_safe_md(s)}</div>",
                                                unsafe_allow_html=True)
                            else:
                                st.caption("Nothing disappeared — language is consistent with prior quarter.")

            # ── EXPANDER 2: Overall Score & Evidence ──────────────────────────
            with st.expander("Overall Score & Evidence", expanded=False):
                if not t and not m:
                    st.info("No filing data available for this quarter.")
                else:
                    # ── Row 1: Signal Scores (left) | Tone Assessment (right) ──
                    ev_left, ev_right = st.columns(2, gap="medium")
                    with ev_left:
                        if t:
                            st.markdown(
                                "<div class='rr-section-header'>Signal Scores</div>",
                                unsafe_allow_html=True)

                            def _sub(label: str) -> None:
                                st.markdown(
                                    f"<div style='font-size:10px;font-weight:700;"
                                    f"text-transform:uppercase;letter-spacing:0.08em;"
                                    f"color:rgba(128,128,128,0.45);margin:10px 0 2px;'>"
                                    f"{label}</div>", unsafe_allow_html=True)

                            # ── 1. Tone Delta ─────────────────────────────────
                            _sub("Tone Delta  ·  0.30")
                            render_bar("Overall Tone",     t.overall_tone,               -1.0, 1.0)
                            # Determine whether actual prior-quarter transcript was available
                            _qd_idx = next((i for i, qd in enumerate(qd_list) if qd["quarter"] == sel_q), None)
                            _prior_avail = (
                                _qd_idx is not None
                                and _qd_idx + 1 < len(qd_list)
                                and bool(qd_list[_qd_idx + 1].get("transcript"))
                            )
                            _tvpq_label = "vs Prior Qtr (actual)" if _prior_avail else "vs Prior Qtr (est.)"
                            render_bar(_tvpq_label,         t.tone_vs_prior_quarter,      -1.0, 1.0)
                            render_bar("Forward Guidance", t.forward_guidance_strength,  -1.0, 1.0)
                            render_bar("Demand Language",  t.demand_language_tone,       -1.0, 1.0)
                            render_bar("Margin Language",  t.margin_language_tone,       -1.0, 1.0)
                            if m:
                                render_bar("MD&A Fwd Tone", m.forward_looking_tone,      -1.0, 1.0)

                            # ── 2. Guidance Quantification ────────────────────
                            _sub("Guidance Quantification  ·  0.25")
                            # Use delta from scorer (current rate − prior rate) when available
                            _guid_delta = (
                                sel_qs.normalized.get("guidance_quantification")
                                if sel_qs and sel_qs.normalized
                                else None
                            )
                            if _guid_delta is not None and _prior_avail:
                                render_bar("Δ Specificity (vs prior qtr)", _guid_delta, -1.0, 1.0)
                                st.caption(
                                    f"Raw rate this qtr: {t.guidance_quantification_rate:.0%}  "
                                    f"→  prior qtr: {qd_list[_qd_idx + 1]['transcript'].guidance_quantification_rate:.0%}"
                                    if _qd_idx is not None and qd_list[_qd_idx + 1].get("transcript") else ""
                                )
                            else:
                                render_bar("Specificity (absolute)",
                                           (t.guidance_quantification_rate - 0.5) * 2, -1.0, 1.0)
                            if m:
                                render_bar("MD&A Specificity",
                                           (m.guidance_specificity - 0.5) * 2,     -1.0, 1.0)

                            # ── 3. Risk Escalation ────────────────────────────
                            _sub("Risk Escalation  ·  0.20")
                            render_bar("Risk Language",  -t.risk_language_escalation,   -1.0, 1.0)
                            render_bar("Hedging",        -t.hedging_intensity,          -1.0, 0.0)
                            if m:
                                render_bar("MD&A Risk",  -m.risk_escalation,            -1.0, 1.0)
                                render_bar("Cost Press.", -m.cost_pressure_signals,     -1.0, 1.0)

                            # ── 4. Q&A Deflection ─────────────────────────────
                            _sub("Q&A Deflection  ·  0.15")
                            render_bar("Q&A Openness",   -t.qa_deflection_score,        -1.0, 0.0)

                        else:
                            st.info("No 8-K data for this quarter.")
                    with ev_right:
                        if t and t.management_tone_summary:
                            st.markdown(
                                "<div class='rr-section-header'>Tone Assessment</div>",
                                unsafe_allow_html=True)
                            st.info(_safe_md(t.management_tone_summary))

                    # ── Row 2: Phrases full-width in 3 columns ─────────────────
                    def _phrase_section(col, label, phrases, color):
                        if not phrases:
                            return
                        with col:
                            st.markdown(
                                f"<div class='rr-section-header' style='margin-top:14px;'>"
                                f"{label}</div>", unsafe_allow_html=True)
                            for ph in phrases[:2]:
                                render_quote_card(ph, color)
                            if len(phrases) > 2:
                                _key = f"show_{label}_{sel_q}".replace(" ", "_")
                                if st.checkbox(f"Show all {len(phrases)}", key=_key):
                                    for ph in phrases[2:]:
                                        render_quote_card(ph, color)

                    if t:
                        pc1, pc2, pc3 = st.columns(3, gap="medium")
                        _phrase_section(pc1, "Bullish Phrases", t.key_bullish_phrases, "positive")
                        _phrase_section(pc2, "Hedging Phrases", t.key_hedging_phrases, "negative")
                        _phrase_section(pc3, "Risk Phrases",    t.key_risk_phrases,    "negative")

                    # ── Row 3: MD&A quotes (2-col) + summary full-width ────────
                    if m and m.key_quotes:
                        st.markdown(
                            "<div class='rr-section-header' style='margin-top:14px;'>"
                            "Key MD&A Quotes</div>", unsafe_allow_html=True)
                        mq_cols = st.columns(2, gap="medium")
                        for i, qp in enumerate(m.key_quotes[:4]):
                            with mq_cols[i % 2]:
                                render_quote_card(qp, "blue")

                    if m and m.mda_summary:
                        st.markdown(
                            "<div class='rr-section-header' style='margin-top:10px;'>"
                            "MD&A Summary</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='rr-card' style='font-size:13px;line-height:1.7;'>"
                            f"{_safe_md(m.mda_summary)}</div>",
                            unsafe_allow_html=True)

                # ── What's New This Quarter (merged) ──────────────────────────
                if t or m:
                    st.markdown("<hr class='rr-divider' style='margin:14px 0;'>",
                                unsafe_allow_html=True)
                    st.markdown(
                        "<div class='rr-section-header' style='font-size:13px;'>"
                        "What's New This Quarter</div>", unsafe_allow_html=True)
                    # ── forecast pills ────────────────────────────────────────
                    if t and t.forecast_items_most_at_risk:
                        st.markdown(
                            "<div class='rr-section-header'>Forecast Items at Risk</div>",
                            unsafe_allow_html=True)
                        chips = "".join(
                            f"<span class='rr-pill' style='border-color:{COLOR_WARNING}44;"
                            f"color:{COLOR_WARNING};'>{item}</span>"
                            for item in t.forecast_items_most_at_risk
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                    # ── catalysts + risks side by side ────────────────────────
                    _has_cat  = bool(m and m.positive_catalysts_cited)
                    _has_risk = bool(m and m.new_risk_factors)
                    if _has_cat and _has_risk:
                        wn_c2, wn_c3 = st.columns(2, gap="medium")
                        _cat_col, _risk_col = wn_c2, wn_c3
                    else:
                        _cat_col = _risk_col = st.container()

                    with _cat_col:
                        if _has_cat:
                            st.markdown(
                                "<div class='rr-section-header' style='margin-top:12px;'>"
                                "Positive Catalysts</div>", unsafe_allow_html=True)
                            for pc in m.positive_catalysts_cited:
                                st.markdown(
                                    f"<div style='font-size:12px;padding:2px 0 2px 10px;"
                                    f"border-left:2px solid {COLOR_POSITIVE}44;"
                                    f"margin-bottom:4px;'>• {_safe_md(pc)}</div>",
                                    unsafe_allow_html=True)
                    with _risk_col:
                        if _has_risk:
                            st.markdown(
                                "<div class='rr-section-header' style='margin-top:12px;'>"
                                "New Risk Factors</div>", unsafe_allow_html=True)
                            for rf in m.new_risk_factors:
                                st.markdown(
                                    f"<div style='font-size:12px;color:{COLOR_NEGATIVE};"
                                    f"padding:2px 0 2px 10px;border-left:2px solid "
                                    f"{COLOR_NEGATIVE}44;margin-bottom:4px;'>"
                                    f"• {_safe_md(rf)}</div>",
                                    unsafe_allow_html=True)
                    # ── rationale ─────────────────────────────────────────────
                    if t and t.analyst_revision_rationale:
                        st.markdown(
                            "<div class='rr-section-header' style='margin-top:16px;'>"
                            "Analyst Revision Rationale</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='rr-card' style='border-color:{dir_c}44;"
                            f"font-size:13px;line-height:1.7;'>"
                            f"{_safe_md(t.analyst_revision_rationale)}</div>",
                            unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Peers & Context
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        # One-line role badge
        st.markdown(
            "<div style='font-size:11px;color:rgba(128,128,128,0.6);"
            "margin-bottom:14px;'>"
            "<b style='color:rgba(200,200,200,0.9);'>Peer signals</b> feed the final score "
            "(capped ±0.20) &nbsp;·&nbsp; "
            "<b style='color:rgba(200,200,200,0.9);'>Market context</b> panels are display-only</div>",
            unsafe_allow_html=True,
        )

        # ── SUPPLY-CHAIN PEERS ────────────────────────────────────────────────
        st.markdown(
            "<div class='rr-section-header' style='font-size:13px;letter-spacing:0.07em;"
            "margin-bottom:10px;'>SUPPLY-CHAIN PEERS</div>",
            unsafe_allow_html=True,
        )

        peer_source3      = results.get("peer_source",
                            {"type": "none", "confidence": "low",
                             "warning": None, "ticker_count": 0})
        peer_tone_scores3 = results.get("peer_tone_scores", [])
        peer_status3      = results.get("peer_status", {})
        peer_sigs3        = results.get("peer_signals", [])
        _peer_score_adj3  = analysis.signal_contributions.get("peer_signals", 0.0)

        if peer_source3.get("warning") and peer_source3.get("confidence") in ("low", "medium"):
            st.warning(peer_source3["warning"])

        if not peer_sigs3:
            st.info("No peer signals available. Enable Supply-chain Peers in the sidebar.")
        else:
            # ── Read-through headline ──────────────────────────────────────────
            _n_pos  = sum(1 for p in peer_sigs3 if p.get("signal_direction") == "positive")
            _n_neg  = sum(1 for p in peer_sigs3 if p.get("signal_direction") == "negative")
            _n_neu  = len(peer_sigs3) - _n_pos - _n_neg
            _net_dir = ("positive" if _n_pos > _n_neg
                        else "negative" if _n_neg > _n_pos else "mixed")
            _net_color = (COLOR_POSITIVE if _net_dir == "positive"
                          else COLOR_NEGATIVE if _net_dir == "negative"
                          else COLOR_NEUTRAL)
            _net_arrow = "▲" if _net_dir == "positive" else "▼" if _net_dir == "negative" else "↔"
            _adj_color = (COLOR_POSITIVE if _peer_score_adj3 > 0
                          else COLOR_NEGATIVE if _peer_score_adj3 < 0 else COLOR_NEUTRAL)
            _verdict_parts = []
            if _n_pos: _verdict_parts.append(f"<span style='color:{COLOR_POSITIVE};'>{_n_pos} positive</span>")
            if _n_neg: _verdict_parts.append(f"<span style='color:{COLOR_NEGATIVE};'>{_n_neg} negative</span>")
            if _n_neu: _verdict_parts.append(f"<span style='color:{COLOR_NEUTRAL};'>{_n_neu} neutral</span>")
            src_type3 = peer_source3.get("type", "none")
            src_badge = ("curated map" if src_type3 == "curated"
                         else "AI-discovered" if src_type3 == "llm_dynamic" else "")
            st.markdown(
                f"<div class='rr-card' style='border-color:{_net_color}44;"
                f"display:flex;align-items:center;justify-content:space-between;"
                f"flex-wrap:wrap;gap:12px;'>"
                f"<div>"
                f"<div style='font-size:11px;text-transform:uppercase;letter-spacing:0.08em;"
                f"color:rgba(128,128,128,0.7);margin-bottom:4px;'>Supply-chain read-through"
                f"{(' · <span style=\"font-style:italic;\">' + src_badge + '</span>') if src_badge else ''}"
                f"</div>"
                f"<div style='font-size:20px;font-weight:800;color:{_net_color};'>"
                f"{_net_arrow} {_net_dir.upper()} &nbsp;"
                f"<span style='font-size:13px;font-weight:400;color:rgba(128,128,128,0.8);'>"
                f"({' · '.join(_verdict_parts)} across {len(peer_sigs3)} peers)</span></div>"
                f"</div>"
                f"<div style='text-align:right;'>"
                f"<div style='font-size:11px;color:rgba(128,128,128,0.6);'>Score adjustment</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_adj_color};'>"
                f"{_peer_score_adj3:+.3f}</div>"
                f"<div style='font-size:10px;color:rgba(128,128,128,0.5);'>capped ±0.20</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # ── Divergences from peer average ──────────────────────────────────
            target_signals3 = {}
            if analysis.quarters and analysis.quarters[0].transcript:
                t0_3 = analysis.quarters[0].transcript
                target_signals3 = {
                    "overall_tone":              t0_3.overall_tone,
                    "hedging_intensity":         t0_3.hedging_intensity,
                    "forward_guidance_strength": t0_3.forward_guidance_strength,
                    "risk_language_escalation":  t0_3.risk_language_escalation,
                    "demand_language_tone":      t0_3.demand_language_tone,
                    "margin_language_tone":      t0_3.margin_language_tone,
                }

            _dim_labels = {
                "overall_tone":              "Overall Tone",
                "hedging_intensity":         "Hedging Intensity",
                "forward_guidance_strength": "Guidance Strength",
                "risk_language_escalation":  "Risk Escalation",
                "demand_language_tone":      "Demand Language",
                "margin_language_tone":      "Margin Language",
            }
            if target_signals3 and peer_tone_scores3:
                _divergences = []
                for dim, label in _dim_labels.items():
                    tgt_val = target_signals3.get(dim)
                    peer_vals = [p[dim] for p in peer_tone_scores3 if dim in p]
                    if tgt_val is None or not peer_vals:
                        continue
                    peer_avg = sum(peer_vals) / len(peer_vals)
                    delta = tgt_val - peer_avg
                    _divergences.append((abs(delta), delta, label, tgt_val, peer_avg))
                _divergences.sort(reverse=True)
                top_divs = _divergences[:2]

                if top_divs:
                    st.markdown(
                        "<div style='font-size:11px;text-transform:uppercase;"
                        "letter-spacing:0.08em;color:rgba(128,128,128,0.6);"
                        "margin:14px 0 6px;'>Largest divergences from peer average"
                        " — company-specific, not sector-wide</div>",
                        unsafe_allow_html=True,
                    )
                    div_cols = st.columns(len(top_divs), gap="medium")
                    for col, (abs_d, delta, label, tgt, avg) in zip(div_cols, top_divs):
                        d_color = COLOR_POSITIVE if delta > 0 else COLOR_NEGATIVE
                        d_arrow = "▲" if delta > 0 else "▼"
                        d_interp = "above" if delta > 0 else "below"
                        col.markdown(
                            f"<div class='rr-card' style='border-color:{d_color}44;'>"
                            f"<div style='font-size:11px;color:rgba(128,128,128,0.65);"
                            f"margin-bottom:6px;'>{label}</div>"
                            f"<div style='font-size:22px;font-weight:800;color:{d_color};'>"
                            f"{d_arrow} {abs_d:.2f}</div>"
                            f"<div style='font-size:12px;margin-top:4px;'>"
                            f"{ticker} is <b>{abs_d:.2f} pts {d_interp}</b> peer avg "
                            f"({tgt:+.2f} vs {avg:+.2f})</div>"
                            f"<div style='font-size:11px;color:rgba(128,128,128,0.55);"
                            f"margin-top:4px;font-style:italic;'>Sector-wide shifts affect "
                            f"all peers equally — this is idiosyncratic.</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Peer insight cards ─────────────────────────────────────────────
            peers_with_insight = [p for p in peer_sigs3 if p.get("key_insight")]
            if peers_with_insight:
                st.markdown(
                    "<div style='font-size:11px;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:rgba(128,128,128,0.6);"
                    "margin:14px 0 6px;'>Read-through from peers</div>",
                    unsafe_allow_html=True,
                )
                for i in range(0, len(peers_with_insight), 2):
                    card_cols = st.columns(2, gap="medium")
                    for col, ps in zip(card_cols, peers_with_insight[i:i+2]):
                        dir_ps   = ps.get("signal_direction", "neutral")
                        dir_ps_c = (COLOR_POSITIVE if dir_ps == "positive"
                                    else COLOR_NEGATIVE if dir_ps == "negative"
                                    else COLOR_NEUTRAL)
                        dir_ps_arrow = ("▲" if dir_ps == "positive"
                                        else "▼" if dir_ps == "negative" else "—")
                        col.markdown(
                            f"<div class='rr-card' style='border-color:{dir_ps_c}33;'>"
                            f"<div style='display:flex;align-items:center;gap:8px;"
                            f"margin-bottom:8px;'>"
                            f"<span style='font-family:monospace;font-weight:800;"
                            f"font-size:14px;'>{ps.get('ticker','')}</span>"
                            f"<span style='font-size:11px;color:rgba(128,128,128,0.6);'>"
                            f"{ps.get('name','')}</span>"
                            f"<span style='margin-left:auto;color:{dir_ps_c};"
                            f"font-weight:800;font-size:16px;'>{dir_ps_arrow}</span></div>"
                            f"<div style='font-size:12px;line-height:1.65;'>"
                            f"{_safe_md(ps.get('key_insight',''))}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Tone comparison chart ──────────────────────────────────────────
            if peer_tone_scores3 and target_signals3:
                _src3 = peer_source3.get("type", "none")
                chart_ticker3 = (
                    f"{ticker} (AI-discovered — verify)"
                    if _src3 == "llm_dynamic" else ticker
                )
                st.plotly_chart(
                    sector_tone_comparison(chart_ticker3, target_signals3,
                                           peer_tone_scores3),
                    use_container_width=True, theme="streamlit",
                )
                st.caption(
                    f"Tone vs. avg of {len(peer_tone_scores3)} supply-chain peers. "
                    "Large deviations are company-specific — sector-wide shifts move all peers equally."
                )

            # ── Peer fetch status ──────────────────────────────────────────────
            if peer_status3:
                _no_note = {"no_filing": "no 8-K", "no_cik": "not on EDGAR",
                            "no_text": "text failed"}
                status_parts = []
                for t_peer, s in peer_status3.items():
                    if s == "ok":
                        status_parts.append(
                            f"<span style='color:{COLOR_POSITIVE};'>✓</span> "
                            f"<span style='font-family:monospace;font-size:11px;'>{t_peer}</span>"
                        )
                    else:
                        note = _no_note.get(s, s.replace("error: ", ""))[:30]
                        status_parts.append(
                            f"<span style='color:rgba(128,128,128,0.4);'>○</span> "
                            f"<span style='font-family:monospace;font-size:11px;"
                            f"color:rgba(128,128,128,0.5);'>{t_peer} ({note})</span>"
                        )
                st.markdown(
                    "<div style='font-size:11px;margin-top:8px;display:flex;"
                    "flex-wrap:wrap;gap:14px;'>" +
                    "".join(status_parts) + "</div>",
                    unsafe_allow_html=True,
                )

        # ── MARKET CONTEXT (display only) ─────────────────────────────────────
        st.markdown("<hr class='rr-divider' style='margin:20px 0 14px;'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='rr-section-header' style='font-size:13px;letter-spacing:0.07em;"
            "margin-bottom:10px;'>MARKET CONTEXT <span style='font-size:10px;"
            "font-weight:400;color:rgba(128,128,128,0.55);letter-spacing:normal;"
            "text-transform:none;'>(display only — not fed into model)</span></div>",
            unsafe_allow_html=True,
        )

        # Earnings surprise + Short interest side by side
        eh_col, si_col = st.columns([0.6, 0.4], gap="medium")
        with eh_col:
            st.plotly_chart(
                analyst_estimate_chart(results.get("earnings_hist", pd.DataFrame())),
                use_container_width=True, theme="streamlit",
            )
        with si_col:
            st.markdown("<div class='rr-section-header'>Short Interest</div>",
                        unsafe_allow_html=True)
            si3        = results.get("short_interest", {}) or {}
            si3_pct    = si3.get("short_percent_of_float")
            si3_ratio  = si3.get("short_ratio")
            si3_shares = si3.get("shares_short")
            if si3_pct is not None:
                si3_color = (COLOR_NEGATIVE if si3_pct > 0.15 else
                             COLOR_WARNING  if si3_pct > 0.07 else COLOR_POSITIVE)
                st.markdown(
                    f"<div class='rr-card' style='border-color:{si3_color}44;'>"
                    f"<div style='font-size:32px;font-weight:700;"
                    f"color:{si3_color};'>{si3_pct*100:.1f}%</div>"
                    f"<div style='font-size:12px;color:rgba(128,128,128,0.75);"
                    f"margin-top:4px;'>of float is short</div>"
                    + (f"<div style='font-size:12px;margin-top:10px;'>"
                       f"Days-to-cover: <b>{si3_ratio:.1f}</b></div>"
                       if si3_ratio else "")
                    + (f"<div style='font-size:12px;'>"
                       f"Shares short: <b>{si3_shares:,}</b></div>"
                       if si3_shares else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
                si3_interp = (
                    "High — structurally bearish positioning."
                    if si3_pct > 0.10 else
                    "Low — market not positioned for a negative surprise."
                    if si3_pct < 0.03 else
                    "Moderate — no strong structural positioning."
                )
                st.caption(si3_interp)
            else:
                st.info("Short interest data not available.")

        # Price history (below context row)
        price_df = results.get("price_history", pd.DataFrame())
        if not price_df.empty:
            signal_annotations = []
            for qd_item, qs_item in zip(results.get("quarter_data", []),
                                        analysis.quarters):
                fd = (qd_item.get("filing") or {}).get("filingDate", "")
                if fd:
                    signal_annotations.append({
                        "date":      fd,
                        "quarter":   qs_item.quarter,
                        "direction": qs_item.direction,
                        "score":     qs_item.weighted_score,
                    })
            st.plotly_chart(
                price_history_chart(price_df, signal_annotations),
                use_container_width=True, theme="streamlit",
            )

# ── About / documentation page ────────────────────────────────────────────────
def render_about():
    st.markdown("""
## What is Check?

**Check** is an AI-powered analyst revision intelligence engine. It detects
leading indicators of sell-side earnings-estimate revisions — *before* those revisions
are published — by reading the same unstructured documents analysts read, but applying
systematic NLP signal extraction to surface patterns that humans routinely underweight
or miss at critical inflection points.

The core insight: **published analyst forecasts frequently lag reality.** Analysts are
institutionally slow to revise at turning points. This tool is designed to identify those
turning points earlier, using only qualitative language signals — not the headline numbers
that everyone already has.

---

## The Core Constraint: No Circular Logic

> *Stock price movements and headline EPS beats/misses are explicitly excluded from all
> signal features. They are coincident indicators — they confirm what already happened,
> not what analysts will do next. Using them would be circular.*

Every Claude prompt contains this hard constraint. The scoring model has no access to
price data, EPS vs. consensus figures, or market cap changes. The only inputs are
**unstructured text** and **qualitative language patterns**.

---

## Data Sources

### 1. SEC EDGAR — 8-K Earnings Releases (Item 2.02)
The SEC requires companies to file an 8-K within 4 business days of any material event.
Item 2.02 ("Results of Operations and Financial Condition") specifically covers earnings
releases. We filter EDGAR's submission API to **Item 2.02 only** — excluding executive
compensation agreements, amendments, and other non-earnings 8-Ks that would add noise.

**What we extract from the text:**
- Management tone and quarter-over-quarter tone shift
- Hedging language density (words like "approximately", "subject to", "if conditions allow")
- Guidance quantification rate — how many forward statements include specific numbers
- Q&A deflection score — how often management redirects analyst questions
- Forward-looking statement bullishness or caution
- Key quoted phrases (hedging, bullish, risk language) — exact verbatim extracts
- Analyst revision rationale — Claude's assessment of what the sell side is missing

**Why this matters:** The press release text is qualitative data that analysts read but
don't systematically score. A management team that quietly increases hedging language
quarter-over-quarter is signaling something the consensus model hasn't priced in.

### 2. SEC EDGAR — 10-Q Management Discussion & Analysis
Every quarterly 10-Q filing contains an "Item 2 — MD&A" section where management is
required to discuss financial condition, results, and forward-looking outlook in narrative
form. We extract this section directly from the primary document, stripping tables,
exhibits, and boilerplate certifications (EX-31, EX-32).

**What we extract:**
- Forward-looking language tone and specificity
- Risk factor escalation — new or elevated risks vs. boilerplate
- Cost pressure and liquidity language signals
- Positive catalysts management explicitly cites
- Key verbatim quotes linked back to the original EDGAR filing

**Why this matters:** MD&A is the most underread section of a 10-Q. It contains
management's own words about what is and isn't working — language that often shifts
subtly before an earnings revision cycle begins.

### 3. GDELT — Global News Intelligence
GDELT (Global Database of Events, Language, and Tone) is a free, real-time database
maintained by Google Jigsaw that monitors news across 100+ languages from global
print, broadcast, and web sources. We use two endpoints:

- **Tone timeline**: Daily average sentiment score for articles mentioning the company
- **Article search**: Recent headlines, publication dates, and source domains

**What we extract:**
- News sentiment trajectory over 180 days
- Supply-chain and regulatory event signals
- Macro tailwinds/headwinds specific to this company
- Competitive landscape changes mentioned in external coverage
- Article-level tone scores, with titles linked to original sources

**Why this matters:** External news often captures demand signals, regulatory shifts,
and competitive pressures *before* they appear in company filings. A semiconductor
company's news tone around a major customer's capex announcements is a legitimate
leading indicator.

### 4. Supply-Chain Peer Filings
For each company, we identify upstream/downstream supply-chain peers and fetch their
most recent 8-K earnings release. Claude then analyzes the peer text specifically for
read-through signals relevant to the target company.

**Example:** TSMC's commentary on AI accelerator demand directly informs NVIDIA's
revenue visibility. Taiwan Semiconductor is not guessing — they are booking orders.

**What we extract per peer:**
- Relevance score (how much does this peer's business affect the target?)
- Signal direction (positive / negative / neutral for target)
- Key verbatim quotes from the peer filing
- Raw tone dimensions (overall tone, hedging, guidance, risk language) — used to build
  the sector peer tone comparison in the Market Context tab

**Peer discovery is dynamic:** For well-known companies, curated supply-chain relationships
are used (e.g. AAPL → TSM, QCOM, AVGO). For all other companies, peers are inferred from
the company's sector and industry via yfinance, using a sector-level supplier/customer map.
A hard cap per sector (2–4 peers) prevents over-fetching for industries with many natural peers.

---

## Control Group — Market Context Tab

The **Market Context tab** is a control group: it shows what the market already knows and
has priced in. None of this data is fed into the revision model — using it would be circular.

| Control Metric | Source | Why It's a Control, Not a Signal |
|----------------|--------|----------------------------------|
| **Price History** | yfinance | Prices are coincident — they reflect consensus expectations already incorporated |
| **Earnings Surprise History** | yfinance | Past misses show whether analysts have been wrong before, but don't predict future revisions |
| **Short Interest** | yfinance | Shows whether the market is already positioned for a negative outcome |
| **Analyst EPS Revisions** | yfinance | We're predicting what analysts will do — we can't use what analysts currently think as an input |
| **Upgrades / Downgrades** | yfinance | Analyst opinion at a point in time — already public, already priced |
| **Sector Peer Tone** | SEC 8-K (same source) | Used for *comparison* only — the gap between company tone and sector average is a signal, but the peer average itself is context |

**How to use the control group:** If the model flags a strong downward signal AND short
interest is high AND the last two quarters showed EPS misses, that is multi-source confirmation.
If the model is bearish but short interest is near zero and the stock is near all-time highs,
that's a more contrarian setup with higher uncertainty.

### 5. yfinance — Analyst Consensus (Control Group)
We pull analyst EPS estimates, revision history, and upgrade/downgrade records from
yfinance purely for **display context**. This data is shown in the Market Context tab
so you can see the current consensus — but it is **never used as a predictive signal**.
Using it would be circular: we are trying to predict what analysts will do, so we cannot
use what analysts currently think as an input.

---

## How Signals Are Scored

Claude extracts **5 consolidated signal dimensions** from each filing, each normalized to a
comparable scale. These are then aggregated with empirically-tuned weights:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Management Tone Delta** | 30% | Quarter-over-quarter shift in language positivity, incorporating overall tone, explicit QoQ comparison (against actual prior-quarter filing), forward guidance strength, and operational language (demand, margins, capex, hiring). The *change* matters more than the absolute level. |
| **Guidance Quantification Δ** | 25% | *Delta* of guidance specificity rate vs. prior quarter — not the absolute rate. A shift from "8–10% revenue growth" to "monitoring conditions" is the signal, not whether guidance exists at all. |
| **Risk Language Escalation** | 20% | Net new risk language vs. prior quarter baseline, unified with hedging intensity and cost pressure signals. New or escalating risk language in MD&A is a leading indicator. |
| **Q&A Deflection** | 15% | How often management redirects, defers, or avoids direct answers during Q&A. High deflection often precedes negative revisions. |
| **News Sentiment** | 10% | GDELT tone trajectory across the analysis window — supply chain, competitive, macro, and regulatory signals. |

In addition, a **Peer Signal Adjustment** (capped at ±0.20) is applied from supply-chain peer filings — allowing upstream signals to shift the final score before the target company has filed anything.

The weighted score is passed through a **sigmoid function** (steepness=2) to convert it
into a probability between 0 and 1, then mapped to direction (upward / downward / neutral)
and magnitude:

| Magnitude | Implied Revision Range |
|-----------|----------------------|
| Negligible | < 1% |
| Small | 1 – 3% |
| Medium | 3 – 7% |
| Large | > 7% |

---

## Forecast Focus

The **Forecast Focus** selector in the sidebar tells Claude which analyst estimate
dimensions to emphasize in the narrative synthesis. It does not change the raw signal
scores — it shapes what the AI thesis, primary drivers, and watch items highlight.

| Focus Item | Best For |
|------------|---------|
| EPS | All companies — the primary analyst estimate |
| Revenue | All companies — top-line growth visibility |
| Gross Margin | SaaS, semiconductors, manufacturers |
| Operating Margin | Industrials, retailers, mature tech |
| Free Cash Flow | Capital-intensive, asset-heavy businesses |
| Guidance (Next Quarter) | Companies with volatile forward visibility |
| Operating Leverage | High fixed-cost businesses — airlines, fabs, utilities |
| R&D / Innovation Velocity | Pharma, biotech, deep tech, semiconductors |
| Demand & Pricing Power | Consumer, enterprise software, industrials |
| International / Segment Growth | Multinationals, geographically diversified |
| Customer Metrics (ARR / NRR) | SaaS, subscription, marketplace businesses |
| Capital Allocation | Banks, conglomerates, mature industrials |

---

## Analysis Pipeline (Step by Step)

1. **Resolve ticker → CIK** via EDGAR's company_tickers.json map
2. **Fetch 8-K filings** — filter submission history to Item 2.02 only, download full press release text
3. **Fetch 10-Q filings** — download primary document, heuristically extract MD&A section
4. **Diff MD&A** — sentence-level Jaccard diff between consecutive 10-Q filings; classify each sentence as new / retained / removed. The removed language is the primary signal.
5. **Fetch GDELT news** — chunked 90-day fetches, tone timeline + article list across the full analysis window
6. **Fetch peer filings** — supply-chain peer discovery (curated map → Claude-powered dynamic), sector-aware cap; each peer's most recent 8-K is extracted for read-through signals
7. **Fetch control group data** — price history (2y), short interest, earnings surprise history (yfinance — display only, never a model input)
8. **Claude extraction** — structured JSON signal extraction per filing with prior-quarter context for tone comparison (cached by hash of text + prior text)
9. **Peer signal extraction** — read-through risk signals per peer; relevance-weighted adjustment fed into final score (capped ±0.20)
10. **Score quarters** — 5-dimension weighted aggregation + peer adjustment → sigmoid → probability per quarter; guidance quantification computed as QoQ delta
11. **Compute trend** — direction of change across lookback window (improving / deteriorating / stable)
12. **Falsifiability check** — align signal directions to realized EPS surprises; compute P(correct | signal fired) vs base rate
13. **Synthesize narrative** — Claude generates investment thesis, primary drivers, bull/bear cases, watch items
14. **Render dashboard** — outputs across 3 analytical tabs (The Case / The Evidence / Peers & Context)

---

## Caching & Data Freshness

| Layer | TTL | Storage | Auto-deleted? |
|-------|-----|---------|---------------|
| HTTP responses (EDGAR, GDELT) | 48 hours | `.cache/*.json` | Yes — purged on app startup |
| Claude LLM extractions | 48 hours | `.cache/*.json` | Yes — purged on app startup |
| Saved reports (named snapshots) | Never | `saved_reports/*.pkl` | No — managed manually via sidebar |

**Force Refresh Cache** in the sidebar immediately clears all cached data for the current
ticker and LLM extractions, forcing a fully live fetch on the next run.

---

## Lookback Window

The slider sets how many quarters of history to analyze (1–20 quarters, up to 5 years).
Longer windows improve trend detection accuracy but increase API calls and runtime linearly.
EDGAR's submission API typically covers 3–5 years of filing history for large-cap companies.
For smaller or newer companies, fewer quarters may be available.

---

## What This Tool Is Not

- It does not predict stock prices
- It does not use insider information
- It does not scrape proprietary data sources
- It does not guarantee that analyst revisions will occur — it estimates probability
- The Analyst Context tab data (yfinance) is for orientation only, not signal

---
""", unsafe_allow_html=False)


# ── Landing page ──────────────────────────────────────────────────────────────
def render_landing():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='text-align:center;padding:60px 20px 32px 20px;'>
            <div style='font-size:56px;margin-bottom:14px;'></div>
            <h2 style='margin:0 0 8px 0;font-size:28px;font-weight:800;letter-spacing:-0.02em;'>
                {APP_NAME}
            </h2>
            <p style='color:rgba(128,128,128,0.75);font-size:15px;max-width:600px;margin:0 auto 12px;line-height:1.7;'>
                Detect leading indicators of analyst earnings-estimate revisions
                using only <em>qualitative</em> signals from SEC filings and news —
                zero stock-price data used.
            </p>
            <p style='color:rgba(128,128,128,0.55);font-size:13px;margin:0 auto 32px;'>
                Pick a ticker in the sidebar → select data sources → click <strong>▶ Run Analysis</strong>
            </p>
            <div style='display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:10px;'>
                {''.join(f'<span class="rr-pill">{t}</span>' for t in ["AAPL","MSFT","NVDA","TSLA","META","AMZN","JPM","LLY"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── How It Works ──────────────────────────────────────────────────────────
    st.markdown("<div style='max-width:900px;margin:0 auto;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px;'>
            <div class='rr-card'>
                <div style='font-size:24px;margin-bottom:8px;'>📥</div>
                <div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Step 1 — Fetch</div>
                <div style='font-size:12px;color:rgba(128,128,128,0.75);line-height:1.6;'>
                    Pulls SEC EDGAR 8-K earnings releases, 10-Q MD&amp;A sections,
                    and GDELT global news — all free APIs, no key required.
                </div>
            </div>
            <div class='rr-card'>
                <div style='font-size:24px;margin-bottom:8px;'>🤖</div>
                <div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Step 2 — Extract</div>
                <div style='font-size:12px;color:rgba(128,128,128,0.75);line-height:1.6;'>
                    Claude reads each filing and scores qualitative signal dimensions:
                    tone, hedging, guidance specificity, Q&amp;A deflection, risk language, and more.
                </div>
            </div>
            <div class='rr-card'>
                <div style='font-size:24px;margin-bottom:8px;'>📈</div>
                <div style='font-weight:700;font-size:13px;margin-bottom:6px;'>Step 3 — Score</div>
                <div style='font-size:12px;color:rgba(128,128,128,0.75);line-height:1.6;'>
                    Weighted signals are aggregated across quarters into a revision probability,
                    direction, magnitude, and a full AI investment narrative.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📖  Full Documentation — Data Sources, Methodology & Signal Weights",
                     expanded=False):
        render_about()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_header()
    params = render_sidebar()

    if "results" not in st.session_state:
        st.session_state.results = None

    if params["run"] and params["ticker"]:
        try:
            st.session_state.results = run_analysis(params)
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.results = None
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.session_state.results = None
    elif params["run"] and not params["ticker"]:
        st.warning("Please select a ticker before running analysis.")

    if st.session_state.results:
        render_dashboard(st.session_state.results)
    else:
        render_landing()


if __name__ == "__main__":
    main()
