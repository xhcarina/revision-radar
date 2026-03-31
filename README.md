# Check
**AI-Powered Analyst Forecast Revision Intelligence**

Check detects leading indicators of sell-side earnings-estimate revisions *before* they are published — by extracting structured signals from SEC filings, earnings call transcripts, and global news using Claude AI. Zero stock-price data is used at any stage of analysis. The only inputs are unstructured text and qualitative language patterns.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/xhcarina/revision-radar.git
cd revision-radar

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# 5. Run
streamlit run app.py
```

App runs at `http://localhost:8501`.

---

## Project Structure

```
revision_radar/
├── app.py                        # Streamlit UI (3-tab dashboard)
├── config.py                     # Signal weights, thresholds, API config
├── event_log.py                  # Lightweight run-event logger
├── requirements.txt
├── data/
│   ├── peer_universe.json        # Curated supply-chain peer map
│   └── peer_cache/               # Claude-generated peer cache (7-day TTL)
└── src/
    ├── data/
    │   ├── sec_client.py         # SEC EDGAR filing fetcher + MD&A extractor
    │   ├── company_client.py     # yfinance metadata, analyst estimates, peer discovery
    │   └── news_client.py        # GDELT news fetcher
    ├── extraction/
    │   ├── llm_extractor.py      # Claude API calls, Pydantic signal models
    │   └── prompts.py            # All prompt templates
    ├── scoring/
    │   ├── scorer.py             # Weighted signal aggregation → revision probability
    │   └── falsifiability.py     # Historical signal accuracy vs. EPS surprise direction
    └── visualization/
        └── charts.py             # Plotly chart builders
```

---

## Data Sources

| Source | What it provides | Library |
|--------|-----------------|---------|
| **SEC EDGAR** | 8-K filings (earnings releases), 10-Q MD&A sections | `requests` + `edgartools` |
| **yfinance** | Company metadata, analyst EPS/revenue consensus, EPS revision counts, earnings history, short interest | `yfinance` |
| **GDELT** | News headlines, article snippets, daily sentiment timeline | `gdeltdoc` |
| **Claude API** | Signal extraction, MD&A delta analysis, news synthesis, supply-chain peer discovery, final narrative | `anthropic` |

> Stock price history is fetched for *display context only* (price chart in the Peers & Context tab). It is never used as a signal input.

---

## Methodology

### 1. Document Ingestion

- **8-K / Earnings releases**: Fetched from SEC EDGAR — ticker → CIK → submission metadata → filing index → largest text exhibit. Item 2.02 only (earnings results). Two-layer cache (raw HTML + cleaned text).
- **10-Q MD&A**: Extracted from quarterly 10-Q filings using heuristic section splitting on standard headers.
- **MD&A Diff**: Consecutive quarters are compared sentence-by-sentence using Jaccard token similarity (threshold 0.5, implemented natively — no ML library required). Sentences are classified as *new*, *retained*, or *removed*. Only the delta is passed to Claude — not the unchanged text. The disappearance of previously confident language is treated as a signal in its own right.

### 2. Signal Extraction (Claude)

Each document type has a dedicated prompt template (`src/extraction/prompts.py`). All prompts explicitly forbid use of stock price, market cap, or headline EPS vs. consensus — preventing circular reasoning. Output is JSON with a fixed schema validated by Pydantic.

| Prompt | Document | Output |
|--------|----------|--------|
| `TRANSCRIPT_ANALYSIS_PROMPT` | 8-K / earnings call | `TranscriptSignals` — tone, hedging, guidance specificity, Q&A deflection, risk language, demand/margin tone |
| `MDA_ANALYSIS_PROMPT` | Full 10-Q MD&A | `MDASignals` — forward tone, risk escalation, cost pressure, liquidity, specificity |
| `MDA_DELTA_PROMPT` | MD&A diff (new + removed language only) | `MDASignals` — scored on language change, not full text |
| `NEWS_SYNTHESIS_PROMPT` | GDELT headlines | `NewsSignals` — sentiment, supply chain, competitive, macro, regulatory |
| `PEER_SIGNAL_PROMPT` | Supply-chain peer filing | Relevance score, signal direction, key insight |
| `NARRATIVE_SYNTHESIS_PROMPT` | All signals combined | `NarrativeOutput` — thesis, bull/bear/watch, primary drivers, revision horizon |

### 3. Signal Aggregation & Scoring

Signals are normalized to `[-1, +1]` then combined as a weighted sum, passed through a sigmoid (steepness = 2) to yield revision probability. The three named core signals — **① Tone Shift** (QoQ), **② Guidance Specificity** (Δ), **③ Disappeared Language** (line count) — are displayed prominently as the primary evidence chips.

| Signal | Weight |
|--------|--------|
| Tone delta (current vs. prior quarter) | 0.22 |
| Hedging intensity | 0.18 |
| Guidance quantification rate | 0.18 |
| Q&A deflection score | 0.14 |
| Risk language escalation | 0.12 |
| News sentiment | 0.10 |
| Forward guidance strength | 0.06 |
| Peer signal adjustment | ±0.20 cap |

**Magnitude thresholds:**

| Label | Score range | EPS revision estimate |
|-------|------------|----------------------|
| Negligible | < 0.15 | < 1% |
| Small | 0.15 – 0.30 | 1 – 3% |
| Medium | 0.30 – 0.50 | 3 – 7% |
| Large | > 0.50 | > 7% |

### 4. Supply-Chain Peer Analysis

Peers are sourced via a two-tier strategy:

1. **Curated map** (`data/peer_universe.json`) — manually validated upstream suppliers for ~50 tickers. Confidence: **high**.
2. **Claude-powered dynamic discovery** — for tickers not in the curated map, Claude Haiku identifies direct upstream suppliers. Each candidate is validated against SEC EDGAR (`get_cik()`). Results cached per-ticker for 7 days. Confidence: **medium**.

Each peer's most recent 8-K is analyzed for read-through signals relevant to the target. The peer section in the dashboard shows: net read-through direction, largest divergences from peer-average tone dimensions (company-specific vs. sector-wide), and per-peer insight cards.

### 5. Falsifiability

Each quarter where a signal fired is checked against the subsequent EPS surprise direction from yfinance. The dashboard shows: directional accuracy, base rate, lift over random, and a per-quarter verdict chip (signal direction vs. actual outcome). This is an approximation — EPS surprise direction proxies revision direction, and lead time cannot be precisely measured without timestamped I/B/E/S data.

---

## Dashboard Tabs

### Tab 1 — The Case
The complete analytical argument on one screen.
- Direction signal (↑/↓/→), magnitude badge, trend badge, revision horizon
- AI revision thesis and "what analysts are missing"
- Bull case / bear case / watch items / key risk / primary drivers
- Three core signal chips: Tone Shift (QoQ), Guidance Specificity (Δ), Disappeared Language
- Management posture + forward signals + risk profile bar clusters
- The Gap: filing signals vs. analyst consensus positioning
- Signal context: filing age, analyst coverage count, large-cap warning
- News signal summary (GDELT)
- Signal validation (historical accuracy vs. base rate)
- Signal trend chart (probability + weighted score across quarters)

### Tab 2 — The Evidence
Quarter-by-quarter deep dive.
- Score breakdown waterfall + quarter scores table (New Signal vs. Historical)
- Per-quarter language diff: new language (green) and disappeared language (dimmed red, no strikethrough)
- Score breakdown expander: signal bars, tone assessment, phrases, MD&A quotes
- "What's New This Quarter" (merged): forecast items at risk, new risk factors, catalysts, revision rationale
- News sentiment bars, GDELT article table with tone scores and source links

### Tab 3 — Peers & Context
- **Supply-chain peers** (top): net read-through headline, largest tone divergences from peer average, per-peer insight cards, sector tone comparison chart, fetch status
- **Market context** (bottom, display only): earnings surprise history, short interest, 2-year price history with filing markers

---

## Limitations

- **Qualitative signals only.** Cannot observe order books, channel checks, or private management commentary.
- **EPS revision proxy.** Revision counts from yfinance give direction (up/down) but not magnitude or exact timing.
- **GDELT coverage.** Broad English-language coverage; may miss niche trade publications or non-English sources.
- **Peer validation.** Claude-generated peers are EDGAR-validated but supply-chain relationships change faster than the 7-day cache TTL.
- **Small sample.** Falsifiability stats are indicative for most tickers (3–8 quarters of history). Not statistically conclusive.
- **Large-cap signal decay.** MD&A for large-cap companies is heavily legal-reviewed; language signals are less idiosyncratic. The dashboard surfaces a warning when market cap > $10B or analyst count > 20.
- **No financial advice.** Research intelligence only. Not a trading signal or investment recommendation.

---

## Python Packages

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard UI |
| `anthropic` | Claude API (extraction, peer discovery, narrative) |
| `yfinance` | Analyst estimates, earnings history, price data, short interest |
| `requests`, `beautifulsoup4` | SEC EDGAR filing access and HTML parsing |
| `gdeltdoc` | GDELT news headlines |
| `pydantic` | Signal schema validation |
| `plotly` | Interactive charts |
| `pandas`, `numpy` | Data manipulation |
| `tenacity` | Claude API retry logic |
| `python-dotenv` | Environment variable loading |

---
