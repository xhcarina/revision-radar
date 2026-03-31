import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── App Identity ──────────────────────────────────────────────────────────────
APP_NAME = "Check"
APP_SUBTITLE = "AI-Powered Analyst Forecast Revision Intelligence"
APP_VERSION = "1.0.0"

# ── Analysis Defaults ─────────────────────────────────────────────────────────
DEFAULT_QUARTERS = 4
MAX_QUARTERS = 20
DEFAULT_HORIZONS = [30, 60, 90]
FORECAST_ITEMS = [
    "EPS",
    "Revenue",
    "Gross Margin",
    "Operating Margin",
    "Free Cash Flow",
    "Guidance (Next Quarter)",
    "Operating Leverage",
    "R&D / Innovation Velocity",
    "Demand & Pricing Power",
    "International / Segment Growth",
    "Customer Metrics (ARR / NRR)",
    "Capital Allocation",
]

# ── Signal Weights (must sum to 1.0) ─────────────────────────────────────────
# Consolidated 5-signal model — redundant dimensions collapsed:
#   • forward_guidance merged into tone_delta (both measure forward positivity)
#   • hedging_intensity merged into risk_escalation (both are bearish language detectors)
SIGNAL_WEIGHTS = {
    "tone_delta":              0.30,   # sentiment shift + forward confidence
    "guidance_quantification": 0.25,   # specificity collapse (number → vague)
    "risk_escalation":         0.20,   # risk/hedging language change (diff primary)
    "qa_deflection":           0.15,   # behavioral evasion under analyst questioning
    "news_sentiment":          0.10,   # external signal (GDELT + Claude synthesis)
}

# ── Magnitude Thresholds (absolute weighted score) ───────────────────────────
MAGNITUDE_THRESHOLDS = {
    "negligible": 0.15,   # < 1%
    "small":      0.30,   # 1-3%
    "medium":     0.50,   # 3-7%
    "large":      1.00,   # 7%+
}

MAGNITUDE_LABELS = {
    "negligible": ("< 1%",    "⬤"),
    "small":      ("1 – 3%",  "⬤⬤"),
    "medium":     ("3 – 7%",  "⬤⬤⬤"),
    "large":      ("> 7%",    "⬤⬤⬤⬤"),
}

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_HEADERS = {
    "User-Agent": "RevisionRadar contact@revisionradar.ai",
    "Accept-Encoding": "gzip, deflate",
}

# ── Cache ─────────────────────────────────────────────────────────────────────
try:
    CACHE_DIR = Path(".cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    CACHE_DIR = Path("/tmp/revision_radar/.cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 48

# ── Plotly / Color Theme ──────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"

COLOR_POSITIVE  = "#00d4aa"   # Teal – upward revision signal
COLOR_NEGATIVE  = "#ff4d6d"   # Red  – downward revision signal
COLOR_NEUTRAL   = "#7c85f0"   # Indigo
COLOR_WARNING   = "#f5a623"   # Amber
COLOR_BG        = "#0e1117"   # Page background
COLOR_CARD      = "#1a1d2e"   # Card / panel background
COLOR_BORDER    = "#2a2d3e"   # Subtle borders
COLOR_TEXT      = "#e0e0e0"   # Primary text
COLOR_SUBTEXT   = "#8888aa"   # Secondary text

SIGNAL_COLORS = {
    "tone_delta":              "#7c85f0",
    "guidance_quantification": "#00d4aa",
    "risk_escalation":         "#ff4d6d",
    "qa_deflection":           "#f5a623",
    "news_sentiment":          "#5bc4f5",
}

# ── Ticker → Company name map (for autocomplete) ──────────────────────────────
# Covers S&P 500 large-caps + major ADRs/ETFs. Add more as needed.
TICKER_COMPANY_MAP: dict[str, str] = {
    # Mega-cap tech
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corp.",
    "NVDA":  "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG":  "Alphabet Inc. (Class C)",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "TSLA":  "Tesla Inc.",
    "AVGO":  "Broadcom Inc.",
    "ORCL":  "Oracle Corp.",
    "CRM":   "Salesforce Inc.",
    "ADBE":  "Adobe Inc.",
    "AMD":   "Advanced Micro Devices",
    "QCOM":  "Qualcomm Inc.",
    "INTC":  "Intel Corp.",
    "TXN":   "Texas Instruments",
    "MU":    "Micron Technology",
    "AMAT":  "Applied Materials",
    "LRCX":  "Lam Research",
    "KLAC":  "KLA Corp.",
    "MRVL":  "Marvell Technology",
    "ARM":   "Arm Holdings",
    "SMCI":  "Super Micro Computer",
    "CRWD":  "CrowdStrike Holdings",
    "PANW":  "Palo Alto Networks",
    "SNOW":  "Snowflake Inc.",
    "DDOG":  "Datadog Inc.",
    "ZS":    "Zscaler Inc.",
    "NET":   "Cloudflare Inc.",
    "MDB":   "MongoDB Inc.",
    "NOW":   "ServiceNow Inc.",
    "WDAY":  "Workday Inc.",
    "INTU":  "Intuit Inc.",
    "TEAM":  "Atlassian Corp.",
    "HUBS":  "HubSpot Inc.",
    "TWLO":  "Twilio Inc.",
    "OKTA":  "Okta Inc.",
    # Financials
    "JPM":   "JPMorgan Chase & Co.",
    "BAC":   "Bank of America Corp.",
    "WFC":   "Wells Fargo & Co.",
    "GS":    "Goldman Sachs Group",
    "MS":    "Morgan Stanley",
    "BLK":   "BlackRock Inc.",
    "C":     "Citigroup Inc.",
    "SCHW":  "Charles Schwab Corp.",
    "AXP":   "American Express Co.",
    "V":     "Visa Inc.",
    "MA":    "Mastercard Inc.",
    "PYPL":  "PayPal Holdings",
    "COF":   "Capital One Financial",
    "BRK-B": "Berkshire Hathaway (B)",
    # Healthcare
    "JNJ":   "Johnson & Johnson",
    "LLY":   "Eli Lilly and Co.",
    "ABBV":  "AbbVie Inc.",
    "MRK":   "Merck & Co.",
    "PFE":   "Pfizer Inc.",
    "TMO":   "Thermo Fisher Scientific",
    "ABT":   "Abbott Laboratories",
    "DHR":   "Danaher Corp.",
    "UNH":   "UnitedHealth Group",
    "CVS":   "CVS Health Corp.",
    "CI":    "Cigna Group",
    "HUM":   "Humana Inc.",
    "ISRG":  "Intuitive Surgical",
    "GILD":  "Gilead Sciences",
    "BIIB":  "Biogen Inc.",
    "REGN":  "Regeneron Pharma",
    "MRNA":  "Moderna Inc.",
    "AMGN":  "Amgen Inc.",
    "BSX":   "Boston Scientific",
    "MDT":   "Medtronic PLC",
    "ZBH":   "Zimmer Biomet",
    # Consumer
    "AMZN":  "Amazon.com Inc.",
    "WMT":   "Walmart Inc.",
    "COST":  "Costco Wholesale",
    "TGT":   "Target Corp.",
    "HD":    "Home Depot Inc.",
    "LOW":   "Lowe's Companies",
    "MCD":   "McDonald's Corp.",
    "SBUX":  "Starbucks Corp.",
    "NKE":   "Nike Inc.",
    "PG":    "Procter & Gamble",
    "KO":    "Coca-Cola Co.",
    "PEP":   "PepsiCo Inc.",
    "PM":    "Philip Morris Intl.",
    "MO":    "Altria Group",
    "EL":    "Estée Lauder Companies",
    "ULTA":  "Ulta Beauty",
    "LULU":  "Lululemon Athletica",
    "RH":    "RH (Restoration Hardware)",
    "ROST":  "Ross Stores",
    "TJX":   "TJX Companies",
    "DG":    "Dollar General",
    "DLTR":  "Dollar Tree",
    "YUM":   "Yum! Brands",
    "CMG":   "Chipotle Mexican Grill",
    "DPZ":   "Domino's Pizza",
    # Industrials / Energy
    "GE":    "GE Aerospace",
    "HON":   "Honeywell Intl.",
    "CAT":   "Caterpillar Inc.",
    "BA":    "Boeing Co.",
    "RTX":   "RTX Corp.",
    "LMT":   "Lockheed Martin",
    "NOC":   "Northrop Grumman",
    "GD":    "General Dynamics",
    "UPS":   "UPS",
    "FDX":   "FedEx Corp.",
    "DE":    "Deere & Company",
    "EMR":   "Emerson Electric",
    "ETN":   "Eaton Corp.",
    "XOM":   "Exxon Mobil Corp.",
    "CVX":   "Chevron Corp.",
    "COP":   "ConocoPhillips",
    "SLB":   "SLB (Schlumberger)",
    "PSX":   "Phillips 66",
    "VLO":   "Valero Energy",
    "NEE":   "NextEra Energy",
    "DUK":   "Duke Energy",
    "SO":    "Southern Company",
    "D":     "Dominion Energy",
    # Telecom / Media
    "T":     "AT&T Inc.",
    "VZ":    "Verizon Communications",
    "TMUS":  "T-Mobile US",
    "NFLX":  "Netflix Inc.",
    "DIS":   "Walt Disney Co.",
    "CMCSA": "Comcast Corp.",
    "WBD":   "Warner Bros. Discovery",
    "PARA":  "Paramount Global",
    "SPOT":  "Spotify Technology",
    "SNAP":  "Snap Inc.",
    "PINS":  "Pinterest Inc.",
    "RDDT":  "Reddit Inc.",
    # Real estate / Other
    "AMT":   "American Tower Corp.",
    "PLD":   "Prologis Inc.",
    "EQIX":  "Equinix Inc.",
    "SPG":   "Simon Property Group",
    "O":     "Realty Income Corp.",
    "WELL":  "Welltower Inc.",
    # International / ADRs
    "TSM":   "TSMC (Taiwan Semiconductor)",
    "ASML":  "ASML Holding",
    "SAP":   "SAP SE",
    "BABA":  "Alibaba Group",
    "NVO":   "Novo Nordisk",
    "TM":    "Toyota Motor Corp.",
    "SONY":  "Sony Group Corp.",
    "SHOP":  "Shopify Inc.",
    "RY":    "Royal Bank of Canada",
    "TD":    "Toronto-Dominion Bank",
    # Auto
    "F":     "Ford Motor Co.",
    "GM":    "General Motors Co.",
    "RIVN":  "Rivian Automotive",
    "LCID":  "Lucid Group",
}

SP500_SAMPLE = sorted(TICKER_COMPANY_MAP.keys())
