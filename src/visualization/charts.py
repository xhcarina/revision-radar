"""
All Plotly visualizations for Revision Radar.

Design language:
  - Dark background (#0e1117) with subtle card panels (#1a1d2e)
  - Teal (#00d4aa) for positive / upward signals
  - Red (#ff4d6d) for negative / downward signals
  - Indigo (#7c85f0) for neutral / informational
  - All charts are returned as plotly.graph_objects.Figure objects
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_WARNING,
    COLOR_BG, COLOR_CARD, COLOR_BORDER, COLOR_TEXT, COLOR_SUBTEXT,
    PLOTLY_TEMPLATE, SIGNAL_COLORS,
)

# NOTE: We use theme="streamlit" in st.plotly_chart so Streamlit handles
# paper_bgcolor / plot_bgcolor and font color to match the OS theme.
# We only set font family, margins, and hover style here.
_LAYOUT_BASE = dict(
    font=dict(family="'Inter', 'Segoe UI', sans-serif", size=13),
    margin=dict(l=20, r=20, t=48, b=20),
    hoverlabel=dict(
        bgcolor="rgba(30,33,48,0.95)",
        bordercolor="rgba(128,128,128,0.3)",
        font=dict(size=12),
    ),
)


def _signal_color(value: float, direction: str = "auto") -> str:
    """Return color based on value or explicit direction."""
    if direction == "upward":
        return COLOR_POSITIVE
    if direction == "downward":
        return COLOR_NEGATIVE
    if value > 0.05:
        return COLOR_POSITIVE
    if value < -0.05:
        return COLOR_NEGATIVE
    return COLOR_NEUTRAL


# ── 1. Revision Probability Gauge ─────────────────────────────────────────────

def gauge_chart(
    probability: float,
    direction: str,
    magnitude: str,
    confidence: float,
) -> go.Figure:
    """
    Large semi-circular gauge showing revision probability.
    The gauge reads 0 → 100. Needle position reflects conviction level.
    """
    # Normalise so that 50 = neutral; >50 = upward; <50 = downward
    if direction == "upward":
        gauge_value = 50 + probability * 50
    elif direction == "downward":
        gauge_value = 50 - (1 - probability) * 50
    else:
        gauge_value = 50.0

    needle_color = _signal_color(1 if direction == "upward" else (-1 if direction == "downward" else 0))

    magnitude_pct_map = {
        "negligible": "< 1%",
        "small":      "1 – 3%",
        "medium":     "3 – 7%",
        "large":      "> 7%",
    }
    mag_label = magnitude_pct_map.get(magnitude, "")
    conf_pct  = int(round(confidence * 100))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(gauge_value, 1),
        number=dict(
            suffix="%",
            font=dict(size=52, color=needle_color),
        ),
        delta=dict(reference=50, valueformat=".1f", suffix="pp vs neutral"),
        title=dict(
            text=(
                f"<b>Revision Conviction</b><br>"
                f"<span style='font-size:14px;color:rgba(128,128,128,0.75)'>"
                f"Direction: <b style='color:{needle_color}'>{direction.title()}</b> &nbsp;|&nbsp; "
                f"Magnitude: <b>{mag_label}</b> &nbsp;|&nbsp; "
                f"Confidence: <b>{conf_pct}%</b>"
                f"</span>"
            ),
            font=dict(size=16, color=None),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="rgba(128,128,128,0.4)",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["Strong↓", "Lean↓", "Neutral", "Lean↑", "Strong↑"],
                tickfont=dict(size=11, color='rgba(128,128,128,0.75)'),
            ),
            bar=dict(color=needle_color, thickness=0.25),
            bgcolor="rgba(128,128,128,0.08)",
            borderwidth=1,
            bordercolor="rgba(128,128,128,0.3)",
            steps=[
                dict(range=[0, 25],   color="rgba(255,77,109,0.12)"),
                dict(range=[25, 42],  color="rgba(255,77,109,0.06)"),
                dict(range=[42, 58],  color="rgba(128,128,128,0.06)"),
                dict(range=[58, 75],  color="rgba(0,212,170,0.06)"),
                dict(range=[75, 100], color="rgba(0,212,170,0.12)"),
            ],
            threshold=dict(
                line=dict(color=needle_color, width=3),
                thickness=0.8,
                value=gauge_value,
            ),
        ),
    ))
    fig.update_layout(**_LAYOUT_BASE, height=340)
    return fig


# ── 2. Signal Timeline ────────────────────────────────────────────────────────

def signal_timeline(quarters_data: list[dict]) -> go.Figure:
    """
    Multi-line chart showing each signal dimension score across quarters.
    quarters_data: list of {quarter: str, signals: dict[str, float]}
    """
    if not quarters_data:
        return _empty_chart("No quarterly data available")

    quarters_data = list(reversed(quarters_data))  # chronological order
    quarters = [d["quarter"] for d in quarters_data]

    # 5-signal consolidated model (matches config.SIGNAL_WEIGHTS)
    signal_keys = [
        ("tone_delta",              "Mgmt Tone Delta"),
        ("guidance_quantification", "Guidance Specificity"),
        ("risk_escalation",         "Risk Escalation (inv.)"),
        ("qa_deflection",           "Q&A Deflection (inv.)"),
        ("news_sentiment",          "News Sentiment"),
    ]

    fig = go.Figure()

    for key, label in signal_keys:
        values = [d.get("signals", {}).get(key, 0.0) for d in quarters_data]
        color = SIGNAL_COLORS.get(key, COLOR_NEUTRAL)
        fig.add_trace(go.Scatter(
            x=quarters,
            y=values,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color, line=dict(color='rgba(0,0,0,0.3)', width=1.5)),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Score: %{{y:.2f}}<extra></extra>",
        ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.4)", dash="dot", width=1))
    fig.add_hrect(y0=-0.1, y1=0.1, fillcolor='rgba(128,128,128,0.12)', opacity=1, line_width=0)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="<b>Qualitative Signal Trends by Quarter</b>",
                   font=dict(size=15), x=0),
        yaxis=dict(title="Signal Score (−1 = bearish, +1 = bullish)",
                   range=[-1.1, 1.1], zeroline=False,
                   gridcolor="rgba(128,128,128,0.2)", gridwidth=0.5),
        xaxis=dict(title="", gridcolor="rgba(128,128,128,0.2)", gridwidth=0.5),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
        height=400,
    )
    return fig


# ── 3. Factor Waterfall / Contribution Chart ──────────────────────────────────

def waterfall_chart(signal_contributions: dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart showing each factor's weighted contribution
    to the overall revision signal.
    """
    from config import SIGNAL_WEIGHTS
    if not signal_contributions:
        return _empty_chart("No signal contributions available")

    label_map = {
        "tone_delta":              "Management Tone",
        "guidance_quantification": "Guidance Specificity",
        "qa_deflection":           "Q&A Transparency",
        "risk_escalation":         "Risk Disclosure",
        "news_sentiment":          "News Sentiment",
        # legacy keys kept for backward compatibility
        "hedging_intensity":       "Hedging Language",
        "forward_guidance":        "Forward Guidance",
    }

    items = []
    for key, weight in SIGNAL_WEIGHTS.items():
        raw_val = signal_contributions.get(key, 0.0)
        contribution = raw_val * weight
        items.append({
            "factor":       label_map.get(key, key),
            "contribution": contribution,
            "weight_pct":   int(weight * 100),
            "hover":        f"(wt {int(weight * 100)}%)",
        })

    # Add peer adjustment as a separate bar (it's a direct additive adjustment, not multiplied by weight)
    peer_adj = signal_contributions.get("peer_signals", 0.0)
    if peer_adj != 0.0:
        items.append({
            "factor":       "Peer Company Signals",
            "contribution": peer_adj,
            "weight_pct":   0,
            "hover":        "(peer adjustment, cap ±0.20)",
        })

    items.sort(key=lambda x: x["contribution"])

    colors = [
        COLOR_POSITIVE if x["contribution"] > 0.005
        else COLOR_NEGATIVE if x["contribution"] < -0.005
        else COLOR_NEUTRAL
        for x in items
    ]

    def _bar_text(x: dict) -> str:
        if x["weight_pct"]:
            return f"{x['contribution']:+.3f}  (wt {x['weight_pct']}%)"
        return f"{x['contribution']:+.3f}  {x['hover']}"

    fig = go.Figure(go.Bar(
        x=[x["contribution"] for x in items],
        y=[x["factor"] for x in items],
        orientation="h",
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.1)', width=0.5)),
        text=[_bar_text(x) for x in items],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color='rgba(128,128,128,0.4)', width=1.5))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="<b>Factor Contributions to Revision Signal</b>",
                   font=dict(size=15), x=0),
        xaxis=dict(title="Weighted contribution (negative = downward pressure)",
                   gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        yaxis=dict(title="", automargin=True),
        height=360,
    )
    return fig


# ── 4. Management Tone Radar ──────────────────────────────────────────────────

def tone_radar(
    quarters_data: list[dict],
    n_quarters: int = 2,
) -> go.Figure:
    """
    Spider / radar chart comparing tone dimensions across the last N quarters.
    quarters_data: list of {quarter: str, transcript: TranscriptSignals}
    """
    dimensions = [
        ("overall_tone",           "Overall Tone"),
        ("forward_guidance_strength", "Fwd Guidance"),
        ("guidance_quantification_rate", "Guide Specificity"),
        ("demand_language_tone",   "Demand Language"),
        ("margin_language_tone",   "Margin Language"),
        ("capex_commitment_tone",  "CapEx Commitment"),
        ("hiring_headcount_tone",  "Hiring Language"),
    ]
    dim_keys  = [d[0] for d in dimensions]
    dim_names = [d[1] for d in dimensions] + [dimensions[0][1]]  # close the loop

    fig = go.Figure()
    # Pre-defined as rgba() strings — Plotly does not accept 8-digit hex (#rrggbbaa)
    fill_colors = [
        "rgba(0,212,170,0.12)",    # COLOR_POSITIVE teal
        "rgba(124,133,240,0.12)",  # COLOR_NEUTRAL indigo
        "rgba(245,166,35,0.12)",   # COLOR_WARNING amber
    ]
    line_colors = [COLOR_POSITIVE, COLOR_NEUTRAL, COLOR_WARNING]

    for i, qd in enumerate(quarters_data[:n_quarters]):
        t = qd.get("transcript")
        if t is None:
            continue
        values = [getattr(t, k, 0.0) for k in dim_keys]
        # Shift from [-1,1] to [0,1] — radar axis must be non-negative
        norm = [(v + 1) / 2 for v in values]
        norm.append(norm[0])  # close polygon

        fig.add_trace(go.Scatterpolar(
            r=norm,
            theta=dim_names,
            fill="toself",
            name=qd.get("quarter", f"Q{i+1}"),
            line=dict(color=line_colors[i % len(line_colors)], width=2),
            fillcolor=fill_colors[i % len(fill_colors)],
            opacity=0.9,
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["−1", "−0.5", "0", "+0.5", "+1"],
                gridcolor="rgba(128,128,128,0.2)",
                linecolor="rgba(128,128,128,0.2)",
                tickfont=dict(size=9, color='rgba(128,128,128,0.75)'),
            ),
            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)", linecolor="rgba(128,128,128,0.2)"),
            bgcolor="rgba(128,128,128,0.08)",
        ),
        title=dict(text="<b>Management Tone Radar</b>",
                   font=dict(size=15), x=0),
        legend=dict(orientation="h", y=-0.1),
        height=420,
    )
    return fig


# ── 5. News Sentiment Timeline ────────────────────────────────────────────────

def news_sentiment_timeline(sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of daily news sentiment + volume from GDELT — single y-axis.
    Volume bars are scaled to the sentiment axis range so they render reliably;
    hover always shows the real article count via customdata.
    Columns: date (datetime), sentiment (float), volume (int), sentiment_ma7 (float)
    """
    if sentiment_df.empty:
        return _empty_chart("No news data available")

    df = sentiment_df.copy()

    # Normalize volume into sentiment axis range so it renders on the same axis
    has_volume = "volume" in df.columns and df["volume"].max() > 0
    if has_volume:
        max_vol = float(df["volume"].max())
        sent_range = max(float(df["sentiment"].abs().max()), 0.15)
        # Scale so tallest volume bar = 90% of the positive sentiment range
        df["_vol_norm"] = df["volume"] / max_vol * sent_range * 0.90

    fig = go.Figure()

    # ── Sentiment bars (behind, teal/red) ────────────────────────────────────
    colors = [COLOR_POSITIVE if s >= 0 else COLOR_NEGATIVE
              for s in df["sentiment"]]
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["sentiment"],
        name="Daily Sentiment",
        marker=dict(color=colors, opacity=0.55),
        selected=dict(marker=dict(opacity=0.55)),
        unselected=dict(marker=dict(opacity=0.55)),
        hovertemplate="%{x|%b %d}<br>Sentiment: %{y:.3f}<extra></extra>",
    ))

    # ── Volume bars (on top, purple) — hover shows real count ────────────────
    if has_volume:
        fig.add_trace(go.Bar(
            x=df["date"],
            y=df["_vol_norm"],
            name="Article Volume",
            marker=dict(color="rgba(124,133,240,0.75)"),
            selected=dict(marker=dict(color="rgba(124,133,240,0.75)")),
            unselected=dict(marker=dict(color="rgba(124,133,240,0.75)")),
            customdata=df["volume"],
            hovertemplate=(
                "%{x|%b %d}<br>"
                "<b>%{customdata} articles</b><br>"
                "<i>Click to see articles</i>"
                "<extra></extra>"
            ),
        ))

    # ── 7-day MA line ─────────────────────────────────────────────────────────
    if "sentiment_ma7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["sentiment_ma7"],
            name="7-day Avg",
            line=dict(color=COLOR_WARNING, width=2.5),
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=1.0)),
            hovertemplate="%{x|%b %d}<br>7d Avg: %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=COLOR_BORDER, dash="dot", width=1))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="<b>News Sentiment Timeline (GDELT)</b>",
                   font=dict(size=15), x=0),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2),
        height=360,
        yaxis=dict(
            title="Sentiment (−1 bearish → +1 bullish)",
            gridcolor="rgba(128,128,128,0.2)",
        ),
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


# ── 6. Analyst Estimate Context Chart ────────────────────────────────────────

def analyst_estimate_chart(earnings_history: pd.DataFrame) -> go.Figure:
    """
    Bar chart of historical EPS: actual vs. estimate, with surprise %.
    This is display-only context — NOT used as an input signal.
    """
    if earnings_history.empty:
        return _empty_chart("No earnings history available")

    df = earnings_history.copy()
    # Attempt flexible column matching
    actual_col   = next((c for c in df.columns if "actual" in c.lower()), None)
    estimate_col = next((c for c in df.columns if "estimate" in c.lower()), None)
    date_col     = next((c for c in df.columns if "date" in c.lower()), None)
    surprise_col = next((c for c in df.columns if "surprise" in c.lower()), None)

    if not actual_col:
        return _empty_chart("Insufficient earnings data")
    has_estimates = estimate_col is not None and df[estimate_col].notna().any()

    # Drop rows only where actual is missing; estimate may be partially absent
    df = df.dropna(subset=[actual_col])
    if df.empty:
        return _empty_chart("No earnings data available")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        x_vals = df[date_col].dt.strftime("%b '%y")
    else:
        x_vals = [f"Q{i+1}" for i in range(len(df))]

    surprise_colors = []
    if surprise_col and surprise_col in df.columns:
        surprise_colors = [
            COLOR_POSITIVE if (pd.notna(v) and v >= 0) else COLOR_NEGATIVE
            for v in df[surprise_col]
        ]
    else:
        surprise_colors = [COLOR_NEUTRAL] * len(df)

    fig = go.Figure()
    # Actual EPS bars
    fig.add_trace(go.Bar(
        x=x_vals,
        y=df[actual_col],
        name="Actual EPS",
        marker=dict(color=surprise_colors),
    ))
    # Consensus estimate as a horizontal dash — only where data exists
    if has_estimates:
        est_mask = df[estimate_col].notna()
        fig.add_trace(go.Scatter(
            x=[x for x, m in zip(x_vals, est_mask) if m],
            y=df.loc[est_mask, estimate_col],
            mode="markers",
            name="Consensus Estimate",
            marker=dict(
                symbol="line-ew",
                size=24,
                line=dict(color=COLOR_NEUTRAL, width=3),
                color=COLOR_NEUTRAL,
            ),
        ))

    has_surprise = surprise_col is not None and surprise_col in df.columns and df[surprise_col].notna().any()
    if has_surprise:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=df[surprise_col],
            mode="lines+markers",
            name="Surprise %",
            yaxis="y2",
            line=dict(color=COLOR_WARNING, width=2, dash="dot"),
            marker=dict(size=7),
        ))

    layout_extra = dict(
        yaxis2=dict(title="Surprise %", overlaying="y", side="right",
                    showgrid=False, zeroline=False)
    ) if has_surprise else {}

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="<b>Historical EPS: Actual vs. Consensus</b> (display context only)",
                   font=dict(size=15), x=0),
        barmode="group",
        yaxis=dict(title="EPS ($)", gridcolor="rgba(128,128,128,0.2)"),
        legend=dict(orientation="h", y=-0.2),
        height=340,
        **layout_extra,
    )
    return fig


# ── 7. Revision Probability Over Horizons ────────────────────────────────────

def horizon_bars(
    prob_30d: float,
    prob_60d: float,
    prob_90d: float,
    direction: str,
) -> go.Figure:
    """
    Three horizontal bars showing revision probability at 30/60/90-day horizons.
    Probability decays with time as uncertainty increases.
    """
    color = COLOR_POSITIVE if direction == "upward" else (
        COLOR_NEGATIVE if direction == "downward" else COLOR_NEUTRAL
    )
    labels = ["30 days", "60 days", "90 days"]
    probs  = [prob_30d, prob_60d, prob_90d]
    texts  = [f"{int(p * 100)}%" for p in probs]

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        marker=dict(
            color=probs,
            colorscale=[[0, COLOR_NEGATIVE], [0.5, COLOR_NEUTRAL], [1, COLOR_POSITIVE]],
            cmin=0, cmax=1,
            line=dict(color='rgba(0,0,0,0.1)', width=0.5),
        ),
        text=texts,
        textposition="inside",
        textfont=dict(size=16, color='#0e1117'),
        hovertemplate="<b>%{y}</b><br>Probability: %{text}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=f"<b>Revision Probability by Horizon</b> — {direction.title()} bias",
                   font=dict(size=15), x=0),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(title=""),
        height=220,
    )
    return fig


# ── 8. Quote Highlight Cards (HTML) ──────────────────────────────────────────

def quote_card_html(
    quotes: list[str],
    sentiment: str = "neutral",
    source: str = "",
    quarter: str = "",
    source_url: str = "",
) -> str:
    """Return an HTML string for quote cards with colour-coded sentiment.
    If source_url is provided, the source label links to the original filing/article."""
    if not quotes:
        return ""
    border = {
        "positive": COLOR_POSITIVE,
        "negative": COLOR_NEGATIVE,
        "neutral":  COLOR_NEUTRAL,
    }.get(sentiment, COLOR_NEUTRAL)

    icon = {"positive": "↑", "negative": "↓", "neutral": "•"}.get(sentiment, "•")
    if source_url:
        source_label = (
            f'<a href="{source_url}" target="_blank" rel="noopener" '
            f'style="color:{border};text-decoration:none;opacity:0.8;">'
            f'{source} ↗</a>'
        )
    else:
        source_label = source

    cards = []
    for q in quotes[:5]:
        if not q:
            continue
        cards.append(f"""
        <div style="
            border-left: 3px solid {border};
            background: var(--secondary-background-color);
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            line-height: 1.5;
        ">
            <span style="color:{border};font-weight:700;margin-right:6px;">{icon}</span>
            <em>&ldquo;{q}&rdquo;</em>
            <div style="margin-top:5px;font-size:11px;opacity:0.6;">
                {source_label} &nbsp;·&nbsp; {quarter}
            </div>
        </div>
        """)
    return "\n".join(cards)


# ── 9. Mini signal sparklines ─────────────────────────────────────────────────

def mini_sparkline(values: list[float], color: str = COLOR_NEUTRAL) -> go.Figure:
    """Tiny inline sparkline for a single signal across quarters."""
    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.13)",  # fixed: Plotly rejects 8-digit hex
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-1.2, 1.2]),
        showlegend=False,
    )
    return fig


# ── 10. Supply chain peer heatmap ─────────────────────────────────────────────

def peer_heatmap(peer_signals: list[dict]) -> go.Figure:
    """
    Heatmap showing supply-chain peer signal direction and strength.
    peer_signals: [{ticker, name, direction, strength, insight}]
    """
    if not peer_signals:
        return _empty_chart("No peer supply-chain data")

    tickers  = [p.get("ticker", "") for p in peer_signals]
    dirs     = [p.get("signal_direction", "neutral") for p in peer_signals]
    strengths = [p.get("signal_strength", 0.0) for p in peer_signals]
    signed   = [s if d == "positive" else (-s if d == "negative" else 0)
                for s, d in zip(strengths, dirs)]

    colors = [
        COLOR_POSITIVE if v > 0.1 else (COLOR_NEGATIVE if v < -0.1 else COLOR_NEUTRAL)
        for v in signed
    ]

    fig = go.Figure(go.Bar(
        x=signed,
        y=tickers,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.2f}" for v in signed],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>Signal: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color='rgba(128,128,128,0.4)', width=1.5))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="<b>Supply-Chain Peer Signals</b>",
                   font=dict(size=15), x=0),
        xaxis=dict(title="Signal strength (negative = downward pressure on target)",
                   gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(title=""),
        height=max(200, len(tickers) * 40 + 80),
    )
    return fig


# ── 11. Price History with Signal Annotations ────────────────────────────────

def price_history_chart(
    price_df: pd.DataFrame,
    signal_annotations: list[dict] | None = None,
) -> go.Figure:
    """
    Price history line chart with colored vertical markers at each filing date
    showing the model's signal direction at that point.

    signal_annotations: [{date, quarter, direction, score}]
    This is CONTROL GROUP data — stock price is never used as a model input.
    """
    if price_df.empty:
        return _empty_chart("No price history available")

    date_col  = next((c for c in price_df.columns if "date" in c.lower()), None)
    close_col = next((c for c in price_df.columns if "close" in c.lower()), None)
    if not date_col or not close_col:
        return _empty_chart("No price history available")

    df = price_df.copy()
    df[date_col]  = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)

    fig = go.Figure()

    # Shaded regions per signal period
    annotations = signal_annotations or []
    # Sort oldest first so regions are drawn left→right
    annotations = sorted(annotations, key=lambda a: a.get("date", ""))
    for i, ann in enumerate(annotations):
        ann_date = pd.to_datetime(ann["date"], errors="coerce")
        if pd.isna(ann_date):
            continue
        next_date = (
            pd.to_datetime(annotations[i + 1]["date"], errors="coerce")
            if i + 1 < len(annotations) else df[date_col].max()
        )
        color = (
            "rgba(0,212,170,0.08)"   if ann["direction"] == "upward"   else
            "rgba(255,77,109,0.08)"  if ann["direction"] == "downward" else
            "rgba(128,128,128,0.06)"
        )
        fig.add_vrect(
            x0=ann_date, x1=next_date,
            fillcolor=color, line_width=0,
            annotation_text=ann["quarter"],
            annotation_position="top left",
            annotation=dict(font=dict(size=9, color="rgba(128,128,128,0.6)"), yshift=4),
        )

    # Price line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[close_col],
        mode="lines",
        name="Close Price",
        line=dict(color=COLOR_NEUTRAL, width=2),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
    ))

    # Signal direction markers at each filing date
    for ann in annotations:
        ann_date = pd.to_datetime(ann["date"], errors="coerce")
        if pd.isna(ann_date):
            continue
        # Find the closest price
        idx = (df[date_col] - ann_date).abs().idxmin()
        price_at = df.loc[idx, close_col]
        color = (COLOR_POSITIVE if ann["direction"] == "upward" else
                 COLOR_NEGATIVE if ann["direction"] == "downward" else COLOR_NEUTRAL)
        symbol = "triangle-up" if ann["direction"] == "upward" else (
            "triangle-down" if ann["direction"] == "downward" else "circle")
        fig.add_trace(go.Scatter(
            x=[ann_date],
            y=[price_at],
            mode="markers",
            marker=dict(symbol=symbol, size=12, color=color,
                        line=dict(color="rgba(0,0,0,0.4)", width=1)),
            name=ann["quarter"],
            showlegend=False,
            hovertemplate=(
                f"<b>{ann['quarter']}</b><br>"
                f"Signal: {ann['direction'].title()} "
                f"(score {ann['score']:+.2f})<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="<b>Price History</b> — shading shows model signal direction per quarter"
                 "<br><span style='font-size:11px;color:rgba(128,128,128,0.6);'>"
                 "Control group: price data is never used as a model input</span>",
            font=dict(size=14), x=0,
        ),
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(title="Price ($)", gridcolor="rgba(128,128,128,0.2)"),
        showlegend=False,
        height=360,
    )
    return fig


# ── 12. Sector Peer Tone Comparison ──────────────────────────────────────────

def sector_tone_comparison(
    target_ticker: str,
    target_signals: dict,
    peer_scores: list[dict],
) -> go.Figure:
    """
    Grouped bar chart comparing the target company's tone/hedging scores
    against an average of sector peers on the same dimensions.

    target_signals: dict with keys matching TranscriptSignals fields
    peer_scores:    [{ticker, overall_tone, hedging_intensity,
                      forward_guidance_strength, risk_language_escalation}]
    """
    dims = [
        ("overall_tone",              "Overall Tone"),
        ("hedging_intensity",         "Hedging Intensity"),
        ("forward_guidance_strength", "Forward Guidance"),
        ("risk_language_escalation",  "Risk Language"),
        ("demand_language_tone",      "Demand Language"),
        ("margin_language_tone",      "Margin Language"),
    ]

    peer_scores = [p for p in peer_scores if p]
    if not peer_scores and not target_signals:
        return _empty_chart("No peer tone data available")

    labels        = [d[1] for d in dims]
    target_values = [target_signals.get(d[0], 0.0) for d in dims]

    fig = go.Figure()

    # Target company bar
    target_colors = [
        COLOR_POSITIVE if v > 0.05 else COLOR_NEGATIVE if v < -0.05 else COLOR_NEUTRAL
        for v in target_values
    ]
    fig.add_trace(go.Bar(
        name=target_ticker,
        x=labels,
        y=target_values,
        marker=dict(color=target_colors, opacity=0.85),
        hovertemplate="<b>" + target_ticker + "</b><br>%{x}: %{y:.2f}<extra></extra>",
    ))

    # Peer average bar (if we have peer data)
    if peer_scores:
        peer_avg = {}
        for d_key, _ in dims:
            vals = [p.get(d_key) for p in peer_scores if p.get(d_key) is not None]
            peer_avg[d_key] = sum(vals) / len(vals) if vals else 0.0

        peer_values = [peer_avg.get(d[0], 0.0) for d in dims]
        fig.add_trace(go.Bar(
            name="Sector Peer Avg",
            x=labels,
            y=peer_values,
            marker=dict(color=COLOR_NEUTRAL, opacity=0.45,
                        line=dict(color=COLOR_NEUTRAL, width=1.5)),
            hovertemplate="<b>Sector Avg</b><br>%{x}: %{y:.2f}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.4)", dash="dot", width=1))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="<b>Company vs. Sector Peer Tone</b> — most recent quarter"
                 "<br><span style='font-size:11px;color:rgba(128,128,128,0.6);'>"
                 "Company-specific stress (above/below peers) is a stronger signal</span>",
            font=dict(size=14), x=0,
        ),
        barmode="group",
        yaxis=dict(title="Score (−1 bearish → +1 bullish)",
                   range=[-1.1, 1.1], gridcolor="rgba(128,128,128,0.2)"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.18),
        height=360,
    )
    return fig


# ── Utility ───────────────────────────────────────────────────────────────────

def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(**_LAYOUT_BASE, height=200)
    return fig


def direction_badge_html(direction: str, probability: float) -> str:
    """Inline HTML badge for direction + probability."""
    color = (COLOR_POSITIVE if direction == "upward"
             else COLOR_NEGATIVE if direction == "downward"
             else COLOR_NEUTRAL)
    arrow = "↑" if direction == "upward" else "↓" if direction == "downward" else "→"
    pct = int(probability * 100)
    return (
        f'<span style="background:{color}22;border:1px solid {color};'
        f'color:{color};padding:3px 10px;border-radius:20px;font-weight:700;'
        f'font-size:14px;">{arrow} {direction.upper()} &nbsp; {pct}%</span>'
    )
