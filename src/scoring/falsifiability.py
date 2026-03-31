"""
Signal falsifiability validator.

Tests the core proposal claim:
    P(revision | signal fired) > P(revision base rate)

Since timestamped I/B/E/S data is unavailable, we approximate using realized
EPS surprises as a proxy. Rationale: if analysts missed in the same direction
our signal predicted, the consensus estimate was wrong in that direction —
i.e., a revision would have been warranted but was never made.

Method
------
1. Align scored quarters to earnings history rows by nearest date.
2. For each quarter where the signal fired (direction != neutral):
   - "Correct" = realized EPS surprise is in the same direction as signal.
3. P(correct | signal fired) = hits / n_fired
4. Base rate P(correct) = fraction of all historical quarters with a
   directional surprise above the significance threshold.
5. Lift = P(correct | fired) / base_rate - 1
6. Test passes when P(correct | fired) > base_rate (lift > 0).

Limitations (surfaced in the UI)
---------------------------------
- Surprise direction ≠ revision direction: a beat can occur even when analysts
  revised estimates upward (and vice versa). This is an approximation.
- Small sample sizes (~4-8 quarters) make the test indicative, not statistically
  conclusive. A z-test is shown with a caveat.
- Lead-time (the metric that matters most) cannot be measured without
  timestamped I/B/E/S data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

# EPS surprise threshold to count as a "significant" / "directional" outcome.
# 2% is conservative — small beats/misses may be noise.
SURPRISE_THRESHOLD_PCT = 2.0


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class QuarterValidation:
    quarter: str                       # e.g. "Q3 2025"
    signal_direction: str              # "upward" | "downward" | "neutral"
    signal_confidence: float
    signal_score: float
    earnings_date: Optional[date]      # matched earnings date from history
    surprise_pct: Optional[float]      # realized EPS surprise %
    surprise_direction: Optional[str]  # "beat" | "miss" | "inline"
    match: Optional[bool]              # True/False when signal fired; None when neutral/no data


@dataclass
class FalsifiabilityResult:
    validations: list[QuarterValidation] = field(default_factory=list)

    # Fired quarters (non-neutral signal with earnings data)
    n_fired: int = 0
    n_fired_correct: int = 0
    directional_accuracy: float = 0.0   # = n_fired_correct / n_fired

    # Base rate across all quarters with earnings data
    n_total: int = 0
    n_total_directional: int = 0
    base_rate: float = 0.0              # fraction of quarters with significant surprise

    # Lift: how much better than random
    lift: float = 0.0                   # directional_accuracy / base_rate - 1
    passes_test: bool = False            # P(correct|fired) > base_rate

    # One-sided z-test (normal approximation of binomial)
    z_score: float = 0.0
    p_value: float = 1.0
    sample_size_warning: bool = True    # always warn when n < 20


# ── Quarter label → approximate end date ─────────────────────────────────────

_QTR_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}
_QTR_END_DAY   = {1: 31, 2: 30, 3: 30, 4: 31}


def _quarter_end_date(quarter_label: str) -> Optional[date]:
    """
    Parse "Q3 2025" → date(2025, 9, 30).
    Also handles "2025-Q3", "3Q25", "FY2025 Q3" etc.
    Returns None if unparseable.
    """
    m = re.search(r"[Qq]?(\d)[^\d]*(\d{4}|\d{2})\b", quarter_label)
    if not m:
        m = re.search(r"(\d{4})[^\d]*[Qq]?(\d)", quarter_label)
        if m:
            year_s, q_s = m.group(1), m.group(2)
        else:
            return None
    else:
        q_s = m.group(1)
        year_raw = m.group(2)
        year_s = ("20" + year_raw) if len(year_raw) == 2 else year_raw

    try:
        q = int(q_s)
        y = int(year_s)
        if q not in _QTR_END_MONTH:
            return None
        return date(y, _QTR_END_MONTH[q], _QTR_END_DAY[q])
    except (ValueError, KeyError):
        return None


# ── Earnings history → per-row lookup ────────────────────────────────────────

def _build_earnings_lookup(earnings_df: pd.DataFrame) -> list[tuple[date, float]]:
    """
    Return a sorted list of (report_date, surprise_pct) from earnings history.
    """
    if earnings_df is None or earnings_df.empty:
        return []

    df = earnings_df.copy()
    # Find date column
    date_col = next(
        (c for c in df.columns
         if c.lower() in ("date", "reporteddate", "quarterdate", "reportdate")),
        None,
    )
    if date_col is None and df.index.name and "date" in df.index.name.lower():
        df = df.reset_index()
        date_col = df.columns[0]
    if date_col is None:
        return []

    # Find surprise column
    surp_col = next(
        (c for c in df.columns if "surprise" in c.lower()),
        None,
    )
    if surp_col is None:
        # Compute from epsActual and epsEstimate
        act_col  = next((c for c in df.columns if "actual" in c.lower()), None)
        est_col  = next((c for c in df.columns if "estimate" in c.lower()), None)
        if act_col and est_col:
            df["_surp"] = (
                (pd.to_numeric(df[act_col], errors="coerce")
                 - pd.to_numeric(df[est_col], errors="coerce"))
                / pd.to_numeric(df[est_col], errors="coerce").abs().replace(0, float("nan"))
                * 100
            )
            surp_col = "_surp"
        else:
            return []

    rows = []
    for _, row in df.iterrows():
        try:
            d = pd.to_datetime(row[date_col]).date()
            s = float(row[surp_col])
            if not math.isnan(s):
                rows.append((d, s))
        except Exception:
            pass
    return sorted(rows)


def _find_nearest(lookup: list[tuple[date, float]], target: date,
                  window_days: int = 90) -> Optional[tuple[date, float]]:
    """Return the (date, surprise_pct) closest to target within window_days."""
    best = None
    best_dist = window_days + 1
    for d, s in lookup:
        dist = abs((d - target).days)
        if dist < best_dist:
            best_dist = dist
            best = (d, s)
    return best if best_dist <= window_days else None


# ── Main computation ──────────────────────────────────────────────────────────

def compute_falsifiability(
    quarters,            # list[QuarterSignals]
    earnings_df: pd.DataFrame,
    surprise_threshold: float = SURPRISE_THRESHOLD_PCT,
) -> FalsifiabilityResult:
    """
    Compute falsifiability statistics by aligning scored quarters to
    realized earnings surprises.

    Parameters
    ----------
    quarters : list[QuarterSignals]
        Scored quarters (newest-first).
    earnings_df : pd.DataFrame
        Output of get_earnings_history(). Must contain date + surprise_pct columns.
    surprise_threshold : float
        Minimum |surprise_pct| to count as a directional outcome (default 2%).
    """
    result = FalsifiabilityResult()
    if not quarters or earnings_df is None or earnings_df.empty:
        return result

    lookup = _build_earnings_lookup(earnings_df)
    if not lookup:
        return result

    validations: list[QuarterValidation] = []

    for qs in quarters:
        qend = _quarter_end_date(qs.quarter)
        matched = _find_nearest(lookup, qend) if qend else None

        earnings_date = matched[0] if matched else None
        surprise_pct  = matched[1] if matched else None

        # Surprise direction
        if surprise_pct is None:
            surprise_dir = None
        elif surprise_pct >= surprise_threshold:
            surprise_dir = "beat"
        elif surprise_pct <= -surprise_threshold:
            surprise_dir = "miss"
        else:
            surprise_dir = "inline"

        # Did our signal match?
        match: Optional[bool] = None
        if qs.direction != "neutral" and surprise_dir is not None and surprise_dir != "inline":
            if qs.direction == "upward" and surprise_dir == "beat":
                match = True
            elif qs.direction == "downward" and surprise_dir == "miss":
                match = True
            else:
                match = False

        validations.append(QuarterValidation(
            quarter=qs.quarter,
            signal_direction=qs.direction,
            signal_confidence=qs.confidence,
            signal_score=qs.weighted_score,
            earnings_date=earnings_date,
            surprise_pct=surprise_pct,
            surprise_direction=surprise_dir,
            match=match,
        ))

    result.validations = validations

    # ── Aggregate statistics ──────────────────────────────────────────────────
    with_data = [v for v in validations if v.surprise_pct is not None]
    result.n_total = len(with_data)
    result.n_total_directional = sum(
        1 for v in with_data
        if v.surprise_direction in ("beat", "miss")
    )
    result.base_rate = (
        result.n_total_directional / result.n_total
        if result.n_total > 0 else 0.0
    )

    fired = [v for v in with_data if v.match is not None]
    result.n_fired = len(fired)
    result.n_fired_correct = sum(1 for v in fired if v.match is True)
    result.directional_accuracy = (
        result.n_fired_correct / result.n_fired
        if result.n_fired > 0 else 0.0
    )

    result.lift = (
        result.directional_accuracy / result.base_rate - 1.0
        if result.base_rate > 0 else 0.0
    )
    result.passes_test = result.directional_accuracy > result.base_rate

    # ── One-sided binomial z-test ─────────────────────────────────────────────
    # H0: p = base_rate   H1: p > base_rate
    if result.n_fired > 0 and 0 < result.base_rate < 1:
        p0 = result.base_rate
        p_hat = result.directional_accuracy
        se = math.sqrt(p0 * (1 - p0) / result.n_fired)
        result.z_score = (p_hat - p0) / se if se > 0 else 0.0
        # Approximate one-sided p-value from z (standard normal CDF)
        result.p_value = _norm_sf(result.z_score)
    else:
        result.z_score = 0.0
        result.p_value = 1.0

    result.sample_size_warning = result.n_fired < 8

    return result


def _norm_sf(z: float) -> float:
    """Survival function (1 - CDF) of standard normal. Approximation via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2))
