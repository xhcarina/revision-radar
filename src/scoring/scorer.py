"""
Signal aggregation and revision probability scoring.

Architecture:
  1. Each extraction module produces typed signal objects.
  2. This module normalises them to a common [-1, +1] scale where
       -1 = strong downward revision signal
       +1 = strong upward revision signal
  3. A weighted sum is computed per SIGNAL_WEIGHTS in config.
  4. The weighted sum is passed through a sigmoid to yield probability.
  5. Direction and magnitude are derived from the signed weighted sum.

The weighted sum, not just the probability, is returned so that the UI
can display individual factor contributions (waterfall chart).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import SIGNAL_WEIGHTS, MAGNITUDE_THRESHOLDS, MAGNITUDE_LABELS
from src.extraction.llm_extractor import (
    TranscriptSignals,
    MDASignals,
    NewsSignals,
    NarrativeOutput,
)


# ─────────────────────────── Data containers ─────────────────────────────────

@dataclass
class QuarterSignals:
    """All extracted signals for a single quarter."""
    quarter: str
    transcript: Optional[TranscriptSignals] = None
    mda:        Optional[MDASignals]        = None
    # Derived after normalization
    normalized: dict[str, float]            = field(default_factory=dict)
    weighted_score: float                   = 0.0
    revision_probability: float             = 0.5
    direction: str                          = "neutral"
    magnitude: str                          = "negligible"
    confidence: float                       = 0.0


@dataclass
class FullAnalysis:
    """Complete analysis result for one company."""
    ticker: str
    company_name: str
    quarters: list[QuarterSignals]          = field(default_factory=list)
    news_signals: Optional[NewsSignals]     = None
    narrative: Optional[NarrativeOutput]   = None
    # Aggregated across quarters
    overall_probability: float              = 0.5
    overall_direction: str                  = "neutral"
    overall_magnitude: str                  = "negligible"
    overall_confidence: float               = 0.0
    signal_contributions: dict[str, float] = field(default_factory=dict)
    # Trend (most recent minus oldest)
    trend_direction: str                    = "stable"    # "improving", "deteriorating", "stable"


# ─────────────────────────── Normalization helpers ───────────────────────────

def _sigmoid(x: float) -> float:
    """Sigmoid squash: R → (0, 1). Steepness = 2."""
    return 1.0 / (1.0 + math.exp(-2.0 * x))


def _signal_direction_score(signal_str: str) -> float:
    """Convert upward/neutral/downward string to +1/0/-1."""
    return {"upward": 1.0, "neutral": 0.0, "downward": -1.0}.get(signal_str, 0.0)


def _normalize_transcript(t: TranscriptSignals) -> dict[str, float]:
    """
    Map TranscriptSignals to the 5 canonical signal dimensions.
    Values in [-1, +1]; positive = upward revision pressure.

    Consolidation vs. prior 7-signal model:
      • forward_guidance absorbed into tone_delta (both measure forward positivity)
      • hedging_intensity absorbed into risk_escalation (both are bearish language detectors)
    """
    ops_signals = (
        t.demand_language_tone  * 0.35
        + t.margin_language_tone  * 0.35
        + t.capex_commitment_tone * 0.15
        + t.hiring_headcount_tone * 0.15
    )
    # tone_delta: overall sentiment + QoQ shift (primary) + forward guidance + ops
    tone_delta = (
        t.overall_tone              * 0.25
        + t.tone_vs_prior_quarter   * 0.40
        + t.forward_guidance_strength * 0.20
        + ops_signals               * 0.15
    )
    guidance_quantification = (t.guidance_quantification_rate - 0.5) * 2

    # risk_escalation: risk language (60%) + hedging (40%) — both bearish, now unified
    risk_escalation = -(t.risk_language_escalation * 0.60 + t.hedging_intensity * 0.40)

    qa_deflection = -t.qa_deflection_score

    explicit = _signal_direction_score(t.revision_signal) * t.revision_confidence
    return {
        "tone_delta":              _blend(tone_delta,              explicit, 0.70),
        "guidance_quantification": _blend(guidance_quantification, explicit, 0.75),
        "risk_escalation":         _blend(risk_escalation,         explicit, 0.75),
        "qa_deflection":           _blend(qa_deflection,           explicit, 0.85),
        "news_sentiment":          0.0,
    }


def _normalize_mda(m: MDASignals) -> dict[str, float]:
    """
    Map MDASignals to the 5 canonical signal dimensions.

    When derived from a diff (mda_is_delta=True in score_quarter), these scores
    are counted 2x — the diff is the primary signal per the project thesis.
    liquidity_concern absorbed into risk_escalation (unified bearish language).
    """
    explicit = _signal_direction_score(m.revision_signal) * m.revision_confidence
    # risk_escalation: risk language (50%) + cost pressure (30%) + liquidity (20%)
    risk_escalation = -(
        m.risk_escalation        * 0.50
        + m.cost_pressure_signals  * 0.30
        + m.liquidity_concern      * 0.20
    )
    return {
        "tone_delta":              _blend(m.forward_looking_tone,             explicit, 0.65),
        "guidance_quantification": _blend((m.guidance_specificity - 0.5) * 2, explicit, 0.70),
        "risk_escalation":         _blend(risk_escalation,                    explicit, 0.70),
        "qa_deflection":           0.0,
        "news_sentiment":          0.0,
    }


def _normalize_news(n: NewsSignals) -> dict[str, float]:
    """Map NewsSignals to the 5 canonical signal dimensions (news_sentiment only)."""
    explicit = _signal_direction_score(n.revision_signal) * n.revision_confidence
    composite = (
        n.news_sentiment_score * 0.30
        + n.supply_chain_signal  * 0.30
        + n.competitive_signal   * 0.20
        + n.macro_signal         * 0.10
        + n.regulatory_signal    * 0.10
    )
    return {
        "tone_delta":              0.0,
        "guidance_quantification": 0.0,
        "risk_escalation":         0.0,
        "qa_deflection":           0.0,
        "news_sentiment":          _blend(composite, explicit, 0.60),
    }


def _blend(derived: float, explicit: float, derived_weight: float) -> float:
    """
    Blend a heuristically-derived score with Claude's explicit classification.
    Clamp to [-1, +1].
    """
    val = derived * derived_weight + explicit * (1 - derived_weight)
    return max(-1.0, min(1.0, val))


# ─────────────────────────── Scoring engine ──────────────────────────────────

class RevisionScorer:

    @staticmethod
    def score_quarter(
        quarter: str,
        transcript: Optional[TranscriptSignals] = None,
        mda: Optional[MDASignals] = None,
        mda_is_delta: bool = False,
    ) -> QuarterSignals:
        """
        Produce a fully scored QuarterSignals from raw extraction outputs.

        When mda_is_delta=True, the MDA signals come from a sentence-level
        QoQ diff (new/removed language only) — the primary signal per the
        project thesis. These are weighted 2x relative to the 8-K transcript,
        which provides cross-validation and fills in qa_deflection (a dimension
        the diff cannot measure). When mda_is_delta=False (full MD&A text),
        both sources are weighted equally.
        """
        qs = QuarterSignals(quarter=quarter, transcript=transcript, mda=mda)

        # Aggregate dimension scores from available sources
        dim_scores: dict[str, list[float]] = {k: [] for k in SIGNAL_WEIGHTS}

        if transcript:
            norm = _normalize_transcript(transcript)
            for k, v in norm.items():
                if k in dim_scores:
                    dim_scores[k].append(v)

        if mda:
            norm = _normalize_mda(mda)
            # Diff-based MDA is the primary signal — count it twice so it
            # carries ~2/3 weight vs the transcript's 1/3. Full-text MDA
            # is equally weighted (no privileged position).
            repeat = 2 if mda_is_delta else 1
            for k, v in norm.items():
                if k in dim_scores and v != 0.0:
                    for _ in range(repeat):
                        dim_scores[k].append(v)

        # Average each dimension
        averaged: dict[str, float] = {
            k: (sum(vs) / len(vs) if vs else 0.0)
            for k, vs in dim_scores.items()
        }
        qs.normalized = averaged

        # Weighted sum
        wsum = sum(averaged.get(k, 0.0) * w for k, w in SIGNAL_WEIGHTS.items())
        qs.weighted_score = round(wsum, 4)

        # Probability (0=strong down, 0.5=neutral, 1=strong up)
        qs.revision_probability = round(_sigmoid(wsum), 4)

        # Direction
        if abs(wsum) < 0.05:
            qs.direction = "neutral"
        elif wsum > 0:
            qs.direction = "upward"
        else:
            qs.direction = "downward"

        # Magnitude
        abs_score = abs(wsum)
        if abs_score < MAGNITUDE_THRESHOLDS["negligible"]:
            qs.magnitude = "negligible"
        elif abs_score < MAGNITUDE_THRESHOLDS["small"]:
            qs.magnitude = "small"
        elif abs_score < MAGNITUDE_THRESHOLDS["medium"]:
            qs.magnitude = "medium"
        else:
            qs.magnitude = "large"

        # Confidence: average of revision_confidence from available sources
        confidences = []
        if transcript:
            confidences.append(transcript.revision_confidence)
        if mda:
            confidences.append(mda.revision_confidence)
        qs.confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.4

        return qs

    @staticmethod
    def _peer_adjustment(peer_signals: list[dict]) -> float:
        """
        Compute a signed score adjustment from peer company signals.

        Each peer dict has:
            signal_direction: "upward" | "neutral" | "downward"
            signal_strength:  float in [0, 1]
            relevance_score:  float in [0, 1]

        The adjustment is a relevance-weighted average of directional scores,
        capped at ±0.20 so peers can shift but never dominate the final score.
        """
        if not peer_signals:
            return 0.0
        total_w = 0.0
        wsum = 0.0
        for ps in peer_signals:
            direction = ps.get("signal_direction", "neutral")
            strength = float(ps.get("signal_strength", 0.5))
            relevance = float(ps.get("relevance_score", 0.5))
            score = _signal_direction_score(direction) * strength
            wsum += score * relevance
            total_w += relevance
        if total_w == 0:
            return 0.0
        raw = wsum / total_w
        # Cap at ±0.20
        return max(-0.20, min(0.20, raw))

    @staticmethod
    def aggregate(
        quarters: list[QuarterSignals],
        news: Optional[NewsSignals] = None,
        peer_signals: Optional[list[dict]] = None,
    ) -> tuple[float, str, str, float, dict[str, float]]:
        """
        Aggregate multiple quarters + news + peer signals into overall revision metrics.

        Returns:
            probability, direction, magnitude, confidence, signal_contributions
        """
        if not quarters:
            return 0.5, "neutral", "negligible", 0.0, {}

        # Weight recent quarters more heavily (geometric decay)
        weights = [0.5 ** i for i in range(len(quarters))]
        total_w = sum(weights) or 1.0  # guard against zero

        # Weighted average of each dimension signal
        contrib: dict[str, float] = {k: 0.0 for k in SIGNAL_WEIGHTS}
        for qs, w in zip(quarters, weights):
            for dim, val in (qs.normalized or {}).items():
                if dim in contrib:
                    contrib[dim] += val * w / total_w

        # Inject news signal
        if news:
            news_norm = _normalize_news(news)
            contrib["news_sentiment"] = news_norm.get("news_sentiment", 0.0)

        # Weighted sum
        wsum = sum(contrib.get(k, 0.0) * wt for k, wt in SIGNAL_WEIGHTS.items())

        # Apply peer adjustment (capped at ±0.20)
        peer_adj = RevisionScorer._peer_adjustment(peer_signals or [])
        wsum_adjusted = wsum + peer_adj

        prob = round(_sigmoid(wsum_adjusted), 4)

        direction = "neutral" if abs(wsum_adjusted) < 0.05 else ("upward" if wsum_adjusted > 0 else "downward")

        abs_score = abs(wsum_adjusted)
        if abs_score < MAGNITUDE_THRESHOLDS["negligible"]:
            magnitude = "negligible"
        elif abs_score < MAGNITUDE_THRESHOLDS["small"]:
            magnitude = "small"
        elif abs_score < MAGNITUDE_THRESHOLDS["medium"]:
            magnitude = "medium"
        else:
            magnitude = "large"

        conf_values = [qs.confidence for qs in quarters]
        confidence = round(sum(conf_values) / len(conf_values), 4) if conf_values else 0.4

        # Store peer adjustment in contributions for waterfall display
        contrib["peer_signals"] = round(peer_adj, 4)

        return prob, direction, magnitude, confidence, contrib

    @staticmethod
    def compute_trend(quarters: list[QuarterSignals]) -> str:
        """Compare most recent quarter score to prior average."""
        if len(quarters) < 2:
            return "stable"
        recent = quarters[0].weighted_score
        prior_avg = sum(q.weighted_score for q in quarters[1:]) / (len(quarters) - 1)
        delta = recent - prior_avg
        if delta > 0.10:
            return "deteriorating" if recent < 0 else "improving"
        if delta < -0.10:
            return "improving" if recent > 0 else "deteriorating"
        return "stable"

    @staticmethod
    def _apply_guidance_delta(quarters: list[QuarterSignals]) -> None:
        """
        Replace each quarter's guidance_quantification signal with a delta
        (current rate − prior rate) rather than the absolute rate.

        The list is ordered newest-first. For each quarter we compare to the
        next element (the prior quarter). The oldest quarter keeps its absolute
        signal (no prior available). Mutates `normalized` in-place.
        """
        for i, qs in enumerate(quarters):
            if qs.transcript is None:
                continue
            current_rate = qs.transcript.guidance_quantification_rate
            if i + 1 < len(quarters) and quarters[i + 1].transcript is not None:
                prior_rate = quarters[i + 1].transcript.guidance_quantification_rate
                # Delta in [−1, +1]; clamp to same range as absolute signal
                delta = max(-1.0, min(1.0, (current_rate - prior_rate) * 2))
            else:
                # No prior quarter: fall back to absolute (centred at 0.5)
                delta = (current_rate - 0.5) * 2
            qs.normalized["guidance_quantification"] = round(delta, 4)
            # Recompute weighted score with updated dimension
            from config import SIGNAL_WEIGHTS
            wsum = sum(qs.normalized.get(k, 0.0) * w for k, w in SIGNAL_WEIGHTS.items())
            qs.weighted_score = round(wsum, 4)
            qs.revision_probability = round(_sigmoid(wsum), 4)

    @classmethod
    def build_full_analysis(
        cls,
        ticker: str,
        company_name: str,
        quarter_data: list[dict],  # [{quarter, transcript_signals, mda_signals}]
        news: Optional[NewsSignals] = None,
        narrative: Optional[NarrativeOutput] = None,
        peer_signals: Optional[list[dict]] = None,
    ) -> FullAnalysis:
        """Entry point: build a complete FullAnalysis from raw signal objects."""
        scored_quarters = []
        for qd in quarter_data:
            qs = cls.score_quarter(
                quarter=qd.get("quarter", ""),
                transcript=qd.get("transcript"),
                mda=qd.get("mda"),
                mda_is_delta=qd.get("mda_is_delta", False),
            )
            scored_quarters.append(qs)

        # Fix #4: recompute guidance_quantification as QoQ delta
        cls._apply_guidance_delta(scored_quarters)

        prob, direction, magnitude, confidence, contrib = cls.aggregate(
            scored_quarters, news, peer_signals=peer_signals or []
        )
        trend = cls.compute_trend(scored_quarters)

        return FullAnalysis(
            ticker=ticker,
            company_name=company_name,
            quarters=scored_quarters,
            news_signals=news,
            narrative=narrative,
            overall_probability=prob,
            overall_direction=direction,
            overall_magnitude=magnitude,
            overall_confidence=confidence,
            signal_contributions=contrib,
            trend_direction=trend,
        )


# ─────────────────────────── Formatting helpers ──────────────────────────────

def magnitude_display(magnitude: str) -> tuple[str, str]:
    """Return (label, dot_string) for a magnitude value."""
    return MAGNITUDE_LABELS.get(magnitude, ("Unknown", ""))


def probability_to_label(prob: float, direction: str) -> str:
    """Human-readable label for the revision probability."""
    if direction == "neutral":
        return "No Revision Expected"
    pct = int(round(prob * 100))
    if direction == "downward":
        pct = int(round((1 - prob) * 100))  # flip: low prob means high downward likelihood
    label_map = [(80, "Very High"), (60, "High"), (40, "Moderate"), (20, "Low"), (0, "Very Low")]
    label = next(l for threshold, l in label_map if pct >= threshold)
    arrow = "↑" if direction == "upward" else "↓"
    return f"{arrow} {label} Probability of {direction.title()} Revision"
