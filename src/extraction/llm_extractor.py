"""
LLM extraction layer — sends documents to Claude and parses structured JSON responses.

Features:
  - Disk-based caching keyed on (text hash, prompt version) to avoid re-billing
  - Automatic chunking for documents that exceed Claude's effective context
  - Retry logic with exponential backoff
  - Pydantic validation of all outputs
  - Graceful degradation: if extraction fails, returns zeroed-out neutral signals
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ANTHROPIC_API_KEY, CACHE_DIR, CACHE_TTL_HOURS
from src.extraction.prompts import (
    SYSTEM_ANALYST,
    TRANSCRIPT_ANALYSIS_PROMPT,
    MDA_ANALYSIS_PROMPT,
    MDA_DELTA_PROMPT,
    NEWS_SYNTHESIS_PROMPT,
    NARRATIVE_SYNTHESIS_PROMPT,
    PEER_SIGNAL_PROMPT,
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_VERSION = "v1.3"   # Bump to invalidate all caches on prompt change
MAX_CHUNK_CHARS = 12_000  # Claude context budget per call

# ── Pydantic signal models ────────────────────────────────────────────────────

class TranscriptSignals(BaseModel):
    overall_tone:                float = Field(0.0, ge=-1, le=1)
    tone_vs_prior_quarter:       float = Field(0.0, ge=-1, le=1)
    hedging_intensity:           float = Field(0.0, ge=0, le=1)
    guidance_quantification_rate: float = Field(0.5, ge=0, le=1)
    forward_guidance_strength:   float = Field(0.0, ge=-1, le=1)
    qa_deflection_score:         float = Field(0.0, ge=0, le=1)
    risk_language_escalation:    float = Field(0.0, ge=-1, le=1)
    capex_commitment_tone:       float = Field(0.0, ge=-1, le=1)
    hiring_headcount_tone:       float = Field(0.0, ge=-1, le=1)
    demand_language_tone:        float = Field(0.0, ge=-1, le=1)
    margin_language_tone:        float = Field(0.0, ge=-1, le=1)
    revision_signal:             str   = Field("neutral")
    revision_confidence:         float = Field(0.0, ge=0, le=1)
    revision_magnitude:          str   = Field("negligible")
    forecast_items_most_at_risk: list[str] = Field(default_factory=list)
    key_hedging_phrases:         list[str] = Field(default_factory=list)
    key_bullish_phrases:         list[str] = Field(default_factory=list)
    key_risk_phrases:            list[str] = Field(default_factory=list)
    management_tone_summary:     str  = Field("")
    analyst_revision_rationale:  str  = Field("")

    @field_validator("revision_signal")
    @classmethod
    def validate_signal(cls, v):
        return v if v in ("upward", "downward", "neutral") else "neutral"

    @field_validator("revision_magnitude")
    @classmethod
    def validate_magnitude(cls, v):
        return v if v in ("negligible", "small", "medium", "large") else "negligible"


class MDASignals(BaseModel):
    forward_looking_tone:      float = Field(0.0, ge=-1, le=1)
    risk_escalation:           float = Field(0.0, ge=-1, le=1)
    cost_pressure_signals:     float = Field(0.0, ge=-1, le=1)
    liquidity_concern:         float = Field(0.0, ge=0, le=1)
    guidance_specificity:      float = Field(0.5, ge=0, le=1)
    new_risk_factors:          list[str] = Field(default_factory=list)
    positive_catalysts_cited:  list[str] = Field(default_factory=list)
    revision_signal:           str   = Field("neutral")
    revision_confidence:       float = Field(0.0, ge=0, le=1)
    key_quotes:                list[str] = Field(default_factory=list)
    mda_summary:               str   = Field("")

    @field_validator("revision_signal")
    @classmethod
    def validate_signal(cls, v):
        return v if v in ("upward", "downward", "neutral") else "neutral"


class NewsSignals(BaseModel):
    news_sentiment_score:  float = Field(0.0, ge=-1, le=1)
    supply_chain_signal:   float = Field(0.0, ge=-1, le=1)
    competitive_signal:    float = Field(0.0, ge=-1, le=1)
    macro_signal:          float = Field(0.0, ge=-1, le=1)
    regulatory_signal:     float = Field(0.0, ge=-1, le=1)
    key_themes:            list[str] = Field(default_factory=list)
    underweighted_factors: list[str] = Field(default_factory=list)
    revision_signal:       str   = Field("neutral")
    revision_confidence:   float = Field(0.0, ge=0, le=1)
    news_summary:          str   = Field("")

    @field_validator("revision_signal")
    @classmethod
    def validate_signal(cls, v):
        return v if v in ("upward", "downward", "neutral") else "neutral"


class NarrativeOutput(BaseModel):
    executive_summary:      str  = Field("")
    primary_drivers:        list[dict] = Field(default_factory=list)
    what_analysts_are_missing: str = Field("")
    bull_case:              str  = Field("")
    bear_case:              str  = Field("")
    key_risk_to_thesis:     str  = Field("")
    watch_items:            list[str] = Field(default_factory=list)
    revision_horizon:       str  = Field("next_quarter")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(text: str, prompt_type: str) -> str:
    h = hashlib.sha256(f"{PROMPT_VERSION}:{prompt_type}:{text[:500]}".encode()).hexdigest()[:24]
    return f"llm_{prompt_type}_{h}"


def _load_llm_cache(key: str) -> Optional[dict]:
    from datetime import datetime, timedelta
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    if datetime.fromisoformat(raw["ts"]) < datetime.now() - timedelta(hours=CACHE_TTL_HOURS):
        return None
    return raw["payload"]


def _save_llm_cache(key: str, payload: dict) -> None:
    from datetime import datetime
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps({"ts": datetime.now().isoformat(), "payload": payload}))


# ── Token / cost tracking ─────────────────────────────────────────────────────
# claude-sonnet-4-6 pricing (per 1M tokens)
_PRICE_INPUT  = 3.00
_PRICE_OUTPUT = 15.00

_usage: dict = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cache_hits": 0}

def reset_usage() -> None:
    """Reset counters at the start of each analysis run."""
    _usage.update({"input_tokens": 0, "output_tokens": 0, "calls": 0, "cache_hits": 0})

def get_usage() -> dict:
    """Return current token usage and estimated USD cost."""
    inp = _usage["input_tokens"]
    out = _usage["output_tokens"]
    cost = (inp * _PRICE_INPUT + out * _PRICE_OUTPUT) / 1_000_000
    return {
        "input_tokens":  inp,
        "output_tokens": out,
        "total_tokens":  inp + out,
        "calls":         _usage["calls"],
        "cache_hits":    _usage["cache_hits"],
        "cost_usd":      round(cost, 4),
    }


# ── Claude API client ────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file and restart the app."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from Claude's response text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Find JSON block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Find raw JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def _call_claude(system: str, user: str, model: str = "claude-sonnet-4-6",
                 max_tokens: int = 2048) -> str:
    client = _get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    _usage["input_tokens"]  += msg.usage.input_tokens
    _usage["output_tokens"] += msg.usage.output_tokens
    _usage["calls"]         += 1
    return msg.content[0].text


def _truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    # Try to cut at a paragraph boundary
    cutoff = text.rfind("\n\n", 0, max_chars)
    return text[: cutoff if cutoff > 0 else max_chars] + "\n\n[... document truncated for analysis ...]"


# ── Public extraction functions ───────────────────────────────────────────────

class LLMExtractor:
    """Stateless extractor — all methods are effectively static."""

    @staticmethod
    def extract_transcript(
        text: str,
        ticker: str,
        company_name: str,
        quarter: str,
        prior_transcript_text: Optional[str] = None,
    ) -> TranscriptSignals:
        """
        Analyze an earnings call transcript or press release.
        Returns a validated TranscriptSignals object.

        If prior_transcript_text is provided, it is injected into the prompt so
        Claude can make an explicit QoQ tone comparison for tone_vs_prior_quarter.
        """
        if not text or len(text) < 100:
            return TranscriptSignals()

        # Include prior text in cache key so different priors get separate cache entries
        prior_fragment = prior_transcript_text[:500] if prior_transcript_text else ""
        cache_key = _cache_key(text + quarter + prior_fragment, "transcript")
        cached = _load_llm_cache(cache_key)
        if cached:
            try:
                _usage["cache_hits"] += 1
                return TranscriptSignals(**cached)
            except Exception:
                pass

        if prior_transcript_text and len(prior_transcript_text) >= 100:
            prior_section = (
                "\n## Prior Quarter Document (for tone comparison only)\n"
                "<prior_document>\n"
                f"{_truncate(prior_transcript_text, max_chars=4000)}\n"
                "</prior_document>\n"
                "Use the prior quarter document above ONLY to calibrate "
                "tone_vs_prior_quarter — a concrete comparison of management "
                "language and sentiment shift between quarters.\n"
            )
        else:
            prior_section = ""

        prompt = TRANSCRIPT_ANALYSIS_PROMPT.format(
            text=_truncate(text),
            ticker=ticker,
            company_name=company_name,
            quarter=quarter,
            prior_quarter_section=prior_section,
        )
        try:
            raw = _call_claude(SYSTEM_ANALYST, prompt)
            data = _extract_json(raw)
            signals = TranscriptSignals(**data)
            _save_llm_cache(cache_key, signals.model_dump())
            return signals
        except Exception as exc:
            print(f"[LLMExtractor] transcript extraction failed: {exc}")
            return TranscriptSignals()

    @staticmethod
    def extract_mda(
        text: str,
        ticker: str,
        company_name: str,
        quarter: str,
    ) -> MDASignals:
        """Analyze a 10-Q MD&A section."""
        if not text or len(text) < 100:
            return MDASignals()

        cache_key = _cache_key(text + quarter, "mda")
        cached = _load_llm_cache(cache_key)
        if cached:
            try:
                _usage["cache_hits"] += 1
                return MDASignals(**cached)
            except Exception:
                pass

        prompt = MDA_ANALYSIS_PROMPT.format(
            text=_truncate(text),
            ticker=ticker,
            company_name=company_name,
            quarter=quarter,
        )
        try:
            raw = _call_claude(SYSTEM_ANALYST, prompt)
            data = _extract_json(raw)
            signals = MDASignals(**data)
            _save_llm_cache(cache_key, signals.model_dump())
            return signals
        except Exception as exc:
            print(f"[LLMExtractor] MDA extraction failed: {exc}")
            return MDASignals()

    @staticmethod
    def extract_mda_delta(
        diff: dict,
        ticker: str,
        quarter: str,
    ) -> MDASignals:
        """
        Analyze a QoQ MD&A diff produced by sec_client.diff_mda().
        Claude sees only the new and removed language — not the full static text.
        Returns the same MDASignals schema as extract_mda().
        """
        new_lang     = (diff.get("new_language")     or "").strip()
        removed_lang = (diff.get("removed_language") or "").strip()

        if not new_lang and not removed_lang:
            return MDASignals()

        cache_key = _cache_key(
            new_lang[:400] + removed_lang[:400] + quarter, "mda_delta"
        )
        cached = _load_llm_cache(cache_key)
        if cached:
            try:
                _usage["cache_hits"] += 1
                return MDASignals(**cached)
            except Exception:
                pass

        prompt = MDA_DELTA_PROMPT.format(
            ticker=ticker,
            current_quarter=diff.get("current_quarter") or quarter,
            prior_quarter=diff.get("prior_quarter") or "prior quarter",
            new_language=_truncate(new_lang, MAX_CHUNK_CHARS // 2),
            removed_language=_truncate(removed_lang, MAX_CHUNK_CHARS // 2),
        )
        try:
            raw = _call_claude(SYSTEM_ANALYST, prompt)
            data = _extract_json(raw)
            signals = MDASignals(**data)
            _save_llm_cache(cache_key, signals.model_dump())
            return signals
        except Exception as exc:
            print(f"[LLMExtractor] MDA delta extraction failed: {exc}")
            return MDASignals()

    @staticmethod
    def extract_news(
        articles_df,  # pd.DataFrame with title, url, tone columns
        ticker: str,
        company_name: str,
    ) -> NewsSignals:
        """Synthesize news signals from a GDELT article DataFrame."""
        if articles_df is None or articles_df.empty:
            return NewsSignals()

        # Format articles into a condensed text block
        lines = []
        for _, row in articles_df.head(30).iterrows():
            tone_str = f"[tone={row.get('tone', 0):.2f}]" if "tone" in row else ""
            domain = row.get("domain", "")
            lines.append(f"• {row.get('title', '')} {tone_str} ({domain})")
        articles_text = "\n".join(lines)

        cache_key = _cache_key(articles_text, "news")
        cached = _load_llm_cache(cache_key)
        if cached:
            try:
                _usage["cache_hits"] += 1
                return NewsSignals(**cached)
            except Exception:
                pass

        prompt = NEWS_SYNTHESIS_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            articles_text=articles_text,
        )
        try:
            raw = _call_claude(SYSTEM_ANALYST, prompt)
            data = _extract_json(raw)
            signals = NewsSignals(**data)
            _save_llm_cache(cache_key, signals.model_dump())
            return signals
        except Exception as exc:
            print(f"[LLMExtractor] news extraction failed: {exc}")
            return NewsSignals()

    @staticmethod
    def synthesize_narrative(
        all_signals: dict,
        ticker: str,
        company_name: str,
        consensus_summary: str = "",
        focus_items: list[str] | None = None,
    ) -> NarrativeOutput:
        """
        Generate the final human-readable investment narrative
        from the aggregated signal dict.
        """
        signals_json = json.dumps(all_signals, indent=2, default=str)
        cache_key = _cache_key(signals_json, "narrative")
        cached = _load_llm_cache(cache_key)
        if cached:
            try:
                _usage["cache_hits"] += 1
                return NarrativeOutput(**cached)
            except Exception:
                pass

        focus_str = ", ".join(focus_items) if focus_items else "EPS, Revenue"
        prompt = NARRATIVE_SYNTHESIS_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            signals_json=_truncate(signals_json, 6000),
            consensus_summary=consensus_summary or "No consensus data available.",
            focus_items=focus_str,
        )
        try:
            raw = _call_claude(SYSTEM_ANALYST, prompt)
            data = _extract_json(raw)
            narrative = NarrativeOutput(**data)
            _save_llm_cache(cache_key, narrative.model_dump())
            return narrative
        except Exception as exc:
            print(f"[LLMExtractor] narrative synthesis failed: {type(exc).__name__}: {exc}")
            return NarrativeOutput(
                executive_summary=f"Narrative synthesis failed: {type(exc).__name__}: {exc}",
                primary_drivers=[],
                what_analysts_are_missing="",
                bull_case="",
                bear_case="",
                key_risk_to_thesis="",
                watch_items=[],
            )

    @staticmethod
    def extract_peer_signal(
        peer_text: str,
        peer_ticker: str,
        peer_name: str,
        target_ticker: str,
        target_name: str,
    ) -> dict:
        """Extract cross-company supply-chain signal from a peer's filing."""
        if not peer_text or len(peer_text) < 100:
            return {}

        cache_key = _cache_key(peer_text[:1000] + peer_ticker, "peer")
        cached = _load_llm_cache(cache_key)
        if cached:
            _usage["cache_hits"] += 1
            return cached

        prompt = PEER_SIGNAL_PROMPT.format(
            peer_ticker=peer_ticker,
            peer_name=peer_name,
            company_name=target_name,
            ticker=target_ticker,
            peer_text=_truncate(peer_text, 6000),
        )
        try:
            from src.extraction.prompts import PEER_SIGNAL_PROMPT as _  # noqa
            raw = _call_claude(SYSTEM_ANALYST, prompt)
            data = _extract_json(raw)
            _save_llm_cache(cache_key, data)
            return data
        except Exception as exc:
            print(f"[LLMExtractor] peer signal failed: {exc}")
            return {}
