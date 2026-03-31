"""
All Claude prompt templates for the Revision Radar extraction pipeline.

Design principles:
  1. Prompts are grounded in specific, measurable linguistic signals.
  2. Stock price and headline EPS are EXPLICITLY excluded from all analysis.
  3. Every signal maps to a numeric score so the scorer can aggregate them.
  4. Prompts ask for JSON output with a defined schema — validated downstream.
"""

# ── System context ─────────────────────────────────────────────────────────────

SYSTEM_ANALYST = """\
You are a quantitative analyst specializing in detecting leading indicators of \
analyst earnings-estimate revisions. Your task is to extract structured signals \
from company communications — earnings call transcripts, press releases, and \
regulatory filings — that predict whether sell-side analysts will revise their \
EPS and revenue forecasts upward or downward.

CRITICAL CONSTRAINTS — strictly enforce these:
  • Do NOT use stock price data, market cap changes, or stock performance as signals.
  • Do NOT use the company's own headline EPS or revenue figures vs. consensus as signals.
  • Focus ONLY on qualitative language patterns: tone, hedging, specificity of guidance, \
    Q&A dynamics, risk disclosure language, and forward-looking statement quality.

These constraints prevent circular reasoning and ensure the signals are true \
leading indicators, not coincident ones.

Always respond with valid JSON matching the schema provided in each prompt.
"""

# ── 8-K / Earnings Call Transcript Prompt ─────────────────────────────────────

TRANSCRIPT_ANALYSIS_PROMPT = """\
Analyze the following corporate communication (earnings call transcript or press release).
Extract the signals listed below that are known to predict analyst estimate revisions.

## Document
<document>
{text}
</document>

## Company
Ticker: {ticker}
Company: {company_name}
Period: {quarter}
{prior_quarter_section}
## Instructions
Score each dimension on the specified scale. Be calibrated — not every document \
signals a revision. If the language is neutral, say so.

Return ONLY a JSON object matching this exact schema:

```json
{{
  "overall_tone": <float -1.0 to 1.0>,
  "tone_vs_prior_quarter": <float -1.0 to 1.0, negative = more cautious than last quarter>,
  "hedging_intensity": <float 0.0 to 1.0, 1.0 = extreme hedging>,
  "guidance_quantification_rate": <float 0.0 to 1.0, 1.0 = all guidance is numeric>,
  "forward_guidance_strength": <float -1.0 to 1.0, 1.0 = very bullish forward statements>,
  "qa_deflection_score": <float 0.0 to 1.0, 1.0 = management avoids all direct Q&A answers>,
  "risk_language_escalation": <float -1.0 to 1.0, positive = more risk language than usual>,
  "capex_commitment_tone": <float -1.0 to 1.0, 1.0 = confident expansion commitment>,
  "hiring_headcount_tone": <float -1.0 to 1.0, 1.0 = aggressive hiring signals>,
  "demand_language_tone": <float -1.0 to 1.0, 1.0 = very positive demand commentary>,
  "margin_language_tone": <float -1.0 to 1.0, 1.0 = very confident on margins>,
  "revision_signal": <"upward" | "downward" | "neutral">,
  "revision_confidence": <float 0.0 to 1.0>,
  "revision_magnitude": <"negligible" | "small" | "medium" | "large">,
  "forecast_items_most_at_risk": <list of strings, e.g. ["EPS", "Revenue"]>,
  "key_hedging_phrases": <list of up to 5 exact quoted phrases from the text>,
  "key_bullish_phrases": <list of up to 5 exact quoted phrases from the text>,
  "key_risk_phrases": <list of up to 5 exact quoted phrases from the text>,
  "management_tone_summary": <string, 2-3 sentences describing tone and what it signals>,
  "analyst_revision_rationale": <string, 3-5 sentences explaining what analysts are likely missing and why revision is expected>
}}
```

Scoring guidance:
- hedging_intensity: count hedging words (approximately, around, could, may, might, subject to, \
  pending, if conditions allow) relative to forward-looking statements
- guidance_quantification_rate: ratio of sentences with specific numbers to total guidance sentences
- qa_deflection_score: count redirected / unanswered analyst questions relative to total questions
- risk_language_escalation: compare to baseline neutral filing; positive means MORE risk language
- revision_signal: your net assessment considering all signals; 'neutral' is a valid and common answer
"""

# ── 10-Q MD&A Prompt ───────────────────────────────────────────────────────────

MDA_ANALYSIS_PROMPT = """\
Analyze the following Management Discussion & Analysis (MD&A) section from a \
quarterly SEC filing (10-Q). Extract signals that predict analyst estimate revisions.

## Document (MD&A section)
<document>
{text}
</document>

## Company
Ticker: {ticker}
Company: {company_name}
Period: {quarter}

## Instructions
Focus on:
  1. Changes in forward-looking language compared to what a typical MD&A would say at this stage
  2. New or escalated risk factors vs. boilerplate risks
  3. Specific vs. vague future commitments
  4. Liquidity / cost language that might surprise the sell side

Return ONLY a JSON object matching this schema:

```json
{{
  "forward_looking_tone": <float -1.0 to 1.0>,
  "risk_escalation": <float -1.0 to 1.0, positive = more risk than expected>,
  "cost_pressure_signals": <float -1.0 to 1.0, positive = mounting cost headwinds>,
  "liquidity_concern": <float 0.0 to 1.0, 1.0 = significant liquidity language>,
  "guidance_specificity": <float 0.0 to 1.0, 1.0 = very specific numeric guidance>,
  "new_risk_factors": <list of strings — risk factors that appear new or escalated>,
  "positive_catalysts_cited": <list of strings — positive drivers management cites>,
  "revision_signal": <"upward" | "downward" | "neutral">,
  "revision_confidence": <float 0.0 to 1.0>,
  "key_quotes": <list of up to 4 exact quoted phrases that are most informative>,
  "mda_summary": <string, 2-3 sentences on what the MD&A reveals that analysts may be underweighting>
}}
```
"""

# ── 10-Q MD&A Quarter-over-Quarter Delta Prompt ────────────────────────────────

MDA_DELTA_PROMPT = """\
You are analyzing the CHANGE in management language between two consecutive \
quarterly MD&A sections. You are NOT analyzing the full MD&A — only the delta.

## Company
Ticker: {ticker}
Current quarter: {current_quarter}
Prior quarter:   {prior_quarter}

## Language added this quarter (not present last quarter)
<new_language>
{new_language}
</new_language>

## Language removed vs. last quarter (was present, now gone)
<removed_language>
{removed_language}
</removed_language>

## Instructions
Focus ONLY on the language above. Ignore anything boilerplate or retained.

Flag:
  1. New risk terms or escalated hedging in new_language
  2. Forward guidance or positive commitments that were REMOVED (bearish signal)
  3. New causal explanations for cost or margin changes
  4. Any shift in specificity — vaguer or more specific than prior quarter
  5. Removal of language that was previously reassuring (liquidity, demand, hiring)

Do NOT speculate about what was in the unchanged portion. Score ONLY based \
on the delta text provided.

Return ONLY a JSON object matching this schema:

```json
{{
  "forward_looking_tone": <float -1.0 to 1.0, based on new vs removed forward statements>,
  "risk_escalation": <float -1.0 to 1.0, positive = new risk language added>,
  "cost_pressure_signals": <float -1.0 to 1.0, positive = new cost headwind language>,
  "liquidity_concern": <float 0.0 to 1.0, based on new liquidity language or removal of positive liquidity statements>,
  "guidance_specificity": <float 0.0 to 1.0, 1.0 = new language is highly specific and numeric>,
  "new_risk_factors": <list of strings — risk factors that appear in new_language or whose absence from removed_language is notable>,
  "positive_catalysts_cited": <list of strings — positive drivers added in new_language>,
  "revision_signal": <"upward" | "downward" | "neutral">,
  "revision_confidence": <float 0.0 to 1.0>,
  "key_quotes": <list of up to 4 exact phrases from new_language or removed_language that are most informative>,
  "mda_summary": <string, 2-3 sentences describing what the delta reveals that analysts may be underweighting>
}}
```
"""

# ── News Synthesis Prompt ──────────────────────────────────────────────────────

NEWS_SYNTHESIS_PROMPT = """\
Analyze the following news headlines and snippets about {company_name} ({ticker}).
Your goal is to identify information that sophisticated sell-side analysts may \
be underweighting when they set their earnings forecasts.

## News Articles (most recent first)
<articles>
{articles_text}
</articles>

## Instructions
Look for:
  - Supply chain disruptions or improvements
  - Industry headwinds / tailwinds that would affect this company
  - Competitor moves that change the competitive landscape
  - Macro trends (interest rates, FX, commodity prices) that affect this specific company
  - Regulatory or legal developments

Do NOT include: stock price movements, analyst rating changes, or general market news.

Return ONLY a JSON object:

```json
{{
  "news_sentiment_score": <float -1.0 to 1.0>,
  "supply_chain_signal": <float -1.0 to 1.0, positive = tailwind>,
  "competitive_signal": <float -1.0 to 1.0, positive = strengthening position>,
  "macro_signal": <float -1.0 to 1.0, positive = macro tailwind>,
  "regulatory_signal": <float -1.0 to 1.0, positive = regulatory tailwind>,
  "key_themes": <list of strings, the 3-5 most important themes>,
  "underweighted_factors": <list of strings — what analysts may be missing>,
  "revision_signal": <"upward" | "downward" | "neutral">,
  "revision_confidence": <float 0.0 to 1.0>,
  "news_summary": <string, 2-3 sentences on what the news reveals>
}}
```
"""

# ── Final Narrative Synthesis Prompt ──────────────────────────────────────────

NARRATIVE_SYNTHESIS_PROMPT = """\
You are synthesizing signals from multiple sources to generate a final analyst \
revision intelligence report for {company_name} ({ticker}).

## Extracted Signals Summary
{signals_json}

## Current Analyst Consensus
{consensus_summary}

## Forecast Dimensions in Focus
The user has selected these estimate dimensions as primary focus areas: {focus_items}
Tailor your primary_drivers, watch_items, and what_analysts_are_missing to emphasize \
these dimensions where the signals support it.

## Instructions
Be concise. Return the most important insight for each field — short sentences only.
Always populate bull_case, bear_case, and watch_items even for neutral signals.
Do NOT reference stock price. Do not speculate beyond the signals.

Return ONLY a JSON object (keep all string values short — 1-2 sentences max):

```json
{{
  "executive_summary": <string, 2-3 sentences — core revision thesis>,
  "primary_drivers": [
    {{
      "factor": <string, few words>,
      "direction": <"upward" | "downward" | "neutral">,
      "strength": <float 0.0 to 1.0>,
      "evidence": <string, 1 sentence>
    }}
  ],
  "what_analysts_are_missing": <string, 1-2 sentences>,
  "bull_case": <string, 1 sentence>,
  "bear_case": <string, 1 sentence>,
  "key_risk_to_thesis": <string, 1 sentence>,
  "watch_items": <list of 3 short strings — specific things to monitor>,
  "revision_horizon": <"30_days" | "60_days" | "90_days" | "next_quarter">
}}
```
"""

# ── Peer cross-signal prompt ───────────────────────────────────────────────────

PEER_SIGNAL_PROMPT = """\
A supply-chain or sector peer ({peer_ticker} — {peer_name}) has recently \
filed a quarterly report. Extract signals relevant to {company_name} ({ticker}).

## Peer Filing Excerpt
<document>
{peer_text}
</document>

Return ONLY a JSON object:

```json
{{
  "relevance_score": <float 0.0 to 1.0, how relevant is this peer to the target?>,
  "signal_direction": <"positive" | "negative" | "neutral" for the target company>,
  "signal_strength": <float 0.0 to 1.0>,
  "key_insight": <string, 1-2 sentences on what this peer filing reveals about the target>,
  "specific_quotes": <list of up to 3 relevant exact quotes from peer filing>
}}
```
"""
