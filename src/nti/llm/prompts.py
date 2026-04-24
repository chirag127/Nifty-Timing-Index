"""LLM Prompt Templates — Blog generation and news analysis."""

from __future__ import annotations

BLOG_SYSTEM_PROMPT = """You are NTI-Writer, the AI analyst for the Nifty Timing Index (NTI).
You write market analysis blog posts for Indian equity investors who follow
a long-only, value investing approach.

RULES:
- NEVER recommend shorting, options selling, or leveraged bearish strategies
- ONLY recommend stocks with PE < 20, PB < 3, and market cap ≥ ₹500 Cr
- Highlight PSU/government stocks as preferred (soft preference)
- Be specific about numbers — no vague claims
- Include the SHAP drivers to explain WHY the NTI score is at its level
- Always include the DISCLAIMER at the end
- Output ONLY the blog post markdown content, no explanations
"""

BLOG_PROMPT_TEMPLATE = """
Write a {blog_type} market update blog post for Indian equity investors.

TODAY'S DATA ({timestamp} IST):
- NTI Danger Score: {nti_score}/100 (Zone: {zone})
- Previous Hour Score: {prev_score}/100 (Zone: {prev_zone})
- Confidence: {confidence}%
- Nifty 50: {nifty_price} ({nifty_change_pct:+.2f}%)
- India VIX: {vix}
- Tickertape MMI: {mmi} ({mmi_zone})
- GIFT Nifty: {gift_nifty_price} ({gift_nifty_signal})
- Put/Call Ratio: {pcr}
- FII Net (today): ₹{fii_net:,.0f} Cr ({fii_direction})
- Nifty P/E: {nifty_pe}x | P/B: {nifty_pb}x | Div Yield: {div_yield}%
- US 10Y Yield: {us_10y}% | USD/INR: {usdinr} | Brent Crude: ${crude}/bbl
- Top SHAP Drivers: {top_drivers_text}

TOP VALUE PICKS (PE < 20, PB < 3, ≥ ₹500 Cr):
{top_stocks_formatted}

RECENT NEWS (from RSS + search APIs):
{news_headlines}

SEARCH CONTEXT (from web search):
{search_context}

CHANGELOG (vs previous hour):
{changelog_text}

INSTRUCTIONS:
- Word target: {word_target} words
- Write in English with Indian market context
- Be specific about numbers, don't be vague
- Explain WHY the NTI score is at this level using the SHAP drivers
- For each top stock pick, explain PE, PB, ROE and why it's a value pick
- Mention PSU stocks specifically (government backing is a positive)
- Include the changelog section clearly
- Add a DISCLAIMER at the end: "This is not investment advice. NTI is a personal market analysis tool."
- Do NOT recommend shorting, options selling, or leveraged bearish strategies
- Output ONLY the blog post markdown content, no explanations
"""

NEWS_ANALYSIS_PROMPT = """
Analyze these recent Indian stock market headlines and provide:
1. Overall danger score (0=buy opportunity, 100=sell danger)
2. Top 3 market-moving events
3. Sector impacts (banking, IT, energy, pharma, metals)
4. Policy/regulatory flags (RBI, SEBI, budget, government announcements)
5. Geopolitical risks affecting Indian equities

Headlines:
{headlines}

Respond ONLY in JSON format:
{{
  "danger_score": <0-100>,
  "reasoning": "<2-sentence explanation>",
  "key_events": ["event1", "event2", "event3"],
  "sector_impacts": {{"banking": "positive/neutral/negative", "it": "...", "energy": "...", "pharma": "...", "metals": "..."}},
  "policy_flag": "<description or 'none'>",
  "geopolitical_risk": <0-100>,
  "news_sentiment": "bullish/neutral/bearish"
}}
"""
