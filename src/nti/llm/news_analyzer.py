"""LLM News Analyzer — Uses LangGraph fusion to analyze news and produce danger scores."""

from __future__ import annotations

import json
import logging

from nti.llm.langgraph_workflows.fusion_blog import generate_blog, run_sequential_generation
from nti.llm.news_client import news_client
from nti.llm.prompts import NEWS_ANALYSIS_PROMPT
from nti.llm.search_client import search_client
from nti.config.settings import settings

logger = logging.getLogger(__name__)


def analyze_news(query: str = "Indian stock market NSE BSE") -> dict:
    """Analyze recent news using LLM to produce danger scores.

    Uses the fusion workflow to get multiple LLM perspectives on the news,
    then parses the results into structured data.

    Returns:
        dict with keys: danger_score, reasoning, key_events, sector_impacts,
        policy_flag, geopolitical_risk, news_sentiment
    """
    if not settings.enable_news or not settings.enable_llm:
        return _default_news_analysis()

    # Fetch news from all enabled providers
    articles = news_client.fetch_news(query=query, hours=settings.blog_news_hours)

    if not articles:
        logger.warning("No news articles found")
        return _default_news_analysis()

    # Format headlines for LLM
    headlines = "\n".join(
        f"- [{a.source}] {a.title}: {a.summary}"
        for a in articles[:30]
    )

    # Enrich with search results if enabled
    search_context = ""
    if settings.enable_search:
        search_results = search_client.search(f"{query} latest news today")
        if search_results:
            search_context = "\n\nWeb Search Results:\n" + "\n".join(
                f"- [{r.source}] {r.title}: {r.snippet}"
                for r in search_results[:10]
            )

    # Generate the analysis prompt
    prompt = NEWS_ANALYSIS_PROMPT.format(headlines=headlines + search_context)

    # Use sequential generation (simpler, cheaper for news analysis)
    result = run_sequential_generation(
        prompt=prompt,
        system_prompt="You are a financial news analyst. Respond ONLY in valid JSON format.",
    )

    content = result.get("final_blog", "")
    if not content:
        return _default_news_analysis()

    # Parse JSON response
    return _parse_news_analysis(content)


def _parse_news_analysis(content: str) -> dict:
    """Parse LLM news analysis response into structured data."""
    default = _default_news_analysis()

    try:
        # Try to extract JSON from the response
        # Sometimes LLMs add markdown code blocks
        json_str = content.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        data = json.loads(json_str.strip())

        return {
            "danger_score": min(100, max(0, float(data.get("danger_score", 50)))),
            "reasoning": str(data.get("reasoning", "")),
            "key_events": data.get("key_events", [])[:3],
            "sector_impacts": data.get("sector_impacts", {}),
            "policy_flag": str(data.get("policy_flag", "none")),
            "geopolitical_risk": min(100, max(0, float(data.get("geopolitical_risk", 0)))),
            "news_sentiment": str(data.get("news_sentiment", "neutral")),
        }

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Could not parse news analysis JSON: {e}")
        # Try to extract danger_score from text
        return default


def _default_news_analysis() -> dict:
    """Default news analysis when LLM is unavailable."""
    return {
        "danger_score": 50,
        "reasoning": "News analysis unavailable — using neutral default",
        "key_events": [],
        "sector_impacts": {},
        "policy_flag": "none",
        "geopolitical_risk": 50,
        "news_sentiment": "neutral",
    }
