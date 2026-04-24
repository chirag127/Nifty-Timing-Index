"""Blog Generator — Uses LangGraph fusion workflow to generate hourly blog posts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from nti.llm.langgraph_workflows.fusion_blog import generate_blog
from nti.llm.news_client import news_client
from nti.llm.news_analyzer import analyze_news
from nti.llm.search_client import search_client
from nti.llm.prompts import BLOG_PROMPT_TEMPLATE, BLOG_SYSTEM_PROMPT
from nti.config.settings import settings
from nti.config.thresholds import get_zone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def generate_hourly_blog(
    nti_score: float,
    prev_score: float,
    confidence: float,
    indicators: dict,
    top_drivers: list[dict],
    top_stocks: list[dict],
    changelog_text: str,
    blog_type: str = "mid_session",
) -> dict:
    """Generate a full hourly blog post using the LangGraph fusion workflow.

    This is the main entry point called by the hourly pipeline.

    Args:
        nti_score: Current NTI danger score (0–100)
        prev_score: Previous hour's NTI score
        confidence: Model confidence percentage
        indicators: Dict of all current indicator values
        top_drivers: List of {indicator, label, shap, direction} dicts
        top_stocks: List of top stock picks from screener
        changelog_text: Pre-generated changelog markdown
        blog_type: "market_open", "mid_session", "market_close", "post_market", "overnight"

    Returns:
        dict with keys: blog_markdown, providers_used, errors, duration_seconds
    """
    if not settings.enable_blog or not settings.enable_llm:
        return {
            "blog_markdown": "",
            "providers_used": [],
            "errors": ["Blog generation disabled"],
            "duration_seconds": 0,
        }

    now = datetime.now(IST)
    zone = get_zone(nti_score)
    prev_zone = get_zone(prev_score)

    # Determine word target
    word_target = settings.blog_word_target_mid
    if blog_type in ("market_open", "market_close"):
        word_target = settings.blog_word_target_full

    # Format top drivers
    drivers_text = ", ".join(
        f"{d.get('label', d.get('indicator', ''))} ({d.get('direction', '')} signal)"
        for d in top_drivers[:3]
    ) or "N/A"

    # Format top stocks
    stocks_text = "\n".join(
        f"- {s.get('symbol', '')} ({s.get('name', '')}): PE={s.get('pe', 'N/A')}, "
        f"PB={s.get('pb', 'N/A')}, ROE={s.get('roe_pct', 'N/A')}%, "
        f"Score={s.get('composite_score', 'N/A')}"
        f"{' [★ PSU]' if s.get('is_psu') else ''}"
        for s in top_stocks[:5]
    ) or "No screener data available for this run"

    # Fetch news and search context
    news_articles = news_client.fetch_news(
        query="Indian stock market NSE",
        hours=settings.blog_news_hours,
    )
    news_text = "\n".join(
        f"- [{a.source}] {a.title}"
        for a in news_articles[:20]
    ) or "No news headlines available"

    search_context = ""
    if settings.enable_search:
        results = search_client.search("Indian stock market Nifty analysis today")
        if results:
            search_context = "\n".join(
                f"- [{r.source}] {r.title}: {r.snippet}"
                for r in results[:10]
            )

    # Build the prompt
    prompt = BLOG_PROMPT_TEMPLATE.format(
        blog_type=blog_type,
        timestamp=now.strftime("%Y-%m-%d %H:%M"),
        nti_score=nti_score,
        zone=zone,
        prev_score=prev_score,
        prev_zone=prev_zone,
        confidence=confidence,
        nifty_price=indicators.get("nifty_price", "N/A"),
        nifty_change_pct=indicators.get("nifty_change_pct", 0),
        vix=indicators.get("india_vix", "N/A"),
        mmi=indicators.get("mmi", "N/A"),
        mmi_zone=indicators.get("mmi_zone", "N/A"),
        gift_nifty_price=indicators.get("gift_nifty_price", "N/A"),
        gift_nifty_signal=indicators.get("gift_nifty_signal", "N/A"),
        pcr=indicators.get("pcr", "N/A"),
        fii_net=indicators.get("fii_cash_net") or 0,
        fii_direction="selling" if (indicators.get("fii_cash_net") or 0) < 0 else "buying",
        nifty_pe=indicators.get("nifty_pe", "N/A"),
        nifty_pb=indicators.get("nifty_pb", "N/A"),
        div_yield=indicators.get("nifty_dy", "N/A"),
        us_10y=indicators.get("us_10y_yield", "N/A"),
        usdinr=indicators.get("usd_inr", "N/A"),
        crude=indicators.get("brent_crude", "N/A"),
        top_drivers_text=drivers_text,
        top_stocks_formatted=stocks_text,
        news_headlines=news_text,
        search_context=search_context,
        changelog_text=changelog_text,
        word_target=word_target,
    )

    # Generate blog using LangGraph fusion workflow
    result = generate_blog(
        prompt=prompt,
        system_prompt=BLOG_SYSTEM_PROMPT,
        news_headlines=news_text,
    )

    blog_markdown = result.get("final_blog", "")

    # Add frontmatter for Astro content collection
    slug = now.strftime("%Y-%m-%d-%H-%M")
    frontmatter = f"""---
title: "NTI Update: Score {nti_score:.0f} | {zone} Zone | Nifty at {indicators.get('nifty_price', 'N/A')} | {now.strftime('%Y-%m-%d %H:%M')} IST"
description: "Nifty Timing Index hourly update: NTI score {nti_score:.0f} ({zone}). {'Zone changed from ' + prev_zone + '!' if zone != prev_zone else 'Score within same zone.'}"
slug: "{slug}"
publishedAt: "{now.isoformat()}"
ntiScore: {nti_score}
ntiZone: "{zone}"
ntiZonePrev: "{prev_zone}"
zoneChanged: {str(zone != prev_zone).lower()}
confidence: {confidence}
nifty50Price: {indicators.get('nifty_price', 0)}
niftyBank: {indicators.get('nifty_bank_price', 0)}
sensex: {indicators.get('sensex_price', 0)}
topDrivers: {list(d.get('indicator', '') for d in top_drivers[:3])}
topStocks: {list(s.get('symbol', '') for s in top_stocks[:5])}
blogType: "{blog_type}"
---

"""

    return {
        "blog_markdown": frontmatter + blog_markdown,
        "slug": slug,
        "providers_used": result.get("providers_used", []),
        "errors": result.get("errors", []),
        "duration_seconds": result.get("duration_seconds", 0),
    }
