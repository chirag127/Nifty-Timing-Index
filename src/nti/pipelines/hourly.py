"""Hourly Pipeline — Orchestrate the full hourly NTI run.

Steps:
1. Scrape all 30 indicators from various sources
2. Normalize indicators to 0–100 scale
3. Run ML inference (or rule-based fallback)
4. Generate changelog (compare vs previous run)
5. Generate blog post via LangGraph fusion
6. Write data files (CSV, JSON, blog .md)
7. Send alerts on zone change
8. Git commit and push

This is the main entry point called by GitHub Actions every hour.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.settings import settings
from nti.config.thresholds import get_zone
from nti.config.holidays import is_market_holiday

# Scrapers
from nti.scrapers.nse_indices import scrape_nse_index_data
from nti.scrapers.yahoo_finance import scrape_global_markets
from nti.scrapers.fred_api import scrape_us_10y_yield
from nti.scrapers.nse_fii_dii import scrape_fii_dii_cash_flow
from nti.scrapers.nse_options import scrape_put_call_ratio
from nti.scrapers.tickertape_mmi import scrape_mmi_selenium
from nti.scrapers.rbi_data import scrape_rbi_repo_rate
from nti.scrapers.mospi_data import scrape_cpi_inflation
from nti.scrapers.amfi_data import scrape_amfi_sip_flows
from nti.scrapers.cnn_fear_greed import scrape_cnn_fear_greed
from nti.scrapers.gift_nifty import scrape_gift_nifty
from nti.scrapers.mmi_alternative import scrape_mmi_alternative

# Indicators
from nti.indicators.normalizer import normalize_all_indicators
from nti.indicators.composite import compute_custom_fg_composite, compute_global_overnight_composite
from nti.indicators.technical_display import compute_rsi, compute_macd

# Model
from nti.model.predictor import run_inference
from nti.model.fallback import run_fallback_inference

# Blog & News
from nti.llm.blog_generator import generate_hourly_blog
from nti.llm.news_analyzer import analyze_news

# Changelog
from nti.changelog.generator import generate_changelog, load_previous_run, save_current_run

# Storage
from nti.storage.csv_writer import write_hourly_csv
from nti.storage.json_api import write_latest_json, write_history_json
from nti.storage.blog_writer import write_blog_post
from nti.storage.git_committer import git_commit_and_push

# Notifications
from nti.notifications.email_sender import send_zone_change_alert, send_big_move_alert

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _safe_scrape(scraper_fn, scraper_name: str) -> dict:
    """Run a scraper safely, returning empty dict on failure.

    Args:
        scraper_fn: Callable that returns a dict
        scraper_name: Human-readable name for logging

    Returns:
        Dict from scraper, or empty dict on failure
    """
    try:
        result = scraper_fn()
        logger.info(f"Scraper {scraper_name}: success ({len(result)} keys)")
        return result
    except Exception as e:
        logger.warning(f"Scraper {scraper_name} failed: {e}")
        return {}


def run_hourly_pipeline(dry_run: bool = False) -> dict:
    """Run the full hourly NTI pipeline.

    This is the main entry point called by GitHub Actions cron.

    Args:
        dry_run: If True, don't write files or push to git

    Returns:
        Dict with pipeline results and metadata
    """
    start_time = time.time()
    pipeline_errors: list[str] = []

    now_ist = datetime.now(IST)
    logger.info(f"=== NTI Hourly Pipeline Started at {now_ist.isoformat()} ===")

    # Check for holidays
    if is_market_holiday(now_ist.date()):
        logger.info(f"Today is a market holiday — running in limited mode")

    # -----------------------------------------------------------------------
    # STEP 1: Scrape all indicators
    # -----------------------------------------------------------------------
    logger.info("Step 1: Scraping indicators...")

    raw_indicators: dict = {}

    # NSE index data (PE, PB, VIX, dividend yield, advances/declines)
    nse_data = _safe_scrape(scrape_nse_index_data, "NSE Indices")
    raw_indicators.update(nse_data)

    # Yahoo Finance (USD/INR, Brent Crude, S&P 500, global indices)
    yf_data = _safe_scrape(scrape_global_markets, "Yahoo Finance")
    raw_indicators.update(yf_data)

    # FRED API (US 10-Year Yield)
    fred_data = _safe_scrape(scrape_us_10y_yield, "FRED API")
    raw_indicators.update(fred_data)

    # FII/DII flows
    fii_dii_data = _safe_scrape(scrape_fii_dii_cash_flow, "FII/DII")
    raw_indicators.update(fii_dii_data)

    # PCR from NSE options
    pcr_data = _safe_scrape(lambda: scrape_put_call_ratio(), "NSE Options PCR")
    raw_indicators.update(pcr_data)

    # Tickertape MMI
    mmi_data = _safe_scrape(scrape_mmi_selenium, "Tickertape MMI")
    raw_indicators.update(mmi_data)

    # RBI repo rate
    rbi_data = _safe_scrape(scrape_rbi_repo_rate, "RBI Repo Rate")
    raw_indicators.update(rbi_data)

    # CPI inflation
    cpi_data = _safe_scrape(scrape_cpi_inflation, "MOSPI CPI")
    raw_indicators.update(cpi_data)

    # AMFI SIP flows
    sip_data = _safe_scrape(scrape_amfi_sip_flows, "AMFI SIP")
    raw_indicators.update(sip_data)

    # CNN Fear & Greed
    cnn_data = _safe_scrape(scrape_cnn_fear_greed, "CNN F&G")
    raw_indicators.update(cnn_data)

    # GIFT Nifty pre-market
    gift_data = _safe_scrape(scrape_gift_nifty, "GIFT Nifty")
    raw_indicators.update(gift_data)

    # Alternative MMI (fallback if Tickertape failed)
    if raw_indicators.get("mmi_value") is None:
        alt_mmi_data = _safe_scrape(scrape_mmi_alternative, "MMI Alternative")
        if alt_mmi_data.get("mmi_value") is not None:
            raw_indicators.update(alt_mmi_data)

    # Count how many indicators we got
    non_none_count = sum(1 for v in raw_indicators.values() if v is not None)
    logger.info(f"Scraped {non_none_count}/{len(raw_indicators)} indicators with values")

    # -----------------------------------------------------------------------
    # STEP 2: Normalize indicators
    # -----------------------------------------------------------------------
    logger.info("Step 2: Normalizing indicators...")

    normalized = normalize_all_indicators(raw_indicators)
    logger.info(f"Normalized {len(normalized)} indicator scores")

    # Compute composites
    vix_norm = normalized.get("vix_normalized")
    pcr_norm = normalized.get("pcr_normalized")
    ad_ratio = raw_indicators.get("advance_decline_ratio")
    hl_ratio = raw_indicators.get("high_low_ratio")

    custom_fg = compute_custom_fg_composite(vix_norm, pcr_norm, ad_ratio, hl_ratio)
    if custom_fg is not None:
        raw_indicators["custom_fg_composite"] = custom_fg
        normalized["custom_fg_composite"] = custom_fg

    # Global overnight composite
    global_overnight = compute_global_overnight_composite(
        raw_indicators.get("dow_jones_change"),
        raw_indicators.get("nasdaq_change"),
        raw_indicators.get("nikkei_change"),
        raw_indicators.get("hang_seng_change"),
    )
    if global_overnight is not None:
        raw_indicators["global_overnight"] = global_overnight

    # -----------------------------------------------------------------------
    # STEP 3: LLM News Analysis
    # -----------------------------------------------------------------------
    logger.info("Step 3: Analyzing news...")

    news_result = analyze_news()
    if news_result:
        raw_indicators["llm_news_danger"] = news_result.get("danger_score")
        raw_indicators["llm_policy_flag"] = news_result.get("policy_flag")
        raw_indicators["llm_geopolitical_score"] = news_result.get("geopolitical_risk")

    # -----------------------------------------------------------------------
    # STEP 4: Run ML inference
    # -----------------------------------------------------------------------
    logger.info("Step 4: Running NTI inference...")

    # Load previous run data for lagged features
    previous_run = load_previous_run()
    prev_score = previous_run.get("nti_score")
    score_yesterday = previous_run.get("nti_score_yesterday")
    pe_5d_ago = previous_run.get("nifty_pe_5d_ago")
    vix_5d_ago = previous_run.get("india_vix_5d_ago")

    if settings.enable_model:
        nti_result = run_inference(
            raw_indicators,
            previous_score=prev_score,
            score_yesterday=score_yesterday,
            pe_5d_ago=pe_5d_ago,
            vix_5d_ago=vix_5d_ago,
        )
    else:
        nti_result = run_fallback_inference(raw_indicators)

    nti_score = nti_result.get("nti_score", 50)
    zone = nti_result.get("zone", get_zone(nti_score))
    confidence = nti_result.get("confidence", 50)
    is_fallback = nti_result.get("is_fallback", True)
    top_drivers = nti_result.get("top_drivers", [])

    logger.info(f"NTI Score: {nti_score:.1f} ({zone}) | Confidence: {confidence:.0f}% | Fallback: {is_fallback}")

    # Store in indicators for writing
    raw_indicators["nti_score"] = nti_score
    raw_indicators["nti_score_prev"] = prev_score
    raw_indicators["zone"] = zone
    raw_indicators["zone_prev"] = previous_run.get("zone", "UNKNOWN")
    raw_indicators["confidence"] = confidence
    raw_indicators["is_fallback"] = is_fallback
    raw_indicators["model_version"] = nti_result.get("model_version", "rule-based")

    # -----------------------------------------------------------------------
    # STEP 5: Generate changelog
    # -----------------------------------------------------------------------
    logger.info("Step 5: Generating changelog...")

    # Load previous stock picks for screener changelog
    prev_stocks = previous_run.get("top_stocks", [])

    changelog_text = generate_changelog(
        current=raw_indicators,
        previous=previous_run,
        current_stocks=[],  # Populated by screener if available
        previous_stocks=prev_stocks,
    )

    # -----------------------------------------------------------------------
    # STEP 6: Generate blog post
    # -----------------------------------------------------------------------
    logger.info("Step 6: Generating blog post...")

    # Determine blog type based on time
    blog_type = _get_blog_type(now_ist)

    # Format top drivers for blog
    top_drivers_text = ""
    if top_drivers:
        driver_lines = []
        for d in top_drivers[:5]:
            indicator = d.get("indicator", d.get("feature", ""))
            direction = d.get("direction", "")
            label = d.get("label", indicator)
            driver_lines.append(f"  {direction.upper()} {label}")
        top_drivers_text = "\n".join(driver_lines)

    # Format top stocks
    top_stocks_formatted = "No screener data available this hour"
    top_stock_symbols: list[str] = []

    # Generate the blog
    blog_result = generate_hourly_blog(
        nti_score=nti_score,
        prev_score=prev_score or nti_score,
        confidence=confidence,
        indicators=raw_indicators,
        top_drivers=top_drivers,
        top_stocks=[],
        changelog_text=changelog_text,
        blog_type=blog_type,
    )

    blog_markdown = blog_result.get("blog_markdown", "")
    blog_slug = now_ist.strftime("%Y-%m-%d-%H-%M")

    # -----------------------------------------------------------------------
    # STEP 7: Write data files
    # -----------------------------------------------------------------------
    if not dry_run:
        logger.info("Step 7: Writing data files...")

        # Write hourly CSV
        write_hourly_csv(raw_indicators)

        # Write API JSON files
        driver_dicts = []
        for d in top_drivers[:5]:
            driver_dicts.append({
                "indicator": d.get("indicator", d.get("feature", "")),
                "label": d.get("label", d.get("indicator", "")),
                "shap": d.get("shap", 0),
                "direction": d.get("direction", ""),
                "current_value": d.get("description", ""),
            })

        write_latest_json(
            indicators=raw_indicators,
            nti_result=nti_result,
            top_drivers=driver_dicts,
            top_stocks=[],
            blog_slug=blog_slug,
        )

        write_history_json()

        # Write blog post
        if blog_markdown:
            write_blog_post(
                blog_markdown=blog_markdown,
                nti_score=nti_score,
                prev_score=prev_score,
                confidence=confidence,
                nifty_price=raw_indicators.get("nifty_price"),
                top_drivers=[d.get("indicator", d.get("feature", "")) for d in top_drivers[:5]],
                top_stocks=top_stock_symbols,
                blog_type=blog_type,
            )

        # Save current run for next comparison
        save_current_run(raw_indicators)

    # -----------------------------------------------------------------------
    # STEP 8: Send alerts
    # -----------------------------------------------------------------------
    if not dry_run and settings.enable_email:
        logger.info("Step 8: Checking for alerts...")

        prev_zone = previous_run.get("zone", "UNKNOWN")

        # Zone change alert
        if zone != prev_zone and prev_zone != "UNKNOWN" and settings.alert_on_zone_change:
            driver_names = [d.get("label", d.get("indicator", "")) for d in top_drivers[:3]]
            try:
                send_zone_change_alert(
                    from_zone=prev_zone,
                    to_zone=zone,
                    score=nti_score,
                    nifty_price=raw_indicators.get("nifty_price", 0),
                    drivers=driver_names,
                )
                logger.info(f"Zone change alert sent: {prev_zone} → {zone}")
            except Exception as e:
                pipeline_errors.append(f"Email alert failed: {e}")

        # Big move alert
        if prev_score is not None and settings.alert_on_big_move:
            move = abs(nti_score - prev_score)
            if move >= settings.alert_big_move_threshold:
                try:
                    send_big_move_alert(prev_score, nti_score, raw_indicators.get("nifty_price", 0))
                    logger.info(f"Big move alert sent: {move:.1f} points")
                except Exception as e:
                    pipeline_errors.append(f"Big move alert failed: {e}")

    # -----------------------------------------------------------------------
    # STEP 9: Git commit and push
    # -----------------------------------------------------------------------
    if not dry_run:
        logger.info("Step 9: Git commit and push...")
        git_commit_and_push()
    else:
        logger.info("[DRY RUN] Skipping git commit and push")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    duration = time.time() - start_time
    raw_indicators["run_duration_seconds"] = round(duration, 1)
    raw_indicators["errors"] = "; ".join(pipeline_errors) if pipeline_errors else ""

    logger.info(
        f"=== NTI Hourly Pipeline Complete ===\n"
        f"  Score: {nti_score:.1f} ({zone})\n"
        f"  Confidence: {confidence:.0f}%\n"
        f"  Fallback: {is_fallback}\n"
        f"  Indicators: {non_none_count} scraped\n"
        f"  Duration: {duration:.1f}s\n"
        f"  Errors: {len(pipeline_errors)}"
    )

    return {
        "nti_score": nti_score,
        "zone": zone,
        "confidence": confidence,
        "is_fallback": is_fallback,
        "indicators_scraped": non_none_count,
        "duration_seconds": duration,
        "blog_slug": blog_slug,
        "errors": pipeline_errors,
    }


def _get_blog_type(now_ist: datetime) -> str:
    """Determine blog type based on time of day."""
    hour = now_ist.hour
    minute = now_ist.minute

    if hour == 9 and minute < 30:
        return "market_open"
    if hour == 15 and minute >= 15:
        return "market_close"
    if 9 <= hour <= 15:
        return "mid_session"
    if 16 <= hour <= 20:
        return "post_market"
    return "overnight"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_hourly_pipeline()
    print(f"\nNTI Score: {result['nti_score']:.1f} ({result['zone']})")
    print(f"Duration: {result['duration_seconds']:.1f}s")
