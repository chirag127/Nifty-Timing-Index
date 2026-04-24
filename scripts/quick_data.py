#!/usr/bin/env python3
"""Quick Data Populate — Scrape indicators + write API JSON without slow LLM steps.

Skips: blog generation, news analysis, git push
Runs: scrapers, normalization, fallback inference, JSON writes
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.thresholds import get_zone

# Scrapers
from nti.scrapers.nse_indices import scrape_nse_index_data
from nti.scrapers.yahoo_finance import scrape_global_markets
from nti.scrapers.fred_api import scrape_us_10y_yield
from nti.scrapers.nse_fii_dii import scrape_fii_dii_cash_flow
from nti.scrapers.nse_options import scrape_put_call_ratio
from nti.scrapers.rbi_data import scrape_rbi_repo_rate
from nti.scrapers.mospi_data import scrape_cpi_inflation
from nti.scrapers.amfi_data import scrape_amfi_sip_flows
from nti.scrapers.cnn_fear_greed import scrape_cnn_fear_greed
from nti.scrapers.gift_nifty import scrape_gift_nifty

# Indicators
from nti.indicators.normalizer import normalize_all_indicators
from nti.indicators.composite import compute_custom_fg_composite, compute_global_overnight_composite

# Model
from nti.model.fallback import run_fallback_inference

# Storage
from nti.storage.json_api import write_latest_json, write_history_json

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")


def _safe_scrape(scraper_fn, scraper_name: str) -> dict:
    """Run a scraper safely, returning empty dict on failure."""
    try:
        result = scraper_fn()
        logger.info(f"Scraper {scraper_name}: success ({len(result)} keys)")
        return result
    except Exception as e:
        logger.warning(f"Scraper {scraper_name} failed: {e}")
        return {}


def run_quick_data() -> dict:
    """Run a fast data-only pipeline (no LLM, no blog, no git)."""
    start_time = time.time()
    now_ist = datetime.now(IST)
    logger.info(f"=== NTI Quick Data Populate at {now_ist.isoformat()} ===")

    # Step 1: Scrape indicators (skip slow ones like Selenium MMI)
    logger.info("Step 1: Scraping indicators...")
    raw_indicators: dict = {}

    scrapers = [
        (scrape_nse_index_data, "NSE Indices"),
        (scrape_global_markets, "Yahoo Finance"),
        (scrape_us_10y_yield, "FRED API"),
        (scrape_fii_dii_cash_flow, "FII/DII"),
        (lambda: scrape_put_call_ratio(), "NSE Options PCR"),
        (scrape_rbi_repo_rate, "RBI Repo Rate"),
        (scrape_cpi_inflation, "MOSPI CPI"),
        (scrape_amfi_sip_flows, "AMFI SIP"),
        (scrape_cnn_fear_greed, "CNN F&G"),
        (scrape_gift_nifty, "GIFT Nifty"),
    ]

    for fn, name in scrapers:
        data = _safe_scrape(fn, name)
        raw_indicators.update(data)

    non_none_count = sum(1 for v in raw_indicators.values() if v is not None)
    logger.info(f"Scraped {non_none_count}/{len(raw_indicators)} indicators with values")

    # Step 2: Normalize
    logger.info("Step 2: Normalizing indicators...")
    normalized = normalize_all_indicators(raw_indicators)

    # Composites
    vix_norm = normalized.get("vix_normalized")
    pcr_norm = normalized.get("pcr_normalized")
    ad_ratio = raw_indicators.get("advance_decline_ratio")
    hl_ratio = raw_indicators.get("high_low_ratio")

    custom_fg = compute_custom_fg_composite(None, vix_norm, pcr_norm, ad_ratio, hl_ratio)
    if custom_fg is not None:
        raw_indicators["custom_fg_composite"] = custom_fg
        normalized["custom_fg_composite"] = custom_fg

    global_overnight = compute_global_overnight_composite(
        None,
        raw_indicators.get("dow_jones_change"),
        raw_indicators.get("nasdaq_change"),
        raw_indicators.get("nikkei_change"),
        raw_indicators.get("hang_seng_change"),
    )
    if global_overnight is not None:
        raw_indicators["global_overnight"] = global_overnight

    # Step 3: Fallback inference
    logger.info("Step 3: Running fallback inference...")
    nti_result = run_fallback_inference(raw_indicators)
    nti_score = nti_result.get("nti_score", 50)
    zone = nti_result.get("zone", get_zone(nti_score))
    confidence = nti_result.get("confidence", 50)
    top_drivers = nti_result.get("top_drivers", [])

    logger.info(f"NTI Score: {nti_score:.1f} ({zone}) | Confidence: {confidence:.0f}%")

    raw_indicators["nti_score"] = nti_score
    raw_indicators["zone"] = zone
    raw_indicators["confidence"] = confidence
    raw_indicators["is_fallback"] = True

    # Step 4: Write API JSON files
    logger.info("Step 4: Writing API JSON files...")

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
        blog_slug=now_ist.strftime("%Y-%m-%d-%H-%M"),
    )

    write_history_json()

    # Also update previous_run.json for next comparison
    prev_path = DATA_DIR / "api" / "previous_run.json"
    try:
        with open(prev_path, "w", encoding="utf-8") as f:
            json.dump(raw_indicators, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save previous_run.json: {e}")

    duration = time.time() - start_time
    logger.info(f"=== Quick Data Populate Complete ({duration:.1f}s) ===")
    logger.info(f"  Score: {nti_score:.1f} ({zone})")
    logger.info(f"  Indicators: {non_none_count}")
    logger.info(f"  Duration: {duration:.1f}s")

    return {
        "nti_score": nti_score,
        "zone": zone,
        "confidence": confidence,
        "indicators_scraped": non_none_count,
        "duration_seconds": duration,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_quick_data()
    print(f"\nNTI Score: {result['nti_score']:.1f} ({result['zone']})")
    print(f"Confidence: {result['confidence']:.0f}%")
    print(f"Indicators: {result['indicators_scraped']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
