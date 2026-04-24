"""Step 1: Scrape — Scrape all 30+ indicators from various sources.

Saves raw_indicators to data/api/step_scrape.json so subsequent steps
can resume even if this step took too long and the pipeline was restarted.

Also fills in missing data from previous_run.json and CSV history.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.settings import settings
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

from nti.storage.csv_writer import get_last_known_value
from nti.changelog.generator import load_previous_run

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")
API_DIR = DATA_DIR / "api"
STEP_FILE = API_DIR / "step_scrape.json"

# Indicators that commonly fail and should be backfilled from history
BACKFILL_INDICATORS = [
    "pcr",
    "fii_cash_net",
    "dii_cash_net",
    "gift_nifty_price",
    "gift_nifty_change",
    "gift_nifty_change_pct",
    "fii_cash_buy",
    "fii_cash_sell",
    "dii_cash_buy",
    "dii_cash_sell",
    "underlying_price",
    "total_call_oi",
    "total_put_oi",
]


def _safe_scrape(scraper_fn, scraper_name: str) -> dict:
    """Run a scraper safely, returning empty dict on failure."""
    try:
        result = scraper_fn()
        logger.info(f"Scraper {scraper_name}: success ({len(result)} keys)")
        return result
    except Exception as e:
        logger.warning(f"Scraper {scraper_name} failed: {e}")
        return {}


def _backfill_missing(raw_indicators: dict, previous_run: dict) -> dict:
    """Fill in None/missing values from previous run and CSV history.

    For indicators that are null because the market is closed or the
    scraper failed, we use the last known good value so the NTI score
    and blog have complete data.

    Priority:
    1. Current value (if not None) — keep it
    2. Previous run value — use last run's value
    3. CSV history — use get_last_known_value from hourly CSVs
    """
    filled = 0
    for key in BACKFILL_INDICATORS:
        current_val = raw_indicators.get(key)
        if current_val is not None:
            continue

        # Try previous run
        prev_val = previous_run.get(key)
        if prev_val is not None:
            raw_indicators[key] = prev_val
            filled += 1
            logger.debug(f"Backfilled {key} from previous_run: {prev_val}")
            continue

        # Try CSV history
        hist_val = get_last_known_value(key)
        if hist_val is not None:
            raw_indicators[key] = hist_val
            filled += 1
            logger.debug(f"Backfilled {key} from CSV history: {hist_val}")

    if filled:
        logger.info(f"Backfilled {filled} missing indicators from previous run / CSV history")
    return raw_indicators


def run_scrape_step(force: bool = False) -> dict:
    """Run the scrape step: fetch all indicators and save to disk.

    Args:
        force: If True, re-run even if step_scrape.json already exists today

    Returns:
        Dict of raw indicator values
    """
    now_ist = datetime.now(IST)
    logger.info(f"=== Step 1: Scrape (started {now_ist.isoformat()}) ===")

    # Check if market is holiday
    is_holiday = is_market_holiday(now_ist.date())
    if is_holiday:
        logger.info("Today is a market holiday — running in limited mode")

    # Check for existing step output (resume support)
    if not force and STEP_FILE.exists():
        try:
            with open(STEP_FILE) as f:
                cached = json.load(f)
            cached_time = cached.get("_scraped_at", "")
            # If scraped within the last 55 minutes, reuse it
            if cached_time:
                cached_dt = datetime.fromisoformat(cached_time)
                age_minutes = (now_ist - cached_dt).total_seconds() / 60
                if age_minutes < 55:
                    logger.info(f"Reusing scrape data from {age_minutes:.0f} min ago ({len(cached)} keys)")
                    return {k: v for k, v in cached.items() if not k.startswith("_")}
        except (json.JSONDecodeError, ValueError, OSError):
            pass  # Re-scrape

    start_time = time.time()
    raw_indicators: dict = {}

    # -------------------------------------------------------------------
    # Scrape all indicators
    # -------------------------------------------------------------------
    logger.info("Scraping indicators...")

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

    # -------------------------------------------------------------------
    # Backfill missing data from previous run and CSV history
    # -------------------------------------------------------------------
    previous_run = load_previous_run()
    raw_indicators = _backfill_missing(raw_indicators, previous_run)

    # Count how many indicators we got
    non_none_count = sum(1 for v in raw_indicators.values() if v is not None)
    logger.info(f"Scraped {non_none_count}/{len(raw_indicators)} indicators with values")

    # -------------------------------------------------------------------
    # Save intermediate state
    # -------------------------------------------------------------------
    API_DIR.mkdir(parents=True, exist_ok=True)
    save_data = dict(raw_indicators)
    save_data["_scraped_at"] = now_ist.isoformat()
    try:
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Scrape step saved to {STEP_FILE}")
    except OSError as e:
        logger.error(f"Failed to save scrape step: {e}")

    duration = time.time() - start_time
    logger.info(f"=== Step 1: Scrape complete ({duration:.1f}s, {non_none_count} indicators) ===")

    return raw_indicators


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="NTI Scrape Step")
    parser.add_argument("--force", action="store_true", help="Force re-scrape even if step file exists")
    args = parser.parse_args()

    result = run_scrape_step(force=args.force)
    non_none = sum(1 for v in result.values() if v is not None)
    print(f"\nScraped {non_none}/{len(result)} indicators")
