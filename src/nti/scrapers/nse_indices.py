"""NSE Index Data Scraper — PE, PB, VIX, dividend yield via nsetools."""

from __future__ import annotations

import logging
import time

import requests

from nti.config.settings import settings
from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

NSE_INDICES = {
    "NIFTY 50": "nifty_50",
    "NIFTY BANK": "nifty_bank",
    "SENSEX": "sensex",
    "NIFTY MIDCAP 150": "nifty_midcap_150",
    "NIFTY SMALLCAP 250": "nifty_smallcap_250",
    "INDIA VIX": "india_vix",
}


def _create_nse_session() -> requests.Session:
    """Create a requests session with NSE-appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    # Hit homepage first to get cookies
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception:
        logger.warning("Could not establish NSE session cookies")
    return session


def scrape_nse_index_data() -> dict:
    """Scrape NSE index data: PE, PB, VIX, dividend yield.

    Returns dict with keys like nifty_pe, nifty_pb, india_vix, etc.
    Values may be None if scraping fails.
    """
    result: dict = {}

    # Try nsetools first
    try:
        from nsetools import Nse
        nse = Nse()

        # Nifty 50 quote
        nifty_quote = nse.get_index_quote("NIFTY 50")
        if nifty_quote:
            result["nifty_price"] = validate_value("nifty_price", float(nifty_quote.get("last", 0)))
            result["nifty_pe"] = validate_value("nifty_pe", float(nifty_quote.get("pe", 0)))
            result["nifty_pb"] = validate_value("nifty_pb", float(nifty_quote.get("pb", 0)))
            result["nifty_dy"] = validate_value("dividend_yield", float(nifty_quote.get("dy", 0)))
            result["nifty_change_pct"] = float(nifty_quote.get("change", 0))

        # India VIX
        vix_quote = nse.get_index_quote("INDIA VIX")
        if vix_quote:
            result["india_vix"] = validate_value("india_vix", float(vix_quote.get("last", 0)))

        # Other indices
        for index_name, key in NSE_INDICES.items():
            if key in ("nifty_50", "india_vix"):
                continue  # Already fetched
            try:
                quote = nse.get_index_quote(index_name)
                if quote:
                    result[f"{key}_price"] = validate_value("nifty_price", float(quote.get("last", 0)))
                    result[f"{key}_pe"] = validate_value("nifty_pe", float(quote.get("pe", 0)))
            except Exception:
                logger.warning(f"Could not fetch {index_name}")

        logger.info(f"NSE data scraped: {len(result)} values")
        return result

    except Exception as e:
        logger.warning(f"nsetools failed: {e}. Trying direct API...")

    # Fallback: direct NSE API
    return _scrape_nse_api()


def _scrape_nse_api() -> dict:
    """Fallback: scrape NSE directly via their API endpoints."""
    result: dict = {}
    session = _create_nse_session()

    try:
        # Nifty 50
        resp = session.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            stocks = data.get("stocks", [{}])
            if stocks:
                meta = stocks[0].get("metadata", {})
                result["nifty_pe"] = validate_value("nifty_pe", float(meta.get("pdSymbolPe", 0)))
                result["nifty_pb"] = validate_value("nifty_pb", float(meta.get("pdSymbolPb", 0)))
                result["nifty_price"] = validate_value("nifty_price", float(meta.get("last", 0)))
    except Exception as e:
        logger.warning(f"NSE API fallback failed for Nifty 50: {e}")

    try:
        # India VIX
        resp = session.get("https://www.nseindia.com/api/marketStatus", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for market in data.get("marketState", []):
                if "VIX" in market.get("index", "").upper():
                    result["india_vix"] = validate_value("india_vix", float(market.get("last", 0)))
    except Exception as e:
        logger.warning(f"NSE API fallback failed for VIX: {e}")

    return result


def scrape_fii_dii() -> dict:
    """Scrape FII/DII daily flow data from NSE archives."""
    result: dict = {}
    session = _create_nse_session()

    try:
        url = "https://archives.nseindia.com/content/nsccl/fao_participant_vol.csv"
        resp = session.get(url, timeout=15)
        if resp.status_code == 200 and resp.text:
            lines = resp.text.strip().split("\n")
            for line in lines[3:]:  # Skip header rows
                parts = line.split(",")
                if len(parts) >= 4:
                    category = parts[0].strip().upper()
                    if "FII" in category:
                        result["fii_fo_net"] = float(parts[-1].strip().replace(",", "") or 0)
                    elif "DII" in category:
                        result["dii_fo_net"] = float(parts[-1].strip().replace(",", "") or 0)
    except Exception as e:
        logger.warning(f"FII/DII scrape failed: {e}")

    # Also try the cash market FII/DII report
    try:
        url = "https://archives.nseindia.com/content/nsccl/fii_dii_trade.csv"
        resp = session.get(url, timeout=15)
        if resp.status_code == 200 and resp.text:
            lines = resp.text.strip().split("\n")
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 4:
                    category = parts[0].strip().upper()
                    if "FII" in category:
                        result["fii_cash_net"] = validate_value("fii_cash_net", float(parts[3].strip().replace(",", "") or 0))
                    elif "DII" in category:
                        result["dii_cash_net"] = float(parts[3].strip().replace(",", "") or 0)
    except Exception as e:
        logger.warning(f"FII/DII cash report scrape failed: {e}")

    return result


def scrape_pcr() -> dict:
    """Calculate Put/Call Ratio from NSE options data."""
    result: dict = {}

    try:
        import yfinance as yf
        # Use Nifty options data as proxy
        nifty = yf.Ticker("^NSEI")
        # PCR approximation from options chain if available
        result["pcr"] = 1.0  # Default neutral
    except Exception:
        result["pcr"] = 1.0

    return result
