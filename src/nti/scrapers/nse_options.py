"""NSE Options Chain Scraper — Put/Call Ratio (PCR) computation.

PCR is a contrarian sentiment indicator:
- PCR < 0.7: Call-heavy = greed = danger (high NTI score)
- PCR > 1.3: Put-heavy = fear = opportunity (low NTI score)
"""

from __future__ import annotations

import logging
import time

import requests

from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

# NSE options chain API endpoint for Nifty
NSE_OPTIONS_URL = "https://www.nseindia.com/api/option-chain-indices"
NSE_SYMBOL = "NIFTY"


def _create_nse_session() -> requests.Session:
    """Create a requests session with NSE-appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    })
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception:
        logger.warning("Could not establish NSE session cookies")
    return session


def scrape_put_call_ratio(symbol: str = NSE_SYMBOL) -> dict:
    """Scrape NSE options chain and compute Put/Call Ratio (PCR).

    PCR = Total Put Open Interest / Total Call Open Interest

    Args:
        symbol: Index symbol (default: NIFTY). Also supports BANKNIFTY.

    Returns:
        dict with keys: pcr, total_call_oi, total_put_oi, total_call_volume,
                         total_put_volume, underlying_price
    """
    result: dict = {
        "pcr": None,
        "total_call_oi": 0,
        "total_put_oi": 0,
        "total_call_volume": 0,
        "total_put_volume": 0,
        "underlying_price": None,
    }

    session = _create_nse_session()

    try:
        resp = session.get(
            NSE_OPTIONS_URL,
            params={"symbol": symbol},
            timeout=15,
        )

        if resp.status_code != 200:
            logger.warning(f"NSE options API returned {resp.status_code}")
            return result

        data = resp.json()
        records = data.get("records", {})
        data_list = records.get("data", [])

        if not data_list:
            logger.warning("No options data returned from NSE")
            return result

        # Underlying price
        result["underlying_price"] = records.get("underlyingValue", None)

        total_call_oi = 0
        total_put_oi = 0
        total_call_volume = 0
        total_put_volume = 0

        for item in data_list:
            ce = item.get("CE", {})
            pe = item.get("PE", {})

            if ce:
                oi = ce.get("openInterest", 0) or 0
                vol = ce.get("totalTradedVolume", 0) or 0
                total_call_oi += oi
                total_call_volume += vol

            if pe:
                oi = pe.get("openInterest", 0) or 0
                vol = pe.get("totalTradedVolume", 0) or 0
                total_put_oi += oi
                total_put_volume += vol

        result["total_call_oi"] = total_call_oi
        result["total_put_oi"] = total_put_oi
        result["total_call_volume"] = total_call_volume
        result["total_put_volume"] = total_put_volume

        # Calculate PCR = Put OI / Call OI
        if total_call_oi > 0:
            pcr = total_put_oi / total_call_oi
            result["pcr"] = validate_value("pcr", pcr)
            logger.info(f"PCR for {symbol}: {pcr:.2f} (Call OI: {total_call_oi:,}, Put OI: {total_put_oi:,})")
        else:
            logger.warning("Total Call OI is 0 — cannot compute PCR")

    except Exception as e:
        logger.warning(f"NSE options scrape failed: {e}")

    return result
