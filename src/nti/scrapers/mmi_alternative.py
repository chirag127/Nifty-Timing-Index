"""Alternative Market Mood Index (MMI) Scraper — RapidAPI + Smallcase fallbacks.

Provides additional MMI/sentiment data sources beyond the primary
Tickertape MMI scraper. These are used as cross-validation and fallback.

Sources:
1. RapidAPI — Market Mood Index API (free tier, requires MMI_RAPIDAPI_KEY)
2. Smallcase MMI API — used by various alert tools (may require auth)

MMI is a sentiment indicator (0–100):
- 0–25: Extreme Fear (contrarian buy)
- 25–45: Fear
- 45–55: Neutral
- 55–75: Greed
- 75–100: Extreme Greed (contrarian sell)

Note: "Moon Market Index" is not a recognized financial indicator.
The correct term is "Market Mood Index" (MMI).
"""

from __future__ import annotations

import logging

import httpx

from nti.config.settings import settings
from nti.config.thresholds import validate_value
from nti.scrapers.tickertape_mmi import get_mmi_zone

logger = logging.getLogger(__name__)

# RapidAPI MMI endpoint
RAPIDAPI_MMI_URL = "https://market-mood-index.p.rapidapi.com/mmi"
RAPIDAPI_HOST = "market-mood-index.p.rapidapi.com"

# Smallcase MMI API (public, no auth required — but may change)
SMALLCASE_MMI_URL = "https://api.smallcase.com/nsm/v1/marketMoodIndex"


def scrape_mmi_rapidapi() -> dict:
    """Scrape Market Mood Index via RapidAPI.

    Requires MMI_RAPIDAPI_KEY to be set in environment.
    Free tier: ~100 requests/month.

    Returns:
        dict with keys: mmi_value, mmi_zone, mmi_source
    """
    result: dict = {"mmi_value": None, "mmi_zone": None, "mmi_source": None}

    api_key = settings.mmi_rapidapi_key
    if not api_key:
        logger.debug("MMI RapidAPI key not configured — skipping")
        return result

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(RAPIDAPI_MMI_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # RapidAPI MMI response format varies; try common fields
            mmi_val = None
            if isinstance(data, dict):
                # Try common response structures
                for key in ("mmi", "value", "mmi_value", "marketMoodIndex", "data"):
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float)) and 0 <= val <= 100:
                            mmi_val = val
                            break
                        elif isinstance(val, dict):
                            for sub_key in ("mmi", "value", "mmi_value"):
                                if sub_key in val and isinstance(val[sub_key], (int, float)):
                                    sub_val = val[sub_key]
                                    if 0 <= sub_val <= 100:
                                        mmi_val = sub_val
                                        break
                            if mmi_val is not None:
                                break

            if mmi_val is not None:
                result["mmi_value"] = validate_value("mmi", mmi_val)
                result["mmi_zone"] = get_mmi_zone(mmi_val)
                result["mmi_source"] = "rapidapi"
                logger.info(f"MMI (RapidAPI): {result['mmi_value']} ({result['mmi_zone']})")
            else:
                logger.warning(f"RapidAPI MMI: unexpected response format: {data}")

    except httpx.HTTPStatusError as e:
        logger.warning(f"RapidAPI MMI HTTP error: {e.response.status_code}")
    except Exception as e:
        logger.warning(f"RapidAPI MMI scrape failed: {e}")

    return result


def scrape_mmi_smallcase() -> dict:
    """Scrape Market Mood Index via Smallcase API.

    Smallcase publishes a free MMI API endpoint.
    No authentication required (but may change).

    Returns:
        dict with keys: mmi_value, mmi_zone, mmi_source
    """
    result: dict = {"mmi_value": None, "mmi_zone": None, "mmi_source": None}

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                SMALLCASE_MMI_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Parse Smallcase response
            mmi_val = None
            if isinstance(data, dict):
                for key in ("mmi", "value", "mmi_value", "marketMoodIndex", "data"):
                    if key in data:
                        val = data[key]
                        if isinstance(val, (int, float)) and 0 <= val <= 100:
                            mmi_val = val
                            break
                        elif isinstance(val, dict):
                            for sub_key in ("mmi", "value", "mmi_value"):
                                if sub_key in val and isinstance(val[sub_key], (int, float)):
                                    sub_val = val[sub_key]
                                    if 0 <= sub_val <= 100:
                                        mmi_val = sub_val
                                        break
                            if mmi_val is not None:
                                break

            if mmi_val is not None:
                result["mmi_value"] = validate_value("mmi", mmi_val)
                result["mmi_zone"] = get_mmi_zone(mmi_val)
                result["mmi_source"] = "smallcase"
                logger.info(f"MMI (Smallcase): {result['mmi_value']} ({result['mmi_zone']})")
            else:
                logger.debug(f"Smallcase MMI: unexpected response format: {str(data)[:200]}")

    except Exception as e:
        logger.debug(f"Smallcase MMI scrape failed (expected — may require auth): {e}")

    return result


def scrape_mmi_alternative() -> dict:
    """Scrape MMI from alternative sources (RapidAPI + Smallcase).

    Used as a fallback when the primary Tickertape MMI scraper fails,
    or for cross-validation of sentiment data.

    Tries sources in order:
    1. RapidAPI (if key configured)
    2. Smallcase API (free, no auth)

    Returns:
        dict with keys: mmi_value, mmi_zone, mmi_source
    """
    # Try RapidAPI first (if configured)
    result = scrape_mmi_rapidapi()
    if result.get("mmi_value") is not None:
        return result

    # Try Smallcase API
    result = scrape_mmi_smallcase()
    if result.get("mmi_value") is not None:
        return result

    logger.info("All alternative MMI sources failed — primary Tickertape MMI should be used")
    return {"mmi_value": None, "mmi_zone": None, "mmi_source": None}
