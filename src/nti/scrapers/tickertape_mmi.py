"""Tickertape MMI Scraper — Market Mood Index via API.

MMI is a sentiment indicator (0–100):
- 0–30: Extreme Fear (contrarian buy)
- 30–50: Fear
- 50–70: Greed
- 70–100: Extreme Greed (contrarian sell)

Source: https://api.tickertape.in/mmi/now
"""

from __future__ import annotations

import logging
import httpx

from nti.config.settings import settings
from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

MMI_API_URL = "https://api.tickertape.in/mmi/now"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "accept-version": "8.14.0",
    "dnt": "1",
    "origin": "https://www.tickertape.in",
    "priority": "u=1, i",
    "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
}

COOKIES = {
    "AMP_d9d4ec74fa": "JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjIwZGExNDExZS02MTk1LTQ5OTAtOGIzYy03MGEwNjNmYmMwMWElMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzcxMjU0NjgxNTEyJTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTc3MTI1NTY0MTI1NiUyQyUyMmxhc3RFdmVudElkJTIyJTNBNDI3JTdE"
}

# MMI zone mapping
MMI_ZONES = {
    (0, 30): "Extreme Fear",
    (30, 50): "Fear",
    (50, 70): "Greed",
    (70, 100): "Extreme Greed",
}


def get_mmi_zone(mmi_value: float) -> str:
    """Get zone name from MMI value."""
    for (low, high), zone in MMI_ZONES.items():
        if low <= mmi_value <= high:
            return zone
    return "Extreme Greed" if mmi_value > 70 else "Extreme Fear"


def scrape_mmi_selenium() -> dict:
    """Legacy alias, calls the API instead."""
    return scrape_mmi_api()


def scrape_mmi_requests_fallback() -> dict:
    """Legacy alias, calls the API instead."""
    return scrape_mmi_api()


def scrape_mmi_api() -> dict:
    """Scrape Tickertape MMI using their JSON API.

    Returns:
        dict with keys: mmi_value, mmi_zone
    """
    result: dict = {"mmi_value": None, "mmi_zone": None}

    try:
        with httpx.Client(timeout=15.0, headers=HEADERS, cookies=COOKIES) as client:
            response = client.get(MMI_API_URL)
            response.raise_for_status()
            data = response.json()
            
            if data.get("success") and "data" in data and "currentValue" in data["data"]:
                val = float(data["data"]["currentValue"])
                if 0 <= val <= 100:
                    result["mmi_value"] = validate_value("mmi", val)
                    result["mmi_zone"] = get_mmi_zone(val)
                    logger.info(f"MMI: {result['mmi_value']} ({result['mmi_zone']})")
                else:
                    logger.warning(f"MMI value {val} out of bounds.")
            else:
                logger.warning(f"API returned success=False or missing data: {data}")

    except Exception as e:
        logger.warning(f"MMI API request failed: {e}")

    return result


def scrape_mmi() -> dict:
    """Scrape MMI using the tickertape API directly.

    Returns:
        dict with keys: mmi_value, mmi_zone
    """
    return scrape_mmi_api()

