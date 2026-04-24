"""FRED API Scraper — US 10-Year Bond Yield."""

from __future__ import annotations

import logging

import httpx

from nti.config.settings import settings
from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)


def scrape_us_10y_yield() -> dict:
    """Fetch US 10-Year Treasury Yield from FRED API (free, 120 req/min)."""
    if not settings.fred_api_key:
        logger.warning("FRED_API_KEY not configured")
        return {"us_10y_yield": None}

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "DGS10",
                    "api_key": settings.fred_api_key,
                    "limit": 1,
                    "sort_order": "desc",
                    "file_type": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            observations = data.get("observations", [])
            if observations:
                value = float(observations[0].get("value", 0))
                validated = validate_value("us_10y_yield", value)
                return {"us_10y_yield": validated}

    except Exception as e:
        logger.warning(f"FRED API failed: {e}")

    return {"us_10y_yield": None}


def scrape_fed_funds_rate() -> dict:
    """Fetch Federal Funds Rate from FRED (secondary use case)."""
    if not settings.fred_api_key:
        return {"fed_funds_rate": None}

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "FEDFUNDS",
                    "api_key": settings.fred_api_key,
                    "limit": 1,
                    "sort_order": "desc",
                    "file_type": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            if observations:
                return {"fed_funds_rate": float(observations[0].get("value", 0))}
    except Exception as e:
        logger.warning(f"FRED Fed Funds failed: {e}")

    return {"fed_funds_rate": None}
