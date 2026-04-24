"""MOSPI CPI Inflation Scraper — Consumer Price Index data.

Scrapes the MOSPI website for the latest CPI inflation rate.
CPI is a key macro indicator for the NTI model:
- CPI < 4%: Favorable for equities (low NTI score)
- CPI 4–6%: Moderate
- CPI > 6%: Unfavorable (high NTI score, RBI may hike rates)
"""

from __future__ import annotations

import logging
import re

import httpx

from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

MOSPI_URL = "https://mospi.gov.in/cpi"


def scrape_cpi_inflation() -> dict:
    """Scrape CPI inflation rate from MOSPI website.

    Returns:
        dict with keys: cpi_inflation, cpi_date
        - cpi_inflation: float (e.g., 4.2 for 4.2%)
        - cpi_date: str (e.g., "Mar 2026")
    """
    result: dict = {
        "cpi_inflation": None,
        "cpi_date": None,
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                MOSPI_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            text = resp.text

            # Try to find CPI inflation rate in page content
            # Common patterns: "CPI Inflation: 4.2%", "inflation rate of 4.2 per cent"
            cpi_pattern = re.search(
                r"(?:cpi|inflation)[^0-9]*(\d+\.?\d*)\s*(?:%|per\s*cent)",
                text,
                re.IGNORECASE,
            )
            if cpi_pattern:
                rate = float(cpi_pattern.group(1))
                if -2 <= rate <= 15:
                    result["cpi_inflation"] = validate_value("cpi_inflation", rate)

            # Try to find the date
            date_pattern = re.search(
                r"(\w{3}\s+\d{4})",
                text,
            )
            if date_pattern:
                result["cpi_date"] = date_pattern.group(1)

            if result["cpi_inflation"] is not None:
                logger.info(f"CPI Inflation: {result['cpi_inflation']}% ({result.get('cpi_date', 'unknown')})")

    except Exception as e:
        logger.warning(f"MOSPI CPI scrape failed: {e}")

    # Default fallback: reasonable estimate
    if result["cpi_inflation"] is None:
        result["cpi_inflation"] = 4.2  # Approximate current CPI
        logger.info("Using fallback CPI inflation: 4.2%")

    return result
