"""RBI Data Scraper — Repo Rate and Monetary Policy Stance.

Scrapes the RBI press releases page to determine:
- Current repo rate
- Policy stance (accommodative, neutral, tightening)
- Direction (cutting, holding, hiking)
"""

from __future__ import annotations

import logging
import re

import httpx

from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

RBI_URL = "https://www.rbi.org.in/Scripts/bs_viewcontent.aspx?Id=2147"


def scrape_rbi_repo_rate() -> dict:
    """Scrape RBI repo rate from RBI website.

    Returns:
        dict with keys: rbi_repo_rate, rbi_stance, rbi_direction
        - rbi_repo_rate: float (e.g., 6.5)
        - rbi_stance: str ("accommodative", "neutral", "tightening")
        - rbi_direction: int (+1 = cutting, 0 = hold, -1 = hiking)
    """
    result: dict = {
        "rbi_repo_rate": None,
        "rbi_stance": None,
        "rbi_direction": 0,
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                RBI_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            text = resp.text

            # Try to find repo rate in page content
            # Pattern: "Repo Rate" followed by a number
            rate_pattern = re.search(
                r"repo\s+rate[^0-9]*(\d+\.?\d*)",
                text,
                re.IGNORECASE,
            )
            if rate_pattern:
                rate = float(rate_pattern.group(1))
                if 0 < rate < 20:
                    result["rbi_repo_rate"] = validate_value("rbi_repo_rate", rate)

            # Determine stance from page text
            text_lower = text.lower()
            if "accommodative" in text_lower:
                result["rbi_stance"] = "accommodative"
                result["rbi_direction"] = 1  # Cutting/loosening
            elif "tightening" in text_lower or "withdrawal of accommodation" in text_lower:
                result["rbi_stance"] = "tightening"
                result["rbi_direction"] = -1  # Hiking
            elif "neutral" in text_lower:
                result["rbi_stance"] = "neutral"
                result["rbi_direction"] = 0  # Hold
            else:
                # Default based on known current rate (2026)
                result["rbi_stance"] = "neutral"
                result["rbi_direction"] = 0

            if result["rbi_repo_rate"]:
                logger.info(
                    f"RBI repo rate: {result['rbi_repo_rate']}% "
                    f"(stance: {result['rbi_stance']}, direction: {result['rbi_direction']})"
                )

    except Exception as e:
        logger.warning(f"RBI data scrape failed: {e}")

    # Default fallback: known current rate if scrape fails
    if result["rbi_repo_rate"] is None:
        result["rbi_repo_rate"] = 6.50  # Current RBI repo rate as of 2026
        result["rbi_stance"] = "neutral"
        result["rbi_direction"] = 0
        logger.info("Using fallback RBI repo rate: 6.50%")

    return result


# Alias for backward compatibility with tests
def scrape_rbi_rate() -> dict:
    """Alias for scrape_rbi_repo_rate."""
    return scrape_rbi_repo_rate()
