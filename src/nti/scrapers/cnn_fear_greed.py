"""CNN Fear & Greed Index Scraper.

The CNN F&G index (0–100) measures US market sentiment:
- 0–25: Extreme Fear
- 25–45: Fear
- 45–55: Neutral
- 55–75: Greed
- 75–100: Extreme Greed

While US-focused, it provides useful global sentiment context.
"""

from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup

from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

CNN_FG_URL = "https://money.cnn.com/data/fear-and-greed/"


def scrape_cnn_fear_greed() -> dict:
    """Scrape CNN Fear & Greed Index.

    Returns:
        dict with keys: cnn_fg_value, cnn_fg_label
        - cnn_fg_value: int (0–100)
        - cnn_fg_label: str ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed")
    """
    result: dict = {
        "cnn_fg_value": None,
        "cnn_fg_label": None,
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                CNN_FG_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )
            text = resp.text

            # Parse with BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")

            # Try to find F&G value — CNN uses various formats
            # Method 1: Look for number in specific elements
            fg_elements = soup.find_all(["div", "span", "li"], string=re.compile(r"\d+"))
            for el in fg_elements:
                text_content = el.get_text(strip=True)
                if re.match(r"^\d{1,3}$", text_content):
                    val = int(text_content)
                    if 0 <= val <= 100:
                        result["cnn_fg_value"] = validate_value("cnn_fg", float(val))
                        break

            # Method 2: Regex in full HTML
            if result["cnn_fg_value"] is None:
                fg_pattern = re.search(
                    r"fear.and.greed[^0-9]*(\d{1,3})",
                    text,
                    re.IGNORECASE,
                )
                if not fg_pattern:
                    fg_pattern = re.search(
                        r"Market.*?Fear.*?Greed.*?(\d{1,3})",
                        text,
                        re.IGNORECASE,
                    )
                if fg_pattern:
                    val = int(fg_pattern.group(1))
                    if 0 <= val <= 100:
                        result["cnn_fg_value"] = validate_value("cnn_fg", float(val))

            # Determine label from value
            if result["cnn_fg_value"] is not None:
                val = result["cnn_fg_value"]
                if val < 25:
                    result["cnn_fg_label"] = "Extreme Fear"
                elif val < 45:
                    result["cnn_fg_label"] = "Fear"
                elif val < 55:
                    result["cnn_fg_label"] = "Neutral"
                elif val < 75:
                    result["cnn_fg_label"] = "Greed"
                else:
                    result["cnn_fg_label"] = "Extreme Greed"

                logger.info(f"CNN F&G: {result['cnn_fg_value']} ({result['cnn_fg_label']})")

    except Exception as e:
        logger.warning(f"CNN F&G scrape failed: {e}")

    # Default fallback
    if result["cnn_fg_value"] is None:
        result["cnn_fg_value"] = 45.0  # Neutral default
        result["cnn_fg_label"] = "Neutral"
        logger.info("Using fallback CNN F&G: 45 (Neutral)")

    return result


# Alias for backward compatibility with tests
def scrape_cnn_fg() -> dict:
    """Alias for scrape_cnn_fear_greed."""
    return scrape_cnn_fear_greed()
