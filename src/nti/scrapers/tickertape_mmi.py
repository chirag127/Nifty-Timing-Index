"""Tickertape MMI Scraper — Market Mood Index via Selenium headless Chrome.

MMI is a sentiment indicator (0–100):
- 0–25: Extreme Fear (contrarian buy)
- 25–45: Fear
- 45–55: Neutral
- 55–75: Greed
- 75–100: Extreme Greed (contrarian sell)

Source: https://www.tickertape.in/market-mood-index
"""

from __future__ import annotations

import logging
import re

from nti.config.settings import settings
from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

MMI_URL = "https://www.tickertape.in/market-mood-index"

# MMI zone mapping
MMI_ZONES = {
    (0, 25): "Extreme Fear",
    (25, 45): "Fear",
    (45, 55): "Neutral",
    (55, 75): "Greed",
    (75, 100): "Extreme Greed",
}


def get_mmi_zone(mmi_value: float) -> str:
    """Get zone name from MMI value."""
    for (low, high), zone in MMI_ZONES.items():
        if low <= mmi_value <= high:
            return zone
    return "Neutral"


def scrape_mmi_selenium() -> dict:
    """Scrape Tickertape MMI using Selenium headless Chrome.

    Returns:
        dict with keys: mmi_value, mmi_zone
    """
    if not settings.enable_chromium_scraping:
        logger.info("Chromium scraping disabled — skipping MMI scrape")
        return {"mmi_value": None, "mmi_zone": None}

    result: dict = {"mmi_value": None, "mmi_zone": None}

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        try:
            driver.get(MMI_URL)

            # Wait for MMI value to appear (up to 15 seconds)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='mmi'], [class*='market-mood']"))
            )

            # Try multiple selectors to find the MMI number
            page_source = driver.page_source

            # Method 1: Look for a number in common MMI-related elements
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, "div, span, h1, h2, h3, p")
                for el in elements:
                    text = el.text.strip()
                    # MMI is a number between 0–100
                    if text and re.match(r"^\d{1,3}(\.\d+)?$", text):
                        val = float(text)
                        if 0 <= val <= 100:
                            result["mmi_value"] = validate_value("mmi", val)
                            result["mmi_zone"] = get_mmi_zone(val)
                            break
            except Exception:
                pass

            # Method 2: Regex search in page source
            if result["mmi_value"] is None:
                mmi_pattern = re.search(r'"mmi"[:\s]+(\d{1,3}(?:\.\d+)?)', page_source)
                if not mmi_pattern:
                    mmi_pattern = re.search(r'marketMoodIndex[^\d]*(\d{1,3}(?:\.\d+)?)', page_source)
                if not mmi_pattern:
                    mmi_pattern = re.search(r'MMI[^\d]*(\d{1,3}(?:\.\d+)?)', page_source, re.IGNORECASE)

                if mmi_pattern:
                    val = float(mmi_pattern.group(1))
                    if 0 <= val <= 100:
                        result["mmi_value"] = validate_value("mmi", val)
                        result["mmi_zone"] = get_mmi_zone(val)

            if result["mmi_value"] is not None:
                logger.info(f"MMI: {result['mmi_value']} ({result['mmi_zone']})")
            else:
                logger.warning("Could not extract MMI value from Tickertape page")

        finally:
            driver.quit()

    except ImportError:
        logger.warning("Selenium/webdriver-manager not installed — cannot scrape MMI")
    except Exception as e:
        logger.warning(f"MMI Selenium scrape failed: {e}")

    return result


def scrape_mmi_requests_fallback() -> dict:
    """Fallback: Try scraping MMI with simple requests (may not work due to JS rendering).

    Returns:
        dict with keys: mmi_value, mmi_zone
    """
    import httpx

    result: dict = {"mmi_value": None, "mmi_zone": None}

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                MMI_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )
            text = resp.text

            # Try to find MMI value in page content
            mmi_pattern = re.search(r'"mmi"[:\s]+(\d{1,3}(?:\.\d+)?)', text)
            if not mmi_pattern:
                mmi_pattern = re.search(r'MMI[^\d]*(\d{1,3}(?:\.\d+)?)', text, re.IGNORECASE)

            if mmi_pattern:
                val = float(mmi_pattern.group(1))
                if 0 <= val <= 100:
                    result["mmi_value"] = validate_value("mmi", val)
                    result["mmi_zone"] = get_mmi_zone(val)

    except Exception as e:
        logger.warning(f"MMI requests fallback failed: {e}")

    return result


def scrape_mmi() -> dict:
    """Scrape MMI — tries Selenium first, then requests fallback.

    Returns:
        dict with keys: mmi_value, mmi_zone
    """
    # Try Selenium (most reliable for JS-heavy pages)
    result = scrape_mmi_selenium()
    if result.get("mmi_value") is not None:
        return result

    # Fallback to simple requests
    result = scrape_mmi_requests_fallback()
    if result.get("mmi_value") is not None:
        return result

    logger.warning("All MMI scrape methods failed — using None")
    return {"mmi_value": None, "mmi_zone": None}
