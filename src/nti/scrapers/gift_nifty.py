"""GIFT Nifty Pre-Market Scraper — GIFT Nifty (formerly SGX Nifty) pre-market indicator.

GIFT Nifty is a USD-denominated Nifty 50 futures contract traded on
NSE International Exchange (NSE IX) at GIFT City, Gujarat. It trades
nearly 21 hours/day, providing critical pre-market signals before
NSE India opens.

Pre-market signals:
- GIFT Nifty trading above previous Nifty close → bullish opening expected
- GIFT Nifty trading below previous Nifty close → bearish opening expected

Data sources (in order of reliability):
1. Moneycontrol GIFT Nifty page (scraped with httpx/BeautifulSoup)
2. NSE IX website (requires JS rendering — Selenium fallback)
3. Economic Times GIFT Nifty page

Note: No free public API exists for GIFT Nifty futures data.
All sources require web scraping.
"""

from __future__ import annotations

import logging
import re
import time

import httpx
from bs4 import BeautifulSoup

from nti.config.settings import settings
from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

# Source URLs for GIFT Nifty data
GIFT_NIFTY_SOURCES = {
    "moneycontrol": "https://www.moneycontrol.com/markets/gift-nifty/",
    "economictimes": "https://economictimes.indiatimes.com/markets/sgx-nifty",
    "nseix": "https://www.nseix.com/live-market-data",
}


def scrape_gift_nifty() -> dict:
    """Scrape GIFT Nifty pre-market data from available sources.

    Tries Moneycontrol first (most reliable for static scraping),
    then falls back to Economic Times.

    Returns:
        dict with keys:
            gift_nifty_price: float or None — Current/last GIFT Nifty price
            gift_nifty_change: float or None — Change from previous close
            gift_nifty_change_pct: float or None — Percentage change
            gift_nifty_signal: str — "Bullish", "Bearish", or "Neutral"
            gift_nifty_source: str — Which source provided the data
    """
    if not settings.gift_nifty_enabled:
        logger.info("GIFT Nifty scraping disabled — skipping")
        return {
            "gift_nifty_price": None,
            "gift_nifty_change": None,
            "gift_nifty_change_pct": None,
            "gift_nifty_signal": "Disabled",
            "gift_nifty_source": None,
        }

    result: dict = {
        "gift_nifty_price": None,
        "gift_nifty_change": None,
        "gift_nifty_change_pct": None,
        "gift_nifty_signal": "Neutral",
        "gift_nifty_source": None,
    }

    # Try Moneycontrol first
    result = _scrape_moneycontrol(result)
    if result.get("gift_nifty_price") is not None:
        return result

    # Fallback to Economic Times
    result = _scrape_economictimes(result)
    if result.get("gift_nifty_price") is not None:
        return result

    logger.warning("All GIFT Nifty scrape methods failed — using None")
    return result


def _scrape_moneycontrol(result: dict) -> dict:
    """Scrape GIFT Nifty from Moneycontrol page.

    Moneycontrol's GIFT Nifty page typically shows:
    - Current price
    - Change from previous close
    - Change percentage
    """
    url = GIFT_NIFTY_SOURCES["moneycontrol"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            text = resp.text
            soup = BeautifulSoup(text, "html.parser")

            # Method 1: Look for GIFT Nifty price in structured elements
            # Moneycontrol typically shows price in specific div/span patterns
            price_patterns = [
                # Pattern: "GIFT Nifty" followed by a number like 24,567.80
                re.compile(r"GIFT\s*Nifty[^0-9]*?([\d,]+\.?\d*)", re.IGNORECASE),
                # Pattern: "SGX Nifty" followed by a number
                re.compile(r"SGX\s*Nifty[^0-9]*?([\d,]+\.?\d*)", re.IGNORECASE),
                # Generic: any large number near "Nifty" context (5-digit index)
                re.compile(r'nifty[^0-9]*?([\d,]+\.?\d*)', re.IGNORECASE),
            ]

            for pattern in price_patterns:
                match = pattern.search(text)
                if match:
                    price_str = match.group(1).replace(",", "")
                    try:
                        price = float(price_str)
                        # GIFT Nifty is typically around Nifty 50 levels (20,000–30,000+)
                        if 15000 <= price <= 50000:
                            result["gift_nifty_price"] = validate_value("gift_nifty_price", price)
                            result["gift_nifty_source"] = "moneycontrol"
                            break
                    except ValueError:
                        continue

            # Try to find change value
            if result["gift_nifty_price"] is not None:
                change_patterns = [
                    re.compile(r'change[^0-9-]*?([+-]?[\d,]+\.?\d*)', re.IGNORECASE),
                    re.compile(r'([+-][\d,]+\.?\d*)\s*\([\d.]+%\)', re.IGNORECASE),
                ]
                for pattern in change_patterns:
                    match = pattern.search(text[max(0, text.find("GIFT") - 200):text.find("GIFT") + 500] if "GIFT" in text else text[:2000])
                    if match:
                        try:
                            change_str = match.group(1).replace(",", "").replace("+", "")
                            result["gift_nifty_change"] = float(change_str)
                            if result["gift_nifty_price"] and result["gift_nifty_change"]:
                                result["gift_nifty_change_pct"] = round(
                                    (result["gift_nifty_change"] / (result["gift_nifty_price"] - result["gift_nifty_change"])) * 100, 2
                                )
                            break
                        except (ValueError, ZeroDivisionError):
                            continue

            if result["gift_nifty_price"] is not None:
                result["gift_nifty_signal"] = _determine_signal(result["gift_nifty_change"])
                logger.info(
                    f"GIFT Nifty (moneycontrol): {result['gift_nifty_price']} "
                    f"({result['gift_nifty_change']}, {result['gift_nifty_signal']})"
                )

    except Exception as e:
        logger.warning(f"GIFT Nifty Moneycontrol scrape failed: {e}")

    return result


def _scrape_economictimes(result: dict) -> dict:
    """Scrape GIFT Nifty from Economic Times page.

    Economic Times has a dedicated SGX/GIFT Nifty page with
    pre-market data.
    """
    url = GIFT_NIFTY_SOURCES["economictimes"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            text = resp.text

            # Look for GIFT/SGX Nifty price patterns
            price_patterns = [
                re.compile(r"GIFT\s*Nifty[^0-9]*?([\d,]+\.?\d*)", re.IGNORECASE),
                re.compile(r"SGX\s*Nifty[^0-9]*?([\d,]+\.?\d*)", re.IGNORECASE),
            ]

            for pattern in price_patterns:
                match = pattern.search(text)
                if match:
                    price_str = match.group(1).replace(",", "")
                    try:
                        price = float(price_str)
                        if 15000 <= price <= 50000:
                            result["gift_nifty_price"] = validate_value("gift_nifty_price", price)
                            result["gift_nifty_source"] = "economictimes"
                            break
                    except ValueError:
                        continue

            if result["gift_nifty_price"] is not None:
                # Try to find change
                change_match = re.search(r'([+-]?[\d,]+\.?\d*)\s*\(([-\d.]+)%\)', text[:5000])
                if change_match:
                    try:
                        result["gift_nifty_change"] = float(change_match.group(1).replace(",", "").replace("+", ""))
                        result["gift_nifty_change_pct"] = float(change_match.group(2))
                    except ValueError:
                        pass

                result["gift_nifty_signal"] = _determine_signal(result["gift_nifty_change"])
                logger.info(
                    f"GIFT Nifty (economictimes): {result['gift_nifty_price']} "
                    f"({result['gift_nifty_change']}, {result['gift_nifty_signal']})"
                )

    except Exception as e:
        logger.warning(f"GIFT Nifty Economic Times scrape failed: {e}")

    return result


def _determine_signal(change: float | None) -> str:
    """Determine pre-market signal from GIFT Nifty change.

    Args:
        change: Price change from previous close (positive = bullish)

    Returns:
        "Bullish" if change > 0, "Bearish" if change < 0, "Neutral" otherwise
    """
    if change is None:
        return "Neutral"
    if change > 50:
        return "Strongly Bullish"
    if change > 0:
        return "Bullish"
    if change < -50:
        return "Strongly Bearish"
    if change < 0:
        return "Bearish"
    return "Neutral"


def scrape_gift_nifty_selenium() -> dict:
    """Scrape GIFT Nifty using Selenium headless Chrome.

    Used as a fallback when static scraping fails (JS-rendered pages).

    Returns:
        Same dict structure as scrape_gift_nifty()
    """
    if not settings.enable_chromium_scraping:
        return {
            "gift_nifty_price": None,
            "gift_nifty_change": None,
            "gift_nifty_change_pct": None,
            "gift_nifty_signal": "Disabled",
            "gift_nifty_source": None,
        }

    result: dict = {
        "gift_nifty_price": None,
        "gift_nifty_change": None,
        "gift_nifty_change_pct": None,
        "gift_nifty_signal": "Neutral",
        "gift_nifty_source": None,
    }

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
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
            url = GIFT_NIFTY_SOURCES["moneycontrol"]
            driver.get(url)

            time.sleep(3)  # Wait for JS to render

            page_source = driver.page_source

            # Search for GIFT Nifty price in rendered HTML
            price_patterns = [
                re.compile(r"GIFT\s*Nifty[^0-9]*?([\d,]+\.?\d*)", re.IGNORECASE),
                re.compile(r"SGX\s*Nifty[^0-9]*?([\d,]+\.?\d*)", re.IGNORECASE),
            ]

            for pattern in price_patterns:
                match = pattern.search(page_source)
                if match:
                    price_str = match.group(1).replace(",", "")
                    try:
                        price = float(price_str)
                        if 15000 <= price <= 50000:
                            result["gift_nifty_price"] = validate_value("gift_nifty_price", price)
                            result["gift_nifty_source"] = "moneycontrol-selenium"
                            break
                    except ValueError:
                        continue

            # Try to extract change data from rendered HTML
            if result["gift_nifty_price"] is not None:
                change_patterns = [
                    re.compile(r'change[^0-9-]*?([+-]?[\d,]+\.?\d*)', re.IGNORECASE),
                    re.compile(r'([+-][\d,]+\.?\d*)\s*\([\d.]+%\)'),
                ]
                for pattern in change_patterns:
                    match = pattern.search(page_source)
                    if match:
                        try:
                            change_str = match.group(1).replace(",", "").replace("+", "")
                            result["gift_nifty_change"] = float(change_str)
                            if result["gift_nifty_price"] and result["gift_nifty_change"]:
                                result["gift_nifty_change_pct"] = round(
                                    (result["gift_nifty_change"] / (result["gift_nifty_price"] - result["gift_nifty_change"])) * 100, 2
                                )
                            break
                        except (ValueError, ZeroDivisionError):
                            continue

                result["gift_nifty_signal"] = _determine_signal(result["gift_nifty_change"])
                logger.info(f"GIFT Nifty (selenium): {result['gift_nifty_price']} ({result['gift_nifty_signal']})")

        finally:
            driver.quit()

    except ImportError:
        logger.warning("Selenium not installed — cannot scrape GIFT Nifty with JS rendering")
    except Exception as e:
        logger.warning(f"GIFT Nifty Selenium scrape failed: {e}")

    return result
