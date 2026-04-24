"""Yahoo Finance Scraper — global markets, INR, crude, S&P 500."""

from __future__ import annotations

import logging

import yfinance as yf

from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

# Yahoo Finance tickers
TICKERS = {
    "usd_inr": "INR=X",
    "brent_crude": "BZ=F",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow_jones": "^DJI",
    "nikkei": "^N225",
    "hang_seng": "^HSI",
    "gold": "GC=F",
    "silver": "SI=F",
    "nifty_50": "^NSEI",
    "sensex": "^BSESN",
    "gift_nifty": "^NSEI",  # Proxy — GIFT Nifty not directly on yfinance
}


def scrape_global_markets() -> dict:
    """Scrape global market data from Yahoo Finance.

    Returns dict with normalized values for USD/INR, crude, S&P 500 change, etc.
    """
    result: dict = {}

    # Fetch individual tickers with error handling
    ticker_map = {
        "usd_inr": "INR=X",
        "brent_crude": "BZ=F",
        "sp500": "^GSPC",
        "nifty_50": "^NSEI",
        "sensex": "^BSESN",
        "nikkei": "^N225",
        "hang_seng": "^HSI",
    }

    for key, symbol in ticker_map.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue

            latest = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2] if len(hist) >= 2 else latest
            change_pct = ((latest - prev) / prev) * 100 if prev else 0

            result[key] = latest
            result[f"{key}_change_pct"] = change_pct

        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")

    # Validate known ranges
    if "usd_inr" in result:
        result["usd_inr"] = validate_value("usd_inr", result["usd_inr"])
    if "brent_crude" in result:
        result["brent_crude"] = validate_value("brent_crude", result["brent_crude"])

    # Global overnight composite (weighted)
    global_changes = []
    weights = {"sp500_change_pct": 0.4, "nikkei_change_pct": 0.2, "hang_seng_change_pct": 0.2, "nifty_50_change_pct": 0.2}
    weighted_sum = 0
    total_weight = 0
    for change_key, weight in weights.items():
        if change_key in result:
            weighted_sum += result[change_key] * weight
            total_weight += weight

    if total_weight > 0:
        result["global_overnight_change"] = weighted_sum / total_weight

    logger.info(f"Global markets: {len(result)} values scraped")
    return result


def scrape_nifty_history(period: str = "1y") -> dict:
    """Fetch Nifty 50 historical OHLCV data for technical indicators."""
    try:
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period=period)
        return {"history": hist, "error": None}
    except Exception as e:
        return {"history": None, "error": str(e)}
