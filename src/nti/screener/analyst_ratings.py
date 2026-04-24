"""Analyst Ratings Fetcher — Finnhub free API (60 calls/min).

Provides analyst buy/sell percentages for stock scoring.
"""

from __future__ import annotations

import logging

import httpx

from nti.config.settings import settings

logger = logging.getLogger(__name__)

FINNHUB_URL = "https://finnhub.io/api/v1/stock/recommendation"


def fetch_analyst_ratings(symbol: str) -> dict:
    """Fetch analyst ratings for a stock from Finnhub.

    Note: Finnhub uses different symbol formats for Indian stocks.
    Some Indian stocks may not be available.

    Args:
        symbol: Stock symbol (e.g., "SBIN")

    Returns:
        dict with keys: analyst_buy_pct, analyst_count, strong_buy, buy, hold, sell, strong_sell
    """
    if not settings.finnhub_api_key:
        return _default_ratings()

    # Try both formats: with and without .NS suffix
    for sym in [symbol, f"{symbol}.NS"]:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    FINNHUB_URL,
                    params={
                        "symbol": sym,
                        "token": settings.finnhub_api_key,
                    },
                )

                if resp.status_code != 200:
                    continue

                data = resp.json()
                if not data:
                    continue

                latest = data[0]
                total = (
                    latest.get("strongBuy", 0)
                    + latest.get("buy", 0)
                    + latest.get("hold", 0)
                    + latest.get("sell", 0)
                    + latest.get("strongSell", 0)
                )

                if total == 0:
                    continue

                buy_pct = (latest.get("strongBuy", 0) + latest.get("buy", 0)) / total * 100

                return {
                    "analyst_buy_pct": round(buy_pct, 1),
                    "analyst_count": total,
                    "strong_buy": latest.get("strongBuy", 0),
                    "buy": latest.get("buy", 0),
                    "hold": latest.get("hold", 0),
                    "sell": latest.get("sell", 0),
                    "strong_sell": latest.get("strongSell", 0),
                }

        except Exception as e:
            logger.warning(f"Finnhub ratings failed for {sym}: {e}")

    return _default_ratings()


def _default_ratings() -> dict:
    """Default analyst ratings when no data available."""
    return {
        "analyst_buy_pct": 50.0,
        "analyst_count": 0,
        "strong_buy": 0,
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "strong_sell": 0,
    }


def batch_fetch_ratings(symbols: list[str], max_concurrent: int = 10) -> dict[str, dict]:
    """Fetch analyst ratings for multiple stocks.

    Args:
        symbols: List of stock symbols
        max_concurrent: Maximum concurrent requests

    Returns:
        Dict of symbol → ratings dict
    """
    results = {}

    for i, symbol in enumerate(symbols):
        if i > 0 and i % 60 == 0:
            # Finnhub rate limit: 60 calls/min
            import time
            logger.info("Finnhub rate limit: pausing 60 seconds")
            time.sleep(60)

        results[symbol] = fetch_analyst_ratings(symbol)

    rated_count = sum(1 for v in results.values() if v["analyst_count"] > 0)
    logger.info(f"Analyst ratings: {rated_count}/{len(symbols)} stocks with data")

    return results
