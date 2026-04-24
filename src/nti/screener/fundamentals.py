"""Fundamentals Fetcher — Batch fetch PE, PB, ROE, market cap via yfinance.

Rate-limited to respect yfinance's informal limits.
Batched in groups of 50 with 2-second delays between batches.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import yfinance as yf

from nti.config.settings import settings

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Safely coerce a value to float, returning default if not possible.

    yfinance sometimes returns strings or unexpected types for numeric fields.
    This ensures all numeric fields are properly typed before filtering.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

BATCH_SIZE = 50
BATCH_DELAY = 2.0  # seconds between batches


def fetch_stock_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch fundamental data for a single stock via yfinance.

    Args:
        symbol: Stock symbol WITHOUT .NS suffix (e.g., "SBIN")

    Returns:
        dict with keys: symbol, pe, pb, market_cap_cr, dividend_yield,
                         roe, debt_equity, current_price, sector, industry
    """
    yf_symbol = f"{symbol}.NS"

    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info

        market_cap = _safe_float(info.get("marketCap"), 0) or 0
        market_cap_cr = market_cap / 1e7 if market_cap else 0  # Convert to Crores

        return {
            "symbol": symbol,
            "pe": _safe_float(info.get("trailingPE")),
            "pb": _safe_float(info.get("priceToBook")),
            "market_cap_cr": round(market_cap_cr, 1),
            "dividend_yield": (_safe_float(info.get("dividendYield"), 0) or 0) * 100,
            "roe": (_safe_float(info.get("returnOnEquity"), 0) or 0) * 100,
            "debt_equity": _safe_float(info.get("debtToEquity")),
            "current_price": _safe_float(info.get("currentPrice")),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }

    except Exception as e:
        logger.warning(f"yfinance failed for {yf_symbol}: {e}")
        return {
            "symbol": symbol,
            "pe": None,
            "pb": None,
            "market_cap_cr": 0,
            "dividend_yield": 0,
            "roe": 0,
            "debt_equity": None,
            "current_price": None,
            "sector": "",
            "industry": "",
        }


def batch_fetch_fundamentals(symbols: list[str]) -> list[dict[str, Any]]:
    """Fetch fundamentals for a batch of stocks with rate limiting.

    Args:
        symbols: List of stock symbols (WITHOUT .NS suffix)

    Returns:
        List of fundamental dicts (one per symbol)
    """
    results = []
    total = len(symbols)

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        logger.info(f"Fetching fundamentals: batch {i // BATCH_SIZE + 1} ({len(batch)} stocks)")

        for symbol in batch:
            fund = fetch_stock_fundamentals(symbol)
            results.append(fund)

        # Rate limiting between batches
        if i + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)

    valid_count = sum(1 for r in results if r.get("pe") is not None)
    logger.info(f"Fundamentals fetched: {valid_count}/{total} stocks with valid PE")
    return results
