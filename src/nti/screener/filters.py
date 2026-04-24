"""Stock Screener Filters — Hard and soft filters.

Hard filters (must pass ALL):
- PE < 20.0 (strictly, None PE = excluded)
- PB < 3.0 (strictly, None PB = excluded)
- Market Cap ≥ ₹500 Cr
- PE > 0 and PB > 0 (negative = excluded)
- Price > 0

Soft filters (shown as warnings but NOT excluded):
- ROE > 12%
- Debt/Equity < 1.5 (or < 1.0 for non-banks)
"""

from __future__ import annotations

import logging

from nti.config.settings import settings
from nti.config.psu_stocks import is_psu

logger = logging.getLogger(__name__)


def passes_hard_filters(stock: dict) -> bool:
    """Check if a stock passes ALL hard filters.

    Args:
        stock: Dict with keys: symbol, pe, pb, market_cap_cr, current_price

    Returns:
        True if stock passes all hard filters
    """
    max_pe = settings.max_pe
    max_pb = settings.max_pb
    min_mcap = settings.min_market_cap_cr

    # PE filter: must exist, be positive, and below max
    pe = stock.get("pe")
    if pe is None or pe <= 0 or pe >= max_pe:
        return False

    # PB filter: must exist, be positive, and below max
    pb = stock.get("pb")
    if pb is None or pb <= 0 or pb >= max_pb:
        return False

    # Market cap filter: must meet minimum
    mcap = stock.get("market_cap_cr", 0)
    if mcap < min_mcap:
        return False

    # Price must be positive
    price = stock.get("current_price")
    if price is not None and price <= 0:
        return False

    return True


def get_soft_warnings(stock: dict) -> list[str]:
    """Get soft filter warnings for a stock.

    These don't exclude the stock but show warnings in the UI.

    Args:
        stock: Dict with keys: symbol, roe, debt_equity, industry

    Returns:
        List of warning strings
    """
    warnings = []

    # ROE warning (support both 'roe' and 'roe_pct' keys)
    roe = stock.get("roe_pct") if stock.get("roe_pct") is not None else stock.get("roe")
    if roe is not None and roe < 12.0:
        warnings.append(f"Low ROE: {roe:.1f}% (< 12%)")

    # Debt/Equity warning
    de = stock.get("debt_equity")
    industry = stock.get("industry", "").lower()
    is_bank = "bank" in industry or "financial" in industry

    if de is not None:
        threshold = 1.5 if is_bank else 1.0
        if de > threshold:
            warnings.append(f"High Debt/Equity: {de:.1f} (> {threshold})")

    return warnings


def apply_all_filters(stocks: list[dict]) -> tuple[list[dict], dict]:
    """Apply hard filters and soft warnings to a list of stocks.

    Args:
        stocks: List of stock dicts with fundamental data

    Returns:
        Tuple of:
        - List of stocks that pass hard filters (with 'warnings' key added)
        - Summary dict with exclusion counts
    """
    passing = []
    exclusion_summary = {
        "pe_too_high": 0,
        "pb_too_high": 0,
        "market_cap_too_small": 0,
        "missing_data": 0,
        "negative_pe": 0,
        "negative_pb": 0,
    }

    for stock in stocks:
        # Check each filter individually for tracking
        pe = stock.get("pe")
        pb = stock.get("pb")
        mcap = stock.get("market_cap_cr", 0)

        if pe is None:
            exclusion_summary["missing_data"] += 1
            continue
        if pe <= 0:
            exclusion_summary["negative_pe"] += 1
            continue
        if pe >= settings.max_pe:
            exclusion_summary["pe_too_high"] += 1
            continue

        if pb is None:
            exclusion_summary["missing_data"] += 1
            continue
        if pb <= 0:
            exclusion_summary["negative_pb"] += 1
            continue
        if pb >= settings.max_pb:
            exclusion_summary["pb_too_high"] += 1
            continue

        if mcap < settings.min_market_cap_cr:
            exclusion_summary["market_cap_too_small"] += 1
            continue

        # Passed all hard filters
        stock["warnings"] = get_soft_warnings(stock)
        stock["is_psu"] = is_psu(stock.get("symbol", ""))
        passing.append(stock)

    logger.info(
        f"Filters: {len(passing)}/{len(stocks)} stocks passed — "
        f"excluded: {exclusion_summary}"
    )

    return passing, exclusion_summary
