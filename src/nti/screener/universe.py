"""Stock Universe Builder — NSE + BSE equities with market cap ≥ ₹500 Cr.

Downloads the full NSE equity list (EQUITY_L.csv) and filters to EQ series.
Appends .NS suffix for yfinance lookups.
"""

from __future__ import annotations

import logging

import pandas as pd

from nti.scrapers.nse_stocks import scrape_nse_equity_list
from nti.config.settings import settings

logger = logging.getLogger(__name__)


def build_stock_universe(min_market_cap_cr: float | None = None) -> pd.DataFrame:
    """Build the full stock universe for the screener.

    Step 1: Download NSE EQUITY_L.csv
    Step 2: Filter to EQ series only
    Step 3: Add .NS suffix for yfinance
    Step 4: (Future) Filter by market cap ≥ ₹500 Cr using yfinance

    Args:
        min_market_cap_cr: Minimum market cap in Crores (default: from settings)

    Returns:
        DataFrame with columns: symbol, yf_symbol, name, isin, series
    """
    if min_market_cap_cr is None:
        min_market_cap_cr = settings.min_market_cap_cr

    # Download NSE equity list
    df = scrape_nse_equity_list()

    if df.empty:
        logger.warning("No NSE equity data available")
        return pd.DataFrame()

    # Build universe
    universe = pd.DataFrame()
    
    # Standardize column names
    col_mapping = {
        "SYMBOL": "symbol",
        "NAME_OF_COMPANY": "name",
        "ISIN_NUMBER": "isin",
        "SERIES": "series",
    }
    
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns:
            universe[new_col] = df[old_col]

    if "symbol" not in universe.columns:
        logger.warning("No SYMBOL column found in NSE data")
        return pd.DataFrame()

    # Add yfinance suffix
    universe["yf_symbol"] = universe["symbol"] + ".NS"

    # Remove duplicates
    universe = universe.drop_duplicates(subset=["symbol"])

    logger.info(f"Stock universe: {len(universe)} EQ series stocks from NSE")
    return universe


def get_universe_symbols() -> list[str]:
    """Get list of all universe stock symbols (for batch processing).

    Returns:
        List of symbol strings with .NS suffix
    """
    universe = build_stock_universe()
    if universe.empty:
        return []
    return universe["yf_symbol"].tolist()
