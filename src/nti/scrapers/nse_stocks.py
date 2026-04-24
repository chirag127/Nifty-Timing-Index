"""NSE Stock Universe Scraper — Full equity list from EQUITY_L.csv.

Downloads the complete list of all NSE-listed equities, which forms the
base universe for the stock screener.
"""

from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def _create_nse_session() -> requests.Session:
    """Create a requests session with NSE-appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception:
        logger.warning("Could not establish NSE session cookies")
    return session


def scrape_nse_equity_list() -> pd.DataFrame:
    """Download the full NSE equity list (EQUITY_L.csv).

    Returns:
        DataFrame with columns: SYMBOL, NAME_OF_COMPANY, SERIES, ISIN_NUMBER,
        DATE_OF_LISTING, PAID_UP_VALUE, MARKET_LOT, FACE_VALUE
    """
    session = _create_nse_session()

    try:
        resp = session.get(NSE_EQUITY_LIST_URL, timeout=30)

        if resp.status_code != 200:
            logger.warning(f"NSE equity list URL returned {resp.status_code}")
            return pd.DataFrame()

        df = pd.read_csv(io.BytesIO(resp.content))

        # Standardize column names (NSE uses various formats)
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

        # Filter to EQ series only (remove ETFs, bonds, rights, etc.)
        if "SERIES" in df.columns:
            eq_df = df[df["SERIES"] == "EQ"].copy()
        else:
            eq_df = df.copy()

        logger.info(f"NSE equity list: {len(df)} total, {len(eq_df)} EQ series stocks")
        return eq_df

    except Exception as e:
        logger.warning(f"NSE equity list scrape failed: {e}")
        return pd.DataFrame()


def get_nse_stock_symbols() -> list[str]:
    """Get list of all NSE EQ-series stock symbols.

    Returns:
        List of symbol strings (e.g., ["SBIN", "TCS", "INFY", ...])
    """
    df = scrape_nse_equity_list()
    if df.empty or "SYMBOL" not in df.columns:
        return []
    return df["SYMBOL"].tolist()
