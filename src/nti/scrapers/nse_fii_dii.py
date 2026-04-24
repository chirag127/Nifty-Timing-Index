"""NSE FII/DII Daily Flow Scraper — Cash + F&O net positions.

Scrapes the participant-wise trading volume CSV from NSE archives.
FII and DII net cash flow is a key sentiment indicator.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date

import pandas as pd
import requests

from nti.config.thresholds import validate_value

logger = logging.getLogger(__name__)

# NSE participant-wise volume CSV URL
FII_DII_URL = "https://archives.nseindia.com/content/nsccl/fao_participant_vol.csv"
FII_CASH_URL = "https://archives.nseindia.com/content/nsccl/fii_participant.csv"


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


def scrape_fii_dii_cash_flow() -> dict:
    """Scrape FII and DII daily net cash flow from NSE.

    Returns dict with keys:
        fii_cash_net: float (Crores, negative = net selling)
        dii_cash_net: float (Crores, negative = net selling)
        fii_cash_buy: float
        fii_cash_sell: float
        dii_cash_buy: float
        dii_cash_sell: float
    """
    result: dict = {
        "fii_cash_net": None,
        "dii_cash_net": None,
        "fii_cash_buy": None,
        "fii_cash_sell": None,
        "dii_cash_buy": None,
        "dii_cash_sell": None,
    }

    session = _create_nse_session()

    try:
        resp = session.get(FII_CASH_URL, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"FII cash URL returned {resp.status_code}")
            return result

        # Parse CSV — NSE format has header rows
        lines = resp.text.strip().split("\n")
        df = None
        for line in lines:
            if "FII" in line.upper() and "DII" in line.upper():
                break

        df = pd.read_csv(io.StringIO(resp.text), skiprows=1)

        # Find FII row
        for _, row in df.iterrows():
            name = str(row.iloc[0]).strip().upper()
            if "FII" in name or "FOREIGN" in name:
                try:
                    buy = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0
                    sell = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
                    net = float(row.iloc[3]) if pd.notna(row.iloc[3]) else (buy - sell)
                    result["fii_cash_buy"] = buy
                    result["fii_cash_sell"] = sell
                    result["fii_cash_net"] = validate_value("fii_cash_net", net)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing FII row: {e}")

            elif "DII" in name or "DOMESTIC" in name:
                try:
                    buy = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0
                    sell = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
                    net = float(row.iloc[3]) if pd.notna(row.iloc[3]) else (buy - sell)
                    result["dii_cash_buy"] = buy
                    result["dii_cash_sell"] = sell
                    result["dii_cash_net"] = validate_value("dii_cash_net", net)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing DII row: {e}")

        if result["fii_cash_net"] is not None:
            logger.info(f"FII net cash: ₹{result['fii_cash_net']:,.0f} Cr")
        if result["dii_cash_net"] is not None:
            logger.info(f"DII net cash: ₹{result['dii_cash_net']:,.0f} Cr")

    except Exception as e:
        logger.warning(f"FII/DII cash flow scrape failed: {e}")

    return result


def scrape_fii_fo_positions() -> dict:
    """Scrape FII F&O (Futures + Options) net positions from NSE.

    Returns dict with keys:
        fii_fo_index_futures_net: float (Crores)
        fii_fo_index_options_net: float
        fii_fo_stock_futures_net: float
        fii_fo_stock_options_net: float
    """
    result: dict = {
        "fii_fo_index_futures_net": None,
        "fii_fo_index_options_net": None,
        "fii_fo_stock_futures_net": None,
        "fii_fo_stock_options_net": None,
    }

    session = _create_nse_session()

    try:
        resp = session.get(FII_DII_URL, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"FII F&O URL returned {resp.status_code}")
            return result

        df = pd.read_csv(io.StringIO(resp.text))

        # Parse FII rows for different F&O segments
        for _, row in df.iterrows():
            name = str(row.iloc[0]).strip().upper()
            if "FII" not in name and "FOREIGN" not in name:
                continue

            # Try to identify the segment from column headers or row content
            row_str = str(row.values)
            if "INDEX FUTURES" in row_str.upper() or "INDEXFUT" in row_str.upper():
                try:
                    net = float(row.iloc[-1]) if pd.notna(row.iloc[-1]) else 0
                    result["fii_fo_index_futures_net"] = validate_value("fii_fo_net", net)
                except (ValueError, IndexError):
                    pass
            elif "INDEX OPTIONS" in row_str.upper() or "INDEXOPT" in row_str.upper():
                try:
                    net = float(row.iloc[-1]) if pd.notna(row.iloc[-1]) else 0
                    result["fii_fo_index_options_net"] = validate_value("fii_fo_net", net)
                except (ValueError, IndexError):
                    pass

    except Exception as e:
        logger.warning(f"FII F&O positions scrape failed: {e}")

    return result


def get_fii_5d_avg() -> dict:
    """Get 5-day average of FII net cash flow.

    For cold-start, this returns a default value.
    With accumulated data, reads from signal CSVs.
    """
    # Default — will be replaced by actual CSV reading once data accumulates
    return {"fii_cash_5d_avg": None}
