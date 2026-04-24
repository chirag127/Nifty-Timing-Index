"""Display-Only Technical Indicators — RSI, MACD.

These are computed and shown in the dashboard/blog but have 0% weight
in the ML model. They provide additional context for readers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    """Compute Relative Strength Index (RSI) from a price series.

    Args:
        series: pandas Series of closing prices (at least period+1 values)
        period: RSI period (default 14)

    Returns:
        RSI value (0–100), or None if insufficient data
    """
    try:
        if len(series) < period + 1:
            return None

        delta = series.diff().dropna()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use exponential moving average for subsequent values
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        latest_rsi = rsi.iloc[-1]

        if pd.notna(latest_rsi):
            return float(latest_rsi)
        return None

    except Exception as e:
        logger.warning(f"RSI computation failed: {e}")
        return None


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict:
    """Compute MACD (Moving Average Convergence Divergence) from a price series.

    Args:
        series: pandas Series of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        dict with keys: macd_line, signal_line, histogram
        All values may be None if insufficient data
    """
    result: dict = {
        "macd_line": None,
        "signal_line": None,
        "histogram": None,
    }

    try:
        if len(series) < slow_period + signal_period:
            return result

        # Compute EMAs
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()

        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema

        # Signal line = EMA of MACD line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram = MACD - Signal
        histogram = macd_line - signal_line

        # Latest values
        if pd.notna(macd_line.iloc[-1]):
            result["macd_line"] = float(macd_line.iloc[-1])
        if pd.notna(signal_line.iloc[-1]):
            result["signal_line"] = float(signal_line.iloc[-1])
        if pd.notna(histogram.iloc[-1]):
            result["histogram"] = float(histogram.iloc[-1])

    except Exception as e:
        logger.warning(f"MACD computation failed: {e}")

    return result


def compute_advance_decline_ratio(raw: dict) -> float | None:
    """Compute Advance/Decline ratio from NSE market stats.

    Args:
        raw: Dict with 'advances' and 'declines' keys

    Returns:
        A/D ratio (e.g., 2.5), or None if data unavailable
    """
    advances = raw.get("advances")
    declines = raw.get("declines")

    if advances is not None and declines is not None and int(declines) > 0:
        return float(advances) / float(declines)
    return None


def compute_52wk_high_low_ratio(raw: dict) -> float | None:
    """Compute 52-week Highs vs Lows ratio.

    Args:
        raw: Dict with 'new_highs' and 'new_lows' keys

    Returns:
        Highs/Lows ratio, or None if data unavailable
    """
    new_highs = raw.get("new_highs")
    new_lows = raw.get("new_lows")

    if new_highs is not None and new_lows is not None and int(new_lows) > 0:
        return float(new_highs) / float(new_lows)
    if new_highs is not None and new_lows is not None and int(new_lows) == 0:
        return 10.0  # All highs, no lows = extreme greed
    return None
