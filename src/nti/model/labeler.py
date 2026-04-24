"""Training Label Generator — Create binary labels from Nifty forward returns.

Labels based on Nifty 50 forward return over next 5 trading days:
- +2.5% or more → label 0 (it was a BUY moment, low danger)
- -2.5% or less → label 1 (it was a SELL moment, high danger)
- Between ±2.5% → None (excluded from training — too noisy)

Conservative threshold because user uses MTF leverage (3x = risk amplification).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Label thresholds (as fractions, not percentages)
BUY_THRESHOLD = 0.025   # +2.5% → label 0 (buy/low danger)
SELL_THRESHOLD = -0.025  # -2.5% → label 1 (sell/high danger)
FORWARD_DAYS = 5         # 5 trading days forward


def create_binary_label(future_5d_return: float) -> int | None:
    """Create a binary label from a 5-day forward return.

    Args:
        future_5d_return: Nifty 50 return over next 5 trading days (as fraction)

    Returns:
        0 = BUY (low danger), 1 = SELL (high danger), None = neutral (excluded)
    """
    if future_5d_return >= BUY_THRESHOLD:
        return 0  # It was a buy moment
    elif future_5d_return <= SELL_THRESHOLD:
        return 1  # It was a sell moment
    else:
        return None  # Neutral — exclude from training


def generate_labels_from_prices(
    nifty_close: pd.Series,
    forward_days: int = FORWARD_DAYS,
) -> pd.Series:
    """Generate training labels from Nifty closing prices.

    Computes 5-day forward return for each day and creates binary labels.

    Args:
        nifty_close: pandas Series of Nifty 50 daily closing prices
        forward_days: Number of forward trading days (default 5)

    Returns:
        pandas Series of labels (0, 1, or NaN for neutral/excluded)
    """
    # Compute forward return
    future_close = nifty_close.shift(-forward_days)
    forward_return = (future_close - nifty_close) / nifty_close

    # Create labels
    labels = forward_return.apply(create_binary_label)

    valid_count = labels.notna().sum()
    buy_count = (labels == 0).sum()
    sell_count = (labels == 1).sum()
    neutral_count = labels.isna().sum()

    logger.info(
        f"Labels generated: {valid_count} valid ({buy_count} buy, {sell_count} sell), "
        f"{neutral_count} neutral excluded"
    )

    return labels


def generate_labels_from_signal_csv(
    signal_csv_path: str,
    nifty_price_col: str = "nifty_price",
) -> pd.DataFrame:
    """Generate labels from an existing signal CSV file.

    Adds a 'label' column to the signal data based on forward returns.

    Args:
        signal_csv_path: Path to the hourly signal CSV
        nifty_price_col: Column name for Nifty price

    Returns:
        DataFrame with original columns + 'label' and 'forward_5d_return' columns
    """
    df = pd.read_csv(signal_csv_path)

    if nifty_price_col not in df.columns:
        logger.warning(f"Column '{nifty_price_col}' not found in {signal_csv_path}")
        return df

    prices = df[nifty_price_col]
    future_prices = prices.shift(-FORWARD_DAYS)
    forward_returns = (future_prices - prices) / prices

    df["forward_5d_return"] = forward_returns
    df["label"] = forward_returns.apply(create_binary_label)

    return df
