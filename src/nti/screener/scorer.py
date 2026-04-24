"""Stock Scorer — Composite score computation with PSU boost.

Scoring formula:
    Value Score (50%): PE rank + PB rank + dividend yield rank
    Quality Score (30%): ROE rank + debt rank
    Analyst Score (20%): analyst buy% + target upside
    PSU Bonus: +10 added to composite score if PSU stock

Final composite score is 0–100 (capped).
"""

from __future__ import annotations

import logging

import pandas as pd

from nti.config.settings import settings
from nti.config.psu_stocks import get_psu_score_boost

logger = logging.getLogger(__name__)


def compute_composite_scores(stocks: list[dict]) -> list[dict]:
    """Compute composite scores for all stocks that passed hard filters.

    Args:
        stocks: List of stock dicts with fundamentals data

    Returns:
        List of stock dicts with added keys:
            value_score, quality_score, analyst_score, composite_score, psu_boost
    """
    if not stocks:
        return []

    df = pd.DataFrame(stocks)

    # --- Value Score (50% weight) ---
    # Lower PE = better value → higher score
    # Lower PB = better value → higher score
    # Higher dividend yield = better value → higher score
    df["pe_rank"] = df["pe"].rank(ascending=True, method="min", na_option="bottom")
    df["pb_rank"] = df["pb"].rank(ascending=True, method="min", na_option="bottom")
    df["dy_rank"] = df["dividend_yield"].rank(ascending=False, method="min", na_option="bottom")

    # Normalize ranks to 0–100
    n = len(df)
    if n > 1:
        df["pe_score"] = (1 - df["pe_rank"] / n) * 100
        df["pb_score"] = (1 - df["pb_rank"] / n) * 100
        df["dy_score"] = (1 - df["dy_rank"] / n) * 100
    else:
        df["pe_score"] = 50.0
        df["pb_score"] = 50.0
        df["dy_score"] = 50.0

    df["value_score"] = (df["pe_score"] * 0.4 + df["pb_score"] * 0.3 + df["dy_score"] * 0.3).round(1)

    # --- Quality Score (30% weight) ---
    # Higher ROE = better quality → higher score
    # Lower Debt/Equity = better quality → higher score
    # Support both 'roe' (from fundamentals) and 'roe_pct' keys
    roe_col = "roe_pct" if "roe_pct" in df.columns else "roe"
    df["roe_rank"] = df[roe_col].rank(ascending=False, method="min", na_option="bottom")
    df["de_rank"] = df["debt_equity"].rank(ascending=True, method="min", na_option="bottom")

    if n > 1:
        df["roe_score"] = (1 - df["roe_rank"] / n) * 100
        df["de_score"] = (1 - df["de_rank"] / n) * 100
    else:
        df["roe_score"] = 50.0
        df["de_score"] = 50.0

    df["quality_score"] = (df["roe_score"] * 0.6 + df["de_score"] * 0.4).round(1)

    # --- Analyst Score (20% weight) ---
    # Default to 50 if no analyst data
    df["analyst_score"] = df.get("analyst_buy_pct", 50.0)
    if isinstance(df["analyst_score"], pd.Series):
        df["analyst_score"] = df["analyst_score"].fillna(50.0).round(1)
    else:
        df["analyst_score"] = 50.0

    # --- Composite Score ---
    df["composite_raw"] = (
        df["value_score"] * 0.50 +
        df["quality_score"] * 0.30 +
        df["analyst_score"] * 0.20
    ).round(1)

    # --- PSU Boost ---
    df["psu_boost"] = df["symbol"].apply(
        lambda s: settings.psu_boost_score if get_psu_score_boost(s) > 0 else 0.0
    )

    df["composite_score"] = (df["composite_raw"] + df["psu_boost"]).clip(upper=100).round(1)

    # Sort by composite score descending
    df = df.sort_values("composite_score", ascending=False)

    # Add rank
    df["rank"] = range(1, len(df) + 1)

    # Convert back to list of dicts
    result = df.to_dict("records")

    # Clean up temporary columns
    clean_keys = {"pe_rank", "pb_rank", "dy_rank", "pe_score", "pb_score", "dy_score",
                   "roe_rank", "de_rank", "roe_score", "de_score", "composite_raw"}
    for stock in result:
        for key in clean_keys:
            stock.pop(key, None)

    psu_count = sum(1 for s in result if s.get("is_psu"))
    logger.info(f"Scored {len(result)} stocks ({psu_count} PSU stocks)")

    return result


def get_top_picks(stocks: list[dict], top_n: int = 50) -> list[dict]:
    """Get top N stock picks sorted by composite score.

    Args:
        stocks: List of scored stock dicts
        top_n: Number of top picks to return

    Returns:
        List of top N stock dicts
    """
    sorted_stocks = sorted(stocks, key=lambda s: s.get("composite_score", 0), reverse=True)
    return sorted_stocks[:top_n]
