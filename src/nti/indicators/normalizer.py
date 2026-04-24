"""Indicator Normalizer — Normalize all 30 raw indicators to 0–100 scale.

All indicators normalized so that:
- 0 = BUY opportunity (low danger)
- 100 = SELL danger (high danger)

This module imports the normalization functions from thresholds.py
and provides a unified interface to normalize an entire indicator dict.
"""

from __future__ import annotations

import logging
from typing import Any

from nti.config.thresholds import (
    normalize_pe,
    normalize_pb,
    normalize_dividend_yield,
    normalize_earnings_yield_spread,
    normalize_mcap_to_gdp,
    normalize_vix,
    normalize_pcr,
    normalize_cpi,
    normalize_us_10y,
    normalize_usdinr_change,
    normalize_crude,
    normalize_fii_cash,
    normalize_sip_flow,
    normalize_mmi,
)

logger = logging.getLogger(__name__)


def normalize_all_indicators(raw: dict[str, Any]) -> dict[str, float]:
    """Normalize all raw indicator values to 0–100 scale.

    Args:
        raw: Dict of raw indicator values from scrapers.
             Keys like 'nifty_pe', 'india_vix', 'pcr', etc.

    Returns:
        Dict of normalized scores (0–100) with '_normalized' suffix keys.
    """
    normalized: dict[str, float] = {}

    # --- Tier 1: Fundamental Valuation ---
    if raw.get("nifty_pe") is not None:
        normalized["nifty_pe_normalized"] = normalize_pe(float(raw["nifty_pe"]))

    if raw.get("nifty_pb") is not None:
        normalized["nifty_pb_normalized"] = normalize_pb(float(raw["nifty_pb"]))

    if raw.get("nifty_pe") is not None and raw.get("us_10y_yield") is not None:
        spread = (1.0 / float(raw["nifty_pe"])) - float(raw["us_10y_yield"])
        normalized["earnings_yield_bond_spread"] = normalize_earnings_yield_spread(spread)

    if raw.get("nifty_dy") is not None:
        normalized["dividend_yield_normalized"] = normalize_dividend_yield(float(raw["nifty_dy"]))

    if raw.get("mcap_to_gdp") is not None:
        normalized["mcap_to_gdp_percentile"] = normalize_mcap_to_gdp(float(raw["mcap_to_gdp"]))

    if raw.get("midcap_pe") is not None:
        normalized["midcap_pe_normalized"] = normalize_pe(float(raw["midcap_pe"]))

    # --- Tier 2: Sentiment (Contrarian) ---
    if raw.get("mmi_value") is not None:
        normalized["mmi_score"] = normalize_mmi(float(raw["mmi_value"]))

    if raw.get("india_vix") is not None:
        normalized["vix_normalized"] = normalize_vix(float(raw["india_vix"]))

    if raw.get("pcr") is not None:
        normalized["pcr_normalized"] = normalize_pcr(float(raw["pcr"]))

    # Custom F&G composite computed separately in composite.py

    if raw.get("fii_cash_net") is not None:
        normalized["fii_cash_5d_avg_normalized"] = normalize_fii_cash(float(raw["fii_cash_net"]))

    if raw.get("cnn_fg_value") is not None:
        # CNN F&G is already 0-100, map directly (greed=high=danger)
        normalized["cnn_fg_normalized"] = float(raw["cnn_fg_value"])

    # --- Tier 3: Macro ---
    if raw.get("rbi_direction") is not None:
        normalized["rbi_rate_direction"] = float(raw["rbi_direction"])

    if raw.get("cpi_inflation") is not None:
        normalized["cpi_normalized"] = normalize_cpi(float(raw["cpi_inflation"]))

    if raw.get("us_10y_yield") is not None:
        normalized["us_10y_normalized"] = normalize_us_10y(float(raw["us_10y_yield"]))

    if raw.get("usdinr_30d_change") is not None:
        normalized["usdinr_30d_change"] = normalize_usdinr_change(float(raw["usdinr_30d_change"]))
    elif raw.get("usd_inr") is not None:
        # If no 30d change, use current level as proxy
        normalized["usdinr_30d_change"] = normalize_usdinr_change(0.0)  # Neutral

    if raw.get("brent_crude") is not None:
        normalized["crude_normalized"] = normalize_crude(float(raw["brent_crude"]))

    if raw.get("sp500_change_pct") is not None:
        # S&P 500 change: positive = bullish global = lower danger for India
        sp500_change = float(raw["sp500_change_pct"])
        normalized["sp500_change_normalized"] = max(0.0, min(100.0, 50.0 - sp500_change * 10))

    # --- Tier 4: Institutional Flow ---
    if raw.get("fii_fo_index_futures_net") is not None:
        normalized["fii_fo_net_normalized"] = normalize_fii_cash(float(raw["fii_fo_index_futures_net"]))

    if raw.get("dii_cash_net") is not None:
        # DII buying = support = lower danger (inverse of FII)
        dii_net = float(raw["dii_cash_net"])
        # DII buying is positive for market, so inverse
        normalized["dii_net_normalized"] = max(0.0, min(100.0, 50.0 - dii_net / 100.0))

    if raw.get("sip_flow_monthly_cr") is not None:
        normalized["sip_flow_normalized"] = normalize_sip_flow(float(raw["sip_flow_monthly_cr"]))

    if raw.get("gift_nifty_change_pct") is not None:
        # GIFT Nifty positive change = lower danger
        gift_change = float(raw["gift_nifty_change_pct"])
        normalized["gift_nifty_change_normalized"] = max(0.0, min(100.0, 50.0 - gift_change * 10))

    # --- Tier 5: LLM News ---
    if raw.get("llm_news_danger_score") is not None:
        normalized["llm_news_danger_score"] = float(raw["llm_news_danger_score"])

    if raw.get("global_overnight_normalized") is not None:
        normalized["global_overnight_normalized"] = float(raw["global_overnight_normalized"])

    # --- Tier 6: Display Only (0% weight, not in model) ---
    if raw.get("rsi_14") is not None:
        rsi = float(raw["rsi_14"])
        # RSI > 70 = overbought = danger, RSI < 30 = oversold = opportunity
        normalized["rsi_14_normalized"] = max(0.0, min(100.0, rsi))

    # Log summary
    if normalized:
        logger.info(f"Normalized {len(normalized)} indicators")

    return normalized


def compute_rule_based_score(normalized: dict[str, float]) -> float:
    """Compute rule-based weighted average NTI score from normalized indicators.

    This is the cold-start fallback when no ML model is available.
    Uses the same weights as defined in the 30-parameter framework.
    """
    weights: dict[str, float] = {
        # Tier 1: Fundamental Valuation (35% total)
        "nifty_pe_normalized": 0.10,
        "nifty_pb_normalized": 0.08,
        "earnings_yield_bond_spread": 0.06,
        "dividend_yield_normalized": 0.05,
        "mcap_to_gdp_percentile": 0.04,
        "midcap_pe_normalized": 0.02,
        # Tier 2: Sentiment (25% total)
        "mmi_score": 0.08,
        "vix_normalized": 0.06,
        "pcr_normalized": 0.04,
        "custom_fg_composite": 0.04,
        "fii_cash_5d_avg_normalized": 0.01,
        # Tier 3: Macro (25% total)
        "rbi_rate_direction": 0.06,
        "cpi_normalized": 0.05,
        "us_10y_normalized": 0.05,
        "usdinr_30d_change": 0.04,
        "crude_normalized": 0.03,
        "sp500_change_normalized": 0.02,
        # Tier 4: Institutional Flow (10% total)
        "fii_fo_net_normalized": 0.04,
        "dii_net_normalized": 0.03,
        "sip_flow_normalized": 0.02,
        "gift_nifty_change_normalized": 0.01,
        # Tier 5: LLM News (5% total)
        "llm_news_danger_score": 0.02,
        "global_overnight_normalized": 0.01,
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for key, weight in weights.items():
        value = normalized.get(key)
        if value is not None:
            weighted_sum += value * weight
            total_weight += weight

    if total_weight == 0:
        return 50.0  # Neutral if no data

    # Normalize to account for missing indicators
    score = weighted_sum / total_weight * (sum(weights.values()) / total_weight)
    return max(0.0, min(100.0, score))
