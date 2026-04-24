"""Feature Engineer — Build ML feature vector from normalized indicators.

Produces the 26 engineered features used by the stacked ensemble model:
- 6 Fundamental features
- 5 Sentiment features
- 5 Macro features
- 2 Flow features
- 2 LLM features
- 6 Lagged + derived features
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from nti.indicators.normalizer import normalize_all_indicators
from nti.indicators.composite import compute_custom_fg_composite, compute_global_overnight_composite

logger = logging.getLogger(__name__)

# Ordered list of all model features
MODEL_FEATURES = [
    # Fundamental (6 features)
    "nifty_pe_normalized",
    "nifty_pb_normalized",
    "earnings_yield_bond_spread",
    "dividend_yield_normalized",
    "mcap_to_gdp_percentile",
    "midcap_pe_normalized",
    # Sentiment (5 features)
    "mmi_score",
    "vix_normalized",
    "pcr_normalized",
    "custom_fg_composite",
    "fii_cash_5d_avg_normalized",
    # Macro (5 features)
    "rbi_rate_direction",
    "cpi_normalized",
    "us_10y_normalized",
    "usdinr_30d_change",
    "crude_normalized",
    # Flows (2 features)
    "fii_fo_net_normalized",
    "dii_net_normalized",
    # LLM (2 features)
    "llm_news_danger_score",
    "global_overnight_normalized",
    # Lagged + derived (6 features)
    "nti_score_lag1",
    "nti_score_lag24",
    "pe_5d_change_normalized",
    "vix_5d_change",
    "day_of_week",
    "days_to_monthly_expiry",
]


def build_feature_vector(
    raw_indicators: dict,
    previous_score: float | None = None,
    score_yesterday: float | None = None,
    pe_5d_ago: float | None = None,
    vix_5d_ago: float | None = None,
) -> dict[str, float]:
    """Build the complete ML feature vector from raw indicators.

    Args:
        raw_indicators: Dict of raw indicator values from scrapers
        previous_score: Previous hourly NTI score (for lag1)
        score_yesterday: Score from same time yesterday (for lag24)
        pe_5d_ago: Nifty PE 5 days ago (for 5d change)
        vix_5d_ago: India VIX 5 days ago (for 5d change)

    Returns:
        Dict of feature_name → float value, with None for missing features
    """
    # Step 1: Normalize all raw indicators
    normalized = normalize_all_indicators(raw_indicators)

    # Step 2: Compute composites
    vix_norm = normalized.get("vix_normalized")
    pcr_norm = normalized.get("pcr_normalized")
    ad_ratio = raw_indicators.get("advance_decline_ratio")
    hl_ratio = raw_indicators.get("high_low_ratio")

    fg_composite = compute_custom_fg_composite(
        vix_normalized=vix_norm,
        pcr_normalized=pcr_norm,
        advance_decline_ratio=ad_ratio,
        high_low_ratio=hl_ratio,
    )
    if fg_composite is not None:
        normalized["custom_fg_composite"] = fg_composite

    # Global overnight composite
    global_overnight = compute_global_overnight_composite(
        dow_change_pct=raw_indicators.get("sp500_change_pct"),
        nasdaq_change_pct=raw_indicators.get("nasdaq_change_pct"),
        nikkei_change_pct=raw_indicators.get("nikkei_change_pct"),
        hang_seng_change_pct=raw_indicators.get("hang_seng_change_pct"),
    )
    if global_overnight is not None:
        normalized["global_overnight_normalized"] = global_overnight

    # Step 3: Add lagged features
    normalized["nti_score_lag1"] = previous_score if previous_score is not None else 50.0
    normalized["nti_score_lag24"] = score_yesterday if score_yesterday is not None else 50.0

    # PE 5-day change
    current_pe = raw_indicators.get("nifty_pe")
    if current_pe is not None and pe_5d_ago is not None and pe_5d_ago > 0:
        pe_change = (float(current_pe) - float(pe_5d_ago)) / float(pe_5d_ago) * 100
        normalized["pe_5d_change_normalized"] = max(0.0, min(100.0, 50.0 + pe_change * 5))
    else:
        normalized["pe_5d_change_normalized"] = 50.0  # Neutral

    # VIX 5-day change
    current_vix = raw_indicators.get("india_vix")
    if current_vix is not None and vix_5d_ago is not None and vix_5d_ago > 0:
        vix_change = (float(current_vix) - float(vix_5d_ago)) / float(vix_5d_ago) * 100
        normalized["vix_5d_change"] = max(-50.0, min(50.0, vix_change))
    else:
        normalized["vix_5d_change"] = 0.0  # Neutral

    # Calendar features
    today = date.today()
    normalized["day_of_week"] = float(today.weekday())  # 0=Mon to 4=Fri

    # Days to monthly F&O expiry (last Thursday of month)
    normalized["days_to_monthly_expiry"] = float(_days_to_expiry(today))

    # Step 4: Build final feature vector (only MODEL_FEATURES, in order)
    features: dict[str, float] = {}
    for feat in MODEL_FEATURES:
        val = normalized.get(feat)
        if val is not None:
            features[feat] = float(val)
        else:
            features[feat] = 50.0  # Neutral default for missing features

    logger.info(f"Feature vector: {len(features)} features built ({sum(1 for v in features.values() if v != 50.0)} non-neutral)")
    return features


def _days_to_expiry(today: date) -> int:
    """Calculate days to next NSE monthly F&O expiry (last Thursday of month).

    If today is past this month's expiry, use next month's expiry.
    """
    # Find last Thursday of current month
    if today.month == 12:
        next_month = date(today.year + 1, 1, 1)
    else:
        next_month = date(today.year, today.month + 1, 1)

    # Last day of current month
    last_day = next_month - timedelta(days=1)

    # Find last Thursday
    offset = (last_day.weekday() - 3) % 7  # 3 = Thursday
    last_thursday = last_day - timedelta(days=offset)

    if today <= last_thursday:
        return (last_thursday - today).days
    else:
        # Use next month's expiry
        if today.month == 12:
            next_expiry_month = date(today.year + 1, 1, 1)
        else:
            next_expiry_month = date(today.year, today.month + 1, 1)

        if next_expiry_month.month == 12:
            after_next = date(next_expiry_month.year + 1, 1, 1)
        else:
            after_next = date(next_expiry_month.year, next_expiry_month.month + 1, 1)

        last_day_next = after_next - timedelta(days=1)
        offset = (last_day_next.weekday() - 3) % 7
        next_last_thursday = last_day_next - timedelta(days=offset)

        return (next_last_thursday - today).days
