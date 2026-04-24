"""Composite Indicators — Custom Fear & Greed and Global Overnight composites.

Custom India F&G Composite:
  30% VIX + 30% PCR + 20% Market Breadth + 20% 52wk High/Low

Global Overnight Composite:
  Weighted average of Dow Jones + NASDAQ + Nikkei + Hang Seng changes
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_custom_fg_composite(
    indicators: dict | None = None,
    /,
    vix_normalized: float | None = None,
    pcr_normalized: float | None = None,
    advance_decline_ratio: float | None = None,
    high_low_ratio: float | None = None,
) -> float:
    """Compute custom India Fear & Greed composite (0–100).

    Weighted composition:
    - 30% India VIX normalized (contrarian: low VIX = complacency = greed = danger)
    - 30% PCR normalized (contrarian: low PCR = call-heavy = greed = danger)
    - 20% Advance/Decline ratio (high A/D = broad participation = greed)
    - 20% 52-week Highs vs Lows (more highs = greed = danger)

    Args:
        indicators: Dict with keys like vix_normalized, pcr_normalized,
                    adv_decline_ratio, high_low_ratio. If provided, used as
                    primary source; keyword args serve as overrides.
        vix_normalized: VIX normalized to 0–100 (from normalizer)
        pcr_normalized: PCR normalized to 0–100 (from normalizer)
        advance_decline_ratio: Raw A/D ratio (e.g., 2.5 = 2.5 advancers per decliner)
        high_low_ratio: Raw 52wk Highs/Lows ratio

    Returns:
        Composite score 0–100 (neutral 50 if insufficient data)
    """
    # Support both dict input and keyword args
    ind = indicators if indicators is not None else {}

    def _get(key: str, kwarg_val):
        """Prefer keyword arg if explicitly provided, else fall back to dict."""
        if kwarg_val is not None:
            return kwarg_val
        return ind.get(key)

    vix = _get("vix_normalized", vix_normalized)
    pcr = _get("pcr_normalized", pcr_normalized)
    ad  = _get("adv_decline_ratio", advance_decline_ratio)
    hl  = _get("high_low_ratio", high_low_ratio)

    components: list[tuple[float, float]] = []  # (value, weight)

    if vix is not None:
        components.append((float(vix), 0.30))

    if pcr is not None:
        components.append((float(pcr), 0.30))

    if ad is not None:
        # A/D ratio > 1.5 = broad advance = greed zone; < 0.5 = broad decline = fear
        ad_normalized = max(0.0, min(100.0, float(ad) / 3.0 * 100))
        components.append((ad_normalized, 0.20))

    if hl is not None:
        # More highs than lows = greed; normalize
        hl_normalized = max(0.0, min(100.0, float(hl) / 5.0 * 100))
        components.append((hl_normalized, 0.20))

    if not components:
        return 50.0  # Neutral if no data

    # Normalize weights to sum to 1.0
    total_weight = sum(w for _, w in components)
    composite = sum(v * w for v, w in components) / total_weight

    result = max(0.0, min(100.0, composite))
    logger.info(f"Custom F&G Composite: {result:.1f} (from {len(components)} components)")
    return result


def compute_global_overnight_composite(
    indicators: dict | None = None,
    /,
    dow_change_pct: float | None = None,
    nasdaq_change_pct: float | None = None,
    nikkei_change_pct: float | None = None,
    hang_seng_change_pct: float | None = None,
) -> float:
    """Compute global overnight markets composite (normalized 0–100).

    Weighted average of major global index changes:
    - 35% S&P 500 / Dow Jones
    - 25% NASDAQ
    - 20% Nikkei (Japan)
    - 20% Hang Seng (Hong Kong)

    Positive changes = bullish global sentiment = lower danger for India.

    Args:
        indicators: Dict with keys like sp500_change, nasdaq_change,
                    nikkei_change, hang_seng_change. If provided, used as
                    primary source; keyword args serve as overrides.
        dow_change_pct: Dow Jones daily change %
        nasdaq_change_pct: NASDAQ daily change %
        nikkei_change_pct: Nikkei 225 daily change %
        hang_seng_change_pct: Hang Seng daily change %

    Returns:
        Normalized 0–100 score (neutral 50 if insufficient data)
    """
    # Support both dict input and keyword args
    ind = indicators if indicators is not None else {}

    def _get(key: str, kwarg_val):
        """Prefer keyword arg if explicitly provided, else fall back to dict."""
        if kwarg_val is not None:
            return kwarg_val
        return ind.get(key)

    dow   = _get("sp500_change", dow_change_pct)
    nasdq = _get("nasdaq_change", nasdaq_change_pct)
    nikk  = _get("nikkei_change", nikkei_change_pct)
    hng   = _get("hang_seng_change", hang_seng_change_pct)

    components: list[tuple[float, float]] = []  # (change_pct, weight)

    if dow is not None:
        components.append((float(dow), 0.35))

    if nasdq is not None:
        components.append((float(nasdq), 0.25))

    if nikk is not None:
        components.append((float(nikk), 0.20))

    if hng is not None:
        components.append((float(hng), 0.20))

    if not components:
        return 50.0  # Neutral if no data

    # Compute weighted average change
    total_weight = sum(w for _, w in components)
    avg_change = sum(v * w for v, w in components) / total_weight

    # Normalize: positive change → lower danger (0), negative → higher danger (100)
    # A +2% global rally → ~20 danger, -2% selloff → ~80 danger
    normalized = max(0.0, min(100.0, 50.0 - avg_change * 15))

    logger.info(f"Global overnight composite: {normalized:.1f} (avg change: {avg_change:+.2f}%)")
    return normalized
