"""Rule-Based Fallback Score — Cold start and model failure fallback.

When no ML model is available (first 4–8 weeks or model load failure),
the system uses a weighted average of all normalized indicator scores
to compute the NTI score.
"""

from __future__ import annotations

import logging

from nti.indicators.normalizer import normalize_all_indicators, compute_rule_based_score
from nti.config.thresholds import get_zone

logger = logging.getLogger(__name__)


def run_fallback_inference(raw_indicators: dict) -> dict:
    """Run rule-based fallback inference to compute NTI score.

    This is used when:
    1. No ML model file exists (cold start, first 4–8 weeks)
    2. ML model loading fails (corrupted file, incompatible version)
    3. ML model inference fails (runtime error)

    The fallback uses weighted average of normalized indicator scores
    with the same weights as defined in the 30-parameter framework.

    Args:
        raw_indicators: Dict of raw indicator values from scrapers

    Returns:
        dict with keys:
            nti_score: float (0–100)
            zone: str
            confidence: float (always 50 for fallback — lower confidence)
            is_fallback: bool (always True)
            model_version: str (always "rule-based")
    """
    normalized = normalize_all_indicators(raw_indicators)
    score = compute_rule_based_score(normalized)
    zone = get_zone(score)

    logger.info(
        f"Fallback score: {score:.1f} ({zone}) — "
        f"computed from {len(normalized)} normalized indicators"
    )

    return {
        "nti_score": round(score, 1),
        "zone": zone,
        "confidence": 50.0,  # Lower confidence for rule-based
        "is_fallback": True,
        "model_version": "rule-based",
        "normalized_indicators": normalized,
    }
