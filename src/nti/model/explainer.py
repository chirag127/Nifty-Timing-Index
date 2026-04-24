"""Model Explainer — SHAP values for model interpretability.

Provides detailed SHAP explanations for each inference run,
showing which indicators are pushing the score up or down.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def explain_prediction(
    model,
    feature_values: dict[str, float],
    feature_names: list[str] | None = None,
    top_n: int = 5,
) -> dict:
    """Generate a detailed SHAP explanation for a prediction.

    Args:
        model: Trained tree-based model (LightGBM, XGBoost, or RF)
        feature_values: Dict of feature_name → value
        feature_names: Ordered list of feature names (default: use dict keys)
        top_n: Number of top features to explain

    Returns:
        dict with keys:
            top_drivers: list of {feature, value, shap, direction, description}
            base_value: float (model's expected value)
            total_shap: float (sum of all SHAP values)
    """
    try:
        import shap

        if feature_names is None:
            feature_names = list(feature_values.keys())

        X = np.array([[feature_values.get(f, 50.0) for f in feature_names]])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1][0]
            base_value = explainer.expected_value[1] if hasattr(explainer, 'expected_value') else 0.5
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0.5

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(sv))[::-1]

        top_drivers = []
        for idx in indices[:top_n]:
            shap_val = float(sv[idx])
            direction = "sell (↑ danger)" if shap_val > 0 else "buy (↓ danger)"
            top_drivers.append({
                "feature": feature_names[idx],
                "value": feature_values.get(feature_names[idx], 50.0),
                "shap": round(shap_val, 4),
                "abs_shap": round(abs(shap_val), 4),
                "direction": direction,
                "description": _get_description(feature_names[idx], shap_val),
            })

        return {
            "top_drivers": top_drivers,
            "base_value": float(base_value),
            "total_shap": round(float(np.sum(sv)), 4),
        }

    except ImportError:
        logger.warning("SHAP not installed — returning empty explanation")
        return {"top_drivers": [], "base_value": 0.5, "total_shap": 0.0}
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return {"top_drivers": [], "base_value": 0.5, "total_shap": 0.0}


def format_shap_for_blog(top_drivers: list[dict]) -> str:
    """Format SHAP drivers as readable text for blog posts.

    Args:
        top_drivers: List of driver dicts from explain_prediction

    Returns:
        Formatted string for inclusion in blog posts
    """
    if not top_drivers:
        return "SHAP drivers unavailable (model not loaded)"

    lines = []
    for i, driver in enumerate(top_drivers[:3], 1):
        direction_emoji = "🔴" if "sell" in driver.get("direction", "") else "🟢"
        lines.append(
            f"{direction_emoji} {'①②③'[i-1]} {driver.get('feature', 'unknown')}: "
            f"{driver.get('description', '')} "
            f"(SHAP: {driver.get('shap', 0):+.4f})"
        )

    return "\n".join(lines)


def _get_description(feature_name: str, shap_val: float) -> str:
    """Get a human-readable description of a SHAP driver."""
    direction = "pushing toward sell" if shap_val > 0 else "pulling toward buy"

    descriptions = {
        "nifty_pe_normalized": f"Nifty PE is {'above average' if shap_val > 0 else 'below average'} ({direction})",
        "mmi_score": f"Market Mood shows {'greed/complacency' if shap_val > 0 else 'fear'} ({direction})",
        "vix_normalized": f"VIX is {'low/complacent' if shap_val > 0 else 'elevated/fearful'} ({direction})",
        "pcr_normalized": f"PCR indicates {'call-heavy/greed' if shap_val > 0 else 'put-heavy/fear'} ({direction})",
        "cpi_normalized": f"CPI inflation is {'high' if shap_val > 0 else 'low'} ({direction})",
        "us_10y_normalized": f"US 10Y yield is {'high' if shap_val > 0 else 'low'} ({direction})",
        "crude_normalized": f"Crude oil is {'expensive' if shap_val > 0 else 'cheap'} ({direction})",
    }
    return descriptions.get(feature_name, f"{feature_name} ({direction})")
