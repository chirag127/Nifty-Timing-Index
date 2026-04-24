"""Model Predictor — Hourly inference with stacked ensemble.

Loads model artifacts, runs inference on current indicators,
produces NTI score (0–100) with confidence and SHAP top drivers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nti.indicators.feature_engineer import MODEL_FEATURES, build_feature_vector
from nti.model.fallback import run_fallback_inference
from nti.config.thresholds import get_zone

logger = logging.getLogger(__name__)

MODEL_DIR = Path("model_artifacts")


def run_inference(
    raw_indicators: dict,
    previous_score: float | None = None,
    score_yesterday: float | None = None,
    pe_5d_ago: float | None = None,
    vix_5d_ago: float | None = None,
    model_dir: Path = MODEL_DIR,
) -> dict:
    """Run hourly NTI inference using the stacked ensemble model.

    If the ML model is unavailable, falls back to rule-based scoring.

    Args:
        raw_indicators: Dict of raw indicator values from scrapers
        previous_score: Previous hourly NTI score
        score_yesterday: Score from same time yesterday
        pe_5d_ago: Nifty PE 5 days ago
        vix_5d_ago: India VIX 5 days ago
        model_dir: Directory containing model artifacts

    Returns:
        dict with keys: nti_score, zone, confidence, is_fallback,
                         model_version, top_drivers
    """
    # Build feature vector
    features = build_feature_vector(
        raw_indicators,
        previous_score=previous_score,
        score_yesterday=score_yesterday,
        pe_5d_ago=pe_5d_ago,
        vix_5d_ago=vix_5d_ago,
    )

    # Try to load ML model
    try:
        import joblib

        lgbm_path = model_dir / "lgbm_model.joblib"
        xgb_path = model_dir / "xgb_model.joblib"
        rf_path = model_dir / "rf_model.joblib"
        meta_path = model_dir / "meta_learner.joblib"
        scaler_path = model_dir / "scaler.joblib"

        if not meta_path.exists():
            logger.info("No trained model found — using rule-based fallback")
            return run_fallback_inference(raw_indicators)

        # Load models
        lgbm_model = joblib.load(lgbm_path) if lgbm_path.exists() else None
        xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None
        rf_model = joblib.load(rf_path) if rf_path.exists() else None
        meta_learner = joblib.load(meta_path)
        scaler = joblib.load(scaler_path)

        # Build feature array in correct order
        feature_values = [features.get(f, 50.0) for f in MODEL_FEATURES]
        X = np.array([feature_values])

        # Layer 1: Base learner predictions
        base_preds = []

        if lgbm_model is not None:
            try:
                pred = lgbm_model.predict_proba(X)[:, 1]
                base_preds.append(float(pred[0]))
            except Exception as e:
                logger.warning(f"LightGBM inference failed: {e}")

        if xgb_model is not None:
            try:
                pred = xgb_model.predict_proba(X)[:, 1]
                base_preds.append(float(pred[0]))
            except Exception as e:
                logger.warning(f"XGBoost inference failed: {e}")

        if rf_model is not None:
            try:
                pred = rf_model.predict_proba(X)[:, 1]
                base_preds.append(float(pred[0]))
            except Exception as e:
                logger.warning(f"RF inference failed: {e}")

        if not base_preds:
            logger.warning("All base learners failed — using fallback")
            return run_fallback_inference(raw_indicators)

        # Layer 2: Meta-learner prediction
        meta_features = np.array([base_preds])
        meta_scaled = scaler.transform(meta_features)
        probability = float(meta_learner.predict_proba(meta_scaled)[:, 1][0])

        # Convert to NTI score
        nti_score = round(probability * 100, 1)

        # Confidence = abs(prob - 0.5) / 0.5 * 100
        confidence = round(abs(probability - 0.5) / 0.5 * 100, 1)

        zone = get_zone(nti_score)

        # Compute SHAP top drivers
        top_drivers = _compute_shap_drivers(lgbm_model, X, MODEL_FEATURES)

        logger.info(
            f"ML inference: score={nti_score} ({zone}), "
            f"confidence={confidence}%, "
            f"base_preds={[f'{p:.3f}' for p in base_preds]}"
        )

        return {
            "nti_score": nti_score,
            "zone": zone,
            "confidence": confidence,
            "is_fallback": False,
            "model_version": "stacked_ensemble",
            "top_drivers": top_drivers,
            "base_predictions": base_preds,
        }

    except ImportError:
        logger.warning("joblib not installed — using fallback")
        return run_fallback_inference(raw_indicators)

    except Exception as e:
        logger.warning(f"ML inference failed: {e} — using fallback")
        return run_fallback_inference(raw_indicators)


def _compute_shap_drivers(
    model,
    X: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> list[dict]:
    """Compute SHAP top drivers for the current prediction.

    Args:
        model: A trained tree-based model (LightGBM or RF)
        X: Feature array (1 sample)
        feature_names: List of feature names
        top_n: Number of top drivers to return

    Returns:
        List of {indicator, label, shap, direction} dicts
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, use the positive class SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1][0]  # Positive class
        else:
            sv = shap_values[0]

        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(sv))[::-1][:top_n]

        drivers = []
        for idx in indices:
            shap_val = float(sv[idx])
            direction = "sell" if shap_val > 0 else "buy"
            drivers.append({
                "indicator": feature_names[idx],
                "label": _human_readable_label(feature_names[idx]),
                "shap": round(abs(shap_val), 4),
                "direction": direction,
            })

        return drivers

    except ImportError:
        logger.warning("SHAP not installed — returning empty drivers")
        return []
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return []


def _human_readable_label(feature_name: str) -> str:
    """Convert feature name to human-readable label."""
    labels = {
        "nifty_pe_normalized": "Nifty PE Ratio",
        "nifty_pb_normalized": "Nifty P/B Ratio",
        "earnings_yield_bond_spread": "Earnings vs Bond Yield",
        "dividend_yield_normalized": "Dividend Yield",
        "mcap_to_gdp_percentile": "Market Cap/GDP (Buffett)",
        "midcap_pe_normalized": "Midcap PE Ratio",
        "mmi_score": "Market Mood Index",
        "vix_normalized": "India VIX",
        "pcr_normalized": "Put/Call Ratio",
        "custom_fg_composite": "Custom Fear & Greed",
        "fii_cash_5d_avg_normalized": "FII 5-Day Avg Flow",
        "rbi_rate_direction": "RBI Policy Direction",
        "cpi_normalized": "CPI Inflation",
        "us_10y_normalized": "US 10Y Bond Yield",
        "usdinr_30d_change": "USD/INR 30-Day Change",
        "crude_normalized": "Brent Crude Oil",
        "fii_fo_net_normalized": "FII F&O Net Position",
        "dii_net_normalized": "DII Net Flow",
        "llm_news_danger_score": "LLM News Danger Score",
        "global_overnight_normalized": "Global Overnight Markets",
        "nti_score_lag1": "Previous Hour Score",
        "nti_score_lag24": "Yesterday Same-Time Score",
        "pe_5d_change_normalized": "PE 5-Day Change",
        "vix_5d_change": "VIX 5-Day Change",
        "day_of_week": "Day of Week",
        "days_to_monthly_expiry": "Days to F&O Expiry",
    }
    return labels.get(feature_name, feature_name.replace("_", " ").title())
