"""Model Trainer — Stacked Ensemble (LightGBM + XGBoost + RF → Logistic meta).

Architecture:
    Layer 1 — Base Learners (5-fold CV):
        Model A: LightGBM (leaf-wise, fast, best for Indian market tabular data)
        Model B: XGBoost (depth-wise, regularized, robust)
        Model C: Random Forest (bagging, variance reduction)
    Layer 2 — Meta-Learner:
        Logistic Regression with L2 regularization
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nti.indicators.feature_engineer import MODEL_FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR = Path("model_artifacts")


def train_stacked_ensemble(
    df: pd.DataFrame,
    label_col: str = "label",
    output_dir: Path = MODEL_DIR,
) -> dict:
    """Train the full stacked ensemble model.

    Args:
        df: Training DataFrame with feature columns + label column
        label_col: Name of the label column (0=buy, 1=sell)
        output_dir: Directory to save model artifacts

    Returns:
        dict with keys: cv_accuracy, cv_roc_auc, feature_importance,
                         training_samples, training_date, duration_seconds
    """
    start_time = time.time()

    # Filter to valid labels only
    train_df = df.dropna(subset=[label_col]).copy()
    train_df[label_col] = train_df[label_col].astype(int)

    if len(train_df) < 50:
        logger.warning(f"Only {len(train_df)} training samples — need at least 50")
        return {"error": "Insufficient training data", "training_samples": len(train_df)}

    # Extract features and labels
    available_features = [f for f in MODEL_FEATURES if f in train_df.columns]
    X = train_df[available_features].fillna(50.0).values
    y = train_df[label_col].values

    logger.info(f"Training on {len(train_df)} samples with {len(available_features)} features")
    logger.info(f"Class distribution: 0={sum(y == 0)}, 1={sum(y == 1)}")

    # --- Layer 1: Base Learners with 5-fold CV ---
    n_splits = min(5, max(2, len(train_df) // 20))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize base learners
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
    }

    xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }

    rf_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    # Generate out-of-fold predictions for meta-learner
    oof_preds = np.zeros((len(train_df), 3))  # 3 base learners

    try:
        import lightgbm as lgb
        has_lgbm = True
    except ImportError:
        has_lgbm = False
        logger.warning("LightGBM not installed — using only XGBoost + RF")

    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False
        logger.warning("XGBoost not installed — using only LightGBM + RF")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # LightGBM
        if has_lgbm:
            lgbm_model = lgb.LGBMClassifier(**lgbm_params)
            lgbm_model.fit(X_train, y_train)
            oof_preds[val_idx, 0] = lgbm_model.predict_proba(X_val)[:, 1]

        # XGBoost
        if has_xgb:
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train)
            oof_preds[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]

        # Random Forest
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        oof_preds[val_idx, 2] = rf_model.predict_proba(X_val)[:, 1]

    # --- Layer 2: Meta-Learner (Logistic Regression) ---
    # Use only columns for base learners that are available
    active_cols = []
    if has_lgbm:
        active_cols.append(0)
    if has_xgb:
        active_cols.append(1)
    active_cols.append(2)  # RF always available

    meta_features = oof_preds[:, active_cols]
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)

    meta_learner = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    meta_learner.fit(meta_features_scaled, y)

    # --- Evaluate on OOF predictions ---
    meta_preds = meta_learner.predict(meta_features_scaled)
    meta_proba = meta_learner.predict_proba(meta_features_scaled)[:, 1]

    cv_accuracy = accuracy_score(y, meta_preds)
    cv_roc_auc = roc_auc_score(y, meta_proba)

    logger.info(f"CV Accuracy: {cv_accuracy:.3f}, CV ROC-AUC: {cv_roc_auc:.3f}")

    # --- Feature Importance (from LightGBM or RF) ---
    feature_importance: dict[str, float] = {}
    if has_lgbm:
        # Retrain on full data for feature importance
        lgbm_full = lgb.LGBMClassifier(**lgbm_params)
        lgbm_full.fit(X, y)
        importances = lgbm_full.feature_importances_
        for feat, imp in zip(available_features, importances):
            feature_importance[feat] = round(float(imp) / max(importances), 3)
    else:
        rf_full = RandomForestClassifier(**rf_params)
        rf_full.fit(X, y)
        importances = rf_full.feature_importances_
        for feat, imp in zip(available_features, importances):
            feature_importance[feat] = round(float(imp) / max(importances), 3)

    # --- Save models ---
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib

        if has_lgbm:
            joblib.dump(lgbm_full if has_lgbm else None, output_dir / "lgbm_model.joblib")
        if has_xgb:
            joblib.dump(xgb_model, output_dir / "xgb_model.joblib")
        joblib.dump(rf_full if not has_lgbm else rf_model, output_dir / "rf_model.joblib")
        joblib.dump(meta_learner, output_dir / "meta_learner.joblib")
        joblib.dump(scaler, output_dir / "scaler.joblib")

    except Exception as e:
        logger.warning(f"Error saving models: {e}")

    # --- Save metadata ---
    duration = time.time() - start_time
    metadata = {
        "version": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "training_samples": len(train_df),
        "cv_accuracy": round(cv_accuracy, 3),
        "cv_roc_auc": round(cv_roc_auc, 3),
        "feature_importance": dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:10]),
        "training_date": pd.Timestamp.now().isoformat(),
        "retrain_duration_seconds": round(duration, 1),
        "base_learners": ["lgbm" if has_lgbm else None, "xgb" if has_xgb else None, "rf"],
        "features_used": available_features,
    }

    metadata_path = Path("data/model/metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model training complete in {duration:.1f}s")
    return metadata
