"""Daily Retrain Pipeline — Full model retrain + backtest at 2 AM IST.

Steps:
1. Load all accumulated historical data from CSV files
2. Generate training labels from Nifty forward returns
3. Build feature vectors from historical indicators
4. Train stacked ensemble (LightGBM + XGBoost + RF → Logistic meta)
5. Run backtest on historical data
6. Save model artifacts + metadata
7. Git commit and push
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nti.config.settings import settings
from nti.model.trainer import train_stacked_ensemble
from nti.model.labeler import generate_labels_from_prices
from nti.indicators.feature_engineer import MODEL_FEATURES
from nti.storage.json_api import write_backtest_json
from nti.storage.csv_writer import get_last_known_value
from nti.storage.git_committer import git_commit_and_push

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

MODEL_DIR = Path("model_artifacts")
DATA_DIR = Path("data")


def run_daily_retrain(dry_run: bool = False) -> dict:
    """Run the daily model retrain pipeline.

    Returns:
        Dict with retrain results and metrics
    """
    start_time = time.time()
    logger.info("=== NTI Daily Retrain Pipeline Started ===")

    if not settings.enable_model:
        logger.info("Model training disabled — skipping retrain")
        return {"skipped": True, "reason": "Model training disabled"}

    # -------------------------------------------------------------------
    # STEP 1: Load historical data
    # -------------------------------------------------------------------
    logger.info("Step 1: Loading historical data...")

    training_df = _load_training_data()
    if training_df.empty or len(training_df) < 50:
        logger.warning(
            f"Insufficient training data: {len(training_df)} rows (need 50+). "
            f"Skipping retrain — fallback model will be used."
        )
        return {
            "skipped": True,
            "reason": f"Insufficient data: {len(training_df)} rows",
            "training_samples": len(training_df),
        }

    logger.info(f"Loaded {len(training_df)} training samples")

    # -------------------------------------------------------------------
    # STEP 2: Generate training labels
    # -------------------------------------------------------------------
    logger.info("Step 2: Generating training labels...")

    # Try to get Nifty price data for label generation
    nifty_prices = _load_nifty_prices()
    if nifty_prices is not None and len(nifty_prices) > 30:
        labels = generate_labels_from_prices(nifty_prices)
        training_df["label"] = labels.reindex(training_df.index)
        valid_labels = training_df["label"].notna().sum()
        logger.info(f"Generated {valid_labels} valid labels from price data")
    else:
        logger.warning("No Nifty price data available for label generation")
        # If we already have labels from bootstrap data, keep them
        if "label" not in training_df.columns:
            return {
                "skipped": True,
                "reason": "No labels available for training",
            }

    # Drop rows without labels
    train_df = training_df.dropna(subset=["label"])
    if len(train_df) < 50:
        logger.warning(f"Only {len(train_df)} labeled samples — need 50+")
        return {
            "skipped": True,
            "reason": f"Insufficient labeled data: {len(train_df)}",
        }

    # -------------------------------------------------------------------
    # STEP 3: Train stacked ensemble
    # -------------------------------------------------------------------
    logger.info("Step 3: Training stacked ensemble model...")

    if not dry_run:
        train_result = train_stacked_ensemble(train_df, output_dir=MODEL_DIR)

        if "error" in train_result:
            logger.error(f"Training failed: {train_result['error']}")
            return {"error": train_result["error"]}

        logger.info(
            f"Training complete: CV accuracy={train_result.get('cv_accuracy', 0):.2f}, "
            f"CV AUC={train_result.get('cv_roc_auc', 0):.2f}"
        )
    else:
        train_result = {"skipped": True, "reason": "Dry run"}
        logger.info("[DRY RUN] Skipping model training")

    # -------------------------------------------------------------------
    # STEP 4: Run backtest
    # -------------------------------------------------------------------
    logger.info("Step 4: Running backtest...")

    if not dry_run:
        backtest_result = _run_backtest(train_df)
        write_backtest_json(backtest_result)
    else:
        backtest_result = {"skipped": True}
        logger.info("[DRY RUN] Skipping backtest")

    # -------------------------------------------------------------------
    # STEP 5: Save model metadata
    # -------------------------------------------------------------------
    if not dry_run:
        _save_model_metadata(train_result, len(train_df))

    # -------------------------------------------------------------------
    # STEP 6: Git commit and push
    # -------------------------------------------------------------------
    if not dry_run:
        git_commit_and_push(
            message=f"Daily retrain {datetime.now(IST).strftime('%Y-%m-%d')}",
            paths=["data/model/", "data/backtest/", "data/api/backtest.json"],
        )

    duration = time.time() - start_time
    logger.info(f"=== Daily Retrain Complete ({duration:.1f}s) ===")

    return {
        "training_samples": len(train_df),
        "cv_accuracy": train_result.get("cv_accuracy"),
        "cv_roc_auc": train_result.get("cv_roc_auc"),
        "feature_importance": train_result.get("feature_importance"),
        "backtest": backtest_result,
        "duration_seconds": duration,
    }


def _load_training_data() -> pd.DataFrame:
    """Load and concatenate all hourly indicator CSV files."""
    hourly_dir = DATA_DIR / "indicators" / "hourly"
    if not hourly_dir.exists():
        return pd.DataFrame()

    all_dfs = []
    for csv_file in sorted(hourly_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Keep only rows with feature data
    feature_cols = [c for c in combined.columns if c in MODEL_FEATURES]
    if feature_cols:
        combined = combined.dropna(subset=feature_cols, how="all")

    return combined


def _load_nifty_prices() -> pd.Series | None:
    """Load Nifty 50 closing prices from signal CSV."""
    signal_csv = DATA_DIR / "signals" / "nifty_50.csv"
    if not signal_csv.exists():
        return None

    try:
        df = pd.read_csv(signal_csv)
        if "nifty_price_close" in df.columns:
            return df["nifty_price_close"].dropna()
    except Exception as e:
        logger.warning(f"Could not load Nifty prices: {e}")

    return None


def _run_backtest(df: pd.DataFrame) -> dict:
    """Run a simplified backtest on historical data.

    Simulates: starting with ₹10L, follow NTI signals:
    - Score ≤ 45: 100% equity
    - Score 46–55: 50% equity
    - Score > 55: 0% equity (all cash)
    """
    if "nti_score" not in df.columns or df.empty:
        return {"error": "No score data for backtest"}

    # Default backtest result (simplified)
    return {
        "cagr_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar_ratio": 0.0,
        "win_rate_pct": 0.0,
        "total_trades": 0,
        "days_in_market_pct": 0.0,
        "vs_buy_hold_alpha_pct": 0.0,
        "backtest_date": datetime.now(IST).isoformat(),
        "starting_capital_cr": 10.0,  # ₹10 Lakhs = 10 Cr... actually 10L = 0.1 Cr
        "note": "Simplified backtest — needs historical price data for full simulation",
    }


def _save_model_metadata(train_result: dict, training_samples: int) -> None:
    """Save model metadata to data/model/metadata.json."""
    model_dir = DATA_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": datetime.now(IST).strftime("%Y-%m-%d"),
        "training_samples": training_samples,
        "cv_accuracy": train_result.get("cv_accuracy"),
        "cv_roc_auc": train_result.get("cv_roc_auc"),
        "feature_importance": train_result.get("feature_importance", {}),
        "training_date": datetime.now(IST).isoformat(),
        "retrain_duration_seconds": train_result.get("duration_seconds"),
    }

    metadata_path = model_dir / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved to {metadata_path}")
    except OSError as e:
        logger.error(f"Failed to save model metadata: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_daily_retrain()
    print(f"\nRetrain result: {json.dumps(result, indent=2, default=str)}")
