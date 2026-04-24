#!/usr/bin/env python3
"""Bootstrap Historical Data — Seed 2 years of historical data for cold start.

Downloads historical Nifty 50 OHLCV data and basic indicator values
from yfinance, creating the initial CSV files needed for:
- ML model training labels (5-day forward returns)
- Lagged feature computation (PE 5d ago, VIX 5d ago)
- Historical NTI score chart on the website

Run this ONCE after setup: python scripts/bootstrap_data.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")

# Nifty indices to bootstrap
INDICES = {
    "nifty_50": "^NSEI",
    "nifty_bank": "^NSEBANK",
    "sensex": "^BSESN",
    "nifty_midcap_150": "^NSEMIDCAP",
    "nifty_smallcap_250": "^NSESMALLCAP",
}


def bootstrap_nifty_prices(years: int = 2) -> dict[str, pd.DataFrame]:
    """Download historical Nifty prices from yfinance.

    Args:
        years: Number of years of historical data to download

    Returns:
        Dict mapping index name → DataFrame with OHLCV data
    """
    import yfinance as yf

    end_date = datetime.now(IST)
    start_date = end_date - timedelta(days=years * 365)

    results: dict[str, pd.DataFrame] = {}

    for name, symbol in INDICES.items():
        logger.info(f"Downloading {name} ({symbol}) from {start_date.date()} to {end_date.date()}...")

        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"No data for {name} ({symbol})")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            results[name] = df
            logger.info(f"  Got {len(df)} rows for {name}")

        except Exception as e:
            logger.warning(f"Failed to download {name}: {e}")

    return results


def bootstrap_pe_data() -> pd.DataFrame:
    """Download historical Nifty PE data from yfinance or nsetools.

    Returns:
        DataFrame with date, pe, pb, dy columns
    """
    import yfinance as yf

    logger.info("Attempting to get historical PE/PB/DY data...")

    try:
        # Try nsetools for current PE data
        from nsetools import Nse
        nse = Nse()

        # Get current index quote
        quote = nse.get_index_quote("NIFTY 50")
        if quote:
            current_pe = quote.get("pe")
            current_pb = quote.get("pb")
            current_dy = quote.get("dy")
            logger.info(f"Current Nifty PE: {current_pe}, PB: {current_pb}, DY: {current_dy}")

    except Exception as e:
        logger.warning(f"nsetools not available for PE data: {e}")

    # Build a minimal PE history DataFrame
    # (Full PE history requires nifty-pe-ratio.com scrape or NSE historical)
    logger.info("Note: Full historical PE data requires manual scrape from nifty-pe-ratio.com")
    logger.info("The system will accumulate PE data from hourly runs going forward.")

    return pd.DataFrame()


def create_signal_csvs(price_data: dict[str, pd.DataFrame]) -> None:
    """Create initial signal CSV files from price data.

    Args:
        price_data: Dict of index name → OHLCV DataFrame
    """
    signals_dir = DATA_DIR / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    for name, df in price_data.items():
        csv_path = signals_dir / f"{name}.csv"

        # Build signal DataFrame with date and close price
        signal_df = pd.DataFrame({
            "date": df.index.date,
            "nifty_price_close": df["close"].values,
        })

        # If file exists, append; otherwise create new
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            # Merge on date, avoiding duplicates
            signal_df = pd.concat([existing, signal_df]).drop_duplicates(
                subset=["date"], keep="last"
            ).sort_values("date")

        signal_df.to_csv(csv_path, index=False)
        logger.info(f"Wrote {len(signal_df)} rows to {csv_path}")


def create_hourly_dir() -> None:
    """Create the hourly indicators directory structure."""
    hourly_dir = DATA_DIR / "indicators" / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {hourly_dir}")


def create_api_placeholder_files() -> None:
    """Create placeholder API JSON files with empty data."""
    import json

    api_dir = DATA_DIR / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    now_ist = datetime.now(IST).isoformat()

    # latest.json
    latest_path = api_dir / "latest.json"
    if not latest_path.exists():
        latest = {
            "timestamp": now_ist,
            "market_status": "closed",
            "primary_index": "nifty_50",
            "indices": {
                "nifty_50": {"score": None, "zone": "UNKNOWN", "confidence": 0},
                "nifty_bank": {"score": None, "zone": "UNKNOWN", "confidence": 0},
                "sensex": {"score": None, "zone": "UNKNOWN", "confidence": 0},
                "nifty_midcap_150": {"score": None, "zone": "UNKNOWN", "confidence": 0},
                "nifty_smallcap_250": {"score": None, "zone": "UNKNOWN", "confidence": 0},
            },
            "indicators": {},
            "top_drivers": [],
            "top_stock_picks": [],
            "latest_blog_slug": "",
            "model_version": "none",
            "model_confidence": 0,
        }
        with open(latest_path, "w") as f:
            json.dump(latest, f, indent=2)
        logger.info(f"Created {latest_path}")

    # history.json
    history_path = api_dir / "history.json"
    if not history_path.exists():
        with open(history_path, "w") as f:
            json.dump({"dates": [], "scores": []}, f, indent=2)
        logger.info(f"Created {history_path}")

    # backtest.json
    backtest_path = api_dir / "backtest.json"
    if not backtest_path.exists():
        with open(backtest_path, "w") as f:
            json.dump({"status": "no_data", "message": "Backtest will run after sufficient data accumulates"}, f, indent=2)
        logger.info(f"Created {backtest_path}")

    # previous_run.json
    prev_path = api_dir / "previous_run.json"
    if not prev_path.exists():
        with open(prev_path, "w") as f:
            json.dump({}, f, indent=2)
        logger.info(f"Created {prev_path}")


def create_model_dir() -> None:
    """Create the model artifacts directory and metadata placeholder."""
    import json

    model_dir = DATA_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "version": "none",
            "training_samples": 0,
            "cv_accuracy": 0,
            "cv_roc_auc": 0,
            "feature_importance": {},
            "training_date": None,
            "retrain_duration_seconds": 0,
            "status": "cold_start",
            "message": "Model will be trained after sufficient data accumulates (4–8 weeks)",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Created {metadata_path}")

    # Also create model_artifacts dir (for joblib files)
    artifacts_dir = Path("model_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Add .gitkeep so the dir is tracked
    (artifacts_dir / ".gitkeep").touch()


def create_screener_dir() -> None:
    """Create screener data directory with placeholder files."""
    import json

    screener_dir = DATA_DIR / "screener"
    screener_dir.mkdir(parents=True, exist_ok=True)

    for run_type in ["pre_market", "post_market"]:
        path = screener_dir / f"latest_{run_type}.json"
        if not path.exists():
            data = {
                "screened_at": None,
                "run_type": run_type,
                "nti_score_at_screen": None,
                "universe_size": 0,
                "passing_filters": 0,
                "top_picks": [],
                "sector_summary": {},
                "exclusion_summary": {},
                "psu_stocks_count": 0,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created {path}")


def create_errors_dir() -> None:
    """Create errors log directory."""
    errors_dir = DATA_DIR / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    (errors_dir / ".gitkeep").touch()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("📊 Nifty Timing Index — Historical Data Bootstrap")
    print("=" * 50)

    # Step 1: Create directory structure
    print("\n📁 Creating data directory structure...")
    create_hourly_dir()
    create_api_placeholder_files()
    create_model_dir()
    create_screener_dir()
    create_errors_dir()

    # Step 2: Download historical prices
    print("\n📈 Downloading historical Nifty prices...")
    try:
        price_data = bootstrap_nifty_prices(years=2)
        if price_data:
            create_signal_csvs(price_data)
            print(f"  ✅ Downloaded data for {len(price_data)} indices")
        else:
            print("  ⚠️  No price data downloaded (yfinance may be rate-limited)")
    except Exception as e:
        print(f"  ⚠️  Price download failed: {e}")
        print("  You can run this again later.")

    # Step 3: Try to get PE data
    print("\n📊 Attempting to get PE/PB/DY data...")
    bootstrap_pe_data()

    print("\n" + "=" * 50)
    print("✅ Bootstrap complete!")
    print("  The system will accumulate data from hourly runs.")
    print("  After 4–8 weeks, enough data will exist for ML model training.")
    print()
    print("  💡 Run a manual hourly pipeline now:")
    print("     uv run python -m nti.pipelines.hourly")


if __name__ == "__main__":
    main()
