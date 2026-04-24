"""CSV Writer — Write hourly signal data to CSV files.

Each run appends a row to the daily CSV file in data/indicators/hourly/.
Historical signals are also appended to data/signals/nifty_50.csv (etc.).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# CSV column order for hourly indicator snapshots
HOURLY_COLUMNS = [
    "timestamp", "date", "hour_ist",
    "nti_score", "nti_score_prev", "zone", "zone_prev", "zone_changed",
    "confidence", "is_fallback", "model_version",
    "nifty_price", "nifty_change_pct",
    "nifty_pe", "nifty_pb", "nifty_dy", "earnings_yield_bond_spread", "mcap_to_gdp",
    "mmi_value", "india_vix", "pcr", "custom_fg_composite", "cnn_fg_value", "fii_cash_net",
    "gift_nifty_price", "gift_nifty_change", "gift_nifty_change_pct", "gift_nifty_signal",
    "rbi_repo_rate", "rbi_stance", "rbi_direction",
    "cpi_inflation", "us_10y_yield", "usd_inr", "brent_crude", "sp500_change",
    "fii_fo_net", "dii_cash_net", "sip_flow_monthly_cr",
    "llm_news_danger", "llm_policy_flag", "llm_geopolitical_score", "global_overnight",
    "rsi_14", "macd", "adv_decline_ratio", "high_low_ratio",
    "top_driver_1", "top_driver_1_shap",
    "top_driver_2", "top_driver_2_shap",
    "top_driver_3", "top_driver_3_shap",
    "blog_slug", "run_duration_seconds", "errors",
]

# Daily signal CSV columns (subset of hourly, one row per day)
SIGNAL_COLUMNS = [
    "date", "nti_score_open", "nti_score_close", "nti_score_avg",
    "zone_open", "zone_close",
    "nifty_price_open", "nifty_price_close", "nifty_change_pct",
    "nifty_pe", "nifty_pb",
    "india_vix_avg", "pcr_avg",
    "fii_cash_net_total", "dii_cash_net_total",
    "confidence_avg", "model_version",
]


def write_hourly_csv(indicators: dict, data_dir: Path | None = None) -> Path:
    """Write hourly indicator snapshot to a daily CSV file.

    Creates data/indicators/hourly/YYYY-MM-DD.csv with one row per hour.

    Args:
        indicators: Dict of all indicator values for this run
        data_dir: Root data directory (default: data/)

    Returns:
        Path to the CSV file written
    """
    if data_dir is None:
        data_dir = Path("data")

    now_ist = datetime.now(IST)
    date_str = now_ist.strftime("%Y-%m-%d")
    csv_dir = data_dir / "indicators" / "hourly"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{date_str}.csv"

    # Add computed fields
    row = dict(indicators)
    row["timestamp"] = now_ist.isoformat()
    row["date"] = date_str
    row["hour_ist"] = now_ist.strftime("%H:%M")

    # Ensure all columns exist (fill missing with empty string)
    for col in HOURLY_COLUMNS:
        if col not in row:
            row[col] = ""

    # Write header if file is new
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HOURLY_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Hourly CSV row written to {csv_path}")
        return csv_path

    except OSError as e:
        logger.error(f"Failed to write hourly CSV: {e}")
        return csv_path


def write_signal_csv(
    index_name: str,
    signal_data: dict,
    data_dir: Path | None = None,
) -> Path:
    """Append a daily signal row to the index signal CSV.

    Creates data/signals/{index_name}.csv with one row per trading day.

    Args:
        index_name: Index identifier (e.g., "nifty_50", "nifty_bank")
        signal_data: Dict of daily aggregate signal data
        data_dir: Root data directory

    Returns:
        Path to the CSV file written
    """
    if data_dir is None:
        data_dir = Path("data")

    csv_dir = data_dir / "signals"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize index name for filename
    safe_name = index_name.lower().replace(" ", "_").replace("^", "")
    csv_path = csv_dir / f"{safe_name}.csv"

    # Ensure all columns exist
    row = dict(signal_data)
    for col in SIGNAL_COLUMNS:
        if col not in row:
            row[col] = ""

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SIGNAL_COLUMNS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Signal CSV row written to {csv_path}")
        return csv_path

    except OSError as e:
        logger.error(f"Failed to write signal CSV: {e}")
        return csv_path


def get_last_known_value(indicator_name: str, data_dir: Path | None = None) -> float | None:
    """Get the last known value for an indicator from the most recent CSV.

    Used as fallback when a scraper fails.

    Args:
        indicator_name: Name of the indicator (e.g., "nifty_pe", "india_vix")
        data_dir: Root data directory

    Returns:
        Last known value as float, or None if not found
    """
    if data_dir is None:
        data_dir = Path("data")

    csv_dir = data_dir / "indicators" / "hourly"
    if not csv_dir.exists():
        return None

    # Find the most recent daily CSV file
    csv_files = sorted(csv_dir.glob("*.csv"), reverse=True)
    if not csv_files:
        return None

    # Search the most recent files (up to 3 days back)
    for csv_path in csv_files[:3]:
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Search from the end (most recent row first)
                for row in reversed(rows):
                    val = row.get(indicator_name, "")
                    if val and val not in ("", "None", "null"):
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            continue
        except (OSError, csv.Error):
            continue

    return None
