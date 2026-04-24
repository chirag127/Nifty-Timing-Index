"""JSON API Writer — Update data/api/*.json files for website consumption.

These static JSON files are copied to the Astro website's public/api/ directory
and served directly by Cloudflare Pages CDN (no server needed).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.thresholds import get_zone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def write_latest_json(
    indicators: dict,
    nti_result: dict,
    top_drivers: list[dict],
    top_stocks: list[dict],
    blog_slug: str = "",
    data_dir: Path | None = None,
) -> Path:
    """Write data/api/latest.json with current NTI data.

    This is the primary API endpoint consumed by the website dashboard.

    Args:
        indicators: Dict of raw indicator values from scrapers
        nti_result: Dict from predictor with nti_score, zone, confidence, etc.
        top_drivers: List of {indicator, label, shap, direction} dicts
        top_stocks: List of top stock pick dicts from screener
        blog_slug: Slug of the latest blog post
        data_dir: Root data directory

    Returns:
        Path to the JSON file written
    """
    if data_dir is None:
        data_dir = Path("data")

    api_dir = data_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    score = nti_result.get("nti_score", 0)
    prev_score = nti_result.get("prev_score", indicators.get("nti_score_prev"))
    zone = nti_result.get("zone", get_zone(score))
    prev_zone = nti_result.get("prev_zone", get_zone(prev_score) if prev_score else "UNKNOWN")
    confidence = nti_result.get("confidence", 50)

    latest = {
        "timestamp": datetime.now(IST).isoformat(),
        "market_status": _get_market_status(),
        "primary_index": "nifty_50",
        "indices": {
            "nifty_50": {
                "score": round(score, 1),
                "score_prev": round(prev_score, 1) if prev_score else None,
                "zone": zone,
                "zone_prev": prev_zone,
                "zone_changed": zone != prev_zone and prev_zone != "UNKNOWN",
                "confidence": round(confidence, 1),
                "price": indicators.get("nifty_price"),
                "change_pct": indicators.get("nifty_change_pct"),
            },
        },
        "indicators": {
            "nifty_pe": indicators.get("nifty_pe"),
            "nifty_pb": indicators.get("nifty_pb"),
            "nifty_dy": indicators.get("nifty_dy"),
            "mmi": indicators.get("mmi_value"),
            "india_vix": indicators.get("india_vix"),
            "pcr": indicators.get("pcr"),
            "fii_cash_net": indicators.get("fii_cash_net"),
            "us_10y": indicators.get("us_10y_yield"),
            "usd_inr": indicators.get("usd_inr"),
            "brent_crude": indicators.get("brent_crude"),
            "llm_news_danger": indicators.get("llm_news_danger"),
            "cpi": indicators.get("cpi_inflation"),
            "rbi_repo_rate": indicators.get("rbi_repo_rate"),
            "cnn_fg": indicators.get("cnn_fg_value"),
            "gift_nifty_price": indicators.get("gift_nifty_price"),
            "gift_nifty_signal": indicators.get("gift_nifty_signal"),
            "gift_nifty_change_pct": indicators.get("gift_nifty_change_pct"),
        },
        "top_drivers": top_drivers[:5],
        "top_stock_picks": [
            {
                "symbol": s.get("symbol", ""),
                "pe": s.get("pe"),
                "pb": s.get("pb"),
                "is_psu": s.get("is_psu", False),
                "composite_score": s.get("composite_score"),
            }
            for s in top_stocks[:10]
        ],
        "latest_blog_slug": blog_slug,
        "model_version": nti_result.get("model_version", "rule-based"),
        "model_confidence": round(confidence, 1),
        "is_fallback": nti_result.get("is_fallback", True),
    }

    json_path = api_dir / "latest.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2, default=str)
        logger.info(f"latest.json written to {json_path}")
    except OSError as e:
        logger.error(f"Failed to write latest.json: {e}")

    return json_path


def write_history_json(data_dir: Path | None = None, days: int = 30) -> Path:
    """Write data/api/history.json with last N days of NTI scores.

    Reads from data/signals/nifty_50.csv to build the history array.

    Args:
        data_dir: Root data directory
        days: Number of days of history to include

    Returns:
        Path to the JSON file written
    """
    if data_dir is None:
        data_dir = Path("data")

    api_dir = data_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    history = []
    signal_csv = data_dir / "signals" / "nifty_50.csv"

    if signal_csv.exists():
        import csv
        try:
            with open(signal_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Take last N days
                for row in rows[-days:]:
                    try:
                        history.append({
                            "date": row.get("date", ""),
                            "nti_score": float(row.get("nti_score_close") or row.get("nti_score_avg") or 0),
                            "zone": row.get("zone_close", ""),
                            "nifty_price": float(row.get("nifty_price_close") or 0),
                            "nifty_change_pct": float(row.get("nifty_change_pct") or 0),
                        })
                    except (ValueError, TypeError):
                        continue
        except (OSError, csv.Error) as e:
            logger.warning(f"Could not read signal CSV for history: {e}")

    json_path = api_dir / "history.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"history": history, "days": len(history)}, f, indent=2)
        logger.info(f"history.json written ({len(history)} days)")
    except OSError as e:
        logger.error(f"Failed to write history.json: {e}")

    return json_path


def write_backtest_json(backtest_data: dict, data_dir: Path | None = None) -> Path:
    """Write data/api/backtest.json with backtest results.

    Args:
        backtest_data: Dict of backtest metrics
        data_dir: Root data directory

    Returns:
        Path to the JSON file written
    """
    if data_dir is None:
        data_dir = Path("data")

    api_dir = data_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    json_path = api_dir / "backtest.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(backtest_data, f, indent=2, default=str)
        logger.info(f"backtest.json written")
    except OSError as e:
        logger.error(f"Failed to write backtest.json: {e}")

    return json_path


def _get_market_status() -> str:
    """Determine current market status (open/closed/holiday)."""
    now_ist = datetime.now(IST)
    hour = now_ist.hour
    weekday = now_ist.weekday()

    # Weekend
    if weekday >= 5:
        return "closed"

    # Market hours: 9:15 AM – 3:30 PM IST
    if 9 <= hour < 16:
        if hour == 9 and now_ist.minute < 15:
            return "pre_market"
        if hour == 15 and now_ist.minute > 30:
            return "post_market"
        return "open"

    if 6 <= hour < 9:
        return "pre_market"

    return "closed"
