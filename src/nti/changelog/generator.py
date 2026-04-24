"""Changelog Generator — Compare current vs previous run to produce changelog markdown.

Every blog post ends with a "What Changed Since Last Hour" section showing:
- NTI score change + zone change (if any)
- Individual indicator changes (with arrows)
- Stock screener changes (entries/exits)
- Zone change alert (highlighted red/green if zone changed)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.thresholds import get_zone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Indicators to track in the changelog (key: display_name)
TRACKED_INDICATORS = {
    "nifty_pe": "Nifty P/E",
    "nifty_pb": "Nifty P/B",
    "nifty_dy": "Dividend Yield",
    "india_vix": "India VIX",
    "mmi_value": "Tickertape MMI",
    "pcr": "Put/Call Ratio",
    "fii_cash_net": "FII Net",
    "dii_cash_net": "DII Net",
    "us_10y_yield": "US 10Y Yield",
    "usd_inr": "USD/INR",
    "brent_crude": "Brent Crude",
    "cpi_inflation": "CPI Inflation",
    "rbi_repo_rate": "RBI Repo Rate",
    "sip_flow_monthly_cr": "SIP Flow",
    "cnn_fg_value": "CNN F&G",
    "custom_fg_composite": "Custom F&G",
    "gift_nifty_price": "GIFT Nifty",
    "gift_nifty_signal": "GIFT Nifty Signal",
}


def load_previous_run(data_dir: Path | None = None) -> dict:
    """Load the previous run data from JSON file.

    Args:
        data_dir: Path to data/api directory. Defaults to data/api/

    Returns:
        Dict of previous run indicators, or empty dict if not found
    """
    if data_dir is None:
        data_dir = Path("data/api")

    prev_file = data_dir / "previous_run.json"
    if not prev_file.exists():
        logger.info("No previous_run.json found — this is the first run")
        return {}

    try:
        with open(prev_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load previous_run.json: {e}")
        return {}


def save_current_run(indicators: dict, data_dir: Path | None = None) -> None:
    """Save current run data for next comparison.

    Args:
        indicators: Dict of current indicator values
        data_dir: Path to data/api directory
    """
    if data_dir is None:
        data_dir = Path("data/api")

    data_dir.mkdir(parents=True, exist_ok=True)
    prev_file = data_dir / "previous_run.json"

    try:
        with open(prev_file, "w") as f:
            json.dump(indicators, f, indent=2, default=str)
    except OSError as e:
        logger.warning(f"Could not save previous_run.json: {e}")


def _format_change(prev: float | None, curr: float | None, unit: str = "") -> str:
    """Format an indicator change with arrows and colors.

    Args:
        prev: Previous value (None = not available)
        curr: Current value (None = not available)
        unit: Unit suffix (e.g., "%", "x", "Cr")

    Returns:
        Formatted change string like "↑ +0.2 (slightly more expensive)"
    """
    if prev is None or curr is None:
        if curr is not None:
            return f"→ {curr}{unit} (new)"
        return "— (no data)"

    diff = curr - prev
    if abs(diff) < 0.001:
        return f"→ {curr}{unit} (no change)"

    arrow = "↑" if diff > 0 else "↓"
    sign = "+" if diff > 0 else ""

    return f"{arrow} {sign}{diff:.2f}{unit} (now {curr}{unit})"


def generate_changelog(
    current: dict,
    previous: dict,
    current_stocks: list[str] | None = None,
    previous_stocks: list[str] | None = None,
) -> str:
    """Generate a changelog markdown section comparing current vs previous run.

    Args:
        current: Dict of current indicator values
        previous: Dict of previous indicator values
        current_stocks: List of current top stock symbols
        previous_stocks: List of previous top stock symbols

    Returns:
        Markdown string for the changelog section
    """
    sections: list[str] = []

    # --- Score Change ---
    curr_score = current.get("nti_score")
    prev_score = previous.get("nti_score")
    curr_zone = get_zone(curr_score) if curr_score is not None else "UNKNOWN"
    prev_zone = get_zone(prev_score) if prev_score is not None else "UNKNOWN"

    if curr_score is not None and prev_score is not None:
        score_diff = curr_score - prev_score
        if abs(score_diff) < 0.1:
            score_line = f"**NTI Score:** {prev_score:.1f} → {curr_score:.1f} (no change, {curr_zone} zone)"
        else:
            sign = "+" if score_diff > 0 else ""
            score_line = f"**NTI Score:** {prev_score:.1f} → {curr_score:.1f} ({sign}{score_diff:.1f} points, {curr_zone} zone)"
    elif curr_score is not None:
        score_line = f"**NTI Score:** {curr_score:.1f} ({curr_zone} zone)"
    else:
        score_line = "**NTI Score:** Unavailable"

    # --- Zone Change Alert ---
    zone_changed = curr_zone != prev_zone and prev_zone != "UNKNOWN"
    if zone_changed:
        # Determine direction
        curr_val = curr_score or 0
        prev_val = prev_score or 0
        if curr_val > prev_val:
            emoji = "🔴"
            direction = "danger increased"
        else:
            emoji = "🟢"
            direction = "danger decreased"

        alert_section = f"""## {emoji} ZONE CHANGE ALERT: {prev_zone} → {curr_zone}

**Previous Score:** {prev_score:.1f} ({prev_zone} zone)
**Current Score:** {curr_score:.1f} ({curr_zone} zone)

**What drove the zone change:**
"""
        # Add indicator changes that may have driven it
        driver_lines = []
        for key, display_name in TRACKED_INDICATORS.items():
            pv = previous.get(key)
            cv = current.get(key)
            try:
                if pv is not None and cv is not None and abs(float(cv) - float(pv)) > 0.01:
                    driver_lines.append(f"- {display_name}: {float(pv):.2f} → {float(cv):.2f}")
            except (ValueError, TypeError):
                if pv is not None and cv is not None and str(pv) != str(cv):
                    driver_lines.append(f"- {display_name}: {pv} → {cv}")

        if driver_lines:
            alert_section += "\n".join(driver_lines[:5]) + "\n"
        else:
            alert_section += "- Multiple small changes contributed to the zone shift\n"

        alert_section += """
**Action implication (NOT investment advice):**
According to NTI's model, this zone suggests """
        if curr_val >= 56:
            alert_section += "REDUCING equity exposure. This does NOT mean shorting — it means gradually moving to cash from existing long positions.\n"
        elif curr_val <= 45:
            alert_section += "ACCUMULATING equity positions gradually. Focus on value stocks with PE < 20 and PB < 3.\n"
        else:
            alert_section += "HOLDING current positions. No strong conviction to buy or sell.\n"

        sections.append(alert_section)
    else:
        # Standard score change header
        no_zone_change = f"\n**No zone change.** Score remains in {curr_zone} ({_get_zone_range(curr_zone)}).\n"
        sections.append(f"## 📋 What Changed Since Last Hour\n\n{score_line}\n{no_zone_change}")

    # --- Indicator Changes Table ---
    indicator_changes = []
    for key, display_name in TRACKED_INDICATORS.items():
        pv = previous.get(key)
        cv = current.get(key)
        if pv is not None or cv is not None:
            unit = _get_unit(key)
            try:
                change_str = _format_change(
                    float(pv) if pv is not None else None,
                    float(cv) if cv is not None else None,
                    unit,
                )
            except (ValueError, TypeError):
                if pv is None or cv is None:
                    change_str = f"→ {cv}{unit}" if cv is not None else "— (no data)"
                elif str(pv) == str(cv):
                    change_str = f"→ {cv}{unit} (no change)"
                else:
                    change_str = f"→ {cv}{unit} (was {pv})"
            indicator_changes.append(f"| {display_name} | {pv if pv is not None else '—'} | {cv if cv is not None else '—'} | {change_str} |")

    if indicator_changes:
        table = "\n**Indicator Changes:**\n\n| Indicator | Previous | Current | Change |\n|-----------|---------|---------|--------|\n"
        table += "\n".join(indicator_changes)
        sections.append(table)

    # --- Stock Screener Changes ---
    if current_stocks and previous_stocks:
        curr_set = set(current_stocks)
        prev_set = set(previous_stocks)

        entered = curr_set - prev_set
        exited = prev_set - curr_set

        stock_changes = []
        if entered:
            stock_changes.append(f"- 🆕 Entered: {', '.join(sorted(entered))}")
        if exited:
            stock_changes.append(f"- ❌ Exited: {', '.join(sorted(exited))}")

        if stock_changes:
            stock_section = "\n**Stock Screener Changes:**\n" + "\n".join(stock_changes)
            sections.append(stock_section)
        else:
            sections.append("\n**Stock Screener:** No changes from previous run.")

    return "\n\n".join(sections)


def _get_zone_range(zone: str) -> str:
    """Get score range string for a zone."""
    from nti.config.thresholds import ZONES
    for (low, high), z in ZONES.items():
        if z == zone:
            return f"{low}–{high}"
    return "unknown"


def _get_unit(key: str) -> str:
    """Get display unit for an indicator key."""
    units = {
        "nifty_pe": "x",
        "nifty_pb": "x",
        "nifty_dy": "%",
        "india_vix": "",
        "mmi_value": "",
        "pcr": "",
        "fii_cash_net": " Cr",
        "dii_cash_net": " Cr",
        "us_10y_yield": "%",
        "usd_inr": "",
        "brent_crude": "",
        "cpi_inflation": "%",
        "rbi_repo_rate": "%",
        "sip_flow_monthly_cr": " Cr",
        "cnn_fg_value": "",
        "custom_fg_composite": "",
        "gift_nifty_price": "",
        "gift_nifty_signal": "",
    }
    return units.get(key, "")
