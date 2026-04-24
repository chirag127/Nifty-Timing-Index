"""Tests for changelog generator — score changes, zone changes, stock changes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nti.changelog.generator import (
    generate_changelog,
    load_previous_run,
    save_current_run,
    _format_change,
    _get_unit,
    _get_zone_range,
)
from nti.config.thresholds import get_zone


# --- _format_change ---

class TestFormatChange:
    def test_both_none(self):
        result = _format_change(None, None)
        assert result == "— (no data)"

    def test_prev_none_curr_exists(self):
        result = _format_change(None, 21.3, "x")
        assert "21.3x" in result
        assert "new" in result

    def test_no_change(self):
        result = _format_change(21.3, 21.3, "x")
        assert "no change" in result

    def test_increase(self):
        result = _format_change(21.3, 21.5, "x")
        assert "↑" in result
        assert "+0.20" in result

    def test_decrease(self):
        result = _format_change(21.5, 21.3, "x")
        assert "↓" in result

    def test_with_unit(self):
        result = _format_change(None, 1240, " Cr")
        assert " Cr" in result


# --- _get_unit ---

class TestGetUnit:
    def test_pe_unit(self):
        assert _get_unit("nifty_pe") == "x"

    def test_fii_unit(self):
        assert _get_unit("fii_cash_net") == " Cr"

    def test_vix_no_unit(self):
        assert _get_unit("india_vix") == ""

    def test_unknown_key(self):
        assert _get_unit("unknown_key") == ""


# --- generate_changelog ---

class TestGenerateChangelog:
    def test_no_previous_data(self):
        """First run (no previous data) should produce a changelog."""
        current = {"nti_score": 42.0, "nifty_pe": 21.3, "india_vix": 16.2}
        previous = {}
        result = generate_changelog(current, previous)
        assert "42" in result
        assert "BUY_LEAN" in result

    def test_same_zone_no_change(self):
        """Score change within same zone should note no zone change."""
        current = {"nti_score": 44.0, "nifty_pe": 21.5}
        previous = {"nti_score": 42.0, "nifty_pe": 21.3}
        result = generate_changelog(current, previous)
        assert "No zone change" in result

    def test_zone_change_detected(self):
        """Zone change should produce alert section."""
        current = {"nti_score": 57.0, "nifty_pe": 22.5, "india_vix": 11.0}
        previous = {"nti_score": 52.0, "nifty_pe": 21.3, "india_vix": 16.2}
        result = generate_changelog(current, previous)
        assert "ZONE CHANGE ALERT" in result
        assert "SELL_LEAN" in result

    def test_stock_changes_detected(self):
        """Stock screener additions/removals should be shown."""
        current = {"nti_score": 42.0}
        previous = {"nti_score": 40.0}
        current_stocks = ["SBIN", "NTPC", "COALINDIA", "NMDC"]
        previous_stocks = ["SBIN", "NTPC", "COALINDIA", "BHEL"]
        result = generate_changelog(current, previous, current_stocks, previous_stocks)
        assert "NMDC" in result
        assert "BHEL" in result
        assert "Entered" in result or "🆕" in result
        assert "Exited" in result or "❌" in result

    def test_indicator_changes_table(self):
        """Indicator changes should be shown in a table."""
        current = {"nti_score": 42.0, "nifty_pe": 21.5, "india_vix": 16.0}
        previous = {"nti_score": 40.0, "nifty_pe": 21.3, "india_vix": 17.2}
        result = generate_changelog(current, previous)
        assert "Indicator Changes" in result

    def test_extreme_zone_change(self):
        """Zone change from buy to sell zone should include action implications."""
        current = {"nti_score": 72.0, "nifty_pe": 25.0}
        previous = {"nti_score": 30.0, "nifty_pe": 15.0}
        result = generate_changelog(current, previous)
        assert "ZONE CHANGE ALERT" in result
        assert "REDUCING" in result or "reducing" in result


# --- load/save previous_run ---

class TestPreviousRunIO:
    def test_save_and_load(self):
        """Save and load previous run data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            indicators = {"nti_score": 42.0, "nifty_pe": 21.3, "india_vix": 16.2}

            save_current_run(indicators, data_dir)
            loaded = load_previous_run(data_dir)

            assert loaded["nti_score"] == 42.0
            assert loaded["nifty_pe"] == 21.3

    def test_load_nonexistent(self):
        """Loading from nonexistent directory returns empty dict."""
        result = load_previous_run(Path("/nonexistent/path"))
        assert result == {}

    def test_load_corrupted_json(self):
        """Loading corrupted JSON returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            prev_file = data_dir / "previous_run.json"
            prev_file.write_text("{corrupted json")

            result = load_previous_run(data_dir)
            assert result == {}


# --- _get_zone_range ---

class TestGetZoneRange:
    def test_extreme_buy_range(self):
        assert _get_zone_range("EXTREME_BUY") == "0–15"

    def test_neutral_range(self):
        assert _get_zone_range("NEUTRAL") == "46–55"

    def test_unknown_zone(self):
        assert _get_zone_range("UNKNOWN") == "unknown"
