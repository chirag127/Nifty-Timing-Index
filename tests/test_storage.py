"""Tests for storage modules — CSV writer, JSON API writer, blog writer."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from nti.storage.csv_writer import (
    write_hourly_csv,
    write_signal_csv,
    get_last_known_value,
    HOURLY_COLUMNS,
    SIGNAL_COLUMNS,
)
from nti.storage.json_api import (
    write_latest_json,
    write_history_json,
    write_backtest_json,
)
from nti.storage.blog_writer import write_blog_post


# --- CSV Writer ---

class TestHourlyCSV:
    def test_creates_csv_file(self):
        """Writing hourly CSV should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indicators = {
                "nti_score": 42.0,
                "nifty_pe": 21.3,
                "india_vix": 16.2,
                "zone": "BUY_LEAN",
            }
            path = write_hourly_csv(indicators, Path(tmpdir))
            assert path.exists()
            assert path.stat().st_size > 0

    def test_csv_has_header(self):
        """CSV file should have a header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indicators = {"nti_score": 42.0}
            path = write_hourly_csv(indicators, Path(tmpdir))

            with open(path, encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "timestamp" in header
                assert "nti_score" in header

    def test_csv_appends_rows(self):
        """Second write should append, not overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indicators1 = {"nti_score": 42.0}
            indicators2 = {"nti_score": 44.0}
            path = write_hourly_csv(indicators1, Path(tmpdir))
            write_hourly_csv(indicators2, Path(tmpdir))

            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
                assert float(rows[0]["nti_score"]) == 42.0
                assert float(rows[1]["nti_score"]) == 44.0


class TestSignalCSV:
    def test_creates_signal_csv(self):
        """Writing signal CSV should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "date": "2026-04-23",
                "nti_score_open": 40.0,
                "nti_score_close": 42.0,
                "nti_score_avg": 41.0,
                "zone_open": "BUY_LEAN",
                "zone_close": "BUY_LEAN",
            }
            path = write_signal_csv("nifty_50", data, Path(tmpdir))
            assert path.exists()
            assert "nifty_50" in path.name


class TestGetLastKnownValue:
    def test_returns_last_value(self):
        """Should return the most recent value for an indicator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indicators1 = {"nti_score": 42.0, "nifty_pe": 21.3}
            indicators2 = {"nti_score": 44.0, "nifty_pe": 21.5}
            write_hourly_csv(indicators1, Path(tmpdir))
            write_hourly_csv(indicators2, Path(tmpdir))

            result = get_last_known_value("nifty_pe", Path(tmpdir))
            assert result == 21.5

    def test_returns_none_when_no_data(self):
        """Should return None when no CSV files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_last_known_value("nifty_pe", Path(tmpdir))
            assert result is None


# --- JSON API Writer ---

class TestLatestJson:
    def test_creates_latest_json(self):
        """Writing latest.json should create a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indicators = {
                "nifty_price": 24180.5,
                "nifty_pe": 21.3,
                "india_vix": 16.2,
            }
            nti_result = {
                "nti_score": 42.0,
                "zone": "BUY_LEAN",
                "confidence": 74.0,
                "model_version": "rule-based",
                "is_fallback": True,
            }
            path = write_latest_json(
                indicators, nti_result, [], [], "2026-04-23-14-00",
                data_dir=Path(tmpdir),
            )
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert data["indices"]["nifty_50"]["score"] == 42.0
            assert data["indices"]["nifty_50"]["zone"] == "BUY_LEAN"
            assert data["primary_index"] == "nifty_50"

    def test_latest_json_has_timestamp(self):
        """latest.json should have a timestamp field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_latest_json(
                {}, {"nti_score": 42.0, "zone": "BUY_LEAN", "confidence": 74.0},
                [], [], data_dir=Path(tmpdir),
            )
            with open(path) as f:
                data = json.load(f)
            assert "timestamp" in data


class TestBacktestJson:
    def test_creates_backtest_json(self):
        """Writing backtest.json should create a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backtest_data = {
                "cagr_pct": 12.5,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": -18.0,
            }
            path = write_backtest_json(backtest_data, Path(tmpdir))
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert data["cagr_pct"] == 12.5


# --- Blog Writer ---

class TestBlogWriter:
    def test_creates_blog_file(self):
        """Writing blog post should create a .md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_blog_post(
                blog_markdown="# Test Blog\n\nThis is a test blog.",
                nti_score=42.0,
                prev_score=40.0,
                confidence=74.0,
                nifty_price=24180.5,
                top_drivers=["nifty_pe_normalized", "mmi_score"],
                top_stocks=["SBIN", "NTPC"],
                blog_type="mid_session",
                website_dir=Path(tmpdir),
            )
            assert path.exists()
            assert path.suffix == ".md"

    def test_blog_has_frontmatter(self):
        """Blog file should have YAML frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_blog_post(
                blog_markdown="Test content",
                nti_score=42.0,
                prev_score=40.0,
                confidence=74.0,
                nifty_price=24180.5,
                top_drivers=["nifty_pe_normalized"],
                top_stocks=["SBIN"],
                blog_type="mid_session",
                website_dir=Path(tmpdir),
            )
            content = path.read_text(encoding="utf-8")
            assert content.startswith("---")
            assert "ntiScore: 42" in content
            assert "ntiZone:" in content
            assert "blogType:" in content

    def test_blog_zone_change_flag(self):
        """Blog frontmatter should set zoneChanged=true when zone changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_blog_post(
                blog_markdown="Test",
                nti_score=57.0,  # SELL_LEAN
                prev_score=42.0,  # BUY_LEAN
                confidence=74.0,
                nifty_price=24180.5,
                top_drivers=[],
                top_stocks=[],
                blog_type="mid_session",
                website_dir=Path(tmpdir),
            )
            content = path.read_text(encoding="utf-8")
            assert "zoneChanged: true" in content
