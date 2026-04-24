"""Tests for stock screener: hard filters, PSU boost, composite scoring."""

from __future__ import annotations

import pytest

from nti.screener.filters import passes_hard_filters, get_soft_warnings
from nti.screener.scorer import compute_composite_scores
from nti.config.psu_stocks import get_psu_score_boost, PSU_STOCKS


# --- Hard Filters ---

class TestPassesHardFilters:
    def test_pe_too_high_fails(self):
        """PE=20.5 must FAIL (>=20 excluded)."""
        stock = {"symbol": "TEST", "pe": 20.5, "pb": 1.5, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "PE=20.5 must FAIL"

    def test_pe_boundary_passes(self):
        """PE=19.99 must PASS."""
        stock = {"symbol": "TEST", "pe": 19.99, "pb": 1.5, "market_cap_cr": 1000}
        assert passes_hard_filters(stock), "PE=19.99 must PASS"

    def test_pe_exact_20_fails(self):
        """PE=20.0 should FAIL (strictly < 20)."""
        stock = {"symbol": "TEST", "pe": 20.0, "pb": 1.5, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "PE=20.0 must FAIL (strictly < 20)"

    def test_pb_too_high_fails(self):
        """PB=3.1 must FAIL."""
        stock = {"symbol": "TEST", "pe": 15, "pb": 3.1, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "PB=3.1 must FAIL"

    def test_pb_exact_3_fails(self):
        """PB=3.0 should FAIL (strictly < 3)."""
        stock = {"symbol": "TEST", "pe": 15, "pb": 3.0, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "PB=3.0 must FAIL (strictly < 3)"

    def test_pb_boundary_passes(self):
        """PB=2.99 must PASS."""
        stock = {"symbol": "TEST", "pe": 15, "pb": 2.99, "market_cap_cr": 1000}
        assert passes_hard_filters(stock), "PB=2.99 must PASS"

    def test_market_cap_too_small_fails(self):
        """Market cap 499 Cr must FAIL."""
        stock = {"symbol": "TEST", "pe": 10, "pb": 1.0, "market_cap_cr": 499}
        assert not passes_hard_filters(stock), "₹499 Cr must FAIL"

    def test_market_cap_boundary_passes(self):
        """Market cap 500 Cr must PASS."""
        stock = {"symbol": "TEST", "pe": 10, "pb": 1.0, "market_cap_cr": 500}
        assert passes_hard_filters(stock), "₹500 Cr must PASS"

    def test_pe_none_fails(self):
        """None PE should fail (can't evaluate)."""
        stock = {"symbol": "TEST", "pe": None, "pb": 1.5, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "None PE must FAIL"

    def test_pb_none_fails(self):
        """None PB should fail."""
        stock = {"symbol": "TEST", "pe": 15, "pb": None, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "None PB must FAIL"

    def test_negative_pe_fails(self):
        """Negative PE (loss-making) should fail."""
        stock = {"symbol": "TEST", "pe": -5, "pb": 1.5, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "Negative PE must FAIL"

    def test_zero_pe_fails(self):
        """PE of 0 is invalid."""
        stock = {"symbol": "TEST", "pe": 0, "pb": 1.5, "market_cap_cr": 1000}
        assert not passes_hard_filters(stock), "PE=0 must FAIL"

    def test_valid_stock_passes(self):
        """A properly valued stock should pass all filters."""
        stock = {
            "symbol": "SBIN",
            "pe": 11.2,
            "pb": 1.8,
            "market_cap_cr": 712000,
            "roe_pct": 19.4,
            "debt_equity": 0.0,
            "dividend_yield": 2.4,
        }
        assert passes_hard_filters(stock), "SBIN-like stock must PASS"

    def test_all_filters_combined(self):
        """Test multiple filter violations together."""
        # High PE + high PB + small cap
        stock = {"symbol": "TEST", "pe": 25, "pb": 4.0, "market_cap_cr": 300}
        assert not passes_hard_filters(stock)


# --- Soft Warnings ---

class TestGetSoftWarnings:
    def test_low_roe_warning(self):
        """ROE < 12% should produce a warning."""
        stock = {"symbol": "TEST", "roe_pct": 8.0}
        warnings = get_soft_warnings(stock)
        assert any("ROE" in w for w in warnings)

    def test_high_debt_warning(self):
        """High debt/equity should produce a warning."""
        stock = {"symbol": "TEST", "debt_equity": 2.0}
        warnings = get_soft_warnings(stock)
        assert any("debt" in w.lower() for w in warnings)

    def test_clean_stock_no_warnings(self):
        """A clean stock should have no warnings."""
        stock = {"symbol": "TEST", "roe_pct": 20.0, "debt_equity": 0.3}
        warnings = get_soft_warnings(stock)
        assert len(warnings) == 0


# --- PSU Boost ---

class TestPSUBoost:
    def test_psu_stock_gets_boost(self):
        """PSU stock should get +10 boost."""
        assert get_psu_score_boost("SBIN") == 10.0

    def test_non_psu_no_boost(self):
        """Non-PSU stock should get 0 boost."""
        assert get_psu_score_boost("HDFCBANK") == 0.0

    def test_psu_list_not_empty(self):
        """PSU_STOCKS list should not be empty."""
        assert len(PSU_STOCKS) > 0

    def test_known_psu_stocks(self):
        """Known PSU stocks should be in the list."""
        for symbol in ["NTPC", "ONGC", "COALINDIA", "SBIN", "BEL"]:
            assert symbol in PSU_STOCKS, f"{symbol} should be a PSU stock"


# --- Composite Scoring ---

class TestCompositeScoring:
    def test_empty_stocks_returns_empty(self):
        """Empty input should return empty output."""
        assert compute_composite_scores([]) == []

    def test_single_stock_gets_score(self):
        """Single stock should get a composite score."""
        stocks = [{
            "symbol": "TEST",
            "pe": 10,
            "pb": 1.0,
            "dividend_yield": 3.0,
            "roe_pct": 20.0,
            "debt_equity": 0.3,
        }]
        result = compute_composite_scores(stocks)
        assert len(result) == 1
        assert "composite_score" in result[0]
        assert 0 <= result[0]["composite_score"] <= 100

    def test_psu_stock_gets_higher_score(self):
        """PSU stock with same fundamentals should score higher than non-PSU."""
        base = {
            "pe": 12,
            "pb": 1.5,
            "dividend_yield": 2.5,
            "roe_pct": 18.0,
            "debt_equity": 0.5,
        }
        stocks = [
            {"symbol": "SBIN", **base},  # PSU
            {"symbol": "HDFCBANK", **base},  # Non-PSU
        ]
        result = compute_composite_scores(stocks)

        psu_stock = next(s for s in result if s["symbol"] == "SBIN")
        non_psu_stock = next(s for s in result if s["symbol"] == "HDFCBANK")

        assert psu_stock["composite_score"] > non_psu_stock["composite_score"], \
            "PSU stock should score higher with same fundamentals"
        assert psu_stock.get("psu_boost", 0) == 10.0

    def test_score_capped_at_100(self):
        """Composite score should be capped at 100."""
        stocks = [{
            "symbol": "PERFECT",
            "pe": 5,  # Very cheap
            "pb": 0.5,  # Very cheap
            "dividend_yield": 8.0,  # Very high
            "roe_pct": 50.0,  # Excellent
            "debt_equity": 0.0,  # No debt
        }]
        result = compute_composite_scores(stocks)
        assert result[0]["composite_score"] <= 100

    def test_multiple_stocks_ranked(self):
        """Multiple stocks should produce different ranks."""
        stocks = [
            {"symbol": "CHEAP", "pe": 8, "pb": 0.8, "dividend_yield": 4.0, "roe_pct": 25.0, "debt_equity": 0.2},
            {"symbol": "EXPENSIVE", "pe": 18, "pb": 2.5, "dividend_yield": 1.0, "roe_pct": 10.0, "debt_equity": 1.5},
        ]
        result = compute_composite_scores(stocks)
        cheap = next(s for s in result if s["symbol"] == "CHEAP")
        expensive = next(s for s in result if s["symbol"] == "EXPENSIVE")
        assert cheap["composite_score"] > expensive["composite_score"]

    def test_all_scores_in_range(self):
        """All scores should be 0-100."""
        stocks = [
            {"symbol": f"S{i}", "pe": 5 + i * 2, "pb": 0.5 + i * 0.2, "dividend_yield": 5.0 - i * 0.3, "roe_pct": 30.0 - i * 2, "debt_equity": 0.1 + i * 0.1}
            for i in range(10)
        ]
        result = compute_composite_scores(stocks)
        for s in result:
            assert 0 <= s["composite_score"] <= 100, f"{s['symbol']} score {s['composite_score']} out of range"
            assert 0 <= s["value_score"] <= 100
            assert 0 <= s["quality_score"] <= 100
