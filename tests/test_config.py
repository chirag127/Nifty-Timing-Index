"""Tests for configuration modules — settings, thresholds, PSU stocks."""

from __future__ import annotations

import os

import pytest

from nti.config.thresholds import (
    get_zone,
    normalize_pe,
    normalize_pb,
    normalize_dividend_yield,
    normalize_earnings_yield_spread,
    normalize_mmi,
    normalize_vix,
    normalize_pcr,
    normalize_fii_cash,
    normalize_cpi,
    normalize_us_10y,
    normalize_usdinr,
    normalize_crude,
    normalize_mcap_to_gdp,
    normalize_usdinr_change,
    normalize_sip_flow,
    validate_value,
    ZONES,
)
from nti.config.psu_stocks import is_psu, get_psu_score_boost, PSU_STOCKS


# --- Zone Mapping ---

class TestGetZone:
    def test_extreme_buy(self):
        assert get_zone(0) == "EXTREME_BUY"
        assert get_zone(15) == "EXTREME_BUY"

    def test_strong_buy(self):
        assert get_zone(16) == "STRONG_BUY"
        assert get_zone(30) == "STRONG_BUY"

    def test_buy_lean(self):
        assert get_zone(31) == "BUY_LEAN"
        assert get_zone(45) == "BUY_LEAN"

    def test_neutral(self):
        assert get_zone(46) == "NEUTRAL"
        assert get_zone(55) == "NEUTRAL"

    def test_sell_lean(self):
        assert get_zone(56) == "SELL_LEAN"
        assert get_zone(69) == "SELL_LEAN"

    def test_strong_sell(self):
        assert get_zone(70) == "STRONG_SELL"
        assert get_zone(84) == "STRONG_SELL"

    def test_extreme_sell(self):
        assert get_zone(85) == "EXTREME_SELL"
        assert get_zone(100) == "EXTREME_SELL"

    def test_boundary_45_46(self):
        """Score 45 = BUY_LEAN, Score 46 = NEUTRAL."""
        assert get_zone(45) == "BUY_LEAN"
        assert get_zone(46) == "NEUTRAL"

    def test_boundary_55_56(self):
        """Score 55 = NEUTRAL, Score 56 = SELL_LEAN."""
        assert get_zone(55) == "NEUTRAL"
        assert get_zone(56) == "SELL_LEAN"


# --- Validation ---

class TestValidateValue:
    def test_valid_nifty_pe(self):
        assert validate_value("nifty_pe", 21.3) == 21.3

    def test_invalid_nifty_pe_too_high(self):
        assert validate_value("nifty_pe", 65.0) is None

    def test_invalid_nifty_pe_negative(self):
        assert validate_value("nifty_pe", -5.0) is None

    def test_valid_india_vix(self):
        assert validate_value("india_vix", 16.2) == 16.2

    def test_invalid_india_vix(self):
        assert validate_value("india_vix", 100.0) is None

    def test_unknown_indicator_passes_through(self):
        """Unknown indicator names should pass through without validation."""
        assert validate_value("custom_indicator", 999.0) == 999.0


# --- PSU Stocks ---

class TestPSUStocks:
    def test_known_psu(self):
        assert is_psu("SBIN") is True
        assert is_psu("NTPC") is True
        assert is_psu("COALINDIA") is True

    def test_case_insensitive(self):
        assert is_psu("sbin") is True

    def test_non_psu(self):
        assert is_psu("HDFCBANK") is False
        assert is_psu("INFY") is False

    def test_psu_boost(self):
        assert get_psu_score_boost("SBIN") == 10.0

    def test_non_psu_no_boost(self):
        assert get_psu_score_boost("HDFCBANK") == 0.0

    def test_psu_list_not_empty(self):
        assert len(PSU_STOCKS) >= 20


# --- Normalize MacmToGDP ---

class TestNormalizeMcapToGdp:
    def test_cheap_market(self):
        assert normalize_mcap_to_gdp(50) == 0.0

    def test_expensive_market(self):
        assert normalize_mcap_to_gdp(130) == 100.0

    def test_mid_range(self):
        score = normalize_mcap_to_gdp(90)
        assert 0 <= score <= 100


# --- Normalize USDINR Change ---

class TestNormalizeUsdinrChange:
    def test_inr_strengthening(self):
        assert normalize_usdinr_change(-4.0) == 15.0

    def test_inr_weakening(self):
        assert normalize_usdinr_change(4.0) == 85.0

    def test_stable(self):
        score = normalize_usdinr_change(0.0)
        assert 40 <= score <= 60


# --- Normalize SIP Flow ---

class TestNormalizeSipFlow:
    def test_normal_sip(self):
        assert normalize_sip_flow(8000) == 30.0

    def test_euphoric_sip(self):
        assert normalize_sip_flow(35000) == 80.0

    def test_mid_range(self):
        score = normalize_sip_flow(20000)
        assert 30 <= score <= 80
