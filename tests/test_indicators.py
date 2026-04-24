"""Tests for indicator normalizer and composite calculations."""

from __future__ import annotations

import pytest

from nti.indicators.normalizer import (
    normalize_pe,
    normalize_pb,
    normalize_vix,
    normalize_cpi,
    normalize_us_10y,
    normalize_crude,
    normalize_pcr,
    normalize_mmi,
    normalize_all_indicators,
)
from nti.indicators.composite import compute_custom_fg_composite, compute_global_overnight_composite
from nti.config.thresholds import get_zone


# --- PE Normalization ---

class TestNormalizePE:
    def test_very_cheap_pe(self):
        """PE <= 12 → score 0 (very cheap, strong buy)."""
        assert normalize_pe(5) == 0
        assert normalize_pe(12) == 0

    def test_very_expensive_pe(self):
        """PE >= 30 → score 100."""
        assert normalize_pe(30) == 100
        assert normalize_pe(60) == 100

    def test_mid_range_pe(self):
        """PE = 16 → should be around 25 (historical mean)."""
        score = normalize_pe(16)
        assert 20 <= score <= 30

    def test_pe_20_is_50(self):
        """PE = 20 → should be approximately 50 (fair value boundary)."""
        score = normalize_pe(20)
        assert 44 <= score <= 56

    def test_pe_none_returns_none(self):
        """None PE should return None."""
        assert normalize_pe(None) is None

    def test_pe_negative_returns_none(self):
        """Negative PE (loss-making) should be handled."""
        result = normalize_pe(-5)
        # Negative PE is typically excluded or capped
        assert result is None or result == 100

    def test_pe_zero_returns_none(self):
        """PE of 0 is invalid."""
        result = normalize_pe(0)
        assert result is None or result == 100

    def test_score_always_in_range(self):
        """Normalized score should always be 0-100."""
        for pe in [13, 15, 17, 19, 21, 23, 25, 27, 29]:
            score = normalize_pe(pe)
            assert 0 <= score <= 100, f"PE={pe} gave score={score} out of range"


# --- PB Normalization ---

class TestNormalizePB:
    def test_very_low_pb(self):
        """PB <= 1.5 → score 0."""
        assert normalize_pb(1.0) == 0
        assert normalize_pb(1.5) == 0

    def test_very_high_pb(self):
        """PB >= 5.0 → score 100."""
        assert normalize_pb(5.0) == 100
        assert normalize_pb(8.0) == 100

    def test_pb_none_returns_none(self):
        assert normalize_pb(None) is None

    def test_score_in_range(self):
        for pb in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            score = normalize_pb(pb)
            assert 0 <= score <= 100, f"PB={pb} gave score={score}"


# --- VIX Normalization (Contrarian) ---

class TestNormalizeVIX:
    def test_low_vix_complacency(self):
        """VIX < 10 → high danger (complacency = contrarian sell)."""
        score = normalize_vix(8)
        assert score >= 80, f"VIX=8 should indicate complacency danger, got {score}"

    def test_high_vix_fear(self):
        """VIX > 25 → low danger (fear = contrarian buy)."""
        score = normalize_vix(30)
        assert score <= 20, f"VIX=30 should indicate buy opportunity, got {score}"

    def test_vix_none_returns_none(self):
        assert normalize_vix(None) is None

    def test_score_in_range(self):
        for vix in [5, 10, 15, 20, 25, 30, 40, 50]:
            score = normalize_vix(vix)
            if score is not None:
                assert 0 <= score <= 100, f"VIX={vix} gave score={score}"


# --- CPI Normalization ---

class TestNormalizeCPI:
    def test_low_cpi_good(self):
        """CPI < 4% → low danger score."""
        score = normalize_cpi(3.5)
        assert score <= 25, f"CPI=3.5 should be low danger, got {score}"

    def test_high_cpi_bad(self):
        """CPI > 7% → high danger score."""
        score = normalize_cpi(8.0)
        assert score >= 70, f"CPI=8.0 should be high danger, got {score}"

    def test_cpi_none_returns_none(self):
        assert normalize_cpi(None) is None


# --- US 10Y Normalization ---

class TestNormalizeUS10Y:
    def test_low_yield_good(self):
        """US 10Y < 3.5% → low danger."""
        score = normalize_us_10y(3.0)
        assert score <= 20, f"US10Y=3.0 should be low danger, got {score}"

    def test_high_yield_bad(self):
        """US 10Y > 5.5% → high danger."""
        score = normalize_us_10y(6.0)
        assert score >= 80, f"US10Y=6.0 should be high danger, got {score}"

    def test_none_returns_none(self):
        assert normalize_us_10y(None) is None


# --- Crude Normalization ---

class TestNormalizeCrude:
    def test_low_crude_good(self):
        """Crude < $70 → low danger for India (importer)."""
        score = normalize_crude(60)
        assert score <= 25, f"Crude=60 should be low danger, got {score}"

    def test_high_crude_bad(self):
        """Crude > $95 → high danger."""
        score = normalize_crude(100)
        assert score >= 75, f"Crude=100 should be high danger, got {score}"

    def test_none_returns_none(self):
        assert normalize_crude(None) is None


# --- PCR Normalization ---

class TestNormalizePCR:
    def test_low_pcr_greed(self):
        """PCR < 0.7 → call buying = greed = danger."""
        score = normalize_pcr(0.5)
        assert score >= 70, f"PCR=0.5 should indicate greed/danger, got {score}"

    def test_high_pcr_fear(self):
        """PCR > 1.3 → put buying = fear = opportunity."""
        score = normalize_pcr(1.5)
        assert score <= 30, f"PCR=1.5 should indicate fear/opportunity, got {score}"

    def test_none_returns_none(self):
        assert normalize_pcr(None) is None


# --- MMI Normalization ---

class TestNormalizeMMI:
    def test_mmi_is_already_0_100(self):
        """MMI is already 0-100 scale."""
        assert normalize_mmi(0) == 0
        assert normalize_mmi(100) == 100
        assert normalize_mmi(50) == 50

    def test_none_returns_none(self):
        assert normalize_mmi(None) is None


# --- Zone Determination ---

class TestGetZone:
    def test_extreme_buy(self):
        assert get_zone(5) == "EXTREME_BUY"
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

    def test_boundary_values(self):
        """Test exact boundary values."""
        assert get_zone(0) == "EXTREME_BUY"
        assert get_zone(100) == "EXTREME_SELL"


# --- Composite Indicators ---

class TestCustomFGComposite:
    def test_composite_in_range(self):
        """Custom F&G composite should be 0-100."""
        indicators = {
            "vix_normalized": 50,
            "pcr_normalized": 50,
            "adv_decline_ratio": 1.0,
            "high_low_ratio": 0.5,
        }
        result = compute_custom_fg_composite(indicators)
        assert 0 <= result <= 100

    def test_extreme_greed(self):
        """High VIX complacency + low PCR (call-heavy) + strong breadth → high composite (danger)."""
        indicators = {
            "vix_normalized": 90,  # Complacency = danger
            "pcr_normalized": 80,  # Low PCR = call buying = greed = danger
            "adv_decline_ratio": 2.5,  # Broad advance = greed
            "high_low_ratio": 4.0,  # More new highs = greed = danger
        }
        result = compute_custom_fg_composite(indicators)
        assert result > 50  # Should indicate danger territory

    def test_empty_indicators(self):
        """Empty indicators should return a default value."""
        result = compute_custom_fg_composite({})
        assert 0 <= result <= 100


class TestGlobalOvernightComposite:
    def test_composite_in_range(self):
        indicators = {
            "sp500_change": -1.0,
            "nasdaq_change": -0.5,
            "nikkei_change": 0.3,
            "hang_seng_change": -0.8,
        }
        result = compute_global_overnight_composite(indicators)
        assert 0 <= result <= 100

    def test_all_markets_down(self):
        """All markets down → high danger."""
        indicators = {
            "sp500_change": -3.0,
            "nasdaq_change": -4.0,
            "nikkei_change": -2.0,
            "hang_seng_change": -3.0,
        }
        result = compute_global_overnight_composite(indicators)
        assert result > 70

    def test_empty_indicators(self):
        result = compute_global_overnight_composite({})
        assert 0 <= result <= 100
