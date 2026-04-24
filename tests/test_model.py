"""Tests for ML model: fallback inference, labeler, and predictor."""

from __future__ import annotations

import pytest
import numpy as np

from nti.model.fallback import run_fallback_inference
from nti.model.labeler import create_binary_label
from nti.config.thresholds import get_zone


# --- Fallback Inference ---

class TestFallbackInference:
    def test_score_in_range(self):
        """Fallback score must be 0-100."""
        indicators = {
            "nifty_pe": 21.3,
            "nifty_pb": 3.1,
            "nifty_dy": 1.28,
            "mmi_value": 31,
            "india_vix": 16.2,
            "pcr": 0.94,
            "fii_cash_net": -1240,
            "rbi_repo_rate": 6.5,
            "cpi_inflation": 4.2,
            "us_10y_yield": 4.32,
            "usd_inr": 83.42,
            "brent_crude": 74.5,
            "sp500_change": -0.3,
            "fii_fo_net": -500,
            "dii_cash_net": 800,
            "sip_flow_monthly_cr": 23000,
            "llm_news_danger": 42,
            "cnn_fg_value": 45,
            "custom_fg_composite": 48,
            "global_overnight": 52,
        }
        result = run_fallback_inference(indicators)
        assert "nti_score" in result
        assert 0 <= result["nti_score"] <= 100, f"Score {result['nti_score']} out of range"

    def test_fallback_returns_zone(self):
        """Fallback should return a valid zone."""
        indicators = {"nifty_pe": 21, "nifty_pb": 3.0, "india_vix": 16, "mmi_value": 35}
        result = run_fallback_inference(indicators)
        assert "zone" in result
        assert result["zone"] in [
            "EXTREME_BUY", "STRONG_BUY", "BUY_LEAN", "NEUTRAL",
            "SELL_LEAN", "STRONG_SELL", "EXTREME_SELL",
        ]

    def test_fallback_returns_confidence(self):
        """Fallback should return a confidence score."""
        indicators = {"nifty_pe": 21}
        result = run_fallback_inference(indicators)
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 100

    def test_fallback_with_empty_indicators(self):
        """Fallback should handle empty indicators gracefully."""
        result = run_fallback_inference({})
        assert 0 <= result["nti_score"] <= 100

    def test_fallback_with_partial_indicators(self):
        """Fallback should handle partial indicators."""
        indicators = {"nifty_pe": 15, "india_vix": 20}
        result = run_fallback_inference(indicators)
        assert 0 <= result["nti_score"] <= 100

    def test_fallback_is_fallback(self):
        """Fallback result should indicate it's a fallback."""
        indicators = {"nifty_pe": 21}
        result = run_fallback_inference(indicators)
        assert result.get("is_fallback", True) is True

    def test_low_pe_gives_low_score(self):
        """Very low PE should give a low danger score (buy territory)."""
        low_pe_indicators = {
            "nifty_pe": 12,  # Very cheap
            "nifty_pb": 1.5,
            "india_vix": 25,  # Elevated fear
            "mmi_value": 20,  # Fear
        }
        result = run_fallback_inference(low_pe_indicators)
        assert result["nti_score"] < 50, f"Low PE+high VIX should give buy signal, got {result['nti_score']}"

    def test_high_pe_gives_high_score(self):
        """Very high PE should give a high danger score (sell territory)."""
        high_pe_indicators = {
            "nifty_pe": 28,  # Very expensive
            "nifty_pb": 4.0,
            "india_vix": 10,  # Complacency
            "mmi_value": 80,  # Greed
        }
        result = run_fallback_inference(high_pe_indicators)
        assert result["nti_score"] > 50, f"High PE+low VIX should give sell signal, got {result['nti_score']}"


# --- Labeler ---

class TestLabeler:
    def test_positive_return_label_0(self):
        """5-day return > 2.5% → label 0 (buy moment)."""
        assert create_binary_label(0.03) == 0

    def test_negative_return_label_1(self):
        """5-day return < -2.5% → label 1 (sell moment)."""
        assert create_binary_label(-0.03) == 1

    def test_neutral_return_none(self):
        """5-day return within ±2.5% → None (too noisy)."""
        assert create_binary_label(0.01) is None
        assert create_binary_label(-0.01) is None
        assert create_binary_label(0.0) is None

    def test_boundary_positive(self):
        """Exactly +2.5% return → should be 0."""
        assert create_binary_label(0.025) == 0

    def test_boundary_negative(self):
        """Exactly -2.5% return → should be 1."""
        assert create_binary_label(-0.025) == 1

    def test_just_outside_positive(self):
        """Just above +2.5% → should be 0."""
        assert create_binary_label(0.026) == 0

    def test_just_outside_negative(self):
        """Just below -2.5% → should be 1."""
        assert create_binary_label(-0.026) == 1

    def test_just_inside_neutral(self):
        """Just inside ±2.5% → None."""
        assert create_binary_label(0.024) is None
        assert create_binary_label(-0.024) is None

    def test_large_positive(self):
        """Large positive return → 0."""
        assert create_binary_label(0.10) == 0

    def test_large_negative(self):
        """Large negative return → 1."""
        assert create_binary_label(-0.10) == 1
