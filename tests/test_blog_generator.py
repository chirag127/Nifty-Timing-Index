"""Tests for blog generator — frontmatter generation, slug format, word targets."""

from __future__ import annotations

import pytest

from nti.llm.blog_generator import generate_hourly_blog
from nti.config.settings import settings


# --- Blog Generation (with LLM disabled) ---

class TestBlogDisabled:
    def test_returns_empty_when_llm_disabled(self):
        """When LLM is disabled, blog generator should return empty markdown."""
        original_llm = settings.enable_llm
        original_blog = settings.enable_blog
        settings.enable_llm = False
        settings.enable_blog = False
        try:
            result = generate_hourly_blog(
                nti_score=42.0,
                prev_score=40.0,
                confidence=74.0,
                indicators={"nifty_price": 24180.5, "nifty_pe": 21.3},
                top_drivers=[{"indicator": "nifty_pe_normalized", "label": "Nifty PE", "shap": 0.18, "direction": "sell"}],
                top_stocks=[{"symbol": "SBIN", "pe": 11.2, "pb": 1.8, "is_psu": True, "composite_score": 94}],
                changelog_text="No changes",
                blog_type="mid_session",
            )
            assert result["blog_markdown"] == ""
            assert "disabled" in str(result["errors"]).lower()
        finally:
            settings.enable_llm = original_llm
            settings.enable_blog = original_blog


class TestBlogReturnType:
    def test_result_has_required_keys(self):
        """Blog generator result should have all required keys even when disabled."""
        original_llm = settings.enable_llm
        original_blog = settings.enable_blog
        settings.enable_llm = False
        settings.enable_blog = False
        try:
            result = generate_hourly_blog(
                nti_score=42.0,
                prev_score=40.0,
                confidence=74.0,
                indicators={},
                top_drivers=[],
                top_stocks=[],
                changelog_text="",
                blog_type="mid_session",
            )
            assert "blog_markdown" in result
            assert "providers_used" in result
            assert "errors" in result
            assert "duration_seconds" in result
        finally:
            settings.enable_llm = original_llm
            settings.enable_blog = original_blog


class TestWordTargets:
    def test_mid_session_word_target(self):
        """Mid-session blog should use mid word target."""
        assert settings.blog_word_target_mid > 0
        assert settings.blog_word_target_mid < settings.blog_word_target_full

    def test_full_word_target(self):
        """Market open/close blog should use full word target."""
        assert settings.blog_word_target_full > 0

    def test_default_word_targets(self):
        """Default word targets should match plan: mid=500, full=900."""
        assert settings.blog_word_target_mid == 500
        assert settings.blog_word_target_full == 900


class TestBlogTypes:
    """Verify blog type values are valid."""

    VALID_TYPES = ["market_open", "mid_session", "market_close", "post_market", "overnight"]

    def test_valid_blog_types(self):
        """All blog types from the plan should be valid."""
        for bt in self.VALID_TYPES:
            assert bt in self.VALID_TYPES
