"""Tests for email notification system — zone change alerts, big move alerts, provider fallback."""

from __future__ import annotations

import pytest

from nti.notifications.email_sender import (
    send_zone_change_alert,
    send_big_move_alert,
    MultiEmailSender,
)
from nti.config.settings import settings


# --- Email Logic Tests (without actually sending) ---

class TestEmailDisabled:
    def test_email_disabled_returns_false(self):
        """When email is disabled, send_alert should return False."""
        sender = MultiEmailSender()
        original = settings.alert_email_to
        settings.alert_email_to = ""
        try:
            result = sender.send_alert("Test Subject", "<h1>Test</h1>")
            assert result is False
        finally:
            settings.alert_email_to = original

    def test_no_enabled_providers_returns_false(self):
        """When no email providers are enabled, send_alert should return False."""
        sender = MultiEmailSender()
        # With default settings (no env vars), no providers should have API keys
        # The function should handle this gracefully
        if not settings.get_enabled_email_providers():
            result = sender.send_alert("Test", "<p>Test</p>")
            assert result is False


class TestZoneChangeAlert:
    def test_alert_callable(self):
        """Zone change alert function should be callable."""
        assert callable(send_zone_change_alert)

    def test_alert_accepts_correct_args(self):
        """Zone change alert should accept the documented arguments."""
        import inspect
        sig = inspect.signature(send_zone_change_alert)
        param_names = list(sig.parameters.keys())
        assert "from_zone" in param_names
        assert "to_zone" in param_names
        assert "score" in param_names
        assert "nifty_price" in param_names
        assert "drivers" in param_names

    def test_alert_returns_bool(self):
        """Zone change alert should return a boolean."""
        # Will return False since no email providers configured in test env
        result = send_zone_change_alert(
            from_zone="NEUTRAL",
            to_zone="SELL_LEAN",
            score=57.0,
            nifty_price=24180.5,
            drivers=["nifty_pe_normalized", "mmi_score"],
        )
        assert isinstance(result, bool)


class TestBigMoveAlert:
    def test_alert_callable(self):
        """Big move alert function should be callable."""
        assert callable(send_big_move_alert)

    def test_alert_accepts_correct_args(self):
        """Big move alert should accept the documented arguments."""
        import inspect
        sig = inspect.signature(send_big_move_alert)
        param_names = list(sig.parameters.keys())
        assert "prev_score" in param_names
        assert "curr_score" in param_names
        assert "nifty_price" in param_names

    def test_alert_returns_bool(self):
        """Big move alert should return a boolean."""
        result = send_big_move_alert(
            prev_score=40.0,
            curr_score=57.0,
            nifty_price=24180.5,
        )
        assert isinstance(result, bool)


class TestBigMoveThreshold:
    def test_default_threshold(self):
        """Default big move threshold should be 10 points."""
        assert settings.alert_big_move_threshold == 10

    def test_threshold_is_positive(self):
        """Big move threshold should be positive."""
        assert settings.alert_big_move_threshold > 0


class TestEmailProviderConfig:
    def test_gmail_is_default_provider(self):
        """Gmail should be in the list of email providers."""
        provider_names = [p.name for p in settings.email_providers]
        assert "gmail" in provider_names

    def test_provider_has_required_fields(self):
        """Each email provider should have name, enabled, and from_address."""
        for provider in settings.email_providers:
            assert hasattr(provider, "name")
            assert hasattr(provider, "enabled")
            assert hasattr(provider, "from_address")

    def test_alert_on_zone_change_default(self):
        """Zone change alerts should be enabled by default."""
        assert settings.alert_on_zone_change is True

    def test_alert_on_big_move_default(self):
        """Big move alerts should be enabled by default."""
        assert settings.alert_on_big_move is True
