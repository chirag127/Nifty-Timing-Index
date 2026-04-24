"""Tests for scraper modules — validation ranges and data format checks.

Note: These tests do NOT make actual HTTP requests to NSE/Yahoo.
They test the validation logic, session creation, and data parsing.
"""

from __future__ import annotations

import pytest

from nti.config.thresholds import validate_value, VALIDATION_RANGES


# --- Validation Ranges ---

class TestValidationRanges:
    def test_nifty_pe_range(self):
        """Nifty PE should be validated between 5 and 60."""
        low, high = VALIDATION_RANGES["nifty_pe"]
        assert low == 5
        assert high == 60

    def test_india_vix_range(self):
        """India VIX should be validated between 5 and 90."""
        low, high = VALIDATION_RANGES["india_vix"]
        assert low == 5
        assert high == 90

    def test_usd_inr_range(self):
        """USD/INR should be validated between 60 and 120."""
        low, high = VALIDATION_RANGES["usd_inr"]
        assert low == 60
        assert high == 120

    def test_brent_crude_range(self):
        """Brent crude should be validated between 30 and 200."""
        low, high = VALIDATION_RANGES["brent_crude"]
        assert low == 30
        assert high == 200

    def test_fii_cash_net_range(self):
        """FII net cash should be validated between -15000 and 15000 Cr."""
        low, high = VALIDATION_RANGES["fii_cash_net"]
        assert low == -15000
        assert high == 15000


# --- validate_value Edge Cases ---

class TestValidateValueEdgeCases:
    def test_boundary_low(self):
        """Value at the lower boundary should pass."""
        assert validate_value("nifty_pe", 5) == 5

    def test_boundary_high(self):
        """Value at the upper boundary should pass."""
        assert validate_value("nifty_pe", 60) == 60

    def test_just_below_range(self):
        """Value just below range should fail validation."""
        assert validate_value("nifty_pe", 4.9) is None

    def test_just_above_range(self):
        """Value just above range should fail validation."""
        assert validate_value("nifty_pe", 60.1) is None

    def test_zero_value(self):
        """Zero value for PE should fail (below range)."""
        assert validate_value("nifty_pe", 0) is None

    def test_negative_fii_is_valid(self):
        """Negative FII flow (selling) should be valid."""
        assert validate_value("fii_cash_net", -5000) == -5000

    def test_positive_fii_is_valid(self):
        """Positive FII flow (buying) should be valid."""
        assert validate_value("fii_cash_net", 5000) == 5000


# --- Scraper Data Format ---

class TestScraperOutputFormat:
    def test_nse_indices_module_has_functions(self):
        """NSE indices module should expose scraping functions."""
        from nti.scrapers import nse_indices
        assert hasattr(nse_indices, "scrape_nse_index_data")
        assert hasattr(nse_indices, "scrape_fii_dii")

    def test_yahoo_finance_module_has_functions(self):
        """Yahoo finance module should expose scraping functions."""
        from nti.scrapers import yahoo_finance
        assert hasattr(yahoo_finance, "scrape_global_markets")

    def test_fred_api_module_has_functions(self):
        """FRED API module should expose scraping functions."""
        from nti.scrapers import fred_api
        assert hasattr(fred_api, "scrape_us_10y_yield")

    def test_rbi_data_module_has_functions(self):
        """RBI data module should expose scraping functions."""
        from nti.scrapers import rbi_data
        assert hasattr(rbi_data, "scrape_rbi_rate")

    def test_mospi_data_module_has_functions(self):
        """MOSPI data module should expose scraping functions."""
        from nti.scrapers import mospi_data
        assert hasattr(mospi_data, "scrape_cpi_inflation")

    def test_cnn_fear_greed_module_has_functions(self):
        """CNN F&G module should expose scraping functions."""
        from nti.scrapers import cnn_fear_greed
        assert hasattr(cnn_fear_greed, "scrape_cnn_fg")

    def test_rss_news_module_has_functions(self):
        """RSS news module should expose scraping functions."""
        from nti.scrapers import rss_news
        assert hasattr(rss_news, "fetch_recent_news")

    def test_tickertape_mmi_module_has_functions(self):
        """Tickertape MMI module should expose scraping functions."""
        from nti.scrapers import tickertape_mmi
        assert hasattr(tickertape_mmi, "scrape_mmi")
