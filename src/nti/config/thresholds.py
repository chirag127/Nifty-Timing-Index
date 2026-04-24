"""Normalization thresholds and validation ranges for all 30 indicators.

All indicators normalized 0–100: 0 = BUY opportunity (low danger), 100 = SELL danger (high danger).
"""

from nti.config.settings import settings

# ---------------------------------------------------------------------------
# NTI Score Zones
# ---------------------------------------------------------------------------

ZONES = {
    (0, 15): "EXTREME_BUY",
    (16, 30): "STRONG_BUY",
    (31, 45): "BUY_LEAN",
    (46, 55): "NEUTRAL",
    (56, 69): "SELL_LEAN",
    (70, 84): "STRONG_SELL",
    (85, 100): "EXTREME_SELL",
}


def get_zone(score: float) -> str:
    """Get zone name from NTI score."""
    for (low, high), zone in ZONES.items():
        if low <= score <= high:
            return zone
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Normalization Functions — Tier 1: Fundamental Valuation
# ---------------------------------------------------------------------------


def normalize_pe(pe: float | None) -> float | None:
    """Nifty P/E → 0–100 score. Lower PE = buy opportunity (0), higher PE = sell danger (100)."""
    if pe is None:
        return None
    if pe <= 0:
        return None  # Invalid PE
    if pe <= 12:
        return 0.0
    if pe >= 30:
        return 100.0
    return min(100.0, max(0.0, (pe - 12) / (30 - 12) * 100))


def normalize_pb(pb: float | None) -> float | None:
    """Nifty P/B → 0–100 score."""
    if pb is None:
        return None
    if pb <= 0:
        return None  # Invalid PB
    if pb <= 1.5:
        return 0.0
    if pb >= 5.0:
        return 100.0
    return min(100.0, max(0.0, (pb - 1.5) / (5.0 - 1.5) * 100))


def normalize_dividend_yield(dy: float) -> float:
    """Dividend yield → 0–100 score. Higher yield = buy signal (lower score)."""
    if dy >= 2.0:
        return 0.0
    if dy <= 0.5:
        return 100.0
    return min(100.0, max(0.0, (2.0 - dy) / (2.0 - 0.5) * 100))


def normalize_earnings_yield_spread(spread: float) -> float:
    """Earnings yield vs bond spread → 0–100. Positive spread = buy (low score)."""
    if spread >= 0.04:
        return 0.0
    if spread <= -0.02:
        return 100.0
    return min(100.0, max(0.0, (-0.02 - spread) / (-0.02 - 0.04) * 100))


def normalize_midcap_pe(pe: float) -> float:
    """Midcap P/E → 0–100 score. Same scale as Nifty PE but wider range."""
    if pe <= 15:
        return 0.0
    if pe >= 40:
        return 100.0
    return min(100.0, max(0.0, (pe - 15) / (40 - 15) * 100))


def normalize_buffett_indicator(pct: float) -> float:
    """Market Cap to GDP % → 0–100. >120% = expensive (100), <60% = cheap (0)."""
    if pct <= 60:
        return 0.0
    if pct >= 120:
        return 100.0
    return min(100.0, max(0.0, (pct - 60) / (120 - 60) * 100))


# ---------------------------------------------------------------------------
# Normalization Functions — Tier 2: Sentiment (Contrarian — inverted)
# ---------------------------------------------------------------------------


def normalize_mmi(mmi: float | None) -> float | None:
    """Tickertape MMI → 0–100. MMI is already 0–100; high MMI (greed) = sell danger."""
    if mmi is None:
        return None
    return min(100.0, max(0.0, mmi))


def normalize_vix(vix: float | None) -> float | None:
    """India VIX → 0–100. Contrarian: LOW VIX (complacency) = danger (high score)."""
    if vix is None:
        return None
    if vix <= 10:
        return 80.0  # Complacency = danger
    if vix >= 30:
        return 10.0  # Fear = opportunity (low score)
    return min(100.0, max(0.0, 100 - (vix - 10) / (30 - 10) * 100))


def normalize_pcr(pcr: float | None) -> float | None:
    """Put/Call Ratio → 0–100. Low PCR (call buying = greed) = danger (high score)."""
    if pcr is None:
        return None
    if pcr <= 0.7:
        return 80.0
    if pcr >= 1.3:
        return 20.0
    return min(100.0, max(0.0, (1.3 - pcr) / (1.3 - 0.7) * (80 - 20) + 20))


def normalize_fii_cash(net_cr: float | None) -> float | None:
    """FII net cash flow → 0–100. Sustained selling = fear = contrarian buy (low score)."""
    if net_cr is None:
        return None
    if net_cr <= -2000:
        return 15.0  # Heavy FII selling = fear = contrarian buy
    if net_cr >= 2000:
        return 85.0  # Heavy FII buying = greed = danger
    return min(100.0, max(0.0, (net_cr + 2000) / 4000 * 70 + 15))


# ---------------------------------------------------------------------------
# Normalization Functions — Tier 3: Macro Fundamentals
# ---------------------------------------------------------------------------


def normalize_cpi(cpi: float | None) -> float | None:
    """CPI Inflation → 0–100. High CPI = sell danger (high score)."""
    if cpi is None:
        return None
    if cpi <= 4:
        return 20.0
    if cpi >= 7:
        return 85.0
    return min(100.0, max(0.0, (cpi - 4) / (7 - 4) * (85 - 20) + 20))


def normalize_us_10y(yield_pct: float | None) -> float | None:
    """US 10-Year Yield → 0–100. Higher yields = headwind for EM (high score)."""
    if yield_pct is None:
        return None
    if yield_pct <= 3.5:
        return 15.0
    if yield_pct >= 5.5:
        return 85.0
    return min(100.0, max(0.0, (yield_pct - 3.5) / (5.5 - 3.5) * (85 - 15) + 15))


def normalize_usdinr(usdinr: float) -> float:
    """USD/INR → 0–100. Depreciating INR (rising) = danger (high score)."""
    if usdinr <= 75:
        return 15.0
    if usdinr >= 90:
        return 85.0
    return min(100.0, max(0.0, (usdinr - 75) / (90 - 75) * (85 - 15) + 15))


def normalize_crude(crude: float | None) -> float | None:
    """Brent Crude → 0–100. Higher crude = headwind for India (high score)."""
    if crude is None:
        return None
    if crude <= 70:
        return 15.0
    if crude >= 95:
        return 80.0
    return min(100.0, max(0.0, (crude - 70) / (95 - 70) * (80 - 15) + 15))


# Alias for normalizer.py imports (same logic as normalize_buffett_indicator)
normalize_mcap_to_gdp = normalize_buffett_indicator


def normalize_usdinr_change(change_pct: float | None) -> float | None:
    """USD/INR 30-day change → 0–100. Depreciating INR (positive change) = danger."""
    if change_pct is None:
        return None
    if change_pct <= -3:
        return 15.0  # INR strengthening = good
    if change_pct >= 3:
        return 85.0  # INR weakening = bad
    return min(100.0, max(0.0, (change_pct + 3) / 6 * (85 - 15) + 15))


def normalize_sip_flow(sip_cr: float | None) -> float | None:
    """AMFI Monthly SIP flows → 0–100. Very high SIP = retail euphoria (danger)."""
    if sip_cr is None:
        return None
    if sip_cr <= 10000:
        return 30.0  # Normal SIP flow = neutral
    if sip_cr >= 30000:
        return 80.0  # Very high SIP = euphoria = danger
    return min(100.0, max(0.0, (sip_cr - 10000) / 20000 * (80 - 30) + 30))


def normalize_sp500_change(change_pct: float) -> float:
    """S&P 500 daily change → 0–100. Very positive = mild greed signal (higher score)."""
    if change_pct <= -2:
        return 20.0  # Fear = opportunity
    if change_pct >= 2:
        return 80.0  # Euphoria = danger
    return min(100.0, max(0.0, (change_pct + 2) / 4 * 60 + 20))


# ---------------------------------------------------------------------------
# Validation Ranges — sanity checks for scraped data
# ---------------------------------------------------------------------------

VALIDATION_RANGES = {
    "nifty_pe": (5, 60),
    "nifty_pb": (0.5, 8),
    "india_vix": (5, 90),
    "mmi": (0, 100),
    "pcr": (0.1, 5.0),
    "fii_cash_net": (-15000, 15000),  # Crores
    "us_10y_yield": (0, 10),
    "usd_inr": (60, 120),
    "brent_crude": (30, 200),
    "cpi_inflation": (-2, 15),
    "nifty_price": (1000, 100000),
    "dividend_yield": (0, 5),
    "gift_nifty_price": (1000, 100000),
    "gift_nifty_change_pct": (-10, 10),
}


def validate_value(name: str, value: float) -> float | None:
    """Validate a scraped value against expected range. Returns None if invalid."""
    if name not in VALIDATION_RANGES:
        return value
    low, high = VALIDATION_RANGES[name]
    if low <= value <= high:
        return value
    return None  # Out of range — treat as missing


# ---------------------------------------------------------------------------
# Indicator Weights (from plan.md Section 4)
# ---------------------------------------------------------------------------

INDICATOR_WEIGHTS = {
    # Tier 1: Fundamental Valuation (35% total)
    "nifty_pe_normalized": 0.10,
    "nifty_pb_normalized": 0.08,
    "earnings_yield_bond_spread": 0.06,
    "dividend_yield_normalized": 0.05,
    "mcap_to_gdp_percentile": 0.04,
    "midcap_pe_normalized": 0.02,
    # Tier 2: Sentiment (25% total)
    "mmi_score": 0.08,
    "vix_normalized": 0.06,
    "pcr_normalized": 0.04,
    "custom_fg_composite": 0.04,
    "cnn_fg": 0.02,
    "fii_cash_5d_avg_normalized": 0.01,
    # Tier 3: Macro (25% total)
    "rbi_rate_direction": 0.06,
    "cpi_normalized": 0.05,
    "us_10y_normalized": 0.05,
    "usdinr_30d_change": 0.04,
    "crude_normalized": 0.03,
    "sp500_change_normalized": 0.02,
    # Tier 4: Institutional Flow (10% total)
    "fii_fo_net_normalized": 0.04,
    "dii_net_normalized": 0.03,
    "sip_flow_normalized": 0.02,
    "gift_nifty_change_normalized": 0.01,
    # Tier 5: LLM News (5% total)
    "llm_news_danger_score": 0.02,
    "llm_policy_flag": 0.01,
    "llm_geopolitical_score": 0.01,
    "global_overnight_normalized": 0.01,
}
