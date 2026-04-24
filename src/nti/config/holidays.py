"""Indian Market Holiday Calendar 2026.

NSE and BSE are closed on these days. The system skips scraping on holidays
and uses the last available data.
"""

from datetime import date

# NSE/BSE Trading Holidays 2026
HOLIDAYS_2026 = {
    # Republic Day
    date(2026, 1, 26),
    # Holi
    date(2026, 3, 10),
    # Id-Ul-Fitr (Ramzan Eid)
    date(2026, 3, 31),
    # Shri Mahavir Jayanti
    date(2026, 4, 2),
    # Good Friday
    date(2026, 4, 3),
    # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 4, 14),
    # Buddha Purnima
    date(2026, 5, 4),
    # Bakri Id / Eid ul-Adha
    date(2026, 5, 27),
    # Moharram
    date(2026, 6, 26),
    # Independence Day
    date(2026, 8, 15),
    # Janmashtami
    date(2026, 8, 20),
    # Mahatma Gandhi Jayanti
    date(2026, 10, 2),
    # Dussehra (Vijay Dashmi)
    date(2026, 10, 20),
    # Diwali (Laxmi Pujan) — special Muhurat Trading session in evening
    date(2026, 11, 8),
    # Guru Nanak Jayanti
    date(2026, 11, 15),
    # Christmas
    date(2026, 12, 25),
}

# Weekends are also non-trading days (handled by checking weekday())


def is_market_holiday(check_date: date | None = None) -> bool:
    """Check if a given date is a market holiday (NSE/BSE closed).

    Args:
        check_date: Date to check. Defaults to today.

    Returns:
        True if market is closed on that date.
    """
    if check_date is None:
        check_date = date.today()

    # Weekend check
    if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
        return True

    # Holiday check
    if check_date in HOLIDAYS_2026:
        return True

    return False


def is_market_hours() -> bool:
    """Check if market is currently in trading hours (9:15 AM – 3:30 PM IST)."""
    from datetime import datetime
    import pytz

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    if is_market_holiday(now.date()):
        return False

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close
