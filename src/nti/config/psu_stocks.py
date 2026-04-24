"""PSU/Government Stock List — Updated quarterly.

PSU stocks get a soft preference (+10 composite score boost) in the screener.
Source: Nifty PSE + Nifty CPSE + Nifty PSU Bank index constituents.
"""

PSU_STOCKS = {
    # Nifty CPSE
    "NTPC", "ONGC", "BEL", "POWERGRID", "COALINDIA",
    "NHPC", "OIL", "NLCINDIA", "COCHINSHIP", "SJVN", "NBCC",
    # Nifty PSE (additional)
    "HPCL", "BPCL", "GAIL", "IOC", "HAL", "RECLTD", "PFC",
    "IRFC", "BHEL", "NMDC", "MOIL", "NFL",
    # PSU Banks (Nifty PSU Bank)
    "SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK",
    "INDIANB", "BANKINDIA", "UCOBANK", "CENTRALBK", "MAHABANK",
    # Insurance PSU
    "GICRE", "NIACL", "ORIENTINS", "UIICL",
}


def is_psu(symbol: str) -> bool:
    """Check if a stock symbol is a PSU/government company."""
    return symbol.upper() in PSU_STOCKS


def get_psu_score_boost(symbol: str) -> float:
    """PSU stocks get a +10 boost to composite score (soft preference)."""
    return 10.0 if is_psu(symbol) else 0.0
