"""Screener Run Pipeline — Full stock screener run (pre-market + post-market).

Steps:
1. Build stock universe from NSE EQUITY_L.csv
2. Fetch fundamentals for each stock via yfinance (batched, rate-limited)
3. Fetch analyst ratings from Finnhub (optional)
4. Apply hard filters (PE < 20, PB < 3, market cap ≥ ₹500 Cr)
5. Compute composite scores with PSU boost
6. Save results to JSON
7. Git commit and push

Runs twice daily: 6 AM IST (pre-market) and 3:45 PM IST (post-market).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.settings import settings
from nti.screener.universe import build_stock_universe
from nti.screener.fundamentals import batch_fetch_fundamentals
from nti.screener.filters import passes_hard_filters, get_soft_warnings
from nti.screener.scorer import compute_composite_scores
from nti.screener.analyst_ratings import fetch_analyst_ratings
from nti.storage.git_committer import git_commit_and_push

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

DATA_DIR = Path("data")


def run_screener(run_type: str = "pre_market", dry_run: bool = False) -> dict:
    """Run the full stock screener pipeline.

    Args:
        run_type: "pre_market" or "post_market"
        dry_run: If True, don't save files or push to git

    Returns:
        Dict with screener results and metadata
    """
    start_time = time.time()
    logger.info(f"=== NTI Screener Run ({run_type}) Started ===")

    if not settings.enable_screener:
        logger.info("Screener disabled — skipping")
        return {"skipped": True, "reason": "Screener disabled"}

    # -------------------------------------------------------------------
    # STEP 1: Build stock universe
    # -------------------------------------------------------------------
    logger.info("Step 1: Building stock universe...")

    universe = build_stock_universe()
    if universe.empty:
        logger.warning("No stocks in universe — cannot run screener")
        return {"error": "Empty stock universe"}

    logger.info(f"Stock universe: {len(universe)} stocks")

    # -------------------------------------------------------------------
    # STEP 2: Fetch fundamentals (batched, rate-limited)
    # -------------------------------------------------------------------
    logger.info("Step 2: Fetching fundamentals (this may take a while)...")

    symbols = universe["symbol"].tolist()
    fundamentals = batch_fetch_fundamentals(symbols)
    logger.info(f"Fetched fundamentals for {len(fundamentals)} stocks")

    # -------------------------------------------------------------------
    # STEP 3: Apply hard filters
    # -------------------------------------------------------------------
    logger.info("Step 3: Applying hard filters (PE < 20, PB < 3, mcap ≥ ₹500 Cr)...")

    passing_stocks = []
    exclusion_summary = {
        "pe_too_high": 0,
        "pb_too_high": 0,
        "market_cap_too_small": 0,
        "missing_data": 0,
        "negative_pe": 0,
    }

    for stock in fundamentals:
        pe = stock.get("pe")
        pb = stock.get("pb")
        mcap = stock.get("market_cap_cr", 0)

        # Count exclusions for summary
        if pe is None or pe <= 0:
            exclusion_summary["negative_pe" if pe is not None and pe <= 0 else "missing_data"] += 1
        elif pe >= settings.max_pe:
            exclusion_summary["pe_too_high"] += 1

        if pb is None:
            exclusion_summary["missing_data"] += 1
        elif pb >= settings.max_pb:
            exclusion_summary["pb_too_high"] += 1

        if mcap < settings.min_market_cap_cr:
            exclusion_summary["market_cap_too_small"] += 1

        if passes_hard_filters(stock):
            # Add soft filter warnings
            stock["warnings"] = get_soft_warnings(stock)

            # Add PSU status
            from nti.config.psu_stocks import is_psu, get_psu_score_boost
            stock["is_psu"] = is_psu(stock.get("symbol", ""))
            stock["psu_boost"] = get_psu_score_boost(stock.get("symbol", ""))

            passing_stocks.append(stock)

    logger.info(f"Stocks passing hard filters: {len(passing_stocks)}")
    logger.info(f"Exclusion summary: {exclusion_summary}")

    # -------------------------------------------------------------------
    # STEP 4: Compute composite scores
    # -------------------------------------------------------------------
    logger.info("Step 4: Computing composite scores...")

    scored_stocks = compute_composite_scores(passing_stocks)

    # Sort by composite score descending
    scored_stocks.sort(key=lambda s: s.get("composite_score", 0), reverse=True)

    # Take top 50
    top_picks = scored_stocks[:50]

    # -------------------------------------------------------------------
    # STEP 5: Fetch analyst ratings for top picks (optional)
    # -------------------------------------------------------------------
    if settings.finnhub_api_key and top_picks:
        logger.info("Step 5: Fetching analyst ratings for top picks...")
        for stock in top_picks[:20]:  # Rate limit: only top 20
            try:
                ratings = fetch_analyst_ratings(stock["symbol"])
                stock["analyst_buy_pct"] = ratings.get("analyst_buy_pct")
                stock["analyst_count"] = ratings.get("analyst_count")
            except Exception as e:
                logger.warning(f"Analyst ratings failed for {stock['symbol']}: {e}")
    else:
        logger.info("Step 5: Skipping analyst ratings (no Finnhub key)")

    # -------------------------------------------------------------------
    # STEP 6: Add MTF risk panel
    # -------------------------------------------------------------------
    for stock in top_picks:
        price = stock.get("current_price", 0)
        if price > 0:
            leverage = settings.mtf_leverage
            stock["mtf_risk"] = {
                "leverage_3x_margin_required_pct": round(100 / leverage, 1),
                "ten_pct_fall_capital_loss_pct": round(leverage * 10, 1),
                "margin_call_distance_pct": round((1 - 1 / leverage) * 100 / leverage * 100, 1),
            }

    # -------------------------------------------------------------------
    # STEP 7: Build output JSON
    # -------------------------------------------------------------------
    now_ist = datetime.now(IST)

    # Add "why_picked" explanation for each stock
    for rank, stock in enumerate(top_picks, 1):
        stock["rank"] = rank
        psu_tag = "PSU bank" if stock.get("is_psu") and "bank" in stock.get("industry", "").lower() else \
                  "PSU" if stock.get("is_psu") else ""
        # Pre-compute optional parts to avoid nested f-string with escaped quotes
        roe_part = f", ROE {stock.get('roe', 0):.1f}%" if stock.get('roe') else ''
        div_part = f", Div Yield {stock.get('dividend_yield', 0):.1f}%" if stock.get('dividend_yield') else ''

        stock["why_picked"] = (
            f"{psu_tag + ' ' if psu_tag else ''}"
            f"Low PE ({stock.get('pe', 0):.1f}x), "
            f"PB ({stock.get('pb', 0):.1f}x)"
            f"{roe_part}{div_part}"
        ).strip()

    output = {
        "screened_at": now_ist.isoformat(),
        "run_type": run_type,
        "universe_size": len(universe),
        "passing_filters": len(passing_stocks),
        "top_picks": top_picks,
        "exclusion_summary": exclusion_summary,
        "psu_stocks_count": sum(1 for s in top_picks if s.get("is_psu")),
    }

    # -------------------------------------------------------------------
    # STEP 8: Save results
    # -------------------------------------------------------------------
    if not dry_run:
        screener_dir = DATA_DIR / "screener"
        screener_dir.mkdir(parents=True, exist_ok=True)

        filename = f"latest_{run_type}.json"
        json_path = screener_dir / filename

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"Screener results saved to {json_path}")
        except OSError as e:
            logger.error(f"Failed to save screener results: {e}")

        # Git commit
        git_commit_and_push(
            message=f"{run_type.replace('_', ' ').title()} screener {now_ist.strftime('%Y-%m-%d')}",
            paths=["data/screener/"],
        )

    duration = time.time() - start_time
    logger.info(
        f"=== Screener Run Complete ({duration:.1f}s) ===\n"
        f"  Universe: {len(universe)}\n"
        f"  Passing filters: {len(passing_stocks)}\n"
        f"  Top picks: {len(top_picks)}\n"
        f"  PSU picks: {sum(1 for s in top_picks if s.get('is_psu'))}"
    )

    return {
        "universe_size": len(universe),
        "passing_filters": len(passing_stocks),
        "top_picks_count": len(top_picks),
        "psu_count": sum(1 for s in top_picks if s.get("is_psu")),
        "duration_seconds": duration,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="NTI Stock Screener")
    parser.add_argument("--type", choices=["pre_market", "post_market"], default="pre_market")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_screener(run_type=args.type, dry_run=args.dry_run)
    print(f"\nScreener result: {json.dumps(result, indent=2, default=str)}")
