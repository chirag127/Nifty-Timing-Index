"""Step 2: Analyze — Normalize indicators, run inference, analyze news, generate changelog.

Loads raw indicators from step_scrape.json (or accepts them directly),
normalizes, runs ML/fallback inference, analyzes news, and generates
changelog. Saves intermediate state to step_analyze.json.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.settings import settings
from nti.config.thresholds import get_zone

# Indicators
from nti.indicators.normalizer import normalize_all_indicators
from nti.indicators.composite import compute_custom_fg_composite, compute_global_overnight_composite

# Model
from nti.model.predictor import run_inference
from nti.model.fallback import run_fallback_inference

# Blog & News
from nti.llm.news_analyzer import analyze_news

# Changelog
from nti.changelog.generator import generate_changelog, load_previous_run, save_current_run

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")
API_DIR = DATA_DIR / "api"
STEP_FILE = API_DIR / "step_analyze.json"
SCRAPE_FILE = API_DIR / "step_scrape.json"


def _load_scrape_data() -> dict:
    """Load scrape data from the intermediate file if not passed directly."""
    if not SCRAPE_FILE.exists():
        raise FileNotFoundError(
            f"Scrape data not found at {SCRAPE_FILE}. "
            "Run the scrape step first."
        )
    with open(SCRAPE_FILE) as f:
        data = json.load(f)
    # Remove internal metadata keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def run_analyze_step(raw_indicators: dict | None = None, force: bool = False) -> dict:
    """Run the analyze step: normalize, inference, news, changelog.

    Args:
        raw_indicators: Dict of raw indicator values. If None, loads from step_scrape.json
        force: If True, re-run even if step_analyze.json already exists

    Returns:
        Dict with keys: raw_indicators, nti_result, news_result, changelog_text
    """
    now_ist = datetime.now(IST)
    logger.info(f"=== Step 2: Analyze (started {now_ist.isoformat()}) ===")

    # Load raw indicators if not provided
    if raw_indicators is None:
        raw_indicators = _load_scrape_data()

    # Check for existing step output (resume support)
    if not force and STEP_FILE.exists():
        try:
            with open(STEP_FILE) as f:
                cached = json.load(f)
            cached_time = cached.get("_analyzed_at", "")
            if cached_time:
                cached_dt = datetime.fromisoformat(cached_time)
                age_minutes = (now_ist - cached_dt).total_seconds() / 60
                if age_minutes < 55:
                    logger.info(f"Reusing analyze data from {age_minutes:.0f} min ago")
                    # Still return the cached data but update raw_indicators
                    cached.pop("_analyzed_at", None)
                    return cached
        except (json.JSONDecodeError, ValueError, OSError):
            pass

    start_time = time.time()

    # -------------------------------------------------------------------
    # Normalize indicators
    # -------------------------------------------------------------------
    logger.info("Normalizing indicators...")
    normalized = normalize_all_indicators(raw_indicators)
    logger.info(f"Normalized {len(normalized)} indicator scores")

    # Compute composites
    vix_norm = normalized.get("vix_normalized")
    pcr_norm = normalized.get("pcr_normalized")
    ad_ratio = raw_indicators.get("advance_decline_ratio")
    hl_ratio = raw_indicators.get("high_low_ratio")

    custom_fg = compute_custom_fg_composite(None, vix_norm, pcr_norm, ad_ratio, hl_ratio)
    if custom_fg is not None:
        raw_indicators["custom_fg_composite"] = custom_fg
        normalized["custom_fg_composite"] = custom_fg

    # Global overnight composite
    global_overnight = compute_global_overnight_composite(
        None,
        raw_indicators.get("dow_jones_change"),
        raw_indicators.get("nasdaq_change"),
        raw_indicators.get("nikkei_change"),
        raw_indicators.get("hang_seng_change"),
    )
    if global_overnight is not None:
        raw_indicators["global_overnight"] = global_overnight

    # -------------------------------------------------------------------
    # LLM News Analysis (with timeout protection)
    # -------------------------------------------------------------------
    logger.info("Analyzing news...")
    news_result = {}
    try:
        news_result = analyze_news()
        if news_result:
            raw_indicators["llm_news_danger"] = news_result.get("danger_score")
            raw_indicators["llm_policy_flag"] = news_result.get("policy_flag")
            raw_indicators["llm_geopolitical_score"] = news_result.get("geopolitical_risk")
    except Exception as e:
        logger.warning(f"News analysis failed (continuing without it): {e}")
        news_result = {
            "danger_score": raw_indicators.get("llm_news_danger", 50),
            "policy_flag": "none",
            "geopolitical_risk": 50,
        }

    # -------------------------------------------------------------------
    # Run ML inference
    # -------------------------------------------------------------------
    logger.info("Running NTI inference...")

    # Load previous run data for lagged features
    previous_run = load_previous_run()
    prev_score = previous_run.get("nti_score")
    score_yesterday = previous_run.get("nti_score_yesterday")
    pe_5d_ago = previous_run.get("nifty_pe_5d_ago")
    vix_5d_ago = previous_run.get("india_vix_5d_ago")

    if settings.enable_model:
        nti_result = run_inference(
            raw_indicators,
            previous_score=prev_score,
            score_yesterday=score_yesterday,
            pe_5d_ago=pe_5d_ago,
            vix_5d_ago=vix_5d_ago,
        )
    else:
        nti_result = run_fallback_inference(raw_indicators)

    nti_score = nti_result.get("nti_score", 50)
    zone = nti_result.get("zone", get_zone(nti_score))
    confidence = nti_result.get("confidence", 50)
    is_fallback = nti_result.get("is_fallback", True)
    top_drivers = nti_result.get("top_drivers", [])

    logger.info(f"NTI Score: {nti_score:.1f} ({zone}) | Confidence: {confidence:.0f}% | Fallback: {is_fallback}")

    # Store in indicators for writing
    raw_indicators["nti_score"] = nti_score
    raw_indicators["nti_score_prev"] = prev_score
    raw_indicators["zone"] = zone
    raw_indicators["zone_prev"] = previous_run.get("zone", "UNKNOWN")
    raw_indicators["confidence"] = confidence
    raw_indicators["is_fallback"] = is_fallback
    raw_indicators["model_version"] = nti_result.get("model_version", "rule-based")

    # -------------------------------------------------------------------
    # Generate changelog
    # -------------------------------------------------------------------
    logger.info("Generating changelog...")

    prev_stocks = previous_run.get("top_stocks", [])
    changelog_text = generate_changelog(
        current=raw_indicators,
        previous=previous_run,
        current_stocks=[],
        previous_stocks=prev_stocks,
    )

    # -------------------------------------------------------------------
    # Save intermediate state
    # -------------------------------------------------------------------
    result = {
        "raw_indicators": raw_indicators,
        "nti_result": nti_result,
        "news_result": news_result,
        "changelog_text": changelog_text,
        "normalized": normalized,
        "_analyzed_at": now_ist.isoformat(),
    }

    API_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Don't save the full normalized dict to keep file size manageable
        save_data = {k: v for k, v in result.items() if k != "normalized"}
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Analyze step saved to {STEP_FILE}")
    except OSError as e:
        logger.error(f"Failed to save analyze step: {e}")

    duration = time.time() - start_time
    logger.info(f"=== Step 2: Analyze complete ({duration:.1f}s) ===")

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="NTI Analyze Step")
    parser.add_argument("--force", action="store_true", help="Force re-analysis even if step file exists")
    args = parser.parse_args()

    result = run_analyze_step(force=args.force)
    nti = result.get("nti_result", {})
    print(f"\nNTI Score: {nti.get('nti_score', 0):.1f} ({nti.get('zone', 'N/A')})")
    print(f"Confidence: {nti.get('confidence', 0):.0f}%")
