"""Hourly Pipeline — Orchestrates the full hourly NTI run using modular steps.

The pipeline is split into 4 independent steps, each with its own
timeout and intermediate state persistence (data/api/step_*.json):

1. SCRAPE  — Scrape all indicators + backfill missing data (~60s)
2. ANALYZE — Normalize, inference, news analysis, changelog (~30s + LLM)
3. BLOG    — Generate blog post via LLM fusion workflow (~90s)
4. PUBLISH — Write CSV/JSON/blog files, alerts, git push (~15s)

If a step times out or fails, the pipeline:
- Saves whatever data was collected so far
- Continues with remaining steps where possible
- Resumes from the last successful step on next run

Each step can also be run independently:
    uv run python -m nti.pipelines.steps.scrape
    uv run python -m nti.pipelines.steps.analyze
    uv run python -m nti.pipelines.steps.blog
    uv run python -m nti.pipelines.steps.publish

This is the main entry point called by GitHub Actions every hour.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

from nti.config.thresholds import get_zone
from nti.config.holidays import is_market_holiday

from nti.pipelines.steps.scrape import run_scrape_step
from nti.pipelines.steps.analyze import run_analyze_step
from nti.pipelines.steps.blog import run_blog_step
from nti.pipelines.steps.publish import run_publish_step

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Per-step timeout in seconds (generous limits)
STEP_TIMEOUTS = {
    "scrape": 180,   # 3 min — scrapers can be slow
    "analyze": 120,  # 2 min — includes news analysis LLM call
    "blog": 180,     # 3 min — LLM blog generation
    "publish": 60,   # 1 min — file writes + git push
}

# Total pipeline timeout (should fit within GitHub Actions 10-min default)
TOTAL_TIMEOUT = 600  # 10 minutes


def _run_step_with_timeout(step_name: str, step_fn, *args, **kwargs) -> tuple[dict | None, str | None]:
    """Run a pipeline step with timeout and error handling.

    Uses multiprocessing so timed-out steps can actually be terminated
    (threading cannot kill a hung thread).

    Args:
        step_name: Name of the step (for logging)
        step_fn: Callable to run
        *args, **kwargs: Arguments to pass to step_fn

    Returns:
        Tuple of (result_dict or None, error_message or None)
    """
    import multiprocessing

    timeout = STEP_TIMEOUTS.get(step_name, 120)
    logger.info(f"Starting step '{step_name}' (timeout: {timeout}s)")

    def _target(result_queue):
        """Run in child process; put result or exception into queue."""
        try:
            result = step_fn(*args, **kwargs)
            result_queue.put(("ok", result))
        except Exception as e:
            result_queue.put(("error", f"Step '{step_name}' failed: {e}"))

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_target, args=(result_queue,), daemon=True)
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        # Process timed out — actually kill it
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=3)
        result_queue.close()
        result_queue.join_thread()
        error = f"Step '{step_name}' timed out after {timeout}s"
        logger.error(error)
        return None, error

    # Get result from queue
    if not result_queue.empty():
        status, payload = result_queue.get_nowait()
        result_queue.close()
        result_queue.join_thread()
        if status == "ok":
            return payload, None
        else:
            logger.error(payload)
            return None, payload

    # Process exited without putting anything in queue (crash)
    result_queue.close()
    result_queue.join_thread()
    exitcode = proc.exitcode
    error = f"Step '{step_name}' crashed (exit code {exitcode})"
    logger.error(error)
    return None, error


def run_hourly_pipeline(dry_run: bool = False) -> dict:
    """Run the full hourly NTI pipeline using modular steps.

    Each step is independently timed out. If a step fails, the pipeline
    attempts to continue with remaining steps using any available data
    from intermediate files (data/api/step_*.json).

    Args:
        dry_run: If True, don't write files or push to git

    Returns:
        Dict with pipeline results and metadata
    """
    start_time = time.time()
    pipeline_errors: list[str] = []
    step_results: dict = {}

    now_ist = datetime.now(IST)
    logger.info(f"=== NTI Hourly Pipeline Started at {now_ist.isoformat()} ===")

    # Check for holidays
    if is_market_holiday(now_ist.date()):
        logger.info("Today is a market holiday — running in limited mode")

    # Default values for step results
    raw_indicators = None
    analyze_data = None
    blog_data = None

    # -------------------------------------------------------------------
    # STEP 1: Scrape all indicators
    # -------------------------------------------------------------------
    raw_indicators, scrape_error = _run_step_with_timeout(
        "scrape", run_scrape_step, force=False
    )
    if scrape_error:
        pipeline_errors.append(scrape_error)
    step_results["scrape"] = "ok" if raw_indicators else "failed"

    # -------------------------------------------------------------------
    # STEP 2: Analyze (normalize, inference, news, changelog)
    # -------------------------------------------------------------------
    if time.time() - start_time > TOTAL_TIMEOUT:
        logger.error(f"Total pipeline timeout exceeded ({TOTAL_TIMEOUT}s) — skipping analyze")
        pipeline_errors.append("Pipeline timed out before analyze step")
        step_results["analyze"] = "skipped"
    else:
        analyze_data, analyze_error = _run_step_with_timeout(
            "analyze", run_analyze_step,
            raw_indicators=raw_indicators,
            force=False,
        )
        if analyze_error:
            pipeline_errors.append(analyze_error)
        step_results["analyze"] = "ok" if analyze_data else "failed"

    # -------------------------------------------------------------------
    # STEP 3: Generate blog post
    # -------------------------------------------------------------------
    if time.time() - start_time > TOTAL_TIMEOUT:
        logger.error(f"Total pipeline timeout exceeded ({TOTAL_TIMEOUT}s) — skipping blog")
        pipeline_errors.append("Pipeline timed out before blog step")
        step_results["blog"] = "skipped"
    else:
        blog_data, blog_error = _run_step_with_timeout(
            "blog", run_blog_step,
            analyze_data=analyze_data,
            force=False,
        )
        if blog_error:
            pipeline_errors.append(blog_error)
        step_results["blog"] = "ok" if blog_data else "failed"

    # -------------------------------------------------------------------
    # STEP 4: Publish (write files, alerts, git push)
    # -------------------------------------------------------------------
    if time.time() - start_time > TOTAL_TIMEOUT:
        logger.error(f"Total pipeline timeout exceeded ({TOTAL_TIMEOUT}s) — skipping publish")
        pipeline_errors.append("Pipeline timed out before publish step")
        step_results["publish"] = "skipped"
    else:
        publish_result, publish_error = _run_step_with_timeout(
            "publish", run_publish_step,
            analyze_data=analyze_data,
            blog_data=blog_data,
            dry_run=dry_run,
        )
        if publish_error:
            pipeline_errors.append(publish_error)
        step_results["publish"] = "ok" if publish_result else "failed"

    # -------------------------------------------------------------------
    # Compile final results
    # -------------------------------------------------------------------
    # Extract key metrics from whichever step succeeded
    nti_score = 50
    zone = "UNKNOWN"
    confidence = 50
    is_fallback = True
    indicators_scraped = 0
    blog_slug = now_ist.strftime("%Y-%m-%d-%H-%M")

    if raw_indicators:
        indicators_scraped = sum(1 for v in raw_indicators.values() if v is not None)

    if analyze_data:
        nti_result = analyze_data.get("nti_result", {})
        raw_ind = analyze_data.get("raw_indicators", {})
        nti_score = nti_result.get("nti_score", raw_ind.get("nti_score", 50))
        zone = nti_result.get("zone", raw_ind.get("zone", get_zone(nti_score)))
        confidence = nti_result.get("confidence", raw_ind.get("confidence", 50))
        is_fallback = nti_result.get("is_fallback", True)
        indicators_scraped = sum(1 for v in raw_ind.values() if v is not None)

    if blog_data:
        blog_slug = blog_data.get("blog_slug", blog_slug)

    duration = time.time() - start_time

    logger.info(
        f"=== NTI Hourly Pipeline Complete ===\n"
        f"  Score: {nti_score:.1f} ({zone})\n"
        f"  Confidence: {confidence:.0f}%\n"
        f"  Fallback: {is_fallback}\n"
        f"  Indicators: {indicators_scraped} scraped\n"
        f"  Steps: {step_results}\n"
        f"  Duration: {duration:.1f}s\n"
        f"  Errors: {len(pipeline_errors)}"
    )

    if pipeline_errors:
        for err in pipeline_errors:
            logger.error(f"  ERROR: {err}")

    return {
        "nti_score": nti_score,
        "zone": zone,
        "confidence": confidence,
        "is_fallback": is_fallback,
        "indicators_scraped": indicators_scraped,
        "duration_seconds": duration,
        "blog_slug": blog_slug,
        "step_results": step_results,
        "errors": pipeline_errors,
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_hourly_pipeline()
    print(f"\nNTI Score: {result['nti_score']:.1f} ({result['zone']})")
    print(f"Confidence: {result.get('confidence', 0):.0f}%")
    print(f"Indicators: {result.get('indicators_scraped', 0)}")
    print(f"Blog slug: {result.get('blog_slug', 'N/A')}")
    print(f"Steps: {result.get('step_results', {})}")
    print(f"Duration: {result.get('duration_seconds', 0):.1f}s")
    if result.get('errors'):
        print(f"Errors: {result['errors']}")
