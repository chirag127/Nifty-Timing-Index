"""Step 4: Publish — Write data files, send alerts, git commit & push.

Loads data from previous step intermediate files, writes all output
files (CSV, JSON, blog .md), sends email alerts on zone changes,
and git commits + pushes everything.

This step is designed to be fast (< 30s) since all heavy computation
is done in previous steps.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.settings import settings
from nti.config.thresholds import get_zone

# Storage
from nti.storage.csv_writer import write_hourly_csv
from nti.storage.json_api import write_latest_json, write_history_json
from nti.storage.blog_writer import write_blog_post
from nti.storage.git_committer import git_commit_and_push

# Changelog
from nti.changelog.generator import save_current_run

# Notifications
from nti.notifications.email_sender import send_zone_change_alert, send_big_move_alert

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")
API_DIR = DATA_DIR / "api"
ANALYZE_FILE = API_DIR / "step_analyze.json"
BLOG_FILE = API_DIR / "step_blog.json"


def _load_step_data() -> tuple[dict, dict]:
    """Load analysis and blog data from intermediate files."""
    # Load analyze data
    if not ANALYZE_FILE.exists():
        raise FileNotFoundError(
            f"Analysis data not found at {ANALYZE_FILE}. "
            "Run the analyze step first."
        )
    with open(ANALYZE_FILE) as f:
        analyze_data = json.load(f)
    analyze_data = {k: v for k, v in analyze_data.items() if not k.startswith("_")}

    # Load blog data
    blog_data = {}
    if BLOG_FILE.exists():
        with open(BLOG_FILE) as f:
            blog_data = json.load(f)
        blog_data = {k: v for k, v in blog_data.items() if not k.startswith("_")}

    return analyze_data, blog_data


def run_publish_step(
    analyze_data: dict | None = None,
    blog_data: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the publish step: write files, send alerts, git push.

    Args:
        analyze_data: Dict from the analyze step. If None, loads from step_analyze.json
        blog_data: Dict from the blog step. If None, loads from step_blog.json
        dry_run: If True, don't write files or push to git

    Returns:
        Dict with keys: files_written, alerts_sent, git_pushed, errors
    """
    now_ist = datetime.now(IST)
    logger.info(f"=== Step 4: Publish (started {now_ist.isoformat()}) ===")

    # Load data if not provided
    if analyze_data is None or blog_data is None:
        _analyze_data, _blog_data = _load_step_data()
        if analyze_data is None:
            analyze_data = _analyze_data
        if blog_data is None:
            blog_data = _blog_data

    start_time = time.time()
    pipeline_errors: list[str] = []
    files_written: list[str] = []
    alerts_sent: list[str] = []

    # Extract data
    raw_indicators = analyze_data.get("raw_indicators", {})
    nti_result = analyze_data.get("nti_result", {})
    changelog_text = analyze_data.get("changelog_text", "")

    nti_score = nti_result.get("nti_score", 50)
    zone = nti_result.get("zone", get_zone(nti_score))
    confidence = nti_result.get("confidence", 50)
    top_drivers = nti_result.get("top_drivers", [])

    blog_markdown = blog_data.get("blog_markdown", "")
    blog_slug = blog_data.get("blog_slug", now_ist.strftime("%Y-%m-%d-%H-%M"))
    blog_valid = blog_data.get("blog_valid", False)
    blog_type = blog_data.get("blog_type", "mid_session")

    # Load previous run for comparison
    from nti.changelog.generator import load_previous_run
    previous_run = load_previous_run()
    prev_score = previous_run.get("nti_score")
    prev_zone = previous_run.get("zone", "UNKNOWN")

    if not dry_run:
        # -------------------------------------------------------------------
        # Write hourly CSV
        # -------------------------------------------------------------------
        try:
            csv_path = write_hourly_csv(raw_indicators)
            files_written.append(str(csv_path))
            logger.info(f"Hourly CSV written to {csv_path}")
        except Exception as e:
            pipeline_errors.append(f"CSV write failed: {e}")
            logger.error(f"Failed to write hourly CSV: {e}")

        # -------------------------------------------------------------------
        # Write API JSON files
        # -------------------------------------------------------------------
        try:
            driver_dicts = []
            for d in top_drivers[:5]:
                driver_dicts.append({
                    "indicator": d.get("indicator", d.get("feature", "")),
                    "label": d.get("label", d.get("indicator", "")),
                    "shap": d.get("shap", 0),
                    "direction": d.get("direction", ""),
                    "current_value": d.get("description", ""),
                })

            latest_path = write_latest_json(
                indicators=raw_indicators,
                nti_result=nti_result,
                top_drivers=driver_dicts,
                top_stocks=[],
                blog_slug=blog_slug,
            )
            files_written.append(str(latest_path))

            history_path = write_history_json()
            files_written.append(str(history_path))

            logger.info("API JSON files written")
        except Exception as e:
            pipeline_errors.append(f"JSON write failed: {e}")
            logger.error(f"Failed to write API JSON: {e}")

        # -------------------------------------------------------------------
        # Write blog post (only if valid)
        # -------------------------------------------------------------------
        if blog_markdown and blog_valid:
            try:
                blog_path = write_blog_post(
                    blog_markdown=blog_markdown,
                    nti_score=nti_score,
                    prev_score=prev_score,
                    confidence=confidence,
                    nifty_price=raw_indicators.get("nifty_price"),
                    top_drivers=[d.get("indicator", d.get("feature", "")) for d in top_drivers[:5]],
                    top_stocks=[],
                    blog_type=blog_type,
                )
                files_written.append(str(blog_path))
                logger.info(f"Blog post written to {blog_path}")
            except Exception as e:
                pipeline_errors.append(f"Blog write failed: {e}")
                logger.error(f"Failed to write blog post: {e}")
        else:
            reason = blog_data.get("blog_validation_reason", "No blog content")
            logger.warning(f"Skipping blog write: {reason}")

        # -------------------------------------------------------------------
        # Save current run for next comparison
        # -------------------------------------------------------------------
        try:
            save_current_run(raw_indicators)
            logger.info("Saved current run data for next comparison")
        except Exception as e:
            pipeline_errors.append(f"Save current run failed: {e}")

        # -------------------------------------------------------------------
        # Send alerts on zone change
        # -------------------------------------------------------------------
        if settings.enable_email:
            try:
                # Zone change alert
                if zone != prev_zone and prev_zone != "UNKNOWN" and settings.alert_on_zone_change:
                    driver_names = [d.get("label", d.get("indicator", "")) for d in top_drivers[:3]]
                    send_zone_change_alert(
                        from_zone=prev_zone,
                        to_zone=zone,
                        score=nti_score,
                        nifty_price=raw_indicators.get("nifty_price", 0),
                        drivers=driver_names,
                    )
                    alerts_sent.append(f"Zone change: {prev_zone} → {zone}")
                    logger.info(f"Zone change alert sent: {prev_zone} → {zone}")

                # Big move alert
                if prev_score is not None and settings.alert_on_big_move:
                    move = abs(nti_score - prev_score)
                    if move >= settings.alert_big_move_threshold:
                        send_big_move_alert(prev_score, nti_score, raw_indicators.get("nifty_price", 0))
                        alerts_sent.append(f"Big move: {move:.1f} points")
                        logger.info(f"Big move alert sent: {move:.1f} points")
            except Exception as e:
                pipeline_errors.append(f"Email alert failed: {e}")

        # -------------------------------------------------------------------
        # Git commit and push
        # -------------------------------------------------------------------
        try:
            git_commit_and_push()
            logger.info("Git commit and push completed")
        except Exception as e:
            pipeline_errors.append(f"Git push failed: {e}")

        # -------------------------------------------------------------------
        # Clean up intermediate step files (keep for debugging)
        # -------------------------------------------------------------------
        # Don't delete step files — they're useful for resume and debugging

    else:
        logger.info("[DRY RUN] Skipping all file writes and git push")

    duration = time.time() - start_time
    logger.info(
        f"=== Step 4: Publish complete ({duration:.1f}s) ===\n"
        f"  Files written: {len(files_written)}\n"
        f"  Alerts sent: {len(alerts_sent)}\n"
        f"  Errors: {len(pipeline_errors)}"
    )

    return {
        "files_written": files_written,
        "alerts_sent": alerts_sent,
        "git_pushed": not dry_run,
        "errors": pipeline_errors,
        "duration_seconds": duration,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_publish_step()
    print(f"\nFiles written: {len(result['files_written'])}")
    print(f"Alerts: {result['alerts_sent']}")
    print(f"Errors: {result['errors']}")
