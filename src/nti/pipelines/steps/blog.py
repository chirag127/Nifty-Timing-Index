"""Step 3: Blog — Generate blog post via LLM fusion workflow.

Loads analysis data from step_analyze.json, generates the blog post,
validates it has proper LLM text content, and saves intermediate state.

If blog generation fails or produces empty/invalid content, the step
gracefully returns an empty blog_markdown so the publish step can
skip writing a broken blog file.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.settings import settings
from nti.config.thresholds import get_zone
from nti.llm.blog_generator import generate_hourly_blog

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path("data")
API_DIR = DATA_DIR / "api"
STEP_FILE = API_DIR / "step_blog.json"
ANALYZE_FILE = API_DIR / "step_analyze.json"

# Minimum content length for a valid blog post (characters)
MIN_BLOG_LENGTH = 200

# Minimum number of words that look like real commentary (not just frontmatter/tables)
MIN_COMMENTARY_WORDS = 50


def _load_analyze_data() -> dict | None:
    """Load analysis data from the intermediate file.

    Returns None (instead of raising) if the file doesn't exist,
    so the blog step can gracefully skip when analyze data is unavailable.
    """
    if not ANALYZE_FILE.exists():
        logger.warning(f"Analysis data not found at {ANALYZE_FILE} — blog step will produce no content")
        return None
    try:
        with open(ANALYZE_FILE) as f:
            data = json.load(f)
        # Remove internal metadata keys
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load analyze data: {e}")
        return None


def _get_blog_type(now_ist: datetime) -> str:
    """Determine blog type based on time of day."""
    hour = now_ist.hour
    minute = now_ist.minute

    if hour == 9 and minute < 30:
        return "market_open"
    if hour == 15 and minute >= 15:
        return "market_close"
    if 9 <= hour <= 15:
        return "mid_session"
    if 16 <= hour <= 20:
        return "post_market"
    return "overnight"


def _validate_blog_content(blog_markdown: str) -> tuple[bool, str]:
    """Validate that a blog post has proper LLM-generated text commentary.

    Checks:
    1. Total content length >= MIN_BLOG_LENGTH characters
    2. Contains actual prose (not just tables/frontmatter/data)
    3. Has at least MIN_COMMENTARY_WORDS of commentary text
    4. Contains a disclaimer

    Returns:
        Tuple of (is_valid, reason)
    """
    if not blog_markdown or not blog_markdown.strip():
        return False, "Blog markdown is empty"

    # Strip frontmatter if present (between --- markers)
    content = blog_markdown.strip()
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()

    # Check total length
    if len(content) < MIN_BLOG_LENGTH:
        return False, f"Blog content too short ({len(content)} chars, need {MIN_BLOG_LENGTH})"

    # Strip markdown tables and structural elements to get pure commentary
    commentary = content
    # Remove table rows (but keep any text inside them)
    commentary = re.sub(r'\|', ' ', commentary)
    # Remove markdown headers (but keep their text)
    commentary = re.sub(r'^#+\s*', '', commentary, flags=re.MULTILINE)
    # Remove bullet markers (but keep text)
    commentary = re.sub(r'^[-*]\s*', '', commentary, flags=re.MULTILINE)
    # Remove links but keep text
    commentary = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', commentary)
    # Remove bold/italic markers
    commentary = re.sub(r'[*_]+', '', commentary)
    # Collapse whitespace
    commentary = re.sub(r'\s+', ' ', commentary).strip()

    # Count words in the commentary (including numbers — they are valid LLM output)
    commentary_words = len(commentary.split())
    if commentary_words < MIN_COMMENTARY_WORDS:
        return False, f"Blog has too little commentary ({commentary_words} words, need {MIN_COMMENTARY_WORDS})"

    return True, "Valid"


def run_blog_step(analyze_data: dict | None = None, force: bool = False) -> dict:
    """Run the blog step: generate blog post via LLM.

    Args:
        analyze_data: Dict from the analyze step. If None, loads from step_analyze.json
        force: If True, re-run even if step_blog.json already exists

    Returns:
        Dict with keys: blog_markdown, blog_slug, blog_type, blog_valid, blog_validation_reason,
                         providers_used, errors, duration_seconds
    """
    now_ist = datetime.now(IST)
    logger.info(f"=== Step 3: Blog (started {now_ist.isoformat()}) ===")

    # Load analyze data if not provided
    if analyze_data is None:
        analyze_data = _load_analyze_data()

    # If no analyze data available, return empty result
    if analyze_data is None:
        logger.warning("No analyze data available — cannot generate blog")
        return {
            "blog_markdown": "",
            "blog_slug": now_ist.strftime("%Y-%m-%d-%H-%M"),
            "blog_type": _get_blog_type(now_ist),
            "blog_valid": False,
            "blog_validation_reason": "No analyze data available",
            "providers_used": [],
            "errors": ["No analyze data available — run scrape and analyze steps first"],
            "duration_seconds": 0,
        }

    # Check for existing step output (resume support)
    if not force and STEP_FILE.exists():
        try:
            with open(STEP_FILE) as f:
                cached = json.load(f)
            cached_time = cached.get("_blog_at", "")
            if cached_time:
                cached_dt = datetime.fromisoformat(cached_time)
                age_minutes = (now_ist - cached_dt).total_seconds() / 60
                if age_minutes < 55:
                    logger.info(f"Reusing blog data from {age_minutes:.0f} min ago")
                    cached.pop("_blog_at", None)
                    return cached
        except (json.JSONDecodeError, ValueError, OSError):
            pass

    start_time = time.time()

    # Extract data from analyze step
    raw_indicators = analyze_data.get("raw_indicators", {})
    nti_result = analyze_data.get("nti_result", {})
    changelog_text = analyze_data.get("changelog_text", "")

    nti_score = nti_result.get("nti_score", 50)
    prev_score = raw_indicators.get("nti_score_prev") or nti_score
    confidence = nti_result.get("confidence", 50)
    top_drivers = nti_result.get("top_drivers", [])
    blog_type = _get_blog_type(now_ist)
    blog_slug = now_ist.strftime("%Y-%m-%d-%H-%M")

    # -------------------------------------------------------------------
    # Generate blog post via LLM
    # -------------------------------------------------------------------
    blog_markdown = ""
    providers_used = []
    errors = []

    try:
        blog_result = generate_hourly_blog(
            nti_score=nti_score,
            prev_score=prev_score,
            confidence=confidence,
            indicators=raw_indicators,
            top_drivers=top_drivers,
            top_stocks=[],
            changelog_text=changelog_text,
            blog_type=blog_type,
        )

        blog_markdown = blog_result.get("blog_markdown", "")
        providers_used = blog_result.get("providers_used", [])
        errors = blog_result.get("errors", [])

    except Exception as e:
        error_msg = f"Blog generation failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # -------------------------------------------------------------------
    # Validate blog content
    # -------------------------------------------------------------------
    blog_valid = False
    blog_validation_reason = ""

    if blog_markdown:
        blog_valid, blog_validation_reason = _validate_blog_content(blog_markdown)
        if not blog_valid:
            logger.warning(f"Blog validation FAILED: {blog_validation_reason}")
            logger.warning("Discarding invalid blog content — will not publish empty/invalid blog")
            blog_markdown = ""
        else:
            logger.info(f"Blog validation PASSED ({len(blog_markdown)} chars)")
    else:
        blog_valid = False
        blog_validation_reason = "Blog generation returned empty content"
        logger.warning(blog_validation_reason)

    # -------------------------------------------------------------------
    # Save intermediate state
    # -------------------------------------------------------------------
    result = {
        "blog_markdown": blog_markdown,
        "blog_slug": blog_slug,
        "blog_type": blog_type,
        "blog_valid": blog_valid,
        "blog_validation_reason": blog_validation_reason,
        "providers_used": providers_used,
        "errors": errors,
        "duration_seconds": time.time() - start_time,
        "_blog_at": now_ist.isoformat(),
    }

    API_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Blog step saved to {STEP_FILE}")
    except OSError as e:
        logger.error(f"Failed to save blog step: {e}")

    duration = time.time() - start_time
    status = "VALID" if blog_valid else "INVALID/EMPTY"
    logger.info(f"=== Step 3: Blog complete ({duration:.1f}s, {status}) ===")

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="NTI Blog Step")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if step file exists")
    args = parser.parse_args()

    result = run_blog_step(force=args.force)
    print(f"\nBlog valid: {result.get('blog_valid', False)}")
    print(f"Blog type: {result.get('blog_type', 'N/A')}")
    print(f"Slug: {result.get('blog_slug', 'N/A')}")
    print(f"Validation: {result.get('blog_validation_reason', 'N/A')}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")
