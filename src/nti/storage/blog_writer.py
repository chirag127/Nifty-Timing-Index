"""Blog Writer — Write blog .md files to the Astro website content directory.

Each blog is a Markdown file with YAML frontmatter that Astro's content
collections automatically pick up and render as static pages.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nti.config.thresholds import get_zone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _strip_existing_frontmatter(markdown: str) -> str:
    """Strip any existing YAML frontmatter from markdown content.

    The LLM sometimes generates its own frontmatter (--- ... ---), which
    causes a double-frontmatter bug when blog_writer adds its own.
    This function strips any existing frontmatter before we add ours.

    Args:
        markdown: Raw markdown content that may contain frontmatter

    Returns:
        Markdown content with frontmatter stripped
    """
    if not markdown or not markdown.strip():
        return markdown

    content = markdown.strip()
    if content.startswith("---"):
        # Find the closing --- of the frontmatter on its own line
        # Frontmatter is: ---\n<yaml>\n---\n<content>
        # We look for the second --- that appears at the start of a line
        lines = content.split("\n")
        if len(lines) >= 2:
            # Skip the first line (opening ---)
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    # Found closing --- — everything after it is the real content
                    remaining = "\n".join(lines[i + 1:]).strip()
                    logger.info(f"Stripped existing frontmatter from blog content ({len(markdown)} → {len(remaining)} chars)")
                    return remaining
            # No closing --- found — the opening --- might be a horizontal rule, leave as-is

    return markdown


def write_blog_post(
    blog_markdown: str,
    nti_score: float,
    prev_score: float | None,
    confidence: float,
    nifty_price: float | None,
    top_drivers: list[str],
    top_stocks: list[str],
    blog_type: str = "mid_session",
    website_dir: Path | None = None,
) -> Path:
    """Write a blog post .md file to the Astro content directory.

    Args:
        blog_markdown: The full blog post markdown content (from LLM)
        nti_score: Current NTI danger score
        prev_score: Previous hour's NTI score
        confidence: Model confidence percentage
        nifty_price: Current Nifty 50 price
        top_drivers: List of top SHAP driver indicator names
        top_stocks: List of top stock pick symbols
        blog_type: "market_open", "mid_session", "market_close", "post_market", "overnight"
        website_dir: Path to the website directory

    Returns:
        Path to the blog .md file written
    """
    if website_dir is None:
        website_dir = Path("website")

    # Strip any existing frontmatter from blog_markdown to prevent
    # double-frontmatter bug (LLM sometimes generates its own frontmatter)
    blog_markdown = _strip_existing_frontmatter(blog_markdown)

    now_ist = datetime.now(IST)
    slug = now_ist.strftime("%Y-%m-%d-%H-%M")
    date_iso = now_ist.isoformat()

    zone = get_zone(nti_score)
    prev_zone = get_zone(prev_score) if prev_score is not None else "UNKNOWN"
    zone_changed = zone != prev_zone and prev_zone != "UNKNOWN"

    # Build YAML frontmatter
    title = f"NTI Update: Score {nti_score:.0f} | {zone.replace('_', ' ')} Zone"
    if nifty_price:
        title += f" | Nifty at {nifty_price:,.0f}"
    title += f" | {now_ist.strftime('%Y-%m-%d %H:%M')} IST"

    description = (
        f"Nifty Timing Index hourly update: NTI score at {nti_score:.0f} "
        f"({zone.replace('_', ' ')} zone). "
        f"Confidence: {confidence:.0f}%."
    )

    # Format top drivers and stocks as YAML lists
    drivers_yaml = "\n".join(f'  - "{d}"' for d in top_drivers[:5])
    stocks_yaml = "\n".join(f'  - "{s}"' for s in top_stocks[:5])

    frontmatter = f"""---
title: "{title}"
description: "{description}"
slug: "{slug}"
publishedAt: "{date_iso}"
ntiScore: {nti_score:.1f}
ntiZone: "{zone}"
ntiZonePrev: "{prev_zone}"
zoneChanged: {str(zone_changed).lower()}
confidence: {confidence:.0f}
nifty50Price: {nifty_price or 0}
topDrivers:
{drivers_yaml if drivers_yaml else '  []'}
topStocks:
{stocks_yaml if stocks_yaml else '  []'}
blogType: "{blog_type}"
---

"""

    # Full file content
    content = frontmatter + blog_markdown

    # Write to website content directory
    blog_dir = website_dir / "src" / "content" / "blog"
    blog_dir.mkdir(parents=True, exist_ok=True)
    blog_path = blog_dir / f"{slug}.md"

    try:
        with open(blog_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Blog post written to {blog_path}")
    except OSError as e:
        logger.error(f"Failed to write blog post: {e}")

    return blog_path
