"""Git Committer — Git add + commit + push from GitHub Actions.

Commits all data files, blog posts, and API JSON, then pushes to GitHub.
The push triggers Cloudflare Pages to rebuild and deploy the website.
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def git_commit_and_push(
    message: str | None = None,
    paths: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    """Git add, commit, and push data files.

    Args:
        message: Commit message. Auto-generated if not provided.
        paths: List of paths to add. Defaults to data/, website/src/content/blog/, README.md
        dry_run: If True, log what would be committed but don't actually commit/push

    Returns:
        True if commit and push succeeded, False otherwise
    """
    if message is None:
        now_ist = datetime.now(IST)
        message = f"NTI Update {now_ist.strftime('%Y-%m-%d %H:%M')} IST"

    if paths is None:
        paths = [
            "data/",
            "website/src/content/blog/",
            "README.md",
        ]

    # Configure git user (for GitHub Actions)
    _run_git(["config", "--local", "user.email", "nti-bot@github.com"])
    _run_git(["config", "--local", "user.name", "NTI Bot"])

    # Add paths
    for path in paths:
        if Path(path).exists():
            if dry_run:
                logger.info(f"[DRY RUN] Would add: {path}")
            else:
                _run_git(["add", path])

    # Check if there are changes to commit
    result = _run_git(["diff-index", "--quiet", "HEAD"], check=False)
    has_changes = result.returncode != 0

    if not has_changes:
        logger.info("No changes to commit")
        return True

    if dry_run:
        logger.info(f"[DRY RUN] Would commit: {message}")
        return True

    # Commit
    result = _run_git(["commit", "-m", message], check=False)
    if result.returncode != 0:
        logger.warning(f"Git commit failed: {result.stderr}")
        return False

    # Push with retry logic for concurrent pipeline executions
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        # Pull latest changes with rebase to avoid merge commits
        _run_git(["pull", "--rebase", "origin", "main"], check=False)

        result = _run_git(["push"], check=False)
        if result.returncode == 0:
            logger.info(f"Git commit and push succeeded: {message}")
            return True
        
        logger.warning(f"Git push failed (attempt {attempt}/{max_retries}): {result.stderr}")
        if attempt < max_retries:
            time.sleep(2)  # Wait briefly before retrying
            
    logger.error("Git push failed after all retries.")
    return False


def _run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command.

    Args:
        args: Git command arguments (e.g., ["add", "data/"])
        check: If True, raise on non-zero exit code

    Returns:
        CompletedProcess result
    """
    cmd = ["git"] + args
    logger.debug(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if check and result.returncode != 0:
        logger.warning(f"Git command failed: {' '.join(cmd)} — {result.stderr.strip()}")

    return result
