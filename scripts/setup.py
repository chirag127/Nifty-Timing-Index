#!/usr/bin/env python3
"""NTI Setup Script — One command to rule them all.

Run this ONCE after filling in .env to set up everything:
  - Validate all required env vars
  - Push all secrets to GitHub via gh CLI
  - Create Cloudflare Pages project
  - Initialize Firebase (Auth + Firestore)
  - Bootstrap initial historical data

Usage: python scripts/setup.py
Requirements: gh CLI, wrangler CLI (optional), firebase CLI (optional)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def load_env(env_file: str = ".env") -> dict[str, str]:
    """Parse .env file into dict, ignoring comments and blank lines."""
    env: dict[str, str] = {}
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"[!!] .env file not found at {env_path.resolve()}")
        print("  Copy .env.example to .env and fill in your values first.")
        sys.exit(1)

    with open(env_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if key.strip() and value:
                    env[key.strip()] = value

    return env


def run_cmd(
    cmd: str,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run shell command and handle errors."""
    print(f"  > {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"  [!!] Command failed: {cmd}")
        if result.stderr:
            print(f"  Error: {result.stderr.strip()}")
        sys.exit(1)
    return result


def check_cli_tool(name: str, install_hint: str) -> bool:
    """Check if a CLI tool is installed."""
    result = run_cmd(f"which {name}", check=False, capture=True)
    if result.returncode != 0:
        print(f"  [--] {name} not found. Install: {install_hint}")
        return False
    print(f"  [OK] {name} found")
    return True


def validate_required_vars(env: dict[str, str]) -> None:
    """Check all required env vars are present."""
    required = [
        "GITHUB_USERNAME",
        "GITHUB_TOKEN",
        "LLM_GROQ_API_KEY",
        "FRED_API_KEY",
        "FINNHUB_API_KEY",
        "ALERT_EMAIL_TO",
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        print(f"\n[!!] Missing required env vars: {', '.join(missing)}")
        print("  Please fill in .env before running setup.")
        sys.exit(1)
    print(f"  [OK] All {len(required)} required env vars found")


def push_secrets_to_github(env: dict[str, str], repo: str) -> None:
    """Push all env vars as GitHub Secrets using gh CLI."""
    print("\n>> Pushing secrets to GitHub...")

    # Write a temp secrets file (only non-empty values, skip sensitive paths)
    secrets_file = Path(".env.secrets.tmp")
    skip_keys = {"GITHUB_TOKEN", "GITHUB_USERNAME", "GITHUB_REPO_NAME"}
    count = 0
    with open(secrets_file, "w") as f:
        for key, value in env.items():
            if value and not key.startswith("_") and key not in skip_keys:
                # Create PUBLIC_ prefixed versions for Astro (client-safe keys only)
                public_safe = {
                    "FIREBASE_API_KEY", "FIREBASE_AUTH_DOMAIN", "FIREBASE_PROJECT_ID",
                    "FIREBASE_STORAGE_BUCKET", "FIREBASE_MESSAGING_SENDER_ID",
                    "FIREBASE_APP_ID", "FIREBASE_MEASUREMENT_ID",
                }
                if key in public_safe:
                    f.write(f"PUBLIC_{key}={value}\n")
                f.write(f"{key}={value}\n")
                count += 1

    # Use gh CLI bulk secret set
    run_cmd(f"gh secret set --env-file {secrets_file} --repo {repo}")
    secrets_file.unlink()  # Delete temp file immediately
    print(f"  [OK] {count} secrets pushed to GitHub")


def create_github_repo(env: dict[str, str]) -> str:
    """Create public GitHub repository if it doesn't exist."""
    repo_name = env.get("GITHUB_REPO_NAME", "Nifty-Timing-Index")
    username = env.get("GITHUB_USERNAME", "chirag127")
    repo = f"{username}/{repo_name}"

    print(f"\n>> Setting up GitHub repository: {repo}")

    # Check if repo already exists
    result = run_cmd(f"gh repo view {repo}", check=False, capture=True)
    if result.returncode == 0:
        print(f"  [OK] Repository already exists: https://github.com/{repo}")
        return repo

    # Create the repo
    run_cmd(
        f'gh repo create {repo} --public '
        f'--description "Nifty Timing Index — Automated Indian market analysis with hourly blogs" '
        f'--clone=false'
    )
    print(f"  [OK] Repository created: https://github.com/{repo}")
    return repo


def setup_cloudflare_pages(env: dict[str, str]) -> None:
    """Create Cloudflare Pages project via wrangler CLI."""
    project_name = "nifty-timing-index"
    print(f"\n>> Setting up Cloudflare Pages: {project_name}")

    # Check if wrangler is installed
    result = run_cmd("which wrangler", check=False, capture=True)
    if result.returncode != 0:
        print("  [--] wrangler CLI not found. Install with: npm install -g wrangler")
        print("  You can also set up Cloudflare Pages manually from the dashboard.")
        print("  Skipping automatic Cloudflare setup.")
        return

    # Try to create the Pages project
    run_cmd(
        f"wrangler pages project create {project_name} --production-branch main",
        check=False,
    )
    print(f"  [OK] Cloudflare Pages project ready")


def setup_firebase(env: dict[str, str]) -> None:
    """Initialize Firebase project with Auth and Firestore."""
    project_id = env.get("FIREBASE_PROJECT_ID", "nifty-timing-index")
    print(f"\n>> Setting up Firebase project: {project_id}")

    # Check if firebase CLI is installed
    result = run_cmd("which firebase", check=False, capture=True)
    if result.returncode != 0:
        print("  [--] Firebase CLI not found. Install with: npm install -g firebase-tools")
        print("  You can set up Firebase manually from the console.")
        print("  Skipping automatic Firebase setup.")
        return

    # Create Firebase project
    run_cmd(
        f"firebase projects:create {project_id} "
        f'--display-name "Nifty Timing Index"',
        check=False,
    )

    # Enable Google Auth (requires manual step)
    print("  [*] Enable Google Auth manually:")
    print(f"     https://console.firebase.google.com/project/{project_id}/authentication/providers")

    # Create Firestore database in asia-south1 (Mumbai)
    run_cmd(
        f"firebase firestore:databases:create --project {project_id} --location asia-south1",
        check=False,
    )
    print(f"  [OK] Firebase project ready: https://console.firebase.google.com/project/{project_id}")


def bootstrap_data() -> None:
    """Run the initial data bootstrap script."""
    print("\n>> Bootstrapping historical data...")
    bootstrap_script = Path("scripts/bootstrap_data.py")
    if bootstrap_script.exists():
        run_cmd("uv run python scripts/bootstrap_data.py", check=False)
    else:
        print("  [--] Bootstrap script not found. Run manually later.")


def main() -> None:
    print("[NTI] Nifty Timing Index -- Setup Script")
    print("=" * 50)

    # Step 0: Check prerequisites
    print("\n[*] Checking prerequisites...")
    check_cli_tool("gh", "https://cli.github.com")
    check_cli_tool("uv", "https://docs.astral.sh/uv/getting-started/installation/")
    check_cli_tool("git", "https://git-scm.com")

    # Load .env
    print("\n>> Loading configuration...")
    env = load_env(".env")
    print(f"  [OK] Loaded {len(env)} env vars from .env")

    # Validate required vars
    validate_required_vars(env)

    # Step 1: GitHub repo
    repo = create_github_repo(env)

    # Step 2: Push secrets
    push_secrets_to_github(env, repo)

    # Step 3: Cloudflare Pages
    setup_cloudflare_pages(env)

    # Step 4: Firebase
    setup_firebase(env)

    # Step 5: Install Python dependencies
    print("\n[*] Installing Python dependencies...")
    run_cmd("uv sync")

    # Step 6: Bootstrap data
    bootstrap_data()

    # Done
    print("\n" + "=" * 50)
    print("[OK] Setup complete! Here's what happens next:")
    print("  1. Push your code to GitHub (git push origin main)")
    print("  2. GitHub Actions will run the first hourly signal automatically")
    print("  3. Cloudflare Pages will deploy the website on push")
    print("  4. First blog post will appear in ~10 minutes after Actions run")
    print()
    print(f"  >> GitHub: https://github.com/{repo}")
    print(f"  >> Actions: https://github.com/{repo}/actions")
    print("  >> Cloudflare: https://dash.cloudflare.com")
    print("  >> Firebase: https://console.firebase.google.com")
    print()
    print("  [*] Run 'uv run python -m nti.pipelines.hourly' for a manual test run.")


if __name__ == "__main__":
    main()
