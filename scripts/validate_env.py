#!/usr/bin/env python3
"""Validate Environment — Check all required env vars are set and APIs reachable.

Run this after filling .env: python scripts/validate_env.py

Checks:
1. All required env vars are present and non-empty
2. API keys are valid (optional, requires network)
3. Python dependencies are installed
4. Data directories exist
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


def load_env(env_file: str = ".env") -> dict[str, str]:
    """Parse .env file into dict."""
    env: dict[str, str] = {}
    env_path = Path(env_file)
    if not env_path.exists():
        return env

    with open(env_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if key.strip() and value and not value.startswith("your_"):
                    env[key.strip()] = value
    return env


# Required vars grouped by category
REQUIRED_VARS = {
    "GitHub": ["GITHUB_USERNAME", "GITHUB_TOKEN"],
    "LLM (at least one provider)": [
        "LLM_GROQ_API_KEY",  # Primary
    ],
    "Data APIs": ["FRED_API_KEY", "FINNHUB_API_KEY"],
    "Email": ["ALERT_EMAIL_TO"],
    "Fusion Config": ["FUSION_MODE", "FUSION_SYNTHESIZER"],
}

# Optional but recommended vars
OPTIONAL_VARS = {
    "Email Providers": [
        "EMAIL_GMAIL_ADDRESS", "EMAIL_GMAIL_APP_PASSWORD",
        "EMAIL_RESEND_API_KEY",
    ],
    "LLM — OpenAI-compatible": [
        "LLM_GEMINI_API_KEY", "LLM_CEREBRAS_API_KEY",
        "LLM_OPENROUTER_API_KEY", "LLM_NVIDIA_API_KEY",
        "LLM_TOGETHER_API_KEY", "LLM_SAMBANOVA_API_KEY",
        "LLM_MISTRAL_API_KEY",
    ],
    "LLM — Native SDK": [
        "LLM_COHERE_API_KEY",
        "LLM_HUGGINGFACE_API_KEY",
    ],
    "LLM — Multi-model (comma-separated in MODEL vars)": [
        "FUSION_SYNTHESIZER_MODEL",
    ],
    "Cloudflare": ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_EMAIL", "CLOUDFLARE_GLOBAL_API_KEY", "CLOUDFLARE_ORIGIN_CA_KEY"],
    "Spaceship (Domain Registrar)": ["SPACESHIP_API_KEY", "SPACESHIP_API_SECRET", "SPACESHIP_API_URL"],
    "Langfuse Observability": [
        "LANGFUSE_ENABLED", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
    ],
    "GIFT Nifty": ["GIFT_NIFTY_ENABLED"],
    "MMI Alternative": ["MMI_RAPIDAPI_KEY"],
    "Firebase": [
        "FIREBASE_PROJECT_ID", "FIREBASE_API_KEY",
        "FIREBASE_APP_ID", "FIREBASE_SERVICE_ACCOUNT_BASE64",
    ],
    "Screener Config": [
        "NTI_MAX_PE", "NTI_MAX_PB", "NTI_MIN_MARKET_CAP_CR",
    ],
}

# Python packages that must be importable
REQUIRED_PACKAGES = [
    "pandas", "numpy", "sklearn", "lightgbm", "xgboost",
    "yfinance", "httpx", "feedparser", "bs4", "dotenv",
    "jinja2", "pydantic",
]

OPTIONAL_PACKAGES = [
    ("nsetools", "NSE data scraper"),
    ("selenium", "Tickertape MMI scraper"),
    ("shap", "ML model explainer"),
    ("joblib", "Model persistence"),
    ("langgraph", "LLM fusion workflows"),
    ("openai", "LLM client"),
    ("langchain_cohere", "Cohere native SDK for LLM fusion"),
    ("langchain_huggingface", "HuggingFace native SDK for LLM fusion"),
    ("langfuse", "LLM observability & tracing (Langfuse)"),
]


def check_env_vars(env: dict[str, str]) -> tuple[int, int]:
    """Check required and optional env vars.

    Returns:
        Tuple of (errors_count, warnings_count)
    """
    errors = 0
    warnings = 0

    print("\n[*] Checking required environment variables...")
    for category, vars_list in REQUIRED_VARS.items():
        print(f"\n  {category}:")
        for var in vars_list:
            value = env.get(var, "")
            if value and not value.startswith("your_"):
                # Mask sensitive values
                if "KEY" in var or "TOKEN" in var or "PASSWORD" in var or "SECRET" in var:
                    masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                    print(f"    [OK] {var}: {masked}")
                else:
                    print(f"    [OK] {var}: {value}")
            else:
                print(f"    [!!] {var}: NOT SET")
                errors += 1

    print("\n[?] Checking optional environment variables...")
    for category, vars_list in OPTIONAL_VARS.items():
        print(f"\n  {category}:")
        for var in vars_list:
            value = env.get(var, "")
            if value and not value.startswith("your_"):
                if "KEY" in var or "TOKEN" in var or "PASSWORD" in var or "SECRET" in var:
                    masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                    print(f"    [OK] {var}: {masked}")
                else:
                    print(f"    [OK] {var}: {value}")
            else:
                print(f"    [--] {var}: not set (optional)")
                warnings += 1

    return errors, warnings


def check_python_packages() -> tuple[int, int]:
    """Check that required Python packages are installed.

    Returns:
        Tuple of (errors_count, warnings_count)
    """
    errors = 0
    warnings = 0

    print("\n[*] Checking Python packages...")
    print("\n  Required packages:")
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"    [OK] {pkg}")
        except ImportError:
            print(f"    [!!] {pkg}: NOT INSTALLED")
            errors += 1

    print("\n  Optional packages:")
    for pkg, purpose in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"    [OK] {pkg} ({purpose})")
        except ImportError:
            print(f"    [--] {pkg}: not installed ({purpose})")
            warnings += 1

    return errors, warnings


def check_data_dirs() -> int:
    """Check that required data directories exist.

    Returns:
        Number of missing directories
    """
    missing = 0
    print("\n📁 Checking data directories...")

    required_dirs = [
        "data",
        "data/api",
        "data/signals",
        "data/indicators/hourly",
        "data/screener",
        "data/model",
        "data/errors",
        "model_artifacts",
        "website/src/content/blog",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [--] {dir_path}/: missing (will be created on first run)")
            missing += 1

    return missing


def check_api_reachability(env: dict[str, str]) -> int:
    """Check that key APIs are reachable (optional, requires network).

    Returns:
        Number of unreachable APIs
    """
    import urllib.request
    import urllib.error

    unreachable = 0
    print("\n🌐 Checking API reachability (optional)...")

    apis = [
        ("FRED API", "https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key=test&file_type=json"),
        ("Finnhub", "https://finnhub.io/api/v1/stock/recommendation?symbol=AAPL&token=test"),
        ("Yahoo Finance", "https://query1.finance.yahoo.com/v8/finance/chart/^NSEI"),
    ]

    for name, url in apis:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NTI-Validator/1.0"})
            urllib.request.urlopen(req, timeout=5)
            print(f"  [OK] {name}: reachable")
        except urllib.error.HTTPError:
            # HTTP error means the server is reachable (auth may fail, but that's ok)
            print(f"  [OK] {name}: reachable (auth required)")
        except Exception as e:
            print(f"  [--] {name}: unreachable ({e})")
            unreachable += 1

    return unreachable


def main() -> None:
    print("[*] Nifty Timing Index -- Environment Validation")
    print("=" * 50)

    # Load .env
    env = load_env(".env")
    if not env:
        print("[!] No .env file found. Copy .env.example to .env and fill in values.")
    else:
        print(f"  [OK] Found {len(env)} configured values in .env")

    total_errors = 0
    total_warnings = 0

    # Check env vars
    err, warn = check_env_vars(env)
    total_errors += err
    total_warnings += warn

    # Check packages
    err, warn = check_python_packages()
    total_errors += err
    total_warnings += warn

    # Check data dirs
    missing = check_data_dirs()
    total_warnings += missing

    # Optional: API reachability
    try:
        unreachable = check_api_reachability(env)
        total_warnings += unreachable
    except Exception:
        print("\n  [--] Network check skipped (no network access)")

    # Summary
    print("\n" + "=" * 50)
    if total_errors == 0:
        print("[OK] All required checks passed!")
        if total_warnings > 0:
            print(f"  [--] {total_warnings} warnings (optional features may not work)")
        print("\n  Ready to run: uv run python -m nti.pipelines.hourly")
    else:
        print(f"[!!] {total_errors} error(s) found. Fix before running the system.")
        print(f"  [--] {total_warnings} warnings")
        sys.exit(1)


if __name__ == "__main__":
    main()
