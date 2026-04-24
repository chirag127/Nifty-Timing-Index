"""Pipeline Steps — Modular sub-steps for the hourly NTI pipeline.

Each step is independently executable, saves intermediate state to disk,
and can be resumed if a previous step already completed.

Steps:
1. scrape  — Scrape all 30+ indicators from various sources
2. analyze — Normalize indicators, run inference, analyze news
3. blog    — Generate blog post via LLM fusion workflow
4. publish — Write data files (CSV, JSON, blog .md), git commit & push
"""
