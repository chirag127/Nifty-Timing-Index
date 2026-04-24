#!/usr/bin/env python3
"""Run hourly pipeline in sequential fusion mode for faster blog generation.

Sets FUSION_MODE=sequential before importing any NTI modules,
so the blog generation uses sequential LLM calls instead of parallel fan-out.
"""
import os
import sys
import logging

# Force sequential mode BEFORE importing nti modules
os.environ["FUSION_MODE"] = "sequential"
os.environ["NTI_FUSION_MAX_CONCURRENT_LLM"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from nti.pipelines.hourly import run_hourly_pipeline

result = run_hourly_pipeline()
print(f"\n{'='*60}")
print(f"NTI Score: {result['nti_score']:.1f} ({result['zone']})")
print(f"Confidence: {result.get('confidence', 0):.0f}%")
print(f"Indicators: {result.get('indicators_scraped', 0)}")
print(f"Blog slug: {result.get('blog_slug', 'N/A')}")
print(f"Duration: {result.get('duration_seconds', 0):.1f}s")
print(f"Errors: {result.get('errors', [])}")
print(f"{'='*60}")
