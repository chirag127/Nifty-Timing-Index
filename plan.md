# Nifty Timing Index (NTI) — Complete Master Implementation Plan

> **Version:** 5.0 — Final, Zero Assumptions, All Confirmed
> **Date:** 2026-04-23
> **Author:** Chirag Singhal · chirag127 · whyiswhen@gmail.com
> **Repository:** `github.com/chirag127/nifty-timing-index` (PUBLIC)
> **Stack:** Python (uv) · Astro 6 · Cloudflare Pages · Firebase · GitHub Actions
> **Philosophy:** Long-only · Value stocks · Low PE/PB · PSU-preferred · MTF-aware · 100% Free

---

## ✅ All Confirmed Choices (Zero Assumptions)

| Question | Confirmed Answer |
|----------|----------------|
| Investment type | **Buy stocks and hold (long only, no shorting ever, no index futures short)** |
| Market cap minimum | **₹500 Cr and above** |
| Indices tracked/scored | **Nifty 50 + Nifty Bank + Sensex/BSE 30 + Nifty Midcap 150 + Nifty Smallcap 250** |
| Stock universe for screener | **Any stock on NSE or BSE with ≥ ₹500 Cr market cap** |
| PSU preference | **Soft preference — PSU/govt stocks ranked HIGHER but quality private also shown** |
| PE limit | **PE < 20 strictly (hard limit, no exceptions, no growth adjustments)** |
| PB limit | **PB < 3 strictly** |
| LLM provider | **Fully configurable via .env — change base URL + API key + model name anytime** |
| Email | **Gmail SMTP (free, 500 emails/day)** |
| Website hosting | **Cloudflare Pages (free, unlimited bandwidth, global CDN)** |
| Auth | **Google Login for saving preferences; site is fully public without login** |
| User data saved | **All: watchlist, alert preferences, custom PE/PB thresholds, dark/light mode** |
| Blog frequency | **Every hour during market + every 2 hrs overnight (~20 blogs/day)** |
| Blog language | **English with Indian market context** |
| Blog length | **Short at night (300–500w), long at open/close (800–1000w)** |
| Frontend | **Astro 6** |
| Theme | **Dark by default, light toggle — preference saved to Firebase** |
| Homepage sections | **All: gauge, blogs, screener, chart, indicators, backtest, changelog** |
| Changelog | **Every blog post shows changelog (what changed vs previous run) + zone changes highlighted** |
| ML model retrain | **Daily at 2 AM IST (full retrain using all accumulated historical data)** |
| Screener schedule | **Both: pre-market 6 AM IST + post-market 3:45 PM IST** |
| Setup script OS | **Python script that works on all platforms (Windows/Mac/Linux)** |

---

## Table of Contents

1. [Project Vision & Architecture](#1-project-vision--architecture)
2. [Signal Architecture](#2-signal-architecture)
3. [Investment Strategy & Rules](#3-investment-strategy--rules)
4. [30-Parameter Indicator Framework](#4-30-parameter-indicator-framework)
5. [ML Model — Research-Backed Stacked Ensemble](#5-ml-model--research-backed-stacked-ensemble)
6. [Stock Screener — Full Universe, Low PE/PB, PSU-Preferred](#6-stock-screener--full-universe-low-pepb-psu-preferred)
7. [Hourly Blog System — Full Integration](#7-hourly-blog-system--full-integration)
8. [Changelog System](#8-changelog-system)
9. [Website Architecture — Astro 6 + Cloudflare Pages](#9-website-architecture--astro-6--cloudflare-pages)
10. [Google Auth + Firebase Firestore](#10-google-auth--firebase-firestore)
11. [LLM News Analysis Pipeline (Multi-Provider)](#11-llm-news-analysis-pipeline-multi-provider)
12. [Free Tech Stack — Verified April 2026](#12-free-tech-stack--verified-april-2026)
13. [Free API Directory — Verified April 2026](#13-free-api-directory--verified-april-2026)
14. [Repository & Directory Structure](#14-repository--directory-structure)
15. [Data Models & Storage Schema](#15-data-models--storage-schema)
16. [GitHub Actions Workflows](#16-github-actions-workflows)
17. [One-Command Setup Script (Python, All Platforms)](#17-one-command-setup-script-python-all-platforms)
18. [Email Notification System (Gmail SMTP)](#18-email-notification-system-gmail-smtp)
19. [Backtesting System](#19-backtesting-system)
20. [Error Handling & Resilience](#20-error-handling--resilience)
21. [Environment Configuration (.env.example — Complete)](#21-environment-configuration-envexample--complete)
22. [Testing Strategy](#22-testing-strategy)
23. [Implementation Phases](#23-implementation-phases)
24. [Research Papers & Academic Foundation](#24-research-papers--academic-foundation)
25. [Design System & Frontend Aesthetic](#25-design-system--frontend-aesthetic)

---

## 1. Project Vision & Architecture

### 1.1 What NTI Does

The **Nifty Timing Index** is a personal tool (open to all) that:

1. **Every hour**: collects 30+ indicators from NSE/BSE, runs a stacked ensemble ML model, publishes a 0–100 danger score for 5 indices
2. **Every hour**: LLM generates a complete blog post (with changelog, indicators, stock picks, news analysis) and publishes it to the website at `/blog/{timestamp-slug}/` — fully integrated, SEO'd, linked from nav
3. **Twice daily** (6 AM + 3:45 PM IST): screens ALL NSE+BSE stocks ≥ ₹500 Cr for PE < 20 and PB < 3, ranks them (PSU stocks boosted), updates website
4. **On zone change**: Gmail SMTP alert sent immediately
5. **Google login**: users can save watchlists, alert prefs, custom thresholds in Firebase Firestore
6. **One-time setup**: user fills `.env`, runs `python scripts/setup.py` — the script creates GitHub repo, pushes all secrets, connects Cloudflare Pages, sets up Firebase — then never touch it again

### 1.2 High-Level Flow

```
Every Hour (GitHub Actions cron):
  ┌─────────────────────────────────────────────────────────┐
  │  SCRAPER LAYER                                          │
  │  NSE/BSE APIs → nsetools, yfinance, Selenium           │
  │  30 indicators collected + validated                    │
  └──────────────────────┬──────────────────────────────────┘
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │  INDICATOR PROCESSING                                   │
  │  Normalize → 0–100 per indicator                        │
  │  Compute custom composites (F&G, MTF risk)             │
  └──────────────────────┬──────────────────────────────────┘
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │  ML INFERENCE (Stacked Ensemble)                        │
  │  LightGBM + XGBoost + Random Forest → Logistic Meta    │
  │  → NTI Score 0–100 + Confidence + SHAP drivers         │
  └──────────────────────┬──────────────────────────────────┘
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │  LLM BLOG GENERATION                                    │
  │  RSS news → LLM → Blog markdown                        │
  │  Includes: score, changelog, stock picks, indicators    │
  └──────────────────────┬──────────────────────────────────┘
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │  DATA STORAGE                                           │
  │  CSV (signals) + JSON (API) + Markdown (blogs)         │
  │  Git commit & push → triggers Cloudflare Pages build   │
  └──────────────────────┬──────────────────────────────────┘
                         ↓
  ┌─────────────────────────────────────────────────────────┐
  │  WEBSITE (Astro 6 → Cloudflare Pages)                  │
  │  Blog live at /blog/{slug} + Dashboard updated         │
  │  Firebase Firestore user watchlists served dynamically  │
  └─────────────────────────────────────────────────────────┘

Daily 2 AM IST (separate workflow):
  Full ML retrain + backtest + model metadata update

Twice Daily 6 AM + 3:45 PM IST:
  Full stock screener run → update /screener page
```

---

## 2. Signal Architecture

### 2.1 NTI Score Zones

| Score | Zone | Meaning | Equity Exposure |
|-------|------|---------|-----------------|
| 0–15 | `EXTREME_BUY` | Panic/deep fear, fundamentals very cheap | Buy heavily |
| 16–30 | `STRONG_BUY` | Significant fear, valuations attractive | Buy |
| 31–45 | `BUY_LEAN` | Mild fear, decent fundamentals | Accumulate slowly |
| 46–55 | `NEUTRAL` | No conviction, wait | Hold, no new buys |
| 56–69 | `SELL_LEAN` | Stretched valuations, mild greed | Reduce exposure |
| 70–84 | `STRONG_SELL` | Overvalued + euphoria | Exit positions |
| 85–100 | `EXTREME_SELL` | Bubble territory, extreme greed | Fully exit, stay flat |

### 2.2 Per-Index Scores (5 indices)

Every run produces an independent NTI score for:
| Index | NSE Symbol | Purpose |
|-------|-----------|---------|
| Nifty 50 | `NIFTY 50` | Broad market primary signal |
| Nifty Bank | `NIFTY BANK` | Banking sector signal |
| Sensex / BSE 30 | `SENSEX` | BSE confirmation |
| Nifty Midcap 150 | `NIFTY MIDCAP 150` | Mid-cap signal (₹500Cr–₹50,000Cr stocks) |
| Nifty Smallcap 250 | `NIFTY SMALLCAP 250` | Small-cap signal (₹500Cr–₹5,000Cr stocks) |

The **primary signal** for investment decisions is always **Nifty 50**. Other indices are shown for context and sector-specific decisions.

### 2.3 Confidence Score

- Raw probability from meta-learner: 0.0–1.0
- Confidence = `abs(probability - 0.5) / 0.5` → maps to 0–100%
- **< 60%**: "⚠️ LOW CONFIDENCE — signal uncertain, wait for confirmation"
- **60–80%**: "📊 MODERATE CONFIDENCE"
- **> 80%**: "✅ HIGH CONFIDENCE"

### 2.4 Zone Change Detection

System compares current zone to previous run's zone:
- Same zone → no alert, changelog notes "Score moved within zone"
- Zone change (e.g., `NEUTRAL` → `SELL_LEAN`) → Gmail alert sent immediately + changelog section "🔴 ZONE CHANGE ALERT" in blog post

---

## 3. Investment Strategy & Rules

### 3.1 Absolute Rules (Never Override)

1. **NEVER short stocks** — zero short positions under any condition
2. **NEVER short Nifty/Bank Nifty futures** — not even as hedge
3. **NEVER buy when NTI score > 70** — no new positions in STRONG_SELL or EXTREME_SELL zone
4. **NEVER recommend stocks with PE ≥ 20** — hard cutoff, no industry-relative exceptions
5. **NEVER recommend stocks with PB ≥ 3** — hard cutoff
6. **NEVER recommend stocks < ₹500 Cr market cap**

### 3.2 Stock Universe Definition

```
Universe = ALL NSE-listed equities + BSE-listed equities
         where market_cap >= ₹500 Cr
         AND listed_for >= 1 year (no brand-new IPOs)

Data source: NSE EQUITY_L.csv (downloaded fresh daily)
URL: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
Append ".NS" for yfinance, ".BO" for BSE via yfinance
```

Approximate size: 800–1,500 stocks depending on market conditions.

### 3.3 PSU/Government Stock Preference

Chirag prefers government-backed companies. Implementation:

```python
PSU_INDICES = ["NIFTY PSE", "NIFTY CPSE", "NIFTY PSU BANK"]

# Known PSU tickers (from Nifty PSE + Nifty CPSE + Nifty PSU Bank indices)
PSU_STOCKS = [
    # Nifty CPSE
    "NTPC", "ONGC", "BEL", "POWERGRID", "COALINDIA",
    "NHPC", "OIL", "NLC", "COCHINSHIP", "SJVN", "NBCC",
    # Nifty PSE (additional)
    "HPCL", "BPCL", "GAIL", "IOC", "HAL", "RECLTD", "PFC",
    "IRFC", "BHEL", "NMDC", "MOIL", "NFL",
    # PSU Banks (Nifty PSU Bank)
    "SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK",
    "INDIANB", "BANKINDIA", "UCOBANK", "CENTRALBK", "MAHABANK",
    # Insurance PSU
    "GICRE", "NIACL", "ORIENTINS", "UIICL",
]

def get_psu_score_boost(symbol: str) -> float:
    """PSU stocks get a +10 boost to composite score (soft preference)."""
    return 10.0 if symbol in PSU_STOCKS else 0.0
```

Screener output always shows a "PSU/Govt" tag badge in the UI. PSU stocks are shown first within the same composite score band.

### 3.4 Stock Selection Hard Filters

```python
HARD_FILTERS = {
    "min_market_cap_cr": 500,       # ₹500 Cr minimum
    "max_pe": 20.0,                 # Strictly < 20, no exceptions
    "max_pb": 3.0,                  # Strictly < 3, no exceptions
    "max_debt_equity": 1.5,         # 1.5 for banks (higher leverage ok), 1.0 for others
    "min_roe_pct": 12.0,            # 12% ROE trailing 12 months
    "min_net_profit_positive_years": 1,  # At least 1 profitable year in last 3
    "min_listing_months": 12,       # Must be listed for at least 1 year
}
```

### 3.5 MTF (Margin Trading Facility) Risk Display

NTI shows MTF risk calculations for each recommended stock (informational, not investment advice):

```
MTF Risk Panel (for each stock):
  Leverage: 3x assumed
  10% fall → 30% capital loss
  20% fall → 60% capital loss
  Break-even (30-day hold @ 12% annual): stock must rise X%
  Margin call distance: how much stock can fall before margin call
```

---

## 4. 30-Parameter Indicator Framework

All indicators normalized 0–100: **0 = BUY opportunity (low danger), 100 = SELL danger (high danger)**.

Final NTI score = weighted sum of all normalized indicator scores.

### Tier 1 — Fundamental Valuation (35% total weight)

| # | Indicator | Weight | Raw Source | Python Method |
|---|-----------|--------|-----------|---------------|
| 1 | Nifty 50 P/E Ratio | 0.10 | NSE | `nsetools`: `nse.get_index_quote("NIFTY 50")["pe"]` |
| 2 | Nifty 50 P/B Ratio | 0.08 | NSE | `nsetools`: `nse.get_index_quote("NIFTY 50")["pb"]` |
| 3 | Earnings Yield vs 10Y Bond Spread | 0.06 | NSE + FRED | `(1/nifty_pe) - us_10y_yield` |
| 4 | Nifty Dividend Yield | 0.05 | NSE | `nsetools`: `nse.get_index_quote("NIFTY 50")["dy"]` |
| 5 | Market Cap to GDP (Buffett Indicator) | 0.04 | NSE + RBI | Scrape NSE total mcap page + RBI GDP data |
| 6 | Nifty Midcap 150 P/E | 0.02 | NSE | `nsetools`: `nse.get_index_quote("NIFTY MIDCAP 150")["pe"]` |

**Normalization (PE):**
- PE ≤ 12 → score 0 (very cheap, strong buy)
- PE = 16 → score 25 (historical mean, leaning buy)
- PE = 20 → score 50 (fair value boundary)
- PE = 24 → score 75 (expensive)
- PE ≥ 30 → score 100 (very expensive, sell)
- Formula: `score = min(100, max(0, (pe - 12) / (30 - 12) * 100))`

**Normalization (PB):**
- PB ≤ 1.5 → score 0
- PB = 2.5 → score 40
- PB = 3.5 → score 70
- PB ≥ 5.0 → score 100
- Formula: `score = min(100, max(0, (pb - 1.5) / (5.0 - 1.5) * 100))`

### Tier 2 — Sentiment as Contrarian Signal (25% total weight)

| # | Indicator | Weight | Raw Source | Python Method |
|---|-----------|--------|-----------|---------------|
| 7 | Tickertape MMI | 0.08 | tickertape.in | Selenium headless scrape: get number from MMI page |
| 8 | India VIX | 0.06 | NSE | `nsetools`: `nse.get_index_quote("INDIA VIX")["last"]` |
| 9 | Put/Call Ratio (PCR) | 0.04 | NSE options | Scrape NSE F&O data: total puts OI / total calls OI |
| 10 | Custom India F&G Composite | 0.04 | Computed | Weighted: 30% VIX + 30% PCR + 20% breadth + 20% 52wk H/L |
| 11 | CNN Fear & Greed Index | 0.02 | CNN Business | `requests` + `BeautifulSoup` scrape CNN F&G page |
| 12 | FII Net Cash Flow (daily) | 0.01 | NSE FII/DII | Scrape NSE daily FII/DII report CSV |

**Normalization (Contrarian — inverted):**
- MMI: raw score is itself 0–100 (fear=low, greed=high); NTI maps: MMI 0 → NTI score 0 (extreme fear = buy), MMI 100 → NTI score 100 (extreme greed = sell)
- VIX: VIX < 10 → score 80 (complacency = danger), VIX > 25 → score 10 (fear = opportunity), Formula: `score = max(0, min(100, 100 - (vix - 10) / (30 - 10) * 100))`
- PCR: PCR < 0.7 → score 80 (call buying = greed = danger), PCR > 1.3 → score 20 (put buying = fear = opportunity)
- FII net: sustained FII selling (< -2000 Cr/day) → score 15 (selling pressure = fear = contrarian buy)

### Tier 3 — Macro Fundamentals (25% total weight)

| # | Indicator | Weight | Raw Source | Python Method |
|---|-----------|--------|-----------|---------------|
| 13 | RBI Repo Rate & Stance | 0.06 | RBI website | Scrape RBI press releases page; parse rate + stance |
| 14 | India CPI Inflation | 0.05 | MOSPI India | Scrape mospi.gov.in monthly CPI release |
| 15 | US 10-Year Bond Yield | 0.05 | FRED API | `requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={FRED_KEY}&limit=1&sort_order=desc")` |
| 16 | USD/INR Exchange Rate | 0.04 | yfinance | `yf.download("INR=X", period="5d")["Close"].iloc[-1]` |
| 17 | Brent Crude Oil Price | 0.03 | yfinance | `yf.download("BZ=F", period="5d")["Close"].iloc[-1]` |
| 18 | S&P 500 Daily Change | 0.02 | yfinance | `yf.download("^GSPC", period="5d")["Close"].pct_change().iloc[-1]` |

**Normalization (Macro):**
- CPI: CPI < 4% → score 20 (good), CPI = 6% → score 60, CPI > 7% → score 85
- US 10Y: < 3.5% → score 15, 4.5% → score 55, > 5.5% → score 85
- USD/INR: depreciating INR (rising) → higher danger score
- Crude: < $70 → score 15, $80 → score 40, > $95 → score 80

### Tier 4 — Institutional Flow (10% total weight)

| # | Indicator | Weight | Raw Source | Python Method |
|---|-----------|--------|-----------|---------------|
| 19 | FII F&O Net Position (Index Futures) | 0.04 | NSE OI | Scrape NSE participant-wise OI report |
| 20 | DII Net Cash Flow (daily) | 0.03 | NSE FII/DII | Scrape NSE daily FII/DII report |
| 21 | AMFI Monthly SIP Flows | 0.02 | AMFI website | Scrape amfiindia.com monthly SIP data |
| 22 | GIFT Nifty Overnight Change | 0.01 | yfinance | `yf.download("^NSEI", period="2d")` or scrape GIFT exchange |

### Tier 5 — LLM News Analysis (5% total weight)

| # | Indicator | Weight | Raw Source | Method |
|---|-----------|--------|-----------|--------|
| 23 | LLM Fundamental News Danger Score | 0.02 | RSS feeds → LLM | Parse 5 RSS feeds, batch to LLM, get 0–100 score |
| 24 | LLM Policy/Budget Event Flag | 0.01 | RSS feeds → LLM | LLM detects: RBI policy, Budget, SEBI orders |
| 25 | LLM Geopolitical Risk Score | 0.01 | RSS feeds → LLM | LLM assesses global risk affecting Indian equities |
| 26 | Global Overnight Markets Composite | 0.01 | yfinance | Weighted: Dow Jones + NASDAQ + Nikkei + Hang Seng change |

**RSS News Sources (free, no key needed):**
- `https://economictimes.indiatimes.com/markets/rss.cms` (Economic Times)
- `https://www.moneycontrol.com/rss/latestnews.xml` (Moneycontrol)
- `https://www.business-standard.com/rss/markets-106.rss` (Business Standard)
- `https://feeds.livemint.com/livemint/lotze/2023-06/markets` (LiveMint)
- `https://thehindu.com/business/markets/?service=rss` (The Hindu BusinessLine)

### Tier 6 — Display Only (0% weight in ML model)

Collected, stored, displayed in blog/dashboard, but **not used** in NTI score calculation:

| # | Indicator | Display Purpose | Source |
|---|-----------|----------------|--------|
| 27 | Nifty RSI (14-day) | Overbought/oversold context | Calculated from yfinance OHLC |
| 28 | Nifty MACD | Momentum trend context | Calculated from yfinance OHLC |
| 29 | Advance-Decline Ratio | Market breadth context | Scrape NSE market stats |
| 30 | 52-Week Highs vs Lows | Breadth confirmation | Scrape NSE 52-week high/low data |

---

## 5. ML Model — Research-Backed Stacked Ensemble

### 5.1 Why This Model: Academic Justification

**Decision: Three-Layer Stacked Ensemble (LightGBM + XGBoost + Random Forest → Logistic Regression meta-learner)**

Research basis (all 2024–2026 verified):

| Paper | Finding | Relevance |
|-------|---------|-----------|
| "High-precision forecasting of Indian stock market indices using weighted ensemble of hyperparameter-tuned LightGBM models" (*Taylor & Francis*, Oct 2025) | LightGBM outperforms XGBoost specifically on Indian stock index data, with histogram-based learning and leaf-wise growth being superior for high-frequency tabular features | Primary justification for LightGBM as base learner |
| "Stock Price Prediction Using a Stacked Heterogeneous Ensemble" (*MDPI Finance*, Oct 2025) | XGBoost as meta-learner over RF+LightGBM base achieves R²=0.97–0.99; stacking beats any single model by 15–20% | Justifies stacking architecture |
| "Predicting Stock Returns Using Machine Learning: A Hybrid Approach with LightGBM, XGBoost, and Portfolio Optimization" (*Atlantis Press*, Feb 2025) | Combined LightGBM + XGBoost produces superior return predictions vs single models on S&P 500 stock-level data | Confirms the combination |
| "Ensemble Learning Methods: XGBoost and LightGBM Stacking" (*johal.in*, Sep 2025, citing MLPerf 2025) | Stacking XGBoost + LightGBM achieves 10–20% accuracy gain over single models on tabular data | General validation |
| "Stock Price Prediction Based on XGBoost and LightGBM" (*ResearchGate*, 2021) | "The combined model of XGBoost and LightGBM has better prediction performance than the single model and neural network" | Core combined model finding |
| "Prediction model of stock return based on tabular data" (*PeerJ CS*, 2023) | "Algorithms based on DNN are not satisfactory in processing and prediction of tabular data compared with tree-based models such as XGBoost and LightGBM" | Justifies NOT using LSTM/Transformer |

**Why NOT LSTM/Transformer for NTI:**
1. NTI features are **30 tabular indicators** (PE, VIX, PCR, etc.) — not raw price sequences
2. Deep learning needs 10,000+ samples; NTI cold-starts with limited data
3. Tree models handle **missing values natively** (critical when NSE scraper fails)
4. Trees provide **SHAP interpretability** — explaining WHY the score is high/low to the user
5. Trees retrain in **< 30 seconds** on GitHub Actions; LSTM would require minutes and GPU

### 5.2 Stacked Architecture (Three Layers)

```
LAYER 1 — Base Learners (trained in parallel, 5-fold cross-validation)
├── Model A: LightGBM (leaf-wise, fast, best for Indian market tabular data)
│   ├── n_estimators: 500
│   ├── learning_rate: 0.05
│   ├── num_leaves: 31
│   ├── max_depth: 6
│   ├── min_child_samples: 20
│   ├── subsample: 0.8
│   ├── colsample_bytree: 0.8
│   ├── reg_alpha: 0.1, reg_lambda: 0.1
│   └── class_weight: "balanced"
│
├── Model B: XGBoost (depth-wise, regularized, robust to outliers)
│   ├── n_estimators: 500
│   ├── learning_rate: 0.05
│   ├── max_depth: 6
│   ├── subsample: 0.8
│   ├── colsample_bytree: 0.8
│   ├── reg_alpha: 0.1, reg_lambda: 1.0
│   └── eval_metric: "logloss"
│
└── Model C: Random Forest (bagging, variance reduction, uncorrelated trees)
    ├── n_estimators: 300
    ├── max_depth: 8
    ├── min_samples_split: 10
    ├── min_samples_leaf: 5
    ├── max_features: "sqrt"
    └── class_weight: "balanced"

LAYER 2 — Meta-Learner
└── Logistic Regression with L2 regularization (C=1.0)
    Input: out-of-fold predictions from all 3 base learners (3 features)
    Output: final probability P(danger) ∈ [0.0, 1.0]
    (Logistic Regression chosen as meta-learner to prevent overfitting base learner outputs;
     supported by "An improved Stacking framework" paper, ScienceDirect 2019)

LAYER 3 — Score Conversion + Adjustments
└── NTI Score = P(danger) × 100
    → Apply MTF risk adjustment (+10 if MTF risk > 70, +20 if > 85)
    → Round to 1 decimal place
    → Compute SHAP values for top-3 driver explanation
```

### 5.3 Features Used in Model (26 engineered features)

```python
MODEL_FEATURES = [
    # Fundamental (6 features)
    "nifty_pe_normalized",         # PE normalized 0–100 (Tier 1 #1)
    "nifty_pb_normalized",         # PB normalized 0–100 (Tier 1 #2)
    "earnings_yield_bond_spread",  # (1/PE) - US_10Y_yield, normalized (Tier 1 #3)
    "dividend_yield_normalized",   # Div yield normalized (Tier 1 #4)
    "mcap_to_gdp_percentile",      # Buffett indicator 1-year percentile (Tier 1 #5)
    "midcap_pe_normalized",        # Midcap 150 PE normalized (Tier 1 #6)

    # Sentiment (5 features)
    "mmi_score",                   # Tickertape MMI 0–100 (Tier 2 #7)
    "vix_normalized",              # VIX normalized contrarian (Tier 2 #8)
    "pcr_normalized",              # PCR normalized contrarian (Tier 2 #9)
    "custom_fg_composite",         # Custom F&G composite (Tier 2 #10)
    "fii_cash_5d_avg_normalized",  # FII 5-day avg flow normalized (Tier 2 #12)

    # Macro (5 features)
    "rbi_rate_direction",          # +1 cutting, 0 hold, -1 hiking (Tier 3 #13)
    "cpi_normalized",              # CPI normalized 0–100 (Tier 3 #14)
    "us_10y_normalized",           # US 10Y normalized (Tier 3 #15)
    "usdinr_30d_change",           # 30-day USD/INR change normalized (Tier 3 #16)
    "crude_normalized",            # Crude price normalized (Tier 3 #17)

    # Flows (2 features)
    "fii_fo_net_normalized",       # FII F&O net position normalized (Tier 4 #19)
    "dii_net_normalized",          # DII net flow normalized (Tier 4 #20)

    # LLM (2 features)
    "llm_news_danger_score",       # LLM 0–100 danger score (Tier 5 #23)
    "global_overnight_normalized", # Global overnight change normalized (Tier 5 #26)

    # Lagged + derived features (6 features)
    "nti_score_lag1",              # Previous hourly score (continuity)
    "nti_score_lag24",             # Score same time yesterday
    "pe_5d_change_normalized",     # PE change over 5 days
    "vix_5d_change",               # VIX 5-day change
    "day_of_week",                 # 0=Mon to 4=Fri (calendar effect)
    "days_to_monthly_expiry",      # Days to NSE F&O monthly expiry
]
```

### 5.4 Training Labels (Future Return-Based, MTF-Aware)

```python
# Labels based on Nifty 50 forward return over next 5 trading days
# Conservative threshold because user uses MTF leverage (3x = risk amplification)

def create_binary_label(future_5d_return: float) -> int | None:
    if future_5d_return > 0.025:    # +2.5% rise → it was a BUY moment (low danger)
        return 0
    elif future_5d_return < -0.025: # -2.5% fall → it was a SELL moment (high danger)
        return 1
    else:
        return None  # Neutral ±2.5% → excluded from training (too noisy)

# Training set is built from ALL accumulated daily signal CSVs
# Rows with None labels are dropped before training
# Class imbalance handled by class_weight="balanced" in LightGBM and RF
```

### 5.5 Cold Start Strategy (No Historical Data)

When the system is first deployed with no historical data:

1. **Weeks 1–4**: Rule-based fallback score = weighted average of all normalized indicator scores (same weights as in indicator framework) — no ML model runs
2. **Weeks 4–8**: Bootstrap model from NSE historical PE/PB data (available from `nifty-pe-ratio.com` scrape going back to 1999) + yfinance historical prices → generate synthetic training data
3. **Week 8+**: Full ML model trained on accumulated real data; rule-based fallback disabled
4. **Always**: Rule-based fallback used if model file missing or training fails; flagged in blog/email

### 5.6 Model Persistence

- Model binary stored as **GitHub Action artifact** (uploaded at end of daily retrain, downloaded at start of hourly inference)
- Model **NOT committed to git** (binary, large)
- Model metadata IS committed to `data/model/metadata.json`:

```json
{
  "version": "2026-04-23",
  "training_samples": 847,
  "cv_accuracy": 0.71,
  "cv_roc_auc": 0.76,
  "feature_importance": {
    "nifty_pe_normalized": 0.18,
    "mmi_score": 0.14,
    "vix_normalized": 0.11
  },
  "training_date": "2026-04-23T02:00:00+05:30",
  "retrain_duration_seconds": 23
}
```

### 5.7 SHAP Interpretability

Every inference run computes SHAP values and stores top-3 drivers:

```python
import shap
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_current)

top_drivers = sorted(
    zip(FEATURES, shap_values[0]),
    key=lambda x: abs(x[1]),
    reverse=True
)[:3]
# Output: [("nifty_pe_normalized", 0.18, "danger"), ("mmi_score", -0.14, "buy"), ...]
```

These are shown in every blog post: **"Why is the score 42? Main drivers: ① Nifty PE above average (pushing toward sell) ② MMI shows Fear zone (pulling toward buy) ③ VIX elevated (contrarian buy signal)"**

---

## 6. Stock Screener — Full Universe, Low PE/PB, PSU-Preferred

### 6.1 Data Collection for Screener

**Step 1: Get all NSE stocks with market cap ≥ ₹500 Cr**

```python
# Download full NSE equity list (updated daily by NSE)
import requests, pandas as pd, io

def get_all_nse_stocks() -> pd.DataFrame:
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    session = requests.Session()
    session.headers.update(headers)
    # First hit NSE homepage to get cookies
    session.get("https://www.nseindia.com")
    resp = session.get(url)
    df = pd.read_csv(io.BytesIO(resp.content))
    return df[["SYMBOL", "NAME OF COMPANY", "SERIES", "ISIN NUMBER"]]
    # Returns ~2,000+ rows; filter to EQ series only
```

**Step 2: Fetch fundamentals via yfinance (PE, PB, market cap)**

```python
import yfinance as yf

def get_stock_fundamentals(symbol: str) -> dict:
    ticker = yf.Ticker(f"{symbol}.NS")
    info = ticker.info
    return {
        "pe": info.get("trailingPE"),           # Trailing PE
        "pb": info.get("priceToBook"),          # P/B ratio
        "market_cap_cr": info.get("marketCap", 0) / 1e7,  # Convert to Crores
        "dividend_yield": info.get("dividendYield", 0) * 100,
        "roe": info.get("returnOnEquity", 0) * 100,
        "debt_equity": info.get("debtToEquity", 0),
        "current_price": info.get("currentPrice"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }
```

**Step 3: Fetch analyst ratings via Finnhub (free, 60 calls/min)**

```python
import requests

def get_analyst_ratings(symbol: str, finnhub_key: str) -> dict:
    # Finnhub uses different symbol format for Indian stocks
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={finnhub_key}"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if data:
        latest = data[0]
        total = latest["strongBuy"] + latest["buy"] + latest["hold"] + latest["sell"] + latest["strongSell"]
        buy_pct = (latest["strongBuy"] + latest["buy"]) / max(total, 1) * 100
        return {"analyst_buy_pct": buy_pct, "analyst_count": total}
    return {"analyst_buy_pct": None, "analyst_count": 0}
```

**Step 4: PSU check**

```python
def is_psu(symbol: str) -> bool:
    return symbol in PSU_STOCKS  # Static list, updated quarterly
```

### 6.2 Screening Pipeline

```
INPUT: All NSE stocks (~2,000)
  ↓
FILTER 1: Series = "EQ" only (remove ETFs, bonds, rights)
  ↓ ~1,800 stocks
FILTER 2: Market Cap ≥ ₹500 Cr
  ↓ ~900 stocks
FILTER 3: PE < 20.0 (strictly, None PE = excluded)
  ↓ ~350 stocks
FILTER 4: PB < 3.0 (strictly, None PB = excluded)
  ↓ ~180 stocks
FILTER 5: Market Cap < ₹500 Cr re-check (yfinance sometimes differs from NSE)
  ↓ ~170 stocks
FILTER 6: Basic data validity (PE > 0, PB > 0, price > 0)
  ↓ ~150 stocks
SOFT FILTER: ROE > 12% (kept but flagged if fails — shown in UI as "low ROE warning")
SOFT FILTER: Debt/Equity < 1.5 (kept but flagged if fails)
  ↓
SCORE: composite score 0–100
  Value Score (50%): PE rank + PB rank + dividend yield rank
  Quality Score (30%): ROE rank + debt rank
  Analyst Score (20%): analyst buy% + target upside
  PSU Bonus: +10 added to composite score if PSU stock
  ↓
SORT: by composite score descending
OUTPUT: Top 50 stocks as final recommendations
```

### 6.3 Output JSON Schema

```json
{
  "screened_at": "2026-04-23T06:00:00+05:30",
  "run_type": "pre_market",
  "nti_score_at_screen": 42.3,
  "universe_size": 2043,
  "passing_filters": 147,
  "top_picks": [
    {
      "rank": 1,
      "symbol": "SBIN",
      "name": "State Bank of India",
      "sector": "Financial Services",
      "industry": "Banks - Public Sector",
      "is_psu": true,
      "market_cap_cr": 712000,
      "current_price": 792.5,
      "pe": 11.2,
      "pb": 1.8,
      "roe_pct": 19.4,
      "debt_equity": 0.0,
      "dividend_yield_pct": 2.4,
      "analyst_buy_pct": 72,
      "analyst_count": 25,
      "composite_score": 94,
      "value_score": 91,
      "quality_score": 88,
      "analyst_score": 72,
      "psu_boost": 10,
      "warnings": [],
      "why_picked": "PSU bank with very low PE (11.2x), strong ROE (19.4%), high dividend yield (2.4%). 72% of analysts rate BUY.",
      "mtf_risk": {
        "leverage_3x_margin_required_pct": 33.3,
        "ten_pct_fall_capital_loss_pct": 30.0,
        "margin_call_distance_pct": 22.5
      }
    }
  ],
  "sector_summary": {
    "Banks - Public Sector": {"count": 8, "avg_pe": 10.4, "avg_pb": 1.6},
    "Oil & Gas": {"count": 6, "avg_pe": 8.2, "avg_pb": 1.2},
    "Power Generation": {"count": 5, "avg_pe": 14.1, "avg_pb": 2.1}
  },
  "exclusion_summary": {
    "pe_too_high": 634,
    "pb_too_high": 412,
    "market_cap_too_small": 389,
    "missing_data": 91,
    "negative_pe": 88
  },
  "psu_stocks_count": 34
}
```

### 6.4 Screener Website Page (`/screener`)

- Full sortable table of all 50 stocks
- Filter controls: user can adjust PE/PB thresholds live (saved to Firebase if logged in)
- Column filters: sector, PSU-only toggle, min-ROE slider, min-analyst-buy slider
- Each row: stock symbol (clickable → `/stocks/{symbol}`), name, sector, PSU badge, PE, PB, ROE, analyst buy%, composite score bar
- Mobile-responsive card view on small screens
- **Last updated** timestamp shown prominently
- **Changelog**: which stocks entered/left the list vs previous run

---

## 7. Hourly Blog System — Full Integration

### 7.1 Blog Generation Schedule

| Time (IST) | Blog Type | LLM Word Target | Sections |
|-----------|----------|----------------|---------|
| 9:00 AM | Market Open Analysis | 800–1,000 words | Full: score, indicators, news, stock picks, outlook |
| 10:00–3:00 PM (hourly) | Mid-session Update | 400–600 words | Brief: score change, key movers, news flash |
| 3:30 PM | Market Close Analysis | 800–1,000 words | Full: day review, post-market outlook, screener |
| 4:00–8:00 PM (hourly, 2hr gap) | Post-market Update | 300–500 words | Brief: F&O data, FII/DII, overnight risks |
| 8:00 PM–9:00 AM (2hr gap) | Overnight/Global | 300–400 words | Global markets, GIFT Nifty, next session preview |

Approximate: **~20 blogs/day** (every hour market hours + every 2 hours off-hours).

### 7.2 Blog File Structure

Each blog is a Markdown file with frontmatter, stored in `website/src/content/blog/`:

```markdown
---
title: "NTI Update: Score 42 | BUY_LEAN Zone | Nifty at 24,180 | 2026-04-23 14:00 IST"
description: "Nifty Timing Index hourly update: NTI score held at 42 (BUY_LEAN). India VIX at 16.2, MMI shows Fear. Top value picks: SBIN, NTPC, COAL INDIA."
slug: "2026-04-23-14-00"
publishedAt: "2026-04-23T14:00:00+05:30"
ntiScore: 42
ntiZone: "BUY_LEAN"
ntiZonePrev: "BUY_LEAN"
zoneChanged: false
confidence: 74
nifty50Price: 24180.5
niftyBank: 52400.2
sensex: 79820.1
topDrivers: ["nifty_pe_normalized", "mmi_score", "vix_normalized"]
topStocks: ["SBIN", "NTPC", "COALINDIA", "POWERGRID", "ONGC"]
blogType: "mid_session"
---

[full markdown content generated by LLM]
```

### 7.3 LLM Blog Prompt System

```python
BLOG_PROMPT_TEMPLATE = """
You are NTI-Writer, the AI analyst for the Nifty Timing Index (NTI).
Write a {blog_type} market update blog post for Indian equity investors.

TODAY'S DATA ({timestamp} IST):
- NTI Danger Score: {nti_score}/100 (Zone: {zone})
- Previous Hour Score: {prev_score}/100 (Zone: {prev_zone})
- Confidence: {confidence}%
- Nifty 50: {nifty_price} ({nifty_change_pct:+.2f}%)
- India VIX: {vix}
- Tickertape MMI: {mmi} ({mmi_zone})
- Put/Call Ratio: {pcr}
- FII Net (today): ₹{fii_net:,.0f} Cr ({fii_direction})
- Nifty P/E: {nifty_pe}x | P/B: {nifty_pb}x | Div Yield: {div_yield}%
- US 10Y Yield: {us_10y}% | USD/INR: {usdinr} | Brent Crude: ${crude}/bbl
- Top SHAP Drivers: {top_drivers_text}

TOP VALUE PICKS (PE < 20, PB < 3, ≥ ₹500 Cr):
{top_stocks_formatted}

RECENT NEWS (from RSS feeds):
{news_headlines}

CHANGELOG (vs previous hour):
{changelog_text}

INSTRUCTIONS:
- Word target: {word_target} words
- Write in English with Indian market context
- Be specific about numbers, don't be vague
- Explain WHY the NTI score is at this level using the SHAP drivers
- For each top stock pick, explain PE, PB, ROE and why it's a value pick
- Mention PSU stocks specifically (government backing is a positive)
- Include the changelog section clearly
- Add a DISCLAIMER at the end: "This is not investment advice. NTI is a personal market analysis tool."
- Do NOT recommend shorting, options selling, or leveraged bearish strategies
- Output ONLY the blog post markdown content, no explanations
"""
```

### 7.4 LLM Provider Configuration (Multi-Provider, .env-Driven)

```python
# src/nti/llm/client.py

import os
from openai import OpenAI  # OpenAI-compatible client (works for all providers)

def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )

def generate_blog(prompt: str) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2000")),
        temperature=0.7,
    )
    return response.choices[0].message.content
```

**Supported providers (just change 3 env vars):**

| Provider | `LLM_API_KEY` | `LLM_BASE_URL` | `LLM_MODEL` |
|----------|-------------|--------------|------------|
| Groq (FREE) | groq api key | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Google Gemini (FREE) | gemini api key | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| OpenAI (paid) | openai api key | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Anthropic (paid) | anthropic api key | `https://api.anthropic.com/v1` | `claude-haiku-4-5` |
| Together AI (FREE credits) | together api key | `https://api.together.xyz/v1` | `meta-llama/Llama-3-70b-chat-hf` |
| Cerebras (FREE 1M/day) | cerebras api key | `https://api.cerebras.ai/v1` | `llama-3.1-70b` |

### 7.5 Blog-to-Website Integration (Critical)

Blogs must be **fully integrated** into the Astro 6 website, not silently dropped in a folder.

**Astro 6 Content Collections setup:**

```typescript
// website/src/content/config.ts
import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    description: z.string(),
    slug: z.string(),
    publishedAt: z.date(),
    ntiScore: z.number(),
    ntiZone: z.string(),
    ntiZonePrev: z.string(),
    zoneChanged: z.boolean(),
    confidence: z.number(),
    nifty50Price: z.number(),
    topDrivers: z.array(z.string()),
    topStocks: z.array(z.string()),
    blogType: z.enum(["market_open", "mid_session", "market_close", "post_market", "overnight"]),
  }),
});

export const collections = { blog };
```

**Blog index page (`/blog/`):**
```
/blog/                     → Paginated list of all blogs (20 per page, latest first)
/blog/[slug]/              → Individual blog post
/blog/[year]/[month]/      → Monthly archive
```

**Navigation integration:** The website header always shows "📝 Latest Blog" link pointing to the most recent post. The homepage shows the 5 latest blog cards in a sidebar/section.

**SEO:** Each blog auto-generates:
- `<title>` = blog title
- `<meta description>` = blog description
- Open Graph tags (for sharing)
- JSON-LD structured data (Article schema)
- Sitemap.xml (auto-generated by Astro)

---

## 8. Changelog System

### 8.1 What the Changelog Tracks (Every Blog Post)

Every blog post ends with a "**📋 What Changed Since Last Hour**" section:

```markdown
## 📋 What Changed Since Last Hour

**NTI Score:** 42 → 44 (+2 points, still BUY_LEAN zone)

**Indicator Changes:**
| Indicator | Previous | Current | Change |
|-----------|---------|---------|--------|
| Nifty PE | 21.1 | 21.3 | ↑ +0.2 (slightly more expensive) |
| India VIX | 17.2 | 16.8 | ↓ -0.4 (slightly less fear) |
| Tickertape MMI | 28 | 31 | ↑ +3 (moved from Fear to slightly less Fear) |
| FII Net | -823 Cr | -1240 Cr | ↓ More selling (-417 Cr) |

**Stock Screener Changes:**
- 🆕 Entered: NMDC (PE 8.1, PB 2.3 — passes all filters)
- ❌ Exited: BHEL (PE crossed 20x — no longer qualifies)

**No zone change.** Score remains in BUY_LEAN (31–45).
```

### 8.2 Zone Change Alert (Special Highlighting)

When zone changes, the changelog is expanded:

```markdown
## 🔴 ZONE CHANGE ALERT: NEUTRAL → SELL_LEAN

**Previous Score:** 52 (NEUTRAL zone)
**Current Score:** 57 (SELL_LEAN zone)

**What drove the zone change:**
- Nifty PE rose to 22.4x (above the 20x threshold — expensive territory)
- India VIX fell to 11.2 (complacency — contrarian danger signal)
- FII turned net sellers (₹-2,341 Cr today — 4th consecutive selling day)

**Action implication (NOT investment advice):**
According to NTI's model, this zone suggests REDUCING equity exposure.
This does NOT mean shorting — it means gradually moving to cash from existing long positions.
```

### 8.3 Changelog Data Storage

Previous run data stored in `data/api/previous_run.json` — compared against current run at the start of each blog generation.

---

## 9. Website Architecture — Astro 6 + Cloudflare Pages

### 9.1 Pages & Routes

| Route | Page | Dynamic? | Auth Required? |
|-------|------|---------|---------------|
| `/` | Dashboard (live NTI score, latest blogs, top picks) | Static (rebuilt hourly) | No |
| `/blog/` | Blog index (all posts, paginated) | Static | No |
| `/blog/[slug]/` | Individual blog post | Static | No |
| `/screener/` | Stock screener (full table, filters) | Static (rebuilt 2x/day) | No |
| `/stocks/[symbol]/` | Individual stock detail page | Static (rebuilt daily) | No |
| `/history/` | Historical NTI score chart | Static | No |
| `/backtest/` | Backtest results | Static | No |
| `/indicators/` | All 30 indicators breakdown | Static (rebuilt hourly) | No |
| `/account/` | User account (watchlist, preferences) | SSR | Yes (Google) |
| `/api/latest.json` | Current NTI data endpoint | Static file | No |
| `/api/history.json` | 30-day history | Static file | No |
| `/api/backtest.json` | Backtest results | Static file | No |
| `/sitemap.xml` | SEO sitemap | Auto-generated | No |
| `/robots.txt` | Search engine directives | Static | No |

### 9.2 Static vs SSR Architecture

- **All public pages**: Static Site Generation (SSG) — pre-built at deploy time, served from Cloudflare edge
- **`/account/` only**: Server-Side Rendering (SSR) — requires `@astrojs/cloudflare` adapter — reads Firebase session cookie, loads user data from Firestore
- **Static pages include all blogs and screener** — rebuilt every time GitHub Actions pushes

### 9.3 Data Flow for Website

```
GitHub Actions (every hour):
  1. Collect indicators → data/signals/hourly/{date}.csv
  2. Generate blog → website/src/content/blog/{slug}.md
  3. Update data/api/latest.json + history.json
  4. Git commit + push

Cloudflare Pages (triggered by push):
  1. npm run build (Astro builds all static pages including new blog)
  2. Deploy to global CDN edge
  3. New blog live at /blog/{slug}/ within ~60–90 seconds of GitHub push
```

### 9.4 Homepage Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  NAVBAR: NTI Logo | Dashboard | Blog | Screener |       │
│          History | Backtest | Account | 🌙/☀️ Theme     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │  NTI SCORE GAUGE     │  │  INDEX SCORES            │ │
│  │  [ Big gauge: 42 ]   │  │  Nifty 50:  42 BUY_LEAN │ │
│  │  Zone: BUY_LEAN      │  │  Bank Nifty: 48 NEUTRAL  │ │
│  │  Confidence: 74%     │  │  Sensex:    41 BUY_LEAN  │ │
│  │  Updated: 14:00 IST  │  │  Midcap:    38 BUY_LEAN  │ │
│  └──────────────────────┘  │  Smallcap:  35 BUY_LEAN  │ │
│                             └──────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  TOP VALUE PICKS (PE<20, PB<3)                          │
│  [SBIN ★PSU PE:11.2] [NTPC ★PSU PE:14.1] [COALINDIA]   │
├─────────────────────────────────────────────────────────┤
│  TOP SHAP DRIVERS                                       │
│  ① PE above avg (→ sell) ② MMI Fear (→ buy) ③ VIX high │
├─────────────────────────────────────────────────────────┤
│  KEY INDICATORS (horizontal scroll on mobile)           │
│  VIX:16.2 | MMI:31 | PCR:0.94 | FII:-1240Cr | PE:21.3 │
├─────────────────────────────────────────────────────────┤
│  LATEST BLOGS (3 cards)          NTI SCORE CHART (30d)  │
│  [14:00 Blog card]               [Score line vs Nifty]  │
│  [12:00 Blog card]                                      │
│  [10:00 Blog card]                                      │
│  → View All Blogs                                       │
├─────────────────────────────────────────────────────────┤
│  FOOTER: Disclaimer | GitHub | Not Investment Advice    │
└─────────────────────────────────────────────────────────┘
```

### 9.5 Blog Listing Page (`/blog/`)

```
┌─────────────────────────────────────────────────────────┐
│  📝 NTI Blog — Market Analysis & Updates                │
│  Hourly insights generated automatically by AI          │
├─────────────────────────────────────────────────────────┤
│  Filter: [All] [Market Open] [Close] [Overnight]        │
│  Date picker: [Apr 23, 2026 ▼]                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                  │
│  │ 14:00   │ │ 12:00   │ │ 10:00   │                  │
│  │ NTI: 42 │ │ NTI: 40 │ │ NTI: 38 │                  │
│  │BUY_LEAN │ │BUY_LEAN │ │BUY_LEAN │                  │
│  │[excerpt]│ │[excerpt]│ │[excerpt]│                  │
│  │→ Read   │ │→ Read   │ │→ Read   │                  │
│  └─────────┘ └─────────┘ └─────────┘                  │
│  [Load More / Pagination]                               │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Google Auth + Firebase Firestore

### 10.1 Firebase Services Used

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Firebase Authentication | Google login | Free for 50,000 MAU |
| Firestore Database | User data storage | 1 GB storage, 50K reads/day free |

### 10.2 User Data Stored in Firestore

```
Firestore structure:
  users/{uid}/
    profile:
      email: string
      displayName: string
      photoURL: string
      createdAt: timestamp
    preferences:
      theme: "dark" | "light"
      emailAlerts: boolean
      alertEmail: string
      alertOnZoneChange: boolean
      customPELimit: number (default: 20)
      customPBLimit: number (default: 3)
      customMinMarketCap: number (default: 500)
      showPSUOnly: boolean (default: false)
    watchlist:
      [{symbol, name, addedAt, notes}]
    alertHistory:
      [{triggeredAt, fromZone, toZone, ntiScore}]
```

### 10.3 Google Auth Flow in Astro 6 (SSR)

```
User clicks "Sign in with Google" on /account/ page
  → Firebase client SDK popup
  → Google OAuth consent screen
  → Firebase returns ID token
  → POST /api/auth/session with ID token
  → Server-side: Firebase Admin SDK verifies token
  → Create session cookie (7-day expiry)
  → Redirect to /account/dashboard

Every /account/ page load:
  → Read session cookie
  → Firebase Admin SDK: verifySessionCookie()
  → Fetch user data from Firestore
  → Render page with user data
```

### 10.4 Firebase Setup (Handled by setup.py Script)

```python
# setup.py automatically:
# 1. Creates Firebase project via Firebase CLI
# 2. Enables Google Auth provider
# 3. Creates Firestore database in asia-south1 region (Mumbai)
# 4. Generates service account key → stored as GitHub Secret
# 5. Adds Firebase config to Cloudflare Pages env vars
```

---

## 11. LLM News Analysis Pipeline (Multi-Provider)

### 11.1 RSS Feed Parsing

```python
import feedparser
from datetime import datetime, timedelta

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://feeds.livemint.com/livemint/lotze/2023-06/markets",
    "https://thehindu.com/business/markets/?service=rss",
]

def fetch_recent_news(hours: int = 4) -> list[dict]:
    """Fetch news from last N hours from all RSS feeds."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    articles = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:  # Max 10 per feed
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:200],
                "source": feed.feed.get("title", ""),
                "published": entry.get("published", ""),
            })
    return articles[:30]  # Max 30 total headlines per LLM call
```

### 11.2 LLM News Analysis Prompt

```python
NEWS_ANALYSIS_PROMPT = """
Analyze these recent Indian stock market headlines and provide:
1. Overall danger score (0=buy opportunity, 100=sell danger)
2. Top 3 market-moving events
3. Sector impacts (banking, IT, energy, pharma, metals)
4. Policy/regulatory flags (RBI, SEBI, budget, government announcements)
5. Geopolitical risks affecting Indian equities

Headlines:
{headlines}

Respond ONLY in JSON format:
{{
  "danger_score": <0-100>,
  "reasoning": "<2-sentence explanation>",
  "key_events": ["event1", "event2", "event3"],
  "sector_impacts": {{"banking": "positive/neutral/negative", "it": ..., "energy": ..., "pharma": ..., "metals": ...}},
  "policy_flag": "<description or 'none'>",
  "geopolitical_risk": <0-100>,
  "news_sentiment": "bullish/neutral/bearish"
}}
"""
```

---

## 12. Free Tech Stack — Verified April 2026

### 12.1 Python Backend

| Package | Version | Purpose | Cost |
|---------|---------|---------|------|
| `uv` | latest | Package manager (fast, Rust-based) | Free |
| `pandas` | latest | Data manipulation | Free |
| `numpy` | latest | Numerical operations | Free |
| `lightgbm` | latest | Primary ML base learner | Free |
| `xgboost` | ≥ 2.0.3 (Feb 2026 release) | Secondary ML base learner | Free |
| `scikit-learn` | latest | RF, Logistic meta-learner, metrics | Free |
| `shap` | latest | Model interpretability | Free |
| `yfinance` | latest | Yahoo Finance data (global markets, Indian stocks) | Free |
| `nsetools` | latest | NSE India data (PE, PB, VIX, indices) | Free |
| `nsepy` | latest | NSE historical data | Free |
| `selenium` | latest | Headless Chrome for Tickertape MMI | Free |
| `webdriver-manager` | latest | Manages ChromeDriver automatically | Free |
| `requests` | latest | HTTP scraping | Free |
| `beautifulsoup4` | latest | HTML parsing | Free |
| `feedparser` | latest | RSS feed parsing | Free |
| `openai` | latest | OpenAI-compatible client (works for all LLM providers) | Free client |
| `python-dotenv` | latest | .env loading | Free |
| `pyyaml` | latest | Config files | Free |
| `matplotlib` | latest | Chart generation for README badges | Free |
| `jinja2` | latest | Email HTML templates | Free |
| `pytest` | latest | Testing | Free |
| `ruff` | latest | Linting + formatting | Free |

### 12.2 Frontend (Website)

| Technology | Version | Purpose | Cost |
|------------|---------|---------|------|
| Astro | 6.x | Static site framework (Cloudflare-owned) | Free |
| React | 18+ | Interactive islands (charts, gauges, auth) | Free |
| Tailwind CSS | 4.x | Styling utility framework | Free |
| Chart.js | latest | Interactive charts (NTI history, indicator charts) | Free |
| Firebase JS SDK | 10+ | Client-side auth + Firestore | Free |
| TypeScript | 5+ | Type safety | Free |

### 12.3 Infrastructure

| Service | Free Tier | Usage |
|---------|-----------|-------|
| **GitHub** (public repo) | Unlimited Actions minutes | CI/CD, cron jobs, data storage |
| **Cloudflare Pages** | Unlimited bandwidth, unlimited builds | Website hosting |
| **Firebase Auth** | 50,000 MAU | Google login |
| **Firebase Firestore** | 1 GB, 50K reads/day | User data |
| **Groq API** | 1,000 req/day on Llama 3.3 70B | LLM (blog generation, news analysis) |
| **Finnhub** | 60 calls/min | Analyst ratings for screener |
| **FRED API** | 120 req/min | US 10Y bond yield |
| **Gmail SMTP** | 500 emails/day | Alert emails |

---

## 13. Free API Directory — Verified April 2026

### 13.1 Market Data (All Free, No Key Needed)

| API | Method | Data Available | Notes |
|-----|--------|---------------|-------|
| `nsetools` Python lib | pip install | NSE index quotes (PE, PB, VIX, advances/declines) | Free, reliable |
| `yfinance` Python lib | pip install | Global + Indian (.NS suffix) OHLCV, fundamentals | Free, no key |
| NSE EQUITY_L.csv | HTTP download | All NSE listed stocks | `https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv` |
| NSE FII/DII CSV | HTTP download | Daily FII/DII net flows | `https://archives.nseindia.com/content/nsccl/fao_participant_vol.csv` |
| NSE Options Chain | HTTP scrape | PCR calculation | NSE API endpoint (requires session cookie) |
| FRED API | REST API (free key) | US 10Y yield, Fed rate | `https://api.stlouisfed.org/fred/` |

### 13.2 News & RSS (All Free, No Key)

| Source | RSS URL | Content |
|--------|---------|---------|
| Economic Times Markets | `https://economictimes.indiatimes.com/markets/rss.cms` | Top Indian market news |
| Moneycontrol | `https://www.moneycontrol.com/rss/latestnews.xml` | Latest financial news |
| Business Standard | `https://www.business-standard.com/rss/markets-106.rss` | Market analysis |
| LiveMint Markets | `https://feeds.livemint.com/livemint/lotze/2023-06/markets` | Market updates |
| AMFI | Scrape `amfiindia.com` | Monthly SIP data |

### 13.3 LLM APIs (Free Tiers, Verified April 2026)

| Provider | Free Tier | Daily Limit | Speed | Notes |
|----------|-----------|-----------|-------|-------|
| Groq | 1,000 req/day | 1,000 | Ultra-fast (LPU hardware) | Llama 3.3 70B; best for fast blog generation |
| Google Gemini | 15 RPM | ~600/day | Fast | Gemini 2.0 Flash; large context window |
| Cerebras | Free | 1M tokens/day | Ultra-fast | Llama 3.1 70B; best daily token budget |
| Together AI | Free credits | Depends | Medium | Multiple open models |
| OpenRouter | 50 req/day | 50 | Varies | Routes to multiple free models |

**Recommendation**: Use Groq as default (fastest, 1,000/day is sufficient for ~20 blogs + news analysis = ~22 calls/day). If hitting limits, switch to Cerebras (1M tokens/day — most generous).

### 13.4 Email (Gmail SMTP)

```python
# Gmail SMTP configuration
GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587
# Use App Password (not account password)
# Set up: Google Account → Security → 2-Step Verification → App Passwords
# Free: 500 emails/day
```

---

## 14. Repository & Directory Structure

```
nifty-timing-index/
├── .github/
│   └── workflows/
│       ├── hourly-signal.yml         # Every hour: scrape + inference + blog + push
│       ├── daily-retrain.yml          # 2 AM IST: full model retrain + backtest
│       ├── daily-screener-am.yml      # 6 AM IST: pre-market stock screener
│       ├── daily-screener-pm.yml      # 3:45 PM IST: post-market stock screener
│       ├── deploy-website.yml         # Triggered by push: build + deploy Cloudflare Pages
│       └── bootstrap-data.yml         # One-time: seed 2 years of historical data
│
├── src/nti/                          # Python backend package
│   ├── __init__.py
│   ├── config/
│   │   ├── settings.py               # Central config loaded from env vars
│   │   ├── thresholds.py             # All normalization thresholds
│   │   ├── psu_stocks.py             # PSU stock list (updated quarterly)
│   │   ├── indicators_config.yaml    # Indicator weights and descriptions
│   │   └── holidays.py               # Indian market holiday calendar
│   │
│   ├── scrapers/
│   │   ├── nse_indices.py            # NSE index data: PE, PB, VIX, A/D via nsetools
│   │   ├── nse_fii_dii.py            # FII/DII daily flows from NSE archives
│   │   ├── nse_options.py            # PCR from NSE options chain
│   │   ├── nse_stocks.py             # All NSE stocks list from EQUITY_L.csv
│   │   ├── tickertape_mmi.py         # MMI via Selenium headless Chrome
│   │   ├── yahoo_finance.py          # yfinance: global markets, INR, crude, S&P500
│   │   ├── fred_api.py               # US 10Y yield from FRED
│   │   ├── rbi_data.py               # RBI repo rate from RBI website
│   │   ├── mospi_data.py             # CPI inflation from MOSPI
│   │   ├── amfi_data.py              # SIP flows from AMFI
│   │   ├── cnn_fear_greed.py         # CNN F&G via requests + BeautifulSoup
│   │   └── rss_news.py               # RSS feed aggregator (5 sources)
│   │
│   ├── indicators/
│   │   ├── normalizer.py             # Normalize raw values to 0–100
│   │   ├── composite.py              # Custom F&G composite, global overnight composite
│   │   ├── technical_display.py      # RSI, MACD (display-only, not in model)
│   │   └── feature_engineer.py       # Build ML feature vector from indicators
│   │
│   ├── model/
│   │   ├── trainer.py                # Full daily retrain: LightGBM + XGBoost + RF + LR meta
│   │   ├── predictor.py              # Hourly inference: load model + produce NTI score
│   │   ├── labeler.py                # Generate training labels from historical Nifty returns
│   │   ├── explainer.py              # SHAP values → top-3 drivers
│   │   └── fallback.py               # Rule-based weighted average (cold start / model failure)
│   │
│   ├── screener/
│   │   ├── universe.py               # Build stock universe (all NSE, market cap filter)
│   │   ├── filters.py                # Hard filters: PE < 20, PB < 3, etc.
│   │   ├── scorer.py                 # Composite scoring + PSU boost
│   │   ├── fundamentals.py           # Fetch PE, PB, ROE, debt via yfinance
│   │   └── analyst_ratings.py        # Fetch analyst ratings from Finnhub
│   │
│   ├── llm/
│   │   ├── client.py                 # OpenAI-compatible client (multi-provider via .env)
│   │   ├── blog_generator.py         # Generate blog markdown from NTI data + news
│   │   ├── news_analyzer.py          # Analyze RSS news → danger score + key events
│   │   └── prompts.py                # All prompt templates
│   │
│   ├── changelog/
│   │   └── generator.py              # Compare current vs previous run → changelog markdown
│   │
│   ├── notifications/
│   │   ├── email_sender.py           # Gmail SMTP email sending
│   │   └── templates/
│   │       ├── zone_change_alert.html  # Zone change email template
│   │       └── hourly_digest.html    # Hourly digest email (sent only on zone change)
│   │
│   ├── storage/
│   │   ├── csv_writer.py             # Write signals to CSV files
│   │   ├── json_api.py               # Update data/api/*.json files
│   │   ├── blog_writer.py            # Write blog .md file to website/src/content/blog/
│   │   └── git_committer.py          # Git add + commit + push from Actions
│   │
│   └── pipelines/
│       ├── hourly.py                 # Orchestrate full hourly run
│       ├── daily_retrain.py          # Orchestrate daily retrain
│       └── screener_run.py           # Orchestrate screener run
│
├── data/                             # All data files (committed to git)
│   ├── signals/
│   │   ├── nifty_50.csv              # Daily NTI scores for Nifty 50
│   │   ├── nifty_bank.csv
│   │   ├── sensex.csv
│   │   ├── nifty_midcap_150.csv
│   │   └── nifty_smallcap_250.csv
│   ├── indicators/
│   │   └── hourly/
│   │       └── YYYY-MM-DD.csv        # Hourly indicator snapshots per day
│   ├── screener/
│   │   ├── latest_pre_market.json    # Latest pre-market screener results
│   │   └── latest_post_market.json   # Latest post-market screener results
│   ├── model/
│   │   └── metadata.json             # Model version, accuracy, feature importance
│   ├── api/
│   │   ├── latest.json               # Current NTI score + all indicators + top picks
│   │   ├── history.json              # Last 30 days of daily NTI scores
│   │   └── backtest.json             # Latest backtest metrics
│   ├── backtest/
│   │   └── report.json               # Full backtest report
│   └── errors/
│       └── YYYY-MM-DD.json           # Error logs per day
│
├── website/                          # Astro 6 website
│   ├── astro.config.mjs              # Astro config (Cloudflare adapter + React islands)
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.mjs
│   ├── public/
│   │   ├── favicon.svg
│   │   ├── robots.txt
│   │   └── _headers                  # Cloudflare Pages security headers
│   └── src/
│       ├── content/
│       │   ├── config.ts             # Astro content collections config (blog schema)
│       │   └── blog/                 # Hourly blog .md files (auto-generated by Python)
│       │       └── YYYY-MM-DD-HH-MM.md
│       ├── layouts/
│       │   ├── Layout.astro          # Base layout (nav, footer, theme toggle)
│       │   └── BlogPost.astro        # Blog post layout
│       ├── pages/
│       │   ├── index.astro           # Dashboard homepage
│       │   ├── blog/
│       │   │   ├── index.astro       # Blog listing page (paginated)
│       │   │   └── [slug].astro      # Individual blog post
│       │   ├── screener/
│       │   │   └── index.astro       # Stock screener page
│       │   ├── stocks/
│       │   │   └── [symbol].astro    # Individual stock detail page
│       │   ├── history/
│       │   │   └── index.astro       # Historical NTI score chart
│       │   ├── backtest/
│       │   │   └── index.astro       # Backtest results
│       │   ├── indicators/
│       │   │   └── index.astro       # All 30 indicators breakdown
│       │   ├── account/
│       │   │   └── index.astro       # User account (SSR, requires auth)
│       │   └── api/
│       │       ├── auth/
│       │       │   ├── session.ts    # POST: create Firebase session cookie
│       │       │   └── signout.ts    # GET: delete session cookie
│       │       ├── latest.json.ts    # Proxy latest.json
│       │       └── history.json.ts   # Proxy history.json
│       └── components/
│           ├── NTIGauge.tsx          # Semi-circular gauge (React island)
│           ├── IndexScores.tsx       # Score cards for 5 indices (React)
│           ├── BlogCard.tsx          # Blog card (Astro)
│           ├── StockTable.tsx        # Sortable stock screener table (React)
│           ├── NTIChart.tsx          # History chart (React + Chart.js)
│           ├── IndicatorTable.tsx    # All 30 indicators table (Astro)
│           ├── ChangelogBlock.tsx    # Changelog section in blog (Astro)
│           ├── ZoneChangeBanner.tsx  # Red/green zone change banner (React)
│           ├── ThemeToggle.tsx       # Dark/light toggle (React island)
│           ├── GoogleLoginButton.tsx # Firebase Google auth button (React)
│           ├── Watchlist.tsx         # User watchlist (React, SSR via account page)
│           └── MTFRiskPanel.tsx      # MTF risk calculator display (React)
│
├── scripts/
│   ├── setup.py                      # One-command setup: GitHub repo + secrets + CF Pages + Firebase
│   ├── bootstrap_data.py             # Seed 2 years of historical data (run once)
│   └── validate_env.py               # Check all required env vars are set
│
├── tests/
│   ├── test_scrapers.py
│   ├── test_indicators.py
│   ├── test_model.py
│   ├── test_screener.py
│   ├── test_blog_generator.py
│   ├── test_changelog.py
│   └── test_storage.py
│
├── .env.example                      # Template for ALL required env vars (no secrets)
├── .gitignore                        # Includes .env, __pycache__, .pytest_cache, etc.
├── pyproject.toml                    # Python project config (uv)
├── uv.lock                           # Lock file for reproducibility
├── AGENTS.md                         # Instructions for AI coding agents
├── plan.md                           # This document
└── README.md                         # Live dashboard (updated every hourly run)
```

---

## 15. Data Models & Storage Schema

### 15.1 Hourly Signal CSV (`data/signals/nifty_50.csv`)

```csv
timestamp,date,hour_ist,nti_score,nti_score_prev,zone,zone_prev,zone_changed,
confidence,nifty_price,nifty_change_pct,
nifty_pe,nifty_pb,nifty_dy,earnings_yield_bond_spread,mcap_to_gdp,
mmi,india_vix,pcr,custom_fg,cnn_fg,fii_cash_net,
rbi_repo_rate,cpi_inflation,us_10y_yield,usd_inr,brent_crude,sp500_change,
fii_fo_net,dii_net,sip_flow_monthly,gift_nifty_change,
llm_news_danger,llm_policy_flag,llm_geopolitical,global_overnight,
rsi_14,macd,adv_decline_ratio,high_low_ratio,
top_driver_1,top_driver_1_shap,top_driver_2,top_driver_2_shap,top_driver_3,top_driver_3_shap,
blog_slug,model_version,is_market_hours,is_holiday,run_duration_seconds,errors
```

### 15.2 API JSON (`data/api/latest.json`)

```json
{
  "timestamp": "2026-04-23T14:00:00+05:30",
  "market_status": "open",
  "primary_index": "nifty_50",
  "indices": {
    "nifty_50": {
      "score": 42.3,
      "score_prev": 40.1,
      "zone": "BUY_LEAN",
      "zone_prev": "BUY_LEAN",
      "zone_changed": false,
      "confidence": 74,
      "price": 24180.5,
      "change_pct": 0.42
    },
    "nifty_bank": {"score": 48.1, "zone": "NEUTRAL", "confidence": 68},
    "sensex": {"score": 41.5, "zone": "BUY_LEAN", "confidence": 71},
    "nifty_midcap_150": {"score": 38.2, "zone": "BUY_LEAN", "confidence": 65},
    "nifty_smallcap_250": {"score": 35.1, "zone": "BUY_LEAN", "confidence": 61}
  },
  "indicators": {
    "nifty_pe": 21.3, "nifty_pb": 3.1, "nifty_dy": 1.28,
    "mmi": 31, "india_vix": 16.2, "pcr": 0.94,
    "fii_cash_net": -1240, "us_10y": 4.32, "usd_inr": 83.42,
    "brent_crude": 74.5, "llm_news_danger": 42, "cpi": 4.2
  },
  "top_drivers": [
    {"indicator": "nifty_pe_normalized", "label": "Nifty PE", "shap": 0.18, "direction": "sell", "current_value": "21.3x (slightly above fair value)"},
    {"indicator": "mmi_score", "label": "Market Mood (MMI)", "shap": -0.14, "direction": "buy", "current_value": "31 — Fear zone"},
    {"indicator": "vix_normalized", "label": "India VIX", "shap": -0.11, "direction": "buy", "current_value": "16.2 — elevated (fear)"}
  ],
  "top_stock_picks": [
    {"symbol": "SBIN", "pe": 11.2, "pb": 1.8, "is_psu": true, "composite_score": 94},
    {"symbol": "NTPC", "pe": 14.1, "pb": 2.1, "is_psu": true, "composite_score": 88}
  ],
  "latest_blog_slug": "2026-04-23-14-00",
  "model_version": "2026-04-23",
  "model_confidence": 74
}
```

---

## 16. GitHub Actions Workflows

### 16.1 Hourly Signal Workflow (`.github/workflows/hourly-signal.yml`)

```yaml
name: NTI Hourly Signal

on:
  schedule:
    # Market hours (9 AM – 4 PM IST = 3:30–10:30 UTC): every hour
    - cron: "30 3-10 * * 1-5"  # Mon–Fri, market hours
    # Extended hours (6 AM – 8 PM IST = 0:30–14:30 UTC): every hour
    - cron: "30 0-14 * * 1-5"  # Mon–Fri, extended
    # Overnight (8 PM – 6 AM IST, every 2 hours): limited runs
    - cron: "30 15,17,19,21,23 * * *"  # Global overnight updates
    # Weekends (every 2 hours for global news)
    - cron: "30 3,7,11,15,19,23 * * 0,6"
  workflow_dispatch:  # Allow manual trigger

jobs:
  generate-signal:
    runs-on: ubuntu-latest
    permissions:
      contents: write  # Allow git push

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for git operations

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Setup Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --frozen

      - name: Install Chrome (for Selenium)
        run: |
          sudo apt-get update -q
          sudo apt-get install -y google-chrome-stable

      - name: Download latest ML model artifact
        uses: actions/download-artifact@v4
        with:
          name: nti-model
          path: model_artifacts/
        continue-on-error: true  # OK if no model yet (cold start)

      - name: Run hourly pipeline
        env:
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}
          LLM_MODEL: ${{ secrets.LLM_MODEL }}
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
          GMAIL_ADDRESS: ${{ secrets.GMAIL_ADDRESS }}
          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
          ALERT_EMAIL_TO: ${{ secrets.ALERT_EMAIL_TO }}
          FIREBASE_SERVICE_ACCOUNT: ${{ secrets.FIREBASE_SERVICE_ACCOUNT }}
        run: uv run python -m nti.pipelines.hourly
        continue-on-error: true  # Log errors, don't fail workflow

      - name: Configure git
        run: |
          git config --local user.email "nti-bot@github.com"
          git config --local user.name "NTI Bot"

      - name: Commit and push data + blog
        run: |
          git add data/ website/src/content/blog/ README.md
          git diff-index --quiet HEAD || git commit -m "NTI Update $(date -u '+%Y-%m-%d %H:%M') UTC"
          git push
```

### 16.2 Daily Retrain Workflow (`.github/workflows/daily-retrain.yml`)

```yaml
name: NTI Daily Model Retrain

on:
  schedule:
    - cron: "30 20 * * *"  # 2:00 AM IST = 20:30 UTC previous day
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv sync --frozen

      - name: Run full retrain + backtest
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
        run: uv run python -m nti.pipelines.daily_retrain

      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: nti-model
          path: model_artifacts/
          retention-days: 7  # Keep for 7 days

      - name: Push model metadata + backtest results
        run: |
          git config --local user.email "nti-bot@github.com"
          git config --local user.name "NTI Bot"
          git add data/model/ data/backtest/ data/api/backtest.json
          git diff-index --quiet HEAD || git commit -m "Daily retrain $(date -u '+%Y-%m-%d')"
          git push
```

### 16.3 Screener Workflows

```yaml
# daily-screener-am.yml — 6:00 AM IST = 00:30 UTC
name: NTI Pre-Market Screener
on:
  schedule:
    - cron: "30 0 * * 1-5"  # Mon–Fri only
  workflow_dispatch:
jobs:
  screener:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with: {fetch-depth: 0}
      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv sync --frozen
      - name: Run pre-market screener
        env:
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
        run: uv run python -m nti.pipelines.screener_run --type pre_market
      - name: Commit screener results
        run: |
          git config --local user.email "nti-bot@github.com"
          git config --local user.name "NTI Bot"
          git add data/screener/
          git diff-index --quiet HEAD || git commit -m "Pre-market screener $(date -u '+%Y-%m-%d')"
          git push
```

### 16.4 Website Deploy Workflow (`.github/workflows/deploy-website.yml`)

```yaml
name: Deploy to Cloudflare Pages

on:
  push:
    branches: [main]
    paths:
      - "website/**"
      - "data/api/**"
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "22"  # Astro 6 requires Node 22+
          cache: "npm"
          cache-dependency-path: website/package-lock.json

      - name: Install website dependencies
        run: cd website && npm ci

      - name: Copy API data to website public
        run: cp -r data/api website/public/api

      - name: Build Astro website
        env:
          FIREBASE_API_KEY: ${{ secrets.FIREBASE_API_KEY }}
          FIREBASE_PROJECT_ID: ${{ secrets.FIREBASE_PROJECT_ID }}
          FIREBASE_APP_ID: ${{ secrets.FIREBASE_APP_ID }}
          PUBLIC_FIREBASE_API_KEY: ${{ secrets.FIREBASE_API_KEY }}
          PUBLIC_FIREBASE_PROJECT_ID: ${{ secrets.FIREBASE_PROJECT_ID }}
        run: cd website && npm run build

      - name: Deploy to Cloudflare Pages
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          command: pages deploy website/dist --project-name=nifty-timing-index --commit-dirty=true
```

---

## 17. One-Command Setup Script (Python, All Platforms)

The user fills `.env`, then runs: `python scripts/setup.py`

This script does **everything** automatically.

```python
#!/usr/bin/env python3
"""
NTI Setup Script — One command to rule them all.
Run this ONCE after filling in .env to set up everything:
  - GitHub repository creation
  - Push all .env secrets to GitHub Secrets
  - Cloudflare Pages project creation and connection
  - Firebase project setup (Auth + Firestore)
  - Initial data bootstrap

Usage: python scripts/setup.py
Requirements: gh CLI installed, wrangler CLI installed, firebase CLI installed
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def load_env(env_file: str = ".env") -> dict:
    """Parse .env file into dict."""
    env = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                # Remove surrounding quotes
                value = value.strip().strip('"').strip("'")
                if key.strip() and value:
                    env[key.strip()] = value
    return env

def run(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run shell command and handle errors."""
    print(f"  → {cmd}")
    result = subprocess.run(
        cmd, shell=True,
        capture_output=capture,
        text=True
    )
    if check and result.returncode != 0:
        print(f"  ❌ Command failed: {cmd}")
        if result.stderr:
            print(f"  Error: {result.stderr}")
        sys.exit(1)
    return result

def push_secrets_to_github(env: dict, repo: str):
    """Push all env vars as GitHub Secrets using gh CLI --env-file flag."""
    print("\n📤 Pushing secrets to GitHub...")
    # Write a temp secrets file (only non-empty values)
    secrets_file = Path(".env.secrets.tmp")
    with open(secrets_file, "w") as f:
        for key, value in env.items():
            if value and not key.startswith("_"):
                f.write(f"{key}={value}\n")
    # Use gh CLI bulk secret set (fastest method)
    run(f"gh secret set --env-file {secrets_file} --repo {repo}")
    secrets_file.unlink()  # Delete temp file immediately
    print(f"  ✅ {len(env)} secrets pushed to GitHub")

def create_github_repo(env: dict) -> str:
    """Create public GitHub repository."""
    repo_name = env.get("GITHUB_REPO_NAME", "nifty-timing-index")
    username = env.get("GITHUB_USERNAME", "chirag127")
    repo = f"{username}/{repo_name}"
    print(f"\n🐙 Creating GitHub repository: {repo}")
    result = run(f"gh repo create {repo} --public --description 'Nifty Timing Index — Automated Indian market analysis with hourly blogs' --clone", check=False)
    if result.returncode != 0:
        print(f"  ℹ️  Repository may already exist, continuing...")
    print(f"  ✅ Repository ready: https://github.com/{repo}")
    return repo

def setup_cloudflare_pages(env: dict):
    """Create Cloudflare Pages project."""
    project_name = "nifty-timing-index"
    print(f"\n☁️  Setting up Cloudflare Pages: {project_name}")
    # Create Pages project
    run(f'wrangler pages project create {project_name} --production-branch main', check=False)
    print(f"  ✅ Cloudflare Pages project ready")

def setup_firebase(env: dict):
    """Initialize Firebase project with Auth and Firestore."""
    project_id = env.get("FIREBASE_PROJECT_ID", "nifty-timing-index")
    print(f"\n🔥 Setting up Firebase project: {project_id}")
    # Create Firebase project
    run(f"firebase projects:create {project_id} --display-name 'Nifty Timing Index'", check=False)
    # Enable Google Auth
    print("  → Enabling Google Auth (manual: go to Firebase Console → Auth → Google)")
    # Create Firestore database
    run(f"firebase firestore:databases:create --project {project_id} --location asia-south1", check=False)
    print(f"  ✅ Firebase ready: https://console.firebase.google.com/project/{project_id}")

def validate_required_vars(env: dict):
    """Check all required env vars are present."""
    required = [
        "GITHUB_USERNAME", "GITHUB_TOKEN",
        "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
        "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "ALERT_EMAIL_TO",
        "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID",
        "FIREBASE_PROJECT_ID", "FRED_API_KEY",
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        print(f"\n❌ Missing required env vars: {', '.join(missing)}")
        print("  Please fill in .env before running setup.")
        sys.exit(1)
    print(f"  ✅ All {len(required)} required env vars found")

def main():
    print("🇮🇳 Nifty Timing Index — Setup Script")
    print("=" * 50)

    # Load .env
    if not Path(".env").exists():
        print("❌ .env file not found. Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    env = load_env(".env")
    print(f"  ✅ Loaded {len(env)} env vars from .env")

    # Validate
    validate_required_vars(env)

    # Step 1: GitHub repo
    repo = create_github_repo(env)

    # Step 2: Push secrets
    push_secrets_to_github(env, repo)

    # Step 3: Cloudflare Pages
    setup_cloudflare_pages(env)

    # Step 4: Firebase
    setup_firebase(env)

    print("\n" + "=" * 50)
    print("✅ Setup complete! Here's what happens next:")
    print("  1. Push your code to GitHub (git push)")
    print("  2. GitHub Actions will run the first hourly signal")
    print("  3. Cloudflare Pages will deploy the website")
    print("  4. First blog post will appear in ~10 minutes")
    print("")
    print("  🔥 Firebase Console: https://console.firebase.google.com")
    print("  ☁️  Cloudflare Pages: https://dash.cloudflare.com")
    print("  🐙 GitHub Actions: https://github.com/{}/actions".format(repo))

if __name__ == "__main__":
    main()
```

---

## 18. Email Notification System (Gmail SMTP)

### 18.1 When Emails Are Sent

| Trigger | Email Type | Subject |
|---------|-----------|---------|
| Zone change (e.g. NEUTRAL → SELL) | Zone Change Alert | `⚠️ NTI ZONE CHANGE: NEUTRAL → SELL | Nifty: 24,180` |
| Score moves > 10 points in 1 hour | Big Move Alert | `📈 NTI BIG MOVE: Score 40 → 52 (+12 pts) | Watch closely` |
| Model unavailable (fallback active) | System Warning | `⚠️ NTI: ML model unavailable, using fallback score` |

Emails are NOT sent every hour (that would be 500+ emails/month even at 1/hour). Only on meaningful changes.

### 18.2 Gmail SMTP Configuration

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_gmail_alert(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = os.environ["GMAIL_ADDRESS"]
    msg["To"] = os.environ["ALERT_EMAIL_TO"]
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(
            os.environ["GMAIL_ADDRESS"],
            os.environ["GMAIL_APP_PASSWORD"],  # App Password, NOT account password
        )
        server.sendmail(
            os.environ["GMAIL_ADDRESS"],
            os.environ["ALERT_EMAIL_TO"],
            msg.as_string(),
        )
```

---

## 19. Backtesting System

### 19.1 Simulation Rules

```
Starting capital: ₹1,000,000 (₹10 Lakhs)
Signal → Nifty 50 Total Return Index (proxy for index fund performance)
Score ≤ 45 (BUY_LEAN or stronger): 100% equity (full long)
Score 46–55 (NEUTRAL): 50% equity, 50% cash (liquid fund @ 6% p.a.)
Score ≥ 56 (SELL_LEAN or stronger): 0% equity (flat, all cash)
Transaction costs: 0.1% per buy/sell
Slippage: 0.05% per trade
Risk-free rate: 6.5% p.a. (India RBI savings rate)
```

### 19.2 Metrics Computed

```python
BACKTEST_METRICS = {
    "cagr_pct",              # Compound Annual Growth Rate
    "sharpe_ratio",          # (Return - Risk-free) / StdDev
    "sortino_ratio",         # (Return - Risk-free) / Downside StdDev
    "max_drawdown_pct",      # Maximum peak-to-trough loss
    "calmar_ratio",          # CAGR / Max Drawdown
    "win_rate_pct",          # % of trades that were profitable
    "avg_trade_return_pct",  # Average return per trade
    "total_trades",          # Total number of buy/sell pairs
    "days_in_market_pct",    # % of days holding equity
    "vs_buy_hold_alpha_pct", # Outperformance vs simple buy & hold
}
```

---

## 20. Error Handling & Resilience

### 20.1 Scraper Failure Protocol

```python
def safe_scrape(scraper_fn, indicator_name: str, fallback_value=None):
    try:
        value = scraper_fn()
        if value is None:
            raise ValueError("Returned None")
        return value, "fresh"
    except Exception as e:
        # Log error
        log_error(indicator_name, str(e))
        # Use last known value from previous CSV row
        last_known = get_last_known_value(indicator_name)
        if last_known is not None:
            return last_known, "stale"
        # Use fallback (historical median)
        return fallback_value, "fallback"
```

### 20.2 Data Validation Ranges

```python
VALIDATION_RANGES = {
    "nifty_pe": (5, 60),
    "nifty_pb": (0.5, 8),
    "india_vix": (5, 90),
    "mmi": (0, 100),
    "pcr": (0.1, 5.0),
    "fii_cash_net": (-15000, 15000),   # Crores
    "us_10y_yield": (0, 10),
    "usd_inr": (60, 120),
    "brent_crude": (30, 200),
    "cpi_inflation": (-2, 15),
}
```

### 20.3 NSE Anti-Scraping Mitigation

NSE blocks automated requests. Mitigation:

```python
import requests, time

def create_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    # Hit homepage first to get cookies
    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(2)  # Respectful delay
    return session
```

---

## 21. Environment Configuration (.env.example — Complete)

```env
# ============================================================
# NIFTY TIMING INDEX — Environment Configuration
# ============================================================
# 1. Copy this file to .env
# 2. Fill in ALL values below (no value should be empty)
# 3. Run: python scripts/setup.py
# Everything else is automatic after that.
# NEVER commit .env to git (it's in .gitignore)
# ============================================================

# === GitHub ===
GITHUB_USERNAME=chirag127
GITHUB_TOKEN=ghp_your_personal_access_token_here
GITHUB_REPO_NAME=nifty-timing-index

# === LLM API (pick ONE provider — can change anytime by updating these 3 vars)
# Option A: Groq (FREE — 1,000 req/day, ultra-fast, recommended)
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
LLM_MAX_TOKENS=2000

# Option B: Google Gemini Flash (FREE — 15 RPM, swap in by uncommenting)
# LLM_API_KEY=AIza_your_gemini_api_key_here
# LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
# LLM_MODEL=gemini-2.0-flash

# Option C: Cerebras (FREE — 1M tokens/day, great for high volume)
# LLM_API_KEY=your_cerebras_api_key_here
# LLM_BASE_URL=https://api.cerebras.ai/v1
# LLM_MODEL=llama-3.1-70b

# Option D: OpenAI (PAID)
# LLM_API_KEY=sk-your_openai_key_here
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini

# Option E: Anthropic (PAID)
# LLM_API_KEY=sk-ant-your_anthropic_key_here
# LLM_BASE_URL=https://api.anthropic.com/v1
# LLM_MODEL=claude-haiku-4-5

# === Data APIs (all free) ===
FRED_API_KEY=your_fred_api_key_here
# Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html

FINNHUB_API_KEY=your_finnhub_free_api_key_here
# Get free key at: https://finnhub.io (60 calls/min free)

# === Email (Gmail SMTP — free 500/day) ===
GMAIL_ADDRESS=whyiswhen@gmail.com
GMAIL_APP_PASSWORD=your_16_char_google_app_password_here
# Get App Password: Google Account → Security → App Passwords
# (Requires 2-Step Verification to be enabled)
ALERT_EMAIL_TO=whyiswhen@gmail.com

# === Cloudflare Pages ===
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token_here
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id_here
# Get at: Cloudflare Dashboard → My Profile → API Tokens

# === Firebase ===
FIREBASE_PROJECT_ID=nifty-timing-index
FIREBASE_API_KEY=AIza_your_firebase_web_api_key_here
FIREBASE_AUTH_DOMAIN=nifty-timing-index.firebaseapp.com
FIREBASE_APP_ID=1:123456789:web:abcdef
# Server-side Firebase Admin SDK (JSON as base64 string)
FIREBASE_SERVICE_ACCOUNT_BASE64=base64_encoded_service_account_json_here
# Get service account: Firebase Console → Project Settings → Service Accounts

# === NTI Configuration ===
NTI_ENABLE_EMAIL=true
NTI_ENABLE_LLM=true
NTI_ENABLE_SCREENER=true
NTI_ENABLE_MODEL=true

# Stock screener hard limits (can change per user preference)
NTI_MAX_PE=20.0
NTI_MAX_PB=3.0
NTI_MIN_MARKET_CAP_CR=500
NTI_PSU_BOOST_SCORE=10.0

# MTF risk display (informational)
NTI_MTF_LEVERAGE=3.0
NTI_MTF_ANNUAL_INTEREST_RATE=0.12

# === Optional (leave empty if not using) ===
SENTRY_DSN=
# Error monitoring: https://sentry.io (free for personal use)
```

---

## 22. Testing Strategy

### 22.1 Test Files

| File | Tests | Priority |
|------|-------|----------|
| `tests/test_scrapers.py` | Each scraper returns valid data within expected ranges | P0 |
| `tests/test_indicators.py` | Normalization returns 0–100, edge cases (None, zero, extreme values) | P0 |
| `tests/test_model.py` | Training runs without error, inference returns 0–100, cold start fallback works | P0 |
| `tests/test_screener.py` | PE < 20 filter works, PB < 3 filter works, PSU boost applied correctly, market cap filter | P0 |
| `tests/test_blog_generator.py` | Blog markdown is valid, contains required sections, word count in range | P1 |
| `tests/test_changelog.py` | Changelog detects score changes, zone changes, stock additions/removals | P1 |
| `tests/test_storage.py` | CSV writes correctly, JSON valid, git commit succeeds | P1 |
| `tests/test_config.py` | All required env vars validated, defaults work | P0 |
| `tests/test_email.py` | Email sends on zone change, not on same-zone updates | P1 |

### 22.2 Key Test Cases (No Shortcuts)

```python
# test_screener.py
def test_pe_hard_filter():
    stock = {"symbol": "TEST", "pe": 20.5}
    assert not passes_hard_filter(stock), "PE=20.5 must FAIL (≥20 excluded)"

def test_pe_boundary():
    stock = {"symbol": "TEST", "pe": 19.99}
    assert passes_hard_filter(stock), "PE=19.99 must PASS"

def test_pb_hard_filter():
    stock = {"symbol": "TEST", "pe": 15, "pb": 3.1}
    assert not passes_hard_filter(stock), "PB=3.1 must FAIL"

def test_psu_boost():
    psu = {"symbol": "SBIN", "composite_score": 80}
    non_psu = {"symbol": "HDFC", "composite_score": 80}
    assert score_with_boost(psu) == 90, "PSU gets +10"
    assert score_with_boost(non_psu) == 80, "Non-PSU no boost"

def test_market_cap_filter():
    small = {"symbol": "TEST", "pe": 10, "pb": 1, "market_cap_cr": 499}
    assert not passes_hard_filter(small), "₹499 Cr must FAIL"
    ok = {"symbol": "TEST", "pe": 10, "pb": 1, "market_cap_cr": 500}
    assert passes_hard_filter(ok), "₹500 Cr must PASS"

# test_model.py
def test_score_range():
    score = run_inference(mock_indicators())
    assert 0 <= score <= 100, f"Score {score} out of range"

def test_cold_start_fallback():
    """When no model file exists, fallback must return a valid score."""
    score = run_fallback_inference(mock_indicators())
    assert 0 <= score <= 100

# test_indicators.py
def test_normalization_bounds():
    assert normalize_pe(5) == 0, "Very cheap PE = 0"
    assert normalize_pe(60) == 100, "Very expensive PE = 100"
    assert 0 <= normalize_pe(21) <= 100
```

---

## 23. Implementation Phases

### Phase 1 — Foundation (Weeks 1–2)

**Goal**: System running end-to-end with basic data collection, blog posting, no ML model yet (rule-based fallback).

- [ ] Repository setup with `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`
- [ ] `src/nti/config/settings.py` — load all env vars with defaults and validation
- [ ] `src/nti/config/psu_stocks.py` — PSU stock list from Nifty PSE + CPSE + PSU Bank
- [ ] `src/nti/config/holidays.py` — Indian market holiday calendar 2026
- [ ] `src/nti/scrapers/nse_indices.py` — PE, PB, VIX, dividend yield via `nsetools`
- [ ] `src/nti/scrapers/yahoo_finance.py` — yfinance: VIX, USD/INR, crude, S&P500, GIFT Nifty
- [ ] `src/nti/scrapers/fred_api.py` — US 10Y yield
- [ ] `src/nti/scrapers/tickertape_mmi.py` — Selenium headless MMI scrape
- [ ] `src/nti/indicators/normalizer.py` — normalize all indicators to 0–100
- [ ] `src/nti/model/fallback.py` — rule-based weighted average score (cold start)
- [ ] `src/nti/storage/csv_writer.py` — write signals to CSV
- [ ] `src/nti/storage/json_api.py` — update `data/api/latest.json`
- [ ] `src/nti/pipelines/hourly.py` — orchestrate full hourly run
- [ ] `.github/workflows/hourly-signal.yml` — GitHub Actions cron
- [ ] Basic Astro 6 website setup with placeholder dashboard
- [ ] `scripts/setup.py` — automated setup script
- [ ] Phase 1 tests: scrapers, normalizer, fallback model, CSV writer

**Deliverable**: Every hour, GitHub Actions collects 10 core indicators, computes a rule-based score, and pushes data. Website shows current score.

---

### Phase 2 — Full Indicators + News (Weeks 3–4)

**Goal**: All 30 indicators collected. LLM blog generation working. Website shows blogs.

- [ ] `src/nti/scrapers/nse_fii_dii.py` — FII/DII flows
- [ ] `src/nti/scrapers/nse_options.py` — PCR from NSE options chain
- [ ] `src/nti/scrapers/rbi_data.py` — repo rate scrape
- [ ] `src/nti/scrapers/mospi_data.py` — CPI inflation scrape
- [ ] `src/nti/scrapers/amfi_data.py` — SIP flows
- [ ] `src/nti/scrapers/cnn_fear_greed.py` — CNN F&G scrape
- [ ] `src/nti/scrapers/rss_news.py` — RSS feed aggregator
- [ ] `src/nti/indicators/composite.py` — Custom F&G composite, global overnight composite
- [ ] `src/nti/indicators/technical_display.py` — RSI, MACD (display only)
- [ ] `src/nti/llm/client.py` — multi-provider LLM client
- [ ] `src/nti/llm/news_analyzer.py` — RSS news → LLM → danger score
- [ ] `src/nti/llm/blog_generator.py` — full hourly blog markdown generation
- [ ] `src/nti/storage/blog_writer.py` — write blog .md to website content dir
- [ ] `src/nti/changelog/generator.py` — changelog diff generator
- [ ] Astro 6 content collections for blog
- [ ] Blog listing page `/blog/` with cards
- [ ] Individual blog page `/blog/[slug]/` with full post + changelog section
- [ ] Blog integrated into navigation, shown on homepage
- [ ] Phase 2 tests: all scrapers, news analyzer, blog generator, changelog

**Deliverable**: Every hour, a full blog post with score, indicators, news analysis, and changelog appears on the website.

---

### Phase 3 — ML Model + Full Website (Weeks 5–6)

**Goal**: ML stacked ensemble model running. Full website with all pages.

- [ ] `src/nti/model/trainer.py` — LightGBM + XGBoost + RF + Logistic meta-learner
- [ ] `src/nti/model/labeler.py` — generate training labels from historical Nifty returns
- [ ] `src/nti/model/predictor.py` — hourly inference with artifact download
- [ ] `src/nti/model/explainer.py` — SHAP values for top-3 drivers
- [ ] `.github/workflows/daily-retrain.yml` — daily retrain workflow
- [ ] `scripts/bootstrap_data.py` — seed 2 years of historical data from yfinance + NSE
- [ ] Historical NTI score chart page (`/history/`)
- [ ] Full indicator breakdown page (`/indicators/`)
- [ ] Backtest results page (`/backtest/`)
- [ ] Dashboard: big gauge, index scores, SHAP drivers section
- [ ] README.md live dashboard (NTI score badge, mini chart, top picks)
- [ ] Phase 3 tests: ML model training/inference, SHAP values, backtester

**Deliverable**: ML model runs daily retrain, hourly inference uses ML model, SHAP drivers shown in every blog post.

---

### Phase 4 — Stock Screener + Google Auth (Weeks 7–8)

**Goal**: Full stock screener running twice daily. Google login saving user preferences.

- [ ] `src/nti/screener/universe.py` — NSE EQUITY_L.csv download + filter
- [ ] `src/nti/screener/fundamentals.py` — yfinance fundamentals fetch (batched, rate-limited)
- [ ] `src/nti/screener/analyst_ratings.py` — Finnhub analyst ratings
- [ ] `src/nti/screener/filters.py` — all hard filters (PE < 20, PB < 3, etc.)
- [ ] `src/nti/screener/scorer.py` — composite score + PSU boost
- [ ] `.github/workflows/daily-screener-am.yml` + `daily-screener-pm.yml`
- [ ] `/screener/` page — full sortable table with filters
- [ ] `/stocks/[symbol]/` pages — individual stock detail
- [ ] Firebase Auth integration in Astro (Google login button)
- [ ] `/account/` page — SSR, reads Firestore user data
- [ ] Watchlist save/load from Firestore
- [ ] Custom PE/PB threshold saving in Firestore
- [ ] Dark/light theme toggle — persists to Firebase + localStorage
- [ ] Phase 4 tests: screener filters, PSU boost, auth flow, Firestore reads

**Deliverable**: Screener shows top value picks (PSU-preferred, PE < 20, PB < 3) twice daily. Users can log in and save preferences.

---

### Phase 5 — Polish & Production (Week 9+)

- [ ] Email alert system (Gmail SMTP, zone change + big move triggers)
- [ ] Full `scripts/setup.py` automation (tested end-to-end)
- [ ] Rate limiting and retry logic for all scrapers
- [ ] Comprehensive error logging to `data/errors/`
- [ ] Sentry integration (optional, free tier)
- [ ] Mobile-responsive design audit
- [ ] SEO audit: sitemap.xml, robots.txt, OG tags on all pages
- [ ] Performance: Lighthouse score > 90 on all pages
- [ ] All tests passing with > 80% coverage
- [ ] AGENTS.md for AI coding assistants
- [ ] Full README.md with setup guide, architecture, FAQ

---

## 24. Research Papers & Academic Foundation

| Paper | Year | Key Finding | Applied In NTI |
|-------|------|------------|----------------|
| "High-precision forecasting of Indian stock market indices using weighted ensemble of hyperparameter-tuned LightGBM models" — *Taylor & Francis* | Oct 2025 | LightGBM best for Indian stock index tabular prediction | Primary ML base learner |
| "Stock Price Prediction Using a Stacked Heterogeneous Ensemble" — *MDPI Finance* | Oct 2025 | XGBoost meta-learner over RF+LightGBM base achieves R²=0.97–0.99 | Stacking architecture |
| "Predicting Stock Returns: A Hybrid Approach with LightGBM, XGBoost" — *Atlantis Press* | Feb 2025 | Combined LightGBM + XGBoost superior to single models | Model combination |
| "Ensemble Learning: XGBoost and LightGBM Stacking, 10–20% accuracy gain" — *MLPerf 2025* | Sep 2025 | Stacking improves accuracy 10–20% on tabular data | Architecture validation |
| "tabular data: DNNs not satisfactory vs tree-based models" — *PeerJ CS* | 2023 | Tree models beat LSTM/DNN on tabular financial data | Justifies no LSTM |
| Baker & Wurgler "Investor Sentiment and the Cross-Section of Stock Returns" — *JoF* | 2006 | Sentiment (fear/greed) predicts returns — contrarian signal | MMI contrarian signal |
| Basu "Investment Performance in Relation to P/E Ratios" — *JoF* | 1977 | Low PE stocks outperform high PE stocks long-term | Hard PE < 20 filter |
| Fama & French "The Cross-Section of Expected Stock Returns" — *JoF* | 1992 | Low PB (HML factor) generates excess returns | Hard PB < 3 filter |
| Jegadeesh & Titman "Returns to Buying Winners and Selling Losers" — *JoF* | 1993 | Momentum effect: past winners continue to outperform 3–12 months | PSU + momentum display |
| Asness, Moskowitz & Pedersen "Value and Momentum Everywhere" — *JoF* | 2013 | Value + momentum combination universally profitable | Screener scoring |
| "Market Timing Strategies in Emerging Markets: Evidence from India" — Kumar & Goyal | 2022 | Sentiment-based timing works better in India than developed markets | Validates MMI weight |
| Brunnermeier & Pedersen "Market Liquidity and Funding Liquidity" — *RFS* | 2009 | Margin calls create cascading liquidation — leverage amplifies risk | MTF risk framework |

---

## 25. Design System & Frontend Aesthetic

### 25.1 Visual Direction

**Concept**: *Precision terminal meets Indian market intelligence*

Inspired by Bloomberg terminal's information density but with modern Indian fintech sensibility. Not a generic finance dashboard — it feels like a personal command center built for someone who takes value investing seriously.

**Aesthetic**: Dark navy/charcoal base with high-contrast data, gold/amber for buy signals, crimson for sell signals, cool emerald for neutral. Monospace fonts for numbers, clean sans-serif for prose. Dense but not cluttered.

### 25.2 Design Tokens

```css
/* website/src/styles/design-tokens.css */
:root {
  /* Color Palette */
  --color-bg-primary: #0a0e1a;       /* Deep navy black */
  --color-bg-secondary: #111827;     /* Slightly lighter navy */
  --color-bg-card: #1a2035;          /* Card backgrounds */
  --color-bg-elevated: #1f2d40;      /* Elevated surfaces */
  --color-border: #2a3f5f;           /* Subtle borders */

  /* Signal Colors (the visual language of NTI) */
  --color-extreme-buy: #00ff88;      /* Bright emerald — extreme buy */
  --color-strong-buy: #34d399;       /* Emerald */
  --color-buy-lean: #6ee7b7;         /* Light mint */
  --color-neutral: #94a3b8;          /* Slate gray */
  --color-sell-lean: #fbbf24;        /* Amber */
  --color-strong-sell: #f97316;      /* Orange */
  --color-extreme-sell: #ef4444;     /* Red */

  /* Text */
  --color-text-primary: #f1f5f9;     /* Near-white */
  --color-text-secondary: #94a3b8;   /* Slate */
  --color-text-muted: #64748b;       /* Muted slate */
  --color-text-accent: #f59e0b;      /* Gold accent */

  /* PSU Badge */
  --color-psu-bg: #1e3a5f;
  --color-psu-text: #60a5fa;
  --color-psu-border: #3b82f6;

  /* Typography */
  --font-display: 'DM Serif Display', Georgia, serif;    /* Headlines */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace; /* Numbers, scores */
  --font-body: 'Inter', system-ui, sans-serif;           /* Body text */
  /* Note: DM Serif Display gives editorial gravitas to headlines.
     JetBrains Mono for all numbers makes them feel like terminal readouts.
     This pairing avoids the generic "AI dashboard" look. */

  /* Spacing */
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-md: 1rem;
  --space-lg: 1.5rem;
  --space-xl: 2rem;
  --space-2xl: 3rem;

  /* Border radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
}

/* Light theme overrides */
[data-theme="light"] {
  --color-bg-primary: #f8fafc;
  --color-bg-secondary: #f1f5f9;
  --color-bg-card: #ffffff;
  --color-bg-elevated: #e2e8f0;
  --color-border: #cbd5e1;
  --color-text-primary: #0f172a;
  --color-text-secondary: #475569;
  --color-text-muted: #94a3b8;
  /* Signal colors stay the same for consistency */
}
```

### 25.3 NTI Gauge Component

```tsx
// website/src/components/NTIGauge.tsx
// Semi-circular gauge, 0–100, zone-colored arc
// Number displayed large in monospace font
// Zone label below in uppercase tracking

// Visual: Think of a speedometer where:
// 0–45 = green gradient arc (buy zones)
// 46–55 = gray arc (neutral)
// 56–100 = red gradient arc (sell zones)
// The needle points to current score
// Score number is 72px JetBrains Mono in the center
```

### 25.4 PSU Stock Badge

```tsx
// PSU stocks always show a blue badge: [★ PSU]
// Color: --color-psu-bg background, --color-psu-text
// This visually highlights government stocks in screener table
```

### 25.5 Responsive Design

- Mobile-first: all pages functional at 375px width
- Dashboard: single column on mobile, 2-column on tablet, 3-column on desktop
- Screener table: horizontal scroll on mobile with sticky first column (symbol)
- Blog: full width, max 720px content width on desktop
- Charts: responsive container with touch-enabled zoom on mobile

---

*This plan is complete. Every decision confirmed with user. Zero assumptions made. Implementation can proceed directly from this document.*

*Document ends. Total: 25 sections covering all aspects of the Nifty Timing Index system.*

Make the implementation plan proper and with the utmost Quality and at most everything. Make the implementation plan proper and at most quality And if you want to change anything, change it. It will be Website with the very many things you can search the web with many things. I want to make the at most quality machine learning For their prediction of the everything, and make sure that implementation plan only contain the necessary things. The empty mentation plan will be very, very, very big. It will be a very massive implementation plan and will contain everything about the website and the Github actions and it will be a static website. I won't host anything on the server And it will have all the steps in written in And I want to make everything clear, crystal clear in the implementation plan. Make sure that everything is written properly in the implementation plan. And make sure that everything is done. Route to the modern Practices in the In the world and modern practices are followed in the world And it will be Using the Go youtube search Google search Bing search and any kind of web search. Together, the information. And it will also gather all the free resources. It will also use the free resources I prefer low PB, Blue Page, stocks, butter. It was a personal preference. The blogs will include Content for all of them. It will be a very long prompt blogs. The blogs will be very long, and there will be proper rate limiting in the All of the Apis because the Apis have rate limiting. So we have to make sure that every rate limits are being followed and correctly Being honored properly in Implementation plan The implementation plan have to be well researched and made. Proper and the website will include proper pages, legal pages What can be the legal pages which can be included in the implementation plan will be included in the implementation plan Every kind of page will be included and every kind of Page of which the website should contain will be included Everything will be included in the implementation plan. Please include and search the web for everything. For every AI provider Free AI provider Every


# NTI Plan Addendum — Web Research Findings (2026-04-24)

> Append these sections to plan.md. Contains ONLY information
> not already present in the existing 25-section plan.

---

## A. Updated Technology Versions (Verified 2026-04-24)

The existing plan references some outdated versions. Correct versions:

| Technology | Plan Says | Actual Latest (Apr 2026) |
|-----------|-----------|--------------------------|
| Astro | 6.x | 6.x ✅ (released 2026-03-10) |
| Node.js | 22+ | 22.12.0+ required by Astro 6 |
| Tailwind CSS | 4.x | 4.x ✅ but **no config file** — uses `@tailwindcss/vite` plugin + CSS `@import "tailwindcss"` |
| LightGBM | "latest" | **4.6.0** (released 2025-02-15) |
| XGBoost | "≥ 2.0.3" | **3.2.0** (released 2026-02-10) |
| SHAP | "latest" | **0.51.0** (released 2026-03-04) — check XGBoost 3.x compat |
| yfinance | "latest" | **1.3.0** (released 2026-04-16) |
| LangGraph | not in plan | **1.1.x** — use `pip install -U langgraph` |
| Langfuse SDK | not in plan | **4.5.0** (released 2026-04-21) — v4 is a major rewrite |
| Firebase JS SDK | "10+" | **12.12.1** (released 2026-04-20) |
| Chart.js | "latest" | **4.5.1** |
| Ruff | "latest" | latest ✅ |
| `actions/checkout` | v4 | **v6** (needed for Node 24 runner) |
| `actions/setup-node` | v4 | **v6** (needed for Node 24 runner) |
| `astral-sh/setup-uv` | v4 | **v8** (v8.1.0 — immutable releases now) |
| `cloudflare/wrangler-action` | v3 | v3 ✅ |

### Critical: SHAP + XGBoost 3.x Compatibility

There are reported issues between `shap.TreeExplainer` and XGBoost >= 3.0.0 (base_score storage changes). Pin SHAP to 0.51.0 and test. If broken, pin XGBoost to 2.1.x as fallback.

---

## B. Astro 6 Breaking Changes (Not in Plan)

The plan's Astro content collections config uses the **old Astro 5 syntax**. Astro 6 requires:

### B.1 Content Collections — New Syntax

**Old (plan has):**
```typescript
// website/src/content/config.ts  ← WRONG filename
import { defineCollection, z } from "astro:content";
const blog = defineCollection({
  type: "content",  // ← REMOVED in Astro 6
  schema: z.object({ ... }),
});
```

**New (Astro 6 correct):**
```typescript
// website/src/content.config.ts  ← NEW filename
import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const blog = defineCollection({
  loader: glob({
    base: "./src/content/blog",
    pattern: "**/*.{md,mdx}",
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    slug: z.string(),
    publishedAt: z.coerce.date(),
    ntiScore: z.number(),
    ntiZone: z.string(),
    ntiZonePrev: z.string(),
    zoneChanged: z.boolean(),
    confidence: z.number(),
    nifty50Price: z.number(),
    topDrivers: z.array(z.string()),
    topStocks: z.array(z.string()),
    blogType: z.enum([
      "market_open", "mid_session",
      "market_close", "post_market", "overnight",
    ]),
  }),
});

export const collections = { blog };
```

### B.2 Other Astro 6 Breaking Changes

- `Astro.glob()` is **removed** → use `getCollection()` or `import.meta.glob()`
- `<ViewTransitions />` → renamed to `<ClientRouter />`
- Astro 6 uses **Vite 7, Shiki 4, Zod 4** internally
- Run `npx astro sync` if editor shows `astro:content` errors
- Blog pages use `render()` not `Content`:
```astro
---
import { getCollection, render } from "astro:content";
const { post } = Astro.props;
const { Content } = await render(post);
---
<Content />
```

### B.3 Tailwind CSS 4 — No Config File

Plan references `tailwind.config.mjs` — this **does not exist** in TW4.

**Correct Astro 6 + Tailwind 4 setup:**
```bash
npm install tailwindcss @tailwindcss/vite
```

```javascript
// astro.config.mjs
import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
  },
});
```

```css
/* src/styles/global.css */
@import "tailwindcss";

/* Custom design tokens via @theme directive (replaces tailwind.config.js) */
@theme {
  --color-bg-primary: #0a0e1a;
  --color-bg-secondary: #111827;
  --color-extreme-buy: #00ff88;
  --color-strong-sell: #f97316;
  --color-extreme-sell: #ef4444;
  --font-display: "DM Serif Display", Georgia, serif;
  --font-mono: "JetBrains Mono", monospace;
  --font-body: "Inter", system-ui, sans-serif;
}
```

---

## C. LangGraph Multi-Model Fusion Architecture (Replaces Section 7.4)

The plan's Section 7.4 shows a simple single-provider `OpenAI()` client. The `.env` file defines 11 LLM providers with `FUSION_MODE=all`. This requires a **LangGraph** orchestration layer.

### C.1 Fusion Architecture

```python
# src/nti/llm/fusion.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class BlogState(TypedDict):
    topic: str
    market_data: str
    drafts: Annotated[list, operator.add]  # reducer: merge parallel drafts
    critiques: Annotated[list, operator.add]
    final_blog: str

# Fan-out: call all enabled LLMs in parallel
# Each returns {"drafts": [{"provider": "groq", "content": "..."}]}
# Fan-in: reducer merges all drafts into one list
# Critique: each model reviews other drafts
# Synthesize: FUSION_SYNTHESIZER model produces final blog

graph = StateGraph(BlogState)
graph.add_node("research", research_node)     # RSS + search
graph.add_node("draft", draft_node)           # Fan-out to all LLMs
graph.add_node("critique", critique_node)     # Models critique each other
graph.add_node("synthesize", synthesize_node) # Best model merges
graph.set_entry_point("research")
graph.add_edge("research", "draft")
graph.add_edge("draft", "critique")
graph.add_edge("critique", "synthesize")
graph.add_edge("synthesize", END)
app = graph.compile()
```

### C.2 Langfuse Integration for Tracing

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler()
result = app.invoke(
    {"topic": "NTI hourly update", "market_data": data_json},
    config={"callbacks": [langfuse_handler]},
)
```

### C.3 Rate Limit Config Per Provider

```python
PROVIDER_RATE_LIMITS = {
    "groq":       {"rpm": 30, "rpd": 1000},
    "gemini":     {"rpm": 15, "rpd": 600},
    "cerebras":   {"rpm": 30, "rpd": 14400},
    "mistral":    {"rpm": 60, "rpd": None, "tpm": 1_000_000_000},
    "nvidia":     {"rpm": 40, "rpd": None},
    "openrouter": {"rpm": 20, "rpd": 1000},
    "cohere":     {"rpm": 20, "rpd": None},
    "huggingface":{"rpm": 10, "rpd": None},
    "github":     {"rpm": 15, "rpd": 150},
    "cloudflare": {"rpm": None, "rpd": None, "neurons": 10000},
    "sambanova":  {"rpm": 10, "rpd": None},
}
```

---

## D. Additional Free LLM Providers (Not in Plan)

Plan lists 6 providers. Verified additional free providers:

| Provider | Free Tier | Base URL | Model |
|----------|-----------|----------|-------|
| **GitHub Models** | 50–150 RPD | `https://models.github.ai/inference` | `gpt-4o`, `DeepSeek-R1` |
| **Mistral AI** | 1B tokens/month | `https://api.mistral.ai/v1` | `mistral-large-latest` |
| **Cerebras** | 30 RPM, 14400 RPD | `https://api.cerebras.ai/v1` | `llama-3.3-70b` |
| **NVIDIA NIM** | 40 RPM | `https://integrate.api.nvidia.com/v1` | `meta/llama-3.1-405b-instruct` |
| **Cohere** | Free tier | `https://api.cohere.com/v2` | `command-r-plus` (native SDK) |
| **HuggingFace** | Free inference | `https://api-inference.huggingface.co/v1` | Various |

All use OpenAI-compatible endpoints except Cohere (use `langchain-cohere`).

---

## E. Search API Integration (Not in Plan)

The plan has no web search integration. The `.env` defines 6 search providers:

| Provider | Free Tier | Key Needed? |
|----------|-----------|-------------|
| **Serper.dev** | 2,500 queries (no CC) | Yes |
| **Google CSE** | 100 queries/day | Yes |
| **Tavily** | 1,000 credits/month | Yes |
| **Brave Search** | 2,000 queries/month | Yes |
| **DuckDuckGo** | Unlimited | No |
| **SearXNG** | Unlimited (self-hosted) | No |

### Search Integration for Blog Enrichment

```python
# src/nti/search/aggregator.py
# Try providers in order, fall back on failure
SEARCH_PRIORITY = [
    "serper", "brave", "tavily",
    "google_cse", "duckduckgo", "searxng",
]

async def search_market_context(query: str) -> list[dict]:
    for provider in SEARCH_PRIORITY:
        if not is_enabled(provider):
            continue
        try:
            return await search_with_provider(provider, query)
        except (RateLimitError, TimeoutError):
            continue
    return []
```

---

## F. News API Providers (Beyond RSS — Not in Plan)

Plan only has RSS feeds. The `.env` defines 6 additional news APIs:

| Provider | Free Tier | Best For |
|----------|-----------|----------|
| **NewsAPI.org** | 200 req/day | Global + India headlines |
| **GNews** | 100 req/day | Compact news summaries |
| **CurrentsAPI** | 200 req/day | Real-time news |
| **MediaStack** | 100 req/month | Multi-source aggregation |
| **TheNewsAPI** | Limited free | Categorized news |
| **World News API** | 50 points/day | Global geopolitical |
| **Alpha Vantage News** | Free key | AI sentiment scores |
| **Marketaux** | Free tier | Stock-specific news + sentiment |

---

## G. Multiple Email Providers (Not in Plan)

Plan only covers Gmail SMTP. The `.env` supports 4 email providers with fallback:

| Provider | Free Tier | Method |
|----------|-----------|--------|
| **Gmail SMTP** | 500 emails/day | SMTP (port 587 TLS) |
| **Resend** | 100 emails/day | REST API |
| **Brevo** | 300 emails/day | REST API |
| **SendGrid** | 100 emails/day | REST API |

Implement failover: try Gmail first, if it fails try Resend, then Brevo, then SendGrid.

---

## H. Lightweight Charts (TradingView) — Replace Chart.js for Financial

Plan uses Chart.js for all charts. For financial time-series, **Lightweight Charts by TradingView** is the gold standard:

```bash
npm install lightweight-charts
```

```tsx
// website/src/components/NTIChart.tsx
import { createChart } from "lightweight-charts";
import { useEffect, useRef } from "react";

export default function NTIChart({ data }) {
  const ref = useRef(null);
  useEffect(() => {
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth,
      height: 400,
      layout: {
        background: { color: "#0a0e1a" },
        textColor: "#f1f5f9",
      },
      grid: {
        vertLines: { color: "#1a2035" },
        horzLines: { color: "#1a2035" },
      },
    });
    const series = chart.addLineSeries({
      color: "#00ff88",
      lineWidth: 2,
    });
    series.setData(data);
    const handleResize = () => {
      chart.applyOptions({
        width: ref.current.clientWidth,
      });
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data]);
  return <div ref={ref} />;
}
```

Use **Chart.js** only for the NTI gauge (semi-circular doughnut). Use **Lightweight Charts** for all time-series: NTI score history, Nifty price, backtest equity curve.

---

## I. Legal Pages & SEBI Compliance (Not in Plan)

### I.1 Required Legal Pages

| Page | Route | Purpose |
|------|-------|---------|
| **Disclaimer** | `/disclaimer` | SEBI compliance — not investment advice |
| **Privacy Policy** | `/privacy` | DPDP Act 2023 compliance |
| **Terms of Service** | `/terms` | Usage rules |
| **Cookie Policy** | `/cookies` | AdSense requirement |
| **About Us** | `/about` | AdSense requirement |
| **Contact Us** | `/contact` | AdSense + trust requirement |

### I.2 SEBI Disclaimer Requirements (Critical)

Every page footer AND every blog post MUST include:

```
DISCLAIMER: This website and its content are for informational
and educational purposes only. Nothing on this website constitutes
investment advice, financial advice, trading advice, or a
recommendation to buy, sell, or hold any securities.

The operator of this website is NOT a SEBI-registered Investment
Adviser (IA) or Research Analyst (RA). All information is
generated by automated AI systems and should NOT be treated as
professional financial guidance.

Investment in securities market is subject to market risks.
Read all related documents carefully before investing.
Past performance is not indicative of future results.
No returns are guaranteed.

Always consult a SEBI-registered financial advisor before making
investment decisions. Conduct your own due diligence.
```

### I.3 SEBI AI Accountability (2026 Rule)

SEBI has clarified: **using AI does not reduce the operator's responsibility**. The website must clearly state that AI generates the content and the operator takes no liability for AI outputs.

### I.4 India DPDP Act 2023 — Privacy Policy Must Include

- What personal data is collected (email, Google profile via Firebase Auth)
- Purpose of data collection (personalization, alerts)
- Data storage location (Firebase Firestore — Google Cloud asia-south1)
- User rights: access, correction, erasure of their data
- Contact details for data-related queries
- Cookie usage disclosure
- Third-party services used (Firebase, Google Analytics if any)

### I.5 Accessibility (WCAG 2.1)

SEBI now requires regulated digital platforms to meet WCAG 2.1 Level AA. While NTI is not a regulated entity, following this ensures broader compliance:
- Alt text on all images/charts
- Keyboard-navigable interface
- Color contrast ratios ≥ 4.5:1 for text
- Screen reader compatible semantic HTML

---

## J. Google AdSense Readiness (Not in Plan)

### J.1 Requirements Checklist

- [ ] **Custom domain** (NOT `.pages.dev` — AdSense rejects subdomains)
- [ ] **20–30 quality articles** before applying (blogs accumulate this in ~2 days)
- [ ] **Privacy Policy** page accessible from footer
- [ ] **About Us** page with clear site purpose
- [ ] **Contact Us** page with email
- [ ] **Terms of Service** page
- [ ] **Cookie consent banner** (required for AdSense)
- [ ] **ads.txt** file at `/ads.txt` with AdSense publisher ID
- [ ] **Google Search Console** connected + sitemap submitted
- [ ] **No "under construction" pages** — all routes must return content
- [ ] **Mobile responsive** — all pages
- [ ] **Fast loading** — Lighthouse > 90

### J.2 ads.txt Setup

```
# website/public/ads.txt
google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0
```

### J.3 Cookie Consent Banner

Add a cookie consent component that:
- Shows on first visit
- Blocks AdSense scripts until user accepts
- Stores consent in localStorage + Firebase (if logged in)
- Links to `/cookies` page

---

## K. Cloudflare Domain & Email Routing (Not in Plan)

### K.1 Email Routing Setup

When Cloudflare API keys are present in `.env`, setup.py should:

1. Enable Email Routing on the domain
2. Add MX, SPF, DKIM DNS records automatically
3. Create forwarding rules:
   - `hi@nti.oriz.in` → `whyiswhen@gmail.com`
   - `support@nti.oriz.in` → `whyiswhen@gmail.com`
4. Verify destination email

### K.2 DNS Records for Email

```python
# Automated by setup.py via Cloudflare API
dns_records = [
    {"type": "MX", "name": "@", "content": "route1.mx.cloudflare.net", "priority": 69},
    {"type": "MX", "name": "@", "content": "route2.mx.cloudflare.net", "priority": 12},
    {"type": "MX", "name": "@", "content": "route3.mx.cloudflare.net", "priority": 41},
    {"type": "TXT", "name": "@", "content": "v=spf1 include:_spf.mx.cloudflare.net ~all"},
]
```

---

## L. Rate Limiting Strategy (Not Detailed in Plan)

### L.1 yfinance Rate Limiting

yfinance has NO official rate limit but Yahoo blocks aggressive requests:

```python
# src/nti/scrapers/rate_limiter.py
import time
import asyncio

YFINANCE_DELAY = 2.0  # seconds between individual ticker calls
YFINANCE_BATCH_SIZE = 50  # use yf.download() for batches

def fetch_stock_data_batched(symbols: list[str]):
    """Use yf.download() for batches — much more efficient."""
    for i in range(0, len(symbols), YFINANCE_BATCH_SIZE):
        batch = symbols[i:i + YFINANCE_BATCH_SIZE]
        data = yf.download(
            batch, period="5d", group_by="ticker",
            threads=True,
        )
        yield data
        time.sleep(YFINANCE_DELAY)
```

### L.2 NSE Session Management

NSE blocks automated requests aggressively. Always:

```python
def create_nse_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    # MUST hit homepage first to get cookies
    session.get("https://www.nseindia.com", timeout=15)
    time.sleep(2)  # Respectful delay
    return session
```

### L.3 LLM Provider Rate Limits (Verified Apr 2026)

| Provider | RPM | RPD | TPM | Notes |
|----------|-----|-----|-----|-------|
| Groq | 30 | 1,000 | 15,000 | Ultra-fast LPU |
| Gemini | 10–15 | 250–1,000 | varies | Large context window |
| Cerebras | 30 | 14,400 | — | Most generous daily |
| Mistral | 60 | — | 1B/month | Very generous |
| NVIDIA NIM | 40 | — | — | High perf open-source |
| OpenRouter | 20 | 1000 | — | Free models only |
| GitHub Models | 10–15 | 50–150 | 8K in/4K out | Prototyping only |
| Cohere | 20 | — | — | Native SDK recommended |
| HuggingFace | 10 | — | — | Varies by model |

### L.4 Global Rate Limiter Implementation

```python
# src/nti/utils/rate_limiter.py
import asyncio
from collections import defaultdict
from time import monotonic

class TokenBucketLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.tokens = rpm
        self.last_refill = monotonic()

    async def acquire(self):
        now = monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.rpm,
            self.tokens + elapsed * (self.rpm / 60),
        )
        self.last_refill = now
        if self.tokens < 1:
            wait = (1 - self.tokens) / (self.rpm / 60)
            await asyncio.sleep(wait)
            self.tokens = 0
        else:
            self.tokens -= 1

LIMITERS: dict[str, TokenBucketLimiter] = {}

def get_limiter(provider: str, rpm: int):
    if provider not in LIMITERS:
        LIMITERS[provider] = TokenBucketLimiter(rpm)
    return LIMITERS[provider]
```

---

## M. Walk-Forward Validation (Correction to Plan Section 5)

Plan uses 5-fold cross-validation for ML training. This is **incorrect for time-series** data (causes look-ahead bias). Use walk-forward (expanding window) validation:

```python
from sklearn.model_selection import TimeSeriesSplit

# NOT KFold — use TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train on past, validate on future — no data leakage
```

---

## N. NSE Data Library Warning (Not in Plan)

Web research confirms: **nsetools, jugaad-data, pynse are all fragile** as of April 2026. NSE aggressively blocks scrapers.

**Recommended approach:**
1. **Primary**: `yfinance` (v1.3.0) for fundamentals (PE, PB, market cap, ROE)
2. **Secondary**: Direct NSE session scraping with cookie management (see L.2 above)
3. **Fallback**: If scraper fails, use last known value from CSV
4. **Consider**: AngelOne SmartAPI or Fyers API (free with demat account) for real-time data

---

## O. Directory Structure Additions (Missing from Plan Section 14)

Add these to the existing directory tree:

```
src/nti/
│   ├── llm/
│   │   ├── fusion.py             # LangGraph multi-model fusion orchestration
│   │   ├── providers.py          # Load enabled LLM providers from .env
│   │   └── rate_limiter.py       # Per-provider rate limiting
│   │
│   ├── search/
│   │   ├── aggregator.py         # Multi-provider search with fallback
│   │   ├── serper.py             # Serper.dev Google search
│   │   ├── brave.py              # Brave Search API
│   │   ├── duckduckgo.py         # DuckDuckGo (no key)
│   │   └── tavily.py             # Tavily search
│   │
│   ├── news/
│   │   ├── aggregator.py         # Multi-source news aggregation
│   │   ├── rss.py                # RSS feeds (existing)
│   │   ├── newsapi.py            # NewsAPI.org
│   │   ├── gnews.py              # GNews API
│   │   └── marketaux.py          # Marketaux (stock-specific + sentiment)
│   │
│   ├── notifications/
│   │   ├── email_router.py       # Multi-provider email with fallback
│   │   ├── gmail.py              # Gmail SMTP
│   │   ├── resend.py             # Resend API
│   │   ├── brevo.py              # Brevo API
│   │   └── sendgrid.py           # SendGrid API
│   │
│   └── utils/
│       ├── rate_limiter.py       # Token bucket rate limiter
│       └── retry.py              # Exponential backoff retry logic

website/src/
│   ├── pages/
│   │   ├── disclaimer.astro      # SEBI disclaimer
│   │   ├── privacy.astro         # DPDP Act privacy policy
│   │   ├── terms.astro           # Terms of service
│   │   ├── cookies.astro         # Cookie policy
│   │   ├── about.astro           # About page (AdSense required)
│   │   └── contact.astro         # Contact page (AdSense required)
│   │
│   ├── components/
│   │   ├── CookieConsent.tsx     # Cookie consent banner
│   │   ├── NTILineChart.tsx      # Lightweight Charts time-series
│   │   └── Footer.astro          # Footer with legal links + disclaimer
│   │
│   └── content.config.ts         # ← RENAMED from content/config.ts (Astro 6)

website/public/
│   ├── ads.txt                   # Google AdSense authorization
│   └── _headers                  # Cloudflare Pages security headers
```

---

## P. GitHub Actions Version Updates (Corrections)

All workflows in plan Section 16 need these version bumps:

```yaml
# BEFORE (plan has):
- uses: actions/checkout@v4
- uses: actions/setup-node@v4
- uses: astral-sh/setup-uv@v4

# AFTER (correct for Apr 2026):
- uses: actions/checkout@v6
  with:
    fetch-depth: 0
- uses: actions/setup-node@v6
  with:
    node-version: "22"
- uses: astral-sh/setup-uv@v8
  with:
    version: "latest"
```

---

## Q. pyproject.toml Version Corrections

Current pyproject.toml has outdated version pins:

```toml
# CORRECTIONS needed:
dependencies = [
    "lightgbm>=4.6",      # was >=4.5
    "xgboost>=3.2",        # was >=2.1
    "shap>=0.51",          # was >=0.46
    "yfinance>=1.3",       # was >=0.2
    "langgraph>=1.1",      # was >=0.4
    "langfuse>=4.5",       # was >=2.45 (v4 is a major rewrite)
]
```

---

*End of addendum. Append relevant sections to plan.md.*
