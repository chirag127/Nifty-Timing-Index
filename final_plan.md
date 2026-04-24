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
