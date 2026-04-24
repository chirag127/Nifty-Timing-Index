"""Multi-Provider Search Client with Fallback Chain.

Supports 6+ free search APIs. Tries providers in order, falling back to the next on failure.
All providers are configurable via .env with enable/disable toggles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from nti.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    source: str  # Which provider returned this result


class MultiSearchClient:
    """Search client that tries multiple providers with fallback.

    Usage:
        client = MultiSearchClient()
        results = client.search("Nifty 50 PE ratio today")
        for r in results:
            print(f"{r.title}: {r.snippet}")
    """

    def __init__(self) -> None:
        self.timeout = 15.0

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search using all enabled providers with fallback.

        Tries each enabled provider in order. Returns results from the
        first successful provider, or empty list if all fail.
        """
        if not settings.enable_search:
            logger.info("Search disabled via NTI_ENABLE_SEARCH=false")
            return []

        for provider in settings.get_enabled_search_providers():
            try:
                results = self._search_provider(provider, query, max_results)
                if results:
                    logger.info(f"Search: got {len(results)} results from {provider.name}")
                    return results
            except Exception as e:
                logger.warning(f"Search provider {provider.name} failed: {e}. Trying next...")
                continue

        logger.warning("All search providers failed")
        return []

    def _search_provider(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search using a specific provider."""
        match provider.name:
            case "serper":
                return self._search_serper(provider, query, max_results)
            case "google_cse":
                return self._search_google_cse(provider, query, max_results)
            case "tavily":
                return self._search_tavily(provider, query, max_results)
            case "brave":
                return self._search_brave(provider, query, max_results)
            case "duckduckgo":
                return self._search_duckduckgo(provider, query, max_results)
            case "searxng":
                return self._search_searxng(provider, query, max_results)
            case _:
                logger.warning(f"Unknown search provider: {provider.name}")
                return []

    def _search_serper(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search via Serper.dev (Google SERP API)."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{provider.base_url}/search",
                headers={"X-API-KEY": provider.api_key},
                json={"q": query, "num": max_results},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="serper",
            ))
        return results

    def _search_google_cse(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search via Google Custom Search Engine."""
        params = {
            "key": provider.api_key,
            "cx": provider.extra.get("cx", ""),
            "q": query,
            "num": min(max_results, 10),
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(provider.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("items", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google_cse",
            ))
        return results

    def _search_tavily(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search via Tavily API."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{provider.base_url}/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": provider.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="tavily",
            ))
        return results

    def _search_brave(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search via Brave Search API."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{provider.base_url}/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": provider.api_key},
                params={"q": query, "count": max_results},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="brave",
            ))
        return results

    def _search_duckduckgo(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search via DuckDuckGo Instant Answer API (limited but free, no key needed)."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{provider.base_url}/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        # DuckDuckGo instant answers
        if data.get("AbstractText"):
            results.append(SearchResult(
                title=data.get("Heading", query),
                url=data.get("AbstractURL", ""),
                snippet=data.get("AbstractText", ""),
                source="duckduckgo",
            ))
        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results - 1]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(SearchResult(
                    title=topic.get("Text", "")[:80],
                    url=topic.get("FirstURL", ""),
                    snippet=topic.get("Text", ""),
                    source="duckduckgo",
                ))
        return results[:max_results]

    def _search_searxng(self, provider, query: str, max_results: int) -> list[SearchResult]:
        """Search via SearXNG instance (self-hosted or public)."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{provider.base_url}/search",
                params={"q": query, "format": "json", "categories": "general"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="searxng",
            ))
        return results


# Singleton
search_client = MultiSearchClient()
