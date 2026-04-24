"""Multi-Provider News Client with Fallback Chain.

Supports RSS feeds (always free) + 6+ free news APIs.
Tries providers in order, falling back to the next on failure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

import httpx

from nti.config.settings import settings

logger = logging.getLogger(__name__)


# RSS Feed URLs — imported from the canonical source in rss_news.py
# (Single source of truth — avoids duplicating 16+ feed URLs across modules)
from nti.scrapers.rss_news import RSS_FEEDS


@dataclass
class NewsArticle:
    """A single news article."""

    title: str
    summary: str
    url: str
    source: str
    published: str


class MultiNewsClient:
    """News client that tries multiple providers with fallback.

    RSS feeds are always enabled (no key needed). Additional APIs
    enrich coverage with broader search capabilities.
    """

    def __init__(self) -> None:
        self.timeout = 15.0

    def fetch_news(self, query: str = "", hours: int = 4, max_articles: int = 30) -> list[NewsArticle]:
        """Fetch news from all enabled providers with fallback.

        Args:
            query: Optional search query for API-based providers
            hours: How many hours back to look for articles
            max_articles: Maximum number of articles to return
        """
        if not settings.enable_news:
            logger.info("News disabled via NTI_ENABLE_NEWS=false")
            return []

        all_articles: list[NewsArticle] = []

        # RSS feeds are always tried first (free, reliable, Indian market focused)
        rss_articles = self._fetch_rss(hours, max_articles)
        all_articles.extend(rss_articles)

        # If we have enough from RSS, return early
        if len(all_articles) >= max_articles:
            return all_articles[:max_articles]

        # Try API-based providers for additional coverage
        for provider in settings.get_enabled_news_providers():
            if provider.name == "rss":
                continue  # Already fetched above

            try:
                articles = self._fetch_from_provider(provider, query, hours, max_articles - len(all_articles))
                all_articles.extend(articles)
                if len(all_articles) >= max_articles:
                    break
            except Exception as e:
                logger.warning(f"News provider {provider.name} failed: {e}. Trying next...")
                continue

        return all_articles[:max_articles]

    def _fetch_rss(self, hours: int, max_articles: int) -> list[NewsArticle]:
        """Fetch news from RSS feeds (free, no key needed)."""
        articles: list[NewsArticle] = []

        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed, skipping RSS")
            return articles

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    articles.append(NewsArticle(
                        title=entry.get("title", ""),
                        summary=entry.get("summary", "")[:200],
                        url=entry.get("link", ""),
                        source=feed_name,
                        published=entry.get("published", ""),
                    ))
            except Exception as e:
                logger.warning(f"RSS feed {feed_name} failed: {e}")
                continue

        return articles[:max_articles]

    def _fetch_from_provider(self, provider, query: str, hours: int, max_articles: int) -> list[NewsArticle]:
        """Fetch news from a specific API provider."""
        match provider.name:
            case "newsapi":
                return self._fetch_newsapi(provider, query, max_articles)
            case "gnews":
                return self._fetch_gnews(provider, query, max_articles)
            case "currents":
                return self._fetch_currents(provider, query, max_articles)
            case "mediastack":
                return self._fetch_mediastack(provider, query, max_articles)
            case "thenewsapi":
                return self._fetch_thenewsapi(provider, query, max_articles)
            case "worldnews":
                return self._fetch_worldnews(provider, query, max_articles)
            case _:
                return []

    def _fetch_newsapi(self, provider, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch via NewsAPI.org."""
        params = {
            "apiKey": provider.api_key,
            "q": query or "Indian stock market NSE",
            "language": "en",
            "pageSize": min(max_articles, 100),
            "sortBy": "publishedAt",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{provider.base_url}/everything", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("articles", [])[:max_articles]:
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("description", "")[:200],
                url=item.get("url", ""),
                source=item.get("source", {}).get("name", "newsapi"),
                published=item.get("publishedAt", ""),
            ))
        return articles

    def _fetch_gnews(self, provider, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch via GNews API."""
        params = {
            "token": provider.api_key,
            "q": query or "Indian stock market",
            "lang": "en",
            "max": min(max_articles, 10),
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{provider.base_url}/top-headlines", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("articles", [])[:max_articles]:
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("description", "")[:200],
                url=item.get("url", ""),
                source=item.get("source", {}).get("name", "gnews"),
                published=item.get("publishedAt", ""),
            ))
        return articles

    def _fetch_currents(self, provider, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch via CurrentsAPI."""
        params = {
            "apiKey": provider.api_key,
            "q": query or "stock market India",
            "language": "en",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{provider.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("news", [])[:max_articles]:
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("description", "")[:200],
                url=item.get("url", ""),
                source=item.get("author", "currents"),
                published=item.get("published", ""),
            ))
        return articles

    def _fetch_mediastack(self, provider, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch via MediaStack."""
        params = {
            "access_key": provider.api_key,
            "keywords": query or "NSE stock market",
            "languages": "en",
            "limit": min(max_articles, 100),
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{provider.base_url}/news", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("data", [])[:max_articles]:
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("description", "")[:200],
                url=item.get("url", ""),
                source=item.get("source", "mediastack"),
                published=item.get("published_at", ""),
            ))
        return articles

    def _fetch_thenewsapi(self, provider, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch via TheNewsAPI."""
        params = {
            "api_token": provider.api_key,
            "search": query or "Indian stock market",
            "language": "en",
            "limit": min(max_articles, 10),
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{provider.base_url}/news/all", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("data", [])[:max_articles]:
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("description", "")[:200],
                url=item.get("url", ""),
                source=item.get("source", "thenewsapi"),
                published=item.get("published_at", ""),
            ))
        return articles

    def _fetch_worldnews(self, provider, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch via World News API."""
        params = {
            "api-key": provider.api_key,
            "text": query or "Indian stock market NSE",
            "language": "en",
            "number": min(max_articles, 50),
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{provider.base_url}/news", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("news", [])[:max_articles]:
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("text", "")[:200],
                url=item.get("url", ""),
                source="worldnews",
                published=item.get("publish_date", ""),
            ))
        return articles


# Singleton
news_client = MultiNewsClient()
