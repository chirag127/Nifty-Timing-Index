"""RSS News Feed Aggregator — 5 free Indian financial news sources.

RSS feeds require no API key and always work. They form the base
of the news analysis pipeline, supplemented by paid API providers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import feedparser
except ImportError:
    feedparser = None  # RSS_FEEDS dict is still importable without feedparser

logger = logging.getLogger(__name__)

# RSS Feed URLs (free, no key needed)
# Expanded to include Financial Times, Reuters, and more Indian market sources.
# Each feed provides Indian/Asian market context for the NTI analysis.
#
# NOTE: Some international feeds (Reuters, FT, Investing.com, NSE) may require
# updates over time as providers change their RSS structure. All feeds are wrapped
# in try/except so failures are logged but don't crash the pipeline.
RSS_FEEDS = {
    # --- Primary Indian Market Feeds ---
    "economictimes": "https://economictimes.indiatimes.com/markets/rss.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "businessstandard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint": "https://feeds.livemint.com/livemint/lotze/2023-06/markets",
    "thehindu": "https://thehindu.com/business/markets/?service=rss",
    # --- Additional Indian Market Feeds ---
    "et_ipo": "https://economictimes.indiatimes.com/markets/ipo/rss.cms",
    "et_commodities": "https://economictimes.indiatimes.com/markets/commodities/rss.cms",
    "et_currencies": "https://economictimes.indiatimes.com/markets/forex/rss.cms",
    "mc_top": "https://www.moneycontrol.com/rss/marketstudies.xml",
    "bs_ipo": "https://www.business-standard.com/rss/ipo-109.rss",
    # --- International / Global Context ---
    "reuters_markets": "https://www.reuters.com/rssFeed/marketsNews",
    "reuters_business": "https://www.reuters.com/rssFeed/businessNews",
    "ft_markets": "https://www.ft.com/rss/home/markets",
    "ft_asia": "https://www.ft.com/rss/home/asia-pacific",
    "investing_india": "https://www.investing.com/rss/news_301.rss",
    # --- NSE / Market Data ---
    "nse_announcements": "https://www.nseindia.com/rss/announcements.xml",
}


@dataclass
class RSSArticle:
    """A single RSS article."""

    title: str
    summary: str
    url: str
    source: str
    published: str


# Alias for backward compatibility with tests
def fetch_recent_news(hours: int = 4, max_per_feed: int = 10, max_total: int = 30) -> list[RSSArticle]:
    """Alias for fetch_rss_news."""
    return fetch_rss_news(hours=hours, max_per_feed=max_per_feed, max_total=max_total)


def fetch_rss_news(hours: int = 4, max_per_feed: int = 10, max_total: int = 30) -> list[RSSArticle]:
    """Fetch recent news from all RSS feeds.

    Args:
        hours: How many hours back to look for articles
        max_per_feed: Maximum articles per feed
        max_total: Maximum total articles across all feeds

    Returns:
        List of RSSArticle objects, sorted by published date (newest first)
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    all_articles: list[RSSArticle] = []

    if feedparser is None:
        logger.warning("feedparser not installed — cannot fetch RSS news")
        return []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)

            count = 0
            for entry in feed.entries:
                if count >= max_per_feed:
                    break

                title = entry.get("title", "").strip()
                if not title:
                    continue

                summary = entry.get("summary", entry.get("description", ""))[:300]
                url = entry.get("link", "")
                published_str = entry.get("published", "")

                # Try to parse and filter by date
                if published_str:
                    try:
                        from dateutil import parser as date_parser
                        pub_date = date_parser.parse(published_str, fuzzy=True)
                        # RSS dates may have timezone info; compare naive UTC
                        pub_utc = pub_date.replace(tzinfo=None) if pub_date.tzinfo is None else pub_date.utctimetuple()
                        if isinstance(pub_utc, tuple):
                            pub_dt = datetime(*pub_utc[:6])
                        else:
                            pub_dt = pub_utc
                        if pub_dt < cutoff:
                            continue
                    except Exception:
                        pass  # Include article even if date parsing fails

                all_articles.append(
                    RSSArticle(
                        title=title,
                        summary=summary,
                        url=url,
                        source=source_name,
                        published=published_str,
                    )
                )
                count += 1

        except Exception as e:
            logger.warning(f"RSS feed {source_name} failed: {e}")
            continue

    # Sort by source diversity and return
    # Interleave sources for balanced coverage
    articles_by_source: dict[str, list[RSSArticle]] = {}
    for article in all_articles:
        articles_by_source.setdefault(article.source, []).append(article)

    interleaved: list[RSSArticle] = []
    max_len = max(len(v) for v in articles_by_source.values()) if articles_by_source else 0
    for i in range(max_len):
        for source in RSS_FEEDS:
            if source in articles_by_source and i < len(articles_by_source[source]):
                interleaved.append(articles_by_source[source][i])

    result = interleaved[:max_total]
    logger.info(f"RSS news: {len(result)} articles from {len(articles_by_source)} sources")
    return result


def format_rss_headlines(articles: list[RSSArticle]) -> str:
    """Format RSS articles as a text block for LLM consumption.

    Returns a string with one line per article:
    - [Source] Title: Summary excerpt
    """
    lines = []
    for a in articles[:30]:
        lines.append(f"- [{a.source}] {a.title}: {a.summary[:150]}")
    return "\n".join(lines)
