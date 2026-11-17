"""Source Collector & Deduplicator for Bounded Research Mode."""

import asyncio
import logging
import urllib.parse
from typing import List, Set
from pydantic import BaseModel

from app.schemas.chat import SourceCitation
from app.services.search.web_search import web_search_service

logger = logging.getLogger(__name__)


class ResearchSource(BaseModel):
    index: int
    title: str
    url: str
    snippet: str
    source_query: str


def normalize_url(url: str) -> str:
    """Strip tracking query parameters and trailing slashes for canonical deduplication."""
    try:
        parsed = urllib.parse.urlparse(url)
        # Strip query and fragment for deduplication
        clean = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
        return clean.lower()
    except Exception:
        return url.strip().lower()


class ResearchCollector:
    """Asynchronously gathers search results across planned sub-queries with deduplication."""

    @classmethod
    async def collect(cls, queries: List[str], max_sources: int = 8) -> List[ResearchSource]:
        async def fetch_for_query(q: str) -> List[tuple]:
            try:
                results = await web_search_service.search(q)
                return [(q, r) for r in results]
            except Exception as e:
                logger.warning(f"Research search failed (sub-query chars={len(q)}): {e}")
                return []

        # Run sub-queries concurrently
        tasks = [fetch_for_query(q) for q in queries]
        nested_results = await asyncio.gather(*tasks)

        seen_urls: Set[str] = set()
        deduped_sources: List[ResearchSource] = []
        source_idx = 1

        for subquery_results in nested_results:
            for query_str, citation in subquery_results:
                norm = normalize_url(citation.url)
                if norm in seen_urls:
                    continue
                seen_urls.add(norm)

                deduped_sources.append(
                    ResearchSource(
                        index=source_idx,
                        title=citation.title,
                        url=citation.url,
                        snippet=citation.snippet,
                        source_query=query_str,
                    )
                )
                source_idx += 1
                if len(deduped_sources) >= max_sources:
                    break
            if len(deduped_sources) >= max_sources:
                break

        return deduped_sources
