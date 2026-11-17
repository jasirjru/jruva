"""Unit & Integration Tests for Bounded Research Mode."""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from app.schemas.chat import SourceCitation
from app.services.research import ResearchPlanner, ResearchCollector, ResearchEngine
from app.services.research.collector import normalize_url


def test_research_planner_query_decomposition():
    # 1. Comparison topic
    topic = "PostgreSQL vs SQLite for production Cloud Run"
    queries = ResearchPlanner.plan_subqueries(topic)
    assert len(queries) <= 3
    assert any("features" in q or "architecture" in q for q in queries)

    # 2. General technical topic
    tech_topic = "How to optimize FastEmbed ONNX runtime performance"
    queries2 = ResearchPlanner.plan_subqueries(tech_topic)
    assert len(queries2) <= 3
    assert len(queries2) >= 1

    # 3. Simple topic
    simple_topic = "Quantum computing advancements"
    queries3 = ResearchPlanner.plan_subqueries(simple_topic)
    assert len(queries3) <= 3


def test_url_normalization_and_deduplication():
    url1 = "https://example.com/docs/article/?utm_source=twitter&ref=123"
    url2 = "https://example.com/docs/article"
    assert normalize_url(url1) == normalize_url(url2)


@pytest.mark.asyncio
async def test_research_collector_deduplication():
    mock_citations = [
        SourceCitation(id="c1", title="Postgres Guide", url="https://db.org/guide?token=1", snippet="Fast reliable DB"),
        SourceCitation(id="c2", title="Postgres Guide Duplicate", url="https://db.org/guide", snippet="Duplicate URL snippet"),
        SourceCitation(id="c3", title="SQLite Guide", url="https://sqlite.org/intro", snippet="Embedded DB"),
    ]

    with patch("app.services.search.web_search.web_search_service.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_citations
        sources = await ResearchCollector.collect(["PostgreSQL vs SQLite"], max_sources=5)

        # 3 citations returned, but 2 share the same canonical URL -> should have exactly 2 deduped sources
        assert len(sources) == 2
        assert sources[0].index == 1
        assert sources[1].index == 2
        assert sources[0].url == "https://db.org/guide?token=1"
        assert sources[1].url == "https://sqlite.org/intro"


@pytest.mark.asyncio
async def test_research_engine_end_to_end_synthesis():
    mock_citations = [
        SourceCitation(id="c1", title="Architecture Spec", url="https://spec.org", snippet="High throughput design"),
    ]


    with patch("app.services.search.web_search.web_search_service.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_citations
        report = await ResearchEngine.conduct_research(
            topic="Modern microservices architecture",
            timeout_seconds=5.0,
        )

        assert report.topic == "Modern microservices architecture"
        assert len(report.sources) >= 1
        assert "[1]" in report.synthesis_context
        assert len(report.structured_citations) >= 1
        assert report.structured_citations[0].title.startswith("[1]")
        assert report.execution_time_ms > 0


@pytest.mark.asyncio
async def test_research_engine_timeout_enforcement():
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(2.0)
        return []

    with patch("app.services.search.web_search.web_search_service.search", side_effect=slow_search):
        with pytest.raises(asyncio.TimeoutError):
            await ResearchEngine.conduct_research(
                topic="Slow query topic",
                timeout_seconds=0.1,
            )
