"""Bounded Research Engine: Coordinates planning, retrieval, deduplication, and grounded synthesis context."""

import asyncio
import time
from typing import List, Optional
from pydantic import BaseModel

from app.schemas.chat import SourceCitation
from app.services.research.collector import ResearchCollector, ResearchSource
from app.services.research.planner import ResearchPlanner


class ResearchReport(BaseModel):
    topic: str
    planned_queries: List[str]
    sources: List[ResearchSource]
    synthesis_context: str
    structured_citations: List[SourceCitation]
    execution_time_ms: float


class ResearchEngine:
    """Bounded, safe deep-research engine with hard step caps and timeout enforcement."""

    DEFAULT_TIMEOUT_SECONDS = 15.0
    MAX_SOURCES = 8

    @classmethod
    async def conduct_research(
        cls,
        topic: str,
        timeout_seconds: Optional[float] = None,
    ) -> ResearchReport:
        start = time.perf_counter()
        timeout = timeout_seconds or cls.DEFAULT_TIMEOUT_SECONDS

        async def _run() -> ResearchReport:
            # 1. Plan sub-queries (strictly capped at 3)
            plan = ResearchPlanner.plan_subqueries(topic)

            # 2. Collect and deduplicate sources
            sources = await ResearchCollector.collect(queries=plan, max_sources=cls.MAX_SOURCES)

            # 3. Build synthesis context with numbered citations
            context_blocks = []
            citations = []
            for src in sources:
                context_blocks.append(
                    f"[{src.index}] \"{src.title}\" ({src.url})\nEvidence: {src.snippet}"
                )
                citations.append(
                    SourceCitation(
                        id=f"research_src_{src.index}",
                        title=f"[{src.index}] {src.title}",
                        url=src.url,
                        snippet=src.snippet,
                    )
                )


            synthesis_prompt = (
                "You are an expert research analyst. Synthesize a comprehensive, unbiased research report "
                f"answering the user topic: \"{topic}\".\n"
                "Ground all key claims in the numbered evidence sources below using inline brackets (e.g. [1], [2]).\n\n"
                "RESEARCH EVIDENCE SOURCES:\n" + "\n\n".join(context_blocks)
            )

            elapsed = (time.perf_counter() - start) * 1000.0
            return ResearchReport(
                topic=topic,
                planned_queries=plan,
                sources=sources,
                synthesis_context=synthesis_prompt,
                structured_citations=citations,
                execution_time_ms=round(elapsed, 2),
            )

        return await asyncio.wait_for(_run(), timeout=timeout)


research_engine = ResearchEngine()
