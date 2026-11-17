"""Tests for Render 512MB memory hardening, demand-driven RAG gating, and graceful degradation."""
import os

import psutil
from app.core.config import settings
from app.services.chat_service import should_check_domain_knowledge
from app.services.rag.qdrant_vector_store import (
    BM25Index,
    BoundedBM25Cache,
    NeuralReranker,
    is_flashrank_enabled,
)
from fastapi.testclient import TestClient


def test_low_memory_configuration_defaults():
    """Verify that low-memory RAG mode is configured for 512MB container constraints."""
    assert settings.LOW_MEMORY_RAG_MODE is True
    assert settings.ENABLE_FLASHRANK is False
    assert settings.MAX_PROCESS_RSS_MB <= 400.0


def test_demand_driven_rag_intent_gating():
    """Verify that general chat and coding do not trigger institutional or curriculum RAG."""
    # General queries must NOT trigger domain RAG
    assert should_check_domain_knowledge("Where is Kuttippuram?") is False
    assert should_check_domain_knowledge("Write a Python binary search function.") is False
    assert should_check_domain_knowledge("Tell me a funny joke about cats.") is False
    assert should_check_domain_knowledge("What is the capital of France?") is False
    assert should_check_domain_knowledge("How do I make chocolate chip cookies?") is False
    assert should_check_domain_knowledge("Where can I find a good restaurant in Malappuram?") is False
    assert should_check_domain_knowledge("Write a poem about the sunrise.") is False

    # JRU institutional queries MUST trigger domain RAG
    assert should_check_domain_knowledge("What is the fee structure for BTech at JRU?") is True
    assert should_check_domain_knowledge("How do I apply for Jharkhand Rai University admissions?") is True
    assert should_check_domain_knowledge("Who is the vice chancellor of JRU?") is True
    assert should_check_domain_knowledge("Where is JRU campus located?") is True

    # STEM curriculum queries MUST trigger domain RAG
    assert should_check_domain_knowledge("Explain calculus derivatives and limits.") is True
    assert should_check_domain_knowledge("Explain quantum mechanics and wavefunctions.") is True


def test_flashrank_conditional_activation():
    """Verify that neural reranker is safely gated under low memory mode."""
    # Under default LOW_MEMORY_RAG_MODE=True and ENABLE_FLASHRANK=False
    enabled, reason = is_flashrank_enabled()
    assert enabled is False
    assert "low_memory" in reason

    reranker = NeuralReranker()
    assert reranker.is_available is False
    assert reranker.degraded is True

    # Calling rerank on degraded reranker must return top RRF items without crashing
    sample_candidates = [
        {"text": "Candidate 1", "score": 0.95},
        {"text": "Candidate 2", "score": 0.85},
        {"text": "Candidate 3", "score": 0.75},
    ]
    results = reranker.rerank("test query", sample_candidates, top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "Candidate 1"
    assert reranker.rag_metadata["rag_reranker_used"] is False
    assert reranker.rag_metadata["rag_degraded"] is True


def test_bounded_bm25_cache_eviction():
    """Verify that BoundedBM25Cache enforces strict LRU maxsize bounds."""
    cache = BoundedBM25Cache(maxsize=3)

    idx1 = BM25Index()
    idx2 = BM25Index()
    idx3 = BM25Index()
    idx4 = BM25Index()

    cache.set("col1", idx1)
    cache.set("col2", idx2)
    cache.set("col3", idx3)

    assert len(cache) == 3
    assert cache.size() == 3
    assert cache.get("col1") is idx1

    # Adding a 4th must evict the least recently used
    cache.set("col4", idx4)
    assert len(cache) == 3
    # col2 was least recently used because col1 was accessed via get()
    assert cache.get("col2") is None
    assert cache.get("col1") is idx1
    assert cache.get("col3") is idx3
    assert cache.get("col4") is idx4


def test_e2e_general_chat_does_not_initialize_heavy_rag(monkeypatch):
    """Verify that a standard general query does not retrieve domain sources or cause memory spikes."""
    from app.api.dependencies import get_chat_service
    from app.main import app

    async def mock_generate(**kwargs):
        return ("Kuttippuram is a town in Malappuram district, Kerala, India.", "mock_provider", "mock_model", False)

    monkeypatch.setattr(get_chat_service().router, "generate_response", mock_generate)
    client = TestClient(app)
    # Warm up endpoint modules and schemas so delta measures query processing, not Python module loading
    client.post("/api/v1/chat", json={"message": "warmup", "model": "auto", "stream": False})
    import gc
    gc.collect()

    rss_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    resp = client.post("/api/v1/chat", json={
        "message": "Where is Kuttippuram?",
        "model": "auto",
        "stream": False,
    })
    rss_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    assert resp.status_code == 200
    data = resp.json()

    # Domain sources must be empty for general chat
    assert data.get("sources") == [] or data.get("sources") is None
    assert "Generation failed" not in data.get("content", "")
    assert "quota" not in data.get("content", "").lower()

    # Delta memory must be minimal (general query does not load heavy models)
    delta_mb = max(0.0, rss_after - rss_before)
    assert delta_mb < 30.0, f"Unexpected memory spike of {delta_mb:.1f} MiB on general query"


def test_concurrent_operations_memory_rss(monkeypatch):
    """Verify that concurrent operations remain bounded under load below 400MB threshold."""
    import concurrent.futures
    import gc

    from app.api.dependencies import get_chat_service
    from app.main import app

    async def mock_generate(**kwargs):
        return ("Concurrent test response with bounded payload.", "mock_provider", "mock_model", False)

    monkeypatch.setattr(get_chat_service().router, "generate_response", mock_generate)
    client = TestClient(app)

    def _worker(idx):
        if idx % 3 == 0:
            return client.get("/api/v1/health/db")
        elif idx % 3 == 1:
            return client.post("/api/v1/chat", json={"message": f"Concurrent query {idx}", "model": "auto", "stream": False})
        else:
            return client.get("/api/v1/models")

    rss_baseline = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_worker, i) for i in range(20)]
        results = [f.result() for f in futures]

    gc.collect()
    rss_peak = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for r in results:
        assert r.status_code == 200

    delta_mb = max(0.0, rss_peak - rss_baseline)
    assert delta_mb < 50.0, f"Memory delta under concurrent load ({delta_mb:.1f} MB) exceeded 50MB budget!"
