"""Locust load test suite for JRU AI Chatbot backend.

Measures p50, p95, p99 latencies across:
1. Health and Model catalog endpoints
2. Sub-15ms semantic response cache hit path
3. Concurrency limiter & burst queue path
"""
import uuid
from locust import HttpUser, task, between


class ChatbotLoadUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.user_id = f"user-{uuid.uuid4().hex[:8]}"
        self.headers = {
            "X-User-ID": self.user_id,
            "x-benchmark-bypass": "jru-benchmark-internal-bypass-key",
        }

    @task(3)
    def test_models_catalog(self):
        """Query models catalog (SLO target: p50 < 3ms, p95 < 10ms)."""
        self.client.get("/api/v1/models", headers=self.headers, name="GET /api/v1/models")

    @task(2)
    def test_image_health(self):
        """Query image provider health endpoint."""
        self.client.get("/api/v1/image/health", headers=self.headers, name="GET /api/v1/image/health")

    @task(4)
    def test_chat_cached_resolution(self):
        """Query chat endpoint with cached prompt to measure in-memory LRU resolution under concurrency."""
        payload = {
            "session_id": str(uuid.uuid4()),
            "message": "benchmark-cached-probe-prompt",
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        self.client.post("/api/v1/chat", json=payload, headers=self.headers, name="POST /api/v1/chat (cached)")

    @task(1)
    def test_image_cached_resolution(self):
        """Query image endpoint with cached S+++ prompt."""
        payload = {
            "prompt": "benchmark-cached-image-prompt",
            "model": "flux",
            "tier": "S+++",
            "width": 1024,
            "height": 1024,
        }
        self.client.post("/api/v1/image/generate", json=payload, headers=self.headers, name="POST /api/v1/image/generate (cached)")
