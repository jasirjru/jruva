"""Realistic Mixed-Traffic Load Test Scenario for JRU AI Chatbot.

Traffic Composition:
1. Most sessions (Weight 7): One uncached first message (cache miss, real LLM call),
   followed by 1-2 cache-hit follow-ups.
2. Minority of sessions (Weight 2): Image generation request.
3. Minority of sessions (Weight 1): RAG document attachment upload + grounded query.
"""
import base64
import uuid
from locust import HttpUser, task, between


class MixedTrafficUser(HttpUser):
    wait_time = between(0.1, 0.4)

    def on_start(self):
        self.user_id = f"user-{uuid.uuid4().hex[:8]}"
        self.headers = {
            "X-User-ID": self.user_id,
            "x-benchmark-bypass": "jru-benchmark-internal-bypass-key",
        }
        self.session_id = str(uuid.uuid4())
        self.cached_prompt = "benchmark-cached-probe-prompt"
        self.doc_b64 = base64.b64encode(
            b"JRU University policy: Academic semester is 16 weeks. Passing grade threshold is 40 percent. "
            b"Library hours are 8am to 8pm Monday through Saturday."
        ).decode("ascii")

    @task(7)
    def chat_multi_turn_session(self):
        """Standard session: 1 uncached LLM call followed by 1-2 cached follow-ups."""
        # Turn 1: Uncached query (cache miss, invokes real backend LLM)
        uncached_payload = {
            "session_id": self.session_id,
            "message": f"Summarize campus engineering guidelines {uuid.uuid4().hex[:6]}",
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        self.client.post(
            "/api/v1/chat",
            json=uncached_payload,
            headers=self.headers,
            name="POST /api/v1/chat (uncached-turn1)",
        )

        # Turn 2: Cached follow-up
        cached_payload_1 = {
            "session_id": self.session_id,
            "message": self.cached_prompt,
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        self.client.post(
            "/api/v1/chat",
            json=cached_payload_1,
            headers=self.headers,
            name="POST /api/v1/chat (cached-turn2)",
        )

        # Turn 3: Cached follow-up
        cached_payload_2 = {
            "session_id": self.session_id,
            "message": self.cached_prompt,
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        self.client.post(
            "/api/v1/chat",
            json=cached_payload_2,
            headers=self.headers,
            name="POST /api/v1/chat (cached-turn3)",
        )

    @task(2)
    def image_generation_flow(self):
        """Minority session: Image generation request."""
        payload = {
            "prompt": f"futuristic university campus quad neon {uuid.uuid4().hex[:4]}",
            "model": "flux",
            "tier": "B",
            "width": 512,
            "height": 512,
        }
        self.client.post(
            "/api/v1/image/generate",
            json=payload,
            headers=self.headers,
            name="POST /api/v1/image/generate (mixed)",
        )

    @task(1)
    def rag_document_query_flow(self):
        """Minority session: Document attachment ingestion + RAG retrieval."""
        rag_payload = {
            "session_id": str(uuid.uuid4()),
            "message": "What is the passing grade threshold and library hours?",
            "attachments": [
                {
                    "name": "campus_policy.txt",
                    "mime_type": "text/plain",
                    "data": self.doc_b64,
                }
            ],
            "model": "openai/gpt-oss-120b",
            "tier": "S+++",
            "stream": False,
        }
        self.client.post(
            "/api/v1/chat",
            json=rag_payload,
            headers=self.headers,
            name="POST /api/v1/chat (rag-grounded)",
        )
