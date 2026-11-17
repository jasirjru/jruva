"""The first question after a cold start must still reach a model.

Production, 10 Nov 2026 08:29 UTC: on a freshly woken Render instance, first-time setup
(library imports, database round trips, a 4 s knowledge lookup) used up the 25 s stream
budget. The router logged "Stream budget exhausted (1.5s remaining). Aborting failover
before groq" and the user got "All free AI models are busy" without any model being tried.

Before calling the router, the budget is now topped up so models get at least
MODEL_WINDOW_SECONDS.
"""
import time
from unittest.mock import patch

import pytest
from app.schemas.chat import ChatRequest
from app.services import chat_service as chat_module
from app.services.chat_service import MODEL_WINDOW_SECONDS, ChatService
from app.services.llm.router import RequestBudget

from tests.test_jruva_lightning_regression_matrix import MockFastProvider


def aged_budget(total: float, elapsed: float) -> RequestBudget:
    budget = RequestBudget(total_seconds=total, first_token_target_seconds=8.0)
    budget.start_time = time.monotonic() - elapsed
    return budget


def test_ensure_window_tops_up_an_almost_spent_budget():
    budget = aged_budget(total=25.0, elapsed=24.0)

    budget.ensure_window(12.0)

    assert budget.remaining_seconds >= 11.9


def test_ensure_window_keeps_a_budget_with_enough_time():
    budget = aged_budget(total=25.0, elapsed=2.0)

    budget.ensure_window(12.0)

    assert budget.total_seconds == 25.0


@pytest.mark.asyncio
async def test_slow_setup_still_reaches_the_model(monkeypatch):
    monkeypatch.setattr(chat_module, "stream_budget_for", lambda _is_deep: aged_budget(total=25.0, elapsed=24.0))
    service = ChatService()
    provider = MockFastProvider(name="groq", tokens=["Photosynthesis ", "makes sugar."])

    with patch.object(service.router, "_resolve_providers", return_value=(provider, None, "qwen/qwen3.8-27b")):
        req = ChatRequest(message="Tell me something nice about rain.", stream=True)
        output = "".join([chunk async for chunk in service.stream_chat_response(req)])

    assert "makes sugar." in output
    assert "timed out" not in output
    assert MODEL_WINDOW_SECONDS >= 10.0
