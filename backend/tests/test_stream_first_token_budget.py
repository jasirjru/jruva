"""Regression: everyday (non-deep) streaming chat must give a model enough time to start answering.

Found 9 Nov 2026 by the locked_ml_v1 smoke run: through Jruva's pipeline, Groq Qwen 3.8 (a reasoning
model that writes nothing until it finishes thinking) took 3.7-4.3 s to its first token. The old 4.0 s
limit cut it off, Gemini got the same 4.0 s, and 3 of 5 questions ended in
"AI capacity is temporarily unavailable" instead of an answer.
"""

from app.services.chat_service import STREAM_FIRST_TOKEN_SECONDS, stream_budget_for

MEASURED_SLOWEST_FIRST_TOKEN_S = 4.3


def test_everyday_chat_first_token_limit_covers_measured_reasoning_latency():
    budget = stream_budget_for(is_deep=False)

    assert budget.first_token_target_seconds == STREAM_FIRST_TOKEN_SECONDS
    assert budget.first_token_target_seconds >= MEASURED_SLOWEST_FIRST_TOKEN_S * 1.5


def test_everyday_chat_total_budget_unchanged():
    assert stream_budget_for(is_deep=False).total_seconds == 25.0


def test_deep_chat_budget_unchanged():
    budget = stream_budget_for(is_deep=True)

    assert budget.total_seconds == 60.0
    assert budget.first_token_target_seconds == 8.0
