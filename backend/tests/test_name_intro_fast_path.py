"""The name-introduction fast path must only answer pure introductions."""
import pytest
from app.services.chat_service import classify_conversational_intent


@pytest.mark.parametrize("message", [
    "My name is Ravi",
    "Hi, my name is Ravi Kumar.",
    "I'm Priya",
    "Hello, I am Arjun. Nice to meet you!",
    "Call me Sam",
])
def test_pure_introduction_gets_greeting(message):
    reply = classify_conversational_intent(message)
    assert reply is not None and reply.startswith("Nice to meet you")


@pytest.mark.parametrize("message", [
    "My name is Ravi and I live in Pune. What is the capital of Australia?",
    "My name is Ravi, explain photosynthesis",
    "I'm Priya. Can you help me write a cover letter?",
    "Call me Sam and tell me a joke",
])
def test_introduction_with_a_request_is_not_swallowed(message):
    reply = classify_conversational_intent(message)
    assert reply is None or not reply.startswith("Nice to meet you")
