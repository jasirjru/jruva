"""Jruva speaks only as Jruva: it never names the models, providers or people behind it."""

import pytest
from app.services.chat_service import (
    DEFAULT_SYSTEM_PROMPT,
    JRUVA_IDENTITY_REPLY,
    classify_conversational_intent,
    sanitize_model_output,
)

BACKEND_NAMES = ("gemini", "gemma", "groq", "openrouter", "ollama", "qwen", "llama", "gpt", "google", "meta", "openai")


@pytest.mark.parametrize(
    "question",
    [
        "who are you",
        "who made you?",
        "who created you",
        "who built this ai",
        "who owns you",
        "who is your owner",
        "who is your creator",
        "who is your developer",
        "which model are you",
        "what model are you using",
        "what llm do you use",
        "which ai model powers you",
        "what is your underlying model",
        "are you chatgpt",
        "are you gemini?",
        "are you llama",
        "what company made you",
        "what is jruva",
    ],
)
def test_identity_questions_get_the_jruva_reply(question):
    assert classify_conversational_intent(question) == JRUVA_IDENTITY_REPLY


def test_identity_reply_names_no_one_but_jruva():
    reply = JRUVA_IDENTITY_REPLY.lower()
    assert "jruva" in reply
    assert not any(name in reply for name in BACKEND_NAMES)


@pytest.mark.parametrize(
    "question",
    ["who is the owner of tesla", "which model of iphone is best", "who made the eiffel tower", "what is gpt-4"],
)
def test_questions_about_other_things_are_not_hijacked(question):
    assert classify_conversational_intent(question) != JRUVA_IDENTITY_REPLY


def test_system_status_does_not_list_providers():
    reply = (classify_conversational_intent("system status") or "").lower()
    assert reply
    assert not any(name in reply for name in BACKEND_NAMES)


@pytest.mark.parametrize("command", ["/status", "/clear", "/reset"])
def test_slash_only_commands_are_gone(command):
    assert classify_conversational_intent(command) is None


@pytest.mark.parametrize("word", ["help", "courses", "attendance", "scholarships", "fee"])
def test_a_leading_slash_adds_nothing(word):
    assert classify_conversational_intent("/" + word) == classify_conversational_intent(word)


def test_system_prompt_forbids_naming_models_providers_or_people():
    prompt = DEFAULT_SYSTEM_PROMPT.lower()
    assert "answer only: 'i'm jruva.'" in prompt
    assert "never name" in prompt


@pytest.mark.parametrize(
    "leak",
    [
        "I am Gemma, an open model trained by Google DeepMind.",
        "I'm a large language model, trained by Google.",
        "I am Llama 3, developed by Meta.",
        "I'm Qwen, created by Alibaba Cloud.",
        "As an AI developed by OpenAI, I can help.",
        "My underlying model is GPT-OSS 120B running on Groq.",
    ],
)
def test_self_descriptions_naming_a_backend_are_rewritten(leak):
    out = sanitize_model_output(leak, "who are you").lower()
    assert "jruva" in out
    assert not any(name in out for name in BACKEND_NAMES)


def test_facts_about_other_ai_products_are_left_alone():
    text = "GPT-4 was developed by OpenAI and released in 2023."
    assert sanitize_model_output(text, "who made gpt-4") == text
