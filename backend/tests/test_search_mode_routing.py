"""Regression: web search must follow the user's explicit Off/On choice, and Auto must not fire on
self-contained tasks. Found by the 10 Nov 2026 audit: the word "schedule" in a five-task prompt
started a web search and the answer showed five unrelated sources.
"""

import pytest
from app.services.search.web_search import web_search_service

BUNDLE = (
    "Complete five independent tasks. Label each answer.\n"
    "1. Schedule four meetings A, B, C, D so that A is first and C is before B.\n"
    "2. Write a Python function that merges overlapping intervals.\n"
    "3. Using only the sample below, what is the pH of the solution?\n"
    "Sample: volume 20 ml, temperature 25 C."
)

AUTO_NO_SEARCH = [
    BUNDLE,
    "Using only the text below, when does billing start? Text: approved on 3 March.",
    "Write a match function in Python that compares two strings",
    "What is the exchange rate formula in economics?",
    "Explain this code:\n```python\nprint('now')\n```",
]

AUTO_SEARCH = [
    "What is the latest news on Chandrayaan?",
    "Who won the IPL final yesterday?",
    "current price of bitcoin",
    "weather in Kochi today",
]


@pytest.mark.parametrize("text", AUTO_NO_SEARCH)
def test_auto_does_not_search_self_contained_tasks(text):
    assert web_search_service.should_search(text, None) is False


@pytest.mark.parametrize("text", AUTO_SEARCH)
def test_auto_still_searches_fresh_info_questions(text):
    assert web_search_service.should_search(text, None) is True


def test_explicit_on_searches_even_for_if_questions():
    assert web_search_service.should_search("If Python 3.14 is out, what changed?", True) is True


def test_explicit_off_never_searches():
    assert web_search_service.should_search("latest news today", False) is False


def test_explicit_on_ignores_trivial_input():
    assert web_search_service.should_search("hi", True) is False
