"""Regression: ordinary text questions must not be routed to image generation.

Found by the 10 Nov 2026 live audit: "Render" (company) matched the verb "render" and
"workspace" matched the noun "space", so a web-search question became an image request.
"""

import pytest
from app.services.image.image_service import image_service

NOT_IMAGE = [
    "Search the web. According to Render's official documentation, how long without inbound "
    "traffic before a free web service spins down, and how many free instance hours are "
    "available per workspace per calendar month?",
    "How do I render a React component conditionally?",
    "What is the capacity of the education department?",
    "Create a schedule for my exams next week",
    "Explain the topic of law of torts",
    "Can you see why my seed data fails?",
    "Generate a summary of this chapter",
]

IS_IMAGE = [
    "Generate a photorealistic red Ferrari parked beside a beach at sunset",
    "draw a cat",
    "please draw me a dragon",
    "create an image of a mountain lake",
    "/imagine a cyberpunk city at night",
]


@pytest.mark.parametrize("text", NOT_IMAGE)
def test_text_questions_are_not_image_requests(text):
    is_img, _ = image_service.detect_intent(text)
    assert not is_img, text


@pytest.mark.parametrize("text", IS_IMAGE)
def test_real_image_requests_still_detected(text):
    is_img, subject = image_service.detect_intent(text)
    assert is_img, text
    assert subject
