"""Jruva must report the Gemini model Google actually serves.

Google retired gemini-3.5-flash by routing it to gemini-3.6-flash (Gemini
deprecations page, checked 8 Nov 2026; a live request for gemini-3.5-flash
returned modelVersion "gemini-3.6-flash"). Saved preferences that still say
gemini-3.5-flash must keep working and resolve to the served model.
"""

import logging

from app.core.model_registry import get_canonical_model
from app.services.llm.gemini_provider import GeminiProvider, _warn_if_served_model_differs


def test_retired_gemini_35_flash_resolves_to_served_36_flash():
    assert get_canonical_model("gemini-3.5-flash").model_id == "gemini-3.6-flash"
    assert get_canonical_model("gemini-3.6-flash").model_id == "gemini-3.6-flash"


def test_provider_maps_retired_id_to_served_model():
    assert GeminiProvider(api_key="test", model="gemini-3.5-flash").model == "gemini-3.6-flash"
    assert GeminiProvider(api_key="test", model="gemini-3.6-flash").model == "gemini-3.6-flash"


def test_flash_lite_is_not_renamed():
    assert GeminiProvider(api_key="test", model="gemini-3.5-flash-lite").model == "gemini-3.5-flash-lite"
    assert get_canonical_model("gemini-3.5-flash-lite").model_id == "gemini-3.5-flash-lite"


def test_warns_when_google_serves_a_different_model(caplog):
    with caplog.at_level(logging.WARNING, logger="app.services.llm.gemini_provider"):
        _warn_if_served_model_differs("gemini-3.5-flash", "gemini-3.6-flash")
    assert "served gemini-3.6-flash" in caplog.text


def test_no_warning_when_served_model_matches_or_is_missing(caplog):
    with caplog.at_level(logging.WARNING, logger="app.services.llm.gemini_provider"):
        _warn_if_served_model_differs("gemini-3.6-flash", "gemini-3.6-flash")
        _warn_if_served_model_differs("gemini-3.6-flash", None)
    assert caplog.text == ""
