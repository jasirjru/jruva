"""Shared helpers for reading provider rate-limit cooldowns and model lists.

Groq and Gemini tell us how long a model stays rate-limited (``retry-after`` header,
or Google's ``RetryInfo.retryDelay`` in the error body). Production logs on
10 Nov 2026 showed waits of 84-971 s, while the health tracker used a fixed 60 s
cooldown, so the router kept re-hitting models that were still blocked.
"""
import json
import re
from collections.abc import Mapping
from typing import Any

DEFAULT_COOLDOWN_SECONDS = 60.0
MIN_COOLDOWN_SECONDS = 5.0
# Daily quotas reset within 24 h; cap at 1 h so a bad header cannot park a model all day.
MAX_COOLDOWN_SECONDS = 3600.0

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")


def _parse_seconds(value: Any) -> float | None:
    if value is None:
        return None
    match = _DURATION_RE.match(str(value))
    return float(match.group(1)) if match else None


def _retry_delay_from_body(body: str | None) -> float | None:
    """Read Google's ``RetryInfo.retryDelay`` (e.g. ``"34s"``) from an error body."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    error = data.get("error") if isinstance(data, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    for detail in details or []:
        if isinstance(detail, dict) and "retryDelay" in detail:
            return _parse_seconds(detail.get("retryDelay"))
    return None


def cooldown_from_response(
    headers: Mapping[str, str] | None,
    body: str | None = None,
    default: float = DEFAULT_COOLDOWN_SECONDS,
) -> float:
    """Return how long (seconds) a rate-limited model should be skipped.

    Uses ``retry-after`` first, then Google's ``retryDelay``; clamps to
    [MIN_COOLDOWN_SECONDS, MAX_COOLDOWN_SECONDS]. Falls back to ``default``.
    """
    seconds = _parse_seconds((headers or {}).get("retry-after"))
    if seconds is None:
        seconds = _retry_delay_from_body(body)
    if seconds is None:
        return default
    return max(MIN_COOLDOWN_SECONDS, min(seconds, MAX_COOLDOWN_SECONDS))


def parse_model_list(raw: str | None, default: list[str]) -> list[str]:
    """Split a comma-separated env value into model IDs, keeping order and dropping duplicates."""
    if not raw or not raw.strip():
        return list(default)
    models: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in models:
            models.append(model)
    return models or list(default)
