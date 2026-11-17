"""Supabase storage must start when SUPABASE_URL and the service key are set.

Production, 10 Nov 2026: Render had STORAGE_BACKEND=supabase, SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY, yet startup failed with "SUPABASE_URL or service key is
missing". Settings did not declare SUPABASE_URL, so pydantic dropped it
(extra="ignore") and validate_production_storage(settings) always saw "".
"""
import pytest
from app.core.config import Settings
from app.main import validate_production_storage

URL = "https://example-project.supabase.co"
KEY = "sb_secret_test_value"


def test_settings_reads_supabase_url_from_environment(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)

    assert Settings().SUPABASE_URL == URL


def test_production_supabase_storage_with_url_and_key_starts(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", KEY)

    validate_production_storage(Settings())


def test_production_supabase_storage_without_url_still_fails(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("STORAGE_BACKEND", "supabase")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", KEY)

    with pytest.raises(RuntimeError, match="Durable storage configuration fatal"):
        validate_production_storage(Settings(SUPABASE_URL=""))
