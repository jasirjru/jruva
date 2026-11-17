"""The Plugins page must not call a feature Operational when production has it switched off."""
from unittest.mock import patch

import pytest
from app.core.config import settings
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def _statuses():
    res = client.get("/api/v1/plugins")
    assert res.status_code == 200
    return {p["id"]: p["status"] for p in res.json()["plugins"]}


@pytest.mark.parametrize(("plugin_id", "flag"), [("document_ocr", "ENABLE_LOCAL_OCR"), ("image_studio", "IMAGE_GENERATION_ENABLED")])
def test_disabled_features_are_reported_unavailable(plugin_id, flag):
    with patch.object(settings, flag, False):
        assert _statuses()[plugin_id] == "unavailable"
    with patch.object(settings, flag, True):
        assert _statuses()[plugin_id] == "ready"


def test_always_on_plugins_keep_their_status():
    statuses = _statuses()
    assert statuses["web_search"] == "ready"
    assert statuses["github"] == "planned"
