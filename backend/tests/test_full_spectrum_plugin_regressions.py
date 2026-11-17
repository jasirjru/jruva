"""Disabling search must prevent private prompts reaching external search providers."""
from unittest.mock import AsyncMock, patch

import pytest
from app.schemas.chat import ChatRequest, SourceCitation
from app.services.chat_service import ChatService
from app.services.workspace_store import workspace_store


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_account_search_preference_controls_external_grounding(monkeypatch, tmp_path, stream, enabled):
    monkeypatch.setattr(workspace_store, "db_path", str(tmp_path / "integrations.db"))
    workspace_store._init_sqlite_db()
    workspace_store.toggle_integration("web_search", enabled, "audit-search-owner")
    service = ChatService()
    service.router.generate_response = AsyncMock(return_value=("Synthetic final answer", "groq", "openai/gpt-oss-120b", False))

    async def synthetic_stream(**kwargs):
        yield "Synthetic final answer", "groq", "openai/gpt-oss-120b", False

    service.router.stream_response = synthetic_stream
    request = ChatRequest(message="Latest test observatory news?", web_search=True, user_id="audit-search-owner", stream=stream)
    source = SourceCitation(id="search-source", title="EXTERNAL SEARCH SENTINEL", url="https://example.org/audit", snippet="Synthetic evidence.")
    with patch("app.services.chat_service.check_user_library_knowledge", return_value=([], [])), patch(
        "app.services.chat_service.web_search_service.search", new=AsyncMock(return_value=[source])
    ):
        if stream:
            result = "".join([event async for event in service.stream_chat_response(request)])
            assert ("EXTERNAL SEARCH SENTINEL" in result) is enabled
        else:
            result = await service.generate_chat_response(request)
            assert bool(result.sources) is enabled
