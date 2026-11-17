"""Release acceptance must identify the commit actually served by the backend."""

from app.api.v1.endpoints.health import health_check


async def test_health_reports_render_commit(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "a" * 40)
    response = await health_check()
    assert response.model_dump().get("commit") == "a" * 40
