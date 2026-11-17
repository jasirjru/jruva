"""Unit tests for WebSearchService and real-time grounding integration."""
import pytest
from unittest.mock import AsyncMock, patch

from app.schemas.chat import ChatRequest, SourceCitation
from app.services.chat_service import ChatService
from app.services.search.web_search import WebSearchService


def test_should_search_heuristics():
    service = WebSearchService()

    # Temporal / real-time intent queries
    assert service.should_search("What is the latest news today?") is True
    assert service.should_search("Champions League winner 2026") is True
    assert service.should_search("What is the current Bitcoin price?") is True
    assert service.should_search("Weather in Tokyo now") is True

    # General knowledge queries
    assert service.should_search("What is python programming?") is False
    assert service.should_search("Explain quantum physics") is False

    # Manual toggle overrides
    assert service.should_search("What is python?", manual_toggle=True) is True
    assert service.should_search("latest news", manual_toggle=False) is False


def test_parse_html_filters_ads():
    service = WebSearchService()
    sample_html = """
    <div class="result results_links">
      <a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=example.com">Ad Title</a>
      <a class="result__snippet">Ad text</a>
    </div>
    <div class="result results_links">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.spacex.com%2Flaunches&rut=123">SpaceX Launches</a>
      <a class="result__snippet">Official updates on Starship flights and tests.</a>
    </div>
    <div class="result results_links">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSpaceX&rut=456">SpaceX - Wikipedia</a>
      <a class="result__snippet">Space Exploration Technologies Corp. is an American spacecraft manufacturer.</a>
    </div>
    """

    citations = service._parse_html(sample_html, max_results=5)
    assert len(citations) == 2
    assert citations[0].title == "SpaceX Launches"
    assert citations[0].url == "https://www.spacex.com/launches"
    assert "Starship" in citations[0].snippet

    assert citations[1].title == "SpaceX - Wikipedia"
    assert citations[1].url == "https://en.wikipedia.org/wiki/SpaceX"


def test_build_grounding_context():
    service = WebSearchService()
    citations = [
        SourceCitation(
            id="src-1",
            title="Starship Flight 7",
            url="https://spacex.com/launches",
            snippet="SpaceX successfully tested Starship in 2026.",
        )
    ]
    context = service.build_grounding_context(citations)
    assert "[REAL-TIME WEB SEARCH GROUNDING]" in context
    assert "Starship Flight 7" in context
    assert "https://spacex.com/launches" in context


@pytest.mark.asyncio
async def test_chat_service_with_web_search():
    mock_citations = [
        SourceCitation(
            id="src-1",
            title="Live Tech News",
            url="https://technews.com/article",
            snippet="New AI release in 2026.",
        )
    ]

    chat_service = ChatService()
    chat_service.router.generate_response = AsyncMock(
        return_value=("Grounded response based on [1]", "groq", "openai/gpt-oss-120b", False)
    )

    with patch("app.services.chat_service.check_domain_knowledge", return_value=([], [])), \
         patch("app.services.chat_service.web_search_service.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_citations

        request = ChatRequest(
            message="What is the latest AI news in 2026?",
            web_search=True,
            stream=False,
        )

        response = await chat_service.generate_chat_response(request)

        assert len(response.sources) == 1
        assert response.sources[0].title == "Live Tech News"
        assert response.sources[0].url == "https://technews.com/article"

        # Verify router was called with grounded prompt containing [REAL-TIME WEB SEARCH GROUNDING]
        call_args = chat_service.router.generate_response.call_args[1]
        assert "[REAL-TIME WEB SEARCH GROUNDING]" in call_args["prompt"]
        assert "Live Tech News" in call_args["prompt"]
