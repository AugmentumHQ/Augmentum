"""Tests for WebSearchTool — SearXNG integration with retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from augmentum.tools.web_search import WebSearchTool


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://searxng:8080/search"),
    )


def _make_tool(client: MagicMock | None = None) -> WebSearchTool:
    if client is None:
        client = AsyncMock()
    return WebSearchTool(http_client=client, base_url="http://searxng:8080")


class TestWebSearchRequest:
    """Request shape and parameter handling."""

    async def test_search_sends_correct_params(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(
            json_data={"results": [{"title": "Test", "url": "https://example.com", "content": "Snippet"}]}
        ))
        tool = _make_tool(client)
        await tool.execute(query="python asyncio")

        client.get.assert_called_once()
        call_kwargs = client.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params["q"] == "python asyncio"
        assert params["format"] == "json"
        assert params["categories"] == "general"

    async def test_search_custom_categories(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(
            json_data={"results": [{"title": "T", "url": "https://x.com", "content": "S"}]}
        ))
        tool = _make_tool(client)
        await tool.execute(query="physics", categories="science")

        params = client.get.call_args.kwargs.get("params") or client.get.call_args[1].get("params")
        assert params["categories"] == "science"

    async def test_search_clamps_num_results(self):
        client = AsyncMock()
        results = [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Snippet {i}"}
            for i in range(25)
        ]
        client.get = AsyncMock(return_value=_mock_response(json_data={"results": results}))
        tool = _make_tool(client)
        result = await tool.execute(query="test", num_results=25)
        # Max is 20
        assert result.metadata["num_results"] <= 20


class TestWebSearchRetry:
    """Retry logic and error handling."""

    async def test_retry_succeeds_on_second_attempt(self):
        client = AsyncMock()
        fail_resp = _mock_response(status_code=500)
        success_resp = _mock_response(json_data={
            "results": [{"title": "OK", "url": "https://ok.com", "content": "Works"}]
        })
        client.get = AsyncMock(side_effect=[
            httpx.HTTPStatusError("500", request=httpx.Request("GET", "http://x"), response=fail_resp),
            success_resp,
        ])
        tool = _make_tool(client)

        with patch("augmentum.tools.web_search.asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(query="retry test")

        assert result.success is True
        assert client.get.call_count == 2

    async def test_network_error_after_retry_returns_error(self):
        # Two consecutive network failures (initial + one retry per the
        # documented SearXNG contract — further retries fight the engine's
        # own suspension logic rather than helping).
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        tool = _make_tool(client)

        with patch("augmentum.tools.web_search.asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(query="fail test")

        assert result.success is False
        assert "search failed" in result.error.lower()
        assert client.get.call_count == 2

    async def test_empty_results_returns_no_results(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(json_data={"results": []}))
        tool = _make_tool(client)

        result = await tool.execute(query="obscure query xyz")
        assert result.success is True
        assert "no results" in result.output.lower()
        assert result.metadata["num_results"] == 0

    async def test_timeout_returns_error(self):
        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        tool = _make_tool(client)

        with patch("augmentum.tools.web_search.asyncio.sleep", new_callable=AsyncMock):
            result = await tool.execute(query="timeout test")

        assert result.success is False


class TestWebSearchOutput:
    """Output formatting and deduplication."""

    async def test_results_formatted_with_index(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(json_data={
            "results": [
                # Titles/snippets must share tokens with the query: the
                # zero-relevance floor refuses raw results with no query
                # overlap when the quality filter strips everything.
                {"title": "First test result", "url": "https://first.com", "content": "First test snippet"},
                {"title": "Second test result", "url": "https://second.com", "content": "Second test snippet"},
            ]
        }))
        tool = _make_tool(client)
        result = await tool.execute(query="test")
        assert "[1]" in result.output
        assert "[2]" in result.output
        assert "https://first.com" in result.output

    async def test_duplicate_urls_deduplicated(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(json_data={
            "results": [
                {"title": "A dedup", "url": "https://same.com", "content": "First dedup"},
                {"title": "B dedup", "url": "https://same.com", "content": "Duplicate dedup"},
                {"title": "C dedup", "url": "https://other.com", "content": "Third dedup"},
            ]
        }))
        tool = _make_tool(client)
        result = await tool.execute(query="dedup")
        assert result.metadata["num_results"] == 2

    async def test_empty_query_returns_error(self):
        tool = _make_tool()
        result = await tool.execute(query="   ")
        assert result.success is False
        assert "empty" in result.error.lower()

    async def test_validate_input_rejects_empty(self):
        tool = _make_tool()
        assert tool.validate_input(query="") is False
        assert tool.validate_input(query="valid query") is True

    async def test_metadata_includes_urls(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_mock_response(json_data={
            "results": [
                {"title": "X meta", "url": "https://x.com", "content": "Y meta"},
            ]
        }))
        tool = _make_tool(client)
        result = await tool.execute(query="meta")
        assert "https://x.com" in result.metadata["urls"]
