"""Tests for WebFetchTool — URL fetching with SSRF protection and content extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.tools.web_fetch import WebFetchTool, _strip_html_tags, _truncate_at_paragraph
from augmentum.utils.safe_http import SafeHttpError


class TestWebFetchSuccess:
    """Successful fetch scenarios."""

    async def test_fetch_returns_content(self):
        tool = WebFetchTool()
        html = "<html><body><p>Hello world</p></body></html>"
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(return_value=(html, {"url": "https://example.com", "content_type": "text/html"}))

        with patch("augmentum.tools.web_fetch._extract_with_trafilatura", return_value="Hello world"):
            result = await tool.execute(url="https://example.com")

        assert result.success is True
        assert "Hello world" in result.output
        assert "example.com" in result.output

    async def test_fetch_metadata_includes_url(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(return_value=("<p>Content</p>", {"url": "https://final.com", "content_type": "text/html"}))

        with patch("augmentum.tools.web_fetch._extract_with_trafilatura", return_value="Content"):
            result = await tool.execute(url="https://example.com")

        assert result.metadata["url"] == "https://final.com"

    async def test_fetch_truncates_long_content(self):
        tool = WebFetchTool()
        long_text = "Word " * 10000
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(return_value=(long_text, {"url": "https://example.com", "content_type": "text/html"}))

        with patch("augmentum.tools.web_fetch._extract_with_trafilatura", return_value=long_text):
            result = await tool.execute(url="https://example.com", max_chars=500)

        assert len(result.output) < len(long_text)
        assert result.metadata["truncated"] is True


class TestWebFetchSSRF:
    """SSRF blocking for private IP ranges."""

    async def test_blocks_private_10x(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(side_effect=SafeHttpError("Blocked: 10.0.0.1"))
        result = await tool.execute(url="http://10.0.0.1/secret")
        assert result.success is False
        assert "blocked" in result.error.lower()

    async def test_blocks_private_172x(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(side_effect=SafeHttpError("Blocked: 172.16.0.1"))
        result = await tool.execute(url="http://172.16.0.1/admin")
        assert result.success is False

    async def test_blocks_private_192x(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(side_effect=SafeHttpError("Blocked: 192.168.1.1"))
        result = await tool.execute(url="http://192.168.1.1/config")
        assert result.success is False

    async def test_blocks_localhost(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(side_effect=SafeHttpError("Blocked: 127.0.0.1"))
        result = await tool.execute(url="http://127.0.0.1:8080/admin")
        assert result.success is False

    async def test_blocks_link_local(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(side_effect=SafeHttpError("Blocked: 169.254.169.254"))
        result = await tool.execute(url="http://169.254.169.254/metadata")
        assert result.success is False


class TestWebFetchValidation:
    """URL validation and edge cases."""

    async def test_rejects_non_http_url(self):
        tool = WebFetchTool()
        result = await tool.execute(url="ftp://example.com/file")
        assert result.success is False
        assert "http" in result.error.lower()

    async def test_validate_input_accepts_http(self):
        tool = WebFetchTool()
        assert tool.validate_input(url="https://example.com") is True
        assert tool.validate_input(url="http://example.com") is True

    async def test_validate_input_rejects_garbage(self):
        tool = WebFetchTool()
        assert tool.validate_input(url="not a url") is False
        assert tool.validate_input(url="") is False

    async def test_fetch_error_returns_failure(self):
        tool = WebFetchTool()
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(side_effect=Exception("Connection reset"))
        result = await tool.execute(url="https://example.com")
        assert result.success is False
        assert "failed" in result.error.lower()


class TestContentExtraction:
    """HTML stripping and trafilatura fallback."""

    async def test_trafilatura_fallback_to_strip(self):
        tool = WebFetchTool()
        html = "<html><body><script>evil()</script><p>Good content</p></body></html>"
        tool._safe_client = AsyncMock()
        tool._safe_client.fetch = AsyncMock(return_value=(html, {"url": "https://example.com", "content_type": "text/html"}))

        with patch("augmentum.tools.web_fetch._extract_with_trafilatura", return_value=None):
            result = await tool.execute(url="https://example.com")

        assert result.success is True
        assert "Good content" in result.output
        assert "evil()" not in result.output

    def test_strip_html_tags_removes_scripts(self):
        html = "<p>Hello</p><script>alert('xss')</script><p>World</p>"
        text = _strip_html_tags(html)
        assert "alert" not in text
        assert "Hello" in text
        assert "World" in text

    def test_truncate_at_paragraph_short_text(self):
        text = "Short text."
        assert _truncate_at_paragraph(text, 1000) == text

    def test_truncate_at_paragraph_long_text(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph that is really long " + "x" * 500
        truncated = _truncate_at_paragraph(text, 60)
        assert len(truncated) <= 63  # allow for "..."
