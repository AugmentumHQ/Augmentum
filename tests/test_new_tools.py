"""Tests for Wikipedia, YouTube transcript, and document parsing tools."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Wikipedia tool
# ---------------------------------------------------------------------------
from augmentum.tools.wikipedia import WikipediaTool


class TestWikipediaTool:
    """Tests for WikipediaTool."""

    def _make_tool(self) -> WikipediaTool:
        client = AsyncMock(spec=httpx.AsyncClient)
        return WikipediaTool(http_client=client)

    def test_name_and_category(self):
        tool = self._make_tool()
        assert tool.name == "wikipedia"
        assert tool.category.value == "search"

    def test_validate_input(self):
        tool = self._make_tool()
        assert tool.validate_input(query="Python") is True
        assert tool.validate_input(query="") is False
        assert tool.validate_input(query=123) is False

    def test_input_schema_has_required(self):
        tool = self._make_tool()
        schema = tool.input_schema
        assert "query" in schema["required"]
        assert "num_results" in schema["properties"]
        assert "full_article" in schema["properties"]

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = self._make_tool()
        result = await tool.execute(query="")
        assert result.success is False
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_no_results(self):
        tool = self._make_tool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"query": {"search": []}}
        tool._client.get = AsyncMock(return_value=mock_resp)

        result = await tool.execute(query="xyznonexistent12345")
        assert result.success is True
        assert "No Wikipedia" in result.output

    @pytest.mark.asyncio
    async def test_successful_lookup(self):
        tool = self._make_tool()

        search_resp = MagicMock()
        search_resp.raise_for_status = MagicMock()
        search_resp.json.return_value = {
            "query": {"search": [{"title": "Python (programming language)"}]}
        }

        extract_resp = MagicMock()
        extract_resp.raise_for_status = MagicMock()
        extract_resp.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "pageid": 12345,
                        "title": "Python (programming language)",
                        "extract": "Python is a high-level programming language.",
                    }
                }
            }
        }

        tool._client.get = AsyncMock(side_effect=[search_resp, extract_resp])
        result = await tool.execute(query="Python programming")
        assert result.success is True
        assert "Python" in result.output
        assert result.metadata["num_results"] == 1

    @pytest.mark.asyncio
    async def test_multiple_results(self):
        tool = self._make_tool()

        search_resp = MagicMock()
        search_resp.raise_for_status = MagicMock()
        search_resp.json.return_value = {
            "query": {"search": [{"title": "Cat"}, {"title": "Dog"}]}
        }

        extract_resp = MagicMock()
        extract_resp.raise_for_status = MagicMock()
        extract_resp.json.return_value = {
            "query": {
                "pages": {
                    "1": {"pageid": 1, "title": "Cat", "extract": "Cats are..."},
                    "2": {"pageid": 2, "title": "Dog", "extract": "Dogs are..."},
                }
            }
        }

        tool._client.get = AsyncMock(side_effect=[search_resp, extract_resp])
        result = await tool.execute(query="pets", num_results=2)
        assert result.success is True
        assert result.metadata["num_results"] == 2

    @pytest.mark.asyncio
    async def test_search_failure(self):
        tool = self._make_tool()
        tool._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        result = await tool.execute(query="test")
        assert result.success is False
        assert "search failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_num_results_clamped(self):
        tool = self._make_tool()

        search_resp = MagicMock()
        search_resp.raise_for_status = MagicMock()
        search_resp.json.return_value = {"query": {"search": []}}
        tool._client.get = AsyncMock(return_value=search_resp)

        # Should clamp to 5 max, 1 min
        await tool.execute(query="test", num_results=100)
        call_args = tool._client.get.call_args_list[0]
        assert call_args.kwargs["params"]["srlimit"] == 5

    def test_cacheable(self):
        tool = self._make_tool()
        assert tool.cacheable is True


# ---------------------------------------------------------------------------
# YouTube transcript tool
# ---------------------------------------------------------------------------
from augmentum.tools.youtube_transcript import (
    YouTubeTranscriptTool,
    _extract_video_id,
    _format_timestamp,
)


class TestYouTubeVideoIdExtraction:
    """Tests for video ID extraction from various URL formats."""

    def test_bare_id(self):
        assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_standard_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_live_url(self):
        assert _extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=60s") == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        assert _extract_video_id("https://example.com/video") is None

    def test_empty_string(self):
        assert _extract_video_id("") is None

    def test_whitespace_around_id(self):
        assert _extract_video_id("  dQw4w9WgXcQ  ") == "dQw4w9WgXcQ"


class TestTimestampFormatting:
    def test_seconds_only(self):
        assert _format_timestamp(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_timestamp(125) == "2:05"

    def test_hours(self):
        assert _format_timestamp(3661) == "1:01:01"

    def test_zero(self):
        assert _format_timestamp(0) == "0:00"


class TestYouTubeTranscriptTool:
    def test_name_and_category(self):
        tool = YouTubeTranscriptTool()
        assert tool.name == "youtube_transcript"
        assert tool.category.value == "fetch"

    def test_validate_input(self):
        tool = YouTubeTranscriptTool()
        assert tool.validate_input(video="dQw4w9WgXcQ") is True
        assert tool.validate_input(video="") is False

    def test_input_schema(self):
        tool = YouTubeTranscriptTool()
        schema = tool.input_schema
        assert "video" in schema["required"]
        assert "language" in schema["properties"]
        assert "timestamps" in schema["properties"]

    @pytest.mark.asyncio
    async def test_invalid_video_id(self):
        tool = YouTubeTranscriptTool()
        result = await tool.execute(video="not-a-valid-url")
        assert result.success is False
        assert "Could not extract" in result.error

    @pytest.mark.asyncio
    async def test_missing_package(self):
        tool = YouTubeTranscriptTool()
        with patch.dict("sys.modules", {"youtube_transcript_api": None}):
            # Force ImportError
            with patch(
                "augmentum.tools.youtube_transcript.YouTubeTranscriptTool._fetch_transcript",
                side_effect=ImportError("No module"),
            ):
                # We need a valid ID to get past the ID check
                result = await tool.execute(video="dQw4w9WgXcQ")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_successful_transcript(self):
        tool = YouTubeTranscriptTool()
        mock_data = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "This is a test", "start": 2.0, "duration": 3.0},
        ]
        # Mock the import check AND the fetch call
        mock_yt_module = MagicMock()
        with patch.dict("sys.modules", {"youtube_transcript_api": mock_yt_module}):
            with patch.object(
                YouTubeTranscriptTool,
                "_fetch_transcript",
                return_value=mock_data,
            ):
                result = await tool.execute(video="dQw4w9WgXcQ")
                assert result.success is True
                assert "[0:00] Hello world" in result.output
                assert "[0:02] This is a test" in result.output
                assert result.metadata["video_id"] == "dQw4w9WgXcQ"
                assert result.metadata["segments"] == 2

    @pytest.mark.asyncio
    async def test_transcript_without_timestamps(self):
        tool = YouTubeTranscriptTool()
        mock_data = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
        ]
        mock_yt_module = MagicMock()
        with patch.dict("sys.modules", {"youtube_transcript_api": mock_yt_module}):
            with patch.object(
                YouTubeTranscriptTool,
                "_fetch_transcript",
                return_value=mock_data,
            ):
                result = await tool.execute(video="dQw4w9WgXcQ", timestamps=False)
                assert result.success is True
                assert "Hello" in result.output
                assert "[" not in result.output

    @pytest.mark.asyncio
    async def test_empty_transcript(self):
        tool = YouTubeTranscriptTool()
        mock_yt_module = MagicMock()
        with patch.dict("sys.modules", {"youtube_transcript_api": mock_yt_module}):
            with patch.object(
                YouTubeTranscriptTool,
                "_fetch_transcript",
                return_value=[],
            ):
                result = await tool.execute(video="dQw4w9WgXcQ")
                assert result.success is True
                assert "empty" in result.output.lower()

    @pytest.mark.asyncio
    async def test_transcripts_disabled(self):
        tool = YouTubeTranscriptTool()
        mock_yt_module = MagicMock()
        with patch.dict("sys.modules", {"youtube_transcript_api": mock_yt_module}):
            with patch.object(
                YouTubeTranscriptTool,
                "_fetch_transcript",
                side_effect=Exception("TranscriptsDisabled"),
            ):
                result = await tool.execute(video="dQw4w9WgXcQ")
                assert result.success is False
                assert "disabled" in result.error.lower()

    def test_cache_settings(self):
        tool = YouTubeTranscriptTool()
        assert tool.cacheable is True
        assert tool.cache_ttl == 3600.0


# ---------------------------------------------------------------------------
# Document parsing tool
# ---------------------------------------------------------------------------
from augmentum.tools.document_parse import DocumentParseTool


class TestDocumentParseTool:
    def test_name_and_category(self):
        tool = DocumentParseTool()
        assert tool.name == "document_parse"
        assert tool.category.value == "file"

    def test_validate_input(self):
        tool = DocumentParseTool()
        assert tool.validate_input(path="file.pdf") is True
        assert tool.validate_input(path="") is False

    def test_input_schema(self):
        tool = DocumentParseTool()
        schema = tool.input_schema
        assert "path" in schema["required"]
        assert "max_chars" in schema["properties"]

    @pytest.mark.asyncio
    async def test_empty_path(self):
        tool = DocumentParseTool()
        result = await tool.execute(path="")
        assert result.success is False
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        tool = DocumentParseTool()
        result = await tool.execute(path="/nonexistent/file.pdf")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            tool = DocumentParseTool()
            result = await tool.execute(path=path)
            assert result.success is False
            assert "Unsupported" in result.error
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_parse_txt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("Hello, this is a test document.\nSecond line.")
            path = f.name
        try:
            tool = DocumentParseTool()
            result = await tool.execute(path=path)
            assert result.success is True
            assert "Hello" in result.output
            assert "Second line" in result.output
            assert result.metadata["type"] == ".txt"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_parse_csv_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("name,age\nAlice,30\nBob,25")
            path = f.name
        try:
            tool = DocumentParseTool()
            result = await tool.execute(path=path)
            assert result.success is True
            assert "Alice" in result.output
            assert result.metadata["type"] == ".csv"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_parse_md_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("# Heading\n\nSome markdown content.")
            path = f.name
        try:
            tool = DocumentParseTool()
            result = await tool.execute(path=path)
            assert result.success is True
            assert "Heading" in result.output
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_truncation(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("A" * 1000)
            path = f.name
        try:
            tool = DocumentParseTool()
            result = await tool.execute(path=path, max_chars=100)
            assert result.success is True
            assert len(result.output) < 200  # 100 + truncation marker
            assert result.metadata["truncated"] is True
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        tool = DocumentParseTool(base_dir="/tmp/safe")
        result = await tool.execute(path="../../etc/passwd")
        assert result.success is False
        assert "traversal" in result.error.lower() or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_resolution(self):
        tool = DocumentParseTool(base_dir="/tmp/safe")
        # The resolve should block this
        result = await tool.execute(path="../../../etc/passwd")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_pdf_missing_lib(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake pdf")
            path = f.name
        try:
            tool = DocumentParseTool()
            with patch.dict("sys.modules", {"pdfplumber": None}), patch(
                "augmentum.tools.document_parse._parse_pdf",
                side_effect=ImportError("pdfplumber not installed"),
            ):
                result = await tool.execute(path=path)
                assert result.success is False
                assert "pdfplumber" in result.error.lower() or "install" in result.error.lower()
        finally:
            os.unlink(path)

    def test_cache_settings(self):
        tool = DocumentParseTool()
        assert tool.cacheable is True
        assert tool.cache_ttl == 600.0


# ---------------------------------------------------------------------------
# Tool registry aliases
# ---------------------------------------------------------------------------
from augmentum.tools.registry import _TOOL_ALIASES


class TestNewToolAliases:
    """Verify aliases resolve to the correct canonical tool names."""

    def test_wikipedia_aliases(self):
        assert _TOOL_ALIASES["wiki"] == "wikipedia"
        assert _TOOL_ALIASES["wiki_search"] == "wikipedia"
        assert _TOOL_ALIASES["encyclopedia"] == "wikipedia"
        assert _TOOL_ALIASES["lookup"] == "wikipedia"

    def test_youtube_aliases(self):
        assert _TOOL_ALIASES["youtube"] == "youtube_transcript"
        assert _TOOL_ALIASES["yt"] == "youtube_transcript"
        assert _TOOL_ALIASES["transcript"] == "youtube_transcript"
        assert _TOOL_ALIASES["captions"] == "youtube_transcript"
        assert _TOOL_ALIASES["subtitles"] == "youtube_transcript"

    def test_document_parse_aliases(self):
        assert _TOOL_ALIASES["parse"] == "document_parse"
        assert _TOOL_ALIASES["parse_pdf"] == "document_parse"
        assert _TOOL_ALIASES["read_pdf"] == "document_parse"
        assert _TOOL_ALIASES["parse_docx"] == "document_parse"
        assert _TOOL_ALIASES["extract_text"] == "document_parse"
