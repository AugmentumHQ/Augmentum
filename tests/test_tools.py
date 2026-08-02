"""Comprehensive tests for the Augmentum tool framework (Phase 4)."""

from __future__ import annotations  # noqa: I001

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from augmentum.models.base import (
    ModelBackend,
)
from augmentum.modes.analytical.engine import AnalyticalEngine
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.file_ops import FileOpsTool
from augmentum.tools.math_verify import MathVerifyTool
from augmentum.tools.python_exec import PythonExecTool
from augmentum.tools.registry import ToolRegistry
from augmentum.tools.web_fetch import WebFetchTool
from augmentum.tools.web_search import WebSearchTool
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError


# =====================================================================
# Helpers
# =====================================================================


def _mock_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
    headers: dict | None = None,
) -> httpx.Response:
    """Build a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        text=text if json_data is None else None,
        headers=headers or {},
        request=httpx.Request("GET", "http://test"),
    )
    return resp


# =====================================================================
# ToolRegistry Tests
# =====================================================================


class TestToolRegistry:
    def test_register_and_get(self):
        """Registered tools should be retrievable by name."""
        registry = ToolRegistry()
        tool = MagicMock(spec=Tool)
        tool.name = "test_tool"
        tool.category = ToolCategory.SEARCH
        registry.register(tool)

        assert registry.get("test_tool") is tool

    def test_get_missing_returns_none(self):
        """Getting a non-existent tool should return None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools_all(self):
        """list_tools with no filter should return all registered tools."""
        registry = ToolRegistry()
        tool1 = MagicMock(spec=Tool)
        tool1.name = "t1"
        tool1.category = ToolCategory.SEARCH
        tool2 = MagicMock(spec=Tool)
        tool2.name = "t2"
        tool2.category = ToolCategory.EXECUTE
        registry.register(tool1)
        registry.register(tool2)

        assert len(registry.list_tools()) == 2

    def test_list_tools_by_category(self):
        """list_tools should filter by category."""
        registry = ToolRegistry()
        tool1 = MagicMock(spec=Tool)
        tool1.name = "search1"
        tool1.category = ToolCategory.SEARCH
        tool2 = MagicMock(spec=Tool)
        tool2.name = "exec1"
        tool2.category = ToolCategory.EXECUTE
        registry.register(tool1)
        registry.register(tool2)

        search_tools = registry.list_tools(category=ToolCategory.SEARCH)
        assert len(search_tools) == 1
        assert search_tools[0].name == "search1"

    def test_get_for_phase_relevant(self):
        """RELEVANT phase should return SEARCH and FETCH tools."""
        registry = ToolRegistry()
        search = MagicMock(spec=Tool)
        search.name = "ws"
        search.category = ToolCategory.SEARCH
        fetch = MagicMock(spec=Tool)
        fetch.name = "wf"
        fetch.category = ToolCategory.FETCH
        execute = MagicMock(spec=Tool)
        execute.name = "ex"
        execute.category = ToolCategory.EXECUTE
        registry.register(search)
        registry.register(fetch)
        registry.register(execute)

        tools = registry.get_for_phase("relevant")
        names = {t.name for t in tools}
        assert names == {"ws", "wf"}

    def test_get_for_phase_apply(self):
        """APPLY phase should return all tool categories (search, fetch, execute, verify, file)."""
        registry = ToolRegistry()
        for name, cat in [
            ("s", ToolCategory.SEARCH),
            ("ft", ToolCategory.FETCH),
            ("e", ToolCategory.EXECUTE),
            ("v", ToolCategory.VERIFY),
            ("f", ToolCategory.FILE),
        ]:
            t = MagicMock(spec=Tool)
            t.name = name
            t.category = cat
            registry.register(t)

        tools = registry.get_for_phase("apply")
        names = {t.name for t in tools}
        assert names == {"s", "ft", "e", "v", "f"}

    def test_get_for_phase_verify(self):
        """VERIFY phase should return VERIFY and EXECUTE tools."""
        registry = ToolRegistry()
        for name, cat in [
            ("s", ToolCategory.SEARCH),
            ("e", ToolCategory.EXECUTE),
            ("v", ToolCategory.VERIFY),
        ]:
            t = MagicMock(spec=Tool)
            t.name = name
            t.category = cat
            registry.register(t)

        tools = registry.get_for_phase("verify")
        names = {t.name for t in tools}
        assert names == {"e", "v"}

    def test_get_for_phase_assess_returns_empty(self):
        """ASSESS phase should return no tools."""
        registry = ToolRegistry()
        t = MagicMock(spec=Tool)
        t.name = "x"
        t.category = ToolCategory.SEARCH
        registry.register(t)

        assert registry.get_for_phase("assess") == []

    def test_get_for_phase_unknown_returns_empty(self):
        """Unknown phase names should return empty list."""
        registry = ToolRegistry()
        assert registry.get_for_phase("nonexistent") == []

    def test_get_for_phase_case_insensitive(self):
        """Phase lookup should be case-insensitive."""
        registry = ToolRegistry()
        t = MagicMock(spec=Tool)
        t.name = "s"
        t.category = ToolCategory.SEARCH
        registry.register(t)

        assert len(registry.get_for_phase("RELEVANT")) == 1
        assert len(registry.get_for_phase("Relevant")) == 1


# =====================================================================
# WebSearchTool Tests
# =====================================================================


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Successful SearXNG search should return formatted results."""
        searxng_response = {
            "results": [
                {
                    "title": "Example Result",
                    "url": "https://example.com",
                    "content": "This is a test snippet.",
                },
                {
                    "title": "Another Result",
                    "url": "https://other.com",
                    "content": "Another snippet.",
                },
            ]
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_httpx_response(
            status_code=200, json_data=searxng_response
        )

        tool = WebSearchTool(http_client=mock_client, base_url="http://searxng:8080")
        result = await tool.execute(query="test query", num_results=5)

        assert result.success is True
        assert "Example Result" in result.output
        assert "https://example.com" in result.output
        assert "Another Result" in result.output
        assert result.metadata["num_results"] == 2

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Search with no results should return a clear message."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_httpx_response(
            status_code=200, json_data={"results": []}
        )

        tool = WebSearchTool(http_client=mock_client)
        result = await tool.execute(query="obscure query")

        assert result.success is True
        assert "No results found" in result.output

    @pytest.mark.asyncio
    async def test_search_network_error(self):
        """Network errors should be handled gracefully."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        tool = WebSearchTool(http_client=mock_client)
        result = await tool.execute(query="test")

        assert result.success is False
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Empty query should be rejected."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = WebSearchTool(http_client=mock_client)
        result = await tool.execute(query="")

        assert result.success is False
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_search_limits_results(self):
        """Should limit results to num_results."""
        searxng_response = {
            "results": [
                {"title": f"Result {i}", "url": f"https://r{i}.com", "content": f"Snippet {i}"}
                for i in range(10)
            ]
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = _mock_httpx_response(
            status_code=200, json_data=searxng_response
        )

        tool = WebSearchTool(http_client=mock_client)
        result = await tool.execute(query="test", num_results=3)

        assert result.success is True
        assert result.metadata["num_results"] == 3
        # Should only contain results 0-2 not result 3+
        assert "Result 0" in result.output
        assert "Result 2" in result.output
        assert "Result 3" not in result.output

    def test_validate_input(self):
        """validate_input should reject empty queries."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = WebSearchTool(http_client=mock_client)
        assert tool.validate_input(query="hello") is True
        assert tool.validate_input(query="") is False
        assert tool.validate_input(query="   ") is False

    def test_tool_metadata(self):
        """Tool should have correct name, description, category."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = WebSearchTool(http_client=mock_client)
        assert tool.name == "web_search"
        assert tool.category == ToolCategory.SEARCH
        assert "SearXNG" in tool.description


# =====================================================================
# SafeHttpClient Tests
# =====================================================================


class TestSafeHttpClient:
    @pytest.mark.asyncio
    async def test_blocks_private_ip_127(self):
        """Should block 127.0.0.1 (loopback)."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("127.0.0.1")

    @pytest.mark.asyncio
    async def test_blocks_private_ip_10(self):
        """Should block 10.x.x.x (private class A)."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("10.0.0.1")

    @pytest.mark.asyncio
    async def test_blocks_private_ip_192_168(self):
        """Should block 192.168.x.x (private class C)."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("192.168.1.1")

    @pytest.mark.asyncio
    async def test_blocks_private_ip_172_16(self):
        """Should block 172.16.x.x (private class B)."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("172.16.0.1")

    @pytest.mark.asyncio
    async def test_blocks_link_local(self):
        """Should block 169.254.x.x (link-local)."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("169.254.1.1")

    @pytest.mark.asyncio
    async def test_blocks_ipv6_loopback(self):
        """Should block ::1 (IPv6 loopback)."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("::1")

    @pytest.mark.asyncio
    async def test_allows_public_ip(self):
        """Public IPs should not be blocked."""
        client = SafeHttpClient()
        # Should not raise
        await client._check_resolved_ips("8.8.8.8")

    @pytest.mark.asyncio
    async def test_allows_public_ip_93(self):
        """Another public IP should pass."""
        client = SafeHttpClient()
        await client._check_resolved_ips("93.184.216.34")

    def test_blocks_file_scheme(self):
        """file:// scheme should be rejected."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            client._validate_url("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        """ftp:// scheme should be rejected."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            client._validate_url("ftp://example.com/file")

    def test_blocks_gopher_scheme(self):
        """gopher:// scheme should be rejected."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked scheme"):
            client._validate_url("gopher://example.com")

    def test_allows_http_scheme(self):
        """http:// scheme should be allowed."""
        client = SafeHttpClient()
        hostname = client._validate_url("http://example.com/page")
        assert hostname == "example.com"

    def test_allows_https_scheme(self):
        """https:// scheme should be allowed."""
        client = SafeHttpClient()
        hostname = client._validate_url("https://example.com/page")
        assert hostname == "example.com"

    def test_rejects_empty_hostname(self):
        """URLs with no hostname should be rejected."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="no hostname"):
            client._validate_url("http://")

    @pytest.mark.asyncio
    async def test_fetch_blocks_private_hostname(self):
        """Fetching a URL that resolves to a private IP should fail."""
        client = SafeHttpClient()

        with (
            patch(
                "augmentum.utils.safe_http._resolve_hostname",
                return_value=["192.168.1.100"],
            ),
            pytest.raises(SafeHttpError, match="Blocked"),
        ):
            await client.fetch("http://evil-redirect.example.com")

    @pytest.mark.asyncio
    async def test_blocks_zero_network(self):
        """Should block 0.0.0.0/8."""
        client = SafeHttpClient()
        with pytest.raises(SafeHttpError, match="Blocked"):
            await client._check_resolved_ips("0.0.0.1")


# =====================================================================
# WebFetchTool Tests
# =====================================================================


class TestWebFetchTool:
    def test_strip_html_tags(self):
        """Basic HTML tag stripping should work."""
        from augmentum.tools.web_fetch import _strip_html_tags

        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        result = _strip_html_tags(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_strip_html_removes_script(self):
        """Script tags and content should be removed."""
        from augmentum.tools.web_fetch import _strip_html_tags

        html = '<html><body><script>alert("xss")</script><p>Safe text</p></body></html>'
        result = _strip_html_tags(html)
        assert "alert" not in result
        assert "Safe text" in result

    def test_truncate_at_paragraph(self):
        """Truncation should prefer paragraph boundaries."""
        from augmentum.tools.web_fetch import _truncate_at_paragraph

        text = "First paragraph here.\n\nSecond paragraph.\n\nThird paragraph is very long indeed."
        result = _truncate_at_paragraph(text, 45)
        # Should cut at the paragraph boundary before max_chars
        assert len(result) <= 45
        assert "First paragraph here." in result

    def test_truncate_short_text(self):
        """Text shorter than max_chars should be returned as-is."""
        from augmentum.tools.web_fetch import _truncate_at_paragraph

        text = "Short."
        result = _truncate_at_paragraph(text, 100)
        assert result == "Short."

    def test_tool_metadata(self):
        """Tool should have correct name, description, category."""
        tool = WebFetchTool()
        assert tool.name == "web_fetch"
        assert tool.category == ToolCategory.FETCH

    def test_validate_input(self):
        """validate_input should reject non-http URLs."""
        tool = WebFetchTool()
        assert tool.validate_input(url="https://example.com") is True
        assert tool.validate_input(url="http://example.com") is True
        assert tool.validate_input(url="ftp://example.com") is False
        assert tool.validate_input(url="not-a-url") is False

    @pytest.mark.asyncio
    async def test_rejects_non_http_url(self):
        """Execute should reject non-http URLs."""
        tool = WebFetchTool()
        result = await tool.execute(url="ftp://example.com/file")

        assert result.success is False
        assert "http" in result.error.lower()


# =====================================================================
# PythonExecTool Tests
# =====================================================================


class TestPythonExecTool:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Successful code execution should return output."""
        executor_response = {
            "success": True,
            "stdout": "Hello, World!",
            "stderr": "",
            "return_value": None,
            "error": None,
            "metrics": {"elapsed_seconds": 0.1},
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_httpx_response(
            status_code=200, json_data=executor_response
        )

        tool = PythonExecTool(http_client=mock_client)
        result = await tool.execute(code='print("Hello, World!")')

        assert result.success is True
        assert "Hello, World!" in result.output

    @pytest.mark.asyncio
    async def test_execution_with_return_value(self):
        """Code with a return value should include it in output."""
        executor_response = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_value": "42",
            "error": None,
            "metrics": {"elapsed_seconds": 0.05},
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_httpx_response(
            status_code=200, json_data=executor_response
        )

        tool = PythonExecTool(http_client=mock_client)
        result = await tool.execute(code="21 * 2")

        assert result.success is True
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_execution_error(self):
        """Failed code execution should return error details."""
        executor_response = {
            "success": False,
            "stdout": "",
            "stderr": "NameError: name 'x' is not defined",
            "return_value": None,
            "error": "NameError: name 'x' is not defined",
            "metrics": {"elapsed_seconds": 0.01},
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_httpx_response(
            status_code=200, json_data=executor_response
        )

        tool = PythonExecTool(http_client=mock_client)
        result = await tool.execute(code="print(x)")

        assert result.success is False
        assert "NameError" in result.output

    @pytest.mark.asyncio
    async def test_executor_unavailable(self):
        """Should handle executor connection errors gracefully."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        tool = PythonExecTool(http_client=mock_client)
        result = await tool.execute(code="1+1")

        assert result.success is False
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_code_rejected(self):
        """Empty code should be rejected."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = PythonExecTool(http_client=mock_client)
        result = await tool.execute(code="")

        assert result.success is False
        assert "No code" in result.error

    def test_tool_metadata(self):
        """Tool should have correct name, description, category."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = PythonExecTool(http_client=mock_client)
        assert tool.name == "python_exec"
        assert tool.category == ToolCategory.EXECUTE


# =====================================================================
# MathVerifyTool Tests
# =====================================================================


class TestMathVerifyTool:
    @pytest.mark.asyncio
    async def test_numeric_simple_eval(self):
        """Basic numeric expression should evaluate correctly."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="2 + 3")

        assert result.success is True
        assert "5" in result.output

    @pytest.mark.asyncio
    async def test_numeric_with_expected_match(self):
        """Matching expected value should report YES."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="6 * 7", expected="42")

        assert result.success is True
        assert "YES" in result.output
        assert result.metadata["match"] is True

    @pytest.mark.asyncio
    async def test_numeric_with_expected_mismatch(self):
        """Non-matching expected value should report NO."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="6 * 7", expected="43")

        assert result.success is True
        assert "NO" in result.output
        assert result.metadata["match"] is False

    @pytest.mark.asyncio
    async def test_numeric_math_functions(self):
        """Math functions like sqrt should be available."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="sqrt(144)")

        assert result.success is True
        assert "12" in result.output

    @pytest.mark.asyncio
    async def test_numeric_constants(self):
        """Math constants pi and e should be available."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="pi", expected="3.14159265358979")

        assert result.success is True
        assert result.metadata["match"] is True

    @pytest.mark.asyncio
    async def test_numeric_division(self):
        """Division should work."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="10 / 4", expected="2.5")

        assert result.success is True
        assert result.metadata["match"] is True

    @pytest.mark.asyncio
    async def test_numeric_negative_numbers(self):
        """Negative numbers should work."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="-5 + 3", expected="-2")

        assert result.success is True
        assert result.metadata["match"] is True

    @pytest.mark.asyncio
    async def test_numeric_invalid_expression(self):
        """Invalid expressions should return an error."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="import os")

        assert result.success is False
        assert "Could not evaluate" in result.error

    @pytest.mark.asyncio
    async def test_empty_expression_rejected(self):
        """Empty expression should be rejected."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="")

        assert result.success is False
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_unknown_verify_type(self):
        """Unknown verify_type should be rejected."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="1+1", verify_type="unknown")

        assert result.success is False
        assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_symbolic_fallback_to_numeric(self):
        """Symbolic mode should fall back to numeric when executor is unavailable."""
        tool = MathVerifyTool(http_client=None)
        result = await tool.execute(
            expression="2 + 3", expected="5", verify_type="symbolic"
        )

        # Falls back to numeric — should still succeed.
        assert result.success is True
        assert result.metadata["match"] is True

    @pytest.mark.asyncio
    async def test_symbolic_with_executor(self):
        """Symbolic mode with working executor should parse SymPy output."""
        executor_response = {
            "success": True,
            "stdout": "simplified: 5\nexpected: 5\nmatch: True",
            "stderr": "",
            "return_value": None,
            "error": None,
            "metrics": {"elapsed_seconds": 0.2},
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = _mock_httpx_response(
            status_code=200, json_data=executor_response
        )

        tool = MathVerifyTool(http_client=mock_client)
        result = await tool.execute(
            expression="2 + 3", expected="5", verify_type="symbolic"
        )

        assert result.success is True
        assert "YES" in result.output

    @pytest.mark.asyncio
    async def test_equation_requires_expected(self):
        """Equation verification without expected should fail."""
        tool = MathVerifyTool()
        result = await tool.execute(expression="x**2 - 4", verify_type="equation")

        assert result.success is False
        assert "expected" in result.error.lower()

    def test_tool_metadata(self):
        """Tool should have correct name, description, category."""
        tool = MathVerifyTool()
        assert tool.name == "math_verify"
        assert tool.category == ToolCategory.VERIFY


# =====================================================================
# FileOpsTool Tests
# =====================================================================


class TestFileOpsTool:
    def test_path_traversal_blocked_dotdot(self):
        """../../../etc/passwd must be blocked."""
        tool = FileOpsTool(base_dir="/data/workdir")
        with pytest.raises(ValueError, match="traversal"):
            tool._resolve_safe_path("../../../etc/passwd")

    def test_path_traversal_blocked_absolute(self):
        """Absolute paths outside base_dir must be blocked."""
        tool = FileOpsTool(base_dir="/data/workdir")
        with pytest.raises(ValueError, match="traversal"):
            tool._resolve_safe_path("/etc/passwd")

    def test_path_traversal_blocked_complex(self):
        """Complex traversal like subdir/../../.. must be blocked."""
        tool = FileOpsTool(base_dir="/data/workdir")
        with pytest.raises(ValueError, match="traversal"):
            tool._resolve_safe_path("subdir/../../../etc/shadow")

    def test_safe_path_resolved(self):
        """A normal relative path should resolve under base_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            resolved = tool._resolve_safe_path("test.txt")
            assert str(resolved).startswith(str(Path(tmp).resolve()))

    def test_safe_path_subdirectory(self):
        """Subdirectory paths should resolve under base_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            resolved = tool._resolve_safe_path("sub/dir/file.txt")
            assert str(resolved).startswith(str(Path(tmp).resolve()))

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        """Should be able to write a file and read it back."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)

            write_result = await tool.execute(
                operation="write", path="hello.txt", content="Hello, tools!"
            )
            assert write_result.success is True

            read_result = await tool.execute(operation="read", path="hello.txt")
            assert read_result.success is True
            assert read_result.output == "Hello, tools!"

    @pytest.mark.asyncio
    async def test_write_creates_subdirectories(self):
        """Writing to a nested path should create parent directories."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)

            result = await tool.execute(
                operation="write",
                path="a/b/c/deep.txt",
                content="Deep file",
            )
            assert result.success is True

            read = await tool.execute(operation="read", path="a/b/c/deep.txt")
            assert read.success is True
            assert read.output == "Deep file"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        """Reading a missing file should return an error."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            result = await tool.execute(operation="read", path="nope.txt")

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """Listing a directory should show its contents."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            # Create some files
            (Path(tmp) / "a.txt").write_text("a")
            (Path(tmp) / "b.txt").write_text("b")
            (Path(tmp) / "subdir").mkdir()

            result = await tool.execute(operation="list", path=".")

            assert result.success is True
            assert "a.txt" in result.output
            assert "b.txt" in result.output
            assert "[dir] subdir" in result.output

    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Checking an existing file should report it exists."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            (Path(tmp) / "present.txt").write_text("here")

            result = await tool.execute(operation="exists", path="present.txt")
            assert result.success is True
            assert "Exists" in result.output

    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Checking a missing file should report it does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)

            result = await tool.execute(operation="exists", path="missing.txt")
            assert result.success is True
            assert "Does not exist" in result.output

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        """Unknown operations should be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            result = await tool.execute(operation="delete", path="x.txt")

            assert result.success is False
            assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_path_traversal_blocked_in_execute(self):
        """Path traversal should be blocked at the execute level."""
        with tempfile.TemporaryDirectory() as tmp:
            tool = FileOpsTool(base_dir=tmp)
            result = await tool.execute(operation="read", path="../../../etc/passwd")

            assert result.success is False
            assert "traversal" in result.error.lower()

    def test_tool_metadata(self):
        """Tool should have correct name, description, category."""
        tool = FileOpsTool()
        assert tool.name == "file_ops"
        assert tool.category == ToolCategory.FILE


# =====================================================================
# ToolResult Tests
# =====================================================================


class TestToolResult:
    def test_default_values(self):
        """ToolResult should have sensible defaults."""
        result = ToolResult(success=True)
        assert result.output == ""
        assert result.error == ""
        assert result.metadata == {}

    def test_with_all_fields(self):
        """All fields should be settable."""
        result = ToolResult(
            success=False,
            output="some output",
            error="something went wrong",
            metadata={"key": "value"},
        )
        assert result.success is False
        assert result.output == "some output"
        assert result.error == "something went wrong"
        assert result.metadata == {"key": "value"}


# =====================================================================
# ToolCategory Tests
# =====================================================================


class TestToolCategory:
    def test_category_values(self):
        """All category values should be correct strings."""
        assert ToolCategory.SEARCH == "search"
        assert ToolCategory.FETCH == "fetch"
        assert ToolCategory.EXECUTE == "execute"
        assert ToolCategory.VERIFY == "verify"
        assert ToolCategory.FILE == "file"

    def test_category_is_string(self):
        """Categories should be usable as strings."""
        assert isinstance(ToolCategory.SEARCH, str)


# =====================================================================
# ToolRegistry.resolve() — fuzzy name matching
# =====================================================================


class TestToolRegistryResolve:
    """Tests for fuzzy tool name resolution."""

    def _make_registry(self) -> ToolRegistry:
        """Build a registry with realistic tool names."""
        registry = ToolRegistry()
        for name, cat in [
            ("web_search", ToolCategory.SEARCH),
            ("web_fetch", ToolCategory.FETCH),
            ("python_exec", ToolCategory.EXECUTE),
            ("math_verify", ToolCategory.VERIFY),
            ("calculator", ToolCategory.VERIFY),
            ("datetime", ToolCategory.VERIFY),
            ("unit_converter", ToolCategory.VERIFY),
            ("file_ops", ToolCategory.FILE),
            ("text_analysis", ToolCategory.VERIFY),
            ("json_tool", ToolCategory.VERIFY),
            ("hash", ToolCategory.VERIFY),
        ]:
            t = MagicMock(spec=Tool)
            t.name = name
            t.category = cat
            registry.register(t)
        return registry

    def test_exact_match(self):
        """Exact name should resolve directly."""
        r = self._make_registry()
        assert r.resolve("web_search").name == "web_search"
        assert r.resolve("calculator").name == "calculator"

    def test_case_insensitive(self):
        """Upper/mixed case should resolve."""
        r = self._make_registry()
        assert r.resolve("Web_Search").name == "web_search"
        assert r.resolve("CALCULATOR").name == "calculator"
        assert r.resolve("Python_Exec").name == "python_exec"

    def test_alias_search(self):
        """Common aliases should resolve to web_search."""
        r = self._make_registry()
        assert r.resolve("search").name == "web_search"
        assert r.resolve("websearch").name == "web_search"
        assert r.resolve("google").name == "web_search"

    def test_alias_fetch(self):
        """Aliases for web_fetch."""
        r = self._make_registry()
        assert r.resolve("fetch").name == "web_fetch"
        assert r.resolve("webfetch").name == "web_fetch"

    def test_alias_python(self):
        """Aliases for python_exec."""
        r = self._make_registry()
        assert r.resolve("python").name == "python_exec"
        assert r.resolve("exec").name == "python_exec"
        assert r.resolve("code").name == "python_exec"
        assert r.resolve("run_python").name == "python_exec"

    def test_alias_calculator(self):
        """Aliases for calculator."""
        r = self._make_registry()
        assert r.resolve("calc").name == "calculator"
        assert r.resolve("calculate").name == "calculator"

    def test_alias_datetime(self):
        """Aliases for datetime."""
        r = self._make_registry()
        assert r.resolve("date").name == "datetime"
        assert r.resolve("time").name == "datetime"

    def test_alias_math_verify(self):
        """Aliases for math_verify."""
        r = self._make_registry()
        assert r.resolve("math").name == "math_verify"
        assert r.resolve("sympy").name == "math_verify"

    def test_markdown_stripping(self):
        """Markdown formatting should be stripped."""
        r = self._make_registry()
        assert r.resolve("`web_search`").name == "web_search"
        assert r.resolve("**calculator**").name == "calculator"
        assert r.resolve("'python_exec'").name == "python_exec"

    def test_hyphen_variant(self):
        """Hyphenated names should resolve."""
        r = self._make_registry()
        assert r.resolve("web-search").name == "web_search"
        assert r.resolve("python-exec").name == "python_exec"
        assert r.resolve("math-verify").name == "math_verify"

    def test_unknown_returns_none(self):
        """Completely unknown names should return None."""
        r = self._make_registry()
        assert r.resolve("nonexistent_tool_xyz") is None

    def test_substring_match(self):
        """Substring of a registered name should resolve."""
        r = self._make_registry()
        # "text_anal" is a substring of "text_analysis"
        assert r.resolve("text_anal").name == "text_analysis"


# =====================================================================
# _parse_tool_call — robust parsing for small models
# =====================================================================


class TestParseToolCall:
    """Tests for the robust tool-call parser."""

    def test_canonical_format(self):
        """Standard TOOL_CALL / TOOL_INPUT should parse correctly."""
        output = (
            "I need to search for this.\n"
            "TOOL_CALL: web_search\n"
            'TOOL_INPUT: {"query": "weather in NYC"}'
        )
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {"query": "weather in NYC"}

    def test_case_insensitive_tool_call(self):
        """tool_call and Tool Call should also work."""
        for variant in [
            "tool_call: web_search\ntool_input: {\"query\": \"test\"}",
            "Tool Call: web_search\nTool Input: {\"query\": \"test\"}",
            "Tool_Call: web_search\nTool_Input: {\"query\": \"test\"}",
        ]:
            name, inp = AnalyticalEngine._parse_tool_call(variant)
            assert name == "web_search", f"Failed for: {variant!r}"
            assert inp == {"query": "test"}, f"Input failed for: {variant!r}"

    def test_markdown_bold_tool_call(self):
        """**TOOL_CALL:** should still parse."""
        output = "**TOOL_CALL:** web_search\n**TOOL_INPUT:** {\"query\": \"test\"}"
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {"query": "test"}

    def test_backtick_wrapped_name(self):
        """Tool name in backticks should parse."""
        output = "TOOL_CALL: `web_search`\nTOOL_INPUT: {\"query\": \"test\"}"
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"

    def test_single_quote_json(self):
        """Single-quoted JSON should be recovered."""
        output = "TOOL_CALL: web_search\nTOOL_INPUT: {'query': 'weather today'}"
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {"query": "weather today"}

    def test_no_tool_call_returns_empty(self):
        """No tool call in output should return empty."""
        name, inp = AnalyticalEngine._parse_tool_call(
            "Just some normal text without tool calls."
        )
        assert name == ""
        assert inp == {}

    def test_tool_name_with_trailing_punctuation(self):
        """Tool name with trailing period/comma should be cleaned."""
        output = "TOOL_CALL: web_search.\nTOOL_INPUT: {\"query\": \"test\"}"
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"

    def test_json_after_tool_call_no_explicit_input_line(self):
        """JSON appearing after TOOL_CALL without TOOL_INPUT label."""
        output = "TOOL_CALL: web_search\n{\"query\": \"weather\"}"
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {"query": "weather"}

    def test_multiline_json(self):
        """Multi-line JSON should be parsed correctly."""
        output = (
            "TOOL_CALL: web_search\n"
            "TOOL_INPUT: {\n"
            '  "query": "weather in Seattle WA",\n'
            '  "num_results": 5\n'
            "}"
        )
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp["query"] == "weather in Seattle WA"
        assert inp["num_results"] == 5

    def test_trailing_comma_json(self):
        """JSON with trailing comma should be recovered."""
        output = 'TOOL_CALL: calculator\nTOOL_INPUT: {"expression": "2+2",}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "calculator"
        assert inp == {"expression": "2+2"}

    def test_using_tool_variant(self):
        """'Using tool: name' should also parse."""
        output = 'Using tool: web_search\nTOOL_INPUT: {"query": "test"}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"

    def test_tool_colon_variant(self):
        """'Tool: name' should also parse."""
        output = 'Tool: calculator\nTOOL_INPUT: {"expression": "1+1"}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "calculator"

    def test_empty_input_returns_empty_dict(self):
        """TOOL_CALL without TOOL_INPUT returns empty dict."""
        output = "TOOL_CALL: web_search"
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {}

    def test_hyphenated_tool_call(self):
        """TOOL-CALL with hyphen should parse."""
        output = 'TOOL-CALL: web_search\nTOOL-INPUT: {"query": "test"}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {"query": "test"}

    def test_equals_separator(self):
        """TOOL_CALL = name (equals instead of colon)."""
        output = 'TOOL_CALL = web_search\nTOOL_INPUT = {"query": "test"}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp == {"query": "test"}

    def test_placeholder_tool_name_rejected(self):
        """Literal 'tool_name' from prompt template should be rejected."""
        output = 'TOOL_CALL: tool_name\nTOOL_INPUT: {"param": "value"}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == ""
        assert inp == {}

    def test_placeholder_angle_bracket_name_rejected(self):
        """<tool_name> from prompt template should be rejected."""
        output = 'TOOL_CALL: <tool_name>\nTOOL_INPUT: {"param": "value"}'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == ""
        assert inp == {}

    def test_key_value_input_parsing(self):
        """key = value format after TOOL_INPUT should parse."""
        output = 'TOOL_CALL: web_search\nTOOL_INPUT:\nquery: weather today\nnum_results: 3'
        name, inp = AnalyticalEngine._parse_tool_call(output)
        assert name == "web_search"
        assert inp.get("query") == "weather today"


# =====================================================================
# _execute_tool protections (extra kwargs, placeholders)
# =====================================================================


class TestExecuteToolProtections:
    """Tests for the runtime protections in _execute_tool."""

    @pytest.mark.asyncio
    async def test_extra_kwargs_stripped(self):
        """Extra kwargs not in the schema should be stripped, not crash."""
        # Build a mock tool with known schema
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "web_fetch"
        mock_tool.input_schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["url"],
        }
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="page content",
        ))

        # Create engine with registry
        backend = MagicMock(spec=ModelBackend)
        registry = ToolRegistry()
        registry.register(mock_tool)
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        # Call with extra kwargs (query, parameters) that aren't in schema
        result = await engine._execute_tool(
            "apply", "web_fetch",
            {"url": "https://example.com", "query": "test", "parameters": ["x"]},
        )

        assert result.success is True
        # Verify execute was called WITHOUT the extra kwargs
        mock_tool.execute.assert_called_once_with(url="https://example.com")

    @pytest.mark.asyncio
    async def test_placeholder_values_rejected(self):
        """Placeholder values like '<code>' should be rejected."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "python_exec"
        mock_tool.input_schema = {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        }

        backend = MagicMock(spec=ModelBackend)
        registry = ToolRegistry()
        registry.register(mock_tool)
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "python_exec", {"code": "<code>"},
        )

        assert result.success is False
        assert "placeholder" in result.error.lower()
        # execute should NOT have been called
        mock_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_angle_bracket_placeholder_rejected(self):
        """Any <something> value for a required field should be rejected."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "web_search"
        mock_tool.input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

        backend = MagicMock(spec=ModelBackend)
        registry = ToolRegistry()
        registry.register(mock_tool)
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "web_search", {"query": "<my_search_term>"},
        )

        assert result.success is False
        assert "placeholder" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fuzzy_name_resolution_in_execute(self):
        """Fuzzy names like 'search' should resolve to 'web_search'."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "web_search"
        mock_tool.category = ToolCategory.SEARCH
        mock_tool.input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="results",
        ))

        backend = MagicMock(spec=ModelBackend)
        registry = ToolRegistry()
        registry.register(mock_tool)
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "search", {"query": "weather today"},
        )

        assert result.success is True
        mock_tool.execute.assert_called_once_with(query="weather today")

    @pytest.mark.asyncio
    async def test_unknown_tool_lists_available(self):
        """Unknown tool error should list all available tools."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "web_search"
        mock_tool.category = ToolCategory.SEARCH

        backend = MagicMock(spec=ModelBackend)
        registry = ToolRegistry()
        registry.register(mock_tool)
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "nonexistent_xyz", {"query": "test"},
        )

        assert result.success is False
        assert "web_search" in result.error
        assert "Available tools:" in result.error


# =====================================================================
# get_phase_prompt with has_tools flag
# =====================================================================


class TestPhasePromptToolAwareness:
    """Tests for tool-first instructions in phase prompts."""

    def test_apply_simple_has_tools_includes_tool_nudge(self):
        """When has_tools=True, APPLY simple prompt should include tool directive."""
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system_prompt, user_content = get_phase_prompt(
            "apply", query="weather today", assess_output="simple",
            is_simple=True, has_tools=True,
        )
        assert "tool" in system_prompt.lower()
        assert "call a tool first" in system_prompt.lower()

    def test_apply_simple_no_tools_no_tool_nudge(self):
        """When has_tools=False, APPLY simple prompt should NOT include tool nudge."""
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system_prompt, user_content = get_phase_prompt(
            "apply", query="weather today", assess_output="simple",
            is_simple=True, has_tools=False,
        )
        assert "call a tool first" not in system_prompt.lower()

    def test_apply_full_has_tools(self):
        """Full APPLY prompt with has_tools=True includes tool directive."""
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system_prompt, user_content = get_phase_prompt(
            "apply", query="test", identify_output="test",
            relevant_output="test", has_tools=True,
        )
        assert "tool" in system_prompt.lower()

    def test_relevant_has_tools(self):
        """RELEVANT prompt with has_tools=True includes tool directive."""
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system_prompt, user_content = get_phase_prompt(
            "relevant", query="test", identify_output="test",
            has_tools=True,
        )
        assert "tool" in system_prompt.lower()

    def test_assess_ignores_has_tools(self):
        """ASSESS prompt should not be affected by has_tools."""
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system_prompt, user_content = get_phase_prompt("assess", query="test", has_tools=True)
        assert "call a tool first" not in system_prompt.lower()

    def test_conclude_ignores_has_tools(self):
        """CONCLUDE prompt should not be affected by has_tools."""
        from augmentum.modes.analytical.prompts import get_phase_prompt

        system_prompt, user_content = get_phase_prompt(
            "conclude", query="test", apply_output="test",
            verify_output="test", has_tools=True,
        )
        assert "call a tool first" not in system_prompt.lower()


# =====================================================================
# _strip_system_echo utility
# =====================================================================


class TestStripSystemEcho:
    """Tests for stripping echoed system prompt from phase output."""

    def test_normal_output_unchanged(self):
        """Output that doesn't start with echo markers is returned as-is."""
        from augmentum.modes.analytical.handler import _strip_system_echo

        output = "ERRORS_FOUND:\n- None\n\nVERIFIED: yes\nCONFIDENCE: 0.9"
        assert _strip_system_echo(output) == output

    def test_echoed_verify_prompt_stripped(self):
        """Echoed verify system prompt should be stripped to content."""
        from augmentum.modes.analytical.handler import _strip_system_echo

        output = (
            "You are an analytical verification engine. Your task is to check...\n\n"
            "## Your Goal\nVerify the analysis...\n\n"
            "ERRORS_FOUND:\n- None\n\nVERIFIED: yes\nCONFIDENCE: 0.85"
        )
        result = _strip_system_echo(output)
        assert result.startswith("ERRORS_FOUND:")
        assert "verification engine" not in result

    def test_echoed_instructions_stripped(self):
        """Echoed ## Instructions header should be stripped."""
        from augmentum.modes.analytical.handler import _strip_system_echo

        output = (
            "## Instructions\n1. Check each step...\n\n"
            "ERRORS_FOUND:\n- None\nVERIFIED: yes"
        )
        result = _strip_system_echo(output)
        assert result.startswith("ERRORS_FOUND:")

    def test_empty_output_unchanged(self):
        """Empty output should be returned as-is."""
        from augmentum.modes.analytical.handler import _strip_system_echo

        assert _strip_system_echo("") == ""
        assert _strip_system_echo("   ") == "   "

    def test_echo_without_content_marker_unchanged(self):
        """If echo detected but no content marker found, return as-is."""
        from augmentum.modes.analytical.handler import _strip_system_echo

        output = "You are an analytical engine. Here is some freeform text."
        assert _strip_system_echo(output) == output
