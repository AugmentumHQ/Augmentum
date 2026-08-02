"""Tests for PythonExecTool — sandboxed code execution with circuit breaker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from augmentum.tools.python_exec import PythonExecTool


def _make_tool(client: MagicMock | None = None) -> PythonExecTool:
    if client is None:
        client = AsyncMock()
    return PythonExecTool(http_client=client, base_url="http://executor:5000")


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://executor:5000/execute"),
    )


class TestPythonExecSuccess:
    """Successful code execution."""

    async def test_execute_returns_stdout(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True,
            "stdout": "42\n",
            "stderr": "",
            "return_value": None,
            "metrics": {"elapsed_seconds": 0.1},
        }))
        tool = _make_tool(client)
        result = await tool.execute(code="print(6 * 7)")
        assert result.success is True
        assert "42" in result.output

    async def test_execute_returns_return_value(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_value": 42,
            "metrics": {},
        }))
        tool = _make_tool(client)
        result = await tool.execute(code="6 * 7")
        assert result.success is True
        assert "42" in result.output

    async def test_execute_sends_correct_payload(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True, "stdout": "", "stderr": "", "return_value": None, "metrics": {},
        }))
        tool = _make_tool(client)
        await tool.execute(code="print(1)", timeout=10)
        call_kwargs = client.post.call_args
        json_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert json_body["code"] == "print(1)"
        assert json_body["timeout"] == 10


class TestPythonExecBlocking:
    """Dangerous code pattern blocking."""

    async def test_blocks_os_system(self):
        tool = _make_tool()
        result = await tool.execute(code='import os\nos.system("rm -rf /")')
        assert result.success is False
        assert "rejected" in result.error.lower()

    async def test_blocks_subprocess(self):
        tool = _make_tool()
        result = await tool.execute(code='import subprocess\nsubprocess.run(["ls"])')
        assert result.success is False

    async def test_blocks_eval(self):
        tool = _make_tool()
        result = await tool.execute(code='eval("__import__(\'os\').system(\'ls\')")')
        assert result.success is False

    async def test_blocks_exec(self):
        tool = _make_tool()
        result = await tool.execute(code='exec("print(1)")')
        assert result.success is False

    async def test_blocks_open(self):
        tool = _make_tool()
        result = await tool.execute(code='f = open("/etc/passwd")')
        assert result.success is False

    async def test_blocks_dunder_import(self):
        tool = _make_tool()
        result = await tool.execute(code='__import__("os")')
        assert result.success is False

    async def test_allows_safe_code(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True, "stdout": "hello\n", "stderr": "", "return_value": None, "metrics": {},
        }))
        tool = _make_tool(client)
        result = await tool.execute(code='print("hello")')
        assert result.success is True


class TestPythonExecErrors:
    """Error handling and timeout."""

    async def test_empty_code_returns_error(self):
        tool = _make_tool()
        result = await tool.execute(code="   ")
        assert result.success is False
        assert "no code" in result.error.lower()

    async def test_executor_http_error(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        tool = _make_tool(client)
        result = await tool.execute(code="print(1)")
        assert result.success is False
        assert "executor" in result.error.lower() or "failed" in result.error.lower()

    async def test_executor_returns_error(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": False,
            "stdout": "",
            "stderr": "Traceback...",
            "error": "NameError: name 'foo' is not defined",
            "metrics": {},
        }))
        tool = _make_tool(client)
        result = await tool.execute(code="foo()")
        assert result.success is False
        assert "NameError" in result.output or "NameError" in result.error

    async def test_timeout_clamped(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True, "stdout": "", "stderr": "", "return_value": None, "metrics": {},
        }))
        tool = _make_tool(client)
        await tool.execute(code="print(1)", timeout=999)
        json_body = client.post.call_args.kwargs.get("json") or client.post.call_args[1].get("json")
        # Max is 120
        assert json_body["timeout"] <= 120


class TestPythonExecCircuitBreaker:
    """Built-in circuit breaker in PythonExecTool."""

    async def test_circuit_opens_after_threshold(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        tool = _make_tool(client)
        tool._FAILURE_THRESHOLD = 2

        await tool.execute(code="print(1)")
        await tool.execute(code="print(2)")
        # Third call should be rejected by circuit breaker
        result = await tool.execute(code="print(3)")
        assert result.success is False
        assert "circuit breaker" in result.error.lower() or "unavailable" in result.error.lower()


class TestCodeIsReviewable:
    """The executed source rides in metadata so the UI can render it.

    Code runs on the user's own machine; they must be able to review what
    ran after the fact. The transient tool_start subtitle collapses code to
    120 chars and is never persisted, so `metadata["code"]` is the only
    durable record on the UI path — including for code that was rejected or
    failed, which is when review matters most.
    """

    async def test_success_carries_untruncated_code(self):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True,
            "stdout": "ok\n",
            "stderr": "",
            "return_value": None,
            "metrics": {"elapsed_seconds": 0.1},
        }))
        source = "\n".join(f"x{i} = {i}" for i in range(200)) + "\nprint('ok')"
        result = await tool_exec(client, source)
        assert result.metadata["code"] == source
        assert result.metadata["language"] == "python"

    async def test_shorthand_reports_the_code_that_actually_ran(self):
        """Calc-wrapped input reports the wrapped source, not the bare one —
        what's shown must be what executed."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_response({
            "success": True, "stdout": "391\n", "stderr": "",
            "return_value": None, "metrics": {},
        }))
        result = await tool_exec(client, "17*23")
        assert "print(17*23)" in result.metadata["code"]

    async def test_rejected_code_still_carries_source(self):
        # Calc-wrapping runs BEFORE the blocklist, so a bare expression is
        # reported (and screened) in its wrapped form. Reporting the wrapped
        # source is the honest answer: it's what the blocklist judged.
        result = await tool_exec(AsyncMock(), "open('/etc/passwd')")
        assert result.success is False
        assert "open('/etc/passwd')" in result.metadata["code"]

    async def test_executor_failure_still_carries_source(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
        result = await tool_exec(client, "print(1)")
        assert result.success is False
        assert result.metadata["code"] == "print(1)"


async def tool_exec(client, code: str):
    return await _make_tool(client).execute(code=code)
