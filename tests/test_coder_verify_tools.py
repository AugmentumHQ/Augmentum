"""Tests for the Tier 1 verification tools: browser_evaluate, http_request, db_inspect.

These tools wrap workspace-side Python subprocess scripts. Test strategy:

  - Mock ``container_manager.run_command`` to return a canned JSON line
    the helper parses. Verifies the tool layer's contract: script
    construction, JSON parsing, structured output, metadata shape.
  - Cover the validation_error paths (missing required args, refused
    actions) without invoking subprocess at all.
  - For db_inspect, exercise each action variant since they have
    branching code in the renderer.
  - For the trim helper, exec the function source standalone to side-step
    the augmentum package's runtime deps (aiosqlite etc.). Lets us pin
    the truncation invariants without spinning up a workspace.

The actual Playwright / requests / sqlite3 calls happen inside the
workspace and aren't reachable from a unit test — those are exercised
in live integration runs against a real workspace container.

Run: python -m pytest tests/test_coder_verify_tools.py -v
"""

from __future__ import annotations

import json
import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.browser import (
    _build_evaluate_wrapper,
    _trim_evaluate_result,
    playwright_screenshot,
)
from augmentum.coder.runtime_tools import (
    BrowserEvaluateTool,
    BrowserScreenshotTool,
    DbInspectTool,
    HttpRequestTool,
    _clamp_timeout,
)
from augmentum.coder.state import CoderState


def _state() -> CoderState:
    return CoderState(session_id="s", workspace_id="ws")


def _cm(run_output: str = "") -> MagicMock:
    cm = MagicMock()
    cm.run_command = AsyncMock(return_value=run_output)
    cm.file_read = AsyncMock(return_value="")
    cm.file_write = AsyncMock(return_value=None)
    return cm


# ---------------------------------------------------------------------------
# BrowserScreenshotTool / playwright_screenshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_playwright_screenshot_defaults_to_best_effort_readiness():
    payload = json.dumps({
        "ok": True,
        "playwright": True,
        "path": "/workspace/.augmentum/browser-screenshots/shot.png",
        "title": "App",
        "full_page": True,
        "degraded": False,
        "warnings": [],
        "console_errors": [],
        "network_failures": [],
    })
    cm = _cm(run_output=payload)

    result = await playwright_screenshot(cm, "ws", url="http://localhost:5173")

    assert result["ok"] is True
    run_call = cm.run_command.await_args_list[1]
    command = run_call.args[1][2]
    script = shlex.split(command)[2]
    assert "wait_until='domcontentloaded'" in script
    assert "page.goto(url, wait_until=wait_until" in script
    assert "page.wait_for_load_state('networkidle'" in script
    # The subprocess budget is now phase-derived, so the helper can return
    # structured degraded output instead of being killed at the old 25s cap.
    assert run_call.kwargs["timeout"] > 25.0


@pytest.mark.asyncio
async def test_playwright_screenshot_preserves_strict_networkidle_option():
    payload = json.dumps({
        "ok": True,
        "playwright": True,
        "path": "/workspace/.augmentum/browser-screenshots/shot.png",
        "title": "App",
        "full_page": False,
        "degraded": False,
        "warnings": [],
        "console_errors": [],
        "network_failures": [],
    })
    cm = _cm(run_output=payload)

    await playwright_screenshot(
        cm,
        "ws",
        url="http://localhost:5173",
        wait_until="networkidle",
        timeout_ms=45_000,
        full_page=False,
    )

    command = cm.run_command.await_args_list[1].args[1][2]
    script = shlex.split(command)[2]
    assert "wait_until='networkidle'" in script
    assert "networkidle_grace_ms=0" in script
    assert "full_page=False" in script


class TestBrowserScreenshotTool:
    @pytest.mark.asyncio
    async def test_degraded_capture_is_successful_and_warns(self):
        payload = json.dumps({
            "ok": True,
            "playwright": True,
            "path": "/workspace/.augmentum/browser-screenshots/shot.png",
            "title": "App",
            "full_page": False,
            "requested_full_page": True,
            "degraded": True,
            "warnings": [
                {"phase": "goto_domcontentloaded", "error": "Timeout 15000ms exceeded"},
                {"phase": "screenshot_fallback", "error": "full-page screenshot failed; viewport screenshot captured"},
            ],
            "console_errors": [{"type": "warning", "text": "layout shift"}],
            "network_failures": [],
        })
        tool = BrowserScreenshotTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws",
            state=_state(),
        )

        result = await tool.execute(url="http://localhost:5173")

        assert result.success is True
        assert result.error == ""
        assert "Screenshot captured" in result.output
        assert "viewport" in result.output
        assert "Capture degraded but usable" in result.output
        assert "goto_domcontentloaded" in result.output
        assert "layout shift" in result.output
        assert result.warnings
        assert result.metadata["browser"]["path"].endswith("shot.png")

    @pytest.mark.asyncio
    async def test_failed_capture_remains_failure(self):
        payload = json.dumps({
            "ok": False,
            "playwright": True,
            "path": "/workspace/.augmentum/browser-screenshots/shot.png",
            "error": "chromium launch failed",
            "warnings": [{"phase": "launch", "error": "browser missing"}],
            "console_errors": [],
            "network_failures": [],
        })
        tool = BrowserScreenshotTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws",
            state=_state(),
        )

        result = await tool.execute(url="http://localhost:5173")

        assert result.success is False
        assert "chromium launch failed" in result.error
        assert "Screenshot failed" in result.output
        assert result.warnings == []

# ---------------------------------------------------------------------------
# BrowserEvaluateTool
# ---------------------------------------------------------------------------


class TestBrowserEvaluateTool:
    @pytest.mark.asyncio
    async def test_missing_expression_validation_error(self):
        tool = BrowserEvaluateTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(expression="")
        assert not result.success
        assert result.validation_error
        assert "expression" in result.error

    @pytest.mark.asyncio
    async def test_no_open_url_validation_error(self):
        cm = _cm()
        # file_read returns empty (no session file), so load_browser_session
        # resolves to "" → tool should refuse.
        cm.file_read = AsyncMock(return_value="")
        tool = BrowserEvaluateTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        result = await tool.execute(expression="document.title")
        assert not result.success
        assert result.validation_error
        assert "browser_open" in result.error

    @pytest.mark.asyncio
    async def test_successful_evaluate_returns_result_json(self):
        # The Playwright subprocess prints a JSON line; the tool parses it.
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": "{\"count\": 3, \"title\": \"Home\"}",
            "result_type": "object",
            "truncated": False,
            "latency_ms": 42,
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="() => ({ count: items.length, title: document.title })",
            url="http://127.0.0.1:5173/",
        )
        assert result.success
        assert '"count": 3' in result.output
        assert '"title": "Home"' in result.output
        # result_type tag should appear in the header line.
        assert "object" in result.output.splitlines()[0]
        assert result.metadata["browser_evaluate"]["latency_ms"] == 42

    @pytest.mark.asyncio
    async def test_evaluation_failure_surfaces_error(self):
        payload = json.dumps({
            "ok": False, "playwright": True,
            "error": "ReferenceError: items is not defined",
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="items.length", url="http://127.0.0.1:5173/",
        )
        assert not result.success
        assert "ReferenceError" in result.error

    @pytest.mark.asyncio
    async def test_console_errors_surfaced_in_output(self):
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": "42", "truncated": False,
            "latency_ms": 10,
            "console_errors": [
                {"type": "error", "text": "TypeError: x is null"},
            ],
            "network_failures": [
                {"status": 404, "method": "GET", "url": "/api/missing"},
            ],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="42", url="http://127.0.0.1:5173/",
        )
        assert result.success
        assert "TypeError" in result.output
        assert "/api/missing" in result.output

    @pytest.mark.asyncio
    async def test_truncated_result_flagged(self):
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": "x" * 100,  # actual content doesn't matter for the flag
            "result_type": "string",
            "truncated": True,
            "latency_ms": 10,
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="bigBlob()", url="http://127.0.0.1:5173/",
        )
        assert result.success
        assert "truncated" in result.output.lower()
        assert "50kb" in result.output.lower()

    @pytest.mark.asyncio
    async def test_args_threaded_into_subprocess_script(self):
        """args param must reach the inline subprocess script as a JSON
        literal so the JS wrapper can bind it to `arg`. We verify by
        inspecting the bash command passed to run_command."""
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": "1", "result_type": "number",
            "truncated": False, "latency_ms": 5,
            "console_errors": [], "network_failures": [],
        })
        cm = _cm(run_output=payload)
        tool = BrowserEvaluateTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        await tool.execute(
            expression="(arg) => arg.x + arg.y",
            args={"x": 41, "y": 1},
            url="http://127.0.0.1:5173/",
        )
        # The bash command bundles the script source — args_json should be
        # embedded as a Python string literal containing the JSON form.
        bash_cmd = cm.run_command.call_args[0][1]
        joined = " ".join(bash_cmd) if isinstance(bash_cmd, list) else str(bash_cmd)
        # JSON form of the args dict; key order is preserved by json.dumps.
        assert '"x": 41' in joined
        assert '"y": 1' in joined

    @pytest.mark.asyncio
    async def test_selector_routes_to_locator_evaluate(self):
        """When selector is set the wrapper signature uses `(el, arg)` and
        the subprocess script takes the locator.evaluate branch."""
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": "\"Foo\"", "result_type": "string",
            "truncated": False, "latency_ms": 8,
            "console_errors": [], "network_failures": [],
        })
        cm = _cm(run_output=payload)
        tool = BrowserEvaluateTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        await tool.execute(
            expression="el.textContent.trim()",
            selector="h1",
            url="http://127.0.0.1:5173/",
        )
        bash_cmd = cm.run_command.call_args[0][1]
        joined = " ".join(bash_cmd) if isinstance(bash_cmd, list) else str(bash_cmd)
        # Element-scoped wrapper signature.
        assert "(el, arg)" in joined
        # Locator branch must be exercised, not page.evaluate.
        assert "page.locator(selector)" in joined
        assert "target_obj.evaluate" in joined

    @pytest.mark.asyncio
    async def test_js_error_returns_structured_detail(self):
        """JS exceptions come back as {js_error, error_detail: {message,
        name, line, column, stack}}. The output surface must show the
        location info so the model can fix it without re-running."""
        payload = json.dumps({
            "ok": False, "playwright": True,
            "js_error": True,
            "error": "items is not defined",
            "error_detail": {
                "message": "items is not defined",
                "name": "ReferenceError",
                "line": 3,
                "column": 17,
                "stack": "ReferenceError: items is not defined\n    at <anonymous>:3:17",
            },
            "latency_ms": 11,
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="items.length", url="http://127.0.0.1:5173/",
        )
        assert not result.success
        # Header must name the error type AND the location.
        assert "ReferenceError" in result.output
        assert "line 3" in result.output
        assert "col 17" in result.output
        # Error string on ToolResult should be the structured form, not
        # "Playwright evaluate failed".
        assert "ReferenceError" in result.error
        assert "items is not defined" in result.error

    @pytest.mark.asyncio
    async def test_selector_missing_returns_clean_error(self):
        payload = json.dumps({
            "ok": False, "playwright": True,
            "error": "selector not found: .never",
            "selector_missing": True,
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="el.textContent",
            selector=".never",
            url="http://127.0.0.1:5173/",
        )
        assert not result.success
        assert ".never" in result.output
        assert "did not match" in result.output.lower()
        assert "selector not found" in result.error

    @pytest.mark.asyncio
    async def test_wrapper_error_surfaces_syntax_hint(self):
        """A JS parse error breaks the wrapper itself. The subprocess
        returns wrapper_error=True; the tool layer should hint at syntax."""
        payload = json.dumps({
            "ok": False, "playwright": True,
            "error": "wrapper produced no envelope (likely a JS syntax error in expression)",
            "wrapper_error": True,
            "raw": None,
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserEvaluateTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            expression="document.title)))",
            url="http://127.0.0.1:5173/",
        )
        assert not result.success
        assert "syntax" in result.output.lower()

    @pytest.mark.asyncio
    async def test_timeouts_clamped_into_subprocess(self):
        """A model-supplied timeout below 500 / above 60_000 must be
        clamped, not crash the wait. We verify via the script payload."""
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": "1", "result_type": "number",
            "truncated": False, "latency_ms": 5,
            "console_errors": [], "network_failures": [],
        })
        cm = _cm(run_output=payload)
        tool = BrowserEvaluateTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        await tool.execute(
            expression="1",
            url="http://127.0.0.1:5173/",
            goto_timeout_ms=10,         # below min → clamps to 500
            timeout_ms=999_999,         # above max → clamps to 60_000
        )
        bash_cmd = cm.run_command.call_args[0][1]
        joined = " ".join(bash_cmd) if isinstance(bash_cmd, list) else str(bash_cmd)
        # Source literals from the f-string interpolation.
        assert "goto_timeout_ms=500" in joined
        assert "timeout_ms=60000" in joined


# ---------------------------------------------------------------------------
# Pure-Python helpers — exercised without a workspace.
# ---------------------------------------------------------------------------


class TestClampTimeout:
    def test_none_returns_default(self):
        assert _clamp_timeout(None, 15_000) == 15_000

    def test_empty_string_returns_default(self):
        assert _clamp_timeout("", 15_000) == 15_000

    def test_below_min_clamps_to_min(self):
        assert _clamp_timeout(10, 15_000) == 500

    def test_above_max_clamps_to_max(self):
        assert _clamp_timeout(999_999, 15_000) == 60_000

    def test_string_int_coerced(self):
        assert _clamp_timeout("5000", 15_000) == 5_000

    def test_garbage_falls_back_to_default(self):
        assert _clamp_timeout("nope", 15_000) == 15_000

    def test_in_range_passes_through(self):
        assert _clamp_timeout(8_000, 15_000) == 8_000


class TestBuildEvaluateWrapper:
    def test_page_scope_signature(self):
        out = _build_evaluate_wrapper("document.title", with_element=False)
        assert out.startswith("(async (arg) =>")
        assert "(el, arg)" not in out

    def test_element_scope_signature(self):
        out = _build_evaluate_wrapper("el.textContent", with_element=True)
        assert "(async (el, arg) =>" in out

    def test_user_expression_interpolated(self):
        out = _build_evaluate_wrapper("MY_USER_EXPR_42", with_element=False)
        assert "MY_USER_EXPR_42" in out

    def test_emits_structured_error_envelope(self):
        out = _build_evaluate_wrapper("x", with_element=False)
        # The catch block must surface the four diagnostic fields.
        for field in ("message", "name", "stack", "line", "column"):
            assert field in out
        # And the success envelope keys.
        for field in ("__aug_ok", "value", "type"):
            assert field in out


class TestTrimEvaluateResult:
    """The trim helper's contract: produce JSON-shape-compatible Python
    values that, after json.dumps, are well-formed and within budget."""

    def test_short_string_pass_through(self):
        assert _trim_evaluate_result("hi", 0, 2000, 50, 50, 8) == "hi"

    def test_long_string_capped_with_count_marker(self):
        s = "x" * 3000
        out = _trim_evaluate_result(s, 0, 100, 50, 50, 8)
        assert out.startswith("x" * 100)
        assert "(3000 chars)" in out

    def test_small_array_pass_through(self):
        assert _trim_evaluate_result([1, 2, 3], 0, 100, 50, 50, 8) == [1, 2, 3]

    def test_large_array_head_plus_sentinel(self):
        out = _trim_evaluate_result(list(range(200)), 0, 100, 50, 50, 8)
        # 25 head (arr_cap // 2) + 1 sentinel string
        assert len(out) == 26
        assert isinstance(out[-1], str)
        assert "175 more items" in out[-1]

    def test_small_object_pass_through(self):
        d = {"a": 1, "b": 2}
        assert _trim_evaluate_result(d, 0, 100, 50, 50, 8) == d

    def test_large_object_keeps_first_n_plus_sentinel(self):
        d = {f"k{i}": i for i in range(120)}
        out = _trim_evaluate_result(d, 0, 100, 50, 50, 8)
        assert "__augmentum_truncated_keys" in out
        assert out["__augmentum_truncated_keys"] == 70
        # First 50 kept.
        assert "k0" in out
        assert "k49" in out
        assert "k50" not in out

    def test_depth_cap_returns_sentinel_string(self):
        deep = {"a": {"a": {"a": {"a": {"a": "leaf"}}}}}
        out = _trim_evaluate_result(deep, 0, 100, 50, 50, 2)
        # Walk down 3 levels — at depth 3 we should see the sentinel.
        cur = out
        for _ in range(5):
            if isinstance(cur, str):
                break
            cur = next(iter(cur.values()))
        assert "truncated" in cur

    def test_output_is_json_parseable(self):
        """The whole point of structure-aware trimming: the trimmed
        value must encode to valid JSON and decode back without error."""
        mixed = {
            "big_str": "x" * 5_000,
            "big_arr": list(range(150)),
            "nested": {"list": ["x" * 9_999] * 80},
            "scalar": 42,
            "null": None,
        }
        trimmed = _trim_evaluate_result(mixed, 0, 100, 20, 10, 5)
        encoded = json.dumps(trimmed)
        decoded = json.loads(encoded)
        assert set(decoded.keys()) == set(mixed.keys())
        assert decoded["scalar"] == 42
        assert decoded["null"] is None
        # Encoded output is well under the budget after aggressive trim.
        assert len(encoded) < 50_000

    def test_non_container_pass_through(self):
        assert _trim_evaluate_result(42, 0, 100, 50, 50, 8) == 42
        assert _trim_evaluate_result(True, 0, 100, 50, 50, 8) is True
        assert _trim_evaluate_result(None, 0, 100, 50, 50, 8) is None
        assert _trim_evaluate_result(3.14, 0, 100, 50, 50, 8) == 3.14


# ---------------------------------------------------------------------------
# HttpRequestTool
# ---------------------------------------------------------------------------


class TestHttpRequestTool:
    @pytest.mark.asyncio
    async def test_missing_url_validation_error(self):
        tool = HttpRequestTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="")
        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_invalid_method_validation_error(self):
        tool = HttpRequestTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x/", method="TRACE")
        assert not result.success
        assert result.validation_error
        assert "method" in result.error.lower()

    @pytest.mark.asyncio
    async def test_successful_get(self):
        payload = json.dumps({
            "ok": True, "status": 200, "reason": "OK",
            "headers": {"Content-Type": "application/json", "Content-Length": "21"},
            "body": '{"status": "healthy"}',
            "body_truncated": False,
            "final_url": "http://127.0.0.1:8080/healthz",
            "latency_ms": 12,
            "error": "",
        })
        tool = HttpRequestTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://127.0.0.1:8080/healthz")
        assert result.success
        assert "200" in result.output
        assert "/healthz" in result.output
        # Header preview should surface content-type.
        assert "application/json" in result.output
        assert '"status": "healthy"' in result.output
        assert result.metadata["http_request"]["status"] == 200

    @pytest.mark.asyncio
    async def test_4xx_response_is_failure(self):
        payload = json.dumps({
            "ok": False, "status": 401, "reason": "Unauthorized",
            "headers": {"WWW-Authenticate": "Bearer realm=api"},
            "body": '{"error": "missing token"}',
            "body_truncated": False,
            "final_url": "http://127.0.0.1:8080/api/secret",
            "latency_ms": 8,
            "error": "",
        })
        tool = HttpRequestTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://127.0.0.1:8080/api/secret")
        assert not result.success
        assert "401" in result.output
        assert "WWW-Authenticate" in result.output

    @pytest.mark.asyncio
    async def test_post_with_body_and_headers(self):
        payload = json.dumps({
            "ok": True, "status": 201, "reason": "Created",
            "headers": {"Location": "/items/42"},
            "body": "",
            "body_truncated": False,
            "final_url": "http://127.0.0.1:8080/items",
            "latency_ms": 33,
            "error": "",
        })
        tool = HttpRequestTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            url="http://127.0.0.1:8080/items",
            method="POST",
            headers={"Authorization": "Bearer x", "Content-Type": "application/json"},
            body='{"name": "widget"}',
        )
        assert result.success
        assert "201" in result.output
        assert "Location" in result.output
        # The actual script should have been invoked with the body baked in.
        # We can sanity-check that by inspecting the cm.run_command call.
        cmd = result.metadata["http_request"]["final_url"]
        assert cmd == "http://127.0.0.1:8080/items"

    @pytest.mark.asyncio
    async def test_unparseable_response_handled(self):
        # When the subprocess crashes the helper returns a structured
        # dict — the tool should still produce a clean failure rather
        # than an exception.
        tool = HttpRequestTool(
            container_manager=_cm(run_output="some unexpected stderr noise"),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://nowhere.example/")
        assert not result.success
        assert result.error  # populated, not blank


# ---------------------------------------------------------------------------
# DbInspectTool
# ---------------------------------------------------------------------------


class TestDbInspectTool:
    @pytest.mark.asyncio
    async def test_missing_db_path_validation_error(self):
        tool = DbInspectTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(db_path="")
        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_invalid_action_validation_error(self):
        tool = DbInspectTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(db_path="/workspace/x.db", action="DROP")
        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_query_refuses_non_select(self):
        tool = DbInspectTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            db_path="/workspace/x.db",
            action="query",
            query="DELETE FROM users",
        )
        assert not result.success
        assert result.validation_error
        assert "read-only" in result.error.lower()

    @pytest.mark.asyncio
    async def test_query_accepts_select(self):
        payload = json.dumps({
            "ok": True, "action": "query",
            "columns": ["id", "name"],
            "rows": [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}],
            "truncated": False,
            "latency_ms": 5,
        })
        tool = DbInspectTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            db_path="/workspace/x.db",
            action="query",
            query="SELECT id, name FROM users LIMIT 10",
        )
        assert result.success
        assert "alice" in result.output
        assert "bob" in result.output

    @pytest.mark.asyncio
    async def test_query_accepts_with_clause(self):
        # WITH ... SELECT must pass the read-only gate too.
        payload = json.dumps({
            "ok": True, "action": "query",
            "columns": ["n"], "rows": [{"n": 1}],
            "truncated": False, "latency_ms": 3,
        })
        tool = DbInspectTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            db_path="/workspace/x.db",
            action="query",
            query="WITH t AS (SELECT 1 AS n) SELECT * FROM t",
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_sample_requires_table(self):
        tool = DbInspectTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            db_path="/workspace/x.db", action="sample",
        )
        assert not result.success
        assert result.validation_error
        assert "table" in result.error.lower()

    @pytest.mark.asyncio
    async def test_schema_action_renders_objects(self):
        payload = json.dumps({
            "ok": True, "action": "schema",
            "schema": [
                {"type": "table", "name": "users", "sql": "CREATE TABLE users (id INTEGER, name TEXT)"},
                {"type": "index", "name": "ix_users_name", "sql": "CREATE INDEX ix_users_name ON users(name)"},
            ],
            "latency_ms": 4,
        })
        tool = DbInspectTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(db_path="/workspace/x.db", action="schema")
        assert result.success
        assert "CREATE TABLE users" in result.output
        assert "ix_users_name" in result.output

    @pytest.mark.asyncio
    async def test_tables_action_renders_row_counts(self):
        payload = json.dumps({
            "ok": True, "action": "tables",
            "tables": [
                {"name": "users", "rows": 42},
                {"name": "orders", "rows": 0},
                {"name": "audit_log", "rows": -1},  # count failed
            ],
            "latency_ms": 7,
        })
        tool = DbInspectTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(db_path="/workspace/x.db", action="tables")
        assert result.success
        assert "users  (42 rows)" in result.output
        assert "orders  (0 rows)" in result.output
        # Count-failure renders as ?, not -1.
        assert "audit_log  (? rows)" in result.output

    @pytest.mark.asyncio
    async def test_integrity_action(self):
        payload = json.dumps({
            "ok": True, "action": "integrity",
            "integrity": ["ok"],
            "table_count": 12,
            "latency_ms": 21,
        })
        tool = DbInspectTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(db_path="/workspace/x.db", action="integrity")
        assert result.success
        assert "integrity_check: ok" in result.output
        assert "Tables: 12" in result.output

    @pytest.mark.asyncio
    async def test_db_not_found_surfaces_error(self):
        payload = json.dumps({
            "ok": False, "error": "db not found: /workspace/missing.db",
        })
        tool = DbInspectTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(db_path="/workspace/missing.db", action="schema")
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# Wave-2 browser primitives: browser_wait / browser_extract / browser_fill_form
# ---------------------------------------------------------------------------

from augmentum.coder.runtime_tools import (  # noqa: E402
    BrowserClickTool,
    BrowserExtractTool,
    BrowserFillFormTool,
    BrowserOpenTool,
    BrowserWaitTool,
)


class TestBrowserWaitTool:
    @pytest.mark.asyncio
    async def test_no_url_and_no_session_validation_error(self):
        tool = BrowserWaitTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(selector=".x")
        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_condition_met(self):
        payload = json.dumps({
            "ok": True, "playwright": True, "waited_ms": 420,
            "title": "App", "body_preview": "", "error": "",
            "console_errors": [],
        })
        tool = BrowserWaitTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://localhost:5173", selector=".results")
        assert result.success
        assert "Condition met after 420ms" in result.output
        assert "'.results' visible" in result.output

    @pytest.mark.asyncio
    async def test_timeout_returns_current_page_text(self):
        payload = json.dumps({
            "ok": False, "playwright": True, "waited_ms": 10000,
            "title": "App", "body_preview": "Still loading spinner...",
            "error": "condition not met within 10000ms: timeout",
            "console_errors": [],
        })
        tool = BrowserWaitTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://localhost:5173", text="Done")
        assert not result.success
        assert "Still loading spinner" in result.output
        assert "condition not met" in result.error


class TestBrowserExtractTool:
    @pytest.mark.asyncio
    async def test_unknown_kind_rejected(self):
        tool = BrowserExtractTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x", kind="everything")
        assert not result.success
        assert "unknown kind" in result.error

    @pytest.mark.asyncio
    async def test_attr_requires_attribute(self):
        tool = BrowserExtractTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x", kind="attr", selector="img")
        assert not result.success
        assert "attribute" in result.error

    @pytest.mark.asyncio
    async def test_links_extraction_pretty_json(self):
        payload = json.dumps({
            "ok": True, "playwright": True,
            "result_json": json.dumps([{"text": "Home", "href": "http://x/"}]),
            "result_type": "array", "truncated": False,
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserExtractTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x", kind="links")
        assert result.success
        assert "Extracted kind=links" in result.output
        assert '"href": "http://x/"' in result.output

    @pytest.mark.asyncio
    async def test_http_fallback_for_links(self):
        # First run_command: playwright missing; second: HTTP fallback.
        no_pw = json.dumps({"ok": False, "playwright": False, "error": "no module"})
        fb = json.dumps({
            "ok": True, "fallback": "http",
            "result_json": json.dumps([{"text": "Docs", "href": "http://x/docs"}]),
            "result_type": "array",
        })
        cm = _cm()
        cm.run_command = AsyncMock(side_effect=[no_pw, fb])
        tool = BrowserExtractTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x", kind="links")
        assert result.success
        assert "plain-HTTP fallback" in result.output
        assert "http://x/docs" in result.output

    @pytest.mark.asyncio
    async def test_dom_kind_without_playwright_fails_with_profile_hint(self):
        no_pw = json.dumps({"ok": False, "playwright": False, "error": "no module"})
        tool = BrowserExtractTool(
            container_manager=_cm(run_output=no_pw),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x", kind="table")
        assert not result.success
        assert "tooling profile" in result.error


class TestBrowserFillFormTool:
    @pytest.mark.asyncio
    async def test_missing_fields_validation_error_with_example(self):
        tool = BrowserFillFormTool(
            container_manager=_cm(), workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x")
        assert not result.success
        assert result.validation_error
        assert '"fields"' in result.error  # self-describing recovery

    @pytest.mark.asyncio
    async def test_fill_and_submit_success(self):
        payload = json.dumps({
            "ok": True, "playwright": True,
            "fields": [{"selector": "#email", "ok": True},
                       {"selector": "#agree", "ok": True}],
            "submitted": True, "submit_error": "", "wait_error": "",
            "title": "Dash", "body_preview": "Welcome back",
            "latency_ms": 900, "console_errors": [], "network_failures": [],
        })
        tool = BrowserFillFormTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            url="http://x",
            fields={"#email": "a@b.c", "#agree": True},
            submit="button[type=submit]",
        )
        assert result.success
        assert "Filled 2/2 fields. Submitted." in result.output
        assert "Welcome back" in result.output

    @pytest.mark.asyncio
    async def test_partial_fill_never_submits(self):
        payload = json.dumps({
            "ok": False, "playwright": True,
            "fields": [{"selector": "#email", "ok": True},
                       {"selector": "#missing", "ok": False, "error": "timeout"}],
            "submitted": False,
            "submit_error": "skipped: not all fields filled (never submits a half-filled form)",
            "wait_error": "", "title": "", "body_preview": "",
            "console_errors": [], "network_failures": [],
        })
        tool = BrowserFillFormTool(
            container_manager=_cm(run_output=payload),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(
            url="http://x", fields={"#email": "a@b.c", "#missing": "x"},
            submit="#go",
        )
        assert not result.success
        assert "FAILED" in result.output
        assert "Submit NOT clicked" in result.output


class TestBrowserClickWaitFor:
    @pytest.mark.asyncio
    async def test_wait_for_threaded_into_script(self):
        payload = json.dumps({"ok": True, "playwright": True, "status": 200,
                              "title": "t", "body_preview": ""})
        cm = _cm(run_output=payload)
        tool = BrowserClickTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x", selector="#go", wait_for="#app")
        assert result.success
        script = " ".join(str(a) for a in cm.run_command.call_args[0][1])
        assert "'#app'" in script  # wait_for_selector value

    @pytest.mark.asyncio
    async def test_wait_for_defaults_to_target_selector(self):
        payload = json.dumps({"ok": True, "playwright": True, "status": 200,
                              "title": "t", "body_preview": ""})
        cm = _cm(run_output=payload)
        tool = BrowserClickTool(
            container_manager=cm, workspace_id="ws", state=_state(),
        )
        await tool.execute(url="http://x", selector="#go")
        script = " ".join(str(a) for a in cm.run_command.call_args[0][1])
        # '#go' appears as both the click selector AND the pre-action
        # wait_for_selector (shlex quoting mangles exact = matching).
        assert script.count("#go") >= 2


class TestBrowserOpenReturnsSnapshot:
    @pytest.mark.asyncio
    async def test_open_output_includes_visible_elements(self):
        snap = json.dumps({
            "url": "http://x", "reachable_url": "http://x", "status": 200,
            "ok": True, "title": "My App",
            "summary": [{"tag": "h1", "text": "Dashboard"},
                        {"tag": "button", "text": "Save"}],
            "inputs": [], "buttons": [], "console_errors": [],
            "network_failures": [], "error": "", "fallback": "http",
            "latency_ms": 12,
        })
        tool = BrowserOpenTool(
            container_manager=_cm(run_output=snap),
            workspace_id="ws", state=_state(),
        )
        result = await tool.execute(url="http://x")
        assert result.success
        assert "Visible elements:" in result.output
        assert "h1: Dashboard" in result.output
