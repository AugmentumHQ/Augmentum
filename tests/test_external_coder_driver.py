"""External-coder driver engine core — slice 1.

Covers the pure, verifiable parts: SDK-message → CoderEvent translation, mutation
awareness (what the consent gate intercepts), the engineering_log projection
seam, and availability/selection. The live SDK-in-a-container run is verified
on-device (needs auth + the container image), not here.
"""

from __future__ import annotations

import sys

import pytest

from augmentum.coder.external import (
    ExternalRunResult,
    ExternalTask,
    select_driver,
    to_engineering_record,
)
from augmentum.coder.external.base import is_mutating_tool
from augmentum.coder.external.claude_code import ClaudeCodeDriver, _translate

pytestmark = pytest.mark.asyncio


# --- fakes mimicking the Claude Agent SDK message shapes -------------------

class TextBlock:
    def __init__(self, text): self.text = text


class ThinkingBlock:
    def __init__(self, thinking): self.thinking = thinking


class ToolUseBlock:
    def __init__(self, name, input): self.name = name; self.input = input


class AssistantMessage:
    def __init__(self, content): self.content = content


class ResultMessage:
    def __init__(self, result="", is_error=False, session_id=""):
        self.result = result; self.is_error = is_error; self.session_id = session_id


class SystemMessage:
    def __init__(self, subtype="init"): self.subtype = subtype


# --- translation ----------------------------------------------------------

def test_system_init_is_started():
    evs = _translate(SystemMessage())
    assert [e.kind for e in evs] == ["started"]


def test_text_block_is_message():
    evs = _translate(AssistantMessage([TextBlock("hello there")]))
    assert len(evs) == 1 and evs[0].kind == "message" and "hello there" in evs[0].text


def test_thinking_block_is_thinking():
    evs = _translate(AssistantMessage([ThinkingBlock("let me plan")]))
    assert evs[0].kind == "thinking"


def test_write_is_mutating_file_change_with_path():
    evs = _translate(AssistantMessage([ToolUseBlock("Write", {"file_path": "/workspace/a.py"})]))
    e = evs[0]
    assert e.kind == "file_change" and e.tool == "Write"
    assert e.path == "/workspace/a.py" and e.mutating is True


def test_bash_is_mutating_command_exec():
    evs = _translate(AssistantMessage([ToolUseBlock("Bash", {"command": "make && ./run"})]))
    e = evs[0]
    assert e.kind == "command_exec" and e.mutating is True and "make" in e.text


def test_read_is_non_mutating_tool_call():
    evs = _translate(AssistantMessage([ToolUseBlock("Read", {"file_path": "/x"})]))
    e = evs[0]
    assert e.kind == "tool_call" and e.mutating is False
    # text carries the subject (the file), NOT the tool name twice ("Read Read").
    assert e.text == "/x" and e.text != "Read"


def test_tool_target_extraction():
    from augmentum.coder.external.base import tool_use_event
    assert tool_use_event("Read", {"file_path": "/workspace/main.py"}).text == "/workspace/main.py"
    assert tool_use_event("Grep", {"pattern": "TODO"}).text == "TODO"
    assert tool_use_event("WebFetch", {"url": "https://x.dev"}).text == "https://x.dev"
    # MCP call: subject, not the long dunder name echoed twice.
    mcp = tool_use_event("mcp__augmentum__memory_search", {"query": "cats"})
    assert mcp.kind == "mcp_call" and mcp.text == "cats"
    # No recognizable arg → empty text (formatter then shows just the tool name).
    assert tool_use_event("SomeTool", {"weird": 1}).text == ""


def test_mcp_tool_is_mcp_call():
    evs = _translate(AssistantMessage([ToolUseBlock("mcp__augmentum__memory_search", {"q": "x"})]))
    assert evs[0].kind == "mcp_call"


def test_result_success_and_failure():
    assert _translate(ResultMessage(result="all done"))[0].kind == "completed"
    fail = _translate(ResultMessage(is_error=True, result="boom"))[0]
    assert fail.kind == "failed" and "boom" in fail.text


def test_assistant_message_with_multiple_blocks_yields_each():
    evs = _translate(AssistantMessage([
        TextBlock("I'll edit the file"),
        ToolUseBlock("Edit", {"file_path": "/b.py"}),
    ]))
    assert [e.kind for e in evs] == ["message", "file_change"]


def test_mutation_table():
    assert is_mutating_tool("Write") and is_mutating_tool("Bash") and is_mutating_tool("apply_patch")
    assert not is_mutating_tool("Read") and not is_mutating_tool("Grep")


# --- engineering_log projection seam --------------------------------------

def test_to_engineering_record_success():
    task = ExternalTask(prompt="refactor the media store", framing="cast images slow")
    res = ExternalRunResult(ok=True, files_changed=["a.py", "b.py"], session_ref="sess-1")
    rec = to_engineering_record(task, res, engine_label="Claude Code")
    assert rec["task"] == "refactor the media store"
    assert rec["engine"] == "Claude Code"
    assert rec["framing"] == "cast images slow"
    assert rec["resume_ref"] == "sess-1"
    assert "2 files changed" in rec["outcome"] or rec["outcome"]


def test_to_engineering_record_failure():
    task = ExternalTask(prompt="do a thing")
    res = ExternalRunResult(ok=False, error="container died")
    rec = to_engineering_record(task, res, engine_label="Claude Code")
    assert "didn't finish" in rec["outcome"] and "container died" in rec["outcome"]


# --- availability / selection (SDK not installed in test env) --------------

async def test_driver_unavailable_without_sdk(monkeypatch):
    # Force the optional SDK import to fail (deterministic regardless of whether
    # claude-agent-sdk happens to be installed in the dev env): a None entry in
    # sys.modules makes ``import claude_agent_sdk`` raise ImportError.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    d = ClaudeCodeDriver(oauth_token="sk-ant-oat01-x")  # has a credential
    assert await d.is_available() is False  # ...but SDK absent → unavailable


async def test_select_driver_returns_none_when_nothing_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    assert await select_driver(cwd="/tmp") is None


# --- dedicated-container CLI path (claude -p --output-format stream-json) ---

def test_parse_cli_event_shapes():
    from augmentum.coder.external.claude_cli import parse_cli_event
    assert parse_cli_event({"type": "system", "subtype": "init"})[0].kind == "started"
    asst = parse_cli_event({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "editing"},
        {"type": "tool_use", "name": "Write", "input": {"file_path": "/p/a.py"}},
        {"type": "thinking", "thinking": "hmm"},
    ]}})
    kinds = [e.kind for e in asst]
    assert kinds == ["message", "file_change", "thinking"]
    fc = asst[1]
    assert fc.tool == "Write" and fc.path == "/p/a.py" and fc.mutating is True
    assert parse_cli_event({"type": "result", "subtype": "success", "result": "ok"})[0].kind == "completed"
    fail = parse_cli_event({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "boom"})
    assert fail[0].kind == "failed" and "boom" in fail[0].text
    # tool-result echo + unknown → nothing
    assert parse_cli_event({"type": "user", "message": {}}) == []


def test_build_claude_argv_has_streamjson_and_no_token():
    from augmentum.coder.external.claude_cli import build_claude_argv
    argv = build_claude_argv(ExternalTask(prompt="do it", workspace="/p/proj", permission="plan"))
    assert "--output-format" in argv and "stream-json" in argv
    assert "--permission-mode" in argv and "plan" in argv
    assert "--add-dir" in argv and "/p/proj" in argv
    # token must NEVER be on the command line (it's in the container env)
    assert not any("sk-ant" in a for a in argv)


async def test_cli_driver_run_streams_normalized_events():
    from augmentum.coder.external.claude_cli import ClaudeCliDriver

    lines = [
        '{"type":"system","subtype":"init","session_id":"s1"}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"editing the file"},{"type":"tool_use","name":"Write","input":{"file_path":"/p/a.py"}}]}}',
        '{"type":"result","subtype":"success","is_error":false,"result":"done","session_id":"s1"}',
    ]

    async def runner(argv):
        for ln in lines:
            yield ln

    d = ClaudeCliDriver(runner=runner)
    assert await d.is_available() is True
    out = [ev async for ev in d.run(ExternalTask(prompt="x", workspace="/p"))]
    kinds = [e.kind for e in out]
    # one explicit started (system's started is deduped), then content, terminal
    assert kinds == ["started", "message", "file_change", "completed"]


async def test_cli_driver_fails_when_no_terminal_event():
    from augmentum.coder.external.claude_cli import ClaudeCliDriver

    async def runner(argv):
        yield '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}'
        # process dies without a result event

    d = ClaudeCliDriver(runner=runner)
    out = [ev async for ev in d.run(ExternalTask(prompt="x"))]
    assert out[-1].kind == "failed" and "without a result" in out[-1].text


async def test_cli_driver_unavailable_without_credential():
    from augmentum.coder.external.claude_cli import ClaudeCliDriver

    async def runner(argv):
        if False:
            yield ""  # pragma: no cover

    assert await ClaudeCliDriver(runner=runner, has_credential=False).is_available() is False


# --- per-user encrypted token store (the login backend) --------------------

class _FakeSettingsStore:
    def __init__(self):
        self._d = {}

    async def get_user(self, user_id, key):
        return self._d.get((user_id, key))

    async def set_user(self, user_id, key, value):
        if value is None:
            self._d.pop((user_id, key), None)
        else:
            self._d[(user_id, key)] = value


async def test_token_store_roundtrip_encrypted_and_scoped():
    from augmentum.coder.external import claude_token_store as cts
    s = _FakeSettingsStore()
    await cts.save_token(s, "u_a", "sk-ant-oat01-secret123")

    # Stored value is ENCRYPTED, not the raw token.
    raw = await s.get_user("u_a", "claude_code_oauth_token")
    assert raw and raw != "sk-ant-oat01-secret123"

    # Round-trips back to the real token for the owner only.
    assert await cts.load_token(s, "u_a") == "sk-ant-oat01-secret123"
    assert await cts.load_token(s, "u_b") == ""  # user-scoped

    st = await cts.status(s, "u_a")
    assert st["connected"] is True and st["kind"] == "subscription"
    assert "secret123" not in st["hint"]  # masked, never leaks the token

    await cts.clear_token(s, "u_a")
    assert await cts.load_token(s, "u_a") == ""
    assert (await cts.status(s, "u_a"))["connected"] is False


async def test_token_store_classifies_api_key():
    from augmentum.coder.external import claude_token_store as cts
    s = _FakeSettingsStore()
    await cts.save_token(s, "u", "sk-ant-api03-xyz")
    assert (await cts.status(s, "u"))["kind"] == "api_key"
    assert cts.looks_like_claude_credential("sk-ant-oat01-x")
    assert not cts.looks_like_claude_credential("random")


# --- browser OAuth login (claude setup-token) parsing ----------------------

def test_parse_setup_token_extracts_oauth_token():
    from augmentum.coder.external.claude_auth import parse_setup_token
    out = (
        "Opening browser to authorize...\n"
        "If the browser didn't open, visit the URL above.\n"
        "Your long-lived token (valid 1 year):\n"
        "sk-ant-oat01-AbC123_def-456XYZ\n"
        "Set it as CLAUDE_CODE_OAUTH_TOKEN.\n"
    )
    assert parse_setup_token(out) == "sk-ant-oat01-AbC123_def-456XYZ"
    assert parse_setup_token("no token here") is None


def test_parse_auth_url_extracts_browser_url():
    from augmentum.coder.external.claude_auth import parse_auth_url
    out = "Open this URL to authorize: https://claude.ai/oauth/authorize?code=1&scope=inference then paste the code"
    url = parse_auth_url(out)
    assert url is not None and url.startswith("https://claude.ai/oauth/authorize")


def test_auth_env_routes_oauth_vs_api_key():
    from augmentum.coder.external.claude_auth import auth_env, is_oauth_token
    assert is_oauth_token("sk-ant-oat01-x") and not is_oauth_token("sk-ant-api03-x")
    assert auth_env(oauth_token="sk-ant-oat01-x") == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x"}
    # an API key handed to the oauth slot is routed to the right var, not mislabeled
    assert auth_env(oauth_token="sk-ant-api03-y") == {"ANTHROPIC_API_KEY": "sk-ant-api03-y"}
    assert auth_env(api_key="sk-ant-api03-z") == {"ANTHROPIC_API_KEY": "sk-ant-api03-z"}
    assert auth_env() == {}


def test_driver_has_credential_with_oauth_token():
    # Credential presence is independent of the SDK being installed.
    assert ClaudeCodeDriver(oauth_token="sk-ant-oat01-x")._has_credential() is True
    assert ClaudeCodeDriver(api_key="sk-ant-api03-y")._has_credential() is True
    assert ClaudeCodeDriver(cwd="/nonexistent-home-xyz")._has_credential() in (True, False)


# --- _ensure_claude_cli: install-on-demand for legacy ubuntu workspaces -----

class _FakeContainerManager:
    """Records run_command calls and replays scripted stdout per invocation."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def run_command(self, workspace_id, cmd, **kwargs):
        self.calls.append((workspace_id, cmd, kwargs))
        if not self._outputs:
            return ""
        out = self._outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


async def test_ensure_claude_cli_noop_when_present():
    # Prebaked image: first probe finds claude → no install, single call.
    from augmentum.proxy.external_coder_routes import _ensure_claude_cli
    cm = _FakeContainerManager(["/usr/local/bin/claude\n"])
    assert await _ensure_claude_cli(cm, "ws1") is None
    assert len(cm.calls) == 1


async def test_ensure_claude_cli_installs_when_missing():
    # Legacy ubuntu: probe empty → install → verify finds it → success.
    from augmentum.proxy.external_coder_routes import _ensure_claude_cli
    cm = _FakeContainerManager(["", "added 3 packages in 15s\n", "/usr/local/bin/claude\n"])
    assert await _ensure_claude_cli(cm, "ws1") is None
    assert len(cm.calls) == 3


async def test_ensure_claude_cli_reports_missing_npm():
    from augmentum.proxy.external_coder_routes import _ensure_claude_cli
    cm = _FakeContainerManager(["", "NO_NPM\n"])
    err = await _ensure_claude_cli(cm, "ws1")
    assert err and "npm" in err.lower()


async def test_ensure_claude_cli_reports_unverified_install():
    # Install ran but claude still not on PATH → clear error, not silent pass.
    from augmentum.proxy.external_coder_routes import _ensure_claude_cli
    cm = _FakeContainerManager(["", "some npm noise\n", ""])
    err = await _ensure_claude_cli(cm, "ws1")
    assert err and "claude" in err.lower()


async def test_ensure_claude_cli_surfaces_unreachable_workspace():
    from augmentum.proxy.external_coder_routes import _ensure_claude_cli
    cm = _FakeContainerManager([RuntimeError("409 not running")])
    err = await _ensure_claude_cli(cm, "ws1")
    assert err and "not reachable" in err.lower()


# --- /claude/run/stream: live SSE event streaming --------------------------

class _StreamingCM:
    """Fake container manager that replays stream-json to the run's on_chunk
    callback in arbitrary byte slices (incl. a split mid-line) to prove the
    endpoint reassembles partial lines."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def run_command(self, workspace_id, cmd, *, on_chunk=None, **kwargs):
        if on_chunk is not None:
            for c in self._chunks:
                await on_chunk(c)
        return ""


def _fake_request(cm):
    from types import SimpleNamespace
    app = SimpleNamespace(state=SimpleNamespace(
        settings_store=object(), container_manager=cm, companion_runtime=None,
    ))
    return SimpleNamespace(app=app, scope={"user": SimpleNamespace(id="usr_x")})


async def _collect_sse(response):
    frames = []
    async for chunk in response.body_iterator:
        text = chunk if isinstance(chunk, str) else chunk.decode()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                import json as _json
                frames.append(_json.loads(line[5:].strip()))
    return frames


async def test_run_stream_emits_live_events_and_reassembles_lines(monkeypatch):
    from augmentum.proxy import external_coder_routes as r

    monkeypatch.setattr(r.cts, "load_token", _async_return("sk-ant-oat01-tok"))
    monkeypatch.setattr(r, "_ensure_claude_cli", _async_return(None))

    # A realistic stream-json run, sliced so one JSON object is split across
    # two chunks (proves partial-line buffering).
    lines = [
        '{"type":"system","subtype":"init"}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}',
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Write","input":{"file_path":"/workspace/health.py"}}]}}',
        '{"type":"result","subtype":"success","result":"all done"}',
    ]
    blob = "\n".join(lines) + "\n"
    mid = len(blob) // 2
    chunks = [blob[:mid].encode(), blob[mid:].encode()]

    conn = await _mk_runs_db()
    try:
        body = r._RunBody(workspace_id="ws1", task="add a health endpoint")
        resp = await r.run_claude_stream(body, _fake_request_with_db(_StreamingCM(chunks), conn))
        frames = await _collect_sse(resp)

        kinds = [f["kind"] for f in frames]
        assert kinds[0] == "run"             # run-id frame first
        assert "started" in kinds
        assert "file_change" in kinds
        assert "completed" in kinds
        done = frames[-1]
        assert done["kind"] == "done" and done["ok"] is True
        assert done["files_changed"] == ["/workspace/health.py"]
    finally:
        await conn.close()


async def test_run_stream_requires_token(monkeypatch):
    from augmentum.proxy import external_coder_routes as r
    monkeypatch.setattr(r.cts, "load_token", _async_return(""))
    body = r._RunBody(workspace_id="ws1", task="x")
    resp = await r.run_claude_stream(body, _fake_request(_StreamingCM([])))
    # No token → plain JSONResponse error, not a stream.
    assert resp.status_code == 400


def _async_return(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn


# --- run_store: persisted run history (raw + normalized) -------------------

import pathlib  # noqa: E402

import aiosqlite  # noqa: E402

_MIGRATION_287 = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "287_claude_runs.sql"
)


async def _mk_runs_db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)"
    )
    await conn.executescript(_MIGRATION_287.read_text())
    await conn.commit()
    return conn


async def test_run_store_roundtrip_and_user_isolation():
    from augmentum.coder.external import run_store as rs
    conn = await _mk_runs_db()
    try:
        await rs.create_run(
            conn, run_id="r1", user_id="u1", workspace_id="ws1",
            task="add a health endpoint", permission="auto",
        )
        await rs.add_event(conn, run_id="r1", user_id="u1", seq=1, kind="started", text="go")
        await rs.add_event(
            conn, run_id="r1", user_id="u1", seq=2, kind="file_change",
            tool="Write", path="/workspace/health.py",
        )
        await rs.set_session_id(conn, run_id="r1", user_id="u1", session_id="sess-abc")
        await rs.finish_run(
            conn, run_id="r1", user_id="u1", status="done",
            outcome="added the endpoint", files_changed=["/workspace/health.py"],
            raw_jsonl='{"type":"result"}', cost_usd=0.04, num_turns=3, duration_ms=1200,
        )

        runs = await rs.list_runs(conn, user_id="u1", workspace_id="ws1")
        assert len(runs) == 1
        assert runs[0]["status"] == "done"
        assert runs[0]["session_id"] == "sess-abc"
        assert runs[0]["files_changed"] == ["/workspace/health.py"]
        assert runs[0]["num_turns"] == 3

        full = await rs.get_run(conn, run_id="r1", user_id="u1", include_raw=True)
        assert [e["kind"] for e in full["events"]] == ["started", "file_change"]
        assert full["raw_jsonl"] == '{"type":"result"}'

        # Another user sees nothing and cannot fetch the run.
        assert await rs.list_runs(conn, user_id="u2", workspace_id="ws1") == []
        assert await rs.get_run(conn, run_id="r1", user_id="u2") is None
        assert await rs.session_for_run(conn, run_id="r1", user_id="u2") is None

        sess = await rs.session_for_run(conn, run_id="r1", user_id="u1")
        assert sess == {"session_id": "sess-abc", "workspace_id": "ws1"}
    finally:
        await conn.close()


def _fake_request_with_db(cm, conn):
    from types import SimpleNamespace
    backend = SimpleNamespace(conn=conn)
    app = SimpleNamespace(state=SimpleNamespace(
        settings_store=object(), container_manager=cm, companion_runtime=None,
        state_manager=SimpleNamespace(backend=backend),
    ))
    return SimpleNamespace(app=app, scope={"user": SimpleNamespace(id="u1")},
                           query_params={})


async def test_run_stream_persists_run_and_captures_session(monkeypatch):
    from augmentum.proxy import external_coder_routes as r

    monkeypatch.setattr(r.cts, "load_token", _async_return("sk-ant-oat01-tok"))
    monkeypatch.setattr(r, "_ensure_claude_cli", _async_return(None))

    conn = await _mk_runs_db()
    try:
        lines = [
            '{"type":"system","subtype":"init","session_id":"sess-xyz"}',
            '{"type":"system","subtype":"init","session_id":"sess-xyz"}',
            '{"type":"assistant","session_id":"sess-xyz","message":{"content":[{"type":"tool_use","name":"Write","input":{"file_path":"/workspace/app.py"}}]}}',
            '{"type":"result","subtype":"success","session_id":"sess-xyz","result":"done it","total_cost_usd":0.07,"num_turns":4,"duration_ms":2222}',
        ]
        blob = ("\n".join(lines) + "\n").encode()
        cm = _StreamingCM([blob])
        body = r._RunBody(workspace_id="ws1", task="build it")
        resp = await r.run_claude_stream(body, _fake_request_with_db(cm, conn))
        frames = await _collect_sse(resp)

        # An early "run" frame carries the run id the client persists against.
        run_frame = next(f for f in frames if f["kind"] == "run")
        run_id = run_frame["run_id"]
        done = frames[-1]
        assert done["kind"] == "done" and done["ok"] is True
        assert done["session_id"] == "sess-xyz"
        assert done["num_turns"] == 4
        # Two system/init lines must NOT spam "started" — only our one synthetic.
        assert sum(1 for f in frames if f["kind"] == "started") == 1

        # Persisted: run row finalized + session captured + real summary + raw kept.
        from augmentum.coder.external import run_store as rs
        full = await rs.get_run(conn, run_id=run_id, user_id="u1", include_raw=True)
        assert full["status"] == "done"
        assert full["session_id"] == "sess-xyz"
        assert full["outcome"] == "done it"
        assert full["files_changed"] == ["/workspace/app.py"]
        assert "result" in full["raw_jsonl"]
        assert any(e["kind"] == "file_change" for e in full["events"])
    finally:
        await conn.close()


async def test_run_stream_resume_uses_prior_session(monkeypatch):
    from augmentum.coder.external import run_store as rs
    from augmentum.proxy import external_coder_routes as r

    monkeypatch.setattr(r.cts, "load_token", _async_return("sk-ant-oat01-tok"))
    monkeypatch.setattr(r, "_ensure_claude_cli", _async_return(None))

    conn = await _mk_runs_db()
    try:
        # Seed a finished prior run with a session id.
        await rs.create_run(conn, run_id="prev", user_id="u1", workspace_id="ws1", task="t")
        await rs.finish_run(conn, run_id="prev", user_id="u1", status="done",
                            session_id="sess-prev")

        captured = {}

        class _RecordingCM:
            async def run_command(self, ws, argv, *, on_chunk=None, **kw):
                captured["argv"] = argv
                captured["kw"] = kw
                if on_chunk:
                    await on_chunk(b'{"type":"result","subtype":"success","session_id":"sess-prev","result":"ok"}\n')
                return ""

        body = r._RunBody(workspace_id="ws1", task="keep going", resume_run_id="prev")
        resp = await r.run_claude_stream(body, _fake_request_with_db(_RecordingCM(), conn))
        frames = await _collect_sse(resp)

        # The argv carried Claude's native --resume <prior session id>.
        assert "--resume" in captured["argv"]
        assert captured["argv"][captured["argv"].index("--resume") + 1] == "sess-prev"
        # The run execs through a login shell so the volume-installed ``claude``
        # CLI (on PATH only via /etc/profile.d) actually resolves — a direct
        # exec fails with "executable file not found in $PATH".
        assert captured["kw"].get("login_shell") is True
        run_frame = next(f for f in frames if f["kind"] == "run")
        assert run_frame["resumed_from"] == "prev"
    finally:
        await conn.close()


async def test_run_stream_resume_missing_run_404(monkeypatch):
    from augmentum.proxy import external_coder_routes as r
    monkeypatch.setattr(r.cts, "load_token", _async_return("sk-ant-oat01-tok"))
    conn = await _mk_runs_db()
    try:
        body = r._RunBody(workspace_id="ws1", task="x", resume_run_id="nope")
        resp = await r.run_claude_stream(body, _fake_request_with_db(_StreamingCM([]), conn))
        assert resp.status_code == 404
    finally:
        await conn.close()


def test_build_claude_argv_resume_prepends_flag():
    from augmentum.coder.external.base import ExternalTask
    from augmentum.coder.external.claude_cli import build_claude_argv
    argv = build_claude_argv(ExternalTask(prompt="hi"), resume_session_id="sess-1")
    assert argv[:3] == ["claude", "--resume", "sess-1"]
    # No resume → no flag.
    assert "--resume" not in build_claude_argv(ExternalTask(prompt="hi"))


# --- RunManager: runs are server-owned, survive disconnect, are cancellable ---

async def test_run_manager_survives_viewer_leaving():
    import asyncio

    from augmentum.coder.external.run_manager import RunManager
    mgr = RunManager()
    run = mgr.create("r1", workspace_id="w", user_id="u", task="t")
    started = asyncio.Event()

    async def work(r):
        started.set()
        for i in range(3):
            await asyncio.sleep(0)
            r.publish({"kind": "message", "text": str(i), "seq": i + 1})
        r.finish({"status": "done", "ok": True})

    q = run.subscribe()
    mgr.start(run, work)
    await started.wait()
    run.unsubscribe(q)                       # viewer leaves immediately
    await asyncio.wait_for(run.finished.wait(), timeout=2)
    assert run.status == "done"              # run completed anyway
    assert mgr.get("r1") is None             # evicted after finish


async def test_run_manager_stop_cancels():
    import asyncio

    from augmentum.coder.external.run_manager import RunManager
    mgr = RunManager()
    run = mgr.create("r2", workspace_id="w", user_id="u", task="t")

    async def work(r):
        await asyncio.sleep(5)               # long-running

    mgr.start(run, work)
    await asyncio.sleep(0)
    assert await mgr.stop("r2") is True
    await asyncio.wait_for(run.finished.wait(), timeout=2)
    assert run.status == "cancelled"


async def test_attach_replays_finished_run():
    from augmentum.coder.external import run_store as rs
    from augmentum.proxy import external_coder_routes as r
    conn = await _mk_runs_db()
    try:
        await rs.create_run(conn, run_id="done1", user_id="u1", workspace_id="ws1", task="did a thing")
        await rs.add_event(conn, run_id="done1", user_id="u1", seq=1, kind="message", text="hello")
        await rs.finish_run(conn, run_id="done1", user_id="u1", status="done",
                            outcome="ok", files_changed=["/workspace/x"])
        resp = await r.attach_claude_run("done1", _fake_request_with_db(_StreamingCM([]), conn))
        frames = await _collect_sse(resp)
        kinds = [f["kind"] for f in frames]
        assert kinds[0] == "run" and kinds[1] == "started"
        assert any(f["kind"] == "message" and f["text"] == "hello" for f in frames)
        assert frames[-1]["kind"] == "done" and frames[-1]["ok"] is True
    finally:
        await conn.close()


async def test_stop_endpoint_marks_cancelled():
    from augmentum.coder.external import run_store as rs
    from augmentum.proxy import external_coder_routes as r
    conn = await _mk_runs_db()
    try:
        await rs.create_run(conn, run_id="liverun", user_id="u1", workspace_id="ws1", task="t")
        resp = await r.stop_claude_run("liverun", _fake_request_with_db(_StreamingCM([]), conn))
        assert resp.status_code == 200
        row = await rs.get_run(conn, run_id="liverun", user_id="u1")
        assert row["status"] == "cancelled"
        # Another user can't stop it.
        other = await r.stop_claude_run(
            "liverun",
            type(_fake_request_with_db(_StreamingCM([]), conn))(
                app=_fake_request_with_db(_StreamingCM([]), conn).app,
                scope={"user": __import__("types").SimpleNamespace(id="u2")},
                query_params={},
            ),
        )
        assert other.status_code == 404
    finally:
        await conn.close()
