"""Tests for the self-edit engine connector (fills app.state.selfedit_driver).

Fakes for the native loop + platform select, so deterministic. Load-bearing:
  - native engine → a working EditDriver over a local-model loop (no token);
  - native engine without a loop → None (debt loop stays a safe dry-run);
  - claude_code/codex → the registry-selected driver, or None when unavailable;
  - the assembled driver actually runs (proves the debt loop would fix via it).
"""

from __future__ import annotations

from augmentum.coder.external.base import CoderEvent
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.engine_select import (
    ENGINE_NATIVE,
    build_selfedit_driver,
    wire_selfedit_driver,
)
from augmentum.selfedit.orchestrator import EditRequest


def _native_loop(events):
    async def loop(_task):
        for ev in events:
            yield ev
    return loop


def _req():
    return EditRequest(
        candidate=Candidate(name="a1", path="/tmp/wt", branch="selfedit/a1",
                            base_ref="HEAD", base_sha="abc"),
        objective="fix the orphan", attempt_id="a1", user_id="u1")


async def test_native_engine_builds_runnable_driver():
    loop = _native_loop([
        {"kind": "tool_call", "tool": "edit_file", "args": {"file_path": "augmentum/x.py"}},
        {"kind": "completed", "text": "removed the orphan"},
    ])
    driver = await build_selfedit_driver(conn=None, engine=ENGINE_NATIVE, native_loop=loop)
    assert driver is not None
    res = await driver(_req())                          # the debt loop would call this
    assert res.ok is True and res.final_text == "removed the orphan"


async def test_native_engine_without_loop_is_none():
    driver = await build_selfedit_driver(conn=None, engine=ENGINE_NATIVE, native_loop=None)
    assert driver is None                              # → debt loop stays a safe dry-run


async def test_claude_engine_uses_registry_select():
    class _FakeDriver:
        id = "claude_code"
        async def is_available(self):
            return True
        async def run(self, task):
            yield CoderEvent(kind="completed", text="patched")

    async def _select(prefer, *, cwd, claude_oauth_token, claude_api_key):
        assert prefer == "claude_code"
        return _FakeDriver()

    driver = await build_selfedit_driver(conn=None, engine="claude_code",
                                         oauth_token="sk-ant-oat01-x", _select=_select)
    assert driver is not None
    res = await driver(_req())
    assert res.ok is True and res.final_text == "patched"


async def test_platform_unavailable_is_none():
    async def _select(prefer, *, cwd, claude_oauth_token, claude_api_key):
        return None

    driver = await build_selfedit_driver(conn=None, engine="codex", _select=_select)
    assert driver is None                              # nothing runnable → safe dry-run


class _AppState:
    pass


async def test_wire_sets_driver_and_repo_when_live():
    app = _AppState()
    loop = _native_loop([{"kind": "completed", "text": "ok"}])
    live = await wire_selfedit_driver(app, None, engine=ENGINE_NATIVE,
                                      native_loop=loop, repo_dir="/repo")
    assert live is True
    assert app.selfedit_driver is not None             # green-lane now enabled
    assert app.selfedit_repo_dir == "/repo"


async def test_wire_leaves_dry_run_when_no_engine():
    app = _AppState()
    live = await wire_selfedit_driver(app, None, engine=ENGINE_NATIVE, native_loop=None)
    assert live is False
    assert app.selfedit_driver is None                 # debt stays a safe dry-run


# --- native engine self-constructs from just a registry (sovereign, no token) ---

class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, msg):
        self.message = msg
        self.model = "m"


class _Backend:
    def __init__(self, msg):
        self._msg = msg

    async def chat(self, _req):
        return _Resp(self._msg)


class _Reg:
    def __init__(self, msg):
        self._b = _Backend(msg)

    async def resolve_model_for_role(self, role, override="", settings=None):
        return self._b, "m"


async def test_native_engine_self_constructs_from_registry():
    # backend returns a `finish` tool call → the self-built loop completes.
    reg = _Reg(_Msg(tool_calls=[{"id": "1", "function": {
        "name": "finish", "arguments": '{"summary":"clean"}'}}]))
    driver = await build_selfedit_driver(conn=None, engine=ENGINE_NATIVE, registry=reg)
    assert driver is not None                          # sovereign path wired from the model list
    res = await driver(_req())
    assert res.ok is True and res.final_text == "clean"
