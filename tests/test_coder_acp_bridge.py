"""ACPEditorChannel — maps EditorChannel requests onto ACP client calls.

Uses a duck-typed fake connection (no ACP SDK needed for most tests) so the
create->wait->output->release terminal dance, the fs read/write mapping, and the
timeout->kill path are all verified deterministically. One test that exercises
env-var conversion is guarded on the SDK being importable.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from augmentum.coder.acp_bridge import ACPEditorChannel
from augmentum.coder.executors import EditorError, RemoteEditorExecutor

SESSION = "sess-1"


class FakeConn:
    """Minimal async stand-in for an ACP AgentSideConnection."""

    def __init__(self, *, content="", output="", exit_code=0, exit_delay=0.0) -> None:
        self.calls: list[tuple] = []
        self.released: list[str] = []
        self.killed: list[str] = []
        self._content = content
        self._output = output
        self._exit_code = exit_code
        self._exit_delay = exit_delay
        self.last_env = None

    async def read_text_file(self, session_id, path, **kw):
        self.calls.append(("read_text_file", session_id, path))
        return SimpleNamespace(content=self._content)

    async def write_text_file(self, session_id, path, content, **kw):
        self.calls.append(("write_text_file", session_id, path, content))
        return None

    async def create_terminal(self, session_id, command, args=None, env=None,
                              cwd=None, output_byte_limit=None, **kw):
        self.calls.append(("create_terminal", session_id, command, args, cwd))
        self.last_env = env
        return SimpleNamespace(terminal_id="term-1")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kw):
        if self._exit_delay:
            await asyncio.sleep(self._exit_delay)
        return SimpleNamespace(exit_code=self._exit_code, signal=None)

    async def terminal_output(self, session_id, terminal_id, **kw):
        return SimpleNamespace(output=self._output, exit_status=self._exit_code)

    async def kill_terminal(self, session_id, terminal_id, **kw):
        self.killed.append(terminal_id)

    async def release_terminal(self, session_id, terminal_id, **kw):
        self.released.append(terminal_id)


@pytest.mark.asyncio
async def test_read_maps_to_read_text_file() -> None:
    conn = FakeConn(content="body")
    ch = ACPEditorChannel(conn, SESSION)
    assert await ch.request("fs/read_text_file", {"path": "/workspace/a.py"}) == {"content": "body"}
    assert conn.calls == [("read_text_file", SESSION, "/workspace/a.py")]


@pytest.mark.asyncio
async def test_write_maps_to_write_text_file() -> None:
    conn = FakeConn()
    ch = ACPEditorChannel(conn, SESSION)
    await ch.request("fs/write_text_file", {"path": "/workspace/a.py", "content": "x"})
    assert conn.calls == [("write_text_file", SESSION, "/workspace/a.py", "x")]


@pytest.mark.asyncio
async def test_terminal_run_full_dance_and_release() -> None:
    conn = FakeConn(output="hello\n", exit_code=0)
    ch = ACPEditorChannel(conn, SESSION)
    res = await ch.request("terminal/run", {"command": ["echo", "hi"], "cwd": "/workspace"})
    assert res == {"output": "hello\n", "exit_code": 0}
    kinds = [c[0] for c in conn.calls]
    assert kinds == ["create_terminal"]  # wait/output aren't recorded but release is
    # create split command/args correctly
    _, _, command, args, cwd = conn.calls[0]
    assert command == "echo" and args == ["hi"] and cwd == "/workspace"
    assert conn.released == ["term-1"]  # always released
    assert conn.killed == []


@pytest.mark.asyncio
async def test_terminal_run_timeout_kills_and_annotates() -> None:
    conn = FakeConn(output="partial", exit_code=0, exit_delay=0.2)
    ch = ACPEditorChannel(conn, SESSION)
    res = await ch.request("terminal/run", {"command": ["sleep", "9"], "timeout": 0.05})
    assert "timed out after 0.05s" in res["output"]
    assert res["exit_code"] is None
    assert conn.killed == ["term-1"]
    assert conn.released == ["term-1"]  # still released after kill


@pytest.mark.asyncio
async def test_unsupported_method_raises() -> None:
    ch = ACPEditorChannel(FakeConn(), SESSION)
    with pytest.raises(EditorError):
        await ch.request("fs/delete", {"path": "/x"})


@pytest.mark.asyncio
async def test_empty_command_rejected() -> None:
    ch = ACPEditorChannel(FakeConn(), SESSION)
    with pytest.raises(EditorError):
        await ch.request("terminal/run", {"command": []})


@pytest.mark.asyncio
async def test_end_to_end_executor_over_acp_channel() -> None:
    # RemoteEditorExecutor on top of the real ACP channel (fake conn beneath).
    conn = FakeConn(content="def f(): ...", output="ok\n")
    ex = RemoteEditorExecutor(ACPEditorChannel(conn, SESSION))
    assert await ex.read_file("/workspace/f.py") == "def f(): ..."
    assert await ex.run_command(["echo", "ok"]) == "ok\n"


@pytest.mark.asyncio
async def test_env_vars_converted_to_acp_envvariable() -> None:
    pytest.importorskip("acp")
    conn = FakeConn(output="")
    ch = ACPEditorChannel(conn, SESSION)
    await ch.request("terminal/run", {"command": ["env"], "environment": {"K": "V"}})
    assert conn.last_env is not None and len(conn.last_env) == 1
    ev = conn.last_env[0]
    assert ev.name == "K" and ev.value == "V"
