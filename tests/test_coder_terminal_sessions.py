"""Tests for coder terminal sessions (PTY + pyte rendering) and the
workspace persistence bootstrap layer."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from augmentum.coder import terminal_sessions
from augmentum.coder.terminal_sessions import (
    MAX_SESSIONS_PER_WORKSPACE,
    TerminalSession,
    TerminalSessionManager,
    encode_keys,
)
from augmentum.coder.terminal_tools import (
    TermCloseTool,
    TermListTool,
    TermOpenTool,
    TermSendTool,
    TermSnapshotTool,
)

# ---------------------------------------------------------------------------
# Fakes for the aiodocker exec surface
# ---------------------------------------------------------------------------

class FakeStream:
    def __init__(self) -> None:
        self._q: asyncio.Queue = asyncio.Queue()
        self.written: list[bytes] = []
        self.closed = False

    def feed(self, data: bytes | None) -> None:
        self._q.put_nowait(data)

    async def read_out(self):
        item = await self._q.get()
        if item is None:
            return None
        return SimpleNamespace(data=item)

    async def write_in(self, data: bytes) -> None:
        self.written.append(data)

    def close(self) -> None:
        self.closed = True
        # Unblock a pending read_out so the pump exits.
        self._q.put_nowait(None)


class FakeExec:
    def __init__(self, stream: FakeStream) -> None:
        self.id = "exec-fake"
        self._stream = stream

    def start(self, detach: bool = False) -> FakeStream:
        return self._stream


class FakeContainerManager:
    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.commands: list[str | None] = []
        self.resizes: list[tuple[int, int]] = []

    async def exec_shell(self, workspace_id, *, command=None, cwd="/workspace"):
        self.commands.append(command)
        stream = FakeStream()
        self.streams.append(stream)
        return FakeExec(stream)

    async def resize_exec(self, exec_id, rows, cols):
        self.resizes.append((rows, cols))


async def _drain(session: TerminalSession) -> None:
    """Give the pump task a few ticks to consume queued output."""
    for _ in range(20):
        await asyncio.sleep(0.01)
        if session.last_output_at:
            return


# ---------------------------------------------------------------------------
# Key encoding
# ---------------------------------------------------------------------------

def test_encode_named_keys():
    assert encode_keys(["enter"]) == b"\r"
    assert encode_keys(["up", "down", "left", "right"]) == b"\x1b[A\x1b[B\x1b[D\x1b[C"
    assert encode_keys(["tab", "escape", "backspace"]) == b"\t\x1b\x7f"
    assert encode_keys(["f1", "f5"]) == b"\x1bOP\x1b[15~"


def test_encode_ctrl_and_alt_combos():
    assert encode_keys(["ctrl+c"]) == b"\x03"
    assert encode_keys(["ctrl+d"]) == b"\x04"
    assert encode_keys(["Ctrl+X"]) == b"\x18"  # case-insensitive
    assert encode_keys(["alt+x"]) == b"\x1bx"
    assert encode_keys(["alt+enter"]) == b"\x1b\r"


def test_encode_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown key"):
        encode_keys(["superkey"])


# ---------------------------------------------------------------------------
# Session rendering
# ---------------------------------------------------------------------------

async def test_session_renders_plain_output():
    cm = FakeContainerManager()
    session = TerminalSession("t1", "ws1", "echo hi", cols=40, rows=10)
    await session.start(cm)
    cm.streams[0].feed(b"hello\r\nworld\r\n")
    await _drain(session)
    snap = session.snapshot()
    assert "hello" in snap
    assert "world" in snap
    assert "[t1] running" in snap
    assert cm.commands == ["echo hi"]
    assert cm.resizes == [(10, 40)]
    await session.close()


async def test_session_renders_ansi_not_escape_codes():
    cm = FakeContainerManager()
    session = TerminalSession("t1", "ws1", "tui", cols=40, rows=10)
    await session.start(cm)
    # Clear screen, home, bold red MENU, cursor repositioning — the raw
    # bytes are escape soup; the snapshot must be the rendered grid.
    cm.streams[0].feed(
        b"\x1b[2J\x1b[H\x1b[1;31mMENU\x1b[0m\r\n> Option A\r\n  Option B"
        b"\x1b[2;1H> Option A (selected)"
    )
    await _drain(session)
    snap = session.snapshot()
    assert "MENU" in snap
    assert "> Option A (selected)" in snap
    assert "\x1b" not in snap
    assert "[2J" not in snap
    await session.close()


async def test_session_exit_detected_and_send_refused():
    cm = FakeContainerManager()
    session = TerminalSession("t1", "ws1", "true", cols=40, rows=10)
    await session.start(cm)
    cm.streams[0].feed(b"bye\r\n")
    cm.streams[0].feed(None)  # EOF — process exited
    for _ in range(50):
        await asyncio.sleep(0.01)
        if session.exited:
            break
    assert session.exited
    assert "[t1] exited" in session.snapshot()
    with pytest.raises(RuntimeError, match="exited"):
        await session.send(b"x")


async def test_session_send_writes_to_stream():
    cm = FakeContainerManager()
    session = TerminalSession("t1", "ws1", "cat", cols=40, rows=10)
    await session.start(cm)
    await session.send(b"hi" + encode_keys(["enter"]))
    assert cm.streams[0].written == [b"hi\r"]
    await session.close()


async def test_settle_with_baseline_waits_for_reaction():
    """A quiet stream must NOT satisfy a post-send settle — the wait means
    'quiet after NEW output', or a snapshot races ahead of the echo."""
    cm = FakeContainerManager()
    session = TerminalSession("t1", "ws1", "repl", cols=40, rows=10)
    await session.start(cm)
    cm.streams[0].feed(b"banner\r\n")
    await _drain(session)
    await asyncio.sleep(0.3)  # stream now long quiet

    async def echo_later():
        await asyncio.sleep(0.2)
        cm.streams[0].feed(b">>> 42\r\n")

    baseline = session.bytes_seen
    task = asyncio.create_task(echo_later())
    await session.settle(3000, baseline_bytes=baseline)
    assert session.bytes_seen > baseline  # settle waited for the reaction
    assert "42" in session.snapshot()
    await task
    await session.close()


async def test_session_scrollback_history():
    cm = FakeContainerManager()
    session = TerminalSession("t1", "ws1", "seq", cols=20, rows=5)
    await session.start(cm)
    cm.streams[0].feed("".join(f"line{i}\r\n" for i in range(12)).encode())
    await _drain(session)
    snap = session.snapshot(history_lines=5)
    assert "scrollback" in snap
    assert "line3" in snap  # scrolled off the 5-row screen, kept in history
    await session.close()


# ---------------------------------------------------------------------------
# Manager lifecycle
# ---------------------------------------------------------------------------

async def test_manager_open_close_and_name_conflict():
    cm = FakeContainerManager()
    mgr = TerminalSessionManager()
    s1 = await mgr.open(cm, "ws1", "cmd1", name="tui")
    assert mgr.get("ws1", "tui") is s1
    with pytest.raises(ValueError, match="already running"):
        await mgr.open(cm, "ws1", "cmd2", name="tui")
    # A different workspace never sees ws1's sessions.
    assert mgr.list("ws2") == []
    assert await mgr.close("ws1", "tui") is True
    assert await mgr.close("ws1", "tui") is False


async def test_manager_replaces_exited_session_under_same_name():
    cm = FakeContainerManager()
    mgr = TerminalSessionManager()
    s1 = await mgr.open(cm, "ws1", "cmd1", name="tui")
    cm.streams[0].feed(None)
    for _ in range(50):
        await asyncio.sleep(0.01)
        if s1.exited:
            break
    s2 = await mgr.open(cm, "ws1", "cmd2", name="tui")
    assert mgr.get("ws1", "tui") is s2
    await mgr.close("ws1", "tui")


async def test_manager_session_cap():
    cm = FakeContainerManager()
    mgr = TerminalSessionManager()
    for i in range(MAX_SESSIONS_PER_WORKSPACE):
        await mgr.open(cm, "ws1", f"cmd{i}")
    with pytest.raises(ValueError, match="term_close"):
        await mgr.open(cm, "ws1", "one too many")
    for s in list(mgr.list("ws1")):
        await mgr.close("ws1", s.id)


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_manager(monkeypatch):
    mgr = TerminalSessionManager()
    monkeypatch.setattr(terminal_sessions, "_MANAGER", mgr)
    return mgr


def _tool(cls, cm):
    return cls(
        container_manager=cm,
        workspace_id="ws1",
        state=None,
    )


async def test_term_tools_roundtrip(fresh_manager):
    cm = FakeContainerManager()
    open_tool = _tool(TermOpenTool, cm)
    result = await open_tool.execute(command="python3 app.py", name="tui", wait_ms=0)
    assert result.success, result.error
    assert result.metadata["session"]["session_id"] == "tui"

    cm.streams[0].feed(b"\x1b[2JWelcome\r\n")
    await asyncio.sleep(0.05)

    send_tool = _tool(TermSendTool, cm)
    result = await send_tool.execute(session_id="tui", keys=["down", "enter"], wait_ms=0)
    assert result.success, result.error
    assert cm.streams[0].written == [b"\x1b[B\r"]
    assert "Welcome" in result.output

    snap_tool = _tool(TermSnapshotTool, cm)
    result = await snap_tool.execute(session_id="tui")
    assert result.success and "Welcome" in result.output

    list_tool = _tool(TermListTool, cm)
    result = await list_tool.execute()
    assert result.success and "tui" in result.output

    close_tool = _tool(TermCloseTool, cm)
    result = await close_tool.execute(session_id="tui")
    assert result.success
    result = await list_tool.execute()
    assert "No terminal sessions" in result.output


async def test_term_tools_validation_errors(fresh_manager):
    cm = FakeContainerManager()
    result = await _tool(TermOpenTool, cm).execute(command="   ")
    assert not result.success and result.validation_error

    result = await _tool(TermSendTool, cm).execute(session_id="nope", text="x")
    assert not result.success and "term_list" in result.error

    open_tool = _tool(TermOpenTool, cm)
    await open_tool.execute(command="cat", name="s", wait_ms=0)
    result = await _tool(TermSendTool, cm).execute(session_id="s")
    assert not result.success and "text" in result.error
    result = await _tool(TermSendTool, cm).execute(session_id="s", keys=["warpdrive"])
    assert not result.success and "Unknown key" in result.error

    result = await _tool(TermCloseTool, cm).execute(session_id="nope")
    assert not result.success and result.validation_error
    await fresh_manager.close("ws1", "s")


async def test_term_send_on_dead_session_returns_final_screen(fresh_manager):
    cm = FakeContainerManager()
    open_tool = _tool(TermOpenTool, cm)
    await open_tool.execute(command="app", name="s", wait_ms=0)
    cm.streams[0].feed(b"crash log line\r\n")
    cm.streams[0].feed(None)
    session = fresh_manager.get("ws1", "s")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if session.exited:
            break
    result = await _tool(TermSendTool, cm).execute(session_id="s", text="x")
    assert not result.success
    assert "crash log line" in result.metadata["final_screen"]
    await fresh_manager.close("ws1", "s")


# ---------------------------------------------------------------------------
# Registration + persistence bootstrap
# ---------------------------------------------------------------------------

def test_term_tools_registered_in_all_coder_tools():
    from augmentum.coder.tools import ALL_CODER_TOOLS

    names = {cls.__name__ for cls in ALL_CODER_TOOLS}
    for expected in (
        "TermOpenTool", "TermSendTool", "TermSnapshotTool",
        "TermListTool", "TermCloseTool",
    ):
        assert expected in names


def test_persistence_setup_lines_in_bootstrap():
    from augmentum.coder.containers import (
        _assemble_keepalive_cmd,
        _persistence_setup_lines,
    )

    lines = _persistence_setup_lines()
    joined = "\n".join(lines)
    assert "/workspace/.venv" in joined
    assert "--system-site-packages" in joined
    assert "/workspace/requirements.txt" in joined
    assert "/workspace/.augmentum/setup.sh" in joined
    assert "/etc/profile.d/augmentum-persist.sh" in joined
    # Every hook is failure-guarded — provisioning may degrade, never die.
    for line in lines:
        if "requirements.txt" in line or "setup.sh" in line or "venv" in line:
            assert line.endswith("|| true"), line
    # And the keepalive wrapper still ends in the unconditional tail.
    cmd = _assemble_keepalive_cmd(lines)
    assert cmd[2].endswith("exec tail -f /dev/null")


def test_build_workspace_setup_lines_includes_persistence():
    from augmentum.coder.containers import ContainerManager

    mgr = object.__new__(ContainerManager)  # method touches no instance state
    lines = ContainerManager._build_workspace_setup_lines(
        mgr,
        workspace_id="ws_test",
        name="t",
        actual_image="augmentum-workspace:standard",
        tooling_profile="standard",
        initialize_workspace=False,
    )
    joined = "\n".join(lines)
    assert "--system-site-packages /workspace/.venv" in joined
    assert "/workspace/requirements.txt" in joined
    # Persistence lines land before the ready marker (last line).
    assert lines[-1] == "touch /workspace/.augmentum/ready"
