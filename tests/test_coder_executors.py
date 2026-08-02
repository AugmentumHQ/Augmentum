"""ContainerExecutor forwarding characterization.

Phase 1 of the "loop in the editor" work introduces a WorkspaceExecutor seam
without moving any container-lifecycle code: ContainerExecutor must forward each
portable op 1:1 to the underlying ContainerManager, with ``workspace_id``
injected and every other argument preserved verbatim. These tests pin that
contract so a later refactor (or the RemoteEditorExecutor sibling) can't silently
drift the container path's behavior.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from augmentum.coder.executors import ContainerExecutor, WorkspaceExecutor

WS = "ws-abc123"


def _cm() -> AsyncMock:
    cm = AsyncMock()
    cm.run_command = AsyncMock(return_value="stdout-out")
    cm.file_read = AsyncMock(return_value="file-contents")
    cm.file_download = AsyncMock(return_value=b"\x00\x01raw")
    cm.file_write = AsyncMock(return_value=None)
    cm.file_list = AsyncMock(return_value=["a", "b"])
    cm.file_upload = AsyncMock(return_value=None)
    return cm


def test_is_a_workspace_executor() -> None:
    assert issubclass(ContainerExecutor, WorkspaceExecutor)
    ex = ContainerExecutor(_cm(), WS)
    assert isinstance(ex, WorkspaceExecutor)


@pytest.mark.asyncio
async def test_run_command_forwards_all_kwargs_and_injects_ws() -> None:
    cm = _cm()
    ex = ContainerExecutor(cm, WS)

    async def sink(_b: bytes) -> None:  # on_chunk stand-in
        return None

    out = await ex.run_command(
        ["ls", "-la"], 12.5,
        idle_timeout=3.0, progress_path="/workspace/x.log",
        on_chunk=sink, environment={"K": "V"}, login_shell=True,
    )
    assert out == "stdout-out"
    cm.run_command.assert_awaited_once_with(
        WS, ["ls", "-la"], 12.5,
        idle_timeout=3.0, progress_path="/workspace/x.log",
        on_chunk=sink, environment={"K": "V"}, login_shell=True,
    )


@pytest.mark.asyncio
async def test_run_command_passthrough_injects_no_defaults() -> None:
    # Transparent wrapper: a bare call forwards ONLY workspace_id + cmd, letting
    # ContainerManager apply its own defaults. It must NOT inject timeout=30.0 or
    # the keyword defaults, or it would change the call the manager (and test
    # mocks) observe. This is what keeps the migration behavior-neutral.
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    await ex.run_command(["echo", "hi"])
    cm.run_command.assert_awaited_once_with(WS, ["echo", "hi"])


@pytest.mark.asyncio
async def test_read_file_forwards() -> None:
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    assert await ex.read_file("/workspace/f.py") == "file-contents"
    cm.file_read.assert_awaited_once_with(WS, "/workspace/f.py")


@pytest.mark.asyncio
async def test_read_file_bytes_uses_file_download_not_file_read_bytes() -> None:
    # The binary-safe reader is file_download (get-archive). ``file_read_bytes``
    # does not exist on ContainerManager — the executor must not reference it.
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    assert await ex.read_file_bytes("/workspace/a.gguf") == b"\x00\x01raw"
    cm.file_download.assert_awaited_once_with(WS, "/workspace/a.gguf")
    assert not hasattr(cm, "file_read_bytes") or not cm.file_read_bytes.called


@pytest.mark.asyncio
async def test_write_file_forwards() -> None:
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    await ex.write_file("/workspace/f.py", "print(1)")
    cm.file_write.assert_awaited_once_with(WS, "/workspace/f.py", "print(1)")


@pytest.mark.asyncio
async def test_list_files_forwards_with_default_path() -> None:
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    assert await ex.list_files() == ["a", "b"]
    cm.file_list.assert_awaited_once_with(WS, "/workspace")


@pytest.mark.asyncio
async def test_list_files_forwards_explicit_path() -> None:
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    await ex.list_files("/workspace/sub")
    cm.file_list.assert_awaited_once_with(WS, "/workspace/sub")


@pytest.mark.asyncio
async def test_upload_files_forwards() -> None:
    cm = _cm()
    ex = ContainerExecutor(cm, WS)
    payload = [("a.txt", b"aaa"), ("d/b.bin", b"\x00")]
    await ex.upload_files("/workspace/dst", payload)
    cm.file_upload.assert_awaited_once_with(WS, "/workspace/dst", payload)


def test_abstract_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        WorkspaceExecutor()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# RemoteEditorExecutor (Phase 2) — round-trip over an EditorChannel
# ---------------------------------------------------------------------------

from augmentum.coder.executors import (  # noqa: E402
    EditorChannel,
    RemoteEditorExecutor,
)


class FakeEditorChannel(EditorChannel):
    """Records requests and returns canned responses keyed by method."""

    def __init__(self, responses: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses or {}

    async def request(self, method: str, params: dict) -> dict:
        self.calls.append((method, params))
        resp = self._responses.get(method)
        if callable(resp):
            return resp(params)
        return resp if resp is not None else {}


def test_remote_is_a_workspace_executor() -> None:
    assert issubclass(RemoteEditorExecutor, WorkspaceExecutor)
    assert isinstance(RemoteEditorExecutor(FakeEditorChannel()), WorkspaceExecutor)


@pytest.mark.asyncio
async def test_remote_read_file_maps_to_fs_read_text_file() -> None:
    ch = FakeEditorChannel({"fs/read_text_file": {"content": "hello\nworld"}})
    ex = RemoteEditorExecutor(ch)
    assert await ex.read_file("/workspace/a.py") == "hello\nworld"
    assert ch.calls == [("fs/read_text_file", {"path": "/workspace/a.py"})]


@pytest.mark.asyncio
async def test_remote_write_file_maps_to_fs_write_text_file() -> None:
    ch = FakeEditorChannel()
    ex = RemoteEditorExecutor(ch)
    await ex.write_file("/workspace/a.py", "print(1)")
    assert ch.calls == [
        ("fs/write_text_file", {"path": "/workspace/a.py", "content": "print(1)"}),
    ]


@pytest.mark.asyncio
async def test_remote_run_command_maps_to_terminal_run_and_forwards_chunk() -> None:
    ch = FakeEditorChannel({"terminal/run": {"output": "done\n", "exit_code": 0}})
    ex = RemoteEditorExecutor(ch)
    seen: list[bytes] = []

    async def sink(b: bytes) -> None:
        seen.append(b)

    out = await ex.run_command(["echo", "hi"], 12.0, environment={"K": "V"}, on_chunk=sink)
    assert out == "done\n"
    method, params = ch.calls[0]
    assert method == "terminal/run"
    assert params["command"] == ["echo", "hi"]
    assert params["cwd"] == "/workspace"
    assert params["timeout"] == 12.0
    assert params["environment"] == {"K": "V"}
    assert seen == [b"done\n"]  # full output forwarded once


@pytest.mark.asyncio
async def test_remote_list_files_lists_via_terminal_and_parses() -> None:
    ls = (
        "total 8\n"
        "drwxr-xr-x 2 root root 4096 1711234567 sub\n"
        "-rw-r--r-- 1 root root  512 1711234568 file.txt\n"
    )
    ch = FakeEditorChannel({"terminal/run": {"output": ls}})
    ex = RemoteEditorExecutor(ch)
    entries = await ex.list_files("/workspace")
    names = {e.name: e.is_dir for e in entries}
    assert names == {"sub": True, "file.txt": False}
    # listed via the terminal, with the ls parser's expected flags
    assert ch.calls[0][0] == "terminal/run"
    assert ch.calls[0][1]["command"] == ["ls", "-la", "--time-style=+%s", "/workspace"]


@pytest.mark.asyncio
async def test_remote_binary_ops_raise_notimplemented() -> None:
    ex = RemoteEditorExecutor(FakeEditorChannel())
    with pytest.raises(NotImplementedError):
        await ex.read_file_bytes("/workspace/a.gguf")
    with pytest.raises(NotImplementedError):
        await ex.upload_files("/workspace", [("a.txt", b"x")])


@pytest.mark.asyncio
async def test_spine_real_tool_runs_against_remote_editor() -> None:
    # The Phase-2 spine in miniature: the SAME FileReadTool, handed a
    # RemoteEditorExecutor instead of a container, reads from the editor.
    from augmentum.coder.state import CoderState
    from augmentum.coder.tools import FileReadTool

    ch = FakeEditorChannel({
        "fs/read_text_file": {"content": "def foo():\n    return 42\n"},
        "terminal/run": {"output": "1711234568\n"},  # stat mtime probe
    })
    ex = RemoteEditorExecutor(ch)
    tool = FileReadTool(
        executor=ex, workspace_id="w", state=CoderState(session_id="s", workspace_id="w"),
    )
    result = await tool.execute(path="/workspace/foo.py")
    assert result.success is True
    assert "return 42" in result.output
    # proves the read went over the editor channel, not a container
    assert ("fs/read_text_file", {"path": "/workspace/foo.py"}) in ch.calls
