"""Tests for the mtime-aware read-before-edit guard.

Prior to 2026-04-20, ``CoderState.files_read`` was a plain ``set[str]``
answering only "has this path been read at all?" — letting the agent
silently clobber files that had changed externally (user edit, git pull,
another agent) since the read.

This suite verifies:
  1. ``CoderState.files_read`` is now ``dict[str, float]`` (path → mtime
     at read time).
  2. ``record_file_read(path, mtime)`` stores the mtime.
  3. ``can_edit(path, current_mtime)`` returns False when current is
     newer than stored (stale), True when ≤ (fresh), True when stored
     is ``inf`` (unknown mtime, backward compat).
  4. ``FileReadTool.execute`` captures mtime via ``stat -c %Y`` after
     the successful read.
  5. ``CodeEditTool`` and ``CodeMultiEditTool`` reject edits on a stale
     path with a clear recovery message.

Run: python -m pytest tests/test_coder_mtime_staleness.py -v
"""
from __future__ import annotations

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import (
    CodeEditTool,
    CodeMultiEditTool,
    FileReadTool,
    FileWriteTool,
)


# ---------------------------------------------------------------------------
# CoderState mtime semantics
# ---------------------------------------------------------------------------


def test_record_file_read_stores_mtime():
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/a.py", mtime=1000.0)
    assert state.files_read == {"/a.py": 1000.0}


def test_record_file_read_without_mtime_stores_inf():
    """Backward compat: callers that can't provide mtime get
    ``float('inf')`` which disables the staleness check for that
    entry (equivalent to pre-2026-04-20 behaviour)."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/b.py")
    assert state.files_read["/b.py"] == float("inf")


def test_can_edit_fresh_read_passes():
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/a.py", mtime=1000.0)
    # File hasn't changed since we read it
    assert state.can_edit("/a.py", current_mtime=1000.0) is True
    assert state.can_edit("/a.py", current_mtime=999.0) is True  # we're ahead


def test_can_edit_stale_read_rejects():
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/a.py", mtime=1000.0)
    # File changed after our read
    assert state.can_edit("/a.py", current_mtime=1001.0) is False


def test_can_edit_never_read_rejects_regardless_of_mtime():
    state = CoderState(session_id="s", workspace_id="w")
    assert state.can_edit("/unknown.py", current_mtime=1000.0) is False
    assert state.can_edit("/unknown.py", current_mtime=None) is False


def test_can_edit_inf_stored_always_fresh():
    """When stored mtime is inf (unknown from a pre-fix read or a
    shell-based read that didn't capture mtime), staleness check is
    disabled — avoids false staleness rejections on legacy sessions."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/old.py")  # no mtime → inf
    assert state.can_edit("/old.py", current_mtime=9999.0) is True


def test_can_edit_without_current_mtime_falls_back_to_membership():
    """Callers that don't pass current_mtime get the original
    "any-read-is-fine" behaviour — equivalent to the pre-2026-04-20
    set-membership check."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/a.py", mtime=1000.0)
    assert state.can_edit("/a.py") is True
    assert state.can_edit("/missing.py") is False


# ---------------------------------------------------------------------------
# FileReadTool captures mtime
# ---------------------------------------------------------------------------


class _StatCM:
    """Container that returns canned file content + stat mtime."""

    def __init__(self, *, content: str, mtime: float | None = 1000.0) -> None:
        self._content = content
        self._mtime = mtime

    async def file_read(self, ws, path):  # noqa: ARG002
        return self._content

    async def file_write(self, ws, path, content):  # noqa: ARG002
        return None

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, ws, cmd, timeout=None):  # noqa: ARG002
        """Handle the stat command the tool issues after file_read."""
        cmd_str = cmd[-1] if isinstance(cmd, list) else str(cmd)
        if "stat -c %Y" in cmd_str:
            if self._mtime is None:
                return ""
            return f"{self._mtime}\n"
        return ""


@pytest.mark.asyncio
async def test_file_read_captures_mtime_into_state():
    state = CoderState(session_id="s", workspace_id="w")
    cm = _StatCM(content="a\nb\nc", mtime=1234.0)
    tool = FileReadTool(container_manager=cm, workspace_id="w", state=state)
    r = await tool.execute(path="/a.py")
    assert r.success
    assert state.files_read["/a.py"] == 1234.0


@pytest.mark.asyncio
async def test_file_read_falls_back_when_stat_fails():
    """Stat failure → files_read stores inf, downstream edits allowed
    (pre-fix behaviour preserved for entries we can't check)."""
    state = CoderState(session_id="s", workspace_id="w")
    cm = _StatCM(content="x", mtime=None)  # stat returns empty
    tool = FileReadTool(container_manager=cm, workspace_id="w", state=state)
    r = await tool.execute(path="/b.py")
    assert r.success
    assert state.files_read["/b.py"] == float("inf")


@pytest.mark.asyncio
async def test_file_read_offset_does_not_overwrite_mtime():
    """Partial re-reads (offset>0) don't re-arm the read-before-edit
    guard, so they also shouldn't overwrite the stored mtime. Tracked
    only on fresh full reads (offset=0)."""
    state = CoderState(session_id="s", workspace_id="w")
    content = "\n".join(f"line{i}" for i in range(100))
    cm = _StatCM(content=content, mtime=500.0)
    tool = FileReadTool(container_manager=cm, workspace_id="w", state=state)

    # First read: offset=0, arms the state
    await tool.execute(path="/big.py", offset=0)
    assert state.files_read["/big.py"] == 500.0

    # Simulate the file changing in the container
    cm._mtime = 600.0
    # Partial re-read at offset>0 — shouldn't touch files_read
    await tool.execute(path="/big.py", offset=20, limit=10)
    assert state.files_read["/big.py"] == 500.0


# ---------------------------------------------------------------------------
# Edit tools reject stale reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_edit_rejects_stale_read():
    """File read at T, edited externally at T+1 → code_edit rejects
    with a clear "re-read" message, does NOT write."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)

    cm = _StatCM(content="def foo():\n    pass\n", mtime=1500.0)  # file changed
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)
    r = await tool.execute(
        path="/foo.py", search="pass", replace="return 1",
    )
    assert r.success is False
    assert "re-read" in r.error.lower() or "modified externally" in r.error.lower()


@pytest.mark.asyncio
async def test_code_edit_passes_when_file_unchanged():
    """File read at T, current mtime == T → edit proceeds."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)

    cm = _StatCM(content="def foo():\n    pass\n", mtime=1000.0)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)
    r = await tool.execute(
        path="/foo.py", search="pass", replace="return 1",
    )
    assert r.success is True


@pytest.mark.asyncio
async def test_code_multi_edit_rejects_stale_read():
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)

    cm = _StatCM(content="a\nb\nc\n", mtime=2000.0)  # file changed
    tool = CodeMultiEditTool(container_manager=cm, workspace_id="w", state=state)
    r = await tool.execute(
        path="/foo.py",
        edits=[{"search": "b", "replace": "B"}],
    )
    assert r.success is False
    assert "re-read" in r.error.lower() or "modified" in r.error.lower()


@pytest.mark.asyncio
async def test_code_edit_rejects_never_read():
    """Original read-before-edit guard still fires for paths never
    read — regression test for the freshness-helper that previously
    short-circuited to None on stat failure."""
    state = CoderState(session_id="s", workspace_id="w")
    # NB: no record_file_read for /foo.py

    cm = _StatCM(content="a", mtime=None)  # stat fails too
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)
    r = await tool.execute(path="/foo.py", search="x", replace="y")
    assert r.success is False
    # Error message should mention reading first
    assert "read" in r.error.lower()


# ---------------------------------------------------------------------------
# Self-write doesn't trigger the freshness guard on the next edit
# ---------------------------------------------------------------------------


class _BumpingStatCM(_StatCM):
    """Variant of _StatCM that bumps mtime on every file_write — the
    real container's behaviour. Lets us assert that the write tools
    refresh ``files_read[path]`` to the post-write mtime so a
    same-turn second edit doesn't false-positive on staleness."""

    async def file_write(self, ws, path, content):  # noqa: ARG002
        if self._mtime is None:
            self._mtime = 1000.0
        self._mtime += 1.0
        self._content = content
        return None


@pytest.mark.asyncio
async def test_file_write_refreshes_read_mtime_for_same_turn_edit():
    """After file_write succeeds, the next edit on the same path must
    pass the freshness check. Pre-fix, files_read[path] still pointed
    at the original read mtime while disk mtime advanced — the second
    edit failed with the misleading "modified externally" error and
    forced a redundant re-read of a file the model just wrote."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)
    cm = _BumpingStatCM(content="old", mtime=1000.0)

    write = FileWriteTool(container_manager=cm, workspace_id="w", state=state)
    wr = await write.execute(path="/foo.py", content="line\npass\n")
    assert wr.success is True
    # Helper must have re-stamped files_read with the post-write mtime.
    assert state.files_read["/foo.py"] == 1001.0

    edit = CodeEditTool(container_manager=cm, workspace_id="w", state=state)
    er = await edit.execute(path="/foo.py", search="pass", replace="return 1")
    assert er.success is True, (
        "Same-turn edit after own write rejected — files_read mtime "
        f"refresh regressed. error={er.error!r}"
    )


@pytest.mark.asyncio
async def test_code_edit_refreshes_read_mtime_for_chained_edits():
    """Two code_edit calls on the same path in one turn. The second
    edit must pass the freshness check after the first edit's write
    bumped on-disk mtime."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)
    cm = _BumpingStatCM(content="pass\nstay\n", mtime=1000.0)

    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)
    r1 = await tool.execute(path="/foo.py", search="pass", replace="return 1")
    assert r1.success is True

    r2 = await tool.execute(path="/foo.py", search="stay", replace="moved")
    assert r2.success is True, (
        "Chained code_edit on same path rejected — second edit saw "
        f"a stale files_read mtime. error={r2.error!r}"
    )


@pytest.mark.asyncio
async def test_code_multi_edit_refreshes_read_mtime():
    """Same regression coverage for the batch variant — code_edit_batch
    must also bump files_read so a follow-up edit doesn't false-fail."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)
    cm = _BumpingStatCM(content="a\nb\nc\n", mtime=1000.0)

    batch = CodeMultiEditTool(container_manager=cm, workspace_id="w", state=state)
    r1 = await batch.execute(
        path="/foo.py",
        edits=[{"search": "a", "replace": "A"}, {"search": "b", "replace": "B"}],
    )
    assert r1.success is True

    edit = CodeEditTool(container_manager=cm, workspace_id="w", state=state)
    r2 = await edit.execute(path="/foo.py", search="c", replace="C")
    assert r2.success is True, (
        "Edit after code_edit_batch on same path rejected — "
        f"batch tool didn't refresh files_read mtime. error={r2.error!r}"
    )


@pytest.mark.asyncio
async def test_write_preserves_baseline_when_stat_fails():
    """If post-write stat fails (returns None), we must NOT erase the
    prior baseline to inf — doing so would silence a genuinely external
    edit arriving later. The pre-write baseline stays intact."""
    state = CoderState(session_id="s", workspace_id="w")
    state.record_file_read("/foo.py", mtime=1000.0)

    class _NoStatBumpingCM(_BumpingStatCM):
        async def _run_command(self, ws, cmd, timeout=None):  # noqa: ARG002
            return ""   # stat returns nothing — _current_mtime → None

    cm = _NoStatBumpingCM(content="old", mtime=1000.0)
    write = FileWriteTool(container_manager=cm, workspace_id="w", state=state)
    r = await write.execute(path="/foo.py", content="new")
    assert r.success is True
    # Baseline preserved — not silently bumped to inf.
    assert state.files_read["/foo.py"] == 1000.0
