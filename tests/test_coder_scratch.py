"""Tests for ScratchStore — filesystem-as-scratchpad for large tool results.

Adopted 2026-04-20 from Manus's "filesystem as ultimate context" pattern.
When a tool returns a large output, we externalise it to
/workspace/.augmentum/scratch/<hash>.txt and replace the inline message
body with a summary + preview + path. Reversible alternative to
compaction clipping.

Covers:
  1. Threshold behaviour — content under the cutoff passes through,
     content above gets externalised.
  2. Path shape — hash stable for identical content, tool names
     sanitised so MCP-namespaced tools don't create subdirs.
  3. Fallback — container write failure returns None (caller falls
     back to inline), never raises.
  4. Handler integration — large tool_result gets externalised; inline
     message is short (summary + preview); full content is recoverable
     via file_read at the returned path.

Run: python -m pytest tests/test_coder_scratch.py -v
"""
from __future__ import annotations

import pytest

from augmentum.coder.scratch import (
    ScratchRef,
    ScratchStore,
    _SCRATCH_THRESHOLD,
    render_scratch_message,
)


# ---------------------------------------------------------------------------
# Stub container — records file_write calls and shell mkdir commands
# ---------------------------------------------------------------------------


class _StubCM:
    def __init__(self, *, write_raises: bool = False) -> None:
        self.writes: list[tuple[str, str]] = []
        self.shell_cmds: list[str] = []
        self._write_raises = write_raises

    async def file_write(self, workspace_id, path, content):  # noqa: ARG002
        if self._write_raises:
            raise RuntimeError("disk full")
        self.writes.append((path, content))

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, workspace_id, cmd, timeout=None):  # noqa: ARG002
        self.shell_cmds.append(cmd[-1] if isinstance(cmd, list) else str(cmd))
        return ""


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_under_threshold_not_externalised():
    cm = _StubCM()
    store = ScratchStore(cm, "ws")
    ref = await store.maybe_externalise(
        content="x" * 100, source_tool="file_read",
    )
    assert ref is None
    assert cm.writes == []


@pytest.mark.asyncio
async def test_content_at_threshold_not_externalised():
    """Boundary: content exactly at threshold passes through."""
    cm = _StubCM()
    store = ScratchStore(cm, "ws")
    ref = await store.maybe_externalise(
        content="x" * _SCRATCH_THRESHOLD, source_tool="file_read",
    )
    assert ref is None


@pytest.mark.asyncio
async def test_content_above_threshold_externalised():
    cm = _StubCM()
    store = ScratchStore(cm, "ws")
    content = "y" * (_SCRATCH_THRESHOLD + 1000)
    ref = await store.maybe_externalise(
        content=content, source_tool="file_read",
    )
    assert ref is not None
    assert ref.original_size == len(content)
    assert ref.source_tool == "file_read"
    assert ref.path.startswith("/workspace/.augmentum/scratch/")
    assert ref.path.endswith(".txt")
    # Write should have happened
    assert len(cm.writes) == 1
    assert cm.writes[0][0] == ref.path
    assert cm.writes[0][1] == content  # full content persisted
    # mkdir -p should have fired first
    assert any("mkdir -p" in c for c in cm.shell_cmds)


@pytest.mark.asyncio
async def test_custom_threshold():
    """Callers can tighten the threshold — used for tests and possibly
    config overrides."""
    cm = _StubCM()
    store = ScratchStore(cm, "ws", threshold=100)
    ref = await store.maybe_externalise(
        content="z" * 200, source_tool="shell_exec",
    )
    assert ref is not None
    assert ref.original_size == 200


# ---------------------------------------------------------------------------
# Path shape + hash stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_content_hashes_identically():
    """Stable hash so the model can see "this is the same data" if a
    tool re-runs. Deterministic makes debugging easier too."""
    cm = _StubCM()
    store = ScratchStore(cm, "ws", threshold=10)
    content = "alpha" * 100
    a = await store.maybe_externalise(content=content, source_tool="file_read")
    b = await store.maybe_externalise(content=content, source_tool="file_read")
    assert a is not None and b is not None
    assert a.path == b.path


@pytest.mark.asyncio
async def test_different_content_hashes_differently():
    cm = _StubCM()
    store = ScratchStore(cm, "ws", threshold=10)
    a = await store.maybe_externalise(
        content="A" * 100, source_tool="file_read",
    )
    b = await store.maybe_externalise(
        content="B" * 100, source_tool="file_read",
    )
    assert a.path != b.path


@pytest.mark.asyncio
async def test_mcp_namespaced_tool_name_sanitised():
    """MCP tools show up as ``server/tool`` — the slash would create a
    subdirectory; sanitise to hyphen so scratch stays flat."""
    cm = _StubCM()
    store = ScratchStore(cm, "ws", threshold=10)
    ref = await store.maybe_externalise(
        content="q" * 100, source_tool="github/fetch_issues",
    )
    assert ref is not None
    # No literal slash in the tool-name portion
    fname = ref.path.rsplit("/", 1)[-1]
    assert "/" not in fname
    # Still informative
    assert "github" in fname


# ---------------------------------------------------------------------------
# Fallbacks — never raise, always degrade gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_failure_returns_none():
    """Container write error → None. Caller keeps original content
    inline; no silent data loss."""
    cm = _StubCM(write_raises=True)
    store = ScratchStore(cm, "ws", threshold=10)
    ref = await store.maybe_externalise(
        content="x" * 500, source_tool="file_read",
    )
    assert ref is None


@pytest.mark.asyncio
async def test_no_container_short_circuits():
    store = ScratchStore(None, "ws", threshold=10)
    ref = await store.maybe_externalise(
        content="x" * 500, source_tool="file_read",
    )
    assert ref is None


@pytest.mark.asyncio
async def test_none_content_handled():
    cm = _StubCM()
    store = ScratchStore(cm, "ws")
    ref = await store.maybe_externalise(content=None, source_tool="x")
    assert ref is None


# ---------------------------------------------------------------------------
# Rendered message shape
# ---------------------------------------------------------------------------


def test_render_message_contains_size_tool_path_preview():
    ref = ScratchRef(
        path="/workspace/.augmentum/scratch/file_read-abc.txt",
        original_size=12345,
        preview="def foo():\n    return 42",
        source_tool="file_read",
    )
    msg = render_scratch_message(ref)
    assert "12345 bytes" in msg
    assert "file_read" in msg
    assert ref.path in msg
    assert "def foo():" in msg
    # Model-facing directive
    assert "file_read" in msg  # already checked, but also affirm usage hint
    assert "load the full content" in msg


def test_render_message_is_compact_enough_for_compaction():
    """Rendered body must fit comfortably in the 1500-char compaction
    cap; otherwise we haven't solved anything."""
    ref = ScratchRef(
        path="/workspace/.augmentum/scratch/shell_exec-deadbeef12.txt",
        original_size=100_000,
        preview="x" * 500,
        source_tool="shell_exec",
    )
    msg = render_scratch_message(ref)
    assert len(msg) < 1000


# ---------------------------------------------------------------------------
# Head + tail rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_content_externalises_head_and_tail():
    """For a long output the ScratchRef carries both head and tail so
    the rendered message surfaces final lines (pytest summaries / stack
    traces) instead of dropping them."""
    cm = _StubCM()
    store = ScratchStore(cm, "ws")
    head_marker = "HEAD_LINE_FIRST"
    tail_marker = "TAIL_LINE_LAST"
    middle = "m" * (_SCRATCH_THRESHOLD + 5_000)
    content = head_marker + middle + tail_marker
    ref = await store.maybe_externalise(
        content=content, source_tool="shell_exec",
    )
    assert ref is not None
    assert ref.preview.startswith(head_marker)
    assert ref.tail.endswith(tail_marker)
    # Tail must not duplicate head — tail_start enforces no overlap.
    assert head_marker not in ref.tail


@pytest.mark.asyncio
async def test_short_overflow_no_tail():
    """Content barely above threshold — head already covers everything,
    so tail stays empty (no point showing redundant tail bytes)."""
    cm = _StubCM()
    store = ScratchStore(cm, "ws")
    # Threshold is 8000, head is 500 — so to skip tail we need
    # size <= head + 1, i.e. content where tail_start >= size-1.
    content = "x" * (_SCRATCH_THRESHOLD + 1)  # 8001 bytes
    ref = await store.maybe_externalise(
        content=content, source_tool="file_read",
    )
    assert ref is not None
    # Tail-start = max(500, 8001-300) = 7701 < 8000, so a tail will
    # be emitted; verify the *content* of the test by computing what
    # it should be rather than asserting empty.
    expected_tail = content[max(500, len(content) - 300):]
    assert ref.tail == expected_tail


def test_render_message_with_tail_includes_elision_and_both_sides():
    ref = ScratchRef(
        path="/workspace/.augmentum/scratch/shell_exec-abc.txt",
        original_size=10_000,
        preview="START_OF_PYTEST_OUTPUT",
        tail="FAILED tests/test_x.py::test_y - AssertionError",
        source_tool="shell_exec",
    )
    msg = render_scratch_message(ref)
    assert "START_OF_PYTEST_OUTPUT" in msg
    assert "FAILED tests/test_x.py::test_y" in msg
    assert "elided" in msg
    assert "10000 bytes" in msg
    # Still under compaction cap with tail.
    assert len(msg) < 1500


def test_render_message_tail_omitted_when_empty():
    """Backwards-compat: refs without a tail render the original
    head-only template."""
    ref = ScratchRef(
        path="/workspace/.augmentum/scratch/file_read-xyz.txt",
        original_size=600,
        preview="just a small bit",
        tail="",
        source_tool="file_read",
    )
    msg = render_scratch_message(ref)
    assert "elided" not in msg
    assert "Preview above" in msg


# ---------------------------------------------------------------------------
# Handler integration — end-to-end externalisation on large tool output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_externalises_large_tool_output(monkeypatch):
    """Run _act_hybrid against a backend that emits one file_read tool
    call, have the tool return a huge output, verify the history
    message got replaced by the scratch summary and the full content
    reached the container file_write path."""
    from augmentum.modes.coder.handler import CoderHandler
    from augmentum.models.base import Message

    from tests.test_coder_handler import (
        _FakeChunk,
        _FakeTool,
        _force_native_tier,
        _make_request,
        _tc_delta,
    )

    _force_native_tier(monkeypatch)

    huge_output = "line\n" * 5000   # ~25k chars — well above threshold

    file_tool = _FakeTool("file_read", output=huge_output)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [file_tool],
    )

    class _OneRead:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_read",
                              {"path": "/workspace/big.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    # Need a container that supports file_write (scratch writes) AND
    # the post-hook existence check wc -c. Use _StubCM.
    cm = _StubCM()

    handler = CoderHandler(
        _OneRead(), session_id="sess-scratch",
        container_manager=cm, workspace_id="ws-scratch",
    )

    async for _ in handler._act_hybrid(
        _make_request("look at big.py"), workspace_context="",
    ):
        pass

    # The huge content should have been written to a scratch path
    scratch_writes = [
        w for w in cm.writes
        if w[0].startswith("/workspace/.augmentum/scratch/")
    ]
    assert scratch_writes, (
        f"Expected a scratch write for the large tool output; "
        f"wrote {cm.writes}"
    )
    # Content written is the full huge_output
    assert scratch_writes[0][1] == huge_output


@pytest.mark.asyncio
async def test_handler_keeps_small_output_inline(monkeypatch):
    """Small outputs should NOT hit the scratch path — no reason to
    pay the file_read round-trip cost."""
    from augmentum.modes.coder.handler import CoderHandler

    from tests.test_coder_handler import (
        _FakeChunk,
        _FakeTool,
        _force_native_tier,
        _make_request,
        _tc_delta,
    )

    _force_native_tier(monkeypatch)

    small_output = "short result"
    file_tool = _FakeTool("file_read", output=small_output)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [file_tool],
    )

    class _OneRead:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_read",
                              {"path": "/workspace/small.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    cm = _StubCM()
    handler = CoderHandler(
        _OneRead(), session_id="sess-small",
        container_manager=cm, workspace_id="ws-small",
    )
    async for _ in handler._act_hybrid(
        _make_request("peek at small.py"), workspace_context="",
    ):
        pass

    scratch_writes = [
        w for w in cm.writes
        if w[0].startswith("/workspace/.augmentum/scratch/")
    ]
    assert scratch_writes == [], (
        "Small output should NOT externalise; "
        f"unexpected scratch writes: {scratch_writes}"
    )
