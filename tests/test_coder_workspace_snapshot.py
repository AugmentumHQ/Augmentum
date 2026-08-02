"""Tests for WorkspaceSnapshot — the auto-refreshing workspace tree.

Covers:
  1. Construction + is_stale/mark_stale flag mechanics
  2. refresh_if_stale returns True only when a scan ran, caches result,
     skips re-scan when not stale, forces on demand
  3. Parse correctness — wc -l output with "total" lines, missing paths,
     the /workspace/ prefix stripping
  4. Render shape — tags, files count, truncation footer, newline-free
     paths
  5. Delta markers — [NEW] on added, [DEL] on removed, [MOD] on line-
     count jumps above both absolute and relative thresholds
  6. Truncation — _max_files cap with "showing N of M" footer
  7. Empty-container handling (graceful empty string)
  8. Container exception handling (preserves last-known-good snapshot)
  9. Handler integration — mutation tools mark snapshot stale,
     _canonical_observation returns the rendered tree

Run: python -m pytest tests/test_coder_workspace_snapshot.py -v
"""
from __future__ import annotations

import pytest

from augmentum.coder.reviews import ReviewRegistry
from augmentum.coder.snapshot import WorkspaceSnapshot

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)


# ---------------------------------------------------------------------------
# Scriptable container — per-test control over what ``find | wc -l`` returns
# ---------------------------------------------------------------------------


class _ScriptedContainer(_ExtendedContainerManager):
    """Container whose ``_run_command`` yields a caller-controlled script.

    Each entry in ``outputs`` is consumed in order; after exhaustion the
    last entry repeats. This lets a test simulate "scan 1", "scan 2 with
    a new file", "scan 3 after deletion" without a real container.
    """

    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, workspace_id, cmd, timeout=None):
        self.calls.append(list(cmd))
        if not self.outputs:
            return ""
        out = self.outputs[0] if len(self.outputs) == 1 else self.outputs.pop(0)
        return out


class _ReviewScriptedContainer(_ScriptedContainer):
    """Workspace-snapshot script + file contents for review-flow tests."""

    def __init__(self, outputs: list[str], files: dict[str, str]) -> None:
        super().__init__(outputs)
        self._files = dict(files)

    async def file_read(self, workspace_id, path):
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]


def _wc_output(*rows: tuple[str, int]) -> str:
    """Build a fake ``wc -l`` body like the real container would emit."""
    lines = [f"   {count} /workspace/{path}" for path, count in rows]
    if len(rows) > 1:
        # Real wc appends a "total" footer when given multiple files.
        lines.append(f"   {sum(r[1] for r in rows)} total")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stale-flag mechanics
# ---------------------------------------------------------------------------


def test_new_snapshot_is_stale():
    s = WorkspaceSnapshot(None, "ws")
    assert s.is_stale()


def test_mark_stale_flips_flag():
    cm = _ScriptedContainer([_wc_output(("a.py", 10))])
    s = WorkspaceSnapshot(cm, "ws")
    import asyncio
    asyncio.get_event_loop().run_until_complete(s.refresh_if_stale())
    assert not s.is_stale()
    s.mark_stale()
    assert s.is_stale()


# ---------------------------------------------------------------------------
# refresh_if_stale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_if_stale_runs_when_stale():
    cm = _ScriptedContainer([_wc_output(("a.py", 10), ("b.py", 20))])
    s = WorkspaceSnapshot(cm, "ws")
    did_refresh = await s.refresh_if_stale()
    assert did_refresh is True
    assert s.refresh_count == 1
    assert not s.is_stale()


@pytest.mark.asyncio
async def test_refresh_if_stale_skips_when_not_stale():
    cm = _ScriptedContainer([_wc_output(("a.py", 10))])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    assert s.refresh_count == 1
    # Second call without mark_stale → no-op
    did_refresh = await s.refresh_if_stale()
    assert did_refresh is False
    assert s.refresh_count == 1


@pytest.mark.asyncio
async def test_refresh_force_bypasses_stale_check():
    cm = _ScriptedContainer([_wc_output(("a.py", 10))])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    await s.refresh_if_stale(force=True)
    assert s.refresh_count == 2


@pytest.mark.asyncio
async def test_refresh_with_no_container_returns_false():
    s = WorkspaceSnapshot(None, "ws")
    assert await s.refresh_if_stale(force=True) is False


@pytest.mark.asyncio
async def test_refresh_container_error_preserves_last_snapshot():
    """Container exception → last-known-good tree preserved, stale stays."""
    cm = _ScriptedContainer([_wc_output(("a.py", 10))])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    before = s.render()

    # Second scan blows up
    async def _broken(*_a, **_kw):
        raise RuntimeError("shell died")
    cm._run_command = _broken

    s.mark_stale()
    did = await s.refresh_if_stale()
    assert did is False
    # Last rendering still available
    assert s.render() == before


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_strips_workspace_prefix():
    cm = _ScriptedContainer([_wc_output(("augmentum/coder/tools.py", 2100))])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    rendered = s.render()
    assert "augmentum/coder/tools.py" in rendered
    # No leading slash — keeps the tree readable
    assert "/workspace/augmentum" not in rendered


@pytest.mark.asyncio
async def test_parse_skips_total_line():
    """``wc -l`` appends a ``N total`` footer for multi-file input."""
    cm = _ScriptedContainer([_wc_output(
        ("a.py", 10), ("b.py", 20), ("c.py", 30),
    )])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    rendered = s.render()
    # Three files in the body, the 60-total footer should NOT appear
    # as a file entry (no "total" path, no "60L" line).
    assert "a.py (10L)" in rendered
    assert "b.py (20L)" in rendered
    assert "c.py (30L)" in rendered
    assert "total" not in rendered.split("<workspace_tree", 1)[-1].split(
        "</workspace_tree>", 1
    )[0].split("\n")[-3]  # total should be nowhere in body


@pytest.mark.asyncio
async def test_parse_handles_malformed_lines():
    """Blank lines, garbage entries, invalid line counts are skipped."""
    cm = _ScriptedContainer([
        "\n\n"
        "  garbage no count here\n"
        "  10 /workspace/good.py\n"
        "  abc /workspace/bad-count.py\n"
        "  42 /workspace/also-good.py\n"
    ])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    rendered = s.render()
    assert "good.py (10L)" in rendered
    assert "also-good.py (42L)" in rendered
    # Garbage / bad-count excluded
    assert "bad-count.py" not in rendered
    assert "garbage" not in rendered


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_empty_emits_explicit_marker():
    # Empty workspace must produce an explicit "(empty)" block rather
    # than "" — without a signal, models confabulate a project from
    # training data (see DTLN incident, 2026-04-21).
    cm = _ScriptedContainer([""])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale(force=True)
    rendered = s.render()
    assert rendered.startswith("<workspace_tree")
    assert rendered.endswith("</workspace_tree>")
    assert 'files="0"' in rendered
    assert "empty" in rendered.lower()
    # with_header=False drops the tags but keeps the body
    body = s.render(with_header=False)
    assert "empty" in body.lower()
    assert "<workspace_tree" not in body


@pytest.mark.asyncio
async def test_render_has_tags_and_counts():
    cm = _ScriptedContainer([_wc_output(
        ("a.py", 10), ("b.py", 20),
    )])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    rendered = s.render()
    assert rendered.startswith("<workspace_tree")
    assert rendered.endswith("</workspace_tree>")
    assert 'files="2"' in rendered
    assert 'total="2"' in rendered
    assert 'refresh="1"' in rendered


@pytest.mark.asyncio
async def test_render_sorted_alphabetically():
    cm = _ScriptedContainer([_wc_output(
        ("z.py", 10), ("a.py", 20), ("m.py", 30),
    )])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    rendered = s.render()
    pos_a = rendered.index("a.py")
    pos_m = rendered.index("m.py")
    pos_z = rendered.index("z.py")
    assert pos_a < pos_m < pos_z


@pytest.mark.asyncio
async def test_render_without_header_drops_tags():
    cm = _ScriptedContainer([_wc_output(("a.py", 10))])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    body = s.render(with_header=False)
    assert "<workspace_tree" not in body
    assert "</workspace_tree>" not in body
    assert "a.py (10L)" in body


# ---------------------------------------------------------------------------
# Delta markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delta_marks_new_file():
    cm = _ScriptedContainer([
        _wc_output(("a.py", 10)),
        _wc_output(("a.py", 10), ("b.py", 5)),
    ])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    s.mark_stale()
    await s.refresh_if_stale()
    rendered = s.render()
    assert "a.py (10L)" in rendered
    assert "b.py (5L) [NEW]" in rendered


@pytest.mark.asyncio
async def test_delta_marks_deleted_file():
    cm = _ScriptedContainer([
        _wc_output(("a.py", 10), ("b.py", 5)),
        _wc_output(("a.py", 10)),
    ])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    s.mark_stale()
    await s.refresh_if_stale()
    rendered = s.render()
    assert "# Deleted since last refresh:" in rendered
    assert "b.py [DEL]" in rendered


@pytest.mark.asyncio
async def test_delta_marks_modified_file_on_big_jump():
    """Line-count jump ≥20 lines or ≥20 % marks [MOD]."""
    cm = _ScriptedContainer([
        _wc_output(("a.py", 100)),
        _wc_output(("a.py", 130)),  # +30 lines (>20 absolute, >20%)
    ])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    s.mark_stale()
    await s.refresh_if_stale()
    rendered = s.render()
    assert "a.py (130L) [MOD]" in rendered


@pytest.mark.asyncio
async def test_delta_ignores_small_line_changes():
    """Line-count changes under both thresholds don't emit [MOD]."""
    cm = _ScriptedContainer([
        _wc_output(("a.py", 1000)),
        _wc_output(("a.py", 1005)),  # +5 lines (<20 absolute, <20%)
    ])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    s.mark_stale()
    await s.refresh_if_stale()
    rendered = s.render()
    # Look at the file-list line specifically — the rendered preamble
    # also mentions "[MOD]" as part of the marker legend, which is fine.
    assert "a.py (1005L) [MOD]" not in rendered
    assert "a.py (1005L)" in rendered


@pytest.mark.asyncio
async def test_delta_empty_on_first_refresh():
    """No previous snapshot → no per-file markers, no deletions section."""
    cm = _ScriptedContainer([_wc_output(("a.py", 10), ("b.py", 20))])
    s = WorkspaceSnapshot(cm, "ws")
    await s.refresh_if_stale()
    rendered = s.render()
    # "[NEW]" and "[DEL]" appear in the preamble legend; the relevant
    # check is that no individual file line carries them.
    assert "a.py (10L) [NEW]" not in rendered
    assert "b.py (20L) [NEW]" not in rendered
    assert "# Deleted since last refresh:" not in rendered


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truncation_caps_at_max_files():
    """A repo over ``max_files`` emits a "showing N of M" footer."""
    rows = [(f"f{i:03d}.py", 10) for i in range(50)]
    cm = _ScriptedContainer([_wc_output(*rows)])
    s = WorkspaceSnapshot(cm, "ws", max_files=10)
    await s.refresh_if_stale()
    rendered = s.render()
    assert 'files="10"' in rendered
    assert 'total="50"' in rendered
    assert "Showing 10 of 50 files" in rendered
    # First 10 present; later ones absent
    assert "f000.py" in rendered
    assert "f009.py" in rendered
    assert "f020.py" not in rendered


@pytest.mark.asyncio
async def test_no_truncation_footer_when_under_cap():
    cm = _ScriptedContainer([_wc_output(("a.py", 10))])
    s = WorkspaceSnapshot(cm, "ws", max_files=50)
    await s.refresh_if_stale()
    rendered = s.render()
    assert "Showing" not in rendered


# ---------------------------------------------------------------------------
# Handler integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_constructs_snapshot_when_container_present():
    """Handler with a container gets a live WorkspaceSnapshot; Phase 1
    passthrough (no container) gets None."""
    from augmentum.modes.coder.handler import CoderHandler
    with_container = CoderHandler(
        _FakeBackend([]),
        session_id="sess-snap",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-snap",
    )
    assert with_container._workspace_snapshot is not None

    no_container = CoderHandler(
        _FakeBackend([]),
        session_id="sess-no-snap",
        container_manager=None,
    )
    assert no_container._workspace_snapshot is None


@pytest.mark.asyncio
async def test_mutation_tool_marks_snapshot_stale(monkeypatch):
    """After a successful code_edit / file_write, the snapshot flips stale."""
    from augmentum.modes.coder.handler import CoderHandler
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_write", output="wrote")],
    )

    class _OneWriteThenStop:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-w", "file_write",
                              {"path": "/workspace/x.py", "content": "x"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    # Use a scripted container so the initial refresh at _act_phase
    # succeeds without blowing up on the real find command.
    cm = _ScriptedContainer([_wc_output(("before.py", 10))])
    handler = CoderHandler(
        _OneWriteThenStop(), session_id="sess-mut",
        container_manager=cm, workspace_id="ws-mut",
    )

    # Pre-mark clean — act_hybrid should NOT re-mark stale unless a
    # mutation succeeded.
    await handler._workspace_snapshot.refresh_if_stale(force=True)
    assert not handler._workspace_snapshot.is_stale()

    async for _ in handler._act_hybrid(
        _make_request("write x.py"), workspace_context="",
    ):
        pass

    # The successful file_write should have flipped stale even though
    # the tool's _FakeTool doesn't actually touch the filesystem.
    assert handler._workspace_snapshot.is_stale(), (
        "Successful mutation tool should have marked the snapshot stale"
    )


@pytest.mark.asyncio
async def test_canonical_observation_returns_snapshot(monkeypatch):
    """The observation refresh path uses the snapshot when available."""
    from augmentum.modes.coder.handler import CoderHandler

    cm = _ScriptedContainer([_wc_output(
        ("a.py", 10), ("b.py", 20),
    )])
    handler = CoderHandler(
        _FakeBackend([]), session_id="sess-obs",
        container_manager=cm, workspace_id="ws-obs",
    )

    obs = await handler._canonical_observation()
    assert "<workspace_tree" in obs
    assert "a.py (10L)" in obs
    assert "b.py (20L)" in obs


@pytest.mark.asyncio
async def test_publish_turn_review_captures_shell_created_files():
    """Review publishing should include files that appeared via shell-side work."""
    from augmentum.modes.coder.handler import CoderHandler

    cm = _ReviewScriptedContainer(
        [
            _wc_output(("a.py", 10)),
            _wc_output(("a.py", 10), ("new.py", 1)),
        ],
        {"/workspace/new.py": "print('hi')\n"},
    )
    registry = ReviewRegistry()
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="ws-shell-review",
        container_manager=cm,
        workspace_id="ws-shell-review",
        review_registry=registry,
    )

    handler._reset_for_new_request()
    await handler._capture_turn_workspace_baseline()
    handler._workspace_snapshot.mark_stale()

    await handler._publish_turn_review("create new.py from shell")

    bundle = registry.get(handler._state.active_turn_id)
    assert bundle is not None
    assert len(bundle.files) == 1
    assert bundle.files[0].path == "/workspace/new.py"
    assert bundle.files[0].status == "added"
    assert bundle.files[0].reversible is True


@pytest.mark.asyncio
async def test_publish_turn_review_ignores_internal_scratch_files():
    """Internal .augmentum scratch files should not surface in review UI."""
    from augmentum.modes.coder.handler import CoderHandler

    cm = _ReviewScriptedContainer(
        [
            _wc_output(("a.py", 10)),
            _wc_output(
                ("a.py", 10),
                ("new.py", 1),
                (".augmentum/scratch/file_read-abc.txt", 5),
            ),
        ],
        {
            "/workspace/new.py": "print('hi')\n",
            "/workspace/.augmentum/scratch/file_read-abc.txt": "oversized output\n",
        },
    )
    registry = ReviewRegistry()
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="ws-shell-review-filtered",
        container_manager=cm,
        workspace_id="ws-shell-review-filtered",
        review_registry=registry,
    )

    handler._reset_for_new_request()
    await handler._capture_turn_workspace_baseline()
    handler._workspace_snapshot.mark_stale()

    await handler._publish_turn_review("explain the project files")

    bundle = registry.get(handler._state.active_turn_id)
    assert bundle is not None
    assert [f.path for f in bundle.files] == ["/workspace/new.py"]
