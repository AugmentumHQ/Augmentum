"""Tests for D2: read-before-edit guard on FileWriteTool + ApplyPatchTool.

CodeEditTool and CodeEditBatchTool already enforce the guard via
``_check_read_freshness``. D2 closes the gap on the two mutating tools
that didn't: FileWriteTool (used to overwrite existing files) and
ApplyPatchTool (used for multi-file diffs).

The cascade in the 2026-05-27 log started exactly here: the model's
first edit attempt was a code_edit with a hallucinated search block
against an unread file. When code_edit bounced, the model pivoted to
file_write — which had no guard, so an overwrite with imagined
contents was only one tool call away. D2 makes that overwrite
structurally impossible.

New-file creation must remain unblocked — the model has nothing to
read first when the path doesn't exist yet.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import ApplyPatchTool, FileWriteTool


def _state() -> CoderState:
    return CoderState(session_id="sess", workspace_id="ws")


def _cm(*, mtime: str = "") -> MagicMock:
    """Build a fake container manager.

    Passing ``mtime`` simulates an existing file at that mtime. An
    empty string makes ``stat`` return blank — same behaviour as
    file-not-found per ``_current_mtime``.
    """
    cm = MagicMock()
    cm.file_write = AsyncMock(return_value=None)
    cm.file_upload = AsyncMock(return_value=None)
    cm.run_command = AsyncMock(return_value=mtime)
    cm._run_command = cm.run_command
    return cm


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------


class TestFileWriteGuard:
    @pytest.mark.asyncio
    async def test_new_file_creation_skips_guard(self):
        """File doesn't exist (stat returns empty) → write allowed
        without any prior read. New-file creation must not be blocked."""
        cm = _cm(mtime="")  # stat -> "" -> _current_mtime returns None
        tool = FileWriteTool(container_manager=cm, workspace_id="ws", state=_state())

        result = await tool.execute(
            path="/workspace/brand_new.py", content="print('first version')",
        )

        assert result.success
        cm.file_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_overwrite_without_read_refuses(self):
        """File exists AND was never read → guard refuses the write.
        This is the exact cascade prevention: model can't blindly
        overwrite a file based on imagined contents."""
        cm = _cm(mtime="1700000000\n")
        state = _state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws", state=state)

        result = await tool.execute(
            path="/workspace/existing.py", content="wholly fabricated rewrite",
        )

        assert not result.success
        assert "haven't read" in result.error or "Re-read" in result.error
        cm.file_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_overwrite_after_read_allowed(self):
        """File exists AND was read this turn → write proceeds."""
        cm = _cm(mtime="1700000000\n")
        state = _state()
        state.record_file_read("/workspace/existing.py", mtime=1700000000)
        tool = FileWriteTool(container_manager=cm, workspace_id="ws", state=state)

        result = await tool.execute(
            path="/workspace/existing.py", content="informed rewrite",
        )

        assert result.success
        cm.file_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stale_read_refuses(self):
        """File was read but mtime is now newer (external edit since
        read) → guard refuses. Same defense as code_edit's existing
        mtime-staleness check."""
        cm = _cm(mtime="1700001000\n")  # newer than stored read mtime
        state = _state()
        state.record_file_read("/workspace/existing.py", mtime=1700000000)
        tool = FileWriteTool(container_manager=cm, workspace_id="ws", state=state)

        result = await tool.execute(
            path="/workspace/existing.py", content="stale rewrite",
        )

        assert not result.success
        cm.file_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_guard_disabled_allows_unread_overwrite(self):
        """When _strict_edit_guard=False the gate is intentionally off
        (config knob for backward compat / tests). Existing
        FileWriteTool tests rely on this — preserve that path."""
        cm = _cm(mtime="1700000000\n")
        tool = FileWriteTool(
            container_manager=cm,
            workspace_id="ws",
            state=_state(),
            strict_edit_guard=False,
        )

        result = await tool.execute(
            path="/workspace/existing.py", content="blind rewrite",
        )

        assert result.success
        cm.file_write.assert_awaited_once()


# ---------------------------------------------------------------------------
# ApplyPatchTool
# ---------------------------------------------------------------------------


PATCH_BODY = """diff --git a/existing.py b/existing.py
--- a/existing.py
+++ b/existing.py
@@ -1 +1 @@
-old
+new
"""


class TestApplyPatchGuard:
    @pytest.mark.asyncio
    async def test_patch_against_existing_unread_file_refuses(self):
        cm = _cm(mtime="1700000000\n")
        tool = ApplyPatchTool(container_manager=cm, workspace_id="ws", state=_state())

        result = await tool.execute(patch=PATCH_BODY)

        assert not result.success
        cm.file_upload.assert_not_called()  # never got past the guard

    @pytest.mark.asyncio
    async def test_patch_against_existing_read_file_passes_guard(self):
        """File was read; the guard passes. Whether the patch itself
        applies is a separate concern — we only care that D2 didn't
        short-circuit before the actual patch logic runs."""
        cm = _cm(mtime="1700000000\n")
        cm.run_command = AsyncMock(return_value="1700000000\n")
        state = _state()
        state.record_file_read("/workspace/existing.py", mtime=1700000000)
        tool = ApplyPatchTool(container_manager=cm, workspace_id="ws", state=state)

        result = await tool.execute(patch=PATCH_BODY)

        # We don't assert success — the patch logic uses git apply
        # which can fail in a mock. We DO assert the guard didn't
        # short-circuit (which would have left file_upload uncalled).
        cm.file_upload.assert_awaited()
