"""Tests for the apply_patch post-write verify+lint gap closure.

Before 2026-05-31, ``apply_patch`` skipped the verify+lint hooks that
``file_write`` / ``code_edit`` / ``code_edit_batch`` all run after
every write. This let multi-file patches land syntactically broken
Python/JS/JSON and the model only discovered the break on the next
test/run — wasting a whole turn.

These tests pin the wiring: after a successful apply, the tool reads
each touched file back and runs the same per-file verify gate the
other write tools use, returning the failure summary in the tool
output and flagging ``verification_failed`` in metadata for the
handler's downstream chunk emission.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import ApplyPatchTool


def _state() -> CoderState:
    return CoderState(session_id="sess", workspace_id="ws")


def _cm_with_apply_outcome(*, file_bodies: dict[str, str]) -> MagicMock:
    """Container manager mock that fakes a successful patch apply +
    serves back the given file bodies via file_read."""
    cm = MagicMock()
    # Patch upload + git apply check + git apply both succeed.
    cm.file_upload = AsyncMock(return_value=None)

    async def _run_command(workspace_id, argv, *, timeout=30.0):
        cmd = " ".join(argv) if isinstance(argv, list) else str(argv)
        if "git apply --check" in cmd:
            return ""  # patch check OK
        if "git apply" in cmd:
            # Mock git status --short output for the changed-file listing.
            return "\n".join(f" M {p[len('/workspace/'):]}" for p in file_bodies)
        if "stat -c" in cmd or "stat --format" in cmd:
            return "1700000000\n"
        return ""

    cm.run_command = AsyncMock(side_effect=_run_command)
    cm._run_command = cm.run_command

    async def _file_read(workspace_id, path):
        return file_bodies.get(path, "")

    cm.file_read = AsyncMock(side_effect=_file_read)

    # Mark all files as already-read so the read-before-edit guard
    # doesn't bounce the patch.
    return cm


@pytest.mark.asyncio
async def test_apply_patch_runs_verify_on_changed_files():
    """A successful patch that lands valid Python passes verify."""
    files = {"/workspace/foo.py": "def foo():\n    return 1\n"}
    cm = _cm_with_apply_outcome(file_bodies=files)
    state = _state()
    # Pre-record file_read so the read-before-edit guard passes.
    state.record_file_read("/workspace/foo.py", mtime=1700000000)

    tool = ApplyPatchTool(container_manager=cm, workspace_id="ws", state=state)
    patch = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1,2 @@\n"
        "-old\n"
        "+def foo():\n"
        "+    return 1\n"
    )
    result = await tool.execute(patch=patch)
    assert result.success
    assert "verification_failed" not in result.metadata


@pytest.mark.asyncio
async def test_apply_patch_surfaces_syntax_error_in_changed_file():
    """A patch that lands broken Python should still apply (it's a
    successful git apply) BUT the verify gate surfaces the parse
    error in the output so the model sees it immediately."""
    broken_py = "def foo(\n    return 1\n"  # missing close paren
    files = {"/workspace/foo.py": broken_py}
    cm = _cm_with_apply_outcome(file_bodies=files)
    state = _state()
    state.record_file_read("/workspace/foo.py", mtime=1700000000)

    tool = ApplyPatchTool(container_manager=cm, workspace_id="ws", state=state)
    patch = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1,2 @@\n"
        "-old\n"
        f"+{broken_py}\n"
    )
    result = await tool.execute(patch=patch)
    # Apply itself succeeded; the model sees the parse error inline.
    assert result.success
    assert "[verify:" in result.output or "verify" in result.output.lower()
    assert result.metadata.get("verification_failed") is True


@pytest.mark.asyncio
async def test_apply_patch_skips_deleted_files_in_verify():
    """A delete-only patch shouldn't crash verify (nothing to read)."""
    cm = MagicMock()
    cm.file_upload = AsyncMock(return_value=None)

    async def _run_command(workspace_id, argv, *, timeout=30.0):
        cmd = " ".join(argv) if isinstance(argv, list) else str(argv)
        if "git apply --check" in cmd:
            return ""
        if "git apply" in cmd:
            return " D removed.py"
        return ""

    cm.run_command = AsyncMock(side_effect=_run_command)
    cm._run_command = cm.run_command
    cm.file_read = AsyncMock(return_value="")

    state = _state()
    state.record_file_read("/workspace/removed.py", mtime=1700000000)
    tool = ApplyPatchTool(container_manager=cm, workspace_id="ws", state=state)

    patch = (
        "diff --git a/removed.py b/removed.py\n"
        "--- a/removed.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-gone\n"
    )
    result = await tool.execute(patch=patch)
    assert result.success
