"""Regression test: code_edit is idempotent on re-sent identical edits.

Observed 2026-04-22 in the Pong-game transcript:

  1. Model calls code_edit(path=X, search=A, replace=B) — succeeds,
     file on disk now contains B.
  2. Backend glitches / scratch-reference confusion / model mis-reads
     its own context → model retries the same code_edit.
  3. Tool looks for A in the file, doesn't find it (it's now B),
     returns "No match found" as an error.
  4. Model reads the error as "something is wrong", retries AGAIN —
     compounding into a ~6-iteration thrash loop before a streak
     breaker fires.

Fix: when ``search`` is absent from the file but ``replace`` is
present (and the two differ), the edit is already applied — return
success with ``metadata={"no_op": True, "tier": "already_applied"}``.
The model gets a clear "yes, that's done" signal instead of a
misleading failure.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import CodeEditTool


def _make_cm(content: str) -> MagicMock:
    cm = MagicMock()
    cm.file_read = AsyncMock(return_value=content)
    cm.file_write = AsyncMock(return_value=None)
    cm._run_command = AsyncMock(return_value=f"{hash(content):x}")
    cm.run_command = cm._run_command
    return cm


def _make_state_with_read(path: str) -> CoderState:
    state = CoderState(session_id="s", workspace_id="w")
    # Mark the file as already read in this turn so the mtime guard
    # doesn't reject the edit. Real state.files_read is dict path→mtime.
    state.files_read[path] = float("inf")
    return state


@pytest.mark.asyncio
async def test_idempotent_on_already_applied_edit():
    """File already in end state → success with no_op marker."""
    path = "/workspace/app.py"
    post_edit_content = (
        "def authenticate(user, password):\n"
        "    return validate_password(user, password)\n"
    )
    cm = _make_cm(post_edit_content)
    state = _make_state_with_read(path)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)

    # Pre-fix search text no longer exists; post-fix replace does
    result = await tool.execute(
        path=path,
        search="def authenticate(user):\n    return user.password",
        replace="def authenticate(user, password):\n    return validate_password(user, password)",
    )

    assert result.success
    assert result.metadata.get("no_op") is True
    assert result.metadata.get("tier") == "already_applied"
    # No write was issued — the file is already correct
    cm.file_write.assert_not_called()
    # User-facing output says the situation plainly
    assert "already applied" in result.output.lower() or "no-op" in result.output.lower()


@pytest.mark.asyncio
async def test_normal_edit_still_applies():
    """Regression: this fix must NOT interfere with normal edits."""
    path = "/workspace/app.py"
    pre_edit_content = (
        "def authenticate(user):\n"
        "    return user.password\n"
    )
    cm = _make_cm(pre_edit_content)
    state = _make_state_with_read(path)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path=path,
        search="def authenticate(user):\n    return user.password",
        replace="def authenticate(user, password):\n    return validate_password(user, password)",
    )

    assert result.success
    assert result.metadata.get("no_op") is not True
    # Actual write happened with the new content
    cm.file_write.assert_called_once()


@pytest.mark.asyncio
async def test_short_replace_does_not_trigger_idempotence():
    """False-positive guard: trivially-short replacements like ``x`` →
    ``y`` would match by coincidence. Require >=20 chars to claim
    idempotence."""
    path = "/workspace/v.py"
    # File contains "y" but no "x" — a short replace with these values
    # would spuriously trigger idempotence. Length guard prevents this.
    cm = _make_cm("y\n")
    state = _make_state_with_read(path)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(path=path, search="x", replace="y")

    # Short replace → falls through to normal tier matching. "x" isn't
    # in the file, so it's a real "no match" error (not idempotence).
    assert not result.success
    assert "no match found" in result.error.lower()


@pytest.mark.asyncio
async def test_identical_search_and_replace_does_not_trigger():
    """Edge case: search == replace means the model sent a no-op edit
    anyway. Don't claim idempotence for this — let the tier matcher
    handle it normally."""
    path = "/workspace/a.py"
    content = "identical_block_appearing_in_file_verbatim_here_ok\n"
    cm = _make_cm(content)
    state = _make_state_with_read(path)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)

    identical = "identical_block_appearing_in_file_verbatim_here_ok"
    result = await tool.execute(
        path=path, search=identical, replace=identical,
    )

    # search == replace — not an idempotence case, runs normal path.
    assert result.success
    assert result.metadata.get("no_op") is not True
