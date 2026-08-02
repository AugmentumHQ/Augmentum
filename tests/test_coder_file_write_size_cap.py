"""Tests for the opt-in oversized file_write rejection.

The default (``coder_file_write_max_tokens = 0``) is uncapped — this
matches Claude Code's ``Write`` and Codex CLI's ``apply_patch``, and
relies on D1 (truncation-detection in the coder handler) to catch
the mid-arguments cutoff at runtime with a structured recovery hint.

Tests below cover the opt-in path: when a workspace owner sets a
positive cap (typical reason: a weak local model with a tiny output
budget), the tool refuses oversized writes pre-emptively with the
right recovery message (existing file → code_edit_batch, new file →
skeleton-first). The final test asserts that 0 keeps the path
uncapped — the production default.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import FileWriteTool


def _state() -> CoderState:
    return CoderState(session_id="sess", workspace_id="ws")


def _cm() -> MagicMock:
    cm = MagicMock()
    cm.file_write = AsyncMock(return_value=None)
    cm.run_command = AsyncMock(return_value="")  # stat returns blank → new file
    cm._run_command = cm.run_command
    return cm


@pytest.mark.asyncio
async def test_small_content_passes_size_cap():
    """Normal-sized writes (well under the cap) work as before."""
    tool = FileWriteTool(container_manager=_cm(), workspace_id="ws", state=_state())
    result = await tool.execute(
        path="/workspace/hello.py", content="print('hello')\n",
    )
    assert result.success


@pytest.mark.asyncio
async def test_oversized_content_refused(monkeypatch):
    """Content over the configured cap is refused with a redirect."""
    # Tight cap so the test doesn't need to allocate megabytes of text.
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "coder_file_write_max_tokens", 100)

    # ~600 tokens of text — well over the 100 cap.
    huge_content = "function bar() {\n  return 42;\n}\n" * 200

    tool = FileWriteTool(container_manager=_cm(), workspace_id="ws", state=_state())
    result = await tool.execute(
        path="/workspace/huge.js", content=huge_content,
    )
    assert not result.success
    assert result.validation_error
    assert "tokens >" in result.error  # token count > cap diagnostic
    assert "code_edit" in result.error  # points at recovery path


@pytest.mark.asyncio
async def test_size_cap_runs_before_read_guard(monkeypatch):
    """When BOTH defenses would refuse (oversized AND unread existing
    file), the size cap fires first. Reason: an unread huge rewrite is
    the exact problem we're trying to prevent, and the size diagnostic
    is the more actionable message — the model needs to learn "don't
    rewrite, edit" before it learns "read first"."""
    monkeypatch.setattr(
        "augmentum.config.settings.coder_file_write_max_tokens", 50,
    )

    # Simulate an existing file (would normally trigger the D2 read
    # guard since we never recorded a file_read).
    cm = _cm()
    cm.run_command = AsyncMock(return_value="1700000000\n")
    cm._run_command = cm.run_command

    huge_content = "x" * 5000  # very over the 50-token cap

    tool = FileWriteTool(container_manager=cm, workspace_id="ws", state=_state())
    result = await tool.execute(
        path="/workspace/existing.py", content=huge_content,
    )
    assert not result.success
    # Size diagnostic, not the read-before-edit one.
    assert "tokens >" in result.error
    # The cap path stats the path once to differentiate the recovery
    # message (existing file → code_edit_batch; new file → skeleton).
    # That single stat is fine; what we must NOT do is run the
    # read-before-edit guard, which would surface the wrong error.
    assert "read the file first" not in result.error.lower()
    assert "has been modified" not in result.error.lower()


@pytest.mark.asyncio
async def test_oversized_existing_file_recommends_code_edit_batch(monkeypatch):
    """The cap message for an EXISTING file leads with code_edit_batch,
    not file_write splitting. That's the wiring fix from the 2026-05-29
    transcript: the model that hit the cap kept retrying smaller
    file_writes (still over cap) because the old message listed
    splitting first. New copy steers existing-file rewrites toward
    targeted SEARCH/REPLACE blocks."""
    monkeypatch.setattr(
        "augmentum.config.settings.coder_file_write_max_tokens", 50,
    )

    cm = _cm()
    # mtime present → existing file
    cm.run_command = AsyncMock(return_value="1700000000\n")
    cm._run_command = cm.run_command

    tool = FileWriteTool(container_manager=cm, workspace_id="ws", state=_state())
    result = await tool.execute(
        path="/workspace/existing.py", content="x" * 5000,
    )
    assert not result.success
    # Primary recommendation names code_edit_batch.
    assert "code_edit_batch" in result.error
    # Split-into-multiple-file_writes is demoted to "last resort".
    assert "Last resort" in result.error


@pytest.mark.asyncio
async def test_oversized_new_file_recommends_skeleton(monkeypatch):
    """For a NEW file the message recommends writing a small skeleton
    first then filling via code_edit, since there's nothing to
    SEARCH/REPLACE against yet."""
    monkeypatch.setattr(
        "augmentum.config.settings.coder_file_write_max_tokens", 50,
    )

    # _cm() returns blank from run_command → no mtime → new file.
    tool = FileWriteTool(container_manager=_cm(), workspace_id="ws", state=_state())
    result = await tool.execute(
        path="/workspace/new.py", content="x" * 5000,
    )
    assert not result.success
    assert "skeleton" in result.error.lower()
    assert "code_edit" in result.error


@pytest.mark.asyncio
async def test_size_cap_zero_disables_check(monkeypatch):
    """Cap=0 is uncapped — the production default. Validates that the
    pre-emptive refusal is fully bypassed, leaving D1 truncation
    detection in the handler as the runtime defense (matching Claude
    Code / Codex CLI behavior)."""
    monkeypatch.setattr(
        "augmentum.config.settings.coder_file_write_max_tokens", 0,
    )

    huge_content = "y" * 100000

    tool = FileWriteTool(container_manager=_cm(), workspace_id="ws", state=_state())
    result = await tool.execute(
        path="/workspace/big.txt", content=huge_content,
    )
    assert result.success
