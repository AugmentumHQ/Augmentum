"""Tests for the 2026-04-20 context-dedup pass.

Two sources of bloat identified in the audit of a "make me a snake game"
session and removed here:

  1. ``build_repo_map`` was always emitting its own ``## Workspace Files``
     listing, which the ``WorkspaceSnapshot`` block already covers more
     completely (state-complete, auto-refreshed, [NEW]/[MOD]/[DEL]
     markers). Duplicating the listing wasted ~500-750 tokens per turn
     AND risked the model picking the wrong source when they diverged.
     Fix: ``skip_file_listing=True`` param; handler passes it when the
     snapshot ran.

  2. ``TOOL_REFERENCE`` (~1.5k tokens) was inlined into ``ACT_SYSTEM`` for
     every coder turn, native tier or not. On native tier the tools are
     already passed structurally via the request's ``tools`` field —
     inlining the prose was pure duplication. Fix: split ``ACT_SYSTEM``
     (rules only) from ``ACT_SYSTEM_WITH_TOOLS`` (rules + catalog), and
     let the handler pick by tier.

  3. ``EDIT_FORMAT_INSTRUCTIONS`` (~125 tokens) — the SEARCH/REPLACE
     contract prose — was appended after ``ACT_SYSTEM`` on every turn.
     The same info now lives in ``CodeEditTool.description`` (native
     tier sees it via the schema) and in ``TOOL_REFERENCE`` (text tier
     sees it there). Handler no longer appends it.

Run: python -m pytest tests/test_coder_prompt_dedup.py -v
"""
from __future__ import annotations

import pytest

from augmentum.coder.prompts import (
    ACT_SYSTEM,
    ACT_SYSTEM_WITH_TOOLS,
    MISSION_ACT_SYSTEM,
    MISSION_ACT_SYSTEM_WITH_TOOLS,
    TOOL_REFERENCE,
)
from augmentum.coder.repomap import build_repo_map
from augmentum.modes.coder.handler import (
    _act_system_for_tier,
    _mission_act_system_for_tier,
)
from augmentum.modes.analytical.tool_calling import ToolCallingTier


# ---------------------------------------------------------------------------
# build_repo_map — skip_file_listing flag
# ---------------------------------------------------------------------------


class _StubContainerManager:
    """Returns canned shell output so build_repo_map runs without Docker."""

    def __init__(self, *, find_output: str = "", grep_output: str = "") -> None:
        self.find_output = find_output
        self.grep_output = grep_output
        self.commands: list[str] = []

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, workspace_id, cmd, timeout=None):
        cmd_str = cmd[-1] if isinstance(cmd, list) else str(cmd)
        self.commands.append(cmd_str)
        if "find" in cmd_str and "wc -l" in cmd_str:
            return self.find_output
        return self.grep_output


@pytest.mark.asyncio
async def test_repo_map_file_listing_present_by_default():
    """Default behaviour (backwards compat): the listing is emitted."""
    cm = _StubContainerManager(
        find_output="  42 /workspace/main.py\n  10 /workspace/utils.py",
        grep_output="/workspace/main.py:1:def foo():",
    )
    out = await build_repo_map(cm, "ws", query="foo")
    assert "## Workspace Files" in out
    assert "main.py (42L)" in out


@pytest.mark.asyncio
async def test_repo_map_skips_listing_when_flagged():
    """When the caller (handler) already has the snapshot-based tree,
    repo_map should drop its own listing — two sources of truth for
    the same data invites divergence."""
    cm = _StubContainerManager(
        find_output="  42 /workspace/main.py\n  10 /workspace/utils.py",
        grep_output="/workspace/main.py:1:def foo():",
    )
    out = await build_repo_map(
        cm, "ws", query="foo", skip_file_listing=True,
    )
    assert "## Workspace Files" not in out
    # Listing block missing — but definitions section should still appear
    assert "main.py (42L)" not in out


@pytest.mark.asyncio
async def test_repo_map_definitions_preserved_when_listing_skipped():
    """Skipping the listing must NOT skip definitions — that's the
    whole point: keep what's unique to repo_map, drop what duplicates
    the snapshot."""
    cm = _StubContainerManager(
        find_output="  42 /workspace/main.py",
        grep_output="/workspace/main.py:1:def foo():\n/workspace/main.py:5:def bar():",
    )
    out = await build_repo_map(
        cm, "ws", query="foo", skip_file_listing=True,
    )
    # The "## Key Definitions" header and the parsed defs should still
    # be present — this is the value repo_map adds over the snapshot.
    assert "Key Definitions" in out
    assert "foo" in out or "bar" in out


# ---------------------------------------------------------------------------
# Tier-aware system prompt selection
# ---------------------------------------------------------------------------


def test_native_tier_skips_tool_reference():
    """On native tier, tools are passed as a structural schema — no
    need to duplicate them in the system prompt."""
    sys_prompt = _act_system_for_tier(ToolCallingTier.NATIVE)
    # ACT_SYSTEM is what we want (rules only)
    assert sys_prompt == ACT_SYSTEM
    # TOOL_REFERENCE signatures should NOT appear
    assert "## Tool Reference" not in sys_prompt
    # A distinctive line from the TOOL_REFERENCE catalog:
    assert "Show directory hierarchy" not in sys_prompt


def test_structured_tier_includes_tool_reference():
    """On structured / text tier the model can't see a native tool
    schema, so the inline catalog IS necessary."""
    sys_prompt = _act_system_for_tier(ToolCallingTier.STRUCTURED)
    assert sys_prompt == ACT_SYSTEM_WITH_TOOLS
    assert "## Tool Reference" in sys_prompt
    assert "Show directory hierarchy" in sys_prompt


def test_text_tier_includes_tool_reference():
    sys_prompt = _act_system_for_tier(ToolCallingTier.TEXT)
    assert sys_prompt == ACT_SYSTEM_WITH_TOOLS
    assert "## Tool Reference" in sys_prompt


def test_act_system_core_preserves_critical_rules():
    """The rules block (STOP IMMEDIATELY, file_read before code_edit,
    etc.) must survive in BOTH variants — those are the guardrails
    that prevent the model from running away."""
    for prompt in (ACT_SYSTEM, ACT_SYSTEM_WITH_TOOLS):
        assert "STOP IMMEDIATELY" in prompt
        assert "file_read before code_edit" in prompt
        assert "<task_complete/>" in prompt


def test_act_system_native_is_meaningfully_smaller():
    """Sanity: the native variant should be clearly smaller than the
    text variant — that's the whole saving. Threshold picked with
    headroom so minor prose edits don't break the test."""
    native = _act_system_for_tier(ToolCallingTier.NATIVE)
    text = _act_system_for_tier(ToolCallingTier.TEXT)
    saved = len(text) - len(native)
    assert saved > 1000, (
        f"Native variant should save >1000 chars vs text; saved {saved}. "
        f"If TOOL_REFERENCE shrank, adjust the bound — don't delete the "
        f"test; it's the regression guard for this optimisation."
    )


def test_mission_native_also_skips_tool_reference():
    """Parity: mission strategy gets the same tier-aware treatment so
    _act_mission's prompt isn't 1.5k tokens heavier than _act_hybrid's."""
    native = _mission_act_system_for_tier(ToolCallingTier.NATIVE)
    text = _mission_act_system_for_tier(ToolCallingTier.TEXT)
    assert native == MISSION_ACT_SYSTEM
    assert text == MISSION_ACT_SYSTEM_WITH_TOOLS
    assert "## Tool Reference" not in native
    assert "## Tool Reference" in text


# ---------------------------------------------------------------------------
# CodeEditTool.description carries the SEARCH/REPLACE contract
# ---------------------------------------------------------------------------


def test_code_edit_tool_description_has_tier_contract():
    """Native tier sees this description via the tool schema — it must
    include the SEARCH/REPLACE rules that EDIT_FORMAT_INSTRUCTIONS used
    to provide in-prompt. Otherwise native-tier models lose that info
    when the separate EDIT_FORMAT block goes away."""
    from augmentum.coder.tools import CodeEditTool

    # Build a throwaway instance just for the description — no
    # container needed to read the property.
    class _DummyCM:
        pass

    class _DummyState:
        pass

    tool = CodeEditTool(
        container_manager=_DummyCM(),
        workspace_id="ws",
        state=_DummyState(),
    )
    desc = tool.description
    # The contract bits EDIT_FORMAT_INSTRUCTIONS used to own
    assert "SEARCH/REPLACE" in desc
    assert "read-before-edit" in desc or "file_read" in desc
    # Tier list
    assert "exact" in desc and "fuzzy" in desc
    # At least one tip
    assert "context" in desc.lower()
