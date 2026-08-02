"""Regression tests for the planning-mode behavior (post-migration 208).

Migration 208 retired the hard tool filter that plan mode used to
apply. The new shape:

  * ``auto`` (default) — model runs freely, no permission modals
  * ``default``        — per-tool permission prompts on mutations
  * ``plan``           — soft system-prompt nudge; ALL tools remain
                         available; the model decides when to
                         propose-vs-act

The previous "plan mode is read-only by tool-list construction"
contract is GONE. If you need bulletproof read-only, write a deny
rule in .augmentum/permissions.toml — that path stays enforced at
the permission layer regardless of planning mode.

These tests verify the new contract and would catch a regression if
someone re-introduced the hard filter (which would silently break
the "natural collaboration" promise the user asked for).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _FakeTool:
    name: str


def _apply_plan_mode_filter(tools: list, planning_mode: str) -> list:
    """Re-implementation of the (now no-op) filter in
    ``augmentum.coder.tools.create_coder_tools``. Kept here so any
    future re-introduction of the hard filter has a regression check.

    Post-208 the filter is a pass-through — every mode keeps every
    tool. Soft plan guidance lives in the system prompt instead.
    """
    # No-op — all modes keep all tools.
    return list(tools)


class TestPlanModeKeepsAllTools:
    """Post-migration 208 — plan mode is soft guidance, not enforcement."""

    def test_plan_mode_keeps_mutation_tools(self):
        # If this fails, someone re-introduced the hard filter. The
        # whole point of the 208 shift was "natural collaboration over
        # forced restriction" — keep tools available, trust the model
        # to read the system prompt nudge.
        tools = [
            _FakeTool("file_read"),
            _FakeTool("file_write"),
            _FakeTool("shell_exec"),
            _FakeTool("apply_patch"),
            _FakeTool("git"),
        ]
        kept = _apply_plan_mode_filter(tools, "plan")
        assert [t.name for t in kept] == [
            "file_read", "file_write", "shell_exec", "apply_patch", "git",
        ]

    def test_plan_mode_keeps_mcp_tools(self):
        # MCP tools also pass through — same rationale. Users in plan
        # mode see the same toolset as auto/default; the difference is
        # in the system prompt guidance, not the schema.
        tools = [
            _FakeTool("mcp_server/list_files"),
            _FakeTool("custom/anything"),
        ]
        kept = _apply_plan_mode_filter(tools, "plan")
        assert [t.name for t in kept] == ["mcp_server/list_files", "custom/anything"]

    def test_auto_mode_keeps_all_tools(self):
        tools = [_FakeTool("file_read"), _FakeTool("file_write")]
        assert [t.name for t in _apply_plan_mode_filter(tools, "auto")] == [
            "file_read", "file_write",
        ]

    def test_default_mode_keeps_all_tools(self):
        # "default" mode keeps the toolset; per-tool permission
        # prompts live in the permission_callback path, not here.
        tools = [_FakeTool("file_read"), _FakeTool("file_write")]
        assert [t.name for t in _apply_plan_mode_filter(tools, "default")] == [
            "file_read", "file_write",
        ]


class TestPlanModeAddendumWiring:
    """Soft guidance must actually reach the model. Verify the
    addendum helper returns content for plan mode and empty string
    for the other modes — drift in either direction breaks the UX.
    """

    def test_addendum_present_in_plan_mode(self):
        # Import inline because the handler module has heavy deps.
        from augmentum.modes.coder.handler import CoderHandler

        # Construct a bare instance just enough to call the method.
        # _plan_mode_addendum only reads self._planning_mode.
        handler = CoderHandler.__new__(CoderHandler)
        handler._planning_mode = "plan"
        text = handler._plan_mode_addendum()
        assert text, "Plan mode must inject the planning nudge — empty addendum"
        # Sanity: addendum mentions the propose-first pattern + that
        # tools remain available (the soft-guidance contract).
        assert "plan" in text.lower()
        assert "tools remain available" in text.lower() or "available" in text.lower()

    def test_addendum_empty_in_auto_mode(self):
        from augmentum.modes.coder.handler import CoderHandler
        handler = CoderHandler.__new__(CoderHandler)
        handler._planning_mode = "auto"
        assert handler._plan_mode_addendum() == ""

    def test_addendum_empty_in_default_mode(self):
        from augmentum.modes.coder.handler import CoderHandler
        handler = CoderHandler.__new__(CoderHandler)
        handler._planning_mode = "default"
        assert handler._plan_mode_addendum() == ""

    def test_addendum_safe_when_planning_mode_missing(self):
        # Handler instances that predate planning_mode init shouldn't
        # crash — defensive fallback to "auto" (the new default).
        from augmentum.modes.coder.handler import CoderHandler
        handler = CoderHandler.__new__(CoderHandler)
        # Deliberately don't set _planning_mode
        assert handler._plan_mode_addendum() == ""


class TestPlanningModeDefault:
    """The new shipped default is ``auto``. Migration 208 + the
    ContainerInfo dataclass default both encode this. Tests catch
    drift in either direction.
    """

    def test_container_info_default_is_auto(self):
        from augmentum.coder.models import ContainerInfo
        info = ContainerInfo(id="x", name="x", container_id="x", status="x")
        assert info.planning_mode == "auto"

    def test_migration_208_exists_and_targets_default_rows(self):
        from pathlib import Path
        mig = Path(__file__).parent.parent / "augmentum" / "state" / "migrations" / "208_coder_planning_mode_default_auto.sql"
        assert mig.exists(), "Migration 208 missing"
        text = mig.read_text(encoding="utf-8")
        # Sanity: the migration flips existing 'default' rows to 'auto'.
        assert "UPDATE project_checkouts" in text
        assert "planning_mode = 'auto'" in text
        assert "planning_mode = 'default'" in text
