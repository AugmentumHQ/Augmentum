"""Tests for agentic mode UI integration (Phase E).

Tests the data flow from handler metadata through to expected UI structures.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)


# ---------------------------------------------------------------------------
# Mode prefix tests
# ---------------------------------------------------------------------------


class TestModePrefix:
    def test_agentic_prefix_in_mode_map(self):
        from augmentum.classifier.router import MODE_PREFIXES

        assert "g/" in MODE_PREFIXES

    def test_agentic_prefix_maps_to_agentic_mode(self):
        from augmentum.classifier.router import MODE_PREFIXES, Mode

        assert MODE_PREFIXES["g/"] == Mode.AGENTIC

    def test_mode_enum_has_agentic(self):
        from augmentum.classifier.router import Mode

        assert hasattr(Mode, "AGENTIC")
        assert Mode.AGENTIC.value == "agentic"


# ---------------------------------------------------------------------------
# Stream chunk metadata tests
# ---------------------------------------------------------------------------


class TestAgenticStreamChunks:
    def test_meta_chunk_has_mode(self):
        from augmentum.modes.agentic.handler import _agentic_meta_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", title="Test")
        chunk = _agentic_meta_chunk("model", "running", task, "content")
        assert chunk.augmentum["mode"] == "agentic"
        assert chunk.augmentum["task_status"] == "running"
        assert chunk.augmentum["task_title"] == "Test"

    def test_meta_chunk_has_progress(self):
        from augmentum.modes.agentic.handler import _agentic_meta_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", title="Test", current_step=2, total_steps=5,
                        step_outputs={0: "done", 1: "done"})
        chunk = _agentic_meta_chunk("model", "running", task)
        assert chunk.augmentum["progress"] == pytest.approx(40.0)
        assert chunk.augmentum["current_step"] == 2
        assert chunk.augmentum["total_steps"] == 5

    def test_flow_step_chunk_has_phases(self):
        from augmentum.modes.agentic.handler import _flow_step_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", title="Test")
        pipeline = ["Plan", "Research", "Draft", "Create"]
        chunk = _flow_step_chunk("model", "Research", "running", pipeline, task)
        phases = chunk.augmentum["phases"]
        assert len(phases) == 4
        assert phases[0]["status"] == "complete"  # Plan
        assert phases[1]["status"] == "running"  # Research
        assert phases[2]["status"] == "pending"  # Draft
        assert phases[3]["status"] == "pending"  # Create

    def test_approval_chunk_has_request(self):
        from augmentum.modes.agentic.autonomy import build_approval_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", title="My Task", current_step=3, total_steps=7)
        chunk = build_approval_chunk("model", task, "Create Report", "create", "Generating PDF")
        assert chunk.augmentum["task_status"] == "approval_pending"
        assert chunk.augmentum["approval_request"]["step_name"] == "Create Report"
        assert chunk.augmentum["approval_request"]["step_role"] == "create"
        assert chunk.augmentum["approval_request"]["current_step"] == 3
        assert chunk.augmentum["approval_request"]["total_steps"] == 7

    def test_plan_approval_chunk_has_plan_md(self):
        from augmentum.modes.agentic.autonomy import build_plan_approval_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(
            id="t1", title="Report",
            plan_md="- [ ] 1. Research\n- [ ] 2. Write",
            total_steps=2,
        )
        chunk = build_plan_approval_chunk("model", task)
        assert chunk.augmentum["plan_md"] == task.plan_md
        assert chunk.augmentum["approval_request"]["step_role"] == "plan"

    def test_inform_chunk_has_action(self):
        from augmentum.modes.agentic.autonomy import build_inform_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", title="Test")
        chunk = build_inform_chunk("model", task, "Create File", "Generated report.pdf")
        # Inform events ride on the real task lifecycle status — the
        # informed action is carried on the dedicated sub-event payload.
        assert chunk.augmentum["task_status"] == task.status.value
        assert chunk.augmentum["informed_action"]["step_name"] == "Create File"
        assert chunk.augmentum["informed_action"]["action"] == "Generated report.pdf"


# ---------------------------------------------------------------------------
# Artifact routes tests
# ---------------------------------------------------------------------------


class TestArtifactRouteConfig:
    def test_artifact_routes_exist(self):
        from augmentum.proxy.artifact_routes import router

        routes = [r.path for r in router.routes]
        assert any("download" in r for r in routes)


# ---------------------------------------------------------------------------
# Handler factory wiring
# ---------------------------------------------------------------------------


class TestHandlerFactoryAgentic:
    def test_agentic_mode_creates_handler(self):
        from augmentum.classifier.router import Mode
        from augmentum.modes.agentic.handler import AgenticHandler

        # Verify Mode.AGENTIC is properly defined
        assert Mode.AGENTIC.value == "agentic"

        # Verify AgenticHandler can be instantiated
        backend = AsyncMock()
        handler = AgenticHandler(backend=backend, session_id="test")
        assert handler is not None


# ---------------------------------------------------------------------------
# Task state progress calculation
# ---------------------------------------------------------------------------


class TestTaskProgress:
    def test_progress_pct_zero(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", current_step=0, total_steps=5)
        assert task.progress_pct == 0.0

    def test_progress_pct_partial(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", current_step=3, total_steps=6,
                        step_outputs={0: "a", 1: "b", 2: "c"})
        assert task.progress_pct == pytest.approx(50.0)

    def test_progress_pct_complete(self):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        task = TaskState(id="t1", current_step=5, total_steps=5,
                        status=TaskStatus.COMPLETED)
        assert task.progress_pct == pytest.approx(100.0)

    def test_progress_pct_no_steps(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", current_step=0, total_steps=0)
        assert task.progress_pct == 0.0


# ---------------------------------------------------------------------------
# Planner plan_to_context for attention anchor
# ---------------------------------------------------------------------------


class TestPlanAttentionAnchor:
    def test_plan_to_context_wraps_plan(self):
        from augmentum.modes.agentic.planner import plan_to_context

        plan_md = "## Task: Test\n- [ ] 1. Step one\n- [ ] 2. Step two"
        ctx = plan_to_context(plan_md)
        assert "## Current Plan" in ctx or plan_md in ctx
        assert "Step one" in ctx

    def test_plan_to_context_empty(self):
        from augmentum.modes.agentic.planner import plan_to_context

        ctx = plan_to_context("")
        assert ctx == "" or ctx.strip() == ""

    def test_mark_current_step(self):
        from augmentum.modes.agentic.planner import mark_current_step

        plan = "- [ ] 1. First\n- [ ] 2. Second\n- [ ] 3. Third"
        marked = mark_current_step(plan, 1)
        assert "CURRENT" in marked
        # Step 2 (index 1) should be marked
        lines = marked.split("\n")
        assert "CURRENT" in lines[1]
