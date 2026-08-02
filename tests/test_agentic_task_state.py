"""Tests for agentic task state tracking and autonomy level gating."""

from __future__ import annotations

from augmentum.modes.agentic.autonomy import (
    build_approval_chunk,
    build_inform_chunk,
    build_plan_approval_chunk,
    needs_plan_approval,
    needs_step_approval,
)
from augmentum.modes.agentic.task_state import TaskState, TaskStatus


class TestTaskState:
    """TaskState dataclass behavior."""

    def test_initial_defaults(self):
        ts = TaskState()
        assert ts.status == TaskStatus.PLANNING
        assert ts.current_step == 0
        assert ts.total_steps == 0
        assert ts.is_complete is False

    def test_advance_step(self):
        ts = TaskState(current_step=0, total_steps=3)
        ts.advance_step()
        assert ts.current_step == 1

    def test_record_step_output(self):
        ts = TaskState()
        ts.record_step_output(0, "Step 0 output")
        ts.record_step_output(1, "Step 1 output")
        assert ts.step_outputs[0] == "Step 0 output"
        assert ts.step_outputs[1] == "Step 1 output"

    def test_is_complete_when_completed(self):
        ts = TaskState(status=TaskStatus.COMPLETED)
        assert ts.is_complete is True

    def test_is_complete_when_failed(self):
        ts = TaskState(status=TaskStatus.FAILED)
        assert ts.is_complete is True

    def test_is_not_complete_when_running(self):
        ts = TaskState(status=TaskStatus.RUNNING)
        assert ts.is_complete is False

    def test_progress_pct_zero_steps(self):
        ts = TaskState(total_steps=0)
        assert ts.progress_pct == 0.0

    def test_progress_pct_partial(self):
        ts = TaskState(total_steps=4, status=TaskStatus.RUNNING)
        ts.record_step_output(0, "done")
        ts.record_step_output(1, "done")
        pct = ts.progress_pct
        assert pct == 50.0

    def test_progress_pct_completed(self):
        ts = TaskState(status=TaskStatus.COMPLETED, total_steps=3)
        assert ts.progress_pct == 100.0

    def test_progress_pct_capped_at_99(self):
        ts = TaskState(total_steps=3, status=TaskStatus.RUNNING)
        ts.record_step_output(0, "done")
        ts.record_step_output(1, "done")
        ts.record_step_output(2, "done")
        # 3/3 = 100, but not COMPLETED status, so capped at 99
        assert ts.progress_pct == 99.0


class TestTaskStatus:
    """TaskStatus enum values."""

    def test_all_statuses(self):
        assert TaskStatus.PLANNING.value == "planning"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.PAUSED.value == "paused"
        assert TaskStatus.APPROVAL_PENDING.value == "approval_pending"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestAutonomyNeedsPlanApproval:
    """Level 1 requires plan approval."""

    def test_level_1_needs_approval(self):
        assert needs_plan_approval(1) is True

    def test_level_2_no_plan_approval(self):
        assert needs_plan_approval(2) is False

    def test_level_3_no_plan_approval(self):
        assert needs_plan_approval(3) is False

    def test_level_4_no_plan_approval(self):
        assert needs_plan_approval(4) is False


class TestAutonomyNeedsStepApproval:
    """Step-level approval gating."""

    def test_level_1_always_needs_approval(self):
        assert needs_step_approval(1, "draft") is True
        assert needs_step_approval(1, "search") is True
        assert needs_step_approval(1, "create") is True

    def test_level_2_needs_approval_for_high_impact(self):
        assert needs_step_approval(2, "create") is True
        assert needs_step_approval(2, "illustrate") is True

    def test_level_2_no_approval_for_low_impact(self):
        assert needs_step_approval(2, "draft") is False
        assert needs_step_approval(2, "search") is False
        assert needs_step_approval(2, "review") is False

    def test_level_2_approval_for_many_tool_calls(self):
        assert needs_step_approval(2, "search", tool_calls_in_step=5) is True

    def test_level_3_no_approval(self):
        assert needs_step_approval(3, "create") is False
        assert needs_step_approval(3, "illustrate") is False

    def test_level_4_no_approval(self):
        assert needs_step_approval(4, "create") is False


class TestBuildApprovalChunk:
    """Approval chunk metadata structure."""

    def test_approval_chunk_has_metadata(self):
        task = TaskState(id="abc123", title="Test Task", current_step=1, total_steps=3)
        chunk = build_approval_chunk(
            model="test-model",
            task=task,
            step_name="Research",
            step_role="search",
            description="Searching the web",
        )
        assert "Approval needed" in chunk.content_delta
        assert chunk.augmentum["task_id"] == "abc123"
        assert chunk.augmentum["task_status"] == "approval_pending"
        assert chunk.augmentum["approval_request"]["step_role"] == "search"


class TestBuildPlanApprovalChunk:
    """Plan approval chunk for level 1."""

    def test_plan_approval_chunk_has_plan(self):
        task = TaskState(
            id="def456",
            title="Build Report",
            plan_md="- [ ] 1. Research\n- [ ] 2. Draft",
            total_steps=2,
        )
        chunk = build_plan_approval_chunk(model="test-model", task=task)
        assert "Plan ready" in chunk.content_delta
        assert "Research" in chunk.content_delta
        assert chunk.augmentum["plan_md"] == task.plan_md


class TestBuildInformChunk:
    """Inform chunk for level 3."""

    def test_inform_chunk_format(self):
        task = TaskState(id="ghi789", title="Auto Task")
        chunk = build_inform_chunk(
            model="test-model",
            task=task,
            step_name="Research",
            action_taken="Searched 3 sources",
        )
        assert "Research" in chunk.content_delta
        assert "Searched 3 sources" in chunk.content_delta
        # Inform events ride on the task's real lifecycle status (so the
        # inspector status pill keeps reading "running" / "planning" / etc.
        # instead of switching to a meaningless "informed" pseudo-status).
        # The action itself is carried on the dedicated sub-event payload.
        assert chunk.augmentum["task_status"] == task.status.value
        assert chunk.augmentum["informed_action"]["step_name"] == "Research"
        assert chunk.augmentum["informed_action"]["action"] == "Searched 3 sources"
