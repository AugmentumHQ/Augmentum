"""Autonomy level enforcement for agentic mode.

Controls how much user approval the agent seeks before acting.

Levels:
    1 - Suggest:    Propose plan, wait for approval at every step.
    2 - Ask:        Execute freely, pause before high-impact actions
                    (file creation, tool calls producing artifacts).
    3 - Inform:     Execute everything, report what was done.
    4 - Autonomous: Run to completion silently, user sees final result.
"""

from __future__ import annotations

from augmentum.models.base import InternalStreamChunk
from augmentum.modes.agentic.task_state import TaskState
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Step roles considered "high impact" — produce external artifacts or side effects.
# "deliver" excluded: it only presents results; "create" and "illustrate" create files.
_HIGH_IMPACT_ROLES = frozenset({"create", "illustrate"})

# Minimum number of tool calls in a step before it's considered high-impact
_TOOL_CALL_THRESHOLD = 3


def needs_plan_approval(autonomy_level: int) -> bool:
    """Level 1 (Suggest) requires explicit plan approval before execution."""
    return autonomy_level <= 1


def needs_step_approval(
    autonomy_level: int,
    step_role: str,
    tool_calls_in_step: int = 0,
) -> bool:
    """Determine if a step requires user approval before execution.

    Level 1: Every step needs approval.
    Level 2: Only high-impact steps (create, illustrate, deliver) or
             steps with many tool calls.
    Level 3-4: No step approval needed.
    """
    if autonomy_level <= 1:
        return True

    if autonomy_level == 2:
        if step_role in _HIGH_IMPACT_ROLES:
            return True
        if tool_calls_in_step >= _TOOL_CALL_THRESHOLD:
            return True

    return False


def build_approval_chunk(
    model: str,
    task: TaskState,
    step_name: str,
    step_role: str,
    description: str = "",
) -> InternalStreamChunk:
    """Build a stream chunk requesting user approval.

    The inspector renders an interactive card (Approve / Modify / Skip) from
    the ``approval_request`` meta below. The ``content_delta`` is a minimal
    plaintext summary so API-only / terminal clients that don't parse the
    meta still see a coherent request; the typed-keyword fallback (``approve``,
    ``skip``, free text = modify) remains a valid path for those surfaces.
    """
    envelope = task.meta_envelope(status_override="approval_pending")
    envelope["approval_request"] = {
        "step_name": step_name,
        "step_role": step_role,
        "description": description,
        "task_title": task.title,
        "current_step": task.current_step,
        "total_steps": task.total_steps,
    }
    return InternalStreamChunk(
        content_delta=(
            f"\n**Approval needed:** {step_name}\n\n"
            f"{description}\n"
        ),
        model=model,
        augmentum=envelope,
    )


def build_plan_approval_chunk(
    model: str,
    task: TaskState,
) -> InternalStreamChunk:
    """Build a chunk requesting plan approval (autonomy level 1).

    See ``build_approval_chunk`` — the same dual-surface contract applies:
    content_delta is the plaintext fallback, ``approval_request`` + ``plan_md``
    on the meta drive the inspector card.
    """
    envelope = task.meta_envelope(status_override="approval_pending")
    envelope["approval_request"] = {
        "step_name": "Plan Approval",
        "step_role": "plan",
        "description": f"Execute {task.total_steps} planned steps",
        "task_title": task.title,
        "current_step": 0,
        "total_steps": task.total_steps,
    }
    return InternalStreamChunk(
        content_delta=(
            f"\n**Plan ready for review:** {task.title}\n\n"
            f"{task.plan_md}\n"
        ),
        model=model,
        augmentum=envelope,
    )


def build_inform_chunk(
    model: str,
    task: TaskState,
    step_name: str,
    action_taken: str,
) -> InternalStreamChunk:
    """Build a chunk informing the user about an action taken (autonomy level 3).

    Inform events fire mid-step and don't change the task's lifecycle
    state — emit the real ``task.status`` (rather than a pseudo-status)
    so the inspector's status pill keeps showing "running".
    """
    envelope = task.meta_envelope()
    envelope["informed_action"] = {
        "step_name": step_name,
        "action": action_taken,
    }
    return InternalStreamChunk(
        content_delta=f"*[{step_name}: {action_taken}]*\n",
        model=model,
        augmentum=envelope,
    )
