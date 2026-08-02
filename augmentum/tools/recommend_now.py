"""``recommend_now`` — companion verbs architecture, Phase 4.

Surface the companion's current substrate-driven recommendation: the
dominant drive, the activity kind that would satiate it, and a
short reason rooted in observed substrate state. The model invokes
this when the user asks "what should I do?" / "any ideas?" /
"recommend something" — instead of synthesizing a recommendation
from chat context (which drifts), it reads the substrate that
management verbs are already maintaining.

Pure read — no DB writes, no LLM calls. Returns a structured
recommendation the model can summarize naturally in chat.

This is the read-side counterpart to ``propose_action`` (Phase 3c
management verb): propose_action *fires* a proposal on substrate
delta-events; recommend_now *answers* a model query against the
same substrate.
"""

from __future__ import annotations

from typing import Any

from augmentum.config import settings
from augmentum.tools.base import (
    CoreVerbAutonomyClass,
    CoreVerbMetadata,
    CoreVerbSafetyClass,
    CostEnvelope,
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Inverse of activity_selector._CANDIDATE_DRIVES (kept in sync with
# propose_action). For each drive, give the activity kind plus a
# one-line natural-language reason the model can pass through.
_DRIVE_RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "curiosity": ("revisit_thread",
                  "There's a thread you started that hasn't closed."),
    "competence": ("creation",
                   "You've been collecting more than making lately."),
    "connection": ("reach_out",
                   "It's been a while since you checked in with someone."),
    "rest": ("no_op",
             "Honestly? Take a breath. Nothing needs you right now."),
}


def _gate(app_state: Any) -> tuple[bool, ToolResult | None, Any]:
    if not bool(getattr(settings, "companion_runtime_enabled", False)):
        return False, ToolResult(
            success=False, error="companion_runtime_disabled",
            metadata={"ok": False, "reason": "companion_runtime_disabled"},
        ), None
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None:
        return False, ToolResult(
            success=False, error="companion_runtime not initialized",
            metadata={"ok": False, "reason": "runtime_not_ready"},
        ), None
    return True, None, runtime


class RecommendNowTool(Tool):
    """Surface a substrate-driven recommendation."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "recommend_now"

    @property
    def description(self) -> str:
        return (
            "Ask Becca's substrate what she'd recommend right now — "
            "based on dominant drive, recent activity, energy. "
            "Returns a structured suggestion (kind + reason) the "
            "model should pass through in natural language. Use "
            "when the user asks 'what should I do?' / 'recommend "
            "something' / 'any ideas?' — NOT for arbitrary "
            "questions or for tasks with explicit user intent."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=True, coder=False, companion=True, flow=False)

    @property
    def cacheable(self) -> bool:
        return False  # Substrate is mutable; recommendation could shift.

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.READ,
            autonomy_class=CoreVerbAutonomyClass.SUGGESTED,
            cost_envelope=CostEnvelope(max_wallclock_ms=2_000, max_db_ops=8),
            cite_self_required=False,
        )

    async def execute(self, **kwargs) -> ToolResult:
        ok, err, runtime = _gate(self._app_state)
        if not ok:
            return err
        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False, error="user_id missing",
                metadata={"ok": False, "reason": "missing_user"},
            )

        # Read drive state — tick_drive keeps it fresh.
        from augmentum.companion_runtime import drives
        state = await drives.load(runtime, user_id=user_id)

        # Pick the most urgent drive. Use the same dominant() logic as
        # DriveState so we don't disagree with the model's other
        # substrate readers.
        dominant = state.dominant()
        urgency = state.urgency(dominant)
        kind, reason = _DRIVE_RECOMMENDATIONS.get(
            dominant, ("no_op", "Nothing urgent on the substrate."),
        )

        # Also read energy to qualify the recommendation — low energy
        # makes the rest-leaning answer more honest even if another
        # drive is technically dominant.
        try:
            from augmentum.companion_runtime import energy
            energy_state = await energy.load(runtime, user_id=user_id)
            energy_level = round(energy_state.level, 2)
        except Exception:
            energy_level = None

        if energy_level is not None and energy_level < 0.3:
            kind = "no_op"
            reason = "Energy is low. Rest is the honest answer."

        summary = (
            f"Most urgent: {dominant} ({urgency:.2f}). "
            f"Suggestion: {kind} — {reason}"
        )
        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "ok": True,
                "kind": kind,
                "drive": dominant,
                "urgency": round(urgency, 3),
                "energy_level": energy_level,
                "reason": reason,
            },
        )
