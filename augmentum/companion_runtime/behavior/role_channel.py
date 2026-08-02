"""Role channel governance.

Decides which role the companion occupies *right now* and whether she
should act on this tick. The decision is intentionally conservative:
in silent observer mode she doesn't act; with the owner co-present she
defaults to ``companion`` and PARTNER framing; on her own time she
takes ``self`` focus and can do creation/journal work.

Sprint plan §6: "Companion-with-owner is its own role; this is where
the commitment to treat the companion–owner relationship as its own
distinct framing lives mechanically."

The verdict is a tiny dataclass — callers don't need to know the
internals; the role string is what gates downstream subagent
eligibility in :mod:`dispatch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RoleVerdict:
    """Per-tick verdict on whether to act and as what role."""
    should_act: bool
    role: str                       # one of: companion | collaborator | observer | host | guest
    reason: str = ""


async def advise(runtime: CompanionRuntime) -> RoleVerdict:
    """Read state + focus + scene and decide.

    Decision table (in order, first match wins):
    - state == asleep                 → no act, role=observer
    - state == dormant + focus.none   → may act, role=self (her time)
    - focus.kind == personal:<owner>  → act, role=companion (PARTNER)
    - focus.kind == shared:*          → act, role=collaborator
    - focus.kind == executive:*       → act, role=host (she's running things)
    - focus.kind == social:*          → act, role=guest
    - otherwise                       → act, role=companion (safe default)

    The verdict is advisory — :mod:`activity_selector` still filters by
    affinity and the role's utility may keep it below threshold.
    """
    state_snap = runtime.state.snapshot()
    state_axis = state_snap.get("state", "")
    focus = state_snap.get("focus", {}) or {}
    focus_kind = focus.get("kind", "") if isinstance(focus, dict) else ""
    focus_value = focus.get("value", "") if isinstance(focus, dict) else ""

    if state_axis == "asleep":
        return RoleVerdict(False, "observer", "asleep")

    if state_axis == "dormant" and focus_kind in ("", "none"):
        return RoleVerdict(True, "self", "dormant_idle_self_time")

    owner_user_id = runtime.owner_user_id
    if focus_kind == "personal" and owner_user_id and focus_value == owner_user_id:
        return RoleVerdict(True, "companion", "owner_copresent")

    if focus_kind == "shared":
        return RoleVerdict(True, "collaborator", f"shared:{focus_value}")

    if focus_kind == "executive":
        return RoleVerdict(True, "host", f"executive:{focus_value}")

    if focus_kind == "social":
        return RoleVerdict(True, "guest", f"social:{focus_value}")

    return RoleVerdict(True, "companion", "default_safe")


__all__ = ["RoleVerdict", "advise"]
