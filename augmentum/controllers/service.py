"""ControllerService -- merge defaults + user remap into a resolved layout.

Engine adapters consume the resolved layout; they don't know or care
that some bindings come from defaults.py and others from the user's
override row. The merge rule is per-action: any non-null entry in the
user's override wins; otherwise the default is used.

Reset = delete the user's row → the next ``resolve()`` falls through
to defaults entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.controllers.defaults import (
    SYSTEM_PROFILES,
    SystemProfile,
    get_system_profile,
)
from augmentum.controllers.store import ControllerRemap, ControllerStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedLayout:
    """The merged (default + override) controller layout for one
    (user, system) pair. This is what engine adapters get.
    """
    system_id: str
    label: str
    pad_routing: str                                # 'index' | 'firstpress'
    multiplayer: int
    actions: dict[str, dict[str, Any]]              # logical_id -> resolved binding

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "label": self.label,
            "pad_routing": self.pad_routing,
            "multiplayer": self.multiplayer,
            "actions": dict(self.actions),
        }


class ControllerService:
    def __init__(self, *, store: ControllerStore) -> None:
        self._store = store

    # ── Reads ────────────────────────────────────────────────────────

    def list_systems(self) -> list[SystemProfile]:
        return list(SYSTEM_PROFILES)

    async def resolve(
        self, *, user_id: str, system_id: str,
    ) -> ResolvedLayout | None:
        profile = get_system_profile(system_id)
        if profile is None:
            return None
        remap = await self._store.get(user_id=user_id, system_id=system_id)
        return _resolve(profile, remap)

    async def list_user_remaps(
        self, *, user_id: str,
    ) -> list[ControllerRemap]:
        return await self._store.list_for_user(user_id=user_id)

    async def get_user_remap(
        self, *, user_id: str, system_id: str,
    ) -> ControllerRemap | None:
        return await self._store.get(user_id=user_id, system_id=system_id)

    # ── Writes ───────────────────────────────────────────────────────

    async def update_remap(
        self,
        *,
        user_id: str,
        system_id: str,
        bindings: dict[str, Any] | None = None,
        pad_routing: str | None = None,
    ) -> ControllerRemap:
        if get_system_profile(system_id) is None:
            raise ValueError(f"unknown system_id {system_id!r}")
        if bindings is not None:
            # Reject binding overrides for unknown actions -- catches
            # typos before they silently no-op at runtime.
            profile = get_system_profile(system_id)
            unknown = set(bindings) - set(profile.actions)
            if unknown:
                raise ValueError(
                    f"unknown actions for {system_id!r}: {sorted(unknown)}"
                )
        return await self._store.upsert(
            user_id=user_id,
            system_id=system_id,
            bindings=bindings,
            pad_routing=pad_routing,
        )

    async def reset_remap(
        self, *, user_id: str, system_id: str,
    ) -> bool:
        return await self._store.delete(user_id=user_id, system_id=system_id)


# ── Internals ───────────────────────────────────────────────────────


def _resolve(
    profile: SystemProfile,
    remap: ControllerRemap | None,
) -> ResolvedLayout:
    pad_routing = remap.pad_routing if remap else "index"
    overrides = remap.bindings if remap else {}
    merged: dict[str, dict[str, Any]] = {}

    for action_id, default in profile.actions.items():
        override = overrides.get(action_id) if isinstance(overrides, dict) else None
        if isinstance(override, dict):
            merged[action_id] = {
                "keyboard": _coalesce(override.get("keyboard"), default.keyboard),
                "gamepad_button": _coalesce(
                    override.get("gamepad_button"), default.gamepad_button,
                ),
                "gamepad_axis": _coalesce(
                    override.get("gamepad_axis"), default.gamepad_axis,
                ),
                "gamepad_axis_sign": _coalesce(
                    override.get("gamepad_axis_sign"), default.gamepad_axis_sign,
                ),
            }
        else:
            merged[action_id] = {
                "keyboard": default.keyboard,
                "gamepad_button": default.gamepad_button,
                "gamepad_axis": default.gamepad_axis,
                "gamepad_axis_sign": default.gamepad_axis_sign,
            }
    return ResolvedLayout(
        system_id=profile.id,
        label=profile.label,
        pad_routing=pad_routing,
        multiplayer=profile.multiplayer,
        actions=merged,
    )


def _coalesce(override, default):
    """Return override when it's an explicit value, default otherwise.

    None means "no override -- use default". An empty string or 0 is a
    valid override value (e.g. binding to KeyboardEvent.code "" means
    "no key, but I want the default cleared"). We treat None
    specifically as the "fall through" sentinel.
    """
    return default if override is None else override
