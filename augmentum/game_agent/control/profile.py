"""Profile data structures + JSON-Schema validation.

Two on-disk JSON shapes:

* **Controller profile** (``control.controller.v1``) declares the wire
  mapping for a physical device. One file per device class (e.g.,
  ``gba.json``, ``gambatte.json``, ``psx.json``, ``pc_keyboard.json``).
* **Game profile** (``control.game.v1``) declares which universal /
  game-specific actions are available for a particular game, and how
  each maps to an abstract controller input. One file per game.

Two in-memory types:

* :class:`ControllerProfile` — validated, immutable controller mapping.
* :class:`GameProfile` — validated, immutable game mapping.

Plus :class:`ComposedProfile` which holds both and resolves
Layer 1 → Layer 3 in a single ``resolve(semantic)`` call.

Validation is strict on load (Pydantic v2). Bad profiles fail loud at
startup rather than silently at session time -- the explicit error
messages call out which field is wrong and where.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from augmentum.game_agent.control.actions import (
    ACTION_DESCRIPTIONS,
    is_universal_action,
)
from augmentum.game_agent.control.controllers import (
    is_keyboard_code,
    is_known_controller_input,
)

# ── Wire transport descriptors ────────────────────────────────────────
#
# A controller profile declares how its button table maps to actual
# bytes on the wire. The bridge / surface adapter consumes the wire
# kind and routes accordingly.

_WIRE_KINDS = frozenset({
    "libretro_joypad",   # libretro simulateInput(port, button_id, value)
    "keyboard",          # KeyboardEvent.code via synthetic events
    "pointer",           # mouse / touch coordinates
    "lua_rpc",           # JSON-RPC payload to a server-side Lua mod
    "synthetic",         # adapter-defined (custom JSON to bridge)
})


# ── Pydantic schemas for JSON loading ─────────────────────────────────


class _ControllerWireSchema(BaseModel):
    """Wire descriptor on a controller profile.

    @example: {"kind": "libretro_joypad", "port": 0}
    @example: {"kind": "keyboard"}
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    port: int = 0

    def validate_kind(self) -> None:
        if self.kind not in _WIRE_KINDS:
            raise ValueError(
                f"unknown wire.kind {self.kind!r}; "
                f"valid: {sorted(_WIRE_KINDS)}"
            )


class _ControllerButtonSchema(BaseModel):
    """One row in a controller profile's button table."""

    model_config = ConfigDict(extra="forbid")

    wire_code: int | str
    label: str = ""


class _ControllerProfileSchema(BaseModel):
    """On-disk shape of a controller-profile JSON file."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,  # accept JSON key 'schema' via alias
    )

    # Pydantic v2 reserves ``schema`` as a class method, so we alias.
    profile_schema: Literal["control.controller.v1"] = Field(alias="schema")
    id: str
    description: str = ""
    wire: _ControllerWireSchema
    buttons: dict[str, _ControllerButtonSchema]


class _GameActionBindingSchema(BaseModel):
    """One action row in a game profile.

    @example: {"binding": "face_east", "hint": "Advance dialog"}
    """

    model_config = ConfigDict(extra="forbid")

    binding: str
    hint: str = ""
    duration_ms_min: int = 120
    duration_ms_max: int = 2000


class _GameProfileSchema(BaseModel):
    """On-disk shape of a game-profile JSON file."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    profile_schema: Literal["control.game.v1"] = Field(alias="schema")
    id: str
    description: str = ""
    applies_to_controllers: list[str]
    applies_to_log_schema: str | None = None
    actions: dict[str, _GameActionBindingSchema]
    hardware_passthrough: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Immutable runtime types ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControllerButton:
    """Resolved Layer-2 → Layer-3 row."""

    name: str          # Layer-2 input name, e.g., "face_east"
    wire_code: int | str
    label: str         # human-friendly: "A", "Cross", "Space"


@dataclass(frozen=True)
class ControllerProfile:
    """Validated, immutable controller profile.

    Use when:
    - The surface adapter is being constructed and needs to know the
      wire mapping for its physical device.
    """

    id: str
    description: str
    wire_kind: str            # one of _WIRE_KINDS
    wire_port: int
    buttons: dict[str, ControllerButton]  # Layer-2 name -> button

    def has(self, layer2_input: str) -> bool:
        return layer2_input in self.buttons

    def wire_for(self, layer2_input: str) -> ControllerButton | None:
        return self.buttons.get(layer2_input)


@dataclass(frozen=True, slots=True)
class GameAction:
    """One agent-facing action in a game profile."""

    name: str          # Layer-1 semantic name
    binding: str       # Layer-2 controller input it maps to
    hint: str
    duration_ms_min: int
    duration_ms_max: int
    is_universal: bool


@dataclass(frozen=True)
class GameProfile:
    """Validated, immutable game profile."""

    id: str
    description: str
    applies_to_controllers: tuple[str, ...]
    applies_to_log_schema: str | None
    actions: dict[str, GameAction]              # Layer-1 -> binding info
    hardware_passthrough: tuple[str, ...]       # Layer-2 buttons exposed raw
    notes: str


@dataclass(frozen=True)
class ComposedProfile:
    """A controller + game pairing.

    The thing the surface adapter actually consumes. Provides:
    * :meth:`semantic_inputs` — the agent's allowed vocabulary
    * :meth:`hints` — semantic_id → human-readable hint, for the prompt
    * :meth:`resolve` — Layer 1 → wire emit info
    """

    controller: ControllerProfile
    game: GameProfile

    def semantic_inputs(self) -> list[str]:
        """All semantic ids the agent may emit, sorted for stable prompts."""

        ids: set[str] = set(self.game.actions.keys())
        # Hardware passthrough surfaces controller-level buttons directly.
        for hw in self.game.hardware_passthrough:
            if self.controller.has(hw):
                ids.add(hw)
        return sorted(ids)

    def hints(self) -> dict[str, str]:
        """semantic_id -> human-readable hint for the prompt's INPUT_HINTS block."""

        out: dict[str, str] = {}
        for action_name, action in self.game.actions.items():
            out[action_name] = action.hint or ACTION_DESCRIPTIONS.get(
                action_name, "(no hint)",
            )
        for hw in self.game.hardware_passthrough:
            btn = self.controller.wire_for(hw)
            if btn is None:
                continue
            label = btn.label or hw
            out[hw] = f"Hardware passthrough: {label}."
        return out

    def resolve(self, semantic: str) -> ControllerButton | None:
        """Resolve a Layer-1 semantic to the wire-level controller button.

        Returns:
        - :class:`ControllerButton` when the semantic maps cleanly
          through the game and controller profiles.
        - ``None`` when the semantic is unknown to this composition.
          Callers (resolver) treat None as ``UnknownSemanticError``.
        """

        action = self.game.actions.get(semantic)
        if action is not None:
            return self.controller.wire_for(action.binding)
        if semantic in self.game.hardware_passthrough:
            return self.controller.wire_for(semantic)
        return None


# ── Loading ───────────────────────────────────────────────────────────


class ProfileLoadError(ValueError):
    """Raised when a profile JSON fails schema validation or sanity rules."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise ProfileLoadError(f"could not open profile {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileLoadError(
            f"profile {path} is not valid JSON: {exc.msg} at line {exc.lineno}",
        ) from exc


def load_controller_profile(path: Path) -> ControllerProfile:
    """Validate and load a controller profile file.

    Raises :class:`ProfileLoadError` on schema mismatch, unknown wire
    kind, or button mapping that references an unknown Layer-2 name.
    """

    raw = _load_json(path)
    try:
        validated = _ControllerProfileSchema.model_validate(raw)
    except ValidationError as exc:
        raise ProfileLoadError(
            f"controller profile {path} failed schema: {exc}",
        ) from exc
    try:
        validated.wire.validate_kind()
    except ValueError as exc:
        raise ProfileLoadError(
            f"controller profile {path}: {exc}",
        ) from exc
    # Each button name must be either a known controller input
    # (preferred) or a keyboard code (when wire.kind == "keyboard").
    is_kbd = validated.wire.kind == "keyboard"
    for btn_name in validated.buttons:
        if is_known_controller_input(btn_name):
            continue
        if is_kbd and is_keyboard_code(btn_name):
            continue
        raise ProfileLoadError(
            f"controller profile {path}: button name {btn_name!r} is not a "
            "known Layer-2 controller input and not a valid keyboard code",
        )
    buttons: dict[str, ControllerButton] = {
        name: ControllerButton(
            name=name, wire_code=b.wire_code, label=b.label or name,
        )
        for name, b in validated.buttons.items()
    }
    return ControllerProfile(
        id=validated.id,
        description=validated.description,
        wire_kind=validated.wire.kind,
        wire_port=validated.wire.port,
        buttons=buttons,
    )


def load_game_profile(path: Path) -> GameProfile:
    """Validate and load a game profile file.

    Raises :class:`ProfileLoadError` on schema mismatch.

    Note: cross-profile validation (each binding references a real
    Layer-2 input on the target controller) happens at compose time,
    NOT here -- the game profile is intentionally loadable without
    requiring every supported controller to be present.
    """

    raw = _load_json(path)
    try:
        validated = _GameProfileSchema.model_validate(raw)
    except ValidationError as exc:
        raise ProfileLoadError(
            f"game profile {path} failed schema: {exc}",
        ) from exc
    actions: dict[str, GameAction] = {}
    for name, binding in validated.actions.items():
        if not name or not all(c.isalnum() or c == "_" for c in name):
            raise ProfileLoadError(
                f"game profile {path}: action name {name!r} must be [a-z0-9_]+",
            )
        if binding.duration_ms_min < 1 or binding.duration_ms_max > 30000:
            raise ProfileLoadError(
                f"game profile {path}: action {name!r} has unreasonable "
                "duration range; valid 1..30000 ms",
            )
        actions[name] = GameAction(
            name=name,
            binding=binding.binding,
            hint=binding.hint,
            duration_ms_min=binding.duration_ms_min,
            duration_ms_max=binding.duration_ms_max,
            is_universal=is_universal_action(name),
        )
    return GameProfile(
        id=validated.id,
        description=validated.description,
        applies_to_controllers=tuple(validated.applies_to_controllers),
        applies_to_log_schema=validated.applies_to_log_schema,
        actions=actions,
        hardware_passthrough=tuple(validated.hardware_passthrough),
        notes=validated.notes,
    )


def compose(
    controller: ControllerProfile,
    game: GameProfile,
) -> ComposedProfile:
    """Combine a controller + game profile, validating they fit.

    Raises :class:`ProfileLoadError` when:
    - The game profile doesn't declare support for this controller id.
    - A game action binds to a Layer-2 input the controller doesn't expose.
    - A hardware passthrough name isn't on the controller.
    """

    if controller.id not in game.applies_to_controllers:
        raise ProfileLoadError(
            f"game profile {game.id!r} does not declare support for "
            f"controller {controller.id!r}; declared: "
            f"{list(game.applies_to_controllers)}",
        )
    for action_name, action in game.actions.items():
        if not controller.has(action.binding):
            raise ProfileLoadError(
                f"game profile {game.id!r} action {action_name!r} binds to "
                f"{action.binding!r} which controller {controller.id!r} "
                "does not expose",
            )
    for hw in game.hardware_passthrough:
        if not controller.has(hw):
            raise ProfileLoadError(
                f"game profile {game.id!r} hardware_passthrough lists "
                f"{hw!r} which controller {controller.id!r} does not expose",
            )
    return ComposedProfile(controller=controller, game=game)


__all__ = [
    "ComposedProfile",
    "ControllerButton",
    "ControllerProfile",
    "GameAction",
    "GameProfile",
    "ProfileLoadError",
    "compose",
    "load_controller_profile",
    "load_game_profile",
]
