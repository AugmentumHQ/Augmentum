"""Canonical per-system controller layouts.

A SystemProfile is a small data structure: which logical actions the
system supports (D-pad, A, B, Start, Select, etc.) plus the canonical
default mapping each action gets on (1) keyboard and (2) the standard
HTML5 gamepad layout.

Logical action ids are namespaced by system (``nes_a``, ``snes_x``,
...) so a remap UI can route "the user just pressed gamepad button 1
while binding nes_a" without ambiguity. Keyboard code values use the
standard ``KeyboardEvent.code`` strings (``KeyZ``, ``ArrowUp``,
``Enter``, ...). Gamepad button indices follow the Standard Gamepad
spec: 0=A/Cross, 1=B/Circle, 2=X/Square, 3=Y/Triangle, 4-5=L/R,
6-7=L2/R2, 8=Select, 9=Start, 12-15=D-pad up/down/left/right.

Adding a new system: one entry below + (separately) a libretro core
mapping in ``augmentum/titles/rom_systems.py`` if the system is
emulator-backed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SystemBinding:
    """One logical action's default mapping across input sources."""
    keyboard: str | None = None              # KeyboardEvent.code (or None)
    gamepad_button: int | None = None        # standard Gamepad button index
    gamepad_axis: int | None = None          # axis index (for stick directions)
    gamepad_axis_sign: int | None = None     # +1 / -1 for axis-as-button


@dataclass(frozen=True)
class SystemProfile:
    """All actions + defaults for one emulator system."""
    id: str
    label: str
    actions: dict[str, SystemBinding]
    multiplayer: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "multiplayer": self.multiplayer,
            "actions": {
                aid: {
                    "keyboard": b.keyboard,
                    "gamepad_button": b.gamepad_button,
                    "gamepad_axis": b.gamepad_axis,
                    "gamepad_axis_sign": b.gamepad_axis_sign,
                }
                for aid, b in self.actions.items()
            },
        }


# ── Helpers (factor out the verbose dpad block) ──────────────────────


def _dpad(prefix: str) -> dict[str, SystemBinding]:
    """D-pad bindings shared across virtually every retro system."""
    return {
        f"{prefix}_up":    SystemBinding(keyboard="ArrowUp",    gamepad_button=12,
                                          gamepad_axis=1, gamepad_axis_sign=-1),
        f"{prefix}_down":  SystemBinding(keyboard="ArrowDown",  gamepad_button=13,
                                          gamepad_axis=1, gamepad_axis_sign=+1),
        f"{prefix}_left":  SystemBinding(keyboard="ArrowLeft",  gamepad_button=14,
                                          gamepad_axis=0, gamepad_axis_sign=-1),
        f"{prefix}_right": SystemBinding(keyboard="ArrowRight", gamepad_button=15,
                                          gamepad_axis=0, gamepad_axis_sign=+1),
    }


# ── System catalog ───────────────────────────────────────────────────
# Ordered roughly by generation. Adding a system: define actions,
# pick reasonable keyboard defaults, register at the bottom of
# SYSTEM_PROFILES.

_NES = SystemProfile(
    id="nes", label="Nintendo Entertainment System",
    actions={
        **_dpad("nes"),
        "nes_a":      SystemBinding(keyboard="KeyZ",  gamepad_button=1),
        "nes_b":      SystemBinding(keyboard="KeyX",  gamepad_button=0),
        "nes_start":  SystemBinding(keyboard="Enter", gamepad_button=9),
        "nes_select": SystemBinding(keyboard="Space", gamepad_button=8),
    },
    multiplayer=2,
)

_SNES = SystemProfile(
    id="snes", label="Super Nintendo",
    actions={
        **_dpad("snes"),
        "snes_a":      SystemBinding(keyboard="KeyX",      gamepad_button=1),
        "snes_b":      SystemBinding(keyboard="KeyZ",      gamepad_button=0),
        "snes_x":      SystemBinding(keyboard="KeyS",      gamepad_button=3),
        "snes_y":      SystemBinding(keyboard="KeyA",      gamepad_button=2),
        "snes_l":      SystemBinding(keyboard="KeyQ",      gamepad_button=4),
        "snes_r":      SystemBinding(keyboard="KeyW",      gamepad_button=5),
        "snes_start":  SystemBinding(keyboard="Enter",     gamepad_button=9),
        "snes_select": SystemBinding(keyboard="Backspace", gamepad_button=8),
    },
    multiplayer=4,
)

_GB = SystemProfile(
    id="gb", label="Game Boy",
    actions={
        **_dpad("gb"),
        "gb_a":      SystemBinding(keyboard="KeyZ",  gamepad_button=1),
        "gb_b":      SystemBinding(keyboard="KeyX",  gamepad_button=0),
        "gb_start":  SystemBinding(keyboard="Enter", gamepad_button=9),
        "gb_select": SystemBinding(keyboard="Space", gamepad_button=8),
    },
    multiplayer=1,
)

_GBC = SystemProfile(
    id="gbc", label="Game Boy Color",
    actions={
        **_dpad("gbc"),
        "gbc_a":      SystemBinding(keyboard="KeyZ",  gamepad_button=1),
        "gbc_b":      SystemBinding(keyboard="KeyX",  gamepad_button=0),
        "gbc_start":  SystemBinding(keyboard="Enter", gamepad_button=9),
        "gbc_select": SystemBinding(keyboard="Space", gamepad_button=8),
    },
    multiplayer=1,
)

_GBA = SystemProfile(
    id="gba", label="Game Boy Advance",
    actions={
        **_dpad("gba"),
        "gba_a":      SystemBinding(keyboard="KeyZ",  gamepad_button=1),
        "gba_b":      SystemBinding(keyboard="KeyX",  gamepad_button=0),
        "gba_l":      SystemBinding(keyboard="KeyA",  gamepad_button=4),
        "gba_r":      SystemBinding(keyboard="KeyS",  gamepad_button=5),
        "gba_start":  SystemBinding(keyboard="Enter", gamepad_button=9),
        "gba_select": SystemBinding(keyboard="Space", gamepad_button=8),
    },
    multiplayer=1,
)

_N64 = SystemProfile(
    id="n64", label="Nintendo 64",
    actions={
        **_dpad("n64"),
        "n64_a":       SystemBinding(keyboard="KeyZ",      gamepad_button=1),
        "n64_b":       SystemBinding(keyboard="KeyX",      gamepad_button=0),
        # C-pad mapped to right-stick directions
        "n64_c_up":    SystemBinding(keyboard="KeyI",      gamepad_axis=3, gamepad_axis_sign=-1),
        "n64_c_down":  SystemBinding(keyboard="KeyK",      gamepad_axis=3, gamepad_axis_sign=+1),
        "n64_c_left":  SystemBinding(keyboard="KeyJ",      gamepad_axis=2, gamepad_axis_sign=-1),
        "n64_c_right": SystemBinding(keyboard="KeyL",      gamepad_axis=2, gamepad_axis_sign=+1),
        "n64_l":       SystemBinding(keyboard="KeyQ",      gamepad_button=4),
        "n64_r":       SystemBinding(keyboard="KeyW",      gamepad_button=5),
        "n64_z":       SystemBinding(keyboard="Space",     gamepad_button=6),
        "n64_start":   SystemBinding(keyboard="Enter",     gamepad_button=9),
        # Analog stick mapped to left-stick axes (axes 0/1)
        "n64_stick_up":    SystemBinding(keyboard="KeyW", gamepad_axis=1, gamepad_axis_sign=-1),
        "n64_stick_down":  SystemBinding(keyboard="KeyS", gamepad_axis=1, gamepad_axis_sign=+1),
        "n64_stick_left":  SystemBinding(keyboard="KeyA", gamepad_axis=0, gamepad_axis_sign=-1),
        "n64_stick_right": SystemBinding(keyboard="KeyD", gamepad_axis=0, gamepad_axis_sign=+1),
    },
    multiplayer=4,
)

_PSX = SystemProfile(
    id="psx", label="PlayStation 1",
    actions={
        **_dpad("psx"),
        "psx_cross":    SystemBinding(keyboard="KeyZ",      gamepad_button=0),
        "psx_circle":   SystemBinding(keyboard="KeyX",      gamepad_button=1),
        "psx_square":   SystemBinding(keyboard="KeyA",      gamepad_button=2),
        "psx_triangle": SystemBinding(keyboard="KeyS",      gamepad_button=3),
        "psx_l1":       SystemBinding(keyboard="KeyQ",      gamepad_button=4),
        "psx_r1":       SystemBinding(keyboard="KeyW",      gamepad_button=5),
        "psx_l2":       SystemBinding(keyboard="KeyE",      gamepad_button=6),
        "psx_r2":       SystemBinding(keyboard="KeyR",      gamepad_button=7),
        "psx_select":   SystemBinding(keyboard="Backspace", gamepad_button=8),
        "psx_start":    SystemBinding(keyboard="Enter",     gamepad_button=9),
        "psx_l3":       SystemBinding(keyboard="KeyT",      gamepad_button=10),
        "psx_r3":       SystemBinding(keyboard="KeyY",      gamepad_button=11),
        "psx_lstick_up":    SystemBinding(gamepad_axis=1, gamepad_axis_sign=-1),
        "psx_lstick_down":  SystemBinding(gamepad_axis=1, gamepad_axis_sign=+1),
        "psx_lstick_left":  SystemBinding(gamepad_axis=0, gamepad_axis_sign=-1),
        "psx_lstick_right": SystemBinding(gamepad_axis=0, gamepad_axis_sign=+1),
        "psx_rstick_up":    SystemBinding(gamepad_axis=3, gamepad_axis_sign=-1),
        "psx_rstick_down":  SystemBinding(gamepad_axis=3, gamepad_axis_sign=+1),
        "psx_rstick_left":  SystemBinding(gamepad_axis=2, gamepad_axis_sign=-1),
        "psx_rstick_right": SystemBinding(gamepad_axis=2, gamepad_axis_sign=+1),
    },
    multiplayer=2,
)

_GENESIS = SystemProfile(
    id="genesis", label="Sega Genesis / Mega Drive",
    actions={
        **_dpad("genesis"),
        "genesis_a":     SystemBinding(keyboard="KeyA",  gamepad_button=2),
        "genesis_b":     SystemBinding(keyboard="KeyS",  gamepad_button=0),
        "genesis_c":     SystemBinding(keyboard="KeyD",  gamepad_button=1),
        "genesis_x":     SystemBinding(keyboard="KeyQ",  gamepad_button=3),
        "genesis_y":     SystemBinding(keyboard="KeyW",  gamepad_button=4),
        "genesis_z":     SystemBinding(keyboard="KeyE",  gamepad_button=5),
        "genesis_start": SystemBinding(keyboard="Enter", gamepad_button=9),
        "genesis_mode":  SystemBinding(keyboard="Space", gamepad_button=8),
    },
    multiplayer=2,
)

_SMS = SystemProfile(
    id="sms", label="Sega Master System",
    actions={
        **_dpad("sms"),
        "sms_1":     SystemBinding(keyboard="KeyZ",  gamepad_button=0),
        "sms_2":     SystemBinding(keyboard="KeyX",  gamepad_button=1),
        "sms_pause": SystemBinding(keyboard="Enter", gamepad_button=9),
    },
    multiplayer=2,
)

_ATARI2600 = SystemProfile(
    id="atari2600", label="Atari 2600",
    actions={
        **_dpad("atari2600"),
        "atari2600_fire":      SystemBinding(keyboard="Space", gamepad_button=0),
        "atari2600_select":    SystemBinding(keyboard="KeyQ",  gamepad_button=8),
        "atari2600_reset":     SystemBinding(keyboard="KeyW",  gamepad_button=9),
    },
    multiplayer=2,
)

_PCE = SystemProfile(
    id="pce", label="TurboGrafx-16 / PC Engine",
    actions={
        **_dpad("pce"),
        "pce_i":      SystemBinding(keyboard="KeyZ",  gamepad_button=1),
        "pce_ii":     SystemBinding(keyboard="KeyX",  gamepad_button=0),
        "pce_select": SystemBinding(keyboard="Space", gamepad_button=8),
        "pce_run":    SystemBinding(keyboard="Enter", gamepad_button=9),
    },
    multiplayer=2,
)

_ARCADE = SystemProfile(
    id="arcade", label="Arcade",
    actions={
        **_dpad("arcade"),
        "arcade_b1":     SystemBinding(keyboard="KeyZ",  gamepad_button=0),
        "arcade_b2":     SystemBinding(keyboard="KeyX",  gamepad_button=1),
        "arcade_b3":     SystemBinding(keyboard="KeyC",  gamepad_button=2),
        "arcade_b4":     SystemBinding(keyboard="KeyV",  gamepad_button=3),
        "arcade_b5":     SystemBinding(keyboard="KeyA",  gamepad_button=4),
        "arcade_b6":     SystemBinding(keyboard="KeyS",  gamepad_button=5),
        "arcade_coin":   SystemBinding(keyboard="Digit5", gamepad_button=8),
        "arcade_start":  SystemBinding(keyboard="Digit1", gamepad_button=9),
    },
    multiplayer=4,
)


SYSTEM_PROFILES: tuple[SystemProfile, ...] = (
    _NES, _SNES, _GB, _GBC, _GBA, _N64, _PSX,
    _GENESIS, _SMS, _ATARI2600, _PCE, _ARCADE,
)


# ── Lookup ───────────────────────────────────────────────────────────


def list_systems() -> list[SystemProfile]:
    return list(SYSTEM_PROFILES)


def get_system_profile(system_id: str) -> SystemProfile | None:
    for p in SYSTEM_PROFILES:
        if p.id == system_id:
            return p
    return None
