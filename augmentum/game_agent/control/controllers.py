"""Layer 2: abstract controller / input device vocabulary.

What the device CAN DO, named POSITIONALLY rather than by letter.
This is the SDL2 ``SDL_GameController`` design adapted to our
agent-control needs: every controller speaks the same abstract
button names, and per-controller profiles map them to wire-level
codes.

Why positional names instead of A/B/X/Y
----------------------------------------
A naive design uses ``face_a``, ``face_b``, ``face_x``, ``face_y``.
This breaks when porting across console families:

* Nintendo: A/B/X/Y in a specific cross pattern
* Sony: Triangle/Square/X/Circle
* Xbox: Y/X/A/B
* Generic third-party gamepads: variable

The "A button" is NOT the same physical position across consoles. SDL
solved this by naming the four face-button slots POSITIONALLY
(``SDL_CONTROLLER_BUTTON_A`` was the south button, ``_B`` east, etc.,
later renamed ``_SOUTH``/``_EAST``/``_WEST``/``_NORTH`` in SDL3 to
match modern Steam Input conventions).

We use the modern positional naming directly: ``face_south``,
``face_east``, ``face_west``, ``face_north``. The game profile says
"confirm binds to face_east" (which is A on GBA, Circle on PSX,
because both consoles have their "confirm" button on the east face).

D-pad
-----
``dpad_up`` / ``dpad_down`` / ``dpad_left`` / ``dpad_right``. Cardinal
directions on the directional pad. Distinct from analog stick
directions even on controllers where the agent only has one or the
other.

Shoulders + triggers
--------------------
``shoulder_l``, ``shoulder_r`` — top-row digital buttons.
``trigger_l``, ``trigger_r`` — bottom-row buttons. Analog on modern
controllers but treated as digital here (the libretro layer presents
them as buttons; analog gradient comes from a different code path).

System
------
``start``, ``select`` — universally present from NES onward.
``home``, ``share`` — modern additions (PS4/PS5/Switch/Xbox).

Sticks
------
Stick CLICKS: ``stick_l_press``, ``stick_r_press`` (L3/R3 / LS/RS).
Stick DIRECTIONS as digital presses: ``stick_l_up`` and family. These
are useful when the game maps stick-directional gestures to discrete
inputs (Smash Bros style C-stick attacks). For continuous analog
control we'd need a different model — not in v1 scope.

Pointer / keyboard
------------------
Non-gamepad surfaces (Luanti, web games, captured desktop) declare
their own controller profile with ``wire.kind`` of ``keyboard``,
``pointer``, or ``lua_rpc``. Their button table maps Layer-2 names
to KeyboardEvent.code / mouse coordinate / RPC payload. The agent's
Layer-1 vocabulary stays the same; only the wire mapping changes.
"""

from __future__ import annotations

# ── D-pad: cardinal directions on the directional cross ──────────────
DPAD_INPUTS: frozenset[str] = frozenset({
    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
})


# ── Face buttons: positional (SDL3 convention) ────────────────────────
#
# South / East / West / North describe the slot on a four-button
# diamond, not the letter. Mapping to letters depends entirely on
# which controller profile is loaded.
FACE_BUTTONS: frozenset[str] = frozenset({
    "face_south", "face_east", "face_west", "face_north",
})


# ── Shoulders + triggers ──────────────────────────────────────────────
SHOULDER_INPUTS: frozenset[str] = frozenset({
    "shoulder_l", "shoulder_r",
    "trigger_l", "trigger_r",
})


# ── Stick clicks (L3 / R3) ────────────────────────────────────────────
STICK_CLICK_INPUTS: frozenset[str] = frozenset({
    "stick_l_press", "stick_r_press",
})


# ── Stick directions as digital presses ───────────────────────────────
#
# Smash-style or any game where the stick tap is treated as a discrete
# direction press. Continuous analog is not represented here -- that
# requires an analog channel in the wire payload.
STICK_DIRECTIONAL_INPUTS: frozenset[str] = frozenset({
    "stick_l_up", "stick_l_down", "stick_l_left", "stick_l_right",
    "stick_r_up", "stick_r_down", "stick_r_left", "stick_r_right",
})


# ── System buttons ────────────────────────────────────────────────────
SYSTEM_BUTTONS: frozenset[str] = frozenset({
    "start", "select", "home", "share",
})


# ── The full abstract controller vocabulary ───────────────────────────
#
# A controller profile is REQUIRED to declare wire mappings only for
# the subset it physically has. GBA has dpad + face_south/east +
# start/select + shoulder_l/r (no trigger, no sticks, no home/share).
# A modern gamepad declares the full set. Game profiles bind to
# whatever the controller exposes.
CONTROLLER_INPUTS: frozenset[str] = (
    DPAD_INPUTS
    | FACE_BUTTONS
    | SHOULDER_INPUTS
    | STICK_CLICK_INPUTS
    | STICK_DIRECTIONAL_INPUTS
    | SYSTEM_BUTTONS
)


# ── Pointer inputs (mouse-driven surfaces) ────────────────────────────
#
# Distinct from CONTROLLER_INPUTS because they take coordinates, not
# just a duration. Game profile bindings can reference these by name
# but the wire payload includes (x, y) at dispatch time.
POINTER_INPUTS: frozenset[str] = frozenset({
    "mouse_left", "mouse_right", "mouse_middle",
    "mouse_move", "mouse_scroll",
})


# ── Keyboard inputs are open-ended strings ────────────────────────────
#
# Keyboard codes (KeyboardEvent.code values) are too numerous to
# enumerate; profiles use them as opaque strings. Validation only
# checks they match the ``[A-Za-z][A-Za-z0-9]*`` shape.
def is_keyboard_code(name: str) -> bool:
    """Recognize a string as a KeyboardEvent.code style identifier."""

    if not name:
        return False
    if not name[0].isalpha():
        return False
    return all(c.isalnum() for c in name)


def is_known_controller_input(name: str) -> bool:
    """True for any abstract gamepad/controller input we recognize.

    Use when:
    - Validating a game profile's bindings refer to controller inputs
      the controller profile actually exposes.
    """

    return name in CONTROLLER_INPUTS or name in POINTER_INPUTS


__all__ = [
    "CONTROLLER_INPUTS",
    "DPAD_INPUTS",
    "FACE_BUTTONS",
    "POINTER_INPUTS",
    "SHOULDER_INPUTS",
    "STICK_CLICK_INPUTS",
    "STICK_DIRECTIONAL_INPUTS",
    "SYSTEM_BUTTONS",
    "is_keyboard_code",
    "is_known_controller_input",
]
