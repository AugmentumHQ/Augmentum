"""Input-style fingerprinting — observed DOM listeners → adapter chain.

The probe instruments a game in a headless browser and records which
input APIs it actually wires up (via ``addEventListener`` traps + a
``navigator.getGamepads`` poll trap). This module is the PURE half: it
maps that observed-event set onto an ordered ``input_chain`` the universal
adapter loader can activate. No browser, no I/O — trivially testable.

Mapping rationale (the loader activates EVERY adapter in the chain, so
each observed style just needs its adapter present):

  - keyboard events (keydown/keyup/keypress)      → ``keyboard``
  - touch events    (touchstart/touchmove/…)      → ``touch``
  - pointer/mouse   (pointerdown/mousedown/…)      → ``pointer``
  - gamepad         (gamepadconnected / getGamepads poll) → ``gamepad_api``

``gamepad_api`` is ALWAYS included: it's cheap, harmless to games that
ignore it, and is the legacy default every play surface already ships.
The non-gamepad adapters are added only when the game demonstrably
listens for them, so we don't synthesise events nothing consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Observed-event substrings → adapter id. Matched case-insensitively
# against the raw event-type names the trap recorded.
_KEYBOARD_EVENTS = ("keydown", "keyup", "keypress")
_TOUCH_EVENTS = ("touchstart", "touchmove", "touchend", "touchcancel")
_POINTER_EVENTS = (
    "pointerdown", "pointerup", "pointermove",
    "mousedown", "mouseup", "mousemove", "click",
)
_GAMEPAD_EVENTS = ("gamepadconnected", "gamepaddisconnected")

# Sentinel the probe emits when it observed a ``navigator.getGamepads``
# call (polling games never addEventListener, they poll every frame).
GETGAMEPADS_POLL = "__getgamepads_poll__"


@dataclass(frozen=True)
class InputFingerprint:
    """Result of fingerprinting a game's input wiring."""

    # Ordered adapter chain for CastProfile.input_chain.
    input_chain: tuple[str, ...]
    # Human-readable styles observed, for notes / diagnostics.
    styles: tuple[str, ...] = ()
    # The raw observed event names that drove the decision.
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def to_chain(self) -> tuple[str, ...]:
        return self.input_chain


def _matches_any(observed_lower: set[str], needles: tuple[str, ...]) -> list[str]:
    """Return the needles present in the observed set (exact, lowercased)."""
    return [n for n in needles if n in observed_lower]


def classify_input_style(observed_events: list[str] | None) -> InputFingerprint:
    """Map a list of observed event names → an ordered input_chain.

    ``observed_events`` is whatever the browser trap recorded:
    ``addEventListener`` type strings plus the ``GETGAMEPADS_POLL``
    sentinel when polling was seen. Unknown / empty input falls back to
    the safe default chain ``('gamepad_api',)``.
    """
    observed_lower = {str(e or "").strip().lower() for e in (observed_events or [])}
    observed_lower.discard("")

    kb = _matches_any(observed_lower, _KEYBOARD_EVENTS)
    touch = _matches_any(observed_lower, _TOUCH_EVENTS)
    pointer = _matches_any(observed_lower, _POINTER_EVENTS)
    gamepad = _matches_any(observed_lower, _GAMEPAD_EVENTS)
    polls_gamepad = GETGAMEPADS_POLL in observed_lower

    chain: list[str] = ["gamepad_api"]   # always present, cheap default
    styles: list[str] = []
    if gamepad or polls_gamepad:
        styles.append("gamepad")
    if kb:
        chain.append("keyboard")
        styles.append("keyboard")
    if touch:
        chain.append("touch")
        styles.append("touch")
    if pointer:
        chain.append("pointer")
        styles.append("pointer")

    evidence = tuple(sorted(observed_lower))
    return InputFingerprint(
        input_chain=tuple(chain),
        styles=tuple(styles),
        evidence=evidence,
    )


# JS the probe injects at document-start to trap input wiring. Kept here
# (not in the driver) so a test can assert it traps the right surface +
# the driver stays a thin runner. It accumulates into a global the probe
# reads back after the game has had a moment to boot.
INSTRUMENTATION_JS = r"""
(() => {
  const seen = new Set();
  const _add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, ...rest) {
    try { if (type) seen.add(String(type).toLowerCase()); } catch (_) {}
    return _add.call(this, type, ...rest);
  };
  // Polling games never addEventListener — trap the poll itself.
  try {
    const _gg = navigator.getGamepads && navigator.getGamepads.bind(navigator);
    if (_gg) {
      navigator.getGamepads = function() {
        seen.add('__getgamepads_poll__');
        return _gg();
      };
    }
  } catch (_) {}
  window.__augProbeObserved = () => Array.from(seen);
})();
"""
