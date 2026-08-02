"""Input-context inference — the agnostic layer under "screens".

Every game, whatever it calls things, is a stack of INPUT CONTEXTS:

* ``free_move`` — inputs move your character/avatar through the world
* ``cursor``    — inputs move a selection, not the character (menus,
  battles, shops, naming grids, save prompts, title screens — every
  menu-shaped mechanic in every game, without naming any of them)
* ``reading``   — text is being presented; one input advances it and
  everything else is swallowed
* ``locked``    — nothing responds (cutscene / animation / loading)

The tracker infers the live context from evidence the loop already
collects — per-button effect scores (ground truth for "did that press
do anything"), position-fact motion, and text-probe activity — so it
needs ZERO game knowledge and works on a title nobody wrote a
translation layer for. Per-game screen rules remain a *refinement* on
top; this is the backbone.

Discriminators
--------------
* text probe changed recently        → reading
* several recent presses, all dead   → locked
* position facts moving              → free_move
* screen reacts but position doesn't → cursor
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A press "worked" when its core-level frame-diff score clears this —
# same threshold the dead-press gate and reflex pack use.
_ALIVE = 48
# Evidence windows (ms): fx older than this no longer describes the
# current context; position motion is meaningful a little longer than
# one walk-step animation.
_FX_WINDOW_MS = 8000
_POS_WINDOW_MS = 4000
_TEXT_WINDOW_MS = 6000

_NAV_PREFIX = "nav_"

# The generic, mechanics-named (never game-named) instruction per mode.
MODE_LINES: dict[str, str] = {
    "reading": (
        "READING: text is being presented — confirm advances it; movement "
        "and menu inputs are swallowed. Read the text, it is context."
    ),
    "cursor": (
        "SELECTION: inputs move a cursor/selection, NOT your character — "
        "nav changes the highlighted option, confirm commits, cancel backs "
        "out. You are not walking anywhere until this closes."
    ),
    "free_move": (
        "FREE MOVEMENT: your character moves through the world — walk "
        "toward your goal (navigate_to when available); confirm interacts "
        "with what you face."
    ),
    "locked": (
        "LOCKED: recent inputs had NO effect (cutscene/animation/loading) "
        '— emit "a":[] and wait for the world to change instead of '
        "pressing more buttons."
    ),
}


@dataclass
class InputContextTracker:
    """Rolling evidence → current input context. Cheap, synchronous."""

    _fx: list[tuple[str, int, int]] = field(default_factory=list)  # (button, score, t)
    _text_until: int = 0
    _last_pos_move_ms: int = -1
    _last_screen_change_ms: int = -1

    # ── evidence feeds ────────────────────────────────────────────

    def feed_fx(self, button: str, score: int, t_ms: int) -> None:
        self._fx.append((button, score, t_ms))
        if len(self._fx) > 24:
            self._fx = self._fx[-24:]

    def feed_text_activity(self, t_ms: int) -> None:
        """A *_text probe changed — a box is printing right now."""

        self._text_until = t_ms + _TEXT_WINDOW_MS

    def end_text_activity(self) -> None:
        """Evidence the box closed (e.g. a movement press worked)."""

        self._text_until = 0

    def feed_position_change(self, t_ms: int) -> None:
        """Any position fact moved (tile/coordinate change)."""

        self._last_pos_move_ms = t_ms

    def feed_screen_change(self, t_ms: int) -> None:
        self._last_screen_change_ms = t_ms

    # ── inference ─────────────────────────────────────────────────

    def infer(self, now_ms: int) -> str:
        """The live context id, or '' when evidence is insufficient."""

        if now_ms < self._text_until:
            return "reading"
        recent = [
            (b, s) for b, s, t in self._fx if now_ms - t <= _FX_WINDOW_MS
        ]
        pos_moving = (
            self._last_pos_move_ms >= 0
            and now_ms - self._last_pos_move_ms <= _POS_WINDOW_MS
        )
        if len(recent) >= 3 and all(s < _ALIVE for _b, s in recent):
            # Everything the agent tried recently bounced off — but a
            # world that is moving on its own (position drifting, e.g.
            # a scripted walk) is a cutscene to WATCH, still locked.
            return "locked"
        if pos_moving:
            return "free_move"
        nav_alive = any(
            s >= _ALIVE for b, s in recent if b.startswith(_NAV_PREFIX)
        )
        act_alive = any(
            s >= _ALIVE for b, s in recent if not b.startswith(_NAV_PREFIX)
        )
        if nav_alive or act_alive:
            # Inputs register, the character isn't moving → a selection
            # context, whatever this game happens to call it.
            return "cursor"
        return ""

    def mode_line(self, now_ms: int) -> str:
        """The MODE= instruction for the current context ('' if unknown)."""

        return MODE_LINES.get(self.infer(now_ms), "")


__all__ = ["MODE_LINES", "InputContextTracker"]
