"""Text-entry macro compiler ("type_text" quickaction).

Naming/keyboard screens cost an LLM agent dozens of turns of blind grid
navigation — the single worst turn-to-progress ratio in a playthrough.
This module turns ONE plan action ``{"s": "type_text", "text": "MAY"}``
into the exact primitive press sequence, compiled deterministically from
a per-game keyboard layout. The model expresses intent; the harness does
the typing — the same shape as a coder-mode quickaction.

Layouts are part of the game's translation layer (keyed by the Phase-G
``game_profile``). Source for the Gen-3 layout: pret/pokeemerald
``src/naming_screen.c`` keyboard tables (verified 2026-07):

* 3 pages (UPPER, lower, symbols), SELECT cycles pages, cursor position
  survives the page switch.
* 4 rows x 8 columns; cursor starts at (0,0) on the UPPER page.
* A enters the highlighted character, START jumps to OK, A confirms.

The compiler avoids wrap-around assumptions: it always walks the
straight-line |dx| + |dy| path, so it stays correct even on keyboards
that don't wrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.game_agent.schema import PlanAction

# Press duration for macro steps: short, deliberate taps.
_TAP_MS = 100
_MAX_TEXT_LEN = 12


@dataclass(frozen=True)
class KeyboardLayout:
    """One game's on-screen keyboard, as the compiler needs it."""

    # Ordered pages; page 0 is where the cursor starts. Each page is a
    # list of equal-length strings (rows); spaces are real "space" keys
    # unless listed in ``dead_cells``.
    pages: tuple[tuple[str, ...], ...]
    # Semantic that cycles to the NEXT page (one step in the cycle).
    page_switch: str = "registered"   # SELECT on GBA
    # Semantic that jumps the cursor to OK, and the one that presses it.
    ok_jump: str = "menu"             # START on GBA
    confirm: str = "confirm"          # A
    start_row: int = 0
    start_col: int = 0

    _index: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def find(self, ch: str) -> tuple[int, int, int] | None:
        """(page, row, col) of a character, or None."""

        if not self._index:
            idx: dict[str, tuple[int, int, int]] = {}
            for p, page in enumerate(self.pages):
                for r, row in enumerate(page):
                    for c, cell in enumerate(row):
                        if cell != " " and cell not in idx:
                            idx[cell] = (p, r, c)
            # Space is enterable on Gen-3 keyboards via the blank cells;
            # map it to the first blank on page 0 if present.
            for r, row in enumerate(self.pages[0]):
                c = row.find(" ")
                if c >= 0:
                    idx.setdefault(" ", (0, r, c))
                    break
            object.__setattr__(self, "_index", idx)
        return self._index.get(ch)


_GEN3_KEYBOARD = KeyboardLayout(
    pages=(
        (  # UPPER — the starting page for name entry
            "ABCDEF .",
            "GHIJKL ,",
            "MNOPQRS ",
            "TUVWXYZ ",
        ),
        (  # lower
            "abcdef .",
            "ghijkl ,",
            "mnopqrs ",
            "tuvwxyz ",
        ),
        (  # symbols (digits + punctuation)
            "01234   ",
            "56789   ",
            "!?♂♀/-  ",
            "…“”‘’   ",
        ),
    ),
)

# Keyed by the Phase-G game_profile id. Gen-3 Pokémon share one screen.
LAYOUTS: dict[str, KeyboardLayout] = {
    "pokemon_rs": _GEN3_KEYBOARD,
    "pokemon_emerald": _GEN3_KEYBOARD,
    "pokemon_frlg": _GEN3_KEYBOARD,
}


def has_text_entry(game_profile: str | None) -> bool:
    return bool(game_profile) and game_profile in LAYOUTS


def compile_text_entry(game_profile: str | None, text: str) -> list[PlanAction]:
    """Compile a string into the primitive press sequence.

    Unknown characters are skipped (never guessed). Always ends with
    OK-jump + confirm, so the macro FINISHES the naming screen. Returns
    just [ok_jump, confirm] when the text is empty/unknown — "accept
    what's there and leave" is the safe degenerate case.
    """

    layout = LAYOUTS.get(game_profile or "")
    if layout is None:
        return []

    def tap(semantic: str) -> PlanAction:
        return PlanAction(semantic=semantic, duration_ms=_TAP_MS)

    seq: list[PlanAction] = []
    page, row, col = 0, layout.start_row, layout.start_col
    for ch in text[:_MAX_TEXT_LEN]:
        pos = layout.find(ch)
        if pos is None:
            continue
        tp, tr, tc = pos
        # Page cycle (SELECT steps forward through the page ring).
        steps = (tp - page) % len(layout.pages)
        seq.extend(tap(layout.page_switch) for _ in range(steps))
        page = tp
        # Straight-line walk — no wrap assumptions.
        vert = "nav_down" if tr > row else "nav_up"
        seq.extend(tap(vert) for _ in range(abs(tr - row)))
        horiz = "nav_right" if tc > col else "nav_left"
        seq.extend(tap(horiz) for _ in range(abs(tc - col)))
        row, col = tr, tc
        seq.append(tap(layout.confirm))
    seq.append(tap(layout.ok_jump))
    seq.append(tap(layout.confirm))
    return seq


__all__ = ["LAYOUTS", "KeyboardLayout", "compile_text_entry", "has_text_entry"]
