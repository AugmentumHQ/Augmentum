"""Per-screen interface rules — context-conditional instruction.

The collapsed premise from mode-aware harnesses (Gemini Plays Pokémon's
battle/overworld/menu prompt switching): **rules are conditional on
machine-read context.** We already pay for a ``screen`` probe that
names the active screen from RAM; this table turns that fact into ONE
``RULE=`` line in the fast delta, so the model gets the law of the
screen it is actually on instead of a general lecture.

Scope discipline (this is a *translation layer*, like INPUT_HINTS and
keyboard layouts): rules describe interface physics ONLY — which
inputs are live and what they mean on this screen. Never walkthrough
content, never "where to go". The model still learns the game itself.
"""

from __future__ import annotations

# game_profile -> screen label (from the screen probe's value_labels)
# -> one-line rule. Keep each line tight: it rides every fast turn.
SCREEN_RULES: dict[str, dict[str, str]] = {
    "pokemon_emerald": {
        "title_screen": "menu (START) then confirm advances past the title.",
        "main_menu": "nav_up/nav_down picks a menu row; confirm enters it.",
        "naming_screen": (
            'use type_text NOW ({"s":"type_text","text":"..."}); '
            "never navigate the letter grid manually."
        ),
        "overworld": (
            "OVERWORLD, no dialog open — you are FREE to move: walk with "
            "navigate_to (collision-safe path) toward your goal; confirm "
            "talks/interacts with what you FACE; menu opens the pause menu."
        ),
        "dialog_open": (
            "a DIALOG/TEXT BOX is open: movement is SWALLOWED — press "
            "confirm to advance and READ each box (it is world lore and "
            "instructions). Do not walk or open menus until the box closes."
        ),
        "battle": (
            "you are IN A BATTLE: nav moves the selection, confirm chooses "
            "it, cancel backs out. FIGHT is top-left. navigate_to/menu do "
            "nothing here."
        ),
        "battle_starting": "battle intro playing: confirm advances text; wait otherwise.",
        "battle_ending": "battle wrap-up: confirm advances text until the overworld returns.",
        "bag_menu": "nav moves between items/pockets, confirm selects, cancel exits the bag.",
        "party_menu": "nav picks a Pokémon, confirm opens its actions, cancel exits.",
        "pokemon_summary": "nav_left/nav_right flips pages, cancel exits the summary.",
        "options_menu": "nav moves rows, nav_left/right changes values, cancel exits.",
        "loading_map": "map is loading: emit no inputs, wait one beat.",
    },
}


# Generic screen rules that apply to ANY game without a profile entry.
# These describe universal interface physics (no game-specific knowledge).
# Keys use the scene narrator's labels (lowercase).
_GENERIC_SCREEN_RULES: dict[str, str] = {
    "title": (
        "TITLE SCREEN — a START or MENU button usually advances past"
        " titles into the game; nav rarely does anything here."
    ),
    "overworld": (
        "OVERWORLD — you can move your character via nav_* inputs;"
        " confirm interacts with what you face; menu opens pause/inventory."
    ),
    "battle": (
        "BATTLE — nav moves the selection cursor; confirm chooses the"
        " highlighted option; cancel backs out."
    ),
    "menu": (
        "MENU — nav moves between items/rows; confirm selects; cancel"
        " backs out one level."
    ),
    "dialog": (
        "DIALOG — text is being presented; confirm advances it; movement"
        " is swallowed until the box closes."
    ),
    "cutscene": (
        "CUTSCENE — no input needed; wait for it to finish before acting."
    ),
    "loading": (
        "LOADING — no input needed; wait for the screen to finish."
    ),
    "unknown": (
        "UNKNOWN SCREEN — the scene is unclear; send a short nav or"
        " confirm to probe what responds."
    ),
}

# Fallback for games without a profile entry: the one interface law
# that is near-universal (text boxes swallow movement everywhere).
_GENERIC_DIALOG_RULE = (
    "a dialog/text box is open: movement is swallowed — confirm advances "
    "the text; read it, it is context. Wait to walk until it closes."
)


def screen_rule(game_profile: str | None, screen: object) -> str:
    """The rule line for this game+screen, or '' when none applies.

    Falls back to :data:`_GENERIC_SCREEN_RULES` when the game has no
    profile entry — so any game with a scene narrator gets a RULE= line
    for common screen types (overworld, dialog, battle, menu, ...).
    """

    if not game_profile or not isinstance(screen, str):
        return ""
    return (
        SCREEN_RULES.get(game_profile, {}).get(screen, "")
        or _GENERIC_SCREEN_RULES.get(screen, "")
    )


def modal_rule(game_profile: str | None) -> str:
    """The 'a text box is open' rule — modal state OVERRIDES screen state.

    Dialog is not a *screen*: the screen probe keeps saying "overworld"
    while a text box is open, so callers must swap to this rule whenever
    modal text is active or the screen rule actively misleads ("you are
    free to move") mid-dialog.
    """

    return (
        SCREEN_RULES.get(game_profile or "", {}).get("dialog_open")
        or _GENERIC_DIALOG_RULE
    )


__all__ = ["SCREEN_RULES", "modal_rule", "screen_rule"]
