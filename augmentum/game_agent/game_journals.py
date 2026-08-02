"""Default journal seeds for known game profiles.

When a player starts a title for the first time (no on-disk journal),
the agent begins completely blind: empty scratchpad, no objectives, no
knowledge of the game's intro sequence.  The first ~8 fast turns fire
before the slow-path planner even writes a journal entry — during that
window the model guesses mechanics, tries navigate_to without a walk
grid, and generally makes avoidable mistakes.

A seed journal pre-loads the structural knowledge that would otherwise
take several sessions to discover:
- What the intro sequence looks like (title → naming → dialog → overworld)
- Which inputs are live on each intro screen
- Constraints that aren't obvious from the screen (navigate_to needs a
  walk_grid; dialog swallows movement; etc.)

Seeds are short, telegraphic, and title-specific.  The agent overwrites
them as it plays, so stale advice naturally ages out.
"""

from __future__ import annotations

from augmentum.game_agent.journal import JournalSections

# game_profile -> default sections.  Loaded when no on-disk journal exists
# (first session for this user+title pair).
_SEEDS: dict[str, JournalSections] = {
    "pokemon_emerald": JournalSections(
        status="Starting new game — in mandatory intro sequence",
        progress="Not yet started gameplay",
        objectives=(
            "FINAL: complete the game\n"
            "MEDIUM: reach the overworld and win the first battle\n"
            "SHORT: complete the intro (title→name→Birch dialog→lorry→Littleroot→overworld)"
        ),
        notes=[
            # Intro sequence — the most important thing to know up front
            "INTRO ORDER: title_screen(press menu/START) → main_menu(confirm 'New Game') "
            "→ naming_screen(type_text in ONE action, e.g. type_text 'RED') "
            "→ long Birch dialog(press confirm MANY times — do NOT navigate) "
            "→ rival naming screen(type_text 'GARY' or any name) "
            "→ lorry/Littleroot cutscene(wait then confirm through text) "
            "→ house dialog → overworld",

            # Hard constraint that burns the first few turns without this note
            "navigate_to only works after walk_grid is loaded: player_x and player_y "
            "must show valid overworld coords (not 0 or garbage intro values). "
            "Use single nav presses during intro; switch to navigate_to once in overworld.",

            # Dialog mechanics
            "During ALL dialog/cutscene: press confirm repeatedly, do NOT use nav or "
            "navigate_to — movement is swallowed while any text box is open. "
            "dialog_text changing in DELTA = the confirm worked, press again.",

            # Naming screen shortcut
            "naming_screen: use type_text IMMEDIATELY (one action types + confirms). "
            "Never navigate the letter grid manually — type_text is faster and error-free.",
        ],
    ),
    "pokemon_rs": JournalSections(
        status="Starting new game — in mandatory intro sequence",
        progress="Not yet started gameplay",
        objectives=(
            "FINAL: complete the game\n"
            "MEDIUM: reach the overworld and win the first battle\n"
            "SHORT: complete the intro (title→name→Birch dialog→lorry→Littleroot→overworld)"
        ),
        notes=[
            "INTRO ORDER: title_screen(menu/START) → main_menu(New Game) "
            "→ naming_screen(type_text) → Birch dialog(confirm many times) "
            "→ rival naming(type_text) → lorry → overworld",
            "navigate_to requires walk_grid — use nav presses during intro sequence.",
            "dialog open: press confirm only, movement is swallowed.",
            "naming_screen: type_text in one action, never navigate the grid.",
        ],
    ),
}


def default_journal_sections(game_profile: str | None) -> JournalSections | None:
    """Return seed sections for ``game_profile``, or None if no seed exists.

    Used by the route layer when creating a CompanionJournal for a
    game that has no on-disk journal yet.  The seed gives the agent
    structural intro knowledge from turn 1 instead of waiting for the
    first slow-path plan to accumulate it through trial and error.
    """

    if not game_profile:
        return None
    return _SEEDS.get(game_profile)


__all__ = ["default_journal_sections"]
