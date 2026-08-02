"""Playtest framing for foundry-driven autonomous play.

The game_agent already drives play from a session ``objective`` string. For
the foundry loop we want a *playtest* framing rather than a "beat the game"
framing: play as a first-time user, pursue the objective, and — critically —
do NOT try to make the run look successful, because the score is computed
externally from what actually happened on screen (see
:mod:`augmentum.game_agent.progress`). Narrating false victory helps nobody;
honest play is what produces a usable defect signal.

This is deliberately a thin string builder, not a new persona subsystem: the
objective field is the seam the orchestrator already consumes.
"""
from __future__ import annotations

_PLAYTEST_FRAME = (
    "You are playtesting a freshly generated game as a first-time player. "
    "Your goal: {objective} "
    "Play naturally and try to make real progress toward that goal. Explore "
    "the controls, react to what is on screen, and keep going until you reach "
    "the goal or clearly cannot. Do not narrate success or declare victory — "
    "your run is graded automatically from the screen, so just play honestly. "
    "If something seems broken (a button does nothing, the screen never "
    "changes, you are stuck), keep trying alternatives rather than stopping."
)


def build_playtest_objective(objective: str) -> str:
    """Wrap a game's objective in the first-time-playtester framing.

    ``objective`` is the game's own goal statement (from the generation
    build spec / ``AUGMENTUM_GAME.objective``). Falls back to a generic
    "reach a win state" when empty so a session always has a target.
    """
    goal = (objective or "").strip() or "reach a win or clear-progress state."
    if not goal.endswith((".", "!", "?")):
        goal += "."
    return _PLAYTEST_FRAME.format(objective=goal)
