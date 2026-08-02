"""Real, self-contained games used as standing evaluation cases.

These are not mocks. Each one is a genuine playable game with its own
rules, rendered to real pixels, that responds to input and can be
finished or failed. They exist so the agent's progress loop can be
exercised end-to-end, unattended and offline, on a game the agent has
no built-in knowledge of — no RAM probe preset, no rule pack, no
walkthrough in the prompt.

That "unfamiliar" property is the point. Anyone can make an agent look
good on a game they wrote a probe for; the question these fixtures
answer is whether the scoring spine still measures real progress when
the only thing available is the screen.
"""

from __future__ import annotations

from augmentum.game_agent.eval_games.tinyquest import (
    TinyQuest,
    TinyQuestAdapter,
)

__all__ = ["TinyQuest", "TinyQuestAdapter"]
