"""Tier-scaled build budgets — checkpoint intervals, NOT hard limits.

When a build reaches one of these, it does not fail: it PAUSES, publishes what
it has, and asks the user to continue (resume on the same workspace) or stop
(see ``facade.PAUSE_STOP_REASONS``). So these numbers are "how far to run
before checking in," not a ceiling one bad task turns into a bug.

Two things they encode:

* **Model tier** (``detect_model_tier``): a weaker / local model needs many
  more turns — and, being self-hosted, far cheaper tokens — to reach the same
  place as a frontier model. So small/local gets much bigger budgets than
  frontier.
* **Scope**: a light *edit* checks in sooner than a full *build*.

``max_iterations`` is deliberately a **generous backstop**, not the real
trigger — a browser-driving builder fires one diagnostic tool per turn
(``browser_evaluate`` / ``screenshot`` / ``snapshot``), so counting each as an
iteration would trip the cap on cheap navigation. **Tokens + wallclock** are
the real triggers; iterations only catches a genuine runaway.

These are sensible defaults; tune per your hardware.
"""

from __future__ import annotations

from augmentum.agents.budget import SubagentBudget

#                (max_iterations, max_tokens, max_wallclock_seconds)
_BUILD: dict[str, tuple[int, int, float]] = {
    "small":    (300, 8_000_000, 3600.0),
    "medium":   (220, 5_000_000, 3000.0),
    "large":    (160, 4_000_000, 2400.0),
    "frontier": (120, 3_000_000, 1800.0),
}
_EDIT: dict[str, tuple[int, int, float]] = {
    "small":    (60, 1_500_000, 1200.0),
    "medium":   (45, 1_000_000,  900.0),
    "large":    (35,   800_000,  720.0),
    "frontier": (25,   500_000,  600.0),
}


def build_budget(model: str, scope: str = "build") -> SubagentBudget:
    """A tier-scaled checkpoint budget for ``model`` at the given ``scope``.

    ``scope`` is ``"edit"`` (a light change to an existing app) or ``"build"``
    (a full build/rebuild). Unknown tiers fall back to ``medium``.
    """
    from augmentum.tools.application_scaffolds import detect_model_tier

    tier = detect_model_tier(model or "")
    table = _EDIT if scope == "edit" else _BUILD
    iters, tokens, wall = table.get(tier, table["medium"])
    return SubagentBudget(
        max_iterations=iters, max_tokens=tokens, max_wallclock_seconds=wall,
    )
