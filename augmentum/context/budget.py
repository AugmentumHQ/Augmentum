"""Per-turn token budget governance for Becca's prompt assembly.

Odysseus's 2026-Q3 roadmap calls out "agent prompt/context bloat" as
their #1 named pain point: tool schemas + skills + memory + documents
+ instructions can eat the context window before the user's actual
request lands. Augmentum is on the same trajectory as the action
catalog grows, observation substrate fills in, persona deepens, and
multi-modal recall flows into Becca's prompt.

This module instruments the assembly so we can SEE the spend per
contributor and catch overrun before it becomes a silent quality
regression. Today the tracker MEASURES; downstream assemblers (memory
recall, knowledge injection, persona builder) can cooperate by reading
``tracker.remaining()`` before deciding how much to include.

Usage:

    tracker = BudgetTracker(total_cap=8000)
    tracker.add("system", system_prompt)
    tracker.add("persona", persona_block)
    tracker.add("memory/active", memory_recall_block)
    tracker.add("documents/rag", rag_block)
    tracker.add("knowledge/pack", pack_block)
    tracker.add("recent_messages", history_block)

    if tracker.over_budget:
        log.warning("context_over_budget", **tracker.snapshot().as_log())

The labels match :mod:`augmentum.security.untrusted` wrap labels where
applicable so a single label conveys both "what is it" and "where in
the assembly did it come from" — useful for cross-referencing the
spend metrics against the prompt-injection wrap audit.

Token counting uses the shared :mod:`augmentum.utils.tokenizer`
helper which lazy-loads tiktoken (cl100k_base) and falls back to a
character estimate when tiktoken isn't available. Counts are
approximate (~15-20% off for Llama/Mistral tokenizers) but consistent
across the contributors, which is what matters for budget allocation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger
from augmentum.utils.tokenizer import count_tokens

log = get_logger(__name__)


# Default per-contributor soft caps as fractions of the total cap.
# Callers can override per-tracker; the defaults match the dominant
# chat-style turn shape (long persona + memory, short tool schemas).
# Sum is intentionally < 1.0 — the remainder is the user's input + the
# completion budget.
_DEFAULT_CONTRIBUTOR_FRACTIONS: dict[str, float] = {
    "system": 0.05,
    "persona": 0.10,
    "memory/active": 0.10,
    "memory/core": 0.05,
    "documents/rag": 0.10,
    "knowledge/pack": 0.10,
    "observation": 0.05,
    "tool_schemas": 0.05,
    "recent_messages": 0.20,
    # 0.20 left for user input + completion
}


# Global hard ceiling so a misconfigured caller can't allocate an
# absurd budget. The largest Augmentum-supported context window today
# is in the low hundreds of thousands of tokens; this cap is well above
# that as a sanity backstop. Bump if a future model genuinely needs
# more.
_ABSOLUTE_MAX_CAP = 2_000_000


@dataclass
class BudgetSnapshot:
    """Frozen view of what the tracker has observed so far.

    Useful for logging, /api/health surfaces, and post-mortem analysis
    when a turn produces unexpected output (often correlated with
    prompt bloat).
    """

    total_cap: int
    used: int
    remaining: int
    over_budget: bool
    by_label: dict[str, int] = field(default_factory=dict)
    started_at: float = 0.0

    def as_log(self) -> dict:
        """Render for structlog. Keeps key names short + lowercase to
        match Augmentum's logging conventions."""
        return {
            "budget_total": self.total_cap,
            "budget_used": self.used,
            "budget_remaining": self.remaining,
            "budget_over": self.over_budget,
            "budget_by_label": dict(self.by_label),
            "budget_elapsed_s": round(time.time() - self.started_at, 3),
        }


class BudgetTracker:
    """Meter per-contributor token spend during a single turn's prompt
    assembly.

    One tracker per turn. Thread-unsafe by design — turns are
    sequential within a session, and the cost of synchronisation
    isn't worth the overhead.

    The tracker does NOT mutate prompts. It records what callers
    submit; enforcing budget caps is the caller's responsibility. A
    cooperative caller reads :meth:`remaining` before deciding how
    much content to include.
    """

    def __init__(
        self,
        total_cap: int,
        *,
        contributor_caps: dict[str, int] | None = None,
        contributor_fractions: dict[str, float] | None = None,
        model_name: str = "",
    ) -> None:
        if total_cap <= 0:
            raise ValueError(f"total_cap must be positive, got {total_cap}")
        if total_cap > _ABSOLUTE_MAX_CAP:
            raise ValueError(
                f"total_cap {total_cap} exceeds absolute max "
                f"{_ABSOLUTE_MAX_CAP} — likely misconfiguration"
            )
        self._total_cap = total_cap
        self._used = 0
        self._by_label: dict[str, int] = {}
        self._started_at = time.time()
        self._model_name = model_name

        # Resolve per-contributor caps. Explicit overrides win;
        # otherwise we compute from fractions (caller-provided or
        # default).
        fractions = contributor_fractions or _DEFAULT_CONTRIBUTOR_FRACTIONS
        derived = {label: int(total_cap * frac) for label, frac in fractions.items()}
        if contributor_caps:
            derived.update(contributor_caps)
        self._contributor_caps = derived

    # ── Reading ──────────────────────────────────────────────────────

    @property
    def total_cap(self) -> int:
        return self._total_cap

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self._total_cap - self._used)

    @property
    def over_budget(self) -> bool:
        return self._used > self._total_cap

    def used_by(self, label: str) -> int:
        """Tokens consumed under a specific label."""
        return self._by_label.get(label, 0)

    def cap_for(self, label: str) -> int:
        """Soft cap configured for this contributor; 0 = no per-label
        cap."""
        return self._contributor_caps.get(label, 0)

    def over_label_cap(self, label: str) -> bool:
        cap = self.cap_for(label)
        return cap > 0 and self.used_by(label) > cap

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            total_cap=self._total_cap,
            used=self._used,
            remaining=self.remaining,
            over_budget=self.over_budget,
            by_label=dict(self._by_label),
            started_at=self._started_at,
        )

    # ── Recording ────────────────────────────────────────────────────

    def add(self, label: str, content: str, *, tokens: int | None = None) -> int:
        """Record ``content`` contributed under ``label``.

        Args:
            label: contributor identity (``system``, ``persona``,
                ``memory/active``, ``documents/rag``, ...). Should
                match the :mod:`augmentum.security.untrusted` wrap
                label where applicable so the budget snapshot and the
                untrusted-wrap audit cross-reference cleanly.
            content: the string the caller plans to include.
            tokens: explicit token count if the caller has already
                computed it (e.g. via a model-specific tokenizer).
                When ``None`` we count via the shared cl100k_base
                tokenizer.

        Returns the token count recorded (so callers can record + react
        in one line: ``cost = tracker.add("memory/active", block)``).
        """
        if not content:
            return 0
        n = tokens if tokens is not None else count_tokens(content)
        if n < 0:
            n = 0
        self._used += n
        self._by_label[label] = self._by_label.get(label, 0) + n
        return n

    def add_count(self, label: str, tokens: int) -> int:
        """Record token spend WITHOUT having the content string handy.

        Useful when the caller knows the count (e.g. from the model's
        usage response) but doesn't want to re-tokenise. Negative
        counts are clamped to zero.
        """
        if tokens <= 0:
            return 0
        self._used += tokens
        self._by_label[label] = self._by_label.get(label, 0) + tokens
        return tokens

    # ── Termination ──────────────────────────────────────────────────

    def commit(self) -> BudgetSnapshot:
        """Finalise the tracker — log if over budget, return snapshot.

        Call at the end of prompt assembly so an overrun produces an
        observable signal. The log includes per-label spend so we can
        see WHICH contributor blew the budget on the offending turn
        (almost always memory or knowledge in early-stage triage).
        """
        snap = self.snapshot()
        if self.over_budget:
            log.warning(
                "context_over_budget",
                model=self._model_name or "unknown",
                **snap.as_log(),
            )
        return snap
