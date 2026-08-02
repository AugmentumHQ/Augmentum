"""Fast-path rule engine.

The fast path is the reflex layer. It runs on every event (typically
~100 Hz) and short-circuits the slow-path LLM for high-frequency,
deterministic responses: tap-throttling, low-HP flee sequences,
release-on-collision, etc.

Rules are plain Python predicates over a sliding window of recent log
entries. They are *not* a config DSL -- the surface adapter ships its
own rule set as Python functions; the engine evaluates them in
priority order on each tick.

Design intent
-------------
* **No mini-language.** A predicate is ``(window, surface_caps) -> RuleMatch | None``.
  The rule author has the full Python surface to inspect events,
  which is fine because rules ship with the adapter (trusted code) --
  not with user content.
* **Priority and cooldown.** Higher priority wins on conflict; each
  rule has a per-rule cooldown so a flapping condition doesn't spam
  the input wire.
* **Replayable.** Every rule firing emits a :class:`RuleFiredEntry`
  into the log, so a session replay reconstructs identical behavior.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from augmentum.game_agent.schema import PlanAction, SurfaceCapsPayload


@dataclass(frozen=True)
class RuleMatch:
    """A rule fired against the current event window."""

    rule_id: str
    matched: dict[str, Any]
    actions: list[PlanAction]


@dataclass
class Rule:
    """A fast-path predicate + action emitter.

    ``predicate`` is invoked with the rolling event window (newest
    last) and the active :class:`SurfaceCapsPayload`. It returns a
    :class:`RuleMatch` to fire, or ``None`` to pass.
    """

    rule_id: str
    predicate: Callable[[list[dict[str, Any]], SurfaceCapsPayload], RuleMatch | None]
    priority: int = 0
    cooldown_ms: int = 0
    # Internal state: last-fired tick for cooldown enforcement.
    _last_fired_t: int | None = field(default=None, init=False, repr=False)


class RuleEngine:
    """Owns the rule set and the rolling event window.

    Use when:
    - The orchestrator wants per-tick reflex evaluation in front of the
      slow-path planner.

    Expects:
    - Rules are registered before the session starts (or hot-swapped
      between sessions). The window is in-memory only -- on process
      restart, replay from the NDJSON log.

    Returns:
    - From :meth:`tick`, a list of :class:`RuleMatch` -- typically zero
      or one entry; multiple are possible when high-priority rules
      stack and don't conflict.
    """

    def __init__(self, *, window_size: int = 128) -> None:
        self._rules: list[Rule] = []
        self._window: deque[dict[str, Any]] = deque(maxlen=window_size)

    def register(self, rule: Rule) -> None:
        # Same-id registration REPLACES (model-authored reflex rules
        # refresh themselves by re-emitting their id with new specs).
        self._rules = [r for r in self._rules if r.rule_id != rule.rule_id]
        self._rules.append(rule)
        # Keep highest priority first; ties keep insertion order so
        # adapter-author intent is preserved.
        self._rules.sort(key=lambda r: -r.priority)

    def unregister(self, rule_id: str) -> bool:
        """Remove a rule by id. Returns True when something was removed."""

        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    @property
    def rule_ids(self) -> list[str]:
        return [r.rule_id for r in self._rules]

    def observe(self, entry: dict[str, Any]) -> None:
        """Add an entry to the rolling window. Called by the orchestrator."""

        self._window.append(entry)

    def tick(
        self,
        now_ms: int,
        caps: SurfaceCapsPayload,
    ) -> list[RuleMatch]:
        """Evaluate rules against the current window.

        Returns the matches that fired this tick. The orchestrator is
        responsible for: (a) logging each match as a
        :class:`RuleFiredEntry`, (b) emitting the matched actions to
        the surface adapter, and (c) writing :class:`InputEntry` lines
        with ``source="rule"`` for each emitted action.
        """

        window_snapshot = list(self._window)
        fired: list[RuleMatch] = []
        for rule in self._rules:
            if (
                rule._last_fired_t is not None
                and now_ms - rule._last_fired_t < rule.cooldown_ms
            ):
                continue
            match = rule.predicate(window_snapshot, caps)
            if match is None:
                continue
            # Filter actions to only those whose semantic is bound on
            # the active surface; an adapter rule should not be able to
            # emit something the surface can't execute.
            allowed = set(caps.semantic_inputs)
            kept = [a for a in match.actions if a.semantic in allowed]
            if not kept:
                continue
            rule._last_fired_t = now_ms
            fired.append(RuleMatch(rule.rule_id, match.matched, kept))
        return fired
