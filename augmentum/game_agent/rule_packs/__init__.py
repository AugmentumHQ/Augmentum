"""Per-surface fast-path rule packs.

A rule pack is a small set of :class:`Rule` predicates tuned for a
specific game / log schema. The orchestrator's :class:`RuleEngine`
evaluates them on every event so reflex behavior (advance dialogue,
flee at critical HP, etc.) responds at probe-tick latency (~250 ms)
instead of waiting for the slow-path LLM (~10-20 s).

This module is a small registry: callers ask for "the rule pack for
log_schema=X", get back a :class:`RuleEngine` with rules
pre-registered, hand it to the orchestrator. New surface presets add
themselves by importing here and registering in :data:`_PACK_BUILDERS`.

Why a registry rather than direct imports in the route layer
------------------------------------------------------------
Routes know about ``log_schema`` strings; the orchestrator knows about
:class:`RuleEngine`. Putting the mapping here keeps both free of
per-game knowledge -- adding a new game means adding ONE entry in
:data:`_PACK_BUILDERS`, not editing every consumer.
"""

from __future__ import annotations

from collections.abc import Callable

from augmentum.game_agent.rule_packs import pokemon_rs
from augmentum.game_agent.rules import RuleEngine

_PACK_BUILDERS: dict[str, Callable[[], RuleEngine]] = {
    "pokemon_rs.v1": pokemon_rs.build_rule_engine,
    # Emerald shares the Gen-3 interface physics; own schema id so the
    # translation layer (walk_grid nav, screen rules) keys correctly.
    "pokemon_emerald.v1": pokemon_rs.build_rule_engine,
    # Add new schemas here: "pokemon_rby.v1": pokemon_rby.build_rule_engine
}


def rule_engine_for_log_schema(schema: str) -> RuleEngine | None:
    """Return a fresh rule engine for ``schema``, or ``None`` if no pack.

    Use when:
    - The route layer is constructing an :class:`Orchestrator` and wants
      surface-appropriate reflex behavior in front of the slow path.

    Returns:
    - A fresh :class:`RuleEngine` with the pack's rules pre-registered,
      or ``None`` when no pack matches. ``None`` is the safe default
      -- the orchestrator falls back to an empty rule engine and only
      the slow path drives behavior.
    """

    builder = _PACK_BUILDERS.get(schema)
    return builder() if builder else None


__all__ = ["rule_engine_for_log_schema"]
