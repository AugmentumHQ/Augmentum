"""Fast-path RuleEngine tests."""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.rules import Rule, RuleEngine, RuleMatch
from augmentum.game_agent.schema import PlanAction, SurfaceCapsPayload


def _caps(*semantics: str) -> SurfaceCapsPayload:
    return SurfaceCapsPayload(
        semantic_inputs=list(semantics),
        log_schema="mock.v1",
        observation_modalities=["log"],  # type: ignore[list-item]
    )


def _flee_predicate(
    window: list[dict[str, Any]], _caps: SurfaceCapsPayload
) -> RuleMatch | None:
    for entry in reversed(window):
        if entry.get("kind") == "event":
            payload = entry.get("payload", {})
            data = payload.get("data", {})
            if data.get("event") == "damage_taken" and data.get("hp", 100) < 20:
                return RuleMatch(
                    rule_id="flee_on_low_hp",
                    matched={"hp": data["hp"]},
                    actions=[PlanAction(semantic="flee", duration_ms=500)],
                )
    return None


def test_predicate_fires_on_match() -> None:
    """@example: a predicate that finds its trigger event fires once per tick."""

    eng = RuleEngine()
    eng.register(Rule(rule_id="flee_on_low_hp", predicate=_flee_predicate))
    eng.observe(
        {
            "t": 100,
            "kind": "event",
            "payload": {"channel": "log", "data": {"event": "damage_taken", "hp": 12}},
        }
    )
    matches = eng.tick(now_ms=100, caps=_caps("flee"))
    assert len(matches) == 1
    assert matches[0].rule_id == "flee_on_low_hp"
    assert matches[0].matched == {"hp": 12}


def test_cooldown_suppresses_repeat() -> None:
    """@example: a rule with cooldown does not fire twice inside the window."""

    eng = RuleEngine()
    eng.register(
        Rule(rule_id="flee_on_low_hp", predicate=_flee_predicate, cooldown_ms=1000)
    )
    eng.observe(
        {
            "t": 100,
            "kind": "event",
            "payload": {"channel": "log", "data": {"event": "damage_taken", "hp": 12}},
        }
    )
    first = eng.tick(now_ms=100, caps=_caps("flee"))
    second = eng.tick(now_ms=500, caps=_caps("flee"))
    third = eng.tick(now_ms=1500, caps=_caps("flee"))
    assert len(first) == 1
    assert second == []
    assert len(third) == 1


def test_unbound_semantic_filtered_out() -> None:
    """@example: a rule that emits a semantic the surface does not accept is dropped."""

    eng = RuleEngine()
    eng.register(Rule(rule_id="flee_on_low_hp", predicate=_flee_predicate))
    eng.observe(
        {
            "t": 100,
            "kind": "event",
            "payload": {"channel": "log", "data": {"event": "damage_taken", "hp": 12}},
        }
    )
    # Caps does NOT include "flee".
    matches = eng.tick(now_ms=100, caps=_caps("noop"))
    assert matches == []
