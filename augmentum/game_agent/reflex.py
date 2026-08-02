"""Model-authored reflex rules ("tier 0").

The FULL planner can emit declarative ``reflex_rules`` in its plan:
small condition→action specs that compile into :class:`Rule` predicates
and fire on the existing fast-path :class:`RuleEngine` — at RAM-tick
speed, with ZERO model involvement. Measured motivation: roughly half of
all actions in a dialog-heavy game are "advance the text box", a
decision that needs no intelligence once proven;每 one currently costs a
~500ms fast turn.

Safety model
------------
The spec is a CLOSED vocabulary (probe compare + fixed actions), not
code: nothing executes beyond string/number comparison, emitted
semantics are filtered against surface caps by the engine (same as every
action path), each rule carries a cooldown and a fires-budget (TTL), and
every firing is logged as a replayable ``rule_fired`` entry. The fast
lane keeps watching the screen, so a misfiring reflex gets escalated and
the planner retracts it with ``{"id": ..., "retract": true}``.

Wire shape (documented to the model in the FULL prompt)::

    {"id": "advance-dialog",
     "when": {"probe": "dialog_text", "changed": true,
              "not_contains": ["?"]},
     "do": [{"s": "confirm", "d": 120}],
     "cooldown_ms": 900, "ttl_fires": 60}

Conditions (all optional except ``probe``, AND-ed):

* ``changed: true``  — the probe's value differs from its previous
  sighting in the event window (first sighting counts as changed).
  Empty/zero values do NOT count as changed unless ``allow_empty`` —
  a dialog box CLOSING must not trigger the advance reflex.
* ``equals: <val>``  — exact match (numbers compared as numbers).
* ``contains / not_contains: [<str>, ...]`` — case-insensitive
  substring tests against the stringified value.
"""

from __future__ import annotations

from typing import Any

import structlog

from augmentum.game_agent.rules import Rule, RuleMatch
from augmentum.game_agent.schema import PlanAction

log = structlog.get_logger(__name__)

# Bounds — a runaway planner can't flood the engine or the input wire.
MAX_ACTIVE_REFLEX_RULES = 8
_MAX_ACTIONS = 4
_MAX_TTL_FIRES = 500
_MIN_COOLDOWN_MS = 250
_DEFAULT_COOLDOWN_MS = 900
_DEFAULT_TTL_FIRES = 60

# Module-level sentinel: "this rule has never fired on any value".
_UNSEEN = object()


def _probe_history(window: list[dict[str, Any]], probe: str) -> list[Any]:
    """The probe's value sightings in the window, oldest first."""

    values: list[Any] = []
    for entry in window:
        if entry.get("kind") != "event":
            continue
        payload = entry.get("payload") or {}
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        probes = data.get("probes")
        if isinstance(probes, dict) and probe in probes:
            values.append(probes[probe])
    return values


def compile_reflex_rule(spec: dict[str, Any]) -> Rule:
    """Compile one declarative spec into an engine :class:`Rule`.

    Raises ``ValueError`` with a model-readable message on any invalid
    shape — the orchestrator logs it as a recoverable ``agent_error``,
    which lands in the planner's LIVE_LOG_TAIL as feedback.
    """

    rule_id = str(spec.get("id") or "").strip()
    if not rule_id or len(rule_id) > 48:
        raise ValueError("reflex rule needs an 'id' (<= 48 chars)")
    when = spec.get("when")
    if not isinstance(when, dict):
        raise ValueError(f"reflex rule {rule_id!r}: 'when' object is required")
    probe = str(when.get("probe") or "").strip()
    if not probe:
        raise ValueError(f"reflex rule {rule_id!r}: when.probe is required")

    changed = bool(when.get("changed", False))
    allow_empty = bool(when.get("allow_empty", False))
    has_equals = "equals" in when
    equals = when.get("equals")
    contains = [str(s).lower() for s in (when.get("contains") or [])]
    not_contains = [str(s).lower() for s in (when.get("not_contains") or [])]
    if not (changed or has_equals or contains or not_contains):
        raise ValueError(
            f"reflex rule {rule_id!r}: 'when' needs at least one condition "
            "(changed / equals / contains / not_contains)"
        )

    raw_actions = spec.get("do")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"reflex rule {rule_id!r}: 'do' must be a non-empty list")
    actions: list[PlanAction] = []
    for item in raw_actions[:_MAX_ACTIONS]:
        if not isinstance(item, dict) or not item.get("s"):
            raise ValueError(f"reflex rule {rule_id!r}: bad action in 'do'")
        try:
            dur = int(item.get("d", 120))
        except (TypeError, ValueError):
            dur = 120
        # Chord extras ("+"): held simultaneously with "s" — lets a
        # reflex encode real-time combos (hold run + jump). Lenient
        # shape check; caps filtering happens in the action worker.
        raw_also = item.get("+") or item.get("also")
        also = None
        if isinstance(raw_also, list):
            also = [str(s) for s in raw_also if isinstance(s, str)][:2] or None
        actions.append(
            PlanAction(
                semantic=str(item["s"]),
                duration_ms=min(2000, max(10, dur)),
                also=also,
            )
        )

    try:
        cooldown_ms = int(spec.get("cooldown_ms", _DEFAULT_COOLDOWN_MS))
    except (TypeError, ValueError):
        cooldown_ms = _DEFAULT_COOLDOWN_MS
    cooldown_ms = max(_MIN_COOLDOWN_MS, cooldown_ms)
    try:
        ttl_fires = int(spec.get("ttl_fires", _DEFAULT_TTL_FIRES))
    except (TypeError, ValueError):
        ttl_fires = _DEFAULT_TTL_FIRES
    ttl_fires = min(_MAX_TTL_FIRES, max(1, ttl_fires))

    state: dict[str, Any] = {"fires": 0, "last_value": _UNSEEN}

    def _predicate(window: list[dict[str, Any]], _caps: Any) -> RuleMatch | None:
        if state["fires"] >= ttl_fires:
            return None
        history = _probe_history(window, probe)
        if not history:
            return None
        value = history[-1]

        if changed:
            # Empty/zero means "the thing went away" (dialog closed);
            # reacting to that is almost always wrong, so it's opt-in.
            if not allow_empty and not value:
                return None
            # "Changed" = differs from the value this rule last fired
            # on (first sighting counts — the bridge only emits probe
            # values when they change, so re-seeing the same value in
            # the window is the same screen, not a new event).
            if state["last_value"] is not _UNSEEN and value == state["last_value"]:
                return None
        text = str(value).lower()
        if has_equals:
            same = (value == equals) or (text == str(equals).lower())
            if not same:
                return None
        if contains and not any(s in text for s in contains):
            return None
        if any(s in text for s in not_contains):
            return None

        state["fires"] += 1
        state["last_value"] = value
        return RuleMatch(
            rule_id=rule_id,
            matched={"probe": probe, "value": str(value)[:120], "fire": state["fires"]},
            actions=list(actions),
        )

    return Rule(
        rule_id=rule_id,
        predicate=_predicate,
        priority=5,  # above default adapter rules; reflexes are deliberate
        cooldown_ms=cooldown_ms,
    )


__all__ = ["MAX_ACTIVE_REFLEX_RULES", "compile_reflex_rule"]
