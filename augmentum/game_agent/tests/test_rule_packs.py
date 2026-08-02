"""Rule pack tests — focus on predicate correctness without an orchestrator.

The predicates are pure functions of (event window, surface caps), so we
build synthetic windows directly and assert the firing decision. No
asyncio, no LLM, no adapter.
"""

from __future__ import annotations

from augmentum.game_agent.rule_packs import (
    rule_engine_for_log_schema,
)
from augmentum.game_agent.rule_packs.pokemon_rs import (
    _auto_continue_dialog,
    _is_player_stationary,
    build_rule_engine,
)
from augmentum.game_agent.schema import SurfaceCapsPayload


def _caps() -> SurfaceCapsPayload:
    return SurfaceCapsPayload(
        semantic_inputs=["a", "b", "up", "down", "left", "right", "start", "select"],
        log_schema="pokemon_rs.v1",
        observation_modalities=["log", "frame"],
    )


def _ram_event(x: int, y: int, t: int = 0) -> dict:
    return {
        "t": t,
        "kind": "event",
        "payload": {
            "channel": "log",
            "data": {
                "event": "ram",
                "probes": {"player_x": x, "player_y": y},
            },
        },
    }


def _input_ack(button: str, effect: int, t: int = 0) -> dict:
    return {
        "t": t,
        "kind": "event",
        "payload": {
            "channel": "log",
            "data": {
                "event": "input_ack",
                "button": button,
                "held_ms": 120,
                "tick_count": 7,
                "effect_score": effect,
            },
        },
    }


# ── _is_player_stationary ─────────────────────────────────────────────


def test_stationary_requires_at_least_two_coord_readings() -> None:
    """@example: with <2 ram events the function refuses to declare stationary.

    ROOT CAUSE:
      Treating "no evidence" as stationary would cause the dialogue
      rule to fire on the very first turn before any probe data has
      arrived -- worst case it mashes A while the player is mid-step
      and races overworld input handling. Conservative refusal.
    """

    assert _is_player_stationary([]) is False
    assert _is_player_stationary([_ram_event(10, 5)]) is False


def test_stationary_detects_no_change_across_recent_readings() -> None:
    """@example: identical coordinates across recent ram events == stationary."""

    window = [
        _ram_event(10, 5, t=100),
        _ram_event(10, 5, t=200),
        _ram_event(10, 5, t=300),
    ]
    assert _is_player_stationary(window) is True


def test_stationary_detects_movement() -> None:
    """@example: any coord change disqualifies the stationary signal."""

    window = [
        _ram_event(10, 5, t=100),
        _ram_event(11, 5, t=200),  # moved right
    ]
    assert _is_player_stationary(window) is False


def test_stationary_ignores_non_ram_events() -> None:
    """@example: input acks / other event types are skipped during scan."""

    window = [
        _ram_event(10, 5, t=100),
        _input_ack("a", 90, t=150),  # not a coord reading
        _input_ack("a", 100, t=180),
        _ram_event(10, 5, t=200),
    ]
    # Two ram events, both (10, 5), separated by non-ram noise → stationary.
    assert _is_player_stationary(window) is True


# ── _auto_continue_dialog ─────────────────────────────────────────────


def test_dialog_rule_fires_on_a_press_with_high_effect_and_stationary() -> None:
    """@example: A-press had visible effect + player not moving → press A again.

    This is the canonical dialogue-advance pattern. The rule reads the
    most recent input_ack, confirms it was for A with effect_score
    above threshold, and confirms the player is stationary across
    recent ram readings.
    """

    window = [
        _ram_event(10, 5, t=100),
        _ram_event(10, 5, t=200),
        _input_ack("a", 250, t=300),  # large change → text appeared
    ]
    match = _auto_continue_dialog(window, _caps())
    assert match is not None
    assert match.rule_id == "auto_continue_dialog"
    assert len(match.actions) == 1
    assert match.actions[0].semantic == "a"
    # Holds for our configured 120ms so the libretro core's input poll
    # catches it across multiple retro_run frames.
    assert match.actions[0].duration_ms == 120


def test_dialog_rule_does_not_fire_when_effect_below_threshold() -> None:
    """@example: a low effect_score means the press didn't really do anything.

    ROOT CAUSE:
      Self-termination logic. When the textbox closes and the next A
      press just shows the same overworld frame, effect_score drops
      near 0. The rule must NOT keep firing -- otherwise it would
      mash A indefinitely after dialogue ends.
    """

    window = [
        _ram_event(10, 5, t=100),
        _ram_event(10, 5, t=200),
        _input_ack("a", 5, t=300),  # below threshold
    ]
    assert _auto_continue_dialog(window, _caps()) is None


def test_dialog_rule_does_not_fire_when_player_moving() -> None:
    """@example: a moving player is in the overworld; don't auto-mash A.

    ROOT CAUSE:
      A textbox can appear *during* a step trigger. If we treat
      mid-walk effect_score as dialogue and start mashing, the
      character can produce surprise jumps when the trigger resolves.
    """

    window = [
        _ram_event(10, 5, t=100),
        _ram_event(11, 5, t=200),  # moved
        _input_ack("a", 200, t=300),
    ]
    assert _auto_continue_dialog(window, _caps()) is None


def test_dialog_rule_does_not_fire_on_non_a_input_ack() -> None:
    """@example: an input_ack for a non-A button is ignored."""

    window = [
        _ram_event(10, 5, t=100),
        _ram_event(10, 5, t=200),
        _input_ack("start", 200, t=300),  # Start, not A
    ]
    assert _auto_continue_dialog(window, _caps()) is None


def test_dialog_rule_skips_when_surface_does_not_expose_a() -> None:
    """@example: a surface without 'a' in semantic_inputs can't take the action."""

    caps_without_a = SurfaceCapsPayload(
        semantic_inputs=["up", "down", "left", "right"],
        log_schema="weird_surface.v1",
        observation_modalities=["log"],  # type: ignore[list-item]
    )
    window = [
        _ram_event(10, 5, t=100),
        _ram_event(10, 5, t=200),
        _input_ack("a", 250, t=300),
    ]
    assert _auto_continue_dialog(window, caps_without_a) is None


# ── Pack registry ─────────────────────────────────────────────────────


def test_registry_returns_engine_for_known_schema() -> None:
    """@example: the pokemon_rs.v1 schema maps to a non-empty rule engine."""

    engine = rule_engine_for_log_schema("pokemon_rs.v1")
    assert engine is not None
    # Same engine shape as the direct builder.
    direct = build_rule_engine()
    assert len(engine._rules) == len(direct._rules)


def test_registry_returns_none_for_unknown_schema() -> None:
    """@example: schemas without a pack fall through to slow-path-only behavior."""

    assert rule_engine_for_log_schema("does_not_exist.v9") is None
    assert rule_engine_for_log_schema("") is None
