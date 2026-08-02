"""Reflex tier tests: spec compiler, engine replace/retract, orchestrator
integration, and fast-delta visibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.reflex import compile_reflex_rule
from augmentum.game_agent.rules import RuleEngine
from augmentum.game_agent.schema import PlanPayload, SurfaceCapsPayload
from augmentum.game_agent.surfaces.mock import MockAdapter

CAPS = SurfaceCapsPayload(
    semantic_inputs=["confirm", "cancel", "nav_up"],
    log_schema="mock.v1",
    observation_modalities=["log"],
)


def _ram_event(probes: dict) -> dict:
    return {
        "t": 0,
        "kind": "event",
        "payload": {"channel": "log", "data": {"event": "ram", "probes": probes}},
    }


def _spec(**over):
    spec = {
        "id": "advance-dialog",
        "when": {"probe": "dialog_text", "changed": True, "not_contains": ["?"]},
        "do": [{"s": "confirm", "d": 120}],
        "cooldown_ms": 300,
        "ttl_fires": 3,
    }
    spec.update(over)
    return spec


def test_reflex_fires_on_changed_text_and_respects_not_contains() -> None:
    rule = compile_reflex_rule(_spec())
    w = [_ram_event({"dialog_text": "Hello there!"})]
    m = rule.predicate(w, CAPS)
    assert m is not None and m.actions[0].semantic == "confirm"

    # Same value again -> not a change -> no fire.
    assert rule.predicate(w, CAPS) is None

    # A question is excluded by not_contains.
    w2 = w + [_ram_event({"dialog_text": "Are you a boy? Or a girl?"})]
    assert rule.predicate(w2, CAPS) is None

    # New statement fires again.
    w3 = w2 + [_ram_event({"dialog_text": "Welcome to LITTLEROOT."})]
    assert rule.predicate(w3, CAPS) is not None


def test_reflex_ignores_empty_values_and_honors_ttl() -> None:
    rule = compile_reflex_rule(_spec(ttl_fires=2))
    # Dialog closing (empty) must not fire the advance reflex.
    assert rule.predicate([_ram_event({"dialog_text": ""})], CAPS) is None

    w = [_ram_event({"dialog_text": "line one."})]
    assert rule.predicate(w, CAPS) is not None
    w.append(_ram_event({"dialog_text": "line two."}))
    assert rule.predicate(w, CAPS) is not None
    w.append(_ram_event({"dialog_text": "line three."}))
    assert rule.predicate(w, CAPS) is None  # TTL of 2 exhausted


def test_reflex_equals_and_contains() -> None:
    rule = compile_reflex_rule(
        {
            "id": "flee-on-low-hp",
            "when": {"probe": "p1_hp_current", "equals": 1},
            "do": [{"s": "cancel"}],
        }
    )
    assert rule.predicate([_ram_event({"p1_hp_current": 5})], CAPS) is None
    assert rule.predicate([_ram_event({"p1_hp_current": 1})], CAPS) is not None

    rule2 = compile_reflex_rule(
        {
            "id": "battle-start",
            "when": {"probe": "battle_text", "contains": ["appeared"]},
            "do": [{"s": "confirm"}],
        }
    )
    assert (
        rule2.predicate([_ram_event({"battle_text": "A wild POOCHYENA appeared!"})], CAPS)
        is not None
    )


def test_reflex_rejects_bad_specs() -> None:
    with pytest.raises(ValueError):
        compile_reflex_rule({"id": "", "when": {"probe": "x", "changed": True}, "do": []})
    with pytest.raises(ValueError):
        compile_reflex_rule({"id": "a", "do": [{"s": "confirm"}]})  # no when
    with pytest.raises(ValueError):
        compile_reflex_rule({"id": "a", "when": {"probe": "x"}, "do": [{"s": "confirm"}]})
    with pytest.raises(ValueError):
        compile_reflex_rule({"id": "a", "when": {"probe": "x", "changed": True}, "do": []})


def test_engine_register_replaces_and_unregisters() -> None:
    eng = RuleEngine()
    eng.register(compile_reflex_rule(_spec()))
    eng.register(compile_reflex_rule(_spec()))  # same id -> replace
    assert eng.rule_ids.count("advance-dialog") == 1
    assert eng.unregister("advance-dialog") is True
    assert eng.rule_ids == []


def _orch(tmp_path: Path) -> Orchestrator:
    async def _llm(_p, _f):
        return ""

    orch = Orchestrator(
        log_path=str(tmp_path / "s.ndjson"),
        surface_kind="mock",
        adapter=MockAdapter(script=[]),
        llm=_llm,
        objective="x",
    )
    orch._clock.start()  # methods under test log against the clock
    return orch


@pytest.mark.asyncio
async def test_orchestrator_applies_and_retracts_reflex_rules(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    plan = PlanPayload(
        confidence=0.5,
        next_check_in_ms=500,
        reflex_rules=[_spec()],
    )
    await orch._integrate_plan(plan, 10.0)
    assert "advance-dialog" in orch._rules.rule_ids
    assert "advance-dialog" in orch._reflex_ids

    plan2 = PlanPayload(
        confidence=0.5,
        next_check_in_ms=500,
        reflex_rules=[{"id": "advance-dialog", "retract": True}],
    )
    await orch._integrate_plan(plan2, 10.0)
    assert "advance-dialog" not in orch._rules.rule_ids
    assert "advance-dialog" not in orch._reflex_ids


@pytest.mark.asyncio
async def test_reflex_fires_end_to_end_and_reaches_fast_delta(tmp_path: Path) -> None:
    """RAM event -> rule fires -> input queued (source=rule) -> semantics
    surfaced to the next fast turn as reflex_did."""

    from augmentum.game_agent.schema import EventPayload

    orch = _orch(tmp_path)
    await orch._integrate_plan(
        PlanPayload(confidence=0.5, next_check_in_ms=500, reflex_rules=[_spec()]),
        10.0,
    )
    await orch._emit_event(
        EventPayload(
            channel="log", data={"event": "ram", "probes": {"dialog_text": "Go north!"}}
        )
    )
    assert orch._recent_reflex == ["confirm"]
    assert orch._action_queue.qsize() == 1

    # The fast delta carries it (and the buffer clears after consumption).
    from augmentum.game_agent.prompt import build_fast_delta

    delta = build_fast_delta(
        t_ms=1000, overlay_delta=None, last_actions=[],
        frame_attached=False, reflex_actions=orch._recent_reflex,
    )
    assert "reflex_did=confirm" in delta


# ── per-button effect feedback + keyframes ────────────────────────────


@pytest.mark.asyncio
async def test_input_ack_effect_scores_collected_and_rendered(tmp_path: Path) -> None:
    from augmentum.game_agent.prompt import build_fast_delta
    from augmentum.game_agent.schema import EventPayload

    orch = _orch(tmp_path)
    await orch._emit_event(
        EventPayload(
            channel="log",
            data={"event": "input_ack", "button": "confirm", "effect_score": 2724},
        )
    )
    await orch._emit_event(
        EventPayload(
            channel="log",
            data={"event": "input_ack", "button": "menu", "effect_score": 0},
        )
    )
    assert orch._recent_fx == [("confirm", 2724), ("menu", 0)]

    delta = build_fast_delta(
        t_ms=1000, overlay_delta=None, last_actions=["confirm", "menu"],
        frame_attached=False, fx=orch._recent_fx,
    )
    assert "fx=confirm:2724,menu:0" in delta


def test_keyframe_sample_picks_oldest_and_middle() -> None:
    assert Orchestrator._keyframe_sample([]) == []
    assert Orchestrator._keyframe_sample([b"a"]) == [b"a"]
    assert Orchestrator._keyframe_sample([b"a", b"b"]) == [b"a"]
    assert Orchestrator._keyframe_sample(
        [b"a", b"b", b"c", b"d", b"e"]
    ) == [b"a", b"c"]


# ── scene narrator + universal dead-nav interceptor ───────────────────


def _ack_event(button: str, score: int) -> dict:
    return {
        "t": 0,
        "kind": "event",
        "payload": {
            "channel": "log",
            "data": {"event": "input_ack", "button": button, "effect_score": score,
                     "request_id": f"r-{button}-{score}"},
        },
    }


def test_dead_nav_during_dialog_fires_confirm_once() -> None:
    from augmentum.game_agent.rule_packs.universal import (
        dead_nav_during_dialog_rule,
    )

    rule = dead_nav_during_dialog_rule()
    w = [
        _ram_event({"dialog_text": "The box is printed with a logo."}),
        _ack_event("nav_right", 3),
    ]
    m = rule.predicate(w, CAPS)
    assert m is not None
    assert [a.semantic for a in m.actions] == ["confirm"]
    # Same dead press again -> already handled, no re-fire.
    assert rule.predicate(w, CAPS) is None
    # Effective press -> no fire.
    w2 = [
        _ram_event({"dialog_text": "More text."}),
        _ack_event("nav_left", 900),
    ]
    assert rule.predicate(w2, CAPS) is None
    # No dialog open -> movement legitimately failed, planner's problem.
    w3 = [
        _ram_event({"dialog_text": ""}),
        _ack_event("nav_up", 2),
    ]
    assert rule.predicate(w3, CAPS) is None


@pytest.mark.asyncio
async def test_scene_tick_gates_on_fingerprint_and_updates_scene(
    tmp_path: Path,
) -> None:
    import io

    from PIL import Image

    def png(color):
        buf = io.BytesIO()
        Image.new("RGB", (240, 160), color).save(buf, "PNG")
        return buf.getvalue()

    frames = [png((200, 30, 30))]
    calls = []

    async def chat(messages, options=None):  # noqa: ANN001, ARG001
        calls.append(messages)
        return {"text": "Interior of a moving truck; exit on the right.",
                "latency_ms": 5.0}

    async def _llm(_p, _f):
        return ""

    adapter = MockAdapter(script=[], observation_modalities=("log", "frame"))

    async def snapshot_frames(n=1):  # noqa: ANN001
        return [frames[-1]]

    adapter.snapshot_frames = snapshot_frames  # type: ignore[attr-defined]
    orch = Orchestrator(
        log_path=str(tmp_path / "s.ndjson"),
        surface_kind="mock",
        adapter=adapter,
        llm=_llm,
        objective="x",
        fast_llm=chat,
    )
    orch._clock.start()

    fp1 = await orch._scene_tick(None)
    assert orch._scene.startswith("Interior of a moving truck")
    assert len(calls) == 1
    assert calls[0][0]["role"] == "system"
    assert calls[0][-1]["images"]

    # Same frame -> gated, no second call.
    fp2 = await orch._scene_tick(fp1)
    assert len(calls) == 1
    assert fp2 == fp1

    # Different frame -> narrates again, with PREVIOUS context.
    frames.append(png((30, 200, 30)))
    await orch._scene_tick(fp2)
    assert len(calls) == 2
    assert "PREVIOUS:" in calls[1][-1]["content"]
