"""Quickaction tests: type_text compiler, screen probe labels, and the
orchestrator's macro expansion + caps injection."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from augmentum.game_agent.control.text_entry import (
    LAYOUTS,
    compile_text_entry,
    has_text_entry,
)
from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.surfaces.mock import MockAdapter


def _sems(seq):
    return [a.semantic for a in seq]


def test_compile_first_char_of_start_page() -> None:
    # 'A' is at the cursor's start cell: just confirm, then START+A (OK).
    seq = compile_text_entry("pokemon_rs", "A")
    assert _sems(seq) == ["confirm", "menu", "confirm"]


def test_compile_walks_grid_and_tracks_cursor() -> None:
    # 'B' is one right of 'A'; then 'A' again means one left back.
    seq = compile_text_entry("pokemon_rs", "BA")
    assert _sems(seq) == [
        "nav_right", "confirm",          # B
        "nav_left", "confirm",           # back to A
        "menu", "confirm",               # OK
    ]


def test_compile_page_switch_for_lowercase_and_digits() -> None:
    seq = compile_text_entry("pokemon_rs", "a")
    # UPPER -> lower is one SELECT (registered); cursor stays at (0,0)='a'.
    assert _sems(seq) == ["registered", "confirm", "menu", "confirm"]

    seq2 = compile_text_entry("pokemon_rs", "5")
    # UPPER -> symbols is two SELECTs; '5' is at row 1 col 0.
    assert _sems(seq2) == [
        "registered", "registered", "nav_down", "confirm", "menu", "confirm",
    ]


def test_compile_skips_unknown_chars_and_caps_length() -> None:
    seq = compile_text_entry("pokemon_rs", "@")  # not on any page
    assert _sems(seq) == ["menu", "confirm"]     # degenerate: accept + OK
    assert compile_text_entry("unknown_game", "MAY") == []
    assert has_text_entry("pokemon_rs") is True
    assert has_text_entry("") is False


def test_layout_shared_across_gen3_profiles() -> None:
    assert LAYOUTS["pokemon_rs"] is LAYOUTS["pokemon_emerald"]


def test_emerald_preset_has_screen_probe_with_labels() -> None:
    from augmentum.game_agent.probes.pokemon_emerald import to_dict

    by_name = {p["name"]: p for p in to_dict()["probes"]}
    screen = by_name["screen"]
    assert screen["address"] == 0x030022C4
    assert screen["type"] == "u32le"
    # JSON-safe: keys stringified for the JS bridge.
    assert screen["value_labels"][str(0x08085E5C)] == "overworld"
    assert screen["value_labels"][str(0x080E2E04)] == "naming_screen"


@pytest.mark.asyncio
async def test_orchestrator_injects_type_text_and_expands_macro(
    tmp_path: Path,
) -> None:
    sems = (
        "confirm", "cancel", "menu", "registered",
        "nav_up", "nav_down", "nav_left", "nav_right",
    )
    adapter = MockAdapter(script=[], semantic_inputs=sems)
    base = adapter.caps()
    caps = base.model_copy(
        update={"game_profile": "pokemon_rs", "controller_profile": "gba"}
    )
    adapter.caps = lambda: caps  # type: ignore[method-assign]

    async def _llm(_p, _f):
        return ""

    orch = Orchestrator(
        log_path=str(tmp_path / "s.ndjson"),
        surface_kind="mock",
        adapter=adapter,
        llm=_llm,
        objective="x",
    )
    # caps injection: type_text allowed + hinted.
    assert "type_text" in orch._caps.semantic_inputs
    assert "type_text" in (orch._caps.input_hints or {})

    # Macro expansion: enqueue type_text, run the worker briefly, and
    # check the primitives reached the resolver in order.
    orch._clock.start()
    from augmentum.game_agent.schema import PlanAction

    await orch._action_queue.put(
        (PlanAction(semantic="type_text", duration_ms=100, text="A"), "agent")
    )
    worker = asyncio.create_task(orch._action_worker())
    await asyncio.sleep(0.3)
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    assert [s for s, _d in adapter.recorded_inputs] == ["confirm", "menu", "confirm"]


def test_playbook_renders_in_both_prompts_and_plan_field_parses() -> None:
    from augmentum.game_agent.prompt import (
        build_fast_system_prompt,
        build_full_prompt,
        parse_plan_output,
    )
    from augmentum.game_agent.schema import SurfaceCapsPayload

    pb = {"notes": ["dialogs swallow movement; close box first"]}
    caps = SurfaceCapsPayload(
        semantic_inputs=["confirm"], log_schema="x.v1",
        observation_modalities=["log"],
    )
    full = build_full_prompt(
        companion=False, surface_kind="emulatorjs", caps=caps,
        objective="win", state="", live_log_tail=[], playbook=pb,
    )
    assert "PLAYBOOK" in full and "swallow movement" in full
    fast = build_fast_system_prompt(caps=caps, objective="win", playbook=pb)
    assert "swallow movement" in fast

    plan = parse_plan_output(
        '{"observations":[],"state_update":"","actions":[],"confidence":0.5,'
        '"next_check_in_ms":500,'
        '"playbook_update":{"notes_append":["grass = wild encounters"]}}',
        caps,
    )
    assert plan.playbook_update == {"notes_append": ["grass = wild encounters"]}


# ── world blackboard + goal stack ─────────────────────────────────────


def test_world_provenance_and_goal_metrics() -> None:
    from augmentum.game_agent.world import WorldState

    w = WorldState()
    w.update_probes({"map_num": 40, "party_count": 0}, t_ms=1000)
    # Scene can't clobber fresh RAM truth...
    assert w.update("map_num", 99, source="scene", t_ms=2000) is False
    assert w.facts["map_num"].value == 40
    # ...but can fill fields RAM never wrote.
    assert w.update("scene_feed", "truck interior", source="scene", t_ms=2000)

    w.apply_goal_update(
        {
            "final": "complete the game",
            "short": {"text": "exit the truck",
                      "metric": {"probe": "map_num", "op": "ne", "value": 40}},
        },
        t_ms=2500,
    )
    assert "SHORT: exit the truck" in w.goals_line()
    assert w.check_goals() == []          # still in the truck
    w.update_probes({"map_num": 1}, t_ms=9000)
    assert w.check_goals() == ["short"]   # metric self-completed
    assert "DONE exit the truck" in w.goals_line()
    # Stall pulse follows the last change.
    assert w.stalled_for_ms(10_000) == 1000


def test_fast_delta_carries_goals_and_stall() -> None:
    from augmentum.game_agent.prompt import build_fast_delta

    d = build_fast_delta(
        t_ms=1000, overlay_delta=None, last_actions=[], frame_attached=True,
        goals="FINAL: beat game | SHORT: exit truck", stalled_s=52,
    )
    assert "GOALS[FINAL: beat game | SHORT: exit truck]" in d
    assert "STALLED=52s" in d
