"""navigate_to quickaction: pathfinder, novelty tracker, screen rules,
and the orchestrator's caps injection + macro expansion."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from augmentum.game_agent.control.navigate import (
    compile_navigation,
    has_navigation,
    parse_nav_target,
)
from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.surfaces.mock import MockAdapter


def _open_grid(px: int, py: int, size: int = 15) -> dict:
    half = size // 2
    return {
        "x0": px - half,
        "y0": py - half,
        "rows": ["." * size for _ in range(size)],
    }


def _sems(seq):
    return [a.semantic for a in seq]


# ── target parsing ────────────────────────────────────────────────────


def test_parse_absolute_and_relative_targets() -> None:
    assert parse_nav_target("12,8", 0, 0) == (12, 8)
    assert parse_nav_target(" 12 , 8 ", 0, 0) == (12, 8)
    assert parse_nav_target("down 5", 10, 10) == (10, 15)
    assert parse_nav_target("Left 3", 10, 10) == (7, 10)
    assert parse_nav_target("sideways 2", 0, 0) is None
    assert parse_nav_target("", 0, 0) is None


# ── path compilation ──────────────────────────────────────────────────


def test_straight_line_walk() -> None:
    seq, end = compile_navigation(_open_grid(5, 5), 5, 5, 8, 5)
    assert _sems(seq) == ["nav_right"] * 3
    assert end == (8, 5)


def test_detour_around_wall() -> None:
    grid = _open_grid(5, 5)
    rows = list(grid["rows"])
    # Vertical wall at x=6 (col index 8) with a gap at y=2 (row 4).
    for r in range(len(rows)):
        if r != 4:
            rows[r] = rows[r][:8] + "#" + rows[r][9:]
    grid["rows"] = rows
    seq, end = compile_navigation(grid, 5, 5, 8, 5)
    assert end == (8, 5)
    sems = _sems(seq)
    assert "nav_up" in sems          # went around via the gap
    assert len(sems) > 3             # longer than the blocked beeline


def test_unreachable_target_paths_to_nearest() -> None:
    grid = _open_grid(5, 5)
    rows = list(grid["rows"])
    # Target tile itself is blocked.
    rows[7] = rows[7][:10] + "#" + rows[7][11:]  # (8,5) -> col 10, row 7
    grid["rows"] = rows
    seq, end = compile_navigation(grid, 5, 5, 8, 5)
    assert end != (8, 5)
    assert abs(end[0] - 8) + abs(end[1] - 5) == 1   # adjacent tile
    assert seq


def test_already_there_and_fully_walled() -> None:
    seq, end = compile_navigation(_open_grid(5, 5), 5, 5, 5, 5)
    assert seq == [] and end == (5, 5)
    grid = _open_grid(5, 5)
    grid["rows"] = ["#" * 15 for _ in range(15)]
    seq, end = compile_navigation(grid, 5, 5, 8, 5)
    assert seq == [] and end == (5, 5)


def test_profile_gate() -> None:
    assert has_navigation("pokemon_emerald") is True
    assert has_navigation("pokemon_rs") is False
    assert has_navigation(None) is False


def test_emerald_preset_ships_hidden_walk_grid() -> None:
    from augmentum.game_agent.probes import hidden_probe_names
    from augmentum.game_agent.probes.pokemon_emerald import to_dict

    by_name = {p["name"]: p for p in to_dict()["probes"]}
    wg = by_name["walk_grid"]
    assert wg["type"] == "grid"
    assert wg["hidden"] is True
    assert wg["grid"]["header_at"] == 0x03005DC0
    assert wg["grid"]["anchor_x"] == "player_x"
    assert hidden_probe_names("pokemon_emerald") == frozenset({"walk_grid"})
    assert hidden_probe_names("pokemon_rby") == frozenset()


def test_extract_landmarks_names_reachable_exits() -> None:
    from augmentum.game_agent.control.navigate import (
        extract_landmarks,
        resolve_nav_target,
    )

    # Open room, walls east+west, open north+south edges.
    grid = _open_grid(5, 5)
    rows = list(grid["rows"])
    for r in range(len(rows)):
        rows[r] = "#" + rows[r][1:-1] + "#"
    grid["rows"] = rows
    marks = extract_landmarks(grid, 5, 5)
    assert set(marks) == {"exit_north", "exit_south"}
    ex, ey = marks["exit_north"]
    assert ey == grid["y0"]                      # on the north edge
    assert grid["x0"] < ex < grid["x0"] + 14     # not in a wall corner
    # Symbolic resolution: the same name the delta advertises resolves
    # at action time; unknown names and empty text resolve to None.
    assert resolve_nav_target("exit_north", grid, 5, 5) == marks["exit_north"]
    assert resolve_nav_target("exit_hyrule", grid, 5, 5) is None
    # Coordinate + relative forms still work through the same resolver.
    assert resolve_nav_target("8,5", grid, 5, 5) == (8, 5)
    assert resolve_nav_target("down 2", grid, 5, 5) == (5, 7)


def test_extract_landmarks_ignores_unreachable_edges() -> None:
    from augmentum.game_agent.control.navigate import extract_landmarks

    # Fully walled room: edges exist but nothing is reachable.
    grid = _open_grid(5, 5)
    rows = list(grid["rows"])
    rows[0] = "#" * 15
    rows[-1] = "#" * 15
    for r in range(1, 14):
        rows[r] = "#" + rows[r][1:-1] + "#"
    grid["rows"] = rows
    assert extract_landmarks(grid, 5, 5) == {}


def test_fast_output_schema_enum_locks_actions() -> None:
    from augmentum.game_agent.agent import _fast_output_schema
    from augmentum.game_agent.schema import SurfaceCapsPayload

    caps = SurfaceCapsPayload(
        semantic_inputs=["confirm", "navigate_to"], log_schema="x.v1",
        observation_modalities=["log"],
    )
    schema = _fast_output_schema(caps)
    items = schema["properties"]["a"]["items"]
    assert items["properties"]["s"]["enum"] == ["confirm", "navigate_to"]
    # text is REQUIRED (run-19 lesson: optional text taught the model to
    # emit type_text with no argument) and ordered before duration.
    assert items["required"] == ["s", "text", "d"]
    assert list(items["properties"]) == ["s", "text", "d"]
    assert schema["required"] == ["a", "why", "next_ms", "esc"]


# ── novelty tracker ───────────────────────────────────────────────────


def test_world_novelty_counts_and_stall() -> None:
    from augmentum.game_agent.world import WorldState

    w = WorldState()
    assert w.note("tile", (0, 1, 5, 5), t_ms=1000) == 1    # novel
    assert w.last_novel_ms == 1000
    assert w.note("tile", (0, 1, 5, 5), t_ms=2000) == 1    # still standing there
    assert w.note("tile", (0, 1, 5, 6), t_ms=3000) == 1    # novel again
    assert w.note("tile", (0, 1, 5, 5), t_ms=4000) == 2    # returned: seen x2
    assert w.last_novel_ms == 3000                          # revisit is NOT novel
    # Menu churn: same two screens flipping never counts as novel again.
    w.note("screen", "overworld", t_ms=5000)
    w.note("screen", "bag_menu", t_ms=6000)
    w.note("screen", "overworld", t_ms=7000)
    w.note("screen", "bag_menu", t_ms=8000)
    assert w.last_novel_ms == 6000
    assert w.novelty_stalled_for_ms(50_000) == 44_000
    w.mark_progress(49_000)
    assert w.novelty_stalled_for_ms(50_000) == 1000


def test_fast_delta_carries_loc_and_rule() -> None:
    from augmentum.game_agent.prompt import build_fast_delta

    d = build_fast_delta(
        t_ms=1000, overlay_delta=None, last_actions=[], frame_attached=True,
        loc="seenx4", rule="you are IN A BATTLE: confirm chooses",
    )
    assert "LOC=seenx4" in d
    assert 'RULE="you are IN A BATTLE: confirm chooses"' in d


def test_screen_rule_lookup() -> None:
    from augmentum.game_agent.rule_packs.screen_rules import (
        modal_rule,
        screen_rule,
    )

    assert "BATTLE" in screen_rule("pokemon_emerald", "battle")
    assert screen_rule("pokemon_emerald", "0x812345") == ""
    assert screen_rule("pokemon_emerald", None) == ""
    # Unprofiled games fall back to GENERIC interface physics for the
    # scene-narrator screen vocabulary (game-agnostic tier) …
    assert "BATTLE" in screen_rule("unknown_game", "battle")
    assert "DIALOG" in screen_rule("unknown_game", "dialog")
    # … but only for known screen labels; junk still yields nothing.
    assert screen_rule("unknown_game", "0x812345") == ""
    # Modal rule: per-game entry when present, generic fallback otherwise.
    assert "DIALOG" in modal_rule("pokemon_emerald")
    assert "dialog" in modal_rule("unknown_game")


# ── orchestrator integration ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_injects_and_expands_navigate_to(
    tmp_path: Path,
) -> None:
    sems = (
        "confirm", "cancel", "menu", "registered",
        "nav_up", "nav_down", "nav_left", "nav_right",
    )
    adapter = MockAdapter(script=[], semantic_inputs=sems)
    base = adapter.caps()
    caps = base.model_copy(
        update={"game_profile": "pokemon_emerald", "controller_profile": "gba"}
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
    assert "navigate_to" in orch._caps.semantic_inputs
    assert "navigate_to" in (orch._caps.input_hints or {})
    # walk_grid is hidden: never in the prompt overlay, always in world.
    assert "walk_grid" in orch._hidden_probes

    orch._clock.start()
    from augmentum.game_agent.schema import PlanAction

    # No grid yet: honest error, no presses.
    await orch._action_queue.put(
        (PlanAction(semantic="navigate_to", duration_ms=100, text="8,5"), "agent")
    )
    worker = asyncio.create_task(orch._action_worker())
    await asyncio.sleep(0.2)
    assert adapter.recorded_inputs == []

    # Seed the blackboard the way a ram event would, then navigate.
    orch._world.update_probes(
        {"player_x": 5, "player_y": 5, "walk_grid": _open_grid(5, 5)},
        t_ms=orch._clock.elapsed_ms(),
    )
    await orch._action_queue.put(
        (PlanAction(semantic="navigate_to", duration_ms=100, text="down 2"), "agent")
    )
    await asyncio.sleep(0.3)
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    assert [s for s, _d in adapter.recorded_inputs] == ["nav_down", "nav_down"]


@pytest.mark.asyncio
async def test_hidden_probe_stays_out_of_overlay(tmp_path: Path) -> None:
    adapter = MockAdapter(script=[], semantic_inputs=("confirm",))
    base = adapter.caps()
    caps = base.model_copy(update={"game_profile": "pokemon_emerald"})
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
    orch._clock.start()
    orch._record(
        {
            "t": 100,
            "kind": "event",
            "payload": {
                "channel": "log",
                "data": {
                    "event": "ram",
                    "probes": {
                        "player_x": 5,
                        "player_y": 5,
                        "screen": "overworld",
                        "walk_grid": _open_grid(5, 5),
                    },
                },
            },
        }
    )
    assert "walk_grid" not in orch._overlay
    assert "player_x" in orch._overlay
    assert orch._world.facts["walk_grid"].value["rows"]
    # Novelty fed: first tile + first screen are novel.
    assert orch._tile_seen == 1
    assert orch._world.last_novel_ms == 100


@pytest.mark.asyncio
async def test_modal_text_window_arms_and_disarms(tmp_path: Path) -> None:
    adapter = MockAdapter(script=[], semantic_inputs=("confirm", "nav_up"))
    base = adapter.caps()
    caps = base.model_copy(update={"game_profile": "pokemon_emerald"})
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
    orch._clock.start()

    def ram_event(t, probes):
        return {
            "t": t, "kind": "event",
            "payload": {"channel": "log", "data": {"event": "ram", "probes": probes}},
        }

    # A printing dialog box arms the modal window → context = reading.
    orch._record(ram_event(1000, {"dialog_text": "PROF. BIRCH is in trouble!"}))
    assert orch._context.infer(1500) == "reading"
    # Junk (single glyph) does not arm.
    orch._context.end_text_activity()
    orch._record(ram_event(2000, {"dialog_text": " "}))
    assert orch._context.infer(2100) != "reading"
    # A successful movement press disarms it (nothing modal is open).
    orch._record(ram_event(3000, {"dialog_text": "Hello!"}))
    assert orch._context.infer(3100) == "reading"
    orch._record(
        {
            "t": 3500, "kind": "event",
            "payload": {
                "channel": "log",
                "data": {"event": "input_ack", "button": "nav_up", "effect_score": 200},
            },
        }
    )
    assert orch._context.infer(3600) != "reading"
    # A dead nav press does NOT disarm (that's the swallowed case).
    orch._record(ram_event(4000, {"battle_text": "Go! TORCHIC!"}))
    orch._record(
        {
            "t": 4200, "kind": "event",
            "payload": {
                "channel": "log",
                "data": {"event": "input_ack", "button": "nav_up", "effect_score": 3},
            },
        }
    )
    assert orch._context.infer(4300) == "reading"


# ── input-context inference (game-agnostic) ───────────────────────────


def test_context_tracker_classifies_all_four_modes() -> None:
    from augmentum.game_agent.context import InputContextTracker

    # reading: text activity wins over everything.
    t = InputContextTracker()
    t.feed_fx("nav_up", 200, 900)
    t.feed_text_activity(1000)
    assert t.infer(1500) == "reading"

    # free_move: position facts moving.
    t = InputContextTracker()
    t.feed_position_change(1000)
    assert t.infer(2000) == "free_move"
    # ...but stale motion no longer counts.
    assert t.infer(9000) != "free_move"

    # cursor: presses register (screen reacts) but position never moves —
    # a battle, a bag, a shop, a naming grid: all the same context.
    t = InputContextTracker()
    t.feed_fx("nav_down", 150, 1000)
    t.feed_fx("confirm", 220, 1500)
    assert t.infer(2000) == "cursor"

    # locked: several recent presses, all dead.
    t = InputContextTracker()
    for i, b in enumerate(("confirm", "nav_up", "menu")):
        t.feed_fx(b, 5, 1000 + i * 200)
    assert t.infer(2000) == "locked"

    # unknown: no evidence yet.
    assert InputContextTracker().infer(1000) == ""


def test_context_mode_lines_are_mechanics_named() -> None:
    from augmentum.game_agent.context import MODE_LINES

    for text in MODE_LINES.values():
        # Agnostic discipline: mechanics language only, never game terms.
        assert "pokemon" not in text.lower()
        assert "emerald" not in text.lower()
    assert "cursor" in MODE_LINES["cursor"].lower()
    assert '"a":[]' in MODE_LINES["locked"]


def test_fast_delta_carries_mode() -> None:
    from augmentum.game_agent.prompt import build_fast_delta

    d = build_fast_delta(
        t_ms=1000, overlay_delta=None, last_actions=[], frame_attached=True,
        mode="READING: text is being presented",
    )
    assert 'MODE="READING: text is being presented"' in d


def test_fast_delta_screen_is_always_on() -> None:
    from augmentum.game_agent.prompt import build_fast_delta

    # SCREEN rides every turn even when NOTHING changed (empty delta) —
    # the model never has to remember the screen from old turns.
    d = build_fast_delta(
        t_ms=1000, overlay_delta=None, last_actions=[], frame_attached=True,
        screen="overworld",
    )
    assert "SCREEN=overworld" in d
    assert "DELTA={}" in d


def test_fresh_window_triggers_full_state_resend() -> None:
    from augmentum.game_agent.agent import FastTurnRunner
    from augmentum.game_agent.schema import SurfaceCapsPayload

    caps = SurfaceCapsPayload(
        semantic_inputs=["confirm"], log_schema="x.v1",
        observation_modalities=["log"],
    )

    async def _chat(_msgs, _options=None):
        return {"text": '{"a":[],"why":"","next_ms":500,"esc":false}'}

    runner = FastTurnRunner(chat_llm=_chat, caps=caps, objective="x")
    # Fresh at construction, fresh again after every reset.
    assert runner.fresh_window() is True
    runner.reset()
    assert runner.fresh_window() is True
    # After a successful turn the window holds the exchange.
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        runner.turn(
            t_ms=100, overlay_delta={"hp": 20}, last_actions=[], frame=None,
        )
    )
    assert runner.fresh_window() is False
    runner.reset()
    assert runner.fresh_window() is True


@pytest.mark.asyncio
async def test_stale_decision_is_dropped(tmp_path: Path) -> None:
    from augmentum.game_agent.prompt import FastPlan
    from augmentum.game_agent.schema import PlanAction

    adapter = MockAdapter(script=[], semantic_inputs=("confirm", "cancel"))
    base = adapter.caps()
    caps = base.model_copy(update={"game_profile": "pokemon_emerald"})
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
    orch._clock.start()
    plan = FastPlan(
        actions=[PlanAction(semantic="cancel", duration_ms=100)],
        why="closing the bag",
        next_check_in_ms=1000,
    )
    # Screen unchanged since capture → plan passes through untouched.
    orch._world.update("screen", "bag_menu", source="ram", t_ms=100)
    same = orch._invalidate_if_stale(plan, screen_before="bag_menu")
    assert same.actions and same.actions[0].semantic == "cancel"
    # Screen moved on mid-think → actions dropped, quick re-look.
    orch._world.update("screen", "overworld", source="ram", t_ms=200)
    dropped = orch._invalidate_if_stale(plan, screen_before="bag_menu")
    assert dropped.actions == []
    assert "mid-think" in dropped.why
    assert dropped.next_check_in_ms == 250
