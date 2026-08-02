"""Fast-turn ("call mode") loop tests.

Covers the micro-plan contract (:func:`parse_fast_output`), the rolling
window discipline (:class:`FastTurnRunner`), and the orchestrator's
full/fast alternation + escalation + timing telemetry.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from augmentum.game_agent.agent import FastTurnRunner
from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.perception import downscale_frame, prepare_frames
from augmentum.game_agent.prompt import (
    PlanParseError,
    build_fast_system_prompt,
    parse_fast_output,
)
from augmentum.game_agent.schema import SurfaceCapsPayload
from augmentum.game_agent.surfaces.mock import MockAdapter

CAPS = SurfaceCapsPayload(
    semantic_inputs=["advance", "confirm", "cancel"],
    log_schema="mock.v1",
    observation_modalities=["log"],
)


# ── parse_fast_output ─────────────────────────────────────────────────


def test_parse_fast_valid_micro_plan() -> None:
    plan = parse_fast_output(
        '{"a":[{"s":"confirm","d":120}],"why":"advance dialog","next_ms":800,"esc":false}',
        CAPS,
    )
    assert [a.semantic for a in plan.actions] == ["confirm"]
    assert plan.actions[0].duration_ms == 120
    assert plan.why == "advance dialog"
    assert plan.next_check_in_ms == 800
    assert plan.escalate is False


def test_parse_fast_tolerates_fences_and_prose() -> None:
    raw = 'Sure!\n```json\n{"a":[],"why":"waiting","next_ms":2000,"esc":true}\n```'
    plan = parse_fast_output(raw, CAPS)
    assert plan.actions == []
    assert plan.escalate is True


def test_parse_fast_rejects_unknown_semantic() -> None:
    with pytest.raises(PlanParseError):
        parse_fast_output('{"a":[{"s":"fly","d":100}]}', CAPS)


def test_parse_fast_clamps_and_defaults() -> None:
    plan = parse_fast_output(
        '{"a":[{"s":"advance","d":99999}],"next_ms":1}', CAPS
    )
    assert plan.actions[0].duration_ms == 2000
    assert plan.next_check_in_ms == 50
    assert plan.why == ""
    assert plan.escalate is False


def test_parse_fast_rejects_non_json() -> None:
    with pytest.raises(PlanParseError):
        parse_fast_output("mash A repeatedly", CAPS)


# ── system prompt ─────────────────────────────────────────────────────


def test_fast_system_prompt_carries_contract_and_context() -> None:
    text = build_fast_system_prompt(
        caps=CAPS,
        objective="win the first battle",
        state="heading north",
        journal={"status": "in truck"},
    )
    assert '"confirm"' in text
    assert "win the first battle" in text
    assert '"status":"in truck"' in text.replace(" ", "") or "in truck" in text
    assert "heading north" in text
    assert '"esc"' in text  # the output contract is spelled out


# ── FastTurnRunner window discipline ──────────────────────────────────


@pytest.mark.asyncio
async def test_runner_window_appends_on_success_only() -> None:
    replies = [
        '{"a":[{"s":"confirm","d":100}],"why":"ok","next_ms":500,"esc":false}',
        "garbage not json",
        '{"a":[],"why":"wait","next_ms":500,"esc":false}',
    ]
    seen: list[list[dict]] = []

    async def chat(messages, options=None):  # noqa: ANN001, ARG001
        seen.append(list(messages))
        return {"text": replies.pop(0), "latency_ms": 5.0}

    runner = FastTurnRunner(chat_llm=chat, caps=CAPS, objective="obj")
    await runner.turn(t_ms=1000, overlay_delta=None, last_actions=[], frame=None)
    assert len(runner._window) == 2  # user + assistant

    with pytest.raises(PlanParseError):
        await runner.turn(t_ms=2000, overlay_delta=None, last_actions=[], frame=None)
    assert len(runner._window) == 2  # garbage did not poison the window

    await runner.turn(t_ms=3000, overlay_delta={"x": 5}, last_actions=["confirm"], frame=b"png")
    assert len(runner._window) == 4

    # Message shape: system first, then the window, then the new user turn.
    last_call = seen[-1]
    assert last_call[0]["role"] == "system"
    assert last_call[-1]["role"] == "user"
    assert last_call[-1]["images"] == [b"png"]
    assert "DELTA=" in last_call[-1]["content"]
    assert "did=confirm" in last_call[-1]["content"]


@pytest.mark.asyncio
async def test_runner_reset_clears_window_and_rebuilds_prefix() -> None:
    async def chat(messages, options=None):  # noqa: ANN001, ARG001
        return {"text": '{"a":[],"why":"","next_ms":500,"esc":false}'}

    runner = FastTurnRunner(chat_llm=chat, caps=CAPS, objective="obj")
    await runner.turn(t_ms=100, overlay_delta=None, last_actions=[], frame=None)
    assert runner._window
    runner.reset(journal={"progress": "beat rival"}, state="scratch")
    assert runner._window == []
    assert "beat rival" in runner._system
    assert "scratch" in runner._system


# ── perception downscale ──────────────────────────────────────────────


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_downscale_caps_longest_edge_preserving_aspect() -> None:
    out = downscale_frame(_png(1440, 960), max_edge=480)
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (480, 320)


def test_downscale_noop_when_small_or_disabled() -> None:
    small = _png(240, 160)
    assert downscale_frame(small, max_edge=480) is small
    big = _png(1440, 960)
    assert downscale_frame(big, max_edge=0) is big


def test_prepare_frames_applies_max_edge() -> None:
    prepared = prepare_frames([_png(1440, 960)], dedup=False, grid=False, max_edge=480)
    with Image.open(io.BytesIO(prepared.frames[0])) as im:
        assert max(im.size) == 480


# ── orchestrator alternation ──────────────────────────────────────────


def _full_reply() -> str:
    return json.dumps(
        {
            "observations": ["full turn"],
            "state_update": "strategic notes",
            "actions": [{"semantic": "advance", "duration_ms": 100}],
            "confidence": 0.5,
            "next_check_in_ms": 200,
        }
    )


@pytest.mark.asyncio
async def test_orchestrator_alternates_full_and_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First turn FULL, then fast turns on the rolling window, with
    llm_timing telemetry for both lanes."""

    from augmentum.config import settings

    monkeypatch.setattr(settings, "game_agent_fast_turns_enabled", True, raising=False)
    monkeypatch.setattr(settings, "game_agent_full_turn_every", 8, raising=False)

    full_calls = 0
    fast_calls = 0

    async def full_llm(_prompt: str, _frames: list[bytes]) -> str:
        nonlocal full_calls
        full_calls += 1
        return _full_reply()

    async def fast_llm(messages, options=None):  # noqa: ANN001, ARG001
        nonlocal fast_calls
        fast_calls += 1
        return {
            "text": '{"a":[{"s":"confirm","d":80}],"why":"keep going","next_ms":200,"esc":false}',
            "latency_ms": 7.5,
            "tok_s": 180.0,
            "cached_tokens": 900,
            "completion_tokens": 25,
        }

    log_path = tmp_path / "session.ndjson"
    orch = Orchestrator(
        log_path=str(log_path),
        surface_kind="mock",
        adapter=MockAdapter(script=[]),
        llm=full_llm,
        objective="alternate",
        fast_llm=fast_llm,
    )

    async def stopper() -> None:
        await asyncio.sleep(1.1)
        orch.stop("completed")

    stop_task = asyncio.create_task(stopper())
    await orch.run()
    await stop_task

    assert full_calls == 1, "one grounding FULL turn"
    assert fast_calls >= 2, "fast turns carried the cadence"

    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    plans = [e for e in lines if e["kind"] == "plan"]
    fast_plans = [p for p in plans if p["payload"]["observations"][0].startswith("[fast]")]
    assert fast_plans, "fast plans mirrored into the log"
    timings = [
        e for e in lines
        if e["kind"] == "event" and e["payload"]["data"].get("event") == "llm_timing"
    ]
    turns = {t["payload"]["data"]["turn"] for t in timings}
    assert turns == {"full", "fast"}
    fast_timing = next(t for t in timings if t["payload"]["data"]["turn"] == "fast")
    assert fast_timing["payload"]["data"]["tok_s"] == 180.0


@pytest.mark.asyncio
async def test_orchestrator_escalates_on_esc_and_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fast turn that asks to escalate (or fails to parse) forces the
    next turn to be FULL."""

    from augmentum.config import settings

    monkeypatch.setattr(settings, "game_agent_fast_turns_enabled", True, raising=False)
    monkeypatch.setattr(settings, "game_agent_full_turn_every", 8, raising=False)

    full_calls = 0
    fast_replies = [
        '{"a":[],"why":"lost, need a plan","next_ms":200,"esc":true}',
        "not json",
        '{"a":[],"why":"ok","next_ms":200,"esc":false}',
    ]

    async def full_llm(_prompt: str, _frames: list[bytes]) -> str:
        nonlocal full_calls
        full_calls += 1
        return _full_reply()

    async def fast_llm(messages, options=None):  # noqa: ANN001, ARG001
        text = fast_replies.pop(0) if fast_replies else '{"a":[],"next_ms":200}'
        return {"text": text}

    log_path = tmp_path / "session.ndjson"
    orch = Orchestrator(
        log_path=str(log_path),
        surface_kind="mock",
        adapter=MockAdapter(script=[]),
        llm=full_llm,
        objective="escalate",
        fast_llm=fast_llm,
    )

    async def stopper() -> None:
        await asyncio.sleep(1.6)
        orch.stop("completed")

    stop_task = asyncio.create_task(stopper())
    await orch.run()
    await stop_task

    # Async semantics: plan #1 is already in flight when the first fast
    # turn escalates (no stacking — esc only zeroes the budget), then the
    # parse error forces plan #2. At least 2 full turns inside 1.6s.
    assert full_calls >= 2

    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    errors = [e for e in lines if e["kind"] == "agent_error"]
    assert any("fast-turn parse error" in e["payload"]["message"] for e in errors)


# ── dialogue lore ─────────────────────────────────────────────────────


def test_emerald_preset_carries_text_probes() -> None:
    from augmentum.game_agent.probes.pokemon_emerald import to_dict

    d = to_dict()
    by_name = {p["name"]: p for p in d["probes"]}
    dialog = by_name["dialog_text"]
    assert dialog["type"] == "text"
    assert dialog["charmap"] == "gen3"
    assert dialog["address"] == 0x02021FC4
    assert by_name["battle_text"]["address"] == 0x02022E2C


def test_orchestrator_collects_lore_from_text_probes(tmp_path: Path) -> None:
    orch = Orchestrator(
        log_path=str(tmp_path / "s.ndjson"),
        surface_kind="mock",
        adapter=MockAdapter(script=[]),
        llm=lambda p, f: None,  # never called in this test
        objective="x",
    )
    orch._collect_lore({"dialog_text": "Hello! Welcome to the world", "hp": 20})
    orch._collect_lore({"dialog_text": "Hello! Welcome to the world of POKEMON!"})
    orch._collect_lore({"dialog_text": "Hello! Welcome to the world of POKEMON!"})
    orch._collect_lore({"dialog_text": "short"})  # < 8 chars → ignored
    orch._collect_lore({"battle_text": "A wild ZIGZAGOON appeared!"})
    # Typewriter growth collapsed into one line; ints never lore.
    assert orch._lore == [
        "Hello! Welcome to the world of POKEMON!",
        "A wild ZIGZAGOON appeared!",
    ]


def test_prompts_render_lore_and_patterns() -> None:
    from augmentum.game_agent.prompt import build_full_prompt

    lore = ["PROF BIRCH: press the A Button to advance!"]
    full = build_full_prompt(
        companion=False, surface_kind="emulatorjs", caps=CAPS,
        objective="win", state="", live_log_tail=[], lore=lore,
    )
    assert "DIALOGUE_LORE" in full
    assert "press the A Button" in full
    assert "GAME PATTERNS" in full
    fast = build_fast_system_prompt(caps=CAPS, objective="win", lore=lore)
    assert "press the A Button" in fast
    assert "GAME PATTERNS" in fast


@pytest.mark.asyncio
async def test_fast_lane_runs_while_planner_thinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FULL turn is a background task: fast turns keep acting during
    the planner's think time instead of blocking on it."""

    import time as _time

    from augmentum.config import settings

    monkeypatch.setattr(settings, "game_agent_fast_turns_enabled", True, raising=False)
    monkeypatch.setattr(settings, "game_agent_full_turn_every", 8, raising=False)

    window: dict = {"start": None, "end": None}
    fast_times: list[float] = []

    async def slow_planner(_prompt: str, _frames: list[bytes]) -> str:
        window["start"] = _time.monotonic()
        await asyncio.sleep(0.6)  # a "thinking" planner
        window["end"] = _time.monotonic()
        return _full_reply()

    async def fast_llm(messages, options=None):  # noqa: ANN001, ARG001
        fast_times.append(_time.monotonic())
        return {"text": '{"a":[{"s":"confirm","d":80}],"why":"go","next_ms":200,"esc":false}'}

    orch = Orchestrator(
        log_path=str(tmp_path / "s.ndjson"),
        surface_kind="mock",
        adapter=MockAdapter(script=[]),
        llm=slow_planner,
        objective="overlap",
        fast_llm=fast_llm,
    )

    async def stopper() -> None:
        await asyncio.sleep(1.2)
        orch.stop("completed")

    stop_task = asyncio.create_task(stopper())
    await orch.run()
    await stop_task

    assert window["start"] is not None and window["end"] is not None
    overlapping = [t for t in fast_times if window["start"] < t < window["end"]]
    assert len(overlapping) >= 2, (
        f"expected fast turns during the 0.6s planning window, got "
        f"{len(overlapping)} (fast lane blocked on the planner)"
    )
