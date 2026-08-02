"""Tests for the foundry event bus + loop event emission."""
from __future__ import annotations

import asyncio

import pytest

from augmentum.coder.foundry.contract import GameBuildSpec
from augmentum.coder.foundry.events import FoundryEventBus
from augmentum.coder.foundry.loop import run_foundry

_FILES = {"index.html": "<canvas></canvas>"}


def _progress(score):
    return {"score": score, "reached_play": True, "inputs_acked": 20,
            "inputs_effective": 15, "effective_input_ratio": 0.75,
            "goals_completed": 1, "duration_ms": 90000,
            "score_per_min": score / 1.5}


def _spec(dimension="2d"):
    return GameBuildSpec(slug="g", title="G", concept="c",
                         objective="win", dimension=dimension)


@pytest.mark.asyncio
async def test_bus_backlog_then_live():
    bus = FoundryEventBus()
    bus.emit("run_start", passes=2)
    seen = []

    async def consume():
        async for ev in bus.subscribe():
            seen.append(ev["type"])

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let subscribe register + drain backlog
    bus.emit("pass_start", index=1)
    await asyncio.sleep(0)
    bus.close()
    await asyncio.wait_for(task, timeout=2)
    assert seen[0] == "run_start"      # backlog first
    assert "pass_start" in seen        # then live


@pytest.mark.asyncio
async def test_late_subscriber_gets_backlog():
    bus = FoundryEventBus()
    bus.emit("run_start", passes=1)
    bus.emit("done", improved=True)
    bus.close()
    seen = [ev["type"] async for ev in bus.subscribe()]
    assert seen == ["run_start", "done"]


@pytest.mark.asyncio
async def test_run_foundry_emits_expected_sequence():
    events = []

    def on_event(t, **d):
        events.append(t)

    async def generate(spec):
        return {"slug": "g", "files": _FILES, "violations": []}

    scores = iter([2.0, 5.0])

    async def play(slug, files, spec, secs):
        return _progress(next(scores))

    await run_foundry(_spec(), generate=generate, play=play, passes=2,
                      on_event=on_event)
    assert events[0] == "run_start"
    assert events[-1] == "done"
    assert events.count("pass_start") == 2
    assert events.count("generating") == 2
    assert events.count("play_start") == 2
    assert events.count("pass_scored") == 2


@pytest.mark.asyncio
async def test_3d_emits_asset_render():
    events = []

    async def asset(spec):
        return {"glb_asset": "assets/g.glb", "render_png_bytes": b"PNG"}

    async def verify(img, obj):
        return []

    async def generate(spec):
        return {"slug": "g", "files": _FILES, "violations": []}

    async def play(slug, files, spec, secs):
        return _progress(3.0)

    await run_foundry(_spec("3d"), generate=generate, play=play,
                      asset=asset, verify=verify, passes=1,
                      on_event=lambda t, **d: events.append((t, d)))
    types = [t for t, _ in events]
    assert "asset_render" in types
    render_ev = next(d for t, d in events if t == "asset_render")
    assert render_ev["image"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_contract_violation_emits_blocker_defects():
    events = []

    async def generate(spec):
        return {"slug": "g", "files": {}, "violations": ["no <canvas>"]}

    async def play(slug, files, spec, secs):
        return _progress(1.0)

    await run_foundry(_spec(), generate=generate, play=play, passes=1,
                      on_event=lambda t, **d: events.append((t, d)))
    scored = next(d for t, d in events if t == "pass_scored")
    assert scored["score"] is None
    assert scored["defects"][0]["kind"] == "contract"
