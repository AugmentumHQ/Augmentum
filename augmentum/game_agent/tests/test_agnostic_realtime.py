"""Tests for the game-agnostic real-time tier.

Three pillars, each of which must work on a title NOBODY wrote a
translation layer for:

* **Chorded input** — ``PlanAction.also`` holds extra buttons
  simultaneously with the primary (run+jump). Wire shape, parser
  leniency, reflex support, and the adapter's single-frame dispatch.
* **Scene probes** — narrator-derived facts ride the same probes spine
  RAM uses (rank-gated), so reflexes/overlay/goals work vision-only.
* **Motion sense** — frame-diff region heuristics feed the input
  context tracker when no RAM position/text probes exist.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from augmentum.game_agent.motion import MotionSense
from augmentum.game_agent.prompt import parse_fast_output, parse_plan_output
from augmentum.game_agent.reflex import compile_reflex_rule
from augmentum.game_agent.schema import PlanAction, SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.bridged import BridgedAdapter
from augmentum.game_agent.world import WorldState

CAPS = SurfaceCapsPayload(
    semantic_inputs=["confirm", "cancel", "nav_left", "nav_right", "menu"],
    log_schema="mock.v1",
    observation_modalities=["log", "frame"],
)


# ── PlanAction.also (schema) ──────────────────────────────────────────


def test_plan_action_accepts_chord_extras() -> None:
    a = PlanAction(semantic="confirm", duration_ms=300, also=["nav_right"])
    assert a.also == ["nav_right"]


def test_plan_action_rejects_more_than_two_extras() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PlanAction(
            semantic="confirm", duration_ms=300,
            also=["nav_right", "nav_left", "menu"],
        )


# ── fast-turn parser ("+") ────────────────────────────────────────────


def test_fast_parse_keeps_known_chord_extras() -> None:
    raw = json.dumps(
        {"a": [{"s": "confirm", "d": 400, "+": ["nav_right"]}],
         "why": "run-jump", "next_ms": 200}
    )
    plan = parse_fast_output(raw, CAPS)
    assert plan.actions[0].also == ["nav_right"]


def test_fast_parse_drops_unknown_and_self_chord_extras() -> None:
    raw = json.dumps(
        {"a": [{"s": "confirm", "d": 400,
                "+": ["warp_drive", "confirm", "nav_left"]}],
         "why": "", "next_ms": 200}
    )
    plan = parse_fast_output(raw, CAPS)
    # Unknown + self-duplicate removed; the legal member survives.
    assert plan.actions[0].also == ["nav_left"]


def test_fast_parse_chord_none_when_all_extras_invalid() -> None:
    raw = json.dumps(
        {"a": [{"s": "confirm", "d": 400, "+": ["warp_drive"]}],
         "why": "", "next_ms": 200}
    )
    plan = parse_fast_output(raw, CAPS)
    assert plan.actions[0].also is None


# ── FULL-plan parser ("also" via pydantic) ────────────────────────────


def test_full_plan_accepts_also_field() -> None:
    raw = json.dumps(
        {
            "observations": ["moving right"],
            "state_update": "",
            "actions": [
                {"semantic": "confirm", "duration_ms": 500,
                 "also": ["nav_right"]}
            ],
            "confidence": 0.8,
            "next_check_in_ms": 300,
        }
    )
    plan = parse_plan_output(raw, CAPS)
    assert plan.actions[0].also == ["nav_right"]


# ── reflex "do" chords ────────────────────────────────────────────────


def test_reflex_do_accepts_chord_extras() -> None:
    rule = compile_reflex_rule(
        {
            "id": "run-right",
            "when": {"probe": "screen", "equals": "overworld"},
            "do": [{"s": "nav_right", "d": 600, "+": ["confirm"]}],
        }
    )
    window = [
        {
            "t": 0,
            "kind": "event",
            "payload": {
                "channel": "vlm",
                "data": {
                    "event": "scene_probes",
                    "probes": {"screen": "overworld"},
                    "probe_source": "scene",
                },
            },
        }
    ]
    m = rule.predicate(window, CAPS)
    assert m is not None
    assert m.actions[0].also == ["confirm"]


def test_reflex_fires_on_scene_probe_events() -> None:
    """Vision-derived probes drive reflexes on games with ZERO RAM."""

    rule = compile_reflex_rule(
        {
            "id": "advance-dialog",
            "when": {"probe": "dialog_text", "changed": True},
            "do": [{"s": "confirm", "d": 120}],
        }
    )
    window = [
        {
            "t": 100,
            "kind": "event",
            "payload": {
                "channel": "vlm",
                "data": {
                    "event": "scene_probes",
                    "probes": {"screen": "dialog",
                               "dialog_text": "Welcome, traveler!"},
                    "probe_source": "scene",
                },
            },
        }
    ]
    m = rule.predicate(window, CAPS)
    assert m is not None and m.actions[0].semantic == "confirm"


# ── semantic resolver chords ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolver_apply_chord_requires_binding() -> None:
    r = SemanticInputResolver()

    async def _noop(_d: int) -> None:
        pass

    r.bind("confirm", _noop)
    assert r.supports_chord is False
    with pytest.raises(RuntimeError):
        await r.apply_chord(["confirm"], 100)


@pytest.mark.asyncio
async def test_resolver_apply_chord_validates_members() -> None:
    r = SemanticInputResolver()
    calls: list[tuple[list[str], int]] = []

    async def _noop(_d: int) -> None:
        pass

    async def _chord(sems: list[str], d: int) -> None:
        calls.append((sems, d))

    r.bind("confirm", _noop)
    r.bind_chord(_chord)
    assert r.supports_chord is True
    with pytest.raises(KeyError):
        await r.apply_chord(["confirm", "unbound"], 100)
    await r.apply_chord(["confirm"], 150)
    assert calls == [(["confirm"], 150)]


# ── bridged adapter chord wire frame ──────────────────────────────────


class _SendCapturingWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._never: asyncio.Future = asyncio.get_event_loop().create_future()

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        return await self._never

    async def close(self) -> None:
        if not self._never.done():
            self._never.cancel()


@pytest.mark.asyncio
async def test_bridged_chord_sends_one_frame_with_parts() -> None:
    ws = _SendCapturingWS()
    adapter = BridgedAdapter(
        websocket=ws,  # type: ignore[arg-type]
        surface_kind="emulatorjs",
        semantic_inputs=["confirm", "nav_right"],
        log_schema="mock.v1",
    )
    resolver = adapter.resolver
    assert resolver.supports_chord is True
    task = asyncio.create_task(resolver.apply_chord(["confirm", "nav_right"], 400))
    for _ in range(50):
        if ws.sent:
            break
        await asyncio.sleep(0)
    assert ws.sent, "chord dispatch never hit the wire"
    payload = ws.sent[0]
    assert payload["action"] == "confirm"
    assert payload["duration_ms"] == 400
    assert payload["chord"] == [{"button": "nav_right"}]
    # ONE wire frame, one request_id — chords are atomic on the wire.
    assert len(ws.sent) == 1
    adapter._pending_acks[payload["request_id"]].set()
    await task


# ── world.update_many (rank-aware bulk write) ─────────────────────────


def test_update_many_accepts_when_unowned() -> None:
    w = WorldState()
    accepted = w.update_many(
        {"screen": "dialog", "dialog_text": "hi there"},
        source="scene", t_ms=1000,
    )
    assert accepted == {"screen": "dialog", "dialog_text": "hi there"}
    assert w.facts["screen"].source == "scene"


def test_update_many_rejected_under_fresh_ram() -> None:
    w = WorldState()
    w.update("screen", "overworld", source="ram", t_ms=900)
    accepted = w.update_many({"screen": "dialog"}, source="scene", t_ms=1200)
    assert accepted == {}
    assert w.facts["screen"].value == "overworld"
    assert w.facts["screen"].source == "ram"


def test_update_many_accepts_when_ram_stale() -> None:
    w = WorldState()
    w.update("screen", "overworld", source="ram", t_ms=100)
    # 5s freshness window elapsed — the narrator may refresh the fact.
    accepted = w.update_many({"screen": "menu"}, source="scene", t_ms=9000)
    assert accepted == {"screen": "menu"}
    assert w.facts["screen"].source == "scene"


def test_update_many_same_value_refresh_is_accepted() -> None:
    w = WorldState()
    w.update_many({"screen": "dialog"}, source="scene", t_ms=1000)
    accepted = w.update_many({"screen": "dialog"}, source="scene", t_ms=2000)
    assert accepted == {"screen": "dialog"}
    assert w.facts["screen"].t_ms == 2000


# ── motion sense ──────────────────────────────────────────────────────

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def _png(draw_fn=None, color=(40, 44, 52)) -> bytes:
    img = Image.new("RGB", (240, 160), color)
    if draw_fn is not None:
        draw_fn(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_motion_static_frames_read_nothing() -> None:
    m = MotionSense()
    assert m.feed(_png()) is None          # first frame: no baseline
    assert m.feed(_png()) is None          # identical: no change


def test_motion_bottom_band_change_is_text_printing() -> None:
    m = MotionSense()
    m.feed(_png())
    # A dialog box strip appearing across the bottom of the frame.
    def _box(d):
        d.rectangle([8, 120, 232, 156], fill=(240, 240, 240))
    r = m.feed(_png(_box))
    assert r is not None
    assert r.text_printing is True
    assert r.screen_changed is False
    assert r.world_motion is False


def test_motion_full_frame_change_is_screen_transition() -> None:
    m = MotionSense()
    m.feed(_png(color=(20, 20, 20)))
    r = m.feed(_png(color=(235, 235, 235)))
    assert r is not None and r.screen_changed is True


def test_motion_broad_midband_change_is_world_motion() -> None:
    m = MotionSense()

    def _terrain(offset):
        def _draw(d):
            # A field of tiles shifted horizontally — a camera scroll:
            # broad change through the mid band, bottom HUD untouched.
            for x in range(0, 240, 24):
                for y in range(8, 104, 24):
                    d.rectangle(
                        [x + offset, y, x + offset + 12, y + 12],
                        fill=(90, 160, 90),
                    )
        return _draw

    m.feed(_png(_terrain(0)))
    r = m.feed(_png(_terrain(12)))
    assert r is not None
    assert r.world_motion is True
    assert r.screen_changed is False
