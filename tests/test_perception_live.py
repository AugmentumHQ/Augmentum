"""Live wiring — the sink delivers onto the REAL initiative queue + bus, and the
runtime adapter resolves regret/budget/snapshot and runs a pass.

Uses an in-memory SQLite backend with migrations applied (real schema, real
``initiative.enqueue`` SQL — catches drift) + a MagicMock runtime with a recording
bus, mirroring test_companion_initiative. Load-bearing:
  - SPEAK enqueues a 'perceived' row AND publishes initiative.surfaced (pointer +
    payload-in-row, the existing pattern);
  - FILE_FOR_PULL enqueues but does NOT surface (pull, no interruption);
  - propose_action surfaces with the proposes_action marker (→ gated confirm);
  - evaluate_user is OFF by default and a cheap no-op with no fusers;
  - when enabled with a fuser, a full pass delivers through the real queue and
    charges the budget exactly once for the interrupt.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from augmentum.companion_runtime.perception import (
    FILE_FOR_PULL,
    SPEAK,
    CompanionPerceptionSink,
    Insight,
    evaluate_user,
)
from augmentum.companion_runtime.perception.budget import BUDGET
from augmentum.companion_runtime.perception.insight import (
    ACT_WITH_CONSENT,
    DeliveryDecision,
)


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


def _fake_runtime(backend):
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    runtime.owner_user_id = "u1"
    runtime.app_state = None
    published: list[tuple[str, dict]] = []

    async def _publish(topic, payload, **kw):
        published.append((topic, payload))
        return None

    runtime.bus.publish_topic = _publish
    runtime._published = published
    return runtime


async def _queue_rows(backend, user_id="u1"):
    cur = await backend.conn.execute(
        "SELECT id, kind, payload, score, status FROM companion_initiative_queue "
        "WHERE user_id = ? ORDER BY id",
        (user_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    return rows


def _insight(**kw) -> Insight:
    base = {"kind": "logi.flight", "summary": "flight slipped to 8",
            "value": 0.9, "confidence": 0.9, "time_critical": True}
    base.update(kw)
    return Insight(**base)


# --- the sink onto the real queue + bus ------------------------------------

@pytest.mark.asyncio
async def test_speak_enqueues_and_surfaces():
    backend = await _boot_backend()
    rt = _fake_runtime(backend)
    sink = CompanionPerceptionSink(rt)

    await sink.speak(_insight(), DeliveryDecision(SPEAK, "r", spent_budget=True))

    rows = await _queue_rows(backend)
    assert len(rows) == 1
    _id, kind, payload, score, status = rows[0]
    assert kind == "perceived"
    body = json.loads(payload)
    assert body["summary"] == "flight slipped to 8" and body["source"] == "perception"
    # surfaced on the existing topic, as a pointer (id/kind/score) + summary
    assert rt._published and rt._published[0][0] == "initiative.surfaced"
    assert rt._published[0][1]["id"] == _id and rt._published[0][1]["kind"] == "perceived"


@pytest.mark.asyncio
async def test_file_for_pull_enqueues_without_surfacing():
    backend = await _boot_backend()
    rt = _fake_runtime(backend)
    sink = CompanionPerceptionSink(rt)

    await sink.file_for_pull(_insight(value=0.6, confidence=0.7),
                             DeliveryDecision(FILE_FOR_PULL, "r"))

    rows = await _queue_rows(backend)
    assert len(rows) == 1 and rows[0][1] == "perceived"
    assert rt._published == []   # pull never interrupts — no bus event


@pytest.mark.asyncio
async def test_propose_action_surfaces_with_marker():
    backend = await _boot_backend()
    rt = _fake_runtime(backend)
    sink = CompanionPerceptionSink(rt)

    ins = _insight(suggested_action="message.send", stakes="disruptive")
    await sink.propose_action(ins, DeliveryDecision(ACT_WITH_CONSENT, "r"))

    assert rt._published[0][0] == "initiative.surfaced"
    ev = rt._published[0][1]
    assert ev["proposes_action"] is True and ev["suggested_action"] == "message.send"


# --- the runtime adapter ---------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_user_off_by_default(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_perception_enabled", False, raising=False)
    backend = await _boot_backend()
    rt = _fake_runtime(backend)
    out = await evaluate_user(rt, user_id="u1")
    assert out == {} and await _queue_rows(backend) == []   # inert when disabled


@pytest.mark.asyncio
async def test_evaluate_user_enabled_no_fusers_is_noop(monkeypatch):
    from augmentum.companion_runtime.perception import fusion as _fusion
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_perception_enabled", True, raising=False)
    _fusion.clear_fusers()
    backend = await _boot_backend()
    rt = _fake_runtime(backend)
    out = await evaluate_user(rt, user_id="u1")
    # enabled but nothing to fuse → empty queue, no crash (the rollout-safe state)
    assert await _queue_rows(backend) == []
    assert all(v == 0 for v in out.values()) if out else True


@pytest.mark.asyncio
async def test_evaluate_user_full_pass_delivers_and_charges_budget(monkeypatch):
    from augmentum.companion_runtime.perception import fusion as _fusion
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_perception_enabled", True, raising=False)
    monkeypatch.setattr(settings, "companion_interruption_budget_per_day", 3, raising=False)
    BUDGET.reset()

    def fuser(ctx):
        return [
            _insight(kind="logi.flight", summary="flight slipped", shape="logi"),
            Insight(kind="info.fyi", summary="a note", shape="info",
                    value=0.7, confidence=0.7, time_critical=False),
        ]
    _fusion.register_fuser("t", fuser)
    backend = await _boot_backend()
    rt = _fake_runtime(backend)
    try:
        out = await evaluate_user(rt, user_id="u1", now=1000.0)
    finally:
        _fusion.clear_fusers()

    # both insights enqueued (one speak, one pull); exactly one surfaced (the
    # flight interrupt); the budget charged exactly one unit.
    rows = await _queue_rows(backend)
    assert len(rows) == 2
    assert out.get(SPEAK) == 1 and out.get(FILE_FOR_PULL) == 1
    surfaced = [p for (t, p) in rt._published if t == "initiative.surfaced"]
    assert len(surfaced) == 1 and surfaced[0]["summary"] == "flight slipped"
    assert BUDGET.remaining("u1", 1000.0) == 2
