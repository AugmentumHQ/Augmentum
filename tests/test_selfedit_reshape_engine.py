"""Tests for the surface-reshape ENGINE (NL ask → change → reshape → record).

Classifier and recorder are injected, so this runs with deterministic fakes. The
load-bearing tests:
  - a mappable ask drives an end-to-end config reshape and lands STATUS_PROMOTED;
  - an UNMAPPABLE ask records nothing and reports honestly (no silent no-op);
  - the recorder hooks fire with the right args, and APPLIED_PENDING is left OPEN
    (not finalized) — awaiting the human pick;
  - the engine carries the requester's actor onto the change (never the anon row).
"""

from __future__ import annotations

from augmentum.selfedit import verifier as V
from augmentum.selfedit.surfaces import (
    CLASS_ADAPTATION,
    STATUS_APPLIED_PENDING,
    STATUS_PROMOTED,
    STATUS_UNMAPPED,
    CaptureArtifact,
    ReshapeChange,
    ReshapeOutcome,
    ReshapeRequest,
    SurfaceAdapter,
    build_config_surface,
    clear_surfaces,
    register_surface,
    run_reshape_request,
)


class FakeStore:
    def __init__(self):
        self.data: dict = {}

    async def read(self, uid, key):
        return self.data.get((uid, key))

    async def write(self, uid, key, val):
        self.data[(uid, key)] = val


def _register_config(store: FakeStore):
    clear_surfaces()
    register_surface(build_config_surface(read=store.read, write=store.write))


async def _classify_theme(request: ReshapeRequest, surfaces: list[str]):
    # A closed-vocab fake: "make it dark" → set config theme=dark.
    if "dark" in request.ask and "config" in surfaces:
        return ReshapeChange(surface="config", change_class=CLASS_ADAPTATION,
                             payload={"key": "theme", "value": "dark"})
    return None  # unmappable


async def test_mappable_ask_drives_end_to_end_and_promotes():
    store = FakeStore()
    _register_config(store)
    req = ReshapeRequest(ask="make it dark please", actor="u1")
    res = await run_reshape_request(req, classify=_classify_theme)
    assert res.mapped and res.status == STATUS_PROMOTED
    assert res.reshape.verdict.tier == V.TIER_VERIFIED
    assert store.data[("u1", "theme")] == "dark"
    assert res.change.actor == "u1"                  # requester carried onto the change


async def test_unmappable_ask_is_honest_noop():
    store = FakeStore()
    _register_config(store)
    req = ReshapeRequest(ask="do something vague", actor="u1")
    res = await run_reshape_request(req, classify=_classify_theme)
    assert res.mapped is False and res.status == STATUS_UNMAPPED
    assert res.reshape is None
    assert store.data == {}                          # nothing happened


async def test_recorder_hooks_fire_and_finalize_terminal():
    store = FakeStore()
    _register_config(store)
    starts: list = []
    finishes: list = []

    async def on_start(aid, request, change):
        starts.append((aid, request.actor, change.surface))

    async def on_finish(aid, actor, result, status):
        finishes.append((aid, actor, status))

    req = ReshapeRequest(ask="make it dark", actor="u7")
    res = await run_reshape_request(req, classify=_classify_theme,
                                    on_start=on_start, on_finish=on_finish)
    assert starts and starts[0][1] == "u7" and starts[0][2] == "config"
    assert finishes and finishes[0] == (res.attempt_id, "u7", STATUS_PROMOTED)


def _taste_adapter() -> SurfaceAdapter:
    """A surface whose oracle only proves liveness → APPLIED_PENDING (needs human)."""
    async def apply(_c):
        return ReshapeOutcome(True, revert_token="t")

    async def revert(_t):
        return True

    def make_verifier(_c):
        async def run(_ctx):
            return V.VerifierResult("liveness", V.ORACLE_MECHANICAL, V.PASS,
                                    confirms_intent=False, required=True)
        return V.Verifier("liveness", V.ORACLE_MECHANICAL, run, ("*",), False)

    async def capture(_c):
        return CaptureArtifact(kind="screenshot", ref="x", summary="s")

    return SurfaceAdapter("taste", (CLASS_ADAPTATION,), apply, revert, make_verifier, capture)


async def test_applied_pending_is_left_open_for_human_verdict():
    clear_surfaces()
    register_surface(_taste_adapter())
    finished: list = []

    async def on_finish(aid, actor, result, status):
        finished.append(status)

    async def classify(_req, _surfaces):
        return ReshapeChange(surface="taste", change_class=CLASS_ADAPTATION,
                             payload={}, actor="u1")

    req = ReshapeRequest(ask="recolor the room", actor="u1")
    res = await run_reshape_request(req, classify=classify, on_finish=on_finish)
    assert res.status == STATUS_APPLIED_PENDING       # kept, but needs the user's pick
    assert res.reshape.needs_human is True
    assert finished == [STATUS_APPLIED_PENDING]        # engine reports it; recorder leaves it open
