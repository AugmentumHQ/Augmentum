"""Sovereign Perception Pipeline — fusion framework + orchestrator end to end.

Proves the full loop on synthetic fusers (real fusers land with real data streams):
  - fuse() runs registered fusers, isolates a broken one, dedups by shape;
  - run_perception threads ONE budget across the batch (strongest insights claim
    interruptions first; a single pass can't over-spend);
  - dispatch routes each decision to the right sink method and charges the budget
    only on a successful delivery;
  - perceive_and_dispatch ties it together with injected regret + budget.
"""

from __future__ import annotations

from augmentum.companion_runtime.perception import (
    ACT_WITH_CONSENT,
    FILE_FOR_PULL,
    SILENT,
    SPEAK,
    FusionContext,
    Insight,
    InterruptionBudgetStore,
    dispatch,
    fuse,
    perceive_and_dispatch,
    run_perception,
)


def _ctx(**kw) -> FusionContext:
    base = {"user_id": "u", "now": 1000.0, "in_conversation": False}
    base.update(kw)
    return FusionContext(**base)


# --- fusion framework ------------------------------------------------------

def test_fuse_runs_fusers_and_collects():
    def f1(ctx):
        return [Insight(kind="a.x", summary="ax", value=0.9, confidence=0.9)]
    def f2(ctx):
        return [Insight(kind="b.y", summary="by", value=0.8, confidence=0.8)]
    out = fuse(_ctx(), fusers=[("f1", f1), ("f2", f2)])
    assert {i.kind for i in out} == {"a.x", "b.y"}


def test_fuse_dedups_by_shape_keeping_strongest():
    # two fusers both fire on the "social" shape — only the stronger survives
    def weak(ctx):
        return [Insight(kind="social.a", summary="w", value=0.4, confidence=0.4)]
    def strong(ctx):
        return [Insight(kind="social.b", summary="s", value=0.9, confidence=0.9)]
    out = fuse(_ctx(), fusers=[("weak", weak), ("strong", strong)])
    assert len(out) == 1 and out[0].summary == "s"   # same shape "social" → strongest


def test_fuse_isolates_a_broken_fuser():
    def boom(ctx):
        raise RuntimeError("bad stream")
    def good(ctx):
        return [Insight(kind="ok.x", summary="ok", value=0.7, confidence=0.7)]
    out = fuse(_ctx(), fusers=[("boom", boom), ("good", good)])
    assert [i.kind for i in out] == ["ok.x"]   # broken fuser skipped, good survives


def test_fuse_returns_strongest_first():
    def f(ctx):
        return [
            Insight(kind="a.x", summary="a", value=0.5, confidence=0.5),  # 0.25
            Insight(kind="b.y", summary="b", value=0.9, confidence=0.9),  # 0.81
        ]
    out = fuse(_ctx(), fusers=[("f", f)])
    assert [i.kind for i in out] == ["b.y", "a.x"]


# --- run_perception: batch budget threading --------------------------------

def test_budget_threaded_across_batch_strongest_first():
    # three strong, time-critical insights; budget = 1. Only the STRONGEST speaks;
    # the rest fall to pull. Proves one pass can't over-spend.
    def f(ctx):
        return [
            Insight(kind="x.a", summary="a", shape="a", value=0.95, confidence=0.95, time_critical=True),
            Insight(kind="x.b", summary="b", shape="b", value=0.90, confidence=0.90, time_critical=True),
            Insight(kind="x.c", summary="c", shape="c", value=0.85, confidence=0.85, time_critical=True),
        ]
    from augmentum.companion_runtime.perception import fusion as _fusion
    _fusion.register_fuser("batch", f)
    try:
        decisions = run_perception(_ctx(), regret_multiplier=1.0, budget_remaining=1)
    finally:
        _fusion.clear_fusers()
    speaks = [d for _, d in decisions if d.channel == SPEAK]
    pulls = [d for _, d in decisions if d.channel == FILE_FOR_PULL]
    assert len(speaks) == 1 and speaks[0].spent_budget          # exactly one interrupt
    assert len(pulls) == 2                                      # the rest deferred
    assert decisions[0][0].summary == "a"                      # strongest got the budget


def test_run_perception_empty_when_no_fusers():
    from augmentum.companion_runtime.perception import fusion as _fusion
    _fusion.clear_fusers()
    assert run_perception(_ctx(), regret_multiplier=1.0, budget_remaining=3) == []


# --- dispatch routing + budget charging ------------------------------------

class _FakeSink:
    def __init__(self):
        self.filed, self.spoke, self.proposed = [], [], []

    async def file_for_pull(self, insight, decision):
        self.filed.append(insight)

    async def speak(self, insight, decision):
        self.spoke.append(insight)

    async def propose_action(self, insight, decision):
        self.proposed.append(insight)


async def test_dispatch_routes_each_channel():
    from augmentum.companion_runtime.perception import (
        DeliveryDecision,
    )
    sink = _FakeSink()
    pairs = [
        (Insight(kind="a.x", summary="pull"), DeliveryDecision(FILE_FOR_PULL, "r")),
        (Insight(kind="b.y", summary="say"), DeliveryDecision(SPEAK, "r", spent_budget=True)),
        (Insight(kind="c.z", summary="act"), DeliveryDecision(ACT_WITH_CONSENT, "r")),
        (Insight(kind="d.w", summary="quiet"), DeliveryDecision(SILENT, "r")),
    ]
    store = InterruptionBudgetStore(cap=3)
    counts = await dispatch(pairs, sink=sink, budget_store=store, user_id="u", now=1000.0)
    assert [i.summary for i in sink.filed] == ["pull"]
    assert [i.summary for i in sink.spoke] == ["say"]
    assert [i.summary for i in sink.proposed] == ["act"]
    assert counts[SPEAK] == 1 and counts["silent"] == 1
    assert store.remaining("u", 1000.0) == 2   # the SPEAK charged exactly one unit


async def test_dispatch_does_not_charge_budget_on_sink_failure():
    # a speak that spent_budget but whose sink throws must NOT burn the budget
    class _BrokenSink(_FakeSink):
        async def speak(self, insight, decision):
            raise RuntimeError("tts down")

    from augmentum.companion_runtime.perception import DeliveryDecision
    store = InterruptionBudgetStore(cap=2)
    pairs = [(Insight(kind="b.y", summary="say"), DeliveryDecision(SPEAK, "r", spent_budget=True))]
    counts = await dispatch(pairs, sink=_BrokenSink(), budget_store=store, user_id="u", now=1000.0)
    assert counts[SPEAK] == 0                      # delivery didn't happen
    assert store.remaining("u", 1000.0) == 2       # budget intact — no non-event charge


# --- perceive_and_dispatch: the single entry -------------------------------

async def test_perceive_and_dispatch_full_loop():
    from augmentum.companion_runtime.perception import fusion as _fusion

    def f(ctx):
        return [
            Insight(kind="logi.flight", summary="flight slipped", shape="logi",
                    value=0.95, confidence=0.95, time_critical=True),
            Insight(kind="info.note", summary="fyi", shape="info",
                    value=0.7, confidence=0.7, time_critical=False),
        ]
    _fusion.register_fuser("t", f)
    sink, store = _FakeSink(), InterruptionBudgetStore(cap=3)
    try:
        counts = await perceive_and_dispatch(
            _ctx(), regret_multiplier=1.0, budget_remaining=3,
            sink=sink, budget_store=store,
        )
    finally:
        _fusion.clear_fusers()
    # flight → interrupt (spoke), note → pull
    assert [i.summary for i in sink.spoke] == ["flight slipped"]
    assert [i.summary for i in sink.filed] == ["fyi"]
    assert counts[SPEAK] == 1 and counts[FILE_FOR_PULL] == 1
    assert store.remaining("u", 1000.0) == 2


async def test_dismissive_user_makes_the_whole_pass_quieter():
    # same insights, regret 0.5 (they dismiss her) → the interrupt downgrades to pull
    from augmentum.companion_runtime.perception import fusion as _fusion

    def f(ctx):
        return [Insight(kind="logi.flight", summary="flight", shape="logi",
                        value=0.9, confidence=0.9, time_critical=True)]
    _fusion.register_fuser("t", f)
    sink, store = _FakeSink(), InterruptionBudgetStore(cap=3)
    try:
        await perceive_and_dispatch(
            _ctx(), regret_multiplier=0.5, budget_remaining=3,
            sink=sink, budget_store=store,
        )
    finally:
        _fusion.clear_fusers()
    assert sink.spoke == [] and [i.summary for i in sink.filed] == ["flight"]
    assert store.remaining("u", 1000.0) == 3   # nothing interrupted → budget untouched
