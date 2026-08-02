"""Verifier router tests — the honest oracle-tier verdict.

The load-bearing test is `test_no_regression_only_is_human_required`: a change
where the code runs and nothing regressed, but no oracle confirmed the *intent*
(the CSS-button case), must land as human_required — NOT verified.
"""

from __future__ import annotations

from augmentum.selfedit import verifier as V


def _v(name, oracle, status, *, confirms_intent=False, confidence=1.0,
       required=True, cost=1, intent_classes=("*",)):
    async def run(ctx):
        return V.VerifierResult(name, oracle, status, confirms_intent=confirms_intent,
                                confidence=confidence, required=required,
                                score=1.0 if status == V.PASS else 0.0)
    return V.Verifier(name, oracle, run, intent_classes, confirms_intent, cost, required)


def _pool(*verifiers):
    return {v.name: v for v in verifiers}


async def test_no_regression_only_is_human_required():
    # The CSS-button case: health/compile pass (no regression) but nothing
    # confirms the change did what was asked.
    pool = _pool(_v("health", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False),
                 _v("compile", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False))
    verdict = await V.verify({}, verifiers=pool)
    assert verdict.passed is True              # didn't break
    assert verdict.tier == V.TIER_HUMAN_REQUIRED   # but NOT "good"
    assert verdict.auto_promotable is False


async def test_mechanical_intent_confirm_is_verified():
    pool = _pool(_v("health", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False),
                 _v("behavior_gate", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=True, cost=5))
    verdict = await V.verify({}, verifiers=pool)
    assert verdict.tier == V.TIER_VERIFIED
    assert verdict.auto_promotable is True


async def test_judgment_confirm_respects_confidence_floor():
    high = _pool(_v("judge", V.ORACLE_JUDGMENT, V.PASS, confirms_intent=True,
                    confidence=0.9, required=False))
    assert (await V.verify({}, verifiers=high)).tier == V.TIER_PROBABLE

    low = _pool(_v("judge", V.ORACLE_JUDGMENT, V.PASS, confirms_intent=True,
                   confidence=0.5, required=False))
    # below the 0.7 floor → judgment doesn't count → nothing confirmed intent
    assert (await V.verify({}, verifiers=low)).tier == V.TIER_HUMAN_REQUIRED


async def test_required_failure_short_circuits_expensive():
    ran = {"expensive": False}

    async def cheap_fail(ctx):
        return V.VerifierResult("cheap", V.ORACLE_MECHANICAL, V.FAIL,
                                confirms_intent=False, required=True, score=0.0)

    async def expensive(ctx):
        ran["expensive"] = True
        return V.VerifierResult("expensive", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=True)

    pool = _pool(V.Verifier("cheap", V.ORACLE_MECHANICAL, cheap_fail, cost=1),
                 V.Verifier("expensive", V.ORACLE_MECHANICAL, expensive, cost=10, confirms_intent=True))
    verdict = await V.verify({}, verifiers=pool)
    assert verdict.tier == V.TIER_FAILED and verdict.passed is False
    assert ran["expensive"] is False  # never burned the expensive check past a hard fail


async def test_human_verdict_kept_and_reverted():
    pool = _pool(_v("health", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False))
    kept = await V.verify({}, verifiers=pool, extra_results=[V.human_verdict(True, note="looks right")])
    assert kept.tier == V.TIER_HUMAN_CONFIRMED

    reverted = await V.verify({}, verifiers=pool, extra_results=[V.human_verdict(False, note="worse")])
    assert reverted.tier == V.TIER_FAILED and reverted.passed is False


async def test_intent_class_filtering():
    fe_only = _v("fe", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=True, intent_classes=("frontend",))
    pool = _pool(fe_only, _v("health", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False))
    # backend change → the frontend-only intent-confirmer doesn't apply → human_required
    backend = await V.verify({}, intent_class="backend", verifiers=pool)
    assert backend.tier == V.TIER_HUMAN_REQUIRED
    # frontend change → it applies → verified
    frontend = await V.verify({}, intent_class="frontend", verifiers=pool)
    assert frontend.tier == V.TIER_VERIFIED


async def test_judgment_verifier_wrapper():
    async def judge(ctx):
        return (True, 0.85, "matches the request")
    v = V.judgment_verifier("llm_judge", judge)
    r = await v.run({})
    assert r.oracle == V.ORACLE_JUDGMENT and r.status == V.PASS
    assert r.confirms_intent is True and r.confidence == 0.85

    async def boom(ctx):
        raise RuntimeError("judge down")
    bad = await V.judgment_verifier("j2", boom).run({})
    assert bad.status == V.SKIP  # a crashing judge skips, doesn't block


async def test_registry_and_serialization():
    V.clear_registry()
    try:
        V.register_verifier(_v("health", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False))
        verdict = await V.verify({})  # uses registry
        assert verdict.tier == V.TIER_HUMAN_REQUIRED
        d = verdict.to_dict()
        assert d["tier"] == V.TIER_HUMAN_REQUIRED and d["results"][0]["name"] == "health"
        assert isinstance(verdict.to_json(), str)
    finally:
        V.clear_registry()
