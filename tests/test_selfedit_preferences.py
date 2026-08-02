"""Learning-loop tests — the archive becomes judgment.

Locks the honest contract: trust is earned by accumulation (one keep proves
nothing), a consistently-kept shape lifts human_required → probable (never to
verified, never silently auto), and an untrusted shape leaves the verdict
untouched.
"""

from __future__ import annotations

import aiosqlite

from augmentum.selfedit import preferences as P
from augmentum.selfedit import verifier as V
from augmentum.selfedit.growth_db import _SCHEMA


async def _store():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    await conn.commit()
    return P.PreferenceStore(conn)


def test_change_shape_normalizes():
    assert P.change_shape("Config", "Adaptation") == "config:adaptation"
    assert P.change_shape("backend", "") == "backend"
    assert P.change_shape("", "") == "system"


def test_calibration_thresholds():
    assert P.is_trusted(3, 0) is True              # 3 keeps, 100%
    assert P.is_trusted(2, 0) is False             # below min samples
    assert P.is_trusted(4, 1) is True              # 80% over 5
    assert P.is_trusted(3, 2) is False             # 60% — below the trust bar
    assert P.confidence(3, 1) == 0.75


async def test_record_and_stat():
    s = await _store()
    try:
        await s.record(user_id="u1", shape="config:adaptation", kept=True)
        await s.record(user_id="u1", shape="config:adaptation", kept=True)
        await s.record(user_id="u1", shape="config:adaptation", kept=False)
        st = await s.stat(user_id="u1", shape="config:adaptation")
        assert st.kept == 2 and st.reverted == 1 and st.samples == 3
        assert round(st.confidence, 3) == 0.667 and st.trusted is False
        # scoped per user
        assert (await s.stat(user_id="u2", shape="config:adaptation")).samples == 0
    finally:
        await s.conn.close()


async def test_summary_orders_by_activity():
    s = await _store()
    try:
        for _ in range(4):
            await s.record(user_id="u1", shape="frontend:style", kept=True)
        await s.record(user_id="u1", shape="backend:bugfix", kept=True)
        rows = await s.summary(user_id="u1")
        assert rows[0].shape == "frontend:style" and rows[0].samples == 4
    finally:
        await s.conn.close()


def _noconfirm_pass():
    # a mechanical no-regression check that passes but does NOT confirm intent →
    # on its own the verdict is human_required (the case the lift targets).
    async def run(_ctx):
        return V.VerifierResult("boot", V.ORACLE_MECHANICAL, V.PASS, confirms_intent=False)
    return V.Verifier("boot", V.ORACLE_MECHANICAL, run, confirms_intent=False)


async def test_lift_untrusted_stays_human_required():
    s = await _store()
    try:
        pv = P.preference_verifier("backend:bugfix", store=s, user_id="u1")
        verdict = await V.verify({}, verifiers={"boot": _noconfirm_pass(), "preference": pv})
        assert verdict.tier == V.TIER_HUMAN_REQUIRED   # no earned evidence yet
    finally:
        await s.conn.close()


async def test_lift_trusted_becomes_probable():
    s = await _store()
    try:
        for _ in range(3):                              # earn trust
            await s.record(user_id="u1", shape="backend:bugfix", kept=True)
        pv = P.preference_verifier("backend:bugfix", store=s, user_id="u1")
        verdict = await V.verify({}, verifiers={"boot": _noconfirm_pass(), "preference": pv})
        assert verdict.tier == V.TIER_PROBABLE          # learned trust → lifted
        assert verdict.auto_promotable is False         # never silently auto-ships taste
    finally:
        await s.conn.close()
