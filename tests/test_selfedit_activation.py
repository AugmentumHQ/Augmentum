"""Verified-skill-graph tests — capability that grows, as a pure fold of the archive.

Locks the contract P1 must keep: edges move ONLY on a real verdict (never raw
frequency), a verified region scores positive and a rolled-back one negative, the
BCM frequency-threshold suppresses monopoly nodes, Oja normalization bounds every
node, cold start is neutral (acts on nothing it hasn't earned), and the whole graph
is a deterministic, reversible projection of the archive (no new state).
"""

from __future__ import annotations

import aiosqlite

from augmentum.selfedit import activation as A
from augmentum.selfedit.growth_db import _SCHEMA


def _attempt(aid, surface, status, files, created, tier="green", target=""):
    return {"id": aid, "surface": surface, "status": status, "tier": tier,
            "files_changed": files, "created_at": created, "target": target}


# --- extractors -----------------------------------------------------------

def test_subsystem_prefix():
    assert A._subsystem("augmentum/selfedit/loop.py") == "augmentum/selfedit"
    assert A._subsystem("./ui/scripts/workshop.js") == "ui/scripts"
    assert A._subsystem("README.md") == "README.md"
    assert A._subsystem("") == ""


def test_atoms_are_namespaced():
    atoms = A.atoms_for_attempt(_attempt("a", "frontend", "promoted",
                                         ["ui/scripts/x.js"], "t", tier="green"))
    assert atoms == {"shape:frontend", "tier:green", "sub:ui/scripts"}


def test_query_atoms_match_archive_vocab():
    assert A.query_atoms(surface="backend", files=["augmentum/selfedit/x.py"]) == {
        "shape:backend", "sub:augmentum/selfedit"}


def test_target_class_is_an_atom():
    # the structured debt class (scanner.metric) becomes a first-class atom so the
    # graph carries per-class trust, not only file/surface regions.
    atoms = A.atoms_for_attempt(_attempt("a", "frontend", "promoted",
                                         ["ui/scripts/x.js"], "t",
                                         target="code_quality.console_log"))
    assert "target:code_quality.console_log" in atoms
    assert A.query_atoms(target="code_quality.console_log") == {
        "target:code_quality.console_log"}


def test_missing_target_adds_no_atom():
    # old archive rows (no target) must fold exactly as before — no phantom atom.
    atoms = A.atoms_for_attempt(_attempt("a", "frontend", "promoted",
                                         ["ui/scripts/x.js"], "t"))
    assert not any(x.startswith("target:") for x in atoms)


def test_score_target_learns_class_trust():
    # a debt class that consistently ships scores positive; an unseen class is cold.
    attempts = [_attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"],
                         f"{i:02d}", target="code_quality.console_log")
                for i in range(6)]
    g = A.build_graph(attempts)
    seen = g.score_target("code_quality", "console_log")
    assert seen.score > 0.15 and seen.confidence > 0.2
    unseen = g.score_target("security", "critical")
    assert unseen.confidence < 0.2  # cold — never a fabricated guess


def test_modulation_only_on_real_verdict():
    assert A.modulation_for_attempt({"status": "promoted"}) == 1.0
    assert A.modulation_for_attempt({"status": "rolled_back"}) == -1.0
    assert A.modulation_for_attempt({"status": "rejected"}) == -0.5
    assert A.modulation_for_attempt({"status": "proposed"}) == 0.0
    assert A.modulation_for_attempt({"status": "editing"}) == 0.0


# --- the fold -------------------------------------------------------------

def test_cold_start_is_neutral():
    g = A.build_graph([])
    s = g.score({"shape:frontend"})
    assert s.score == 0.0 and s.confidence == 0.0
    assert "unknown" in s.rationale or "no learned" in s.rationale


def test_pending_attempts_move_no_weight():
    # only proposed/editing attempts → structure seen, but zero edges, zero support.
    g = A.build_graph([
        _attempt("a", "frontend", "proposed", ["ui/scripts/x.js"], "1"),
        _attempt("b", "frontend", "editing", ["ui/scripts/y.js"], "2"),
    ])
    assert g.attempts == 2
    assert sum(len(r) for r in g.edges.values()) == 0
    assert g.score({"shape:frontend", "sub:ui/scripts"}).support == 0.0


def test_verified_region_scores_positive():
    attempts = [
        _attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/workshop.js"], f"{i:02d}")
        for i in range(5)
    ]
    g = A.build_graph(attempts)
    s = g.score({"shape:frontend", "sub:ui/scripts"})
    assert s.score > 0.15
    assert s.confidence > 0.4
    assert "verified-success" in s.rationale


def test_failed_region_scores_negative():
    attempts = [
        _attempt(f"a{i}", "backend", "rolled_back", ["augmentum/risky/mod.py"], f"{i:02d}")
        for i in range(5)
    ]
    g = A.build_graph(attempts)
    s = g.score({"shape:backend", "sub:augmentum/risky"})
    assert s.score < -0.15
    assert "repeated-failure" in s.rationale


def test_confidence_grows_with_evidence():
    few = A.build_graph([
        _attempt("a", "frontend", "promoted", ["ui/scripts/x.js"], "01"),
    ])
    many = A.build_graph([
        _attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:02d}")
        for i in range(12)
    ])
    q = {"shape:frontend", "sub:ui/scripts"}
    assert many.score(q).confidence > few.score(q).confidence


def test_bcm_threshold_suppresses_monopoly_node():
    # 'tier:green' appears in EVERY attempt → its frequency → 1 → BCM gain → 0, so it
    # accrues far less edge weight than a discriminating subsystem that co-occurs
    # only with successes. Anti-monopoly / diversity for free.
    attempts = []
    for i in range(8):
        status = "promoted" if i % 2 == 0 else "rejected"
        sub = "augmentum/win" if status == "promoted" else "augmentum/lose"
        attempts.append(_attempt(f"a{i}", "backend", status,
                                 [f"{sub}/m.py"], f"{i:02d}"))
    g = A.build_graph(attempts)
    green_net = abs(sum(g.edges.get("tier:green", {}).values()))
    win_net = abs(sum(g.edges.get("sub:augmentum/win", {}).values()))
    # the ubiquitous node is dampened relative to the discriminating one.
    assert win_net > green_net


def test_oja_normalization_bounds_every_node():
    # even a hammered edge can't run away: each node's outgoing L2 norm ≤ 1.
    attempts = [
        _attempt(f"a{i}", "frontend", "promoted",
                 ["ui/scripts/a.js", "ui/scripts/b.js"], f"{i:02d}")
        for i in range(50)
    ]
    g = A.build_graph(attempts)
    for a, row in g.edges.items():
        norm = sum(w * w for w in row.values()) ** 0.5
        assert norm <= 1.0 + 1e-9, f"{a} norm {norm} exceeds Oja bound"


def test_fold_is_deterministic_and_order_independent_of_input_list():
    # same archive, shuffled input order → identical graph (sort is by created_at,id).
    base = [
        _attempt("a", "frontend", "promoted", ["ui/scripts/x.js"], "01"),
        _attempt("b", "backend", "rolled_back", ["augmentum/y/z.py"], "02"),
        _attempt("c", "frontend", "promoted", ["ui/scripts/x.js"], "03"),
    ]
    g1 = A.build_graph(base)
    g2 = A.build_graph(list(reversed(base)))
    assert g1.edges == g2.edges and g1.activity == g2.activity


def test_top_regions_orders_trust():
    attempts = (
        [_attempt(f"w{i}", "frontend", "promoted", ["ui/good/x.js"], f"1{i}") for i in range(4)]
        + [_attempt(f"l{i}", "backend", "rolled_back", ["augmentum/bad/y.py"], f"2{i}") for i in range(4)]
    )
    g = A.build_graph(attempts)
    regions = dict(g.top_regions(20))
    assert regions["sub:ui/good"] > 0 > regions["sub:augmentum/bad"]


# --- routing hint (router beats cascade) ----------------------------------

def _score(s, c):
    return A.ActivationScore(score=s, confidence=c)


def test_start_rung_neutral_or_positive_starts_cheap():
    assert A.recommend_start_rung(_score(0.0, 0.9), 3) == 0     # neutral
    assert A.recommend_start_rung(_score(0.5, 0.9), 3) == 0     # positive
    assert A.recommend_start_rung(_score(-0.4, 0.9), 3) == 1    # failure-prone → skip 1


def test_start_rung_requires_real_evidence():
    # confidently negative skips; the same score with thin evidence does not.
    assert A.recommend_start_rung(_score(-0.5, 0.9), 3) >= 1
    assert A.recommend_start_rung(_score(-0.5, 0.1), 3) == 0


def test_start_rung_strongly_negative_skips_two():
    assert A.recommend_start_rung(_score(-0.7, 0.9), 4) == 2


def test_start_rung_never_skips_whole_ladder():
    # cap honors max_skip (e.g. keep the frontier rung opt-in): even a terrible
    # region can't skip past the last allowed rung.
    assert A.recommend_start_rung(_score(-1.0, 1.0), 3, max_skip=1) == 1
    assert A.recommend_start_rung(_score(-1.0, 1.0), 1) == 0     # single rung → no skip


# --- calibration (the signal earns the right to act) ----------------------

def test_calibration_cold_start_is_shadow():
    cal = A.backtest_calibration([])
    assert cal.graduated is False and cal.n_predictions == 0
    assert "shadow" in cal.rationale


def test_calibration_thin_archive_stays_shadow():
    # a handful of attempts can't make enough confident calls to graduate.
    attempts = [_attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:02d}")
                for i in range(5)]
    cal = A.backtest_calibration(attempts)
    assert cal.graduated is False
    assert cal.n_predictions < A.CALIB_MIN_GRADUATE


def test_calibration_graduates_on_a_consistent_region():
    # a region that consistently ships → the prior-graph keeps predicting success
    # and keeps being right → the signal graduates (earns the right to act).
    attempts = [_attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:02d}")
                for i in range(20)]
    cal = A.backtest_calibration(attempts)
    assert cal.n_predictions >= A.CALIB_MIN_GRADUATE
    assert cal.accuracy >= A.CALIB_ACC_FLOOR
    assert cal.graduated is True
    assert "graduated" in cal.rationale


def test_calibration_no_future_leakage():
    # every prediction is built only from strictly-prior attempts (prequential):
    # the very first scored attempt has at least MIN_HISTORY priors behind it.
    attempts = [_attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:02d}")
                for i in range(12)]
    cal = A.backtest_calibration(attempts)
    # 12 attempts, first MIN_HISTORY have no/short history → fewer than 12 calls.
    assert cal.n_predictions <= 12 - A.MIN_HISTORY


def test_calibration_truncates_large_archive_loudly():
    attempts = [_attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:04d}")
                for i in range(A.CALIB_CAP + 30)]
    cal = A.backtest_calibration(attempts)
    assert cal.truncated is True
    assert cal.n_attempts == A.CALIB_CAP


async def test_load_graph_and_calibration_single_pass():
    conn = await _conn_with([
        _attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:02d}")
        for i in range(20)
    ])
    try:
        graph, cal = await A.load_graph_and_calibration(conn, user_id="u1")
        assert graph.attempts == 20
        assert cal.graduated is True
    finally:
        await conn.close()


# --- async reader ---------------------------------------------------------

async def _conn_with(attempts):
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    import json
    for a in attempts:
        await conn.execute(
            "INSERT INTO self_edit_attempts (id, user_id, surface, tier, status, "
            "files_changed, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (a["id"], a.get("user_id", "u1"), a["surface"], a.get("tier", "green"),
             a["status"], json.dumps(a["files_changed"]), a["created_at"], a["created_at"]),
        )
    await conn.commit()
    return conn


async def test_load_graph_from_growth_conn():
    conn = await _conn_with([
        _attempt(f"a{i}", "frontend", "promoted", ["ui/scripts/x.js"], f"{i:02d}")
        for i in range(4)
    ])
    try:
        g = await A.load_graph(conn, user_id="u1")
        assert g.attempts == 4
        assert g.score({"shape:frontend", "sub:ui/scripts"}).score > 0
    finally:
        await conn.close()


async def test_load_graph_is_user_scoped():
    conn = await _conn_with([
        {**_attempt("a", "frontend", "promoted", ["ui/scripts/x.js"], "01"), "user_id": "u1"},
        {**_attempt("b", "frontend", "promoted", ["ui/scripts/x.js"], "02"), "user_id": "u2"},
    ])
    try:
        g = await A.load_graph(conn, user_id="u1")
        assert g.attempts == 1
    finally:
        await conn.close()
