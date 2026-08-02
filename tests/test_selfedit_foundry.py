"""Oracle Foundry (Move 2) — the coverage map, the worklist, the composed
oracle ask, and the Goodhart guard (authored-oracle never auto-promotes).

Locks the load-bearing behaviors:
* the (surface × intent-class) fold and its best-tier strength order;
* rows without a stored verdict tier (git-ingested history) never count as
  interruptions — absence of evidence is not an interruption;
* the worklist threshold keeps one-offs off the list but leaves them visible;
* the composed ask carries the [authored-oracle] marker, the tamper-gate
  constraint, and FULL evidence text (never truncated);
* the marker routes classification to the red-tier class, and
  decide_promotion refuses to auto-promote it even when verified + opted-in.
"""

from __future__ import annotations

from augmentum.selfedit import foundry, promote
from augmentum.selfedit import verifier as V
from augmentum.selfedit.intent import CLASS_AUTHORED_ORACLE, ORACLE_MARKER, classify_intent


def _attempt(*, surface="backend", intent_class="bugfix", tier="human_required",
             status="gated", objective="fix the thing", attempt_id="a1"):
    return {
        "id": attempt_id, "objective": objective, "surface": surface,
        "status": status,
        "gate_verdict": {"tier": tier, "intent_class": intent_class, "results": []},
    }


# ---------------------------------------------------------------------------
# coverage_map — the fold
# ---------------------------------------------------------------------------

def test_coverage_map_folds_one_cell():
    rows = [
        _attempt(tier="human_required", status="promoted", attempt_id="a1"),
        _attempt(tier="verified", status="live", attempt_id="a2"),
        _attempt(tier="human_required", status="rolled_back", attempt_id="a3"),
    ]
    cells = foundry.coverage_map(rows)
    assert len(cells) == 1
    c = cells[0]
    assert (c.surface, c.intent_class) == ("backend", "bugfix")
    assert c.total == 3
    assert c.best_tier == V.TIER_VERIFIED       # strongest ever achieved wins
    assert c.interruptions == 2
    assert c.kept == 2 and c.reverted == 1
    assert dict(c.by_tier) == {"human_required": 2, "verified": 1}


def test_coverage_map_best_tier_strength_order():
    # probable outranks human_required; human_confirmed outranks probable
    rows = [_attempt(tier="human_required"), _attempt(tier="probable")]
    assert foundry.coverage_map(rows)[0].best_tier == V.TIER_PROBABLE
    rows.append(_attempt(tier="human_confirmed"))
    assert foundry.coverage_map(rows)[0].best_tier == V.TIER_HUMAN_CONFIRMED


def test_rows_without_verdict_tier_are_excluded():
    # git-ingested history carries no verifier trace by design — it must not
    # show up as interruptions (that would fabricate human cost).
    rows = [
        {"id": "g1", "surface": "backend", "status": "live", "gate_verdict": {}},
        {"id": "g2", "surface": "backend", "status": "live", "gate_verdict": None},
        _attempt(tier="human_required"),
    ]
    cells = foundry.coverage_map(rows)
    assert len(cells) == 1 and cells[0].total == 1


def test_coverage_map_orders_densest_human_cost_first():
    rows = (
        [_attempt(intent_class="style", surface="frontend", tier="human_required",
                  attempt_id=f"s{i}") for i in range(3)]
        + [_attempt(intent_class="bugfix", tier="human_required", attempt_id="b1")]
    )
    cells = foundry.coverage_map(rows)
    assert cells[0].intent_class == "style" and cells[1].intent_class == "bugfix"


# ---------------------------------------------------------------------------
# worklist — the threshold
# ---------------------------------------------------------------------------

def test_worklist_excludes_covered_and_one_offs():
    rows = (
        # covered cell (has a verified) — never on the worklist, however busy
        [_attempt(intent_class="debt", tier="verified", attempt_id=f"d{i}")
         for i in range(5)]
        # recurring interruptions — worklist material
        + [_attempt(intent_class="style", surface="frontend", tier="human_required",
                    attempt_id=f"s{i}") for i in range(2)]
        # a one-off — visible in the map, off the worklist
        + [_attempt(intent_class="feature", tier="human_required", attempt_id="f1")]
    )
    cells = foundry.coverage_map(rows)
    work = foundry.foundry_worklist(cells)
    assert [c.intent_class for c in work] == ["style"]
    assert {c.intent_class for c in cells} == {"debt", "style", "feature"}


def test_probables_count_toward_the_cluster():
    # a judgment-oracle guess is still uncovered ground — 1 interruption +
    # 1 probable clears the default cluster floor of 2.
    rows = [_attempt(tier="human_required"), _attempt(tier="probable")]
    work = foundry.foundry_worklist(foundry.coverage_map(rows))
    assert len(work) == 1 and work[0].oracle_worthy


# ---------------------------------------------------------------------------
# the composed ask
# ---------------------------------------------------------------------------

def test_oracle_objective_contents():
    long_evidence = "wire the frobnicator setting through all four layers " * 20
    cell = foundry.coverage_map([
        _attempt(tier="human_required", objective=long_evidence, attempt_id="e1"),
        _attempt(tier="human_required", objective="second ask", attempt_id="e2"),
    ])[0]
    text = foundry.oracle_objective(cell)
    assert ORACLE_MARKER in text
    assert "'bugfix'" in text and "'backend'" in text
    assert "register_verifier" in text and 'intent_classes=("bugfix",)' in text
    assert ".claude/" in text                      # the tamper-gate constraint
    assert "must be able to FAIL" in text          # anti-reward-hacking
    assert long_evidence.strip() in text           # evidence NEVER truncated
    assert "second ask" in text


def test_marker_routes_to_red_tier_class():
    intent = classify_intent(foundry.oracle_objective(
        foundry.coverage_map([_attempt(tier="human_required")])[0]))
    assert intent.intent_class == CLASS_AUTHORED_ORACLE
    # no mechanical oracle may confirm the examiner's own authorship
    assert intent.mechanically_confirmable is False


def test_marker_beats_softer_token_heuristics():
    # the ask mentions "test"/"audit"-adjacent words; the marker must still win
    intent = classify_intent(f"{ORACLE_MARKER} author a reproducing test for "
                             "the audit-flagged debt class")
    assert intent.intent_class == CLASS_AUTHORED_ORACLE


# ---------------------------------------------------------------------------
# the Goodhart guard
# ---------------------------------------------------------------------------

def test_authored_oracle_never_auto_promotes_even_verified_and_opted_in():
    verdict = V.Verdict(tier=V.TIER_VERIFIED, passed=True,
                        intent_class=CLASS_AUTHORED_ORACLE)
    d = promote.decide_promotion(verdict, surface="backend",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is False and "authored-oracle" in d.reason


def test_ordinary_verified_backend_still_auto_promotes():
    # the guard is class-scoped — it must not regress the normal green lane
    verdict = V.Verdict(tier=V.TIER_VERIFIED, passed=True, intent_class="bugfix")
    d = promote.decide_promotion(verdict, surface="backend",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is True


# ---------------------------------------------------------------------------
# coverage_summary — the route payload
# ---------------------------------------------------------------------------

def test_coverage_summary_gauge_and_embedded_objectives():
    rows = (
        [_attempt(intent_class="debt", tier="verified", status="live",
                  attempt_id=f"d{i}") for i in range(3)]
        + [_attempt(intent_class="style", surface="frontend", tier="human_required",
                    attempt_id=f"s{i}") for i in range(2)]
    )
    out = foundry.coverage_summary(rows)
    g = out["gauge"]
    assert g["graded_attempts"] == 5
    assert g["verified_attempts"] == 3 and g["verified_share"] == 0.6
    assert g["interruptions"] == 2
    assert g["cells_total"] == 2 and g["cells_covered"] == 1
    assert len(out["worklist"]) == 1
    assert ORACLE_MARKER in out["worklist"][0]["oracle_objective"]


def test_coverage_summary_empty_archive():
    out = foundry.coverage_summary([])
    assert out["cells"] == [] and out["worklist"] == []
    assert out["gauge"]["verified_share"] == 0.0
