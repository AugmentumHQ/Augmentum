"""Evidence-grounding tests — objective composition + the mechanical confirm
oracle (finding-set diff), without needing the real scanners (extract_findings is
monkeypatched). The confirm oracle is the SOTA leap: a resolved-and-none-added
diff makes the verdict reach `verified`/auto-promotable instead of human_required.
"""

from __future__ import annotations

from augmentum.selfedit import evidence as E
from augmentum.selfedit.verifier import FAIL, PASS, SKIP


def _f(key: str, symbol: str = "x", file: str = "ui/styles/a.css") -> E.Finding:
    return E.Finding(key=key, metric="code_quality.dead_css", symbol=symbol, file=file,
                     detail=f"detail for {symbol}")


def _patch_candidate(monkeypatch, findings: list[E.Finding]):
    async def fake_extract(tree_dir, metric_key, *, timeout=180.0):
        return findings
    monkeypatch.setattr(E, "extract_findings", fake_extract)


# --- objective composition (pure) -------------------------------------------

def test_objective_carries_specifics_and_contract():
    findings = [_f("dead_css:btn-foo@ui/styles/a.css", "btn-foo"),
                _f("dead_css:old-card@ui/styles/b.css", "old-card", "ui/styles/b.css")]
    obj = E.build_evidence_objective("Remove a dead CSS rule.", "code_quality",
                                     "dead_css", findings)
    assert "btn-foo" in obj and "old-card" in obj          # the actual flagged items
    assert "ui/styles/a.css" in obj                        # where to look
    assert "search" in obj                                 # told to confirm
    assert "HOW THIS IS CHECKED" in obj                    # the verification contract
    assert "re-run" in obj or "re-run" in obj.lower()


def test_objective_caps_long_lists():
    findings = [_f(f"dead_css:c{i}@a.css", f"c{i}") for i in range(40)]
    obj = E.build_evidence_objective("base", "code_quality", "dead_css", findings, max_items=10)
    assert "and 30 more" in obj


# --- the confirm oracle (finding-set diff) ----------------------------------

async def test_confirm_pass_when_resolved_none_added(monkeypatch):
    # baseline {a,b}; candidate {b} → resolved {a}, added {} → PASS (verified).
    _patch_candidate(monkeypatch, [_f("b")])
    v = E.findings_confirm_verifier(metric_key="code_quality.dead_css",
                                    baseline_keys=frozenset({"a", "b"}))
    res = await v.run({"candidate_dir": "/cand"})
    assert res.status == PASS and res.confirms_intent and res.oracle == "mechanical"
    assert "resolved 1" in res.detail


async def test_confirm_fail_when_finding_added(monkeypatch):
    # baseline {a,b}; candidate {a,b,c} → added {c} → FAIL (catches a 1-for-1 swap
    # the coarse count-based audit would miss).
    _patch_candidate(monkeypatch, [_f("a"), _f("b"), _f("c")])
    v = E.findings_confirm_verifier(metric_key="code_quality.dead_css",
                                    baseline_keys=frozenset({"a", "b"}))
    res = await v.run({"candidate_dir": "/cand"})
    assert res.status == FAIL and res.required is True
    assert "new" in res.detail


async def test_confirm_fail_when_nothing_resolved(monkeypatch):
    # candidate == baseline → resolved nothing → FAIL: the grounded objective was
    # to fix a flagged finding, so a no-resolution change (e.g. a junk helper file)
    # is rejected and the ladder climbs, not settled at human_required.
    _patch_candidate(monkeypatch, [_f("a"), _f("b")])
    v = E.findings_confirm_verifier(metric_key="code_quality.dead_css",
                                    baseline_keys=frozenset({"a", "b"}))
    res = await v.run({"candidate_dir": "/cand"})
    assert res.status == FAIL and res.required is True


async def test_confirm_skip_without_candidate_dir():
    v = E.findings_confirm_verifier(metric_key="code_quality.dead_css",
                                    baseline_keys=frozenset({"a"}))
    res = await v.run({})
    assert res.status == SKIP


# --- enrich_target glue ------------------------------------------------------

async def test_enrich_target_grounds_when_findings_exist(monkeypatch):
    _patch_candidate(monkeypatch, [_f("dead_css:foo@a.css", "foo")])
    enr = await E.enrich_target("/tree", "code_quality", "dead_css", "Remove a dead rule.")
    assert enr.grounded
    assert "foo" in enr.objective
    assert len(enr.verifiers) == 1


async def test_enrich_target_empty_for_unknown_metric(monkeypatch):
    enr = await E.enrich_target("/tree", "code_quality", "mixed_errors", "fix errors")
    assert not enr.grounded and enr.objective == "" and enr.verifiers == []


async def test_enrich_target_empty_when_no_findings(monkeypatch):
    _patch_candidate(monkeypatch, [])
    enr = await E.enrich_target("/tree", "code_quality", "dead_css", "Remove a dead rule.")
    assert not enr.grounded
