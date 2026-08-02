"""Debt advisor tests — LLM agency over the list, fenced by the deterministic floor.

Locks the safety properties: the model can ORDER and REASON, but it can't invent a
finding (hallucinated ids dropped) and it can't reclassify (kind carried from the
deterministic triage). A failing model degrades to available=False so the caller
falls back to the catalog.
"""

from __future__ import annotations

import json

from augmentum.selfedit import debt
from augmentum.selfedit.debt_advisor import _render_activation, advise
from augmentum.selfedit.scanners import parse_audit_json

_REPORT = parse_audit_json(json.dumps({
    "score": 80.0,
    "metrics": {
        "code_quality": {"silent_catches": 4, "dead_css": 9, "missing_css": 110},
        "coverage": {"coverage_gaps": 6},
        "security": {"low": 12, "total": 12},
        "dead_code": {"orphaned_endpoints": 22},
    },
}))


def _triage():
    return debt.triage(_REPORT)


def _chat_returning(obj):
    async def _chat(_prompt: str) -> str:
        return "```json\n" + json.dumps(obj) + "\n```"
    return _chat


async def test_advisor_orders_and_reasons_over_real_findings():
    chat = _chat_returning({
        "summary": "Clear the cheap mechanical wins first; security needs your eyes.",
        "recommendations": [
            {"id": "code_quality.silent_catches", "rationale": "cheap + correctness",
             "approach": "log.warning", "effort": "S", "group": "quality"},
            {"id": "coverage.coverage_gaps", "rationale": "lock behavior",
             "approach": "add tests", "effort": "M"},
            {"id": "security.low", "rationale": "review before touching",
             "approach": "human review", "effort": "L"},
        ],
    })
    adv = await advise(_triage(), chat=chat)
    assert adv.available is True and adv.summary
    ids = [r.target_id for r in adv.recommendations]
    assert ids == ["code_quality.silent_catches", "coverage.coverage_gaps", "security.low"]
    assert [r.rank for r in adv.recommendations] == [1, 2, 3]  # best-first


def test_render_activation_splits_trusted_and_risky():
    block = _render_activation({"top_regions": [
        ["sub:ui/scripts", 0.42], ["sub:augmentum/risky", -0.5], ["tier:green", 0.001],
    ]})
    assert "verified-success regions" in block and "sub:ui/scripts" in block
    assert "repeated-failure regions" in block and "sub:augmentum/risky" in block
    assert "tier:green" not in block  # below the 0.01 noise floor


def test_render_activation_empty_when_no_evidence():
    assert _render_activation(None) == ""
    assert _render_activation({"top_regions": []}) == ""


async def test_advisor_consumes_activation_without_breaking_floor():
    # the activation block is advisory context only — it must not change which ids
    # survive the safety filter (still dropped if not a real finding).
    chat = _chat_returning({"summary": "ok", "recommendations": [
        {"id": "code_quality.silent_catches", "rationale": "r", "approach": "a"},
        {"id": "made.up", "rationale": "r", "approach": "a"},
    ]})
    adv = await advise(_triage(), chat=chat, activation={"top_regions": [
        ["sub:augmentum/selfedit", 0.5]]})
    assert [r.target_id for r in adv.recommendations] == ["code_quality.silent_catches"]


async def test_hallucinated_finding_is_dropped():
    chat = _chat_returning({"summary": "x", "recommendations": [
        {"id": "code_quality.silent_catches", "rationale": "real", "approach": "a"},
        {"id": "made_up.metric", "rationale": "invented", "approach": "b"},
        {"id": "code_quality.totally_fake", "rationale": "invented", "approach": "c"},
    ]})
    adv = await advise(_triage(), chat=chat)
    ids = {r.target_id for r in adv.recommendations}
    assert ids == {"code_quality.silent_catches"}  # only the real flagged id survives


async def test_advisor_cannot_reclassify_kind():
    # The model 'recommends' a structural item; its kind MUST stay structural
    # (carried from the deterministic triage), not whatever the model implies.
    chat = _chat_returning({"summary": "x", "recommendations": [
        {"id": "security.low", "rationale": "let's auto-fix it", "approach": "just patch it"},
        {"id": "code_quality.silent_catches", "rationale": "y", "approach": "z"},
    ]})
    adv = await advise(_triage(), chat=chat)
    by_id = {r.target_id: r for r in adv.recommendations}
    assert by_id["security.low"].kind == debt.KIND_STRUCTURAL       # not auto-lane
    assert by_id["code_quality.silent_catches"].kind == debt.KIND_MECHANICAL


async def test_failing_model_degrades_to_unavailable():
    async def _boom(_prompt: str) -> str:
        raise RuntimeError("model down")
    adv = await advise(_triage(), chat=_boom)
    assert adv.available is False and "unavailable" in adv.note


async def test_empty_triage_is_handled():
    clean = debt.triage(parse_audit_json(json.dumps({"score": 100.0, "metrics": {}})))
    async def _chat(_p):  # pragma: no cover - not called
        return "{}"
    adv = await advise(clean, chat=_chat)
    assert adv.available is True and adv.recommendations == []
