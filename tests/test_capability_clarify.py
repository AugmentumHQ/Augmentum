"""The clarify / triage gate — ask before you build (fixes Case B).

Load-bearing:
  - triage classifies ready / clarify / refuse with a closed-world contract;
  - malformed model output repairs once, then degrades to a SAFE clarify (never
    a silent guess);
  - apply_clarifications folds answers into a clarified request for synthesis.
"""

from __future__ import annotations

import json

from augmentum.selfedit.capabilities import (
    apply_clarifications,
    triage_capability_request,
)


async def test_triage_ready():
    async def mi(_p: str) -> str:
        return json.dumps({"status": "ready", "shape": "verb",
                           "normalized": "verb unit.c_to_f: speak the Fahrenheit of args.celsius"})
    res = await triage_capability_request("convert C to F", model_invoke=mi)
    assert res.status == "ready" and res.shape == "verb" and "c_to_f" in res.normalized


async def test_triage_clarify_parses_questions():
    async def mi(_p: str) -> str:
        return json.dumps({"status": "clarify", "questions": [
            {"question": "Which messages?", "header": "Scope", "multiSelect": False,
             "options": [{"label": "Unread only", "description": "..."},
                         {"label": "All", "description": "..."}]}]})
    res = await triage_capability_request("summarize my messages", model_invoke=mi)
    assert res.status == "clarify" and len(res.questions) == 1
    q = res.questions[0]
    assert q.header == "Scope" and len(q.options) == 2 and q.options[0].label == "Unread only"
    # renders to AskUserQuestion shape
    assert q.to_dict()["multiSelect"] is False and len(q.to_dict()["options"]) == 2


async def test_triage_refuse():
    async def mi(_p: str) -> str:
        return json.dumps({"status": "refuse", "reason": "sending email is irreversible/external"})
    res = await triage_capability_request("email my mom when I'm sad", model_invoke=mi)
    assert res.status == "refuse" and "irreversible" in res.reason


async def test_triage_repairs_then_succeeds():
    calls: list[str] = []

    async def mi(p: str) -> str:
        calls.append(p)
        if len(calls) == 1:
            return json.dumps({"status": "clarify", "questions": [
                {"question": "x?", "options": [{"label": "only one"}]}]})  # <2 options → invalid
        return json.dumps({"status": "ready", "normalized": "do x", "shape": "verb"})

    res = await triage_capability_request("x", model_invoke=mi)
    assert res.status == "ready" and len(calls) == 2 and "invalid" in calls[1].lower()


async def test_triage_degrades_to_safe_clarify_on_junk():
    async def mi(_p: str) -> str:
        return "I can't help with that."   # never valid JSON → degrade, not guess
    res = await triage_capability_request("x", model_invoke=mi)
    assert res.status == "clarify" and res.questions   # asks, never silently proceeds


async def test_triage_refuse_without_reason_is_rejected():
    calls = {"n": 0}

    async def mi(_p: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"status": "refuse"})        # no reason → invalid
        return json.dumps({"status": "refuse", "reason": "ok now"})

    res = await triage_capability_request("x", model_invoke=mi)
    assert res.status == "refuse" and res.reason == "ok now" and calls["n"] == 2


def test_apply_clarifications_folds_answers():
    out = apply_clarifications("summarize my messages",
                               {"Which messages?": "Unread only", "Format?": ["bullets", "short"]})
    assert "summarize my messages" in out
    assert "Which messages? → Unread only" in out
    assert "bullets, short" in out


def test_apply_clarifications_noop_without_answers():
    assert apply_clarifications("do x", {}) == "do x"
