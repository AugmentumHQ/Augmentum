"""Lead agent decision-parser + prompt-renderer tests.

The decision loop itself requires a backend + queue and is exercised
when the orchestrator is wired through it. Here we cover:

  * Parser: tolerates fenced JSON, bare JSON, last-block-wins.
  * Schema validation: each action has its own required fields.
  * Prompt renderer: includes queue summary + findings section, never
    crashes on edge cases (empty queue, no findings).
"""

from __future__ import annotations

import json
import time

from augmentum.bug_finder.findings import (
    ClaimSignature, Finding, FindingStatus, Severity,
)
from augmentum.bug_finder.lead import (
    LeadDecision,
    parse_lead_decision,
    render_lead_prompt,
)
from augmentum.bug_finder.task_queue import BugFinderTask


# ---------------------------------------------------------------------------
# parse_lead_decision
# ---------------------------------------------------------------------------


def _block(payload: dict) -> str:
    return "thinking...\n\n```json\n" + json.dumps(payload) + "\n```\n"


def test_parse_dispatch_action() -> None:
    d = parse_lead_decision(_block({
        "action": "dispatch",
        "task_id": "tsk_abc",
        "rationale": "highest priority pending",
    }))
    assert d is not None
    assert d.action == "dispatch"
    assert d.task_id == "tsk_abc"
    assert d.rationale.startswith("highest")
    assert not d.is_terminal
    assert d.is_valid


def test_parse_enqueue_action() -> None:
    d = parse_lead_decision(_block({
        "action": "enqueue",
        "new_task": {
            "kind": "investigate",
            "target": {"thread_anchor": "auth.py:login"},
            "reason": "same pattern as finding #3",
            "priority": 7,
        },
        "rationale": "compound the auth-bypass thread",
    }))
    assert d is not None
    assert d.action == "enqueue"
    assert d.new_task is not None
    assert d.new_task["kind"] == "investigate"


def test_parse_done_action() -> None:
    d = parse_lead_decision(_block({
        "action": "done",
        "rationale": "queue empty, goal satisfied",
    }))
    assert d is not None
    assert d.is_terminal


def test_parse_drop_task_action() -> None:
    d = parse_lead_decision(_block({
        "action": "drop_task",
        "task_id": "tsk_x",
        "rationale": "later evidence shows bug is elsewhere",
    }))
    assert d is not None
    assert d.action == "drop_task"


def test_parse_drop_finding_action() -> None:
    d = parse_lead_decision(_block({
        "action": "drop_finding",
        "finding_id": "fnd_x",
        "rationale": "low-confidence speculative",
    }))
    assert d is not None
    assert d.action == "drop_finding"


def test_parse_empty_returns_none() -> None:
    assert parse_lead_decision("") is None
    assert parse_lead_decision("no fenced json here") is None


def test_parse_malformed_json_returns_none() -> None:
    assert parse_lead_decision("```json\n{not valid}\n```") is None


def test_parse_unknown_action_returns_none() -> None:
    """Unrecognized action values shouldn't translate to LeadDecision."""
    assert parse_lead_decision(_block({"action": "panic"})) is None


def test_parse_dispatch_without_task_id_returns_none() -> None:
    """Schema rejection: dispatch needs task_id."""
    assert parse_lead_decision(_block({"action": "dispatch"})) is None


def test_parse_enqueue_without_new_task_returns_none() -> None:
    assert parse_lead_decision(_block({"action": "enqueue"})) is None


def test_parse_enqueue_with_invalid_new_task_returns_none() -> None:
    """new_task must be a dict with kind + target dict."""
    assert parse_lead_decision(_block({
        "action": "enqueue",
        "new_task": {"kind": "investigate"},  # no target
    })) is None


def test_parse_last_json_block_wins() -> None:
    """Models often emit drafts before the final commit — last wins."""
    raw = (
        "Draft:\n```json\n" + json.dumps({"action": "dispatch", "task_id": "DRAFT"}) + "\n```\n"
        "Final:\n```json\n" + json.dumps({"action": "dispatch", "task_id": "FINAL"}) + "\n```\n"
    )
    d = parse_lead_decision(raw)
    assert d is not None
    assert d.task_id == "FINAL"


def test_parse_tolerates_bare_json_object() -> None:
    d = parse_lead_decision(json.dumps({
        "action": "done", "rationale": "x",
    }))
    assert d is not None
    assert d.is_terminal


# ---------------------------------------------------------------------------
# LeadDecision dataclass
# ---------------------------------------------------------------------------


def test_lead_decision_is_terminal_predicate() -> None:
    assert not LeadDecision(action="dispatch", task_id="x").is_terminal
    assert LeadDecision(action="done").is_terminal


def test_lead_decision_is_frozen() -> None:
    d = LeadDecision(action="done")
    try:
        d.action = "x"  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("LeadDecision should be frozen")


# ---------------------------------------------------------------------------
# render_lead_prompt
# ---------------------------------------------------------------------------


def _task(task_id: str, kind: str, file: str, priority: int = 5) -> BugFinderTask:
    return BugFinderTask(
        task_id=task_id, user_id="u", run_id="r", workspace_id="",
        kind=kind, target={"file": file, "function": "foo"},
        reason="planner", priority=priority,
        status="pending", parent_task_id="", created_by="planner",
        result_summary="", created_at=int(time.time()), completed_at=0,
    )


def _finding(fid: str, sev: str, status: str, claim: str) -> Finding:
    return Finding(
        id=fid, file="x.py", function="foo",
        claim=claim,
        claim_signature=ClaimSignature.MISSING_VALIDATION.value,
        severity=sev,
        evidence_paths=("x.py:1",),
        status=status,
    )


def test_render_lead_prompt_empty_queue_and_findings() -> None:
    out = render_lead_prompt(
        iteration=1, max_iterations=20,
        queue=[], findings=[],
        user_goal_block="",
        tokens_remaining=100_000,
    )
    assert "Iteration: 1/20" in out
    assert "queue empty" in out
    assert "explore mode" in out.lower()


def test_render_lead_prompt_includes_queue_and_findings() -> None:
    queue = [
        _task("tsk_1", "detect", "auth.py", priority=8),
        _task("tsk_2", "detect", "billing.py", priority=4),
    ]
    findings = [
        _finding("fnd_1", Severity.HIGH.value, FindingStatus.CONFIRMED.value,
                 "auth bypass via missing session check"),
        _finding("fnd_2", Severity.LOW.value, FindingStatus.SPECULATIVE.value,
                 "logging inconsistency"),
    ]
    out = render_lead_prompt(
        iteration=3, max_iterations=20,
        queue=queue, findings=findings,
        user_goal_block="## User goal\nfind auth bypasses",
        tokens_remaining=50_000,
    )
    # Queue
    assert "PENDING (2)" in out
    assert "auth.py" in out
    # Findings section — highest severity first
    assert "fnd_1" in out
    assert "HIGH" in out.upper()
    # Token meter
    assert "50,000" in out
    # User goal echoed
    assert "find auth bypasses" in out


def test_render_lead_prompt_truncates_finding_claim() -> None:
    """Long claims should be truncated to keep the prompt bounded."""
    long_claim = "x" * 400
    findings = [
        _finding("fnd_long", Severity.MEDIUM.value, FindingStatus.SPECULATIVE.value,
                 long_claim),
    ]
    out = render_lead_prompt(
        iteration=1, max_iterations=20,
        queue=[], findings=findings,
        user_goal_block="",
        tokens_remaining=100_000,
    )
    # Claim is rendered, but not full length
    assert "fnd_long" in out
    assert long_claim not in out  # truncated
