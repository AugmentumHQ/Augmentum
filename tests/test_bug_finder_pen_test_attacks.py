"""Unit tests for the Phase 2 pen-test attack primitives.

These run without a real subprocess — we monkeypatch
``pen_test.execute_probe`` to return canned responses so the
primitives can be exercised in isolation. The E2E integration test
``test_bug_finder_pen_test_e2e.py`` covers real-HTTP correctness;
this file covers the orchestration logic + edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from augmentum.bug_finder import pen_test_attacks
from augmentum.bug_finder.agent_tools import (
    AuthzMatrixProbeTool, ConcurrentProbeTool,
)
from augmentum.bug_finder.pen_test import ProbeReceipt, ProbeResponse
from augmentum.bug_finder.pen_test_attacks import (
    AuthzMatrixRow,
    AuthzMatrixVerdict,
    ConcurrentProbeOutcome,
    ConcurrentProbeVerdict,
    authz_matrix_probe,
    concurrent_probe,
)


def _stub_responses_by_url(
    monkeypatch: pytest.MonkeyPatch,
    by_url: dict[str, tuple[int, str]],
    default: tuple[int, str] = (404, "not found"),
) -> list[dict]:
    """Make execute_probe return canned (status, body) per URL.

    Returns a probe-log list the test can inspect to verify which
    URLs were actually hit.
    """
    log: list[dict] = []

    async def _fake_probe(req, *, workspace_root=None, **_kwargs):
        log.append({
            "method": req.method, "url": req.url,
            "headers": dict(req.headers),
        })
        status, body = by_url.get(req.url, default)
        resp = ProbeResponse(
            ok=True, status=status, body_excerpt=body,
            body_size=len(body.encode("utf-8")),
            latency_ms=10,
        )
        receipt = ProbeReceipt(
            ok=True, method=req.method, url=req.url,
            response_status=status, response_body_excerpt=body,
            host_policy="loopback",
        )
        return resp, receipt

    monkeypatch.setattr(pen_test_attacks, "execute_probe", _fake_probe)
    return log


# ---------------------------------------------------------------------------
# authz_matrix_probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authz_matrix_skips_same_tenant_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe where attacker_token equals the victim's user_id is
    the baseline (legitimate) case — must NOT be in the result set."""
    log = _stub_responses_by_url(monkeypatch, {})
    await authz_matrix_probe(
        base_url="http://localhost:8000",
        endpoint_pattern="/notes/{victim_id}",
        attacker_tokens=("alice", "bob"),
        # Alice owns 1, Bob owns 2. Only cross-tenant pairs should
        # be probed: (alice → bob/2) and (bob → alice/1). Same-tenant
        # pairs (alice → alice/1) and (bob → bob/2) must be skipped.
        victims=(("alice", "1"), ("bob", "2")),
    )
    sent_urls = {entry["url"] for entry in log}
    assert sent_urls == {
        "http://localhost:8000/notes/2",   # alice attacking bob/2
        "http://localhost:8000/notes/1",   # bob attacking alice/1
    }


@pytest.mark.asyncio
async def test_authz_matrix_flags_2xx_as_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2xx response on a cross-tenant probe is a leak."""
    _stub_responses_by_url(monkeypatch, {
        "http://localhost/notes/2": (200, "bob private data"),
        "http://localhost/notes/1": (200, "alice private data"),
    })
    verdict = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/notes/{victim_id}",
        attacker_tokens=("alice", "bob"),
        victims=(("alice", "1"), ("bob", "2")),
    )
    assert verdict.vulnerable
    assert len(verdict.leaked_rows) == 2


@pytest.mark.asyncio
async def test_authz_matrix_no_leak_on_403_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hardened endpoint that returns 403 on cross-tenant access
    must NOT be flagged as vulnerable."""
    _stub_responses_by_url(monkeypatch, {
        "http://localhost/notes/2": (403, "forbidden"),
        "http://localhost/notes/1": (403, "forbidden"),
    })
    verdict = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/notes/{victim_id}",
        attacker_tokens=("alice", "bob"),
        victims=(("alice", "1"), ("bob", "2")),
    )
    assert not verdict.vulnerable
    assert "correctly enforces tenant isolation" in verdict.rationale


@pytest.mark.asyncio
async def test_authz_matrix_leak_indicator_upgrades_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a 2xx body contains the victim's indicator string, the
    row's ``leak_indicator_matched`` flag is True — that's the
    upgrade from ambiguous to confirmed leak."""
    _stub_responses_by_url(monkeypatch, {
        # Alice attacks bob/2 → 200 with bob's known indicator
        "http://localhost/notes/2": (
            200, '{"owner":"bob","body":"private to bob"}',
        ),
        # Bob attacks alice/1 → 200 but indicator doesn't match
        "http://localhost/notes/1": (
            200, '{"owner":"alice","body":"unrelated content"}',
        ),
    })
    verdict = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/notes/{victim_id}",
        attacker_tokens=("alice", "bob"),
        victims=(("alice", "1"), ("bob", "2")),
        leak_indicators={
            "bob": "private to bob",
            "alice": "alice secret data not present",
        },
    )
    confirmed = [r for r in verdict.leaked_rows if r.leak_indicator_matched]
    ambiguous = [
        r for r in verdict.leaked_rows
        if not r.leak_indicator_matched
    ]
    assert len(confirmed) == 1, (
        f"expected exactly one confirmed leak (alice attacking bob/2); "
        f"got: {[(r.victim_id, r.leak_indicator_matched) for r in verdict.leaked_rows]}"
    )
    assert len(ambiguous) == 1, "the other 2xx is ambiguous"
    assert confirmed[0].victim_id == "2"


@pytest.mark.asyncio
async def test_authz_matrix_pattern_must_contain_victim_id_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sending a pattern that doesn't reference {victim_id} is a
    caller bug — surface it as a verdict with a clear rationale
    instead of silently sweeping the same URL N times."""
    verdict = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/no-substitution",
        attacker_tokens=("a",),
        victims=(("b", "1"),),
    )
    assert verdict.rows == ()
    assert "victim_id" in verdict.rationale


@pytest.mark.asyncio
async def test_authz_matrix_empty_inputs_returns_clean_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty attacker_tokens or victims must not crash — return a
    verdict with no rows + a rationale."""
    v1 = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=(),
        victims=(("a", "1"),),
    )
    assert v1.rows == ()
    v2 = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=("a",),
        victims=(),
    )
    assert v2.rows == ()


@pytest.mark.asyncio
async def test_authz_matrix_custom_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apps that use ``X-API-Key`` (or other custom headers) must be
    probable. The format-string is parameterized for that exact case."""
    log = _stub_responses_by_url(monkeypatch, {
        "http://localhost/x/2": (200, "ok"),
        "http://localhost/x/1": (200, "ok"),
    })
    await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=("alice", "bob"),
        victims=(("alice", "1"), ("bob", "2")),
        auth_header_name="X-API-Key",
        auth_header_format="{token}",
    )
    assert log[0]["headers"].get("X-API-Key") in {"alice", "bob"}
    # No standard Authorization header sent
    assert "Authorization" not in log[0]["headers"]


@pytest.mark.asyncio
async def test_authz_matrix_handles_multiple_victims_per_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same user owning multiple resources must produce one
    probe per (attacker, victim_resource) pair."""
    _stub_responses_by_url(monkeypatch, {})
    verdict = await authz_matrix_probe(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=("alice", "bob"),
        victims=(
            ("alice", "1"), ("alice", "3"),
            ("bob", "2"),
        ),
    )
    # alice attacks bob/2; bob attacks alice/1 and alice/3 → 3 rows
    assert len(verdict.rows) == 3


# ---------------------------------------------------------------------------
# Tool wrapper (AuthzMatrixProbeTool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_validates_required_args() -> None:
    tool = AuthzMatrixProbeTool()
    # Missing everything
    res = await tool.execute()
    assert not res.success
    assert res.validation_error


@pytest.mark.asyncio
async def test_tool_validates_attacker_and_victim_non_empty() -> None:
    tool = AuthzMatrixProbeTool()
    res = await tool.execute(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=[],
        victims=[["a", "1"]],
    )
    assert not res.success
    assert res.validation_error
    res = await tool.execute(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=["a"],
        victims=[],
    )
    assert not res.success


@pytest.mark.asyncio
async def test_tool_emits_parseable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_responses_by_url(monkeypatch, {
        "http://localhost/x/2": (403, "forbidden"),
        "http://localhost/x/1": (403, "forbidden"),
    })
    tool = AuthzMatrixProbeTool()
    res = await tool.execute(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=["alice", "bob"],
        victims=[["alice", "1"], ["bob", "2"]],
    )
    assert res.success
    data = json.loads(res.output)
    assert data["vulnerable"] is False
    assert data["leaked_count"] == 0
    assert len(data["rows"]) == 2


@pytest.mark.asyncio
async def test_tool_propagates_vulnerable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_responses_by_url(monkeypatch, {
        "http://localhost/x/2": (200, "bob data"),
        "http://localhost/x/1": (200, "alice data"),
    })
    tool = AuthzMatrixProbeTool()
    res = await tool.execute(
        base_url="http://localhost",
        endpoint_pattern="/x/{victim_id}",
        attacker_tokens=["alice", "bob"],
        victims=[["alice", "1"], ["bob", "2"]],
    )
    data = json.loads(res.output)
    assert data["vulnerable"] is True
    assert data["leaked_count"] == 2
    assert res.metadata["vulnerable"] is True


# ---------------------------------------------------------------------------
# Role-isolation invariant — still holds with the new tool
# ---------------------------------------------------------------------------


def test_authz_matrix_probe_in_pen_test_canonical_set() -> None:
    from augmentum.agents.tools import (
        DETECTOR_TOOL_NAMES,
        FIXER_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        PEN_TEST_TOOL_NAMES,
        PEN_TESTER_TOOL_NAMES,
        PLANNER_TOOL_NAMES,
        READ_ONLY_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
    )
    assert "authz_matrix_probe" in PEN_TEST_TOOL_NAMES
    assert "authz_matrix_probe" in PEN_TESTER_TOOL_NAMES
    for role_names in (
        READ_ONLY_TOOL_NAMES,
        PLANNER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
        FIXER_TOOL_NAMES,
    ):
        assert "authz_matrix_probe" not in role_names


# ---------------------------------------------------------------------------
# concurrent_probe
# ---------------------------------------------------------------------------


def _stub_concurrent_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[tuple[int, str]],
) -> list[str]:
    """Make each call to execute_probe return the NEXT canned response
    from ``responses``. Useful for simulating a TOCTOU race where the
    first N succeed and later ones fail."""
    sequence_idx = {"i": 0}
    log: list[str] = []

    async def _fake_probe(req, *, workspace_root=None, **_kwargs):
        i = sequence_idx["i"]
        sequence_idx["i"] += 1
        status, body = responses[min(i, len(responses) - 1)]
        log.append(req.url)
        resp = ProbeResponse(
            ok=True, status=status, body_excerpt=body,
            body_size=len(body), latency_ms=5,
        )
        receipt = ProbeReceipt(
            ok=True, method=req.method, url=req.url,
            response_status=status, host_policy="loopback",
        )
        return resp, receipt

    monkeypatch.setattr(pen_test_attacks, "execute_probe", _fake_probe)
    return log


@pytest.mark.asyncio
async def test_concurrent_probe_no_violation_on_atomic_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An atomic endpoint: only ``expected_success_count`` requests
    return 2xx, the rest return 409. Must NOT be flagged as
    vulnerable."""
    # 1 success, 9 conflicts — the textbook atomic claim outcome
    _stub_concurrent_responses(
        monkeypatch,
        [(200, '{"claimed":true}')]
        + [(409, '{"error":"sold out"}')] * 9,
    )
    verdict = await concurrent_probe(
        base_url="http://localhost",
        path="/inventory/claim",
        replicas=10,
        expected_success_count=1,
    )
    assert not verdict.vulnerable
    assert verdict.success_count == 1
    assert not verdict.uniqueness_violation
    assert "concurrency-safe" in verdict.rationale


@pytest.mark.asyncio
async def test_concurrent_probe_flags_uniqueness_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-atomic endpoint: 5 of 10 requests succeed when only 1
    should. Must flag uniqueness_violation."""
    _stub_concurrent_responses(
        monkeypatch,
        [(200, '{"claimed":true}')] * 5
        + [(409, '{"error":"sold out"}')] * 5,
    )
    verdict = await concurrent_probe(
        base_url="http://localhost",
        path="/inventory/claim",
        replicas=10,
        expected_success_count=1,
    )
    assert verdict.vulnerable
    assert verdict.uniqueness_violation
    assert verdict.success_count == 5
    assert "uniqueness violation" in verdict.rationale


@pytest.mark.asyncio
async def test_concurrent_probe_flags_5xx_under_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the handler can't cope with concurrency it returns 5xx —
    that's a real bug class (deadlock, exception on contention)."""
    _stub_concurrent_responses(
        monkeypatch,
        [(200, "ok")] * 3 + [(500, "deadlock")] * 7,
    )
    verdict = await concurrent_probe(
        base_url="http://localhost", path="/x",
        replicas=10, expected_success_count=10,  # expect all to succeed
    )
    assert verdict.vulnerable
    assert verdict.error_class_divergence
    assert verdict.error_count == 7


@pytest.mark.asyncio
async def test_concurrent_probe_inconsistency_without_other_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status diverges across identical requests (non-deterministic)
    but no uniqueness or error signal. Flag inconsistency only —
    that's a weaker signal but still useful."""
    _stub_concurrent_responses(
        monkeypatch,
        [(200, "ok"), (200, "ok"), (200, "ok"),
         (404, "not found"), (404, "not found")],
    )
    verdict = await concurrent_probe(
        base_url="http://localhost", path="/x",
        replicas=5, expected_success_count=5,  # 3 of 5 succeed
    )
    # 3 < 5, so uniqueness is fine (we expected up to 5)
    # But 200s and 404s mixed = inconsistency
    assert verdict.inconsistency
    assert "diverged" in verdict.rationale.lower()


@pytest.mark.asyncio
async def test_concurrent_probe_replicas_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primitive must not let the LLM accidentally fire 10,000
    concurrent requests. The tool wrapper clamps; this test confirms
    the underlying primitive fires exactly the requested count and
    no more."""
    log: list[str] = []
    request_count = {"i": 0}

    async def _fake_probe(req, *, workspace_root=None, **_kwargs):
        request_count["i"] += 1
        log.append(req.url)
        resp = ProbeResponse(ok=True, status=200, body_excerpt="ok")
        receipt = ProbeReceipt(
            ok=True, method=req.method, url=req.url,
            response_status=200, host_policy="loopback",
        )
        return resp, receipt

    monkeypatch.setattr(pen_test_attacks, "execute_probe", _fake_probe)
    await concurrent_probe(
        base_url="http://localhost", path="/x",
        replicas=7, expected_success_count=7,
    )
    assert request_count["i"] == 7


# ---------------------------------------------------------------------------
# ConcurrentProbeTool wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_tool_validates_required_args() -> None:
    tool = ConcurrentProbeTool()
    res = await tool.execute()
    assert not res.success
    assert res.validation_error


@pytest.mark.asyncio
async def test_concurrent_tool_emits_parseable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_concurrent_responses(
        monkeypatch,
        [(200, "ok")] * 2 + [(409, "conflict")] * 3,
    )
    tool = ConcurrentProbeTool()
    res = await tool.execute(
        base_url="http://localhost", path="/x",
        replicas=5, expected_success_count=1,
    )
    assert res.success
    data = json.loads(res.output)
    assert data["uniqueness_violation"] is True
    assert data["success_count"] == 2
    assert data["replicas"] == 5


@pytest.mark.asyncio
async def test_concurrent_tool_clamps_replicas_to_safe_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool's input_schema specifies maximum 50 replicas. The
    execute method must enforce that — an LLM that requests 1000
    must get clamped, not allowed."""
    request_count = {"i": 0}

    async def _fake_probe(req, *, workspace_root=None, **_kwargs):
        request_count["i"] += 1
        resp = ProbeResponse(ok=True, status=200, body_excerpt="ok")
        receipt = ProbeReceipt(
            ok=True, method=req.method, url=req.url,
            response_status=200, host_policy="loopback",
        )
        return resp, receipt

    monkeypatch.setattr(pen_test_attacks, "execute_probe", _fake_probe)
    tool = ConcurrentProbeTool()
    await tool.execute(
        base_url="http://localhost", path="/x", replicas=10000,
    )
    # Must NEVER fire more than 50
    assert request_count["i"] <= 50


def test_concurrent_probe_in_pen_test_canonical_set() -> None:
    from augmentum.agents.tools import (
        PEN_TEST_TOOL_NAMES, PEN_TESTER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES, FIXER_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES, LEAD_TOOL_NAMES,
        PLANNER_TOOL_NAMES, READ_ONLY_TOOL_NAMES, VERIFIER_TOOL_NAMES,
    )
    assert "concurrent_probe" in PEN_TEST_TOOL_NAMES
    assert "concurrent_probe" in PEN_TESTER_TOOL_NAMES
    for role_names in (
        READ_ONLY_TOOL_NAMES, PLANNER_TOOL_NAMES, DETECTOR_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES, LEAD_TOOL_NAMES,
        VERIFIER_TOOL_NAMES, FIXER_TOOL_NAMES,
    ):
        assert "concurrent_probe" not in role_names
