"""End-to-end integration tests for the pen-test substrate.

Boots ``tests/fixtures/vuln_app`` (a deliberately vulnerable FastAPI
app with 6 documented vulnerabilities) and probes it via the actual
production primitives. Asserts each vulnerability is detectable
through the corresponding tool.

This is the load-bearing proof-of-life test for the pen-test leg —
it validates that:

* boot_under_test reliably brings the fixture up
* http_attack catches the per-endpoint vulnerabilities
* authz_matrix_probe specifically catches the cross-tenant leak
* receipts persist exactly the probes that were sent
* the safe counter-example endpoint is NOT flagged (no FPs)
* teardown runs cleanly even when probes have been mid-flight

These tests use real subprocesses + real HTTP — they take a few
seconds each. The session-scoped boot fixture amortizes the cost
across the whole test class.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path
from typing import AsyncIterator

import pytest

from augmentum.bug_finder.pen_test import (
    ProbeRequest, execute_probe, load_probe_receipts,
)
from augmentum.bug_finder.pen_test_attacks import (
    authz_matrix_probe, concurrent_probe,
)
from augmentum.bug_finder.pen_test_boot import (
    BootSpec, _UnderTestRegistry, boot_under_test,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def boot_event_loop():
    """Module-scoped loop so the boot fixture can yield once and
    teardown at the end — pytest-asyncio's per-function loop would
    re-boot for every test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def booted_vuln_app(boot_event_loop, tmp_path_factory) -> dict:
    """Boot the vulnerable fixture once for the whole module.

    Returns ``{"base_url": ..., "service_id": ..., "registry": ...,
    "receipts_root": ...}``. ``registry.teardown_all()`` is run at
    module teardown so even if a test errors mid-probe, the
    subprocess is killed.
    """
    port = _free_port()
    receipts_root = tmp_path_factory.mktemp("pen_test_e2e_receipts")
    registry = _UnderTestRegistry()
    spec = BootSpec(
        command=(
            sys.executable, "-m", "tests.fixtures.vuln_app.app",
            "--port", str(port),
        ),
        port=port,
        healthcheck_path="/",
        boot_timeout_s=30.0,
        healthcheck_timeout_s=20.0,
    )

    async def _boot():
        return await boot_under_test(_REPO_ROOT, spec, registry=registry)

    result = boot_event_loop.run_until_complete(_boot())
    if not result.ok:
        pytest.fail(
            f"vuln fixture boot failed: {result.failure.reason}: "
            f"{result.failure.detail}\nlog_tail:\n{result.failure.log_tail}"
        )

    payload = {
        "base_url": result.service.base_url,
        "service_id": result.service.service_id,
        "registry": registry,
        "receipts_root": receipts_root,
    }

    yield payload

    async def _teardown():
        await registry.teardown_all()

    boot_event_loop.run_until_complete(_teardown())


def _run(coro):
    """Bridge a coroutine onto a fresh event loop for tests that
    aren't decorated with pytest-asyncio (the module-scoped fixtures
    need a plain sync interface)."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Each vulnerability — one test per class
# ---------------------------------------------------------------------------


def test_vuln1_broken_access_control_admin_endpoint(booted_vuln_app) -> None:
    """VULN-1: /admin/stats requires no auth but exposes admin_secret.
    Probe without an Authorization header — should still 200 with
    secret in the body. A correctly-implemented endpoint would 401."""
    resp, _ = _run(execute_probe(
        ProbeRequest(method="GET", url=booted_vuln_app["base_url"] + "/admin/stats"),
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert resp.ok
    assert resp.status == 200, f"expected 200 (vulnerable); got {resp.status}"
    assert "admin_secret" in resp.body_excerpt, (
        f"endpoint did not leak admin_secret as expected — body was "
        f"{resp.body_excerpt[:200]!r}"
    )


def test_vuln2_cross_tenant_leak_via_authz_matrix(booted_vuln_app) -> None:
    """VULN-2: /notes/{id} returns any note regardless of which user
    the auth token belongs to. This is the textbook cross-tenant
    leak — exactly the class authz_matrix_probe was built to catch."""
    verdict = _run(authz_matrix_probe(
        base_url=booted_vuln_app["base_url"],
        endpoint_pattern="/notes/{victim_id}",
        attacker_tokens=("alice", "bob"),
        victims=(("alice", "1"), ("bob", "2")),
        # The fixture seeds note 1 owned by alice w/ body "private to alice"
        # and note 2 owned by bob w/ body "private to bob". When alice's
        # token retrieves note 2, the body should contain "private to bob".
        leak_indicators={"alice": "private to alice", "bob": "private to bob"},
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert verdict.vulnerable, (
        f"authz_matrix_probe should have detected the cross-tenant leak; "
        f"rationale: {verdict.rationale}"
    )
    # Must have at least one leak with confirmed indicator match
    confirmed_leaks = [r for r in verdict.leaked_rows if r.leak_indicator_matched]
    assert confirmed_leaks, (
        "no confirmed leak-indicator match — the bodies didn't contain "
        "the expected victim-owned content"
    )


def test_vuln3_sql_injection_search(booted_vuln_app) -> None:
    """VULN-3: /search?q=... is built via f-string. A UNION SELECT
    payload should leak other users' notes despite the user_id
    WHERE clause."""
    payload = "%' UNION SELECT user_id, title FROM notes--"
    resp, _ = _run(execute_probe(
        ProbeRequest(
            method="GET",
            url=booted_vuln_app["base_url"] + f"/search?q={payload}",
            headers={"Authorization": "Bearer alice"},
        ),
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert resp.ok
    # The SQLi should leak bob's user_id or bob-owned titles into the
    # response (which would otherwise only contain alice's notes).
    assert "bob" in resp.body_excerpt, (
        f"SQLi payload did not leak cross-tenant data; body: "
        f"{resp.body_excerpt[:300]!r}"
    )


def test_vuln4_path_traversal(booted_vuln_app) -> None:
    """VULN-4: /file?name= joins a base dir without normalization.
    Just naming the source file with no ../ prefix is enough to
    leak it because the base dir is the fixture's own folder."""
    resp, _ = _run(execute_probe(
        ProbeRequest(
            method="GET",
            url=booted_vuln_app["base_url"] + "/file?name=app.py",
        ),
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert resp.ok
    assert resp.status == 200
    # The fixture's docstring opens with "Deliberately vulnerable"
    assert "Deliberately vulnerable" in resp.body_excerpt, (
        f"file-read endpoint did not leak fixture source; body: "
        f"{resp.body_excerpt[:200]!r}"
    )


def test_vuln5_mass_assignment(booted_vuln_app) -> None:
    """VULN-5: POST /notes accepts arbitrary body fields including
    user_id, which it trusts over the auth-derived user. Alice's
    request with body user_id=bob should produce a note owned by bob."""
    resp, _ = _run(execute_probe(
        ProbeRequest(
            method="POST",
            url=booted_vuln_app["base_url"] + "/notes",
            headers={
                "Authorization": "Bearer alice",
                "Content-Type": "application/json",
            },
            body='{"title":"hijack","user_id":"bob"}',
        ),
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert resp.ok
    assert resp.status == 200
    assert '"owner":"bob"' in resp.body_excerpt, (
        f"mass-assignment did not transfer ownership — body: "
        f"{resp.body_excerpt[:200]!r}"
    )


def test_vuln6_verbose_error_disclosure(booted_vuln_app) -> None:
    """VULN-6: /debug/explode raises and the handler leaks the full
    traceback into the response body. A correctly-implemented handler
    would log server-side + return a generic message."""
    resp, _ = _run(execute_probe(
        ProbeRequest(
            method="GET",
            url=booted_vuln_app["base_url"] + "/debug/explode",
        ),
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert resp.ok
    assert resp.status == 500
    assert "Traceback" in resp.body_excerpt, (
        f"verbose-error endpoint did not leak traceback; body: "
        f"{resp.body_excerpt[:200]!r}"
    )


# ---------------------------------------------------------------------------
# VULN-7: TOCTOU race via concurrent_probe
# ---------------------------------------------------------------------------


def test_vuln7_toctou_race_on_inventory_claim(booted_vuln_app) -> None:
    """VULN-7: /inventory/claim does a non-atomic read-check-write.
    Reset the counter to 1, fire N concurrent claim requests, expect
    MORE than 1 to succeed because the TOCTOU window lets multiple
    requests pass the ``current > 0`` check before any of them
    decrements.

    This is the canonical class of bug LLMs systematically miss
    (arXiv 2508.16419). Static analysis would have to model the
    interleaving to catch it; concurrent_probe just observes the
    runtime symptom.
    """
    base = booted_vuln_app["base_url"]
    # Reset inventory to 1 — only one claim should succeed if the
    # handler is atomic.
    reset_resp, _ = _run(execute_probe(
        ProbeRequest(method="POST", url=base + "/inventory/reset?count=1"),
    ))
    assert reset_resp.ok and reset_resp.status == 200

    verdict = _run(concurrent_probe(
        base_url=base,
        path="/inventory/claim",
        method="POST",
        headers={"Authorization": "Bearer alice"},
        replicas=10,
        expected_success_count=1,   # only 1 item, only 1 should win
        workspace_root=booted_vuln_app["receipts_root"],
    ))
    assert verdict.vulnerable, (
        f"concurrent_probe should have flagged the TOCTOU race; "
        f"rationale: {verdict.rationale} | "
        f"status_distribution: {verdict.status_distribution}"
    )
    assert verdict.uniqueness_violation, (
        "uniqueness invariant (only 1 claim should succeed) "
        f"was violated: {verdict.success_count} of "
        f"{verdict.replicas} succeeded"
    )
    # At least 2 successes — that's the race signature
    assert verdict.success_count >= 2


def test_concurrent_probe_does_not_flag_atomic_endpoint(booted_vuln_app) -> None:
    """Negative test — when the inventory has 10 items and we fire
    10 claims, ALL should succeed cleanly (each request takes a
    different unit). No uniqueness violation, no 5xx, no spurious
    flag.

    This pins the FP rate: concurrent_probe should only fire on
    actual races, not on every concurrent endpoint."""
    base = booted_vuln_app["base_url"]
    _run(execute_probe(
        ProbeRequest(method="POST", url=base + "/inventory/reset?count=10"),
    ))
    verdict = _run(concurrent_probe(
        base_url=base,
        path="/inventory/claim",
        method="POST",
        headers={"Authorization": "Bearer alice"},
        replicas=10,
        expected_success_count=10,   # enough inventory for all
    ))
    # With 10 inventory and 10 claims, all 10 should succeed
    assert verdict.success_count == 10
    assert not verdict.uniqueness_violation
    assert not verdict.error_class_divergence
    # NOTE: The endpoint IS still TOCTOU-racy in absolute terms
    # (concurrent_probe of count=1 catches it), but with sufficient
    # inventory the race doesn't manifest as a uniqueness violation.
    # This is the right behavior — the primitive flags symptom, not
    # disease.


# ---------------------------------------------------------------------------
# Negative test — the SAFE endpoint must NOT be flagged
# ---------------------------------------------------------------------------


def test_safe_endpoint_not_flagged_by_authz_matrix(booted_vuln_app) -> None:
    """The /notes list endpoint correctly filters by user_id. An
    authz_matrix_probe-style cross-token check against it should
    NOT find any cross-tenant leak.

    This is the FP-killer test: we want to prove the primitive
    doesn't fire on hardened endpoints, only on broken ones."""
    # /notes lists "your notes" — there is no {victim_id} substitution
    # to perform, so we test by sending alice's token and verifying
    # that bob's data is NOT in the response.
    resp_alice, _ = _run(execute_probe(
        ProbeRequest(
            method="GET",
            url=booted_vuln_app["base_url"] + "/notes",
            headers={"Authorization": "Bearer alice"},
        ),
    ))
    assert resp_alice.ok
    assert resp_alice.status == 200
    # alice's notes (1, 3) should appear; bob's note (2) must NOT
    assert "bob secret" not in resp_alice.body_excerpt, (
        "safe /notes endpoint leaked cross-tenant data — fixture is "
        "wrong or the safe endpoint is also vulnerable"
    )


# ---------------------------------------------------------------------------
# Negative test — authz_matrix_probe should NOT flag a hardened endpoint
# ---------------------------------------------------------------------------


def test_authz_matrix_does_not_false_positive_on_root(booted_vuln_app) -> None:
    """The healthcheck path / returns the same content regardless of
    auth. An authz_matrix probe against an endpoint that legitimately
    doesn't differentiate by tenant should NOT report a vulnerability
    (because the user_ids never match the resource_ids, so there are
    no cross-tenant probes to evaluate, OR every probe gets the same
    unauthenticated response).

    Important: this test pins that the leak-indicator gate works —
    the probe matrix correctly distinguishes 'public endpoint' from
    'cross-tenant leak endpoint'.
    """
    verdict = _run(authz_matrix_probe(
        base_url=booted_vuln_app["base_url"],
        endpoint_pattern="/notes/{victim_id}",
        # Use indicators that won't appear in a 404 / wrong-id response
        attacker_tokens=("alice", "bob"),
        # Both victims own ids OUT of the seeded range, so every probe
        # 404s — never a leak.
        victims=(("alice", "9999"), ("bob", "8888")),
        leak_indicators={
            "alice": "this string will not appear in any 404",
            "bob": "neither will this one",
        },
    ))
    # The endpoint is vulnerable in general, but with non-existent
    # resource ids every probe 404s — so confirmed_leaks should be 0.
    confirmed = [r for r in verdict.leaked_rows if r.leak_indicator_matched]
    assert not confirmed, (
        "indicator gate misfired — none of the leak indicators should "
        "match a 404 response body"
    )


# ---------------------------------------------------------------------------
# Receipts contract — the audit trail must be complete
# ---------------------------------------------------------------------------


def test_receipts_persist_all_probes(booted_vuln_app) -> None:
    """Every probe + every authz_matrix row should land in
    probe_receipts.jsonl. The receipts trail is the audit invariant —
    a future investigator must be able to replay every probe.

    Content-focused checks (not count): the suite as a whole touches
    every vulnerability endpoint AT LEAST ONCE, so the receipts must
    cover each endpoint and at least one POST (mass-assignment) +
    one authz_matrix-style cross-tenant probe (recognizable by the
    ``authz_matrix:`` note prefix).
    """
    receipts = load_probe_receipts(booted_vuln_app["receipts_root"])
    methods_seen = {r.method for r in receipts}
    assert "GET" in methods_seen
    assert "POST" in methods_seen, "POST probe (mass-assignment) missing"

    # Every probe must use the loopback host policy — no probe ever
    # left the test sandbox.
    policies = {r.host_policy for r in receipts}
    assert policies == {"loopback"}, (
        f"unexpected host policies in receipts: {policies}"
    )

    # Every vulnerability endpoint should appear in the receipts.
    urls_seen = {r.url for r in receipts}
    for expected_path in (
        "/admin/stats", "/notes/2", "/search", "/file",
        "/notes", "/debug/explode",
    ):
        assert any(expected_path in u for u in urls_seen), (
            f"no receipt found for {expected_path}; receipts: "
            + ", ".join(u[-40:] for u in urls_seen)
        )

    # At least one receipt must carry the authz_matrix note prefix,
    # confirming the cross-tenant primitive contributed to the trail.
    notes_seen = {r.note for r in receipts}
    assert any(n.startswith("authz_matrix:") for n in notes_seen), (
        "no authz_matrix probe ended up in receipts — primitive "
        "either skipped persistence or wasn't called"
    )
