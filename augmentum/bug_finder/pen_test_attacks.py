"""Phase 2 higher-level pen-test attack primitives.

``http_attack`` (Phase 1a) is the load-bearing primitive — one request
in, one response out. The Phase 2 primitives are compositions on top
of it that target specific bug classes that benefit from sequenced
probing:

* ``authz_matrix_probe`` — given two user identities + a victim's
  resource id, systematically tests cross-tenant access. The canonical
  "hard to detect statically, easy to verify dynamically" class.
* (Future) ``injection_sweep`` — fires a curated payload corpus
  (SQLi / XSS / path traversal / SSTI) against one endpoint and
  reports which ones triggered anomalous responses.
* (Future) ``concurrent_probe`` — fires N parallel requests against
  one endpoint to surface TOCTOU / uniqueness-invariant races.
* (Future) ``session_replay_attack`` — replays a captured session
  with auth-related mutations to find session-handling flaws.

The primitives share the safety model of ``http_attack``: host
allow-list, bounded sizes, receipts trail, no implicit external
hosts. Each one is implemented as a thin orchestrator over
``execute_probe`` so the audit trail stays consistent.

This module's API is the contract — the LLM-callable tools wrap it
in ``agent_tools.py``. Tests in
``tests/test_bug_finder_pen_test_attacks.py`` pin the contract.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from augmentum.bug_finder.pen_test import (
    ProbeRequest,
    ProbeResponse,
    execute_probe,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# authz_matrix_probe — cross-tenant access check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthzMatrixRow:
    """One cell of the authz-probe matrix.

    ``attacker_token`` is the auth value sent in the probe header.
    ``victim_id`` is the resource owner the URL points to. The
    primitive is interested in the case where they DIFFER: that's
    the cross-tenant test.
    """

    attacker_token: str
    victim_id: str
    url: str
    status: int
    body_excerpt: str
    latency_ms: int
    leak_indicator_matched: bool = False
    error: str = ""


@dataclass(frozen=True)
class AuthzMatrixVerdict:
    """Verdict from one matrix sweep.

    ``leaked_rows`` collects every cell where the attacker fetched a
    victim's resource AND received a 2xx response (or matched the
    ``leak_indicator``). The rationale string summarizes for the LLM.
    """

    endpoint_pattern: str
    rows: tuple[AuthzMatrixRow, ...]
    leaked_rows: tuple[AuthzMatrixRow, ...]
    rationale: str

    @property
    def vulnerable(self) -> bool:
        return len(self.leaked_rows) > 0


def _matched_leak_indicator(
    body: str, leak_indicator: str,
) -> bool:
    """``True`` when a leak-indicator string substring-matches the
    response body.

    The caller supplies an indicator that should ONLY appear if the
    cross-tenant read succeeded — e.g. the victim's known title, the
    victim's user_id literal. Empty indicator means "treat any 2xx
    as a leak"."""
    if not leak_indicator:
        return False
    return leak_indicator in body


async def authz_matrix_probe(
    *,
    base_url: str,
    endpoint_pattern: str,
    auth_header_name: str = "Authorization",
    auth_header_format: str = "Bearer {token}",
    attacker_tokens: tuple[str, ...],
    victims: tuple[tuple[str, str], ...],
    leak_indicators: dict[str, str] | None = None,
    workspace_root: Path | None = None,
    finding_id: str = "",
    run_id: str = "",
) -> AuthzMatrixVerdict:
    """Systematically probe cross-tenant access against one endpoint.

    For each (attacker_token, victim_id) pair where attacker_token's
    user != victim_id, send a request to ``endpoint_pattern.format(
    victim_id=...)`` carrying attacker_token. A 2xx response is
    treated as a leak; an explicit ``leak_indicators[victim_id]``
    substring match upgrades that to "confirmed leak".

    Args:
        base_url: The booted app's base URL (from boot_under_test).
        endpoint_pattern: Path template, e.g. ``/notes/{victim_id}``.
            Must contain ``{victim_id}``.
        auth_header_name: Header to carry the token (usually
            ``Authorization``). Configurable for apps using custom
            headers (X-Auth-Token, etc.).
        auth_header_format: Format string for the header value.
            ``{token}`` will be replaced with the actual token.
        attacker_tokens: Auth tokens of users whose access this
            primitive tests. Most useful with 2+ tokens.
        victims: Tuples of (user_id, resource_id). The user_id is
            used to filter out same-tenant requests; the resource_id
            substitutes into the pattern. Example:
            ``(("alice", "1"), ("bob", "2"))`` says "alice owns
            resource 1; bob owns resource 2".
        leak_indicators: Optional ``{user_id -> indicator-string}``
            map. When set, a 2xx that contains the indicator is
            confirmed-leaked; a 2xx that doesn't is still tagged as
            ambiguous-leak.
        workspace_root: Receipts trail destination. ``None`` = no
            persistence.
        finding_id / run_id: Audit-trail linkage.

    Returns ``AuthzMatrixVerdict`` with every row and the leaked
    subset. ``vulnerable`` is the headline boolean.
    """
    if "{victim_id}" not in endpoint_pattern:
        return AuthzMatrixVerdict(
            endpoint_pattern=endpoint_pattern,
            rows=(),
            leaked_rows=(),
            rationale=(
                "endpoint_pattern must contain "
                "'{victim_id}' for the substitution"
            ),
        )
    if not attacker_tokens or not victims:
        return AuthzMatrixVerdict(
            endpoint_pattern=endpoint_pattern,
            rows=(),
            leaked_rows=(),
            rationale="empty attacker_tokens or victims — nothing to probe",
        )

    leak_indicators = leak_indicators or {}

    # Build the matrix: every (attacker_token, victim_user, victim_resource)
    # triple where attacker_token's identity differs from victim_user.
    # We don't try to introspect what user a token maps to — the caller
    # is expected to use token strings that match the user_ids (the
    # common test pattern) OR supply tokens that are recognizable as
    # different from each victim's user_id.
    rows: list[AuthzMatrixRow] = []
    base = base_url.rstrip("/")
    for token in attacker_tokens:
        for victim_user, victim_resource in victims:
            # Skip same-tenant requests — those are baseline behavior,
            # not cross-tenant exploit attempts.
            if token == victim_user:
                continue
            path = endpoint_pattern.format(victim_id=victim_resource)
            url = base + ("/" + path.lstrip("/"))
            header_value = auth_header_format.format(token=token)
            req = ProbeRequest(
                method="GET", url=url,
                headers={auth_header_name: header_value},
                note=(
                    f"authz_matrix: attacker={token}, "
                    f"victim={victim_user}/{victim_resource}"
                ),
                finding_id=finding_id, run_id=run_id,
            )
            resp, _ = await execute_probe(
                req, workspace_root=workspace_root,
            )
            leak_indicator = leak_indicators.get(victim_user, "")
            leak_match = _matched_leak_indicator(
                resp.body_excerpt, leak_indicator,
            )
            rows.append(AuthzMatrixRow(
                attacker_token=token,
                victim_id=str(victim_resource),
                url=url,
                status=resp.status,
                body_excerpt=resp.body_excerpt[:400],
                latency_ms=resp.latency_ms,
                leak_indicator_matched=leak_match,
                error=resp.error,
            ))

    # Anything 2xx is suspicious; indicator-match upgrades to confirmed.
    leaked = tuple(
        r for r in rows
        if 200 <= r.status < 300
    )

    if not rows:
        rationale = (
            "no cross-tenant pairs to test — every attacker_token "
            "matched a victim user_id, so all probes were same-tenant"
        )
    elif not leaked:
        rationale = (
            f"tested {len(rows)} cross-tenant pairs; every one was "
            "rejected (3xx/4xx/5xx). The endpoint correctly enforces "
            "tenant isolation."
        )
    else:
        any_confirmed = any(r.leak_indicator_matched for r in leaked)
        rationale = (
            f"{len(leaked)} of {len(rows)} cross-tenant probes "
            f"returned 2xx; "
            + ("at least one body matched the leak indicator (confirmed leak)"
               if any_confirmed
               else "no indicator string supplied or none matched "
                    "(ambiguous — body may or may not contain victim data)")
        )

    return AuthzMatrixVerdict(
        endpoint_pattern=endpoint_pattern,
        rows=tuple(rows),
        leaked_rows=leaked,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# concurrent_probe — TOCTOU / race / uniqueness-invariant detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrentProbeOutcome:
    """One response captured in a concurrent sweep."""

    status: int
    body_excerpt: str
    latency_ms: int
    error: str = ""


@dataclass(frozen=True)
class ConcurrentProbeVerdict:
    """Result of one concurrent sweep against a single endpoint.

    Three signals distinguish "broken" from "fine":

    * ``inconsistency`` — non-deterministic status divergence across
      otherwise identical requests. A correctly-implemented endpoint
      should yield the same status for the same request.
    * ``uniqueness_violation`` — when the caller expects a "claim"
      semantic (only one request should succeed), more 2xx responses
      than the supplied ``expected_success_count`` indicates a TOCTOU
      race on the underlying invariant.
    * ``error_class_divergence`` — 5xx responses suggest the endpoint
      can't handle concurrency at all (deadlock, exception under load).
    """

    endpoint: str
    replicas: int
    outcomes: tuple[ConcurrentProbeOutcome, ...]
    status_distribution: dict[int, int]
    success_count: int
    error_count: int
    expected_success_count: int
    inconsistency: bool
    uniqueness_violation: bool
    error_class_divergence: bool
    rationale: str

    @property
    def vulnerable(self) -> bool:
        return (
            self.uniqueness_violation
            or self.error_class_divergence
        )


async def concurrent_probe(
    *,
    base_url: str,
    path: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: str = "",
    replicas: int = 10,
    expected_success_count: int = 1,
    workspace_root: Path | None = None,
    finding_id: str = "",
    run_id: str = "",
) -> ConcurrentProbeVerdict:
    """Fire ``replicas`` identical requests in parallel and analyze
    the response distribution.

    Targets the bug class published research (arXiv 2508.16419) flags
    as systematically missed by LLM static analysis: TOCTOU races,
    uniqueness-invariant violations, scheduler-timing-dependent bugs.
    Static review can't model concurrency outcomes; only timed
    parallel requests can.

    ``expected_success_count`` is the caller's claim about how many
    2xx responses SHOULD occur if the endpoint is correctly atomic.
    For ``POST /claim_one_item`` against a 1-item inventory, this is
    1. When more than ``expected_success_count`` requests get 2xx, the
    invariant has been violated.

    Args:
        base_url: Booted target's base URL.
        path: Path to attack (e.g. ``/inventory/claim``).
        method: HTTP method (default POST since most race-prone
            endpoints are mutations).
        headers: Common headers for all replicas.
        body: Common body for all replicas.
        replicas: Parallel request count. 10 is enough to surface most
            races; very wide windows may need more.
        expected_success_count: Caller's atomicity claim.
        workspace_root: Receipts trail destination.
        finding_id / run_id: Audit-trail linkage.
    """
    headers = headers or {}
    url = base_url.rstrip("/") + "/" + path.lstrip("/")

    async def _one(seq: int) -> ConcurrentProbeOutcome:
        req = ProbeRequest(
            method=method, url=url,
            headers=dict(headers), body=body,
            note=f"concurrent_probe:{seq}/{replicas}",
            finding_id=finding_id, run_id=run_id,
        )
        resp, _ = await execute_probe(
            req, workspace_root=workspace_root,
        )
        return ConcurrentProbeOutcome(
            status=resp.status,
            body_excerpt=resp.body_excerpt[:200],
            latency_ms=resp.latency_ms,
            error=resp.error,
        )

    # Fire them all together. asyncio.gather with default kwargs
    # schedules all coroutines as concurrently as the event loop
    # allows; for httpx-based requests this is genuine parallelism
    # to the target.
    outcomes = await asyncio.gather(*[_one(i) for i in range(replicas)])
    outcomes_tuple = tuple(outcomes)

    status_dist = Counter(o.status for o in outcomes_tuple)
    success_count = sum(
        1 for o in outcomes_tuple if 200 <= o.status < 300
    )
    error_count = sum(
        1 for o in outcomes_tuple if o.status >= 500
    )

    # Heuristic 1: distinct non-error statuses — endpoint isn't
    # deterministic under load. (Many endpoints legitimately return
    # different statuses for different inputs; here every request is
    # identical, so divergence IS the signal.)
    non_error_statuses = {
        s for s in status_dist
        if s and s < 500
    }
    inconsistency = len(non_error_statuses) > 1

    # Heuristic 2: more 2xx than the caller's atomicity claim.
    uniqueness_violation = success_count > expected_success_count

    # Heuristic 3: any 5xx (server can't handle concurrency).
    error_class_divergence = error_count > 0

    # Rationale construction — only flag a problem when we actually
    # have one. The 200/409 mix on an atomic claim endpoint is the
    # EXPECTED outcome, not an inconsistency signal — that's why we
    # cross-check against ``expected_success_count`` before treating
    # status divergence as suspicious.
    parts: list[str] = []
    if uniqueness_violation:
        parts.append(
            f"uniqueness violation: {success_count} of {replicas} "
            f"requests succeeded (caller expected at most "
            f"{expected_success_count}) — non-atomic invariant"
        )
    if error_class_divergence:
        parts.append(
            f"server errors under load: {error_count} of {replicas} "
            f"returned 5xx — handler not concurrency-safe"
        )

    is_expected_atomic_outcome = (
        success_count == expected_success_count
        and error_count == 0
        and not uniqueness_violation
    )

    # Inconsistency is interesting ONLY when it's not the expected
    # atomic-claim mix. Two cases qualify:
    #   1. More than 2 distinct non-error statuses (3+ classes is
    #      unusual for any well-defined endpoint).
    #   2. Status divergence WITHOUT a matching expected_success_count
    #      (e.g. caller said "all 10 should be idempotent 200s" but
    #      we got 200s + 404s).
    inconsistency_is_suspicious = (
        inconsistency
        and not is_expected_atomic_outcome
        and (
            len(non_error_statuses) > 2
            or success_count != expected_success_count
        )
    )
    if (
        inconsistency_is_suspicious
        and not (uniqueness_violation or error_class_divergence)
    ):
        parts.append(
            "response status diverged across identical concurrent "
            f"requests: {dict(status_dist)} — non-deterministic"
        )

    if not parts:
        rationale = (
            f"{replicas} concurrent requests landed within the "
            f"expected envelope ({success_count} success of "
            f"{expected_success_count} allowed, {error_count} 5xx). "
            "Endpoint appears concurrency-safe."
        )
    else:
        rationale = " | ".join(parts)

    return ConcurrentProbeVerdict(
        endpoint=url,
        replicas=replicas,
        outcomes=outcomes_tuple,
        status_distribution=dict(status_dist),
        success_count=success_count,
        error_count=error_count,
        expected_success_count=expected_success_count,
        inconsistency=inconsistency,
        uniqueness_violation=uniqueness_violation,
        error_class_divergence=error_class_divergence,
        rationale=rationale,
    )
