"""Phase 1a tests for the http_attack probe primitive.

Covers:
* Host policy (default allow-list; explicit external opt-in)
* Request validation (methods, URLs, sizes)
* Transport (success / 4xx / 5xx / network error all surface cleanly)
* Header redaction in receipts (sensitive header values never persisted)
* Receipts JSONL append-and-tail-load
* Role-allow-list isolation — http_attack must NOT be reachable from
  any existing role's tool list until the pen_tester role lands in
  Phase 1c.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from augmentum.bug_finder import pen_test
from augmentum.bug_finder.agent_tools import (
    HTTPAttackTool,
    PEN_TEST_TOOL_NAMES,
    build_deterministic_tools,
    build_pen_test_tools,
)
from augmentum.bug_finder.pen_test import (
    ProbeRequest,
    ProbeResponse,
    append_probe_receipt,
    execute_probe,
    is_host_allowed,
    load_probe_receipts,
)


# ---------------------------------------------------------------------------
# Host policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host,expected", [
    ("localhost", True),
    ("127.0.0.1", True),
    ("::1", True),
    ("0.0.0.0", True),
    ("host.docker.internal", True),
    ("172.17.0.5", True),           # docker bridge
    ("172.20.10.1", True),           # docker bridge
    ("172.31.255.254", True),        # docker bridge upper edge
    ("my-service", True),            # compose single-label
    ("api", True),                   # compose single-label
])
def test_default_host_policy_permits_local_targets(
    host: str, expected: bool,
) -> None:
    assert is_host_allowed(host) is expected


@pytest.mark.parametrize("host", [
    "google.com",
    "internal.company.net",
    "192.168.1.5",                   # RFC1918 — could be user's LAN
    "10.0.0.1",                      # RFC1918
    "172.32.0.1",                    # just past docker bridge range
    "8.8.8.8",
    "169.254.169.254",               # cloud metadata endpoint
])
def test_default_host_policy_refuses_external(host: str) -> None:
    """RFC1918 ranges + arbitrary DNS names + cloud metadata are NOT
    allowed by default. Anything outside the explicit allow-list
    requires an opt-in."""
    assert is_host_allowed(host) is False


def test_external_override_admits_arbitrary_hosts() -> None:
    """The opt-in waives the host check entirely; useful only when
    the caller owns the external target."""
    assert is_host_allowed("google.com", allow_external=True) is True
    assert is_host_allowed("169.254.169.254", allow_external=True) is True


def test_empty_host_is_refused() -> None:
    assert is_host_allowed("") is False
    assert is_host_allowed("   ") is False


# ---------------------------------------------------------------------------
# Method validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_method_returns_clean_refusal() -> None:
    req = ProbeRequest(method="TRACE", url="http://localhost/x")
    resp, receipt = await execute_probe(req)
    assert not resp.ok
    assert "method not allowed" in resp.error.lower()
    assert receipt.host_policy == "refused"
    # Hostname was never looked up — refusal happens at validation
    assert receipt.response_status == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", [
    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
])
async def test_allowed_methods_reach_transport(method: str) -> None:
    """Allowed methods get past validation. We point at a port nothing
    listens on so transport fails — but the receipt records the
    attempt, which is the contract."""
    req = ProbeRequest(method=method, url="http://127.0.0.1:1/x")
    resp, receipt = await execute_probe(req)
    # Either connect-refused or timeout — either way: ok=False with
    # an error, NOT a method refusal.
    assert not resp.ok
    assert "method not allowed" not in (resp.error or "").lower()
    assert receipt.host_policy == "loopback"


# ---------------------------------------------------------------------------
# URL / scheme validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_scheme_refused() -> None:
    req = ProbeRequest(method="GET", url="file:///etc/passwd")
    resp, receipt = await execute_probe(req)
    assert not resp.ok
    assert "unsupported scheme" in resp.error
    assert receipt.host_policy == "refused"


@pytest.mark.asyncio
async def test_external_host_refused_without_override() -> None:
    req = ProbeRequest(method="GET", url="https://example.com/")
    resp, receipt = await execute_probe(req)
    assert not resp.ok
    assert "allow_external" in resp.error
    assert receipt.host_policy == "refused"


@pytest.mark.asyncio
async def test_external_host_admitted_with_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_external bypasses the host check. We don't actually want
    to hit example.com in CI, so swap the httpx client for a stub
    that returns a canned 204."""

    class _StubResponse:
        def __init__(self) -> None:
            self.status_code = 204
            self.headers = httpx.Headers({"x-test": "ok"})
            self.content = b""
            self.url = httpx.URL("https://example.com/")

    class _StubClient:
        def __init__(self, *_, **__) -> None:
            pass

        async def __aenter__(self) -> "_StubClient":
            return self

        async def __aexit__(self, *_) -> None:
            return None

        async def request(self, *_args, **_kwargs) -> _StubResponse:
            return _StubResponse()

    monkeypatch.setattr(pen_test.httpx, "AsyncClient", _StubClient)
    req = ProbeRequest(
        method="GET", url="https://example.com/",
        allow_external=True,
    )
    resp, receipt = await execute_probe(req)
    assert resp.ok
    assert resp.status == 204
    assert receipt.host_policy == "external_override"


# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_body_refused() -> None:
    huge = "x" * (pen_test.MAX_REQUEST_BODY_BYTES + 1)
    req = ProbeRequest(method="POST", url="http://localhost/x", body=huge)
    resp, receipt = await execute_probe(req)
    assert not resp.ok
    assert "too large" in resp.error
    assert receipt.request_body_size > pen_test.MAX_REQUEST_BODY_BYTES


# ---------------------------------------------------------------------------
# Response capture + truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_response_captures_excerpt_and_marks_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_payload = b"A" * (pen_test.DEFAULT_MAX_RESPONSE_BYTES * 4)

    class _StubResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = httpx.Headers({"content-type": "application/octet-stream"})
            self.content = big_payload
            self.url = httpx.URL("http://localhost/big")

    class _StubClient:
        def __init__(self, *_, **__) -> None:
            pass

        async def __aenter__(self) -> "_StubClient":
            return self

        async def __aexit__(self, *_) -> None:
            return None

        async def request(self, *_args, **_kwargs) -> _StubResponse:
            return _StubResponse()

    monkeypatch.setattr(pen_test.httpx, "AsyncClient", _StubClient)

    req = ProbeRequest(method="GET", url="http://localhost/big")
    resp, receipt = await execute_probe(req)
    assert resp.ok
    assert resp.status == 200
    assert resp.body_size == len(big_payload)
    assert resp.body_truncated is True
    # Excerpt must NOT be the full payload
    assert len(resp.body_excerpt.encode("utf-8")) < len(big_payload)
    assert receipt.response_truncated is True


# ---------------------------------------------------------------------------
# Header redaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensitive_headers_redacted_in_receipt() -> None:
    """Sensitive header values must never persist in the receipts
    trail — they'd leak credentials into the workspace substrate."""
    req = ProbeRequest(
        method="GET",
        url="http://127.0.0.1:1/x",
        headers={
            "Authorization": "Bearer s3cr3t-token",
            "Cookie": "session=abc123",
            "X-API-Key": "live-key-must-not-leak",
            "Accept": "application/json",
        },
    )
    _, receipt = await execute_probe(req)
    assert receipt.request_headers["Authorization"] == "<redacted>"
    assert receipt.request_headers["Cookie"] == "<redacted>"
    assert receipt.request_headers["X-API-Key"] == "<redacted>"
    # Non-sensitive headers preserved
    assert receipt.request_headers["Accept"] == "application/json"


# ---------------------------------------------------------------------------
# Receipts persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_appends_receipt_when_workspace_root_supplied(
    tmp_path: Path,
) -> None:
    req = ProbeRequest(
        method="GET", url="http://127.0.0.1:1/x",
        finding_id="F-001", run_id="R-42", note="phase 1a smoke",
    )
    await execute_probe(req, workspace_root=tmp_path)
    receipts = load_probe_receipts(tmp_path)
    assert len(receipts) == 1
    r = receipts[0]
    assert r.finding_id == "F-001"
    assert r.run_id == "R-42"
    assert r.method == "GET"
    assert r.url == "http://127.0.0.1:1/x"
    assert r.host_policy == "loopback"
    assert r.note == "phase 1a smoke"


@pytest.mark.asyncio
async def test_probe_does_not_append_receipt_when_workspace_root_is_none(
    tmp_path: Path,
) -> None:
    """In-memory mode (no workspace_root) skips disk writes — useful
    for unit tests and throwaway probing."""
    req = ProbeRequest(method="GET", url="http://127.0.0.1:1/x")
    await execute_probe(req, workspace_root=None)
    # Nothing should have been written anywhere in tmp_path
    assert list(tmp_path.rglob("*.jsonl")) == []


@pytest.mark.asyncio
async def test_probe_receipts_jsonl_format_is_per_line_json(
    tmp_path: Path,
) -> None:
    """Cross-tool replay depends on JSONL — one JSON object per line."""
    for i in range(3):
        await execute_probe(
            ProbeRequest(
                method="GET", url=f"http://127.0.0.1:1/{i}",
                note=f"hit {i}",
            ),
            workspace_root=tmp_path,
        )
    receipts_file = (
        tmp_path / ".augmentum" / "bug_finder" / "probe_receipts.jsonl"
    )
    assert receipts_file.is_file()
    lines = receipts_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        d = json.loads(line)
        assert "method" in d and "url" in d and "host_policy" in d


# ---------------------------------------------------------------------------
# Tool wrapper (HTTPAttackTool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_validates_required_args() -> None:
    tool = HTTPAttackTool()
    res = await tool.execute()
    assert not res.success
    assert res.validation_error
    assert "required" in res.error


@pytest.mark.asyncio
async def test_tool_emits_parseable_json(tmp_path: Path) -> None:
    tool = HTTPAttackTool(workspace_root_for_receipts=tmp_path)
    res = await tool.execute(
        method="GET", url="http://127.0.0.1:1/x",
    )
    # Transport will fail (port 1 closed) — that's fine; the tool
    # call itself returns parseable JSON.
    data = json.loads(res.output)
    assert "ok" in data
    assert data["ok"] is False
    assert "host_policy" in data
    assert data["host_policy"] == "loopback"


@pytest.mark.asyncio
async def test_tool_refusal_surfaces_as_structured_output() -> None:
    tool = HTTPAttackTool()
    res = await tool.execute(
        method="GET", url="https://google.com/",
    )
    data = json.loads(res.output)
    assert data["ok"] is False
    assert data["host_policy"] == "refused"
    assert "allow_external" in data.get("error", "")


# ---------------------------------------------------------------------------
# Role isolation — the bedrock safety claim
# ---------------------------------------------------------------------------


def test_pen_test_tool_names_not_in_deterministic_set() -> None:
    """Adding ``http_attack`` to ``DETERMINISTIC_TOOL_NAMES`` would
    silently grant probing capability to planner / detector /
    investigator / lead. This must stay false until the pen_tester
    role is the one consuming it."""
    from augmentum.agents.tools import (
        DETERMINISTIC_TOOL_NAMES,
        PEN_TEST_TOOL_NAMES as canonical_pen_test,
    )
    assert canonical_pen_test.isdisjoint(DETERMINISTIC_TOOL_NAMES)
    assert PEN_TEST_TOOL_NAMES == canonical_pen_test


def test_pen_test_tool_names_not_in_any_existing_role() -> None:
    """No existing role's allow-list contains a probe tool. The
    pen_tester role (Phase 1c) is the only legitimate consumer."""
    from augmentum.agents.tools import (
        COMPREHENDER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES,
        FIXER_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        PEN_TEST_TOOL_NAMES as canonical_pen_test,
        PLANNER_TOOL_NAMES,
        READ_ONLY_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
    )
    for role_names in (
        READ_ONLY_TOOL_NAMES,
        PLANNER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES,
        COMPREHENDER_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
        FIXER_TOOL_NAMES,
    ):
        assert canonical_pen_test.isdisjoint(role_names)


def test_build_deterministic_tools_does_not_include_http_attack(
    tmp_path: Path,
) -> None:
    """Belt-and-braces: even if the allow-list constants drift, the
    tool builder must not silently inject http_attack into the
    deterministic toolset."""
    tools = build_deterministic_tools(tmp_path)
    names = {t.name for t in tools}
    assert "http_attack" not in names


def test_build_pen_test_tools_includes_http_attack(
    tmp_path: Path,
) -> None:
    """Phase 1a contract — http_attack is one of the tools the
    pen-test builder returns. The exact-set contract is owned by
    later phases (see test_build_pen_test_tools_returns_all_three_tools
    in test_bug_finder_pen_test_boot.py)."""
    tools = build_pen_test_tools(tmp_path)
    names = {t.name for t in tools}
    assert "http_attack" in names
