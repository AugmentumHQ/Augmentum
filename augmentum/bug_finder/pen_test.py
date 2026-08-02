"""Dynamic-probe primitives for the bug_finder's pen-test leg.

Phase 1a (this module) ships the load-bearing primitive: an HTTP
attack executor with safety guards plus a receipts trail. Later
phases add the under-test boot orchestrator (1b), the ``pen_tester``
subagent role (1c), and the higher-level attack primitives — authz
matrix, injection sweep, concurrent probe (Phase 2).

The architectural pitch: static analysis identifies WHERE to look;
the dynamic probe verifies whether the defenses actually hold. Both
legs share the same orchestrator and findings shape, so a probe that
exploits a static finding strengthens its severity; a probe that
fails to exploit downgrades it.

**Safety contract** — the bug_finder runs inside a disposable Docker
container, but probing means actual HTTP traffic to a real listener.
This module enforces:

* Host allow-list — default permits only loopback, the docker bridge
  range, and ``host.docker.internal``. External hosts require an
  explicit ``allow_external=True`` opt-in.
* Bounded sizes — request body capped, response body captured with
  a hard limit so probe receipts can't blow up the substrate.
* Bounded latency — wallclock timeout per probe.
* Receipts JSONL — every probe (success, refusal, transport error)
  appends to ``.augmentum/bug_finder/probe_receipts.jsonl``. Replayable.

The receipts substrate is intentionally separate from the fixer's
``receipts.jsonl``: different shape, different lifecycle, different
trust framing. A probe is evidence of behavior, not a code edit.
"""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from augmentum.bug_finder.workspace_substrate import (
    ensure_substrate,
    substrate_dir,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Limits + defaults
# ---------------------------------------------------------------------------


# Conservative defaults — a pen-test probe should be small and fast.
# The model can request a higher limit but the executor will clamp.
DEFAULT_TIMEOUT_S: float = 30.0
MAX_TIMEOUT_S: float = 120.0
DEFAULT_MAX_RESPONSE_BYTES: int = 16 * 1024     # 16 KB captured for receipts
HARD_MAX_RESPONSE_BYTES: int = 256 * 1024       # 256 KB ceiling for `output`
MAX_REQUEST_BODY_BYTES: int = 1 * 1024 * 1024   # 1 MB
DEFAULT_MAX_REDIRECTS: int = 3
ALLOWED_METHODS: frozenset[str] = frozenset({
    "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
})


# ---------------------------------------------------------------------------
# Host policy
# ---------------------------------------------------------------------------


# The "always safe" set — addresses that can ONLY refer to either the
# container itself or its docker bridge / compose network. Permitted
# without any opt-in.
_LOOPBACK_HOSTNAMES: frozenset[str] = frozenset({
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    # The Docker DNS name that resolves the host's bridge from inside
    # a container. Useful when the under-test app boots on the host.
    "host.docker.internal",
    # Docker compose default service-network suffix
    "docker.internal",
})

# Docker bridge default range (172.16.0.0/12) — anything in this block
# is overwhelmingly likely to be another container on the same machine.
# RFC1918 private ranges (10/8, 192.168/16) are NOT in the default
# allow-list because they could be the user's LAN.
_DOCKER_BRIDGE_NETWORK = ipaddress.ip_network("172.16.0.0/12")


def _normalize_host(host: str) -> str:
    return (host or "").strip().lower().rstrip(".")


def is_host_allowed(host: str, *, allow_external: bool = False) -> bool:
    """Return True when the host is safe to probe under the policy.

    ``allow_external=True`` waives the host check entirely — the
    caller has explicitly opted into probing arbitrary destinations.
    Reserved for cases where the user runs the bug_finder against a
    known external staging environment they own.

    Default policy permits only loopback, ``host.docker.internal``,
    and the Docker bridge network (172.16/12). RFC1918 ranges are
    NOT default-allowed; they could be the user's LAN.
    """
    if allow_external:
        return True
    h = _normalize_host(host)
    if not h:
        return False
    if h in _LOOPBACK_HOSTNAMES:
        return True
    # Compose-internal short DNS names look like ``my-service`` — no
    # dots, no scheme. Treat them as docker-internal IF we're running
    # inside a container. We can't reliably check that without poking
    # the filesystem, so for Phase 1a we accept any single-label
    # hostname (no dots) — the model has to opt in to a real DNS name
    # via allow_external.
    if "." not in h:
        return True
    try:
        addr = ipaddress.ip_address(h)
    except ValueError:
        # Not a literal IP and not loopback or single-label — that's
        # an external DNS name. Refuse unless overridden.
        return False
    if addr in _DOCKER_BRIDGE_NETWORK:
        return True
    return False


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeRequest:
    """One HTTP attack request, fully specified by the caller.

    The executor clamps timeouts and body sizes; the request shape is
    otherwise verbatim. Headers and body are passed through so the
    pen_tester can craft any payload (injection strings, malformed
    JSON, oversized fields) without the executor sanitizing them.
    """

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    follow_redirects: bool = False
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    allow_external: bool = False
    # Identification for the receipts trail. Empty when the probe is
    # standalone (e.g. a manual experiment), populated when the
    # pen_tester role dispatches it as part of an audit.
    finding_id: str = ""
    run_id: str = ""
    note: str = ""             # Free-form caller annotation


@dataclass(frozen=True)
class ProbeResponse:
    """Captured response from one probe.

    ``ok`` is true iff the transport succeeded — the HTTP status code
    is in ``status`` regardless. A 500 is still ``ok=True``; a DNS
    failure is ``ok=False`` with ``error`` populated.
    """

    ok: bool
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    body_excerpt: str = ""
    body_size: int = 0
    body_truncated: bool = False
    latency_ms: int = 0
    final_url: str = ""
    error: str = ""


@dataclass
class ProbeReceipt:
    """Append-only audit row for one probe.

    Different shape from ``receipts.Receipt`` (file-action receipts)
    because the semantics are different: a probe is observed
    behavior, not a code edit. We keep the request + response excerpt
    so a future investigator can replay the same payload.
    """

    ts: int = 0
    run_id: str = ""
    finding_id: str = ""
    method: str = ""
    url: str = ""
    request_headers: dict[str, str] = field(default_factory=dict)
    request_body_excerpt: str = ""
    request_body_size: int = 0
    response_status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body_excerpt: str = ""
    response_body_size: int = 0
    response_truncated: bool = False
    latency_ms: int = 0
    ok: bool = False
    error: str = ""
    host_policy: str = ""      # "loopback" | "docker_bridge" | "external_override" | "refused"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Header redaction
# ---------------------------------------------------------------------------


# Header names we redact in receipts. The probe still SENDS them
# (the under-test app needs the real value); we just don't keep the
# raw value in the audit trail. Lower-cased compare.
_REDACTED_HEADER_NAMES: frozenset[str] = frozenset({
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "proxy-authorization",
})


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy with sensitive headers replaced by ``"<redacted>"``."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _REDACTED_HEADER_NAMES:
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def _classify_host(host: str, *, allow_external: bool) -> str:
    """Human-readable policy bucket for the receipts trail."""
    h = _normalize_host(host)
    if h in _LOOPBACK_HOSTNAMES:
        return "loopback"
    if "." not in h:
        return "single_label"
    try:
        addr = ipaddress.ip_address(h)
        if addr in _DOCKER_BRIDGE_NETWORK:
            return "docker_bridge"
    except ValueError:
        pass
    return "external_override" if allow_external else "refused"


def _excerpt_bytes(
    raw: bytes, *, limit: int,
) -> tuple[str, int, bool]:
    """Decode ``raw`` for the receipts trail. Returns
    ``(excerpt_str, total_size, truncated)``.

    Always decodes as utf-8 with ``replace`` — pen-test responses
    routinely contain binary or partial encodings; we want SOMETHING
    in the receipt for the LLM to reason about, not an exception.
    """
    total = len(raw)
    if total <= limit:
        return raw.decode("utf-8", errors="replace"), total, False
    return (
        raw[:limit].decode("utf-8", errors="replace") + "\n…[truncated]",
        total,
        True,
    )


async def execute_probe(
    req: ProbeRequest,
    *,
    workspace_root: Path | None = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> tuple[ProbeResponse, ProbeReceipt]:
    """Send one probe. Always returns a ``ProbeResponse`` + receipt
    pair; never raises.

    Side effects: when ``workspace_root`` is supplied, the receipt is
    appended to ``probe_receipts.jsonl``. Pass ``None`` for in-memory
    use (tests, throwaway probing).

    The contract is "the caller can always inspect what happened" —
    transport errors, host-policy refusals, malformed URLs all
    surface as receipts with ``ok=False`` + ``error`` populated.
    """
    started = time.monotonic()
    redacted_req_headers = _redact_headers(req.headers)
    req_body_size = len(req.body.encode("utf-8")) if req.body else 0

    receipt = ProbeReceipt(
        ts=int(time.time()),
        run_id=req.run_id,
        finding_id=req.finding_id,
        method=(req.method or "").upper(),
        url=req.url,
        request_headers=redacted_req_headers,
        request_body_excerpt=req.body[:512],
        request_body_size=req_body_size,
        note=req.note,
    )

    # --- Validate method ---
    method = (req.method or "").upper().strip()
    if method not in ALLOWED_METHODS:
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False, error=f"method not allowed: {method}",
            latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        receipt.host_policy = "refused"
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt

    # --- Validate URL + host policy ---
    try:
        parsed = urlparse(req.url)
    except ValueError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False, error=f"invalid URL: {exc}", latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        receipt.host_policy = "refused"
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt

    if parsed.scheme not in {"http", "https"}:
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False,
            error=f"unsupported scheme: {parsed.scheme!r}",
            latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        receipt.host_policy = "refused"
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt

    host = parsed.hostname or ""
    if not is_host_allowed(host, allow_external=req.allow_external):
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False,
            error=(
                f"host {host!r} not in default allow-list "
                "(loopback / docker bridge / single-label). Pass "
                "allow_external=True to opt in explicitly."
            ),
            latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        receipt.host_policy = "refused"
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt

    receipt.host_policy = _classify_host(
        host, allow_external=req.allow_external,
    )

    # --- Validate sizes ---
    if req_body_size > MAX_REQUEST_BODY_BYTES:
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False,
            error=(
                f"request body too large: {req_body_size} > "
                f"{MAX_REQUEST_BODY_BYTES}"
            ),
            latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt

    # --- Clamp limits ---
    timeout = max(0.5, min(req.timeout_s or DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S))
    cap_bytes = max(
        1024,
        min(max_response_bytes or DEFAULT_MAX_RESPONSE_BYTES, HARD_MAX_RESPONSE_BYTES),
    )

    # --- Send ---
    try:
        async with httpx.AsyncClient(
            follow_redirects=req.follow_redirects,
            max_redirects=max(0, req.max_redirects),
            timeout=httpx.Timeout(timeout),
        ) as client:
            response = await client.request(
                method,
                req.url,
                headers=req.headers or None,
                content=req.body.encode("utf-8") if req.body else None,
            )
            raw = response.content or b""
            excerpt, total, truncated = _excerpt_bytes(raw, limit=cap_bytes)
            elapsed = int((time.monotonic() - started) * 1000)
            resp_headers = {k: v for k, v in response.headers.items()}
            resp = ProbeResponse(
                ok=True,
                status=int(response.status_code),
                headers=resp_headers,
                body_excerpt=excerpt,
                body_size=total,
                body_truncated=truncated,
                latency_ms=elapsed,
                final_url=str(response.url),
            )
            receipt.ok = True
            receipt.response_status = resp.status
            receipt.response_headers = _redact_headers(resp_headers)
            receipt.response_body_excerpt = excerpt
            receipt.response_body_size = total
            receipt.response_truncated = truncated
            receipt.latency_ms = elapsed
            _maybe_append_receipt(workspace_root, receipt)
            return resp, receipt
    except httpx.HTTPError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False,
            error=f"{type(exc).__name__}: {exc}"[:300],
            latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        elapsed = int((time.monotonic() - started) * 1000)
        resp = ProbeResponse(
            ok=False,
            error=f"{type(exc).__name__}: {exc}"[:300],
            latency_ms=elapsed,
        )
        receipt.ok = False
        receipt.error = resp.error
        receipt.latency_ms = elapsed
        _maybe_append_receipt(workspace_root, receipt)
        return resp, receipt


# ---------------------------------------------------------------------------
# Receipts JSONL
# ---------------------------------------------------------------------------


def _probe_receipts_path(workspace_root: Path) -> Path:
    return substrate_dir(workspace_root) / "probe_receipts.jsonl"


def _maybe_append_receipt(
    workspace_root: Path | None,
    receipt: ProbeReceipt,
) -> None:
    """Best-effort write. Silently no-ops when ``workspace_root`` is
    ``None`` (in-memory / test mode)."""
    if workspace_root is None:
        return
    append_probe_receipt(workspace_root, receipt)


def append_probe_receipt(
    workspace_root: Path,
    receipt: ProbeReceipt,
) -> None:
    """Append one probe receipt to the workspace's JSONL trail."""
    ensure_substrate(workspace_root)
    if receipt.ts == 0:
        receipt.ts = int(time.time())
    try:
        with _probe_receipts_path(workspace_root).open(
            "a", encoding="utf-8",
        ) as fp:
            fp.write(json.dumps(receipt.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning(
            "bug_finder_probe_receipt_append_failed",
            workspace=str(workspace_root), error=str(exc),
        )


def load_probe_receipts(
    workspace_root: Path,
    *,
    limit: int = 200,
) -> list[ProbeReceipt]:
    """Tail-load probe receipts. Best-effort; returns [] on any error."""
    p = _probe_receipts_path(workspace_root)
    if not p.is_file():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[ProbeReceipt] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        out.append(ProbeReceipt(
            ts=int(d.get("ts") or 0),
            run_id=str(d.get("run_id") or ""),
            finding_id=str(d.get("finding_id") or ""),
            method=str(d.get("method") or ""),
            url=str(d.get("url") or ""),
            request_headers=dict(d.get("request_headers") or {}),
            request_body_excerpt=str(d.get("request_body_excerpt") or ""),
            request_body_size=int(d.get("request_body_size") or 0),
            response_status=int(d.get("response_status") or 0),
            response_headers=dict(d.get("response_headers") or {}),
            response_body_excerpt=str(d.get("response_body_excerpt") or ""),
            response_body_size=int(d.get("response_body_size") or 0),
            response_truncated=bool(d.get("response_truncated")),
            latency_ms=int(d.get("latency_ms") or 0),
            ok=bool(d.get("ok")),
            error=str(d.get("error") or ""),
            host_policy=str(d.get("host_policy") or ""),
            note=str(d.get("note") or ""),
        ))
    return out
