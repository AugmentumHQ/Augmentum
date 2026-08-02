"""FabricPeerMiddleware: authenticate cross-peer HTTP requests.

Runs OUTSIDE AuthMiddleware in the ASGI chain. When a peer dispatches
an LLM (or future routed) request, it sends the user request straight
to our normal API endpoint (/v1/chat/completions) with extra fabric
headers proving "I'm peer X, this request is on behalf of user U."

If those headers are present AND the signature verifies AND the
claimed user exists on this node, we pre-populate ``scope["user"]``
and ``scope["fabric_peer"]`` so the downstream chain (AuthMiddleware
+ handler) treats the request as a normal authenticated user request.
``AuthMiddleware`` has a small tolerance addition (three lines) that
honors pre-set ``scope["user"]`` rather than re-validating.

When headers are absent OR invalid, the middleware passes through
unchanged. The downstream AuthMiddleware then runs its normal
validation, which will reject if the request lacks proper user auth.

Critical invariants:

  - Default off: when ``settings.fabric_enabled`` is False, this
    middleware is a pure pass-through (it's never added to the chain
    in lifespan, see fabric/lifespan.py).
  - Unknown peer: any ``X-Fabric-Sender`` not in fabric_nodes is
    treated as if no fabric headers were present (pass through). We
    explicitly do NOT 401 here -- that's AuthMiddleware's job. An
    attacker probing fabric headers should see identical behavior to
    a plain auth-less request.
  - Replay defense: ±300s timestamp window. Same as the pair-request
    flow in peer_auth.py.
  - WebSocket pass-through: fabric peer-to-peer traffic is HTTP only
    (the inter-peer WebSocket is control plane). WS requests skip
    fabric checks entirely.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# Same replay tolerance window as pair_request signing. Five minutes
# is enough to forgive clock skew between LAN peers (typically <1s
# with NTP), strict enough that a captured signed request can't be
# replayed hours later.
_REPLAY_WINDOW_S = 300

# Hard cap on buffered inbound peer-request body. LLM chat bodies are
# typically <500 KB; 16 MB leaves generous headroom for large
# multi-turn contexts while preventing a peer (or a forged sender)
# from forcing us to buffer arbitrary bytes. Over-cap requests fall
# through unauthenticated and AuthMiddleware 401s them, same as if
# no fabric headers had been present.
_MAX_PEER_BODY_BYTES = 16 * 1024 * 1024

# Fabric hop count: defense-in-depth against accidental A→B→C→A
# fan-out loops. Every cross-peer request carries an X-Fabric-Hop-Count
# header (default 0 from an originating user request, incremented by
# 1 when a receiver re-dispatches to another peer). The middleware
# 508s any incoming request whose count is at or above the max.
#
# Today's endpoints don't fan out (knowledge searches local packs only,
# LLM uses resolve_backend_for_model not _with_fabric, etc.) so the
# count is always 0 in practice. The guard exists so a future
# fan-out feature can't reintroduce the loop class without us
# noticing — the receiver refuses third-hop requests automatically.
#
# Max of 2 = "an originating request can be forwarded once, but not
# again". 508 = LOOP_DETECTED, the most semantically apt HTTP status.
_FABRIC_MAX_HOP_COUNT = 2

# Path allowlist for fabric-authenticated requests. A paired peer is
# a guest-with-purpose: it should be able to invoke modality endpoints
# (LLM, image, TTS/STT, knowledge) but NOT account-management or
# admin surfaces on the receiver. Without this guard, a paired peer
# could POST to /api/auth/keys with its signed envelope and mint a
# persistent sk-aug- API key for its per-peer service user — the key
# would survive unpairing, defeating revocation.
#
# Allowlist (not denylist) so future sensitive routes added under
# /api/* are denied by default rather than silently reachable.
# Anything not matching falls through unauthenticated; AuthMiddleware
# then 401s as if no fabric headers had been present.
_FABRIC_ALLOWED_PATH_PREFIXES: tuple[str, ...] = (
    "/v1/",           # OpenAI-compat surface: chat completions, audio, models
    "/api/fabric/",   # fabric-internal endpoints: image/generate, knowledge/search, render, pair
)


def _fabric_path_allowed(path: str) -> bool:
    """True if ``path`` is reachable by a paired peer's signed request."""
    if not path:
        return False
    return any(path.startswith(prefix) for prefix in _FABRIC_ALLOWED_PATH_PREFIXES)


class FabricPeerMiddleware:
    """ASGI middleware that recognises cross-peer signed requests."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        # WebSocket + lifespan + everything-not-http pass through.
        # Cross-peer dispatch is HTTP-only at this phase.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        # Extract headers WITHOUT consuming the body -- ASGI receive
        # is single-use; touching it here would break the downstream
        # handler. We only read headers (already parsed by ASGI).
        headers = self._headers_dict(scope)
        sender = headers.get("x-fabric-sender", "")
        if not sender:
            # No fabric headers -- not a peer request. Pass through.
            await self.app(scope, receive, send)
            return

        # Path allowlist gate. A signed envelope grants the sender a
        # local per-peer service user; that user should only reach the
        # modality surface, not account/admin routes. Reject early so
        # we don't even spend cycles verifying signatures on routes
        # the peer was never meant to touch. Pass-through behavior
        # (AuthMiddleware 401s) is identical to a request that simply
        # forgot its fabric headers — no signal about the allowlist
        # leaks to an attacker.
        path = scope.get("path", "")
        if not _fabric_path_allowed(path):
            log.info(
                "fabric_peer_path_not_allowed",
                sender=sender, path=path,
            )
            await self.app(scope, receive, send)
            return

        signature_b64 = headers.get("x-fabric-signature", "")
        timestamp_str = headers.get("x-fabric-timestamp", "")
        user_id_claim = headers.get("x-fabric-user-id", "")

        # Signature + timestamp are the load-bearing fields — they prove
        # the request came from a paired peer and isn't a replay.
        # ``X-Fabric-User-Id`` was load-bearing pre-3d26639 (drove the
        # local user lookup); after the per-peer service user model it
        # is informational only. Internal LLM dispatch sites (narrative,
        # flow, tools, jobs, role resolution, ...) call
        # ``resolve_backend_with_fabric`` without a request context, so
        # they pass user_id="" through to ``FabricBackend`` and the
        # resulting envelope has an empty ``X-Fabric-User-Id``. Pre-fix:
        # this middleware required user_id_claim to be non-empty as
        # part of an ``all(...)`` check, silently bailed when it wasn't,
        # AuthMiddleware then 401'd — fabric_routing_to_peer fired but
        # the proxied chat returned 401 with no fabric_peer_* log on
        # the receiver side (the diagnostic fingerprint of this bug).
        if not signature_b64 or not timestamp_str:
            log.debug(
                "fabric_peer_headers_incomplete",
                sender=sender, has_sig=bool(signature_b64),
                has_ts=bool(timestamp_str), has_user=bool(user_id_claim),
            )
            await self.app(scope, receive, send)
            return

        # Look up the sender's pubkey + the claimed user from app.state.
        # If we can't (DB not ready, peer unknown, user unknown), pass
        # through. Pass-through is the safe default -- we never claim
        # "this is authenticated" without proof, so the worst case is
        # the request fails AuthMiddleware as if no fabric headers
        # were present.
        app = scope.get("app")
        if app is None or not hasattr(app, "state"):
            await self.app(scope, receive, send)
            return

        coordinator = getattr(app.state, "fabric_coordinator", None)
        sm = getattr(app.state, "state_manager", None)
        session_manager = getattr(app.state, "session_manager", None)

        db_conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        if coordinator is None or db_conn is None or session_manager is None:
            await self.app(scope, receive, send)
            return

        # Resolve sender pubkey from fabric_nodes (the durable identity
        # record, populated at pair time).
        from augmentum.fabric.peer_auth import lookup_peer_pubkey
        pubkey_b64 = await lookup_peer_pubkey(db_conn, sender)
        if pubkey_b64 is None:
            log.info("fabric_peer_unknown_sender", sender=sender)
            await self.app(scope, receive, send)
            return

        # Timestamp window check before signature verify (cheap reject).
        try:
            ts = int(timestamp_str)
        except (TypeError, ValueError):
            await self.app(scope, receive, send)
            return
        now = int(time.time())
        if abs(now - ts) > _REPLAY_WINDOW_S:
            log.info(
                "fabric_peer_timestamp_stale",
                sender=sender, skew_s=abs(now - ts),
            )
            await self.app(scope, receive, send)
            return

        # Drain the request body so we can verify body integrity. ASGI
        # receive() is single-use; once consumed we must synthesize a
        # replacement to feed the downstream handler the same bytes.
        # Cap at _MAX_PEER_BODY_BYTES — over-cap = pass through (the
        # request will then 401 in AuthMiddleware, matching the
        # behavior of any other unauthenticated request).
        body_chunks: list[tuple[bytes, bool]] = []
        total = 0
        too_large = False
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # Disconnect or other event mid-body. Treat as
                # malformed; let downstream see the same event by
                # synthesizing a single end-of-body chunk first.
                body_chunks.append((b"", False))
                await self._replay_app(scope, body_chunks, send)
                return
            chunk = message.get("body", b"") or b""
            more = bool(message.get("more_body", False))
            total += len(chunk)
            if total > _MAX_PEER_BODY_BYTES:
                too_large = True
            body_chunks.append((chunk, more))
            if not more:
                break

        if too_large:
            log.info(
                "fabric_peer_body_too_large", sender=sender, total_bytes=total,
            )
            await self._replay_app(scope, body_chunks, send)
            return

        body = b"".join(c for c, _ in body_chunks)
        body_hash = hashlib.sha256(body).hexdigest()

        # Hop-count parse. Missing header = 0 (originator). Malformed =
        # treat as 0 so a typo doesn't accidentally lock out a peer
        # that's playing fair (signature still has to verify with the
        # value we use here — see canonical bytes below).
        hop_header = headers.get("x-fabric-hop-count", "0")
        try:
            hop_count = int(hop_header.strip())
            if hop_count < 0:
                hop_count = 0
        except (TypeError, ValueError):
            hop_count = 0

        # 508 LOOP_DETECTED: refuse any request whose hop count is
        # already at the limit. Defense-in-depth — current endpoints
        # don't fan out, but a future feature that does could re-
        # introduce A→B→C→A triangles. The guard catches them
        # automatically without re-auditing every modality endpoint.
        if hop_count >= _FABRIC_MAX_HOP_COUNT:
            log.warning(
                "fabric_peer_hop_count_exceeded",
                sender=sender, hop_count=hop_count, max=_FABRIC_MAX_HOP_COUNT,
                path=scope.get("path", ""),
            )
            await self._send_loop_detected(send)
            return

        # Build canonical bytes the sender signed + verify.
        method = scope.get("method", "")
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"").decode("latin-1", errors="ignore")
        path_with_query = f"{path}?{query_string}" if query_string else path

        canonical = _peer_request_canonical_bytes(
            sender=sender,
            user_id=user_id_claim,
            method=method,
            path=path_with_query,
            timestamp=ts,
            body_sha256=body_hash,
            hop_count=hop_count,
        )
        try:
            pub_bytes = base64.b64decode(pubkey_b64)
            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
            sig_bytes = base64.b64decode(signature_b64)
            pub.verify(sig_bytes, canonical)
        except Exception as exc:
            log.warning(
                "fabric_peer_signature_invalid",
                sender=sender, error=str(exc)[:160],
            )
            await self._replay_app(scope, body_chunks, send)
            return

        # Signature valid. Resolve the LOCAL user the cross-peer
        # dispatch will run as.
        #
        # User accounts don't span peers — a request arriving from peer
        # P with X-Fabric-User-Id of P's local user_id can't be looked
        # up on our side (we don't have P's accounts). Instead, every
        # peer maps to a per-peer service user on the receiver:
        # ``fabric:<short-node-id>``, created lazily on first signed
        # request from that peer. Data isolation moves from per-user
        # to per-peer; the original ``X-Fabric-User-Id`` claim is now
        # informational metadata for audit/telemetry only.
        user = await session_manager.get_or_create_fabric_peer_user(
            sender, hostname=headers.get("x-fabric-hostname", ""),
        )
        if user is None or not user.is_active:
            log.warning(
                "fabric_peer_user_provision_failed",
                sender=sender, claimed_user=user_id_claim,
            )
            await self._replay_app(scope, body_chunks, send)
            return

        # Authenticated. Mark the scope so downstream handlers can
        # tell this is a fabric-routed request (e.g. for telemetry,
        # different routing policies, etc.) and attach the user so
        # AuthMiddleware's tolerance check honors it.
        scope["user"] = user
        scope["fabric_peer"] = {
            "sender_node_id": sender,
            "verified_at": now,
            "trust_tier": "local",  # phase 5 federation will set this differently
        }

        log.debug(
            "fabric_peer_authenticated",
            sender=sender, user_id=user_id_claim, path=path,
        )

        # Phase 9.3 — register this in-flight request so an inbound
        # MSG_CANCEL_REQUEST envelope can cancel the right task.
        # request_id is optional in the header (legacy clients omit
        # it; the cancel backstop just won't fire for them — TCP-
        # close still handles cancellation via the existing chain).
        request_id = headers.get("x-fabric-request-id", "").strip()
        if request_id:
            import asyncio as _asyncio
            current_task = _asyncio.current_task()
            if current_task is not None:
                coordinator.register_inflight(request_id, current_task)

        # Phase 9-lifecycle — emit job_started back to the originator
        # so its UI can render "[peer] is working on your request"
        # instead of a dead-air spinner. Fire-and-forget; the WS push
        # might fail (peer's WS to us dropped) and that's fine — the
        # data plane (HTTPS) is independent.
        from augmentum.fabric.protocol import (
            MSG_JOB_COMPLETED,
            MSG_JOB_FAILED,
            MSG_JOB_STARTED,
        )
        path_for_kind = scope.get("path", "")
        if request_id:
            await coordinator.send_to_peer(
                sender, msg_type=MSG_JOB_STARTED,
                payload={
                    "request_id": request_id,
                    "kind": _infer_kind(path_for_kind),
                    "path": path_for_kind,
                },
            )

        failure_reason = ""
        try:
            await self._replay_app(scope, body_chunks, send)
        except Exception as exc:
            failure_reason = type(exc).__name__
            raise
        finally:
            if request_id:
                coordinator.unregister_inflight(request_id)
                if failure_reason:
                    await coordinator.send_to_peer(
                        sender, msg_type=MSG_JOB_FAILED,
                        payload={
                            "request_id": request_id,
                            "reason": failure_reason,
                        },
                    )
                else:
                    await coordinator.send_to_peer(
                        sender, msg_type=MSG_JOB_COMPLETED,
                        payload={"request_id": request_id, "ok": True},
                    )

    async def _send_loop_detected(self, send) -> None:
        """Emit a 508 LOOP_DETECTED response for hop-count overflow.

        Used when a fabric request arrives with X-Fabric-Hop-Count at
        or above ``_FABRIC_MAX_HOP_COUNT``. We don't propagate to the
        downstream handler — the request is structurally broken (it
        traversed the fabric too many times) and the right answer is
        to refuse loudly so the operator notices.
        """
        body = b'{"error":{"type":"loop_detected","message":"fabric hop count exceeded"}}'
        await send({
            "type": "http.response.start",
            "status": 508,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _replay_app(
        self, scope, body_chunks: list[tuple[bytes, bool]], send,
    ) -> None:
        """Call the downstream app with a fresh receive() that replays
        the body we already consumed. Each chunk is yielded as a
        separate http.request event preserving the original
        ``more_body`` flag so chunked encoding round-trips correctly.

        After the body drains, receive() must BLOCK until the response
        finishes — not return an immediate ``http.disconnect``. Starlette's
        ``StreamingResponse`` runs the body generator concurrently with a
        disconnect-listener that loops on ``receive()``; a synthetic
        disconnect right after the (small) request body would fire that
        listener and cancel the stream mid-flight. Buffered responses
        (JSONResponse etc.) never call receive() again, so they were
        unaffected — but every streaming fabric response (TTS audio, LLM
        inference, render) was being truncated. We unblock the pending
        receive() once the response's terminal body frame is sent, then
        report the disconnect (now a no-op, the response is already done).
        """
        # Defensive copy + index so the closure mutation is local.
        chunks = list(body_chunks)
        idx = 0
        response_done = asyncio.Event()

        async def replay_receive():
            nonlocal idx
            if idx < len(chunks):
                body, more = chunks[idx]
                idx += 1
                return {
                    "type": "http.request", "body": body, "more_body": more,
                }
            # Body fully replayed. Mirror a real ASGI server: block until
            # the client (here: the completed response) produces an event,
            # rather than synthesising an instant disconnect that aborts
            # streaming responses.
            await response_done.wait()
            return {"type": "http.disconnect"}

        async def replay_send(message):
            if (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_done.set()
            await send(message)

        try:
            await self.app(scope, replay_receive, replay_send)
        finally:
            # Unblock any receive() still parked (e.g. handler raised before
            # sending a terminal body frame) so the task can be cancelled.
            response_done.set()

    @staticmethod
    def _headers_dict(scope) -> dict[str, str]:
        """Build a lowercase-keyed string dict from ASGI scope headers.

        ASGI headers are list[tuple[bytes, bytes]] with names already
        lowercased. We decode latin-1 (the ASGI header spec) and skip
        anything not decodable as UTF-8 in the values.
        """
        out: dict[str, str] = {}
        for name, value in scope.get("headers", []):
            try:
                key = name.decode("latin-1", errors="ignore").lower()
                val = value.decode("latin-1", errors="ignore")
            except (AttributeError, UnicodeDecodeError):
                # Malformed header tuple — ASGI guarantees bytes, so this
                # only fires under a fuzzer / mock; safe to skip.
                continue
            # Headers can repeat; keep the first (HTTP spec for most
            # auth-style headers).
            out.setdefault(key, val)
        return out


def _infer_kind(path: str) -> str:
    """Best-effort mapping of request path → high-level kind for the
    lifecycle event payload. The UI uses this to decide which queue
    indicator (LLM / image / knowledge) to update. Unknown paths
    surface as "rpc" — generic but non-empty.
    """
    if "/v1/chat" in path or "/api/chat" in path:
        return "llm.inference"
    if "/api/image/generate" in path:
        return "image.generation"
    if "/api/knowledge/search" in path:
        return "knowledge.search"
    if "/api/image/" in path:
        return "image.fetch"
    return "rpc"


def _peer_request_canonical_bytes(
    *,
    sender: str,
    user_id: str,
    method: str,
    path: str,
    timestamp: int,
    body_sha256: str,
    hop_count: int = 0,
) -> bytes:
    """Build the canonical byte representation that BOTH sides sign + verify.

    Format mirrors the pair-request canonical bytes in peer_auth.py --
    line-separated key=value fields, version-tagged at the top, no
    JSON to avoid ordering ambiguity across implementations.

    ``body_sha256`` is the lowercase hex SHA-256 of the raw request
    body bytes as transmitted (``hashlib.sha256(b"").hexdigest()`` for
    empty bodies). The hash is not carried in a separate header: the
    receiver re-hashes the body it sees + rebuilds these canonical
    bytes locally, so the signature verifies only if the receiver's
    hash matches the sender's. That removes a tampering surface (no
    way to send a mismatched hash-vs-body) and avoids a redundant
    header on every signed request.

    Bumped to ``v3`` (hop coverage added). The fabric is default-off
    and unreleased, so both sides upgrade in lockstep — no compat
    shim. Hop is part of the canonical bytes (not just a transport
    header) so an in-path attacker can't decrement it to break the
    loop guard. Default hop=0 means existing tests + originator-side
    calls don't have to thread the field through.
    """
    parts = [
        "v3",
        f"sender={sender}",
        f"user={user_id}",
        f"method={method.upper()}",
        f"path={path}",
        f"ts={timestamp}",
        f"body_sha256={body_sha256}",
        f"hop={hop_count}",
    ]
    return "\n".join(parts).encode("utf-8")


def build_signed_peer_headers(
    *,
    identity,
    user_id: str,
    method: str,
    path: str,
    body: bytes,
    hop_count: int = 0,
) -> dict[str, str]:
    """Build the X-Fabric-* header set for an outbound peer request.

    Called by FabricBackend before issuing the proxy HTTPS call. The
    caller MUST pass the exact body bytes that will be transmitted --
    httpx's ``json=`` keyword serialises internally and would produce
    different bytes than what was hashed; use ``content=body`` on the
    httpx call alongside this header set.

    The receiving peer's FabricPeerMiddleware re-derives the canonical
    bytes from the body it sees + verifies against the pinned pubkey
    for the claimed sender.

    ``hop_count`` defaults to 0 for an originating user-initiated
    request. When a receiver re-dispatches (a future fan-out feature
    — none today), it MUST pass ``hop_count=inbound_hop + 1`` so the
    next receiver can refuse at ``_FABRIC_MAX_HOP_COUNT``. The value
    is part of the signed canonical bytes so an in-path attacker
    can't decrement it to defeat the loop guard.

    ``identity`` must be a ``FabricIdentity``.
    """
    ts = int(time.time())
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = _peer_request_canonical_bytes(
        sender=identity.node_id,
        user_id=user_id,
        method=method,
        path=path,
        timestamp=ts,
        body_sha256=body_hash,
        hop_count=hop_count,
    )
    signature = identity.sign(canonical)
    return {
        "X-Fabric-Sender": identity.node_id,
        "X-Fabric-User-Id": user_id,
        "X-Fabric-Timestamp": str(ts),
        "X-Fabric-Signature": base64.b64encode(signature).decode("ascii"),
        "X-Fabric-Hop-Count": str(hop_count),
    }
