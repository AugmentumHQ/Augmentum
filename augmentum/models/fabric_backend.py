"""FabricBackend: a remote Augmentum peer exposed as a ModelBackend.

When the RoutingDirector picks a peer to serve an LLM request, it
hands the caller one of these. From the dispatch layer's perspective
this is just another ``ModelBackend`` -- ``chat()`` and ``chat_stream()``
behave like any other OpenAI-compat backend, except the HTTP request
goes to a peer in the user's fabric rather than to a public cloud
API.

The peer's existing ``/v1/chat/completions`` endpoint handles the
real inference. We send the request over the peer's Caddy HTTPS edge
(same TLS endpoint that serves the UI) and stream chunks back.

Phase 3 caveats — explicit limitations to fix in 3.x:

  - Cross-peer authentication is NOT yet wired. The current ``chat``
    + ``chat_stream`` send a signed ``X-Fabric-Sender`` header so
    later phases can verify on the peer side, but the peer's existing
    AuthMiddleware doesn't understand fabric signatures yet and will
    reject the request with 401. Result: when this path actually
    fires (fabric_enabled + paired peer + model only on peer),
    inference fails cleanly with an auth error. The user sees a
    bubbled-up error message; no silent failure mode. Phase 3.x adds
    a ``FabricPeerMiddleware`` that accepts signed cross-peer
    requests.

  - LAN-only assumption baked in. The peer's ``addr`` field comes
    from pairing; over the public internet a Tailscale (or similar)
    overlay is required. No relay support, no NAT traversal.

  - No streaming back-pressure handling beyond what httpx provides.

What's NOT a caveat:

  - The ModelBackend ABC contract is fully honored. Existing dispatch
    code that calls ``backend.chat_stream(...)`` works identically
    whether ``backend`` is a local OpenAIBackend or a FabricBackend.
    No call-site special-casing.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.fabric.capabilities import LLMInferenceCapability
    from augmentum.fabric.identity import FabricIdentity

log = get_logger(__name__)


# ── Typed peer-call errors (Phase 9-failure-surfacing) ───────────
#
# The dispatch layer + UI care about distinguishing three failure
# kinds: "peer was unreachable" (likely transient), "peer doesn't
# have the model it advertised" (capability staleness — drop the
# stale capability + maybe retry locally), and "peer returned a
# protocol error" (5xx / unexpected shape — likely needs operator
# attention). Today everything bubbled as plain RuntimeError; the
# UI couldn't tell what to render.

class FabricPeerError(RuntimeError):
    """Base for fabric peer-call failures. Subclasses RuntimeError
    so existing handlers' generic ``except RuntimeError`` paths
    continue to work — typing is additive, not breaking.
    """


class PeerUnreachableError(FabricPeerError):
    """Couldn't reach the peer at all — connect refused, TCP-reset,
    read timeout. Best signalled to the UI as a transient network
    issue; the matrix view + UI peer-row already show offline state
    from missed heartbeats so a retry might just succeed.
    """


class PeerModelMissingError(FabricPeerError):
    """Peer's heartbeat advertised the model but a request for it
    came back 4xx with a "model not found" shape. The peer must
    have swapped models between heartbeats (or the operator
    unloaded it on their end). FabricBackend automatically asks the
    coordinator to drop the stale capability so subsequent dispatch
    won't route to the same peer.
    """


class PeerProtocolError(FabricPeerError):
    """Peer returned a 5xx or an unexpected 4xx — the peer is
    reachable but mis-behaving. Surface to the operator; not
    transient.
    """


_MODEL_MISSING_SIGNALS = (
    "model not found",
    "no such model",
    "model 'unknown'",
    "model not loaded",
)


# ── Pre-dispatch model-load coordination ─────────────────────────────
#
# Before sending an LLM request to a peer we POST /api/fabric/load_model
# and poll /api/fabric/load_status until "ready" (or timeout/failure).
# This separates load failures (OOM, missing file) from dispatch
# failures (network, signature) — pre-fix everything looked like a
# generic 500 mid-stream on the receiver and the operator had no way
# to tell why.

# Total wall-clock budget for a cold load. 35B-Q3 on consumer hardware
# is ~30s end-to-end (disk → VRAM); 120s leaves headroom for slower
# tiers + first-time CUDA kernel compile. Tunable via settings if a
# specific deployment hits the ceiling.
_LOAD_TIMEOUT_S = 120.0

# Polling cadence. 250ms is fast enough for the operator-facing UX
# (the next status check fires before any user notices the gap) and
# slow enough that we don't hammer a peer that's compiling kernels.
_LOAD_POLL_INTERVAL_S = 0.25

# Prefill-progress poll cadence. Prefill happens AFTER the load gate
# (model already ready), while the sender is consuming the inference
# SSE — so the load poll loop is no longer running. When the peer
# relays a ``prefill`` stage_start, we keep a lightweight side-poll of
# /load_status alive (it embeds prefill_progress while the model is
# READY + a prompt is processing) and mirror each snapshot into the
# coordinator cache, until the first content delta ends prefill. 400ms
# matches the local prefill bar's perceived cadence without hammering.
_PREFILL_POLL_INTERVAL_S = 0.4


def _is_model_missing_response(status_code: int, body: str) -> bool:
    """Heuristic: did the peer signal "I don't have that model"?
    Conservative — only matches on explicit text. A peer that
    returns generic 4xx for an unrelated reason (rate limit, auth
    expired) doesn't trigger capability invalidation.
    """
    if status_code != 404 and status_code != 400:
        return False
    low = body.lower()
    return any(sig in low for sig in _MODEL_MISSING_SIGNALS)


# Substrings in a peer's load_status ``reason`` field that mean the
# load is structurally broken for this peer + model + quant combo —
# retrying with the same inputs will fail again. Used to invalidate
# the peer's advertised capability so the dropdown stops offering it
# until the peer's next heartbeat reasserts the model.
#
# Examples observed in production logs:
#   "RuntimeError: llama-server exited during startup with code -11"
#     (SIGSEGV — model + llama-server version incompatibility, e.g.
#     IQ4_NL on an older build)
#   "OOM during prefill" / "CUDA out of memory at layer ..."
#     (peer's VRAM budget can't host this quant; same model, smaller
#     quant might fit but this exact one can't on this peer)
#   "sched_reserve" / "Gated Delta Net"
#     (partial-offload-incompatible architecture; see
#     _PARTIAL_OFFLOAD_INCOMPATIBLE_ARCHS in llama_server_manager.py)
_STRUCTURAL_LOAD_FAILURE_SIGNALS = (
    "exited during startup",
    "exit code -",
    "out of memory",
    "cuda oom",
    "sched_reserve",
    "gated delta net",
    "unsupported architecture",
    "format not supported",
)


def _is_structural_load_failure(reason: str) -> bool:
    """Heuristic: is this a peer-side load failure that won't recover
    on retry with the same inputs?
    """
    low = reason.lower()
    return any(sig in low for sig in _STRUCTURAL_LOAD_FAILURE_SIGNALS)


class FabricBackend(ModelBackend):
    """A remote peer Augmentum instance exposed as a ModelBackend.

    Constructed by the RoutingDirector when it picks a peer. One
    instance per (peer, request); short-lived. Holds a reference to
    a long-lived ``httpx.AsyncClient`` for the actual HTTP work.
    """

    # Peers may have models with different shapes than the local box;
    # the local supports_mid_conversation_system policy doesn't apply.
    # Use the conservative default (single leading system block) so
    # the peer's own backend implementation handles the rest.
    supports_mid_conversation_system: bool = False

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        peer_node_id: str,
        peer_addr: str,
        advertised_capability: LLMInferenceCapability,
        identity: FabricIdentity | None = None,
        user_id: str = "",
        coordinator: Any | None = None,
        pinned_wire_name: str = "",
    ) -> None:
        self._client = http_client
        self._peer_node_id = peer_node_id
        self._peer_addr = peer_addr.rstrip("/")
        self._capability = advertised_capability
        # Phase 3.x: identity + user_id are required for the peer to
        # authenticate the proxied request. Without them the peer's
        # FabricPeerMiddleware sees no fabric headers and the request
        # falls through to AuthMiddleware, which 401s (no user token
        # in the proxy call). Optional in the constructor so Phase 3
        # tests + the original director path keep working with their
        # legacy invocation; production lifespan wiring always supplies
        # both. When identity is None, _fabric_headers() returns just
        # the legacy informational header (no signature).
        self._identity = identity
        self._user_id = user_id
        # Phase 9.2+: coordinator reference for sending control-plane
        # messages back to the peer (cancellation, etc.). Optional so
        # legacy tests without a coordinator keep working — when None
        # we silently skip the WS backstop and rely on TCP-close.
        self._coordinator = coordinator
        # Phase 8.x: when the operator picked a specific peer entry
        # from the dropdown (``<model>@fabric:<short_id>``) the
        # resolver passes the original suffixed name through here so
        # we can stamp it onto the response / stream chunks. That
        # keeps the persisted ``model_used`` aligned with what the
        # operator selected, so the chat renderer's peer-badge
        # lookup still works post-dispatch and a regenerate routes
        # back to the same peer. Empty string = auto-routed call;
        # response stays clean (auto entry is matched in that case).
        self._pinned_wire_name = pinned_wire_name

    # ── ModelBackend contract ─────────────────────────────────────

    async def ensure_peer_model_loaded(self, model_id: str) -> None:
        """Make sure the peer is serving ``model_id`` before dispatching.

        Thin drain of :meth:`_drive_peer_load` — preserves the original
        contract (await-to-completion, raises the typed peer errors)
        for the non-streaming ``chat()`` path and direct callers. The
        streaming path consumes ``_drive_peer_load`` directly so it can
        turn the load into a ``model_load`` stage for the UI.

        Raises:
          PeerModelMissingError: peer 404'd — model isn't on disk
            on the peer; capability is invalidated so subsequent
            dispatch attempts route elsewhere.
          PeerProtocolError: load failed (OOM, subprocess crash,
            etc.) or timed out. Carries the receiver's reason.
          PeerUnreachableError: HTTP transport failure.
        """
        async for _event in self._drive_peer_load(model_id):
            pass

    async def _drive_peer_load(self, model_id: str):
        """Run the peer model-load handshake, yielding lifecycle markers
        and recording the peer's progress into the coordinator cache.

        Yields:
          ``"cold_start"`` — once, when the peer was NOT already serving
            the model and polling begins. The streaming path turns this
            into a ``model_load`` stage_start (UI bar appears + stall
            watchdog suspends).
          ``"ready"`` — the peer is serving the model.

        Side effect: each poll that carries ``load_progress`` /
        ``prefill_progress`` (cross-peer state transparency) is written
        to the coordinator cache keyed by model_id, so the originator's
        ``/api/engine/v2/{load,prefill}_progress`` endpoints surface a
        peer load through the SAME UI poller as a local one.

        Raises the same typed errors as :meth:`ensure_peer_model_loaded`.
        """
        import asyncio as _asyncio
        import time as _time

        if not model_id:
            return

        # Phase 1: POST /load_model — synchronous handshake. Returns
        # status=ready when already loaded (common warm path) so we
        # skip the polling loop entirely.
        kick_url = f"{self._http_scheme()}://{self._peer_addr}/api/fabric/load_model"
        kick_body = json.dumps({"model_id": model_id}).encode("utf-8")
        kick_headers = self._fabric_headers_for(
            method="POST", path="/api/fabric/load_model", body=kick_body,
        )
        try:
            resp = await self._client.post(
                kick_url, content=kick_body, headers=kick_headers, timeout=30.0,
            )
        except httpx.TransportError as exc:
            raise PeerUnreachableError(
                f"fabric peer unreachable during load handshake: {exc}"
            ) from None

        if resp.status_code == 404:
            # Peer doesn't have this model on disk — its advertised
            # capability is stale. Drop it so subsequent dispatch
            # attempts route to a peer that actually has the file.
            self._invalidate_peer_capability(model_id)
            raise PeerModelMissingError(
                f"peer {self._peer_node_id} does not have model "
                f"{model_id!r} on disk: {resp.text[:200]}"
            )
        if resp.status_code >= 400:
            raise PeerProtocolError(
                f"fabric peer load_model returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )

        try:
            kick_data = resp.json()
        except Exception:
            raise PeerProtocolError(
                "fabric peer load_model returned non-JSON response"
            ) from None

        status = str(kick_data.get("status", ""))
        if status == "ready":
            log.debug(
                "fabric_peer_load_already_ready",
                peer=self._peer_node_id, model=model_id,
            )
            yield "ready"
            return

        # Cold load: signal the streaming path to open a model_load
        # stage before we settle into the (potentially 30-120s) poll.
        yield "cold_start"

        # Phase 2: poll /load_status. The receiver's background load
        # task drives the state machine; we just wait for it to settle.
        #
        # URL-encode model_id with safe="" so chars like '+' / ' ' / '&'
        # / '=' don't break signature canonicalisation. The sender's
        # signed path string must byte-for-byte match what the receiver
        # reconstructs from ASGI's raw query_string. httpx URL-encodes
        # safe chars on the wire (most model IDs are alphanumeric +
        # ".-_" which round-trip identically), so we encode explicitly
        # at construction time and sign the encoded form.
        from urllib.parse import quote as _quote
        encoded_model_id = _quote(model_id, safe="")
        status_path = f"/api/fabric/load_status?model_id={encoded_model_id}"
        status_url = (
            f"{self._http_scheme()}://{self._peer_addr}{status_path}"
        )
        deadline = _time.monotonic() + _LOAD_TIMEOUT_S
        log.info(
            "fabric_peer_load_polling",
            peer=self._peer_node_id, model=model_id,
            current=kick_data.get("current_model"),
        )

        while _time.monotonic() < deadline:
            await _asyncio.sleep(_LOAD_POLL_INTERVAL_S)
            status_headers = self._fabric_headers_for(
                method="GET",
                path=status_path,
                body=b"",
            )
            try:
                poll_resp = await self._client.get(
                    status_url, headers=status_headers, timeout=10.0,
                )
            except httpx.TransportError as exc:
                # Transient network blip mid-poll — keep polling until
                # the deadline. Don't fail eagerly; the load could
                # finish while the network heals.
                log.debug(
                    "fabric_peer_load_poll_blip",
                    peer=self._peer_node_id, model=model_id, error=str(exc)[:120],
                )
                continue

            if poll_resp.status_code >= 400:
                raise PeerProtocolError(
                    f"fabric peer load_status returned {poll_resp.status_code}: "
                    f"{poll_resp.text[:200]}"
                )

            try:
                poll_data = poll_resp.json()
            except Exception as exc:
                # Malformed status from a peer — log so the operator
                # can see "peer X is sending non-JSON" rather than
                # silently burning the poll loop with no telemetry.
                log.debug(
                    "fabric_peer_poll_non_json",
                    error=str(exc)[:160],
                    body=poll_resp.text[:160],
                )
                continue

            # Cross-peer state transparency: stash whatever progress the
            # peer reported so the originator's progress endpoints can
            # render it. Cheap + best-effort; never affects the load.
            self._record_peer_progress(model_id, poll_data)

            poll_status = str(poll_data.get("status", ""))
            if poll_status == "ready":
                log.info(
                    "fabric_peer_load_complete",
                    peer=self._peer_node_id, model=model_id,
                )
                yield "ready"
                return
            if poll_status == "failed":
                reason = str(poll_data.get("reason", "unknown"))
                # Subprocess startup crashes (SIGSEGV / OOM-class /
                # incompatible-quant) are structural for this peer:
                # the same load attempt will fail again next dispatch,
                # so the dropdown shouldn't keep offering this model
                # on this peer until the heartbeat reasserts it.
                # Matches the 404 (model-missing) path's invalidation
                # contract — operators see one clean removal instead
                # of repeated PeerProtocolError → ConnectError →
                # "response already started" cascades on retry.
                if _is_structural_load_failure(reason):
                    self._invalidate_peer_capability(model_id)
                    log.warning(
                        "fabric_peer_load_structural_failure",
                        peer=self._peer_node_id, model=model_id,
                        reason=reason[:200],
                    )
                raise PeerProtocolError(
                    f"fabric peer {self._peer_node_id} failed to load "
                    f"{model_id!r}: {reason}"
                )
            if poll_status == "superseded":
                # Manager moved on between load completion and our poll.
                # Re-kick the load — the operator on the peer may have
                # swapped to a different model.
                log.info(
                    "fabric_peer_load_superseded_retrying",
                    peer=self._peer_node_id, model=model_id,
                )
                async for _event in self._drive_peer_load(model_id):
                    yield _event
                return
            # status in {loading, unknown} → keep polling.

        raise PeerProtocolError(
            f"fabric peer {self._peer_node_id} did not load {model_id!r} "
            f"within {_LOAD_TIMEOUT_S:.0f}s"
        )

    def _record_peer_progress(self, model_id: str, poll_data: dict) -> None:
        """Forward a peer's load/prefill progress (embedded in a
        load_status poll response) into the coordinator cache. Both
        snapshots are already in the wire shape produced by the shared
        builders on the receiver, so the originator's progress endpoints
        relay them verbatim. Best-effort: never raises into the load.
        """
        if self._coordinator is None or not isinstance(poll_data, dict):
            return
        # The UI polls /api/engine/v2/load_progress with
        # ``app.state.currentModel``, which is the bare model name on an
        # auto-routed call but the suffixed ``<model>@fabric:<node>`` when
        # the operator pinned this peer. ``model_id`` here is the clean
        # name dispatched to the peer. Record under both so the bar lines
        # up regardless of how the model was selected.
        keys = {model_id}
        if self._pinned_wire_name:
            keys.add(self._pinned_wire_name)
        try:
            load = poll_data.get("load_progress")
            if isinstance(load, dict) and load.get("active"):
                rec = getattr(self._coordinator, "record_peer_load_progress", None)
                if callable(rec):
                    for k in keys:
                        rec(k, load)
            prefill = poll_data.get("prefill_progress")
            if isinstance(prefill, dict) and prefill.get("active"):
                rec = getattr(self._coordinator, "record_peer_prefill_progress", None)
                if callable(rec):
                    for k in keys:
                        rec(k, prefill)
        except Exception:
            log.debug(
                "fabric_peer_progress_record_failed",
                peer=self._peer_node_id, model=model_id, exc_info=True,
            )

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        """Non-streaming proxy. Builds an OpenAI-shape payload and
        forwards to the peer's /v1/chat/completions. Returns the
        normalised response.
        """
        import time as _time

        # Pre-dispatch load gate. No-ops when the peer is already
        # serving this model. Surfaces load failures distinctly from
        # dispatch failures so the operator gets actionable error
        # messages instead of a generic 500 mid-stream.
        await self.ensure_peer_model_loaded(request.model)

        payload = self._build_payload(request, stream=False)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        # Dispatch goes through /api/fabric/inference, a purpose-built
        # endpoint that bypasses the receiver's heavy chat orchestration
        # (mode classification, memory, knowledge packs, narrative,
        # tools, post-stream hooks). The OpenAI /v1/chat/completions
        # endpoint is for end users — peers get their own dedicated
        # data-plane URL so receiver-side orchestration deps can't kill
        # cross-peer chats (2026-05-23 incident root cause).
        url = f"{self._http_scheme()}://{self._peer_addr}/api/fabric/inference"
        request_id = f"req-{uuid.uuid4().hex[:16]}"
        started_ms = _time.monotonic() * 1000.0

        try:
            resp = await self._client.post(
                url, content=body,
                headers=self._fabric_headers(body, request_id=request_id),
                timeout=120.0,
            )
        except httpx.TransportError as exc:
            log.warning(
                "fabric_chat_proxy_failed",
                peer=self._peer_node_id, model=request.model, error=str(exc)[:200],
            )
            raise PeerUnreachableError(f"fabric peer unreachable: {exc}") from None

        if resp.status_code >= 400:
            log.warning(
                "fabric_chat_proxy_status",
                peer=self._peer_node_id, status=resp.status_code,
                body=resp.text[:300],
            )
            # Capability-staleness detection: if the peer's 4xx says
            # "I don't have that model", drop the advertisement and
                # raise the typed error so callers can react.
            if _is_model_missing_response(resp.status_code, resp.text):
                self._invalidate_peer_capability(request.model)
                raise PeerModelMissingError(
                    f"peer {self._peer_node_id} no longer has model "
                    f"{request.model!r}: {resp.text[:200]}"
                )
            raise PeerProtocolError(
                f"fabric peer returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        # Phase 10 — record latency for the scoring function.
        if self._coordinator is not None:
            try:
                elapsed_ms = _time.monotonic() * 1000.0 - started_ms
                self._coordinator.record_peer_latency(
                    self._peer_node_id, kind="llm.inference",
                    latency_ms=elapsed_ms,
                )
            except Exception:
                log.debug("fabric_peer_latency_record_failed", exc_info=True)
        parsed = _parse_openai_chat_response(data, fallback_model=request.model)
        if self._pinned_wire_name:
            parsed.model = self._pinned_wire_name
        return parsed

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        # Pre-dispatch load gate, surfaced as a ``model_load`` stage so
        # the sender UI shows the SAME progress bar + suspends the same
        # stall watchdog as a local cold load (cross-peer state
        # transparency). The peer's live progress rides into the
        # coordinator cache via _drive_peer_load → the originator's
        # /api/engine/v2/load_progress endpoint. A warm peer yields no
        # cold_start, so the stage never opens and there's no flicker.
        #
        # Load failures still raise cleanly before any inference chunk —
        # _drive_peer_load raises, we close the stage as failed, then
        # re-raise so the caller's error handling is unchanged.
        import time as _time

        stage_id = f"stg_fabric_load_{uuid.uuid4().hex[:8]}"
        stage_open = False
        load_started_ms = 0.0
        try:
            async for event in self._drive_peer_load(request.model):
                if event == "cold_start":
                    stage_open = True
                    load_started_ms = _time.monotonic() * 1000.0
                    yield InternalStreamChunk(augmentum={
                        "stage_start": {
                            "id": stage_id,
                            "stage": "model_load",
                            "label": "Loading model",
                            "detail": request.model,
                            "started_at": _time.time(),
                            "request_id": "",
                        },
                        "heartbeat": True,
                        "phase": "model_load",
                        "phase_status": "starting",
                        "t": _time.time(),
                    })
                elif event == "ready" and stage_open:
                    yield InternalStreamChunk(augmentum={
                        "stage_complete": {
                            "id": stage_id,
                            "stage": "model_load",
                            "success": True,
                            "duration_ms": int(
                                _time.monotonic() * 1000.0 - load_started_ms
                            ),
                            "detail": request.model,
                            "error": "",
                            "request_id": "",
                        },
                    })
                    stage_open = False
        except BaseException as exc:
            if stage_open:
                yield InternalStreamChunk(augmentum={
                    "stage_complete": {
                        "id": stage_id,
                        "stage": "model_load",
                        "success": False,
                        "duration_ms": int(
                            _time.monotonic() * 1000.0 - load_started_ms
                        ),
                        "detail": request.model,
                        "error": str(exc)[:200],
                        "request_id": "",
                    },
                })
            raise

        async for chunk in self._chat_stream_after_load(request):
            yield chunk

    async def _poll_prefill_into_cache(self, model_id: str) -> None:
        """Mirror the peer's prefill progress into the coordinator cache
        during the prefill window (cross-peer state transparency, P2).

        Started when the peer relays a ``prefill`` stage_start; cancelled
        on the first content delta / stream end (see
        :meth:`_chat_stream_after_load`). Reuses /api/fabric/load_status
        — which embeds ``prefill_progress`` while the model is READY and a
        prompt is processing — and :meth:`_record_peer_progress`, which
        records under both the clean + pinned model names. The
        originator's /api/engine/v2/prefill_progress endpoint reads that
        cache, so the existing UI poller renders the bar identically to a
        local prefill. Best-effort: never raises into the stream.
        """
        import asyncio as _asyncio
        import time as _time
        from urllib.parse import quote as _quote

        if self._coordinator is None or not model_id:
            return
        encoded = _quote(model_id, safe="")
        status_path = f"/api/fabric/load_status?model_id={encoded}"
        status_url = f"{self._http_scheme()}://{self._peer_addr}{status_path}"
        # Bounded by the same ceiling as a cold load — a prefill that
        # outruns it is pathological and the bar simply stops updating.
        deadline = _time.monotonic() + _LOAD_TIMEOUT_S
        while _time.monotonic() < deadline:
            headers = self._fabric_headers_for(
                method="GET", path=status_path, body=b"",
            )
            try:
                resp = await self._client.get(
                    status_url, headers=headers, timeout=10.0,
                )
                if resp.status_code < 400:
                    self._record_peer_progress(model_id, resp.json())
            except _asyncio.CancelledError:
                raise
            except Exception:
                # Transient blip / non-JSON / record failure — keep going;
                # the cache entry just goes stale (8s TTL) and the bar
                # pauses. Never let a side-poll error touch the stream.
                log.debug(
                    "fabric_prefill_poll_blip",
                    peer=self._peer_node_id, model=model_id, exc_info=True,
                )
            await _asyncio.sleep(_PREFILL_POLL_INTERVAL_S)

    async def _chat_stream_after_load(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Streaming proxy. Forwards to the peer and yields chunks as
        they arrive. Cancellation propagates two ways:

          1. **TCP-close (Link 1, primary)**: when the caller cancels
             this async generator, the ``async with`` exit closes the
             httpx stream context, which TCP-FINs the peer connection.
             The peer's StreamingResponse + chat_egress see the
             disconnect and cancel their own LLM generation.
          2. **WS backstop (Phase 9.4)**: we also mint a request_id
             at the top of the call, send it in the X-Fabric-Request-Id
             header (covered by body integrity), and on
             ``asyncio.CancelledError`` push a MSG_CANCEL_REQUEST
             envelope to the peer over the existing WS. Belt-and-
             suspenders: if TCP-close is slow to propagate (proxy in
             the middle, kernel buffer), the WS message arrives
             instantly and the peer cancels its in-flight handler.
        """
        import asyncio as _asyncio

        payload = self._build_payload(request, stream=True)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        # Dispatch via the dedicated /api/fabric/inference endpoint —
        # see chat() for the rationale.
        url = f"{self._http_scheme()}://{self._peer_addr}/api/fabric/inference"
        request_id = f"req-{uuid.uuid4().hex[:16]}"

        # Track whether the stream completed naturally. Async-
        # generator cleanup can come from CancelledError (caller's
        # task cancelled) OR GeneratorExit (caller called aclose())
        # OR the receiver returning early. Both should fire the
        # backstop; matching exception types is fragile. A simple
        # "did we reach the end of the loop" flag is robust.
        completed_normally = False
        # Cross-peer prefill transparency (P2): a side-poll task that
        # mirrors the peer's prefill progress into the coordinator cache.
        # Started when the peer relays a ``prefill`` stage_start, stopped
        # on the first content delta (prefill is done) / stream end.
        prefill_task: _asyncio.Task | None = None
        try:
            async with self._client.stream(
                "POST", url, content=body,
                headers=self._fabric_headers(body, request_id=request_id),
                timeout=600.0,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                    log.warning(
                        "fabric_stream_proxy_status",
                        peer=self._peer_node_id, status=resp.status_code, body=body,
                    )
                    if _is_model_missing_response(resp.status_code, body):
                        self._invalidate_peer_capability(request.model)
                        raise PeerModelMissingError(
                            f"peer {self._peer_node_id} no longer has model "
                            f"{request.model!r}: {body}"
                        )
                    raise PeerProtocolError(
                        f"fabric peer returned {resp.status_code}: {body}"
                    )

                async for chunk in _parse_openai_sse_stream(resp, request.model):
                    if self._pinned_wire_name and chunk.model:
                        chunk.model = self._pinned_wire_name
                    # Relayed ``prefill`` stage_start → start mirroring the
                    # peer's prefill progress into the cache. The first
                    # content delta means prefill finished; stop the poll.
                    stage_start = (chunk.augmentum or {}).get("stage_start")
                    if (prefill_task is None
                            and isinstance(stage_start, dict)
                            and stage_start.get("stage") == "prefill"):
                        prefill_task = _asyncio.create_task(
                            self._poll_prefill_into_cache(request.model)
                        )
                    if chunk.content_delta and prefill_task is not None:
                        prefill_task.cancel()
                        prefill_task = None
                    yield chunk
                completed_normally = True
        except httpx.TransportError as exc:
            log.warning(
                "fabric_stream_proxy_failed",
                peer=self._peer_node_id, model=request.model, error=str(exc)[:200],
            )
            raise PeerUnreachableError(
                f"fabric peer unreachable mid-stream: {exc}"
            ) from None
        finally:
            # Tear down the prefill side-poll if it's still running (stream
            # ended during prefill, or cancelled before first token).
            if prefill_task is not None and not prefill_task.done():
                prefill_task.cancel()
            # WS-backstop cancel: only when the stream did NOT
            # complete naturally (caller cancelled / closed early).
            # Fire-and-forget; coordinator absorbs send errors.
            # Skipped when there's no coordinator reference (legacy /
            # test paths) or no identity.
            if (not completed_normally
                    and self._coordinator is not None
                    and self._identity is not None):
                try:
                    from augmentum.fabric.protocol import MSG_CANCEL_REQUEST
                    await self._coordinator.send_to_peer(
                        self._peer_node_id,
                        msg_type=MSG_CANCEL_REQUEST,
                        payload={"request_id": request_id},
                    )
                except Exception:
                    # Never let backstop-send raise into the
                    # cancellation propagation. The original
                    # CancelledError/GeneratorExit propagation is
                    # what matters; the WS hint is best-effort.
                    log.debug(
                        "fabric_cancel_backstop_failed",
                        peer=self._peer_node_id, request_id=request_id,
                        exc_info=True,
                    )

    async def list_models(self) -> list[ModelInfo]:
        """Just the one model this FabricBackend was built for. The
        coordinator already knows the full peer model list; we don't
        re-query it on every backend instantiation.
        """
        cap = self._capability
        return [
            ModelInfo(
                name=cap.model_id,
                model=cap.model_id,
                size=0,
                digest="",
                modified_at="",
                details={
                    "family": cap.model_family,
                    "params_b": cap.params_b,
                },
                context_length=cap.ctx_max,
            )
        ]

    async def show_model(self, name: str) -> ModelDetails:
        """Return capability-derived details for the model. Used by
        the context-length probe and a couple of UI surfaces. We
        don't proxy the actual /v1/models/{name} call -- the peer's
        capability advertisement is the source of truth.
        """
        cap = self._capability
        return ModelDetails(
            modelfile="",
            parameters="",
            template="",
            details={
                "family": cap.model_family,
                "parameter_size": f"{cap.params_b}B" if cap.params_b else "",
                "context_window": cap.ctx_max,
            },
            model_info={},
            family=cap.model_family,
            parameter_size=f"{cap.params_b}B" if cap.params_b else "",
            quantization_level="",
        )

    # ── Internals ─────────────────────────────────────────────────

    def _invalidate_peer_capability(self, model_id: str) -> None:
        """Drop the stale llm.inference capability for ``model_id``
        from the coordinator's view of this peer. Called when the
        peer 4xxs with a model-missing signal — its heartbeat lied
        (or the operator swapped models between heartbeats).

        Next dispatch for the same model won't route to this peer
        (find_peers_with_capability filters by model_id, and that
        match is now gone). The peer's next heartbeat will refresh
        the list with its current actual capabilities, so this
        invalidation is self-healing.
        """
        if self._coordinator is None:
            return
        try:
            self._coordinator.invalidate_peer_capability(
                self._peer_node_id, kind="llm.inference", model_id=model_id,
            )
        except Exception:
            log.debug(
                "fabric_capability_invalidate_failed",
                peer=self._peer_node_id, model=model_id, exc_info=True,
            )

    def _http_scheme(self) -> str:
        """Always wss/https when the addr is bare (Caddy + TLS). When
        the addr already starts with a scheme, respect it."""
        if "://" in self._peer_addr:
            return self._peer_addr.split("://", 1)[0]
        return "https"

    def _fabric_headers_for(
        self, *, method: str, path: str, body: bytes,
    ) -> dict[str, str]:
        """Build signed headers for an arbitrary peer endpoint.

        Used by ``ensure_peer_model_loaded`` to call /load_model +
        /load_status. Mirrors ``_fabric_headers`` but parameterised
        on method + path so the SHA-256 body hash covers exactly what
        the peer's FabricPeerMiddleware will re-derive.

        Empty ``self._user_id`` is fine — internal dispatch sites
        (jobs, narrative refresh, draft_section, …) call into the
        fabric without a request context. The receiver-side middleware
        accepts an empty user_id_claim under the per-peer service
        user model. The only hard requirement is the signing identity.
        """
        if self._identity is None:
            return {"Content-Type": "application/json"}
        from augmentum.fabric.peer_middleware import build_signed_peer_headers

        signed = build_signed_peer_headers(
            identity=self._identity,
            user_id=self._user_id,
            method=method,
            path=path,
            body=body,
        )
        return {"Content-Type": "application/json", **signed}

    def _fabric_headers(self, body: bytes, *, request_id: str = "") -> dict[str, str]:
        """Headers identifying this request as fabric-routed.

        Phase 3.x produces signed headers consumed by the peer's
        FabricPeerMiddleware: sender (our node_id), user_id (who
        this request is on behalf of), timestamp, and an ed25519
        signature over the canonical request bytes. From Phase 3.y
        the signed canonical bytes include sha256(body) so the peer
        can detect body tampering. When identity isn't supplied
        (legacy constructor / tests) we fall back to the
        informational-only header set — the peer will 401.

        Empty ``self._user_id`` still produces a signed envelope (the
        ``X-Fabric-User-Id`` header is empty but the signature covers
        it). Internal dispatch sites without a request context
        (jobs, narrative refresh, draft_section, …) need this — the
        receiver's middleware accepts an empty user_id_claim under
        the per-peer service user model. Half-fixed in 1c213d1
        (receiver side); this is the matching sender-side change.

        Phase 9.3: ``request_id`` (when supplied) goes out as
        ``X-Fabric-Request-Id``. The peer-side middleware reads it
        + registers the in-flight asyncio.Task in the coordinator
        registry so a later MSG_CANCEL_REQUEST envelope can cancel
        the right handler.

        The ``body`` argument is the exact bytes that will be sent
        via ``httpx.post(content=body, ...)``. Don't use ``json=``
        on the httpx call when invoking this -- httpx's serialiser
        would produce different bytes than what we hashed.
        """
        base = {"Content-Type": "application/json"}
        if request_id:
            base["X-Fabric-Request-Id"] = request_id
        if self._identity is None:
            # Legacy unsigned path (tests only — production lifespan
            # always supplies identity). The peer 401s.
            return base

        from augmentum.fabric.peer_middleware import build_signed_peer_headers

        signed = build_signed_peer_headers(
            identity=self._identity,
            user_id=self._user_id,
            method="POST",
            # Must match the URL path that chat() / chat_stream() POST
            # to — receiver re-derives canonical bytes from its parsed
            # path and signature verification fails on mismatch.
            path="/api/fabric/inference",
            body=body,
        )
        return {**base, **signed}

    def _build_payload(
        self, request: InternalChatRequest, *, stream: bool,
    ) -> dict:
        """Build the OpenAI-shape payload to send to the peer.

        Forwards the full user-preference surface that the receiver's
        ``/api/fabric/inference`` endpoint reconstructs into its own
        ``InternalChatRequest`` — sampler params, stop sequences,
        native tool calling, and the whole thinking-control family
        (``think`` / ``chat_template_kwargs`` / ``preserve_thinking``
        / ``reasoning_effort``). The peer's local backend does the
        family-specific mapping (e.g. ``enable_thinking`` kwarg), so
        this stays a faithful relay, not a re-implementation.

        ``think`` is sent as an explicit boolean even when False:
        for always-on-by-default families (Qwen 3.x) the OFF state
        requires an explicit ``enable_thinking: false`` on the peer,
        so omission and False are NOT equivalent.

        Per-field ``None`` checks mirror the local backends — absent
        means "peer's default applies". Old receivers ignore unknown
        keys, so version skew degrades to the previous behavior
        instead of erroring.
        """
        payload: dict = {
            "model": request.model,
            "messages": _messages_to_openai(request.messages),
            "stream": stream,
            "think": bool(request.think),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.format:
            payload["format"] = request.format
        if request.chat_template_kwargs:
            payload["chat_template_kwargs"] = request.chat_template_kwargs
        if request.preserve_thinking is not None:
            payload["preserve_thinking"] = bool(request.preserve_thinking)
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.raw_options:
            payload["raw_options"] = request.raw_options
        if request.continue_last_assistant:
            payload["continue_last_assistant"] = True
        if request.is_background_task:
            payload["is_background_task"] = True
        if request.kv_session_key:
            payload["session_id"] = request.kv_session_key
        return payload


# ── OpenAI-shape ↔ Internal-shape helpers ─────────────────────────


def _messages_to_openai(messages: list[Message]) -> list[dict]:
    """Convert internal Message list to OpenAI chat-format dicts.

    ``images`` and ``thinking`` ride as sibling keys in our internal
    convention — the receiver is another Augmentum, and its
    ``fabric_inference`` coercion rebuilds ``Message`` objects from
    them, so the peer's own backend handles provider-shape conversion
    (mtmd multimodal arrays, ``reasoning_content`` round-trip).
    """
    out: list[dict] = []
    for m in messages:
        item: dict = {"role": m.role, "content": m.content}
        if getattr(m, "tool_call_id", None):
            item["tool_call_id"] = m.tool_call_id
        if getattr(m, "tool_calls", None):
            item["tool_calls"] = m.tool_calls
        if getattr(m, "images", None):
            item["images"] = m.images
        if getattr(m, "thinking", None):
            item["thinking"] = m.thinking
        out.append(item)
    return out


def _parse_openai_chat_response(
    data: dict, *, fallback_model: str,
) -> InternalChatResponse:
    """Parse the JSON body of a non-streaming /v1/chat/completions
    response into our InternalChatResponse shape.
    """
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("peer returned empty choices list")
    msg = choices[0].get("message", {})
    finish = choices[0].get("finish_reason")
    usage_d = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(usage_d.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage_d.get("completion_tokens", 0) or 0),
        total_tokens=int(usage_d.get("total_tokens", 0) or 0),
    )
    return InternalChatResponse(
        message=Message(
            role=msg.get("role", "assistant"),
            content=msg.get("content", "") or "",
            # Same bug class as LlamaCppBackend's earlier missing-field
            # gap — tool_calls and reasoning_content arrive in the peer's
            # OpenAI response but were silently dropped by this parser.
            # Result: any tool-calling client (Claude Code, Cursor agent
            # mode) routed via fabric got back empty content + null
            # tool_calls AND no reasoning surfaced for thinking models.
            tool_calls=msg.get("tool_calls"),
            thinking=msg.get("reasoning_content"),
        ),
        model=data.get("model", fallback_model),
        finish_reason=finish,
        usage=usage,
    )


async def _parse_openai_sse_stream(
    resp: httpx.Response, fallback_model: str,
) -> AsyncIterator[InternalStreamChunk]:
    """Iterate a /v1/chat/completions SSE stream + emit InternalStreamChunks.

    Tolerant of irregularities: missing fields, prefixed comments,
    `[DONE]` terminator. Mirrors the parser shape used by OpenAIBackend
    so behavior is consistent across local vs fabric-routed inference.
    """
    async for line in resp.aiter_lines():
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            yield InternalStreamChunk(done=True)
            return
        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError:
            log.debug("fabric_stream_bad_chunk", line=data_str[:200])
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        finish = choices[0].get("finish_reason")
        usage_d = obj.get("usage")
        chunk = InternalStreamChunk(
            content_delta=delta.get("content", "") or "",
            # The peer's local backend already ran family-specific
            # reasoning extraction; relay its ``reasoning_content``
            # verbatim. Without this, thinking models routed via
            # fabric silently lose their entire reasoning stream
            # (the UI shows no thinking block at all).
            thinking_delta=delta.get("reasoning_content", "") or "",
            role=delta.get("role"),
            finish_reason=finish,
            usage=(
                Usage(
                    prompt_tokens=int(usage_d.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage_d.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage_d.get("total_tokens", 0) or 0),
                )
                if usage_d else None
            ),
            model=obj.get("model", fallback_model),
            done=False,
        )
        # Relay the peer's augmentum metadata. The receiver forwards its
        # stage/status events (notably the ``prefill`` stage_start) as a
        # top-level ``augmentum`` block — mirror it straight back into
        # ``chunk.augmentum`` so it bubbles to the UI's stage handler and
        # drives the SAME prefill progress bar + stall-watchdog suspension
        # a local prefill shows (cross-peer state transparency, P2).
        aug = obj.get("augmentum")
        merged_aug = dict(aug) if isinstance(aug, dict) else {}
        # Native tool-call deltas ride in delta.tool_calls, mirroring
        # OpenAICompatBackend's parser — chat_egress's accumulator reads
        # them from chunk.augmentum.
        tc_deltas = delta.get("tool_calls")
        if tc_deltas:
            merged_aug["tool_calls"] = tc_deltas
        if merged_aug:
            chunk.augmentum = merged_aug
        yield chunk
