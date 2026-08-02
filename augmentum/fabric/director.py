"""RoutingDirector: pick which peer (or local) should serve a request.

Lives on ``app.state.fabric_director`` when fabric is enabled. The
dispatch hook in ``openai_routes`` (and friends) consults this at
request entry; the director either says "stay local" (returns None,
the common case) or returns a FabricBackend wrapping the chosen peer.

Phase 3 policy is intentionally simple:

  1. If the local backend can serve the request: stay local. Always.
     (Local-first invariant -- the moment your own box can handle it,
     network costs and routing complexity are pure waste.)
  2. If no peer advertises the capability either: stay local. The
     local backend will fail with a clean error rather than the
     director silently dropping the request.
  3. Otherwise: pick the first peer advertising the model. Future
     phases add scoring (warm KV bonus, load aversion, latency, etc.)
     by replacing this with a ranked-selection function.

Critical invariant: ``maybe_route_llm`` returns ``None`` whenever it
can. Returning a peer routes the request through the fabric data
plane; returning None preserves the pre-fabric dispatch flow exactly.
A bug in the director that returns a peer when local could serve
silently degrades latency for every user. The local-first test in
``test_fabric_director`` is the load-bearing assertion here -- never
relax it without thinking very hard.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from augmentum.cast.render import (
    RenderJob,
    RenderRoute,
    capability_flag_for,
    tier_rank,
)
from augmentum.fabric.capabilities import (
    KIND_CAST_RENDER,
    KIND_IMAGE_GENERATION,
    KIND_KNOWLEDGE_SEARCH,
    KIND_LLM_INFERENCE,
    CastRenderCapability,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from augmentum.fabric.coordinator import FabricCoordinator
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)


class RoutingDirector:
    """Routing brain for fabric-aware request dispatch.

    Stateless apart from references to its source-of-truth singletons
    (coordinator for peer/capability state, http_client for FabricBackend
    instances it builds). Safe to query from any task; the underlying
    coordinator reads are lock-free dict lookups.
    """

    def __init__(
        self,
        coordinator: FabricCoordinator,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._coordinator = coordinator
        self._http_client = http_client
        # Identity is required for signing outbound peer requests. We
        # pull it from the coordinator at request time rather than at
        # construct time so phase 3.x signing works even if the
        # coordinator was initialised before the identity was loaded
        # (defensive against future startup-order changes; today it's
        # always set before this director is built).
        # See ``maybe_route_llm`` for the wiring point.

    async def maybe_route_llm(
        self,
        *,
        model: str,
        user_id: str,
        session_id: str,
        local_backend: ModelBackend,
        local_known: bool = True,
    ) -> ModelBackend | None:
        """Decide whether to route this LLM request to a peer.

        Returns ``None`` when local should serve (the common case).
        Returns a ``FabricBackend`` instance when a remote peer is
        selected.

        Decision tree:

          - ``local_backend`` is already a ``FabricBackend`` → we're
            already routing; return None to avoid recursive ping-pong.
          - ``local_known=True`` → local map has the model; local-first
            is the load-bearing invariant, return None.
          - Otherwise → search connected peers for the model. First
            match wins (Phase 10 scoring kicks in when 2+ candidates).
            Returns None when no peer advertises it.

        ``local_known`` background: ``provider_registry.resolve_backend_
        for_model`` returns ``(default_backend, original_name)`` even
        when the model isn't in ``_model_map`` (it logs
        ``model_not_in_map_using_default`` and forges ahead). The
        director can't tell the difference from the (backend, model)
        tuple alone. Callers pass ``local_known=clean_model in
        registry._model_map`` to surface the signal — without it, a
        peer-only model name falls through to the local default
        backend's chat call, which fails with "no model selected" or
        a connect error to a non-running llama-server.
        """
        from augmentum.models.fabric_backend import FabricBackend

        # Already routing? Don't re-route. Caller should never pass a
        # FabricBackend in here, but defensive-by-default — a confused
        # caller doing so would otherwise ping-pong forever.
        if isinstance(local_backend, FabricBackend):
            return None

        # Local-first. When the resolver's local map has the model, we
        # NEVER override to a peer — even if peers advertise the same
        # model. This is the load-bearing invariant; the local_first test
        # in test_fabric_director is the canonical check.
        if local_known:
            return None

        # Local doesn't have it; try peers.
        log.debug(
            "fabric_routing_local_unknown_model",
            model=model, user_id=user_id, session_id=session_id,
        )
        candidates = self._find_llm_peers(model)
        if not candidates:
            # No peer can serve it either. Return None; the resolver
            # helper raises ModelUnavailableError to surface this
            # cleanly to the operator with the peer diagnostic.
            log.info(
                "fabric_routing_no_peer_match",
                model=model,
                user_id=user_id,
                session_id=session_id,
                connected_peers=self._coordinator.connected_peer_ids(),
            )
            return None

        # Phase 10 — score-based selection. Local-first is preserved
        # by the earlier _local_can_serve gate; when we reach here we
        # already know local can't serve, so we're ranking peers
        # against each other on (free capacity, observed latency,
        # cost). First-match was correct when there was one signal
        # (capability match); now there are multiple.
        chosen_node_id, chosen_capability = self._score_llm_candidates(
            model, candidates,
        )
        log.info(
            "fabric_routing_to_peer",
            model=model,
            peer_node_id=chosen_node_id,
            user_id=user_id,
            session_id=session_id,
            candidate_count=len(candidates),
        )

        # Build the FabricBackend lazily here (rather than at peer-
        # connect time) so each request gets a fresh wrapper bound to
        # the current peer state. Cheap construction; no I/O.
        from augmentum.models.fabric_backend import FabricBackend

        peer_state = self._coordinator.peer_state(chosen_node_id)
        if peer_state is None or peer_state.paired is None:
            # Defensive: peer was unregistered between candidate search
            # and FabricBackend construction. Fall through to local.
            return None

        return FabricBackend(
            http_client=self._http_client,
            peer_node_id=chosen_node_id,
            peer_addr=peer_state.paired.addr,
            advertised_capability=chosen_capability,
            # Phase 3.x: propagate signing identity + originating
            # user so the receiving peer's FabricPeerMiddleware
            # authenticates the proxied request. Identity comes from
            # the coordinator (set at lifespan startup); user_id
            # comes from the dispatch context.
            identity=self._coordinator._identity,
            user_id=user_id,
            # Phase 9.2/9.4: coordinator reference enables the
            # WS-backstop cancellation send when the caller cancels
            # the async generator mid-stream.
            coordinator=self._coordinator,
        )

    def _find_llm_peers(self, model: str):
        """Return [(node_id, capability)] for peers advertising the model.

        Empty list when nobody has it. Coordinator filters to connected
        peers by default; we don't bother attempting unreachable peers.

        Emits a structured info log on every call so the operator can
        diagnose "the UI shows the peer model but the request fails"
        without re-instrumenting. Demote to debug once fabric routing
        is past the beta-stability window.
        """
        all_llm = self._coordinator.find_peers_with_capability(KIND_LLM_INFERENCE)
        matches = [
            (node_id, cap)
            for node_id, cap in all_llm
            if getattr(cap, "model_id", "") == model
        ]
        if not matches:
            log.info(
                "fabric_find_llm_peers_no_match",
                wanted=model,
                connected_peers=self._coordinator.connected_peer_ids(),
                advertised_models=[getattr(c, "model_id", "") for _, c in all_llm],
            )
        return matches

    async def route_llm_to_pinned_peer(
        self,
        *,
        model: str,
        peer_id_prefix: str,
        user_id: str,
        session_id: str,
    ) -> ModelBackend | None:
        """Route an LLM request to a *specific* peer the operator picked.

        ``peer_id_prefix`` is the short node-id (first 12 chars by
        convention) that ``/api/tags`` baked into the dropdown entry's
        ``@fabric:<id>`` suffix. We do a prefix match against currently-
        connected peers so the wire form stays human-tolerable; if the
        prefix is ambiguous or no peer matches, return None and the
        resolver raises ``ModelUnavailableError`` with a diagnostic.

        Unlike :meth:`maybe_route_llm`, this path does NOT consult the
        scoring weights — the user already made the call. It also does
        NOT honour local-first: a pinned dispatch is the user saying
        "use that box," even if the local engine has the model.
        Disconnected target → None (hard fail). Returns ``None`` for
        all the "can't satisfy" cases; resolver translates that into
        the typed error.
        """
        from augmentum.models.fabric_backend import FabricBackend

        connected = self._coordinator.connected_peer_ids()
        matches = [n for n in connected if n.startswith(peer_id_prefix)]
        if len(matches) != 1:
            log.info(
                "fabric_pinned_peer_unresolved",
                wanted_prefix=peer_id_prefix,
                model=model,
                connected_peers=connected,
                match_count=len(matches),
            )
            return None
        chosen_node_id = matches[0]

        peer_state = self._coordinator.peer_state(chosen_node_id)
        if peer_state is None or peer_state.paired is None:
            return None

        # Confirm the peer still advertises the model. Heartbeats can
        # drop a model between dropdown emit and request dispatch (the
        # user took 20 min, the peer unloaded, …) — fail clearly rather
        # than handing FabricBackend a model the peer will 404 on.
        chosen_capability = None
        for cap in peer_state.capabilities:
            if (
                getattr(cap, "kind", "") == KIND_LLM_INFERENCE
                and getattr(cap, "model_id", "") == model
            ):
                chosen_capability = cap
                break
        if chosen_capability is None:
            log.info(
                "fabric_pinned_peer_no_longer_advertises_model",
                peer_node_id=chosen_node_id, model=model,
            )
            return None

        log.info(
            "fabric_routing_to_pinned_peer",
            model=model, peer_node_id=chosen_node_id,
            peer_id_prefix=peer_id_prefix,
            user_id=user_id, session_id=session_id,
        )
        # ``pinned_wire_name`` echoes the suffixed form back to the
        # chat renderer via ``response.model``, so the persisted
        # ``model_used`` keeps the operator's pin and regenerate
        # stays bound to the same peer.
        pinned_wire_name = f"{model}@fabric:{chosen_node_id[:12]}"
        return FabricBackend(
            http_client=self._http_client,
            peer_node_id=chosen_node_id,
            peer_addr=peer_state.paired.addr,
            advertised_capability=chosen_capability,
            identity=self._coordinator._identity,
            user_id=user_id,
            coordinator=self._coordinator,
            pinned_wire_name=pinned_wire_name,
        )

    def peer_diagnostic_for_llm(self, model: str) -> dict[str, Any]:
        """Build the operator-facing diagnostic for an LLM-routing miss.

        Surfaces what the director actually saw: which peers are paired,
        which are connected vs offline, and what each one advertises in
        its current LLM capability list. Consumed by
        ``ProviderRegistry.resolve_backend_with_fabric`` when raising
        ``ModelUnavailableError`` so the route layer can render a
        meaningful error to the operator instead of letting a fallback
        backend produce a confusing upstream message.
        """
        paired = self._coordinator.known_peer_ids()
        connected = self._coordinator.connected_peer_ids()
        peers: dict[str, dict[str, Any]] = {}
        for node_id in paired:
            state = self._coordinator.peer_state(node_id)
            if state is None:
                continue
            llm_models = [
                getattr(c, "model_id", "")
                for c in state.capabilities
                if getattr(c, "kind", "") == KIND_LLM_INFERENCE
            ]
            hostname = ""
            if state.paired is not None:
                hostname = state.paired.hostname or ""
            peers[node_id] = {
                "connected": state.connected,
                "hostname": hostname,
                "llm_models": llm_models,
                "advertises_wanted": model in llm_models,
            }
        return {
            "wanted_model": model,
            "connected_peers": connected,
            "offline_peers": [n for n in paired if n not in connected],
            "peers": peers,
        }

    # Phase 10 — scoring weights. Defaults match operator intuition
    # "don't burn money when I have hardware": when a free peer can
    # serve, the cost penalty dominates the slot/latency bonuses
    # unless the free peer is dramatically worse on those axes.
    # Operators who want pure speed (cloud-first) or pure cost
    # (always cheapest) can tune via settings.fabric_score_* knobs
    # (future). The point release of these defaults is calibrated so
    # that an 8-free-slot $15/1M-token cloud peer loses to a
    # 0-free-slot local peer — money-saving as the strong prior.
    _SCORE_FREE_SLOTS_WEIGHT = 10.0       # +10 per free slot
    _SCORE_LATENCY_DIVISOR_MS = 100.0     # 100ms latency = -10 score
    _SCORE_COST_USD_PER_M_WEIGHT = 10.0   # $1/1M output tokens = -10 score
    _SCORE_VRAM_PRESSURE_MB_DIVISOR = 1024.0  # 1 GB free VRAM = +1 score

    def _score_llm_candidates(self, model: str, candidates):
        """Rank peer candidates + return the best (node_id, capability).

        Score components:
          - free_slots * SCORE_FREE_SLOTS_WEIGHT
            (more idle capacity = better; peer at queue depth 0
            beats a peer with 4 pending requests)
          - (latency_ms / SCORE_LATENCY_DIVISOR_MS) penalty
            (only when measured; un-measured peers default neutral)
          - (output_cost_per_1m_tokens * SCORE_COST_WEIGHT) penalty
            (cloud-routed peers carrying $/M token costs lose to
            zero-cost local-network peers)
          - VRAM-free bonus (informational; advertised in
            capability.device dict)

        Returns the highest-scoring candidate. Ties broken by
        first-seen order (list iteration is stable).
        """
        best_score = float("-inf")
        best = candidates[0]
        for node_id, cap in candidates:
            score = 0.0

            # Free-slot bonus.
            score += getattr(cap, "free_slots", 0) * self._SCORE_FREE_SLOTS_WEIGHT

            # Latency penalty (when measured).
            latency = self._coordinator.peer_latency_ms(node_id, "llm.inference")
            if latency is not None and latency > 0:
                score -= latency / self._SCORE_LATENCY_DIVISOR_MS

            # Cost penalty. Output costs dominate the bill for chat;
            # input costs are a secondary factor. Convert per-token
            # → per-1M-tokens to keep the weight unitful.
            out_cost = getattr(cap, "output_cost_per_token", 0.0) or 0.0
            cost_per_m = out_cost * 1_000_000.0
            score -= cost_per_m * self._SCORE_COST_USD_PER_M_WEIGHT

            # VRAM-free bonus (informational; small contribution).
            device = getattr(cap, "device", None) or {}
            vram_free = device.get("vram_free_mb", 0) if isinstance(device, dict) else 0
            score += (vram_free or 0) / self._SCORE_VRAM_PRESSURE_MB_DIVISOR

            log.debug(
                "fabric_score_candidate",
                peer_node_id=node_id, model=model,
                free_slots=getattr(cap, "free_slots", 0),
                latency_ms=latency, cost_per_m=cost_per_m,
                score=round(score, 2),
            )

            if score > best_score:
                best_score = score
                best = (node_id, cap)
        return best

    # ── Knowledge-pack search fanout ─────────────────────────────

    async def fanout_knowledge_search(
        self,
        *,
        query: str,
        requested_pack_ids: list[str],
        local_pack_ids: set[str],
        user_id: str,
        limit: int,
        local_search_fn: Callable[[list[str]], Awaitable[list[Any]]],
    ) -> list[Any]:
        """Run a knowledge search across local packs + peer packs in parallel.

        Splits ``requested_pack_ids`` by location:

          - packs present in ``local_pack_ids`` → run via ``local_search_fn``
            (the caller supplies its own bound PackManager.search to keep
            the fabric layer free of knowledge-layer imports).
          - packs advertised by a connected peer → run via
            :func:`knowledge_client.search_remote_packs` against that peer.
          - packs that nobody has → silently dropped (the existing local
            search already drops unknown pack_ids).

        All branches run concurrently with ``asyncio.gather`` so a slow
        peer doesn't block fast ones; failures are absorbed (caller gets
        an empty list from that peer rather than an exception). Returns
        the concatenated raw result lists -- the caller is responsible
        for re-ranking + deduping.

        ``local_search_fn`` is called with the local subset of pack_ids
        and must return whatever the caller's local search returns
        (typically ``list[PackResult]``). The return value of THIS
        function is opaque to the fabric layer; callers should expect
        a list of items in the same shape as ``local_search_fn`` plus
        deserialised peer results (one dict per remote chunk).
        """
        # Partition the request.
        local_subset: list[str] = []
        peer_assignments: dict[str, list[str]] = {}  # node_id → pack_ids
        peer_addrs: dict[str, str] = {}  # node_id → addr (for the search call)

        for pid in requested_pack_ids:
            if pid in local_pack_ids:
                local_subset.append(pid)
                continue
            # Not local — does any connected peer have it?
            peer_for_pack = self._find_pack_peer(pid)
            if peer_for_pack is not None:
                node_id, addr = peer_for_pack
                peer_assignments.setdefault(node_id, []).append(pid)
                peer_addrs[node_id] = addr

        # Nothing useful to do? Short-circuit.
        if not local_subset and not peer_assignments:
            return []

        tasks: list[Awaitable[list[Any]]] = []
        if local_subset:
            tasks.append(local_search_fn(local_subset))
        if peer_assignments:
            from augmentum.fabric.knowledge_client import (
                RemoteSearchError,
                search_remote_packs,
            )

            async def _do_peer(node_id: str, pids: list[str]) -> list[Any]:
                try:
                    return await search_remote_packs(
                        http_client=self._http_client,
                        identity=self._coordinator._identity,
                        user_id=user_id,
                        peer_addr=peer_addrs[node_id],
                        query=query,
                        pack_ids=pids,
                        limit=limit,
                    )
                except RemoteSearchError as exc:
                    log.info(
                        "fabric_knowledge_fanout_peer_failed",
                        peer_node_id=node_id, error=str(exc)[:160],
                    )
                    return []

            for node_id, pids in peer_assignments.items():
                tasks.append(_do_peer(node_id, pids))

        # Collect everything in parallel; any individual leg's failure
        # was already absorbed above into an empty list.
        gathered = await asyncio.gather(*tasks, return_exceptions=False)
        merged: list[Any] = []
        for leg in gathered:
            if isinstance(leg, list):
                merged.extend(leg)
        log.debug(
            "fabric_knowledge_fanout_result",
            local_count=len(local_subset), peer_count=len(peer_assignments),
            total_results=len(merged),
        )
        return merged

    def _find_pack_peer(self, pack_id: str) -> tuple[str, str] | None:
        """Return (node_id, addr) of the first connected peer that
        advertises this pack. ``None`` when nobody has it.

        Phase 6 picks the first match. A future enhancement could pick
        the lowest-latency / freshest-pack peer.
        """
        matches = self._coordinator.find_peers_with_capability(KIND_KNOWLEDGE_SEARCH)
        for node_id, cap in matches:
            if getattr(cap, "pack_id", "") != pack_id:
                continue
            state = self._coordinator.peer_state(node_id)
            if state is None or state.paired is None:
                continue
            return node_id, state.paired.addr
        return None

    # ── Image generation routing ──────────────────────────────────

    async def maybe_route_image(
        self,
        *,
        model_id: str,
        local_can_serve: bool,
    ) -> tuple[str, str] | None:
        """Decide whether to route an image-gen request to a peer.

        Returns ``(peer_node_id, peer_addr)`` when a connected peer
        advertises the requested model AND the local pipeline can't
        serve it. Returns ``None`` for stay-local (the common case).

        Local-first invariant: if local can serve the request, we
        always stay local — same shape as :meth:`maybe_route_llm`.
        Image generation involves transferring image bytes back
        across the fabric (typically a few MB per render), so the
        cost of routing is higher than for LLM streaming. The
        rule remains "use local whenever possible".

        ``local_can_serve`` is supplied by the caller because the
        image dispatch path doesn't expose a clean ABC the way
        LLM dispatch does — the route handler has to ask the
        pipeline_registry/cloud-resolver and pass the boolean in.

        The caller is responsible for the bytes proxy via
        :func:`augmentum.fabric.image_client.generate_image_via_peer`
        + writing the returned bytes into local image_output. The
        director only points at WHO to ask.
        """
        if local_can_serve:
            return None

        candidates = self._find_image_peers(model_id)
        if not candidates:
            log.debug(
                "fabric_image_routing_no_peer_match",
                model_id=model_id,
            )
            return None

        # Phase 7: pick the first eligible peer. Scoring (free GPU,
        # warm pipeline, latency) is a follow-up.
        chosen_node_id, _ = candidates[0]
        state = self._coordinator.peer_state(chosen_node_id)
        if state is None or state.paired is None:
            return None

        log.info(
            "fabric_image_routing_to_peer",
            model_id=model_id, peer_node_id=chosen_node_id,
            candidate_count=len(candidates),
        )
        return chosen_node_id, state.paired.addr

    def _find_image_peers(self, model_id: str):
        """Return [(node_id, capability)] for peers advertising the
        image model. Empty list when nobody has it. Coordinator
        filters to connected peers by default.
        """
        matches = self._coordinator.find_peers_with_capability(KIND_IMAGE_GENERATION)
        return [
            (node_id, cap)
            for node_id, cap in matches
            if getattr(cap, "model_id", "") == model_id
        ]

    # ── Cast-render routing ───────────────────────────────────────

    async def maybe_route_render(
        self,
        *,
        job: RenderJob,
    ) -> RenderRoute | None:
        """Decide where a render job should run.

        Returns:
          - ``RenderRoute(location="local", ...)`` when local can serve.
            Local-first invariant — same shape as ``maybe_route_llm``.
          - ``RenderRoute(location="peer", ...)`` when local can't but
            a connected peer can. Caller dispatches the render request
            over fabric.
          - ``None`` when neither local nor any peer can serve. Caller
            should surface a clean "no capable render node" error.

        Doesn't actually dispatch — pure routing decision. The render
        pipeline (future) builds on this; today the director's role is
        just to point at the right node.

        Phase 0: tier-aware selection. The chosen peer's tier is the
        only ranking signal — higher tier wins. Load-aware scoring
        slots in when we add per-peer load reporting; doesn't change
        the call surface.
        """
        flag = capability_flag_for(job.kind)
        if not flag:
            # Unknown kind — no node, local or peer, can be sure to
            # serve. Defensive: better to fail closed than dispatch
            # a job that nobody actually understands.
            log.debug("fabric_render_routing_unknown_kind", kind=job.kind)
            return None

        local_cap = self._local_render_capability()
        if local_cap is not None and getattr(local_cap, flag, False):
            return RenderRoute(
                location="local",
                node_id=self._coordinator._identity.node_id,
                tier=local_cap.tier,
            )

        candidates = self._find_render_peers(flag)
        if not candidates:
            log.debug(
                "fabric_render_routing_no_capable_node",
                kind=job.kind,
                flag=flag,
                target_device_id=job.target_device_id,
            )
            return None

        # Rank: higher tier first; ties broken by iteration order
        # (currently arbitrary, replaceable when proximity / load
        # scoring lands without changing this method's signature).
        candidates.sort(key=lambda nc: -tier_rank(nc[1].tier))
        chosen_node_id, chosen_cap = candidates[0]

        state = self._coordinator.peer_state(chosen_node_id)
        if state is None or state.paired is None:
            # Defensive: peer unregistered between candidate search and
            # route construction. No local fallback possible (we already
            # know local can't serve), so signal "nobody."
            return None

        log.info(
            "fabric_render_routing_to_peer",
            kind=job.kind,
            peer_node_id=chosen_node_id,
            peer_tier=chosen_cap.tier,
            target_device_id=job.target_device_id,
            candidate_count=len(candidates),
        )
        return RenderRoute(
            location="peer",
            node_id=chosen_node_id,
            tier=chosen_cap.tier,
        )

    def _local_render_capability(self) -> CastRenderCapability | None:
        """Pick the cast-render cap out of this node's local capability
        list. Empty / missing capability means the extractor hasn't run
        yet (very rare — happens during a narrow startup window).
        """
        for cap in self._coordinator.local_capabilities():
            if isinstance(cap, CastRenderCapability):
                return cap
        return None

    def _find_render_peers(self, flag: str):
        """Return [(node_id, capability)] for connected peers whose
        cast-render capability has ``flag`` set to True. Coordinator
        filters to connected peers by default — offline peers can't
        do work for us regardless of what they advertised earlier.
        """
        matches = self._coordinator.find_peers_with_capability(KIND_CAST_RENDER)
        return [
            (node_id, cap)
            for node_id, cap in matches
            if getattr(cap, flag, False)
        ]
