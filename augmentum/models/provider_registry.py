"""Backend discovery and selection."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.models.base import ModelBackend
from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.models.ollama import OllamaBackend
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import (
    ProviderProfile,
    get_profile,
    get_profile_for_url,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from augmentum.state.provider_store import ProviderStore

log = get_logger(__name__)

_MODEL_MAP_TTL = 30  # seconds
# When a probe round was DEGRADED (one or more backends failed / timed out),
# we serve a last-known-good map but want the next caller to re-probe soon
# instead of waiting the full TTL — otherwise a single cold-start cloud
# timeout right after restart hides a provider's models for 30s.
#
# Fix #3: the fast re-probe is reserved for a LOCAL backend dropping (the
# "the user's chat model just vanished" case worth spinning on). A slow but
# stable cloud catalog (openrouter's huge /models) is NOT worth re-probing
# every 5s — it just re-times-out and spams — so a cloud-only degradation
# rides the full TTL instead.
_MODEL_MAP_DEGRADED_RETRY_S = 5  # seconds

# Per-backend probe deadlines (fix #2). A local engine answers /models in
# milliseconds; a cloud provider with a large catalog (openrouter lists
# hundreds of models) routinely needs more than the old flat 6s, so it would
# perpetually time out → degrade every round. Give clouds a longer leash so
# they actually succeed instead of churning.
_PROBE_DEADLINE_LOCAL_S = 6.0
_PROBE_DEADLINE_CLOUD_S = 15.0

# Last-known-good model map persisted across restarts. The in-memory
# carry-forward (UNVERIFIED tier) only survives within one process; on a cold
# restart ``prev_map`` is empty, so a slow/cold cloud probe (deepseek,
# openrouter) leaves the user's models ABSENT for the first probe cycle — the
# "my models aren't all there for 30s, deepseek fails, I have to refresh
# twice" symptom. Persisting the map and loading it at startup as UNVERIFIED
# means the full last-working catalog is resolvable IMMEDIATELY after restart
# and the first probe merely confirms/refines it.
_MODEL_MAP_CACHE_FILE = "model_map_cache.json"

# Match a parameter-count token in a model name: 0.8B, 1B, 7B, 70B, 122B, etc.
# Decimal point optional; case-insensitive on the suffix.
_PARAM_COUNT_RE = re.compile(r"(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])")

# Suffix the /api/tags route attaches to peer-advertised LLM entries so
# the operator can pin a specific peer in the dropdown when 2+ peers
# serve the same model. The colon-prefix keeps this form distinct from
# the existing `<model>@<backend_key>` local-collision suffix, which
# uses backend keys like ``ollama`` / ``llamacpp`` / ``local-1234``
# (none contain a colon). See ``ollama_routes.py::ollama_tags`` for the
# emit side and ``_parse_fabric_pin`` below for the parse side.
_FABRIC_PIN_MARKER = "@fabric:"


def _parse_fabric_pin(model_name: str) -> tuple[str, str]:
    """Split a fabric-pinned model name into (clean_model, peer_id_prefix).

    Returns ``(model_name, "")`` when no pin is present, so callers can
    treat the result uniformly.

    The pin form is ``<model_id>@fabric:<node_id_short>``; ``node_id_short``
    is the first 12 chars of the peer's node_id, matching the hostname
    fallback in ``ollama_routes``. Director-side resolution does a
    prefix match against ``coordinator.connected_peer_ids()`` so the
    suffix doesn't need to be the full id.
    """
    idx = model_name.rfind(_FABRIC_PIN_MARKER)
    if idx < 0:
        return model_name, ""
    clean = model_name[:idx]
    peer_prefix = model_name[idx + len(_FABRIC_PIN_MARKER):]
    if not clean or not peer_prefix:
        # Malformed — treat as unpinned so the caller surfaces the
        # downstream "model not found" error rather than us inventing
        # an error type for a typo.
        return model_name, ""
    return clean, peer_prefix


def _strip_mode_prefix(model_name: str) -> str:
    """Strip a mode prefix (``d/``, ``a/``, ``n/``, ...) from a model name.

    Mode prefixes select HOW a request is handled, never WHERE it runs. On
    classified paths ``RequestClassifier._check_model_prefix`` consumes the
    prefix (mutating ``request.model``) before resolution ever sees it, so a
    prefixed name is never a literal model id anywhere in the platform.
    Classifier-less ingresses — the /v1/messages tools path (Claude Code),
    ``X-Augmentum-Mode`` header-override turns, internal callers — pass the
    prefixed name straight to the resolver, where it can't match any catalog
    entry and (with fabric on) raises ModelUnavailableError quoting the
    prefixed name. Stripping here mirrors the classifier's semantics at the
    resolution choke point so every ingress behaves the same.
    """
    from augmentum.classifier.router import MODE_PREFIXES

    for prefix in MODE_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def _strip_backend_suffix(model_name: str, backend_key: str) -> str:
    """Drop a load-balancer disambiguation suffix (``<model>@<backend>``).

    When one model id is served by more than one backend (e.g. several
    Gemini API-key siblings for load balancing), ``refresh_model_map`` stores
    the map key as ``"<model>@<backend_key>"`` so the picker can address each.
    The BACKEND, however, must receive the BARE model id — a leaked
    ``@backend`` rides into the upstream request and 404s the provider
    (observed 2026-07-29: ``gemini-2.5-flash@google-gemini-3`` →
    ``models/gemini-2.5-flash@google-gemini-3 is not found``). Strips ONLY the
    exact ``@<backend_key>`` the map appended, so a real ``@`` in a model id
    (none in practice) is never touched.
    """
    suffix = f"@{backend_key}"
    return model_name[: -len(suffix)] if model_name.endswith(suffix) else model_name


def _model_too_small(model_name: str, min_billions: float) -> bool:
    """Return True if the model's parameter count (parsed from its name) is
    below ``min_billions``. Returns False if no size token can be parsed —
    we never reject a model whose size we can't verify.
    """
    if min_billions <= 0:
        return False
    match = _PARAM_COUNT_RE.search(model_name)
    if not match:
        return False
    try:
        return float(match.group(1)) < min_billions
    except ValueError:
        return False


class ModelUnavailableError(RuntimeError):
    """No local backend serves the requested model AND no connected
    fabric peer advertises it.

    Raised by ``ProviderRegistry.resolve_backend_with_fabric`` before
    any dispatch happens, so the caller can surface a clean operator-
    facing error instead of letting the default backend produce a
    confusing upstream message ("model not found" from llama-server,
    connect-refused from a non-running engine, etc.).

    ``peer_diagnostic`` carries the structured state the resolver
    actually looked at: which peers are connected, which are offline,
    and what each peer's current LLM capability list contains. The UI
    + the route layer surface this so the operator can tell at a
    glance whether the peer dropped offline or its advertised model
    set drifted.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str,
        peer_diagnostic: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.model = model
        self.peer_diagnostic = peer_diagnostic


def create_backend_from_profile(
    profile: ProviderProfile | None,
    *,
    api_key: str = "",
    http_client: httpx.AsyncClient | None = None,
    chat_client: httpx.AsyncClient | None = None,
    provider_type: str = "",
    base_url: str = "",
    **kwargs: Any,
) -> ModelBackend:
    """Create a backend from a provider profile or explicit type."""
    # "anthropic" is the UI preset key; "claude" is the canonical adapter
    # type. Accept both so the dropdown selection routes to the native
    # backend rather than the OpenAI-compat shim.
    if provider_type in ("claude", "anthropic"):
        from augmentum.models.adapters.claude import ClaudeBackend

        return ClaudeBackend(
            client=http_client,
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com/v1",
            **kwargs,
        )
    # "google" is the UI preset key; "gemini" is the canonical adapter
    # type. Accept both so the dropdown selection and any already-stored
    # rows route to the native backend rather than the OpenAI-compat shim.
    if provider_type in ("gemini", "google"):
        from augmentum.models.adapters.gemini import GeminiBackend

        return GeminiBackend(
            client=http_client,
            api_key=api_key,
            base_url=base_url or "https://generativelanguage.googleapis.com",
            **kwargs,
        )
    # Default: OpenAI-compatible with optional profile
    return OpenAIBackend(
        client=http_client,
        base_url=base_url or (profile.base_url if profile else ""),
        api_key=api_key,
        profile=profile,
        chat_client=chat_client,
    )


class ProviderRegistry:
    """Manages available model backends and selects the right one per request."""

    def __init__(
        self, http_client: httpx.AsyncClient, *,
        chat_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._backends: dict[str, ModelBackend] = {}
        self._default = settings.default_backend
        self._http_client = http_client
        self._chat_http_client = chat_http_client or http_client
        self._model_map: dict[str, str] = {}  # model_name -> backend_key
        self._model_map_ts: float = 0
        # Three-tier freshness for ``_model_map`` entries. A map key is
        # VERIFIED when its owning backend responded on the last probe,
        # UNVERIFIED when it's a last-known-good catalog carried forward
        # because the backend went unresponsive, and "unknown" (absent)
        # when it's not in the map at all. This set holds the UNVERIFIED
        # keys; everything else in ``_model_map`` is verified. See
        # ``refresh_model_map`` and ``model_freshness``.
        self._model_unverified: set[str] = set()
        # Coalesce the "resolved to a stale entry" trace: warn once per model
        # while it stays unverified, reset when it returns to verified, so a
        # per-turn role resolve doesn't spam the log. See
        # ``_note_unverified_resolution``.
        self._unverified_served_logged: set[str] = set()
        # Explicit model->backend routing pins. Unlike ``_model_map``
        # (rebuilt by probing each backend's catalog), pins are set
        # deliberately when a model is loaded into a dedicated resident
        # slot — the secondary local engine ("Slot B"). A pin:
        #   1. wins in ``resolve_backend_for_model`` BEFORE the catalog
        #      map, so the pinned model routes to its slot's resident
        #      process with no cold-swap and no 30s partial-map race;
        #   2. is injected into the final map on every refresh, so the
        #      model appears exactly once in the picker (routed to its
        #      slot) instead of colliding into ``name@engine`` /
        #      ``name@engine_secondary`` variants.
        # Paired with ``_map_excluded_backends`` so the slot's backend
        # never contributes its (full, overlapping) GGUF catalog to the
        # probe — only its one pinned model is addressable.
        self._model_pins: dict[str, str] = {}  # model_name -> backend_key
        self._map_excluded_backends: set[str] = set()
        self._dismissed_urls: set[str] = set()  # URLs the user dismissed (persisted)
        self._provider_urls: set[str] = set()  # URLs from the providers DB table
        # Tracks the last error type per backend so periodic probes don't
        # warn on every cycle when a backend stays in the same failure
        # state. Cleared by the success path. See ``_probe`` in
        # ``rebuild_model_map``.
        self._probe_failure_state: dict[str, str] = {}
        # Coalesce the round-level ``model_map_degraded_round`` warning (fix
        # #1). The per-backend probe logs already warn-once-then-debug, but
        # the degraded-round SUMMARY fired every cycle — so a chronically slow
        # cloud backend machine-gunned the log every re-probe. Hold the set of
        # backends that failed the LAST round we warned about; only re-warn
        # when that set CHANGES (a new backend dropped, or one recovered),
        # else debug. ``None`` = last round was fully healthy.
        self._degraded_round_logged: frozenset[str] | None = None
        # Snapshot of the map last written to / read from disk, so we only
        # rewrite the cache file (and emit the ``models.changed`` UI event)
        # when the catalog actually changes — not every probe round.
        self._persisted_snapshot: dict[str, str] = {}
        # Immutable boot catalog — the model->backend map as loaded from the
        # on-disk cache at startup, NEVER mutated by a refresh. Cloud/runtime
        # backends register (``register_backend``) SECONDS after the cache
        # loads in ``__init__``; a ``refresh_model_map`` that runs in that
        # window drops their cached entries via the carry-forward "backend not
        # registered yet — let it drop" clause, leaving the models unroutable
        # until a (flaky) cloud probe re-adds them. ``register_backend`` restores
        # a backend's entries from this snapshot the instant it registers, so
        # the post-restart "cloud model not in map" race can't strand them.
        self._boot_catalog: dict[str, str] = {}
        # Fabric routing director — attached by ``fabric.lifespan`` after
        # both objects exist. None on solo installs where fabric is
        # disabled. ``resolve_backend_with_fabric`` consults this to
        # swap in peer backends for models that aren't in our local
        # ``_model_map``; ``resolve_backend_for_model`` stays unchanged
        # (callers that explicitly want the LOCAL-only resolver still
        # have it).
        self._fabric_director: Any = None
        # Per-user provider visibility (migration 305). Maps a DB-provider
        # backend key -> (owner_user_id, shared). Builtin/env backends are
        # NEVER in this map and are therefore visible to everyone. A private
        # provider (shared=False, non-empty owner) is visible + resolvable
        # ONLY to its owner. Populated by ``load_runtime_providers`` and the
        # provider routes; consulted by the model LIST surfaces
        # (/v1/models, /api/tags) and the user-facing RESOLVE path
        # (``resolve_backend_with_fabric``). This is a per-user *view* over
        # the process-global backend registry — the same pattern as the
        # multi-tenant pref-leak fix, not a per-user sub-registry.
        self._provider_meta: dict[str, tuple[str, bool]] = {}
        self._init_backends(http_client)
        self._load_persisted_model_map()

    def set_fabric_director(self, director: Any) -> None:
        """Attach the fabric routing director.

        Called once at lifespan startup from ``fabric.lifespan`` after
        both the registry and the director have been constructed.
        Idempotent; passing ``None`` clears the reference (used by
        ``stop_fabric``).
        """
        self._fabric_director = director

    def _local_backend_has_loaded(self, model_name: str) -> bool:
        """Source-of-truth check: is ``model_name`` actively loaded in any
        local managed backend?

        ``_model_map`` is populated by ``list_models()`` probes with a
        30s TTL and a 6s per-probe deadline. After a UI refresh kicks
        off a rebuild, a user who clicks send before the probes return
        gets routed against a partial/empty map and 30s'd out — even
        though the manager has the model actively serving. The manager's
        ``model_id`` is the authoritative answer to "is this model
        loaded RIGHT NOW", no probe required.
        """
        if not model_name:
            return False
        for backend in self._backends.values():
            manager = getattr(backend, "_manager", None)
            if manager is None:
                continue
            if getattr(manager, "model_id", "") == model_name:
                return True
        return False

    async def resolve_backend_with_fabric(
        self,
        model_name: str,
        *,
        user_id: str = "",
        session_id: str = "",
    ) -> tuple[ModelBackend | None, str]:
        """Resolve a backend, with fabric peer routing when applicable.

        Drop-in superset of ``resolve_backend_for_model`` for any LLM
        dispatch site:

          - On fabric-disabled solo installs (director is None) →
            identical behaviour to ``resolve_backend_for_model``.
          - On fabric-enabled installs where the requested model IS in
            the local ``_model_map`` → still local; no peer round-trip.
          - On fabric-enabled installs where the requested model isn't
            local but a connected peer advertises it → returns a
            ``FabricBackend`` wrapping that peer. The caller dispatches
            without knowing or caring.

        Background: ``resolve_backend_for_model`` silently falls back to
        the default backend with a ``model_not_in_map_using_default``
        warning when the requested name isn't in the map. Sites that
        called it and then dispatched directly (voice WS, narrative,
        reasoning, flow, tool layers, bug-finder, game-agent, ...) all
        had the same bug: a peer-only model fell through to the local
        default backend, which then errored with "no model selected" or
        a connect-refused on a non-running engine. Use this method
        instead at every dispatch site.

        ``user_id`` / ``session_id`` are passed through to the
        director's telemetry; empty strings are fine for internal
        callers (the receiver derives its local user identity from
        the signed envelope's sender_node_id under the per-peer
        service user model).
        """
        # Empty-name expansion when the user's primary chat model
        # carries an ``@fabric:`` pin. Internal callers (voice address
        # classifier, narrative refresh, ...) pass ``""`` to mean "use
        # the user's current chat model". ``resolve_backend_for_model``
        # then expands from ``primary_chat_model`` itself, but its
        # ``@``-suffix branch tries to look up ``"fabric:<peer_id>"`` as
        # a local backend key — which silently misses, falls back to the
        # default engine's first model, and lazy-loads a 40B local file
        # the user never asked for. Expanding HERE keeps the pin intact
        # so the ``_parse_fabric_pin`` step below routes correctly.
        if not model_name:
            primary = (getattr(settings, "primary_chat_model", "") or "").strip()
            if primary and _FABRIC_PIN_MARKER in primary:
                model_name = primary
        # Mode prefixes never reach the map/peer catalogs — consume one here
        # (covers fabric pins + the local_known/error paths below) exactly as
        # the classifier would have on a classified path.
        model_name = _strip_mode_prefix(model_name)
        # Phase 8.x — operator-pinned peer dispatch. When the model name
        # carries an ``@fabric:<id>`` suffix the user picked a specific
        # peer from the dropdown; bypass the scoring path and route to
        # exactly that peer. Hard-fail with a typed error when the peer
        # is offline or no longer advertises the model (the un-pinned
        # entry in the dropdown is the user's escape hatch for "I just
        # want it to work").
        clean_pin, peer_pin = _parse_fabric_pin(model_name)
        if peer_pin:
            director = self._fabric_director
            if director is None:
                raise ModelUnavailableError(
                    f"model {clean_pin!r} carries a fabric pin "
                    f"({peer_pin!r}) but fabric is disabled on this node",
                    model=clean_pin,
                    peer_diagnostic={
                        "connected_peers": [],
                        "offline_peers": [],
                        "peers": {},
                    },
                )
            peer_backend = await director.route_llm_to_pinned_peer(
                model=clean_pin,
                peer_id_prefix=peer_pin,
                user_id=user_id,
                session_id=session_id,
            )
            if peer_backend is not None:
                return peer_backend, clean_pin
            diag = director.peer_diagnostic_for_llm(clean_pin)
            raise ModelUnavailableError(
                f"pinned peer {peer_pin!r} cannot serve model "
                f"{clean_pin!r} (connected_peers="
                f"{diag.get('connected_peers') or '[]'})",
                model=clean_pin,
                peer_diagnostic=diag,
            )

        backend, clean_model = await self.resolve_backend_for_model(model_name)
        # Private-provider gate. Resolution is process-global, so a model
        # served by another user's private provider is reachable by id even
        # though it never appears in this user's model list. Refuse it at the
        # user-facing dispatch boundary. Only enforced when a real user_id is
        # present — internal/trusted callers (role resolution, background
        # tasks) pass "" and keep the unrestricted local resolver.
        if user_id and backend is not None:
            bkey = self._backend_key_for(backend)
            if bkey and not self.provider_visible_to(bkey, user_id):
                raise ModelUnavailableError(
                    f"model {clean_model!r} is served by a private provider "
                    f"not shared with this user",
                    model=clean_model,
                    peer_diagnostic={
                        "connected_peers": [],
                        "offline_peers": [],
                        "peers": {},
                    },
                )
        director = self._fabric_director
        if director is None:
            return backend, clean_model
        # Two-key check: a model is "locally known" if EITHER the clean
        # name OR the original ``@<backend>``-disambiguated key is in
        # the map. ``refresh_model_map`` registers multi-provider models
        # ONLY under their disambiguated form (e.g. ``z-ai/glm-5.1@nim``
        # + ``z-ai/glm-5.1@openrouter``), so the clean-name check alone
        # silently classifies every disambiguated pick as "not local",
        # which then trips the fail-fast guard below — even though
        # ``resolve_backend_for_model`` correctly returned a specific,
        # working backend via the ``@``-suffix branch. Accepting the
        # original ``model_name`` as proof-of-locality closes that gap.
        local_known = (
            clean_model in self._model_map
            or (model_name and model_name in self._model_map)
            or self._local_backend_has_loaded(clean_model)
        )
        peer_backend = await director.maybe_route_llm(
            model=clean_model,
            user_id=user_id,
            session_id=session_id,
            local_backend=backend,
            local_known=local_known,
        )
        if peer_backend is not None:
            return peer_backend, clean_model
        # Fabric is wired but the model isn't local AND no peer matched.
        # ``resolve_backend_for_model`` silently fell back to the default
        # backend in this case; dispatching against it produces a confusing
        # upstream error ("model not found" / connect-refused). Raise a
        # typed error with peer diagnostics so the route layer can surface
        # what we actually checked. Skip the empty-name path — that's the
        # role-resolver "use default first model" idiom and not a real
        # failure.
        if not local_known and clean_model and model_name:
            diag = director.peer_diagnostic_for_llm(clean_model)
            raise ModelUnavailableError(
                f"model {clean_model!r} is not served by any local backend "
                f"or connected fabric peer "
                f"(connected_peers={diag.get('connected_peers') or '[]'}, "
                f"offline_peers={diag.get('offline_peers') or '[]'})",
                model=clean_model,
                peer_diagnostic=diag,
            )
        return backend, clean_model

    def _init_backends(self, client: httpx.AsyncClient) -> None:
        if settings.ollama_base_url:
            self._backends["ollama"] = OllamaBackend(client, settings.ollama_base_url)

        engine_url = settings.engine_base_url or os.environ.get("AUGMENTUM_ENGINE_URL", "")
        if engine_url:
            from augmentum.models.engine import AugmentumEngineBackend
            self._backends["engine"] = AugmentumEngineBackend(client, engine_url)
            log.info("engine_backend_registered", base_url=engine_url)

        # Skip OpenAI backend if its URL points at the local engine (avoids
        # duplicate "OpenAI (builtin)" entry — engine already registered above)
        openai_is_engine = (
            engine_url
            and settings.openai_base_url
            and engine_url.rstrip("/") in settings.openai_base_url
        )
        if settings.openai_api_key and not openai_is_engine:
            self._backends["openai"] = OpenAIBackend(
                client, settings.openai_base_url, settings.openai_api_key,
                chat_client=self._chat_http_client,
            )
            log.info("openai_backend_registered", base_url=settings.openai_base_url)

        if settings.llamacpp_base_url:
            self._backends["llamacpp"] = LlamaCppBackend(
                client, settings.llamacpp_base_url, settings.llamacpp_api_key
            )
            log.info("llamacpp_backend_registered", base_url=settings.llamacpp_base_url)

        # Dedicated voice/intent-classifier sidecar. Auto-registers when
        # compose.classifier.yaml is enabled (it sets the env var). It's a
        # stock llama-server, so register it as a LlamaCppBackend (no manager
        # = pure HTTP client, same as the in-process sibling). This matters:
        # LlamaCppBackend forwards ``chat_template_kwargs`` to the server,
        # which OpenAIBackend drops. Without it, a Gemma-class judgment model
        # never receives ``enable_thinking: False`` → its chat template injects
        # <|think|> by default and burns the budget on a reasoning trace
        # (voice_router_parse_failed, thinking_chars>0). Only used for the
        # local sidecar; the cloud fallback path uses its own backend, so the
        # kwarg never reaches a provider that would 400 on it.
        classifier_url = (
            getattr(settings, "classifier_base_url", "")
            or os.environ.get("AUGMENTUM_CLASSIFIER_BASE_URL", "")
        )
        if classifier_url:
            self._backends["classifier"] = LlamaCppBackend(
                client, classifier_url, "not-needed",
            )
            log.info("classifier_sidecar_registered", base_url=classifier_url)

        # Optional vLLM + llama-swap fallback tier for architectures the bundled
        # llama-server can't load (safetensors + trust_remote_code). Auto-
        # registers when compose.vllm.yaml is enabled (it sets the env var).
        # llama-swap is a plain OpenAI-compatible front door that swaps vLLM
        # upstreams by model name, so register it as an OpenAIBackend (NOT
        # LlamaCppBackend — it's not a llama-server; it doesn't consume the
        # chat_template_kwargs path). Never auto-selected: it only serves models
        # the user explicitly downloaded as safetensors and picked. See spec
        # 2026-07-22-unsupported-arch-serving-vllm-safetensors-design.md
        vllm_url = (
            getattr(settings, "vllm_base_url", "")
            or os.environ.get("AUGMENTUM_VLLM_BASE_URL", "")
        )
        if vllm_url:
            self._backends["vllm"] = OpenAIBackend(
                client, vllm_url, "not-needed",
                chat_client=self._chat_http_client,
            )
            log.info("vllm_fallback_registered", base_url=vllm_url)

        # Claude (Anthropic)
        if settings.anthropic_api_key:
            self._backends["claude"] = create_backend_from_profile(
                None,
                provider_type="claude",
                api_key=settings.anthropic_api_key,
                http_client=client,
                chat_client=self._chat_http_client,
                base_url=settings.anthropic_base_url,
            )
            log.info("claude_backend_registered", base_url=settings.anthropic_base_url)

        # Gemini (Google)
        if settings.google_api_key:
            self._backends["gemini"] = create_backend_from_profile(
                None,
                provider_type="gemini",
                api_key=settings.google_api_key,
                http_client=client,
                chat_client=self._chat_http_client,
                vertex=settings.google_vertex,
                vertex_project=settings.google_vertex_project,
                vertex_region=settings.google_vertex_region,
            )
            log.info("gemini_backend_registered")

        # Profile-based backends
        _PROFILE_KEY_MAP = {
            "openrouter": "openrouter_api_key",
            "mistral": "mistral_api_key",
            "deepseek": "deepseek_api_key",
            "xai": "xai_api_key",
            "groq": "groq_api_key",
            "cohere": "cohere_api_key",
            "perplexity": "perplexity_api_key",
            "fireworks": "fireworks_api_key",
        }
        for profile_id, key_attr in _PROFILE_KEY_MAP.items():
            api_key = getattr(settings, key_attr, "")
            if api_key:
                profile = get_profile(profile_id)
                if profile:
                    self._backends[profile_id] = create_backend_from_profile(
                        profile, api_key=api_key, http_client=client,
                        chat_client=self._chat_http_client,
                    )
                    log.info("profile_backend_registered", profile=profile_id)

        # Install training-trace capture on every configured backend. Idempotent
        # and inert unless a capture_turn scope is active (see trace_context) —
        # the hot path is one ContextVar read.
        for _backend in self._backends.values():
            self._hook(_backend)

        log.info(
            "backends_initialized",
            configured=list(self._backends.keys()) or ["none"],
            default=self._default,
        )

    def _hook(self, backend: ModelBackend) -> ModelBackend:
        """Install backend-boundary training capture (idempotent). A failure here
        must never block backend registration, so it's swallowed to debug."""
        try:
            from augmentum.training.trace_context import install_capture_hook

            install_capture_hook(backend)
        except Exception:
            log.debug("capture_hook_install_failed", exc_info=True)
        try:
            # AFTER the capture hook, so the primer wrapper is outermost and
            # capture snapshots record what the model actually received.
            from augmentum.prompts.native_serve import install_primer_hook

            install_primer_hook(backend)
        except Exception:
            log.warning("primer_hook_install_failed", exc_info=True)
        return backend

    def get_backend(self, name: str | None = None) -> ModelBackend:
        """Get a backend by name, or the default.

        Returns ``None`` when *name* is given explicitly but not found
        (allows callers to handle gracefully), raises when the *default*
        is missing.
        """
        key = name or self._default
        backend = self._backends.get(key)
        if backend is None and name is not None:
            return None  # type: ignore[return-value]
        if backend is None:
            if not self._backends:
                raise ValueError(
                    "No model backends are connected. Add a provider in "
                    "Settings > Manage Providers, or set "
                    "AUGMENTUM_DEFAULT_BACKEND=engine to use the bundled engine."
                )
            raise ValueError(
                f"Backend '{key}' not available. Available: {list(self._backends.keys())}"
            )
        return backend

    @property
    def backends(self) -> dict[str, ModelBackend]:
        """Expose the full backend map (used by ModelManager)."""
        return self._backends

    @property
    def default_backend(self) -> ModelBackend:
        return self.get_backend()

    @property
    def available_backends(self) -> list[str]:
        return list(self._backends.keys())

    # Discovery results stored for the frontend to display
    _discovered: list[dict] = []

    async def load_dismissed_discoveries(self, settings_store) -> None:
        """Load dismissed discovery URLs from persistent settings."""
        import json
        raw = await settings_store.get("dismissed_discovery_urls")
        if raw:
            try:
                self._dismissed_urls = set(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                self._dismissed_urls = set()

    async def save_dismissed_discoveries(self, settings_store) -> None:
        """Persist dismissed discovery URLs to settings store."""
        import json
        await settings_store.set("dismissed_discovery_urls", json.dumps(list(self._dismissed_urls)))

    def dismiss_discovery_url(self, url: str) -> None:
        """Mark a URL as dismissed (call save_dismissed_discoveries after)."""
        self._dismissed_urls.add(url.rstrip("/"))

    def undismiss_discovery_url(self, url: str) -> None:
        """Remove a URL from dismissed list (e.g., if user manually adds it)."""
        self._dismissed_urls.discard(url.rstrip("/"))

    async def populate_provider_urls(self, provider_store: ProviderStore) -> None:
        """Cache URLs from the providers DB table for discovery filtering."""
        try:
            providers = await provider_store.list_providers()
            self._provider_urls = {p.base_url.rstrip("/") for p in providers if p.base_url}
        except Exception as exc:
            log.warning("populate_provider_urls_failed", error=str(exc))

    async def discover_local_backends(self) -> None:
        """Auto-discover LLM servers running on the local network.

        Scans well-known ports, fingerprints each service to determine what
        it actually is (Ollama, LM Studio, llama.cpp, generic OpenAI), pulls
        model lists, and registers reachable backends.

        Uses host.docker.internal (Docker Desktop) or configurable host IP.
        """
        import httpx

        host = os.environ.get("AUGMENTUM_HOST_IP", "host.docker.internal")

        # Ports to scan — covers all major local LLM platforms
        scan_ports = [11434, 1234, 8080, 5000, 3000, 8000, 8888, 9090]

        self._discovered = []

        async with httpx.AsyncClient(timeout=2.0) as client:
            for port in scan_ports:
                url = f"http://{host}:{port}"

                # Skip if user previously dismissed this URL
                if url.rstrip("/") in self._dismissed_urls:
                    continue

                # Skip if already in the providers DB (user-configured)
                if url.rstrip("/") in self._provider_urls:
                    continue

                # Skip if already registered as an active backend
                already = any(
                    url in str(getattr(b, '_base_url', '')) or
                    url in str(getattr(b, 'base_url', ''))
                    for b in self._backends.values()
                )
                if already:
                    continue

                info = await self._identify_service(client, url)
                if not info:
                    continue

                service_type = info["type"]
                service_name = info["name"]
                models = info.get("models", [])

                # Register with appropriate backend class
                key = None
                backend = None

                if service_type == "ollama" and "ollama" not in self._backends:
                    backend = OllamaBackend(self._http_client, url)
                    key = "ollama"
                elif service_type == "llamacpp" and "llamacpp" not in self._backends:
                    backend = LlamaCppBackend(self._http_client, url, "")
                    key = "llamacpp"
                elif service_type == "openai":
                    backend = OpenAIBackend(self._http_client, url, "not-needed")
                    key = f"local-{port}"
                else:
                    # Already have this type registered, add with port suffix
                    if service_type == "ollama":
                        backend = OllamaBackend(self._http_client, url)
                    else:
                        backend = OpenAIBackend(self._http_client, url, "not-needed")
                    key = f"{service_type}-{port}"

                if key and backend:
                    self._backends[key] = backend
                    self._hook(backend)
                    discovery = {
                        "key": key,
                        "name": service_name,
                        "type": service_type,
                        "url": url,
                        "port": port,
                        "models": models[:10],  # cap for API response size
                        "model_count": len(models),
                    }
                    self._discovered.append(discovery)
                    log.info("discovered_local_backend",
                             name=service_name, key=key, url=url,
                             models=len(models))

        if self._discovered:
            log.info("local_discovery_complete",
                     found=[(d["key"], d["name"], d["model_count"])
                            for d in self._discovered])
            if self._default not in self._backends and self._discovered:
                self._default = self._discovered[0]["key"]
                log.info("default_backend_auto_discovered",
                         selected=self._discovered[0]["name"])
        else:
            log.info("local_discovery_complete", found="none")

    @staticmethod
    async def _identify_service(client, url: str) -> dict | None:
        """Fingerprint a service to determine what it is and list its models.

        Checks multiple endpoints to distinguish between:
        - Ollama (has /api/tags with 'models' array)
        - LM Studio (has /v1/models, /lmstudio/models endpoint)
        - llama.cpp (has /v1/models, /health, slots-based)
        - Generic OpenAI-compatible (has /v1/models)
        """

        # Try Ollama first (/api/tags is unique to Ollama)
        try:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                if "models" in data:
                    models = [m.get("name", "?") for m in data["models"]]
                    return {"type": "ollama", "name": "Ollama", "models": models}
        except Exception as exc:
            log.debug("backend_probe_ollama_tags_failed", url=url, error=str(exc))

        # Try /v1/models (OpenAI-compatible — LM Studio, llama.cpp, vLLM, etc.)
        try:
            resp = await client.get(f"{url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models_data = data.get("data", [])
                models = [m.get("id", "?") for m in models_data]

                # Fingerprint: LM Studio has /lmstudio/models
                try:
                    lms = await client.get(f"{url}/lmstudio/models")
                    if lms.status_code == 200:
                        return {"type": "openai", "name": "LM Studio", "models": models}
                except Exception as exc:
                    log.debug("backend_probe_lmstudio_failed", url=url, error=str(exc))

                # Fingerprint: llama.cpp has /health with slots info
                try:
                    health = await client.get(f"{url}/health")
                    if health.status_code == 200:
                        hdata = health.json()
                        if "slots_idle" in hdata or "slots_processing" in hdata:
                            return {"type": "llamacpp", "name": "llama.cpp", "models": models}
                except Exception as exc:
                    log.debug("backend_probe_llamacpp_health_failed", url=url, error=str(exc))

                # Fingerprint: Augmentum Engine has /health with engine-specific fields
                try:
                    eng = await client.get(f"{url}/health")
                    if eng.status_code == 200:
                        edata = eng.json()
                        if "loaded_model" in edata or "engine" in str(edata).lower():
                            return {"type": "openai", "name": "Augmentum Engine", "models": models}
                except Exception as exc:
                    log.debug("backend_probe_engine_health_failed", url=url, error=str(exc))

                # Fingerprint: vLLM has specific model format
                if models and any("/" in m.get("id", "") for m in models_data):
                    return {"type": "openai", "name": "vLLM", "models": models}

                # Generic OpenAI-compatible
                name = "LLM Server"
                if models:
                    # Try to guess from model names
                    joined = " ".join(models).lower()
                    if "gpt" in joined:
                        name = "OpenAI"
                    elif "claude" in joined:
                        name = "Anthropic"
                return {"type": "openai", "name": name, "models": models}
        except Exception as exc:
            log.debug("backend_probe_v1_models_failed", url=url, error=str(exc))

        # Try bare /health (some servers only have this)
        try:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                return {"type": "openai", "name": "LLM Server", "models": []}
        except Exception as exc:
            log.debug("backend_probe_bare_health_failed", url=url, error=str(exc))

        return None

    async def probe_backends(self) -> None:
        """Probe all registered backends and drop unreachable ones.

        Auto-selects the default backend from whatever is actually
        reachable, preferring the configured default if it responds.
        """
        if not self._backends:
            log.info("no_backends_configured", hint="Add a provider in Settings or set AUGMENTUM_DEFAULT_BACKEND=engine")
            return

        reachable: list[str] = []
        unreachable: list[str] = []

        for key, backend in list(self._backends.items()):
            try:
                await backend.list_models()
                reachable.append(key)
            except Exception:
                unreachable.append(key)

        # Remove unreachable backends
        for key in unreachable:
            del self._backends[key]
            log.info("backend_unreachable_removed", backend=key)

        # Auto-select default if the configured one isn't available
        if self._default not in self._backends and reachable:
            old_default = self._default
            self._default = reachable[0]
            log.info(
                "default_backend_auto_selected",
                configured=old_default,
                selected=self._default,
                reason=f"{old_default} not reachable",
            )

        if not self._backends:
            log.warning("no_backends_reachable")
        else:
            log.info(
                "backends_probed",
                reachable=reachable,
                unreachable=unreachable,
                default=self._default,
            )

    # --- Runtime management ---

    def set_provider_meta(self, key: str, owner_user_id: str, shared: bool) -> None:
        """Record ownership + sharing for a DB-provider backend key.

        Called when a runtime provider is loaded, created, or updated.
        Builtin/env backends never get an entry (they stay globally
        visible via the ``None`` branch in :meth:`provider_visible_to`).
        """
        self._provider_meta[key] = (owner_user_id or "", bool(shared))

    def clear_provider_meta(self, key: str) -> None:
        """Drop a provider's visibility metadata (on delete/unregister)."""
        self._provider_meta.pop(key, None)

    def provider_visible_to(self, key: str, user_id: str = "") -> bool:
        """Whether backend ``key`` is visible/usable by ``user_id``.

        Builtin/env backends (absent from ``_provider_meta``) are global
        infrastructure → always visible. A DB provider is visible when it
        is shared or owned by this user — nothing else.

        An UNSHARED provider with an empty owner is hidden from everyone.
        The first version of this predicate treated empty-owner as
        "global → always visible", which made Unshare a silent no-op on
        every pre-305 admin-created provider (owner was backfilled to '');
        that's the leak this guards against. The share route stamps the
        acting admin as owner on unshare, so the ownerless-unshared state
        shouldn't persist — treating it as hidden is the fail-safe.
        """
        meta = self._provider_meta.get(key)
        if meta is None:
            return True
        owner, shared = meta
        if shared:
            return True
        return bool(user_id) and bool(owner) and owner == user_id

    def _backend_key_for(self, backend: ModelBackend) -> str:
        """Reverse-lookup a backend's registry key by identity ("" if none)."""
        for k, b in self._backends.items():
            if b is backend:
                return k
        return ""

    def register_backend(self, key: str, backend: ModelBackend) -> None:
        """Register a backend at runtime."""
        self._backends[key] = backend
        self._hook(backend)
        # Auto-select as default if the current default isn't available
        if self._default not in self._backends:
            old = self._default
            self._default = key
            log.info("default_backend_auto_selected", configured=old, selected=key, reason="previous default unavailable")
        # Restore this backend's last-known catalog the instant it registers,
        # so its models are routable immediately even if an earlier
        # ``refresh_model_map`` ran during startup (before this backend existed)
        # and dropped them — the post-restart "cloud model not in map" race.
        # Only fills gaps: a fresh probe result already in the map wins. A later
        # successful probe re-verifies; until then these serve as UNVERIFIED.
        if self._boot_catalog:
            restored = {
                m: bk
                for m, bk in self._boot_catalog.items()
                if bk == key and m not in self._model_map
            }
            if restored:
                self._model_map.update(restored)
                self._model_unverified |= set(restored)
                log.info("backend_catalog_restored", backend=key, models=len(restored))
        log.info("backend_registered", key=key)

    def unregister_backend(self, key: str) -> None:
        """Remove a backend. Refuses to remove the default."""
        if key == self._default:
            raise ValueError(f"Cannot unregister the default backend '{key}'")
        removed = self._backends.pop(key, None)
        if removed:
            # Clean stale entries from BOTH the live map and the immutable boot
            # snapshot, so a deliberately-removed backend's models stop being
            # carried forward by refresh_model_map (distinguishes a real removal
            # from a not-yet-registered backend during the startup window).
            self._model_map = {
                m: k for m, k in self._model_map.items() if k != key
            }
            self._boot_catalog = {
                m: k for m, k in self._boot_catalog.items() if k != key
            }
            self._provider_meta.pop(key, None)
            log.info("backend_unregistered", key=key)

    # --- Model map ---

    def _is_cloud_backend(self, key: str) -> bool:
        """True iff ``key``'s backend talks to a remote (non-local) endpoint.

        Used to give cloud catalogs a longer probe deadline (fix #2) and to
        keep the degraded fast-re-probe local-only (fix #3). Locality is read
        from the backend's ``_base_url`` via the same loopback/RFC-1918/docker
        rule the request path uses; a backend with no base_url (the in-process
        local engine) is local by definition.
        """
        backend = self._backends.get(key)
        base_url = getattr(backend, "_base_url", "") if backend else ""
        if not base_url:
            return False
        from augmentum.models.openai_compat import is_local_engine_url

        return not is_local_engine_url(base_url)

    def probe_deadline_for(self, key: str) -> float:
        """Per-backend ``list_models`` deadline (fix #2). Cloud catalogs are
        large and slow; locals are instant. Shared by ``refresh_model_map``
        and ``/api/tags`` so both wait long enough for deepseek/openrouter on
        a cold first probe instead of dropping them at a flat 6s."""
        return (
            _PROBE_DEADLINE_CLOUD_S
            if self._is_cloud_backend(key)
            else _PROBE_DEADLINE_LOCAL_S
        )

    def _model_map_cache_path(self) -> str:
        # str() guards against a non-str data_dir (e.g. a mocked settings in
        # tests) reaching os.path.join and raising at construction time.
        return os.path.join(str(settings.data_dir), _MODEL_MAP_CACHE_FILE)

    def _load_persisted_model_map(self) -> None:
        """Seed ``_model_map`` from the on-disk cache at startup, marking every
        entry UNVERIFIED so it's resolvable immediately but re-confirmed by the
        first probe. ``_model_map_ts`` stays 0 so the first request still
        triggers a fresh probe. Best-effort: a missing/corrupt cache is a
        clean cold start, never an error.
        """
        try:
            path = self._model_map_cache_path()
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:
            log.warning("model_map_cache_load_failed", error=repr(exc))
            return
        loaded = payload.get("map") if isinstance(payload, dict) else None
        if not isinstance(loaded, dict) or not loaded:
            return
        # Keep only sane str->str entries.
        clean = {
            str(name): str(key)
            for name, key in loaded.items()
            if isinstance(name, str) and isinstance(key, str) and name and key
        }
        if not clean:
            return
        self._model_map = clean
        self._model_unverified = set(clean.keys())
        self._persisted_snapshot = dict(clean)
        # Stable copy used by register_backend to bridge the startup window
        # where an early refresh races a cloud backend's registration.
        self._boot_catalog = dict(clean)
        # ts left at 0 → first resolve probes for real; until then we serve
        # this last-known-good catalog so nothing is "missing" after restart.
        log.info("model_map_cache_loaded", models=len(clean), path=path)

    def _persist_model_map(self, new_map: dict[str, str]) -> None:
        """Atomically write the current map to disk (off the event loop). Only
        called when the catalog actually changed (see ``refresh_model_map``).
        """
        tmp = ""
        try:
            path = self._model_map_cache_path()
            tmp = f"{path}.tmp"
            data = json.dumps({"map": new_map, "saved_at": time.time()})
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp, path)
            self._persisted_snapshot = dict(new_map)
        except Exception as exc:
            log.warning("model_map_cache_write_failed", error=repr(exc))
            if tmp:
                with contextlib.suppress(Exception):
                    os.remove(tmp)

    async def refresh_model_map(self, *, force: bool = False) -> dict[str, str]:
        """Probe all backends and rebuild the model-name -> backend-key map.

        Uses a TTL cache to avoid hammering backends on every request.
        """
        now = time.monotonic()
        if not force and self._model_map and (now - self._model_map_ts) < _MODEL_MAP_TTL:
            return self._model_map

        new_map: dict[str, str] = {}
        # Track raw names to detect collisions
        name_counts: dict[str, list[str]] = {}  # model_name -> [backend_keys]

        # Probe backends in parallel with a per-backend deadline. Same
        # rationale as ollama_routes.ollama_tags: one slow cloud provider
        # (deepseek/openrouter with a degraded route) used to block the
        # whole probe for the sum of their httpx connect timeouts. With
        # asyncio.gather + per-backend wait_for, the worst case is the
        # slowest individual backend, not their sum, and one failure
        # doesn't gate the rest. The deadline is per-backend (fix #2): cloud
        # catalogs are large and slow, locals are instant.
        async def _probe(key: str, backend) -> tuple[str, list | None]:
            deadline = self.probe_deadline_for(key)
            try:
                models = await asyncio.wait_for(
                    backend.list_models(), timeout=deadline,
                )
                # Recovery: this backend had been failing — surface the
                # return to health at warning so dashboards see the edge.
                if self._probe_failure_state.pop(key, None):
                    log.warning("model_map_probe_recovered", backend=key)
                return key, list(models)
            except TimeoutError:
                err_key = "timeout"
                if self._probe_failure_state.get(key) != err_key:
                    log.warning(
                        "model_map_probe_timeout", backend=key,
                        timeout_s=deadline,
                    )
                    self._probe_failure_state[key] = err_key
                else:
                    log.debug(
                        "model_map_probe_timeout", backend=key,
                        timeout_s=deadline, repeat=True,
                    )
                return key, None
            except Exception as exc:
                # str(httpx.ConnectTimeout(TimeoutError())) is empty so we'd
                # log `error=` with nothing useful; repr surfaces the type.
                # Coalesce: warn once per (backend, error-type) pair, then
                # downgrade to debug until either the error type CHANGES or
                # the backend recovers (handled in the success branch above).
                # Previously every probe cycle warned again — at ~10
                # backends × ~24h × N restarts that was the top noise event.
                err_repr = repr(exc)
                err_key = type(exc).__name__
                if self._probe_failure_state.get(key) != err_key:
                    log.warning(
                        "model_map_probe_failed", backend=key, error=err_repr,
                    )
                    self._probe_failure_state[key] = err_key
                else:
                    log.debug(
                        "model_map_probe_failed", backend=key,
                        error=err_repr, repeat=True,
                    )
                return key, None

        probe_results = await asyncio.gather(
            *(
                _probe(k, b)
                for k, b in self._backends.items()
                if k not in self._map_excluded_backends
            ),
            return_exceptions=False,
        )
        # Distinguish a FAILED probe (``models is None`` — timeout/error) from
        # a backend that legitimately serves nothing right now (empty list).
        # Only the former triggers last-known-good carry-forward below; the
        # latter correctly drops that backend's models.
        failed_keys: set[str] = set()
        succeeded_keys: set[str] = set()
        for key, models in probe_results:
            if models is None:
                failed_keys.add(key)
                continue
            succeeded_keys.add(key)
            for m in models:
                name_counts.setdefault(m.name, []).append(key)
        any_probe_failed = bool(failed_keys)

        # Three-tier freshness instead of binary drop/keep. Previously the
        # whole map was rebuilt from scratch every cycle, so a single
        # failed/timed-out probe (cold cloud TLS after restart, a /models call
        # exceeding the 6s deadline, a brief network blip) dropped EVERY model
        # that backend serves — and the full-TTL cache below then hid them for
        # 30s. A user whose ``primary_chat_model`` lives on that provider got
        # ``model_not_in_map_using_default`` → (with fabric on) a typed
        # ModelUnavailableError. Instead:
        #   VERIFIED   — backend responded THIS round (in ``succeeded_keys``):
        #                fresh catalog, so a model a provider genuinely dropped
        #                still disappears.
        #   UNVERIFIED — backend was verified before but didn't respond this
        #                round: re-seed its last-known-good catalog from the
        #                prior map and FLAG it stale (still resolvable so a
        #                transient outage doesn't break dispatch, but callers
        #                can tell it's unconfirmed).
        #   (absent)   — backend that has never produced a catalog has nothing
        #                to carry, so it simply doesn't appear.
        # ``stale_backends`` collects the registered-but-unresponsive backends
        # we re-seed; their surviving map keys become the UNVERIFIED tier.
        stale_backends: set[str] = set()
        prev_map = self._model_map
        # Carry forward the last-known catalog of any backend that did NOT
        # return a fresh catalog THIS round — whether its probe FAILED or it
        # simply wasn't probed because it isn't registered yet. Cloud/runtime
        # backends register SECONDS after the on-disk cache loads at startup
        # (see register_backend / _boot_catalog); a refresh that runs in that
        # window must not drop their models — and, critically, must not persist
        # that degraded map to disk, which would poison the cache so the models
        # stay unroutable across EVERY future restart (the reported "cloud model
        # not in map after restart" class). Source from BOTH the live map and
        # the immutable boot snapshot so a model survives even if an earlier
        # racy round already dropped it from the live map.
        #
        # Genuine removals still happen: a SUCCESSFUL probe that no longer lists
        # a model replaces that backend's entries (succeeded_keys wins below),
        # and ``unregister_backend`` purges a deliberately-removed backend from
        # BOTH the live map and the boot snapshot so it stops being carried.
        carry_source = {**self._boot_catalog, **prev_map}
        for prev_name, owner_key in carry_source.items():
            if owner_key in succeeded_keys:
                continue  # responded this round — fresh data wins
            if owner_key in self._map_excluded_backends:
                continue
            base_name = prev_name.split("@", 1)[0]
            owners = name_counts.setdefault(base_name, [])
            if owner_key not in owners:
                owners.append(owner_key)
            stale_backends.add(owner_key)

        # Build map — disambiguate collisions with @backend suffix
        for model_name, backend_keys in name_counts.items():
            if len(backend_keys) == 1:
                new_map[model_name] = backend_keys[0]
            else:
                for bk in backend_keys:
                    new_map[f"{model_name}@{bk}"] = bk

        # Inject routing pins last so they override any catalog collision:
        # a model resident in a dedicated slot maps uniquely to that slot's
        # backend and appears exactly once in the picker.
        for pinned_model, pinned_key in self._model_pins.items():
            new_map[pinned_model] = pinned_key

        # Mark the UNVERIFIED tier: any surviving entry owned by a stale
        # (carried-forward) backend. Pins are authoritative — set at model-load
        # time, not probe-derived — so they're always verified.
        unverified = {
            mname for mname, bkey in new_map.items() if bkey in stale_backends
        }
        unverified.difference_update(self._model_pins.keys())

        self._model_map = new_map
        prev_unverified = self._model_unverified
        self._model_unverified = unverified
        # Forget warn-dedup state for models that recovered to verified, so a
        # future degradation traces afresh instead of staying silent.
        self._unverified_served_logged.intersection_update(unverified)
        if any_probe_failed:
            sig = frozenset(failed_keys)
            chronic = sig == self._degraded_round_logged
            local_failed = any(not self._is_cloud_backend(k) for k in failed_keys)
            # Fast 5s re-probe on a LOCAL drop (chat-model recovery) or a NEW/
            # changed failure — cold start, or a freshly-flaky backend. But if
            # the SAME backends keep failing round after round (chronic), back
            # off to the full TTL so a permanently-slow cloud /models isn't
            # re-hit every 5s. This unifies fix #1 (log dedup) with fix #3 (no
            # re-probe storm) WITHOUT starving a cold-start cloud catalog: the
            # first failure always gets one fast retry, only the repeat settles.
            fast_retry = local_failed or not chronic
            self._model_map_ts = (
                now - (_MODEL_MAP_TTL - _MODEL_MAP_DEGRADED_RETRY_S)
                if fast_retry else now
            )
            # Fix #1: only WARN when the unhealthy set changed since the last
            # warned round; otherwise debug. Stops a chronically slow backend
            # from machine-gunning the log on every re-probe.
            level = log.debug if chronic else log.warning
            level(
                "model_map_degraded_round",
                healthy_backends=sorted(succeeded_keys),
                failed_backends=sorted(failed_keys),
                unverified_models=len(unverified),
                retry_in_s=_MODEL_MAP_DEGRADED_RETRY_S if fast_retry else _MODEL_MAP_TTL,
                mapped_models=len(new_map),
                repeat=chronic,
            )
            self._degraded_round_logged = sig
        else:
            self._model_map_ts = now
            # Recovered to a fully-healthy round — surface the edge once and
            # reset so a future degradation warns afresh.
            if self._degraded_round_logged is not None:
                log.info("model_map_fully_recovered")
                self._degraded_round_logged = None

        # Persist + notify when the catalog actually changes — cold-start
        # fill-in, an unverified→verified upgrade, or a real add/remove. The
        # disk cache lets the next restart serve this list immediately; the
        # ``models.changed`` event makes connected UIs refetch live instead of
        # needing a manual double-refresh.
        map_changed = new_map != self._persisted_snapshot
        unverified_changed = unverified != prev_unverified
        if map_changed:
            await asyncio.to_thread(self._persist_model_map, dict(new_map))
        if map_changed or unverified_changed:
            self._publish_models_changed(total=len(new_map), unverified=len(unverified))
        return new_map

    def _publish_models_changed(self, *, total: int, unverified: int) -> None:
        """Fire-and-forget UI notification that the model catalog changed, so
        connected clients refetch ``/api/tags`` (no manual double-refresh after
        a cold start). Best-effort; never breaks the probe."""
        try:
            from augmentum.proxy import system_events

            system_events.publish(
                "models.changed",
                {"total": total, "unverified": unverified},
            )
        except Exception:
            log.debug("models_changed_publish_failed", exc_info=True)

    def model_freshness(self, model_name: str) -> str:
        """Three-tier freshness of ``model_name`` in the current map.

        ``"verified"``   — the owning backend responded on the last probe.
        ``"unverified"`` — last-known-good carried forward; the backend is
                           currently unresponsive (still resolvable, but the
                           catalog is unconfirmed).
        ``"unknown"``    — not in the map at all (never advertised, or its
                           backend was removed).

        Lets the picker, diagnostics, and dispatch sites distinguish a live
        model from a stale one instead of treating presence-in-map as health.
        """
        if not model_name or model_name not in self._model_map:
            return "unknown"
        return "unverified" if model_name in self._model_unverified else "verified"

    def is_model_verified(self, model_name: str) -> bool:
        """True only for the VERIFIED tier (backend confirmed it this probe)."""
        return self.model_freshness(model_name) == "verified"

    def _note_unverified_resolution(self, model_name: str, backend_key: str) -> None:
        """Trace when a request resolves to an UNVERIFIED catalog entry.

        The owning backend was unresponsive on the last probe, so we're
        serving its last-known-good catalog — a dispatch failure here is
        expected-and-diagnosable rather than mysterious. Coalesced: warns once
        per model until it returns to the verified tier (reset in
        ``refresh_model_map``) so a per-turn role resolve doesn't spam the log.
        """
        if model_name not in self._model_unverified:
            return
        if model_name in self._unverified_served_logged:
            return
        self._unverified_served_logged.add(model_name)
        log.warning(
            "model_resolved_unverified",
            model=model_name,
            backend=backend_key,
            hint="serving last-known-good catalog; owning backend was "
                 "unresponsive on the last probe — dispatch may fail until it "
                 "recovers",
        )

    def invalidate_model_map(self):
        """Force refresh of model map on next query.

        Also clears any per-backend list-models cache that exposes an
        ``invalidate_models_cache`` hook (e.g., LlamaCppBackend's 15s TTL
        on the managed-server scan). Without this, a freshly-downloaded
        GGUF stays invisible to /api/tags consumers until the per-backend
        TTL expires, even though the registry's own model_map gets
        rebuilt on the next request.
        """
        self._model_map_ts = 0
        for backend in self._backends.values():
            hook = getattr(backend, "invalidate_models_cache", None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    log.warning(
                        "invalidate_models_cache_failed",
                        backend=type(backend).__name__,
                        exc_info=True,
                    )

    def set_lb_registry(self, lb_registry) -> None:
        """Attach the load balancer registry for virtual model resolution."""
        self._lb_registry = lb_registry

    def exclude_backend_from_map(self, backend_key: str) -> None:
        """Keep ``backend_key`` out of EVERY catalog listing.

        Used for dedicated resident slots (the secondary local engine):
        the slot shares its model_dirs with the primary engine, so if it
        advertised its full GGUF catalog every model name would collide
        into ``name@engine`` / ``name@engine_secondary`` variants and the
        picker UX would break. The slot is still a registered backend
        (reachable via its pin and by ``_local_backend_has_loaded``) — it
        just doesn't appear in any catalog enumeration.

        "Listing" here means ALL the per-backend probe+dedup paths, not
        only ``refresh_model_map``: ``/api/tags`` (ollama_routes),
        ``/v1/models`` (openai_routes), and ``ModelManager.list_all_models``
        each iterate ``backends`` and disambiguate duplicates with an
        ``@key`` / `` (key)`` suffix. They all consult
        :meth:`is_listing_excluded` so an excluded backend never inflates
        a collision or the backend count. Idempotent.
        """
        self._map_excluded_backends.add(backend_key)

    def is_listing_excluded(self, backend_key: str) -> bool:
        """True when ``backend_key`` must be skipped in catalog listings.

        Consulted by every model-enumeration path (model map, /api/tags,
        /v1/models, list_all_models) so a routing-only backend — one
        reachable solely via an explicit pin — never pollutes the picker.
        """
        return backend_key in self._map_excluded_backends

    def pin_model(self, model_name: str, backend_key: str) -> None:
        """Route ``model_name`` to ``backend_key``, overriding the catalog map.

        Call when a model is loaded into a dedicated resident slot so chat
        requests for that model hit the slot's live process instead of
        swapping the primary engine. Invalidates the map so the injected
        pin takes effect on the next resolve. No-op on empty inputs.
        """
        model_name = (model_name or "").strip()
        backend_key = (backend_key or "").strip()
        if not model_name or not backend_key:
            return
        self._model_pins[model_name] = backend_key
        self._model_map[model_name] = backend_key  # immediate visibility
        log.info("model_pinned", model=model_name, backend=backend_key)

    def unpin_model(self, model_name: str = "") -> None:
        """Drop a routing pin (or all pins when ``model_name`` is empty).

        After unpinning, the model falls back to normal catalog routing —
        i.e. swap-on-demand against the primary engine. Invalidates the
        map so the change is visible on the next resolve.
        """
        if model_name:
            self._model_pins.pop(model_name.strip(), None)
        else:
            self._model_pins.clear()
        self.invalidate_model_map()

    def pinned_backend_for(self, model_name: str) -> str:
        """Return the pinned backend key for ``model_name``, or ''."""
        return self._model_pins.get((model_name or "").strip(), "")

    async def resolve_backend_for_model(self, model_name: str) -> tuple[ModelBackend, str]:
        """Look up a model in the model map and return (backend, clean_model_name).

        If model has an ``@backend`` suffix, parse it out.
        Supports ``lb/`` prefixed virtual models via load balancer registry.
        When ``model_name`` is empty, prefers ``settings.primary_chat_model``
        before falling back to the first model on the default backend.
        """
        # Mode prefixes are consumed by the classifier on classified paths;
        # strip here too so direct callers (role resolution, internal tools,
        # anything bypassing resolve_backend_with_fabric) match the same
        # semantics. See _strip_mode_prefix.
        model_name = _strip_mode_prefix(model_name)

        # Check for load balancer prefix
        from augmentum.models.load_balancer import LB_PREFIX

        if model_name.startswith(LB_PREFIX):
            lb_registry = getattr(self, "_lb_registry", None)
            if lb_registry:
                balancer_name = model_name[len(LB_PREFIX):]
                lb = lb_registry.get_by_name(balancer_name)
                if lb and lb.members:
                    # Return the fallback-aware FACADE rather than one member's
                    # backend, so every call site (all chat modes, companion
                    # tasks, aux routes) gets member selection + fallback for
                    # free. The facade owns strategy selection + fallback order
                    # per call — do NOT ``lb.select()`` here, it would
                    # double-advance round-robin. This supersedes the old
                    # set_balancer_context/get_balancer_context path (the latter
                    # had zero consumers, so fallback never fired). See
                    # models/balancer_backend.py and
                    # docs/load-balancer-first-class-fallback.md.
                    from augmentum.models.balancer_backend import BalancerBackend
                    log.info(
                        "balancer_resolved",
                        balancer=lb.config.name,
                        strategy=lb.config.strategy,
                        members=len(lb.members),
                        fallback_enabled=lb.config.fallback_enabled,
                    )
                    # Nominal model id for the caller's display/token-count; the
                    # facade overrides request.model per attempt internally.
                    return BalancerBackend(lb, self), lb.members[0].model_name

        # Check for explicit @backend suffix
        if "@" in model_name:
            parts = model_name.rsplit("@", 1)
            clean_name = parts[0]
            backend_key = parts[1]
            backend = self._backends.get(backend_key)
            if backend:
                return backend, clean_name

        # Explicit routing pin wins over the catalog map — and is checked
        # BEFORE refresh_model_map so a model resident in a dedicated slot
        # resolves instantly, never blocked on a 30s partial-map rebuild.
        if model_name:
            pinned_key = self._model_pins.get(model_name)
            if pinned_key:
                backend = self._backends.get(pinned_key)
                if backend:
                    return backend, model_name

        # Check model map (refresh if needed)
        await self.refresh_model_map()
        backend_key = self._model_map.get(model_name)
        if backend_key:
            backend = self._backends.get(backend_key)
            if backend:
                self._note_unverified_resolution(model_name, backend_key)
                # Map keys for multi-backend models carry an "@<backend>"
                # disambiguation suffix; the backend must get the bare id.
                return backend, _strip_backend_suffix(model_name, backend_key)

        # Normalized-id fallback. A model saved into a role/chat setting in
        # its canonical HF form ("unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL")
        # never exact-matches the SAME GGUF served by a local llama-server,
        # which advertises the FILENAME STEM that load picks up
        # ("unsloth_gemma-4-E2B-it-qat-GGUF_UD-Q4_K_XL", see
        # llama_server_manager.model_id = Path(model_path).stem). The two
        # differ only by '/'+':' vs '_', so the strict lookup above misses
        # and the request falls through to the default backend / fabric and
        # fails — even though the model IS loaded (this is the classifier
        # sidecar drop bug). Collapse both forms with the canonical
        # alnum-lowercase normalizer and retry. Require an UNAMBIGUOUS hit
        # (exactly one served entry collapses to the same key) so two
        # genuinely different models can never be silently cross-routed;
        # ambiguous cases fall through to the existing behaviour untouched.
        if model_name:
            from augmentum.models.openai_compat import _normalized_model_name
            want = _normalized_model_name(model_name)
            if want:
                norm_matches = [
                    (mname, bkey)
                    for mname, bkey in self._model_map.items()
                    if _normalized_model_name(mname.split("@", 1)[0]) == want
                ]
                if len(norm_matches) == 1:
                    actual_name, backend_key = norm_matches[0]
                    backend = self._backends.get(backend_key)
                    if backend:
                        clean = _strip_backend_suffix(actual_name, backend_key)
                        log.info(
                            "model_resolved_via_normalized_name",
                            requested=model_name,
                            resolved=clean,
                            backend=backend_key,
                        )
                        self._note_unverified_resolution(actual_name, backend_key)
                        return backend, clean

        # Fall back to the user's primary chat model, then the default backend.
        default = self.default_backend
        if not model_name:
            primary_model = (getattr(settings, "primary_chat_model", "") or "").strip()
            if primary_model:
                # Explicit backend suffixes remain valid even before the next
                # model-map refresh.
                if "@" in primary_model:
                    clean_name, backend_key = primary_model.rsplit("@", 1)
                    backend = self._backends.get(backend_key)
                    if backend:
                        log.info(
                            "model_empty_resolved_from_primary",
                            requested=primary_model,
                            resolved=clean_name,
                            backend=backend_key,
                        )
                        return backend, clean_name

                # Load balancer aliases resolve to one of their member models.
                if primary_model.startswith(LB_PREFIX):
                    lb_registry = getattr(self, "_lb_registry", None)
                    balancer_name = primary_model[len(LB_PREFIX):]
                    if lb_registry and lb_registry.get_by_name(balancer_name):
                        backend, clean_name = await self.resolve_backend_for_model(primary_model)
                        log.info(
                            "model_empty_resolved_from_primary",
                            requested=primary_model,
                            resolved=clean_name,
                        )
                        return backend, clean_name

                primary_backend_key = self._model_map.get(primary_model)
                if primary_backend_key:
                    backend = self._backends.get(primary_backend_key)
                    if backend:
                        log.info(
                            "model_empty_resolved_from_primary",
                            requested=primary_model,
                            resolved=primary_model,
                            backend=primary_backend_key,
                        )
                        self._note_unverified_resolution(primary_model, primary_backend_key)
                        return backend, primary_model

                log.warning(
                    "primary_chat_model_unavailable",
                    requested=primary_model,
                    default_backend=self._default,
                )

            # When neither the request nor primary model is available, pick
            # the first model hosted by the default backend as a last resort.
            if default:
                for mapped_model, mapped_key in self._model_map.items():
                    if mapped_key == self._default:
                        log.info("model_empty_resolved_from_default", resolved=mapped_model, backend=self._default)
                        return default, mapped_model
        log.warning("model_not_in_map_using_default", requested=model_name, default_backend=self._default)
        return default, model_name

    async def model_is_resolvable(self, model_name: str) -> bool:
        """True when ``model_name`` resolves to a real backend WITHOUT the
        default-backend fallback at the bottom of ``resolve_backend_for_model``.

        Callers that hold a user-configured model reference (narrative memory
        model, role overrides, …) use this to detect a stale reference and
        surface the choice to the user instead of silently riding whatever the
        default backend has loaded (never auto-select).
        """
        name = _strip_mode_prefix((model_name or "").strip())
        if not name:
            return False

        from augmentum.models.load_balancer import LB_PREFIX

        if name.startswith(LB_PREFIX):
            lb_registry = getattr(self, "_lb_registry", None)
            lb = lb_registry.get_by_name(name[len(LB_PREFIX):]) if lb_registry else None
            return bool(lb and lb.members)

        if "@" in name:
            backend_key = name.rsplit("@", 1)[1]
            return self._backends.get(backend_key) is not None

        pinned_key = self._model_pins.get(name)
        if pinned_key and self._backends.get(pinned_key):
            return True

        await self.refresh_model_map()
        if self._model_map.get(name):
            return True

        # Same unambiguous normalized-id retry as resolve_backend_for_model.
        from augmentum.models.openai_compat import _normalized_model_name
        want = _normalized_model_name(name)
        if want:
            matches = [
                mname for mname in self._model_map
                if _normalized_model_name(mname.split("@", 1)[0]) == want
            ]
            return len(matches) == 1
        return False

    async def resolve_model_for_role(
        self,
        role: str,
        override: str = "",
        settings: object | None = None,
    ) -> tuple[ModelBackend, str]:
        """Resolve a model through the role-based fallback chain.

        Roles map onto the three engine slots one-for-one: ``primary`` → Slot A,
        ``utility`` → Slot B, ``classifier`` → Slot C. ``heavyweight`` has no
        slot — it's for quality-critical work that usually wants a remote model.

        Resolution order:
        1. ``override`` (per-feature setting) if non-empty
        2. Role-specific setting (``classifier_model``, ``utility_model``,
           ``heavyweight_model``)
        3. The role's slot when it holds a resolvable model — Slot C's sidecar
           for ``classifier``, Slot B's ``engine_secondary_model`` for
           ``utility``. Skipped when the slot is disabled or empty.
        4. ``primary_chat_model`` — what "Auto — use Primary" in the UI promises:
           the model the user is actually chatting with.
        5. Default backend via ``resolve_backend_for_model("")`` (last resort —
           picks first model on default backend, which is rarely what the user
           wants for utility/distiller roles).

        Step 5's blind fallback may log a warning via ``role_min_param_billions``
        if the resolved model looks too small for the requested role. The guard
        is soft (warn-only) and never blocks — it just upgrades to
        ``primary_chat_model`` if available, or proceeds with the small model
        otherwise. A user who deliberately picks a small chat model (e.g.
        Qwen3.5-2B) is never blocked because step 4 returns first.
        """
        # NOTE: the numbered steps above are the user-visible contract. Every
        # step is a user-settable knob — do not add an implicit preference
        # between them, and do not add a step that picks a model the user
        # never named (see "never auto-select" in CLAUDE.md).
        # 1. Per-feature override — explicit user choice, no second-guessing.
        if override:
            return await self.resolve_backend_with_fabric(override)

        # 2-4. Role-specific settings, then the role's own slot.
        if settings:
            if role == "classifier":
                cm = getattr(settings, "classifier_model", "")
                if cm:
                    return await self.resolve_backend_with_fabric(cm)
                # Dedicated SmolLM-135M sidecar — preferred for this
                # latency-critical hop when present. An explicit
                # classifier_model above still wins; this only kicks in when
                # the role is left on "auto".
                sidecar = self._backends.get("classifier")
                if sidecar is not None:
                    return sidecar, getattr(
                        settings, "classifier_sidecar_model",
                        "smollm2-135m-instruct",
                    )
            # ``utility_model`` is the utility tier's explicit setting. The
            # classifier borrows it ONLY as tier-absent degradation: for
            # ``role == "classifier"`` this line is unreachable unless both
            # ``classifier_model`` is blank AND no Slot C backend exists, i.e.
            # the classifier tier isn't present at all. Deliberate — better a
            # small utility model than the blind default-backend fallback.
            if role in ("classifier", "utility"):
                um = getattr(settings, "utility_model", "")
                if um:
                    return await self.resolve_backend_with_fabric(um)
            # Slot B is the UTILITY tier — memory consolidation/compaction/
            # reflection, titles, and other recurring sub-turn reasoning tasks.
            #
            # This used to return the Slot C classifier sidecar, which was a
            # historical accident, not the design: Slot C predates Slot B
            # having a persisted, re-pinned model id, so it was the only small
            # resident thing to point at. The cost was real — utility work is
            # chunky (compaction, summarisation) while the classifier runs on a
            # ~2.5s voice/architect budget, and both queued on the SAME
            # llama-server process. Three slots, three tiers: A primary,
            # B utility, C classifier.
            #
            # Reached BY NAME through Slot B's registry pin (same idiom as
            # ``agents/dispatch.py::_fast_model_spec``), so no direct
            # ``_backends`` lookup is needed and fabric peers still apply. The
            # resolvability guard means an empty or stale Slot B degrades to
            # ``primary_chat_model`` below rather than raising — utility work
            # keeps running on a box where the slot was never loaded.
            #
            # A manual ``utility_model`` (or a per-feature ``override``) above
            # still wins, so a user who runs their OWN model is never overridden.
            if role == "utility" and getattr(settings, "engine_secondary_enabled", False):
                slot_b = (getattr(settings, "engine_secondary_model", "") or "").strip()
                if slot_b and await self.model_is_resolvable(slot_b):
                    return await self.resolve_backend_with_fabric(slot_b)
            # Heavyweight / frontier slot — quality-critical work that
            # warrants a stronger (usually paid / remote) model. Used
            # by the Bug Finder verifier, the stagnation-escalation
            # buddy, the future /second-opinion command, narrative
            # summariser escalation, and classifier hard-case
            # fallback. Per-feature overrides (e.g. the per-workspace
            # ``coder_workspaces.bug_finder_verifier_model``) are
            # passed via the ``override`` arg above and take priority;
            # this is the global default that kicks in when no
            # override is set.
            if role == "heavyweight":
                hm = getattr(settings, "heavyweight_model", "")
                if hm:
                    return await self.resolve_backend_with_fabric(hm)

            # 4. "Auto — use Primary" — fall back to the chat model the user
            # actively selected in the UI. Pushed by the chat model picker.
            pm = getattr(settings, "primary_chat_model", "")
            if pm:
                resolved = await self.resolve_backend_with_fabric(pm)
                # Skip the size guard — user explicitly chose this model for chat,
                # so it's fit-for-purpose by definition.
                return resolved

        # 5. Default backend's first model. Last resort. Soft size-guard:
        # log a warning if the resolved model looks too small for the role,
        # and try primary as a free upgrade — but never block. A user with
        # only a small model available should still get *something* back.
        backend, model_name = await self.resolve_backend_with_fabric("")
        if settings and model_name:
            min_b = float(getattr(settings, "role_min_param_billions", 0) or 0)
            if min_b > 0 and _model_too_small(model_name, min_b):
                log.warning(
                    "role_resolution_small_fallback",
                    role=role,
                    resolved_model=model_name,
                    min_billions=min_b,
                    hint="Pick a chat model in the UI to set primary_chat_model, "
                         "or set utility_model explicitly. Proceeding with the "
                         "small model — distiller-format outputs may be empty.",
                )
        return backend, model_name

    async def load_runtime_providers(self, provider_store: ProviderStore) -> None:
        """Load enabled providers from the database and register them.

        Profile resolution order, lowest-priority last:
        1. The ``profile_id`` stored on the row (the user's explicit choice
           at create or last-edit time).
        2. URL-pattern match (handles legacy rows from before migration 112
           and rows created without the user picking a profile).
        3. ``None`` — backend constructed without provider-specific
           post-processing.

        Only OpenAI-compatible providers consult profiles; native adapters
        (claude, gemini) handle provider quirks inside the backend itself.
        """
        providers = await provider_store.list_providers(enabled_only=True)
        for p in providers:
            try:
                profile: ProviderProfile | None = None
                if p.provider_type == "openai":
                    if p.profile_id:
                        profile = get_profile(p.profile_id)
                    if profile is None:
                        profile = get_profile_for_url(p.base_url)

                backend = create_backend_from_profile(
                    profile,
                    api_key=p.api_key or "",
                    http_client=self._http_client,
                    chat_client=self._chat_http_client,
                    provider_type=p.provider_type,
                    base_url=p.base_url,
                )
                self.register_backend(p.id, backend)
                self.set_provider_meta(p.id, p.owner_user_id, p.shared)
                log.info(
                    "runtime_provider_loaded",
                    id=p.id,
                    name=p.name,
                    base_url=p.base_url,
                    provider_type=p.provider_type,
                    profile=profile.id if profile else "",
                )
            except Exception:
                log.warning(
                    "runtime_provider_load_failed",
                    id=p.id,
                    exc_info=True,
                )
