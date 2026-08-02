"""Handler factory — resolves the correct ModeHandler for a classified request."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from augmentum.classifier.router import Mode
from augmentum.config import settings
from augmentum.modes.agentic.handler import AgenticHandler
from augmentum.modes.analytical.handler import AnalyticalHandler
from augmentum.modes.coder.handler import CoderHandler
from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.handler import NarrativeHandler
from augmentum.modes.passthrough.handler import PassthroughHandler
from augmentum.proxy.session import (
    SESSION_HEADER,
    derive_kv_session_key,
    get_client_id,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.datastructures import State

    from augmentum.models.base import InternalChatRequest, ModelBackend
    from augmentum.modes.base import ModeHandler

log = get_logger(__name__)


def get_session_id_from_request(
    request: InternalChatRequest,
    client_id: str = "",
) -> str:
    """Derive a deterministic session ID from the parsed internal request.

    Uses the system prompt content (which contains the character card and is
    stable across turns) as the primary fingerprint.  Falls back to the first
    user message when there is no system prompt.

    When ``session_client_isolation`` is enabled and a *client_id* is provided,
    the client identity is prefixed to the hash input so that the same system
    prompt from different clients produces different session IDs.
    """
    parts: list[str] = []
    for msg in request.messages:
        if msg.role == "system":
            parts.append(msg.content)
            break
    if not parts:
        for msg in request.messages:
            if msg.role == "user":
                parts.append(msg.content)
                break
    if parts:
        content = "|".join(parts)
        if settings.session_client_isolation and client_id:
            content = f"{client_id}:{content}"
        fingerprint = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"ses_{fingerprint}"
    return "ses_default"


def resolve_session_keys(
    fastapi_request: Request,
    internal_req: InternalChatRequest,
    *,
    user_id: str = "",
    workspace_id: str = "",
) -> tuple[str, str]:
    """Return ``(session_id, kv_session_key)`` for a chat-style request.

    Two keys, two roles:

    * ``session_id`` is a coarse routing key — handler-cache lookup,
      narrative-engine selection, log correlation. Hash collisions between
      unrelated callers are tolerable here because the consumers either
      hold no cross-call state, or scope it tightly enough that a
      collision degrades to "share an idle handler instance."

    * ``kv_session_key`` controls llama-server slot save/restore. We only
      populate it when we have a *trustworthy* session source — the
      explicit ``X-Augmentum-Session`` header from the in-app UI (which
      assigns a unique id per branch and thus survives regenerations
      cleanly), or a coder workspace id. External API clients that don't
      send the header get an empty kv_session_key, which opts them out of
      slot save/restore. llama-server's per-slot token-prefix cache covers
      within-conversation reuse without needing a stable identifier, and
      we avoid the false-match class of bugs where two unrelated
      conversations share enough surface (system prompt, first user
      message) to collide on a fingerprint.
    """
    if workspace_id:
        return workspace_id, derive_kv_session_key(user_id, workspace_id)
    header = fastapi_request.headers.get(SESSION_HEADER, "")
    if header:
        return header, derive_kv_session_key(user_id, header)
    fp = get_session_id_from_request(
        internal_req, client_id=get_client_id(fastapi_request),
    )
    return fp, ""


def apply_prompt_cache_key(
    internal_req: InternalChatRequest,
    *,
    user_id: str = "",
    session_id: str = "",
) -> None:
    """Set the sticky prompt-cache routing hints on an outbound request.

    OpenAI-family providers shard their prompt cache by machine; ``prompt_cache_key``
    is the hint that keeps successive turns of one conversation landing on the same
    shard, which is the difference between an ~80% hit rate and a full re-charge of
    the prompt on every turn. On a long chat that is the single largest lever on
    cost, so it belongs on EVERY chat ingress — not just the external
    OpenAI-compatible one, which was the only caller until now. The in-app UI posts
    to the Ollama-shaped ``/api/chat`` and was therefore paying full price.

    ``session_id`` is the primary axis (per-chat prefix reuse) and ``user_id`` the
    secondary, so two users on one box never share a cache shard. Coder requests
    need no third axis: ``resolve_session_keys`` already returns the workspace id
    *as* the session id when one is present.

    ``openai_compat`` only forwards these when the provider profile sets
    ``supports_prompt_cache_key``, so this is inert for every other backend.
    """
    if not (user_id or session_id):
        return
    if internal_req.raw_options is None:
        internal_req.raw_options = {}
    # Never clobber an explicit key from the caller — lets a specialised
    # dispatch path substitute its own axis downstream.
    if "prompt_cache_key" not in internal_req.raw_options:
        internal_req.raw_options["prompt_cache_key"] = (
            f"aug-{(user_id or 'anon')[:32]}-{(session_id or 'nosess')[:32]}"
        )
    # 24h retention — GPT-5.5's default, but explicit for older 5.x families
    # whose 5-10min in-memory retention expires mid-run on long agentic turns.
    internal_req.raw_options.setdefault("prompt_cache_retention", "24h")


def _tool_name(t: dict) -> str:
    fn = t.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(t.get("name", ""))


def compute_direct_prefix_cache_key(
    user_id: str, request: InternalChatRequest,
) -> str:
    """Stable KV-slot key for a direct-mode (external harness) request.

    External coding harnesses (OpenCode, etc.) hit the OpenAI path via ``d/``
    direct mode and don't send our ``X-Augmentum-Session`` header, so
    ``resolve_session_keys`` leaves ``kv_session_key`` empty and they opt out of
    llama-server slot save/restore. For a LONE conversation that's fine — slot
    0's in-place prompt cache covers within-conversation reuse — but a web-UI or
    companion turn landing on slot 0 between two harness turns evicts the
    conversation and forces a full re-prefill on the next turn.

    Mirror the Claude Code fix (``compute_prefix_cache_key``): hash the STABLE
    head of the conversation — system message(s) + the first user message +
    tool names — so every turn of one conversation collides on a single key and
    routes to the same slot (save/restore beats re-prefill), while different
    conversations get distinct keys. The first user message + tools are folded
    in so two sessions that share an identical system prompt + tool set (same
    harness, same project) don't false-merge onto one slot — the exact collision
    class that got the bare system-message hash removed (see
    ``llama_cpp._session_key_for_request``). No-op for cloud backends, which
    ignore ``kv_session_key`` entirely. Returns "" for the anon tenant.
    """
    if not user_id:
        return ""
    h = hashlib.sha256()
    h.update(b"oc-prefix-v1\x00")
    seen_user = False
    for msg in request.messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg.role == "system":
            h.update(b"sys\x00")
            h.update(content.encode("utf-8", "ignore"))
            h.update(b"\x00")
        elif msg.role == "user" and not seen_user:
            h.update(b"user0\x00")
            h.update(content.encode("utf-8", "ignore"))
            h.update(b"\x00")
            seen_user = True
    names = sorted(
        _tool_name(t) for t in (getattr(request, "tools", None) or [])
        if isinstance(t, dict)
    )
    if names:
        h.update(b"tools\x00")
        for n in names:
            h.update(n.encode("utf-8", "ignore"))
            h.update(b"\x00")
    return f"{user_id}:oc:{h.hexdigest()[:12]}"


def _get_or_create_engine(session_id: str, app_state: State, *, user_id: str = "") -> NarrativeEngine:
    """Return the NarrativeEngine for *session_id*, creating one if needed.

    Engines are stored in an OrderedDict used as an LRU cache.  On cache hit
    the entry is moved to the end (most-recently-used position).  On cache
    miss the least-recently-used entry is evicted before inserting a new one
    when the cache is at capacity.

    When *user_id* is provided the cache key becomes ``(user_id, session_id)``
    so that the same session_id under different users gets separate engines.
    When user_id is empty (auth disabled) the key remains a plain string for
    backward compatibility.
    """
    engines: dict = app_state.narrative_engines
    cache_key: str | tuple[str, str] = (user_id, session_id) if user_id else session_id
    if cache_key in engines:
        engines.move_to_end(cache_key)
        return engines[cache_key]

    # Evict LRU entries if at capacity
    max_engines: int = getattr(settings, "narrative_max_engines", 50)
    while len(engines) >= max_engines:
        evicted_id, _ = engines.popitem(last=False)
        log.info("narrative_engine_evicted", evicted_session_id=evicted_id)

    log.info("narrative_engine_created", session_id=session_id, user_id=user_id or "(none)")
    engines[cache_key] = NarrativeEngine(
        session_id=session_id,
        context_budget=settings.narrative_context_budget,
        character_pct=settings.narrative_character_budget_pct,
        scene_pct=settings.narrative_scene_budget_pct,
        plot_pct=settings.narrative_plot_budget_pct,
        lore_pct=settings.narrative_lore_budget_pct,
        consistency_pct=settings.narrative_consistency_budget_pct,
    )
    return engines[cache_key]


def _resolve_passthrough_tools(
    app_state: State,
    header_tools: str | None = None,
    query: str | None = None,
) -> list[str]:
    """Merge header-requested tools with config defaults.

    When any tool is enabled, zero-cost utility tools (calculator, datetime,
    unit_converter, etc.) are auto-included — they have no external deps and
    are always useful for the LLM.

    Scheduling-substrate tools ride through the SAME auto-include step
    as the zero-cost utilities — no bespoke injection path: present
    whenever the user has any tool enabled AND a scheduling dispatcher
    exists (companion runtime OR SchedulerService), absent on toolless
    configs, removed by the "none" header like everything else. The
    model is the arbiter of whether to use them; keyword-gating them
    made any paraphrase outside the regex vocabulary unschedulable.
    The ``query`` parameter is retained for signature compatibility but
    no longer gates anything.

    Args:
        app_state: Application state (for tool_registry existence check).
        header_tools: Comma-separated tool names from X-Augmentum-Tools header.
                      "all" enables every registered tool.
        query: Latest user message, used to intent-gate the companion
               scheduling substrate. None preserves legacy always-on injection.

    Returns:
        List of tool name strings (may be empty).
    """
    if not getattr(app_state, "tool_registry", None):
        return []

    names: set[str] = set()

    # Config defaults
    defaults = settings.passthrough_tools.strip()
    if defaults:
        names.update(t.strip() for t in defaults.split(",") if t.strip())

    # Header override / addition. IMPORTANT: an EMPTY selector is an
    # explicit "no tools" — the chat UI sends X-Augmentum-Tools: ""
    # when the user toggles everything off, and analytical already
    # honors that as none. Passthrough historically treated "" as
    # "header absent" and fell back to config defaults + ride-alongs,
    # silently overriding the user's all-off choice (caught live
    # 2026-07-02: schedule tools appeared with tools fully disabled).
    # Only a truly ABSENT header (None — headerless API clients) falls
    # back to config defaults.
    explicit_none = False
    if header_tools is not None:
        header_tools = header_tools.strip()
        if header_tools.lower() in ("", "none"):
            explicit_none = True
        elif header_tools.lower() == "all":
            registry = app_state.tool_registry
            return [t.name for t in registry.list_tools()]
        else:
            names.update(t.strip() for t in header_tools.split(",") if t.strip())

    if explicit_none:
        # The user explicitly chose "no tools" — honor that for the whole
        # set including substrate. The companion's chat tools defer to
        # explicit user intent.
        return []

    # Auto-include ride-along tools when any tool is active — ONE
    # mechanism for utilities and substrate alike. The scheduling tools
    # join only when a dispatcher exists to fire the result (companion
    # runtime or SchedulerService — mirrors standing_gate); the model
    # decides whether to use them, no keyword gating. Toolless configs
    # stay toolless (pure-streaming fast path preserved).
    if names:
        from augmentum.proxy.config_routes import (
            PASSTHROUGH_AUTO_TOOLS,
            PASSTHROUGH_SCHEDULE_AUTO_TOOLS,
        )

        names.update(PASSTHROUGH_AUTO_TOOLS)
        _scheduling_up = (
            getattr(app_state, "companion_runtime", None) is not None
            or getattr(app_state, "scheduler_service", None) is not None
        )
        if _scheduling_up:
            names.update(PASSTHROUGH_SCHEDULE_AUTO_TOOLS)
            # Create-verbs-first ordering for text-tier prompt rendering
            # (tiny models bias toward earlier schema entries).
            ordered = [
                n for n in PASSTHROUGH_SCHEDULE_AUTO_TOOLS if n in names
            ]
            return [
                *ordered,
                *(n for n in names if n not in PASSTHROUGH_SCHEDULE_AUTO_TOOLS),
            ]

    return list(names)


def resolve_auto_invoke_tools(header_tools: str | None) -> set[str]:
    """Tool names eligible for DIRECT auto-invocation this turn.

    Only tools EXPLICITLY named in the ``X-Augmentum-Tools`` header count as
    per-turn button intent. Config-default tools (``settings.passthrough_tools``),
    the blanket ``"all"`` (UI "all tools" / training override), ``"none"``, and
    an absent header are NOT per-turn intent — they expose schemas to the model
    but must not auto-fire (otherwise ``auto_invoke_when_enabled`` tools like
    youtube / build_application run on every message and steal the turn).

    Returns the explicit name set, or an empty set (auto-invoke nothing) for
    ``all`` / ``none`` / empty / absent.
    """
    if not header_tools:
        return set()
    raw = header_tools.strip()
    if raw.lower() in ("all", "none", ""):
        return set()
    return {t.strip() for t in raw.split(",") if t.strip()}


def resolve_analytical_tools(header_tools: str | None) -> list[str] | None:
    """Resolve the analytical-mode tool filter from the raw header value.

    Unlike :func:`_resolve_passthrough_tools`, this *does not* merge in
    ``settings.passthrough_tools`` config defaults — for analytical mode
    the textbox tool selector is the source of truth, not server config.
    The phase capability map (``_PHASE_CATEGORIES``) still bounds what
    can be reached, but the user's selection picks within that bound.

    Returns:
        - ``None`` when the header is absent — caller should fall back to
          the phase-default capability set (legacy behaviour for callers
          that don't pipe a header through, like the voice route).
        - ``[]`` when the user explicitly chose "none" or sent an empty
          selector — no tools, even if a phase would otherwise allow them.
        - A list of tool names when the user picked specific tools or
          requested ``"all"`` (which expands to nothing here — falls
          through to ``None`` so the phase default applies unfiltered).
    """
    if header_tools is None:
        return None
    raw = header_tools.strip()
    if not raw:
        return []
    low = raw.lower()
    if low == "none":
        return []
    if low == "all":
        return None  # no filter — let the phase capability stand
    return [t.strip() for t in raw.split(",") if t.strip()]


def get_handler_for_mode(
    mode: Mode,
    backend: ModelBackend,
    session_id: str,
    app_state: State,
    *,
    passthrough_tools: list[str] | None = None,
    analytical_enabled_tools: list[str] | None = None,
    tool_synthesis_hint: str = "",
    workspace_id: str = "",
    flow_tune: dict | None = None,
    explicit_flow_id: str = "",
    user_id: str = "",
    coder_strategy: str = "",
) -> ModeHandler:
    """Build the appropriate ModeHandler based on the classified mode.

    Falls back to PassthroughHandler for unknown / unimplemented modes and on
    any error during NarrativeHandler construction.

    Args:
        passthrough_tools: Optional list of tool names to enable in passthrough mode.
    """
    # Resolve image subsystem state — shared across all modes
    image_queue = getattr(app_state, "image_queue", None)
    image_enabled = bool(settings.image_enabled and image_queue is not None)

    if mode == Mode.NARRATIVE:
        try:
            engine = _get_or_create_engine(session_id, app_state, user_id=user_id)
            state_manager = getattr(app_state, "state_manager", None)

            # Cache handlers alongside engines so group chat state persists
            handlers = getattr(app_state, "narrative_handlers", None)
            handler_key: str | tuple[str, str] = (user_id, session_id) if user_id else session_id
            if handlers is not None and handler_key in handlers:
                handler = handlers[handler_key]
                # Update backend in case model changed
                handler._backend = backend
                handlers.move_to_end(handler_key)
                return handler

            handler = NarrativeHandler(
                backend=backend,
                engine=engine,
                state_manager=state_manager,
                session_id=session_id,
                image_queue=image_queue,
                image_enabled=image_enabled,
                app_state=app_state,
                user_id=user_id,
            )

            if handlers is not None:
                # Evict oldest if at capacity (match engine cache size)
                max_handlers: int = getattr(settings, "narrative_max_engines", 50)
                while len(handlers) >= max_handlers:
                    evicted_id, _ = handlers.popitem(last=False)
                    log.info("narrative_handler_evicted", evicted_session_id=evicted_id)
                handlers[handler_key] = handler

            return handler
        except Exception:
            log.warning(
                "narrative_handler_fallback",
                session_id=session_id,
                exc_info=True,
            )
            return PassthroughHandler(
                backend=backend,
                image_queue=image_queue,
                image_enabled=image_enabled,
                session_id=session_id,
                tool_registry=getattr(app_state, "tool_registry", None),
                enabled_tools=passthrough_tools,
                tool_synthesis_hint=tool_synthesis_hint,
                custom_flow_store=getattr(app_state, "custom_flow_store", None),
                user_id=user_id,
                app_state=app_state,
            )

    if mode == Mode.ANALYTICAL:
        try:
            tool_registry = getattr(app_state, "tool_registry", None)
            prompt_cache = getattr(app_state, "prompt_cache", None)

            # Lazy-register MemoryRecallTool so UARF can query memory on-demand
            if tool_registry and not tool_registry.get("memory_recall"):
                memory_store = getattr(app_state, "memory_store", None)
                if memory_store:
                    from augmentum.tools.memory_recall import MemoryRecallTool

                    tool_registry.register(MemoryRecallTool(memory_store))
                    log.info("memory_recall_registered_lazily")
            # Note: narrative_engines dict access is also safe — sync code
            # between await points cannot interleave on a single event loop.

            flow_store = getattr(app_state, "flow_store", None)

            provider_registry = getattr(app_state, "provider_registry", None)

            circuit_breaker = getattr(app_state, "circuit_breaker", None)

            return AnalyticalHandler(
                backend=backend,
                tool_registry=tool_registry,
                prompt_cache=prompt_cache,
                image_queue=image_queue,
                image_enabled=image_enabled,
                session_id=session_id,
                flow_store=flow_store,
                provider_registry=provider_registry,
                circuit_breaker=circuit_breaker,
                flow_tune=flow_tune,
                explicit_flow_id=explicit_flow_id,
                # Textbox tool selector → header → here. The handler intersects
                # this with each phase's category capability so the user's pick
                # is the source of truth. ``None`` (header absent) keeps the
                # legacy "use whatever the phase allows" behaviour. Empty list
                # = "user explicitly chose no tools".
                enabled_tools=analytical_enabled_tools,
                user_id=user_id,
            )
        except Exception:
            log.warning(
                "analytical_handler_fallback",
                session_id=session_id,
                exc_info=True,
            )
            return PassthroughHandler(
                backend=backend,
                image_queue=image_queue,
                image_enabled=image_enabled,
                session_id=session_id,
                tool_registry=getattr(app_state, "tool_registry", None),
                enabled_tools=passthrough_tools,
                tool_synthesis_hint=tool_synthesis_hint,
                custom_flow_store=getattr(app_state, "custom_flow_store", None),
                user_id=user_id,
                app_state=app_state,
            )

    if mode == Mode.AGENTIC:
        try:
            tool_registry = getattr(app_state, "tool_registry", None)
            flow_store = getattr(app_state, "flow_store", None)
            artifact_store = getattr(app_state, "artifact_store", None)
            build_run_store = getattr(app_state, "build_run_store", None)
            task_store = getattr(app_state, "task_store", None)
            tool_call_cache = getattr(app_state, "tool_call_cache", None)

            # Cache by (user_id, session_id) so in-flight task state (working
            # memory, plan scratch) survives between request turns in the
            # same session — matches the narrative pattern at line 209.
            handlers = getattr(app_state, "agentic_handlers", None)
            handler_key: str | tuple[str, str] = (user_id, session_id) if user_id else session_id
            if handlers is not None and handler_key in handlers:
                handler = handlers[handler_key]
                handler._backend = backend
                handler._flow_tune = flow_tune
                # Refresh the per-turn flow selection on the cached handler.
                # Without this, the first request's selection would stick
                # for the life of the session — switching flows mid-chat
                # would silently no-op.
                handler._explicit_flow_id = explicit_flow_id
                # Keep the cache reference fresh in case lifespan rebuilt it.
                handler._tool_call_cache = tool_call_cache
                handler._build_run_store = build_run_store
                handlers.move_to_end(handler_key)
                return handler

            handler = AgenticHandler(
                backend=backend,
                tool_registry=tool_registry,
                session_id=session_id,
                user_id=user_id,
                task_store=task_store,
                tool_call_cache=tool_call_cache,
                flow_store=flow_store,
                artifact_store=artifact_store,
                build_run_store=build_run_store,
                flow_tune=flow_tune,
                explicit_flow_id=explicit_flow_id,
            )

            if handlers is not None:
                max_handlers: int = getattr(settings, "narrative_max_engines", 50)
                while len(handlers) >= max_handlers:
                    evicted_id, _ = handlers.popitem(last=False)
                    log.info("agentic_handler_evicted", evicted_session_id=evicted_id)
                handlers[handler_key] = handler

            return handler
        except Exception:
            log.warning(
                "agentic_handler_fallback",
                session_id=session_id,
                exc_info=True,
            )
            return PassthroughHandler(
                backend=backend,
                image_queue=image_queue,
                image_enabled=image_enabled,
                session_id=session_id,
                tool_registry=getattr(app_state, "tool_registry", None),
                enabled_tools=passthrough_tools,
                tool_synthesis_hint=tool_synthesis_hint,
                custom_flow_store=getattr(app_state, "custom_flow_store", None),
                user_id=user_id,
                app_state=app_state,
            )

    if mode == Mode.CODER:
        try:
            # Permission callback — bridges the handler's per-tool gate
            # (AUGMENTUM_CODER_PERMISSIONS=confirm_mutations) to the
            # registry that the UI's approval modal polls. Captures
            # user_id at handler-build time so each request is scoped to
            # the caller, not whichever client polled most recently.
            permission_registry = getattr(app_state, "permission_registry", None)
            permission_callback = None
            if permission_registry is not None:
                # Consult the per-workspace policy file BEFORE
                # prompting the user. Three possible verdicts:
                #   * "allow" → return True immediately (no modal).
                #   * "deny"  → return False immediately (no modal).
                #   * "ask"   → fall through to the modal, exactly
                #              the legacy behaviour.
                # Policy is re-read on every check (no caching) so
                # an operator-edited rule takes effect on the next
                # tool call — important for the "I just hit an
                # 'ask' prompt I want to permanently allow" flow.
                from augmentum.coder.policy import load_policy as _load_policy

                async def _permission_callback(tool_name: str, tool_input: dict) -> bool:
                    container_manager = getattr(app_state, "container_manager", None)
                    # Plan-mode auto-approve. Read planning_mode fresh
                    # each call so a Shift+Tab cycle the user just hit
                    # takes effect on the next tool dispatch without
                    # waiting for the next turn boundary. ``auto``
                    # mode bypasses both the policy AND the modal —
                    # used when the user explicitly trusts the run.
                    auto_mode = False
                    if container_manager is not None:
                        try:
                            info = await container_manager._get_workspace(workspace_id)
                            mode_now = (getattr(info, "planning_mode", "") or "").strip().lower()
                            auto_mode = mode_now == "auto"
                        except Exception:
                            auto_mode = False
                    if auto_mode:
                        return True
                    try:
                        policy = await _load_policy(container_manager, workspace_id)
                        verdict = policy.decide(tool_name, tool_input or {})
                    except Exception:
                        # Defensive: any policy error degrades to
                        # ask-the-user. Logged inside load_policy.
                        verdict = "ask"
                    if verdict in ("allow", "deny"):
                        # Durable audit row for policy-decided outcomes.
                        # Modal outcomes (user/timeout/disconnect) are
                        # audited inside the registry; auto_mode is
                        # deliberately unaudited (per-call noise — see
                        # migration 260 header).
                        from augmentum.coder.permission_audit import (
                            resolve_store as _resolve_audit_store,
                        )
                        store = _resolve_audit_store(app_state)
                        if store is not None:
                            await store.record(
                                tool_name=tool_name,
                                decision="allowed" if verdict == "allow" else "denied",
                                decided_by="policy",
                                user_id=user_id or "",
                                workspace_id=workspace_id or "",
                                tool_input=tool_input,
                            )
                        return verdict == "allow"
                    return await permission_registry.request(
                        user_id=user_id or "",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        workspace_id=workspace_id or "",
                    )
                permission_callback = _permission_callback

            state_manager = getattr(app_state, "state_manager", None)
            return CoderHandler(
                backend,
                session_id=session_id,
                tool_registry=getattr(app_state, "tool_registry", None),
                container_manager=getattr(app_state, "container_manager", None),
                workspace_id=workspace_id,
                permission_callback=permission_callback,
                review_registry=getattr(app_state, "review_registry", None),
                user_id=user_id or "",
                state_manager=state_manager,
                power_registry=getattr(app_state, "power_registry", None),
                settings_store=getattr(app_state, "settings_store", None),
                mcp_client=getattr(app_state, "mcp_client", None),
                coder_strategy=coder_strategy,
                provider_registry=getattr(app_state, "provider_registry", None),
                coder_run_broker=getattr(app_state, "coder_run_broker", None),
                jobs_store=getattr(app_state, "jobs_store", None),
                vision_router=getattr(app_state, "vision_router", None),
            )
        except Exception:
            log.warning("coder_handler_init_failed", exc_info=True)
            return PassthroughHandler(
                backend=backend,
                image_queue=image_queue,
                image_enabled=image_enabled,
                session_id=session_id,
                tool_registry=getattr(app_state, "tool_registry", None),
                enabled_tools=passthrough_tools,
                tool_synthesis_hint=tool_synthesis_hint,
                custom_flow_store=getattr(app_state, "custom_flow_store", None),
                user_id=user_id,
                app_state=app_state,
            )

    if mode == Mode.DIRECT:
        # Raw pass-through tier — see augmentum/modes/direct/handler.py.
        # The route layer should have short-circuited BEFORE reaching the
        # factory; this branch exists so any callsite that asks the
        # factory directly (tests, future internal dispatchers) gets the
        # right handler instead of falling through to passthrough.
        from augmentum.modes.direct.handler import DirectHandler
        return DirectHandler(backend=backend)

    if mode == Mode.BECCA_DIRECT:
        # The companion's own prompt pipeline as a chat-side handler.
        # Only reachable when the chat router picks the ``becca_direct``
        # subagent, which itself only registers when both
        # ``companion_runtime_enabled`` and ``companion_becca_direct_enabled``
        # are on. Internally, the handler falls back to passthrough on
        # any "not ready" condition so the chat path is robust to the
        # companion being unavailable.
        try:
            from augmentum.modes.becca_direct.handler import BeccaDirectHandler
            return BeccaDirectHandler(
                backend=backend,
                app_state=app_state,
                session_id=session_id,
                user_id=user_id or "",
            )
        except Exception:
            log.warning("becca_direct_handler_init_failed", exc_info=True)
            # Fall through to passthrough — companion is optional.

    # Unknown mode — fall through to passthrough
    tool_registry = getattr(app_state, "tool_registry", None)
    custom_flow_store = getattr(app_state, "custom_flow_store", None)
    return PassthroughHandler(
        backend=backend,
        image_queue=image_queue,
        image_enabled=image_enabled,
        session_id=session_id,
        tool_registry=tool_registry,
        enabled_tools=passthrough_tools or [],
        tool_synthesis_hint=tool_synthesis_hint,
        custom_flow_store=custom_flow_store,
        user_id=user_id,
        app_state=app_state,
    )


async def register_flow_tools_async(
    tool_registry,
    custom_flow_store,
    background_chain_manager,
    backend,
    *,
    provider_registry=None,
) -> int:
    """Register custom flows as callable tools in the tool registry.

    Each flow becomes a FlowTool that, when invoked by the LLM, launches
    the flow's step chain via the BackgroundChainManager.

    Returns the number of flow tools registered.
    """
    from augmentum.tools.flow_tool import FlowTool

    # Remove any previously registered flow tools (re-sync after CRUD)
    existing = [t.name for t in tool_registry.list_tools() if t.name.startswith("flow_")]
    for name in existing:
        tool_registry.unregister(name)

    flows = await custom_flow_store.list_flows()
    count = 0
    for flow in flows:
        if not flow.get("enabled", True):
            continue

        async def _launcher(
            flow_dict, query, session_id, *,
            user_id: str = "", request_context=None,
            _bg=background_chain_manager, _be=backend, _tr=tool_registry,
        ):
            # provider_registry was bound on _bg at construction time; no
            # need to re-pass it per launch. tool_registry MUST ride along:
            # _run_chain hard-fails without it, so every FlowTool launch
            # (chat function-call or ATP) died with "Backend or tool
            # registry not available" before this was bound.
            return await _bg.launch(
                flow_dict, query, session_id,
                user_id=user_id,
                backend=_be,
                tool_registry=_tr,
                request_context=request_context,
            )

        tool = FlowTool(flow=flow, chain_launcher=_launcher)
        tool_registry.register(tool)
        count += 1

    return count
