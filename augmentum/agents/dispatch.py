"""SubagentDispatcher — orchestrates one ``task_dispatch`` invocation.

Lifecycle for a single spawn:

1. **Resolve** the role (registry → built-ins).
2. **Resolve** the model (preferred → fallback chain via fabric-aware
   resolver). Override from the tool call wins if non-empty.
3. **Filter** the workspace tool registry to the role's allow-list +
   any tool_overrides from the call.
4. **Bridge** parent context (slim/workspace/hot) into the child's
   initial user message.
5. **Persist** a breadcrumb (start_run).
6. **Run** the loop via ``run_subagent`` under a parent-cancel-aware
   asyncio task with a per-role concurrency semaphore.
7. **Persist** the result (complete_run) and return the
   ``SubagentResult`` to the caller.

Concurrency: per-role semaphore caps parallel spawns of the *same* role
under one parent turn. Inter-role parallelism is naturally bounded by
the lead model's parallel tool-call count.
"""

from __future__ import annotations

import asyncio
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from augmentum.agents.budget import SubagentBudget
from augmentum.agents.context_bridge import (
    build_initial_user_message,
    extract_orientation,
    extract_recent_tool_digests,
    extract_workspace_facts,
)
from augmentum.agents.guards import role_guard
from augmentum.agents.loop import (
    SubagentProgress,
    SubagentProgressCallback,
    SubagentResult,
    SubagentSpec,
    run_subagent,
)
from augmentum.agents.persistence import SubagentRunStore
from augmentum.agents.registry import AgentRegistry
from augmentum.agents.resolve import (
    SubagentModelUnavailableError,
    resolve_subagent_model,
)
from augmentum.agents.spec import AgentRole
from augmentum.agents.tools import filter_tools
from augmentum.models.provider_registry import ProviderRegistry
from augmentum.tools.base import Tool
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Tracks the current subagent nesting depth in the asyncio context.
# 0 = lead model (the coder turn itself). 1 = a subagent the lead
# spawned. 2 = a sub-sub-agent the first subagent spawned. Etc.
#
# asyncio.Task copies parent contextvars by default, so ``set()`` in
# the dispatcher before awaiting ``run_subagent`` propagates into
# every tool call the subagent makes — including its own
# ``task_dispatch`` calls, which then read this value and compare to
# ``settings.coder_subagent_max_depth`` before spawning further.
#
# Refusal lives in ``SubagentDispatcher.dispatch`` (single chokepoint)
# rather than in the TaskDispatchTool, so anyone bypassing the tool
# layer (e.g. internal direct calls in future power-forge paths)
# still gets the depth cap.
_current_subagent_depth: ContextVar[int] = ContextVar(
    "augmentum_subagent_depth", default=0,
)


def current_subagent_depth() -> int:
    """Read the current nesting depth (test + diagnostics)."""
    return _current_subagent_depth.get(0)


# Roles eligible for the cheap-model arbitrage (see
# ``SubagentDispatcher._fast_model_spec``). These are the breadth-first,
# read-only, high-volume roles where a fast/cheap model is the right
# tradeoff — the lead fans out many of them and reads structured
# summaries back. Deep-judgment roles (review / security_review /
# threat_model / audit_zone / plan) are deliberately excluded: their
# value depends on a capable model, so they inherit the lead's model.
_FAST_FANOUT_ROLES: frozenset[str] = frozenset({"explore", "research"})


# Process-wide map: instance_id → dispatcher that owns the running
# subagent. Lets external surfaces (the cancel REST endpoint, an admin
# CLI) locate a specific in-flight subagent without threading a
# per-handler reference. Dispatchers self-register at dispatch start
# and unregister at end (or cancel). Single source of truth — no
# duplicate state with dispatcher._active_tasks; the registry only
# carries the dispatcher reference, the task is still owned by the
# dispatcher's _active_tasks dict.
_active_subagent_owners: dict[str, "SubagentDispatcher"] = {}


def find_subagent_owner(instance_id: str) -> "SubagentDispatcher | None":
    """Return the dispatcher that owns ``instance_id`` if it's
    currently in-flight. Used by the cancel route + admin tools."""
    return _active_subagent_owners.get(instance_id)


def list_active_subagents() -> list[str]:
    """instance_ids of every in-flight subagent across all
    dispatchers in this process. UI fleet view + tests."""
    return list(_active_subagent_owners.keys())


@dataclass(frozen=True)
class DispatchRequest:
    """One ``task_dispatch`` tool call's inputs, normalized."""

    role: str
    prompt: str
    model_override: str = ""
    success_criteria: tuple[str, ...] = ()
    """Definition-of-done the lead hands down: each item is one concrete,
    checkable condition the subagent must satisfy (or explicitly report it
    can't). Injected into the child's first message as a ``<success_criteria>``
    block so the subagent self-checks against real intent instead of the
    lead's freehand prose alone. Empty = no explicit contract."""
    constraints: tuple[str, ...] = ()
    """Hard limits the subagent must respect (e.g. "don't touch the
    migration files", "read-only — propose, don't apply"). Injected as a
    ``<constraints>`` block. Empty = none beyond the role's own rules."""
    tool_overrides_add: tuple[str, ...] = ()
    tool_overrides_remove: tuple[str, ...] = ()
    context_mode_override: str = ""
    parent_run_id: str = ""
    parent_turn_id: str = ""
    workspace_id: str = ""
    session_id: str = ""
    user_id: str = ""

    progress_callback: SubagentProgressCallback | None = None
    """Optional sink for inner-loop progress events. Set by callers
    that want to surface live activity (the coder TaskDispatchTool
    wires this to the parent's chat_egress). Errors in the callback
    are swallowed by the loop so a misbehaving sink can't break the
    subagent run."""


@dataclass(frozen=True)
class DispatchOutcome:
    """What the dispatcher returns to ``task_dispatch.execute``.

    Carries the SubagentResult plus the resolved metadata the caller
    needs to render the tool result + a UI card.
    """

    subagent_id: str
    role: str
    model_spec: str
    model_resolved: str
    result: SubagentResult


class SubagentDispatcher:
    """Stateful dispatcher — one per coder session is fine.

    Holds per-role concurrency semaphores plus the registry / store
    references.  Cheap to construct; no background tasks.
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        provider_registry: ProviderRegistry,
        store: SubagentRunStore | None = None,
        tool_registry_provider: Any = None,
        coder_state_provider: Any = None,
    ) -> None:
        self._registry = registry
        self._provider_registry = provider_registry
        self._store = store
        self._tool_provider = tool_registry_provider
        self._state_provider = coder_state_provider
        self._sems: dict[str, asyncio.Semaphore] = {}
        # Active subagents keyed by instance_id → the asyncio.Task
        # running ``run_subagent``. Lets ``cancel(instance_id)`` reach
        # a specific in-flight subagent without disturbing siblings.
        # Cleared on completion/cancel in the dispatch coroutine.
        self._active_tasks: dict[str, asyncio.Task] = {}
        # User-visible cancel reasons keyed by instance_id. Stored at
        # cancel() time so the dispatch path can label the result
        # cleanly when the CancelledError bubbles up. Trimmed when the
        # corresponding task is reaped.
        self._cancel_reasons: dict[str, str] = {}

    # ------------------------------------------------------------------

    def _sem_for(self, role: AgentRole) -> asyncio.Semaphore:
        sem = self._sems.get(role.name)
        if sem is None:
            sem = asyncio.Semaphore(max(1, role.max_concurrent))
            self._sems[role.name] = sem
        return sem

    # ------------------------------------------------------------------

    async def dispatch(self, req: DispatchRequest) -> DispatchOutcome:
        """Spawn one subagent, await completion, return the outcome.

        Errors that happen BEFORE the loop starts (role missing, no
        model available, nesting depth exceeded) raise ``ValueError`` /
        ``SubagentModelUnavailableError``; errors INSIDE the loop come
        back as a populated SubagentResult with ``stop_reason="error"``.
        """
        # 0. Depth cap. Read the caller's current depth from the
        #    context (0 = the lead model calling out from the coder
        #    turn). We refuse BEFORE registry / model resolution so a
        #    runaway recursion can't burn budget just to be rejected.
        current_depth = _current_subagent_depth.get(0)
        from augmentum.config import settings as _settings
        max_depth = int(getattr(_settings, "coder_subagent_max_depth", 1) or 1)
        if current_depth >= max_depth:
            log.warning(
                "subagent_depth_cap_hit",
                current_depth=current_depth,
                max_depth=max_depth,
                requested_role=req.role,
            )
            raise ValueError(
                f"task_dispatch refused: spawning this subagent would "
                f"nest to depth {current_depth + 1}, exceeding "
                f"coder_subagent_max_depth={max_depth}. Either complete "
                f"the work yourself, or — if you're operating as the "
                f"lead — raise the setting via PUT /api/config/tools "
                f"with `coder_subagent_max_depth`."
            )

        # 1. Refresh registry if any role file changed on disk.
        try:
            self._registry.refresh_if_stale()
        except Exception:
            log.warning("agent_registry_refresh_failed", exc_info=True)

        try:
            role = self._registry.get(req.role)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        # 2. Resolve model. ``preferred_model=""`` → role inherits
        #    parent's current model (read from provider_registry's default).
        preferred = role.preferred_model
        fallbacks = list(role.fallback_models)
        if not preferred and not req.model_override:
            # Empty-string resolver path returns the default backend's
            # first model — close enough to "parent's model".
            preferred = ""
            # Model arbitrage: the cheap fan-out roles (explore, research)
            # run on a fast/cheap model when one is configured, so the lead
            # gets a felt cost win from delegating rather than just context
            # relief. Only applies when the role didn't pin its own model
            # and the caller didn't override — an explicit choice always
            # wins. Falls back to the lead's model ("") if the fast model
            # can't resolve (e.g. Slot B not loaded yet), so this never
            # blocks a dispatch.
            fast_spec = self._fast_model_spec(role.name)
            if fast_spec:
                preferred = fast_spec
                if "" not in fallbacks:
                    fallbacks.append("")

        resolved = await resolve_subagent_model(
            role=role.name,
            preferred=preferred,
            fallbacks=fallbacks,
            registry=self._provider_registry,
            user_id=req.user_id,
            session_id=req.session_id,
            override=req.model_override,
        )

        # 3. Resolve tool subset. Start from the role's allow-list,
        #    apply per-call overrides, then filter the live tool registry.
        allowed = set(role.tools)
        for name in req.tool_overrides_add:
            allowed.add(name.strip())
        for name in req.tool_overrides_remove:
            allowed.discard(name.strip())
        if not role.can_spawn_subagents:
            allowed.discard("task_dispatch")

        all_tools: list[Tool] = self._materialize_tools()
        selected = filter_tools(all_tools, frozenset(allowed))

        # 4. Bridge context.
        mode = req.context_mode_override.strip() or role.context_mode
        state = self._materialize_coder_state()
        orientation = extract_orientation(state)
        facts = extract_workspace_facts(state)
        digests = extract_recent_tool_digests(state) if mode == "hot" else []
        initial_user = build_initial_user_message(
            prompt=req.prompt,
            context_mode=mode,
            orientation=orientation,
            workspace_facts=facts,
            recent_tool_digests=digests,
            success_criteria=req.success_criteria,
            constraints=req.constraints,
        )

        # 5. Build the spec.
        guard_name = role.tool_guard if role.tool_guard != "none" else ""
        guard = role_guard(guard_name) if guard_name else None
        # Verification gate: an independent judge checks the subagent's
        # output against the lead's success_criteria before its stop is
        # honored. Only armed when the lead actually handed down criteria —
        # which mutating roles always do (criteria are the contract) and
        # read-only fan-out usually doesn't, so cheap explores stay cheap.
        verify_enabled = bool(
            getattr(_settings, "coder_subagent_verify_enabled", True)
        )
        verify_reentry = int(
            getattr(_settings, "coder_subagent_verify_reentry", 1) or 0
        )
        spec = SubagentSpec(
            role=role.name,
            model=resolved.model_id,
            system_prompt=role.system_prompt,
            initial_user_message=initial_user,
            tools=selected,
            budget=role.budget,
            tool_guard=guard,
            instance_id=f"sa_{role.name}_{uuid.uuid4().hex[:8]}",
            verify=verify_enabled and bool(req.success_criteria),
            task_prompt=req.prompt,
            success_criteria=tuple(req.success_criteria),
            verify_max_reentry=max(0, verify_reentry),
            progress_callback=req.progress_callback,
        )

        # 6. Persist breadcrumb.
        if self._store is not None and role.log_persistence:
            try:
                await self._store.start_run(
                    subagent_id=spec.instance_id,
                    role=role.name,
                    prompt=req.prompt[:4_000],
                    model_spec=resolved.spec,
                    model_resolved=resolved.model_id,
                    backend_key=_backend_key(resolved.backend),
                    context_mode=mode,
                    parent_run_id=req.parent_run_id,
                    parent_turn_id=req.parent_turn_id,
                    workspace_id=req.workspace_id,
                    session_id=req.session_id,
                    user_id=req.user_id,
                )
            except Exception:
                log.warning(
                    "subagent_persist_start_failed",
                    instance_id=spec.instance_id,
                    exc_info=True,
                )

        # 7. Run under the per-role semaphore + with the depth
        #    contextvar bumped so any nested ``task_dispatch`` the
        #    subagent makes sees the new depth and re-runs the cap
        #    check at the top of this method. ``reset`` in the finally
        #    block restores prior depth so concurrent same-parent
        #    dispatches don't interfere with each other. The asyncio
        #    Task wrapper lets ``cancel(instance_id)`` reach this
        #    specific run; we register/unregister around the await.
        sem = self._sem_for(role)
        async with sem:
            log.info(
                "subagent_dispatch_start",
                instance_id=spec.instance_id,
                role=role.name,
                model_spec=resolved.spec,
                model_resolved=resolved.model_id,
                tools=len(selected),
                context_mode=mode,
                depth=current_depth + 1,
            )
            depth_token = _current_subagent_depth.set(current_depth + 1)
            cancelled_via_cancel = False
            try:
                # Run inside an asyncio.Task so cancel() can target it.
                # Without this wrap the only handle on the in-flight
                # work is the awaiting frame itself, which the cancel
                # API can't reach.
                run_task = asyncio.create_task(
                    run_subagent(spec, backend=resolved.backend),
                    name=f"subagent:{spec.instance_id}",
                )
                self._active_tasks[spec.instance_id] = run_task
                _active_subagent_owners[spec.instance_id] = self
                try:
                    result = await run_task
                except asyncio.CancelledError:
                    cancelled_via_cancel = (
                        spec.instance_id in self._cancel_reasons
                    )
                    if not cancelled_via_cancel:
                        # External cancellation (parent cancel, shutdown)
                        # — re-raise so the surrounding context unwinds.
                        raise
                    cancel_reason = self._cancel_reasons.pop(
                        spec.instance_id, "user cancelled",
                    )
                    # Synthesise a cancelled-result so the caller gets
                    # the same shape as a clean exit. Recovery hint
                    # tells the parent model what to do next.
                    from augmentum.agents.loop import _compute_recovery_hint
                    wallclock_ms = int(
                        (asyncio.get_event_loop().time()
                         - run_task.get_loop().time()) * 1000,
                    )
                    if wallclock_ms < 0:
                        wallclock_ms = 0
                    recovery = _compute_recovery_hint(
                        stop_reason="cancelled",
                        stop_detail=cancel_reason,
                        stuck_pattern=None,
                        role=role.name,
                        iterations=0,
                        instance_id=spec.instance_id,
                    )
                    result = SubagentResult(
                        role=role.name,
                        instance_id=spec.instance_id,
                        output=f"[cancelled: {cancel_reason}]",
                        tokens_in=0,
                        tokens_out=0,
                        wallclock_ms=wallclock_ms,
                        iterations=0,
                        tool_calls=0,
                        stop_reason="cancelled",
                        stop_detail=cancel_reason,
                        model_resolved=resolved.model_id,
                        recovery_hint=recovery,
                    )
            finally:
                self._active_tasks.pop(spec.instance_id, None)
                self._cancel_reasons.pop(spec.instance_id, None)
                _active_subagent_owners.pop(spec.instance_id, None)
                _current_subagent_depth.reset(depth_token)

        log.info(
            "subagent_dispatch_complete",
            instance_id=spec.instance_id,
            role=role.name,
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            tokens_total=result.tokens_in + result.tokens_out,
            wallclock_ms=result.wallclock_ms,
        )

        # 8. Persist completion.
        if self._store is not None and role.log_persistence:
            try:
                await self._store.complete_run(
                    result,
                    subagent_id=spec.instance_id,
                    user_id=req.user_id,
                )
            except Exception:
                log.warning(
                    "subagent_persist_complete_failed",
                    instance_id=spec.instance_id,
                    exc_info=True,
                )

        return DispatchOutcome(
            subagent_id=spec.instance_id,
            role=role.name,
            model_spec=resolved.spec,
            model_resolved=resolved.model_id,
            result=result,
        )

    # ------------------------------------------------------------------

    def is_running(self, instance_id: str) -> bool:
        """True iff the given subagent is currently in-flight."""
        task = self._active_tasks.get(instance_id)
        return task is not None and not task.done()

    def list_running(self) -> list[str]:
        """instance_ids of every in-flight subagent. Useful for the UI
        fleet view + admin cancel-all flows."""
        return [
            iid for iid, task in self._active_tasks.items()
            if not task.done()
        ]

    def cancel(self, instance_id: str, *, reason: str = "user cancelled") -> bool:
        """Cancel one in-flight subagent. Returns True if the cancel
        signal was delivered, False if no such subagent is running.

        The dispatch coroutine intercepts the resulting
        ``asyncio.CancelledError`` and synthesises a clean
        ``SubagentResult`` with ``stop_reason="cancelled"`` plus a
        recovery hint, so the parent loop gets the same shape on a
        cancel as on a stuck/budget exit. Siblings are unaffected —
        the cancel targets ONLY the named instance.
        """
        task = self._active_tasks.get(instance_id)
        if task is None or task.done():
            return False
        # Record the human-readable reason BEFORE issuing the cancel
        # so the dispatch coroutine's except-block can find it. Race
        # is harmless: even if the task completes between the two
        # operations, the reason is cleaned up in the finally block.
        self._cancel_reasons[instance_id] = (reason or "user cancelled")[:200]
        task.cancel()
        log.info(
            "subagent_cancel_requested",
            instance_id=instance_id,
            reason=reason[:200],
        )
        return True

    def _materialize_tools(self) -> list[Tool]:
        """Pull the live tool registry. Wrapped in a try so a misbehaving
        provider can't blow up the dispatcher."""
        if self._tool_provider is None:
            return []
        try:
            tools = self._tool_provider()
        except Exception:
            log.warning("subagent_tool_provider_failed", exc_info=True)
            return []
        if not isinstance(tools, list):
            return list(tools) if tools else []
        return tools

    def _fast_model_spec(self, role_name: str) -> str:
        """Resolve the fast/cheap model spec for a cheap fan-out role.

        Returns ``""`` (meaning "inherit the lead's model") unless the
        role is one of the designated cheap fan-out roles AND a fast
        model is available. Precedence:

        1. ``settings.coder_subagent_fast_model`` when set explicitly.
        2. the Slot B resident model (``engine_secondary_model``) when
           Slot B is enabled and currently holds a model.
        3. ``""`` — inherit the lead's model (unchanged behavior).

        Read-only roles that do a deep capable-model job (``review``,
        ``security_review``, ``threat_model``, ``audit_zone``, ``plan``)
        are intentionally NOT downgraded — only the breadth-first
        ``explore`` / ``research`` roles, where a fast model is the
        right tradeoff.
        """
        if role_name not in _FAST_FANOUT_ROLES:
            return ""
        try:
            from augmentum.config import settings as _settings
        except Exception:
            return ""
        explicit = (getattr(_settings, "coder_subagent_fast_model", "") or "").strip()
        if explicit:
            return explicit
        # Default to the Slot B resident model when the slot is enabled and
        # actually holds a model. The fallback chain (with "" appended)
        # degrades to the lead's model if the slot isn't resolvable.
        if getattr(_settings, "engine_secondary_enabled", False):
            slot_b = (getattr(_settings, "engine_secondary_model", "") or "").strip()
            if slot_b:
                return slot_b
        return ""

    def _materialize_coder_state(self) -> Any:
        if self._state_provider is None:
            return None
        try:
            return self._state_provider()
        except Exception:
            log.warning("subagent_state_provider_failed", exc_info=True)
            return None


def _backend_key(backend: Any) -> str:
    """Best-effort backend identity for persistence rows."""
    for attr in ("backend_id", "name", "id"):
        val = getattr(backend, attr, "")
        if val:
            return str(val)[:64]
    return type(backend).__name__[:64]


# Helper for typed budget overrides on a per-call basis. Not currently
# wired into the tool schema (budget is role-level only in v1) but kept
# here so Phase 2 can surface it without restructuring.
def override_budget(
    base: SubagentBudget,
    *,
    iterations: int | None = None,
    wallclock_s: float | None = None,
    tokens: int | None = None,
) -> SubagentBudget:
    return SubagentBudget(
        max_iterations=iterations if iterations is not None else base.max_iterations,
        max_wallclock_seconds=wallclock_s if wallclock_s is not None else base.max_wallclock_seconds,
        max_tokens=tokens if tokens is not None else base.max_tokens,
    )
