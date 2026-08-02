"""Frozen legacy strategy implementations for Coder mode.

Reachable only via ``AUGMENTUM_CODER_STRATEGY=legacy``. Preserved as a
rollback path after the 2026-04-20 switch to ``_act_hybrid`` as the
default. **Do not add features here** — extend ``_act_hybrid`` or
``_act_canonical`` in ``handler.py`` instead.

These methods are implemented as a mixin so ``self.*`` references the
full ``CoderHandler`` instance (state, backend, container manager,
etc). Any module-level helpers or imports referenced below resolve via
``handler.py`` at call time — the circular import is safe because the
mixin class body doesn't evaluate them at definition time.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator

import structlog

from augmentum.coder.harness import (
    REWOO_FIX_SYSTEM,
    REWOO_PLAN_SYSTEM,
    parse_rewoo_plan,
    record_result,
    select_harness,
)
from augmentum.coder.prompts import (
    ACT_SYSTEM,
    EDIT_FORMAT_INSTRUCTIONS,
    MISSION_ACT_SYSTEM,
    MISSION_PLAN_SYSTEM,
    MISSION_REPLAN_SYSTEM,
)
from augmentum.coder.state import CoderPhase

# ``create_coder_tools`` is imported lazily via ``_bind_handler_helpers``
# so tests that monkeypatch ``handler.create_coder_tools`` still affect
# legacy code paths. See the forwarded list below.
from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.analytical.tool_calling import ToolCallingTier, select_tier
from augmentum.promises import (
    ActEvent,
    ActEventKind,
    MissionRunner,
    Promise,
    PromiseContext,
    PromiseStatus,
    RunnerEvent,
    RunnerEventKind,
    Verification,
    VerificationKind,
    parse_mission_json,
    parse_prose_plan,
    render_mission_log,
)
from augmentum.promises.verify import VerifyFn, default_verify_fns

log = structlog.get_logger(__name__)


# Module-level helpers (``_strip_tool_json``, ``_tool_to_schema``,
# ``_promise_summary``, ``_preview_len``, etc.) live in ``handler.py``.
# The legacy mixin methods reference them by bare name, which Python
# resolves via ``LOAD_GLOBAL`` against *this* module's ``__dict__``.
#
# A direct ``from augmentum.modes.coder.handler import _strip_tool_json``
# at module top would circular-import (handler imports this module for
# the mixin class). And PEP 562 ``__getattr__`` does NOT intercept
# bare-name lookups inside methods — only external attribute access.
#
# Instead, ``handler.py`` calls :func:`_bind_handler_helpers` at the
# bottom of its module body (after all helpers are defined), which
# injects them into this module's namespace. First-legacy-call is the
# earliest they're needed, and handler.py has finished loading by then.
_HELPERS_BOUND = False


class _LiveProxy:
    """Callable proxy that resolves ``handler.<attr>`` on every call.

    Needed for names that tests monkeypatch (e.g. ``create_coder_tools``)
    — caching the reference at bind-time would make monkeypatches on
    ``handler.create_coder_tools`` invisible to legacy code paths.
    """
    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        from augmentum.modes.coder import handler as _h
        return getattr(_h, self._attr)(*args, **kwargs)


# Names that tests commonly monkeypatch on handler.py — bind these as
# live proxies rather than cached references so the patches propagate.
_LIVE_NAMES = ("create_coder_tools",)


def _bind_handler_helpers() -> None:
    """Late-bind handler.py's module-level helpers into this module.

    Called once by handler.py after it finishes defining everything.
    Idempotent — safe to re-invoke.
    """
    global _HELPERS_BOUND
    if _HELPERS_BOUND:
        return
    from augmentum.modes.coder import handler as _h

    forwarded = (
        "_strip_tool_json", "_strip_cot_tokens",
        "_extract_tool_calls_from_text", "_tool_to_schema",
        "_preview_len", "_promise_summary",
        "_estimate_complexity", "_needs_observation_loop",
        "_extract_steps_from_request",
        "_act_system_for_tier", "_mission_act_system_for_tier",
        "_render_mission_observations", "_batch_signature",
        "_tool_fingerprint", "_execute_tool",
        "_parse_plan_steps", "_has_unclaimed_code_block",
        "_has_content_loop", "_is_transient_backend_error",
        "_short_error_reason", "_soft_failure_target",
        "_intent_key", "_env_int",
        # Constants (uppercase) that the legacy methods reference.
        "_MUTATING_TOOLS",
    )
    g = globals()
    for name in forwarded:
        if hasattr(_h, name):
            g[name] = getattr(_h, name)
    # Live proxies — each call resolves handler.<name> fresh, so test
    # monkeypatches on handler take effect in legacy code paths.
    for name in _LIVE_NAMES:
        g[name] = _LiveProxy(name)
    _HELPERS_BOUND = True


# --- Loop budget defaults (earned-autonomy model) ---
# Referenced only by the legacy strategies below. Hybrid / canonical
# use their own module-level constants defined in handler.py.
_INITIAL_ITERATIONS = 20
_ITERATIONS_CEILING = 75         # hard safety backstop
_NO_PROGRESS_THRESHOLD = 8       # consecutive iterations with no write → soft-stop
_INITIAL_FANOUT = 5              # max parallel tool calls per iteration
_MIN_FANOUT = 1                  # floor when the model misbehaves

# Budget deltas per observed signal
_BUDGET_EARN_PER_SUCCESS = 1     # successful tool call returning output
_BUDGET_EARN_ON_WRITE = 2        # successful code_edit / file_write bonus
_BUDGET_LOSE_PER_FAILURE = 1     # per failed tool call (capped at 2/iteration)
_BUDGET_LOSE_ON_REPEAT = 2       # same batch signature twice in a row
_BUDGET_LOSE_ON_PARSE_FAIL = 1   # model emitted unparseable tool JSON
_BUDGET_LOSE_ON_BAD_FANOUT = 1   # shrink fanout_limit by 1 when batch is bad

# Legacy alias kept for other strategies that still reference it.
_MAX_ITERATIONS = _ITERATIONS_CEILING


class LegacyStrategyMixin:
    """Legacy heuristic strategies — frozen rollback path.

    Mixed into ``CoderHandler``. All methods preserve the exact
    behaviour they had in the pre-extraction handler. The env-var gate
    in ``_act_phase`` controls whether any of them run.
    """

    # ------------------------------------------------------------------
    # Legacy heuristic dispatcher — only reachable when
    # AUGMENTUM_CODER_STRATEGY=legacy. Preserves the pre-hybrid routing
    # between _act_mission / _act_decompose / _act_architect / _act_direct
    # so it can be resurrected for debugging or regression comparisons.
    # ------------------------------------------------------------------
    async def _act_phase_legacy(
        self,
        request: InternalChatRequest,
        workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        harness = select_harness(request.model)
        plan_steps = self._state.plan_steps or []

        user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )

        from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel
        complexity = ComplexityAnalyzer().analyze(request)
        is_complex = (
            complexity.level == ComplexityLevel.COMPLEX
            or len(plan_steps) >= 4
            or _estimate_complexity(user_msg) >= 4
        )
        needs_observation = _needs_observation_loop(user_msg)

        if needs_observation:
            log.info("coder.strategy_selected", model=request.model,
                     strategy="mission", reason="legacy:observation_needed")
            async for chunk in self._act_mission(request, workspace_context):
                yield chunk
        elif harness == "react":
            log.info("coder.strategy_selected", model=request.model,
                     strategy="mission", reason="legacy")
            async for chunk in self._act_mission(request, workspace_context):
                yield chunk
        elif is_complex and harness != "react":
            if not plan_steps:
                plan_steps = _extract_steps_from_request(user_msg)
            log.info("coder.strategy_selected", model=request.model,
                     strategy="decompose", reason="legacy",
                     steps=len(plan_steps))
            async for chunk in self._act_decompose(request, workspace_context, plan_steps):
                yield chunk
        elif harness == "architect":
            log.info("coder.strategy_selected", model=request.model,
                     strategy="architect", reason="legacy")
            async for chunk in self._act_architect(request, workspace_context):
                yield chunk
        else:
            log.info("coder.strategy_selected", model=request.model,
                     strategy="direct", reason="legacy")
            async for chunk in self._act_direct(request, workspace_context):
                yield chunk

    # ------------------------------------------------------------------
    # Direct strategy (ReWOO — single LLM call → JSON array → execute)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Decompose strategy — break complex tasks into independent subtasks
    # ------------------------------------------------------------------

    async def _act_decompose(
        self,
        request: InternalChatRequest,
        workspace_context: str,
        plan_steps: list[str],
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute a complex task by running each plan step as an independent subtask."""
        total = len(plan_steps)

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model,
            extra={"strategy": "decompose", "total_steps": total},
        )

        yield InternalStreamChunk(
            content_delta=f"Decomposing into {total} steps...\n\n",
            model=request.model,
            augmentum={"mode": "coder", "phase": "executing", "status": "decomposing"},
        )

        completed_summaries = []

        for i, step in enumerate(plan_steps):
            step_num = i + 1
            self._state.current_step = step_num

            # Show step header
            yield InternalStreamChunk(
                content_delta=f"Step {step_num}/{total}: {step}\n",
                model=request.model,
                augmentum={
                    "mode": "coder", "phase": "executing",
                    "status": "step_start",
                    "step": step_num, "total": total,
                    "description": step,
                },
            )

            # Build focused context for this subtask
            # Include: workspace state + what prior steps accomplished
            subtask_context = workspace_context
            if completed_summaries:
                subtask_context += "\n\n## Completed Steps\n"
                for s in completed_summaries:
                    subtask_context += f"- {s}\n"

            # Build a focused request with just this step as the task
            subtask_messages = [
                Message(role="user", content=f"Execute this step: {step}"),
            ]
            subtask_request = InternalChatRequest(
                model=request.model,
                messages=subtask_messages,
                stream=True,
                temperature=request.temperature,
            )

            # Run this subtask through the Direct strategy
            # (each step is simple enough for one LLM call)
            subtask_tools = create_coder_tools(
                self._container_manager, self._workspace_id, self._state,
            )
            tool_map = {t.name: t for t in subtask_tools}
            known_tools = set(tool_map.keys())

            # Build ReWOO prompt for this specific step
            step_system = REWOO_PLAN_SYSTEM
            if subtask_context:
                step_system += f"\n\n{subtask_context}"

            step_messages = self._build_messages(subtask_request, step_system)
            step_llm_request = InternalChatRequest(
                model=request.model,
                messages=step_messages,
                stream=True,
                temperature=request.temperature,
            )

            # Get tool calls for this step
            content_parts = []
            try:
                async for chunk in self._backend.chat_stream(step_llm_request):
                    if chunk.content_delta:
                        content_parts.append(chunk.content_delta)
            except Exception as exc:
                yield InternalStreamChunk(
                    content_delta=f"\n  [Step {step_num} error: {str(exc)[:150]}]\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "error"},
                )
                completed_summaries.append(f"Step {step_num}: FAILED — {str(exc)[:80]}")
                continue

            step_content = "".join(content_parts)
            step_calls = parse_rewoo_plan(step_content, known_tools)

            if not step_calls:
                # Model output prose instead of tools — show it and move on
                clean = re.sub(r'\n{3,}', '\n', step_content).strip()
                if clean:
                    yield InternalStreamChunk(
                        content_delta=f"  {clean}\n",
                        model=request.model,
                        augmentum={"mode": "coder", "phase": "executing", "status": "streaming"},
                    )
                completed_summaries.append(f"Step {step_num}: {step} (no tools called)")
                continue

            # Execute tool calls for this step
            step_errors = []
            step_files_changed = []

            for tc in step_calls:
                tool_input = tc.tool_input
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except json.JSONDecodeError:
                        tool_input = {}

                tool_result, checkpoint_hash, tool_id = await self._execute_tool_with_verification(
                    tc.tool_name, tool_input, tool_map,
                )

                # Emit tool metadata
                yield self._meta_chunk(
                    phase="executing", status="tool_call",
                    model=request.model,
                    extra={"tool_call": {"id": tool_id, "tool": tc.tool_name, "input": tool_input}},
                )

                result_preview = (tool_result.output or tool_result.error)[:_preview_len(tc.tool_name)]
                extra = {"tool_result": {
                    "id": tool_id, "tool": tc.tool_name,
                    "success": tool_result.success,
                    "output_preview": result_preview,
                }}
                if checkpoint_hash:
                    extra["tool_result"]["checkpoint"] = checkpoint_hash
                    step_files_changed.append(checkpoint_hash)

                yield self._meta_chunk(
                    phase="executing", status="tool_result",
                    model=request.model, extra=extra,
                )

                if not tool_result.success:
                    step_errors.append(f"{tc.tool_name}: {tool_result.error}")
                elif tc.tool_name in ("shell_exec", "shell_read") and tool_result.output:
                    out_lower = tool_result.output.lower()
                    if any(sig in out_lower for sig in ["error", "traceback", "syntaxerror", "failed"]):
                        step_errors.append(f"{tc.tool_name} output: {tool_result.output[:200]}")

            # Step summary
            if step_errors:
                summary = f"Step {step_num}: {step} — {len(step_errors)} error(s)"
                yield InternalStreamChunk(
                    content_delta=f"  ⚠ {len(step_errors)} error(s) in step {step_num}\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "step_warning"},
                )
            else:
                ckpt_info = f", ckpt {step_files_changed[-1]}" if step_files_changed else ""
                summary = f"Step {step_num}: {step} — done{ckpt_info}"
                yield InternalStreamChunk(
                    content_delta=f"  ✓ Step {step_num} complete{ckpt_info}\n",
                    model=request.model,
                    augmentum={
                        "mode": "coder", "phase": "executing",
                        "status": "step_complete",
                        "step": step_num, "total": total,
                    },
                )

            completed_summaries.append(summary)

        # Synthesize a final response from all step results
        total_errors = sum(1 for s in completed_summaries if "error" in s.lower() or "FAILED" in s)
        synth_results = [
            {"tool": "decompose", "success": "FAILED" not in s and "error" not in s.lower(),
             "output_preview": s}
            for s in completed_summaries
        ]
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content[:500]
                break
        async for chunk in self._synthesize_response(request, user_query, synth_results):
            yield chunk

        yield self._meta_chunk(
            phase="executing", status="complete", model=request.model,
        )

        record_result(request.model, "decompose", total_errors == 0)
        self._state.phase = CoderPhase.WAITING

    # ------------------------------------------------------------------
    # Direct strategy (ReWOO — single LLM call → JSON array → execute)
    # ------------------------------------------------------------------

    async def _act_direct(
        self, request: InternalChatRequest, workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Direct/ReWOO strategy: one LLM call -> JSON array -> execute all."""
        tools = create_coder_tools(self._container_manager, self._workspace_id, self._state)
        tool_map = {t.name: t for t in tools}
        known_tools = set(tool_map.keys())

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model,
            extra={"strategy": "direct"},
        )

        # Build the ReWOO prompt with workspace context
        # NOTE: Do NOT inject the plan text — it confuses small models.
        # The user's message IS the task. ReWOO plans in one shot.
        rewoo_system = REWOO_PLAN_SYSTEM
        if workspace_context:
            rewoo_system += f"\n\n## Workspace\n{workspace_context}"

        messages = self._build_messages(request, rewoo_system)

        # ONE LLM call — get the JSON array of tool calls
        rewoo_request = InternalChatRequest(
            model=request.model,
            messages=messages,
            stream=True,
            temperature=request.temperature,
        )

        content_parts = []
        try:
            async for chunk in self._backend.chat_stream(rewoo_request):
                if chunk.content_delta:
                    content_parts.append(chunk.content_delta)
        except Exception as exc:
            yield InternalStreamChunk(
                content_delta=f"\n[Agent error: {str(exc)[:200]}]\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "error"},
            )
            return

        full_content = "".join(content_parts)

        # Parse the JSON array of tool calls
        tool_calls = parse_rewoo_plan(full_content, known_tools)

        if not tool_calls:
            # Model didn't output valid tool calls — show what it said
            clean = re.sub(r'\n{3,}', '\n\n', full_content).strip()
            if clean:
                yield InternalStreamChunk(
                    content_delta=clean + "\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "streaming"},
                )
            record_result(request.model, "direct", False)
            self._state.phase = CoderPhase.WAITING
            return

        # Execute all tool calls sequentially
        errors = []
        collected_results = []  # For synthesis
        for tc in tool_calls:
            tool_name = tc.tool_name
            tool_input = tc.tool_input

            # Parse string args
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {}

            # Emit tool_call
            tool_result, checkpoint_hash, tool_id = await self._execute_tool_with_verification(
                tool_name, tool_input, tool_map,
            )

            # Emit metadata
            yield self._meta_chunk(
                phase="executing", status="tool_call",
                model=request.model,
                extra={"tool_call": {"id": tool_id, "tool": tool_name, "input": tool_input}},
            )

            result_preview = (tool_result.output or tool_result.error)[:_preview_len(tool_name)]
            extra = {"tool_result": {
                "id": tool_id, "tool": tool_name,
                "success": tool_result.success,
                "output_preview": result_preview,
            }}
            if checkpoint_hash:
                extra["tool_result"]["checkpoint"] = checkpoint_hash

            yield self._meta_chunk(
                phase="executing", status="tool_result",
                model=request.model, extra=extra,
            )

            # Collect for synthesis
            collected_results.append({
                "tool": tool_name,
                "success": tool_result.success,
                "output_preview": (tool_result.output or tool_result.error or "")[:500],
                "error": tool_result.error if not tool_result.success else None,
            })

            if not tool_result.success:
                errors.append(f"{tool_name}: {tool_result.error}")
            elif tool_name in ("shell_exec", "shell_read") and tool_result.output:
                # Stream shell output to user (they need to see results)
                shell_out = tool_result.output.strip()
                if shell_out:
                    yield InternalStreamChunk(
                        content_delta=f"{shell_out}\n",
                        model=request.model,
                        augmentum={"mode": "coder", "phase": "executing", "status": "shell_output"},
                    )
                # Detect errors in shell output even when the tool "succeeded"
                out_lower = tool_result.output.lower()
                if any(sig in out_lower for sig in [
                    "error", "traceback", "syntaxerror", "nameerror",
                    "typeerror", "importerror", "failed", "exception",
                    "command not found", "no such file",
                ]):
                    errors.append(f"{tool_name} output suggests failure:\n{tool_result.output[:300]}")

        # If errors occurred, try ONE correction pass
        if errors:
            yield InternalStreamChunk(
                content_delta=f"\n[{len(errors)} error(s) — attempting fix...]\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "fixing"},
            )

            fix_system = REWOO_FIX_SYSTEM.format(
                errors="\n".join(f"- {e}" for e in errors),
                previous_plan=full_content[:1000],
            )
            if workspace_context:
                fix_system += f"\n\n{workspace_context}"

            fix_messages = self._build_messages(request, fix_system)
            fix_request = InternalChatRequest(
                model=request.model, messages=fix_messages,
                stream=True, temperature=request.temperature,
            )

            fix_parts = []
            try:
                async for chunk in self._backend.chat_stream(fix_request):
                    if chunk.content_delta:
                        fix_parts.append(chunk.content_delta)
            except Exception:
                log.warning("coder_fix_stream_failed", exc_info=True)

            fix_content = "".join(fix_parts)
            fix_calls = parse_rewoo_plan(fix_content, known_tools)

            for tc in fix_calls:
                tool_input = tc.tool_input if isinstance(tc.tool_input, dict) else {}
                tool_result, checkpoint_hash, tool_id = await self._execute_tool_with_verification(
                    tc.tool_name, tool_input, tool_map,
                )
                yield self._meta_chunk(
                    phase="executing", status="tool_call",
                    model=request.model,
                    extra={"tool_call": {"id": tool_id, "tool": tc.tool_name, "input": tool_input}},
                )
                # Stream shell output from fix pass too
                if tool_result.success and tc.tool_name in ("shell_exec", "shell_read") and tool_result.output:
                    shell_out = tool_result.output.strip()
                    if shell_out:
                        yield InternalStreamChunk(
                            content_delta=f"{shell_out}\n",
                            model=request.model,
                            augmentum={"mode": "coder", "phase": "executing", "status": "shell_output"},
                        )
                result_preview = (tool_result.output or tool_result.error)[:_preview_len(tc.tool_name)]
                yield self._meta_chunk(
                    phase="executing", status="tool_result",
                    model=request.model,
                    extra={"tool_result": {"id": tool_id, "tool": tc.tool_name, "success": tool_result.success, "output_preview": result_preview}},
                )

        # Post-task verification — confirm files exist and tests pass
        collected_results = await self._verify_task_results(collected_results, tool_map)

        # Emit verification failures to the user
        for tr in collected_results:
            if tr.get("tool") == "verification" and not tr.get("success"):
                yield InternalStreamChunk(
                    content_delta=f"\n{tr['output_preview']}\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "verification_failed"},
                )
            elif tr.get("tool") == "test_run":
                status_text = "✓ Tests passed" if tr.get("success") else "✖ Tests failed"
                yield InternalStreamChunk(
                    content_delta=f"\n{status_text}\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "test_result"},
                )

        # Synthesize a user-facing response from the tool results
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content[:500]
                break
        async for chunk in self._synthesize_response(request, user_query, collected_results):
            yield chunk

        yield self._meta_chunk(
            phase="executing", status="complete", model=request.model,
        )

        record_result(request.model, "direct", len(errors) == 0)
        self._state.phase = CoderPhase.WAITING

    # ------------------------------------------------------------------
    # Architect strategy (reason freely → format as tool calls → execute)
    # ------------------------------------------------------------------

    async def _act_architect(
        self, request: InternalChatRequest, workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Architect strategy: reason freely -> format as tool calls -> execute."""
        tools = create_coder_tools(self._container_manager, self._workspace_id, self._state)
        tool_map = {t.name: t for t in tools}
        known_tools = set(tool_map.keys())

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model,
            extra={"strategy": "architect"},
        )

        # ARCHITECT CALL: reason about the solution (no formatting constraints)
        architect_system = (
            "You are a senior developer. Analyze this task and describe the solution "
            "step by step. Be specific about file paths, function names, and exact changes.\n"
            "Do NOT write tool calls or JSON. Just describe what needs to happen.\n"
            "Use python3 (not python) for all Python commands."
        )
        if workspace_context:
            architect_system += f"\n\n{workspace_context}"
        if self._state.plan:
            architect_system += f"\n\nPlan:\n{self._state.plan}"

        architect_messages = self._build_messages(request, architect_system)
        architect_request = InternalChatRequest(
            model=request.model, messages=architect_messages,
            stream=True, temperature=request.temperature,
        )

        arch_parts = []
        try:
            async for chunk in self._backend.chat_stream(architect_request):
                if chunk.content_delta:
                    arch_parts.append(chunk.content_delta)
        except Exception as exc:
            yield InternalStreamChunk(
                content_delta=f"\n[Architect error: {str(exc)[:200]}]\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "error"},
            )
            return

        architect_output = "".join(arch_parts)

        # Show architect's reasoning to the user (clean prose)
        clean_arch = re.sub(r'\n{3,}', '\n\n', architect_output).strip()
        if clean_arch:
            yield InternalStreamChunk(
                content_delta=clean_arch + "\n\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "architect_reasoning"},
            )

        # EDITOR CALL: translate reasoning into tool calls
        editor_system = REWOO_PLAN_SYSTEM
        if workspace_context:
            editor_system += f"\n\n{workspace_context}"

        editor_messages = [
            Message(role="system", content=editor_system),
            Message(role="user", content=f"Implement this solution:\n\n{architect_output}"),
        ]

        editor_request = InternalChatRequest(
            model=request.model, messages=editor_messages,
            stream=True, temperature=request.temperature,
        )

        editor_parts = []
        try:
            async for chunk in self._backend.chat_stream(editor_request):
                if chunk.content_delta:
                    editor_parts.append(chunk.content_delta)
        except Exception as exc:
            yield InternalStreamChunk(
                content_delta=f"\n[Editor error: {str(exc)[:200]}]\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "error"},
            )
            return

        editor_content = "".join(editor_parts)
        tool_calls = parse_rewoo_plan(editor_content, known_tools)

        if not tool_calls:
            yield InternalStreamChunk(
                content_delta="\n[Could not generate tool calls from the solution description.]\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "no_tools"},
            )
            record_result(request.model, "architect", False)
            self._state.phase = CoderPhase.WAITING
            return

        # Execute all tool calls (same as Direct)
        errors = []
        collected_results = []
        for tc in tool_calls:
            tool_input = tc.tool_input
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {}

            tool_result, checkpoint_hash, tool_id = await self._execute_tool_with_verification(
                tc.tool_name, tool_input, tool_map,
            )

            yield self._meta_chunk(
                phase="executing", status="tool_call",
                model=request.model,
                extra={"tool_call": {"id": tool_id, "tool": tc.tool_name, "input": tool_input}},
            )

            result_preview = (tool_result.output or tool_result.error)[:_preview_len(tc.tool_name)]
            extra = {"tool_result": {
                "id": tool_id, "tool": tc.tool_name,
                "success": tool_result.success,
                "output_preview": result_preview,
            }}
            if checkpoint_hash:
                extra["tool_result"]["checkpoint"] = checkpoint_hash
            yield self._meta_chunk(
                phase="executing", status="tool_result",
                model=request.model, extra=extra,
            )

            collected_results.append({
                "tool": tc.tool_name,
                "success": tool_result.success,
                "output_preview": (tool_result.output or tool_result.error or "")[:500],
            })

            if not tool_result.success:
                errors.append(f"{tc.tool_name}: {tool_result.error}")

        # Post-task verification
        collected_results = await self._verify_task_results(collected_results, tool_map)

        # Synthesize response
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content[:500]
                break
        async for chunk in self._synthesize_response(request, user_query, collected_results):
            yield chunk

        yield self._meta_chunk(
            phase="executing", status="complete", model=request.model,
        )

        record_result(request.model, "architect", len(errors) == 0)
        self._state.phase = CoderPhase.WAITING

    # ------------------------------------------------------------------
    # ReAct strategy (multi-turn agent loop — existing logic)
    # ------------------------------------------------------------------

    async def _act_react(
        self, request: InternalChatRequest, workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """ReAct strategy: multi-turn agent loop with parallel tool calls and
        a dynamic, signal-driven iteration budget.

        The loop runs until the budget reaches zero, a soft-stop fires
        (no-progress / repeat / task_complete / no_tool_calls), or the
        backend raises. Successful tool calls earn additional iterations
        (bounded by ``_ITERATIONS_CEILING``); failures and parse fallbacks
        shrink the budget and the per-iteration fan-out.
        """
        # Initialize loop budget on state
        self._state.iterations_remaining = _INITIAL_ITERATIONS
        self._state.iterations_ceiling = _ITERATIONS_CEILING
        self._state.iterations_since_progress = 0
        self._state.fanout_limit = _INITIAL_FANOUT
        self._state.consecutive_failures = 0

        # Track consecutive *batch* repeats (sorted signature of all calls)
        _last_batch_sig: tuple[str, ...] = ()
        _batch_repeat_count = 0

        # Termination reason surfaced in final meta chunk
        termination_reason = "unknown"

        # Local ToolResult import used for gather exception conversion + lint wrap
        from augmentum.tools.base import ToolResult as _TR

        # Build all tools
        tools = create_coder_tools(
            self._container_manager,
            self._workspace_id,
            self._state,
        )
        tool_map = {t.name: t for t in tools}

        # Select tool calling tier based on backend capabilities
        tier = select_tier(self._backend, request.model)
        log.info("coder.tool_tier", tier=tier.value, model=request.model)

        # Only include tools schema for native tier
        tool_schemas = [_tool_to_schema(t) for t in tools] if tier == ToolCallingTier.NATIVE else None

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model,
            extra={"strategy": "react"},
        )

        # Build act system prompt: datetime + user system + ACT_SYSTEM + EDIT_FORMAT + context
        act_system = (
            f"{ACT_SYSTEM}\n\n{EDIT_FORMAT_INSTRUCTIONS}"
        )
        if workspace_context:
            act_system += f"\n\n{workspace_context}"
        if self._state.plan:
            act_system += f"\n\n## Current Plan\n\n{self._state.plan}"

        # Conversation history for the agent loop
        messages = self._build_messages(request, act_system)

        iteration = 0
        while self._state.iterations_remaining > 0:
            iteration += 1
            self._state.iterations_remaining -= 1

            log.debug(
                "coder.act_iteration",
                session_id=self._session_id,
                iteration=iteration,
                budget=self._state.iterations_remaining,
                fanout=self._state.fanout_limit,
            )

            # Surface current budget so UI and (eventually) prompt can see it
            yield self._meta_chunk(
                phase="executing",
                status="budget",
                model=request.model,
                extra={
                    "budget": {
                        "iteration": iteration,
                        "iterations_remaining": self._state.iterations_remaining,
                        "iterations_ceiling": self._state.iterations_ceiling,
                        "iterations_since_progress": self._state.iterations_since_progress,
                        "fanout_limit": self._state.fanout_limit,
                    }
                },
            )

            act_request = InternalChatRequest(
                model=request.model,
                messages=list(messages),
                stream=True,
                temperature=request.temperature,
                tools=tool_schemas,  # None for text tier — model uses prompt-based tool calls
            )

            content_parts: list[str] = []
            _tc_acc: dict[int, dict] = {}

            # Always buffer the full LLM response before emitting.
            # Streaming raw deltas leaks tool-call JSON fragments to
            # the user (text-tier models embed JSON inline, and even
            # native-tier content can contain noisy reasoning).
            # Tool activity is communicated via structured metadata
            # chunks (tool_call / tool_result), not content streaming.

            try:
                async for chunk in self._backend.chat_stream(act_request):
                    if chunk.content_delta:
                        content_parts.append(chunk.content_delta)
                    if chunk.augmentum and "tool_calls" in chunk.augmentum:
                        for tc_delta in chunk.augmentum["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in _tc_acc:
                                _tc_acc[idx] = {
                                    "id": tc_delta.get("id", str(uuid.uuid4())),
                                    "name": "",
                                    "arguments_parts": [],
                                }
                            acc = _tc_acc[idx]
                            if tc_delta.get("id"):
                                acc["id"] = tc_delta["id"]
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                acc["name"] = fn["name"]
                            if fn.get("arguments"):
                                acc["arguments_parts"].append(fn["arguments"])
            except Exception as exc:
                log.warning("coder.act_stream_failed",
                            iteration=iteration, error=str(exc))
                yield InternalStreamChunk(
                    content_delta=f"\n\n[Agent error on iteration {iteration}: {str(exc)[:200]}]\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "error"},
                )
                termination_reason = "backend_error"
                break

            full_content = "".join(content_parts)

            # Clean and emit buffered prose (strip tool JSON, CoT tokens,
            # control tags, code blocks).
            clean_text = _strip_tool_json(full_content)
            clean_text = _strip_cot_tokens(clean_text)
            clean_text = re.sub(r'<task_complete/>', '', clean_text)
            clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

            if clean_text:
                yield InternalStreamChunk(
                    content_delta=clean_text + "\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "streaming"},
                )

            # Assemble accumulated tool call deltas into complete calls
            assembled_tool_calls = []
            for idx in sorted(_tc_acc.keys()):
                acc = _tc_acc[idx]
                if acc["name"]:
                    args_str = "".join(acc["arguments_parts"])
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    assembled_tool_calls.append({
                        "id": acc["id"],
                        "name": acc["name"],
                        "input": args,
                    })

            # Try to extract tool calls: native chunks first, then text parsing.
            # Track whether we fell back (signal for fanout_limit demotion).
            parse_fallback_used = False
            response_tool_calls = assembled_tool_calls
            if not response_tool_calls and full_content:
                from augmentum.modes.analytical.tool_calling import parse_json_tool_calls
                known = set(tool_map.keys())
                parsed = parse_json_tool_calls(full_content, known_tools=known)
                if parsed:
                    response_tool_calls = [
                        {"id": str(uuid.uuid4()), "name": n, "input": a}
                        for n, a in parsed
                    ]
                    parse_fallback_used = True
                else:
                    response_tool_calls = _extract_tool_calls_from_text(full_content)
                    if response_tool_calls:
                        parse_fallback_used = True

            # Append assistant message to history
            if tier == ToolCallingTier.NATIVE:
                formatted_tc = None
                if response_tool_calls:
                    formatted_tc = []
                    for tc in response_tool_calls:
                        formatted_tc.append({
                            "id": tc.get("id", str(uuid.uuid4())),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {})),
                            },
                        })
                assistant_msg = Message(
                    role="assistant",
                    content=full_content or "",
                    tool_calls=formatted_tc,
                )
            else:
                clean_content = _strip_tool_json(full_content)
                final_content = clean_content.strip() or full_content.strip() or "(executing tool)"
                assistant_msg = Message(role="assistant", content=final_content)
            messages.append(assistant_msg)

            # Completion signal — agent emits <task_complete/> on its own line
            if "<task_complete/>" in full_content and not response_tool_calls:
                log.info("coder.act_complete",
                         session_id=self._session_id, iteration=iteration)
                termination_reason = "task_complete"
                break

            # No tool calls → agent is done (or stuck)
            if not response_tool_calls:
                log.info("coder.act_no_tool_calls",
                         session_id=self._session_id, iteration=iteration)
                termination_reason = "no_tool_calls"
                break

            # Apply fan-out cap — earned autonomy shrinks this on bad batches
            fanout = max(_MIN_FANOUT, self._state.fanout_limit)
            if len(response_tool_calls) > fanout:
                log.debug("coder.fanout_clipped",
                          requested=len(response_tool_calls), cap=fanout)
                response_tool_calls = response_tool_calls[:fanout]

            # Batch-level repeat detection — signature is the sorted set of
            # (name, input) across all calls in this turn. Same batch twice
            # in a row → stop.
            batch_sig_list = []
            for tc in response_tool_calls:
                name = tc.get("name") or ""
                raw_inp = tc.get("input") or {}
                if isinstance(raw_inp, str):
                    try:
                        raw_inp = json.loads(raw_inp)
                    except json.JSONDecodeError:
                        raw_inp = {}
                batch_sig_list.append(f"{name}:{json.dumps(raw_inp, sort_keys=True)}")
            batch_sig = tuple(sorted(batch_sig_list))

            if batch_sig and batch_sig == _last_batch_sig:
                _batch_repeat_count += 1
                if _batch_repeat_count >= 2:
                    log.warning("coder.repeat_detected",
                                iteration=iteration,
                                repeats=_batch_repeat_count,
                                batch=list(batch_sig))
                    yield InternalStreamChunk(
                        content_delta="\n\n[Agent stopped: repeated the same batch of actions. Please provide more specific instructions.]\n",
                        model=request.model,
                        augmentum={"mode": "coder", "phase": "executing", "status": "repeat_stopped"},
                    )
                    termination_reason = "repeat_stopped"
                    break
            else:
                _batch_repeat_count = 0
            _last_batch_sig = batch_sig

            # Normalize each call and emit tool_call metadata in order
            normalized_calls: list[tuple[str, str, dict]] = []
            for tc in response_tool_calls:
                tool_id = tc.get("id") or str(uuid.uuid4())
                tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                raw_input = tc.get("input") or tc.get("function", {}).get("arguments", {})
                if isinstance(raw_input, str):
                    try:
                        tool_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        tool_input = {}
                else:
                    tool_input = raw_input or {}

                normalized_calls.append((tool_id, tool_name, tool_input))

                yield self._meta_chunk(
                    phase="executing",
                    status="tool_call",
                    model=request.model,
                    extra={
                        "tool_call": {
                            "id": tool_id,
                            "tool": tool_name,
                            "input": tool_input,
                        }
                    },
                )

            # Execute the batch in two phases:
            #   1. Read-only calls in parallel (asyncio.gather)
            #   2. Mutating calls sequentially, after all reads complete
            # This makes read-before-edit safe when both target the same path,
            # and serialises same-file edits deterministically.
            gathered: list = [None] * len(normalized_calls)

            parallel_indices: list[int] = []
            serial_indices: list[int] = []
            for idx, (_tid, name, _inp) in enumerate(normalized_calls):
                if name in _MUTATING_TOOLS:
                    serial_indices.append(idx)
                else:
                    parallel_indices.append(idx)

            if parallel_indices:
                parallel_coros = [
                    _execute_tool(
                        tool_map=tool_map,
                        tool_name=normalized_calls[i][1],
                        tool_input=normalized_calls[i][2],
                    )
                    for i in parallel_indices
                ]
                parallel_results = await asyncio.gather(
                    *parallel_coros, return_exceptions=True,
                )
                for i, res in zip(parallel_indices, parallel_results, strict=True):
                    gathered[i] = res

            for i in serial_indices:
                # _execute_tool never raises — wraps exceptions as failed ToolResult
                gathered[i] = await _execute_tool(
                    tool_map=tool_map,
                    tool_name=normalized_calls[i][1],
                    tool_input=normalized_calls[i][2],
                )

            # Post-process results sequentially: LSP + git checkpoint per
            # successful mutation (git ops must not run concurrently), plus
            # meta emission + history feed.
            successes = 0
            failures = 0
            writes = 0

            for (tool_id, tool_name, tool_input), result_or_exc in zip(normalized_calls, gathered, strict=True):
                self._state.tool_calls_made += 1

                if isinstance(result_or_exc, BaseException):
                    log.warning("coder.tool_exception",
                                tool_name=tool_name, error=str(result_or_exc))
                    tool_result = _TR(
                        success=False,
                        error=f"Tool {tool_name!r} raised: {result_or_exc}",
                    )
                else:
                    tool_result = result_or_exc

                # LSP feedback loop for mutating tools
                if tool_result.success and tool_name in ("code_edit", "file_write"):
                    lint_output = await self._run_lint_check(tool_input.get("path", ""))
                    if lint_output:
                        tool_result = _TR(
                            success=tool_result.success,
                            output=(tool_result.output or "") + f"\n\n[Lint check]\n{lint_output}",
                            metadata=tool_result.metadata,
                        )

                # Auto-checkpoint after successful file modifications
                checkpoint_hash = None
                if tool_result.success and tool_name in ("code_edit", "file_write"):
                    path = tool_input.get("path", "unknown")
                    short_path = path.replace("/workspace/", "")
                    checkpoint_hash = await self._container_manager.git_checkpoint(
                        self._workspace_id,
                        f"Agent: {tool_name} {short_path}",
                    )

                result_preview = (tool_result.output or tool_result.error or "")[:_preview_len(tool_name)]

                tool_result_extra = {
                    "tool_result": {
                        "id": tool_id,
                        "tool": tool_name,
                        "success": tool_result.success,
                        "output_preview": result_preview,
                    }
                }
                if checkpoint_hash:
                    tool_result_extra["tool_result"]["checkpoint"] = checkpoint_hash

                yield self._meta_chunk(
                    phase="executing",
                    status="tool_result",
                    model=request.model,
                    extra=tool_result_extra,
                )

                # Feed result back into message history
                result_content = (
                    tool_result.output if tool_result.success else f"ERROR: {tool_result.error}"
                )

                if tier == ToolCallingTier.NATIVE:
                    messages.append(Message(
                        role="tool",
                        content=result_content or "",
                        tool_call_id=tool_id,
                    ))
                else:
                    result_text = f"[Tool result: {tool_name}]\n{result_content}"
                    if messages and messages[-1].role == "user":
                        messages[-1] = Message(
                            role="user",
                            content=messages[-1].content + "\n\n" + result_text,
                        )
                    else:
                        messages.append(Message(role="user", content=result_text))

                # Tally signals for budget scoring
                if tool_result.success:
                    successes += 1
                    if tool_name in ("code_edit", "file_write"):
                        writes += 1
                else:
                    failures += 1

            # --- Budget scoring: earn on success, lose on failure/parse issues ---
            earned = (
                successes * _BUDGET_EARN_PER_SUCCESS
                + writes * _BUDGET_EARN_ON_WRITE
            )
            lost = min(2, failures) * _BUDGET_LOSE_PER_FAILURE
            if parse_fallback_used:
                lost += _BUDGET_LOSE_ON_PARSE_FAIL
                self._state.fanout_limit = max(
                    _MIN_FANOUT, self._state.fanout_limit - _BUDGET_LOSE_ON_BAD_FANOUT
                )

            new_budget = self._state.iterations_remaining + earned - lost
            self._state.iterations_remaining = max(
                0, min(self._state.iterations_ceiling, new_budget)
            )

            # Track progress for no-progress soft-stop
            if writes > 0:
                self._state.iterations_since_progress = 0
                self._state.consecutive_failures = 0
            else:
                self._state.iterations_since_progress += 1
                if failures > 0 and successes == 0:
                    self._state.consecutive_failures += 1
                else:
                    self._state.consecutive_failures = 0

            if self._state.iterations_since_progress >= _NO_PROGRESS_THRESHOLD:
                log.info("coder.act_no_progress",
                         session_id=self._session_id,
                         iterations_since_progress=self._state.iterations_since_progress)
                yield InternalStreamChunk(
                    content_delta=f"\n\n[Agent stopped: {_NO_PROGRESS_THRESHOLD} iterations with no file changes. Summarize or ask for a new direction.]\n",
                    model=request.model,
                    augmentum={"mode": "coder", "phase": "executing", "status": "no_progress"},
                )
                termination_reason = "no_progress"
                break

        else:
            # while-else: budget reached zero without a break
            termination_reason = "budget_exhausted"
            log.warning(
                "coder.act_budget_exhausted",
                session_id=self._session_id,
                iterations=iteration,
            )
            yield self._meta_chunk(
                phase="executing",
                status="max_iterations_reached",
                model=request.model,
                extra={"reason": termination_reason},
            )

        # Done
        self._state.phase = CoderPhase.WAITING
        yield self._meta_chunk(
            phase="waiting",
            status="complete",
            model=request.model,
            extra={
                "tool_calls_made": self._state.tool_calls_made,
                "progress_pct": self._state.progress_pct,
                "termination_reason": termination_reason,
                "iterations_used": iteration,
                "iterations_remaining": self._state.iterations_remaining,
                "fanout_limit": self._state.fanout_limit,
            },
        )

    # ------------------------------------------------------------------
    # Mission strategy — structured promises + verified completion.
    # The chain stays alive until every promise verifies (or is
    # deterministically rejected), replacing the prose-driven ReAct loop.
    # ------------------------------------------------------------------

    async def _act_mission(
        self, request: InternalChatRequest, workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute the user's request as a verified mission.

        Flow:
        1. Plan a mission (``list[Promise]``) from the user message.
        2. For each promise, run a bounded inner loop of tool calls.
        3. Between promises, run the verification spec deterministically.
        4. On verify failure: retry with feedback; on exhaustion: reject.
        5. Terminate when all promises fulfill, or cascade failure.
        """
        # Plan the mission if we haven't yet (resumed sessions keep theirs).
        # When the planner fails we still want to give the agent a chance to
        # do something, but we stop pretending the result was "fulfilled" if
        # no actual work happened — see the post-runner gate below.
        fallback_triggered = False
        tool_calls_before = self._state.tool_calls_made
        if not self._state.mission:
            mission = await self._plan_mission(request, workspace_context)
            if not mission:
                fallback_triggered = True
                yield InternalStreamChunk(
                    content_delta=(
                        "\n[Mission planner returned nothing — falling back "
                        "to a single open-ended promise.]\n"
                    ),
                    model=request.model,
                    augmentum={
                        "mode": "coder", "phase": "planning",
                        "status": "planner_fallback",
                    },
                )
                user_msg = self._last_user_message(request) or "respond to user"
                mission = [Promise(
                    description=user_msg[:200],
                    verify=Verification.always(),
                )]
            self._state.mission = mission
        mission = self._state.mission

        yield self._meta_chunk(
            phase="executing", status="mission_started",
            model=request.model,
            extra={
                "strategy": "mission",
                "mission": [_promise_summary(p) for p in mission],
            },
        )
        # Render the mission log once so the user sees the whole plan upfront.
        yield InternalStreamChunk(
            content_delta=render_mission_log(mission) + "\n\n",
            model=request.model,
            augmentum={
                "mode": "coder", "phase": "executing", "status": "mission_log",
            },
        )

        # Build verifiers that run inside the workspace container.
        verify_fns = self._build_verify_fns()

        # Build the act_fn that drives one promise toward completion.
        act_fn = self._build_mission_act_fn(request, workspace_context)
        replan_fn = self._build_mission_replan_fn(request, workspace_context)

        runner = MissionRunner()
        fulfilled = 0
        rejected = 0
        try:
            async for ev in runner.run(
                mission, act_fn, verify_fns, replan_fn=replan_fn,
            ):
                if ev.kind == RunnerEventKind.PROMISE_FULFILLED:
                    fulfilled += 1
                elif ev.kind == RunnerEventKind.PROMISE_REJECTED:
                    rejected += 1
                async for chunk in self._runner_event_to_chunks(ev, request.model):
                    yield chunk
        except Exception as exc:  # defensive — runner itself shouldn't raise
            log.warning("coder.mission_runner_failed", error=str(exc), exc_info=True)
            yield InternalStreamChunk(
                content_delta=f"\n[Mission runner error: {exc}]\n",
                model=request.model,
                augmentum={"mode": "coder", "phase": "executing", "status": "error"},
            )

        # Honest-success gate: if the planner fallback fired AND no tools
        # actually ran during the mission, don't claim fulfilment. The
        # ``Verification.always()`` on the fallback promise would otherwise
        # auto-pass, making the agent lie about work it didn't do.
        tools_run = self._state.tool_calls_made - tool_calls_before
        effective_fulfilled = fulfilled
        effective_rejected = rejected
        if fallback_triggered and tools_run == 0:
            effective_fulfilled = 0
            effective_rejected = max(1, rejected)
            log.info(
                "coder.mission_fallback_unresolved",
                session_id=self._session_id,
                fulfilled=fulfilled,
                tools_run=tools_run,
            )
            yield InternalStreamChunk(
                content_delta=(
                    "\n[I couldn't turn that request into a concrete plan, "
                    "and no tools executed. Try rephrasing, or break the "
                    "task into smaller concrete actions.]\n"
                ),
                model=request.model,
                augmentum={
                    "mode": "coder", "phase": "executing",
                    "status": "fallback_unresolved",
                },
            )

        record_result(
            request.model, "mission",
            success=(effective_rejected == 0 and effective_fulfilled > 0),
        )

        self._state.phase = CoderPhase.WAITING
        yield self._meta_chunk(
            phase="waiting", status="complete", model=request.model,
            extra={
                "tool_calls_made": self._state.tool_calls_made,
                "fulfilled": effective_fulfilled,
                "rejected": effective_rejected,
                "fallback_triggered": fallback_triggered,
            },
        )

    # --- Mission helpers --------------------------------------------------

    async def _plan_mission(
        self, request: InternalChatRequest, workspace_context: str,
    ) -> list[Promise]:
        """Ask the model for a JSON mission. Returns [] on failure."""
        plan_system = MISSION_PLAN_SYSTEM
        if workspace_context:
            plan_system += f"\n\n{workspace_context}"

        plan_request = InternalChatRequest(
            model=request.model,
            messages=self._build_messages(request, plan_system),
            stream=True,
            temperature=0.0,  # deterministic structured output
        )
        parts: list[str] = []
        try:
            async for chunk in self._backend.chat_stream(plan_request):
                if chunk.content_delta:
                    parts.append(chunk.content_delta)
        except Exception as exc:  # noqa: BLE001
            log.warning("coder.mission_plan_stream_failed", error=str(exc))
            return []

        raw = "".join(parts)
        mission = parse_mission_json(raw)
        source = "json"
        if not mission:
            # Fallback: smaller models often emit numbered prose instead
            # of JSON. Extract a weakly-verified plan so the mission can
            # still run — each step will have an ``always`` verifier and
            # rely on the act layer for progress.
            mission = parse_prose_plan(raw)
            source = "prose" if mission else "empty"
        log.info(
            "coder.mission_planned",
            session_id=self._session_id,
            promises=len(mission),
            raw_len=len(raw),
            source=source,
        )
        return mission

    def _build_verify_fns(self) -> dict[VerificationKind, VerifyFn]:
        """Build verifier registry bound to this session's container.

        The shell/file verifiers run commands inside the workspace so
        that verify specs target paths the agent can actually reach.
        """
        workspace_id = self._workspace_id
        mgr = self._container_manager

        async def run_shell(cmd: str, timeout: float) -> tuple[int, str]:
            """Run *cmd* in the workspace and return (exit_code, output).

            ContainerManager._run_command only returns stdout, so we
            wrap the command to capture the exit code via a sentinel
            line and merge stderr into stdout.
            """
            if mgr is None:
                return 1, "no container manager available"
            sentinel = "__AUGMENTUM_EXIT__:"
            wrapped = f"({cmd}) 2>&1; printf '\\n{sentinel}%s\\n' \"$?\""
            try:
                output = await mgr._run_command(
                    workspace_id, ["bash", "-c", wrapped], timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                return 1, f"shell error: {exc}"
            # Parse trailing sentinel line
            lines = (output or "").splitlines()
            exit_code = 1
            body_lines = lines
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if line.startswith(sentinel):
                    try:
                        exit_code = int(line[len(sentinel):].strip())
                    except ValueError:
                        exit_code = 1
                    body_lines = lines[:i]
                    break
            return exit_code, "\n".join(body_lines)

        return default_verify_fns(run_shell)

    def _build_mission_replan_fn(
        self, request: InternalChatRequest, workspace_context: str,
    ):
        """Return a replan callback for ``MissionRunner``.

        After each top-level promise resolves, ask the model to redraft
        the remaining tail given the accumulated evidence. Returning
        ``None`` leaves the existing tail in place; returning a list
        replaces every still-pending top-level tail promise. An empty
        list ends the mission early.

        Replanning is gated on three conditions:
        - there is at least one still-pending tail promise to rewrite
        - OR the just-resolved promise was REJECTED (fresh strategy needed)
        - AND we haven't already replanned too many times (budget 3)
        """
        replan_budget = [3]  # mutable closure counter

        async def replan(
            mission: list[Promise], resolved: Promise,
        ) -> list[Promise] | None:
            # Find tail
            try:
                idx = mission.index(resolved)
            except ValueError:
                return None
            tail = mission[idx + 1 :]
            pending_tail = [p for p in tail if p.status == PromiseStatus.PENDING]
            if not pending_tail and resolved.status != PromiseStatus.REJECTED:
                return None
            if replan_budget[0] <= 0:
                return None
            replan_budget[0] -= 1

            # Render the mission so far for the planner: fulfilled +
            # rejected promises with their evidence, followed by a list
            # of still-pending descriptions for context.
            progress_lines: list[str] = []
            for p in mission[: idx + 1]:
                tag = {
                    PromiseStatus.FULFILLED: "DONE",
                    PromiseStatus.REJECTED: "FAILED",
                }.get(p.status, p.status.value.upper())
                ev = (p.evidence or "").replace("\n", " ")[:400]
                progress_lines.append(f"- [{tag}] {p.description} — {ev}")
            pending_lines = [
                f"- [PENDING] {p.description}" for p in pending_tail
            ]

            user_goal = self._last_user_message(request) or ""
            replan_context = (
                f"## Original user request\n{user_goal}\n\n"
                "## Progress so far\n" + "\n".join(progress_lines or ["(none)"]) + "\n\n"
                "## Remaining pending steps (replace these)\n"
                + "\n".join(pending_lines or ["(none)"])
            )

            plan_system = MISSION_REPLAN_SYSTEM
            if workspace_context:
                plan_system += f"\n\n{workspace_context}"

            plan_request = InternalChatRequest(
                model=request.model,
                messages=[
                    Message(role="system", content=plan_system),
                    Message(role="user", content=replan_context),
                ],
                stream=True,
                temperature=0.0,
            )
            parts: list[str] = []
            try:
                async for chunk in self._backend.chat_stream(plan_request):
                    if chunk.content_delta:
                        parts.append(chunk.content_delta)
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.warning("coder.mission_replan_stream_failed", error=str(exc))
                return None

            raw = "".join(parts).strip()
            # Empty string OR empty array both mean "finish the mission here"
            if not raw or raw == "[]":
                log.info(
                    "coder.mission_replan_empty",
                    after_id=resolved.id,
                    session_id=self._session_id,
                )
                return []
            new_tail = parse_mission_json(raw)
            if not new_tail:
                # Fall back: keep existing tail rather than replacing with empty
                log.info(
                    "coder.mission_replan_unparseable",
                    after_id=resolved.id,
                    raw_len=len(raw),
                    session_id=self._session_id,
                )
                return None
            log.info(
                "coder.mission_replanned",
                after_id=resolved.id,
                new_count=len(new_tail),
                session_id=self._session_id,
            )
            return new_tail

        return replan

    def _build_mission_act_fn(
        self, request: InternalChatRequest, workspace_context: str,
    ):
        """Build the act_fn that drives ONE promise per invocation.

        The act_fn runs a bounded inner loop (max 3 tool calls) in
        service of the current promise, streaming PROGRESS events for
        each tool call/result, and emits ATTEMPT_COMPLETE when the
        model stops emitting tool calls. Handles ``<decompose/>`` and
        ``<cannot_fulfill/>`` control tokens for richer flow control.
        """
        tools = create_coder_tools(
            self._container_manager, self._workspace_id, self._state,
        )
        tool_map = {t.name: t for t in tools}
        known_tools = set(tool_map.keys())
        tier = select_tier(self._backend, request.model)
        tool_schemas = (
            [_tool_to_schema(t) for t in tools]
            if tier == ToolCallingTier.NATIVE else None
        )

        base_system = f"{MISSION_ACT_SYSTEM}\n\n{EDIT_FORMAT_INSTRUCTIONS}"
        if workspace_context:
            base_system += f"\n\n{workspace_context}"

        max_tools_per_attempt = 3

        async def act(
            promise: Promise, ctx: PromiseContext,
        ) -> AsyncIterator[ActEvent]:
            # Per-attempt system: base + mission log (with live status) + focus block
            mission_log = render_mission_log(ctx.mission, header="## Mission Log")
            observations = _render_mission_observations(ctx.mission, promise)
            focus = (
                f"## Your Current Promise\n"
                f"Description: {promise.description}\n"
            )
            if promise.verify.kind == VerificationKind.SHELL:
                focus += (
                    f"Postcondition: shell `{promise.verify.spec.get('cmd', '')}`"
                    f" must exit 0.\n"
                )
            elif promise.verify.kind == VerificationKind.FILE:
                path = promise.verify.spec.get("path", "")
                must = promise.verify.spec.get("must_exist", True)
                state = "exist" if must else "be absent"
                focus += f"Postcondition: `{path}` must {state}.\n"
            elif promise.verify.kind == VerificationKind.ANY_OF:
                checks = promise.verify.spec.get("checks") or []
                focus += (
                    f"Postcondition: ANY of these {len(checks)} checks may "
                    f"pass. You do not need to match a specific path — "
                    f"achieving any listed outcome counts as success.\n"
                )
            if promise.attempts > 0 and promise.evidence:
                prior_fps = promise.attempt_fingerprints or []
                prior_tools_hint = ""
                if prior_fps:
                    prior_tools_hint = (
                        f"\nYou already tried: "
                        f"{', '.join(prior_fps[-3:])}. Do NOT repeat these.\n"
                    )
                focus += (
                    f"\n### Retry guidance (attempt {promise.attempts + 1}/{promise.max_attempts})\n"
                    f"Previous attempt failed: {promise.evidence[:400]}\n"
                    f"{prior_tools_hint}"
                    f"You MUST choose a substantially different approach — "
                    f"different tool, different command, different assumption, "
                    f"or a smaller scope. Before acting, consider: did the "
                    f"last attempt fail because a tool was missing, a path "
                    f"was wrong, or an assumption was invalid? Address the "
                    f"root cause, do not just retry.\n"
                )

            sys_parts = [base_system, mission_log]
            if observations:
                sys_parts.append(observations)
            sys_parts.append(focus)
            sys_prompt = "\n\n".join(sys_parts)
            messages = self._build_messages(request, sys_prompt)
            last_output = ""
            fingerprints_this_attempt: list[str] = []

            for _inner in range(max_tools_per_attempt):
                attempt_req = InternalChatRequest(
                    model=request.model,
                    messages=list(messages),
                    stream=True,
                    temperature=request.temperature,
                    tools=tool_schemas,
                )

                content_parts: list[str] = []
                tc_acc: dict[int, dict] = {}
                stream_exc: Exception | None = None

                # Retry the stream on transient backend errors (429 / 503 /
                # network hiccups). Backoff: 5s, 15s, 45s. On final
                # failure — or any non-transient error — surface via
                # CANNOT_FULFILL so the mission reports honestly.
                retry_schedule = [5.0, 15.0, 45.0]
                for retry_idx in range(len(retry_schedule) + 1):
                    content_parts = []
                    tc_acc = {}
                    stream_exc = None
                    try:
                        async for chunk in self._backend.chat_stream(attempt_req):
                            if chunk.content_delta:
                                content_parts.append(chunk.content_delta)
                            if chunk.augmentum and "tool_calls" in chunk.augmentum:
                                for delta in chunk.augmentum["tool_calls"]:
                                    idx = delta.get("index", 0)
                                    slot = tc_acc.setdefault(idx, {
                                        "id": delta.get("id", str(uuid.uuid4())),
                                        "name": "",
                                        "arguments_parts": [],
                                    })
                                    if delta.get("id"):
                                        slot["id"] = delta["id"]
                                    fn = delta.get("function", {})
                                    if fn.get("name"):
                                        slot["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        slot["arguments_parts"].append(fn["arguments"])
                        break  # stream succeeded
                    except Exception as exc:  # noqa: BLE001
                        stream_exc = exc
                        if not _is_transient_backend_error(exc):
                            break
                        if retry_idx >= len(retry_schedule):
                            break  # retries exhausted
                        wait = retry_schedule[retry_idx]
                        log.warning(
                            "coder.mission_rate_limited_retry",
                            promise_id=promise.id,
                            attempt=retry_idx + 1,
                            wait=wait,
                            error=str(exc)[:200],
                        )
                        yield ActEvent(
                            kind=ActEventKind.PROGRESS,
                            payload={
                                "type": "rate_limited",
                                "wait_seconds": wait,
                                "attempt": retry_idx + 1,
                                "max_retries": len(retry_schedule),
                                "reason": _short_error_reason(exc),
                            },
                        )
                        await asyncio.sleep(wait)

                if stream_exc is not None:
                    log.warning(
                        "coder.mission_stream_failed",
                        error=str(stream_exc), promise_id=promise.id,
                    )
                    yield ActEvent(
                        kind=ActEventKind.CANNOT_FULFILL,
                        payload=f"backend stream error: {stream_exc}",
                    )
                    return

                full_content = "".join(content_parts)

                # --- Control tokens ---------------------------------
                if "<cannot_fulfill/>" in full_content:
                    reason = full_content.split("<cannot_fulfill/>", 1)[1].strip()
                    yield ActEvent(
                        kind=ActEventKind.CANNOT_FULFILL,
                        payload=reason[:300] or "model declared inability",
                    )
                    return

                if "<decompose/>" in full_content:
                    after = full_content.split("<decompose/>", 1)[1]
                    children = parse_mission_json(after)
                    if children:
                        yield ActEvent(
                            kind=ActEventKind.NEEDS_DECOMPOSITION,
                            payload=children,
                        )
                        return
                    log.warning(
                        "coder.mission_decompose_parse_failed",
                        content=full_content[:200],
                    )
                    # Fall through to normal handling if decomposition unparseable

                # --- Assemble tool calls ----------------------------
                assembled = []
                for idx in sorted(tc_acc):
                    slot = tc_acc[idx]
                    if slot["name"]:
                        args_str = "".join(slot["arguments_parts"])
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {}
                        assembled.append({
                            "id": slot["id"], "name": slot["name"], "input": args,
                        })

                response_tool_calls = assembled
                if not response_tool_calls and full_content:
                    from augmentum.modes.analytical.tool_calling import parse_json_tool_calls
                    parsed = parse_json_tool_calls(full_content, known_tools=known_tools)
                    if parsed:
                        response_tool_calls = [
                            {"id": str(uuid.uuid4()), "name": n, "input": a}
                            for n, a in parsed
                        ]
                    else:
                        response_tool_calls = _extract_tool_calls_from_text(full_content)

                # Append assistant message for next inner iteration
                if tier == ToolCallingTier.NATIVE:
                    formatted_tc = None
                    if response_tool_calls:
                        formatted_tc = [{
                            "id": tc.get("id", str(uuid.uuid4())),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {})),
                            },
                        } for tc in response_tool_calls]
                    messages.append(Message(
                        role="assistant",
                        content=full_content or "",
                        tool_calls=formatted_tc,
                    ))
                else:
                    clean = _strip_tool_json(full_content).strip() or "(executing)"
                    messages.append(Message(role="assistant", content=clean))

                if not response_tool_calls:
                    # Model believes the current promise is done.
                    break

                # Execute the first tool call (one per inner iteration).
                tc = response_tool_calls[0]
                tool_id = tc.get("id") or str(uuid.uuid4())
                tool_name = tc.get("name") or ""
                raw_in = tc.get("input") or {}
                if isinstance(raw_in, str):
                    try:
                        tool_input = json.loads(raw_in)
                    except json.JSONDecodeError:
                        tool_input = {}
                else:
                    tool_input = raw_in

                # Fingerprint the tool call so we can detect "retry with
                # the same exact command" and force a strategy change
                # instead of burning attempts.
                fp = _tool_fingerprint(tool_name, tool_input)
                if (
                    promise.attempts > 0
                    and fp in (promise.attempt_fingerprints or [])
                ):
                    log.info(
                        "coder.mission_retry_identical_tool_call",
                        promise_id=promise.id, tool=tool_name,
                    )
                    yield ActEvent(
                        kind=ActEventKind.CANNOT_FULFILL,
                        payload=(
                            f"model retried identical tool call ({tool_name}) — "
                            f"strategy change required, triggering replan"
                        ),
                    )
                    return
                fingerprints_this_attempt.append(fp)

                yield ActEvent(
                    kind=ActEventKind.PROGRESS,
                    payload={
                        "type": "tool_call", "id": tool_id,
                        "tool": tool_name, "input": tool_input,
                    },
                )

                tool_result = await _execute_tool(
                    tool_map=tool_map,
                    tool_name=tool_name, tool_input=tool_input,
                )
                self._state.tool_calls_made += 1

                # LSP lint feedback on edits (same as legacy path)
                if tool_result.success and tool_name in ("code_edit", "file_write"):
                    lint_output = await self._run_lint_check(
                        tool_input.get("path", ""),
                    )
                    if lint_output:
                        from augmentum.tools.base import ToolResult as _TR
                        tool_result = _TR(
                            success=tool_result.success,
                            output=(tool_result.output or "") + f"\n\n[Lint check]\n{lint_output}",
                            metadata=tool_result.metadata,
                        )

                # Auto-checkpoint on successful file modifications
                checkpoint_hash = None
                if (
                    tool_result.success
                    and tool_name in ("code_edit", "file_write")
                    and self._container_manager is not None
                ):
                    path = tool_input.get("path", "unknown")
                    short = path.replace("/workspace/", "")
                    checkpoint_hash = await self._container_manager.git_checkpoint(
                        self._workspace_id, f"Agent: {tool_name} {short}",
                    )

                result_text = (
                    tool_result.output if tool_result.success
                    else f"ERROR: {tool_result.error}"
                )
                last_output = result_text

                progress_payload: dict = {
                    "type": "tool_result", "id": tool_id, "tool": tool_name,
                    "success": tool_result.success,
                    "output_preview": (result_text or "")[:_preview_len(tool_name)],
                }
                if checkpoint_hash:
                    progress_payload["checkpoint"] = checkpoint_hash
                yield ActEvent(
                    kind=ActEventKind.PROGRESS, payload=progress_payload,
                )

                # Feed result back into history for the next inner iteration.
                if tier == ToolCallingTier.NATIVE:
                    messages.append(Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=tool_id,
                    ))
                else:
                    feedback = (
                        f"[Tool result: {tool_name}]\n{result_text}\n\n"
                        f"Continue with the next tool call toward the "
                        f"current promise, or stop if it is now done."
                    )
                    if messages and messages[-1].role == "user":
                        messages[-1] = Message(
                            role="user",
                            content=messages[-1].content + "\n\n" + feedback,
                        )
                    else:
                        messages.append(Message(role="user", content=feedback))

            # Record every tool call this attempt made so the next
            # attempt (if any) can check for identical-retry loops.
            if fingerprints_this_attempt:
                promise.attempt_fingerprints = (
                    (promise.attempt_fingerprints or [])
                    + fingerprints_this_attempt
                )

            # Inner loop done — signal the runner to verify.
            yield ActEvent(
                kind=ActEventKind.ATTEMPT_COMPLETE,
                evidence=last_output.strip()[:2000] or "no tool output",
            )

        return act

    async def _runner_event_to_chunks(
        self, ev: RunnerEvent, model: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Translate a ``RunnerEvent`` into one or more UI chunks."""
        kind = ev.kind
        if kind == RunnerEventKind.MISSION_STARTED:
            return  # already emitted by _act_mission
        if kind == RunnerEventKind.PROMISE_STARTED:
            yield self._meta_chunk(
                phase="executing", status="promise_started", model=model,
                extra={"promise": _promise_summary(ev.promise)},
            )
            return
        if kind == RunnerEventKind.PROMISE_PROGRESS:
            payload = ev.payload or {}
            ptype = payload.get("type")
            if ptype == "tool_call":
                yield self._meta_chunk(
                    phase="executing", status="tool_call", model=model,
                    extra={"tool_call": {k: v for k, v in payload.items() if k != "type"}},
                )
            elif ptype == "tool_result":
                extra: dict = {
                    "tool_result": {
                        k: v for k, v in payload.items()
                        if k not in ("type", "checkpoint")
                    },
                }
                if "checkpoint" in payload:
                    extra["tool_result"]["checkpoint"] = payload["checkpoint"]
                yield self._meta_chunk(
                    phase="executing", status="tool_result", model=model,
                    extra=extra,
                )
            elif ptype == "rate_limited":
                yield self._meta_chunk(
                    phase="executing", status="rate_limited", model=model,
                    extra={
                        "promise": _promise_summary(ev.promise),
                        "wait_seconds": payload.get("wait_seconds", 0),
                        "attempt": payload.get("attempt", 1),
                        "max_retries": payload.get("max_retries", 3),
                        "reason": payload.get("reason", ""),
                    },
                )
            return
        if kind == RunnerEventKind.PROMISE_VERIFYING:
            yield self._meta_chunk(
                phase="executing", status="promise_verifying", model=model,
                extra={"promise": _promise_summary(ev.promise)},
            )
            return
        if kind == RunnerEventKind.PROMISE_FULFILLED:
            yield self._meta_chunk(
                phase="executing", status="promise_fulfilled", model=model,
                extra={"promise": _promise_summary(ev.promise)},
            )
            yield InternalStreamChunk(
                content_delta=f"[x] {ev.promise.description}\n",
                model=model,
                augmentum={"mode": "coder", "phase": "executing", "status": "streaming"},
            )
            return
        if kind == RunnerEventKind.PROMISE_RETRY:
            reason = (ev.payload or {}).get("reason", "")
            yield self._meta_chunk(
                phase="executing", status="promise_retry", model=model,
                extra={
                    "promise": _promise_summary(ev.promise),
                    "reason": reason[:200],
                },
            )
            return
        if kind == RunnerEventKind.PROMISE_REJECTED:
            reason = (ev.payload or {}).get("reason", "")
            yield self._meta_chunk(
                phase="executing", status="promise_rejected", model=model,
                extra={
                    "promise": _promise_summary(ev.promise),
                    "reason": reason[:200],
                },
            )
            yield InternalStreamChunk(
                content_delta=(
                    f"[!] {ev.promise.description} — "
                    f"{(ev.promise.evidence or reason or 'failed')[:160]}\n"
                ),
                model=model,
                augmentum={"mode": "coder", "phase": "executing", "status": "streaming"},
            )
            return
        if kind == RunnerEventKind.PROMISE_DECOMPOSED:
            yield self._meta_chunk(
                phase="executing", status="promise_decomposed", model=model,
                extra={
                    "promise": _promise_summary(ev.promise),
                    "children": [
                        _promise_summary(c)
                        for c in (ev.promise.children if ev.promise else [])
                    ],
                },
            )
            return
        if kind == RunnerEventKind.MISSION_REPLANNED:
            payload = ev.payload or {}
            new_tail = payload.get("new_tail") or []
            yield self._meta_chunk(
                phase="executing", status="mission_replanned", model=model,
                extra={
                    "after_id": payload.get("after_id"),
                    "new_tail": new_tail,
                },
            )
            # Show the user what changed so the UI panel stays coherent.
            if new_tail:
                tail_preview = "\n".join(f"  - {t}" for t in new_tail[:6])
                body = (
                    "\n[Replanned remaining steps based on observed "
                    f"evidence:\n{tail_preview}]\n"
                )
            else:
                body = (
                    "\n[Replanned: remaining steps are no longer needed given "
                    "the observed evidence — finishing mission.]\n"
                )
            yield InternalStreamChunk(
                content_delta=body,
                model=model,
                augmentum={
                    "mode": "coder", "phase": "executing",
                    "status": "streaming",
                },
            )
            return
        if kind == RunnerEventKind.MISSION_COMPLETED:
            yield self._meta_chunk(
                phase="executing", status="mission_completed", model=model,
                extra=dict(ev.payload or {}),
            )
            return
        if kind == RunnerEventKind.MISSION_FAILED:
            yield self._meta_chunk(
                phase="executing", status="mission_failed", model=model,
                extra=dict(ev.payload or {}),
            )
            return
