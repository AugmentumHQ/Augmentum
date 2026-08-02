"""Flow executor — runs a reasoning flow's steps in sequence.

This module provides ``execute_flow_stream()`` which replaces the hardcoded
phase logic in the analytical handler when a custom flow is active. It:

1. Filters steps by complexity gate
2. Executes each step with variable substitution
3. Handles role-based behavior (classify, search, verify, respond)
4. Streams phase metadata and content to the UI
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
    Message,
    ModelBackend,
)
from augmentum.modes.analytical.engine import (
    _CONFIDENCE_THRESHOLD,
    AnalyticalEngine,
)
from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    build_structured_output_schema,
    coerce_tool_params,
    extract_structured_text,
    parse_native_tool_call,
    parse_python_style_tool_call,
    parse_structured_output,
    select_tier,
    tools_to_native_format,
)
from augmentum.reasoning.models import FlowStep, ReasoningFlow
from augmentum.reasoning.variables import (
    FLOW_STEP_USER_TEMPLATE,
    StepContext,
    build_user_message,
    resolve_variables,
)
from augmentum.tools.base import invoke_tool
from augmentum.utils.datetime_context import get_datetime_context
from augmentum.utils.logging import get_logger

# Hard ceiling on what a single step's output may contribute to the recorded
# context (chars). The per-step bulk-concat budget for {all_outputs} lives in
# StepContext; this is just a runaway-generation guard so one pathological step
# can't blow the next step's prompt past the model's context window.
_MAX_RECORDED_STEP_OUTPUT = 20_000

# How many chars of raw search context to staple onto a respond/deliver step
# that didn't already pull {search_results} in via its own template.
_RESPOND_SOURCES_CHARS = 8_000

# Appended to the system prompt for every respond/deliver step. Drowns out
# RLHF-trained disclaimer patterns ("I don't have real-time info", "as an AI",
# "based on the research notes", "please consult a professional") that the
# model emits even when the carried-forward chain and stapled <sources> make
# the disclaimer factually wrong for this turn. Can be bypassed per-request
# via flow_tune={"voice_guard": false} (used by the A/B harness).
_RESPOND_VOICE_GUARD = (
    "\n\n## Voice\n"
    "You are presenting findings to the user. The current date is stated "
    "above and any sources gathered this turn are in your context — speak "
    "FROM them, not ABOUT them.\n\n"
    "Never use these patterns:\n"
    "- \"I don't have access to real-time information\" / \"my knowledge "
    "cutoff\" / \"as of my last training\" / \"I cannot verify directly\" "
    "— the date is given and any sources gathered are present\n"
    "- \"As an AI...\" / \"As a language model...\" / \"I am an AI\" — "
    "don't narrate your nature\n"
    "- \"Based on the research notes\" / \"the analysis shows\" / "
    "\"according to the work notes\" / \"the information provided to me\" "
    "— the user doesn't see the pipeline; state the facts directly\n"
    "- \"You should verify this with...\" / \"please consult official "
    "sources\" / \"for legal/medical/financial purposes consult a "
    "professional\" — when sources support a claim, state it; when they "
    "don't, say sources disagree or it's not yet confirmed\n\n"
    "Cite gathered sources inline with [1], [2]. If sources genuinely "
    "conflict, say so plainly (\"reports differ on X\"). If something is "
    "outside what the sources cover and you don't know, say \"I'm not "
    "sure\" directly — never disclaim as an AI.\n"
)

if TYPE_CHECKING:
    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)


async def _resolve_step_backend(
    step: FlowStep,
    default_backend: ModelBackend,
    default_model: str,
    provider_registry: ProviderRegistry | None,
) -> tuple[ModelBackend, str]:
    """Resolve the backend and model for a step, respecting model_override."""
    if not step.model_override:
        return default_backend, default_model

    override = step.model_override

    # Special "verify" keyword — use cross-model verification config
    if override == "verify":
        from augmentum.config import settings as cfg

        override = cfg.uarf_verify_model
        if not override:
            return default_backend, default_model

    # Resolve via provider registry
    if provider_registry:
        try:
            resolved_backend, resolved_model = await provider_registry.resolve_backend_with_fabric(override)
            return resolved_backend, resolved_model
        except Exception:
            log.warning("step_model_override_failed", model=override, exc_info=True)

    return default_backend, default_model


# ---------------------------------------------------------------------------
# Stream chunk helpers (mirrors handler.py helpers)
# ---------------------------------------------------------------------------


def _flow_phase_chunk(
    model: str,
    step_name: str,
    status: str,
    pipeline: list[str],
    *,
    confidence: float | None = None,
    complexity: str = "",
    content: str = "",
    tool_calls: list[dict] | None = None,
    flow_name: str = "",
    step_model: str = "",
    default_model: str = "",
) -> InternalStreamChunk:
    """Build a stream chunk carrying flow step metadata."""
    phases_list = []
    step_idx = -1
    for i, p in enumerate(pipeline):
        if p == step_name:
            step_idx = i
            break

    for i, p in enumerate(pipeline):
        if p == step_name:
            phases_list.append({"name": p, "status": status})
        elif i < step_idx:
            phases_list.append({"name": p, "status": "complete"})
        else:
            phases_list.append({"name": p, "status": "pending"})

    meta: dict = {
        "mode": "analytical",
        "phase": step_name,
        "phase_status": status,
        "phases": phases_list,
    }
    if flow_name:
        meta["flow_name"] = flow_name
    if complexity:
        meta["complexity"] = complexity
    if confidence is not None:
        meta["confidence"] = confidence
    if tool_calls:
        meta["tool_calls"] = tool_calls
    if step_model and default_model and step_model != default_model:
        meta["step_model"] = step_model

    return InternalStreamChunk(
        content_delta=content,
        model=model,
        augmentum=meta,
    )


def _flow_content_chunk(
    model: str,
    step_name: str,
    complexity: str,
    content_delta: str,
) -> InternalStreamChunk:
    """Build a lightweight chunk carrying step content delta."""
    meta: dict = {
        "mode": "analytical",
        "phase": step_name,
        "phase_content_delta": content_delta,
    }
    if complexity:
        meta["complexity"] = complexity
    return InternalStreamChunk(
        content_delta="",
        model=model,
        augmentum=meta,
    )


# ---------------------------------------------------------------------------
# Step filtering
# ---------------------------------------------------------------------------


def filter_steps_by_complexity(
    steps: list[FlowStep], complexity: str,
) -> list[FlowStep]:
    """Filter flow steps based on complexity gating.

    Steps with empty ``complexity_gate`` always run.
    Steps with a gate only run if the detected complexity is in the gate list.
    """
    result = []
    for step in steps:
        if not step.enabled:
            continue
        if not step.complexity_gate or complexity in step.complexity_gate:
            result.append(step)
    return result


# ---------------------------------------------------------------------------
# Tool resolution for a step
# ---------------------------------------------------------------------------


# Marks a step output that is an ERROR STRING, not analysis. These used to
# be returned bare, recorded via record_step and yielded with phase_status
# "complete" — so a step that never ran was indistinguishable from one that
# succeeded, and "LLM call timed out after 120.0s" was fed to the next step
# as if it were the analysis. Failure must not launder into success.
_STEP_ERROR_PREFIX = "[[STEP_ERROR]] "


def _step_failed(output: str) -> bool:
    return output.startswith(_STEP_ERROR_PREFIX)


def _strip_step_error(output: str) -> str:
    return output.removeprefix(_STEP_ERROR_PREFIX)


# Roles that benefit from memory context injection.
_MEMORY_RECALL_ROLES = frozenset({"classify", "plan", "search", "respond", "deliver"})

# Tools the flow machinery adds to a step on its own. They must never be
# mistaken for the user having pinned tools to that step: ``memory_recall``
# is auto-injected into every search-role step (templates.py::_step and
# _resolve_tools_for_step), which silently made the auto-search branch's
# ``not step.tool_names`` guard permanently false and disabled web search
# in EVERY analytical flow. Anything auto-injected belongs here.
_AUTO_INJECTED_TOOLS = frozenset({"memory_recall"})


def _pinned_tools(step: FlowStep) -> list[str]:
    """Tools explicitly pinned to ``step`` by its author, minus auto-injected."""
    return [n for n in (step.tool_names or []) if n not in _AUTO_INJECTED_TOOLS]

# Pass-through sentinel values for ``FlowStep.tool_choice`` — forwarded as
# strings; the OpenAI-compat / llama-server / Anthropic adapters all accept
# the OpenAI vocabulary directly (and ``anthropic_compat._translate_tool_choice``
# rewrites it to Anthropic's ``{"type": "any"/"auto"/"none"}`` form on the way
# out).
_TOOL_CHOICE_PASSTHROUGH = frozenset({"auto", "required", "none"})


def _translate_step_tool_choice(
    step: FlowStep, tools: list,
) -> str | dict | None:
    """Translate the step's ``tool_choice`` field into the value to pass
    on ``InternalChatRequest.tool_choice``.

    Returns:
      - ``None`` when no override applies (empty string, or no tools
        resolved — the provider would 400 on tool_choice without a tools
        list).
      - ``"auto"`` / ``"required"`` / ``"none"`` straight through.
      - A pinned-tool dict (OpenAI shape) when the value matches a name
        in the resolved tool set: ``{"type": "function", "function":
        {"name": <name>}}``. The Anthropic adapter rewrites this to its
        own shape via ``_translate_tool_choice``.
      - ``None`` when the value names a tool that isn't in the resolved
        set for this step — we silently drop rather than 400 the call,
        and a warning is logged for visibility.
    """
    raw = (step.tool_choice or "").strip()
    if not raw:
        return None
    if not tools:
        # No tools resolved — sending tool_choice would 400 on most
        # providers ("tool_choice with no tools"). Drop quietly so a
        # template authored for the tool-eligible path still runs when
        # tune disables every tool.
        return None
    if raw in _TOOL_CHOICE_PASSTHROUGH:
        return raw
    # Specific tool name. Validate against the resolved set so a typo
    # doesn't become a runtime 400 mid-flow.
    tool_names = {t.name for t in tools}
    if raw not in tool_names:
        log.warning(
            "step_tool_choice_unknown_tool",
            step=step.name,
            tool=raw,
            available=sorted(tool_names),
        )
        return None
    return {"type": "function", "function": {"name": raw}}


def _resolve_tools_for_step(
    step: FlowStep,
    tool_registry: ToolRegistry | None,
    *,
    exclude: frozenset[str] | None = None,
) -> list:
    """Resolve available tools for a step based on its config.

    Automatically injects memory_recall for roles that benefit from user
    context (classify, plan, search, respond, deliver), matching the
    behaviour of builtin templates.
    """
    if not tool_registry:
        return []

    tools = []
    seen = set()

    # Specific tool names take priority
    if step.tool_names:
        for name in step.tool_names:
            tool = tool_registry.get(name)
            if tool and tool.name not in seen:
                if exclude and tool.name in exclude:
                    continue
                tools.append(tool)
                seen.add(tool.name)

    # Tool categories add additional tools. Category expansion respects
    # SurfaceExposure.flow — conversational action verbs (note.create,
    # media.play, schedulers…) share categories with real capabilities
    # but must never auto-enter a flow step's schema. Explicit
    # ``tool_names`` pins above bypass this on purpose: the flow author
    # asked for that exact tool.
    if step.tool_categories:
        for category in step.tool_categories:
            cat_tools = tool_registry.get_for_phase(category, exclude=exclude)
            for tool in cat_tools:
                if not getattr(getattr(tool, "surfaces", None), "flow", True):
                    continue
                if tool.name not in seen:
                    tools.append(tool)
                    seen.add(tool.name)

    # Auto-inject memory_recall for appropriate roles so custom flows get
    # the same memory context that builtins do.
    if (
        step.role in _MEMORY_RECALL_ROLES
        and "memory_recall" not in seen
    ):
        recall_tool = tool_registry.get("memory_recall")
        if recall_tool:
            tools.append(recall_tool)

    return tools


# ---------------------------------------------------------------------------
# Main flow executor
# ---------------------------------------------------------------------------


async def execute_flow_stream(
    flow: ReasoningFlow,
    engine: AnalyticalEngine,
    backend: ModelBackend,
    model: str,
    query: str,
    *,
    tool_registry: ToolRegistry | None = None,
    provider_registry: ProviderRegistry | None = None,
    conversation_context: str = "",
    search_context: str = "",
    user_system: str = "",
    flow_tune: dict | None = None,
) -> AsyncIterator[InternalStreamChunk]:
    """Execute a reasoning flow, streaming step results.

    This is the flow-based alternative to the hardcoded phase orchestration
    in ``AnalyticalHandler.handle_stream()``.
    """
    from augmentum.modes.analytical.prompts import (
        get_native_tool_prompt_section,
        get_structured_tool_prompt_section,
        get_tool_prompt_section,
    )

    ctx = StepContext(query=query, model=model)
    ctx.conversation = conversation_context
    ctx.search_results = search_context
    _fn = flow.name  # flow name for stream metadata

    # First pass: check if there's a classify step to determine complexity
    classify_step = next((s for s in flow.steps if s.role == "classify" and s.enabled), None)

    # Complexity hint from client header — skip classify step entirely
    _valid_complexities = {"simple", "moderate", "complex"}
    complexity_hint = (flow_tune or {}).get("complexity") if flow_tune else None
    if isinstance(complexity_hint, str) and complexity_hint.lower() in _valid_complexities:
        complexity_hint = complexity_hint.lower()
    else:
        complexity_hint = None

    if complexity_hint and classify_step:
        # Client provided a complexity hint — skip the classify LLM call
        ctx.complexity = complexity_hint
        engine._state.complexity = complexity_hint
        log.info("complexity_hint_applied", hint=complexity_hint, skipped_step=classify_step.name)
        active_steps = filter_steps_by_complexity(
            [s for s in flow.steps if s.id != classify_step.id],
            ctx.complexity,
        )
        pipeline_names = [s.name for s in active_steps]
    elif classify_step:
        # Execute classify step first to get complexity
        pipeline_names = [s.name for s in flow.steps if s.enabled]
        yield _flow_phase_chunk(model, classify_step.name, "running", pipeline_names, flow_name=flow.name)
        yield InternalStreamChunk(content_delta="", role="assistant", model=model)

        # Try heuristic first
        heuristic = (
            AnalyticalEngine._heuristic_assess(query)
            if settings.uarf_heuristic_assess else None
        )

        if heuristic is not None:
            classify_output = (
                f"TYPE: heuristic\nDOMAIN: general\nREASONING_STEPS: 1\n"
                f"COMPLEXITY: {heuristic}\nRATIONALE: Heuristic classification."
            )
            ctx.complexity = heuristic
            yield _flow_content_chunk(
                model, classify_step.name, "",
                f"Heuristic: {heuristic} complexity\n",
            )
        else:
            # LLM classify — resolve per-step model override
            cls_backend, cls_model = await _resolve_step_backend(
                classify_step, backend, model, provider_registry,
            )
            system_prompt = resolve_variables(classify_step.system_prompt, ctx)
            system_prompt = f"{get_datetime_context()}\n\n{system_prompt}"
            if user_system:
                system_prompt = f"{user_system}\n\n{system_prompt}"
            user_msg = build_user_message(classify_step.user_template, ctx)

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_msg),
            ]

            classify_output = ""
            req = InternalChatRequest(model=cls_model, messages=messages, stream=True)
            async for chunk in cls_backend.chat_stream(req):
                if chunk.content_delta:
                    classify_output += chunk.content_delta
                    yield _flow_content_chunk(
                        model, classify_step.name, "", chunk.content_delta,
                    )

            ctx.complexity = AnalyticalEngine._parse_complexity(classify_output)

        ctx.record_step(classify_step.name, classify_output)
        engine._state.complexity = ctx.complexity

        # Now filter remaining steps by complexity
        active_steps = filter_steps_by_complexity(
            [s for s in flow.steps if s.id != classify_step.id],
            ctx.complexity,
        )
        pipeline_names = [classify_step.name] + [s.name for s in active_steps]

        yield _flow_phase_chunk(
            model, classify_step.name, "complete", pipeline_names,
            complexity=ctx.complexity, flow_name=_fn,
        )
    else:
        ctx.complexity = "moderate"
        active_steps = [s for s in flow.steps if s.enabled]
        pipeline_names = [s.name for s in active_steps]

    # Apply per-message tune overrides (Quick Tune panel)
    _tune_skip: frozenset[int] = frozenset()
    _tune_disable_tools: frozenset[str] = frozenset()
    if flow_tune and isinstance(flow_tune, dict):
        skip_raw = flow_tune.get("skip_steps")
        if isinstance(skip_raw, list):
            _tune_skip = frozenset(int(x) for x in skip_raw if isinstance(x, (int, float)))
            if _tune_skip:
                active_steps = [s for s in active_steps if s.sort_order not in _tune_skip]
                pipeline_names = [s.name for s in active_steps]
                log.info("flow_tune_skip_steps", skipped=list(_tune_skip))
        dt_raw = flow_tune.get("disable_tools")
        if isinstance(dt_raw, list):
            _tune_disable_tools = frozenset(str(t) for t in dt_raw if isinstance(t, str))
            if _tune_disable_tools:
                log.info("flow_tune_disable_tools", disabled=list(_tune_disable_tools))

    # Guard: at least one step must remain after tune filtering
    if not active_steps:
        log.warning("flow_tune_all_steps_skipped", flow=flow.name)
        active_steps = [s for s in flow.steps if s.enabled][:1]
        pipeline_names = [s.name for s in active_steps]

    # Check if any active step will stream to the user
    has_respond_step = False

    # Detect if auto-search is needed (for search role steps)
    needs_search = (
        flow.auto_search
        and tool_registry is not None
        and tool_registry.get("web_search") is not None
        and AnalyticalEngine._needs_search(query, ctx.get_step_output(classify_step.name) if classify_step else "")
    )
    if needs_search:
        engine._state.needs_search = True

    tool_exclude = frozenset({"web_search"}) if needs_search else None
    tc_cursor = len(engine.state.tool_calls)

    # Execute remaining steps in order
    for step in active_steps:
        step_name = step.name
        if step.stream_to_user:
            has_respond_step = True

        if step.role == "search" and needs_search and not _pinned_tools(step):
            # Auto-search step — use existing search machinery.
            # Skipped when the user pinned specific tool_names on the step
            # (e.g. only "image_search"), so explicit tagging routes through
            # the standard tool loop instead of the hardcoded web_search path.
            yield _flow_phase_chunk(
                model, step_name, "running", pipeline_names,
                complexity=ctx.complexity, flow_name=_fn,
            )

            # Direct URL fetch — if the query contains URLs, fetch them
            # directly via the web_fetch tool instead of searching *about* them.
            search_query = query
            if settings.search_direct_fetch_enabled:
                from augmentum.modes.analytical.state import ToolCallRecord
                from augmentum.tools.web import _extract_urls

                fetch_tool = tool_registry.get("web_fetch") if tool_registry else None
                urls = _extract_urls(query)
                if urls and fetch_tool:
                    url_list = "\n".join(f"  → {u}" for u in urls)
                    yield _flow_content_chunk(
                        model, step_name, ctx.complexity,
                        f"Fetching {len(urls)} URL(s) directly:\n{url_list}\n",
                    )
                    fetched_parts = []
                    for url in urls[:3]:  # cap at 3 URLs
                        result = await invoke_tool(fetch_tool, {
                            "url": url,
                            "max_chars": settings.search_direct_fetch_max_chars,
                            "_user_id": engine._user_id,
                        })
                        engine._state.tool_calls.append(ToolCallRecord(
                            phase="search",
                            tool_name="web_fetch",
                            input_data={"url": url},
                            output=(result.output or result.error or "")[:500],
                            success=result.success,
                        ))
                        if result.success and result.output:
                            fetched_parts.append(f"## {url}\n{result.output}")
                    if fetched_parts:
                        fetched_context = "\n\n".join(fetched_parts)
                        yield _flow_content_chunk(
                            model, step_name, ctx.complexity,
                            f"Fetched {len(fetched_parts)}/{len(urls)} page(s).\n",
                        )
                        engine._state.search_context = fetched_context
                        engine._state.search_result_count += len(fetched_parts)

                    # Strip URLs from query for search generation
                    import re
                    search_query = re.sub(r"https?://\S+", "", query).strip()
                    if not search_query or len(search_query) < 5:
                        # URL-only query — skip normal search
                        ctx.search_results = engine._state.search_context
                        ctx.record_step(step_name, f"Direct fetch complete: {len(urls)} URL(s).")
                        new_calls = [
                            {
                                "phase": tc.phase, "tool": tc.tool_name,
                                "input": tc.input_data,
                                "output": (tc.output or "")[:500],
                                "success": tc.success,
                            }
                            for tc in engine.state.tool_calls[tc_cursor:]
                        ]
                        tc_cursor = len(engine.state.tool_calls)
                        yield _flow_phase_chunk(
                            model, step_name, "complete", pipeline_names,
                            complexity=ctx.complexity, tool_calls=new_calls or None, flow_name=_fn,
                        )
                        continue

            yield _flow_content_chunk(
                model, step_name, ctx.complexity, "Generating search queries...\n",
            )

            queries = await engine._generate_search_queries(
                model, search_query, num_queries=settings.uarf_auto_search_queries,
                conversation_context=ctx.conversation,
            )
            queries_text = "\n".join(f"  - {q}" for q in queries)
            yield _flow_content_chunk(
                model, step_name, ctx.complexity,
                f"Queries:\n{queries_text}\n\nSearching...\n",
            )

            tc_start = len(engine.state.tool_calls)
            await engine._execute_auto_search(
                queries,
                results_per_query=settings.uarf_auto_search_results_per_query,
                max_context_chars=settings.uarf_auto_search_max_context_chars,
            )

            num_results = engine._state.search_result_count
            yield _flow_content_chunk(
                model, step_name, ctx.complexity,
                f"Found {num_results} results.\n",
            )

            ctx.search_results = engine._state.search_context
            ctx.record_step(step_name, f"Search complete: {num_results} results found.")

            new_calls = [
                {
                    "phase": tc.phase, "tool": tc.tool_name,
                    "input": tc.input_data,
                    "output": (tc.output or "")[:500],
                    "success": tc.success,
                }
                for tc in engine.state.tool_calls[tc_start:]
            ]
            tc_cursor = len(engine.state.tool_calls)
            yield _flow_phase_chunk(
                model, step_name, "complete", pipeline_names,
                complexity=ctx.complexity, tool_calls=new_calls or None, flow_name=_fn,
            )
            continue

        # --- Standard step execution ---
        # Resolve per-step model override
        step_backend, step_model = await _resolve_step_backend(
            step, backend, model, provider_registry,
        )

        yield _flow_phase_chunk(
            model, step_name, "running", pipeline_names,
            complexity=ctx.complexity, flow_name=_fn,
            step_model=step_model, default_model=model,
        )

        # Resolve tools for this step (merge base excludes with tune overrides)
        effective_exclude = (tool_exclude | _tune_disable_tools) if tool_exclude else (_tune_disable_tools or None)
        tools = _resolve_tools_for_step(step, tool_registry, exclude=effective_exclude)

        # Build system prompt with variable substitution + date context
        system_prompt = resolve_variables(step.system_prompt, ctx)
        system_prompt = f"{get_datetime_context()}\n\n{system_prompt}"
        if user_system:
            system_prompt = f"{user_system}\n\n{system_prompt}"

        # Add tool instructions to system prompt
        tier = select_tier(step_backend, step_model) if tools else ToolCallingTier.TEXT
        native_tools: list[dict] | None = None
        structured_schema: dict | None = None

        if tools:
            if tier == ToolCallingTier.NATIVE:
                system_prompt += get_native_tool_prompt_section(tools)
                native_tools = tools_to_native_format(tools)
            elif tier == ToolCallingTier.STRUCTURED:
                structured_schema = build_structured_output_schema(tools)
                system_prompt += get_structured_tool_prompt_section(
                    tools, structured_schema,
                )
            else:
                system_prompt += get_tool_prompt_section(tools)

        # Voice guard for respond/deliver steps — appended LAST so it's the
        # most salient instruction in the system prompt. Bypass with
        # flow_tune={"voice_guard": false} (per-request A/B knob).
        _vg_off = bool(
            flow_tune and isinstance(flow_tune, dict)
            and flow_tune.get("voice_guard") is False
        )
        if step.role in ("respond", "deliver") and not _vg_off:
            system_prompt += _RESPOND_VOICE_GUARD

        # Keep {step:_delivery_context} resolvable: it should mean "everything
        # produced so far". Steps such as Math's Respond (and the shared
        # agentic Deliver template, if an agentic flow is ever resolved while
        # in analytical mode) reference it. Refreshed every step so the final
        # respond/deliver step sees the full accumulated chain.
        ctx._step_outputs["_delivery_context"] = ctx.all_outputs

        # Build user message — pass tools_section so {tools} variable resolves.
        # Steps without an explicit template get FLOW_STEP_USER_TEMPLATE, which
        # carries {all_outputs} (the whole prior chain) rather than only the
        # immediately-preceding step.
        tools_section = get_tool_prompt_section(tools) if tools else ""
        user_msg = build_user_message(
            step.user_template or FLOW_STEP_USER_TEMPLATE, ctx, tools_section,
        )

        # The final answer step is told to cite sources — make sure it actually
        # has them. If the step's template didn't pull {search_results} in,
        # staple the raw search context on so it cites real URLs instead of
        # inventing plausible-looking ones.
        if (
            step.role in ("respond", "deliver")
            and ctx.search_results
            and ctx.search_results not in user_msg
        ):
            user_msg = (
                f"{user_msg}\n\n<sources>\n"
                f"{ctx.search_results[:_RESPOND_SOURCES_CHARS]}\n</sources>"
            )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_msg),
        ]

        # Execute with tool loop
        step_output = ""
        tool_calls_made = 0
        allowed_tool_names = {t.name for t in tools}
        max_tools = flow.max_tool_calls_per_step

        # Per-step tool_choice override (e.g. "required" for classify
        # steps that must always emit a structured tool call). None when
        # the step left tool_choice empty or has no native tools.
        step_tool_choice = (
            _translate_step_tool_choice(step, tools)
            if tier in (ToolCallingTier.NATIVE, ToolCallingTier.STRUCTURED)
            else None
        )

        if step.stream_to_user:
            # Final respond step — stream directly to user
            if tools and tier in (ToolCallingTier.NATIVE, ToolCallingTier.STRUCTURED):
                # Handle tools non-streaming first, then stream final
                step_output = await _tool_loop_nonstreaming(
                    engine, step_backend, step_model, messages, tools,
                    tier=tier, native_tools=native_tools,
                    structured_schema=structured_schema,
                    max_calls=max_tools,
                    step_name=step_name, complexity=ctx.complexity,
                    tool_exclude=tool_exclude, tool_registry=tool_registry,
                    tool_choice=step_tool_choice,
                )
                # Now stream the final response
                req = InternalChatRequest(model=step_model, messages=messages, stream=True)
                parts: list[str] = []
                async for chunk in step_backend.chat_stream(req):
                    if chunk.content_delta:
                        parts.append(chunk.content_delta)
                    yield chunk
                step_output = "".join(parts)
            else:
                # Stream everything (tier 3 or no tools)
                req = InternalChatRequest(model=step_model, messages=messages, stream=True)
                parts = []
                async for chunk in step_backend.chat_stream(req):
                    if chunk.content_delta:
                        parts.append(chunk.content_delta)
                    yield chunk
                step_output = "".join(parts)
        else:
            # Internal step — stream to reasoning panel only
            if tools and tier in (ToolCallingTier.NATIVE, ToolCallingTier.STRUCTURED):
                step_output = await _tool_loop_with_chunks(
                    engine, step_backend, step_model, messages, tools, step_name,
                    pipeline_names, ctx.complexity,
                    tier=tier, native_tools=native_tools,
                    structured_schema=structured_schema,
                    max_calls=max_tools,
                    tool_exclude=tool_exclude, tool_registry=tool_registry,
                    tool_choice=step_tool_choice,
                )
                if step_output:
                    yield _flow_content_chunk(
                        model, step_name, ctx.complexity, step_output,
                    )
            else:
                # Tier 3 streaming to reasoning panel
                req = InternalChatRequest(model=step_model, messages=messages, stream=True)
                parts = []
                async for chunk in step_backend.chat_stream(req):
                    if chunk.content_delta:
                        parts.append(chunk.content_delta)
                        yield _flow_content_chunk(
                            model, step_name, ctx.complexity, chunk.content_delta,
                        )
                step_output = "".join(parts)

        # Record the FULL step output for downstream steps. `output_cap` is no
        # longer used to shred the recorded copy — that discarded ~90% of each
        # analysis step before the final answer ever saw it (a step capped at
        # 400-800 chars routinely produced 5-8k). Bulk concatenation that feeds
        # late synthesis/respond steps is budget-capped per-step inside
        # StepContext.all_outputs instead; here we only guard against a
        # pathological runaway generation blowing the next prompt.
        if len(step_output) > _MAX_RECORDED_STEP_OUTPUT:
            step_output = step_output[:_MAX_RECORDED_STEP_OUTPUT] + "\n\n[... output truncated]"

        # A failed step contributes NOTHING downstream — recording the error
        # text would put "LLM call timed out" into the next step's context as
        # if it were findings.
        step_errored = _step_failed(step_output)
        if step_errored:
            step_output = _strip_step_error(step_output)
            log.warning("flow_step_failed", step=step_name, detail=step_output[:200])
        else:
            ctx.record_step(step_name, step_output)

        # Check for escalation signal from Quick Answer
        if step.role == "respond" and flow.escalation_flow and "[NEEDS_RESEARCH]" in step_output:
            # Strip the signal from the output
            step_output = step_output.replace("[NEEDS_RESEARCH]", "").rstrip()
            ctx.record_step(step_name, step_output)
            # Yield an escalation hint chunk that the UI can show as a button
            yield InternalStreamChunk(
                content_delta="",
                model=model,
                augmentum={
                    "escalation": {
                        "target_flow": flow.escalation_flow,
                        "reason": "uncertainty_detected",
                        "message": "This question may benefit from deeper research with source verification.",
                    },
                },
            )

        # Source validation — check draft outputs against search results
        if (
            settings.source_validation_enabled
            and step.role == "draft"
            and ctx.search_results
        ):
            from augmentum.reasoning.source_validator import validate_draft_sources

            try:
                validation = await validate_draft_sources(
                    step_output, ctx.search_results,
                )
                if not validation.is_valid:
                    warning = (
                        f"SOURCE VALIDATION WARNING: {validation.unsourced_ratio:.0%} of paragraphs "
                        f"({len(validation.unsourced_paragraphs)}/{validation.total_count}) "
                        f"appear unsourced.\n"
                        f"Unsourced paragraphs:\n"
                        + "\n".join(f"  - {p}" for p in validation.unsourced_paragraphs)
                    )
                    ctx.record_step(f"{step_name}_validation", warning)
                    log.warning(
                        "draft_source_validation_failed",
                        step=step_name,
                        unsourced_ratio=validation.unsourced_ratio,
                        unsourced_count=len(validation.unsourced_paragraphs),
                    )
            except Exception:
                log.warning("source_validation_error", exc_info=True)

        # Role-based parsing
        confidence = None
        if step.role == "verify":
            confidence = AnalyticalEngine._parse_confidence(step_output)
            verified = AnalyticalEngine._parse_verified(step_output)
            needs_backtrack = not verified or confidence < _CONFIDENCE_THRESHOLD

            if needs_backtrack and engine._state.backtrack_count < engine._state.max_backtracks:
                engine._state.backtrack_count += 1
                backtrack_reason = AnalyticalEngine._extract_verification_issues(step_output)
                log.info(
                    "flow_backtrack",
                    step=step_name,
                    confidence=confidence,
                    reason=backtrack_reason[:100],
                )
                # Record backtrack in context for downstream steps
                ctx.record_step(
                    f"{step_name}_backtrack",
                    f"Verification failed (confidence: {confidence:.2f}). "
                    f"Issues: {backtrack_reason}",
                )
                # Emit backtrack event for UI
                yield InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum={
                        "backtrack": {
                            "step": step_name,
                            "confidence": confidence,
                            "reason": backtrack_reason[:200],
                            "count": engine._state.backtrack_count,
                        },
                    },
                )

        # Review role — if NEEDS_REVISION, re-run the most recent draft step
        if step.role == "review" and "NEEDS_REVISION" in step_output.upper():
            # Find the most recent draft step that ran before this review
            draft_step = None
            for prior in active_steps:
                if prior.role == "draft" and prior.sort_order < step.sort_order:
                    draft_step = prior

            if draft_step and engine._state.backtrack_count < engine._state.max_backtracks:
                engine._state.backtrack_count += 1
                revision_name = f"{draft_step.name} (Revised)"
                log.info(
                    "flow_revision",
                    review_step=step_name,
                    draft_step=draft_step.name,
                )

                # Signal revision in progress
                yield InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum={
                        "revision": {
                            "review_step": step_name,
                            "draft_step": draft_step.name,
                            "count": engine._state.backtrack_count,
                        },
                    },
                )
                yield _flow_phase_chunk(
                    model, revision_name, "running", pipeline_names,
                    complexity=ctx.complexity, flow_name=_fn,
                )

                # Build revision prompt from original draft prompt + review feedback
                revision_system = resolve_variables(draft_step.system_prompt, ctx)
                _us_prefix = f"{user_system}\n\n" if user_system else ""
                revision_system = (
                    f"{_us_prefix}{get_datetime_context()}\n\n{revision_system}\n\n"
                    f"## REVISION REQUIRED\n"
                    f"A review found these issues:\n{step_output}\n\n"
                    f"Address ALL issues. Output the COMPLETE revised version. "
                    f"Do NOT add commentary about what you changed."
                )
                rev_exclude = (tool_exclude | _tune_disable_tools) if tool_exclude else (_tune_disable_tools or None)
                draft_tools = _resolve_tools_for_step(draft_step, tool_registry, exclude=rev_exclude)
                draft_tools_section = get_tool_prompt_section(draft_tools) if draft_tools else ""
                revision_user = build_user_message(
                    draft_step.user_template or FLOW_STEP_USER_TEMPLATE, ctx, draft_tools_section,
                )
                revision_messages = [
                    Message(role="system", content=revision_system),
                    Message(role="user", content=revision_user),
                ]

                # Execute revision — use the draft step's model override if set
                rev_backend, rev_model = await _resolve_step_backend(
                    draft_step, backend, model, provider_registry,
                )
                req = InternalChatRequest(
                    model=rev_model, messages=revision_messages, stream=True,
                )
                rev_parts: list[str] = []
                async for chunk in rev_backend.chat_stream(req):
                    if chunk.content_delta:
                        rev_parts.append(chunk.content_delta)
                        yield _flow_content_chunk(
                            model, revision_name, ctx.complexity,
                            chunk.content_delta,
                        )
                revised_output = "".join(rev_parts)

                # Apply output cap from original draft step
                if (
                    draft_step.output_cap
                    and draft_step.output_cap > 0
                    and len(revised_output) > draft_step.output_cap
                ):
                    revised_output = revised_output[:draft_step.output_cap] + "..."

                # Overwrite draft output so downstream steps use revised version
                ctx.record_step(draft_step.name, revised_output)

                yield _flow_phase_chunk(
                    model, revision_name, "complete", pipeline_names,
                    complexity=ctx.complexity, flow_name=_fn,
                )

        # Emit tool call records
        new_tools_list = [
            {
                "phase": tc.phase, "tool": tc.tool_name,
                "input": tc.input_data,
                "output": (tc.output or "")[:500],
                "success": tc.success,
            }
            for tc in engine.state.tool_calls[tc_cursor:]
        ]
        tc_cursor = len(engine.state.tool_calls)

        yield _flow_phase_chunk(
            model, step_name, "error" if step_errored else "complete",
            pipeline_names,
            complexity=ctx.complexity,
            confidence=confidence,
            tool_calls=new_tools_list or None,
            flow_name=_fn,
            step_model=step_model, default_model=model,
        )

    # -----------------------------------------------------------------------
    # Fallback: if no step streamed to the user, synthesize a final response
    # using all step outputs as context so the user always gets an answer.
    # -----------------------------------------------------------------------
    if not has_respond_step and active_steps:
        log.info("flow_synthesize_response", flow=flow.name)

        # Gather all step outputs into a context block (budget-bounded via
        # StepContext.all_outputs so a long pipeline can't overflow the prompt).
        combined = ctx.all_outputs
        if not combined:
            step_summaries = [
                f"## {s.name}\n{ctx.get_step_output(s.name)}"
                for s in active_steps if ctx.get_step_output(s.name)
            ]
            combined = "\n\n".join(step_summaries)

        sources_block = (
            f"\n\n### Sources\n{ctx.search_results[:_RESPOND_SOURCES_CHARS]}"
            if ctx.search_results else ""
        )
        _us_prefix = f"{user_system}\n\n" if user_system else ""
        _voice_guard_tail = ""
        _vg_on = True
        if flow_tune and isinstance(flow_tune, dict) and flow_tune.get("voice_guard") is False:
            _vg_on = False
        if _vg_on:
            _voice_guard_tail = _RESPOND_VOICE_GUARD
        respond_system = (
            f"{_us_prefix}{get_datetime_context()}\n\n"
            "You are a helpful assistant. The user asked a question and "
            "several analysis steps have already been completed. Use the "
            "analysis results below to provide a clear, comprehensive "
            "final answer to the user. Cite sources with their URLs when the "
            "analysis drew on them.\n\n"
            f"### Analysis Results\n{combined}{sources_block}"
            f"{_voice_guard_tail}"
        )
        respond_messages = [
            Message(role="system", content=respond_system),
            Message(role="user", content=query),
        ]
        req = InternalChatRequest(
            model=model, messages=respond_messages, stream=True,
        )
        async for chunk in backend.chat_stream(req):
            if chunk.content_delta:
                yield chunk


# ---------------------------------------------------------------------------
# Tool loop helpers
# ---------------------------------------------------------------------------


async def _tool_loop_nonstreaming(
    engine: AnalyticalEngine,
    backend: ModelBackend,
    model: str,
    messages: list[Message],
    tools: list,
    *,
    tier: ToolCallingTier,
    native_tools: list[dict] | None,
    structured_schema: dict | None,
    max_calls: int,
    step_name: str,
    complexity: str,
    tool_exclude: frozenset[str] | None,
    tool_registry: ToolRegistry | None,
    tool_choice: str | dict | None = None,
) -> str:
    """Run non-streaming tool loop for tier 1/2. Returns final output."""
    import asyncio

    allowed_tool_names = {t.name for t in tools}
    calls = 0
    _timeout = settings.tool_execution_timeout

    while calls < max_calls:
        req = InternalChatRequest(
            model=model, messages=list(messages), stream=False,
            tools=native_tools, format=structured_schema,
            tool_choice=tool_choice,
        )
        try:
            response = await asyncio.wait_for(backend.chat(req), timeout=_timeout)
        except TimeoutError:
            log.warning("tool_loop_llm_timeout", step=step_name, timeout=_timeout)
            return f"{_STEP_ERROR_PREFIX}LLM call timed out after {_timeout}s during tool loop."
        output = response.message.content if response.message else ""

        # Parse tool call
        tool_name = ""
        tool_input: dict = {}

        if tier == ToolCallingTier.NATIVE:
            parsed = parse_native_tool_call(response)
            if parsed:
                tool_name, tool_input = parsed
        else:
            parsed = parse_structured_output(output)
            if parsed:
                tool_name, tool_input = parsed
            elif output:
                output = extract_structured_text(output)

        # Fallback: Python-style function call (primary text format)
        # e.g. web_search(query="current weather") — matches model pre-training
        if not tool_name:
            py_parsed = parse_python_style_tool_call(output, known_tools=allowed_tool_names)
            if py_parsed:
                tool_name, tool_input = py_parsed
                log.info("python_style_tool_call_detected", tool=tool_name)

        # Fallback: legacy TOOL_CALL: format (silent backward compat)
        if not tool_name:
            fallback_name, fallback_input = AnalyticalEngine._parse_tool_call(output)
            if fallback_name:
                tool_name, tool_input = fallback_name, fallback_input

        if not tool_name:
            return output

        # Validate
        resolved = tool_registry.resolve(tool_name) if tool_registry else None
        if not resolved or resolved.name not in allowed_tool_names:
            return output

        tool_input = coerce_tool_params(resolved, tool_input)
        tool_result = await engine._execute_tool(
            step_name.lower(), tool_name, tool_input, exclude=tool_exclude,
        )

        # Validation errors don't count against budget — give model a
        # chance to correct its tool call
        if not tool_result.validation_error:
            calls += 1

        if tool_result.validation_error:
            followup = (
                f"Your tool call failed: {tool_result.error}\n"
                "Please correct your tool call and try again."
            )
        else:
            followup = (
                f"## Tool Result ({resolved.name})\n"
                f"{'Success' if tool_result.success else 'Error'}: "
                f"{tool_result.output or tool_result.error}\n\n"
                "Continue your analysis incorporating this information."
            )

        messages.append(Message(role="assistant", content=output))
        messages.append(Message(role="user", content=followup))

    # Exhausted calls — get final response. Switch a "required" override
    # to None here so the model can answer instead of being forced into
    # yet another tool call we'd have nowhere to execute. A pinned
    # specific-tool override is similarly demoted; "none"/"auto" stay
    # as-is.
    final_tool_choice = tool_choice
    if isinstance(tool_choice, dict) or tool_choice == "required":
        final_tool_choice = None
    req = InternalChatRequest(
        model=model, messages=list(messages), stream=False,
        tools=native_tools, format=structured_schema,
        tool_choice=final_tool_choice,
    )
    try:
        response = await asyncio.wait_for(backend.chat(req), timeout=_timeout)
    except TimeoutError:
        log.warning("tool_loop_final_llm_timeout", step=step_name, timeout=_timeout)
        return f"{_STEP_ERROR_PREFIX}LLM call timed out after {_timeout}s."
    output = response.message.content if response.message else ""
    if tier == ToolCallingTier.STRUCTURED and output:
        output = extract_structured_text(output)
    return output


async def _tool_loop_with_chunks(
    engine: AnalyticalEngine,
    backend: ModelBackend,
    model: str,
    messages: list[Message],
    tools: list,
    step_name: str,
    pipeline: list[str],
    complexity: str,
    *,
    tier: ToolCallingTier,
    native_tools: list[dict] | None,
    structured_schema: dict | None,
    max_calls: int,
    tool_exclude: frozenset[str] | None,
    tool_registry: ToolRegistry | None,
    tool_choice: str | dict | None = None,
) -> str:
    """Run tool loop returning final output (chunks yielded by caller)."""
    # Same as nonstreaming but without yielding chunks directly
    return await _tool_loop_nonstreaming(
        engine, backend, model, messages, tools,
        tier=tier, native_tools=native_tools,
        structured_schema=structured_schema,
        max_calls=max_calls,
        step_name=step_name, complexity=complexity,
        tool_exclude=tool_exclude, tool_registry=tool_registry,
        tool_choice=tool_choice,
    )
