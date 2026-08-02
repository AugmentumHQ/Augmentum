"""Analytical mode handler — processes requests through the UARF analytical engine.

Streams ALL phases in real-time to the UI (not just CONCLUDE). Phase reasoning
content is carried in ``augmentum.phase_content_delta`` metadata so the UI can
display it in the inline thinking block. Tool calls are emitted as they happen.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    Usage,
)
from augmentum.modes.analytical.engine import (
    _MAX_TOOL_CALLS_PER_PHASE,
    AnalyticalEngine,
)
from augmentum.modes.analytical.prompts import (
    get_native_tool_prompt_section,
    get_phase_prompt,
    get_structured_tool_prompt_section,
    get_tool_prompt_section,
)
from augmentum.modes.analytical.state import AnalyticalPhase, PhaseResult, ToolCallRecord
from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    build_structured_output_schema,
    coerce_tool_params,
    extract_structured_text,
    parse_native_tool_call,
    parse_native_tool_calls_all,
    parse_python_style_tool_call,
    parse_structured_output,
    select_tier,
    tools_to_native_format,
)
from augmentum.modes.base import ModeHandler
from augmentum.modes.v_command import extract_v_command, generate_direct_image
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.cache.prompt_cache import PromptCache
    from augmentum.image.queue import GenerationQueue
    from augmentum.reasoning.store import FlowStore
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

# Ordered list for UI display
_FULL_PHASES = ["ASSESS", "IDENTIFY", "RELEVANT", "APPLY", "CONCLUDE"]
_MODERATE_PHASES = ["ASSESS", "GATHER", "APPLY", "CONCLUDE"]
_SIMPLE_PHASES = ["ASSESS", "APPLY", "CONCLUDE"]
_FULL_PHASES_WITH_SEARCH = [
    "ASSESS", "SEARCH", "IDENTIFY", "RELEVANT", "APPLY", "CONCLUDE",
]
_MODERATE_PHASES_WITH_SEARCH = ["ASSESS", "SEARCH", "GATHER", "APPLY", "CONCLUDE"]
_SIMPLE_PHASES_WITH_SEARCH = ["ASSESS", "SEARCH", "APPLY", "CONCLUDE"]


def _phase_chunk(
    model: str,
    phase: str,
    status: str,
    pipeline: list[str],
    *,
    confidence: float | None = None,
    complexity: str = "",
    content: str = "",
    tool_calls: list[dict] | None = None,
) -> InternalStreamChunk:
    """Build a stream chunk carrying UARF phase metadata."""
    phases_list = []
    for p in pipeline:
        if p == phase:
            phases_list.append({"name": p, "status": status})
        elif pipeline.index(p) < pipeline.index(phase):
            phases_list.append({"name": p, "status": "complete"})
        else:
            phases_list.append({"name": p, "status": "pending"})

    meta: dict = {
        "mode": "analytical",
        "phase": phase,
        "phase_status": status,
        "phases": phases_list,
    }
    if complexity:
        meta["complexity"] = complexity
    if confidence is not None:
        meta["confidence"] = confidence
    if tool_calls:
        meta["tool_calls"] = tool_calls

    return InternalStreamChunk(
        content_delta=content,
        model=model,
        augmentum=meta,
    )


def _phase_content_chunk(
    model: str,
    phase: str,
    complexity: str,
    content_delta: str,
) -> InternalStreamChunk:
    """Build a lightweight chunk carrying phase content delta for the thinking block.

    Unlike ``_phase_chunk``, this doesn't include the full phases list (too
    verbose for every token). The UI uses the phase name to route content to
    the correct phase display area.
    """
    meta: dict = {
        "mode": "analytical",
        "phase": phase,
        "phase_content_delta": content_delta,
    }
    if complexity:
        meta["complexity"] = complexity
    return InternalStreamChunk(
        content_delta="",  # Empty — not response body content
        model=model,
        augmentum=meta,
    )


def _merge_native_tc_delta(accumulator: dict[int, dict], delta: dict) -> None:
    """Merge one streaming native ``tool_call`` SSE delta into the accumulator.

    Mirrors the passthrough resolver: each delta is
    ``{index, id?, type?, function: {name?, arguments?}}``. The first delta
    for an index carries id/type/name; later deltas carry arguments fragments
    that concatenate into the final JSON string. Lets the NATIVE tool loop
    stream prose live while rebuilding tool_calls from the side-channel.
    """
    idx = int(delta.get("index", 0) or 0)
    entry = accumulator.get(idx)
    if entry is None:
        entry = {"id": "", "type": "function",
                 "function": {"name": "", "arguments": ""}}
        accumulator[idx] = entry
    if delta.get("id"):
        entry["id"] = delta["id"]
    if delta.get("type"):
        entry["type"] = delta["type"]
    fn_delta = delta.get("function") or {}
    if fn_delta.get("name"):
        entry["function"]["name"] += fn_delta["name"] or ""
    if fn_delta.get("arguments"):
        entry["function"]["arguments"] += fn_delta["arguments"] or ""


def _extract_tool_calls(engine: AnalyticalEngine, start_idx: int = 0) -> list[dict]:
    """Extract tool call records from engine state as serialisable dicts.

    Returns records from ``start_idx`` onwards so callers can track which
    calls are new since the last phase.
    """
    calls = engine.state.tool_calls[start_idx:]
    return [_tool_call_dict(tc) for tc in calls]


def _tool_call_dict(tc) -> dict:
    """Convert a single ToolCallRecord to a serialisable dict.

    Includes the structured ``card`` envelope when the tool produced one,
    so the frontend can render a typed ToolCard (preview/edit/download)
    instead of dumping ``output`` as plain markdown.
    """
    out: dict = {
        "phase": tc.phase,
        "tool": tc.tool_name,
        "input": tc.input_data,
        "output": (tc.output or "")[:500],
        "success": tc.success,
    }
    card = getattr(tc, "card", None)
    if card:
        out["card"] = card
    return out


# Phrases from system prompts that small models sometimes echo verbatim.
# If the output starts with one of these, strip the echoed preamble.
_ECHO_MARKERS = (
    "You are an analytical",
    "## Your Goal",
    "## Instructions",
    "Your task is to",
)

# Markers/punctuation that can begin a tool-call syntax in text-tier output.
_TOOL_CALL_LEGACY_MARKER = "TOOL_CALL:"


def _find_tool_call_start(buf: str, known_tools: set[str]) -> int | None:
    """Return earliest index in buf where a tool call begins, or None.

    Detects either the legacy ``TOOL_CALL:`` marker or a Python-style
    ``known_tool_name(`` invocation at a word boundary. Used to suppress
    tool-call syntax (and giant tool args like an ebook body) from the
    visible stream before the call is parsed and executed.
    """
    earliest: int | None = None
    idx = buf.find(_TOOL_CALL_LEGACY_MARKER)
    if idx >= 0:
        earliest = idx
    for name in known_tools:
        needle = name + "("
        pos = 0
        while True:
            i = buf.find(needle, pos)
            if i < 0:
                break
            # Word boundary on the left: start of buf or non-identifier char
            if i == 0 or not (buf[i - 1].isalnum() or buf[i - 1] == "_"):
                if earliest is None or i < earliest:
                    earliest = i
                break
            pos = i + 1
    return earliest


def _strip_tool_call_for_history(text: str, known_tools: set[str]) -> str:
    """Replace the tool-call portion of an assistant turn with a placeholder.

    Keeps prose preceding the call (the model's reasoning) and discards the
    raw call syntax so later phases can't accidentally re-quote a 50KB body
    back into the visible stream.
    """
    cut = _find_tool_call_start(text, known_tools)
    if cut is None:
        return text
    head = text[:cut].rstrip()
    return f"{head}\n[tool_call elided]" if head else "[tool_call elided]"


def _strip_system_echo(output: str) -> str:
    """Strip echoed system prompt text from the start of phase output.

    Small models sometimes reproduce parts of their system prompt before
    writing the actual response.  Detect common echoed prefixes and strip
    them, keeping only the model's real output.
    """
    stripped = output.lstrip()
    if not stripped:
        return output

    # Check if output starts with a known system prompt phrase
    starts_with_echo = any(stripped.startswith(marker) for marker in _ECHO_MARKERS)
    if not starts_with_echo:
        return output

    # Find the first real content section (starts after double newline)
    # Look for structured output markers that signal real content
    content_markers = (
        "ERRORS_FOUND:", "UNSUPPORTED_CLAIMS:", "CONTRADICTIONS:",
        "VERIFIED:", "CONFIDENCE:", "VERIFICATION_NOTES:",
        "RELEVANT_KNOWLEDGE:", "APPLICABLE_METHODS:",
        "REASONING:", "PRELIMINARY_ANSWER:",
        "STEP 1:", "KEY_CONCEPTS:",
    )
    for marker in content_markers:
        idx = stripped.find(marker)
        if idx > 0:
            log.debug(
                "stripped_system_echo",
                echo_len=idx,
                marker=marker,
            )
            return stripped[idx:]

    # Fallback: couldn't find content marker, return as-is
    return output


class AnalyticalHandler(ModeHandler):
    """Processes requests through the UARF analytical engine with real-time streaming."""

    def __init__(
        self,
        backend: ModelBackend,
        tool_registry: ToolRegistry | None = None,
        prompt_cache: PromptCache | None = None,
        image_queue: GenerationQueue | None = None,
        image_enabled: bool = False,
        session_id: str = "",
        flow_store: FlowStore | None = None,
        provider_registry: object | None = None,
        circuit_breaker: object | None = None,
        flow_tune: dict | None = None,
        explicit_flow_id: str = "",
        enabled_tools: list[str] | None = None,
        user_id: str = "",
    ) -> None:
        self._backend = backend
        self._tool_registry = tool_registry
        self._prompt_cache = prompt_cache
        self._image_queue = image_queue
        self._image_enabled = image_enabled
        self._session_id = session_id
        self._flow_store = flow_store
        self._provider_registry = provider_registry
        self._circuit_breaker = circuit_breaker
        self._flow_tune = flow_tune
        self._explicit_flow_id = explicit_flow_id
        self._user_id = user_id
        # User's textbox tool selection (the source of truth). ``None`` ⇒
        # caller did not pipe a header through, fall back to phase defaults.
        # Empty list ⇒ user explicitly chose "no tools". Otherwise: filter
        # phase capability ∩ this set on every ``get_for_phase`` call.
        self._enabled_tools: frozenset[str] | None = (
            frozenset(enabled_tools) if enabled_tools is not None else None
        )

    async def _handle(self, request: InternalChatRequest) -> InternalChatResponse:
        """Process a non-streaming analytical request.

        Runs the full UARF pipeline and returns the CONCLUDE phase output.
        """
        import asyncio

        # Per-turn search dedup shared across every UARF phase + auto-search of
        # this turn (turn_search_dedup): the same image/video/page returned in
        # one phase is remembered so later phases surface only NEW results.
        # Auto-search runs its queries via asyncio.gather, whose tasks capture
        # this context, so they share the one dedup object installed here.
        from augmentum.tools.turn_search_dedup import TurnSearchDedup, set_turn_dedup
        set_turn_dedup(TurnSearchDedup())

        has_v, v_instruction, cleaned = extract_v_command(request)
        image_task = None
        if has_v and self._image_enabled and self._image_queue:
            image_task = asyncio.create_task(
                generate_direct_image(v_instruction, self._image_queue, self._session_id, user_id=self._user_id)
            )
            request = cleaned

        # Knowledge library context (Discovery Engine)
        try:
            from augmentum.discovery import (
                inject_system_context,
                retrieve_knowledge_context,
            )
            _user_query = request.messages[-1].content if request.messages else ""
            if _user_query and isinstance(_user_query, str):
                _knowledge_ctx = await retrieve_knowledge_context(
                    request.app.state, _user_query,
                    user_id=self._user_id,
                )
                if _knowledge_ctx:
                    inject_system_context(request.messages, _knowledge_ctx)
        except Exception:
            log.debug("analytical_knowledge_retrieve_failed", exc_info=True)

        engine = AnalyticalEngine(
            self._backend,
            tool_registry=self._tool_registry,
            prompt_cache=self._prompt_cache,
            provider_registry=self._provider_registry,
            circuit_breaker=self._circuit_breaker,
            enabled_tools=self._enabled_tools,
            user_id=self._user_id,
        )
        result = await engine.process(request)

        content = result.conclusion
        if image_task:
            image_url = await image_task
            if image_url:
                content += f"\n\n![Generated Image]({image_url})"

        return InternalChatResponse(
            message=Message(role="assistant", content=content),
            model=request.model,
            finish_reason="stop",
            usage=Usage(total_tokens=result.total_tokens),
        )

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    async def _stream_llm(
        self, model: str, messages: list[Message],
        *, tools: list[dict] | None = None,
        format: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream an LLM call, yielding raw content deltas (strings)."""
        req = InternalChatRequest(
            model=model, messages=list(messages), stream=True,
            tools=tools, format=format,
        )
        async for chunk in self._backend.chat_stream(req):
            if chunk.content_delta:
                yield chunk.content_delta

    async def _call_llm(
        self, model: str, messages: list[Message],
        *, tools: list[dict] | None = None,
        format: dict | None = None,
    ):
        """Non-streaming LLM call. Returns the full InternalChatResponse."""
        req = InternalChatRequest(
            model=model, messages=list(messages), stream=False,
            tools=tools, format=format,
        )
        return await self._backend.chat(req)

    async def _stream_llm_native(
        self, model: str, messages: list[Message],
        *, tools: list[dict] | None = None,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream a NATIVE-tier call, yielding FULL chunks (not just content
        strings like ``_stream_llm``) so the caller can forward content_delta
        live while accumulating native ``tool_call`` deltas off the augmentum
        side-channel. Backs the stream-first NATIVE tool loop in
        ``_stream_phase`` (replaces the old non-streaming peek-then-blob)."""
        req = InternalChatRequest(
            model=model, messages=list(messages), stream=True, tools=tools,
        )
        async for chunk in self._backend.chat_stream(req):
            yield chunk

    async def _stream_phase(
        self,
        engine: AnalyticalEngine,
        phase: AnalyticalPhase,
        model: str,
        pipeline: list[str],
        complexity: str,
        *,
        system_prompt: str,
        query: str,
        enable_tools: bool = False,
        result_out: dict,
        user_system: str = "",
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream a single UARF phase with optional tool-calling loop.

        Automatically selects the best tool-calling tier:

        - **Tier 1/2:** Tool-calling portion runs non-streaming so we can
          parse the complete response for ``tool_calls`` (native) or JSON
          (structured). After all tool calls, the final response streams.
        - **Tier 3:** Streams tokens, checks for ``TOOL_CALL`` at end,
          executes tools, and loops (existing behaviour).

        Yields ``InternalStreamChunk`` objects carrying phase content deltas
        and tool call events. Writes the final accumulated output to
        ``result_out["output"]`` when the generator completes.
        """
        phase_upper = phase.value.upper()
        phase_value = phase.value

        # Tool setup — exclude web_search when auto-search already handled it
        tools: list = []
        tool_exclude = frozenset({"web_search"}) if engine._state.needs_search else None
        if enable_tools and self._tool_registry:
            tools = self._tool_registry.get_for_phase(
                phase_value,
                exclude=tool_exclude,
                allowed_names=self._enabled_tools,
            )
            # Pre-filter: reduce tool count based on query analysis
            if tools and settings.tool_prefilter_enabled and engine._state.query:
                from augmentum.tools.filter import filter_tools_for_query
                tools = filter_tools_for_query(
                    engine._state.query, tools,
                    min_tools=settings.tool_prefilter_min_tools,
                )
            log.info(
                "stream_phase_tools",
                phase=phase_upper,
                enable_tools=enable_tools,
                tool_count=len(tools),
                tool_names=[t.name for t in tools],
            )
        else:
            log.info(
                "stream_phase_no_tools",
                phase=phase_upper,
                enable_tools=enable_tools,
                has_registry=self._tool_registry is not None,
            )

        # Select tool-calling tier
        tier = select_tier(self._backend, model) if tools else ToolCallingTier.TEXT

        # Tier-specific prompt and request params
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
            else:  # TEXT
                system_prompt += get_tool_prompt_section(tools)

            # Proactive suggestions for search/math-heavy phases
            if phase in (AnalyticalPhase.RELEVANT, AnalyticalPhase.APPLY, AnalyticalPhase.GATHER):
                suggestions = AnalyticalEngine._get_proactive_suggestions(query, tools=tools)
                if suggestions:
                    system_prompt += (
                        "\n\n## Proactive Suggestions\n"
                        "Based on the query, consider using these tools:\n"
                        + "\n".join(f"- {s}" for s in suggestions)
                    )

        # Prepend user personalization to the phase system prompt
        if user_system:
            system_prompt = f"{user_system}\n\n{system_prompt}"

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query),
        ]

        final_output = ""
        tool_calls_made = 0
        allowed_tool_names = {t.name for t in tools}

        # ---- Tier 1/2: tool loop. NATIVE streams content live + accumulates
        #      native tool_call deltas off the side-channel; STRUCTURED keeps
        #      the non-streaming peek (it must parse a complete constrained-JSON
        #      response). (2026-06-30: NATIVE was peek-then-blob; now stream-first.)
        if tools and tier in (ToolCallingTier.NATIVE, ToolCallingTier.STRUCTURED):
            while tool_calls_made < _MAX_TOOL_CALLS_PER_PHASE:
                streamed_live = False
                if tier == ToolCallingTier.NATIVE:
                    # Stream prose to the user AS it generates; native tool_call
                    # deltas arrive on the augmentum side-channel and never
                    # interleave with content, so we forward content immediately.
                    _tc_acc: dict[int, dict] = {}
                    _parts: list[str] = []
                    _finish: str | None = None
                    async for chunk in self._stream_llm_native(
                        model, messages, tools=native_tools,
                    ):
                        tc_deltas = (chunk.augmentum or {}).get("tool_calls")
                        if tc_deltas:
                            for d in tc_deltas:
                                _merge_native_tc_delta(_tc_acc, d)
                            # Native tool_call deltas normally don't interleave
                            # with prose, but if a model DOES bundle content into
                            # the same chunk, stream it live too. Capturing it
                            # only into _parts would lose it: streamed_live=True
                            # suppresses the terminal emit, so unstreamed prose
                            # here would reach history but never the visible
                            # stream when the call is later dropped/unparsed.
                            if chunk.content_delta:
                                _parts.append(chunk.content_delta)
                                yield _phase_content_chunk(
                                    model, phase_upper, complexity,
                                    chunk.content_delta,
                                )
                            if chunk.finish_reason:
                                _finish = chunk.finish_reason
                            continue
                        if chunk.content_delta:
                            _parts.append(chunk.content_delta)
                            yield _phase_content_chunk(
                                model, phase_upper, complexity, chunk.content_delta,
                            )
                        if chunk.finish_reason:
                            _finish = chunk.finish_reason
                    streamed_live = True
                    current_output = "".join(_parts)
                    # Synthetic response so the existing native parsers run unchanged.
                    response = InternalChatResponse(
                        message=Message(
                            role="assistant",
                            content=current_output,
                            tool_calls=[_tc_acc[i] for i in sorted(_tc_acc)] or None,
                        ),
                        model=model,
                        finish_reason=_finish or ("tool_calls" if _tc_acc else "stop"),
                    )
                else:  # STRUCTURED
                    response = await self._call_llm(
                        model, messages,
                        tools=native_tools, format=structured_schema,
                    )
                    current_output = response.message.content if response.message else ""

                # Parse tool call(s) based on tier
                parsed_calls: list[tuple[str, dict]] = []

                if tier == ToolCallingTier.NATIVE:
                    # Extract ALL tool calls for parallel execution
                    parsed_calls = parse_native_tool_calls_all(response)
                    # Single-call fallback for models that return one
                    if not parsed_calls:
                        parsed = parse_native_tool_call(response)
                        if parsed:
                            parsed_calls = [parsed]
                else:  # STRUCTURED
                    parsed = parse_structured_output(current_output)
                    if parsed:
                        parsed_calls = [parsed]
                    elif current_output:
                        current_output = extract_structured_text(current_output)

                # Fallback: Python-style function call (primary text format)
                if not parsed_calls:
                    py_parsed = parse_python_style_tool_call(
                        current_output, known_tools=allowed_tool_names,
                    )
                    if py_parsed:
                        parsed_calls = [py_parsed]
                        log.info(
                            "stream_tool_python_style_detected",
                            phase=phase_upper, tool=py_parsed[0],
                        )

                # Fallback: legacy TOOL_CALL: format
                if not parsed_calls:
                    fallback_name, fallback_input = AnalyticalEngine._parse_tool_call(
                        current_output,
                    )
                    if fallback_name:
                        log.info(
                            "stream_tool_legacy_text_fallback",
                            phase=phase_upper,
                            tier=tier.value,
                            tool=fallback_name,
                        )
                        parsed_calls = [(fallback_name, fallback_input)]

                if not parsed_calls:
                    # No tool call — finish. NATIVE already streamed the prose
                    # live above; only STRUCTURED needs to emit it now (avoids
                    # a double-emit of the whole phase response).
                    if current_output and not streamed_live:
                        yield _phase_content_chunk(
                            model, phase_upper, complexity, current_output,
                        )
                    final_output = current_output
                    break

                # Validate and resolve all parsed calls
                validated_calls: list[tuple[str, dict, object]] = []
                for tc_name, tc_input in parsed_calls:
                    resolved = (
                        self._tool_registry.resolve(tc_name)
                        if self._tool_registry else None
                    )
                    if resolved is None or resolved.name not in allowed_tool_names:
                        log.info(
                            "stream_tool_not_allowed_for_phase",
                            phase=phase_upper,
                            tool=tc_name,
                            resolved=resolved.name if resolved else None,
                            allowed=list(allowed_tool_names),
                        )
                        continue
                    tc_input = coerce_tool_params(resolved, tc_input)
                    validated_calls.append((resolved.name, tc_input, resolved))

                if not validated_calls:
                    # All parsed calls were disallowed — finish. NATIVE already
                    # streamed the prose live; only STRUCTURED emits it now.
                    if current_output and not streamed_live:
                        yield _phase_content_chunk(
                            model, phase_upper, complexity, current_output,
                        )
                    final_output = current_output
                    break

                # Respect remaining budget
                remaining = _MAX_TOOL_CALLS_PER_PHASE - tool_calls_made
                validated_calls = validated_calls[:remaining]

                for vc_name, vc_input, _ in validated_calls:
                    log.info(
                        "stream_phase_tool_call",
                        phase=phase_upper,
                        tier=tier.value,
                        tool_name=vc_name,
                        tool_input=vc_input,
                    )

                # Execute tools — parallel if multiple, sequential if one
                from augmentum.tools.base import ToolResult

                if len(validated_calls) == 1:
                    vc_name, vc_input, _ = validated_calls[0]
                    tool_results = [
                        (vc_name, await engine._execute_tool(
                            phase_value, vc_name, vc_input,
                            exclude=tool_exclude,
                        )),
                    ]
                else:
                    log.info(
                        "stream_parallel_tool_execution",
                        phase=phase_upper,
                        count=len(validated_calls),
                        tools=[c[0] for c in validated_calls],
                    )

                    async def _safe_execute(
                        name: str, inp: dict,
                    ) -> tuple[str, ToolResult]:
                        try:
                            result = await engine._execute_tool(
                                phase_value, name, inp,
                                exclude=tool_exclude,
                            )
                        except Exception as exc:
                            log.warning(
                                "stream_parallel_tool_failed",
                                tool=name, error=str(exc),
                            )
                            result = ToolResult(success=False, error=str(exc))
                        return (name, result)

                    tool_results = await asyncio.gather(
                        *[_safe_execute(c[0], c[1]) for c in validated_calls],
                    )

                # Count non-validation-error calls and emit events
                for tc_name, tc_result in tool_results:
                    if not tc_result.validation_error:
                        tool_calls_made += 1
                    log.info(
                        "stream_phase_tool_result",
                        phase=phase_upper,
                        tool_name=tc_name,
                        success=tc_result.success,
                        output_len=len(tc_result.output or tc_result.error or ""),
                    )

                # Yield tool call events for UI
                recent_records = engine._state.tool_calls[-len(tool_results):]
                yield _phase_chunk(
                    model, phase_upper, "running", pipeline,
                    complexity=complexity,
                    tool_calls=[_tool_call_dict(r) for r in recent_records],
                )

                # Build combined result message
                has_validation_error = any(
                    r.validation_error for _, r in tool_results
                )
                if has_validation_error and len(tool_results) == 1:
                    # Single validation error — corrective prompt
                    followup = (
                        f"Your tool call failed: {tool_results[0][1].error}\n"
                        "Please correct your tool call and try again."
                    )
                else:
                    result_parts = []
                    for tc_name, tc_result in tool_results:
                        status = "Success" if tc_result.success else "Error"
                        result_parts.append(
                            f"## Tool Result ({tc_name})\n"
                            f"{status}: {tc_result.output or tc_result.error}"
                        )
                    followup = (
                        "\n\n".join(result_parts)
                        + "\n\nContinue your analysis incorporating "
                        + ("these results." if len(tool_results) > 1 else "this information.")
                    )
                messages.append(Message(
                    role="assistant",
                    content=_strip_tool_call_for_history(
                        current_output, allowed_tool_names,
                    ),
                ))
                messages.append(Message(role="user", content=followup))
            else:
                # Exhausted tool call limit — final answer. NATIVE streams it
                # live (no tools left to parse); STRUCTURED keeps the peek.
                if tier == ToolCallingTier.NATIVE:
                    _parts = []
                    async for chunk in self._stream_llm_native(model, messages):
                        if chunk.content_delta:
                            _parts.append(chunk.content_delta)
                            yield _phase_content_chunk(
                                model, phase_upper, complexity, chunk.content_delta,
                            )
                    final_output = "".join(_parts)
                else:
                    response = await self._call_llm(
                        model, messages,
                        tools=native_tools, format=structured_schema,
                    )
                    current_output = response.message.content if response.message else ""
                    if tier == ToolCallingTier.STRUCTURED and current_output:
                        current_output = extract_structured_text(current_output)
                    if current_output:
                        yield _phase_content_chunk(
                            model, phase_upper, complexity, current_output,
                        )
                    final_output = current_output

        # ---- Tier 3: Streaming tool loop (existing behaviour) ----
        else:
            # Hold-back size for partial tool-name detection across delta seams.
            _max_name_len = max((len(n) for n in allowed_tool_names), default=0)
            _hold_tail = _max_name_len + len(_TOOL_CALL_LEGACY_MARKER) + 2
            while True:
                current_output = ""
                visible_up_to = 0
                suppressed = False
                pending_emitted = False
                async for delta in self._stream_llm(model, messages):
                    current_output += delta
                    if suppressed or not allowed_tool_names:
                        if not allowed_tool_names:
                            # No tools — emit normally
                            yield _phase_content_chunk(
                                model, phase_upper, complexity, delta,
                            )
                        # else: swallow — tool-call args must not stream visibly
                        continue
                    cut = _find_tool_call_start(current_output, allowed_tool_names)
                    if cut is not None:
                        # Flush any prose preceding the tool call, then suppress.
                        if cut > visible_up_to:
                            yield _phase_content_chunk(
                                model, phase_upper, complexity,
                                current_output[visible_up_to:cut],
                            )
                            visible_up_to = cut
                        suppressed = True
                        if not pending_emitted:
                            # Lightweight signal so UI can show "calling tool…"
                            yield _phase_content_chunk(
                                model, phase_upper, complexity, "",
                            )
                            pending_emitted = True
                        continue
                    # No call detected yet — emit only the safe prefix, hold a tail
                    safe_end = max(visible_up_to, len(current_output) - _hold_tail)
                    if safe_end > visible_up_to:
                        yield _phase_content_chunk(
                            model, phase_upper, complexity,
                            current_output[visible_up_to:safe_end],
                        )
                        visible_up_to = safe_end
                # Stream finished — flush any held tail if no call was detected
                if not suppressed and visible_up_to < len(current_output):
                    yield _phase_content_chunk(
                        model, phase_upper, complexity,
                        current_output[visible_up_to:],
                    )

                final_output = current_output

                # Check for tool calls (skip if no tools or at limit)
                if not tools or tool_calls_made >= _MAX_TOOL_CALLS_PER_PHASE:
                    log.debug(
                        "stream_phase_no_tool_check",
                        phase=phase_upper,
                        has_tools=bool(tools),
                        calls_made=tool_calls_made,
                    )
                    break

                # Python-style first (primary text format)
                py_parsed = parse_python_style_tool_call(
                    current_output, known_tools=allowed_tool_names,
                )
                if py_parsed:
                    tool_name, tool_input = py_parsed
                else:
                    # Legacy TOOL_CALL: format fallback
                    tool_name, tool_input = AnalyticalEngine._parse_tool_call(
                        current_output,
                    )
                if not tool_name:
                    log.debug(
                        "stream_phase_no_tool_parsed",
                        phase=phase_upper,
                        output_len=len(current_output),
                        output_tail=current_output[-200:] if current_output else "",
                    )
                    break

                # Validate tool is allowed for this phase
                resolved = (
                    self._tool_registry.resolve(tool_name)
                    if self._tool_registry else None
                )
                if resolved is None or resolved.name not in allowed_tool_names:
                    log.info(
                        "stream_tool_not_allowed_for_phase",
                        phase=phase_upper,
                        tool=tool_name,
                        resolved=resolved.name if resolved else None,
                        allowed=list(allowed_tool_names),
                    )
                    break

                log.info(
                    "stream_phase_tool_call",
                    phase=phase_upper,
                    tier="text",
                    tool_name=tool_name,
                    tool_input=tool_input,
                )

                # Execute tool (pass exclude to block auto-searched tools)
                tool_result = await engine._execute_tool(
                    phase_value, tool_name, tool_input,
                    exclude=tool_exclude,
                )
                if not tool_result.validation_error:
                    tool_calls_made += 1
                log.info(
                    "stream_phase_tool_result",
                    phase=phase_upper,
                    tool_name=tool_name,
                    success=tool_result.success,
                    output_len=len(tool_result.output or tool_result.error or ""),
                )

                # Yield tool call event
                tc_record = engine._state.tool_calls[-1]
                yield _phase_chunk(
                    model, phase_upper, "running", pipeline,
                    complexity=complexity,
                    tool_calls=[_tool_call_dict(tc_record)],
                )

                # Inject tool result and loop — corrective prompt for validation errors
                if tool_result.validation_error:
                    followup = (
                        f"Your tool call failed: {tool_result.error}\n"
                        "Please correct your tool call and try again."
                    )
                else:
                    followup = (
                        f"## Tool Result ({tool_name})\n"
                        f"{'Success' if tool_result.success else 'Error'}: "
                        f"{tool_result.output or tool_result.error}\n\n"
                        "Continue your analysis incorporating this information."
                    )
                messages.append(Message(
                    role="assistant",
                    content=_strip_tool_call_for_history(
                        current_output, allowed_tool_names,
                    ),
                ))
                messages.append(Message(role="user", content=followup))

        result_out["output"] = final_output

    # ------------------------------------------------------------------
    # Streaming auto-search
    # ------------------------------------------------------------------

    async def _stream_auto_search(
        self,
        engine: AnalyticalEngine,
        model: str,
        query: str,
        pipeline: list[str],
        complexity: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Run the auto-search phase with streaming progress events.

        1. Direct-fetches any URLs found in the query (no LLM needed).
        2. Generates search queries via a small LLM call.
        3. Executes all queries in parallel against web_search.
        4. Emits phase chunks so the UI shows SEARCH progress.
        """
        # Phase start
        yield _phase_chunk(model, "SEARCH", "running", pipeline, complexity=complexity)

        # Step 0: Direct URL fetch — if the query contains URLs, fetch them
        # immediately via web_fetch tool instead of searching *about* them.
        # Same curation rule as auto-search above: don't pull in web_fetch
        # if the user didn't enable it.
        web_fetch_allowed = (
            self._enabled_tools is None
            or "web_fetch" in self._enabled_tools
        )
        if settings.search_direct_fetch_enabled and web_fetch_allowed:
            from augmentum.tools.web import _extract_urls

            fetch_tool = self._tool_registry.get("web_fetch") if self._tool_registry else None
            urls = _extract_urls(query)
            if urls and fetch_tool:
                url_list = "\n".join(f"  → {u}" for u in urls)
                yield _phase_content_chunk(
                    model, "SEARCH", complexity,
                    f"Fetching {len(urls)} URL(s) directly:\n{url_list}\n",
                )

                tc_before = len(engine.state.tool_calls)
                fetched_parts = []
                for url in urls[:3]:  # cap at 3 URLs
                    result = await fetch_tool.execute(
                        url=url,
                        max_chars=settings.search_direct_fetch_max_chars,
                    )
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
                    engine._state.search_context = "\n\n".join(fetched_parts)
                    engine._state.search_result_count += len(fetched_parts)
                    yield _phase_content_chunk(
                        model, "SEARCH", complexity,
                        f"Fetched {len(fetched_parts)}/{len(urls)} page(s).\n\n",
                    )
                else:
                    yield _phase_content_chunk(
                        model, "SEARCH", complexity,
                        "Could not fetch URL content. Falling back to search.\n\n",
                    )

                # Strip URLs from query for search query generation
                import re as _re
                search_query = _re.sub(r"https?://\S+", "", query).strip()
                if not search_query or len(search_query) < 5:
                    # Query was URL-only — skip normal search
                    yield _phase_chunk(
                        model, "SEARCH", "complete", pipeline,
                        complexity=complexity,
                        tool_calls=_extract_tool_calls(engine, tc_before) or None,
                    )
                    return
                query = search_query

        # Step 1: Generate queries
        yield _phase_content_chunk(model, "SEARCH", complexity, "Generating search queries...\n")
        queries = await engine._generate_search_queries(
            model, query, num_queries=settings.uarf_auto_search_queries,
            conversation_context=engine._state.conversation_context,
        )
        queries_text = "\n".join(f"  • {q}" for q in queries)
        yield _phase_content_chunk(
            model, "SEARCH", complexity, f"Queries:\n{queries_text}\n\nSearching...\n",
        )

        # Step 2: Execute searches (already records ToolCallRecords in state)
        tc_start = len(engine.state.tool_calls)
        await engine._execute_auto_search(
            queries,
            results_per_query=settings.uarf_auto_search_results_per_query,
            max_context_chars=settings.uarf_auto_search_max_context_chars,
        )

        # Emit tool call records for the UI
        new_calls = _extract_tool_calls(engine, tc_start)
        num_results = sum(1 for tc in new_calls if tc.get("success"))
        yield _phase_content_chunk(
            model, "SEARCH", complexity,
            f"Found results from {num_results}/{len(queries)} searches.\n",
        )

        # System-level search retry: too few usable results
        if (
            engine._state.search_result_count < settings.uarf_search_retry_min_results
            and engine._state.search_retry_count < settings.uarf_search_retry_max
        ):
            engine._state.search_retry_count += 1
            broadened = AnalyticalEngine._broaden_queries(
                engine._state.search_queries, query,
            )
            if broadened:
                yield _phase_content_chunk(
                    model, "SEARCH", complexity,
                    f"Insufficient results ({engine._state.search_result_count}). "
                    "Retrying with broader queries...\n",
                )
                broadened_text = "\n".join(f"  • {q}" for q in broadened)
                yield _phase_content_chunk(
                    model, "SEARCH", complexity,
                    f"Retry queries:\n{broadened_text}\n\nSearching...\n",
                )

                retry_tc_start = len(engine.state.tool_calls)
                existing_context = engine._state.search_context
                await engine._execute_auto_search(
                    broadened,
                    results_per_query=settings.uarf_auto_search_results_per_query,
                    max_context_chars=settings.uarf_auto_search_max_context_chars,
                )
                new_context = engine._state.search_context
                engine._state.search_context = existing_context
                engine._merge_search_context(new_context)

                retry_calls = _extract_tool_calls(engine, retry_tc_start)
                retry_results = sum(1 for tc in retry_calls if tc.get("success"))
                yield _phase_content_chunk(
                    model, "SEARCH", complexity,
                    f"Retry found results from {retry_results}/{len(broadened)} searches.\n",
                )
                # Combine all tool calls for phase-complete event
                new_calls = _extract_tool_calls(engine, tc_start)

        # Phase complete
        yield _phase_chunk(
            model, "SEARCH", "complete", pipeline,
            complexity=complexity,
            tool_calls=new_calls or None,
        )

    # ------------------------------------------------------------------
    # Main streaming entry point
    # ------------------------------------------------------------------

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Process a streaming analytical request with real-time phase output.

        Streams ALL phases to the UI in real-time. Phase reasoning content
        is carried in ``augmentum.phase_content_delta`` metadata so the UI
        can display it in the inline thinking block. Tool call events are
        emitted as they happen. The CONCLUDE phase streams directly to the
        response body as the final answer.

        If a reasoning flow is available (from the FlowStore), delegates to
        the flow executor instead of the hardcoded UARF pipeline.
        """
        import asyncio

        # Per-turn search dedup shared across every UARF phase + auto-search of
        # this turn (turn_search_dedup): the same image/video/page returned in
        # one phase is remembered so later phases surface only NEW results.
        # Auto-search runs its queries via asyncio.gather, whose tasks capture
        # this context, so they share the one dedup object installed here.
        from augmentum.tools.turn_search_dedup import TurnSearchDedup, set_turn_dedup
        set_turn_dedup(TurnSearchDedup())

        has_v, v_instruction, cleaned = extract_v_command(request)
        image_task = None
        if has_v and self._image_enabled and self._image_queue:
            image_task = asyncio.create_task(
                generate_direct_image(v_instruction, self._image_queue, self._session_id, user_id=self._user_id)
            )
            request = cleaned

        # Knowledge library context (Discovery Engine)
        try:
            from augmentum.discovery import (
                inject_system_context,
                retrieve_knowledge_context,
            )
            _user_query = request.messages[-1].content if request.messages else ""
            if _user_query and isinstance(_user_query, str):
                _knowledge_ctx = await retrieve_knowledge_context(
                    request.app.state, _user_query,
                    user_id=self._user_id,
                )
                if _knowledge_ctx:
                    inject_system_context(request.messages, _knowledge_ctx)
        except Exception:
            log.debug("analytical_knowledge_retrieve_failed", exc_info=True)

        engine = AnalyticalEngine(
            self._backend,
            tool_registry=self._tool_registry,
            prompt_cache=self._prompt_cache,
            provider_registry=self._provider_registry,
            circuit_breaker=self._circuit_breaker,
            enabled_tools=self._enabled_tools,
            user_id=self._user_id,
        )

        query = engine._extract_query(request)
        engine._state.query = query
        model = request.model

        # Extract user's personalization / custom system prompt (if any).
        # The UI prepends it to the system message; we thread it into each
        # phase so the LLM sees it alongside the UARF instructions.
        user_system = ""
        if request.messages and request.messages[0].role == "system":
            user_system = request.messages[0].content.strip()

        # Build conversation context from prior messages
        conv_raw = AnalyticalEngine._build_conversation_context(
            request,
            max_turns=settings.uarf_conversation_turns,
            max_chars=settings.uarf_conversation_max_chars,
        )
        conversation_context = ""
        if conv_raw:
            conversation_context = (
                "\n## Conversation History\n"
                "The following is the recent conversation for context. "
                "Use it to resolve references (e.g. 'that', 'it', 'the same') "
                "but focus your analysis on the current query.\n\n"
                f"{conv_raw}\n"
            )
            engine._state.conversation_context = conversation_context

        # --- Try flow-based execution if a flow store is available ---
        if self._flow_store:
            try:
                from augmentum.reasoning.executor import execute_flow_stream
                from augmentum.reasoning.resolver import resolve_flow

                flow = await resolve_flow(
                    self._flow_store, model=model, query=query,
                    explicit_flow_id=self._explicit_flow_id,
                    user_id=self._user_id,
                )
                if flow:
                    log.info("using_reasoning_flow", flow=flow.name, steps=len(flow.steps))
                    async for chunk in execute_flow_stream(
                        flow, engine, self._backend, model, query,
                        tool_registry=self._tool_registry,
                        provider_registry=self._provider_registry,
                        conversation_context=conversation_context,
                        search_context=engine._state.search_context,
                        user_system=user_system,
                        flow_tune=self._flow_tune,
                    ):
                        yield chunk

                    if image_task:
                        image_url = await image_task
                        if image_url:
                            yield InternalStreamChunk(
                                content_delta=f"\n\n![Generated Image]({image_url})",
                            )
                    return
            except Exception:
                log.warning("flow_execution_failed_falling_back", exc_info=True)

        # --- Fallback: hardcoded UARF pipeline ---

        # --- Phase 1: ASSESS (always first, determines pipeline) ---
        yield _phase_chunk(model, "ASSESS", "running", _FULL_PHASES)
        yield InternalStreamChunk(content_delta="", role="assistant", model=model)

        # Try heuristic first — skip LLM call for unambiguous queries
        heuristic_complexity = (
            AnalyticalEngine._heuristic_assess(query)
            if settings.uarf_heuristic_assess else None
        )

        if heuristic_complexity is not None:
            assess_output = (
                "TYPE: heuristic\n"
                "DOMAIN: general\n"
                "REASONING_STEPS: 1\n"
                f"COMPLEXITY: {heuristic_complexity}\n"
                "RATIONALE: Determined by surface-level heuristic (no LLM call)."
            )
            complexity = heuristic_complexity
            yield _phase_content_chunk(
                model, "ASSESS", "",
                f"Heuristic: {heuristic_complexity} complexity\n",
            )
            log.info(
                "stream_heuristic_assess",
                complexity=heuristic_complexity,
            )
        else:
            system_prompt, user_content = get_phase_prompt(
                "assess", query=query,
                conversation_context=engine._state.conversation_context,
            )
            result: dict = {}
            async for chunk in self._stream_phase(
                engine, AnalyticalPhase.ASSESS, model, _FULL_PHASES, "",
                system_prompt=system_prompt, query=user_content, result_out=result,
                user_system=user_system,
            ):
                yield chunk

            assess_output = result["output"]
            complexity = AnalyticalEngine._parse_complexity(assess_output)

        engine._state.complexity = complexity
        engine._state.phase_results[AnalyticalPhase.ASSESS.value] = PhaseResult(
            phase=AnalyticalPhase.ASSESS, output=assess_output, tokens_used=0,
        )
        is_simple = complexity == "simple"

        # Detect if auto-search is needed.
        # Respect the user's tool curation: if ``_enabled_tools`` was provided
        # and ``web_search`` is not in it, don't silently fire web_search
        # behind their back. ``_enabled_tools is None`` means "no curation
        # specified" (legacy callers), and we keep the legacy behavior.
        web_search_allowed = (
            self._enabled_tools is None
            or "web_search" in self._enabled_tools
        )
        needs_search = (
            settings.uarf_auto_search
            and self._tool_registry is not None
            and self._tool_registry.get("web_search") is not None
            and web_search_allowed
            and AnalyticalEngine._needs_search(query, assess_output)
        )

        is_moderate = complexity == "moderate"

        if needs_search:
            engine._state.needs_search = True
            if is_simple:
                pipeline = _SIMPLE_PHASES_WITH_SEARCH
            elif is_moderate:
                pipeline = _MODERATE_PHASES_WITH_SEARCH
            else:
                pipeline = _FULL_PHASES_WITH_SEARCH
        else:
            if is_simple:
                pipeline = _SIMPLE_PHASES
            elif is_moderate:
                pipeline = _MODERATE_PHASES
            else:
                pipeline = _FULL_PHASES

        yield _phase_chunk(model, "ASSESS", "complete", pipeline, complexity=complexity)

        # --- Auto-search phase (if triggered) ---
        if needs_search:
            async for chunk in self._stream_auto_search(
                engine, model, query, pipeline, complexity,
            ):
                yield chunk

        tc_cursor = len(engine.state.tool_calls)  # skip auto-search tool records

        if is_simple:
            # ---- Simple path: single merged RESPOND call ----
            # Instead of APPLY (analysis) + CONCLUDE (synthesis) as two
            # separate LLM calls, we handle tools non-streaming via the
            # engine and store the respond_prompt for streaming in CONCLUDE.

            # Build respond prompt with tool support
            respond_system, respond_user = get_phase_prompt(
                "respond", query=query,
                has_tools=self._tool_registry is not None,
                search_context=engine._state.search_context,
                conversation_context=engine._state.conversation_context,
            )

            # Handle tool calls non-streaming (same as engine does)
            tool_exclude = frozenset({"web_search"}) if engine._state.needs_search else None
            tools: list = []
            if self._tool_registry:
                tools = self._tool_registry.get_for_phase(
                    "respond",
                    exclude=tool_exclude,
                    allowed_names=self._enabled_tools,
                )
                if tools and settings.tool_prefilter_enabled and engine._state.query:
                    from augmentum.tools.filter import filter_tools_for_query
                    tools = filter_tools_for_query(
                        engine._state.query, tools,
                        min_tools=settings.tool_prefilter_min_tools,
                    )

            tier = select_tier(self._backend, model) if tools else ToolCallingTier.TEXT
            native_tools: list[dict] | None = None
            structured_schema: dict | None = None

            if tools:
                if tier == ToolCallingTier.NATIVE:
                    respond_system += get_native_tool_prompt_section(tools)
                    native_tools = tools_to_native_format(tools)
                elif tier == ToolCallingTier.STRUCTURED:
                    structured_schema = build_structured_output_schema(tools)
                    respond_system += get_structured_tool_prompt_section(
                        tools, structured_schema,
                    )
                else:
                    respond_system += get_tool_prompt_section(tools)

                suggestions = AnalyticalEngine._get_proactive_suggestions(query, tools=tools)
                if suggestions:
                    respond_system += (
                        "\n\n## Proactive Suggestions\n"
                        "Based on the query, consider using these tools:\n"
                        + "\n".join(f"- {s}" for s in suggestions)
                    )

            respond_messages = [
                Message(role="system", content=respond_system),
                Message(role="user", content=respond_user),
            ]

            # Non-streaming tool loop
            allowed_tool_names = {t.name for t in tools}
            tool_calls_made = 0
            while tools and tool_calls_made < _MAX_TOOL_CALLS_PER_PHASE:
                response = await self._call_llm(
                    model, respond_messages,
                    tools=native_tools, format=structured_schema,
                )
                output = response.message.content if response.message else ""

                # Parse tool call
                tool_name = ""
                tool_input: dict = {}
                if tier == ToolCallingTier.NATIVE:
                    parsed = parse_native_tool_call(response)
                    if parsed:
                        tool_name, tool_input = parsed
                elif tier == ToolCallingTier.STRUCTURED:
                    parsed = parse_structured_output(output)
                    if parsed:
                        tool_name, tool_input = parsed
                    elif output:
                        output = extract_structured_text(output)
                        break
                else:
                    tool_name, tool_input = AnalyticalEngine._parse_tool_call(output)

                if not tool_name:
                    # Python-style first (primary text format)
                    py_parsed = parse_python_style_tool_call(
                        output, known_tools=allowed_tool_names,
                    )
                    if py_parsed:
                        tool_name, tool_input = py_parsed
                        log.info(
                            "respond_tool_python_style_detected",
                            tool=tool_name,
                        )
                    # Legacy TOOL_CALL: format fallback
                    if not tool_name:
                        fallback_name, fallback_input = AnalyticalEngine._parse_tool_call(output)
                        if fallback_name:
                            tool_name, tool_input = fallback_name, fallback_input
                    if not tool_name:
                        break

                # Resolve and execute
                resolved = self._tool_registry.resolve(tool_name) if self._tool_registry else None
                if not resolved or resolved.name not in allowed_tool_names:
                    break

                tool_input = coerce_tool_params(resolved, tool_input)
                tool_result = await engine._execute_tool(
                    "respond", tool_name, tool_input, exclude=tool_exclude,
                )
                if not tool_result.validation_error:
                    tool_calls_made += 1

                # Emit tool event to reasoning panel
                yield _phase_content_chunk(
                    model, "APPLY", complexity,
                    f"Tool: {resolved.name} → "
                    f"{'OK' if tool_result.success else 'Error'}\n",
                )

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
                        "Continue and provide your final answer."
                    )
                respond_messages.append(Message(role="assistant", content=output))
                respond_messages.append(Message(role="user", content=followup))

            new_tools = _extract_tool_calls(engine, tc_cursor)
            tc_cursor = len(engine.state.tool_calls)
            yield _phase_chunk(
                model, "APPLY", "complete", pipeline,
                complexity=complexity, tool_calls=new_tools or None,
            )

            # Store the respond messages for streaming in CONCLUDE
            engine._respond_messages = respond_messages
            engine._respond_native_tools = native_tools
            engine._respond_schema = structured_schema
        else:
            # ---- Non-simple path ----
            # Moderate: GATHER → APPLY → VERIFY
            # Complex:  IDENTIFY → RELEVANT → APPLY → VERIFY

            if is_moderate:
                # --- GATHER (merged IDENTIFY + RELEVANT) ---
                yield _phase_chunk(
                    model, "GATHER", "running", pipeline, complexity=complexity,
                )

                system_prompt, user_content = get_phase_prompt(
                    "gather", query=query,
                    has_tools=self._tool_registry is not None,
                    search_context=engine._state.search_context,
                    conversation_context=engine._state.conversation_context,
                )
                result = {}
                async for chunk in self._stream_phase(
                    engine, AnalyticalPhase.GATHER, model, pipeline, complexity,
                    system_prompt=system_prompt, query=user_content,
                    enable_tools=True, result_out=result,
                    user_system=user_system,
                ):
                    yield chunk

                gather_output = result["output"]
                engine._state.phase_results[AnalyticalPhase.GATHER.value] = PhaseResult(
                    phase=AnalyticalPhase.GATHER, output=gather_output, tokens_used=0,
                )
                # Mirror to IDENTIFY slot for downstream compatibility
                engine._state.phase_results[AnalyticalPhase.IDENTIFY.value] = PhaseResult(
                    phase=AnalyticalPhase.IDENTIFY, output=gather_output, tokens_used=0,
                )
                identify_output = gather_output
                relevant_output = ""

                new_tools = _extract_tool_calls(engine, tc_cursor)
                tc_cursor = len(engine.state.tool_calls)
                yield _phase_chunk(
                    model, "GATHER", "complete", pipeline,
                    complexity=complexity, tool_calls=new_tools or None,
                )
            else:
                # --- IDENTIFY ---
                yield _phase_chunk(
                    model, "IDENTIFY", "running", pipeline, complexity=complexity,
                )

                system_prompt, user_content = get_phase_prompt(
                    "identify", query=query, assess_output=assess_output,
                    search_context=engine._state.search_context,
                    conversation_context=engine._state.conversation_context,
                )
                result = {}
                async for chunk in self._stream_phase(
                    engine, AnalyticalPhase.IDENTIFY, model, pipeline, complexity,
                    system_prompt=system_prompt, query=user_content, result_out=result,
                    user_system=user_system,
                ):
                    yield chunk

                identify_output = result["output"]
                engine._state.phase_results[AnalyticalPhase.IDENTIFY.value] = PhaseResult(
                    phase=AnalyticalPhase.IDENTIFY, output=identify_output, tokens_used=0,
                )
                yield _phase_chunk(
                    model, "IDENTIFY", "complete", pipeline, complexity=complexity,
                )

                # --- RELEVANT (with tools) ---
                yield _phase_chunk(
                    model, "RELEVANT", "running", pipeline, complexity=complexity,
                )

                system_prompt, user_content = get_phase_prompt(
                    "relevant", query=query, identify_output=identify_output,
                    has_tools=self._tool_registry is not None,
                    search_context=engine._state.search_context,
                    conversation_context=engine._state.conversation_context,
                )
                result = {}
                async for chunk in self._stream_phase(
                    engine, AnalyticalPhase.RELEVANT, model, pipeline, complexity,
                    system_prompt=system_prompt, query=user_content,
                    enable_tools=True, result_out=result,
                    user_system=user_system,
                ):
                    yield chunk

                relevant_output = result["output"]
                engine._state.phase_results[AnalyticalPhase.RELEVANT.value] = PhaseResult(
                    phase=AnalyticalPhase.RELEVANT, output=relevant_output, tokens_used=0,
                )
                new_tools = _extract_tool_calls(engine, tc_cursor)
                tc_cursor = len(engine.state.tool_calls)
                yield _phase_chunk(
                    model, "RELEVANT", "complete", pipeline,
                    complexity=complexity, tool_calls=new_tools or None,
                )

            # --- APPLY (with tools) ---
            yield _phase_chunk(
                model, "APPLY", "running", pipeline, complexity=complexity,
            )

            backtrack_context = ""
            system_prompt, user_content = get_phase_prompt(
                "apply", query=query,
                identify_output=identify_output,
                relevant_output=relevant_output,
                backtrack_context=backtrack_context,
                has_tools=self._tool_registry is not None,
                search_context=engine._state.search_context,
                conversation_context=engine._state.conversation_context,
            )
            result = {}
            async for chunk in self._stream_phase(
                engine, AnalyticalPhase.APPLY, model, pipeline, complexity,
                system_prompt=system_prompt, query=user_content,
                enable_tools=True, result_out=result,
                user_system=user_system,
            ):
                yield chunk

            apply_output = result["output"]
            engine._state.phase_results[AnalyticalPhase.APPLY.value] = PhaseResult(
                phase=AnalyticalPhase.APPLY, output=apply_output, tokens_used=0,
            )
            new_tools = _extract_tool_calls(engine, tc_cursor)
            tc_cursor = len(engine.state.tool_calls)
            yield _phase_chunk(
                model, "APPLY", "complete", pipeline,
                complexity=complexity, tool_calls=new_tools or None,
            )

            # VERIFY phase removed from user-visible path. Training-cutoff bias
            # caused the LLM reviewer to flag real search results as fabricated
            # (any date past the model's cutoff triggered "VERIFIED: no"), and
            # its output was either fed back into a backtrack loop that
            # rewrote APPLY or injected into CONCLUDE — poisoning the final
            # answer. Auto-verify (math/code tool checks) also gated on that
            # same flow; removed with it for now. If a background telemetry
            # verify is wanted later, kick it off here as a fire-and-forget
            # task whose result never touches APPLY or CONCLUDE.

        # --- CONCLUDE (streamed to response body as the final answer) ---

        if is_simple:
            # Simple path: stream the single merged response directly.
            yield _phase_chunk(
                model, "CONCLUDE", "running", pipeline, complexity=complexity,
            )
            respond_messages = getattr(engine, "_respond_messages", None)
            if respond_messages:
                if user_system and respond_messages and respond_messages[0].role == "system":
                    respond_messages[0] = Message(
                        role="system",
                        content=f"{user_system}\n\n{respond_messages[0].content}",
                    )
                conclude_request = InternalChatRequest(
                    model=model,
                    messages=respond_messages,
                    stream=True,
                )
                conclude_output_parts: list[str] = []
                async for chunk in self._backend.chat_stream(conclude_request):
                    if chunk.content_delta:
                        conclude_output_parts.append(chunk.content_delta)
                    yield chunk

                conclude_text = "".join(conclude_output_parts)
                engine._state.phase_results[AnalyticalPhase.APPLY.value] = PhaseResult(
                    phase=AnalyticalPhase.APPLY, output=conclude_text, tokens_used=0,
                )
                engine._state.phase_results[AnalyticalPhase.CONCLUDE.value] = PhaseResult(
                    phase=AnalyticalPhase.CONCLUDE, output=conclude_text, tokens_used=0,
                )
            yield _phase_chunk(
                model, "CONCLUDE", "complete", pipeline, complexity=complexity,
            )

        elif is_moderate:
            # Moderate path: APPLY output is the answer — use a lightweight
            # CONCLUDE call to polish it into a natural response.
            yield _phase_chunk(
                model, "CONCLUDE", "running", pipeline, complexity=complexity,
            )
            apply_output = engine._get_phase_output(AnalyticalPhase.APPLY)

            system_prompt, user_content = get_phase_prompt(
                "conclude",
                query=query,
                apply_output=apply_output,
                is_simple=True,  # use the shorter conclude prompt
                conversation_context=engine._state.conversation_context,
            )

            if user_system:
                system_prompt = f"{user_system}\n\n{system_prompt}"
            conclude_request = InternalChatRequest(
                model=model,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_content),
                ],
                stream=True,
            )
            async for chunk in self._backend.chat_stream(conclude_request):
                yield chunk

            yield _phase_chunk(
                model, "CONCLUDE", "complete", pipeline, complexity=complexity,
            )

        else:
            # Complex path: final synthesis from APPLY output only.
            yield _phase_chunk(
                model, "CONCLUDE", "running", pipeline, complexity=complexity,
            )
            apply_output = engine._get_phase_output(AnalyticalPhase.APPLY)

            system_prompt, user_content = get_phase_prompt(
                "conclude",
                query=query,
                apply_output=apply_output,
                is_simple=False,
                conversation_context=engine._state.conversation_context,
            )

            if user_system:
                system_prompt = f"{user_system}\n\n{system_prompt}"
            conclude_request = InternalChatRequest(
                model=model,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_content),
                ],
                stream=True,
            )
            async for chunk in self._backend.chat_stream(conclude_request):
                yield chunk

            yield _phase_chunk(
                model, "CONCLUDE", "complete", pipeline, complexity=complexity,
            )

        # Append generated image after CONCLUDE if /v was detected
        if image_task:
            image_url = await image_task
            if image_url:
                yield InternalStreamChunk(
                    content_delta=f"\n\n![Generated Image]({image_url})",
                )
