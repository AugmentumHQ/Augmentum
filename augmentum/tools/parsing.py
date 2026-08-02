"""Universal tool call parsing, schema injection, and execution.

Provides a single entry point for all handlers (passthrough, analytical,
agentic) to parse tool calls from LLM responses, inject tool schemas into
requests, and execute tools with type coercion.

The parser runs a tiered waterfall:

1. **Native** — ``tool_calls`` field (OpenAI function calling)
2. **Structured** — JSON schema-constrained output (Ollama ``format``)
3. **Text fallbacks** (always checked as fallbacks):
   a. ``TOOL_CALL: / TOOL_INPUT:`` blocks (multi-block)
   b. JSON array ``[{"name": ..., "arguments": ...}]``
   c. Python-style ``tool_name(arg="val")``
   d. XML ``<function=name>{...}</function>`` (multi-block)
   e. ReAct ``Action: name / Action Input: {...}``

A **parser affinity cache** remembers which parser last succeeded for
each model and tries it first on subsequent calls.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import (
        InternalChatRequest,
        InternalChatResponse,
        ModelBackend,
    )
    from augmentum.tools.base import Tool
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

# Matches <function=name>{...}</function> blocks (Hermes/vLLM style).
# Group 1: function name, Group 2: body.
_XML_FUNCTION_RE = re.compile(
    r"<function=([^>]+)>(.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ParsedToolCall:
    """A single parsed tool call from an LLM response."""

    name: str
    args: dict = field(default_factory=dict)
    call_id: str = ""
    parser: str = ""  # which parser matched (for diagnostics)

    def __post_init__(self) -> None:
        if not self.call_id:
            self.call_id = f"call_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Parser affinity cache
# ---------------------------------------------------------------------------

_parser_affinity: dict[str, str] = {}


def get_parser_affinity() -> dict[str, str]:
    """Return a copy of the affinity cache (for diagnostics/testing)."""
    return dict(_parser_affinity)


def clear_parser_affinity() -> None:
    """Reset the affinity cache (useful in tests)."""
    _parser_affinity.clear()


# ---------------------------------------------------------------------------
# Public API: parse_tool_calls
# ---------------------------------------------------------------------------


def parse_tool_calls(
    response: InternalChatResponse,
    tools: list[Tool],
    backend: ModelBackend,
) -> list[ParsedToolCall]:
    """Parse tool calls from an LLM response using the full parser waterfall.

    Uses parser affinity caching: if a parser previously succeeded for this
    model, tries it first.  On cache miss, falls back to the full waterfall
    and updates the cache on success.

    Parameters
    ----------
    response : InternalChatResponse
        The LLM response to parse.
    tools : list[Tool]
        Available tools (used for name validation).
    backend : ModelBackend
        The backend that produced the response (used for tier selection).

    Returns
    -------
    list[ParsedToolCall]
        Parsed tool calls, empty list if none found.
    """
    from augmentum.modes.analytical.tool_calling import (
        ToolCallingTier,
        select_tier,
    )

    tier = select_tier(backend, response.model or "")
    model = response.model or ""
    known = {t.name for t in tools}
    text = response.message.content if response.message else ""

    # Build ordered parser list: (key, callable) → returns list or None
    parsers: list[tuple[str, Any]] = []

    if tier == ToolCallingTier.NATIVE:
        parsers.append(("native", lambda: _parse_native(response)))
    if tier == ToolCallingTier.STRUCTURED:
        parsers.append(("structured", lambda: _parse_structured(text)))

    # Text-based parsers — always available as fallbacks
    parsers.extend([
        ("tool_call_text", lambda: _parse_text_blocks(text, known)),
        ("json_array", lambda: _parse_json_array(text, known)),
        ("python_style", lambda: _parse_python_style(text, known)),
        ("xml_function", lambda: _parse_xml_multi(text, known)),
        ("react", lambda: _parse_react(text, known)),
        # Last resort: bare JSON args without a tool name wrapper.
        # Infers the tool by matching JSON keys against tool input schemas.
        ("bare_json_args", lambda: _parse_bare_json_args(text, tools)),
    ])

    # Affinity: try cached parser first
    cached_key = _parser_affinity.get(model)
    if cached_key:
        for key, fn in parsers:
            if key == cached_key:
                result = fn()
                if result:
                    return result
                break

    # Full waterfall
    for key, fn in parsers:
        if key == cached_key:
            continue
        result = fn()
        if result:
            _parser_affinity[model] = key
            log.info("parser_affinity_set", model=model, parser=key)
            return result

    return []


# ---------------------------------------------------------------------------
# Public API: inject_tool_schemas / clear_tool_schemas
# ---------------------------------------------------------------------------


_TOOL_RESTRAINT = (
    "Choose the right approach for each query:\n"
    "- web_search: current events, recent news, real-time data\n"
    "- wikipedia: factual topics, history, biographies, science, definitions\n"
    "- calculator: math expressions and numeric computation\n"
    "- web_fetch: when the user shares a URL to read\n"
    "- image_generation: when the user asks for a picture or visual\n"
    "- Direct answer (no tools): opinions, greetings, creative writing, common knowledge"
)


def inject_tool_schemas(
    request: InternalChatRequest,
    tools: list[Tool],
    backend: ModelBackend,
) -> str:
    """Inject tool schemas into a request based on the detected tier.

    Also injects a restraint instruction so the model doesn't over-eagerly
    call tools for queries it can answer directly.

    Returns the tier name (``"native"``, ``"structured"``, or ``"text"``).
    """
    from augmentum.models.base import Message
    from augmentum.modes.analytical.tool_calling import (
        ToolCallingTier,
        build_structured_output_schema,
        select_tier,
        tools_to_native_format,
    )

    tier = select_tier(backend, request.model)

    if tier == ToolCallingTier.NATIVE:
        request.tools = tools_to_native_format(tools)
    elif tier == ToolCallingTier.STRUCTURED:
        request.format = build_structured_output_schema(tools)  # type: ignore[assignment]
    else:
        prompt = build_text_tool_prompt(tools)
        if request.messages:
            last = request.messages[-1]
            request.messages[-1] = Message(
                role=last.role,
                content=f"{last.content}\n\n{prompt}",
                images=last.images,
            )

    # Inject restraint guidance into system prompt so the model doesn't
    # call tools for simple queries it can answer from knowledge.
    if request.messages and request.messages[0].role == "system":
        request.messages[0] = Message(
            role="system",
            content=f"{request.messages[0].content}\n\n{_TOOL_RESTRAINT}",
            images=request.messages[0].images,
        )
    else:
        request.messages.insert(0, Message(role="system", content=_TOOL_RESTRAINT))

    return tier.value


def clear_tool_schemas(request: InternalChatRequest) -> None:
    """Remove tool schemas from a request so the LLM responds freely."""
    request.tools = None
    request.format = None  # type: ignore[assignment]


def build_text_tool_prompt(tools: list[Tool]) -> str:
    """Build a Tier 3 text prompt listing available tools."""
    lines = [
        "You have access to the following tools. To use one, respond with:",
        "TOOL_CALL: tool_name",
        'TOOL_INPUT: {"param": "value"}',
        "",
        "You may call multiple tools in one response by repeating the format.",
        "",
        "Available tools:",
    ]
    for t in tools:
        schema = t.input_schema or {}
        params = schema.get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'string')}" for k, v in params.items()
        )
        lines.append(f"- {t.name}({param_str}): {t.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API: coerce_and_execute
# ---------------------------------------------------------------------------


async def coerce_and_execute(
    tool: Tool,
    args: dict,
    *,
    registry: ToolRegistry | None = None,
    timeout: float | None = None,
) -> tuple[str, dict]:
    """Coerce parameter types, execute a tool, and return (output, metadata).

    Handles type coercion, timeout, error capture, and output truncation.
    """
    import asyncio

    from augmentum.modes.analytical.tool_calling import coerce_tool_params

    args = coerce_tool_params(tool, args)
    timeout = timeout or tool.timeout
    metadata: dict = {}
    import time
    start = time.monotonic()

    try:
        result = await asyncio.wait_for(tool.execute(**args), timeout=timeout)
        elapsed = (time.monotonic() - start) * 1000
        if registry:
            registry.metrics.record(tool.name, success=result.success, elapsed_ms=elapsed)
        output = result.output if result.success else f"Error: {result.error}"
        metadata = result.metadata or {}
    except TimeoutError:
        elapsed = (time.monotonic() - start) * 1000
        if registry:
            registry.metrics.record(tool.name, success=False, elapsed_ms=elapsed)
        output = f"Error: Tool '{tool.name}' timed out after {timeout}s"
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        if registry:
            registry.metrics.record(tool.name, success=False, elapsed_ms=elapsed)
        output = f"Error: {exc}"

    # Truncate
    max_chars = settings.tool_result_max_chars
    if len(output) > max_chars:
        tail = settings.tool_result_truncation_tail
        output = output[:max_chars - tail] + "\n...\n" + output[-tail:]

    return output, metadata


# ---------------------------------------------------------------------------
# Internal parsers — each returns list[ParsedToolCall] or None
# ---------------------------------------------------------------------------


def _parse_native(response: InternalChatResponse) -> list[ParsedToolCall] | None:
    from augmentum.modes.analytical.tool_calling import parse_native_tool_calls_all

    raw_calls = parse_native_tool_calls_all(response)
    if not raw_calls:
        return None
    raw_tc = (response.message.tool_calls or []) if response.message else []
    results = []
    for i, (name, args) in enumerate(raw_calls):
        tc_id = ""
        if i < len(raw_tc):
            tc_id = raw_tc[i].get("id", "") or ""
        results.append(ParsedToolCall(
            name=name, args=args,
            call_id=tc_id or f"call_{uuid.uuid4().hex[:8]}",
            parser="native",
        ))
    return results


def _parse_structured(text: str) -> list[ParsedToolCall] | None:
    from augmentum.modes.analytical.tool_calling import parse_structured_output

    parsed = parse_structured_output(text)
    if not parsed:
        return None
    name, args = parsed
    return [ParsedToolCall(name=name, args=args, parser="structured")]


def _parse_text_blocks(text: str, known: set[str]) -> list[ParsedToolCall] | None:
    """Parse TOOL_CALL/TOOL_INPUT blocks — supports multiple in one response.

    Also handles the hybrid format where models write
    ``TOOL_CALL: tool_name(arg: "value")`` by falling back to the
    python-style parser per block.
    """
    if not text:
        return None
    from augmentum.modes.analytical.engine import AnalyticalEngine
    from augmentum.modes.analytical.tool_calling import parse_python_style_tool_call

    blocks = re.split(
        r'(?=(?:^|\n)\s*(?:\*{0,2})TOOL[_ ]?CALL)',
        text, flags=re.IGNORECASE,
    )
    results: list[ParsedToolCall] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        tc_name, tc_input = AnalyticalEngine._parse_tool_call(block)
        if tc_name and (not known or tc_name in known):
            results.append(ParsedToolCall(name=tc_name, args=tc_input, parser="tool_call_text"))
        else:
            py_parsed = parse_python_style_tool_call(block, known)
            if py_parsed:
                results.append(
                    ParsedToolCall(name=py_parsed[0], args=py_parsed[1], parser="python_style")
                )
    return results if results else None


def _parse_json_array(text: str, known: set[str]) -> list[ParsedToolCall] | None:
    if not text:
        return None
    from augmentum.modes.analytical.tool_calling import parse_json_tool_calls

    calls = parse_json_tool_calls(text, known)
    if not calls:
        return None
    return [ParsedToolCall(name=n, args=a, parser="json_array") for n, a in calls]


def _parse_python_style(text: str, known: set[str]) -> list[ParsedToolCall] | None:
    if not text:
        return None
    from augmentum.modes.analytical.tool_calling import parse_python_style_tool_call

    parsed = parse_python_style_tool_call(text, known)
    if not parsed:
        return None
    return [ParsedToolCall(name=parsed[0], args=parsed[1], parser="python_style")]


def _parse_xml_multi(text: str, known: set[str]) -> list[ParsedToolCall] | None:
    """Parse all <function=name>...</function> blocks in the text."""
    if not text:
        return None

    matches = list(_XML_FUNCTION_RE.finditer(text))
    if not matches:
        return None

    results: list[ParsedToolCall] = []
    for m in matches:
        name = m.group(1).strip()
        if known and name not in known:
            continue
        body = m.group(2).strip()
        args: dict = {}
        try:
            parsed_json = json.loads(body)
            if isinstance(parsed_json, dict):
                args = parsed_json
        except (json.JSONDecodeError, ValueError, TypeError):
            if body:
                args = {"query": body}
        results.append(ParsedToolCall(name=name, args=args, parser="xml_function"))
    return results if results else None


def _parse_bare_json_args(text: str, tools: list[Tool]) -> list[ParsedToolCall] | None:
    """Last-resort parser: match a bare JSON object to a tool by parameter names.

    Some models output the tool's arguments as a raw JSON block without any
    wrapper (no ``name``, ``TOOL_CALL:``, or function-call syntax).  We infer
    the tool by checking which tool's required parameters best match the
    JSON object's keys.

    Only matches when there's a single unambiguous best match with ≥50%
    key overlap on the tool's schema properties.
    """
    if not text or not tools:
        return None

    # Strip markdown code fences
    clean = text
    if "```" in clean:
        clean = re.sub(r"```[\w]*\n?", "", clean).replace("```", "").strip()

    # Find the first JSON object in the text
    start = clean.find("{")
    if start == -1:
        return None
    end = clean.rfind("}")
    if end <= start:
        return None

    try:
        data = json.loads(clean[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict) or not data:
        return None

    data_keys = set(data.keys())

    # Score each tool by how well the JSON keys match its schema properties
    best_tool = None
    best_score = 0.0
    for tool in tools:
        schema = tool.input_schema or {}
        props = set(schema.get("properties", {}).keys())
        if not props:
            continue
        # Overlap: how many JSON keys are valid params for this tool
        overlap = len(data_keys & props)
        if overlap == 0:
            continue
        # Score = fraction of JSON keys that are valid params
        score = overlap / len(data_keys)
        # Require at least one required param to match if the tool has required params
        required = set(schema.get("required", []))
        if required and not (data_keys & required):
            continue
        if score > best_score:
            best_score = score
            best_tool = tool

    # Require ≥50% key overlap to avoid false positives
    if not best_tool or best_score < 0.5:
        return None

    # Filter args to only include valid params for the matched tool
    valid_props = set((best_tool.input_schema or {}).get("properties", {}).keys())
    filtered_args = {k: v for k, v in data.items() if k in valid_props}

    log.info("bare_json_args_matched", tool=best_tool.name, score=f"{best_score:.0%}",
             keys=list(data_keys))
    return [ParsedToolCall(name=best_tool.name, args=filtered_args, parser="bare_json_args")]


def _parse_react(text: str, known: set[str]) -> list[ParsedToolCall] | None:
    if not text:
        return None
    from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call

    parsed = parse_action_input_tool_call(text, known)
    if not parsed:
        return None
    return [ParsedToolCall(name=parsed[0], args=parsed[1], parser="react")]
