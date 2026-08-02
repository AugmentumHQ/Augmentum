"""Three-tier tool calling fallback chain for the UARF analytical engine.

Tier 1 — Native function calling: tools API parameter + tool_calls response
Tier 2 — Structured output: Ollama format JSON Schema for grammar-constrained decoding
Tier 3 — Text-based parsing: TOOL_CALL: / TOOL_INPUT: regex (existing approach)

Selection is automatic based on backend type and model capabilities, with an
optional config override for testing (``AUGMENTUM_UARF_TOOL_TIER_OVERRIDE``).
"""

from __future__ import annotations

import contextlib
import json
import re
from enum import Enum
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.tools.params import POSITIONAL_GUESS_KEY
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatResponse, ModelBackend
    from augmentum.tools.base import Tool

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class ToolCallingTier(str, Enum):
    NATIVE = "native"        # Tier 1: tools API parameter
    STRUCTURED = "structured"  # Tier 2: Ollama format JSON Schema
    TEXT = "text"            # Tier 3: TOOL_CALL: regex parsing


# ---------------------------------------------------------------------------
# Model families known to support native function calling via Ollama
# ---------------------------------------------------------------------------
# Deliberately conservative — better to fall back to structured output
# than to send `tools` to a model that ignores them.

_NATIVE_TOOL_FAMILIES: tuple[str, ...] = (
    "qwen",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "llama4",
    "mistral-nemo",
    "mistral-small",
    "mistral-large",
    "command-r",
    "firefunction",
    "hermes",
    "nemotron",
    "granite",
    # 2026-04-22: expanded for LM Studio. These all ship chat
    # templates with proper tool-call rendering (verified against
    # LM Studio's bundled jinja templates) — without them, weak/local
    # models drop to TEXT tier and the validation_error_streak
    # detector fires from text-extraction failures even though the
    # model would emit valid tool_calls under the API's native path.
    "deepseek",       # deepseek-r1, deepseek-coder-v2, deepseek-v3
    "glm",            # GLM-4-9B / 32B+ (chatglm3 doesn't qualify)
    "gpt-oss",        # OpenAI's open-weight gpt-oss family
    "minimax",        # minimax-m1, minimax-m2, minimax-m2.5
    "phi-4",          # phi-4 (phi-3 doesn't, intentionally narrow)
    "yi-",            # yi-coder-9b, yi-1.5-34b-chat (hyphen suffix)
)

# Cloud API providers known to support native function calling.
# Matched against the OpenAIBackend base_url hostname.
_NATIVE_CLOUD_HOSTS: tuple[str, ...] = (
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "api.deepseek.com",
    "api.anthropic.com",
    "openrouter.ai",
    "api.together.xyz",
    "api.groq.com",
    "api.mistral.ai",
    "api.fireworks.ai",
    "api.perplexity.ai",
    "api.cohere.ai",
    "api.cohere.com",
)


# ---------------------------------------------------------------------------
# Tier selection
# ---------------------------------------------------------------------------


def select_tier(backend: ModelBackend, model_name: str) -> ToolCallingTier:
    """Select the best tool-calling tier for a backend + model combination.

    Returns the highest-reliability tier the backend/model can support.
    Respects ``settings.uarf_tool_tier_override`` when set.
    """
    # Config override — useful for testing and debugging
    override = settings.uarf_tool_tier_override
    if override:
        try:
            tier = ToolCallingTier(override)
            log.info("tool_tier_override", tier=tier.value)
            return tier
        except ValueError:
            log.warning("tool_tier_override_invalid", value=override)

    # Import backend classes lazily to avoid circular imports
    from augmentum.models.engine import AugmentumEngineBackend
    from augmentum.models.llama_cpp import LlamaCppBackend
    from augmentum.models.ollama import OllamaBackend
    from augmentum.models.openai_compat import OpenAIBackend

    if isinstance(backend, OpenAIBackend):
        # Cloud APIs (OpenAI, Gemini, etc.) support native function calling.
        # Local OpenAI-compatible servers (LM Studio, text-generation-webui, etc.)
        # usually don't — their models see tool schemas but can't produce
        # proper tool_calls responses.
        if _is_cloud_api(getattr(backend, "_base_url", "")):
            return ToolCallingTier.NATIVE
        # Local OpenAI-compat server — check model family, fall back to TEXT
        model_lower = model_name.lower()
        for family in _NATIVE_TOOL_FAMILIES:
            if family in model_lower:
                return ToolCallingTier.NATIVE
        log.info("openai_compat_text_tier", model=model_name,
                 reason="local server, unknown model family")
        return ToolCallingTier.TEXT

    if isinstance(backend, AugmentumEngineBackend):
        # The Augmentum Engine is llama-server's OpenAI-compat surface with
        # ``--jinja`` always on (see CLAUDE.md "--jinja is mandatory"), so it
        # renders each GGUF's native tool-call template. Assume NATIVE rather
        # than gating on a hardcoded family list: modern instruction-tuned
        # models are trained on the OpenAI tool-calling standard, and a static
        # allowlist goes stale the moment a new model ships — silently
        # demoting it to a worse tier for no reason. A model whose template
        # genuinely lacks tool support just emits no tool_calls (same outcome
        # as the old STRUCTURED path for the common case), minus the upkeep.
        # ``_NATIVE_TOOL_FAMILIES`` is retained only for the local
        # OpenAI-compat heuristic below, which has no runtime fallback yet.
        return ToolCallingTier.NATIVE

    if isinstance(backend, OllamaBackend):
        # Ollama implements the native tool-calling API for any model whose
        # template declares tool support; assume NATIVE for the same reason as
        # the engine above (don't gate on a stale hardcoded list).
        return ToolCallingTier.NATIVE

    if isinstance(backend, LlamaCppBackend):
        # LlamaCppBackend IS llama-server with ``--jinja`` always on — it
        # renders each GGUF's native tool-call template and emits/streams
        # native ``tool_calls`` deltas (llama_cpp.py: "llama-server emits
        # native tool_calls when --jinja is on"; chat_stream parses the
        # tool_call deltas). It is functionally identical to
        # AugmentumEngineBackend above, so it gets the same NATIVE tier.
        # Demoting it to TEXT forced the non-streaming peek-then-blob path
        # for every local-model chat turn that didn't fire a tool — the
        # blob the user saw. (2026-06-30: fixed the misclassification; the
        # streaming-with-tools path already existed, this backend just
        # wasn't routed to it.)
        return ToolCallingTier.NATIVE

    # Unknown backend — safest fallback (text-regex tool parsing).
    return ToolCallingTier.TEXT


def _is_cloud_api(base_url: str) -> bool:
    """Check if a base URL belongs to a known cloud API provider."""
    from urllib.parse import urlparse

    try:
        hostname = urlparse(base_url).hostname or ""
    except Exception:
        return False

    hostname = hostname.lower()
    return any(host in hostname for host in _NATIVE_CLOUD_HOSTS)


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def tools_to_native_format(tools: list[Tool]) -> list[dict]:
    """Convert Tool objects to OpenAI function calling format.

    Dedupes by function name: some enabled-tool sets (notably the blanket
    "all" selection) surface the same name twice, and providers such as
    DeepSeek / OpenAI HARD-400 on "Tool names must be unique." Keep the
    first occurrence of each name.
    """
    result = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {
                    "type": "object",
                    "properties": {},
                },
            },
        })
    return result


def build_structured_output_schema(tools: list[Tool]) -> dict:
    """Build a JSON Schema for Tier 2 structured output via Ollama grammar decoding.

    The model outputs either:
      {"action": "tool_call", "tool_name": "...", "tool_input": {...}}
    or:
      {"action": "text_response", "text": "..."}

    The enum constraints on ``action`` and ``tool_name`` prevent hallucinated
    tool names and guarantee parseable structure.
    """
    tool_names = [t.name for t in tools]
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tool_call", "text_response"],
            },
            "tool_name": {
                "type": "string",
                "enum": tool_names,
            },
            "tool_input": {
                "type": "object",
            },
            "text": {
                "type": "string",
            },
        },
        "required": ["action"],
    }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def _native_args(raw: object, tool_name: str) -> dict:
    """Normalise a native tool call's ``arguments`` field into a dict.

    ``arguments`` arrives as a dict (Ollama-style) or a JSON string
    (OpenAI-style). When the string is NOT valid JSON, this used to fall
    through to ``{}`` — silently discarding the value the model supplied and
    failing downstream as "requires: expression" while the number sat right
    there in the response. Small models emit exactly that: a bare
    ``"34.46 * 0.99"``, or a whole ``calculator("34.46 * 0.99")`` echo.

    Both salvageable, neither by guessing: a python-style echo is re-parsed
    by the real parser, and a bare scalar is handed to the coercion layer
    tagged with ``POSITIONAL_GUESS_KEY`` so it gets bound to the tool's
    actual required parameter rather than assumed to be ``query``.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}

    raw = raw.strip()
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed

    # The model echoed the whole call — let the python-style parser own it.
    if tool_name and raw.startswith(f"{tool_name}("):
        echoed = parse_python_style_tool_call(raw, known_tools={tool_name})
        if echoed:
            return echoed[1]

    # A bare positional value. Tag it rather than naming it ourselves.
    return {"query": _unquote(raw), POSITIONAL_GUESS_KEY: ["query"]}


def parse_native_tool_call(response: InternalChatResponse) -> tuple[str, dict] | None:
    """Extract tool name and arguments from a native tool_calls response.

    Handles both OpenAI-style (string arguments) and Ollama-style (dict arguments).
    Returns ``(tool_name, args_dict)`` or ``None`` if no tool call found.
    """
    if not response.message or not response.message.tool_calls:
        return None

    tc = response.message.tool_calls[0]  # execute first tool call only
    func = tc.get("function", tc)
    name = func.get("name", "")
    if not name:
        return None

    return (name, _native_args(func.get("arguments", {}), name))


def parse_native_tool_calls_all(response: InternalChatResponse) -> list[tuple[str, dict]]:
    """Extract ALL tool calls from a native tool_calls response.

    Returns a list of ``(tool_name, args_dict)`` tuples. Returns empty list
    if no tool calls found.
    """
    if not response.message or not response.message.tool_calls:
        return []

    results = []
    for tc in response.message.tool_calls:
        func = tc.get("function", tc)
        name = func.get("name", "")
        if not name:
            continue

        results.append((name, _native_args(func.get("arguments", {}), name)))

    return results


def parse_structured_output(text: str) -> tuple[str, dict] | None:
    """Parse Tier 2 structured JSON output.

    Returns ``(tool_name, tool_input)`` if the model chose ``tool_call``,
    or ``None`` if the model chose ``text_response`` or JSON is malformed.
    """
    with contextlib.suppress(json.JSONDecodeError, ValueError, TypeError):
        data = json.loads(text)
        if isinstance(data, dict) and data.get("action") == "tool_call":
            tool_name = data.get("tool_name", "")
            tool_input = data.get("tool_input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}
            if tool_name:
                return (tool_name, tool_input)
    return None


def extract_structured_text(text: str) -> str:
    """Extract the text content from a Tier 2 ``text_response`` output.

    Returns the ``text`` field value, or the raw text if parsing fails.
    """
    with contextlib.suppress(json.JSONDecodeError, ValueError, TypeError):
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("text", text)
    return text


# ---------------------------------------------------------------------------
# Python-style function call parsing (fallback for models that emit code)
# ---------------------------------------------------------------------------

# Matches the start of a Python-style function call: tool_name(
_PYTHON_CALL_START_RE = re.compile(
    r"([a-z_][a-z0-9_]*)\s*\(",
    re.IGNORECASE,
)


def parse_json_tool_calls(
    text: str,
    known_tools: set[str] | None = None,
) -> list[tuple[str, dict]] | None:
    """Parse JSON-format tool calls that models sometimes embed in text content.

    Catches patterns like:
    - ``[{"name": "web", "arguments": {"query": "..."}}]``  (array)
    - ``{"name": "web", "arguments": {"query": "..."}}``    (single object)

    Returns list of ``(tool_name, args_dict)`` or ``None``.
    """
    text = text.strip()
    # Strip code fences
    if "```" in text:
        text = re.sub(r"```[\w]*\n?", "", text).replace("```", "").strip()

    # Try to find JSON in the text
    for start_ch, end_ch in [("[", "]"), ("{", "}")]:
        start = text.find(start_ch)
        if start == -1:
            continue
        end = text.rfind(end_ch)
        if end <= start:
            continue
        candidate = text[start : end + 1]
        with contextlib.suppress(json.JSONDecodeError, ValueError, TypeError):
            data = json.loads(candidate)
            # Normalise to list
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                continue
            results = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                if not name:
                    continue
                if known_tools and name not in known_tools:
                    continue
                args = item.get("arguments", item.get("parameters", item.get("input", {})))
                if isinstance(args, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        args = json.loads(args)
                if not isinstance(args, dict):
                    args = {}
                results.append((name, args))
            if results:
                return results
    return None


def parse_python_style_tool_call(
    text: str,
    known_tools: set[str] | None = None,
) -> tuple[str, dict] | None:
    """Parse a Python-style function call from LLM text output.

    Catches patterns like:
    - ``web_search("current weather")``
    - ``web_search(query="current weather")``
    - ``calculator(expression="2 + 2")``
    - Code-fenced versions of the above

    Only matches if the function name is in ``known_tools`` (when provided)
    to avoid false positives on arbitrary code.

    Returns ``(tool_name, args_dict)`` or ``None``.
    """
    # Strip code fences for easier matching
    stripped = text
    if "```" in stripped:
        stripped = re.sub(r"```[\w]*\n?", "", stripped).replace("```", "")

    for m in _PYTHON_CALL_START_RE.finditer(stripped):
        name = m.group(1)

        # Only match known tool names to avoid false positives
        if known_tools and name not in known_tools:
            continue

        # Extract balanced parenthesized arguments
        args_str = _extract_balanced_parens(stripped, m.end() - 1)
        if args_str is None:
            continue

        if not args_str:
            return (name, {})

        args = _parse_python_args(args_str, name)
        if args is not None:
            return (name, args)

    return None


def _extract_balanced_parens(text: str, open_pos: int) -> str | None:
    """Extract content between balanced parentheses starting at open_pos.

    Returns the content between ( and ) or None if unbalanced.
    """
    if open_pos >= len(text) or text[open_pos] != "(":
        return None

    depth = 0
    in_quote = ""

    for i in range(open_pos, len(text)):
        ch = text[i]
        if ch in ('"', "'") and not in_quote:
            in_quote = ch
        elif ch == in_quote:
            in_quote = ""
        elif not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[open_pos + 1 : i].strip()

    return None


def _parse_python_args(args_str: str, tool_name: str) -> dict | None:
    """Parse Python-style function arguments into a dict.

    Handles:
    - Keyword args: ``key="value", key2="value2"``
    - Positional string: ``"some query"`` → maps to first required param
    - Mixed: ``"query", max_results=5``
    """
    # Try as JSON object first (some models wrap in braces)
    if args_str.startswith("{"):
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            return json.loads(args_str)

    result: dict = {}
    positional: list[str] = []

    # Split on commas that aren't inside quotes
    parts = _split_args(args_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for keyword argument: key="value" or key=value
        kw_match = re.match(r'([a-z_]\w*)\s*=\s*(.*)', part, re.IGNORECASE)
        if kw_match:
            key = kw_match.group(1)
            val = _unquote(kw_match.group(2).strip())
            # Try to parse as JSON for numbers/bools
            result[key] = _coerce_value(val)
        else:
            # Positional argument
            positional.append(_unquote(part))

    # Map positional args to "query" (most common single-param tool pattern).
    #
    # This is a GUESS: we have the tool name but no schema here (the caller
    # only knows tool NAMES), so "query" is a statistical bet that is wrong
    # for calculator(expression), read_file(path), and every other tool whose
    # first param isn't a search string. Record the guess under
    # ``POSITIONAL_GUESS_KEY`` so the coercion layer — which DOES have the
    # schema — can rebind it to the real parameter. Without the marker the
    # rebinder cannot tell this synthesized "query" apart from a genuine
    # model typo like ``catgory=``, and rebinding a typo'd *optional* param
    # into the missing *required* one invents data the model never supplied.
    if positional and not result:
        # Single positional → use "query" as the key (most tools use this)
        if len(positional) == 1:
            result["query"] = positional[0]
        else:
            result["query"] = ", ".join(positional)
        result[POSITIONAL_GUESS_KEY] = ["query"]
    elif positional:
        # Mix of positional and keyword — add positionals as "query"
        if "query" not in result:
            result["query"] = positional[0]
            result[POSITIONAL_GUESS_KEY] = ["query"]

    return result if result else None


def _split_args(s: str) -> list[str]:
    """Split argument string by commas, respecting quotes."""
    parts: list[str] = []
    current: list[str] = []
    in_quote = ""
    depth = 0

    for ch in s:
        if ch in ('"', "'") and not in_quote:
            in_quote = ch
            current.append(ch)
        elif ch == in_quote:
            in_quote = ""
            current.append(ch)
        elif ch in ("(", "[", "{"):
            depth += 1
            current.append(ch)
        elif ch in (")", "]", "}"):
            depth -= 1
            current.append(ch)
        elif ch == "," and not in_quote and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current))

    return parts


def _unquote(s: str) -> str:
    """Remove surrounding quotes from a string value."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _coerce_value(s: str) -> str | int | float | bool:
    """Coerce a string value to its Python type if possible."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    with contextlib.suppress(ValueError):
        return int(s)
    with contextlib.suppress(ValueError):
        return float(s)
    return s


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def coerce_tool_params(tool: Tool, params: dict) -> dict:
    """Coerce parameter types based on the tool's JSON Schema.

    Thin delegate to :func:`augmentum.tools.params.coerce_params`, which
    is the single implementation shared by every dispatch path (chat,
    coder, MCP, ATP, subagents) via ``Tool.invoke``. Kept as a name so
    existing call sites and tests keep working.

    NOTE: this does NOT fan out a list supplied for a string param —
    that changes the number of calls, so it lives in ``Tool.invoke``.
    Prefer ``await tool.invoke(params)`` over calling this directly.
    """
    from augmentum.tools.params import coerce_params

    return coerce_params(tool, params)


# ---------------------------------------------------------------------------
# Additional format parsers (ReAct, XML, fuzzy)
# ---------------------------------------------------------------------------

# ReAct-style: Action: tool_name / Action Input: {...}
_ACTION_NAME_RE = re.compile(
    r"(?:^|\n)\s*\*{0,2}(?:Action|Tool|Function)\*{0,2}\s*[:=]\s*\*{0,2}\s*(\S+?)\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ACTION_INPUT_RE = re.compile(
    r"(?:^|\n)\s*\*{0,2}(?:Action\s*Input|Tool\s*Input|Input|Parameters|Arguments|Args)\*{0,2}\s*[:=]\s*",
    re.IGNORECASE | re.MULTILINE,
)


def parse_action_input_tool_call(
    text: str,
    known_tools: set[str] | None = None,
) -> tuple[str, dict] | None:
    """Parse ReAct-style Action/Action Input tool calls.

    Catches patterns like::

        Action: web_search
        Action Input: {"query": "weather today"}

        Thought: I need to search...
        Action: web_search
        Action Input: {"query": "..."}

        **Tool**: calculator
        **Input**: {"expression": "2+2"}

    Returns ``(tool_name, args_dict)`` or ``None``.
    """
    name_match = _ACTION_NAME_RE.search(text)
    if not name_match:
        return None

    name = name_match.group(1).strip().strip("`*\"'")
    if not name or name.lower() in ("tool_name", "toolname", "name", "none"):
        return None
    if known_tools and name not in known_tools:
        return None

    # Find the input/arguments
    input_match = _ACTION_INPUT_RE.search(text, name_match.end())
    if input_match:
        after = text[input_match.end():]
        # Try JSON extraction
        from augmentum.modes.analytical.engine import _balanced_json_extract, _try_parse_json
        json_str = _balanced_json_extract(after)
        if json_str:
            result = _try_parse_json(json_str)
            if result:
                return (name, result)
        # Try plain string value (e.g. Action Input: weather today)
        first_line = after.strip().split("\n")[0].strip().strip('"\'`')
        if first_line and not first_line.startswith("{"):
            return (name, {"query": first_line})

    return (name, {})


# XML-style: <tool_call>, <tool_use>, <function_call> blocks
_XML_TOOL_BLOCK_RE = re.compile(
    r"<(tool_call|tool_use|function_call)[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_XML_NAME_RE = re.compile(
    r"<(?:name|function|tool)>(.*?)</(?:name|function|tool)>",
    re.IGNORECASE | re.DOTALL,
)
_XML_INPUT_RE = re.compile(
    r"<(?:input|arguments|parameters|args)>(.*?)</(?:input|arguments|parameters|args)>",
    re.IGNORECASE | re.DOTALL,
)
# Also catch <tool_call>{"name": "...", ...}</tool_call> (Hermes/Qwen style)
_XML_INLINE_JSON_RE = re.compile(
    r"<(?:tool_call|function_call)>\s*(\{.*?\})\s*</(?:tool_call|function_call)>",
    re.IGNORECASE | re.DOTALL,
)


def parse_xml_tool_calls(
    text: str,
    known_tools: set[str] | None = None,
) -> list[tuple[str, dict]] | None:
    """Parse XML-style tool call blocks.

    Catches patterns like::

        <tool_use>
          <name>web_search</name>
          <input>{"query": "weather"}</input>
        </tool_use>

        <tool_call>
        {"name": "web_search", "arguments": {"query": "..."}}
        </tool_call>

        <function_call name="web_search">{"query": "..."}</function_call>

    Returns list of ``(tool_name, args_dict)`` or ``None``.
    """
    from augmentum.modes.analytical.engine import _try_parse_json

    results: list[tuple[str, dict]] = []

    # Pattern 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    for m in _XML_INLINE_JSON_RE.finditer(text):
        with contextlib.suppress(json.JSONDecodeError, ValueError, TypeError):
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                name = data.get("name", "")
                if name and (not known_tools or name in known_tools):
                    args = data.get("arguments", data.get("parameters", data.get("input", {})))
                    if isinstance(args, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            args = json.loads(args)
                    if not isinstance(args, dict):
                        args = {}
                    results.append((name, args))

    if results:
        return results

    # Pattern 2: <tool_use><name>...</name><input>...</input></tool_use>
    for m in _XML_TOOL_BLOCK_RE.finditer(text):
        block = m.group(2)
        name_m = _XML_NAME_RE.search(block)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        if not name or (known_tools and name not in known_tools):
            continue

        args: dict = {}
        input_m = _XML_INPUT_RE.search(block)
        if input_m:
            raw_input = input_m.group(1).strip()
            if raw_input.startswith("{"):
                args = _try_parse_json(raw_input)
            elif raw_input:
                args = {"query": raw_input}
        results.append((name, args))

    return results if results else None


def parse_fuzzy_tool_call(
    text: str,
    known_tools: set[str],
) -> tuple[str, dict] | None:
    """Last-resort fuzzy parser: find a known tool name near a JSON object.

    Catches cases where the model mentions the tool name in conversational
    text alongside a JSON argument block, e.g.::

        I'll use web_search with {"query": "weather today"}
        Let me call the calculator tool: {"expression": "2+2"}

    Safety constraints to avoid false positives:
    - Requires ``known_tools`` to be non-empty
    - Tool name must appear within 200 chars before a valid JSON object
    - JSON keys must overlap with the tool's expected parameters (if known)
    - Rejects text with negation signals ("don't", "instead of", "no need")
      near the tool name to avoid intercepting refusals

    Returns ``(tool_name, args_dict)`` or ``None``.
    """
    if not known_tools:
        return None

    from augmentum.modes.analytical.engine import _balanced_json_extract, _try_parse_json

    text_lower = text.lower()

    # Reject if the text contains negation near a tool name — the model
    # is likely explaining why it WON'T use the tool.
    _NEGATION_SIGNALS = ("don't", "dont", "do not", "won't", "will not",
                         "instead of", "no need", "not necessary",
                         "rather than", "without using", "skip")

    for tool_name in known_tools:
        idx = text_lower.find(tool_name.lower())
        if idx == -1:
            continue

        # Check for negation within 60 chars before the tool name
        before = text_lower[max(0, idx - 60) : idx]
        if any(neg in before for neg in _NEGATION_SIGNALS):
            continue

        # Look for a JSON object within 200 chars after the tool name
        search_start = idx + len(tool_name)
        search_window = text[search_start : search_start + 200]
        json_str = _balanced_json_extract(search_window)
        if json_str:
            args = _try_parse_json(json_str)
            if args:
                return (tool_name, args)

    return None
