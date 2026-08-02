"""Analytical engine — core orchestrator for the UARF analytical pipeline.

Processing pipeline per query:
1. ASSESS  — Determine query complexity (simple/moderate/complex)
2. IDENTIFY — Identify key concepts, unknowns, constraints
3. RELEVANT — Gather relevant information and frameworks
4. APPLY   — Apply reasoning to solve the problem
5. VERIFY  — Check work for errors and contradictions
6. CONCLUDE — Synthesize final polished answer

Routing based on complexity:
- Simple:   ASSESS -> APPLY -> CONCLUDE (3 calls)
- Moderate: ASSESS -> IDENTIFY -> RELEVANT -> APPLY -> VERIFY -> CONCLUDE (6 calls)
- Complex:  Full pipeline + decomposition support (6+ calls)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    Message,
    ModelBackend,
)
from augmentum.modes.analytical.auto_verify import run_auto_verification
from augmentum.modes.analytical.prompts import (
    SEARCH_CONTEXT_SECTION,
    SEARCH_QUERY_PROMPT,
    get_native_tool_prompt_section,
    get_phase_prompt,
    get_structured_tool_prompt_section,
    get_tool_prompt_section,
)
from augmentum.modes.analytical.state import (
    AnalyticalPhase,
    AnalyticalResult,
    AnalyticalState,
    PhaseResult,
    ToolCallRecord,
)
from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    build_structured_output_schema,
    coerce_tool_params,
    extract_structured_text,
    select_tier,
    tools_to_native_format,
)
from augmentum.tools.base import ToolResult, invoke_tool
from augmentum.utils.datetime_context import get_datetime_context
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.cache.prompt_cache import PromptCache
    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.tools.circuit_breaker import ToolCircuitBreaker
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

# Defaults — actual values read from settings at use-time so the API can update them.
# Exported for handler.py imports (backward compat).
_MAX_PHASE_RETRIES = 2
_CONFIDENCE_THRESHOLD = 0.5


def _get_max_phase_retries() -> int:
    return getattr(settings, "analytical_max_phase_retries", _MAX_PHASE_RETRIES)


def _get_confidence_threshold() -> float:
    return getattr(settings, "analytical_confidence_threshold", _CONFIDENCE_THRESHOLD)


def _get_max_backtracks() -> int:
    return getattr(settings, "analytical_max_backtracks", 3)

# Max tool calls allowed per phase execution
_MAX_TOOL_CALLS_PER_PHASE = 3

# Patterns for proactive tool suggestions
_URL_PATTERN = re.compile(r"https?://\S+")
_MATH_PATTERN = re.compile(
    r"(?:calculate|compute|solve|evaluate|what is \d|\d\s*[\+\-\*/\^])", re.IGNORECASE
)
_SEARCH_PATTERN = re.compile(
    r"(?:search|find|look up|what is|who is|current|latest|recent|when did|where is)",
    re.IGNORECASE,
)
_MEMORY_PATTERN = re.compile(
    r"(?:my |i (?:told|said|mentioned)|remember|previously|you know|"
    r"we (?:discussed|talked)|last time|what(?:'s| is) my|"
    r"do you (?:know|remember)|i (?:previously|already)|"
    r"as i (?:mentioned|said)|from (?:before|earlier|last))",
    re.IGNORECASE,
)
_YOUTUBE_PATTERN = re.compile(
    r"(?:youtube\.com/(?:watch|embed|shorts|live)|youtu\.be/|"
    r"youtube\s+video|video\s+transcript|summarize\s+(?:this\s+)?video)",
    re.IGNORECASE,
)
_WIKIPEDIA_PATTERN = re.compile(
    r"(?:wikipedia|wiki\b|encyclopedia|"
    r"(?:tell|teach)\s+me\s+about|what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:history|origin|definition)\s+of|"
    r"who\s+(?:is|was|were)\s+\w+\s+\w+|"
    r"define\s+\w+|biography\s+of)",
    re.IGNORECASE,
)
_DOCUMENT_PATTERN = re.compile(
    r"(?:\.(?:pdf|docx|pptx|xlsx|csv)\b|"
    r"(?:parse|read|extract|open|analyze)\s+(?:the\s+)?(?:document|file|pdf|spreadsheet|presentation))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Auto-search heuristics
# ---------------------------------------------------------------------------

# Temporal keywords that signal the query needs fresh data.
_TEMPORAL_PATTERN = re.compile(
    r"(?:today|tonight|right now|this (?:week|month|year)|yesterday|"
    r"(?:20\d{2})|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})",
    re.IGNORECASE,
)

# ASSESS TYPE values that indicate the query needs external facts.
_FACTUAL_TYPE_PATTERN = re.compile(
    r"(?:factual|current[_ ]events?|real[- ]time|news|lookup|data[_ ]retrieval)",
    re.IGNORECASE,
)

# Regex to strip leading numbering from LLM-generated query lines.
_QUERY_LINE_PREFIX = re.compile(r"^\s*(?:\d+[\.\)\-:]\s*|[-*]\s*|[\"'])")

# Preamble patterns that LLMs prepend before actual queries.
_PREAMBLE_PATTERN = re.compile(
    r"^(?:here\s+are|sure|okay|certainly|of\s+course|the\s+following|below\s+are|i'?d\s+suggest)",
    re.IGNORECASE,
)

# Auto-search tunables are read from ``settings`` at call-time so that
# env-var overrides (AUGMENTUM_UARF_AUTO_SEARCH_*) take effect without
# restarting the process.  The module-level constants remain as fallbacks
# for unit tests that don't spin up the full config.
_AUTO_SEARCH_QUERIES = 3
_AUTO_SEARCH_RESULTS_PER_QUERY = 4
_AUTO_SEARCH_MAX_CONTEXT_CHARS = 4000

# ---------------------------------------------------------------------------
# Refusal detection — catches safety-trained models that refuse instead of
# producing useful search queries or using provided search context.
# ---------------------------------------------------------------------------
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i can't provide",
    "i cannot provide",
    "i'm unable to",
    "i am unable to",
    "i don't have access",
    "i do not have access",
    "i can't access",
    "i cannot access",
    "i'm not able to",
    "i am not able to",
    "i can't browse",
    "i cannot browse",
    "i can't search",
    "i cannot search",
    "i don't have the ability",
    "i do not have the ability",
    "as an ai",
    "as a language model",
    "as an llm",
    "i'm sorry, but i",
    "i apologize, but i",
    "real-time information",
    "real-time data",
    "i can't give you",
    "i cannot give you",
    "i'm afraid i",
    "i am afraid i",
)

# ---------------------------------------------------------------------------
# Robust tool-call parsing helpers (tolerant of small-model formatting)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Patterns that tolerate inline markdown (bold ``**`` / backtick `` ` ``)
# so we NEVER strip ``*`` from the full output.  Stripping ``*`` globally
# destroys content like ``"25 * 1.08"`` inside JSON values.
# ---------------------------------------------------------------------------

# TOOL_CALL patterns — handle **TOOL_CALL:**, `TOOL_CALL`:, TOOL CALL:, etc.
# [*`]* eats optional leading/trailing markdown; [:=] is the separator.
_TOOL_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"[*`]*TOOL[_ \-]?CALL[*`]*\s*[:=]\s*[*`]*\s*(\S+?)[*`]*\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"(?:using\s+)?tool[*`]*\s*[:=]\s*[*`]*\s*(\S+?)[*`]*\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# TOOL_INPUT patterns — same markdown tolerance
_TOOL_INPUT_PATTERN = re.compile(
    r"[*`]*TOOL[_ \-]?INPUT[*`]*\s*[:=]\s*[*`]*",
    re.IGNORECASE,
)

# Strip residual punctuation from captured tool names
_NAME_CLEAN_RE = re.compile(r"[`*\"\'\[\](){}<>]")

# Placeholder tool names that small models copy from the prompt template.
_PLACEHOLDER_NAMES = frozenset({
    "tool_name", "toolname", "name", "tool",
})


def _extract_tool_name(output: str) -> str:
    """Extract the tool name from the LLM output.

    Patterns handle inline markdown (bold / backtick) directly so we
    never strip ``*`` from the full output (which would destroy content
    like ``"25 * 1.08"`` in JSON values).  Rejects placeholder names
    like ``tool_name`` that small models copy from prompt templates.
    """
    for pattern in _TOOL_NAME_PATTERNS:
        m = pattern.search(output)
        if m:
            raw = m.group(1)
            raw = _NAME_CLEAN_RE.sub("", raw).strip().rstrip(".,;:")
            if raw and raw.lower() not in _PLACEHOLDER_NAMES:
                return raw
    return ""


def _balanced_json_extract(text: str) -> str:
    """Extract the first balanced ``{...}`` block from *text*."""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _try_parse_json(raw: str) -> dict:
    """Try to parse JSON, with fallbacks for common LLM formatting issues."""
    # 1. Direct parse
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        return json.loads(raw)

    # 2. Escape literal newlines/tabs inside JSON strings.
    #    Models often put multiline code with real line breaks inside JSON
    #    values (e.g. {"code": "import math\nprint(1)"}).  This is invalid
    #    JSON — newlines in strings must be escaped as \\n.
    fixed = _escape_newlines_in_json_strings(raw)
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        return json.loads(fixed)

    # 3. Single quotes → double quotes (common small-model mistake)
    fixed = fixed.replace("'", '"')
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        return json.loads(fixed)

    # 4. Strip trailing commas before closing braces
    fixed2 = re.sub(r",\s*}", "}", fixed)
    fixed2 = re.sub(r",\s*]", "]", fixed2)
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        return json.loads(fixed2)

    return {}


def _escape_newlines_in_json_strings(raw: str) -> str:
    """Escape literal newlines and tabs inside JSON string values.

    Walks the string tracking whether we're inside a JSON string
    (between unescaped double quotes) and replaces raw control
    characters with their escape sequences.
    """
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and in_string and i + 1 < len(raw):
            # Already escaped — pass through both chars
            result.append(ch)
            result.append(raw[i + 1])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _extract_tool_input(output: str) -> dict:
    """Extract tool input JSON from the LLM output.

    IMPORTANT: JSON is extracted from the ORIGINAL output (never
    markdown-stripped) so that characters like ``*`` in values
    such as ``"25 * 1.08"`` are preserved.

    Resolution order:
    1. Explicit TOOL_INPUT line with JSON object
    2. Any JSON object appearing after the TOOL_CALL line
    3. Key=value pairs after TOOL_INPUT
    4. Empty dict (caller will get a missing-required-fields error)
    """
    # --- 1. Explicit TOOL_INPUT: {...} (search original text) ---
    m = _TOOL_INPUT_PATTERN.search(output)
    if m:
        after = output[m.end():]
        json_str = _balanced_json_extract(after)
        if json_str:
            result = _try_parse_json(json_str)
            if result:
                return result

        # Try key=value parsing from the text after TOOL_INPUT
        kv = _parse_key_value_lines(after)
        if kv:
            return kv

        # --- 1b. Bare string after TOOL_INPUT (e.g. TOOL_INPUT: "15*23+7") ---
        bare = after.split("\n")[0].strip()
        if bare:
            # Unquote
            if len(bare) >= 2 and bare[0] == bare[-1] and bare[0] in ('"', "'"):
                bare = bare[1:-1]
            if bare:
                return {"query": bare}

    # --- 2. Any JSON object after the TOOL_CALL line ---
    for pattern in _TOOL_NAME_PATTERNS:
        name_m = pattern.search(output)
        if name_m:
            after_name = output[name_m.end():]
            json_str = _balanced_json_extract(after_name)
            if json_str:
                result = _try_parse_json(json_str)
                if result:
                    return result
            break

    return {}


def _parse_key_value_lines(text: str) -> dict:
    """Parse ``key = value`` or ``key: value`` lines into a dict.

    Only considers the first few lines after TOOL_INPUT to avoid
    grabbing unrelated content.
    """
    lines = text.strip().splitlines()[:8]
    result: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kv = re.match(
            r'["`\']?(\w+)["`\']?\s*[:=]\s*["`\']?(.*?)["`\']?\s*$',
            line,
        )
        if kv:
            key, val = kv.group(1), kv.group(2).strip().strip("\"'`")
            if key and val:
                result[key] = val
    return result


class AnalyticalEngine:
    """Core UARF analytical engine — orchestrates multi-phase reasoning."""

    def __init__(
        self,
        backend: ModelBackend,
        tool_registry: ToolRegistry | None = None,
        prompt_cache: PromptCache | None = None,
        provider_registry: ProviderRegistry | None = None,
        circuit_breaker: ToolCircuitBreaker | None = None,
        enabled_tools: frozenset[str] | None = None,
        user_id: str = "",
    ) -> None:
        self._backend = backend
        self._tool_registry = tool_registry
        self._prompt_cache = prompt_cache
        self._provider_registry = provider_registry
        self._circuit_breaker = circuit_breaker
        # Owner of this run. Threaded into every tool call as ``_user_id``
        # (see ``_execute_tool``) — without it every user-scoped tool
        # (memory_recall, all the artifact creators, build_application)
        # fails with "requires a user_id".
        self._user_id = user_id
        # User's textbox tool selection — passed through to ``get_for_phase``
        # so the selector is the source of truth for what the model sees.
        self._enabled_tools = enabled_tools
        self._state = AnalyticalState()
        self._state.max_backtracks = _get_max_backtracks()

        # Per-engine tool result cache (lives for one UARF pipeline run)
        from augmentum.tools.cache import ToolResultCache
        self._tool_cache = ToolResultCache() if settings.tool_cache_enabled else None

        # Datetime context pinned for this entire UARF run — every phase
        # gets the same string so the shared prefix (see prompts.
        # build_shared_prefix) is byte-identical and llama-server's
        # cache_prompt can reuse the prefix KV across phases.
        self._run_datetime_ctx = get_datetime_context()

    @property
    def state(self) -> AnalyticalState:
        """Expose internal state for testing and inspection."""
        return self._state

    # ------------------------------------------------------------------
    # Refusal detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Return True if *text* looks like a model refusal rather than content.

        Used to detect when a small safety-trained model outputs refusal
        boilerplate instead of search queries or analytical content.
        """
        lower = text.lower().strip()
        return any(phrase in lower for phrase in _REFUSAL_PHRASES)

    # ------------------------------------------------------------------
    # Auto-search: system-driven web search (bypasses LLM tool-calling)
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_search(query: str, assess_output: str = "") -> bool:
        """Determine if the query needs automatic web search.

        Uses four heuristic signals (any triggers auto-search):
        1. ``_URL_PATTERN`` — query contains a URL (direct fetch).
        2. ``_SEARCH_PATTERN`` — keywords like "current", "who is", etc.
        3. ``_TEMPORAL_PATTERN`` — time-sensitive words: "today", "2026", etc.
        4. ``_FACTUAL_TYPE_PATTERN`` — ASSESS TYPE values: "factual", "news", etc.
        """
        if _URL_PATTERN.search(query):
            return True
        if _SEARCH_PATTERN.search(query):
            return True
        if _TEMPORAL_PATTERN.search(query):
            return True
        return bool(assess_output and _FACTUAL_TYPE_PATTERN.search(assess_output))

    async def _generate_search_queries(
        self,
        model: str,
        query: str,
        num_queries: int = _AUTO_SEARCH_QUERIES,
        conversation_context: str = "",
    ) -> list[str]:
        """Generate search queries from the user's request via a small LLM call.

        ``conversation_context`` (recent turns) lets the model resolve a
        follow-up — "compare both", "tell me more about it" — into queries
        that name the actual topics, instead of literally searching the
        pronoun. Without it, "do a full comparison on both" searched for the
        phrase "full comparison" and returned unrelated junk. Falls back to
        the original query if the model returns garbage or refusal text.
        """
        conv_block = ""
        if conversation_context and conversation_context.strip():
            conv_block = (
                "\nRecent conversation (resolve what the latest request "
                "refers to — pronouns like 'both', 'it', 'that one'):\n"
                f"{conversation_context.strip()}\n"
            )
        prompt = SEARCH_QUERY_PROMPT.format(
            query=query,
            datetime_context=get_datetime_context(),
            conversation=conv_block,
        )
        request = InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content=query),
            ],
            stream=False,
        )
        try:
            response = await self._backend.chat(request)
            raw = response.message.content if response.message else ""
        except Exception as exc:
            log.warning("auto_search_query_gen_failed", error=str(exc))
            raw = ""

        # Parse: one query per line, strip numbering and quotes.
        # Filter out lines that are refusal boilerplate (safety-trained models
        # sometimes mix refusal text with actual queries).
        queries: list[str] = []
        has_refusal = False
        for line in raw.splitlines():
            cleaned = _QUERY_LINE_PREFIX.sub("", line).strip().rstrip("\"'")
            if not cleaned or len(cleaned) <= 3:  # noqa: PLR2004
                continue
            # Skip individual lines that are refusals
            if self._is_refusal(cleaned):
                has_refusal = True
                log.debug("auto_search_query_line_refusal", line=cleaned[:100])
                continue
            # Skip preamble lines ("Here are three search queries:", etc.)
            if _PREAMBLE_PATTERN.search(cleaned):
                log.debug("auto_search_query_line_preamble", line=cleaned[:100])
                continue
            queries.append(cleaned)
            if len(queries) >= num_queries:
                break

        if has_refusal:
            log.warning(
                "auto_search_query_gen_refusal_detected",
                valid_queries=len(queries),
                raw=raw[:200],
            )

        if not queries:
            log.warning("auto_search_query_gen_empty_fallback", raw=raw[:200])
            queries = [query]

        self._state.search_queries = queries
        log.info("auto_search_queries_generated", queries=queries)
        return queries

    async def _execute_auto_search(
        self,
        queries: list[str],
        results_per_query: int = _AUTO_SEARCH_RESULTS_PER_QUERY,
        max_context_chars: int = _AUTO_SEARCH_MAX_CONTEXT_CHARS,
    ) -> str:
        """Execute web searches in parallel and build a formatted context block.

        Deduplicates results by URL, caps total context at
        ``max_context_chars``, and stores the context in
        ``self._state.search_context``.

        When ``search_expansion_enabled`` is set, queries are expanded using
        zero-cost heuristics (synonym substitution, type-specific
        reformulation, domain-scoped site: prefixes) before execution.
        This typically increases query coverage by 2-3x at no LLM cost.
        """
        if not self._tool_registry:
            return ""
        search_tool = self._tool_registry.get("web_search")
        if not search_tool:
            log.warning("auto_search_no_web_search_tool")
            return ""

        # Enhance queries with topic-aware site: hints from the
        # preferred sources registry (zero-cost, no LLM).
        if settings.search_expansion_enabled:
            from augmentum.tools.web import _build_search_query

            enhanced: list[str] = []
            seen_norm: set[str] = set()
            for q in queries:
                norm = q.strip().lower()
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                enhanced.append(_build_search_query(q))
            if len(enhanced) != len(queries):
                log.info(
                    "search_expansion",
                    original=len(queries),
                    enhanced=len(enhanced),
                )
            queries = enhanced

        # Execute all queries in parallel
        async def _run_one(q: str) -> tuple[str, ToolResult]:
            try:
                result = await invoke_tool(search_tool, {
                    "query": q,
                    "num_results": results_per_query,
                    "_user_id": self._user_id,
                })
            except Exception as exc:
                log.warning("auto_search_exec_failed", query=q, error=str(exc))
                result = ToolResult(success=False, error=str(exc))
            # Record in state
            self._state.tool_calls.append(ToolCallRecord(
                phase="search",
                tool_name="web_search",
                input_data={"query": q, "num_results": results_per_query},
                output=(result.output or result.error or "")[:500],
                success=result.success,
            ))
            return q, result

        task_results = await asyncio.gather(
            *[_run_one(q) for q in queries],
            return_exceptions=True,
        )

        # Build formatted context, deduplicating by URL
        seen_urls: set[str] = set()
        context_parts: list[str] = []
        result_num = 0

        for item in task_results:
            if isinstance(item, Exception):
                log.warning("auto_search_gather_exception", error=str(item))
                continue
            q, result = item
            if not result.success or not result.output:
                continue
            context_parts.append(f"Search: \"{q}\"")
            # Parse individual results from the formatted output
            for block in result.output.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                # Extract URL for dedup
                url = ""
                for line in block.splitlines():
                    if line.strip().startswith("URL:"):
                        url = line.strip()[4:].strip()
                        break
                if url and url in seen_urls:
                    continue
                # Count only real result blocks (those with a URL). Url-less
                # blocks are the untrusted-wrapper markers and the search tool's
                # own "(N already shown)" dedup note — counting them inflated
                # search_result_count and could suppress a legitimate retry
                # (the cross-round dedup made the note frequent).
                if url:
                    seen_urls.add(url)
                    result_num += 1
                # Annotate with source quality from preferred sources registry
                if settings.search_credibility_enabled and url:
                    from augmentum.tools.preferred_sources import describe_source

                    desc = describe_source(url)
                    if desc:
                        block += f"\n    {desc}"
                context_parts.append(block)

        if not context_parts:
            log.warning("auto_search_no_results")
            self._state.search_context = ""
            self._state.search_result_count = 0
            return ""

        search_results_text = "\n\n".join(context_parts)

        # Truncate if needed
        if len(search_results_text) > max_context_chars:
            search_results_text = search_results_text[:max_context_chars] + "\n[... truncated]"

        # Escape braces in untrusted search content to prevent .format() injection
        safe_search_text = search_results_text.replace("{", "{{").replace("}", "}}")

        formatted = SEARCH_CONTEXT_SECTION.format(
            search_results=safe_search_text,
        )
        self._state.search_context = formatted
        self._state.search_result_count = result_num
        log.info(
            "auto_search_complete",
            num_queries=len(queries),
            num_results=result_num,
            context_len=len(formatted),
        )
        return formatted

    # ------------------------------------------------------------------
    # Search retry helpers
    # ------------------------------------------------------------------

    # Regex used by _broaden_queries to strip temporal qualifiers.
    _TEMPORAL_STRIP = re.compile(
        r"\b(?:today|tonight|yesterday|right now|this (?:week|month|year)|"
        r"(?:20\d{2})|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})\b",
        re.IGNORECASE,
    )
    _STOP_WORDS: frozenset[str] = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "what", "who", "where",
        "when", "how", "which", "that", "this", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "do", "does", "did", "can", "could",
        "will", "would", "should", "be", "been", "being", "have", "has", "had",
    })

    @staticmethod
    def _broaden_queries(
        original_queries: list[str],
        original_query: str,
    ) -> list[str]:
        """Generate broadened search queries for system-level retry.

        Deterministic — no LLM call — reliable for small models.
        Strategies: strip temporal words, extract key nouns, use raw query.
        Returns up to 3 new queries that differ from the originals.
        """
        seen = {q.lower().strip() for q in original_queries}
        broadened: list[str] = []

        # 1. Strip temporal qualifiers from each original query
        for q in original_queries:
            stripped = AnalyticalEngine._TEMPORAL_STRIP.sub("", q).strip()
            stripped = re.sub(r"\s{2,}", " ", stripped).strip()
            if stripped and stripped.lower() not in seen and len(stripped) > 3:  # noqa: PLR2004
                broadened.append(stripped)
                seen.add(stripped.lower())

        # 2. Extract key terms from the original user query
        words = original_query.split()
        key_words = [
            w for w in words
            if w.lower().strip("?.,!") not in AnalyticalEngine._STOP_WORDS
        ]
        if key_words:
            key_query = " ".join(key_words[:5])
            if key_query.lower() not in seen and len(key_query) > 3:  # noqa: PLR2004
                broadened.append(key_query)
                seen.add(key_query.lower())

        # 3. Use the raw original query as fallback
        raw = original_query.strip()
        if raw.lower() not in seen and len(raw) > 3:  # noqa: PLR2004
            broadened.append(raw)
            seen.add(raw.lower())

        return broadened[:3]

    async def _generate_refined_queries(
        self,
        model: str,
        query: str,
        verify_issues: str,
        num_queries: int = _AUTO_SEARCH_QUERIES,
    ) -> list[str]:
        """Generate refined search queries based on VERIFY feedback.

        Uses verification issues to target missing information.
        Falls back to :meth:`_broaden_queries` if the LLM call fails or
        the model produces refusal text.
        """
        from augmentum.modes.analytical.prompts import SEARCH_RETRY_PROMPT

        prompt = SEARCH_RETRY_PROMPT.format(
            query=query, issues=verify_issues,
            datetime_context=get_datetime_context(),
        )
        request = InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content=query),
            ],
            stream=False,
        )
        try:
            response = await self._backend.chat(request)
            raw = response.message.content if response.message else ""
        except Exception as exc:
            log.warning("search_retry_query_gen_failed", error=str(exc))
            raw = ""

        # Parse lines, filtering out refusal boilerplate
        queries: list[str] = []
        for line in raw.splitlines():
            cleaned = _QUERY_LINE_PREFIX.sub("", line).strip().rstrip("\"'")
            if not cleaned or len(cleaned) <= 3:  # noqa: PLR2004
                continue
            if self._is_refusal(cleaned):
                log.debug("search_retry_query_line_refusal", line=cleaned[:100])
                continue
            if _PREAMBLE_PATTERN.search(cleaned):
                log.debug("search_retry_query_line_preamble", line=cleaned[:100])
                continue
            queries.append(cleaned)
            if len(queries) >= num_queries:
                break

        if not queries:
            log.warning("search_retry_query_gen_empty_fallback", raw=raw[:200])
            queries = self._broaden_queries(self._state.search_queries, query)

        log.info("search_retry_queries_generated", queries=queries)
        return queries

    def _merge_search_context(self, new_context: str) -> None:
        """Merge new search results into existing search_context.

        Appends new result blocks, deduplicating by URL.  Respects the
        configured ``uarf_auto_search_max_context_chars`` budget.
        """
        if not new_context:
            return
        if not self._state.search_context:
            self._state.search_context = new_context
            return

        # Collect URLs already in the existing context
        existing_urls: set[str] = set()
        for line in self._state.search_context.splitlines():
            stripped = line.strip()
            if stripped.startswith("URL:"):
                existing_urls.add(stripped[4:].strip())

        # Filter new blocks: skip any whose URL is already present
        new_blocks: list[str] = []
        for block in new_context.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            url = ""
            for line in block.splitlines():
                if line.strip().startswith("URL:"):
                    url = line.strip()[4:].strip()
                    break
            if url and url in existing_urls:
                continue
            new_blocks.append(block)

        if not new_blocks:
            return

        appended = "\n\n".join(new_blocks)
        max_chars = settings.uarf_auto_search_max_context_chars

        existing = self._state.search_context
        if "[... truncated]" in existing:
            existing = existing[: existing.index("[... truncated]")]

        combined = existing.rstrip() + "\n\n" + appended
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n[... truncated]"

        self._state.search_context = combined

    @staticmethod
    def _heuristic_assess(query: str) -> str | None:
        """Return complexity if unambiguous from surface features, else None.

        Allows skipping the LLM ASSESS call for ~40% of queries that are
        obviously simple or complex based on length, structure, and keywords.
        """
        q = query.strip().lower()
        words = q.split()

        # Complex: multiple sub-questions (check first — takes priority)
        if q.count("?") >= 2:
            return "complex"

        # Complex: comparison / analysis signals in longer queries
        if len(words) > 20 and any(w in q for w in (
            "compare", "contrast", "analyze the impact", "evaluate the",
        )):
            return "complex"

        # Simple: short factual questions
        if len(words) <= 10 and any(
            q.startswith(w) for w in (
                "what is", "who is", "where is", "when was", "define ",
            )
        ):
            return "simple"

        # Ambiguous — fall through to LLM
        return None

    async def process(self, request: InternalChatRequest) -> AnalyticalResult:
        """Run the full UARF pipeline on the request.

        Returns an AnalyticalResult with the final conclusion and all phase outputs.
        """
        self._current_request = request
        # Extract the user query from the request
        self._state.query = self._extract_query(request)
        model = request.model

        # Build conversation context from prior messages
        conv_raw = self._build_conversation_context(
            request,
            max_turns=settings.uarf_conversation_turns,
            max_chars=settings.uarf_conversation_max_chars,
        )
        if conv_raw:
            self._state.conversation_context = (
                "\n## Conversation History\n"
                "The following is the recent conversation for context. "
                "Use it to resolve references (e.g. 'that', 'it', 'the same') "
                "but focus your analysis on the current query.\n\n"
                f"{conv_raw}\n"
            )

        # Thread memory availability hint into context if present
        if request.memory_hint:
            hint_section = f"\n## User Memory Context\n{request.memory_hint}\n"
            self._state.conversation_context = (
                hint_section + self._state.conversation_context
            )

        # Phase 1: ASSESS — try heuristic first, fall back to LLM
        heuristic_complexity = (
            self._heuristic_assess(self._state.query)
            if settings.uarf_heuristic_assess else None
        )

        if heuristic_complexity is not None:
            # Synthesize a minimal ASSESS output for downstream phases
            assess_output = (
                "TYPE: heuristic\n"
                "DOMAIN: general\n"
                "REASONING_STEPS: 1\n"
                f"COMPLEXITY: {heuristic_complexity}\n"
                "RATIONALE: Determined by surface-level heuristic (no LLM call)."
            )
            self._state.phase_results[AnalyticalPhase.ASSESS.value] = PhaseResult(
                phase=AnalyticalPhase.ASSESS,
                output=assess_output,
                tokens_used=0,
            )
            self._state.complexity = heuristic_complexity
            log.info(
                "analytical_heuristic_assess",
                complexity=heuristic_complexity,
                query_preview=self._state.query[:80],
            )
        else:
            assess_result = await self._run_phase(
                AnalyticalPhase.ASSESS,
                model=model,
                query=self._state.query,
            )
            self._state.complexity = self._parse_complexity(assess_result.output)

        is_simple = self._state.complexity == "simple"

        log.info(
            "analytical_assess_complete",
            complexity=self._state.complexity,
            query_preview=self._state.query[:80],
        )

        # Auto-search: detect and execute before downstream phases.
        # Skip when the user curated tools and didn't include web_search —
        # auto-search shouldn't force a tool the user opted out of.
        web_search_allowed = (
            self._enabled_tools is None
            or "web_search" in self._enabled_tools
        )
        assess_output = self._get_phase_output(AnalyticalPhase.ASSESS)
        if (
            settings.uarf_auto_search
            and self._tool_registry
            and self._tool_registry.get("web_search")
            and web_search_allowed
            and self._needs_search(self._state.query, assess_output)
        ):
            self._state.needs_search = True
            queries = await self._generate_search_queries(
                model,
                self._state.query,
                num_queries=settings.uarf_auto_search_queries,
            )
            await self._execute_auto_search(
                queries,
                results_per_query=settings.uarf_auto_search_results_per_query,
                max_context_chars=settings.uarf_auto_search_max_context_chars,
            )

            # System-level search retry: too few usable results
            if (
                self._state.search_result_count < settings.uarf_search_retry_min_results
                and self._state.search_retry_count < settings.uarf_search_retry_max
            ):
                self._state.search_retry_count += 1
                broadened = self._broaden_queries(
                    self._state.search_queries, self._state.query,
                )
                if broadened:
                    log.info(
                        "search_retry_system",
                        reason="insufficient_results",
                        original_count=self._state.search_result_count,
                        retry_queries=broadened,
                    )
                    existing_context = self._state.search_context
                    await self._execute_auto_search(
                        broadened,
                        results_per_query=settings.uarf_auto_search_results_per_query,
                        max_context_chars=settings.uarf_auto_search_max_context_chars,
                    )
                    new_context = self._state.search_context
                    self._state.search_context = existing_context
                    self._merge_search_context(new_context)

        if is_simple:
            # Simple: ASSESS -> RESPOND (2 calls)
            return await self._run_simple_pipeline(model)

        if self._state.complexity == "moderate":
            # Moderate: ASSESS -> GATHER -> APPLY -> VERIFY (4 calls)
            return await self._run_moderate_pipeline(model)

        # Complex: ASSESS -> IDENTIFY -> RELEVANT -> APPLY -> VERIFY -> CONCLUDE (6 calls)
        return await self._run_full_pipeline(model)

    async def _run_simple_pipeline(self, model: str) -> AnalyticalResult:
        """Run the shortened pipeline for simple queries.

        Uses a single merged RESPOND call instead of separate APPLY + CONCLUDE.
        The response is the final answer directly — no intermediate synthesis.
        Tools are still available for search/calc if needed.
        """
        # Single RESPOND call with tool access
        await self._run_phase_with_tools(
            AnalyticalPhase.RESPOND,
            model=model,
            query=self._state.query,
            is_simple=True,
        )

        # Copy RESPOND output to both APPLY and CONCLUDE for consistency
        # (downstream code and UI expect these phases to exist)
        respond_output = self._get_phase_output(AnalyticalPhase.RESPOND)
        self._state.phase_results[AnalyticalPhase.APPLY.value] = PhaseResult(
            phase=AnalyticalPhase.APPLY,
            output=respond_output,
            tokens_used=0,
        )
        self._state.phase_results[AnalyticalPhase.CONCLUDE.value] = PhaseResult(
            phase=AnalyticalPhase.CONCLUDE,
            output=respond_output,
            tokens_used=0,
        )

        return self._build_result()

    async def _run_moderate_pipeline(self, model: str) -> AnalyticalResult:
        """Run the moderate pipeline: ASSESS -> GATHER -> APPLY -> VERIFY.

        GATHER merges IDENTIFY + RELEVANT into one call.  APPLY output is
        used as the final answer (no separate CONCLUDE) unless verification
        fails, which triggers backtracking.
        """
        # GATHER (merged IDENTIFY+RELEVANT with tool access)
        await self._run_phase_with_tools(
            AnalyticalPhase.GATHER,
            model=model,
            query=self._state.query,
        )
        gather_output = self._get_phase_output(AnalyticalPhase.GATHER)

        # Copy GATHER to IDENTIFY slot so downstream (APPLY, VERIFY) can
        # reference it via the standard identify_output parameter.
        self._state.phase_results[AnalyticalPhase.IDENTIFY.value] = PhaseResult(
            phase=AnalyticalPhase.IDENTIFY,
            output=gather_output,
            tokens_used=0,
        )

        # APPLY — verify+backtrack removed (training-cutoff bias caused the
        # LLM reviewer to reject real search results as fabricated).
        await self._run_phase_with_tools(
            AnalyticalPhase.APPLY,
            model=model,
            query=self._state.query,
            identify_output=gather_output,
            relevant_output="",  # absorbed into GATHER
        )

        # Use APPLY output directly as the conclusion (skip CONCLUDE call)
        apply_output = self._get_phase_output(AnalyticalPhase.APPLY)
        self._state.phase_results[AnalyticalPhase.CONCLUDE.value] = PhaseResult(
            phase=AnalyticalPhase.CONCLUDE,
            output=apply_output,
            tokens_used=0,
        )

        return self._build_result()

    async def _run_full_pipeline(self, model: str) -> AnalyticalResult:
        """Run the full 6-phase pipeline for complex queries."""
        assess_output = self._get_phase_output(AnalyticalPhase.ASSESS)

        # IDENTIFY
        await self._run_phase(
            AnalyticalPhase.IDENTIFY,
            model=model,
            query=self._state.query,
            assess_output=assess_output,
        )
        identify_output = self._get_phase_output(AnalyticalPhase.IDENTIFY)

        # RELEVANT (with tools if available)
        await self._run_phase_with_tools(
            AnalyticalPhase.RELEVANT,
            model=model,
            query=self._state.query,
            identify_output=identify_output,
        )
        relevant_output = self._get_phase_output(AnalyticalPhase.RELEVANT)

        # APPLY — verify+backtrack removed (training-cutoff bias caused the
        # LLM reviewer to reject real search results as fabricated).
        await self._run_phase_with_tools(
            AnalyticalPhase.APPLY,
            model=model,
            query=self._state.query,
            identify_output=identify_output,
            relevant_output=relevant_output,
        )

        # CONCLUDE synthesizes from APPLY only (no verification input).
        apply_output = self._get_phase_output(AnalyticalPhase.APPLY)
        await self._run_phase(
            AnalyticalPhase.CONCLUDE,
            model=model,
            query=self._state.query,
            apply_output=apply_output,
        )

        return self._build_result()

    async def _run_phase_with_tools(
        self,
        phase: AnalyticalPhase,
        *,
        model: str,
        query: str = "",
        assess_output: str = "",
        identify_output: str = "",
        relevant_output: str = "",
        apply_output: str = "",
        verify_output: str = "",
        backtrack_context: str = "",
        is_simple: bool = False,
    ) -> PhaseResult:
        """Run a phase with optional tool-calling loop.

        Automatically selects the best tool-calling tier for the backend
        and model:

        - **Tier 1 (Native):** Pass tool definitions via ``tools`` API param,
          parse ``tool_calls`` from response.  Highest reliability.
        - **Tier 2 (Structured):** Use Ollama ``format`` JSON Schema for
          grammar-constrained output.  Guarantees valid JSON.
        - **Tier 3 (Text):** Parse ``TOOL_CALL:`` / ``TOOL_INPUT:`` markers
          from raw text.  Universal fallback.

        Falls back to the plain ``_run_phase`` path when no tools are
        available or the model doesn't request any.
        """
        # Determine available tools for this phase.
        # When auto-search already ran, exclude web_search from the tool loop
        # so the model can't attempt (and fail at) manual search calls.
        exclude = frozenset({"web_search"}) if self._state.needs_search else None
        tools = (
            self._tool_registry.get_for_phase(
                phase.value,
                exclude=exclude,
                allowed_names=self._enabled_tools,
            )
            if self._tool_registry
            else []
        )

        # Pre-filter tools based on query analysis (3-5% accuracy gain per removed tool)
        if tools and settings.tool_prefilter_enabled and self._state.query:
            from augmentum.tools.filter import filter_tools_for_query
            tools = filter_tools_for_query(
                self._state.query, tools,
                min_tools=settings.tool_prefilter_min_tools,
            )

        # Select tool-calling tier
        tier = select_tier(self._backend, model) if tools else ToolCallingTier.TEXT
        if tools:
            log.info(
                "tool_tier_selected",
                phase=phase.value,
                tier=tier.value,
                tool_count=len(tools),
            )

        # Add proactive suggestions to the query context
        proactive_hint = ""
        if tools and phase in (AnalyticalPhase.RELEVANT, AnalyticalPhase.APPLY, AnalyticalPhase.RESPOND):
            suggestions = self._get_proactive_suggestions(query, tools=tools)
            if suggestions:
                proactive_hint = (
                    "\n\n## Proactive Suggestions\n"
                    "Based on the query, consider using these tools:\n"
                    + "\n".join(f"- {s}" for s in suggestions)
                )

        # Build the system + user prompts. The shared prefix (datetime +
        # query + conv context) lives in system for prefix-cache reuse
        # across phases; search_context is phase-scoped and stays in user.
        system_prompt, user_content = get_phase_prompt(
            phase.value,
            query=query,
            assess_output=assess_output,
            identify_output=identify_output,
            relevant_output=relevant_output,
            apply_output=apply_output,
            verify_output=verify_output,
            backtrack_context=backtrack_context,
            is_simple=is_simple,
            search_context=self._state.search_context,
            conversation_context=self._state.conversation_context,
            auto_verify_summary=self._state.auto_verify_summary if phase == AnalyticalPhase.VERIFY else "",
            datetime_ctx=self._run_datetime_ctx,
        )

        # Tier-specific prompt and request setup
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

        if proactive_hint:
            system_prompt += proactive_hint

        # Inject artifact template context when artifact tools are available
        if tools:
            artifact_tool_names = [t.name for t in tools if hasattr(t, 'category') and str(getattr(t, 'category', '')) == 'ToolCategory.ARTIFACT']
            if not artifact_tool_names:
                # Fallback: check by name
                artifact_tool_names = [t.name for t in tools if t.name in ('create_document', 'create_presentation', 'create_spreadsheet', 'create_chart')]
            if artifact_tool_names:
                try:
                    from augmentum.tools.artifact_templates import get_template_for_tool_call
                    for tn in artifact_tool_names:
                        tpl_ctx = get_template_for_tool_call(tn, query)
                        if tpl_ctx:
                            system_prompt += f"\n\n## Design Template\n{tpl_ctx}"
                            break
                except Exception as exc:
                    log.warning("analytical_phase_template_inject_failed", error=str(exc))

        # Initial LLM call
        self._state.current_phase = phase
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]
        phase_request = InternalChatRequest(
            model=model, messages=list(messages), stream=False,
            tools=native_tools,
            format=structured_schema,
        )
        response = await self._backend.chat(phase_request)
        output = response.message.content if response.message else ""
        total_tokens = response.usage.total_tokens if response.usage else 0

        # Tool-call loop
        allowed_tool_names = {t.name for t in tools}
        tool_calls_made = 0
        while tools and tool_calls_made < _MAX_TOOL_CALLS_PER_PHASE:
            # Parse tool call(s) using universal parser (full waterfall
            # with affinity cache: native/structured/text/xml/react/json)
            from augmentum.tools.parsing import parse_tool_calls as _universal_parse

            parsed = _universal_parse(response, tools, self._backend)
            parsed_calls: list[tuple[str, dict]] = [
                (p.name, p.args) for p in parsed
            ]

            # Structured tier: no tool call may mean model chose text_response
            if (
                not parsed_calls
                and tier == ToolCallingTier.STRUCTURED
                and output
            ):
                output = extract_structured_text(output)
                break

            if not parsed_calls:
                break

            # Validate and resolve all tool calls
            validated_calls: list[tuple[str, dict, object]] = []  # (name, input, tool)
            for tc_name, tc_input in parsed_calls:
                resolved = (
                    self._tool_registry.resolve(tc_name)
                    if self._tool_registry else None
                )
                if resolved is None or resolved.name not in allowed_tool_names:
                    log.info(
                        "tool_not_allowed_for_phase",
                        phase=phase.value,
                        tool=tc_name,
                        resolved=resolved.name if resolved else None,
                        allowed=list(allowed_tool_names),
                    )
                    continue
                tc_input = coerce_tool_params(resolved, tc_input)
                validated_calls.append((resolved.name, tc_input, resolved))

            if not validated_calls:
                break

            # Check budget — don't exceed max tool calls
            remaining = _MAX_TOOL_CALLS_PER_PHASE - tool_calls_made
            validated_calls = validated_calls[:remaining]

            # Execute tools — parallel if multiple, sequential if one
            if len(validated_calls) == 1:
                tc_name, tc_input, _ = validated_calls[0]
                tool_results = [
                    (tc_name, await self._execute_tool(
                        phase.value, tc_name, tc_input, exclude=exclude,
                    ))
                ]
            else:
                # Parallel execution of independent tool calls
                log.info(
                    "parallel_tool_execution",
                    phase=phase.value,
                    count=len(validated_calls),
                    tools=[c[0] for c in validated_calls],
                )

                async def _safe_execute(name: str, inp: dict) -> tuple[str, ToolResult]:
                    try:
                        result = await self._execute_tool(
                            phase.value, name, inp, exclude=exclude,
                        )
                    except Exception as exc:
                        log.warning("parallel_tool_failed", tool=name, error=str(exc))
                        result = ToolResult(success=False, error=str(exc))
                    return (name, result)

                tool_results = await asyncio.gather(
                    *[_safe_execute(c[0], c[1]) for c in validated_calls],
                )

            tool_calls_made += len(tool_results)

            # Build combined result message
            result_parts = []
            for tc_name, tc_result in tool_results:
                status = "Success" if tc_result.success else "Error"
                result_parts.append(
                    f"## Tool Result ({tc_name})\n"
                    f"{status}: {tc_result.output or tc_result.error}"
                )

            messages.append(Message(role="assistant", content=output))
            messages.append(
                Message(
                    role="user",
                    content=(
                        "\n\n".join(result_parts) + "\n\n"
                        "Continue your analysis incorporating this information."
                    ),
                ),
            )
            phase_request = InternalChatRequest(
                model=model, messages=list(messages), stream=False,
                tools=native_tools,
                format=structured_schema,
            )
            response = await self._backend.chat(phase_request)
            output = response.message.content if response.message else ""
            if response.usage:
                total_tokens += response.usage.total_tokens

        # Parse phase-specific outputs
        confidence = 0.0
        needs_backtrack = False
        backtrack_reason = ""

        if phase == AnalyticalPhase.VERIFY:
            confidence = self._parse_confidence(output)
            verified = self._parse_verified(output)
            needs_backtrack = not verified or confidence < _get_confidence_threshold()
            if needs_backtrack:
                backtrack_reason = self._extract_verification_issues(output)

        result = PhaseResult(
            phase=phase,
            output=output,
            confidence=confidence,
            needs_backtrack=needs_backtrack,
            backtrack_reason=backtrack_reason,
            tokens_used=total_tokens,
        )
        self._state.phase_results[phase.value] = result

        log.info(
            "analytical_phase_complete",
            phase=phase.value,
            tier=tier.value if tools else "none",
            tokens=total_tokens,
            tool_calls=tool_calls_made,
            confidence=confidence if phase == AnalyticalPhase.VERIFY else None,
        )

        return result

    async def _execute_tool(
        self, phase: str, tool_name: str, tool_input: dict,
        *,
        exclude: frozenset[str] | None = None,
    ) -> ToolResult:
        """Execute a tool and record the call in state.

        Uses ``ToolRegistry.resolve()`` for fuzzy name matching so that
        small models that write ``search`` instead of ``web_search``
        still trigger the correct tool.

        Args:
            exclude: Optional set of tool names to reject.  Used to
                prevent redundant web_search calls when auto-search
                already ran.
        """
        tool = self._tool_registry.resolve(tool_name) if self._tool_registry else None

        # Block excluded tools (e.g. web_search after auto-search)
        if tool is not None and exclude and tool.name in exclude:
            error_msg = (
                f"Tool '{tool.name}' is not available in this phase "
                f"(search was already performed automatically). "
                f"Use the search results provided in the context instead."
            )
            record = ToolCallRecord(
                phase=phase,
                tool_name=tool.name,
                input_data=tool_input,
                output=error_msg,
                success=False,
            )
            self._state.tool_calls.append(record)
            log.info(
                "tool_excluded",
                phase=phase,
                tool=tool.name,
                reason="auto_search_already_ran",
            )
            return ToolResult(success=False, error=error_msg, validation_error=True)

        if tool is None:
            available = (
                [t.name for t in self._tool_registry.list_tools()]
                if self._tool_registry
                else []
            )
            error_msg = (
                f"Unknown tool: {tool_name}. "
                f"Available tools: {', '.join(available)}"
            )
            record = ToolCallRecord(
                phase=phase,
                tool_name=tool_name,
                input_data=tool_input,
                output=error_msg,
                success=False,
            )
            self._state.tool_calls.append(record)
            return ToolResult(success=False, error=error_msg, validation_error=True)

        # Use canonical name for all downstream records / logging
        if tool.name != tool_name:
            log.info(
                "tool_name_resolved",
                raw=tool_name,
                resolved=tool.name,
                phase=phase,
            )
            tool_name = tool.name

        # --- Artifact pipeline intercept ---
        from augmentum.tools.artifact_pipeline import ARTIFACT_TOOLS

        if tool_name in ARTIFACT_TOOLS:
            try:
                from augmentum.tools.artifact_pipeline import (
                    ArtifactRequest,
                    PipelineContext,
                    build_backend_pipeline_caller,
                    execute_artifact_pipeline,
                )

                current_req = getattr(self, "_current_request", None)

                fmt_map = {
                    "create_document": tool_input.get("format", "pdf"),
                    "create_presentation": "pptx",
                    "create_spreadsheet": "xlsx",
                    "create_chart": "chart",
                }
                fmt = fmt_map.get(tool_name, "pdf")
                topic = tool_input.get("title", tool_input.get("topic", "Document"))

                art_req = ArtifactRequest(
                    format=fmt,
                    topic=topic,
                    title=tool_input.get("title"),
                    theme=tool_input.get("theme"),
                    tool_params=tool_input,
                )

                msg_history = []
                if current_req and current_req.messages:
                    msg_history = [
                        {"role": m.role, "content": m.content}
                        for m in current_req.messages
                    ]

                ctx = PipelineContext(
                    message_history=msg_history,
                    tool_results=list(self._state.tool_calls),
                )

                model = current_req.model if current_req else ""
                caller = build_backend_pipeline_caller(self._backend, model=model)

                # Resolve render tools from registry
                render_tools: dict = {}
                search_tool = None
                fetch_tool = None
                if self._tool_registry:
                    for tname in ("create_document", "create_presentation",
                                  "create_spreadsheet", "create_chart"):
                        t = self._tool_registry.resolve(tname)
                        if t:
                            render_tools[tname] = t
                    search_tool = self._tool_registry.resolve("web_search")
                    fetch_tool = self._tool_registry.resolve("web_fetch")

                pipeline_result = await execute_artifact_pipeline(
                    art_req, ctx, caller,
                    _search_tool=search_tool,
                    _fetch_tool=fetch_tool,
                    _render_tools=render_tools,
                )

                output = (
                    f"Created {pipeline_result.display_name}: {pipeline_result.download_url}"
                    if pipeline_result.download_url
                    else f"Created artifact: {pipeline_result.artifact_id}"
                )

                record = ToolCallRecord(
                    phase=phase,
                    tool_name=tool_name,
                    input_data=tool_input,
                    output=output,
                    success=True,
                )
                self._state.tool_calls.append(record)

                return ToolResult(
                    success=True,
                    output=output,
                    metadata=pipeline_result.metadata,
                )
            except Exception as exc:
                log.warning("artifact_pipeline_fallthrough",
                            tool=tool_name, error=str(exc))
                # Fall through to normal execution

        # Validate required fields from the tool's input schema
        schema = tool.input_schema or {}
        required_fields = schema.get("required", [])
        known_fields = set(schema.get("properties", {}).keys())
        missing = [f for f in required_fields if f not in tool_input]
        if missing:
            # Build a helpful error message showing required params and their types
            props = schema.get("properties", {})
            type_map = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
            param_hints = []
            for f in required_fields:
                ptype = props.get(f, {}).get("type", "string")
                short = type_map.get(ptype, ptype)
                param_hints.append(f'{f}="<{short}>"')
            error_msg = (
                f"Tool '{tool_name}' requires: {', '.join(required_fields)}. "
                f"Correct format:\n"
                f"{tool_name}({', '.join(param_hints)})"
            )
            record = ToolCallRecord(
                phase=phase,
                tool_name=tool_name,
                input_data=tool_input,
                output=error_msg,
                success=False,
            )
            self._state.tool_calls.append(record)
            log.warning(
                "tool_missing_required_input",
                phase=phase,
                tool=tool_name,
                missing=missing,
            )
            return ToolResult(success=False, error=error_msg, validation_error=True)

        # Strip unknown kwargs — prevents "got an unexpected keyword argument"
        if known_fields:
            extra = set(tool_input.keys()) - known_fields
            if extra:
                log.info(
                    "tool_input_stripped_extra",
                    phase=phase,
                    tool=tool_name,
                    extra=list(extra),
                )
                tool_input = {k: v for k, v in tool_input.items() if k in known_fields}

        # Reject placeholder values that small models copy from the prompt
        # e.g. "<code>", "<query>", "<url>", "<value>", "value", "param"
        placeholder_values = {"<code>", "<query>", "<url>", "<value>", "<key>",
                              "<string>", "<expression>", "value", "param"}
        placeholder_found = []
        for k in required_fields:
            v = tool_input.get(k)
            if isinstance(v, str) and (
                v in placeholder_values
                or (v.startswith("<") and v.endswith(">"))
            ):
                placeholder_found.append(f"{k}={v!r}")
        if placeholder_found:
            error_msg = (
                f"Tool '{tool_name}' was called with placeholder values: "
                f"{', '.join(placeholder_found)}. "
                f"Replace them with actual values."
            )
            record = ToolCallRecord(
                phase=phase,
                tool_name=tool_name,
                input_data=tool_input,
                output=error_msg,
                success=False,
            )
            self._state.tool_calls.append(record)
            log.warning(
                "tool_placeholder_values",
                phase=phase,
                tool=tool_name,
                placeholders=placeholder_found,
            )
            return ToolResult(success=False, error=error_msg, validation_error=True)

        # --- Circuit breaker: skip tools with too many recent failures ---
        if (
            self._circuit_breaker
            and settings.tool_circuit_breaker_enabled
            and self._circuit_breaker.is_open(tool.name)
        ):
            # Carry the underlying cause, not just the symptom — otherwise
            # the breaker masks the real bug from the model AND from
            # whoever reads the transcript afterwards.
            _cause = self._circuit_breaker.last_error(tool.name)
            raw_err = (
                f"Tool '{tool.name}' is temporarily unavailable "
                f"(too many recent failures)."
            )
            if _cause:
                raw_err += f" Last error: {_cause}"
            error_msg = tool.enrich_error(raw_err, tool_input)
            record = ToolCallRecord(
                phase=phase, tool_name=tool_name,
                input_data=tool_input, output=error_msg, success=False,
            )
            self._state.tool_calls.append(record)
            log.info("tool_circuit_breaker_open", tool=tool.name, phase=phase)
            return ToolResult(success=False, error=error_msg)

        # --- Cache: skip re-execution for identical calls ---
        tool_cacheable = getattr(tool, "cacheable", False) is True
        tool_cache_ttl = getattr(tool, "cache_ttl", 300.0)
        if not isinstance(tool_cache_ttl, (int, float)):
            tool_cache_ttl = 300.0
        if self._tool_cache and tool_cacheable:
            cached = self._tool_cache.get(tool.name, tool_input, tool_cache_ttl)
            if cached is not None:
                record = ToolCallRecord(
                    phase=phase, tool_name=tool_name,
                    input_data=tool_input,
                    output=cached.output if cached.success else cached.error,
                    success=cached.success,
                )
                self._state.tool_calls.append(record)
                log.info("tool_cache_hit", tool=tool.name, phase=phase)
                if self._tool_registry:
                    self._tool_registry.metrics.record(
                        tool.name, success=cached.success, elapsed_ms=0, cached=True,
                    )
                return cached

        # --- Execute with per-tool timeout ---
        tool_timeout = getattr(tool, "timeout", None)
        if not isinstance(tool_timeout, (int, float)):
            tool_timeout = 30.0
        import time as _time
        _t0 = _time.monotonic()
        try:
            # ``invoke_tool``, not ``execute``: invoke is the mandated
            # dispatch seam (param coercion, list-for-string fan-out, typed
            # failures), and the helper tolerates MCP-bridged/duck-typed
            # registry entries that only expose ``execute``.
            #
            # ``_user_id`` is what every user-scoped tool reads via
            # ``extract_user_id``. Analytical passed neither seam nor id, so
            # all 38 user-scoped tools failed here 100% of the time
            # ("No user context", "ArtifactStore.save requires a user_id").
            result = await asyncio.wait_for(
                invoke_tool(tool, {**tool_input, "_user_id": self._user_id}),
                timeout=tool_timeout,
            )
        except TimeoutError:
            raw_err = f"Tool '{tool_name}' timed out after {tool_timeout}s"
            error_msg = tool.enrich_error(raw_err, tool_input)
            log.warning("tool_timeout", tool=tool_name, timeout=tool_timeout)
            result = ToolResult(success=False, error=error_msg)
        except Exception as exc:
            error_msg = tool.enrich_error(str(exc), tool_input)
            log.warning("tool_execution_failed", tool=tool_name, error=str(exc), exc_info=True)
            result = ToolResult(success=False, error=error_msg)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000

        # --- Metrics: record call outcome ---
        if self._tool_registry:
            self._tool_registry.metrics.record(
                tool.name, success=result.success, elapsed_ms=_elapsed_ms,
            )

        # --- Circuit breaker: record success/failure ---
        if self._circuit_breaker and settings.tool_circuit_breaker_enabled:
            if result.success:
                self._circuit_breaker.record_success(tool.name)
            else:
                self._circuit_breaker.record_failure(
                    tool.name, error=result.error or "",
                )

        # --- Cache: store successful results ---
        if self._tool_cache and tool_cacheable and result.success:
            self._tool_cache.put(tool.name, tool_input, result)

        # --- Truncate long results to save context window ---
        if result.output and len(result.output) > settings.tool_result_max_chars:
            from augmentum.tools.result_processing import truncate_tool_result
            result = ToolResult(
                success=result.success,
                output=truncate_tool_result(
                    result.output,
                    max_chars=settings.tool_result_max_chars,
                    tail_chars=settings.tool_result_truncation_tail,
                ),
                error=result.error,
                metadata=result.metadata,
            )

        record = ToolCallRecord(
            phase=phase,
            tool_name=tool_name,
            input_data=tool_input,
            output=result.output if result.success else result.error,
            success=result.success,
            card=result.card if result.success else None,
        )
        self._state.tool_calls.append(record)

        log.info(
            "tool_executed",
            phase=phase,
            tool=tool_name,
            success=result.success,
        )
        return result

    # ------------------------------------------------------------------
    # Tool-call parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_call(output: str) -> tuple[str, dict]:
        """Parse a tool-call block from the LLM response.

        Designed to be forgiving of small-model formatting mistakes:

        * Case-insensitive matching (``TOOL_CALL``, ``Tool Call``, ``tool_call``)
        * Markdown-tolerant (strips ``**``, backticks, etc.)
        * Handles ``Tool Call``, ``TOOL CALL``, ``tool-call`` with/without colon
        * Accepts ``TOOL_INPUT`` in JSON *or* ``key = value`` format
        * Recovers single-quote JSON (``{'query': 'foo'}``)
        * Extracts a bare JSON object following the tool name if no
          explicit ``TOOL_INPUT`` line is found.

        Returns ``(tool_name, input_dict)`` or ``("", {})`` if not found.
        """
        tool_name = _extract_tool_name(output)
        if not tool_name:
            return "", {}

        tool_input = _extract_tool_input(output)

        return tool_name, tool_input

    @staticmethod
    def _get_proactive_suggestions(query: str, *, tools: list | None = None) -> list[str]:
        """Suggest tools that should be used based on query content."""
        suggestions: list[str] = []
        if _URL_PATTERN.search(query):
            suggestions.append("web_fetch — the query contains a URL")
        if _MATH_PATTERN.search(query):
            suggestions.append("math_verify or calculator — the query involves math")
        if _SEARCH_PATTERN.search(query):
            suggestions.append("web_search — the query asks about facts or current info")
        if (
            _MEMORY_PATTERN.search(query)
            and tools
            and any(t.name == "memory_recall" for t in tools)
        ):
            suggestions.append(
                "memory_recall — the query references personal context or prior conversations"
            )
        if (
            _YOUTUBE_PATTERN.search(query)
            and tools
            and any(t.name == "youtube_transcript" for t in tools)
        ):
            suggestions.append(
                "youtube_transcript — the query references a YouTube video or asks for a transcript"
            )
        if (
            _WIKIPEDIA_PATTERN.search(query)
            and tools
            and any(t.name == "wikipedia" for t in tools)
        ):
            suggestions.append(
                "wikipedia — the query asks about a topic that Wikipedia can answer"
            )
        if (
            _DOCUMENT_PATTERN.search(query)
            and tools
            and any(t.name == "document_parse" for t in tools)
        ):
            suggestions.append(
                "document_parse — the query references a document file to read or analyze"
            )
        return suggestions

    # ------------------------------------------------------------------
    # Original _run_phase kept for phases that never need tools (ASSESS, CONCLUDE)
    # ------------------------------------------------------------------

    async def _run_phase(
        self,
        phase: AnalyticalPhase,
        *,
        model: str,
        query: str = "",
        assess_output: str = "",
        identify_output: str = "",
        relevant_output: str = "",
        apply_output: str = "",
        verify_output: str = "",
        backtrack_context: str = "",
        is_simple: bool = False,
    ) -> PhaseResult:
        """Run a single phase by calling the model with the phase prompt.

        Builds the prompt from templates, calls the backend, parses the
        response, stores the result in state, and returns it.
        """
        self._state.current_phase = phase

        # Build the system + user prompts for this phase. Shared prefix
        # (datetime + query + conv context) is pinned via datetime_ctx so
        # it's byte-identical across all phases of this UARF run.
        system_prompt, user_content = get_phase_prompt(
            phase.value,
            query=query,
            assess_output=assess_output,
            identify_output=identify_output,
            relevant_output=relevant_output,
            apply_output=apply_output,
            verify_output=verify_output,
            backtrack_context=backtrack_context,
            is_simple=is_simple,
            search_context=self._state.search_context,
            conversation_context=self._state.conversation_context,
            datetime_ctx=self._run_datetime_ctx,
            auto_verify_summary=self._state.auto_verify_summary if phase == AnalyticalPhase.VERIFY else "",
        )

        # Build the request
        phase_request = InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_content),
            ],
            stream=False,
        )

        # Check cache before calling backend
        if self._prompt_cache is not None:
            cached = await self._prompt_cache.get(phase_request)
            if cached is not None:
                log.debug("analytical_cache_hit", phase=phase.value)
                response = cached
            else:
                response = await self._backend.chat(phase_request)
                await self._prompt_cache.put(phase_request, response)
        else:
            # Call the backend
            response = await self._backend.chat(phase_request)
        output = response.message.content if response.message else ""

        # Calculate tokens used
        tokens_used = 0
        if response.usage:
            tokens_used = response.usage.total_tokens

        # Parse phase-specific outputs
        confidence = 0.0
        needs_backtrack = False
        backtrack_reason = ""

        if phase == AnalyticalPhase.VERIFY:
            confidence = self._parse_confidence(output)
            verified = self._parse_verified(output)
            needs_backtrack = not verified or confidence < _get_confidence_threshold()
            if needs_backtrack:
                backtrack_reason = self._extract_verification_issues(output)

        # Build and store result
        result = PhaseResult(
            phase=phase,
            output=output,
            confidence=confidence,
            needs_backtrack=needs_backtrack,
            backtrack_reason=backtrack_reason,
            tokens_used=tokens_used,
        )
        self._state.phase_results[phase.value] = result

        log.info(
            "analytical_phase_complete",
            phase=phase.value,
            tokens=tokens_used,
            confidence=confidence if phase == AnalyticalPhase.VERIFY else None,
        )

        return result

    def _build_result(self) -> AnalyticalResult:
        """Build the final AnalyticalResult from accumulated state."""
        conclude_output = self._get_phase_output(AnalyticalPhase.CONCLUDE)
        total_tokens = sum(
            r.tokens_used for r in self._state.phase_results.values()
        )

        return AnalyticalResult(
            conclusion=conclude_output,
            phase_results=dict(self._state.phase_results),
            complexity=self._state.complexity,
            total_tokens=total_tokens,
            backtrack_count=self._state.backtrack_count,
        )

    def _get_phase_output(self, phase: AnalyticalPhase) -> str:
        """Get the output text from a completed phase."""
        result = self._state.phase_results.get(phase.value)
        return result.output if result else ""

    @staticmethod
    def _extract_query(request: InternalChatRequest) -> str:
        """Extract the user query from the request messages."""
        for msg in reversed(request.messages):
            if msg.role == "user":
                return msg.content
        return ""

    @staticmethod
    def _build_conversation_context(
        request: InternalChatRequest,
        max_turns: int = 5,
        max_chars: int = 2000,
    ) -> str:
        """Build a formatted conversation history block from prior messages.

        Extracts up to *max_turns* user+assistant pairs from the request
        messages (excluding the final user message, which is the current
        query).  Each pair is formatted as::

            User: <message>
            Assistant: <response>

        Returns an empty string when there are no prior turns.  The output
        is capped at *max_chars* to stay within the context budget, and
        curly braces are escaped for safe ``str.format()`` usage.
        """
        _MSG_TRUNCATE = 300  # noqa: N806 — max chars per individual message

        # Collect non-system messages in order
        conversation = [m for m in request.messages if m.role in ("user", "assistant")]

        # The last user message is the current query — exclude it
        if not conversation or conversation[-1].role != "user":
            return ""
        prior = conversation[:-1]
        if not prior:
            return ""

        # Walk backwards collecting user+assistant pairs (most-recent first)
        pairs: list[str] = []
        i = len(prior) - 1
        while i >= 0 and len(pairs) < max_turns:
            msg = prior[i]
            if msg.role == "assistant" and i > 0 and prior[i - 1].role == "user":
                user_text = prior[i - 1].content.strip()
                asst_text = msg.content.strip()
                if len(user_text) > _MSG_TRUNCATE:
                    user_text = user_text[: _MSG_TRUNCATE - 3] + "..."
                if len(asst_text) > _MSG_TRUNCATE:
                    asst_text = asst_text[: _MSG_TRUNCATE - 3] + "..."
                pairs.append(f"User: {user_text}\nAssistant: {asst_text}")
                i -= 2
            else:
                # Unpaired message — include as-is
                text = msg.content.strip()
                if len(text) > _MSG_TRUNCATE:
                    text = text[: _MSG_TRUNCATE - 3] + "..."
                pairs.append(f"{msg.role.capitalize()}: {text}")
                i -= 1

        if not pairs:
            return ""

        # Reverse to chronological order
        pairs.reverse()
        context = "\n\n".join(pairs)

        # Cap total length
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[... earlier conversation truncated]"

        # Escape braces so str.format() on prompt templates doesn't break
        return context.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _parse_complexity(assess_output: str) -> str:
        """Parse the COMPLEXITY line from ASSESS phase output."""
        match = re.search(
            r"COMPLEXITY:\s*(simple|moderate|complex)",
            assess_output,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower()
        return "moderate"

    @staticmethod
    def _parse_confidence(verify_output: str) -> float:
        """Parse the CONFIDENCE line from VERIFY phase output.

        Handles multiple formats that models produce:
        - Decimal 0.0-1.0:  ``CONFIDENCE: 0.85``
        - Percentage:        ``CONFIDENCE: 85%``
        - Integer 0-100:     ``CONFIDENCE: 85``
        - Fraction:          ``CONFIDENCE: 0.85/1.0`` or ``85/100``
        """
        # Try decimal 0.0-1.0 first (most precise match)
        m = re.search(
            r"CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)\b", verify_output, re.IGNORECASE,
        )
        if m:
            return max(0.0, min(1.0, float(m.group(1))))

        # Try percentage: 85%, 85 percent
        m = re.search(
            r"CONFIDENCE:\s*(\d{1,3})\s*%", verify_output, re.IGNORECASE,
        )
        if m:
            return max(0.0, min(1.0, float(m.group(1)) / 100))

        # Try fraction: 0.85/1.0, 85/100
        m = re.search(
            r"CONFIDENCE:\s*([\d.]+)\s*/\s*([\d.]+)", verify_output, re.IGNORECASE,
        )
        if m:
            try:
                return max(0.0, min(1.0, float(m.group(1)) / float(m.group(2))))
            except (ValueError, ZeroDivisionError):
                pass

        # Try bare integer 0-100 (no decimal point, no % sign)
        m = re.search(
            r"CONFIDENCE:\s*(\d{1,3})\b", verify_output, re.IGNORECASE,
        )
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val / 100 if val > 1 else val))

        return 0.0

    @staticmethod
    def _parse_verified(verify_output: str) -> bool:
        """Parse the VERIFIED line from VERIFY phase output."""
        match = re.search(
            r"VERIFIED:\s*(yes|no)",
            verify_output,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower() == "yes"
        return False

    @staticmethod
    def _parse_search_needed(verify_output: str) -> bool:
        """Parse the SEARCH_NEEDED line from VERIFY phase output."""
        match = re.search(
            r"SEARCH_NEEDED:\s*(yes|no)",
            verify_output,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower() == "yes"
        return False

    @staticmethod
    def _extract_verification_issues(verify_output: str) -> str:
        """Extract error descriptions from VERIFY output."""
        issues: list[str] = []

        for section in ("ERRORS_FOUND", "UNSUPPORTED_CLAIMS", "CONTRADICTIONS"):
            pattern = rf"{section}:\s*\n((?:- .+\n?)+)"
            match = re.search(pattern, verify_output)
            if match:
                content = match.group(1).strip()
                if content.strip("- \n").lower() != "none":
                    issues.append(f"{section}:\n{content}")

        return "\n\n".join(issues) if issues else "Verification failed with low confidence."
