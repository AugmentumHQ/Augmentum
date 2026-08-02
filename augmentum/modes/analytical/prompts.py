"""Phase prompt templates for the UARF analytical pipeline.

Each phase has a system prompt (instructions only) and user content (query +
phase data).  ``get_phase_prompt()`` returns ``(system_prompt, user_content)``
so that instructions live in the system message and data lives in the user
message — no duplication.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Auto-search prompts
# ---------------------------------------------------------------------------

SEARCH_QUERY_PROMPT = """\
{datetime_context}

Write 3 short web search queries to find information for the user's LATEST request.

The latest request is often a follow-up that refers back to earlier turns — "compare both", "tell me more about it", "the second one", "do that for X too". When it does, use the recent conversation to resolve what it actually refers to and write self-contained queries that name those specific topics/entities. NEVER search a bare pronoun or stand-in ("both", "it", "them", "that one") — resolve it from the conversation first. If the latest request is already self-contained, just use it directly.

Use the current date above when the request involves recent events, news, or time-sensitive topics.
Output ONLY the 3 search queries, one per line. No explanations.

Example output:
latest news headlines today
current weather forecast
stock market performance today
{conversation}
Latest request: {query}
"""

SEARCH_CONTEXT_SECTION = """\

## [SOURCE: WEB SEARCH] Live Search Results
> Retrieved from the internet moments ago. This is REAL, CURRENT data.

RULES:
- Base your answer on these results. Cite sources when relevant.
- If results are incomplete, answer with what IS available and note gaps.
- NEVER fabricate facts not present in the results. If the results name one city, do NOT substitute a different one.
- When results contain specific data (locations, numbers, names), use them EXACTLY as stated.
- The current date is often past your training cutoff. That is EXPECTED and does NOT make these results fake. Do not refuse, hedge, or call real events "fabricated" because their date is past your knowledge cutoff — report them faithfully.

--- BEGIN WEB RESULTS ---
{search_results}
--- END WEB RESULTS ---
"""

SEARCH_CONTEXT_VERIFY_SECTION = """\

## [SOURCE: WEB SEARCH] Reference URLs
> Source URLs from web searches, for cross-referencing claims.

{search_urls}
"""

SEARCH_CONTEXT_SUMMARY_SECTION = """\

## Search Context
{result_count} web results retrieved. Full results will be available in the \
analysis phase. Topics covered: {topics}
"""

SEARCH_RETRY_PROMPT = """\
{datetime_context}

Write 3 different web search queries to find the missing information below.
Use the current date above when the question involves recent events, news, or time-sensitive topics.
Output ONLY the 3 search queries, one per line. No explanations.

Original question: {query}

Missing information:
{issues}
"""


def scope_search_context(search_context: str, phase: str) -> str:
    """Return a phase-appropriate slice of the search context.

    - **apply / respond / gather**: full search context (these do the real work)
    - **identify / relevant**: brief summary only (awareness, not data dump)
    - **verify**: only source URLs (for cross-referencing)
    - **assess / conclude / other**: no search context
    """
    if not search_context:
        return ""

    if phase in ("apply", "respond", "gather"):
        return search_context

    if phase in ("identify", "relevant"):
        return _build_search_summary(search_context)

    if phase == "verify":
        # Extract just the URLs and titles for reference checking
        urls: list[str] = []
        for line in search_context.splitlines():
            stripped = line.strip()
            if stripped.startswith("URL:"):
                urls.append(stripped)
            elif stripped.startswith("[") and "]" in stripped:
                urls.append(stripped)
        if not urls:
            return ""
        return SEARCH_CONTEXT_VERIFY_SECTION.format(search_urls="\n".join(urls))

    # assess, conclude, other phases — no search context
    return ""


def _build_search_summary(search_context: str) -> str:
    """Build a brief summary of search results for awareness phases.

    Instead of dumping 4000 chars of search results into IDENTIFY/RELEVANT,
    provide a 1-2 line summary so the model knows search happened and what
    topics were covered, without triggering re-summarization.
    """
    if not search_context:
        return ""

    # Count result blocks and extract search query topics
    result_count = 0
    topics: list[str] = []
    for line in search_context.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            result_count += 1
        if stripped.startswith('Search: "') and stripped.endswith('"'):
            topic = stripped[9:-1]  # strip 'Search: "' and trailing '"'
            if topic:
                topics.append(topic)

    if not result_count and not topics:
        return ""

    topic_str = "; ".join(topics[:3]) if topics else "various related topics"
    return SEARCH_CONTEXT_SUMMARY_SECTION.format(
        result_count=result_count or "Several",
        topics=topic_str,
    )


# ---------------------------------------------------------------------------
# Phase system prompts — concise, positive-only, professional.
#
# Design principles:
# - Tell the model what to DO, never what NOT to do
# - One clear role identity per phase
# - Minimal output format (only structure needed for parsing or downstream)
# - Word limits where outputs feed into later phases
# ---------------------------------------------------------------------------

_ASSESS_SYSTEM = """\
Classify this query's complexity. Output ONLY the format below.

TYPE: <factual|analytical|mathematical|comparison|multi-step|creative>
DOMAIN: <specific domain or "general">
REASONING_STEPS: <estimated count>
COMPLEXITY: <simple|moderate|complex>
RATIONALE: <one sentence>

Definitions:
- simple: direct factual recall, single reasoning step
- moderate: needs synthesis across 2-4 steps, single domain
- complex: multi-domain, requires decomposition, 5+ steps, or comparison

Examples:

Query: "What is the capital of France?"
TYPE: factual
DOMAIN: geography
REASONING_STEPS: 1
COMPLEXITY: simple
RATIONALE: Direct factual recall, single-step lookup

Query: "Compare the economic policies of the US and EU regarding AI regulation"
TYPE: comparison
DOMAIN: economics, technology policy
REASONING_STEPS: 5
COMPLEXITY: complex
RATIONALE: Multi-domain comparison requiring policy analysis and cross-domain reasoning
"""

_IDENTIFY_SYSTEM = """\
Decompose this query into its working parts. Concise bullet points only.

CONCEPTS:
- <key terms and entities central to the query>

UNKNOWNS:
- <what must be determined or resolved>

CONSTRAINTS:
- <boundary conditions, scope limitations>

SUB_PROBLEMS:
- <independent parts to solve, if any>

One line per item. Stay under 150 words total.
"""

_GATHER_SYSTEM = """\
Analyze what this query requires and assemble the knowledge to solve it.

1. Identify the key concepts and what must be determined
2. Gather relevant facts, principles, data, and applicable methods
3. Use a tool if you need current data or calculations
4. Flag information gaps

Output structured bullet points. Stay under 250 words.
"""

_RELEVANT_SYSTEM = """\
Gather the specific evidence and methods needed to solve this query.

For each identified concept or sub-problem:
- Relevant facts, data, principles, or formulas
- The applicable method or approach

Use a tool if you need current data or calculations.
Flag gaps where information is uncertain or missing.
Stay under 200 words.
"""

_APPLY_SYSTEM = """\
Solve this query using the evidence and methods provided.

- Address each component or sub-problem systematically
- Show your reasoning at each step
- Use tools for additional data or calculations if needed
- State assumptions explicitly
- Trust search results over your training-data priors. If a result's date is past your training cutoff, that does NOT mean it is fake — report it faithfully.

Conclude with a clear, complete answer.
"""

_APPLY_SIMPLE_SYSTEM = """\
Answer this query using the provided context and your knowledge.

Show brief reasoning, then give your answer.
Use a tool if you need current data or calculations.
Trust search results as authoritative for current events, even when the date is past your training cutoff.
Be direct and accurate.
"""

_VERIFY_SYSTEM = """\
Review this analysis for errors. You are the last check before the user sees it.

Examine:
1. Logical validity of each reasoning step
2. Calculation correctness
3. Claims without supporting evidence
4. Internal contradictions
5. Whether the conclusion follows from the reasoning

If automated verification results are provided, treat them as ground truth.

Output ONLY this format:

ERRORS_FOUND:
- <error, or "None">

UNSUPPORTED_CLAIMS:
- <claim, or "None">

CONTRADICTIONS:
- <contradiction, or "None">

VERIFIED: <yes|no>
CONFIDENCE: <0.0 to 1.0>
SEARCH_NEEDED: <yes|no>
VERIFICATION_NOTES: <one-sentence summary>
"""

_CONCLUDE_SYSTEM = """\
Write the final answer for the user.

Synthesize the analysis into one clear, polished response. Rules:
- Lead with the direct answer, then support with evidence and reasoning.
- If sources were searched, cite them inline using [1], [2], etc.
- Trust the analysis and search results provided. Do NOT second-guess facts, dates, or events based on your training cutoff — if a date is past your knowledge, that is expected and does not make the information fake.
- Never refuse or claim you "cannot access" information that the analysis has already retrieved.
- If data or calculations were involved, interpret the numbers in context.
- If artifacts were created (documents, images), include the download link prominently.
- Write in a natural, conversational tone. No phase names, no internal references.
- Do NOT start with "Based on my analysis" or "After researching" — just answer.
"""

_CONCLUDE_SIMPLE_SYSTEM = """\
Write the final answer for the user.

Present the answer clearly and directly. If search results provided sources, \
cite them inline using [1], [2]. Trust the search results even when dates are past your training cutoff. Write naturally — no preambles, no meta-commentary, no refusals.
"""

_RESPOND_SIMPLE_SYSTEM = """\
Answer this question directly.

If search results are provided, use them as your primary source and trust them — even when the date is past your training cutoff, the results are real and current.
If you need data or calculations, call a tool first.
Do not refuse or claim you cannot access information that search has already retrieved.
Be accurate, clear, and complete.
"""


def _build_tool_signature(tool) -> str:
    """Build a Python-style function signature from a tool's schema.

    Example output: ``web_search(query: str, num_results?: int)``
    """
    schema = tool.input_schema or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    params: list[str] = []
    for name, prop in props.items():
        ptype = prop.get("type", "str")
        # Map JSON Schema types to readable short forms
        type_map = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
        short_type = type_map.get(ptype, ptype)
        optional = "?" if name not in required else ""
        params.append(f"{name}{optional}: {short_type}")

    return f"{tool.name}({', '.join(params)})"


def get_tool_prompt_section(tools: list) -> str:
    """Generate a tool prompt section using Python-style function call format.

    Models are pre-trained on function call syntax from code corpora, making
    this format significantly more reliable than custom ``TOOL_CALL:`` markers.

    Args:
        tools: List of Tool instances available for the current phase.

    Returns:
        A formatted string describing tools and the calling convention,
        or an empty string if no tools are available.
    """
    if not tools:
        return ""

    lines = [
        "\n## Tools",
        "You can call tools as functions when you need real-time data, facts, or calculations.",
        "",
        "Available tools:",
    ]

    for tool in tools:
        sig = _build_tool_signature(tool)
        lines.append(f"  {sig}")
        lines.append(f"    {tool.description}")

    # Concrete example
    lines.append("")
    if any(t.name == "web_search" for t in tools):
        lines.append("Example:")
        lines.append('web_search(query="current weather in <city, region>")')
    elif tools:
        schema = tools[0].input_schema or {}
        req = schema.get("required", [])
        if req:
            ex_args = ", ".join(f'{k}="..."' for k in req)
            lines.append("Example:")
            lines.append(f"{tools[0].name}({ex_args})")

    lines.extend([
        "",
        "Call a tool on its own line. The result will be provided, then continue your response.",
    ])

    return "\n".join(lines)


def get_native_tool_prompt_section(tools: list | None = None) -> str:
    """Prompt section for Tier 1 — native function calling.

    Tool definitions are passed via the ``tools`` API parameter.  Includes
    a Python-style fallback instruction so that models which ignore native
    tool calling (e.g. DeepSeek via OpenAI-compatible API) still produce
    parseable tool calls.
    """
    section = (
        "\n\n## Tools\n"
        "You have tools available. If the query requires real-time information, "
        "current data, or calculations, call the appropriate tool before "
        "writing your analysis."
    )

    # Add fallback format so models that ignore native tool calling
    # still produce something we can parse
    if tools:
        section += (
            "\n\nIf function calling is not available, call tools as:\n"
            'tool_name(param="value")\n'
            "on its own line."
        )

    return section


def get_structured_tool_prompt_section(tools: list, schema: dict) -> str:
    """Prompt section for Tier 2 — structured JSON output.

    Explains the constrained JSON output format. The ``schema`` is passed
    via Ollama's ``format`` parameter so the model's output is grammar-
    constrained, but it still needs to know the available tools.

    Args:
        tools: List of Tool instances available for the current phase.
        schema: The JSON Schema that will be enforced by the backend.

    Returns:
        A formatted prompt string describing the structured output format.
    """
    if not tools:
        return ""

    lines = [
        "\n\n## Tools — JSON Output Mode",
        "Respond ONLY with a JSON object:",
        "",
        'To call a tool: {"action": "tool_call", "tool_name": "<name>", "tool_input": {...}}',
        'To respond: {"action": "text_response", "text": "your analysis here"}',
        "",
        "Available:",
    ]
    for tool in tools:
        schema_desc = tool.input_schema or {}
        required = schema_desc.get("required", [])
        req_hint = f" (required: {', '.join(required)})" if required else ""
        lines.append(f"- {tool.name}: {tool.description}{req_hint}")

    lines.extend([
        "",
        "Use tool_call for real-time data or calculations. "
        "Otherwise use text_response. Output ONLY the JSON.",
    ])
    return "\n".join(lines)


# Maximum characters of phase output to forward to downstream phases.
# Keeps context windows lean and prevents models from re-summarizing
# walls of text.
_PHASE_OUTPUT_CAP = 800


def cap_phase_output(output: str, max_chars: int = _PHASE_OUTPUT_CAP) -> str:
    """Truncate a phase output for forwarding to downstream phases.

    Preserves the beginning of the text (which usually contains the most
    important structured content) and appends a truncation marker.
    """
    if not output or len(output) <= max_chars:
        return output
    # Try to cut at a paragraph or line break
    cut = output[:max_chars]
    last_break = max(cut.rfind("\n\n"), cut.rfind("\n"))
    if last_break > max_chars // 2:
        cut = cut[:last_break]
    return cut.rstrip() + "\n[... condensed]"


def build_shared_prefix(
    datetime_ctx: str,
    query: str,
    conversation_context: str,
    *,
    include_conversation: bool = True,
) -> str:
    """Build the UARF run-invariant prefix prepended to every phase's system prompt.

    Identical bytes across every phase in a single UARF run lets llama-server
    (with ``cache_prompt=true``) skip KV prefill for this portion on phase
    2..N — the Tier 1.4 speedup from the engine-v2 competitive analysis.

    The conversation-context toggle preserves a historical behavior: the VERIFY
    phase suppressed conversation context to keep token budget focused on the
    analysis it was critiquing. VERIFY still gets a shared prefix, just a
    smaller one (date + query only), so VERIFY's prefix is a proper prefix of
    the other phases' prefix — which means on most templates, KV from phases
    1..N can still seed VERIFY's prefill up to where they diverge.
    """
    parts: list[str] = [datetime_ctx, "", f"## User Query\n{query}"]
    if include_conversation and conversation_context:
        parts.extend(["", conversation_context])
    return "\n\n".join(p for p in parts if p != "") + "\n\n---\n"


def get_phase_prompt(
    phase: str,
    *,
    query: str = "",
    assess_output: str = "",
    identify_output: str = "",
    relevant_output: str = "",
    apply_output: str = "",
    verify_output: str = "",
    backtrack_context: str = "",
    is_simple: bool = False,
    has_tools: bool = False,
    search_context: str = "",
    conversation_context: str = "",
    auto_verify_summary: str = "",
    datetime_ctx: str = "",
) -> tuple[str, str]:
    """Get the formatted system + user prompts for a given phase.

    System prompt is structured as ``<shared_prefix><phase_instructions>``.
    The shared prefix (datetime + query + conversation context) is identical
    across phases in a single UARF run — this enables llama-server's
    cache_prompt to skip re-prefilling it for phases 2..N.  User content
    holds phase-specific prior outputs and scoped search context.

    Args:
        phase: The phase name (assess, identify, relevant, gather, apply,
            verify, conclude, respond).
        query: The original user query.
        assess_output: Output from the ASSESS phase.
        identify_output: Output from the IDENTIFY phase (or GATHER).
        relevant_output: Output from the RELEVANT phase.
        apply_output: Output from the APPLY phase.
        verify_output: Output from the VERIFY phase.
        backtrack_context: Context from a failed verification (for retries).
        is_simple: Whether the query was assessed as simple.
        has_tools: Whether tools are available for this phase.
        search_context: Pre-fetched search results context (from auto-search).
        conversation_context: Formatted prior conversation turns for reference.
        auto_verify_summary: Automated verification results for VERIFY phase.
        datetime_ctx: Pre-computed datetime context string. Pass the same
            value across all phases of one UARF run so the shared prefix is
            byte-identical and the prefix cache actually reuses.  If omitted,
            a fresh datetime is fetched here (per-phase, prefix will NOT
            share across phases — caller should cache).

    Returns:
        ``(system_prompt, user_content)`` tuple.
    """
    system_templates = {
        "assess": _ASSESS_SYSTEM,
        "identify": _IDENTIFY_SYSTEM,
        "gather": _GATHER_SYSTEM,
        "relevant": _RELEVANT_SYSTEM,
        "apply": _APPLY_SIMPLE_SYSTEM if is_simple else _APPLY_SYSTEM,
        "verify": _VERIFY_SYSTEM,
        "conclude": _CONCLUDE_SIMPLE_SYSTEM if is_simple else _CONCLUDE_SYSTEM,
        "respond": _RESPOND_SIMPLE_SYSTEM,
    }

    phase_instructions = system_templates.get(phase, "")
    if not phase_instructions:
        return ("", "")

    if not datetime_ctx:
        # Caller didn't pin one — fetch now. This breaks cross-phase prefix
        # sharing (because the datetime string changes every second). Fine
        # for tests and one-off callers; production UARF should pin.
        from augmentum.utils.datetime_context import get_datetime_context
        datetime_ctx = get_datetime_context()

    # Shared prefix — identical across UARF phases when datetime_ctx is pinned.
    # VERIFY historically skipped conversation context; keep that behavior but
    # arrange the prefix so VERIFY's version is a proper prefix of the full
    # variant, maximizing token-level sharing.
    shared_prefix = build_shared_prefix(
        datetime_ctx,
        query,
        conversation_context,
        include_conversation=(phase != "verify"),
    )

    system = f"{shared_prefix}\n{phase_instructions}"

    # Tool use nudge for phases with tool access
    if has_tools and phase in ("apply", "relevant", "respond", "gather"):
        system += (
            "\n\nYou have access to tools. If this query requires real-time "
            "information, factual data, or calculations, call a tool first."
        )

    # ------------------------------------------------------------------
    # Build user content — ONLY phase-specific data.  Query + conversation
    # context live in the shared prefix above (system-side).
    # ------------------------------------------------------------------
    user_parts: list[str] = []

    # Scope search context to what this phase actually needs
    scoped_search = scope_search_context(search_context, phase)

    if phase == "identify":
        if assess_output:
            user_parts.append(f"## Assessment\n{cap_phase_output(assess_output)}")
        if scoped_search:
            user_parts.append(scoped_search)

    elif phase == "gather":
        # Merged IDENTIFY+RELEVANT for moderate queries — gets full search
        if scoped_search:
            user_parts.append(scoped_search)

    elif phase == "relevant":
        if identify_output:
            user_parts.append(
                f"## Identified Components\n{cap_phase_output(identify_output)}"
            )
        if scoped_search:
            user_parts.append(scoped_search)

    elif phase == "respond":
        if scoped_search:
            user_parts.append(scoped_search)

    elif phase == "apply":
        if is_simple:
            if assess_output:
                user_parts.append(
                    f"## Assessment\n{cap_phase_output(assess_output)}"
                )
        else:
            if identify_output:
                user_parts.append(
                    f"## Key Components\n{cap_phase_output(identify_output)}"
                )
            if relevant_output:
                user_parts.append(
                    f"## Evidence & Methods\n{cap_phase_output(relevant_output)}"
                )
        if scoped_search:
            user_parts.append(scoped_search)
        if backtrack_context:
            user_parts.append(backtrack_context)

    elif phase == "verify":
        if apply_output:
            # VERIFY gets the full APPLY output — it needs to see everything
            user_parts.append(f"## Analysis to Verify\n{apply_output}")
        if auto_verify_summary:
            user_parts.append(auto_verify_summary)
        if scoped_search:
            user_parts.append(scoped_search)

    elif phase == "conclude":
        if apply_output:
            user_parts.append(f"## Analysis\n{apply_output}")

    # Chat templates can be finicky with an empty user turn. When a phase
    # has no prior-output context (ASSESS always; others when upstream
    # phases produced nothing), fall back to a phase-label cue so the user
    # message has content. Kept short so it doesn't push out of the shared-
    # prefix region.
    if not user_parts:
        user_parts.append(f"Proceed with the {phase} phase.")

    return (system, "\n\n".join(user_parts))
