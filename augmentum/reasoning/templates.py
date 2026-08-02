"""Built-in reasoning flow templates.

Each template is a factory function returning a ReasoningFlow with pre-configured
steps.  Templates are seeded into the database on first run and marked as
``is_builtin=True`` (clone-only, not directly editable).
"""

from __future__ import annotations

import uuid

from augmentum.reasoning.models import FlowStep, ReasoningFlow


def _id() -> str:
    return uuid.uuid4().hex[:16]


_MEMORY_RECALL_ROLES = frozenset({
    "classify", "plan", "search", "respond", "deliver",
})


def _step(
    name: str,
    system_prompt: str,
    *,
    role: str = "analyze",
    tool_categories: list[str] | None = None,
    tool_names: list[str] | None = None,
    complexity_gate: list[str] | None = None,
    stream_to_user: bool = False,
    output_cap: int = 800,
    user_template: str = "",
    sort_order: int = 0,
    model_override: str = "",
    tool_choice: str = "",
) -> FlowStep:
    resolved_names = list(tool_names or [])

    # Only inject memory_recall into steps that benefit from user context
    # (planning, searching, responding).  Internal analysis/verify/review
    # steps work on pipeline data and don't need memory — saves tokens.
    if role in _MEMORY_RECALL_ROLES and "memory_recall" not in resolved_names:
        resolved_names.append("memory_recall")

    return FlowStep(
        id=_id(),
        sort_order=sort_order,
        name=name,
        system_prompt=system_prompt,
        user_template=user_template,
        role=role,
        tool_categories=tool_categories or [],
        tool_names=resolved_names,
        complexity_gate=complexity_gate or [],
        stream_to_user=stream_to_user,
        model_override=model_override,
        output_cap=output_cap,
        tool_choice=tool_choice,
    )


# ---------------------------------------------------------------------------
# Shared deliver prompt — base system message for all deliver-role steps.
#
# Positive framing only. Small models (3-7B) follow affirmative instructions
# reliably; negative ("Do NOT ...") constraints are frequently violated.
# Runtime delivery guidance in AgenticHandler is additive, not duplicated.
# ---------------------------------------------------------------------------

_DELIVER_SYSTEM_BASE = (
    "You are writing the final response the user sees. "
    "Respond directly to the request inside <user_request>. "
    "Lead with the key finding or completed artifact. "
    "Surface any download link prominently. "
    "Cite evidence inline with [1], [2] when prior steps gathered sources. "
    "If a step produced partial or failed results, acknowledge it briefly and honestly. "
    "Write conversationally — you are replying to the user, not narrating a pipeline. "
    "Treat <work_notes> as reference material: extract facts, then present the result in your own words."
)

_DELIVER_USER_TEMPLATE = (
    "<work_notes>\n{step:_delivery_context}\n</work_notes>\n\n"
    "<user_request>{query}</user_request>"
)


# ---------------------------------------------------------------------------
# Shared classify prompt (used by Research, Code Review, Math, etc.)
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
Classify complexity. Output ONLY:

TYPE: <factual|analytical|mathematical|comparison|multi-step|creative>
DOMAIN: <domain or "general">
COMPLEXITY: <simple|moderate|complex>
RATIONALE: <one sentence>

simple = direct recall. moderate = 2-4 steps. complex = multi-domain, 5+ steps."""


# ---------------------------------------------------------------------------
# Template: Quick Answer (default flow)
# ---------------------------------------------------------------------------

def quick_answer_flow() -> ReasoningFlow:
    """Single-step flow — lets the model answer directly with full tool access."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Quick Answer",
        description=(
            "Single step with full tool access. Best for most queries. "
            "Use specialized flows only when you need multi-step pipelines."
        ),
        icon="",
        is_builtin=True,
        escalation_flow="Research",
        steps=[
            _step(
                "Respond",
                (
                    "Answer the user's question directly.\n\n"
                    "Tool guidance — use the RIGHT tool for the task:\n"
                    "- Current events, facts, or anything time-sensitive → web_search\n"
                    "- Fetch a specific URL the user provided → web_fetch\n"
                    "- Arithmetic or unit conversion → calculator, unit_converter\n"
                    "- Equations, symbolic math, proofs → math_verify\n"
                    "- Code execution, data processing → python_exec\n"
                    "- Date/time questions → datetime\n\n"
                    "If you can answer from knowledge alone, do so without tools.\n"
                    "Be thorough but concise."
                    "\n\nUNCERTAINTY PROTOCOL:\n"
                    "If you realize this question needs research (current events, "
                    "recent data, multi-source verification), end your response with:\n"
                    "[NEEDS_RESEARCH]\n"
                    "This signals the system to offer a deeper research pass."
                ),
                role="respond",
                stream_to_user=True,
                output_cap=0,
                tool_categories=["search", "execute", "verify", "fetch"],
                sort_order=0,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Research
# ---------------------------------------------------------------------------

def research_flow() -> ReasoningFlow:
    """Heavy research flow with cross-referencing and fact-checking."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Research",
        description="Deep research with search, cross-referencing, and fact-checking. Best for current events and factual queries.",
        icon="",
        is_builtin=True,
        trigger_domains=["news", "current_events", "science", "history"],
        trigger_keywords=["news", "headlines", "latest", "current events", "what happened"],
        steps=[
            _step("Classify", _CLASSIFY_PROMPT, role="classify", output_cap=300, sort_order=0),
            _step(
                "Search",
                (
                    "Search the web for information on this topic.\n\n"
                    "1. Search for the main topic\n"
                    "2. Search for a related angle or subtopic\n"
                    "3. For each result, note: source name, key facts, URL\n\n"
                    "Aim for 3-5 sources. Prefer recent, authoritative content."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Cross-Reference",
                "Compare sources: agreements, contradictions, single-source claims, "
                "and authority/recency. Structured comparison.",
                tool_categories=["fetch"],
                output_cap=800,
                sort_order=2,
            ),
            _step(
                "Synthesize",
                "Combine cross-referenced information. Lead with well-sourced claims. "
                "Note disagreements and gaps.",
                sort_order=3,
            ),
            _step(
                "Fact-Check",
                "Verify key claims against sources. Output ONLY:\n\n"
                "VERIFIED: <yes|no>\nCONFIDENCE: <0.0 to 1.0>\n\n"
                "Do NOT re-search.",
                role="verify",
                output_cap=500,
                sort_order=4,
            ),
            _step(
                "Respond",
                "Write the final answer for the user. "
                "Cite sources with URLs. Note any uncertainty. "
                "Do NOT search again — all research is done. "
                "Do NOT echo or reference the research notes — synthesize them into a natural response.",
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=5,
                user_template="<research_notes>\n{all_outputs}\n</research_notes>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Code Review
# ---------------------------------------------------------------------------

def code_review_flow() -> ReasoningFlow:
    """Code-focused analysis pipeline."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Code Review",
        description="Structured code analysis: parse, find issues, suggest fixes, verify. Best for debugging and code review.",
        icon="",
        is_builtin=True,
        trigger_domains=["code", "programming", "debugging"],
        trigger_keywords=["debug", "review", "fix", "error", "bug", "refactor"],
        steps=[
            _step(
                "Parse Code",
                (
                    "Analyze the code:\n"
                    "1. What language and framework?\n"
                    "2. What are the key functions/classes?\n"
                    "3. What is the data flow?\n"
                    "4. What external dependencies are used?"
                ),
                output_cap=600,
                sort_order=0,
            ),
            _step(
                "Find Issues",
                (
                    "Find problems in the code. For each issue state:\n"
                    "- SEVERITY: critical / high / medium / low\n"
                    "- ISSUE: what is wrong\n"
                    "- LOCATION: which function/line\n\n"
                    "Check for: bugs, security issues, performance problems, bad practices."
                ),
                output_cap=800,
                sort_order=1,
            ),
            _step(
                "Suggest Fixes",
                (
                    "For each issue, provide a fix:\n"
                    "1. Show the corrected code in a code block\n"
                    "2. Explain why the fix works (one sentence)\n"
                    "3. Note any side effects\n\n"
                    "If the code is Python, use python_exec to test your fix."
                ),
                tool_names=["python_exec"],
                sort_order=2,
            ),
            _step(
                "Verify Fixes",
                (
                    "Review the suggested fixes:\n"
                    "1. Does each fix solve the identified issue?\n"
                    "2. Does any fix introduce new problems?\n"
                    "3. Is the code idiomatic for the language?\n\n"
                    "VERIFIED: <yes|no>\n"
                    "CONFIDENCE: <0.0 to 1.0>"
                ),
                role="verify",
                tool_names=["python_exec"],
                output_cap=500,
                sort_order=3,
            ),
            _step(
                "Respond",
                "Present the code review for the user. "
                "Lead with critical issues. Include corrected code blocks. "
                "Do NOT re-analyze — all review steps are done. "
                "Do NOT echo or reference the analysis notes — synthesize them into a natural response.",
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=4,
                user_template="<analysis_notes>\n{all_outputs}\n</analysis_notes>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Debate / Steel Man
# ---------------------------------------------------------------------------

def debate_flow() -> ReasoningFlow:
    """Forces consideration of both sides before concluding."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Debate",
        description="Steel-man both sides of a question before synthesizing. Great for opinion, ethics, and policy questions.",
        icon="",
        is_builtin=True,
        trigger_keywords=["should", "debate", "argue", "opinion", "ethics"],
        steps=[
            _step(
                "Understand Position",
                (
                    "Identify:\n"
                    "1. What is the core question?\n"
                    "2. What are the two (or more) sides?\n"
                    "3. What values or assumptions does each side hold?\n"
                    "4. What are they actually disagreeing about?"
                ),
                output_cap=400,
                sort_order=0,
            ),
            _step(
                "Argue For",
                (
                    "Make the STRONGEST case FOR the proposition.\n"
                    "Use the best arguments, evidence, and logic available.\n"
                    "Be fair — present this side as its best advocates would.\n"
                    "Use search if you need supporting evidence."
                ),
                tool_categories=["search"],
                output_cap=600,
                sort_order=1,
            ),
            _step(
                "Argue Against",
                (
                    "Make the STRONGEST case AGAINST the proposition.\n"
                    "Use the best counterarguments and evidence.\n"
                    "Be equally fair — present this side as its best advocates would.\n"
                    "Use search if you need supporting evidence."
                ),
                tool_categories=["search"],
                output_cap=600,
                sort_order=2,
            ),
            _step(
                "Synthesize",
                (
                    "Compare both cases:\n"
                    "1. Where do they agree?\n"
                    "2. Where are the real disagreements?\n"
                    "3. Which arguments are stronger and why?\n"
                    "4. Is there a middle ground?"
                ),
                output_cap=600,
                sort_order=3,
            ),
            _step(
                "Respond",
                "Present a balanced analysis for the user. "
                "Show both sides fairly. Support any conclusion with the reasoning. "
                "Do NOT re-argue — synthesis is done. "
                "Do NOT echo or reference the debate notes — synthesize them into a natural response.",
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=4,
                user_template="<debate_notes>\n{all_outputs}\n</debate_notes>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Math / Science
# ---------------------------------------------------------------------------

def math_flow() -> ReasoningFlow:
    """Math-focused pipeline with calculator verification."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Math & Science",
        description="Parse, set up, solve, and verify with calculator tools. Best for math, physics, and quantitative problems.",
        icon="",
        is_builtin=True,
        trigger_domains=["math", "mathematics", "physics", "chemistry", "engineering"],
        trigger_keywords=["calculate", "solve", "equation", "prove", "derive"],
        steps=[
            _step(
                "Parse Problem",
                (
                    "Break down the problem:\n"
                    "1. GIVEN: what information is provided?\n"
                    "2. FIND: what is being asked?\n"
                    "3. TYPE: algebra, calculus, statistics, geometry, etc.\n"
                    "4. FORMULAS: which formulas or theorems apply?"
                ),
                output_cap=400,
                sort_order=0,
            ),
            _step(
                "Set Up",
                (
                    "Write the formal setup:\n"
                    "1. Define variables (e.g., let x = ...)\n"
                    "2. Write the equations\n"
                    "3. State the approach (e.g., solve for x, integrate, etc.)\n"
                    "4. Note any assumptions"
                ),
                output_cap=400,
                sort_order=1,
            ),
            _step(
                "Solve",
                (
                    "Solve step by step. Show all work.\n\n"
                    "IMPORTANT: Use tools for ALL calculations. Do not do mental math.\n\n"
                    "For arithmetic, use calculator:\n"
                    'calculator(expression="2 * 3.14159 * 5")\n\n'
                    "For symbolic math (simplify, solve equations), use math_verify:\n"
                    'math_verify(expression="solve(x**2 - 4, x)")\n\n'
                    "For complex computations, use python_exec:\n"
                    'python_exec(code="import numpy as np; print(np.linalg.det([[1,2],[3,4]]))")'
                ),
                tool_names=["calculator", "math_verify", "python_exec"],
                tool_choice="required",
                output_cap=0,
                sort_order=2,
            ),
            _step(
                "Verify",
                (
                    "Check the solution:\n"
                    "1. Plug the answer back into the original equation\n"
                    "2. Check units and dimensions\n"
                    "3. Does the answer make intuitive sense?\n"
                    "4. Use calculator to re-check key computations\n\n"
                    "VERIFIED: <yes|no>\n"
                    "CONFIDENCE: <0.0 to 1.0>"
                ),
                role="verify",
                tool_names=["calculator", "math_verify"],
                output_cap=500,
                sort_order=3,
            ),
            _step(
                "Respond",
                "Present the solution for the user. Show setup, key steps, "
                "final answer, and verification result. Do NOT re-solve — "
                "all computation is done. "
                "Do NOT echo or reference the work notes — synthesize them into a natural response.",
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=4,
                user_template="<work_notes>\n{step:_delivery_context}\n</work_notes>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Creative Writing
# ---------------------------------------------------------------------------

def creative_flow() -> ReasoningFlow:
    """Creative writing pipeline focused on brainstorming and drafting."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Creative Writing",
        description="Brainstorm, draft, and refine. No tools needed. Prompts focus on creativity and voice.",
        icon="",
        is_builtin=True,
        trigger_domains=["writing", "creative", "fiction", "poetry"],
        trigger_keywords=["write", "story", "poem", "creative", "draft"],
        auto_search=False,
        steps=[
            _step(
                "Understand Intent",
                (
                    "What does the user want?\n"
                    "1. FORMAT: story, poem, essay, script, etc.\n"
                    "2. TONE: serious, humorous, dark, whimsical, etc.\n"
                    "3. THEMES: key ideas or constraints\n"
                    "4. AUDIENCE: who is this for?\n"
                    "5. LENGTH: short, medium, or long?"
                ),
                output_cap=300,
                sort_order=0,
            ),
            _step(
                "Brainstorm",
                (
                    "Generate 3 different approaches. For each:\n"
                    "1. CONCEPT: the core idea or hook\n"
                    "2. OPENING: a sample first line\n"
                    "3. WHY: what makes this angle interesting\n\n"
                    "Be creative. Avoid the most obvious approach."
                ),
                output_cap=600,
                sort_order=1,
            ),
            _step(
                "Draft",
                (
                    "Write the full draft using the strongest approach.\n\n"
                    "Craft it with:\n"
                    "- A strong opening that hooks the reader\n"
                    "- Vivid, specific language (not generic)\n"
                    "- Consistent voice throughout\n"
                    "- A satisfying ending\n\n"
                    "Output ONLY the prose itself. No preamble "
                    "(\"Okay, let's...\", \"Here is...\"), no notes about "
                    "your choices, no trailing questions or offers to revise. "
                    "Start with the first line of the piece."
                ),
                stream_to_user=False,
                output_cap=0,
                sort_order=2,
            ),
            _step(
                "Refine",
                (
                    "Polish this draft. Output the COMPLETE refined version — "
                    "not a list of changes. Cut unnecessary words, strengthen "
                    "weak sentences, fix transitions.\n\n"
                    "Output ONLY the polished prose. No preamble, no notes "
                    "about what you changed, no trailing questions. Start "
                    "with the first line of the piece.\n\n"
                    "Draft to polish:\n{step:Draft}"
                ),
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=3,
                user_template="Polish the draft.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Shared style block for the 2026-07 flows (Explainer / Live Lookup /
# Summarize) — targets the AI-isms observed across 107 real analytical
# turns: report-voice bleed into casual answers, restate-the-question
# openers, tool-failure apologies leaking to the user.
# ---------------------------------------------------------------------------

_CONVERSATIONAL_STYLE = (
    "STYLE RULES (hard):\n"
    "- Write like a sharp colleague talking, not a report generator.\n"
    "- No markdown headers or bold-lead formatting unless the content "
    "genuinely needs structure; short questions get prose.\n"
    "- Never open by restating the question or with 'I understand "
    "you're asking'.\n"
    "- Banned phrases: 'delve', 'crucial', 'landscape', 'multifaceted', "
    "'navigate the complexities', \"it's important to note\", "
    "'in conclusion'.\n"
    "- If a tool failed, work with what you have — never apologize "
    "about internal tooling to the user."
)


# ---------------------------------------------------------------------------
# Template: Explainer
# ---------------------------------------------------------------------------

def explainer_flow() -> ReasoningFlow:
    """Layered explanations for "what is X / how does X work" questions.

    Fills the gap the session data exposed: definitional questions were
    keyword-routing into Math & Science, whose Parse-Problem/Set-Up
    template forced GIVEN/FIND framing onto questions with nothing to
    solve — and whose Verify step restated instead of checking. This
    flow grounds with one search, explains in layers, then runs a
    verdict-format accuracy check that treats the search results as
    ground truth newer than the model's training data.

    Live-validated 2026-07-02 ("what is the Poincaré Conjecture" —
    correct statement, Perelman/Ricci-flow history, clean verdict,
    ~55s vs ~130s on the misrouted Math path).
    """
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Explainer",
        description=(
            "Layered explanations for 'what is X / how does X work' "
            "questions. Grounds facts with a quick search, explains from "
            "intuition to mechanics, then checks accuracy against the "
            "sources."
        ),
        icon="",
        is_builtin=True,
        escalation_flow="Research",
        trigger_keywords=[
            "what is", "what are", "how does", "how do", "eli5",
            "meaning of", "define",
        ],
        steps=[
            _step(
                "Ground",
                (
                    "Run ONE web/wikipedia search to ground the explanation: "
                    "correct names, dates, current status, common "
                    "misconceptions. Note 2-4 key facts with their source. "
                    "If the topic is stable textbook material and search "
                    "adds nothing, output the key facts from knowledge and "
                    "say so."
                ),
                role="search",
                tool_categories=["search"],
                output_cap=900,
                sort_order=0,
            ),
            _step(
                "Explain",
                (
                    "Explain the topic in layers:\n"
                    "1. The essence in one plain sentence.\n"
                    "2. The intuition — a concrete analogy or example a "
                    "smart friend would get.\n"
                    "3. The mechanics — how it actually works, precise but "
                    "not jargon-first; define any term you must use.\n"
                    "4. Why it matters / where you'd meet it.\n"
                    "Match depth to the question — a casual ask gets 150 "
                    "words, a deep topic gets more. Use the grounded facts; "
                    "do not invent specifics.\n\n" + _CONVERSATIONAL_STYLE
                ),
                role="analyze",
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Accuracy Check",
                (
                    "Check the explanation against the grounded facts from "
                    "the Ground step. The search results are NEWER than your "
                    "training data — where they conflict with your priors, "
                    "the results win. Output EXACTLY three lines and then "
                    "STOP:\n"
                    "ERRORS_FOUND: <none | brief numbered list>\n"
                    "VERIFIED: <yes|no>\n"
                    "CONFIDENCE: <0.0-1.0>\n"
                    "Any text after the CONFIDENCE line is a protocol "
                    "violation."
                ),
                role="verify",
                output_cap=150,
                sort_order=2,
            ),
            _step(
                "Respond",
                (
                    "Deliver the final explanation, fixing anything the "
                    "Accuracy Check flagged. Do not mention the pipeline or "
                    "the check.\n\n" + _CONVERSATIONAL_STYLE
                ),
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=3,
                user_template="<work>\n{all_outputs}\n</work>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Live Lookup
# ---------------------------------------------------------------------------

def live_lookup_flow() -> ReasoningFlow:
    """Fast grounded answers for time-sensitive lookups.

    The highest-volume real analytical use case (weather, prices,
    "as of today" facts — 41 of 107 observed turns wanted live data)
    previously ran the 6-step Research pipeline or an unverified
    moderate path. Two steps: search (with fetch fallback when
    snippets lack the actual figures), then answer STRICTLY from
    findings with as-of dates and sources.

    Live-validated 2026-07-02 (real weather query — current temp,
    high/low, a source discrepancy surfaced honestly, ~21s).
    """
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Live Lookup",
        description=(
            "Fast grounded answers for time-sensitive lookups — weather, "
            "prices, schedules, 'as of today' facts. Searches, then answers "
            "strictly from the results with dates and sources."
        ),
        icon="",
        is_builtin=True,
        escalation_flow="Research",
        trigger_keywords=[
            "weather", "today", "right now", "price of", "how much is",
            "as of", "forecast",
        ],
        steps=[
            _step(
                "Search",
                (
                    "Run 1-3 targeted searches for the live data the user "
                    "wants. If the search snippets don't contain the actual "
                    "figures (common for weather/prices), FETCH the most "
                    "promising result page and read the data out of it. For "
                    "each useful finding note: the specific figure/fact, its "
                    "as-of date, source name, URL. Prefer primary or recent "
                    "sources. No analysis — just the findings."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                output_cap=0,
                sort_order=0,
            ),
            _step(
                "Respond",
                (
                    "Answer strictly from the findings. They are NEWER than "
                    "your training data — trust them over your priors.\n"
                    "- Lead with the answer itself, first sentence.\n"
                    "- Include the as-of date and source (URL) for the "
                    "load-bearing facts.\n"
                    "- If sources disagree, say so plainly with both "
                    "values.\n"
                    "- If the findings don't contain the answer, say what's "
                    "missing — never fill the gap from memory.\n\n"
                    + _CONVERSATIONAL_STYLE
                ),
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=1,
                user_template="<findings>\n{all_outputs}\n</findings>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Summarize
# ---------------------------------------------------------------------------

def summarize_flow() -> ReasoningFlow:
    """Faithful summaries/extractions from pasted text.

    Observed demand: "tldr this", "pull the physical descriptions",
    "tell me what you think about this <paste>". Two steps, no tools:
    a faithfulness-focused distillation (numbers/names preserved
    exactly, no invention), then delivery in the requested shape.

    Live-validated 2026-07-02 (earnings-paragraph TL;DR — every figure
    exact, tight bullets + takeaway, ~20s).
    """
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Summarize",
        description=(
            "Faithful summaries and extractions from pasted text — TL;DRs, "
            "key points, pulling specific details out of a wall of text. "
            "Never invents; preserves numbers and names exactly."
        ),
        icon="",
        is_builtin=True,
        # Phrase supersets ("tldr this" alongside "tldr") are deliberate:
        # both match the same ask, doubling its score, so an explicit
        # summarize-verb beats a flow whose keyword merely appears inside
        # the PASTED text (observed: "tldr this: ...quarterly report..."
        # tied with the Report flow on the content word "report").
        trigger_keywords=[
            "summarize", "summarize this", "summarise", "tldr", "tldr this",
            "tl;dr", "key points", "extract", "condense", "boil down",
            "sum this up",
        ],
        steps=[
            _step(
                "Distill",
                (
                    "Read the provided text and the user's ask. Produce the "
                    "faithful distillation they asked for:\n"
                    "- Summary ask → the core claims/events in order of "
                    "importance, plus what the author wants the reader to "
                    "do or believe.\n"
                    "- Extraction ask ('pull the X') → exactly the "
                    "requested items, quoted or tightly paraphrased.\n"
                    "Rules: never add facts not in the text; preserve "
                    "numbers, names, and dates EXACTLY; if the text doesn't "
                    "contain what was asked for, say so instead of "
                    "inventing; note real ambiguities in one line."
                ),
                role="analyze",
                output_cap=0,
                sort_order=0,
            ),
            _step(
                "Respond",
                (
                    "Deliver the distillation in the shape the user asked "
                    "for. Default when unspecified: 3-6 tight bullet "
                    "points, then one takeaway line. Length proportional to "
                    "the source — a paragraph in, two sentences out.\n\n"
                    + _CONVERSATIONAL_STYLE
                ),
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=1,
                user_template="<distilled>\n{previous_output}\n</distilled>\n\n{query}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def auto_routing_flow() -> ReasoningFlow:
    """Meta-flow: automatically selects the best flow per query via keyword/domain matching."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Auto Routing",
        description="Automatically selects the best flow for each query using keyword and domain matching. Falls back to Standard.",
        icon="",
        is_default=True,
        is_builtin=True,
        auto_search=False,
        steps=[
            _step(
                "Route",
                "This is a routing placeholder. The system selects the appropriate flow automatically.",
                role="respond",
                stream_to_user=True,
                output_cap=0,
                sort_order=0,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Agentic — Report
# ---------------------------------------------------------------------------

_AGENTIC_PLAN_SYSTEM = """\
Create a plan for this task. Output ONLY a markdown checklist:

## Task: <short title>

- [ ] 1. <step description>
- [ ] 2. <step description>

Rules:
- 4-8 steps
- Each step is one concrete action
- Steps should be in logical order
- Do not include explanations, just the checklist"""


# ---------------------------------------------------------------------------
# Quality anchors — shared instruction blocks for artifact content generation
# ---------------------------------------------------------------------------

_ANTI_FILLER = (
    "NEVER use these filler patterns: "
    '"In this section we will explore," "It is worth noting that," '
    '"In today\'s fast-paced world," "Let\'s delve into," '
    '"It is important to note," "Furthermore," "Moreover," "Indeed," '
    '"In conclusion," "As we can see," "This is significant because." '
    "Write direct, concrete prose. Every sentence must add information "
    "the previous one did not contain."
)

_REPORT_QUALITY = """\
Write in clear, flowing prose using complete paragraphs. Do NOT use bullet \
lists unless presenting truly discrete items. Incorporate information \
naturally into sentences.

Match the standard to the document the Plan step identified.

EVIDENCE report (research, analysis, whitepaper, brief grounded in sources):
- Open each section with the KEY FINDING, not background or definitions
  GOOD: "Average temperatures in the study region increased 1.8 degrees \
between 1970 and 2024, according to NCEI monitoring data."
  BAD: "Climate change is a complex phenomenon that affects many regions \
around the world."
- Every factual claim requires a specific number, date, or name with \
a [Source Name] citation
- Never use vague quantifiers: instead of "many" write "7 of 10," instead \
of "significant" write "23% increase," instead of "recently" write \
"since March 2024"
- Each section: Finding -> Evidence (cited data) -> Analysis -> Implication

COMPOSITION (essay, personal statement, op-ed, reflective or persuasive \
writing with no external dataset):
- Open each section with a clear point or claim, then develop it
- Support points with reasoning and concrete, specific examples — do NOT \
invent statistics, studies, or [Source] citations to sound authoritative
- Keep one consistent voice and a through-line argument from intro to \
conclusion; every section should advance the thesis
- Cite a real source only when you genuinely drew on one

In BOTH cases write substantive paragraphs (3-5 per section), not one-liners.
- """ + _ANTI_FILLER

_NARRATIVE_QUALITY = """\
Write vivid, immersive prose. Show, never tell.

Quality standards:
- Each scene must be at least 150 words — not 2-3 sentences
- Every scene includes ALL of these elements:
  * Setting: where are we? What does the character see, hear, smell?
  * Action: what is the character physically DOING?
  * Dialogue: natural conversation that reveals character or advances plot
  * Emotion through behavior — NEVER label emotions directly
    GOOD: "Her fingers tightened around the strap of her bag."
    BAD: "She felt nervous."
- Vary sentence rhythm: mix short punchy sentences with longer \
descriptive ones. Break patterns from your previous paragraphs.
- End each scene with a hook: a question, discovery, or choice \
that pulls the reader into the next scene
- Use concrete sensory details, not abstract descriptions
  GOOD: "The wooden floorboards groaned under her boots."
  BAD: "The old house was creepy."
- Dialogue tags: use "said" or nothing. Never "exclaimed," \
"murmured," "opined." Let the dialogue itself carry the tone.
- """ + _ANTI_FILLER

_SLIDE_QUALITY = """\
Design slides for maximum impact with minimum text.

Quality standards:
- Each slide carries ONE idea. Match the title style to the deck the Plan \
step identified:
  * PERSUASIVE / business deck — the title is an ASSERTION, the takeaway
    GOOD: "Customer retention improved 23% after onboarding redesign"
    BAD: "Customer Retention Analysis"
  * EDUCATIONAL / informational deck — the title names the concept or the \
question the slide answers; do NOT invent data to force an assertion
    GOOD: "How photosynthesis converts light into sugar"
    GOOD: "What problem does a load balancer solve?"
    BAD: "Photosynthesis boosts plant output 23%" (fabricated figure)
- Maximum 4 bullet points per slide, each 6 words or fewer
  Bullets are visual anchors, not the content
- Speaker notes carry the real substance: 3-5 sentences per slide
  * Do NOT repeat what is on the slide
  * Include specific data, context, or examples the presenter would say aloud
  * Open with a transition from the previous slide
  * Close with a bridge to the next
- Build a narrative across slides: persuasive → Situation, Evidence, Insight, \
Action; educational → Hook, Concept, Example, Recap
- Use "two_column" layout for comparisons, "blank" for full-page visuals"""

_DATA_QUALITY = """\
Match the rigor to the data the Plan step found.

QUANTITATIVE (real numbers available) — the difference between summary and \
analysis:
- SUMMARY (bad): "Sales increased in Q3"
- ANALYSIS (good): "Q3 sales rose 23% to $4.2M, outpacing the industry \
average of 12%, likely driven by the June pricing restructure — but this \
masks a 15% decline in unit volume, suggesting price sensitivity that may \
limit Q4 growth"
- For every finding: state the metric and its exact value; compare to a \
benchmark/baseline/prior period; calculate the change (%, absolute, ratio); \
state what it means. Never write "significant increase" without the number.
- Structure data so it is ready for charting: labeled categories with \
numeric values.

QUALITATIVE (a conceptual comparison with no real metrics — e.g. two \
philosophies, approaches, or trade-offs):
- Compare on clear, named criteria; for each, give a reasoned judgment with a \
concrete example — not a made-up score
- Do NOT fabricate numeric ratings, percentages, or a 1-10 scale to imitate \
data. An honest qualitative verdict beats fake precision.
- Land a clear bottom line: which option fits which need, and why.
- """ + _ANTI_FILLER

_TUTORIAL_QUALITY = """\
Write instructions that a reader can follow without guessing.

Match the form to the subject. A programming / CLI / config topic needs code; \
a physical, manual, or conceptual topic (changing a tire, kneading bread, \
de-escalating a conflict) needs plain-language actions. NEVER wrap a \
real-world action in code — no print("engage the brake"), no fake "expected \
console output" for something that happens off-screen.

Quality standards:
- CODE topic — every code block must be COMPLETE and RUNNABLE (never "..." or \
"// add your code here"). For each step: state what the reader accomplishes \
(one sentence), show the code, show the expected output, explain WHY it \
works. Specify exact versions of languages, frameworks, and dependencies.
- NON-CODE topic — for each step: state what the reader accomplishes (one \
sentence), give the concrete action in plain language, describe the result \
they should observe (what they see / feel / hear — NOT console output), \
explain WHY it works. List the physical tools or materials required, not \
software versions.
- Progress from simple to complex: start with the minimal first action, then \
build up one step at a time
- After each major step, include a "Common mistake" note: what goes wrong, \
how the reader will recognize it, and the fix
- """ + _ANTI_FILLER

_FACTCHECK_QUALITY = """\
Evaluate claims with rigor. For each claim:
- State the claim exactly as presented
- Classify: factual / statistical / mathematical / opinion (skip opinions)
- Present evidence FOR with source and date
- Present evidence AGAINST with source and date
- Note source credibility: official government/academic > major news > blog
- Assign a verdict: TRUE / MOSTLY TRUE / MIXED / MOSTLY FALSE / FALSE / \
UNVERIFIABLE
- State your confidence and what additional evidence would change the verdict
- Never assign TRUE or FALSE without at least two independent sources"""

_DOC_CREATION_GUIDANCE = """\
Structure the document for a professional reader:
- Each heading should be an ASSERTION (a finding), not a topic label
  GOOD: "Regional temperatures rose 1.8 degrees over five decades"
  BAD: "Temperature Trends"
- Each section body: 3+ paragraphs of flowing prose with [Source] citations
- Final section: "Sources" listing all cited URLs
- Do NOT create single-sentence sections — if a section has less than \
a paragraph of content, merge it with an adjacent section"""

_PPTX_CREATION_GUIDANCE = """\
Structure slides for a professional presentation:
- Title slide: clear title, subtitle with scope or date, author if known
- Content slides: assertion headline, 3-4 short bullets, detailed speaker notes
- Use "two_column" layout for side-by-side comparisons
- End with a summary or "Next Steps" slide
- 8-15 slides total for a standard presentation"""


def agentic_report_flow() -> ReasoningFlow:
    """Agentic flow for generating research reports and documents."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Report",
        description=(
            "Research a topic, draft sections, review for accuracy, and "
            "deliver a polished document (PDF/DOCX). Best for reports, "
            "essays, and written deliverables."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "report", "document", "essay", "paper", "article", "brief",
            "summary", "memo", "whitepaper", "analysis",
        ],
        auto_search=True,
        autonomy_level=3,
        max_tool_calls_per_step=5,
        steps=[
            _step(
                "Plan",
                (
                    _AGENTIC_PLAN_SYSTEM + "\n\n"
                    "Consider the user's context and preferences when planning.\n\n"
                    "On its own line, decide:\n"
                    "- FORMAT: 'evidence' for a research/analytical document that "
                    "should be grounded in external sources and data; "
                    "'composition' for an essay, personal statement, op-ed, or "
                    "reflective/persuasive piece argued from reasoning and the "
                    "user's own material rather than cited data"
                ),
                role="plan",
                tool_names=[],
                stream_to_user=False,
                output_cap=600,
                sort_order=0,
            ),
            _step(
                "Research",
                (
                    "If the Plan chose 'composition' and the topic is personal "
                    "or subjective with no external facts to gather, output "
                    "'No external research required.' and stop.\n\n"
                    "Otherwise, search for 3-5 high-quality sources on this "
                    "topic.\n\n"
                    "For each source, extract SPECIFIC data:\n"
                    "- Source: <name> (<URL>)\n"
                    "- Key numbers: exact figures, dates, percentages\n"
                    "- Key quotes: notable statements from experts or officials\n\n"
                    "Prioritize official data (government, academic, major organizations) "
                    "over opinion pieces. Fetch the actual page content when a source "
                    "looks promising — snippets are not enough.\n\n"
                    "Do NOT write analysis. Just extract data with sources."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                stream_to_user=False,
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Outline",
                (
                    "Create a document outline. 4-6 sections total.\n\n"
                    "For an EVIDENCE report, each heading is an ASSERTION (a key "
                    "finding), not a topic label, and you list the specific data "
                    "points and sources that support it.\n"
                    'GOOD heading: "Regional temperatures rose 1.8 degrees over five decades"\n'
                    'BAD heading: "Temperature Trends"\n\n'
                    "For a COMPOSITION, each heading marks a stage of the "
                    "argument or narrative (it need not be a data finding); list "
                    "the point each section makes and the example that carries "
                    "it.\n\n"
                    "Output ONLY the outline. Do NOT write prose."
                ),
                role="analyze",
                stream_to_user=False,
                output_cap=800,
                sort_order=2,
            ),
            _step(
                "Draft",
                (
                    _REPORT_QUALITY + "\n\n"
                    "Write the full document following the outline. For an "
                    "EVIDENCE report, use ONLY facts from the Research step — do "
                    "not invent data. For a COMPOSITION, develop the argument "
                    "from reasoning and the user's request — do not fabricate "
                    "statistics or citations. Output the complete document in "
                    "markdown. Start directly with the content.\n\n"
                    "Target 800-1500 words across all sections.\n\n"
                    "IMPORTANT: Format each section with a marker line:\n"
                    "## SECTION: <assertion heading>\n"
                    "<section body text>\n\n"
                    "This exact format is required for document generation."
                ),
                role="draft",
                tool_names=[],
                stream_to_user=False,
                output_cap=4000,
                sort_order=3,
            ),
            _step(
                "Review",
                (
                    "Review the draft. For an EVIDENCE report check:\n"
                    "1. Any claim without a [Source] citation?\n"
                    "2. Any vague language ('many,' 'significant,' 'recently')?\n"
                    "3. Any section with fewer than 2 paragraphs?\n"
                    "4. Does each section open with a finding (not background)?\n"
                    "For a COMPOSITION check instead:\n"
                    "1. Any fabricated statistic or fake citation? (flag it)\n"
                    "2. Is the voice consistent and the thesis advanced each section?\n"
                    "3. Any section with fewer than 2 paragraphs?\n"
                    "In BOTH: are transitions between sections logical?\n\n"
                    "Output ONLY:\nISSUES:\n1. <issue and fix>\n\n"
                    "VERDICT: PASS or NEEDS_REVISION\n\n"
                    "Do NOT rewrite the document."
                ),
                role="review",
                tool_names=[],
                stream_to_user=False,
                output_cap=600,
                sort_order=4,
            ),
            _step(
                "Create Document",
                (
                    _DOC_CREATION_GUIDANCE + "\n\n"
                    "Call create_document with the draft content.\n\n"
                    'create_document(title="...", format="pdf", sections=['
                    '{"heading": "...", "body": "..."}])\n\n'
                    "Use the FULL draft text for each section body. "
                    "Do NOT summarize or shorten."
                ),
                role="create",
                tool_names=["create_document"],
                tool_categories=["artifact"],
                stream_to_user=False,
                output_cap=0,
                sort_order=5,
            ),
            _step(
                "Deliver",
                _DELIVER_SYSTEM_BASE + (
                    " Include: key findings, supporting evidence, and the "
                    "document download link if one was produced."
                ),
                role="deliver",
                tool_names=[],
                stream_to_user=True,
                output_cap=600,
                sort_order=6,
                user_template=_DELIVER_USER_TEMPLATE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Agentic — Presentation
# ---------------------------------------------------------------------------

def agentic_presentation_flow() -> ReasoningFlow:
    """Agentic flow for generating slide presentations."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Presentation",
        description=(
            "Research, structure slides, draft content, optionally generate "
            "illustrations, and deliver a polished PPTX. Best for slide decks, "
            "pitch decks, and visual presentations."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "presentation", "slides", "slide deck", "pptx", "powerpoint",
            "pitch deck", "keynote", "talk",
        ],
        auto_search=True,
        autonomy_level=3,
        max_tool_calls_per_step=5,
        steps=[
            _step(
                "Plan",
                (
                    _AGENTIC_PLAN_SYSTEM + "\n\n"
                    "Consider the user's preferences on presentation style.\n\n"
                    "On its own line, decide:\n"
                    "- FORMAT: 'persuasive' for a pitch, business case, or "
                    "results deck that argues a position from data; "
                    "'educational' for a lecture, training, or explainer that "
                    "teaches a concept and may have no metrics to assert"
                ),
                role="plan",
                tool_names=[],
                stream_to_user=False,
                output_cap=600,
                sort_order=0,
            ),
            _step(
                "Research",
                (
                    "Search for data and talking points. Output ONLY a source list.\n\n"
                    "For each source:\n"
                    "- Source: <name> (<URL>)\n"
                    "- Key data: <bullet points>\n\n"
                    "Do NOT write a response. Just list sources with data."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                stream_to_user=False,
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Structure Slides",
                (
                    "Plan the slide deck structure. For each slide:\n\n"
                    "### Slide N: <title>\n"
                    "- Layout: title / content / two_column\n"
                    "- Key point from research\n"
                    "- Needs illustration: yes/no\n\n"
                    "Title style follows the FORMAT: a persuasive deck uses an "
                    "ASSERTION ('Three factors drove 80% of cost overruns'); an "
                    "educational deck names the concept or question ('How a load "
                    "balancer distributes traffic'). Do NOT invent metrics to "
                    "manufacture an assertion.\n\n"
                    "Target 8-15 slides. Start with title, end with action items "
                    "or a recap."
                ),
                role="analyze",
                stream_to_user=False,
                output_cap=1200,
                sort_order=2,
            ),
            _step(
                "Draft Content",
                (
                    _SLIDE_QUALITY + "\n\n"
                    "Output slide content following the structure above.\n\n"
                    "### Slide N: <assertion title>\n"
                    "- Bullet 1 (max 6 words)\n"
                    "- Bullet 2\n"
                    "**Notes:** 3-5 sentences of presenter context — specific data, "
                    "examples, talking points NOT on the slide.\n\n"
                    "Use ONLY facts from the Research step."
                ),
                role="draft",
                tool_names=[],
                stream_to_user=False,
                output_cap=3000,
                sort_order=3,
            ),
            _step(
                "Illustrate Slides",
                (
                    "For each slide that needs an image, run the two-pass "
                    "query crafter:\n"
                    "1. Write an SEO description (subject / source family / "
                    "format / aesthetic) — internal scratch, never sent to "
                    "search.\n"
                    "2. Reduce to a 5-10 word query and image_search returns "
                    "4 candidates per slide. Top-ranked candidate is the "
                    "primary; the user can widen the pool post-render via "
                    "the picker.\n\n"
                    "The runtime handles the per-slide loop, prompt wrapping, "
                    "and pool storage — this step's role is to declare the "
                    "intent. Slides ranked text-only get no image."
                ),
                role="illustrate",
                tool_names=["image_search"],
                tool_categories=["image"],
                stream_to_user=False,
                output_cap=0,
                sort_order=4,
            ),
            _step(
                "Create Presentation",
                (
                    _PPTX_CREATION_GUIDANCE + "\n\n"
                    "Call create_presentation with ALL slides from the draft.\n\n"
                    'create_presentation(title="...", slides=[{"layout": "content", '
                    '"title": "Assertion headline here", '
                    '"body": "- Bullet 1\\n- Bullet 2", '
                    '"notes": "Detailed talking points...", '
                    '"image_url": "/api/image/abc123"}])\n\n'
                    "Include the FULL speaker notes. Do NOT shorten or omit them."
                ),
                role="create",
                tool_names=["create_presentation"],
                tool_categories=["artifact"],
                stream_to_user=False,
                output_cap=0,
                sort_order=5,
            ),
            _step(
                "Review",
                (
                    "Output ONLY an issues list:\n\n"
                    "ISSUES:\n"
                    "1. <issue and fix>\n\n"
                    "VERDICT: PASS or NEEDS_REVISION\n\n"
                    "Use text_analysis to check readability per slide. "
                    "Do NOT rewrite the presentation."
                ),
                role="review",
                tool_names=["text_analysis"],
                stream_to_user=False,
                output_cap=500,
                sort_order=6,
            ),
            _step(
                "Deliver",
                _DELIVER_SYSTEM_BASE + (
                    " Include: slide count, key talking points, the download "
                    "link, and one or two delivery tips."
                ),
                role="deliver",
                tool_names=[],
                stream_to_user=True,
                output_cap=600,
                sort_order=7,
                user_template=_DELIVER_USER_TEMPLATE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Agentic — Storybook
# ---------------------------------------------------------------------------

def agentic_storybook_flow() -> ReasoningFlow:
    """Agentic flow for generating illustrated storybooks.

    Three steps: Plan → Write → Create Book.
    Illustration is handled automatically by create_ebook's _auto_illustrate()
    which generates a cover + per-chapter images with IP-Adapter consistency.
    """
    fid = _id()
    steps = [
            _step(
                "Plan Story",
                (
                    "Output ONLY a structured story plan. No prose.\n\n"
                    "0. MODE: 'fiction' for an invented story; 'retelling' for a "
                    "true account (history, biography, real events). For a "
                    "retelling, stay faithful to what actually happened — "
                    "describe real people and places accurately and do NOT "
                    "invent their appearance or fabricate events.\n"
                    "1. TITLE: story title\n"
                    "2. AUDIENCE: age group and reading level\n"
                    "3. CHARACTERS: name, appearance, personality (physical details — "
                    "hair color, clothing, distinguishing features). For a "
                    "retelling, describe real figures only from known fact.\n"
                    "4. SETTING: time, place, atmosphere\n"
                    "5. PLOT: beginning → rising action → climax → resolution\n"
                    "6. STORY ARC: map the content onto discrete chapters so "
                    "each chapter moves forward. For FICTION use a dramatic arc; "
                    "for a RETELLING organize chapters chronologically or "
                    "thematically around the real events instead of a fictional "
                    "climax. Dramatic-arc structure:\n"
                    "   - Chapter 1 (SETUP): introduce protagonist, setting, "
                    "and normal world\n"
                    "   - Chapter 2 (INCITING INCIDENT): the event that "
                    "disrupts normal life\n"
                    "   - Middle chapters (RISING ACTION): escalating "
                    "obstacles; raise the stakes every chapter — do NOT "
                    "repeat the same beat with different scenery\n"
                    "   - Penultimate chapter (CLIMAX): the confrontation "
                    "or turning point\n"
                    "   - Final chapter (RESOLUTION): how the protagonist "
                    "is changed; tie off loose threads\n"
                    "7. CHAPTERS: list 4-8 chapters. For EACH chapter include "
                    "a one-line summary AND its arc stage (setup / inciting / "
                    "rising / climax / resolution). Every chapter must move "
                    "the story forward — no filler.\n\n"
                    "Consider the user's story preferences."
                ),
                role="plan",
                tool_names=[],
                stream_to_user=False,
                output_cap=1000,
                sort_order=0,
            ),
            _step(
                "Write Story",
                (
                    _NARRATIVE_QUALITY + "\n\n"
                    "Write the full story chapter by chapter following the plan. "
                    "Start directly with the story — no preamble.\n\n"
                    "Target 1500-2500 words total. Each chapter: 150-300 words minimum.\n\n"
                    "IMPORTANT: Format each chapter with a marker line:\n"
                    "## SECTION: Chapter N: <chapter title>\n"
                    "<chapter body text — pure prose only>\n\n"
                    "This exact format is required for book generation.\n\n"
                    "Rules:\n"
                    "- Write ONLY prose text in chapter bodies\n"
                    "- For a RETELLING, stay faithful to known facts: keep "
                    "sensory detail grounded, and present only dialogue that is "
                    "actually recorded — do NOT invent quotes or events\n"
                    "- Do NOT include illustration markers, image references, or "
                    "parenthetical stage directions like '(Illustration #1 — ...)'\n"
                    "- Illustrations are generated automatically — just write the story"
                ),
                role="draft",
                stream_to_user=False,
                output_cap=6000,
                sort_order=1,
            ),
            _step(
                "Create Book",
                (
                    "Call create_ebook to assemble the illustrated EPUB. "
                    "Illustrations are generated automatically — do NOT call "
                    "image_generation yourself.\n\n"
                    "Pass the chapters from the Write Story step:\n\n"
                    'create_ebook(title="<title from plan>", author="Augmentum", '
                    'chapters=[{"heading": "Chapter 1: The Discovery", '
                    '"body": "Once upon a time..."}, ...])\n\n'
                    "Rules:\n"
                    "- Call create_ebook EXACTLY ONCE\n"
                    "- Include ALL chapters with full body text\n"
                    "- Do NOT include image_url fields — images are auto-generated\n"
                    "- Do NOT add commentary — just call the tool\n"
                    "- After the tool returns, present the download link to the user"
                ),
                role="create",
                tool_names=["create_ebook"],
                stream_to_user=True,
                output_cap=0,
                sort_order=2,
            ),
    ]

    # Storybook steps don't need memory_recall — strip auto-injected instances.
    # The flow operates on its own working memory, not cross-session recall.
    for s in steps:
        if "memory_recall" in s.tool_names:
            s.tool_names.remove("memory_recall")

    return ReasoningFlow(
        id=fid,
        name="Storybook",
        description=(
            "Plan a story, write chapters with prose, and assemble an "
            "illustrated EPUB storybook with auto-generated cover and "
            "chapter illustrations. "
            "Best for children's stories, illustrated tales, and creative projects."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "storybook", "story", "children's book", "illustrated",
            "fairy tale", "picture book", "tale", "fable",
        ],
        auto_search=False,
        autonomy_level=3,
        max_tool_calls_per_step=1,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Template: Agentic — Data & Comparison
# ---------------------------------------------------------------------------

def agentic_data_comparison_flow() -> ReasoningFlow:
    """Agentic flow for data analysis, comparison, and visualization."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Data & Comparison",
        description=(
            "Research data or competitors, analyze trends and differences, "
            "generate charts and spreadsheets. Handles both data analysis "
            "and competitive comparison."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "analyze data", "data analysis", "chart", "graph", "compare",
            "statistics", "trend", "spreadsheet", "dataset", "metrics",
            "visualization", "visualize", "competitor", "competitive",
            "versus", "vs", "alternative", "benchmark", "comparison",
            "which is better", "pros and cons",
        ],
        auto_search=True,
        autonomy_level=3,
        max_tool_calls_per_step=8,
        steps=[
            _step(
                "Plan",
                (
                    _AGENTIC_PLAN_SYSTEM + "\n\n"
                    "Determine: is this a data analysis task or a comparison?\n"
                    "- Data analysis: identify metrics, time ranges, data sources\n"
                    "- Comparison: identify entities, dimensions to compare\n\n"
                    "Also decide, on its own line:\n"
                    "- DATA: 'quantitative' if real numbers/metrics exist to "
                    "analyze and chart; 'qualitative' if this is a conceptual "
                    "comparison with no real metrics (a chart would be "
                    "fabricated)\n\n"
                    "Consider prior context and user preferences."
                ),
                role="plan",
                tool_names=[],
                stream_to_user=False,
                output_cap=600,
                sort_order=0,
            ),
            _step(
                "Research",
                (
                    "Search for data. Output ONLY structured data notes.\n\n"
                    "For data analysis:\n"
                    "- Source: <name> (<URL>)\n"
                    "- Numbers: <extracted statistics>\n\n"
                    "For comparison:\n"
                    "### <Entity Name>\n"
                    "- Features: ...\n"
                    "- Pricing: ...\n"
                    "- Strengths / Weaknesses: ...\n\n"
                    "Extract specific numbers — charts need real data. "
                    "Fetch official sources. Do NOT write analysis."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                stream_to_user=False,
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Analyze",
                (
                    _DATA_QUALITY + "\n\n"
                    "Output structured analysis — no introductions.\n\n"
                    "For data analysis:\n"
                    "1. Key trends with exact numbers and time ranges\n"
                    "2. Derived metrics: calculate growth rates, averages, ratios\n"
                    "3. Outliers: what is unexpected and why?\n\n"
                    "For comparison:\n"
                    "1. Feature matrix with scores per dimension\n"
                    "2. Winner per category with justification\n"
                    "3. Unique differentiators per entity\n\n"
                    "Use calculator/python_exec for all computations.\n"
                    "Structure results so they are ready for charting: "
                    "labeled categories with numeric values."
                ),
                role="analyze",
                tool_names=["python_exec", "calculator", "math_verify", "unit_converter"],
                stream_to_user=False,
                output_cap=0,
                sort_order=2,
            ),
            _step(
                "Create Charts",
                (
                    "Create charts ONLY when you have REAL numeric data.\n\n"
                    "If the Plan marked this 'qualitative' (a conceptual "
                    "comparison with no real metrics), SKIP this step — output "
                    "'No quantitative data to chart.' Do NOT invent numbers to "
                    "fill a chart.\n\n"
                    "When you do have real numbers, call the create_chart tool "
                    "(do not just describe charts).\n"
                    "Pick the best chart type per insight:\n"
                    "- bar: comparing categories or entities\n"
                    "- line: trends over time\n"
                    "- pie: proportions of a whole\n"
                    "- scatter: correlations\n"
                    "- area: cumulative trends\n\n"
                    "Example:\n"
                    'create_chart(title="Feature Comparison", chart_type="bar", '
                    'x_label="Feature", y_label="Score", '
                    'labels=["Speed", "Price", "Support"], '
                    'datasets=[{"name": "Product A", "values": [8, 6, 9]}, '
                    '{"name": "Product B", "values": [7, 9, 7]}])\n\n'
                    "Create 1-3 charts. No commentary — just call the tool."
                ),
                role="create",
                tool_names=["create_chart"],
                tool_categories=["artifact"],
                stream_to_user=False,
                output_cap=0,
                sort_order=3,
            ),
            _step(
                "Create Spreadsheet",
                (
                    "Export the findings to a spreadsheet via "
                    "create_spreadsheet.\n\n"
                    "Quantitative — numeric sheets:\n"
                    "1. Raw Data — all collected data points\n"
                    "2. Analysis — scores, rankings, or derived metrics\n"
                    "3. Summary — key findings in tabular form\n"
                    "Qualitative — a comparison table is still useful: criteria "
                    "as rows, entities as columns, concise text judgments per "
                    "cell (no fabricated numbers).\n\n"
                    "Example:\n"
                    'create_spreadsheet(title="Analysis Results", sheets=['
                    '{"name": "Data", "headers": ["Metric", "A", "B"], '
                    '"rows": [["Speed", 8, 7]]}])\n\n'
                    "Include ALL data. No commentary — just call the tool."
                ),
                role="create",
                tool_names=["create_spreadsheet"],
                tool_categories=["artifact"],
                stream_to_user=False,
                output_cap=0,
                sort_order=4,
            ),
            _step(
                "Deliver",
                _DELIVER_SYSTEM_BASE + (
                    " Include: an executive summary, key insights with "
                    "specific numbers, embedded chart images, the spreadsheet "
                    "download link, and any caveats worth flagging."
                ),
                role="deliver",
                tool_names=[],
                stream_to_user=True,
                output_cap=600,
                sort_order=5,
                user_template=_DELIVER_USER_TEMPLATE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Agentic — Fact-Checker
# ---------------------------------------------------------------------------

def agentic_fact_checker_flow() -> ReasoningFlow:
    """Agentic flow for verifying claims, articles, and statements."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Fact-Checker",
        description=(
            "Decompose claims, research each one independently, cross-reference "
            "sources, verify math/data, and produce a verdict report with "
            "confidence ratings. Best for debunking, verification, and due diligence."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "fact check", "verify", "debunk", "is it true", "claim",
            "accurate", "misinformation", "source check", "true or false",
        ],
        auto_search=True,
        autonomy_level=3,
        max_tool_calls_per_step=8,
        steps=[
            _step(
                "Decompose Claims",
                (
                    "Output ONLY a numbered claim list. No prose.\n\n"
                    "For each claim:\n"
                    "C1: <claim in one sentence>\n"
                    "Type: factual / statistical / mathematical\n"
                    "Evidence needed: <what would confirm or refute>\n\n"
                    "Skip opinions — only include verifiable claims."
                ),
                role="analyze",
                tool_names=[],
                stream_to_user=False,
                output_cap=800,
                sort_order=0,
            ),
            _step(
                "Research Claims",
                (
                    "For EACH claim, search for evidence. Output ONLY evidence notes.\n\n"
                    "For each claim:\n"
                    "C1 FOR: <supporting evidence + source>\n"
                    "C1 AGAINST: <counter-evidence + source>\n\n"
                    "Fetch primary sources. Note credibility (official/academic/news/blog). "
                    "Do NOT write analysis or conclusions."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                stream_to_user=False,
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Verify Data",
                (
                    "For claims involving numbers, math, or statistics:\n\n"
                    "1. Use math_verify to check mathematical claims\n"
                    "2. Use calculator to verify arithmetic\n"
                    "3. Use python_exec for statistical analysis\n"
                    "4. Use unit_converter if unit claims are involved\n\n"
                    "For non-numerical claims, compare against the source evidence "
                    "gathered in the Research step.\n\n"
                    "Output ONLY verification results: C1: PASS/FAIL + reason."
                ),
                role="verify",
                tool_names=[
                    "math_verify", "calculator", "python_exec",
                    "unit_converter",
                ],
                stream_to_user=False,
                output_cap=0,
                sort_order=2,
            ),
            _step(
                "Rate & Judge",
                (
                    _FACTCHECK_QUALITY + "\n\n"
                    "Output a structured verdict for each claim:\n\n"
                    "C1: <claim restated>\n"
                    "VERDICT: TRUE / MOSTLY TRUE / MIXED / MOSTLY FALSE / FALSE / UNVERIFIABLE\n"
                    "EVIDENCE: <strongest supporting evidence + source>\n"
                    "COUNTER: <strongest counter-evidence if any>\n"
                    "CONFIDENCE: high / medium / low\n\n"
                    "Do NOT write introductions or conclusions."
                ),
                role="analyze",
                stream_to_user=False,
                output_cap=1500,
                sort_order=3,
            ),
            _step(
                "Create Report",
                (
                    "Call create_document to build the PDF. No commentary.\n\n"
                    'create_document(title="Fact-Check Report: <topic>", format="pdf", sections=['
                    '{"heading": "Executive Summary", '
                    '"body": "Overall: ..."}, {"heading": "Claim 1: ...", '
                    '"body": "Verdict: TRUE\\n\\nEvidence: ..."}])\n\n'
                    "Include executive summary, one section per claim, and sources."
                ),
                role="create",
                tool_names=["create_document"],
                tool_categories=["artifact"],
                stream_to_user=False,
                output_cap=0,
                sort_order=4,
            ),
            _step(
                "Deliver",
                _DELIVER_SYSTEM_BASE + (
                    " Include: the verdict table, key evidence per claim, the "
                    "report download link, and any caveats about confidence."
                ),
                role="deliver",
                tool_names=[],
                stream_to_user=True,
                output_cap=600,
                sort_order=5,
                user_template=_DELIVER_USER_TEMPLATE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Agentic — Tutorial Builder
# ---------------------------------------------------------------------------

def agentic_tutorial_flow() -> ReasoningFlow:
    """Agentic flow for creating illustrated how-to guides and tutorials."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Tutorial Builder",
        description=(
            "Research best practices, write step-by-step instructions, "
            "test code examples, generate diagrams, and assemble into a "
            "polished tutorial document. Best for how-to guides, technical "
            "documentation, and learning materials."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "tutorial", "how to", "guide", "walkthrough", "step by step",
            "instructions", "learn", "teach", "documentation", "howto",
        ],
        auto_search=True,
        autonomy_level=3,
        max_tool_calls_per_step=5,
        steps=[
            _step(
                "Plan",
                (
                    _AGENTIC_PLAN_SYSTEM + "\n\n"
                    "Consider the user's skill level "
                    "or preferences on this topic.\n\n"
                    "Also determine, each on its own line:\n"
                    "- FORMAT: 'code' when this is a programming, CLI, config, "
                    "or software topic where the reader runs commands or writes "
                    "code; 'procedure' for physical, manual, or conceptual "
                    "topics (changing a tire, cooking, exercises, soft skills) "
                    "where there is no code to run\n"
                    "- Target audience (beginner/intermediate/advanced)\n"
                    "- Prerequisites needed (software + versions for a code "
                    "topic; tools + materials for a procedure)\n"
                    "- Whether diagrams/illustrations would help"
                ),
                role="plan",
                tool_names=[],
                stream_to_user=False,
                output_cap=600,
                sort_order=0,
            ),
            _step(
                "Research",
                (
                    "Search for sources. Output ONLY a source list.\n\n"
                    "For each source:\n"
                    "- Source: <name> (<URL>)\n"
                    "- Key info: <bullet points>\n\n"
                    "Prioritize official docs. Do NOT write a tutorial."
                ),
                role="search",
                tool_categories=["search", "fetch"],
                stream_to_user=False,
                output_cap=0,
                sort_order=1,
            ),
            _step(
                "Draft Tutorial",
                (
                    _TUTORIAL_QUALITY + "\n\n"
                    "Follow the FORMAT the Plan step chose.\n\n"
                    "Structure:\n"
                    "1. Introduction — what the reader will accomplish and why "
                    "it matters (2-3 sentences)\n"
                    "2. Prerequisites — for a code topic: exact versions and "
                    "install commands. For a procedure: the physical tools, "
                    "materials, or conditions required\n"
                    "3. Steps — each step: goal sentence → the action (complete "
                    "runnable code for a code topic; a concrete plain-language "
                    "instruction for a procedure) → the result to expect → why "
                    "it works\n"
                    "4. Common mistakes — after each major step, what goes "
                    "wrong and the fix\n"
                    "5. Next steps — where to go from here\n\n"
                    "Use ONLY facts from the Research step. Do NOT invent "
                    "commands, APIs, or steps. For a procedure topic write the "
                    "actions in plain language — do NOT emit code or print() "
                    "statements for real-world actions.\n"
                    "Mark diagram spots with [DIAGRAM: description].\n"
                    "Start directly with the content.\n"
                    "Target 1000-2000 words.\n\n"
                    "IMPORTANT: Format each section with a marker line:\n"
                    "## SECTION: <section heading>\n"
                    "<section body text>\n\n"
                    "This exact format is required for guide generation."
                ),
                role="draft",
                stream_to_user=False,
                output_cap=5000,
                sort_order=2,
            ),
            _step(
                "Verify Examples",
                (
                    "Output ONLY results.\n\n"
                    "If the draft contains real code: run each code block with "
                    "python_exec and report PASS/FAIL.\n"
                    "If it contains math claims: verify with math_verify.\n"
                    "If the draft has NO code (a physical or conceptual "
                    "procedure): do NOT invent code to run — output "
                    "'No code to verify.' and instead spot-check that each "
                    "step's stated result plausibly follows from its action, "
                    "flagging any step that does not.\n"
                    "Do NOT rewrite the tutorial."
                ),
                role="verify",
                tool_names=[
                    "python_exec", "math_verify", "calculator", "unit_converter",
                ],
                stream_to_user=False,
                output_cap=0,
                sort_order=3,
            ),
            _step(
                "Illustrate",
                (
                    "Illustrate the guide. The runtime runs this step "
                    "deterministically: it parses your sections and, for each "
                    "one, gathers image candidates from BOTH real-photo search "
                    "and a capability-matched image generator, then lets the "
                    "user pick which lands in the document.\n\n"
                    "FORMAT awareness (from the Plan step):\n"
                    "- 'procedure' (physical/manual/conceptual how-to — changing "
                    "a tire, cooking, exercises): REAL PHOTOGRAPHS lead. A "
                    "synthetic image is offered only as an alternate and only "
                    "when a photoreal model is installed — a stylised/anime "
                    "model is NEVER used to depict a real-world action.\n"
                    "- 'code'/technical: clean labeled DIAGRAMS lead, with "
                    "photos as alternates.\n\n"
                    "This step only declares intent — emit no commentary and do "
                    "not hand-write image prompts."
                ),
                role="illustrate",
                tool_names=["image_search", "image_generation"],
                tool_categories=["image"],
                stream_to_user=False,
                output_cap=0,
                sort_order=4,
            ),
            _step(
                "Create Guide",
                (
                    "Call create_document to build the PDF. No commentary.\n\n"
                    'create_document(title="Tutorial: <topic>", format="pdf", sections=['
                    '{"heading": "Introduction", '
                    '"body": "..."}, {"heading": "Step 1: ...", '
                    '"body": "...", "image_url": "/api/image/abc123"}])\n\n'
                    "Include full tutorial text, verified code, and diagram images."
                ),
                role="create",
                tool_names=["create_document"],
                tool_categories=["artifact"],
                stream_to_user=False,
                output_cap=0,
                sort_order=5,
            ),
            _step(
                "Deliver",
                _DELIVER_SYSTEM_BASE + (
                    " Include: a topic summary, the section list, the "
                    "download link, and a concrete quick-start first step."
                ),
                role="deliver",
                tool_names=[],
                stream_to_user=True,
                output_cap=600,
                sort_order=6,
                user_template=_DELIVER_USER_TEMPLATE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Template: Agentic - Application
# ---------------------------------------------------------------------------

def agentic_application_flow() -> ReasoningFlow:
    """Agentic flow for building editable web applications."""
    fid = _id()
    return ReasoningFlow(
        id=fid,
        name="Application",
        description=(
            "Plan, build, verify, and deliver an editable web application. "
            "Best for apps, websites, dashboards, games, calculators, forms, "
            "and interactive tools."
        ),
        icon="",
        is_builtin=True,
        auto_select=True,
        trigger_domains=["agentic"],
        trigger_keywords=[
            "app", "application", "website", "web app", "site", "game",
            "dashboard", "landing page", "calculator", "tool", "form",
            "portfolio", "build me", "make me", "create an app",
        ],
        auto_search=False,
        autonomy_level=3,
        max_tool_calls_per_step=2,
        steps=[
            _step(
                "Plan App",
                (
                    "Turn the user request into a concise app plan.\n\n"
                    "Include:\n"
                    "- Core user goal\n"
                    "- Main screens or states\n"
                    "- Required interactions\n"
                    "- Visual direction suited to the app domain\n"
                    "- Any important edge cases\n\n"
                    "Keep it concrete. Do not generate code."
                ),
                role="plan",
                tool_names=[],
                output_cap=700,
                sort_order=0,
            ),
            _step(
                "Build Application",
                (
                    "Call build_application with the user's request and the "
                    "app plan. Build a complete, polished, editable web app. "
                    "Use scaffold='game' for canvas/browser games, "
                    "scaffold='dashboard' for chart-heavy dashboards, "
                    "scaffold='form' for form/tool workflows, otherwise "
                    "scaffold='static'."
                ),
                role="create",
                tool_names=["build_application"],
                tool_categories=["artifact"],
                output_cap=0,
                sort_order=1,
                user_template="<app_plan>\n{all_outputs}\n</app_plan>\n\n{query}",
            ),
            _step(
                "Review Result",
                (
                    "Review the build result from the prior step. Summarize "
                    "what was produced, any limitations or warnings, and the "
                    "best next action for the user. Do not call more tools."
                ),
                role="review",
                tool_names=[],
                output_cap=500,
                sort_order=2,
            ),
            _step(
                "Deliver",
                _DELIVER_SYSTEM_BASE + (
                    " Include the app name, what is ready to try, and point "
                    "the user to the available app actions such as Open, Edit, "
                    "Modify, Download, or Open in Code when present."
                ),
                role="deliver",
                tool_names=[],
                stream_to_user=True,
                output_cap=600,
                sort_order=3,
                user_template=_DELIVER_USER_TEMPLATE,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES: dict[str, callable] = {
    "auto_routing": auto_routing_flow,
    "quick_answer": quick_answer_flow,
    "research": research_flow,
    "code_review": code_review_flow,
    "debate": debate_flow,
    "math": math_flow,
    "creative": creative_flow,
    "explainer": explainer_flow,
    "live_lookup": live_lookup_flow,
    "summarize": summarize_flow,
    "agentic_report": agentic_report_flow,
    "agentic_presentation": agentic_presentation_flow,
    "agentic_storybook": agentic_storybook_flow,
    "agentic_data_comparison": agentic_data_comparison_flow,
    "agentic_fact_checker": agentic_fact_checker_flow,
    "agentic_tutorial": agentic_tutorial_flow,
    "agentic_application": agentic_application_flow,
}


def get_template(name: str) -> ReasoningFlow | None:
    """Get a built-in flow template by name."""
    factory = BUILTIN_TEMPLATES.get(name)
    return factory() if factory else None


def list_templates() -> list[dict[str, str]]:
    """List available template names and descriptions."""
    result = []
    for name, factory in BUILTIN_TEMPLATES.items():
        flow = factory()
        result.append({
            "name": name,
            "display_name": flow.name,
            "description": flow.description,
            "step_count": len(flow.steps),
        })
    return result
