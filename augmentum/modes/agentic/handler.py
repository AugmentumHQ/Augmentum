"""Agentic mode handler — multi-step autonomous task execution.

Combines the passthrough chain execution engine (wave-based parallel
execution, dependency DAGs, error recovery, plan mutation) with the
agentic UX layer (checkpoints, approval gates, artifact tools, streaming
task inspector).

Tool-heavy steps delegate to ``execute_chain()`` from ``chain.py``,
getting parallel waves, template substitution, and LLM-driven error
recovery.  Generative steps (draft, review, deliver) use direct LLM
calls with accumulated working memory from prior chain results.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.agentic.autonomy import (
    build_approval_chunk,
    build_inform_chunk,
    build_plan_approval_chunk,
    needs_plan_approval,
    needs_step_approval,
)
from augmentum.modes.agentic.planner import (
    PLAN_SYSTEM_PROMPT,
    mark_current_step,
    parse_plan,
    plan_to_context,
    update_plan_step,
)
from augmentum.modes.agentic.task_state import (
    TaskState,
    TaskStatus,
    TaskStore,
    ToolCallCache,
)
from augmentum.modes.agentic.working_memory import WorkingMemory
from augmentum.modes.base import ModeHandler
from augmentum.reasoning.variables import StepContext, build_user_message, resolve_variables
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend
    from augmentum.reasoning.models import FlowStep, ReasoningFlow
    from augmentum.reasoning.store import FlowStore
    from augmentum.tools.chain import ChainPlan
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

# Step roles that the agentic handler understands
_AGENTIC_ROLES = {"plan", "draft", "create", "illustrate", "review", "deliver"}
# Roles from the analytical pipeline that also work in agentic flows
_ANALYTICAL_ROLES = {"classify", "search", "analyze", "verify", "respond"}
_ALL_ROLES = _AGENTIC_ROLES | _ANALYTICAL_ROLES

# Roles that are tool-heavy by nature (always use chain executor)
_CHAIN_ROLES = frozenset({"search", "verify"})

# Chain-step tools whose artifact output is a build *input* (slide images,
# illustrations) rather than a deliverable — flagged ``intermediate`` so the
# frontend keeps the per-step card out of the chat transcript.
_INTERMEDIATE_VISUAL_TOOLS = frozenset({"image_generation", "image_search"})

# Roles where the primary action is content generation.  Even if these
# steps list supplementary tools (e.g. calculator for Draft, text_analysis
# for Review), the MAIN output is LLM-generated prose — not tool results.
# Routing these through the chain executor would produce tool output
# instead of the actual content.
_GENERATIVE_ROLES = frozenset({"plan", "draft", "review", "deliver", "respond"})

# Synthetic tool schema used by review steps to submit a structured verdict.
# The tool is never executed — its call is intercepted and stored on wmem.
_SUBMIT_REVIEW_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": (
            "Submit your review verdict. Call this exactly once with PASS if the "
            "draft is ready, or REVISE with a concrete list of issues that must "
            "be fixed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["PASS", "REVISE"],
                    "description": "PASS if ready, REVISE if issues must be addressed.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One or two sentences explaining the verdict.",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete issues to fix. Required when verdict=REVISE; empty list when verdict=PASS.",
                },
            },
            "required": ["verdict", "reasoning"],
        },
    },
}
_SUBMIT_REVIEW_INSTRUCTION = (
    "When you have finished reviewing, call the submit_review tool with your "
    "verdict. Do NOT emit VERDICT: text — use the tool."
)

# Verbs that indicate a multi-step task (not a simple question)
_TASK_VERBS = frozenset({
    "create", "generate", "write", "build", "research", "analyze",
    "compare", "make", "prepare", "draft", "design", "produce",
    "develop", "compile", "compose", "construct", "summarize",
})


# Slide-marker grammar for the create_presentation step.
# Models reliably emit one of these forms but not always the canonical
# `### Slide N:` requested by the prompt — accept the common variants.
_SLIDE_MARKER_RE = re.compile(
    r"^(?:"
    r"\#{1,3}\s+(?:Slide\s+\d+[:.]\s*)?"      # #/##/### with optional "Slide N:" prefix
    r"|\*{0,2}\s*Slide\s+\d+[:.]\s*\*{0,2}\s*"  # bare or bolded "Slide N:" line
    r")",
    flags=re.MULTILINE | re.IGNORECASE,
)
_NOTES_PREFIX_RE = re.compile(r"^\*?\*?[Nn]otes:\*?\*?\s*")
# A draft larger than this that parses to zero slides is almost certainly
# a marker-format mismatch, not a legitimate one-slide deck.
_SLIDE_PARSE_COLLAPSE_THRESHOLD = 600

# Cap on synthetic illustrations generated per Illustrate step — bounds the
# step's runtime (each generation is a full diffusion job). Real-photo search
# candidates are cheap and uncapped beyond their per-unit count.
_ILLUSTRATE_MAX_GENERATED = 8


# Plan titles models commonly emit when they don't bother extracting a real
# title from the user's query. Treat these as not-actually-a-title and fall
# back to the original_query so the artifact pipeline gets a real topic.
_META_TITLES = frozenset({
    "plan", "task", "document", "presentation", "deck", "slides",
    "report", "spreadsheet", "chart", "untitled",
})


# Step-name fragments that mean "this step produced the textual content
# we want to render as an artifact" — i.e., the deliverable draft. Ordered
# from most to least specific so a later "Draft Content" beats an earlier
# "Structure Slides" when both exist.
_DRAFT_STEP_HINTS = ("draft", "write", "content", "structure", "outline", "compose")
# Step-name fragments that *do* produce text but explicitly aren't the draft —
# they emit tool-call status, review verdicts, or delivery summaries.
_NON_DRAFT_STEP_HINTS = ("illustrate", "review", "deliver", "image", "render")


def _pick_artifact_draft(wmem) -> str:
    """Pick the working-memory step whose output is the artifact draft.

    The agentic flow records several steps: Plan, Research, Structure
    Slides, Draft Content, Illustrate, Create. A naive "most recent step
    with >200 chars" loop grabs Illustrate's image_generation status
    string and ships it as the deliverable. Prefer explicitly draft-shaped
    step names; only fall back to "latest large generative output" when
    no draft step is identifiable.
    """
    names = list(wmem.all_step_names)
    candidates: list[tuple[int, str, str]] = []
    for idx, name in enumerate(names):
        name_lc = name.lower()
        if any(skip in name_lc for skip in _NON_DRAFT_STEP_HINTS):
            continue
        output = wmem.get_step_output(name)
        if not output or len(output) < 200:
            continue
        # Skip chain (tool-result) steps — they're not the draft text.
        is_chain = name in getattr(wmem, "_chain_results", {})
        if is_chain:
            continue
        score = 0
        for hint in _DRAFT_STEP_HINTS:
            if hint in name_lc:
                score += 2
        # Tie-break by recency (later step preferred) and length.
        candidates.append((score * 10000 + idx * 100 + len(output) // 100, name, output))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]

    # Last-resort fallback — keeps behaviour for ad-hoc tasks that don't
    # name a draft step. Still skips the explicit non-draft hints.
    for name in reversed(names):
        name_lc = name.lower()
        if any(skip in name_lc for skip in _NON_DRAFT_STEP_HINTS):
            continue
        output = wmem.get_step_output(name)
        if output and len(output) > 200:
            return output
    return ""


def _pick_artifact_draft_from_outputs(task) -> str:
    """Draft picker for code paths without a live WorkingMemory.

    The picker REST routes run outside the handler's streaming loop —
    they only have ``task.step_outputs`` (the persisted snapshot of each
    step's output text). Walk the recorded outputs in order, prefer
    "draft" / "structure" / "outline" / "content" hits, skip illustrate
    / review / deliver outputs (which are status text, not the draft).

    Step names aren't stored on the task — only their integer indices —
    so the heuristic is "longest non-illustrate output", which empirically
    lines up with the Draft Content step for the bundled Presentation
    flow.
    """
    outputs = getattr(task, "step_outputs", {}) or {}
    if not outputs:
        return ""
    # Sort by index and pick the longest output whose body doesn't look
    # like a tool-result status string (image_generation / image_search /
    # create_* receipts).
    status_markers = (
        "image generated successfully",
        "found ", "illustrated ", "image_search",
        "presentation '", "document '",
    )
    best = ""
    best_len = 0
    for idx in sorted(outputs.keys()):
        text = outputs.get(idx) or ""
        if not isinstance(text, str) or len(text) < 200:
            continue
        head = text[:200].lower()
        if any(m in head for m in status_markers):
            continue
        if len(text) > best_len:
            best = text
            best_len = len(text)
    return best


def _apply_image_picks(slides: list[dict], wmem, task) -> list[dict]:
    """Project Illustrate Slides picks onto a parsed slide list.

    For each slide whose 1-based index appears in the task's
    ``slide_image_picks``, sets ``image_url`` to the primary candidate's
    embed_url and ``additional_images`` to the appended candidates' URLs.

    Reads picks + candidates from wmem first (current run), falling back
    to the task state (resumed run, or when this helper is called from
    the /pick re-render path without a working memory).

    Slides not present in the picks dict are returned unchanged.
    """
    picks = getattr(wmem, "_slide_image_picks", None) or getattr(
        task, "slide_image_picks", {}
    ) or {}
    candidates = getattr(wmem, "_image_candidates", None) or getattr(
        task, "image_candidates", {}
    ) or {}
    if not picks or not candidates:
        return slides

    def _candidate_by_id(slide_idx: int, cid: str) -> dict | None:
        pool = candidates.get(slide_idx) or []
        for c in pool:
            if c.get("candidate_id") == cid:
                return c
        return None

    out: list[dict] = []
    for i, slide in enumerate(slides):
        slide_idx = i + 1  # 1-based — matches the crafter's index field
        pick = picks.get(slide_idx)
        if not pick:
            out.append(slide)
            continue
        new_slide = dict(slide)
        primary_id = pick.get("primary", "")
        primary = _candidate_by_id(slide_idx, primary_id) if primary_id else None
        if primary:
            new_slide["image_url"] = primary.get("embed_url", "")
        additional_ids = pick.get("additional") or []
        additional_urls = []
        for cid in additional_ids[:3]:  # hard cap matches the picker UI
            c = _candidate_by_id(slide_idx, cid)
            if c and c.get("embed_url"):
                additional_urls.append(c["embed_url"])
        if additional_urls:
            new_slide["additional_images"] = additional_urls
        out.append(new_slide)
    return out


def _resolve_artifact_topic(task) -> str:
    """Pick the strongest available description of the user's ask.

    Plan-step titles often collapse to meta-words ("Plan"). When that
    happens, the original query carries more signal — use it as the
    drafter's topic so the LLM has actual content to write about.
    """
    title = (getattr(task, "title", "") or "").strip()
    query = (getattr(task, "original_query", "") or "").strip()
    if not title:
        return query or "Document"
    if title.lower() in _META_TITLES:
        return query or title
    # Single bare word — almost always too thin to draft from.
    if " " not in title and query:
        return query
    return title


def _detect_plan_format(task, wmem) -> str:
    """Read the Plan step's ``FORMAT: code|procedure`` label, if present.

    Shape-aware flows (Tutorial) emit a FORMAT line so downstream steps can
    branch physical procedures (real photos) from code topics (synthetic
    diagrams). Scans the live working memory's plan output first, then the
    persisted ``step_outputs`` snapshot. Returns "code" / "procedure" / "".
    """
    blobs: list[str] = []
    try:
        for name in getattr(wmem, "all_step_names", []) or []:
            if "plan" in name.lower():
                out = wmem.get_step_output(name)
                if out:
                    blobs.append(out)
    except Exception:
        pass
    outputs = getattr(task, "step_outputs", {}) or {}
    blobs.extend(str(v) for v in outputs.values() if v)
    m = re.search(
        r"FORMAT:\s*['\"]?(code|procedure)", "\n".join(blobs), re.IGNORECASE,
    )
    return m.group(1).lower() if m else ""


def _parse_slide_draft(
    draft: str, *, fallback_title: str = "Content"
) -> tuple[list[dict], str]:
    """Parse a slide-deck draft into structured slides.

    Returns ``(slides, warning)``. ``warning`` is non-empty when the
    parser fell back to a single slide on a draft large enough that
    the LLM almost certainly *meant* multiple slides. Callers should
    surface the warning to the agent/user rather than silently shipping
    a one-slide deck.
    """
    slides: list[dict] = []
    parts = _SLIDE_MARKER_RE.split(draft)
    # parts[0] is pre-marker preamble (e.g. "Here's the presentation:")
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n")
        title = lines[0].strip()
        # Strip trailing markdown emphasis the marker regex couldn't span,
        # e.g. **Slide 1: Title**  → opener consumed, closer left behind.
        title = title.rstrip("*").rstrip()
        if not title:
            continue
        body_lines: list[str] = []
        notes = ""
        in_notes = False
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.lower().startswith(("**notes:**", "notes:")):
                in_notes = True
                notes_text = _NOTES_PREFIX_RE.sub("", stripped)
                if notes_text:
                    notes = notes_text
                continue
            if in_notes:
                notes = f"{notes}\n{stripped}" if notes else stripped
            else:
                body_lines.append(line)
        slides.append({
            "layout": "content",
            "title": title,
            "body": "\n".join(body_lines).strip(),
            "notes": notes.strip(),
        })

    if slides:
        return slides, ""

    # No markers found — emit one slide containing the full draft so the
    # user gets *something*, but flag the collapse if the draft is large
    # enough that the LLM clearly meant a multi-slide deck.
    slides = [{"layout": "content", "title": fallback_title, "body": draft}]
    if len(draft) > _SLIDE_PARSE_COLLAPSE_THRESHOLD:
        warning = (
            f"slide parser found no slide markers in a {len(draft)}-char "
            "draft; collapsed to a single slide. Re-draft using "
            "'### Slide N: <title>' markers to get a proper deck."
        )
        return slides, warning
    return slides, ""


def _parse_document_sections(
    draft: str, *, fallback_title: str = "Content"
) -> list[dict]:
    """Parse a document draft into ``[{heading, body, level}]`` sections.

    Recognises ``## SECTION: <heading>`` markers (the artifact draft
    contract), falling back to bare ``## <heading>`` and finally a single
    whole-draft section. Shared by ``_execute_create_document_step`` and the
    Illustrate loop so per-section image indices line up — the pick pool is
    keyed by 1-based section order, so both must parse identically.
    """

    def _strip_ai_preamble(text: str) -> str:
        for prefix in ("Here is ", "Here's ", "Below is ",
                       "I've created ", "I have created "):
            if text.lower().startswith(prefix.lower()):
                nl = text.find("\n")
                if 0 < nl < 200:
                    return text[nl:].strip()
                break
        return text

    def _split(marker: str) -> list[dict]:
        out: list[dict] = []
        parts = re.split(marker, draft, flags=re.MULTILINE)
        preamble = parts[0].strip() if parts else ""
        if preamble and len(preamble) > 50:
            preamble = _strip_ai_preamble(preamble)
            if preamble and len(preamble) > 50:
                out.append({"heading": "Introduction", "body": preamble, "level": 1})
        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
            lines = part.split("\n", 1)
            heading = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if heading and body:
                out.append({"heading": heading, "body": body, "level": 1})
        return out

    sections = _split(r"^## SECTION:\s*")
    if not sections:
        sections = _split(r"^##\s+")
    if not sections:
        sections = [{"heading": fallback_title, "body": draft, "level": 1}]
    return sections


class AgenticHandler(ModeHandler):
    """Handle multi-step agentic tasks with plan management and checkpoints."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        tool_registry: ToolRegistry | None = None,
        session_id: str = "",
        user_id: str = "",
        task_store: TaskStore | None = None,
        tool_call_cache: ToolCallCache | None = None,
        flow_store: FlowStore | None = None,
        artifact_store=None,
        build_run_store=None,
        flow_tune: dict | None = None,
        explicit_flow_id: str = "",
    ) -> None:
        self._backend = backend
        self._tool_registry = tool_registry
        self._session_id = session_id
        self._user_id = user_id
        self._task_store = task_store
        self._tool_call_cache = tool_call_cache
        self._flow_store = flow_store
        self._artifact_store = artifact_store
        self._build_run_store = build_run_store
        self._flow_tune = flow_tune
        # Caller's flow selection from the X-Augmentum-Flow header. Honoured
        # by _run_new_task via the canonical resolve_flow() — without this
        # the handler keyword-matched against trigger_keywords and routinely
        # picked the Application flow on any "build/app/tool"-containing
        # query, regardless of what the user actually selected.
        self._explicit_flow_id = explicit_flow_id

    async def _handle(self, request: InternalChatRequest) -> InternalChatResponse:
        """Non-streaming agentic execution — collects all output."""
        full_output: list[str] = []
        async for chunk in self._handle_stream(request):
            if chunk.content_delta:
                full_output.append(chunk.content_delta)

        return InternalChatResponse(
            message=Message(role="assistant", content="".join(full_output)),
            model=request.model,
            finish_reason="stop",
        )

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream agentic task execution with live progress."""
        query = _extract_query(request)
        model = request.model

        existing_task = None
        if self._task_store:
            existing_task = await self._task_store.get_incomplete_for_session(
                self._session_id, user_id=self._user_id,
            )

        if existing_task and not _is_new_task_request(query):
            if existing_task.status == TaskStatus.APPROVAL_PENDING:
                async for chunk in self._handle_approval_response(
                    existing_task, query, model, request
                ):
                    yield chunk
                return
            async for chunk in self._resume_task(existing_task, model, request):
                yield chunk
            return

        async for chunk in self._run_new_task(query, model, request):
            yield chunk

    # ------------------------------------------------------------------
    # New task execution
    # ------------------------------------------------------------------

    async def _run_new_task(
        self,
        query: str,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Plan and execute a brand new agentic task."""
        flow = await self._resolve_agentic_flow(query, model)
        if not flow:
            async for chunk in self._run_ad_hoc(query, model, request):
                yield chunk
            return

        # Dynamic flows skip the fixed-DAG executor and run a ReAct-style
        # loop over the full tool registry. Useful when the task shape
        # isn't known in advance and a rigid plan would be premature.
        if getattr(flow, "kind", "workflow") == "dynamic":
            async for chunk in self._run_dynamic_loop(flow, query, model, request):
                yield chunk
            return

        autonomy = flow.autonomy_level if flow.autonomy_level else settings.agentic_default_autonomy
        task = TaskState(
            session_id=self._session_id,
            user_id=self._user_id,
            flow_id=flow.id,
            status=TaskStatus.PLANNING,
            autonomy_level=autonomy,
            original_query=query,
            title="",
        )

        if self._task_store:
            task = await self._task_store.create(task)

        yield _agentic_meta_chunk(model, "planning", task)

        # Generate a plan that maps to the actual flow steps
        enabled_steps = [s for s in flow.steps if s.enabled]
        step_list = "\n".join(f"  {i+1}. {s.name}" for i, s in enumerate(enabled_steps))
        flow_plan_prompt = (
            "The following steps WILL be executed, in order, exactly once:\n"
            f"{step_list}\n\n"
            "Write a ONE-LINE description for each step describing what it "
            "will do for the user's specific request. Be concrete — mention "
            "specific topics, data sources, or actions relevant to this query.\n"
            "\n"
            "RULES:\n"
            "- Every step listed above WILL run. Do NOT mark any step as "
            "'(Optional)', '(if needed)', '(if time permits)', conditional, "
            "or maybe.\n"
            "- Do NOT add steps that are not in the list above.\n"
            "- Do NOT skip steps.\n"
            "- Do NOT change the step names — keep them exactly as given.\n"
            "- Each description is a single committed sentence in present "
            "or future tense (e.g. 'Generates three illustrations for the "
            "key chapters', not 'May generate illustrations if time allows').\n"
            "\n"
            "Format:\n"
            "## Task: <short title>\n\n"
            + "\n".join(
                f"- [ ] {i+1}. {s.name}: <what this step does for this query>"
                for i, s in enumerate(enabled_steps)
            )
        )

        plan_text = await self._call_llm(
            model=model,
            system_prompt=flow_plan_prompt,
            user_message=query,
            request=request,
            max_tokens=800,
        )

        title, step_descs = parse_plan(plan_text)
        task.title = title or query[:60]
        task.plan_md = plan_text
        task.total_steps = sum(1 for s in flow.steps if s.enabled)
        task.status = TaskStatus.RUNNING

        if self._task_store:
            await self._task_store.update(task)

        yield _agentic_meta_chunk(model, "plan_ready", task)

        if needs_plan_approval(task.autonomy_level):
            task.status = TaskStatus.APPROVAL_PENDING
            if self._task_store:
                await self._task_store.update(task)
            yield build_plan_approval_chunk(model, task)
            return

        async for chunk in self._execute_flow_steps(flow, task, query, model, request):
            yield chunk

    async def _run_dynamic_loop(
        self,
        flow: ReasoningFlow,
        query: str,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """ReAct-style agent loop: model decides next tool or final answer.

        On each iteration the LLM sees the user query, the running tool
        trace, and the full tool registry (as native tool_use schemas). It
        either emits tool_calls (which we execute and feed back) or a
        final message (which we stream to the user). Terminates on:
        * A final message with no tool_calls, OR
        * ``settings.agentic_dynamic_max_steps`` iterations reached.
        """
        task = TaskState(
            session_id=self._session_id,
            user_id=self._user_id,
            flow_id=flow.id,
            status=TaskStatus.RUNNING,
            autonomy_level=flow.autonomy_level or settings.agentic_default_autonomy,
            original_query=query,
            title=query[:60],
            total_steps=0,
        )
        if self._task_store:
            task = await self._task_store.create(task)
        yield _agentic_meta_chunk(model, "running", task)

        # Build native tool schemas from the full registry
        from augmentum.modes.analytical.tool_calling import tools_to_native_format
        if self._tool_registry:
            available = [t for t in self._tool_registry.all_tools()] \
                if hasattr(self._tool_registry, "all_tools") else []
        else:
            available = []
        native_tools = tools_to_native_format(available) if available else None

        system_prompt = (
            "You are an autonomous agent. Decide each step by calling tools "
            "or writing a final response. Prefer tools for any factual or "
            "computational claim. Stop calling tools once you have enough "
            "information to answer. Cite sources inline as [1], [2]."
        )

        trace: list[str] = [f"User request: {query}"]
        max_steps = settings.agentic_dynamic_max_steps
        for step_i in range(max_steps):
            user_msg = "\n\n".join(trace) + "\n\nDecide the next action."
            try:
                msg = await self._call_llm_full(
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_msg,
                    request=request,
                    max_tokens=2048,
                    tools=native_tools,
                )
            except Exception as exc:
                log.error("dynamic_loop_llm_failed", task_id=task.id, step=step_i, exc_info=True)
                task.status = TaskStatus.FAILED
                task.error = f"LLM call failed: {exc}"
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "failed", task)
                return

            if not msg.tool_calls:
                # Final message — stream to user and exit.
                yield InternalStreamChunk(
                    content_delta=msg.content or "",
                    model=model,
                    augmentum={"mode": "agentic", "task_id": task.id},
                )
                task.status = TaskStatus.COMPLETED
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "completed", task)
                return

            # Execute the tool calls (cap to max_tool_calls_per_step)
            cap = flow.max_tool_calls_per_step or 3
            results = await self._dispatch_structured_tool_calls(
                msg.tool_calls[:cap], available,
            )
            task.tool_calls_made += len(results)
            if self._task_store:
                await self._task_store.update(task)
            for name, output in results:
                trace.append(f"## Tool: {name}\n{output[:1500]}")

        # Exhausted step budget — synthesize a best-effort final answer.
        log.warning("dynamic_loop_budget_exhausted", task_id=task.id, steps=max_steps)
        final_user = "\n\n".join(trace) + (
            "\n\nYou have reached the step budget. Write the final answer now "
            "with whatever you have, acknowledging any gaps."
        )
        final = await self._call_llm(
            model=model,
            system_prompt=system_prompt,
            user_message=final_user,
            request=request,
            max_tokens=1024,
        )
        yield InternalStreamChunk(
            content_delta=final,
            model=model,
            augmentum={"mode": "agentic", "task_id": task.id},
        )
        task.status = TaskStatus.COMPLETED
        if self._task_store:
            await self._task_store.update(task)
        yield _agentic_meta_chunk(model, "completed", task)

    async def _run_ad_hoc(
        self,
        query: str,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Fallback: no defined flow — use the chain planner for adaptive execution."""
        task = TaskState(
            session_id=self._session_id,
            user_id=self._user_id,
            status=TaskStatus.PLANNING,
            autonomy_level=settings.agentic_default_autonomy,
            original_query=query,
        )
        if self._task_store:
            task = await self._task_store.create(task)

        yield _agentic_meta_chunk(model, "planning", task)

        plan_text = await self._call_llm(
            model=model,
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_message=query,
            request=request,
        )

        title, step_descs = parse_plan(plan_text)
        task.title = title or query[:60]
        task.plan_md = plan_text
        task.total_steps = len(step_descs)
        task.status = TaskStatus.RUNNING

        if self._task_store:
            await self._task_store.update(task)

        yield _agentic_meta_chunk(model, "plan_ready", task)

        # Complexity gate: simple questions get a direct response, not a pipeline
        words = query.split()
        is_simple = len(words) < 15 and not any(
            w.lower().rstrip(".,!?") in _TASK_VERBS for w in words
        )
        if is_simple:
            sp = self._inject_date(
                "You are a helpful assistant. Answer the user's question "
                "directly and concisely."
            )
            user_system = _extract_user_system(request)
            if user_system:
                sp = f"{user_system}\n\n{sp}"
            req = InternalChatRequest(
                model=model,
                messages=[Message(role="system", content=sp),
                          Message(role="user", content=query)],
                stream=True,
            )
            async for chunk in self._backend.chat_stream(req):
                if chunk.content_delta:
                    yield InternalStreamChunk(
                        content_delta=chunk.content_delta,
                        model=model,
                        augmentum={"mode": "agentic", "task_id": task.id},
                    )
            task.status = TaskStatus.COMPLETED
            if self._task_store:
                await self._task_store.update(task)
            yield _agentic_meta_chunk(model, "completed", task)
            return

        if needs_plan_approval(task.autonomy_level):
            task.status = TaskStatus.APPROVAL_PENDING
            if self._task_store:
                await self._task_store.update(task)
            yield build_plan_approval_chunk(model, task)
            return

        # Try chain-based execution (wave parallelism, error recovery)
        if self._tool_registry:
            from augmentum.tools.chain import (
                ToolChainPlanner,
                build_synthesis_prompt,
            )

            tools = self._tool_registry.list_tools()
            planner = ToolChainPlanner(self._backend, self._tool_registry)

            try:
                chain_result = await asyncio.wait_for(
                    planner.plan_and_execute(
                        request, tools,
                        extra_tool_args={
                            "task_id": task.id,
                            "_task_id": task.id,
                            "session_id": self._session_id,
                            "_session_id": self._session_id,
                            "_request_model": model,
                        },
                        cache_user_id=self._user_id or "",
                    ),
                    timeout=settings.agentic_step_timeout,
                )
            except Exception:
                chain_result = None
                log.warning(
                    "ad_hoc_chain_plan_failed_fallback",
                    task_id=task.id,
                    exc_info=True,
                )

            if chain_result is not None:
                results, chain_plan = chain_result

                # Use the chain step's tool + reason for a meaningful label
                # — the inspector pipeline shows these directly to the user.
                def _step_label(s):
                    base = (s.tool or "").strip() or f"step{s.id}"
                    reason = (s.reason or "").strip()
                    if reason:
                        # Keep labels short — first sentence / 60 chars max.
                        snippet = reason.split(".")[0].strip()[:60]
                        return f"{base}: {snippet}" if snippet else base
                    return base

                pipeline = [_step_label(s) for s in chain_plan.steps]
                for s, label in zip(chain_plan.steps, pipeline, strict=True):
                    r = results.get(s.id)
                    if r:
                        # Persist the full tool output — the inspector and
                        # context-injection paths apply their own display
                        # caps (preview ~200 chars, synth ~500 chars). Storing
                        # the full result lets resume and later synthesis
                        # work from the same data the step actually produced.
                        task.record_step_output(s.id, r.output)
                        task.tool_calls_made += 1
                        yield _flow_step_chunk(
                            model, label, "complete", pipeline, task,
                        )

                if self._task_store:
                    await self._task_store.update(task)

                synth_prompt = build_synthesis_prompt(
                    chain_plan, results, user_query=query,
                )
                synth_request = InternalChatRequest(
                    model=model,
                    messages=[*request.messages, Message(role="user", content=synth_prompt)],
                    stream=True,
                )
                async for chunk in self._backend.chat_stream(synth_request):
                    if chunk.content_delta:
                        yield InternalStreamChunk(
                            content_delta=chunk.content_delta,
                            model=model,
                            augmentum={"mode": "agentic", "task_id": task.id},
                        )

                task.status = TaskStatus.COMPLETED
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "completed", task)
                return

        # Fallback: sequential LLM execution (no chain planner available).
        # Use the parsed plan-step descriptions as the pipeline labels so
        # the inspector shows real step names, not "Step 1, Step 2, …".
        def _short(desc: str) -> str:
            line = (desc or "").strip().split("\n", 1)[0]
            # Strip trailing "(Optional)" / parenthetical hints that crowd the label.
            return line[:60] if len(line) <= 60 else line[:57] + "…"
        pipeline = [_short(d) or f"Step {i + 1}" for i, d in enumerate(step_descs)]
        wmem = WorkingMemory(goal=query, plan_md=plan_text)

        for i, step_desc in enumerate(step_descs):
            task.current_step = i
            marked_plan = mark_current_step(task.plan_md, i)
            plan_ctx = plan_to_context(marked_plan)
            is_last_step = i == len(step_descs) - 1

            step_content = ""
            if is_last_step:
                step_content = f"### {step_desc}\n\n"
            step_label = pipeline[i] if i < len(pipeline) else f"Step {i + 1}"
            yield _flow_step_chunk(
                model, step_label, "running", pipeline, task,
                content=step_content,
            )

            prior_context = wmem.build_context_for_step(f"Step {i + 1}")
            step_system = (
                "You are executing one step of a multi-step task. "
                "Complete this step thoroughly and provide your output."
            )
            step_user = f"Original request: {query}\n\n"
            if prior_context:
                step_user += f"{prior_context}\n\n"
            step_user += f"Current step: {step_desc}\n{plan_ctx}"

            if is_last_step:
                step_output_parts: list[str] = []
                async for chunk in self._stream_llm(
                    model=model,
                    system_prompt=step_system,
                    user_message=step_user,
                    request=request,
                ):
                    step_output_parts.append(chunk.content_delta)
                    yield InternalStreamChunk(
                        content_delta=chunk.content_delta,
                        model=model,
                        augmentum={"mode": "agentic", "task_id": task.id},
                    )
                step_output = "".join(step_output_parts)
            else:
                step_output = await self._call_llm(
                    model=model,
                    system_prompt=step_system,
                    user_message=step_user,
                    request=request,
                )

            wmem.record_generative_output(f"Step {i + 1}", step_output)
            task.record_step_output(i, step_output)
            task.plan_md = update_plan_step(task.plan_md, i)

            if step_output:
                yield InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum={
                        "mode": "agentic",
                        "task_id": task.id,
                        "phase": step_label,
                        "phase_content_delta": step_output,
                    },
                )

            if self._task_store:
                await self._task_store.update(task)

            yield _flow_step_chunk(model, step_label, "complete", pipeline, task)

        task.status = TaskStatus.COMPLETED
        if self._task_store:
            await self._task_store.update(task)
        yield _agentic_meta_chunk(model, "completed", task)

    # ------------------------------------------------------------------
    # Flow-based execution
    # ------------------------------------------------------------------

    async def _execute_flow_steps(
        self,
        flow: ReasoningFlow,
        task: TaskState,
        query: str,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute each step in the resolved agentic flow.

        Tool-heavy steps (those with tool_names/tool_categories) are
        delegated to the chain executor for wave-based parallel execution,
        error recovery, and plan mutation.  Generative steps get direct
        LLM calls with accumulated working memory context.
        """
        wmem = WorkingMemory(goal=query, plan_md=task.plan_md)
        ctx = StepContext(query=query, model=model)
        ctx.plan = task.plan_md
        ctx.conversation = _build_conversation(request)

        pipeline = [
            {
                "name": s.name,
                "role": s.role,
                "tools": list(s.tool_names or []),
            }
            for s in flow.steps if s.enabled
        ]
        max_steps = settings.agentic_max_steps

        # Track mid-flow re-planning: if a draft fails review even after
        # revision, a planner LLM proposes an additional step. Capped at
        # settings.agentic_max_insertions per flow to prevent runaway growth.
        insertions_made = 0
        steps_executed = 0
        for step_idx, step in enumerate(flow.steps):
            if not step.enabled:
                continue

            if steps_executed >= max_steps:
                log.warning("agentic_max_steps_reached", task_id=task.id, max_steps=max_steps)
                task.status = TaskStatus.FAILED
                task.error = f"Safety limit: max {max_steps} steps reached"
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "failed", task)
                return

            steps_executed += 1
            task.current_step = step_idx
            ctx.plan = mark_current_step(task.plan_md, step_idx)

            # Autonomy check — tool count comes from the step spec, so the
            # level-2 threshold for "many tool calls" can actually fire.
            tool_count = len(step.tool_names or [])
            if needs_step_approval(task.autonomy_level, step.role, tool_count):
                task.status = TaskStatus.APPROVAL_PENDING
                if self._task_store:
                    await self._task_store.update(task)
                yield build_approval_chunk(
                    model, task, step.name, step.role,
                    description=f"Execute step: {step.name} (role: {step.role})",
                )
                return

            # Emit step-start
            step_content = ""
            if step.stream_to_user:
                streaming_count = sum(1 for s in flow.steps if s.enabled and s.stream_to_user)
                if streaming_count > 1:
                    prior_streamed = any(
                        s.stream_to_user for s in flow.steps[:step_idx] if s.enabled
                    )
                    step_content = (
                        f"\n\n---\n\n### {step.name}\n\n"
                        if prior_streamed
                        else f"### {step.name}\n\n"
                    )
            yield _flow_step_chunk(model, step.name, "running", pipeline, task, content=step_content)

            # Pre-flight health check — verify required tools are available
            # before spending time and tokens on a step that will fail.
            step_tools = self._resolve_step_tools(step)
            unhealthy = []
            for st in step_tools:
                try:
                    if not st.health_check():
                        unhealthy.append(st.name)
                except Exception:
                    unhealthy.append(st.name)
            if unhealthy:
                log.warning("agentic_step_unhealthy_tools",
                            step=step.name, unhealthy=unhealthy)
                yield _flow_step_chunk(
                    model, step.name, "warning", pipeline, task,
                    content=f"⚠ Tools unavailable: {', '.join(unhealthy)}. Proceeding without them.\n",
                )
                # Remove unhealthy tools so the step can still try
                step_tools = [t for t in step_tools if t.name not in unhealthy]

            # Classify and execute
            is_chain = _is_chain_step(step)

            # Build focused delivery context for deliver steps.
            # Provide structured, readable context so the LLM can synthesize
            # a polished final response. Preserve paragraph structure instead
            # of flattening to single lines.
            if step.role == "deliver":
                delivery_parts = []
                for sname, skind, _ in wmem._steps:
                    if skind == "generative":
                        soutput = wmem._generative_outputs.get(sname, "")
                        if soutput and len(soutput) > 100:
                            # Preserve paragraph structure — truncate by paragraphs not chars
                            paras = [p.strip() for p in soutput.split("\n\n") if p.strip()]
                            budget = 2000  # chars per generative step
                            kept = []
                            used = 0
                            for p in paras:
                                if used + len(p) > budget:
                                    break
                                kept.append(p)
                                used += len(p)
                            if kept:
                                delivery_parts.append(f"## {sname}\n" + "\n\n".join(kept))
                                if len(kept) < len(paras):
                                    delivery_parts[-1] += f"\n\n[...{len(paras) - len(kept)} more paragraphs]"
                    elif skind == "chain":
                        sresults = wmem._chain_results.get(sname, {})
                        # Collect both artifact links and key findings
                        artifact_lines = []
                        finding_lines = []
                        for r in sresults.values():
                            if not r.success:
                                continue
                            if r.metadata and (
                                r.metadata.get("download_url")
                                or r.metadata.get("url")
                                or r.metadata.get("image_id")
                            ):
                                label = r.tool_name
                                url = r.metadata.get("download_url") or r.metadata.get("url", "")
                                artifact_lines.append(f"- [{label}]: {url}")
                            elif r.output and len(r.output) > 50:
                                # Include key search findings (truncated)
                                finding_lines.append(f"- {r.tool_name}: {r.output[:300]}")
                        parts = []
                        if finding_lines:
                            parts.extend(finding_lines[:5])  # top 5 findings
                        if artifact_lines:
                            parts.extend(artifact_lines)
                        if parts:
                            delivery_parts.append(f"## {sname}\n" + "\n".join(parts))
                if wmem._artifacts:
                    delivery_parts.append("## Created Artifacts\n" + "\n".join(
                        f"- **{a.get('display_name', '?')}**: {a.get('download_url', a.get('url', ''))}"
                        for a in wmem._artifacts
                    ))
                # Delivery rules now live in the step's system prompt
                # (templates._DELIVER_SYSTEM_BASE). Only the raw context is
                # injected here; runtime-specific nudges can be appended
                # conditionally (e.g., when artifacts exist).
                raw_context = "\n\n".join(delivery_parts) if delivery_parts else ctx.all_outputs
                # Ground truth FIRST — so Deliver reports what the artifacts
                # actually contain, not the planned outline (the run that broke
                # this: deck collapsed to 1 slide, review said REVISE, Deliver
                # still claimed "10 slides").
                _status = _artifact_status_preamble(wmem, ctx)
                if _status:
                    raw_context = _status + "\n\n---\n\n" + raw_context
                ctx._step_outputs["_delivery_context"] = raw_context

            # Unified artifact pipeline intercept for create steps
            from augmentum.tools.artifact_pipeline import ARTIFACT_TOOLS

            is_artifact_step = (
                step.role == "create"
                and self._tool_registry
                and any(tn in ARTIFACT_TOOLS for tn in (step.tool_names or []))
            )

            # Illustrate dispatcher — when an illustrate step lists either
            # image tool we run the deterministic per-unit crafter + dual
            # search/generation loop (capability-matched, FORMAT-aware)
            # instead of letting the chain executor free-fire image_generation
            # with the user's default model (the "anime tire" failure).
            _illus_tools = step.tool_names or []
            is_illustrate_search_step = (
                step.role == "illustrate"
                and self._tool_registry
                and ("image_search" in _illus_tools
                     or "image_generation" in _illus_tools)
            )

            try:
                timeout = settings.agentic_step_timeout

                if is_artifact_step:
                    step_output = await asyncio.wait_for(
                        self._execute_artifact_pipeline_step(
                            step, model, request, task, wmem,
                        ),
                        timeout=timeout,
                    )
                elif is_illustrate_search_step:
                    step_output = await asyncio.wait_for(
                        self._execute_illustrate_step(
                            step, model, request, task, wmem,
                        ),
                        timeout=timeout,
                    )
                elif is_chain:
                    # Tool-heavy → chain executor with wave parallelism
                    progress_queue: asyncio.Queue[InternalStreamChunk | None] = asyncio.Queue()

                    async def _run_chain(_s=step, _q=progress_queue, _idx=step_idx):
                        try:
                            return await self._execute_chain_step(
                                _s, ctx, model, request, task, wmem, _q,
                                step_idx=_idx,
                            )
                        finally:
                            await _q.put(None)

                    step_task = asyncio.create_task(_run_chain())

                    while True:
                        chunk = await asyncio.wait_for(progress_queue.get(), timeout=timeout)
                        if chunk is None:
                            break
                        yield chunk

                    step_output = await asyncio.wait_for(step_task, timeout=10.0)
                else:
                    # Generative → direct LLM call with working memory
                    step_output = await asyncio.wait_for(
                        self._execute_generative_step(step, ctx, model, request, task, wmem),
                        timeout=timeout,
                    )

            except TimeoutError:
                log.error("agentic_step_timeout", step=step.name, task_id=task.id,
                          timeout=settings.agentic_step_timeout)
                task.status = TaskStatus.FAILED
                task.error = f"Step '{step.name}' timed out after {settings.agentic_step_timeout}s"
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "failed", task)
                return
            except Exception as e:
                log.error("agentic_step_failed", step=step.name, task_id=task.id,
                          error=str(e), exc_info=True)
                task.status = TaskStatus.FAILED
                task.error = f"Step '{step.name}' failed: {e}"
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "failed", task)
                return

            # Record output
            ctx.record_step(step.name, step_output)
            task.record_step_output(step_idx, step_output)
            task.plan_md = update_plan_step(task.plan_md, step_idx)

            # Review → Revision loop: if review finds issues, re-run the draft
            needs_revision = False
            structured_verdict = None
            if step.role == "review":
                # Prefer the structured verdict from submit_review when present.
                structured_verdict = wmem.get_review_verdict(step.name)
                if structured_verdict:
                    needs_revision = structured_verdict["verdict"] == "REVISE"
                else:
                    # Legacy fallback: regex over prose for models that didn't
                    # call submit_review or for flows still using text verdicts.
                    _verdict = re.search(
                        r"(?:VERDICT|verdict)\s*[:=]\s*\*{0,2}\s*(PASS|NEEDS_REVISION|FAIL|REVISE|REVISION)",
                        step_output,
                        re.IGNORECASE,
                    )
                    if _verdict:
                        v = _verdict.group(1).upper()
                        needs_revision = v in ("NEEDS_REVISION", "FAIL", "REVISE", "REVISION")
                    else:
                        _revision_signals = re.search(
                            r"(needs?\s+revision|should\s+be\s+revised|requires?\s+changes|not\s+ready|major\s+issues)",
                            step_output,
                            re.IGNORECASE,
                        )
                        if _revision_signals:
                            needs_revision = True
                        else:
                            log.warning("review_verdict_not_found", step=step.name, output_len=len(step_output))
            if step.role == "review" and needs_revision:
                draft_step = None
                draft_step_idx = None
                for prior_idx in range(step_idx):
                    prior = flow.steps[prior_idx]
                    if prior.enabled and prior.role == "draft":
                        draft_step = prior
                        draft_step_idx = prior_idx

                if draft_step and steps_executed < max_steps:
                    log.info("agentic_revision", review_step=step.name,
                             draft_step=draft_step.name, task_id=task.id)

                    yield InternalStreamChunk(
                        content_delta="",
                        model=model,
                        augmentum={
                            "mode": "agentic",
                            "task_id": task.id,
                            "revision": {
                                "review_step": step.name,
                                "draft_step": draft_step.name,
                            },
                        },
                    )

                    revision_system = resolve_variables(
                        draft_step.system_prompt, ctx,
                    ) if draft_step.system_prompt else ""
                    if structured_verdict and structured_verdict.get("issues"):
                        _issues_md = "\n".join(
                            f"- {i}" for i in structured_verdict["issues"]
                        )
                        _reasoning = structured_verdict.get("reasoning", "")
                        revision_system += (
                            "\n\n## REVISION REQUIRED\n"
                            f"{_reasoning}\n\nIssues to address:\n{_issues_md}\n\n"
                            "Address ALL issues. Output the COMPLETE revised version. "
                            "Do NOT add commentary about what you changed."
                        )
                    else:
                        revision_system += (
                            "\n\n## REVISION REQUIRED\n"
                            f"A review found these issues:\n{step_output}\n\n"
                            "Address ALL issues. Output the COMPLETE revised version. "
                            "Do NOT add commentary about what you changed."
                        )
                    prior_context = wmem.build_context_for_step(draft_step.name)
                    revision_user = build_user_message(
                        draft_step.user_template, ctx, "",
                    )
                    if prior_context:
                        revision_user = f"{prior_context}\n\n{revision_user}"

                    gen_limit = max(256, (draft_step.output_cap or 800) * 2)
                    revised = await self._call_llm(
                        model=model,
                        system_prompt=revision_system,
                        user_message=revision_user,
                        request=request,
                        max_tokens=gen_limit,
                    )

                    wmem.record_generative_output(draft_step.name, revised)
                    ctx.record_step(draft_step.name, revised)
                    task.record_step_output(draft_step_idx, revised)
                    steps_executed += 1

                    # --- Re-review + optional re-planning ---
                    # Re-run the review against the revised draft. If it still
                    # fails and we haven't hit the insertion cap, call the
                    # planner for a remediation step, execute it, and redraft
                    # once more. This gives the flow one structured chance to
                    # recover from a persistent issue before giving up.
                    if insertions_made < settings.agentic_max_insertions:
                        re_review_tools = self._resolve_step_tools(step)
                        re_tools_section = _build_tools_section(re_review_tools) if re_review_tools else ""
                        re_review_system = resolve_variables(
                            step.system_prompt, ctx, re_tools_section,
                        ) if step.system_prompt else ""
                        re_review_user = build_user_message(step.user_template, ctx, re_tools_section)
                        try:
                            re_msg = await self._call_llm_full(
                                model=model,
                                system_prompt=(re_review_system + "\n\n" + _SUBMIT_REVIEW_INSTRUCTION)
                                    if settings.agentic_native_tool_use else re_review_system,
                                user_message=re_review_user,
                                request=request,
                                max_tokens=max(256, (step.output_cap or 500) * 2),
                                tools=[_SUBMIT_REVIEW_SCHEMA] if settings.agentic_native_tool_use else None,
                            )
                        except Exception:
                            re_msg = None

                        re_verdict = None
                        if re_msg and re_msg.tool_calls:
                            for call in re_msg.tool_calls:
                                fn = call.get("function") or {}
                                if fn.get("name") == "submit_review":
                                    self._record_submit_review(step.name, fn, wmem)
                                    re_verdict = wmem.get_review_verdict(step.name)
                                    break

                        if re_verdict and re_verdict["verdict"] == "REVISE":
                            inserted_output = await self._run_replan_insertion(
                                issues=re_verdict.get("issues", []),
                                reasoning=re_verdict.get("reasoning", ""),
                                draft_step=draft_step,
                                ctx=ctx,
                                wmem=wmem,
                                model=model,
                                request=request,
                                task=task,
                            )
                            if inserted_output:
                                insertions_made += 1
                                log.info("agentic_replan_inserted",
                                         task_id=task.id, draft_step=draft_step.name,
                                         insertions_made=insertions_made)
                                # Redraft with the remediation context in wmem.
                                redraft_system = resolve_variables(
                                    draft_step.system_prompt, ctx,
                                ) if draft_step.system_prompt else ""
                                redraft_system += (
                                    "\n\n## REMEDIATION CONTEXT\n"
                                    f"{inserted_output}\n\n"
                                    "Use the remediation context above to produce a "
                                    "corrected draft that resolves the outstanding issues."
                                )
                                redraft_user = build_user_message(
                                    draft_step.user_template, ctx, "",
                                )
                                prior_ctx2 = wmem.build_context_for_step(draft_step.name)
                                if prior_ctx2:
                                    redraft_user = f"{prior_ctx2}\n\n{redraft_user}"
                                redraft = await self._call_llm(
                                    model=model,
                                    system_prompt=redraft_system,
                                    user_message=redraft_user,
                                    request=request,
                                    max_tokens=gen_limit,
                                )
                                wmem.record_generative_output(draft_step.name, redraft)
                                ctx.record_step(draft_step.name, redraft)
                                task.record_step_output(draft_step_idx, redraft)
                                steps_executed += 1

            # Emit step content to sidebar
            if step_output:
                yield InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum={
                        "mode": "agentic",
                        "task_id": task.id,
                        "phase": step.name,
                        "phase_content_delta": step_output,
                    },
                )

            # Checkpoint
            if self._task_store:
                await self._task_store.update(task)

            yield _flow_step_chunk(model, step.name, "complete", pipeline, task)

            # Stream to user if marked. For deliver-role steps, also emit
            # typed delivery deltas (artifact_card, citation) alongside the
            # prose. Frontends that understand the augmentum.delivery
            # payload can render progressive cards; older renderers ignore
            # it and use the prose content_delta as before.
            if step.stream_to_user and step_output:
                if step.role == "deliver":
                    for card in _build_artifact_cards(wmem):
                        yield InternalStreamChunk(
                            content_delta="",
                            model=model,
                            augmentum={
                                "mode": "agentic",
                                "task_id": task.id,
                                "delivery": {"kind": "artifact_card", "payload": card},
                            },
                        )
                yield InternalStreamChunk(
                    content_delta=step_output + "\n\n",
                    model=model,
                    augmentum={"mode": "agentic", "task_id": task.id},
                )
                if step.role == "deliver":
                    citations = _extract_citations(step_output, wmem)
                    for cit in citations:
                        yield InternalStreamChunk(
                            content_delta="",
                            model=model,
                            augmentum={
                                "mode": "agentic",
                                "task_id": task.id,
                                "delivery": {"kind": "citation", "payload": cit},
                            },
                        )

            # Inform mode
            if task.autonomy_level == 3 and step.role in ("create", "illustrate"):
                summary = step_output[:200] if step_output else "Completed"
                yield build_inform_chunk(model, task, step.name, summary)

        # Synthesize if no step streamed to user
        any_streamed = any(s.stream_to_user for s in flow.steps if s.enabled)
        if not any_streamed:
            log.info("agentic_synthesize_response", task_id=task.id)
            combined = wmem.to_synthesis_context()
            respond_system = (
                "You are a helpful assistant. The user asked a question and "
                "several analysis steps have already been completed. Use the "
                "analysis results below to provide a clear, comprehensive "
                "final answer to the user.\n\n"
                f"### Analysis Results\n{combined}"
            )
            respond_sp = self._inject_date(respond_system)
            user_system = _extract_user_system(request)
            if user_system:
                respond_sp = f"{user_system}\n\n{respond_sp}"
            req = InternalChatRequest(
                model=model,
                messages=[
                    Message(role="system", content=respond_sp),
                    Message(role="user", content=query),
                ],
                stream=True,
            )
            async for chunk in self._backend.chat_stream(req):
                if chunk.content_delta:
                    yield chunk

        task.status = TaskStatus.COMPLETED
        if self._task_store:
            await self._task_store.update(task)
        yield _agentic_meta_chunk(model, "completed", task)

    # ------------------------------------------------------------------
    # Step execution: chain (tool-heavy) vs generative
    # ------------------------------------------------------------------

    async def _execute_chain_step(
        self,
        step: FlowStep,
        ctx: StepContext,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
        wmem: WorkingMemory,
        progress_queue: asyncio.Queue[InternalStreamChunk | None] | None = None,
        step_idx: int = -1,
    ) -> str:
        """Execute a tool-heavy step via the chain executor."""
        from augmentum.tools.chain import execute_chain  # noqa: I001
        from augmentum.tools.chain import StepResult as ChainStepResult

        tools = self._resolve_step_tools(step)
        if not tools or not self._tool_registry:
            return await self._execute_generative_step(
                step, ctx, model, request, task, wmem,
            )

        chain_plan = await self._build_chain_plan(
            step, tools, ctx, model, request, wmem,
        )

        if not chain_plan or not chain_plan.steps:
            log.info("chain_plan_empty_fallback", step=step.name)
            return await self._execute_generative_step(
                step, ctx, model, request, task, wmem,
            )

        async def on_step_start(chain_step):
            if progress_queue:
                envelope = task.meta_envelope()
                envelope["chain_step"] = {
                    "id": chain_step.id,
                    "tool": chain_step.tool,
                    "status": "running",
                    "reason": chain_step.reason,
                }
                await progress_queue.put(InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum=envelope,
                ))

        async def on_step_done(result: ChainStepResult):
            task.tool_calls_made += 1
            if progress_queue:
                envelope = task.meta_envelope()
                envelope["chain_step"] = {
                    "id": result.step_id,
                    "tool": result.tool_name,
                    "status": "done" if result.success else "failed",
                    "preview": result.output[:200],
                }
                await progress_queue.put(InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum=envelope,
                ))
                payload = _artifact_delivery_payload(result.metadata or {})
                if result.success and payload:
                    # Visuals produced mid-build (slide images, illustrations)
                    # are inputs to a later assembly step, not deliverables —
                    # flag them so the frontend keeps them out of the chat
                    # transcript and renders them compactly in the inspector.
                    if result.tool_name in _INTERMEDIATE_VISUAL_TOOLS:
                        payload["intermediate"] = True
                    delivery_envelope = task.meta_envelope()
                    delivery_envelope["delivery"] = {
                        "kind": "artifact_card",
                        "payload": payload,
                    }
                    await progress_queue.put(InternalStreamChunk(
                        content_delta="",
                        model=model,
                        augmentum=delivery_envelope,
                    ))

        async def on_replan(step_id: int, decision: str):
            log.info("agentic_chain_replan", step_id=step_id, decision=decision,
                     task_id=task.id)
            if progress_queue:
                envelope = task.meta_envelope()
                envelope["chain_replan"] = {"step_id": step_id, "decision": decision}
                await progress_queue.put(InternalStreamChunk(
                    content_delta="",
                    model=model,
                    augmentum=envelope,
                ))

        # Bind a progress callback for the duration of this chain step so
        # tools (e.g. ``create_ebook`` during illustration) can stream
        # per-event payloads (chapter prompts, image thumbnails, planner
        # output) without threading a queue reference through every layer.
        # Each payload arrives merged into the meta envelope under whatever
        # key the tool chose, and the inspector renders it via the
        # per-flow renderer.
        from augmentum.modes.agentic.progress_bus import bind_callback, reset_binding

        async def _emit_tool_progress(payload: dict) -> None:
            if not progress_queue:
                return
            envelope = task.meta_envelope()
            # Merge top-level keys directly so the renderer can dispatch
            # by event-type key (chapter_illustration, ebook_plan, …)
            # without unwrapping a nested object.
            envelope.update(payload)
            await progress_queue.put(InternalStreamChunk(
                content_delta="",
                model=model,
                augmentum=envelope,
            ))

        build_store = self._build_run_store
        build_id = ""
        build_run_created = False
        build_status_state: dict = {}
        progress_callback = _emit_tool_progress
        if build_store and self._user_id and any(s.tool == "build_application" for s in chain_plan.steps):
            try:
                created = await build_store.create(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    task_id=task.id,
                    kind="application",
                    status="running",
                    name=task.title or step.name,
                    request={
                        "description": task.original_query or _extract_query(request),
                        "agentic_step": step.name,
                    },
                )
                build_id = created.get("id", "")
                build_run_created = bool(build_id)
                build_status_state = {
                    "id": build_id,
                    "kind": "application",
                    "user_id": self._user_id,
                    "session_id": self._session_id,
                    "task_id": task.id,
                    "name": task.title or step.name,
                    "status": "running",
                    "passes": [],
                    "error": None,
                    "project": None,
                }
            except Exception:
                log.warning("agentic_build_run_create_failed", task_id=task.id, exc_info=True)

            async def _emit_tool_progress_with_build(payload: dict) -> None:
                progress = payload.get("project_progress") if isinstance(payload, dict) else None
                if build_run_created and isinstance(progress, dict):
                    progress["build_id"] = build_id
                    progress["task_id"] = task.id
                    try:
                        from augmentum.builds.runtime import (
                            apply_project_progress,
                            progress_payload_from_state,
                        )
                        apply_project_progress(build_status_state, progress)
                        await build_store.update(
                            build_id,
                            user_id=self._user_id,
                            status="running",
                            name=build_status_state.get("name") or task.title or step.name,
                            progress=progress_payload_from_state(build_status_state),
                        )
                    except Exception:
                        log.warning("agentic_build_run_progress_failed", build_id=build_id, exc_info=True)
                await _emit_tool_progress(payload)

            progress_callback = _emit_tool_progress_with_build

        heartbeat_task = None
        if build_run_created and build_store and self._user_id:
            try:
                from augmentum.builds.runtime import heartbeat_build_run
                heartbeat_task = asyncio.create_task(
                    heartbeat_build_run(
                        build_store,
                        build_id=build_id,
                        user_id=self._user_id,
                        state=build_status_state,
                    )
                )
            except Exception:
                log.warning("agentic_build_run_heartbeat_start_failed", build_id=build_id, exc_info=True)

        # Constrain replan/substitution to the flow's curated tool set.
        # Without this, a failed step (e.g. arg-resolution truncation on
        # create_ebook) could be replaced by ANY tool in the global
        # registry — that's how Storybook ended up generating a spurious
        # create_document PDF when the LLM-driven arg JSON failed to
        # parse. Restricting the substitute candidates to the names this
        # step was authorized to use means the worst-case recovery is
        # retry/skip/abort, not a different artifact type being silently
        # inserted into the user's chat.
        allowed_tools: set[str] | None = (
            set(step.tool_names) if step.tool_names else None
        )

        progress_token = bind_callback(progress_callback)
        try:
            results = await execute_chain(
                chain_plan,
                self._backend,
                self._tool_registry,
                request_context=request,
                on_step_start=on_step_start,
                on_step_done=on_step_done,
                on_replan=on_replan,
                max_steps=settings.agentic_max_steps,
                extra_tool_args={
                    "task_id": task.id,
                    "_task_id": task.id,
                    "session_id": self._session_id,
                    "_session_id": self._session_id,
                    "_request_model": model,
                    "_build_id": build_id,
                    "_progress_callback": progress_callback,
                },
                tool_cache=self._tool_call_cache,
                cache_task_id=task.id,
                cache_user_id=self._user_id,
                cache_step_idx_base=step_idx,
                allowed_tool_names=allowed_tools,
            )
        except Exception as exc:
            if build_run_created:
                try:
                    await build_store.update(
                        build_id,
                        user_id=self._user_id,
                        status="failed",
                        error=str(exc),
                    )
                except Exception:
                    log.warning("agentic_build_run_failure_update_failed", build_id=build_id, exc_info=True)
            raise
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
            reset_binding(progress_token)

        # Record in working memory
        wmem.record_chain_results(step.name, results)

        # Capture artifact metadata
        for r in results.values():
            if r.success and r.metadata and (
                r.metadata.get("download_url") or r.metadata.get("id")
                or r.metadata.get("url") or r.metadata.get("image_id")
            ):
                wmem.record_artifact(r.metadata)

        if build_run_created:
            app_result = next((r for r in results.values() if r.tool_name == "build_application"), None)
            try:
                if app_result and app_result.success:
                    project = (app_result.metadata or {}).get("project") or {}
                    artifact_id = (app_result.metadata or {}).get("artifact_id") or project.get("artifactId") or ""
                    from augmentum.builds.runtime import progress_payload_from_state
                    # Honor the real terminal status from the coder builder — a
                    # budget/stuck stop is "paused" (resumable checkpoint), not
                    # "completed". Hardcoding "completed" here would erase the
                    # continue/stop gate on refresh.
                    final_status = project.get("status") or "completed"
                    build_status_state["status"] = final_status
                    build_status_state["project"] = project
                    build_status_state["artifact_id"] = artifact_id
                    build_status_state["qualityStatus"] = project.get("qualityStatus") or project.get("quality_status") or build_status_state.get("qualityStatus", "clean")
                    build_status_state["warnings"] = project.get("warnings") or build_status_state.get("warnings", [])
                    build_status_state["blockingErrors"] = project.get("blockingErrors") or project.get("blocking_errors") or build_status_state.get("blockingErrors", [])
                    await build_store.update(
                        build_id,
                        user_id=self._user_id,
                        status=final_status,
                        name=project.get("name") or task.title or step.name,
                        artifact_id=artifact_id,
                        result={
                            "artifact_id": artifact_id,
                            "project": project,
                            "output": app_result.output,
                        },
                        progress=progress_payload_from_state(build_status_state),
                    )
                elif app_result:
                    build_status_state["status"] = "failed"
                    build_status_state["error"] = app_result.output or "Build failed"
                    await build_store.update(
                        build_id,
                        user_id=self._user_id,
                        status="failed",
                        error=app_result.output or "Build failed",
                    )
            except Exception:
                log.warning("agentic_build_run_finish_failed", build_id=build_id, exc_info=True)

        # Build summary for checkpoint storage
        parts = []
        for s in chain_plan.steps:
            r = results.get(s.id)
            if r:
                status = "OK" if r.success else "FAILED"
                parts.append(f"[{r.tool_name} ({status})]: {r.output[:500]}")
        return "\n".join(parts)

    async def _build_chain_plan(
        self,
        step: FlowStep,
        tools: list,
        ctx: StepContext,
        model: str,
        request: InternalChatRequest,
        wmem: WorkingMemory,
    ) -> ChainPlan | None:
        """Build a ChainPlan for a tool-heavy step via the LLM planner."""
        from augmentum.tools.chain import ChainPlan, ChainStep, ToolChainPlanner

        step_objective = step.name
        if step.user_template:
            step_objective = resolve_variables(step.user_template, ctx)
        elif step.system_prompt:
            step_objective = resolve_variables(step.system_prompt, ctx)

        # Inject working memory summaries + task-level plan as attention anchor.
        # Artifact-creation steps (role=create) get the full draft + image URLs
        # so the chain planner can construct complete tool calls.
        if step.role == "create":
            prior_context = wmem.format_for_create_context()
        else:
            prior_context = wmem.format_for_plan_context()
        if prior_context:
            step_objective = f"{step_objective}\n\n{prior_context}"

        task_plan = plan_to_context(ctx.plan)
        if task_plan:
            step_objective += task_plan

        # Inject artifact template context when step involves document/presentation creation
        if step.tool_names:
            try:
                from augmentum.tools.artifact_templates import get_template_for_tool_call
                user_msg = ctx._step_outputs.get("_user_message", "") or step.name
                for tool_name in step.tool_names:
                    tpl_ctx = get_template_for_tool_call(tool_name, user_msg)
                    if tpl_ctx:
                        step_objective += f"\n\n## Design Template\n{tpl_ctx}"
                        break  # one template is enough
            except Exception as exc:
                log.debug("agentic_chain_template_inject_failed", error=str(exc))

        focused_request = InternalChatRequest(
            model=model,
            messages=[Message(role="user", content=step_objective)],
            stream=False,
        )

        planner = ToolChainPlanner(self._backend, self._tool_registry)
        plan_messages = planner._build_plan_prompt(focused_request, tools)

        try:
            plan = await planner._get_plan(model, plan_messages)
            if plan:
                log.info(
                    "agentic_chain_plan_built",
                    step=step.name,
                    chain_steps=len(plan.steps),
                    tools=[s.tool for s in plan.steps],
                )
                return plan
        except Exception:
            log.warning("agentic_chain_plan_failed", step=step.name, exc_info=True)

        # Fallback: one chain step per tool in tool_names
        fallback_steps = []
        for i, tool_name in enumerate(step.tool_names):
            resolved = self._tool_registry.resolve(tool_name) if self._tool_registry else None
            if resolved:
                fallback_steps.append(ChainStep(
                    id=i + 1,
                    tool=resolved.name,
                    needs=[i] if i > 0 else [],
                    reason=f"Use {resolved.name} for: {step.name}",
                ))
        if not fallback_steps and tools:
            fallback_steps.append(ChainStep(
                id=1,
                tool=tools[0].name,
                reason=step.name,
            ))
        return ChainPlan(steps=fallback_steps, source=f"agentic:{step.name}") if fallback_steps else None

    async def _execute_generative_step(
        self,
        step: FlowStep,
        ctx: StepContext,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
        wmem: WorkingMemory,
    ) -> str:
        """Execute a pure LLM step (draft, review, deliver) with working memory context.

        If the step declares tools and the LLM outputs ``TOOL_CALL:`` patterns,
        the tools are executed and results fed back for a second LLM pass.
        This allows generative steps to use supplementary tools (calculator,
        web_search, etc.) while keeping prose generation as the primary output.
        """
        tools = self._resolve_step_tools(step)
        tools_section = _build_tools_section(tools) if tools else ""

        system_prompt = resolve_variables(
            step.system_prompt, ctx, tools_section,
        ) if step.system_prompt else ""
        user_message = build_user_message(step.user_template, ctx, tools_section)

        # Inject working memory context
        prior_context = wmem.build_context_for_step(step.name)
        if prior_context:
            user_message = f"{prior_context}\n\n{user_message}"

        plan_context = plan_to_context(ctx.plan)
        if plan_context:
            user_message += plan_context

        # Derive token limit from output_cap
        gen_limit = None
        if step.output_cap and step.output_cap > 0:
            gen_limit = max(256, step.output_cap * 2)

        # --- Native tool_use path (preferred) ---
        # When tools are declared and the backend supports structured tool_calls,
        # pass them via InternalChatRequest.tools and consume response.tool_calls.
        # This is dramatically more reliable than TOOL_CALL: text patterns,
        # especially on small models. Falls back to text-pattern path when the
        # backend returns no structured tool_calls.
        output = ""
        tool_results: list[tuple[str, str]] = []
        submit_review_seen = False
        # Review steps get a structured verdict channel via a synthetic
        # submit_review tool. The tool is virtual — we intercept the call
        # instead of executing it.
        use_submit_review = step.role == "review" and settings.agentic_native_tool_use
        native_capable = (tools and settings.agentic_native_tool_use) or use_submit_review
        if native_capable:
            from augmentum.modes.analytical.tool_calling import tools_to_native_format
            native_tools = tools_to_native_format(tools) if tools else []
            if use_submit_review:
                native_tools.append(_SUBMIT_REVIEW_SCHEMA)
                system_prompt = (
                    (system_prompt + "\n\n") if system_prompt else ""
                ) + _SUBMIT_REVIEW_INSTRUCTION
            msg = await self._call_llm_full(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
                request=request,
                max_tokens=gen_limit,
                tools=native_tools or None,
            )
            output = msg.content or ""
            if msg.tool_calls:
                non_review_calls = []
                for call in msg.tool_calls:
                    fn = call.get("function") or {}
                    if use_submit_review and fn.get("name") == "submit_review":
                        self._record_submit_review(step.name, fn, wmem)
                        submit_review_seen = True
                        continue
                    non_review_calls.append(call)
                if non_review_calls and tools:
                    tool_results = await self._dispatch_structured_tool_calls(non_review_calls, tools)
        if not output and not tool_results and not submit_review_seen:
            output = await self._call_llm(
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
                request=request,
                max_tokens=gen_limit,
            )

        # --- Text-pattern fallback ---
        # If tools were declared but native tool_calls weren't emitted, parse
        # TOOL_CALL: patterns from the text. Kept for small/older models that
        # don't reliably emit structured tool_calls.
        if tools and not tool_results and "TOOL_CALL:" in output:
            tool_results = await self._execute_inline_tool_calls(output, tools)

        if tool_results:
            # Re-call LLM with tool results appended so the final output
            # incorporates real tool data.
            results_block = "\n\n".join(
                f"## Tool Result ({name})\n{result}"
                for name, result in tool_results
            )
            user_with_results = (
                f"{user_message}\n\n"
                f"{results_block}\n\n"
                "Incorporate the tool results above into your response. "
                "Write the final content directly — do not emit further tool calls."
            )
            output = await self._call_llm(
                model=model,
                system_prompt=system_prompt,
                user_message=user_with_results,
                request=request,
                max_tokens=gen_limit,
            )

        wmem.record_generative_output(step.name, output)

        if step.output_cap and len(output) > step.output_cap * 4:
            output = output[: step.output_cap * 4]

        return output

    async def _run_replan_insertion(
        self,
        *,
        issues: list[str],
        reasoning: str,
        draft_step: FlowStep,
        ctx: StepContext,
        wmem: WorkingMemory,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
    ) -> str:
        """Synthesize a remediation step when review keeps failing.

        Rather than mutate flow.steps (which would break ongoing iteration
        and cached flows), we run the remediation inline as a one-shot
        generative call. Its output is recorded in wmem under a synthetic
        step name so it feeds the redraft via the normal prior-context path.
        """
        if not issues and not reasoning:
            return ""
        bullets = "\n".join(f"- {i}" for i in issues) if issues else "(no issue list supplied)"
        remediation_system = (
            "You are a planning aide. The prior draft failed review twice. "
            "Produce a concise remediation brief the next draft can act on. "
            "Do not rewrite the draft — produce guidance: what is missing, "
            "what evidence must be cited, what sections must be added or removed."
        )
        remediation_user = (
            f"## Review reasoning\n{reasoning}\n\n"
            f"## Outstanding issues\n{bullets}\n\n"
            f"## User request\n{ctx.query}\n\n"
            "Write a 5-10 bullet remediation brief."
        )
        try:
            output = await self._call_llm(
                model=model,
                system_prompt=remediation_system,
                user_message=remediation_user,
                request=request,
                max_tokens=512,
            )
        except Exception:
            log.warning("agentic_replan_llm_failed", task_id=task.id, exc_info=True)
            return ""
        synthetic_name = f"Remediation for {draft_step.name}"
        wmem.record_generative_output(synthetic_name, output)
        return output

    def _record_submit_review(self, step_name: str, fn: dict, wmem: WorkingMemory) -> None:
        """Intercept a submit_review tool call and stash its structured payload."""
        import json as _json
        args_raw = fn.get("arguments", {})
        if isinstance(args_raw, str):
            try:
                args = _json.loads(args_raw) if args_raw else {}
            except _json.JSONDecodeError:
                args = {}
        else:
            args = args_raw or {}
        verdict = str(args.get("verdict", "")).strip().upper() or "PASS"
        reasoning = str(args.get("reasoning", "") or "")
        issues_raw = args.get("issues") or []
        issues = [str(i) for i in issues_raw if str(i).strip()]
        wmem.record_review_verdict(step_name, verdict, reasoning, issues)
        log.info("review_verdict_structured", step=step_name, verdict=verdict,
                 issue_count=len(issues))

    def _inject_tool_context(self, tool, args: dict) -> None:
        """Add ``_context`` (user_id + session_id) to ``args`` when the tool
        accepts it. Tools like ``create_ebook`` and ``image_generation``
        persist DB rows keyed by these fields, and without them the writes
        hit FOREIGN KEY failures. The passthrough handler injects the same
        dict via its own tool executor — this keeps agentic-mode parity so
        every tool-call path sees the same context.
        """
        if not self._user_id and not self._session_id:
            return
        import inspect
        try:
            sig = inspect.signature(tool.execute)
        except (TypeError, ValueError):
            return
        accepts_context = (
            "_context" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
        if not accepts_context:
            return
        ctx = dict(args.get("_context") or {})
        if self._user_id and not ctx.get("user_id"):
            ctx["user_id"] = self._user_id
        if self._session_id and not ctx.get("session_id"):
            ctx["session_id"] = self._session_id
        # Mode stamp for the offer substrate's allowed_modes gate.
        ctx.setdefault("mode", "agentic")
        args["_context"] = ctx

    async def _dispatch_structured_tool_calls(
        self,
        tool_calls: list[dict],
        tools: list,
    ) -> list[tuple[str, str]]:
        """Execute structured tool_calls emitted by the LLM (native tool_use).

        Each call is ``{"function": {"name": str, "arguments": str|dict}, ...}``
        (OpenAI/Ollama format). Returns ``(tool_name, result_text)`` tuples.
        Capped at 3 to match the text-pattern path.
        """
        import json as _json

        results: list[tuple[str, str]] = []
        tool_map = {t.name: t for t in tools}

        for call in tool_calls[:3]:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    args = _json.loads(args_raw) if args_raw else {}
                except _json.JSONDecodeError:
                    results.append((name, "Error: Invalid JSON arguments"))
                    continue
            else:
                args = args_raw or {}

            tool = tool_map.get(name)
            if not tool and self._tool_registry:
                tool = self._tool_registry.resolve(name)
            if not tool:
                results.append((name, f"Error: Unknown tool '{name}'"))
                continue

            try:
                from augmentum.modes.analytical.tool_calling import coerce_tool_params
                args = coerce_tool_params(tool, args)
                self._inject_tool_context(tool, args)
                result = await asyncio.wait_for(
                    tool.execute(**args),
                    timeout=min(tool.timeout, 30.0),
                )
                if result.success:
                    results.append((tool.name, result.output))
                else:
                    results.append((tool.name, f"Error: {tool.enrich_error(result.error, args)}"))
            except Exception as exc:
                results.append((name, f"Error: {exc}"))

            log.info("generative_native_tool", step_tool=name,
                     success=bool(results[-1][1] and not results[-1][1].startswith("Error:")))

        return results

    async def _execute_inline_tool_calls(
        self,
        output: str,
        tools: list,
    ) -> list[tuple[str, str]]:
        """Parse and execute TOOL_CALL: patterns from generative step output.

        Returns list of ``(tool_name, result_text)`` tuples.
        Limited to 3 calls to prevent runaway execution.
        """
        import json as _json
        import re as _re

        results: list[tuple[str, str]] = []
        tool_map = {t.name: t for t in tools}

        # Find each `TOOL_CALL: name` header then walk the following braces
        # with a depth counter — a single regex can't match balanced JSON and
        # the previous `\{[^}]*\}` silently truncated any nested arg (e.g.
        # `{"items": [{"id": 1}]}` was cut to `{"items": [{"id": 1}`).
        header_re = _re.compile(r"TOOL_CALL:\s*(\w+)\s*\n\s*(?=\{)")
        matches: list[tuple[str, str]] = []
        for m in header_re.finditer(output):
            name = m.group(1)
            start = m.end()
            depth = 0
            in_str = False
            escape = False
            end = -1
            for i in range(start, len(output)):
                ch = output[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                matches.append((name, output[start:end]))
            if len(matches) >= 3:
                break

        for name, args_str in matches[:3]:  # cap at 3
            tool = tool_map.get(name)
            if not tool and self._tool_registry:
                # Try fuzzy resolution
                tool = self._tool_registry.resolve(name)
            if not tool:
                results.append((name, f"Error: Unknown tool '{name}'"))
                continue

            try:
                args = _json.loads(args_str)
            except _json.JSONDecodeError:
                results.append((name, "Error: Invalid JSON arguments"))
                continue

            try:
                from augmentum.modes.analytical.tool_calling import coerce_tool_params
                args = coerce_tool_params(tool, args)
                self._inject_tool_context(tool, args)
                result = await asyncio.wait_for(
                    tool.execute(**args),
                    timeout=min(tool.timeout, 30.0),  # cap for inline calls
                )
                if result.success:
                    results.append((tool.name, result.output))
                else:
                    results.append((tool.name, f"Error: {tool.enrich_error(result.error, args)}"))
            except Exception as exc:
                results.append((name, f"Error: {exc}"))

            log.info("generative_inline_tool", step_tool=name, success=bool(results[-1][1] and not results[-1][1].startswith("Error:")))

        return results

    async def _execute_illustrate_step(
        self,
        step: FlowStep,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
        wmem: WorkingMemory,
    ) -> str:
        """Build a per-unit image candidate pool for the artifact.

        Parses the draft into illustratable units (slides for a deck,
        ``## SECTION:`` blocks for a document), runs the two-pass query
        crafter, and for each unit gathers candidates from BOTH sources the
        step allows:

        - ``image_search`` → real photographs (``kind="photo"``)
        - ``image_generation`` → a synthetic image from a *capability-matched*
          model (``kind="generated"``), only when the step lists
          image_generation AND a suitable model is installed.

        The primary pick per unit follows the topic FORMAT: a physical
        ``procedure`` (changing a tire, cooking) leads with a real photo; a
        ``code`` / technical topic leads with a generated diagram. Generation
        NEVER falls back to a stylised default — an anime checkpoint is never
        used to draw a real-world how-to, which is the failure this fixes.
        The full pool is persisted so the picker + Create step can swap among
        candidates. Presentations list only image_search, so they take the
        photo-only path unchanged.
        """
        import uuid as _uuid

        from augmentum.tools.artifact_pipeline import (
            build_agentic_pipeline_caller,
            craft_initial_slide_queries,
        )

        # 1. Pull the draft and parse it into units. Use the SAME parser the
        # matching Create step uses so unit indices line up with the pool.
        draft = _pick_artifact_draft(wmem)
        if not draft:
            log.warning("illustrate_step_no_draft", task_id=task.id)
            return "No draft available — illustrate skipped."

        fallback = _resolve_artifact_topic(task) or "Content"
        if _SLIDE_MARKER_RE.search(draft):
            unit_kind = "slide"
            parsed, _warn = _parse_slide_draft(draft, fallback_title=fallback)
            units = [{"title": s.get("title", ""), "body": s.get("body", "")}
                     for s in parsed]
        else:
            unit_kind = "section"
            parsed = _parse_document_sections(draft, fallback_title=fallback)
            units = [{"title": s.get("heading", ""), "body": s.get("body", "")}
                     for s in parsed]
        if not units:
            return "No units parsed from draft — illustrate skipped."

        # 2. Two-pass query crafter over the units (slide-shaped input).
        caller = build_agentic_pipeline_caller(self._call_llm, model, request)
        crafted = await craft_initial_slide_queries(units, caller)
        if not crafted:
            log.warning("illustrate_step_crafter_failed", task_id=task.id,
                        units=len(units))
            return (
                f"Query crafter returned no usable output for "
                f"{len(units)} {unit_kind}s — they will ship text-only."
            )

        # 3. Resolve tools. image_search is the photo source; image_generation
        # is added only when the step lists it (tutorials do, decks don't).
        reg = self._tool_registry
        image_search = reg.resolve("image_search") if reg else None
        if not image_search:
            log.warning("illustrate_step_no_image_search_tool", task_id=task.id)
            return "image_search tool not registered — units will ship text-only."

        fmt = _detect_plan_format(task, wmem)
        gen_allowed = bool(reg) and "image_generation" in (step.tool_names or [])
        gen_model = ""
        if gen_allowed:
            img_gen = reg.resolve("image_generation")
            if img_gen and hasattr(img_gen, "select_model_for"):
                # code topic → diagram-capable; everything else → photoreal.
                if fmt == "code":
                    gen_model = await img_gen.select_model_for(need_diagram=True)
                else:
                    gen_model = await img_gen.select_model_for(need_photoreal=True)
            gen_allowed = bool(img_gen) and bool(gen_model)
            if "image_generation" in (step.tool_names or []) and not gen_allowed:
                log.info("illustrate_step_gen_skipped", task_id=task.id,
                         reason="no_capable_model", fmt=fmt or "unknown")
        gen_is_primary = gen_allowed and fmt == "code"

        # 4. Gather candidates per unit from both sources.
        candidates_by_unit: dict[int, list[dict]] = {}
        picks_by_unit: dict[int, dict] = {}
        units_with_query = 0
        units_with_results = 0
        generated_count = 0

        for unit_data in crafted:
            idx = unit_data["index"]
            query = unit_data.get("query", "")
            if not query:
                continue
            units_with_query += 1
            prefer_charts = bool(unit_data.get("prefer_charts"))
            description = unit_data.get("description", "")

            photo_pool: list[dict] = []
            try:
                result = await image_search.execute(
                    query=query, count=4, prefer_charts=prefer_charts,
                    task_id=task.id, session_id=self._session_id,
                    user_id=self._user_id,
                )
            except Exception as exc:
                log.warning("illustrate_step_image_search_failed",
                            unit_index=idx, query=query, error=str(exc))
                result = None
            if result and result.success and result.metadata:
                for img in (result.metadata.get("images") or []):
                    if not isinstance(img, dict):
                        continue
                    embed_url = img.get("embed_url") or img.get("url") or ""
                    if not embed_url:
                        continue
                    photo_pool.append({
                        "candidate_id": _uuid.uuid4().hex[:12],
                        "kind": "photo",
                        "query": query,
                        "description": description,
                        "prefer_charts": prefer_charts,
                        "embed_url": embed_url,
                        "thumb_url": img.get("thumb_url") or embed_url,
                        "source": img.get("source", ""),
                        "title": img.get("title", ""),
                    })

            gen_pool: list[dict] = []
            if gen_allowed and generated_count < _ILLUSTRATE_MAX_GENERATED:
                gen_cand = await self._generate_illustration_candidate(
                    img_gen, gen_model, description or query or units[idx - 1].get("title", ""),
                    fmt, task, _uuid,
                )
                if gen_cand:
                    gen_pool.append(gen_cand)
                    generated_count += 1

            # Order the pool so the FORMAT-appropriate source is primary.
            pool = (gen_pool + photo_pool) if gen_is_primary else (photo_pool + gen_pool)
            if pool:
                candidates_by_unit[idx] = pool
                picks_by_unit[idx] = {"primary": pool[0]["candidate_id"], "additional": []}
                units_with_results += 1

        # 5. Persist for the picker / resume, and stash on wmem for the
        # downstream Create step's _apply_image_picks.
        task.image_candidates = candidates_by_unit
        task.slide_image_picks = picks_by_unit
        if self._task_store:
            await self._task_store.update_image_candidates(
                task.id, candidates_by_unit, user_id=self._user_id,
            )
            await self._task_store.update_slide_image_picks(
                task.id, picks_by_unit, user_id=self._user_id,
            )
        wmem._slide_image_picks = picks_by_unit
        wmem._image_candidates = candidates_by_unit

        # 6. Summary for the working-memory transcript.
        lines = [
            f"Illustrated {units_with_results} of {units_with_query} "
            f"image-eligible {unit_kind}s (of {len(units)} total); "
            f"{generated_count} generated, "
            f"{'diagram' if fmt == 'code' else 'photo'}-primary."
        ]
        for idx in sorted(candidates_by_unit.keys()):
            pool = candidates_by_unit[idx]
            primary = pool[0]
            kinds = "+".join(sorted({c.get("kind", "photo") for c in pool}))
            lines.append(
                f"  {unit_kind.title()} {idx}: {len(pool)} candidates ({kinds}); "
                f"primary {primary.get('kind', 'photo')} "
                f"from {primary.get('source') or gen_model or 'search'}"
            )
        return "\n".join(lines)

    async def _generate_illustration_candidate(
        self, img_gen, gen_model: str, subject: str, fmt: str, task, _uuid,
    ) -> dict | None:
        """Generate one synthetic illustration candidate, or None on failure.

        Wraps the crafted subject in a FORMAT-appropriate prompt and forces
        the capability-matched ``gen_model`` (never the user's stylised
        default).
        """
        subject = (subject or "").strip()
        if not subject:
            return None
        if fmt == "code":
            prompt = (
                f"Clean flat technical diagram: {subject}. Labeled components, "
                "minimal, high-contrast, instructional vector style, no watermark."
            )
        else:
            prompt = (
                f"Realistic instructional photograph: {subject}. Clear, well-lit, "
                "true-to-life, sharp focus, no text overlay, no watermark."
            )
        try:
            args = {
                "prompt": prompt,
                "aspect": "landscape",
                "model": gen_model,
                "task_id": task.id,
                "session_id": self._session_id,
                "user_id": self._user_id,
            }
            self._inject_tool_context(img_gen, args)
            res = await img_gen.execute(**args)
        except Exception as exc:
            log.warning("illustrate_step_generate_failed",
                        task_id=task.id, model=gen_model, error=str(exc))
            return None
        if not res or not res.success or not res.metadata:
            return None
        url = res.metadata.get("url") or ""
        if not url:
            return None
        return {
            "candidate_id": _uuid.uuid4().hex[:12],
            "kind": "generated",
            "query": subject,
            "description": prompt,
            "prefer_charts": False,
            "embed_url": url,
            "thumb_url": url,
            "source": gen_model,
            "title": "Generated illustration",
        }

    async def _execute_artifact_pipeline_step(
        self,
        step: FlowStep,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
        wmem: WorkingMemory,
    ) -> str:
        """Execute an artifact creation step via the unified pipeline.

        Determines format from tool names, builds context from working
        memory and message history, and delegates to execute_artifact_pipeline.
        Falls back to the legacy create methods on failure.
        """
        from augmentum.tools.artifact_pipeline import (
            ArtifactRequest,
            PipelineContext,
            build_agentic_pipeline_caller,
            execute_artifact_pipeline,
        )

        try:
            # Determine format from tool names
            fmt = "pdf"
            for tn in (step.tool_names or []):
                if tn == "create_presentation":
                    fmt = "pptx"
                elif tn == "create_spreadsheet":
                    fmt = "xlsx"
                elif tn == "create_chart":
                    fmt = "chart"
                elif tn == "create_document":
                    fmt = "pdf"

            # task.title comes from the plan step's "## Task: <title>" line,
            # which models often abbreviate to a single meta-word ("Plan",
            # "Task", "Document"). That makes a useless topic for the
            # drafter. Prefer the user's original query as the topic, and
            # keep task.title for the artifact's display title.
            topic = _resolve_artifact_topic(task)
            art_req = ArtifactRequest(
                format=fmt,
                topic=topic,
                title=task.title or topic,
                tool_params={},
            )

            # Build context from working memory and message history
            msg_history = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ] if request.messages else []

            ctx = PipelineContext(
                message_history=msg_history,
                working_memory=wmem,
                generated_images=[],
                user_id=self._user_id,
            )

            caller = build_agentic_pipeline_caller(self._call_llm, model, request)

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

            # Surface Illustrate Slides picks to the pipeline so it skips
            # re-illustration and uses our candidates instead. The pipeline
            # reads ``ctx.slide_image_picks`` / ``ctx.image_candidates``
            # when present and applies them via the same shape as
            # _apply_image_picks above.
            picks = getattr(wmem, "_slide_image_picks", None) or task.slide_image_picks
            candidates = getattr(wmem, "_image_candidates", None) or task.image_candidates
            if picks and candidates:
                ctx.slide_image_picks = picks
                ctx.image_candidates = candidates

            result = await execute_artifact_pipeline(
                art_req, ctx, caller,
                _search_tool=search_tool,
                _fetch_tool=fetch_tool,
                _render_tools=render_tools,
            )

            if result.metadata:
                wmem.record_artifact(result.metadata)

            output = (
                f"Created {result.display_name}: {result.download_url}"
                if result.download_url
                else f"Created artifact: {result.artifact_id}"
            )
            return output

        except Exception as exc:
            log.warning("artifact_pipeline_step_fallthrough",
                        step=step.name, error=str(exc))
            # Fall back to legacy create methods
            if "create_document" in (step.tool_names or []):
                return await self._execute_create_document_step(
                    step, model, request, task, wmem,
                )
            elif "create_presentation" in (step.tool_names or []):
                return await self._execute_create_presentation_step(
                    step, model, request, task, wmem,
                )
            return f"Error: Artifact pipeline failed: {exc}"

    async def _execute_create_document_step(
        self,
        step: FlowStep,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
        wmem: WorkingMemory,
    ) -> str:
        """Create a document by parsing section markers from the draft output.

        Instead of asking the LLM to restructure the draft into tool args
        (which loses content via summarization), this reads the draft from
        working memory, parses ``## SECTION: <heading>`` markers, and calls
        create_document directly with the parsed sections.
        """
        if not self._tool_registry:
            return "Error: tool registry not available"

        tool = self._tool_registry.resolve("create_document")
        if not tool:
            return "Error: create_document tool not found"

        draft = _pick_artifact_draft(wmem)
        if not draft:
            return "Error: no draft content found in working memory"

        # Parse the draft into sections via the shared parser so section
        # indices line up with the Illustrate loop's pick pool.
        sections = _parse_document_sections(
            draft, fallback_title=task.title or "Content",
        )

        # Project Illustrate picks (real-photo / generated primary per
        # section) onto each section's image_url before the renderer runs.
        sections = _apply_image_picks(sections, wmem, task)

        # Pick format from the user's original ask when possible —
        # "make me a Word doc" → docx, "PDF report" → pdf. Falls back
        # to pdf when the user's phrasing is neutral, matching the
        # tool's own default.
        from augmentum.tools.artifact_document import infer_document_format
        inferred = infer_document_format(task.original_query or task.title or "")
        chosen_format = inferred or "pdf"
        if inferred:
            log.info(
                "create_document_format_inferred",
                format=inferred,
                task_id=task.id,
            )

        # Call the tool directly
        try:
            doc_args = {
                "title": task.title or "Document",
                "format": chosen_format,
                "sections": sections,
                "task_id": task.id,
                "session_id": self._session_id,
                "theme": "",
            }
            self._inject_tool_context(tool, doc_args)
            result = await tool.execute(**doc_args)
            if result.success and result.metadata:
                meta = dict(result.metadata)
                if isinstance(getattr(result, "card", None), dict):
                    meta["card"] = result.card
                wmem.record_artifact(meta)
            return result.output if result.success else f"Error: {result.error}"
        except Exception as e:
            log.error("create_document_direct_failed", error=str(e), exc_info=True)
            return f"Error: Document creation failed: {e}"

    async def _execute_create_presentation_step(
        self,
        step: FlowStep,
        model: str,
        request: InternalChatRequest,
        task: TaskState,
        wmem: WorkingMemory,
    ) -> str:
        """Create a presentation by parsing slide markers from draft output.

        Parses ``### Slide N: <title>`` markers and bullet/notes blocks
        from the Draft Content step, mapping directly to create_presentation
        sections without LLM restructuring.
        """
        if not self._tool_registry:
            return "Error: tool registry not available"

        tool = self._tool_registry.resolve("create_presentation")
        if not tool:
            return "Error: create_presentation tool not found"

        # Prefer the explicit Draft / Structure / Outline step output.
        # Walking blindly backwards picks up the Illustrate step's chain
        # output (image_generation status strings) and ships them as the
        # slide body — the symptom that prompted this rewrite.
        draft = _pick_artifact_draft(wmem)
        if not draft:
            return "Error: no draft content found in working memory"

        slides, parse_warning = _parse_slide_draft(
            draft, fallback_title=_resolve_artifact_topic(task) or "Content",
        )
        if parse_warning:
            log.warning(
                "pptx_slide_parse_collapse",
                draft_chars=len(draft),
                task_id=task.id,
            )

        # Project Illustrate Slides picks (image_url + additional_images)
        # onto each slide before the renderer sees them.
        slides = _apply_image_picks(slides, wmem, task)

        try:
            pres_args = {
                "title": task.title or "Presentation",
                "slides": slides,
                "task_id": task.id,
                "session_id": self._session_id,
                "theme": "",
            }
            self._inject_tool_context(tool, pres_args)
            result = await tool.execute(**pres_args)
            if result.success and result.metadata:
                meta = dict(result.metadata)
                if isinstance(getattr(result, "card", None), dict):
                    meta["card"] = result.card
                wmem.record_artifact(meta)
            output = result.output if result.success else f"Error: {result.error}"
            if parse_warning and result.success:
                output = f"{output}\n\nWarning: {parse_warning}"
            return output
        except Exception as e:
            log.error("create_presentation_direct_failed", error=str(e), exc_info=True)
            return f"Error: Presentation creation failed: {e}"

    # ------------------------------------------------------------------
    # Resume / Approval
    # ------------------------------------------------------------------

    async def _resume_task(
        self,
        task: TaskState,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Resume an incomplete task from its last checkpoint."""
        yield _agentic_meta_chunk(model, "resuming", task)

        # Ad-hoc tasks (no flow_id) — resume via chain execution
        if not task.flow_id:
            if self._tool_registry:
                from augmentum.tools.chain import ToolChainPlanner
                query = task.original_query or _extract_query(request)
                planner = ToolChainPlanner(
                    backend=self._backend,
                    tool_registry=self._tool_registry,
                    model=model,
                )
                try:
                    plan = await planner.plan(query, task.plan_md)
                    async for chunk in self._execute_chain_plan(
                        plan, query, model, request, task,
                    ):
                        yield chunk
                    task.status = TaskStatus.COMPLETED
                    if self._task_store:
                        await self._task_store.update(task)
                    yield _agentic_meta_chunk(model, "completed", task)
                    return
                except Exception as exc:
                    log.warning("ad_hoc_resume_failed", error=str(exc))
            # Fallback: direct LLM execution of each step
            query = task.original_query or _extract_query(request)
            async for chunk in self._execute_ad_hoc_steps(task, query, model, request):
                yield chunk
            return

        if task.flow_id and self._flow_store:
            flow = await self._flow_store.get(task.flow_id, user_id=self._user_id)
            if flow:
                query = task.original_query or _extract_query(request)

                wmem = WorkingMemory(goal=query, plan_md=task.plan_md)
                ctx = StepContext(query=query, model=model)
                ctx.plan = task.plan_md
                ctx.conversation = _build_conversation(request)

                for idx, output in task.step_outputs.items():
                    if idx < len(flow.steps):
                        step_name = flow.steps[idx].name
                        ctx.record_step(step_name, output)
                        wmem.record_generative_output(step_name, output)

                pipeline = [
                    {
                        "name": s.name,
                        "role": s.role,
                        "tools": list(s.tool_names or []),
                    }
                    for s in flow.steps if s.enabled
                ]

                for step_idx, step in enumerate(flow.steps):
                    # Skip disabled steps and any step whose output is already
                    # checkpointed — `current_step` can point at either the
                    # last completed or the in-progress step depending on
                    # when persistence fired, so checking step_outputs is the
                    # authoritative "did this step finish" signal on resume.
                    if not step.enabled or step_idx in task.step_outputs:
                        continue

                    task.current_step = step_idx
                    ctx.plan = mark_current_step(task.plan_md, step_idx)

                    step_content = ""
                    if step.stream_to_user:
                        step_content = f"### {step.name}\n\n"
                    yield _flow_step_chunk(
                        model, step.name, "running", pipeline, task,
                        content=step_content,
                    )

                    is_chain = _is_chain_step(step)
                    if is_chain:
                        progress_queue: asyncio.Queue[InternalStreamChunk | None] = asyncio.Queue()

                        async def _run_chain(_s=step, _q=progress_queue, _idx=step_idx):
                            try:
                                return await self._execute_chain_step(
                                    _s, ctx, model, request, task, wmem, _q,
                                    step_idx=_idx,
                                )
                            finally:
                                await _q.put(None)

                        step_task = asyncio.create_task(_run_chain())
                        while True:
                            chunk = await progress_queue.get()
                            if chunk is None:
                                break
                            yield chunk
                        step_output = await step_task
                    else:
                        step_output = await self._execute_generative_step(
                            step, ctx, model, request, task, wmem,
                        )

                    ctx.record_step(step.name, step_output)
                    task.record_step_output(step_idx, step_output)
                    task.plan_md = update_plan_step(task.plan_md, step_idx)

                    if step_output:
                        yield InternalStreamChunk(
                            content_delta="",
                            model=model,
                            augmentum={
                                "mode": "agentic",
                                "task_id": task.id,
                                "phase": step.name,
                                "phase_content_delta": step_output,
                            },
                        )

                    if self._task_store:
                        await self._task_store.update(task)

                    yield _flow_step_chunk(model, step.name, "complete", pipeline, task)

                    if step.stream_to_user and step_output:
                        yield InternalStreamChunk(
                            content_delta=step_output + "\n\n",
                            model=model,
                            augmentum={"mode": "agentic", "task_id": task.id},
                        )

                task.status = TaskStatus.COMPLETED
                if self._task_store:
                    await self._task_store.update(task)
                yield _agentic_meta_chunk(model, "completed", task)
                return

        yield InternalStreamChunk(
            content_delta=(
                f"You have an incomplete task: **{task.title}** "
                f"(step {task.current_step + 1}/{task.total_steps}). "
                "Send a new message to continue, or start a new task.\n"
            ),
            model=model,
            augmentum={"mode": "agentic", "task_id": task.id},
        )

    async def _handle_approval_response(
        self,
        task: TaskState,
        user_response: str,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Handle user's response to an approval request."""
        response_lower = user_response.strip().lower()

        if response_lower in ("approve", "accept", "yes", "ok", "continue", "go", "do it", "start", "run"):
            task.status = TaskStatus.RUNNING
            if self._task_store:
                await self._task_store.update(task)
            async for chunk in self._resume_task(task, model, request):
                yield chunk
        elif response_lower in ("skip", "no", "cancel", "stop"):
            task.status = TaskStatus.COMPLETED
            task.error = "Cancelled by user"
            if self._task_store:
                await self._task_store.update(task)
            yield _agentic_meta_chunk(model, "cancelled", task)
        else:
            task.status = TaskStatus.RUNNING
            task.plan_md += f"\n\nUser modification: {user_response}"
            if self._task_store:
                await self._task_store.update(task)
            async for chunk in self._resume_task(task, model, request):
                yield chunk

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _execute_ad_hoc_steps(
        self,
        task: TaskState,
        query: str,
        model: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute ad-hoc plan steps via sequential LLM calls.

        Fallback for tasks without a flow — parses the plan markdown
        and executes each step as a prompted LLM generation.
        """
        _, step_descs = parse_plan(task.plan_md)
        if not step_descs:
            step_descs = [query]

        prior_outputs: list[str] = []

        for i, step_desc in enumerate(step_descs):
            if i <= task.current_step and task.current_step > 0:
                continue  # skip already-completed steps

            task.current_step = i
            if self._task_store:
                await self._task_store.update(task)

            yield InternalStreamChunk(
                content_delta=f"### Step {i + 1}: {step_desc}\n\n",
                model=model,
                augmentum={"mode": "agentic", "task_id": task.id,
                           "phase": f"step_{i + 1}"},
            )

            prior_context = ""
            if prior_outputs:
                prior_context = "\n\n".join(
                    f"Step {j + 1} output:\n{out[:1000]}"
                    for j, out in enumerate(prior_outputs)
                )

            step_prompt = (
                f"You are executing step {i + 1} of a plan.\n\n"
                f"Original request: {query}\n\n"
                f"Full plan:\n{task.plan_md}\n\n"
                f"Current step: {step_desc}\n\n"
                + (f"Prior step results:\n{prior_context}\n\n" if prior_context else "")
                + "Execute this step thoroughly. Output the result."
            )

            step_output = ""
            req = InternalChatRequest(
                model=model,
                messages=[
                    Message(role="system", content=step_prompt),
                    Message(role="user", content=f"Execute step {i + 1}: {step_desc}"),
                ],
                stream=True,
            )
            async for chunk in self._backend.chat_stream(req):
                if chunk.content_delta:
                    step_output += chunk.content_delta
                    yield InternalStreamChunk(
                        content_delta=chunk.content_delta,
                        model=model,
                        augmentum={"mode": "agentic", "task_id": task.id},
                    )

            prior_outputs.append(step_output)
            # Store full output; the next-step prompt below already trims
            # to 1000 chars per prior result when building `prior_context`,
            # so context budget is bounded at injection, not at persistence.
            task.record_step_output(i, step_output)
            yield InternalStreamChunk(
                content_delta="\n\n", model=model,
            )

        task.status = TaskStatus.COMPLETED
        if self._task_store:
            await self._task_store.update(task)
        yield _agentic_meta_chunk(model, "completed", task)

    @staticmethod
    def _inject_date(system_prompt: str) -> str:
        """Prepend current date/time to system prompt."""
        from augmentum.utils.datetime_context import get_datetime_context
        return f"{get_datetime_context()}\n\n{system_prompt}"

    async def _call_llm(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str,
        request: InternalChatRequest,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming LLM call, returns full text.

        Args:
            max_tokens: Generation limit.  Prevents verbose models from
                producing unbounded output on internal steps.
        """
        messages = []
        if system_prompt:
            sp = self._inject_date(system_prompt)
            user_system = _extract_user_system(request)
            if user_system:
                sp = f"{user_system}\n\n{sp}"
            messages.append(Message(role="system", content=sp))
        messages.append(Message(role="user", content=user_message))

        llm_request = InternalChatRequest(
            model=model,
            messages=messages,
            stream=False,
            temperature=request.temperature,
            max_tokens=max_tokens,
        )
        response = await self._backend.chat(llm_request)
        return response.message.content if response.message else ""

    async def _call_llm_full(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str,
        request: InternalChatRequest,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Message:
        """Non-streaming LLM call that returns the full Message (incl. tool_calls)."""
        messages = []
        if system_prompt:
            sp = self._inject_date(system_prompt)
            user_system = _extract_user_system(request)
            if user_system:
                sp = f"{user_system}\n\n{sp}"
            messages.append(Message(role="system", content=sp))
        messages.append(Message(role="user", content=user_message))

        llm_request = InternalChatRequest(
            model=model,
            messages=messages,
            stream=False,
            temperature=request.temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        response = await self._backend.chat(llm_request)
        return response.message if response.message else Message(role="assistant", content="")

    async def _stream_llm(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Streaming LLM call, yields chunks."""
        messages = []
        if system_prompt:
            sp = self._inject_date(system_prompt)
            user_system = _extract_user_system(request)
            if user_system:
                sp = f"{user_system}\n\n{sp}"
            messages.append(Message(role="system", content=sp))
        messages.append(Message(role="user", content=user_message))

        llm_request = InternalChatRequest(
            model=model,
            messages=messages,
            stream=True,
            temperature=request.temperature,
        )
        async for chunk in self._backend.chat_stream(llm_request):
            yield chunk

    # ------------------------------------------------------------------
    # Flow + tool resolution
    # ------------------------------------------------------------------

    async def _resolve_agentic_flow(
        self, query: str, model: str,
    ) -> ReasoningFlow | None:
        """Find the agentic flow to run for this turn.

        Priority chain (mirrors :func:`augmentum.reasoning.resolver.resolve_flow`
        but with an agentic-domain guard on the auto-routing leg):

        1. **Explicit selection** — the flow id from the chat UI's flow
           picker (``X-Augmentum-Flow`` header, threaded in via
           ``self._explicit_flow_id``). The user picked it; honour it
           unconditionally. Skip if it's the Auto Routing meta-flow.
        2. **User's default flow** (the "Set as default" toggle in the
           side panel). Only used when it's tagged ``agentic`` — a
           default of, say, the analytical "Research" flow would not
           run cleanly through the agentic chain executor, so we let it
           fall through to step 3 in that case.
        3. **Keyword auto-routing across agentic-domain flows only** —
           legacy behaviour, but scoped to the agentic domain so an
           analytical flow can't accidentally win on a shared keyword.

        Without step 1, ``"build me a tool"`` (and most other "app/tool/site"
        phrasings) keyword-matched the built-in Application flow, which then
        ran ``build_application`` regardless of the user's selection. That was
        the "every request becomes a build request" bug.
        """
        if not self._flow_store:
            return None

        # 1. Explicit selection from the flow picker.
        if self._explicit_flow_id:
            try:
                explicit = await self._flow_store.get_flow(
                    self._explicit_flow_id, user_id=self._user_id,
                )
            except Exception:
                explicit = None
            if explicit and explicit.name != "Auto Routing":
                log.info(
                    "agentic_flow_resolved",
                    method="explicit", flow=explicit.name,
                )
                return explicit

        # 2. User's pinned default flow (only if it's an agentic flow).
        try:
            default = await self._flow_store.get_default_flow(user_id=self._user_id)
        except Exception:
            default = None
        if (
            default
            and default.name != "Auto Routing"
            and "agentic" in (default.trigger_domains or [])
        ):
            log.info(
                "agentic_flow_resolved",
                method="default", flow=default.name,
            )
            return default

        # 3. Auto-routing — keyword match across agentic-domain flows only.
        flows = await self._flow_store.list_all(user_id=self._user_id)
        agentic_flows = [
            f for f in flows if "agentic" in (f.trigger_domains or [])
        ]
        if not agentic_flows:
            return None

        query_lower = query.lower()
        best_flow: ReasoningFlow | None = None
        best_score = 0

        for flow in agentic_flows:
            score = sum(
                1 for kw in (flow.trigger_keywords or [])
                if kw.lower() in query_lower
            )
            if score > best_score:
                best_score = score
                best_flow = flow

        if best_flow:
            log.info(
                "agentic_flow_resolved",
                method="auto_routing", flow=best_flow.name, score=best_score,
            )
        return best_flow

    def _resolve_step_tools(self, step: FlowStep) -> list:
        """Resolve tools available for a step based on its config."""
        if not self._tool_registry:
            return []

        tools = []

        for name in step.tool_names:
            tool = self._tool_registry.resolve(name)
            if tool:
                tools.append(tool)

        if step.tool_categories:
            from augmentum.tools.base import ToolCategory
            for cat_str in step.tool_categories:
                try:
                    cat = ToolCategory(cat_str)
                    tools.extend(self._tool_registry.list_tools(cat))
                except ValueError:
                    pass

        seen = set()
        unique = []
        for t in tools:
            if t.name not in seen:
                seen.add(t.name)
                unique.append(t)

        return unique


# ---------------------------------------------------------------------------
# Stream chunk helpers
# ---------------------------------------------------------------------------


def _agentic_meta_chunk(
    model: str,
    status: str,
    task: TaskState,
    content: str = "",
) -> InternalStreamChunk:
    """Build a chunk carrying agentic task metadata.

    Delegates state collection to ``task.meta_envelope`` so this stays in
    lock-step with every other emission helper — see the docstring there
    for the contract.
    """
    return InternalStreamChunk(
        content_delta=content,
        model=model,
        augmentum=task.meta_envelope(status_override=status),
    )


def _artifact_status_preamble(wmem, ctx) -> str:
    """A hard "here is what the artifacts ACTUALLY contain" block for the
    deliver step's context — so it reports reality instead of regurgitating
    the planned outline. Returns "" when there's nothing notable to flag.

    Surfaces three things: (1) any review verdict that's still REVISE/FAIL,
    (2) artifact-degeneration signals scraped from the run's step outputs
    (a deck collapsed to one slide, an empty deck, parser warnings), and
    (3) the real sub-unit count of each produced artifact (slides/pages/etc.)
    where the tool recorded one.
    """
    lines: list[str] = []

    # 1. Outstanding review verdicts.
    for sname, v in (getattr(wmem, "_review_verdicts", None) or {}).items():
        if not isinstance(v, dict):
            continue
        verdict = str(v.get("verdict", "")).strip().upper()
        if verdict in ("REVISE", "REVISION", "NEEDS_REVISION", "FAIL"):
            issues = v.get("issues") or []
            detail = "; ".join(str(i) for i in issues) if issues else str(v.get("reasoning", "") or "")
            lines.append(
                f"- A review step ('{sname}') returned **{verdict}** — the deliverable is NOT finished."
                + (f" Outstanding issues: {detail}" if detail else "")
            )

    # 2. Degeneration signals in the run's outputs (any step).
    try:
        blob = ctx.all_outputs or ""
    except Exception:
        blob = ""
    for pat, msg in (
        (r"no slide markers|collapsed to (?:a )?single slide|found no slide\b",
         "the slide draft had no `### Slide N:` markers, so the deck collapsed to ONE slide — it is NOT a usable multi-slide presentation"),
        (r"\bno slides? (?:found|present)\b|empty (?:deck|presentation)",
         "the presentation came out empty"),
        (r"parser? (?:warning|error|failed)|parse[_ ](?:warning|error)",
         "an artifact parser reported a problem during assembly"),
    ):
        if re.search(pat, blob, re.IGNORECASE):
            lines.append(f"- ⚠ {msg}.")
            break  # one note is enough

    # 3. Ground truth on the produced artifacts.
    _UNIT = {"pptx": "slide", "docx": "section", "pdf": "page", "xlsx": "sheet", "md": "section"}
    for a in (getattr(wmem, "_artifacts", None) or []):
        if not isinstance(a, dict):
            continue
        name = a.get("display_name") or a.get("filename") or a.get("name") or "artifact"
        fmt = str(a.get("format") or a.get("page_type") or "").lower()
        n = None
        for k in ("slide_count", "slides", "page_count", "pages",
                  "section_count", "sections", "sheet_count", "sheets"):
            val = a.get(k)
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                n = val
                break
            if isinstance(val, list | tuple):
                n = len(val)
                break
        if n is not None:
            unit = _UNIT.get(fmt, "section")
            lines.append(f"- The produced **{name}** ({fmt or 'file'}) actually contains **{n} {unit}{'' if n == 1 else 's'}**.")
        else:
            lines.append(f"- Produced **{name}** ({fmt or 'file'}).")

    if not lines:
        return ""
    return (
        "## ⚠ ARTIFACT STATUS — READ THIS FIRST; IT OVERRIDES THE OUTLINE BELOW\n"
        + "\n".join(lines)
        + "\n\nReport ONLY what the artifacts ACTUALLY contain. Do NOT quote the "
        "*planned* slide/section count, or figures the produced file does not "
        "have. If a review returned REVISE/FAIL, or an artifact is degenerate "
        "(e.g. a deck collapsed to one slide), tell the user plainly that the "
        "deliverable needs more work and exactly what's wrong — do NOT present "
        "it as finished or polished."
    )


def _artifact_delivery_payload(a: dict) -> dict | None:
    """Project one artifact metadata dict into a frontend delivery payload."""
    if not isinstance(a, dict):
        return None
    url = a.get("download_url") or a.get("url") or ""
    artifact_id = a.get("artifact_id") or a.get("id") or ""
    raw_card = a.get("card") if isinstance(a.get("card"), dict) else None
    nested_meta = a.get("metadata") if isinstance(a.get("metadata"), dict) else {}
    # Reconstruct download_url from id if backend stored one but not the other.
    if not url and artifact_id:
        url = f"/api/artifacts/{artifact_id}/download"
    if not url and not artifact_id:
        return None
    name = a.get("display_name") or a.get("filename") or a.get("name") or "Artifact"
    fmt = a.get("format", "") or a.get("page_type", "") or nested_meta.get("format", "")
    page_type = a.get("page_type", "") or nested_meta.get("page_type", "")
    return {
        "id": artifact_id,
        "name": name,
        "display_name": name,
        "title": name,
        "download_url": url,
        "format": fmt,
        "kind": fmt,
        "size_bytes": int(a.get("size_bytes") or 0),
        "page_type": page_type,
        "card": raw_card,
    }


def _build_artifact_cards(wmem: WorkingMemory) -> list[dict]:
    """Project recorded artifacts into frontend-friendly card payloads.

    Contract (augmentum.delivery.payload for kind='artifact_card'):
      {
        "id": str,            # artifact_id — required for Open/Edit routing
        "name": str,          # display name (also exposed as display_name)
        "display_name": str,  # alias of name for renderers using either key
        "title": str,         # alias of name for ToolCard-style renderers
        "download_url": str,  # relative or absolute URL
        "format": str,        # e.g. pdf|pptx|xlsx|docx|md|png
        "kind": str,          # alias of format for ToolCard-style renderers
        "size_bytes": int,    # file size for the size pill on the card
        "page_type": str,     # ebook | document | presentation | spreadsheet | chart
      }

    The duplicated keys (``name``/``display_name``/``title`` and
    ``format``/``kind``) let the same payload feed both the inspector
    task panel and any ToolCard-shaped consumer without each side
    needing to know the other's vocabulary.
    """
    cards: list[dict] = []
    for a in wmem._artifacts:
        payload = _artifact_delivery_payload(a)
        if payload:
            cards.append(payload)
    return cards


def _extract_citations(text: str, wmem: WorkingMemory) -> list[dict]:
    """Pull [1], [2]... references out of the delivered prose.

    Contract (augmentum.delivery.payload for kind='citation'):
      {"index": int, "source": str | None}

    ``source`` is left None here; a future pass can link citation indices
    to specific URLs captured in wmem chain results. Emitting the indices
    is enough for the frontend to render a citation footer row.
    """
    import re as _re
    seen: set[int] = set()
    out: list[dict] = []
    for m in _re.finditer(r"\[(\d+)\]", text):
        idx = int(m.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        out.append({"index": idx, "source": None})
    # Cap to avoid runaway emission on degenerate outputs.
    return out[:20]


def _normalize_pipeline(pipeline: list) -> list[dict]:
    """Coerce mixed-shape pipeline entries to the dict form.

    Ad-hoc + chain-planner paths still pass ``list[str]``; flow paths
    pass ``list[dict]`` with name/role/tools. Both flow into TaskState
    identically once normalised here.
    """
    out: list[dict] = []
    for p in pipeline:
        if isinstance(p, dict):
            out.append(p)
        else:
            out.append({"name": p if isinstance(p, str) else ""})
    return out


def _flow_step_chunk(
    model: str,
    step_name: str,
    status: str,
    pipeline: list,
    task: TaskState,
    *,
    content: str = "",
) -> InternalStreamChunk:
    """Build a chunk for flow step progress (matches analytical phase format).

    Mirrors the ``_agentic_meta_chunk`` envelope so the inspector receives
    a consistent snapshot, then layers the per-step ``phase`` /
    ``phase_status`` keys on top. Pipeline is also stamped onto ``task``
    so subsequent meta chunks can compute ``phases`` without re-passing
    the pipeline list.
    """
    normalised = _normalize_pipeline(pipeline)
    if normalised and normalised != task.pipeline:
        task.pipeline = normalised
    names = [p.get("name", "") for p in normalised]
    step_idx = names.index(step_name) if step_name in names else task.current_step

    envelope = task.meta_envelope(
        active_step_index=step_idx,
        active_step_status=status,
    )
    envelope["phase"] = step_name
    envelope["phase_status"] = status
    return InternalStreamChunk(
        content_delta=content,
        model=model,
        augmentum=envelope,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _is_chain_step(step: FlowStep) -> bool:
    """Classify whether a flow step should use the chain executor.

    Generative-primary roles (draft, review, deliver, respond, plan) are
    ALWAYS generative — even if they list supplementary tools.
    Tool-primary roles (search, verify, create, illustrate) use the
    chain executor.
    """
    if step.role in _GENERATIVE_ROLES:
        return False
    if step.tool_names or step.tool_categories:
        return True
    return step.role in _CHAIN_ROLES


def _extract_query(request: InternalChatRequest) -> str:
    """Extract the user's query from the last user message."""
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _extract_user_system(request: InternalChatRequest) -> str:
    """Pull the frontend-supplied system prompt (if any) from the request.

    Computed per-call instead of cached on the handler — with cross-request
    handler caching the instance would otherwise carry one request's system
    prompt into another's LLM call.
    """
    if request.messages and request.messages[0].role == "system":
        return request.messages[0].content.strip()
    return ""


def _build_conversation(request: InternalChatRequest) -> str:
    """Build conversation context from request messages."""
    parts: list[str] = []
    for msg in request.messages[:-1]:
        if msg.role in ("user", "assistant") and msg.content:
            parts.append(f"**{msg.role}:** {msg.content[:500]}")
    return "\n".join(parts[-6:])


def _is_new_task_request(query: str) -> bool:
    """Heuristic: does this look like a new task request vs. a continuation?"""
    new_task_signals = [
        "create", "generate", "make", "build", "write", "produce",
        "research", "analyze", "prepare", "draft", "design",
    ]
    query_lower = query.lower()
    if len(query.split()) >= 5:
        return any(s in query_lower for s in new_task_signals)
    return False


def _build_tools_section(tools: list) -> str:
    """Build a text description of available tools for injection into prompts."""
    if not tools:
        return ""
    lines = ["## Available Tools\n"]
    for tool in tools:
        schema_hint = ""
        if tool.input_schema:
            props = tool.input_schema.get("properties", {})
            if props:
                params = ", ".join(props.keys())
                schema_hint = f" (params: {params})"
        lines.append(f"- **{tool.name}**: {tool.description}{schema_hint}")
    lines.append(
        "\nTo use a tool, output: TOOL_CALL: <tool_name>\n<json_arguments>"
    )
    return "\n".join(lines)
