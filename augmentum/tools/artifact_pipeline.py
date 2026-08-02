"""Unified artifact content assembly pipeline.

Intercepts artifact tool calls across all modes, researches the topic,
drafts format-appropriate content, wires images, and delegates to
existing render tools.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ResearchItem:
    """A single piece of research (fact, quote, data point)."""
    content: str
    source_url: str = ""
    topic: str = ""


@dataclass
class DataTable:
    """Structured tabular data (for xlsx/chart)."""
    headers: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    source: str = ""


@dataclass
class ImageRef:
    """Reference to an available image."""
    url: str
    description: str = ""
    source: str = ""  # "generated", "web", "artifact"


@dataclass
class ArtifactRequest:
    """What the caller wants created."""
    format: str  # pdf, docx, pptx, xlsx, chart
    topic: str
    title: str | None = None
    theme: str | None = None
    tool_params: dict = field(default_factory=dict)


@dataclass
class PipelineContext:
    """What's already available from the calling mode."""
    message_history: list[dict] = field(default_factory=list)
    working_memory: Any = None  # Optional WorkingMemory from agentic
    tool_results: list[Any] = field(default_factory=list)
    generated_images: list[str] = field(default_factory=list)
    # Per-slide image picks + candidate pool from a prior Illustrate Slides
    # step. When present the pptx pipeline skips its own illustrate phase
    # and projects these picks onto the slides instead.
    slide_image_picks: dict[int, dict] = field(default_factory=dict)
    image_candidates: dict[int, list[dict]] = field(default_factory=dict)
    # Owner of the artifacts the pipeline produces. ArtifactStore.save
    # requires user_id (it's a user-scoped table); the render tools call
    # Tool.extract_user_id(kwargs), which reads ``_user_id`` from kwargs
    # or ``_context.user_id``. ``execute_artifact_pipeline`` injects this
    # into the assembled kwargs before invoking the render tool so an
    # empty value here raises a clear error at save time rather than
    # silently dropping into the anon row.
    user_id: str = ""


@dataclass
class ContentInventory:
    """Result of the context scan phase."""
    topic: str
    existing_research: list[ResearchItem] = field(default_factory=list)
    existing_data: list[DataTable] = field(default_factory=list)
    existing_images: list[ImageRef] = field(default_factory=list)
    existing_draft: str | None = None
    coverage_gaps: list[str] = field(default_factory=list)
    needs_research: bool = True
    needs_images: bool = False


@dataclass
class ArtifactResult:
    """What the pipeline returns."""
    artifact_id: str
    download_url: str
    display_name: str
    metadata: dict = field(default_factory=dict)
    source_json: dict = field(default_factory=dict)


# Type alias for the LLM caller adapter
LLMCaller = Callable[[str, str], Awaitable[str]]


def build_backend_pipeline_caller(backend, model: str = "") -> LLMCaller:
    """Wrap a ModelBackend.chat() into the LLMCaller interface."""

    async def caller(system: str, user: str) -> str:
        from augmentum.models.base import InternalChatRequest, Message

        req = InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user),
            ],
            stream=False,
        )
        resp = await backend.chat(req)
        return resp.content or ""

    return caller


def build_agentic_pipeline_caller(call_llm_fn, model: str, request) -> LLMCaller:
    """Wrap an agentic handler's _call_llm into the LLMCaller interface."""

    async def caller(system: str, user: str) -> str:
        return await call_llm_fn(
            model=model,
            system_prompt=system,
            user_message=user,
            request=request,
        )

    return caller

def resolve_pipeline_tools(app) -> dict:
    """Resolve research and render tools from FastAPI app state."""
    registry = getattr(app.state, "tool_registry", None)
    if not registry:
        return {"search_tool": None, "fetch_tool": None, "image_search_tool": None}
    return {
        "search_tool": registry.resolve("web_search"),
        "fetch_tool": registry.resolve("web_fetch"),
        "image_search_tool": registry.resolve("image_search"),
    }


# Artifact tool names that trigger the pipeline
ARTIFACT_TOOLS = frozenset({
    "create_document",
    "create_presentation",
    "create_spreadsheet",
    "create_chart",
})

# Tools whose output is genuine research/data (URLs, facts, quotes).
_RESEARCH_TOOLS = frozenset({
    "web_search", "web_fetch",
    "wikipedia", "wikipedia_search", "arxiv_search",
    "knowledge_pack_search", "rag_search", "document_search",
})

# Tools that emit status/instruction text or artifact receipts rather than
# research content. Treating these as "research" poisons the draft prompt —
# the LLM sees only "Image generated successfully. Do NOT call image_generation
# again..." and regurgitates it into slide bodies.
_NON_RESEARCH_TOOLS = frozenset({
    "image_generation", "image_search",
    "create_document", "create_presentation", "create_spreadsheet",
    "create_chart", "create_ebook", "create_application",
    "create_audio", "create_video",
})

# Image-need keywords
_IMAGE_KEYWORDS = re.compile(
    r"\b(illustrat\w*|with images|visual|photos?|diagrams?|pictures?|infographic)\b",
    re.IGNORECASE,
)

# URL extraction pattern
_URL_PATTERN = re.compile(r"https?://[^\s\)\"'>]+")


# ---------------------------------------------------------------------------
# Phase 1: Context Scan
# ---------------------------------------------------------------------------


async def scan_context(
    request: ArtifactRequest,
    context: PipelineContext,
    llm_caller: LLMCaller,
) -> ContentInventory:
    """Phase 1: Inspect available context and produce a ContentInventory."""
    t0 = time.monotonic()
    inventory = ContentInventory(topic=request.topic)

    # --- Extract from message history ---
    for msg in context.message_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue
        if role == "tool":
            tool_name = (msg.get("name") or msg.get("tool_name") or "").strip()
            if tool_name in _NON_RESEARCH_TOOLS:
                continue
            # When we can't identify the tool, accept only content that
            # looks like research (carries a URL) to avoid poisoning the
            # pool with status text from artifact / image-generation calls.
            urls = _URL_PATTERN.findall(content)
            if not tool_name and not urls:
                continue
            source = urls[0] if urls else ""
            inventory.existing_research.append(
                ResearchItem(content=content, source_url=source, topic=request.topic)
            )

    # --- Extract from working memory (agentic) ---
    wmem = context.working_memory
    if wmem is not None:
        steps = getattr(wmem, "_steps", [])
        chain_results = getattr(wmem, "_chain_results", {}) or {}
        for name, kind, _ in steps:
            if kind == "chain":
                # Iterate per-tool so we can filter image_generation /
                # create_* status text out of the research pool. Those tools
                # emit instruction-shaped output ("Image generated successfully.
                # Do NOT call image_generation again...") that the drafter
                # otherwise regurgitates into slide/section bodies.
                tool_results = chain_results.get(name) or {}
                if tool_results:
                    name_lc = name.lower()
                    research_step = (
                        "research" in name_lc or "search" in name_lc
                    )
                    for tr in tool_results.values():
                        tool_output = getattr(tr, "output", "") or ""
                        tool_name = getattr(tr, "tool_name", "") or ""
                        tool_meta = getattr(tr, "metadata", None) or {}
                        if not getattr(tr, "success", False):
                            continue
                        if tool_name == "image_generation":
                            img_url = tool_meta.get("url") or ""
                            if img_url:
                                inventory.existing_images.append(
                                    ImageRef(
                                        url=img_url,
                                        description=tool_meta.get("prompt", ""),
                                        source="generated",
                                    )
                                )
                            continue
                        if tool_name in _NON_RESEARCH_TOOLS:
                            continue
                        if not tool_output or len(tool_output) < 20:
                            continue
                        # Whitelist when we know it's a research tool,
                        # otherwise only accept output that *looks* like
                        # research (contains a URL) on research-named steps.
                        is_research = (
                            tool_name in _RESEARCH_TOOLS
                            or (research_step and _URL_PATTERN.search(tool_output))
                        )
                        if not is_research:
                            continue
                        urls = _URL_PATTERN.findall(tool_output)
                        source = urls[0] if urls else ""
                        inventory.existing_research.append(
                            ResearchItem(
                                content=tool_output,
                                source_url=source,
                                topic=request.topic,
                            )
                        )
                    continue
                # Fall through to legacy aggregate handling when
                # _chain_results is unavailable (older callers, test fakes).
                output = wmem.get_step_output(name)
                if not output or len(output) < 20:
                    continue
                urls = _URL_PATTERN.findall(output)
                source = urls[0] if urls else ""
                inventory.existing_research.append(
                    ResearchItem(content=output, source_url=source, topic=request.topic)
                )
            else:
                output = wmem.get_step_output(name)
                if not output or len(output) < 20:
                    continue
                name_lc = name.lower()
                if "research" in name_lc or "search" in name_lc:
                    urls = _URL_PATTERN.findall(output)
                    source = urls[0] if urls else ""
                    inventory.existing_research.append(
                        ResearchItem(content=output, source_url=source, topic=request.topic)
                    )
                elif kind == "generative" and (
                    "draft" in name_lc
                    or "write" in name_lc
                    or "structure" in name_lc
                    or "outline" in name_lc
                ):
                    # Prefer the richest generative source as the canonical
                    # draft. A later "Draft Content" step beats an earlier
                    # "Structure Slides" outline.
                    if (
                        inventory.existing_draft is None
                        or len(output) > len(inventory.existing_draft)
                    ):
                        inventory.existing_draft = output

    # --- Extract from analytical tool results ---
    for tr in context.tool_results:
        output = getattr(tr, "output", "")
        tool_name = getattr(tr, "tool_name", "")
        if not output or not getattr(tr, "success", False):
            continue
        if tool_name in ("web_search", "web_fetch"):
            urls = _URL_PATTERN.findall(output)
            source = urls[0] if urls else ""
            inventory.existing_research.append(
                ResearchItem(content=output, source_url=source, topic=request.topic)
            )

    # --- Collect existing images ---
    for img_url in context.generated_images:
        inventory.existing_images.append(
            ImageRef(url=img_url, description="", source="generated")
        )

    # --- Image need detection ---
    if request.format == "pptx" or _IMAGE_KEYWORDS.search(request.topic):
        inventory.needs_images = True

    # --- Coverage gap detection via LLM ---
    research_summary = "\n".join(
        r.content[:500] for r in inventory.existing_research[:10]
    ) if inventory.existing_research else "(none yet)"
    try:
        gap_response = await llm_caller(
            "You identify gaps in research coverage. Reply ONLY with a JSON array of strings.",
            f"Topic: {request.topic}\n\nResearch gathered so far:\n{research_summary}\n\n"
            f"What subtopics are missing or thin? Reply as a JSON array of strings. "
            f"If coverage is sufficient, reply with an empty array: []",
        )
        gaps = json.loads(gap_response.strip())
        if isinstance(gaps, list):
            inventory.coverage_gaps = [str(g) for g in gaps]
    except (json.JSONDecodeError, Exception) as exc:
        log.warning("coverage_gap_detection_failed", error=str(exc))

    # --- Determine if research is needed ---
    if not inventory.existing_research and not inventory.existing_draft or inventory.coverage_gaps:
        inventory.needs_research = True
    else:
        inventory.needs_research = False

    elapsed = time.monotonic() - t0
    log.info("context_scan_complete",
             research_items=len(inventory.existing_research),
             has_draft=inventory.existing_draft is not None,
             needs_research=inventory.needs_research,
             needs_images=inventory.needs_images,
             gaps=len(inventory.coverage_gaps),
             elapsed_s=round(elapsed, 2))
    return inventory


# ---------------------------------------------------------------------------
# Phase 2: Research
# ---------------------------------------------------------------------------

# Research timeout
_RESEARCH_TIMEOUT_S = 30


async def run_research(
    inventory: ContentInventory,
    search_tool,
    fetch_tool,
) -> None:
    """Phase: Text research — fill coverage gaps with web search + fetch."""
    if not inventory.needs_research:
        return

    t0 = time.monotonic()
    gaps = inventory.coverage_gaps or [inventory.topic]

    for gap in gaps:
        if time.monotonic() - t0 > _RESEARCH_TIMEOUT_S:
            log.warning("research_timeout", elapsed_s=round(time.monotonic() - t0, 2))
            break

        query = f"{inventory.topic} {gap}"
        try:
            search_result = await asyncio.wait_for(
                search_tool.execute(query=query, num_results=3),
                timeout=10,
            )
        except (TimeoutError, Exception) as exc:
            log.warning("research_search_failed", gap=gap, error=str(exc))
            continue

        if not search_result.success:
            log.warning("research_search_no_results", gap=gap)
            continue

        # Extract URLs from search results and fetch top one
        urls = _URL_PATTERN.findall(search_result.output)
        inventory.existing_research.append(
            ResearchItem(
                content=search_result.output,
                source_url=urls[0] if urls else "",
                topic=gap,
            )
        )

        # Fetch first URL for deeper content
        if urls:
            try:
                fetch_result = await asyncio.wait_for(
                    fetch_tool.execute(url=urls[0], max_chars=10000),
                    timeout=10,
                )
                if fetch_result.success and fetch_result.output:
                    inventory.existing_research.append(
                        ResearchItem(
                            content=fetch_result.output[:5000],
                            source_url=urls[0],
                            topic=gap,
                        )
                    )
            except (TimeoutError, Exception) as exc:
                log.warning("research_fetch_failed", url=urls[0], error=str(exc))

    log.info("research_complete",
             items=len(inventory.existing_research),
             elapsed_s=round(time.monotonic() - t0, 2))


async def run_research_data(
    inventory: ContentInventory,
    search_tool,
    fetch_tool,
) -> None:
    """Phase: Data research — search for tabular/numeric data (xlsx/chart)."""
    t0 = time.monotonic()
    data_queries = [
        f"{inventory.topic} data table statistics",
        f"{inventory.topic} comparison numbers",
    ]

    for query in data_queries:
        if time.monotonic() - t0 > _RESEARCH_TIMEOUT_S:
            break

        try:
            search_result = await asyncio.wait_for(
                search_tool.execute(query=query, num_results=3),
                timeout=10,
            )
        except (TimeoutError, Exception) as exc:
            log.warning("data_research_search_failed", error=str(exc))
            continue

        if not search_result.success:
            continue

        urls = _URL_PATTERN.findall(search_result.output)
        inventory.existing_research.append(
            ResearchItem(content=search_result.output, source_url=urls[0] if urls else "", topic=inventory.topic)
        )

        # Fetch for tabular data
        if urls:
            try:
                fetch_result = await asyncio.wait_for(
                    fetch_tool.execute(url=urls[0], max_chars=15000),
                    timeout=10,
                )
                if fetch_result.success and fetch_result.output:
                    inventory.existing_research.append(
                        ResearchItem(content=fetch_result.output[:8000], source_url=urls[0], topic=inventory.topic)
                    )
            except (TimeoutError, Exception) as exc:
                log.warning("data_research_fetch_failed", error=str(exc))

    log.info("data_research_complete",
             items=len(inventory.existing_research),
             elapsed_s=round(time.monotonic() - t0, 2))


# ---------------------------------------------------------------------------
# Phase 4: Draft
# ---------------------------------------------------------------------------


async def draft_sections(
    inventory: ContentInventory,
    llm_caller: LLMCaller,
) -> str:
    """Phase: Draft document sections with SECTION markers."""
    if inventory.existing_draft and "## SECTION:" in inventory.existing_draft:
        return inventory.existing_draft
    if inventory.existing_draft and "## " in inventory.existing_draft:
        return inventory.existing_draft

    from augmentum.tools.artifact_templates import get_pipeline_template_context

    research_context = "\n\n".join(
        f"[Source: {r.source_url}]\n{r.content}" if r.source_url else r.content
        for r in inventory.existing_research[:15]
    )
    template_ctx = get_pipeline_template_context(inventory.topic, "pdf")

    system = (
        "You are a professional document writer. Structure content into clear sections "
        "using ## SECTION: heading markers. Include all factual data from the research. "
        "Cite sources inline with [Source: URL]. Be thorough and detailed.\n"
        + (template_ctx if template_ctx else "")
    )
    user = (
        f"Topic: {inventory.topic}\n\n"
        f"Research:\n{research_context}\n\n"
        f"Write a comprehensive document with ## SECTION: heading markers for each section. "
        f"Include an introduction and conclusion."
    )

    t0 = time.monotonic()
    draft = await llm_caller(system, user)
    log.info("draft_sections_complete", chars=len(draft), elapsed_s=round(time.monotonic() - t0, 2))
    return draft


# ---------------------------------------------------------------------------
# Slide image-search query crafting (Phase 1 of post-render picker substrate)
#
# These three prompts power the new agentic Illustrate Slides step + the
# /api/agentic/tasks/{id}/expand and /generate-instead endpoints. They share
# a two-pass shape: write an SEO description first (subject / source family /
# format / aesthetic), then derive a 5-10 word query. The description is
# private — never sent to SearXNG — but is persisted on each candidate so
# expansion + generate-instead can read it later without re-asking the LLM.
# ---------------------------------------------------------------------------

_SLIDE_ILLUSTRATE_PROMPT = (
    "You craft image-search queries for presentation slides. For each slide, "
    "do two passes:\n\n"
    "1. SEO description (1-2 sentences, internal scratch). Commit to: "
    "subject (what we see), source family (gov data / encyclopedia / stock "
    "photo / news / university / industry), format (chart / photo / diagram "
    "/ screenshot / illustration), aesthetic (clean / technical / lifestyle "
    "/ editorial).\n\n"
    "2. Query (5-10 words). Convert the description into ranking-friendly "
    "terms. Lead with the subject and one specificity hook — a year, a "
    "place, an org, a technical noun. No filler words.\n\n"
    "Set \"query\": \"\" for slides that need no image (title cards, "
    "pure-text bullets, section dividers, summary lists).\n\n"
    "Set \"prefer_charts\": true when the description names chart / graph / "
    "diagram / figure / plot.\n\n"
    "Output ONLY valid JSON, no preamble, no markdown fence."
)

_SLIDE_EXPAND_PROMPT = (
    "You expand a slide's image options with deliberately DIFFERENT angles. "
    "The user already has one or more queries; you produce K new ones that "
    "approach the same slide from a different aspect, source family, format, "
    "or framing than what exists.\n\n"
    "Diversification axes (use AT LEAST two different ones across your K "
    "queries):\n"
    "- aspect: subject vs context vs metaphor vs data vs process\n"
    "- source family: data portal vs encyclopedia vs stock photo vs news "
    "vs industry vs academic\n"
    "- format: chart vs photo vs diagram vs screenshot vs illustration\n"
    "- specificity: different year, place, org, or technical noun than the "
    "existing queries\n\n"
    "Two-pass each: SEO description first, then derived 5-10 word query. "
    "Same JSON contract as the initial crafter.\n\n"
    "Output ONLY valid JSON, no preamble, no markdown fence."
)

_SLIDE_GENERATE_FROM_DESC_PROMPT = (
    "You convert a slide image description into a prompt for an image "
    "generation model. The description was written for searching the web "
    "— adapt it for generation.\n\n"
    "Rules:\n"
    "- Lead with the concrete visual subject (what we literally see in the "
    "frame), not what it represents.\n"
    "- Pick a style: photorealistic / illustration / vector / painterly / "
    "technical-diagram. Match the description's aesthetic.\n"
    "- AVOID asking for things gen models do poorly: precise numeric "
    "charts, legible text, real logos, real people, real screenshots. If "
    "the description asks for one, pivot to a visual stand-in (chart → "
    "\"a person studying a graph on a laptop\"; screenshot → \"a stylized "
    "interface mockup\").\n"
    "- Stay under 60 words. Concrete > flowery.\n\n"
    "Output ONLY the prompt string. No quotes. No preamble. No newlines."
)


def _strip_json_fence(raw: str) -> str:
    """Drop ```json … ``` fences models add despite "no markdown" instructions."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`")
    return cleaned.strip()


async def craft_initial_slide_queries(
    slides: list[dict],
    llm_caller: LLMCaller,
) -> list[dict]:
    """Run the initial two-pass crafter once over every slide in a deck.

    ``slides`` is the parsed-draft slide list (title + body). Returns a list
    of ``{index, description, query, prefer_charts}`` dicts in slide order.
    Slides the model decides need no image get ``query == ""`` and are
    surfaced through the same list so callers can keep indices aligned.

    Parsing failure → retry once with stricter "no markdown" instruction;
    second failure → return empty list (caller decides whether to skip
    illustration entirely or apply a category-blind fallback).
    """
    slides_payload = [
        {"index": i + 1, "title": s.get("title", ""), "body": s.get("body", "")[:600]}
        for i, s in enumerate(slides)
    ]
    user_msg = (
        f"Slides:\n{json.dumps(slides_payload, ensure_ascii=False)}\n\n"
        "Output JSON:\n"
        "{\"slides\": ["
        "{\"index\": <int>, \"description\": \"<str>\", "
        "\"query\": \"<str>\", \"prefer_charts\": <bool>}"
        "]}\n\n"
        "Example slide: {\"index\": 3, \"title\": \"Solar panel costs fell "
        "90% since 2010\", \"body\": \"- $0.36/W today vs $3.40/W in 2010\\n"
        "- Driven by polysilicon supply scale\"}\n"
        "Example output for that slide:\n"
        "{\"index\": 3, \"description\": \"Line chart of crystalline-silicon "
        "solar module $/W from 2010 through 2024, ideally IRENA or NREL "
        "source, clean professional chart not infographic.\", "
        "\"query\": \"solar module cost decline 2010 IRENA chart\", "
        "\"prefer_charts\": true}"
    )

    t0 = time.monotonic()
    raw = await llm_caller(_SLIDE_ILLUSTRATE_PROMPT, user_msg)
    parsed = _parse_slide_query_payload(raw)
    if not parsed:
        log.warning("slide_illustrate_first_parse_failed", raw_len=len(raw or ""))
        retry = await llm_caller(
            "Output ONLY a JSON object. No markdown, no fences, no preamble.",
            user_msg,
        )
        parsed = _parse_slide_query_payload(retry)
        if not parsed:
            log.warning("slide_illustrate_retry_failed",
                        raw_len=len(retry or ""))
            return []
    log.info("slide_illustrate_complete",
             slides_total=len(slides_payload),
             slides_with_query=sum(1 for x in parsed if x.get("query")),
             elapsed_s=round(time.monotonic() - t0, 2))
    return parsed


async def craft_expansion_queries(
    *,
    slide_title: str,
    slide_body: str,
    existing_queries: list[dict],
    k: int,
    llm_caller: LLMCaller,
) -> list[dict]:
    """Run the diversity-prompt crafter for a single slide.

    ``existing_queries`` carries the queries already in this slide's pool
    so the diversity prompt can exclude their angles. ``k`` is how many
    new queries to ask for (target_count - current_count).

    Returns a list of ``{description, query, prefer_charts}`` dicts.
    Empty list on double-parse failure — caller falls back to a single
    re-run of the same query with a larger count.
    """
    user_msg = (
        f"Slide: \"{slide_title}\"\n"
        f"Body: \"{slide_body[:600]}\"\n\n"
        f"Existing queries (do NOT repeat their angle):\n"
        f"{json.dumps(existing_queries, ensure_ascii=False)}\n\n"
        f"Form {k} new queries, each from a different angle than the "
        f"existing set and each other.\n\n"
        "Output JSON:\n"
        "{\"queries\": [{\"description\": \"<str>\", \"query\": \"<str>\", "
        "\"prefer_charts\": <bool>}]}"
    )
    t0 = time.monotonic()
    raw = await llm_caller(_SLIDE_EXPAND_PROMPT, user_msg)
    parsed = _parse_expansion_payload(raw, k)
    if parsed is None:
        log.warning("slide_expand_first_parse_failed", raw_len=len(raw or ""))
        retry = await llm_caller(
            "Output ONLY a JSON object. No markdown, no fences, no preamble.",
            user_msg,
        )
        parsed = _parse_expansion_payload(retry, k)
        if parsed is None:
            log.warning("slide_expand_retry_failed",
                        raw_len=len(retry or ""))
            return []
    log.info("slide_expand_complete", k=k, returned=len(parsed),
             elapsed_s=round(time.monotonic() - t0, 2))
    return parsed


async def convert_description_to_gen_prompt(
    *,
    description: str,
    slide_title: str,
    llm_caller: LLMCaller,
) -> str:
    """Convert the SEO description for a slide into an image_generation prompt.

    Returns the prompt string, with newlines stripped. On LLM failure
    returns a fallback constructed from the description + slide title so
    the caller doesn't dead-end the picker.
    """
    user_msg = (
        f"Description: \"{description}\"\n"
        f"Slide title: \"{slide_title}\""
    )
    try:
        raw = await llm_caller(_SLIDE_GENERATE_FROM_DESC_PROMPT, user_msg)
    except Exception as exc:
        log.warning("slide_generate_convert_failed", error=str(exc))
        raw = ""
    cleaned = (raw or "").strip().strip('"').replace("\n", " ").strip()
    if not cleaned:
        # Last-resort fallback. Strip out chart/numeric language so the
        # gen model doesn't try to render a chart — the description
        # already names the subject; the slide title gives context.
        fallback = re.sub(
            r"\b(chart|graph|diagram|figure|plot|infographic|screenshot)\b",
            "scene",
            description or slide_title,
            flags=re.IGNORECASE,
        )
        cleaned = (
            f"A photorealistic scene representing: {fallback[:200]}. "
            "Soft natural lighting, clean composition."
        )
    return cleaned[:500]


def _parse_slide_query_payload(raw: str) -> list[dict]:
    """Parse the initial crafter's JSON envelope into a normalised list."""
    if not raw:
        return []
    try:
        parsed = json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(parsed, dict):
        items = parsed.get("slides")
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
        out.append({
            "index": idx,
            "description": str(item.get("description") or "").strip(),
            "query": str(item.get("query") or "").strip(),
            "prefer_charts": bool(item.get("prefer_charts")),
        })
    return out


def _parse_expansion_payload(raw: str, expected_k: int) -> list[dict] | None:
    """Parse the expansion crafter's JSON. Returns None on hard failure."""
    if not raw:
        return None
    try:
        parsed = json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict):
        items = parsed.get("queries")
    elif isinstance(parsed, list):
        items = parsed
    else:
        return None
    if not isinstance(items, list):
        return None
    out: list[dict] = []
    for item in items[:expected_k]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query") or "").strip()
        if not q:
            continue
        out.append({
            "description": str(item.get("description") or "").strip(),
            "query": q,
            "prefer_charts": bool(item.get("prefer_charts")),
        })
    return out


async def draft_slides(
    inventory: ContentInventory,
    llm_caller: LLMCaller,
) -> str:
    """Phase: Draft presentation slides with Slide N markers."""
    if inventory.existing_draft and "### Slide" in inventory.existing_draft:
        return inventory.existing_draft

    from augmentum.tools.artifact_templates import get_pipeline_template_context

    research_context = "\n\n".join(
        f"[Source: {r.source_url}]\n{r.content}" if r.source_url else r.content
        for r in inventory.existing_research[:15]
    )
    template_ctx = get_pipeline_template_context(inventory.topic, "pptx")

    system = (
        "You are a professional presentation designer. Create slides using "
        "### Slide N: Title markers. Keep bullet points concise. Include speaker notes "
        "with **Notes:** prefix. Suggest layout per slide (title/content/two_column/blank).\n"
        + (template_ctx if template_ctx else "")
    )
    user = (
        f"Topic: {inventory.topic}\n\n"
        f"Research:\n{research_context}\n\n"
        f"Create a presentation with ### Slide N: Title markers. "
        f"Include a title slide, content slides, and a summary slide."
    )

    t0 = time.monotonic()
    draft = await llm_caller(system, user)

    # If the model ignored the marker grammar, retry once with a stricter
    # prompt. Without this the downstream parser collapses the entire
    # draft into one slide titled after the topic — the failure that
    # ships a deck reading "Plan / Image generated successfully…".
    if "### Slide" not in draft:
        log.warning("draft_slides_missing_markers_retrying", chars=len(draft))
        strict_system = (
            "Output ONLY slide markers — no preamble, no commentary. "
            "Every slide MUST start with a literal '### Slide N: <title>' line. "
            "First slide is the title slide; final slide is a summary."
        )
        strict_user = (
            f"Topic: {inventory.topic}\n\n"
            f"Research:\n{research_context}\n\n"
            f"Write 5-10 slides. Required line format:\n"
            f"### Slide 1: <title>\n- bullet\n- bullet\n**Notes:** <speaker notes>\n"
        )
        retry_draft = await llm_caller(strict_system, strict_user)
        if "### Slide" in retry_draft:
            draft = retry_draft
        else:
            log.warning(
                "draft_slides_retry_also_missing_markers",
                research_items=len(inventory.existing_research),
            )

    log.info("draft_slides_complete", chars=len(draft), elapsed_s=round(time.monotonic() - t0, 2))
    return draft


def _parse_sheets_json(raw: str) -> list[dict] | None:
    """Parse a sheets JSON payload, tolerating markdown fences. None on failure."""
    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed.get("sheets", [parsed] if "headers" in parsed else [])
    if isinstance(parsed, list):
        return parsed
    return None


async def draft_sheet_structure(
    inventory: ContentInventory,
    llm_caller: LLMCaller,
) -> list[dict]:
    """Phase: Structure data into spreadsheet sheets.

    Injects the best-matching XLSX template's design guidance, requires a
    minimum row density with a worked example, and re-drafts once if the
    first attempt comes back empty/thin (the local-model failure mode).
    """
    from augmentum.tools.artifact_templates import get_pipeline_template_context
    from augmentum.tools.artifact_validate import (
        MIN_SHEET_ROWS,
        sheet_quality,
        sheet_repair_user,
    )

    research_context = "\n\n".join(
        r.content for r in inventory.existing_research[:15]
    )
    template_ctx = get_pipeline_template_context(inventory.topic, "xlsx")

    system = (
        "You organize data into spreadsheet structure. Output ONLY valid JSON — "
        "no markdown, no prose.\n"
        + (template_ctx + "\n\n" if template_ctx else "")
        + "Data requirements:\n"
        f"- Provide AT LEAST {MIN_SHEET_ROWS} data rows (more when the data "
        "supports it).\n"
        "- Every row must have one value per header — no ragged rows.\n"
        "- Use real values from the data. If a figure is unknown, infer a "
        "reasonable estimate and mark it '(est)' — never leave cells as TBD, "
        "N/A, or blank."
    )
    user = (
        f"Topic: {inventory.topic}\n\n"
        f"Data:\n{research_context}\n\n"
        "Output JSON exactly in this shape (fill with real data):\n"
        '{"sheets": [{"name": "Summary", '
        '"headers": ["Item", "2024", "2025", "Change"], '
        '"rows": [["Revenue", 120, 145, 25], ["Costs", 80, 92, 12], '
        '["Profit", 40, 53, 13], ["Margin %", 33, 37, 4]], '
        '"column_formats": {}, "summary_row": "none"}]}'
    )

    t0 = time.monotonic()
    raw = await llm_caller(system, user)
    sheets = _parse_sheets_json(raw)
    if sheets is None:
        log.warning("sheet_structure_parse_failed", raw_len=len(raw))

    # Validate-and-repair: re-draft once when the result is degenerate
    # (parse failure, no rows, ragged, too few). This is the substrate guard
    # that lifts thin local-model output above its weight.
    quality = sheet_quality(sheets or [])
    if sheets is None or quality.needs_repair:
        reason = quality.reason if sheets is not None else "invalid JSON"
        log.info("sheet_structure_repair", reason=reason)
        sys2, user2 = sheet_repair_user(inventory.topic, research_context, reason)
        raw2 = await llm_caller(sys2, user2)
        repaired = _parse_sheets_json(raw2)
        if repaired is not None and not sheet_quality(repaired).degenerate:
            sheets = repaired
        elif repaired is not None and sheets is None:
            # First attempt was unparseable; keep whatever the repair gave.
            sheets = repaired

    if not sheets:
        sheets = [{
            "name": "Data", "headers": ["Item", "Value"], "rows": [],
            "column_formats": {}, "summary_row": "none",
        }]

    log.info("draft_sheets_complete", elapsed_s=round(time.monotonic() - t0, 2),
             sheets=len(sheets))
    return sheets


def _parse_chart_json(raw: str) -> dict | None:
    """Parse a chart JSON payload, tolerating markdown fences. None on failure."""
    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def draft_chart_dataset(
    inventory: ContentInventory,
    llm_caller: LLMCaller,
    explicit_chart_type: str | None = None,
) -> dict:
    """Phase: Extract chart dataset from research.

    Injects the best-matching chart template's design guidance, requires a
    minimum point density with a worked example, and re-drafts once if the
    first attempt is empty/thin/all-zero (the local-model failure mode).
    """
    from augmentum.tools.artifact_templates import get_pipeline_template_context
    from augmentum.tools.artifact_validate import (
        MIN_CHART_POINTS,
        chart_quality,
        chart_repair_user,
    )

    research_context = "\n\n".join(
        r.content for r in inventory.existing_research[:15]
    )
    template_ctx = get_pipeline_template_context(inventory.topic, "chart")

    system = (
        "You extract chart data from research. Output ONLY valid JSON — no "
        "markdown, no prose. Pick the best chart_type for the data.\n"
        + (template_ctx + "\n\n" if template_ctx else "")
        + "Data requirements:\n"
        f"- Provide AT LEAST {MIN_CHART_POINTS} category labels (more when the "
        "data supports it).\n"
        "- Every dataset must give one real number for EVERY label "
        "(equal-length arrays).\n"
        "- Use actual figures from the data; if one is missing, infer a "
        "reasonable value — never empty arrays, all-zero series, or placeholders."
    )
    user = (
        f"Topic: {inventory.topic}\n\n"
        f"Data:\n{research_context}\n\n"
        "Output JSON exactly in this shape (fill with real data):\n"
        '{"chart_type": "bar", "x_label": "Quarter", "y_label": "Revenue ($M)", '
        '"labels": ["Q1", "Q2", "Q3", "Q4"], '
        '"datasets": [{"name": "2025", "values": [12.4, 15.1, 18.9, 22.3]}]}'
    )

    t0 = time.monotonic()
    raw = await llm_caller(system, user)
    chart = _parse_chart_json(raw)
    if chart is None:
        log.warning("chart_dataset_parse_failed")

    # Validate-and-repair: re-draft once when degenerate.
    quality = chart_quality(
        (chart or {}).get("labels", []), (chart or {}).get("datasets", []),
    )
    if chart is None or quality.needs_repair:
        reason = quality.reason if chart is not None else "invalid JSON"
        log.info("chart_dataset_repair", reason=reason)
        sys2, user2 = chart_repair_user(inventory.topic, research_context, reason)
        raw2 = await llm_caller(sys2, user2)
        repaired = _parse_chart_json(raw2)
        if repaired is not None:
            rq = chart_quality(repaired.get("labels", []), repaired.get("datasets", []))
            if not rq.degenerate or chart is None:
                chart = repaired

    if chart is None:
        chart = {"chart_type": "bar", "labels": [], "datasets": [], "x_label": "", "y_label": ""}

    # User override for chart type
    if explicit_chart_type:
        chart["chart_type"] = explicit_chart_type

    log.info("draft_chart_complete", elapsed_s=round(time.monotonic() - t0, 2),
             labels=len(chart.get("labels") or []))
    return chart


# ---------------------------------------------------------------------------
# Phase 5: Illustrate
# ---------------------------------------------------------------------------


async def run_illustrate(
    sections: list[dict],
    existing_images: list[ImageRef],
    image_search_tool,
    llm_caller: LLMCaller,
    max_searches: int = 3,
) -> dict[str, str | None]:
    """Phase: Find images for sections/slides and build heading->URL mapping."""
    t0 = time.monotonic()

    # Pool existing images
    pool = {img.url: img.description or img.url for img in existing_images}

    # Search for additional images if pool is thin
    searches_done = 0
    for section in sections:
        if searches_done >= max_searches:
            break
        heading = section.get("heading", section.get("title", ""))
        if not heading:
            continue

        try:
            result = await asyncio.wait_for(
                image_search_tool.execute(query=heading, count=1),
                timeout=10,
            )
            if result.success and result.metadata.get("url"):
                url = result.metadata["url"]
                pool[url] = f"Image for: {heading}"
                searches_done += 1
        except (TimeoutError, Exception) as exc:
            log.warning("illustrate_search_failed", heading=heading, error=str(exc))

    if not pool:
        return {}

    # LLM matching
    headings = [s.get("heading", s.get("title", "")) for s in sections]
    pool_desc = "\n".join(f"- {url}: {desc}" for url, desc in pool.items())

    try:
        raw = await llm_caller(
            "Match images to sections. Output ONLY a JSON object mapping section heading to image URL or null.",
            f"Sections: {json.dumps(headings)}\n\nAvailable images:\n{pool_desc}\n\n"
            f"Output: {{\"Section Heading\": \"image_url_or_null\", ...}}",
        )
        clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        mapping = json.loads(clean)
        if not isinstance(mapping, dict):
            mapping = {}
    except (json.JSONDecodeError, Exception) as exc:
        log.warning("illustrate_matching_failed", error=str(exc))
        mapping = {}

    log.info("illustrate_complete",
             pool_size=len(pool), mapped=sum(1 for v in mapping.values() if v),
             elapsed_s=round(time.monotonic() - t0, 2))
    return mapping


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SECTION_MARKER = re.compile(r"^##\s+SECTION:\s*(.+)", re.MULTILINE)
_HEADING_MARKER = re.compile(r"^##\s+(.+)", re.MULTILINE)
_SLIDE_MARKER = re.compile(r"^###\s+Slide\s+\d+:\s*(.+)", re.MULTILINE)
_SLIDE_HEADING = re.compile(r"^###\s+(.+)", re.MULTILINE)


def parse_draft_sections(draft: str, fallback_title: str = "Document") -> list[dict]:
    """Parse draft text into section dicts."""
    # Strategy 1: ## SECTION: markers
    markers = list(_SECTION_MARKER.finditer(draft))
    if markers:
        sections = []
        for i, m in enumerate(markers):
            heading = m.group(1).strip()
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(draft)
            body = draft[start:end].strip()
            sections.append({"heading": heading, "body": body, "level": 1})
        return sections

    # Strategy 2: ## heading markers
    markers = list(_HEADING_MARKER.finditer(draft))
    if markers:
        sections = []
        for i, m in enumerate(markers):
            heading = m.group(1).strip()
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(draft)
            body = draft[start:end].strip()
            sections.append({"heading": heading, "body": body, "level": 1})
        return sections

    # Strategy 3: whole draft as one section
    return [{"heading": fallback_title, "body": draft.strip(), "level": 1}]


def parse_draft_slides(draft: str, fallback_title: str = "Presentation") -> list[dict]:
    """Parse draft text into slide dicts."""
    markers = list(_SLIDE_MARKER.finditer(draft))
    if not markers:
        markers = list(_SLIDE_HEADING.finditer(draft))
    if not markers:
        # Marker-less fallback. Split the body on blank lines so the deck
        # gets some shape instead of one giant blob, and log loudly so we
        # know the drafter ignored the marker contract.
        log.warning(
            "parse_draft_slides_no_markers_collapsing",
            draft_chars=len(draft),
            fallback_title=fallback_title,
        )
        return _split_markerless_draft(draft, fallback_title)

    slides = []
    for i, m in enumerate(markers):
        title = m.group(1).strip()
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(draft)
        block = draft[start:end].strip()

        # Extract notes
        notes = ""
        notes_match = re.search(r"\*\*Notes?:\*\*\s*(.*?)$", block, re.DOTALL | re.MULTILINE)
        if notes_match:
            notes = notes_match.group(1).strip()
            block = block[:notes_match.start()].strip()

        layout = "title" if i == 0 and len(block) < 100 else "content"
        slides.append({"layout": layout, "title": title, "body": block, "notes": notes})

    return slides


def _split_markerless_draft(draft: str, fallback_title: str) -> list[dict]:
    """Cut a marker-less draft into multiple slides on paragraph breaks.

    A one-slide collapse (title + entire-blob body) is almost always
    wrong — the drafter meant a deck but skipped the grammar. Splitting
    on blank lines gives the user something resembling a presentation
    until they re-run with a stricter prompt.
    """
    cleaned = (draft or "").strip()
    if not cleaned:
        return [{
            "layout": "content",
            "title": fallback_title,
            "body": "",
            "notes": "",
        }]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if len(paragraphs) <= 1:
        return [{
            "layout": "content",
            "title": fallback_title,
            "body": cleaned,
            "notes": "",
        }]

    slides: list[dict] = [{
        "layout": "title",
        "title": fallback_title,
        "body": paragraphs[0][:200],
        "notes": "",
    }]
    for i, para in enumerate(paragraphs[1:], start=1):
        first_line, _, rest = para.partition("\n")
        title = first_line.strip().lstrip("#").strip() or f"{fallback_title} ({i})"
        body = rest.strip() or first_line.strip()
        slides.append({
            "layout": "content",
            "title": title[:120],
            "body": body,
            "notes": "",
        })
    return slides


# ---------------------------------------------------------------------------
# Image-pick projection (pipeline side — mirrors handler._apply_image_picks)
# ---------------------------------------------------------------------------

def _apply_pipeline_image_picks(
    slides: list[dict],
    picks: dict[int, dict],
    candidates: dict[int, list[dict]],
) -> list[dict]:
    """Project Illustrate Slides picks onto pipeline-parsed slides.

    Same shape as ``_apply_image_picks`` in the handler — kept here so
    the pipeline can run without importing from the agentic handler
    (which would create a layering inversion). Indices are 1-based to
    match the crafter contract.
    """
    if not picks or not candidates:
        return slides

    def _candidate(slide_idx: int, cid: str) -> dict | None:
        pool = candidates.get(slide_idx) or candidates.get(str(slide_idx)) or []
        for c in pool:
            if c.get("candidate_id") == cid:
                return c
        return None

    out: list[dict] = []
    for i, slide in enumerate(slides):
        slide_idx = i + 1
        pick = picks.get(slide_idx) or picks.get(str(slide_idx))
        if not pick:
            out.append(slide)
            continue
        new_slide = dict(slide)
        primary_id = pick.get("primary", "")
        primary = _candidate(slide_idx, primary_id) if primary_id else None
        if primary and primary.get("embed_url"):
            new_slide["image_url"] = primary["embed_url"]
        additional_urls = []
        for cid in (pick.get("additional") or [])[:3]:
            c = _candidate(slide_idx, cid)
            if c and c.get("embed_url"):
                additional_urls.append(c["embed_url"])
        if additional_urls:
            new_slide["additional_images"] = additional_urls
        out.append(new_slide)
    return out


# ---------------------------------------------------------------------------
# Assemble phases
# ---------------------------------------------------------------------------

def assemble_sections(
    sections: list[dict],
    image_map: dict[str, str | None],
    research: list[ResearchItem],
    title: str,
    tool_params: dict,
) -> dict:
    """Assemble final document tool input from parsed sections + images + references."""
    for section in sections:
        heading = section["heading"]
        img = image_map.get(heading)
        if img:
            section["image_url"] = img

    # Add references section
    source_urls = list(dict.fromkeys(r.source_url for r in research if r.source_url))
    if source_urls:
        refs_body = "\n".join(f"- {url}" for url in source_urls)
        sections.append({"heading": "References", "body": refs_body, "level": 1})

    return {
        "title": title,
        "format": tool_params.get("format", "pdf"),
        "author": tool_params.get("author", ""),
        "theme": tool_params.get("theme", ""),
        "sections": sections,
        **{k: v for k, v in tool_params.items() if k not in ("format", "author", "theme", "title", "sections")},
    }


def assemble_slides(
    slides: list[dict],
    image_map: dict[str, str | None],
    research: list[ResearchItem],
    title: str,
    tool_params: dict,
) -> dict:
    """Assemble final presentation tool input from parsed slides + images."""
    for slide in slides:
        img = image_map.get(slide.get("title", ""))
        if img:
            slide["image_url"] = img

        # Add source URLs to speaker notes
        source_urls = [r.source_url for r in research if r.source_url]
        if source_urls and slide.get("notes"):
            slide["notes"] += "\n\nSources: " + ", ".join(source_urls[:3])

    return {
        "title": title,
        "subtitle": tool_params.get("subtitle", ""),
        "author": tool_params.get("author", ""),
        "theme": tool_params.get("theme", ""),
        "slides": slides,
        **{k: v for k, v in tool_params.items() if k not in ("title", "subtitle", "author", "theme", "slides")},
    }


def assemble_sheets(
    sheets: list[dict],
    research: list[ResearchItem],
    title: str,
    tool_params: dict,
) -> dict:
    """Assemble final spreadsheet tool input."""
    # Add sources sheet if research has URLs
    source_urls = list(dict.fromkeys(r.source_url for r in research if r.source_url))
    if source_urls:
        sheets.append({
            "name": "Sources",
            "headers": ["URL"],
            "rows": [[url] for url in source_urls],
            "freeze_header": True,
            "column_formats": {},
            "summary_row": "none",
        })

    return {
        "title": title,
        "theme": tool_params.get("theme", ""),
        "sheets": sheets,
        **{k: v for k, v in tool_params.items() if k not in ("title", "theme", "sheets")},
    }


def assemble_chart(
    chart_data: dict,
    title: str,
    tool_params: dict,
) -> dict:
    """Assemble final chart tool input.

    Threads the document theme + presentation hints (value_format, sort,
    subtitle, caption) into the render tool so build-mode charts match the
    surrounding artifact's brand and format numbers professionally. The
    drafting LLM may set value_format/sort/subtitle; otherwise the renderer
    auto-detects format from the axis labels.
    """
    _reserved = {
        "title", "chart_type", "labels", "datasets", "x_label", "y_label",
        "show_values", "theme", "value_format", "sort", "subtitle", "caption",
    }
    return {
        "title": title,
        "subtitle": chart_data.get("subtitle", tool_params.get("subtitle", "")),
        "chart_type": chart_data.get("chart_type", "bar"),
        "labels": chart_data.get("labels", []),
        "datasets": chart_data.get("datasets", []),
        "x_label": chart_data.get("x_label", ""),
        "y_label": chart_data.get("y_label", ""),
        "show_values": tool_params.get("show_values", False),
        "theme": tool_params.get("theme", ""),
        "value_format": chart_data.get("value_format", tool_params.get("value_format", "auto")),
        "sort": chart_data.get("sort", tool_params.get("sort", "none")),
        "caption": chart_data.get("caption", tool_params.get("caption", "")),
        **{k: v for k, v in tool_params.items() if k not in _reserved},
    }


# ---------------------------------------------------------------------------
# Format strategies
# ---------------------------------------------------------------------------

_IMAGE_DOWNLOAD_TIMEOUT = 10


async def download_web_image(url: str, *, _test_bytes: bytes | None = None) -> str | None:
    """Download a web image to a temp file. Returns local path.

    Internal /api/ URLs are returned unchanged.
    """
    if url.startswith("/api/"):
        return url
    if _test_bytes is not None:
        suffix = os.path.splitext(url.split("?")[0])[-1] or ".jpg"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.write(fd, _test_bytes)
        os.close(fd)
        return path
    try:
        import httpx

        async with httpx.AsyncClient(timeout=_IMAGE_DOWNLOAD_TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                log.warning("image_download_failed", url=url, status=resp.status_code)
                return None
            suffix = os.path.splitext(url.split("?")[0])[-1] or ".jpg"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.write(fd, resp.content)
            os.close(fd)
            return path
    except Exception as exc:
        log.warning("image_download_error", url=url, error=str(exc))
        return None


_FORMAT_TO_TOOL = {
    "pdf": "create_document",
    "docx": "create_document",
    "pptx": "create_presentation",
    "xlsx": "create_spreadsheet",
    "chart": "create_chart",
}


async def execute_artifact_pipeline(
    request: ArtifactRequest,
    context: PipelineContext,
    llm_caller: LLMCaller,
    progress_cb: Callable | None = None,
    *,
    _search_tool=None,
    _fetch_tool=None,
    _image_search_tool=None,
    _render_tools: dict | None = None,
) -> ArtifactResult:
    """Run the pipeline, capturing its internal LLM generation (scan → draft →
    illustrate → assemble) as its OWN ``:W`` builder training row.

    The artifact creator is a generative SUB-AGENT, not the orchestrator's
    reasoning: a ``force_new`` capture scope keeps its multi-step generation
    out of the calling chat/agentic trace (so that row shows a tool_call +
    result reference, not the inline document body) AND records it as a
    standalone builder row teaching "given a build spec, produce the artifact."
    Capture-only; never alters the build.
    """
    from augmentum.training.trace_context import begin_capture, end_capture

    _w_ctx, _w_tok = begin_capture(
        user_id=getattr(context, "user_id", "") or "",
        mode="builder",
        force_new=True,
    )
    try:
        return await _execute_artifact_pipeline_inner(
            request, context, llm_caller, progress_cb,
            _search_tool=_search_tool, _fetch_tool=_fetch_tool,
            _image_search_tool=_image_search_tool, _render_tools=_render_tools,
        )
    finally:
        end_capture(_w_ctx, _w_tok)


async def _execute_artifact_pipeline_inner(
    request: ArtifactRequest,
    context: PipelineContext,
    llm_caller: LLMCaller,
    progress_cb: Callable | None = None,
    *,
    _search_tool=None,
    _fetch_tool=None,
    _image_search_tool=None,
    _render_tools: dict | None = None,
) -> ArtifactResult:
    """Execute the full artifact assembly pipeline.

    _search_tool, _fetch_tool, _image_search_tool, _render_tools are
    injected for testing. In production, resolved from app state.
    """
    t0 = time.monotonic()
    fmt = request.format
    title = request.title or request.topic[:80]

    log.info("artifact_pipeline_start", format=fmt, topic=request.topic[:100])

    # --- Phase 1: Context Scan ---
    inventory = await scan_context(request, context, llm_caller)

    # --- Phase 2+: Format-specific strategy ---
    tool_name = _FORMAT_TO_TOOL.get(fmt, "create_document")
    render_tool = (_render_tools or {}).get(tool_name)

    if fmt in ("pdf", "docx"):
        # Research -> Draft -> Illustrate -> Assemble -> Render
        if inventory.needs_research and _search_tool:
            await run_research(inventory, _search_tool, _fetch_tool)

        draft = await draft_sections(inventory, llm_caller)
        sections = parse_draft_sections(draft, fallback_title=title)

        image_map: dict[str, str | None] = {}
        if inventory.needs_images and _image_search_tool:
            image_map = await run_illustrate(
                sections, inventory.existing_images, _image_search_tool, llm_caller
            )

        # Download web images to local paths for render tool compatibility
        resolved_image_map: dict[str, str | None] = {}
        for heading, img_url in image_map.items():
            if img_url and img_url.startswith("http"):
                local_path = await download_web_image(img_url)
                resolved_image_map[heading] = local_path
            else:
                resolved_image_map[heading] = img_url

        assembled = assemble_sections(sections, resolved_image_map, inventory.existing_research, title, {**request.tool_params, "format": fmt})

    elif fmt == "pptx":
        # Research -> Draft Slides -> Illustrate -> Assemble -> Render
        if inventory.needs_research and _search_tool:
            await run_research(inventory, _search_tool, _fetch_tool)

        draft = await draft_slides(inventory, llm_caller)
        slides = parse_draft_slides(draft, fallback_title=title)

        # If the caller already ran the new Illustrate Slides step, honour
        # those picks and skip pipeline-internal illustration entirely.
        picks = getattr(context, "slide_image_picks", None) or {}
        candidates = getattr(context, "image_candidates", None) or {}
        if picks and candidates:
            slides = _apply_pipeline_image_picks(slides, picks, candidates)
            assembled = assemble_slides(
                slides, {}, inventory.existing_research, title,
                request.tool_params,
            )
        else:
            image_map = {}
            if inventory.needs_images and _image_search_tool:
                image_map = await run_illustrate(
                    [{"heading": s["title"], "body": s["body"]} for s in slides],
                    inventory.existing_images, _image_search_tool, llm_caller,
                )

            # Download web images to local paths for render tool compatibility
            resolved_image_map: dict[str, str | None] = {}
            for heading, img_url in image_map.items():
                if img_url and img_url.startswith("http"):
                    local_path = await download_web_image(img_url)
                    resolved_image_map[heading] = local_path
                else:
                    resolved_image_map[heading] = img_url

            assembled = assemble_slides(slides, resolved_image_map, inventory.existing_research, title, request.tool_params)

    elif fmt == "xlsx":
        # Research Data -> Structure Rows -> Assemble -> Render
        if _search_tool:
            await run_research_data(inventory, _search_tool, _fetch_tool)

        sheets = await draft_sheet_structure(inventory, llm_caller)
        assembled = assemble_sheets(sheets, inventory.existing_research, title, request.tool_params)

    elif fmt == "chart":
        # Research Data -> Assemble Dataset -> Render
        if _search_tool:
            await run_research_data(inventory, _search_tool, _fetch_tool)

        chart_data = await draft_chart_dataset(
            inventory, llm_caller,
            explicit_chart_type=request.tool_params.get("chart_type"),
        )
        # Charts adopt the artifact's theme so embedded charts match the
        # surrounding document's brand color.
        chart_params = {**request.tool_params}
        chart_params.setdefault("theme", request.theme or "")
        assembled = assemble_chart(chart_data, title, chart_params)

    else:
        log.warning("unknown_artifact_format", format=fmt)
        assembled = {"title": title, "format": fmt}

    # --- Render ---
    if not render_tool:
        raise ValueError(f"No render tool available for {tool_name}")

    # Thread the pipeline's user_id into the render tool's kwargs so
    # ArtifactStore.save (which requires user_id on every save) gets a
    # real owner via Tool.extract_user_id. Without this the agentic
    # build path raised "ArtifactStore.save requires a user_id".
    if context.user_id:
        assembled.setdefault("_user_id", context.user_id)

    result = await render_tool.execute(**assembled)

    if not result.success:
        raise RuntimeError(f"Render failed: {result.error}")

    meta = dict(result.metadata or {})
    if isinstance(getattr(result, "card", None), dict):
        meta["card"] = result.card
    elapsed = time.monotonic() - t0
    log.info("artifact_pipeline_complete",
             format=fmt, artifact_id=meta.get("id", ""),
             elapsed_s=round(elapsed, 2))

    return ArtifactResult(
        artifact_id=meta.get("id", ""),
        download_url=meta.get("download_url", ""),
        display_name=meta.get("display_name", title),
        metadata=meta,
        source_json=assembled,
    )
