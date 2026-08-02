"""Server-Side Orchestrated Search (SSOS) for passthrough mode.

Executes tools directly based on heuristic intent classification —
no LLM involvement in tool selection, query formulation, or execution.
The LLM is used only for final synthesis of gathered results.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.tools.base import invoke_tool
from augmentum.tools.intent import QueryIntent, classify_intent
from augmentum.tools.query_formulator import formulate_queries
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)


@dataclass
class _EventEmitter:
    """Bundles the three SSOS event callbacks so each ``_execute_*`` can
    fire ``tool_start`` / ``tool_complete`` without a callback-soup
    parameter list. All callbacks are optional; missing ones are no-ops.

    Each tool gets a fresh call_id so the UI can pair start↔complete
    events when multiple tools fire in one orchestration (rare today
    but the contract holds for future composite intents).
    """

    on_tool_start: Callable[[str, str, dict], Awaitable[None]] | None = None
    on_tool_complete: Callable[[str, bool, str, dict, str, int], Awaitable[None]] | None = None
    on_status: Callable[[str], Awaitable[None]] | None = None
    _active: dict[str, str] = field(default_factory=dict)

    async def status(self, stage: str) -> None:
        if self.on_status:
            await self.on_status(stage)

    async def tool_start(self, tool_name: str, args: dict) -> str:
        tc_id = f"ssos_{uuid.uuid4().hex[:8]}"
        self._active[tool_name] = tc_id
        if self.on_tool_start:
            await self.on_tool_start(tc_id, tool_name, args)
        return tc_id

    async def tool_complete(
        self, tool_name: str, *, success: bool, snippet: str,
        metadata: dict, dur_ms: int,
    ) -> None:
        tc_id = self._active.pop(tool_name, "")
        if self.on_tool_complete:
            await self.on_tool_complete(
                tool_name, success, snippet, metadata, tc_id, dur_ms,
            )

@dataclass
class _ExecResult:
    """ToolResult-shaped view over the handler executor's (output,
    metadata) return, so every SSOS call site keeps its existing
    ``result.success / .output / .metadata / .error`` logic while the
    actual execution routes through ONE executor (see _run_tool)."""

    success: bool
    output: str
    metadata: dict
    error: str = ""


# Extract code from fenced code blocks
_FENCED_CODE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _extract_fenced_code(text: str) -> str | None:
    """Extract code from the first fenced code block."""
    m = _FENCED_CODE_RE.search(text)
    return m.group(1).strip() if m else None


# Minimum confidence to use SSOS instead of falling back to tool-calling
_CONFIDENCE_THRESHOLD = 0.5

_SYNTHESIS_INSTRUCTION = (
    "Answer the user's question using ONLY the search results above. "
    "Be thorough and detailed. Include specific facts, figures, dates, and sources. "
    "All information must remain faithful to the search results without alteration — "
    "do not change dates, numbers, names, or any factual details. "
    "Cite sources inline using [1], [2], etc. matching the result numbers. "
    "End with a Sources section listing each cited URL."
)


# ----------------------------------------------------------------------
# Model-initiated capabilities ("soft tool triggering")
#
# Unlike the heuristic SSOS fast-path above (server classifies intent and
# FORCES a tool), these let the *model* decide. When Auto is on and no
# deterministic heuristic fired, we inject a compact hint describing these
# capabilities and the trigger protocol, then let the model emit a marker as
# its first visible line if it wants one. No function-calling schemas — works
# on small local models. See docs/.../soft-tool-triggering.
#
# This is deliberately a general, registry-driven primitive: adding a new
# model-callable capability later (Phase 2 gated tools, Phase 3 inline
# representations) is one ModelCapability entry, not a new code path.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapability:
    """A native tool the model may reach for via the in-band trigger protocol.

    ``kind`` selects the dispatch shape:
      * ``lookup``  — run the tool, then a synthesis pass over the result
                      (Phase 1: data-returning lookups).
      * ``gated``   — propose to the user, await Approve/Skip (Phase 2).
      * ``augment`` — render an artifact alongside the reply (Phase 3).
    """

    name: str          # trigger name the model emits, e.g. "web_search"
    tool: str          # registry tool name to execute
    kind: str          # "lookup" | "gated" | "augment"
    primary_arg: str   # free text after the marker maps to this kwarg
    fallback_hint: str  # used if the tool exposes no model_hint
    extra_kwargs: tuple[tuple[str, object], ...] = ()  # static execute kwargs
    synthesis: str = "default"  # which synthesis instruction to apply
    # Gated capabilities only: the tool needs STRUCTURED input (title + a list
    # of sections), so the single-string ``args`` is a brief that a planner
    # expands into the tool's full input before the confirmation offer.
    needs_plan: bool = False


# Phase 1 wires only ``lookup`` capabilities. The four data-returning tools
# the model now drives instead of the retired regex `search` heuristic.
_CAPABILITIES: tuple[ModelCapability, ...] = (
    ModelCapability(
        name="web_search", tool="web_search", kind="lookup",
        primary_arg="query",
        fallback_hint="Search the web for current events, news, or facts you're unsure of.",
        extra_kwargs=(("num_results", 5),), synthesis="search",
    ),
    # The read half of web research. web_search returns snippets; web_fetch
    # opens ONE page and returns its full text — the only way to get tables,
    # day-by-day records, or any verified detail that lives on the page rather
    # than in a two-line search result. Without this exposed, a model that
    # finds the authoritative source (e.g. an NWS climate page) can't read it,
    # so it re-searches in circles. Pairs with web_search; kind=lookup so it
    # runs inline and feeds back into the loop.
    ModelCapability(
        name="web_fetch", tool="web_fetch", kind="lookup",
        primary_arg="url",
        fallback_hint=(
            "Read the full contents of a specific web page by URL. Use it after "
            "web_search to open a promising result for the details, tables, or "
            "exact figures that snippets don't include."
        ),
        synthesis="search",
    ),
    ModelCapability(
        name="wikipedia", tool="wikipedia", kind="lookup",
        primary_arg="query",
        fallback_hint="Look up encyclopedic facts: history, biographies, science, definitions.",
        extra_kwargs=(("num_results", 3),), synthesis="wikipedia",
    ),
    ModelCapability(
        name="youtube", tool="youtube", kind="lookup",
        primary_arg="query",
        fallback_hint="Find a video, or pass a YouTube URL to read its transcript.",
        synthesis="youtube",
    ),
    ModelCapability(
        name="image_search", tool="image_search", kind="lookup",
        primary_arg="query",
        fallback_hint="Find photos or pictures from the web.",
        extra_kwargs=(("count", 4),), synthesis="image_search",
    ),
    # Exact calculation + math verification. A lookup (runs + feeds the result
    # back), not gated — verifying arithmetic should be silent and instant, with
    # no confirm chip. The shorthand harness in python_exec means the model can
    # pass a bare expression ("17*23", "sqrt(2)*100") and get the printed answer,
    # so it can check its own math mid-reasoning instead of doing it in its head.
    ModelCapability(
        name="python_exec", tool="python_exec", kind="lookup",
        primary_arg="code",
        fallback_hint=(
            "Run Python for exact calculation, math verification, or data/text "
            "crunching. Pass a bare expression (e.g. `17*23`, `sqrt(2)*100`) and "
            "it's evaluated and printed with math functions preloaded. Prefer "
            "this over mental arithmetic for anything non-trivial."
        ),
        synthesis="default",
    ),
)

# Phase 2 — GATED capabilities. Heavy / costly / long-running tools the model
# may *request* but never fire unilaterally: instead of running, we surface a
# confirmation offer (the existing offer-chip with Accept / Not now / Never), so
# the user always gets an exit when the intent was inferred wrong. The model
# recognizes desire by emitting the same ``[[tool:NAME]]`` marker; the server
# turns that into an offer rather than an execution. Only single-string-arg
# tools fit the marker cleanly (create_ebook needs a topic→outline planner —
# follow-up). These run in Auto mode, where intent is INFERRED; the explicit
# tool checkboxes already carry the user's consent and run immediately.
_GATED_CAPABILITIES: tuple[ModelCapability, ...] = (
    ModelCapability(
        name="image_generation", tool="image_generation", kind="gated",
        primary_arg="prompt",
        fallback_hint=(
            "Generate an image from a description. Ask the user first — "
            "it costs GPU time, so propose it rather than assuming."
        ),
    ),
    ModelCapability(
        name="build_application", tool="build_application", kind="gated",
        primary_arg="description",
        fallback_hint=(
            "Build a complete web app from a description. Ask first — it "
            "spins up a workspace and runs for a few minutes."
        ),
    ),
    # Structured creators — the marker carries a one-line brief; a planner
    # expands it into the tool's full input and the chip shows the outline.
    ModelCapability(
        name="create_ebook", tool="create_ebook", kind="gated",
        primary_arg="brief", needs_plan=True,
        fallback_hint=(
            "Write and illustrate a complete ebook from a one-line idea. "
            "Ask first — it's a long task; the user confirms the outline."
        ),
    ),
    ModelCapability(
        name="create_presentation", tool="create_presentation", kind="gated",
        primary_arg="brief", needs_plan=True,
        fallback_hint=(
            "Build a slide deck from a one-line topic. Ask first — the user "
            "confirms the outline before it's generated."
        ),
    ),
    ModelCapability(
        name="create_document", tool="create_document", kind="gated",
        primary_arg="brief", needs_plan=True,
        fallback_hint=(
            "Write a structured document from a one-line brief. Ask first — "
            "the user confirms the outline."
        ),
    ),
    # Data visualisation / tabulation (2026-07-26). Listed as ``gated`` for
    # registry membership only — ``_should_gate_capability`` runs BOTH inline,
    # so the model calls the real tool with its real structured schema and the
    # result lands in the reply. They were the only artifact creators missing
    # from Auto, so a data-shaped answer had no chart schema on the roster and
    # the model described a chart in prose instead of drawing one. The
    # research→dataset→render chain they need already exists
    # (``artifact_pipeline`` ``fmt == "chart"`` / ``"xlsx"`` branches).
    #
    # ``needs_plan`` stays False: the INLINE branch exposes the real tool, so
    # the model supplies ``labels``/``datasets`` (or ``sheets``) directly and
    # there is no one-line brief for a planner to expand. If either is ever
    # re-gated to PROPOSE, the single-arg proxy would hand it ``{brief: ...}``,
    # which does NOT match these schemas — add a ``gated_planner`` PlanSpec
    # first.
    ModelCapability(
        name="create_chart", tool="create_chart", kind="gated",
        primary_arg="brief",
        fallback_hint=(
            "Draw a chart (bar, line, pie, scatter, area) from data and show "
            "it in the reply. Use it whenever numbers are easier to read as a "
            "picture — comparisons, trends over time, breakdowns of a whole. "
            "Pass real labels and values, never placeholders."
        ),
    ),
    ModelCapability(
        name="create_spreadsheet", tool="create_spreadsheet", kind="gated",
        primary_arg="brief",
        fallback_hint=(
            "Build an .xlsx spreadsheet with real headers and rows. Use it "
            "when the user wants data they can sort, filter, or keep."
        ),
    ),
)

_CAPABILITIES_BY_NAME: dict[str, ModelCapability] = {
    c.name: c for c in (*_CAPABILITIES, *_GATED_CAPABILITIES)
}

# First-visible-line trigger: ``[[tool:NAME]] free text args``. Distinct from
# every thinking.py delimiter (<think>, <|channel|>, [THINK], <|channel>thought)
# — none start with "[[tool:" — and parsed on the post-thinking visible stream.
_TRIGGER_RE = re.compile(r"^\s*\[\[tool:([a-z_]+)\]\]\s*(.*)$", re.IGNORECASE)


class SSOSOrchestrator:
    """Server-side orchestrated search — bypasses LLM tool calling entirely.

    Flow:
    1. Classify intent from user message (heuristics, 0ms)
    2. Formulate search queries (deterministic)
    3. Execute tools directly (parallel)
    4. Build context with consolidated results + citations
    5. Return prepared request for a single LLM synthesis call
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        user_id: str = "",
        app_state: object | None = None,
    ) -> None:
        self._registry = tool_registry
        self._user_id = user_id
        self._app_state = app_state
        # Bound by the owning PassthroughHandler (bind_executor) so
        # every SSOS execution rides the handler's _execute_tool.
        self._executor = None

    async def is_enabled(self) -> bool:
        """Return True if the caller has enabled SSOS for their account.

        Per-user preference at ``user_settings(user_id, ui.autoTools)``,
        falling back to the install-wide ``ui.autoTools`` default if
        present. Default when unset: False — the LLM tool-calling path
        is the more capable default; SSOS is opt-in.
        """
        if not self._user_id or self._app_state is None:
            return False
        store = getattr(self._app_state, "settings_store", None)
        if store is None:
            return False
        val = await store.get_user_or_global(self._user_id, "ui.autoTools")
        return val == "true"

    def bind_executor(self, executor) -> None:
        """Route ALL SSOS tool executions through the handler's
        ``_execute_tool`` (2026-07-02 unification): same param coercion,
        user/session context injection, metrics, ToolResult cards, and
        progress plumbing as the native tool-calling loop — one executor
        and one visual language for tool cards, regardless of which
        brain (heuristic, marker, or native) requested the call."""
        self._executor = executor

    async def _run_tool(self, tool, params: dict) -> _ExecResult:
        """Execute ``tool`` via the bound executor (preferred) or a
        legacy direct call (standalone/test construction). Never raises.

        Executor failure detection leans on _execute_tool's stable
        failure shape (``"Error: …"`` prefix) — it returns text, not a
        ToolResult, by design.
        """
        if tool is None:
            return _ExecResult(False, "", {}, "tool unavailable")
        if self._executor is not None:
            try:
                output, metadata = await self._executor(tool, dict(params))
            except Exception as exc:  # noqa: BLE001 — belt over the executor's own guards
                log.warning("ssos_executor_error",
                            tool=getattr(tool, "name", "?"), exc_info=True)
                return _ExecResult(False, "", {}, str(exc)[:200])
            ok = bool(output) and not str(output).startswith("Error:")
            return _ExecResult(
                ok, output or "", metadata or {},
                "" if ok else (output or "tool failed"),
            )
        # Legacy fallback — minimal context injection so user-scoped
        # tools still work when the orchestrator runs unbound.
        kwargs = dict(params)
        if self._user_id and "_context" not in kwargs:
            import inspect
            ctx: dict = {"user_id": self._user_id}
            fi = getattr(self._app_state, "file_index", None) \
                if self._app_state else None
            if fi is not None:
                ctx["file_index"] = fi
            sig = inspect.signature(tool.execute)
            if "_context" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            ):
                kwargs["_context"] = ctx
        timeout = getattr(tool, "timeout", 30.0)
        if not isinstance(timeout, (int, float)):
            timeout = 30.0  # mocks / exotic tools without a real number
        try:
            result = await asyncio.wait_for(
                invoke_tool(tool, kwargs), timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("ssos_tool_error",
                        tool=getattr(tool, "name", "?"), exc_info=True)
            return _ExecResult(False, "", {}, str(exc)[:200])
        return _ExecResult(
            bool(result.success), result.output or "",
            result.metadata or {}, result.error or "",
        )

    async def try_orchestrate(
        self,
        request: InternalChatRequest,
        *,
        on_tool_start: Callable[[str, str, dict], Awaitable[None]] | None = None,
        on_tool_complete: Callable[[str, bool, str, dict, str, int], Awaitable[None]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> InternalChatRequest | None:
        """Attempt server-side orchestration for the request.

        Returns a prepared InternalChatRequest ready for a single LLM
        synthesis call, or None if SSOS can't handle this query (caller
        should fall back to the LLM tool-calling pipeline).

        When the streaming caller supplies the optional callbacks, SSOS
        emits the same ``tool_start`` / ``tool_complete`` / ``status``
        events the LLM tool-calling path emits, so the UI shows live
        progress (otherwise the user sits in dead-air for the whole
        search → synthesis cycle).
        """
        if not await self.is_enabled():
            return None

        # Extract last user message
        user_msg = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_msg = (msg.content or "").strip()
                break

        if not user_msg:
            return None

        # Step 1: Classify intent
        intent = classify_intent(user_msg)

        if intent.confidence < _CONFIDENCE_THRESHOLD:
            log.debug("ssos_low_confidence", confidence=intent.confidence, action=intent.action)
            return None

        emit = _EventEmitter(
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            on_status=on_status,
        )

        # Step 2: Execute based on action.
        #
        # NOTE: the former ``search`` branch is intentionally gone — web search
        # is now model-initiated (the open-slot regex intent was brittle; see
        # the soft-trigger capabilities above). ``search``/``none`` fall through
        # to ``return None``, and the handler routes them to the model pass.
        # Only the *deterministic* heuristics (URL/math/convert/datetime/code/
        # files) still force a tool here — those are unambiguous and 0-cost.
        if intent.action == "fetch_url":
            await emit.status("fetching")
            result_text = await self._execute_fetch(intent, emit=emit)
        elif intent.action == "calculate":
            result_text = await self._execute_calculate(intent, emit=emit)
        elif intent.action == "convert":
            result_text = await self._execute_convert(user_msg, emit=emit)
        elif intent.action == "datetime":
            result_text = await self._execute_datetime(user_msg, emit=emit)
        elif intent.action == "code":
            await emit.status("running_code")
            result_text = await self._execute_code(user_msg, emit=emit)
        elif intent.action == "files":
            result_text = await self._execute_files(user_msg, emit=emit)
        elif intent.action == "build_app":
            # Skip SSOS for app builder — let the tool-calling path handle it
            # so the user gets progress streaming and the project card UI.
            return None
        else:
            return None

        if not result_text:
            log.warning("ssos_no_results", action=intent.action)
            return None

        # Step 3: Build final message array
        await emit.status("composing")
        return self._build_synthesis_request(request, result_text, intent)

    # ------------------------------------------------------------------
    # Tool executors
    # ------------------------------------------------------------------

    async def _execute_search(
        self, intent: QueryIntent, user_msg: str,
        *, emit: _EventEmitter | None = None,
    ) -> tuple[str | None, list[dict]]:
        """Execute web search with formulated queries.

        Returns ``(formatted_text, structured_results)``. ``formatted_text``
        is the prompt-ready block for synthesis; ``structured_results`` is
        the list of ``{title, url, snippet}`` dicts surfaced to the UI as
        ``result_metadata`` on the ``tool_complete`` event so the chat
        bubble can render source cards without re-parsing prose.
        """
        search_tool = self._registry.get("web_search")
        if not search_tool:
            return None, []

        queries = formulate_queries(intent, user_msg)
        log.info("ssos_search", queries=queries, source_type=intent.source_type)

        if emit:
            await emit.tool_start("web_search", {"queries": queries})

        # Execute all queries in parallel
        start = time.monotonic()
        tasks = [
            self._run_tool(search_tool, {"query": q, "num_results": 5})
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        dur_ms = int((time.monotonic() - start) * 1000)

        # Merge structured results from metadata. Each web_search call
        # returns metadata={"results": [{title, url, snippet, ...}, ...]} —
        # parse that directly instead of re-parsing the formatted output
        # text. Dedup by URL across all parallel calls.
        merged: list[dict] = []
        seen_urls: set[str] = set()
        any_success = False

        for result in results:
            if isinstance(result, Exception):
                log.debug("ssos_search_error", error=str(result))
                continue
            if not result.success:
                continue
            any_success = True
            structured = (result.metadata or {}).get("results", [])
            if not isinstance(structured, list):
                continue
            for entry in structured:
                if not isinstance(entry, dict):
                    continue
                url = (entry.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append({
                    "title": entry.get("title") or "Untitled",
                    "url": url,
                    "snippet": entry.get("snippet") or "",
                })

        if not merged:
            if emit:
                await emit.tool_complete(
                    "web_search", success=any_success,
                    snippet="(no results)" if any_success else "search failed",
                    metadata={}, dur_ms=dur_ms,
                )
            return None, []

        # Render single, renumbered block from the structured list.
        lines: list[str] = []
        for i, r in enumerate(merged, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"    {r['snippet']}")
            lines.append("")
        output = "\n".join(lines).rstrip()

        # Cap total result size to prevent context bloat
        max_chars = settings.tool_result_max_chars
        if len(output) > max_chars:
            truncated = output[:max_chars]
            last_block = truncated.rfind("\n[")
            if last_block > max_chars // 2:
                truncated = truncated[:last_block]
            log.info("ssos_results_truncated", original=len(output), truncated=len(truncated))
            output = truncated

        log.debug("ssos_search_results", chars=len(output), unique_urls=len(merged))

        if emit:
            preview = (
                f"Found {len(merged)} result(s): "
                + ", ".join(r["title"][:60] for r in merged[:3])
            )
            await emit.tool_complete(
                "web_search", success=True, snippet=preview[:240],
                metadata={"results": merged, "result_count": len(merged)},
                dur_ms=dur_ms,
            )

        return output, merged

    async def _execute_fetch(
        self, intent: QueryIntent, *, emit: _EventEmitter | None = None,
    ) -> str | None:
        """Fetch and extract content from a URL."""
        fetch_tool = self._registry.get("web_fetch")
        if not fetch_tool or not intent.url:
            return None

        if emit:
            await emit.tool_start("web_fetch", {"url": intent.url})
        start = time.monotonic()
        try:
            result = await self._run_tool(fetch_tool, {
                "url": intent.url,
                "max_chars": settings.search_autofetch_max_chars,
            })
            dur_ms = int((time.monotonic() - start) * 1000)
            if result.success and result.output:
                if emit:
                    await emit.tool_complete(
                        "web_fetch", success=True,
                        snippet=result.output[:240],
                        metadata={"url": intent.url, **(result.metadata or {})},
                        dur_ms=dur_ms,
                    )
                return f"[1] Content from {intent.url}\n    URL: {intent.url}\n    {result.output}"
            if emit:
                await emit.tool_complete(
                    "web_fetch", success=False,
                    snippet=result.error or "fetch failed",
                    metadata={"url": intent.url}, dur_ms=dur_ms,
                )
        except Exception as exc:
            log.warning("ssos_fetch_error", url=intent.url, exc_info=True)
            if emit:
                await emit.tool_complete(
                    "web_fetch", success=False, snippet=str(exc)[:240],
                    metadata={"url": intent.url},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
        return None

    async def _execute_calculate(
        self, intent: QueryIntent, *, emit: _EventEmitter | None = None,
    ) -> str | None:
        """Execute a calculation."""
        calc_tool = self._registry.get("calculator")
        if not calc_tool or not intent.math_expr:
            return None

        if emit:
            await emit.tool_start("calculator", {"expression": intent.math_expr})
        start = time.monotonic()
        try:
            result = await self._run_tool(
                calc_tool, {"expression": intent.math_expr})
            dur_ms = int((time.monotonic() - start) * 1000)
            if result.success:
                if emit:
                    await emit.tool_complete(
                        "calculator", success=True,
                        snippet=str(result.output)[:240],
                        metadata={"expression": intent.math_expr},
                        dur_ms=dur_ms,
                    )
                return f"Calculation: {intent.math_expr} = {result.output}"
            if emit:
                await emit.tool_complete(
                    "calculator", success=False,
                    snippet=result.error or "calculation failed",
                    metadata={"expression": intent.math_expr},
                    dur_ms=dur_ms,
                )
        except Exception as exc:
            log.warning("ssos_calc_error", expr=intent.math_expr, exc_info=True)
            if emit:
                await emit.tool_complete(
                    "calculator", success=False, snippet=str(exc)[:240],
                    metadata={"expression": intent.math_expr},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
        return None

    async def _execute_convert(
        self, user_msg: str, *, emit: _EventEmitter | None = None,
    ) -> str | None:
        """Execute a unit conversion."""
        converter = self._registry.get("unit_converter")
        if not converter:
            return None

        if emit:
            await emit.tool_start("unit_converter", {"query": user_msg})
        start = time.monotonic()
        try:
            # Pass the full message — the converter tool parses it
            result = await self._run_tool(converter, {"query": user_msg})
            dur_ms = int((time.monotonic() - start) * 1000)
            if result.success:
                if emit:
                    await emit.tool_complete(
                        "unit_converter", success=True,
                        snippet=str(result.output)[:240], metadata={},
                        dur_ms=dur_ms,
                    )
                return f"Conversion: {result.output}"
            if emit:
                await emit.tool_complete(
                    "unit_converter", success=False,
                    snippet=result.error or "conversion failed",
                    metadata={}, dur_ms=dur_ms,
                )
        except Exception as exc:
            log.warning("ssos_convert_error", exc_info=True)
            if emit:
                await emit.tool_complete(
                    "unit_converter", success=False, snippet=str(exc)[:240],
                    metadata={},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
        return None

    async def _execute_datetime(
        self, user_msg: str, *, emit: _EventEmitter | None = None,
    ) -> str | None:
        """Execute a datetime query."""
        dt_tool = self._registry.get("datetime")
        if not dt_tool:
            return None

        if emit:
            await emit.tool_start("datetime", {"action": "now"})
        start = time.monotonic()
        try:
            # Get current time first — almost all datetime queries need it
            result = await self._run_tool(dt_tool, {
                "action": "now", "timezone": settings.timezone or None})
            dur_ms = int((time.monotonic() - start) * 1000)
            if result.success:
                if emit:
                    await emit.tool_complete(
                        "datetime", success=True,
                        snippet=str(result.output)[:240], metadata={},
                        dur_ms=dur_ms,
                    )
                return f"Current date/time: {result.output}"
            if emit:
                await emit.tool_complete(
                    "datetime", success=False,
                    snippet=result.error or "datetime failed",
                    metadata={}, dur_ms=dur_ms,
                )
        except Exception as exc:
            log.warning("ssos_datetime_error", exc_info=True)
            if emit:
                await emit.tool_complete(
                    "datetime", success=False, snippet=str(exc)[:240],
                    metadata={},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
        return None

    async def _execute_code(
        self, user_msg: str, *, emit: _EventEmitter | None = None,
    ) -> str | None:
        """Extract and execute code from a fenced code block."""
        exec_tool = self._registry.get("python_exec")
        if not exec_tool:
            return None

        code = _extract_fenced_code(user_msg)
        if not code:
            return None

        if emit:
            await emit.tool_start("python_exec", {"code_chars": len(code)})
        start = time.monotonic()
        try:
            result = await self._run_tool(exec_tool, {"code": code})
            dur_ms = int((time.monotonic() - start) * 1000)
            if result.success:
                output = result.output or "(no output)"
                if emit:
                    await emit.tool_complete(
                        "python_exec", success=True, snippet=output[:240],
                        metadata={}, dur_ms=dur_ms,
                    )
                return f"Code execution result:\n```\n{output}\n```"
            else:
                error = result.output or "Unknown error"
                if emit:
                    await emit.tool_complete(
                        "python_exec", success=False, snippet=error[:240],
                        metadata={}, dur_ms=dur_ms,
                    )
                return f"Code execution failed:\n```\n{error}\n```"
        except TimeoutError:
            if emit:
                await emit.tool_complete(
                    "python_exec", success=False, snippet="timeout (30s)",
                    metadata={},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
            return "Code execution timed out (30s limit)"
        except Exception as exc:
            log.warning("ssos_code_error", exc_info=True)
            if emit:
                await emit.tool_complete(
                    "python_exec", success=False, snippet=str(exc)[:240],
                    metadata={},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
        return None

    async def _execute_files(
        self, user_msg: str, *, emit: _EventEmitter | None = None,
    ) -> str | None:
        """Search the user's files via the search_files tool."""
        tool = self._registry.get("search_files")
        if not tool:
            return None
        if not self._user_id:
            return None

        file_index = getattr(self._app_state, "file_index", None) if self._app_state else None
        if not file_index:
            return None

        if emit:
            await emit.tool_start("search_files", {"query": user_msg, "limit": 5})
        start = time.monotonic()
        try:
            result = await self._run_tool(
                tool, {"query": user_msg, "limit": 5})
            dur_ms = int((time.monotonic() - start) * 1000)
            if result.success and result.output:
                if emit:
                    await emit.tool_complete(
                        "search_files", success=True,
                        snippet=result.output[:240],
                        metadata=result.metadata or {}, dur_ms=dur_ms,
                    )
                return f"User's files:\n{result.output}"
            if emit:
                await emit.tool_complete(
                    "search_files", success=False,
                    snippet=(result.error if not result.success else "(no results)") or "",
                    metadata={}, dur_ms=dur_ms,
                )
        except Exception as exc:
            log.warning("ssos_files_error", exc_info=True)
            if emit:
                await emit.tool_complete(
                    "search_files", success=False, snippet=str(exc)[:240],
                    metadata={},
                    dur_ms=int((time.monotonic() - start) * 1000),
                )
        return None

    async def _execute_build_app(self, user_msg: str, model: str = "") -> str | None:
        """Execute the application builder tool directly via SSOS."""
        tool = self._registry.get("build_application")
        if not tool:
            return None

        try:
            result = await self._run_tool(
                tool, {"description": user_msg})
            if result.success:
                return result.output
            else:
                return f"Build failed: {result.error}"
        except TimeoutError:
            return "Application build timed out"
        except Exception:
            log.warning("ssos_build_app_error", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_synthesis_request(
        self,
        original: InternalChatRequest,
        result_text: str,
        intent: QueryIntent,
    ) -> InternalChatRequest:
        """Synthesis request for the heuristic fast-path (keyed on intent)."""
        return self._assemble_synthesis(
            original, self._synthesis_context(intent.action, result_text),
        )

    @staticmethod
    def _synthesis_context(kind: str, result_text: str) -> str:
        """Wrap a tool result in the right per-kind synthesis instruction.

        ``kind`` is either a heuristic ``intent.action`` (search/fetch_url/
        code/files/…) or a capability ``synthesis`` label (wikipedia/youtube/
        image_search). Unknown kinds get the generic precise-result framing.
        """
        if kind == "search":
            return (
                f"<search_results>\n{result_text}\n</search_results>"
                f"\n\n{_SYNTHESIS_INSTRUCTION}"
            )
        if kind == "wikipedia":
            return (
                f"<wikipedia>\n{result_text}\n</wikipedia>"
                "\n\nAnswer the user's question using the article(s) above. Stay "
                "faithful to the facts. Cite sources inline using [1], [2], … and "
                "end with a Sources section listing the article URLs."
            )
        if kind == "youtube":
            return (
                f"<video_results>\n{result_text}\n</video_results>"
                "\n\nUse the video result(s) above to answer the user. If a "
                "transcript is present, summarize the key points. If these are "
                "search results, briefly describe the most relevant video(s). "
                "The video cards render separately — don't paste raw URLs."
            )
        if kind == "image_search":
            return (
                f"<image_results>\n{result_text}\n</image_results>"
                "\n\nBriefly introduce the images found for the user's request. "
                "The images render as cards separately — describe what you found "
                "in a sentence or two, don't paste raw URLs or markdown images."
            )
        if kind == "fetch_url":
            return (
                f"<fetched_content>\n{result_text}\n</fetched_content>"
                "\n\nUse the fetched content above to answer the user's question. "
                "Summarize the key information — don't just paste the raw content. "
                "If they asked about a specific topic, focus on that."
            )
        if kind == "code":
            return (
                f"<code_output>\n{result_text}\n</code_output>"
                "\n\nExplain the code output above in context of the user's question. "
                "Interpret the results — what do the numbers or output mean? "
                "If there was an error, explain what went wrong."
            )
        if kind == "files":
            return (
                f"<user_files>\n{result_text}\n</user_files>"
                "\n\nPresent the user's files naturally. Describe what you found — "
                "file names, types, and descriptions. If the user asked for something "
                "specific, highlight the most relevant files. Group by type if there "
                "are many results."
            )
        # Calculator/converter/datetime/tool failures — precise, brief
        return (
            f"<tool_result>\n{result_text}\n</tool_result>"
            "\n\nUse the result above to answer the user's question. "
            "Present the answer clearly. Include the exact value and "
            "add context if it helps (e.g. unit explanations, date details)."
        )

    def _assemble_synthesis(
        self, original: InternalChatRequest, context: str,
    ) -> InternalChatRequest:
        """Build a tools-stripped synthesis request: [system, …history, context].

        Shared by the heuristic path and the model-initiated path. Strips the
        tool-restraint guidance from the system prompt and drops tool schemas
        so the LLM does pure synthesis over ``context``.
        """
        from augmentum.tools.parsing import _TOOL_RESTRAINT

        messages: list[Message] = []

        # System prompt (original, stripped of any tool restraint)
        if original.messages and original.messages[0].role == "system":
            content = original.messages[0].content
            content = content.replace("\n\n" + _TOOL_RESTRAINT, "")
            content = content.replace(_TOOL_RESTRAINT, "")
            messages.append(Message(role="system", content=content))

        # Original conversation (system already handled, skip it)
        for msg in original.messages:
            if msg.role == "system":
                continue
            messages.append(msg)

        messages.append(Message(role="user", content=context))

        return InternalChatRequest(
            model=original.model,
            messages=messages,
            stream=original.stream,
            temperature=original.temperature,
            top_p=original.top_p,
            max_tokens=original.max_tokens,
            stop=original.stop,
            frequency_penalty=original.frequency_penalty,
            presence_penalty=original.presence_penalty,
            seed=original.seed,
            # No tools — LLM just synthesizes
            tools=None,
            format=original.format,
            keep_alive=original.keep_alive,
            raw_options=original.raw_options,
        )

    # ------------------------------------------------------------------
    # Model-initiated capabilities (soft tool triggering)
    # ------------------------------------------------------------------

    @staticmethod
    def lookup_capabilities() -> tuple[ModelCapability, ...]:
        """Capabilities the model may drive in Phase 1 (``kind == 'lookup'``)."""
        return tuple(c for c in _CAPABILITIES if c.kind == "lookup")

    @staticmethod
    def gated_capabilities() -> tuple[ModelCapability, ...]:
        """Heavy capabilities the model may *request* — surfaced as a
        confirmation offer instead of run (Phase 2, ``kind == 'gated'``)."""
        return _GATED_CAPABILITIES

    @staticmethod
    def offerable_capabilities() -> tuple[ModelCapability, ...]:
        """All capabilities advertised to the model (lookups + gated)."""
        return (*SSOSOrchestrator.lookup_capabilities(), *_GATED_CAPABILITIES)

    def build_soft_trigger_hint(self, caps: tuple[ModelCapability, ...] | None = None) -> str:
        """System-prompt block describing the capabilities + trigger protocol.

        Tool descriptions come from each tool's own ``model_hint`` (single
        source of truth), falling back to the descriptor's ``fallback_hint``.
        """
        caps = caps if caps is not None else self.offerable_capabilities()
        lines: list[str] = []
        any_gated = False
        for cap in caps:
            tool = self._registry.get(cap.tool)
            hint = (getattr(tool, "model_hint", "") or "").strip() if tool else ""
            suffix = ""
            if cap.kind == "gated":
                any_gated = True
                suffix = "  (the user will confirm before it runs)"
            lines.append(f"- {cap.name} — {hint or cap.fallback_hint}{suffix}")
        tools_block = "\n".join(lines)
        gated_note = (
            "\n\nSome tools are heavier (marked above): emit the marker to "
            "PROPOSE them when the user clearly wants one — they don't run "
            "until the user taps Accept, so it's safe to offer."
            if any_gated else ""
        )
        return (
            "You can use a tool before answering. If — and only if — a tool "
            "would genuinely help, make your VERY FIRST line EXACTLY:\n"
            "[[tool:NAME]] your query here\n"
            "with nothing else on that line. For lookup tools the result comes "
            "back to you and you answer from it. If no tool is needed, just "
            "answer normally and do not mention tools. Use at most one tool. "
            "Available tools:\n"
            f"{tools_block}{gated_note}"
        )

    def build_decide_request(self, original: InternalChatRequest) -> InternalChatRequest:
        """Clone ``original`` with the soft-trigger hint appended to the system
        prompt. No tool schemas — the model signals in-band instead."""
        hint = self.build_soft_trigger_hint()
        messages: list[Message] = []
        if original.messages and original.messages[0].role == "system":
            base = original.messages[0].content or ""
            messages.append(Message(role="system", content=f"{base}\n\n{hint}".strip()))
            rest = original.messages[1:]
        else:
            messages.append(Message(role="system", content=hint))
            rest = original.messages
        messages.extend(rest)

        return InternalChatRequest(
            model=original.model,
            messages=messages,
            stream=original.stream,
            temperature=original.temperature,
            top_p=original.top_p,
            max_tokens=original.max_tokens,
            stop=original.stop,
            frequency_penalty=original.frequency_penalty,
            presence_penalty=original.presence_penalty,
            seed=original.seed,
            tools=None,
            format=original.format,
            keep_alive=original.keep_alive,
            raw_options=original.raw_options,
        )

    @staticmethod
    def parse_trigger(first_line: str) -> tuple[ModelCapability, str] | None:
        """Parse a ``[[tool:NAME]] args`` first line into (capability, args).

        Returns None on no match, malformed marker, unknown tool, or a tool not
        registered as a Phase-1 lookup — the caller then treats the line as
        ordinary content (never errors).
        """
        if not first_line:
            return None
        m = _TRIGGER_RE.match(first_line)
        if not m:
            return None
        cap = _CAPABILITIES_BY_NAME.get(m.group(1).lower())
        if cap is None or cap.kind != "lookup":
            return None
        args = (m.group(2) or "").strip()
        if not args:
            return None
        return cap, args

    @staticmethod
    def match_trigger(first_line: str) -> tuple[ModelCapability, str] | None:
        """Like ``parse_trigger`` but matches ANY registered capability —
        lookup OR gated. The caller branches on ``cap.kind``: lookups run +
        synthesize; gated capabilities surface a confirmation offer. Returns
        None on no match / malformed / unknown / empty args."""
        if not first_line:
            return None
        m = _TRIGGER_RE.match(first_line)
        if not m:
            return None
        cap = _CAPABILITIES_BY_NAME.get(m.group(1).lower())
        if cap is None or cap.kind not in ("lookup", "gated"):
            return None
        args = (m.group(2) or "").strip()
        if not args:
            return None
        return cap, args

    async def propose_gated(
        self, cap: ModelCapability, args: str, *,
        extra: dict | None = None, reason: str | None = None,
        thread_id: str = "", turn_id: str = "", session_id: str = "",
        mode: str = "passthrough",
    ) -> bool:
        """Surface a confirmation offer for a gated capability instead of
        running it. Reuses the offer substrate (chip with Accept / Not now /
        Never); the catalog accept handler runs the real tool. ``extra`` /
        ``reason`` override the defaults (used by planned tools to carry the
        structured outline). Returns True if the offer surfaced. Best-effort."""
        if self._app_state is None or not self._user_id:
            return False
        sm = getattr(self._app_state, "state_manager", None)
        backend = getattr(sm, "backend", None)
        conn = getattr(backend, "conn", None)
        if conn is None:
            return False
        hub = getattr(self._app_state, "notification_hub", None)
        # Default: the args ARE the "what" (prompt / description) — show them in
        # the chip body. Planned tools pass a structured ``extra`` + an outline
        # ``reason`` instead.
        payload_extra = extra if extra is not None else {
            "args": args, "primary_arg": cap.primary_arg,
        }
        # Carry the originating chat session so the accept handler can thread it
        # into the tool — without it the generated image/artifact lands only in
        # the user's gallery, not inline in the chat it was requested from.
        if session_id and "session_id" not in payload_extra:
            payload_extra = {**payload_extra, "session_id": session_id}
        payload_reason = reason if reason is not None else args[:200]
        try:
            from augmentum.config import settings as _settings
            from augmentum.offers.dispatcher import propose_offer
            res = await propose_offer(
                conn, hub=hub, user_id=self._user_id,
                kind="gated_tool", target_id=cap.tool,
                reason=payload_reason,
                extra=payload_extra,
                thread_id=thread_id, turn_id=turn_id, mode=mode,
                max_per_turn=int(getattr(_settings, "offers_max_per_turn", 2)),
            )
            return bool(res.ok)
        except Exception:
            log.warning("gated_propose_failed", tool=cap.tool, exc_info=True)
            return False

    async def run_named_tool(
        self, cap: ModelCapability, args: str,
        *, emit: _EventEmitter | None = None,
    ) -> tuple[str | None, dict]:
        """Execute a capability's tool with ``args`` mapped to its primary arg.

        Returns ``(synthesis_text, ui_metadata)``. ``synthesis_text`` is None on
        failure (caller still synthesizes a graceful answer). Fires the same
        ``tool_start``/``tool_complete`` events the heuristic path uses.
        """
        tool = self._registry.get(cap.tool)
        kwargs: dict = {cap.primary_arg: args, **dict(cap.extra_kwargs)}
        if emit:
            await emit.tool_start(cap.tool, {cap.primary_arg: args})
        start = time.monotonic()
        result = await self._run_tool(tool, kwargs) if tool else None
        dur_ms = int((time.monotonic() - start) * 1000)

        if result is None or not result.success or not (result.output or "").strip():
            err = (getattr(result, "error", "") if result else "") or "tool unavailable"
            if emit:
                await emit.tool_complete(
                    cap.tool, success=False, snippet=err[:240], metadata={}, dur_ms=dur_ms,
                )
            return None, {}

        ui_meta = {
            k: v for k, v in (result.metadata or {}).items() if not k.startswith("_")
        }
        if emit:
            await emit.tool_complete(
                cap.tool, success=True, snippet=(result.output or "")[:240],
                metadata=ui_meta, dur_ms=dur_ms,
            )
        return result.output, ui_meta

    def build_tool_synthesis_request(
        self, original: InternalChatRequest, cap: ModelCapability, result_text: str,
    ) -> InternalChatRequest:
        """Synthesis request for a model-initiated lookup (keyed on capability)."""
        return self._assemble_synthesis(
            original, self._synthesis_context(cap.synthesis, result_text),
        )

    def build_tool_failure_request(
        self, original: InternalChatRequest, cap: ModelCapability,
    ) -> InternalChatRequest:
        """Synthesis request when the tool failed — answer from knowledge."""
        context = (
            f"(The {cap.name} tool didn't return anything useful.) "
            "Answer the user's question as best you can from your own knowledge. "
            "If you're not sure, say so briefly — don't invent specifics."
        )
        return self._assemble_synthesis(original, context)
