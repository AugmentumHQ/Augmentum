"""Tool chain execution — wave-based multi-step tool orchestration.

Supports both adaptive chains (model-planned) and custom flows
(user-defined). Both feed into the same WaveExecutor which resolves
dependencies and runs independent steps in parallel.
"""

from __future__ import annotations

import asyncio
import json as json_mod
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.base import Tool
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ChainStep:
    """A single step in a tool chain."""

    id: int
    tool: str
    input: dict | None = None  # pre-specified args (may contain {{templates}})
    needs: list[int] = field(default_factory=list)
    reason: str = ""


@dataclass
class ChainPlan:
    """A complete execution plan — list of steps with dependencies."""

    steps: list[ChainStep]
    source: str = "adaptive"  # "adaptive" or "custom:{flow_id}"


@dataclass
class StepResult:
    """Result of executing a single chain step."""

    step_id: int
    tool_name: str
    output: str
    metadata: dict
    success: bool


# ---------------------------------------------------------------------------
# Complexity detection
# ---------------------------------------------------------------------------

_MULTI_STEP_RE = re.compile(
    r"\b(?:then|after that|first\b.*\bthen|next|finally|"
    r"step\s*\d|and\s+also\s+|and\s+then|once\s+(?:you|that|done))\b",
    re.IGNORECASE,
)


def detect_complexity(query: str, matched_tools: list[Tool]) -> bool:
    """Heuristic to decide whether a query needs multi-step chain planning.

    Returns True when the query uses explicit multi-step language
    (e.g. "first... then", "after that", "and then").

    NOTE: Category-based detection was removed because ``matched_tools``
    contains ALL registered tools, not query-relevant ones.  With 20+
    tools spanning 7+ categories the old ``len(categories) >= 2`` check
    fired on every single query, routing simple requests through the
    heavyweight chain planner instead of the simple tool loop.
    """
    if _MULTI_STEP_RE.search(query):
        return True
    return False


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


_TEMPLATE_MAX_CHARS = 10_000


def resolve_templates(input_dict: dict, results: dict[int, StepResult]) -> dict:
    """Replace ``{{step.N.output}}`` / ``{{step.N.metadata.KEY}}`` in values.

    Resolved values are truncated to ``_TEMPLATE_MAX_CHARS`` to prevent
    cost amplification when a tool returns very large output.
    """
    resolved = {}
    for key, value in input_dict.items():
        if isinstance(value, str):
            val = _resolve_string(value, results)
            if len(val) > _TEMPLATE_MAX_CHARS:
                val = val[:_TEMPLATE_MAX_CHARS] + "…[truncated]"
            resolved[key] = val
        else:
            resolved[key] = value
    return resolved


_TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}")


def _resolve_string(text: str, results: dict[int, StepResult]) -> str:
    """Resolve all ``{{...}}`` placeholders in a string."""

    def _replace(match: re.Match) -> str:
        expr = match.group(1).strip()
        parts = expr.split(".")
        if len(parts) >= 3 and parts[0] == "step" and parts[1].isdigit():
            step_id = int(parts[1])
            result = results.get(step_id)
            if not result:
                return match.group(0)  # unresolved — leave as-is
            if parts[2] == "output":
                return result.output
            if parts[2] == "metadata" and len(parts) >= 4:
                val = result.metadata
                for p in parts[3:]:
                    if isinstance(val, dict):
                        val = val.get(p, "")
                    elif isinstance(val, list) and p.isdigit():
                        idx = int(p)
                        val = val[idx] if idx < len(val) else ""
                    else:
                        val = ""
                        break
                return str(val)
        return match.group(0)

    return _TEMPLATE_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Plan parsing — structured JSON (primary)
# ---------------------------------------------------------------------------

# JSON schema for Ollama grammar-constrained decoding.  OpenAI/LlamaCpp
# backends use format="json" which doesn't enforce a schema but the prompt
# instructs the model to produce this shape.
PLAN_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tool": {"type": "string"},
                    "reason": {"type": "string"},
                    "needs": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["id", "tool", "reason"],
            },
        },
    },
    "required": ["steps"],
}


def parse_plan_from_json(
    text: str,
    tool_registry: ToolRegistry,
) -> ChainPlan | None:
    """Parse a JSON plan response into a ChainPlan.

    Handles raw JSON, markdown-fenced JSON, and JSON embedded in prose.
    Returns None if parsing fails or no valid tools are found.
    """
    if not text:
        return None

    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try to find JSON object in the text
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return None

    try:
        data = json_mod.loads(cleaned[json_start:json_end + 1])
    except (json_mod.JSONDecodeError, ValueError):
        return None

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    steps: list[ChainStep] = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            continue
        step_id = s.get("id", i + 1)
        if not isinstance(step_id, int):
            try:
                step_id = int(step_id)
            except (ValueError, TypeError):
                step_id = i + 1

        tool_name = str(s.get("tool", ""))
        reason = str(s.get("reason", ""))
        needs_raw = s.get("needs", [])
        needs = []
        if isinstance(needs_raw, list):
            for n in needs_raw:
                try:
                    needs.append(int(n))
                except (ValueError, TypeError):
                    pass

        # Resolve tool name via registry
        if tool_name:
            resolved = tool_registry.resolve(tool_name)
            tool_name = resolved.name if resolved else ""

        steps.append(ChainStep(
            id=step_id,
            tool=tool_name,
            needs=needs,
            reason=reason,
        ))

    if not steps or not any(s.tool for s in steps):
        return None

    return ChainPlan(steps=steps, source="adaptive")


# ---------------------------------------------------------------------------
# Plan parsing — text fallback
# ---------------------------------------------------------------------------


def _extract_tool_name(
    text: str,
    tool_registry: ToolRegistry,
) -> str:
    """Try to find a valid tool name in *text* using multiple patterns."""
    # Pattern 1: "using tool_name" / "via tool_name" / "**Using**: `tool`"
    for pattern in (
        r"(?:using|via|with|call|tool:?)\s*:?\s*[`\"]?(\w+)[`\"]?",
        r"`(\w+)`",  # backtick-wrapped word
    ):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            candidate = m.group(1)
            resolved = tool_registry.resolve(candidate)
            if resolved:
                return resolved.name
    return ""


def _extract_dependencies(text: str) -> list[int] | None:
    """Extract step dependencies from text. Returns None if not found."""
    needs_match = re.search(
        r"needs?\s+steps?\s+([\d,\s]+)",
        text, re.IGNORECASE,
    )
    if needs_match:
        return [int(n) for n in re.findall(r"\d+", needs_match.group(1))]
    return None


def parse_plan_from_response(
    response: InternalChatResponse,
    tool_registry: ToolRegistry,
) -> ChainPlan | None:
    """Extract a numbered plan from the LLM's text response.

    Handles both compact single-line plans::

        1. Search for X using web_search
        2. Get transcript using youtube_transcript (needs step 1)

    And verbose multi-line plans (markdown headers, bullet sub-items)::

        ### 1. Fetch Video Information
        **What**: Get basic metadata
        **Using**: `youtube_transcript` (needs step 1)

    Returns None if parsing fails or the model didn't produce a plan.
    """
    text = response.message.content if response.message else ""
    # Some reasoning models put the plan in their thinking block instead of
    # content.  Fall back to thinking when content is empty or very short.
    if (not text or len(text.strip()) < 20) and response.message:
        thinking = getattr(response.message, "thinking", None) or ""
        if thinking:
            text = thinking

    # Match numbered step headers — "1. ...", "### 1. ...", "1) ..."
    step_re = re.compile(
        r"^[#\s]*(\d+)[.)]\s+(.+?)$",
        re.MULTILINE,
    )
    matches = list(step_re.finditer(text))
    if len(matches) < 2:
        return None  # not a multi-step plan

    steps: list[ChainStep] = []
    for i, m in enumerate(matches):
        step_id = int(m.group(1))
        header = m.group(2).strip()

        # Grab the full block for this step (text until next step header)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Try to extract tool name from the full block (header + body)
        tool_name = _extract_tool_name(block, tool_registry)

        # Extract dependencies from the full block
        needs = _extract_dependencies(block)
        if needs is None:
            # Implicit: each step depends on the previous one
            if steps:
                needs = [steps[-1].id]
            else:
                needs = []

        steps.append(ChainStep(
            id=step_id,
            tool=tool_name,
            needs=needs,
            reason=header,
        ))

    if not steps:
        return None

    # If no tools were resolved at all, not a valid chain plan
    if not any(s.tool for s in steps):
        return None

    return ChainPlan(steps=steps, source="adaptive")


# ---------------------------------------------------------------------------
# Plan prompt formatting
# ---------------------------------------------------------------------------


def format_plan_progress(
    plan: ChainPlan,
    results: dict[int, StepResult],
    current_step_ids: list[int] | None = None,
) -> str:
    """Format a plan progress block for injection into the LLM context.

    Shows completed steps with result previews, active steps, and pending steps.
    """
    lines = ["Plan progress:"]
    current = set(current_step_ids or [])
    for step in plan.steps:
        if step.id in results:
            r = results[step.id]
            status = "✓" if r.success else "✗"
            preview = r.output[:100].replace("\n", " ")
            lines.append(
                f"  {step.id}. {status} {step.reason} → [{preview}]"
            )
        elif step.id in current:
            lines.append(f"  {step.id}. → {step.reason} — EXECUTE THIS NOW")
        else:
            lines.append(f"  {step.id}. ○ {step.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wave executor
# ---------------------------------------------------------------------------


async def execute_step(
    step: ChainStep,
    prior_results: dict[int, StepResult],
    backend: ModelBackend,
    tool_registry: ToolRegistry,
    *,
    request_context: InternalChatRequest | None = None,
    plan: ChainPlan | None = None,
    all_results: dict[int, StepResult] | None = None,
    extra_tool_args: dict | None = None,
    tool_cache: Any | None = None,
    cache_task_id: str = "",
    cache_step_idx: int = -1,
    cache_user_id: str = "",
) -> StepResult:
    """Execute a single chain step.

    If the step has pre-specified ``input``, resolve templates and execute
    directly. Otherwise, make a focused LLM call to determine arguments.

    Args:
        extra_tool_args: Additional kwargs merged into every tool call
            (e.g. ``{"task_id": "abc", "session_id": "xyz"}`` for artifact
            tools).  Values are set only if not already present in the
            resolved args (LLM-specified values take precedence).
    """
    from augmentum.modes.analytical.tool_calling import coerce_tool_params

    tool = tool_registry.resolve(step.tool)
    if not tool:
        return StepResult(
            step_id=step.id,
            tool_name=step.tool,
            output=f"Error: Unknown tool '{step.tool}'",
            metadata={},
            success=False,
        )

    # Determine arguments
    if step.input:
        args = resolve_templates(step.input, prior_results)
    else:
        # Focused LLM call for arg resolution
        args = await _resolve_args_via_llm(
            step, tool, prior_results, backend, request_context,
            plan=plan, all_results=all_results,
        )

    args = coerce_tool_params(tool, args)

    # Merge extra args (agentic task_id, session_id, etc.) — only for tools
    # whose input schema declares those parameters or accept **kwargs.
    # ARTIFACT tools (create_document, export_*, draft_section) and IMAGE
    # tools (image_generation) accept these; search/calc/etc. do not.
    if extra_tool_args:
        schema_props = set((tool.input_schema or {}).get("properties", {}).keys())
        from augmentum.tools.base import ToolCategory
        tool_cat = getattr(tool, "category", None)
        accepts_kwargs = tool_cat in (ToolCategory.ARTIFACT, ToolCategory.IMAGE)
        for k, v in extra_tool_args.items():
            if k in schema_props or accepts_kwargs:
                args.setdefault(k, v)

    # Pass request context to tools that need the current model/backend:
    # - FlowTool: resolves backend for background chains
    # - DraftSectionTool: uses the user's model for content generation
    # - Any ARTIFACT tool with **kwargs that might need it
    # - IMAGE tools (image_generation, image_search): persist to user-scoped
    #   tables so they need `_user_id` / `_context["user_id"]` exactly like
    #   the artifact tools do. Without IMAGE in this condition, chain-routed
    #   image_generation jobs landed with empty user_id and the
    #   ``image_persist_skipped_missing_user_id`` warning fired for every
    #   illustration.
    if request_context is not None:
        from augmentum.tools.base import ToolCategory
        tool_cat = getattr(tool, "category", None)
        # Tools that write to user-scoped tables (artifacts, images,
        # custom flows) need user_id threaded into their kwargs.
        # ActionTool primitives also write user-scoped data (notes,
        # memory) and self-flag via ``needs_user_context = True``.
        if (
            hasattr(tool, "flow_id")
            or getattr(tool, "needs_user_context", False)
            or tool_cat in (ToolCategory.ARTIFACT, ToolCategory.IMAGE)
        ):
            args["_request_context"] = request_context
            # FlowTool reads `_user_id` (top-level kwarg); artifact + image
            # tools (artifact_ebook, image_generation, image_search) read
            # `_context["user_id"]`. Populate both slots so either
            # extraction pattern works.
            if cache_user_id:
                args["_user_id"] = cache_user_id
                ctx = args.get("_context")
                if not isinstance(ctx, dict):
                    ctx = {}
                    args["_context"] = ctx
                ctx.setdefault("user_id", cache_user_id)

    # Cache lookup: on resume, skip tools we already ran successfully.
    # ``cache_user_id`` is required for a hit — the cache rows are tenant
    # scoped (migration 087) so a missing user_id resolves to the anon
    # row and will not match real stored results.
    if tool_cache is not None and cache_task_id and cache_step_idx >= 0:
        from augmentum.modes.agentic.task_state import hash_tool_call
        _cache_hash = hash_tool_call(tool.name, args)
        try:
            _cached = await tool_cache.get(
                cache_task_id, cache_step_idx, _cache_hash,
                user_id=cache_user_id,
            )
        except Exception:
            _cached = None
        if _cached and _cached.success:
            return StepResult(
                step_id=step.id,
                tool_name=_cached.tool_name,
                output=_cached.output,
                metadata=_cached.metadata,
                success=True,
            )

    # Execute the tool
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            tool.execute(**args),
            timeout=tool.timeout,
        )
        elapsed = (time.monotonic() - start) * 1000
        if tool_registry:
            tool_registry.metrics.record(
                tool.name, success=result.success, elapsed_ms=elapsed,
            )
        if result.success:
            output = result.output
        else:
            # Enrich error with tool-specific recovery hints
            output = f"Error: {tool.enrich_error(result.error, args)}"
        metadata = result.metadata or {}
        # Thread ToolResult.card through metadata so the chain runner's
        # tool_call emission can surface it in the UI (artifact cards).
        if getattr(result, "card", None):
            metadata["card"] = result.card
        success = result.success
    except TimeoutError:
        elapsed = (time.monotonic() - start) * 1000
        if tool_registry:
            tool_registry.metrics.record(
                tool.name, success=False, elapsed_ms=elapsed,
            )
        raw_err = f"Tool '{tool.name}' timed out after {tool.timeout}s"
        output = f"Error: {tool.enrich_error(raw_err, args)}"
        metadata = {}
        success = False
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        if tool_registry:
            tool_registry.metrics.record(
                tool.name, success=False, elapsed_ms=elapsed,
            )
        output = f"Error: {tool.enrich_error(str(exc), args)}"
        metadata = {}
        success = False

    # Truncate long output
    max_chars = settings.tool_result_max_chars
    if len(output) > max_chars:
        tail = settings.tool_result_truncation_tail
        output = output[: max_chars - tail] + "\n...\n" + output[-tail:]

    # Cache write: persist successful tool results so a resumed task
    # can replay them without re-executing the tool.
    if tool_cache is not None and cache_task_id and cache_step_idx >= 0 and success:
        from augmentum.modes.agentic.task_state import hash_tool_call
        try:
            await tool_cache.put(
                cache_task_id,
                cache_step_idx,
                hash_tool_call(tool.name, args),
                tool_name=tool.name,
                output=output,
                metadata=metadata,
                success=True,
                user_id=cache_user_id,
            )
        except Exception as exc:
            # Cache is an optimization; never fail the step over it.
            log.debug("chain_tool_cache_write_failed", tool=tool.name, error=str(exc))

    return StepResult(
        step_id=step.id,
        tool_name=tool.name,
        output=output,
        metadata=metadata,
        success=success,
    )


async def _resolve_args_via_llm(
    step: ChainStep,
    tool: Tool,
    prior_results: dict[int, StepResult],
    backend: ModelBackend,
    request_context: InternalChatRequest | None,
    *,
    plan: ChainPlan | None = None,
    all_results: dict[int, StepResult] | None = None,
) -> dict:
    """Make a focused LLM call to determine tool arguments for a step.

    When *plan* and *all_results* are provided and the attention anchor
    setting is enabled, the full plan progress is prepended to the user
    content so the model always sees the current state of execution
    (Manus todo.md pattern).
    """
    import json as json_mod

    # Build context from dependency results
    context_parts = []
    for dep_id in step.needs:
        dep = prior_results.get(dep_id)
        if dep:
            status = "OK" if dep.success else "FAILED"
            preview = dep.output[:4000]
            context_parts.append(
                f"[Step {dep_id} — {dep.tool_name} ({status})]: {preview}"
            )

    # Include any user query context
    user_query = ""
    if request_context and request_context.messages:
        for msg in reversed(request_context.messages):
            if msg.role == "user":
                user_query = msg.content or ""
                break

    schema = tool.input_schema or {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    param_desc = []
    for k, v in properties.items():
        req = " (required)" if k in required else ""
        param_desc.append(f"  - {k}: {v.get('type', 'string')}{req} — {v.get('description', '')}")

    from augmentum.utils.datetime_context import get_datetime_context

    system_prompt = (
        f"{get_datetime_context()}\n\n"
        f"You need to call the tool '{tool.name}': {tool.description}\n"
        f"Parameters:\n" + "\n".join(param_desc) + "\n\n"
        "Respond with ONLY a JSON object containing the parameters. No explanation."
    )

    # --- Attention anchor (Manus pattern) ---
    # Prepend plan progress so the model sees the full execution state,
    # keeping the plan in the model's recent attention window.
    anchor = ""
    if (
        settings.passthrough_chain_attention_anchor
        and plan is not None
        and all_results is not None
    ):
        anchor = format_plan_progress(plan, all_results, current_step_ids=[step.id]) + "\n\n"

    user_content = anchor + f"Task: {step.reason}\n"
    if user_query:
        user_content += f"User's original request: {user_query}\n"
    if context_parts:
        user_content += "Results from prior steps:\n" + "\n".join(context_parts)
    # When a dependency failed and error-as-observation is on, add adaptation guidance
    failed_deps = [
        prior_results[d] for d in step.needs
        if d in prior_results and not prior_results[d].success
    ]
    if failed_deps and settings.passthrough_chain_error_as_observation:
        user_content += (
            "\n\nNOTE: Some prior steps FAILED. Their error output is shown above. "
            "Adapt your approach — use what succeeded and work around what failed."
        )

    # Artifact tools (create_ebook, create_document, create_presentation,
    # create_spreadsheet) emit very large JSON args because each section /
    # chapter body is inlined as a JSON string. A multi-chapter storybook
    # with full prose bodies is ~12-20KB of JSON. Without an explicit
    # budget the backend's default response cap (often ~2-4K tokens) cuts
    # the model off mid-string — JSON fails to parse, args resolve to
    # ``{}``, the step "fails", and the chain replanner kicks in producing
    # a phantom recovery artifact (e.g. a create_document PDF) plus a
    # degraded retry that drops chapters to fit under the cap. Setting a
    # generous budget here is one-shot per step (no accumulation) and the
    # backend still enforces its own ceiling, so this only matters when
    # the default was the bottleneck.
    from augmentum.tools.base import ToolCategory
    _large_output = getattr(tool, "category", None) in (
        ToolCategory.ARTIFACT, ToolCategory.IMAGE,
    )
    _max_tokens = 16384 if _large_output else 4096

    llm_request = InternalChatRequest(
        model=request_context.model if request_context else "",
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ],
        stream=False,
        max_tokens=_max_tokens,
    )

    try:
        resp = await backend.chat(llm_request)
        text = resp.message.content if resp.message else ""
        # Extract JSON from response
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        return json_mod.loads(text)
    except json_mod.JSONDecodeError as exc:
        # Distinguish truncation (the common failure mode for artifact
        # tools) from a malformed-but-complete response. A response that
        # ends without a closing brace is almost always a token-budget
        # truncation; surface it with the response length so the
        # signal-to-noise on chain_replan decisions stays high.
        _tail = (text or "").rstrip()
        _truncated = bool(_tail) and not _tail.endswith(("}", "]"))
        log.warning(
            "chain_arg_resolution_failed",
            step=step.id,
            tool=tool.name,
            error=str(exc),
            response_chars=len(text or ""),
            looks_truncated=_truncated,
            max_tokens=_max_tokens,
        )
        return {}
    except Exception as exc:
        log.warning(
            "chain_arg_resolution_failed",
            step=step.id, tool=tool.name, error=str(exc),
        )
        return {}


async def execute_chain_streaming(
    plan: ChainPlan,
    backend: ModelBackend,
    tool_registry: ToolRegistry,
    queue: asyncio.Queue[StepResult | None],
    *,
    request_context: InternalChatRequest | None = None,
    max_steps: int | None = None,
    extra_tool_args: dict | None = None,
    allowed_tool_names: set[str] | None = None,
    tool_cache: Any | None = None,
    cache_task_id: str = "",
    cache_user_id: str = "",
    cache_step_idx_base: int = -1,
) -> dict[int, StepResult]:
    """Execute a chain plan, pushing each StepResult to *queue* as it completes.

    A ``None`` sentinel is pushed after all steps finish (or on error) so
    that consumers know the stream is done.

    Returns the same results dict as :func:`execute_chain`.
    """
    async def _push_done(result: StepResult) -> None:
        await queue.put(result)

    try:
        return await execute_chain(
            plan, backend, tool_registry,
            request_context=request_context,
            on_step_done=_push_done,
            max_steps=max_steps,
            extra_tool_args=extra_tool_args,
            allowed_tool_names=allowed_tool_names,
            tool_cache=tool_cache,
            cache_task_id=cache_task_id,
            cache_user_id=cache_user_id,
            cache_step_idx_base=cache_step_idx_base,
        )
    finally:
        await queue.put(None)


async def _replan_on_failure(
    step: ChainStep,
    failed_result: StepResult,
    prior_results: dict[int, StepResult],
    plan: ChainPlan,
    backend: ModelBackend,
    request_context: InternalChatRequest | None,
    *,
    allowed_tool_names: set[str] | None = None,
) -> str:
    """Ask the LLM what to do about a failed step.

    Returns one of: "retry", "substitute:<tool_name>", "skip", "abort".

    When ``allowed_tool_names`` is provided, the substitute candidates shown
    to the LLM are restricted to that set — substitutions outside it are
    rejected by the caller. Without this bound, a chain that started with
    a curated tool set could mutate into using anything in the registry.
    """

    succeeded = [
        f"Step {r.step_id} ({r.tool_name}): {r.output[:200]}"
        for r in prior_results.values() if r.success
    ]
    user_goal = ""
    if request_context and request_context.messages:
        for msg in reversed(request_context.messages):
            if msg.role == "user":
                user_goal = msg.content or ""
                break

    if allowed_tool_names:
        available_tools = sorted(allowed_tool_names)
    else:
        available_tools = [s.tool for s in plan.steps if s.tool]

    mutation_option = ""
    if settings.passthrough_chain_plan_mutation:
        mutation_option = "- mutate (restructure the remaining plan — add, remove, or replace steps)\n"

    system = (
        "A multi-step tool chain is executing. One step failed.\n"
        "Decide what to do. Respond with EXACTLY one word or phrase:\n"
        "- retry (try the same step again)\n"
        "- substitute:<tool_name> (use a different tool)\n"
        "- skip (skip this step, let dependents proceed with partial context)\n"
        + mutation_option +
        "- abort (stop the chain, cascade failure to dependents)\n"
        "\nRespond with ONLY the decision, nothing else."
    )

    user_content = (
        f"User's goal: {user_goal}\n\n"
        f"Failed step {step.id} ({step.tool}): {failed_result.output}\n"
        f"Reason: {step.reason}\n\n"
    )
    if succeeded:
        user_content += "Succeeded so far:\n" + "\n".join(succeeded) + "\n\n"
    user_content += f"Available tools: {', '.join(set(available_tools))}"

    llm_request = InternalChatRequest(
        model=request_context.model if request_context else "",
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user_content),
        ],
        stream=False,
    )

    try:
        resp = await backend.chat(llm_request)
        text = (resp.message.content if resp.message else "").strip().lower()
        if text.startswith("substitute:"):
            return text  # e.g. "substitute:wikipedia"
        if text in ("retry", "skip", "abort", "mutate"):
            return text
        # Unrecognized — default to abort for safety
        log.warning("replan_unrecognized_decision", decision=text)
        return "abort"
    except Exception:
        log.warning("replan_llm_call_failed", exc_info=True)
        return "abort"


async def _mutate_plan(
    plan: ChainPlan,
    failed_step: ChainStep,
    failed_result: StepResult,
    completed_results: dict[int, StepResult],
    remaining_steps: list[ChainStep],
    backend: ModelBackend,
    tool_registry: ToolRegistry,
    request_context: InternalChatRequest | None,
    *,
    allowed_tool_names: set[str] | None = None,
) -> list[ChainStep] | None:
    """Ask the LLM to restructure the remaining plan after a failure.

    Returns replacement steps for the remaining portion of the plan,
    or ``None`` if parsing fails (caller should fall back to abort).

    When ``allowed_tool_names`` is provided, the LLM only sees that subset
    in the prompt and any mutated step naming a tool outside the set is
    rejected. Without this bound, the failure-recovery path could pull in
    tools the user never enabled.
    """
    import json as json_mod

    user_goal = ""
    if request_context and request_context.messages:
        for msg in reversed(request_context.messages):
            if msg.role == "user":
                user_goal = msg.content or ""
                break

    completed_summary = "\n".join(
        f"  Step {r.step_id} ({r.tool_name}): {'OK' if r.success else 'FAILED'} — {r.output[:150]}"
        for r in completed_results.values()
    )

    if allowed_tool_names:
        available = sorted(allowed_tool_names)
    else:
        available = [t.name for t in tool_registry.list_tools()]

    remaining_summary = "\n".join(
        f"  Step {s.id} ({s.tool}): {s.reason} [depends on: {s.needs}]"
        for s in remaining_steps
    )

    system = (
        "A multi-step tool chain failed at one step. You need to restructure "
        "the REMAINING steps to still achieve the user's goal.\n\n"
        "Respond with a JSON array of step objects. Each step:\n"
        '  {"id": <int>, "tool": "<tool_name>", "reason": "<why>", '
        '"needs": [<dependency_ids>], "input": null}\n\n'
        "Rules:\n"
        "- Keep step IDs sequential starting from the next available ID\n"
        "- You can add, remove, or replace steps\n"
        "- Dependencies can reference completed step IDs or new step IDs\n"
        "- Only use tools from the available list\n"
        "- Respond with ONLY the JSON array, no explanation"
    )

    next_id = max((s.id for s in plan.steps), default=0) + 1

    user_content = (
        f"User's goal: {user_goal}\n\n"
        f"Failed step {failed_step.id} ({failed_step.tool}): {failed_result.output}\n\n"
        f"Completed steps:\n{completed_summary}\n\n"
        f"Remaining steps (to be restructured):\n{remaining_summary}\n\n"
        f"Available tools: {', '.join(available)}\n"
        f"Next available step ID: {next_id}"
    )

    llm_request = InternalChatRequest(
        model=request_context.model if request_context else "",
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user_content),
        ],
        stream=False,
    )

    try:
        resp = await backend.chat(llm_request)
        text = (resp.message.content if resp.message else "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        raw = json_mod.loads(text)
        if not isinstance(raw, list):
            return None

        new_steps = []
        for item in raw:
            new_steps.append(ChainStep(
                id=int(item["id"]),
                tool=str(item["tool"]),
                reason=str(item.get("reason", "")),
                needs=[int(d) for d in item.get("needs", [])],
                input=item.get("input"),
            ))

        # Validate all tools exist AND are within the allowed set (if any).
        # The LLM was already instructed to stay within ``available``; this
        # second check prevents a phantom name from sneaking through when
        # the model ignores the constraint.
        for s in new_steps:
            resolved = tool_registry.resolve(s.tool)
            if not resolved:
                log.warning("mutate_unknown_tool", tool=s.tool)
                return None
            if allowed_tool_names and resolved.name not in allowed_tool_names:
                log.warning(
                    "mutate_tool_outside_allowed_set",
                    tool=s.tool, resolved=resolved.name,
                )
                return None

        # Cap mutated steps
        max_steps = settings.passthrough_chain_max_steps
        if len(new_steps) > max_steps:
            new_steps = new_steps[:max_steps]

        log.info(
            "chain_plan_mutated",
            old_remaining=len(remaining_steps),
            new_steps=len(new_steps),
            tools=[s.tool for s in new_steps],
        )
        return new_steps

    except Exception:
        log.warning("mutate_plan_failed", exc_info=True)
        return None


async def execute_chain(
    plan: ChainPlan,
    backend: ModelBackend,
    tool_registry: ToolRegistry,
    *,
    request_context: InternalChatRequest | None = None,
    on_step_start: object = None,
    on_step_done: object = None,
    on_replan: object = None,
    max_steps: int | None = None,
    extra_tool_args: dict | None = None,
    tool_cache: Any | None = None,
    cache_task_id: str = "",
    cache_user_id: str = "",
    cache_step_idx_base: int = -1,
    allowed_tool_names: set[str] | None = None,
) -> dict[int, StepResult]:
    """Execute a chain plan using wave-based parallel execution.

    Steps whose dependencies are all satisfied run in parallel.
    Continues until all steps are complete or no progress can be made.

    When a step fails and ``passthrough_chain_max_retries > 0``, the LLM
    is consulted for a re-plan decision (retry/substitute/skip/abort).

    Args:
        on_step_start: Optional ``async (step: ChainStep)`` callback.
        on_step_done: Optional ``async (result: StepResult)`` callback.
        on_replan: Optional ``async (step_id, decision)`` callback.
        max_steps: Override for max steps (default from config).

    Returns:
        Dict mapping step_id → StepResult.
    """
    _max = max_steps or settings.passthrough_chain_max_steps
    max_retries = settings.passthrough_chain_max_retries
    results: dict[int, StepResult] = {}
    retry_counts: dict[int, int] = {}
    total_replans = 0
    max_total_replans = max_retries * len(plan.steps)
    remaining = list(plan.steps[:_max])
    remaining_ids = {s.id for s in remaining}

    wave_num = 0
    while remaining:
        wave_num += 1
        # Find steps whose dependencies are all satisfied
        ready = [
            s for s in remaining
            if all(d in results for d in s.needs)
        ]
        if not ready:
            # Circular dependency or unresolvable deps — bail
            log.warning(
                "chain_stuck",
                remaining=[s.id for s in remaining],
                results=list(results.keys()),
            )
            # Mark all remaining as failed
            for s in remaining:
                unsatisfied = [d for d in s.needs if d not in results]
                results[s.id] = StepResult(
                    step_id=s.id,
                    tool_name=s.tool,
                    output=f"Error: Unresolvable dependencies: steps {unsatisfied}",
                    metadata={},
                    success=False,
                )
            break

        # Handle steps with failed dependencies
        wave_ready = []
        for s in ready:
            failed_deps = [
                d for d in s.needs
                if d in results and not results[d].success
            ]
            if failed_deps:
                if settings.passthrough_chain_error_as_observation:
                    # Error-as-observation: let the step proceed.
                    # The failed dependency output stays in prior_results so
                    # _resolve_args_via_llm sees it and can adapt.
                    log.info(
                        "chain_error_as_observation",
                        step=s.id,
                        failed_deps=failed_deps,
                    )
                    wave_ready.append(s)
                else:
                    # Legacy cascade: skip the step entirely
                    results[s.id] = StepResult(
                        step_id=s.id,
                        tool_name=s.tool,
                        output=f"Error: Skipped — dependency steps {failed_deps} failed",
                        metadata={},
                        success=False,
                    )
                    remaining.remove(s)
                    if on_step_done:
                        await on_step_done(results[s.id])  # type: ignore[misc]
                    continue
            else:
                wave_ready.append(s)

        if not wave_ready:
            continue

        log.info(
            "chain_wave",
            wave=wave_num,
            steps=[s.id for s in wave_ready],
            tools=[s.tool for s in wave_ready],
        )

        # Fire start callbacks
        if on_step_start:
            for s in wave_ready:
                await on_step_start(s)  # type: ignore[misc]

        # Execute wave in parallel (semaphore caps concurrency)
        sem = asyncio.Semaphore(settings.passthrough_chain_max_parallel)

        async def _guarded(step: ChainStep, sem=sem) -> StepResult:
            async with sem:
                return await execute_step(
                    step, results, backend, tool_registry,
                    request_context=request_context,
                    plan=plan, all_results=results,
                    extra_tool_args=extra_tool_args,
                    tool_cache=tool_cache,
                    cache_task_id=cache_task_id,
                    cache_step_idx=cache_step_idx_base,
                    cache_user_id=cache_user_id,
                )

        tasks = [_guarded(s) for s in wave_ready]
        wave_results = await asyncio.gather(*tasks, return_exceptions=True)

        for step, result in zip(wave_ready, wave_results, strict=False):
            if isinstance(result, Exception):
                sr = StepResult(
                    step_id=step.id,
                    tool_name=step.tool,
                    output=f"Error: {result}",
                    metadata={},
                    success=False,
                )
            else:
                sr = result

            # Re-plan on failure if retries are enabled
            if (
                not sr.success
                and max_retries > 0
                and retry_counts.get(step.id, 0) < max_retries
                and total_replans < max_total_replans
            ):
                decision = await _replan_on_failure(
                    step, sr, results, plan, backend, request_context,
                    allowed_tool_names=allowed_tool_names,
                )
                total_replans += 1
                log.info("chain_replan", step=step.id, decision=decision)

                if on_replan:
                    await on_replan(step.id, decision)  # type: ignore[misc]

                if decision == "retry":
                    retry_counts[step.id] = retry_counts.get(step.id, 0) + 1
                    # Re-execute same step — don't add to results, don't remove from remaining.
                    # Retries bypass the cache on purpose: the first attempt's cached
                    # result was almost certainly the failure we're recovering from.
                    retry_sr = await execute_step(
                        step, results, backend, tool_registry,
                        request_context=request_context,
                        plan=plan, all_results=results,
                        extra_tool_args=extra_tool_args,
                    )
                    sr = retry_sr
                elif decision.startswith("substitute:"):
                    new_tool = decision.split(":", 1)[1].strip()
                    resolved = tool_registry.resolve(new_tool)
                    # Reject substitutions outside the allowed set — without
                    # this, the LLM could swap to any tool in the global
                    # registry, bypassing the user's curated selection.
                    if resolved and (
                        not allowed_tool_names
                        or resolved.name in allowed_tool_names
                    ):
                        step.tool = resolved.name
                        retry_counts[step.id] = retry_counts.get(step.id, 0) + 1
                        retry_sr = await execute_step(
                            step, results, backend, tool_registry,
                            request_context=request_context,
                            plan=plan, all_results=results,
                            extra_tool_args=extra_tool_args,
                        )
                        sr = retry_sr
                    elif resolved:
                        log.info(
                            "chain_substitute_outside_allowed_set",
                            step=step.id, tool=resolved.name,
                            allowed=sorted(allowed_tool_names or ()),
                        )
                    # If tool not found or disallowed, fall through with original failure
                elif decision == "skip":
                    # Mark failed but DON'T cascade — dependents proceed
                    sr = StepResult(
                        step_id=step.id,
                        tool_name=step.tool,
                        output=f"Skipped: {sr.output}",
                        metadata=sr.metadata,
                        success=True,  # Mark as "success" so dependents run
                    )
                elif decision == "mutate" and settings.passthrough_chain_plan_mutation:
                    # LLM restructures the remaining plan
                    mutated = await _mutate_plan(
                        plan, step, sr, results, remaining,
                        backend, tool_registry, request_context,
                        allowed_tool_names=allowed_tool_names,
                    )
                    if mutated is not None:
                        # Replace remaining steps with mutated plan
                        remaining.clear()
                        remaining.extend(mutated)
                        remaining_ids.clear()
                        remaining_ids.update(s.id for s in mutated)
                        # Update plan.steps to include completed + new
                        plan.steps = [
                            s for s in plan.steps if s.id in results
                        ] + mutated
                        sr = StepResult(
                            step_id=step.id,
                            tool_name=step.tool,
                            output=f"Plan mutated after failure: {sr.output}",
                            metadata=sr.metadata,
                            success=True,  # Allow execution to continue
                        )
                    # If mutation failed, fall through to abort behavior
                # "abort" — use original failure, which will cascade

            results[step.id] = sr
            if step in remaining:
                remaining.remove(step)
            if on_step_done:
                await on_step_done(sr)  # type: ignore[misc]

    return results


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------


def build_synthesis_prompt(
    plan: ChainPlan,
    results: dict[int, StepResult],
    user_query: str = "",
) -> str:
    """Build the final synthesis prompt with all step results."""
    parts = ["All steps complete. Results:\n"]
    tool_names_used = set()
    artifact_produced = False
    for step in plan.steps:
        r = results.get(step.id)
        if r:
            status = "✓" if r.success else "✗"
            # When a step produced a structured artifact card, the UI already
            # shows a card with title / preview / download. Don't echo the
            # full tool output into the synthesis prompt — weak models will
            # parrot it verbatim, producing the "[tool (OK)]: ..." duplicate
            # users see in chat. A single-line acknowledgement is enough.
            has_card = bool((r.metadata or {}).get("card"))
            if r.success and has_card:
                parts.append(
                    f"[Step {step.id} — {r.tool_name} {status}]: "
                    "(artifact delivered; card is rendered in the UI)\n"
                )
                artifact_produced = True
            else:
                parts.append(f"[Step {step.id} — {r.tool_name} {status}]: {r.output}\n")
            if r.success:
                tool_names_used.add(r.tool_name)
    parts.append("")
    if user_query:
        parts.append(
            f"The user's original request was: {user_query}\n"
        )
    if artifact_produced:
        parts.append(
            "The requested artifact has already been delivered and is visible "
            "to the user as a card with preview / download buttons. Respond "
            "with ONE short friendly sentence confirming it is ready — do NOT "
            "re-state the filename, URL, chapter count, or any tool output."
        )
    else:
        parts.append(
            "Answer the user's question using the results above. "
            "Focus on substance and insight — do NOT just list raw stats or "
            "repeat tool output verbatim. Provide a thoughtful, complete response."
        )
    # Add tool-specific guidance for search results
    if tool_names_used & {"web_search", "web"}:
        parts.append(
            "Cite sources inline using [1], [2], etc. and include a Sources list at the end."
        )
    if "python_exec" in tool_names_used:
        parts.append(
            "Interpret the code output — explain what the numbers or results mean in context."
        )
    return "\n".join(parts)


def _extract_user_query(request: InternalChatRequest) -> str:
    """Extract the last user message from a request."""
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.content or ""
    return ""


# ---------------------------------------------------------------------------
# Adaptive chain planner
# ---------------------------------------------------------------------------


class ToolChainPlanner:
    """Orchestrates adaptive chain execution — detect, plan, execute, synthesize."""

    def __init__(
        self,
        backend: ModelBackend,
        tool_registry: ToolRegistry,
    ) -> None:
        self._backend = backend
        self._registry = tool_registry

    async def plan_and_execute(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
        *,
        on_step_start: object = None,
        on_step_done: object = None,
        extra_tool_args: dict | None = None,
        tool_cache: Any | None = None,
        cache_task_id: str = "",
        cache_user_id: str = "",
        cache_step_idx_base: int = -1,
    ) -> tuple[dict[int, StepResult], ChainPlan] | None:
        """Run adaptive chain: plan in one LLM call, then execute waves.

        Returns (results, plan) or None if planning fails.

        The ``tools`` list is the constraint — both the planner prompt and
        the failure-recovery (substitute/mutate) paths only see these
        names, so the chain can't drift outside the user's curated set.

        ``cache_user_id`` is the *required* slot for IMAGE/ARTIFACT tools to
        persist to their user-scoped tables. Callers MUST pass it whenever
        their handler has a user_id; without it, image/artifact rows land
        with empty user_id and become invisible in the Library.
        """
        plan_prompt = self._build_plan_prompt(request, tools)
        plan = await self._get_plan(request.model, plan_prompt)

        if not plan:
            log.info("chain_plan_parse_failed")
            return None

        log.info(
            "chain_plan_created",
            steps=len(plan.steps),
            tools=[s.tool for s in plan.steps],
        )

        # Phase 2+: Wave execution
        allowed_tool_names = {t.name for t in tools} if tools else None
        results = await execute_chain(
            plan, self._backend, self._registry,
            request_context=request,
            on_step_start=on_step_start,
            on_step_done=on_step_done,
            extra_tool_args=extra_tool_args,
            allowed_tool_names=allowed_tool_names,
            tool_cache=tool_cache,
            cache_task_id=cache_task_id,
            cache_user_id=cache_user_id,
            cache_step_idx_base=cache_step_idx_base,
        )

        return results, plan

    async def _get_plan(
        self,
        model: str,
        messages: list[Message],
    ) -> ChainPlan | None:
        """Request a plan from the LLM with a tier of fallbacks.

        Tier 1: JSON-constrained output (``format="json"``). Best when the
            backend supports it (Ollama, llama.cpp, llama-server). Some
            backends (LM Studio) reject it — we drop straight to tier 2.
        Tier 2: No format constraint, parse JSON from the response.
        Tier 3: Strict retry — if both prior tiers parsed nothing, send a
            second request with an *aggressive* "ONLY JSON, NO PROSE"
            preamble. This rescues weak instruction-followers that
            ignored the original prompt.
        Tier 4: Text-pattern fallback (regex extraction from prose).
        """
        # Tier 1: JSON-constrained
        response = await self._call_planner(model, messages, json_format=True)
        plan = self._parse_plan_response(response)
        if plan:
            return plan

        # Tier 2: No format constraint
        if response is None:
            response = await self._call_planner(model, messages, json_format=False)
            plan = self._parse_plan_response(response)
            if plan:
                return plan

        # Tier 3: Strict retry with aggressive preamble
        retry_messages = [
            Message(
                role="system",
                content=(
                    "Your previous output could not be parsed as a valid JSON "
                    "execution plan. RESPOND WITH ONLY A SINGLE JSON OBJECT — "
                    "no prose, no explanation, no markdown fences. The JSON "
                    "MUST have a top-level 'steps' array. Each step MUST have "
                    "'id' (int), 'tool' (string from the tools list), 'reason' "
                    "(string), and 'needs' (array of int).\n\n"
                    + (messages[0].content if messages and messages[0].role == "system" else "")
                ),
            ),
            *(m for m in messages if m.role != "system"),
        ]
        response = await self._call_planner(model, retry_messages, json_format=True)
        plan = self._parse_plan_response(response)
        if plan:
            log.info("chain_plan_strict_retry_succeeded")
            return plan

        # Last resort — log a snippet of what the model returned so the
        # failure is debuggable from logs alone.
        snippet = ""
        if response and response.message:
            snippet = (response.message.content or "")[:200]
        log.warning("chain_plan_all_tiers_failed", returned_snippet=snippet)
        return None

    async def _call_planner(
        self,
        model: str,
        messages: list[Message],
        *,
        json_format: bool,
    ):
        """Single planner request. Returns None on backend error."""
        # Once a backend has rejected format=json in this session, don't
        # keep sending it — LM Studio's 400 is deterministic, and retrying
        # burns a round-trip for every tier that asks for json_format.
        if json_format and getattr(self, "_planner_json_unsupported", False):
            json_format = False
        try:
            req = InternalChatRequest(
                model=model,
                messages=messages,
                stream=False,
                think=False,
                format="json" if json_format else None,
            )
            return await self._backend.chat(req)
        except Exception:
            if json_format:
                self._planner_json_unsupported = True
                log.info("chain_plan_json_format_unsupported", fallback="text")
            else:
                log.warning("chain_plan_request_failed", exc_info=True)
            return None

    def _parse_plan_response(self, response) -> ChainPlan | None:
        """Try every parser on a planner response. Returns None on no match."""
        if response is None:
            return None
        text = response.message.content if response.message else ""
        plan = parse_plan_from_json(text, self._registry)
        if not plan and response.message:
            thinking = getattr(response.message, "thinking", None)
            if thinking:
                plan = parse_plan_from_json(thinking, self._registry)
        if not plan:
            plan = parse_plan_from_response(response, self._registry)
        return plan

    def _build_plan_prompt(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
    ) -> list[Message]:
        """Build an isolated planning prompt with only the user query.

        The planner gets a clean, focused system prompt plus the last user
        message — no conversation history, memory context, or frontend
        system prompts.  Full context is preserved in *request* for the
        synthesis call that runs after execution.
        """
        tool_lines = []
        for t in tools:
            schema = t.input_schema or {}
            params = schema.get("properties", {})
            param_str = ", ".join(
                f"{k}: {v.get('type', 'string')}" for k, v in params.items()
            )
            tool_lines.append(f"  {t.name}({param_str}) — {t.description}")

        # Extract just the last user message
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content or ""
                break

        from augmentum.utils.datetime_context import get_datetime_context

        system = (
            f"{get_datetime_context()}\n\n"
            "You are a task planner. Output ONLY a single JSON object — no "
            "prose, no markdown, no code fences, no commentary before or "
            "after.\n\n"
            "REQUIRED SHAPE:\n"
            '{"steps": [{"id": int, "tool": str, "reason": str, "needs": [int]}]}\n\n'
            "Available tools (use the exact name shown):\n"
            + "\n".join(tool_lines)
            + "\n\n"
            "RULES:\n"
            "- ``id``: integer starting at 1, increments by 1.\n"
            "- ``tool``: must be one of the names listed above. Do NOT invent "
            "tool names.\n"
            "- ``reason``: one short sentence describing what this step does "
            "for the user's request.\n"
            "- ``needs``: list of prior step ids that must finish first. "
            "Empty list ``[]`` means the step can run in parallel with other "
            "needs-empty steps.\n"
            "- Use the fewest steps that actually answer the request.\n"
            "- If the request needs no tools (pure conversation, math the "
            "model can do in its head, etc.), still emit a ``steps`` array "
            "with at least one step using the most appropriate tool from "
            "the list.\n\n"
            "EXAMPLES:\n"
            "User: 'What's the weather in Paris and what should I wear?'\n"
            '{"steps":[\n'
            '  {"id":1,"tool":"web_search","reason":"Look up current Paris weather","needs":[]},\n'
            '  {"id":2,"tool":"web_fetch","reason":"Read the top result for forecast details","needs":[1]}\n'
            "]}\n\n"
            "User: 'Build me a 3-chapter ebook about a kitchen mouse named Pip.'\n"
            '{"steps":[\n'
            '  {"id":1,"tool":"create_ebook","reason":"Author a 3-chapter illustrated EPUB about Pip the kitchen mouse","needs":[]}\n'
            "]}\n\n"
            "Output the JSON object now. Nothing else."
        )

        return [
            Message(role="system", content=system),
            Message(role="user", content=user_query),
        ]

    async def synthesize(
        self,
        request: InternalChatRequest,
        plan: ChainPlan,
        results: dict[int, StepResult],
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream the final synthesis response."""
        synth_prompt = build_synthesis_prompt(
            plan, results, user_query=_extract_user_query(request),
        )
        synth_request = InternalChatRequest(
            model=request.model,
            messages=[
                *request.messages,
                Message(role="user", content=synth_prompt),
            ],
            stream=True,
        )
        timeout = settings.passthrough_chain_synthesis_timeout
        gen = self._backend.chat_stream(synth_request)
        while True:
            try:
                chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                yield chunk
            except StopAsyncIteration:
                break
            except TimeoutError:
                log.warning("chain_synthesis_timeout", timeout=timeout)
                yield InternalStreamChunk(
                    content_delta="\n\n*(Response generation timed out)*",
                )
                break
