"""Passthrough mode — forwards requests directly to the backend with minimal overhead.

Supports optional per-request tool enablement via X-Augmentum-Tools header or
the passthrough_tools config setting. When tools are enabled, the handler runs
a simple tool-call loop: inject tool schemas → LLM responds → execute tool
calls → feed results back → repeat until the LLM produces a final text response.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
)
from augmentum.modes.base import ModeHandler
from augmentum.modes.v_command import extract_v_command, generate_direct_image
from augmentum.security.untrusted import (
    MARKER_OPEN_PREFIX,
    ensure_policy_in_system,
)
from augmentum.tools.base import invoke_tool
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.image.queue import GenerationQueue
    from augmentum.modes.analytical.tool_calling import ToolCallingTier
    from augmentum.tools.base import Tool
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

_TOOL_RESULT_MAX = settings.tool_result_max_chars

# Hard ceiling for the "unlimited" setting. Not a budget — a backstop,
# mirroring coder mode's 150-iteration ceiling. The real loop brakes are
# the no-progress and repeat-call guards below; this only guarantees a
# genuinely stuck turn terminates instead of running forever.
_ITERATION_CEILING = 150

# Gated capabilities that are ALWAYS run inline, never proposed as a
# confirmation chip. See ``_should_gate_capability`` for the reasoning: these
# are second-scale, in-process, and non-destructive, so a confirm would only
# add friction to the visualisation the model chose to show. Kept as a
# module-level set so ``_should_gate_capability`` has one obvious place to
# extend and tests can assert membership without instantiating a handler.
_NEVER_GATED_CAPABILITIES: frozenset[str] = frozenset({
    "create_chart",
    "create_spreadsheet",
})


# Per-turn system directives for tools that RENDER something into the reply.
# One shared failure mode: the model writes the content as prose instead of
# emitting the tool call, so nothing appears. Injected only when the named tool
# is on this turn's menu — see ``_inject_call_dont_narrate_directives``.
#
# Why a directive and not just the tool description: the description competes
# with a persona/system prompt that is often actively steering AWAY from tool
# use (stay in character, be conversational, answer directly). A system-level
# line is the only lever that reliably outranks it.
#
# Keep each entry SHORT. It rides in the system prompt on every turn the tool is
# available, so it costs prefix tokens; long enough to be unambiguous, short
# enough not to reshape the persona.
_CALL_DONT_NARRATE_DIRECTIVES: tuple[tuple[str, str], ...] = (
    (
        "image_generation",
        # Observed: deepseek-v4-flash composed a full scene description after a
        # '---' divider but never emitted the tool_call, so the user's "show me"
        # produced prose and no image.
        "[IMAGE TOOL] When the user wants to SEE or be SHOWN something "
        "visual — or you decide to show them an image — you MUST call the "
        "image_generation tool, putting the full visual description in its "
        "`prompt` argument. Describing the image in your reply does NOT "
        "create it; only the tool call renders an image. Keep your spoken "
        "reply in character, but make the tool call so the image appears.",
    ),
    (
        "create_chart",
        # Same class as the image case: the model explains the shape of the data
        # ("revenue climbs steadily, peaking in Q3") or dumps a markdown table
        # instead of drawing it. Charts are the ONE capability the user
        # explicitly wants offered proactively, so this says so outright.
        "[CHART TOOL] When your answer compares quantities, shows a trend "
        "over time, or breaks a whole into parts, call the create_chart tool "
        "with the real labels and values — the chart appears inline in your "
        "reply. You do not need to be asked: prefer drawing the data over "
        "writing a table of figures or describing the pattern in words. A "
        "table or a description renders NO chart; only the tool call does. "
        "Keep your prose short and let the chart carry the numbers.",
    ),
)


def _capability_dep_available(tool: object) -> bool:
    """Whether an Auto-exposed capability can actually run right now.

    Applied ONLY to the never-gated creators: their render deps (matplotlib,
    openpyxl) are lazy and, for matplotlib, Docker-only, so on a bare-metal
    install the tool registers but cannot draw. Offering it anyway means the
    model promises a chart and then apologises.

    Deliberately NOT applied to every inline capability. ``image_generation``
    is exposed in Auto precisely so the model never denies it can generate
    images (see ``_resolve_auto_capability_tools``); dropping it whenever its
    provider is briefly unhealthy would reintroduce that denial. A failing
    health check there is a transient service issue, not a missing dependency.
    An unhealthy check is treated as available, so a slow or throwing probe
    never silently removes a working tool.
    """
    name = getattr(tool, "name", "")
    if name not in _NEVER_GATED_CAPABILITIES:
        return True
    try:
        return bool(tool.health_check())
    except Exception:
        log.warning("capability_health_check_failed", tool=name, exc_info=True)
        return True


async def _resolve_user_chain_limit(handler: object) -> int | None:
    """Read the caller's ``ui.toolChainLimit`` preference, or None if unset.

    Same per-user channel ``ui.autoTools`` uses. Returns None (not a
    number) when unset or unreadable so the caller falls through to the
    install default — distinct from an explicit 0, which means unlimited.
    """
    user_id = getattr(handler, "_user_id", "")
    app_state = getattr(handler, "_app_state", None)
    if not user_id or app_state is None:
        return None
    store = getattr(app_state, "settings_store", None)
    if store is None:
        return None
    try:
        raw = await store.get_user_or_global(user_id, "ui.toolChainLimit")
    except Exception:  # noqa: BLE001 — a settings read must never kill a turn
        log.debug("tool_chain_limit_read_failed", exc_info=True)
        return None
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        log.warning("tool_chain_limit_unparseable", value=raw)
        return None


def _max_iterations(request: object = None, user_limit: int | None = None) -> int:
    """Resolve the tool-call round-trip budget for THIS turn.

    Read per-call, never captured at import. The old module-level
    constant meant changing the setting required a container restart,
    so the value in the UI and the value in force could silently
    disagree.

    Resolution order: an explicit per-request override → the caller's
    ``ui.toolChainLimit`` preference → the install default → 5. ``0``
    at any level means unlimited and resolves to
    :data:`_ITERATION_CEILING`.
    """
    override = getattr(request, "tool_max_iterations", None)
    raw = override if isinstance(override, int) and override >= 0 else None
    if raw is None and isinstance(user_limit, int) and user_limit >= 0:
        raw = user_limit
    if raw is None:
        raw = getattr(settings, "passthrough_tool_max_iterations", 5)
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        raw = 5
    if raw <= 0:
        return _ITERATION_CEILING
    return min(raw, _ITERATION_CEILING)


def _tool_repeat_key(name: str, args: dict | None) -> str:
    """Identity for the repeat guard: tool name + a hash of its args.

    The guard must block a *genuine* repeat (same tool, same inputs — an
    infinite loop) while ALLOWING the same tool with DIFFERENT inputs. A
    non-cacheable tool like ``web`` is still legitimately called several
    times in one turn with different queries (search, read, refine). Keying
    the guard on name alone killed that — the model's second, different
    search was blocked and the turn ended with no answer. Keying on
    name+args fixes it: identical calls are blocked, distinct ones pass.
    """
    import hashlib
    import json
    try:
        blob = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        blob = repr(args)
    return f"{name}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]}"

# Module-level build state — shared across handler instances, survives request lifecycle.
# Polled by /api/artifacts/build-status endpoint.
ACTIVE_BUILDS: dict[str, dict] = {}

_CHAT_SYNTHESIS_HINT = (
    "Synthesize the results into a clear, well-structured response. "
    "Cite sources inline using [1], [2], etc. when referencing search results. "
    "Prefer prose with inline citations over raw bullet lists of results."
)


# Per-tool synthesis guidance — tells the LLM *how* to present results
# from specific tools.  Keyed by tool name, injected after execution so
# the LLM has clear direction for its final response.  These are hidden
# from the user — they shape quality without appearing in output.
_TOOL_SYNTHESIS_GUIDANCE: dict[str, str] = {
    "web": (
        "Synthesize search findings into a coherent answer with inline citations [1], [2], etc. "
        "Draw connections between sources. Don't list results — weave them into flowing prose. "
        "End with a brief Sources list of the most relevant URLs."
    ),
    "web_search": (
        "Synthesize search findings into a coherent answer with inline citations [1], [2], etc. "
        "Draw connections between sources. Don't list results — weave them into flowing prose. "
        "End with a brief Sources list of the most relevant URLs."
    ),
    "wikipedia": (
        "Present the information naturally — don't prefix with 'According to Wikipedia.' "
        "Extract the key facts and context the user actually needs. "
        "If multiple articles were retrieved, synthesize them into a unified answer."
    ),
    "youtube": (
        "Summarize the video's key points and insights — don't paste raw transcript text. "
        "Use timestamps only when they add value (e.g. referencing a specific moment). "
        "Capture the speaker's main arguments and conclusions."
    ),
    "python_exec": (
        "Explain what the code produced and what it means. If there are numbers or data, "
        "interpret them in context. Don't just paste output — give the user insight. "
        "If the code errored, explain what went wrong and suggest a fix."
    ),
    "image_generation": (
        "Briefly describe the image you created and how it matches the request. "
        "Do NOT call image_generation again unless the user explicitly asks. "
        "The image is already visible — focus on any remaining parts of their question."
    ),
    "image_search": (
        "Reference the found image naturally in your response. "
        "Describe what it shows and how it relates to the user's question."
    ),
    "create_ebook": (
        "Let the user know their ebook is ready. Mention the title, chapter count, "
        "and illustration count. Provide the download link prominently. "
        "Don't reproduce the chapter content — the book speaks for itself."
    ),
    "export_markdown": (
        "Confirm the file was saved. Provide the download link. "
        "Don't reproduce the exported content in your response."
    ),
    "export_csv": (
        "Confirm the data was exported. Provide the download link. "
        "Briefly note what the data contains (row count, columns) without reproducing it."
    ),
    "export_code": (
        "Confirm the code file was saved. Provide the download link. "
        "Briefly note what the code does without reproducing it."
    ),
    "search_files": (
        "Present the user's files naturally — names, types, and descriptions. "
        "If they asked for something specific, highlight the most relevant matches. "
        "Don't dump raw metadata — describe what you found conversationally."
    ),
    "schedule_briefing": (
        "The briefing was created — the tool's output already reads as "
        "a natural confirmation. Relay it conversationally in one or "
        "two sentences (e.g. 'Got it — set up your Bitcoin briefing for "
        "2:09am daily, first fires tomorrow morning.'). Do NOT call "
        "schedule_briefing again, do NOT quote the structured output "
        "verbatim, do NOT echo TOOL_CALL/TOOL_INPUT or JSON. If the "
        "output mentions defaulted fields, mapped gather_tools, or "
        "ignored unknown values, surface those to the user briefly so "
        "they can correct. The first run time is in UTC — if it would "
        "help, you may translate to the user's local clock."
    ),
    "cancel_briefing": (
        "Confirm the cancellation conversationally in one short "
        "sentence. Do NOT call cancel_briefing again. Do NOT quote the "
        "raw output. If the tool reported ambiguity ('multiple match'), "
        "list the matching titles and ask the user which one."
    ),
    "list_briefings": (
        "Present the briefings naturally — title, time, cadence. "
        "Group active vs. delivered if both exist. Do NOT call "
        "list_briefings again. Do NOT dump raw metadata fields; if "
        "there's an error flag (⚠) or pause state, mention it plainly."
    ),
}


def _build_tool_synthesis_guidance(succeeded_tools: set[str]) -> str:
    """Build combined synthesis guidance for all tools that executed successfully."""
    parts = []
    seen = set()
    for tool_name in succeeded_tools:
        guidance = _TOOL_SYNTHESIS_GUIDANCE.get(tool_name, "")
        if guidance and guidance not in seen:
            parts.append(guidance)
            seen.add(guidance)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model size detection — determines synthesis prompt verbosity
# ---------------------------------------------------------------------------

# Patterns that indicate small models (< ~14B parameters)
_SMALL_MODEL_PATTERNS = re.compile(
    r"(?:^|[:\-/])(\d+(?:\.\d+)?)[bB]\b",
)


def _estimate_model_tier(model_name: str, backend) -> str:
    """Estimate model capability tier from name and backend type.

    Returns "small" (<14B), "medium" (14-70B), or "large" (70B+ / cloud API).
    """
    if not model_name:
        return "medium"  # safe default

    # Cloud APIs are always capable
    from augmentum.models.openai_compat import OpenAIBackend
    if isinstance(backend, OpenAIBackend):
        base_url = getattr(backend, "_base_url", "")
        if any(host in base_url for host in (
            "openai.com", "anthropic.com", "googleapis.com",
            "together.xyz", "groq.com", "deepseek.com",
            "openrouter.ai", "mistral.ai",
        )):
            return "large"

    # Parse parameter count from model name
    m = _SMALL_MODEL_PATTERNS.search(model_name.lower())
    if m:
        try:
            param_b = float(m.group(1))
            if param_b < 14:
                return "small"
            if param_b < 70:
                return "medium"
            return "large"
        except ValueError:
            pass

    return "medium"


# ---------------------------------------------------------------------------
# Result quality gating — prevents synthesizing empty/garbage results
# ---------------------------------------------------------------------------

def _assess_result_quality(
    tool_name: str, output: str, success: bool, failure_kind: str = "",
) -> str:
    """Score tool result quality. Returns 'good', 'partial', 'empty' or 'broken'.

    ``broken`` means the tool never actually ran (it raised, or the call
    was malformed). That is NOT the same as ``empty`` — "the search
    found nothing" is evidence about the world, "the search crashed" is
    evidence about us. Collapsing the two is what produced the
    2026-07-26 failure: a tool crash was scored ``empty``, the synthesis
    addendum told the model the tools "returned no useful results" and
    to answer from its own knowledge, and the model went on to doubt
    source-backed material it had already cited. Keep them distinct.
    """
    if not success:
        return "broken" if failure_kind in ("internal_error", "invalid_input") else "empty"
    text = output.strip()
    if not text or text == "(no output)":
        return "empty"

    # Search tools: check for actual result content
    if tool_name in ("web", "web_search"):
        # Count numbered result markers [N]
        import re as _re
        result_count = len(_re.findall(r"\[\d+\]", text))
        if result_count < 1:
            return "partial"

    # Code execution: check for errors in output
    if tool_name == "python_exec":
        if "Traceback" in text or "Error:" in text:
            return "partial"

    # Image: check for actual URL
    if tool_name in ("image_generation", "image_search"):
        if "/api/" not in text and "http" not in text:
            return "partial"

    return "good"


def _quality_synthesis_addendum(qualities: dict[str, str]) -> str:
    """Build synthesis addendum based on result quality scores."""
    all_empty = all(q == "empty" for q in qualities.values())
    any_empty = any(q == "empty" for q in qualities.values())
    any_partial = any(q == "partial" for q in qualities.values())
    any_broken = any(q == "broken" for q in qualities.values())

    # A tool that FAILED gets its own guidance, checked first. Telling
    # the model "no useful results, answer from your own knowledge"
    # after a crash invites it to treat already-established, cited facts
    # as unverified — the exact regression this branch exists to stop.
    # The honest instruction is: this is our fault, don't reinterpret it
    # as evidence, and don't retract what you already sourced.
    if any_broken:
        _all = all(q == "broken" for q in qualities.values())
        scope = "The tool" if _all else "One of the tools"
        return (
            f" {scope} failed with an internal error and never ran — this is a "
            "problem on our side, NOT a sign that the information doesn't "
            "exist or that anything you established earlier was wrong. Do not "
            "revise or walk back facts you already sourced. Say plainly that "
            "the lookup failed, answer from what you already have, and offer "
            "to retry."
        )
    if all_empty:
        return (
            " The tools returned no useful results. "
            "Answer from your own knowledge and note that you couldn't verify externally."
        )
    if any_empty:
        return (
            " Some tools returned no results. Use what's available and "
            "note any gaps where you're relying on your own knowledge."
        )
    if any_partial:
        return (
            " Some tool results are incomplete or contain errors. "
            "Use what's available but note any limitations."
        )
    return ""


# ---------------------------------------------------------------------------
# Model-adaptive synthesis prompts
# ---------------------------------------------------------------------------

_SYNTH_EXPLICIT = (
    "IMPORTANT: You just used tools to gather information. Now write your "
    "final response to the user. Follow these rules exactly:\n"
    "1. DO NOT include any tool names, function calls, or JSON in your response\n"
    "2. DO NOT start with 'Based on the search results' or 'According to' — just answer naturally\n"
    "3. Write in clear prose — no bullet-point dumps of raw results\n"
    "4. If search results were provided, mention sources with [1], [2] numbers\n"
    "5. Focus on what the user actually asked\n\n"
)

_SYNTH_STANDARD = (
    "Use the tool results above to answer the user's question. "
    "Do NOT repeat the raw tool output — synthesize a natural response. "
)

_SYNTH_COMPACT = (
    "Synthesize the results into a clear response. "
    "Cite sources [1], [2] where applicable. Don't repeat raw output. "
)


def _build_synthesis_fallback(results: dict) -> str:
    """Build a minimal response from raw step results when synthesis times out."""
    parts = ["*(Chain completed but response generation timed out. Raw results below:)*\n"]
    for r in results.values():
        if hasattr(r, "success") and r.success and hasattr(r, "output"):
            tool_name = getattr(r, "tool_name", "step")
            parts.append(f"**{tool_name}:** {r.output[:500]}")
    return "\n\n".join(parts)


async def _stream_with_timeout(gen, timeout: float):
    """Wrap an async generator with a per-chunk timeout."""
    while True:
        try:
            chunk = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
            yield chunk
        except StopAsyncIteration:
            break
        except TimeoutError:
            yield InternalStreamChunk(
                content_delta="\n\n*(Response generation timed out)*",
            )
            break


class PassthroughHandler(ModeHandler):
    """Directly proxies requests to the model backend.

    When a ToolRegistry is provided and tools are requested (via header or
    config), the handler adds tool schemas to the request and executes any
    tool calls the LLM returns before forwarding the final text response.

    Supports multi-step tool chains when ``passthrough_chain_enabled`` is set:
    1. Check for matching custom flow (trigger pattern or /flow command)
    2. If complex query detected, use adaptive chain planning
    3. Otherwise, fall back to the simple tool loop
    """

    def __init__(
        self,
        backend: ModelBackend,
        image_queue: GenerationQueue | None = None,
        image_enabled: bool = False,
        session_id: str = "",
        tool_registry: ToolRegistry | None = None,
        enabled_tools: list[str] | None = None,
        tool_synthesis_hint: str = "",
        custom_flow_store: object = None,
        user_id: str = "",
        app_state: object | None = None,
    ) -> None:
        self._backend = backend
        self._image_queue = image_queue
        self._image_enabled = image_enabled
        self._session_id = session_id
        self._tool_registry = tool_registry
        self._enabled_tools = enabled_tools or []
        self._tool_synthesis_hint = tool_synthesis_hint
        self._custom_flow_store = custom_flow_store
        self._user_id = user_id
        self._app_state = app_state
        # Direct ("auto-invoke when enabled") tool execution. True for the
        # chat path, where enabled_tools reflects explicit per-tool button
        # toggles — turning a tool's button on IS the intent signal. The
        # voice path flips this off: it enables tools via a blanket ['all']
        # sentinel (not per-tool intent), so auto-firing youtube/etc. every
        # turn is wrong there. See voice_routes after get_handler_for_mode.
        self._auto_invoke_enabled = True
        # Per-turn allowlist for direct invocation. ``None`` = legacy (every
        # enabled auto-invoke tool may fire — preserved for callers that don't
        # set it). A set restricts auto-invocation to tools the user selected
        # for THIS turn (the X-Augmentum-Tools header), so config-default /
        # blanket-"all" tools (youtube, build_application) no longer fire on
        # every message. Chat routes set this from the request header.
        self._auto_invoke_tools: set[str] | None = None
        # Whether heavy "gated" capabilities are PROPOSED via a confirmation
        # chip instead of running inline. Gating exists to guard against acting
        # on MISREAD intent — which only happens when the input itself is
        # uncertain (voice/STT). Default False: in text chat the user typed the
        # request, so intent is explicit and image_generation runs inline (its
        # live tool card + inline result, the pre-offer-substrate behavior). The
        # voice path flips this True so a stray STT artifact can't silently kick
        # off a generation. See voice_routes after get_handler_for_mode, and
        # _should_gate_capability for the per-tool policy.
        self._gate_heavy_tools = False
        # Accumulates image URLs generated by tools during a request
        self._generated_images: list[str] = []
        # First non-streaming response (used as fallback in _handle)
        self._first_response = None
        # SSOS orchestrator — zero-cost heuristic tool path (independent of enabled_tools)
        self._ssos = None
        if tool_registry:
            from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
            self._ssos = SSOSOrchestrator(
                tool_registry, user_id=user_id, app_state=app_state,
            )
            # One executor for every tool action (2026-07-02): SSOS
            # heuristic + marker paths run through _execute_tool, so
            # coercion, user/session context, metrics, cards, and the
            # tool-card presentation match the native loop exactly.
            self._ssos.bind_executor(self._execute_tool)

    # ------------------------------------------------------------------
    # Tool resolution
    # ------------------------------------------------------------------

    def _resolve_tools(self) -> list[Tool]:
        """Return Tool objects for the enabled tool names."""
        if not self._tool_registry or not self._enabled_tools:
            return []

        tools: list[Tool] = []
        seen: set[str] = set()
        for name in self._enabled_tools:
            tool = self._tool_registry.resolve(name.strip())
            if tool and tool.name not in seen:
                tools.append(tool)
                seen.add(tool.name)
            elif not tool:
                log.warning("passthrough_tool_not_found", name=name)
        self._append_schedule_substrate(tools, seen)
        return tools

    def _resolve_auto_capability_tools(self) -> list[Tool]:
        """Resolve Auto-mode lookup capabilities into native Tool objects.

        Auto mode = no tools explicitly selected. Instead of the old
        ``[[tool:NAME]]`` text-marker "soft-trigger" (which models trained on
        native function-calling reliably missed — they emit native tool_calls
        or reason in the thinking channel, so the marker never appeared on the
        first visible line and the call silently vanished into the thinking
        dropdown with no follow-up), expose the SSOS lookup capabilities
        (web_search, wikipedia, youtube, image_search) as real tool schemas and
        let the native tool-calling loop drive them. The model decides whether
        to call one; if it doesn't, it just answers.
        """
        if not self._ssos or not self._tool_registry:
            return []
        out: list[Tool] = []
        seen: set[str] = set()
        for cap in self._ssos.lookup_capabilities():
            name = (cap.tool or "").strip()
            if not name or name in seen:
                continue
            tool = self._tool_registry.resolve(name)
            if tool is not None and tool.name not in seen:
                out.append(tool)
                seen.add(tool.name)
        # Gated capabilities (image_generation, build_application, the
        # structured creators) split by policy (_should_gate_capability):
        #   * PROPOSE  → exposed as a single-arg PROXY tool. The model can
        #     REQUEST it via native function-calling; the parsed call is
        #     intercepted (_first_gated) and turned into a confirmation offer
        #     rather than executed (used for inferred multi-step builds, and
        #     for image_generation under voice where STT may misfire).
        #   * INLINE   → exposed as the REAL tool so the native loop executes
        #     it directly, restoring its live tool card + inline result. This
        #     is image_generation in text chat: the user typed the request, so
        #     intent is explicit and there's nothing to confirm.
        # Either way the model SEES image_generation in Auto mode, so it never
        # (correctly, from its roster) denies it can generate images.
        from augmentum.modes.passthrough.gated_proxy import build_gated_proxy_tools
        gated_caps = self._ssos.gated_capabilities()
        inline_caps = [c for c in gated_caps if not self._should_gate_capability(c.tool)]
        propose_caps = [c for c in gated_caps if self._should_gate_capability(c.tool)]
        for cap in inline_caps:
            real = self._tool_registry.resolve(cap.tool)
            if real is not None and real.name not in seen:
                if not _capability_dep_available(real):
                    continue
                out.append(real)
                seen.add(real.name)
        for proxy in build_gated_proxy_tools(propose_caps, self._tool_registry):
            if proxy.name not in seen:
                out.append(proxy)
                seen.add(proxy.name)
        self._append_schedule_substrate(out, seen)
        return out

    def _append_schedule_substrate(self, tools: list[Tool], seen: set[str]) -> None:
        """Union the always-on scheduling substrate into the tool pool.

        The 2026-07-02 scheduling-substrate policy: timed action is an
        app-level capability, so the model must ALWAYS be able to infer a
        scheduling call — the regex tier in ``tools/filter.py`` is a
        fast-path, not a gate. Without this, phrasings the patterns miss
        ("set a tracker for bitcoin every 15 minutes") leave the model
        with no scheduling schema at all and it silently can't schedule.
        Names resolve against the registry, so installs without the
        scheduling tools registered are unaffected.
        """
        from augmentum.tools.filter import _SCHEDULE_INJECTION_ORDER

        if not self._tool_registry:
            return
        for name in _SCHEDULE_INJECTION_ORDER:
            tool = self._tool_registry.resolve(name)
            if tool is not None and tool.name not in seen:
                tools.append(tool)
                seen.add(tool.name)

    def _should_gate_capability(self, tool_name: str) -> bool:
        """Whether a gated capability is PROPOSED (confirmation chip) rather
        than run inline.

        Gating guards against acting on misread intent, so it belongs only
        where the intent is uncertain:

        * **Explicitly enabled** (the user ticked the tool's button) — never
          gate. The button IS the consent; run inline. This is the
          orchestrator's own contract (see ``_GATED_CAPABILITIES`` doc).
        * **image_generation** — gate only under ``_gate_heavy_tools`` (voice,
          where an STT artifact could trigger a stray generation). In text the
          request was typed, so it runs inline with its live card + inline
          image, the pre-offer-substrate behavior.
        * **create_chart / create_spreadsheet** — never gate. They render in
          about a second (in-process matplotlib / openpyxl), write nothing the
          user must undo, and persist an artifact row they can delete. A
          "shall I draw this?" chip is pure friction on the case they exist
          for: the model noticing a data-shaped answer reads better as a
          picture and just showing it, the way a generated image already
          appears inline.
        * **Multi-step builders** (build_application + the remaining artifact
          creators) — keep the confirm when intent was merely inferred: they
          spin up multi-minute work with their own native progress surfaces, so
          a confirm before spending those minutes is legitimate, not a
          regression.
        """
        if tool_name in _NEVER_GATED_CAPABILITIES:
            return False
        if tool_name in self._enabled_tools:
            return False
        # Explicit verbal invocation = consent (2026-07-08). The gate exists
        # for INFERRED intent; when the live user message names the capability
        # ("use the multi-file app builder"), proposing again creates an
        # eternal offer loop — the user says yes in words, the model re-calls
        # the proxy, another chip appears, forever. Naming the tool is the
        # same consent as ticking its checkbox.
        if self._explicit_capability_consent(tool_name):
            return False
        if tool_name == "image_generation":
            return self._gate_heavy_tools
        return True

    _EXPLICIT_CAP_PHRASES: dict[str, tuple[str, ...]] = {
        "build_application": (
            "app builder", "application builder", "multi-file", "multi file",
            "multifile", "build_application", "the builder tool",
        ),
        "image_generation": ("image generator", "image_generation"),
        "create_ebook": ("ebook builder", "ebook tool", "create_ebook"),
        "create_presentation": (
            "presentation builder", "slides builder", "create_presentation",
        ),
        "create_document": ("document builder", "create_document"),
    }

    def _stash_turn_user_text(self, request) -> None:
        """Remember the live turn's user message for gate-consent checks."""
        try:
            for m in reversed(getattr(request, "messages", []) or []):
                if getattr(m, "role", "") == "user":
                    self._turn_user_text = str(getattr(m, "content", "") or "")
                    return
        except Exception:
            self._turn_user_text = ""

    def _explicit_capability_consent(self, tool_name: str) -> bool:
        """True when the current user message explicitly names *tool_name*'s
        capability. Conservative: only capability-NAMING phrases count —
        generic acceptance ("yes", "go ahead") stays with the offer chips,
        which carry the original proposal's context."""
        text = (getattr(self, "_turn_user_text", "") or "").lower()
        if not text:
            return False
        phrases = self._EXPLICIT_CAP_PHRASES.get(tool_name, ())
        return any(ph in text for ph in phrases)

    def _first_gated(
        self, calls: list[tuple[str, dict, str]],
    ) -> tuple[object, str] | None:
        """If any parsed tool call targets a gated capability, return
        ``(capability, brief)``; otherwise ``None``.

        Gated capabilities are PROPOSED, not run — the caller surfaces a
        confirmation offer (Accept / Not now / Never) instead of executing, so
        the user has an exit when the intent was inferred wrong. ``brief`` is the
        capability's ``primary_arg`` value (a prompt / description / one-line
        brief), falling back to the first non-empty string argument.
        """
        if not self._ssos:
            return None
        by_name = {c.tool: c for c in self._ssos.gated_capabilities()}
        for name, args, _tc in calls:
            cap = by_name.get(name)
            # Only PROPOSE-policy capabilities are intercepted; an inline-policy
            # one (e.g. image_generation in text chat) falls through to normal
            # execution so it keeps its live tool card + inline result.
            if cap is None or not self._should_gate_capability(name):
                continue
            brief = ""
            if isinstance(args, dict):
                brief = str(args.get(cap.primary_arg) or "").strip()
                if not brief:
                    brief = next(
                        (str(v).strip() for v in args.values()
                         if isinstance(v, str) and v.strip()),
                        "",
                    )
            else:
                brief = str(args or "").strip()
            return cap, brief
        return None

    async def _gated_response(
        self, cap: object, brief: str, model: str,
    ) -> InternalChatResponse:
        """Synthetic assistant response for a proposed gated capability: surface
        the confirmation offer and carry the warm lead-in as the turn's answer
        (used by the non-streaming resolvers)."""
        lead = await self._surface_gated_offer(cap, brief, model=model)
        return InternalChatResponse(
            message=Message(role="assistant", content=lead),
            model=model,
            finish_reason="stop",
        )

    def _inject_auto_capability_self_model(
        self, request: InternalChatRequest, tools: list[Tool],
    ) -> None:
        """Give the model a STABLE self-model of its Auto-mode capabilities.

        Auto mode filters tool SCHEMAS per turn by query relevance, so on a turn
        whose query doesn't mention images the ``image_generation`` proxy is
        dropped — and the model, seeing only search tools that turn, wrongly
        DENIES it can make images at all ("my toolset today is search and
        Wikipedia"). This injects a compact, always-present capability line
        (the same digest the companion uses) built from the FULL resolved Auto
        toolset, so the model's self-knowledge stays constant even as the
        per-turn schema changes. Only capabilities whose tools actually resolved
        on this install are claimed — nothing is over-promised.
        """
        if not tools:
            return
        from augmentum.companion_runtime.capability_digest import (
            build_capability_digest,
        )
        digest = build_capability_digest({t.name for t in tools})
        if not digest:
            return
        for m in request.messages:
            if m.role == "system":
                if digest not in (m.content or ""):
                    m.content = f"{(m.content or '').rstrip()}\n\n{digest}".strip()
                return
        request.messages.insert(0, Message(role="system", content=digest))

    def _chain_extra_tool_args(self) -> dict:
        """Build the ``extra_tool_args`` dict for chain calls.

        Threads ``session_id`` so chain-routed IMAGE/ARTIFACT tools satisfy
        their FK constraint on insert (see GenerationJob.session_id and
        artifact_store ownership). ``user_id`` is passed via ``cache_user_id``
        instead — that drives the chain's own ``_user_id`` / ``_context``
        injection at ``chain.py`` line 478.
        """
        extras: dict = {}
        if self._session_id:
            extras["session_id"] = self._session_id
        return extras

    def _collect_direct_invoke_tools(self) -> list[Tool]:
        """Return Tool objects that opt into ``auto_invoke_when_enabled``.

        The handler calls these directly (with the user's message as the
        query) when they're present in ``enabled_tools``, bypassing the
        LLM's tool-selection step. The button is the intent signal.

        Long-running tools (``long_running=True``) take over the whole
        response — they're run in the background with their own
        progress UI, and other direct tools are skipped for that turn.
        """
        # Voice and other blanket-enable callers disable direct invocation —
        # there's no per-tool intent signal, so the LLM must choose.
        if not self._auto_invoke_enabled:
            return []
        tools = self._resolve_tools()
        candidates = [
            t for t in tools
            if getattr(t, "auto_invoke_when_enabled", False)
        ]
        # Restrict to the per-turn header selection when one was supplied.
        # Without this, a tool merely present in the config-default set (or the
        # blanket-"all"/training override) auto-fires on every message and
        # hijacks the turn onto the direct-invoke path — which also drops tool
        # schemas from training traces. ``None`` keeps the legacy behavior.
        if self._auto_invoke_tools is not None:
            candidates = [t for t in candidates if t.name in self._auto_invoke_tools]
        return candidates

    def _note_direct_invoke_capture(self, tools: list[Tool]) -> None:
        """Record direct-invoked tool schemas into the active training trace.

        The direct-invoke path synthesizes tool results into the prompt and
        clears ``request.tools``, so the backend-boundary capture hook sees no
        schemas — a deterministic-tool turn would otherwise be recorded as a
        plain chat turn. Note them explicitly. No-op when capture is inactive
        and never affects inference.
        """
        if not tools:
            return
        try:
            from augmentum.modes.analytical.tool_calling import tools_to_native_format
            from augmentum.training.trace_context import note_tool_schemas
            note_tool_schemas(tools_to_native_format(tools))
        except Exception:
            log.debug("direct_invoke_capture_note_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self, tool: Tool, params: dict,
        progress_queue: asyncio.Queue | None = None,
    ) -> tuple[str, dict]:
        """Execute a single tool and return (truncated_output, metadata).

        If *progress_queue* is provided, a ``_progress_callback`` kwarg is
        injected into the tool params so long-running tools (e.g. application
        builder) can stream intermediate progress to the client.
        """
        # --- Artifact pipeline intercept ---
        from augmentum.tools.artifact_pipeline import ARTIFACT_TOOLS

        if tool.name in ARTIFACT_TOOLS:
            try:
                from augmentum.tools.artifact_pipeline import (
                    ArtifactRequest,
                    PipelineContext,
                    build_backend_pipeline_caller,
                    execute_artifact_pipeline,
                )

                # Determine format from tool name
                fmt_map = {
                    "create_document": params.get("format", "pdf"),
                    "create_presentation": "pptx",
                    "create_spreadsheet": "xlsx",
                    "create_chart": "chart",
                }
                fmt = fmt_map.get(tool.name, "pdf")
                topic = params.get("title", params.get("topic", "Document"))

                art_req = ArtifactRequest(
                    format=fmt,
                    topic=topic,
                    title=params.get("title"),
                    theme=params.get("theme"),
                    tool_params=params,
                )

                # Build context from current request messages
                msg_history = []
                current_req = getattr(self, "_current_request", None)
                if current_req and current_req.messages:
                    msg_history = [
                        {"role": m.role, "content": m.content}
                        for m in current_req.messages
                    ]

                ctx = PipelineContext(
                    message_history=msg_history,
                    generated_images=list(self._generated_images),
                )

                model = getattr(self, "_current_model", "")
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

                result = await execute_artifact_pipeline(
                    art_req, ctx, caller,
                    _search_tool=search_tool,
                    _fetch_tool=fetch_tool,
                    _render_tools=render_tools,
                )

                output = (
                    f"Created {result.display_name}: {result.download_url}"
                    if result.download_url
                    else f"Created artifact: {result.artifact_id}"
                )
                return output, result.metadata
            except Exception as exc:
                log.warning("artifact_pipeline_fallthrough",
                            tool=tool.name, error=str(exc))
                # Fall through to normal execution

        from augmentum.modes.analytical.tool_calling import coerce_tool_params

        try:
            params = coerce_tool_params(tool, params)
        except Exception:  # noqa: BLE001 — coercion is best-effort normalization;
            # a malformed schema must not kill the execution itself.
            log.debug("tool_param_coerce_failed",
                      tool=getattr(tool, "name", "?"), exc_info=True)

        # Build internal params — these are only injected if the tool accepts them
        _internal: dict = {}
        if progress_queue is not None:
            async def _progress_cb(data: dict) -> None:
                from augmentum.models.base import InternalStreamChunk
                content_text = data.pop("_content_delta", "")
                await progress_queue.put(InternalStreamChunk(
                    content_delta=content_text,
                    augmentum=data,
                ))
            _internal["_progress_callback"] = _progress_cb
        if hasattr(self, "_current_model"):
            _internal["_request_model"] = self._current_model
        _ctx: dict = {}
        if self._user_id:
            _ctx["user_id"] = self._user_id
        if self._session_id:
            # Needed for tools that persist DB rows FK'd to sessions —
            # image_generation's save_generation() writes session_id,
            # and an empty value caused FOREIGN KEY failures for every
            # illustration during ebook creation.
            _ctx["session_id"] = self._session_id
        if self._app_state is not None:
            _ctx["file_index"] = getattr(self._app_state, "file_index", None)
        # Mode stamp for the offer substrate's allowed_modes gate.
        _ctx["mode"] = "passthrough"
        # Provenance stamp — set by owners that borrow this executor
        # for a different actor (the companion's native loop sets
        # "companion"). Deliberately a SEPARATE key from mode so the
        # offer substrate's allowed_modes gate is untouched. Tools that
        # persist content (image_generation) thread it into the row's
        # origin column.
        _origin = getattr(self, "_ctx_origin", "")
        if _origin:
            _ctx["origin"] = _origin
            # Artifact provenance choke point (Phase 7): any artifact
            # this tool call saves carries the same origin — covers
            # create_document/.../image_search/remove_background
            # without per-tool wiring. Task-local; reset is unneeded
            # (the var is re-set per call and defaults '' elsewhere).
            try:
                from augmentum.tools.artifact_storage import ARTIFACT_ORIGIN
                ARTIFACT_ORIGIN.set(_origin)
            except Exception:  # noqa: BLE001
                pass
        else:
            # Explicit reset — a user-chat tool call in a task that
            # previously ran a companion call must not inherit the
            # stale stamp.
            try:
                from augmentum.tools.artifact_storage import ARTIFACT_ORIGIN
                ARTIFACT_ORIGIN.set("")
            except Exception:  # noqa: BLE001
                pass
        if _ctx:
            _internal["_context"] = _ctx

        # Only inject internal params the tool's execute() can accept.
        # Signature introspection is best-effort — exotic callables and
        # test mocks can defeat inspect; skip injection rather than kill
        # the execution.
        if _internal:
            import inspect
            try:
                sig = inspect.signature(tool.execute)
            except Exception:  # noqa: BLE001
                sig = None
                log.debug("tool_signature_inspect_failed",
                          tool=getattr(tool, "name", "?"))
            if sig is not None:
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                for key, val in _internal.items():
                    if accepts_kwargs or key in sig.parameters:
                        params[key] = val

        metadata: dict = {}
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                # invoke() — not execute() — so schema coercion and
                # list fan-out apply here exactly as they do on every
                # other surface. See Tool.invoke.
                invoke_tool(tool, params),
                timeout=tool.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000
            if self._tool_registry:
                self._tool_registry.metrics.record(
                    tool.name, success=result.success, elapsed_ms=elapsed,
                )
            if result.success:
                output = result.output
            else:
                # Enrich error with tool-specific recovery hints
                output = f"Error: {tool.enrich_error(result.error, params)}"
            metadata = result.metadata or {}
            # Carry WHY it failed to the synthesis layer. Without this the
            # quality scorer can only see success=False and would score a
            # crash the same as a genuine no-results search.
            if not result.success and getattr(result, "failure_kind", ""):
                metadata["failure_kind"] = result.failure_kind
            # Pass the structured ToolResult.card through so _on_tool_result
            # can attach it to the tool_call event. Without this, artifact
            # tools (ebook, etc.) return a card the frontend never sees and
            # the user just gets the raw output text.
            if getattr(result, "card", None):
                metadata["card"] = result.card
        except TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            if self._tool_registry:
                self._tool_registry.metrics.record(
                    tool.name, success=False, elapsed_ms=elapsed,
                )
            raw_err = f"Tool '{tool.name}' timed out after {tool.timeout}s"
            output = f"Error: {tool.enrich_error(raw_err, params)}"
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            if self._tool_registry:
                self._tool_registry.metrics.record(
                    tool.name, success=False, elapsed_ms=elapsed,
                )
            output = f"Error: {tool.enrich_error(str(exc), params)}"

        # Truncate long output
        if len(output) > _TOOL_RESULT_MAX:
            tail = settings.tool_result_truncation_tail
            output = output[: _TOOL_RESULT_MAX - tail] + "\n...\n" + output[-tail:]

        return output, metadata

    # ------------------------------------------------------------------
    # Direct-invoke tools (button-authoritative)
    # ------------------------------------------------------------------

    async def _invoke_direct_tool(
        self, tool: Tool, user_msg: str,
    ) -> tuple[str, dict, bool, int]:
        """Run an auto-invoke-enabled tool with the user's message as query.

        Returns ``(output, metadata, success, duration_ms)``. The caller
        is responsible for emitting tool_start/complete events and for
        injecting the output into the synthesis request.
        """
        start = time.monotonic()
        output, metadata = await self._execute_tool(tool, {"query": user_msg})
        elapsed_ms = int((time.monotonic() - start) * 1000)
        success = not output.startswith("Error:")
        return output, metadata, success, elapsed_ms

    def _build_direct_invoke_events(
        self, tool_name: str, user_msg: str,
        output: str, metadata: dict, success: bool, duration_ms: int,
    ) -> list[InternalStreamChunk]:
        """Emit the three unified events (start/call/complete) for a
        direct-invoked tool.

        Returned as a list so both the streaming and non-streaming
        paths can yield / drop them consistently.
        """
        from augmentum.tools.events import make_tool_complete, make_tool_start
        tc_id = f"direct_{uuid.uuid4().hex[:8]}"
        args = {"query": user_msg}
        snippet = output[:240]

        # Strip internal keys; UI only needs user-facing metadata.
        result_metadata = {
            k: v for k, v in (metadata or {}).items() if not k.startswith("_")
        }

        chunks: list[InternalStreamChunk] = []
        chunks.append(InternalStreamChunk(
            content_delta="",
            augmentum={"tool_start": make_tool_start(
                tc_id, tool_name, args, phase="passthrough",
            )},
        ))
        tool_data = {
            "tool": tool_name,
            "success": success,
            "output_preview": snippet,
            "phase": "passthrough",
            "result_metadata": result_metadata,
        }
        if metadata.get("card"):
            tool_data["card"] = metadata["card"]
        chunks.append(InternalStreamChunk(
            content_delta="",
            augmentum={"tool_call": tool_data},
        ))
        extra = {"result_metadata": result_metadata} if result_metadata else None
        chunks.append(InternalStreamChunk(
            content_delta="",
            augmentum={"tool_complete": make_tool_complete(
                tc_id, tool_name,
                success=success,
                output_preview=snippet,
                error="" if success else snippet,
                duration_ms=duration_ms,
                extra=extra,
            )},
        ))
        return chunks

    def _inject_direct_invoke_synthesis(
        self,
        request: InternalChatRequest,
        tool_outputs: list[tuple[str, str]],
    ) -> None:
        """Mutate ``request`` so the next backend call synthesizes over
        the direct-invoked tool results instead of calling tools.

        Tool outputs are consolidated into a system message inserted
        before the trailing user turn. The model is told the UI is
        already showing the structured result — it only needs to write
        a brief caption, not recap titles/URLs.
        """
        if not tool_outputs:
            return
        blocks = []
        for name, out in tool_outputs:
            blocks.append(f"[{name}]\n{out}")
        joined = "\n\n".join(blocks)
        guidance = (
            "The tools above already ran directly (the user turned them "
            "on from the picker). Their structured results — video "
            "cards, images, project previews — are being rendered by "
            "the UI right now, next to your message. Write a short "
            "caption (1–3 sentences) acknowledging what was found and "
            "inviting the user to explore. Do NOT recap titles, URLs, "
            "durations, thumbnails or other data the UI is already "
            "showing. If the tool returned an error, explain it briefly "
            "and suggest a next step."
        )
        synthesis_msg = Message(
            role="system",
            content=f"<tool_results>\n{joined}\n</tool_results>\n\n{guidance}",
        )
        # Insert just before the user's message so the model sees the
        # results as fresh context to the current turn.
        insert_idx = len(request.messages) - 1
        for i in range(len(request.messages) - 1, -1, -1):
            if request.messages[i].role == "user":
                insert_idx = i
                break
        request.messages.insert(insert_idx, synthesis_msg)
        # Drop tool schemas — the synthesis pass has nothing left to decide.
        request.tools = None

    async def _run_long_running_direct_tool_stream(
        self, tool: Tool, user_msg: str, model: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Fire-and-forget path for ``long_running`` direct-invoke tools.

        Spawns a background task and yields a kickoff chunk so the user
        can keep chatting. Progress is written to a tool-specific state
        dict (e.g. ``ACTIVE_BUILDS``) that the client polls.

        Currently specialized for ``build_application``. The tool-side
        plumbing (progress callback, ACTIVE_BUILDS wiring, cancellation,
        checkpoint recovery) lives here rather than on the tool because
        it's interleaved with request-scope state (session, model).
        """
        if tool.name != "build_application":
            # Placeholder for future long-running tools — fall through
            # to the normal direct-invoke path rather than silently
            # dropping the call.
            output, meta, success, dur_ms = await self._invoke_direct_tool(
                tool, user_msg,
            )
            for chunk in self._build_direct_invoke_events(
                tool.name, user_msg, output, meta, success, dur_ms,
            ):
                yield chunk
            return

        log.info("app_builder.direct_trigger", model=model)

        # Build mode runs on the SAME coder-workspace builder as the Library
        # "Build an app" button (augmentum.builds.facade.run_build), via the
        # shared start_app_build seam — a real Docker workspace + Playwright
        # behavior gate, not the retired in-process quickjs pipeline. The
        # workspace persists, so a rough build can be opened and continued with
        # the agent instead of being a dead-end artifact with no way to fix it.
        # Docker is a hard prerequisite for Augmentum, so there is no
        # lightweight fallback — if the stack is down we say so plainly rather
        # than silently shipping an unverified app.
        from augmentum.builds.dispatch import start_app_build
        ack = await start_app_build(
            self._app_state,
            objective=user_msg,
            user_id=self._user_id,
            session_id=self._session_id,
            model=model,
        )
        if not ack.get("ok"):
            msg = f"I couldn't start the build — {ack.get('detail', 'the builder is unavailable right now.')}"
            for chunk in self._build_direct_invoke_events(
                "build_application", user_msg, msg, {}, False, 0,
            ):
                yield chunk
            return

        build_id = ack["build_id"]
        project_name = ack.get("name") or "your app"

        yield InternalStreamChunk(
            content_delta=(
                f"Building **{project_name}** in a coder workspace — it's "
                "browser-tested as it goes, and the workspace stays open so we "
                "can keep working on it together. Feel free to keep chatting; "
                "the build card tracks progress.\n"
            ),
            augmentum={
                "build_started": {
                    "build_id": build_id,
                    "name": project_name,
                },
            },
        )

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    def _parse_tool_calls(
        self, response: InternalChatResponse, tools: list[Tool],
    ) -> list[tuple[str, dict, str]]:
        """Parse tool calls from the LLM response.

        Returns list of ``(tool_name, args_dict, tool_call_id)`` tuples.
        Tries all parsers regardless of tier so that text-tier tool calls
        (``TOOL_CALL: name / TOOL_INPUT: {...}``) are always caught.
        """
        from augmentum.modes.analytical.engine import AnalyticalEngine
        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier,
            parse_action_input_tool_call,
            parse_fuzzy_tool_call,
            parse_json_tool_calls,
            parse_native_tool_calls_all,
            parse_python_style_tool_call,
            parse_structured_output,
            parse_xml_tool_calls,
            select_tier,
        )

        tier = select_tier(self._backend, response.model or "")
        known = {t.name for t in tools}
        # OpenAI-compat backends sanitize function names (dots/dashes → "_")
        # before the model ever sees them, so a dotted registry id like
        # "weather.today" comes back from the model as "weather_today". Match on
        # the SAME normalized key the native loop's resolver uses, or every
        # dotted-namespace verb (weather.today, memory.recall, device.dial,
        # device.set_alarm, companion.introspect, …) is dropped here as "unknown"
        # the moment the model actually calls it — which both silences the tool
        # AND orphans the assistant tool_calls message (→ backend 400
        # "assistant message with 'tool_calls' must be followed by tool message"
        # on the next turn). See companion_runtime.native_loop._tool_name_collision_key.
        from augmentum.companion_runtime.native_loop import _tool_name_collision_key
        known_norm = {_tool_name_collision_key(n) for n in known}

        # Tier 1: native tool_calls
        if tier == ToolCallingTier.NATIVE:
            raw_calls = parse_native_tool_calls_all(response)
            if raw_calls:
                result = []
                # Extract tool_call_ids from raw tool_calls
                raw_tc = (response.message.tool_calls or []) if response.message else []
                for i, (name, args) in enumerate(raw_calls):
                    # Drop calls for tools the user didn't enable. Capable
                    # models hallucinate well-known names (web_search,
                    # image_generation) from training priors even when the
                    # schema only lists the curated set; without this filter
                    # the registry resolves any name globally and the
                    # phantom call runs anyway. Text-tier parsers below
                    # already enforce this — NATIVE was the gap.
                    if (
                        known
                        and name not in known
                        and _tool_name_collision_key(name) not in known_norm
                    ):
                        log.info(
                            "passthrough_native_tool_filtered",
                            tool=name, enabled=sorted(known),
                        )
                        continue
                    tc_id = ""
                    if i < len(raw_tc):
                        tc_id = raw_tc[i].get("id", "") or ""
                    if not tc_id:
                        tc_id = f"call_{uuid.uuid4().hex[:8]}"
                    result.append((name, args, tc_id))
                if result:
                    return result

        # Tier 2: structured JSON output
        if tier == ToolCallingTier.STRUCTURED:
            text = response.message.content if response.message else ""
            parsed = parse_structured_output(text)
            if parsed:
                name, args = parsed
                return [(name, args, f"call_{uuid.uuid4().hex[:8]}")]

        # Text-based fallbacks — always checked for all tiers so that models
        # producing text-format tool calls are caught even when tier is NATIVE
        # (e.g. local OpenAI-compat servers that output text instead of tool_calls)
        text = response.message.content if response.message else ""
        if text:
            # TOOL_CALL: / TOOL_INPUT: format (what _build_text_tool_prompt asks for)
            tc_name, tc_input = AnalyticalEngine._parse_tool_call(text)
            if tc_name and (not known or tc_name in known):
                log.info("passthrough_text_tool_parsed", tool=tc_name, format="tool_call")
                return [(tc_name, tc_input, f"call_{uuid.uuid4().hex[:8]}")]

            # JSON array/object tool calls in text content
            # (models sometimes emit [{"name": "tool", "arguments": {...}}])
            json_calls = parse_json_tool_calls(text, known)
            if json_calls:
                log.info("passthrough_text_tool_parsed", format="json",
                         tools=[c[0] for c in json_calls])
                return [
                    (name, args, f"call_{uuid.uuid4().hex[:8]}")
                    for name, args in json_calls
                ]

            # XML-style: <tool_call>, <tool_use>, <function_call> blocks
            xml_calls = parse_xml_tool_calls(text, known)
            if xml_calls:
                log.info("passthrough_text_tool_parsed", format="xml",
                         tools=[c[0] for c in xml_calls])
                return [
                    (name, args, f"call_{uuid.uuid4().hex[:8]}")
                    for name, args in xml_calls
                ]

            # ReAct-style: Action: tool / Action Input: {...}
            react_parsed = parse_action_input_tool_call(text, known)
            if react_parsed:
                name, args = react_parsed
                log.info("passthrough_text_tool_parsed", tool=name, format="react")
                return [(name, args, f"call_{uuid.uuid4().hex[:8]}")]

            # Python-style call in text
            py_parsed = parse_python_style_tool_call(text, known)
            if py_parsed:
                name, args = py_parsed
                log.info("passthrough_text_tool_parsed", tool=name, format="python")
                return [(name, args, f"call_{uuid.uuid4().hex[:8]}")]

            # Last resort: known tool name near a JSON object in free text
            fuzzy_parsed = parse_fuzzy_tool_call(text, known)
            if fuzzy_parsed:
                name, args = fuzzy_parsed
                log.info("passthrough_text_tool_parsed", tool=name, format="fuzzy")
                return [(name, args, f"call_{uuid.uuid4().hex[:8]}")]

        return []

    # ------------------------------------------------------------------
    # Tool schema injection
    # ------------------------------------------------------------------

    def _inject_tool_schemas(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
    ) -> ToolCallingTier:
        """Inject tool schemas into the request. Returns the tier used.

        For small models (< ~14B), appends each tool's ``model_hint`` to
        its description to compensate for weaker instruction following.
        """
        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier,
            build_structured_output_schema,
            select_tier,
            tools_to_native_format,
        )

        # Models tend to NARRATE what a rendering tool would produce (an image
        # "she paints a sunrise…", a chart as a markdown table) instead of
        # CALLING it, so nothing actually renders. For each such tool on the
        # menu, add an explicit call-don't-describe directive — see
        # _CALL_DONT_NARRATE_DIRECTIVES.
        # Dedupe by name up front so EVERY tier branch below (native /
        # enhanced / structured / text) emits unique function names. Blanket
        # "all" tool sets can surface the same name twice, and DeepSeek/OpenAI
        # HARD-400 on "Tool names must be unique."
        if tools:
            _seen: set[str] = set()
            _deduped: list[Tool] = []
            for _t in tools:
                if _t.name not in _seen:
                    _seen.add(_t.name)
                    _deduped.append(_t)
            tools = _deduped

        self._inject_call_dont_narrate_directives(request, tools)

        tier = select_tier(self._backend, request.model)
        model_tier = getattr(self, "_model_tier", "medium")

        # For small models, build enhanced schemas with model_hint appended
        if model_tier == "small" and tier == ToolCallingTier.NATIVE:
            enhanced = []
            for tool in tools:
                hint = tool.model_hint
                desc = f"{tool.description} {hint}" if hint else tool.description
                enhanced.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": desc,
                        "parameters": tool.input_schema or {
                            "type": "object",
                            "properties": {},
                        },
                    },
                })
            request.tools = enhanced
        elif tier == ToolCallingTier.NATIVE:
            request.tools = tools_to_native_format(tools)
        elif tier == ToolCallingTier.STRUCTURED:
            request.format = build_structured_output_schema(tools)  # type: ignore[assignment]
        else:
            tool_desc = self._build_text_tool_prompt(tools, small_model=model_tier == "small")
            if request.messages:
                last = request.messages[-1]
                request.messages[-1] = Message(
                    role=last.role,
                    content=f"{last.content}\n\n{tool_desc}",
                    images=last.images,
                )
        return tier

    @classmethod
    def _inject_call_dont_narrate_directives(
        cls, request: InternalChatRequest, tools: list[Tool],
    ) -> None:
        """Add a per-turn directive for each rendering tool that's on the menu.

        These tools share one failure mode: the model produces the CONTENT in
        prose instead of emitting the tool call, so nothing renders. The tool is
        on the roster and never invoked. Each directive is scoped to turns where
        its tool is actually available and is idempotent across the multi-call
        tool loop, so the system prompt doesn't grow on every iteration.

        Table-driven on purpose: this is one bug class with several instances
        (image, chart, …), so a new rendering tool is one entry here rather than
        another near-identical injector to keep in sync.
        """
        for tool_name, directive in _CALL_DONT_NARRATE_DIRECTIVES:
            if any(getattr(t, "name", "") == tool_name for t in tools):
                cls._append_system_directive(request, directive)

    @staticmethod
    def _append_system_directive(
        request: InternalChatRequest, directive: str,
    ) -> None:
        """Append ``directive`` to the first system message, or create one.

        No-op if the text is already present — the tool loop calls schema
        injection once per iteration.
        """
        for m in request.messages:
            if m.role == "system":
                if directive not in (m.content or ""):
                    m.content = f"{(m.content or '').rstrip()}\n\n{directive}".strip()
                return
        request.messages.insert(0, Message(role="system", content=directive))

    @staticmethod
    def _clear_tool_schemas(request: InternalChatRequest) -> None:
        """Remove tool schemas from a request so the LLM responds freely."""
        request.tools = None
        request.format = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Stale tool result condensation
    # ------------------------------------------------------------------

    @staticmethod
    def _condense_stale_tool_results(messages: list[Message]) -> None:
        """Replace verbose tool results from prior turns with brief summaries.

        When a frontend sends the full conversation history, old tool results
        (search results, image metadata, etc.) pollute the context and cause
        LLMs — especially smaller ones — to re-trigger the same tools.

        A tool result is "stale" if an assistant message follows it, meaning
        the LLM already synthesized those results into a response.  The most
        recent batch (not yet followed by an assistant response) is left
        untouched so the current tool loop works normally.

        Operates in-place on the message list.
        """
        # Find indices of all tool result messages
        tool_indices: list[int] = []
        for i, msg in enumerate(messages):
            if msg.role == "tool" or msg.role == "user" and msg.content and "## Tool Result" in msg.content:
                tool_indices.append(i)

        if not tool_indices:
            return

        # Find the last assistant message index — tool results before it are stale
        last_assistant_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx < 0:
            return  # No assistant response yet — nothing is stale

        _CONDENSED = "[Previous tool results omitted — already synthesized in the response below.]"

        for idx in tool_indices:
            if idx >= last_assistant_idx:
                continue  # Current turn — keep intact
            msg = messages[idx]
            content = msg.content or ""
            # Only condense if substantially long (short results aren't worth it)
            if len(content) > 200:
                messages[idx] = Message(
                    role=msg.role,
                    content=_CONDENSED,
                    tool_call_id=getattr(msg, "tool_call_id", None),
                )

    # ------------------------------------------------------------------
    # Tool call execution (appends messages to request)
    # ------------------------------------------------------------------

    async def _execute_and_append(
        self,
        request: InternalChatRequest,
        response: InternalChatResponse,
        calls: list[tuple[str, dict, str]],
        *,
        on_tool_start: object = None,
        on_tool_result: object = None,
        text_tier: bool = False,
        progress_queue: asyncio.Queue | None = None,
    ) -> set[str]:
        """Append assistant + tool result messages for a set of tool calls.

        Returns the set of tool names that executed **successfully**.

        Args:
            on_tool_result: Optional async callback ``(name, success, snippet)``
                called after each tool execution for UI transparency.
            text_tier: If True, use user-role messages for tool results instead
                of the native tool role (local models may not handle role="tool").
        """
        assistant_content = response.message.content if response.message else ""
        assistant_tool_calls = response.message.tool_calls if response.message else None
        # Preserve native reasoning_content across the synthesis round-trip.
        # DeepSeek's reasoner returns 400 ("reasoning_content in thinking
        # mode must be passed back") if a prior assistant turn with
        # tool_calls is replayed without it. The openai_compat backend
        # serializes Message.thinking → reasoning_content on the way out;
        # we just have to keep it on the in-flight messages list.
        assistant_thinking = response.message.thinking if response.message else None

        if text_tier:
            # TEXT tier: strip tool call artifacts and use plain messages
            assistant_content = self._strip_tool_call_text(assistant_content)
            assistant_content = self._strip_json_tool_calls(assistant_content)
            assistant_content = self._strip_fuzzy_tool_calls(assistant_content, calls)
            if assistant_content:
                request.messages.append(Message(
                    role="assistant",
                    content=assistant_content,
                    thinking=assistant_thinking,
                ))
        else:
            if not assistant_tool_calls and calls:
                # Tool calls were parsed from text — synthesize native format
                assistant_tool_calls = [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for name, args, tc_id in calls
                ]
                assistant_content = self._strip_tool_call_text(assistant_content)
                assistant_content = self._strip_json_tool_calls(assistant_content)
                assistant_content = self._strip_fuzzy_tool_calls(assistant_content, calls)

            request.messages.append(Message(
                role="assistant",
                content=assistant_content,
                tool_calls=assistant_tool_calls,
                thinking=assistant_thinking,
            ))

        # Execute tools and collect results + quality scores
        result_parts: list[str] = []
        succeeded: set[str] = set()
        _result_qualities: dict[str, str] = {}
        _saw_untrusted = False
        for tool_name, tool_args, tc_id in calls:
            tool = self._tool_registry.resolve(tool_name) if self._tool_registry else None
            # Fire tool_start BEFORE execution so the UI can render "searching
            # for {query}" / "fetching {url}" immediately. Legacy `on_tool_result`
            # still fires after with the full payload for backward compatibility.
            if on_tool_start:
                await on_tool_start(tc_id, tool_name, tool_args)  # type: ignore[misc]
            _exec_start = time.monotonic()
            if not tool:
                log.warning("passthrough_tool_unresolved", name=tool_name)
                result_text = f"Error: Unknown tool '{tool_name}'"
                tool_meta = {}
                _result_qualities[tool_name] = "empty"
                if on_tool_result:
                    dur_ms = int((time.monotonic() - _exec_start) * 1000)
                    await on_tool_result(tool_name, False, result_text[:200], {}, tc_id, dur_ms)  # type: ignore[misc]
            else:
                result_text, tool_meta = await self._execute_tool(tool, tool_args, progress_queue=progress_queue)
                success = not result_text.startswith("Error:")
                if not success:
                    log.warning("tool_execution_failed", tool=tool.name, error=result_text[:200])
                if success:
                    succeeded.add(tool.name)
                # Track result quality for synthesis guidance
                _result_qualities[tool.name] = _assess_result_quality(
                    tool.name, result_text, success,
                    failure_kind=(tool_meta or {}).get("failure_kind", ""),
                )
                # Track generated images so we can append them to the response
                if success and tool_meta.get("image_id"):
                    self._generated_images.append(tool_meta["url"])
                if on_tool_result:
                    snippet = result_text[:200] if success else result_text
                    dur_ms = int((time.monotonic() - _exec_start) * 1000)
                    await on_tool_result(tool.name, success, snippet, tool_meta, tc_id, dur_ms)  # type: ignore[misc]

            # Untrusted-content tools (web/browse/knowledge/files) wrap their
            # own output at source via wrap_untrusted, so the <<<UNTRUSTED:…>>>
            # markers already travel with result_text. Detect them so we can
            # guarantee the governing policy preamble is present (below).
            if MARKER_OPEN_PREFIX in result_text:
                _saw_untrusted = True
            if text_tier:
                result_parts.append(
                    f"## Tool Result ({tool_name})\n{result_text}"
                )
            else:
                request.messages.append(Message(
                    role="tool",
                    content=result_text,
                    tool_call_id=tc_id,
                ))

        # If any tool returned wrapped untrusted content, guarantee the safety
        # policy preamble is in the system prompt for the synthesis call. Tools
        # wrap at source, but the policy is otherwise only injected by the
        # memory / knowledge recall paths — a tool-only turn (web search, no
        # memory) would carry the markers without their governing policy.
        if _saw_untrusted:
            ensure_policy_in_system(request)

        # Build model-adaptive synthesis guidance
        model_tier = getattr(self, "_model_tier", "medium")
        if model_tier == "small":
            _synth_base = _SYNTH_EXPLICIT
        elif model_tier == "large":
            _synth_base = _SYNTH_COMPACT
        else:
            _synth_base = _SYNTH_STANDARD

        _synth_parts = [_synth_base]
        # Tool-specific guidance based on which tools ran
        _tool_guidance = _build_tool_synthesis_guidance(succeeded)
        if _tool_guidance:
            _synth_parts.append(_tool_guidance)
        # Quality-based addendum (empty results, partial failures)
        _quality_add = _quality_synthesis_addendum(_result_qualities)
        if _quality_add:
            _synth_parts.append(_quality_add)
        # Route-level hint (voice, chat, etc.)
        if self._tool_synthesis_hint:
            _synth_parts.append(self._tool_synthesis_hint)
        _synth_prompt = " ".join(_synth_parts)

        if text_tier and result_parts:
            # Inject all tool results as a single user message so the model
            # sees them as context and responds with a natural answer.
            request.messages.append(Message(
                role="user",
                content="\n\n".join(result_parts) + "\n\n" + _synth_prompt,
            ))
        else:
            # Native tier: tool results are role="tool" messages. Always
            # append synthesis direction so every model — from 8B locals
            # to frontier APIs — knows how to shape the response.
            request.messages.append(Message(
                role="user",
                content=_synth_prompt,
            ))

        return succeeded

    # ------------------------------------------------------------------
    # Tool loop (non-streaming — used by handle())
    # ------------------------------------------------------------------

    async def _run_tool_loop(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
    ) -> InternalChatResponse | None:
        """Run the tool call loop. Returns the final response (no pending tool calls)."""
        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier,
            extract_structured_text,
        )

        # No keyword pre-filtering in passthrough (2026-07-02): every
        # enabled tool is presented and the model decides. The regex
        # relevance filter made tools invisible on any paraphrase
        # outside its vocabulary — the wrong failure mode (the model
        # DENIES the capability). Curation happens at the enable layer
        # (config defaults / X-Augmentum-Tools), not per-turn.

        tier = self._inject_tool_schemas(request, tools)
        is_text = tier == ToolCallingTier.TEXT
        last_response: InternalChatResponse | None = None
        # Track non-cacheable (side-effect) tools already executed so the
        # LLM can't loop them.  Tools like image_generation and
        # python_exec have cacheable=False — once called, they shouldn't
        # be re-invoked in subsequent iterations of the same request.
        _executed_non_repeatable: set[str] = set()
        tool_map = {t.name: t for t in tools}

        # Per-turn search dedup + productivity guard (turn_search_dedup). The
        # dedup itself is installed at the turn boundary (_handle / _handle_stream)
        # so it's shared across every loop of the turn; here we just consult it
        # and stop early when a round of searching produced nothing new.
        from augmentum.tools.turn_search_dedup import get_turn_dedup, is_search_round
        _dedup = get_turn_dedup()
        _no_progress_rounds = 0

        _max_iters = _max_iterations(
            request, await _resolve_user_chain_limit(self),
        )
        for iteration in range(_max_iters):
            response = await self._backend.chat(request)
            last_response = response
            calls = self._parse_tool_calls(response, tools)

            # Gated capability proposed → surface a confirmation offer and end
            # the turn with the lead-in (never execute it inline).
            gated = self._first_gated(calls)
            if gated is not None:
                self._clear_tool_schemas(request)
                return await self._gated_response(
                    gated[0], gated[1], request.model or "",
                )

            if not calls:
                if tier == ToolCallingTier.STRUCTURED and response.message:
                    response.message.content = extract_structured_text(
                        response.message.content,
                    )
                return response

            # Filter genuine repeats (same tool + same args). The same tool
            # with different args (e.g. a refined web search) is allowed.
            filtered = []
            for c in calls:
                if _tool_repeat_key(c[0], c[1]) in _executed_non_repeatable:
                    log.info("passthrough_tool_repeat_blocked", tool=c[0],
                             iteration=iteration)
                    continue
                filtered.append(c)

            if not filtered:
                # All calls were identical repeats. Don't return the tool-call
                # stub (no prose) — drop the tools and let the model answer
                # from what it already gathered.
                self._clear_tool_schemas(request)
                return await self._backend.chat(request)

            log.info(
                "passthrough_tool_calls",
                iteration=iteration,
                tools=[c[0] for c in filtered],
            )
            if _dedup is not None:
                _dedup.begin_round()
            succeeded = await self._execute_and_append(
                request, response, filtered, text_tier=is_text,
            )

            # Productivity guard: if this round called a search tool but
            # surfaced zero NEW results (all duplicates of earlier rounds),
            # it's spinning. After two such rounds, stop and synthesize from
            # what's already gathered rather than burning the iteration cap.
            if _dedup is not None and is_search_round(filtered):
                if _dedup.round_new_count() == 0:
                    _no_progress_rounds += 1
                    if _no_progress_rounds >= 2:
                        log.info("passthrough_tool_no_progress_stop",
                                 iteration=iteration, new=0)
                        self._clear_tool_schemas(request)
                        return await self._backend.chat(request)
                else:
                    _no_progress_rounds = 0

            # Block only an IDENTICAL re-call of a non-cacheable tool that
            # succeeded. Failed calls (wrong args, timeout) stay retryable so
            # the model can self-correct.
            for c in filtered:
                tool_obj = tool_map.get(c[0])
                if tool_obj and not tool_obj.cacheable and c[0] in succeeded:
                    _executed_non_repeatable.add(_tool_repeat_key(c[0], c[1]))

        log.warning("passthrough_tool_max_iterations", max=_max_iters)
        # Same guard as the streaming path: the model exhausted the tool budget
        # without writing an answer, so ``last_response`` is a tool-call stub
        # with no prose — returning it shows the user a blank turn. Drop the
        # tools and make one final synthesis pass so the model answers from
        # what it gathered. Fall back to last_response only if that yields
        # nothing (never worse than before).
        self._clear_tool_schemas(request)
        final = await self._backend.chat(request)
        if final and final.message and (final.message.content or "").strip():
            return final
        return last_response

    # ------------------------------------------------------------------
    # Chain execution (multi-step tool chains)
    # ------------------------------------------------------------------

    async def _flow_list_message(self) -> str:
        """Build a user-friendly listing of available flows."""
        from augmentum.tools.custom_flows import CustomFlowStore

        if not self._custom_flow_store or not isinstance(self._custom_flow_store, CustomFlowStore):
            return "No flow store is configured."
        all_flows = await self._custom_flow_store.list_flows(enabled_only=True)
        if not all_flows:
            return "No flows are configured. Create flows in Settings → Flows."
        lines = ["Available flows:"]
        for f in all_flows:
            desc = f.get("description") or ""
            trigger = f.get("trigger_pattern") or ""
            line = f"  • {f['name']}"
            if desc:
                line += f" — {desc}"
            if trigger:
                line += f"  (auto-trigger: `{trigger}`)"
            lines.append(line)
        lines.append("\nUsage: `/flow <name>`")
        return "\n".join(lines)

    async def _try_chain_execution(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
    ) -> InternalChatResponse | None:
        """Try to execute the request as a multi-step chain.

        Resolution order:
        1. Custom flow (trigger pattern match)
        2. Adaptive chain (complexity detection + LLM planning)
        3. None (fall back to simple tool loop)
        """
        if not settings.passthrough_chain_enabled:
            return None

        # Gated proxies are propose-only — they must never be planned into a
        # chain (they'd execute the no-op proxy). The simple tool loop runs
        # afterwards with the full list and handles the gated intercept.
        tools = [t for t in tools if not getattr(t, "is_gated_proxy", False)]

        from augmentum.tools.chain import (
            ChainPlan,
            ToolChainPlanner,
            build_synthesis_prompt,
            detect_complexity,
            execute_chain,
        )
        from augmentum.tools.custom_flows import CustomFlowStore, flow_to_plan

        user_query = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content or ""
                    break

        if not user_query:
            return None

        plan: ChainPlan | None = None

        # 1. Check custom flows
        if self._custom_flow_store and isinstance(self._custom_flow_store, CustomFlowStore):
            stripped = user_query.strip().lower()
            # Bare /flow — list available flows
            if stripped == "/flow":
                msg = await self._flow_list_message()
                return InternalChatResponse(
                    message=Message(role="assistant", content=msg),
                    model=request.model,
                )
            # /flow <name> — find and run
            if stripped.startswith("/flow "):
                flow_name = user_query.strip()[6:].strip()
                flow = await self._custom_flow_store.fuzzy_find(flow_name)
                if flow:
                    plan = flow_to_plan(flow)
                    for step in plan.steps:
                        if step.input:
                            for k, v in step.input.items():
                                if isinstance(v, str) and "{{query}}" in v:
                                    step.input[k] = v.replace("{{query}}", flow_name)
                else:
                    all_flows = await self._custom_flow_store.list_flows(enabled_only=True)
                    names = [f["name"] for f in all_flows]
                    if names:
                        msg = f"No flow found matching \"{flow_name}\". Available flows: {', '.join(names)}"
                    else:
                        msg = "No flows are configured. Create flows in Settings → Flows."
                    return InternalChatResponse(
                        message=Message(role="assistant", content=msg),
                        model=request.model,
                    )
            else:
                flow = await self._custom_flow_store.match_query(user_query)
                if flow:
                    plan = flow_to_plan(flow)
                    for step in plan.steps:
                        if step.input:
                            for k, v in step.input.items():
                                if isinstance(v, str) and "{{query}}" in v:
                                    step.input[k] = v.replace("{{query}}", user_query)

        # 2. Adaptive chain planning
        if not plan and detect_complexity(user_query, tools):
            planner = ToolChainPlanner(self._backend, self._tool_registry)
            # Thread user_id + session_id through extra_tool_args and
            # cache_user_id so chain-routed IMAGE/ARTIFACT tools persist
            # to user-scoped tables (image_generations FK + ownership).
            # Without these slots, image_generation lands with empty
            # user_id and the row is never written.
            _extra = self._chain_extra_tool_args()
            result = await planner.plan_and_execute(
                request, tools,
                extra_tool_args=_extra,
                cache_user_id=self._user_id or "",
            )
            if result:
                results, plan = result
                # Skip execute_chain — already done by planner
                synth = build_synthesis_prompt(plan, results)
                synth_request = InternalChatRequest(
                    model=request.model,
                    messages=[
                        *request.messages,
                        Message(role="user", content=synth),
                    ],
                    stream=False,
                )
                try:
                    response = await asyncio.wait_for(
                        self._backend.chat(synth_request),
                        timeout=settings.passthrough_chain_synthesis_timeout,
                    )
                except TimeoutError:
                    log.warning("chain_synthesis_timeout", timeout=settings.passthrough_chain_synthesis_timeout)
                    fallback = _build_synthesis_fallback(results)
                    response = InternalChatResponse(
                        message=Message(role="assistant", content=fallback),
                        model=synth_request.model or "",
                    )
                # Track images from results
                for r in results.values():
                    if r.success and r.metadata.get("image_id"):
                        self._generated_images.append(r.metadata["url"])
                return response

        # 3. Execute custom flow plan (if found in step 1)
        if plan:
            log.info("chain_custom_flow", source=plan.source, steps=len(plan.steps))
            _extra = self._chain_extra_tool_args()
            _allowed = {t.name for t in tools} if tools else None
            results = await execute_chain(
                plan, self._backend, self._tool_registry,
                request_context=request,
                extra_tool_args=_extra,
                cache_user_id=self._user_id or "",
                allowed_tool_names=_allowed,
            )
            synth = build_synthesis_prompt(plan, results)
            synth_request = InternalChatRequest(
                model=request.model,
                messages=[
                    *request.messages,
                    Message(role="user", content=synth),
                ],
                stream=False,
            )
            try:
                response = await asyncio.wait_for(
                    self._backend.chat(synth_request),
                    timeout=settings.passthrough_chain_synthesis_timeout,
                )
            except TimeoutError:
                log.warning("chain_synthesis_timeout", timeout=settings.passthrough_chain_synthesis_timeout)
                fallback = _build_synthesis_fallback(results)
                response = InternalChatResponse(
                    model=synth_request.model or "",
                    choices=[{"message": {"role": "assistant", "content": fallback}}],
                )
            for r in results.values():
                if r.success and r.metadata.get("image_id"):
                    self._generated_images.append(r.metadata["url"])
            return response

        return None

    async def _try_chain_execution_stream(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
    ) -> AsyncIterator[InternalStreamChunk] | None:
        """Streaming variant of chain execution.

        Returns an async iterator of chunks, or None if no chain applies.
        """
        if not settings.passthrough_chain_enabled:
            return None

        # Gated proxies are propose-only — exclude from chains (the simple loop
        # below handles the gated intercept with the full tool list).
        tools = [t for t in tools if not getattr(t, "is_gated_proxy", False)]

        from augmentum.tools.chain import (
            ChainPlan,
            ChainStep,
            StepResult,
            ToolChainPlanner,
            build_synthesis_prompt,
            detect_complexity,
            execute_chain_streaming,
        )
        from augmentum.tools.custom_flows import CustomFlowStore, flow_to_plan

        user_query = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content or ""
                    break

        if not user_query:
            return None

        plan: ChainPlan | None = None

        # 1. Check custom flows
        if self._custom_flow_store and isinstance(self._custom_flow_store, CustomFlowStore):
            stripped = user_query.strip().lower()
            # Bare /flow — list available flows
            if stripped == "/flow":
                msg = await self._flow_list_message()

                async def _flow_list_stream():
                    yield InternalStreamChunk(content_delta=msg, done=True)

                return _flow_list_stream()
            # /flow <name> — find and run
            if stripped.startswith("/flow "):
                flow_name = user_query.strip()[6:].strip()
                flow = await self._custom_flow_store.fuzzy_find(flow_name)
                if flow:
                    plan = flow_to_plan(flow)
                    for step in plan.steps:
                        if step.input:
                            for k, v in step.input.items():
                                if isinstance(v, str) and "{{query}}" in v:
                                    step.input[k] = v.replace("{{query}}", flow_name)
                else:
                    all_flows = await self._custom_flow_store.list_flows(enabled_only=True)
                    names = [f["name"] for f in all_flows]
                    if names:
                        msg = f"No flow found matching \"{flow_name}\". Available flows: {', '.join(names)}"
                    else:
                        msg = "No flows are configured. Create flows in Settings → Flows."

                    async def _no_flow_stream():
                        yield InternalStreamChunk(content_delta=msg, done=True)

                    return _no_flow_stream()
            else:
                flow = await self._custom_flow_store.match_query(user_query)
                if flow:
                    plan = flow_to_plan(flow)
                    for step in plan.steps:
                        if step.input:
                            for k, v in step.input.items():
                                if isinstance(v, str) and "{{query}}" in v:
                                    step.input[k] = v.replace("{{query}}", user_query)

        # 2. Adaptive chain planning
        if not plan and detect_complexity(user_query, tools):
            async def _adaptive_stream():
                planner = ToolChainPlanner(self._backend, self._tool_registry)

                yield InternalStreamChunk(
                    content_delta="",
                    augmentum={"chain": {"status": "planning", "source": "adaptive"}},
                )

                step_queue: asyncio.Queue[InternalStreamChunk | None] = asyncio.Queue()

                async def _on_start(step: ChainStep):
                    await step_queue.put(InternalStreamChunk(
                        content_delta="",
                        augmentum={"chain_step": {
                            "id": step.id, "tool": step.tool,
                            "status": "running", "reason": step.reason,
                        }},
                    ))

                async def _on_done(result: StepResult):
                    await step_queue.put(InternalStreamChunk(
                        content_delta="",
                        augmentum={"chain_step": {
                            "id": result.step_id, "tool": result.tool_name,
                            "status": "done" if result.success else "error",
                            "preview": result.output[:200],
                        }},
                    ))
                    # Also surface the structured ToolResult.card so artifact
                    # tools (create_ebook etc.) render as a proper card in the
                    # chat bubble rather than a raw output echo.
                    _card = (result.metadata or {}).get("card") if result.success else None
                    if _card:
                        await step_queue.put(InternalStreamChunk(
                            content_delta="",
                            augmentum={"tool_call": {
                                "tool": result.tool_name,
                                "success": True,
                                "output_preview": result.output[:200],
                                "phase": "chain",
                                "card": _card,
                            }},
                        ))

                async def _run_planner():
                    try:
                        return await asyncio.wait_for(
                            planner.plan_and_execute(
                                request, tools,
                                on_step_start=_on_start,
                                on_step_done=_on_done,
                                extra_tool_args=self._chain_extra_tool_args(),
                                cache_user_id=self._user_id or "",
                            ),
                            timeout=settings.passthrough_chain_timeout,
                        )
                    finally:
                        await step_queue.put(None)

                plan_task = asyncio.create_task(_run_planner())

                # Yield step chunks as they arrive
                while True:
                    chunk = await step_queue.get()
                    if chunk is None:
                        break
                    yield chunk

                try:
                    result = await plan_task
                except TimeoutError:
                    log.warning("chain_timeout", timeout=settings.passthrough_chain_timeout)
                    yield InternalStreamChunk(
                        content_delta="",
                        augmentum={"chain": {"status": "fallback"}},
                    )
                    return

                if not result:
                    yield InternalStreamChunk(
                        content_delta="",
                        augmentum={"chain": {"status": "fallback"}},
                    )
                    return

                results, adaptive_plan = result

                yield InternalStreamChunk(
                    content_delta="",
                    augmentum={"chain": {"status": "synthesizing", "total_steps": len(adaptive_plan.steps)}},
                )

                for r in results.values():
                    if r.success and r.metadata.get("image_id"):
                        self._generated_images.append(r.metadata["url"])

                async for chunk in planner.synthesize(request, adaptive_plan, results):
                    yield chunk

            return _adaptive_stream()

        # 3. Execute custom flow
        if plan:
            async def _custom_flow_stream():
                yield InternalStreamChunk(
                    content_delta="",
                    augmentum={"chain": {
                        "status": "running",
                        "total_steps": len(plan.steps),
                        "source": plan.source,
                    }},
                )

                step_queue: asyncio.Queue[StepResult | None] = asyncio.Queue()

                _allowed = {t.name for t in tools} if tools else None
                exec_task = asyncio.create_task(
                    asyncio.wait_for(
                        execute_chain_streaming(
                            plan, self._backend, self._tool_registry,
                            step_queue,
                            request_context=request,
                            extra_tool_args=self._chain_extra_tool_args(),
                            cache_user_id=self._user_id or "",
                            allowed_tool_names=_allowed,
                        ),
                        timeout=settings.passthrough_chain_timeout,
                    )
                )

                # Consume step results as they complete
                results: dict[int, StepResult] = {}
                try:
                    while True:
                        sr = await step_queue.get()
                        if sr is None:
                            break
                        results[sr.step_id] = sr
                        yield InternalStreamChunk(
                            content_delta="",
                            augmentum={"chain_step": {
                                "id": sr.step_id, "tool": sr.tool_name,
                                "status": "done" if sr.success else "error",
                                "preview": sr.output[:200],
                            }},
                        )
                        # Emit a tool_call event too when the step produced a
                        # structured card (artifact tools like create_ebook).
                        # chain_step only renders the pipeline pill; tool_call
                        # is what triggers the chat-bubble artifact card.
                        _card = (sr.metadata or {}).get("card") if sr.success else None
                        if _card:
                            yield InternalStreamChunk(
                                content_delta="",
                                augmentum={"tool_call": {
                                    "tool": sr.tool_name,
                                    "success": True,
                                    "output_preview": sr.output[:200],
                                    "phase": "chain",
                                    "card": _card,
                                }},
                            )
                except asyncio.CancelledError:
                    exec_task.cancel()
                    return

                # Await the task to get final results and propagate errors
                try:
                    results = await exec_task
                except TimeoutError:
                    log.warning("chain_timeout", source=plan.source)
                    yield InternalStreamChunk(
                        content_delta="",
                        augmentum={"chain": {"status": "fallback"}},
                    )
                    return

                yield InternalStreamChunk(
                    content_delta="",
                    augmentum={"chain": {"status": "synthesizing"}},
                )

                for r in results.values():
                    if r.success and r.metadata.get("image_id"):
                        self._generated_images.append(r.metadata["url"])

                synth = build_synthesis_prompt(plan, results)
                synth_request = InternalChatRequest(
                    model=request.model,
                    messages=[
                        *request.messages,
                        Message(role="user", content=synth),
                    ],
                    stream=True,
                )
                async for chunk in _stream_with_timeout(
                    self._backend.chat_stream(synth_request),
                    timeout=settings.passthrough_chain_synthesis_timeout,
                ):
                    yield chunk

            return _custom_flow_stream()

        return None

    # ------------------------------------------------------------------
    # Tool resolution for streaming (resolves tool calls, then caller streams)
    # ------------------------------------------------------------------

    async def _resolve_tool_calls(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
        *,
        on_tool_status: object = None,
        on_tool_start: object = None,
        on_tool_result: object = None,
        on_narration: object = None,
        progress_queue: asyncio.Queue | None = None,
    ) -> bool:
        """Resolve all tool calls, mutating request.messages with results.

        Returns True if any tools were executed (caller should stream final
        answer). Returns False if the model never called tools.

        Stores request.model on self so _execute_tool can pass it to tools.

        When tools were used, the request is left ready for a final streaming
        call — tool schemas are cleared so the LLM responds freely.

        Args:
            on_narration: Optional async callback ``(text)`` called when the
                LLM produces text content alongside tool calls (e.g.
                "Let me search for that…").  Useful for voice pipelines
                that want to TTS the narration before tools execute.
        """
        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier,
            extract_structured_text,
        )

        self._current_model = request.model or ""
        self._model_tier = _estimate_model_tier(request.model or "", self._backend)

        # No keyword pre-filtering in passthrough (2026-07-02): every
        # enabled tool is presented and the model decides — see the
        # matching note in _handle_non_streaming. Curation happens at
        # the enable layer, not per-turn.

        tier = self._inject_tool_schemas(request, tools)
        is_text = tier == ToolCallingTier.TEXT
        _executed_non_repeatable: set[str] = set()
        tool_map = {t.name: t for t in tools}

        # Per-turn dedup (installed at _handle_stream) + productivity guard.
        from augmentum.tools.turn_search_dedup import get_turn_dedup, is_search_round
        _dedup = get_turn_dedup()
        _no_progress_rounds = 0

        _max_iters = _max_iterations(
            request, await _resolve_user_chain_limit(self),
        )
        for iteration in range(_max_iters):
            response = await self._backend.chat(request)
            calls = self._parse_tool_calls(response, tools)

            # Gated capability proposed → surface a confirmation offer; the
            # lead-in becomes the turn's answer (yielded by the caller as a
            # single chunk). Never executed inline.
            gated = self._first_gated(calls)
            if gated is not None:
                self._clear_tool_schemas(request)
                self._first_response = await self._gated_response(
                    gated[0], gated[1], request.model or "",
                )
                return False

            if not calls:
                if iteration == 0:
                    # Model never used tools — return response via _first_response
                    if tier == ToolCallingTier.STRUCTURED and response.message:
                        response.message.content = extract_structured_text(
                            response.message.content,
                        )
                    # For TEXT tier, strip any tool call artifacts the model
                    # produced without actually completing a valid call
                    if is_text and response.message:
                        content = response.message.content or ""
                        content = self._strip_tool_call_text(content)
                        response.message.content = content
                    self._first_response = response
                    return False

                # Tools were used in prior iterations. This non-streaming call
                # produced the final answer, but we discard it so the caller
                # can stream the answer instead (better UX for long responses).
                self._clear_tool_schemas(request)
                return True

            # Filter genuine repeats (same tool + same args). A refined
            # call to the same tool (e.g. a new search query) passes through.
            filtered = []
            for c in calls:
                if _tool_repeat_key(c[0], c[1]) in _executed_non_repeatable:
                    log.info("passthrough_tool_repeat_blocked", tool=c[0],
                             iteration=iteration)
                    continue
                filtered.append(c)

            if not filtered:
                # Identical repeats only. Returning True hands off to the
                # caller's final streaming pass, which synthesises an answer
                # from the gathered results (no tools attached).
                self._clear_tool_schemas(request)
                return True

            # Forward narration text the LLM produced alongside tool calls
            # (e.g. "Let me look that up…" before calling web_search).
            if on_narration and response.message:
                narration = (response.message.content or "").strip()
                # Strip tool call artifacts from the narration
                narration = self._strip_tool_call_text(narration)
                narration = self._strip_json_tool_calls(narration)
                if narration:
                    await on_narration(narration)  # type: ignore[misc]

            if on_tool_status:
                tool_names = [c[0] for c in filtered]
                await on_tool_status(tool_names)  # type: ignore[misc]

            log.info(
                "passthrough_tool_calls",
                iteration=iteration,
                tools=[c[0] for c in filtered],
            )
            if _dedup is not None:
                _dedup.begin_round()
            succeeded = await self._execute_and_append(
                request, response, filtered,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
                text_tier=is_text,
                progress_queue=progress_queue,
            )

            # Productivity guard: two rounds of searching that surface nothing
            # new → stop and let the caller stream a synthesis from what was
            # gathered, instead of spinning to the iteration cap.
            if _dedup is not None and is_search_round(filtered):
                if _dedup.round_new_count() == 0:
                    _no_progress_rounds += 1
                    if _no_progress_rounds >= 2:
                        log.info("passthrough_stream_no_progress_stop",
                                 iteration=iteration, new=0)
                        self._clear_tool_schemas(request)
                        return True
                else:
                    _no_progress_rounds = 0

            # Block only an IDENTICAL re-call (same tool + same args) of a
            # non-cacheable tool that succeeded.
            for c in filtered:
                tool_obj = tool_map.get(c[0])
                if tool_obj and not tool_obj.cacheable and c[0] in succeeded:
                    _executed_non_repeatable.add(_tool_repeat_key(c[0], c[1]))

        # Max iterations — clear schemas and let caller stream
        self._clear_tool_schemas(request)
        return True

    @staticmethod
    def _merge_native_tool_call_delta(
        accumulator: dict[int, dict], delta: dict,
    ) -> None:
        """Merge one streaming ``tool_call`` SSE delta into the accumulator.

        Native streaming format (OpenAI/DeepSeek/Anthropic-compat): each
        delta is ``{index, id?, type?, function: {name?, arguments?}}``.
        The first delta for a given index carries ``id``/``type``/``name``;
        later deltas carry ``arguments`` fragments that concatenate into
        the final JSON string.
        """
        idx = int(delta.get("index", 0) or 0)
        entry = accumulator.get(idx)
        if entry is None:
            entry = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
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

    async def _resolve_tool_calls_streaming(
        self,
        request: InternalChatRequest,
        tools: list[Tool],
        *,
        output_queue: asyncio.Queue,
        on_tool_status: object = None,
        on_tool_start: object = None,
        on_tool_result: object = None,
    ) -> None:
        """Stream-first variant of :meth:`_resolve_tool_calls` for the NATIVE tool tier.

        Pushes content chunks to ``output_queue`` AS the model emits them
        instead of waiting for the full non-streaming response. Native
        tool calling guarantees ``content_delta`` and ``tool_calls`` don't
        interleave within a single response, so we can forward content
        chunks immediately while accumulating tool_call deltas separately.

        When a response ends with no tool_calls: the prose was streamed
        in full; we exit (model answered directly).

        When a response ends with tool_calls: we build a synthetic
        ``InternalChatResponse`` from the accumulated deltas, run the
        tools (events still flow through callbacks → output_queue), and
        loop to the next iteration — which will stream the composed
        answer naturally.

        Unlike :meth:`_resolve_tool_calls` this returns no boolean; the
        caller doesn't need to make a follow-up streaming call because
        EVERY model emission lands directly in the queue.

        Used only when ``select_tier`` returns NATIVE. STRUCTURED and
        TEXT tier requests stay on the original peek-then-stream path
        in :meth:`_handle_stream` because their tool-call parsing
        depends on having the full response text up-front.
        """
        from augmentum.modes.analytical.tool_calling import ToolCallingTier

        self._current_model = request.model or ""
        self._model_tier = _estimate_model_tier(request.model or "", self._backend)

        # No keyword pre-filtering in passthrough (2026-07-02): every
        # enabled tool is presented and the model decides — see the
        # matching note in _handle_non_streaming.

        tier = self._inject_tool_schemas(request, tools)
        # Defensive: the caller should only invoke us for NATIVE tier,
        # but if the runtime tier resolves to something else (e.g.
        # uarf_tool_tier_override flipped it), fall back to clearing
        # schemas and letting the caller stream raw. Avoids running
        # the streaming loop with a tier whose responses can't be
        # parsed mid-stream.
        if tier != ToolCallingTier.NATIVE:
            log.warning(
                "passthrough_streaming_tier_mismatch",
                expected="native", actual=tier.value, model=request.model,
            )
            self._clear_tool_schemas(request)
            async for chunk in self._backend.chat_stream(request):
                await output_queue.put(chunk)
            return

        tool_map = {t.name: t for t in tools}
        _executed_non_repeatable: set[str] = set()

        # Per-turn dedup (installed at _handle_stream) + productivity guard.
        from augmentum.tools.turn_search_dedup import get_turn_dedup, is_search_round
        _dedup = get_turn_dedup()
        _no_progress_rounds = 0

        _max_iters = _max_iterations(
            request, await _resolve_user_chain_limit(self),
        )
        for iteration in range(_max_iters):
            accumulated: dict[int, dict] = {}
            final_finish_reason: str | None = None
            final_model: str = request.model or ""
            # Capture any assistant content/thinking the model emits
            # alongside tool_calls so the synthesised response carries
            # them. DeepSeek's reasoner sometimes prefixes a tool_call
            # with reasoning_content — we want to thread it back into
            # the assistant message so the next turn passes
            # reasoning_content per DeepSeek's API requirement.
            assistant_content_parts: list[str] = []
            assistant_thinking_parts: list[str] = []

            async for chunk in self._backend.chat_stream(request):
                # Pluck tool_call deltas out of the augmentum metadata —
                # don't forward those chunks. The model is dictating a
                # tool call, not visible content. We'll execute it after
                # the stream completes.
                tc_deltas = (chunk.augmentum or {}).get("tool_calls")
                if tc_deltas:
                    for tc in tc_deltas:
                        self._merge_native_tool_call_delta(accumulated, tc)
                    # Also capture any content/thinking that arrived in
                    # the same chunk before the tool_call decision —
                    # this is rare but the model is free to do it. The
                    # content was already in the chunk; just track it.
                    if chunk.content_delta:
                        assistant_content_parts.append(chunk.content_delta)
                    if chunk.thinking_delta:
                        assistant_thinking_parts.append(chunk.thinking_delta)
                    if chunk.finish_reason:
                        final_finish_reason = chunk.finish_reason
                    continue

                # Regular content / thinking / status chunk — forward
                # immediately. This is the streaming win: the user sees
                # tokens as they generate instead of waiting for the
                # whole response.
                if chunk.content_delta:
                    assistant_content_parts.append(chunk.content_delta)
                if chunk.thinking_delta:
                    assistant_thinking_parts.append(chunk.thinking_delta)
                if chunk.finish_reason:
                    final_finish_reason = chunk.finish_reason
                if chunk.model:
                    final_model = chunk.model
                await output_queue.put(chunk)

            if not accumulated:
                # No tool calls — model answered directly. Prose was
                # streamed in full above. Clear schemas (in case the
                # caller reuses the request) and exit.
                self._clear_tool_schemas(request)
                return

            # Build a synthetic non-streaming response from the
            # accumulated tool_call deltas so the existing
            # _parse_tool_calls + _execute_and_append plumbing works
            # unchanged.
            tool_calls_list = [
                accumulated[idx] for idx in sorted(accumulated.keys())
            ]
            synth_response = InternalChatResponse(
                message=Message(
                    role="assistant",
                    content="".join(assistant_content_parts),
                    tool_calls=tool_calls_list,
                    thinking="".join(assistant_thinking_parts) or None,
                ),
                model=final_model,
                finish_reason=final_finish_reason or "tool_calls",
            )

            calls = self._parse_tool_calls(synth_response, tools)

            # Gated capability proposed → surface a confirmation offer and stream
            # the lead-in instead of executing. The user confirms via the chip
            # (Accept / Not now / Never) before the heavy tool ever runs.
            gated = self._first_gated(calls)
            if gated is not None:
                lead = await self._surface_gated_offer(
                    gated[0], gated[1], model=request.model or "",
                )
                await output_queue.put(InternalStreamChunk(content_delta=lead))
                await output_queue.put(InternalStreamChunk(
                    content_delta="", done=True, finish_reason="stop",
                ))
                self._clear_tool_schemas(request)
                return

            if not calls:
                # tool_call deltas didn't parse to a known tool (e.g.
                # the model hallucinated a name not in the schema and
                # the native filter dropped it). Clear schemas and let
                # the next iteration's stream just produce prose.
                self._clear_tool_schemas(request)
                return

            # Filter genuine repeats (same tool + same args) executed in
            # earlier iterations. Different args for the same tool (e.g. a
            # second web search with a refined query) are allowed through.
            filtered = []
            for c in calls:
                if _tool_repeat_key(c[0], c[1]) in _executed_non_repeatable:
                    log.info(
                        "passthrough_tool_repeat_blocked_streaming",
                        tool=c[0], iteration=iteration,
                    )
                    continue
                filtered.append(c)

            if not filtered:
                # Every requested call was an identical repeat. The model is
                # mid-turn and hasn't written its answer yet — ending here
                # leaves the user with reasoning and no result (the live
                # "searches twice then stops" bug). Drop the tools and let
                # one more pass synthesise an answer from what's gathered.
                self._clear_tool_schemas(request)
                async for chunk in self._backend.chat_stream(request):
                    await output_queue.put(chunk)
                return

            # Skip narration emit deliberately: in the non-streaming
            # path the model's "Let me search…" content surfaces as a
            # separate tool_narration event because the full response
            # came back at once. Here it already streamed inline above,
            # so re-emitting would duplicate.
            if on_tool_status:
                tool_names = [c[0] for c in filtered]
                await on_tool_status(tool_names)  # type: ignore[misc]

            log.info(
                "passthrough_tool_calls_streaming",
                iteration=iteration,
                tools=[c[0] for c in filtered],
            )
            if _dedup is not None:
                _dedup.begin_round()
            succeeded = await self._execute_and_append(
                request, synth_response, filtered,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
                text_tier=False,  # native tier only
                progress_queue=output_queue,
            )

            # Productivity guard: two rounds of searching with nothing new →
            # drop tools and stream one synthesis pass over what's gathered.
            if _dedup is not None and is_search_round(filtered):
                if _dedup.round_new_count() == 0:
                    _no_progress_rounds += 1
                    if _no_progress_rounds >= 2:
                        log.info("passthrough_stream_consume_no_progress_stop",
                                 iteration=iteration, new=0)
                        self._clear_tool_schemas(request)
                        async for chunk in self._backend.chat_stream(request):
                            await output_queue.put(chunk)
                        return
                else:
                    _no_progress_rounds = 0

            # Block only an IDENTICAL re-call (same tool + same args) of a
            # non-cacheable tool that succeeded — a true infinite-loop guard.
            for c in filtered:
                tool_obj = tool_map.get(c[0])
                if tool_obj and not tool_obj.cacheable and c[0] in succeeded:
                    _executed_non_repeatable.add(_tool_repeat_key(c[0], c[1]))

        # Max iterations hit. The model kept requesting tools and never wrote
        # an answer. If we just return here and EVERY iteration was tool-only
        # (the model looped tool calls without prose — common for small custom
        # SFT models), NOTHING streamed this entire turn: the user sees a blank,
        # hung response (gen_tokens=0). The other loop-exit guards
        # (no-progress, all-repeats) each drop the tools and stream one final
        # synthesis pass for exactly this reason; the max-iterations exit was
        # missing it. Mirror them: clear schemas and force one tool-free pass so
        # the model MUST answer in prose from what it already gathered. A
        # guaranteed terminator ensures the turn can never end without a
        # visible answer AND a finish_reason (never a silent red-button hang).
        log.warning(
            "passthrough_stream_tool_max_iterations",
            max=_max_iters, model=request.model,
        )
        self._clear_tool_schemas(request)
        streamed_any = False
        async for chunk in self._backend.chat_stream(request):
            if chunk.content_delta or chunk.thinking_delta:
                streamed_any = True
            await output_queue.put(chunk)
        if not streamed_any:
            # Even the synthesis pass produced no visible content — never leave
            # the turn without an answer + terminator, or the UI hangs forever.
            await output_queue.put(InternalStreamChunk(
                content_delta=(
                    "I couldn't quite pull that together just now. "
                    "Could you rephrase, or try again in a moment?"
                ),
                done=True,
                finish_reason="stop",
            ))

    # ------------------------------------------------------------------
    # Non-streaming handler
    # ------------------------------------------------------------------

    async def _handle(self, request: InternalChatRequest) -> InternalChatResponse:
        has_v, v_instruction, cleaned = extract_v_command(
            request,
            fallback_text=request.messages[-1].content if request.messages else "",
        )
        image_task = None
        if has_v and self._image_enabled and self._image_queue:
            image_task = asyncio.create_task(
                generate_direct_image(v_instruction, self._image_queue, self._session_id, user_id=self._user_id)
            )
            request = cleaned

        self._current_request = request
        self._generated_images.clear()

        # Install one per-turn search dedup shared by every tool loop below
        # (turn_search_dedup): web/image/youtube results returned in one round are
        # remembered so later rounds surface only NEW items. Scoped to this
        # request's asyncio task, so no cross-turn leak and no reset needed.
        from augmentum.tools.turn_search_dedup import TurnSearchDedup, set_turn_dedup
        set_turn_dedup(TurnSearchDedup())

        # Direct-invoke tools run BEFORE SSOS so the tool button is
        # authoritative over SSOS's heuristic intent. See _handle_stream
        # for the matching rationale. Non-streaming path drops long-
        # running tools entirely — their fire-and-forget UX depends on
        # streaming kickoff chunks and a progress-polling client.
        _direct_tools = [
            t for t in self._collect_direct_invoke_tools()
            if not getattr(t, "long_running", False)
        ]
        if _direct_tools:
            _user_msg = ""
            for _m in reversed(request.messages):
                if _m.role == "user":
                    _user_msg = (_m.content or "").strip()
                    break
            if _user_msg:
                log.info(
                    "passthrough.direct_invoke",
                    tools=[t.name for t in _direct_tools],
                    model=request.model,
                    streaming=False,
                )
                tool_outputs: list[tuple[str, str]] = []
                for _tool in _direct_tools:
                    output, _meta, _success, _dur = await self._invoke_direct_tool(
                        _tool, _user_msg,
                    )
                    tool_outputs.append((_tool.name, output))
                self._inject_direct_invoke_synthesis(request, tool_outputs)
                self._note_direct_invoke_capture(_direct_tools)
                response = await self._backend.chat(request)
                if image_task:
                    image_url = await image_task
                    if image_url and response.message:
                        response.message.content += f"\n\n![Generated Image]({image_url})"
                return response

        # SSOS fast path — heuristic intent detection, no LLM tool calling.
        # Self-gates on the per-user `ui.autoTools` preference; returns None
        # when disabled, when intent confidence is low, or when no handler
        # branch matches.
        if self._ssos:
            self._current_model = request.model or ""
            synthesized = await self._ssos.try_orchestrate(request)
            if synthesized:
                response = await self._backend.chat(synthesized)
                if image_task:
                    image_url = await image_task
                    if image_url and response.message:
                        response.message.content += f"\n\n![Generated Image]({image_url})"
                return response

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
            # Knowledge retrieval is best-effort enrichment — passthrough
            # proceeds without injected context.
            log.debug("passthrough_knowledge_retrieve_failed", exc_info=True)

        self._stash_turn_user_text(request)
        tools = self._resolve_tools()

        if not tools and self._ssos and await self._ssos.is_enabled():
            # Auto mode: expose the lookup capabilities as NATIVE tools (mirror
            # of the streaming path) instead of the [[tool:NAME]] soft-trigger
            # text protocol, which native-tool-trained models reliably missed
            # (emitting native tool_calls / reasoning in the thinking channel).
            tools = self._resolve_auto_capability_tools()
            # Stable capability self-model so the model never denies a
            # capability the per-turn schema filter dropped this turn.
            self._inject_auto_capability_self_model(request, tools)

        if not tools:
            # No tools (Auto off, or none resolved) → plain chat.
            response = await self._backend.chat(request)
        else:
            # Inject artifact template context when artifact tools are available
            try:
                artifact_names = [t.name for t in tools if t.name in ('create_document', 'create_presentation', 'create_spreadsheet', 'create_chart')]
                if artifact_names and request.messages:
                    from augmentum.tools.artifact_templates import get_template_for_tool_call
                    user_text = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
                    for tn in artifact_names:
                        tpl_ctx = get_template_for_tool_call(tn, user_text)
                        if tpl_ctx:
                            sys_msg = next((m for m in request.messages if m.role == "system"), None)
                            if sys_msg:
                                sys_msg.content += f"\n\n## Design Template\n{tpl_ctx}"
                            else:
                                # No system message — create one with template context
                                from augmentum.models.base import Message
                                request.messages.insert(0, Message(
                                    role="system",
                                    content=f"## Design Template\n{tpl_ctx}",
                                ))
                            break
            except Exception:
                # Artifact template lookup is enrichment — fall through
                # without inserting design-template context.
                log.debug("passthrough_artifact_template_inject_failed", exc_info=True)
            self._condense_stale_tool_results(request.messages)
            # Try chain execution first (custom flow → adaptive → simple loop)
            response = await self._try_chain_execution(request, tools)
            if response is None:
                response = await self._run_tool_loop(request, tools)
            if response is None:
                # Shouldn't happen but be defensive
                response = await self._backend.chat(request)

        if image_task:
            image_url = await image_task
            if image_url and response.message:
                response.message.content += f"\n\n![Generated Image]({image_url})"

        # Append any images generated by tools (e.g. image_generation) so
        # they appear in the response regardless of frontend or streaming mode
        if self._generated_images and response.message:
            for img_url in self._generated_images:
                response.message.content += f"\n\n![Generated Image]({img_url})"

        return response

    # ------------------------------------------------------------------
    # Streaming handler
    # ------------------------------------------------------------------

    # Friendly lead-in verbs for gated capabilities — the brief line shown
    # alongside the confirmation chip. Falls back to the humanized tool name.
    _GATED_LEADIN: dict[str, str] = {
        "image_generation": "create that image",
        "build_application": "build that app",
        "create_ebook": "write that ebook",
        "create_presentation": "put that deck together",
        "create_document": "write that up",
    }

    async def _surface_gated_offer(self, cap, args: str, *, model: str = "") -> str:
        """Publish a confirmation offer for a gated capability instead of
        running it, and return a short lead-in line to show with the chip.

        The model PROPOSED a heavy tool; we turn that into an Accept / Not now /
        Never chip so the user has an exit when the intent was inferred wrong.
        Structured creators (``needs_plan``) get a planner pass first: the brief
        is expanded into the tool's full input so the chip shows the OUTLINE.
        Best-effort: a failed/suppressed offer just means no chip.
        """
        verb = self._GATED_LEADIN.get(cap.name, cap.name.replace("_", " "))
        extra = None
        reason = None
        planned = False
        # A self-contained spec rendered into the CHAT MESSAGE itself (not only
        # the offer chip), so the proposal is never invisible: even if the chip
        # is missed or suppressed, the user sees exactly what would be built and
        # can approve in words. Built deterministically — no extra LLM call that
        # could stall the proposal.
        spec_lead = ""
        if getattr(cap, "needs_plan", False):
            try:
                from augmentum.modes.passthrough.gated_planner import (
                    expand_brief,
                    outline_summary,
                )
                structured = await expand_brief(cap.tool, args, self._backend, model)
                if structured:
                    extra = {"structured": structured, "primary_arg": cap.primary_arg}
                    reason = outline_summary(cap.tool, structured) or args[:200]
                    planned = True
            except Exception:
                log.warning("gated_plan_failed", tool=getattr(cap, "tool", "?"), exc_info=True)
        elif cap.name == "build_application" and args.strip():
            from augmentum.tools.artifact_application import derive_project_name
            name = derive_project_name(args) or "Web App"
            what = args.strip().rstrip(".")
            reason = f"{name}: {what[:200]}"
            spec_lead = (
                f"Here's what I'd build — **{name}**:\n\n"
                f"- {what}\n"
                f"- Delivered as a multi-file web app (HTML/CSS/JS) with a live "
                f"preview and a downloadable zip.\n\n"
            )
        try:
            surfaced = await self._ssos.propose_gated(
                cap, args, extra=extra, reason=reason,
                thread_id=self._session_id or "",
                session_id=self._session_id or "", mode="passthrough",
            )
        except Exception:
            log.warning("gated_offer_failed", tool=getattr(cap, "tool", "?"), exc_info=True)
            surfaced = False
        if surfaced:
            if planned:
                return f"Here's a draft to {verb} — confirm the outline below and I'll build it."
            if spec_lead:
                return spec_lead + "Want me to go ahead? Confirm below, or tell me what to change."
            return f"I can {verb} — confirm below and I'll get started."
        if spec_lead:
            return spec_lead + "Just say the word and I'll build it."
        return f"I could {verb} if you'd like — just say the word."

    async def _soft_trigger_nonstream(
        self, request: InternalChatRequest,
    ):
        """Non-streaming model-initiated capability pass (Auto mode).

        Decide call (original + soft-trigger hint). If the first visible line is
        a ``[[tool:NAME]] args`` marker: a LOOKUP runs + synthesizes; a GATED
        capability surfaces a confirmation offer instead of firing. Otherwise
        the decide response IS the answer, returned verbatim.
        """
        decide_req = self._ssos.build_decide_request(request)
        resp = await self._backend.chat(decide_req)
        text = (resp.message.content if resp.message else "") or ""
        first_line = text.lstrip().split("\n", 1)[0]
        matched = self._ssos.match_trigger(first_line)
        if matched is None:
            return resp
        cap, args = matched
        if cap.kind == "gated":
            lead = await self._surface_gated_offer(cap, args, model=request.model or "")
            if resp.message is not None:
                resp.message.content = lead
            return resp
        result_text, _meta = await self._ssos.run_named_tool(cap, args)
        if result_text is None:
            synth = self._ssos.build_tool_failure_request(request, cap)
        else:
            synth = self._ssos.build_tool_synthesis_request(request, cap, result_text)
        return await self._backend.chat(synth)

    async def _soft_trigger_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Streaming model-initiated capability pass (Auto mode).

        Streams a decide call with the soft-trigger hint, buffering the first
        VISIBLE line (post-thinking). If it's a ``[[tool:NAME]] args`` marker for
        a registered lookup capability, suppress it, run the tool (live events),
        then stream a synthesis pass. Otherwise flush the buffer and stream the
        reply verbatim — a single call, no extra cost for the common case.
        """
        from augmentum.modes.passthrough.orchestrator import _EventEmitter
        from augmentum.tools.events import make_tool_complete, make_tool_start
        from augmentum.utils.thinking import ThinkingStreamBuffer

        _PROBE_LIMIT = 240
        decide_req = self._ssos.build_decide_request(request)
        # local_engine: the asymmetric "starts inside think" assumption (GLM /
        # DeepSeek-V4 / Qwen3 …) is valid only for a local llama-server. This
        # probe streams VISIBLE content to the user, so a cloud active model
        # would otherwise have its answer routed into thinking and emptied (#17).
        _fam = getattr(self._backend, "reasoning_family", None)
        tbuf = ThinkingStreamBuffer(
            family=_fam(request.model) if callable(_fam) else None,
            model=request.model or "",
            local_engine=self._backend.is_local_engine(),
        )
        probe = ""
        decided = False
        matched = None

        async for chunk in self._backend.chat_stream(decide_req):
            visible, thinking = "", ""
            if chunk.content_delta or chunk.thinking_delta:
                visible, thinking = tbuf.process(
                    chunk.content_delta or "", chunk.thinking_delta or "",
                )
            if thinking:
                yield InternalStreamChunk(thinking_delta=thinking, model=chunk.model)

            if decided:  # no-marker path resolved earlier — stream verbatim
                if visible:
                    yield InternalStreamChunk(content_delta=visible, model=chunk.model)
                if chunk.done or chunk.finish_reason or chunk.usage:
                    yield InternalStreamChunk(
                        done=chunk.done, finish_reason=chunk.finish_reason,
                        usage=chunk.usage, model=chunk.model,
                    )
                continue

            if visible:
                probe += visible
            stripped = probe.lstrip()
            if "\n" in stripped or len(stripped) >= _PROBE_LIMIT:
                matched = self._ssos.match_trigger(stripped.split("\n", 1)[0])
                decided = True
                if matched is None:
                    if probe:
                        yield InternalStreamChunk(content_delta=probe, model=chunk.model)
                    probe = ""
                else:
                    break  # stop the decide stream — run the tool instead

        if matched is None:
            # Stream ended (or never produced a newline). Flush the parser tail
            # and decide on whatever we buffered.
            tail_c, tail_t = tbuf.flush()
            if tail_t:
                yield InternalStreamChunk(thinking_delta=tail_t)
            if not decided:
                probe += tail_c or ""
                matched = self._ssos.match_trigger(probe.lstrip().split("\n", 1)[0]) if probe.strip() else None
                if matched is None:
                    if probe:
                        yield InternalStreamChunk(content_delta=probe)
                    return
            elif tail_c:
                yield InternalStreamChunk(content_delta=tail_c)
                return
            else:
                return

        # Marker matched.
        cap, args = matched

        # GATED capability → don't fire; surface a confirmation offer (chip with
        # Accept / Not now / Never) and end the turn with a short lead-in. The
        # marker line itself was buffered, never shown.
        if cap.kind == "gated":
            lead = await self._surface_gated_offer(cap, args, model=request.model or "")
            yield InternalStreamChunk(content_delta=lead, model=request.model)
            yield InternalStreamChunk(done=True, finish_reason="stop", model=request.model)
            return

        # LOOKUP → run the tool with live events, then synthesize.
        _q: asyncio.Queue[InternalStreamChunk | None] = asyncio.Queue()

        async def _on_start(tc_id: str, name: str, a: dict) -> None:
            await _q.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_start": make_tool_start(tc_id, name, a, phase="auto")},
            ))

        async def _on_complete(
            name: str, success: bool, snippet: str, meta: dict,
            tc_id: str, dur_ms: int,
        ) -> None:
            result_metadata = {
                k: v for k, v in (meta or {}).items() if not k.startswith("_")
            }
            await _q.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_call": {
                    "tool": name, "success": success,
                    "output_preview": snippet, "phase": "auto",
                    "result_metadata": result_metadata,
                }},
            ))
            await _q.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_complete": make_tool_complete(
                    tc_id, name, success=success, output_preview=snippet,
                    error="" if success else snippet, duration_ms=dur_ms,
                    extra={"result_metadata": result_metadata} if result_metadata else None,
                )},
            ))

        emit = _EventEmitter(on_tool_start=_on_start, on_tool_complete=_on_complete)

        async def _run():
            try:
                return await self._ssos.run_named_tool(cap, args, emit=emit)
            finally:
                await _q.put(None)

        run_task = asyncio.create_task(_run())
        while True:
            ev = await _q.get()
            if ev is None:
                break
            yield ev
        result_text, _meta = await run_task

        if result_text is None:
            synth = self._ssos.build_tool_failure_request(request, cap)
        else:
            synth = self._ssos.build_tool_synthesis_request(request, cap, result_text)
        async for chunk in self._backend.chat_stream(synth):
            yield chunk

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        has_v, v_instruction, cleaned = extract_v_command(
            request,
            fallback_text=request.messages[-1].content if request.messages else "",
        )
        image_task = None
        if has_v and self._image_enabled and self._image_queue:
            image_task = asyncio.create_task(
                generate_direct_image(v_instruction, self._image_queue, self._session_id, user_id=self._user_id)
            )
            request = cleaned

        self._current_request = request
        self._generated_images.clear()

        # Install one per-turn search dedup shared by every tool loop below
        # (turn_search_dedup): web/image/youtube results returned in one round are
        # remembered so later rounds surface only NEW items. Scoped to this
        # request's asyncio task, so no cross-turn leak and no reset needed.
        from augmentum.tools.turn_search_dedup import TurnSearchDedup, set_turn_dedup
        set_turn_dedup(TurnSearchDedup())

        # Direct-invoke tools run BEFORE SSOS so that an explicit button
        # selection (e.g. YouTube) beats SSOS's heuristic intent that
        # would otherwise route a "videos about cats" query to
        # web_search. The button is the authoritative intent signal.
        _direct_tools = self._collect_direct_invoke_tools()
        if _direct_tools:
            _user_msg = ""
            for _m in reversed(request.messages):
                if _m.role == "user":
                    _user_msg = (_m.content or "").strip()
                    break
            if _user_msg:
                # Long-running direct tools (app_builder) take over the
                # response — they spawn a background task and yield a
                # kickoff chunk. Other direct tools are skipped for
                # this turn since the user clearly wanted the big one.
                _long_running = next(
                    (t for t in _direct_tools
                     if getattr(t, "long_running", False)),
                    None,
                )
                if _long_running:
                    log.info(
                        "passthrough.direct_invoke.long_running",
                        tool=_long_running.name,
                        model=request.model,
                    )
                    async for chunk in self._run_long_running_direct_tool_stream(
                        _long_running, _user_msg, request.model or "",
                    ):
                        yield chunk
                    return

                log.info(
                    "passthrough.direct_invoke",
                    tools=[t.name for t in _direct_tools],
                    model=request.model,
                )
                tool_outputs: list[tuple[str, str]] = []
                for _tool in _direct_tools:
                    output, meta, success, dur_ms = await self._invoke_direct_tool(
                        _tool, _user_msg,
                    )
                    for chunk in self._build_direct_invoke_events(
                        _tool.name, _user_msg, output, meta, success, dur_ms,
                    ):
                        yield chunk
                    tool_outputs.append((_tool.name, output))
                # Splice synthesis context + drop tool schemas, then
                # stream the model's brief caption over the results.
                self._inject_direct_invoke_synthesis(request, tool_outputs)
                self._note_direct_invoke_capture(_direct_tools)
                async for chunk in self._backend.chat_stream(request):
                    yield chunk
                if image_task:
                    image_url = await image_task
                    if image_url:
                        yield InternalStreamChunk(
                            content_delta=f"\n\n![Generated Image]({image_url})",
                        )
                return

        # SSOS fast path — heuristic intent detection, no LLM tool calling.
        # Self-gates on the per-user `ui.autoTools` preference; returns None
        # when disabled, when intent confidence is low, or when no handler
        # branch matches.
        #
        # Run SSOS as a background task and pump its progress events
        # through a queue so the user sees `tool_start` /
        # `tool_complete` LIVE during the 5–15s search wait, not after.
        if self._ssos:
            self._current_model = request.model or ""
            from augmentum.tools.events import (
                make_tool_complete,
                make_tool_start,
            )

            _ssos_q: asyncio.Queue[InternalStreamChunk | None] = asyncio.Queue()

            async def _ssos_status(stage: str) -> None:
                await _ssos_q.put(InternalStreamChunk(
                    content_delta="", augmentum={"status": stage},
                ))

            async def _ssos_tool_start(tc_id: str, name: str, args: dict) -> None:
                await _ssos_q.put(InternalStreamChunk(
                    content_delta="",
                    augmentum={"tool_start": make_tool_start(
                        tc_id, name, args, phase="ssos",
                    )},
                ))

            async def _ssos_tool_complete(
                name: str, success: bool, snippet: str, meta: dict,
                tc_id: str, dur_ms: int,
            ) -> None:
                # Strip leading-underscore keys (private plumbing; same
                # convention as the LLM tool path).
                result_metadata = {
                    k: v for k, v in (meta or {}).items()
                    if not k.startswith("_")
                }
                await _ssos_q.put(InternalStreamChunk(
                    content_delta="",
                    augmentum={"tool_call": {
                        "tool": name,
                        "success": success,
                        "output_preview": snippet,
                        "phase": "ssos",
                        "result_metadata": result_metadata,
                    }},
                ))
                await _ssos_q.put(InternalStreamChunk(
                    content_delta="",
                    augmentum={"tool_complete": make_tool_complete(
                        tc_id, name, success=success,
                        output_preview=snippet,
                        error="" if success else snippet,
                        duration_ms=dur_ms,
                        extra={"result_metadata": result_metadata}
                              if result_metadata else None,
                    )},
                ))

            async def _run_ssos():
                try:
                    return await self._ssos.try_orchestrate(
                        request,
                        on_status=_ssos_status,
                        on_tool_start=_ssos_tool_start,
                        on_tool_complete=_ssos_tool_complete,
                    )
                finally:
                    await _ssos_q.put(None)

            ssos_task = asyncio.create_task(_run_ssos())

            # Drain progress events live until SSOS finishes.
            while True:
                ev = await _ssos_q.get()
                if ev is None:
                    break
                yield ev

            synthesized = await ssos_task
            if synthesized:
                async for chunk in self._backend.chat_stream(synthesized):
                    yield chunk
                if image_task:
                    image_url = await image_task
                    if image_url:
                        yield InternalStreamChunk(
                            content_delta=f"\n\n![Generated Image]({image_url})",
                        )
                return

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
            # Knowledge retrieval is best-effort enrichment — streaming
            # proceeds without injected context.
            log.debug("passthrough_stream_knowledge_retrieve_failed", exc_info=True)

        tools = self._resolve_tools()
        log.info("passthrough_stream_tools", enabled=self._enabled_tools, resolved=len(tools))

        if not tools:
            # Auto mode (no explicit tools selected). When Auto-tools is on,
            # expose the model-driven lookup capabilities (web_search,
            # wikipedia, youtube, image_search) as NATIVE tool schemas and let
            # the native tool-calling path below handle them. This replaces the
            # old [[tool:NAME]] "soft-trigger" text protocol, which fought the
            # way modern models are trained: they emit native tool_calls (or
            # reason in the thinking channel), so the text marker was missed and
            # the call vanished into the thinking dropdown with no follow-up.
            # The deterministic SSOS heuristics already ran above.
            if self._ssos and await self._ssos.is_enabled():
                self._stash_turn_user_text(request)
                tools = self._resolve_auto_capability_tools()
                # Stable capability self-model so the model never denies a
                # capability the per-turn schema filter dropped this turn.
                self._inject_auto_capability_self_model(request, tools)
                if tools:
                    log.info(
                        "passthrough_auto_native_tools",
                        tools=[t.name for t in tools], model=request.model,
                    )
        if not tools:
            # Auto-tools off, or no capabilities resolved → pure passthrough
            # streaming (no tool schemas).
            yield InternalStreamChunk(
                content_delta="", augmentum={"status": "thinking"},
            )
            _stream_chunk_count = 0
            _stream_content_bytes = 0
            _stream_thinking_bytes = 0
            async for chunk in self._backend.chat_stream(request):
                _stream_chunk_count += 1
                if chunk.content_delta:
                    _stream_content_bytes += len(chunk.content_delta)
                if chunk.thinking_delta:
                    _stream_thinking_bytes += len(chunk.thinking_delta)
                yield chunk
            log.info(
                "passthrough_stream_no_tools_done",
                chunks=_stream_chunk_count,
                content_bytes=_stream_content_bytes,
                thinking_bytes=_stream_thinking_bytes,
                continue_last_assistant=request.continue_last_assistant,
                model=request.model,
            )
            if image_task:
                image_url = await image_task
                if image_url:
                    yield InternalStreamChunk(
                        content_delta=f"\n\n![Generated Image]({image_url})",
                    )
            return

        # Condense verbose tool results from prior turns so the LLM doesn't
        # get confused by stale search results / image prompts in context.
        self._condense_stale_tool_results(request.messages)

        # Try chain execution first (custom flow → adaptive → simple loop)
        chain_stream = await self._try_chain_execution_stream(request, tools)
        if chain_stream is not None:
            _fallback = False
            async for chunk in chain_stream:
                if chunk.augmentum and chunk.augmentum.get("chain", {}).get("status") == "fallback":
                    _fallback = True
                    break
                yield chunk
            if not _fallback:
                # Chain handled it — yield tool-generated images and return
                for img_url in self._generated_images:
                    yield InternalStreamChunk(
                        content_delta=f"\n\n![Generated Image]({img_url})",
                    )
                if image_task:
                    image_url = await image_task
                    if image_url:
                        yield InternalStreamChunk(
                            content_delta=f"\n\n![Generated Image]({image_url})",
                        )
                return

        # Tool-enabled streaming strategy. Two paths depending on the
        # tool-calling tier select_tier resolves for the active backend:
        #
        # NATIVE (cloud APIs + most modern local models that emit native
        # ``tool_calls`` SSE deltas — DeepSeek, OpenAI, Anthropic,
        # llama-server's Jinja-template path, Ollama's recent native
        # tool-call API): use ``_resolve_tool_calls_streaming`` to
        # forward content chunks AS the model emits them. Native models
        # guarantee content_delta and tool_calls don't interleave, so
        # we can stream prose immediately and accumulate tool_call
        # deltas separately. When tool calls land, execute them and
        # let the next iteration stream the composed answer. This is
        # what the user actually sees — no more "wait for the whole
        # response to materialise, then dump as one chunk" with cloud
        # providers.
        #
        # STRUCTURED / TEXT (local model families without reliable
        # native tool-calling, where tool calls arrive embedded in
        # prose or as a constrained JSON schema): we still need the
        # full response text to parse — the older peek-then-stream
        # path applies. Step 1 makes a non-streaming call, parses
        # tool calls, executes them if found, then streams the final
        # answer. If no tools were used, the cached full response is
        # yielded as a single chunk via ``self._first_response``.

        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier as _ToolCallingTier,
        )
        from augmentum.modes.analytical.tool_calling import (
            select_tier as _select_tier,
        )
        _tier = _select_tier(self._backend, request.model or "")
        # Diagnostic (2026-06-14): NATIVE → stream-first; STRUCTURED/TEXT →
        # non-streaming peek. select_tier returns non-NATIVE when the model
        # string is empty or the backend class is unrecognized — either drops
        # tool turns off the streaming path. Surfaces the exact inputs so a
        # live test pins which one is empty/wrong.
        log.info(
            "passthrough_tier_decided",
            tier=_tier.value,
            model=request.model or "(empty)",
            backend=type(self._backend).__name__,
            n_tools=len(tools),
        )

        _tool_queue: asyncio.Queue[InternalStreamChunk | None] = asyncio.Queue()

        def _status(stage: str) -> InternalStreamChunk:
            return InternalStreamChunk(
                content_delta="", augmentum={"status": stage},
            )

        async def _on_narration(text: str) -> None:
            await _tool_queue.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_narration": text},
            ))

        async def _on_tool_status(tool_names: list[str]) -> None:
            await _tool_queue.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_status": "running", "tool_names": tool_names},
            ))

        async def _on_tool_start(tc_id: str, tool_name: str, args: dict) -> None:
            """Emit unified tool_start event so UI can render live progress."""
            from augmentum.tools.events import make_tool_start
            await _tool_queue.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_start": make_tool_start(tc_id, tool_name, args, phase="passthrough")},
            ))

        async def _on_tool_result(
            name: str, success: bool, snippet: str, meta: dict,
            tc_id: str = "", dur_ms: int = 0,
        ) -> None:
            from augmentum.tools.events import make_tool_complete
            # Forward the tool's full metadata dict to the UI under
            # `result_metadata`. The frontend deliverable dispatcher reads
            # this and renders tool-specific UI (YouTube panel, image
            # gallery, project card, etc.) without the model having to
            # echo the payload in prose. Any tool that wants a rich UI
            # just needs to return the relevant keys in ToolResult.metadata
            # — no handler changes required.
            result_metadata = {k: v for k, v in (meta or {}).items() if not k.startswith("_")}

            tool_data: dict = {
                "tool": name,
                "success": success,
                "output_preview": snippet,
                "phase": "passthrough",
                "result_metadata": result_metadata,
            }
            # Structured ToolCard envelope (artifact tools) — kept at top
            # level because _renderInlineToolCard reads it directly.
            if meta.get("card"):
                tool_data["card"] = meta["card"]
            await _tool_queue.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_call": tool_data},
            ))
            # Unified tool_complete event — same metadata, carried under
            # `result_metadata` for the deliverable dispatcher.
            extra = {"result_metadata": result_metadata} if result_metadata else None
            await _tool_queue.put(InternalStreamChunk(
                content_delta="",
                augmentum={"tool_complete": make_tool_complete(
                    tc_id or "", name,
                    success=success,
                    output_preview=snippet,
                    error="" if success else snippet,
                    duration_ms=dur_ms,
                    extra=extra,
                )},
            ))

        self._first_response = None

        yield _status("thinking")

        if _tier == _ToolCallingTier.NATIVE:
            # Native tier: stream-first path. The resolver pushes
            # content chunks AND tool events through the same queue.
            # When the resolver returns the entire user-visible answer
            # has already streamed — no follow-up chat_stream needed.
            async def _run_streaming_tools():
                try:
                    await self._resolve_tool_calls_streaming(
                        request, tools,
                        output_queue=_tool_queue,
                        on_tool_status=_on_tool_status,
                        on_tool_start=_on_tool_start,
                        on_tool_result=_on_tool_result,
                    )
                finally:
                    await _tool_queue.put(None)

            stream_task = asyncio.create_task(_run_streaming_tools())
            while True:
                chunk = await _tool_queue.get()
                if chunk is None:
                    break
                yield chunk
            await stream_task  # surface any exception that escaped
        else:
            # Structured / text tier: peek-then-stream. The resolver
            # makes a non-streaming call (we need the full response
            # text to parse tool calls embedded in prose or constrained
            # JSON), then either yields the cached response or streams
            # the composed answer after tools run.
            async def _run_tools():
                try:
                    return await self._resolve_tool_calls(
                        request, tools,
                        on_tool_status=_on_tool_status,
                        on_tool_start=_on_tool_start,
                        on_tool_result=_on_tool_result,
                        on_narration=_on_narration,
                        progress_queue=_tool_queue,
                    )
                finally:
                    await _tool_queue.put(None)

            tool_task = asyncio.create_task(_run_tools())
            while True:
                chunk = await _tool_queue.get()
                if chunk is None:
                    break
                yield chunk
            tools_used = await tool_task

            if tools_used:
                # Tools were executed — request.messages now has tool
                # results. Stream the composed final answer.
                yield _status("composing")
                first_content = True
                async for chunk in self._backend.chat_stream(request):
                    if first_content and (chunk.content_delta or chunk.thinking_delta):
                        first_content = False
                    yield chunk
            elif self._first_response and self._first_response.message:
                # No tools used — model answered directly via the
                # non-streaming call. Yield the cached response as a
                # single chunk. Acceptable for structured/text tier
                # because the models in this tier are local (low
                # latency) — the chunk-vs-stream perception delta is
                # negligible. Cloud providers go through the native
                # branch above where this dump never happens.
                content = self._first_response.message.content
                yield InternalStreamChunk(
                    content_delta=content,
                    model=self._first_response.model,
                )
                yield InternalStreamChunk(
                    content_delta="",
                    done=True,
                    finish_reason=self._first_response.finish_reason or "stop",
                    usage=self._first_response.usage,
                    model=self._first_response.model,
                )

        # Append tool-generated images as markdown so all frontends see them
        # (Augmentum UI also renders via metadata channel, but this ensures
        # Open WebUI and other external clients display the image too)
        for img_url in self._generated_images:
            yield InternalStreamChunk(
                content_delta=f"\n\n![Generated Image]({img_url})",
            )

        if image_task:
            image_url = await image_task
            if image_url:
                yield InternalStreamChunk(
                    content_delta=f"\n\n![Generated Image]({image_url})",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_json_tool_calls(text: str) -> str:
        """Strip JSON tool call blocks from assistant content.

        Removes patterns like ``[{"name": "web", ...}]`` so the LLM
        doesn't see its own raw tool call JSON on the next iteration.
        """
        stripped = text.strip()
        # Try to remove a JSON array or object that looks like a tool call
        stripped = re.sub(
            r'```(?:json)?\s*\[.*?\]\s*```',
            '', stripped, flags=re.DOTALL,
        )
        stripped = re.sub(
            r'\[[\s\n]*\{[\s\n]*"name".*?\]',
            '', stripped, flags=re.DOTALL,
        )
        stripped = re.sub(
            r'\{[\s\n]*"name".*?\}',
            '', stripped, count=1, flags=re.DOTALL,
        )
        return stripped.strip()

    @staticmethod
    def _strip_tool_call_text(text: str) -> str:
        """Strip tool call artifacts from assistant content.

        Removes all known tool call formats so the model doesn't see
        its own raw tool invocation on the next iteration:
        - TOOL_CALL: / TOOL_INPUT: (Tier 3 text format)
        - Action: / Action Input: (ReAct format)
        - <tool_call>...</tool_call>, <tool_use>...</tool_use> (XML format)
        """
        # Remove TOOL_CALL: line and everything after it
        stripped = re.sub(
            r'(?i)\n*\s*\*{0,2}TOOL[_ -]?CALL\*{0,2}\s*[:=].*',
            '', text, flags=re.DOTALL,
        )
        # Remove ReAct Action: / Action Input: blocks
        stripped = re.sub(
            r'(?i)\n*\s*\*{0,2}(?:Action|Tool|Function)\*{0,2}\s*[:=]\s*\S+.*',
            '', stripped, flags=re.DOTALL,
        )
        # Remove XML tool call blocks
        stripped = re.sub(
            r'<(?:tool_call|tool_use|function_call)[^>]*>.*?</(?:tool_call|tool_use|function_call)>',
            '', stripped, flags=re.DOTALL | re.IGNORECASE,
        )
        return stripped.strip()

    @staticmethod
    def _strip_fuzzy_tool_calls(text: str, calls: list[tuple[str, dict, str]]) -> str:
        """Strip fuzzy/python-style tool call artifacts from assistant content.

        For parsers that don't have a clean delimiter (fuzzy, python-style),
        remove the tool name + JSON object from the text so the model doesn't
        see its own raw tool invocation on the next iteration.
        """
        for tool_name, _args, _tc_id in calls:
            # Remove tool_name(...) patterns (python-style)
            text = re.sub(
                rf'\b{re.escape(tool_name)}\s*\([^)]*\)',
                '', text, count=1, flags=re.DOTALL,
            )
            # Remove tool_name + nearby JSON object
            text = re.sub(
                rf'\b{re.escape(tool_name)}\b[^{{]{{0,50}}\{{[^}}]*\}}',
                '', text, count=1, flags=re.DOTALL,
            )
        return text.strip()

    @staticmethod
    def _build_text_tool_prompt(tools: list[Tool], *, small_model: bool = False) -> str:
        """Build a Tier 3 text prompt listing available tools."""
        lines = [
            "You have access to the following tools. To use one, respond ONLY with:",
            "TOOL_CALL: tool_name",
            "TOOL_INPUT: {\"param\": \"value\"}",
            "",
            "IMPORTANT: Only use a tool when it clearly helps answer the user's question. "
            "If you can answer directly from your knowledge, do so without calling a tool.",
            "",
            "Available tools:",
        ]
        for t in tools:
            schema = t.input_schema or {}
            params = schema.get("properties", {})
            required = set(schema.get("required", []))
            parts = []
            for k, v in params.items():
                ptype = v.get("type", "string")
                marker = " (required)" if k in required else ""
                parts.append(f"{k}: {ptype}{marker}")
            param_str = ", ".join(parts)
            desc = t.description
            if small_model and t.model_hint:
                desc = f"{desc} {t.model_hint}"
            lines.append(f"- {t.name}({param_str}): {desc}")
        return "\n".join(lines)
