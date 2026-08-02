"""Plan phase for Coder mode.

Extracted from ``handler.py`` so plan-specific logic (grammar, parsers,
preamble stripping, plan-to-task seeding) lives in one focused place.
The mixin preserves ``self.*`` access to the full ``CoderHandler``
instance — state, backend, container manager, workspace snapshot,
workspace guide cache — without changing any behaviour.

The separate file also means plan-phase fixes (like the 2026-04-21
preamble strip that kept "The user is asking me..." out of chat) land
here rather than in a 6000-line god file.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import structlog

from augmentum.coder.prompts import PLAN_META, PLAN_SYSTEM
from augmentum.coder.state import CoderPhase
from augmentum.coder.tools import READ_ONLY_TOOLS
from augmentum.models.base import InternalChatRequest, InternalStreamChunk
from augmentum.modes.coder.chat_egress import (
    ReasoningRelay,
    StreamProgressTracker,
    emit_relay,
)

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from augmentum.modes.coder.turn_context import TurnContext


# Plan-phase output MUST contain one of these markers (PLAN_SYSTEM
# enforces the grammar). Anything the model emits before the first match
# is planner reasoning prose and must not leak into chat.
#
# Case-sensitive + word-boundary to match the grammar the prompt teaches
# ("Plan:" / "Question:" as section headers). Case-insensitive would
# false-positive on prose like "the plan: is to..." or "my question:".
#
# No ^/\n lookbehind — observed 2026-04-21 that weak models concatenate
# their pre-marker monologue with the marker on a single logical line
# ("The user is asking... Plan: inspect workspace..."). Requiring a
# newline made the filter silently miss those and the preamble leaked
# to chat. Any occurrence in the stream is treated as the cut point.
_PLAN_MARKER_RE = re.compile(r"\b(Plan|Question)\s*:")


# "Question:" detection is slightly more permissive than _PLAN_MARKER_RE:
# it allows up to 200 chars of "Let me think..." preamble before the
# marker, because the act-phase short-circuit needs a best-effort
# answer even when the model didn't follow grammar exactly. Used by
# _plan_is_question to decide whether to skip the act phase entirely.
_PLAN_QUESTION_RE = re.compile(
    r"^.{0,200}?Question\s*:",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _plan_is_question(plan_text: str) -> bool:
    """True iff the plan-phase output looks like a clarification question
    rather than an executable plan.

    Checks for the "Question:" marker the PLAN_SYSTEM VAGUE branch
    emits. Handles small amounts of leading preamble (weaker models
    often write a sentence of throat-clearing before the structured
    output).
    """
    if not plan_text:
        return False
    return bool(_PLAN_QUESTION_RE.match(plan_text.strip()))


def _parse_plan_steps(plan_text: str) -> list[str]:
    """Extract numbered step descriptions from a plan string.

    Accepts the plan shapes weaker models actually emit today:

    - ``1. Do this thing``
    - ``Step 1: Do this thing``
    - fallback plain imperative body lines under ``Plan:``

    Returns a list of step descriptions. When the model omitted numbering but
    still emitted one imperative action per line, those body lines are treated
    as recoverable steps instead of collapsing to ``steps=0``.
    """
    steps: list[str] = []
    body_lines: list[str] = []
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = re.match(r"^\d+\.\s+(.+)$", stripped)
        if m:
            steps.append(m.group(1).strip())
            continue

        m = re.match(r"^step\s+\d+\s*[:.\-]\s+(.+)$", stripped, re.IGNORECASE)
        if m:
            steps.append(m.group(1).strip())
            continue

        if (
            stripped.lower().startswith("plan:")
            or stripped.startswith("#")
            or stripped.startswith("```")
        ):
            continue

        if stripped.startswith(("- ", "* ")):
            body_lines.append(stripped[2:].strip())
            continue

        body_lines.append(stripped)

    if steps:
        return steps

    if len(body_lines) >= 2:
        return body_lines

    return steps


# ---------------------------------------------------------------------------
# Late-bind: module-level helpers from handler.py that the plan phase
# references (``_tool_to_schema``, ``_strip_cot_tokens``,
# ``_strip_tool_json``). Same pattern as ``_legacy._bind_handler_helpers``
# — the circular-import can't be resolved at module-top, so handler.py
# injects these refs at its module bottom.
# ---------------------------------------------------------------------------

_HELPERS_BOUND = False


class _LiveProxy:
    """Call-time proxy for names that tests monkeypatch on handler."""
    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        from augmentum.modes.coder import handler as _h
        return getattr(_h, self._attr)(*args, **kwargs)


_LIVE_NAMES = ("create_coder_tools",)


def _bind_handler_helpers() -> None:
    """Late-bind handler.py's module-level helpers into this module."""
    global _HELPERS_BOUND
    if _HELPERS_BOUND:
        return
    from augmentum.modes.coder import handler as _h

    forwarded = (
        "_tool_to_schema",
        "_strip_cot_tokens",
        "_strip_tool_json",
    )
    g = globals()
    for name in forwarded:
        if hasattr(_h, name):
            g[name] = getattr(_h, name)
    for name in _LIVE_NAMES:
        g[name] = _LiveProxy(name)
    _HELPERS_BOUND = True


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class PlanPhaseMixin:
    """Plan phase for the CoderHandler.

    Isolated in its own module so plan-specific fixes don't require
    navigating thousands of lines of unrelated strategy code.
    """

    async def _plan_phase(
        self,
        request: InternalChatRequest,
        turn_context: TurnContext,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Generate the step-by-step plan and parse it into state."""
        self._state.phase = CoderPhase.PLANNING

        # Signal planning has started
        yield self._meta_chunk(
            phase="planning",
            status="started",
            model=request.model,
        )

        # Build plan context: authoritative workspace snapshot (file
        # tree) + repo_map's definitions + dependencies sections. The
        # two complement each other — snapshot is state-complete,
        # repo_map's ranked definitions are query-relevant. Neither
        # duplicates the other's content (we pass skip_file_listing=True
        # below so repo_map drops its own listing).
        plan_context = turn_context.to_plan_context()
        # Ensure workspace guide is cached (used by _build_messages)
        await self._get_workspace_guide()

        # Build the plan request: guide (auto-prepended) + PLAN_SYSTEM + repo map
        plan_system = PLAN_SYSTEM
        if self._workspace_has_populated_repo_for_turn():
            plan_system += (
                "\n\n## Existing Repository Orientation\n"
                "This workspace already contains a populated existing repository. "
                "Do NOT ask what to build from scratch, do NOT claim the repo is "
                "empty, and do NOT spend turns rediscovering /workspace root. Start "
                "from the existing workspace_tree, then inspect one or two high-signal "
                "files or subdirectories (for example README, package manifest, src/, "
                "tests/, docs/) to identify a small real improvement that fits the "
                "current codebase."
            )
            if getattr(self, "_workspace_git_url_for_turn", ""):
                plan_system += (
                    "\nA git clone URL is attached for this workspace, so treat it as "
                    "a real cloned repository even if a tool result is sparse or "
                    "surprising. Re-anchor on the existing codebase instead of asking "
                    "for another repo URL."
                )
        if plan_context:
            plan_system += f"\n\n{plan_context}"
        plan_messages = self._build_messages(request, plan_system)
        yield self._token_budget_chunk(
            plan_messages,
            scope="plan_prompt",
            phase="planning",
            model=request.model,
        )
        compacted, before, after = self._maybe_compact_messages(plan_messages)
        if compacted:
            yield self._meta_chunk(
                phase="planning",
                status="compaction",
                model=request.model,
                extra={
                    "scope": "plan_prompt",
                    "tokens_before": before,
                    "tokens_after": after,
                },
            )
            yield self._token_budget_chunk(
                plan_messages,
                scope="plan_prompt",
                phase="planning",
                model=request.model,
                compacted=True,
            )

        # Build the plan-phase tool schema.
        #
        # Two filters compose: the long-standing ``READ_ONLY_TOOLS`` allow-list
        # (also used by phase_act for parallelism classification) gates which
        # tools are *capable* of read-only work; ``PLAN_META.disallowed_tools``
        # is the authoritative deny-list declared with the prompt. Today the
        # two lists are complementary in practice, so this is a zero-behavior
        # change activation — but the prompt metadata is now consulted on
        # every plan-phase schema build, so a future deny (or a new tool that
        # ships in disallowed_tools) takes effect without a code change here.
        all_tools = (
            create_coder_tools(
                self._container_manager,
                self._workspace_id,
                self._state,
                executor=getattr(self, "_executor", None),
                profile_store=getattr(self, "_profile_store", None),
                service_store=getattr(self, "_service_store", None),
                user_id=getattr(self, "_user_id", ""),
                subagent_dispatcher=self._get_subagent_dispatcher(),
                jobs_store=getattr(self, "_jobs_store", None),
                db_conn=self._resolve_archive_conn(),
            )
            if self._container_manager is not None
            else []
        )
        # Defense-in-depth: a tool listed in both surfaces means the
        # classification has drifted (allow-list says read-only, deny-list
        # says forbid). The deny still wins below; log so we clean it up.
        overlap = set(PLAN_META.disallowed_tools) & READ_ONLY_TOOLS
        if overlap:
            log.warning(
                "coder_plan_tool_classification_overlap",
                tools=sorted(overlap),
                prompt=PLAN_META.name,
                prompt_version=PLAN_META.version,
            )
        ro_tool_schemas = [
            _tool_to_schema(t)
            for t in all_tools
            if t.name in READ_ONLY_TOOLS
            and t.name not in PLAN_META.disallowed_tools
        ]
        log.info(
            "coder_plan_schema_built",
            tool_count=len(ro_tool_schemas),
            disallowed_count=len(PLAN_META.disallowed_tools),
            prompt=PLAN_META.name,
            prompt_version=PLAN_META.version,
        )

        # Resolve the plan-phase model via the role declared on PLAN_META
        # (currently "utility" — the smarter background-task tier in the
        # provider registry). When no registry is wired (older construction
        # paths, tests) or the role doesn't resolve, fall back to the bound
        # backend so single-model setups are unchanged. This is the Cline
        # cost-tier discipline: planning gets the thoughtful model, acting
        # gets the fast one. On single-GPU setups both resolve to the same
        # model — degrades gracefully.
        plan_backend = self._backend
        plan_model = request.model
        plan_role = PLAN_META.model_role
        registry = getattr(self, "_provider_registry", None)
        if plan_role and registry is not None:
            try:
                from augmentum.config import settings as _plan_settings
                resolved = await registry.resolve_model_for_role(
                    plan_role, settings=_plan_settings,
                )
                if resolved and resolved[0] is not None:
                    plan_backend, plan_model = resolved
                    log.info(
                        "coder_plan_role_resolved",
                        role=plan_role,
                        model=plan_model,
                        prompt=PLAN_META.name,
                        prompt_version=PLAN_META.version,
                    )
            except Exception as exc:
                log.warning(
                    "coder_plan_role_resolve_failed",
                    role=plan_role,
                    error=str(exc),
                )

        # Same thinking policy as the act loops: default OFF, the coder
        # composer's per-turn toggle wins. The toggle exists FOR planning
        # turns ("toggle on before the plan turn, off before the execute
        # turn"), so the plan model must see it. ``think`` mirrors the
        # resolved bool so cloud backends (openai_compat folds
        # ``request.think`` into provider-specific toggles) agree with
        # local engines (which consume the explicit kwarg directly).
        from augmentum.modes.coder.phase_act import _iteration_thinking_kwargs
        plan_thinking_kwargs = _iteration_thinking_kwargs(request)
        plan_request = InternalChatRequest(
            model=plan_model,
            messages=plan_messages,
            stream=True,
            temperature=request.temperature,
            tools=ro_tool_schemas or None,
            # Explicit field-adds (not dataclass_replace) so the plan call
            # keeps its historical no-kv-affinity behavior — adopting the
            # act request's kv_session_key here would evict the act loop's
            # warm slot prefix for a differently-framed one-shot.
            think=plan_thinking_kwargs["enable_thinking"],
            chat_template_kwargs=plan_thinking_kwargs,
        )

        # Stream plan generation while collecting full text for parsing.
        # Clean CoT delimiter tokens (<|mask_start|>, <think>, etc.) per delta
        # so they don't leak into the chat view.
        #
        # Weaker models often narrate their reasoning in plain prose
        # ("The user is asking me...", "Let me plan my response...") BEFORE
        # emitting the "Plan:" / "Question:" marker that PLAN_SYSTEM
        # requires. _strip_cot_tokens only removes wrapper tokens, not
        # prose. So: buffer every delta until we see the marker, then
        # stream from the marker onward. If a stream ends without the
        # marker, the plan is malformed — log and stay silent rather than
        # dumping the preamble into chat.
        plan_parts: list[str] = []
        preamble_buffer = ""
        thinking_preamble_buffer = ""
        marker_found = False
        thinking_marker_found = False
        # Live reasoning relay for the pre-marker thinking stream. Before
        # 2026-07-02 the preamble was buffered for marker detection and
        # then silently discarded — with the thinking toggle ON for a plan
        # turn, the user watched dead air while the model reasoned. Now
        # the pre-marker text streams to the collapsible reasoning block
        # (coalesced), while the post-marker release into the plan bubble
        # is unchanged. ``_relayed_upto`` tracks how much of the buffer
        # has been relayed; ``_REASONING_HOLDBACK`` keeps the last few
        # chars unrelayed so a "Plan:" marker split across chunk
        # boundaries never half-leaks into the reasoning block.
        reasoning_relay = ReasoningRelay(phase="planning", model=plan_model)
        thinking_relayed_upto = 0
        _REASONING_HOLDBACK = 16  # > len("Question :") — marker never splits past this
        progress = StreamProgressTracker()
        yield progress.begin(phase="planning", model=plan_model)
        async for chunk in plan_backend.chat_stream(plan_request):
            if chunk.content_delta:
                plan_parts.append(chunk.content_delta)
            raw_delta = _strip_cot_tokens(chunk.content_delta or "")

            if marker_found:
                visible_delta = raw_delta
            elif raw_delta:
                preamble_buffer += raw_delta
                match = _PLAN_MARKER_RE.search(preamble_buffer)
                if match:
                    marker_found = True
                    visible_delta = preamble_buffer[match.start():]
                    preamble_buffer = ""
                else:
                    visible_delta = ""
            else:
                visible_delta = ""

            # Suppress the same pre-marker preamble that leaks through
            # ``thinking_delta`` for reasoning-model backends. Observed
            # 2026-04-22: Qwen / GLM / DeepSeek-R1 emitted "The user is
            # asking... I need to... Plan: describe..." via thinking_delta
            # (not content_delta), which the UI renders inline in the
            # thinking bubble. Mirror the content filter: buffer until a
            # Plan:/Question: marker appears, then release from the
            # marker onward. Separate buffer from content so a missing
            # marker on one channel doesn't starve the other.
            raw_thinking = chunk.thinking_delta or ""
            if thinking_marker_found:
                visible_thinking = raw_thinking
            elif raw_thinking:
                thinking_preamble_buffer += raw_thinking
                t_match = _PLAN_MARKER_RE.search(thinking_preamble_buffer)
                if t_match:
                    thinking_marker_found = True
                    visible_thinking = thinking_preamble_buffer[t_match.start():]
                    # Relay the not-yet-relayed remainder of the reasoning
                    # preamble, then flush so all reasoning lands before
                    # the plan text starts streaming.
                    tail = thinking_preamble_buffer[
                        thinking_relayed_upto:t_match.start()
                    ]
                    ev = reasoning_relay.add(tail) if tail else None
                    if ev is not None:
                        yield ev
                    ev = reasoning_relay.flush()
                    if ev is not None:
                        yield ev
                    thinking_preamble_buffer = ""
                    thinking_relayed_upto = 0
                else:
                    visible_thinking = ""
                    # Relay the safe prefix live (holdback guards a marker
                    # spanning chunk boundaries; the held-back tail is
                    # relayed once the marker question is settled).
                    safe = max(
                        thinking_relayed_upto,
                        len(thinking_preamble_buffer) - _REASONING_HOLDBACK,
                    )
                    if safe > thinking_relayed_upto:
                        ev = reasoning_relay.add(
                            thinking_preamble_buffer[thinking_relayed_upto:safe]
                        )
                        if ev is not None:
                            yield ev
                        thinking_relayed_upto = safe
            else:
                visible_thinking = ""

            # Progress sub-state transition. Emitted BEFORE the content
            # chunk so the UI can swap its label (e.g. "Thinking…" →
            # "Writing…") before the next visible delta lands.
            progress_chunk = progress.update(
                chunk, phase="planning", model=request.model,
            )
            if progress_chunk is not None:
                yield progress_chunk

            yield emit_relay(
                chunk,
                phase="planning", status="streaming",
                model_fallback=request.model,
                content_override=visible_delta,
                thinking_override=visible_thinking,
            )

        if not marker_found and preamble_buffer.strip():
            log.warning(
                "coder_plan_missing_marker",
                sample=preamble_buffer[:200],
                model=request.model,
            )
        if (
            not thinking_marker_found
            and thinking_preamble_buffer.strip()
        ):
            log.warning(
                "coder_plan_missing_marker_thinking",
                sample=thinking_preamble_buffer[:200],
                model=request.model,
            )
        # Stream ended with reasoning still held back (no marker arrived,
        # or the trailing holdback window) — relay the remainder. Batched,
        # never dropped.
        if thinking_preamble_buffer and thinking_relayed_upto < len(
            thinking_preamble_buffer
        ):
            _tail_ev = reasoning_relay.add(
                thinking_preamble_buffer[thinking_relayed_upto:]
            )
            if _tail_ev is not None:
                yield _tail_ev
        _final_reasoning = reasoning_relay.flush()
        if _final_reasoning is not None:
            yield _final_reasoning

        full_plan_text = "".join(plan_parts)

        # Strip any tool call JSON the model embedded in the plan
        clean_plan = _strip_tool_json(full_plan_text).strip()

        # Parse numbered steps from the plan
        steps = _parse_plan_steps(clean_plan)
        self._state.plan = clean_plan
        self._state.plan_steps = steps

        # Seed the task list from the plan so the sticky reminder shows
        # the real plan immediately — without this, act-phase iteration 1
        # renders an empty "(call task_list to plan your work)" hint even
        # though we just generated a plan. First step starts in_progress;
        # the model updates as it completes each.
        if steps and not self._state.tasks:
            seeded: list[dict] = []
            for i, step in enumerate(steps):
                seeded.append({
                    "content":    step,
                    "activeForm": step,
                    "status":     "in_progress" if i == 0 else "pending",
                })
            self._state.set_tasks(seeded)

        # Persist the plan to /workspace/.augmentum/plan.md — the
        # attention-anchor pattern from Manus. The file becomes a
        # user-inspectable artifact the agent can edit via normal
        # file_write / code_edit tools as the plan evolves. The
        # sticky reminder reads this file every iteration so the
        # plan content always lives in the context tail (compaction-
        # safe). Best-effort write: failure doesn't block the act
        # phase — in-memory state.tasks still carries the seeds.
        if clean_plan and self._container_manager is not None:
            try:
                await self._container_manager._run_command(
                    self._workspace_id,
                    ["bash", "-c", "mkdir -p /workspace/.augmentum"],
                    timeout=3.0,
                )
                await self._container_manager.file_write(
                    self._workspace_id,
                    "/workspace/.augmentum/plan.md",
                    clean_plan,
                )
            except Exception:
                log.debug("plan.persist_failed", exc_info=True)

        log.info(
            "coder.plan_complete",
            session_id=self._session_id,
            steps=len(steps),
        )

        yield self._meta_chunk(
            phase="planning",
            status="complete",
            model=request.model,
            extra={"step_count": len(steps)},
        )
