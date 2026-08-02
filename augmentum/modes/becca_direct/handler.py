"""BeccaDirectHandler — the chat-side bridge to her own prompt pipeline.

Bridges two worlds:

1. The chat substrate, which expects a :class:`ModeHandler` that takes
   an :class:`InternalChatRequest` and yields
   :class:`InternalStreamChunk` objects through ``handle_stream``. This
   is the streaming wire protocol every legacy mode (passthrough,
   narrative, agentic, coder, analytical) implements.

2. Her own prompt pipeline (currently voice-only), which composes
   prompts via :func:`compose_becca_prompt` over a per-user
   :class:`CompanionRuntime` + an :class:`Intent`. This is the path
   where she speaks as herself.

The handler builds an Intent from the chat request, gathers the same
context the voice pipeline gathers, composes her system prompt, then
streams through the primary tier — yielding chunks the chat route's
``StreamingResponse`` consumes.

**Why this matters (the thesis).** Until the chat path uses her own
prompt pipeline, half her turns route through legacy handlers that
aren't her. Her exemplar library only accumulates from voice turns,
which is a small fraction of total interaction. ``becca_direct`` is
the seam that lets every chat turn become her turn, which lets the
clock start on capability accumulation. See
``docs/superpowers/specs/2026-05-23-accumulation-thesis.md``.

**Fall-through discipline.** When the runtime isn't up, when her
persona kernel digest is empty, when prompt composition bypasses for
any reason — the handler delegates to a transparent passthrough so
the chat turn completes cleanly. *Becca being unavailable should
never break the chat path.* The "companion is optional" property is
preserved structurally.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
)
from augmentum.modes.base import ModeHandler
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from starlette.datastructures import State

log = get_logger(__name__)

# Per-tool invocation ceiling for the chat path. Voice uses cancellation
# events for finer control; chat just guards with a wall-clock cap so a
# wedged primitive can't pin the chat turn open.
_TOOL_INVOKE_TIMEOUT_S = 90.0


class BeccaDirectHandler(ModeHandler):
    """Routes a chat turn through Becca's own prompt composer + tier stream.

    Construction takes the same dependencies as a regular ModeHandler
    plus the ``app_state`` (for runtime + memory + provider registry
    access). ``user_id`` is required for per-user identity scoping —
    her kernel + relationship slice + affect read are all per-user.

    Designed so the chat route can call the same ``handle_stream``
    interface it calls on every other handler. The handler internally
    delegates to her pipeline, but the streaming chunk shape is
    standard ``InternalStreamChunk`` — the wire layer doesn't change.
    """

    # Datetime injection is handled by the parent class for the chat
    # request; her composer doesn't need additional injection since
    # the personality kernel + transcript carry the relevant freshness.
    _INJECT_DATETIME = True

    def __init__(
        self,
        backend: ModelBackend,
        *,
        app_state: State,
        session_id: str = "",
        user_id: str = "",
        surface: str = "",
    ) -> None:
        self._backend = backend
        self._app_state = app_state
        self._session_id = session_id
        self._user_id = user_id
        # Origin surface. "" = chat (default). "voice"/"assist" = the cert-free
        # on-phone path → compose the spoken, short voice prompt + the
        # phone-assist framing. Chat callers never set this, so chat is unchanged.
        self._surface = (surface or "").strip().lower()

    # ── ModeHandler interface ─────────────────────────────────────────

    # Appended to her composed system prompt when motion cues are enabled.
    # Keep the vocabulary in sync with ``MOTION_CUE_INTENT`` in
    # ui/scripts/companion-animation-router.js (which maps each word to roles).
    _MOTION_CUE_DIRECTIVE = (
        "\n\nAvatar motion (optional): when a reply genuinely carries a feeling"
        " or a physical beat — and only occasionally, never every message — you"
        " may end it with a single hidden tag on its own line, e.g."
        " [motion:happy]. The tag is removed from what the user reads; it only"
        " animates your on-screen avatar. Choose at most one of: happy, excited,"
        " dancing, bow, wave, shrug, think, curious, sad, laugh, nod, tender,"
        " shy, proud. Most replies need none — stillness reads as calm."
    )

    async def _handle(self, request: InternalChatRequest) -> InternalChatResponse:
        """Non-streaming path. Collects the full streamed response into
        a single InternalChatResponse. Used when the wire format is
        non-stream (rare — chat is almost always streaming)."""
        chunks: list[str] = []
        async for chunk in self._handle_stream(request):
            if chunk.content_delta:
                chunks.append(chunk.content_delta)
        return InternalChatResponse(
            message=Message(role="assistant", content="".join(chunks)),
        )

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Streaming path. The canonical entry point.

        Sequence:
        1. Verify runtime + identity are ready; fall through to
           passthrough on any "not ready" signal.
        2. Build an Intent from the request (last user turn + recent
           transcript in metadata).
        3. Gather composition context (relationship, recalled memory,
           facets, focus).
        4. Compose her system prompt via ``compose_becca_prompt``.
        5. Replace the request's system message with the composed one.
        6. Stream through the primary tier.
        7. Yield each delta as an ``InternalStreamChunk``.
        """
        # Resolve the companion via the unified façade.
        # ``app.state.companions[name]`` is the canonical path post-Step-2;
        # we fall back to ``app.state.companion_runtime`` only if the
        # façade hasn't been mounted (shouldn't happen in production but
        # keeps tests + dev installs robust).
        companions = getattr(self._app_state, "companions", None) or {}
        companion = companions.get("becca")
        if companion is None:
            # Compat path: build a façade on the fly if companions dict
            # isn't populated. Runtime must still be present.
            runtime = getattr(self._app_state, "companion_runtime", None)
            if runtime is not None and getattr(runtime, "_started", False):
                try:
                    from augmentum.companion import Companion
                    companion = Companion(runtime)
                except Exception:
                    companion = None

        if companion is None or not companion.started:
            async for chunk in self._fall_through_passthrough(request, reason="companion_not_ready"):
                yield chunk
            return

        # Per-user identity check. Empty user_id means we can't pull
        # her per-user kernel — fall back rather than respond as the
        # legacy seed identity to a fresh user.
        if not self._user_id:
            async for chunk in self._fall_through_passthrough(request, reason="no_user_id"):
                yield chunk
            return

        view = companion.for_user(self._user_id)
        runtime = companion.runtime

        # Speaking-tier pin. When the user pins the companion to the
        # utility tier (a low-latency small model kept separate from a
        # heavier primary chat model — companion_speak_tier="utility"),
        # re-resolve the backend + model HERE so factory-dispatched chat
        # turns speak on the pinned model, matching voice and the
        # subagent path (both already route through tiers.primary). Gated
        # on the pin being active: the default "primary" path leaves the
        # request untouched so the user's per-chat model choice still
        # flows through unchanged (the dogfooding promise).
        from augmentum.config import settings as _tier_settings
        if (getattr(_tier_settings, "companion_speak_tier", "primary") or "primary") == "utility":
            try:
                from augmentum.companion_runtime import tiers as _tiers
                _pin_backend, _pin_model = await _tiers.primary(runtime)
                if _pin_backend is not None and _pin_model:
                    self._backend = _pin_backend
                    request = dataclasses.replace(request, model=_pin_model)
            except Exception:
                log.debug("becca_direct_speak_tier_resolve_failed", exc_info=True)

        # Persona kernel digest must be present. When empty (fresh
        # install before the doc has been digested), there's nothing
        # to compose her from — fall through.
        try:
            identity = await view.identity()
        except Exception:
            log.debug("becca_direct_identity_fetch_failed", exc_info=True)
            async for chunk in self._fall_through_passthrough(request, reason="identity_lookup_failed"):
                yield chunk
            return

        digest = (identity.persona_kernel_digest or "").strip()
        if not digest:
            async for chunk in self._fall_through_passthrough(request, reason="empty_kernel_digest"):
                yield chunk
            return

        # Build Intent + gather context + compose. Any failure here
        # also falls through — composition is best-effort relative to
        # the user getting a response.
        try:
            from augmentum.companion_runtime.prompt_compose import compose_becca_prompt
            from augmentum.companion_runtime.runtime import Intent
        except Exception:
            log.warning("becca_direct_import_failed", exc_info=True)
            async for chunk in self._fall_through_passthrough(request, reason="import_failed"):
                yield chunk
            return

        intent = self._intent_from_request(request)
        try:
            ctx = await self._gather_ctx(runtime, intent)
        except Exception:
            log.debug("becca_direct_ctx_gather_failed", exc_info=True)
            ctx = {}

        try:
            composed = await compose_becca_prompt(intent, runtime, ctx)
        except Exception:
            log.warning("becca_direct_compose_failed", exc_info=True)
            async for chunk in self._fall_through_passthrough(request, reason="compose_failed"):
                yield chunk
            return

        if composed.bypass_reason or not composed.system_text:
            log.info("becca_direct_compose_bypass", reason=composed.bypass_reason or "empty_system_text")
            async for chunk in self._fall_through_passthrough(request, reason=composed.bypass_reason or "compose_empty"):
                yield chunk
            return

        # Avatar motion cues — let the model optionally end a reply with a
        # hidden ``[motion:xxx]`` tag that drives her on-screen avatar. The tag
        # is stripped client-side (motion-cue.js) before render/save, and the
        # cue maps to animation roles via the user's curated/rated/uploadable
        # clip pool. One model call, no extra tool round-trip. Setting-gated.
        from augmentum.config import settings as _mc_settings
        system_text = composed.system_text
        if getattr(_mc_settings, "companion_motion_cues_enabled", True):
            system_text = system_text + self._MOTION_CUE_DIRECTIVE

        # Substitute the system message with the composed one. The
        # rest of the request stays — user/assistant history, model,
        # streaming flag, tool config — so the backend sees a normal
        # chat request, just with her prompt in front.
        rewritten = self._substitute_system_message(request, system_text)

        # Emit bus event so observers know the chat path went through
        # her composer. Cheap signal that this turn is a "her" turn for
        # downstream accumulation (exemplar library, etc.).
        try:
            await runtime.bus.publish_topic(
                "becca_direct.invoked",
                {
                    "user_id": self._user_id,
                    "session_id": self._session_id,
                    "layers_used": composed.layers_used,
                    "refusal_mode": composed.refusal_mode,
                },
                source_companion_id=runtime.companion_id,
            )
        except Exception:
            log.debug("becca_direct_bus_emit_failed", exc_info=True)

        # NATIVE TOOL LOOP — Companion Agency MVP #1 (specs 2026-06-10).
        # Coder navigates a workspace; the companion navigates the
        # application — on the SAME engine. Tools arrive as grammar-
        # constrained schemas (passthrough's tier machinery), results
        # round-trip, and multi-intent turns ("while you do that, also
        # check the news") fall out for free. Acting stops being a
        # persona style choice. Any failure falls through to the legacy
        # tag path so the rollout is structurally safe.
        from augmentum.config import settings as _settings
        if getattr(_settings, "companion_native_toolloop", True):
            try:
                async for chunk in self._stream_native_loop(
                    rewritten, runtime=runtime, intent=intent,
                ):
                    yield chunk
                return
            except Exception:
                log.exception("becca_native_loop_failed_falling_back")

        # Stream through the backend, sifting tool/handoff tags out of
        # the visible text. The runtime + user_id are needed so detected
        # tool tags can dispatch via ``companion_runtime.tools.execute_tool``.
        try:
            async for chunk in self._stream_backend(rewritten, runtime=runtime):
                yield chunk
        except Exception:
            log.exception("becca_direct_stream_failed")
            # On stream failure, still try to close the chat turn cleanly.
            # We can't fall back to passthrough mid-stream — the user has
            # already started receiving tokens. Emit an empty done chunk.
            yield InternalStreamChunk(
                content_delta="",
                done=True,
            )

    # ── Internal: native tool loop (Companion Agency MVP #1) ──────────

    async def _stream_native_loop(
        self,
        request: InternalChatRequest,
        *,
        runtime: Any,
        intent: Any,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Run the companion turn on the shared NATIVE tool loop.

        The loop core lives in ``companion_runtime/native_loop.py``
        (extracted 2026-06-10 for the headless-agency work so voice
        consumes the same loop). This wrapper only maps loop events
        onto the becca_tool_* chunk shapes the client already renders.
        """
        from augmentum.companion_runtime.native_loop import native_loop_events

        registry = getattr(self._app_state, "tool_registry", None)
        model_name = request.model or ""

        async for kind, payload in native_loop_events(
            request,
            backend=self._backend,
            runtime=runtime,
            intent=intent,
            registry=registry,
            user_id=self._user_id,
            session_id=self._session_id,
            app_state=self._app_state,
        ):
            if kind == "tool_call":
                yield InternalStreamChunk(
                    content_delta="",
                    model=model_name,
                    augmentum={"becca_tool_call": payload},
                )
            elif kind == "tool_result":
                yield InternalStreamChunk(
                    content_delta="",
                    model=model_name,
                    augmentum={"becca_tool_result": {
                        **payload,
                        "ui_effects": [],
                        "error": "" if payload.get("ok") else "tool_failed",
                    }},
                )
            elif kind == "text":
                yield InternalStreamChunk(
                    content_delta=payload.get("text", ""), model=model_name,
                )
            elif kind == "ui_effects":
                yield InternalStreamChunk(
                    content_delta="",
                    model=model_name,
                    augmentum={"becca_tool_result": {
                        "tool": "surface",
                        "ok": True,
                        "payload_summary": "",
                        "duration_ms": 0,
                        "ui_effects": payload.get("effects", []),
                        "error": "",
                    }},
                )
            elif kind == "metrics":
                # Final-turn telemetry (tok/s) → forwarded for the assist
                # overlay's orb. Carried on its own augmentum key so chat
                # consumers (which don't read it) are unaffected.
                yield InternalStreamChunk(
                    content_delta="",
                    model=model_name,
                    augmentum={"becca_metrics": payload},
                )

        yield InternalStreamChunk(content_delta="", done=True, model=model_name)

    # ── Internal: Intent + ctx + streaming ────────────────────────────

    def _intent_from_request(self, request: InternalChatRequest) -> Any:
        """Build an :class:`Intent` from the chat request.

        Pulls the last user-role message as ``intent.text``. Earlier
        turns are passed via ``metadata.recent_turns`` so the composer
        can render them in Layer 9 (transcript window). The
        ``voice_channel`` flag is *not* set — that gates the voice-only
        addendum in the composer, and this is the chat path.
        """
        from augmentum.companion_runtime.runtime import Intent

        # Last user message → intent.text. The composer needs only the
        # current user turn here; recent_turns carries the rest.
        last_user_text = ""
        for msg in reversed(request.messages or []):
            if getattr(msg, "role", "") == "user":
                last_user_text = getattr(msg, "content", "") or ""
                break

        # Recent turns for the transcript window. Skip the current
        # user turn (already passed as intent.text) — composer's
        # _transcript_window expects history WITHOUT the current.
        history: list[dict] = []
        current_user_seen = False
        for msg in reversed(request.messages or []):
            role = getattr(msg, "role", "")
            if not current_user_seen and role == "user":
                current_user_seen = True
                continue
            if role in ("user", "assistant"):
                history.append({
                    "role": role,
                    "content": getattr(msg, "content", "") or "",
                })
        history.reverse()

        # On-phone path ("voice"/"assist"): tell the composer this turn is
        # SPOKEN (Layer 8.5 → short, no-markdown, TTS-safe + the lean voice
        # transcript window) and that she's acting from inside the user's phone
        # (Layer 8.7 → phone-assist framing). Chat leaves _surface="" so neither
        # fires and chat composes exactly as before.
        is_phone = self._surface in ("voice", "assist")
        metadata: dict[str, Any] = {
            "session_id": self._session_id,
            "recent_turns": history,
        }
        if is_phone:
            metadata["voice_channel"] = True
            metadata["phone_assist"] = True
            metadata["surface"] = self._surface
        return Intent(
            text=last_user_text,
            user_id=self._user_id,
            source="user_chat",
            device_id="",
            explicit_mode="",
            metadata=metadata,
        )

    async def _gather_ctx(self, runtime, intent) -> dict[str, Any]:
        """Pre-fetch composition inputs in parallel.

        Mirrors :meth:`BeccaVoice._gather_ctx` so chat + voice see the
        same context shape going into the composer. Differences kept
        intentional: tools/channels enumeration uses the same bridges;
        relationship + facets + recall + focus all read the same
        per-user state.
        """
        from augmentum.companion_runtime import tool_bridge
        from augmentum.companion_runtime.state import FocusValue

        user_id = intent.user_id or ""

        async def _rel() -> str:
            try:
                memory = getattr(runtime, "memory", None)
                if memory is None or not user_id:
                    return ""
                return await memory.get_relationship_profile(user_id) or ""
            except Exception:
                log.debug("becca_direct_rel_fetch_failed", exc_info=True)
                return ""

        async def _recalled_memory() -> str:
            memory = getattr(runtime, "memory", None)
            if memory is None or not user_id or not (intent.text or "").strip():
                return ""
            try:
                rows = await memory.recall(intent.text, user_id=user_id, k=6)
            except Exception:
                log.debug("becca_direct_recall_failed", exc_info=True)
                return ""
            if not rows:
                return ""
            lines: list[str] = []
            for row in rows:
                text = ""
                try:
                    text = getattr(row, "content", None) or (
                        row.get("content") if isinstance(row, dict) else ""
                    ) or ""
                except Exception:
                    text = ""
                text = (text or "").strip().replace("\n", " ")
                if not text:
                    continue
                lines.append(f"- {text[:240]}")
                if len(lines) >= 6:
                    break
            return "\n".join(lines)

        async def _facets() -> dict[str, float]:
            store = getattr(runtime, "personality_store", None)
            if store is None or not user_id:
                return {}
            try:
                from augmentum.personality.graph import compose_facet_affects
                return await compose_facet_affects(
                    store,
                    user_id=user_id,
                    companion_id=runtime.companion_id,
                    recent_hours=24,
                    limit=8,
                )
            except Exception:
                log.debug("becca_direct_facets_failed", exc_info=True)
                return {}

        async def _threads() -> list[str]:
            # Open commitments fill the Layer-5 open-threads slot —
            # same ledger as the voice path (commitments.py).
            try:
                from augmentum.companion_runtime import commitments
                return await commitments.open_threads(
                    runtime, user_id=user_id,
                )
            except Exception:  # noqa: BLE001
                log.debug("becca_direct_threads_fetch_failed", exc_info=True)
                return []

        async def _engineering() -> list[str]:
            # Recent collaborative coding work carried across sessions —
            # same ledger as voice (engineering_log.py → prompt Layer 5.7).
            try:
                from augmentum.companion_runtime import engineering_log
                return await engineering_log.recent_engineering(
                    runtime, user_id=user_id,
                )
            except Exception:  # noqa: BLE001
                log.debug("becca_direct_engineering_fetch_failed", exc_info=True)
                return []

        rel, facets, threads, recalled, eng_threads = await asyncio.gather(
            _rel(), _facets(), _threads(), _recalled_memory(), _engineering(),
        )

        try:
            focus_str = runtime.state.snapshot().get("focus") or "none"
            fv = FocusValue.from_str(focus_str) if isinstance(focus_str, str) else None
            focus = (
                {"kind": fv.kind.value, "value": fv.payload}
                if fv is not None
                else {}
            )
        except Exception:
            focus = {}

        # Tool + channel enumeration — same bridge voice uses. The chat
        # path consumes tool tags via ``_stream_backend``'s TagSieve loop,
        # so the composer's tool block is now load-bearing: every
        # advertised primitive is dispatchable from the chat surface.
        # Roster scoring text: current turn + previous user turn, so
        # follow-ups ("try that again") keep the referred-to verbs in
        # the relevance-ranked roster. Mirrors BeccaVoice's
        # _roster_scoring_text.
        roster_text = intent.text or ""
        try:
            turns = (intent.metadata or {}).get("recent_turns") or []
            prev_user = next(
                (
                    (t.get("content") or "")
                    for t in reversed(turns)
                    if t.get("role") == "user"
                ),
                "",
            )
            if prev_user:
                roster_text = f"{roster_text} {prev_user[:200]}"
        except Exception:  # noqa: BLE001 — context is a bonus
            log.debug("roster_context_window_failed", exc_info=True)
        try:
            from augmentum.companion_runtime.context_budget import (
                derive_roster_char_budget,
                resolve_context_length,
            )

            _ctx_len = await resolve_context_length(runtime)
            tools = tool_bridge.enumerate_tools(
                roster_text,
                context_budget_chars=derive_roster_char_budget(_ctx_len),
            )
            channels = tool_bridge.enumerate_channels()
        except Exception:
            log.debug("becca_direct_tool_bridge_failed", exc_info=True)
            tools = []
            channels = []

        # Layer 5.6 — relevant skills (thesis Step 3). Reads through
        # the Companion façade so the feature flag + thresholds are
        # honored in one place. Empty list when the feature is off or
        # the graph hasn't accumulated yet — composer's helper
        # silently skips empty layers, so no injection happens until
        # there's real evidence to draw on.
        relevant_skills: list = []
        relevant_lessons: list = []
        try:
            companions = getattr(self._app_state, "companions", None) or {}
            companion = companions.get(runtime.companion_id)
            if companion is not None and self._user_id:
                view = companion.for_user(self._user_id)
                relevant_skills = await view.relevant_skills(intent.text or "")
                # Layer 5.7 — relevant lessons (mig 270): corrections to
                # honor for this situation. Same façade + flag discipline;
                # empty until the registry has a relevant hit.
                relevant_lessons = await view.relevant_lessons(intent.text or "")
        except Exception:
            log.debug("becca_direct_skills_fetch_failed", exc_info=True)
            relevant_skills = []
            relevant_lessons = []

        # Presence — same perception organ as the voice path. The
        # perception contract (fidelity-marked index + results ring +
        # blind line) replaces the old always-push excerpt/note-tail;
        # this call also advances the ring's turn clock.
        now_lines: list[str] = []
        try:
            from augmentum.architect.primitives.grove_match import (
                _conn_from_runtime,
            )
            from augmentum.companion_runtime.presence_context import (
                perception_lines,
            )
            _conn = await _conn_from_runtime(runtime)
            now_lines = await perception_lines(
                self._app_state, _conn, user_id, self._session_id,
                scoring_text=intent.text or "",
            )
        except Exception:  # noqa: BLE001
            log.debug("becca_direct_presence_ctx_failed", exc_info=True)

        return {
            "relationship_profile": rel,
            "recalled_memory": recalled,
            "facets": facets,
            "open_threads": threads,
            "engineering_threads": eng_threads,
            "focus": focus,
            "now_context": now_lines,
            "tools": tools,
            "channels": channels,
            "relevant_skills": relevant_skills,
            "relevant_lessons": relevant_lessons,
        }

    def _substitute_system_message(
        self,
        request: InternalChatRequest,
        system_text: str,
    ) -> InternalChatRequest:
        """Return a copy of ``request`` with the system message
        replaced by ``system_text``.

        Uses ``dataclasses.replace`` so any future fields on
        InternalChatRequest propagate cleanly (see the field-addition
        warning in ``augmentum/models/base.py``).
        """
        import dataclasses

        new_messages: list[Message] = []
        replaced = False
        for msg in request.messages or []:
            if not replaced and getattr(msg, "role", "") == "system":
                new_messages.append(Message(role="system", content=system_text))
                replaced = True
            else:
                new_messages.append(msg)
        if not replaced:
            new_messages.insert(0, Message(role="system", content=system_text))

        return dataclasses.replace(request, messages=new_messages)

    async def _stream_backend(
        self,
        request: InternalChatRequest,
        *,
        runtime: Any | None = None,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream from the backend with mid-stream tool-tag dispatch.

        Closes the gap called out in :meth:`_gather_ctx`'s tools-block
        comment ("parser is voice-only"). Becca's composer already
        advertises the tool roster on the chat path — this consumer
        actually fires the tags she emits, mirroring voice's
        ``TagSieve`` loop.

        Wire shape:
          - Clean prose chunks pass through as standard content_delta.
          - ``<tool:NAME ... />`` tags trigger
            ``companion_runtime.tools.execute_tool`` and emit two
            augmentum-metadata chunks: ``becca_tool_call`` (announce)
            then ``becca_tool_result`` (outcome). Becca's primary stream
            continues — synthesis of the result back into her voice is a
            follow-up (see voice's ``_synthesize``).
          - ``<handoff:CHANNEL ... />`` tags emit a single
            ``becca_handoff`` chunk and terminate the turn (the UI is
            expected to mount the channel surface).
          - Budget caps from voice (``MAX_TOOLS_PER_TURN`` /
            ``MAX_TOOL_BUDGET_S``) apply identically; once exhausted, a
            ``becca_tool_budget_exhausted`` chunk lands and further
            tool tags are dropped silently.

        Falls back to the previous straight-passthrough behavior when
        ``runtime`` is ``None`` (defensive — caller always passes it in
        production, but tests sometimes don't).
        """
        if runtime is None:
            async for chunk in self._raw_stream(request):
                yield chunk
            return

        from augmentum.companion_runtime import tools as tool_bridge
        from augmentum.companion_runtime.tool_protocol import TagSieve
        from augmentum.companion_runtime.voice import (
            MAX_TOOL_BUDGET_S,
            MAX_TOOLS_PER_TURN,
        )

        # Salvage-enabled: mangled tag prefixes recover when the name
        # resolves against the known verb set (mirrors BeccaVoice).
        sieve = TagSieve(known_tools=tool_bridge.known_tool_names)
        tools_used = 0
        tool_budget_s = 0.0
        budget_announced = False

        async for chunk in self._raw_stream(request):
            delta = chunk.content_delta or ""
            if not delta:
                # Forward non-content chunks (role/finish/done/augmentum)
                # untouched. The sieve only cares about prose deltas.
                yield chunk
                continue

            for clean, tag in sieve.feed(delta):
                if clean:
                    yield InternalStreamChunk(
                        content_delta=clean,
                        model=chunk.model,
                    )
                if tag is None:
                    continue

                # Handoff terminates the turn — UI mounts the channel
                # surface from the augmentum chunk and stops rendering
                # primary output. Don't execute the channel handoff here;
                # the chat side just signals.
                if tag.kind == "handoff":
                    yield InternalStreamChunk(
                        content_delta="",
                        model=chunk.model,
                        augmentum={"becca_handoff": {
                            "channel": tag.name,
                            "args": dict(tag.args),
                        }},
                    )
                    yield InternalStreamChunk(
                        content_delta="",
                        done=True,
                        model=chunk.model,
                    )
                    return

                # Tool tag. Enforce per-turn budget.
                if (
                    tools_used >= MAX_TOOLS_PER_TURN
                    or tool_budget_s >= MAX_TOOL_BUDGET_S
                ):
                    if not budget_announced:
                        yield InternalStreamChunk(
                            content_delta="",
                            model=chunk.model,
                            augmentum={"becca_tool_budget_exhausted": {
                                "tools_used": tools_used,
                                "budget_s": round(tool_budget_s, 2),
                            }},
                        )
                        budget_announced = True
                    continue

                yield InternalStreamChunk(
                    content_delta="",
                    model=chunk.model,
                    augmentum={"becca_tool_call": {
                        "tool": tag.name,
                        "args": dict(tag.args),
                    }},
                )

                t_start = time.monotonic()
                result_chunk = await self._invoke_tool_safe(
                    tag, runtime, model=chunk.model,
                )
                tool_budget_s += time.monotonic() - t_start
                tools_used += 1
                yield result_chunk

        # Primary stream ended naturally — flush any trailing sieve
        # buffer. By definition this is text (no complete tag waiting),
        # so emit as content.
        tail = sieve.flush()
        if tail:
            yield InternalStreamChunk(content_delta=tail)

    async def _raw_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Bare backend stream — used by ``_stream_backend`` as the source
        and as the fallback when no runtime is available.

        Prefers ``chat_stream``, falls back to ``chat`` for backends that
        don't stream.
        """
        chat_stream = getattr(self._backend, "chat_stream", None)
        if chat_stream is not None:
            async for chunk in chat_stream(request):
                yield chunk
            return

        chat = getattr(self._backend, "chat", None)
        if chat is None:
            raise RuntimeError(
                "becca_direct: primary backend has neither chat_stream nor chat",
            )
        resp = await chat(request)
        from augmentum.models.base import response_text
        content = response_text(resp)
        yield InternalStreamChunk(
            content_delta=content or "",
            done=True,
        )

    async def _invoke_tool_safe(
        self,
        tag: Any,
        runtime: Any,
        *,
        model: str,
    ) -> InternalStreamChunk:
        """Execute a tool tag with timeout + crash guards. Returns a
        single ``becca_tool_result`` augmentum chunk describing the
        outcome — never raises.
        """
        from augmentum.companion_runtime import tools as tool_bridge

        try:
            result = await asyncio.wait_for(
                tool_bridge.execute_tool(
                    tag, runtime,
                    user_id=self._user_id,
                    session_id=self._session_id,
                ),
                timeout=_TOOL_INVOKE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.info("becca_direct_tool_timeout", tool=tag.name)
            return InternalStreamChunk(
                content_delta="",
                model=model,
                augmentum={"becca_tool_result": {
                    "tool": tag.name,
                    "ok": False,
                    "error": "timeout",
                }},
            )
        except Exception:
            log.exception("becca_direct_tool_crashed", tool=tag.name)
            return InternalStreamChunk(
                content_delta="",
                model=model,
                augmentum={"becca_tool_result": {
                    "tool": tag.name,
                    "ok": False,
                    "error": "crash",
                }},
            )

        return InternalStreamChunk(
            content_delta="",
            model=model,
            augmentum={"becca_tool_result": {
                "tool": tag.name,
                "ok": result.ok,
                "payload_summary": tool_bridge.summarize_payload(result.payload),
                "duration_ms": result.duration_ms,
                "ui_effects": [
                    {"kind": e.kind, "target": e.target, "payload": e.payload}
                    for e in result.ui_effects
                ],
                "error": (
                    result.error.category if result.error else ""
                ),
            }},
        )

    async def _fall_through_passthrough(
        self,
        request: InternalChatRequest,
        *,
        reason: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Fall back to passthrough behavior when becca_direct can't run.

        This is the "companion is optional" guarantee — under any
        not-ready condition (runtime down, kernel empty, compose failure)
        the chat turn still completes cleanly via the standard
        passthrough handler. The user never sees a degraded experience
        because Becca was unavailable.
        """
        log.info("becca_direct_falling_through", reason=reason)
        try:
            from augmentum.modes.passthrough.handler import PassthroughHandler
            tool_registry = getattr(self._app_state, "tool_registry", None)
            fallback = PassthroughHandler(
                backend=self._backend,
                session_id=self._session_id,
                tool_registry=tool_registry,
                user_id=self._user_id,
                app_state=self._app_state,
            )
            async for chunk in fallback._handle_stream(request):  # noqa: SLF001
                yield chunk
        except Exception:
            log.exception("becca_direct_fallthrough_failed")
            yield InternalStreamChunk(
                content_delta="",
                done=True,
            )


__all__ = ["BeccaDirectHandler"]
