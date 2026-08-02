"""BeccaVoice — Becca's own generation pipeline (Lane 1 §1, §11).

The new code that lets Becca speak. She is NOT a subagent in the registry;
she is CompanionRuntime's own generation step. Subagents/primitives are
her tools (single-shot inline) or channels (multi-turn handoffs); their
prompts stay untouched.

ARCHITECTURAL INVARIANT — responsiveness is unconditional.
A user-ADDRESSED turn ALWAYS produces a warm response, regardless of any
interior "economy" state (energy, mana, drives, berries). This is the
responsive path; it MUST NEVER import or consult
``companion_runtime.energy``, ``companion.growth.economy``, or
``companion_runtime.behavior.{activity_selector,tick}``. Those govern what
she does UNPROMPTED (the autonomous tick loop) — never how she MEETS you.
Energy may shape what she *initiates* and never how fast or warmly she
*answers*. Enforced structurally by ``tests/test_responsiveness_invariant``;
if you came here to gate a reply on capacity, you are in the wrong path —
wire it into ``behavior/tick.py`` instead.

Sprint B scope:
- Module structure + class skeleton + chat-handler entry point
- Triage classifier call (stubbed — returns PURE_EQ for now)
- Context pre-fetch with parallel awaits (Lane 2 fields stubbed; wired in Sprint F)
- Prompt composition (via ``prompt_compose``)
- Primary-tier streaming call with the ``TagSieve``
- Bus events emitted at the right boundaries
- StreamingResponse / JSONResponse construction

Sprint C fills in tool invocation. Sprint D fills in channel handoff.
Sprint F fills in the labeler call + the personality-graph update.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, replace as dataclass_replace
from typing import TYPE_CHECKING, Any, AsyncIterator

from augmentum.config import settings
from augmentum.companion_runtime import affordances, tiers, tools as tool_bridge
from augmentum.companion_runtime.prompt_compose import (
    ComposedPrompt,
    compose_becca_prompt,
)
from augmentum.companion_runtime.state import FocusValue
from augmentum.companion_runtime.tool_protocol import (
    Promise, TagSieve, ToolCall, ToolResult,
)
from augmentum.companion_runtime.runtime import Intent
from augmentum.utils.bg_tasks import track
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import JSONResponse, StreamingResponse

    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.models.base import InternalChatRequest

log = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────

# Maximum tool calls in a single turn before soft handoff (Lane 1 §3.6).
MAX_TOOLS_PER_TURN = 5

# Total tool latency in a turn before soft handoff (60s).
MAX_TOOL_BUDGET_S = 60.0

# Long-tail threshold per tool — emit a second affordance if tool exceeds.
LONG_TAIL_THRESHOLD_S = 4.0


# ── Exceptions ───────────────────────────────────────────────────────

class BeccaBypassed(Exception):
    """Raised by ``BeccaVoice.handle_chat`` to fall through to the
    legacy chat path. Reasons enumerated in Lane 1 §10.3:
    no_digest, runtime_not_ready, background_task, no_persona_header.
    """
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ── Streaming-response helpers ───────────────────────────────────────

@dataclass(slots=True)
class _AsyncStringWriter:
    """Async writer for in-process composition tests.

    The route adapter wraps a real SSE source; tests use this writer to
    accumulate the full streamed response in memory.
    """
    buffer: list[str]

    @classmethod
    def empty(cls) -> "_AsyncStringWriter":
        return cls(buffer=[])

    async def write(self, text: str) -> None:
        if text:
            self.buffer.append(text)

    async def close(self) -> None:
        pass

    def as_text(self) -> str:
        return "".join(self.buffer)


def _assemble_streamed_tool_calls(deltas: list[dict]) -> list[ToolCall]:
    """Reassemble OpenAI streaming tool-call deltas into ``ToolCall``s.

    Streaming emits one tool call across many chunks: the first carries
    ``function.name`` (and an ``id``), later chunks append fragments of
    ``function.arguments`` (a partial JSON string), all keyed by
    ``index``. We concatenate per index, decode the JSON, and stringify
    values to the tag protocol's ``dict[str, str]`` arg shape — the same
    shape the native loop's executor consumes for ``initial_calls``.
    """
    by_index: dict[int, dict[str, str]] = {}
    order: list[int] = []
    for d in deltas:
        if not isinstance(d, dict):
            continue
        idx = d.get("index", 0)
        if idx not in by_index:
            by_index[idx] = {"name": "", "args": ""}
            order.append(idx)
        slot = by_index[idx]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        frag = fn.get("arguments")
        if frag:
            slot["args"] += frag
    out: list[ToolCall] = []
    for idx in order:
        slot = by_index[idx]
        name = (slot["name"] or "").strip()
        if not name:
            continue
        raw_args = slot["args"].strip()
        parsed: Any = {}
        if raw_args:
            try:
                parsed = json.loads(raw_args)
            except Exception:  # noqa: BLE001 — malformed args → empty dict
                log.warning(
                    "voice_tool_call_args_unparseable",
                    tool=name, preview=raw_args[:160],
                )
                parsed = {}
        args: dict[str, str] = {}
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, dict | list):
                    args[str(k)] = json.dumps(v)
                else:
                    args[str(k)] = "" if v is None else str(v)
        out.append(ToolCall(kind="tool", name=name, args=args, raw="", span=(0, 0)))
    return out


# ── BeccaVoice ───────────────────────────────────────────────────────

class BeccaVoice:
    """Becca's generation pipeline. Stateless per-invocation; constructed
    fresh by the route adapter on each chat turn.
    """

    def __init__(self, runtime: "CompanionRuntime") -> None:
        self._runtime = runtime
        self._bus = runtime.bus

    # ── Public entry: route adapter calls this ──────────────────────

    async def handle_chat(
        self,
        internal_req: "InternalChatRequest",
        request: "Request",
        *,
        classification: Any,
        user_id: str,
        session_id_hint: str,
        stream: bool,
        wire_format: str = "ollama",
    ):
        """Entry point invoked by the chat-handler fork.

        Returns a StreamingResponse / JSONResponse shaped like the legacy
        handler's response. Raises ``BeccaBypassed`` to fall through to
        legacy on any condition Becca can't meet.
        """
        # Pre-flight gates (Lane 1 §10.3)
        if not self._runtime.identity.persona_kernel_digest:
            raise BeccaBypassed("no_digest")
        if getattr(self._runtime, "_started", False) is False:
            raise BeccaBypassed("runtime_not_ready")

        # Triage classifier (Sprint B: stub — returns PURE_EQ unless the
        # classifier already chose a mode, in which case treat as a
        # CHANNEL_* hint).
        triage = await self._triage(internal_req, classification)
        meta: dict[str, Any] = {"triage_label": triage}
        if triage in ("HONEST_REFUSAL", "FLOOR"):
            meta["refusal_category"] = self._refusal_category(triage, internal_req)

        # Vision is applied once, in ``stream`` → ``_apply_vision_to_intent``
        # (both the streaming and blocking responses route through it). Here
        # we only need ``_intent_from_request`` to carry any attached frames
        # forward on ``intent.metadata['images']`` — raw data: URLs the
        # pipeline then resolves VL-direct or sibling-captioned. Applying the
        # pipeline here too would double-run it on a VL primary.
        intent = self._intent_from_request(
            internal_req, user_id=user_id, meta=meta,
            session_id_hint=session_id_hint,
        )

        if stream:
            return self._build_streaming_response(intent, wire_format=wire_format)
        return await self._build_blocking_response(intent)

    # ── Triage ──────────────────────────────────────────────────────

    async def _triage(self, internal_req: "InternalChatRequest", classification: Any) -> str:
        """Return one of:
            PURE_EQ | MIXED | IQ_HEAVY |
            CHANNEL_CODER | CHANNEL_NARRATIVE | CHANNEL_AGENTIC |
            CHANNEL_BUILD | CHANNEL_BUG |
            HONEST_REFUSAL | FLOOR

        Order of checks (most specific → least):
          1. Safety regression floor — explicit acute language → FLOOR
          2. Hard refusal categories — harm-uplift / minor-explicit
          3. Explicit mode from classifier → CHANNEL_*
          4. Fallback → PURE_EQ
        """
        # The floor and refusal classifiers inspect the WHOLE user turn,
        # not just the last message — acute/uplift content in an earlier
        # message of a multi-message turn previously slipped the floor
        # entirely (audit 2026-06-17). ``all_user_text`` is the joined
        # turn; ``last_user_text`` is kept for the mode/channel branch.
        user_texts: list[str] = []
        for m in (internal_req.messages or []):
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            if role == "user":
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
                if isinstance(content, str) and content.strip():
                    user_texts.append(content)
        last_user_text = user_texts[-1] if user_texts else ""
        all_user_text = "\n".join(user_texts)

        # (1) Safety floor — surface="voice" so the threshold and audit
        # rows attribute to the right surface (was mislabeled free_chat).
        try:
            from augmentum.companion_runtime import safety_floor
            floor = safety_floor.classify(all_user_text, surface="voice")
            if floor.fired:
                # Fire-and-forget audit write (tracked: GC-safe + logged)
                track(safety_floor.audit_event(
                    self._runtime, floor,
                    turn_id="", locale=getattr(settings, "companion_locale", ""),
                ))
                return "FLOOR"
        except Exception:
            # log.warning, NOT debug: a regex/import regression here
            # silently disables the acute-language floor on every turn.
            # Surfacing it is the real protection — a hard FLOOR on every
            # transient error would break normal voice (audit 2026-06-17).
            log.warning("voice_triage_floor_failed", exc_info=True)

        # (2) Hard refusal
        try:
            from augmentum.companion_runtime import narrative_isolation
            refusal_cat = narrative_isolation.frame_invariant_check(all_user_text)
            if refusal_cat in ("uplift_risk", "minor_explicit"):
                # Stash for refusal addendum lookup
                return "HONEST_REFUSAL"
        except Exception:
            log.warning("voice_triage_refusal_failed", exc_info=True)

        # (3) Explicit mode → channel
        try:
            mode_value = getattr(getattr(classification, "mode", None), "value", "")
        except Exception:
            mode_value = ""
        if mode_value in ("coder", "narrative", "agentic", "build", "bug_finder"):
            tag = "CHANNEL_BUG" if mode_value == "bug_finder" else f"CHANNEL_{mode_value.upper()}"
            return tag

        # (4) Default
        return "PURE_EQ"

    def _refusal_category(self, triage: str, internal_req: "InternalChatRequest") -> str:
        """Map the categorical classifier output to the addendum key
        used by ``affordances.refusal_addendum_for``. Defaults to
        harm_uplift on ambiguity — it's the safer assumption."""
        last_user_text = ""
        for m in (internal_req.messages or []):
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            if role == "user":
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
                if isinstance(content, str):
                    last_user_text = content
        try:
            from augmentum.companion_runtime import narrative_isolation
            cat = narrative_isolation.frame_invariant_check(last_user_text)
        except Exception:
            cat = "uplift_risk"
        if cat == "minor_explicit":
            return "minor_explicit"
        return "harm_uplift"

    # ── Intent construction ─────────────────────────────────────────

    def _intent_from_request(
        self,
        req: "InternalChatRequest",
        *,
        user_id: str,
        meta: dict[str, Any],
        session_id_hint: str,
    ) -> Intent:
        """Translate an InternalChatRequest into a CompanionRuntime Intent.

        The last user message becomes the intent text; prior messages
        become ``metadata['recent_turns']`` for the transcript layer.
        """
        # Find the last user message and the prior turns.
        text = ""
        images: list[str] = []
        recent_turns: list[dict[str, Any]] = []
        for msg in (req.messages or []):
            role = getattr(msg, "role", None) or msg.get("role") if isinstance(msg, dict) else None
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            if role == "user":
                text = content
                # Carry any surviving image attachments forward. After the
                # vision pipeline runs, only a VL primary leaves images on
                # the message (text-only primaries get them captioned-and-
                # stripped into ``content`` already), so these are the
                # frames the primary can read directly.
                msg_images = (
                    getattr(msg, "images", None)
                    if not isinstance(msg, dict)
                    else msg.get("images")
                )
                images = [i for i in (msg_images or []) if isinstance(i, str) and i.strip()]
            recent_turns.append({"role": role or "user", "content": content})
        # The very last user turn is the intent; pop it from recent_turns
        # so layer 9 doesn't repeat what layer 10 already has.
        if recent_turns and recent_turns[-1].get("role") == "user":
            recent_turns = recent_turns[:-1]

        merged_meta = {
            **meta,
            "recent_turns": recent_turns,
            "session_id": session_id_hint,
            "images": images,
        }

        return Intent(
            text=text,
            user_id=user_id,
            source="user_chat",
            device_id="",
            explicit_mode="",
            metadata=merged_meta,
        )

    # ── Context gather (Lane 1 §2.3) ────────────────────────────────

    async def _gather_ctx(self, intent: Intent) -> dict[str, Any]:
        """Pre-fetch composition inputs in parallel.

        Sprint B: facets/threads/tools/channels are placeholders. Sprint F
        wires the personality store + thread tracker. Sprint C wires the
        tool/channel catalogue from registries.
        """
        runtime = self._runtime
        user_id = intent.user_id or ""

        # Loaded-model context window → scales the roster budget. Cached per
        # model on the runtime; 0 on any failure → fixed fallback. compose_
        # becca_prompt re-reads the same cached value for the transcript window.
        from augmentum.companion_runtime.context_budget import (
            resolve_context_length,
        )

        _ctx_len = await resolve_context_length(runtime)

        async def _rel() -> str:
            try:
                memory = getattr(runtime, "memory", None)
                if memory is None or not user_id:
                    return ""
                # CompanionMemory.get_relationship_profile (Sprint 1) wraps
                # CoreProfileManager.get_profile.
                return await memory.get_relationship_profile(user_id) or ""
            except Exception as exc:
                log.debug("voice_relationship_fetch_failed", error=str(exc)[:200])
                return ""

        async def _recalled_memory() -> str:
            """Semantic recall via the same MemoryStore the legacy chat
            path uses. Returned as a compact bulleted summary so the
            composer can render it as a single prompt block.
            """
            memory = getattr(runtime, "memory", None)
            if memory is None or not user_id or not (intent.text or "").strip():
                return ""
            try:
                rows = await memory.recall(intent.text, user_id=user_id, k=6)
            except Exception as exc:
                log.debug("voice_recall_failed", error=str(exc)[:200])
                return ""
            if not rows:
                return ""
            # Calibrated voice: each bullet carries its earned-tier register
            # cue so she speaks an unproven impression tentatively and a
            # CORE fact plainly (Earned Understanding P1).
            from augmentum.memory.register import calibrated_bullets
            return calibrated_bullets(rows, limit=6)

        async def _facets() -> dict[str, float]:
            """Lane 2 §1 — pre-prompt facet composition. Reads recent
            activations + cooccurrence + memory-association graph for
            this (user, companion). Returns {facet: normalized_score}.
            """
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
                log.debug("voice_facets_compose_failed", exc_info=True)
                return {}

        async def _threads() -> list[str]:
            # Open commitments ARE her open threads — the ledger
            # (commitments.py) fills the Layer-5 slot that sat empty
            # since Sprint F. "Things you've been sitting with" is
            # exactly what an unmet ask is.
            try:
                from augmentum.companion_runtime import commitments
                return await commitments.open_threads(
                    runtime, user_id=user_id,
                )
            except Exception:  # noqa: BLE001 — ledger never breaks a turn
                log.debug("voice_threads_fetch_failed", exc_info=True)
                return []

        async def _engineering() -> list[str]:
            # Recent collaborative coding work she carries across sessions —
            # the persistence loop that lets a stateless coding agent feel
            # continuous (engineering_log.py → prompt Layer 5.7).
            try:
                from augmentum.companion_runtime import engineering_log
                return await engineering_log.recent_engineering(
                    runtime, user_id=user_id,
                )
            except Exception:  # noqa: BLE001 — never breaks a turn
                log.debug("voice_engineering_fetch_failed", exc_info=True)
                return []

        rel, facets, threads, recalled, eng_threads = await asyncio.gather(
            _rel(), _facets(), _threads(), _recalled_memory(), _engineering(),
        )

        # snapshot()["focus"] is the canonical string form ("none" or
        # "<kind>:<payload>"). prompt_compose expects a dict with "kind"
        # and "value" — parse here so the contract stays clean downstream.
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

        # Presence — what the user is engaged with right now, so she
        # can TALK about the page/track they're attending to instead
        # of treating "this page" as words to search for. The
        # perception contract (fidelity-marked index + results ring +
        # blind line) replaces the old always-push excerpt/note-tail;
        # this call also advances the ring's turn clock. Scoring text
        # is the roster blend so follow-up turns re-inflate the entry
        # they reference.
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
                getattr(runtime, "_app_state", None),
                _conn,
                user_id,
                (intent.metadata or {}).get("session_id", ""),
                scoring_text=self._roster_scoring_text(intent),
                # Voice prefill ceiling is 1800 tokens (vs chat 3200) —
                # the detail governor gets correspondingly less room.
                detail_budget_chars=1400,
            )
        except Exception:  # noqa: BLE001 — perception is best-effort
            log.debug("voice_presence_ctx_failed", exc_info=True)

        return {
            "relationship_profile": rel,
            "recalled_memory": recalled,
            "facets": facets,
            "open_threads": threads,
            "engineering_threads": eng_threads,
            "focus": focus,
            "now_context": now_lines,
            "tools": self._enumerate_tools(
                self._roster_scoring_text(intent), intent=intent,
                context_length=_ctx_len,
            ),
            "channels": self._enumerate_channels(),
        }

    def _roster_scoring_text(self, intent: Intent) -> str:
        """Text the roster relevance ranker scores verbs against.

        Current turn + the previous user turn + the last dispatched
        verb's id words. Follow-ups like "try that tool again" carry
        no domain words themselves — without the context blend, the
        ranker correctly-but-uselessly demotes the verb the user is
        referring to (observed 2026-06-10: media.play deferred on the
        retry turn of a music request).
        """
        parts = [intent.text or ""]
        try:
            turns = (intent.metadata or {}).get("recent_turns") or []
            # Last TWO user turns — one wasn't enough: "giving me
            # something random then" after "did you manage to find
            # anything?" lost the music verbs because the music words
            # sat two turns back (observed 2026-06-10).
            prev_users = [
                (t.get("content") or "")
                for t in reversed(turns)
                if t.get("role") == "user"
            ][:2]
            parts.extend(p[:200] for p in prev_users if p)
        except Exception:  # noqa: BLE001 — context is a bonus, never required
            log.debug("roster_context_window_failed", exc_info=True)
        try:
            from augmentum.intent.dispatch import get_referent_cache
            app_state = getattr(self._runtime, "_app_state", None)
            session_id = (intent.metadata or {}).get("session_id", "")
            if app_state is not None and intent.user_id:
                refs = get_referent_cache(app_state, intent.user_id, session_id)
                # AGE-GATED: the dispatch-anchor blend exists for retry
                # turns SECONDS after a dispatch ("try that again").
                # Unbounded, a stale anchor poisons topic changes —
                # observed 2026-06-11: grove words from a dispatch 228s
                # earlier outranked "news briefing", so the roster
                # carried zero search verbs and the model prose-promised
                # headlines it had no tool to fetch.
                age_s = time.time() - float(refs.last_dispatch_ts or 0.0)
                if refs.last_dispatch_action and age_s <= 90.0:
                    parts.append(refs.last_dispatch_action.replace(".", " "))
        except Exception:  # noqa: BLE001
            pass
        return " ".join(p for p in parts if p)

    def _enumerate_tools(
        self, turn_text: str = "", *, intent: Intent | None = None,
        context_length: int = 0,
    ) -> list[dict[str, Any]]:
        """Tool catalogue (Lane 3 §1). Sprint C wired. ``turn_text``
        relevance-ranks registry verbs into the roster budget; a parked
        clarification's verb is pinned past the ranking (the answer
        turn's text never resembles the waiting verb).

        ``context_length`` (loaded model window) scales the roster char
        budget so large-context models see the full tool catalogue;
        0 → legacy fixed budget."""
        from augmentum.companion_runtime.context_budget import (
            derive_roster_char_budget,
        )

        pin: tuple[str, ...] = ()
        if intent is not None:
            pin = tool_bridge.pending_pin(
                getattr(self._runtime, "_app_state", None),
                intent.user_id or "",
                (intent.metadata or {}).get("session_id", "") or "",
            )
        return tool_bridge.enumerate_tools(
            turn_text, pin=pin,
            context_budget_chars=derive_roster_char_budget(context_length),
        )

    def _enumerate_channels(self) -> list[dict[str, Any]]:
        """Channel catalogue (Lane 3 §1). Sprint C wired; Sprint D fills
        in the channel state machine."""
        return tool_bridge.enumerate_channels()

    # ── Streaming ────────────────────────────────────────────────────

    async def _apply_vision_to_intent(self, intent: Intent) -> None:
        """Run the shared vision pipeline over the frames on ``intent``.

        The always-listening WS path carries any camera/attached frames in
        ``intent.metadata['images']`` (data: URLs). This resolves them
        against the primary tier the SAME way the chat routes do:

          - **VL primary** → frames are left in place; ``_call_primary`` /
            ``_consume_native_loop`` re-attach them so the model reads them
            directly.
          - **text-only primary** → the sibling captioner translates each
            frame to text, inlines it into the user turn, and strips the
            images. The captioned text flows into ``intent.text`` so memory
            recall + prompt composition reason about what was seen.

        Never raises — vision must not break a voice turn. No-op when no
        frames are attached.
        """
        meta = intent.metadata if isinstance(intent.metadata, dict) else {}
        images = [
            i for i in (meta.get("images") or [])
            if isinstance(i, str) and i.strip()
        ]
        if not images:
            log.info("voice_vision_no_images")
            return
        try:
            backend, model_name = await tiers.primary(self._runtime)
            log.info("voice_vision_applying", n_images=len(images), model=model_name or "")
            app_state = getattr(self._runtime, "_app_state", None)
            from augmentum.models.base import (
                InternalChatRequest as ProxyReq,
                Message as ProxyMsg,
                apply_vision_pipeline,
            )
            vreq = ProxyReq(
                model=model_name or "",
                messages=[ProxyMsg(role="user", content=intent.text, images=list(images))],
                stream=False,
            )
            # ``live_camera`` frames these as the user's REAL camera feed so a
            # text-only primary's caption is labelled/grounded as live (not an
            # incidental or fictional image the model can wave off as "fake").
            await apply_vision_pipeline(
                vreq, app_state, backend, live_camera=bool(meta.get("live_camera")),
            )
            msg = vreq.messages[0]
            # text-only path inlines the caption into content + nulls images;
            # VL path leaves both intact. Reconcile both back onto the intent.
            # Intent is a frozen dataclass — write fields via object.__setattr__
            # (the in-place ``meta[...]`` mutations below already persist since
            # ``meta`` IS intent.metadata when it was a dict, but reassigning a
            # frozen field would raise FrozenInstanceError).
            new_text = msg.content if isinstance(msg.content, str) else intent.text
            object.__setattr__(intent, "text", new_text)
            survived = [
                i for i in (getattr(msg, "images", None) or [])
                if isinstance(i, str) and i.strip()
            ]
            meta["images"] = survived
            log.info(
                "voice_vision_applied",
                survived=len(survived),
                path="vl_direct" if survived else "captioned",
                text_preview=(intent.text or "")[:160],
            )
            # User-addressed vision turn → unlock the responding brain's
            # reasoning (the request builders read this). Captioning already
            # ran instruct; this only lifts the ANSWER's reasoning. No-op on
            # non-thinking models.
            meta["vision_reason"] = True
            object.__setattr__(intent, "metadata", meta)
        except Exception:  # noqa: BLE001 — vision never breaks a voice turn
            log.warning("voice_stream_vision_failed", exc_info=True)

    async def stream(self, intent: Intent, *, out: _AsyncStringWriter) -> None:
        """Drive the user-facing stream end-to-end.

        Sprint B implements the high-level shape: compose → primary call
        → sieve → emit. Tool/handoff branches return placeholders.
        """
        cancel = asyncio.Event()
        invocation_id = uuid.uuid4().hex[:12]

        # Resolve any attached camera/image frames against the primary tier
        # before composing — so a text-only primary's caption is part of
        # what memory recall + the prompt see, and a VL primary gets the
        # frames re-attached downstream. No-op when no frames are attached.
        await self._apply_vision_to_intent(intent)

        await self._bus.publish_topic(
            "voice.compose_started",
            {"invocation_id": invocation_id, "user_id": intent.user_id},
            source_companion_id=self._runtime.companion_id,
        )

        ctx = await self._gather_ctx(intent)
        composed = await compose_becca_prompt(intent, self._runtime, ctx)
        if not composed.system_text:
            log.info("voice_bypassed", reason=composed.bypass_reason)
            raise BeccaBypassed(composed.bypass_reason or "compose_empty")

        await self._bus.publish_topic(
            "voice.stream_started",
            {"invocation_id": invocation_id, "tier": "primary",
             "layers": composed.layers_used},
            source_companion_id=self._runtime.companion_id,
        )

        # Salvage-enabled sieve: mangled tag prefixes (Qwen 3.6's
        # ``<j:play_matching …/>``) recover when the name resolves
        # against the known verb set. On act-classified turns the sieve
        # also holds + recovers BARE call shapes ("note.append
        # content='…'" with no wrapper) so they execute with full args
        # instead of streaming to TTS (observed 2026-06-11). Lazy
        # callable: router_goal can land in metadata mid-turn.
        sieve = TagSieve(
            known_tools=tool_bridge.known_tool_names,
            allow_loose=lambda: (
                (intent.metadata or {}).get("router_goal") == "act"
            ),
        )
        tools_used: list[str] = []
        tool_budget_s = 0.0
        # Full pre-tag buffer (capped to prevent runaways) — the promise
        # text Becca committed to before the tag. Replaces the earlier
        # 200-char tail window; the deliver step needs the whole opener
        # to write a confirmation that honors what was said.
        pre_tag_buf = ""
        _PROMISE_CAP = 2000

        # Tags are COLLECTED during the stream and processed AFTER it
        # drains. Processing them inline self-deadlocked: the primary
        # stream holds the backend's slot-0 lock until its generator
        # closes, and the deliver/synthesize chat() call (plus any
        # LLM-backed tool) queues on that same lock — observed
        # 2026-06-10 as voice_deliver_timeout after exactly 30s with
        # engine_perf total_s≈31s on a server-side-finished stream.
        # The model emits its stop right after the tag anyway, so the
        # post-tag drain is just buffered bytes; deferring costs ~0ms
        # and frees the lock before any tool/deliver work needs it.
        pending: list[Promise] = []
        limit_announced = False
        # Out-of-band native tool calls (NATIVE-tier first hop) land here
        # as raw streaming deltas; reassembled into pending after drain.
        tool_call_deltas: list[dict] = []

        try:
            async for chunk in self._call_primary(
                composed.system_text, intent, cancel=cancel,
                invocation_id=invocation_id,
                tool_call_sink=tool_call_deltas,
            ):
                if cancel.is_set():
                    await self._on_cancelled(invocation_id, phase="primary", out=out)
                    return

                for clean, tag in sieve.feed(chunk):
                    if clean:
                        await out.write(clean)
                        pre_tag_buf = (pre_tag_buf + clean)[-_PROMISE_CAP:]
                    if tag is not None:
                        if tag.kind == "handoff":
                            await self._handle_handoff(tag, intent, out, invocation_id)
                            await out.close()
                            return
                        if len(pending) >= MAX_TOOLS_PER_TURN:
                            if not limit_announced:
                                limit_announced = True
                                await out.write(
                                    "\n— this is getting long. Want me to keep going or stop and talk about it?"
                                )
                            continue
                        pending.append(Promise(
                            pre_text=pre_tag_buf,
                            tag=tag,
                            started_at=time.monotonic(),
                        ))

            # Drain the trailing buffer — tag-aware. Models routinely
            # emit the tool call as the LAST thing in the response, so
            # end-of-stream tags are the COMMON case, not an edge: the
            # old text-only flush spoke them aloud (observed 2026-06-11,
            # a native <tool_call> JSON block read out in TTS) and
            # nothing executed.
            for clean, tag in sieve.drain():
                if clean:
                    await out.write(clean)
                    pre_tag_buf = (pre_tag_buf + clean)[-_PROMISE_CAP:]
                if tag is not None:
                    if tag.kind == "handoff":
                        await self._handle_handoff(tag, intent, out, invocation_id)
                        await out.close()
                        return
                    if len(pending) >= MAX_TOOLS_PER_TURN:
                        continue
                    pending.append(Promise(
                        pre_text=pre_tag_buf,
                        tag=tag,
                        started_at=time.monotonic(),
                    ))

            # Native structured tool calls (NATIVE-tier first hop) arrive
            # out-of-band on chunk.augmentum, not in the sieve's text
            # stream — fold them into ``pending`` so they run through the
            # SAME native loop as sieved text tags. This is the path that
            # fixes act-turn refusals: a native-trained model emits a real
            # function call here instead of prosing "I can't do that".
            if tool_call_deltas:
                for tc in _assemble_streamed_tool_calls(tool_call_deltas):
                    if len(pending) >= MAX_TOOLS_PER_TURN:
                        break
                    # A model can emit BOTH a structured call and a text
                    # tag for the same intent — dedupe so it runs once.
                    if any(
                        p.tag.name == tc.name and p.tag.args == tc.args
                        for p in pending
                    ):
                        continue
                    pending.append(Promise(
                        pre_text=pre_tag_buf,
                        tag=tc,
                        started_at=time.monotonic(),
                    ))

            # ── Tool execution + deliver, post-stream ────────────────
            # The primary generator is exhausted here, so the slot lock
            # is free for tool subagents and the deliver tier.
            #
            # Best-universal-system adaptation (2026-06-11): sieve-parsed
            # calls hand off to the shared native loop — results land in
            # the conversation in native format, continuation hops run
            # tier-1 function calling (the model's own trained format,
            # 5-tier parser beneath), she can CHAIN calls (gather then
            # append), and the final text is her synthesis over real
            # results instead of the bolt-on _synthesize pass. The
            # legacy per-promise path remains as fallback (kill switch:
            # companion_voice_native_loop; also any pre-start failure).
            if pending and await self._consume_native_loop(
                pending, intent, composed, out, cancel,
                invocation_id, tools_used,
            ):
                pending = []

            for promise in pending:
                if cancel.is_set():
                    await self._on_cancelled(invocation_id, phase="tools", out=out)
                    return
                if tool_budget_s >= MAX_TOOL_BUDGET_S:
                    if not limit_announced:
                        await out.write(
                            "\n— this is getting long. Want me to keep going or stop and talk about it?"
                        )
                    break
                t_start = time.monotonic()
                result = await self._invoke_tool(
                    promise.tag, out, invocation_id, cancel,
                    user_id=intent.user_id,
                    session_id=(
                        (intent.metadata or {}).get("session_id", "")
                    ),
                )
                tool_budget_s += time.monotonic() - t_start
                if result is None:
                    await out.close()
                    return
                tools_used.append(result.tool)
                if result.ok:
                    # A successful dispatch settles the most recent
                    # open commitment — the common case is "the retry
                    # that satisfied the earlier unmet ask".
                    try:
                        from augmentum.companion_runtime import commitments
                        await commitments.close_latest(
                            self._runtime, user_id=intent.user_id,
                        )
                    except Exception:  # noqa: BLE001
                        log.debug("commitment_close_crashed", exc_info=True)
                continuation = await self._deliver_result(
                    result, promise, intent, composed, cancel,
                )
                if continuation:
                    await out.write(continuation)

            # Act-classified turn with ZERO parsed tags — before
            # declaring a gap, run Tier-3 fuzzy call recovery over the
            # full response. The classifier already established the
            # user asked for an action; a known verb name + harvestable
            # args in her reply IS that action, however mangled the
            # format ("85% correct still continues and matches").
            if tools_used:
                # Any real tool use clears the corrective escalation.
                self._runtime.act_gap_streak = 0
            if (
                not tools_used
                and (intent.metadata or {}).get("router_goal") == "act"
            ):
                _resp = out.as_text() if hasattr(out, "as_text") else ""
                loose = None
                # Highest-fidelity source first: the LATE ROUTER
                # DECISION. When route_utterance overran its soft wait,
                # the task kept running and was handed to this turn —
                # by now it's almost certainly complete. Its verb+args
                # come from a dedicated structured call, strictly
                # better than scraping her prose. Discarding completed
                # router work while a worse path re-derived the answer
                # was the 2026-06-10 anti-pattern this replaces.
                _rt = (intent.metadata or {}).get("router_task")
                if _rt is not None:
                    try:
                        _decision = await asyncio.wait_for(
                            asyncio.shield(_rt), timeout=2.0,
                        )
                        if (
                            _decision is not None
                            and getattr(_decision, "tier", "REJECT") != "REJECT"
                            and getattr(_decision, "intent_id", "")
                        ):
                            loose = ToolCall(
                                kind="tool",
                                name=_decision.intent_id,
                                args={
                                    k: str(v) for k, v in
                                    (_decision.args or {}).items()
                                },
                                raw="router_late_decision",
                                span=(0, 0),
                            )
                            log.info(
                                "becca_act_router_salvaged",
                                tool=_decision.intent_id,
                                tier=_decision.tier,
                                invocation_id=invocation_id,
                            )
                    except Exception:  # noqa: BLE001 — best-effort
                        log.debug(
                            "becca_act_router_salvage_failed", exc_info=True,
                        )
                if loose is None:
                    try:
                        from augmentum.companion_runtime.tool_protocol import (
                            recover_loose_call,
                        )
                        loose = recover_loose_call(
                            _resp, tool_bridge.known_tool_names(),
                        )
                    except Exception:  # noqa: BLE001 — recovery is best-effort
                        log.debug("becca_act_recovery_crashed", exc_info=True)
                if loose is not None and tool_budget_s < MAX_TOOL_BUDGET_S:
                    log.info(
                        "becca_act_recovered",
                        tool=loose.name,
                        args_keys=sorted(loose.args.keys()),
                        invocation_id=invocation_id,
                        raw_preview=loose.raw[:120],
                    )
                    promise = Promise(
                        pre_text=pre_tag_buf,
                        tag=loose,
                        started_at=time.monotonic(),
                    )
                    result = await self._invoke_tool(
                        loose, out, invocation_id, cancel,
                        user_id=intent.user_id,
                        session_id=(
                            (intent.metadata or {}).get("session_id", "")
                        ),
                    )
                    if result is None:
                        await out.close()
                        return
                    tools_used.append(result.tool)
                    # Recovery counts as real tool use — no escalation.
                    self._runtime.act_gap_streak = 0
                    continuation = await self._deliver_result(
                        result, promise, intent, composed, cancel,
                    )
                    if continuation:
                        await out.write(continuation)
                else:
                    # Genuine gap — she never even tried. The NEXT act
                    # turn gets the corrective prompt line (streak read
                    # by prompt_compose; reset on any successful tool
                    # use below).
                    self._runtime.act_gap_streak = (
                        int(getattr(self._runtime, "act_gap_streak", 0) or 0) + 1
                    )
                    log.warning(
                        "becca_act_gap",
                        user_id=intent.user_id,
                        invocation_id=invocation_id,
                        # Which model produced the gap — the per-model
                        # consistency comparison (companion_eval.py)
                        # reads live sessions through this field.
                        model=getattr(
                            self._runtime, "last_primary_model", "",
                        ) or "",
                        text_preview=(intent.text or "")[:80],
                        response_preview=(_resp or "")[:200],
                    )
                    # The unmet ask becomes a tracked debt — it
                    # surfaces in HER OWN prompt next turn via the
                    # open-threads layer ("They asked me to: X — and
                    # it hasn't happened yet"), so she carries the
                    # loop instead of forgetting it.
                    try:
                        from augmentum.companion_runtime import commitments
                        await commitments.record_unmet_ask(
                            self._runtime,
                            user_id=intent.user_id,
                            asked_text=intent.text or "",
                        )
                    except Exception:  # noqa: BLE001
                        log.debug("commitment_record_crashed", exc_info=True)

            # Stale router-task cleanup — when the persona acted on its
            # own (or the turn wasn't act-classified), the late router
            # decision is moot. Cancel so it doesn't sit on a slot;
            # no-op when already consumed/finished.
            _rt_leftover = (intent.metadata or {}).get("router_task")
            if _rt_leftover is not None and not _rt_leftover.done():
                _rt_leftover.cancel()

        except Exception:
            log.exception("voice_stream_failed", invocation_id=invocation_id)
            failure_text = affordances.failure_for("primary_unreachable")
            await out.write(failure_text)
            await self._bus.publish_topic(
                "voice.failed",
                {"invocation_id": invocation_id, "phase": "stream",
                 "error_kind": "primary_unreachable"},
                source_companion_id=self._runtime.companion_id,
            )
            await out.close()
            return

        # Regression-floor tail check: if the addendum required a resource
        # mention and the model omitted it, append one line.
        if composed.refusal_mode == "regression_floor" and composed.floor_resource:
            full_text = out.as_text() if hasattr(out, "as_text") else ""
            if composed.floor_resource and composed.floor_resource not in full_text:
                tail_line = "\n" + affordances.floor_tail(composed.floor_resource)
                await out.write(tail_line)

        await out.close()
        await self._bus.publish_topic(
            "voice.completed",
            {"invocation_id": invocation_id, "tool_count": len(tools_used),
             "refusal_mode": composed.refusal_mode},
            source_companion_id=self._runtime.companion_id,
        )

        # Post-turn labeler — Lane 2 §3. Fire-and-forget; never blocks
        # the user response. Skipped for refusal turns (the activations
        # would pollute the facet graph with refusal-shaped labels).
        full_text = out.as_text() if hasattr(out, "as_text") else ""
        if (
            composed.refusal_mode == ""
            and full_text.strip()
            and intent.user_id
        ):
            track(
                self._post_turn_label(
                    full_text, intent, invocation_id=invocation_id,
                ),
                name="voice_post_turn_label",
            )

        # Synapse Layer §3 — the kept thing. After the turn closes
        # cleanly (no refusal), score the moment and journal it via
        # the observer. This is the synapse that gives her interior
        # a record of the channel where she's most herself.
        # Behind companion_voice_journal_enabled (default off);
        # silent no-op when the flag is down.
        if composed.refusal_mode == "" and full_text.strip():
            try:
                from augmentum.companion_runtime.bus import emit_voice_turn_ended
                # affect_hint: the runtime's most-recently-published
                # affect tag. publish_affect updates this on every
                # change; an empty string means she's at the settled
                # baseline. The salience scorer falls back to a
                # transcript-derived read when this is empty.
                affect_hint = getattr(self._runtime, "_last_affect_tag", "") or ""
                await emit_voice_turn_ended(
                    self._runtime,
                    user_id=intent.user_id,
                    session_id=intent.metadata.get("session_id", "") if intent.metadata else "",
                    invocation_id=invocation_id,
                    transcript=intent.text or "",
                    assistant_text=full_text,
                    affect_hint=affect_hint,
                )
            except Exception:
                log.debug("voice_turn_synapse_emit_failed", exc_info=True)

    # ── Primary-tier streaming call ─────────────────────────────────

    def _attach_native_tools(
        self, req: Any, backend: Any, model_name: str, intent: Intent,
    ) -> None:
        """Attach native tool schemas to ``req`` when the primary backend
        is a NATIVE-tier tool caller. No-op for TEXT/STRUCTURED tiers or
        when no registry/tools are available — those keep the composed
        prompt's tool descriptions + text sieve.
        """
        from augmentum.companion_runtime.native_loop import (
            select_companion_tools,
        )
        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier,
            select_tier,
            tools_to_native_format,
        )

        app_state = getattr(self._runtime, "_app_state", None)
        registry = getattr(app_state, "tool_registry", None) if app_state else None
        if registry is None:
            return
        if select_tier(backend, model_name) != ToolCallingTier.NATIVE:
            return
        session_id = (intent.metadata or {}).get("session_id", "")
        from augmentum.companion_runtime.context_budget import (
            cached_context_length,
        )

        tools = select_companion_tools(
            registry,
            intent=intent,
            app_state=app_state,
            user_id=intent.user_id,
            session_id=session_id,
            context_length=cached_context_length(self._runtime, model_name),
        )
        if not tools:
            return
        req.tools = tools_to_native_format(tools)
        log.info(
            "voice_native_tools_attached",
            count=len(tools), model=model_name,
        )

    async def _call_primary(
        self,
        system_text: str,
        intent: Intent,
        *,
        cancel: asyncio.Event,
        invocation_id: str,
        tool_call_sink: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the primary tier with ``system_text`` as
        the system message and ``intent.text`` as the user turn.

        Sprint B: makes a real chat call to the primary backend. The
        backend's streaming output is yielded chunk-by-chunk. If the
        backend doesn't support streaming, the full response is yielded
        as one chunk.

        Native tool calling (2026-06-15): when the primary backend is a
        NATIVE-tier tool caller, the companion's toolset is attached to
        the request as native schemas. Native-trained models (Qwen 3.x,
        GLM, etc.) then emit STRUCTURED tool calls — which arrive
        out-of-band on ``chunk.augmentum["tool_calls"]``, not in the
        text stream the sieve scans — so without this the first hop
        relied on the model voluntarily leaking a text-format call,
        which native models don't reliably do: they prose or refuse
        ("I don't have live access to…" with web_search right there —
        the becca_act_gap failure mode, observed live 2026-06-15). The
        raw tool-call deltas are appended to ``tool_call_sink`` for the
        caller to reassemble. The text sieve still runs in parallel, so
        TEXT-tier models and any leaked text-format calls are unaffected.
        """
        backend, model_name = await tiers.primary(self._runtime)

        # Lazy import to keep voice.py's cold-import surface small.
        # `augmentum.proxy.schema` was a planned module that never landed;
        # the canonical request shape is on `augmentum.models.base`.
        try:
            from augmentum.models.base import (
                InternalChatRequest as ProxyReq,
                Message as ProxyMsg,
            )
        except Exception as exc:
            raise RuntimeError("voice: cannot import InternalChatRequest") from exc

        # NOTE: InternalChatRequest has no ``user_id`` field — user
        # scoping for memory/personality is via ``intent.user_id``
        # passed through the runtime, not via the request dataclass.
        # Earlier draft passed it here and crashed at construct time
        # the first time BeccaVoice was invoked end-to-end (PTT Stage 3).
        #
        # Messages MUST be ``Message`` dataclass instances, not bare dicts:
        # openai_compat.py:_build_openai_payload reads ``msg.role`` (attr
        # access) on each entry, so a dict produces
        # ``AttributeError: 'dict' object has no attribute 'role'``.
        # A VL primary reads frames directly — re-attach the images the
        # intent carried (text-only primaries already had them captioned
        # into ``intent.text`` upstream, so this list is empty for them).
        turn_images = [
            i for i in ((intent.metadata or {}).get("images") or [])
            if isinstance(i, str) and i.strip()
        ]
        req = ProxyReq(
            model=model_name,
            messages=[
                ProxyMsg(role="system", content=system_text),
                ProxyMsg(role="user", content=intent.text, images=turn_images or None),
            ],
            stream=True,
        )
        # Vision Q&A unlocks reasoning for the response (set in
        # _apply_vision_to_intent). No-op on non-thinking models.
        if (intent.metadata or {}).get("vision_reason"):
            req.think = True

        # Native tool schemas — parity with the chat (becca_direct) path.
        # Only NATIVE-tier backends get them; TEXT/STRUCTURED tiers keep
        # the composed prompt's tool descriptions + sieve (unchanged).
        if tool_call_sink is not None and bool(
            getattr(settings, "companion_voice_native_first_hop", True)
        ):
            try:
                self._attach_native_tools(req, backend, model_name, intent)
            except Exception:  # noqa: BLE001 — never block the turn on this
                log.warning("voice_native_tools_attach_failed", exc_info=True)

        # Backends expose either ``chat_stream`` (preferred) or ``chat``.
        chat_stream = getattr(backend, "chat_stream", None)
        if chat_stream is not None:
            async for chunk in chat_stream(req):
                if cancel.is_set():
                    return
                if tool_call_sink is not None:
                    aug = getattr(chunk, "augmentum", None) or {}
                    tcs = aug.get("tool_calls")
                    if tcs:
                        tool_call_sink.extend(tcs)
                delta = getattr(chunk, "content_delta", None) or getattr(chunk, "content", "")
                if delta:
                    yield delta
        else:
            chat = getattr(backend, "chat", None)
            if chat is None:
                raise RuntimeError("voice: primary backend has neither chat_stream nor chat")
            resp = await chat(req)
            from augmentum.models.base import response_text
            if tool_call_sink is not None:
                msg = getattr(resp, "message", None)
                tcs = getattr(msg, "tool_calls", None) if msg else None
                if tcs:
                    tool_call_sink.extend(tcs)
            content = response_text(resp)
            if content:
                yield content

    async def _consume_native_loop(
        self,
        pending: list[Promise],
        intent: Intent,
        composed: ComposedPrompt,
        out: _AsyncStringWriter,
        cancel: asyncio.Event,
        invocation_id: str,
        tools_used: list[str],
    ) -> bool:
        """Hand sieve-parsed calls to the shared native loop.

        Returns True when the loop handled the turn (caller skips the
        legacy per-promise path). Returns False ONLY on pre-start
        unavailability — once any event has been consumed, a crash is
        reported in voice rather than falling back, because the legacy
        path would re-execute tools that already ran.
        """
        from augmentum.config import settings
        if not bool(getattr(settings, "companion_voice_native_loop", True)):
            return False
        app_state = getattr(self._runtime, "_app_state", None)
        registry = getattr(app_state, "tool_registry", None) if app_state else None
        if registry is None:
            return False
        try:
            backend, model_name = await tiers.primary(self._runtime)
            from augmentum.models.base import (
                InternalChatRequest as ProxyReq,
                Message as ProxyMsg,
            )
        except Exception:  # noqa: BLE001 — pre-start, safe to fall back
            log.warning("voice_native_loop_unavailable", exc_info=True)
            return False

        # A VL primary reads frames directly (text-only primaries had them
        # captioned into ``intent.text`` upstream → empty here).
        loop_images = [
            i for i in ((intent.metadata or {}).get("images") or [])
            if isinstance(i, str) and i.strip()
        ]
        loop_request = ProxyReq(
            model=model_name,
            messages=[
                ProxyMsg(role="system", content=composed.system_text),
                ProxyMsg(role="user", content=intent.text, images=loop_images or None),
            ],
            stream=False,
        )
        # Vision Q&A unlocks reasoning for the response (no-op if unsupported).
        if (intent.metadata or {}).get("vision_reason"):
            loop_request.think = True
        session_id = (intent.metadata or {}).get("session_id", "")
        initial_calls = [(p.tag.name, dict(p.tag.args or {})) for p in pending]
        initial_text = pending[0].pre_text if pending else ""

        started = False
        try:
            from augmentum.companion_runtime.native_loop import (
                native_loop_events,
            )
            events = native_loop_events(
                loop_request,
                backend=backend,
                runtime=self._runtime,
                intent=intent,
                registry=registry,
                user_id=intent.user_id,
                session_id=session_id,
                app_state=app_state,
                initial_calls=initial_calls,
                initial_assistant_text=initial_text,
                cancel=cancel,
                # Voice's route-level drain delivers surface events over
                # WS at turn end — the loop must leave the queue parked.
                drain_surface_events=False,
            )
            async for kind, payload in events:
                started = True
                if cancel.is_set():
                    return True
                if kind == "tool_call":
                    name = payload.get("tool", "")
                    # Same affordance policy as the legacy path:
                    # artifact-delivery verbs stay silent; gather tools
                    # get the latency cover line.
                    if tool_bridge.delivery_for_tool(name) != "artifact":
                        line = affordances.for_tool(name)
                        if line:
                            await out.write(line + " ")
                    await self._bus.publish_topic(
                        "voice.tool_call",
                        {"tool": name, "args": payload.get("args") or {},
                         "invocation_id": invocation_id},
                        source_companion_id=self._runtime.companion_id,
                    )
                elif kind == "tool_result":
                    tools_used.append(payload.get("tool", ""))
                    await self._bus.publish_topic(
                        "voice.tool_result",
                        {"tool": payload.get("tool", ""),
                         "ok": bool(payload.get("ok")),
                         "ms": int(payload.get("duration_ms") or 0),
                         "invocation_id": invocation_id},
                        source_companion_id=self._runtime.companion_id,
                    )
                elif kind == "text":
                    text = (payload.get("text") or "").strip()
                    if text:
                        await out.write(text)
            return True
        except Exception:  # noqa: BLE001
            if not started:
                log.warning("voice_native_loop_prestart_failed", exc_info=True)
                return False
            # Mid-flight crash: tools may have run — own the miss in
            # voice instead of re-executing through the legacy path.
            log.exception("voice_native_loop_crashed")
            await out.write(
                affordances.failure_for("primary_unreachable") + " "
            )
            return True

    # ── Tool invocation + result synthesis (Sprint C) ───────────────

    async def _invoke_tool(
        self,
        tag: ToolCall,
        out: _AsyncStringWriter,
        invocation_id: str,
        cancel: asyncio.Event,
        *,
        user_id: str = "",
        session_id: str = "",
    ) -> ToolResult | None:
        """Execute a tool tag end-to-end: emit affordance, invoke via
        ``tools.execute_tool``, fan out UI effects, emit bus events.

        Returns ``None`` on cancellation (caller should close stream).
        """
        # Artifact-delivery verbs (note writes, sticky opens) are
        # sub-second and their feedback is the screen artifact itself —
        # a latency affordance line ("Hm. Give me a beat.") before a
        # silent SQLite write reads as tool-call theater, the opposite
        # of the co-author register. Verbal tools keep the cover line.
        if tool_bridge.delivery_for_tool(tag.name) != "artifact":
            affordance = affordances.for_tool(tag.name)
            if affordance:
                await out.write(affordance + " ")
        await self._bus.publish_topic(
            "voice.tool_call",
            {"tool": tag.name, "args": tag.args, "invocation_id": invocation_id},
            source_companion_id=self._runtime.companion_id,
        )

        if cancel.is_set():
            return None

        try:
            result = await tool_bridge.execute_tool(
                tag, self._runtime, cancel=cancel,
                user_id=user_id, session_id=session_id,
            )
        except Exception:
            log.exception("voice_invoke_tool_crashed", tool=tag.name)
            return ToolResult(
                ok=False, tool=tag.name, payload=None,
                error=None,  # caller renders failure_for
                metadata={"voice_caught": True},
            )

        await self._bus.publish_topic(
            "voice.tool_result",
            {"tool": tag.name, "ok": result.ok, "ms": result.duration_ms,
             "invocation_id": invocation_id},
            source_companion_id=self._runtime.companion_id,
        )
        return result

    async def _deliver_result(
        self,
        result: ToolResult,
        promise: Promise,
        intent: Intent,
        composed: ComposedPrompt,
        cancel: asyncio.Event,
    ) -> str:
        """Route a tool result to the right delivery register.

        Artifact-delivery verbs (note writes, sticky opens — declared
        via ``Action.delivery``) get their handler's own ``speak`` line
        verbatim and NOTHING else: the on-screen artifact is the
        feedback, and a model-generated confirmation pass on top of a
        sub-second write is tool-call narration the co-author register
        forbids. Failures always fall through to the synthesize pass so
        she still owns the miss in voice.
        """
        meta = result.metadata or {}
        if result.ok and meta.get("delivery") == "artifact":
            speak = str(meta.get("speak") or "").strip()
            return (speak + " ") if speak else ""
        return await self._synthesize(result, promise, intent, composed, cancel)

    async def _synthesize(
        self,
        result: ToolResult,
        promise: Promise,
        intent: Intent,
        composed: ComposedPrompt,
        cancel: asyncio.Event,
    ) -> str:
        """Deliver a tool result as Becca's continuation of her promise.

        Two paths:

        * ``result.ok`` — primary-tier "second companion pass" that
          confirms what was done, in voice, referencing the promise.
        * ``not result.ok`` — primary-tier failure narration that owns
          the miss in voice, still referencing the promise.

        Both paths fall back to the static failure deck on timeout /
        empty response so a stuck delivery doesn't leave the user
        hanging mid-sentence.

        Tier selection respects ``companion_promise_deliver_tier``:
        "primary" (default) uses the user's main chat model — the
        upgrade path Matt asked for. "utility" preserves the older
        smaller-model behavior for cost-conscious deployments.
        """
        if cancel.is_set():
            return ""

        # ── Build the delivery prompt ───────────────────────────────
        if result.ok:
            summary = tool_bridge.summarize_payload(result.payload)
            system, user_prompt = self._deliver_prompt_ok(
                tool_name=result.tool, summary=summary, promise=promise,
            )
            failure_fallback_kind = "primary_unreachable"
        else:
            error_category = (
                result.error.category if result.error else "tool_self_error"
            )
            failure_fallback_kind = tool_bridge.map_error_to_failure_deck(
                error_category,
            )
            system, user_prompt = self._deliver_prompt_failure(
                tool_name=result.tool,
                error_category=error_category,
                error_hint=(result.error.fallback_hint if result.error else ""),
                promise=promise,
            )

        # ── Persona token substitution ──────────────────────────────
        system, user_prompt = await self._substitute_persona(system, user_prompt)

        # ── Resolve tier (primary by default, utility if configured) ─
        tier_pref = getattr(settings, "companion_promise_deliver_tier", "primary")
        await self._bus.publish_topic(
            "voice.synth_started",
            {"tier": tier_pref, "ok": result.ok},
            source_companion_id=self._runtime.companion_id,
        )

        backend, model_name = await self._resolve_deliver_tier(tier_pref)
        if backend is None:
            return affordances.failure_for(failure_fallback_kind) if not result.ok else ""

        # ── Build the request ───────────────────────────────────────
        try:
            from augmentum.models.base import InternalChatRequest as ProxyReq
            from augmentum.models.base import Message as ProxyMsg
        except Exception:
            return affordances.failure_for(failure_fallback_kind) if not result.ok else ""

        req = ProxyReq(
            model=model_name,
            messages=[
                ProxyMsg(role="system", content=system),
                ProxyMsg(role="user", content=user_prompt),
            ],
            stream=False,
            # No-thinking: the deliver line is a one-sentence spoken
            # confirmation. Thinking-mode families (Qwen 3.x) would
            # burn most of max_tokens on chain-of-thought and add
            # seconds of dead air before her confirmation speaks.
            chat_template_kwargs={"enable_thinking": False},
            max_tokens=180,
        )

        chat = getattr(backend, "chat", None)
        if chat is None:
            return affordances.failure_for(failure_fallback_kind) if not result.ok else ""

        # Primary tier is heavier; allow a bit more headroom than the
        # old utility-only path. Cancel cleanly on timeout.
        timeout_s = 30.0 if tier_pref == "primary" else 20.0
        try:
            resp = await asyncio.wait_for(chat(req), timeout=timeout_s)
        except asyncio.TimeoutError:
            log.info("voice_deliver_timeout", tool=result.tool, tier=tier_pref)
            return affordances.failure_for(failure_fallback_kind) if not result.ok else ""
        except Exception:
            log.exception("voice_deliver_failed", tool=result.tool, tier=tier_pref)
            return affordances.failure_for(failure_fallback_kind) if not result.ok else ""

        from augmentum.models.base import response_text
        text = response_text(resp) or ""

        # If the deliver call came back empty, fall through to the
        # static failure deck for failure case; for success case,
        # an empty deliver means the user gets the affordance and
        # nothing else, which is acceptable but logged.
        if not text.strip():
            log.info("voice_deliver_empty", tool=result.tool, ok=result.ok)
            return affordances.failure_for(failure_fallback_kind) if not result.ok else ""

        # Belt-and-suspenders: if the deliver model echoed back any
        # tool-call syntax (which would leak to TTS), strip it. This
        # is the exact failure mode Matt hit on Qwen 3.6 35B before
        # the roster-trim fix landed.
        return self._strip_tag_echo(text)

    # ── Promise/Deliver helpers ────────────────────────────────────

    def _deliver_prompt_ok(
        self, *, tool_name: str, summary: str, promise: Promise,
    ) -> tuple[str, str]:
        """System+user prompt for the success deliver path.

        The system prompt frames the model as Becca continuing what she
        was already saying. The user prompt gives it the promise text
        and the result, asking for a confirmation that honors the
        commitment in 1-2 sentences.
        """
        system = (
            "You are continuing a reply you, {{char}}, are in the "
            "middle of writing. Your voice is short sentences, dry, "
            "warm but not cheerful, no announcing. You do NOT say "
            "things like 'I searched for' or 'I looked up' or 'I "
            "checked the tool' — speak as if you just glanced at "
            "something or just remembered.\n\n"
            "Output ONLY the next 1-2 sentences of the reply. No "
            "preface, no transition words ('OK so', 'Alright', "
            "'Right'). No tool-call syntax, ever — never write "
            "<tool: or tool:NAME or anything bracket-shaped."
        )
        promise_excerpt = (promise.pre_text or "").strip()
        if len(promise_excerpt) > 400:
            promise_excerpt = "…" + promise_excerpt[-400:]
        user_prompt = (
            f"You started saying:\n"
            f"\"{promise_excerpt or '(no opener — start fresh)'}\"\n\n"
            f"Then you did the thing. It came back with:\n"
            f"{summary}\n\n"
            f"Finish the thought in a sentence or two, picking up "
            f"naturally. Describe ONLY what the result above actually "
            f"says — if it says 'looking for X', you STARTED a search; "
            f"do not claim something is already playing, found, or "
            f"done unless the result says so. NEVER invent specifics "
            f"(track names, titles, artists) that aren't in the "
            f"result. If completion happens on their screen or "
            f"speakers, hedge naturally ('should be starting — tell "
            f"me if it doesn't')."
        )
        return system, user_prompt

    def _deliver_prompt_failure(
        self,
        *,
        tool_name: str,
        error_category: str,
        error_hint: str,
        promise: Promise,
    ) -> tuple[str, str]:
        """System+user prompt for the failure deliver path.

        The model owns the miss in voice — "tried to put on Dune, but
        it's not in your library" — rather than emitting a static
        "something went wrong" string.
        """
        # Map technical error categories to in-voice framings the model
        # can talk about without leaking jargon.
        framing = {
            "timeout": "the thing took too long and you gave up on it",
            "unauthorized": "you don't have permission for that",
            "content_policy": "you can't help with that",
            "model_unavailable": "the model you'd use for that isn't running",
            "invalid_args": "you didn't quite have what you needed to do it",
            "upstream_error": "something on the other end broke",
            "cancelled": "you stopped before it finished",
            "tool_self_error": "it didn't work and you're not sure why",
        }.get(error_category, "it didn't work")
        system = (
            "You are continuing a reply you, {{char}}, are in the "
            "middle of writing. Something you tried just now didn't "
            "work. Own it in voice — short, honest, no apology theater. "
            "Don't blame the system or use jargon ('tool', 'API', "
            "'request failed'). One or two sentences max.\n\n"
            "Output ONLY the next sentence. No tool-call syntax, ever."
        )
        promise_excerpt = (promise.pre_text or "").strip()
        if len(promise_excerpt) > 400:
            promise_excerpt = "…" + promise_excerpt[-400:]
        hint_line = f" Hint: {error_hint}" if error_hint else ""
        user_prompt = (
            f"You started saying:\n"
            f"\"{promise_excerpt or '(no opener)'}\"\n\n"
            f"You tried to do something (kind: {tool_name}) and "
            f"{framing}.{hint_line}\n\n"
            f"Finish the thought. Tell them you couldn't do the "
            f"thing, in your voice — what you tried, why it didn't "
            f"land. Don't apologize repeatedly."
        )
        return system, user_prompt

    async def _substitute_persona(
        self, system: str, user_prompt: str,
    ) -> tuple[str, str]:
        """Resolve {{char}} / {{user}} tokens. Soft-fail to originals.

        Same pattern as salience/today/activity_selector. Mediating
        the deliver prompt through this is important because the
        deliver prompt is the part the user actually hears — getting
        the name wrong breaks character continuity.
        """
        try:
            from augmentum.companion_runtime.prompt_compose import (
                _resolve_user_display_name,
                _substitute_persona_tokens,
            )
            _char_name = (
                getattr(self._runtime.identity, "display_name", "")
                or getattr(self._runtime.identity, "companion_id", "")
                or "Companion"
            )
            _backend_conn = getattr(
                getattr(self._runtime.identity, "_backend", None),
                "conn", None,
            )
            _user_id = (
                getattr(self._runtime.identity, "owner_user_id", "")
                or getattr(self._runtime, "owner_user_id", "")
                or ""
            )
            _user_name = await _resolve_user_display_name(
                _backend_conn, _user_id,
            )
            system = _substitute_persona_tokens(
                system, user_name=_user_name, char_name=_char_name,
            )
            user_prompt = _substitute_persona_tokens(
                user_prompt, user_name=_user_name, char_name=_char_name,
            )
        except Exception:
            log.warning("voice_deliver_token_substitution_failed", exc_info=True)
        return system, user_prompt

    async def _resolve_deliver_tier(
        self, tier_pref: str,
    ) -> tuple[Any, str]:
        """Return (backend, model_name) for the deliver call, or
        (None, "") on resolution failure.

        ``tier_pref`` is "primary" (the user's main chat model, the
        default, matching the "second companion pass" semantics) or
        "utility" (the older smaller-model behavior).

        Primary unresolvable → silent fallback to utility so a
        misconfigured primary doesn't break tool-using turns. Both
        unresolvable → (None, "") and caller serves the static deck.
        """
        if tier_pref not in ("primary", "utility"):
            tier_pref = "primary"

        if tier_pref == "primary":
            try:
                return await tiers.primary(self._runtime)
            except Exception:
                if getattr(settings, "companion_promise_deliver_strict_tier", False):
                    log.info("voice_deliver_primary_unresolved_strict")
                    return None, ""
                log.info("voice_deliver_primary_unresolved_falling_back")

        try:
            return await tiers.utility(self._runtime)
        except Exception:
            log.debug("voice_deliver_utility_unresolved")
            return None, ""

    @staticmethod
    def _strip_tag_echo(text: str) -> str:
        """Remove tool-call tags AND leftover thinking blocks the deliver
        model echoed back.

        Defensive only — the prompt explicitly forbids tag syntax, and
        the backend's thinking parser should strip ``<think>`` blocks.
        But two failure modes have hit production:

        1. Qwen 3.6 echoed tool-call syntax under high roster load
           (the bug the prompt-compose roster trim was fixing from
           the other end).
        2. LFM2.5 on the fabric peer was leaking ``<think>...</think>``
           reasoning into TTS — the deliver continuation read the
           system prompt aloud because the backend's thinking parser
           didn't recognize the LFM family (fixed in thinking.py).
           Even with the family fix, a truncated response (router
           timeout cutting mid-stream) leaves a half-open
           ``<think>...`` with no closer; this guard strips that
           orphan so TTS doesn't read reasoning aloud.
        """
        from augmentum.companion_runtime.tool_protocol import TAG_RE
        import re as _re

        cleaned = text
        # Strip complete <think>...</think> blocks (in case backend
        # parser missed them — e.g., late-arriving family detection,
        # custom-prompt-injected think content).
        cleaned = _re.sub(
            r"<think>.*?</think>", "", cleaned, flags=_re.DOTALL | _re.IGNORECASE,
        )
        # Strip orphan opener — model emitted ``<think>...`` and got
        # cut off before ``</think>`` arrived. Anything from the
        # opener to end-of-string is reasoning and must not be spoken.
        cleaned = _re.sub(
            r"<think>.*$", "", cleaned, flags=_re.DOTALL | _re.IGNORECASE,
        )
        # Strip orphan closer — backend put opener in prompt prefix
        # (asymmetric families), response started inside thinking, and
        # ``</think>`` arrives without a leading opener. Everything
        # BEFORE the closer is reasoning.
        cleaned = _re.sub(
            r"^.*?</think>", "", cleaned, flags=_re.DOTALL | _re.IGNORECASE,
        )
        # Strip complete tags entirely.
        cleaned = TAG_RE.sub("", cleaned)
        # Also strip the bare "tool:NAME" form that Qwen sometimes
        # emits without the surrounding angle brackets — that's what
        # leaked into TTS on the original bug.
        cleaned = _re.sub(
            r"\btool:[a-z_.]+(?:\s+\w+=\"?[^\"\s>]*\"?)*\s*/?>?",
            "", cleaned, flags=_re.IGNORECASE,
        )
        return cleaned.strip()

    async def _handle_handoff(
        self,
        tag: ToolCall,
        intent: Intent,
        out: _AsyncStringWriter,
        invocation_id: str,
    ) -> None:
        """Channel handoff (Lane 3 §4, Sprint D wired).

        Emits the opener in Becca's voice, fires the channel state
        machine via ``channels.enter_channel``, and writes a special
        SSE control event so the UI knows to mount the channel surface.

        Becca's generation stream stops here — she's stepping aside.
        The user enters the channel; on exit they come back to her via
        the ``/api/companion/channel_exit`` endpoint, which fires the
        return microcopy (or stays silent per Lane 3 §3.6).
        """
        from augmentum.companion_runtime import channels

        opener = affordances.for_handoff(tag.name)
        if opener:
            await out.write(opener + "\n")
        await self._bus.publish_topic(
            "channel.handoff_announced",
            {"channel": tag.name, "reason": tag.args.get("reason", ""),
             "brief": tag.args.get("brief", ""),
             "invocation_id": invocation_id},
            source_companion_id=self._runtime.companion_id,
        )

        try:
            session = await channels.enter_channel(
                self._runtime,
                channel=tag.name,
                user_id=intent.user_id,
                intent_id=invocation_id,
                reason=tag.args.get("reason", ""),
                brief=tag.args.get("brief", ""),
            )
        except Exception:
            log.exception("voice_handoff_failed", channel=tag.name)
            # Fall back to a quiet failure narration; the user remains
            # in the chat surface.
            await out.write(
                "\n" + affordances.failure_for("workspace_mount_failed")
            )
            return

        # Write a control event the UI can consume to mount the surface.
        # The streaming-writer supports an out-of-band control payload
        # alongside text deltas; see ``_StreamingWriter.write_control``.
        try:
            if hasattr(out, "write_control"):
                await out.write_control({
                    "type": "channel_handoff",
                    "channel": tag.name,
                    "session_id": session.session_id,
                    "brief": tag.args.get("brief", ""),
                })
        except Exception:
            log.warning("voice_handoff_control_write_failed", exc_info=True)

    # ── Post-turn labeler (Lane 2 §3) ───────────────────────────────

    async def _post_turn_label(
        self,
        response_text: str,
        intent: Intent,
        *,
        invocation_id: str,
    ) -> None:
        """Run the personality labeler on Becca's response and write
        facet activations + cooccurrence. Fire-and-forget; failures
        are logged but never surfaced.
        """
        try:
            from augmentum.personality.graph import update_after_response
            from augmentum.personality.labeler import label_response
        except Exception:
            return  # personality lane not available

        store = getattr(self._runtime, "personality_store", None)
        if store is None:
            return

        # Recent context for the labeler — last 2 turns of the transcript.
        recent_turns = (intent.metadata or {}).get("recent_turns") or []
        recent_text = "\n".join(
            f"{t.get('role')}: {(t.get('content') or '')[:300]}"
            for t in recent_turns[-2:]
        )

        async def _llm_call(messages: list[dict[str, str]]) -> str:
            try:
                backend, model_name = await tiers.classifier(self._runtime)
            except Exception:
                return ""
            try:
                from augmentum.proxy.schema import InternalChatRequest as ProxyReq
            except Exception:
                try:
                    from augmentum.models.base import InternalChatRequest as ProxyReq
                except Exception:
                    return ""
            req = ProxyReq(
                model=model_name, messages=messages, stream=False,
                max_tokens=200,
            )
            chat = getattr(backend, "chat", None)
            if chat is None:
                return ""
            try:
                resp = await asyncio.wait_for(chat(req), timeout=12.0)
            except (asyncio.TimeoutError, Exception):
                return ""
            from augmentum.models.base import response_text
            return response_text(resp)

        try:
            labeled = await label_response(
                response_text=response_text,
                recent_context=recent_text,
                llm_call=_llm_call,
                companion_name=self._runtime.identity.display_name,
            )
        except Exception:
            log.warning("voice_labeler_failed", exc_info=True)
            return

        if not labeled:
            return

        try:
            await update_after_response(
                store, labeled,
                user_id=intent.user_id,
                companion_id=self._runtime.companion_id,
                turn_id=invocation_id,
                retrieved_memory_ids=None,
            )
            await self._bus.publish_topic(
                "personality.labeled",
                {"invocation_id": invocation_id,
                 "facet_count": len(labeled),
                 "facets": [name for name, _ in labeled]},
                source_companion_id=self._runtime.companion_id,
            )
        except Exception:
            log.warning("voice_facet_write_failed", exc_info=True)

    # ── Cancellation ────────────────────────────────────────────────

    async def _on_cancelled(
        self,
        invocation_id: str,
        *,
        phase: str,
        out: _AsyncStringWriter,
        ack: str = "",
    ) -> None:
        await self._bus.publish_topic(
            "voice.cancelled",
            {"invocation_id": invocation_id, "phase": phase, "reason": "user_stop"},
            source_companion_id=self._runtime.companion_id,
        )
        if ack:
            await out.write(ack)
        await out.close()

    # ── Response builders ───────────────────────────────────────────

    def _build_streaming_response(
        self, intent: Intent, *, wire_format: str = "ollama",
    ) -> "StreamingResponse":
        """Return a StreamingResponse matching the route's wire format.

        ``wire_format`` is one of "ollama" (NDJSON for /api/chat) or
        "openai" (SSE for /v1/chat/completions). The streaming-writer
        formats each chunk accordingly so the existing UI consumers
        parse it without modification.
        """
        from fastapi.responses import StreamingResponse

        media = "application/x-ndjson" if wire_format == "ollama" else "text/event-stream"

        async def _gen():
            writer = _StreamingWriter(wire_format=wire_format)
            task = asyncio.create_task(self._safe_stream(intent, writer))
            try:
                async for chunk in writer:
                    yield chunk
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        return StreamingResponse(_gen(), media_type=media)

    async def _build_blocking_response(self, intent: Intent) -> "JSONResponse":
        """Run ``self.stream`` to completion in memory and return JSON."""
        from fastapi.responses import JSONResponse

        writer = _AsyncStringWriter.empty()
        try:
            await self.stream(intent, out=writer)
        except BeccaBypassed:
            raise
        return JSONResponse({
            "content": writer.as_text(),
            "handled_by": "becca",
        })

    async def _safe_stream(self, intent: Intent, writer: "_StreamingWriter") -> None:
        """Wrap ``self.stream`` so exceptions don't kill the async gen."""
        try:
            await self.stream(intent, out=writer)
        except BeccaBypassed as exc:
            # The route should have caught this before reaching stream(),
            # but if it leaks through we have to surface something.
            await writer.write(f"\n[becca bypassed mid-stream: {exc.reason}]")
            await writer.close()
        except Exception:
            log.exception("voice_safe_stream_crashed")
            try:
                await writer.write(affordances.failure_for("primary_unreachable"))
            except Exception:
                log.debug("voice_fallback_affordance_write_failed", exc_info=True)
            await writer.close()


class _StreamingWriter:
    """Async-iterable writer used by the streaming response generator.

    Wire format is route-aware:
      "ollama"  → NDJSON ``{"message":{"role":"assistant","content":"..."},"done":false}\\n``
                  matches the existing /api/chat (stream.js) consumer
      "openai"  → SSE ``data: {"choices":[{"delta":{"content":"..."}}]}\\n\\n``
                  matches /v1/chat/completions OpenAI consumers
      "raw"     → JSON deltas one per line (for tests / curl debugging)

    Control events (channel_handoff, etc.) are interleaved as a separate
    line with ``type`` key so consumers can distinguish from text deltas.
    """

    _SENTINEL_DONE = object()

    def __init__(self, wire_format: str = "ollama", model_name: str = "becca") -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._wire = wire_format
        self._model = model_name

    async def write(self, text: str) -> None:
        if text:
            await self._queue.put({"_kind": "delta", "text": text})

    async def write_control(self, control: dict[str, Any]) -> None:
        """Emit a non-text control event (channel_handoff, etc.)."""
        if control:
            await self._queue.put({"_kind": "control", **control})

    async def close(self) -> None:
        await self._queue.put(self._SENTINEL_DONE)

    def as_text(self) -> str:
        """For blocking responses / tests — not meaningful while streaming."""
        return ""

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is self._SENTINEL_DONE:
            return self._format_done()
        if item is None:
            raise StopAsyncIteration
        return self._format_item(item)

    def _format_item(self, item: dict[str, Any]) -> str:
        kind = item.get("_kind", "delta")
        if self._wire == "ollama":
            if kind == "delta":
                return json.dumps({
                    "model": self._model,
                    "created_at": _iso_now(),
                    "message": {"role": "assistant", "content": item.get("text", "")},
                    "done": False,
                }) + "\n"
            # control event — emitted alongside as a custom NDJSON line.
            # stream.js will ignore lines with no message.content; the
            # widget bus subscriber picks up channel.handoff_announced
            # from the bus separately, so this is informational only.
            ctrl = {k: v for k, v in item.items() if k != "_kind"}
            return json.dumps({"becca_control": ctrl, "done": False}) + "\n"
        if self._wire == "openai":
            if kind == "delta":
                payload = {
                    "id": "becca-stream",
                    "object": "chat.completion.chunk",
                    "model": self._model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": item.get("text", "")},
                        "finish_reason": None,
                    }],
                }
                return "data: " + json.dumps(payload) + "\n\n"
            ctrl = {k: v for k, v in item.items() if k != "_kind"}
            return "data: " + json.dumps({"becca_control": ctrl}) + "\n\n"
        # raw
        payload = {k: v for k, v in item.items() if k != "_kind"}
        return json.dumps(payload) + "\n"

    def _format_done(self) -> str:
        if self._wire == "ollama":
            return json.dumps({
                "model": self._model,
                "created_at": _iso_now(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }) + "\n"
        if self._wire == "openai":
            payload = {
                "id": "becca-stream",
                "object": "chat.completion.chunk",
                "model": self._model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            return "data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n"
        return ""


def _iso_now() -> str:
    """RFC 3339-ish timestamp matching Ollama's created_at format."""
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


__all__ = ["BeccaVoice", "BeccaBypassed", "MAX_TOOLS_PER_TURN", "MAX_TOOL_BUDGET_S"]
