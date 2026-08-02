"""Shared native function-calling loop — the companion's hands.

Extracted from ``becca_direct/handler.py::_stream_native_loop``
(Companion Agency MVP #1) so BOTH delivery surfaces drive one loop:

  * **Chat** (``becca_direct``) wraps the events into
    ``InternalStreamChunk``s — becca_tool_call/result cards + prose.
  * **Voice** (``companion_runtime/voice.py`` — headless-agency P1,
    next phase) consumes the same events for TTS: iteration text is
    the spoken promise, final text is the verbal answer built from
    gathered results.

The loop borrows passthrough's primitives wholesale: tier injection
(``_inject_tool_schemas`` — native/structured/text), the 5-tier
tolerant parser (``_parse_tool_calls`` — native → JSON → XML → ReAct
→ fuzzy: the "85% correct still matches" principle), and the executor
(``_execute_and_append`` — context injection + result messages). This
module owns tool SELECTION (relevance roster ∩ chat ToolRegistry +
core capability tools) and the event protocol.

Headless-agency delivery contract (spec 2026-06-10): data tools
return results INTO the loop; her trail records where she went so
"take me there" can jump the user to her position.

Event protocol (async generator yields ``(kind, payload)``):
  ("tool_call",   {"tool", "args"})
  ("tool_result", {"tool", "ok", "payload_summary", "duration_ms"})
  ("text",        {"text"})                      # final prose
  ("ui_effects",  {"effects": [...]})            # drained surface events
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Core capability tools she always carries, beyond her verb roster.
# web_fetch is the headless page eye: it runs the browse_fetch
# dispatch chain (Wikipedia REST / Reddit / PDF / trafilatura)
# server-side — no panel opens on the user's screen. Already existed
# in the chat registry; headless agency is wiring, not building.
CORE_TOOL_NAMES = [
    "web_search", "web_fetch", "image_generation",
    # research = the iterative, retry-until-found web primitive. Carried
    # everywhere so scheduled requests / chat / voice all get the same
    # "try a different query/resource, then answer or admit the miss"
    # robustness regardless of how strong the local model is.
    "research",
    "memory_recall", "wikipedia", "context_peek",
    # Reference utilities — pure reads, voice-manifest core since the
    # manifest's first cut, but absent HERE until 2026-06-12: the
    # becca_direct loop had no calculator, so "what's 15% of 240" got
    # small-model mental math (wiring program P0, confirmed live).
    # NOTE: no "datetime" here — the DateTimeTool was removed from the
    # registry (server.py::_build_tool_registry) because the current
    # date/time is injected into every system prompt; listing it here just
    # produced a tool_resolve_failed warning every turn.
    "calculator", "unit_converter",
    # Creation tools (wiring program Phase 7, 2026-06-12) — artifacts
    # land in the canonical artifacts store with origin='companion'
    # (migration 263 + the ARTIFACT_ORIGIN contextvar stamped on this
    # path); she narrates where the thing went. youtube = transcripts;
    # image_search/remove_background round out the visual workflow.
    "create_document", "create_spreadsheet", "create_presentation",
    "create_chart", "convert_document", "youtube", "image_search",
    "remove_background",
    # Consumption-entity ladder, Gate 1 (2026-06-12) — catalog-grounded
    # "what should I pick up next" picks from the user's own library
    # (continue-the-series / new-arrivals / same-author / same-shelf).
    "media_recommendations",
    # Scheduling substrate (2026-07-07) — timed action is an app-level
    # capability (2026-07-02 policy), so the companion carries the same
    # always-on scheduling roster as chat and voice. These tools already
    # declared surfaces.companion=True; this makes the loop honor it.
    # Data-returning (confirmation payloads), headless-doctrine safe.
    "schedule_briefing", "schedule_request", "watch_for",
    "schedule_deadline", "schedule_action",
    "list_briefings", "cancel_briefing",
]

# Phone action verbs. On a phone surface (voice/assist) these must always be
# offered — they're the assistant basics. Relevance-ranking was clipping them
# out of the budgeted roster (e.g. "open Spotify" found no device.launch_app and
# mis-fired navigate.open_surface instead), so we append them unconditionally for
# phone turns, exempt from the budget like CORE_TOOL_NAMES. bluetooth_list is a
# read; the rest are client-executed effects. Desktop chat never gets these
# (they only mean anything on the phone).
DEVICE_TOOL_NAMES = [
    "device.set_alarm", "device.set_timer", "device.launch_app",
    "device.dial", "device.compose_text", "device.add_contact",
    "device.bluetooth_list",
]

# Data tools whose execution means "she went somewhere" — these feed
# the trail (her positions, jumpable via "take me there").
_TRAIL_KINDS = {
    "web_search": "search",
    "web_fetch": "page",
    "wikipedia": "page",
}

TRAIL_CAP = 20


def _visual_surface_event(name: str, meta: Any) -> dict[str, Any] | None:
    """Build an ``intent_action`` surface payload that opens the native
    panel for a visual tool result — ``image_search`` → image viewer,
    ``youtube`` → watch panel. Returns None for non-visual tools or
    empty results.

    These results belong on a screen, not in narration ("here are six
    URLs" is noise out loud). On the voice path the loop parks this on
    the ReferentCache and the route drain emits it as ``intent_action``,
    which the client's intent-action-router opens. Mirrors the legacy
    voice forwarding keys (voice_routes ``_process_voice_turn``) so the
    client needs only two new channels. Chat is unaffected — it renders
    these inline and the loop only parks this when not self-draining.
    """
    if not isinstance(meta, dict):
        return None
    surface: dict[str, Any] | None = None
    if name == "image_search":
        images = meta.get("images") or []
        if images:
            surface = {
                "channel": "image.search",
                "payload": {"images": images[:6]},
            }
    elif name == "youtube":
        mode = meta.get("youtube_mode") or ""
        if mode == "search" and meta.get("results"):
            surface = {
                "channel": "youtube.open",
                "payload": {
                    "youtube_mode": "search",
                    "results": meta.get("results") or [],
                },
            }
        elif mode == "direct" and meta.get("video_id"):
            surface = {
                "channel": "youtube.open",
                "payload": {
                    "youtube_mode": "direct",
                    "video_id": meta.get("video_id", ""),
                    "title": meta.get("title", ""),
                    "channel": meta.get("channel", ""),
                    "thumbnail": meta.get("thumbnail", ""),
                    "url": meta.get("url", ""),
                },
            }
    if surface is None:
        return None
    return {
        "type": "intent_action", "v": 1,
        "action": name, "tier": 3, "short_circuit": False,
        "surface": surface,
    }


# Tools that, on the voice path, hand the work to a client-side surface
# with its own progress UI instead of blocking the turn server-side.
# Image generation is the canonical long task: the image panel already
# owns the VRAM check + live progress + abort, so routing there lets the
# conversation keep going while it renders — detach-and-notify, the
# 2026-06-12 long-horizon contract, achieved by wiring not building.
_VOICE_HANDOFF_TOOLS = {"image_generation"}

_HANDOFF_IMAGE_KEYS = (
    "model", "negative_prompt", "width", "height", "steps",
    "cfg_scale", "preset",
)


def _detach_long_tasks_enabled() -> bool:
    """Admin gate for the voice long-task detach (defaults on)."""
    try:
        from augmentum.config import settings
        return bool(getattr(settings, "companion_voice_detach_long_tasks", True))
    except Exception:  # noqa: BLE001
        return True


def _handoff_surface_event(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Surface event that hands a long task to its client panel.

    ``image_generation`` → ``image.generate`` (the image panel runs it
    with the canonical progress UI). Returns None when there's nothing
    actionable (no prompt) so the caller falls back to normal execution.
    """
    if name == "image_generation":
        prompt = str(args.get("prompt") or args.get("query") or "").strip()
        if not prompt:
            return None
        payload: dict[str, Any] = {"prompt": prompt}
        for k in _HANDOFF_IMAGE_KEYS:
            if args.get(k) not in (None, ""):
                payload[k] = args[k]
        return {
            "type": "intent_action", "v": 1,
            "action": name, "tier": 3, "short_circuit": False,
            "surface": {"channel": "image.generate", "payload": payload},
        }
    return None


def _urls_from_meta(meta: Any, *, cap: int = 8) -> list[str]:
    """Pull source URLs out of a tool result's metadata, in priority order.

    Handles the shapes our read tools emit: research/web_search expose
    ``citations`` (with ``url``), ``results`` (with ``url``), and a flat
    ``urls`` list; web_fetch exposes a single ``url``. De-duplicated,
    order-preserving, capped.
    """
    if not isinstance(meta, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _add(u: Any) -> None:
        if isinstance(u, str) and u.startswith(("http://", "https://")) and u not in seen:
            seen.add(u)
            out.append(u)

    for c in meta.get("citations") or []:
        if isinstance(c, dict):
            _add(c.get("url"))
    for r in meta.get("results") or []:
        if isinstance(r, dict):
            _add(r.get("url"))
    for u in meta.get("urls") or []:
        _add(u)
    _add(meta.get("url"))
    return out[:cap]


def _tool_name_collision_key(name: str) -> str:
    """Backend-strict identity for a tool name. OpenAI-compat backends
    validate function names after their own normalization — DeepSeek
    rejected a payload with 'Tool names must be unique' (2026-06-13,
    voice_native_loop_crashed) even though our raw strings differed.
    Treat dots/dashes/case as the same character class an upstream
    sanitizer would."""
    return name.strip().lower().replace(".", "_").replace("-", "_")


def assemble_native_tools(registry: Any, requested: list[str]) -> list[Any]:
    """Resolve requested tool names into a payload-safe, unique list.

    Two failure modes this guards (both observed live 2026-06-13):

    * The registry's fuzzy resolver (substring/alias fallback) can map
      two DIFFERENT requested names onto the SAME tool — deduping by
      requested name alone then ships a duplicate function name and a
      strict backend 400s the whole turn.
    * It can also remap a name to something surprising ('browse' →
      'web', 'image' → 'note.attach_image'). That's logged loudly here
      so the roster drift is visible instead of silent.
    """
    tools: list[Any] = []
    seen_requested: set[str] = set()
    seen_resolved: set[str] = set()
    for name in requested:
        if name in seen_requested:
            continue
        seen_requested.add(name)
        tool = registry.resolve(name)
        if tool is None:
            continue
        key = _tool_name_collision_key(tool.name)
        if key in seen_resolved:
            log.info(
                "native_loop_tool_deduped",
                requested=name, resolved=tool.name,
            )
            continue
        seen_resolved.add(key)
        if tool.name != name:
            # A successful alias resolution (recall->memory_recall,
            # image->image_generation, browse->web, files_read->search_files,
            # code_run->python_exec) — expected by design, fires every turn.
            # debug, not warning: real roster drift surfaces as a behaviour
            # change, and the dedup branch above already logs at info. Keeping
            # this at warning buried the log under routine remaps.
            log.debug(
                "native_loop_tool_remapped",
                requested=name, resolved=tool.name,
            )
        tools.append(tool)
    return tools


def select_companion_tools(
    registry: Any,
    *,
    intent: Any,
    app_state: Any,
    user_id: str,
    session_id: str,
    context_length: int = 0,
) -> list[Any]:
    """Pick the companion's toolset for a turn: relevance-ranked verb
    roster ∩ registry, plus the core capability tools, resolved to
    payload-safe ``Tool`` objects.

    Shared by the native loop (hop 2+) and the voice streaming first hop
    so BOTH expose the identical toolset — the first hop can't reliably
    trigger a tool the second hop would have offered, and vice versa.

    Scoring text blends the previous two user turns with the current one
    (mirrors voice.py::_roster_scoring_text): an answer turn like
    "Springfield, Illinois" carries none of the asking turn's vocabulary,
    and scoring it alone clipped the needed verb out of the roster
    (companion_eval clarify-weather scenario, 2026-06-11). A parked
    clarification pins its verb past the relevance ranking.
    """
    from augmentum.companion_runtime import tools as tool_bridge

    _scoring_parts = [getattr(intent, "text", "") or ""]
    try:
        _recent = (getattr(intent, "metadata", None) or {}).get("recent_turns") or []
        _prev_users = [
            (t.get("content") or "") for t in _recent
            if (t.get("role") or "") == "user"
        ][-2:]
        _scoring_parts = _prev_users + _scoring_parts
    except Exception:  # noqa: BLE001 — blend is best-effort
        log.debug("roster_scoring_blend_failed", exc_info=True)
    _pin = tool_bridge.pending_pin(app_state, user_id, session_id)
    # Subagent catalogue entries (analytical, passthrough) are multi-step
    # handoffs dispatched via the subagent path, NOT entries in the global
    # ToolRegistry — feeding their names to assemble_native_tools only ever
    # produced a tool_resolve_failed warning. They stay in the PROMPT roster
    # (enumerate_tools) so the model still knows about them; they're just not
    # resolved as native function-call tools here.
    _subagent_keys = {
        k for k, v in tool_bridge.TOOL_CATALOG.items()
        if v.get("registry") == "subagent"
    }
    from augmentum.companion_runtime.context_budget import (
        derive_roster_char_budget,
    )

    roster_names = [
        t["name"]
        for t in tool_bridge.enumerate_tools(
            " ".join(p for p in _scoring_parts if p), pin=_pin,
            context_budget_chars=derive_roster_char_budget(context_length),
        )
        if t["name"] not in _subagent_keys
    ]
    # Phone turns always carry the device action verbs (budget-exempt), so
    # "open <app>" / "set a timer" / "call <name>" can actually fire instead of
    # being clipped from the roster and mis-routed to a web-surface verb.
    _meta = getattr(intent, "metadata", None) or {}
    device_names = (
        DEVICE_TOOL_NAMES
        if (_meta.get("voice_channel") or _meta.get("phone_assist"))
        else []
    )
    return assemble_native_tools(
        registry, roster_names + CORE_TOOL_NAMES + device_names,
    )


def _append_trail(
    app_state: Any, user_id: str, session_id: str,
    *, kind: str, label: str, ref: str = "",
) -> None:
    """Record a position on her trail. Never raises."""
    try:
        import time

        from augmentum.intent.dispatch import get_referent_cache
        refs = get_referent_cache(app_state, user_id, session_id)
        trail = getattr(refs, "trail", None)
        if trail is None:
            return
        trail.append({
            "kind": kind,
            "label": (label or "")[:160],
            "ref": (ref or "")[:300],
            "ts": time.time(),
        })
        del trail[:-TRAIL_CAP]
        # Continuity: persist the trail so "take me there" survives
        # restarts and voice session-id churn (fire-and-forget).
        from augmentum.companion_runtime.working_state import schedule_save
        schedule_save(app_state, user_id, refs)
    except Exception:  # noqa: BLE001
        log.debug("becca_trail_append_failed", exc_info=True)


async def native_loop_events(
    request: Any,
    *,
    backend: Any,
    runtime: Any,
    intent: Any,
    registry: Any,
    user_id: str,
    session_id: str,
    app_state: Any,
    initial_calls: list[tuple[str, dict[str, Any]]] | None = None,
    initial_assistant_text: str = "",
    cancel: Any = None,
    drain_surface_events: bool = True,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Run the companion turn on the native tool loop, yielding events.

    ``request`` must already carry Becca's composed system message —
    the helper PassthroughHandler is borrowed for loop primitives only;
    its own prompt pipeline is never invoked.

    Voice handoff (best-universal-system adaptation, 2026-06-11):
    ``initial_calls`` are tool calls ALREADY parsed from a streamed
    first hop (the voice sieve) — they execute first, their results
    land in ``request.messages`` in native format, and the loop
    continues from there: every subsequent hop runs tier-1 native
    function calling with the 5-tier tolerant parser, and the final
    text is the model's own synthesis over the gathered results.
    ``initial_assistant_text`` is the prose that preceded those calls
    (the spoken promise) — preserved as the assistant message content.
    ``cancel`` (asyncio.Event-like) is checked between hops.
    ``drain_surface_events=False`` leaves pending_surface_events
    parked for the caller's own delivery channel (the voice route
    drains them into WS ``intent_action`` events at turn end —
    draining here would silently eat the sticky-note open).
    """
    from augmentum.intent.dispatch import get_referent_cache
    from augmentum.models.base import response_text
    from augmentum.modes.analytical.tool_calling import (
        ToolCallingTier,
        extract_structured_text,
    )
    from augmentum.modes.passthrough.handler import (
        PassthroughHandler,
        _max_iterations,
    )
    from augmentum.tools.turn_search_dedup import (
        TurnSearchDedup,
        is_search_round,
        set_turn_dedup,
    )

    if registry is None:
        raise RuntimeError("becca_native_loop: no tool_registry")

    # Per-turn search dedup shared across this turn's tool rounds (chat + voice
    # consume this generator). Web/image/youtube results returned in one round
    # are remembered so later rounds surface only NEW items; a round that
    # searches but adds nothing new trips the productivity guard below. Scoped
    # to the consuming request's task — no cross-turn leak.
    from augmentum.training.trace_context import begin_capture, end_capture
    _cap_ctx, _cap_tok = begin_capture(
        user_id=user_id, session_id=session_id, mode="becca_direct",
    )

    _dedup = TurnSearchDedup()
    set_turn_dedup(_dedup)
    _no_progress_rounds = 0

    # Tool selection — relevance-ranked roster ∩ registry + core
    # capability tools. Shared with the voice first hop via
    # ``select_companion_tools`` so both surfaces offer the same set.
    from augmentum.companion_runtime.context_budget import (
        resolve_context_length,
    )

    _ctx_len = await resolve_context_length(runtime)
    tools = select_companion_tools(
        registry,
        intent=intent,
        app_state=app_state,
        user_id=user_id,
        session_id=session_id,
        context_length=_ctx_len,
    )
    if not tools:
        raise RuntimeError("becca_native_loop: no tools resolved")

    helper = PassthroughHandler(
        backend=backend,
        session_id=session_id,
        tool_registry=registry,
        user_id=user_id,
        app_state=app_state,
    )
    # Provenance, not silos: tools executed from HER loop stamp their
    # persisted artifacts origin='companion' (image_generations etc.) —
    # same stores, same surfaces, filterable. Separate from the mode
    # stamp so offer gating is unaffected.
    helper._ctx_origin = "companion"  # noqa: SLF001

    tier = helper._inject_tool_schemas(request, tools)  # noqa: SLF001
    is_text = tier == ToolCallingTier.TEXT
    tool_map = {t.name: t for t in tools}
    executed_non_repeatable: set[str] = set()
    final = None

    async def _run_calls(response, calls):
        """Execute one batch of calls: yields tool_call/tool_result
        events, appends assistant+result messages to ``request``, and
        updates trail / commitments / non-repeatable state."""
        # Voice detach (long-horizon contract): if EVERY call this hop
        # hands off to a client surface (image gen → image panel), don't
        # block the turn on server-side work. Park the surface event,
        # feed a synthetic "started — acknowledge briefly" tool result so
        # she says "putting that together, it'll pop up" and the turn
        # ends, and let the panel render it with its own progress while
        # the user keeps talking. Mixed batches fall through to normal
        # execution (safe — never split one response's tool_calls). Voice
        # only: chat shows the image inline and wants it in the thread.
        if (
            not drain_surface_events
            and calls
            and all(c[0] in _VOICE_HANDOFF_TOOLS for c in calls)
            and _detach_long_tasks_enabled()
        ):
            handoff = []
            for _name, _args, _tc in calls:
                sev = _handoff_surface_event(_name, dict(_args or {}))
                if sev is None:
                    handoff = None
                    break
                handoff.append((_name, _args, _tc, sev))
            if handoff:
                import json as _json

                from augmentum.models.base import Message
                _ac = getattr(getattr(response, "message", None), "content", "") or ""
                request.messages.append(Message(
                    role="assistant",
                    content=_ac,
                    tool_calls=[
                        {"id": tc, "type": "function", "function": {
                            "name": nm,
                            "arguments": _json.dumps(dict(ar or {})),
                        }}
                        for nm, ar, tc in calls
                    ],
                ))
                try:
                    _refs = get_referent_cache(app_state, user_id, session_id)
                except Exception:  # noqa: BLE001
                    _refs = None
                for nm, ar, tc, sev in handoff:
                    yield ("tool_call", {"tool": nm, "args": dict(ar or {})})
                    if _refs is not None:
                        _refs.pending_surface_events.append(sev)
                    request.messages.append(Message(
                        role="tool",
                        content=(
                            "Handed to the image panel — it's rendering there "
                            "now and will appear when ready. Acknowledge in ONE "
                            "short, natural sentence; do NOT describe the image "
                            "or claim it's done."
                        ),
                        tool_call_id=tc,
                    ))
                    yield ("tool_result", {
                        "tool": nm, "ok": True,
                        "payload_summary": "handed to image panel",
                        "duration_ms": 0, "urls": [],
                    })
                    executed_non_repeatable.add(nm)
                return

        for tool_name, tool_args, _tc_id in calls:
            yield ("tool_call", {
                "tool": tool_name,
                "args": dict(tool_args or {}),
            })

        results: list[tuple[str, bool, str, int, dict]] = []

        async def _on_result(
            name, success, snippet, meta, _tc, dur_ms, _results=results,
        ):
            _results.append((
                name, bool(success), snippet or "", int(dur_ms),
                meta if isinstance(meta, dict) else {},
            ))

        succeeded = await helper._execute_and_append(  # noqa: SLF001
            request, response, calls,
            on_tool_result=_on_result,
            text_tier=is_text,
        )

        call_args = {name: dict(args or {}) for name, args, _tc in calls}
        for name, ok, snippet, dur_ms, meta in results:
            yield ("tool_result", {
                "tool": name,
                "ok": ok,
                "payload_summary": snippet[:400],
                "duration_ms": dur_ms,
                # Source URLs the tool actually used — lets a caller (e.g.
                # prompt_fire) attach citations to its delivered result so
                # the answer isn't a wall of "[1]/[3]" with no links.
                "urls": _urls_from_meta(meta),
            })
            if ok:
                # Trail: data tools are positions she visited.
                trail_kind = _TRAIL_KINDS.get(name)
                if trail_kind:
                    args = call_args.get(name, {})
                    _append_trail(
                        app_state, user_id, session_id,
                        kind=trail_kind,
                        label=str(
                            args.get("query") or args.get("url")
                            or args.get("title") or name
                        ),
                        ref=str(args.get("url") or ""),
                    )
                # Results ring: what she just looked at survives a few
                # turns as a digest (full payload kept as detail for
                # re-inflation / peek). Digest is INDEXICAL — the topic,
                # never the findings; half-enumerated findings get
                # confabulated on follow-up turns.
                try:
                    from augmentum.companion_runtime import ring as _ring
                    args = call_args.get(name, {})
                    topic = str(
                        args.get("query") or args.get("url")
                        or args.get("title") or args.get("prompt") or ""
                    )[:80]
                    _ring.record(
                        get_referent_cache(app_state, user_id, session_id),
                        kind="tool",
                        slot=f"tool:{name}",
                        label=f"{name}: {topic}" if topic else name,
                        digest="result gathered — details available",
                        detail=snippet,
                    )
                except Exception:  # noqa: BLE001
                    log.debug("becca_ring_record_failed", exc_info=True)
                # Visual results (image_search / youtube) belong on a
                # screen — park a surface event so the voice route drain
                # opens the native panel. Voice only: chat renders these
                # inline, and it self-drains the queue (drain_surface_events
                # True), so adding it there would double up.
                if not drain_surface_events:
                    sev = _visual_surface_event(name, meta)
                    if sev is not None:
                        try:
                            get_referent_cache(
                                app_state, user_id, session_id,
                            ).pending_surface_events.append(sev)
                        except Exception:  # noqa: BLE001
                            log.debug(
                                "becca_visual_surface_push_failed",
                                exc_info=True,
                            )
                # Successful dispatch settles the most recent open
                # commitment (same policy as the voice path).
                try:
                    from augmentum.companion_runtime import commitments
                    await commitments.close_latest(runtime, user_id=user_id)
                except Exception:  # noqa: BLE001
                    log.debug("commitment_close_crashed", exc_info=True)

        for name, _args, _tc in calls:
            tool_obj = tool_map.get(name)
            if tool_obj is not None and not tool_obj.cacheable and name in succeeded:
                executed_non_repeatable.add(name)

    # Seeded calls from a streamed first hop (voice sieve). Executed
    # via the same machinery so their results land in request.messages
    # in native format — the loop's next hop sees them as proper tool
    # results and can chain or synthesize.
    if initial_calls:
        if cancel is not None and cancel.is_set():
            end_capture(_cap_ctx, _cap_tok, error="cancelled")
            return
        from augmentum.models.base import InternalChatResponse, Message
        synthetic = InternalChatResponse(
            message=Message(
                role="assistant",
                content=initial_assistant_text or "",
            ),
            model=getattr(request, "model", "") or "",
        )
        seeded = [
            (name, dict(args or {}), f"sieve_{i}")
            for i, (name, args) in enumerate(initial_calls)
        ]
        seeded = [c for c in seeded if c[0] not in executed_non_repeatable]
        if seeded:
            async for ev in _run_calls(synthetic, seeded):
                yield ev

    _gen_s = 0.0
    # Resolved per turn (live setting + per-request override), not
    # captured at import — see passthrough.handler._max_iterations.
    _max_iters = _max_iterations(request)
    for _iteration in range(_max_iters):
        if cancel is not None and cancel.is_set():
            end_capture(_cap_ctx, _cap_tok, error="cancelled")
            return
        _gen_t0 = time.monotonic()
        try:
            response = await backend.chat(request)
        except Exception as exc:
            # A backend rejecting the TOOL SCHEMA (strict validators:
            # duplicate names, unsupported shapes) must not kill the
            # turn — from the user's seat she just never answers
            # (voice_native_loop_crashed, 2026-06-13). Strip tools and
            # answer in words once; a second failure is a real backend
            # problem and propagates.
            if _iteration == 0 and getattr(request, "tools", None):
                log.warning(
                    "native_loop_tools_rejected_retrying_bare",
                    error=str(exc)[:200],
                )
                request.tools = None
                response = await backend.chat(request)
            else:
                raise
        _gen_s = time.monotonic() - _gen_t0
        final = response
        calls = helper._parse_tool_calls(response, tools)  # noqa: SLF001
        calls = [c for c in calls if c[0] not in executed_non_repeatable]
        if not calls:
            break
        _dedup.begin_round()
        async for ev in _run_calls(response, calls):
            yield ev

        # Productivity guard: a round that searched but surfaced nothing new is
        # spinning. After two such rounds, stop hopping and let her synthesize
        # from what's gathered (the natural break → final-prose path below).
        if is_search_round(calls):
            if _dedup.round_new_count() == 0:
                _no_progress_rounds += 1
                if _no_progress_rounds >= 2:
                    log.info("becca_native_loop_no_progress_stop",
                             iteration=_iteration, new=0)
                    break
            else:
                _no_progress_rounds = 0
    else:
        log.warning("becca_native_loop_max_iterations", max=_max_iters)

    # Final prose — her synthesis over the verified results.
    # thinking_fallback=False: a think-only response (reasoning burned
    # the token budget before any visible content) must NOT become her
    # reply — companion_eval caught a turn whose "answer" was the raw
    # plan ("I have a tool memory.save... Plan: 1. Use...", 2026-06-11).
    # On voice that text goes to TTS. An empty reply is an honest
    # failure the caller can see; leaked chain-of-thought is not.
    text = (
        response_text(final, thinking_fallback=False)
        if final is not None else ""
    )
    if not text and final is not None and response_text(final):
        log.warning(
            "becca_think_only_response",
            model=getattr(request, "model", "") or "",
            hint="reasoning consumed the whole token budget; no visible content",
        )
    if tier == ToolCallingTier.STRUCTURED and text:
        text = extract_structured_text(text)
    if text:
        # Final-turn telemetry — the real tokens/sec for the answer generation,
        # emitted as an ADDITIVE event (consumers that don't switch on "metrics"
        # ignore it, so chat/voice are unaffected). The blocking loop has no
        # token stream, so this is the whole-generation rate (not a live tick);
        # ttft isn't meaningful here. completion_tokens comes from the backend's
        # usage when present, else a length estimate.
        try:
            usage = getattr(final, "usage", None)
            comp = int(getattr(usage, "completion_tokens", 0) or 0)
            if comp <= 0:
                comp = max(1, len(text) // 4)
            if _gen_s > 0:
                yield ("metrics", {
                    "tok_per_s": round(comp / _gen_s, 1),
                    "gen_ms": int(_gen_s * 1000),
                    "completion_tokens": comp,
                })
        except Exception:  # noqa: BLE001 — telemetry never breaks a turn
            log.debug("native_loop_metrics_failed", exc_info=True)
        yield ("text", {"text": text})
        # Record the assembled answer into THIS scope's trace. The loop emits
        # its final text as a ("text", …) event rather than as backend content,
        # so the backend-boundary hook can capture an empty response (the voice
        # route's symptom). Capture-only; the writer uses this only when the
        # flattened chain produced no text of its own.
        try:
            from augmentum.training.trace_context import note_final_response
            note_final_response(text)
        except Exception:  # noqa: BLE001 — capture must never break the turn
            pass

    # Surface-event drain — ActionTools parked intent_action payloads
    # on the per-session ReferentCache. Consumers route each through
    # their surface channel (chat: ui_effects chunk; voice: WS drain —
    # voice passes drain_surface_events=False so its route-level drain
    # finds the queue intact).
    if not drain_surface_events:
        end_capture(_cap_ctx, _cap_tok)
        return
    try:
        refs = get_referent_cache(app_state, user_id, session_id)
        if refs.pending_surface_events:
            queue = refs.pending_surface_events
            refs.pending_surface_events = []
            effects = [
                {
                    "kind": (ev.get("surface") or {}).get("channel", ""),
                    "target": "_inline",
                    "payload": (ev.get("surface") or {}).get("payload", {}),
                }
                for ev in queue
                if (ev.get("surface") or {}).get("channel")
            ]
            if effects:
                yield ("ui_effects", {"effects": effects})
    except Exception:  # noqa: BLE001
        log.debug("becca_native_loop_drain_failed", exc_info=True)

    end_capture(_cap_ctx, _cap_tok)
