"""WebSocket voice chat endpoint — real-time STT → LLM → TTS pipeline.

Handles the full voice conversation loop: receives audio from the client,
transcribes via STT, routes through the existing chat pipeline, buffers
LLM output into sentences, and streams TTS audio back per sentence.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, NamedTuple

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from augmentum.architect import (
    ArchitectResult,
    dispatch_architect_command,
)
from augmentum.architect.address import AddressDecision
from augmentum.architect.router import (
    ConfidenceStack,
    dispatch_router_decision,
    route_utterance,
)
from augmentum.architect.voice_router import classify_voice
from augmentum.classifier.router import Mode
from augmentum.config import settings
from augmentum.intent import (
    SessionContext as IntentSessionContext,
)
from augmentum.intent import (
    dispatch as intent_dispatch,
)
from augmentum.intent import (
    get_referent_cache as get_intent_referents,
)
from augmentum.intent import (
    serialize_action_event,
)
from augmentum.intent.matcher import match_intent
from augmentum.memory.integration import recall_and_inject
from augmentum.models.base import InternalChatRequest, Message
from augmentum.proxy.handler_factory import (
    _resolve_passthrough_tools,
    get_handler_for_mode,
)
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop
from augmentum.voice import smart_turn
from augmentum.voice.audio_processor import AudioProcessor, normalize_pcm
from augmentum.voice.denoiser import SpeechEnhancer
from augmentum.voice.kokoro_tts import KokoroTTS
from augmentum.voice.moonshine_stt import MoonshineSTTSession
from augmentum.voice.pipeline import (
    SentenceBuffer,
    VoiceSession,
    clean_for_tts,
    effective_chunking_mode,
    is_backchannel,
    prefetch_tts_audio,
    send_prefetched_audio,
    stream_tts_sentence,
    transcribe_audio,
    warmup_tts,
)
from augmentum.voice.pipeline_resolver import resolve as resolve_pipeline_target
from augmentum.voice.speaker import SpeakerVerifier, VoicePrint
from augmentum.voice.streaming_stt import (
    BatchSTTFallback,
    StreamingSTTSession,
    TranscriptEvent,
    is_streaming_stt_capable,
)
from augmentum.voice.tts import _resolve_tts_provider, maybe_emit_normalized_schedule
from augmentum.voice.vad import FRAME_BYTES, SAMPLE_RATE, VadProcessor

log = get_logger(__name__)

router = APIRouter(tags=["voice"])


# ---------------------------------------------------------------------------
# Voice-appropriate tools — sourced from the shared intent manifest
# ---------------------------------------------------------------------------
#
# The set of voice-callable tools (notes, memory, navigation, media
# transport, web search, image gen, etc.) lives in
# augmentum/intent/manifest.py. That module is the single source of
# truth — both this route AND the architect's voice router read from
# it. Adding a new primitive: register the action, then add its id to
# the appropriate bucket in manifest.py (CORE / INTERACTIVE /
# DISRUPTIVE / COSTLY). Don't expand the hard-coded fallback here.
#
# ``_VOICE_TOOLS`` retains its name + frozenset shape for any code
# that still imports it directly; it's the all-buckets union.
#
# GOTCHA (fixed 2026-07-07): the union must be computed at LOOKUP time,
# never snapshotted at import. This module imports during app build,
# but ``manifest.bind_registry`` runs later in lifespan — a module-level
# ``all_voice_tools()`` freeze predates the binding and permanently
# excludes every ``Tool.surfaces.voice`` opt-in (scheduling substrate,
# document convert, background remove, …), silently defeating the
# manifest's read-at-access-time design.
from datetime import UTC

from augmentum.intent.manifest import (
    DEFAULT_AMBIENT_POLICY,
    all_voice_tools,
    capability_line,
    voice_tools_for,
)
from augmentum.intent.manifest import (
    VOICE_TOOL_CAPABILITIES as _VOICE_TOOL_CAPABILITIES_BASE,
)


def _voice_tools() -> frozenset[str]:
    """Live all-buckets union — see GOTCHA above."""
    return all_voice_tools()


def __getattr__(name: str):  # legacy import surface for _VOICE_TOOLS
    if name == "_VOICE_TOOLS":
        return all_voice_tools()
    raise AttributeError(name)

# Live-vision (camera) frame budget. The companion turn attaches at most
# this many of the freshest webcam frames; more than ~2 frames competes
# with the chat model + TTS for GPU and rarely adds signal for a single
# spoken turn (see project_hardware_tiers — per-frame VL prefill is the
# real cost). _LIVE_VISION_STALE_S drops frames older than this so a turn
# never reasons about what the camera saw minutes ago.
_LIVE_VISION_MAX_FRAMES = 2
_LIVE_VISION_STALE_S = 8.0
# How long an offered media pick (pending_candidates) stays eligible for the
# router to re-present as a tappable "the second one". Past this, the offer is
# stale — re-presenting it lets an old recommend get replayed for an unrelated
# later request (e.g. an audiobook offer answering "throw in some music").
_OFFERED_CANDIDATES_TTL_S = 120.0

# Base voice instruction — always injected during voice chat so the LLM
# knows it's in a spoken conversation and can handle STT artifacts.
_VOICE_BASE_INSTRUCTION = (
    "You are in a live voice conversation. The user's messages are "
    "transcribed from speech and may contain misspellings, homophones, "
    "or garbled words from speech-to-text errors — infer the intended "
    "meaning from context rather than interpreting them literally. "
    "Keep your responses concise and conversational — the user will "
    "hear them spoken aloud. Do NOT use markdown formatting, bullet "
    "lists, code blocks, or special characters."
)

# Additional instruction appended when voice tools are active.
_VOICE_TOOL_ADDENDUM = (
    " When you decide to use a tool, briefly tell the user what you're "
    "about to do BEFORE calling it (e.g. 'Let me search for that…'). "
    "If the user asks something that requires personal context — their "
    "location, name, preferences, or past conversations — use the "
    "memory_recall tool to check before answering."
)

# Per-tool capability description — sourced from the shared manifest so
# new primitives only need to register their line in one place. Re-
# exported under the legacy name for any importer outside this file.
_VOICE_TOOL_CAPABILITIES: dict[str, str] = dict(_VOICE_TOOL_CAPABILITIES_BASE)

_XR_SURFACE_INSTRUCTIONS: dict[str, str] = {
    "chat": (
        " The user has selected the Chat surface in VR. Treat the next turn "
        "as part of the current conversation unless they ask to switch tasks."
    ),
    "analytical": (
        " The user has selected the Analyze surface in VR. Favor careful "
        "reasoning, evidence comparison, citations, and explicit uncertainty."
    ),
    "agentic": (
        " The user has selected the Build surface in VR. Help turn the request "
        "into tracked steps, tool use, progress updates, and completion checks."
    ),
    "browse": (
        " The user has selected the Browse surface in VR. If they ask for "
        "current information, pages, sources, or summaries, use the available "
        "web/search tools instead of guessing. If a page has playable media, "
        "treat play media as a handoff to the Media surface."
    ),
    "files": (
        " The user has selected the Files surface in VR. Expect requests about "
        "documents, attachments, comparison, and saved context."
    ),
    "coder": (
        " The user has selected the Coder surface in VR. Focus on software work: "
        "plans, diffs, commands, test results, previews, and approvals."
    ),
    "narrative": (
        " The user has selected the Story surface in VR. Preserve character, "
        "scene, and roleplay context while still responding naturally by voice."
    ),
    "notes": (
        " The user has selected the Notes surface in VR. Treat spoken requests "
        "as dictation, clipping, organization, or recap work unless they ask otherwise."
    ),
    "studio": (
        " The user has selected the Studio surface in VR. Focus on visual artifact "
        "creation, variants, edits, and reviewable output."
    ),
    "media": (
        " The user has selected the Media surface in VR. Expect playback, search, "
        "watch-together, captions, comics, images, audiobooks, local files, games, "
        "and discussion requests."
    ),
    "devices": (
        " The user has selected the Devices surface in VR. Focus on casting, "
        "connected devices, playback control, pairing, and session status."
    ),
    "games": (
        " The user has selected the Games surface in VR. Focus on launching, "
        "resuming, streaming, saves, and controller state."
    ),
}


def _xr_surface_addendum(surface: str) -> str:
    return _XR_SURFACE_INSTRUCTIONS.get((surface or "").strip().lower(), "")


def _xr_panel_action_addendum(action: str) -> str:
    action = (action or "").strip().lower()[:80]
    if not action:
        return ""
    label = action.replace("_", " ").replace("-", " ")
    return (
        f" The user selected the '{label}' action on the active VR panel. "
        "Treat it as their immediate interface intent and ask only if a missing "
        "detail blocks the action."
    )


def _xr_user_signal_addendum(signals: list[dict[str, Any]]) -> str:
    if not signals:
        return ""
    lines: list[str] = []
    for signal in signals[-4:]:
        if not isinstance(signal, dict):
            continue
        summary = str(signal.get("summary") or "").strip()
        typ = str(signal.get("type") or "").strip()
        confidence = signal.get("confidence")
        if not summary and typ:
            summary = f"The headset detected user action: {typ.replace('_', ' ')}."
        if not summary:
            continue
        try:
            conf = float(confidence or 0)
        except (TypeError, ValueError):
            conf = 0.0
        suffix = f" (confidence {conf:.2f})" if conf > 0 else ""
        lines.append(f"- {summary[:240]}{suffix}")
    if not lines:
        return ""
    return (
        " Recent XR nonverbal user context before this spoken turn:\n"
        + "\n".join(lines)
        + "\nUse this as situational context. Do not over-explain that sensors detected it; "
        "respond naturally if it is relevant, otherwise keep it passive."
    )


def _voice_capability_addendum(resolved: list[str]) -> str:
    """Return a one-paragraph capability list for the system prompt.

    Local models (notably Qwen3) sometimes refuse to use tools that *are*
    listed in their tool schema, claiming "I can only search the web" or
    similar. Naming each capability explicitly closes that refusal
    channel. The phrasing here is deliberately conservative: an earlier
    version said "USE the tool — never say you can't" and pushed the
    model the other way (it called web_search on every greeting). The
    rule of thumb is: tools are available *if needed*, not required.
    """
    capabilities = [
        line for line in (capability_line(name) for name in resolved) if line
    ]
    if not capabilities:
        return ""
    return (
        " The following tools are available IF a tool is genuinely the right "
        "way to answer (e.g. user asks for an image, a current fact, a video, "
        "a calculation): "
        + "; ".join(capabilities)
        + ". For greetings, opinions, follow-ups, casual chat, or anything "
        "you can answer from your own knowledge, do NOT call a tool — just "
        "answer directly. Don't refuse a tool you have, but don't reach for "
        "one when none is needed."
    )

# Synthesis hint injected after tool results so the model shapes its
# response for spoken delivery — depth over breadth.
_VOICE_SYNTHESIS_HINT = (
    "The user is listening, not reading. Prioritize depth over breadth: "
    "cover fewer points with meaningful context and detail rather than "
    "listing many items briefly. Speak naturally — no bullet points, "
    "no numbering, no headers."
)


def _resolve_voice_tools(
    app_state: Any,
    session_tools: list[str],
    *,
    ambient: bool = False,
) -> list[str]:
    """Resolve voice-mode tools via the shared chat resolver, then filter
    to the voice-safe allowlist and the ambient policy.

    Routes through :func:`_resolve_passthrough_tools` so config defaults
    (``settings.passthrough_tools``), auto-included zero-cost utilities,
    and ``all``/``none`` semantics stay consistent with the chat UI path.
    Tools whose output can't be meaningfully spoken (file_ops,
    python_exec, etc.) are unconditionally stripped via the manifest's
    universe set. When ``ambient`` is True (passive companion widget,
    not foreground voice call), the manifest's ambient policy further
    filters out disruptive / costly tools per the operator's
    ``companion_ambient_tool_policy`` setting.

    Args:
        app_state: Application state (for tool_registry).
        session_tools: Raw ``tools`` list from the WS config message.
            Empty list disables tools entirely. ``['all']`` is the on
            sentinel from the current voice UI; future granular UIs may
            send specific names (those will be honored where voice-safe).
        ambient: True when the surface is the always-on companion widget
            (``persona_id == 'becca'``). False for foreground voice call
            modal — full set regardless of policy.

    Returns:
        List of voice-safe tool names. Empty when tools are off or none
        of the resolved tools survive the policy filter.
    """
    if not session_tools:
        return []

    # The WS list maps directly onto the chat path's header semantics
    # (comma-joined names, with ``"all"`` and ``"none"`` as sentinels).
    header_value = ",".join(str(t) for t in session_tools)
    resolved = _resolve_passthrough_tools(app_state, header_value)

    # Outer filter — voice-universe (drops file_ops / python_exec /
    # similar non-spoken tools that the chat resolver may have included).
    allowed = [name for name in resolved if name in _voice_tools()]

    # Inner filter — ambient policy. Foreground calls bypass.
    policy = (
        getattr(settings, "companion_ambient_tool_policy", DEFAULT_AMBIENT_POLICY)
        or DEFAULT_AMBIENT_POLICY
    )
    custom = list(
        getattr(settings, "companion_ambient_tool_allowlist", []) or []
    )
    policy_allow = voice_tools_for(
        ambient=ambient,
        policy=policy,
        custom_allowlist=custom,
    )
    return [name for name in allowed if name in policy_allow]


def _is_ambient_session(session: Any) -> bool:
    """Detect the passive always-on companion surface.

    Voice arrives on two surfaces today:

    * ``persona_id == 'becca'`` — the always-on widget. STT runs while
      the user is doing other things; the companion may surface a
      result, set a reminder, etc. without explicit invitation. This
      is the surface the ambient tool policy is designed for.
    * Foreground voice call modal (no ``persona_id``) — the user has
      explicitly opened a session and is paying attention to it. No
      ambient policy applies; the full set of voice tools is in play.
    """
    return getattr(session, "persona_id", "") == "becca"


def _refresh_pipeline_targets(session: VoiceSession) -> None:
    """Recompute the resolver targets for every pipeline component.

    Called at session start with empty ``client_caps`` (yielding the
    'server' target everywhere — current behavior) and again every time
    a ``capabilities`` frame arrives from the client. Results land on
    ``session.pipeline_targets`` for downstream dispatch sites to
    consult. Today the legacy paths still own actual dispatch; the
    targets are observability + the seam through which client engines
    will land in later phases.
    """
    surface = "companion" if _is_ambient_session(session) else "call"
    policy_attr = f"voice_pipeline_mode_{surface}"
    policy = getattr(settings, policy_attr, "auto") or "auto"
    targets: dict[str, str] = {}
    for component in ("vad", "stt", "tts", "denoise"):
        try:
            targets[component] = resolve_pipeline_target(
                component,
                surface,
                client_caps=session.client_caps,
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001 — resolver MUST NOT break the call
            log.warning(
                "pipeline_resolver_failed",
                session=session.session_id,
                component=component,
                surface=surface,
                policy=policy,
                error=str(exc),
            )
            targets[component] = "server"
    session.pipeline_targets = targets
    log.info(
        "pipeline_targets_resolved",
        session=session.session_id,
        surface=surface,
        policy=policy,
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOOL_CUES: dict[str, str] = {
    "web_search": "Searching the web.",
    "web": "Searching the web.",
    "web_fetch": "Fetching that page.",
    "image_generation": "Generating an image.",
    "calculator": "Calculating.",
    "wikipedia": "Checking Wikipedia.",
    "memory_recall": "Checking my memory.",
    "datetime": "Checking the time.",
    "unit_converter": "Converting that.",
}


def _tool_cue(tool_names: list[str]) -> str:
    """Return a short spoken cue for the tools being used."""
    for name in tool_names:
        if name in _TOOL_CUES:
            return _TOOL_CUES[name]
    return "One moment."


def _parse_mode(mode_str: str) -> Mode:
    """Convert mode string to Mode enum."""
    modes = {
        "passthrough": Mode.PASSTHROUGH,
        "narrative": Mode.NARRATIVE,
        "analytical": Mode.ANALYTICAL,
        "agentic": Mode.AGENTIC,
    }
    return modes.get(mode_str, Mode.PASSTHROUGH)


async def _send_json(ws: WebSocket, data: dict[str, Any]) -> None:
    """Send a JSON control message, silencing errors on closed sockets."""
    try:
        await ws.send_text(json.dumps(data))
    except Exception:
        log.warning("voice_ws_send_failed", msg_type=data.get("type", "?"))


async def _commit_user_turn(
    websocket: WebSocket,
    session: Any,
    transcript: str,
    *,
    emit: bool = True,
) -> None:
    """Authoritative commit of the user's side of a voice turn.

    SINGLE source of truth for "this utterance is now real conversation."
    Both the VAD path (``_process_voice_turn``) and the streaming path
    (``_process_voice_turn_from_transcript``) MUST funnel through here so the
    contract can't drift between them.

    Two effects, in order:
      1. Append the turn to the in-memory pipeline session (LLM context).
      2. Emit ``user_committed`` so the call client persists the user side of
         the chat tree — symmetric with the assistant's ``turn_complete``.

    This MUST be reached only AFTER the intent / backchannel / staging gates:
    a command short-circuit returns before calling this, so commands never
    persist as conversation. The earlier ``transcript`` echo is display-only;
    relying on it for persistence is what dropped user turns whenever a stale
    client stage flag or a learned-command match swallowed them ("only
    assistant turns saved").

    ``emit=False`` is used for Stage Send: the client already persisted the
    edited text when the user pressed Send, so emitting would double it. The
    companion widget (``becca-ptt.js``) deliberately ignores ``user_committed``
    — it persists server-side via the becca runtime, not the chat tree.
    """
    session.add_user_message(transcript)
    if emit:
        await _send_json(websocket, {"type": "user_committed", "text": transcript})


def _silence_task_exception(task: asyncio.Task[Any]) -> None:
    """Done-callback: consume any pending exception on a fire-and-forget
    background task so asyncio doesn't emit "Task exception was never
    retrieved" warnings. Used for prefetch TTS tasks whose results are
    consumed via the producer/consumer queue on the happy path, but
    which may complete with a transport error before the consumer
    reaches them on abnormal exit (WS disconnect, cancellation).
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        log.debug("voice_bg_task_exception_consumed", error=str(exc))


def _truncate_to_heard(
    full_response: str,
    emitted_sentences: list[str],
    played_count: int,
) -> str:
    """Return only the text the user actually heard before interrupting.

    *emitted_sentences* are the raw (pre-TTS-cleaning) sentence strings in
    emission order.  *played_count* is the number the client reports having
    fully played back.  We rejoin those sentences to reconstruct the heard
    portion of the original response.
    """
    if played_count <= 0 or not emitted_sentences:
        return ""

    heard = emitted_sentences[:played_count]
    return " ".join(heard)


def _apply_pending_heard_rewrite(session: VoiceSession) -> None:
    """Shrink the last assistant message to the heard-only portion.

    Audio-only surfaces (the Becca widget) have no transcript the user
    can read the unspoken tail in, so history must record what was
    heard, not what was generated. But at commit time a barge-in is
    still provisional — a false alarm replays the tail, making the full
    text accurate after all. So the turn commits full text and stashes
    the heard portion in ``session.pending_heard_rewrite``; this applies
    it once a real transcript vindicates the interrupt. An empty heard
    portion means no audio was streamed at all — the message is dropped
    entirely (she never said it).
    """
    heard = session.pending_heard_rewrite
    if heard is None:
        return
    session.pending_heard_rewrite = None
    msgs = session.messages
    if not msgs or msgs[-1].get("role") != "assistant":
        return
    if heard.strip():
        full_chars = len(msgs[-1].get("content") or "")
        msgs[-1]["content"] = heard.strip()
        log.info("voice_interrupt_heard_rewrite",
                 session_id=session.session_id,
                 heard_chars=len(heard.strip()), full_chars=full_chars)
    else:
        msgs.pop()
        log.info("voice_interrupt_unheard_dropped",
                 session_id=session.session_id)


# ---------------------------------------------------------------------------
# Voice Turn Processing
# ---------------------------------------------------------------------------


async def _process_voice_turn(
    audio_bytes: bytes,
    websocket: WebSocket,
    session: VoiceSession,
    app_state: Any,
) -> None:
    """Process one voice turn: STT → LLM → sentence-buffered TTS."""
    conn = None
    sm = getattr(app_state, "state_manager", None)
    if sm:
        from augmentum.state.backends.sqlite import SQLiteBackend
        if isinstance(sm.backend, SQLiteBackend):
            conn = sm.backend.conn

    if not conn:
        await _send_json(websocket, {"type": "error", "message": "Database not available"})
        return

    # --- STT ---
    await _send_json(websocket, {"type": "processing"})

    # Detect audio format for the STT filename hint.
    # Client-side VAD now sends raw PCM16 (from AudioWorklet), not WebM.
    # Legacy MediaRecorder fallback still sends WebM.
    if len(audio_bytes) > 12 and audio_bytes[:4] == b"RIFF":
        _stt_filename = "recording.wav"
    elif len(audio_bytes) > 4 and audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        _stt_filename = "recording.webm"
    else:
        _stt_filename = "recording.pcm"  # raw PCM — no transcode needed

    try:
        transcript = await asyncio.wait_for(
            transcribe_audio(audio_bytes, conn, filename=_stt_filename,
                             user_id=session.user_id),
            timeout=15.0,
        )
    except Exception as exc:
        log.warning("voice_stt_error", error=str(exc))
        await _send_json(websocket, {"type": "error", "message": f"STT failed: {str(exc)[:200]}"})
        return

    if not transcript:
        log.info("voice_stt_empty", session_id=session.session_id, audio_bytes=len(audio_bytes))
        await _send_json(websocket, {"type": "transcript", "text": ""})
        await _send_json(websocket, {"type": "listening"})
        return

    log.info("voice_stt_ok", session_id=session.session_id, chars=len(transcript))
    log.debug("voice_stt_transcript", session_id=session.session_id, transcript=transcript[:100])
    await _send_json(websocket, {"type": "transcript", "text": transcript})

    # Staging: every auto-captured utterance is composition for the box.
    # Skip intent dispatch, the addressing classifier, AND backchannel
    # filtering entirely — opening the stage manager already claimed intent,
    # so a "near-miss / unaddressed" verdict must not drop words the user is
    # clearly dictating. The transcript was already emitted above; fall
    # straight through to the staging wait below.
    if not session.staging:
        # Intent dispatch — control + navigation utterances ("stop", "open
        # browse", "bye becca") short-circuit the LLM via a surface event.
        # Soft-augmentation actions (memory.recall, future referent ops)
        # return an augmented transcript that the LLM receives instead.
        _intent = await _maybe_dispatch_intent(transcript, session, websocket, app_state)
        if _intent.handled:
            return
        transcript = _intent.transcript

        # Filter backchannels and junk transcripts before touching the LLM.
        if is_backchannel(transcript):
            log.debug("voice_backchannel_filtered", transcript=transcript)
            await _send_json(websocket, {"type": "listening"})
            return

    # Staging mode: transcript sent above, wait for client "stage_send"
    if session.staging:
        await _send_json(websocket, {"type": "listening"})
        return

    # Add user message to session history + authoritative user-turn commit
    # (see _commit_user_turn). This is past the intent/backchannel/staging
    # gates, so it is where the turn becomes real conversation. Fold in any
    # recently-dropped-but-held ambient speech so a follow-up can resurrect it.
    transcript = _resurface_held_ambient(session, transcript)
    await _commit_user_turn(websocket, session, transcript)

    # --- LLM ---
    await _send_json(websocket, {"type": "llm_start"})
    session.interrupted = False
    session.is_speaking = True
    # Fresh turn supersedes any pending barge-in recovery (see the
    # mirrored reset in _run_becca_voice_turn).
    session.tts_started = False
    session.bargein_pending = False
    session.undelivered_tts = []
    session.pending_heard_rewrite = None

    # Resolve voice-appropriate tools — shared resolver + voice-safe
    # filter + ambient policy. Becca-path widget gets the policy-filtered
    # set; foreground voice call gets the full universe.
    _ambient = _is_ambient_session(session)
    passthrough_tools = _resolve_voice_tools(
        app_state, session.tools, ambient=_ambient,
    )
    log.info("voice_tools_resolve",
             session_id=session.session_id,
             session_tools=session.tools,
             resolved=passthrough_tools,
             ambient=_ambient,
             path="full_stt")

    # Build voice-aware system instruction
    voice_instruction = _VOICE_BASE_INSTRUCTION
    if passthrough_tools:
        voice_instruction += _VOICE_TOOL_ADDENDUM
        voice_instruction += _voice_capability_addendum(passthrough_tools)
    voice_instruction += _xr_surface_addendum(session.active_xr_surface)
    voice_instruction += _xr_panel_action_addendum(session.active_xr_action)
    voice_instruction += _xr_user_signal_addendum(session.drain_xr_user_signals())

    # Build messages for the LLM
    messages = []
    if session.system_prompt:
        messages.append(Message(
            role="system",
            content=f"{session.system_prompt}\n\n{voice_instruction}",
        ))
    else:
        messages.append(Message(role="system", content=voice_instruction))
    for msg in session.get_recent_messages():
        messages.append(Message(
            role=msg["role"],
            content=msg["content"],
            thinking=msg.get("reasoning_content") or None,
        ))

    # Resolve backend and handler — fabric-aware so peer-only models the
    # operator picked from the voice UI's model selector route correctly.
    mode = _parse_mode(session.mode)
    registry = app_state.provider_registry
    effective_model = session.model if session.model and session.model != "default" else ""
    backend, resolved_model = await registry.resolve_backend_with_fabric(
        effective_model,
        user_id=session.user_id or "",
        session_id=getattr(session, "session_id", "") or "",
    )

    internal_req = InternalChatRequest(
        model=resolved_model or effective_model or "",
        messages=messages,
        stream=True,
    )

    # Manual-mode group voice: drain the one-shot speaker override the
    # client set via PIP tap. Consume + clear so the override only
    # applies to this single turn — without the clear, the user would
    # have to tap PIP every turn just to NOT pin a speaker.
    if session.speaker_override:
        internal_req.speaker_override = session.speaker_override
        log.info("voice_speaker_override_consumed", speaker=session.speaker_override)
        session.speaker_override = ""

    # Inject relevant memories into the system prompt (user context).
    # user_id is REQUIRED — without it the recall falls into the global
    # "default" bucket and mixes memories across tenants on the voice path.
    await recall_and_inject(
        internal_req, app_state,
        user_id=session.user_id,
        mode=session.mode,
        session_id=session.session_id,
    )

    handler = get_handler_for_mode(
        mode=mode,
        backend=backend,
        session_id=session.session_id,
        app_state=app_state,
        passthrough_tools=passthrough_tools,
        tool_synthesis_hint=_VOICE_SYNTHESIS_HINT if passthrough_tools else "",
        user_id=session.user_id,
    )
    # Voice enables tools via a blanket ['all'] sentinel, not per-tool button
    # toggles — so "auto-invoke when enabled" tools (youtube, image_search, …)
    # must NOT auto-fire on every turn (that turned every spoken message into
    # a video search). Keep them available for the LLM to choose; just don't
    # force them. Chat keeps auto-invoke (its toggles ARE the intent signal).
    if hasattr(handler, "_auto_invoke_enabled"):
        handler._auto_invoke_enabled = False
    # Voice input is uncertain (an STT artifact could read as "draw me a…"), so
    # keep image_generation behind a confirmation chip here instead of firing
    # inline. Text chat leaves this False — a typed request is explicit intent.
    if hasattr(handler, "_gate_heavy_tools"):
        handler._gate_heavy_tools = True

    # Stream LLM response, buffer into sentences, TTS each
    tts_chunking = settings.voice_tts_chunking or "sentence"
    sentence_buffer = SentenceBuffer(
        min_chars=settings.voice_sentence_min_chars,
        mode=tts_chunking,
    )
    full_response = ""
    full_thinking = ""  # accumulated reasoning_content for replay on next turn
    emitted_sentences: list[str] = []  # Track sentences sent to TTS
    is_narrative = session.mode == "narrative"
    _narrated_this_turn = False  # Track whether the LLM narrated before tools

    # --- Emotion-aware TTS ---
    _emotion_aware = settings.tts_emotion_aware and is_narrative
    _voice_style = settings.tts_voice_style or ""
    _entity_emotion = ""
    # Resolved once here so the preflight below can use it AND the
    # parallel-pipeline section further down doesn't redefine it.
    _tts_voice = session.character_voice or session.voice
    # Detect if TTS provider uses inline tags (Fish Speech) vs instruct param (Qwen)
    # or Turbo paralinguistic tags ([laugh], [cough], [chuckle])
    _use_inline_tags = False
    _use_turbo_tags = False
    try:
        _pre_provider, _ = await _resolve_tts_provider(conn, _tts_voice)
        if _pre_provider:
            from augmentum.proxy.audio_routes import is_inline_emotion_provider
            _use_inline_tags = is_inline_emotion_provider(_pre_provider)
            _use_turbo_tags = _pre_provider.get("id") == "chatterbox-turbo"
            # Fast local providers chunk on whole sentences — the
            # clause tier's comma seams are audible there (no-op when
            # the operator set a non-default chunking mode).
            sentence_buffer.set_mode(effective_chunking_mode(
                tts_chunking, _pre_provider.get("id", ""),
            ))
    except Exception as exc:
        log.warning("voice_tts_preflight_failed", error=str(exc))
    if _emotion_aware:
        engines = getattr(app_state, "narrative_engines", {})
        _ekey = (session.user_id, session.session_id) if session.user_id else session.session_id
        engine = engines.get(_ekey)
        if engine:
            for entity in engine.state.entities.values():
                if hasattr(entity, "entity_type") and entity.entity_type.value == "character":
                    if entity.state.emotional_state:
                        _entity_emotion = entity.state.emotional_state
                        break

    # --- Parallel LLM + TTS pipeline ---
    # The LLM producer streams tokens and pushes TTS chunks into a queue.
    # The TTS consumer pulls from the queue and streams audio concurrently.
    # This means the LLM keeps generating while TTS is synthesizing,
    # instead of blocking the LLM stream during each TTS call.
    # Queue items are (text, instruct) tuples; None signals end-of-stream.

    _tts_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize=12)
    _tts_speed = session.speed

    async def _llm_producer():
        """Consume LLM stream, push text chunks to TTS queue."""
        nonlocal full_response, full_thinking, _narrated_this_turn, _tts_voice
        from augmentum.training.trace_context import begin_capture, end_capture
        _cap_ctx, _cap_tok = begin_capture(
            user_id=session.user_id or "", session_id=session.session_id or "",
            mode="voice",
        )
        _cap_err = ""
        try:
            async for chunk in handler.handle_stream(internal_req):
                if session.interrupted:
                    break

                aug = getattr(chunk, "augmentum", None) or {}

                # Group chat: per-turn speaker. Forwarded to the client
                # so the avatar viewport auto-swaps to whichever
                # character is about to talk. The voice update is
                # conditional (only fires when the character has a
                # voice set in their card), but the visual swap
                # should always happen — they're orthogonal.
                if aug.get("group_speaker"):
                    new_voice = aug.get("group_speaker_voice") or ""
                    if new_voice and new_voice != session.character_voice:
                        log.info("group_voice_switch",
                                 speaker=aug["group_speaker"],
                                 old_voice=session.character_voice,
                                 new_voice=new_voice)
                        session.character_voice = new_voice
                        _tts_voice = new_voice
                    await _send_json(websocket, {
                        "type": "group_speaker",
                        "speaker": aug["group_speaker"],
                        "voice": new_voice,
                    })

                if aug.get("status"):
                    await _send_json(websocket, {
                        "type": "status", "stage": aug["status"],
                    })

                if aug.get("tool_narration"):
                    narration = aug["tool_narration"]
                    # Show narration in UI text but skip TTS — the tool cue
                    # already notifies the user, and the final LLM response
                    # will cover the same ground. Speaking both is redundant.
                    full_response += narration + " "
                    await _send_json(websocket, {
                        "type": "llm_delta",
                        "text": narration,
                    })

                if aug.get("tool_status"):
                    tool_names = aug.get("tool_names", [])
                    await _send_json(websocket, {
                        "type": "tool_activity",
                        "status": aug["tool_status"],
                        "tools": tool_names,
                    })
                    if not _narrated_this_turn:
                        _narrated_this_turn = True
                        cue = _tool_cue(tool_names)
                        await _tts_queue.put((cue, ""))

                # Chain execution progress — narrated via TTS only. The
                # `chain_status` WS event used to be emitted here too, but
                # no frontend code consumed it; the audit's WS-contract
                # check flagged it as a dead emission. The voice UX
                # surfaces chain progress through spoken cues rather than
                # any visual chain panel, so the emit was pure noise.
                if aug.get("chain"):
                    chain_info = aug["chain"]
                    status = chain_info.get("status", "")
                    if status == "planning" and not _narrated_this_turn:
                        _narrated_this_turn = True
                        await _tts_queue.put(("Let me work through this step by step.", ""))
                    elif status == "synthesizing":
                        await _tts_queue.put(("Putting it all together.", ""))

                if aug.get("chain_step"):
                    step_info = aug["chain_step"]
                    await _send_json(websocket, {"type": "chain_step", **step_info})
                    if step_info.get("status") == "running" and not _narrated_this_turn:
                        _narrated_this_turn = True
                        tool = step_info.get("tool", "")
                        cue = _tool_cue([tool]) if tool else "Working on the next step."
                        await _tts_queue.put((cue, ""))

                if aug.get("tool_call"):
                    tc = aug["tool_call"]
                    msg: dict = {
                        "type": "tool_result",
                        "tool": tc.get("tool", ""),
                        "success": tc.get("success", False),
                        "preview": tc.get("output_preview", "")[:200],
                    }
                    if tc.get("image_url"):
                        msg["image_url"] = tc["image_url"]
                        msg["image_id"] = tc.get("image_id", "")
                    # YouTube tool: surface video metadata so the voice
                    # frontend can render a 3-card picker (search mode) or
                    # hand off to the YouTube panel for playback (direct
                    # mode). Pulled from result_metadata which the handler
                    # builds from the tool's ToolResult.metadata.
                    rmeta = tc.get("result_metadata") or {}
                    if msg["tool"] == "youtube" and rmeta:
                        msg["youtube_mode"] = rmeta.get("youtube_mode", "")
                        if rmeta.get("youtube_mode") == "search":
                            msg["videos"] = rmeta.get("results", [])
                        elif rmeta.get("youtube_mode") == "direct":
                            msg["video"] = {
                                "video_id": rmeta.get("video_id", ""),
                                "title": rmeta.get("title", ""),
                                "channel": rmeta.get("channel", ""),
                                "thumbnail": rmeta.get("thumbnail", ""),
                                "url": rmeta.get("url", ""),
                            }
                    # image_search: forward the result images so the voice
                    # frontend can render a grid AND persist them into the
                    # saved turn. They live in tool metadata, not the spoken
                    # text, so without this they vanish from chat + history.
                    if msg["tool"] == "image_search" and rmeta.get("images"):
                        msg["images"] = rmeta["images"]
                    await _send_json(websocket, msg)

                if chunk.thinking_delta:
                    full_thinking += chunk.thinking_delta

                if chunk.content_delta:
                    full_response += chunk.content_delta
                    await _send_json(websocket, {
                        "type": "llm_delta",
                        "text": chunk.content_delta,
                    })

                    sentence = sentence_buffer.add_token(chunk.content_delta)
                    if sentence:
                        _instruct = ""
                        if _use_turbo_tags:
                            # Turbo: convert *laughs* → [laugh] before cleaning
                            from augmentum.voice.emotion import inject_turbo_tags
                            clean = clean_for_tts(inject_turbo_tags(sentence), is_narrative, preserve_brackets=True)
                        else:
                            clean = clean_for_tts(sentence, is_narrative)
                        if clean:
                            if _emotion_aware:
                                if _use_inline_tags:
                                    # Fish Speech: embed emotion as [tag]text[/tag]
                                    from augmentum.voice.emotion import (
                                        extract_emotion_tag,
                                        wrap_with_emotion_tag,
                                    )
                                    _etag = extract_emotion_tag(sentence, _entity_emotion)
                                    clean = wrap_with_emotion_tag(clean, _etag)
                                else:
                                    # Qwen / instruct-based: separate instruct parameter
                                    from augmentum.voice.emotion import extract_emotion_instruct
                                    _instruct = extract_emotion_instruct(sentence, _entity_emotion)
                            if not _instruct:
                                _instruct = _voice_style
                            emitted_sentences.append(sentence)
                            await _tts_queue.put((clean, _instruct))

            # Flush remaining buffer
            if not session.interrupted:
                remaining = sentence_buffer.flush()
                if remaining:
                    _instruct = ""
                    if _use_turbo_tags:
                        from augmentum.voice.emotion import inject_turbo_tags
                        clean = clean_for_tts(inject_turbo_tags(remaining), is_narrative, preserve_brackets=True)
                    else:
                        clean = clean_for_tts(remaining, is_narrative)
                    if clean:
                        if _emotion_aware:
                            if _use_inline_tags:
                                from augmentum.voice.emotion import (
                                    extract_emotion_tag,
                                    wrap_with_emotion_tag,
                                )
                                _etag = extract_emotion_tag(remaining, _entity_emotion)
                                clean = wrap_with_emotion_tag(clean, _etag)
                            else:
                                from augmentum.voice.emotion import extract_emotion_instruct
                                _instruct = extract_emotion_instruct(remaining, _entity_emotion)
                        if not _instruct:
                            _instruct = _voice_style
                        emitted_sentences.append(remaining)
                        await _tts_queue.put((clean, _instruct))
        except Exception as _exc:
            _cap_err = type(_exc).__name__
            raise
        finally:
            end_capture(_cap_ctx, _cap_tok, error=_cap_err)
            # Signal consumer that production is done
            await _tts_queue.put(None)

    async def _tts_consumer():
        """Pull text chunks from queue, stream TTS audio to client.

        Priority: built-in Kokoro (in-process) → external provider (HTTP).
        Uses lookahead prefetch for external providers.
        """
        # Resolve TTS provider once for the entire turn
        _cached_provider, _ = await _resolve_tts_provider(conn, _tts_voice)

        # Use built-in Kokoro only if it's the resolved provider (not overriding user's choice)
        _kokoro = None
        _use_stream_pcm = False
        _lipsync_engine = getattr(settings, "voice_lipsync_engine", "amplitude")
        if _cached_provider and _cached_provider.get("id") == "kokoro-builtin":
            _ki = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
            if not _ki.is_available:
                await load_model_off_loop(_ki.load_model)
            if _ki.is_available:
                _kokoro = _ki
            if _cached_provider:
                from augmentum.voice.tts import _STREAM_PCM_PROVIDERS
                _use_stream_pcm = _cached_provider.get("id", "") in _STREAM_PCM_PROVIDERS

        _prefetch_task: asyncio.Task | None = None
        _prefetched_text: str = ""
        _prefetched_chunks: list[bytes] = []

        # WAV for engines whose mp3 path buffers/encodes the whole sentence
        # (Qwen, CSM) -- streams chunks instead. See _voice_fmt_for.
        _default_fmt = _voice_fmt_for(_cached_provider)

        # Snapshot for the false-barge-in replay path (mirrors the
        # becca-path snapshot in _run_becca_voice_turn).
        session.tts_params = {
            "voice": _tts_voice,
            "speed": _tts_speed,
            "provider": _cached_provider,
            "format": _default_fmt,
        }

        while True:
            # Check if we have prefetched audio ready
            if _prefetch_task and _prefetch_task.done():
                try:
                    _prefetched_chunks = _prefetch_task.result()
                except Exception:
                    _prefetched_chunks = []
                _prefetch_task = None

            item = await _tts_queue.get()
            if item is None:
                if _prefetch_task:
                    _prefetch_task.cancel()
                break  # Producer finished
            text, instruct = item
            if session.interrupted:
                if _prefetch_task:
                    _prefetch_task.cancel()
                    _prefetch_task = None
                _prefetched_text = ""
                _prefetched_chunks = []
                # Drain remaining items without processing — but keep
                # them so a false barge-in can replay the tail (see
                # _replay_undelivered_tts).
                session.undelivered_tts.append(text)
                continue
            if not session.tts_started:
                _turn_stamp(session, "first_audio")
                _emit_turn_waterfall(session)
            session.tts_started = True

            _tts_fmt = "pcm" if _use_stream_pcm else _default_fmt
            await _send_json(websocket, {"type": "tts_start", "sentence": text, "format": _tts_fmt})

            # Phoneme-schedule emission for external providers (Phase 2). No-op
            # for Kokoro path (which emits its own absolute-time schedule from
            # _stream_kokoro_with_schedule) and when the universal flag is off.
            # Best-effort: never blocks audio, never raises.
            await maybe_emit_normalized_schedule(text, websocket, _cached_provider)

            if _kokoro and _lipsync_engine in ("phoneme", "auto"):
                # Built-in Kokoro + phoneme lip-sync: route through the
                # schedule-aware path (stream_tts_sentence -> Kokoro schedule
                # helper) so a viseme_schedule is emitted before each sentence's
                # audio. Deliberately bypasses the prefetch pipeline: the
                # schedule is keyed to that sentence's exact synth duration, and
                # prefetch_tts_audio is HTTP-only anyway (returns [] for the
                # in-process provider). Amplitude mode uses the fast path below.
                ok = await stream_tts_sentence(
                    text, websocket, conn,
                    voice=_tts_voice,
                    speed=_tts_speed,
                    session=session,
                    provider=_cached_provider,
                    instruct=instruct,
                    stream_pcm=_use_stream_pcm,
                    response_format=_default_fmt,
                )
            elif _kokoro:
                # Built-in Kokoro + amplitude lip-sync: fast in-process
                # generation (no schedule needed — the client drives mouth
                # shapes from the TTS analyser node).
                try:
                    # Strip provider_id prefix from voice if present
                    voice_name = _tts_voice
                    if voice_name and "::" in voice_name:
                        _, voice_name = voice_name.split("::", 1)

                    audio_bytes = await _kokoro.generate(
                        text,
                        voice=voice_name or "af_heart",
                        speed=_tts_speed,
                        response_format=_tts_fmt,
                    )
                    if audio_bytes and not session.interrupted:
                        await websocket.send_bytes(audio_bytes)
                        ok = True
                    elif audio_bytes and session.interrupted:
                        # Audio was generated but user started speaking.
                        # Wait briefly for speech_discard to clear the flag
                        # (false alarm — user didn't actually say anything).
                        for _wait_i in range(10):  # up to 500ms
                            await asyncio.sleep(0.05)
                            if not session.interrupted:
                                break
                        if not session.interrupted:
                            # False alarm resolved — send the buffered audio
                            await websocket.send_bytes(audio_bytes)
                            ok = True
                            log.debug("tts_resumed_after_discard", text_len=len(text))
                        else:
                            ok = False  # real interrupt — discard
                    else:
                        ok = True  # no audio generated (empty text)
                except Exception as exc:
                    log.warning("kokoro_builtin_tts_error", error=str(exc),
                                text_len=len(text))
                    await _send_json(websocket, {
                        "type": "tts_error",
                        "message": "Built-in TTS failed — check voice settings",
                    })
                    ok = False
            else:
                # External provider: HTTP with prefetch pipeline
                # Use prefetched audio if available for this sentence
                if _prefetched_chunks and _prefetched_text == text:
                    log.debug("tts_prefetch_hit", text_len=len(text))
                    ok = await send_prefetched_audio(
                        _prefetched_chunks, websocket, session=session,
                    )
                    _prefetched_text = ""
                    _prefetched_chunks = []
                else:
                    if _prefetched_text and _prefetched_text != text:
                        log.debug("tts_prefetch_miss", expected=_prefetched_text[:30],
                                  got=text[:30])
                    _prefetched_text = ""
                    _prefetched_chunks = []

                    # Peek ahead: start prefetching next sentence concurrently
                    if not _tts_queue.empty() and not _prefetch_task:
                        try:
                            next_item = _tts_queue._queue[0]  # type: ignore[attr-defined]
                            if next_item is not None and not session.interrupted:
                                next_text, next_instruct = next_item
                                _prefetched_text = next_text
                                _prefetch_task = asyncio.create_task(
                                    prefetch_tts_audio(
                                        next_text, conn,
                                        voice=_tts_voice, speed=_tts_speed,
                                        provider=_cached_provider,
                                        instruct=next_instruct,
                                        response_format=_default_fmt,
                                        user_id=session.user_id,
                                        session_id=session.session_id,
                                    )
                                )
                                # Track for guaranteed cleanup on abnormal exit
                                # (WS disconnect, LLM timeout, exception). Without
                                # this, asyncio.create_task spawns a sibling task
                                # that survives parent gather cancellation —
                                # leaks the in-flight prefetch HTTP request,
                                # burning provider quota.
                                _prefetch_task.add_done_callback(_silence_task_exception)
                                _active_bg_tasks.append(_prefetch_task)
                        except (IndexError, AttributeError):
                            pass

                    ok = await stream_tts_sentence(
                        text, websocket, conn,
                        voice=_tts_voice,
                        speed=_tts_speed,
                        session=session,
                        provider=_cached_provider,
                        instruct=instruct,
                        stream_pcm=_use_stream_pcm,
                        response_format=_default_fmt,
                    )

            if not ok and session.interrupted:
                if _prefetch_task:
                    _prefetch_task.cancel()
                    _prefetch_task = None
                _prefetched_text = ""
                _prefetched_chunks = []

    # Registry of background tasks (prefetch TTS HTTP calls) that need
    # guaranteed cancellation if the gather is cancelled abnormally —
    # WS disconnect, LLM timeout, or exception. The consumer's own
    # cancellation paths handle the normal lifecycle; this is the
    # safety net for paths the consumer never reaches.
    _active_bg_tasks: list[asyncio.Task] = []
    # The assistant turn is committed to session history exactly once, from a
    # finally block — so it survives whichever way we leave the gather:
    # normal completion, timeout/error, or task cancellation (client
    # interrupt / barge-in finalize). _commit_turn does no awaiting, so it
    # runs to completion even while a CancelledError is unwinding us.
    _heard_text = ""
    _turn_committed = False

    def _commit_turn() -> None:
        nonlocal _turn_committed, _heard_text
        if _turn_committed:
            return
        _turn_committed = True
        session.is_speaking = False
        session.tts_ended_at = time.monotonic()
        if session.interrupted:
            # Reconstruct what the user actually heard — it rides the
            # "interrupted" event either way. What gets PERSISTED splits
            # by surface:
            #
            # * Audio-only widget (persona becca): no transcript to read
            #   the unspoken tail in — heard IS delivered. Persist the
            #   heard portion (fall back to full when unreconstructable:
            #   interrupt landed before the first sentence finished, or
            #   the client never reported played_sentences — dropping it
            #   silently would just blind the next turn).
            # * Foreground voice call: the chat transcript rendered the
            #   full reply as it streamed, so the text channel delivered
            #   everything even if TTS never finished reading it (code
            #   blocks, long answers). Persist full — the user expects
            #   the whole response visible in chat after the call.
            _heard_text = _truncate_to_heard(
                full_response, emitted_sentences, session.played_sentences,
            )
            if _is_ambient_session(session):
                persisted = _heard_text or full_response
            else:
                persisted = full_response
            if persisted:
                session.add_assistant_message(persisted, thinking=full_thinking)
                log.info(
                    "voice_interrupt_persisted",
                    total_sentences=len(emitted_sentences),
                    played=session.played_sentences,
                    heard_chars=len(_heard_text),
                    persisted_chars=len(persisted),
                    full_chars=len(full_response),
                    fell_back=not _heard_text,
                )
            session.played_sentences = 0
        elif full_response:
            session.add_assistant_message(full_response, thinking=full_thinking)

    try:
        # Run LLM and TTS concurrently with overall timeout
        await asyncio.wait_for(
            asyncio.gather(_llm_producer(), _tts_consumer()),
            timeout=settings.request_timeout,
        )
    except TimeoutError:
        log.warning("voice_llm_timeout", timeout=settings.request_timeout)
        await _send_json(websocket, {"type": "error", "message": "Response timed out"})
    except Exception as exc:
        log.warning("voice_llm_error", error=str(exc))
        msg = str(exc)
        if "429" in msg:
            friendly = "The model provider is rate-limiting requests. Please wait a moment."
        elif "503" in msg or "502" in msg:
            friendly = "The model backend is temporarily unavailable."
        else:
            friendly = f"LLM error: {msg[:200]}"
        await _send_json(websocket, {"type": "error", "message": friendly})
    finally:
        # Cancel any prefetch tasks that escaped the consumer's normal
        # cleanup. cancel() on already-done/cancelled tasks is a no-op,
        # so this is safe to call unconditionally. Don't await — our
        # own task may be in cancellation state; the event loop will
        # drive each cancelled task to its CancelledError finalization
        # which aborts the underlying httpx request.
        for _bg in _active_bg_tasks:
            if not _bg.done():
                _bg.cancel()
        # Commit the assistant turn before any cancellation finishes
        # unwinding us — this is the only place the turn is persisted.
        _commit_turn()

    if session.interrupted:
        await _send_json(websocket, {
            "type": "interrupted",
            "heard_text": _heard_text,
        })
    else:
        await _send_json(websocket, {"type": "tts_end"})
        # Flush any LLM-invoked surface events (sticky-note opens,
        # navigation, etc.) before announcing turn completion so the
        # UI updates land in the natural order.
        await _flush_pending_surface_events(
            websocket, app_state, session.user_id, session.session_id,
        )
        await _send_json(websocket, {
            "type": "turn_complete",
            "full_text": full_response,
        })

    await _send_json(websocket, {"type": "listening"})


def _state_conn(app_state: Any):
    """Return the shared SQLite connection from app state, or None."""
    sm = getattr(app_state, "state_manager", None)
    if sm:
        from augmentum.state.backends.sqlite import SQLiteBackend
        if isinstance(sm.backend, SQLiteBackend):
            return sm.backend.conn
    return None


async def _companion_voice_for_user(app_state: Any, user_id: str) -> str:
    """Resolve the companion (Becca) TTS voice for ``user_id``.

    Priority: per-user ``ui.companionVoice`` (Companion tab) →
    per-user ``ui.voiceDefaultVoice`` (the app-level default voice) →
    "" (caller falls through to the DB default TTS provider).

    The companion widget never sends a ``voice`` over its WS config —
    without this, Becca sessions silently fall back to whichever
    provider row happens to sort first in ``audio_providers``,
    ignoring the user's voice settings entirely.
    """
    if not user_id:
        return ""
    store = getattr(app_state, "settings_store", None)
    if store is None:
        return ""
    try:
        for key in ("ui.companionVoice", "ui.voiceDefaultVoice"):
            # Per-user ONLY — a global fallback would speak in the
            # OWNER's chosen voice for every user, and voices are never
            # auto-selected on a user's behalf.
            value = await store.get_user(user_id, key)
            if value and value.strip():
                return value.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "companion_voice_resolve_failed",
            user_id=user_id,
            error=str(exc)[:200],
        )
    return ""


async def _replay_undelivered_tts(
    websocket: WebSocket,
    session: VoiceSession,
    app_state: Any,
) -> bool:
    """Re-stream TTS sentences that a false barge-in drained.

    When a VAD blip confirms a barge-in mid-turn, the TTS consumer
    drains the rest of the reply into ``session.undelivered_tts`` while
    the turn still commits its full text to history — so the assistant
    "remembers" saying something the user never heard. Once the blip is
    exposed as noise (the segment is discarded, or a real speech_end
    produces an empty transcript), this replays the drained sentences
    using the turn's snapshotted TTS parameters.

    Returns True if at least one sentence was streamed.
    """
    sentences = [s for s in (session.undelivered_tts or []) if s.strip()]
    session.undelivered_tts = []
    if not sentences:
        # Nothing was drained — every sentence streamed before the
        # interrupt flag rose, so the committed full text is accurate
        # and any deferred heard-rewrite is moot.
        session.pending_heard_rewrite = None
        return False
    conn = _state_conn(app_state)
    if conn is None:
        log.warning("voice_tts_replay_no_db", session_id=session.session_id)
        return False
    params = session.tts_params or {}
    log.info("voice_tts_replay_after_false_bargein",
             session_id=session.session_id, sentences=len(sentences))
    session.interrupted = False
    session.is_speaking = True
    replayed = False
    delivered: list[str] = []
    try:
        for idx, sentence in enumerate(sentences):
            if session.interrupted:
                # Re-interrupted mid-replay — keep the tail so a second
                # false alarm can resume instead of losing it.
                session.undelivered_tts = sentences[idx:]
                break
            try:
                await _send_json(websocket, {
                    "type": "tts_start", "sentence": sentence,
                    "format": params.get("format") or settings.voice_tts_format,
                })
                await stream_tts_sentence(
                    sentence, websocket, conn,
                    voice=params.get("voice") or session.voice,
                    speed=params.get("speed") or session.speed,
                    session=session,
                    provider=params.get("provider"),
                )
                await _send_json(websocket, {"type": "tts_end"})
                replayed = True
                delivered.append(sentence)
            except Exception as exc:
                log.warning("voice_tts_replay_failed",
                            session_id=session.session_id,
                            error=str(exc)[:200])
                # Keep the unstreamed tail (including the failed
                # sentence) so the reconciliation below treats this as
                # partial delivery, not success.
                session.undelivered_tts = sentences[idx:]
                break
    finally:
        session.is_speaking = False
        session.tts_ended_at = time.monotonic()

    # Reconcile the audio-only deferred heard-rewrite with what this
    # replay actually delivered. Fully drained → the original full-text
    # commit is accurate, drop the rewrite. Partially delivered (re-
    # interrupt / TTS failure) → extend the heard portion by the
    # delivered sentences (TTS-cleaned text — close enough to the
    # original prose for voice context) so a later vindication shrinks
    # history to the true heard boundary.
    if session.pending_heard_rewrite is not None:
        if session.undelivered_tts:
            if delivered:
                session.pending_heard_rewrite = (
                    f"{session.pending_heard_rewrite} " + " ".join(delivered)
                ).strip()
        else:
            session.pending_heard_rewrite = None
    return replayed


def _voice_fmt_for(provider: dict | None) -> str:
    """Audio format to request on the voice path. Prefer 'wav' for engines
    whose mp3 path buffers the whole sentence server-side: Qwen's optimized
    backend and the CSM sidecar (which ffmpeg-encodes compressed formats in
    one shot, so mp3 = wait for the full sentence). wav uses their native
    chunk-streaming path -- first audio arrives as it decodes. Other engines
    keep the configured format: Kokoro/Pocket already stream mp3 per-chunk,
    so there's nothing to gain and mp3 saves bandwidth on remote access."""
    if provider:
        from augmentum.proxy.audio_routes import _is_csm_provider, _is_qwen_provider
        if _is_qwen_provider(provider) or _is_csm_provider(provider.get("id", "")):
            return "wav"
    return settings.voice_tts_format


async def _run_becca_voice_turn(
    transcript: str,
    websocket: WebSocket,
    session: VoiceSession,
    app_state: Any,
    conn,
) -> bool:
    """Route this voice turn through BeccaVoice instead of the legacy
    mode-handler chain. Returns True if Becca handled the turn; False if
    the caller should fall through to the legacy path.

    PTT Stage 3. Becca's pipeline composes her own system prompt (the
    persona kernel + facets + relationship + tools + a voice-channel
    addendum) and streams tokens through ``compose_becca_prompt`` →
    primary tier → ``TagSieve``. We catch her ``out.write`` chunks here,
    feed them into the existing ``SentenceBuffer`` + ``stream_tts_sentence``
    pipeline so audio starts before the full response lands. Tool calls
    + channel handoffs route through her runtime; the voice loop just
    surfaces the prose she emits.

    Fall-through reasons: no CompanionRuntime on app.state, the runtime
    has no persona-kernel digest (so ``BeccaVoice`` would raise
    ``BeccaBypassed`` anyway), or an unhandled BeccaBypassed during the
    stream. In all cases ``session.messages`` is left intact so the
    legacy path can re-process from the same transcript.
    """
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None:
        log.info("becca_voice_no_runtime", session_id=session.session_id)
        return False
    digest = getattr(getattr(runtime, "identity", None), "persona_kernel_digest", "") or ""
    if not digest.strip() or not getattr(runtime, "_started", False):
        log.info(
            "becca_voice_runtime_not_ready",
            session_id=session.session_id,
            has_digest=bool(digest.strip()),
            started=getattr(runtime, "_started", False),
        )
        return False

    # Lazy import — keeps cold path light for non-Becca sessions.
    from augmentum.companion_runtime.runtime import Intent
    from augmentum.companion_runtime.voice import BeccaBypassed, BeccaVoice

    # Build Intent: the freshly-transcribed turn is the user text;
    # session.messages provides recent_turns (we already appended the
    # current turn via add_user_message, so slice it off).
    history = list(session.get_recent_messages())
    if history and history[-1].get("role") == "user" and history[-1].get("content") == transcript:
        history = history[:-1]
    recent_turns = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in history
    ]

    # Late router decision — a still-running route_utterance task from
    # the soft-timeout path. The act-gap consumer in BeccaVoice awaits
    # it and dispatches the completed decision when the persona path
    # produced no action. Popped here so a stale task never leaks into
    # the next turn.
    _router_task = getattr(session, "pending_router_task", None)
    try:
        session.pending_router_task = None
    except Exception:  # noqa: BLE001
        log.debug("router_task_clear_failed", exc_info=True)

    # Live vision: hand the freshest camera frame(s) to this turn so the
    # companion can SEE what the user is showing. Consumed once + cleared
    # (a stale frame from a prior glance never bleeds into a later turn),
    # and dropped entirely if older than the staleness window. The vision
    # pipeline downstream routes them VL-direct or sibling-captioned.
    _turn_frames: list[str] = []
    if (
        session.latest_frames
        and (time.monotonic() - session.latest_frame_ts) <= _LIVE_VISION_STALE_S
    ):
        _turn_frames = list(session.latest_frames)
    session.latest_frames = []
    log.info(
        "becca_turn_frames",
        session_id=session.session_id,
        n_frames=len(_turn_frames),
        had_buffered=bool(session.latest_frames) or len(_turn_frames) > 0,
        age_s=(round(time.monotonic() - session.latest_frame_ts, 2)
               if session.latest_frame_ts else None),
    )

    intent = Intent(
        text=transcript,
        user_id=session.user_id,
        source="user_chat",
        device_id="",
        explicit_mode="",
        metadata={
            "recent_turns": recent_turns,
            "session_id": session.session_id,
            "images": _turn_frames,  # live-vision frames for this turn
            # These came from the live camera buffer (video_frame WS), so the
            # vision pipeline + prompt layer frame them as a REAL live feed,
            # not an incidental/fictional image the model can call "fake".
            "live_camera": bool(_turn_frames),
            "voice_channel": True,  # consumed by prompt_compose (Layer 8.5)
            # Address-classifier goal for this turn ("act" switches the
            # tool roster into act mode — see prompt_compose Layer 6).
            "router_goal": getattr(session, "last_router_goal", "") or "",
            "router_task": _router_task,
        },
    )

    # TTS pipeline setup — mirrors the legacy turn handler. One queue;
    # producer (BeccaVoice writer) puts plain sentences; consumer
    # streams each through ``stream_tts_sentence``.
    _voice = session.character_voice or session.voice
    _speed = session.speed
    _provider, _voice_resolved = await _resolve_tts_provider(conn, _voice)
    # Stream-friendly format (wav for CSM/Qwen so the sidecar streams chunks
    # instead of buffering + mp3-encoding the whole sentence).
    _becca_fmt = _voice_fmt_for(_provider)

    # Cross-speaker CSM context: if her voice is CSM and we captured the
    # user's utterance this turn, hand it to the sidecar so her prosody
    # reacts to how they sounded. Fire-and-forget — the push lands in ~ms
    # while the LLM is still composing, well before the first sentence
    # synthesizes. No-op for non-CSM voices. Inverse: the
    # ``companion_csm_cross_speaker`` setting (default on).
    if (
        getattr(settings, "companion_csm_cross_speaker", True)
        and session.last_user_audio
    ):
        from augmentum.proxy.audio_routes import push_user_context
        _user_clip = session.last_user_audio
        _user_clip_sr = session.last_user_audio_sr
        session.last_user_audio = b""  # consume once, per turn
        _ctx_task = asyncio.create_task(push_user_context(
            provider=_provider,
            session_id=session.session_id,
            pcm_audio=_user_clip,
            sample_rate=_user_clip_sr,
            transcript=transcript,
            user_id=session.user_id,
        ))
        _ctx_task.add_done_callback(_silence_task_exception)

    # Cross-modal mood -> voice: tag her fine-tuned CSM voice with her current
    # affect so she *sounds* how she feels. Recency note: _last_affect_tag is
    # the last published change; applied as-is for now and gated off by default
    # (companion_csm_emotion_tag) — enable + tune the mapping by ear once a
    # fine-tuned voice is live. No-op for non-CSM voices.
    session.csm_emotion = ""
    if getattr(settings, "companion_csm_emotion_tag", False):
        from augmentum.proxy.audio_routes import _is_csm_provider
        if _is_csm_provider((_provider or {}).get("id", "")):
            from augmentum.voice.companion_emotion import emotion_for_affect
            session.csm_emotion = emotion_for_affect(getattr(runtime, "_last_affect_tag", ""))

    # Same idea for OpenAI-omni style voices (Higgs Audio v3 via the generic
    # openai-tts provider): stash her recency-gated affect so the TTS layer can
    # render a natural style cue — (warm)/(excited)/(gentle). Stored raw here;
    # tts.py maps it to the provider's vocabulary so the design stays
    # model-agnostic. Gated by companion_voice_emotion_tag (default off).
    session.voice_affect = ""
    if getattr(settings, "companion_voice_emotion_tag", False):
        session.voice_affect = str(getattr(runtime, "_last_affect_tag", "") or "")

    # Snapshot for the false-barge-in replay path — it streams outside
    # this turn's scope and must not re-run provider resolution.
    session.tts_params = {
        "voice": _voice_resolved,
        "speed": _speed,
        "provider": _provider,
        "format": _becca_fmt,
    }

    sentence_buffer = SentenceBuffer(
        min_chars=settings.voice_sentence_min_chars,
        # Provider already resolved above — fast local providers get
        # whole-sentence chunking (no comma seams in her voice).
        mode=effective_chunking_mode(
            settings.voice_tts_chunking,
            (_provider or {}).get("id", ""),
        ),
    )
    tts_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=12)
    full_response_parts: list[str] = []
    # Raw (pre-TTS-cleaning) sentences: every sentence emitted by the
    # producer, and the subset whose TTS audio was actually streamed to
    # the client — so an interrupted audio-only turn can record what
    # was heard. "Streamed" is a server-side proxy for played (the
    # widget doesn't report played_sentences); it can overcount by the
    # one sentence in flight when the interrupt landed, never
    # undercount.
    emitted_sentences: list[str] = []
    heard_sentences: list[str] = []

    await _send_json(websocket, {"type": "llm_start"})
    session.interrupted = False
    session.is_speaking = True
    # Fresh turn supersedes any pending barge-in recovery: a real
    # transcript arrived, so the prior interrupt was legitimate and the
    # old turn's undelivered tail must not replay over this reply.
    session.tts_started = False
    session.bargein_pending = False
    session.undelivered_tts = []
    session.pending_heard_rewrite = None

    class _BeccaWriter:
        """Adapter — BeccaVoice writes prose; we buffer into sentences
        for the TTS consumer. ``write_control`` is a no-op here: the
        voice surface doesn't render channel-handoff UI today (that
        path stays in the chat surface).
        """
        async def write(self, text: str) -> None:
            if not text:
                return
            full_response_parts.append(text)
            try:
                await _send_json(websocket, {"type": "llm_delta", "text": text})
            except Exception:
                log.debug("voice_ws_llm_delta_send_failed", exc_info=True)
            sentence = sentence_buffer.add_token(text)
            if sentence:
                emitted_sentences.append(sentence)
                await tts_q.put(sentence)

        async def write_control(self, control: dict[str, Any]) -> None:
            # Surface channel-handoff intent as a UI hint but don't TTS
            # the control payload. Becca tends to write a prose opener
            # before the tag fires, so the user already hears context.
            try:
                await _send_json(websocket, {
                    "type": "becca_control",
                    "control": control,
                })
            except Exception:
                log.debug("voice_ws_control_send_failed", exc_info=True)

        async def close(self) -> None:
            remaining = sentence_buffer.flush()
            if remaining:
                emitted_sentences.append(remaining)
                await tts_q.put(remaining)
            await tts_q.put(None)

        def as_text(self) -> str:
            return "".join(full_response_parts)

    writer = _BeccaWriter()
    is_narrative = session.mode == "narrative"

    async def _tts_consumer():
        while True:
            item = await tts_q.get()
            if item is None:
                break
            clean = clean_for_tts(item, is_narrative)
            if not clean:
                continue
            if session.interrupted:
                # Drain, but keep what was drained: if the interrupt
                # turns out to be a false barge-in (segment discarded or
                # empty STT), _replay_undelivered_tts streams these so
                # the reply that history records as spoken is actually
                # heard.
                session.undelivered_tts.append(clean)
                continue
            if not session.tts_started:
                _turn_stamp(session, "first_audio")
                _emit_turn_waterfall(session)
            session.tts_started = True
            try:
                await _send_json(websocket, {
                    "type": "tts_start", "sentence": clean,
                    "format": _becca_fmt,
                })
            except Exception:
                return
            try:
                await stream_tts_sentence(
                    clean, websocket, conn,
                    voice=_voice_resolved, speed=_speed,
                    session=session, provider=_provider,
                    response_format=_becca_fmt,
                )
            except Exception as exc:
                log.warning("becca_voice_tts_failed", error=str(exc))
                continue
            heard_sentences.append(item)
            try:
                await _send_json(websocket, {"type": "tts_end"})
            except Exception:
                return

    async def _becca_producer():
        from augmentum.training.trace_context import begin_capture, end_capture
        _cap_ctx, _cap_tok = begin_capture(
            user_id=session.user_id or "", session_id=session.session_id or "",
            mode="voice",
        )
        _cap_err = ""
        try:
            becca = BeccaVoice(runtime)
            await becca.stream(intent, out=writer)
        except BeccaBypassed as exc:
            _cap_err = "BeccaBypassed"
            log.info("becca_voice_bypassed", reason=exc.reason)
            raise  # propagate so the outer try sees it and falls through
        except Exception:
            _cap_err = "producer_crashed"
            log.exception("becca_voice_producer_crashed")
        finally:
            end_capture(_cap_ctx, _cap_tok, error=_cap_err)
            await writer.close()

    handled = True
    try:
        await asyncio.gather(_becca_producer(), _tts_consumer())
    except BeccaBypassed:
        # Drain the TTS consumer if it's still queued so we don't leak.
        try:
            tts_q.put_nowait(None)
        except asyncio.QueueFull:
            pass
        handled = False
    finally:
        session.is_speaking = False
        session.tts_ended_at = time.monotonic()

    if not handled:
        return False

    full_response = writer.as_text()
    if full_response.strip():
        session.add_assistant_message(full_response)
        if session.interrupted:
            # Audio-only surface: history must eventually record what
            # was heard, not what was generated — there's no transcript
            # to read the unspoken tail in. But the interrupt is still
            # provisional: a false barge-in replays the tail, making
            # the full text accurate after all. So commit full now and
            # stash the heard-only portion; vindication (a real
            # transcript in _finalize_speech) applies the rewrite, a
            # completed replay discards it.
            heard = " ".join(heard_sentences).strip()
            session.pending_heard_rewrite = heard
            log.info(
                "voice_becca_interrupt_committed",
                session_id=session.session_id,
                total_sentences=len(emitted_sentences),
                heard_sentences=len(heard_sentences),
                heard_chars=len(heard),
                full_chars=len(full_response),
            )

    try:
        await _flush_pending_surface_events(
            websocket, app_state, session.user_id, session.session_id,
        )
        await _send_json(websocket, {
            "type": "turn_complete", "full_text": full_response,
        })
        await _send_json(websocket, {"type": "listening"})
    except Exception:
        log.debug("voice_ws_turn_complete_send_failed", exc_info=True)
    return True


async def _process_voice_turn_from_transcript(
    transcript: str,
    websocket: WebSocket,
    session: VoiceSession,
    app_state: Any,
    *,
    from_stage_send: bool = False,
) -> None:
    """Process a voice turn where STT already happened (server-side VAD path).

    Identical to ``_process_voice_turn`` but skips the STT step since
    the streaming STT (or batch fallback) already produced the transcript.

    Args:
        from_stage_send: If True, bypass the staging guard (the user already
            edited the text and pressed Send in the staging UI).
    """
    if not transcript:
        # A confirmed barge-in whose audio transcribed to nothing was a
        # false alarm — the interrupt cancelled (or muted) a reply over
        # noise. Un-interrupt and replay whatever the TTS consumer
        # drained so the turn the user triggered actually gets heard.
        if session.bargein_pending:
            session.bargein_pending = False
            session.interrupted = False
            log.info("voice_false_bargein_empty_stt",
                     session_id=session.session_id)
            try:
                replayed = await _replay_undelivered_tts(
                    websocket, session, app_state,
                )
            except Exception:
                log.exception("voice_tts_replay_crashed",
                              session_id=session.session_id)
                replayed = False
            if replayed:
                return

        # Both streaming and the batch fallback returned nothing — this
        # is a genuine "I didn't catch that" moment. Surface it as a
        # visible signal instead of going silent so the user knows to
        # retry rather than wait.
        log.info("voice_stt_empty_streaming", session_id=session.session_id)
        await _send_json(websocket, {
            "type": "voice_no_speech",
            "reason": "stt_empty",
            "message": "I didn't catch that — try again?",
        })
        await _send_json(websocket, {"type": "transcript", "text": ""})
        await _send_json(websocket, {"type": "listening"})
        return

    log.info("voice_stt_ok", session_id=session.session_id, chars=len(transcript))
    log.debug("voice_stt_transcript", session_id=session.session_id, transcript=transcript[:100])

    # Stage Send: skip transcript echo — client already has the text.
    # Go straight to LLM processing without flashing through listening state.
    if from_stage_send:
        await _send_json(websocket, {"type": "processing"})
    else:
        await _send_json(websocket, {"type": "transcript", "text": transcript})

    # Intent dispatch — control + navigation utterances bypass the LLM.
    # On the actual Send (from_stage_send) this still runs so explicit
    # user-typed commands like "open browse" short-circuit. But during
    # DRAFTING (staging & not send) it is skipped entirely: the addressing
    # classifier must not drop a dictated utterance the user is clearly
    # composing (they claimed intent by opening the stage manager).
    if not (session.staging and not from_stage_send):
        _intent = await _maybe_dispatch_intent(
            transcript, session, websocket, app_state,
            from_stage_send=from_stage_send,
        )
        if _intent.handled:
            return
        transcript = _intent.transcript

    if not from_stage_send and is_backchannel(transcript):
        log.debug("voice_backchannel_filtered", transcript=transcript)
        await _send_json(websocket, {"type": "listening"})
        return

    # Staging mode: transcript already sent, wait for client "stage_send"
    if session.staging and not from_stage_send:
        await _send_json(websocket, {"type": "listening"})
        return

    # Authoritative user-turn commit (server-VAD / streaming path). Routed
    # through the shared helper so the VAD and streaming paths can't drift.
    # Stage Send suppresses the emit — the client already persisted the
    # edited text when the user pressed Send, so emitting would double it.
    # Fold in any recently-dropped-but-held ambient speech so a follow-up
    # (spoken or a stage-manager Send) can resurrect it.
    transcript = _resurface_held_ambient(session, transcript)
    await _commit_user_turn(websocket, session, transcript, emit=not from_stage_send)

    # From here on, identical to _process_voice_turn after STT.
    conn = None
    sm = getattr(app_state, "state_manager", None)
    if sm:
        from augmentum.state.backends.sqlite import SQLiteBackend
        if isinstance(sm.backend, SQLiteBackend):
            conn = sm.backend.conn

    if not conn:
        await _send_json(websocket, {"type": "error", "message": "Database not available"})
        return

    # PTT Stage 3 — route through BeccaVoice when persona_id=becca on
    # the WS query. Falls through to the legacy mode-handler chain when
    # the runtime isn't ready or BeccaVoice bypasses.
    if session.persona_id == "becca":
        try:
            handled = await _run_becca_voice_turn(
                transcript, websocket, session, app_state, conn,
            )
        except Exception:
            log.exception("becca_voice_turn_crashed",
                          session_id=session.session_id)
            handled = False
        if handled:
            return
        log.info("becca_voice_fallthrough_to_legacy",
                 session_id=session.session_id)

    _ambient = _is_ambient_session(session)
    passthrough_tools = _resolve_voice_tools(
        app_state, session.tools, ambient=_ambient,
    )
    log.info("voice_tools_resolve",
             session_id=session.session_id,
             session_tools=session.tools,
             resolved=passthrough_tools,
             ambient=_ambient,
             path="from_transcript")

    voice_instruction = _VOICE_BASE_INSTRUCTION
    if passthrough_tools:
        voice_instruction += _VOICE_TOOL_ADDENDUM
        voice_instruction += _voice_capability_addendum(passthrough_tools)
    voice_instruction += _xr_surface_addendum(session.active_xr_surface)
    voice_instruction += _xr_panel_action_addendum(session.active_xr_action)
    voice_instruction += _xr_user_signal_addendum(session.drain_xr_user_signals())

    messages = []
    if session.system_prompt:
        messages.append(Message(
            role="system",
            content=f"{session.system_prompt}\n\n{voice_instruction}",
        ))
    else:
        messages.append(Message(role="system", content=voice_instruction))
    for msg in session.get_recent_messages():
        messages.append(Message(
            role=msg["role"],
            content=msg["content"],
            thinking=msg.get("reasoning_content") or None,
        ))

    mode = _parse_mode(session.mode)
    registry = app_state.provider_registry
    effective_model = session.model if session.model and session.model != "default" else ""
    backend, resolved_model = await registry.resolve_backend_with_fabric(
        effective_model,
        user_id=session.user_id or "",
        session_id=getattr(session, "session_id", "") or "",
    )

    internal_req = InternalChatRequest(
        model=resolved_model or effective_model or "",
        messages=messages,
        stream=True,
    )

    # Live vision (call-surface camera): attach the freshest frame(s) to
    # this turn so the companion sees what the user is showing during a
    # call, then run the same pipeline the chat/becca paths use (VL-direct
    # or sibling-captioned, reasoning unlocked for the response). This is
    # the path passthrough/narrative calls take; the becca path consumes
    # frames itself. Consumed once + cleared; dropped if stale.
    _call_frames: list[str] = []
    if (
        session.latest_frames
        and (time.monotonic() - session.latest_frame_ts) <= _LIVE_VISION_STALE_S
    ):
        _call_frames = list(session.latest_frames)[:_LIVE_VISION_MAX_FRAMES]
    session.latest_frames = []
    if _call_frames:
        for _m in reversed(internal_req.messages):
            if _m.role == "user":
                _m.images = _call_frames
                break
        try:
            from augmentum.models.base import (
                apply_vision_pipeline,
                ensure_live_camera_framing,
            )
            await apply_vision_pipeline(
                internal_req, app_state, backend,
                reason_on_vision=True, live_camera=True,
            )
            # This passthrough/narrative path does NOT compose a companion
            # prompt, so prompt_compose Layer 8.6 never runs. Inject the
            # live-camera reality anchor as a system message so a VISION-
            # capable primary (which reads the frames directly, with no
            # caption label) still knows it's seeing a real object, not
            # fiction. No-op-safe for the text-only path (idempotent).
            ensure_live_camera_framing(internal_req)
        except Exception as exc:  # noqa: BLE001 — vision never breaks a call turn
            log.warning("voice_call_vision_pipeline_failed", error=str(exc))

    # Drain the one-shot manual-mode speaker override (mirrors the
    # voice-turn path at the top of this file).
    if session.speaker_override:
        internal_req.speaker_override = session.speaker_override
        log.info("voice_speaker_override_consumed", speaker=session.speaker_override)
        session.speaker_override = ""

    tts_chunking = settings.voice_tts_chunking or "sentence"
    sentence_buffer = SentenceBuffer(
        min_chars=settings.voice_sentence_min_chars,
        mode=tts_chunking,
    )
    full_response = ""
    full_thinking = ""  # accumulated reasoning_content for replay on next turn
    emitted_sentences: list[str] = []
    is_narrative = session.mode == "narrative"
    _narrated_this_turn = False

    # --- Emotion-aware TTS ---
    _emotion_aware2 = settings.tts_emotion_aware and is_narrative
    _voice_style2 = settings.tts_voice_style or ""
    _entity_emotion2 = ""
    _use_turbo_tags2 = False
    _use_inline_tags2 = False
    try:
        _pp2, _ = await _resolve_tts_provider(
            conn, session.character_voice or session.voice,
        )
        if _pp2:
            from augmentum.proxy.audio_routes import is_inline_emotion_provider
            _use_turbo_tags2 = _pp2.get("id") == "chatterbox-turbo"
            _use_inline_tags2 = is_inline_emotion_provider(_pp2)
            sentence_buffer.set_mode(effective_chunking_mode(
                tts_chunking, _pp2.get("id", ""),
            ))
    except Exception as exc:
        log.warning(
            "voice_tts_preflight_failed",
            path="from_transcript",
            error=str(exc),
        )
    if _emotion_aware2:
        engines = getattr(app_state, "narrative_engines", {})
        _ekey2 = (session.user_id, session.session_id) if session.user_id else session.session_id
        engine = engines.get(_ekey2)
        if engine:
            for entity in engine.state.entities.values():
                if hasattr(entity, "entity_type") and entity.entity_type.value == "character":
                    if entity.state.emotional_state:
                        _entity_emotion2 = entity.state.emotional_state
                        break

    # Registry of background prefetch tasks needing guaranteed cleanup —
    # see matching declaration in _process_voice_turn.
    _active_bg_tasks: list[asyncio.Task] = []
    # Commit the assistant turn exactly once, from a finally — see the
    # matching helper in _process_voice_turn for the full rationale.
    _heard_text = ""
    _turn_committed = False

    def _commit_turn() -> None:
        nonlocal _turn_committed, _heard_text
        if _turn_committed:
            return
        _turn_committed = True
        session.is_speaking = False
        session.tts_ended_at = time.monotonic()
        if session.interrupted:
            _heard_text = _truncate_to_heard(
                full_response, emitted_sentences, session.played_sentences,
            )
            persisted = _heard_text or full_response
            if persisted:
                session.add_assistant_message(persisted, thinking=full_thinking)
                log.info(
                    "voice_interrupt_persisted",
                    total_sentences=len(emitted_sentences),
                    played=session.played_sentences,
                    heard_chars=len(_heard_text),
                    persisted_chars=len(persisted),
                    full_chars=len(full_response),
                    fell_back=not _heard_text,
                )
            session.played_sentences = 0
        elif full_response:
            session.add_assistant_message(full_response, thinking=full_thinking)

    try:
        async def _recall_and_stream():
            nonlocal full_response, full_thinking, _narrated_this_turn

            # Memory recall (may involve embedding computation).
            # user_id required to keep recall scoped to this tenant — see
            # the matching call above for rationale.
            log.info("voice_pre_recall", session_id=session.session_id)
            await recall_and_inject(
                internal_req, app_state,
                user_id=session.user_id,
                mode=session.mode,
                session_id=session.session_id,
            )
            log.info("voice_post_recall", session_id=session.session_id)

            handler = get_handler_for_mode(
                mode=mode,
                backend=backend,
                session_id=session.session_id,
                app_state=app_state,
                passthrough_tools=passthrough_tools,
                tool_synthesis_hint=_VOICE_SYNTHESIS_HINT if passthrough_tools else "",
                user_id=session.user_id,
            )
            # Blanket ['all'] tool enable on voice → don't auto-fire
            # auto-invoke tools (youtube/etc.) every turn; let the LLM choose.
            # (Mirrors the _process_voice_turn path above.)
            if hasattr(handler, "_auto_invoke_enabled"):
                handler._auto_invoke_enabled = False
            # Uncertain STT input → keep image_generation behind a confirm chip
            # (mirrors the _process_voice_turn path above).
            if hasattr(handler, "_gate_heavy_tools"):
                handler._gate_heavy_tools = True

            await _send_json(websocket, {"type": "llm_start"})
            session.interrupted = False
            session.is_speaking = True

            # Parallel LLM + TTS — same pattern as _process_voice_turn
            # Queue items are (text, instruct) tuples; None signals end-of-stream.
            _tts_q: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize=12)
            _voice = session.character_voice or session.voice
            _speed = session.speed

            async def _llm_prod():
                nonlocal full_response, full_thinking, _narrated_this_turn, _voice
                try:
                    async for chunk in handler.handle_stream(internal_req):
                        if session.interrupted:
                            break

                        aug = getattr(chunk, "augmentum", None) or {}

                        # Group chat: per-turn speaker. Visual swap is
                        # forwarded unconditionally; voice update only
                        # fires when the character has an assigned voice.
                        # Mirrors the main voice-turn path at the top.
                        if aug.get("group_speaker"):
                            new_voice = aug.get("group_speaker_voice") or ""
                            if new_voice and new_voice != session.character_voice:
                                log.info("group_voice_switch",
                                         speaker=aug["group_speaker"],
                                         old_voice=session.character_voice,
                                         new_voice=new_voice)
                                session.character_voice = new_voice
                                _voice = new_voice
                            await _send_json(websocket, {
                                "type": "group_speaker",
                                "speaker": aug["group_speaker"],
                                "voice": new_voice,
                            })

                        if aug.get("status"):
                            await _send_json(websocket, {
                                "type": "status", "stage": aug["status"],
                            })

                        if aug.get("tool_narration"):
                            narration = aug["tool_narration"]
                            # Show narration in UI text but skip TTS — the tool cue
                            # already notifies the user, and the final LLM response
                            # will cover the same ground. Speaking both is redundant.
                            full_response += narration + " "
                            await _send_json(websocket, {"type": "llm_delta", "text": narration})

                        if aug.get("tool_status"):
                            tool_names = aug.get("tool_names", [])
                            await _send_json(websocket, {
                                "type": "tool_activity", "status": aug["tool_status"],
                                "tools": tool_names,
                            })
                            if not _narrated_this_turn:
                                _narrated_this_turn = True
                                await _tts_q.put((_tool_cue(tool_names), ""))

                        if aug.get("tool_call"):
                            tc = aug["tool_call"]
                            tr_msg: dict = {
                                "type": "tool_result", "tool": tc.get("tool", ""),
                                "success": tc.get("success", False),
                                "preview": tc.get("output_preview", "")[:200],
                            }
                            rmeta = tc.get("result_metadata") or {}
                            if tr_msg["tool"] == "youtube" and rmeta:
                                tr_msg["youtube_mode"] = rmeta.get("youtube_mode", "")
                                if rmeta.get("youtube_mode") == "search":
                                    tr_msg["videos"] = rmeta.get("results", [])
                                elif rmeta.get("youtube_mode") == "direct":
                                    tr_msg["video"] = {
                                        "video_id": rmeta.get("video_id", ""),
                                        "title": rmeta.get("title", ""),
                                        "channel": rmeta.get("channel", ""),
                                        "thumbnail": rmeta.get("thumbnail", ""),
                                        "url": rmeta.get("url", ""),
                                    }
                            # image_search: forward result images (see the
                            # _process_voice_turn path above for rationale).
                            if tr_msg["tool"] == "image_search" and rmeta.get("images"):
                                tr_msg["images"] = rmeta["images"]
                            await _send_json(websocket, tr_msg)

                        if chunk.thinking_delta:
                            full_thinking += chunk.thinking_delta

                        if chunk.content_delta:
                            full_response += chunk.content_delta
                            await _send_json(websocket, {"type": "llm_delta", "text": chunk.content_delta})

                            sentence = sentence_buffer.add_token(chunk.content_delta)
                            if sentence:
                                _inst = ""
                                if _emotion_aware2:
                                    if _use_inline_tags2:
                                        from augmentum.voice.emotion import (
                                            extract_emotion_tag,
                                            wrap_with_emotion_tag,
                                        )
                                        # Fish: extract tag from raw sentence before cleaning
                                        _etag2 = extract_emotion_tag(sentence, _entity_emotion2)
                                    else:
                                        from augmentum.voice.emotion import extract_emotion_instruct
                                        _inst = extract_emotion_instruct(sentence, _entity_emotion2)
                                if not _inst:
                                    _inst = _voice_style2
                                if _use_turbo_tags2:
                                    from augmentum.voice.emotion import inject_turbo_tags
                                    clean = clean_for_tts(inject_turbo_tags(sentence), is_narrative, preserve_brackets=True)
                                else:
                                    clean = clean_for_tts(sentence, is_narrative)
                                if clean:
                                    if _emotion_aware2 and _use_inline_tags2:
                                        clean = wrap_with_emotion_tag(clean, _etag2)
                                    emitted_sentences.append(sentence)
                                    await _tts_q.put((clean, _inst))

                    if not session.interrupted:
                        remaining = sentence_buffer.flush()
                        if remaining:
                            _inst = ""
                            if _emotion_aware2:
                                if _use_inline_tags2:
                                    from augmentum.voice.emotion import (
                                        extract_emotion_tag,
                                        wrap_with_emotion_tag,
                                    )
                                    _etag2 = extract_emotion_tag(remaining, _entity_emotion2)
                                else:
                                    from augmentum.voice.emotion import extract_emotion_instruct
                                    _inst = extract_emotion_instruct(remaining, _entity_emotion2)
                            if not _inst:
                                _inst = _voice_style2
                            if _use_turbo_tags2:
                                from augmentum.voice.emotion import inject_turbo_tags
                                clean = clean_for_tts(inject_turbo_tags(remaining), is_narrative, preserve_brackets=True)
                            else:
                                clean = clean_for_tts(remaining, is_narrative)
                            if clean:
                                if _emotion_aware2 and _use_inline_tags2:
                                    clean = wrap_with_emotion_tag(clean, _etag2)
                                emitted_sentences.append(remaining)
                                await _tts_q.put((clean, _inst))
                finally:
                    await _tts_q.put(None)

            async def _tts_cons():
                # Resolve TTS provider once for the entire turn
                _cp, _ev = await _resolve_tts_provider(conn, _voice)

                # Use built-in Kokoro only if it's the resolved provider
                _kokoro2 = None
                _use_spcm = False
                _lipsync_engine2 = getattr(settings, "voice_lipsync_engine", "amplitude")
                if _cp and _cp.get("id") == "kokoro-builtin":
                    _ki2 = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
                    if not _ki2.is_available:
                        await load_model_off_loop(_ki2.load_model)
                    if _ki2.is_available:
                        _kokoro2 = _ki2
                elif _cp:
                    from augmentum.voice.tts import _STREAM_PCM_PROVIDERS
                    _use_spcm = _cp.get("id", "") in _STREAM_PCM_PROVIDERS

                _pf_task: asyncio.Task | None = None
                _pf_text: str = ""
                _pf_chunks: list[bytes] = []

                while True:
                    if _pf_task and _pf_task.done():
                        try:
                            _pf_chunks = _pf_task.result()
                        except Exception:
                            _pf_chunks = []
                        _pf_task = None

                    item = await _tts_q.get()
                    if item is None:
                        if _pf_task:
                            _pf_task.cancel()
                        break
                    text, instruct = item
                    if session.interrupted:
                        # Wait briefly — speech_discard may clear the flag
                        for _wi2 in range(10):  # up to 500ms
                            await asyncio.sleep(0.05)
                            if not session.interrupted:
                                break
                        if session.interrupted:
                            # Real interrupt — skip remaining sentences
                            if _pf_task:
                                _pf_task.cancel()
                                _pf_task = None
                            _pf_text = ""
                            _pf_chunks = []
                            continue
                        # False alarm cleared — proceed with this sentence

                    _tts_fmt2 = "pcm" if _use_spcm else _voice_fmt_for(_cp)
                    await _send_json(websocket, {"type": "tts_start", "sentence": text, "format": _tts_fmt2})

                    # Phoneme-schedule emission for external providers (Phase 2).
                    # Mirrors _tts_consumer above. No-op for Kokoro and when
                    # the universal flag is off.
                    await maybe_emit_normalized_schedule(text, websocket, _cp)

                    if _kokoro2 and _lipsync_engine2 in ("phoneme", "auto"):
                        # Built-in Kokoro + phoneme lip-sync: schedule-aware
                        # path so a viseme_schedule precedes each sentence's
                        # audio. No prefetch (schedule is per-sentence-duration;
                        # prefetch is HTTP-only). Amplitude uses the fast path.
                        ok = await stream_tts_sentence(
                            text, websocket, conn,
                            voice=_voice,
                            speed=_speed,
                            session=session,
                            provider=_cp,
                            instruct=instruct,
                            stream_pcm=_use_spcm,
                        )
                    elif _kokoro2:
                        # Built-in Kokoro + amplitude lip-sync: fast in-process
                        # generation (no schedule needed).
                        try:
                            voice_name = _voice
                            if voice_name and "::" in voice_name:
                                _, voice_name = voice_name.split("::", 1)
                            audio_bytes = await _kokoro2.generate(
                                text,
                                voice=voice_name or "af_heart",
                                speed=_speed,
                                response_format=_tts_fmt2,
                            )
                            if audio_bytes and not session.interrupted:
                                await websocket.send_bytes(audio_bytes)
                                ok = True
                            elif audio_bytes and session.interrupted:
                                # Wait for speech_discard to clear false alarm
                                for _wi in range(10):  # up to 500ms
                                    await asyncio.sleep(0.05)
                                    if not session.interrupted:
                                        break
                                if not session.interrupted:
                                    await websocket.send_bytes(audio_bytes)
                                    ok = True
                                    log.debug("tts_resumed_after_discard")
                                else:
                                    ok = False
                            else:
                                ok = True
                        except Exception as exc:
                            log.warning("kokoro_builtin_tts_error", error=str(exc))
                            await _send_json(websocket, {
                                "type": "tts_error",
                                "message": "Built-in TTS failed — check voice settings",
                            })
                            ok = False
                    elif _pf_chunks and _pf_text == text:
                        log.debug("tts_prefetch_hit", text_len=len(text))
                        ok = await send_prefetched_audio(
                            _pf_chunks, websocket, session=session,
                        )
                        _pf_text = ""
                        _pf_chunks = []
                    else:
                        _pf_text = ""
                        _pf_chunks = []

                        if not _tts_q.empty() and not _pf_task:
                            try:
                                next_item = _tts_q._queue[0]  # type: ignore[attr-defined]
                                if next_item is not None and not session.interrupted:
                                    next_text, next_instruct = next_item
                                    _pf_text = next_text
                                    _pf_task = asyncio.create_task(
                                        prefetch_tts_audio(
                                            next_text, conn,
                                            voice=_voice, speed=_speed,
                                            provider=_cp,
                                            instruct=next_instruct,
                                            user_id=session.user_id,
                                        )
                                    )
                                    # See _process_voice_turn for rationale —
                                    # guarantee cancellation on abnormal exit.
                                    _pf_task.add_done_callback(_silence_task_exception)
                                    _active_bg_tasks.append(_pf_task)
                            except (IndexError, AttributeError):
                                pass

                        ok = await stream_tts_sentence(
                            text, websocket, conn,
                            voice=_voice, speed=_speed, session=session,
                            provider=_cp, instruct=instruct,
                            stream_pcm=_use_spcm,
                        )

                    if not ok and session.interrupted:
                        if _pf_task:
                            _pf_task.cancel()
                            _pf_task = None
                        _pf_text = ""
                        _pf_chunks = []

            await asyncio.gather(_llm_prod(), _tts_cons())

        await asyncio.wait_for(
            _recall_and_stream(),
            timeout=settings.request_timeout,
        )
    except TimeoutError:
        log.warning("voice_turn_timeout", timeout=settings.request_timeout)
        await _send_json(websocket, {"type": "error", "message": "Response timed out"})
    except Exception as exc:
        log.warning("voice_turn_error", error=str(exc))
        msg = str(exc)
        if "429" in msg:
            friendly = "The model provider is rate-limiting requests. Please wait a moment."
        elif "503" in msg or "502" in msg:
            friendly = "The model backend is temporarily unavailable."
        else:
            friendly = f"Error: {msg[:200]}"
        await _send_json(websocket, {"type": "error", "message": friendly})
    finally:
        # See _process_voice_turn for rationale — drain any prefetch
        # tasks that escaped the consumer's normal cleanup paths.
        for _bg in _active_bg_tasks:
            if not _bg.done():
                _bg.cancel()
        # Commit the assistant turn before any cancellation finishes
        # unwinding us — this is the only place the turn is persisted.
        _commit_turn()

    if session.interrupted:
        await _send_json(websocket, {"type": "interrupted", "heard_text": _heard_text})
    else:
        await _send_json(websocket, {"type": "tts_end"})
        await _flush_pending_surface_events(
            websocket, app_state, session.user_id, session.session_id,
        )
        await _send_json(websocket, {"type": "turn_complete", "full_text": full_response})

    await _send_json(websocket, {"type": "listening"})


# ---------------------------------------------------------------------------
# Server-Side VAD + Streaming STT Helpers
# ---------------------------------------------------------------------------


async def _get_stt_config(app_state: Any) -> dict[str, Any] | None:
    """Resolve the configured STT provider for streaming STT setup."""
    sm = getattr(app_state, "state_manager", None)
    if not sm:
        return None
    from augmentum.state.backends.sqlite import SQLiteBackend
    if not isinstance(sm.backend, SQLiteBackend):
        return None
    conn = sm.backend.conn
    from augmentum.proxy.audio_routes import _get_default_provider
    provider = await _get_default_provider(conn, "stt")
    return provider


async def _open_streaming_stt(
    provider: dict[str, Any],
    on_transcript: Any,
) -> StreamingSTTSession | None:
    """Open a streaming STT WebSocket if the provider supports it."""
    base_url = provider.get("base_url", "")
    api_key = provider.get("api_key", "")
    model = provider.get("default_model", "nova-3")

    if not is_streaming_stt_capable(base_url):
        return None

    session = StreamingSTTSession(
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpointing_ms=settings.voice_stt_endpointing_ms,
        on_transcript=on_transcript,
    )
    try:
        await asyncio.wait_for(session.connect(), timeout=10.0)
        return session
    except (TimeoutError, Exception) as exc:
        log.warning("streaming_stt_connect_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------


@router.websocket("/voice/sessions/{voice_session_id}/stream")
async def voice_session_stream(websocket: WebSocket, voice_session_id: str) -> None:
    """Subscriber WS for a voice session's fanout.

    Cast receivers (and future voice renderers) open this WS to
    receive a mirror of every TTS audio frame + JSON control message
    + viseme event published to the session. One subscriber per
    connection; multiple connections to the same session are fine
    (multi-TV households).

    Authentication: session cookie fallback (see middleware.py) or
    standard ws-ticket. The fanout itself enforces user ownership —
    a different user attempting to subscribe sees an empty stream.

    Lifecycle: subscribe on connect → forward events until either
    the source voice WS disconnects (sentinel arrives) or this WS
    closes. The subscription auto-tears-down via the async iterator's
    finally clause.
    """
    await websocket.accept()
    user = websocket.scope.get("user")
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    fanout = getattr(websocket.app.state, "voice_fanout", None)
    if fanout is None:
        await websocket.close(code=1011, reason="Voice fanout not initialised")
        return

    from augmentum.voice.fanout import (
        FANOUT_KIND_BYTES,
        FANOUT_KIND_JSON,
        FANOUT_KIND_TEXT,
    )

    log.info(
        "voice_fanout_subscriber_attached",
        voice_session_id=voice_session_id, user_id=user.id,
    )
    try:
        async for event in fanout.subscribe(voice_session_id, user_id=user.id):
            try:
                if event.kind == FANOUT_KIND_BYTES:
                    await websocket.send_bytes(event.payload)
                elif event.kind == FANOUT_KIND_JSON:
                    await websocket.send_json(event.payload)
                elif event.kind == FANOUT_KIND_TEXT:
                    await websocket.send_text(event.payload)
            except Exception as exc:
                # WS dropped mid-send — stop iterating, the finally
                # clause on the iterator detaches us from the fanout.
                log.debug(
                    "voice_fanout_subscriber_send_failed",
                    voice_session_id=voice_session_id, error=str(exc)[:160],
                )
                break
    except WebSocketDisconnect:
        pass
    finally:
        log.info(
            "voice_fanout_subscriber_detached",
            voice_session_id=voice_session_id, user_id=user.id,
        )


class _IntentOutcome(NamedTuple):
    """Result of running the intent registry over a transcript.

    ``handled`` — True iff the turn short-circuited (caller should
    return; the WS event has already been emitted).

    ``transcript`` — what the caller should pass to the LLM. May be
    augmented with a ``<recall_hits>`` / similar context block when a
    soft-augmentation action fires. Falls back to the original
    transcript when no action matched OR the match was short-circuit.
    """

    handled: bool
    transcript: str


def _smart_turn_veto_confidence(prob: float, threshold: float) -> float:
    """Confidence of a smart-turn VETO, in [0, threshold].

    ``prob`` is the model's probability that the turn is COMPLETE; a
    veto fires when it lands below the threshold. The veto's confidence
    is how far below: threshold − prob. 0 = borderline coin-flip veto,
    threshold = the model is certain the user is NOT done.

    History (2026-06-11): the override gate originally compared ``prob``
    itself against the floor — backwards. prob=0.03 ("97% sure the user
    is still mid-thought") read as a "3% confidence veto" and was
    overridden, hard-cutting the user off exactly when the model was
    most certain they were still talking.
    """
    return max(0.0, threshold - prob)


# Canonical Whisper/Moonshine hallucinations — phantom text the model
# invents from silence, mic bumps, breaths, and room noise. They pass
# the duration/word-rate/repetition filters (they're short, normal-rate,
# non-repeating real words) so they need a phrase-aware gate.
#
# ALWAYS: caption/credit artifacts + lone fillers that are virtually
# never a real standalone companion utterance — dropped unconditionally.
_STT_HALLUCINATION_ALWAYS: frozenset[str] = frozenset({
    "thanks for watching", "thank you for watching", "thanks for watching everyone",
    "please subscribe", "like and subscribe", "don't forget to subscribe",
    "subscribe to my channel", "see you next time", "see you in the next video",
    "subtitles by the amara.org community", "subtitles by", "transcription by",
    "thanks for listening", "you", "...", ".", "..", "♪", "♪♪",
    "[music]", "[applause]", "[ silence ]", "[silence]", "[blank_audio]",
})
# AMBIENT-ONLY: plausible as a real deliberate word, but on an ambient
# (non-button, non-wake) capture a LONE one of these is almost always a
# phantom from a bump/breath. An explicit ptt/wake capture keeps them —
# the user deliberately spoke.
_STT_HALLUCINATION_IF_AMBIENT: frozenset[str] = frozenset({
    "thank you", "thanks", "okay", "ok", "bye", "yeah", "yep", "uh",
    "um", "hmm", "mm", "mhm", "oh", "so", "and", "the", "a", "i",
})


def _is_stt_hallucination(text: str, *, explicit: bool) -> bool:
    """Whether a final transcript is a canonical STT phantom (no real
    speech behind it). Matches the WHOLE normalized transcript, so a
    phantom phrase embedded in real speech ("thank you, can you…") is
    NOT dropped. ``explicit`` (ptt/wake) keeps the ambient-only set —
    the user deliberately spoke."""
    norm = " ".join(text.strip().lower().split()).rstrip("!?")
    bare = norm.rstrip(".")
    if norm in _STT_HALLUCINATION_ALWAYS or bare in _STT_HALLUCINATION_ALWAYS:
        return True
    return not explicit and (
        norm in _STT_HALLUCINATION_IF_AMBIENT
        or bare in _STT_HALLUCINATION_IF_AMBIENT
    )


def _should_defer_veto(
    deferral_count: int, max_deferrals: int, vad_is_speaking: bool,
) -> bool:
    """Whether a smart-turn veto deadline should DEFER (keep listening)
    rather than finalize, when the deadline is reached.

    Defers only when VAD still reads speech AND we're under the deferral
    cap. The cap is what stops background noise — which Silero misreads
    as speech and which would otherwise satisfy ``vad_is_speaking``
    forever — from extending the turn without bound ("feels super long",
    2026-06-13). max_deferrals=0 disables deferral entirely (finalize at
    the first deadline).
    """
    return bool(vad_is_speaking) and deferral_count < max_deferrals


def _turn_stamp(session: Any, stage: str) -> None:
    """Record a monotonic timestamp for the per-turn latency waterfall."""
    try:
        timing = getattr(session, "turn_timing", None)
        if timing is None:
            return
        timing[stage] = time.monotonic()
    except Exception:  # noqa: BLE001 — telemetry must never break a turn
        log.debug("voice_turn_stamp_failed", stage=stage, exc_info=True)


def _emit_turn_waterfall(session: Any) -> None:
    """One structured line for the whole speech_end -> first_audio budget.

    Reconstructing this from five scattered log lines is how the latency
    review actually went (2026-06-13) — this makes each turn's stage
    timings legible at a glance and any future regression obvious.
    Emitted once per turn at first audio; deltas are ms from speech_end.
    """
    try:
        timing = getattr(session, "turn_timing", None)
        if not timing or "speech_end" not in timing or "first_audio" not in timing:
            return
        t0 = timing["speech_end"]

        def _ms(stage: str):
            t = timing.get(stage)
            return round((t - t0) * 1000) if t is not None else None

        log.info(
            "voice_turn_waterfall",
            session_id=getattr(session, "session_id", ""),
            stt_ms=_ms("stt_done"),
            dispatch_ms=_ms("dispatch"),
            ack_ms=_ms("ack"),
            route_ms=_ms("route_done"),
            first_audio_ms=_ms("first_audio"),
        )
    except Exception:  # noqa: BLE001
        log.debug("voice_turn_waterfall_failed", exc_info=True)


async def _speak_ack_clip(websocket: WebSocket, conn: Any, session: Any) -> None:
    """Speak a short, varied acknowledgement the instant a deliberate
    turn dispatches — latency masking (latency MVP, 2026-06-13).

    The reply's first audio is ~2s out; a "mm-hm" at ~150ms reads as
    "she heard me" and the real reply lands on an already-engaged
    listener. Constraints that keep it from ever hurting:

      - In-process Kokoro provider ONLY: fast synth AND the SAME voice
        as the reply. An external/cloud ack would add a round trip
        and/or sound like a different person — skipped silently.
      - Variety via the shuffle-bag (ack_clips): no line repeats until
        the pool exhausts, and silent slots mean some turns get no ack
        at all, so it never becomes a tic in a long session.
      - Fired INLINE before dispatch so it cleanly precedes the reply
        on the client's single voice playback queue (no interleave
        race with the turn's own TTS).
      - Best-effort: tight synth budget, never raises, never blocks the
        turn on failure. The ~150ms it costs is masked by the ack it
        plays.
    """
    try:
        if not bool(getattr(settings, "voice_ack_clips_enabled", False)):
            return
        from augmentum.voice import ack_clips
        line = ack_clips.next_ack(session.session_id)
        if not line:
            return  # silent slot — variety
        _voice = session.character_voice or session.voice
        provider, resolved_voice = await _resolve_tts_provider(conn, _voice)
        if not provider or provider.get("id") != "kokoro-builtin":
            return  # only the fast in-process, same-voice path
        ki = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
        if not ki.is_available:
            return  # don't pay a cold model load on the latency path
        fmt = _voice_fmt_for(provider)
        voice_name = resolved_voice
        if voice_name and "::" in voice_name:
            _, voice_name = voice_name.split("::", 1)
        audio = await asyncio.wait_for(
            ki.generate(
                line, voice=voice_name or "af_heart",
                speed=session.speed, response_format=fmt,
            ),
            timeout=1.0,
        )
        if not audio:
            return
        await _send_json(websocket, {
            "type": "tts_start", "sentence": line, "format": fmt,
        })
        await websocket.send_bytes(audio)
        await _send_json(websocket, {"type": "tts_end"})
        log.debug("voice_ack_spoken", line=line, session_id=session.session_id)
    except Exception:  # noqa: BLE001 — ack is pure nicety
        log.debug("voice_ack_failed", exc_info=True)


def _trim_prefix_to_post_tts(
    prefix: bytes, tts_ended_at: float | None, now: float,
) -> bytes:
    """Drop prefix audio recorded before her TTS finished.

    The VAD prefix ring keeps the last N ms of audio before
    speech_start so STT doesn't lose the user's leading words. With a
    generous ring (1500ms) and a fast reply, the ring can reach back
    INTO her TTS playback — her last words would transcribe into the
    user's utterance. Clamp to audio captured after tts_ended_at.

    During active TTS (barge-in) ``tts_ended_at`` still holds the
    PREVIOUS turn's end — elapsed is large, nothing is trimmed; the
    client AEC owns that overlap as before.
    """
    if not prefix or not tts_ended_at:
        return prefix
    elapsed_ms = (now - float(tts_ended_at)) * 1000.0
    if elapsed_ms <= 0:
        return prefix
    allowed = max(1, int(elapsed_ms / 32.0)) * FRAME_BYTES
    if len(prefix) <= allowed:
        return prefix
    return prefix[-allowed:]


def _is_open_thread(refs: Any, last_tts_text: str) -> bool:
    """Is the conversation demonstrably waiting on the user?

    True when her last spoken line ended with a question mark, or a
    verb parked a pending clarification (``ReferentCache.pending_intent``,
    freshness-gated). Used to stretch the follow-up relaxation window —
    the user may read / think for half a minute before answering, and
    their answer must not be dropped as ambient noise.
    """
    try:
        from augmentum.intent.dispatch import get_fresh_pending_intent
        if get_fresh_pending_intent(refs) is not None:
            return True
        return (last_tts_text or "").rstrip().endswith("?")
    except Exception:  # noqa: BLE001 — extension is best-effort
        return False


async def _dispatch_fast_path_control(
    transcript: str,
    session: Any,
    websocket: WebSocket,
    ctx: Any,
) -> _IntentOutcome | None:
    """Pre-router pass for conversation-control verbs.

    Stop / repeat / slower / louder / goodbye / nevermind / strike are
    addressed to the conversation infrastructure, not Becca's reasoning
    (see ``intent/builtin/control.py``). They must fire sub-100ms and
    must NOT be swallowed by the always-listening router's converse-skip
    (``_skip_legacy_intent``) or wait on LLM latency. Running them here,
    before the router/architect, guarantees "stop" and "scratch that"
    always land.

    Purely additive: a miss returns ``None`` and ``_maybe_dispatch_intent``
    continues its normal flow unchanged.
    """
    try:
        outcome = await intent_dispatch(
            transcript, session=ctx, fast_path_only=True,
        )
    except Exception as exc:
        log.warning("voice_fastpath_dispatch_error", error=str(exc))
        return None
    if outcome is None:
        return None
    match, result = outcome
    if not result.short_circuit:
        return None

    payload = serialize_action_event(match, result)
    try:
        await _send_json(websocket, payload)
    except Exception as exc:
        log.warning("voice_fastpath_emit_failed", error=str(exc))
        return _IntentOutcome(handled=False, transcript=transcript)

    # conversation.strike scrubs the poisoned exchange from in-call
    # history server-side — a surface_emit alone can't touch
    # session.messages (the model's working context). Pop here so the
    # NEXT turn reasons from clean history. The strike utterance itself
    # is never added (we return handled before add_user_message).
    if match.action_id == "conversation.strike":
        try:
            removed = session.strike_last_exchange()
            log.info(
                "voice_conversation_strike",
                removed=removed,
                session_id=getattr(ctx, "session_id", ""),
            )
        except Exception:
            log.warning("voice_conversation_strike_failed", exc_info=True)

    try:
        await _send_json(websocket, {"type": "listening"})
    except Exception:
        log.debug("voice_fastpath_listening_send_failed", exc_info=True)
    log.info(
        "voice_fastpath_handled",
        action=match.action_id, session_id=getattr(ctx, "session_id", ""),
    )
    return _IntentOutcome(handled=True, transcript=transcript)


# Held-ambient buffer — utterances the addressing classifier dropped as "not
# clearly addressed". They used to be discarded (the ambient_speech bus topic
# has no consumer, so they were simply lost): the user sees text under STT,
# waits, and nothing happens — the model never saw it. Instead we keep the
# last few on the session, marked. When the user next addresses her, they're
# folded into that turn so she can RE-EVALUATE something she wasn't sure was
# for her ("actually, about what you said a moment ago…"). Windowed + capped
# so stale room-chatter can't accumulate.
_HELD_AMBIENT_WINDOW_S = 90.0
_HELD_AMBIENT_MAX = 3


def _hold_ambient(session: Any, text: str) -> None:
    """Stash a dropped utterance for possible re-evaluation on follow-up."""
    text = (text or "").strip()
    if not text:
        return
    buf = getattr(session, "held_ambient", None)
    if not isinstance(buf, list):
        buf = []
    buf.append({"text": text[:500], "ts": time.monotonic()})
    session.held_ambient = buf[-_HELD_AMBIENT_MAX:]


def _resurface_held_ambient(session: Any, transcript: str) -> str:
    """Prepend a marked note of recently-held (within-window) dropped
    utterances to an ADDRESSED turn's transcript, then clear the buffer. The
    marker tells the model these were only *possibly* meant for her a moment
    ago — she decides whether the current turn is a follow-up on them. No-op
    (returns transcript unchanged) when nothing is held."""
    buf = getattr(session, "held_ambient", None)
    if not isinstance(buf, list) or not buf:
        return transcript
    now = time.monotonic()
    fresh = [
        b for b in buf
        if now - float(b.get("ts", 0.0)) <= _HELD_AMBIENT_WINDOW_S
    ]
    session.held_ambient = []
    if not fresh:
        return transcript
    said = " / ".join(b["text"] for b in fresh)
    note = (
        "[Possibly-background: a moment ago the user said "
        f"\"{said}\" and you weren't sure it was addressed to you. They may "
        "be following up on it now — re-evaluate if this turn relates.]\n"
    )
    return note + transcript


# The addressing decision is BINARY: engage, or stay silent. The only goals
# that mean "stay silent" are ``idle`` (bare ack — "thanks") and ``drop``
# (noise / not-for-the-assistant). Everything else engages. We test against the
# silent set, NOT an "engage" allow-list, deliberately: act/converse/clarify are
# not meaningfully distinct at this layer (clarify is already aliased to the
# converse signal downstream), and an UNKNOWN/future goal should ENGAGE, never
# be silently dropped — the never-ignore invariant. act-vs-talk is decided later
# by whether the turn calls a tool, not pre-guessed here.
_SILENT_GOALS = ("idle", "drop")


def _promote_explicit_goal(*, explicit_capture: bool, goal: str) -> str:
    """Promote an ``idle``/``drop`` verdict to ``converse`` for EXPLICITLY-
    captured input (PTT mic open, or stage-manager Send).

    Explicit capture IS intent by construction, so an ``idle``/``drop`` goal is
    a small-model misclassification, never a reason to stay silent. Pure so the
    invariant is unit-testable — this rule regressed once (promoted ``idle``
    only, missing ``drop``) and silently discarded a coherent paragraph a user
    typed + Sent (Matt 2026-07-26)."""
    if explicit_capture and goal in ("idle", "drop"):
        return "converse"
    return goal


def _explicit_addressed_effective(
    *,
    from_stage_send: bool,
    explicit_capture: bool,
    coherent: bool,
    addressed: bool,
    confidence: float,
    goal: str,
    effective_threshold: float,
    in_followup: bool,
) -> bool:
    """Whether the companion should engage this turn (``addressed_effective``).

    INVARIANT (Matt 2026-07-26): ``from_stage_send`` — typed text + the Send
    button — is UNAMBIGUOUS intent. The result is ALWAYS ``True``; no coherence
    or goal veto may zero it out. The companion must never be able to ignore
    something the user explicitly typed and sent.

    Explicit PTT (``explicit_capture`` without stage-send) keeps the coherence
    veto (a cough shouldn't force a reply) but drops the addressed/confidence
    veto — the user chose to open the mic. Ambient input passes the ordinary
    confidence-or-followup gate. ``goal`` should already be the promoted goal
    from :func:`_promote_explicit_goal`."""
    if from_stage_send:
        return True
    if explicit_capture:
        return coherent and goal not in _SILENT_GOALS
    addressed_gate = (addressed and confidence >= effective_threshold) or in_followup
    return addressed_gate and coherent and goal not in _SILENT_GOALS


async def _maybe_dispatch_intent(
    transcript: str,
    session: Any,
    websocket: WebSocket,
    app_state: Any,
    *,
    from_stage_send: bool = False,
) -> _IntentOutcome:
    """Run the intent registry over a final transcript.

    Three return shapes (encoded in :class:`_IntentOutcome`):

    1. **Short-circuit** (``handled=True``) — an action like
       ``open browse`` or ``stop`` fired. The WS ``intent_action`` +
       ``listening`` events have been emitted; the caller should
       return immediately and skip the LLM.

    2. **Soft augmentation** (``handled=False``, ``transcript`` is
       augmented) — an action like ``memory.recall`` returned a
       ``prompt_addendum``. The caller continues to the LLM with the
       augmented transcript so the model has the recall hits in
       context.

    3. **Pass-through** (``handled=False``, ``transcript`` unchanged)
       — no action matched. Normal UARF / handler flow.

    Note-capture mode is handled separately at the end of this
    function — non-intent utterances become note appends.
    """
    ctx = IntentSessionContext(
        user_id=getattr(session, "user_id", "") or "",
        session_id=getattr(session, "session_id", "") or "",
        mode=getattr(session, "mode", None),
        app_state=app_state,
    )

    # Fast-path conversation control (stop / repeat / scratch that / …)
    # runs BEFORE the router so it can't be swallowed by the always-
    # listening converse-skip or wait on LLM latency. A miss returns
    # None and flow continues unchanged.
    _fp = await _dispatch_fast_path_control(transcript, session, websocket, ctx)
    if _fp is not None:
        return _fp

    # Always-listening gate — when the companion mode is
    # ``always_listening`` and this is the companion path
    # (persona_id=becca), run the address classifier first. Non-
    # addressed utterances go to the ambient observation sink and
    # the function returns handled=True without invoking the LLM or
    # speaking — the user simply gets silence, as if they were
    # talking to themselves (which they were).
    activation_mode = (getattr(settings, "companion_activation_mode", "wake_word") or "wake_word").lower()
    is_becca_path = getattr(session, "persona_id", "") == "becca"

    # Media-aware threshold boost. When the user is currently
    # listening to media (YouTube, audiobook, Grove music, etc.),
    # the mic picks up the playback and VAD treats it as continuous
    # speech. To prevent the architect responding to YouTube
    # narration, RAISE the address-classifier threshold so only
    # very clear addressing (imperative_start with recent activity)
    # passes. Reads the most recent surface.audio.kind_changed event
    # from the runtime's recent deque — the architect observer is
    # already wired to feed that.
    _media_active = False
    try:
        runtime = getattr(app_state, "companion_runtime", None)
        if runtime is not None:
            obs_state = getattr(runtime, "observed_state", None)
            recent = getattr(obs_state, "recent", None) if obs_state else None
            if recent:
                # Scan last few entries for a current media-tier signal.
                # Per-entry shape: {topic, payload, t}. We accept entries
                # within the last 90s since AudioBus events fire on tier
                # state changes (not per-frame).
                import time as _time
                cutoff = _time.time() - 90.0
                for entry in reversed(list(recent)):
                    try:
                        if entry.get("t", 0) < cutoff:
                            break
                        topic = entry.get("topic", "")
                        payload = entry.get("payload", {}) or {}
                        if topic == "surface.audio.kind_changed":
                            kinds = payload.get("kinds") or []
                            tiers = payload.get("tiers") or []
                            # MEDIA / AMBIENT tiers indicate playback;
                            # SPEECH is her own TTS (handled elsewhere).
                            if "media" in tiers or "ambient" in tiers:
                                if kinds:  # non-empty kinds = currently playing
                                    _media_active = True
                            break
                    except (AttributeError, TypeError):
                        continue
    except Exception:  # noqa: BLE001 — observation lookup is best-effort
        _media_active = False

    if activation_mode == "always_listening" and is_becca_path:
        import time as _time
        tts_ts = getattr(session, "tts_ended_at", None)

        # Voice router (MVP — 2026-05-28) — single structured LLM call
        # replaces the regex Tier 1 + LLM Tier 3 stack. Returns coherent
        # + addressed + confidence + goal in one shot. See
        # augmentum/architect/voice_router.py for the design.
        #
        # Falls back to ``AddressDecision`` shape so the existing
        # downstream routing (question-skip, architect dispatch, etc.)
        # doesn't need to change. The ``signal`` field encodes the
        # ``goal`` from the router so the downstream skip-logic can
        # tell act-vs-converse apart.
        last_tts_text = ""
        try:
            msgs = getattr(session, "messages", None) or []
            for m in reversed(msgs):
                if (m or {}).get("role") == "assistant":
                    last_tts_text = (m.get("content") or "")[:200]
                    break
        except Exception:  # noqa: BLE001 — context is optional
            last_tts_text = ""
        refs = get_intent_referents(app_state, ctx.user_id, ctx.session_id)
        last_dispatch = refs.last_dispatch_summary or ""
        active_surface_ctx = getattr(session, "mode", "") or ""
        seconds_since_tts: float | None = None
        if tts_ts is not None:
            try:
                seconds_since_tts = _time.monotonic() - float(tts_ts)
            except (TypeError, ValueError):
                seconds_since_tts = None

        vr = await classify_voice(
            transcript,
            app_state=app_state,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            last_assistant_response=last_tts_text,
            last_dispatch_summary=last_dispatch,
            seconds_since_last_tts=seconds_since_tts,
            active_surface=active_surface_ctx,
        )

        # Media-active gate: when audio is playing the mic picks up
        # narration / music that the router can mistake for addressed
        # speech. Raise the confidence bar so only high-confidence
        # ADDRESSED routes through. Same intent as the old
        # ``companion_address_media_boost`` setting, applied at the
        # post-classification gate instead of pre-regex.
        base_threshold = float(getattr(settings, "companion_address_threshold", 0.70))
        if _media_active:
            boost = float(getattr(settings, "companion_address_media_boost", 0.20) or 0.0)
            effective_threshold = min(0.99, base_threshold + boost)
        else:
            effective_threshold = base_threshold

        # Explicit capture: the utterance arrived through a deliberate
        # channel (PTT press / wake word — stamped by _finalize_speech
        # from the client's start_recording source tag). Addressing is
        # a given; the router still contributes coherence + goal, but
        # its addressed/confidence judgment must not veto a press.
        # Typed text sent from the stage manager is addressed BY CONSTRUCTION:
        # the user composed it and pressed Send. Treat it like an explicit
        # capture (PTT/wake) so the addressing classifier's confidence — which
        # is meaningless on typed text (Gemma scored a full typed paragraph
        # 0.3) — can never veto it. The router still contributes coherence +
        # goal for tool-roster routing; only its addressed/confidence judgment
        # is overridden.
        explicit_capture = bool(
            getattr(session, "last_utterance_explicit", False)
        ) or from_stage_send

        goal = _promote_explicit_goal(explicit_capture=explicit_capture, goal=vr.goal)
        if goal != vr.goal:
            # idle/drop on explicitly-captured input is a misclassification —
            # promoted to converse so the companion always responds. See
            # _promote_explicit_goal (Matt 2026-07-26: typed+Sent paragraph
            # silently dropped as goal=drop).
            log.info(
                "voice_router_explicit_goal_promoted",
                session_id=ctx.session_id,
                goal_raw=vr.goal,
                from_stage_send=from_stage_send,
                text_preview=transcript[:80],
            )

        # Opt-in capture of this routing decision for on-device intent-model
        # training (gated; default OFF — only the user who enabled it writes).
        # The store swallows its own errors so it can never break the voice
        # turn. See augmentum/intent/capture_store.py.
        if getattr(settings, "intent_capture_enabled", False):
            sm = getattr(app_state, "state_manager", None)
            if sm is not None:
                from augmentum.state.backends.sqlite import SQLiteBackend
                if isinstance(sm.backend, SQLiteBackend):
                    from augmentum.intent.capture_store import record_intent_capture
                    await record_intent_capture(
                        sm.backend.conn,
                        user_id=ctx.user_id,
                        session_id=ctx.session_id,
                        surface="voice_router",
                        input_text=transcript,
                        last_assistant_response=last_tts_text,
                        last_dispatch_summary=last_dispatch,
                        active_surface=active_surface_ctx,
                        seconds_since_last_tts=seconds_since_tts,
                        media_active=_media_active,
                        explicit_capture=explicit_capture,
                        goal=vr.goal,
                        effective_goal=goal,
                        coherent=vr.coherent,
                        addressed=vr.addressed,
                        confidence=vr.confidence,
                        teacher_model=vr.model,
                        parsed_from=vr.parsed_from,
                        reasoning=vr.reasoning,
                        latency_ms=vr.latency_ms,
                    )

        # Map the router's goal onto a signal string. Downstream code
        # uses ``signal`` to decide whether to skip the architect
        # (conversational replies) or dispatch (action commands). The
        # ``vr_*`` prefix marks these as router-sourced so logs are
        # filterable.
        goal_to_signal = {
            "act": "vr_act",
            "converse": "vr_converse",
            "clarify": "vr_converse",  # clarify falls to chat path for now
            "idle": "vr_idle",
            "drop": "vr_drop",
        }
        signal = goal_to_signal.get(goal, "vr_drop")

        # Conversation-window relaxation: when she spoke seconds ago the
        # exchange is LIVE — a coherent reply-shaped utterance (goal says
        # a response is warranted) shouldn't die on the addressed/
        # confidence veto. Observed 2026-06-11 as "too greedy with
        # drops": commentary on what's playing, dropped mid-conversation.
        # goal=drop/idle still gates (her own TTS echo classifies as
        # drop), so this only admits utterances the router itself judged
        # reply-worthy. Set companion_followup_window_s=0 to disable.
        followup_window_s = float(
            getattr(settings, "companion_followup_window_s", 12.0) or 0.0
        )
        # Open-thread extension: when her last line ended with a
        # question, or a verb parked a pending clarification, the
        # conversation is demonstrably waiting on the user — they may
        # read / think for half a minute before answering ("what city
        # should I use?" → 30s later → "Rochester"). Hold the relaxed
        # gate for the longer window so the answer isn't dropped as
        # ambient noise.
        open_thread = _is_open_thread(refs, last_tts_text)
        if open_thread:
            open_window_s = float(getattr(
                settings, "companion_open_thread_window_s", 45.0,
            ) or 0.0)
            followup_window_s = max(followup_window_s, open_window_s)
        in_followup = (
            seconds_since_tts is not None
            and seconds_since_tts <= followup_window_s
        )
        addressed_effective = _explicit_addressed_effective(
            from_stage_send=from_stage_send,
            explicit_capture=explicit_capture,
            coherent=vr.coherent,
            addressed=vr.addressed,
            confidence=vr.confidence,
            goal=goal,
            effective_threshold=effective_threshold,
            in_followup=in_followup,
        )
        decision = AddressDecision(addressed_effective, vr.confidence, signal)

        # Stash the router's goal on the session so the downstream
        # companion LLM path (prompt_compose Layer 6) can switch the
        # tool roster into act mode. Without this the classifier KNOWS
        # the user asked for an action (goal=act conf=0.95) but the
        # prompt never hears about it — observed 2026-06-10 as Becca
        # role-playing "I thought I just did that" instead of emitting
        # the tool tag.
        try:
            session.last_router_goal = goal
        except Exception:  # noqa: BLE001 — session shape defensive
            log.debug("router_goal_stash_failed", exc_info=True)

        log.info(
            "voice_address_classifier",
            session_id=ctx.session_id,
            addressed=decision.addressed,
            confidence=round(decision.confidence, 2),
            signal=decision.signal,
            goal=goal,
            router_goal=vr.goal,
            coherent=vr.coherent,
            threshold=round(effective_threshold, 2),
            media_active=_media_active,
            explicit_capture=explicit_capture,
            in_followup=in_followup,
            open_thread=open_thread,
            text_preview=transcript[:80],
        )

        # Near-miss: coherent, reply-shaped speech that landed just under the
        # addressing bar. Hoisted to a single source of truth so BOTH the
        # decision telemetry below and the ambient-drop branch render a
        # consistent "heard you" tell. Explicit captures can't near-miss
        # (they're addressed by construction). Band 0 disables.
        _near_band = float(
            getattr(settings, "companion_address_near_miss_band", 0.25) or 0.0
        )
        near_miss = (
            not explicit_capture
            and _near_band > 0.0
            and vr.coherent
            and goal in ("act", "converse", "clarify")
            and decision.confidence >= max(0.0, effective_threshold - _near_band)
        )

        # Admit near-misses instead of dropping them. A near-miss is, by the
        # conditions above, coherent + reply-shaped speech (goal act/converse/
        # clarify) that landed just under the confidence bar — far more often
        # "the user, a hair under threshold" than ambient noise, which the
        # router routes to goal=drop/idle (never a near_miss, still dropped
        # below). Rescuing it here makes the DROP decision require real
        # evidence of non-address — genuinely ambient, incoherent, or her own
        # TTS echo — rather than merely sub-threshold confidence. This is the
        # single biggest source of "I was clearly talking to her and she
        # dropped it". Toggle ``companion_address_admit_near_miss`` off to
        # restore the strict confident-address-or-drop policy.
        if (
            near_miss
            and not decision.addressed
            and bool(getattr(settings, "companion_address_admit_near_miss", True))
        ):
            log.info(
                "voice_address_near_miss_admitted",
                session_id=ctx.session_id,
                confidence=round(decision.confidence, 2),
                threshold=round(effective_threshold, 2),
                goal=goal,
                text_preview=transcript[:80],
            )
            decision = AddressDecision(True, decision.confidence, decision.signal)
            near_miss = False

        # Decision telemetry → the widget. Every classified turn ships its
        # verdict so the UI can render a subtle per-decision tell and an
        # opt-in decision HUD — closing the "did she even hear me? what did
        # she decide?" gap that previously required reading the logs. ``goal``
        # is the EFFECTIVE goal (post idle→converse promotion); ``router_goal``
        # is the raw verdict. Fire-and-forget; _send_json swallows closed
        # sockets, and a telemetry hiccup must never break the turn.
        try:
            await _send_json(websocket, {
                "type": "voice_decision",
                "goal": goal,
                "router_goal": vr.goal,
                "addressed": bool(decision.addressed),
                "explicit": explicit_capture,
                "confidence": round(decision.confidence, 2),
                "near_miss": near_miss,
                "transcript": transcript[:120],
            })
        except Exception as exc:  # noqa: BLE001 — telemetry never breaks the turn
            log.debug("voice_decision_emit_failed", error=str(exc)[:160])

        # A typed message + Send is a COMMITTED chat turn — it can NEVER enter
        # the silent-drop branch, regardless of any classifier verdict. This is
        # a structural guarantee (not a rescued flag): the user pressed Send, so
        # the companion responds, full stop. (Matt: every stage-manager message
        # receives a response.) The `and not from_stage_send` is the belt to the
        # addressed_effective braces — if a future edit breaks the flag, typed
        # input still cannot be dropped here.
        if not decision.addressed and not from_stage_send:
            # Ambient utterance — emit as observation for the kernel's
            # Working / Observations layers and return handled so the
            # voice pipeline doesn't fire the LLM. The user wasn't
            # talking to Becca; she shouldn't respond. Explicit captures
            # land here only when incoherent — garbled STT has no
            # observational value, so skip the ambient publish for them.
            if not explicit_capture:
                # Keep it passively: coherent, reply-shaped speech that got
                # dropped is exactly what a follow-up might resurrect. Held on
                # the session (not thrown away) so the next addressed turn can
                # re-evaluate it. Incoherent STT has nothing to follow up on.
                if vr.coherent:
                    _hold_ambient(session, transcript)
                try:
                    runtime = getattr(app_state, "companion_runtime", None)
                    bus = getattr(runtime, "bus", None) if runtime else None
                    if bus is not None:
                        await bus.publish_topic(
                            "surface.companion.ambient_speech",
                            {
                                "user_id": ctx.user_id,
                                "session_id": ctx.session_id,
                                "text": transcript[:500],
                                "confidence": round(decision.confidence, 2),
                                "signal": decision.signal,
                            },
                            propagation="FACTUAL_ONLY",
                        )
                except Exception as exc:  # noqa: BLE001 — observability never breaks the path
                    log.debug("voice_ambient_emit_failed", error=str(exc)[:160])

            # Terminal feedback — this branch previously returned with
            # no client message, leaving the widget stuck in its
            # "thinking" state until the next press. Explicit captures
            # that came out incoherent get the same "didn't catch that"
            # hint as empty STT; every path re-arms the widget.
            # (_send_json silences closed-socket errors internally.)
            if explicit_capture:
                await _send_json(websocket, {
                    "type": "voice_no_speech",
                    "reason": "incoherent" if not vr.coherent else "unaddressed",
                    "message": "I didn't catch that — try again?",
                })

            # Near-miss tell: she HEARD coherent, reply-shaped speech that
            # landed just under the addressing bar. A user who *did* address
            # her gets a faint acknowledgement (rendered as a non-spoken
            # widget cue) instead of a silent void — but clearly-ambient
            # speech (incoherent, idle/drop goal, or confidence well below
            # the bar) stays silent so she doesn't flicker at every word
            # across the room. ``near_miss`` is computed once above (shared
            # with the decision telemetry); only the log fires here.
            if near_miss:
                log.info(
                    "voice_address_near_miss",
                    session_id=ctx.session_id,
                    confidence=round(decision.confidence, 2),
                    threshold=round(effective_threshold, 2),
                    goal=goal,
                    text_preview=transcript[:80],
                )

            await _send_json(websocket, {
                "type": "listening",
                "heard": bool(transcript.strip()),
                "near_miss": near_miss,
                "confidence": round(decision.confidence, 2),
            })
            return _IntentOutcome(handled=True, transcript=transcript)

    # Signal-based routing — questions skip the architect entirely.
    # When the address classifier matched on a QUESTION cue (WH-form
    # without an action verb), the user is asking, not commanding.
    # Routing through the architect risks an over-eager primitive
    # claiming the question and producing a deterministic canned
    # response where a thoughtful conversational reply belongs.
    #
    # ACTION cues (imperative_start, direct_request, second_person_
    # question, continuation, llm_classifier promotion) all fall
    # through to the architect normally — those are explicit commands
    # or polite-imperative forms.
    # Signals that route to the chat path instead of the architect:
    #   * Legacy regex question forms (still used by PTT + wake-word).
    #   * ``vr_converse`` from the voice router — a conversational
    #     reply the LLM judged inappropriate for primitive dispatch.
    _QUESTION_SIGNALS = {
        "wh_question_with_you", "wh_question_opener",
        "vr_converse",
    }
    # ``vr_converse`` only earns the skip when it's an actual LLM
    # verdict (parsed_from content/thinking). The timeout/parse
    # fallbacks default to converse as a SAFETY posture, not a
    # judgment — observed 2026-06-11: the classifier timed out on
    # "can you throw in some music and open a note for me?", the
    # fallback stamped vr_converse, and this gate then blocked the
    # architect from dispatching a clearly-actionable polite request.
    # On degraded turns the architect's own matcher (REJECT-biased)
    # is the better arbiter; an unmatched ask falls through to the
    # conversational path exactly as before.
    _vr_llm_verdict = (
        "vr" in locals()
        and getattr(vr, "parsed_from", "") in ("content", "thinking")
    )
    _skip_architect_for_question = (
        activation_mode == "always_listening"
        and is_becca_path
        and "decision" in locals()
        and decision.signal in _QUESTION_SIGNALS
        and (decision.signal != "vr_converse" or _vr_llm_verdict)
    )
    if _skip_architect_for_question:
        log.info(
            "voice_architect_skipped_question",
            session_id=ctx.session_id,
            signal=decision.signal,
            text_preview=transcript[:80],
        )

    # Architect dispatch — runs BEFORE the legacy intent path.
    #
    # Surface derivation: ``persona_id == "becca"`` flags the companion
    # path (becca-ptt + wake-word). That's the ambient widget context
    # where the user CAN see the screen — architect primitives that
    # need a visual result (image gen, navigation, grove playback) are
    # safe to dispatch.
    #
    # The full-screen voice call (no persona_id) is its own ambient
    # context — avatar + transcript modal, no other UI accessible. We
    # still call the architect with ``surface="voice"`` so primitives
    # explicitly scoped to in-call usage (none today) can fire, but
    # the bulk of the architect's primitives are scoped to ['becca',
    # 'chat'] and won't match here — control falls through to the
    # legacy intent dispatcher (which handles "stop", "repeat", etc.).
    architect_surface = "becca" if getattr(session, "persona_id", "") == "becca" else "voice"
    architect_outcome: ArchitectResult | None = None
    # Router-first dispatch (Phase 1 — see
    # docs/superpowers/specs/2026-05-28-confidence-tier-dispatch-design.md).
    # Gated by architect_router_enabled. Becca path only — voice call modal
    # falls through to the legacy dispatcher and the existing tool/LLM path
    # (call modal UX hasn't yet adopted the tier model). On REJECT or any
    # exception, fall through to the legacy template-as-gate path so the
    # rollout is safe.
    _router_enabled = (
        getattr(settings, "architect_router_enabled", False)
        and is_becca_path
        and not _skip_architect_for_question
    )
    if _router_enabled:
        import time as _time_router
        try:
            template_hint = match_intent(transcript, mode=ctx.mode)
        except Exception as exc:  # noqa: BLE001 — hint is optional
            log.debug("voice_router_template_hint_failed", error=str(exc)[:160])
            template_hint = None
        try:
            _router_refs = get_intent_referents(
                app_state, ctx.user_id, ctx.session_id,
            )
        except Exception:
            _router_refs = None
        _last_disp_age = (
            (_time_router.time() - _router_refs.last_dispatch_ts)
            if _router_refs and _router_refs.last_dispatch_ts else 0.0
        )
        # Parked intent — a clarify question waiting for its answer.
        # Surfaced through the stack so the ROUTER decides whether this
        # utterance fills it (the elegant path: no separate fill-and-
        # dispatch machinery; the decision LLM already owns dispatch).
        _pending = None
        if _router_refs is not None:
            from augmentum.intent.dispatch import get_fresh_pending_intent
            _pending = get_fresh_pending_intent(_router_refs)
        # Presence snapshot — the page on screen / media playing, so
        # "tell me about this page" resolves to the page instead of
        # becoming a literal search for the words "this page".
        _presence: dict = {}
        try:
            from augmentum.companion_runtime.presence_context import now_context
            _presence = await now_context(
                _state_conn(app_state), ctx.user_id, app_state=app_state,
            )
        except Exception:  # noqa: BLE001 — perception is best-effort
            log.debug("voice_presence_snapshot_failed", exc_info=True)
        _stack = ConfidenceStack(
            stt_confidence=1.0,  # STT word-level confidence plumbing — future
            address_signal=decision.signal if "decision" in locals() else "",
            address_confidence=(
                decision.confidence if "decision" in locals() else 0.0
            ),
            speaker_verified=False,  # per-utterance verification plumbing — future
            template_hint_id=template_hint.action_id if template_hint else "",
            template_hint_args=(
                dict(template_hint.args) if template_hint else {}
            ),
            audio_tier_media=_media_active,
            audio_tier_speech_other=False,
            last_dispatch_id=(
                _router_refs.last_dispatch_action or ""
                if _router_refs else ""
            ),
            last_dispatch_args=(
                dict(_router_refs.last_dispatch_args or {})
                if _router_refs else {}
            ),
            last_dispatch_age_s=_last_disp_age,
            active_surface=architect_surface,
            pending_intent_id=(_pending or {}).get("action_id", ""),
            pending_intent_args=dict((_pending or {}).get("args") or {}),
            pending_intent_missing=list((_pending or {}).get("missing") or []),
            pending_intent_question=(_pending or {}).get("question", ""),
            offered_candidates=(
                # Only present a STILL-FRESH offer. A stale one (minutes old)
                # must not be re-offered to the model — that's how an audiobook
                # recommend got replayed for a later "throw in some music".
                list(_router_refs.pending_candidates or [])[:4]
                if (
                    _router_refs
                    and (_time_router.time() - float(getattr(_router_refs, "pending_candidates_at", 0.0) or 0.0))
                    <= _OFFERED_CANDIDATES_TTL_S
                )
                else []
            ),
            # Generic accept-resolution metadata for the offered picks, so a
            # spoken "the second one" re-dispatches the offering verb (coder.
            # delegate → workspace_id, etc.), not the hard-coded media.play.
            offered_intent=(
                getattr(_router_refs, "pending_candidates_intent", "") or ""
                if _router_refs else ""
            ),
            offered_id_field=(
                getattr(_router_refs, "pending_candidates_id_field", "") or ""
                if _router_refs else ""
            ),
            current_page_url=((_presence.get("page") or {}).get("url", "")),
            current_page_title=((_presence.get("page") or {}).get("label", "")),
            now_playing_label=((_presence.get("playing") or {}).get("label", "")),
        )
        # Salvage-don't-discard: the router runs as a TASK with a soft
        # wait. Inside the soft window → dispatch its decision here as
        # before. Past it → the task KEEPS RUNNING (route_utterance's
        # internal timeout, architect_router_timeout_ms, remains the
        # hard hang-guard) and the handle is stashed on the session;
        # the BeccaVoice act-gap consumer awaits it and dispatches the
        # completed decision if the persona path produced no action.
        # Before this, a decision that finished 254ms past the soft
        # deadline was cancelled and DISCARDED while a slower, less
        # reliable path re-derived a worse answer (2026-06-10).
        _soft_wait_s = min(
            2.8,
            float(getattr(settings, "architect_router_timeout_ms", 4000)) / 1000.0,
        )
        router_decision = None
        _router_task = asyncio.create_task(route_utterance(
            transcript,
            app_state=app_state,
            stack=_stack,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
        ))
        try:
            router_decision = await asyncio.wait_for(
                asyncio.shield(_router_task), timeout=_soft_wait_s,
            )
        except TimeoutError:
            log.info(
                "voice_router_soft_timeout_continuing",
                soft_wait_s=_soft_wait_s,
                session_id=ctx.session_id,
            )
            try:
                session.pending_router_task = _router_task
            except Exception:  # noqa: BLE001 — session shape defensive
                _router_task.cancel()
        except Exception as exc:  # noqa: BLE001
            log.warning("voice_router_call_error", error=str(exc)[:200])
            router_decision = None

        if router_decision is not None and router_decision.tier != "REJECT":
            try:
                architect_outcome = await dispatch_router_decision(
                    router_decision,
                    transcript=transcript,
                    surface=architect_surface,
                    session=ctx,
                    app_state=app_state,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "voice_router_dispatch_error", error=str(exc)[:200],
                )
                architect_outcome = None

    # Legacy template-as-gate fallback. Runs when:
    #   * the router flag is off, OR
    #   * the router rejected the utterance, OR
    #   * the router-dispatched primitive returned None (handler opt-out).
    # This preserves the existing architect path until the router has
    # 30+ days of telemetry validating its choices (Phase 4 doc).
    if architect_outcome is None and not _skip_architect_for_question:
        try:
            architect_outcome = await dispatch_architect_command(
                transcript, surface=architect_surface, session=ctx, app_state=app_state,
            )
        except Exception as exc:
            log.warning("voice_architect_dispatch_error", error=str(exc))
            architect_outcome = None

    if architect_outcome is not None:
        match = architect_outcome.match
        result = architect_outcome.action_result
        if result.short_circuit:
            payload = serialize_action_event(match, result)
            # Surface the inferrer's filled args so the UI can show
            # "Putting on Miles Davis" with the actual track context.
            if architect_outcome.inferred_args:
                payload["inferred_args"] = {
                    k: v for k, v in architect_outcome.inferred_args.items()
                    if v not in (None, "", [])
                }
            try:
                await _send_json(websocket, payload)
            except Exception as exc:
                log.warning("voice_architect_emit_failed", error=str(exc))
                return _IntentOutcome(handled=False, transcript=transcript)
            try:
                await _send_json(websocket, {"type": "listening"})
            except Exception:
                log.debug("voice_architect_listening_send_failed", exc_info=True)

            # Persist the turn to session history so Becca remembers
            # she did this on the next conversational turn. Without
            # this, a follow-up like "what did you put on?" hits an
            # empty context because the architect short-circuit
            # bypasses the normal _commit_turn path. The spoken line
            # is what she said aloud — record it verbatim.
            try:
                session.add_user_message(transcript)
                spoken_line = result.speak or result.toast or ""
                if spoken_line:
                    session.add_assistant_message(spoken_line)
            except Exception:
                log.debug("voice_architect_session_persist_failed", exc_info=True)

            log.info(
                "voice_architect_handled",
                action=match.action_id, session_id=ctx.session_id,
            )
            return _IntentOutcome(handled=True, transcript=transcript)
        # Architect handler returned soft-augmentation — fall through
        # to LLM with the addendum applied, same as legacy path.
        if result.prompt_addendum:
            log.info(
                "voice_architect_augmented",
                action=match.action_id,
                addendum_chars=len(result.prompt_addendum),
            )
            augmented = f"{result.prompt_addendum}\n\n{transcript}"
            return _IntentOutcome(handled=False, transcript=augmented)

    # Skip the legacy intent dispatcher when voice_router judged this a
    # conversational utterance, not an action. Without this gate the
    # global intent registry greedily matches verbs like "play" / "find"
    # in conversational asks ("can you play me a song you like?" →
    # grove.play_matching with garbage query). The architect already
    # short-circuits in the same way for vr_act; this mirrors that gate
    # for vr_converse / vr_clarify / vr_idle. PTT + wake-word don't set
    # a vr_* signal so they still hit the legacy dispatcher.
    _skip_legacy_intent = (
        activation_mode == "always_listening"
        and is_becca_path
        and "decision" in locals()
        and decision.signal.startswith("vr_")
        and decision.signal != "vr_act"
    )
    if _skip_legacy_intent:
        log.info(
            "voice_intent_skipped_for_converse",
            session_id=ctx.session_id,
            signal=decision.signal,
            text_preview=transcript[:80],
        )
        outcome = None
    else:
        try:
            outcome = await intent_dispatch(transcript, session=ctx)
        except Exception as exc:
            log.warning("voice_intent_dispatch_error", error=str(exc))
            outcome = None

    if outcome is not None:
        match, result = outcome
        if result.short_circuit:
            payload = serialize_action_event(match, result)
            try:
                await _send_json(websocket, payload)
            except Exception as exc:
                log.warning("voice_intent_emit_failed", error=str(exc))
                return _IntentOutcome(handled=False, transcript=transcript)
            try:
                await _send_json(websocket, {"type": "listening"})
            except Exception:
                log.debug("voice_intent_listening_send_failed", exc_info=True)
            log.info(
                "voice_intent_handled",
                action=match.action_id, session_id=ctx.session_id,
            )
            return _IntentOutcome(handled=True, transcript=transcript)
        # Soft augmentation — prepend the addendum to the user message
        # so the LLM has the recall hits / referent context. The model
        # composes the spoken reply naturally.
        if result.prompt_addendum:
            log.info(
                "voice_intent_augmented",
                action=match.action_id,
                addendum_chars=len(result.prompt_addendum),
            )
            augmented = f"{result.prompt_addendum}\n\n{transcript}"
            return _IntentOutcome(handled=False, transcript=augmented)

    # Note-capture mode: when active, every non-intent utterance is
    # appended to the active note instead of running through UARF.
    # Auto-exits if the deadline (refreshed by each append) has
    # passed — protects against a session that was idle long enough
    # the user moved on. The model-cleanup pass is a follow-up;
    # v1 appends raw transcript.
    refs = get_intent_referents(app_state, ctx.user_id, ctx.session_id)
    if refs.note_capture_mode and refs.active_note_id:
        if (
            refs.note_capture_deadline
            and time.monotonic() > refs.note_capture_deadline
        ):
            log.info(
                "voice_capture_auto_exit",
                session_id=ctx.session_id,
                note_id=refs.active_note_id,
            )
            auto_exit_note_id = refs.active_note_id
            auto_exit_baseline = refs.note_capture_baseline_chars or 0
            refs.note_capture_mode = False
            refs.note_capture_deadline = 0.0
            refs.note_capture_baseline_chars = 0
            # Run cleanup the same way as the explicit end-capture path
            # so an idle-timeout doesn't leave the user with unrefined
            # raw transcript while an explicit "save this" gets the
            # polish. Failures fall through silently.
            cleaned_payload: dict[str, Any] = {"note_id": auto_exit_note_id}
            try:
                from augmentum.intent.capture_cleanup import apply_cleanup_to_note
                changed, new_content = await apply_cleanup_to_note(
                    getattr(app_state, "notes_store", None),
                    auto_exit_note_id,
                    user_id=ctx.user_id,
                    baseline_chars=auto_exit_baseline,
                    app_state=app_state,
                )
                if changed:
                    cleaned_payload["content"] = new_content
            except Exception as exc:
                # Cleanup failure leaves the raw transcript intact (the
                # cleanup helper already short-circuits on its own
                # error paths). Warning level so a recurring outage on
                # the utility model is visible in logs.
                log.warning("voice_capture_autoexit_cleanup_failed", error=str(exc)[:200])
            try:
                await _send_json(websocket, {
                    "type": "intent_action",
                    "v": 1,
                    "action": "note.end_capture",
                    "short_circuit": True,
                    "surface": {
                        "channel": "note.capture_ended",
                        "payload": cleaned_payload,
                    },
                })
            except Exception as exc:
                log.warning(
                    "voice_capture_ended_emit_failed",
                    error=str(exc)[:200],
                )
        elif await _capture_append(
            transcript, refs, websocket, app_state, ctx.user_id,
        ):
            return _IntentOutcome(handled=True, transcript=transcript)

    return _IntentOutcome(handled=False, transcript=transcript)


async def _flush_pending_surface_events(
    websocket: WebSocket,
    app_state: Any,
    user_id: str,
    session_id: str,
) -> None:
    """Drain the per-session ``pending_surface_events`` queue.

    When the LLM invokes an action via tool-calling
    (ActionTool.execute), the surface payload is stashed on the
    session's referent cache because chain.py has no WebSocket
    handle. This helper drains the queue and emits each payload as a
    WS ``intent_action`` event so the frontend router can act —
    typically right before ``turn_complete`` so a sticky note pops
    while Becca is still speaking her confirmation.

    Safe to call repeatedly — drains to empty and never raises.
    """
    if not user_id:
        return
    try:
        refs = get_intent_referents(app_state, user_id, session_id)
    except Exception:
        return
    if not refs.pending_surface_events:
        return
    queue = refs.pending_surface_events
    refs.pending_surface_events = []
    for payload in queue:
        try:
            await _send_json(websocket, payload)
        except Exception as exc:
            log.warning(
                "voice_intent_flush_failed",
                action=payload.get("action"), error=str(exc),
            )
            return


async def _capture_append(
    transcript: str,
    refs: Any,
    websocket: WebSocket,
    app_state: Any,
    user_id: str,
) -> bool:
    """Append a transcript to the active capture-mode note.

    Returns True if the utterance was consumed (turn short-circuits).
    Falls back to letting the caller dispatch normally if the notes
    store is unavailable or the active note has been deleted.
    """
    from datetime import datetime
    notes_store = getattr(app_state, "notes_store", None)
    if notes_store is None:
        return False
    try:
        note = await notes_store.get(refs.active_note_id, user_id=user_id)
    except Exception:
        note = None
    if not note:
        # Active note was deleted under us — exit capture mode quietly.
        refs.note_capture_mode = False
        return False
    content = (note.get("content") or "")
    if content and not content.endswith("\n"):
        content += "\n"
    content += transcript
    try:
        await notes_store.update(
            refs.active_note_id,
            {
                "content": content,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            user_id=user_id,
        )
    except Exception as exc:
        log.warning("voice_capture_append_failed", error=str(exc))
        return False
    # Refresh the idle deadline — user is still actively dictating.
    if refs.note_capture_deadline:
        from augmentum.intent.builtin.notes import _CAPTURE_IDLE_TIMEOUT_S
        refs.note_capture_deadline = time.monotonic() + _CAPTURE_IDLE_TIMEOUT_S
    try:
        await _send_json(websocket, {
            "type": "intent_action",
            "v": 1,
            "action": "note.append",
            "short_circuit": True,
            "surface": {
                "channel": "note.update_sticky",
                "payload": {
                    "note_id": refs.active_note_id,
                    "content": content,
                },
            },
        })
        await _send_json(websocket, {"type": "listening"})
    except Exception:
        return False
    log.info(
        "voice_capture_appended",
        note_id=refs.active_note_id,
        chars=len(transcript),
    )
    return True


def _process_audio_frame_sync(
    frame: bytes,
    speech_enhancer: SpeechEnhancer | None,
    audio_proc: AudioProcessor | None,
    vad: VadProcessor,
) -> tuple[bytes, Any]:
    """Run the synchronous per-frame ML chain off the event loop.

    DTLN denoise, WebRTC noise suppression, and Silero VAD are all
    ONNX/PyTorch models that block the calling thread for a few ms per
    frame in the happy case and tens of ms when CPU is contended (LLM
    prefill, model loads). Running them on the asyncio loop at 31 Hz
    starved DB queries and HTTP handlers and surfaced as `slow_db_op`,
    `slow_request`, and `event_loop_stall` warnings. Batching the three
    into one ``asyncio.to_thread`` keeps the loop responsive while
    preserving frame ordering (the caller awaits per frame).

    Returns the processed frame plus whatever VadEvent the VAD emits.
    """
    if speech_enhancer:
        try:
            frame = speech_enhancer.process_frame(frame)
        except Exception as exc:
            log.debug("speech_enhancer_error", error=str(exc))
    if audio_proc:
        try:
            frame = audio_proc.process_frame(frame)
        except Exception as exc:
            log.warning("audio_proc_error", error=str(exc))
    event = vad.process_frame(frame)
    return frame, event


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    """Real-time voice chat via WebSocket.

    Supports two modes:

    **Server-side VAD** (``voice_server_vad=True``, default):
      Client sends raw PCM16 16 kHz mono frames (1024 bytes = 32 ms each)
      continuously.  The server runs Silero VAD on every frame and detects
      speech boundaries.  When speech ends, the accumulated audio is
      transcribed (streaming STT when available, batch otherwise) and
      processed through the LLM pipeline.

    **Client-side VAD** (``voice_server_vad=False``, legacy):
      Client runs its own spectral VAD and sends WebM/Opus audio with
      JSON control messages (start_recording, stop_recording, vad_speech_end).

    Protocol:
      Client → Server:
        Binary frames: raw PCM16 audio (server VAD) or WebM/Opus (client VAD)
        JSON: {"type":"config", ...} | {"type":"interrupt"} |
              {"type":"start_recording"} | {"type":"stop_recording"} (PTT) |
              {"type":"vad_speech_start"} | {"type":"vad_speech_end"} (client VAD)

      Server → Client:
        Binary frames: TTS audio chunks (MP3)
        JSON: {"type":"listening"} | {"type":"processing"} |
              {"type":"transcript","text":"..."} | {"type":"llm_start"} |
              {"type":"llm_delta","text":"..."} | {"type":"tts_start","sentence":"..."} |
              {"type":"tts_end"} | {"type":"turn_complete","full_text":"..."} |
              {"type":"interrupted"} | {"type":"error","message":"..."} |
              {"type":"vad_state","speaking":bool} |
              {"type":"partial_transcript","text":"..."}
    """
    if not settings.voice_enabled:
        await websocket.close(code=1008, reason="Voice chat is disabled")
        return

    await websocket.accept()

    # Extract user from scope (set by AuthMiddleware via WS ticket)
    user = websocket.scope.get("user")
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    params = websocket.query_params
    session_id = params.get("session_id", "voice_default")
    model = params.get("model", "")
    mode = params.get("mode", "passthrough")
    persona_id = (params.get("persona_id") or "").strip().lower()

    # Optional fanout binding — when the client passes
    # ``voice_session_id``, every emit on this WS is also published
    # to the central VoiceFanout so cast receivers / future renderers
    # can subscribe. Set by the cast-shelf "Call on TV" flow. The
    # rest of the handler uses ``websocket`` identically; the wrapped
    # proxy quacks like a Starlette WebSocket.
    fanout_session_id = (params.get("voice_session_id") or "").strip()
    voice_fanout = getattr(websocket.app.state, "voice_fanout", None)
    if fanout_session_id and voice_fanout is not None:
        from augmentum.voice.fanout import wrap_websocket
        await voice_fanout.open_session(fanout_session_id, user_id=user.id)
        websocket = wrap_websocket(websocket, voice_fanout, fanout_session_id)

    session = VoiceSession(
        session_id=session_id,
        model=model,
        mode=mode,
        user_id=user.id,
        persona_id=persona_id,
    )

    # Compute resolver targets with empty client_caps. Capability frames
    # from the browser will recompute these via the 'capabilities' message
    # handler below. Today targets land at 'server' for every component;
    # legacy dispatch is unchanged. See ``_refresh_pipeline_targets``.
    _refresh_pipeline_targets(session)

    use_server_vad = settings.voice_server_vad
    audio_buffer = bytearray()
    is_recording = False
    # PTT gating for server VAD: when True, client is in PTT mode and
    # the button is not held — server skips VAD processing on incoming frames.
    ptt_gate_closed = False  # False = process audio normally (auto mode default)
    # Post-release grace: for a short window after the PTT button is
    # released, keep the VAD path open so a continuation ("…and also
    # eggs") isn't silently discarded between release and her reply.
    # Speech caught here finalizes normally and coalesces into the
    # in-flight turn when she hasn't started speaking yet.
    _ptt_grace_until: float | None = None

    # Server-side VAD state
    vad: VadProcessor | None = None
    stt_session: StreamingSTTSession | None = None
    batch_stt: BatchSTTFallback | None = None
    stt_provider: dict[str, Any] | None = None
    # Accumulated final transcript fragments from streaming STT
    _final_transcript_parts: list[str] = []
    # PCM audio buffer for batch STT fallback
    _speech_pcm_buffer = bytearray()
    # Safety-valve deadline for smart-turn vetoes — when set, the next
    # frame past this wall-clock time forces _finalize_speech even if
    # smart-turn keeps voting "not done". Prevents the airpods/low-
    # quality-audio case where smart-turn locks the user into an
    # infinite "waiting" loop. Cleared on successful finalize / discard.
    _smart_turn_veto_deadline: float | None = None
    # How many times the veto deadline has been deferred for "still
    # speaking" within the current utterance. Background noise that
    # Silero reads as speech satisfies the is_speaking check and would
    # otherwise defer the deadline forever (the "turn feels super long"
    # report, 2026-06-13). Reset on finalize; capped by
    # voice_smart_turn_max_deferrals.
    _veto_deferral_count: int = 0
    # Frame liveness + watchdog handle for the veto valve. The valve
    # checks live in the frame loop, so a client that stops streaming
    # after a veto (widget capture cycle ending, mic suspend) stranded
    # the buffered utterance forever — observed 2026-06-13. The
    # watchdog task is the no-frames rescue; it defers to the in-loop
    # legs whenever frames are still arriving (no concurrent finalize).
    _last_frame_at: float = time.monotonic()
    _veto_watchdog_task: asyncio.Task | None = None
    # Barge-in candidate: wall-clock when VAD first signalled speech
    # during active TTS. The interrupt only fires once we've seen
    # voice_bargein_min_speech_ms of sustained speech — filters out
    # notification beeps, door slams, and other transients that pass
    # VAD's first frame but don't sustain. Cleared on discard / finalize.
    _barge_in_candidate_start: float | None = None
    # VAD warmup deadline — monotonic timestamp. While now() is less
    # than this, speech_start events from server VAD are suppressed.
    # Set when the client sends start_recording (or implicitly on
    # WS connect for the becca-ptt always-listening path). Without
    # this, the mic activation pop + AGC/AEC settling phase produces
    # a false-positive speech_start in the first 60ms before any real
    # speech has happened — Silero gets stuck "speaking" on silence
    # and the session never finalizes.
    _vad_warmup_until: float = 0.0

    # Client-VAD liveness watchdog. When VAD is client-owned (browser
    # Silero), the server retires its own VadProcessor — so a client VAD
    # that loads but fails at INFERENCE (e.g. an onnxruntime-web version
    # skew throwing "t.getValue is not a function" inside the worklet)
    # goes silently deaf: frames keep arriving with clear voice energy but
    # no ``vad_speech_start`` ever comes, and the load-time fallback in
    # voice.js already passed. These track that case so the frame loop can
    # re-create the server VadProcessor and take over the turn.
    _client_vad_owned_since: float | None = None
    _client_vad_events: int = 0
    _client_vad_voiced_frames: int = 0
    _client_vad_fallback_done: bool = False

    # Set initial VAD warmup window for the becca-companion path. Mic
    # activation pop + AGC/AEC settling produces false-positive VAD
    # trips in the first 60-500ms; suppress speech_start during that
    # window. Other paths (voice call modal) don't have this issue
    # because the user just held PTT or clicked call — they're
    # deliberately starting to speak immediately.
    if getattr(session, "persona_id", "") == "becca":
        _warmup_ms = int(getattr(settings, "companion_always_listening_warmup_ms", 500) or 500)
        if _warmup_ms > 0:
            _vad_warmup_until = time.monotonic() + (_warmup_ms / 1000.0)
        # Companion voice — the widget sends no voice over its config
        # message, so seed it server-side from the user's profile
        # (Companion tab setting, falling back to their default voice).
        # A later explicit client config {voice: ...} still overrides.
        _companion_voice = await _companion_voice_for_user(
            websocket.app.state, user.id,
        )
        if _companion_voice:
            session.voice = _companion_voice
            # Conversation-scoped CSM residency: if her voice is CSM, warm the
            # model now so the first utterance isn't cold, and cache the
            # provider for the unload-on-close in the finally. Inverse:
            # companion_csm_residency = "timer" (sidecar idle timer only) or
            # "always" (warm, never unload). Best-effort; never blocks setup.
            if getattr(settings, "companion_csm_residency", "session") in ("session", "always"):
                try:
                    from augmentum.proxy.audio_routes import (
                        _is_csm_provider,
                        csm_warm,
                        resolve_voice_provider,
                    )
                    from augmentum.state.backends.sqlite import SQLiteBackend
                    _sm = getattr(websocket.app.state, "state_manager", None)
                    _rconn = (
                        _sm.backend.conn
                        if _sm and isinstance(_sm.backend, SQLiteBackend)
                        else None
                    )
                    if _rconn is not None:
                        _prov, _ = await resolve_voice_provider(_rconn, _companion_voice)
                        if _prov and _is_csm_provider(_prov.get("id", "")):
                            session.csm_provider = _prov
                            _w = asyncio.create_task(
                                csm_warm(provider=_prov, user_id=user.id)
                            )
                            _w.add_done_callback(_silence_task_exception)
                except Exception:
                    log.debug("csm_warm_on_open_failed", exc_info=True)

    # Speaker verification state
    speaker_verifier: SpeakerVerifier | None = None
    voiceprint: VoicePrint | None = None
    _needs_enrollment = False
    _frame_count = 0

    # Master bypass — raw mic straight to VAD/STT, no enhancement at all.
    _preprocess_bypass = bool(settings.voice_preprocess_bypass)
    if _preprocess_bypass:
        log.info("voice_preprocess_bypassed",
                 note="raw capture → VAD/STT; all denoise/NS/AGC disabled")

    # Speech enhancer — DTLN neural denoiser + highpass filter (before AGC/VAD)
    speech_enhancer: SpeechEnhancer | None = None
    if not _preprocess_bypass and (
        settings.voice_denoise_enabled or settings.voice_highpass_hz > 0
    ):
        speech_enhancer = SpeechEnhancer(
            highpass_hz=settings.voice_highpass_hz,
            model_dir=settings.voice_denoise_model_dir,
            denoise_enabled=settings.voice_denoise_enabled,
        )
        try:
            await load_model_off_loop(speech_enhancer.load_model)
            log.info("speech_enhancer_ready",
                     neural=speech_enhancer.has_neural,
                     highpass=settings.voice_highpass_hz)
        except Exception as exc:
            log.warning("speech_enhancer_load_failed", error=str(exc))
            speech_enhancer = None

    # Audio preprocessor — AGC + noise suppression before VAD/STT
    audio_proc: AudioProcessor | None = None
    if not _preprocess_bypass and (
        settings.voice_audio_agc or settings.voice_audio_ns
    ):
        audio_proc = AudioProcessor(
            agc_enabled=settings.voice_audio_agc,
            ns_enabled=settings.voice_audio_ns,
            agc_target_dbfs=settings.voice_audio_agc_target_dbfs,
            ns_level=settings.voice_audio_ns_level,
        )

    if use_server_vad:
        # Per-session VAD speech threshold. Becca-companion path with
        # an override gets the higher threshold (mic is always hot in
        # always-listening mode; default 0.5 trips on ambient floor).
        # All other paths use the global default.
        _vad_threshold = settings.voice_vad_speech_threshold
        _vad_prefix_padding = settings.voice_vad_prefix_padding_ms
        if getattr(session, "persona_id", "") == "becca":
            if (getattr(settings, "companion_always_listening_vad_threshold", 0.0) or 0.0) > 0.0:
                _vad_threshold = float(settings.companion_always_listening_vad_threshold)
            # Prefix-padding override — keep more audio BEFORE the
            # detected speech_start so STT doesn't lose the user's
            # leading words. Always-hot mics commonly trip VAD on a
            # mid-utterance loud syllable; the default 300ms drops
            # everything before that.
            _al_prefix = int(getattr(settings, "companion_always_listening_prefix_padding_ms", 0) or 0)
            if _al_prefix > 0:
                _vad_prefix_padding = _al_prefix
        # Endpointing inversion (latency MVP, 2026-06-13): when smart-turn
        # gates the end-of-turn, the VAD endpoints EARLY and smart-turn
        # (~65ms) confirms "actually done" — cutting ~1s of dead air per
        # turn with zero effect on the generated reply. The fast endpoint
        # is only SAFE behind a working gate, so we start at the legacy
        # silence and the smart-turn warmup callback narrows it once the
        # model is confirmed loaded. _fast_endpoint_ms==0 keeps legacy.
        _fast_endpoint_ms = int(getattr(settings, "voice_fast_endpoint_ms", 0) or 0)
        _legacy_silence_ms = settings.voice_silence_threshold_ms
        _want_fast_endpoint = bool(settings.voice_smart_turn) and _fast_endpoint_ms > 0
        vad = VadProcessor(
            speech_threshold=_vad_threshold,
            silence_duration_ms=_legacy_silence_ms,
            min_speech_ms=settings.voice_vad_min_speech_ms,
            prefix_padding_ms=_vad_prefix_padding,
            min_start_frames=settings.voice_vad_min_start_frames,
        )
        try:
            await load_model_off_loop(vad.load_model)
        except Exception as exc:
            log.warning("vad_model_load_failed", error=str(exc))
            use_server_vad = False
            vad = None

        if use_server_vad:
            stt_provider = await _get_stt_config(websocket.app.state)

    # Load speaker verification if enabled
    if settings.voice_speaker_verify:
        try:
            speaker_verifier = SpeakerVerifier(
                verify_threshold=settings.voice_speaker_threshold,
            )
            await load_model_off_loop(speaker_verifier.load_model)
        except Exception as exc:
            log.warning("speaker_verify_init_failed", error=str(exc))
            speaker_verifier = None

        # Check enrollment status regardless of whether verifier loaded —
        # we still want to collect voice samples for later processing
        conn = None
        sm = getattr(websocket.app.state, "state_manager", None)
        if sm:
            from augmentum.state.backends.sqlite import SQLiteBackend
            if isinstance(sm.backend, SQLiteBackend):
                conn = sm.backend.conn
        if conn:
            try:
                from augmentum.proxy.voice_enrollment_routes import (
                    is_enrollment_declined,
                    load_voiceprint,
                )
                # Scope voiceprint lookup to the WS-authenticated user —
                # falling back to IP was the cross-tenant leak.
                ws_user = websocket.scope.get("user")
                ws_uid = ws_user.id if ws_user else ""
                voiceprint = await load_voiceprint(conn, ws_uid)

                if not voiceprint:
                    declined = await is_enrollment_declined(conn, ws_uid)
                    if not declined:
                        _needs_enrollment = True
                        log.info("voice_no_enrollment", user_id=ws_uid)
                    else:
                        log.debug("voice_enrollment_declined", user_id=ws_uid)
            except Exception as exc:
                log.warning("voice_enrollment_check_failed", error=str(exc))

    # Determine if server-side STT is available (Moonshine or remote provider)
    _has_server_stt = bool(
        (settings.voice_moonshine_enabled and MoonshineSTTSession.is_available())
        or stt_provider
    )

    log.info("voice_connected", session_id=session_id, mode=mode,
             server_vad=use_server_vad,
             server_stt=_has_server_stt,
             speaker_verify=speaker_verifier is not None,
             enrolled=voiceprint is not None)
    await _send_json(websocket, {
        "type": "listening",
        "server_vad": use_server_vad,
        "server_stt": _has_server_stt,
        "needs_enrollment": _needs_enrollment,
    })

    # --- Pre-warm cold paths ---
    # Fire-and-forget tasks that run concurrently while the user gets ready
    # to speak.  Each warms a different subsystem so the first voice turn
    # doesn't pay cumulative cold-start latency.

    def _swallow_exc(t: asyncio.Task) -> None:
        """Prevent 'exception was never retrieved' warnings on warmup tasks."""
        if not t.cancelled():
            t.exception()  # mark as retrieved

    _warmup_conn = None
    sm = getattr(websocket.app.state, "state_manager", None)
    if sm:
        from augmentum.state.backends.sqlite import SQLiteBackend
        if isinstance(sm.backend, SQLiteBackend):
            _warmup_conn = sm.backend.conn

    # 1a. Built-in Kokoro TTS — pre-load ONNX model (~88 MB INT8)
    _use_kokoro_builtin = False
    if settings.tts_kokoro_builtin and not settings.tts_kokoro_url:
        KokoroTTS.configure(model_dir=settings.tts_kokoro_model_dir)
        async def _warmup_kokoro_tts() -> None:
            nonlocal _use_kokoro_builtin
            try:
                await load_model_off_loop(KokoroTTS.instance().load_model)
                if KokoroTTS.instance().is_available:
                    _use_kokoro_builtin = True
                    log.info("kokoro_builtin_warmup_complete")
            except Exception as exc:
                log.debug("kokoro_builtin_warmup_failed", error=str(exc))

        _kokoro_warmup = asyncio.create_task(_warmup_kokoro_tts())
        _kokoro_warmup.add_done_callback(_swallow_exc)

    # 1b. External TTS engine — resolve provider, open HTTP pool, force model load
    #     (e.g. Chatterbox loads into VRAM on first request)
    if _warmup_conn and not _use_kokoro_builtin:
        _warmup_voice = session.character_voice or session.voice
        _tts_warmup = asyncio.create_task(
            warmup_tts(_warmup_conn, voice=_warmup_voice, user_id=session.user_id),
        )
        _tts_warmup.add_done_callback(_swallow_exc)

    # 2a. Moonshine local STT — pre-load model (~600MB, 2-5s first time)
    #     so speech_start doesn't block on model init.
    #
    # Moonshine is the BUILT-IN streaming STT. It must not pre-empt a remote
    # STT provider the user explicitly set as default (Speaches/Deepgram/…) —
    # that was the "I selected Speaches but it kept using Moonshine" bug:
    # ``voice_moonshine_enabled`` defaults True, so Moonshine always won and
    # the ``not _use_moonshine`` gate below blocked the chosen provider. Use
    # Moonshine only when the default STT IS the built-in (base_url
    # "builtin") or none is configured; a real remote provider takes over.
    _stt_is_remote = bool(
        stt_provider and stt_provider.get("base_url", "") not in ("", "builtin")
    )
    _use_moonshine = False
    if use_server_vad and settings.voice_moonshine_enabled and not _stt_is_remote:
        MoonshineSTTSession.configure(
            model_path=settings.voice_moonshine_model,
            model_arch=settings.voice_moonshine_arch,
        )
        if MoonshineSTTSession.is_available():
            _use_moonshine = True
            async def _warmup_moonshine() -> None:
                try:
                    await asyncio.to_thread(MoonshineSTTSession.warmup)
                    log.info("moonshine_warmup_complete",
                             model=settings.voice_moonshine_model)
                except Exception as exc:
                    log.debug("moonshine_warmup_failed", error=str(exc))

            _mn_warmup = asyncio.create_task(_warmup_moonshine())
            _mn_warmup.add_done_callback(_swallow_exc)

    # 2b. Streaming STT WebSocket — pre-connect to Deepgram (fallback when
    #     Moonshine isn't available or for non-English).
    if (use_server_vad and not _use_moonshine
            and settings.voice_streaming_stt
            and stt_provider
            and is_streaming_stt_capable(stt_provider.get("base_url", ""))):
        async def _warmup_stt() -> None:
            nonlocal stt_session
            try:
                stt_session = await _open_streaming_stt(
                    stt_provider, _on_stt_transcript,
                )
                if stt_session:
                    log.info("stt_warmup_complete",
                             provider=stt_provider.get("id", ""))
            except Exception as exc:
                log.debug("stt_warmup_failed", error=str(exc))

        _stt_warmup = asyncio.create_task(_warmup_stt())
        _stt_warmup.add_done_callback(_swallow_exc)

    # 3. Embedding model — FastEmbed lazy-loads on first use (~130MB ONNX
    #    model).  Pre-loading it here means recall_and_inject on the first
    #    turn doesn't stall waiting for the model download/init.
    if settings.memory_enabled:
        async def _warmup_embeddings() -> None:
            try:
                from augmentum.memory.embeddings import EmbeddingService
                await load_model_off_loop(EmbeddingService.get_model)
                log.info("embedding_warmup_complete")
            except Exception as exc:
                log.debug("embedding_warmup_failed", error=str(exc))

        _emb_warmup = asyncio.create_task(_warmup_embeddings())
        _emb_warmup.add_done_callback(_swallow_exc)

    # 4. SmartTurn — learned turn-completion model (8MB ONNX, ~12ms inference)
    _smart_turn_available = False
    if settings.voice_smart_turn and use_server_vad:
        async def _warmup_smart_turn() -> None:
            nonlocal _smart_turn_available
            try:
                loaded = await load_model_off_loop(smart_turn.load_model)
                _smart_turn_available = loaded
                if loaded:
                    log.info("smart_turn_warmup_complete")
                else:
                    log.debug("smart_turn_warmup_skipped",
                              note="dependencies not available")
            except Exception as exc:
                log.debug("smart_turn_warmup_failed", error=str(exc))
                loaded = False
            # Endpointing inversion: narrow the VAD endpoint ONLY now that
            # smart-turn is confirmed loaded to gate it. Until this fires
            # (and forever, if the model never loads) the full legacy
            # silence wait protects against premature cutoff.
            if loaded and _want_fast_endpoint and vad is not None:
                vad.silence_duration_ms = _fast_endpoint_ms
                log.info("voice_fast_endpoint_armed",
                         endpoint_ms=_fast_endpoint_ms,
                         legacy_ms=_legacy_silence_ms,
                         session_id=session_id)

        _st_warmup = asyncio.create_task(_warmup_smart_turn())
        _st_warmup.add_done_callback(_swallow_exc)

    # 5. LLM model map — pre-refresh so resolve_backend_for_model doesn't
    #    need to query all backends on the first turn.
    async def _warmup_model_map() -> None:
        try:
            registry = getattr(websocket.app.state, "provider_registry", None)
            if registry:
                await registry.refresh_model_map()
                log.info("model_map_warmup_complete")
        except Exception as exc:
            log.debug("model_map_warmup_failed", error=str(exc))

    _mm_warmup = asyncio.create_task(_warmup_model_map())
    _mm_warmup.add_done_callback(_swallow_exc)

    # --- Streaming STT transcript callback ---
    async def _on_stt_transcript(event: TranscriptEvent) -> None:
        """Handle transcripts from the streaming STT session.

        Applies three hallucination filters based on Whisper-family research:
        1. Duration: segments < 0.3s are unreliable (pre-speech noise)
        2. Word rate: > 8 words/sec is implausibly fast (hallucinated output)
        3. Repetition: repeated words ("okay okay okay") are looping artifacts
        """
        if event.text:
            if event.is_final:
                text = event.text.strip()
                words = text.split()
                word_count = len(words)

                # Filter 1: duration — very short segments are hallucination-prone
                # Moonshine's first ~200ms often produces phantom words from noise.
                if event.duration > 0 and event.duration < 0.3 and word_count <= 2:
                    log.debug("stt_filtered_short_duration",
                              text=text, duration=f"{event.duration:.2f}s")
                    return

                # Filter 2: word rate — implausibly fast output is hallucinated
                # Normal speech is 2-4 words/sec; Moonshine caps at ~6.5 tokens/sec.
                if event.duration > 0 and word_count > 2:
                    words_per_sec = word_count / event.duration
                    if words_per_sec > 8.0:
                        log.debug("stt_filtered_high_word_rate",
                                  text=text[:80], wps=f"{words_per_sec:.1f}")
                        return

                # Filter 3: repetition — looping the same word is a common artifact
                if word_count >= 3:
                    unique = set(w.lower().rstrip(".,!?") for w in words)
                    if len(unique) == 1:
                        log.debug("stt_filtered_repetition", text=text[:80])
                        return

                # Dedup: don't append if identical to last final part
                if not _final_transcript_parts or _final_transcript_parts[-1] != text:
                    _final_transcript_parts.append(text)
            # Send partial transcripts to the client for live display
            await _send_json(websocket, {
                "type": "partial_transcript",
                "text": event.text,
                "is_final": event.is_final,
            })

    # --- Helper: finalize speech and process turn ---
    _finalize_in_flight = False

    async def _finalize_speech() -> None:
        """Guard + guaranteed-release wrapper around _finalize_speech_impl.

        The impl has several early returns — speaker-reject, hallucination
        drop, and staging park — each of which used to `return` WITHOUT
        clearing the in-flight latch (it was reset only on the fall-through
        dispatch path). Any one of them left `_finalize_in_flight` stuck
        True, so the guard below deafened EVERY subsequent utterance:
        first turn works, all following turns silently drop before STT.
        The staging park made this fire on every stage-manager turn.
        Fix-the-class: release the latch in a finally — ONE release point,
        so no current or future early return (or exception) can leak it.
        """
        nonlocal _finalize_in_flight

        if _finalize_in_flight:
            log.debug("finalize_speech_skipped_already_in_flight")
            return
        _finalize_in_flight = True
        try:
            await _finalize_speech_impl()
        finally:
            _finalize_in_flight = False

    async def _finalize_speech_impl() -> None:
        """Called when server VAD detects speech_end.

        Verifies the speaker (if enrolled), closes the streaming STT
        session (if any), assembles the final transcript, and dispatches
        the LLM turn. The in-flight latch is owned by _finalize_speech;
        this body may return early freely.
        """
        nonlocal stt_session, batch_stt, _smart_turn_veto_deadline, _barge_in_candidate_start
        nonlocal _veto_deferral_count

        # Latency waterfall: this is t0 (the endpoint). Fresh slate.
        session.turn_timing = {}
        _turn_stamp(session, "speech_end")
        _veto_deferral_count = 0  # fresh slate for the next utterance

        # Speaker verification: check if this speech belongs to the enrolled user
        # Skip verification for very short audio — produces unreliable embeddings
        min_verify_bytes = int(settings.voice_speaker_verify_seconds * SAMPLE_RATE * 2)
        if (speaker_verifier and voiceprint
                and len(_speech_pcm_buffer) >= min_verify_bytes):
            try:
                verify_audio = bytes(_speech_pcm_buffer)
                score = await asyncio.to_thread(
                    speaker_verifier.verify, verify_audio, voiceprint,
                )
                # Use quality-adaptive threshold: high-quality enrollments
                # get stricter, low-quality get more lenient
                threshold = speaker_verifier.effective_threshold(voiceprint)
                log.info("speaker_verify", score=f"{score:.4f}",
                         threshold=f"{threshold:.4f}",
                         quality=f"{voiceprint.quality_score:.4f}",
                         audio_seconds=f"{len(verify_audio) / (SAMPLE_RATE * 2):.1f}")

                if score < threshold:
                    # Not the enrolled user — discard this speech
                    log.info("speaker_rejected", score=f"{score:.4f}",
                             threshold=f"{threshold:.4f}")
                    await _send_json(websocket, {
                        "type": "speaker_rejected",
                        "score": round(score, 4),
                    })
                    # Clean up STT resources
                    if stt_session:
                        try:
                            await asyncio.wait_for(stt_session.close(), timeout=5.0)
                        except (TimeoutError, Exception) as exc2:
                            log.warning("stt_close_timeout", error=str(exc2))
                        stt_session = None
                    batch_stt = None
                    _final_transcript_parts.clear()
                    _speech_pcm_buffer.clear()
                    await _send_json(websocket, {"type": "listening"})
                    return
            except Exception as exc:
                log.warning("speaker_verify_failed", error=str(exc))
                # Fail open — allow the turn to proceed

        await _send_json(websocket, {"type": "processing"})

        transcript = ""

        if stt_session:
            # Close streaming STT and give it a moment for final transcript
            try:
                await asyncio.wait_for(stt_session.close(), timeout=5.0)
            except (TimeoutError, Exception) as exc:
                log.warning("stt_close_timeout", error=str(exc))
            stt_session = None
            # Yield to event loop so any pending transcript callbacks
            # from send_audio() (dispatched via create_task) complete
            # before we read _final_transcript_parts.
            await asyncio.sleep(0)
            transcript = " ".join(_final_transcript_parts).strip()

            # Empty-streaming fallback: streaming STT sometimes returns
            # nothing on short or quiet utterances even when VAD+smart-turn
            # cleared the audio. Re-transcribe the captured PCM via the
            # batch path before giving up — turning a silent drop into a
            # real transcript on the cases that matter.
            min_stt_bytes = int(0.5 * SAMPLE_RATE * 2)
            if not transcript and len(_speech_pcm_buffer) >= min_stt_bytes:
                conn = None
                sm = getattr(websocket.app.state, "state_manager", None)
                if sm:
                    from augmentum.state.backends.sqlite import SQLiteBackend
                    if isinstance(sm.backend, SQLiteBackend):
                        conn = sm.backend.conn
                if conn:
                    try:
                        pcm_audio = bytes(_speech_pcm_buffer)
                        if settings.voice_stt_normalize:
                            pcm_audio = normalize_pcm(pcm_audio)
                        wav_audio = _pcm_to_wav(pcm_audio)
                        log.info("stt_streaming_empty_batch_retry",
                                 pcm_bytes=len(pcm_audio),
                                 duration_ms=int(len(pcm_audio) / (SAMPLE_RATE * 2) * 1000))
                        fallback = await asyncio.wait_for(
                            transcribe_audio(
                                wav_audio, conn, filename="recording.wav",
                                user_id=session.user_id,
                            ),
                            timeout=15.0,
                        )
                        if fallback:
                            transcript = fallback.strip()
                            log.info("stt_streaming_empty_batch_recovered",
                                     chars=len(transcript))
                    except Exception as exc:
                        log.warning("stt_streaming_empty_batch_failed",
                                    error=str(exc))
        elif batch_stt:
            # Batch fallback: transcribe accumulated PCM
            pcm_audio = bytes(_speech_pcm_buffer)
            batch_stt = None

            # Peak-normalize before STT — biggest single improvement for
            # quiet speakers (15-30% WER reduction on Whisper).
            if settings.voice_stt_normalize:
                pcm_audio = normalize_pcm(pcm_audio)

            # Need at least 500ms of audio for reliable STT
            min_stt_bytes = int(0.5 * SAMPLE_RATE * 2)
            if len(pcm_audio) >= min_stt_bytes:
                conn = None
                sm = getattr(websocket.app.state, "state_manager", None)
                if sm:
                    from augmentum.state.backends.sqlite import SQLiteBackend
                    if isinstance(sm.backend, SQLiteBackend):
                        conn = sm.backend.conn
                if conn:
                    try:
                        # Wrap raw PCM16 as WAV for the batch STT API
                        wav_audio = _pcm_to_wav(pcm_audio)
                        log.info("batch_stt_start",
                                 pcm_bytes=len(pcm_audio),
                                 wav_bytes=len(wav_audio),
                                 duration_ms=int(len(pcm_audio) / (SAMPLE_RATE * 2) * 1000))
                        transcript = await asyncio.wait_for(
                            transcribe_audio(
                                wav_audio, conn, filename="recording.wav",
                                user_id=session.user_id,
                            ),
                            timeout=15.0,
                        )
                        log.info("batch_stt_result",
                                 transcript=transcript[:200] if transcript else "(empty)")
                    except Exception as exc:
                        log.warning("batch_stt_error", error=str(exc))
            else:
                log.info("batch_stt_too_short",
                         pcm_bytes=len(pcm_audio),
                         min_bytes=min_stt_bytes,
                         duration_ms=int(len(pcm_audio) / (SAMPLE_RATE * 2) * 1000))

        # Stash the user's utterance for cross-speaker CSM context BEFORE we
        # clear the buffer — the companion turn (if CSM is her voice) feeds
        # it to the sidecar so her prosody reacts to how they sounded. Cheap
        # bytes copy; only consumed on the companion path, ignored otherwise.
        if transcript:
            _turn_stamp(session, "stt_done")

        if transcript and _speech_pcm_buffer:
            session.last_user_audio = bytes(_speech_pcm_buffer)
            session.last_user_audio_sr = SAMPLE_RATE

        _final_transcript_parts.clear()
        _speech_pcm_buffer.clear()
        _smart_turn_veto_deadline = None  # fresh slate for next utterance
        _barge_in_candidate_start = None

        # Capture provenance for the address gate: a press- or wake-
        # initiated capture is the user deliberately speaking to the
        # companion; an auto re-arm of the open always-listening mic
        # (or a follow-up window) is not, and keeps ambient gating.
        session.last_utterance_explicit = (
            getattr(session, "capture_source", "") in ("ptt", "wake")
        )

        # STT hallucination gate — Moonshine/Whisper invent phantom text
        # ("Thank you.", "Thanks for watching") from mic bumps, breaths,
        # and silence; they pass the duration/rate/repetition filters
        # because they're short, normal-rate, real words. A lone phantom
        # must not become a turn — it makes her speak to no one and, left
        # in history, poisons the next turn's context. Explicit ptt/wake
        # captures keep the ambiguous set (the user deliberately spoke).
        if transcript and _is_stt_hallucination(
            transcript, explicit=session.last_utterance_explicit,
        ):
            log.info(
                "voice_stt_hallucination_dropped",
                text=transcript[:60],
                explicit=session.last_utterance_explicit,
            )
            try:
                await _send_json(websocket, {"type": "listening"})
            except Exception:  # noqa: BLE001 — WS may be gone
                log.debug("voice_hallucination_listening_send_failed")
            return

        if transcript:
            # Real words vindicate any provisional barge-in — the
            # interrupt was the user actually speaking. Drop the
            # rollback state so a later noise blip can't resurrect the
            # interrupted reply over this new turn, and shrink the
            # interrupted reply's history entry to what was actually
            # heard (audio-only surfaces stash this at commit time).
            session.bargein_pending = False
            session.undelivered_tts = []
            _apply_pending_heard_rewrite(session)

        # Staging mode: send transcript to client for editing, don't start LLM.
        # The client will send a "stage_send" message with the edited text.
        if session.staging:
            await _send_json(websocket, {
                "type": "transcript", "text": transcript, "is_final": True,
            })
            await _send_json(websocket, {"type": "listening"})
            return

        # Continuation coalescing: a new utterance landing while the
        # PRIOR turn is still thinking (no audible reply yet) is the same
        # thought split by a VAD pause — "add milk to the list" … "and
        # also eggs" — not a barge-in. Dispatching the fragment alone
        # made the halves collide as two competing turns (second one
        # context-free, first one cancelled half-done). Merge them and
        # re-dispatch as ONE turn.
        #
        # Gate on tts_started, NOT is_speaking: is_speaking flips True at
        # llm_start (before ANY audio), so the old gate slammed the merge
        # window shut the instant she started THINKING — leaving an
        # ~800ms hole (widened by the fast endpoint + ack clips) where a
        # continuation was dispatched alone and lost its first half
        # ("can't finish my thought", 2026-06-13). tts_started flips only
        # when she actually makes REPLY sound, which is the real boundary
        # the comment always intended — and matches the barge-in arming,
        # which already requires tts_started. An ack clip never sets
        # tts_started, so "mm-hm" + continuation correctly merges.
        dispatch_text = transcript
        if session.current_task and not session.current_task.done():
            prev_text = str(getattr(session, "pending_turn_transcript", "") or "")
            if transcript and prev_text and not session.tts_started:
                dispatch_text = f"{prev_text} {transcript}".strip()
                # The merged turn inherits the FIRST half's provenance:
                # a PTT-opened thought stays explicit even when the
                # continuation arrived through the ambient-gated
                # follow-up mic.
                session.last_utterance_explicit = (
                    session.last_utterance_explicit
                    or bool(getattr(session, "pending_turn_explicit", False))
                )
                log.info(
                    "voice_turn_coalesced",
                    session_id=session_id,
                    prev_preview=prev_text[:60],
                    new_preview=transcript[:60],
                )
            session.interrupted = True
            session.current_task.cancel()
            await asyncio.sleep(0.05)

        session.pending_turn_transcript = dispatch_text
        session.pending_turn_explicit = bool(session.last_utterance_explicit)

        # Latency mask: speak a short, varied ack the instant a
        # DELIBERATE turn dispatches (ptt/wake — never ambient/auto, or
        # she'd "mm-hm" at overheard speech). Inline so it cleanly
        # precedes the reply on the client's playback queue; the ~150ms
        # it costs is masked by the ack itself. No-op unless enabled +
        # in-process Kokoro voice.
        if session.last_utterance_explicit:
            await _speak_ack_clip(
                websocket, _state_conn(websocket.app.state), session,
            )
            _turn_stamp(session, "ack")

        _turn_stamp(session, "dispatch")
        session.current_task = asyncio.create_task(
            _process_voice_turn_from_transcript(
                dispatch_text, websocket, session, websocket.app.state,
            )
        )
        # NB: _finalize_in_flight is released by the _finalize_speech wrapper's
        # finally — not here — so every early return above is covered too.

    async def _veto_watchdog() -> None:
        """Frame-independent rescue for a smart-turn-vetoed utterance.

        Every other valve lives in the audio-frame loop, so a client
        that stops (or starves) its PCM stream after a veto stranded
        the buffered speech forever — the user's words were heard,
        parked as a continuation, and never dispatched (2026-06-13,
        becca widget). This task enforces the same deadline on wall
        clock. It only acts when frames have actually stalled (>1s
        since the last one): while frames flow, the in-loop legs own
        finalization and this task must not race them.
        """
        nonlocal _smart_turn_veto_deadline
        try:
            while True:
                deadline = _smart_turn_veto_deadline
                if deadline is None:
                    return
                now = time.monotonic()
                if now < deadline:
                    await asyncio.sleep(min(deadline - now + 0.1, 1.0))
                    continue
                if (time.monotonic() - _last_frame_at) <= 1.0:
                    # Frames flowing — the in-loop valve will catch it.
                    await asyncio.sleep(0.5)
                    continue
                if vad is not None and vad.is_speaking:
                    await asyncio.sleep(0.5)
                    continue
                log.info(
                    "smart_turn_veto_deadline_reached",
                    path="watchdog",
                    buffer_bytes=len(_speech_pcm_buffer),
                )
                _smart_turn_veto_deadline = None
                import contextlib as _ctxlib
                with _ctxlib.suppress(Exception):  # WS may be gone
                    await _send_json(websocket, {
                        "type": "vad_state", "speaking": False,
                    })
                await _finalize_speech()
                return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.warning("smart_turn_veto_watchdog_failed", exc_info=True)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ----- Binary frame: audio data -----
            if "bytes" in message and message["bytes"]:
                raw = message["bytes"]
                _frame_count += 1
                _last_frame_at = time.monotonic()
                if _frame_count == 1:
                    log.info("voice_first_audio_frame",
                             bytes=len(raw), session_id=session_id)
                elif _frame_count % 500 == 0:  # ~16s of audio
                    log.info("voice_audio_frames",
                             count=_frame_count, session_id=session_id)

                # Client-VAD liveness watchdog. VAD is client-owned but the
                # browser has sent zero speech events — if clearly-voiced
                # audio keeps arriving, the client Silero/ORT is broken at
                # inference and the call is going deaf. Re-create the server
                # VadProcessor and take over. One-shot; a working client VAD
                # (``_client_vad_events`` > 0) never reaches this.
                if (not use_server_vad
                        and not _client_vad_fallback_done
                        and _client_vad_owned_since is not None
                        and _client_vad_events == 0
                        and len(raw) >= 2):
                    try:
                        _s = np.frombuffer(raw, dtype=np.int16)
                        _rms = float(np.sqrt(np.mean(_s.astype(np.float32) ** 2))) if _s.size else 0.0
                    except Exception:
                        _rms = 0.0
                    # silence reads ~5-50 RMS, speech ~10000 (see vad_diagnostic).
                    if _rms > 600.0:
                        _client_vad_voiced_frames += 1
                    _cv_elapsed = time.monotonic() - _client_vad_owned_since
                    # ~1.5s of voiced audio (≈47 × 32ms frames) + a 3s grace
                    # for client VAD to prove itself = it isn't working.
                    if _client_vad_voiced_frames >= 47 and _cv_elapsed > 3.0:
                        _client_vad_fallback_done = True
                        log.warning(
                            "voice_client_vad_watchdog_fallback",
                            session_id=session_id,
                            voiced_frames=_client_vad_voiced_frames,
                            elapsed_s=round(_cv_elapsed, 1),
                        )
                        try:
                            vad = VadProcessor(
                                speech_threshold=settings.voice_vad_speech_threshold,
                                silence_duration_ms=settings.voice_silence_threshold_ms,
                                min_speech_ms=settings.voice_vad_min_speech_ms,
                                prefix_padding_ms=settings.voice_vad_prefix_padding_ms,
                                min_start_frames=settings.voice_vad_min_start_frames,
                            )
                            await load_model_off_loop(vad.load_model)
                            if stt_provider is None:
                                stt_provider = await _get_stt_config(websocket.app.state)
                            use_server_vad = True
                            try:
                                await _send_json(websocket, {"type": "vad_fallback", "source": "server"})
                            except Exception:
                                log.debug("vad_fallback_notify_failed")
                        except Exception as exc:
                            log.warning("voice_server_vad_recreate_failed", error=str(exc))
                            vad = None

                if use_server_vad and vad:
                    # Smart-turn safety valve, gate-independent leg: a
                    # vetoed utterance must be rescuable even when the
                    # PTT gate is closed — the per-frame check below
                    # sits past the gate's ``continue``, which stranded
                    # vetoed speech forever once the grace window
                    # expired (observed 2026-06-13: veto at prob 0.007,
                    # then silence — turn never finalized). The
                    # watchdog task is the no-frames-at-all rescue;
                    # this leg covers frames-flowing-but-gated.
                    if (_smart_turn_veto_deadline is not None
                            and time.monotonic() >= _smart_turn_veto_deadline
                            and not vad.is_speaking):
                        log.info("smart_turn_veto_deadline_reached",
                                 path="gated_frame",
                                 buffer_bytes=len(_speech_pcm_buffer))
                        _smart_turn_veto_deadline = None
                        await _send_json(websocket, {
                            "type": "vad_state", "speaking": False,
                        })
                        await _finalize_speech()
                        continue

                    # PTT gate: skip VAD when PTT mode is active but button
                    # is released — EXCEPT inside the post-release grace
                    # window, where continuation speech still flows.
                    if ptt_gate_closed:
                        if (
                            _ptt_grace_until is not None
                            and time.monotonic() < _ptt_grace_until
                        ):
                            pass  # grace — keep listening
                        else:
                            _ptt_grace_until = None
                            audio_buffer.clear()
                            continue

                    # Server-side VAD: process each 32 ms PCM16 frame
                    audio_buffer.extend(raw)

                    while len(audio_buffer) >= FRAME_BYTES:
                        frame = bytes(audio_buffer[:FRAME_BYTES])
                        del audio_buffer[:FRAME_BYTES]

                        # Barge-in confirmation: a speech_start during
                        # TTS armed the candidate timestamp; once we've
                        # observed voice_bargein_min_speech_ms of actual
                        # VOICED audio, trip the interrupt. This is what
                        # kills the "notification beep cancelled my TTS"
                        # failure mode.
                        #
                        # Measured as vad.voiced_ms, NOT wall-clock since
                        # the candidate armed: vad.is_speaking stays True
                        # through the TRAILING state, so elapsed time
                        # counts trailing silence as "sustained speech" —
                        # a ~100ms blip plus its silence tail used to
                        # confirm a barge-in that the segment-end check
                        # then discarded as sub-min_speech_ms noise.
                        # bargein_pending marks the interrupt as
                        # provisional so the discard / empty-STT paths
                        # can roll it back and replay drained TTS.
                        if (_barge_in_candidate_start is not None
                                and session.is_speaking
                                and vad.is_speaking):
                            voiced_ms = vad.voiced_ms
                            if voiced_ms >= settings.voice_bargein_min_speech_ms:
                                log.info("bargein_confirmed",
                                         voiced_ms=int(voiced_ms),
                                         min_ms=settings.voice_bargein_min_speech_ms)
                                session.interrupted = True
                                session.bargein_pending = True
                                _barge_in_candidate_start = None

                        # Smart-turn safety valve: if a prior speech_end
                        # was vetoed and the deadline has passed, force
                        # finalization now. Catches the airpods/low-quality-
                        # audio case where smart-turn keeps voting "not
                        # done" and the user has no way out.
                        if (_smart_turn_veto_deadline is not None
                                and time.monotonic() >= _smart_turn_veto_deadline):
                            _max_deferrals = int(getattr(
                                settings, "voice_smart_turn_max_deferrals", 3,
                            ) or 0)
                            if _should_defer_veto(
                                _veto_deferral_count, _max_deferrals,
                                vad is not None and vad.is_speaking,
                            ):
                                # VAD still reads speech at the deadline.
                                # If that's a genuine resume, forcing
                                # finalization would cut a mid-word
                                # (observed 2026-06-11) — so defer. BUT
                                # background noise Silero misreads as
                                # speech satisfies this too and would
                                # defer forever ("turn feels super long",
                                # 2026-06-13); the deferral cap bounds it.
                                # A real multi-pause thought past the cap
                                # is recoverable via continuation-merge.
                                _veto_deferral_count += 1
                                _smart_turn_veto_deadline = (
                                    time.monotonic()
                                    + settings.voice_smart_turn_max_wait_s
                                )
                                log.info("smart_turn_veto_deadline_deferred",
                                         reason="user_resumed_speaking",
                                         deferral=_veto_deferral_count,
                                         max_deferrals=_max_deferrals)
                            else:
                                log.info("smart_turn_veto_deadline_reached",
                                         waited_s=settings.voice_smart_turn_max_wait_s,
                                         buffer_bytes=len(_speech_pcm_buffer))
                                _smart_turn_veto_deadline = None
                                await _send_json(websocket, {
                                    "type": "vad_state", "speaking": False,
                                })
                                await _finalize_speech()
                                continue

                        # DTLN + WebRTC + Silero in one thread hop so the
                        # event loop isn't blocked at 31 Hz. See
                        # _process_audio_frame_sync for why this matters.
                        frame, event = await asyncio.to_thread(
                            _process_audio_frame_sync,
                            frame, speech_enhancer, audio_proc, vad,
                        )

                        # Diagnostic: log audio energy and VAD state every 100 frames
                        if not ptt_gate_closed and _frame_count % 100 == 0:
                            import struct
                            samples = struct.unpack(f'<{len(frame)//2}h', frame)
                            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                            peak = max(abs(s) for s in samples)
                            log.info("vad_diagnostic",
                                     frame=_frame_count, rms=f"{rms:.0f}",
                                     peak=peak, vad_speaking=vad.is_speaking,
                                     ptt_open=not ptt_gate_closed)

                        if event and event.kind == "speech_start":
                            # Warmup suppression: discard speech_start during
                            # the first N ms after session start. AGC/AEC are
                            # still settling and Silero often false-trips on
                            # mic activation pop / DC offset / ambient floor.
                            # Without this, the session can lock in "speaking"
                            # state forever waiting for a speech_end that
                            # never comes.
                            if _vad_warmup_until and time.monotonic() < _vad_warmup_until:
                                remaining_ms = int((_vad_warmup_until - time.monotonic()) * 1000)
                                log.info(
                                    "server_vad_warmup_suppressed",
                                    session_id=session_id,
                                    remaining_ms=remaining_ms,
                                )
                                vad.soft_reset()
                                continue

                            # Echo suppression: ignore speech_start shortly after
                            # TTS ends — the mic likely picked up speaker output.
                            _echo_cooldown_s = 0.8  # 800ms post-TTS suppression
                            if (session.tts_ended_at
                                    and (time.monotonic() - session.tts_ended_at) < _echo_cooldown_s):
                                log.info("server_vad_echo_suppressed",
                                         session_id=session_id,
                                         elapsed=f"{time.monotonic() - session.tts_ended_at:.2f}s")
                                vad.soft_reset()
                                continue

                            log.info("server_vad_speech_start",
                                     session_id=session_id)
                            await _send_json(websocket, {
                                "type": "vad_state", "speaking": True,
                            })

                            # Barge-in candidate. We DEFER setting
                            # ``session.interrupted = True`` until we have
                            # voice_bargein_min_speech_ms of sustained
                            # speech — see the per-frame check at the top
                            # of the inner loop. This filters notification
                            # beeps, door slams, and other transient
                            # noises that pass VAD's first frame but don't
                            # sustain. If the noise truly continues, the
                            # interrupt fires after the gate; if not, the
                            # speech_discard handler clears the candidate
                            # and TTS keeps playing.
                            #
                            # Note: STT session still opens immediately
                            # below — no audio is lost if it turns out to
                            # be real speech.
                            #
                            # tts_started gates arming: is_speaking flips
                            # True at llm_start, before any audio exists
                            # (prompt eval can run >1s). A mic blip in
                            # that window used to arm + confirm a barge-in
                            # against a reply that hadn't made a sound —
                            # there is nothing to barge in on until TTS
                            # has emitted at least one sentence.
                            if session.is_speaking and session.tts_started:
                                _barge_in_candidate_start = time.monotonic()

                            # Veto continuation: an honored smart-turn
                            # veto kept the STT session + buffers alive
                            # so the user's resumed speech EXTENDS the
                            # utterance. Clearing here destroyed the
                            # first segment's finals + audio — the
                            # "fumbles, loses the first half" failure
                            # (2026-06-11). Only a genuinely fresh
                            # utterance starts from empty.
                            _continuing = (
                                stt_session is not None
                                or (_smart_turn_veto_deadline is not None
                                    and batch_stt is not None)
                            )
                            if not _continuing:
                                _final_transcript_parts.clear()
                                _speech_pcm_buffer.clear()

                            # Include prefix audio for STT context —
                            # clamped to post-TTS audio so a generous
                            # ring can't pull her own voice into the
                            # user's transcript.
                            prefix = _trim_prefix_to_post_tts(
                                vad.get_prefix_audio(),
                                session.tts_ended_at,
                                time.monotonic(),
                            )
                            if prefix:
                                _speech_pcm_buffer.extend(prefix)

                            # Open streaming STT — priority: Moonshine > Deepgram > batch
                            # Moonshine: local, native streaming, lowest latency
                            if _use_moonshine and not stt_session:
                                try:
                                    stt_session = MoonshineSTTSession(
                                        on_transcript=_on_stt_transcript,
                                    )
                                    await stt_session.connect()
                                    if not stt_session._connected:
                                        stt_session = None
                                except Exception as exc:
                                    log.warning("moonshine_open_failed", error=str(exc))
                                    stt_session = None

                            # Deepgram: cloud streaming fallback
                            if (not stt_session
                                    and settings.voice_streaming_stt
                                    and stt_provider
                                    and is_streaming_stt_capable(
                                        stt_provider.get("base_url", ""))):
                                stt_session = await _open_streaming_stt(
                                    stt_provider, _on_stt_transcript,
                                )

                            if stt_session and prefix:
                                await stt_session.send_audio(prefix)

                            # Batch fallback: accumulate frames, transcribe on speech_end.
                            # A veto continuation keeps the existing
                            # accumulator — replacing it would drop the
                            # pre-pause segment.
                            if not stt_session:
                                if batch_stt is None or not _continuing:
                                    batch_stt = BatchSTTFallback()
                                if prefix:
                                    batch_stt.add_frame(prefix)

                        # Forward audio to STT during speech
                        if vad.is_speaking:
                            # Enforce max audio length to prevent unbounded memory growth
                            max_pcm_bytes = settings.voice_max_audio_seconds * SAMPLE_RATE * 2
                            if len(_speech_pcm_buffer) < max_pcm_bytes:
                                _speech_pcm_buffer.extend(frame)
                            else:
                                # Max audio reached — capture partial transcript
                                # and push to stage manager instead of losing speech.
                                log.info("voice_max_audio_reached",
                                         seconds=settings.voice_max_audio_seconds)
                                vad.reset()
                                await _send_json(websocket, {
                                    "type": "vad_state", "speaking": False,
                                })

                                # Close STT to flush final transcript
                                partial = ""
                                if stt_session:
                                    try:
                                        await asyncio.wait_for(stt_session.close(), timeout=5.0)
                                    except (TimeoutError, Exception):
                                        pass
                                    stt_session = None
                                    await asyncio.sleep(0)
                                    partial = " ".join(_final_transcript_parts).strip()
                                _final_transcript_parts.clear()
                                _speech_pcm_buffer.clear()

                                # Send to client — stage manager auto-activates
                                # with the partial transcript so user can review & send
                                await _send_json(websocket, {
                                    "type": "max_audio_reached",
                                    "transcript": partial,
                                    "seconds": settings.voice_max_audio_seconds,
                                })
                                await _send_json(websocket, {"type": "listening"})
                                continue
                            if stt_session:
                                await stt_session.send_audio(frame)
                            elif batch_stt:
                                batch_stt.add_frame(frame)

                        if event and event.kind == "speech_end":
                            log.info("server_vad_speech_end",
                                     session_id=session_id)

                            # SmartTurn: check if the user is actually done
                            # speaking or just pausing mid-thought.
                            if (settings.voice_smart_turn
                                    and _smart_turn_available
                                    and len(_speech_pcm_buffer) > 3200):  # >100ms of audio
                                try:
                                    turn_audio = np.frombuffer(
                                        bytes(_speech_pcm_buffer), dtype=np.int16,
                                    ).astype(np.float32) / 32768.0
                                    is_complete, prob = await smart_turn.predict_turn_complete_async(
                                        turn_audio,
                                        threshold=settings.voice_smart_turn_threshold,
                                    )
                                    log.info("smart_turn_result",
                                             complete=is_complete,
                                             prob=f"{prob:.3f}",
                                             audio_s=f"{len(turn_audio)/16000:.1f}")
                                    if not is_complete:
                                        # Confidence floor: borderline vetoes
                                        # (prob just under the threshold)
                                        # shouldn't override a clean VAD
                                        # end-of-speech — trust VAD. Veto
                                        # confidence = threshold − prob (see
                                        # _smart_turn_veto_confidence): prob
                                        # 0.03 is a near-CERTAIN "still
                                        # talking", prob 0.45 is the coin
                                        # flip. The original comparison used
                                        # prob directly and overrode exactly
                                        # the most confident vetoes —
                                        # observed 2026-06-11 as hard mid-
                                        # sentence cutoffs. The runaway-veto
                                        # safety net is the deadline below,
                                        # not this gate.
                                        min_conf = settings.voice_smart_turn_min_veto_confidence
                                        veto_conf = _smart_turn_veto_confidence(
                                            prob, settings.voice_smart_turn_threshold,
                                        )
                                        if veto_conf < min_conf:
                                            log.info("smart_turn_veto_overridden_low_confidence",
                                                     prob=f"{prob:.3f}",
                                                     veto_conf=f"{veto_conf:.3f}",
                                                     min_conf=f"{min_conf:.3f}")
                                            # Fall through to normal finalization
                                        else:
                                            # User is still thinking — resume listening
                                            # without finalizing. Keep STT session and
                                            # PCM buffer intact for the continuation.
                                            # Arm the safety-valve deadline on the
                                            # first veto so we eventually force
                                            # finalization if smart-turn keeps
                                            # vetoing (or if user stays silent).
                                            if _smart_turn_veto_deadline is None:
                                                _smart_turn_veto_deadline = (
                                                    time.monotonic()
                                                    + settings.voice_smart_turn_max_wait_s
                                                )
                                                # Fresh veto sequence —
                                                # zero the deferral cap
                                                # counter (robust against
                                                # a stale count surviving
                                                # a prior discard).
                                                _veto_deferral_count = 0
                                            # Wall-clock rescue for the
                                            # frames-stop case — see
                                            # _veto_watchdog.
                                            if (_veto_watchdog_task is None
                                                    or _veto_watchdog_task.done()):
                                                _veto_watchdog_task = (
                                                    asyncio.create_task(
                                                        _veto_watchdog(),
                                                    )
                                                )
                                            vad.soft_reset()  # reset silence timer, keep speech state
                                            await _send_json(websocket, {
                                                "type": "vad_state", "speaking": False,
                                            })
                                            await _send_json(websocket, {
                                                "type": "status",
                                                "stage": "waiting",
                                            })
                                            continue
                                except Exception as exc:
                                    log.warning("smart_turn_error", error=str(exc))
                                    # Fall through to normal finalization

                            await _send_json(websocket, {
                                "type": "vad_state", "speaking": False,
                            })
                            await _finalize_speech()

                        elif event and event.kind == "speech_discard":
                            log.info("server_vad_speech_discard",
                                     session_id=session_id)
                            await _send_json(websocket, {
                                "type": "vad_state", "speaking": False,
                            })
                            # A parked smart-turn continuation holds REAL
                            # speech — a trailing sub-min-speech blip must
                            # not wipe it. Without this guard the discard
                            # path closed the STT session mid-utterance
                            # and the user's words flushed into the void
                            # (observed 2026-06-13: ".. that's all" hit
                            # moonshine's final flush DURING close and
                            # never dispatched). Finalize the parked
                            # utterance instead of discarding.
                            if (_smart_turn_veto_deadline is not None
                                    and (len(_speech_pcm_buffer) > 0
                                         or stt_session is not None
                                         or batch_stt is not None)):
                                log.info(
                                    "speech_discard_finalizing_parked_veto",
                                    buffer_bytes=len(_speech_pcm_buffer),
                                )
                                _smart_turn_veto_deadline = None
                                await _finalize_speech()
                                continue
                            # False alarm — the noise that confirmed a
                            # barge-in didn't survive the VAD's own
                            # min-speech check. Roll the interrupt back
                            # unconditionally (the old ``is_speaking``
                            # guard raced the turn's finally block and
                            # lost whenever the reply finished before the
                            # discard arrived) and replay any sentences
                            # the TTS consumer drained while the stale
                            # flag was up. Gated on bargein_pending so a
                            # client-initiated interrupt (explicit stop)
                            # is never rolled back by a VAD event.
                            if session.bargein_pending:
                                session.bargein_pending = False
                                if session.interrupted:
                                    session.interrupted = False
                                    log.info("server_vad_interrupt_cleared",
                                             session_id=session_id)
                                # Always spawn — with nothing drained the
                                # replay is a no-op that also discards
                                # any deferred heard-rewrite (the full
                                # text was delivered after all).
                                _replay_task = asyncio.create_task(
                                    _replay_undelivered_tts(
                                        websocket, session,
                                        websocket.app.state,
                                    )
                                )
                                _replay_task.add_done_callback(
                                    _silence_task_exception,
                                )
                            # Too short to be real speech — clean up
                            if stt_session:
                                try:
                                    await asyncio.wait_for(stt_session.close(), timeout=5.0)
                                except (TimeoutError, Exception) as exc:
                                    log.warning("stt_close_timeout", error=str(exc))
                                stt_session = None
                            batch_stt = None
                            _final_transcript_parts.clear()
                            _speech_pcm_buffer.clear()
                            _smart_turn_veto_deadline = None
                            # Barge-in candidate was a false alarm — the
                            # transient noise that tripped speech_start
                            # didn't sustain. Abandon it so TTS keeps
                            # playing uninterrupted.
                            if _barge_in_candidate_start is not None:
                                log.info("bargein_discarded_transient",
                                         elapsed_ms=int(
                                             (time.monotonic() - _barge_in_candidate_start) * 1000
                                         ))
                                _barge_in_candidate_start = None

                else:
                    # Client-side VAD / PTT: buffer WebM/Opus audio
                    if is_recording:
                        audio_buffer.extend(raw)

                continue

            # ----- Text frame: JSON control message -----
            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "config":
                    if "model" in data:
                        session.model = data["model"]
                    if "mode" in data:
                        session.mode = data["mode"]
                    if "voice" in data:
                        session.voice = data["voice"]
                        # In narrative mode, user manually changing the voice pill
                        # overrides the character preset for this call only.
                        # Clear character_voice so `character_voice or voice`
                        # resolves to the user's choice.
                        if session.mode == "narrative" and session.character_voice:
                            log.info("voice_override_character",
                                     old=session.character_voice,
                                     new=data["voice"])
                            session.character_voice = ""
                    if "speed" in data:
                        session.speed = max(0.5, min(2.0, float(data["speed"])))
                    if "system_prompt" in data:
                        session.system_prompt = str(data["system_prompt"])[:8192]
                    if "character_voice" in data:
                        session.character_voice = data["character_voice"]
                    if "speaker_override" in data:
                        # Manual-mode group voice: PIP tap on the client
                        # tells us which character should respond next.
                        # One-shot — the handler consumes and clears it.
                        session.speaker_override = str(data.get("speaker_override") or "")[:128]
                        log.info("voice_speaker_override_set",
                                 speaker=session.speaker_override)
                    if "messages" in data:
                        session.messages = data["messages"]
                    if "tools" in data:
                        session.tools = list(data["tools"]) if data["tools"] else []
                        log.info("voice_tools_config",
                                 session_id=session_id,
                                 tools=session.tools)
                    if "input_mode" in data:
                        session.staging = data["input_mode"] == "staging"
                    if "xr_surface" in data:
                        session.active_xr_surface = str(data.get("xr_surface") or "voice")[:80]
                        if "xr_panel_action" not in data:
                            session.active_xr_action = ""
                        log.info("voice_xr_surface_config",
                                 session_id=session_id,
                                 surface=session.active_xr_surface)
                    if "xr_panel_action" in data:
                        session.active_xr_action = str(data.get("xr_panel_action") or "")[:80]
                        log.info("voice_xr_panel_action_config",
                                 session_id=session_id,
                                 action=session.active_xr_action)
                    if "xr_user_context" in data:
                        raw_signals = data.get("xr_user_context")
                        if isinstance(raw_signals, list):
                            session.add_xr_user_signals(raw_signals)

                elif msg_type == "capabilities":
                    # Client reports which voice-pipeline primitives it can
                    # run locally (Wasm-loaded models). Filtered into
                    # ``session.client_caps`` for the pipeline_resolver to
                    # consult on every dispatch. Schema:
                    #   {"type":"capabilities","vad":[...],"stt":[...],
                    #    "tts":[...],"denoise":[...]}
                    # Pre-handshake or empty-array clients fall through to
                    # server-side dispatch, matching legacy behavior.
                    caps: dict[str, list[str]] = {}
                    for key in ("vad", "stt", "tts", "denoise"):
                        raw = data.get(key)
                        if isinstance(raw, list):
                            caps[key] = [str(x) for x in raw if isinstance(x, str)]
                    session.client_caps = caps
                    # Re-resolve targets now that caps are known.
                    _refresh_pipeline_targets(session)
                    # If the resolver picked client-side VAD, retire the
                    # server VadProcessor and switch the message handler
                    # to the legacy client-VAD code paths (the
                    # ``not use_server_vad`` branches that consume
                    # ``vad_speech_start`` / ``vad_speech_end`` /
                    # ``vad_discard`` from the client). The existing
                    # streaming audio path continues to populate
                    # ``audio_buffer`` between start/end markers — same
                    # protocol as the old amplitude-VAD path, just now
                    # driven by real Silero events from the browser.
                    vad_target = session.pipeline_targets.get("vad", "server")
                    if vad_target.startswith("client:") and use_server_vad:
                        if vad is not None:
                            vad.reset()
                            vad = None
                        use_server_vad = False
                        # Arm the liveness watchdog: from here the server is
                        # deaf unless the browser sends speech events. If it
                        # never does while clear audio flows, the frame loop
                        # re-creates server VAD.
                        _client_vad_owned_since = time.monotonic()
                        _client_vad_events = 0
                        _client_vad_voiced_frames = 0
                        log.info(
                            "voice_vad_source_switched",
                            session=session.session_id,
                            vad_source="client",
                            engine=vad_target.removeprefix("client:"),
                        )
                    log.info(
                        "voice_client_capabilities_registered",
                        session=session.session_id,
                        caps=caps,
                        pipeline_targets=session.pipeline_targets,
                    )

                elif msg_type == "stage_send":
                    # User pressed Send in staging mode — process the edited text
                    edited_text = str(data.get("text", "")).strip()
                    if edited_text:
                        raw_signals = data.get("xr_user_context")
                        if isinstance(raw_signals, list):
                            session.add_xr_user_signals(raw_signals)
                        if session.current_task and not session.current_task.done():
                            session.interrupted = True
                            session.current_task.cancel()
                            await asyncio.sleep(0.05)
                        session.current_task = asyncio.create_task(
                            _process_voice_turn_from_transcript(
                                edited_text, websocket, session, websocket.app.state,
                                from_stage_send=True,
                            )
                        )

                elif msg_type == "xr_user_signal":
                    signal = data.get("signal")
                    if isinstance(signal, dict):
                        session.add_xr_user_signals([signal])
                        log.info("voice_xr_user_signal",
                                 session_id=session_id,
                                 signal_type=str(signal.get("type") or "")[:80])

                elif msg_type == "ptt_active" and use_server_vad:
                    active = data.get("active", False)
                    ptt_gate_closed = not active
                    if active:
                        _ptt_grace_until = None
                    log.info("voice_ptt_gate",
                             active=active, session_id=session_id)
                    if not active and vad:
                        has_speech = vad.is_speaking or len(_speech_pcm_buffer) > 0
                        vad.reset()
                        await _send_json(websocket, {
                            "type": "vad_state", "speaking": False,
                        })
                        if has_speech:
                            # Button released after speech — finalize (STT → LLM)
                            await _finalize_speech()
                        else:
                            # Button released before any speech detected — return to listening
                            await _send_json(websocket, {"type": "listening"})
                        # Arm the post-release grace window: the press
                        # opened a conversation; speech in the next few
                        # seconds is a continuation, not room noise.
                        _grace_s = float(getattr(
                            settings, "companion_followup_window_s", 12.0,
                        ) or 0.0)
                        if _grace_s > 0:
                            _ptt_grace_until = time.monotonic() + _grace_s

                elif msg_type == "playback_state":
                    # Client reports actual SPEAKER playback boundaries.
                    # tts_ended_at is otherwise set at turn commit
                    # (generation end) — seconds ahead of the speakers
                    # for queued replies, which mis-anchored the post-
                    # TTS echo-suppression window: her audible tail
                    # tripped phantom speech_starts, and a user reply
                    # landing right as she finished got soft_reset
                    # mid-word. Re-anchor to playback end when the
                    # client tells us (older clients never send this —
                    # commit-time anchor remains their behavior).
                    _pb_active = bool(data.get("active"))
                    session.client_playback_active = _pb_active
                    if not _pb_active:
                        session.tts_ended_at = time.monotonic()
                    log.debug(
                        "voice_playback_state",
                        active=_pb_active, session_id=session_id,
                    )

                elif msg_type == "ping":
                    # Client heartbeat — keeps idle WS alive through proxies
                    # during PTT pauses. Reply with pong so a future client
                    # enhancement can detect a half-open WS by missed pongs.
                    await _send_json(websocket, {"type": "pong"})

                elif msg_type == "video_frame":
                    # Live-camera mode: the client pushes the latest webcam
                    # frame(s) as ``data:`` URLs. We keep only the freshest
                    # set (RAM-only, overwritten) so the next companion turn
                    # can SEE what the user is showing — a VL primary reads
                    # them directly, a text-only primary gets them captioned.
                    # Gated by ``companion_live_vision_enabled`` so the frame
                    # buffer stays empty (and the camera path inert) unless
                    # the deployment opted in.
                    if not bool(getattr(settings, "companion_live_vision_enabled", False)):
                        log.info("live_vision_frame_gated_off", session_id=session.session_id)
                        continue
                    _frames_in = data.get("frames")
                    if not isinstance(_frames_in, list):
                        _single = data.get("frame")
                        _frames_in = [_single] if isinstance(_single, str) else []
                    _frames = [
                        f for f in _frames_in
                        if isinstance(f, str) and f.startswith("data:") and len(f) < 4_000_000
                    ][:_LIVE_VISION_MAX_FRAMES]
                    if _frames:
                        session.latest_frames = _frames
                        session.latest_frame_ts = time.monotonic()
                    log.info(
                        "live_vision_frame_rx",
                        session_id=session.session_id,
                        gate_on=bool(getattr(settings, "companion_live_vision_enabled", False)),
                        n_in=len(_frames_in) if isinstance(_frames_in, list) else 0,
                        n_kept=len(_frames),
                    )

                elif msg_type == "interrupt":
                    session.interrupted = True
                    session.played_sentences = int(data.get("played_sentences", 0))
                    # Close any active streaming STT
                    if stt_session:
                        try:
                            await asyncio.wait_for(stt_session.close(), timeout=5.0)
                        except (TimeoutError, Exception) as exc:
                            log.warning("stt_close_timeout", error=str(exc))
                        stt_session = None
                    batch_stt = None
                    _final_transcript_parts.clear()
                    _speech_pcm_buffer.clear()
                    if vad:
                        vad.reset()

                    # Cancelling unwinds the turn task past its own
                    # "interrupted"/"listening" emit, so ack here regardless.
                    # The turn still persists what the user heard — that's
                    # done from a finally block that survives cancellation.
                    if session.current_task and not session.current_task.done():
                        session.current_task.cancel()
                    await _send_json(websocket, {"type": "interrupted"})
                    await _send_json(websocket, {"type": "listening"})

                # --- Server-VAD path: re-arm warmup on every capture cycle ---
                # The becca-ptt client sends start_recording on each
                # triggerWakeCapture (initial + every re-arm via
                # _scheduleAlwaysListeningRearm). Reset the warmup
                # deadline so the first 500ms of each fresh capture
                # cycle is protected from VAD false-positives. No-op
                # for non-becca paths (no warmup is configured).
                elif msg_type == "start_recording" and use_server_vad:
                    # Capture provenance: the becca widget tags each
                    # capture cycle with how it started. "ptt" (button
                    # press) and "wake" (wake word) are deliberate
                    # addressing — the address gate skips the ambient
                    # veto for them. "followup" / "auto" (open-mic
                    # re-arms) keep ambient gating. Untagged → "auto"
                    # so older clients keep ambient semantics.
                    _src = str(data.get("source", "") or "").lower()
                    session.capture_source = (
                        _src if _src in ("ptt", "wake", "followup") else "auto"
                    )
                    if getattr(session, "persona_id", "") == "becca":
                        _wms = int(getattr(settings, "companion_always_listening_warmup_ms", 500) or 500)
                        if _wms > 0:
                            _vad_warmup_until = time.monotonic() + (_wms / 1000.0)
                            log.debug(
                                "voice_vad_warmup_armed",
                                session_id=session_id, warmup_ms=_wms,
                            )

                # --- Client-side VAD / PTT messages (legacy) ---
                elif msg_type == "start_recording" and not use_server_vad:
                    audio_buffer.clear()
                    is_recording = True
                    session.interrupted = False

                elif msg_type == "stop_recording" and not use_server_vad:
                    is_recording = False
                    if len(audio_buffer) < 100:
                        await _send_json(websocket, {"type": "listening"})
                        continue

                    audio_data = bytes(audio_buffer)
                    audio_buffer.clear()

                    # Client-side PTT: a press/release cycle is always a
                    # deliberate utterance.
                    session.last_utterance_explicit = True

                    if session.current_task and not session.current_task.done():
                        session.interrupted = True
                        session.current_task.cancel()
                        await asyncio.sleep(0.05)

                    session.current_task = asyncio.create_task(
                        _process_voice_turn(
                            audio_data, websocket, session,
                            websocket.app.state,
                        )
                    )

                elif msg_type == "vad_speech_end" and not use_server_vad:
                    is_recording = False
                    if len(audio_buffer) < 100:
                        await _send_json(websocket, {"type": "listening"})
                        continue

                    audio_data = bytes(audio_buffer)
                    audio_buffer.clear()

                    # Client auto-VAD capture — not a deliberate press.
                    session.last_utterance_explicit = False

                    if session.current_task and not session.current_task.done():
                        session.interrupted = True
                        session.current_task.cancel()
                        await asyncio.sleep(0.05)

                    session.current_task = asyncio.create_task(
                        _process_voice_turn(
                            audio_data, websocket, session,
                            websocket.app.state,
                        )
                    )

                elif msg_type == "vad_discard" and not use_server_vad:
                    audio_buffer.clear()
                    is_recording = False

                elif msg_type == "vad_speech_start" and not use_server_vad:
                    is_recording = True
                    # A real client speech event proves client VAD works —
                    # disarm the liveness watchdog.
                    _client_vad_events += 1
                    if session.is_speaking:
                        session.interrupted = True
                        if session.current_task and not session.current_task.done():
                            session.current_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("voice_websocket_error", error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception as send_exc:
            log.debug("voice_websocket_error_send_failed", error=str(send_exc))
    finally:
        log.info("voice_cleanup_start", session_id=session_id)
        # Stop the smart-turn veto watchdog — its WS handle is dying.
        if _veto_watchdog_task is not None and not _veto_watchdog_task.done():
            _veto_watchdog_task.cancel()
        # Free this session's ack shuffle-bag.
        try:
            from augmentum.voice import ack_clips
            ack_clips.reset(session_id)
        except Exception:  # noqa: BLE001
            log.debug("voice_ack_reset_failed", exc_info=True)
        # Conversation-scoped CSM residency: the voice WS closed, so release
        # the model's VRAM + clear this session's cross-speaker context.
        # Only in "session" mode ("always" keeps it resident; "timer" never
        # set csm_provider). Best-effort — cleanup must never raise.
        _csm_prov = getattr(session, "csm_provider", None)
        if _csm_prov and getattr(settings, "companion_csm_residency", "session") == "session":
            try:
                from augmentum.proxy.audio_routes import csm_unload
                await csm_unload(
                    provider=_csm_prov,
                    session_id=session.session_id,
                    user_id=session.user_id,
                )
                log.info("voice_cleanup_csm_unloaded", session_id=session_id)
            except Exception:
                log.debug("voice_cleanup_csm_unload_failed", exc_info=True)
        # Clean up streaming STT
        if stt_session:
            try:
                await asyncio.wait_for(stt_session.close(), timeout=5.0)
                log.info("voice_cleanup_stt_closed", session_id=session_id)
            except (TimeoutError, Exception) as exc:
                log.warning("voice_cleanup_stt_close_failed",
                            session_id=session_id, error=str(exc))
            stt_session = None
        # Clean up batch STT fallback
        if batch_stt:
            batch_stt = None
            log.info("voice_cleanup_batch_stt_cleared", session_id=session_id)
        # Reset VAD state
        if vad:
            vad.reset()
            log.info("voice_cleanup_vad_reset", session_id=session_id)
        if speech_enhancer:
            speech_enhancer.reset()
        # Clear audio buffers to free memory
        audio_buffer.clear()
        _speech_pcm_buffer.clear()
        _final_transcript_parts.clear()
        # Cancel any in-flight LLM/TTS task
        if session.current_task and not session.current_task.done():
            session.current_task.cancel()
            log.info("voice_cleanup_task_cancelled", session_id=session_id)
        # Clear note-capture mode + any pending surface events for this
        # session. Without this, a session that disconnected mid-capture
        # would still treat the user's NEXT voice utterance as a
        # capture append after they reconnect — confusing.
        try:
            refs = get_intent_referents(
                websocket.app.state, session.user_id, session_id,
            )
            if refs.note_capture_mode:
                refs.note_capture_mode = False
                refs.note_capture_deadline = 0.0
                log.info(
                    "voice_cleanup_capture_mode_cleared",
                    session_id=session_id,
                )
            if refs.pending_surface_events:
                refs.pending_surface_events.clear()
        except Exception as exc:
            log.debug(
                "voice_cleanup_intent_refs_failed",
                session_id=session_id, error=str(exc),
            )
        # Close the fanout session if one was opened. Without this the
        # VoiceFanout keeps subscriber queues alive for the session_id,
        # blocking future re-opens with the same id and pinning memory.
        # The fanout module's own docstring documents this as the
        # required disconnect contract — we were silently breaking it.
        if fanout_session_id and voice_fanout is not None:
            try:
                await voice_fanout.close_session(fanout_session_id)
                log.info(
                    "voice_cleanup_fanout_closed",
                    session_id=session_id,
                    fanout_session_id=fanout_session_id,
                )
            except Exception as exc:
                log.warning(
                    "voice_cleanup_fanout_close_failed",
                    session_id=session_id,
                    fanout_session_id=fanout_session_id,
                    error=str(exc),
                )
        log.info("voice_disconnected", session_id=session_id)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 mono bytes in a minimal WAV header."""
    import struct
    data_size = len(pcm_bytes)
    # WAV header: 44 bytes
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,      # ChunkSize
        b"WAVE",
        b"fmt ",
        16,                  # Subchunk1Size (PCM)
        1,                   # AudioFormat (PCM = 1)
        1,                   # NumChannels (mono)
        sample_rate,         # SampleRate
        sample_rate * 2,     # ByteRate (SampleRate * NumChannels * BitsPerSample/8)
        2,                   # BlockAlign (NumChannels * BitsPerSample/8)
        16,                  # BitsPerSample
        b"data",
        data_size,           # Subchunk2Size
    )
    return header + pcm_bytes
