"""Voice chat pipeline — STT → LLM → sentence-buffered TTS.

Orchestrates the full voice turn: receives audio, transcribes via STT,
routes through the existing chat handler pipeline, buffers LLM tokens
into sentences, and streams TTS audio back per sentence.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.proxy.audio_routes import (
    _FABRIC_PROVIDER_PREFIX,
    _audio_client,
    _build_headers,
    _build_tts_stream,
    _get_default_provider,
    _get_provider_by_id,
    _is_deepgram,
    _moonshine_batch_transcribe,
)
from augmentum.utils.http_client import normalize_base_url
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sentence Buffer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Backchannel / Junk Transcript Detection
# ---------------------------------------------------------------------------

# Short utterances that are conversational acknowledgements, not real turns.
# Mirrors the client-side BACKCHANNEL_RE but is authoritative — the server
# must filter these before sending to the LLM.
# Backchannel / noise filtering.
# Primary STT is Moonshine (not Whisper) — Moonshine doesn't hallucinate
# phrases like "thanks for watching". Only filter truly empty transcripts.
# Let the LLM handle short real utterances like "yeah", "ok", "hey".
_BACKCHANNEL_RE = None  # Disabled — let everything through to the LLM
_BACKCHANNEL_MAX_WORDS = 0

_STT_NOISE_PHRASES = frozenset({
    "...",
    ".",
    "",
})


def is_backchannel(transcript: str) -> bool:
    """Return True if the transcript is a backchannel or STT noise artifact."""
    text = transcript.strip()
    if not text:
        return True
    # Known STT hallucinations (Whisper in particular)
    if text.lower() in _STT_NOISE_PHRASES:
        return True
    # Short backchannel utterances
    # Regex filter disabled — let LLM handle short utterances naturally
    if _BACKCHANNEL_RE and len(text.split()) <= _BACKCHANNEL_MAX_WORDS:
        if _BACKCHANNEL_RE.match(text):
            return True
    return False


# Patterns that look like sentence endings but aren't
_FALSE_ENDINGS = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|approx|dept|est|vol)"
    r"\.\s*$",
    re.IGNORECASE,
)


# TTS providers that synthesize fast enough locally that clause-level
# chunking buys no perceptible TTFA — but costs a prosody reset and an
# audible seam at every split. Pocket handles up to ~8KB natively
# (see pocket_tts.py); whole sentences are the right unit for it.
SMOOTH_CHUNKING_PROVIDERS: frozenset[str] = frozenset({
    "pockettts-builtin",
})


def effective_chunking_mode(configured: str, provider_id: str) -> str:
    """Resolve the chunking mode for a turn, provider-aware.

    An explicit non-default setting (clause / paragraph / full / smooth)
    is always honored. The DEFAULT ("sentence") upgrades to "smooth"
    for fast local providers where the clause tier is pure harm.
    """
    mode = (configured or "sentence").strip() or "sentence"
    if mode == "sentence" and provider_id in SMOOTH_CHUNKING_PROVIDERS:
        return "smooth"
    return mode


class SentenceBuffer:
    """Accumulate streaming LLM tokens, emit at clause/sentence boundaries.

    Uses a two-tier chunking strategy for minimal time-to-first-audio:

    **Tier 1 — Clause-level (first 2 chunks):**
    Emit at any natural pause: commas, semicolons, colons, em-dashes,
    or after ~30 chars at a word boundary.  Gets audio playing within
    ~1-2 seconds of LLM output starting.

    **Tier 2 — Sentence-level (subsequent chunks):**
    Once the user is already hearing audio, switch to sentence boundaries
    for better TTS prosody.  The perceptual latency is already hidden
    behind the first chunk's audio playback.

    Tracks fenced code blocks (```) so TTS detection is suppressed
    while inside a code block.  When the block closes, its content is
    silently discarded rather than sent to TTS.

    **Modes:**
    - ``sentence`` (default): Two-tier — clause-level for the first 2 chunks,
      then one sentence per chunk for better prosody.
    - ``clause``: Always use clause-level breaks for lowest latency.
    - ``paragraph``: First chunk clause-broken for fast TTFA, then 3 sentences
      per chunk (or up to a ``\\n\\n``, whichever fires first). Best prose
      delivery; the word-boundary safety cap still applies on runaways.
    - ``full``: No intermediate emission — only ``flush()`` returns text.
    """

    _MODE_PRESETS: dict[str, tuple[list[int], int, int]] = {
        #              (schedule,                 clause_tier, sentences_per_chunk)
        "clause":     ([20, 30, 40, 50, 60],     999,         1),
        "sentence":   ([30, 50, 100, 150, 200],  2,           1),
        # smooth: whole sentences from chunk 0, generous runaway caps.
        # For FAST LOCAL providers (Pocket TTS) where clause splitting
        # is pure cost: synthesis is near-instant so the clause tier's
        # TTFA win is negligible, while every split resets prosody and
        # adds an audible seam — "Sure," [pause] "let me check"
        # (2026-06-11). The word-boundary fallback only fires on
        # genuinely unpunctuated runs.
        "smooth":     ([350, 400, 450, 500, 550], 0,          1),
        "paragraph":  ([50, 80, 200, 350, 450],  1,           3),
        "full":       ([],                        0,           0),
    }

    def __init__(self, min_chars: int = 10, mode: str = "sentence"):
        self.buffer = ""
        self.min_chars = min_chars
        self.mode = mode if mode in self._MODE_PRESETS else "sentence"
        self._in_code_block = False
        self._code_fence_count = 0
        self._chunk_index = 0

        schedule, clause_tier, sentences_per_chunk = self._MODE_PRESETS[self.mode]
        self._chunk_schedule = list(schedule)
        self._clause_tier_chunks = clause_tier
        self._sentences_per_chunk = sentences_per_chunk

    def set_mode(self, mode: str) -> None:
        """Switch preset BEFORE streaming begins.

        Used by the provider-aware upgrade: the TTS provider often
        resolves a few lines after the buffer is constructed, so the
        mode decision arrives late. No-op once tokens have flowed —
        changing the schedule mid-stream would mis-index the chunk
        counter.
        """
        if self._chunk_index or self.buffer or mode not in self._MODE_PRESETS:
            return
        self.mode = mode
        schedule, clause_tier, sentences_per_chunk = self._MODE_PRESETS[mode]
        self._chunk_schedule = list(schedule)
        self._clause_tier_chunks = clause_tier
        self._sentences_per_chunk = sentences_per_chunk

    def add_token(self, token: str) -> str | None:
        """Add a token. Returns a chunk if a boundary was reached."""
        self.buffer += token

        # Detect code fence transitions
        fence_count = self.buffer.count("```")
        if fence_count > self._code_fence_count:
            self._code_fence_count = fence_count
            if fence_count % 2 == 1:
                self._in_code_block = True
            else:
                self._in_code_block = False
                self.buffer = re.sub(r"```[\s\S]*?```", "", self.buffer)
                self._code_fence_count = self.buffer.count("```")

        if self._in_code_block:
            return None

        return self._try_extract()

    def flush(self) -> str | None:
        """Return any remaining buffered text."""
        if self._in_code_block:
            idx = self.buffer.rfind("```")
            if idx >= 0:
                self.buffer = self.buffer[:idx]
            self._in_code_block = False

        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            self._chunk_index += 1
            return sentence
        return None

    def _try_extract(self) -> str | None:
        # Full mode: never emit intermediate chunks — only flush() returns text
        if not self._chunk_schedule:
            return None

        schedule_threshold = self._chunk_schedule[
            min(self._chunk_index, len(self._chunk_schedule) - 1)
        ]
        use_clause_breaks = self._chunk_index < self._clause_tier_chunks
        # Target sentences per emission: paragraph mode wants more after the
        # first-tier chunks (which always stay at 1 for fast TTFA).
        target_sentences = (
            1 if self._chunk_index < self._clause_tier_chunks
            else max(1, self._sentences_per_chunk)
        )

        # --- Priority 0 (paragraph mode only): explicit \n\n break wins ---
        if self._sentences_per_chunk > 1:
            para_match = re.search(r"\n[ \t]*\n", self.buffer)
            if para_match:
                end = para_match.end()
                candidate = self.buffer[:end].strip()
                if len(candidate) >= self.min_chars:
                    self.buffer = self.buffer[end:]
                    self._chunk_index += 1
                    return candidate

        # --- Priority 1: Sentence-ending punctuation (Nth real one) ---
        if target_sentences >= 1:
            real_count = 0
            chosen_end = -1
            for m in re.finditer(r'[.!?]["\')]*(?:\s|$)', self.buffer):
                prefix = self.buffer[:m.end()].rstrip()
                if _FALSE_ENDINGS.search(prefix):
                    continue
                real_count += 1
                if real_count >= target_sentences:
                    chosen_end = m.end()
                    break
            if chosen_end > 0:
                candidate = self.buffer[:chosen_end].strip()
                if len(candidate) >= self.min_chars:
                    self.buffer = self.buffer[chosen_end:]
                    self._chunk_index += 1
                    return candidate

        # --- Priority 2: Clause-level breaks (Tier 1 only — first N chunks) ---
        if use_clause_breaks and target_sentences == 1 and len(self.buffer) >= self.min_chars:
            # Break at comma, semicolon, colon, em-dash, or parenthetical close
            clause_match = re.search(r'[,;:\u2014)\]]+\s', self.buffer)
            if clause_match and clause_match.end() >= self.min_chars:
                end = clause_match.end()
                candidate = self.buffer[:end].strip()
                if len(candidate) >= self.min_chars:
                    self.buffer = self.buffer[end:]
                    self._chunk_index += 1
                    return candidate

        # --- Priority 3: Word-boundary fallback at progressive threshold ---
        if len(self.buffer) >= schedule_threshold:
            last_space = self.buffer.rfind(" ", 0, len(self.buffer))
            if last_space > self.min_chars:
                chunk = self.buffer[:last_space].strip()
                self.buffer = self.buffer[last_space:]
                self._chunk_index += 1
                return chunk

        return None


# ---------------------------------------------------------------------------
# Voice Session State
# ---------------------------------------------------------------------------


@dataclass
class VoiceSession:
    """Per-connection state for a voice chat WebSocket."""

    session_id: str
    model: str = ""
    mode: str = "passthrough"
    user_id: str = ""  # auth user owning this voice connection (auth set by AuthMiddleware)
    persona_id: str = ""  # "becca" routes voice turns through BeccaVoice; "" → legacy
    voice: str = ""
    speed: float = 1.0
    character_voice: str = ""  # Narrative: voice for current character
    # Group narrative manual mode: name of the character the user has
    # picked to respond next via the avatar PIP tap. One-shot — consumed
    # on the next handler invocation and cleared. Empty = no override.
    speaker_override: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    staging: bool = False       # Staging mode: transcribe only, don't auto-send to LLM
    interrupted: bool = False
    is_speaking: bool = False
    tts_ended_at: float = 0.0  # monotonic timestamp when TTS finished (for echo cooldown)
    # True while the CLIENT reports speaker playback in progress
    # (playback_state messages). tts_ended_at re-anchors to playback
    # end when these arrive — generation end runs seconds ahead of the
    # speakers for queued replies. Older clients never send it.
    client_playback_active: bool = False
    # Per-turn latency waterfall — monotonic stamps keyed by stage
    # (speech_end / stt_done / dispatch / first_audio). Reset at each
    # finalize; one structured voice_turn_waterfall line emitted when
    # the first audio byte goes out, so every turn's budget is visible
    # in one place instead of reconstructed across five log lines.
    turn_timing: dict = field(default_factory=dict)
    current_task: asyncio.Task | None = None
    played_sentences: int = 0  # Sentences user heard before interrupting
    tools: list[str] = field(default_factory=list)  # Passthrough tool names

    # Cross-speaker CSM context: the user's last spoken utterance (raw PCM16
    # mono from server VAD) + its sample rate, stashed by the voice WS for
    # the companion turn to feed CSM so her prosody reacts to how they
    # sounded. RAM-only, overwritten each turn; never persisted.
    last_user_audio: bytes = b""
    last_user_audio_sr: int = 16000
    # Her current mood as a CSM emotion tag for this turn (cross-modal voice).
    # Set at turn start from her affect when companion_csm_emotion_tag is on;
    # consumed in stream_tts_sentence for the fine-tuned CSM voice.
    csm_emotion: str = ""
    # Raw affect tag for this turn, for OpenAI-omni style voices (Higgs etc.).
    # Set when companion_voice_emotion_tag is on; tts.py maps it to a natural
    # style cue per provider, keeping the affect->voice seam model-agnostic.
    voice_affect: str = ""
    # Conversation-scoped CSM residency: the resolved CSM provider for this
    # session (set at WS open when her voice is CSM), so the WS-close path
    # can unload its VRAM + clear its context. None = not a CSM session.
    csm_provider: dict | None = None

    # --- Live vision (camera frames) ---
    # Most-recent camera frame(s) the client pushes over the WS as
    # ``video_frame`` messages (``data:`` URLs, capped). RAM-only, consumed
    # at turn start (attached to the companion turn's images) and cleared so
    # each turn sees only the freshest frame. Empty unless the client has a
    # live camera open. ``latest_frame_ts`` (monotonic) gates staleness so a
    # turn never reasons about a frame from minutes ago.
    latest_frames: list[str] = field(default_factory=list)
    latest_frame_ts: float = 0.0

    # --- Barge-in recovery (false-positive interrupts) ---
    # tts_started: at least one sentence has been handed to TTS this turn.
    # Barge-in candidates only arm once this is True — a turn in prompt
    # eval has produced no audio, so there is nothing to barge in on.
    tts_started: bool = False
    # bargein_pending: a VAD barge-in was confirmed and has not yet been
    # vindicated by a real transcript. If the interrupting "speech" is
    # then discarded (or its STT comes back empty) the interrupt was a
    # false alarm — the recovery path clears ``interrupted`` and replays
    # ``undelivered_tts``.
    bargein_pending: bool = False
    # Sentences the TTS consumer drained while ``interrupted`` was set.
    # Kept so a false barge-in can replay them — otherwise the reply is
    # committed to history but the user never hears it.
    undelivered_tts: list[str] = field(default_factory=list)
    # Snapshot of the turn's resolved TTS parameters (voice / speed /
    # provider / format) so the replay path can stream without re-running
    # the turn's resolution logic.
    tts_params: dict = field(default_factory=dict)
    # Audio-only honest-history deferral: when a turn commits while a
    # provisional barge-in is outstanding, the full text goes into
    # history (a false barge-in will replay the tail, making it
    # accurate) and this holds the heard-only portion. Vindication (a
    # real transcript) rewrites the last assistant message to it; a
    # completed replay discards it. None = nothing pending.
    pending_heard_rewrite: str | None = None

    # --- Capture provenance (explicit vs ambient addressing) ---
    # How the current capture cycle started, tagged by the client on each
    # ``start_recording``: "ptt" (button press), "wake" (wake word),
    # "followup" (auto re-arm inside a follow-up window), "auto"
    # (always-listening re-arm). Empty = untagged legacy client.
    capture_source: str = ""
    # True when the most recently finalized utterance came from a
    # deliberate capture (ptt / wake). The address gate skips the ambient
    # addressed-classifier veto for explicit captures — the user pressed
    # a button or said the wake word; addressing is a given.
    last_utterance_explicit: bool = False

    # XR surface context: which VR panel the user is focused on, what action
    # they last selected, and a rolling buffer of nonverbal signals (gaze,
    # gesture, etc.) drained into the next system-prompt addendum. All three
    # are written by voice_routes.py's surface_config / xr_user_signal
    # handlers; drain_xr_user_signals() is called once per turn.
    active_xr_surface: str = ""
    active_xr_action: str = ""
    xr_user_signals: list[dict] = field(default_factory=list)

    # Client-side voice pipeline capabilities, populated from the first
    # ``{"type": "capabilities", ...}`` frame the browser sends after the
    # WS handshake. Empty dict (the default) means "client reported no
    # local primitives" — the resolver falls back to server-side dispatch
    # for every component, matching pre-handshake behavior byte-for-byte.
    # Shape:
    #   {"vad": ["silero-wasm"], "stt": ["moonshine-wasm"],
    #    "tts": ["pocket-onnx-wasm"], "denoise": ["rnnoise-wasm"]}
    # Unknown keys are ignored; missing keys read as empty lists.
    client_caps: dict[str, list[str]] = field(default_factory=dict)

    # Resolved dispatch target per pipeline component, computed by
    # ``pipeline_resolver.resolve(...)`` at session start and re-computed
    # when ``client_caps`` changes. Shape:
    #   {"vad": "server" | "client:silero-wasm" | "fabric:tower:x",
    #    "stt": ..., "tts": ..., "denoise": ...}
    # Today this is observability-only — actual dispatch still flows
    # through the legacy paths. The handlers that own each component
    # consult ``pipeline_targets`` to decide whether to bypass server-
    # side logic, and that swap happens incrementally as client engines
    # land (Silero VAD first; see voice/vad_client_bridge.py).
    pipeline_targets: dict[str, str] = field(default_factory=dict)

    # Maximum messages to send to the LLM (sliding window).
    # 60 messages = 30 turns of conversation — enough context without
    # overflowing the model's context window on long calls.
    MAX_HISTORY: int = 60
    # Cap the XR signal buffer so a long call with chatty sensors can't
    # grow unbounded. _xr_user_signal_addendum already only renders the
    # last 4 entries, so 16 gives some headroom for back-to-back drains.
    MAX_XR_SIGNALS: int = 16

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str, thinking: str = "") -> None:
        # Carry reasoning_content through the in-call history so DeepSeek
        # and other reasoning-required providers see it on the next turn.
        # Empty thinking is omitted so the dict round-trips cleanly through
        # any consumer that only expects {role, content}.
        msg: dict[str, str] = {"role": "assistant", "content": text}
        if thinking:
            msg["reasoning_content"] = thinking
        self.messages.append(msg)

    def get_recent_messages(self) -> list[dict[str, str]]:
        """Return the most recent messages within the sliding window."""
        if len(self.messages) <= self.MAX_HISTORY:
            return self.messages
        return self.messages[-self.MAX_HISTORY:]

    def strike_last_exchange(self) -> int:
        """Remove the trailing user→assistant exchange from in-call history.

        The server side of ``conversation.strike`` ("scratch that /
        disregard that last recording"). When mangled STT got a reply
        and is now poisoning context, this pops the trailing assistant
        message (if it's the tail) and the user message that prompted
        it, so the next turn reasons from clean history. Returns how many
        messages were removed (0-2).

        The strike utterance itself never reaches history — the dispatch
        short-circuits before ``add_user_message`` — so this only touches
        the prior exchange. Trailing-assistant-first ordering means a
        bare user turn with no reply yet (she's still thinking) strikes
        just that user message, which is the right thing.
        """
        removed = 0
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
            removed += 1
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
            removed += 1
        return removed

    def add_xr_user_signals(self, signals: list[dict]) -> None:
        """Append XR nonverbal signals, dropping non-dict entries and
        trimming the buffer to MAX_XR_SIGNALS most-recent."""
        for sig in signals:
            if isinstance(sig, dict):
                self.xr_user_signals.append(sig)
        if len(self.xr_user_signals) > self.MAX_XR_SIGNALS:
            self.xr_user_signals = self.xr_user_signals[-self.MAX_XR_SIGNALS:]

    def drain_xr_user_signals(self) -> list[dict]:
        """Return and clear the pending XR signal buffer. Called once per
        LLM turn so each signal is folded into a single system addendum."""
        if not self.xr_user_signals:
            return []
        signals = self.xr_user_signals
        self.xr_user_signals = []
        return signals


# ---------------------------------------------------------------------------
# STT Helper
# ---------------------------------------------------------------------------

_AUDIO_CONTENT_TYPES = {
    ".webm": "audio/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".opus": "audio/opus",
}


def _content_type_for(filename: str) -> str:
    """Infer audio content type from filename extension."""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return _AUDIO_CONTENT_TYPES.get(ext, "audio/webm")


def _transcode_to_wav(audio_bytes: bytes) -> tuple[bytes, str, str]:
    """Transcode audio to WAV via ffmpeg for maximum STT compatibility.

    Returns (wav_bytes, filename, content_type).
    Falls back to original bytes if ffmpeg is unavailable.
    """
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("ffmpeg"):
        log.debug("ffmpeg_not_found_skipping_transcode")
        return audio_bytes, "recording.webm", "audio/webm"

    import os
    inp_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name

        out_path = inp_path.rsplit(".", 1)[0] + ".wav"

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", inp_path, "-ar", "16000", "-ac", "1",
             "-f", "wav", out_path],
            capture_output=True, timeout=10,
        )

        os.unlink(inp_path)
        inp_path = None

        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                wav_bytes = f.read()
            os.unlink(out_path)
            log.debug("audio_transcoded", from_size=len(audio_bytes), to_size=len(wav_bytes))
            return wav_bytes, "recording.wav", "audio/wav"

        if os.path.exists(out_path):
            os.unlink(out_path)

        log.debug("ffmpeg_transcode_failed", stderr=result.stderr[:200] if result.stderr else "")
        return audio_bytes, "recording.webm", "audio/webm"
    except Exception:
        log.debug("transcode_error", exc_info=True)
        for p in (inp_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        return audio_bytes, "recording.webm", "audio/webm"


async def transcribe_audio(
    audio_bytes: bytes,
    conn,
    filename: str = "recording.webm",
    language: str = "",
    *,
    user_id: str = "",
) -> str:
    """Send audio to the configured STT provider and return transcript text.

    ``user_id`` is required when the resolved STT provider is fabric-
    routed (the receiving peer's middleware verifies the signed
    envelope against it). Local providers ignore it.
    """
    provider = await _get_default_provider(conn, "stt")
    if not provider:
        raise RuntimeError("No STT provider configured")

    raw_url = provider.get("base_url", "")
    if not raw_url or raw_url in ("builtin", "built-in"):
        # Use Moonshine batch transcription (local, in-process)
        return await _moonshine_batch_transcribe(audio_bytes, filename)

    # Transcode webm→wav for broader STT compatibility
    # Run in a thread to avoid blocking the event loop (ffmpeg subprocess)
    if filename.endswith(".webm"):
        import asyncio
        audio_bytes, filename, _ = await asyncio.to_thread(
            _transcode_to_wav, audio_bytes,
        )

    base_url = normalize_base_url(provider["base_url"])
    headers = _build_headers(provider["api_key"], base_url=base_url)
    model = provider["default_model"]
    content_type = _content_type_for(filename)
    is_fabric = provider.get("id", "").startswith(_FABRIC_PROVIDER_PREFIX)

    async with _audio_client(base_url) as client:
        if _is_deepgram(base_url):
            # Deepgram: POST /v1/listen with raw audio body
            params: dict[str, str] = {}
            if model:
                params["model"] = model
            if language:
                params["language"] = language
            dg_headers = {**headers, "Content-Type": content_type}
            resp = await client.post(
                f"{base_url}/v1/listen",
                content=audio_bytes,
                params=params,
                headers=dg_headers,
            )
            resp.raise_for_status()
            dg_data = resp.json()
            channels = dg_data.get("results", {}).get("channels", [])
            if channels:
                alts = channels[0].get("alternatives", [])
                if alts:
                    return alts[0].get("transcript", "").strip()
            return ""

        # OpenAI-compatible: multipart form
        url = f"{base_url}/v1/audio/transcriptions"

        if is_fabric:
            # Delegate cross-peer dispatch to the shared audio_client
            # so signing + multipart construction stays in one place.
            from augmentum.proxy.audio_routes import _fabric_coordinator
            coord = _fabric_coordinator
            identity = getattr(coord, "_identity", None) if coord is not None else None
            if identity is None or not user_id:
                log.warning(
                    "fabric_stt_call_missing_credentials",
                    provider_id=provider.get("id", ""),
                    has_coord=coord is not None,
                    has_user=bool(user_id),
                )
                raise RuntimeError(
                    "Fabric STT requires an active coordinator and a user_id"
                )
            # Build a factory that just yields the existing pipeline-bound
            # client so we share its connect pool + timeouts instead of
            # spinning up a fresh one inside the helper.
            import contextlib as _contextlib
            @_contextlib.asynccontextmanager
            async def _reuse_client(_base):  # type: ignore[no-untyped-def]
                yield client
            from augmentum.fabric.audio_client import stt_transcribe_via_peer
            return await stt_transcribe_via_peer(
                http_client_factory=_reuse_client,
                identity=identity, user_id=user_id,
                peer_base_url=base_url,
                audio_bytes=audio_bytes, filename=filename,
                content_type=content_type,
                model=model, language=language,
                response_format="json",
            )

        files_data = {
            "file": (filename, audio_bytes, content_type),
        }
        form_data: dict[str, str] = {}
        if model:
            form_data["model"] = model
        if language:
            form_data["language"] = language

        resp = await client.post(
            url, files=files_data, data=form_data, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "").strip()


# ---------------------------------------------------------------------------
# TTS Streaming Helper
# ---------------------------------------------------------------------------


async def warmup_tts(conn, voice: str = "", *, user_id: str = "") -> None:
    """Pre-warm the TTS pipeline: resolve provider, establish HTTP connection,
    and send a tiny silent request so the first real sentence doesn't pay
    cold-start costs.  Fire-and-forget — errors are silently logged.

    ``user_id`` is required when the resolved provider is fabric-routed
    (the peer's middleware authenticates against it). Local providers
    ignore it.
    """
    try:
        target_provider_id = ""
        tts_voice = voice
        if voice and "::" in voice:
            target_provider_id, tts_voice = voice.split("::", 1)

        if target_provider_id:
            provider = await _get_provider_by_id(conn, target_provider_id)
            if not provider:
                provider = await _get_default_provider(conn, "tts")
        else:
            provider = await _get_default_provider(conn, "tts")

        if not provider:
            return

        base_url = normalize_base_url(provider["base_url"])
        tts_voice = tts_voice or provider["default_voice"]
        tts_model = provider["default_model"]

        # Send a single period — minimal text that forces model load + HTTP
        # connection pool establishment without producing audible output.
        stream = _build_tts_stream(
            base_url, tts_model, tts_voice, ".",
            settings.voice_tts_format, 1.0, provider["api_key"],
            pre_cleaned=True,
            provider_id=provider.get("id", ""),
            user_id=user_id,
        )
        async for _ in stream:
            pass  # Discard audio — we just want the side effects

        log.info("tts_warmup_complete", provider=provider.get("id", ""),
                 voice=tts_voice)
    except Exception as exc:
        log.debug("tts_warmup_failed", error=str(exc))


# Re-export from tts module for backward compatibility
# Re-export from text_cleaning module — single source of truth.
# The canonical implementation lives in augmentum/voice/text_cleaning.py.
