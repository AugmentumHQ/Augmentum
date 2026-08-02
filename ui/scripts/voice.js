/* ==========================================================================
   Augmentum — Voice Chat Module
   WebSocket voice chat: STT → LLM → sentence-buffered TTS
   Supports push-to-talk and auto-detect (VAD) modes.
   ========================================================================== */

import { app, showToast, escapeHtml, safeParseJSON } from './app.js';
import { chat } from './chat.js';
import { activateChatModelByName, getSettings, save as saveSettings, syncVoicePrefsToBackend } from './settings.js';
import * as avatarModule from './avatar.js';
import { PoseTriggerEngine } from './avatar-pose-trigger.js';
import * as avatarXR from './avatar-xr.js';
import { getModels, getVoices, onChange as onCacheChange } from './model-cache.js';
import { voiceBadgeRich, peerSourceCount } from './voice-display.js';
import { AudioBus } from './audio-bus.js';
import { getWsTicket } from './auth.js';
import { ViewStack } from './view-stack.js';
import { acquireMic, streamMicLabel } from './voice/mic-device.js';
import { createUtteranceRecorder } from './voice/batch-stt.js';
import { CompanionCameraView } from './companion-camera.js';
import { LiveVisionLoop } from './live-vision.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

// iOS Safari (iPhone + iPadOS 13+) suppresses the <audio> element's native
// output once createMediaElementSource() binds it to an AudioContext, and
// then routes everything through the AudioContext destination — which
// honors the silent switch and aggressively suspends. Skipping the bind on
// iOS gives up amplitude-driven lipsync (phoneme schedule still works) in
// exchange for reliable playback. iPadOS reports as MacIntel; the
// maxTouchPoints check distinguishes it from a desktop Mac.
const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
            || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

let ws = null;
let audioContext = null;
let analyserNode = null;
// Live-camera (call surface): the composite view + the frame loop that
// streams video_frame over the call WS. Null until the user taps the
// camera button; torn down on hang-up.
let _callCamera = null;
let _callVisionLoop = null;
// When VR mode reroutes TTS audio through a spatial panner, holds the
// downstream destination node so restoreTtsRoute() can disconnect cleanly.
let _ttsRouteDest = null;

// --- Spatial audio routing (consumed by avatar-xr.js for VR) -------------
// voice.js owns the TTS node chain (ttsGainNode -> analyserNode -> speakers,
// with analyserNode also feeding the lip-sync FFT). In VR, avatar-xr.js wants
// the audio to come from the avatar's head via a positional panner — but it
// must not reach into the graph itself, so it asks for a route. The reroute
// happens DOWNSTREAM of analyserNode, so lip-sync FFT keeps reading the same
// analyserNode unchanged.

/** The TTS analyser node (lip-sync FFT source), or null before voice init. */
export function getTtsAnalyser() {
  return analyserNode;
}

/**
 * Route TTS audio (downstream of the analyser) into `destNode` — e.g. a
 * three.js PositionalAudio's gain node. Returns true on success, false and a
 * no-op if the audio graph isn't ready. Calling again re-routes to the new
 * destNode. On failure the analyser is reconnected straight to the speakers
 * so audio still plays (flat, not spatialized).
 */
export function routeTtsToNode(destNode) {
  if (!analyserNode || !destNode) return false;
  const ctx = analyserNode.context;
  if (!ctx || ctx.state === 'closed') return false;
  try {
    try { analyserNode.disconnect(_ttsRouteDest || ctx.destination); } catch { /* already gone */ }
    analyserNode.connect(destNode);
    _ttsRouteDest = destNode;
    return true;
  } catch (e) {
    console.warn('[Voice] routeTtsToNode failed — falling back to flat audio', e);
    try { analyserNode.connect(ctx.destination); } catch { /* swallow */ }
    _ttsRouteDest = null;
    return false;
  }
}

/** Undo routeTtsToNode(): reconnect the analyser straight to the speakers. */
export function restoreTtsRoute() {
  const dest = _ttsRouteDest;
  _ttsRouteDest = null;
  if (!analyserNode || !dest) return false;
  try { analyserNode.disconnect(dest); } catch { /* may already be gone */ }
  const ctx = analyserNode.context;
  if (ctx && ctx.state !== 'closed') {
    try { analyserNode.connect(ctx.destination); } catch { /* may already be connected */ }
  }
  return true;
}
let micStream = null;
// Hold-to-talk recorder for PTT mode. MediaRecorder → server batch STT
// (local Moonshine), injected as text via stage_send — bypasses the
// streaming PCM + VAD path for manual holds. null unless a hold is active.
let _pttRecorder = null;
let mediaRecorder = null;
let isRecording = false;
let isConnected = false;
let inputMode = 'ptt'; // 'ptt' | 'auto'
// Active client-side VAD (Silero WASM) — null when disabled, the
// pipeline policy is 'server', or the model failed to load. Wired up
// after the WS opens and the mic stream is granted.
let _clientVad = null;
let animFrameId = null;
let callStartTime = null;
let durationIntervalId = null;
let _pttFallbackTimer = null; // Safety timer: returns to listening if server doesn't respond
let _pendingXrUserSignals = [];

// Pose trigger engine — picks VRMAs based on conversation events.
// Instantiated when avatar mode activates with a VRM character.
// Disposed when avatar deactivates. Null when not in avatar mode.
let _poseTrigger = null;

/**
 * Build the sessionInfo object passed to avatarModule.activateAvatar.
 *
 * For solo characters this is just `{ mode, characterId }`. For group
 * chats it also resolves member NAMES (which is what session.groupMembers
 * stores — see narrative/index.js:7538) into `{id, name}` objects, since
 * avatar.js's group branch fetches each member's paired VRM by character
 * id (`/api/avatar/for-session?character_id=X`). Without this resolution
 * the group branch never fires (groupMembers absent) or fires with
 * undefined ids (silent fallback to a default avatar) — either way you
 * see only one character on screen.
 */
async function _resolveAvatarSessionInfo(activeSession, mode) {
  let charId = activeSession?.characterId || '';
  let groupMembers = null;

  if (activeSession?.groupId
      && Array.isArray(activeSession.groupMembers)
      && activeSession.groupMembers.length > 0) {
    try {
      const { narrative } = await import('./narrative/index.js');
      const charList = narrative.characters || [];
      const resolved = activeSession.groupMembers
        .map(name => {
          const c = charList.find(x => x.name === name);
          return c ? { id: c.id, name: c.name } : null;
        })
        .filter(Boolean);
      if (resolved.length > 0) groupMembers = resolved;
    } catch { /* narrative not loaded — fall back to single-avatar mode */ }
  }

  if (!charId && mode === 'narrative') {
    try {
      const { narrative } = await import('./narrative/index.js');
      charId = narrative.activeCharId || '';
    } catch { /* ignore */ }
  }

  // groupMode (round_robin / random / manual / llm_decide) flows down
  // from narrative.session — needed for per-mode PIP tap routing in
  // the avatar module (manual = address-and-swap, others = view-only).
  const groupMode = activeSession?.groupMode || '';

  return { mode, characterId: charId, groupMembers, groupMode };
}

/**
 * Resolve a short label identifying the chat / character / workspace the
 * voice call is bound to, for display on the persistent pet-mode pill.
 * Best-effort: returns '' if nothing useful is available rather than
 * forcing a placeholder, so the badge stays hidden in that case.
 */
async function _resolveCallHomeLabel(activeSession, mode) {
  const _trim = (s, n = 22) => {
    const t = String(s || '').trim();
    return t.length > n ? t.slice(0, n - 1) + '…' : t;
  };

  if (mode === 'narrative') {
    try {
      const { narrative } = await import('./narrative/index.js');
      // Group call: list up to 2 names then "+N more" so the badge stays
      // short on a 132px pill. Falls back to "Group" if the resolution
      // can't find any matches.
      if (Array.isArray(activeSession?.groupMembers) && activeSession.groupMembers.length > 1) {
        const names = activeSession.groupMembers.slice(0, 2).join(', ');
        const extra = activeSession.groupMembers.length - 2;
        return _trim(extra > 0 ? `${names} +${extra}` : names);
      }
      const charId = activeSession?.characterId || narrative.activeCharId || '';
      if (charId) {
        const char = (narrative.characters || []).find(c => c.id === charId);
        if (char?.name) return _trim(char.name);
      }
    } catch { /* narrative not loaded — fall through */ }
    return 'Story';
  }

  if (mode === 'coder') {
    try {
      const { coder } = await import('./coder.js');
      const ws = coder?.activeWorkspace?.();
      if (ws?.name) return _trim(ws.name);
    } catch { /* coder module unavailable — fall through */ }
    return 'Coder';
  }

  // Chat-family (passthrough / analytical / agentic): use session title,
  // falling back to a mode label so the pill is still distinguishable
  // from a narrative call.
  const title = activeSession?.title || activeSession?.name || '';
  if (title) return _trim(title);
  if (mode === 'analytical') return 'Analyze';
  if (mode === 'agentic') return 'Agentic';
  return 'Chat';
}

/** Write the home label into the pet pill's badge slot, hiding when empty. */
function _renderPillBadge() {
  const badge = document.getElementById('voice-pill-badge');
  if (!badge) return;
  if (_callHomeLabel) {
    badge.textContent = _callHomeLabel;
    badge.hidden = false;
  } else {
    badge.textContent = '';
    badge.hidden = true;
  }
}

/**
 * Send a one-shot "this character speaks next" hint to the voice WS.
 * Called by the avatar module on PIP tap when the group's generation
 * mode is "manual". Backend stores this on the VoiceSession and
 * applies it as `speaker_override` on the next chat request, then
 * clears — so manual override only affects the very next turn.
 */
export function sendVoiceSpeakerOverride(speakerName) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  if (!speakerName) return false;
  try {
    ws.send(JSON.stringify({ type: 'config', speaker_override: speakerName }));
    return true;
  } catch {
    return false;
  }
}

function _initPoseTriggerEngine() {
  if (_poseTrigger) return;
  if (!avatarModule.avatarState?.active || avatarModule.avatarState.mode !== 'vrm') return;
  // Engine is now a thin event router — selection lives in the
  // MovementConductor (avatar.js exports it as movementConductor and
  // also stashes it on avatarState.conductor for discoverability).
  _poseTrigger = new PoseTriggerEngine({
    conductor: avatarModule.movementConductor,
    character: avatarModule.avatarState?.avatarProfile || {},
  });
  _poseTrigger.start();
  _poseTrigger.onCallOpened();
}

function _disposePoseTriggerEngine() {
  if (!_poseTrigger) return;
  _poseTrigger.dispose();
  _poseTrigger = null;
}
let _disconnectTimer = null;  // Auto-cleanup timer after unexpected disconnect
let _heartbeatTimer = null;   // Periodic ping while connected — keeps idle WS alive through proxies
let _callSessionId = null;    // Pinned at call start — voice turns route back to this session even if user navigates
let _callHomeLabel = '';      // Short label identifying the bound session ("Becca", "Group: …", "Chat: …") — shown on the pet-mode pill so the user knows what they're still talking to after mode-switching away

// --- Mid-call reconnect state ---
// Separate from the initial-connect retry counter (_connectAttempt) which only
// covers handshake failure. These cover unexpected drops AFTER a successful
// connect — flaky WiFi, server restart, sleeping phone, etc. Cleared on
// successful reconnect (in ws.onopen) and on user-initiated teardown.
let _reconnectAttempt = 0;
let _reconnectTimer = null;
let _initialConnectRetryTimer = null;
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BACKOFF_MS = [3000, 6000, 12000];  // matches attempt index 0/1/2

// 25s sits below the typical 30-60s idle thresholds on Caddy / cloud LBs and
// well below uvicorn's default ws_ping_interval. PTT mode in particular sends
// no audio frames while the user is silent, so without this the WS gets
// reaped during long pauses and the next user action lands on a dead socket.
const _HEARTBEAT_MS = 25000;

// Minimize/expand state
let isMinimized = false;
let _preMinimizeInputMode = null;
let _pillCanvas = null;
let _pillCtx = null;
let _pillTimerEl = null;
let _pillEl = null;
let _pillAvatarHost = null;   // .voice-pill-avatar — VRM canvas reparents here in pet mode
let _pillPetMode = false;     // true while a minimized avatar call is showing as a desktop-pet
let _pillPos = null;          // {x,y} top-left in px once the user has dragged the pill; null = default corner
let _pillDragStart = null;
let _pillDragMoved = false;
let _pillJustDragged = false; // set on dragend so the trailing click doesn't also expand the call
let _pillDragArmed = false;   // mobile: true after a long-press, allowing drag without a tap accidentally triggering expand
let _pillLongPressTimer = null;
const _PILL_LONGPRESS_MS = 400;    // mobile long-press dwell before drag arms
const _PILL_LONGPRESS_SLOP = 6;    // px movement that cancels the long-press (vs jitter)
let _minimizeAnim = null;
let _pillOrbAnimId = null;
const _PILL_POS_KEY = 'augmentum.voicePillPos';

// Server-side VAD mode — set by server on connect
let serverVadActive = false;
let _serverSttAvailable = true;  // Assume yes until server tells us otherwise

// Browser SpeechRecognition fallback (when no server STT configured)
let _browserStt = null;
let _browserSttInterim = '';  // Current interim transcript
let pcmWorkletNode = null;          // AudioWorkletNode for raw PCM capture
let micSourceNode = null;           // Persistent ref to prevent GC of audio pipeline

// VAD state (client-side auto-detect mode — legacy fallback)
let vadActive = false;
let vadSilenceStart = null;
const VAD_SILENCE_MS_MIN = 800;     // Fastest turn end (rapid conversation)
const VAD_SILENCE_MS_MAX = 2200;    // Slowest turn end (thoughtful speaker)
const VAD_SILENCE_MS_DEFAULT = 1500; // Default silence threshold
let vadSilenceMs = VAD_SILENCE_MS_DEFAULT;
const VAD_MIN_SPEECH_MS = 300;      // Min speech before we consider it real
// Adaptive silence: track recent speech durations to scale threshold
const _recentSpeechDurations = [];  // Last N speech durations in ms
const _ADAPT_WINDOW = 5;            // Number of turns to average over
const VAD_ECHO_COOLDOWN_MS = 800;   // Must match server-side _echo_cooldown_s in voice_routes.py
let vadSpeechStart = null;
let vadSuppressed = false;          // Suppress during TTS playback (echo)
let vadSuppressUntil = 0;           // Timestamp-based cooldown after TTS ends

// [1] Spectral VAD — frequency-band speech detection (replaces naive RMS)
const VAD_SPEECH_LOW_HZ = 85;       // Lower bound of human speech fundamental
const VAD_SPEECH_HIGH_HZ = 4000;    // Upper bound of speech formants
const VAD_SPECTRAL_RATIO = 0.35;    // Min ratio of speech-band energy to total
const VAD_PROB_THRESHOLD = 0.04;    // Speech probability threshold for new speech detection
const VAD_INTERRUPT_THRESHOLD = 0.12; // Higher threshold during TTS playback (ignore background noise)
const VAD_SMOOTHING = 0.7;          // Exponential smoothing for speech probability
let vadSpeechProb = 0;              // Smoothed speech probability [0-1]

// [3] Prefix padding — continuous capture with ring buffer (auto mode)
let vadStreamingActive = false;      // Whether chunks go to server vs ring buffer
let prefixBuffer = [];               // Ring buffer of recent audio ArrayBuffers
const PREFIX_BUFFER_SIZE = 2;        // Keep last 2 chunks (~500ms at 250ms/chunk)

// [4] Audio ducking — reduce TTS volume when user speaks before full barge-in
let ttsGainNode = null;
let duckingActive = false;
// Held while voice mode is actively speaking out loud, so audiobook/Grove
// duck. Released when sentenceQueue drains or a barge-in interrupts.
let _voiceBusClaim = null;
const DUCK_RAMP_S = 0.05;           // Seconds to ramp down volume
const DUCK_LEVEL = 0.15;            // Volume level during ducking (0-1)
const DUCK_TO_CANCEL_MS = 900;      // Escalate ducking to full barge-in cancel (was 500ms — too sensitive to noise)
let duckStartTime = null;

// [5] Backchannel filtering — DISABLED.
// Primary STT is Moonshine, not Whisper. Moonshine doesn't hallucinate.
// Let the LLM handle all speech naturally — even "yeah", "ok", "hey".
const BACKCHANNEL_RE = null;
const BACKCHANNEL_MAX_WORDS = 0;

// [6] Playback position tracking — know what user heard on interrupt
let currentResponseText = '';        // Full AI response text accumulated from llm_delta
let currentResponseSaved = false;    // Whether the current turn was saved to chat
let _turnImages = [];                // Tool-result image URLs for THIS turn (image_search /
                                     // image_generation) — appended as markdown when the turn
                                     // is persisted so they survive in chat + history.
let playedSentenceCount = 0;         // Sentences fully played to user
let queuedSentenceCount = 0;         // Sentences added to queue

// Audio playback — per-sentence accumulation
let sentenceAudioChunks = [];        // Chunks for current sentence
let sentenceQueue = [];              // Queue of complete sentence blobs/{format,chunks} objects
let isPlaying = false;
let currentAudio = null;
let _currentAudioCleanup = null;
let turnDone = false;  // True when server signals turn_complete/tts_end — gates listening transition
let currentTtsFormat = 'mp3';        // Current sentence audio format ('mp3', 'pcm', etc.)
let currentTtsSentence = '';         // Text of current sentence being accumulated (for pacing)
let pendingVisemeSchedules = [];     // FIFO of {duration_ms, events} from server (phoneme lipsync)
const PCM_SAMPLE_RATE = 24000;       // Qwen3-TTS PCM output: int16 @ 24kHz
let _ttsPlaybackActive = false;       // True while decoded TTS is actually playing
let _ttsPlaybackTailTimer = null;     // Short hold for speaker/output latency
const TTS_PLAYBACK_TAIL_MS = 220;

// ---------------------------------------------------------------------------
// Per-Mode Voice Preferences
// ---------------------------------------------------------------------------
let _voicePrefsTimer;
function _saveVoicePrefs() {
  clearTimeout(_voicePrefsTimer);
  _voicePrefsTimer = setTimeout(async () => {
    const mode = app.state.mode || 'passthrough';
    const prefs = {
      avatar_active: avatarModule.avatarState.active,
      stage_active: stageActive,
      input_mode: inputMode,
    };
    try {
      await fetch(`/api/config/voice-prefs/${mode}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(prefs),
      });
    } catch { /* best-effort */ }
  }, 500);
}

// DOM refs
let overlay, canvas, ctx, statusText;
let transcriptUser, transcriptAi, pttBtn, pttLabel;
let modeTogglePtt, modeToggleAuto, infoEl, durationEl;
let toolsToggleBtn;

// Voice selector pill state
let _voiceInfoDimTimer = null;
let _voiceModelPanel = null;
let _voiceVoicePanel = null;
let _voiceCachedModels = null;
let _voiceCachedVoices = null;
let _previewAudio = null; // Currently playing voice preview
let voiceToolsEnabled = false;

// Refresh the local voice cache when the central cache changes — a voice
// cloned/mixed by this user, or an admin adding/removing an audio provider
// (audio_routes.py emits voices.changed over the SSE bus; fires on first
// populate too). If the voice picker is open, re-render it live so the new
// voice appears without closing the panel — the installed PWA has no
// manual refresh.
onCacheChange('voices', (voices) => {
  _voiceCachedVoices = Array.isArray(voices) ? voices : null;
  if (_voiceVoicePanel?.classList.contains('visible')) {
    const search = _voiceVoicePanel.querySelector('.voice-selector-search input');
    _renderVoicePanel(search?.value.trim().toLowerCase() || '');
  }
});

// ---------------------------------------------------------------------------
// Orb Visualization State
// ---------------------------------------------------------------------------
const ORB_POINTS = 64;
const ORB_TWO_PI = Math.PI * 2;

const ORB_PROFILES = {
  connecting: { stiffness: 0.06, damping: 0.90, breathHz: 0.4, audioGain: 0.0, radiusScale: 0.90, hslInner: [215,18,48], hslOuter: [220,14,36], shadowBlur: 30 },
  listening:  { stiffness: 0.08, damping: 0.85, breathHz: 0.8, audioGain: 0.3, radiusScale: 1.00, hslInner: [187,42,60], hslOuter: [192,38,46], shadowBlur: 42 },
  recording:  { stiffness: 0.15, damping: 0.82, breathHz: 1.1, audioGain: 1.0, radiusScale: 1.05, hslInner: [18,58,64],  hslOuter: [12,50,50],  shadowBlur: 55 },
  processing: { stiffness: 0.20, damping: 0.88, breathHz: 0.6, audioGain: 0.1, radiusScale: 0.85, hslInner: [222,35,64], hslOuter: [228,30,50], shadowBlur: 45 },
  speaking:   { stiffness: 0.12, damping: 0.85, breathHz: 0.9, audioGain: 0.7, radiusScale: 1.00, hslInner: [272,40,66], hslOuter: [260,35,54], shadowBlur: 50 },
  composing:  { stiffness: 0.06, damping: 0.90, breathHz: 0.5, audioGain: 0.15, radiusScale: 0.95, hslInner: [198,22,52], hslOuter: [205,18,40], shadowBlur: 32 },
};

let orbRadius = [];
let orbVelocity = [];
let orbTarget = [];
let orbMidRadius = [];
let orbMidVelocity = [];
let orbOuterRadius = [];
let orbOuterVelocity = [];

const MID_DELAY_FRAMES = 6;
let midDelayBuffer = [];
let midDelayIndex = 0;

let orbEnergy = null;
let orbTargetProfile = 'connecting';
let _orbRmsSmoothed = 0;
let _orbWrapEl = null;

let ttsPulseActive = false;
let ttsPulsePhase = 0;

// Progressive disclosure
let _conversingTimer = null;
let _userTranscriptDimTimer = null;
let _aiTranscriptDimTimer = null;

function orbNoise(angle, time) {
  return Math.sin(angle * 3.0 + time * 0.7) * 0.3
       + Math.sin(angle * 5.0 - time * 1.1) * 0.2
       + Math.sin(angle * 7.0 + time * 0.5) * 0.15;
}

function lerpHSL(a, b, t) {
  let dh = b[0] - a[0];
  if (dh > 180) dh -= 360;
  if (dh < -180) dh += 360;
  return [
    (a[0] + dh * t + 360) % 360,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function hslStr(hsl, alpha) {
  return `hsla(${hsl[0]|0}, ${hsl[1]|0}%, ${hsl[2]|0}%, ${alpha})`;
}

function lerpVal(a, b, t) { return a + (b - a) * t; }

function _enterConversing() {
  if (overlay) overlay.classList.add('conversing');
  clearTimeout(_conversingTimer);
  _conversingTimer = null;
}

function _scheduleExitConversing() {
  // Don't exit conversing while audio is still playing or queued —
  // wait until playback actually finishes.
  if (isPlaying || sentenceQueue.length > 0) return;
  clearTimeout(_conversingTimer);
  _conversingTimer = setTimeout(() => {
    // Double-check: audio may have started since the timeout was scheduled
    if (isPlaying || sentenceQueue.length > 0) return;
    if (overlay) overlay.classList.remove('conversing');
    _conversingTimer = null;
  }, 3000);
}

function _resetUserTranscriptDim() {
  clearTimeout(_userTranscriptDimTimer);
  if (transcriptUser) transcriptUser.classList.remove('dimmed');
  _userTranscriptDimTimer = setTimeout(() => {
    if (transcriptUser) transcriptUser.classList.add('dimmed');
  }, 6000);
}

// ---------------------------------------------------------------------------
// Conversation Transcript Log - stable streaming bubbles
// ---------------------------------------------------------------------------

let _currentAiBubble = null;   // Active AI response bubble element
let _currentAiText = '';       // Active AI response text, rendered as one stable bubble
let _aiRenderTimer = null;     // Throttles DOM updates during fast streams
let _lastAiRenderAt = 0;
const _AI_STREAM_RENDER_MS = 90;

/** Add a user speech bubble to the conversation log. Replaces any live partial bubble. */
function _addUserBubble(text) {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;
  // Remove live partial bubble if present
  const live = log.querySelector('.voice-log-live');
  if (live) live.remove();
  const bubble = document.createElement('div');
  bubble.className = 'voice-log-bubble voice-log-user';
  bubble.textContent = text;
  log.appendChild(bubble);
  _scrollLog(log);
}

/** Show/update a live partial transcript bubble (while user is still speaking). */
function _updateLivePartial(text) {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;
  let live = log.querySelector('.voice-log-live');
  if (!live) {
    live = document.createElement('div');
    live.className = 'voice-log-bubble voice-log-user voice-log-live';
    log.appendChild(live);
  }
  live.textContent = text;
  _scrollLog(log);
}

/** Start a new AI response bubble in the conversation log. */
function _startNewAiBubble() {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;
  if (_aiRenderTimer) {
    clearTimeout(_aiRenderTimer);
    _aiRenderTimer = null;
  }
  _currentAiBubble = document.createElement('div');
  _currentAiBubble.className = 'voice-log-bubble voice-log-ai voice-log-streaming';
  log.appendChild(_currentAiBubble);
  _currentAiText = '';
  _lastAiRenderAt = 0;
  _scrollLog(log);
}

/** Append streaming AI delta text into the active assistant bubble. */
function _appendAiDelta(text) {
  return _appendAiDeltaStable(text);
}

function _cleanAiStreamText(text) {
  // Strip image markdown and raw image URLs; images are shown as cards via tool_result.
  let cleaned = text.replace(/!\[[^\]]*\]\([^)]+\)/g, '');
  cleaned = cleaned.replace(/https?:\/\/\S*\/api\/image\/\S*/g, '');
  cleaned = cleaned.replace(/\/api\/image\/\S*/g, '');
  return cleaned;
}

function _appendVoiceTextNode(container, text) {
  if (text) container.appendChild(document.createTextNode(text));
}

function _appendVoiceLink(container, label, url) {
  const a = document.createElement('a');
  a.href = url;
  a.textContent = label;
  a.className = 'voice-word-link';
  a.addEventListener('click', (e) => {
    e.preventDefault();
    _openLinkInBrowse(url);
  });
  container.appendChild(a);
}

function _renderVoiceBubbleText(bubble, text) {
  bubble.textContent = '';
  const linkRe = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|(https?:\/\/[^\s]+)/g;
  let cursor = 0;
  let match;
  while ((match = linkRe.exec(text)) !== null) {
    _appendVoiceTextNode(bubble, text.slice(cursor, match.index));
    if (match[2]) {
      _appendVoiceLink(bubble, match[1], match[2]);
    } else if (match[3]) {
      const rawUrl = match[3];
      const trailing = rawUrl.match(/[),.!?;:]+$/)?.[0] || '';
      const url = trailing ? rawUrl.slice(0, -trailing.length) : rawUrl;
      _appendVoiceLink(bubble, url.replace(/^https?:\/\/(www\.)?/, '').slice(0, 44), url);
      if (trailing) _appendVoiceTextNode(bubble, trailing);
    }
    cursor = match.index + match[0].length;
  }
  _appendVoiceTextNode(bubble, text.slice(cursor));
}

function _renderCurrentAiBubble(force = false) {
  if (!_currentAiBubble) return;
  if (_aiRenderTimer) {
    clearTimeout(_aiRenderTimer);
    _aiRenderTimer = null;
  }
  const now = performance.now();
  if (!force && _lastAiRenderAt > 0 && now - _lastAiRenderAt < _AI_STREAM_RENDER_MS) {
    _aiRenderTimer = setTimeout(() => _renderCurrentAiBubble(true), _AI_STREAM_RENDER_MS);
    return;
  }
  _lastAiRenderAt = now;
  _renderVoiceBubbleText(_currentAiBubble, _currentAiText);

  const log = _currentAiBubble.closest('.voice-transcript-log');
  if (log) _scrollLog(log, false);
}

function _bestAssistantTurnText(serverText = '') {
  const streamed = (currentResponseText || '').trim();
  const server = (serverText || '').trim();
  if (!streamed) return server;
  if (!server) return streamed;
  return streamed.length >= server.length ? streamed : server;
}

/** Append streaming AI delta text into one stable, readable bubble. */
function _appendAiDeltaStable(text) {
  if (!_currentAiBubble) return;
  const cleaned = _cleanAiStreamText(text);
  if (!cleaned.trim() && text.trim()) return; // delta was entirely an image ref
  _currentAiText += cleaned;
  _renderCurrentAiBubble(false);
}

function _finalizeAiBubble(text = '') {
  if (!_currentAiBubble) return;
  const finalText = _cleanAiStreamText(text).trim();
  if (finalText) _currentAiText = finalText;
  _renderCurrentAiBubble(true);
  _currentAiBubble.classList.remove('voice-log-streaming');
}

function _scrollLog(log, force = true) {
  requestAnimationFrame(() => {
    const distanceFromBottom = log.scrollHeight - log.scrollTop - log.clientHeight;
    if (!force && distanceFromBottom > 72) return;
    log.scrollTop = log.scrollHeight;
  });
}

function _resetVoiceTranscriptLog() {
  const log = document.querySelector('.voice-transcript-log');
  if (log) log.querySelectorAll('.voice-log-bubble').forEach(el => el.remove());
  if (_aiRenderTimer) {
    clearTimeout(_aiRenderTimer);
    _aiRenderTimer = null;
  }
  _currentAiBubble = null;
  _currentAiText = '';
  _lastAiRenderAt = 0;
}

/**
 * Remove the trailing assistant+user exchange from the transcript log.
 * The UI half of ``conversation.strike`` — the server already scrubbed
 * the same exchange from the model's context. Mirrors the server's
 * trailing-assistant-first ordering: drop a streaming/finished AI bubble
 * if it's the tail, then the user bubble before it. Also clears any
 * live partial so an in-progress STT bubble doesn't survive the strike.
 */
export function strikeLastExchangeUI() {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;
  const live = log.querySelector('.voice-log-live');
  if (live) live.remove();
  const bubbles = log.querySelectorAll('.voice-log-bubble');
  let i = bubbles.length - 1;
  if (i >= 0 && bubbles[i].classList.contains('voice-log-ai')) {
    if (bubbles[i] === _currentAiBubble) {
      _currentAiBubble = null;
      _currentAiText = '';
    }
    bubbles[i].remove();
    i -= 1;
  }
  if (i >= 0 && bubbles[i].classList.contains('voice-log-user')) {
    bubbles[i].remove();
  }
}

function _appendHistoryBubble(role, text) {
  const log = document.querySelector('.voice-transcript-log');
  const content = (text || '').trim();
  if (!log || !content) return;
  const bubble = document.createElement('div');
  bubble.className = role === 'user'
    ? 'voice-log-bubble voice-log-user voice-log-history'
    : 'voice-log-bubble voice-log-ai voice-log-history';
  if (role === 'assistant') {
    _renderVoiceBubbleText(bubble, content);
  } else {
    bubble.textContent = content;
  }
  log.appendChild(bubble);
}

function _voiceMessageText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (part?.type === 'text') return part.text || '';
        if (part?.type === 'image_url') return '[image]';
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  return '';
}

function _seedVoiceTranscriptLogFromChat(limit = 30) {
  _resetVoiceTranscriptLog();
  const session = chat.getActiveSession();
  if (!session) return;
  const messages = chat.buildMessagesForAPI(session)
    .map((msg) => ({ role: msg.role, content: _voiceMessageText(msg.content) }))
    .filter((msg) => (msg.role === 'user' || msg.role === 'assistant') && msg.content.trim())
    .slice(-limit);
  for (const msg of messages) {
    _appendHistoryBubble(msg.role, msg.content);
  }
  const log = document.querySelector('.voice-transcript-log');
  if (log) _scrollLog(log);
}

/** Open a link in the browse panel and minimize the voice call. */
function _openLinkInBrowse(url) {
  // Minimize the call — user can keep talking via the pill
  if (isConnected && !isMinimized) {
    minimizeVoiceCall();
  }
  // Open URL in the browse reader panel
  document.dispatchEvent(new CustomEvent('augmentum:browse-url', {
    detail: { url },
  }));
}

function _resetAiTranscriptDim() {
  clearTimeout(_aiTranscriptDimTimer);
  const log = document.querySelector('.voice-transcript-log');
  if (log) log.classList.remove('dimmed');
  if (transcriptAi) transcriptAi.classList.remove('dimmed');
  _aiTranscriptDimTimer = setTimeout(() => {
    if (!isPlaying && sentenceQueue.length === 0) {
      if (log) log.classList.add('dimmed');
      if (transcriptAi) transcriptAi.classList.add('dimmed');
    }
  }, 8000);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
export function initVoice() {
  overlay = document.getElementById('voice-overlay');
  if (!overlay) return;

  canvas = overlay.querySelector('.voice-orb');
  statusText = overlay.querySelector('.voice-status-text');

  // Resize orb canvas on window resize (phone rotation, window drag).
  // rAF-coalesce so a drag-resize firing 60+ events/sec doesn't trigger
  // 60 canvas context resets per second and stall the GPU.
  let _orbResizePending = false;
  window.addEventListener('resize', () => {
    if (_orbResizePending) return;
    _orbResizePending = true;
    requestAnimationFrame(() => {
      _orbResizePending = false;
      if (canvas) resizeCanvas();
    });
  });
  transcriptUser = overlay.querySelector('.voice-transcript-user');
  transcriptAi = overlay.querySelector('.voice-transcript-ai');
  pttBtn = overlay.querySelector('.voice-ptt-btn');
  pttLabel = overlay.querySelector('.voice-ptt-label');
  modeTogglePtt = overlay.querySelector('[data-voice-mode="ptt"]');
  modeToggleAuto = overlay.querySelector('[data-voice-mode="auto"]');
  infoEl = overlay.querySelector('.voice-info');
  durationEl = overlay.querySelector('.voice-info-duration');
  const avatarTranscriptToggle = document.getElementById('voice-avatar-transcript-toggle');
  const applyAvatarTranscriptCollapsed = (collapsed) => {
    overlay.classList.toggle('avatar-transcript-collapsed', collapsed);
    if (!avatarTranscriptToggle) return;
    avatarTranscriptToggle.setAttribute('aria-pressed', String(collapsed));
    const tip = collapsed ? 'Show transcript' : 'Hide transcript';
    avatarTranscriptToggle.title = tip;
    avatarTranscriptToggle.setAttribute('aria-label', tip);
    avatarTranscriptToggle.classList.toggle('active', !collapsed);
  };
  applyAvatarTranscriptCollapsed(localStorage.getItem('augmentum-avatar-transcript-collapsed') !== '0');
  if (avatarTranscriptToggle) {
    avatarTranscriptToggle.addEventListener('click', () => {
      const collapsed = !overlay.classList.contains('avatar-transcript-collapsed');
      localStorage.setItem('augmentum-avatar-transcript-collapsed', collapsed ? '1' : '0');
      applyAvatarTranscriptCollapsed(collapsed);
    });
  }

  // Call button
  const callBtn = document.getElementById('voice-call-btn');
  if (callBtn) {
    callBtn.addEventListener('click', toggleVoiceCall);
  }

  // Wake-word trigger hook — invoked by becca-wake when a trained
  // phrase fires. No-op if already in a call (refractory is already
  // enforced on the server side per source).
  window.__beccaTriggerVoiceCall = () => {
    if (isConnected) return;
    try { startVoiceCall(); }
    catch (err) { console.warn('[voice] wake-triggered call failed', err); }
  };

  // PTT button — mousedown/up + touch
  if (pttBtn) {
    pttBtn.addEventListener('mousedown', startPtt);
    pttBtn.addEventListener('mouseup', stopPtt);
    pttBtn.addEventListener('mouseleave', stopPtt);
    pttBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startPtt(); });
    pttBtn.addEventListener('touchend', (e) => { e.preventDefault(); stopPtt(); });
  }

  // End call
  const endBtn = overlay.querySelector('.voice-end-btn');
  if (endBtn) {
    endBtn.addEventListener('click', endVoiceCall);
  }

  // Reconnect (shown on unexpected disconnect)
  document.getElementById('voice-reconnect-btn')?.addEventListener('click', reconnectVoiceCall);
  // Autoplay-block re-arm — the click IS the gesture that re-authorizes
  // audio on iOS, so the handler must do the resume/prime work inline.
  document.getElementById('voice-audio-unlock-btn')?.addEventListener('click', _unlockAudioPlayback);
  window.addEventListener('augmentum:xr-open-surface', _handleXrSurfaceRequest);
  window.addEventListener('augmentum:xr-switch-mode', _handleXrModeSwitchRequest);
  window.addEventListener('augmentum:xr-session-state', _handleXrSessionState);
  window.addEventListener('augmentum:xr-user-signal', _handleXrUserSignal);

  // Pause / Resume button
  const pauseBtn = document.getElementById('voice-pause-btn');
  if (pauseBtn) {
    pauseBtn.addEventListener('click', togglePause);
  }

  // Live camera — reveal only when the capability is enabled. Lets the
  // companion see you (front) or the world (back) during the call; the
  // same composite the presence widget's eye uses.
  const cameraBtn = document.getElementById('voice-camera-btn');
  if (cameraBtn) {
    let _camEnabled = false;
    try { _camEnabled = !!getSettings().companionLiveVisionEnabled; } catch (_) {}
    cameraBtn.style.display = _camEnabled ? '' : 'none';
    cameraBtn.addEventListener('click', _toggleCallCamera);
  }
  const cameraFlipBtn = document.getElementById('voice-camera-flip-btn');
  if (cameraFlipBtn) {
    cameraFlipBtn.addEventListener('click', () => {
      if (_callCamera) _callCamera.flip();
    });
  }

  // Scene image button — background generation, doesn't block conversation
  const sceneImgBtn = document.getElementById('voice-scene-img-btn');
  if (sceneImgBtn) {
    sceneImgBtn.addEventListener('click', () => _generateSceneImage(sceneImgBtn));
  }

  // Enter VR button — only revealed when the browser/device reports
  // immersive-vr support. Click loads the modern-room scene over the
  // existing avatar scene and requests an XR session. See avatar-xr.js.
  const vrBtn = document.getElementById('voice-vr-btn');
  if (vrBtn) {
    avatarXR.isXRSupported().then((ok) => {
      if (ok) vrBtn.hidden = false;
    });
    vrBtn.addEventListener('click', () => {
      _enterVoiceXR('vr', vrBtn).catch((err) => {
        console.error('[voice] VR enter failed:', err);
        showToast(_friendlyVrError(err), 'error', 4500);
      });
    });
  }

  const mrBtn = document.getElementById('voice-mr-btn');
  if (mrBtn) {
    avatarXR.isMRSupported().then((ok) => {
      if (ok) mrBtn.hidden = false;
    });
    mrBtn.addEventListener('click', () => {
      _enterVoiceXR('mr', mrBtn).catch((err) => {
        console.error('[voice] MR enter failed:', err);
        showToast(_friendlyVrError(err), 'error', 4500);
      });
    });
  }

  // Mode toggle
  if (modeTogglePtt) modeTogglePtt.addEventListener('click', () => setInputMode('ptt'));
  if (modeToggleAuto) modeToggleAuto.addEventListener('click', () => setInputMode('auto'));

  // Tools toggle
  toolsToggleBtn = document.getElementById('voice-tools-toggle');
  if (toolsToggleBtn) {
    toolsToggleBtn.addEventListener('click', () => {
      voiceToolsEnabled = !voiceToolsEnabled;
      toolsToggleBtn.classList.toggle('active', voiceToolsEnabled);
      // Send updated config to server if connected
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'config',
          tools: voiceToolsEnabled ? ['all'] : [],
        }));
      }
    });
  }

  // Spacebar PTT — skip when typing in an editable element (stage manager, search, etc.)
  const _isTyping = () => {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
  };
  document.addEventListener('keydown', (e) => {
    if (!isConnected || !overlay.classList.contains('active')) return;
    if (e.code === 'Space' && inputMode === 'ptt' && !e.repeat && !_isTyping()) {
      e.preventDefault();
      startPtt();
    }
    if (e.key === 'Escape') {
      endVoiceCall();
    }
  });
  document.addEventListener('keyup', (e) => {
    if (!isConnected || !overlay.classList.contains('active')) return;
    if (e.code === 'Space' && inputMode === 'ptt' && !_isTyping()) {
      e.preventDefault();
      stopPtt();
    }
  });

  // Detect a half-open WS the moment the tab regains visibility.
  document.addEventListener('visibilitychange', _onVisibilityChange);

  // Voice selector pills
  const pillModel = document.getElementById('voice-pill-model');
  const pillVoice = document.getElementById('voice-pill-voice');
  if (pillModel) {
    pillModel.addEventListener('click', (e) => { e.stopPropagation(); _openPanel('model'); });
  }
  if (pillVoice) {
    pillVoice.addEventListener('click', (e) => { e.stopPropagation(); _openPanel('voice'); });
  }
  document.addEventListener('click', _closeAllPanels);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') _closeAllPanels();
  });

  // Dim timer — wire mouseenter/click on voice-info
  const voiceInfoEl = document.getElementById('voice-info');
  if (voiceInfoEl) {
    voiceInfoEl.addEventListener('mouseenter', _resetDimTimer);
    voiceInfoEl.addEventListener('click', _resetDimTimer);
  }

  // Touch disclosure — reveal dimmed controls on tap
  const touchRevealEls = [
    document.getElementById('voice-info'),
    document.querySelector('.voice-top-right'),
  ];
  for (const el of touchRevealEls) {
    if (!el) continue;
    el.addEventListener('touchstart', () => {
      el.classList.add('touch-revealed');
      setTimeout(() => el.classList.remove('touch-revealed'), 4000);
    }, { passive: true });
  }

  // Restore input mode preference
  const savedMode = localStorage.getItem('augmentum-voice-mode');
  if (savedMode === 'auto' || savedMode === 'ptt') {
    inputMode = savedMode;
  }
  applyInputMode();

  // Pre-fetch models and voices at app startup so pills populate instantly on call start
  _fetchAndCacheSelectors();

  // Stage manager (narrative formatting)
  _initStage();

  // Avatar toggle
  const avatarToggle = document.getElementById('voice-avatar-toggle');
  if (avatarToggle) {
    // Short press: toggle avatar on/off
    // Long press (500ms): open avatar picker to switch mid-call
    let _avatarLongPress = null;
    let _avatarLongFired = false;

    avatarToggle.addEventListener('pointerdown', () => {
      _avatarLongFired = false;
      _avatarLongPress = setTimeout(() => {
        _avatarLongFired = true;
        if (avatarModule.avatarState.active) {
          avatarModule.showAvatarPicker();
        }
      }, 500);
    });

    avatarToggle.addEventListener('pointerup', async () => {
      clearTimeout(_avatarLongPress);
      if (_avatarLongFired) return; // long press already handled
      if (avatarModule.avatarState.active) {
        _disposePoseTriggerEngine();
        avatarModule.deactivateAvatar();
      } else {
        const sessionInfo = await _resolveAvatarSessionInfo(
          chat.getActiveSession(),
          app.state.mode || 'passthrough',
        );
        await avatarModule.activateAvatar(analyserNode, sessionInfo);
        _initPoseTriggerEngine();
      }
      _saveVoicePrefs();
    });

    avatarToggle.addEventListener('pointercancel', () => {
      clearTimeout(_avatarLongPress);
    });
  }

  // Pill element refs
  _pillEl = document.getElementById('voice-pill');
  if (_pillEl) {
    _pillCanvas = _pillEl.querySelector('.voice-pill-orb');
    _pillTimerEl = _pillEl.querySelector('.voice-pill-timer');
    _pillAvatarHost = _pillEl.querySelector('.voice-pill-avatar');
    _pillCtx = _pillCanvas?.getContext('2d');
    _pillEl.addEventListener('click', _onPillClick);
    _pillEl.addEventListener('pointerdown', _onPillPointerDown);
    _pillPos = _loadPillPos();
    // Keep a dragged pill on-screen if the window shrinks.
    window.addEventListener('resize', () => { if (isMinimized && _pillPos) _applyPillPos(); });
  }

  // Minimize button
  const minimizeBtn = document.getElementById('voice-minimize-btn');
  if (minimizeBtn) {
    minimizeBtn.addEventListener('click', minimizeVoiceCall);
  }

  // Command composer — programmatic minimize from command chain
  document.addEventListener('voice:minimize', () => minimizeVoiceCall());
}

// ---------------------------------------------------------------------------
// Model & Voice Selector Pills
// ---------------------------------------------------------------------------

const _DIM_TIMEOUT_MS = 15000;

function _resetDimTimer() {
  const el = document.getElementById('voice-info');
  if (!el) return;
  el.classList.remove('dimmed');
  clearTimeout(_voiceInfoDimTimer);
  _voiceInfoDimTimer = setTimeout(() => {
    el.classList.add('dimmed');
  }, _DIM_TIMEOUT_MS);
}

function _clearDimTimer() {
  clearTimeout(_voiceInfoDimTimer);
  _voiceInfoDimTimer = null;
  const el = document.getElementById('voice-info');
  if (el) el.classList.remove('dimmed');
}

function _resolveVoiceDisplayName(voiceId) {
  if (!voiceId) return '';
  // voice IDs can be "provider_id::voice_name" — show the voice part
  const parts = voiceId.split('::');
  return parts.length > 1 ? parts[parts.length - 1] : voiceId;
}

function _closeAllPanels() {
  if (_previewAudio) { _previewAudio.pause(); _previewAudio = null; }
  if (_voiceModelPanel) _voiceModelPanel.classList.remove('visible');
  if (_voiceVoicePanel) _voiceVoicePanel.classList.remove('visible');
  const pillModel = document.getElementById('voice-pill-model');
  const pillVoice = document.getElementById('voice-pill-voice');
  if (pillModel) pillModel.classList.remove('open');
  if (pillVoice) pillVoice.classList.remove('open');
}

function _ensurePanel(type) {
  if (type === 'model' && _voiceModelPanel) return _voiceModelPanel;
  if (type === 'voice' && _voiceVoicePanel) return _voiceVoicePanel;

  const pill = document.getElementById(type === 'model' ? 'voice-pill-model' : 'voice-pill-voice');
  if (!pill) return null;

  const panel = document.createElement('div');
  panel.className = 'voice-selector-panel';
  panel.addEventListener('click', (e) => e.stopPropagation());

  const searchWrap = document.createElement('div');
  searchWrap.className = 'voice-selector-search';
  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.placeholder = type === 'model' ? 'Search models\u2026' : 'Search voices\u2026';
  searchInput.addEventListener('input', () => {
    const filter = searchInput.value.trim().toLowerCase();
    if (type === 'model') _renderModelPanel(filter);
    else _renderVoicePanel(filter);
  });
  searchWrap.appendChild(searchInput);
  panel.appendChild(searchWrap);

  const list = document.createElement('div');
  list.className = 'voice-selector-list';
  panel.appendChild(list);

  pill.appendChild(panel);

  if (type === 'model') _voiceModelPanel = panel;
  else _voiceVoicePanel = panel;

  return panel;
}

function _openPanel(type) {
  _resetDimTimer();
  // Close the other panel
  if (type === 'model' && _voiceVoicePanel) {
    _voiceVoicePanel.classList.remove('visible');
    document.getElementById('voice-pill-voice')?.classList.remove('open');
  }
  if (type === 'voice' && _voiceModelPanel) {
    _voiceModelPanel.classList.remove('visible');
    document.getElementById('voice-pill-model')?.classList.remove('open');
  }

  const panel = _ensurePanel(type);
  if (!panel) return;

  const pill = document.getElementById(type === 'model' ? 'voice-pill-model' : 'voice-pill-voice');
  const isOpen = panel.classList.contains('visible');
  if (isOpen) {
    panel.classList.remove('visible');
    if (pill) pill.classList.remove('open');
    return;
  }

  if (pill) pill.classList.add('open');
  if (type === 'model') _renderModelPanel('');
  else _renderVoicePanel('');
  panel.classList.add('visible');

  // Auto-focus search
  const searchInput = panel.querySelector('.voice-selector-search input');
  if (searchInput) {
    searchInput.value = '';
    requestAnimationFrame(() => searchInput.focus());
  }
}

function _renderModelPanel(filter) {
  const panel = _voiceModelPanel;
  if (!panel) return;
  const list = panel.querySelector('.voice-selector-list');
  if (!list) return;

  const models = _voiceCachedModels || [];
  const current = app.state.currentModel || '';

  // Recently used (from same localStorage key as main selector)
  let recentNames = [];
  try {
    recentNames = JSON.parse(localStorage.getItem('augmentum-recent-models') || '[]');
  } catch { /* ignore */ }

  const filtered = filter
    ? models.filter(m => (m.name || m.model || '').toLowerCase().includes(filter))
    : models;

  let html = '';

  if (!filter && recentNames.length > 0) {
    // Show recent models that exist in available list
    const availableNames = new Set(models.map(m => m.name || m.model || ''));
    const recentAvail = recentNames.filter(n => availableNames.has(n)).slice(0, 5);
    if (recentAvail.length > 0) {
      html += '<div class="voice-selector-recent-header">Recently Used</div>';
      for (const name of recentAvail) {
        const active = name === current ? ' active' : '';
        html += `<button class="voice-selector-item${active}" data-value="${escapeHtml(name)}">
          <span class="voice-selector-item-name">${escapeHtml(name)}</span>
        </button>`;
      }
      html += '<div class="voice-selector-divider"></div>';
    }
  }

  if (filtered.length === 0) {
    html += '<div class="voice-selector-empty">No models found</div>';
  } else {
    for (const m of filtered) {
      const name = m.name || m.model || '';
      if (!name) continue;
      const active = name === current ? ' active' : '';
      html += `<button class="voice-selector-item${active}" data-value="${escapeHtml(name)}">
        <span class="voice-selector-item-name">${escapeHtml(name)}</span>
      </button>`;
    }
  }

  list.innerHTML = html;

  // Wire click handlers
  for (const item of list.querySelectorAll('.voice-selector-item')) {
    item.addEventListener('click', async () => {
      const value = item.dataset.value;
      try {
        const activated = await activateChatModelByName(value, {
          addRecent: true,
          promptForMissingEngineProfile: false,
        });
        if (!activated) return;
      } catch (err) {
        showToast(err.message || 'Could not switch to that model', 'error');
        return;
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config', model: value }));
      }
      const label = document.getElementById('voice-pill-model-label');
      if (label) label.textContent = value || 'default';

      _closeAllPanels();
    });
  }
}

function _renderVoicePanel(filter) {
  const panel = _voiceVoicePanel;
  if (!panel) return;
  const list = panel.querySelector('.voice-selector-list');
  if (!list) return;

  const voices = _voiceCachedVoices || [];
  const settings = safeParseJSON(localStorage.getItem('augmentum_settings'), {});
  const activeSession = chat.getActiveSession();
  const currentVoice = activeSession?.characterVoice || settings.voiceDefaultVoice || '';

  // "Provider default" option
  let html = '';
  const defaultActive = !currentVoice ? ' active' : '';

  if (!filter || 'provider default'.includes(filter)) {
    html += `<button class="voice-selector-item${defaultActive}" data-value="">
      <span class="voice-selector-item-name">Provider default</span>
    </button>`;
  }

  // Separate recommended blends, recommended voices, and the rest
  const recommendedBlends = [];
  const recommendedVoices = [];
  const otherByProvider = {};

  for (const v of voices) {
    const matchesFilter = !filter || (v.name || v.id || '').toLowerCase().includes(filter)
      || (v.description || '').toLowerCase().includes(filter);
    if (!matchesFilter) continue;

    if (v.is_mix && v.recommended) {
      recommendedBlends.push(v);
    } else if (v.recommended && v.provider_id === 'kokoro-builtin') {
      recommendedVoices.push(v);
    } else {
      const prov = v.provider_name || v.provider_id || 'TTS';
      if (!otherByProvider[prov]) otherByProvider[prov] = [];
      otherByProvider[prov].push(v);
    }
  }

  // Helper to render a single voice item
  const _renderItem = (v) => {
    const id = v.provider_id ? `${v.provider_id}::${v.id || v.name}` : (v.id || v.name || '');
    const name = v.name || v.id || '';
    const active = id === currentVoice ? ' active' : '';
    const grade = v.grade ? `<span class="voice-grade voice-grade-${escapeHtml(v.grade.replace(/[^A-Za-z]/g, ''))}">${escapeHtml(v.grade)}</span>` : '';
    const desc = v.description ? `<span class="voice-desc">${escapeHtml(v.description)}</span>` : '';
    const gender = v.gender ? `<span class="voice-gender">${v.gender === 'F' ? '\u2640' : v.gender === 'M' ? '\u2642' : '\u26A5'}</span>` : '';
    const mixBadge = v.is_mix ? '<span class="voice-mix-badge">blend</span>' : '';
    // Fabric peer badge \u2014 small icon for peer-only voices, kept for
    // continuity with prior UX. Shared-source voices (local + N peers)
    // get the new "\u2022 N" badge via voiceBadgeRich; peer-only voices
    // still use the icon badge so the call panel stays glanceable.
    const peer = v.augmentum_peer || null;
    const peerBadge = (peer && peer.icon)
      ? `<span class="voice-peer-badge" title="Served by ${escapeHtml(peer.hostname || 'a fabric peer')}">${escapeHtml(peer.icon)}</span>`
      : '';
    // Shared-source badge only \u2014 peer-only handled by peerBadge above.
    const sharedBadge = (peerSourceCount(v) > 0 && !peer) ? voiceBadgeRich(v) : '';
    return `<button class="voice-selector-item${active}" data-value="${escapeHtml(id)}">
      <span class="voice-selector-item-top">
        ${gender}<span class="voice-selector-item-name">${escapeHtml(name)}</span>${grade}${mixBadge}${peerBadge}${sharedBadge}
      </span>
      ${desc}
      <span class="voice-preview-btn" data-provider="${escapeHtml(v.provider_id || '')}" data-voice="${escapeHtml(v.id || v.name || '')}" title="Preview voice">
        <svg viewBox="0 0 24 24" fill="currentColor" width="12" height="12"><path d="M8 5v14l11-7z"/></svg>
      </span>
    </button>`;
  };

  // Recommended blends first
  if (recommendedBlends.length > 0) {
    html += '<div class="voice-selector-group">Recommended Blends</div>';
    for (const v of recommendedBlends) html += _renderItem(v);
  }

  // Recommended individual voices
  if (recommendedVoices.length > 0) {
    html += '<div class="voice-selector-group">Recommended Voices</div>';
    for (const v of recommendedVoices) html += _renderItem(v);
  }

  // All other voices grouped by provider
  for (const [provName, provVoices] of Object.entries(otherByProvider)) {
    html += `<div class="voice-selector-group">${escapeHtml(provName)}</div>`;
    for (const v of provVoices) html += _renderItem(v);
  }

  if (!html || (filter && html.trim() === '')) {
    html = '<div class="voice-selector-empty">No voices found</div>';
  }

  list.innerHTML = html;

  // Voice preview — play a short sample
  for (const previewBtn of list.querySelectorAll('.voice-preview-btn')) {
    previewBtn.addEventListener('click', async (e) => {
      e.stopPropagation(); // Don't select the voice, just preview
      const providerId = previewBtn.dataset.provider;
      const voiceId = previewBtn.dataset.voice;

      // Toggle — if already playing this preview, stop it
      if (_previewAudio && !_previewAudio.paused && previewBtn.classList.contains('playing')) {
        _previewAudio.pause();
        _previewAudio = null;
        previewBtn.classList.remove('playing');
        return;
      }

      // Stop any other preview
      if (_previewAudio) {
        _previewAudio.pause();
        _previewAudio = null;
      }
      list.querySelectorAll('.voice-preview-btn.playing').forEach(b => b.classList.remove('playing'));

      previewBtn.classList.add('playing');

      try {
        const params = new URLSearchParams();
        if (providerId) params.set('provider_id', providerId);
        if (voiceId) params.set('voice', voiceId);

        const resp = await fetch(`/api/audio/voices/preview?${params}`, { method: 'POST' });
        if (!resp.ok) throw new Error('Preview failed');

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        _previewAudio = new Audio(url);
        _previewAudio.volume = 0.7;
        _previewAudio.onended = () => {
          previewBtn.classList.remove('playing');
          URL.revokeObjectURL(url);
          _previewAudio = null;
        };
        _previewAudio.onerror = () => {
          previewBtn.classList.remove('playing');
          URL.revokeObjectURL(url);
          _previewAudio = null;
        };
        await _previewAudio.play();
      } catch {
        previewBtn.classList.remove('playing');
      }
    });
  }

  // Wire click handlers
  for (const item of list.querySelectorAll('.voice-selector-item')) {
    item.addEventListener('click', () => {
      const value = item.dataset.value;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config', voice: value }));
      }
      const label = document.getElementById('voice-pill-voice-label');
      if (label) label.textContent = _resolveVoiceDisplayName(value) || 'default';

      // In narrative mode with a character voice, this is a temporary
      // override for this call only — don't persist to the character card.
      // In other modes, save as the global default.
      const activeSession = chat.getActiveSession();
      if (!(activeSession && activeSession.mode === 'narrative' && activeSession.characterVoice)) {
        const s = getSettings();
        s.voiceDefaultVoice = value;
        saveSettings();
        syncVoicePrefsToBackend();
      }

      _closeAllPanels();
    });
  }
}

async function _fetchAndCacheSelectors() {
  // Use centralized cache — instant if already warmed at app startup
  _voiceCachedModels = await getModels();
  _voiceCachedVoices = await getVoices();

  // Set initial pill labels
  const modelLabel = document.getElementById('voice-pill-model-label');
  if (modelLabel) modelLabel.textContent = app.state.currentModel || 'default';

  const settings = safeParseJSON(localStorage.getItem('augmentum_settings'), {});
  const activeSession = chat.getActiveSession();
  const savedVoice = activeSession?.characterVoice || settings.voiceDefaultVoice || '';
  const voiceLabel = document.getElementById('voice-pill-voice-label');
  if (voiceLabel) voiceLabel.textContent = _resolveVoiceDisplayName(savedVoice) || 'default';
}

// ---------------------------------------------------------------------------
// Stage Manager — format actions/dialogue before sending (narrative mode)
// ---------------------------------------------------------------------------
let stageActive = false;
let stageSentPending = false; // True after stage Send until turn completes — allows LLM responses through
let _stageCooldown = false;   // true briefly after Send to block stale STT transcripts
let _stageNextMode = '';      // '' | 'action' | 'dialogue'
let stageEl = null;
let stageTextEl = null;
let stageChunks = [];  // undo history
let pinActive = true;  // pin defaults on (button starts with .active class)
let pinEl = null;       // the pinned context display element
let lastAiText = '';    // last AI response (from voice or chat history)
let _preComposingState = 'listening'; // state before composing, to restore on stage deactivate
let _preXrInputMode = null;
let _preXrStageActive = null;

function _setStageActive(active, { focus = false, savePrefs = true } = {}) {
  const toggle = document.getElementById('voice-stage-toggle');
  if (!toggle || !stageEl || !stageTextEl) return false;
  const next = !!active;
  if (stageActive === next) return true;

  stageActive = next;
  stageEl.hidden = !stageActive;
  toggle.classList.toggle('active', stageActive);
  const pinBtn = document.getElementById('voice-pin-transcript');
  if (pinBtn) pinBtn.classList.add('hidden');
  if (pinEl) pinEl.hidden = true;
  if (!stageActive) {
    stageSentPending = false;
    _stageCooldown = false;
    _stageNextMode = '';
    document.getElementById('voice-stage-action')?.classList.remove('active');
    document.getElementById('voice-stage-dialogue')?.classList.remove('active');
  }
  if (stageActive) {
    _updatePin();
    if (focus) stageTextEl.focus();
    setState('composing');
  } else {
    setState(_preComposingState || 'listening');
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'config',
      input_mode: stageActive ? 'staging' : inputMode,
    }));
  }
  if (savePrefs) _saveVoicePrefs();
  return true;
}

function _applyXrVoiceMode(active) {
  if (active) {
    if (_preXrInputMode == null) _preXrInputMode = inputMode;
    if (_preXrStageActive == null) _preXrStageActive = stageActive;
    if (isRecording && inputMode === 'ptt') stopPtt();
    if (stageActive) _setStageActive(false);
    if (inputMode !== 'auto') setInputMode('auto');
    return;
  }
  if (_preXrInputMode && inputMode !== _preXrInputMode) {
    setInputMode(_preXrInputMode);
  }
  if (_preXrStageActive) {
    _setStageActive(true, { focus: false });
  }
  _preXrInputMode = null;
  _preXrStageActive = null;
}

function _initStage() {
  const toggle = document.getElementById('voice-stage-toggle');
  stageEl = document.getElementById('voice-stage');
  stageTextEl = document.getElementById('voice-stage-text');
  if (!toggle || !stageEl || !stageTextEl) return;

  // Create pinned context element inside stage panel (above the editable area)
  pinEl = document.createElement('div');
  pinEl.className = 'voice-stage-pin';
  pinEl.hidden = true;
  stageEl.insertBefore(pinEl, stageTextEl);

  toggle.addEventListener('click', () => {
    _setStageActive(!stageActive, { focus: true });
  });

  // Pin button — toggle context display inside stage
  const pinBtn = document.getElementById('voice-pin-transcript');
  if (pinBtn) {
    pinBtn.addEventListener('click', () => {
      pinActive = !pinActive;
      pinBtn.classList.toggle('active', pinActive);
      _updatePin();
    });
  }

  // --- Mode-based Action/Dialogue buttons ---
  // Two behaviors:
  // 1. If there's a selection → wrap just the selection
  // 2. If no selection → toggle "mode" so the NEXT spoken chunk auto-wraps
  // 3. If text exists and no selection → wrap the last appended chunk
  //
  // Active mode is shown by button staying highlighted. Speaking while
  // a mode is active auto-wraps the transcript and clears the mode.

  const actionBtn = document.getElementById('voice-stage-action');
  const dialogueBtn = document.getElementById('voice-stage-dialogue');

  actionBtn?.addEventListener('click', () => {
    const sel = window.getSelection();
    const hasSelection = sel && sel.rangeCount > 0 && sel.toString().trim()
      && stageTextEl.contains(sel.anchorNode);

    if (hasSelection) {
      // Wrap selection with *asterisks* (strips existing wrapping first)
      _wrapSelection('*', '*');
      // Activate mode so next spoken chunk also auto-wraps
      _stageNextMode = 'action';
      actionBtn.classList.add('active');
      dialogueBtn?.classList.remove('active');
    } else if (stageChunks.length > 0 && _stageNextMode !== 'action') {
      // Swap wrapping on last chunk — strip existing, apply *asterisks*
      const last = stageChunks[stageChunks.length - 1];
      const stripped = _stripWrap(last);
      stageChunks[stageChunks.length - 1] = '*' + stripped + '*';
      _rebuildStageText();
    } else {
      // Toggle mode — next spoken chunk will be auto-wrapped
      _stageNextMode = _stageNextMode === 'action' ? '' : 'action';
      actionBtn.classList.toggle('active', _stageNextMode === 'action');
      dialogueBtn?.classList.remove('active');
    }
  });

  dialogueBtn?.addEventListener('click', () => {
    const sel = window.getSelection();
    const hasSelection = sel && sel.rangeCount > 0 && sel.toString().trim()
      && stageTextEl.contains(sel.anchorNode);

    if (hasSelection) {
      // Wrap selection with "quotes" (strips existing wrapping first)
      _wrapSelection('"', '"');
      // Activate mode so next spoken chunk also auto-wraps
      _stageNextMode = 'dialogue';
      dialogueBtn.classList.add('active');
      actionBtn?.classList.remove('active');
    } else if (stageChunks.length > 0 && _stageNextMode !== 'dialogue') {
      // Swap wrapping on last chunk — strip existing, apply "quotes"
      const last = stageChunks[stageChunks.length - 1];
      const stripped = _stripWrap(last);
      stageChunks[stageChunks.length - 1] = '"' + stripped + '"';
      _rebuildStageText();
    } else {
      // Toggle mode — next spoken chunk will be auto-wrapped
      _stageNextMode = _stageNextMode === 'dialogue' ? '' : 'dialogue';
      dialogueBtn.classList.toggle('active', _stageNextMode === 'dialogue');
      actionBtn?.classList.remove('active');
    }
  });

  // Undo — remove last appended chunk
  document.getElementById('voice-stage-undo')?.addEventListener('click', () => {
    if (stageChunks.length > 0) {
      stageChunks.pop();
      _rebuildStageText();
    }
  });

  // Clear
  document.getElementById('voice-stage-clear')?.addEventListener('click', () => {
    stageChunks = [];
    stageTextEl.textContent = '';
    _stageNextMode = '';
    actionBtn?.classList.remove('active');
    dialogueBtn?.classList.remove('active');
  });

  // Stage text input — sync chunks from DOM and forward typing to avatar awareness
  stageTextEl.addEventListener('input', () => {
    // Sync chunks with manual edits — treat entire text as one chunk
    const currentText = (stageTextEl.textContent || '').trim();
    if (currentText) {
      stageChunks = [currentText];
    } else {
      stageChunks = [];
    }
    if (avatarModule?.avatarState?.active) {
      avatarModule.onStateChange('user_typing');
    }
    // Show pin as reference when user starts typing
    if (pinEl?.hidden && pinActive && lastAiText) {
      _updatePin();
    }
  });

  // Send — dispatch the staged (user-edited) text to the server for LLM processing
  document.getElementById('voice-stage-send')?.addEventListener('click', () => {
    const text = (stageTextEl.textContent || stageTextEl.innerText || '').trim();
    if (!text) {
      // Visual feedback: briefly flash the placeholder
      stageTextEl.classList.add('shake');
      setTimeout(() => stageTextEl.classList.remove('shake'), 400);
      return;
    }
    stageSentPending = true;
    _stageCooldown = true; // Block incoming transcripts from reinserting
    // Stop any playing/queued audio from the previous response
    stopPlayback();
    // Send the edited text to the server — it will process through LLM
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'stage_send',
        text,
        xr_user_context: _drainPendingXrUserSignals(),
      }));
    }
    _addUserBubble(text);
    addMessageToChat('user', text);
    // Show sent text briefly in user transcript, then clear for next turn
    if (transcriptUser) {
      transcriptUser.textContent = text;
      setTimeout(() => {
        if (transcriptUser.textContent === text) {
          transcriptUser.textContent = '';
        }
      }, 2000);
    }
    // Clear stage for next message
    stageChunks = [];
    stageTextEl.textContent = '';
    _stageNextMode = '';
    document.getElementById('voice-stage-action')?.classList.remove('active');
    document.getElementById('voice-stage-dialogue')?.classList.remove('active');
    stageTextEl.focus();
    // Release cooldown after a short delay (covers any in-flight STT transcripts)
    setTimeout(() => { _stageCooldown = false; }, 1500);
  });

  // Seed lastAiText from chat history so the pin has context on first activation
  _seedLastAiFromChat();
}

/** Append a transcript chunk to the staging area (called instead of immediate send).
 *  If an action/dialogue mode is active, auto-wraps the chunk and clears the mode. */
function _stageAppend(text) {
  // Auto-wrap based on active mode
  let wrapped = text;
  if (_stageNextMode === 'action') {
    wrapped = '*' + text + '*';
    _stageNextMode = '';
    document.getElementById('voice-stage-action')?.classList.remove('active');
  } else if (_stageNextMode === 'dialogue') {
    wrapped = '"' + text + '"';
    _stageNextMode = '';
    document.getElementById('voice-stage-dialogue')?.classList.remove('active');
  }

  stageChunks.push(wrapped);
  _rebuildStageText();
  stageTextEl.focus();
  _placeCursorAtEnd();
  // Show pin as reference now that user is composing
  _updatePin();
}

/** Rebuild the stage text from chunks (single source of truth). */
function _rebuildStageText() {
  if (!stageTextEl) return;
  stageTextEl.textContent = stageChunks.join(' ');
  _placeCursorAtEnd();
}

/** Place cursor at end of contenteditable. */
function _placeCursorAtEnd() {
  if (!stageTextEl || !stageTextEl.childNodes.length) return;
  try {
    const range = document.createRange();
    range.selectNodeContents(stageTextEl);
    range.collapse(false);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  } catch { /* ignore on mobile */ }
}

/** Strip existing action/dialogue wrapping from text. */
function _stripWrap(text) {
  let s = text;
  // Strip outer quotes
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith('\u201c') && s.endsWith('\u201d'))) {
    s = s.slice(1, -1);
  }
  // Strip outer asterisks
  if (s.startsWith('*') && s.endsWith('*')) {
    s = s.slice(1, -1);
  }
  return s;
}

/** Wrap the current text selection with prefix/suffix.
 *  Strips existing wrapping first so action↔dialogue swaps cleanly. */
function _wrapSelection(prefix, suffix) {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  const selectedText = range.toString();
  if (!selectedText.trim()) return;

  // Strip existing wrapping before applying new one
  const stripped = _stripWrap(selectedText);
  const wrapped = prefix + stripped + suffix;

  // Replace selection with wrapped version
  range.deleteContents();
  range.insertNode(document.createTextNode(wrapped));
  sel.collapseToEnd();

  // Sync chunks with the actual contenteditable text
  const currentText = (stageTextEl.textContent || '').trim();
  stageChunks = currentText ? [currentText] : [];
}

/** Update the pinned context display with the last AI message.
 *  When the pin is visible, clear the AI transcript to avoid showing
 *  the same response twice. */
function _updatePin() {
  if (!pinEl) return;
  if (!pinActive || !stageActive || !lastAiText) {
    pinEl.hidden = true;
    return;
  }
  // Show the last AI response as ghost reference text.
  // Do NOT clear transcriptAi — the user may be reading the streaming response.
  // The pin is supplementary context, not a replacement for the transcript.
  pinEl.textContent = lastAiText.trim();
  pinEl.hidden = false;
}

/** Hide pin when new AI text is streaming (avoid showing old + new simultaneously). */
function _hidePinDuringStream() {
  if (pinEl && !pinEl.hidden) {
    pinEl.hidden = true;
  }
}

/** Seed lastAiText from the chat session history (for call start context). */
function _seedLastAiFromChat() {
  const session = chat.getActiveSession();
  if (!session) return;
  // Walk the tree from activeLeafId backwards to find the last assistant message
  const tree = session.tree;
  let nodeId = session.activeLeafId;
  while (nodeId && tree[nodeId]) {
    const node = tree[nodeId];
    if (node.role === 'assistant' && node.content) {
      lastAiText = node.content;
      return;
    }
    nodeId = node.parentId;
  }
}

// ---------------------------------------------------------------------------
// Connection Management
// ---------------------------------------------------------------------------
function toggleVoiceCall() {
  if (isConnected) {
    endVoiceCall();
  } else {
    startVoiceCall();
  }
}

function _friendlyVrError(err) {
  const code = err?.code || err?.name || '';
  if (code === 'immersive-vr-unsupported' || code === 'NotSupportedError') {
    return 'Open Voice VR in a WebXR-capable headset browser over HTTPS.';
  }
  if (code === 'immersive-ar-unsupported') {
    return 'Mixed reality is not available in this headset browser. Use VR instead.';
  }
  if (code === 'SecurityError') {
    return 'Voice VR needs HTTPS and headset permission to start.';
  }
  if (code === 'NotAllowedError') {
    return 'Headset permission was not granted.';
  }
  if (code === 'renderer-not-xr-enabled') {
    return 'The avatar renderer is not ready for VR yet.';
  }
  return err?.message || 'Could not enter VR.';
}

async function _enterVoiceXR(mode = 'vr', triggerBtn = null) {
  const wantsMR = mode === 'mr';
  const supported = wantsMR
    ? await avatarXR.isMRSupported()
    : await avatarXR.isXRSupported();
  if (!supported) {
    showToast(
      wantsMR
        ? 'Mixed reality is not available in this headset browser.'
        : 'Open Voice VR in a WebXR-capable headset browser over HTTPS.',
      'error',
      4500,
    );
    return false;
  }

  const state = avatarModule.avatarState;
  if (!state?.active || !state.scene || !state.renderer || !state.camera) {
    console.warn('[voice] VR enter ignored - avatar not active');
    showToast('Turn on avatar mode before entering VR.', 'info', 3500);
    return false;
  }
  if (state.mode !== 'vrm' || !state.vrm?.scene) {
    console.warn('[voice] VR enter ignored - VRM avatar not ready');
    showToast('Wait for the VRM avatar to appear before entering VR.', 'info', 3500);
    return false;
  }

  if (avatarXR.isInVR()) {
    await avatarXR.exitVR();
    await new Promise((resolve) => setTimeout(resolve, 180));
  }

  // VR is solo-only for now - group call has a second VRM in a separate
  // render stack which the XR rig is not aware of yet.
  if (state.secondaryVrm) {
    console.warn('[voice] VR enter ignored - disabled in group calls');
    showToast('Voice VR is currently available for solo avatar calls.', 'info', 3500);
    return false;
  }

  if (triggerBtn) triggerBtn.disabled = true;
  try {
    const enter = wantsMR ? avatarXR.enterMR : avatarXR.enterVR;
    const ok = await enter({
      scene: state.scene,
      renderer: state.renderer,
      camera: state.camera,
      vrm: state.vrm,
      voiceSessionId: _callSessionId || app.state.currentSessionId || '',
      onError: (err) => {
        console.error('[voice] VR enter failed:', err?.message || err);
        showToast(_friendlyVrError(err), 'error', 4500);
      },
    });
    if (!ok && !avatarXR.isInVR()) {
      console.warn('[voice] VR did not start');
    }
    return ok;
  } finally {
    if (triggerBtn) triggerBtn.disabled = false;
  }
}

async function _enterVoiceVR(vrBtn = null) {
  return _enterVoiceXR('vr', vrBtn);
}

function _handleXrModeSwitchRequest(event) {
  const mode = String(event?.detail?.mode || 'vr').trim() === 'mr' ? 'mr' : 'vr';
  setTimeout(() => {
    _enterVoiceXR(mode).catch((err) => {
      console.error('[voice] XR mode switch failed:', err);
      showToast(_friendlyVrError(err), 'error', 4500);
    });
  }, 180);
}

function _handleXrSessionState(event) {
  _applyXrVoiceMode(!!event?.detail?.active);
}

function _sanitizeXrUserSignal(detail = {}) {
  const type = String(detail.type || '').trim().slice(0, 80);
  if (!type) return null;
  return {
    type,
    summary: String(detail.summary || '').replace(/\s+/g, ' ').trim().slice(0, 240),
    at: String(detail.at || new Date().toISOString()).slice(0, 48),
    mode: String(detail.mode || '').slice(0, 16),
    confidence: Math.max(0, Math.min(1, Number(detail.confidence || 0))),
    zone: String(detail.zone || '').slice(0, 40),
    hand: Number.isFinite(Number(detail.hand)) ? Number(detail.hand) : null,
    distance_m: Number.isFinite(Number(detail.distance_m)) ? Number(detail.distance_m) : null,
  };
}

function _sendXrUserSignalToVoice(signal) {
  if (!signal) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'xr_user_signal', signal }));
  }
}

function _handleXrUserSignal(event) {
  const signal = _sanitizeXrUserSignal(event?.detail || {});
  if (!signal) return;
  _pendingXrUserSignals.push(signal);
  _pendingXrUserSignals = _pendingXrUserSignals.slice(-8);
  _sendXrUserSignalToVoice(signal);
}

function _drainPendingXrUserSignals() {
  const out = _pendingXrUserSignals.slice(-6);
  _pendingXrUserSignals = [];
  return out;
}

async function openVrEntry() {
  const supported = await avatarXR.isXRSupported();
  if (!supported) {
    showToast('Open Voice VR in a WebXR-capable headset browser over HTTPS.', 'error', 4500);
    return false;
  }
  if (!isConnected) {
    showToast('Starting Voice VR. Tap Enter VR once the avatar appears.', 'info', 4500);
    await startVoiceCall();
    return false;
  }
  const vrBtn = document.getElementById('voice-vr-btn');
  if (vrBtn) {
    vrBtn.hidden = false;
    try { vrBtn.focus({ preventScroll: true }); } catch { /* ignore */ }
  }
  showToast('Tap Enter VR once the avatar is visible.', 'info', 3500);
  return true;
}

function _handleXrSurfaceRequest(event) {
  const action = String(event?.detail?.action || '').trim();
  if (!action) return;
  const panelAction = String(
    event?.detail?.primaryAction || event?.detail?.panelAction || ''
  ).trim();

  const modeBySurface = {
    chat: 'passthrough',
    analytical: 'analytical',
    agentic: 'agentic',
    coder: 'coder',
    narrative: 'narrative',
  };
  const needsTools = new Set([
    'browse',
    'files',
    'coder',
    'notes',
    'studio',
    'media',
    'devices',
    'games',
  ]).has(action);
  if (needsTools) {
    voiceToolsEnabled = true;
    const toolsToggle = document.getElementById('voice-tools-toggle');
    if (toolsToggle) toolsToggle.classList.add('active');
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    const cfg = {
      type: 'config',
      xr_surface: action,
      xr_panel_action: panelAction,
    };
    if (modeBySurface[action]) cfg.mode = modeBySurface[action];
    if (needsTools) cfg.tools = ['all'];
    ws.send(JSON.stringify(cfg));
  }

  const label = event?.detail?.label || action;
  const actionLabel = event?.detail?.primaryActionLabel || panelAction;
  showToast(
    actionLabel ? `Voice routed to ${label}: ${actionLabel}` : `Voice routed to ${label}`,
    'info',
    2500,
  );
}

async function startVoiceCall() {
  // Clean up any orphaned connection from a previous call
  if (ws) {
    try { ws.close(); } catch { /* ignore */ }
    ws = null;
  }

  // Hide the Becca companion widget for the duration of the call. It
  // overlaps the call surface and its standalone VRM would fight the
  // call's own avatar for GL slots. ``__beccaReactivateVRM`` restores
  // both visibility and the VRM pipeline when the call ends.
  try { window.__beccaHideForCall?.(); } catch (_) { /* widget gone or pre-mount */ }

  // Secure context check — getUserMedia requires HTTPS on mobile browsers
  if (!window.isSecureContext) {
    showToast('Voice requires a secure (HTTPS) connection', 'error');
    return;
  }

  // Request mic access BEFORE showing overlay — on mobile the permission
  // dialog appears behind the fullscreen overlay and can't be tapped.
  // acquireMic threads the user's chosen deviceId from settings, applies
  // per-device constraint heuristics (BT disables AGC, USB gaming mics
  // disable NS, etc.), and falls back through device/constraint errors.
  try {
    micStream = await acquireMic({ usage: 'streaming' });
  } catch (fallbackErr) {
    const reason = fallbackErr.name === 'NotFoundError' ? 'No microphone found'
      : fallbackErr.name === 'NotAllowedError' ? 'Microphone permission denied — check browser site settings'
      : fallbackErr.name === 'NotReadableError' ? 'Microphone in use by another app'
      : `Microphone error: ${fallbackErr.name || fallbackErr.message}`;
    showToast(reason, 'error');
    console.error('[Voice] acquireMic failed:', fallbackErr.name, fallbackErr.message);
    return;
  }
  // Surface the active mic label in the overlay so the user can verify
  // Chrome honored their pick rather than silently falling back to default.
  // Closes the diagnostic loop the prior "no device picker" silence opened.
  const micLabelEl = document.getElementById('voice-mic-label');
  if (micLabelEl) {
    const label = streamMicLabel(micStream) || 'Default microphone';
    micLabelEl.textContent = label;
  }

  // Show overlay now that mic is granted
  overlay.classList.add('active');
  setState('connecting');

  // Register with ViewStack so hang-up tears down cleanly and whatever was
  // underneath (chat / coder / narrative) is re-focused on exit. Without this
  // the screen went blank if the previous surface wasn't auto-restored.
  _voiceTornDown = false;
  ViewStack.pushOverlay('voice', {
    // Side-channel: the call is mode-agnostic. Without sticky, switching
    // story→chat or chat→analyze through orb-nav would pop the overlay and
    // tear the call down — surprising when the pet pill is happily docked.
    // Hang-up still pops normally via endVoiceCall().
    sticky: true,
    onClose: () => _teardownVoiceCall(),
    restoreFocus: () => {
      // After voice ends, surface a predictable chat-input focus. On narrative
      // or coder we skip — those modes have their own focus convention.
      const mode = app.state.mode;
      if (mode === 'passthrough' || mode === 'analytical' || mode === 'agentic') {
        const input = document.getElementById('chat-input');
        if (input && input.offsetParent !== null) {
          try { input.focus({ preventScroll: true }); } catch { /* ignore */ }
        }
      }
    },
  });

  // Start starfield background
  _initStarfield();

  // Populate model and voice selector pills
  _fetchAndCacheSelectors();
  _resetDimTimer();

  // Seed last AI message from chat history for pin context
  _seedLastAiFromChat();

  // Set up audio analysis
  audioContext = new AudioContext();
  // Ensure AudioContext is running (browsers may auto-suspend without user gesture).
  // Mobile browsers may defer resumption even after await — retry and warn.
  if (audioContext.state === 'suspended') {
    try {
      await audioContext.resume();
    } catch (e) {
      console.warn('[Voice] AudioContext resume failed:', e);
    }
  }
  // If still not running after resume attempt, warn but continue
  if (audioContext.state !== 'running') {
    console.warn('[Voice] AudioContext state after resume:', audioContext.state);
  }
  // Prime with a silent buffer inside this user-gesture call. iOS treats
  // subsequent audio through this context as gesture-authorized, even if
  // the context briefly suspends (lock screen, tab switch back).
  try {
    const silentSrc = audioContext.createBufferSource();
    silentSrc.buffer = audioContext.createBuffer(1, 1, audioContext.sampleRate);
    silentSrc.connect(audioContext.destination);
    silentSrc.start(0);
  } catch { /* best-effort prime */ }
  // Audio graph:
  //   Mic → micAnalyser (orb viz, dead-end — no speaker output)
  //   TTS → ttsGainNode → ttsAnalyser → destination (speakers + lip sync FFT)
  //
  // Two separate analysers: one for mic (orb), one for TTS (lip sync).
  // This prevents mic audio from leaking to speakers via the analyser.

  analyserNode = audioContext.createAnalyser();       // TTS analyser (lip sync + glow)
  analyserNode.fftSize = 2048;
  analyserNode.smoothingTimeConstant = 0.85;

  // Mic source — for orb visualization and PCM streaming to server
  micSourceNode = audioContext.createMediaStreamSource(micStream);
  // Mic connects to its own analyser for orb viz (when avatar is off)
  // This is a dead-end node — mic audio does NOT reach speakers
  const micAnalyser = audioContext.createAnalyser();
  micAnalyser.fftSize = 2048;
  micAnalyser.smoothingTimeConstant = 0.85;
  micSourceNode.connect(micAnalyser);
  // Store mic analyser for orb (swap which analyser the orb reads based on mode)
  analyserNode._micAnalyser = micAnalyser;

  // [4] GainNode for TTS ducking — route through TTS analyser for lip sync
  ttsGainNode = audioContext.createGain();
  ttsGainNode.connect(analyserNode);                 // TTS → analyser
  analyserNode.connect(audioContext.destination);      // analyser → speakers

  // Resize canvas now that overlay is visible
  resizeCanvas();

  // Ensure a chat session exists so voice messages persist to the chat tree
  if (!app.state.currentSessionId) {
    await chat.createSession();
  }

  _seedVoiceTranscriptLogFromChat();

  // Show avatar toggle if enabled in settings
  const avatarToggleBtn = document.getElementById('voice-avatar-toggle');
  if (avatarToggleBtn && getSettings().avatarEnabled) {
    avatarToggleBtn.style.display = '';
  }

  // Connect WebSocket (with retry on initial connection failure)
  const MAX_CONNECT_RETRIES = 2;
  let _connectAttempt = 0;

  const _attemptWsConnect = async () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const sessionId = app.state.currentSessionId || 'voice_default';
    _callSessionId = sessionId;
    const model = (app.state.currentModel && app.state.currentModel !== 'default')
      ? app.state.currentModel
      : (localStorage.getItem('augmentum-selected-model') || '');
    const mode = app.state.mode || 'passthrough';
    // Capture once at connect time — the resolved label belongs to the
    // session the call is bound to, not whatever the user navigates to
    // afterwards. Best-effort; failures leave the badge hidden.
    try { _callHomeLabel = await _resolveCallHomeLabel(chat.getActiveSession(), mode); }
    catch { _callHomeLabel = ''; }
    const ticket = await getWsTicket();
    const wsUrl = `${proto}://${location.host}/ws/voice?ticket=${encodeURIComponent(ticket)}&session_id=${encodeURIComponent(sessionId)}&model=${encodeURIComponent(model)}&mode=${encodeURIComponent(mode)}`;

    try {
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
    } catch (err) {
      overlay.classList.remove('active');
      showToast('Failed to connect voice chat', 'error');
      cleanupMic();
      return;
    }

    setState('connecting');
    _wireWsHandlers();
  };

  // Mid-call reconnect state machine. Drives the user-visible
  // reconnect attempts when either (a) ws.onclose fires unexpectedly
  // or (b) _attemptWsConnect throws before getting a socket open
  // (e.g. ticket fetch returned 401). Without (b), an expired auth
  // session during reconnect would leave the user stuck at
  // "Reconnecting… (1/3)" forever — no socket = no onclose = no
  // forward motion.
  const _handleMidCallDisconnect = (reason) => {
    // Hard auth failure (401 from /api/auth/ws-ticket) — the user's
    // session expired. No amount of retrying will fix it. Surface a
    // distinct, actionable message and tear down the call cleanly.
    if (reason?.status === 401) {
      setState('disconnected');
      showToast('Session expired — please sign in again', 'error');
      endVoiceCall();
      return;
    }

    if (_reconnectAttempt < MAX_RECONNECT_ATTEMPTS) {
      const delay = RECONNECT_BACKOFF_MS[_reconnectAttempt];
      _reconnectAttempt++;
      setState('reconnecting');
      if (statusText) {
        statusText.textContent = `Reconnecting… (${_reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS})`;
      }
      console.warn(
        `[Voice] reconnect ${_reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms`,
        reason?.error?.message || '',
      );
      _reconnectTimer = setTimeout(() => {
        _reconnectTimer = null;
        _attemptWsConnect().catch((err) => {
          // Couldn't construct the socket — typically a ticket-fetch
          // failure. Re-enter the state machine so we either schedule
          // the next attempt with the right backoff, or exhaust and
          // tear down. Without this, the chain stops cold.
          _handleMidCallDisconnect({
            status: err?.status,
            error: err,
          });
        });
      }, delay);
    } else {
      setState('disconnected');
      showToast('Call dropped — could not reconnect', 'error');
      _disconnectTimer = setTimeout(() => {
        if (!isConnected) endVoiceCall();
      }, 30000);
    }
  };

  const _wireWsHandlers = () => {

  ws.onopen = async () => {
    // Reset mid-call reconnect state — we're connected. If we're returning
    // from a successful reconnect, surface a brief "Reconnected" toast so
    // the user knows their call is live again (the prior "Reconnecting…"
    // status would otherwise just silently flip to "Listening").
    const wasReconnecting = _reconnectAttempt > 0;
    _reconnectAttempt = 0;
    if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    if (_disconnectTimer) { clearTimeout(_disconnectTimer); _disconnectTimer = null; }
    if (wasReconnecting) showToast('Voice reconnected', 'info');

    isConnected = true;
    setState('listening');
    startOrb();
    startDurationTimer();
    _startHeartbeat();
    // Ducking is now per-utterance through AudioBus — no call-level duck
    // so Grove stays audible while the user is speaking or idle.

    const callBtn = document.getElementById('voice-call-btn');
    if (callBtn) callBtn.classList.add('in-call');

    // Send initial config — character voice (narrative) or default voice
    const voiceSettings = safeParseJSON(localStorage.getItem('augmentum_settings'), {});
    const activeSession = chat.getActiveSession();
    const characterVoice = activeSession?.characterVoice || '';
    const defaultVoice = voiceSettings.voiceDefaultVoice || '';

    // Include existing chat history so the voice session has full context
    const existingMessages = activeSession
      ? chat.buildMessagesForAPI(activeSession)
      : [];

    ws.send(JSON.stringify({
      type: 'config',
      model: app.state.currentModel || '',
      voice: defaultVoice,
      character_voice: characterVoice,
      speed: parseFloat(voiceSettings.voiceSpeed) || 1.0,
      system_prompt: activeSession?.narrativeSystemPrompt || '',
      messages: existingMessages,
      tools: voiceToolsEnabled ? ['all'] : [],
      input_mode: inputMode,
      xr_user_context: _drainPendingXrUserSignals(),
    }));

    // Announce client-side voice pipeline capabilities. The resolver on
    // the server consults this on every dispatch to decide whether VAD /
    // STT / TTS / denoise should run client-side. We try to load Silero
    // VAD lazily here; if it succeeds, we advertise it AND attach it to
    // the WS so vad_speech_start/end events flow to the server. On
    // failure we fall back to empty caps (server VAD takes over, no
    // user-visible regression).
    const policyMode = voiceSettings.voicePipelineModeCall || 'auto';
    const tryClientVad = (policyMode === 'auto' || policyMode === 'local');
    const caps = { vad: [], stt: [], tts: [], denoise: [] };

    if (tryClientVad && micStream) {
      try {
        const mod = await import('./voice/vad-client.js');
        _clientVad = await mod.attachVadToWebSocket(ws, micStream, {
          onEvent: (evt) => {
            if (evt.kind === 'ready') console.log('[Voice] Silero VAD ready');
            if (evt.kind === 'error') console.warn('[Voice] Silero VAD error', evt.error);
          },
        });
        caps.vad = [mod.VadClient.capabilityId];
      } catch (err) {
        console.warn('[Voice] client VAD unavailable, falling back to server VAD', err);
        if (policyMode === 'local') {
          showToast('Local VAD failed to load — voice features may not work.', 'error');
        }
        _clientVad = null;
      }
    }

    ws.send(JSON.stringify({ type: 'capabilities', ...caps }));

    // If starting in PTT mode, tell the server to gate audio until button is pressed
    if (inputMode === 'ptt') {
      ws.send(JSON.stringify({ type: 'ptt_active', active: false }));
    }

    // NOTE: Server VAD mode is set by the first "listening" message
    // (which includes server_vad flag). Do NOT start VAD monitoring here —
    // wait for the server to tell us which mode to use in handleServerMessage.
    // Starting legacy VAD here races with the server's response and can leave
    // a mediaRecorder running alongside the PCM worklet.

    // Restore per-mode voice preferences
    try {
      const prefMode = app.state.mode || 'passthrough';
      const resp = await fetch(`/api/config/voice-prefs/${prefMode}`);
      if (resp.ok) {
        const prefs = await resp.json();
        if (prefs.stage_active) {
          const stageToggle = document.getElementById('voice-stage-toggle');
          if (stageToggle && !stageActive) stageToggle.click();
        }
        if (prefs.input_mode && prefs.input_mode !== inputMode) {
          setInputMode(prefs.input_mode);
        }
        if (prefs.avatar_active && getSettings().avatarEnabled) {
          const sessionInfo = await _resolveAvatarSessionInfo(
            chat.getActiveSession(),
            prefMode,
          );
          avatarModule.activateAvatar(analyserNode, sessionInfo)
            .then(() => _initPoseTriggerEngine())
            .catch(() => {});
        }
      }
    } catch { /* use defaults */ }
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      handleAudioChunk(event.data);
      return;
    }
    try {
      const msg = JSON.parse(event.data);
      handleServerMessage(msg);
    } catch { /* ignore */ }
  };

  ws.onerror = (ev) => {
    const audioState = audioContext ? audioContext.state : 'no-context';
    const wsState = ws ? ws.readyState : 'no-ws';
    console.error('[Voice] WebSocket error — audioContext:', audioState, 'ws.readyState:', wsState);
    // Retry handled in onclose — don't show toast yet if retries remain
  };

  ws.onclose = (event) => {
    _stopHeartbeat();
    if (isConnected) {
      // Unexpected mid-call disconnect — hand to the reconnect state
      // machine. Users on flaky networks are the primary failure case
      // (mobile data hiccup, WiFi handoff, brief server restart);
      // left alone they stare at "Disconnected" with no signal that
      // recovery is even being attempted.
      isConnected = false;
      if (navigator.vibrate) navigator.vibrate([20, 50, 20]);
      _handleMidCallDisconnect();
    } else if (_connectAttempt < MAX_CONNECT_RETRIES) {
      // Initial connection failed — retry with a fresh ticket
      _connectAttempt++;
      console.warn(`[Voice] Connection failed, retrying (${_connectAttempt}/${MAX_CONNECT_RETRIES})…`);
      if (_initialConnectRetryTimer) clearTimeout(_initialConnectRetryTimer);
      _initialConnectRetryTimer = setTimeout(() => {
        _initialConnectRetryTimer = null;
        if (_voiceTornDown) return;
        _attemptWsConnect().catch((err) => {
          console.warn('[Voice] initial reconnect failed:', err);
          if (!_voiceTornDown) endVoiceCall();
        });
      }, 800 * _connectAttempt);
    } else if (_connectAttempt >= MAX_CONNECT_RETRIES) {
      // Exhausted retries
      showToast('Voice connection error — check console for details', 'error');
      overlay.classList.remove('active');
      const btn = document.getElementById('voice-call-btn');
      if (btn) btn.classList.remove('in-call');
      cleanupMic();
    }
  };

  }; // end _wireWsHandlers

  // Initial connect — distinct from reconnect, no state machine to drive,
  // just surface the failure clearly so the user knows what happened.
  // Hits this path mainly when the user's session cookie expired between
  // page load and clicking the call button.
  try {
    await _attemptWsConnect();
  } catch (err) {
    if (err?.status === 401) {
      showToast('Session expired — please sign in again', 'error');
    } else {
      showToast('Failed to start voice call — check console', 'error');
      console.warn('[Voice] initial connect failed:', err);
    }
    // Tear down the partial setup (overlay, mic, audio context) so the
    // user isn't left looking at a frozen "Connecting" state.
    endVoiceCall();
  }
}

// Public facade — all call sites (UI button, WS disconnect timer, duplicate
// start guard) come through here. Routes through ViewStack so overlay state
// and restore-focus stay consistent. The actual teardown in _teardownVoiceCall
// is idempotent via _voiceTornDown so either path is safe.
let _voiceTornDown = true;  // starts torn down; set false on startVoiceCall
function endVoiceCall() {
  if (_voiceTornDown) return;
  if (ViewStack.hasOverlay('voice')) {
    ViewStack.popOverlay('voice');  // onClose → _teardownVoiceCall
    return;
  }
  _teardownVoiceCall();
}

function _teardownVoiceCall() {
  if (_voiceTornDown) return;
  _voiceTornDown = true;

  // Dispose the pose trigger engine (idle interval + state) before the
  // wider teardown, so its callbacks can't fire against deactivated avatar.
  _disposePoseTriggerEngine();

  // Two-phase teardown:
  //   1. _safe(...) wrappers run all heavy cleanup. Any individual failure
  //      is logged but does NOT prevent the rest from running.
  //   2. _forceVisualReset() runs unconditionally at the end, so the overlay
  //      collapses and the underlying chat surface is restored even if some
  //      step above threw. Without this, a failure mid-cleanup (dead WS,
  //      revoked mic, half-finished minimize anim) used to leave the
  //      overlay's clip-path / classes pinned and the chat UI invisible.
  const _safe = (label, fn) => {
    try { fn(); }
    catch (err) { console.warn('[Voice] teardown step failed:', label, err); }
  };

  // Release the live camera + frame loop before the wider teardown so the
  // webcam light goes out the instant the call ends.
  _safe('camera', _stopCallCamera);

  _safe('callBtn', () => {
    const callBtn = document.getElementById('voice-call-btn');
    if (callBtn) callBtn.classList.remove('in-call');
  });

  _safe('heartbeat', _stopHeartbeat);

  _safe('disconnectTimer', () => {
    if (_disconnectTimer) { clearTimeout(_disconnectTimer); _disconnectTimer = null; }
  });

  _safe('reconnectTimer', () => {
    // Cancel any pending reconnect attempt and reset the counter so a
    // future call starts from a clean slate. Without this, a user who
    // hangs up mid-reconnect would still have a setTimeout firing
    // _attemptWsConnect against a teardown-state module, racing the
    // mic/audio/ws cleanup that just finished.
    if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    _reconnectAttempt = 0;
  });

  _safe('initialConnectRetryTimer', () => {
    if (_initialConnectRetryTimer) {
      clearTimeout(_initialConnectRetryTimer);
      _initialConnectRetryTimer = null;
    }
  });

  _safe('stageFlags', () => {
    stageSentPending = false;
    _stageCooldown = false;
  });

  _safe('flushPendingAssistant', () => {
    if (!currentResponseSaved && currentResponseText && currentResponseText.trim()) {
      addMessageToChat('assistant', currentResponseText.trim());
    }
  });

  _safe('transcriptTimers', () => {
    clearTimeout(_conversingTimer);
    clearTimeout(_userTranscriptDimTimer);
    clearTimeout(_aiTranscriptDimTimer);
  });

  _safe('panels', _closeAllPanels);
  _safe('voiceBus', () => {
    _voiceBusClaim?.release();
    _voiceBusClaim = null;
  });
  _safe('dimTimer', _clearDimTimer);
  _safe('orb', stopOrb);
  _safe('starfield', _stopStarfield);
  _safe('vad', stopVadMonitoring);
  _safe('pcm', stopPcmStreaming);
  _safe('browserStt', _stopBrowserStt);
  _safe('duration', stopDurationTimer);
  _safe('ducking', stopDucking);
  _safe('mic', cleanupMic);
  _safe('playback', stopPlayback);

  _safe('pttFallback', () => {
    if (_pttFallbackTimer) { clearTimeout(_pttFallbackTimer); _pttFallbackTimer = null; }
  });

  _safe('ws', () => {
    if (ws) {
      // Detach handlers BEFORE closing to prevent onclose from retrying
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      try { ws.close(); } catch { /* already closed */ }
      ws = null;
    }
  });

  _safe('audioContext', () => {
    if (!audioContext) return;
    // Disconnect graph paths to `destination` before close. On mobile
    // browsers (iOS Safari especially), close() doesn't reliably
    // release the underlying audio thread when the graph still has
    // active connections — over many call cycles this leaks AudioContext
    // instances and eventually trips the per-page concurrent-context
    // cap, breaking voice mode for the user.
    //
    // Each disconnect is wrapped because most of these nodes were
    // already disconnected by prior teardown steps (stopPcmStreaming,
    // stopVadMonitoring → stopAutoCapture). Disconnecting an already-
    // disconnected node throws, hence per-node try/catch.
    try { ttsGainNode?.disconnect(); } catch { /* already disconnected */ }
    try { analyserNode?.disconnect(); } catch { /* already disconnected */ }
    try { micSourceNode?.disconnect(); } catch { /* already disconnected */ }

    // Capture and null synchronously so any concurrent code that checks
    // `audioContext` sees the teardown immediately, regardless of how
    // long the suspend → close promise chain takes to settle.
    const ctx = audioContext;
    audioContext = null;
    analyserNode = null;
    micSourceNode = null;
    ttsGainNode = null;
    _ttsRouteDest = null;

    // Suspend → close. Suspending first lets the audio thread drain
    // pending samples cleanly, which mobile browsers handle better
    // than a hard close. Both calls are fire-and-forget — settling
    // takes 50-200ms on mobile and we don't want to block the rest
    // of teardown waiting for it.
    ctx.suspend()
      .catch(() => { /* may fail if already closed/suspended */ })
      .then(() => ctx.close())
      .catch((err) => console.warn('[Voice] AudioContext close failed:', err));
  });

  _safe('avatar', () => avatarModule.dispose());

  // Call is over → ask the companion widget (if mounted) to bring its
  // standalone VRM back up. deactivateAvatar() used to do this from its
  // finally block on every deactivate, which churned WebGL contexts on
  // mid-call avatar toggles / switches. Reactivation belongs here — the
  // single point that means "the call has actually ended".
  _safe('beccaReactivate', () => {
    try { window.__beccaReactivateVRM?.(); } catch (_) { /* widget gone */ }
  });

  _safe('avatarToggleBtn', () => {
    const avatarToggleBtn = document.getElementById('voice-avatar-toggle');
    if (avatarToggleBtn) avatarToggleBtn.style.display = 'none';
  });

  _safe('transcripts', () => {
    if (transcriptUser) transcriptUser.textContent = '';
    if (transcriptAi) transcriptAi.textContent = '';
    if (transcriptAi) {
      transcriptAi.querySelectorAll('.voice-image-card, .voice-image-grid, .voice-video-list').forEach(el => el.remove());
    }
    const log = document.querySelector('.voice-transcript-log');
    if (log) log.querySelectorAll('.voice-log-bubble').forEach(el => el.remove());
    if (_aiRenderTimer) {
      clearTimeout(_aiRenderTimer);
      _aiRenderTimer = null;
    }
    _currentAiBubble = null;
    _currentAiText = '';
  });

  // State flags — these are pure assignments, but wrap defensively anyway
  // so a future addition that throws can't block the visual reset.
  _safe('stateFlags', () => {
    isConnected = false;
    isMinimized = false;
    _preMinimizeInputMode = null;
    currentResponseText = '';
    playedSentenceCount = 0;
    queuedSentenceCount = 0;
    vadSpeechProb = 0;
    serverVadActive = false;
    _callSessionId = null;
    _callHomeLabel = '';
  });

  // Reset the badge DOM alongside the in-memory label so a future call
  // doesn't briefly flash the prior call's name before its own resolve
  // completes.
  _safe('pillBadge', _renderPillBadge);

  // Final phase — runs no matter what failed above.
  _forceVisualReset();
}

/** Force the voice overlay back to a hidden, neutral state.
 *
 *  Why this is more aggressive than just removing classes: minimize/expand
 *  use ``Element.animate({ fill: 'forwards' })``. Their references
 *  (``_minimizeAnim``) are nulled in ``onfinish`` for housekeeping, but the
 *  animation **effect** (a clipPath circle + opacity 0 pinned at the pill
 *  position) persists in the browser's animation timeline indefinitely.
 *  Class removal alone leaves the orphaned effect in place — the overlay
 *  appears clipped or opaque even though every class and inline style
 *  says otherwise. This was the cause of the "starfield-only after end
 *  call" stuck-DOM screenshot.
 *
 *  Also wipes the starfield canvas pixels — ``_stopStarfield()`` cancels
 *  the rAF loop, but whatever was painted in the last frame stays drawn
 *  on the canvas surface. With the overlay's opacity fading to 0 over
 *  400ms (and animation-effect leaks above) the residue can be visible
 *  for the entire duration of the user staring at it.
 */
function _forceVisualReset() {
  // 1. Cancel EVERY web animation on the overlay AND its descendants —
  //    `subtree: true` covers _minimizeAnim plus any orphaned
  //    fill:forwards effects on _pillEl, the avatar drawer, the orb,
  //    etc. The default getAnimations() (no subtree) only sees the
  //    overlay element itself, missing animations whose effect persists
  //    on a child even though the overlay's classes look clean.
  if (overlay && typeof overlay.getAnimations === 'function') {
    try {
      const anims = overlay.getAnimations({ subtree: true });
      if (anims.length > 0) {
        // One-line breadcrumb when residual animations are present at
        // teardown. Helpful for diagnosing the next "stuck overlay" report.
        console.warn('[Voice] cancelling residual overlay animations:', anims.length);
      }
      for (const a of anims) {
        try { a.cancel(); } catch { /* already-finished */ }
      }
    } catch { /* getAnimations not supported — fall through */ }
  }
  _minimizeAnim = null;

  _stopPillOrb();
  if (_pillEl) {
    _pillEl.hidden = true;
    // Drop pet-mode layout; deactivateAvatar() owns reparenting/cleanup of
    // the canvas itself and resets avatarState.petMode.
    _pillEl.classList.remove('pet-mode');
    _pillPetMode = false;
    // Pill animations live OUTSIDE the overlay subtree (line 1886's
    // scale-in animation runs on _pillEl which is a sibling of overlay
    // in <body>). Cancel its own animations too, otherwise a fill:forwards
    // effect from a previous expand→pill cycle can leave the pill visible
    // even with hidden=true on some engines.
    if (typeof _pillEl.getAnimations === 'function') {
      try {
        for (const a of _pillEl.getAnimations({ subtree: true })) {
          try { a.cancel(); } catch { /* already-finished */ }
        }
      } catch { /* unsupported */ }
    }
  }

  // 2. Wipe the starfield canvas so the last-painted frame doesn't
  //    survive teardown as a static image plastered over chat.
  if (_starsCanvas && _starsCtx) {
    try {
      _starsCtx.clearRect(0, 0, _starsCanvas.width, _starsCanvas.height);
    } catch { /* canvas detached */ }
  }

  if (overlay) {
    overlay.classList.remove(
      'active', 'minimized', 'minimizing', 'expanding', 'conversing',
      'camera-mode-active',
    );
    overlay.style.removeProperty('clip-path');
    overlay.style.removeProperty('opacity');
    overlay.style.removeProperty('transform');

    const orbWrap = overlay.querySelector('.voice-orb-wrap');
    if (orbWrap) orbWrap.style.visibility = '';

    // 3. Force a one-frame display:none flicker to flush any GPU-cached
    //    layer (composited dark background, residual filter/transform)
    //    that survives a class+style reset. Without this, some engines
    //    (notably WebKit on iOS and some Chromium versions when the
    //    overlay had been promoted to its own composite layer) keep
    //    painting the last frame's --voice-bg-deep until the next
    //    relayout. Restoring display synchronously on the next frame
    //    means the overlay re-mounts in its CSS-default (opacity:0,
    //    pointer-events:none) state — invisible — and the chat surface
    //    underneath gets the paint pass it was missing.
    overlay.style.display = 'none';
    // Force a synchronous layout read so the display:none is applied
    // before we clear it.
    void overlay.offsetHeight;
    requestAnimationFrame(() => {
      // Always restore — a new call started before the rAF still wants
      // the inline `display:none` cleared so the CSS default `flex`
      // takes over. Without this, an immediate end-then-start would
      // leave the new call's overlay invisible.
      overlay.style.removeProperty('display');
    });
  }

  // 4. Belt-and-suspenders avatar viewport reset. avatarModule.dispose()
  //    in the _safe('avatar', …) step above is supposed to handle this,
  //    but deactivateAvatar() runs through a long chain of disposers
  //    (atmosphere, subtitle, drawer, scene materials/textures, presence,
  //    animator, observers…). If any throw, deactivateAvatar's own
  //    finally guarantees the viewport is cleared — but we re-assert
  //    here regardless so a future regression in that path can't leave
  //    the last avatar frame painted on top of chat.
  try {
    const vp = document.getElementById('voice-avatar-viewport');
    if (vp) { vp.style.display = 'none'; vp.innerHTML = ''; }
  } catch { /* viewport already gone */ }
}

function minimizeVoiceCall() {
  if (!isConnected || isMinimized) return;
  isMinimized = true;

  // Save and force auto mode
  _preMinimizeInputMode = inputMode;
  if (inputMode !== 'auto') {
    setInputMode('auto');
  }

  // Close drawer if open
  if (avatarModule.avatarState.drawer) {
    avatarModule.avatarState.drawer.toggle(false);
  }

  _animateMinimize();
}

function expandVoiceCall() {
  if (!isConnected || !isMinimized) return;
  isMinimized = false;

  // Restore input mode
  if (_preMinimizeInputMode && _preMinimizeInputMode !== inputMode) {
    setInputMode(_preMinimizeInputMode);
  }
  _preMinimizeInputMode = null;

  _animateExpand();
}

function _animateMinimize() {
  if (!overlay || !_pillEl) return;

  if (_minimizeAnim) { _minimizeAnim.cancel(); _minimizeAnim = null; }

  // Phase 1: Chrome dissolves
  overlay.classList.add('minimizing');

  // Where the overlay's clip-path collapses to — the pill's resting spot,
  // which the user may have dragged elsewhere.
  const pillRect = _pillPos
    ? { x: _pillPos.x + 35, y: _pillPos.y + 35 }
    : { x: window.innerWidth - 16 - 70, y: window.innerHeight - 16 - 24 };

  // Phase 2+3: After chrome fades, collapse overlay + show pill
  setTimeout(() => {
    const orbWrap = overlay.querySelector('.voice-orb-wrap');
    if (orbWrap) orbWrap.style.visibility = 'hidden';

    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight * 0.45;
    const maxRadius = Math.max(window.innerWidth, window.innerHeight);

    _minimizeAnim = overlay.animate([
      { clipPath: `circle(${maxRadius}px at ${centerX}px ${centerY}px)`, opacity: 1 },
      { clipPath: `circle(${maxRadius * 0.3}px at ${centerX + (pillRect.x - centerX) * 0.3}px ${centerY + (pillRect.y - centerY) * 0.3}px)`, opacity: 0.8, offset: 0.4 },
      { clipPath: `circle(30px at ${pillRect.x}px ${pillRect.y}px)`, opacity: 0 },
    ], {
      duration: 450,
      easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
      fill: 'forwards',
    });

    _minimizeAnim.onfinish = () => {
      _minimizeAnim = null;
      overlay.classList.remove('minimizing', 'active');
      overlay.classList.add('minimized');
      overlay.style.clipPath = '';

      _pillEl.hidden = false;
      _pillEl.setAttribute('data-state', overlay.getAttribute('data-state') || 'listening');

      // If a solo VRM avatar is live, reparent it into the pill as a small
      // desktop-pet instead of drawing the audio orb. The avatar keeps its
      // WebGL context and idle/gesture loop — only its DOM parent + render
      // size change — so it stays warm and keeps fidgeting/dancing on the pill.
      _pillPetMode = false;
      if (_pillAvatarHost && avatarModule.canEnterPetMode?.()) {
        _pillEl.classList.add('pet-mode');
        _pillPetMode = !!avatarModule.enterPetMode(_pillAvatarHost);
        if (!_pillPetMode) _pillEl.classList.remove('pet-mode');
      }
      if (!_pillPetMode) _startPillOrb();
      // Badge is meaningful in both pet-mode and orb-chip mode; render
      // whenever the pill becomes visible. _renderPillBadge() no-ops
      // gracefully when there's no label to show.
      _renderPillBadge();

      // Restore a dragged position (clamped to the current pill size — orb
      // pill and pet frame differ — and viewport).
      _applyPillPos();

      _pillEl.animate([
        { transform: 'scale(0.5) translateY(10px)', opacity: 0 },
        { transform: 'scale(1.06)', opacity: 1, offset: 0.7 },
        { transform: 'scale(1)', opacity: 1 },
      ], { duration: 250, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)' });
    };
  }, 200);
}

function _animateExpand() {
  if (!overlay || !_pillEl) return;

  if (_minimizeAnim) { _minimizeAnim.cancel(); _minimizeAnim = null; }

  // Hand the avatar canvas back to the full viewport before the overlay
  // expands, so there's no flash of an empty viewport.
  if (_pillPetMode) {
    avatarModule.exitPetMode?.();
    _pillPetMode = false;
  }
  _pillEl.classList.remove('pet-mode');
  _stopPillOrb();

  const pillBox = _pillEl.getBoundingClientRect();
  const originX = pillBox.left + pillBox.width / 2;
  const originY = pillBox.top + pillBox.height / 2;
  const maxRadius = Math.max(window.innerWidth, window.innerHeight);

  _pillEl.hidden = true;

  overlay.classList.remove('minimized');
  overlay.classList.add('expanding', 'active');

  const orbWrap = overlay.querySelector('.voice-orb-wrap');
  if (orbWrap) orbWrap.style.visibility = '';

  resizeCanvas();

  const centerX = window.innerWidth / 2;
  const centerY = window.innerHeight * 0.45;

  _minimizeAnim = overlay.animate([
    { clipPath: `circle(30px at ${originX}px ${originY}px)`, opacity: 0.6 },
    { clipPath: `circle(${maxRadius * 0.4}px at ${centerX + (originX - centerX) * 0.5}px ${centerY + (originY - centerY) * 0.5}px)`, opacity: 0.9, offset: 0.5 },
    { clipPath: `circle(${maxRadius}px at ${centerX}px ${centerY}px)`, opacity: 1 },
  ], {
    duration: 450,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    fill: 'forwards',
  });

  _minimizeAnim.onfinish = () => {
    _minimizeAnim = null;
    overlay.classList.remove('expanding');
    overlay.style.clipPath = '';
  };
}

function _startPillOrb() {
  if (!_pillCanvas || !_pillCtx || !analyserNode) return;
  const dpr = window.devicePixelRatio || 1;
  _pillCanvas.width = 36 * dpr;
  _pillCanvas.height = 36 * dpr;

  const orbAnalyser = analyserNode._micAnalyser || analyserNode;
  const frequencyData = new Uint8Array(orbAnalyser.frequencyBinCount);

  function drawPillOrb() {
    _pillOrbAnimId = requestAnimationFrame(drawPillOrb);
    orbAnalyser.getByteFrequencyData(frequencyData);

    const w = _pillCanvas.width;
    const h = _pillCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const baseR = w * 0.35;

    let sum = 0;
    for (let i = 0; i < 64; i++) sum += frequencyData[i];
    const energy = sum / (64 * 255);

    _pillCtx.clearRect(0, 0, w, h);

    const stateColors = {
      listening: [94, 196, 212],
      recording: [224, 144, 112],
      processing: [138, 156, 197],
      speaking: [176, 142, 216],
    };
    const state = overlay?.getAttribute('data-state') || 'listening';
    const [r, g, b] = stateColors[state] || stateColors.listening;

    const grad = _pillCtx.createRadialGradient(cx, cy, baseR * 0.2, cx, cy, baseR * 1.5);
    grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${0.3 + energy * 0.4})`);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    _pillCtx.fillStyle = grad;
    _pillCtx.fillRect(0, 0, w, h);

    const radius = baseR * (0.85 + energy * 0.3);
    _pillCtx.beginPath();
    _pillCtx.arc(cx, cy, radius, 0, Math.PI * 2);
    _pillCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.6 + energy * 0.3})`;
    _pillCtx.fill();

    _pillCtx.beginPath();
    _pillCtx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
    _pillCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.8 + energy * 0.2})`;
    _pillCtx.fill();

    if (_pillEl) _pillEl.setAttribute('data-state', state);
  }

  drawPillOrb();
}

function _stopPillOrb() {
  if (_pillOrbAnimId) {
    cancelAnimationFrame(_pillOrbAnimId);
    _pillOrbAnimId = null;
  }
}

// ---- Pill drag-to-reposition ---------------------------------------------
// The minimized voice pill (and, in avatar calls, the desktop-pet frame it
// becomes) can be dragged anywhere on screen. The position is per-device and
// persisted in localStorage, so it survives mode switches, call end/restart,
// and reload. A small drag threshold means a tap still expands the call.

function _loadPillPos() {
  try {
    const raw = localStorage.getItem(_PILL_POS_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (p && Number.isFinite(p.x) && Number.isFinite(p.y)) return { x: p.x, y: p.y };
  } catch { /* corrupt / unavailable — fall through */ }
  return null;
}

function _savePillPos() {
  try {
    if (_pillPos) localStorage.setItem(_PILL_POS_KEY, JSON.stringify(_pillPos));
    else localStorage.removeItem(_PILL_POS_KEY);
  } catch { /* storage unavailable — position is just session-only then */ }
}

function _clampPillPos(x, y) {
  const w = _pillEl?.offsetWidth || 132;
  const h = _pillEl?.offsetHeight || 48;
  return {
    x: Math.max(8, Math.min((window.innerWidth - w - 8) || 8, x)),
    y: Math.max(8, Math.min((window.innerHeight - h - 8) || 8, y)),
  };
}

function _setPillXY(x, y) {
  _pillEl.style.left = `${x}px`;
  _pillEl.style.top = `${y}px`;
  _pillEl.style.right = 'auto';
  _pillEl.style.bottom = 'auto';
}

function _applyPillPos() {
  if (!_pillEl) return;
  if (!_pillPos) {
    for (const prop of ['left', 'top', 'right', 'bottom']) _pillEl.style.removeProperty(prop);
    return;
  }
  _pillPos = _clampPillPos(_pillPos.x, _pillPos.y);
  _setPillXY(_pillPos.x, _pillPos.y);
}

function _isMobileViewport() {
  return window.matchMedia('(max-width: 767px)').matches;
}

function _cancelLongPress() {
  if (_pillLongPressTimer) {
    clearTimeout(_pillLongPressTimer);
    _pillLongPressTimer = null;
  }
}

function _armPillDrag() {
  _pillDragArmed = true;
  _pillEl?.classList.add('drag-armed');
  // Subtle haptic if the device supports it — confirms the drag is live
  // without needing visual focus, which matters when the user is mid-
  // gesture and looking at the pill rather than feedback chrome.
  try { navigator.vibrate?.(15); } catch { /* ignore */ }
}

function _onPillPointerDown(e) {
  if (!_pillEl || !isMinimized) return;
  if (e.button !== undefined && e.button !== 0) return;  // primary button / touch / pen only
  const rect = _pillEl.getBoundingClientRect();
  _pillDragStart = { px: e.clientX, py: e.clientY, ox: rect.left, oy: rect.top };
  _pillDragMoved = false;
  _pillDragArmed = !_isMobileViewport();  // desktop: drag is always armed (existing behavior)
  try { _pillEl.setPointerCapture(e.pointerId); } catch { /* not capturable — drag still works via document */ }
  _pillEl.addEventListener('pointermove', _onPillPointerMove);
  _pillEl.addEventListener('pointerup', _onPillPointerUp);
  _pillEl.addEventListener('pointercancel', _onPillPointerUp);

  // Mobile: arm the drag after a long-press dwell. If the user moves more
  // than the slop threshold before the timer fires, they're scrolling or
  // tapping — cancel the arm so the gesture stays passive.
  if (!_pillDragArmed) {
    _cancelLongPress();
    _pillLongPressTimer = setTimeout(() => {
      _pillLongPressTimer = null;
      if (!_pillDragStart) return;  // released before arming
      _armPillDrag();
    }, _PILL_LONGPRESS_MS);
  }
}

function _onPillPointerMove(e) {
  if (!_pillDragStart) return;
  const dx = e.clientX - _pillDragStart.px;
  const dy = e.clientY - _pillDragStart.py;
  // Pre-arm (mobile): movement above slop cancels the long-press so the
  // user can still scroll the page underneath without dragging the pill.
  if (!_pillDragArmed) {
    if (Math.hypot(dx, dy) >= _PILL_LONGPRESS_SLOP) _cancelLongPress();
    return;
  }
  if (!_pillDragMoved && Math.hypot(dx, dy) < 4) return;  // jitter tolerance — keep tap-to-expand working
  _pillDragMoved = true;
  _pillEl.classList.add('dragging');
  const { x, y } = _clampPillPos(_pillDragStart.ox + dx, _pillDragStart.oy + dy);
  _pillPos = { x, y };
  _setPillXY(x, y);
}

function _onPillPointerUp(e) {
  if (!_pillDragStart) return;
  _cancelLongPress();
  try { _pillEl.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
  _pillEl.removeEventListener('pointermove', _onPillPointerMove);
  _pillEl.removeEventListener('pointerup', _onPillPointerUp);
  _pillEl.removeEventListener('pointercancel', _onPillPointerUp);
  _pillDragStart = null;
  _pillDragArmed = false;
  _pillEl.classList.remove('dragging', 'drag-armed');
  if (_pillDragMoved) {
    _savePillPos();
    // The browser fires a click right after the drag's pointerup — flag it so
    // _onPillClick ignores that one. Self-clears in case no click follows.
    _pillJustDragged = true;
    setTimeout(() => { _pillJustDragged = false; }, 50);
  }
}

function _onPillClick() {
  if (_pillJustDragged) { _pillJustDragged = false; return; }
  expandVoiceCall();
}

async function reconnectVoiceCall() {
  if (_disconnectTimer) { clearTimeout(_disconnectTimer); _disconnectTimer = null; }
  if (ws) { try { ws.close(); } catch {} ws = null; }

  setState('connecting');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  // Stay pinned to the originating session so the server's VoiceSession
  // context and client-side message routing both stick with the call's origin.
  const sessionId = _callSessionId || app.state.currentSessionId || 'voice_default';
  const model = (app.state.currentModel && app.state.currentModel !== 'default')
    ? app.state.currentModel
    : (localStorage.getItem('augmentum-selected-model') || '');
  const mode = app.state.mode || 'passthrough';
  const ticket = await getWsTicket();
  const wsUrl = `${proto}://${location.host}/ws/voice?ticket=${encodeURIComponent(ticket)}&session_id=${encodeURIComponent(sessionId)}&model=${encodeURIComponent(model)}&mode=${encodeURIComponent(mode)}`;

  try {
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
  } catch (err) {
    setState('disconnected');
    showToast('Failed to reconnect', 'error');
    return;
  }

  // Re-attach handlers since ws is a new object
  ws.onopen = async () => {
    isConnected = true;
    setState('listening');
    _startHeartbeat();
    if (navigator.vibrate) navigator.vibrate([5, 10, 5]);
    showToast('Voice reconnected', 'success');

    // Resend config
    const voiceSettings = safeParseJSON(localStorage.getItem('augmentum_settings'), {});
    ws.send(JSON.stringify({
      type: 'config',
      voice: voiceSettings.voiceDefaultVoice || '',
      speed: parseFloat(voiceSettings.voiceSpeed) || 1.0,
      input_mode: inputMode,
    }));

    if (inputMode === 'ptt') {
      ws.send(JSON.stringify({ type: 'ptt_active', active: false }));
    }
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      handleAudioChunk(event.data);
      return;
    }
    try {
      const msg = JSON.parse(event.data);
      handleServerMessage(msg);
    } catch { /* ignore */ }
  };

  ws.onerror = () => {
    setState('disconnected');
  };

  ws.onclose = (event) => {
    _stopHeartbeat();
    if (isConnected) {
      isConnected = false;
      setState('disconnected');
      _disconnectTimer = setTimeout(() => {
        if (!isConnected) endVoiceCall();
      }, 30000);
    }
  };
}

function _startHeartbeat() {
  _stopHeartbeat();
  _heartbeatTimer = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      _stopHeartbeat();
      return;
    }
    try {
      ws.send(JSON.stringify({ type: 'ping' }));
    } catch (err) {
      // send() throws on a half-open WS — let onclose handle the surface
      // transition so the user gets the reconnect button rather than a
      // stale UI on the next interaction.
      _stopHeartbeat();
    }
  }, _HEARTBEAT_MS);
}

function _stopHeartbeat() {
  if (_heartbeatTimer) {
    clearInterval(_heartbeatTimer);
    _heartbeatTimer = null;
  }
}

/** Detect a half-open WS the moment the user comes back to the tab.
 *
 *  Backgrounded tabs throttle timers and can miss the WS close event when
 *  the server (or a proxy) reaps the idle connection. The heartbeat above
 *  prevents the close in most cases, but it doesn't help if the tab itself
 *  was suspended. On visibility return, if voice is supposed to be active
 *  but the WS is no longer OPEN, surface the disconnected state immediately
 *  so the user lands on the reconnect button instead of a frozen overlay.
 */
function _onVisibilityChange() {
  if (document.visibilityState !== 'visible') return;
  if (!overlay || !overlay.classList.contains('active')) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    if (isConnected) {
      isConnected = false;
      _stopHeartbeat();
      setState('disconnected');
      // Reuse the existing 30s auto-cleanup so an abandoned tab still
      // tears itself down rather than holding the overlay forever.
      if (!_disconnectTimer) {
        _disconnectTimer = setTimeout(() => {
          if (!isConnected) endVoiceCall();
        }, 30000);
      }
    }
  }
}

function cleanupMic() {
  if (_clientVad) {
    _clientVad.destroy().catch(() => {});
    _clientVad = null;
  }
  if (_pttRecorder) {
    try { _pttRecorder.cancel(); } catch (_) {}
    _pttRecorder = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  mediaRecorder = null;
  isRecording = false;
}

// ---------------------------------------------------------------------------
// Duration Timer
// ---------------------------------------------------------------------------
function startDurationTimer() {
  if (durationIntervalId) { clearInterval(durationIntervalId); durationIntervalId = null; }
  callStartTime = Date.now();
  if (durationEl) durationEl.textContent = '0:00';
  durationIntervalId = setInterval(() => {
    if (!callStartTime || !durationEl) return;
    const elapsed = Math.floor((Date.now() - callStartTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    const formattedDuration = `${mins}:${secs.toString().padStart(2, '0')}`;
    durationEl.textContent = formattedDuration;
    if (_pillTimerEl && isMinimized) {
      _pillTimerEl.textContent = formattedDuration;
    }
  }, 1000);
}

function stopDurationTimer() {
  if (durationIntervalId) {
    clearInterval(durationIntervalId);
    durationIntervalId = null;
  }
  callStartTime = null;
}

// ---------------------------------------------------------------------------
// Server Message Handling
// ---------------------------------------------------------------------------
// Message types suppressed during staging (until user hits Send)
const _STAGING_SUPPRESSED = ['llm_start', 'llm_delta', 'tts_start', 'tts_end'];

function handleServerMessage(msg) {
  // Staging guard: suppress LLM/TTS messages when staging is active but user hasn't sent yet
  if (stageActive && !stageSentPending && _STAGING_SUPPRESSED.includes(msg.type)) return;

  switch (msg.type) {
    case 'listening':
      setState('listening');
      vadSuppressed = false;
      if (_pttFallbackTimer) { clearTimeout(_pttFallbackTimer); _pttFallbackTimer = null; }
      // First "listening" message tells us if server VAD is active
      if (msg.server_vad !== undefined) {
        serverVadActive = !!msg.server_vad;
        if (voice._debug) console.debug('[Voice] server_vad:', serverVadActive);
        if (serverVadActive) {
          // Server handles VAD — stop any legacy client VAD/capture first
          stopVadMonitoring();
          // Start streaming raw PCM frames
          startPcmStreaming();
        } else if (inputMode === 'auto') {
          // Legacy client VAD
          startVadMonitoring();
        }
      }
      // Check server STT availability — activate browser fallback if needed
      if (msg.server_stt !== undefined) {
        _serverSttAvailable = !!msg.server_stt;
        if (!_serverSttAvailable && _SpeechRecognition) {
          _startBrowserStt();
        } else {
          _stopBrowserStt();
        }
      }
      // Check if voice enrollment is needed (first-time use)
      if (msg.needs_enrollment) {
        checkAndShowEnrollment(true);
      }
      break;

    case 'speaker_rejected':
      // Server rejected speech — not the enrolled user.
      // Show brief status feedback so user knows why nothing happened.
      if (statusText) {
        statusText.textContent = 'Voice not recognized';
        statusText.style.color = 'rgba(224, 144, 112, 0.7)';
        setTimeout(() => {
          if (statusText.textContent === 'Voice not recognized') {
            statusText.textContent = 'Listening';
            statusText.style.color = '';
          }
        }, 2000);
      }
      if (navigator.vibrate) navigator.vibrate([10, 30, 10]);
      break;

    case 'voice_no_speech': {
      // Server captured audio but STT (both streaming and batch fallback)
      // returned nothing. Surface it so the user retries instead of
      // waiting on dead air.
      const label = (msg && msg.message) || "I didn't catch that — try again?";
      if (statusText) {
        statusText.textContent = label;
        statusText.style.color = 'rgba(224, 144, 112, 0.7)';
        setTimeout(() => {
          if (statusText.textContent === label) {
            statusText.textContent = 'Listening';
            statusText.style.color = '';
          }
        }, 2500);
      }
      if (navigator.vibrate) navigator.vibrate([10, 30, 10]);
      break;
    }

    case 'vad_state':
      // Server-side VAD state change — update UI
      if (msg.speaking) {
        setState('recording');
        _poseTrigger?.onUserStartedSpeaking();
      } else {
        // Speech ended server-side — processing will follow
        _poseTrigger?.onUserStoppedSpeaking();
      }
      break;

    case 'partial_transcript':
      // Live partial transcript from streaming STT — show in log as live bubble
      if (msg.text) {
        if (transcriptUser) transcriptUser.textContent = msg.text;
        _updateLivePartial(msg.text);
      }
      // Feed to presence engine for listening reactions. The fan-out
      // helper hits both primary and secondary (group) engines so both
      // characters react to the user holding the floor.
      avatarModule.onUserTranscript?.(msg.text, !!msg.is_final);
      // Feed to subtitle renderer
      if (avatarModule?.avatarState?.subtitle) {
        avatarModule.avatarState.subtitle.setUserSpeech(msg.text);
      }
      _resetUserTranscriptDim();
      break;

    case 'processing':
      if (voice._debug) console.debug('[Voice] server: processing audio');
      setState('processing');
      break;

    case 'transcript':
      // Server acknowledged speech — cancel PTT fallback timer
      if (_pttFallbackTimer) { clearTimeout(_pttFallbackTimer); _pttFallbackTimer = null; }
      if (msg.text && msg.text.trim()) {
        const trimmed = msg.text.trim();
        // Feed final transcript to presence engine — fan-out covers
        // both primary and secondary (group) engines.
        avatarModule.onUserTranscript?.(trimmed, true);
        // Feed final transcript to pose trigger engine — checks for
        // greeting/farewell/good-news keywords + drives idle reset
        _poseTrigger?.onUserTranscriptFinal(trimmed);
        if (voice._debug) console.debug('[Voice] transcript:', trimmed);
        // [5] Backchannel filtering (disabled — let LLM handle all speech)
        if (BACKCHANNEL_RE && trimmed.split(/\s+/).length <= BACKCHANNEL_MAX_WORDS && BACKCHANNEL_RE.test(trimmed)) {
          if (voice._debug) console.debug('[Voice] backchannel filtered:', trimmed);
          setState('listening');
          break;
        }
        // Stage manager: append to staging area instead of sending immediately
        // Skip if in cooldown (just sent — this transcript is from the previous turn)
        if (stageActive && stageTextEl && !_stageCooldown) {
          _stageAppend(trimmed);
          setState('listening');
          break;
        }
        if (stageActive && _stageCooldown) {
          // Stale transcript from before Send — discard silently
          setState('listening');
          break;
        }
        // Check for voice commands before sending to LLM
        const cmdEvent = new CustomEvent('voice:pre-send', {
          detail: { text: trimmed },
          cancelable: true,
        });
        document.dispatchEvent(cmdEvent);
        if (cmdEvent.defaultPrevented) {
          // Command was handled — show in log but don't send to LLM
          _addUserBubble(trimmed);
          setState('listening');
          break;
        }
        if (transcriptUser) transcriptUser.textContent = trimmed;
        _resetUserTranscriptDim();
        // Add user bubble to the in-call conversation LOG (display only).
        // Persisting to the chat tree happens on the server-authoritative
        // 'user_committed' signal — not here — so a stale stage flag or a
        // learned-command match can't drop the user side of the call while
        // the assistant side is kept ("only assistant turns saved").
        _addUserBubble(trimmed);
      } else {
        if (voice._debug) console.debug('[Voice] empty transcript — no speech detected by STT');
        // Empty transcript (noise, no speech detected) — return to listening
        setState('listening');
      }
      break;

    case 'user_committed':
      // The server has accepted this utterance as a real conversational
      // turn (it is generating a reply now) and is telling us to record the
      // user side. This is the SINGLE authoritative persistence point for
      // user turns — symmetric with the assistant's 'turn_complete' — and is
      // deliberately independent of the in-call display state (stage flag,
      // command match, cooldown) that previously gated it. Stage Send and
      // browser-STT turns persist client-side instead and are not echoed
      // here (the server suppresses 'user_committed' for them), so this
      // never double-saves.
      if (msg.text && msg.text.trim()) {
        addMessageToChat('user', msg.text.trim());
      }
      break;

    case 'group_speaker':
      // Server-emitted at speaker-resolution time (BEFORE the LLM
      // generates this turn's tokens). Auto-swap the avatar viewport
      // to follow whichever character is about to speak. Voice TTS
      // routing already happens server-side; this is purely the
      // visual swap on the client.
      if (msg.speaker) {
        import('./avatar.js')
          .then(m => m.onSpeakerSwitch?.(msg.speaker))
          .catch(() => {});
      }
      break;

    case 'llm_start':
      // Use 'processing' until first audio actually plays — avoids the
      // visual "speaking" state before the user hears anything.
      setState('processing');
      _stageCooldown = false; // LLM responding — safe to accept new transcripts
      _hidePinDuringStream(); // Hide old pin — new response streaming in
      // Start new AI turn in the transcript log
      if (transcriptAi) {
        transcriptAi.classList.remove('stage-hidden');
        _startNewAiBubble();
      }
      // Suppress VAD during AI speech to prevent echo triggering
      vadSuppressed = true;
      turnDone = false;
      // [6] Reset playback tracking for new response
      currentResponseText = '';
      currentResponseSaved = false;
      _turnImages = [];  // fresh per-turn image accumulator
      playedSentenceCount = 0;
      queuedSentenceCount = 0;
      _enterConversing();
      _resetAiTranscriptDim();
      break;

    case 'llm_delta':
      if (msg.text) {
        // [6] Accumulate full response text for playback tracking
        currentResponseText += msg.text;
        // Append with word-by-word reveal animation
        _appendAiDelta(msg.text);
        _resetAiTranscriptDim();
        // Forward delta to avatar for emotion extraction
        if (avatarModule?.avatarState?.active) {
          avatarModule.onLLMDelta(msg.text);
        }
      }
      break;

    case 'viseme_schedule':
      // Phoneme-driven lip-sync schedule. Arrives before tts_start for the
      // same sentence (server emits it in-order on the same WS). FIFO queue
      // — popped when the corresponding sentence's audio begins playback.
      //
      // Two schedule shapes are accepted:
      //   Absolute: events carry `t` in ms; `duration_ms` is set. Emitted
      //     by the Kokoro path which knows audio duration at synth time.
      //   Normalized: events carry `t_norm` in [0.0, 1.0]; `duration_ms`
      //     is null; `normalized: true`. Emitted by external-provider
      //     paths that stream audio without up-front duration. The
      //     attach helpers below rescale these once the audio decoder
      //     reports actual duration.
      if (msg.events && Array.isArray(msg.events)) {
        pendingVisemeSchedules.push({
          duration_ms: msg.duration_ms || 0,
          events: msg.events,
          sentence: msg.sentence || '',
          normalized: msg.normalized === true,
        });
      }
      break;

    case 'tts_start':
      // Flush any prior sentence's chunks before starting new one
      if (sentenceAudioChunks.length > 0) {
        _flushSentenceChunks();
      }
      sentenceAudioChunks = [];
      currentTtsFormat = msg.format || 'mp3';
      currentTtsSentence = msg.sentence || '';
      // Group chat: switch active speaker avatar when character attribution present
      if (avatarModule.avatarState.active && msg.character) {
        avatarModule.onSpeakerSwitch(msg.character);
      }
      // Pose trigger: AI is about to speak — classify sentiment of the
      // first sentence and possibly fire a reactive animation.
      _poseTrigger?.onResponseStarted(msg.sentence || '');
      break;

    case 'tts_end':
      // Flush any remaining chunks for the last sentence
      if (sentenceAudioChunks.length > 0) {
        _flushSentenceChunks();
        sentenceAudioChunks = [];
      }
      _scheduleExitConversing();
      _poseTrigger?.onResponseEndedSpeaking();
      break;

    case 'tts_error':
      // TTS failed — notify user so they know why audio didn't play
      if (voice._debug) console.debug('[Voice] TTS error:', msg.message);
      showToast(msg.message || 'TTS failed', 'error');
      break;

    case 'turn_complete':
      stageSentPending = false;
      {
        const finalText = _bestAssistantTurnText(msg.full_text);
        const saveText = _assistantSaveText(finalText);
        if (saveText) {
          _finalizeAiBubble(finalText);  // live display stays clean — images already shown as cards
          addMessageToChat('assistant', saveText);  // saved turn carries the images as markdown
          currentResponseSaved = true;
          lastAiText = finalText;
          // If stage is active, hide the transcript and show pin instead
          // (avoids duplicate: pin shows context, transcript showed streaming)
          if (stageActive && transcriptAi) {
            transcriptAi.classList.add('stage-hidden');
            _updatePin();
          }
        }
      }
      // Mark turn as done — actual listening transition happens when
      // playSentenceQueue() drains the last audio blob.
      // Do NOT transition to listening here — audio may still be playing
      // or queued. playSentenceQueue() checks turnDone when the queue empties.
      turnDone = true;
      // Only transition immediately if no audio was ever queued for this turn
      // (e.g. text-only response with no TTS)
      if (!isPlaying && sentenceQueue.length === 0 && queuedSentenceCount === 0) {
        setState('listening');
        vadSuppressed = false;
        vadSuppressUntil = Date.now() + VAD_ECHO_COOLDOWN_MS;
        _scheduleExitConversing();
      }
      break;

    case 'interrupted':
      stageSentPending = false;
      stopPlayback();
      // Persist what the user heard. Fall back to the full streamed text
      // when the server couldn't reconstruct a heard prefix (interrupt
      // before the first sentence finished) — better a slightly-long
      // bubble than a dropped turn. Guarded so a duplicate 'interrupted'
      // (server-VAD soft interrupt racing a client interrupt message)
      // doesn't double-add the message.
      if (!currentResponseSaved) {
        const interruptedText = (msg.heard_text || currentResponseText || '').trim();
        const interruptedSave = _assistantSaveText(interruptedText);
        if (interruptedSave) {
          _finalizeAiBubble(interruptedText);
          addMessageToChat('assistant', interruptedSave);
          lastAiText = interruptedText;
        }
        currentResponseSaved = true;
      }
      setState('listening');
      vadSuppressed = false;
      if (overlay) overlay.classList.remove('conversing');
      break;

    case 'max_audio_reached': {
      // Max recording length hit — activate stage manager with partial transcript
      // so the user can review and send instead of losing their speech.
      const partial = (msg.transcript || '').trim();
      showToast(`Recording limit (${msg.seconds || 30}s) — review in stage manager`, 'info');

      // Activate stage manager if not already active
      if (!stageActive) {
        const stageToggle = document.getElementById('voice-stage-toggle');
        if (stageToggle) stageToggle.click();
      }

      // Inject partial transcript into stage text
      if (partial && stageTextEl) {
        const existing = stageTextEl.textContent.trim();
        stageTextEl.textContent = existing ? `${existing} ${partial}` : partial;
        stageChunks = [stageTextEl.textContent.trim()];
        _placeCursorAtEnd();
      }
      break;
    }

    case 'status': {
      const _voiceStatusLabels = {
        thinking: 'Thinking',
        composing: 'Composing response',
      };
      if (statusText) statusText.textContent = _voiceStatusLabels[msg.stage] || msg.stage;
      break;
    }

    case 'tool_activity': {
      if (voice._debug) console.debug('[Voice] tool activity:', msg.status, msg.tools);
      const _voiceToolLabels = {
        web: 'Searching the web', web_search: 'Searching the web',
        web_fetch: 'Fetching page', wikipedia: 'Checking Wikipedia',
        youtube_transcript: 'Looking up video', calculator: 'Calculating',
        datetime: 'Checking the time', unit_converter: 'Converting units',
        image_generation: 'Generating image', python_exec: 'Running code',
        document_parse: 'Reading document', memory_recall: 'Checking memory',
      };
      const names = msg.tools || [];
      const label = _voiceToolLabels[names[0]] || `Using ${names.join(', ')}`;
      if (statusText) statusText.textContent = label;
      break;
    }

    case 'tool_result':
      if (voice._debug) console.debug('[Voice] tool result:', msg.tool, msg.success);
      if (statusText) statusText.textContent = 'Reviewing results';
      if (msg.tool === 'image_generation' && msg.success) {
        if (msg.image_id) {
          const _genUrl = `/api/image/${msg.image_id}`;
          _showVoiceImage(_genUrl);
          if (!_turnImages.includes(_genUrl)) _turnImages.push(_genUrl);
        } else {
          _fetchAndShowVoiceImage();
        }
      }
      if (msg.tool === 'image_search' && msg.success && Array.isArray(msg.images) && msg.images.length) {
        _showVoiceImages(msg.images);
        for (const im of msg.images) {
          // Persist the local artifact URL (/api/...) — survives server sync.
          const u = im && (im.embed_url || im.download_url || im.url);
          if (u && !_turnImages.includes(u)) _turnImages.push(u);
        }
      }
      if (msg.tool === 'youtube' && msg.success) {
        if (msg.youtube_mode === 'search' && Array.isArray(msg.videos) && msg.videos.length) {
          _showVoiceVideos(msg.videos);
        } else if (msg.youtube_mode === 'direct' && msg.video?.video_id) {
          _playVoiceVideo(msg.video);
        }
      }
      break;

    case 'error':
      stageSentPending = false;
      console.warn('[Voice] server error:', msg.message);
      showToast(msg.message || 'Voice error', 'error');
      setState('listening');
      vadSuppressed = false;
      break;
    case 'intent_action':
      // Server-side action registry short-circuited the turn. Surface
      // effects (open browse, cancel TTS, etc.) run via the shared
      // router. The call modal stays open — user can keep talking.
      import('./intent-action-router.js')
        .then(m => m.dispatchIntentAction?.(msg))
        .catch(err => console.warn('[Voice] intent dispatch failed', err));
      setState('listening');
      break;
  }
}

// ---------------------------------------------------------------------------
// State Display
// ---------------------------------------------------------------------------
function setState(newState) {
  if (!overlay) return;

  // Haptic feedback on state transitions (mobile)
  if (navigator.vibrate) {
    if (newState === 'recording') navigator.vibrate(15);
    else if (newState === 'speaking') navigator.vibrate([5, 30, 5]);
    else if (newState === 'listening') navigator.vibrate(5);
  }

  overlay.setAttribute('data-state', newState);
  if (_pillEl && isMinimized) {
    _pillEl.setAttribute('data-state', newState);
  }
  if (statusText) {
    const labels = {
      connecting: 'Connecting',
      reconnecting: 'Reconnecting…',
      listening: 'Listening',
      recording: 'Recording',
      processing: 'Processing',
      speaking: 'Speaking',
      composing: 'Composing',
      disconnected: 'Disconnected',
    };
    statusText.textContent = labels[newState] || newState;
  }

  // Update orb energy profile
  if (ORB_PROFILES[newState]) {
    orbTargetProfile = newState;
  }

  // Track pre-composing state so we can restore when stage deactivates
  if (newState !== 'composing') {
    _preComposingState = newState;
  }

  // TTS synthetic pulse
  ttsPulseActive = (newState === 'speaking');
  if (!ttsPulseActive) ttsPulsePhase = 0;

  // Notify avatar of state change (for animation layer switching)
  avatarModule.onStateChange(newState);
}

// ---------------------------------------------------------------------------
// Push-to-Talk
// ---------------------------------------------------------------------------
function startPtt() {
  if (!isConnected || inputMode !== 'ptt' || isRecording) return;
  if (_pttFallbackTimer) { clearTimeout(_pttFallbackTimer); _pttFallbackTimer = null; }
  // Local-batch STT path: record the held utterance and transcribe via the
  // server's /v1/audio/transcriptions endpoint (ffmpeg → Moonshine batch) on
  // release, then inject the text (stage_send). The button press IS the
  // boundary, so we skip streaming PCM + server/client VAD entirely — the
  // same clean, on-device approach as the chat mic button (which works
  // flawlessly). Stays local: audio goes to our server, not a vendor cloud.
  if (micStream) {
    try {
      _pttRecorder = createUtteranceRecorder(micStream);
      _pttRecorder.start();
      isRecording = true;
      setState('recording');
      if (pttBtn) pttBtn.classList.add('pressed');
      if (navigator.vibrate) navigator.vibrate(20);
      return;
    } catch (err) {
      console.warn('[Voice] PTT recorder failed, using legacy capture', err);
      _pttRecorder = null;
    }
  }
  if (serverVadActive) {
    isRecording = true;
    setState('recording');
    if (pttBtn) pttBtn.classList.add('pressed');
    // Tell server to start processing VAD on incoming audio
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ptt_active', active: true }));
    }
    return;
  }
  startRecording();
  if (navigator.vibrate) navigator.vibrate(20);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'start_recording' }));
  }
}

function stopPtt() {
  if (!isRecording || inputMode !== 'ptt') return;
  if (navigator.vibrate) navigator.vibrate(10);
  if (_pttRecorder) {
    // Local-batch path: stop, transcribe, inject text into the existing
    // turn pipeline. Reply + TTS stream back over the call WS as usual.
    isRecording = false;
    if (pttBtn) pttBtn.classList.remove('pressed');
    setState('processing');
    const rec = _pttRecorder;
    _pttRecorder = null;
    (async () => {
      let transcript = '';
      try { transcript = await rec.stop(); }
      catch (err) { console.warn('[Voice] PTT batch STT failed', err); }
      if (transcript && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stage_send', text: transcript }));
        // stage_send tells the server the client owns persistence (it
        // suppresses 'user_committed'), so this sender MUST record the
        // user turn itself — same contract the staging Send button and
        // browser-STT senders honor. Without these two lines PTT turns
        // showed the assistant's reply but silently dropped the user's
        // words from the chat log.
        _addUserBubble(transcript);
        addMessageToChat('user', transcript);
        if (transcriptUser) {
          transcriptUser.textContent = transcript;
          _resetUserTranscriptDim();
        }
      } else {
        // Nothing heard — back to listening instead of a stuck spinner.
        setState('listening');
      }
    })();
    return;
  }
  if (serverVadActive) {
    isRecording = false;
    if (pttBtn) pttBtn.classList.remove('pressed');
    // Tell server to stop processing VAD — finalize any in-progress speech
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ptt_active', active: false }));
    }
    // Transition UI back to processing (server will send 'listening' after STT/LLM)
    // If VAD never triggered (too short/quiet), fall back to listening after a timeout
    setState('processing');
    _pttFallbackTimer = setTimeout(() => {
      const curState = overlay?.getAttribute('data-state');
      if (curState === 'processing') setState('listening');
    }, 3000);
    return;
  }
  stopRecording();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'stop_recording' }));
  }
}

// ---------------------------------------------------------------------------
// Recording (shared by PTT and Auto)
// ---------------------------------------------------------------------------
function startRecording() {
  if (isRecording || !micStream) return;
  isRecording = true;
  setState('recording');
  if (pttBtn) pttBtn.classList.add('pressed');

  mediaRecorder = new MediaRecorder(micStream, {
    mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm'
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) {
      e.data.arrayBuffer().then(buf => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(buf);
        }
      });
    }
  };

  mediaRecorder.start(250); // Send chunks every 250ms
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  if (pttBtn) pttBtn.classList.remove('pressed');

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

// ---------------------------------------------------------------------------
// Browser SpeechRecognition Fallback
// ---------------------------------------------------------------------------
// Activates when the server has no STT provider (no Moonshine, no Deepgram).
// Uses the browser's built-in speech recognition:
//   - Safari/iOS: runs on-device via Apple Neural Engine (private, fast)
//   - Chrome: sends to Google servers (high quality, not private)
//   - Firefox: not supported
// Feeds transcripts into the same pipeline as server STT.

const _SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function _startBrowserStt() {
  if (!_SpeechRecognition) {
    console.warn('[Voice] Browser SpeechRecognition not available');
    return;
  }
  if (_browserStt) return; // already running

  _browserStt = new _SpeechRecognition();
  _browserStt.continuous = true;
  _browserStt.interimResults = true;
  _browserStt.maxAlternatives = 1;

  // Match the user's browser language (can be overridden by settings later)
  _browserStt.lang = navigator.language || 'en-US';

  _browserStt.onresult = (event) => {
    let interim = '';
    let final = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        final += transcript;
      } else {
        interim += transcript;
      }
    }

    // Show interim in user transcript for live feedback
    if (interim && transcriptUser) {
      transcriptUser.textContent = interim;
      transcriptUser.classList.remove('dimmed');
    }

    if (final) {
      const trimmed = final.trim();
      if (!trimmed) return;

      // Stage manager: append to staging area
      if (stageActive && stageTextEl && !_stageCooldown) {
        _stageAppend(trimmed);
        if (transcriptUser) transcriptUser.textContent = '';
        return;
      }

      // Direct mode: send transcript to server as a stage_send
      // (server processes it through LLM without needing its own STT)
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'stage_send',
          text: trimmed,
          xr_user_context: _drainPendingXrUserSignals(),
        }));
      }
      if (transcriptUser) {
        transcriptUser.textContent = trimmed;
        _resetUserTranscriptDim();
      }
      addMessageToChat('user', trimmed);
    }
  };

  _browserStt.onerror = (event) => {
    // 'no-speech' and 'aborted' are normal — restart silently
    if (event.error === 'no-speech' || event.error === 'aborted') return;
    console.warn('[Voice] Browser STT error:', event.error);
    // 'not-allowed' means permission denied — don't retry
    if (event.error === 'not-allowed') {
      _browserStt = null;
      return;
    }
  };

  _browserStt.onend = () => {
    // Auto-restart if still in a call (recognition stops after silence)
    if (isConnected && _browserStt && !_serverSttAvailable) {
      try { _browserStt.start(); } catch { /* already started */ }
    }
  };

  try {
    _browserStt.start();
    if (voice._debug) console.debug('[Voice] Browser STT started (fallback mode)');
  } catch (err) {
    console.warn('[Voice] Browser STT start failed:', err.message);
    _browserStt = null;
  }
}

function _stopBrowserStt() {
  if (_browserStt) {
    try { _browserStt.stop(); } catch { /* ignore */ }
    _browserStt = null;
    _browserSttInterim = '';
  }
}

// ---------------------------------------------------------------------------
// VAD (Auto-Detect Mode)
// ---------------------------------------------------------------------------
let vadIntervalId = null;

// [1] Spectral VAD — analyze energy in speech frequency band vs total spectrum
function computeSpeechProb() {
  // Use mic analyser for VAD (not TTS analyser)
  const vadAnalyser = analyserNode?._micAnalyser || analyserNode;
  if (!vadAnalyser || !audioContext) return 0;
  const data = new Uint8Array(vadAnalyser.frequencyBinCount);
  vadAnalyser.getByteFrequencyData(data);

  const binHz = audioContext.sampleRate / vadAnalyser.fftSize;
  const lo = Math.floor(VAD_SPEECH_LOW_HZ / binHz);
  const hi = Math.min(Math.ceil(VAD_SPEECH_HIGH_HZ / binHz), data.length - 1);

  let speechSum = 0, totalSum = 0;
  for (let i = 0; i < data.length; i++) {
    totalSum += data[i];
    if (i >= lo && i <= hi) speechSum += data[i];
  }

  if (totalSum < 100) return 0; // Near silence — nothing meaningful

  const ratio = speechSum / totalSum;
  const level = speechSum / ((hi - lo + 1) * 255); // Normalized 0-1

  // High ratio = energy concentrated in speech band = likely human voice
  // Low ratio = broadband noise (fan, traffic, music) = suppress
  return ratio > VAD_SPECTRAL_RATIO ? level : level * 0.2;
}

// [3] Continuous background capture for prefix padding (auto mode only)
// Uses the PCM AudioWorklet (16kHz mono) instead of MediaRecorder (WebM).
// Raw PCM frames are sent to the server which can feed them directly to
// Moonshine without ffmpeg transcode. MediaRecorder WebM chunks can't be
// decoded by ffmpeg after the initial EBML header rotates out of the buffer.
let _autoPcmNode = null;

async function startAutoCapture() {
  if (_autoPcmNode || !micStream || !audioContext || !micSourceNode) return;
  isRecording = true;

  // Ensure AudioContext is running
  if (audioContext.state === 'suspended') {
    try { await audioContext.resume(); } catch {}
  }

  const nativeSampleRate = audioContext.sampleRate;

  // Register PCM worklet if not already done (may exist from server VAD path)
  try {
    const moduleUrl = _buildPcmProcessorUrl(nativeSampleRate);
    await audioContext.audioWorklet.addModule(moduleUrl).catch(() => {});
    URL.revokeObjectURL(moduleUrl);
  } catch { /* ignore */ }

  try {
    _autoPcmNode = new AudioWorkletNode(audioContext, 'pcm-capture-processor', {
      processorOptions: { targetSampleRate: 16000, nativeSampleRate },
    });
  } catch (err) {
    console.warn('[Voice] Auto capture worklet failed, falling back to MediaRecorder:', err);
    _startAutoCaptureMediaRecorder();
    return;
  }

  _autoPcmNode.port.onmessage = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm16 = e.data; // Int16Array, 512 samples = 1024 bytes
    if (vadStreamingActive) {
      // Speech confirmed — stream PCM to server
      ws.send(pcm16.buffer);
    } else {
      // No speech — buffer for prefix padding
      prefixBuffer.push(pcm16.buffer);
      if (prefixBuffer.length > PREFIX_BUFFER_SIZE) prefixBuffer.shift();
    }
  };

  // Connect mic → worklet (audio flows, worklet posts frames)
  micSourceNode.connect(_autoPcmNode);
  _autoPcmNode.connect(audioContext.destination);
}

// Legacy MediaRecorder fallback (only if AudioWorklet unavailable)
function _startAutoCaptureMediaRecorder() {
  mediaRecorder = new MediaRecorder(micStream, {
    mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm'
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) {
      e.data.arrayBuffer().then(buf => {
        if (vadStreamingActive && ws?.readyState === WebSocket.OPEN) {
          ws.send(buf);
        } else {
          prefixBuffer.push(buf);
          if (prefixBuffer.length > PREFIX_BUFFER_SIZE) prefixBuffer.shift();
        }
      });
    }
  };

  mediaRecorder.start(250);
}

function stopAutoCapture() {
  if (_autoPcmNode) {
    try { _autoPcmNode.disconnect(); } catch {}
    _autoPcmNode = null;
  }
  if (mediaRecorder?.state !== 'inactive') {
    try { mediaRecorder.stop(); } catch {}
  }
  mediaRecorder = null;
  isRecording = false;
  vadStreamingActive = false;
  prefixBuffer = [];
}

// ---------------------------------------------------------------------------
// Server-Side VAD: Raw PCM16 Streaming via AudioWorklet
// ---------------------------------------------------------------------------

/**
 * Start streaming raw PCM16 16kHz mono frames to the server.
 *
 * Uses an AudioWorkletNode to capture audio at 16 kHz (resampled from
 * the mic's native rate) and sends 512-sample frames (1024 bytes, 32 ms)
 * as binary WebSocket messages.  The server runs Silero VAD on each frame.
 */
async function startPcmStreaming() {
  if (pcmWorkletNode) return;
  if (!audioContext || !micStream || !micSourceNode) {
    console.warn('[Voice] PCM streaming skipped — missing audioContext/micStream/micSourceNode');
    serverVadActive = false;
    if (inputMode === 'auto') startVadMonitoring();
    return;
  }

  // Ensure AudioContext is running (may be suspended after tab switch)
  if (audioContext.state === 'suspended') {
    try { await audioContext.resume(); } catch { /* ignore */ }
  }

  const nativeSampleRate = audioContext.sampleRate;

  // Register the PCM worklet processor
  let moduleUrl;
  try {
    moduleUrl = _buildPcmProcessorUrl(nativeSampleRate);
    await audioContext.audioWorklet.addModule(moduleUrl);
  } catch (err) {
    console.warn('[Voice] AudioWorklet not supported, falling back to client VAD:', err);
    if (moduleUrl) URL.revokeObjectURL(moduleUrl);
    serverVadActive = false;
    if (inputMode === 'auto') startVadMonitoring();
    return;
  }
  // Revoke the blob URL now that the module is loaded
  if (moduleUrl) URL.revokeObjectURL(moduleUrl);

  pcmWorkletNode = new AudioWorkletNode(audioContext, 'pcm-capture-processor', {
    processorOptions: { targetSampleRate: 16000, nativeSampleRate },
  });

  // The worklet posts 512-sample PCM16 frames as Int16Array
  let _pcmFramesSent = 0;
  pcmWorkletNode.port.onmessage = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm16 = e.data; // Int16Array, 512 samples = 1024 bytes
    ws.send(pcm16.buffer);
    _pcmFramesSent++;
    if (_pcmFramesSent === 1) {
      console.debug('[Voice] First PCM frame sent to server');
    }
  };

  // Connect mic → worklet using the persistent source node (prevents GC disconnect)
  micSourceNode.connect(pcmWorkletNode);

  // CRITICAL: Connect worklet output to destination so Chrome keeps calling process().
  // Without this, Chrome optimizes away AudioWorkletNodes whose output is dangling
  // (not connected to the destination graph), stopping process() after the first call.
  // The worklet outputs silence (zeros) so this adds no audible noise.
  pcmWorkletNode.connect(audioContext.destination);

  // Detect if AudioWorklet silently fails (mobile browsers may load the module
  // but never call process()). If no frames arrive within 2s, fall back to
  // client-side VAD with MediaRecorder.
  setTimeout(() => {
    if (_pcmFramesSent === 0 && pcmWorkletNode) {
      console.warn('[Voice] AudioWorklet produced no frames after 2s — falling back to client VAD');
      stopPcmStreaming();
      serverVadActive = false;
      // Fall back to legacy client-side capture (MediaRecorder + client VAD)
      if (inputMode === 'auto') {
        startAutoCapture();
        startVadMonitoring();
      }
      // PTT mode: startRecording/stopRecording handle capture on button press
    }
  }, 2000);

  console.debug('[Voice] PCM streaming started, native rate:', nativeSampleRate,
    'audioContext.state:', audioContext.state);
}

function stopPcmStreaming() {
  if (pcmWorkletNode) {
    pcmWorkletNode.disconnect();
    pcmWorkletNode = null;
  }
}

/**
 * Build a Blob URL for the AudioWorklet processor.
 *
 * The processor resamples from the browser's native sample rate to 16 kHz
 * and emits 512-sample PCM16 frames (32 ms at 16 kHz = exactly what Silero
 * VAD expects).
 */
function _buildPcmProcessorUrl(nativeSampleRate) {
  const code = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this._targetRate = opts.targetSampleRate || 16000;
    this._nativeRate = opts.nativeSampleRate || sampleRate;
    this._ratio = this._nativeRate / this._targetRate;
    this._frameSize = 512; // Silero VAD frame size at 16 kHz

    // Anti-aliasing low-pass applied at the NATIVE rate BEFORE decimation.
    // Downsampling to 16 kHz drops the Nyquist limit to 8 kHz; without this
    // filter, all energy above 8 kHz — including speech sibilants (s/f/sh/t,
    // which live in 8-16 kHz) — folds back into the speech band as noise and
    // wrecks STT accuracy. Two cascaded Butterworth biquads (~24 dB/oct),
    // cutoff at 0.45*target (~7.2 kHz). The old code did naive linear
    // interpolation with NO anti-alias filter at all.
    this._lp1 = this._makeLowpass(this._nativeRate, this._targetRate * 0.45);
    this._lp2 = this._makeLowpass(this._nativeRate, this._targetRate * 0.45);

    // Resampler state carried ACROSS render quanta. The old code restarted its
    // read cursor at 0 on every 128-sample process() call, dropping ~2 input
    // samples per block and resetting sub-sample phase at each seam (a periodic
    // discontinuity). These persist the continuous fractional read position and
    // the not-yet-consumed input tail so the stream resamples seamlessly.
    this._inBuf = new Float32Array(0);
    this._readPos = 0;
    this._outBuf = new Float32Array(0);
  }

  _makeLowpass(sr, fc) {
    // RBJ cookbook low-pass biquad, Q = 1/sqrt(2) (Butterworth, maximally flat).
    const w0 = 2 * Math.PI * (fc / sr);
    const cosw = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
    const a0 = 1 + alpha;
    return {
      b0: ((1 - cosw) / 2) / a0,
      b1: (1 - cosw) / a0,
      b2: ((1 - cosw) / 2) / a0,
      a1: (-2 * cosw) / a0,
      a2: (1 - alpha) / a0,
      x1: 0, x2: 0, y1: 0, y2: 0,
    };
  }

  _filterOne(st, x) {
    const y = st.b0 * x + st.b1 * st.x1 + st.b2 * st.x2 - st.a1 * st.y1 - st.a2 * st.y2;
    st.x2 = st.x1; st.x1 = x;
    st.y2 = st.y1; st.y1 = y;
    return y;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const samples = input[0]; // Float32, mono, at native rate

    // 1) Anti-alias low-pass at native rate (stateful across blocks).
    const filtered = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      filtered[i] = this._filterOne(this._lp2, this._filterOne(this._lp1, samples[i]));
    }

    // 2) Append to the persistent input buffer.
    const inBuf = new Float32Array(this._inBuf.length + filtered.length);
    inBuf.set(this._inBuf);
    inBuf.set(filtered, this._inBuf.length);
    this._inBuf = inBuf;

    // 3) Resample with a CONTINUOUS fractional cursor (phase carried across
    //    quanta), interpolating across the block seam.
    const out = [];
    let pos = this._readPos;
    while (pos + 1 < this._inBuf.length) {
      const lo = Math.floor(pos);
      const frac = pos - lo;
      out.push(this._inBuf[lo] * (1 - frac) + this._inBuf[lo + 1] * frac);
      pos += this._ratio;
    }

    // 4) Drop consumed input, keep the tail + sub-sample phase for next call.
    const consumed = Math.floor(pos);
    if (consumed > 0) this._inBuf = this._inBuf.slice(consumed);
    this._readPos = pos - consumed;

    if (out.length > 0) {
      // 5) Accumulate resampled output and emit complete 512-sample PCM16 frames.
      const outBuf = new Float32Array(this._outBuf.length + out.length);
      outBuf.set(this._outBuf);
      outBuf.set(out, this._outBuf.length);
      this._outBuf = outBuf;

      while (this._outBuf.length >= this._frameSize) {
        const frame = this._outBuf.slice(0, this._frameSize);
        this._outBuf = this._outBuf.slice(this._frameSize);
        const pcm16 = new Int16Array(this._frameSize);
        for (let i = 0; i < this._frameSize; i++) {
          const s = Math.max(-1, Math.min(1, frame[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        this.port.postMessage(pcm16);
      }
    }

    return true;
  }
}

registerProcessor('pcm-capture-processor', PcmCaptureProcessor);
`;
  const blob = new Blob([code], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}

function startVadMonitoring() {
  if (!analyserNode) return;
  if (vadIntervalId) { clearInterval(vadIntervalId); vadIntervalId = null; }

  // Apply user's silence threshold from settings as the starting point
  // for the adaptive algorithm (slider range: 400-3000ms)
  const userThreshold = getSettings().voiceSilenceThreshold;
  if (userThreshold && userThreshold >= 400 && userThreshold <= 3000) {
    vadSilenceMs = userThreshold;
  }

  // [3] Start continuous capture — chunks go to ring buffer until speech detected
  startAutoCapture();

  vadIntervalId = setInterval(() => {
    if (!isConnected || inputMode !== 'auto') return;

    // Suppress VAD during TTS playback to prevent echo loops
    if (vadSuppressed) {
      if (voice._debug && !voice._lastSuppressLog) {
        console.debug('[VAD] suppressed — TTS playing');
        voice._lastSuppressLog = true;
      }
      return;
    }
    voice._lastSuppressLog = false;
    // Cooldown period after TTS ends — speakers have latency
    if (Date.now() < vadSuppressUntil) return;

    // [1] Spectral speech probability with exponential smoothing
    const rawProb = computeSpeechProb();
    vadSpeechProb = VAD_SMOOTHING * vadSpeechProb + (1 - VAD_SMOOTHING) * rawProb;

    // Throttled debug: log every 500ms when prob is notable
    if (voice._debug && vadSpeechProb > 0.01) {
      const now = Date.now();
      if (!voice._lastProbLog || now - voice._lastProbLog > 500) {
        console.debug('[VAD] prob:', vadSpeechProb.toFixed(4), 'raw:', rawProb.toFixed(4),
          'threshold:', VAD_PROB_THRESHOLD, vadActive ? '(active)' : '');
        voice._lastProbLog = now;
      }
    }

    // Use a higher threshold during TTS playback to avoid background noise
    // (TV, pets, door) interrupting the AI mid-response. The user must
    // speak noticeably louder/closer to trigger a barge-in.
    const activeThreshold = isPlaying ? VAD_INTERRUPT_THRESHOLD : VAD_PROB_THRESHOLD;

    if (vadSpeechProb > activeThreshold) {
      vadSilenceStart = null;

      // [4] If speaking during TTS playback, duck audio first
      if (isPlaying && !duckingActive) {
        startDucking();
      }
      // [4] Escalate ducking to full barge-in after sustained speech
      if (duckingActive && duckStartTime &&
          Date.now() - duckStartTime > DUCK_TO_CANCEL_MS) {
        stopDucking();
        sendInterrupt();
      }

      if (!vadActive && !isPlaying) {
        vadActive = true;
        vadSpeechStart = Date.now();
        vadStreamingActive = true;
        setState('recording');

        // Send speech_start FIRST so server enables recording before
        // the prefix buffer chunks arrive as binary frames
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'vad_speech_start' }));
        }

        // [3] Flush prefix buffer — audio from before VAD fired
        const prefixCount = prefixBuffer.length;
        for (const chunk of prefixBuffer) {
          if (ws?.readyState === WebSocket.OPEN) ws.send(chunk);
        }
        prefixBuffer = [];

        if (voice._debug) console.debug('[VAD] speech started, flushed', prefixCount, 'prefix chunks');
      }
    } else if (vadActive) {
      if (!vadSilenceStart) {
        vadSilenceStart = Date.now();
      } else if (Date.now() - vadSilenceStart > vadSilenceMs) {
        const speechDuration = vadSpeechStart ? Date.now() - vadSpeechStart : 0;
        vadStreamingActive = false;

        if (speechDuration > VAD_MIN_SPEECH_MS) {
          // Real speech — tell server to process
          if (voice._debug) console.debug('[VAD] speech end — duration:', speechDuration, 'ms, silence threshold:', vadSilenceMs, 'ms');
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'vad_speech_end' }));
          }
          setState('processing');

          // Adapt silence threshold based on recent speech pace
          _recentSpeechDurations.push(speechDuration);
          if (_recentSpeechDurations.length > _ADAPT_WINDOW) _recentSpeechDurations.shift();
          const avgDuration = _recentSpeechDurations.reduce((a, b) => a + b, 0) / _recentSpeechDurations.length;
          // Short turns (<2s avg) → shorter silence; long turns (>5s) → longer silence
          const ratio = Math.min(Math.max(avgDuration / 3000, 0), 1);
          vadSilenceMs = Math.round(VAD_SILENCE_MS_MIN + ratio * (VAD_SILENCE_MS_MAX - VAD_SILENCE_MS_MIN));
          if (voice._debug) console.debug('[VAD] adapted silence threshold:', vadSilenceMs, 'ms (avg speech:', Math.round(avgDuration), 'ms)');
        } else {
          // Too short — background noise, discard
          if (voice._debug) console.debug('[VAD] discarded — duration:', speechDuration, 'ms (min:', VAD_MIN_SPEECH_MS, 'ms)');
          if (ws?.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'vad_discard' }));
          }
          setState('listening');
        }
        vadActive = false;
        vadSpeechStart = null;
        vadSilenceStart = null;
        vadSpeechProb = 0; // Reset smoothing
      }
    }
  }, 50);
}

function stopVadMonitoring() {
  if (vadIntervalId) {
    clearInterval(vadIntervalId);
    vadIntervalId = null;
  }
  stopAutoCapture();
  stopDucking();
  vadActive = false;
  vadSpeechStart = null;
  vadSilenceStart = null;
  vadSpeechProb = 0;
  vadSilenceMs = getSettings().voiceSilenceThreshold || VAD_SILENCE_MS_DEFAULT;
  _recentSpeechDurations.length = 0;
  duckStartTime = null;
}

// [4] Audio ducking helpers
function startDucking() {
  if (!ttsGainNode || !audioContext || duckingActive) return;
  duckingActive = true;
  duckStartTime = Date.now();
  ttsGainNode.gain.linearRampToValueAtTime(
    DUCK_LEVEL, audioContext.currentTime + DUCK_RAMP_S
  );
}

function stopDucking() {
  duckingActive = false;
  duckStartTime = null;
  if (!ttsGainNode || !audioContext) return;
  ttsGainNode.gain.linearRampToValueAtTime(
    1.0, audioContext.currentTime + DUCK_RAMP_S
  );
}

// ---------------------------------------------------------------------------
// Input Mode (PTT vs Auto)
// ---------------------------------------------------------------------------
function setInputMode(mode) {
  inputMode = mode;
  localStorage.setItem('augmentum-voice-mode', mode);
  applyInputMode();

  if (isConnected) {
    if (mode === 'auto') {
      if (serverVadActive) {
        // Server VAD handles speech detection — just open the gate
        // so server processes all incoming PCM frames (already streaming).
        // Do NOT start legacy client VAD (MediaRecorder) alongside PCM streaming.
        stopVadMonitoring();
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ptt_active', active: true }));
        }
      } else {
        // No server VAD — use legacy client-side VAD (MediaRecorder + WebM)
        startVadMonitoring();
      }
    } else {
      stopVadMonitoring();
      // Close the gate — PTT mode waits for button press
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ptt_active', active: false }));
      }
    }
  }
  _saveVoicePrefs();
}

function applyInputMode() {
  if (!overlay) return;
  overlay.setAttribute('data-input-mode', inputMode);
  if (modeTogglePtt) modeTogglePtt.classList.toggle('active', inputMode === 'ptt');
  if (modeToggleAuto) modeToggleAuto.classList.toggle('active', inputMode === 'auto');
}

// ---------------------------------------------------------------------------
// Audio Playback — per-sentence accumulation
// ---------------------------------------------------------------------------
function _setTtsPlaybackActive(active, tailMs = 0) {
  if (_ttsPlaybackTailTimer) {
    clearTimeout(_ttsPlaybackTailTimer);
    _ttsPlaybackTailTimer = null;
  }

  if (active) {
    if (!_ttsPlaybackActive) {
      _ttsPlaybackActive = true;
      avatarModule.onTtsPlaybackChange?.(true);
      _reportPlaybackState(true);
    }

    const state = overlay?.getAttribute('data-state');
    if (state === 'processing' || state === 'listening' || state === 'composing') {
      setState('speaking');
    }
    return;
  }

  const finish = () => {
    _ttsPlaybackTailTimer = null;
    if (_ttsPlaybackActive) {
      _ttsPlaybackActive = false;
      avatarModule.onTtsPlaybackChange?.(false);
      _reportPlaybackState(false);
    }
  };

  if (tailMs > 0) {
    _ttsPlaybackTailTimer = setTimeout(finish, tailMs);
  } else {
    finish();
  }
}

function _reportPlaybackState(active) {
  // The server's echo suppression + barge-in windows were anchored to
  // GENERATION end (tts_ended_at at turn commit), seconds ahead of the
  // speakers for any queued reply — her own audible tail could trip
  // phantom speech_starts and a user replying right as she finished
  // got clipped. Report real playback boundaries so the server can
  // anchor to what the room actually hears. Best-effort: older
  // servers ignore unknown message types.
  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'playback_state', active: !!active }));
    }
  } catch (_) { /* WS racing shutdown — harmless */ }
}

function handleAudioChunk(arrayBuffer) {
  // Accumulate chunks; they'll be blobbed together on tts_start/tts_end boundaries
  sentenceAudioChunks.push(arrayBuffer);
}

/**
 * Flush accumulated chunks into the sentence queue with format metadata.
 */
function _flushSentenceChunks() {
  const text = currentTtsSentence;
  if (currentTtsFormat === 'pcm') {
    sentenceQueue.push({ format: 'pcm', chunks: sentenceAudioChunks.slice(), text });
  } else {
    const mimeMap = { mp3: 'audio/mpeg', opus: 'audio/ogg', wav: 'audio/wav' };
    const blob = new Blob(sentenceAudioChunks, { type: mimeMap[currentTtsFormat] || 'audio/mpeg' });
    sentenceQueue.push({ format: 'blob', blob, text });
  }
  queuedSentenceCount++;
  if (!isPlaying) playSentenceQueue();
}

/**
 * Convert int16 PCM ArrayBuffers to a single AudioBuffer for Web Audio playback.
 */
function _pcmChunksToAudioBuffer(chunks) {
  // Concatenate all chunks into one Int16Array
  let totalSamples = 0;
  for (const buf of chunks) totalSamples += buf.byteLength / 2;
  const int16 = new Int16Array(totalSamples);
  let offset = 0;
  for (const buf of chunks) {
    const view = new Int16Array(buf);
    int16.set(view, offset);
    offset += view.length;
  }
  // Convert int16 → float32 for Web Audio
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }
  const audioBuffer = audioContext.createBuffer(1, float32.length, PCM_SAMPLE_RATE);
  audioBuffer.getChannelData(0).set(float32);
  return audioBuffer;
}

// ---------------------------------------------------------------------------
// Natural Sentence Pacing
// ---------------------------------------------------------------------------
// Computes a pause duration (ms) after a sentence based on how it ends.
// Mirrors natural speech breathing — short factual sentences get a quick beat,
// trailing ellipses get a thinking pause, em-dashes feel interrupted.

function _sentencePauseMs(text) {
  if (!text) return 0;
  const t = text.trim();
  if (!t) return 0;

  let pause;

  // Ellipsis: trailing off, thinking — long pause
  if (/[.…]{2,}\s*$/.test(t) || t.endsWith('…')) {
    pause = 400;
  }
  // Em-dash or double-dash: interrupted, cut off — barely a beat
  else if (t.endsWith('—') || t.endsWith('--')) {
    pause = 60;
  }
  // End of quoted dialogue:  ." or ?" or !" — beat after speaker finishes
  else if (/[.!?]['""\u201D\u2019]+\s*$/.test(t)) {
    pause = 280;
  }
  // Exclamation — energy carries forward
  else if (t.endsWith('!')) {
    pause = 160;
  }
  // Question — slight uptick pause
  else if (t.endsWith('?')) {
    pause = 220;
  }
  // Colon or semicolon at end (rare but possible from clause chunking)
  else if (t.endsWith(':') || t.endsWith(';')) {
    pause = 140;
  }
  // Standard period
  else if (t.endsWith('.')) {
    pause = 190;
  }
  // No clear punctuation (clause chunk, word-boundary fallback)
  else {
    pause = 100;
  }

  // Short dramatic sentences ("No." / "She ran." / "It was over.")
  // carry more weight — the brevity IS the emphasis
  if (t.length < 20) {
    pause += 120;
  } else if (t.length < 40) {
    pause += 50;
  }

  // Very long sentences — listener is already in rhythm, less recovery needed
  if (t.length > 250) {
    pause = Math.max(pause - 60, 80);
  }

  return pause;
}

let _sentencePauseTimer = null;

let _listeningGraceTimer = null;

function playSentenceQueue() {
  // [7] Don't advance to next sentence while paused
  if (paused) return;
  // Autoplay-blocked: the head sentence is requeued and waiting for the
  // user's unlock tap — draining now would just re-reject every item.
  if (_audioPlaybackBlocked) return;

  if (sentenceQueue.length === 0) {
    isPlaying = false;
    _voiceBusClaim?.release();
    _voiceBusClaim = null;
    if (turnDone) {
      // Turn is done AND queue is empty — but give a short grace period
      // in case more TTS chunks are about to arrive from the server.
      // Only transition to listening after the grace period confirms
      // no new audio arrived.
      if (_listeningGraceTimer) clearTimeout(_listeningGraceTimer);
      _listeningGraceTimer = setTimeout(() => {
        _listeningGraceTimer = null;
        // Re-check: if audio started playing during the grace period, abort
        if (isPlaying || sentenceQueue.length > 0) return;
        const state = overlay?.getAttribute('data-state');
        if (state === 'speaking') {
          setState('listening');
          vadSuppressed = false;
          vadSuppressUntil = Date.now() + VAD_ECHO_COOLDOWN_MS;
        }
        _scheduleExitConversing();
        _resetAiTranscriptDim();
      }, 800); // 800ms grace — enough for network jitter between TTS chunks
    }
    return;
  }

  // Cancel any pending listening transition — we have more audio to play
  if (_listeningGraceTimer) { clearTimeout(_listeningGraceTimer); _listeningGraceTimer = null; }

  isPlaying = true;
  if (!_voiceBusClaim) {
    _voiceBusClaim = AudioBus.claim({
      id: 'voice-mode-tts',
      tier: 'speech',
      kind: 'speech',   // drives lipsync via the call's own analyser
      duck: () => {},   // voice-mode TTS is the dominator
      unduck: () => {},
    });
  }

  // Transition to 'speaking' when audio is playing — this is the definitive
  // visual state for "AI is talking". Set it from processing OR listening
  // (turn_complete may arrive before audio finishes queuing).
  const state = overlay?.getAttribute('data-state');
  if (state === 'processing' || state === 'listening') {
    setState('speaking');
  }

  const item = sentenceQueue.shift();

  if (item.format === 'pcm') {
    // PCM: play via AudioContext (BufferSource)
    _playPcmItem(item);
    return;
  }

  if (IS_IOS) {
    // iOS Safari gates each NEW <audio> element behind a user gesture —
    // play() from a WebSocket handler throws NotAllowedError (Android/
    // desktop unlock the whole page on first tap, which is why only
    // iPhone/iPad went silent). The call-start ritual (silent buffer
    // prime) gesture-authorized the AudioContext, so decode the blob and
    // play through THAT instead of a fresh element.
    _playBlobViaWebAudio(item);
    return;
  }

  _playBlobViaElement(item);
}

// Blob (mp3/opus/wav): play via Audio element
function _playBlobViaElement(item) {
  const blob = item.blob;
  const url = URL.createObjectURL(blob);

  const audio = new Audio(url);
  currentAudio = audio;

  // This sentence's viseme_schedule is popped off the pending FIFO when
  // playback begins (onplaying → _attachVisemeScheduleFromAudio does the
  // shift). If playback never reaches onplaying (decode error / play()
  // rejection), the schedule is still at the head of the queue and the NEXT
  // sentence would pop it — a persistent off-by-one lip/audio desync that
  // survives to stopPlayback. The failure handlers below drop the orphan so
  // the FIFO stays aligned. Guarded so we never drop a *consumed* schedule
  // (which would then steal the next sentence's).
  let _scheduleConsumed = false;
  const _dropOrphanSchedule = () => {
    if (_scheduleConsumed) return;
    _scheduleConsumed = true;
    if (pendingVisemeSchedules.length) pendingVisemeSchedules.shift();
  };

  // [4] Route through GainNode for ducking control.
  // Skip on iOS: createMediaElementSource() suppresses the element's
  // native output and routes through AudioContext.destination, which on
  // iOS Safari is silenced by suspended contexts and respects routing
  // quirks that Android doesn't. Native <audio> playback is more reliable
  // there. The trade-off is no amplitude-driven lipsync on iPad/iPhone,
  // but phoneme-schedule lipsync (driven by the server's viseme_schedule
  // events, not the analyser) still works.
  let _mediaSourceNode = null;
  if (!IS_IOS && ttsGainNode && audioContext?.state === 'running') {
    try {
      _mediaSourceNode = audioContext.createMediaElementSource(audio);
      _mediaSourceNode.connect(ttsGainNode);
    } catch {
      // Fallback: play without ducking if createMediaElementSource fails
    }
  }

  function _cleanupAudioNode() {
    if (_cleanupAudioNode._done) return;
    _cleanupAudioNode._done = true;
    if (_mediaSourceNode) {
      try { _mediaSourceNode.disconnect(); } catch {}
      _mediaSourceNode = null;
    }
    URL.revokeObjectURL(url);
    if (_currentAudioCleanup === _cleanupAudioNode) _currentAudioCleanup = null;
    if (currentAudio === audio) currentAudio = null;
  }
  _currentAudioCleanup = _cleanupAudioNode;

  const onFinish = () => {
    _setTtsPlaybackActive(false, TTS_PLAYBACK_TAIL_MS);
    _cleanupAudioNode();
    playedSentenceCount++;
    // Natural pacing — pause before next sentence based on how this one ended
    const pause = _sentencePauseMs(item.text);
    if (pause > 0 && sentenceQueue.length > 0) {
      _sentencePauseTimer = setTimeout(playSentenceQueue, pause);
    } else {
      playSentenceQueue();
    }
  };

  audio.onended = () => {
    _detachVisemeSchedule();
    onFinish();
  };
  audio.onplaying = () => {
    // Playback started — the attach below consumes this sentence's schedule
    // off the pending FIFO, so the orphan-drop must not fire for it.
    _scheduleConsumed = true;
    _setTtsPlaybackActive(true);
    _attachVisemeScheduleFromAudio(audio);
  };

  audio.onerror = () => {
    _setTtsPlaybackActive(false);
    _detachVisemeSchedule();
    _dropOrphanSchedule();
    _cleanupAudioNode();
    playSentenceQueue();
  };

  audio.play().then(() => _setTtsPlaybackActive(true)).catch((err) => {
    _setTtsPlaybackActive(false);
    // play() rejected → onplaying won't fire. Detach any active schedule
    // (defensive, in case it raced).
    _detachVisemeSchedule();
    _cleanupAudioNode();
    if (err && err.name === 'NotAllowedError') {
      // Autoplay policy block (iOS Safari per-element gate, or a
      // browser that revoked the page's media permission). Silently
      // skipping every sentence was the invisible-failure mode — the
      // transcript streamed while audio died. Requeue THIS sentence
      // (schedule stays at the FIFO head for the retry — do not drop
      // the orphan) and surface a tap-to-enable pill; the tap is the
      // gesture that re-authorizes playback.
      _blockPlayback(item);
      return;
    }
    // Any other failure: skip this sentence, keep the queue moving.
    _dropOrphanSchedule();
    playSentenceQueue();
  });
}

// ---------------------------------------------------------------------------
// Autoplay-block recovery — requeue + visible re-arm (2026-07-06).
// The failure this encodes: iOS voice calls streamed the transcript but
// every sentence's play() was rejected (NotAllowedError) and swallowed,
// so the user heard nothing and saw no error.
// ---------------------------------------------------------------------------

let _audioPlaybackBlocked = false;

function _blockPlayback(item) {
  console.warn('[Voice] Audio playback blocked by the browser autoplay policy — tap the "Enable audio" pill to resume.');
  sentenceQueue.unshift(item);
  _audioPlaybackBlocked = true;
  isPlaying = false;
  _setTtsPlaybackActive(false);
  document.getElementById('voice-audio-unlock-btn')?.classList.add('visible');
}

async function _unlockAudioPlayback() {
  // Runs inside the tap gesture — the one context where iOS lets us
  // (re)authorize audio. Resume + silent-prime the context, then drain
  // the queue from the requeued sentence.
  try {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state !== 'running') await audioContext.resume();
    const b = audioContext.createBuffer(1, 1, audioContext.sampleRate);
    const s = audioContext.createBufferSource();
    s.buffer = b;
    s.connect(audioContext.destination);
    s.start();
  } catch (e) {
    console.warn('[Voice] audio unlock failed:', e);
  }
  document.getElementById('voice-audio-unlock-btn')?.classList.remove('visible');
  _audioPlaybackBlocked = false;
  if (!isPlaying) playSentenceQueue();
}

// iOS path: decode the sentence blob through the gesture-authorized
// AudioContext and play it as a BufferSource — same machinery as the PCM
// path, so viseme schedules and pacing behave identically. Falls back to
// the <audio> element path if decode fails (unsupported codec).
async function _playBlobViaWebAudio(item) {
  try {
    if (!audioContext) throw new DOMException('no AudioContext', 'NotAllowedError');
    if (audioContext.state !== 'running') {
      // 'suspended' OR iOS's non-standard 'interrupted' (Siri, lock
      // screen, route change) — resume() outside a gesture may no-op;
      // the state check below is the real gate.
      try { await audioContext.resume(); } catch { /* checked below */ }
    }
    if (audioContext.state !== 'running') {
      throw new DOMException('AudioContext not running', 'NotAllowedError');
    }
    const raw = await item.blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(raw);
    _playAudioBufferItem(item, audioBuffer);
  } catch (err) {
    if (err && err.name === 'NotAllowedError') {
      _blockPlayback(item);
      return;
    }
    console.warn('[Voice] WebAudio decode path failed, falling back to <audio> element:', err);
    _playBlobViaElement(item);
  }
}

// Rescale a normalized viseme schedule (t_norm in [0, 1]) to absolute ms
// using the audio's decoded duration. Applies the same leading-silence
// policy as the Kokoro absolute path (min(30 ms, duration * 0.25)) so
// both engines land on the same visual cadence. Mutates `sched` in
// place, flipping `normalized` to false so downstream sees a normal
// absolute schedule.
function _rescaleNormalizedSchedule(sched, durMs) {
  if (!sched || !sched.normalized || !Array.isArray(sched.events)) return;
  if (!isFinite(durMs) || durMs <= 0) return;
  const leadMs = Math.min(30, durMs * 0.25);
  const usableMs = Math.max(1, durMs - leadMs);
  sched.events = sched.events.map(e => ({
    t: leadMs + (e.t_norm || 0) * usableMs,
    v: e.v,
    w: e.w || 0,
  }));
  sched.duration_ms = durMs;
  sched.normalized = false;
}

function _attachVisemeScheduleFromAudio(audioEl) {
  if (!pendingVisemeSchedules.length) return;
  const sched = pendingVisemeSchedules.shift();
  const ls = avatarModule?.avatarState?.lipSync;
  if (!ls || typeof ls.setVisemeSchedule !== 'function') return;
  if (sched.normalized) {
    // HTMLAudio: duration becomes valid after `loadedmetadata` fires —
    // by attach time (just before/at play()) it's reliably available
    // unless the audio source is broken, in which case we skip the
    // schedule and amplitude lip-sync takes over.
    const durMs = (audioEl.duration && isFinite(audioEl.duration))
      ? audioEl.duration * 1000
      : 0;
    if (durMs <= 0) return;
    _rescaleNormalizedSchedule(sched, durMs);
  }
  ls.setVisemeSchedule(sched, () => audioEl.currentTime * 1000);
}

function _attachVisemeScheduleFromPcm(audioCtx, startCtxTime, audioBuffer) {
  if (!pendingVisemeSchedules.length) return;
  const sched = pendingVisemeSchedules.shift();
  const ls = avatarModule?.avatarState?.lipSync;
  if (!ls || typeof ls.setVisemeSchedule !== 'function') return;
  if (sched.normalized) {
    // PCM path: AudioBuffer.duration is exact (samples / sampleRate)
    // and available immediately on decode, so no readiness check is
    // needed beyond null-guarding the buffer reference.
    const durMs = (audioBuffer && isFinite(audioBuffer.duration))
      ? audioBuffer.duration * 1000
      : 0;
    if (durMs <= 0) return;
    _rescaleNormalizedSchedule(sched, durMs);
  }
  ls.setVisemeSchedule(sched, () => (audioCtx.currentTime - startCtxTime) * 1000);
}

function _detachVisemeSchedule() {
  const ls = avatarModule?.avatarState?.lipSync;
  if (ls && typeof ls.clearVisemeSchedule === 'function') ls.clearVisemeSchedule();
}

function _playPcmItem(item) {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  // '!== running' (not '=== suspended'): iOS Safari also has a
  // non-standard 'interrupted' state (Siri, lock screen, route change)
  // that a suspended-only check walks straight past.
  if (audioContext.state !== 'running') audioContext.resume();

  try {
    const audioBuffer = _pcmChunksToAudioBuffer(item.chunks);
    _playAudioBufferItem(item, audioBuffer);
  } catch {
    _setTtsPlaybackActive(false);
    if (pendingVisemeSchedules.length) pendingVisemeSchedules.shift();
    playSentenceQueue();
  }
}

// Shared BufferSource playback for the PCM path and the iOS blob-decode
// path — one implementation so pacing, ducking, and viseme handling can't
// drift between them.
function _playAudioBufferItem(item, audioBuffer) {
  // The pending viseme_schedule FIFO is popped when the sentence's audio
  // starts (_attachVisemeScheduleFromPcm). If start() throws before that,
  // drop this sentence's orphaned schedule so the next sentence doesn't
  // pop a stale one (persistent off-by-one desync).
  let _scheduleConsumed = false;
  try {
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;

    // Route through gain node for ducking if available
    if (ttsGainNode) {
      source.connect(ttsGainNode);
    } else {
      source.connect(audioContext.destination);
    }

    source.onended = () => {
      _setTtsPlaybackActive(false, TTS_PLAYBACK_TAIL_MS);
      _detachVisemeSchedule();
      playedSentenceCount++;
      const pause = _sentencePauseMs(item.text);
      if (pause > 0 && sentenceQueue.length > 0) {
        _sentencePauseTimer = setTimeout(playSentenceQueue, pause);
      } else {
        playSentenceQueue();
      }
    };

    currentAudio = source;
    _setTtsPlaybackActive(true);
    const startCtxTime = audioContext.currentTime;
    source.start();
    _scheduleConsumed = true;
    _attachVisemeScheduleFromPcm(audioContext, startCtxTime, audioBuffer);
  } catch {
    _setTtsPlaybackActive(false);
    // Playback never started — drop this sentence's orphaned schedule so
    // the next sentence doesn't pop a stale one, then skip to it.
    if (!_scheduleConsumed && pendingVisemeSchedules.length) pendingVisemeSchedules.shift();
    playSentenceQueue();
  }
}

function stopPlayback() {
  if (_sentencePauseTimer) { clearTimeout(_sentencePauseTimer); _sentencePauseTimer = null; }
  if (_listeningGraceTimer) { clearTimeout(_listeningGraceTimer); _listeningGraceTimer = null; }
  _setTtsPlaybackActive(false);
  _audioPlaybackBlocked = false;
  document.getElementById('voice-audio-unlock-btn')?.classList.remove('visible');
  sentenceAudioChunks = [];
  sentenceQueue = [];
  pendingVisemeSchedules = [];
  _detachVisemeSchedule();
  isPlaying = false;
  turnDone = false;
  paused = false;
  _updatePauseBtn();
  if (currentAudio) {
    const audioToStop = currentAudio;
    // Audio element uses .pause(), BufferSource uses .stop()
    if (typeof audioToStop.pause === 'function') audioToStop.pause();
    else if (typeof audioToStop.stop === 'function') try { audioToStop.stop(); } catch { /* already stopped */ }
    if (currentAudio === audioToStop) currentAudio = null;
  }
  if (_currentAudioCleanup) {
    const cleanup = _currentAudioCleanup;
    _currentAudioCleanup = null;
    try { cleanup(); } catch { /* already cleaned */ }
  }
  _voiceBusClaim?.release();
  _voiceBusClaim = null;
}

// [7] Pause / Resume — manual button pauses TTS playback without interrupting
let paused = false;

function togglePause() {
  if (paused) {
    resumePlayback();
  } else {
    pausePlayback();
  }
}

// ── Live camera (call surface) ───────────────────────────────────────
// Composites the user's camera into the avatar viewport (camera fill + VRM
// corner PIP) and streams frames over the call WS as video_frame messages.
// Same treatment as the presence widget's eye; scoped to the call.

function _sendCallVideoFrames(frames) {
  if (!Array.isArray(frames) || !frames.length) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try { ws.send(JSON.stringify({ type: 'video_frame', frames })); } catch (_) {}
}

function _setCameraBtnState(on) {
  const btn = document.getElementById('voice-camera-btn');
  if (btn) {
    btn.dataset.camState = on ? 'on' : 'off';
    btn.classList.toggle('active', on);
    const onIcon = btn.querySelector('.voice-camera-on-icon');
    const offIcon = btn.querySelector('.voice-camera-off-icon');
    if (onIcon) onIcon.style.display = on ? '' : 'none';
    if (offIcon) offIcon.style.display = on ? 'none' : '';
    btn.setAttribute('aria-label', on ? 'Turn off camera' : 'Show the companion your camera');
    btn.title = on ? 'Turn off camera' : 'Show camera';
  }
  // Camera is a call-MODE switch: flip the whole surface into full camera
  // mode (camera full-bleed, VRM corner PIP, floating controls). CSS keys
  // off ``camera-mode-active`` on the overlay.
  if (overlay) overlay.classList.toggle('camera-mode-active', on);
}

async function _toggleCallCamera() {
  if (_callCamera) { _stopCallCamera(); return; }
  const host = document.getElementById('voice-avatar-viewport');
  if (!host) { showToast('Avatar view unavailable for camera', 'error', 3000); return; }

  const view = new CompanionCameraView({
    host,
    onStream: (stream) => {
      if (stream) {
        // (Re)point the frame loop at the live stream — flip swaps streams.
        if (!_callVisionLoop) {
          _callVisionLoop = new LiveVisionLoop({
            stream,
            send: (frames) => _sendCallVideoFrames(frames),
            // Don't burn GPU mid-reply or on a hidden tab.
            shouldCapture: () =>
              !!ws && ws.readyState === WebSocket.OPEN
              && (typeof document === 'undefined' || !document.hidden),
          });
          _callVisionLoop.start();
        } else {
          _callVisionLoop.setStream(stream);
        }
      }
    },
    onError: (err) => {
      console.warn('[voice] camera open failed', err);
      showToast('Camera unavailable', 'error', 3000);
    },
  });
  _callCamera = view;
  const ok = await view.start({ facingMode: 'user' });
  if (!ok) { _callCamera = null; return; }
  _setCameraBtnState(true);

  // Reveal the flip control only when there's more than one camera.
  const flipBtn = document.getElementById('voice-camera-flip-btn');
  if (flipBtn) {
    let multi = false;
    try { multi = await view.hasMultipleCameras(); } catch (_) { multi = false; }
    flipBtn.style.display = (multi && _callCamera) ? '' : 'none';
  }
}

function _stopCallCamera() {
  if (_callVisionLoop) { try { _callVisionLoop.stop(); } catch (_) {} _callVisionLoop = null; }
  if (_callCamera) { try { _callCamera.stop(); } catch (_) {} _callCamera = null; }
  const flipBtn = document.getElementById('voice-camera-flip-btn');
  if (flipBtn) flipBtn.style.display = 'none';
  _setCameraBtnState(false);
}

function pausePlayback() {
  if (!isPlaying && sentenceQueue.length === 0) return;
  paused = true;
  if (currentAudio && typeof currentAudio.pause === 'function' && !currentAudio.paused) {
    currentAudio.pause();
    _setTtsPlaybackActive(false);
  }
  _updatePauseBtn();
}

function resumePlayback() {
  paused = false;
  if (currentAudio && typeof currentAudio.play === 'function' && currentAudio.paused && currentAudio.readyState >= 2) {
    currentAudio.play()
      .then(() => _setTtsPlaybackActive(true))
      .catch(() => {});
  } else if (!isPlaying && sentenceQueue.length > 0) {
    playSentenceQueue();
  }
  _updatePauseBtn();
}

function _updatePauseBtn() {
  const btn = document.getElementById('voice-pause-btn');
  if (!btn) return;
  const pauseIcon = btn.querySelector('.voice-pause-icon');
  const resumeIcon = btn.querySelector('.voice-resume-icon');
  if (pauseIcon) pauseIcon.style.display = paused ? 'none' : '';
  if (resumeIcon) resumeIcon.style.display = paused ? '' : 'none';
  btn.classList.toggle('paused', paused);
  btn.title = paused ? 'Resume playback' : 'Pause playback';
}

function sendInterrupt() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    // [6] Send playback position so server can truncate context to what user heard
    ws.send(JSON.stringify({
      type: 'interrupt',
      played_sentences: playedSentenceCount,
      queued_sentences: queuedSentenceCount,
    }));
  }
  paused = false;
  _updatePauseBtn();
  stopDucking();
  stopPlayback();
}

// ---------------------------------------------------------------------------
// Canvas Resize
// ---------------------------------------------------------------------------
function resizeCanvas() {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx = canvas.getContext('2d');
}

// ---------------------------------------------------------------------------
// Orb Visualization — parametric blob with spring physics
// ---------------------------------------------------------------------------
function initOrbState() {
  orbRadius = new Float32Array(ORB_POINTS).fill(1.0);
  orbVelocity = new Float32Array(ORB_POINTS).fill(0);
  orbTarget = new Float32Array(ORB_POINTS).fill(1.0);
  orbMidRadius = new Float32Array(ORB_POINTS).fill(1.0);
  orbMidVelocity = new Float32Array(ORB_POINTS).fill(0);
  orbOuterRadius = new Float32Array(ORB_POINTS).fill(1.0);
  orbOuterVelocity = new Float32Array(ORB_POINTS).fill(0);
  midDelayBuffer = [];
  for (let i = 0; i < MID_DELAY_FRAMES; i++) {
    midDelayBuffer.push(new Float32Array(ORB_POINTS).fill(1.0));
  }
  midDelayIndex = 0;
  const profile = ORB_PROFILES[orbTargetProfile] || ORB_PROFILES.connecting;
  orbEnergy = { ...profile };
}

// ---------------------------------------------------------------------------
// Starfield Background
// ---------------------------------------------------------------------------

let _starsCanvas = null;
let _starsCtx = null;
let _starsAnimId = null;
let _starsResizeHandler = null;
let _starsResizeTimer = null;
let _stars = [];

function _resizeStarfieldCanvas() {
  if (!_starsCanvas || !_starsCtx) return { w: 0, h: 0 };
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  _starsCanvas.width = Math.round(w * dpr);
  _starsCanvas.height = Math.round(h * dpr);
  _starsCanvas.style.width = `${w}px`;
  _starsCanvas.style.height = `${h}px`;
  _starsCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  _starsCtx.imageSmoothingEnabled = true;
  _starsCtx.imageSmoothingQuality = 'high';
  return { w, h };
}

function _starfieldScale(w, h) {
  const referenceArea = 1440 * 900;
  return Math.min(1.75, Math.max(0.55, (w * h) / referenceArea));
}

function _generateStars(w, h) {
  const scale = _starfieldScale(w, h);
  _stars = [];
  const smallCount = Math.round(130 * scale);
  const mediumCount = Math.round(36 * scale);
  const brightCount = Math.round(10 * scale);

  // Small dim stars
  for (let i = 0; i < smallCount; i++) {
    _stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1 + 0.5,
      alpha: Math.random() * 0.36 + 0.14,
      speed: Math.random() * 0.02 + 0.005,
      twinkleSpeed: Math.random() * 0.008 + 0.002,
      twinklePhase: Math.random() * Math.PI * 2,
      color: [255, 255, 255],
    });
  }

  // Medium stars with color tint
  for (let i = 0; i < mediumCount; i++) {
    const tint = Math.random() > 0.5
      ? [200, 220, 255]   // blue-white
      : [255, 230, 200];  // warm white
    _stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.2 + 1,
      alpha: Math.random() * 0.34 + 0.34,
      speed: Math.random() * 0.03 + 0.01,
      twinkleSpeed: Math.random() * 0.01 + 0.003,
      twinklePhase: Math.random() * Math.PI * 2,
      color: tint,
    });
  }

  // Bright stars with glow
  for (let i = 0; i < brightCount; i++) {
    _stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.5 + 1.5,
      alpha: Math.random() * 0.3 + 0.6,
      speed: Math.random() * 0.015 + 0.005,
      twinkleSpeed: Math.random() * 0.015 + 0.005,
      twinklePhase: Math.random() * Math.PI * 2,
      color: [220, 230, 255],
      glow: true,
    });
  }
}

function _initStarfield() {
  _starsCanvas = document.getElementById('voice-stars-canvas');
  if (!_starsCanvas) return;
  _starsCtx = _starsCanvas.getContext('2d');

  if (_starsAnimId) {
    cancelAnimationFrame(_starsAnimId);
    _starsAnimId = null;
  }
  if (_starsResizeHandler) {
    window.removeEventListener('resize', _starsResizeHandler);
    _starsResizeHandler = null;
  }

  const { w, h } = _resizeStarfieldCanvas();
  _generateStars(w, h);

  _starsResizeHandler = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const previousW = Math.max(1, _starsCanvas.width / dpr);
    const previousH = Math.max(1, _starsCanvas.height / dpr);
    const next = _resizeStarfieldCanvas();
    for (const s of _stars) {
      s.x = (s.x / previousW) * next.w;
      s.y = (s.y / previousH) * next.h;
    }
    if (_starsResizeTimer) window.clearTimeout(_starsResizeTimer);
    _starsResizeTimer = window.setTimeout(() => {
      _generateStars(next.w, next.h);
      _starsResizeTimer = null;
    }, 120);
  };
  window.addEventListener('resize', _starsResizeHandler, { passive: true });

  _animateStarfield();
}

function _animateStarfield() {
  if (!_starsCtx || !_starsCanvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = _starsCanvas.width / dpr;
  const h = _starsCanvas.height / dpr;
  _starsCtx.clearRect(0, 0, w, h);

  const t = Date.now() * 0.001;

  for (const s of _stars) {
    // Slow drift
    s.x += s.speed * 0.3;
    s.y += s.speed * 0.15;
    if (s.x > w + 5) s.x = -5;
    if (s.y > h + 5) s.y = -5;

    // Twinkle
    const twinkle = Math.sin(t * s.twinkleSpeed * 10 + s.twinklePhase) * 0.3 + 0.7;
    const a = s.alpha * twinkle;

    const [r, g, b] = s.color;

    if (s.glow) {
      // Draw glow halo first
      const grad = _starsCtx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r * 4);
      grad.addColorStop(0, `rgba(${r},${g},${b},${a * 0.3})`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      _starsCtx.fillStyle = grad;
      _starsCtx.beginPath();
      _starsCtx.arc(s.x, s.y, s.r * 4, 0, Math.PI * 2);
      _starsCtx.fill();
    }

    // Draw star
    _starsCtx.fillStyle = `rgba(${r},${g},${b},${a})`;
    _starsCtx.beginPath();
    _starsCtx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
    _starsCtx.fill();
  }

  _starsAnimId = requestAnimationFrame(_animateStarfield);
}

function _stopStarfield() {
  if (_starsAnimId) {
    cancelAnimationFrame(_starsAnimId);
    _starsAnimId = null;
  }
  if (_starsResizeHandler) {
    window.removeEventListener('resize', _starsResizeHandler);
    _starsResizeHandler = null;
  }
  if (_starsResizeTimer) {
    window.clearTimeout(_starsResizeTimer);
    _starsResizeTimer = null;
  }
}

function startOrb() {
  if (!analyserNode || !canvas) return;
  resizeCanvas();
  if (!ctx) return;
  initOrbState();
  _orbWrapEl = document.querySelector('.voice-orb-wrap');

  // Orb uses mic analyser for user speech visualization (TTS analyser is for lip sync)
  const orbAnalyser = analyserNode._micAnalyser || analyserNode;
  const frequencyData = new Uint8Array(orbAnalyser.frequencyBinCount);
  const startTime = performance.now() / 1000;

  function drawOrb() {
    animFrameId = requestAnimationFrame(drawOrb);

    // Skip orb rendering when avatar is active (Three.js handles its own canvas)
    if (avatarModule.avatarState.active) return;

    const enrollEl = document.getElementById('voice-enrollment');
    if (enrollEl && !enrollEl.hidden) return;

    const now = performance.now() / 1000;
    const time = now - startTime;

    // Use mic analyser for orb (responds to user speech), TTS analyser for lip sync
    const activeAnalyser = (orbTargetProfile === 'speaking' && analyserNode) ? analyserNode : orbAnalyser;
    activeAnalyser.getByteFrequencyData(frequencyData);

    // Pipe audio energy to CSS custom property for ring reactivity
    let rmsSum = 0;
    for (let i = 0; i < frequencyData.length; i++) rmsSum += frequencyData[i];
    const rmsNorm = Math.min(1.0, (rmsSum / frequencyData.length) / 128);
    // Smooth the value to avoid jitter
    _orbRmsSmoothed = _orbRmsSmoothed * 0.85 + rmsNorm * 0.15;
    // Set CSS custom property on the orb wrapper for ring scale
    if (_orbWrapEl) {
      _orbWrapEl.style.setProperty('--audio-energy', _orbRmsSmoothed.toFixed(3));
    }

    const width = canvas.width;
    const height = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    const cx = width / 2;
    const cy = height / 2;
    const maxR = Math.min(cx, cy) * 0.38;

    ctx.clearRect(0, 0, width, height);

    // Interpolate energy toward target profile
    const target = ORB_PROFILES[orbTargetProfile] || ORB_PROFILES.connecting;
    const easeSpeed = 0.04;
    orbEnergy.stiffness = lerpVal(orbEnergy.stiffness, target.stiffness, easeSpeed);
    orbEnergy.damping = lerpVal(orbEnergy.damping, target.damping, easeSpeed);
    orbEnergy.breathHz = lerpVal(orbEnergy.breathHz, target.breathHz, easeSpeed);
    orbEnergy.audioGain = lerpVal(orbEnergy.audioGain, target.audioGain, easeSpeed);
    orbEnergy.radiusScale = lerpVal(orbEnergy.radiusScale, target.radiusScale, easeSpeed);
    orbEnergy.shadowBlur = lerpVal(orbEnergy.shadowBlur, target.shadowBlur, easeSpeed);
    orbEnergy.hslInner = lerpHSL(orbEnergy.hslInner, target.hslInner, easeSpeed);
    orbEnergy.hslOuter = lerpHSL(orbEnergy.hslOuter, target.hslOuter, easeSpeed);

    const baseR = maxR * orbEnergy.radiusScale;
    const breathAmp = baseR * 0.05;
    const breathing = Math.sin(time * orbEnergy.breathHz * ORB_TWO_PI) * breathAmp;
    const stiffness = orbEnergy.stiffness;
    const damping = orbEnergy.damping;
    const audioGain = orbEnergy.audioGain;

    const binCount = frequencyData.length;
    const usableBins = Math.floor(binCount * 0.5);
    const halfPoints = ORB_POINTS / 2;

    // Pre-compute mirrored audio values — symmetric around the circle
    // Maps first half of points to freq bins, mirrors to second half
    const audioValues = new Float32Array(ORB_POINTS);
    for (let i = 0; i < halfPoints; i++) {
      const binIndex = Math.floor((i / halfPoints) * usableBins);
      const val = (frequencyData[binIndex] || 0) / 255;
      audioValues[i] = val;
      audioValues[ORB_POINTS - 1 - i] = val;  // mirror
    }

    // TTS synthetic pulse — advance once per frame
    if (ttsPulseActive) ttsPulsePhase += 0.005;

    for (let i = 0; i < ORB_POINTS; i++) {
      const angle = (i / ORB_POINTS) * ORB_TWO_PI;
      const audioVal = audioValues[i];

      let pulseVal = 0;
      if (ttsPulseActive) {
        pulseVal = (Math.sin(ttsPulsePhase * ORB_TWO_PI * 2) * 0.5 + 0.5) * 0.4;
        pulseVal *= (0.6 + Math.sin(angle * 3 + time) * 0.4);
      }

      const audioDisp = Math.max(audioVal, pulseVal) * audioGain * baseR * 0.45;
      const noise = orbNoise(angle, time) * baseR * 0.035;

      orbTarget[i] = baseR + breathing + audioDisp + noise;

      orbVelocity[i] += (orbTarget[i] - orbRadius[i]) * stiffness;
      orbVelocity[i] *= damping;
      orbRadius[i] += orbVelocity[i];
    }

    // Mid ring delay buffer
    midDelayBuffer[midDelayIndex] = Float32Array.from(orbTarget);
    midDelayIndex = (midDelayIndex + 1) % MID_DELAY_FRAMES;
    const delayedTargets = midDelayBuffer[midDelayIndex];

    for (let i = 0; i < ORB_POINTS; i++) {
      orbMidVelocity[i] += (delayedTargets[i] - orbMidRadius[i]) * stiffness * 0.8;
      orbMidVelocity[i] *= damping;
      orbMidRadius[i] += orbMidVelocity[i];
    }

    // Outer halo (heavily smoothed)
    const avgTarget = orbTarget.reduce((a, b) => a + b, 0) / ORB_POINTS;
    for (let i = 0; i < ORB_POINTS; i++) {
      orbOuterVelocity[i] += (avgTarget - orbOuterRadius[i]) * 0.03;
      orbOuterVelocity[i] *= 0.92;
      orbOuterRadius[i] += orbOuterVelocity[i];
    }

    // Gradient — processing state orbits center
    let gx = cx, gy = cy;
    if (orbTargetProfile === 'processing') {
      const orbitR = baseR * 0.1;
      gx = cx + Math.cos(time * ORB_TWO_PI * 0.25) * orbitR * dpr;
      gy = cy + Math.sin(time * ORB_TWO_PI * 0.25) * orbitR * dpr;
    }

    const gradient = ctx.createRadialGradient(gx, gy, 0, cx, cy, baseR * dpr);
    gradient.addColorStop(0, hslStr(orbEnergy.hslInner, 1.0));
    gradient.addColorStop(1, hslStr(orbEnergy.hslOuter, 0.6));

    // Layer 3: Outer halo
    ctx.save();
    ctx.globalAlpha = 0.08;
    ctx.shadowColor = hslStr(orbEnergy.hslInner, 0.5);
    ctx.shadowBlur = orbEnergy.shadowBlur * dpr;
    ctx.fillStyle = gradient;
    drawBlobPath(ctx, cx, cy, orbOuterRadius, 1.0, dpr);
    ctx.fill();
    ctx.restore();

    // Layer 2: Mid ring
    ctx.save();
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = gradient;
    drawBlobPath(ctx, cx, cy, orbMidRadius, 0.7, dpr);
    ctx.fill();
    ctx.restore();

    // Layer 1: Inner core
    ctx.save();
    ctx.globalAlpha = 0.80;
    ctx.fillStyle = gradient;
    drawBlobPath(ctx, cx, cy, orbRadius, 0.4, dpr);
    ctx.fill();
    ctx.restore();
  }

  drawOrb();
}

function drawBlobPath(ctx, cx, cy, radii, layerScale, dpr) {
  ctx.beginPath();
  const n = radii.length;
  const points = [];
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * ORB_TWO_PI - Math.PI / 2;
    const r = radii[i] * layerScale * dpr;
    points.push({
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
    });
  }

  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 0; i < n; i++) {
    const curr = points[i];
    const next = points[(i + 1) % n];
    const prev = points[(i - 1 + n) % n];
    const next2 = points[(i + 2) % n];

    const tension = 0.3;
    const cp1x = curr.x + (next.x - prev.x) * tension;
    const cp1y = curr.y + (next.y - prev.y) * tension;
    const cp2x = next.x - (next2.x - curr.x) * tension;
    const cp2y = next.y - (next2.y - curr.y) * tension;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, next.x, next.y);
  }
  ctx.closePath();
}

function stopOrb() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  if (ctx && canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// ---------------------------------------------------------------------------
// Chat Integration
// ---------------------------------------------------------------------------
/** Fire-and-forget scene image generation — runs in background, delivers to transcript. */
function _generateSceneImage(btn) {
  if (!app.state.currentSessionId) return;
  btn.disabled = true;
  btn.classList.add('active');
  // Start polling the image queue so the camera button reflects the
  // *real* state (model loading vs queued vs generating step N) rather
  // than a generic spinner.
  const stopPolling = _pollSceneImageStatus(btn);

  const mode = app.state.mode || 'passthrough';
  const finishOnce = () => {
    stopPolling();
    btn.disabled = false;
    btn.classList.remove('active');
    btn.removeAttribute('data-img-stage');
    btn.title = 'Generate scene image';
  };
  if (mode === 'narrative') {
    _generateNarrativeScene(btn, finishOnce);
  } else {
    _generateFromConversation(btn, finishOnce);
  }
}

/** Poll /api/image/generation-status while the camera button is active.
 *  Updates btn.title with stage + elapsed. Returns a cancel function. */
function _pollSceneImageStatus(btn) {
  const t0 = Date.now();
  let cancelled = false;
  let timer = null;

  const tick = async () => {
    if (cancelled) return;
    try {
      const resp = await fetch('/api/image/generation-status');
      if (resp.ok) {
        const data = await resp.json();
        const elapsed = Math.floor((Date.now() - t0) / 1000);
        let label;
        if (!data.active) {
          label = data.queue_size
            ? `Queued (${data.queue_size} ahead) · ${elapsed}s`
            : `Preparing… · ${elapsed}s`;
        } else {
          // Prefer pre-queue stage (distiller running) over the job
          // stage so the 3-10s LLM call shows correctly. Include the
          // step counter when in the diffusion phase so the user
          // sees progress per second instead of static "Generating".
          const stage = (data.pre_queue?.stage || data.stage || data.status || 'generating').trim();
          const parts = [stage];
          if (data.steps_total > 0) {
            parts.push(`step ${data.steps_done}/${data.steps_total}`);
          }
          parts.push(`${elapsed}s`);
          label = parts.join(' · ');
        }
        btn.title = label;
        btn.dataset.imgStage = (data.stage || data.status || '').toLowerCase();
      }
    } catch { /* network blip — keep trying */ }
    if (!cancelled) timer = setTimeout(tick, 800);
  };
  // First poll after a short delay so server has time to enqueue the job.
  timer = setTimeout(tick, 300);

  return () => { cancelled = true; if (timer) clearTimeout(timer); };
}

/** Narrative mode: use distiller pipeline (has character card, world state, etc.) */
function _generateNarrativeScene(btn, done) {
  const session = chat.getActiveSession();
  const messages = session ? chat.buildMessagesForAPI(session) : [];

  fetch('/api/image/generate-scene', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: app.state.currentSessionId,
      instruction: 'the current scene',
      messages: messages.slice(-6),
    }),
  })
    .then(resp => resp.ok ? resp.json() : Promise.reject(resp.statusText))
    .then(data => { if (data.url) _showVoiceImage(data.url); })
    .catch(() => { /* silent */ })
    .finally(done);
}

/** Non-narrative modes: ask LLM to condense conversation into a prompt, then generate. */
async function _generateFromConversation(btn, done) {
  const session = chat.getActiveSession();
  const messages = session ? chat.buildMessagesForAPI(session) : [];
  const recent = messages.slice(-6).map(m => `${m.role}: ${m.content}`).join('\n');
  const summary = recent.slice(-1500) || 'a scene';

  try {
    // Step 1: LLM condenses conversation into image prompt
    const enhanceResp = await fetch('/api/image/enhance-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: `Based on this conversation, describe the scene visually:\n${summary}` }),
    });
    if (!enhanceResp.ok) throw new Error(enhanceResp.statusText || `enhance ${enhanceResp.status}`);
    const data = await enhanceResp.json();
    const prompt = data.prompt || summary;

    // Step 2: Generate image from condensed prompt
    const imageResp = await fetch('/api/image/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    if (!imageResp.ok) throw new Error(imageResp.statusText || `generate ${imageResp.status}`);
    const imageData = await imageResp.json();
    const id = imageData.image_id || imageData.id || '';
    if (id) _showVoiceImage(`/api/image/${id}`);
  } catch (err) {
    console.warn('[voice] conversation image generation failed', err);
  } finally {
    done();
  }
}

/** Show a generated image in the conversation log. */
/** Build the text to PERSIST for an assistant voice turn: spoken text plus any
 *  tool-result images (image_search / image_generation) appended as markdown.
 *  The live overlay shows images as cards and strips them from the spoken text,
 *  but the saved turn must keep them — they live in tool metadata, not the
 *  text — or they vanish from chat + history. Dedup against text + cap at 6. */
function _assistantSaveText(text) {
  let out = (text || '').trim();
  const extras = [];
  for (const u of _turnImages) {
    if (u && !out.includes(u) && !extras.includes(u)) extras.push(u);
    if (extras.length >= 6) break;
  }
  if (extras.length) {
    out += (out ? '\n\n' : '') + extras.map(u => `![](${u})`).join('\n');
  }
  return out;
}

/** Render image_search results as a tappable grid in the voice transcript log.
 *  Mirrors _showVoiceVideos; tapping opens the full image in a new tab. */
function _showVoiceImages(images) {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;
  // One image result set visible at a time — drop previous image/video cards.
  log.querySelectorAll('.voice-image-grid, .voice-image-card').forEach(el => el.remove());

  const wrap = document.createElement('div');
  wrap.className = 'voice-image-grid voice-log-bubble';
  const top = images.slice(0, 6);
  wrap.innerHTML = top.map((im) => {
    // image_search downloads to a local artifact and returns embed_url
    // (/api/artifacts/.../download); fall back through the other URL keys.
    const src = (im && (im.embed_url || im.download_url || im.url || im.thumbnail || im.source_url)) || '';
    const link = (im && (im.source_url || im.embed_url || im.url)) || src;
    const title = (im && im.title) || '';
    return `
      <a class="voice-image-grid-cell" href="${escapeHtml(link)}" target="_blank" rel="noopener"
         title="${escapeHtml(title)}">
        <img src="${escapeHtml(src)}" alt="" loading="lazy"
             onerror="this.closest('.voice-image-grid-cell').style.display='none'">
      </a>`;
  }).join('');
  log.appendChild(wrap);
  _scrollLog(log);
}

function _showVoiceImage(imageUrl) {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;

  // Remove any existing image cards — only 1 visible at a time
  log.querySelectorAll('.voice-image-card').forEach(el => el.remove());

  const card = document.createElement('div');
  card.className = 'voice-image-card voice-log-bubble';
  card.innerHTML = `
    <button class="voice-image-dismiss" aria-label="Dismiss image">&times;</button>
    <img src="${escapeHtml(imageUrl)}" alt="Generated image" loading="lazy"
         onerror="this.parentElement.style.display='none'">
  `;
  card.querySelector('.voice-image-dismiss').addEventListener('click', (e) => {
    e.stopPropagation();
    card.remove();
  });
  card.addEventListener('click', () => window.open(imageUrl, '_blank'));
  log.appendChild(card);
  _scrollLog(log);
}

/** Render up to 3 YouTube search candidates as cards in the voice transcript log.
 *  Tap (or "play the first one"-style voice command — Stage 2) hands the chosen
 *  video off to the YouTube panel via the existing `media:play` event, which
 *  loads the embed + scroll-synced transcript. The voice overlay auto-minimizes
 *  so the picker doesn't sit on top of the player.
 */
function _showVoiceVideos(videos) {
  const log = document.querySelector('.voice-transcript-log');
  if (!log) return;
  // Only one picker visible at a time — drop any previous video/image cards.
  log.querySelectorAll('.voice-video-list, .voice-image-card').forEach(el => el.remove());

  const wrap = document.createElement('div');
  wrap.className = 'voice-video-list voice-log-bubble';
  const top = videos.slice(0, 3);
  wrap.innerHTML = top.map((v, i) => `
    <button class="voice-video-card" data-video-id="${escapeHtml(v.video_id || '')}" type="button">
      <span class="voice-video-card-num">${i + 1}</span>
      <img class="voice-video-card-thumb"
           src="${escapeHtml(v.thumbnail || '')}"
           alt=""
           loading="lazy"
           onerror="this.style.visibility='hidden'">
      <span class="voice-video-card-meta">
        <span class="voice-video-card-title">${escapeHtml(v.title || 'Untitled')}</span>
        <span class="voice-video-card-sub">${escapeHtml(v.channel || '')}${v.duration ? ' · ' + escapeHtml(v.duration) : ''}</span>
      </span>
    </button>
  `).join('');

  wrap.addEventListener('click', (e) => {
    const card = e.target.closest('.voice-video-card');
    if (!card) return;
    const videoId = card.dataset.videoId;
    if (!videoId) return;
    const chosen = top.find(v => v.video_id === videoId);
    if (chosen) _playVoiceVideo(chosen);
  });

  log.appendChild(wrap);
  _scrollLog(log);
}

/** Hand a chosen video off to the YouTube panel + minimize the voice overlay
 *  so the user can actually see the player. Uses the existing `media:play`
 *  event the YouTube panel already listens for.
 *
 *  Dispatches on `window` (not `document`) — youtube-panel.js's listener is
 *  registered with `window.addEventListener('media:play', …)`, and custom
 *  events don't bubble from `document` to `window`, so a document dispatch
 *  silently no-ops. */
async function _playVoiceVideo(video) {
  if (!video || !video.video_id) return;
  // youtube-panel.js registers its `media:play` listener at module load, but
  // it's LAZY-imported (app.js only loads it when the media surface is
  // opened). In a voice call where the user never opened media, the module
  // isn't loaded, so dispatching media:play would no-op — the call would
  // minimize but the player never appears. Import it first so the listener
  // exists before we dispatch.
  try { await import('./youtube-panel.js'); } catch { /* panel unavailable */ }
  // youtube-panel.js's `media:play` listener calls openFromSearch, which
  // resolves the panel DOM, removes its `hidden` class, switches to the
  // discover tab, creates the player, and starts the transcript fetch.
  // Single dispatch is sufficient — no companion `media:open-panel`
  // needed.
  window.dispatchEvent(new CustomEvent('media:play', {
    detail: {
      videoId: video.video_id,
      title: video.title || '',
      channel: video.channel || video.author || '',
    },
  }));
  // Auto-minimize so the YouTube panel is visible.
  if (typeof minimizeVoiceCall === 'function' && isConnected && !isMinimized) {
    try { minimizeVoiceCall(); } catch { /* ignore */ }
  }
}

/** Fallback: fetch the latest image from the library and show it. */
async function _fetchAndShowVoiceImage() {
  try {
    const resp = await fetch('/api/image/history?limit=1&sort=newest');
    if (!resp.ok) return;
    const data = await resp.json();
    const entry = (data.entries || [])[0];
    if (!entry) return;
    _showVoiceImage(`/api/image/${entry.image_id}`);
  } catch { /* silent */ }
}

function addMessageToChat(role, text) {
  document.dispatchEvent(new CustomEvent('augmentum:voice-message', {
    detail: { role, text, sessionId: _callSessionId }
  }));
}

// ---------------------------------------------------------------------------
// Voice Enrollment
// ---------------------------------------------------------------------------
let enrollmentEl = null;
let enrollmentPhrases = [];
let enrollmentStep = 0;         // 0, 1, 2
let enrollmentRecorder = null;
let enrollmentRecording = false;
let enrollmentSamples = [];     // Array of Blob

async function checkAndShowEnrollment(needsEnrollment) {
  enrollmentEl = document.getElementById('voice-enrollment');
  if (!enrollmentEl) return;

  if (!needsEnrollment) {
    enrollmentEl.hidden = true;
    return;
  }

  // Fetch phrases from the server, with hardcoded fallback
  try {
    const resp = await fetch('/api/voice/enrollment/phrases');
    if (resp.ok) {
      const data = await resp.json();
      enrollmentPhrases = data.phrases || [];
    }
  } catch { /* use fallback */ }

  if (!enrollmentPhrases.length) {
    enrollmentPhrases = [
      'The quick brown fox jumps over the lazy dog near the riverbank.',
      'She sells seashells by the seashore on a beautiful sunny morning.',
      'How vexingly quick daft zebras jump over the bright yellow fence.',
      'I enjoy listening to music and reading books on rainy afternoons.',
      'Please remember to pick up some milk and bread on your way home.',
    ];
  }

  // Populate phrase text into the DOM
  const phraseEls = enrollmentEl.querySelectorAll('.voice-enrollment-phrase');
  phraseEls.forEach((el, i) => {
    const textEl = el.querySelector('.phrase-text');
    if (textEl) {
      textEl.textContent = enrollmentPhrases[i] || '';
    }
  });

  enrollmentStep = 0;
  enrollmentSamples = [];
  _updateEnrollmentUI();

  // Show the enrollment modal
  enrollmentEl.hidden = false;

  // Wire buttons — use addEventListener for reliable touch support
  const recordBtn = document.getElementById('enrollment-record-btn');
  const skipBtn = document.getElementById('enrollment-skip-btn');

  // Remove previous listeners by cloning
  if (recordBtn) {
    const newRecord = recordBtn.cloneNode(true);
    recordBtn.parentNode.replaceChild(newRecord, recordBtn);
    newRecord.addEventListener('click', _handleEnrollRecordTap);
    newRecord.addEventListener('touchend', (e) => {
      e.preventDefault();
      _handleEnrollRecordTap();
    });
  }

  if (skipBtn) {
    const newSkip = skipBtn.cloneNode(true);
    skipBtn.parentNode.replaceChild(newSkip, skipBtn);
    newSkip.addEventListener('click', _handleEnrollSkipTap);
    newSkip.addEventListener('touchend', (e) => {
      e.preventDefault();
      _handleEnrollSkipTap();
    });
  }
}

function _handleEnrollRecordTap() {
  if (enrollmentRecording) {
    _stopEnrollmentRecording();
  } else {
    _startEnrollmentRecording();
  }
}

async function _handleEnrollSkipTap() {
  if (!enrollmentEl) return;
  enrollmentEl.hidden = true;
  enrollmentStep = 0;
  enrollmentSamples = [];
  try {
    await fetch('/api/voice/enrollment/decline', { method: 'POST' });
  } catch { /* best-effort */ }
  if (isConnected) {
    // We're in an active voice call — resume it without enrollment
    setState('listening');
  } else {
    // Opened enrollment outside a voice call (e.g. from Settings)
    overlay.classList.remove('active');
    cleanupMic();
  }
}

function _updateEnrollmentUI() {
  if (!enrollmentEl) return;

  const phraseEls = enrollmentEl.querySelectorAll('.voice-enrollment-phrase');
  phraseEls.forEach((el, i) => {
    el.classList.remove('active', 'recording', 'done');
    if (i < enrollmentStep) {
      el.classList.add('done');
    } else if (i === enrollmentStep) {
      el.classList.add('active');
    }
  });

  // Progress bar
  const fill = enrollmentEl.querySelector('.voice-enrollment-fill');
  if (fill) {
    fill.style.width = `${(enrollmentStep / 5) * 100}%`;
  }

  // Record button text
  const recordBtn = document.getElementById('enrollment-record-btn');
  if (recordBtn) {
    const span = recordBtn.querySelector('span');
    if (span && enrollmentStep < 5) {
      span.textContent = `Record Phrase ${enrollmentStep + 1}`;
    }
  }
}

async function _startEnrollmentRecording() {
  if (enrollmentRecording || enrollmentStep >= 5) return;

  // Ensure mic access — may not exist if enrollment triggered from settings
  if (!micStream) {
    try {
      micStream = await acquireMic({ usage: 'enrollment' });
    } catch {
      showToast('Microphone access denied', 'error');
      return;
    }
  }

  enrollmentRecording = true;

  const phraseEls = enrollmentEl.querySelectorAll('.voice-enrollment-phrase');
  if (phraseEls[enrollmentStep]) {
    phraseEls[enrollmentStep].classList.add('recording');
    phraseEls[enrollmentStep].classList.remove('active');
  }

  const recordBtn = document.getElementById('enrollment-record-btn');
  if (recordBtn) {
    recordBtn.classList.add('recording');
    const span = recordBtn.querySelector('span');
    if (span) span.textContent = 'Stop Recording';
  }

  // Record using MediaRecorder (WAV not supported, use webm and let server handle)
  const chunks = [];
  enrollmentRecorder = new MediaRecorder(micStream, {
    mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm'
  });

  enrollmentRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  enrollmentRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'audio/webm' });
    enrollmentSamples.push(blob);

    const phraseEls = enrollmentEl.querySelectorAll('.voice-enrollment-phrase');
    if (phraseEls[enrollmentStep]) {
      phraseEls[enrollmentStep].classList.remove('recording');
      phraseEls[enrollmentStep].classList.add('done');
    }

    enrollmentStep++;
    enrollmentRecording = false;

    const recordBtn = document.getElementById('enrollment-record-btn');
    if (recordBtn) recordBtn.classList.remove('recording');

    if (enrollmentStep >= 5) {
      _submitEnrollment();
    } else {
      _updateEnrollmentUI();
    }
  };

  enrollmentRecorder.start();

  // Auto-stop after 6 seconds
  setTimeout(() => {
    if (enrollmentRecording && enrollmentRecorder?.state === 'recording') {
      _stopEnrollmentRecording();
    }
  }, 6000);
}

function _stopEnrollmentRecording() {
  if (!enrollmentRecording || !enrollmentRecorder) return;
  enrollmentRecording = false;
  if (enrollmentRecorder.state === 'recording') {
    enrollmentRecorder.stop();
  }
}

async function _submitEnrollment() {
  const recordBtn = document.getElementById('enrollment-record-btn');
  if (recordBtn) {
    recordBtn.classList.add('processing');
    const span = recordBtn.querySelector('span');
    if (span) span.textContent = 'Processing...';
  }

  const fill = enrollmentEl.querySelector('.voice-enrollment-fill');
  if (fill) fill.style.width = '100%';

  // Submit the 5 samples to the server
  const formData = new FormData();
  enrollmentSamples.forEach((blob, i) => {
    formData.append(`sample${i + 1}`, blob, `phrase${i + 1}.webm`);
  });

  try {
    const resp = await fetch('/api/voice/enrollment', {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();

    if (data.enrolled) {
      // Show success
      const actions = enrollmentEl.querySelector('.voice-enrollment-actions');
      const result = enrollmentEl.querySelector('.voice-enrollment-result');
      const quality = enrollmentEl.querySelector('.enrollment-quality');

      if (actions) actions.hidden = true;
      if (result) result.hidden = false;
      if (quality) {
        const pct = Math.round((data.quality || 0) * 100);
        quality.textContent = `Quality: ${pct}% · ${data.samples} samples`;
      }

      // Auto-dismiss after 2 seconds and resume voice call
      setTimeout(() => {
        if (enrollmentEl) enrollmentEl.hidden = true;
        if (isConnected) {
          // Resume the active voice call now that enrollment is done
          setState('listening');
        } else {
          // Opened enrollment outside a voice call (e.g. from Settings)
          overlay.classList.remove('active');
          cleanupMic();
        }
      }, 2000);
    } else {
      showToast(data.error || 'Enrollment failed — please try again from Settings', 'error');
      // Dismiss enrollment UI instead of looping
      if (enrollmentEl) enrollmentEl.hidden = true;
      if (isConnected) {
        setState('listening');
      } else {
        overlay.classList.remove('active');
        cleanupMic();
      }
      enrollmentStep = 0;
      enrollmentSamples = [];
      if (recordBtn) recordBtn.classList.remove('processing');
    }
  } catch (err) {
    showToast('Enrollment failed: ' + err.message, 'error');
    if (enrollmentEl) enrollmentEl.hidden = true;
    if (isConnected) {
      setState('listening');
    } else {
      overlay.classList.remove('active');
      cleanupMic();
    }
    enrollmentStep = 0;
    enrollmentSamples = [];
    if (recordBtn) recordBtn.classList.remove('processing');
  }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
// Expose enrollment trigger for settings panel
window.voiceCheckEnrollment = () => checkAndShowEnrollment(true);

// Deferred trigger — covers the edge case where settings.js fires the
// enroll request before voice.js has finished its module eval (in which
// case window.voiceCheckEnrollment wouldn't exist yet at dispatch time).
window.addEventListener('voice-enroll-request', () => checkAndShowEnrollment(true));

export const voice = {
  get isConnected() { return isConnected; },
  endVoiceCall,
  openVrEntry,
  /** Enable with: voice._debug = true in console */
  _debug: false,
};
