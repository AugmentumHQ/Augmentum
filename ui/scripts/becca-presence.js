/**
 * becca-presence.js
 *
 * Mounts the persistent .becca-presence widget when companion_persona_mode
 * is on. The widget owns DOM, drag, surface adaptation, real-talk pill,
 * and bus-driven glow. VRM rendering itself is delegated to the unified
 * avatar.js pipeline via ``activateAvatarStandalone`` — same scene/
 * camera/renderer/animator/presence stack as a voice call, just without
 * the audio-analyser binding and the cinematic experience layers. When
 * a call later starts, the same canvas can be reparented into the call
 * viewport (no re-instantiation).
 *
 * Responsibilities:
 *   - Mount / unmount the widget DOM
 *   - Activate the unified VRM render path into the widget stage
 *   - Surface adaptation (fullscreen / screen-share / discreet / mobile)
 *   - Drag-to-reposition with per-surface persistence
 *   - Real-talk pill always visible
 *   - Glow attribute driven by bus events
 *   - The Real-Talk panel
 *
 * Pose orchestration lives in xr-companion-binding.js as before — the
 * unified PresenceEngine drives breath / blink / micro-gestures
 * autonomously.
 */

import { Raycaster, Vector2 } from 'three';
import {
  activateAvatarStandalone,
  deactivateAvatar,
  avatarState,
  onLLMDelta,
  onStateChange,
  onTtsPlaybackChange,
  onUserTranscript,
  movementConductor,
  pauseAvatarRender,
  resumeAvatarRender,
  setAvatarFrameCap,
} from './avatar.js';
import { BeccaPttSession } from './becca-ptt.js';
import { bus } from './activity-bus.js';
import { BeccaWakeSession } from './becca-wake.js';
import { CompanionCameraView } from './companion-camera.js';
import { CompanionAnimationRouter } from './companion-animation-router.js';
import {
  registerUserAnimations,
  registerAtlasOverrides,
  listEffectiveEntries,
  listRoles,
  listFamilies,
  familyOf,
} from './anim-atlas.js';
import { createLifetime } from './_lifecycle.js';
import { createReconnector } from './_ws-reconnect.js';
import { scheduleAutosize } from './utils/textarea-autosize.js';
// Companion brief panel — registers window.openCompanionBrief (called by the
// surface-event router on coder_run_completed) + a __previewBrief dev hook.
import './brief-panel.js';
import { userScopedKey } from './auth.js';
import { getSettings, syncVoicePrefsToBackend } from './settings.js';
import { setCompanionVoiceGain } from './chat/tts.js';

// Per-user localStorage (multi-tenant fix 2026-06). The widget's
// avatar/voice/size/wake prefs are one user's choices — namespacing the
// keys by the logged-in user id keeps them from leaking into another
// tenant's session on a shared browser. Follows the userScopedKey
// contract: when no user is known yet (pre-login boot), skip the
// read/write rather than fall back to the bare (leaky) key.
function _uGet(base) {
  const k = userScopedKey(base);
  if (!k) return null;
  try { return localStorage.getItem(k); } catch (_) { return null; }
}
function _uSet(base, val) {
  const k = userScopedKey(base);
  if (!k) return;
  try { localStorage.setItem(k, val); } catch (_) { /* private mode / quota */ }
}
function _uRemove(base) {
  const k = userScopedKey(base);
  if (!k) return;
  try { localStorage.removeItem(k); } catch (_) { /* non-critical */ }
}

const STORAGE_KEY_PREFIX = 'becca.presence.pos.';
const BECCA_VRM_URL = '/api/avatar/bundled_f_becca.vrm';
const AUDIO_ROLE_STORAGE_KEY = 'becca.presence.audio_role';
// localStorage cache key for dance history. Server
// (/api/dance/history) is authoritative; this cache only seeds the
// in-memory list on cold load so the timeline panel renders instantly
// before the GET completes. Cap is the per-device in-memory limit;
// the server keeps a larger retention window so the cap is a render-
// budget concern, not a data-retention one.
const DANCE_HISTORY_STORAGE_KEY = 'becca.dance.history';
const DANCE_HISTORY_CAP = 50;

// Channel-exit helper exposed on window so any channel surface (narrative,
// coder, agentic, bug_finder) can call back into companion runtime when
// the user leaves the channel. The handoff session_id is provided by
// whatever surface mounted the channel. Lane 3 §3 — this is the
// re-engagement entry point that triggers Becca's return microcopy.
window.augmentumExitCompanionChannel = async function exitCompanionChannel(sessionId, exitReason = 'user_explicit') {
  if (!sessionId) return null;
  try {
    const resp = await fetch('/api/companion/channel_exit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ session_id: sessionId, exit_reason: exitReason }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (_) {
    return null;
  }
};
const SIZE_STORAGE_KEY = 'becca.presence.size';
const RESIZE_MIN_W = 240, RESIZE_MIN_H = 320;
const RESIZE_MAX_W = 800, RESIZE_MAX_H = 1000;
const AVATAR_CHOICE_STORAGE_KEY = 'becca.presence.avatar';

// Mount-scoped lifetime: tracks every DOM listener, observer, and
// timer that should be torn down on unmount. Newly added handlers
// MUST register through this rather than module-scope detach refs —
// the legacy pattern (one ``_detachX()`` per handler) accumulated
// silent leaks because individual sites stopped being mirrored.
// See ui/scripts/_lifecycle.js for the helper contract.
let _lifetime = null;
// Tracks whether an avatar VRM swap is in flight; gates the picker
// tiles so rapid double-clicks can't fork the standalone pipeline.
let _swapInFlight = false;
// Tracks the return-flash auto-clear timer so back-to-back returns
// don't have an earlier timer wipe a freshly-set bubble.
let _returnFlashTimer = null;

let _root = null;
let _stage = null;
let _statusRow = null;
let _wakeFlash = null;
let _returnFlash = null;            // transient bubble for channel return microcopy
let _channelClearTimer = null;      // 15-min safety reset if channel.exited never arrives
let _activityClearTimer = null;     // 5s clear for activity-chosen status row labels
let _viewportClampHandler = null;   // window-resize listener (debounced)
let _viewportClampTimer = null;
let _audioModeBtn = null;
let _micDspBtn = null;
// Per-user, device-local: disable the browser's mic echo-cancellation so it
// stops muffling music/media output while the companion is listening. Mirrors
// the mic deviceId preference (also device-local) — which physical mic + its
// DSP is a property of THIS browser, not a cross-device account setting.
const MIC_AEC_DISABLED_KEY = 'becca.mic.aecDisabled';
function _micAecDisabled() {
  try { return _uGet(MIC_AEC_DISABLED_KEY) === 'true'; } catch (_) { return false; }
}
let _eyeBtn = null;          // live-camera ("eye") toggle, next to the ear
let _flipBtn = null;         // front<->back flip, shown only when camera on + multi-cam
let _cameraView = null;      // CompanionCameraView while the live camera is on
let _hudReparented = null;
let _wsBus = null;
let _busReconnector = null;   // full-jitter backoff scheduler for the presence WS
let _activeSurface = 'private';
let _vrmActive = false;     // true once the unified avatar pipeline is up
let _audioRole = 'assistant';  // 'assistant' (listens for wake) | 'host' (lets media play)
let _ttsEventHandler = null;
let _audioBusEventHandler = null;
let _ttsEndHoldTimer = null;
let _wakeFlashTimer = null;
let _statusState = 'idle';     // 'idle' | 'listening' | 'thinking' | 'speaking' | 'hosting'
let _statusOverride = null;    // when set, replaces state label (e.g. audio source)
let _danceActive = false;
let _danceRotateTimer = null;
let _danceHistory = [];          // ring buffer: { animId, label, playedAt, durationSec }
let _timelinePanel = null;
let _timelineHandle = null;
let _timelineCloseTimer = null;
let _vrmUrl = null;          // current VRM URL — module-scope so swap can update it
let _avatarPickerEl = null;  // inline picker overlay (when open)
let _pttSession = null;      // lazy BeccaPttSession (Stage 2 hold-to-talk)
let _pttBtnEl = null;        // ref for state-driven CSS
let _animationRouter = null; // routes companion surface events into avatar systems
// Stage-manager text input. 'voice' = legacy (PTT auto-sends); 'stage' =
// compose bar visible, spoken utterances DRAFT into the box for edit-then-send
// and the user can also type. Per-user, device-local — the input surface is a
// property of THIS browser, like the mic/AEC prefs. Default 'voice' (never
// auto-flip an existing user into a new input model).
const INPUT_MODE_STORAGE_KEY = 'becca.input_mode';
let _inputMode = 'voice';    // 'voice' | 'stage'
let _inputModeBtn = null;    // dock toggle
let _composeBarEl = null;    // the compose row (textarea + send)
let _composeInputEl = null;  // the textarea
let _talkMode = 'off';       // 'off' | 'auto' — drives data-talk-mode + wake-word lifecycle
const TALK_MODE_STORAGE_KEY = 'becca.talk_mode';
let _wakeSession = null;     // lazy BeccaWakeSession (always-on wake listening)
let _wakePausedForCall = false;
let _wakePrefsHandler = null;
let _wakeResumeTimer = null; // deferred post-call resume to break feedback loops
// Follow-up window — after a wake-triggered turn, keep the PTT session armed
// for a grace period so the user can ask a follow-up without re-saying the
// wake word. Driven by explicit ``becca-ptt:turn-complete`` /
// ``turn-aborted`` events from BeccaPttSession; cleared by silence abort,
// manual PTT press, ``bye becca``, or dispose.
let _followUpActive = false;
const FOLLOWUP_DEFAULT_WINDOW_MS = 30000;
let _heartbeatTimer = null;
let _heartbeatBadTicks = 0;        // consecutive ticks where the VRM stage looked dead
let _heartbeatReloadGuard = 0;     // wall-clock timestamp of last forced reload
let _gazePointerHandler = null;    // global pointermove (tracks last cursor)
let _gazeLeaveHandler = null;
let _gazeLastCx = 0, _gazeLastCy = 0;
let _passthroughHandler = null;    // pointermove → toggle stage pointer-events
let _passthroughRafScheduled = false;
let _passthroughLastEvt = null;
let _passthroughRaycaster = null;
let _passthroughNdc = null;
// Visibility-state pause handler. Wired on mount, removed on unmount.
// When the tab is hidden the widget can't be seen, so the 3D render
// loop + soft animations + idle polls suspend until the user returns.
// We deliberately do NOT pause wake-word / always-listening / presence
// WebSockets here — those are the FEATURE of having a persistent
// companion (ambient invocation from any visible tab).
let _visibilityHandler = null;
let _pausedForVisibility = false;
// Render gate — the avatar render loop + load probe are the only expensive
// things tied to *seeing* the avatar. They should run only when the avatar is
// actually on-screen: the tab is visible AND the active surface isn't one that
// hides the widget (a minimized voice call display:none's the root, so the
// uncapped render loop would otherwise keep burning GPU/CPU on frames nobody
// can see). Affect poll / heartbeat / wake WS keep running regardless — a
// hidden-but-present companion is the feature, only the *rendering* pauses.
let _occluded = false;       // active surface hides the widget
let _renderPaused = false;   // current render-gate state (true = paused)
// Load-adaptive frame-rate cap for the floating widget. It renders uncapped
// by default, which is wasteful when the page is contended — but the right
// trigger is ACTUAL overload, not a hard mode rule. When there's main-thread
// headroom she stays at 30fps; as the page gets busy (a coder tool-call
// flurry, heavy layout, another busy tab in the same process) she yields
// frames so input + streaming stay responsive, then recovers when it clears.
// Complements _pauseForVisibility, which still hard-pauses on tab-hidden.
//
// Pressure signal = event-loop lag: a self-rescheduling timer measures how
// late it fires vs its scheduled interval. Lateness == the main thread was
// too busy to service the timer on time (same idea as the server-side
// event_loop_stall watchdog). EMA-smoothed so a lone GC spike doesn't slam
// the rate, and because the avatar is itself one of the loads being measured
// the throttle self-settles (back off → lag drops → hold, AIMD-style).
const _LOAD_PROBE_MS = 500;     // lag-probe cadence
const _FPS_HEADROOM = 30;       // responsive page: full presence rate
const _FPS_SATURATED = 8;       // saturated page: ambient-alive only
const _FPS_SPEAKING_FLOOR = 20; // never below this while she's actually talking
const _LAG_GOOD_MS = 20;        // <= this → no throttle
const _LAG_BAD_MS = 250;        // >= this → floor
let _loadProbeTimer = null;
let _loadProbeLast = 0;
let _lagEmaMs = 0;
// Slice 1 — chrome melt-away, transcript chip, mini-controls drawer,
// "heard you" microcopy + verb-fired tick toast. All live in the dock
// region of the widget and share one idle-timer + reveal-handler set
// so chrome only ever has one "alive vs dim" state at a time.
let _transcriptChip = null;
let _transcriptClearTimer = null;
// "Read this page / chat / file" handoff chip — shown only when the
// foreground surface has registered loadable content (companion-context.js).
let _contextChip = null;
let _loadableHandler = null;
let _loadableBusy = false;
let _miniControlsEl = null;
let _miniControlsHideTimer = null;
let _miniControlsActivityHandlers = null;
let _chromeIdleTimer = null;
let _chromeActivityHandler = null;
let _heardYouTimer = null;
let _verbTickTimer = null;
let _longPressTimer = null;
let _longPressPointerId = null;
let _verbFiredHandler = null;
let _miniControlsPrefsHandler = null;
// Track A — voice decision legibility. _decisionHudEl is the opt-in panel;
// _decisionLog is the rolling most-recent-first buffer it renders.
let _decisionHudEl = null;
let _decisionTellTimer = null;
let _statusTellFaintLabel = null;   // the one override label that renders faint
const _decisionLog = [];
const DECISION_LOG_CAP = 8;
const DECISION_TELL_HOLD_MS = 1300;
const DECISION_TELL_AMBIENT_HOLD_MS = 850;   // confident ambient drop: shorter/fainter
const CHROME_IDLE_MS = 4000;
const LONGPRESS_MS = 450;
const MINI_CONTROLS_TIMEOUT_MS = 12000;
const HEARD_YOU_HOLD_MS = 520;
const VERB_TICK_HOLD_MS = 1400;
const TRANSCRIPT_MAX_CHARS = 84;

/**
 * Mount the companion widget. Idempotent — calling twice is a no-op.
 * Returns the root element.
 */
export function mountBeccaPresence(opts = {}) {
  if (_root) return _root;
  // Fullscreen lock-screen presence (?presence=1) owns the singleton avatar via
  // presence-fullscreen.js — don't also mount the floating widget or they'll
  // fight over avatarState (only one standalone avatar can be active).
  try {
    if (new URLSearchParams(window.location.search).get('presence') === '1') return null;
  } catch (_) { /* no-op */ }
  // Create the per-mount lifetime scope first so every subsequent
  // attach/observer registration can route through it. Disposed in
  // unmountBeccaPresence.
  _lifetime = createLifetime();
  document.body.classList.remove('becca-dismissed');
  _hideSummonPip();

  _root = document.createElement('div');
  _root.className = 'becca-presence';
  _root.setAttribute('role', 'complementary');
  _root.setAttribute('aria-label', 'Companion presence');
  _root.setAttribute('tabindex', '0');

  _stage = document.createElement('div');
  _stage.className = 'becca-presence__stage';
  const placeholder = _buildPlaceholder();
  _stage.appendChild(placeholder);
  _root.appendChild(_stage);

  _wakeFlash = _buildWakeFlash();
  _root.appendChild(_wakeFlash);

  _returnFlash = _buildReturnFlash();
  _root.appendChild(_returnFlash);

  _timelineHandle = _buildTimelineHandle();
  _root.appendChild(_timelineHandle);
  _timelinePanel = _buildTimelinePanel();
  _root.appendChild(_timelinePanel);
  // Cache-first: render whatever was in localStorage from the last
  // session, then reconcile against server. Same pattern for ratings.
  _loadDanceHistory();
  _refreshDanceHistoryFromServer();
  try { movementConductor.refreshRatingsFromServer?.(); } catch (_) {}
  // Server-authoritative user atlas: merge uploads into the conductor's
  // selection pool. Failure is silent — the conductor still has the
  // bundled ATLAS to draw from.
  _refreshUserAnimations();
  // Phase C: fetch the user's loops + active state, apply to the
  // conductor so the very first auto-dispatch respects the loop.
  _refreshLoops();

  _root.appendChild(_buildDismissButton());

  // Bottom dock — PTT button on the left, audio-mode toggle on the
  // right, status row centered between them. Keeps controls adjacent
  // so dragging is intuitive and the top of the widget stays clean.
  const dock = document.createElement('div');
  dock.className = 'becca-presence__dock';
  const pttBtn = _buildPttButton();
  _audioModeBtn = _buildAudioModeButton();
  _micDspBtn = _buildMicDspButton();
  _eyeBtn = _buildEyeButton();
  _inputModeBtn = _buildInputModeButton();
  _statusRow = _buildStatusRow();
  _affectRow = _buildAffectRow();
  dock.appendChild(pttBtn);
  dock.appendChild(_statusRow);
  dock.appendChild(_audioModeBtn);
  dock.appendChild(_micDspBtn);
  dock.appendChild(_eyeBtn);
  dock.appendChild(_inputModeBtn);
  _root.appendChild(dock);
  // Stage-manager compose bar — sits below the dock, hidden unless the user
  // switched input to 'stage'. Talk to draft, edit, Send; or just type.
  _composeBarEl = _buildComposeBar();
  _root.appendChild(_composeBarEl);
  // Affect row sits below the dock — a quiet italic line that's
  // hidden until she has a confident read.
  _root.appendChild(_affectRow);
  // Restore the saved input mode (default 'voice') now that the dock exists.
  try { _applyInputMode(_readInputMode(), { savePrefs: false }); } catch (_) {}

  // Slice 1 — transcript chip + mini-controls drawer.
  // Chip surfaces partial STT above the status row (closes the
  // "did she hear me?" gap). Drawer reveals on long-press and
  // exposes the talk/listen/settings handles without taking up
  // permanent chrome space — preserves the "she's just there" feel.
  _transcriptChip = _buildTranscriptChip();
  _root.appendChild(_transcriptChip);
  _contextChip = _buildContextChip();
  _root.appendChild(_contextChip);
  _miniControlsEl = _buildMiniControls();
  _root.appendChild(_miniControlsEl);
  // Track A — opt-in decision HUD. Sits above the dock; hidden unless
  // companion_voice_decision_hud is on. Renders the last few routing
  // verdicts (transcript → goal → confidence) so the user can see what
  // she decided without reading logs.
  _decisionHudEl = _buildDecisionHud();
  _root.appendChild(_decisionHudEl);
  try { _renderDecisionHud(); } catch (_) { /* best effort */ }

  for (const corner of ['nw', 'ne', 'sw', 'se']) {
    const h = document.createElement('div');
    h.className = `becca-presence__resize becca-presence__resize--${corner}`;
    h.dataset.corner = corner;
    h.setAttribute('aria-hidden', 'true');
    _root.appendChild(h);
  }

  document.body.appendChild(_root);
  _restoreSize();
  _restorePosition();
  _restoreAudioRole();
  _restoreTalkMode();
  _attachDragHandlers();
  _attachResizeHandlers();
  _attachStagePassthrough();
  _attachViewportClamp();
  _attachCursorGaze();
  _attachKeyboardShortcuts();
  // Hold-anywhere-on-widget → crisis panel was retired in 2026 — the
  // gesture fired during drag tests and finger-rests. _openRealTalkPanel
  // is now only invoked deliberately (companion runtime / safety floor /
  // dedicated settings entry).
  _attachTtsListener();
  _attachAudioBusListener();
  _ensureAnimationRouter();
  _setStatusState('idle');
  // Start the soft affect indicator. Polls /api/companion/affect_read
  // every 90s; row stays hidden when no confident read exists.
  _startAffectPoll();

  // Visibility pause hook. Stops the 3D render loop + idle polls when
  // the tab is hidden (user switched to another tab/window). Wake-
  // word + always-listening WebSockets stay running — those are the
  // feature. See the ``_pauseForVisibility`` doc-comment.
  _attachVisibilityPause();

  // Slice 1 — chrome melt + long-press drawer + verb-fired tick
  // listener. Each lives inside the widget root so a single root
  // remove() during unmount drops the DOM event handlers with it;
  // the timer + module-window cleanup happens in unmountBeccaPresence.
  _attachChromeIdle();
  _attachLongPressHandle();
  _attachVerbTickListener();
  _attachLoadableListener();

  // Connect to the presence bus for glow / surface / pose updates.
  // Async because we have to fetch an auth ticket first. The reconnector
  // owns retry/backoff; _connectBus sets _wsBus and wires onclose →
  // schedule() itself, so start() is the only call needed here.
  _busReconnector = createReconnector({
    connect: _connectBus,
    base: 2000, cap: 30000, name: 'presence',
  });
  _busReconnector.start();

  // Apply current surface (handles screen-share detection, etc.)
  _applySurface(_detectSurface());
  _watchSurfaceChanges();

  // Defer the standalone activation by a short tick so voice.js's
  // auto-avatar-on-WS-open path gets first shot at the GL slot. If a
  // call is about to claim the avatar (avatar_active pref), our
  // ``activateAvatarStandalone`` would just return false anyway and
  // we'd race the tear-down. Letting the call go first means:
  //   - If voice.js activates avatar mode  → widget stays on placeholder
  //   - If voice.js doesn't activate       → widget activates standalone
  //   - Call ends → deactivateAvatar fires __beccaReactivateVRM → widget
  // 250ms is enough for ws.onopen + the pref fetch to settle on a LAN.
  _vrmUrl = opts.vrmUrl || _restoreAvatarChoice()?.vrm_url || BECCA_VRM_URL;
  // Ensure the server has a head-framed thumbnail for the active VRM so
  // the summon pip can paint her face the moment the user dismisses.
  // Independent of the live-stage activation below — runs in parallel,
  // no-ops fast when the server already has a real PNG.
  _ensurePortraitForActiveAvatar(_vrmUrl);
  setTimeout(() => {
    if (!_root) return;  // unmounted during the delay
    if (avatarState.active || avatarState.loading) {
      console.info('[becca] avatar claimed externally (call?) — widget stays on placeholder');
      return;
    }
    _activateInto(_stage, _vrmUrl, placeholder);
  }, 250);

  // Re-activation hook fired at the END of a voice call (by
  // _teardownVoiceCall in voice.js). Restores visibility if we were
  // hidden for the call, and brings the standalone VRM pipeline back
  // up without a page reload.
  window.__beccaReactivateVRM = () => {
    if (!_root) return;
    _root.style.display = '';
    // Re-detect surface immediately so _activeSurface returns to
    // 'private' (or whatever fits the post-call viewport) instead of
    // staying stuck on 'voice'. The MutationObserver in
    // _watchSurfaceChanges also covers this when voice.js mutates the
    // pill's class, but voice.js's ordering between pill teardown and
    // this hook is not guaranteed.
    _applySurface(_detectSurface());
    const pip = document.getElementById(_SUMMON_BTN_ID);
    if (pip) pip.style.display = '';
    _vrmActive = false;
    let ph = _stage.querySelector('.becca-presence__placeholder');
    if (!ph) { ph = _buildPlaceholder(); _stage.appendChild(ph); }
    _activateInto(_stage, _vrmUrl, ph);
    if (!_heartbeatTimer) _startHeartbeat();
    // Resume wake listening on a 3s delay. Resuming instantly is a
    // feedback loop: the mic may still be carrying TTS-tail audio (or
    // a final user utterance like "thanks bye") that immediately
    // re-fires the detector and re-opens the call. The cooldown lets
    // the acoustic environment quiet down. Idempotent if the user
    // re-enters a call during the delay — the timer is cleared on
    // __beccaHideForCall.
    if (_wakePausedForCall && _wakeSession) {
      if (_wakeResumeTimer) { clearTimeout(_wakeResumeTimer); _wakeResumeTimer = null; }
      _wakeResumeTimer = setTimeout(() => {
        _wakeResumeTimer = null;
        if (_wakeSession && _wakePausedForCall) {
          try { _wakeSession.resume(); } catch (_) {}
          _wakePausedForCall = false;
        }
      }, 3000);
    }
    // Re-arm always-listening capture after the call ends. Same 3s
    // grace as wake-resume so the acoustic environment quiets first
    // (TTS tail, "thanks bye" residue won't immediately re-fire).
    if (_alwaysListeningMode()) {
      setTimeout(() => {
        if (_root) _ensureAlwaysListening().catch(() => {});
      }, 3000);
    }
  };

  // Hide-for-call hook fired by voice.js when a call STARTS. The widget
  // disappears (DOM stays mounted, state preserved) and gives up its
  // GL slot. ``__beccaReactivateVRM`` reverses both when the call ends.
  // A user-initiated dismiss (× button) is a different path and is
  // untouched — that fully unmounts the widget.
  window.__beccaHideForCall = () => {
    if (!_root) return;
    // The call surface owns the camera during a call — release the widget's
    // live-camera view so two getUserMedia streams don't run at once.
    _stopCameraView();
    _refreshEyeVisibility();
    if (_vrmActive) {
      try { deactivateAvatar(); } catch (_) {}
      _vrmActive = false;
    }
    // Track the call surface explicitly so position persistence,
    // status logic, and re-detect-on-return all agree on state.
    _activeSurface = 'voice';
    _root.style.display = 'none';
    _setOccluded(true);  // keep the render gate coherent across the call
    const pip = document.getElementById(_SUMMON_BTN_ID);
    if (pip) pip.style.display = 'none';
    if (_heartbeatTimer) { clearInterval(_heartbeatTimer); _heartbeatTimer = null; }
    _heartbeatBadTicks = 0;
    // Pause wake listening for the call. The call owns the mic now; the
    // wake detector should not fire on the user's mid-call utterances
    // or on Becca's own TTS bouncing back through the speaker.
    if (_wakeResumeTimer) { clearTimeout(_wakeResumeTimer); _wakeResumeTimer = null; }
    if (_wakeSession && _wakeSession.state !== 'paused' && _wakeSession.state !== 'idle') {
      try { _wakeSession.pause(); } catch (_) {}
      _wakePausedForCall = true;
    }
    // Call modal takes the mic — abandon any follow-up window.
    _followUpActive = false;
    // Suspend always-listening for the call duration. Re-armed in
    // __beccaReactivateVRM on call end.
    if (_alwaysListeningRearmTimer) {
      clearTimeout(_alwaysListeningRearmTimer);
      _alwaysListeningRearmTimer = null;
    }
    _alwaysListeningArmed = false;
  };

  // Wake-word flash hook — the future wake-word detector (Slice 3)
  // calls window.__beccaFlashWake('hey becca') on a detection. Brief
  // overlay near the top of the widget, ~1.6s decay.
  window.__beccaFlashWake = (phrase) => _flashWake(phrase);

  // Heartbeat — watchdog that catches the case where the VRM stage
  // goes dark (lost WebGL context that never restores, render loop
  // stalled, animator threw mid-tick) without us hearing about it.
  // Ticks every 5s, requires two consecutive "dead" reads before
  // forcing a reload so transient context-lost/restore doesn't churn.
  _startHeartbeat();

  // Wake-word listening — opt-in via the Wake-word section of the
  // Companion settings tab. When enabled, opens a persistent
  // /ws/voice/wake stream and fires the voice-call entrypoint on
  // detection. Pauses for the duration of any active call so the mic
  // isn't double-claimed.
  _updateWakeStateAttribute(_wakeEnabled() ? 'connecting' : 'off');
  _ensureWakeSession();

  // Always-listening — kick the persistent PTT capture loop. The mic
  // stays open while the widget is mounted; the server's address
  // classifier (augmentum/architect/address.py) filters every
  // finalized utterance so Becca only responds when actually
  // addressed. Non-addressed speech becomes ambient observation.
  if (_alwaysListeningMode()) {
    _ensureAlwaysListening();
  }
  // iOS audio unlock. The arm above auto-starts at page load with no user
  // gesture, so on iPad the AudioContext stays suspended and no PCM frames
  // ever reach the server — the user's always_listening choice silently
  // never engages. Unlock on the first interaction, then cleanly re-arm
  // whichever mode they chose. A no-op on desktop where audio already runs.
  _primeAudioOnFirstGesture();
  // Tab-visibility recovery — mobile browsers suspend the WebSocket
  // and AudioContext when the tab is backgrounded. On return, we may
  // be stuck armed against a dead session. Force a clean rearm so
  // always-on actually stays on across tab switches.
  _alwaysListeningVisibilityHandler = () => {
    if (document.visibilityState !== 'visible') return;
    if (!_alwaysListeningMode()) return;
    if (_wakePausedForCall) return;
    console.info('[becca] always-listening: visibility restored, rearming');
    _clearAlwaysListeningWatchdog();
    _alwaysListeningArmed = false;
    _scheduleAlwaysListeningRearm();
  };
  document.addEventListener('visibilitychange', _alwaysListeningVisibilityHandler);
  _wakePrefsHandler = () => {
    // Settings panel or PTT button changed wake prefs — restart the
    // session so the new toggle state / phrase takes effect without a
    // page reload. Both surfaces flip the same localStorage flag.
    if (_wakeSession) {
      try { _wakeSession.dispose(); } catch (_) {}
      _wakeSession = null;
      _wakePausedForCall = false;
    }
    if (_wakeEnabled()) {
      _updateWakeStateAttribute('connecting');
      _ensureWakeSession();
    } else {
      _updateWakeStateAttribute('off');
    }
    // Keep the cycle button's talk-mode in sync with the canonical wake
    // flag — handles the case where the Settings panel flipped it, not
    // this button. Skip the full _setTalkMode path to avoid a redundant
    // re-dispatch of becca:wake-prefs-changed.
    const expected = _wakeEnabled() ? 'auto' : 'off';
    if (_talkMode !== expected) {
      _talkMode = expected;
      _uSet(TALK_MODE_STORAGE_KEY, expected);
      if (_pttBtnEl) {
        _pttBtnEl.dataset.talkMode = expected;
        const iconEl = _pttBtnEl.querySelector('.becca-presence__ptt-icon');
        if (iconEl) iconEl.innerHTML = _pttIconSvg(expected);
      }
    }
  };
  window.addEventListener('becca:wake-prefs-changed', _wakePrefsHandler);

  // Settings re-fetched (e.g. the live-camera capability was toggled) —
  // re-evaluate the eye's visibility live instead of waiting for the next
  // VRM activation / page reload.
  _lifetime.addEventListener(window, 'becca:settings-refreshed', () => {
    _refreshEyeVisibility();
  });

  // Live-switch handler — Settings → "How she listens" card click
  // dispatches this. Tear down whichever listening path was active for
  // the old mode and start the one for the new mode, without a page
  // reload. window.__beccaSettings has already been refreshed by the
  // settings panel before this fires, so _alwaysListeningMode() reads
  // the new value. Lifetime-tracked so unmount detaches it — pre-fix
  // this was an anonymous listener that leaked once per mount cycle.
  _lifetime.addEventListener(window, 'becca:activation-mode-changed', () => {
    // 1. Cancel any pending always-listening rearm AND stop the
    //    currently-active capture. Without the captureStop the mic
    //    keeps streaming for up to 30s after the user toggles to
    //    wake_word / ptt_only — and the server keeps processing
    //    transcripts with the wrong activation mode in mind.
    if (_alwaysListeningRearmTimer) {
      clearTimeout(_alwaysListeningRearmTimer);
      _alwaysListeningRearmTimer = null;
    }
    _alwaysListeningArmed = false;
    if (_pttSession && !_alwaysListeningMode()) {
      try { _pttSession.captureStop?.(); } catch (_) {}
    }
    // 2. Wake session — start or stop based on new _wakeEnabled().
    if (_wakeSession && !_wakeEnabled()) {
      try { _wakeSession.dispose(); } catch (_) {}
      _wakeSession = null;
      _wakePausedForCall = false;
      _updateWakeStateAttribute('off');
    } else if (!_wakeSession && _wakeEnabled()) {
      _updateWakeStateAttribute('connecting');
      _ensureWakeSession();
    }
    // 3. Always-listening — kick the loop when the new mode wants it.
    if (_alwaysListeningMode()) {
      _ensureAlwaysListening().catch((err) =>
        console.warn('[becca] activation switch -> always-listening failed', err));
    }
    // 4. ptt_only — nothing to start; both wake + listening loops
    //    have been wound down above.
  });

  return _root;
}

/**
 * Watchdog over the VRM stage. The widget thinks it has an avatar up
 * when ``_vrmActive`` is true. The unified pipeline is healthy when
 * ``avatarState.active`` is true AND ``avatarState._contextLost`` is
 * false. Anything else with two consecutive ticks of disagreement —
 * we owned the stage, the stage is dead, and nothing else is using
 * the GL slot — triggers a fresh standalone activation. Throttled so
 * we can't churn faster than once every 15s. Reads the live module
 * ``_vrmUrl`` each tick so avatar swaps are picked up automatically.
 */
function _startHeartbeat() {
  if (_heartbeatTimer) clearInterval(_heartbeatTimer);
  _heartbeatBadTicks = 0;
  // Async-aware tick wrapper. setInterval can't await, but the wrapper
  // returns immediately after kicking the async function; per-tick
  // overlap is harmless because the bad-ticks counter + reload-guard
  // both serialize state.
  _heartbeatTimer = setInterval(() => { _heartbeatTick(); }, 5000);
}

// Engine-loading awareness for the heartbeat. The pipeline's static
// assets (VRM file, body-atlas json, for-session route) share the
// uvicorn worker that's busy when llama-server is in `starting` state,
// so a reactivate-reload during a model swap loses the race and ends
// with a dark stage. Probe /api/engine/v2/status (cached 5s) only at
// the moment the heartbeat is about to act — adds 0 fetches on healthy
// ticks, 1 fetch right before each candidate reload.
let _engineLoadingCache = { value: false, fetchedAt: 0 };
const _ENGINE_LOADING_CACHE_TTL_MS = 5000;

async function _isEngineLoading() {
  const now = Date.now();
  if (now - _engineLoadingCache.fetchedAt < _ENGINE_LOADING_CACHE_TTL_MS) {
    return _engineLoadingCache.value;
  }
  try {
    const r = await fetch('/api/engine/v2/status', {
      credentials: 'same-origin',
      cache: 'no-store',
    });
    if (!r.ok) {
      // 503 / 401 / 404 — treat as "not loading" rather than locking
      // the heartbeat. Local engine may not exist (cloud-only setups);
      // we don't want to gate avatar recovery on something we don't
      // require to be present.
      _engineLoadingCache = { value: false, fetchedAt: now };
      return false;
    }
    const body = await r.json();
    const state = String(body && body.state || '').toLowerCase();
    const loading = state === 'starting';
    _engineLoadingCache = { value: loading, fetchedAt: now };
    return loading;
  } catch {
    _engineLoadingCache = { value: false, fetchedAt: now };
    return false;
  }
}

async function _heartbeatTick() {
  if (!_root) {
    if (_heartbeatTimer) { clearInterval(_heartbeatTimer); _heartbeatTimer = null; }
    return;
  }
  // Live camera owns the stage (full-screen). If the VRM isn't active yet,
  // DON'T force a reactivate-reload — that calls deactivateAvatar() and tears
  // the camera down (the "camera-first → VRM summon kills the camera" bug). The
  // camera fills the view regardless; VRM recovery resumes once filming stops.
  if (_cameraView) {
    _heartbeatBadTicks = 0;
    return;
  }
  // Call mode owns the GL slot — leave it alone. avatarState.active is
  // true with _standalone false during a voice call.
  if (avatarState.active && !avatarState._standalone) {
    _heartbeatBadTicks = 0;
    _hideCatchingUpOverlay();
    return;
  }
  // A fresh activation is in flight — don't race it.
  if (avatarState.loading) {
    _heartbeatBadTicks = 0;
    _hideCatchingUpOverlay();
    return;
  }
  // Healthy: widget owns the stage, the pipeline is active, no context loss.
  const owned = _vrmActive;
  const healthy = avatarState.active && !avatarState._contextLost;
  if (owned && healthy) {
    _heartbeatBadTicks = 0;
    _hideCatchingUpOverlay();
    return;
  }
  // Not owned and not loading and no call — widget never activated (race
  // with call start, or initial defer never landed). Treat as a soft
  // recovery: schedule reactivation on the next tick if it persists.
  _heartbeatBadTicks += 1;
  if (_heartbeatBadTicks < 2) return;

  // Before forcing a reactivate-reload, check if the engine is in
  // mid-swap. If yes, hold off — the reload would race the model
  // boot for the same uvicorn worker and lose. Show the user a calm
  // "catching up" overlay so the empty stage doesn't read as broken.
  const engineLoading = await _isEngineLoading();
  if (engineLoading) {
    _showCatchingUpOverlay();
    _heartbeatBadTicks = 0;  // reset so we re-evaluate fresh next tick
    return;
  }

  const now = Date.now();
  if (now - _heartbeatReloadGuard < 15000) return;
  _heartbeatReloadGuard = now;
  _heartbeatBadTicks = 0;
  console.warn('[becca] heartbeat: VRM stage dark — reloading companion');
  // Tear down whatever stale state the pipeline holds (renderer, scene,
  // canvas) before re-activating, otherwise activateAvatarStandalone's
  // ``active || loading`` gate could block us, or the new render would
  // mount alongside a corpse.
  try { if (avatarState.active) deactivateAvatar(); } catch (_) {}
  _vrmActive = false;
  let ph = _stage?.querySelector('.becca-presence__placeholder');
  if (!ph && _stage) { ph = _buildPlaceholder(); _stage.appendChild(ph); }
  _activateInto(_stage, _vrmUrl, ph);
}


/* ── Engine-loading "catching up" overlay ─────────────────────────
 *
 * A calm semi-transparent layer that floats inside the stage when the
 * heartbeat detects the VRM is dark AND the engine is mid-swap. Lives
 * alongside the placeholder element — the placeholder still covers
 * non-engine stalls (network blip, fresh mount, etc.); this overlay
 * only kicks in for the model-swap window where the empty stage
 * reads as "broken" rather than "loading".
 *
 * Sibling to the VRM canvas inside _stage. The heartbeat hides it as
 * soon as VRM activates / engine returns to ready, and unmountBecca
 * Presence drops it as part of stage cleanup.
 */
let _catchingUpEl = null;

function _showCatchingUpOverlay() {
  if (!_stage) return;
  if (_catchingUpEl && _stage.contains(_catchingUpEl)) return;
  _catchingUpEl = document.createElement('div');
  _catchingUpEl.className = 'becca-presence__catching-up';
  _catchingUpEl.setAttribute('aria-live', 'polite');
  _catchingUpEl.innerHTML = `
    <div class="becca-presence__catching-up-spinner" aria-hidden="true"></div>
    <div class="becca-presence__catching-up-text">catching her breath…</div>
  `;
  _stage.appendChild(_catchingUpEl);
}

function _hideCatchingUpOverlay() {
  if (!_catchingUpEl) return;
  try { _catchingUpEl.remove(); } catch (_) {}
  _catchingUpEl = null;
}

function _activateInto(stageEl, vrmUrl, placeholder) {
  activateAvatarStandalone({ host: stageEl, vrmUrl }).then(ok => {
    if (ok && _root) {
      // Drop the catching-up overlay the moment VRM activates so it
      // doesn't float on top of the live canvas.
      _hideCatchingUpOverlay();
      _vrmActive = true;
      // Begin sampling main-thread load and adapting her frame rate to it.
      _startLoadProbe();
      try { placeholder?.remove(); } catch (_) {}
      // The eye can now composite against a live VRM — reveal it if the
      // live-vision capability is enabled.
      _refreshEyeVisibility();
      console.info('[becca] VRM stage live (unified avatar.js path)');
    } else {
      console.warn('[becca] VRM stage skipped or failed — placeholder retained');
    }
  }).catch(err => {
    console.warn('[becca] VRM stage failed — placeholder retained', err);
  });
}


// Ensure the active avatar has a server-side thumbnail so the summon
// pip (and any future ambient affordance) can paint her face. Uses the
// shared offscreen renderer (preserveDrawingBuffer:true on a dedicated
// canvas) rather than the live avatar.js renderer, which runs
// preserveDrawingBuffer:false for perf and yields a transparent buffer
// when sampled outside its render loop. Fires the moment we know the
// VRM URL — no dependency on the live canvas being ready or readable,
// no 3s timer.
async function _ensurePortraitForActiveAvatar(vrmUrl) {
  if (!vrmUrl) return;
  // Prefer the persisted choice — usually fastest + works offline.
  // Fall back to /api/avatar/for-session so the default (bundled)
  // avatar still gets a per-user snapshot on first use.
  const persisted = _restoreAvatarChoice();
  let avatarId = persisted?.avatar_id || '';
  if (!avatarId) {
    try {
      const r = await fetch('/api/avatar/for-session', { credentials: 'same-origin' });
      if (r.ok) {
        const body = await r.json();
        // Server returns `avatar_id` (see augmentum/proxy/avatar_routes.py).
        // Fall back to `id` so any future contract realignment doesn't
        // re-break this code path.
        avatarId = body?.avatar_id || body?.id || '';
      }
    } catch { /* swallow */ }
  }
  if (!avatarId) return;
  try {
    const mod = await import('./avatar-thumbnail.js');
    await mod.ensureAvatarThumbnail(avatarId, vrmUrl);
    // Tell anyone listening (the summon pip in particular) that the
    // active avatar's portrait may have refreshed so they can re-paint.
    window.dispatchEvent(new CustomEvent('companion:avatar-thumb-ready', {
      detail: { avatar_id: avatarId },
    }));
  } catch (err) {
    console.warn('[becca] portrait ensure failed', err);
  }
}

// ── Audio-mode button (corner) ────────────────────────────────────
//
// Two states:
//
//   assistant — default. She's actively listening, treats audio as
//       conversational. When wake-word lands (Slice 3) this is the
//       mode that gates the detector ON.
//
//   host — she's hosting your audio: video, music, podcasts pass
//       through her body for embodiment (lipsync on speech, dance on
//       music — Phase 2/3 work). Wake-word detector is gated OFF in
//       this mode so background dialogue can't "wake" her accidentally.
//
// State is persisted to localStorage so reloads remember the choice.
// ``window.__beccaAudioRole`` is the source of truth other modules
// (future wake-word detector, content classifier) read from.

// ── Dismiss + summon pip ──────────────────────────────────────────
//
// Dismissing is per-tab and per-session: each browser/tab mounts its
// own widget from the global ``companion_persona_mode`` setting, but
// dismissing one shouldn't affect the others (different conversations
// often run in different tabs). When dismissed, a tiny ~14×14 pip
// appears in the bottom-right corner of the screen — click it to
// summon her back without touching settings or reloading.

const _SUMMON_BTN_ID = 'becca-summon-btn';

function _buildDismissButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__dismiss';
  btn.setAttribute('aria-label',
    'Dismiss the companion for this tab. Click the pip at the bottom-right to summon back.');
  btn.title = 'Dismiss (this tab only)';
  btn.innerHTML = `<svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true">
    <path d="M2.5 2.5 L9.5 9.5 M9.5 2.5 L2.5 9.5" stroke="currentColor"
          stroke-width="1.6" stroke-linecap="round" fill="none"/>
  </svg>`;
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    _dismissWidget();
  });
  return btn;
}

function _dismissWidget() {
  if (!_root) return;
  // Unmount the full widget (cleans up listeners, conductor stop,
  // dance loop, TTS hookups). Then surface the docked summon button
  // in the composer toolbar so the user can bring her back without a
  // reload. ``becca-dismissed`` on <body> also lets the header logo
  // act as an alternate summon affordance on surfaces where the
  // toolbar isn't visible (see becca-bootstrap.js).
  unmountBeccaPresence();
  document.body.classList.add('becca-dismissed');
  _ensureSummonPip();
}

function _ensureSummonPip() {
  const btn = document.getElementById(_SUMMON_BTN_ID);
  if (!btn) return;
  btn.classList.remove('hidden');
  if (!btn._beccaSummonBound) {
    btn._beccaSummonBound = true;
    btn.addEventListener('click', () => {
      btn.classList.add('hidden');
      mountBeccaPresence();
    });
  }
  // Pin the active companion's portrait as the button background so
  // the affordance shows her face instead of a generic silhouette.
  // Falls back gracefully (silhouette stays visible) when no thumb
  // is available yet.
  _paintSummonPipPortrait(btn);
  if (!btn._beccaPortraitListenerBound) {
    btn._beccaPortraitListenerBound = true;
    window.addEventListener('companion:avatar-thumb-ready', () => {
      const live = document.getElementById(_SUMMON_BTN_ID);
      if (live) _paintSummonPipPortrait(live);
    });
  }
}

function _hideSummonPip() {
  const btn = document.getElementById(_SUMMON_BTN_ID);
  if (btn) btn.classList.add('hidden');
}


async function _paintSummonPipPortrait(btn) {
  if (!btn) return;
  let avatarId = _restoreAvatarChoice()?.avatar_id || '';
  if (!avatarId) {
    try {
      const r = await fetch('/api/avatar/for-session', { credentials: 'same-origin' });
      if (r.ok) {
        const body = await r.json();
        // Server returns `avatar_id`; see comment in _ensurePortraitForActiveAvatar.
        avatarId = body?.avatar_id || body?.id || '';
      }
    } catch { /* swallow */ }
  }
  if (!avatarId) return;
  // Quick HEAD-equivalent: fetch with cache-bust so an upload that
  // just landed gets picked up. The placeholder header tells us not
  // to bother painting (silhouette stays).
  try {
    const url = `/api/avatar/${encodeURIComponent(avatarId)}/thumbnail?t=${Date.now()}`;
    const r = await fetch(url, { credentials: 'same-origin', cache: 'no-store' });
    if (!r.ok) return;
    const isPlaceholder = r.headers.get('X-Avatar-Thumbnail-Placeholder') === '1';
    if (isPlaceholder) return;
    btn.classList.add('has-portrait');
    btn.style.backgroundImage = `url("${url}")`;
  } catch (err) {
    console.warn('[becca] summon portrait fetch failed', err);
  }
}

// Exposed so bootstrap can drop the user straight into the dismissed-
// with-button state when companion_auto_summon is off (the widget was
// never mounted, so _dismissWidget never ran). Idempotent.
export function ensureSummonPip() {
  _ensureSummonPip();
}

// Hide the summon button — called from bootstrap on (re)mount.
export function hideSummonPip() {
  _hideSummonPip();
}


// ── Timeline panel — on-the-fly dance curation ────────────────────
//
// Subtle reveal handle on the right edge of the widget. Click to slide
// in a panel showing the last ~10 played dances; each row has four
// rating buttons:
//
//   ♡ like     bias × 1.6 — picked more often
//   ✕ dislike  bias × 0.35 — still possible, less likely
//   ⊘ broken   bias = 0 — skipped at selection until cleared
//   ⤓ longer   +8s added to per-id slot (stacks, capped at +60s)
//
// Ratings persist to localStorage via the conductor's recordRating().
// Playback history persists separately so the user can retro-rate a
// clip they remember from a prior session.

function _buildTimelineHandle() {
  const h = document.createElement('button');
  h.type = 'button';
  h.className = 'becca-presence__timeline-handle';
  h.setAttribute('aria-label', 'Open dance timeline — rate, dislike, mark broken');
  h.innerHTML = `<svg viewBox="0 0 8 24" width="6" height="20" aria-hidden="true">
    <rect x="0" y="2" width="2" height="20" rx="1" fill="currentColor"/>
  </svg>`;
  h.addEventListener('click', (e) => {
    e.stopPropagation();
    _setTimelineOpen(true);
  });
  return h;
}

function _buildTimelinePanel() {
  const p = document.createElement('div');
  p.className = 'becca-presence__timeline';
  p.hidden = true;
  p.setAttribute('aria-hidden', 'true');
  // Click inside the panel postpones auto-close. Click outside closes.
  p.addEventListener('click', (e) => {
    e.stopPropagation();
    _bumpTimelineAutoClose();
  });
  return p;
}

// Shared header markup. Four actions: loops (↻), upload (+),
// swap avatar (⇄), close (×). Plus a hidden file input for upload.
// When a loop is active, the loops button gets data-active="1" so
// CSS can highlight it. The header also surfaces the active loop's
// name as a small chip below the title row when one is active.
function _timelineHeaderHtml() {
  const loopActive = _activeLoopId
    ? _loopsCache.find(l => l.id === _activeLoopId)
    : null;
  const activeChip = loopActive
    ? `<div class="becca-timeline__active-loop" title="Host loop — click to clear">
         <span class="dot"></span>
         <span class="name">${_escapeHtml(loopActive.name)}</span>
       </div>`
    : '';
  return `
    <div class="becca-timeline__header">
      <span>Dance timeline</span>
      <div class="becca-timeline__header-actions">
        <button type="button" class="becca-timeline__loops"
                data-active="${loopActive ? '1' : '0'}"
                aria-label="Manage animations and host loops"
                title="${loopActive
                  ? `Animations · host loop: ${_escapeHtml(loopActive.name)}`
                  : 'Animations — tag, disable, build host loops'}">↻</button>
        <button type="button" class="becca-timeline__upload"
                aria-label="Upload your own animation"
                title="Upload .vrma / .bvh">+</button>
        <button type="button" class="becca-timeline__swap" aria-label="Change avatar"
                title="Change avatar">⇄</button>
        <button type="button" class="becca-timeline__close" aria-label="Close">×</button>
      </div>
      <input type="file" class="becca-timeline__upload-input"
             accept=".vrma,.bvh" hidden>
    </div>
    ${activeChip}`;
}

// Voice-volume row — her TTS output gain. Sits at the top of the timeline
// panel (alongside her animation history) so the companion-widget knobs live
// together. Persists per-user via /api/config/ui (settings.companionVoiceVolume)
// and applies live to the shared TTS gain node. Default is a boost: Kokoro TTS
// runs soft and gets buried under Grove music / host-mode media.
let _voiceVolSaveTimer = null;

function _voiceCurrentPct() {
  let mult = 2.0;
  try {
    const v = getSettings()?.companionVoiceVolume;
    if (v != null && !isNaN(Number(v))) mult = Number(v);
  } catch (_) { /* settings not ready — show the default */ }
  return Math.round(mult * 100);
}

function _voiceVolumeRowHtml() {
  const pct = _voiceCurrentPct();
  return `
    <div class="becca-timeline__voicevol" title="How loud her spoken replies are">
      <svg class="becca-timeline__voicevol-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path fill="currentColor" d="M3 10v4h4l5 5V5L7 10H3zm13.5 2a4.5 4.5 0 0 0-2.5-4.03v8.06A4.5 4.5 0 0 0 16.5 12z"/>
      </svg>
      <span class="becca-timeline__voicevol-label">Her voice</span>
      <input type="range" class="becca-timeline__voicevol-slider"
             min="50" max="400" step="10" value="${pct}"
             aria-label="Companion voice volume">
      <span class="becca-timeline__voicevol-val">${pct}%</span>
    </div>`;
}

function _bindVoiceVolume() {
  if (!_timelinePanel) return;
  const slider = _timelinePanel.querySelector('.becca-timeline__voicevol-slider');
  const valEl = _timelinePanel.querySelector('.becca-timeline__voicevol-val');
  if (!slider) return;
  // Don't let a drag inside the slider count as an outside-click close.
  slider.addEventListener('click', (e) => e.stopPropagation());
  slider.addEventListener('input', (e) => {
    e.stopPropagation();
    _bumpTimelineAutoClose();
    const pct = parseInt(slider.value, 10) || 100;
    const mult = pct / 100;
    if (valEl) valEl.textContent = `${pct}%`;
    try { getSettings().companionVoiceVolume = mult; } catch (_) { /* settings unavailable */ }
    // Live — the gain node is persistent, so this also lifts the clip she's
    // speaking right now, not just the next one.
    try { setCompanionVoiceGain(mult); } catch (_) { /* tts graph not ready */ }
    clearTimeout(_voiceVolSaveTimer);
    _voiceVolSaveTimer = setTimeout(() => {
      try { syncVoicePrefsToBackend(); } catch (_) { /* best-effort server save */ }
    }, 400);
  });
}

// Wire header buttons. Called after both render paths so the upload
// button works whether the timeline is empty or populated.
function _bindTimelineHeaderActions() {
  if (!_timelinePanel) return;
  _timelinePanel.querySelector('.becca-timeline__close')
    ?.addEventListener('click', () => _setTimelineOpen(false));
  _timelinePanel.querySelector('.becca-timeline__swap')
    ?.addEventListener('click',
      (e) => { e.stopPropagation(); _openAvatarPicker(); });
  _timelinePanel.querySelector('.becca-timeline__loops')
    ?.addEventListener('click',
      (e) => { e.stopPropagation(); _openLoopsOverlay(); });
  // Active-loop chip is a quick-clear affordance — click it to
  // deactivate without opening the overlay.
  _timelinePanel.querySelector('.becca-timeline__active-loop')
    ?.addEventListener('click', async (e) => {
      e.stopPropagation();
      await _activateLoop(null);
      _renderTimeline();
    });
  const uploadBtn = _timelinePanel.querySelector('.becca-timeline__upload');
  const uploadInput = _timelinePanel.querySelector('.becca-timeline__upload-input');
  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      _bumpTimelineAutoClose();
      uploadInput.click();
    });
    uploadInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = '';  // reset so the same file re-triggers next time
      uploadBtn.disabled = true;
      uploadBtn.dataset.uploading = '1';
      try {
        await _uploadAnimationFile(file);
      } finally {
        uploadBtn.disabled = false;
        delete uploadBtn.dataset.uploading;
        _bumpTimelineAutoClose();
      }
    });
  }
}

function _renderTimeline() {
  if (!_timelinePanel) return;
  const ratings = movementConductor.getAllRatings?.() || {};
  if (!_danceHistory.length) {
    _timelinePanel.innerHTML = `
      ${_timelineHeaderHtml()}
      ${_voiceVolumeRowHtml()}
      <div class="becca-timeline__empty">
        nothing played yet — start music in host mode and she'll dance.
        <div class="becca-timeline__empty-sub">
          tap <strong>+</strong> to upload your own .vrma / .bvh.
        </div>
      </div>`;
    _bindTimelineHeaderActions();
    _bindVoiceVolume();
    return;
  }
  const rows = _danceHistory.map(_renderTimelineRow).join('');
  _timelinePanel.innerHTML = `
    ${_timelineHeaderHtml()}
    ${_voiceVolumeRowHtml()}
    <ul class="becca-timeline__rows">${rows}</ul>`;
  _bindTimelineHeaderActions();
  _bindVoiceVolume();
  // Bind rating buttons
  _timelinePanel.querySelectorAll('[data-anim][data-rating]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.anim;
      const kind = btn.dataset.rating;
      try {
        movementConductor.recordRating?.(id, kind);
        btn.dataset.justRated = '1';
        setTimeout(() => { btn.dataset.justRated = '0'; }, 400);
      } catch (_) {}
      _renderTimeline();  // refresh active-state highlights
      _bumpTimelineAutoClose();
    });
  });
}

function _renderTimelineRow(entry) {
  const r = movementConductor.getAllRatings?.()[entry.animId] || {};
  const ago = _timeAgo(entry.playedAt);
  const dur = Math.round(entry.durationSec || 0);
  const bonus = r.slotBonusSec ? `<span class="bonus">+${r.slotBonusSec}s</span>` : '';
  // Active states light up via data-rating-active attr on the row
  const active = r.kind || '';
  return `
    <li class="becca-timeline__row" data-rating-active="${active}">
      <div class="meta">
        <span class="label">${_escapeHtml(entry.label)}</span>
        <span class="time">${ago} · ${dur}s ${bonus}</span>
      </div>
      <div class="actions">
        <button type="button" class="btn-rating btn-like"
                data-anim="${_escapeHtml(entry.animId)}" data-rating="${active === 'like' ? 'clear' : 'like'}"
                title="${active === 'like' ? 'Remove like' : 'Like — pick more often'}">♡</button>
        <button type="button" class="btn-rating btn-dislike"
                data-anim="${_escapeHtml(entry.animId)}" data-rating="${active === 'dislike' ? 'clear' : 'dislike'}"
                title="${active === 'dislike' ? 'Remove dislike' : 'Dislike — pick less often'}">✕</button>
        <button type="button" class="btn-rating btn-broken"
                data-anim="${_escapeHtml(entry.animId)}" data-rating="${active === 'broken' ? 'clear' : 'broken'}"
                title="${active === 'broken' ? 'Mark working again' : 'Broken — skip this clip'}">⊘</button>
        <button type="button" class="btn-rating btn-longer"
                data-anim="${_escapeHtml(entry.animId)}" data-rating="longer"
                title="Longer — give this clip ~8 more seconds next time">⤓</button>
      </div>
    </li>`;
}

function _escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Truncate at the last word boundary inside ``maxChars`` and append
// an ellipsis. Falls back to a hard slice if no usable boundary exists
// (when the input has no spaces or all spaces are too close to the
// start). Used for the wake-flash overlay and return-flash bubble so
// neither cuts mid-word.
function _truncateGraceful(s, maxChars) {
  const str = String(s ?? '');
  if (str.length <= maxChars) return str;
  const cut = str.slice(0, maxChars);
  const sp = cut.lastIndexOf(' ');
  return (sp > Math.floor(maxChars / 3) ? cut.slice(0, sp) : cut) + '…';
}

// ── Avatar swap (inline picker, surfaced from timeline header) ────
//
// "Change avatar" lives in the timeline panel because the timeline is
// already the curation surface — it's where the user goes to make her
// theirs. Clicking it opens a compact grid overlay scoped to the
// widget. Selecting an avatar resolves its VRM URL via /api/avatar/
// for-session, persists the choice, and reactivates the standalone
// pipeline with the new model.

async function _openAvatarPicker() {
  if (_avatarPickerEl) return;  // already open
  // Tuck the timeline away while the picker is on screen. The picker's
  // back-arrow label is literally "Back to timeline", so the user
  // expects the timeline to be the resume context — leaving it visible
  // behind the picker grid (and at a higher z-index, occluding hits on
  // avatar tiles) was the bug. Remember whether it was open so we can
  // restore it on close.
  const wasTimelineOpen = _root?.dataset?.timelineOpen === 'true';
  if (wasTimelineOpen) _setTimelineOpen(false);

  const overlay = document.createElement('div');
  overlay.className = 'becca-presence__avatar-picker';
  overlay.dataset.restoreTimeline = wasTimelineOpen ? '1' : '0';
  overlay.innerHTML = `
    <div class="becca-presence__avatar-picker-header">
      <button type="button" class="becca-avatar-picker__back" aria-label="Back to timeline">←</button>
      <span>Change avatar</span>
    </div>
    <div class="becca-presence__avatar-picker-grid" data-state="loading">
      <div class="becca-avatar-picker__empty">Loading…</div>
    </div>
  `;
  _root.appendChild(overlay);
  _avatarPickerEl = overlay;

  overlay.querySelector('.becca-avatar-picker__back')
    ?.addEventListener('click', (e) => { e.stopPropagation(); _closeAvatarPicker(); });

  let avatars = [];
  try {
    const resp = await fetch('/api/avatar/list', { credentials: 'same-origin' });
    if (resp.ok) {
      const data = await resp.json();
      avatars = (data.avatars || []).filter(a => a.type !== 'portrait');
    }
  } catch (_) { /* network — render empty state below */ }

  // Re-check the open flag after BOTH awaits (fetch + json parse). A
  // user-initiated close that lands mid-fetch would otherwise reach the
  // innerHTML assignment below against a detached grid node.
  if (!_avatarPickerEl) return;
  const grid = overlay.querySelector('.becca-presence__avatar-picker-grid');
  if (!grid) return;
  if (!avatars.length) {
    grid.dataset.state = 'empty';
    grid.innerHTML = `<div class="becca-avatar-picker__empty">No VRM avatars available.</div>`;
    return;
  }
  grid.dataset.state = 'ready';
  grid.innerHTML = avatars.map(a => {
    const id = a.id || a.avatar_id || '';
    const charId = a.character_id || '';
    const name = a.name || 'VRM';
    const thumb = _safeAvatarThumbUrl(a.thumbnail_url || a.portrait_url || '');
    const active = (id && id === _restoreAvatarChoice()?.avatar_id) ? ' is-active' : '';
    return `
      <button type="button" class="becca-avatar-picker__item${active}"
              data-avatar-id="${_escapeHtml(id)}"
              data-character-id="${_escapeHtml(charId)}"
              title="${_escapeHtml(name)}">
        ${thumb
          ? `<img src="${_escapeHtml(thumb)}" alt="" loading="lazy">`
          : `<div class="becca-avatar-picker__thumb-fallback">?</div>`}
        <span class="becca-avatar-picker__name">${_escapeHtml(name)}</span>
      </button>`;
  }).join('');
  grid.querySelectorAll('.becca-avatar-picker__item').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      // Rapid double-click can fork the standalone pipeline: tile A's
      // deactivate races tile B's activate and the wrong avatar ends up
      // visible while _vrmUrl holds the other. Gate all picker tiles
      // for the duration of any swap so only the first click wins.
      if (_swapInFlight) return;
      _swapInFlight = true;
      _avatarPickerEl?.classList.add('is-swapping');
      try {
        const avatarId = btn.dataset.avatarId || '';
        const characterId = btn.dataset.characterId || '';
        await _swapWidgetVrm({ avatarId, characterId });
      } finally {
        _swapInFlight = false;
        _avatarPickerEl?.classList.remove('is-swapping');
      }
      _closeAvatarPicker();
    });
  });
}

// Defence in depth for `<img src>` in the avatar picker. Thumbnail
// URLs come from /api/avatar/list (server-controlled), but we'd rather
// fail safe than rely on that trust boundary forever. Same-origin paths
// only — anything exotic (javascript:, data:, http://attacker/) drops
// to empty so the fallback ``?`` glyph renders instead.
function _safeAvatarThumbUrl(url) {
  if (typeof url !== 'string' || !url) return '';
  if (url.startsWith('/api/') || url.startsWith('/static/')) return url;
  try {
    const parsed = new URL(url, window.location.origin);
    if (parsed.origin === window.location.origin) return parsed.pathname + parsed.search;
  } catch (_) { /* unparseable — fall through */ }
  return '';
}

function _closeAvatarPicker() {
  if (!_avatarPickerEl) return;
  const restoreTimeline = _avatarPickerEl.dataset?.restoreTimeline === '1';
  try { _avatarPickerEl.remove(); } catch (_) {}
  _avatarPickerEl = null;
  if (restoreTimeline) _setTimelineOpen(true);
}

// ── Animations overlay (loops + master list + editor) ────────────
//
// One management surface, evolved from the Phase C loops overlay:
//   - saved loops (activate / rename / edit members / delete)
//   - the full merged master list (bundled atlas + user uploads) with
//     preview / inline metadata editor / disable-or-delete per row
//   - member-pick mode: the same list flips to checkboxes when
//     creating a loop or editing one's membership
// Mirrors the avatar picker's overlay-with-back-arrow pattern — it
// floats over the widget body, hides the timeline beneath, and
// restores the timeline on close.

// Overlay-local state. Reset on close so a re-open starts clean.
let _libFilter = '';
let _libFamily = '';      // family chip filter ('' = all, grouped view)
let _editorId = null;     // animation id with the inline editor open
let _pickState = null;    // {loopId: string|null, selected: Set} | null

// Studio dock geometry. When a side of the widget has room, the
// manager mounts on document.body as a fixed panel beside her instead
// of covering the stage — so a ▶ preview is actually watchable while
// you edit the clip's metadata. A rAF follower keeps the panel glued
// to the widget through drags, corner resizes, and the width/height
// CSS transitions. Narrow viewports (no side room) fall back to the
// classic cover-the-widget overlay.
const STUDIO_PANEL_W = 380;
const STUDIO_MIN_PANEL_W = 280;
const STUDIO_GAP = 10;
let _studioFollowRaf = 0;

function _studioHasRoom() {
  if (!_root) return false;
  const rect = _root.getBoundingClientRect();
  const vw = window.innerWidth || document.documentElement.clientWidth;
  const space = Math.max(rect.left, vw - rect.right)
    - STUDIO_GAP - VIEWPORT_MARGIN;
  return space >= STUDIO_MIN_PANEL_W;
}

function _startStudioFollow(overlay) {
  let lastKey = '';
  const step = () => {
    if (_loopsOverlayEl !== overlay || !_root) { _studioFollowRaf = 0; return; }
    const rect = _root.getBoundingClientRect();
    const vw = window.innerWidth || document.documentElement.clientWidth;
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const spaceLeft = rect.left - STUDIO_GAP - VIEWPORT_MARGIN;
    const spaceRight = vw - rect.right - STUDIO_GAP - VIEWPORT_MARGIN;
    // Re-pick the side every frame — dragging her across the screen
    // flips the panel to whichever side has more room.
    const side = spaceLeft >= spaceRight ? 'left' : 'right';
    const space = Math.max(spaceLeft, spaceRight);
    const w = Math.round(
      Math.max(STUDIO_MIN_PANEL_W, Math.min(STUDIO_PANEL_W, space)));
    const left = Math.round(Math.max(
      VIEWPORT_MARGIN,
      side === 'left' ? rect.left - STUDIO_GAP - w : rect.right + STUDIO_GAP,
    ));
    const top = Math.round(Math.max(
      VIEWPORT_MARGIN, Math.min(rect.top, vh - 340 - VIEWPORT_MARGIN)));
    const height = Math.round(Math.max(
      340,
      Math.min(Math.max(rect.height, 340), vh - top - VIEWPORT_MARGIN)));
    const key = `${side}|${w}|${left}|${top}|${height}`;
    if (key !== lastKey) {
      lastKey = key;
      overlay.style.left = `${left}px`;
      overlay.style.top = `${top}px`;
      overlay.style.width = `${w}px`;
      overlay.style.height = `${height}px`;
      overlay.dataset.side = side;
    }
    _studioFollowRaf = requestAnimationFrame(step);
  };
  step();
}

async function _openLoopsOverlay() {
  if (_loopsOverlayEl) return;
  const wasTimelineOpen = _root?.dataset?.timelineOpen === 'true';
  if (wasTimelineOpen) _setTimelineOpen(false);
  const studio = _studioHasRoom();
  const overlay = document.createElement('div');
  overlay.className = 'becca-presence__loops-overlay';
  if (studio) overlay.classList.add('becca-presence__loops-overlay--studio');
  overlay.dataset.restoreTimeline = wasTimelineOpen ? '1' : '0';
  overlay.innerHTML = `
    <div class="becca-loops__header">
      <button type="button" class="becca-loops__back"
              aria-label="${studio ? 'Close animations' : 'Back to timeline'}"
              >${studio ? '×' : '←'}</button>
      <span>Animations</span>
      <button type="button" class="becca-loops__create becca-loops__upload"
              title="Upload your own .vrma / .bvh">+ add</button>
      <button type="button" class="becca-loops__create becca-loops__new"
              title="Pick clips for a new host loop">+ new loop</button>
      <button type="button" class="becca-loops__create becca-loops__history"
              title="Create a loop from recent history">+ from history</button>
      <input type="file" class="becca-loops__upload-input"
             accept=".vrma,.bvh" hidden>
    </div>
    <div class="becca-loops__body"></div>`;
  if (studio) document.body.appendChild(overlay);
  else _root.appendChild(overlay);
  _loopsOverlayEl = overlay;
  if (studio) _startStudioFollow(overlay);
  overlay.querySelector('.becca-loops__back')
    ?.addEventListener('click', (e) => {
      e.stopPropagation();
      _closeLoopsOverlay();
    });
  overlay.querySelector('.becca-loops__new')
    ?.addEventListener('click', (e) => {
      e.stopPropagation();
      _pickState = { loopId: null, selected: new Set() };
      _editorId = null;
      _renderLoopsBody();
    });
  overlay.querySelector('.becca-loops__history')
    ?.addEventListener('click', async (e) => {
      e.stopPropagation();
      await _createLoopFromHistory();
      _renderLoopsBody();
    });
  // + add — same hidden-input upload path as the timeline header, but
  // lands the new clip ready to tag: filter cleared so it's visible,
  // inline editor opened on it.
  const uploadBtn = overlay.querySelector('.becca-loops__upload');
  const uploadInput = overlay.querySelector('.becca-loops__upload-input');
  uploadBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    uploadInput?.click();
  });
  uploadInput?.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';  // reset so the same file re-triggers next time
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'uploading…';
    let anim = null;
    try {
      anim = await _uploadAnimationFile(file);
      if (anim?.id) {
        _libFilter = '';
        _pickState = null;
        _editorId = anim.id;
      }
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.textContent = '+ add';
      _renderLoopsBody();
      if (anim?.id) {
        overlay.querySelector('.becca-lib__editor')
          ?.scrollIntoView({ block: 'center' });
      }
    }
  });
  await Promise.all([_refreshLoops(), _refreshUserAnimations()]);
  _renderLoopsBody();
}

function _closeLoopsOverlay() {
  if (!_loopsOverlayEl) return;
  const restoreTimeline = _loopsOverlayEl.dataset?.restoreTimeline === '1';
  if (_studioFollowRaf) {
    cancelAnimationFrame(_studioFollowRaf);
    _studioFollowRaf = 0;
  }
  try { _loopsOverlayEl.remove(); } catch (_) {}
  _loopsOverlayEl = null;
  _libFilter = '';
  _libFamily = '';
  _editorId = null;
  _pickState = null;
  if (restoreTimeline) _setTimelineOpen(true);
}

function _renderLoopsBody() {
  if (!_loopsOverlayEl) return;
  const body = _loopsOverlayEl.querySelector('.becca-loops__body');
  if (!body) return;
  body.innerHTML = `
    ${_pickState ? _pickBannerHtml() : _loopsSectionHtml()}
    ${_librarySectionHtml()}`;
  if (_pickState) _bindPickBanner(body); else _bindLoopsSection(body);
  _bindLibrarySection(body);
}

// ── Saved loops section ────────────────────────────────────────────

function _loopsSectionHtml() {
  if (!_loopsCache.length) {
    return `
      <div class="becca-loops__empty">
        <p>
          The host loop is the set she dances through when you flip her
          into <strong>host</strong> mode and music is playing. Tap
          <strong>+ new loop</strong> to pick clips from the list below.
        </p>
      </div>`;
  }
  const rows = _loopsCache.map((loop) => {
    const isActive = loop.id === _activeLoopId;
    const count = (loop.animation_ids || []).length;
    return `
      <li class="becca-loops__row" data-active="${isActive ? '1' : '0'}">
        <div class="meta">
          <span class="name">${_escapeHtml(loop.name)}</span>
          <span class="count">${count} ${count === 1 ? 'clip' : 'clips'}</span>
        </div>
        <div class="actions">
          <button type="button" class="btn-activate"
                  data-loop="${_escapeHtml(loop.id)}"
                  title="${isActive ? 'Clear host loop (full atlas in host mode)' : 'Set as host loop'}">
            ${isActive ? '✓ host loop' : 'set as host'}
          </button>
          <button type="button" class="btn-loop-rename"
                  data-loop="${_escapeHtml(loop.id)}"
                  data-name="${_escapeHtml(loop.name)}"
                  title="Rename loop">✎</button>
          <button type="button" class="btn-loop-members"
                  data-loop="${_escapeHtml(loop.id)}"
                  title="Edit which clips are in this loop">clips</button>
          <button type="button" class="btn-delete"
                  data-loop="${_escapeHtml(loop.id)}"
                  data-name="${_escapeHtml(loop.name)}"
                  title="Delete loop">×</button>
        </div>
      </li>`;
  }).join('');
  return `<ul class="becca-loops__rows">${rows}</ul>`;
}

function _bindLoopsSection(body) {
  body.querySelectorAll('.btn-activate').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const loopId = btn.dataset.loop;
      const target = loopId === _activeLoopId ? null : loopId;
      await _activateLoop(target);
      _renderLoopsBody();
    });
  });
  body.querySelectorAll('.btn-loop-rename').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const name = prompt('Loop name', btn.dataset.name || '');
      if (!name?.trim()) return;
      await _updateLoop(btn.dataset.loop, { name: name.trim() });
      _renderLoopsBody();
    });
  });
  body.querySelectorAll('.btn-loop-members').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const loop = _loopsCache.find(l => l.id === btn.dataset.loop);
      if (!loop) return;
      _pickState = {
        loopId: loop.id,
        selected: new Set(loop.animation_ids || []),
      };
      _editorId = null;
      _renderLoopsBody();
    });
  });
  body.querySelectorAll('.btn-delete').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      await _deleteLoop(btn.dataset.loop, btn.dataset.name);
      _renderLoopsBody();
    });
  });
}

// ── Member-pick banner (new loop / edit members) ───────────────────

function _pickBannerHtml() {
  const editing = _pickState.loopId
    ? _loopsCache.find(l => l.id === _pickState.loopId)
    : null;
  const title = editing
    ? `Picking clips for "${_escapeHtml(editing.name)}"`
    : 'Picking clips for a new loop';
  return `
    <div class="becca-lib__pickbar">
      <span class="title">${title}</span>
      <span class="count">${_pickState.selected.size} selected</span>
      <button type="button" class="btn-pick-save">save</button>
      <button type="button" class="btn-pick-cancel">cancel</button>
    </div>`;
}

function _bindPickBanner(body) {
  body.querySelector('.btn-pick-save')
    ?.addEventListener('click', async (e) => {
      e.stopPropagation();
      const ids = [..._pickState.selected];
      if (!ids.length) { alert('Pick at least one clip.'); return; }
      if (_pickState.loopId) {
        await _updateLoop(_pickState.loopId, { animation_ids: ids });
      } else {
        const name = prompt('Loop name', 'my mix');
        if (!name?.trim()) return;
        await _createLoop(name.trim(), ids);
      }
      _pickState = null;
      await _refreshLoops();
      _renderLoopsBody();
    });
  body.querySelector('.btn-pick-cancel')
    ?.addEventListener('click', (e) => {
      e.stopPropagation();
      _pickState = null;
      _renderLoopsBody();
    });
}

// ── Master list (bundled + uploads) ────────────────────────────────

function _libraryEntries() {
  const filter = _libFilter.trim().toLowerCase();
  let entries = listEffectiveEntries();
  // Uploads first — they're the ones the user is actively working on.
  entries.sort((a, b) => (a.bundled === b.bundled) ? 0 : (a.bundled ? 1 : -1));
  if (_libFamily) entries = entries.filter((e) => familyOf(e) === _libFamily);
  if (!filter) return entries;
  return entries.filter((e) => {
    const hay = [
      e.id, e.label || '', familyOf(e),
      ...(e.roles || []),
    ].join(' ').toLowerCase();
    return hay.includes(filter);
  });
}

// Rows markup, grouped under family headers in the "all" view so the
// ~150-entry list scans as buckets; flat when a family chip or text
// filter has already narrowed it.
function _libraryRowsHtml(entries) {
  if (_libFamily || _libFilter.trim()) {
    return entries.map(_libraryRowHtml).join('');
  }
  const groups = new Map();   // family → rows[], insertion = listFamilies order
  for (const { family } of listFamilies()) groups.set(family, []);
  for (const e of entries) {
    const f = familyOf(e);
    if (!groups.has(f)) groups.set(f, []);
    groups.get(f).push(_libraryRowHtml(e));
  }
  let out = '';
  for (const [family, rows] of groups) {
    if (!rows.length) continue;
    out += `<li class="becca-lib__group">${_escapeHtml(family)}
              <span class="count">${rows.length}</span></li>`;
    out += rows.join('');
  }
  return out;
}

function _librarySectionHtml() {
  const entries = _libraryEntries();
  const roleOptions = listRoles()
    .map(r => `<option value="${_escapeHtml(r)}"></option>`).join('');
  const chips = [
    `<button type="button" class="becca-lib__fam" data-fam=""
             data-on="${_libFamily ? '0' : '1'}">all</button>`,
    ...listFamilies().map(({ family, count }) =>
      `<button type="button" class="becca-lib__fam"
               data-fam="${_escapeHtml(family)}"
               data-on="${_libFamily === family ? '1' : '0'}"
               >${_escapeHtml(family)} <span class="count">${count}</span></button>`),
  ].join('');
  const rows = _libraryRowsHtml(entries);
  return `
    <div class="becca-lib__header">
      <span>All animations</span>
      <input type="search" class="becca-lib__filter"
             placeholder="filter by name, role, family"
             value="${_escapeHtml(_libFilter)}">
    </div>
    <div class="becca-lib__fams">${chips}</div>
    <datalist id="becca-lib-roles">${roleOptions}</datalist>
    <ul class="becca-lib__rows">${rows ||
      '<li class="becca-loops__empty">nothing matches that filter</li>'}</ul>`;
}

function _libraryRowHtml(entry) {
  const id = _escapeHtml(entry.id);
  const name = _escapeHtml(entry.label || entry.id);
  const roles = (entry.roles || []).map(_escapeHtml).join(' · ');
  const badges = [
    entry.type === 'bvh' ? 'bvh' : '',
    entry.bundled ? '' : 'yours',
    entry.disabled ? 'off' : '',
  ].filter(Boolean).map(b => `<span class="badge badge-${b}">${b}</span>`).join('');
  const picked = _pickState?.selected?.has(entry.id);
  const pickBox = _pickState
    ? `<input type="checkbox" class="lib-pick" data-id="${id}"
              ${picked ? 'checked' : ''} ${entry.disabled ? 'disabled' : ''}>`
    : '';
  const removeBtn = entry.bundled
    ? `<button type="button" class="lib-off" data-id="${id}"
               title="${entry.disabled
                 ? 'Re-enable — back into her selection pool'
                 : 'Disable — she stops picking this one'}">
         ${entry.disabled ? '↺' : '⊘'}</button>`
    : `<button type="button" class="lib-del btn-delete" data-id="${id}"
               data-name="${name}" title="Delete upload">×</button>`;
  const editor = (_editorId === entry.id) ? _editorHtml(entry) : '';
  return `
    <li class="becca-lib__row" data-id="${id}"
        data-off="${entry.disabled ? '1' : '0'}">
      <div class="becca-lib__rowline">
        ${pickBox}
        <div class="meta">
          <span class="name">${name} ${badges}</span>
          <span class="tags">${roles || '<em>no roles — untaggable</em>'}</span>
        </div>
        <div class="actions">
          <button type="button" class="lib-play" data-id="${id}"
                  title="Preview on her now">▶</button>
          <button type="button" class="lib-edit" data-id="${id}"
                  title="Edit roles + feel">✎</button>
          ${removeBtn}
        </div>
      </div>
      ${editor}
    </li>`;
}

function _bindLibrarySection(body) {
  // Re-render only the list so the filter input keeps focus.
  const rerenderRows = () => {
    const listEl = body.querySelector('.becca-lib__rows');
    if (!listEl) return;
    listEl.innerHTML = _libraryRowsHtml(_libraryEntries()) ||
      '<li class="becca-loops__empty">nothing matches that filter</li>';
    _bindLibraryRows(body);
  };
  const filterEl = body.querySelector('.becca-lib__filter');
  if (filterEl) {
    filterEl.addEventListener('input', () => {
      _libFilter = filterEl.value;
      rerenderRows();
    });
    filterEl.addEventListener('click', (e) => e.stopPropagation());
  }
  body.querySelectorAll('.becca-lib__fam').forEach((chip) => {
    chip.addEventListener('click', (e) => {
      e.stopPropagation();
      _libFamily = chip.dataset.fam || '';
      body.querySelectorAll('.becca-lib__fam').forEach((c) => {
        c.dataset.on = (c.dataset.fam || '') === _libFamily ? '1' : '0';
      });
      rerenderRows();
    });
  });
  _bindLibraryRows(body);
}

function _bindLibraryRows(body) {
  body.querySelectorAll('.lib-pick').forEach((box) => {
    box.addEventListener('change', () => {
      if (!_pickState) return;
      if (box.checked) _pickState.selected.add(box.dataset.id);
      else _pickState.selected.delete(box.dataset.id);
      const count = _loopsOverlayEl?.querySelector('.becca-lib__pickbar .count');
      if (count) count.textContent = `${_pickState.selected.size} selected`;
    });
  });
  body.querySelectorAll('.lib-play').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      // Preview is always ONE pass — loop:false even for loop-tagged
      // entries, so a looping upload can't leave her dancing forever.
      // The conductor's safety release returns her to idle if the
      // clip's finished event never fires (malformed uploads).
      try {
        movementConductor.playById(btn.dataset.id, { loop: false });
      } catch (err) {
        console.warn('[becca-lib] preview play failed', btn.dataset.id, err);
      }
    });
  });
  body.querySelectorAll('.lib-edit').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _editorId = (_editorId === btn.dataset.id) ? null : btn.dataset.id;
      _renderLoopsBody();
    });
  });
  body.querySelectorAll('.lib-off').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      await _toggleBundledDisabled(btn.dataset.id);
    });
  });
  body.querySelectorAll('.lib-del').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      await _deleteUserAnimation(btn.dataset.id, btn.dataset.name);
      _renderLoopsBody();
    });
  });
  _bindEditor(body);
}

async function _toggleBundledDisabled(atlasId) {
  const entry = listEffectiveEntries().find(e => e.id === atlasId);
  if (!entry) return;
  try {
    const resp = await fetch(
      `/api/animations/overrides/${encodeURIComponent(atlasId)}`,
      {
        method: 'PUT',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disabled: !entry.disabled }),
      },
    );
    if (!resp.ok) { alert('Update failed'); return; }
    await _refreshUserAnimations();
    _renderLoopsBody();
  } catch (_) {
    alert('Update failed (network)');
  }
}

// ── Inline metadata editor ─────────────────────────────────────────
//
// Expands under a master-list row. Roles are the field that matters —
// they're the verbs she selects by — so they lead; the feel sliders
// follow; everything else folds under Advanced.

const _EDITOR_MODES = ['chat-call', 'chat-passive', 'narrative', 'passthrough'];

function _editorHtml(entry) {
  const em = entry.emotion || {};
  const slider = (key, label) => `
    <label class="ed-slider">
      <span>${label}</span>
      <input type="range" min="0" max="1" step="0.05"
             data-em="${key}" value="${em[key] ?? 0.5}">
    </label>`;
  const modeBoxes = _EDITOR_MODES.map(m => `
    <label class="ed-mode">
      <input type="checkbox" data-mode="${m}"
             ${(entry.modes || []).includes(m) || (entry.modes || []).includes('*') ? 'checked' : ''}>
      <span>${m}</span>
    </label>`).join('');
  const labelField = entry.bundled ? '' : `
    <label class="ed-field">
      <span>name</span>
      <input type="text" class="ed-label" maxlength="60"
             value="${_escapeHtml(entry.label || '')}">
    </label>`;
  return `
    <div class="becca-lib__editor" data-id="${_escapeHtml(entry.id)}">
      ${labelField}
      <label class="ed-field">
        <span>roles — what she reaches for, like verbs</span>
        <input type="text" class="ed-roles" list="becca-lib-roles"
               placeholder="greet, celebrate, comfort"
               value="${_escapeHtml((entry.roles || []).join(', '))}">
      </label>
      <div class="ed-sliders">
        ${slider('warmth', 'warmth')}
        ${slider('energy', 'energy')}
        ${slider('openness', 'openness')}
        ${slider('focus', 'focus')}
      </div>
      <details class="ed-advanced">
        <summary>advanced</summary>
        <label class="ed-field"><span>cost (0 micro → 1 theatrical)</span>
          <input type="range" class="ed-cost" min="0" max="1" step="0.05"
                 value="${entry.cost ?? 0.5}"></label>
        <label class="ed-field"><span>cooldown (seconds)</span>
          <input type="number" class="ed-cooldown" min="0" step="1"
                 value="${entry.cooldown ?? 300}"></label>
        <div class="ed-checks">
          <label><input type="checkbox" class="ed-loop"
                 ${entry.loop ? 'checked' : ''}><span>loops</span></label>
          <label><input type="checkbox" class="ed-explicit"
                 ${entry.explicitOnly ? 'checked' : ''}><span>only when asked</span></label>
        </div>
        <div class="ed-modes">${modeBoxes}</div>
        <label class="ed-field"><span>notes</span>
          <textarea class="ed-notes" rows="2"
                    maxlength="500">${_escapeHtml(entry.notes || '')}</textarea></label>
      </details>
      <div class="ed-actions">
        <button type="button" class="ed-save">save</button>
        <button type="button" class="ed-cancel">cancel</button>
        ${entry.bundled ? `<button type="button" class="ed-reset"
            title="Drop your edits — back to the bundled defaults">reset</button>` : ''}
      </div>
    </div>`;
}

function _bindEditor(body) {
  const ed = body.querySelector('.becca-lib__editor');
  if (!ed) return;
  ed.addEventListener('click', (e) => e.stopPropagation());
  const id = ed.dataset.id;
  ed.querySelector('.ed-cancel')?.addEventListener('click', () => {
    _editorId = null;
    _renderLoopsBody();
  });
  ed.querySelector('.ed-reset')?.addEventListener('click', async () => {
    try {
      await fetch(`/api/animations/overrides/${encodeURIComponent(id)}`, {
        method: 'DELETE', credentials: 'same-origin',
      });
    } catch (_) { /* row may not exist — same outcome */ }
    _editorId = null;
    await _refreshUserAnimations();
    _renderLoopsBody();
  });
  ed.querySelector('.ed-save')?.addEventListener('click', async () => {
    await _saveEntryEditor(id, ed);
  });
}

async function _saveEntryEditor(id, ed) {
  const entry = listEffectiveEntries().find(e => e.id === id);
  if (!entry) return;
  const roles = (ed.querySelector('.ed-roles')?.value || '')
    .split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
  const emotion = {};
  ed.querySelectorAll('[data-em]').forEach((s) => {
    emotion[s.dataset.em] = parseFloat(s.value);
  });
  const modes = [...ed.querySelectorAll('[data-mode]')]
    .filter(b => b.checked).map(b => b.dataset.mode);
  const cost = parseFloat(ed.querySelector('.ed-cost')?.value ?? '0.5');
  const cooldown = Math.max(0,
    parseFloat(ed.querySelector('.ed-cooldown')?.value ?? '300') || 0);
  const loop = !!ed.querySelector('.ed-loop')?.checked;
  const explicitOnly = !!ed.querySelector('.ed-explicit')?.checked;
  const notes = (ed.querySelector('.ed-notes')?.value || '').trim();

  let url; let payload;
  if (entry.bundled) {
    url = `/api/animations/overrides/${encodeURIComponent(id)}`;
    payload = {
      patch: { roles, emotion, modes, cost, cooldown, loop, explicitOnly, notes },
    };
  } else {
    url = `/api/animations/${encodeURIComponent(id)}`;
    payload = {
      roles, emotion, modes, cost,
      cooldown_sec: cooldown,
      loop_flag: loop,
      explicit_only: explicitOnly,
      notes,
    };
    const label = (ed.querySelector('.ed-label')?.value || '').trim();
    if (label) payload.label = label;
  }
  try {
    const resp = await fetch(url, {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err?.error || 'Save failed');
      return;
    }
    _editorId = null;
    await _refreshUserAnimations();
    _renderLoopsBody();
  } catch (_) {
    alert('Save failed (network)');
  }
}

// ── Loop CRUD helpers shared by sections ───────────────────────────

async function _updateLoop(loopId, updates) {
  try {
    const resp = await fetch(
      `/api/dance/loops/${encodeURIComponent(loopId)}`,
      {
        method: 'PUT',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      },
    );
    if (!resp.ok) { alert('Update failed'); return; }
    await _refreshLoops();
    // Membership edits to the ACTIVE loop must reach the conductor.
    _applyActiveLoopToConductor();
  } catch (_) {
    alert('Update failed (network)');
  }
}

async function _createLoop(name, animationIds) {
  try {
    const resp = await fetch('/api/dance/loops', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, animation_ids: animationIds }),
    });
    if (!resp.ok) { alert('Create failed'); return; }
    await _refreshLoops();
  } catch (_) {
    alert('Create failed (network)');
  }
}

async function _swapWidgetVrm({ avatarId, characterId }) {
  // Resolve the actual VRM URL via the session-aware endpoint. Same
  // path the call uses, so character_id-driven defaults work the same.
  const params = new URLSearchParams({ mode: 'passthrough' });
  if (avatarId) params.set('avatar_id', avatarId);
  if (characterId) params.set('character_id', characterId);
  let vrmUrl = null;
  try {
    const resp = await fetch(`/api/avatar/for-session?${params.toString()}`,
                             { credentials: 'same-origin' });
    if (resp.ok) {
      const data = await resp.json();
      if (data.type !== 'portrait') vrmUrl = data.vrm_url || null;
    }
  } catch (_) { /* network/parse failure — null check below warns + bails */ }
  if (!vrmUrl) {
    console.warn('[becca] avatar swap: no vrm_url resolved');
    return;
  }
  _vrmUrl = vrmUrl;
  _persistAvatarChoice({ avatar_id: avatarId, character_id: characterId, vrm_url: vrmUrl });
  // Kick off a thumbnail ensure for the new avatar so a quick post-swap
  // dismiss still gets her face on the pip rather than the silhouette.
  _ensurePortraitForActiveAvatar(_vrmUrl);
  // Tear down the live standalone pipeline (if any) before re-activating
  // with the new model. Same dance as the heartbeat reload path.
  try { if (avatarState.active && avatarState._standalone) deactivateAvatar(); } catch (_) {}
  _vrmActive = false;
  let ph = _stage?.querySelector('.becca-presence__placeholder');
  if (!ph && _stage) { ph = _buildPlaceholder(); _stage.appendChild(ph); }
  _activateInto(_stage, _vrmUrl, ph);
}

function _persistAvatarChoice(choice) {
  _uSet(AVATAR_CHOICE_STORAGE_KEY, JSON.stringify(choice));
}

function _restoreAvatarChoice() {
  try {
    const raw = _uGet(AVATAR_CHOICE_STORAGE_KEY);
    if (!raw) return null;
    const c = JSON.parse(raw);
    if (c && typeof c.vrm_url === 'string') return c;
  } catch (_) { /* corrupted localStorage — caller falls back to default */ }
  return null;
}

function _timeAgo(ts) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

function _setTimelineOpen(open) {
  if (!_root || !_timelinePanel) return;
  if (open) {
    _renderTimeline();
    _timelinePanel.hidden = false;
    _timelinePanel.setAttribute('aria-hidden', 'false');
    _root.dataset.timelineOpen = 'true';
    _bumpTimelineAutoClose();
    document.addEventListener('click', _onTimelineOutsideClick, { capture: true });
  } else {
    _timelinePanel.hidden = true;
    _timelinePanel.setAttribute('aria-hidden', 'true');
    _root.dataset.timelineOpen = 'false';
    if (_timelineCloseTimer) { clearTimeout(_timelineCloseTimer); _timelineCloseTimer = null; }
    document.removeEventListener('click', _onTimelineOutsideClick, { capture: true });
  }
}

function _onTimelineOutsideClick(e) {
  // Close on click anywhere outside the widget root.
  if (_root && !_root.contains(e.target)) _setTimelineOpen(false);
}

function _bumpTimelineAutoClose() {
  if (_timelineCloseTimer) clearTimeout(_timelineCloseTimer);
  // Auto-close after 12s of no interaction inside the panel.
  _timelineCloseTimer = setTimeout(() => _setTimelineOpen(false), 12000);
}

// ── PTT button ────────────────────────────────────────────────────
//
// Bottom-left of the dock — mirrors the audio-mode toggle on the right.
// Stage 1 (tonight): tapping opens voice mode pre-armed in PTT input
// mode, so the user can talk immediately without picking PTT from the
// voice overlay.
//
// Stage 2/3 (next sprint, task #138): the press handler swaps to a
// no-overlay path that talks to /ws/voice?persona_id=becca directly
// from the widget and streams response audio back through the existing
// TTS analyser. The button itself doesn't change — only what its press
// handler does. window.__beccaPttPress is the swappable seam.

/**
 * Talk-mode cycle. One affordance with two states:
 *
 *   off  — silent. Click to enter auto. Hold still triggers one-shot PTT.
 *   auto — wake-word listener is running. Saying the trained phrase opens
 *          a capture turn; server VAD ends it; BeccaVoice replies. Click
 *          again to return to off. Hold still works for a deliberate
 *          one-shot regardless of mode.
 *
 * Both this button and Settings → Companion → Wake word converge on the
 * ``becca.wake.enabled`` localStorage flag + the ``becca:wake-prefs-changed``
 * DOM event, so flipping either surface restarts the wake session.
 */
function _toggleTalkMode() {
  _setTalkMode(_talkMode === 'auto' ? 'off' : 'auto');
}

function _setTalkMode(mode) {
  if (mode !== 'off' && mode !== 'auto') return;
  _talkMode = mode;
  _uSet(TALK_MODE_STORAGE_KEY, mode);
  // Flip the shared wake-enabled flag; both this button and the Settings
  // panel listen for ``becca:wake-prefs-changed`` to (re)start the session.
  try {
    _uSet('becca.wake.enabled', mode === 'auto' ? 'true' : 'false');
    window.dispatchEvent(new CustomEvent('becca:wake-prefs-changed'));
  } catch (_) { /* private browsing / quota — wake pref is non-essential */ }
  if (_pttBtnEl) {
    _pttBtnEl.dataset.talkMode = mode;
    const iconEl = _pttBtnEl.querySelector('.becca-presence__ptt-icon');
    if (iconEl) iconEl.innerHTML = _pttIconSvg(mode);
    const label = mode === 'auto'
      ? 'Auto — listening for wake word. Click to turn off. Hold to talk now.'
      : 'Off — click for auto-listen. Hold to talk now.';
    _pttBtnEl.setAttribute('aria-label', label);
    _pttBtnEl.title = label;
  }
}

function _restoreTalkMode() {
  let mode = null;
  try {
    const saved = _uGet(TALK_MODE_STORAGE_KEY);
    if (saved === 'auto' || saved === 'off') mode = saved;
  } catch (_) { /* unreadable storage — mode stays null; migration below covers it */ }
  // Migration: if talk_mode was never written but the legacy
  // ``becca.wake.enabled`` flag is on (user enabled wake via the Settings
  // panel before this cycle button existed), carry that into auto so we
  // don't silently disable their listener.
  if (mode === null) {
    try {
      mode = _uGet('becca.wake.enabled') === 'true' ? 'auto' : 'off';
    } catch (_) { mode = 'off'; }
  }
  _setTalkMode(mode);
}

function _pttIconSvg(mode) {
  if (mode === 'auto') {
    return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round"
      stroke-linejoin="round" aria-hidden="true">
      <line x1="3" y1="12" x2="3" y2="12"/>
      <line x1="7" y1="9" x2="7" y2="15"/>
      <line x1="11" y1="5" x2="11" y2="19"/>
      <line x1="15" y1="8" x2="15" y2="16"/>
      <line x1="19" y1="11" x2="19" y2="13"/>
      <line x1="21" y1="12" x2="21" y2="12"/>
    </svg>`;
  }
  return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="3" width="6" height="11" rx="3"/>
    <path d="M5 11a7 7 0 0 0 14 0"/>
    <line x1="12" y1="18" x2="12" y2="22"/>
    <line x1="8" y1="22" x2="16" y2="22"/>
  </svg>`;
}

function _buildPttButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__ptt';
  btn.dataset.pttState = 'idle';
  btn.dataset.talkMode = _talkMode;
  // Icon lives in its own span so _setTalkMode can swap the SVG without
  // touching the button's other state-driven attributes.
  const iconWrap = document.createElement('span');
  iconWrap.className = 'becca-presence__ptt-icon';
  iconWrap.innerHTML = _pttIconSvg(_talkMode);
  btn.appendChild(iconWrap);

  // Dual gesture: short click toggles talk mode (off ↔ auto, wake-word
  // listener); hold (≥TAP_MS) is a classic PTT one-shot regardless of
  // mode. Threshold-based — if the user releases within TAP_MS the press
  // is a click; otherwise the press has already escalated to a hold and
  // started recording. Hold still works in auto mode for a deliberate
  // override — useful when the wake model misses the trained phrase.
  const TAP_MS = 180;
  let _holdTimer = null;
  let _holdFired = false;
  let _pressTs = 0;

  const beginHold = () => {
    _holdFired = true;
    _onPttDown();
  };

  btn.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    e.preventDefault();
    try { btn.setPointerCapture(e.pointerId); } catch (_) {}
    _holdFired = false;
    _pressTs = performance.now();
    if (_holdTimer) clearTimeout(_holdTimer);
    _holdTimer = setTimeout(beginHold, TAP_MS);
  });

  const releaseHandler = (e) => {
    e.stopPropagation();
    if (_holdTimer) { clearTimeout(_holdTimer); _holdTimer = null; }
    if (_holdFired) {
      // Hold path — release ends the PTT capture.
      _onPttUp();
    } else {
      // Click path. If a wake-triggered auto-capture is in flight, cancel
      // it (the user changed their mind). Otherwise toggle talk mode:
      // off → auto (wake listener starts) or auto → off (wake listener
      // disposes).
      const sess = _pttSession;
      if (sess && sess._autoCaptureMode && sess._state === 'recording') {
        try { sess.captureStop(); } catch (_) {}
      } else {
        _toggleTalkMode();
      }
    }
  };
  btn.addEventListener('pointerup', releaseHandler);
  btn.addEventListener('pointercancel', releaseHandler);
  // Don't release on pointerleave — pointer capture keeps events
  // flowing to the button, but some browsers fire leave first.

  // Keyboard support — same tap/hold semantics. Holding the key
  // produces auto-repeat keydown events; the first one starts the
  // hold timer, repeats are filtered out.
  btn.addEventListener('keydown', (e) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      _holdFired = false;
      _pressTs = performance.now();
      if (_holdTimer) clearTimeout(_holdTimer);
      _holdTimer = setTimeout(beginHold, TAP_MS);
    }
  });
  btn.addEventListener('keyup', (e) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      if (_holdTimer) { clearTimeout(_holdTimer); _holdTimer = null; }
      if (_holdFired) {
        _onPttUp();
      } else {
        const sess = _pttSession;
        if (sess && sess._autoCaptureMode && sess._state === 'recording') {
          try { sess.captureStop(); } catch (_) {}
        } else {
          _toggleTalkMode();
        }
      }
    }
  });

  _pttBtnEl = btn;
  return btn;
}

// ── Wake-word listening ────────────────────────────────────────────
//
// Opt-in by design. Always-on mic listening is a strong default for a
// personal AI OS. The primary entry point is the talk-mode cycle button
// in the dock — click flips to auto, which sets ``becca.wake.enabled``
// and dispatches ``becca:wake-prefs-changed``. The Settings → Companion
// → Wake word panel is the secondary entry and writes the same flag.
//
// Avatar-id selection still lives in localStorage until a dedicated
// picker lands (#118):
//
//   localStorage.setItem('becca.wake.avatar_ids', JSON.stringify(['wake-hey-samantha']))
//
// On detection, the trained phrase arms BeccaPttSession via
// ``triggerWakeCapture`` for an open-mic turn (server VAD ends it), so
// there's no modal — she just answers.

function _wakeEnabled() {
  // In always-listening mode the wake-word detector is bypassed entirely —
  // continuous STT + server address classifier replaces it. We still
  // honour the user's localStorage toggle for wake_word + ptt_only modes,
  // but always_listening overrides it to off so the wake WS doesn't
  // run alongside the always-listening capture (would double-claim mic).
  if (_alwaysListeningMode()) return false;
  try { return _uGet('becca.wake.enabled') === 'true'; }
  catch (_) { return false; }
}

// Read the server-side companion_activation_mode setting that becca-
// bootstrap.js exposes on window.__beccaSettings. Defaults to
// "wake_word" when settings haven't loaded yet so the legacy behaviour
// is the safe fallback.
function _alwaysListeningMode() {
  try {
    const m = (window.__beccaSettings || {}).companion_activation_mode;
    return (m || "wake_word").toLowerCase() === "always_listening";
  } catch (_) { return false; }
}

// Follow-up mode is opt-out — once the user has enabled wake-word, the
// natural-conversation default is to NOT require the wake word for every
// turn. localStorage 'false' explicitly disables; absence means on.
function _followUpEnabled() {
  try { return _uGet('becca.followup.enabled') !== 'false'; }
  catch (_) { return true; }
}

function _followUpWindowMs() {
  try {
    const raw = _uGet('becca.followup.window_s');
    if (raw) {
      const n = parseInt(raw, 10);
      if (Number.isFinite(n) && n >= 5 && n <= 120) return n * 1000;
    }
  } catch (_) { /* unreadable storage — return default */ }
  return FOLLOWUP_DEFAULT_WINDOW_MS;
}

function _wakeAvatarIds() {
  try {
    const raw = _uGet('becca.wake.avatar_ids');
    if (raw) {
      const arr = JSON.parse(raw);
      if (Array.isArray(arr) && arr.length) return arr;
    }
  } catch (_) { /* corrupted storage — return default avatar set */ }
  return ['wake-hey-samantha'];
}

/** Sync ``data-wake-state`` attribute on the PTT button. */
function _updateWakeStateAttribute(state) {
  if (!_pttBtnEl) return;
  _pttBtnEl.dataset.wakeState = state;
}

// ── Always-listening capture loop ────────────────────────────────
// In always_listening mode, the BeccaPttSession stays continuously
// armed: triggerWakeCapture opens a capture window, server VAD detects
// the utterance boundary, the address classifier on the server
// decides addressed-vs-ambient, and on every turn-complete /
// turn-aborted we re-arm. The user never has to invoke wake-word or
// press PTT — they just talk naturally, and Becca only responds
// when actually addressed.

let _alwaysListeningArmed = false;
let _alwaysListeningRearmTimer = null;
let _alwaysListeningWatchdogTimer = null;
let _alwaysListeningVisibilityHandler = null;
// Long capture window — VAD will still finalize on natural utterance
// boundaries (silence + smart-turn), but the silence-no-speech timer
// in BeccaPttSession needs a generous bound so we don't bail every
// few seconds during quiet stretches.
const ALWAYS_LISTENING_MAX_MS = 30000;
const ALWAYS_LISTENING_SILENCE_MS = 25000;
const ALWAYS_LISTENING_REARM_DELAY_MS = 200;
// Hard watchdog: if neither turn-complete nor turn-aborted arrives
// within this many ms after we arm, force a rearm. triggerWakeCapture
// returns the instant it sends start_recording; from then on the only
// thing clearing _alwaysListeningArmed is one of those two events. WS
// drops, mobile tab suspends, and server stalls would otherwise leave
// the flag stuck true forever — silently killing always-on. The
// watchdog is the floor that keeps the loop alive no matter what.
const ALWAYS_LISTENING_WATCHDOG_MS = ALWAYS_LISTENING_MAX_MS + 5000;

function _clearAlwaysListeningWatchdog() {
  if (_alwaysListeningWatchdogTimer) {
    clearTimeout(_alwaysListeningWatchdogTimer);
    _alwaysListeningWatchdogTimer = null;
  }
}

// ── iOS first-gesture audio unlock ───────────────────────────────────
// iOS Safari refuses to unlock an AudioContext without a genuine user
// gesture, and accepting the mic-permission prompt does NOT count. Wake /
// always-listening capture auto-starts at page load (no gesture), so on
// iPad the context stays suspended and no PCM frames reach the server — the
// mic looks open but nothing is heard (and a frame-less capture sits stuck
// in 'recording'). We can't beat the iOS rule, but we can unlock on the very
// first interaction and then cleanly (re)start whichever mode the user chose.
let _audioPrimed = false;
function _primeAudioOnFirstGesture() {
  if (_audioPrimed) return;
  const handler = () => {
    if (_audioPrimed) return;
    _audioPrimed = true;
    document.removeEventListener('pointerdown', handler, true);
    document.removeEventListener('touchend', handler, true);
    // Unlock synchronously — awaiting anything here drops the gesture on iOS.
    try { _ensurePttSession()?.primeAudioSync?.(); } catch (_) {}
    try { _wakeSession?.primeAudioSync?.(); } catch (_) {}
    // Now that audio can unlock, (re)engage the user's chosen mode.
    try {
      if (_alwaysListeningMode()) {
        // The page-load arm couldn't unlock audio and is stuck in a
        // frame-less 'recording' state. Tear it down, then rearm cleanly.
        const sess = _pttSession;
        try { if (sess && sess._state === 'recording') sess.captureStop(); } catch (_) {}
        _clearAlwaysListeningWatchdog();
        _alwaysListeningArmed = false;
        _scheduleAlwaysListeningRearm();
      } else if (_wakeEnabled()) {
        const ws = _ensureWakeSession();
        if (ws && (ws.state === 'idle' || ws.state === 'error')) {
          ws.start().catch(() => {});
        }
      }
    } catch (_) {}
  };
  // Capture phase + both event types so any first touch/click anywhere
  // unlocks — the user shouldn't have to hit a specific control.
  document.addEventListener('pointerdown', handler, true);
  document.addEventListener('touchend', handler, true);
}

async function _ensureAlwaysListening() {
  if (!_alwaysListeningMode()) return;
  const session = _ensurePttSession();
  if (!session) {
    console.info('[becca] always-listening: no ptt session, deferring');
    return;
  }
  if (_alwaysListeningArmed) return;
  if (_wakePausedForCall) return;  // call modal owns the mic
  _alwaysListeningArmed = true;
  console.info('[becca] always-listening: arming capture', {
    ptt_state: session.state,
  });
  _clearAlwaysListeningWatchdog();
  _alwaysListeningWatchdogTimer = setTimeout(() => {
    _alwaysListeningWatchdogTimer = null;
    if (_alwaysListeningArmed && _alwaysListeningMode() && _root) {
      console.warn('[becca] always-listening watchdog fired — forcing rearm');
      _scheduleAlwaysListeningRearm();
    }
  }, ALWAYS_LISTENING_WATCHDOG_MS);
  try {
    await session.triggerWakeCapture({
      maxMs: ALWAYS_LISTENING_MAX_MS,
      silenceMs: ALWAYS_LISTENING_SILENCE_MS,
      // Open-mic re-arm, not a deliberate press/wake — the server's
      // ambient address gate stays on for these captures.
      source: 'auto',
    });
  } catch (err) {
    console.warn('[becca] always-listening capture failed', err);
    _clearAlwaysListeningWatchdog();
    _alwaysListeningArmed = false;
    // Capture-start itself failed (mic denied, WS closed, etc). Don't
    // give up — keep trying on the rearm cadence so a transient failure
    // doesn't permanently kill always-on.
    _scheduleAlwaysListeningRearm();
  }
}

function _scheduleAlwaysListeningRearm() {
  if (!_alwaysListeningMode()) return;
  _clearAlwaysListeningWatchdog();
  _alwaysListeningArmed = false;
  if (_alwaysListeningRearmTimer) clearTimeout(_alwaysListeningRearmTimer);
  // Small delay so back-to-back turn boundaries don't busy-loop the
  // mic acquisition. Also gives the previous turn's TTS tail time to
  // finish playing, since server-side echo cooldown still applies.
  _alwaysListeningRearmTimer = setTimeout(() => {
    _alwaysListeningRearmTimer = null;
    if (!_root) return;  // widget unmounted
    _ensureAlwaysListening().catch(() => {});
  }, ALWAYS_LISTENING_REARM_DELAY_MS);
}

// Diagnostic helper — call __beccaAlwaysListeningState() from DevTools
// to see why the loop isn't running. Catches the four failure modes:
// settings haven't loaded, mode is wrong, PTT session not constructed,
// flag stuck armed without a rearm event.
window.__beccaAlwaysListeningState = () => ({
  mode_setting: (window.__beccaSettings || {}).companion_activation_mode || null,
  mode_active: _alwaysListeningMode(),
  armed: _alwaysListeningArmed,
  paused_for_call: _wakePausedForCall,
  has_root: !!_root,
  has_ptt_session: !!_pttSession,
  ptt_state: _pttSession?.state || null,
  rearm_pending: !!_alwaysListeningRearmTimer,
  watchdog_pending: !!_alwaysListeningWatchdogTimer,
});

function _ensureWakeSession() {
  if (_wakeSession) return _wakeSession;
  if (!_wakeEnabled()) return null;
  _wakeSession = new BeccaWakeSession({
    avatarIds: _wakeAvatarIds(),
    echoCancelDisabled: _micAecDisabled(),
    onWake: (detection) => {
      console.info('[becca] wake fired', detection);
      try { window.__beccaFlashWake?.(detection.phrase); } catch (_) {}
      // Engage follow-up mode for the duration of the upcoming turn (and
      // any follow-up turns within the window). Cleared on silence abort,
      // manual PTT, or dispose.
      _followUpActive = _followUpEnabled();
      // Inline path: arm the BeccaPttSession for an auto-capture turn
      // instead of opening the full voice-call modal. Server VAD
      // detects when the user finishes their question and auto-stops;
      // BeccaVoice composes the reply and TTS plays through her body
      // via chat/tts.js, same path the manual PTT button uses. No
      // overlay — she just answers.
      const session = _ensurePttSession();
      if (session?.triggerWakeCapture) {
        session.triggerWakeCapture().catch(err => {
          console.warn('[becca] wake capture failed', err);
        });
      } else {
        // Legacy fallback only if PTT session can't be set up
        // (mic permission revoked mid-session, etc.) — opens the
        // modal so the user at least gets a way to speak.
        try { window.__beccaTriggerVoiceCall?.(); } catch (_) {}
      }
    },
    onStateChange: (s) => {
      console.debug('[becca-wake] state ->', s);
      // Map BeccaWakeSession's state machine to the button's
      // ``data-wake-state``. The button's CSS keys off this for
      // pulsing/dim/error visuals.
      const map = {
        idle: 'off',
        connecting: 'connecting',
        listening: 'listening',
        paused: 'paused',
        error: 'error',
      };
      _updateWakeStateAttribute(map[s] || 'off');
    },
  });
  _wakeSession.start().catch((err) => {
    console.warn('[becca] wake start failed', err);
  });
  return _wakeSession;
}

function _ensurePttSession() {
  if (_pttSession) return _pttSession;
  const animationRouter = _ensureAnimationRouter();
  _pttSession = new BeccaPttSession({
    echoCancelDisabled: _micAecDisabled(),
    // Stage-manager: in 'stage' mode a finished utterance drafts into the
    // compose box instead of firing a turn. Null in 'voice' mode (legacy).
    onDraftTranscript: _inputMode === 'stage' ? _draftIntoCompose : null,
    onStateChange: (state, detail) => {
      if (_pttBtnEl) _pttBtnEl.dataset.pttState = state;
      try { animationRouter?.onPttStateChange(state); } catch (err) {
        console.warn('[becca] animation router state failed', err);
      }
      // Slice 1 — every PTT state change is user-relevant activity:
      // keep chrome alive so the user can see the status row pulse.
      try { _pokeChromeAlive(); } catch (_) {}
      // Bridge into interoception so her body reacts to the press.
      // The session itself doesn't know about interoception; this
      // module owns the coupling.
      try {
        if (state === 'recording') {
          // PTT is active — pause wake listening so the wake detector
          // doesn't fire on the user's own utterance (which always
          // gets through with the existing PTT mic path) or on Becca's
          // TTS playback that follows.
          if (_wakeResumeTimer) { clearTimeout(_wakeResumeTimer); _wakeResumeTimer = null; }
          if (_wakeSession && _wakeSession.state !== 'paused' && _wakeSession.state !== 'idle') {
            try { _wakeSession.pause(); } catch (_) {}
            _wakePausedForCall = true;
          }
        } else if (state === 'processing') {
          // Slice 1 — "did she hear me?" feedback. Briefly show
          // "heard you" before the existing thinking pulse so the
          // user has positive acknowledgement that STT finalised and
          // the request is in flight. _triggerHeardYou auto-advances
          // to 'thinking' after 520ms.
          _triggerHeardYou();
          // Drop the partial transcript chip the moment STT finalises —
          // the status row's "heard you" replaces it and prevents the
          // stale partial from sitting on screen while she thinks.
          _hideTranscript();
          // Thinking beat — user finished, we're waiting on the model.
          // The animation router owns the immediate pose-family handoff;
          // keep the conversation-state flag as the avatar.js fallback so
          // the time between input and first phoneme still reads as
          // deliberate consideration, not dead air.
          bus.set('becca_conversation', 'processing');
          try { avatarState.presence?._queueSyntheticAction?.('call_thoughtful_pause', 0.95); } catch (_) {}
        } else if (state === 'speaking') {
          // tts.js emits augmentum:tts-playback which interoception
          // already consumes for ``tts_start`` — no duplicate trigger.
          // Wake stays paused — her TTS would re-fire the detector
          // otherwise.
          bus.set('becca_conversation', 'speaking');
        } else if (state === 'armed' || state === 'idle') {
          // Turn complete — clear conversation-state flag so the
          // pose-intent loop returns to flow-driven idle families.
          // NOTE: follow-up re-arm + window expiry are driven by the
          // explicit ``becca-ptt:turn-complete`` / ``turn-aborted``
          // events (see listeners below) — NOT by this state
          // transition. Multi-sentence TTS bounces state through
          // ``speaking → armed`` per sentence inside _flushTtsChunks,
          // so we can't use ``armed`` as a turn-boundary signal.
          bus.set('becca_conversation', null);
          // Schedule wake resume on a 3s cooldown — but skip if a
          // follow-up window is active. While follow-up is alive the
          // mic belongs to PTT, not wake. The ``turn-aborted``
          // listener handles resume on silence.
          if (_wakePausedForCall && _wakeSession && !_followUpActive) {
            if (_wakeResumeTimer) clearTimeout(_wakeResumeTimer);
            _wakeResumeTimer = setTimeout(() => {
              _wakeResumeTimer = null;
              if (_wakeSession && _wakePausedForCall) {
                try { _wakeSession.resume(); } catch (_) {}
                _wakePausedForCall = false;
              }
            }, 3000);
          }
        } else if (state === 'error') {
          console.warn('[becca] PTT error:', detail?.message);
        }
      } catch (_) {}
    },
    onTranscript: (text, meta = {}) => {
      try { animationRouter?.onTranscript(text, meta); } catch (err) {
        console.warn('[becca] animation router transcript failed', err);
      }
      // Slice 1 — partial transcripts feed the chip so the user
      // sees her words land in real time. Final transcript is
      // covered by the 'heard you' status state; the chip is
      // hidden when 'processing' fires.
      try { _showTranscript(text); } catch (err) {
        console.warn('[becca] transcript chip failed', err);
      }
    },
    onLLMDelta: (text) => {
      try { animationRouter?.onLLMDelta(text); } catch (err) {
        console.warn('[becca] animation router delta failed', err);
      }
    },
    onTtsStart: (sentence) => {
      try { animationRouter?.onTtsStart(sentence); } catch (err) {
        console.warn('[becca] animation router tts-start failed', err);
      }
    },
    onTtsEnd: () => {
      try { animationRouter?.onTtsEnd(); } catch (err) {
        console.warn('[becca] animation router tts-end failed', err);
      }
    },
  });
  return _pttSession;
}

function _ensureAnimationRouter() {
  if (_animationRouter) return _animationRouter;
  _animationRouter = new CompanionAnimationRouter({
    conductor: movementConductor,
    avatarState,
    hooks: {
      onStateChange,
      onUserTranscript,
      onLLMDelta,
      onPoseShift: _onPoseShift,
    },
  });
  return _animationRouter;
}

// ── Pose-shift surfacing ─────────────────────────────────────────────
//
// CompanionAnimationRouter._applyPoseIntent fires `onPoseShift` every
// time a backend topic (voice.tool_call, voice.completed, affect.changed,
// channel.entering, etc.) sets a new posture. With quiet mode on, the
// VRMA layer is suppressed but the procedural pose layer still shifts.
// Without surfacing the verb, the user has no signal that her body
// reacted at all. We push it into the status row as a transient
// override so they see "thinking" / "settling" / "reaching out" briefly.
//
// Rules:
//   - Skip override when the base state is 'speaking' or 'hosting' —
//     those carry their own meaning and shouldn't be clobbered.
//   - Cap visible duration at min(durationMs, 5000) — pose families
//     run for 6-18s but the label gets stale after ~5s.
//   - Verbs that don't carry user meaning (transient internals like
//     `reach_out` are user-friendly, but `formal_behind` isn't) are
//     mapped through `_POSE_VERB_LABEL` to a clean phrase or skipped.
const _POSE_VERB_LABEL = {
  thinking:        'thinking',
  think:           'thinking',
  reflect:         'reflecting',
  create:          'creating',
  concerned:       'concerned',
  attentive:       'paying attention',
  listening:       'listening',
  listen:          'listening',
  engage:          'engaged',
  curious:         'curious',
  reach_out:       'reaching out',
  speaking:        '',     // speaking state already surfaces this
  speak:           '',
  settle:          'settling',
  settled:         'settled',
  present:         '',
  host:            '',     // hosting state already surfaces this
  formal:          '',
  formal_behind:   'stepping back',
  handoff:         'handing off',
  step_aside:      'stepping aside',
  wind_down:       'winding down',
  dormant:         'dormant',
  asleep:          'asleep',
  closed:          'closed off',
  boundary:        '',     // safety/boundary states route through other UI
  unsure:          'unsure',
  frustrated:      'frustrated',
  melancholy:      '',     // affect-driven; let the mood overlay carry this
  surface_attention: 'looking at the surface',
  media_attention: 'watching',
  world_attention: 'looking around',
  confident:       'confident',
};

let _poseShiftTimer = null;

function _onPoseShift(info) {
  if (!_root) return;
  // Don't fight the base state when she has something more important
  // to say (actively speaking or hosting media). Those win.
  if (_statusState === 'speaking' || _statusState === 'hosting') return;
  const verb = String(info?.verb || '').toLowerCase();
  if (!verb) return;
  const label = _POSE_VERB_LABEL[verb];
  if (!label) return;   // explicitly suppressed or unmapped verb
  _setStatusOverride(`· ${label}`);
  // Clear after a short window, capped at the pose's own duration.
  const durMs = Math.min(Math.max(Number(info?.durationMs) || 0, 1500), 5000);
  if (_poseShiftTimer) { clearTimeout(_poseShiftTimer); _poseShiftTimer = null; }
  _poseShiftTimer = setTimeout(() => {
    _poseShiftTimer = null;
    // Only clear if our override is still the active one — another
    // override (audio source, channel handoff) may have superseded.
    if (_statusOverride === `· ${label}`) _setStatusOverride(null);
  }, durMs);
}

// ── Follow-up window event listeners ─────────────────────────────────
// Driven by explicit signals from BeccaPttSession (turn-complete,
// turn-aborted) and the intent-action-router (conversation.close) so
// we don't have to disambiguate the per-sentence ``armed`` transitions
// that fire inside _flushTtsChunks for multi-sentence TTS replies.
document.addEventListener('becca-ptt:turn-complete', () => {
  // These listeners are registered at module-import time (not inside
  // mountBeccaPresence) so they outlive any dismiss. Guard on _root
  // so we don't schedule timers or run state machines against a
  // torn-down widget — the activation-mode and PTT session refs may
  // still be live in a stale way during the dismiss → resummon window.
  if (!_root) return;
  // Slice 1 — clear the transcript chip on turn-complete. The chip
  // hides on 'processing' too, but a turn that bypasses the partial-
  // STT path (intent_action shortcut) still needs an explicit clear
  // so a stale partial from a prior turn can't linger.
  try { _hideTranscript(); } catch (_) {}
  // Always-listening: re-arm capture for the next utterance
  // regardless of follow-up state. The whole point is continuous
  // capture with classifier filtering.
  if (_alwaysListeningMode()) {
    _scheduleAlwaysListeningRearm();
    return;
  }
  if (!_followUpActive) return;
  if (!_followUpEnabled()) {
    _followUpActive = false;
    return;
  }
  const session = _pttSession;
  if (!session?.triggerWakeCapture) {
    _followUpActive = false;
    return;
  }
  const windowMs = _followUpWindowMs();
  console.info('[becca] follow-up armed', { window_ms: windowMs });
  // Follow-up windows auto-open the mic after her reply — keep the
  // ambient address gate so TV/background noise can't claim the turn.
  session.triggerWakeCapture({ silenceMs: windowMs, source: 'followup' }).catch((err) => {
    console.warn('[becca] follow-up capture failed', err);
    _followUpActive = false;
  });
});

document.addEventListener('becca-ptt:decision', (e) => {
  // Track A — the server's routing verdict for this turn. Record it for the
  // opt-in HUD and flash a subtle per-goal tell. Module-level (like the
  // sibling becca-ptt listeners) so it survives a dismiss; _root-guarded so
  // it no-ops against a torn-down widget.
  if (!_root) return;
  const detail = e?.detail || {};
  try { _recordDecision(detail); } catch (_) { /* HUD is best-effort */ }
  try { _flashDecisionTell(detail); } catch (_) { /* tell is best-effort */ }
});

document.addEventListener('becca-ptt:no-speech', (e) => {
  // "I didn't catch that" — STT came back empty, or an explicit
  // press/wake capture was incoherent. The PTT session already
  // re-armed itself; we surface the hint in the transcript chip
  // (auto-clears on its 8s safety timer or the next turn's events)
  // and keep the always-listening loop alive.
  if (!_root) return;
  try {
    _showTranscript(e?.detail?.message || "Didn't catch that — try again?");
  } catch (_) {}
  if (_alwaysListeningMode()) {
    _scheduleAlwaysListeningRearm();
  }
});

document.addEventListener('becca-ptt:turn-aborted', (e) => {
  // Module-level — see guard rationale above on the turn-complete handler.
  if (!_root) return;
  // Slice 1 — clear any partial transcript on abort (silence/timeout).
  try { _hideTranscript(); } catch (_) {}
  // Near-miss: she HEARD coherent, reply-shaped speech that landed just
  // under the addressing bar (server's judgment). Show a faint, non-spoken
  // "heard you" so a turn she heard but chose not to answer is never a
  // silent void. Clearly-ambient drops (reason 'no-reply' — incoherent /
  // idle / well below the bar) stay silent so she doesn't flicker at every
  // word across the room. This is the trust repair that keeps always-on
  // honest without losing the gate's nuance.
  if (e?.detail?.near_miss) {
    try { _flashHeardAmbient(); } catch (_) {}
  }
  // Always-listening: silence-abort just means "nothing to say right
  // now" — re-arm immediately so a delayed utterance still gets
  // captured. No follow-up gating in this mode.
  if (_alwaysListeningMode()) {
    _scheduleAlwaysListeningRearm();
    return;
  }
  if (!_followUpActive) return;
  console.info('[becca] follow-up window expired', e.detail);
  _followUpActive = false;
  // Kick a wake resume directly — the state-machine path above only
  // runs if a fresh ``armed`` transition fires, which a silence abort
  // may not produce (state was already armed momentarily).
  if (_wakePausedForCall && _wakeSession) {
    if (_wakeResumeTimer) clearTimeout(_wakeResumeTimer);
    _wakeResumeTimer = setTimeout(() => {
      _wakeResumeTimer = null;
      if (_wakeSession && _wakePausedForCall) {
        try { _wakeSession.resume(); } catch (_) {}
        _wakePausedForCall = false;
      }
    }, 3000);
  }
});

// "Bye Becca" path — the intent-action-router emits this when the
// server-side ``control.goodbye`` action fires. Force-close the
// follow-up window even if a turn just landed (otherwise the
// turn-complete listener above would re-arm us straight away).
document.addEventListener('augmentum:intent', (e) => {
  // Module-level — see guard rationale on the turn-complete handler.
  if (!_root) return;
  if (e?.detail?.channel !== 'conversation.close') return;
  _followUpActive = false;
});

// Avatar motion cue from a chat reply ([motion:xxx], stripped client-side).
// Animate only when she's actually on screen; the router maps cue→roles and
// plays through the user's curated/rated/uploadable pool.
document.addEventListener('augmentum:motion-cue', (e) => {
  const cue = e?.detail?.cue;
  if (!cue || !_vrmActive) return;
  try { _ensureAnimationRouter()?.onMotionCue?.(cue); }
  catch (err) { console.debug('[becca] motion-cue failed', err?.message); }
});

async function _onPttDown() {
  // Manual PTT press is an explicit intent — exit any follow-up window
  // so wake listening resumes normally rather than auto-rearming a
  // wake-style capture.
  _followUpActive = false;
  // Stage 1 escape hatch: if a power user installed __beccaPttPress
  // (the swappable seam), defer to it. Otherwise use Stage 2 inline.
  if (typeof window.__beccaPttPress === 'function') {
    try { await window.__beccaPttPress(); } catch (_) {}
    return;
  }
  const session = _ensurePttSession();
  await session.captureStart();
}

function _onPttUp() {
  if (!_pttSession) return;
  _pttSession.captureStop();
}

// ── Stage-manager text input ─────────────────────────────────────────
// A compose bar under the dock. In 'stage' mode a spoken utterance drafts
// into the box (via BeccaPttSession.onDraftTranscript) instead of firing a
// turn, and the user can also type. Send routes through the SAME stage_send
// path a spoken turn uses (BeccaPttSession.sendText), so reply + TTS come
// back and the widget animates identically. 'voice' mode = legacy behaviour.

function _readInputMode() {
  try { return _uGet(INPUT_MODE_STORAGE_KEY) === 'stage' ? 'stage' : 'voice'; }
  catch (_) { return 'voice'; }
}

function _buildInputModeButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__input-mode';
  btn.dataset.mode = _inputMode;
  btn.setAttribute('aria-label', 'Toggle text input: voice-only or compose');
  // The dock is pointer-events:none; each child re-enables itself.
  btn.style.pointerEvents = 'auto';
  const iconWrap = document.createElement('span');
  iconWrap.className = 'becca-presence__input-mode-icon';
  iconWrap.innerHTML = _inputModeIcon(_inputMode);
  btn.appendChild(iconWrap);
  btn.addEventListener('click', () => {
    _applyInputMode(_inputMode === 'stage' ? 'voice' : 'stage');
  });
  return btn;
}

function _inputModeIcon(mode) {
  // 'stage' = keyboard glyph (compose active); 'voice' = speech bubble crossed
  // to a keyboard hint. Inline SVG, one selector deep, no extra fetch.
  if (mode === 'stage') {
    return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="2" y="6" width="20" height="12" rx="2"/>
      <path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M8 14h8"/>
    </svg>`;
  }
  return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>`;
}

function _buildComposeBar() {
  const bar = document.createElement('div');
  bar.className = 'becca-presence__compose';
  bar.hidden = true;
  bar.style.pointerEvents = 'auto';

  const input = document.createElement('textarea');
  input.className = 'becca-presence__compose-input';
  input.rows = 1;
  input.setAttribute('placeholder', 'Message — or hold to talk, then edit');
  input.setAttribute('aria-label', 'Message the companion');
  input.addEventListener('input', () => scheduleAutosize(input, 160));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      _sendCompose();
    } else if (e.key === 'Escape') {
      input.blur();
    }
  });
  _composeInputEl = input;

  const send = document.createElement('button');
  send.type = 'button';
  send.className = 'becca-presence__compose-send';
  send.setAttribute('aria-label', 'Send');
  send.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4z"/></svg>`;
  send.addEventListener('click', _sendCompose);

  bar.appendChild(input);
  bar.appendChild(send);
  return bar;
}

/** Draft a spoken transcript into the compose box (stage mode). Appends to
 *  any text already there so the user can speak in multiple passes. */
function _draftIntoCompose(text) {
  if (!_composeInputEl) return;
  const t = (text || '').trim();
  if (!t) return;
  const existing = _composeInputEl.value.trim();
  _composeInputEl.value = existing ? `${existing} ${t}` : t;
  scheduleAutosize(_composeInputEl, 160);
  try { _composeInputEl.focus(); } catch (_) {}
  try { _pokeChromeAlive(); } catch (_) {}
}

/** Send the composed text through the same stage_send turn a spoken utterance
 *  uses. Clears the box; reply + TTS stream back over the session WS. */
function _sendCompose() {
  if (!_composeInputEl) return;
  const text = _composeInputEl.value.trim();
  if (!text) {
    _composeInputEl.classList.add('shake');
    setTimeout(() => _composeInputEl?.classList.remove('shake'), 400);
    return;
  }
  const session = _ensurePttSession();
  _composeInputEl.value = '';
  scheduleAutosize(_composeInputEl, 160);
  try { Promise.resolve(session?.sendText?.(text)).catch(() => {}); } catch (_) {}
}

/** Apply an input mode: toggle the compose bar + wire (or clear) the PTT
 *  session's draft callback. Persists per-user unless savePrefs is false. */
function _applyInputMode(mode, { savePrefs = true } = {}) {
  _inputMode = mode === 'stage' ? 'stage' : 'voice';
  if (_inputModeBtn) {
    _inputModeBtn.dataset.mode = _inputMode;
    const icon = _inputModeBtn.querySelector('.becca-presence__input-mode-icon');
    if (icon) icon.innerHTML = _inputModeIcon(_inputMode);
  }
  if (_composeBarEl) _composeBarEl.hidden = _inputMode !== 'stage';
  // Wire the draft callback onto the live session (and any future one, via
  // _ensurePttSession which reads _inputMode when it constructs).
  if (_pttSession) {
    _pttSession.onDraftTranscript = _inputMode === 'stage' ? _draftIntoCompose : null;
    // Flip the server-side staging gate too — required for auto/streaming mode
    // (server-side STT), where the client-side draft callback alone never sees
    // the transcript.
    try { _pttSession.setStaging(_inputMode === 'stage'); } catch (_) {}
  }
  if (_inputMode === 'stage') {
    try { _composeInputEl?.focus(); } catch (_) {}
  }
  if (savePrefs) { try { _uSet(INPUT_MODE_STORAGE_KEY, _inputMode); } catch (_) {} }
}

function _buildAudioModeButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__audio-mode';
  btn.dataset.role = _audioRole;
  btn.setAttribute('aria-label', 'Toggle audio mode: assistant or host');
  // Icon lives in its own span so role swaps can mutate the SVG without
  // touching the button's other state-driven attributes — mirrors the
  // PTT button pattern at _buildPttButton().
  const iconWrap = document.createElement('span');
  iconWrap.className = 'becca-presence__audio-mode-icon';
  iconWrap.innerHTML = _audioRoleIcon(_audioRole);
  btn.appendChild(iconWrap);
  btn.addEventListener('click', _toggleAudioRole);
  return btn;
}

function _audioRoleIcon(role) {
  // Two inline SVGs — assistant (ear shape, "she's listening for you") vs.
  // host (headphones, "she's hosting your audio"). Inline so styling is
  // one CSS selector deep and there are no extra HTTP fetches.
  if (role === 'host') {
    return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 14v-2a8 8 0 0 1 16 0v2"/>
      <rect x="2" y="14" width="5" height="6" rx="1.5"/>
      <rect x="17" y="14" width="5" height="6" rx="1.5"/>
    </svg>`;
  }
  // Default: assistant — a small ear / sound icon
  return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M6 8.5a6 6 0 0 1 12 0c0 2.5-1.5 4-3 5s-2 1.5-2 3a2 2 0 0 1-4 0"/>
    <circle cx="10" cy="9" r="1.2" fill="currentColor"/>
  </svg>`;
}

function _toggleAudioRole() {
  _setAudioRole(_audioRole === 'assistant' ? 'host' : 'assistant');
}

// ── Mic DSP (echo-cancellation) toggle ───────────────────────────────
// Small button that lets the user turn the browser's mic echo-cancellation
// OFF. With an AEC-enabled capture stream live, Chrome/Android route the
// whole device output through their echo canceller, which low-passes
// (muffles) music/media the companion is playing. Turning it off keeps
// playback full-range; the trade-off is the mic can then hear playback, so
// wake-word/barge-in leans on server-side VAD. Off by default (AEC on).

function _buildMicDspButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__mic-dsp';
  const disabled = _micAecDisabled();
  btn.dataset.aec = disabled ? 'off' : 'on';
  const iconWrap = document.createElement('span');
  iconWrap.className = 'becca-presence__mic-dsp-icon';
  iconWrap.innerHTML = _micDspIcon(disabled);
  btn.appendChild(iconWrap);
  // The dock is pointer-events:none; each child re-enables itself. Force it
  // inline so clickability never depends on a fresh stylesheet (same guard
  // the eye button uses).
  btn.style.pointerEvents = 'auto';
  // Hidden for now — the echo-cancellation override (mic-device.js +
  // becca-wake.js + becca-ptt.js) stays fully wired for a future re-enable,
  // but the control is off the dock until AEC muffling is worth surfacing as
  // a user-facing choice. Flip to '' (or gate on a capability flag) to show.
  btn.style.display = 'none';
  _applyMicDspLabel(btn, disabled);
  btn.addEventListener('click', _toggleMicDsp);
  return btn;
}

function _micDspIcon(aecDisabled) {
  // Speaker with sound waves (full-range audio, AEC off) vs. a slashed
  // speaker (AEC on = playback muffled while listening). Inline SVG, one CSS
  // selector deep, no extra fetches — mirrors the audio-mode/eye icons.
  if (aecDisabled) {
    return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 9v6h4l5 4V5L8 9H4Z"/>
      <path d="M16 9a3 3 0 0 1 0 6"/>
      <path d="M19 6a7 7 0 0 1 0 12"/>
    </svg>`;
  }
  return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M4 9v6h4l5 4V5L8 9H4Z"/>
    <line x1="16" y1="9" x2="22" y2="15"/>
    <line x1="22" y1="9" x2="16" y2="15"/>
  </svg>`;
}

function _applyMicDspLabel(btn, aecDisabled) {
  const label = aecDisabled
    ? 'Echo cancellation off — music stays full-range while listening. Click to re-enable.'
    : 'Echo cancellation on. Click to turn off so music isn’t muffled while the companion listens.';
  btn.setAttribute('aria-label', label);
  btn.setAttribute('aria-pressed', aecDisabled ? 'true' : 'false');
  btn.title = label;
}

function _toggleMicDsp() {
  const next = !_micAecDisabled();
  _uSet(MIC_AEC_DISABLED_KEY, next ? 'true' : 'false');
  if (_micDspBtn) {
    _micDspBtn.dataset.aec = next ? 'off' : 'on';
    const iconEl = _micDspBtn.querySelector('.becca-presence__mic-dsp-icon');
    if (iconEl) iconEl.innerHTML = _micDspIcon(next);
    _applyMicDspLabel(_micDspBtn, next);
  }
  // Apply live to BOTH companion mic paths — the PTT mic (only open during a
  // hold) and the wake mic (open the whole time she's passively listening;
  // THIS is the one that muffles playback while music plays). PTT re-acquires
  // in place; the wake session is restarted (dispose + recreate) since its
  // PCM worklet can't cleanly re-bind to a fresh source node in place.
  try { _pttSession?.setEchoCancelDisabled?.(next); } catch (_) { /* best effort */ }
  try {
    if (_wakeSession) {
      _wakeSession.dispose();
      _wakeSession = null;
      _wakePausedForCall = false;
      if (_wakeEnabled()) {
        _updateWakeStateAttribute('connecting');
        _ensureWakeSession();
      }
    }
  } catch (_) { /* best effort */ }
}

// ── Live-camera "eye" ────────────────────────────────────────────────
// A small eye next to the ear. Pressing it opens the user's camera into
// the stage (camera-fill backdrop + the VRM shrunk to a corner PIP) and
// streams frames to the companion via the PTT session's live-vision loop.
// Gated on companion_live_vision_enabled so it's invisible on deployments
// that haven't opted in (the server ignores frames there anyway).

function _liveVisionEnabled() {
  return !!(window.__beccaSettings || {}).companion_live_vision_enabled;
}

function _buildEyeButton() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__eye';
  btn.dataset.camState = 'off';
  btn.setAttribute('aria-label', 'Show the companion your camera');
  const iconWrap = document.createElement('span');
  iconWrap.className = 'becca-presence__eye-icon';
  iconWrap.innerHTML = _eyeIcon(false);
  btn.appendChild(iconWrap);
  // The dock is pointer-events:none and each button re-enables itself via
  // a stylesheet rule. CSS caches separately from JS, so if an older
  // becca-presence.css is cached the eye isn't in that allowlist and taps
  // pass through it (visible but dead). Force it inline so clickability
  // never depends on the stylesheet being fresh.
  btn.style.pointerEvents = 'auto';
  btn.addEventListener('click', _toggleCameraView);
  // Visible whenever the capability is enabled. The composite works even
  // before the VRM finishes loading (camera fills; the canvas joins as a
  // corner PIP automatically once it's live), so we don't gate on
  // _vrmActive — that just hid the eye when the avatar was slow/failed.
  btn.style.display = _liveVisionEnabled() ? '' : 'none';
  return btn;
}

function _eyeIcon(on) {
  // Open eye (camera on) vs eye with a slash (off). Inline SVG, one CSS
  // selector deep, no extra fetches — mirrors the audio-mode icon pattern.
  if (on) {
    return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>`;
  }
  return `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M2 12s3.5-7 10-7c1.6 0 3 .4 4.3 1"/>
    <path d="M20.8 8.4A17 17 0 0 1 22 12s-3.5 7-10 7c-1.6 0-3-.4-4.3-1"/>
    <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/>
    <line x1="3" y1="3" x2="21" y2="21"/>
  </svg>`;
}

function _refreshEyeVisibility() {
  if (!_eyeBtn) return;
  const on = _liveVisionEnabled();
  _eyeBtn.style.display = on ? '' : 'none';
  // If the capability was disabled while the camera was on, tear it down.
  if (_cameraView && !on) _stopCameraView();
}

function _setEyeState(on) {
  if (!_eyeBtn) return;
  _eyeBtn.dataset.camState = on ? 'on' : 'off';
  const iconEl = _eyeBtn.querySelector('.becca-presence__eye-icon');
  if (iconEl) iconEl.innerHTML = _eyeIcon(on);
  _eyeBtn.setAttribute('aria-label',
    on ? 'Turn off the camera' : 'Show the companion your camera');
  // Camera takes the WHOLE screen: fullscreen the widget root. The stage
  // can't escape on its own — the root has contain:layout + a drag transform
  // that trap position:fixed children — so we expand the root itself (CSS:
  // .becca-presence--camera) and the camera fills it, VRM → corner PIP.
  if (_root) _root.classList.toggle('becca-presence--camera', on);
}

async function _toggleCameraView() {
  console.info('[becca] eye tapped — liveVision=', _liveVisionEnabled(),
    'hasStage=', !!_stage, 'cameraOn=', !!_cameraView,
    'mediaDevices=', !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));
  if (!_liveVisionEnabled()) return;
  if (_cameraView) { _stopCameraView(); return; }
  if (!_stage) { console.warn('[becca] eye: no _stage to mount camera'); return; }
  const view = new CompanionCameraView({
    host: _stage,
    onStream: (stream) => {
      // Wire the live stream into the PTT session's frame loop (sends
      // video_frame over the voice WS). On stop (stream === null) the loop
      // is torn down in _stopCameraView.
      try {
        if (stream) _ensurePttSession()?.startLiveVision?.(stream);
      } catch (_) { /* best effort */ }
    },
    onError: (err) => console.warn('[becca] camera open failed', err),
  });
  _cameraView = view;
  const ok = await view.start({ facingMode: 'user' });
  if (!ok) { _cameraView = null; return; }
  _setEyeState(true);
  _ensureFlipButton(view);
}

function _stopCameraView() {
  try { _ensurePttSession()?.stopLiveVision?.(); } catch (_) {}
  if (_cameraView) { try { _cameraView.stop(); } catch (_) {} _cameraView = null; }
  if (_flipBtn) { try { _flipBtn.remove(); } catch (_) {} _flipBtn = null; }
  _setEyeState(false);
  // Camera released → re-sync the surface/layout that was frozen while filming.
  try { if (_root) _applySurface(_detectSurface()); } catch (_) { /* best effort */ }
}

async function _ensureFlipButton(view) {
  // Front<->back flip — only meaningful with >1 camera (mobile). Overlaid
  // on the stage so it's reachable while the camera fills the view.
  if (!_stage || _flipBtn) return;
  let multi = false;
  try { multi = await view.hasMultipleCameras(); } catch (_) { multi = false; }
  if (!multi || !_cameraView) return;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__cam-flip';
  btn.setAttribute('aria-label', 'Flip camera (front / back)');
  btn.innerHTML = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M3 7h13l-2.5-2.5M21 17H8l2.5 2.5"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>`;
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_cameraView) _cameraView.flip();
  });
  _flipBtn = btn;
  _stage.appendChild(btn);
}

function _setAudioRole(role) {
  if (role !== 'assistant' && role !== 'host') return;
  _audioRole = role;
  _uSet(AUDIO_ROLE_STORAGE_KEY, role);
  window.__beccaAudioRole = role;
  if (_audioModeBtn) {
    _audioModeBtn.dataset.role = role;
    const iconEl = _audioModeBtn.querySelector('.becca-presence__audio-mode-icon');
    if (iconEl) iconEl.innerHTML = _audioRoleIcon(role);
    _audioModeBtn.setAttribute('aria-label',
      role === 'host'
        ? 'Audio host mode — media plays through her. Click to return to assistant.'
        : 'Assistant mode — listens for you. Click to enter host mode.');
  }
  if (_root) _root.dataset.audioRole = role;
  // Status row baseline follows the role unless an active state
  // (speaking, hosting media) is already overriding.
  if (_statusState !== 'speaking' && !_statusOverride) {
    _setStatusState(role === 'host' ? 'hosting' : 'idle');
  }
  // Flipping out of host mode while dancing → stop. Flipping into host
  // mode while media is already playing → AudioBus event hasn't re-fired,
  // so synthesize a re-evaluation by reading the bus's debug snapshot.
  if (role !== 'host' && _danceActive) {
    _stopDanceLoop();
  } else if (role === 'host' && !_danceActive) {
    try {
      const bus = window.AudioBus;
      if (bus && typeof bus.debug === 'function') {
        const snap = bus.debug();
        const activeSources = (snap.sources || []).filter(s => s.active);
        const activeTiers = activeSources.map(s => s.tier);
        const activeKinds = activeSources.map(s => s.kind).filter(Boolean);
        _maybeUpdateDanceLoop(activeTiers, activeKinds);
      }
    } catch (_) { /* AudioBus not available — harmless */ }
  }
}

function _restoreAudioRole() {
  let role = 'assistant';
  try {
    const saved = _uGet(AUDIO_ROLE_STORAGE_KEY);
    if (saved === 'host' || saved === 'assistant') role = saved;
  } catch (_) { /* unreadable — keep default role */ }
  _setAudioRole(role);
}

// ── Status row + wake-flash ────────────────────────────────────────
//
// The status row replaces the old "real talk" pill at the bottom. It
// adapts based on what's happening: by default it shows her state
// (idle / listening / thinking / speaking), when audio is playing it
// shows the source, and when a wake word fires it briefly flashes the
// wake-flash overlay near the top of the widget.
//
// pointer-events:none in CSS means drag still works through the row.
// Real-talk affordance moves to a long-press anywhere on the widget.

const _STATUS_LABELS = Object.freeze({
  idle: 'idle',
  listening: 'listening',
  heard: 'heard you',
  thinking: 'thinking',
  speaking: 'speaking',
  hosting: 'hosting',
});

function _buildStatusRow() {
  const el = document.createElement('div');
  el.className = 'becca-presence__status';
  el.setAttribute('aria-live', 'polite');
  el.dataset.state = 'idle';
  el.textContent = _STATUS_LABELS.idle;
  return el;
}

// Affect indicator — Synapse §2 visibility surface. A single line in
// italic muted type that surfaces what she's been picking up on the
// user. Hidden when she doesn't have a confident read; otherwise
// reads like a quiet noticing. Updates on a 90s cadence (cheap
// /api/companion/affect_read poll); silent failure when unreachable.
//
// Tasteful by design: small text, no badge, no animation. Just a
// short italic phrase under the status row. Visible to users who
// look at the widget; invisible otherwise.
let _affectRow = null;
let _affectPollTimer = null;
const _AFFECT_POLL_INTERVAL_MS = 90_000;

function _buildAffectRow() {
  const el = document.createElement('div');
  el.className = 'becca-presence__affect';
  el.setAttribute('aria-live', 'polite');
  el.hidden = true;
  return el;
}

async function _refreshAffectRead() {
  if (!_affectRow) return;
  let data = null;
  try {
    const resp = await fetch('/api/companion/affect_read', {
      credentials: 'same-origin',
    });
    if (resp.ok) data = await resp.json();
  } catch (_) {
    // Silent — leaving the existing read on screen is better than
    // erasing it because the network blipped.
    return;
  }
  const obs = data && data.observation;
  if (!obs || !obs.phrase) {
    _affectRow.hidden = true;
    _affectRow.textContent = '';
    return;
  }
  // Format: "she's picking up: soft today" (italic via CSS).
  // Hedged reads get a soft suffix.
  const suffix = obs.hedged ? ' — could be wrong' : '';
  _affectRow.textContent = `picking up: ${obs.phrase}${suffix}`;
  _affectRow.dataset.tag = obs.tag || '';
  _affectRow.hidden = false;
}

// ───────────────────────────────────────────────────────────────────
// Slice 1: transcript chip + "heard you" microcopy
//
// "Did she actually hear me?" was the daily-use pain point. The chip
// surfaces partial STT in real-time as a small italic bubble above the
// status row, then fades on turn-complete. The status row briefly
// shows "heard you" with a sage dot when STT finalizes, before flowing
// into "thinking". Both close the perceptual gap between the user
// finishing their utterance and Becca responding.

function _buildTranscriptChip() {
  const el = document.createElement('div');
  el.className = 'becca-presence__transcript';
  el.setAttribute('aria-live', 'polite');
  el.setAttribute('aria-atomic', 'true');
  el.dataset.visible = '0';
  return el;
}

function _showTranscript(text) {
  if (!_transcriptChip) return;
  const trimmed = String(text || '').trim();
  if (!trimmed) {
    _hideTranscript();
    return;
  }
  const display = _truncateGraceful(trimmed, TRANSCRIPT_MAX_CHARS);
  // textContent assignment auto-escapes — no HTML can leak from a
  // model-generated transcript.
  _transcriptChip.textContent = display;
  _transcriptChip.dataset.visible = '1';
  // Safety: even with explicit turn-complete handling, a stuck-open
  // chip is worse than a missed hide. 8s wall-clock max display.
  if (_transcriptClearTimer) clearTimeout(_transcriptClearTimer);
  _transcriptClearTimer = setTimeout(_hideTranscript, 8000);
}

function _hideTranscript() {
  if (!_transcriptChip) return;
  _transcriptChip.dataset.visible = '0';
  if (_transcriptClearTimer) {
    clearTimeout(_transcriptClearTimer);
    _transcriptClearTimer = null;
  }
}

// ── "Read this …" context handoff chip ───────────────────────────────
//
// Hidden until the foreground surface registers loadable content
// (companion-context.js::setCompanionLoadable). Pressing it hands her
// the full page/chat/file so deixis + quoting are grounded; the server
// stashes it for context_peek('loaded') so only a digest hits her prompt.

function _buildContextChip() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__context-chip';
  btn.dataset.visible = '0';
  btn.addEventListener('click', _onContextChipClick);
  return btn;
}

function _attachLoadableListener() {
  if (_loadableHandler) return;
  _loadableHandler = (ev) => {
    const detail = (ev && ev.detail) || {};
    _renderContextChip(Boolean(detail.active), detail.label || '');
  };
  document.addEventListener('augmentum:loadable-changed', _loadableHandler);
}

function _detachLoadableListener() {
  if (!_loadableHandler) return;
  document.removeEventListener('augmentum:loadable-changed', _loadableHandler);
  _loadableHandler = null;
}

function _renderContextChip(active, label) {
  if (!_contextChip) return;
  if (active && label && !_loadableBusy) {
    _contextChip.textContent = label;
    _contextChip.dataset.visible = '1';
    _contextChip.disabled = false;
  } else if (!_loadableBusy) {
    _contextChip.dataset.visible = '0';
  }
}

async function _onContextChipClick() {
  if (!_contextChip || _loadableBusy) return;
  _loadableBusy = true;
  const original = _contextChip.textContent;
  _contextChip.textContent = 'one sec…';
  _contextChip.disabled = true;
  try {
    const m = await import('./companion-context.js');
    const res = await m.loadCompanionContext();
    _loadableBusy = false;
    if (res && res.ok) {
      // Confirm + fold away — the affordance's job is done for this view.
      _contextChip.textContent = 'got it ✓';
      try { window.__augmentum?.showToast?.('Caught her up.', 'info', 1800); } catch (_) {}
      setTimeout(() => { if (_contextChip) _contextChip.dataset.visible = '0'; }, 1100);
    } else {
      _contextChip.textContent = original;
      _contextChip.disabled = false;
      try { window.__augmentum?.showToast?.("Couldn't read that one.", 'info', 2000); } catch (_) {}
    }
  } catch (err) {
    _loadableBusy = false;
    if (_contextChip) {
      _contextChip.textContent = original;
      _contextChip.disabled = false;
    }
    console.warn('[becca-presence] context load failed', err);
  }
}

// "heard you" — 520ms interstitial between STT-final and thinking.
// Called by _ensurePttSession's onStateChange when transitioning into
// 'processing'. The status row flows: listening → heard → thinking →
// speaking → idle. CSS supplies the sage-dot pulse.
function _triggerHeardYou() {
  if (!_statusRow) return;
  _setStatusState('heard');
  if (_heardYouTimer) clearTimeout(_heardYouTimer);
  _heardYouTimer = setTimeout(() => {
    _heardYouTimer = null;
    // Only advance to 'thinking' if we're still in the 'heard'
    // interstitial — another state transition (e.g. speaking) may
    // have overtaken us.
    if (_statusState === 'heard') _setStatusState('thinking');
  }, HEARD_YOU_HOLD_MS);
}

// Faint "I heard you, wasn't sure that was for me" tell. Rendered on a
// NEAR-MISS turn-abort — the server heard coherent, reply-shaped speech
// that landed just under the addressing bar. Deliberately quiet:
//   - reuses the status-row OVERRIDE slot (same channel as audio-source /
//     verb-tick labels), so it never enters the 'heard → thinking' pulse
//     (that interstitial promises a reply; a near-miss does not), and
//   - is NEVER spoken (a TTS reply would defeat the address gate's whole
//     point — she shouldn't answer what wasn't for her).
// Clears itself after a short beat. Persona-agnostic copy ("you" = the
// user, not the companion) per the UI-string rule.
const NEAR_MISS_HOLD_MS = 1400;
const _NEAR_MISS_LABEL = '· heard you';
let _nearMissTimer = null;
function _flashHeardAmbient() {
  if (!_statusRow) return;
  if (_nearMissTimer) { clearTimeout(_nearMissTimer); _nearMissTimer = null; }
  _setStatusOverride(_NEAR_MISS_LABEL);
  _nearMissTimer = setTimeout(() => {
    _nearMissTimer = null;
    // Only clear if our label is still the active override — an audio
    // source / channel handoff may have superseded it in the meantime.
    if (_statusOverride === _NEAR_MISS_LABEL) _setStatusOverride(null);
  }, NEAR_MISS_HOLD_MS);
}

// ── Track A: voice decision legibility (tell + HUD) ───────────────────
//
// The server emits a ``voice_decision`` over the voice WS the instant it
// classifies a turn (act / converse / clarify / idle / drop) — BEFORE any
// reply or dispatch. becca-ptt.js re-emits it as a ``becca-ptt:decision``
// DOM event. We render two things from it: a subtle per-goal status-row
// tell (always on), and an opt-in HUD listing the recent verdicts. Both
// answer "did she hear me, and what did she decide?" without log-diving.
//
// converse/clarify get NO tell on purpose — they already flow
// heard→thinking→speaking with a spoken reply, so a tell would double up.
const _DECISION_TELL_LABELS = Object.freeze({
  act: '· on it',
  idle: '· got it',
  drop: '· not for me',
});

function _decisionTellFor(detail) {
  const goal = detail?.goal || 'drop';
  // A near-miss drop (coherent, reply-shaped, just under the bar) earns the
  // warmer "heard you" rather than the flat "not for me" — she nearly took
  // it, and that nuance repairs trust (same label as the near-miss flash).
  if (goal === 'drop' && detail?.nearMiss) return _NEAR_MISS_LABEL;
  return _DECISION_TELL_LABELS[goal] || null;
}

function _flashDecisionTell(detail) {
  if (!_statusRow) return;
  const label = _decisionTellFor(detail);
  if (!label) return;
  // Confident ambient drop is the high-frequency case (background TV / side
  // talk in always-listening) — keep it faintest + shortest so it reads as a
  // flicker-tick, not an announcement. Matt opted into ambient tells.
  const ambientDrop =
    detail?.goal === 'drop' && !detail?.nearMiss && !detail?.explicit;
  if (_decisionTellTimer) { clearTimeout(_decisionTellTimer); _decisionTellTimer = null; }
  // Faintness is bound to THIS label in _renderStatus, so a superseding
  // override (verb tick, audio source) can never inherit the fade.
  _statusTellFaintLabel = ambientDrop ? label : null;
  _setStatusOverride(label);
  const hold = ambientDrop ? DECISION_TELL_AMBIENT_HOLD_MS : DECISION_TELL_HOLD_MS;
  _decisionTellTimer = setTimeout(() => {
    _decisionTellTimer = null;
    _statusTellFaintLabel = null;
    if (_statusOverride === label) _setStatusOverride(null);
    else _renderStatus();   // someone else owns the row — just drop the fade
  }, hold);
}

function _hudEnabled() {
  try { return (window.__beccaSettings || {}).companion_voice_decision_hud === true; }
  catch (_) { return false; }
}

function _buildDecisionHud() {
  const el = document.createElement('div');
  el.className = 'becca-presence__decision-hud';
  el.dataset.visible = '0';
  // Diagnostic surface — keep it out of the AT tree; the status-row tell +
  // spoken replies are the accessible signal path.
  el.setAttribute('aria-hidden', 'true');
  return el;
}

function _fmtDecisionClock(ts) {
  try {
    const d = new Date(ts);
    const p = (n) => String(n).padStart(2, '0');
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  } catch (_) { return ''; }
}

function _renderDecisionHud() {
  if (!_decisionHudEl) return;
  const show = _hudEnabled();
  _decisionHudEl.dataset.visible = show ? '1' : '0';
  if (!show) { _decisionHudEl.replaceChildren(); return; }
  if (!_decisionLog.length) {
    const empty = document.createElement('div');
    empty.className = 'becca-presence__decision-empty';
    empty.textContent = 'listening — no turns yet';
    _decisionHudEl.replaceChildren(empty);
    return;
  }
  const rows = _decisionLog.map((d) => {
    const row = document.createElement('div');
    row.className = 'becca-presence__decision-row';
    row.dataset.goal = d.goal;
    const time = document.createElement('span');
    time.className = 'becca-presence__decision-time';
    time.textContent = _fmtDecisionClock(d.ts);
    const goal = document.createElement('span');
    goal.className = 'becca-presence__decision-goal';
    goal.textContent = d.goal;
    const conf = document.createElement('span');
    conf.className = 'becca-presence__decision-conf';
    conf.textContent = (typeof d.conf === 'number') ? d.conf.toFixed(2) : '';
    const text = document.createElement('span');
    text.className = 'becca-presence__decision-text';
    // textContent (not a template literal) — transcript is user speech, so
    // this is the XSS-safe path by construction.
    text.textContent = d.transcript ? `"${d.transcript}"` : '';
    row.append(time, goal, conf, text);
    return row;
  });
  _decisionHudEl.replaceChildren(...rows);
}

function _recordDecision(detail) {
  _decisionLog.unshift({
    ts: Date.now(),
    goal: detail?.goal || 'drop',
    conf: typeof detail?.confidence === 'number' ? detail.confidence : null,
    transcript: detail?.transcript || '',
    addressed: !!detail?.addressed,
    nearMiss: !!detail?.nearMiss,
  });
  if (_decisionLog.length > DECISION_LOG_CAP) _decisionLog.length = DECISION_LOG_CAP;
  _renderDecisionHud();
}

// Persist the HUD toggle through the same settings API the rest of the
// widget uses (mirror of _toggleAlwaysListeningPref) so it survives reloads
// and syncs across devices.
async function _toggleDecisionHudPref() {
  const next = !_hudEnabled();
  try {
    const resp = await fetch('/api/config/tools', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ companion_voice_decision_hud: next }),
    });
    if (!resp.ok) {
      console.warn('[becca] decision-hud write rejected', resp.status);
      return;
    }
  } catch (err) {
    console.warn('[becca] decision-hud write failed', err);
    return;
  }
  window.__beccaSettings = window.__beccaSettings || {};
  window.__beccaSettings.companion_voice_decision_hud = next;
  _renderDecisionHud();
  _refreshMiniControls();
}

// ── Slice 1: verb dispatch tick toast ────────────────────────────────
//
// Closes the loop on delegated-mode actions ("she did the thing"). The
// intent-action-router emits a ``becca:verb-fired`` document event
// every time a server-side action resolves and reaches the dispatch
// surface. We render a brief tick in the status row — sage check-dot
// plus a friendly verb label ("noted ✓" / "casting ✓" / "opening ✓").
// Lives for VERB_TICK_HOLD_MS, then the status row returns to whatever
// state was active beneath it.

const _VERB_TICK_LABELS = Object.freeze({
  'tts.cancel':            'stopped',
  'tts.repeat_last':       'replaying',
  'tts.resynth_last':      'replaying',
  'tts.volume_bump':       'volume up',
  'turn.abort':            'cancelled',
  'conversation.close':    'goodbye',
  'navigate.open_surface': 'opening',
  'navigate.back':         'going back',
  'note.open_sticky':      'note up',
  'note.update_sticky':    'noted',
  'note.capture_started':  'capturing',
  'note.capture_ended':    'captured',
  'grove.play':            'playing',
  'image.generate':        'generating',
  'browse.search':         'searching',
  'browse.open_url':       'opening',
  'discovery.open':        'opening discovery',
  'timer.set':             'timer set',
  'media.resume':          'resuming',
  'media.transport':       'playback',
  'files.open':            'opening file',
  'files.search_open':     'searching files',
  // Slice 2 chat verb
  'chat.new':              'new chat',
  // Companion Direct Action — media.play's near-tie offer cards
  'companion.candidates':  'your pick',
});

function _verbTickLabel(channel) {
  if (!channel || typeof channel !== 'string') return 'done';
  if (_VERB_TICK_LABELS[channel]) return _VERB_TICK_LABELS[channel];
  // Default: strip namespace, replace punctuation, append nothing
  // ("media.transport.pause" → "transport pause").
  const tail = channel.split('.').slice(-2).join(' ').replace(/_/g, ' ');
  return tail || 'done';
}

function _showVerbTick(channel, customLabel) {
  if (!_statusRow) return;
  const label = customLabel || _verbTickLabel(channel);
  if (!label) return;
  _statusRow.dataset.tick = '1';
  // Override the visible label briefly. _setStatusOverride is the
  // existing slot the status row uses for "hosting audio" mid-stream
  // labels, so the tick reuses that channel cleanly.
  _setStatusOverride(`${label} ✓`);
  if (_verbTickTimer) clearTimeout(_verbTickTimer);
  _verbTickTimer = setTimeout(() => {
    _verbTickTimer = null;
    if (_statusRow) _statusRow.removeAttribute('data-tick');
    _setStatusOverride(null);
  }, VERB_TICK_HOLD_MS);
  // Acting on a goal counts as activity — keep chrome alive for the
  // user to see the tick.
  _pokeChromeAlive();
}

function _attachVerbTickListener() {
  if (_verbFiredHandler) return;
  _verbFiredHandler = (ev) => {
    const detail = (ev && ev.detail) || {};
    _showVerbTick(detail.channel || '', detail.label || '');
  };
  document.addEventListener('becca:verb-fired', _verbFiredHandler);
}

function _detachVerbTickListener() {
  if (!_verbFiredHandler) return;
  document.removeEventListener('becca:verb-fired', _verbFiredHandler);
  _verbFiredHandler = null;
  if (_verbTickTimer) { clearTimeout(_verbTickTimer); _verbTickTimer = null; }
}

// ── Slice 1: chrome melt-away ────────────────────────────────────────
//
// After 4s of no interaction AND no live state, the dock fades to
// 0.18 opacity. Hovering / focusing / touching / typing into the
// widget pokes it alive instantly. Live states (listening / heard /
// thinking / speaking) automatically suppress idle so the user can
// always see what she's doing during a turn. CSS handles the visual
// fade; this code drives the data-chrome-idle attribute + timers.
//
// PTT button stays at 0.18 along with the rest (still findable by
// position memory; reveal on hover/touch makes it crisp).

function _isLiveStatusState() {
  return _statusState === 'listening'
      || _statusState === 'heard'
      || _statusState === 'thinking'
      || _statusState === 'speaking';
}

function _setChromeIdle(idle) {
  if (!_root) return;
  if (idle && _isLiveStatusState()) idle = false;
  _root.dataset.chromeIdle = idle ? '1' : '0';
}

function _pokeChromeAlive() {
  _setChromeIdle(false);
  if (_chromeIdleTimer) clearTimeout(_chromeIdleTimer);
  _chromeIdleTimer = setTimeout(() => {
    _chromeIdleTimer = null;
    if (!_isLiveStatusState()) _setChromeIdle(true);
  }, CHROME_IDLE_MS);
}

function _attachChromeIdle() {
  if (_chromeActivityHandler || !_root) return;
  _chromeActivityHandler = () => _pokeChromeAlive();
  // pointerdown captures clicks even on the dock (which has its own
  // handlers) because the root listener runs first. focusin covers
  // keyboard navigation. We deliberately omit mousemove (too chatty
  // and would prevent idle from ever settling on hover).
  _root.addEventListener('pointerdown', _chromeActivityHandler, { passive: true });
  _root.addEventListener('pointerenter', _chromeActivityHandler, { passive: true });
  _root.addEventListener('focusin', _chromeActivityHandler);
  _root.addEventListener('touchstart', _chromeActivityHandler, { passive: true });
  // Start at 'alive' — first idle settle happens 4s after mount.
  _pokeChromeAlive();
}

function _detachChromeIdle() {
  if (!_chromeActivityHandler || !_root) return;
  _root.removeEventListener('pointerdown', _chromeActivityHandler);
  _root.removeEventListener('pointerenter', _chromeActivityHandler);
  _root.removeEventListener('focusin', _chromeActivityHandler);
  _root.removeEventListener('touchstart', _chromeActivityHandler);
  _chromeActivityHandler = null;
  if (_chromeIdleTimer) { clearTimeout(_chromeIdleTimer); _chromeIdleTimer = null; }
}

// ── Slice 1: long-press → mini-controls drawer ───────────────────────
//
// 450ms long-press anywhere on the widget reveals a small inline strip
// above the dock with three handles: talk-mode toggle, always-listening
// toggle, and a settings handle. Surfaces opt-in controls without
// exposing them in default chrome. Auto-hides after 12s of no
// interaction within the drawer.
//
// We deliberately listen on pointerdown/up rather than touchstart/end
// so the same handler works on mouse + touch + pen. Drag handlers
// already use pointerdown on the root, so we coordinate by only
// arming the long-press timer when the press hasn't already been
// consumed as a drag-start (the drag handler sets a flag via
// dataset.dragging).

function _buildMiniControls() {
  const el = document.createElement('div');
  el.className = 'becca-presence__mini-controls';
  el.dataset.visible = '0';
  el.setAttribute('role', 'toolbar');
  el.setAttribute('aria-label', 'Companion quick controls');
  // Talk mode (off / auto wake follow-up)
  const talkBtn = _buildMiniBtn('talk', 'talk: auto', 'Toggle wake-follow-up between off and auto');
  talkBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const next = _talkMode === 'auto' ? 'off' : 'auto';
    try { _setTalkMode(next); } catch (err) { console.warn('[becca] talk-mode toggle failed', err); }
    _refreshMiniControls();
    _bumpMiniControlsTimer();
  });
  // Always-listening (companion_activation_mode)
  const listenBtn = _buildMiniBtn('listen', 'listen: off', 'Toggle always-listening: when on, she hears you without a PTT press');
  listenBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleAlwaysListeningPref().catch(err =>
      console.warn('[becca] always-listening toggle failed', err));
    _bumpMiniControlsTimer();
  });
  // Decision HUD toggle (companion_voice_decision_hud) — shows the last
  // few routing verdicts so the user can SEE what she decided per turn.
  const hudBtn = _buildMiniBtn('hud', 'hud: off', 'Toggle the voice decision HUD: shows act/converse/idle/drop per turn');
  hudBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _toggleDecisionHudPref().catch(err =>
      console.warn('[becca] decision-hud toggle failed', err));
    _bumpMiniControlsTimer();
  });
  // Settings handle (opens companion-self modal)
  const settingsBtn = _buildMiniBtn('settings', 'settings', 'Open companion settings');
  settingsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _hideMiniControls();
    import('./settings.js').then(m => m.openSettings?.()).catch(err =>
      console.warn('[becca] settings open failed', err));
  });
  el.appendChild(talkBtn);
  el.appendChild(listenBtn);
  el.appendChild(hudBtn);
  el.appendChild(settingsBtn);
  // Hover anywhere in the strip resets the auto-hide timer so the
  // user has time to read + decide. Once they tap outside, the
  // timeout takes over.
  el.addEventListener('pointerenter', () => _bumpMiniControlsTimer());
  el.addEventListener('pointermove', () => _bumpMiniControlsTimer());
  return el;
}

function _buildMiniBtn(slot, label, title) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'becca-presence__mini-btn';
  btn.dataset.slot = slot;
  btn.setAttribute('title', title);
  btn.setAttribute('aria-label', title);
  const dot = document.createElement('span');
  dot.className = 'becca-presence__mini-btn-dot';
  btn.appendChild(dot);
  const lbl = document.createElement('span');
  lbl.className = 'becca-presence__mini-btn-label';
  lbl.textContent = label;
  btn.appendChild(lbl);
  return btn;
}

function _refreshMiniControls() {
  if (!_miniControlsEl) return;
  const talkBtn = _miniControlsEl.querySelector('[data-slot="talk"]');
  if (talkBtn) {
    talkBtn.dataset.active = _talkMode === 'auto' ? '1' : '0';
    const lbl = talkBtn.querySelector('.becca-presence__mini-btn-label');
    if (lbl) lbl.textContent = _talkMode === 'auto' ? 'talk: auto' : 'talk: off';
  }
  const listenBtn = _miniControlsEl.querySelector('[data-slot="listen"]');
  if (listenBtn) {
    const on = _alwaysListeningMode();
    listenBtn.dataset.active = on ? '1' : '0';
    const lbl = listenBtn.querySelector('.becca-presence__mini-btn-label');
    if (lbl) lbl.textContent = on ? 'listen: on' : 'listen: off';
  }
  const hudBtn = _miniControlsEl.querySelector('[data-slot="hud"]');
  if (hudBtn) {
    const on = _hudEnabled();
    hudBtn.dataset.active = on ? '1' : '0';
    const lbl = hudBtn.querySelector('.becca-presence__mini-btn-label');
    if (lbl) lbl.textContent = on ? 'hud: on' : 'hud: off';
  }
}

function _revealMiniControls() {
  if (!_miniControlsEl) return;
  _refreshMiniControls();
  _miniControlsEl.dataset.visible = '1';
  _pokeChromeAlive();
  _bumpMiniControlsTimer();
}

function _hideMiniControls() {
  if (!_miniControlsEl) return;
  _miniControlsEl.dataset.visible = '0';
  if (_miniControlsHideTimer) {
    clearTimeout(_miniControlsHideTimer);
    _miniControlsHideTimer = null;
  }
}

function _bumpMiniControlsTimer() {
  if (_miniControlsHideTimer) clearTimeout(_miniControlsHideTimer);
  _miniControlsHideTimer = setTimeout(() => {
    _miniControlsHideTimer = null;
    _hideMiniControls();
  }, MINI_CONTROLS_TIMEOUT_MS);
}

// Toggle companion_activation_mode via the existing settings API so
// the change persists across devices and is honoured by all surfaces
// that read the flag. Mirrors the pattern in settings.js — write +
// fire the same prefs-changed event the existing wake-prefs listener
// already consumes.
async function _toggleAlwaysListeningPref() {
  const current = _alwaysListeningMode();
  const next = current ? 'wake_word_only' : 'always_listening';
  const body = JSON.stringify({ companion_activation_mode: next });
  try {
    const resp = await fetch('/api/config/tools', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body,
    });
    if (!resp.ok) {
      console.warn('[becca] activation-mode write rejected', resp.status);
      return;
    }
  } catch (err) {
    console.warn('[becca] activation-mode write failed', err);
    return;
  }
  // Mirror to the local cache the rest of the widget reads. The
  // settings panel uses the same cache key, so any open settings
  // surface stays in sync. window is always defined here (we
  // already passed the mount + _root check via reach-through).
  window.__beccaSettings = window.__beccaSettings || {};
  window.__beccaSettings.companion_activation_mode = next;
  window.dispatchEvent(new CustomEvent('becca:wake-prefs-changed', {
    detail: { companion_activation_mode: next },
  }));
  _refreshMiniControls();
}

function _attachLongPressHandle() {
  if (!_root) return;
  const onDown = (ev) => {
    // Skip if the press landed on a button — those have their own
    // interactions (PTT, dismiss, audio-mode, mini-control buttons,
    // resize corners). Long-press is a "press the body" gesture.
    if (ev.target?.closest('button, .becca-presence__resize')) return;
    if (ev.target?.closest('.becca-presence__mini-controls')) return;
    // Drag handler may take this press — give it priority by deferring
    // the long-press timer briefly. If a drag starts, the timer is
    // cleared on the first pointermove.
    if (_longPressTimer) clearTimeout(_longPressTimer);
    _longPressPointerId = ev.pointerId;
    _longPressTimer = setTimeout(() => {
      _longPressTimer = null;
      _revealMiniControls();
    }, LONGPRESS_MS);
  };
  const onMoveOrUp = () => {
    if (_longPressTimer) {
      clearTimeout(_longPressTimer);
      _longPressTimer = null;
    }
    _longPressPointerId = null;
  };
  _root.addEventListener('pointerdown', onDown, { passive: true });
  _root.addEventListener('pointermove', onMoveOrUp, { passive: true });
  _root.addEventListener('pointerup', onMoveOrUp, { passive: true });
  _root.addEventListener('pointercancel', onMoveOrUp, { passive: true });
  // Refresh mini-controls when settings prefs change (e.g. settings
  // panel toggled always-listening) so the strip stays in sync.
  _miniControlsPrefsHandler = () => { _refreshMiniControls(); _renderDecisionHud(); };
  window.addEventListener('becca:wake-prefs-changed', _miniControlsPrefsHandler);
}

function _detachLongPressHandle() {
  // Listeners on _root are dropped automatically when _root is
  // removed in unmountBeccaPresence — we just clean up the timer
  // + the window-level prefs handler.
  if (_longPressTimer) { clearTimeout(_longPressTimer); _longPressTimer = null; }
  _longPressPointerId = null;
  if (_miniControlsHideTimer) {
    clearTimeout(_miniControlsHideTimer);
    _miniControlsHideTimer = null;
  }
  if (_miniControlsPrefsHandler) {
    try { window.removeEventListener('becca:wake-prefs-changed', _miniControlsPrefsHandler); } catch (_) {}
    _miniControlsPrefsHandler = null;
  }
}

// ───────────────────────────────────────────────────────────────────
// Visibility pause — stop the 3D render loop + soft animations + idle
// polls when the tab is hidden. Resume on visible.
//
// What pauses (the cost we don't pay while hidden):
//   - Three.js setAnimationLoop (the 60fps rAF + WebGL render). Biggest
//     single perf win — pre-fix, this ran continuously in background
//     tabs.
//   - 90s affect_read poll.
//   - 5s heartbeat watchdog (the VRM-alive check).
//   - 7-14s auto-glance scheduler (raycast + animator update).
//   - CSS keyframe animations (via the ``is-tab-hidden`` class which
//     applies ``animation-play-state: paused`` site-wide).
//
// What stays running (the *feature* of a persistent companion):
//   - Wake-word WebSocket (so "hey Becca" from any visible tab works).
//   - Always-listening PTT capture (same logic).
//   - Presence WebSocket (server can still push events for when the
//     user returns).
//
// Idempotent — multiple visibilitychange events with the same state
// produce one pause or one resume.
// ───────────────────────────────────────────────────────────────────

function _onVisibilityChange() {
  if (document.hidden) _pauseForVisibility();
  else _resumeFromVisibility();
}

// Combined render gate. Render is active only when the tab is visible AND
// the widget isn't occluded by a surface. Both the visibility handler and
// the surface handler funnel through here so they can't fight (e.g. the tab
// becoming visible while a voice call still occludes the widget must NOT
// resume the render loop). Idempotent.
function _shouldRender() {
  return !document.hidden && !_occluded;
}

function _updateRenderGate() {
  const want = _shouldRender();
  if (!want && !_renderPaused) {
    _renderPaused = true;
    try { pauseAvatarRender(); } catch (_) {}
    _stopLoadProbe();  // render loop is paused — nothing to adapt
  } else if (want && _renderPaused) {
    _renderPaused = false;
    try { resumeAvatarRender(); } catch (_) {}
    if (_vrmActive && _root) _startLoadProbe();  // re-arm load-adaptive capping
  }
}

// Surface occlusion hook — called from _applySurface / __beccaHideForCall.
// A surface that display:none's the widget (voice call) should pause the
// render loop even though the tab is foregrounded.
function _setOccluded(occluded) {
  occluded = !!occluded;
  if (_occluded === occluded) return;
  _occluded = occluded;
  _updateRenderGate();
}

function _pauseForVisibility() {
  if (_pausedForVisibility) return;
  _pausedForVisibility = true;
  _updateRenderGate();  // render + load probe owned by the gate
  _stopAffectPoll();
  if (_heartbeatTimer) { clearInterval(_heartbeatTimer); _heartbeatTimer = null; }
  if (_glanceAutoTimer) { clearTimeout(_glanceAutoTimer); _glanceAutoTimer = null; }
  if (_root) _root.classList.add('is-tab-hidden');
}

function _resumeFromVisibility() {
  if (!_pausedForVisibility) return;
  _pausedForVisibility = false;
  if (_root) _root.classList.remove('is-tab-hidden');
  _updateRenderGate();  // resumes render ONLY if not still occluded
  // Resume the timers we stopped. Guard each against the widget
  // having been unmounted while hidden (rare but possible — e.g.
  // user dismisses via the pip while in a different tab).
  if (_root) {
    _startAffectPoll();
    _startHeartbeat();
    _scheduleAutoGlance();
  }
}

function _attachVisibilityPause() {
  if (_visibilityHandler) return;
  _visibilityHandler = _onVisibilityChange;
  document.addEventListener('visibilitychange', _visibilityHandler);
}

function _detachVisibilityPause() {
  if (!_visibilityHandler) return;
  document.removeEventListener('visibilitychange', _visibilityHandler);
  _visibilityHandler = null;
  _pausedForVisibility = false;
}

/**
 * Map the smoothed event-loop lag onto a target frame rate and push it to
 * the render loop. Responsive page (lag <= GOOD) → 30fps; saturated (lag >=
 * BAD) → 8fps; linear in between. While she's actually talking we never drop
 * below the speaking floor so lipsync stays legible. No-op until the VRM is
 * live. setAvatarFrameCap is idempotent, so steady-state calls are free.
 */
function _applyAdaptiveFrameCap() {
  if (!_vrmActive) return;
  const lag = _lagEmaMs;
  let fps;
  if (lag <= _LAG_GOOD_MS) {
    fps = _FPS_HEADROOM;
  } else if (lag >= _LAG_BAD_MS) {
    fps = _FPS_SATURATED;
  } else {
    const t = (lag - _LAG_GOOD_MS) / (_LAG_BAD_MS - _LAG_GOOD_MS);
    fps = Math.round(_FPS_HEADROOM - t * (_FPS_HEADROOM - _FPS_SATURATED));
  }
  // Keep talking watchable even under load — visemes at <20fps read as a
  // stutter. (Heavy load rarely coincides with speech: when she's the focus
  // the page is usually quiet, so this floor seldom binds.)
  if (fps < _FPS_SPEAKING_FLOOR && bus?.state?.is_speaking) {
    fps = _FPS_SPEAKING_FLOOR;
  }
  try { setAvatarFrameCap(fps); } catch (_) {}
}

/**
 * Self-rescheduling lag probe. Each tick measures how late it fired vs the
 * scheduled interval (= main-thread contention), folds it into an EMA, and
 * re-derives the frame cap. setTimeout-chained (not setInterval) so a stall
 * doesn't produce a burst of catch-up callbacks that muddy the measurement.
 * Runs only while the VRM is live and the tab is visible.
 */
function _startLoadProbe() {
  if (_loadProbeTimer) return;
  _loadProbeLast = performance.now();
  _applyAdaptiveFrameCap(); // seed at headroom (lag EMA starts at 0)
  const tick = () => {
    _loadProbeTimer = null;
    const now = performance.now();
    const lateBy = Math.max(0, (now - _loadProbeLast) - _LOAD_PROBE_MS);
    _loadProbeLast = now;
    // alpha 0.3 — responsive enough to catch a flurry within ~1s, damped
    // enough that one spike doesn't slam the rate.
    _lagEmaMs = _lagEmaMs * 0.7 + lateBy * 0.3;
    _applyAdaptiveFrameCap();
    if (_vrmActive && _shouldRender()) {
      _loadProbeTimer = setTimeout(tick, _LOAD_PROBE_MS);
    }
  };
  _loadProbeTimer = setTimeout(tick, _LOAD_PROBE_MS);
}

function _stopLoadProbe() {
  if (_loadProbeTimer) { clearTimeout(_loadProbeTimer); _loadProbeTimer = null; }
  _lagEmaMs = 0;
}

function _startAffectPoll() {
  if (_affectPollTimer) clearInterval(_affectPollTimer);
  _refreshAffectRead();
  _affectPollTimer = setInterval(_refreshAffectRead, _AFFECT_POLL_INTERVAL_MS);
}

function _stopAffectPoll() {
  if (_affectPollTimer) {
    clearInterval(_affectPollTimer);
    _affectPollTimer = null;
  }
}

function _buildWakeFlash() {
  const el = document.createElement('div');
  el.className = 'becca-presence__wake-flash';
  el.setAttribute('aria-hidden', 'true');
  return el;
}

// Transient bubble shown above the status row when a channel.exited event
// arrives with non-empty microcopy. Mirrors the wake-flash pattern —
// JS adds/removes data-return-flash="true" on _root to toggle visibility.
function _buildReturnFlash() {
  const el = document.createElement('div');
  el.className = 'becca-presence__return-flash';
  el.setAttribute('aria-live', 'polite');
  return el;
}

function _renderStatus() {
  if (!_statusRow) return;
  // Live interaction states (listening/thinking/speaking) preempt the
  // mood overlay so what's happening RIGHT NOW always wins. When the
  // live state goes back to idle/hosting, the mood label (if still
  // within its TTL) shows again.
  const liveStates = new Set(['listening', 'thinking', 'speaking']);
  const moodActive = _moodOverlay && _moodOverlay.expiresAt > Date.now();
  const showMood = moodActive && !liveStates.has(_statusState);
  let label = showMood
    ? _moodOverlay.label
    : (_statusOverride || _STATUS_LABELS[_statusState] || _statusState);
  // Surface the active host loop name when hosting — gives the user a
  // standing reminder of which preferred set is driving the dance pool.
  // Override and mood still win above; this only adorns the base
  // ``hosting`` label.
  if (!showMood && !_statusOverride && _statusState === 'hosting' && _activeLoopId) {
    const loop = _loopsCache.find(l => l.id === _activeLoopId);
    if (loop?.name) {
      label = `${label} · ${loop.name}`;
    }
  }
  _statusRow.textContent = label;
  _statusRow.dataset.state = _statusState;
  // Track A — fade only the specific confident-ambient-drop tell, never a
  // label that superseded it.
  _statusRow.dataset.tell =
    (_statusOverride && _statusOverride === _statusTellFaintLabel) ? 'faint' : '';
  if (showMood) {
    _statusRow.dataset.mood = _moodOverlay.tag || '';
    if (_moodOverlay.body) _statusRow.title = _moodOverlay.body;
    else _statusRow.removeAttribute('title');
  } else {
    delete _statusRow.dataset.mood;
    _statusRow.removeAttribute('title');
  }
}

function _setStatusState(state) {
  if (!_STATUS_LABELS.hasOwnProperty(state)) return;
  _statusState = state;
  _renderStatus();
}

function _setStatusOverride(text) {
  _statusOverride = text || null;
  _renderStatus();
}

// ── Mood overlay (companion.state notifications) ────────────────────
//
// Phase 3c narrate_state_to_user emits notifications on
// ``channel_id="companion.state"``. Rather than fire a generic toast,
// notifications.js routes them here so the mood lives in the same
// slot as idle/hosting. Falls back to a toast when this widget isn't
// mounted (see notifications.js).
//
// Tag values mirror the four PAD quadrants from narrate_state_to_user
// (``energized``/``settled``/``restless``/``subdued``) and drive the
// ::before dot colour in becca-presence.css.

let _moodOverlay = null;          // {label, body, tag, expiresAt}
let _moodTimer = null;
const _MOOD_DEFAULT_TTL_MS = 10 * 60 * 1000;  // 10 min

export function setBeccaMood(label, body = '', ttlMs = _MOOD_DEFAULT_TTL_MS, tag = '') {
  if (!label) return;
  _moodOverlay = {
    label: String(label),
    body: body ? String(body) : '',
    tag: tag ? String(tag).toLowerCase() : '',
    expiresAt: Date.now() + Math.max(1000, Number(ttlMs) || _MOOD_DEFAULT_TTL_MS),
  };
  if (_moodTimer) { clearTimeout(_moodTimer); _moodTimer = null; }
  _moodTimer = setTimeout(() => {
    _moodOverlay = null;
    _moodTimer = null;
    _renderStatus();
  }, _moodOverlay.expiresAt - Date.now());
  _renderStatus();
}

export function clearBeccaMood() {
  if (_moodTimer) { clearTimeout(_moodTimer); _moodTimer = null; }
  _moodOverlay = null;
  _renderStatus();
}

if (typeof window !== 'undefined') {
  // Expose globally so notifications.js can route without a circular
  // import (notifications boots before this widget mounts in some
  // layouts; the global is a no-op until the widget owns the slot).
  window.beccaShowMood = setBeccaMood;
  window.beccaClearMood = clearBeccaMood;
}

/**
 * Flash a wake-word indicator near the top of the widget.
 * Called externally (later from the wake-word detector slice) via
 * ``window.__beccaFlashWake(phrase)``.
 */
function _flashWake(phrase) {
  if (!_root || !_wakeFlash) return;
  _wakeFlash.textContent = _truncateGraceful(phrase, 32);
  _root.dataset.wakeFlash = 'true';
  if (_wakeFlashTimer) clearTimeout(_wakeFlashTimer);
  _wakeFlashTimer = setTimeout(() => {
    if (_root) _root.dataset.wakeFlash = 'false';
  }, 1600);
}

// ── Reflexive dance loop ──────────────────────────────────────────
//
// Three layers cooperate here:
//
//   1. The atlas pool (25 dances across VRMA + BVH + BOOTH) selected
//      by MovementConductor. The conductor applies role+emotion
//      scoring, recency tracking, cooldowns, and energy budget — we
//      just hand it the intent.
//
//   2. Adaptive duration. ``avatarState.vrmaCurrentDuration`` carries
//      the real clip length the moment playVrma resolves. Rotations
//      schedule against actual duration, not the atlas's informational
//      ``duration`` field. Short clips get a minimum slot so micro-
//      dances don't feel like a slideshow; long ones get their full
//      run.
//
//   3. Mood-adaptive selection. ``_currentMood()`` translates
//      PresenceEngine state (presence / flow / temperature / resonance)
//      to the atlas's emotion vector (warmth / energy / openness /
//      focus). Each new dance reads a fresh mood — so as her internal
//      state drifts with the conversation / media context, her dance
//      picks drift with her. Between dances we hold ~1.8s in a pose
//      that fits the same mood — gives the rotation a heartbeat
//      instead of a jump cut.
//
// Speech-tier audio (TTS, voice call) preempts dance entirely — the
// TTS listener calls _stopDanceLoop so lipsync can take over the body.
// Resumes after speech clears if media tier is still active.

const _DANCE_MIN_SLOT_SEC = 18.0;    // floor for "one dance slot" — gives
                                     // procedural breath/sway room between
                                     // clips so the rhythm isn't a slideshow
const _DANCE_SLOT_PADDING_SEC = 3.5; // tail after clip end before next; the
                                     // procedural animator settles back to
                                     // neutral during this window
const _DANCE_RETRY_MS = 6000;        // when conductor suppresses (budget)

function _clamp01(v) { return Math.max(0, Math.min(1, v)); }

/**
 * Translate PresenceEngine state into the atlas's emotion vector.
 * Returns a fresh object every call so callers can stash a snapshot.
 *
 * Mapping rationale:
 *   warmth  = baseline + resonance (sync feels warm) + temperature
 *   energy  = baseline + temperature (intensity) + |flow|
 *   openness = baseline + presence + resonance
 *   focus   = baseline + |flow| (focused on something) + temperature
 */
function _currentMood() {
  const p = avatarState?.presence;
  if (!p) {
    return { warmth: 0.6, energy: 0.7, openness: 0.65, focus: 0.6 };
  }
  return {
    warmth:   _clamp01(0.35 + (p.resonance   ?? 0) * 0.35 + (p.temperature ?? 0.2) * 0.2),
    energy:   _clamp01(0.3  + (p.temperature ?? 0.2) * 0.55 + Math.abs(p.flow ?? 0) * 0.15),
    openness: _clamp01(0.3  + (p.presence    ?? 0.3) * 0.4  + (p.resonance ?? 0)   * 0.3),
    focus:    _clamp01(0.4  + Math.abs(p.flow ?? 0) * 0.3   + (p.temperature ?? 0.2) * 0.2),
  };
}

function _scheduleDanceRotate(delayMs) {
  if (_danceRotateTimer) clearTimeout(_danceRotateTimer);
  _danceRotateTimer = setTimeout(() => {
    if (!_danceActive) return;
    // No "landing pose" call between dances — playVrma's natural
    // hold-then-release + the procedural animator's breath/sway IS
    // the gap. Trying to schedule a specific pose here turned into
    // her sitting down between dances because her mood was reading
    // as low-energy when nothing's been feeding presence. The fresh
    // mood read happens inside _playNextDance for the next pick.
    _playNextDance();
  }, delayMs);
}

async function _playNextDance() {
  if (!_root || !_vrmActive) return;
  // Don't fight the call avatar — only dance when she's in the widget.
  if (!avatarState._standalone) return;

  const mood = _currentMood();
  let anim = null;
  try {
    anim = await movementConductor.play({
      roles: ['dance', 'show-off'],
      emotion: mood,
    }, {
      // The rotation below owns this play's lifecycle (slot timing
      // includes user "longer" bonuses) — disable the conductor's
      // stuck-state watchdog so it can't cut a long slot mid-dance.
      safetyCapMs: 0,
    });
  } catch (e) {
    console.warn('[becca] dance request failed', e);
  }
  if (!_danceActive) return;  // toggled off during await

  if (!anim) {
    // Smart retry: when the conductor's `lastResult.reason` is
    // 'budget-floor', we know exactly when budget refills — schedule
    // for that moment instead of polling on a fixed 6s timer (which
    // is too long when budget is almost ready, too short when deeply
    // depleted). For 'no-fit' (everything on cooldown) keep the fixed
    // _DANCE_RETRY_MS — predicting cooldown expiry across the pool
    // would be more code than it's worth at this scale.
    const reason = movementConductor.lastResult?.reason;
    let waitMs = _DANCE_RETRY_MS;
    if (reason === 'budget-floor'
        && typeof movementConductor.estimatedBudgetRefillMs === 'function') {
      const refill = movementConductor.estimatedBudgetRefillMs(0.55);
      // Clamp [2s, 15s] — never less than 2s (gives the rest of the
      // tick loop room) or more than 15s (something else may unblock
      // before then; better to re-evaluate).
      waitMs = Math.max(2000, Math.min(15000, refill || _DANCE_RETRY_MS));
    }
    console.debug('[becca] dance suppressed by conductor',
      { reason, waitMs, budget: movementConductor.energyBudget?.toFixed(2) });
    _scheduleDanceRotate(waitMs);
    return;
  }

  // Real clip duration trumps atlas guess. playVrma just published it.
  const actualDur = avatarState.vrmaCurrentDuration > 0
    ? avatarState.vrmaCurrentDuration
    : (anim.duration || 10.0);
  // User "longer" ratings add seconds to the slot for this specific id.
  const slotBonus = movementConductor.getSlotBonus?.(anim.id) || 0;
  const slotSec = Math.max(actualDur + _DANCE_SLOT_PADDING_SEC, _DANCE_MIN_SLOT_SEC)
                + slotBonus;
  console.info('[becca] dance picked', anim.id,
    `actual=${actualDur.toFixed(1)}s slot=${slotSec.toFixed(1)}s bonus=${slotBonus}s`,
    'mood=', mood);

  // Log the playback so the timeline panel can show it for rating.
  _appendDanceHistory({
    animId: anim.id,
    label: _danceLabel(anim),
    playedAt: Date.now(),
    durationSec: actualDur,
  });

  _scheduleDanceRotate(slotSec * 1000);
}

function _danceLabel(anim) {
  // Friendly label: strip routing prefixes, replace dashes with spaces.
  const id = String(anim?.id || '');
  return id
    .replace(/^bvh-dance-/, '')
    .replace(/^dance-/, '')
    .replace(/-/g, ' ')
    .replace(/_/g, ' ');
}

function _appendDanceHistory(entry) {
  _danceHistory.unshift(entry);
  if (_danceHistory.length > DANCE_HISTORY_CAP) {
    _danceHistory.length = DANCE_HISTORY_CAP;
  }
  _uSet(DANCE_HISTORY_STORAGE_KEY, JSON.stringify(_danceHistory));
  if (_timelinePanel && !_timelinePanel.hidden) _renderTimeline();
  // Fire-and-forget server append. Failure leaves the entry only in
  // the local cache; a future _refreshDanceHistoryFromServer() will
  // reconcile by taking the server's view as authoritative.
  fetch('/api/dance/history', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anim_id: entry.animId,
      label: entry.label,
      played_at: entry.playedAt,
      duration_sec: entry.durationSec,
      mode: entry.mode || null,
    }),
  }).catch(() => { /* offline / unauthed — cache stays */ });
}

function _loadDanceHistory() {
  try {
    const raw = _uGet(DANCE_HISTORY_STORAGE_KEY);
    if (raw) _danceHistory = JSON.parse(raw) || [];
  } catch (_) { _danceHistory = []; }
}

// Server-authoritative refresh. Fetches the per-user history and
// replaces the in-memory list. Called once at widget mount alongside
// the ratings refresh. Silent on failure so unauthed/offline use
// keeps the localStorage view.
async function _refreshDanceHistoryFromServer() {
  try {
    const resp = await fetch(
      `/api/dance/history?limit=${DANCE_HISTORY_CAP}`,
      { credentials: 'same-origin' },
    );
    if (!resp.ok) return;
    const data = await resp.json();
    const entries = data?.entries || [];
    // Normalize snake_case server shape → camelCase JS shape used
    // throughout the widget (animId, playedAt, durationSec).
    _danceHistory = entries.map(e => ({
      animId: e.anim_id,
      label: e.label,
      playedAt: e.played_at,
      durationSec: e.duration_sec,
      mode: e.mode || null,
    }));
    _uSet(DANCE_HISTORY_STORAGE_KEY, JSON.stringify(_danceHistory));
    if (_timelinePanel && !_timelinePanel.hidden) _renderTimeline();
  } catch (_) { /* keep cache */ }
}

// ── User-uploaded animations ───────────────────────────────────────
//
// Fetch the user's uploads from /api/animations/list and register them
// onto the atlas registry. Called at mount + after every successful
// upload/delete so the conductor sees the latest pool.
async function _refreshUserAnimations() {
  try {
    const [listResp, ovResp] = await Promise.all([
      fetch('/api/animations/list', { credentials: 'same-origin' }),
      fetch('/api/animations/overrides', { credentials: 'same-origin' }),
    ]);
    if (listResp.ok) {
      const data = await listResp.json();
      registerUserAnimations(data?.animations || []);
    }
    if (ovResp.ok) {
      const ovData = await ovResp.json();
      registerAtlasOverrides(ovData?.overrides || []);
    }
    if (_timelinePanel && !_timelinePanel.hidden) _renderTimeline();
    // Registry changed shape — push the merged role vocabulary so
    // server-side consumers (gesture verb tool schemas) see the live
    // verbs including user-defined roles. Fire-and-forget.
    _pushRolesSnapshot();
  } catch (_) { /* unauthed / offline — keep current registry */ }
}

// Last vocabulary actually accepted by the server — push only deltas
// so mount + every upload/edit doesn't spam identical PUTs.
let _lastRolesPushed = '';

async function _pushRolesSnapshot() {
  try {
    const roles = listRoles();
    const key = JSON.stringify(roles);
    if (key === _lastRolesPushed) return;
    const resp = await fetch('/api/animations/roles-snapshot', {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ roles }),
    });
    if (resp.ok) _lastRolesPushed = key;
  } catch (_) { /* offline — retried on next registry refresh */ }
}

// Trigger the hidden file input and POST the chosen file. Default
// metadata is a sensible all-purpose dance clip; the user can edit
// tags later via the inline manage list (Phase B v1 keeps the upload
// dialog minimal — just file + auto-derived label).
async function _uploadAnimationFile(file) {
  if (!file) return null;
  const ext = (file.name || '').toLowerCase().slice(-5);
  if (!ext.endsWith('.vrma') && !ext.endsWith('.bvh')) {
    alert('Animation must be .vrma or .bvh');
    return null;
  }
  const fd = new FormData();
  fd.append('file', file);
  fd.append('metadata', JSON.stringify({
    // Sensible defaults — user can rename/retag from the upload row.
    roles: ['dance', 'idle-fill'],
    modes: ['chat-call', 'narrative'],
    cost: 0.5,
    cooldown_sec: 300,
    loop_flag: true,
  }));
  try {
    const resp = await fetch('/api/animations/upload', {
      method: 'POST',
      credentials: 'same-origin',
      body: fd,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err?.error || 'Upload failed');
      return null;
    }
    const data = await resp.json();
    await _refreshUserAnimations();
    return data?.animation || null;
  } catch (e) {
    alert('Upload failed (network)');
    return null;
  }
}

// Delete a user-uploaded animation. Confirm to avoid accidental
// data loss — uploads are one-by-one work for the user, so the
// confirmation cost is dwarfed by the un-recovery cost.
async function _deleteUserAnimation(animationId, label) {
  if (!animationId?.startsWith('user:')) return;
  if (!confirm(`Delete "${label || animationId}"?`)) return;
  try {
    const resp = await fetch(
      `/api/animations/${encodeURIComponent(animationId)}`,
      { method: 'DELETE', credentials: 'same-origin' },
    );
    if (!resp.ok) {
      alert('Delete failed');
      return;
    }
    await _refreshUserAnimations();
  } catch (_) {
    alert('Delete failed (network)');
  }
}

// ── Loops (Phase C) ────────────────────────────────────────────────
//
// A curated subset of animation ids the conductor draws from when
// active. Loops mix bundled and user-uploaded ids freely. State is
// server-authoritative — the widget fetches the active loop on mount
// and applies it to the conductor.

let _loopsCache = [];        // mirror of /api/dance/loops "loops" field
let _activeLoopId = null;     // currently active loop id (or null)
let _loopsOverlayEl = null;   // open-loops overlay DOM (null = closed)

async function _refreshLoops() {
  try {
    const resp = await fetch('/api/dance/loops', {
      credentials: 'same-origin',
    });
    if (!resp.ok) return;
    const data = await resp.json();
    _loopsCache = data?.loops || [];
    _activeLoopId = data?.active_id || null;
    _applyActiveLoopToConductor();
  } catch (_) { /* offline / unauthed */ }
}

function _applyActiveLoopToConductor() {
  if (!_activeLoopId) {
    try { movementConductor.setActiveLoop?.(null); } catch (_) {}
    return;
  }
  const loop = _loopsCache.find(l => l.id === _activeLoopId);
  if (loop && Array.isArray(loop.animation_ids)) {
    try {
      movementConductor.setActiveLoop?.(loop.animation_ids);
    } catch (_) {}
  }
}

async function _activateLoop(loopId) {
  try {
    const resp = await fetch('/api/dance/loops/active', {
      method: 'PUT',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ loop_id: loopId }),
    });
    if (!resp.ok) return;
    await _refreshLoops();
  } catch (_) { /* swallow */ }
}

async function _deleteLoop(loopId, name) {
  if (!confirm(`Delete loop "${name || loopId}"?`)) return;
  try {
    const resp = await fetch(
      `/api/dance/loops/${encodeURIComponent(loopId)}`,
      { method: 'DELETE', credentials: 'same-origin' },
    );
    if (!resp.ok) return;
    await _refreshLoops();
  } catch (_) { /* swallow */ }
}

// Create a loop from whatever's currently in dance history. Uses the
// distinct anim ids in the buffer as the loop membership. Naming
// auto-derives from the most-recent label so the user can rename
// later via Phase D.
async function _createLoopFromHistory() {
  if (!_danceHistory.length) {
    alert('No recent dances to build a loop from.');
    return;
  }
  const ids = [...new Set(_danceHistory.map(e => e.animId))]
    .filter(Boolean);
  if (!ids.length) return;
  const seed = (_danceHistory[0]?.label || 'untitled').slice(0, 30);
  const name = prompt('Loop name', `${seed} mix`);
  if (!name?.trim()) return;
  try {
    const resp = await fetch('/api/dance/loops', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name.trim(),
        animation_ids: ids,
        notes: 'created from recent history',
      }),
    });
    if (!resp.ok) {
      alert('Create failed');
      return;
    }
    await _refreshLoops();
  } catch (_) {
    alert('Create failed (network)');
  }
}

function _startDanceLoop() {
  if (_danceActive) return;
  _danceActive = true;
  // Unmute the conductor while hosting. Default quiet mode keeps her
  // still (just procedural breath/sway + look-at-user) so VRMAs only
  // fire when we explicitly want them — currently host + music. The
  // active-loop constraint (the user's preferred host loop) is set
  // independently via setActiveLoop().
  try { movementConductor.setQuietMode(false); } catch (_) {}
  _playNextDance();
}

function _stopDanceLoop() {
  if (!_danceActive) return;
  _danceActive = false;
  if (_danceRotateTimer) { clearTimeout(_danceRotateTimer); _danceRotateTimer = null; }
  try { movementConductor.stop(); } catch (_) {}
  // Back to quiet — pose-trigger and animation-router auto-fires
  // (e.g. "user started speaking", "voice.tool_call") will no-op
  // until either host re-engages or an explicit click fires.
  try { movementConductor.setQuietMode(true); } catch (_) {}
}

function _maybeUpdateDanceLoop(tiers, kinds) {
  // Dance only when host mode is on AND the active audio is MUSIC.
  // Audiobookshelf narration, video dialogue, podcasts, etc. all play
  // on the same media tier as music — without the kind tag we'd dance
  // to all of them. Tagging-aware: now we branch on content type.
  //
  // Speech (TTS, voice call) always preempts via the TTS listener
  // (which calls _stopDanceLoop directly + lipsyncs).
  const hasMusic = kinds && kinds.includes('music');
  const hasSpeech = tiers && tiers.includes('speech');
  const shouldDance = _audioRole === 'host' && hasMusic && !hasSpeech;
  console.debug('[becca] dance-eval',
    { role: _audioRole, tiers, kinds, shouldDance, currentlyDancing: _danceActive });
  if (shouldDance && !_danceActive) _startDanceLoop();
  else if (!shouldDance && _danceActive) _stopDanceLoop();
}

// ── AudioBus state subscription ────────────────────────────────────
//
// audio-bus.js emits 'augmentum:audio-bus-state' on every tier change.
// We use it to surface audio activity on the status row and to drive
// the dance loop. Detail carries the highest-tier label + the list of
// active tiers.

function _attachAudioBusListener() {
  if (_audioBusEventHandler) return;
  _audioBusEventHandler = (e) => {
    if (!_root) return;
    const detail = e.detail || {};
    const tiers = detail.activeTiers || [];
    const kinds = detail.activeKinds || [];
    // Speech-tier audio is handled by the TTS listener (sets analyser +
    // speaking state). We surface non-speech sources here, with the
    // status label reflecting CONTENT, not just "audio is playing".
    const nonSpeechKinds = kinds.filter(k => k !== 'speech' && k !== 'sfx');
    if (nonSpeechKinds.length) {
      _setStatusOverride(_statusLabelForKind(nonSpeechKinds));
    } else {
      _setStatusOverride(null);
    }
    try {
      _ensureAnimationRouter()?.onAudioBusState?.(detail, { audioRole: _audioRole });
    } catch (err) {
      console.warn('[becca] animation router audio-state failed', err);
    }
    _maybeUpdateDanceLoop(tiers, kinds);
  };
  window.addEventListener('augmentum:audio-bus-state', _audioBusEventHandler);
}

function _statusLabelForKind(kinds) {
  // Pick the most informative label. Priority: music → narration →
  // mixed → dialogue → ambient → unknown. Each maps to a short
  // present-tense phrase fit for the status row.
  if (kinds.includes('music'))     return 'hosting music';
  if (kinds.includes('narration')) return 'listening · audiobook';
  if (kinds.includes('dialogue'))  return 'listening · dialogue';
  if (kinds.includes('mixed'))     return 'listening · video';
  if (kinds.includes('ambient'))   return 'ambient';
  return 'hosting audio';
}

function _detachAudioBusListener() {
  if (!_audioBusEventHandler) return;
  window.removeEventListener('augmentum:audio-bus-state', _audioBusEventHandler);
  _audioBusEventHandler = null;
}

// ── TTS event subscription ────────────────────────────────────────
//
// chat/tts.js emits 'augmentum:tts-playback' with
// { active, analyser, text, source }.
// When TTS starts: bind the analyser to the avatar pipeline so her
// mouth + presence-state move with the speech. When it ends: unbind.
//
// Works in both audio-roles — Phase 1 ships universal lipsync; Phase
// 3 (audio classifier) will branch on content kind.

const TTS_PROGRESSIVE_END_HOLD_MS = 180;

// Monotonic token incremented on every playback-start event. End
// events capture the token at schedule time; the deferred teardown
// aborts if a newer start arrived in between. Without this, a stale
// end event (an aborted/superseded clip's cleanup firing after the
// next clip already started) nulled the analyser mid-utterance and
// the mouth stayed closed for the whole sentence.
let _ttsTurnToken = 0;

function _clearTtsEndHold() {
  if (_ttsEndHoldTimer) {
    clearTimeout(_ttsEndHoldTimer);
    _ttsEndHoldTimer = null;
  }
}

function _isProgressiveTtsSessionActive() {
  try {
    const snap = window.AudioBus?.debug?.();
    return Array.isArray(snap?.active) && snap.active.includes('chat-tts-progressive');
  } catch (_) {
    return false;
  }
}

function _finishTtsPlaybackTurn() {
  _ttsEndHoldTimer = null;
  avatarState.analyserNode = null;
  try { avatarState.lipSync?.setDryPulse?.(false); } catch (_) {}
  try { onTtsPlaybackChange(false); } catch (_) {}
  try { _ensureAnimationRouter()?.onTtsEnd?.(); } catch (err) {
    console.warn('[becca] animation router tts-playback end failed', err);
  }
  // Fall back: if audio bus override is still set (e.g., music playing),
  // _setStatusOverride takes precedence and the speaking state hides.
  _setStatusState(_audioRole === 'host' ? 'hosting' : 'idle');
}

function _scheduleTtsPlaybackTurnEnd({ progressive = false } = {}) {
  _clearTtsEndHold();
  // The hold applies to EVERY end now (it was progressive-only).
  // Two reasons: (1) auto-read queues per-sentence clips — an
  // immediate teardown between clips hard-reset the lipsync every
  // sentence boundary (mouth snap) and re-bound 10ms later; (2) the
  // token guard below needs a deferral window to observe an
  // out-of-order start. 180ms is short enough that a real turn end
  // still closes the mouth promptly (the engine's own silence
  // collapse handles the visual).
  const token = _ttsTurnToken;
  _ttsEndHoldTimer = setTimeout(() => {
    if (token !== _ttsTurnToken) {
      // A newer clip started after this end was scheduled — its
      // lifecycle owns the speaking state now. Do not tear down.
      _ttsEndHoldTimer = null;
      return;
    }
    if (progressive && _isProgressiveTtsSessionActive()) {
      _scheduleTtsPlaybackTurnEnd({ progressive: true });
      return;
    }
    _finishTtsPlaybackTurn();
  }, TTS_PROGRESSIVE_END_HOLD_MS);
}

function _attachTtsListener() {
  if (_ttsEventHandler) return;
  _ttsEventHandler = (e) => {
    const { active, analyser, analyserDry, text, source } = e.detail || {};
    console.info('[becca] tts-playback event',
      { active, hasAnalyser: !!analyser, dry: !!analyserDry,
        vrmActive: _vrmActive,
        standalone: avatarState._standalone, rooted: !!_root });
    if (!_root || !_vrmActive) return;
    // Skip if we've been reparented out (call mode owns its own analyser).
    if (!avatarState._standalone) return;
    if (active) {
      _ttsTurnToken += 1;
      _clearTtsEndHold();
      if (analyser) avatarState.analyserNode = analyser;
      // Dry clip — audio bypasses the analyser (iOS native playback /
      // failed bind). Flip the lipsync engine to its synthetic
      // dry-pulse so the mouth still moves with the speech instead of
      // staying shut on a silent analyser.
      try { avatarState.lipSync?.setDryPulse?.(!!analyserDry); } catch (_) {}
      try { onTtsPlaybackChange(!!analyser || !!analyserDry); } catch (_) {}
      try { _ensureAnimationRouter()?.onTtsStart?.(text || ''); } catch (err) {
        console.warn('[becca] animation router tts-playback start failed', err);
      }
      _setStatusState('speaking');
      // Speech preempts dance — VRMA would override her body and break
      // lipsync. Stop the loop; AudioBus listener will resume it when
      // speech tier clears (if host + media still active).
      _stopDanceLoop();
    } else {
      _scheduleTtsPlaybackTurnEnd({ progressive: String(source || '') === 'progressive' });
    }
  };
  window.addEventListener('augmentum:tts-playback', _ttsEventHandler);
}

function _detachTtsListener() {
  if (!_ttsEventHandler) return;
  window.removeEventListener('augmentum:tts-playback', _ttsEventHandler);
  _ttsEventHandler = null;
  _clearTtsEndHold();
}

/**
 * Lightweight visual placeholder shown until the unified VRM stage
 * is live (and on any activation failure). A round monogram with a
 * soft radial gradient — reads as "someone in the corner" without
 * claiming to be the live avatar.
 */
function _buildPlaceholder() {
  const wrap = document.createElement('div');
  wrap.className = 'becca-presence__placeholder';
  wrap.innerHTML = `
    <div class="becca-placeholder-disc" aria-hidden="true">
      <span>b</span>
    </div>
    <div class="becca-placeholder-name">becca</div>
  `;
  return wrap;
}

/**
 * Unmount the widget, returning the canvas to its original viewport.
 */
export function unmountBeccaPresence() {
  if (!_root) return;
  try { delete window.__beccaReactivateVRM; } catch (_) { window.__beccaReactivateVRM = null; }
  try { delete window.__beccaHideForCall; } catch (_) { window.__beccaHideForCall = null; }
  try { delete window.__beccaFlashWake; } catch (_) { window.__beccaFlashWake = null; }
  _detachTtsListener();
  _detachAudioBusListener();
  _detachCursorGaze();
  _detachStagePassthrough();
  _detachVisibilityPause();
  _stopLoadProbe();
  // Slice 1 — chrome-idle timer, verb-tick listener, long-press
  // timers + window-level prefs listener. The DOM elements
  // (transcript chip, mini-controls drawer) come down with _root.
  _detachChromeIdle();
  _detachVerbTickListener();
  _detachLoadableListener();
  _detachLongPressHandle();
  if (_heardYouTimer) { clearTimeout(_heardYouTimer); _heardYouTimer = null; }
  if (_transcriptClearTimer) { clearTimeout(_transcriptClearTimer); _transcriptClearTimer = null; }
  // Status-row / pose one-shots — these reschedule on every activity-chosen,
  // pose-shift, and near-miss event, so they're not registered on the
  // lifetime (addTimeout would accumulate a teardown per reschedule); cancel
  // any pending one here, matching the clearTimeout idiom used above.
  if (_activityClearTimer) { clearTimeout(_activityClearTimer); _activityClearTimer = null; }
  if (_poseShiftTimer) { clearTimeout(_poseShiftTimer); _poseShiftTimer = null; }
  if (_nearMissTimer) { clearTimeout(_nearMissTimer); _nearMissTimer = null; }
  _stopDanceLoop();
  _stopAffectPoll();
  _setTimelineOpen(false);
  _closeAvatarPicker();
  if (_animationRouter) { try { _animationRouter.dispose(); } catch (_) {} _animationRouter = null; }
  _stopCameraView();   // release the camera before the PTT session goes
  if (_pttSession) { try { _pttSession.dispose(); } catch (_) {} _pttSession = null; }
  _pttBtnEl = null;
  _eyeBtn = null;
  if (_wakeSession) { try { _wakeSession.dispose(); } catch (_) {} _wakeSession = null; }
  _wakePausedForCall = false;
  if (_wakeResumeTimer) { clearTimeout(_wakeResumeTimer); _wakeResumeTimer = null; }
  // Always-listening cleanup
  if (_alwaysListeningRearmTimer) {
    clearTimeout(_alwaysListeningRearmTimer);
    _alwaysListeningRearmTimer = null;
  }
  _clearAlwaysListeningWatchdog();
  _alwaysListeningArmed = false;
  if (_alwaysListeningVisibilityHandler) {
    try { document.removeEventListener('visibilitychange', _alwaysListeningVisibilityHandler); } catch (_) {}
    _alwaysListeningVisibilityHandler = null;
  }
  try { delete window.__beccaAlwaysListeningState; } catch (_) { window.__beccaAlwaysListeningState = null; }
  if (_wakePrefsHandler) {
    try { window.removeEventListener('becca:wake-prefs-changed', _wakePrefsHandler); } catch (_) {}
    _wakePrefsHandler = null;
  }
  if (_heartbeatTimer) { clearInterval(_heartbeatTimer); _heartbeatTimer = null; }
  _heartbeatBadTicks = 0;
  if (_wakeFlashTimer) { clearTimeout(_wakeFlashTimer); _wakeFlashTimer = null; }
  if (_timelineCloseTimer) { clearTimeout(_timelineCloseTimer); _timelineCloseTimer = null; }
  if (_channelClearTimer) { clearTimeout(_channelClearTimer); _channelClearTimer = null; }
  if (_viewportClampTimer) { clearTimeout(_viewportClampTimer); _viewportClampTimer = null; }
  if (_viewportClampHandler) {
    try { window.removeEventListener('resize', _viewportClampHandler); } catch (_) {}
    _viewportClampHandler = null;
  }
  if (_returnFlashTimer) { clearTimeout(_returnFlashTimer); _returnFlashTimer = null; }
  // Drop every lifetime-tracked listener / observer / timer in one
  // call. Handlers registered through ``_lifetime.addEventListener``
  // (drag/resize/keyboard/surface/activation-mode/voice-pill observer)
  // are detached here in LIFO order.
  if (_lifetime) {
    try { _lifetime.dispose(); } catch (_) { /* per-step errors already logged */ }
    _lifetime = null;
  }
  _swapInFlight = false;
  if (_busReconnector) {
    // Stop first so the onclose below doesn't schedule a reconnect into
    // a torn-down widget.
    try { _busReconnector.stop(); } catch (_) {}
    _busReconnector = null;
  }
  if (_wsBus) {
    try { _wsBus.close(); } catch (_) {}
    _wsBus = null;
  }
  if (_vrmActive) {
    try { deactivateAvatar(); } catch (_) {}
    _vrmActive = false;
  }
  // Drop overlay reference before _stage goes null so the element
  // doesn't outlive its parent.
  _hideCatchingUpOverlay();
  // The studio-docked animations panel mounts on document.body, not
  // _root — close explicitly or it outlives the widget.
  _closeLoopsOverlay();
  _root.remove();
  _root = null;
  _stage = null;
  _statusRow = null;
  _wakeFlash = null;
  _returnFlash = null;
  _audioModeBtn = null;
  _micDspBtn = null;
  _timelinePanel = null;
  _timelineHandle = null;
  _transcriptChip = null;
  _contextChip = null;
  _loadableBusy = false;
  _miniControlsEl = null;
}

/**
 * Toggle discreet mode (Lane 4 §11). Cmd/Ctrl+Shift+. shortcut, or
 * settings toggle, or long-press on touch.
 */
export function toggleDiscreetMode() {
  if (!_root) return;
  const isDiscreet = _root.dataset.surface === 'discreet';
  _applySurface(isDiscreet ? _detectSurface() : 'discreet');
  _showDiscreetFlash(!isDiscreet);
}

// ── Real-talk panel ───────────────────────────────────────────────
//
// Crisis-resources dialog. Invoked deliberately by external callers
// (companion runtime, safety floor); there is intentionally no
// gesture / pill / hotkey for opening it from the widget surface —
// see [[user-realtalk-affordance]] in memory + the mount-time comment.

function _openRealTalkPanel() {
  if (document.querySelector('.becca-realtalk-panel')) return;  // already open

  // Remember the element that had focus so we can return focus on close
  // — standard dialog hygiene. If nothing was focused, fall back to the
  // widget root.
  const previouslyFocused = document.activeElement instanceof HTMLElement
    ? document.activeElement
    : null;

  const panel = document.createElement('div');
  panel.className = 'becca-realtalk-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  panel.setAttribute('aria-labelledby', 'becca-realtalk-title');
  panel.tabIndex = -1;
  panel.innerHTML = `
    <h3 id="becca-realtalk-title">real talk</h3>
    <p>if you're in a hard place right now, here are people who answer the phone. they're better at this than i am.</p>
    <div class="becca-resource-card">
      <strong>988</strong>
      <small>call or text — US Suicide &amp; Crisis Lifeline · 24/7</small>
    </div>
    <div class="becca-resource-card">
      <strong>741741</strong>
      <small>text HOME — Crisis Text Line · 24/7</small>
    </div>
    <p>i'll be here when you come back. take the time you need.</p>
    <button type="button" class="becca-close">close</button>
  `;

  const close = () => {
    panel.remove();
    document.removeEventListener('keydown', keyHandler);
    // Return focus to wherever it came from (or fall back to the widget
    // root if that target is gone). Without this, screen-reader and
    // keyboard users land on document.body after the dialog closes.
    try {
      if (previouslyFocused && document.contains(previouslyFocused)) {
        previouslyFocused.focus();
      } else if (_root) {
        _root.focus();
      }
    } catch (_) { /* focus can throw on disabled / detached nodes */ }
    // After close, Becca's pose flips to interior_looking_aside for ~12s.
    // The pose orchestrator (xr-companion-binding) listens for
    // realtalk.closed and handles the transition.
    if (window.__beccaBus) {
      try {
        window.__beccaBus.dispatchEvent(new CustomEvent('realtalk.closed'));
      } catch (_) { /* listener throws are non-fatal — bus is best-effort */ }
    }
  };

  // Combined Esc + Tab trap. Tab inside a dialog should cycle through
  // the dialog's focusables, not escape to the document behind it —
  // otherwise a keyboard user can lose track of where they are.
  const keyHandler = (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return;
    }
    if (e.key !== 'Tab') return;
    const focusables = panel.querySelectorAll(
      'button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
    );
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  panel.querySelector('.becca-close').addEventListener('click', close);
  panel.addEventListener('click', (e) => { if (e.target === panel) close(); });
  document.addEventListener('keydown', keyHandler);

  document.body.appendChild(panel);
  // Initial focus on the close button — least surprising target for
  // keyboard users in a small dialog with no form controls. Defer to
  // the next microtask so the browser has finished inserting the node
  // before we attempt to focus it.
  Promise.resolve().then(() => {
    try { panel.querySelector('.becca-close')?.focus(); } catch (_) { /* swallow */ }
  });

  // Fire bus event so Becca's pose pauses (interior_looking_aside).
  if (window.__beccaBus) {
    try {
      window.__beccaBus.dispatchEvent(new CustomEvent('realtalk.opened'));
    } catch (_) { /* listener throws are non-fatal — bus is best-effort */ }
  }

  // Silent telemetry — anonymized; per Lane 4 §6.5 the user does NOT
  // see "you've opened this N times" anywhere.
  try {
    fetch('/api/companion/safety_floor_audit_event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'realtalk_panel_opened' }),
    }).catch(() => {});
  } catch (_) {}
}

// ── Drag-to-reposition ────────────────────────────────────────────

// ── Stage click-through ───────────────────────────────────────────
//
// The root frame is pointer-events:none (see CSS) so the empty
// rectangle around her body doesn't intercept clicks meant for the
// page underneath — especially noticeable when the user has her
// resized large. Chrome (dock, dismiss, resize handles, timeline)
// individually opts back into pointer events.
//
// The stage canvas defaults to 'auto'. A throttled pointermove
// listener raycasts the cursor into the active VRM scene; when the
// cursor is over empty scene pixels the stage flips to 'none' so the
// click falls through to whatever's behind. When the cursor crosses
// onto her body the stage flips back to 'auto' and normal drag/click
// resumes. Net result: she is the only solid hit-target inside the
// widget bounds, with no canvas-level pixel readback (no need for
// preserveDrawingBuffer in the shared avatar.js renderer).
//
// Fail-open: any path that can't make a confident decision (VRM not
// loaded, raycast throws mid-swap, drag/resize in progress, cursor
// outside the box) leaves the stage as 'auto' so the user never gets
// stuck unable to grab her.

function _attachStagePassthrough() {
  if (!_root || !_stage || _passthroughHandler) return;
  _passthroughHandler = (e) => {
    _passthroughLastEvt = e;
    if (_passthroughRafScheduled) return;
    _passthroughRafScheduled = true;
    requestAnimationFrame(() => {
      _passthroughRafScheduled = false;
      _evaluateStagePassthrough(_passthroughLastEvt);
    });
  };
  // passive — we never call preventDefault; this is purely a hit-test
  // that mutates a style attribute.
  window.addEventListener('pointermove', _passthroughHandler, { passive: true });
}

function _detachStagePassthrough() {
  if (_passthroughHandler) {
    try { window.removeEventListener('pointermove', _passthroughHandler); } catch (_) {}
    _passthroughHandler = null;
  }
  _passthroughRafScheduled = false;
  _passthroughLastEvt = null;
}

function _evaluateStagePassthrough(e) {
  if (!_root || !_stage || !e) return;
  // Drag / resize in progress — keep the stage solid so pointermove
  // continues to flow to the root-level drag handler without flicker.
  if (_root.classList.contains('dragging') || _root.classList.contains('resizing')) {
    _stage.style.pointerEvents = 'auto';
    return;
  }
  const rect = _root.getBoundingClientRect();
  const inBox = e.clientX >= rect.left && e.clientX <= rect.right
             && e.clientY >= rect.top  && e.clientY <= rect.bottom;
  if (!inBox) {
    // Cursor isn't over the widget — leave stage 'auto' so a pointerdown
    // that lands inside before the next move tick still hits her.
    _stage.style.pointerEvents = 'auto';
    return;
  }
  // Placeholder showing (no VRM yet) — treat the whole stage as her so
  // the user can still drag the disc-with-name.
  if (!_vrmActive || !avatarState.vrm || !avatarState.camera) {
    _stage.style.pointerEvents = 'auto';
    return;
  }
  const vrmRoot = avatarState.vrm.scene || avatarState.vrm;
  if (!vrmRoot) {
    _stage.style.pointerEvents = 'auto';
    return;
  }
  if (!_passthroughRaycaster) {
    _passthroughRaycaster = new Raycaster();
    _passthroughNdc = new Vector2();
  }
  const sRect = _stage.getBoundingClientRect();
  if (sRect.width <= 0 || sRect.height <= 0) return;
  _passthroughNdc.x = ((e.clientX - sRect.left) / sRect.width) * 2 - 1;
  _passthroughNdc.y = -((e.clientY - sRect.top) / sRect.height) * 2 + 1;
  _passthroughRaycaster.setFromCamera(_passthroughNdc, avatarState.camera);
  let hits;
  try {
    hits = _passthroughRaycaster.intersectObject(vrmRoot, true);
  } catch (_) {
    // Skinned-mesh raycast can throw briefly during a VRM swap when the
    // bone hierarchy is being reattached. Fail open.
    _stage.style.pointerEvents = 'auto';
    return;
  }
  _stage.style.pointerEvents = (hits && hits.length > 0) ? 'auto' : 'none';
}

function _attachDragHandlers() {
  if (!_root || !_lifetime) return;
  let dragging = false;
  let offsetX = 0, offsetY = 0;
  let startX = 0, startY = 0;
  let moved = false;

  const onDown = (e) => {
    // Don't initiate drag on the status row, audio-mode button, wake-flash
    // overlay, timeline handle/panel, dismiss button, or HUD region —
    // they're not part of the body.
    if (e.target.closest && (
      e.target.closest('.becca-presence__status') ||
      e.target.closest('.becca-presence__audio-mode') ||
      e.target.closest('.becca-presence__ptt') ||
      e.target.closest('.becca-presence__wake-flash') ||
      e.target.closest('.becca-presence__timeline-handle') ||
      e.target.closest('.becca-presence__timeline') ||
      e.target.closest('.becca-presence__dismiss') ||
      e.target.closest('.becca-presence__resize') ||
      e.target.closest('.becca-presence__hud')
    )) return;
    dragging = true;
    moved = false;
    const rect = _root.getBoundingClientRect();
    startX = e.clientX;
    startY = e.clientY;
    offsetX = e.clientX - rect.left;
    offsetY = e.clientY - rect.top;
  };

  const onMove = (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (!moved && Math.hypot(dx, dy) > 6) {
      moved = true;
      _root.classList.add('dragging');
    }
    if (moved) {
      _root.style.left = `${e.clientX - offsetX}px`;
      _root.style.top = `${e.clientY - offsetY}px`;
      _root.style.right = 'auto';
      _root.style.bottom = 'auto';
    }
  };

  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    if (moved) {
      _root.classList.remove('dragging');
      _persistPosition();
    }
  };

  // Lifetime-tracked so unmount tears them down — the pre-2026-06-09
  // version registered these directly and leaked two window listeners
  // per dismiss→resummon cycle.
  _lifetime.addEventListener(_root, 'pointerdown', onDown);
  _lifetime.addEventListener(window, 'pointermove', onMove);
  _lifetime.addEventListener(window, 'pointerup', onUp);
}

function _persistPosition() {
  if (!_root) return;
  const surface = _activeSurface || 'private';
  const rect = _root.getBoundingClientRect();
  const key = STORAGE_KEY_PREFIX + surface;
  _uSet(key, JSON.stringify({
    left: rect.left, top: rect.top,
  }));
}

function _restorePosition() {
  if (!_root) return;
  const surface = _activeSurface || 'private';
  const key = STORAGE_KEY_PREFIX + surface;
  try {
    const raw = _uGet(key);
    if (raw) {
      const pos = JSON.parse(raw);
      if (typeof pos.left === 'number' && typeof pos.top === 'number') {
        _root.style.left = `${pos.left}px`;
        _root.style.top = `${pos.top}px`;
        _root.style.right = 'auto';
        _root.style.bottom = 'auto';
      }
    }
  } catch (_) { /* unreadable storage — clamp below restores sane default */ }
  // Always clamp — even with no saved position the default CSS anchor
  // is bottom/right which is fine on its own, but a saved position from
  // a wider window can land the widget off-screen. Clamp to viewport
  // with a small margin so the user can still grab it.
  _clampWidgetToViewport();
}

// Window-resize listener that re-clamps the widget into the viewport
// when the user resizes/snaps the browser window. Debounced so we don't
// run on every resize tick. The clamp itself is cheap but the
// getBoundingClientRect call triggers layout; debouncing keeps that
// out of the resize hot path on slower devices.
function _attachViewportClamp() {
  if (_viewportClampHandler) return;
  _viewportClampHandler = () => {
    if (_viewportClampTimer) clearTimeout(_viewportClampTimer);
    _viewportClampTimer = setTimeout(() => {
      _clampWidgetToViewport();
      _viewportClampTimer = null;
    }, 120);
  };
  window.addEventListener('resize', _viewportClampHandler);
}

const VIEWPORT_MARGIN = 12;
function _clampWidgetToViewport() {
  if (!_root) return;
  if (_cameraView) return;  // camera full-screen owns layout — don't clamp it
  // If the widget is still using the default bottom/right anchor (no
  // explicit left/top set), leave it alone — CSS handles it correctly
  // at any viewport size.
  const usesExplicitAnchor = _root.style.left && _root.style.left !== 'auto';
  if (!usesExplicitAnchor) return;
  const rect = _root.getBoundingClientRect();
  const vw = window.innerWidth || document.documentElement.clientWidth;
  const vh = window.innerHeight || document.documentElement.clientHeight;
  let left = rect.left;
  let top = rect.top;
  if (left + rect.width > vw - VIEWPORT_MARGIN) left = vw - rect.width - VIEWPORT_MARGIN;
  if (top + rect.height > vh - VIEWPORT_MARGIN) top = vh - rect.height - VIEWPORT_MARGIN;
  if (left < VIEWPORT_MARGIN) left = VIEWPORT_MARGIN;
  if (top < VIEWPORT_MARGIN) top = VIEWPORT_MARGIN;
  if (Math.round(left) !== Math.round(rect.left) || Math.round(top) !== Math.round(rect.top)) {
    _root.style.left = `${left}px`;
    _root.style.top = `${top}px`;
    _root.style.right = 'auto';
    _root.style.bottom = 'auto';
  }
}

// ── Resize handles ────────────────────────────────────────────────
//
// Four corner grab zones let the user stretch/shrink the widget.
// Persisted to a single localStorage key (size doesn't surface-vary
// the way position does — the user wants her the same size in chat
// and in a stream-share). Mobile media queries in CSS override the
// stored size. The VRM renderer auto-rescales via avatar.js's
// ResizeObserver on the stage host.

function _attachResizeHandlers() {
  if (!_root || !_lifetime) return;
  let resizing = false;
  let corner = null;
  let startX = 0, startY = 0;
  let startW = 0, startH = 0;
  let startLeft = 0, startTop = 0;

  // Resize handles are children of _root, so they're discarded when
  // _root.remove() runs in unmount — no detach needed on these. The
  // window-level move/up/cancel handlers below are NOT children and
  // must be lifetime-tracked.
  const handles = _root.querySelectorAll('.becca-presence__resize');
  handles.forEach(h => {
    h.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      e.preventDefault();
      resizing = true;
      corner = h.dataset.corner;
      const rect = _root.getBoundingClientRect();
      startX = e.clientX; startY = e.clientY;
      startW = rect.width; startH = rect.height;
      startLeft = rect.left; startTop = rect.top;
      _root.classList.add('resizing');
      // Pin to left/top so the resize math doesn't fight the default
      // right/bottom CSS anchor.
      _root.style.left = `${startLeft}px`;
      _root.style.top = `${startTop}px`;
      _root.style.right = 'auto';
      _root.style.bottom = 'auto';
      try { h.setPointerCapture(e.pointerId); } catch (_) {}
    });
  });

  const onMove = (e) => {
    if (!resizing) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    let newW = startW, newH = startH;
    if (corner.includes('e')) newW = startW + dx;
    if (corner.includes('w')) newW = startW - dx;
    if (corner.includes('s')) newH = startH + dy;
    if (corner.includes('n')) newH = startH - dy;

    const clampedW = Math.max(RESIZE_MIN_W, Math.min(RESIZE_MAX_W, newW));
    const clampedH = Math.max(RESIZE_MIN_H, Math.min(RESIZE_MAX_H, newH));

    // For west/north corners the left/top edge moves with the drag.
    // Compute from the clamped size so we stop at the edge instead of
    // overshooting when the user drags past the min/max.
    let newL = startLeft, newT = startTop;
    if (corner.includes('w')) newL = startLeft + (startW - clampedW);
    if (corner.includes('n')) newT = startTop + (startH - clampedH);

    _root.style.width = `${clampedW}px`;
    _root.style.height = `${clampedH}px`;
    _root.style.left = `${newL}px`;
    _root.style.top = `${newT}px`;
  };

  const onUp = () => {
    if (!resizing) return;
    resizing = false;
    _root.classList.remove('resizing');
    _persistSize();
    _persistPosition();
  };

  _lifetime.addEventListener(window, 'pointermove', onMove);
  _lifetime.addEventListener(window, 'pointerup', onUp);
  _lifetime.addEventListener(window, 'pointercancel', onUp);
}

function _persistSize() {
  if (!_root) return;
  const rect = _root.getBoundingClientRect();
  _uSet(SIZE_STORAGE_KEY, JSON.stringify({
    width: rect.width, height: rect.height,
  }));
}

function _restoreSize() {
  if (!_root) return;
  try {
    const raw = _uGet(SIZE_STORAGE_KEY);
    if (!raw) return;
    const size = JSON.parse(raw);
    if (typeof size.width === 'number' && typeof size.height === 'number') {
      const w = Math.max(RESIZE_MIN_W, Math.min(RESIZE_MAX_W, size.width));
      const h = Math.max(RESIZE_MIN_H, Math.min(RESIZE_MAX_H, size.height));
      _root.style.width = `${w}px`;
      _root.style.height = `${h}px`;
    }
  } catch (_) { /* corrupted size record — fall back to CSS default */ }
}

// ── Cursor glance (occasional + click-driven) ─────────────────────
//
// Becca doesn't track the cursor continuously — that would override
// her natural blink/contact/avert rhythm and make her feel like a
// security camera. Instead:
//
//   - On click anywhere, she glances at the click point for ~1.4s
//     then releases.
//   - On a 7-14s random cadence, she glances at the current cursor
//     position for ~1.0s then releases (only if cursor is reachable
//     within her eye range — keeps the glance subtle).
//
// During a glance, ``_externalGaze`` is set; the animator's saccades
// + microsaccades layer on top so the eyes still feel alive. After
// the glance window, we clear and the auto state machine resumes.
//
// Passive pointermove listener tracks the latest cursor position but
// does NOT drive gaze; it just feeds the glance timers.

const GAZE_MAX_YAW_DEG = 16;
const GAZE_MAX_PITCH_DEG = 11;
const GLANCE_CLICK_DURATION_MS = 1400;
const GLANCE_AUTO_DURATION_MS = 1000;
const GLANCE_AUTO_INTERVAL_MIN_MS = 7000;
const GLANCE_AUTO_INTERVAL_MAX_MS = 14000;

let _glanceReleaseTimer = null;
let _glanceAutoTimer = null;
let _gazeClickHandler = null;

function _attachCursorGaze() {
  if (_gazePointerHandler) return;
  _gazePointerHandler = (e) => {
    _gazeLastCx = e.clientX;
    _gazeLastCy = e.clientY;
  };
  _gazeLeaveHandler = () => { /* keep last known position */ };
  _gazeClickHandler = (e) => _glanceAt(e.clientX, e.clientY, GLANCE_CLICK_DURATION_MS);
  window.addEventListener('pointermove', _gazePointerHandler, { passive: true });
  document.addEventListener('mouseleave', _gazeLeaveHandler);
  window.addEventListener('pointerdown', _gazeClickHandler, { passive: true });
  _scheduleAutoGlance();
}

function _detachCursorGaze() {
  if (_gazePointerHandler) {
    window.removeEventListener('pointermove', _gazePointerHandler);
    _gazePointerHandler = null;
  }
  if (_gazeLeaveHandler) {
    document.removeEventListener('mouseleave', _gazeLeaveHandler);
    _gazeLeaveHandler = null;
  }
  if (_gazeClickHandler) {
    window.removeEventListener('pointerdown', _gazeClickHandler);
    _gazeClickHandler = null;
  }
  if (_glanceReleaseTimer) { clearTimeout(_glanceReleaseTimer); _glanceReleaseTimer = null; }
  if (_glanceAutoTimer) { clearTimeout(_glanceAutoTimer); _glanceAutoTimer = null; }
  _releaseCursorGaze();
}

function _scheduleAutoGlance() {
  if (_glanceAutoTimer) clearTimeout(_glanceAutoTimer);
  const wait = GLANCE_AUTO_INTERVAL_MIN_MS
             + Math.random() * (GLANCE_AUTO_INTERVAL_MAX_MS - GLANCE_AUTO_INTERVAL_MIN_MS);
  _glanceAutoTimer = setTimeout(() => {
    if (!_root || _root.style.display === 'none') {
      _scheduleAutoGlance();
      return;
    }
    // Only auto-glance if we have a cursor position AND we're not
    // currently glancing already (don't extend an existing glance).
    if ((_gazeLastCx || _gazeLastCy) && !avatarState.animator?._externalGaze) {
      _glanceAt(_gazeLastCx, _gazeLastCy, GLANCE_AUTO_DURATION_MS);
    }
    _scheduleAutoGlance();
  }, wait);
}

function _glanceAt(cx, cy, durationMs) {
  const animator = avatarState.animator;
  if (!animator || !_root || _root.style.display === 'none') return;
  // Widget center on screen, biased toward where her head actually is.
  const rect = _root.getBoundingClientRect();
  const wx = rect.left + rect.width / 2;
  const wy = rect.top + rect.height * 0.32;
  const dx = cx - wx;
  const dy = cy - wy;
  const nx = Math.max(-1, Math.min(1, dx / (window.innerWidth / 2)));
  const ny = Math.max(-1, Math.min(1, dy / (window.innerHeight / 2)));
  // Empirically the animator's yaw/pitch sign convention is opposite
  // to the world-space reasoning — positive yaw lands viewer-right,
  // positive pitch lands viewer-down. Cursor at screen-right should
  // produce positive yaw; cursor at screen-bottom positive pitch.
  const yaw = nx * GAZE_MAX_YAW_DEG;
  const pitch = ny * GAZE_MAX_PITCH_DEG;
  animator._externalGaze = { yaw, pitch };

  if (_glanceReleaseTimer) clearTimeout(_glanceReleaseTimer);
  _glanceReleaseTimer = setTimeout(_releaseCursorGaze, durationMs);
}

function _releaseCursorGaze() {
  const animator = avatarState.animator;
  if (animator) animator._externalGaze = null;
  if (_glanceReleaseTimer) { clearTimeout(_glanceReleaseTimer); _glanceReleaseTimer = null; }
}

// ── Surface adaptation ────────────────────────────────────────────

function _detectSurface() {
  // Voice-pill is always in the DOM as a permanent affordance; it's
  // only visually present when a call has been minimized AND it has
  // the .pet-mode class (see voice.css ~line 3690). Check both so we
  // don't perma-hide ourselves on a page that has the element but no
  // active call.
  const voicePill = document.querySelector('.voice-pill-float');
  if (voicePill && voicePill.classList.contains('pet-mode')) {
    const cs = getComputedStyle(voicePill);
    if (cs.display !== 'none' && cs.visibility !== 'hidden') {
      return 'voice';
    }
  }
  // Fullscreen content
  if (document.fullscreenElement) return 'fullscreen';
  // Mobile sizes
  if (window.matchMedia && window.matchMedia('(max-width: 600px)').matches) return 'mobile-portrait';
  // Screen-share detection is browser-specific; for v0 we honor an
  // explicit toggle via window.__beccaScreenShare.
  if (window.__beccaScreenShare) return 'screen-share';
  return 'private';
}

function _applySurface(surface) {
  if (!_root) return;
  _activeSurface = surface;
  // While the live camera is on, IT owns the widget layout (full-screen). Don't
  // re-detect / hide (display:none on 'voice') / resize underneath it — that's
  // the summon/call race that snapped the camera back to widget size. Surface
  // re-syncs when the camera stops (see _stopCameraView).
  if (_cameraView) return;
  if (surface === 'voice') {
    _root.style.display = 'none';
    // Widget is display:none — pause the (otherwise uncapped) render loop
    // so a minimized call doesn't keep burning frames nobody can see.
    _setOccluded(true);
  } else {
    _root.style.display = '';
    _root.dataset.surface = surface;
    _setOccluded(false);
  }
}

function _watchSurfaceChanges() {
  if (!_lifetime) return;
  const reapply = () => _applySurface(_detectSurface());
  _lifetime.addEventListener(document, 'fullscreenchange', reapply);
  _lifetime.addEventListener(window, 'resize', reapply);

  // Voice-pill detection — the previous code only re-evaluated surface
  // on fullscreenchange / resize, so a call ending without either of
  // those events would leave _activeSurface stuck on 'voice' (with the
  // widget visible only because __beccaReactivateVRM directly toggled
  // display). Now we observe class/style changes on the pill itself,
  // AND watch the body for pill (re)creation in case the call modal
  // wasn't mounted yet at widget-mount time.
  let pillObserver = null;
  const attachPillObserver = (pill) => {
    if (!pill || pillObserver) return;
    pillObserver = new MutationObserver(reapply);
    pillObserver.observe(pill, {
      attributes: true,
      attributeFilter: ['class', 'style'],
    });
    _lifetime.addObserver(pillObserver);
  };
  attachPillObserver(document.querySelector('.voice-pill-float'));

  // Catch the pill if it's mounted after the widget — cheap childList
  // observer scoped to body, disconnected by lifetime on unmount.
  const bodyObserver = new MutationObserver(() => {
    if (pillObserver) return;
    const pill = document.querySelector('.voice-pill-float');
    if (pill) attachPillObserver(pill);
  });
  bodyObserver.observe(document.body, { childList: true });
  _lifetime.addObserver(bodyObserver);
}

// ── Bus listener ──────────────────────────────────────────────────

async function _connectBus() {
  // Augmentum WS routes are auth-gated by short-lived tickets — match
  // the pattern voice.js / terminal.js use. Without the ticket query
  // param the connection 403s before reaching the route handler.
  let ticket;
  try {
    const r = await fetch('/api/auth/ws-ticket', {
      method: 'POST', credentials: 'same-origin',
    });
    if (!r.ok) {
      console.warn('[becca-presence] ws-ticket failed', r.status);
      // Returning null lets the reconnector back off and retry with a
      // fresh ticket — maybe the session just needs a refresh.
      return null;
    }
    const data = await r.json();
    ticket = data.ticket;
  } catch (e) {
    return null;
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/companion/presence`
            + `?ticket=${encodeURIComponent(ticket)}`
            + `&slice_key=becca-widget`;
  let ws;
  try {
    ws = new WebSocket(url);
  } catch (e) {
    console.warn('[becca-presence] bus connect failed', e);
    return null;
  }
  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch (_) { return; }
    _handleBusEvent(msg);
  };
  ws.onerror = () => {};
  ws.onclose = () => {
    // Tickets are short-TTL so we always re-fetch rather than reusing.
    // The reconnector applies full-jitter backoff and resets once the
    // fresh socket reaches OPEN.
    _wsBus = null;
    if (_root && _busReconnector) _busReconnector.schedule();
  };
  _wsBus = ws;
  return ws;
}

// Map server-side affect tags → avatar emotion names. The runtime's
// affect_tag vocabulary (curious / patient / melancholy / alert /
// tender / weary / warm / frustrated / delighted / unsure / concerned)
// is richer than the avatar's emotion enum, so we fold synonyms onto
// the closest visual cousin. Unmapped tags fall through to 'curious'
// — a non-zero neutral that still signals interior activity.
const _AFFECT_TO_EMOTION = {
  curious:    'curious',
  alert:      'curious',
  warm:       'happy',
  delighted:  'happy',
  tender:     'happy',
  patient:    'relaxed',
  settled:    'relaxed',
  weary:      'sad',
  melancholy: 'sad',
  unsure:     'nervous',
  concerned:  'nervous',
  frustrated: 'angry',
};

// Map (attention_state, role.dominant) → an idle emotion the bridge
// nudges toward when the state machine transitions. Stateful idle
// vs audio-cued spikes — interior baseline.
const _STATE_TO_EMOTION = {
  present:    'curious',
  dormant:    'relaxed',
  asleep:     'sad',     // closest to "drowsy" in the avatar vocab
};

// Continuous PAD (valence, arousal) → discrete emotion enum. Mirrors
// the precedence ladder in avatar-presence.js:_deriveEmotion so the
// bridge produces visually-consistent results regardless of whether
// the trigger was audio-derived or interior-PAD-derived. Below the
// "low signal" floor we return null and the override never fires —
// neutral noise shouldn't compete with audio-derived emotion.
function _padToEmotion(valence, arousal) {
  // Low signal — let audio-derived emotion stand. The avatar's
  // ambient face is plenty expressive on its own.
  if (Math.abs(valence) < 0.12 && Math.abs(arousal - 0.4) < 0.12) return null;

  if (arousal > 0.55 && valence > 0.3) return 'excited';
  if (arousal > 0.55 && valence < -0.3) return 'angry';
  if (arousal > 0.5) return 'surprised';
  if (valence > 0.35) return 'happy';
  if (valence < -0.4 && arousal < 0.3) return 'sad';
  if (valence < -0.2) return 'nervous';
  if (arousal < 0.25 && Math.abs(valence) < 0.18) return 'relaxed';
  return 'curious';
}

// Activity verbs surfaced on the status row when behavior.* events
// fire. Short and present-tense so the row reads as an action in flight.
const _ACTIVITY_LABEL = {
  journal:           'reflecting',
  revisit:           'noticing',
  revisit_thread:    'noticing',
  creation:          'writing',
  observation:       'watching',
  reach_out:         'reaching',
  scene_update:      'attending',
  dream:             'dreaming',
};


function _handleBusEvent(msg) {
  if (!_root) return;
  const { topic, payload } = msg;
  try {
    const routed = _animationRouter?.onRuntimeBusEvent(msg);
    if (routed && typeof routed.catch === 'function') {
      routed.catch(err => console.warn('[becca] animation router bus failed', err));
    }
  } catch (err) {
    console.warn('[becca] animation router bus failed', err);
  }
  // Headless tool fires — scheduled verb_fire tasks and timer
  // then-actions publish their surface_emit on the bus with an EMPTY
  // session_id (live calls carry one and reach the client through the
  // per-session intent_action queue instead; forwarding those too
  // would double-dispatch every effect). Route to the intent-action
  // router so "pause the music in 20 minutes" actually pauses it.
  if (
    payload
    && payload.source === 'companion_tool_call'
    && !payload.session_id
    && payload.payload !== undefined
  ) {
    import('./intent-action-router.js')
      .then(m => m.dispatchIntentAction({
        surface: { channel: topic, payload: payload.payload },
      }))
      .catch(err => console.warn('[becca] headless effect dispatch failed', err));
    return;
  }
  if (topic === 'initiative.surfaced' || topic === 'behavior.reach_out') {
    _setGlow('reaching', 2400);
  } else if (topic === 'voice.tool_call') {
    _setGlow('cool', 1600);
  } else if (topic === 'personality.labeled' || topic === 'voice.completed') {
    _setGlow('warm', 1400);
  } else if (topic === 'affect.changed') {
    // New: interior affect bridges into the visual layer. Avatar's
    // expression follows her interior reality for the next ~8s,
    // then audio-derived takes back over.
    const tag = String(payload?.tag || '').toLowerCase();
    const emotion = _AFFECT_TO_EMOTION[tag];
    if (emotion && avatarState?.presence?.setEmotionOverride) {
      avatarState.presence.setEmotionOverride(emotion, 8000);
    }
    _setGlow('warm', 1400);
  } else if (topic === 'affect.pad') {
    // Continuous PAD substrate from the runtime's perception layer.
    // Bridges valence/arousal into the avatar's emotion baseline so
    // her face drifts in response to interior facet activations even
    // between discrete affect-tag changes. Longer decay (12s) than
    // affect.changed because PAD shifts are slower and we don't want
    // a flickery face mid-conversation.
    const emotion = _padToEmotion(
      Number(payload?.valence) || 0,
      Number(payload?.arousal) || 0,
    );
    if (emotion && avatarState?.presence?.setEmotionOverride) {
      avatarState.presence.setEmotionOverride(emotion, 12000);
    }
  } else if (topic === 'state.transition' || topic === 'role.transition') {
    // State-axis transitions nudge the idle emotional baseline. We
    // pull a shorter 4s override so role-shift micro-pulses don't
    // hold the face mid-expression for too long.
    const to = String(payload?.to || '').toLowerCase();
    const emotion = _STATE_TO_EMOTION[to];
    if (emotion && avatarState?.presence?.setEmotionOverride) {
      avatarState.presence.setEmotionOverride(emotion, 4000);
    }
  } else if (topic === 'behavior.activity_chosen' || topic === 'behavior.creation_made') {
    // Briefly surface what she's doing on the status row. Reuses the
    // channel-state mechanism for the label — same affordance the
    // user already understands ("→ reflecting", "→ noticing").
    const kind = String(payload?.kind || '').toLowerCase();
    const label = _ACTIVITY_LABEL[kind] || kind || '';
    if (label) {
      _setStatusOverride(`· ${label}`);
      _setGlow('warm', 1400);
      // Clear the override after the activity's typical duration so
      // the row returns to listening/idle. 5s is enough that the
      // user reads it; short enough that a longer activity isn't
      // mislabeled past its actual completion.
      if (_activityClearTimer) clearTimeout(_activityClearTimer);
      _activityClearTimer = setTimeout(() => {
        _setStatusOverride(null);
        _activityClearTimer = null;
      }, 5000);
    }
  } else if (topic === 'channel.entering') {
    // Step aside: shrink to peripheral. Label the status row so the
    // user knows WHERE she stepped aside to (not just that she did).
    _root.style.opacity = '0.6';
    _root.style.transform = 'scale(0.9)';
    _setChannelState((payload && payload.channel) || '');
    // Self-healing fallback: if no exited event arrives within 15min
    // (currently always — no UI surface calls channel_exit yet), reset
    // visuals so the widget doesn't appear permanently dimmed.
    if (_channelClearTimer) clearTimeout(_channelClearTimer);
    _channelClearTimer = setTimeout(() => {
      if (_root) {
        _root.style.opacity = '';
        _root.style.transform = '';
        _setChannelState(null);
      }
      _channelClearTimer = null;
    }, 15 * 60 * 1000);
  } else if (topic === 'channel.user_idle') {
    // She's been waiting a while. Subtle pulse via CSS hook.
    if (_statusRow) _statusRow.dataset.channelIdle = '1';
  } else if (topic === 'channel.exiting') {
    // Stop the idle pulse — she's coming back.
    if (_statusRow) delete _statusRow.dataset.channelIdle;
  } else if (topic === 'channel.exited') {
    if (_channelClearTimer) { clearTimeout(_channelClearTimer); _channelClearTimer = null; }
    _root.style.opacity = '';
    _root.style.transform = '';
    _setChannelState(null);
    const microcopy = payload && payload.microcopy;
    if (microcopy) _showReturnFlash(microcopy);
  }
}

// Channel-handoff state on the status row. Reuses _setStatusOverride
// (the mechanism hosting/audio-source uses) for the label, and sets a
// data-channel-handoff attribute so CSS can recolor the status dot.
// Pass null to clear and restore the underlying idle/listening/etc state.
function _setChannelState(channelName) {
  if (!channelName) {
    if (_statusRow) {
      delete _statusRow.dataset.channelHandoff;
      delete _statusRow.dataset.channelIdle;
    }
    _setStatusOverride(null);
    return;
  }
  const label = _formatChannelName(channelName);
  if (_statusRow) _statusRow.dataset.channelHandoff = channelName;
  _setStatusOverride(`→ ${label}`);
}

function _formatChannelName(name) {
  // bug_finder -> "bug finder"; coder/narrative/agentic stay as-is.
  return String(name || '').replace(/_/g, ' ').toLowerCase();
}

// Transient floating bubble — the return-microcopy line surfaced when
// a channel exits. Auto-clears after ~6s so it doesn't linger.
function _showReturnFlash(text) {
  if (!_root || !_returnFlash) return;
  _returnFlash.textContent = _truncateGraceful(text, 110);
  _root.dataset.returnFlash = 'true';
  // Cancel any prior decay so back-to-back returns extend the visible
  // window rather than letting the earlier timer wipe the newer text.
  if (_returnFlashTimer) clearTimeout(_returnFlashTimer);
  _returnFlashTimer = setTimeout(() => {
    _returnFlashTimer = null;
    if (_root && _root.dataset.returnFlash === 'true') {
      delete _root.dataset.returnFlash;
    }
  }, 6000);
}

function _setGlow(kind, durationMs) {
  if (!_root) return;
  _root.dataset.glow = kind;
  setTimeout(() => {
    if (_root && _root.dataset.glow === kind) {
      delete _root.dataset.glow;
    }
  }, durationMs);
}

// ── Keyboard shortcuts ────────────────────────────────────────────

function _attachKeyboardShortcuts() {
  if (!_lifetime) return;
  _lifetime.addEventListener(document, 'keydown', (e) => {
    // Cmd/Ctrl + Shift + . — discreet mode toggle (Lane 4 §11)
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === '.') {
      e.preventDefault();
      toggleDiscreetMode();
    }
    // Alt + B / Option + B — focus the widget
    if (e.altKey && (e.key === 'b' || e.key === 'B')) {
      e.preventDefault();
      if (_root) _root.focus();
    }
  });
}

function _showDiscreetFlash(entering) {
  const flash = document.createElement('div');
  flash.style.cssText = `
    position: fixed; inset: 0;
    z-index: 2147483647;
    pointer-events: none;
    display: flex; align-items: center; justify-content: center;
    color: rgba(248, 232, 200, 0.85);
    font: 500 18px/1 system-ui, sans-serif;
    background: rgba(0, 0, 0, 0.32);
    opacity: 0;
    transition: opacity 220ms ease-out;
  `;
  flash.textContent = entering ? 'discreet · on' : 'back · quietly';
  document.body.appendChild(flash);
  requestAnimationFrame(() => { flash.style.opacity = '1'; });
  setTimeout(() => {
    flash.style.opacity = '0';
    setTimeout(() => flash.remove(), 250);
  }, 240);
}

// Auto-mount when companion_persona_mode is detected. The bootstrap
// gate is the global ``window.__companionPersonaMode`` set by the UI's
// settings layer; the settings layer should set this from a /api/config
// fetch at boot.
if (typeof window !== 'undefined') {
  window.mountBeccaPresence = mountBeccaPresence;
  window.unmountBeccaPresence = unmountBeccaPresence;
  window.toggleBeccaDiscreet = toggleDiscreetMode;
}
