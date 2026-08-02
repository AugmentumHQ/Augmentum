/**
 * avatar-xr.js — WebXR / VR mode for the production avatar runtime.
 *
 * Hooked from voice.js when the user clicks the "Enter VR" button.
 * Reads the existing avatar scene/renderer/camera/vrm out of avatar.js's
 * avatarState; bolts on the modern-room scene + sitting pose + an XR rig
 * positioned at the user's saved seat, then requests an immersive-vr
 * session. On exit, restores the original scene state so the desktop
 * voice call continues seamlessly.
 *
 * No three.js dependency at import time — THREE + addon loaders are
 * passed in by the caller (or lazy-imported from the local bundle).
 *
 * Persistence: seat coordinates live under the user_settings key
 * `ui.xrSeatLayout` (allowlisted in config_routes.py:_UI_SETTINGS).
 * Frontend stringifies JSON, server stores as string, frontend parses.
 */

import { extractErrorMessage } from './app.js';
import { POSE_PRESETS, BONE_ROTATION_ORDERS, applyPosePreset } from './avatar-pose-presets.js';
import { armAxisSignFromProfile, fingerAxisSignFromProfile } from './avatar-vrm-profile.js';
// Read-only handle for in-VR status HUD. avatarState is the same object
// voice.js mutates on every WS state event, so reading it per-frame here
// is the cheapest way to keep the HUD in sync.
import { avatarState as appAvatarState } from './avatar.js';
import { bus } from './activity-bus.js';
import {
  initXRContact, tickXRContact, teardownXRContact,
} from './avatar-xr-contact.js';
import {
  initXRCompliance, tickXRCompliance, teardownXRCompliance,
} from './avatar-xr-compliance.js';
import {
  initXRRapier, tickXRRapier, teardownXRRapier,
} from './avatar-xr-rapier.js';
import {
  initBodyPhysicsCoordinator, tickBodyPhysicsCoordinator,
  teardownBodyPhysicsCoordinator,
} from './body-physics-coordinator.js';
import {
  initBodyPhysicsHUD, teardownBodyPhysicsHUD,
} from './body-physics-hud.js';
import {
  initAvatarAudioReactions, teardownAvatarAudioReactions,
} from './avatar-audio-reactions.js';
import * as xrSession from './xr-session.js';
import { describeXrSurface, formatXrActionLabel } from './xr-surface-adapters.js';
import { createXrSurfaceDataStore } from './xr-surface-data.js';
import { createXrWorkspaceRuntime } from './xr-workspace-runtime.js';
import { getWsTicket } from './auth.js';
import {
  buildXrEmbedUrl,
  ensureXrWebEmbedRoot,
  getXrWebEmbedState,
  hideXrWebEmbed,
  setXrWebEmbedAnchor,
  setXrWebEmbedCapabilities,
  setXrWebEmbedPresenting,
  showXrWebEmbed,
} from './xr-web-embed.js';
// Spatial audio: voice.js owns the TTS node graph; we request a route into
// the avatar's positional panner rather than rewiring it ourselves. This is
// a runtime-only use (called from _setupSpatialAudio), so the voice.js <->
// avatar-xr.js import cycle is harmless.
import * as voiceAudio from './voice.js';

const SETTINGS_KEY = 'xrSeatLayout';

// User defaults. The headset rig is placed at the saved seat, while Becca
// uses a separate room/seat avatar anchor that is clamped into the authored
// room bounds before the sit-pose lock runs.
const DEFAULT_SEAT = Object.freeze({
  x: -0.30, y: 0, z: 2.25, rotY: -Math.PI / 2, envId: 'modern-room',
});
const LEGACY_DEFAULT_SEAT_ROT_Y = Math.PI;

const DEFAULT_HUB_ANCHOR = Object.freeze({
  x: -0.42, y: 1.02, z: -1.06, rotY: 0.16, scale: 1,
});

const DEFAULT_XR_AVATAR_ANCHOR = Object.freeze({
  x: 0.75, y: -0.24, z: 1.95, rotY: 3.11,
});

const MODERN_ROOM_AVATAR_BOUNDS = Object.freeze({
  minX: -1.08,
  maxX: 1.08,
  minZ: -1.12,
  maxZ: 1.95,
});

const XR_HUB_SURFACES = Object.freeze([
  {
    id: 'chat',
    label: 'Chat',
    action: 'chat',
    hint: 'conversation + pins',
    embedUrl: '/ui/?xrEmbed=1&mode=passthrough&xrSurface=chat',
    primaryActions: ['reply', 'summarize', 'pin'],
  },
  {
    id: 'analytical',
    label: 'Analyze',
    action: 'analytical',
    hint: 'research + reasoning',
    placement: 'left-stage',
    embedUrl: '/ui/?xrEmbed=1&mode=analytical&xrSurface=analytical',
    primaryActions: ['search', 'compare', 'explain'],
  },
  {
    id: 'agentic',
    label: 'Build',
    action: 'agentic',
    hint: 'tasks + execution',
    placement: 'right-stage',
    embedUrl: '/ui/?xrEmbed=1&mode=agentic&xrSurface=agentic',
    primaryActions: ['plan', 'execute', 'check_status'],
  },
  {
    id: 'narrative',
    label: 'Story',
    action: 'narrative',
    hint: 'characters + scene',
    embedUrl: '/ui/?xrEmbed=1&mode=narrative&xrSurface=narrative',
    primaryActions: ['continue_scene', 'switch_speaker', 'summarize_scene'],
  },
  {
    id: 'files',
    label: 'Files',
    action: 'files',
    hint: 'docs + context',
    embedUrl: '/ui/?xrEmbed=1&surface=files&xrSurface=files',
    primaryActions: ['open', 'attach', 'compare'],
  },
  {
    id: 'browse',
    label: 'Browse',
    action: 'browse',
    hint: 'search + sources',
    embedUrl: '/ui/?xrEmbed=1&surface=browse&xrSurface=browse',
    primaryActions: ['search', 'summarize_page', 'save_source', 'play_media'],
  },
  {
    id: 'coder',
    label: 'Coder',
    action: 'coder',
    hint: 'plans + diffs + tests',
    embedUrl: '/ui/?xrEmbed=1&mode=coder&xrSurface=coder',
    primaryActions: ['show_plan', 'review_diff', 'run_checks'],
  },
  {
    id: 'notes',
    label: 'Notes',
    action: 'notes',
    hint: 'dictation + clips',
    embedUrl: '/ui/?xrEmbed=1&surface=notes&xrSurface=notes',
    primaryActions: ['dictate', 'clip', 'organize'],
  },
  {
    id: 'studio',
    label: 'Studio',
    action: 'studio',
    hint: 'images + artifacts',
    embedUrl: '/ui/?xrEmbed=1&surface=studio&xrSurface=studio',
    primaryActions: ['generate', 'variant', 'edit'],
  },
  {
    id: 'media',
    label: 'Media',
    action: 'media',
    hint: 'watch + read + listen',
    embedUrl: '/ui/?xrEmbed=1&surface=media&xrSurface=media',
    primaryActions: ['continue', 'shows_movies', 'comics', 'audiobooks', 'images', 'local_files', 'games'],
  },
  {
    id: 'devices',
    label: 'Devices',
    action: 'devices',
    hint: 'cast + pair',
    embedUrl: '/ui/?xrEmbed=1&surface=devices&xrSurface=devices',
    primaryActions: ['cast', 'volume', 'pair'],
  },
  {
    id: 'games',
    label: 'Games',
    action: 'games',
    hint: 'launch + stream',
    embedUrl: '/ui/?xrEmbed=1&surface=games&xrSurface=games',
    primaryActions: ['launch', 'resume', 'controller_mode', 'stop_stream'],
  },
]);

const XR_PANEL_POSES = Object.freeze({
  'left-near':     { x: -0.78, y: 1.22, z: -0.54, rotY: 0.18 },
  'right-near':    { x: 0.78,  y: 1.22, z: -0.54, rotY: -0.18 },
  'center-stage':  { x: 0.00,  y: 1.24, z: -0.76, rotY: 0.00 },
  'left-stage':    { x: -0.46, y: 1.22, z: -0.86, rotY: 0.12 },
  'right-stage':   { x: 0.46,  y: 1.22, z: -0.86, rotY: -0.12 },
  'left-shelf':    { x: -1.02, y: 1.15, z: -0.72, rotY: 0.32 },
  'right-wall':    { x: 1.02,  y: 1.20, z: -0.82, rotY: -0.34 },
  'left-desk':     { x: -0.66, y: 0.96, z: -0.42, rotY: 0.14 },
  'far-wall':      { x: 0.00,  y: 1.30, z: -1.24, rotY: 0.00 },
  'far-center':    { x: 0.00,  y: 1.18, z: -1.36, rotY: 0.00 },
  'right-console': { x: 0.96,  y: 1.02, z: -0.55, rotY: -0.30 },
  'far-right':     { x: 0.94,  y: 1.20, z: -1.12, rotY: -0.30 },
  default:         { x: 0.00,  y: 1.16, z: -0.88, rotY: 0.00 },
});

const XR_HAND_PINCH_START_M = 0.032;
const XR_HAND_PINCH_END_M = 0.052;
const XR_HAND_AIM_MAX_DISTANCE_M = 2.4;
const XR_HAND_AIM_IDLE_DISTANCE_M = 1.1;
const XR_HAND_PALM_SUMMON_MS = 700;
const XR_HAND_PALM_SUMMON_COOLDOWN_MS = 2600;
const XR_HAND_MENU_SUPPRESS_AFTER_HIDE_MS = 3600;
const XR_HAND_PALM_MAX_MOTION_M = 0.07;
const XR_HAND_MENU_PALM_FACING_DOT = 0.54;
const XR_HAND_NATIVE_SELECT_STALE_MS = 900;
const XR_HAND_NATIVE_BUTTON_THRESHOLD = 0.62;
const XR_HAND_PROBE_EVENT_LIMIT = 18;
const XR_HAND_DEBUG_REDRAW_MS = 160;
const XR_SURFACE_PANEL_REFRESH_MS = 250;
const XR_BROWSER_PANEL_REFRESH_MS = 900;
const XR_BROWSER_PANEL_STREAM_RECONNECT_MS = 850;
const XR_BROWSER_PANEL_STREAM_MAX_RECONNECTS = 5;
const XR_BROWSER_PANEL_WIDTH = 1440;
const XR_BROWSER_PANEL_HEIGHT = 900;
const XR_HAND_PINCH_SELECT_COOLDOWN_MS = 420;
const XR_GAZE_MAX_DISTANCE_M = 2.8;
const XR_GAZE_IDLE_DISTANCE_M = 1.18;
const XR_GAZE_DWELL_MS = 950;
const XR_GAZE_DWELL_COOLDOWN_MS = 900;
const XR_USER_SIGNAL_COOLDOWN_MS = 1400;
const XR_USER_WAVE_COOLDOWN_MS = 6500;
const XR_USER_WAVE_WINDOW_MS = 2300;
const XR_USER_WAVE_MIN_SPAN_M = 0.075;
const XR_USER_WAVE_MIN_TRAVEL_M = 0.16;
const XR_USER_WAVE_MIN_REVERSALS = 1;
const XR_USER_HAND_NEAR_AVATAR_M = 0.28;
const XR_USER_HAND_CONTACT_AVATAR_M = 0.15;
const XR_GESTURE_RESPONSE_MODE_KEY = 'augmentum.xr.gestureResponseMode';
const XR_GAZE_DWELL_SELECT_KEY = 'augmentum.xr.gazeDwellSelect';
const XR_FRAMEBUFFER_SCALE_KEY = 'augmentum.xr.framebufferScale';
const XR_FIXED_FOVEATION_KEY = 'augmentum.xr.fixedFoveation';
const XR_PANEL_MIN_SCALE = 0.72;
const XR_PANEL_MAX_SCALE = 1.65;
const XR_PANEL_MAX_CURVE_M = 0.12;
const XR_PANEL_LAYOUT_PRESETS = Object.freeze({
  work: {
    id: 'work',
    label: 'Work',
    logicalW: 896,
    logicalH: 560,
    canvasW: 1440,
    canvasH: 900,
    widthM: 0.98,
    heightM: 0.62,
    curveM: 0,
    viewDistanceM: 1.08,
    viewY: 1.18,
    browserW: 1440,
    browserH: 900,
  },
  manga: {
    id: 'manga',
    label: 'Manga',
    logicalW: 720,
    logicalH: 1040,
    canvasW: 1080,
    canvasH: 1560,
    widthM: 0.62,
    heightM: 0.9,
    curveM: 0.025,
    viewDistanceM: 0.86,
    viewY: 1.16,
    browserW: 900,
    browserH: 1400,
  },
  tv: {
    id: 'tv',
    label: 'TV',
    logicalW: 1280,
    logicalH: 720,
    canvasW: 1920,
    canvasH: 1080,
    widthM: 1.58,
    heightM: 0.89,
    curveM: 0.065,
    viewDistanceM: 2.05,
    viewY: 1.26,
    browserW: 1600,
    browserH: 900,
  },
});
const XR_MODE_VR = 'vr';
const XR_MODE_MR = 'mr';
const XR_EXPERIMENTAL_DOM_OVERLAY_KEY = 'augmentum.xr.experimentalDomOverlay';
const MR_PLACEMENT_SETTLE_MS = 1600;
const MR_PLACEMENT_MIN_DISTANCE_M = 0.65;
const MR_PLACEMENT_MAX_DISTANCE_M = 2.6;
const MR_PLACEMENT_DEFAULT_DISTANCE_M = 1.2;
const MR_PLACEMENT_RETICLE_Y_OFFSET_M = 0.018;

// Bones whose rotation we re-stamp every frame to keep the seated pose
// stable against AvatarAnimator's procedural writes. Lower-body only —
// spine/chest/arms/head stay unlocked so breathing, gaze, and gestures
// continue to play. Hips position is also re-stamped (separately) since
// the animator's simplex-noise sway perturbs hip Y.
const SIT_LOCKED_BONE_ROTATIONS = Object.freeze([
  'hips',
  'leftUpperLeg', 'rightUpperLeg',
  'leftLowerLeg', 'rightLowerLeg',
  'leftFoot',     'rightFoot',
]);

// ── Capability detection (cached) ────────────────────────────────────
const _sessionSupportPromises = new Map();

export function isSessionModeSupported(sessionMode = 'immersive-vr') {
  if (_sessionSupportPromises.has(sessionMode)) return _sessionSupportPromises.get(sessionMode);
  if (!navigator.xr?.isSessionSupported) {
    const unsupported = Promise.resolve(false);
    _sessionSupportPromises.set(sessionMode, unsupported);
    return unsupported;
  }
  const promise = navigator.xr
    .isSessionSupported(sessionMode)
    .catch(() => false);
  _sessionSupportPromises.set(sessionMode, promise);
  return promise;
}

// navigator.xr.isSessionSupported returns true on plain Chrome (desktop or
// Android) whenever the WebXR API surface is present, even with no
// headset connected — so feature detection alone reveals the VR/MR
// buttons on every device. Gate the UI on a UA that identifies a real
// immersive browser. Known headset UAs: Meta Quest Browser
// (OculusBrowser), Pico (Pico Neo / VRBrowser), Wolvic, HTC Vive's
// embedded browser (Vive-VR), visionOS (Vision Pro), Magic Leap. Desktop
// users with a tethered headset can still hit `openVrEntry()` via deep
// link if needed; the toast tells them what's expected.
const _HEADSET_UA_RE = /OculusBrowser|Quest|Pico Neo|VRBrowser|Vive-VR|Wolvic|VisionOS|Apple Vision|MagicLeap/i;

function _isPlausibleHeadsetClient() {
  return _HEADSET_UA_RE.test(navigator.userAgent || '');
}

export function isXRSupported() {
  if (!_isPlausibleHeadsetClient()) return Promise.resolve(false);
  return isSessionModeSupported('immersive-vr');
}

export function isMRSupported() {
  if (!_isPlausibleHeadsetClient()) return Promise.resolve(false);
  return isSessionModeSupported('immersive-ar');
}

// ── Seat persistence ─────────────────────────────────────────────────
async function loadSavedSeat() {
  try {
    const r = await fetch('/api/config/ui');
    if (!r.ok) return { ...DEFAULT_SEAT };
    const data = await r.json();
    if (!data?.[SETTINGS_KEY]) return { ...DEFAULT_SEAT };
    const parsed = JSON.parse(data[SETTINGS_KEY]);
    return { ...DEFAULT_SEAT, ...parsed };
  } catch (err) {
    console.warn('[avatar-xr] loadSavedSeat failed, using default', err);
    return { ...DEFAULT_SEAT };
  }
}

function _nearNumber(value, target, epsilon = 0.035) {
  return Math.abs(Number(value) - target) <= epsilon;
}

function _finiteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function _clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function _normalizeSeatForXrVisibility(seat = {}) {
  const normalized = { ...DEFAULT_SEAT, ...(seat || {}) };
  const metadata = normalized.metadata || {};
  const looksLikeLegacyDefault = (
    !metadata.calibrated
    && _nearNumber(normalized.x, -0.30)
    && (_nearNumber(normalized.z, 2.25) || _nearNumber(normalized.z, 2.30))
    && _nearNumber(normalized.rotY, LEGACY_DEFAULT_SEAT_ROT_Y, 0.08)
  );
  if (!looksLikeLegacyDefault) return normalized;
  return {
    ...normalized,
    rotY: DEFAULT_SEAT.rotY,
    metadata: {
      ...metadata,
      xrAutoFacingApplied: true,
    },
  };
}

export async function saveSeat(seat) {
  let serverSaved = false;
  try {
    await xrSession.saveSeat('default', {
      label: 'Default seat',
      x: Number(seat?.x ?? DEFAULT_SEAT.x),
      y: Number(seat?.y ?? DEFAULT_SEAT.y),
      z: Number(seat?.z ?? DEFAULT_SEAT.z),
      rotY: Number(seat?.rotY ?? DEFAULT_SEAT.rotY),
      envId: seat?.envId || DEFAULT_SEAT.envId,
      metadata: { source: 'avatar-xr' },
    });
    serverSaved = true;
  } catch (err) {
    console.warn('[avatar-xr] server saveSeat failed', err);
  }
  try {
    const r = await fetch('/api/config/ui', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [SETTINGS_KEY]: JSON.stringify(seat) }),
    });
    return r.ok || serverSaved;
  } catch (err) {
    console.warn('[avatar-xr] saveSeat failed', err);
    return serverSaved;
  }
}

export async function getCurrentSeat() {
  return loadSavedSeat();
}

// ── Internal state ───────────────────────────────────────────────────
const xrState = {
  active: false,
  // Refs to caller's runtime — captured on enter, cleared on exit.
  THREE: null,
  scene: null,
  renderer: null,
  camera: null,
  vrm: null,
  // Restoration snapshots — saved on enter, applied on exit.
  savedCameraParent: null,
  savedCameraPosition: null,
  savedCameraQuaternion: null,
  savedSceneBackground: null,
  savedVrmPosition: null,
  savedVrmRotationY: null,
  savedVrmVisible: true,
  savedLookAtTarget: undefined,   // VRM lookAt target before we hijack it
  savedVrmOnBeforeRender: undefined,
  // GPU memory-pressure mitigation. Saved on enter, restored on exit.
  // VR + local llama-server contend for VRAM on shared-memory GPUs;
  // dropping these in XR keeps the LLM's allocation on-GPU and prevents
  // the "voice path blocked" symptom that's actually VRAM exhaustion.
  savedPixelRatio: null,
  savedFramebufferScale: null,
  savedClearAlpha: null,
  // Spatial audio for TTS (avatar voice from her head, not "everywhere").
  // Set up after session start; routes voice.js's analyserNode through a
  // PositionalAudio panner. Defensive — if any of this fails the existing
  // flat audio graph is left untouched.
  audioListener: null,
  positionalAudio: null,
  audioContext: null,
  audioRerouted: false,
  mrShadow: null,
  mrPlacement: null,
  boundaryLine: null,
  avatarXrPose: null,
  // Owned objects — disposed on exit.
  roomScene: null,
  xrRig: null,
  controllers: [],
  targetRayControllers: [],
  hands: [],
  handStates: new Map(),
  controllerListeners: [],   // [{ ctrl, ev, fn }] for removeEventListener
  sessionEndHandler: null,
  // In-VR HUD (camera-anchored status panel).
  hudCanvas: null,
  hudCtx: null,
  hudTexture: null,
  hudPlane: null,
  hudLastLabel: '',
  // World-locked mode hub. Lets the user route the live voice call into
  // Augmentum surfaces without exiting VR or hunting through DOM menus.
  modeHubGroup: null,
  modeHubPanel: null,
  modeHubCanvas: null,
  modeHubCtx: null,
  modeHubTexture: null,
  modeHubButtons: [],
  modeHubSurfaces: [],
  modeHubActiveAction: 'voice',
  modeHubStatus: 'Voice call is live. Select a surface to bring it into focus.',
  modeHubLastKey: '',
  hubDrag: null,
  hubUserPlaced: false,
  xrWorkspace: null,
  xrSurfaceData: null,
  spatialPanels: new Map(),
  lastLiveSurfaceRefreshAt: 0,
  panelDrag: null,
  panelDragWorldPoint: null,
  panelCameraWorld: null,
  handTempA: null,
  handTempB: null,
  handTempC: null,
  handTempD: null,
  handTempQuat: null,
  raycaster: null,
  rayOrigin: null,
  rayDirection: null,
  rayMatrix: null,
  gazeCursor: null,
  gazeCursorMat: null,
  gazeHit: null,
  gazeTargetKey: '',
  gazeTargetStartedAt: 0,
  gazeDwellArmed: false,
  lastGazeDwellAt: 0,
  lastNativeSelectAt: 0,
  lastComfortSelectAt: 0,
  handMenuGroup: null,
  handMenuPanel: null,
  handMenuCanvas: null,
  handMenuCtx: null,
  handMenuTexture: null,
  handMenuButtons: [],
  handMenuLastKey: '',
  handMenuHandIndex: null,
  handMenuShownAt: 0,
  handMenuHiddenAt: 0,
  handDebugGroup: null,
  handDebugPanel: null,
  handDebugCanvas: null,
  handDebugCtx: null,
  handDebugTexture: null,
  handDebugVisible: false,
  handDebugLastKey: '',
  handDebugLastDrawAt: 0,
  handProbeEvents: [],
  handProbeInputs: new Map(),
  nativeHandIntents: new Map(),
  xrFrameStats: {
    lastTime: 0,
    averageMs: 0,
    maxMs: 0,
    longFrames: 0,
    lastLongFrameAt: 0,
  },
  userSignals: [],
  userSignalLastAt: new Map(),
  lastAvatarWaveBackAt: 0,
  pendingHubConfirm: null,
  xrMode: XR_MODE_VR,
  sessionMode: 'immersive-vr',
  referenceSpaceType: 'local-floor',
  xrCapabilities: {},
  capabilityListener: null,
  webEmbedLastAnchorKey: '',
  serverSession: null,
  serverSessionId: null,
  sessionStartedAt: 0,
};

export function isInVR() {
  return xrState.active && !!xrState.renderer?.xr?.isPresenting;
}

export function isInMR() {
  return isInVR() && xrState.xrMode === XR_MODE_MR;
}

function _xrError(code, message, cause = null) {
  const err = new Error(message);
  err.code = code;
  if (cause) err.cause = cause;
  return err;
}

// TEMP experiment: route every surface panel through the dom-overlay
// iframe path (xr-web-embed.js) instead of the stereo-safe 3D textured
// panel (_startBrowserSurfacePanel + CDP screenshot stream).
// Tradeoff: native browser rendering (full page, real scroll, logged-in
// state, no server-side chromium) at the cost of being a monocular 2D
// layer instead of stereo geometry. Flip _wantDomOverlayPanels to revert.
function _wantDomOverlayPanels() {
  return true;
}

// True only when (a) we want dom-overlay panels AND (b) the active
// session actually granted dom-overlay. If the runtime denied the
// feature we fall back to the stereo CDP path so the user never sees
// a blank "live page" because the iframe is hidden behind the WebGL
// scene canvas during immersive.
function _useDomOverlayPanels() {
  if (!_wantDomOverlayPanels()) return false;
  const session = xrState.renderer?.xr?.getSession?.();
  if (!session) return false;
  // domOverlayState is the spec field; xrCapabilities.domOverlay is the
  // bool we record after session start. Either being truthy means the
  // feature was granted.
  return !!session.domOverlayState || !!xrState.xrCapabilities?.domOverlay;
}

function _shouldRequestDomOverlay() {
  if (_wantDomOverlayPanels()) return true;
  try {
    return window.localStorage?.getItem(XR_EXPERIMENTAL_DOM_OVERLAY_KEY) === '1';
  } catch {
    return false;
  }
}

function _isGazeDwellSelectEnabled() {
  try {
    const raw = String(window.localStorage?.getItem(XR_GAZE_DWELL_SELECT_KEY) || '').trim().toLowerCase();
    return raw === '1' || raw === 'true' || raw === 'yes';
  } catch {
    return false;
  }
}

function _isGazeFocusVisible() {
  return _isGazeDwellSelectEnabled() || _isMixedRealityAvatarPlacementMode();
}

function _xrFramebufferScale() {
  try {
    const raw = Number(window.localStorage?.getItem(XR_FRAMEBUFFER_SCALE_KEY));
    if (Number.isFinite(raw) && raw > 0) {
      return Math.max(0.72, Math.min(1.25, raw));
    }
  } catch {}
  return 1;
}

function _xrFixedFoveationFeature() {
  try {
    const raw = String(window.localStorage?.getItem(XR_FIXED_FOVEATION_KEY) || '').trim().toLowerCase();
    if (raw === 'low' || raw === 'medium' || raw === 'high') {
      return `${raw}-fixed-foveation-level`;
    }
  } catch {}
  return '';
}

// ── Lazy-load three addon loaders + THREE namespace + RoomScene ─────
// ES modules are URL-singletons, so importing the same three bundle
// here returns the exact module instance avatar.js already loaded —
// no double-load, no class-mismatch issues.
async function loadXRDeps() {
  const [THREE, { GLTFLoader }, { DRACOLoader }, roomMod, ctrlMod, handMod] = await Promise.all([
    import('../lib/three/three.module.min.js'),
    import('../lib/three/GLTFLoader.js'),
    import('../lib/three/DRACOLoader.js'),
    import('./scene/room-scene.js'),
    import('../lib/three/XRControllerModelFactory.js'),
    import('../lib/three/XRHandModelFactory.js'),
  ]);
  return {
    THREE,
    GLTFLoader,
    DRACOLoader,
    RoomScene: roomMod.RoomScene,
    XRControllerModelFactory: ctrlMod.XRControllerModelFactory,
    XRHandModelFactory: handMod.XRHandModelFactory,
  };
}

// ── Enter VR ─────────────────────────────────────────────────────────
/**
 * Enter immersive VR. Caller passes the live avatar runtime handles.
 * Returns true if the session started, false otherwise. On failure,
 * scene state is fully restored (no half-applied changes).
 *
 * Pre-condition: caller's renderer must have `renderer.xr.enabled = true`
 * set at construction time. avatar.js's createScene handles this.
 */
function _xrOptionalFeatures(xrMode) {
  const common = [
    'bounded-floor',
    'hand-tracking',
    'layers',
  ];
  const foveation = _xrFixedFoveationFeature();
  if (foveation) common.push(foveation);
  if (_shouldRequestDomOverlay()) common.push('dom-overlay');
  if (xrMode !== XR_MODE_MR) return common;
  return [
    ...common,
    'hit-test',
    'anchors',
    'plane-detection',
    'mesh-detection',
    'depth-sensing',
    'secondary-views',
  ];
}

function _xrSafeOptionalFeatures(xrMode) {
  const safe = ['bounded-floor', 'hand-tracking'];
  if (_shouldRequestDomOverlay()) safe.push('dom-overlay');
  const foveation = _xrFixedFoveationFeature();
  if (foveation) safe.push(foveation);
  return xrMode === XR_MODE_MR
    ? [...safe, 'hit-test']
    : safe;
}

async function _requestXrSession(sessionMode, xrMode, webEmbedRoot) {
  const attempts = [
    { requiredFeatures: ['local-floor'], optionalFeatures: _xrOptionalFeatures(xrMode) },
    { requiredFeatures: ['local-floor'], optionalFeatures: _xrSafeOptionalFeatures(xrMode) },
  ];
  if (sessionMode === 'immersive-ar') {
    attempts.push(
      { requiredFeatures: ['local'], optionalFeatures: _xrOptionalFeatures(xrMode) },
      { requiredFeatures: ['local'], optionalFeatures: _xrSafeOptionalFeatures(xrMode) },
    );
  }

  let lastErr = null;
  for (const init of attempts) {
    try {
      const sessionInit = { ...init };
      if (webEmbedRoot && init.optionalFeatures?.includes?.('dom-overlay')) {
        sessionInit.domOverlay = { root: webEmbedRoot };
      }
      if (init.optionalFeatures?.includes?.('depth-sensing')) {
        sessionInit.depthSensing = {
          usagePreference: ['cpu-optimized', 'gpu-optimized'],
          dataFormatPreference: ['luminance-alpha', 'float32'],
        };
      }
      const session = await navigator.xr.requestSession(sessionMode, sessionInit);
      return {
        session,
        requiredFeatures: init.requiredFeatures,
        optionalFeatures: init.optionalFeatures,
        referenceSpaceType: init.requiredFeatures.includes('local-floor') ? 'local-floor' : 'local',
      };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || _xrError('request-session-failed', 'Could not start XR session');
}

export async function enterVR({
  scene,
  renderer,
  camera,
  vrm,
  voiceSessionId = '',
  onError,
  sessionMode = 'immersive-vr',
  xrMode = XR_MODE_VR,
} = {}) {
  if (xrState.active) return false;
  if (!scene || !renderer || !camera) {
    const err = _xrError('missing-runtime', 'enterVR requires scene, renderer, camera');
    onError?.(err);
    return false;
  }
  if (!renderer.xr?.enabled) {
    const err = _xrError('renderer-not-xr-enabled', 'renderer.xr.enabled must be true before enterVR()');
    onError?.(err);
    return false;
  }
  const supported = await isSessionModeSupported(sessionMode);
  if (!supported) {
    onError?.(_xrError(`${sessionMode}-unsupported`, `${sessionMode} not supported on this device`));
    return false;
  }
  const isMixedReality = xrMode === XR_MODE_MR || sessionMode === 'immersive-ar';

  let requestedSession = null;
  let sessionPromise = null;
  let sessionRequest = null;
  try {
    const webEmbedRoot = ensureXrWebEmbedRoot();
    sessionPromise = _requestXrSession(sessionMode, isMixedReality ? XR_MODE_MR : XR_MODE_VR, webEmbedRoot)
      .then((result) => {
        sessionRequest = result;
        requestedSession = result.session;
        return result.session;
    });

    const serverSessionPromise = xrSession.createSession({
      surface: isMixedReality ? 'mixed-reality' : 'voice',
      voice_session_id: voiceSessionId || '',
      room_id: 'modern-room',
      seat_id: 'default',
      device_hint: {
        presentation: isMixedReality ? 'mixed-reality' : 'virtual-reality',
        session_mode: sessionMode,
      },
    }).catch((err) => {
      console.warn('[avatar-xr] server XR session unavailable; continuing local-only', err);
      return null;
    });
    const [savedSeat, deps, serverSession] = await Promise.all([
      loadSavedSeat(),
      loadXRDeps(),
      serverSessionPromise,
    ]);
    const seat = _normalizeSeatForXrVisibility(serverSession?.seat_layout || savedSeat);
    const {
      THREE, GLTFLoader, DRACOLoader, RoomScene,
      XRControllerModelFactory, XRHandModelFactory,
    } = deps;

    xrState.serverSession = serverSession;
    xrState.serverSessionId = serverSession?.id || null;
    xrState.sessionStartedAt = performance.now();
    xrState.xrMode = isMixedReality ? XR_MODE_MR : XR_MODE_VR;
    xrState.sessionMode = sessionMode;
    xrSession.recordEvent(xrState.serverSessionId, 'preflight_ok', {
      pwa: xrSession.isPwaLaunchContext(),
      room_id: seat.envId || 'modern-room',
      presentation: xrState.xrMode,
      seat_rot_y: seat.rotY,
      auto_facing: !!seat.metadata?.xrAutoFacingApplied,
    });

    // Snapshot for restoration
    xrState.THREE = THREE;
    xrState.scene = scene;
    xrState.renderer = renderer;
    xrState.camera = camera;
    xrState.vrm = vrm || null;
    xrState.savedCameraParent = camera.parent;
    xrState.savedCameraPosition = camera.position.clone();
    xrState.savedCameraQuaternion = camera.quaternion.clone();
    xrState.savedSceneBackground = scene.background;
    if (vrm) {
      xrState.savedVrmPosition = vrm.scene.position.clone();
      xrState.savedVrmRotationY = vrm.scene.rotation.y;
      xrState.savedVrmVisible = vrm.scene.visible !== false;
      xrState.savedVrmOnBeforeRender = vrm.scene.onBeforeRender;
      _prepareVrmSceneForXR(vrm);
      // Contact reactor: lets the AI avatar sense + react to user-hand
      // proximity / touch via the BodyAtlas + BodyMesh substrates that
      // avatar.js attached at load. See ui/scripts/avatar-xr-contact.js
      // for the integration module.
      try { initXRContact({ three: THREE, vrm, renderer }); }
      catch (err) { console.debug('[xr] contact reactor init skipped:', err?.message); }
      // SDF compliance: subtle body "give" when the user pokes/hovers near
      // the avatar. Reads user-hand positions through the contact reactor
      // (so both share one source of truth) and writes rotational deltas
      // on torso/neck/shoulder/head bones via the BodyAtlas gradient.
      try { initXRCompliance({ three: THREE, vrm }); }
      catch (err) { console.debug('[xr] compliance init skipped:', err?.message); }

      // Rapier ragdoll: global chain dynamics (secondary motion). Reads the
      // post-compliance pose each frame, runs a PD-spring kinematic body
      // chain, exposes per-bone deltas. Async init — wrapper starts the
      // import + RAPIER.init() in the background; tick() no-ops until ready.
      try { initXRRapier({ three: THREE, vrm }); }
      catch (err) { console.debug('[xr] rapier init skipped:', err?.message); }

      // Audio reactions: region-specific WebAudio cues fired from contact
      // events. AudioContext created lazily on first reaction.
      try { initAvatarAudioReactions({ vrm }); }
      catch (err) { console.debug('[xr] audio reactions init skipped:', err?.message); }

      // Coordinator: 5s settings poll → push body_physics_* values onto
      // live compliance + rapier instances via their public mutable
      // properties. Late-binds both channels.
      try { initBodyPhysicsCoordinator(); }
      catch (err) { console.debug('[xr] body-physics coordinator init skipped:', err?.message); }

      // HUD: hidden by default; Ctrl+Shift+B toggles. Polls peers via
      // late binding so init order doesn't matter.
      try { initBodyPhysicsHUD(); }
      catch (err) { console.debug('[xr] body-physics HUD init skipped:', err?.message); }
    }

    // Build the authored room for VR. MR keeps the scene transparent so
    // the WebXR compositor can blend passthrough behind our avatar/UI.
    if (isMixedReality) {
      scene.background = null;
    } else {
      xrState.roomScene = new RoomScene({ THREE, GLTFLoader, DRACOLoader, scene });
      await xrState.roomScene.load();
      xrState.roomScene.setEnvironment(seat.envId || 'modern-room');
    }

    // XR rig: a Group at the user's seat holding the camera. Camera's
    // local transform is owned by the headset once the session starts.
    const rigSeat = isMixedReality
      ? { x: 0, y: 0, z: 0, rotY: 0 }
      : seat;
    xrState.xrRig = new THREE.Group();
    xrState.xrRig.position.set(rigSeat.x, rigSeat.y, rigSeat.z);
    xrState.xrRig.rotation.y = rigSeat.rotY;
    scene.add(xrState.xrRig);
    camera.position.set(0, 0, 0);
    camera.quaternion.set(0, 0, 0, 1);
    xrState.xrRig.add(camera);

    // Apply seated pose to the active VRM, then install a per-frame
    // pose-lock callback that re-stamps the lower-body geometry after
    // AvatarAnimator runs. Without the lock, the animator's simplex-noise
    // hip sway and procedural writes drift the seated pose visibly —
    // user reports "sitting in the air" or "up and in the couch".
    if (vrm) {
      if (isMixedReality) {
        xrState.avatarXrPose = null;
        _applyMixedRealityCompanionPose(vrm, THREE);
        _setupMixedRealityGrounding(THREE, scene, vrm);
      } else {
        xrState.avatarXrPose = _resolveXrAvatarPose(seat, serverSession?.room_manifest);
        _applySittingPoseOnce(vrm, xrState.avatarXrPose);
        vrm.scene.onBeforeRender = () => _stampLockedSitPose(vrm, appAvatarState?.avatarFsm?.current?.());
      }
      // Eye-gaze on the user. three-vrm 2.x's vrm.update(delta) calls
      // lookAt.update() internally each frame, so a single assignment
      // makes Becca track the XR camera (which three.js drives from
      // the headset pose). Save the prior target so non-VR callers
      // that depend on the original behavior get it back on exit.
      if (vrm.lookAt) {
        xrState.savedLookAtTarget = vrm.lookAt.target;
        vrm.lookAt.target = camera;
      }
      _recordVrmVisibilitySnapshot('after_pose');
    }

    // Controllers + hands — real Touch controller meshes via grip space,
    // hand tracking via getHand, laser lines toggled on connected/
    // disconnected to handle Quest's mid-session controller↔hand swap.
    _setupControllers(
      THREE, renderer, xrState.xrRig,
      XRControllerModelFactory, XRHandModelFactory,
    );

    // Status HUD — small camera-anchored text panel showing voice state.
    // Without this, the immersive view has no UI affordance whatsoever
    // (the DOM voice overlay is hidden once the session presents).
    _setupHUD(THREE, camera);
    _setupModeHub(
      THREE,
      xrState.xrRig,
      serverSession?.room_manifest,
      serverSession?.surface_catalog || serverSession?.surfaces || null,
      serverSession?.room_state || null,
    );
    _setupControllerlessComfortUI(THREE, xrState.xrRig);
    xrState.xrWorkspace = createXrWorkspaceRuntime({
      surfaces: xrState.modeHubSurfaces,
      sessionId: xrState.serverSessionId,
      presentation: xrState.xrMode,
      capabilities: xrState.xrCapabilities,
    });
    xrState.xrSurfaceData = createXrSurfaceDataStore();
    xrState.xrSurfaceData.start();
    for (const panel of xrState.spatialPanels?.values?.() || []) {
      xrState.xrWorkspace.openSurface(panel.surface.action, {
        source: 'restore',
        label: panel.surface.label,
      }, { emit: false });
    }

    // XR render quality. The Quest's headset path renders
    // independently of the desktop mirror pixel ratio. The framebuffer
    // scale is configurable, but defaults to full-res because downscale
    // and fixed foveation make panels and avatar edges shimmer.
    xrState.savedPixelRatio = renderer.getPixelRatio();
    renderer.setPixelRatio(1);
    if (typeof renderer.xr.getFramebufferScaleFactor === 'function') {
      xrState.savedFramebufferScale = renderer.xr.getFramebufferScaleFactor();
    }
    if (typeof renderer.xr.setFramebufferScaleFactor === 'function') {
      renderer.xr.setFramebufferScaleFactor(_xrFramebufferScale());
    }
    if (typeof renderer.getClearAlpha === 'function') {
      xrState.savedClearAlpha = renderer.getClearAlpha();
    }
    if (isMixedReality && typeof renderer.setClearAlpha === 'function') {
      renderer.setClearAlpha(0);
    }

    // Reference space — set before requestSession.
    renderer.xr.setReferenceSpaceType('local-floor');

    const session = await sessionPromise;
    xrState.referenceSpaceType = sessionRequest?.referenceSpaceType || 'local-floor';
    renderer.xr.setReferenceSpaceType(xrState.referenceSpaceType);
    const domOverlayType = session.domOverlayState?.type || '';
    setXrWebEmbedPresenting(true, domOverlayType);
    _setupXrCapabilityTracking(
      session,
      renderer,
      sessionRequest?.optionalFeatures || _xrSafeOptionalFeatures(xrState.xrMode),
      domOverlayType,
    );

    // Hook session-end so we restore even if the user takes off the
    // headset or the session ends externally (Quest system menu, etc).
    xrState.sessionEndHandler = () => {
      session.removeEventListener('end', xrState.sessionEndHandler);
      const durationMs = xrState.sessionStartedAt
        ? Math.max(0, Math.round(performance.now() - xrState.sessionStartedAt))
        : 0;
      xrSession.recordEvent(xrState.serverSessionId, 'session_end', { duration_ms: durationMs });
      xrSession.patchSession(xrState.serverSessionId, {
        status: 'ended',
        last_snapshot: {
          ended_at: new Date().toISOString(),
          duration_ms: durationMs,
          seat: {
            x: xrState.xrRig?.position?.x,
            y: xrState.xrRig?.position?.y,
            z: xrState.xrRig?.position?.z,
            rotY: xrState.xrRig?.rotation?.y,
          },
          active_surface: xrState.modeHubActiveAction,
        },
      });
      _restoreNonXRState();
      xrState.active = false;
    };
    session.addEventListener('end', xrState.sessionEndHandler);

    // The desktop avatar loop can still run while the browser is waiting
    // for the headset session to finish starting. Re-zero the camera at
    // the handoff so WebXR, not the portrait auto-framer, owns headset pose.
    camera.position.set(0, 0, 0);
    camera.quaternion.set(0, 0, 0, 1);
    camera.updateMatrixWorld?.(true);

    await renderer.xr.setSession(session);

    camera.position.set(0, 0, 0);
    camera.quaternion.set(0, 0, 0, 1);
    camera.updateMatrixWorld?.(true);

    if (isMixedReality) {
      _setupMixedRealityPlacementAwareness(THREE, session);
    }
    appAvatarState.xrFrameHandler = _handleXrFrame;

    // Spatial audio — best-effort. If anything in the audio re-routing
    // fails, we leave the existing flat-audio graph untouched and
    // continue with VR. Audio path is the highest-risk surface to
    // mutate, so we'd rather lose 3D positioning than lose voice.
    _setupSpatialAudio(THREE, camera, vrm);

    xrState.active = true;
    _recordVrmVisibilitySnapshot('session_started');
    try {
      window.dispatchEvent(new CustomEvent('augmentum:xr-session-state', {
        detail: {
          active: true,
          mode: xrState.xrMode,
          sessionMode: xrState.sessionMode,
          sessionId: xrState.serverSessionId,
        },
      }));
    } catch { /* event listener errors are non-fatal — session continues */ }
    await xrSession.patchSession(xrState.serverSessionId, { status: 'running' });
    xrSession.recordEvent(xrState.serverSessionId, 'session_started', {
      session_mode: sessionMode,
      presentation: xrState.xrMode,
      reference_space: xrState.referenceSpaceType,
      optional_features: sessionRequest?.optionalFeatures || [],
      required_features: sessionRequest?.requiredFeatures || [],
      framebuffer_scale: _xrFramebufferScale(),
      foveation_hint: _xrFixedFoveationFeature() || 'off',
      dom_overlay: domOverlayType || 'unavailable',
      environment_blend_mode: session.environmentBlendMode || '',
      interaction_mode: session.interactionMode || '',
      capabilities: xrState.xrCapabilities,
    });
    return true;
  } catch (err) {
    console.error('[avatar-xr] enterVR failed', err);
    if (requestedSession && !xrState.active) {
      try { await requestedSession.end(); } catch { /* may already be ending */ }
    } else if (sessionPromise) {
      sessionPromise
        .then((session) => session.end().catch(() => {}))
        .catch(() => {});
    }
    xrSession.recordEvent(xrState.serverSessionId, 'session_failed', {
      code: err?.code || err?.name || 'enter-failed',
      message: err?.message || String(err),
    });
    xrSession.patchSession(xrState.serverSessionId, {
      status: 'failed',
      last_snapshot: {
        failed_at: new Date().toISOString(),
        message: err?.message || String(err),
      },
    });
    onError?.(err);
    _restoreNonXRState();
    return false;
  }
}

// ── Exit VR ──────────────────────────────────────────────────────────
export function enterMR(options = {}) {
  return enterVR({
    ...options,
    sessionMode: 'immersive-ar',
    xrMode: XR_MODE_MR,
  });
}

export async function exitVR() {
  if (!xrState.active) return;
  const session = xrState.renderer?.xr?.getSession();
  if (session) {
    try { await session.end(); } catch { /* may already be ending */ }
    // sessionEndHandler does the actual restore
    return;
  }
  // No session — restore manually (defensive)
  _restoreNonXRState();
  xrState.active = false;
}

// ── Internals ────────────────────────────────────────────────────────
function _restoreNonXRState() {
  const endedMode = xrState.xrMode;
  const endedSessionMode = xrState.sessionMode;
  const endedSessionId = xrState.serverSessionId;
  // Tear down contact reactor: clears any in-flight expression spikes and
  // detaches the controller/hand listeners so we don't leak across sessions.
  try { teardownXRContact(); }
  catch (err) { console.debug('[xr] contact teardown failed:', err?.message); }
  try { teardownXRCompliance(); }
  catch (err) { console.debug('[xr] compliance teardown failed:', err?.message); }
  // teardownXRRapier is async (awaits a 500ms-bounded init join); fire-and-
  // forget here so session cleanup isn't blocked by a slow Rapier import.
  try { void teardownXRRapier(); }
  catch (err) { console.debug('[xr] rapier teardown failed:', err?.message); }
  try { teardownAvatarAudioReactions(); }
  catch (err) { console.debug('[xr] audio reactions teardown failed:', err?.message); }
  try { teardownBodyPhysicsCoordinator(); }
  catch (err) { console.debug('[xr] coordinator teardown failed:', err?.message); }
  try { teardownBodyPhysicsHUD(); }
  catch (err) { console.debug('[xr] hud teardown failed:', err?.message); }
  _contactLastTickMs = 0;
  if (appAvatarState.xrFrameHandler === _handleXrFrame) {
    appAvatarState.xrFrameHandler = null;
  }
  try {
    window.dispatchEvent(new CustomEvent('augmentum:xr-session-state', {
      detail: {
        active: false,
        mode: endedMode,
        sessionMode: endedSessionMode,
        sessionId: endedSessionId,
      },
    }));
  } catch { /* event listener errors are non-fatal — session is already ending */ }
  try { xrState.xrWorkspace?.dispose?.(); } catch {}
  try { xrState.xrSurfaceData?.stop?.(); } catch {}
  const activeSession = xrState.renderer?.xr?.getSession?.();
  if (activeSession && xrState.capabilityListener) {
    try { activeSession.removeEventListener('inputsourceschange', xrState.capabilityListener); } catch {}
  }
  hideXrWebEmbed();
  setXrWebEmbedAnchor({ visible: false });
  setXrWebEmbedCapabilities({});
  setXrWebEmbedPresenting(false);

  // Detach the per-frame sit-pose lock so the animator regains control
  // on desktop. Must happen before restoring VRM transform — otherwise
  // the lock could re-fire one more time and clobber the restoration.
  if (xrState.vrm?.scene) {
    xrState.vrm.scene.onBeforeRender =
      xrState.savedVrmOnBeforeRender !== undefined
        ? xrState.savedVrmOnBeforeRender
        : () => {};
  }

  // Restore VRM eye-gaze target (typically null on solo desktop; was
  // hijacked to point at the XR camera on enter). Done before the rig
  // teardown so lookAt isn't pointing at a soon-to-be-detached camera.
  if (xrState.vrm?.lookAt && xrState.savedLookAtTarget !== undefined) {
    xrState.vrm.lookAt.target = xrState.savedLookAtTarget;
  }

  // Tear down spatial audio routing — restore voice.js's original
  // analyserNode → destination edge so non-VR mode is bit-identical
  // to today. Best-effort — voice.js may have already cleaned up if
  // the call ended; in that case the disconnect/connect are no-ops.
  _teardownSpatialAudio();
  _disposeMixedRealityPlacement();
  _disposeBoundaryVisual();
  if (xrState.mrShadow) {
    xrState.mrShadow.parent?.remove(xrState.mrShadow);
    xrState.mrShadow.geometry?.dispose();
    xrState.mrShadow.material?.dispose();
    xrState.mrShadow = null;
  }

  // Restore VRM transform
  if (xrState.vrm && xrState.savedVrmPosition && xrState.savedVrmRotationY != null) {
    xrState.vrm.scene.visible = xrState.savedVrmVisible !== false;
    xrState.vrm.scene.position.copy(xrState.savedVrmPosition);
    xrState.vrm.scene.rotation.y = xrState.savedVrmRotationY;
  }

  // Restore camera parent + local transform
  if (xrState.camera) {
    if (xrState.savedCameraParent) {
      xrState.savedCameraParent.add(xrState.camera);
    } else if (xrState.scene) {
      xrState.scene.add(xrState.camera);
    }
    if (xrState.savedCameraPosition) {
      xrState.camera.position.copy(xrState.savedCameraPosition);
    }
    if (xrState.savedCameraQuaternion) {
      xrState.camera.quaternion.copy(xrState.savedCameraQuaternion);
    }
  }

  // Tear down HUD plane (dispose its canvas-backed texture + geometry).
  if (xrState.hudPlane) {
    xrState.hudPlane.parent?.remove(xrState.hudPlane);
    xrState.hudPlane.geometry?.dispose();
    xrState.hudPlane.material?.map?.dispose();
    xrState.hudPlane.material?.dispose();
  }
  _disposeSpatialPanels();
  _disposeTargetRayVisuals();
  _disposeControllerlessComfortUI();

  // Detach all controller event listeners (selectstart / squeezestart /
  // future ones). Controllers themselves are freed when the rig is
  // removed below.
  for (const { ctrl, ev, fn } of xrState.controllerListeners) {
    ctrl.removeEventListener(ev, fn);
  }

  // Tear down rig
  if (xrState.xrRig && xrState.scene) {
    xrState.scene.remove(xrState.xrRig);
  }

  // Tear down room (disposes GLB references but the GLB file stays
  // browser-cached, so re-entering VR is fast).
  if (xrState.roomScene) {
    xrState.roomScene.dispose();
  }

  // Restore scene background
  if (xrState.scene && xrState.savedSceneBackground !== undefined) {
    xrState.scene.background = xrState.savedSceneBackground;
  }

  // Restore GPU pressure-mitigation knobs.
  if (xrState.renderer) {
    if (xrState.savedPixelRatio != null) {
      xrState.renderer.setPixelRatio(xrState.savedPixelRatio);
    }
    if (
      xrState.savedFramebufferScale != null
      && typeof xrState.renderer.xr?.setFramebufferScaleFactor === 'function'
    ) {
      xrState.renderer.xr.setFramebufferScaleFactor(xrState.savedFramebufferScale);
    }
    if (
      xrState.savedClearAlpha != null
      && typeof xrState.renderer.setClearAlpha === 'function'
    ) {
      xrState.renderer.setClearAlpha(xrState.savedClearAlpha);
    }
  }

  // Reset all internal references
  xrState.THREE = null;
  xrState.scene = null;
  xrState.renderer = null;
  xrState.camera = null;
  xrState.vrm = null;
  xrState.savedCameraParent = null;
  xrState.savedCameraPosition = null;
  xrState.savedCameraQuaternion = null;
  xrState.savedSceneBackground = null;
  xrState.savedVrmPosition = null;
  xrState.savedVrmRotationY = null;
  xrState.savedVrmVisible = true;
  xrState.savedLookAtTarget = undefined;
  xrState.savedVrmOnBeforeRender = undefined;
  xrState.roomScene = null;
  xrState.xrRig = null;
  xrState.controllers = [];
  xrState.targetRayControllers = [];
  xrState.hands = [];
  xrState.handStates = new Map();
  xrState.controllerListeners = [];
  xrState.sessionEndHandler = null;
  xrState.hudCanvas = null;
  xrState.hudCtx = null;
  xrState.hudTexture = null;
  xrState.hudPlane = null;
  xrState.hudLastLabel = '';
  if (xrState.modeHubGroup?.parent) {
    xrState.modeHubGroup.parent.remove(xrState.modeHubGroup);
  }
  if (xrState.modeHubPanel) {
    xrState.modeHubPanel.geometry?.dispose();
    xrState.modeHubPanel.material?.map?.dispose();
    xrState.modeHubPanel.material?.dispose();
  }
  xrState.modeHubGroup = null;
  xrState.modeHubPanel = null;
  xrState.modeHubCanvas = null;
  xrState.modeHubCtx = null;
  xrState.modeHubTexture = null;
  xrState.modeHubButtons = [];
  xrState.modeHubSurfaces = [];
  xrState.modeHubActiveAction = 'voice';
  xrState.modeHubStatus = 'Voice call is live. Select a surface to bring it into focus.';
  xrState.modeHubLastKey = '';
  xrState.hubDrag = null;
  xrState.hubUserPlaced = false;
  xrState.xrWorkspace = null;
  xrState.xrSurfaceData = null;
  xrState.spatialPanels = new Map();
  xrState.lastLiveSurfaceRefreshAt = 0;
  xrState.panelDrag = null;
  xrState.panelDragWorldPoint = null;
  xrState.panelCameraWorld = null;
  xrState.handTempA = null;
  xrState.handTempB = null;
  xrState.handTempC = null;
  xrState.handTempD = null;
  xrState.handTempQuat = null;
  xrState.raycaster = null;
  xrState.rayOrigin = null;
  xrState.rayDirection = null;
  xrState.rayMatrix = null;
  xrState.gazeCursor = null;
  xrState.gazeCursorMat = null;
  xrState.gazeHit = null;
  xrState.gazeTargetKey = '';
  xrState.gazeTargetStartedAt = 0;
  xrState.gazeDwellArmed = false;
  xrState.lastGazeDwellAt = 0;
  xrState.lastNativeSelectAt = 0;
  xrState.lastComfortSelectAt = 0;
  xrState.handMenuGroup = null;
  xrState.handMenuPanel = null;
  xrState.handMenuCanvas = null;
  xrState.handMenuCtx = null;
  xrState.handMenuTexture = null;
  xrState.handMenuButtons = [];
  xrState.handMenuLastKey = '';
  xrState.handMenuHandIndex = null;
  xrState.handMenuShownAt = 0;
  xrState.handMenuHiddenAt = 0;
  xrState.handDebugGroup = null;
  xrState.handDebugPanel = null;
  xrState.handDebugCanvas = null;
  xrState.handDebugCtx = null;
  xrState.handDebugTexture = null;
  xrState.handDebugVisible = false;
  xrState.handDebugLastKey = '';
  xrState.handDebugLastDrawAt = 0;
  xrState.handProbeEvents = [];
  xrState.handProbeInputs = new Map();
  xrState.nativeHandIntents = new Map();
  xrState.xrFrameStats = {
    lastTime: 0,
    averageMs: 0,
    maxMs: 0,
    longFrames: 0,
    lastLongFrameAt: 0,
  };
  xrState.userSignals = [];
  xrState.userSignalLastAt = new Map();
  xrState.lastAvatarWaveBackAt = 0;
  xrState.pendingHubConfirm = null;
  xrState.xrCapabilities = {};
  xrState.capabilityListener = null;
  xrState.webEmbedLastAnchorKey = '';
  xrState.savedPixelRatio = null;
  xrState.savedFramebufferScale = null;
  xrState.savedClearAlpha = null;
  xrState.audioListener = null;
  xrState.positionalAudio = null;
  xrState.audioContext = null;
  xrState.audioRerouted = false;
  xrState.mrShadow = null;
  xrState.mrPlacement = null;
  xrState.boundaryLine = null;
  xrState.avatarXrPose = null;
  xrState.xrMode = XR_MODE_VR;
  xrState.sessionMode = 'immersive-vr';
  xrState.referenceSpaceType = 'local-floor';
  xrState.serverSession = null;
  xrState.serverSessionId = null;
  xrState.sessionStartedAt = 0;
}

function _detectXrCapabilities(session, renderer, optionalFeatures = [], domOverlayType = '') {
  const sources = Array.from(session?.inputSources || []);
  const enabledFeatures = Array.isArray(session?.enabledFeatures)
    ? Array.from(session.enabledFeatures)
    : [];
  const gl = renderer?.getContext?.();
  let layers = false;
  if (typeof window.XRWebGLBinding === 'function' && gl) {
    try {
      const binding = new window.XRWebGLBinding(session, gl);
      layers = !!(
        binding
        && (
          typeof binding.createQuadLayer === 'function'
          || typeof binding.createCylinderLayer === 'function'
          || typeof binding.createProjectionLayer === 'function'
        )
      );
    } catch {
      layers = false;
    }
  }
  return {
    sessionMode: xrState.sessionMode,
    presentation: xrState.xrMode,
    requested: Array.from(optionalFeatures),
    enabledFeatures,
    domOverlay: !!session?.domOverlayState,
    domOverlayType: domOverlayType || session?.domOverlayState?.type || '',
    environmentBlendMode: session?.environmentBlendMode || '',
    interactionMode: session?.interactionMode || '',
    handTracking: sources.some((source) => !!source.hand),
    handInputSources: sources.filter((source) => !!source.hand).length,
    trackedPointers: sources.filter((source) => source.targetRayMode === 'tracked-pointer').length,
    layers,
    mediaLayers: layers && typeof window.XRMediaBinding === 'function',
    anchors: enabledFeatures.includes('anchors'),
    hitTest: enabledFeatures.includes('hit-test'),
    planeDetection: enabledFeatures.includes('plane-detection'),
    meshDetection: enabledFeatures.includes('mesh-detection'),
    depthSensing: enabledFeatures.includes('depth-sensing'),
    secondaryViews: enabledFeatures.includes('secondary-views'),
    boundedFloor: false,
    foveationHint: optionalFeatures.find((feature) => String(feature || '').includes('fixed-foveation-level')) || '',
  };
}

function _publishXrCapabilities(capabilities, eventName = 'capabilities_detected') {
  xrState.xrCapabilities = { ...capabilities };
  setXrWebEmbedCapabilities(xrState.xrCapabilities);
  xrState.xrWorkspace?.updateCapabilities?.(xrState.xrCapabilities, { source: eventName });
  xrSession.recordEvent(xrState.serverSessionId, eventName, xrState.xrCapabilities);
}

function _setupXrCapabilityTracking(session, renderer, optionalFeatures, domOverlayType) {
  const capabilities = _detectXrCapabilities(session, renderer, optionalFeatures, domOverlayType);
  _publishXrCapabilities(capabilities);

  xrState.capabilityListener = () => {
    const next = {
      ..._detectXrCapabilities(session, renderer, optionalFeatures, domOverlayType),
      boundedFloor: xrState.xrCapabilities?.boundedFloor || false,
    };
    _publishXrCapabilities(next, 'capabilities_changed');
  };
  try {
    session.addEventListener('inputsourceschange', xrState.capabilityListener);
  } catch { /* some XR runtimes don't expose inputsourceschange — skip silently */ }

  if (typeof session?.requestReferenceSpace === 'function') {
    session.requestReferenceSpace('bounded-floor')
      .then((space) => {
        _setupBoundaryVisual(space);
        const next = {
          ...xrState.xrCapabilities,
          boundedFloor: true,
          boundsGeometryPoints: Array.isArray(space?.boundsGeometry)
            ? space.boundsGeometry.length
            : 0,
        };
        _publishXrCapabilities(next, 'bounded_floor_available');
      })
      .catch(() => {
        const next = { ...xrState.xrCapabilities, boundedFloor: false };
        _publishXrCapabilities(next, 'bounded_floor_unavailable');
        if (xrState.xrMode === XR_MODE_MR) _setupFallbackBoundaryVisual();
      });
  }
}

function _applyMixedRealityCompanionPose(vrm) {
  if (!vrm?.scene) return;
  if (vrm.humanoid?.resetNormalizedPose) vrm.humanoid.resetNormalizedPose();
  vrm.scene.position.set(0.58, 0, -1.55);
  vrm.scene.lookAt(0, 1.15, 0);
  vrm.scene.rotation.x = 0;
  vrm.scene.rotation.z = 0;
}

function _prepareVrmSceneForXR(vrm) {
  if (!vrm?.scene) return;
  vrm.scene.visible = true;
  vrm.scene.traverse?.((obj) => {
    if (obj.isMesh || obj.isSkinnedMesh) obj.frustumCulled = false;
  });
  vrm.scene.updateMatrixWorld?.(true);
}

function _recordVrmVisibilitySnapshot(label = 'snapshot') {
  const THREE = xrState.THREE;
  const scene = xrState.vrm?.scene;
  if (!THREE || !scene || !xrState.camera) return;
  try {
    scene.updateMatrixWorld?.(true);
    xrState.camera.updateMatrixWorld?.(true);
    const box = new THREE.Box3().setFromObject(scene);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    const cameraWorld = new THREE.Vector3();
    const forward = new THREE.Vector3();
    const toAvatar = new THREE.Vector3();
    xrState.camera.getWorldPosition(cameraWorld);
    xrState.camera.getWorldDirection(forward);
    toAvatar.copy(center).sub(cameraWorld);
    const distance = toAvatar.length();
    const angleDeg = distance > 0.0001
      ? Math.acos(Math.max(-1, Math.min(1, forward.dot(toAvatar.normalize())))) * 180 / Math.PI
      : 0;
    xrSession.recordEvent(xrState.serverSessionId, 'vrm_visibility_snapshot', {
      label,
      visible: scene.visible !== false,
      parent: scene.parent?.name || scene.parent?.type || '',
      position: {
        x: _roundPoseValue(scene.position.x),
        y: _roundPoseValue(scene.position.y),
        z: _roundPoseValue(scene.position.z),
      },
      distance: _roundPoseValue(distance),
      angle_deg: Math.round(angleDeg),
      bounds: {
        x: _roundPoseValue(size.x),
        y: _roundPoseValue(size.y),
        z: _roundPoseValue(size.z),
      },
    });
  } catch (err) {
    console.debug('[avatar-xr] VRM visibility snapshot failed', err);
  }
}

function _setupMixedRealityGrounding(THREE, scene, vrm) {
  if (!THREE || !scene || !vrm?.scene) return;
  const geometry = new THREE.CircleGeometry(0.46, 48);
  const material = new THREE.MeshBasicMaterial({
    color: 0x000000,
    transparent: true,
    opacity: 0.22,
    depthWrite: false,
  });
  const shadow = new THREE.Mesh(geometry, material);
  shadow.name = 'AugmentumMRCompanionShadow';
  shadow.rotation.x = -Math.PI / 2;
  shadow.renderOrder = -10;
  shadow.position.set(vrm.scene.position.x, 0.012, vrm.scene.position.z);
  scene.add(shadow);
  xrState.mrShadow = shadow;
}

function _setupMixedRealityPlacementAwareness(THREE, session) {
  if (!THREE || !session || xrState.xrMode !== XR_MODE_MR || !xrState.scene || !xrState.vrm?.scene) return;
  _disposeMixedRealityPlacement();

  const reticleGeometry = new THREE.RingGeometry(0.18, 0.225, 48);
  const reticleMaterial = new THREE.MeshBasicMaterial({
    color: 0x78ffd6,
    transparent: true,
    opacity: 0.82,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const reticle = new THREE.Mesh(reticleGeometry, reticleMaterial);
  reticle.name = 'AugmentumMRPlacementReticle';
  reticle.rotation.x = -Math.PI / 2;
  reticle.renderOrder = 6;
  reticle.frustumCulled = false;
  reticle.visible = false;
  xrState.scene.add(reticle);

  const placement = {
    THREE,
    session,
    reticle,
    referenceSpace: null,
    viewerSpace: null,
    hitTestSource: null,
    startedAt: performance.now(),
    avatarPlaced: false,
    placedAt: 0,
    manualMode: true,
    userPlaced: false,
    candidateValid: false,
    candidateSource: '',
    manualRayActiveAt: 0,
    anchor: null,
    anchorPending: false,
    anchorSupported: false,
    anchorFailed: false,
    tempPosition: new THREE.Vector3(),
    candidatePosition: new THREE.Vector3(),
    rayPoint: new THREE.Vector3(),
    tempCamera: new THREE.Vector3(),
    tempForward: new THREE.Vector3(),
    tempMatrix: new THREE.Matrix4(),
    tempQuaternion: new THREE.Quaternion(),
    tempScale: new THREE.Vector3(1, 1, 1),
    seenPlaneCount: null,
    seenMeshCount: null,
    lastAwarenessAt: 0,
    lastHitAt: 0,
    lastHitErrorAt: 0,
  };
  xrState.mrPlacement = placement;
  _enterMixedRealityAvatarPlacementMode('session-start', { hideAvatar: true });

  _requestMixedRealityReferenceSpace(session, xrState.referenceSpaceType || 'local-floor')
    .then((space) => {
      if (!_isCurrentMixedRealityPlacement(placement)) return;
      placement.referenceSpace = space;
    })
    .catch((err) => {
      if (!_isCurrentMixedRealityPlacement(placement)) return;
      xrSession.recordEvent(xrState.serverSessionId, 'mr_reference_space_unavailable', {
        code: err?.name || err?.code || 'reference-space-error',
        message: err?.message || String(err),
      });
    });

  session.requestReferenceSpace('viewer')
    .then((viewerSpace) => {
      if (!_isCurrentMixedRealityPlacement(placement)) return null;
      placement.viewerSpace = viewerSpace;
      return _requestMixedRealityHitTestSource(session, viewerSpace);
    })
    .then((source) => {
      if (!source) return;
      if (!_isCurrentMixedRealityPlacement(placement)) {
        try { source.cancel?.(); } catch {}
        return;
      }
      placement.hitTestSource = source;
      xrSession.recordEvent(xrState.serverSessionId, 'mr_placement_ready', {
        reference_space: xrState.referenceSpaceType || 'local-floor',
        hit_test: true,
      });
    })
    .catch((err) => {
      if (!_isCurrentMixedRealityPlacement(placement)) return;
      xrSession.recordEvent(xrState.serverSessionId, 'mr_hit_test_unavailable', {
        code: err?.name || err?.code || 'hit-test-error',
        message: err?.message || String(err),
      });
    });
}

async function _requestMixedRealityReferenceSpace(session, preferredType) {
  try {
    return await session.requestReferenceSpace(preferredType);
  } catch (err) {
    if (preferredType !== 'local') {
      return session.requestReferenceSpace('local');
    }
    throw err;
  }
}

async function _requestMixedRealityHitTestSource(session, viewerSpace) {
  if (typeof session?.requestHitTestSource !== 'function') {
    throw _xrError('hit-test-api-unavailable', 'XR hit-test source is unavailable');
  }
  const offsetRay = _createMixedRealityPlacementRay();
  const attempts = [];
  if (offsetRay) attempts.push({ space: viewerSpace, offsetRay, entityTypes: ['plane', 'mesh', 'point'] });
  attempts.push({ space: viewerSpace, entityTypes: ['plane', 'mesh', 'point'] });
  if (offsetRay) attempts.push({ space: viewerSpace, offsetRay });
  attempts.push({ space: viewerSpace });

  let lastErr = null;
  for (const options of attempts) {
    try {
      return await session.requestHitTestSource(options);
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || _xrError('hit-test-source-failed', 'Could not create XR hit-test source');
}

function _createMixedRealityPlacementRay() {
  try {
    const XRRayCtor = globalThis.XRRay;
    if (typeof XRRayCtor !== 'function') return null;
    return new XRRayCtor(
      { x: 0, y: -0.08, z: 0, w: 1 },
      { x: 0, y: -0.35, z: -1, w: 0 },
    );
  } catch {
    return null;
  }
}

function _isCurrentMixedRealityPlacement(placement) {
  return !!placement && xrState.mrPlacement === placement;
}

function _disposeMixedRealityPlacement() {
  const placement = xrState.mrPlacement;
  if (!placement) return;
  try { placement.hitTestSource?.cancel?.(); } catch {}
  _clearMixedRealityAnchor(placement);
  placement.reticle?.parent?.remove(placement.reticle);
  placement.reticle?.geometry?.dispose();
  placement.reticle?.material?.dispose();
  xrState.mrPlacement = null;
}

function _clearMixedRealityAnchor(placement) {
  if (!placement) return;
  try { placement.anchor?.delete?.(); } catch {}
  placement.anchor = null;
  placement.anchorPending = false;
}

function _setMixedRealityAvatarVisible(visible) {
  if (xrState.vrm?.scene) {
    xrState.vrm.scene.visible = visible;
  }
  if (xrState.mrShadow) {
    xrState.mrShadow.visible = visible;
  }
}

function _setMixedRealityPlacementStatus(message) {
  xrState.modeHubStatus = message;
  xrState.modeHubLastKey = '';
  _refreshModeHub(true);
}

function _enterMixedRealityAvatarPlacementMode(source = 'manual', { hideAvatar = false } = {}) {
  const placement = xrState.mrPlacement;
  if (!placement || xrState.xrMode !== XR_MODE_MR) return false;
  _clearMixedRealityAnchor(placement);
  placement.manualMode = true;
  placement.candidateValid = false;
  placement.candidateSource = '';
  placement.manualRayActiveAt = 0;
  placement.startedAt = performance.now();
  placement.anchorFailed = false;
  if (placement.reticle) placement.reticle.visible = false;
  if (hideAvatar) _setMixedRealityAvatarVisible(false);
  _setMixedRealityPlacementStatus('Place avatar: point at the floor, then trigger, pinch, or dwell.');
  _refreshHandMenu(true);
  xrSession.recordEvent(xrState.serverSessionId, 'mr_avatar_placement_mode', { source });
  return true;
}

function _restartMixedRealityPlacement(source = 'recenter') {
  let placement = xrState.mrPlacement;
  if (!placement) {
    const session = xrState.renderer?.xr?.getSession?.();
    if (xrState.THREE && session) _setupMixedRealityPlacementAwareness(xrState.THREE, session);
    placement = xrState.mrPlacement;
  }
  if (!placement) return;
  _clearMixedRealityAnchor(placement);
  placement.startedAt = performance.now();
  placement.avatarPlaced = false;
  placement.userPlaced = false;
  placement.placedAt = 0;
  placement.anchorFailed = false;
  placement.lastHitAt = 0;
  placement.candidateValid = false;
  placement.candidateSource = '';
  if (placement.reticle) placement.reticle.visible = false;
  _enterMixedRealityAvatarPlacementMode(source, { hideAvatar: true });
  xrSession.recordEvent(xrState.serverSessionId, 'mr_placement_restarted', { source });
}

function _recordXrFrameTiming(time) {
  const stats = xrState.xrFrameStats;
  if (!stats) return;
  const now = Number.isFinite(time) ? time : performance.now();
  if (stats.lastTime > 0) {
    const deltaMs = Math.max(0, now - stats.lastTime);
    stats.averageMs = stats.averageMs > 0
      ? stats.averageMs * 0.92 + deltaMs * 0.08
      : deltaMs;
    stats.maxMs = Math.max(deltaMs, stats.maxMs * 0.96);
    if (deltaMs >= 24) {
      stats.longFrames += 1;
      if (now - stats.lastLongFrameAt > 3500) {
        stats.lastLongFrameAt = now;
        xrSession.recordEvent(xrState.serverSessionId, 'xr_render_long_frame', {
          delta_ms: Math.round(deltaMs),
          average_ms: Math.round(stats.averageMs),
          max_ms: Math.round(stats.maxMs),
          long_frames: stats.longFrames,
        });
      }
    }
  }
  stats.lastTime = now;
}

function _handleXrFrame(time, frame) {
  if (!xrState.active) return;
  const now = Number.isFinite(time) ? time : performance.now();
  _recordXrFrameTiming(now);
  if (xrState.xrMode === XR_MODE_MR) {
    _updateMixedRealityPlacementFromFrame(now, frame);
  }
  _updateXrInteractions();
}

function _updateMixedRealityPlacementFromFrame(now, frame) {
  const placement = xrState.mrPlacement;
  if (!placement || !frame || !placement.referenceSpace) return;
  _updateMixedRealitySceneAwareness(placement, frame, now);
  if (placement.manualMode && placement.anchor) _clearMixedRealityAnchor(placement);
  if (_updateMixedRealityAnchorPose(placement, frame)) return;
  if (!placement.hitTestSource || typeof frame.getHitTestResults !== 'function') return;

  let hits = null;
  try {
    hits = frame.getHitTestResults(placement.hitTestSource);
  } catch (err) {
    if (now - placement.lastHitErrorAt > 2500) {
      placement.lastHitErrorAt = now;
      xrSession.recordEvent(xrState.serverSessionId, 'mr_hit_test_frame_failed', {
        code: err?.name || err?.code || 'hit-test-frame-error',
      });
    }
    return;
  }

  if (!hits?.length) {
    if (placement.reticle && (!placement.manualMode || now - placement.manualRayActiveAt > 350)) {
      placement.reticle.visible = false;
    }
    return;
  }

  const hit = hits[0];
  const pose = hit.getPose?.(placement.referenceSpace);
  if (!pose || !_positionFromXrPose(placement, pose, placement.tempPosition)) return;
  const canAnchorHere = _isMixedRealityAnchorCandidate(placement, placement.tempPosition);
  if (!_sanitizeMixedRealityPlacementPoint(placement, placement.tempPosition)) return;
  placement.lastHitAt = now;

  if (placement.manualMode) {
    if (now - placement.manualRayActiveAt > 350) {
      _setMixedRealityPlacementCandidate(placement, placement.tempPosition, 'headset-hit-test');
    }
    return;
  }

  if (placement.userPlaced) {
    if (placement.reticle) placement.reticle.visible = false;
    return;
  }

  if (placement.reticle) {
    placement.reticle.position.copy(placement.tempPosition);
    placement.reticle.position.y += MR_PLACEMENT_RETICLE_Y_OFFSET_M;
    placement.reticle.visible = !placement.avatarPlaced || now - placement.startedAt <= MR_PLACEMENT_SETTLE_MS;
  }

  if (!placement.avatarPlaced || now - placement.startedAt <= MR_PLACEMENT_SETTLE_MS) {
    const firstPlacement = !placement.avatarPlaced;
    _applyMixedRealityAvatarPosition(placement.tempPosition, placement.tempCamera);
    placement.avatarPlaced = true;
    placement.placedAt = placement.placedAt || now;
    if (firstPlacement) {
      xrSession.recordEvent(xrState.serverSessionId, 'mr_avatar_placed', {
        source: 'hit-test',
        x: _roundPoseValue(placement.tempPosition.x),
        y: _roundPoseValue(placement.tempPosition.y),
        z: _roundPoseValue(placement.tempPosition.z),
      });
      _recordVrmVisibilitySnapshot('mr_hit_test_placed');
    }
  }

  if (canAnchorHere) _tryCreateMixedRealityAnchor(placement, hit);
}

function _updateMixedRealitySceneAwareness(placement, frame, now) {
  try {
    const planeCount = typeof frame.detectedPlanes?.size === 'number' ? frame.detectedPlanes.size : null;
    const meshCount = typeof frame.detectedMeshes?.size === 'number' ? frame.detectedMeshes.size : null;
    if (planeCount == null && meshCount == null) return;
    if (planeCount === placement.seenPlaneCount && meshCount === placement.seenMeshCount) return;
    if (now - placement.lastAwarenessAt < 2000) return;
    placement.seenPlaneCount = planeCount;
    placement.seenMeshCount = meshCount;
    placement.lastAwarenessAt = now;
    xrSession.recordEvent(xrState.serverSessionId, 'mr_scene_awareness', {
      planes: planeCount,
      meshes: meshCount,
    });
  } catch {
    // Some runtimes gate experimental scene-understanding fields per-frame.
  }
}

function _updateMixedRealityAnchorPose(placement, frame) {
  const anchor = placement.anchor;
  if (!anchor) return false;
  try {
    if (frame.trackedAnchors && !frame.trackedAnchors.has(anchor)) {
      _clearMixedRealityAnchor(placement);
      return false;
    }
    const pose = frame.getPose?.(anchor.anchorSpace, placement.referenceSpace);
    if (!pose || !_positionFromXrPose(placement, pose, placement.tempPosition)) return false;
    if (!_sanitizeMixedRealityPlacementPoint(placement, placement.tempPosition, { clampDistance: false })) return false;
    _applyMixedRealityAvatarPosition(placement.tempPosition, placement.tempCamera);
    if (placement.reticle) placement.reticle.visible = false;
    return true;
  } catch {
    return false;
  }
}

function _tryCreateMixedRealityAnchor(placement, hit) {
  if (
    placement.anchor
    || placement.anchorPending
    || placement.anchorFailed
    || typeof hit?.createAnchor !== 'function'
  ) {
    return;
  }
  placement.anchorPending = true;
  hit.createAnchor()
    .then((anchor) => {
      if (!_isCurrentMixedRealityPlacement(placement)) {
        try { anchor?.delete?.(); } catch {}
        return;
      }
      placement.anchor = anchor;
      placement.anchorPending = false;
      placement.anchorSupported = true;
      xrSession.recordEvent(xrState.serverSessionId, 'mr_anchor_created', {});
    })
    .catch((err) => {
      if (!_isCurrentMixedRealityPlacement(placement)) return;
      placement.anchorPending = false;
      placement.anchorFailed = true;
      xrSession.recordEvent(xrState.serverSessionId, 'mr_anchor_unavailable', {
        code: err?.name || err?.code || 'anchor-error',
      });
    });
}

function _positionFromXrPose(placement, pose, out) {
  const transform = pose?.transform;
  if (!transform || !out) return false;
  if (transform.matrix) {
    placement.tempMatrix.fromArray(transform.matrix);
    placement.tempMatrix.decompose(out, placement.tempQuaternion, placement.tempScale);
    return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
  }
  const pos = transform.position;
  if (!pos) return false;
  out.set(Number(pos.x), Number(pos.y), Number(pos.z));
  return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
}

function _isMixedRealityAnchorCandidate(placement, point) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
    return false;
  }
  const floorReference = xrState.referenceSpaceType === 'local-floor' || xrState.referenceSpaceType === 'bounded-floor';
  if (floorReference && (point.y < -0.2 || point.y > 0.35)) return false;
  if (xrState.camera) {
    xrState.camera.updateMatrixWorld?.(true);
    xrState.camera.getWorldPosition(placement.tempCamera);
    const distance = Math.hypot(point.x - placement.tempCamera.x, point.z - placement.tempCamera.z);
    return distance >= MR_PLACEMENT_MIN_DISTANCE_M && distance <= MR_PLACEMENT_MAX_DISTANCE_M;
  }
  return true;
}

function _sanitizeMixedRealityPlacementPoint(placement, point, { clampDistance = true } = {}) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) {
    return false;
  }

  const floorReference = xrState.referenceSpaceType === 'local-floor' || xrState.referenceSpaceType === 'bounded-floor';
  if (floorReference) {
    if (point.y < -0.35 || point.y > 1.05) return false;
    point.y = 0;
  } else if (point.y < -2.4 || point.y > 0.75) {
    return false;
  }

  if (clampDistance && xrState.camera) {
    xrState.camera.updateMatrixWorld?.(true);
    xrState.camera.getWorldPosition(placement.tempCamera);
    const dx = point.x - placement.tempCamera.x;
    const dz = point.z - placement.tempCamera.z;
    const distance = Math.hypot(dx, dz);
    if (distance < 0.001) {
      xrState.camera.getWorldDirection(placement.tempForward);
      placement.tempForward.y = 0;
      if (placement.tempForward.lengthSq() < 0.0001) placement.tempForward.set(0, 0, -1);
      placement.tempForward.normalize();
      point.x = placement.tempCamera.x + placement.tempForward.x * 1.2;
      point.z = placement.tempCamera.z + placement.tempForward.z * 1.2;
    } else if (distance < MR_PLACEMENT_MIN_DISTANCE_M || distance > MR_PLACEMENT_MAX_DISTANCE_M) {
      const clampedDistance = Math.max(MR_PLACEMENT_MIN_DISTANCE_M, Math.min(MR_PLACEMENT_MAX_DISTANCE_M, distance));
      const scale = clampedDistance / distance;
      point.x = placement.tempCamera.x + dx * scale;
      point.z = placement.tempCamera.z + dz * scale;
    }
  }

  return true;
}

function _isMixedRealityAvatarPlacementMode() {
  return xrState.xrMode === XR_MODE_MR && !!xrState.mrPlacement?.manualMode;
}

function _mixedRealityPointFromCurrentRay(placement, out) {
  if (!placement || !out || !xrState.rayOrigin || !xrState.rayDirection) return false;
  const floorY = 0;
  const origin = xrState.rayOrigin;
  const direction = xrState.rayDirection;
  let distance = MR_PLACEMENT_DEFAULT_DISTANCE_M;
  if (Math.abs(direction.y) > 0.03) {
    const t = (floorY - origin.y) / direction.y;
    if (Number.isFinite(t) && t > 0.18 && t < 5.0) {
      distance = t;
    }
  }
  distance = Math.max(0.35, Math.min(4.8, distance));
  out.copy(direction).multiplyScalar(distance).add(origin);
  out.y = floorY;
  return _sanitizeMixedRealityPlacementPoint(placement, out);
}

function _setMixedRealityPlacementCandidate(placement, point, source = 'controller-ray') {
  if (!placement || !point) return false;
  placement.candidatePosition.copy(point);
  placement.candidateValid = true;
  placement.candidateSource = source;
  if (source === 'controller-ray' || source === 'gaze') placement.manualRayActiveAt = performance.now();
  if (placement.reticle) {
    placement.reticle.position.copy(point);
    placement.reticle.position.y += MR_PLACEMENT_RETICLE_Y_OFFSET_M;
    placement.reticle.visible = true;
  }
  return true;
}

function _updateMixedRealityPlacementCandidateFromCurrentRay(source = 'controller-ray') {
  const placement = xrState.mrPlacement;
  if (!placement || xrState.xrMode !== XR_MODE_MR) return null;
  if (!_mixedRealityPointFromCurrentRay(placement, placement.rayPoint)) return null;
  _setMixedRealityPlacementCandidate(placement, placement.rayPoint, source);
  return placement.candidatePosition;
}

function _defaultMixedRealityPlacementPoint(placement, out) {
  if (!placement || !out || !xrState.camera) return false;
  xrState.camera.updateMatrixWorld?.(true);
  xrState.camera.getWorldPosition(placement.tempCamera);
  xrState.camera.getWorldDirection(placement.tempForward);
  placement.tempForward.y = 0;
  if (placement.tempForward.lengthSq() < 0.0001) placement.tempForward.set(0, 0, -1);
  placement.tempForward.normalize();
  out.copy(placement.tempCamera).add(placement.tempForward.multiplyScalar(MR_PLACEMENT_DEFAULT_DISTANCE_M));
  out.y = 0;
  return _sanitizeMixedRealityPlacementPoint(placement, out);
}

function _placeMixedRealityAvatarFromController(controller, source = 'select') {
  const placement = xrState.mrPlacement;
  if (!placement || xrState.xrMode !== XR_MODE_MR) return false;
  if (controller && _setRayFromController(controller)) {
    _updateMixedRealityPlacementCandidateFromCurrentRay('controller-ray');
  }
  return _commitMixedRealityAvatarPlacement(placement, source);
}

function _placeMixedRealityAvatarFromGaze(source = 'gaze') {
  const placement = xrState.mrPlacement;
  if (!placement || xrState.xrMode !== XR_MODE_MR) return false;
  _updateMixedRealityPlacementCandidateFromGaze('gaze');
  return _commitMixedRealityAvatarPlacement(placement, source);
}

function _commitMixedRealityAvatarPlacement(placement, source = 'select') {
  if (!placement || xrState.xrMode !== XR_MODE_MR) return false;
  if (!placement.candidateValid) {
    if (!_defaultMixedRealityPlacementPoint(placement, placement.candidatePosition)) return false;
    placement.candidateValid = true;
    placement.candidateSource = 'view-forward';
  }
  _clearMixedRealityAnchor(placement);
  _setMixedRealityAvatarVisible(true);
  _applyMixedRealityAvatarPosition(placement.candidatePosition, placement.tempCamera);
  placement.manualMode = false;
  placement.userPlaced = true;
  placement.avatarPlaced = true;
  placement.placedAt = performance.now();
  placement.candidateValid = false;
  if (placement.reticle) placement.reticle.visible = false;
  _setMixedRealityPlacementStatus('Avatar placed. Use Place Her, palm menu, or squeeze to move her again.');
  _refreshHandMenu(true);
  xrSession.recordEvent(xrState.serverSessionId, 'mr_avatar_user_placed', {
    source,
    candidate_source: placement.candidateSource || '',
    x: _roundPoseValue(xrState.vrm?.scene?.position?.x),
    y: _roundPoseValue(xrState.vrm?.scene?.position?.y),
    z: _roundPoseValue(xrState.vrm?.scene?.position?.z),
  });
  _recordVrmVisibilitySnapshot('mr_user_placed');
  return true;
}

function _applyMixedRealityAvatarPosition(position, scratchTarget = null) {
  const avatar = xrState.vrm?.scene;
  if (!avatar || !position) return;
  avatar.position.copy(position);
  avatar.rotation.x = 0;
  avatar.rotation.z = 0;
  _faceMixedRealityAvatarToCamera(avatar, scratchTarget);
  if (xrState.mrShadow) {
    xrState.mrShadow.position.x = avatar.position.x;
    xrState.mrShadow.position.y = avatar.position.y + 0.012;
    xrState.mrShadow.position.z = avatar.position.z;
  }
  avatar.updateMatrixWorld?.(true);
}

function _faceMixedRealityAvatarToCamera(avatar, scratchTarget = null) {
  if (!avatar || !xrState.camera) return;
  const target = scratchTarget || xrState.handTempA;
  if (!target) return;
  xrState.camera.getWorldPosition(target);
  const dx = target.x - avatar.position.x;
  const dz = target.z - avatar.position.z;
  if (Math.abs(dx) > 0.001 || Math.abs(dz) > 0.001) {
    avatar.rotation.y = Math.atan2(dx, dz);
  }
}

function _setupBoundaryVisual(space) {
  const THREE = xrState.THREE;
  if (!THREE || !xrState.scene) return;
  if (xrState.xrMode !== XR_MODE_MR) return;
  const bounds = Array.isArray(space?.boundsGeometry) ? space.boundsGeometry : [];
  if (bounds.length < 3) return;
  _disposeBoundaryVisual();
  const points = bounds.map((p) => new THREE.Vector3(Number(p.x || 0), 0.018, Number(p.z || 0)));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: xrState.xrMode === XR_MODE_MR ? 0x78ffd6 : 0x8aa8ff,
    transparent: true,
    opacity: 0.24,
    depthTest: false,
    depthWrite: false,
  });
  const line = new THREE.LineLoop(geometry, material);
  line.name = 'AugmentumXRBoundary';
  line.renderOrder = 3;
  xrState.scene.add(line);
  xrState.boundaryLine = line;
}

function _setupFallbackBoundaryVisual() {
  const THREE = xrState.THREE;
  if (!THREE || !xrState.scene || xrState.boundaryLine) return;
  const points = [];
  for (let i = 0; i < 96; i++) {
    const a = (i / 96) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(a) * 1.45, 0.018, Math.sin(a) * 1.45));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x78ffd6,
    transparent: true,
    opacity: 0.26,
    depthTest: false,
    depthWrite: false,
  });
  const line = new THREE.LineLoop(geometry, material);
  line.name = 'AugmentumXRFallbackBoundary';
  line.renderOrder = 3;
  xrState.scene.add(line);
  xrState.boundaryLine = line;
}

function _disposeBoundaryVisual() {
  const line = xrState.boundaryLine;
  if (!line) return;
  line.parent?.remove(line);
  line.geometry?.dispose();
  line.material?.dispose();
  xrState.boundaryLine = null;
}

function _updateMixedRealityCompanion() {
  if (xrState.xrMode !== XR_MODE_MR || !xrState.vrm?.scene || !xrState.camera) return;
  const avatar = xrState.vrm.scene;
  _faceMixedRealityAvatarToCamera(avatar);
  if (xrState.mrShadow) {
    xrState.mrShadow.position.x = avatar.position.x;
    xrState.mrShadow.position.y = avatar.position.y + 0.012;
    xrState.mrShadow.position.z = avatar.position.z;
  }
}

function _setupControllers(THREE, renderer, rig, XRControllerModelFactory, XRHandModelFactory) {
  const ctrlFactory = new XRControllerModelFactory();
  const handFactory = new XRHandModelFactory();

  // Shared laser line geometry/material — created once, instanced per
  // controller on `connected` events. Tagged with userData.isXRLaser so
  // the disconnected handler removes only our line, not other children.
  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1),
  ]);

  // Grip squeeze on either controller recenters the rig to the saved
  // seat — useful after long sessions when the headset's local-floor
  // reference space drifts, or when the user has turned to look at
  // something off-axis and wants to face the avatar again.
  const onSqueezeStart = (event) => {
    const controller = event?.target || null;
    const inputSource = _inputSourceFromXrEvent(event, controller);
    _recordXrInputProbeEvent('squeezestart', controller, inputSource);
    if (_tryOpenPalmPinchMenuForInputSource(inputSource, 'native-squeeze', controller)) return;
    _recenterRig().catch(() => {});
  };

  for (const i of [0, 1]) {
    // RAY space (target ray origin) — receives select/squeeze events
    // and hosts the optional aim-laser. Quest swaps between controller
    // and hand-tracking input modes mid-session; the connected event
    // tells us which one is currently active.
    const controller = renderer.xr.getController(i);
    controller.userData.xrControllerIndex = i;
    rig.add(controller);

    const onSelectStart = (event) => {
      const inputSource = _inputSourceFromXrEvent(event, controller);
      _recordXrInputProbeEvent('selectstart', controller, inputSource);
      if (_tryOpenPalmPinchMenuForInputSource(inputSource, 'native-selectstart', controller)) return;
      _startPanelInteraction(controller, inputSource);
    };
    const onSelect = (event) => {
      const inputSource = _inputSourceFromXrEvent(event, controller);
      const now = performance.now();
      if (_isHandInputSource(inputSource) && now - xrState.lastComfortSelectAt < XR_HAND_PINCH_SELECT_COOLDOWN_MS) {
        return;
      }
      xrState.lastNativeSelectAt = now;
      _recordXrInputProbeEvent('select', controller, inputSource);
      if (_tryOpenPalmPinchMenuForInputSource(inputSource, 'native-select', controller)) return;
      if (xrState.panelDrag?.controller === controller) return;
      if (xrState.hubDrag?.controller === controller) return;
      _selectXrTarget(controller, inputSource);
    };
    const onSelectEnd = (event) => {
      const inputSource = _inputSourceFromXrEvent(event, controller);
      _recordXrInputProbeEvent('selectend', controller, inputSource);
      if (_endPanelInteraction(controller)) return;
      _endHubInteraction(controller);
    };

    const onConnected = (event) => {
      const data = event.data;
      if (!data) return;
      controller.userData.xrInputSource = data;
      controller.userData.xrIsHand = !!data.hand;
      controller.userData.xrTargetRayMode = data.targetRayMode || '';
      controller.userData.xrHandedness = data.handedness || '';
      const pairedHand = xrState.hands?.[i];
      if (pairedHand) {
        pairedHand.userData.xrInputSource = data;
        pairedHand.userData.xrHandedness = data.handedness || '';
      }
      _recordXrInputProbeEvent('connected', controller, data);
      // Quest exposes both Touch controllers and hand tracking through
      // WebXR target-ray space. We render that native ray; selection itself
      // is driven by WebXR select/selectstart/selectend events, not by our
      // own thumb-index distance checks.
      if (data.targetRayMode === 'tracked-pointer') {
        const lineMat = new THREE.LineBasicMaterial({
          color: data.hand ? 0x84ffd8 : 0x8aa8ff,
          transparent: true,
          opacity: data.hand ? 0.72 : 0.62,
          depthTest: true,
          depthWrite: false,
        });
        const line = new THREE.Line(lineGeo.clone(), lineMat);
        line.userData.isXRLaser = true;
        line.frustumCulled = false;
        line.scale.z = 1.15;
        controller.add(line);
        const reticleGeo = new THREE.RingGeometry(0.014, 0.023, 32);
        const reticleMat = new THREE.MeshBasicMaterial({
          color: data.hand ? 0x84ffd8 : 0x8aa8ff,
          transparent: true,
          opacity: 0.95,
          depthTest: false,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        const reticle = new THREE.Mesh(reticleGeo, reticleMat);
        reticle.name = `AugmentumXRTargetReticle:${i}`;
        reticle.renderOrder = 120;
        reticle.frustumCulled = false;
        reticle.visible = false;
        rig.add(reticle);
        controller.userData.xrTargetVisual = { line, lineMat, reticle, reticleGeo, reticleMat };
        if (!xrState.targetRayControllers.includes(controller)) {
          xrState.targetRayControllers.push(controller);
        }
      }
    };
    const onDisconnected = () => {
      _recordXrInputProbeEvent('disconnected', controller, controller.userData.xrInputSource);
      xrState.targetRayControllers = xrState.targetRayControllers.filter((c) => c !== controller);
      const lasers = controller.children.filter((c) => c.userData?.isXRLaser);
      for (const line of lasers) controller.remove(line);
      const visual = controller.userData.xrTargetVisual;
      if (visual) {
        visual.reticle?.parent?.remove(visual.reticle);
        visual.line?.geometry?.dispose();
        visual.lineMat?.dispose();
        visual.reticleGeo?.dispose();
        visual.reticleMat?.dispose();
      }
      controller.userData.xrTargetVisual = null;
      controller.userData.xrInputSource = null;
      controller.userData.xrIsHand = false;
      controller.userData.xrTargetRayMode = '';
      controller.userData.xrHandedness = '';
    };

    controller.addEventListener('selectstart',  onSelectStart);
    controller.addEventListener('select',       onSelect);
    controller.addEventListener('selectend',    onSelectEnd);
    controller.addEventListener('squeezestart', onSqueezeStart);
    controller.addEventListener('connected',    onConnected);
    controller.addEventListener('disconnected', onDisconnected);

    xrState.controllers.push(controller);
    xrState.controllerListeners.push({ ctrl: controller, ev: 'selectstart',  fn: onSelectStart });
    xrState.controllerListeners.push({ ctrl: controller, ev: 'select',       fn: onSelect });
    xrState.controllerListeners.push({ ctrl: controller, ev: 'selectend',    fn: onSelectEnd });
    xrState.controllerListeners.push({ ctrl: controller, ev: 'squeezestart', fn: onSqueezeStart });
    xrState.controllerListeners.push({ ctrl: controller, ev: 'connected',    fn: onConnected });
    xrState.controllerListeners.push({ ctrl: controller, ev: 'disconnected', fn: onDisconnected });

    // GRIP space (physical hand pose, where the controller mesh lives).
    // XRControllerModelFactory loads the right profile (Touch / Touch Pro
    // / Index / generic) from the WebXR input-profiles registry on
    // jsdelivr — already in our CSP allowlist.
    const grip = renderer.xr.getControllerGrip(i);
    grip.userData.xrControllerIndex = i;
    grip.add(ctrlFactory.createControllerModel(grip));
    rig.add(grip);
    xrState.controllers.push(grip);

    // HAND space (only populated when hand tracking is active). Mesh
    // mode renders a realistic skinned hand from the Oculus profile;
    // when controllers are in use, the hand model stays invisible.
    const hand = renderer.xr.getHand(i);
    hand.userData.xrHandIndex = i;
    hand.userData.xrControllerIndex = i;
    hand.add(handFactory.createHandModel(hand, 'mesh'));
    rig.add(hand);
    xrState.controllers.push(hand);
    xrState.hands.push(hand);
  }
}

function _normalizeHubSurfaces(surfaceCatalog = null) {
  if (!Array.isArray(surfaceCatalog) || surfaceCatalog.length === 0) {
    return XR_HUB_SURFACES;
  }
  return surfaceCatalog
    .filter((s) => s?.action && s?.label)
    .map((s) => ({
      id: s.id || s.action,
      label: s.label,
      action: s.action,
      hint: s.hubHint || s.hint || s.voiceCue || s.panelKind || '',
      placement: s.placement || '',
      panelKind: s.panelKind || '',
      voiceCue: s.voiceCue || '',
      embedUrl: s.embedUrl || '',
      primaryActions: Array.isArray(s.primaryActions) ? s.primaryActions : [],
      contextSources: Array.isArray(s.contextSources) ? s.contextSources : [],
    }));
}

function _setupModeHub(THREE, rig, roomManifest = null, surfaceCatalog = null, roomState = null) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 768;
  const ctx = canvas.getContext('2d');

  const tex = new THREE.CanvasTexture(canvas);
  _configurePanelTexture(tex);

  const geom = new THREE.PlaneGeometry(1.18, 0.88);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const panel = new THREE.Mesh(geom, mat);
  panel.name = 'AugmentumXRHubPanel';
  panel.renderOrder = 40;
  panel.userData.xrModeHub = true;

  const group = new THREE.Group();
  group.name = 'AugmentumXRHub';
  const anchor = roomState?.hub?.pose || DEFAULT_HUB_ANCHOR;
  group.visible = roomState?.hub?.enabled !== false;
  group.position.set(
    Number(anchor.x ?? 0),
    Number(anchor.y ?? 1.05),
    Number(anchor.z ?? -1.08),
  );
  group.rotation.y = Number(anchor.rotY ?? 0);
  group.scale.setScalar(Math.max(0.82, Math.min(1.18, Number(anchor.scale ?? 1) || 1)));
  group.add(panel);
  rig.add(group);

  xrState.modeHubGroup = group;
  xrState.modeHubPanel = panel;
  xrState.modeHubCanvas = canvas;
  xrState.modeHubCtx = ctx;
  xrState.modeHubTexture = tex;
  xrState.modeHubButtons = [];
  xrState.modeHubSurfaces = _normalizeHubSurfaces(surfaceCatalog);
  xrState.modeHubActiveAction = 'voice';
  xrState.modeHubStatus = 'Voice call is live. Select a surface to bring it into focus.';
  xrState.modeHubLastKey = '';
  xrState.hubUserPlaced = !!roomState?.hub?.pose;
  xrState.raycaster = new THREE.Raycaster();
  xrState.rayOrigin = new THREE.Vector3();
  xrState.rayDirection = new THREE.Vector3();
  xrState.rayMatrix = new THREE.Matrix4();
  xrState.panelDragWorldPoint = new THREE.Vector3();
  xrState.panelCameraWorld = new THREE.Vector3();
  xrState.handTempA = new THREE.Vector3();
  xrState.handTempB = new THREE.Vector3();
  xrState.handTempC = new THREE.Vector3();
  xrState.handTempD = new THREE.Vector3();
  xrState.handTempQuat = new THREE.Quaternion();

  _refreshModeHub(true);
  _restoreSpatialPanels(roomState?.surfacePanels || {});
}

function _setupControllerlessComfortUI(THREE, rig) {
  if (!THREE || !rig) return;
  const gazeGeo = new THREE.RingGeometry(0.018, 0.029, 32);
  const gazeMat = new THREE.MeshBasicMaterial({
    color: 0x84ffd8,
    transparent: true,
    opacity: 0.74,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const gazeCursor = new THREE.Mesh(gazeGeo, gazeMat);
  gazeCursor.name = 'AugmentumXRGazeCursor';
  gazeCursor.renderOrder = 150;
  gazeCursor.frustumCulled = false;
  gazeCursor.visible = false;
  rig.add(gazeCursor);

  const canvas = document.createElement('canvas');
  canvas.width = 896;
  canvas.height = 560;
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  _configurePanelTexture(texture);
  const geometry = new THREE.PlaneGeometry(0.74, 0.462);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const panel = new THREE.Mesh(geometry, material);
  panel.name = 'AugmentumXRHandMenuPanel';
  panel.renderOrder = 130;
  panel.frustumCulled = false;
  panel.userData.xrHandMenu = true;

  const group = new THREE.Group();
  group.name = 'AugmentumXRHandMenu';
  group.visible = false;
  group.add(panel);
  rig.add(group);

  const debugCanvas = document.createElement('canvas');
  debugCanvas.width = 1024;
  debugCanvas.height = 576;
  const debugCtx = debugCanvas.getContext('2d');
  const debugTexture = new THREE.CanvasTexture(debugCanvas);
  _configurePanelTexture(debugTexture);
  const debugGeometry = new THREE.PlaneGeometry(0.92, 0.518);
  const debugMaterial = new THREE.MeshBasicMaterial({
    map: debugTexture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const debugPanel = new THREE.Mesh(debugGeometry, debugMaterial);
  debugPanel.name = 'AugmentumXRHandProbePanel';
  debugPanel.renderOrder = 135;
  debugPanel.frustumCulled = false;

  const debugGroup = new THREE.Group();
  debugGroup.name = 'AugmentumXRHandProbe';
  debugGroup.visible = false;
  debugGroup.add(debugPanel);
  rig.add(debugGroup);

  xrState.gazeCursor = gazeCursor;
  xrState.gazeCursorMat = gazeMat;
  xrState.handMenuGroup = group;
  xrState.handMenuPanel = panel;
  xrState.handMenuCanvas = canvas;
  xrState.handMenuCtx = ctx;
  xrState.handMenuTexture = texture;
  xrState.handMenuButtons = [];
  xrState.handMenuLastKey = '';
  xrState.handDebugGroup = debugGroup;
  xrState.handDebugPanel = debugPanel;
  xrState.handDebugCanvas = debugCanvas;
  xrState.handDebugCtx = debugCtx;
  xrState.handDebugTexture = debugTexture;
  xrState.handDebugVisible = false;
  xrState.handDebugLastKey = '';
  xrState.handDebugLastDrawAt = 0;
  _refreshHandMenu(true);
}

function _disposeControllerlessComfortUI() {
  if (xrState.gazeCursor) {
    xrState.gazeCursor.parent?.remove(xrState.gazeCursor);
    xrState.gazeCursor.geometry?.dispose();
    xrState.gazeCursor.material?.dispose();
  }
  if (xrState.handMenuGroup) {
    xrState.handMenuGroup.parent?.remove(xrState.handMenuGroup);
  }
  if (xrState.handMenuPanel) {
    xrState.handMenuPanel.geometry?.dispose();
    xrState.handMenuPanel.material?.map?.dispose();
    xrState.handMenuPanel.material?.dispose();
  }
  if (xrState.handDebugGroup) {
    xrState.handDebugGroup.parent?.remove(xrState.handDebugGroup);
  }
  if (xrState.handDebugPanel) {
    xrState.handDebugPanel.geometry?.dispose();
    xrState.handDebugPanel.material?.map?.dispose();
    xrState.handDebugPanel.material?.dispose();
  }
  xrState.gazeCursor = null;
  xrState.gazeCursorMat = null;
  xrState.gazeHit = null;
  xrState.handMenuGroup = null;
  xrState.handMenuPanel = null;
  xrState.handMenuCanvas = null;
  xrState.handMenuCtx = null;
  xrState.handMenuTexture = null;
  xrState.handMenuButtons = [];
  xrState.handMenuHiddenAt = 0;
  xrState.handDebugGroup = null;
  xrState.handDebugPanel = null;
  xrState.handDebugCanvas = null;
  xrState.handDebugCtx = null;
  xrState.handDebugTexture = null;
  xrState.handDebugVisible = false;
  xrState.handDebugLastKey = '';
  xrState.handDebugLastDrawAt = 0;
}

function _refreshHandMenu(force = false) {
  const ctx = xrState.handMenuCtx;
  const texture = xrState.handMenuTexture;
  const canvas = xrState.handMenuCanvas;
  if (!ctx || !texture || !canvas) return;
  const surfaces = (xrState.modeHubSurfaces || XR_HUB_SURFACES)
    .filter((surface) => surface?.action && surface.action !== 'voice');
  const key = [
    xrState.xrMode,
    xrState.modeHubActiveAction,
    xrState.handDebugVisible ? 'probe-on' : 'probe-off',
    xrState.modeHubGroup?.visible === false ? 'hub-hidden' : 'hub-visible',
    xrState.spatialPanels?.size || 0,
    surfaces.map((surface) => surface.action).join(','),
    _isMixedRealityAvatarPlacementMode() ? 'placing' : 'idle',
  ].join(':');
  if (!force && key === xrState.handMenuLastKey) return;
  xrState.handMenuLastKey = key;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(9, 13, 23, 0.94)';
  _roundRect(ctx, 0, 0, w, h, 22);
  ctx.fill();
  ctx.strokeStyle = 'rgba(132, 255, 216, 0.58)';
  ctx.lineWidth = 4;
  ctx.stroke();

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#f5fff9';
  ctx.font = 'bold 38px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('Quick Menu', 32, 24);
  ctx.fillStyle = 'rgba(222, 244, 255, 0.72)';
  ctx.font = '18px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('Look to aim. Pinch or trigger to commit.', 34, 72);

  const buttons = [];
  const utilDefs = [
    xrState.xrMode === XR_MODE_MR
      ? { action: 'place-avatar', label: 'Place Her', color: 'gold' }
      : null,
    {
      action: 'toggle-hub',
      label: xrState.modeHubGroup?.visible === false ? 'Show Hub' : 'Hide Hub',
      color: 'cyan',
    },
    {
      action: 'close-panels',
      label: 'Close All',
      color: 'plain',
      disabled: !(xrState.spatialPanels?.size > 0),
    },
    { action: 'toggle-probe', label: xrState.handDebugVisible ? 'Hide Probe' : 'Probe', color: 'plain' },
    { action: 'hide-menu', label: 'Dismiss', color: 'plain' },
  ].filter(Boolean);
  const utilGap = 12;
  const utilW = Math.floor((w - 64 - utilGap * (utilDefs.length - 1)) / utilDefs.length);
  const utilH = 62;
  let x = 32;
  const utilY = 112;
  for (const def of utilDefs) {
    const palette = def.color === 'gold'
      ? ['rgba(255, 214, 118, 0.19)', 'rgba(255, 232, 168, 0.72)', '#fff4d6']
      : def.color === 'cyan'
        ? ['rgba(120, 255, 214, 0.16)', 'rgba(132, 255, 216, 0.62)', '#e4fff8']
        : ['rgba(255, 255, 255, 0.10)', 'rgba(210, 224, 255, 0.42)', '#f4f7ff'];
    ctx.fillStyle = def.disabled ? 'rgba(255, 255, 255, 0.045)' : palette[0];
    _roundRect(ctx, x, utilY, utilW, utilH, 14);
    ctx.fill();
    ctx.strokeStyle = def.disabled ? 'rgba(180, 190, 215, 0.22)' : palette[1];
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = def.disabled ? 'rgba(210, 218, 236, 0.46)' : palette[2];
    ctx.font = 'bold 19px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(_fitCanvasText(ctx, def.label, utilW - 22), x + 12, utilY + 21);
    buttons.push({ id: def.action, action: def.action, label: def.label, disabled: !!def.disabled, x, y: utilY, w: utilW, h: utilH });
    x += utilW + utilGap;
  }

  ctx.fillStyle = 'rgba(222, 244, 255, 0.64)';
  ctx.font = '17px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('Open Surface', 34, 198);

  const cols = 4;
  const gapX = 16;
  const gapY = 12;
  const cardW = Math.floor((w - 64 - gapX * (cols - 1)) / cols);
  const cardH = 78;
  const startX = 32;
  const startY = 226;
  surfaces.slice(0, 12).forEach((surface, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const cardX = startX + col * (cardW + gapX);
    const cardY = startY + row * (cardH + gapY);
    const open = xrState.spatialPanels?.has?.(surface.action);
    const selected = surface.action === xrState.modeHubActiveAction;
    ctx.fillStyle = selected
      ? 'rgba(132, 172, 255, 0.30)'
      : (open ? 'rgba(120, 255, 214, 0.14)' : 'rgba(255, 255, 255, 0.075)');
    _roundRect(ctx, cardX, cardY, cardW, cardH, 14);
    ctx.fill();
    ctx.strokeStyle = selected
      ? 'rgba(205, 220, 255, 0.84)'
      : (open ? 'rgba(132, 255, 216, 0.58)' : 'rgba(144, 160, 210, 0.34)');
    ctx.lineWidth = selected ? 3 : 2;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 23px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(_fitCanvasText(ctx, surface.label || surface.action, cardW - 28), cardX + 14, cardY + 12);
    const hint = surface.hint || surface.hubHint || surface.placement || '';
    if (hint) {
      ctx.fillStyle = 'rgba(225, 232, 250, 0.66)';
      ctx.font = '14px ui-monospace, Menlo, Consolas, monospace';
      _wrapCanvasText(ctx, hint, cardX + 14, cardY + 43, cardW - 28, 18, 1);
    }
    buttons.push({
      id: `surface:${surface.action}`,
      kind: 'surface',
      action: 'open-surface',
      surfaceAction: surface.action,
      label: surface.label || surface.action,
      x: cardX,
      y: cardY,
      w: cardW,
      h: cardH,
    });
  });

  ctx.fillStyle = 'rgba(170, 190, 230, 0.56)';
  ctx.font = '15px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('Quick Menu opens only after a steady palm hold, then pinch/select.', 34, h - 34);

  xrState.handMenuButtons = buttons;
  texture.needsUpdate = true;
}

function _isHandMenuOpenSuppressed(now = performance.now()) {
  return !!xrState.handMenuHiddenAt
    && now - xrState.handMenuHiddenAt < XR_HAND_MENU_SUPPRESS_AFTER_HIDE_MS;
}

function _showHandMenuNearHand(state = null, source = 'hand') {
  if (!xrState.handMenuGroup) return false;
  xrState.handMenuGroup.visible = true;
  xrState.handMenuHiddenAt = 0;
  xrState.handMenuHandIndex = Number.isFinite(state?.index) ? state.index : null;
  xrState.handMenuShownAt = performance.now();
  _refreshHandMenu(true);
  if (state && _updateHandMenuPoseFromState(state)) {
    xrSession.recordEvent(xrState.serverSessionId, 'hand_menu_shown', { source, anchored: 'hand' });
    return true;
  }
  _positionHandMenuNearView();
  xrSession.recordEvent(xrState.serverSessionId, 'hand_menu_shown', { source, anchored: 'view' });
  return true;
}

function _hideHandMenu(source = 'manual') {
  if (!xrState.handMenuGroup) return false;
  const now = performance.now();
  xrState.handMenuGroup.visible = false;
  xrState.handMenuHandIndex = null;
  xrState.handMenuHiddenAt = now;
  xrState.handMenuLastKey = '';
  for (const state of xrState.handStates?.values?.() || []) {
    state.palmStartedAt = 0;
    state.lastPalmSummonAt = now;
  }
  xrSession.recordEvent(xrState.serverSessionId, 'hand_menu_hidden', { source });
  return true;
}

function _positionHandMenuNearView() {
  if (!xrState.handMenuGroup || !xrState.camera || !xrState.xrRig) return false;
  const cameraWorld = xrState.handTempA;
  const forward = xrState.handTempB;
  const target = xrState.handTempC;
  if (!cameraWorld || !forward || !target) return false;
  xrState.camera.updateMatrixWorld?.(true);
  xrState.camera.getWorldPosition(cameraWorld);
  xrState.camera.getWorldDirection(forward);
  target.copy(cameraWorld).add(forward.multiplyScalar(0.82));
  xrState.xrRig.worldToLocal(target);
  target.y = Math.max(0.78, Math.min(1.42, target.y - 0.10));
  xrState.handMenuGroup.position.copy(target);
  xrState.handMenuGroup.lookAt(cameraWorld);
  return true;
}

function _updateHandMenuPoseFromState(state) {
  if (!state || !xrState.handMenuGroup || !xrState.camera || !xrState.xrRig) return false;
  const target = xrState.handTempA;
  const cameraWorld = xrState.handTempB;
  const toCamera = xrState.handTempC;
  if (!target || !cameraWorld || !toCamera) return false;
  target.copy(state.wristWorld);
  target.y += 0.105;
  xrState.camera.updateMatrixWorld?.(true);
  xrState.camera.getWorldPosition(cameraWorld);
  toCamera.copy(cameraWorld).sub(target);
  toCamera.y = 0;
  if (toCamera.lengthSq() > 0.0001) {
    toCamera.normalize();
    target.addScaledVector(toCamera, 0.08);
  }
  xrState.xrRig.worldToLocal(target);
  target.y = Math.max(0.66, Math.min(1.58, target.y));
  xrState.handMenuGroup.position.copy(target);
  xrState.handMenuGroup.lookAt(cameraWorld);
  return true;
}

function _toggleHandDebugPanel(source = 'hand-menu') {
  if (!xrState.handDebugGroup) return false;
  xrState.handDebugVisible = !xrState.handDebugVisible;
  xrState.handDebugGroup.visible = xrState.handDebugVisible;
  xrState.handDebugLastKey = '';
  xrState.handDebugLastDrawAt = 0;
  _refreshHandMenu(true);
  if (xrState.handDebugVisible) {
    _positionHandDebugNearView();
    _updateHandDebugPanel(true);
  }
  xrSession.recordEvent(xrState.serverSessionId, 'hand_probe_toggled', {
    visible: xrState.handDebugVisible,
    source,
  });
  return true;
}

function _positionHandDebugNearView() {
  if (!xrState.handDebugGroup || !xrState.camera || !xrState.xrRig) return false;
  const cameraWorld = xrState.handTempA;
  const forward = xrState.handTempB;
  const right = xrState.handTempC;
  const target = xrState.handTempD;
  if (!cameraWorld || !forward || !right || !target) return false;
  xrState.camera.updateMatrixWorld?.(true);
  xrState.camera.getWorldPosition(cameraWorld);
  xrState.camera.getWorldDirection(forward);
  right.setFromMatrixColumn(xrState.camera.matrixWorld, 0).normalize();
  target.copy(cameraWorld)
    .addScaledVector(forward, 1.08)
    .addScaledVector(right, 0.50);
  xrState.xrRig.worldToLocal(target);
  target.y = Math.max(0.78, Math.min(1.52, target.y - 0.03));
  xrState.handDebugGroup.position.copy(target);
  xrState.handDebugGroup.lookAt(cameraWorld);
  return true;
}

function _formatProbeAge(now, at) {
  if (!at) return '-';
  const ms = Math.max(0, now - Number(at || 0));
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${_roundProbeValue(ms / 1000, 1)}s`;
}

function _handProbeRow(state, now) {
  const intent = _nativeIntentForHandState(state, now);
  const handedness = state.handedness || intent?.handedness || `hand ${state.index}`;
  const pinchDistance = state.thumbWorld.distanceTo(state.indexWorld);
  const source = state.nativeSelectActive
    ? 'native select'
    : (state.nativeButtonActive ? `native button ${state.nativeActiveButtons.join(',')}` : (state.pinching ? 'geometry pinch' : 'open'));
  const waveSpan = Math.max(0, Number(state.waveMax || 0) - Number(state.waveMin || 0));
  const profiles = (intent?.profiles || []).slice(0, 2).join(', ') || 'no profile';
  return [
    `${handedness}: ${source}`,
    `pinch ${_roundProbeValue(pinchDistance, 3)}m  palm ${_roundProbeValue(state.palmFacingScore, 2)}${state.palmMenuReady ? ' ready' : ''}`,
    `wave span ${_roundProbeValue(waveSpan, 3)} travel ${_roundProbeValue(state.waveTravel, 3)} turns ${state.waveTurns || 0}`,
    profiles,
  ];
}

function _drawProbeLine(ctx, text, x, y, maxWidth, color = '#dce8ff', font = '22px ui-monospace, Menlo, Consolas, monospace') {
  ctx.fillStyle = color;
  ctx.font = font;
  ctx.fillText(_fitCanvasText(ctx, text, maxWidth), x, y);
}

function _updateHandDebugPanel(force = false) {
  if (!xrState.handDebugVisible || !xrState.handDebugGroup) return;
  const now = performance.now();
  if (!force && now - xrState.handDebugLastDrawAt < XR_HAND_DEBUG_REDRAW_MS) return;
  xrState.handDebugLastDrawAt = now;
  _positionHandDebugNearView();

  const ctx = xrState.handDebugCtx;
  const canvas = xrState.handDebugCanvas;
  const texture = xrState.handDebugTexture;
  if (!ctx || !canvas || !texture) return;
  const handStates = (xrState.hands || []).map((hand) => _handStateFor(hand));
  const frameStats = xrState.xrFrameStats || {};
  const inputRows = Array.from(xrState.handProbeInputs.values())
    .filter((input) => input?.hasHand)
    .slice(-2);
  const eventRows = (xrState.handProbeEvents || []).slice(-7);
  const key = JSON.stringify({
    hands: handStates.map((state) => ({
      h: state.handedness,
      p: _roundProbeValue(state.thumbWorld.distanceTo(state.indexWorld), 3),
      palm: _roundProbeValue(state.palmFacingScore, 2),
      ready: !!state.palmMenuReady,
      source: state.pinchSource || '',
      ns: !!state.nativeSelectActive,
      nb: state.nativeActiveButtons || [],
      w: _roundProbeValue(Number(state.waveMax || 0) - Number(state.waveMin || 0), 3),
      t: state.waveTurns || 0,
    })),
    inputs: inputRows.map((input) => ({
      key: input.key,
      profiles: input.profiles,
      buttons: input.activeButtons,
      axes: input.axes,
    })),
    events: eventRows.map((event) => `${event.type}:${event.hand}:${(event.buttons || []).join(',')}`),
    frame: {
      avg: Math.round(frameStats.averageMs || 0),
      max: Math.round(frameStats.maxMs || 0),
      long: frameStats.longFrames || 0,
    },
  });
  if (!force && key === xrState.handDebugLastKey) return;
  xrState.handDebugLastKey = key;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(8, 12, 22, 0.94)';
  _roundRect(ctx, 0, 0, w, h, 24);
  ctx.fill();
  ctx.strokeStyle = 'rgba(132, 255, 216, 0.54)';
  ctx.lineWidth = 4;
  ctx.stroke();

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  _drawProbeLine(ctx, 'XR Hand Probe', 34, 26, 560, '#f4fff9', 'bold 34px ui-monospace, Menlo, Consolas, monospace');
  _drawProbeLine(ctx, 'Quick Menu gesture: steady palm hold + pinch/select', 36, 68, 760, 'rgba(220, 244, 255, 0.72)', '20px ui-monospace, Menlo, Consolas, monospace');
  _drawProbeLine(
    ctx,
    `XR frame avg ${Math.round(frameStats.averageMs || 0)}ms  max ${Math.round(frameStats.maxMs || 0)}ms  long ${frameStats.longFrames || 0}`,
    620,
    30,
    360,
    'rgba(191, 248, 232, 0.86)',
    '18px ui-monospace, Menlo, Consolas, monospace',
  );

  let y = 112;
  for (const state of handStates) {
    const lines = _handProbeRow(state, now);
    _drawProbeLine(ctx, lines[0], 38, y, 440, '#f5fff9', 'bold 23px ui-monospace, Menlo, Consolas, monospace');
    _drawProbeLine(ctx, lines[1], 38, y + 30, 430, '#bff8e8');
    _drawProbeLine(ctx, lines[2], 38, y + 58, 430, '#bdccff');
    _drawProbeLine(ctx, lines[3], 38, y + 86, 430, 'rgba(220, 230, 255, 0.70)', '18px ui-monospace, Menlo, Consolas, monospace');
    y += 128;
  }
  if (!handStates.length) {
    _drawProbeLine(ctx, 'No WebXR hand objects yet.', 38, y, 430, '#ffddb3');
    y += 44;
  }

  const rightX = 520;
  _drawProbeLine(ctx, 'Native input', rightX, 112, 420, '#f5fff9', 'bold 23px ui-monospace, Menlo, Consolas, monospace');
  let inputY = 148;
  if (!inputRows.length) {
    _drawProbeLine(ctx, 'No hand inputSource/gamepad data yet.', rightX, inputY, 430, '#ffddb3');
    inputY += 32;
  }
  for (const input of inputRows) {
    const profile = (input.profiles || []).slice(0, 2).join(', ') || 'no profile';
    const buttons = (input.activeButtons || []).length ? (input.activeButtons || []).join(',') : '-';
    _drawProbeLine(ctx, `${input.handedness || input.key}: ${input.targetRayMode || '-'}`, rightX, inputY, 430, '#dce8ff');
    _drawProbeLine(ctx, `buttons ${buttons}  axes ${(input.axes || []).join(',') || '-'}`, rightX, inputY + 28, 430, '#bff8e8', '18px ui-monospace, Menlo, Consolas, monospace');
    _drawProbeLine(ctx, profile, rightX, inputY + 52, 430, 'rgba(220, 230, 255, 0.68)', '18px ui-monospace, Menlo, Consolas, monospace');
    inputY += 88;
  }

  _drawProbeLine(ctx, 'Recent events', rightX, 338, 420, '#f5fff9', 'bold 23px ui-monospace, Menlo, Consolas, monospace');
  let eventY = 374;
  for (const event of eventRows.slice().reverse()) {
    const buttons = (event.buttons || []).length ? ` b:${event.buttons.join(',')}` : '';
    const value = Number.isFinite(Number(event.value)) ? ` v:${event.value}` : '';
    _drawProbeLine(ctx, `${_formatProbeAge(now, event.at)} ${event.hand || ''} ${event.type}${buttons}${value}`, rightX, eventY, 440, '#dce8ff', '18px ui-monospace, Menlo, Consolas, monospace');
    eventY += 24;
  }

  texture.needsUpdate = true;
}

function _roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function _wrapCanvasText(ctx, text, x, y, maxWidth, lineHeight, maxLines = 2) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
      if (lines.length >= maxLines) break;
    } else {
      line = test;
    }
  }
  if (line && lines.length < maxLines) lines.push(line);
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], x, y + i * lineHeight);
  }
}

function _fitCanvasText(ctx, text, maxWidth) {
  const raw = String(text || '').replace(/\s+/g, ' ').trim();
  if (!raw || !ctx || !Number.isFinite(maxWidth) || maxWidth <= 0) return '';
  if (ctx.measureText(raw).width <= maxWidth) return raw;
  let lo = 0;
  let hi = raw.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    const candidate = raw.slice(0, mid) + '...';
    if (ctx.measureText(candidate).width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return raw.slice(0, Math.max(0, lo)) + '...';
}

function _drawXrPanelPill(ctx, text, x, y, color = 'rgba(120, 180, 255, 0.18)') {
  const label = String(text || '').trim();
  if (!label) return 0;
  ctx.font = 'bold 16px ui-monospace, Menlo, Consolas, monospace';
  const w = Math.min(210, Math.max(54, ctx.measureText(label).width + 28));
  ctx.fillStyle = color;
  _roundRect(ctx, x, y, w, 30, 12);
  ctx.fill();
  ctx.strokeStyle = 'rgba(160, 190, 245, 0.34)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = 'rgba(235, 242, 255, 0.92)';
  ctx.fillText(_fitCanvasText(ctx, label, w - 22), x + 14, y + 7);
  return w;
}

function _drawSurfaceLiveContent(ctx, content, x, y, width, height) {
  if (!content) return false;
  const items = Array.isArray(content.items) ? content.items.filter((i) => i?.label).slice(0, 4) : [];
  const lines = Array.isArray(content.lines) ? content.lines.filter(Boolean).slice(0, 3) : [];
  const metrics = Array.isArray(content.metrics) ? content.metrics.filter(Boolean).slice(0, 3) : [];
  if (!items.length && !lines.length && !metrics.length) return false;

  ctx.fillStyle = 'rgba(255, 255, 255, 0.075)';
  _roundRect(ctx, x, y, width, height, 16);
  ctx.fill();
  ctx.strokeStyle = 'rgba(130, 160, 220, 0.30)';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = 'rgba(235, 242, 255, 0.86)';
  ctx.font = 'bold 18px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('Live content', x + 18, y + 16);
  ctx.fillStyle = 'rgba(172, 192, 230, 0.72)';
  ctx.font = '15px ui-monospace, Menlo, Consolas, monospace';
  const status = content.status || content.source || '';
  if (status) ctx.fillText(_fitCanvasText(ctx, status, width - 210), x + 154, y + 18);

  let pillX = x + 18;
  for (const metric of metrics) {
    const used = _drawXrPanelPill(ctx, metric, pillX, y + 46);
    pillX += used ? used + 10 : 0;
  }

  let rowY = metrics.length ? y + 88 : y + 52;
  const rowGap = 39;
  const maxY = y + height - 18;
  const drawLine = (label, detail, muted = '') => {
    if (rowY + 32 > maxY) return false;
    ctx.fillStyle = 'rgba(245, 248, 255, 0.94)';
    ctx.font = 'bold 18px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(_fitCanvasText(ctx, label, width * 0.34), x + 20, rowY);
    ctx.fillStyle = 'rgba(203, 216, 245, 0.82)';
    ctx.font = '16px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(_fitCanvasText(ctx, detail || muted, width - 250), x + 220, rowY + 1);
    rowY += rowGap;
    return true;
  };

  if (items.length) {
    for (const item of items) {
      if (!drawLine(item.label || 'Item', item.detail || item.kind || '')) break;
    }
  } else {
    for (const line of lines) {
      if (!drawLine('Update', line)) break;
    }
  }
  return true;
}

function _refreshModeHub(force = false) {
  const ctx = xrState.modeHubCtx;
  const tex = xrState.modeHubTexture;
  if (!ctx || !tex) return;

  const contextLabel = _resolveContextLabel();
  const voiceLabel = _resolveVoiceLabel();
  const surfaces = xrState.modeHubSurfaces || XR_HUB_SURFACES;
  const key = [
    xrState.xrMode,
    xrState.modeHubActiveAction,
    xrState.modeHubStatus,
    contextLabel,
    voiceLabel,
    surfaces.map((s) => s.action || s.id || '').join(','),
    xrState.hubDrag ? 'dragging' : '',
  ].join('|');
  if (!force && key === xrState.modeHubLastKey) return;
  xrState.modeHubLastKey = key;

  const w = xrState.modeHubCanvas.width;
  const h = xrState.modeHubCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, 'rgba(8, 12, 22, 0.95)');
  bg.addColorStop(1, 'rgba(18, 24, 40, 0.94)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  const hubMoveButton = { id: 'move-hub', kind: 'move-hub', action: 'move-hub', x: 0, y: 0, w: 784, h: 122 };
  ctx.fillStyle = xrState.hubDrag ? 'rgba(92, 128, 255, 0.45)' : 'rgba(72, 100, 200, 0.20)';
  ctx.fillRect(0, 0, hubMoveButton.w, hubMoveButton.h);
  ctx.strokeStyle = 'rgba(120, 160, 255, 0.62)';
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, w - 4, h - 4);

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#eef3ff';
  ctx.font = 'bold 54px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(xrState.xrMode === XR_MODE_MR ? 'Augmentum MR' : 'Augmentum Hub', 54, 42);
  ctx.fillStyle = 'rgba(230, 238, 255, 0.62)';
  ctx.font = '17px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('grab title bar to place', 428, 60);

  ctx.fillStyle = 'rgba(188, 204, 236, 0.88)';
  ctx.font = '26px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(`${voiceLabel}  |  ${contextLabel || 'Voice'}`, 58, 108);

  ctx.fillStyle = 'rgba(230, 236, 255, 0.90)';
  ctx.font = '22px ui-monospace, Menlo, Consolas, monospace';
  _wrapCanvasText(ctx, xrState.modeHubStatus, 58, 151, 705, 29, 2);

  const exitX = 808;
  const exitY = 54;
  const exitW = 152;
  const exitH = 58;
  const switchX = 808;
  const switchY = 122;
  const switchW = 152;
  const switchH = 58;
  const placeX = 808;
  const placeY = 190;
  const placeW = 152;
  const placeH = 58;
  ctx.fillStyle = 'rgba(255, 120, 120, 0.14)';
  _roundRect(ctx, exitX, exitY, exitW, exitH, 16);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 160, 160, 0.52)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#ffe6e6';
  ctx.font = 'bold 24px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('Exit VR', exitX + 24, exitY + 16);

  const switchLabel = xrState.xrMode === XR_MODE_MR ? 'Enter VR' : 'Enter MR';
  ctx.fillStyle = 'rgba(120, 255, 214, 0.12)';
  _roundRect(ctx, switchX, switchY, switchW, switchH, 16);
  ctx.fill();
  ctx.strokeStyle = 'rgba(140, 255, 224, 0.48)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#defef4';
  ctx.font = 'bold 24px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(switchLabel, switchX + 18, switchY + 16);

  if (xrState.xrMode === XR_MODE_MR) {
    ctx.fillStyle = _isMixedRealityAvatarPlacementMode()
      ? 'rgba(255, 215, 120, 0.24)'
      : 'rgba(255, 215, 120, 0.12)';
    _roundRect(ctx, placeX, placeY, placeW, placeH, 16);
    ctx.fill();
    ctx.strokeStyle = _isMixedRealityAvatarPlacementMode()
      ? 'rgba(255, 236, 180, 0.86)'
      : 'rgba(255, 218, 140, 0.50)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fff3d6';
    ctx.font = 'bold 24px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText('Place Her', placeX + 12, placeY + 16);
  }

  const buttons = [hubMoveButton];
  const columns = surfaces.length > 10 ? 3 : 2;
  const cardW = columns === 3 ? 286 : 430;
  const cardH = columns === 3 ? 78 : 74;
  const gapX = columns === 3 ? 24 : 42;
  const gapY = columns === 3 ? 10 : 12;
  const startX = columns === 3 ? 48 : 58;
  const startY = xrState.xrMode === XR_MODE_MR ? 284 : 216;
  for (let i = 0; i < surfaces.length; i++) {
    const surface = surfaces[i];
    const col = i % columns;
    const row = Math.floor(i / columns);
    const x = startX + col * (cardW + gapX);
    const y = startY + row * (cardH + gapY);
    const selected = surface.action === xrState.modeHubActiveAction;
    ctx.fillStyle = selected ? 'rgba(92, 128, 255, 0.72)' : 'rgba(255, 255, 255, 0.095)';
    _roundRect(ctx, x, y, cardW, cardH, 18);
    ctx.fill();
    ctx.strokeStyle = selected ? 'rgba(220, 230, 255, 0.85)' : 'rgba(128, 150, 200, 0.48)';
    ctx.lineWidth = selected ? 4 : 2;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = `${columns === 3 ? 'bold 23px' : 'bold 25px'} ui-monospace, Menlo, Consolas, monospace`;
    ctx.fillText(surface.label, x + 22, y + 14);
    ctx.fillStyle = 'rgba(225, 232, 250, 0.78)';
    ctx.font = `${columns === 3 ? '15px' : '16px'} ui-monospace, Menlo, Consolas, monospace`;
    const detail = surface.hint || surface.placement || '';
    _wrapCanvasText(ctx, detail, x + 22, y + 43, cardW - 42, 20, 1);
    buttons.push({ ...surface, x, y, w: cardW, h: cardH });
  }

  buttons.push({ id: 'exit', label: 'Exit VR', action: 'exit', x: exitX, y: exitY, w: exitW, h: exitH });
  buttons.push({
    id: 'switch-xr-mode',
    label: switchLabel,
    action: 'switch-xr-mode',
    x: switchX,
    y: switchY,
    w: switchW,
    h: switchH,
  });
  if (xrState.xrMode === XR_MODE_MR) {
    buttons.push({
      id: 'place-avatar',
      label: 'Place Her',
      action: 'place-avatar',
      x: placeX,
      y: placeY,
      w: placeW,
      h: placeH,
    });
  }

  ctx.fillStyle = 'rgba(170, 190, 230, 0.72)';
  ctx.font = xrState.xrMode === XR_MODE_MR
    ? '18px ui-monospace, Menlo, Consolas, monospace'
    : '20px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(
    xrState.xrMode === XR_MODE_MR
      ? 'Look to aim, pinch/trigger to commit  |  Palm hold + pinch opens Quick Menu'
      : 'Look to aim, pinch/trigger to commit  |  Palm hold + pinch opens Quick Menu',
    58,
    710,
  );

  xrState.modeHubButtons = buttons;
  tex.needsUpdate = true;
}

function _setRayFromController(controller) {
  if (
    !controller
    || !xrState.rayMatrix
    || !xrState.rayOrigin
    || !xrState.rayDirection
    || !xrState.raycaster
  ) {
    return false;
  }
  controller.updateMatrixWorld?.(true);
  xrState.rayMatrix.identity().extractRotation(controller.matrixWorld);
  xrState.rayOrigin.setFromMatrixPosition(controller.matrixWorld);
  xrState.rayDirection.set(0, 0, -1).applyMatrix4(xrState.rayMatrix);
  xrState.raycaster.near = 0;
  xrState.raycaster.far = 5;
  xrState.raycaster.set(xrState.rayOrigin, xrState.rayDirection);
  return true;
}

function _inputSourceFromXrEvent(event, controller) {
  return event?.data || controller?.userData?.xrInputSource || null;
}

function _isHandInputSource(inputSource) {
  return !!inputSource?.hand;
}

function _roundProbeValue(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  const scale = 10 ** digits;
  return Math.round(n * scale) / scale;
}

function _controllerIndexForProbe(controller) {
  const index = Number(controller?.userData?.xrControllerIndex);
  return Number.isFinite(index) ? index : null;
}

function _snapshotXrInputSource(inputSource = null) {
  const gamepad = inputSource?.gamepad || null;
  const buttons = Array.from(gamepad?.buttons || []).map((button, index) => ({
    index,
    pressed: !!button?.pressed,
    touched: !!button?.touched,
    value: _roundProbeValue(button?.value || 0, 2),
  }));
  const activeButtons = buttons
    .filter((button) => (
      button.pressed
      || button.touched
      || Number(button.value || 0) >= XR_HAND_NATIVE_BUTTON_THRESHOLD
    ))
    .map((button) => button.index);
  return {
    handedness: inputSource?.handedness || '',
    targetRayMode: inputSource?.targetRayMode || '',
    profiles: Array.from(inputSource?.profiles || []),
    hasHand: !!inputSource?.hand,
    gamepadId: gamepad?.id || '',
    gamepadMapping: gamepad?.mapping || '',
    axes: Array.from(gamepad?.axes || []).slice(0, 6).map((axis) => _roundProbeValue(axis, 2)),
    buttons,
    activeButtons,
  };
}

function _inputProbeKeys(inputSource = null, controller = null) {
  const keys = [];
  const controllerIndex = _controllerIndexForProbe(controller);
  if (controllerIndex != null) keys.push(`index:${controllerIndex}`);
  if (inputSource?.handedness) keys.push(`handedness:${inputSource.handedness}`);
  if (!keys.length && inputSource?.targetRayMode) keys.push(`ray:${inputSource.targetRayMode}`);
  return Array.from(new Set(keys));
}

function _ensureNativeHandIntent(keys, snapshot, now) {
  if (!keys.length) return null;
  let intent = keys.map((key) => xrState.nativeHandIntents.get(key)).find(Boolean);
  if (!intent) {
    intent = {
      key: keys[0],
      handedness: snapshot.handedness || '',
      index: null,
      profiles: [],
      targetRayMode: '',
      nativeSelectActive: false,
      nativeSqueezeActive: false,
      nativeButtonActive: false,
      gamepadButtons: [],
      activeButtons: [],
      lastSeenAt: 0,
      lastSelectStartAt: 0,
      lastSelectAt: 0,
      lastSelectEndAt: 0,
      lastSqueezeAt: 0,
      lastButtonAt: 0,
      source: 'native',
    };
  }
  const indexKey = keys.find((key) => key.startsWith('index:'));
  if (indexKey) {
    const parsed = Number(indexKey.slice('index:'.length));
    if (Number.isFinite(parsed)) intent.index = parsed;
  }
  intent.handedness = snapshot.handedness || intent.handedness || '';
  intent.profiles = snapshot.profiles || intent.profiles || [];
  intent.targetRayMode = snapshot.targetRayMode || intent.targetRayMode || '';
  intent.gamepadId = snapshot.gamepadId || intent.gamepadId || '';
  intent.gamepadMapping = snapshot.gamepadMapping || intent.gamepadMapping || '';
  intent.axes = snapshot.axes || [];
  intent.lastSeenAt = now;
  for (const key of keys) xrState.nativeHandIntents.set(key, intent);
  return intent;
}

function _recordXrInputProbeEvent(type, controller = null, inputSource = null, detail = {}) {
  const now = performance.now();
  const snapshot = _snapshotXrInputSource(inputSource);
  const keys = _inputProbeKeys(inputSource, controller);
  for (const key of keys) {
    xrState.handProbeInputs.set(key, { ...snapshot, key, at: now });
  }
  if (snapshot.hasHand) {
    const intent = _ensureNativeHandIntent(keys, snapshot, now);
    if (intent) {
      if (type === 'selectstart') {
        intent.nativeSelectActive = true;
        intent.lastSelectStartAt = now;
        intent.source = 'native-select';
      } else if (type === 'select') {
        intent.lastSelectAt = now;
        intent.source = 'native-select';
      } else if (type === 'selectend') {
        intent.nativeSelectActive = false;
        intent.lastSelectEndAt = now;
      } else if (type === 'squeezestart') {
        intent.nativeSqueezeActive = true;
        intent.lastSqueezeAt = now;
        intent.source = 'native-squeeze';
      } else if (type === 'squeezeend') {
        intent.nativeSqueezeActive = false;
      }
      intent.activeButtons = snapshot.activeButtons || [];
      intent.gamepadButtons = snapshot.buttons || [];
      intent.nativeButtonActive = intent.activeButtons.length > 0;
      if (intent.nativeButtonActive) intent.lastButtonAt = now;
    }
  }
  xrState.handProbeEvents.push({
    at: now,
    type,
    hand: snapshot.handedness || (keys[0] || '').replace('index:', '#') || '',
    source: snapshot.hasHand ? 'hand' : 'controller',
    buttons: snapshot.activeButtons || [],
    value: detail.value,
  });
  xrState.handProbeEvents = xrState.handProbeEvents.slice(-XR_HAND_PROBE_EVENT_LIMIT);
}

function _updateXrInputProbes(now = performance.now()) {
  const session = xrState.renderer?.xr?.getSession?.();
  const sources = Array.from(session?.inputSources || []);
  for (const inputSource of sources) {
    const snapshot = _snapshotXrInputSource(inputSource);
    const keys = _inputProbeKeys(inputSource, null);
    for (const key of keys) {
      xrState.handProbeInputs.set(key, { ...snapshot, key, at: now });
    }
    if (!snapshot.hasHand || !keys.length) continue;
    const intent = _ensureNativeHandIntent(keys, snapshot, now);
    if (!intent) continue;
    const prevButtons = Array.isArray(intent.gamepadButtons) ? intent.gamepadButtons : [];
    intent.activeButtons = snapshot.activeButtons || [];
    intent.gamepadButtons = snapshot.buttons || [];
    intent.nativeButtonActive = intent.activeButtons.length > 0;
    if (intent.nativeButtonActive) intent.lastButtonAt = now;
    for (const button of intent.gamepadButtons) {
      const prev = prevButtons.find((candidate) => candidate.index === button.index) || {};
      const wasActive = !!prev.pressed || !!prev.touched || Number(prev.value || 0) >= XR_HAND_NATIVE_BUTTON_THRESHOLD;
      const isActive = !!button.pressed || !!button.touched || Number(button.value || 0) >= XR_HAND_NATIVE_BUTTON_THRESHOLD;
      if (wasActive === isActive) continue;
      _recordXrInputProbeEvent(
        isActive ? `button${button.index}:down` : `button${button.index}:up`,
        null,
        inputSource,
        { value: button.value },
      );
    }
  }
}

function _nativeIntentForHandState(state, now = performance.now()) {
  if (!state) return null;
  const keys = [`index:${state.index}`];
  if (state.handedness) keys.push(`handedness:${state.handedness}`);
  const intents = keys.map((key) => xrState.nativeHandIntents.get(key)).filter(Boolean);
  if (!intents.length) return null;
  const merged = Object.assign({}, ...intents);
  merged.nativeSelectActive = intents.some((intent) => (
    intent.nativeSelectActive
    && now - Number(intent.lastSelectStartAt || 0) <= XR_HAND_NATIVE_SELECT_STALE_MS
  ));
  merged.nativeButtonActive = intents.some((intent) => (
    intent.nativeButtonActive
    && now - Number(intent.lastButtonAt || 0) <= XR_HAND_NATIVE_SELECT_STALE_MS
  ));
  merged.activeButtons = Array.from(new Set(intents.flatMap((intent) => intent.activeButtons || [])));
  return merged;
}

function _handStateForInputSource(inputSource = null, controller = null) {
  if (!_isHandInputSource(inputSource)) return null;
  const controllerIndex = _controllerIndexForProbe(controller);
  let hand = null;
  if (controllerIndex != null) {
    hand = (xrState.hands || []).find((candidate) => (
      Number(candidate?.userData?.xrHandIndex) === controllerIndex
    ));
  }
  if (!hand && inputSource?.handedness) {
    hand = (xrState.hands || []).find((candidate) => (
      candidate?.userData?.xrHandedness === inputSource.handedness
    ));
  }
  if (!hand) return null;
  hand.userData.xrInputSource = inputSource;
  hand.userData.xrHandedness = inputSource.handedness || hand.userData.xrHandedness || '';
  const state = _handStateFor(hand);
  state.handedness = hand.userData.xrHandedness || state.handedness || '';
  return state;
}

function _firstInteractiveRayHit() {
  const hits = xrState.raycaster?.intersectObjects?.(_interactiveRayMeshes(), false) || [];
  for (const hit of hits) {
    const described = _describeInteractiveRayHit(hit);
    if (described) return described;
  }
  return null;
}

function _updateTargetRayVisuals() {
  if (!xrState.raycaster || !xrState.xrRig) return;
  const localPoint = xrState.panelDragWorldPoint;
  const cameraWorld = xrState.panelCameraWorld;
  for (const controller of xrState.targetRayControllers || []) {
    const visual = controller?.userData?.xrTargetVisual;
    if (!visual?.line) continue;
    if (!_setRayFromController(controller)) {
      visual.line.visible = false;
      if (visual.reticle) visual.reticle.visible = false;
      continue;
    }
    xrState.raycaster.far = XR_HAND_AIM_MAX_DISTANCE_M;
    const hit = _firstInteractiveRayHit();
    let placementPoint = null;
    let hasHit = !!hit?.point;
    let actionable = !!hit?.button;
    if (!hasHit && _isMixedRealityAvatarPlacementMode()) {
      placementPoint = _updateMixedRealityPlacementCandidateFromCurrentRay('controller-ray');
      hasHit = !!placementPoint;
      actionable = !!placementPoint;
    }
    const targetPoint = hit?.point || placementPoint;
    const distance = hasHit
      ? Math.max(0.08, Math.min(XR_HAND_AIM_MAX_DISTANCE_M, hit?.distance || XR_HAND_AIM_IDLE_DISTANCE_M))
      : XR_HAND_AIM_IDLE_DISTANCE_M;
    const rayDistance = placementPoint
      ? Math.max(0.08, Math.min(XR_HAND_AIM_MAX_DISTANCE_M, xrState.rayOrigin.distanceTo(placementPoint)))
      : distance;
    const isHandRay = _isHandInputSource(controller.userData.xrInputSource);
    visual.line.scale.z = rayDistance;
    visual.line.visible = hasHit || !isHandRay;
    if (visual.lineMat) {
      visual.lineMat.opacity = hasHit ? (actionable ? 0.92 : 0.42) : 0.16;
      visual.lineMat.color.setHex(actionable ? 0x84ffd8 : (isHandRay ? 0x8bdcff : 0x8aa8ff));
    }

    if (visual.reticle && hasHit && actionable && targetPoint && localPoint) {
      localPoint.copy(targetPoint);
      xrState.xrRig.worldToLocal(localPoint);
      visual.reticle.position.copy(localPoint);
      if (xrState.camera && cameraWorld) {
        xrState.camera.getWorldPosition(cameraWorld);
        visual.reticle.lookAt(cameraWorld);
      }
      visual.reticle.scale.setScalar(1.08);
      if (visual.reticleMat) {
        visual.reticleMat.color.setHex(0x84ffd8);
        visual.reticleMat.opacity = 0.82;
      }
      visual.reticle.visible = true;
    } else if (visual.reticle) {
      visual.reticle.visible = false;
    }
  }
}

function _hasPhysicalControllerRay() {
  return (xrState.targetRayControllers || []).some((controller) => (
    controller?.userData?.xrInputSource && !controller.userData.xrIsHand
  ));
}

function _isControllerlessComfortActive() {
  return isInVR() && !_hasPhysicalControllerRay();
}

function _setRayFromGaze() {
  if (!xrState.camera || !xrState.raycaster || !xrState.rayOrigin || !xrState.rayDirection) return false;
  xrState.camera.updateMatrixWorld?.(true);
  xrState.camera.getWorldPosition(xrState.rayOrigin);
  xrState.camera.getWorldDirection(xrState.rayDirection);
  if (xrState.rayDirection.lengthSq() < 0.0001) return false;
  xrState.rayDirection.normalize();
  xrState.raycaster.near = 0;
  xrState.raycaster.far = XR_GAZE_MAX_DISTANCE_M;
  xrState.raycaster.set(xrState.rayOrigin, xrState.rayDirection);
  return true;
}

function _mixedRealityPointFromGaze(placement, out) {
  if (!placement || !out || !_setRayFromGaze()) return false;
  const floorY = 0;
  let distance = XR_GAZE_IDLE_DISTANCE_M;
  if (Math.abs(xrState.rayDirection.y) > 0.025) {
    const t = (floorY - xrState.rayOrigin.y) / xrState.rayDirection.y;
    if (Number.isFinite(t) && t > 0.22 && t < 5.0) distance = t;
  }
  distance = Math.max(0.45, Math.min(4.8, distance));
  out.copy(xrState.rayDirection).multiplyScalar(distance).add(xrState.rayOrigin);
  out.y = floorY;
  return _sanitizeMixedRealityPlacementPoint(placement, out);
}

function _updateMixedRealityPlacementCandidateFromGaze(source = 'gaze') {
  const placement = xrState.mrPlacement;
  if (!placement || xrState.xrMode !== XR_MODE_MR) return null;
  if (!_mixedRealityPointFromGaze(placement, placement.rayPoint)) return null;
  _setMixedRealityPlacementCandidate(placement, placement.rayPoint, source);
  return placement.candidatePosition;
}

function _gazeHitKey(hit) {
  if (!hit) return '';
  if (hit.type === 'mr-placement') return 'mr-placement';
  if (hit.type === 'hand-menu') return `hand-menu:${hit.button?.action || 'panel'}`;
  if (hit.type === 'hub') return `hub:${hit.button?.action || 'panel'}`;
  if (hit.type === 'surface') {
    return `surface:${hit.panel?.surface?.action || ''}:${hit.button?.kind || 'body'}:${hit.button?.action || ''}`;
  }
  return hit.type || '';
}

function _canGazeDwellSelect(hit) {
  if (!hit) return false;
  if (!_isGazeDwellSelectEnabled()) return false;
  if (hit.type === 'mr-placement') return true;
  if (hit.type === 'hand-menu') return !!hit.button;
  if (hit.type === 'hub') return !!hit.button && hit.button.kind !== 'move-hub';
  if (hit.type === 'surface') return !!hit.button && hit.button.kind !== 'move';
  return false;
}

function _selectGazeTarget(source = 'gaze') {
  const hit = xrState.gazeHit;
  if (!hit) return false;
  const selected = hit.type === 'mr-placement'
    ? _placeMixedRealityAvatarFromGaze(source)
    : _activateInteractiveHit(hit, source);
  if (selected) xrState.lastComfortSelectAt = performance.now();
  return selected;
}

function _updateGazeComfort() {
  const cursor = xrState.gazeCursor;
  if (!cursor) return;
  if (!_isControllerlessComfortActive() || !_isGazeFocusVisible() || !_setRayFromGaze()) {
    cursor.visible = false;
    xrState.gazeHit = null;
    xrState.gazeTargetKey = '';
    xrState.gazeTargetStartedAt = 0;
    return;
  }

  const hit = _isGazeDwellSelectEnabled() ? _firstInteractiveRayHit() : null;
  let placementPoint = null;
  let gazeHit = hit;
  if (!gazeHit && _isMixedRealityAvatarPlacementMode()) {
    placementPoint = _updateMixedRealityPlacementCandidateFromGaze('gaze');
    if (placementPoint) {
      gazeHit = {
        type: 'mr-placement',
        point: placementPoint,
        distance: xrState.rayOrigin.distanceTo(placementPoint),
      };
    }
  }

  const targetPoint = gazeHit?.point || placementPoint;
  xrState.gazeHit = gazeHit || null;
  if (!targetPoint || !xrState.xrRig) {
    cursor.visible = false;
    xrState.gazeTargetKey = '';
    xrState.gazeTargetStartedAt = 0;
    return;
  }

  const now = performance.now();
  const key = _gazeHitKey(gazeHit);
  if (key !== xrState.gazeTargetKey) {
    xrState.gazeTargetKey = key;
    xrState.gazeTargetStartedAt = now;
    xrState.gazeDwellArmed = true;
  }

  xrState.panelDragWorldPoint.copy(targetPoint);
  xrState.xrRig.worldToLocal(xrState.panelDragWorldPoint);
  cursor.position.copy(xrState.panelDragWorldPoint);
  if (xrState.camera && xrState.panelCameraWorld) {
    xrState.camera.getWorldPosition(xrState.panelCameraWorld);
    cursor.lookAt(xrState.panelCameraWorld);
  }
  const dwellSelectable = key && _canGazeDwellSelect(gazeHit);
  const progress = dwellSelectable
    ? Math.max(0, Math.min(1, (now - xrState.gazeTargetStartedAt) / XR_GAZE_DWELL_MS))
    : 0;
  cursor.scale.setScalar(1 + progress * 0.42);
  if (xrState.gazeCursorMat) {
    xrState.gazeCursorMat.color.setHex(dwellSelectable ? 0x84ffd8 : 0x8bdcff);
    xrState.gazeCursorMat.opacity = dwellSelectable ? 0.76 + progress * 0.22 : 0.46;
  }
  cursor.visible = true;

  if (
    key
    && xrState.gazeDwellArmed
    && dwellSelectable
    && now - xrState.gazeTargetStartedAt >= XR_GAZE_DWELL_MS
    && now - xrState.lastGazeDwellAt >= XR_GAZE_DWELL_COOLDOWN_MS
  ) {
    xrState.gazeDwellArmed = false;
    xrState.lastGazeDwellAt = now;
    _selectGazeTarget('gaze-dwell');
  }
}

function _updateXrInteractions() {
  if (xrState.hudCtx && xrState.hudTexture) {
    _refreshHUD(xrState.hudCtx, xrState.hudTexture);
  }
  _updateTargetRayVisuals();
  _updateGazeComfort();
  _updateHandInputs();
  _updatePanelDrag();
  _updateHubDrag();
  _refreshLiveSurfacePanels();
  _updateMixedRealityCompanion();
  _updateWebEmbedAnchor();
  _updateHandDebugPanel();
  // Contact reactor: noop when not presenting or when the reactor was
  // never initialized (e.g. VRM didn't have a BodyMesh).
  _tickContactReactor();
}

let _contactLastTickMs = 0;
function _tickContactReactor() {
  const now = performance.now();
  const dtMs = _contactLastTickMs ? (now - _contactLastTickMs) : 16;
  _contactLastTickMs = now;
  try { tickXRContact(dtMs); }
  catch (err) { console.debug('[xr] contact tick failed:', err?.message); }
  // Compliance runs immediately after the reactor so it picks up the
  // same per-frame hand positions, and writes bone deltas before the
  // scene renders (vrm.update bakes the final pose into the skinned
  // mesh after this returns).
  try { tickXRCompliance(dtMs); }
  catch (err) { console.debug('[xr] compliance tick failed:', err?.message); }

  // Rapier ticks AFTER compliance so its kinematic targets read the
  // post-compliance bone transforms. Deltas exposed via getBoneDeltas()
  // for downstream blending.
  try { tickXRRapier(dtMs); }
  catch (err) { console.debug('[xr] rapier tick failed:', err?.message); }

  // Coordinator pushes server settings → channel instance properties.
  // Cheap (property writes only); also runs on its own 5s refresh timer.
  try { tickBodyPhysicsCoordinator(); }
  catch (err) { console.debug('[xr] coordinator tick failed:', err?.message); }
}

function _refreshLiveSurfacePanels() {
  if (!xrState.spatialPanels?.size) return;
  const now = performance.now();
  if (now - xrState.lastLiveSurfaceRefreshAt < XR_SURFACE_PANEL_REFRESH_MS) return;
  xrState.lastLiveSurfaceRefreshAt = now;
  for (const panel of xrState.spatialPanels.values()) {
    if (panel?.group?.visible === false) continue;
    if (panel.browser?.status === 'active') {
      if (
        panel.browser.streamStatus === 'idle'
        || (panel.browser.streamStatus === 'fallback'
          && Number(panel.browser.streamReconnects || 0) < XR_BROWSER_PANEL_STREAM_MAX_RECONNECTS)
      ) {
        _connectBrowserPanelStream(panel);
      }
      _updateBrowserPanelFrame(panel);
    }
    _refreshSurfacePanel(panel);
  }
}

function _projectWorldToOverlay(pointWorld, camera, width, height) {
  const ndc = pointWorld.clone().project(camera);
  if (!Number.isFinite(ndc.x) || !Number.isFinite(ndc.y) || ndc.z < -1 || ndc.z > 1) {
    return null;
  }
  return {
    x: (ndc.x * 0.5 + 0.5) * width,
    y: (0.5 - ndc.y * 0.5) * height,
    z: ndc.z,
  };
}

function _updateWebEmbedAnchor() {
  if (!xrState.active || !xrState.camera || !xrState.renderer) return;
  // Do not drive a DOM/iframe overlay from an XR eye camera. A projected
  // 2D overlay is monocular on Quest and can land in a different apparent
  // place per eye. The actual headset UI stays on stereo WebGL planes —
  // UNLESS the experiment toggle (_useDomOverlayPanels) is on, in which
  // case we intentionally use the monocular layer for everything.
  if (xrState.renderer.xr?.isPresenting && !_useDomOverlayPanels()) {
    if (xrState.webEmbedLastAnchorKey !== 'immersive-suppressed') {
      xrState.webEmbedLastAnchorKey = 'immersive-suppressed';
      setXrWebEmbedAnchor({ visible: false });
    }
    return;
  }
  const panel = xrState.spatialPanels?.get?.(xrState.modeHubActiveAction);
  if (!panel?.mesh || !panel?.group?.visible) {
    if (xrState.webEmbedLastAnchorKey !== 'hidden') {
      xrState.webEmbedLastAnchorKey = 'hidden';
      setXrWebEmbedAnchor({ visible: false });
    }
    return;
  }

  const renderCamera = xrState.renderer.xr?.isPresenting
    ? xrState.renderer.xr.getCamera(xrState.camera)
    : xrState.camera;
  const camera = renderCamera?.cameras?.[0] || renderCamera || xrState.camera;
  const width = window.innerWidth || xrState.renderer.domElement?.clientWidth || 1280;
  const height = window.innerHeight || xrState.renderer.domElement?.clientHeight || 720;
  if (!camera || !width || !height) return;

  panel.mesh.updateWorldMatrix?.(true, false);
  camera.updateMatrixWorld?.(true);
  camera.updateProjectionMatrix?.();

  const geom = panel.mesh.geometry?.parameters || {};
  const meshW = Number(geom.width || 1);
  const meshH = Number(geom.height || 0.62);
  const THREE = xrState.THREE;
  const corners = [
    new THREE.Vector3(-meshW / 2, meshH / 2, 0),
    new THREE.Vector3(meshW / 2, meshH / 2, 0),
    new THREE.Vector3(meshW / 2, -meshH / 2, 0),
    new THREE.Vector3(-meshW / 2, -meshH / 2, 0),
  ].map((v) => {
    v.applyMatrix4(panel.mesh.matrixWorld);
    return _projectWorldToOverlay(v, camera, width, height);
  }).filter(Boolean);

  if (corners.length < 4) {
    if (xrState.webEmbedLastAnchorKey !== 'offscreen') {
      xrState.webEmbedLastAnchorKey = 'offscreen';
      setXrWebEmbedAnchor({ visible: false });
    }
    return;
  }

  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const projectedWidth = maxX - minX;
  const projectedHeight = maxY - minY;
  const anchor = {
    visible: true,
    left: minX - Math.max(20, projectedWidth * 0.10),
    top: minY - 70,
    width: Math.max(520, projectedWidth * 1.20),
    height: Math.max(360, projectedHeight * 1.35 + 72),
  };
  const key = [
    panel.surface.action,
    Math.round(anchor.left / 4),
    Math.round(anchor.top / 4),
    Math.round(anchor.width / 4),
    Math.round(anchor.height / 4),
  ].join(':');
  if (key === xrState.webEmbedLastAnchorKey) return;
  xrState.webEmbedLastAnchorKey = key;
  setXrWebEmbedAnchor(anchor);
}

function _activateHubButton(button) {
  if (!button) return false;
  if (button.action === 'exit') {
    if (!_confirmHubAction(button.action, 'Pinch Exit VR again to confirm.')) return true;
    xrSession.recordEvent(xrState.serverSessionId, 'hub_exit_selected', {});
    exitVR().catch(() => {});
    return true;
  }
  if (button.action === 'switch-xr-mode') {
    const targetMode = xrState.xrMode === XR_MODE_MR ? XR_MODE_VR : XR_MODE_MR;
    if (!_confirmHubAction(button.action, `Pinch ${button.label} again to confirm.`)) return true;
    xrSession.recordEvent(xrState.serverSessionId, 'hub_switch_mode_selected', {
      from: xrState.xrMode,
      to: targetMode,
    });
    try {
      window.dispatchEvent(new CustomEvent('augmentum:xr-switch-mode', {
        detail: { mode: targetMode, source: 'xr-hub', sessionId: xrState.serverSessionId },
      }));
    } catch { /* listener errors are non-fatal — exitVR below still proceeds */ }
    exitVR().catch(() => {});
    return true;
  }
  if (button.action === 'place-avatar') {
    xrState.pendingHubConfirm = null;
    _enterMixedRealityAvatarPlacementMode('hub-button', { hideAvatar: false });
    return true;
  }
  xrState.pendingHubConfirm = null;
  _openHubSurface(button);
  return true;
}

function _confirmHubAction(action, message) {
  const now = performance.now();
  const pending = xrState.pendingHubConfirm;
  if (pending?.action === action && now - pending.at < 2500) {
    xrState.pendingHubConfirm = null;
    return true;
  }
  xrState.pendingHubConfirm = { action, at: now };
  xrState.modeHubStatus = message;
  _refreshModeHub(true);
  xrSession.recordEvent(xrState.serverSessionId, 'hub_action_confirm_requested', { action });
  return false;
}

function _handStateFor(hand) {
  let state = xrState.handStates.get(hand);
  if (state) return state;
  const THREE = xrState.THREE;
  state = {
    hand,
    index: Number(hand?.userData?.xrHandIndex || 0),
    handedness: hand?.userData?.xrHandedness || '',
    pinching: false,
    pinchStartedAt: 0,
    pinchSource: '',
    palmStartedAt: 0,
    lastPalmSummonAt: 0,
    palmAnchorWorld: new THREE.Vector3(),
    palmFacingScore: 0,
    palmMenuReady: false,
    nativeSelectActive: false,
    nativeButtonActive: false,
    nativeActiveButtons: [],
    pinchWorld: new THREE.Vector3(),
    indexWorld: new THREE.Vector3(),
    thumbWorld: new THREE.Vector3(),
    middleWorld: new THREE.Vector3(),
    wristWorld: new THREE.Vector3(),
    prevWristWorld: new THREE.Vector3(),
    indexBaseWorld: new THREE.Vector3(),
    pinkyBaseWorld: new THREE.Vector3(),
    waveStartedAt: 0,
    waveLastLateral: null,
    waveLastDir: 0,
    waveTurns: 0,
    waveMin: 0,
    waveMax: 0,
    waveTravel: 0,
    waveSamples: [],
    lastWaveAt: 0,
    lastAvatarProximityAt: 0,
    lastAvatarProximityZone: '',
  };
  xrState.handStates.set(hand, state);
  return state;
}

function _handJoint(hand, names) {
  for (const name of names) {
    const joint = hand?.joints?.[name] || hand?.getObjectByName?.(name);
    if (joint) return joint;
  }
  return null;
}

function _readHandState(state) {
  const hand = state.hand;
  state.handedness = hand?.userData?.xrHandedness || state.handedness || '';
  const thumb = _handJoint(hand, ['thumb-tip']);
  const index = _handJoint(hand, ['index-finger-tip']);
  if (!thumb || !index || thumb.visible === false || index.visible === false) return false;
  state.prevWristWorld.copy(state.wristWorld);
  thumb.getWorldPosition(state.thumbWorld);
  index.getWorldPosition(state.indexWorld);
  state.pinchWorld.copy(state.thumbWorld).add(state.indexWorld).multiplyScalar(0.5);

  const middle = _handJoint(hand, ['middle-finger-tip']);
  const wrist = _handJoint(hand, ['wrist']);
  const indexBase = _handJoint(hand, ['index-finger-metacarpal', 'index-finger-proximal']);
  const pinkyBase = _handJoint(hand, ['pinky-finger-metacarpal', 'little-finger-metacarpal']);
  if (middle) middle.getWorldPosition(state.middleWorld);
  else state.middleWorld.copy(state.indexWorld);
  if (wrist) wrist.getWorldPosition(state.wristWorld);
  else state.wristWorld.copy(state.pinchWorld);
  if (indexBase) indexBase.getWorldPosition(state.indexBaseWorld);
  else state.indexBaseWorld.copy(state.indexWorld);
  if (pinkyBase) pinkyBase.getWorldPosition(state.pinkyBaseWorld);
  else state.pinkyBaseWorld.copy(state.middleWorld);
  return true;
}

function _disposeTargetRayVisuals() {
  for (const controller of xrState.targetRayControllers || []) {
    const visual = controller?.userData?.xrTargetVisual;
    if (!visual) continue;
    visual.line?.parent?.remove(visual.line);
    visual.reticle?.parent?.remove(visual.reticle);
    visual.line?.geometry?.dispose();
    visual.lineMat?.dispose();
    visual.reticleGeo?.dispose();
    visual.reticleMat?.dispose();
    controller.userData.xrTargetVisual = null;
  }
  xrState.targetRayControllers = [];
}

function _buttonAtCanvasPoint(buttons, x, y) {
  return buttons?.find?.((b) => (
    x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h
  )) || null;
}

function _interactiveRayMeshes() {
  const meshes = [];
  if (xrState.handMenuPanel && xrState.handMenuGroup?.visible !== false) meshes.push(xrState.handMenuPanel);
  if (xrState.modeHubPanel && xrState.modeHubGroup?.visible !== false) meshes.push(xrState.modeHubPanel);
  for (const panel of xrState.spatialPanels?.values?.() || []) {
    if (panel?.mesh && panel?.group?.visible !== false) meshes.push(panel.mesh);
  }
  return meshes;
}

function _describeInteractiveRayHit(hit) {
  if (!hit?.object) return null;
  if (hit.object.userData?.xrHandMenu) {
    const uv = hit.uv;
    const button = uv && xrState.handMenuCanvas
      ? _buttonAtCanvasPoint(
        xrState.handMenuButtons,
        uv.x * xrState.handMenuCanvas.width,
        (1 - uv.y) * xrState.handMenuCanvas.height,
      )
      : null;
    return {
      type: 'hand-menu',
      button,
      distance: hit.distance,
      point: hit.point,
    };
  }
  if (hit.object.userData?.xrModeHub) {
    const uv = hit.uv;
    const button = uv && xrState.modeHubCanvas
      ? _buttonAtCanvasPoint(
        xrState.modeHubButtons,
        uv.x * xrState.modeHubCanvas.width,
        (1 - uv.y) * xrState.modeHubCanvas.height,
      )
      : null;
    return {
      type: 'hub',
      button,
      distance: hit.distance,
      point: hit.point,
    };
  }

  const action = hit.object.userData?.xrSurfacePanel;
  const panel = action ? xrState.spatialPanels.get(action) : null;
  if (!panel) return null;
  const uv = hit.uv;
  const x = uv ? uv.x * panel.canvas.width : 0;
  const y = uv ? (1 - uv.y) * panel.canvas.height : 0;
  const button = uv
    ? _buttonAtCanvasPoint(
      panel.buttons,
      x,
      y,
    )
    : null;
  return {
    type: 'surface',
    panel,
    button,
    distance: hit.distance,
    point: hit.point,
    x,
    y,
    uv,
  };
}

function _isHandPinching(state) {
  const d = state.thumbWorld.distanceTo(state.indexWorld);
  return state.pinching ? d < XR_HAND_PINCH_END_M : d < XR_HAND_PINCH_START_M;
}

function _isHandOpenForUserSignal(state) {
  if (!state) return false;
  return (
    state.thumbWorld.distanceTo(state.indexWorld) > 0.06
    && state.thumbWorld.distanceTo(state.middleWorld) > 0.07
    && state.indexWorld.distanceTo(state.middleWorld) > 0.025
  );
}

function _isHandOpenEnoughForWave(state) {
  if (!state || state.pinching) return false;
  return (
    state.thumbWorld.distanceTo(state.indexWorld) > 0.043
    && state.thumbWorld.distanceTo(state.middleWorld) > 0.048
  );
}

function _updatePalmMenuPose(state, { selectionActive = false } = {}) {
  if (!state || !xrState.camera) return false;
  const thumbIndex = state.thumbWorld.distanceTo(state.indexWorld);
  const thumbMiddle = state.thumbWorld.distanceTo(state.middleWorld);
  const indexMiddle = state.indexWorld.distanceTo(state.middleWorld);
  const fingersReadable = thumbMiddle > 0.055 && indexMiddle > 0.022;
  const selectionOrOpenIndex = selectionActive || state.pinching || thumbIndex > 0.052;
  const a = xrState.handTempA;
  const b = xrState.handTempB;
  const normal = xrState.handTempC;
  const toCamera = xrState.handTempD;
  if (!a || !b || !normal || !toCamera) return false;
  a.copy(state.indexBaseWorld).sub(state.wristWorld);
  b.copy(state.pinkyBaseWorld).sub(state.wristWorld);
  if (a.lengthSq() < 0.0001 || b.lengthSq() < 0.0001) {
    state.palmFacingScore = 0;
    state.palmMenuReady = false;
    return false;
  }
  normal.crossVectors(a, b).normalize();
  xrState.camera.getWorldPosition(toCamera);
  toCamera.sub(state.wristWorld).normalize();
  state.palmFacingScore = Math.abs(normal.dot(toCamera));
  state.palmMenuReady = (
    fingersReadable
    && selectionOrOpenIndex
    && state.palmFacingScore >= XR_HAND_MENU_PALM_FACING_DOT
    && state.wristWorld.y > 0.52
  );
  return state.palmMenuReady;
}

function _trySummonHandMenuFromPalmPinch(state, now, source = 'hand-pinch') {
  if (!state || !xrState.camera || !xrState.modeHubGroup || xrState.handMenuGroup?.visible) return false;
  if (_isHandMenuOpenSuppressed(now)) return false;
  if (!_updatePalmMenuPose(state, { selectionActive: true })) return false;
  if (!state.palmStartedAt || now - state.palmStartedAt < XR_HAND_PALM_SUMMON_MS) return false;
  if (state.wristWorld.distanceTo(state.palmAnchorWorld) > XR_HAND_PALM_MAX_MOTION_M) return false;
  if (now - state.lastPalmSummonAt < XR_HAND_PALM_SUMMON_COOLDOWN_MS) return false;
  state.lastPalmSummonAt = now;
  state.palmStartedAt = 0;
  _recordXrUserSignal('open_palm_menu', {
    hand: state.index,
    handedness: state.handedness || '',
    source,
    palm_facing: _roundProbeValue(state.palmFacingScore, 2),
    confidence: Math.min(0.96, 0.72 + state.palmFacingScore * 0.18),
  }, { cooldownMs: XR_HAND_PALM_SUMMON_COOLDOWN_MS });
  _showHandMenuNearHand(state, source);
  return true;
}

function _tryOpenPalmPinchMenuForInputSource(inputSource = null, source = 'native-select', controller = null) {
  const state = _handStateForInputSource(inputSource, controller);
  if (!state || !_readHandState(state)) return false;
  return _trySummonHandMenuFromPalmPinch(state, performance.now(), source);
}

function _resetUserWaveState(state) {
  if (!state) return;
  state.waveStartedAt = 0;
  state.waveLastLateral = null;
  state.waveLastDir = 0;
  state.waveTurns = 0;
  state.waveMin = 0;
  state.waveMax = 0;
  state.waveTravel = 0;
  state.waveSamples = [];
}

function _xrGestureResponseMode() {
  try {
    const raw = window.localStorage?.getItem(XR_GESTURE_RESPONSE_MODE_KEY);
    return raw === 'passive' ? 'passive' : 'friendly';
  } catch {
    return 'friendly';
  }
}

function _userSignalSummary(type, detail = {}) {
  if (type === 'wave') return 'The user waved toward the avatar before speaking.';
  if (type === 'hand_near_avatar') {
    const zone = detail.zone ? ` near the avatar's ${detail.zone}` : ' near the avatar';
    return `The user's hand moved${zone}.`;
  }
  if (type === 'hand_contact_avatar') {
    const zone = detail.zone ? ` at the avatar's ${detail.zone}` : ' near the avatar';
    return `The user's hand entered close contact range${zone}.`;
  }
  if (type === 'open_palm_menu') return 'The user deliberately opened the XR Quick Menu.';
  return 'The headset detected a nonverbal user action.';
}

function _recordXrUserSignal(type, detail = {}, { cooldownMs = XR_USER_SIGNAL_COOLDOWN_MS } = {}) {
  if (!type) return false;
  const now = performance.now();
  const key = `${type}:${detail.hand ?? ''}:${detail.zone || ''}`;
  const lastAt = xrState.userSignalLastAt?.get?.(key) || 0;
  if (now - lastAt < cooldownMs) return false;
  xrState.userSignalLastAt?.set?.(key, now);
  const signal = {
    type,
    summary: _userSignalSummary(type, detail),
    at: new Date().toISOString(),
    mode: xrState.xrMode,
    presentation: xrState.sessionMode,
    confidence: Number(detail.confidence || 0.8),
    ...detail,
  };
  xrState.userSignals.push(signal);
  xrState.userSignals = xrState.userSignals.slice(-6);
  xrSession.recordEvent(xrState.serverSessionId, 'xr_user_signal', signal);
  try {
    window.dispatchEvent(new CustomEvent('augmentum:xr-user-signal', { detail: signal }));
  } catch { /* listener errors are non-fatal — signal already recorded server-side */ }
  return true;
}

function _maybeAvatarWaveBack(sourceSignal) {
  if (_xrGestureResponseMode() === 'passive') return;
  const now = performance.now();
  if (now - xrState.lastAvatarWaveBackAt < XR_USER_WAVE_COOLDOWN_MS) return;
  xrState.lastAvatarWaveBackAt = now;
  const conductor = appAvatarState?.conductor;
  if (conductor?.playById) {
    conductor.playById('hello', { explicit: true }).catch(() => {});
  } else if (appAvatarState?.presence?.onExplicitGesture) {
    appAvatarState.presence.onExplicitGesture('wave');
  } else {
    appAvatarState?.animator?.triggerGesture?.('wave');
  }
  xrSession.recordEvent(xrState.serverSessionId, 'xr_avatar_wave_back', {
    source: sourceSignal?.type || 'wave',
  });
}

function _updateUserWaveRecognition(state, now) {
  if (!state || !xrState.camera || state.pinching) return;
  if (!_isHandOpenEnoughForWave(state)) {
    _resetUserWaveState(state);
    return;
  }
  if (state.wristWorld.y < 0.70) {
    _resetUserWaveState(state);
    return;
  }
  const right = xrState.handTempA;
  if (!right) return;
  xrState.camera.updateMatrixWorld?.(true);
  right.setFromMatrixColumn(xrState.camera.matrixWorld, 0).normalize();
  const lateral = state.wristWorld.dot(right);
  const samples = Array.isArray(state.waveSamples) ? state.waveSamples : [];
  samples.push({ at: now, lateral });
  while (samples.length && now - samples[0].at > XR_USER_WAVE_WINDOW_MS) samples.shift();
  state.waveSamples = samples;
  if (samples.length < 4) {
    state.waveStartedAt = samples[0]?.at || now;
    state.waveLastLateral = lateral;
    return;
  }

  let min = samples[0].lateral;
  let max = samples[0].lateral;
  let travel = 0;
  let turns = 0;
  let lastDir = 0;
  for (let i = 1; i < samples.length; i++) {
    const value = samples[i].lateral;
    min = Math.min(min, value);
    max = Math.max(max, value);
    const delta = value - samples[i - 1].lateral;
    travel += Math.abs(delta);
    if (Math.abs(delta) < 0.009) continue;
    const dir = delta > 0 ? 1 : -1;
    if (lastDir && dir !== lastDir) turns += 1;
    lastDir = dir;
  }

  state.waveStartedAt = samples[0].at;
  state.waveLastLateral = lateral;
  state.waveLastDir = lastDir;
  state.waveTurns = turns;
  state.waveMin = min;
  state.waveMax = max;
  state.waveTravel = travel;
  const span = max - min;
  if (
    turns >= XR_USER_WAVE_MIN_REVERSALS
    && span >= XR_USER_WAVE_MIN_SPAN_M
    && travel >= XR_USER_WAVE_MIN_TRAVEL_M
    && now - state.lastWaveAt >= XR_USER_WAVE_COOLDOWN_MS
  ) {
    state.lastWaveAt = now;
    const signal = {
      hand: state.index,
      confidence: Math.min(0.98, 0.66 + span + turns * 0.08),
      span: _roundPoseValue(span),
      travel: _roundPoseValue(travel),
      reversals: turns,
    };
    _resetUserWaveState(state);
    if (_recordXrUserSignal('wave', signal, { cooldownMs: XR_USER_WAVE_COOLDOWN_MS })) {
      _maybeAvatarWaveBack({ type: 'wave' });
    }
  }
}

function _avatarWorldPointForZone(zone, out) {
  if (!out || !xrState.vrm?.scene) return false;
  const humanoid = xrState.vrm.humanoid;
  const boneName = zone === 'head'
    ? 'head'
    : (zone === 'torso' ? 'chest' : 'hips');
  const node = humanoid?.getNormalizedBoneNode?.(boneName)
    || humanoid?.getNormalizedBoneNode?.('upperChest')
    || xrState.vrm.scene;
  node.updateMatrixWorld?.(true);
  node.getWorldPosition(out);
  return true;
}

function _updateHandAvatarProximity(state, now) {
  if (!state || !xrState.vrm?.scene || xrState.vrm.scene.visible === false) return;
  const point = xrState.handTempA;
  if (!point) return;
  const zones = ['head', 'torso'];
  let bestZone = '';
  let bestDistance = Infinity;
  for (const zone of zones) {
    if (!_avatarWorldPointForZone(zone, point)) continue;
    const wristDistance = state.wristWorld.distanceTo(point);
    const indexDistance = state.indexWorld.distanceTo(point);
    const distance = Math.min(wristDistance, indexDistance);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestZone = zone;
    }
  }
  if (!bestZone || bestDistance > XR_USER_HAND_NEAR_AVATAR_M) {
    state.lastAvatarProximityZone = '';
    return;
  }
  const type = bestDistance <= XR_USER_HAND_CONTACT_AVATAR_M
    ? 'hand_contact_avatar'
    : 'hand_near_avatar';
  const zoneKey = `${type}:${bestZone}`;
  if (
    state.lastAvatarProximityZone === zoneKey
    && now - state.lastAvatarProximityAt < 2600
  ) {
    return;
  }
  state.lastAvatarProximityAt = now;
  state.lastAvatarProximityZone = zoneKey;
  _recordXrUserSignal(type, {
    hand: state.index,
    zone: bestZone,
    distance_m: _roundPoseValue(bestDistance),
    confidence: bestDistance <= XR_USER_HAND_CONTACT_AVATAR_M ? 0.9 : 0.72,
  }, { cooldownMs: type === 'hand_contact_avatar' ? 2600 : 4200 });
}

function _updateHandUserSignalRecognition(state, now) {
  _updateUserWaveRecognition(state, now);
  _updateHandAvatarProximity(state, now);
}

function _updateHandInputs() {
  const now = performance.now();
  _updateXrInputProbes(now);
  for (const hand of xrState.hands || []) {
    const state = _handStateFor(hand);
    if (!_readHandState(state)) {
      state.pinching = false;
      state.pinchStartedAt = 0;
      state.palmStartedAt = 0;
      state.nativeSelectActive = false;
      state.nativeButtonActive = false;
      state.nativeActiveButtons = [];
      continue;
    }
    if (
      xrState.handMenuGroup?.visible
      && xrState.handMenuHandIndex != null
      && xrState.handMenuHandIndex === state.index
    ) {
      _updateHandMenuPoseFromState(state);
    }
    const nativeIntent = _nativeIntentForHandState(state, now);
    const nativeSelectActive = !!nativeIntent?.nativeSelectActive;
    const nativeButtonActive = !!nativeIntent?.nativeButtonActive;
    const geometryPinching = _isHandPinching(state);
    state.nativeSelectActive = nativeSelectActive;
    state.nativeButtonActive = nativeButtonActive;
    state.nativeActiveButtons = nativeIntent?.activeButtons || [];
    const pinching = nativeSelectActive || nativeButtonActive || geometryPinching;
    const pinchSource = nativeSelectActive
      ? 'native-select'
      : (nativeButtonActive ? 'native-button' : 'geometry-pinch');
    if (pinching && !state.pinching) {
      state.pinching = true;
      state.pinchStartedAt = now;
      state.pinchSource = pinchSource;
      if (_trySummonHandMenuFromPalmPinch(state, now, pinchSource)) {
        continue;
      }
      if (pinchSource !== 'native-select') {
        _handleHandPinchComfortSelect(state, now);
      }
    } else if (!pinching && state.pinching) {
      state.pinching = false;
      state.pinchStartedAt = 0;
      state.pinchSource = '';
    }
    _updatePalmSummon(state, now);
    _updateHandUserSignalRecognition(state, now);
  }
}

function _handleHandPinchComfortSelect(state, now) {
  if (!_isControllerlessComfortActive()) return false;
  if (!_isGazeDwellSelectEnabled()) return false;
  if (now - xrState.lastNativeSelectAt < 220) return false;
  if (now - xrState.lastComfortSelectAt < XR_HAND_PINCH_SELECT_COOLDOWN_MS) return false;
  const selected = _selectGazeTarget('hand-pinch');
  if (selected) {
    xrState.lastComfortSelectAt = now;
    xrSession.recordEvent(xrState.serverSessionId, 'hand_pinch_comfort_select', {
      hand: state.index,
      target: _gazeHitKey(xrState.gazeHit),
    });
  }
  return selected;
}

function _updatePalmSummon(state, now) {
  if (!xrState.camera || !xrState.modeHubGroup || xrState.handMenuGroup?.visible || _isHandMenuOpenSuppressed(now)) {
    state.palmStartedAt = 0;
    return;
  }
  const ready = _updatePalmMenuPose(state, { selectionActive: false });
  if (!ready || state.pinching) {
    state.palmStartedAt = 0;
    return;
  }
  if (!state.palmStartedAt) {
    state.palmStartedAt = now;
    state.palmAnchorWorld.copy(state.wristWorld);
    return;
  }
  if (state.wristWorld.distanceTo(state.palmAnchorWorld) > XR_HAND_PALM_MAX_MOTION_M) {
    state.palmStartedAt = now;
    state.palmAnchorWorld.copy(state.wristWorld);
    return;
  }
}

function _summonHubNearView(source = 'hand') {
  if (!xrState.camera || !xrState.xrRig || !xrState.modeHubGroup) return;
  if (source === 'hand-palm' && xrState.hubUserPlaced) {
    xrState.modeHubStatus = 'Hub stays where you placed it. Grab the title bar to move it.';
    _refreshModeHub(true);
    return;
  }
  const cameraWorld = xrState.handTempA;
  const forward = xrState.handTempB;
  const target = xrState.handTempC;
  if (!cameraWorld || !forward || !target) return;
  xrState.camera.getWorldPosition(cameraWorld);
  xrState.camera.getWorldDirection(forward);
  target.copy(cameraWorld).add(forward.multiplyScalar(1.12));
  xrState.xrRig.worldToLocal(target);
  target.y = Math.max(0.92, Math.min(1.48, target.y - 0.04));
  xrState.modeHubGroup.visible = true;
  xrState.modeHubGroup.position.copy(target);
  xrState.modeHubGroup.lookAt(cameraWorld);
  xrState.modeHubStatus = 'Hub summoned. Pinch a surface or grab a panel.';
  _refreshModeHub(true);
  _refreshHandMenu(true);
  xrSession.recordEvent(xrState.serverSessionId, 'hub_summoned_by_hand', { source });
}

function _panelPoseForSurface(surface) {
  const base = XR_PANEL_POSES[surface?.placement] || XR_PANEL_POSES.default;
  return { ...base };
}

function _formatPanelLabel(value) {
  return formatXrActionLabel(value);
}

function _roundPoseValue(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}

function _panelLayoutPreset(id) {
  const key = String(id || 'work').trim().toLowerCase();
  return XR_PANEL_LAYOUT_PRESETS[key] || XR_PANEL_LAYOUT_PRESETS.work;
}

function _normalizePanelLayout(layout = {}) {
  const preset = _panelLayoutPreset(layout.preset || layout.id);
  const curve = Number.isFinite(Number(layout.curveM ?? layout.curve))
    ? Number(layout.curveM ?? layout.curve)
    : Number(preset.curveM || 0);
  return {
    preset: preset.id,
    label: preset.label,
    logicalW: Math.max(480, Math.min(1400, Number(layout.logicalW || preset.logicalW || preset.canvasW))),
    logicalH: Math.max(360, Math.min(1200, Number(layout.logicalH || preset.logicalH || preset.canvasH))),
    canvasW: Math.max(720, Math.min(2048, Number(layout.canvasW || preset.canvasW))),
    canvasH: Math.max(540, Math.min(1800, Number(layout.canvasH || preset.canvasH))),
    widthM: Math.max(0.42, Math.min(1.9, Number(layout.widthM || preset.widthM))),
    heightM: Math.max(0.28, Math.min(1.15, Number(layout.heightM || preset.heightM))),
    curveM: Math.max(0, Math.min(XR_PANEL_MAX_CURVE_M, curve)),
    browserW: Math.max(720, Math.min(1920, Number(layout.browserW || preset.browserW))),
    browserH: Math.max(480, Math.min(1600, Number(layout.browserH || preset.browserH))),
  };
}

function _serializablePanelLayout(layout = {}) {
  const normalized = _normalizePanelLayout(layout);
  return {
    preset: normalized.preset,
    curveM: _roundPoseValue(normalized.curveM),
  };
}

function _panelCanvasScale(panel) {
  const layout = _normalizePanelLayout(panel?.layout || {});
  const canvas = panel?.canvas || {};
  const logicalW = Math.max(1, Number(layout.logicalW || layout.canvasW || canvas.width || 1));
  const logicalH = Math.max(1, Number(layout.logicalH || layout.canvasH || canvas.height || 1));
  return {
    sx: Math.max(0.1, Number(canvas.width || logicalW) / logicalW),
    sy: Math.max(0.1, Number(canvas.height || logicalH) / logicalH),
    logicalW,
    logicalH,
  };
}

function _scaleCanvasButton(button, sx, sy) {
  if (!button) return button;
  return {
    ...button,
    x: button.x * sx,
    y: button.y * sy,
    w: button.w * sx,
    h: button.h * sy,
  };
}

function _scaleCanvasButtons(buttons, sx, sy) {
  return (buttons || []).map((button) => _scaleCanvasButton(button, sx, sy));
}

function _panelLayoutKey(panel) {
  const layout = _normalizePanelLayout(panel?.layout || {});
  return [
    layout.preset,
    layout.logicalW,
    layout.logicalH,
    layout.canvasW,
    layout.canvasH,
    _roundPoseValue(layout.widthM),
    _roundPoseValue(layout.heightM),
    _roundPoseValue(layout.curveM),
  ].join(':');
}

function _createSurfacePanelGeometry(layout) {
  const THREE = xrState.THREE;
  const normalized = _normalizePanelLayout(layout);
  const segments = normalized.curveM > 0.001 ? 28 : 1;
  const geometry = new THREE.PlaneGeometry(normalized.widthM, normalized.heightM, segments, 1);
  if (normalized.curveM > 0.001) {
    const pos = geometry.attributes.position;
    const halfW = normalized.widthM / 2;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const t = halfW > 0 ? Math.max(-1, Math.min(1, x / halfW)) : 0;
      const z = normalized.curveM * (1 - Math.cos(Math.abs(t) * Math.PI / 2));
      pos.setZ(i, z);
    }
    pos.needsUpdate = true;
    geometry.computeVertexNormals();
  }
  return geometry;
}

function _configurePanelTexture(texture) {
  const THREE = xrState.THREE;
  if (!texture || !THREE) return texture;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  const maxAnisotropy = xrState.renderer?.capabilities?.getMaxAnisotropy?.();
  if (Number.isFinite(maxAnisotropy) && maxAnisotropy > 1) {
    texture.anisotropy = Math.min(8, maxAnisotropy);
  }
  return texture;
}

function _surfacePanelPose(target) {
  const group = target?.group || target;
  return {
    open: true,
    x: _roundPoseValue(group.position.x),
    y: _roundPoseValue(group.position.y),
    z: _roundPoseValue(group.position.z),
    rotY: _roundPoseValue(group.rotation.y),
    scale: _roundPoseValue(group.scale.x || 1),
    ...(target?.layout ? { layout: _serializablePanelLayout(target.layout) } : {}),
  };
}

function _applySurfacePanelPose(group, pose) {
  const base = pose || XR_PANEL_POSES.default;
  group.position.set(
    Number(base.x ?? 0),
    Number(base.y ?? 1.16),
    Number(base.z ?? -0.88),
  );
  group.rotation.set(0, Number(base.rotY ?? 0), 0);
  const scale = Math.max(
    XR_PANEL_MIN_SCALE,
    Math.min(XR_PANEL_MAX_SCALE, Number(base.scale ?? 1) || 1),
  );
  group.scale.setScalar(scale);
}

function _positionPanelForLayoutPreset(panel, presetId) {
  if (!panel?.group || !xrState.camera || !xrState.xrRig) return false;
  const preset = _panelLayoutPreset(presetId);
  const cameraWorld = xrState.handTempA;
  const forward = xrState.handTempB;
  const target = xrState.handTempC;
  if (!cameraWorld || !forward || !target) return false;
  xrState.camera.updateMatrixWorld?.(true);
  xrState.camera.getWorldPosition(cameraWorld);
  xrState.camera.getWorldDirection(forward);
  forward.y = Math.max(-0.12, Math.min(0.12, forward.y));
  if (forward.lengthSq() < 0.0001) return false;
  forward.normalize();
  target.copy(cameraWorld).add(forward.multiplyScalar(preset.viewDistanceM || 1.1));
  xrState.xrRig.worldToLocal(target);
  target.y = Math.max(0.72, Math.min(1.62, Number(preset.viewY || 1.18)));
  panel.group.position.copy(target);
  panel.group.lookAt(cameraWorld);
  return true;
}

function _applyPanelLayout(panel, layout, { persist = false, moveToPreset = false, restartBrowser = true } = {}) {
  if (!panel?.mesh || !panel?.canvas || !panel?.texture) return false;
  const next = _normalizePanelLayout(layout);
  const prevKey = _panelLayoutKey(panel);
  const nextKey = _panelLayoutKey({ layout: next });
  panel.layout = next;

  if (panel.canvas.width !== next.canvasW || panel.canvas.height !== next.canvasH) {
    panel.canvas.width = next.canvasW;
    panel.canvas.height = next.canvasH;
  }
  const oldGeometry = panel.mesh.geometry;
  panel.mesh.geometry = _createSurfacePanelGeometry(next);
  oldGeometry?.dispose?.();
  panel.texture.needsUpdate = true;
  panel.lastKey = '';
  panel.lastContentVersion = 0;

  if (moveToPreset) _positionPanelForLayoutPreset(panel, next.preset);

  const hadBrowser = !!panel.browser;
  if (hadBrowser && restartBrowser) {
    _disposeBrowserSurfacePanel(panel);
  }
  _refreshSurfacePanel(panel, true);
  if (hadBrowser && restartBrowser) {
    _startBrowserSurfacePanel(panel, 'panel-layout').catch(() => {});
  }
  if (persist && prevKey !== nextKey) {
    _persistSpatialPanels(panel.surface?.action || xrState.modeHubActiveAction);
    xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_layout_changed', {
      surface: panel.surface?.action || '',
      layout: _serializablePanelLayout(next),
    });
  }
  return true;
}

function _browserPanelEmbedUrl(surface, primaryAction = '') {
  try {
    return buildXrEmbedUrl(surface, {
      source: 'xr-browser-panel',
      sessionId: xrState.serverSessionId,
      primaryAction,
    });
  } catch {
    return surface?.embedUrl || '/ui/';
  }
}

async function _startBrowserSurfacePanel(panel, source = 'xr-panel') {
  if (!panel?.surface?.action) return false;
  if (_useDomOverlayPanels()) {
    // Experiment mode: skip the CDP screenshot stream entirely and let
    // the dom-overlay iframe carry the surface instead.
    _showWebEmbedForSurface(panel.surface, source, { primaryAction: panel.selectedAction || '' });
    return true;
  }
  if (panel.browser?.status === 'starting' || panel.browser?.status === 'active') return true;
  const url = _browserPanelEmbedUrl(panel.surface, panel.selectedAction || '');
  const layout = _normalizePanelLayout(panel.layout || {});
  panel.browser = {
    status: 'starting',
    id: '',
    url,
    frameUrl: '',
    revision: 0,
    image: null,
    imageUrl: '',
    loadingFrame: false,
    lastFrameAt: 0,
    streamWs: null,
    streamStatus: 'idle',
    streamClosed: false,
    streamReconnectTimer: 0,
    streamReconnects: 0,
    streamFrames: 0,
    streamLastFrameAt: 0,
    decodingFrame: false,
    pendingStreamFrame: null,
    contentRect: { x: 0, y: 0, w: panel.canvas?.width || layout.canvasW, h: panel.canvas?.height || layout.canvasH },
    error: '',
  };
  panel.lastKey = '';
  _refreshSurfacePanel(panel, true);
  xrState.modeHubStatus = `${panel.surface.label}: launching live page`;
  _refreshModeHub(true);
  try {
    const resp = await fetch('/api/xr/browser-panels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        width: layout.browserW || XR_BROWSER_PANEL_WIDTH,
        height: layout.browserH || XR_BROWSER_PANEL_HEIGHT,
        device_scale_factor: 1,
        format: 'jpeg',
        quality: 92,
      }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(extractErrorMessage(body, `HTTP ${resp.status}`));
    const browser = panel.browser;
    if (!browser) return false;
    browser.status = 'active';
    browser.id = body.panel?.id || '';
    browser.frameUrl = body.panel?.frame_url || '';
    browser.revision = Number(body.panel?.revision || 0);
    xrState.modeHubStatus = `${panel.surface.label}: live page active`;
    xrSession.recordEvent(xrState.serverSessionId, 'xr_browser_panel_started', {
      surface: panel.surface.action,
      source,
      url,
      panel_id: browser.id,
    });
    _refreshModeHub(true);
    _connectBrowserPanelStream(panel);
    _updateBrowserPanelFrame(panel, { force: true });
    return true;
  } catch (err) {
    if (panel.browser) {
      panel.browser.status = 'error';
      panel.browser.error = err?.message || String(err);
    }
    xrState.modeHubStatus = `${panel.surface.label}: live page unavailable`;
    xrSession.recordEvent(xrState.serverSessionId, 'xr_browser_panel_failed', {
      surface: panel.surface.action,
      url,
      message: err?.message || String(err),
    });
    _refreshSurfacePanel(panel, true);
    _refreshModeHub(true);
    return false;
  }
}

function _disposeBrowserSurfacePanel(panel) {
  const browser = panel?.browser;
  if (!browser) return;
  browser.streamClosed = true;
  if (browser.streamReconnectTimer) {
    clearTimeout(browser.streamReconnectTimer);
    browser.streamReconnectTimer = 0;
  }
  if (browser.streamWs) {
    try { browser.streamWs.close(); } catch {}
    browser.streamWs = null;
  }
  if (browser.imageUrl) {
    try { URL.revokeObjectURL(browser.imageUrl); } catch {}
  }
  if (browser.id) {
    fetch(`/api/xr/browser-panels/${encodeURIComponent(browser.id)}`, {
      method: 'DELETE',
    }).catch(() => {});
  }
  panel.browser = null;
  panel.lastKey = '';
}

function _browserPanelStreamUrl(panelId, ticket) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/api/xr/browser-panels/${encodeURIComponent(panelId)}/stream?ticket=${encodeURIComponent(ticket)}`;
}

function _queueBrowserPanelStreamFrame(panel, frame) {
  const browser = panel?.browser;
  if (!browser || browser.streamClosed) return;
  browser.pendingStreamFrame = frame;
  if (!browser.decodingFrame) _consumeBrowserPanelStreamFrame(panel);
}

function _consumeBrowserPanelStreamFrame(panel) {
  const browser = panel?.browser;
  if (!browser || browser.streamClosed || browser.decodingFrame) return;
  const frame = browser.pendingStreamFrame;
  if (!frame?.data) return;
  browser.pendingStreamFrame = null;
  browser.decodingFrame = true;
  const image = new Image();
  image.onload = () => {
    const active = panel.browser === browser && !browser.streamClosed;
    if (active) {
      if (browser.imageUrl) {
        try { URL.revokeObjectURL(browser.imageUrl); } catch {}
        browser.imageUrl = '';
      }
      browser.image = image;
      browser.revision = Number(frame.revision || browser.revision + 1);
      browser.streamFrames = Number(browser.streamFrames || 0) + 1;
      browser.streamLastFrameAt = performance.now();
      browser.lastFrameAt = browser.streamLastFrameAt;
      panel.lastKey = '';
      _refreshSurfacePanel(panel, true);
    }
    browser.decodingFrame = false;
    if (active && browser.pendingStreamFrame) _consumeBrowserPanelStreamFrame(panel);
  };
  image.onerror = () => {
    browser.error = 'stream frame decode failed';
    browser.decodingFrame = false;
    if (panel.browser === browser && browser.pendingStreamFrame) _consumeBrowserPanelStreamFrame(panel);
  };
  const mediaType = frame.media_type || frame.mediaType || 'image/jpeg';
  image.src = `data:${mediaType};base64,${frame.data}`;
}

function _scheduleBrowserPanelStreamReconnect(panel) {
  const browser = panel?.browser;
  if (!browser || browser.streamClosed || browser.status !== 'active') return;
  if (browser.streamReconnectTimer) return;
  const attempt = Number(browser.streamReconnects || 0) + 1;
  if (attempt > XR_BROWSER_PANEL_STREAM_MAX_RECONNECTS) {
    browser.streamStatus = 'fallback';
    _refreshSurfacePanel(panel, true);
    return;
  }
  browser.streamReconnects = attempt;
  const delay = XR_BROWSER_PANEL_STREAM_RECONNECT_MS * attempt;
  browser.streamReconnectTimer = setTimeout(() => {
    browser.streamReconnectTimer = 0;
    _connectBrowserPanelStream(panel);
  }, delay);
}

async function _connectBrowserPanelStream(panel) {
  const browser = panel?.browser;
  if (!browser?.id || browser.status !== 'active' || browser.streamClosed) return false;
  if (browser.streamStatus === 'connecting' || browser.streamStatus === 'open') return true;
  browser.streamStatus = 'connecting';
  browser.error = '';
  let ticket = '';
  try {
    ticket = await getWsTicket();
  } catch (err) {
    browser.streamStatus = 'fallback';
    browser.error = err?.message || 'stream auth failed';
    _scheduleBrowserPanelStreamReconnect(panel);
    return false;
  }
  if (panel.browser !== browser || browser.streamClosed) return false;
  let ws;
  try {
    ws = new WebSocket(_browserPanelStreamUrl(browser.id, ticket));
  } catch (err) {
    browser.streamStatus = 'fallback';
    browser.error = err?.message || 'stream unavailable';
    _scheduleBrowserPanelStreamReconnect(panel);
    return false;
  }
  browser.streamWs = ws;
  ws.onopen = () => {
    if (panel.browser !== browser) return;
    browser.streamStatus = 'open';
    browser.streamReconnects = 0;
    browser.error = '';
    _refreshSurfacePanel(panel, true);
  };
  ws.onmessage = (event) => {
    if (panel.browser !== browser || browser.streamClosed) return;
    let msg = null;
    try { msg = JSON.parse(event.data); } catch { return; }
    if (msg.type === 'frame') {
      browser.streamStatus = 'open';
      _queueBrowserPanelStreamFrame(panel, msg);
    } else if (msg.type === 'error') {
      browser.error = msg.message || 'stream error';
    }
  };
  ws.onerror = () => {
    if (panel.browser !== browser) return;
    browser.streamStatus = 'error';
  };
  ws.onclose = () => {
    if (panel.browser !== browser) return;
    if (browser.streamWs === ws) browser.streamWs = null;
    if (!browser.streamClosed && browser.status === 'active') {
      browser.streamStatus = 'closed';
      _scheduleBrowserPanelStreamReconnect(panel);
    }
  };
  return true;
}

async function _updateBrowserPanelFrame(panel, { force = false } = {}) {
  const browser = panel?.browser;
  if (!browser || browser.status !== 'active' || !browser.id || browser.loadingFrame) return false;
  if (!force && (browser.streamStatus === 'open' || browser.streamStatus === 'connecting')) return false;
  const now = performance.now();
  if (!force && now - browser.lastFrameAt < XR_BROWSER_PANEL_REFRESH_MS) return false;
  browser.loadingFrame = true;
  browser.lastFrameAt = now;
  try {
    const resp = await fetch(
      `/api/xr/browser-panels/${encodeURIComponent(browser.id)}/frame?rev=${Date.now()}`,
      { cache: 'no-store' },
    );
    if (!resp.ok) throw new Error(`frame HTTP ${resp.status}`);
    const rev = Number(resp.headers.get('X-Augmentum-XR-Panel-Revision') || 0);
    const blob = await resp.blob();
    const imageUrl = URL.createObjectURL(blob);
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = imageUrl;
    });
    if (browser.imageUrl) {
      try { URL.revokeObjectURL(browser.imageUrl); } catch {}
    }
    browser.imageUrl = imageUrl;
    browser.image = image;
    browser.revision = rev || browser.revision + 1;
    panel.lastKey = '';
    _refreshSurfacePanel(panel, true);
    return true;
  } catch (err) {
    browser.error = err?.message || String(err);
    panel.lastKey = '';
    _refreshSurfacePanel(panel, true);
    return false;
  } finally {
    browser.loadingFrame = false;
  }
}

async function _sendBrowserPanelInput(panel, x, y, type = 'click', extra = {}) {
  const browser = panel?.browser;
  if (!browser?.id || browser.status !== 'active') return false;
  if (type === 'refresh') {
    try {
      const resp = await fetch(`/api/xr/browser-panels/${encodeURIComponent(browser.id)}/input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'refresh', x: 0.5, y: 0.5, normalized: true }),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(extractErrorMessage(body, `input HTTP ${resp.status}`));
      browser.revision = Number(body.revision || browser.revision);
      if (browser.streamStatus !== 'open') await _updateBrowserPanelFrame(panel, { force: true });
      return true;
    } catch (err) {
      browser.error = err?.message || String(err);
      _refreshSurfacePanel(panel, true);
      return false;
    }
  }
  const rect = browser.contentRect || { x: 0, y: 0, w: panel.canvas.width, h: panel.canvas.height };
  const nx = (Number(x || 0) - rect.x) / Math.max(1, rect.w);
  const ny = (Number(y || 0) - rect.y) / Math.max(1, rect.h);
  if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return false;
  try {
    const resp = await fetch(`/api/xr/browser-panels/${encodeURIComponent(browser.id)}/input`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type,
        x: Math.max(0, Math.min(1, nx)),
        y: Math.max(0, Math.min(1, ny)),
        normalized: true,
        ...extra,
      }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(extractErrorMessage(body, `input HTTP ${resp.status}`));
    browser.revision = Number(body.revision || browser.revision);
    if (browser.streamStatus !== 'open') await _updateBrowserPanelFrame(panel, { force: true });
    return true;
  } catch (err) {
    browser.error = err?.message || String(err);
    _refreshSurfacePanel(panel, true);
    return false;
  }
}

function _drawPanelLayoutControls(ctx, panel, x, y, { compact = false } = {}) {
  if (!ctx || !panel) return [];
  const layout = _normalizePanelLayout(panel.layout || {});
  const defs = [
    { action: 'panel-layout-work', label: compact ? 'Work' : 'Work', preset: 'work' },
    { action: 'panel-layout-manga', label: compact ? 'Manga' : 'Manga', preset: 'manga' },
    { action: 'panel-layout-tv', label: compact ? 'TV' : 'TV', preset: 'tv' },
    { action: 'panel-layout-smaller', label: '-' },
    { action: 'panel-layout-larger', label: '+' },
    { action: 'panel-layout-curve', label: layout.curveM > 0.001 ? 'Flat' : 'Curve' },
  ];
  const buttonH = compact ? 34 : 40;
  const gap = compact ? 8 : 10;
  const buttons = [];
  let bx = x;
  for (const def of defs) {
    const bw = def.action === 'panel-layout-smaller' || def.action === 'panel-layout-larger'
      ? (compact ? 34 : 40)
      : def.action === 'panel-layout-manga'
      ? (compact ? 70 : 84)
      : (def.action === 'panel-layout-curve' ? (compact ? 66 : 78) : (compact ? 58 : 68));
    const selected = def.preset && layout.preset === def.preset;
    ctx.fillStyle = selected ? 'rgba(132, 172, 255, 0.34)' : 'rgba(255, 255, 255, 0.11)';
    _roundRect(ctx, bx, y, bw, buttonH, compact ? 10 : 12);
    ctx.fill();
    ctx.strokeStyle = selected ? 'rgba(220, 232, 255, 0.82)' : 'rgba(190, 215, 255, 0.40)';
    ctx.lineWidth = selected ? 3 : 2;
    ctx.stroke();
    ctx.fillStyle = '#f2f6ff';
    ctx.font = `bold ${compact ? 14 : 16}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.fillText(_fitCanvasText(ctx, def.label, bw - 12), bx + 8, y + (compact ? 9 : 11));
    buttons.push({
      kind: 'panel-layout',
      action: def.action,
      label: def.label,
      preset: def.preset || '',
      x: bx,
      y,
      w: bw,
      h: buttonH,
    });
    bx += bw + gap;
  }
  return buttons;
}

function _activatePanelLayoutControl(panel, button) {
  if (!panel || !button) return false;
  const current = _normalizePanelLayout(panel.layout || {});
  if (button.action === 'panel-layout-smaller' || button.action === 'panel-layout-larger') {
    const currentScale = Number(panel.group?.scale?.x || 1) || 1;
    const factor = button.action === 'panel-layout-smaller' ? 0.88 : 1.12;
    const scale = Math.max(XR_PANEL_MIN_SCALE, Math.min(XR_PANEL_MAX_SCALE, currentScale * factor));
    panel.group?.scale?.setScalar?.(scale);
    panel.lastKey = '';
    _refreshSurfacePanel(panel, true);
    _persistSpatialPanels(panel.surface?.action || xrState.modeHubActiveAction);
    xrState.modeHubStatus = `${panel.surface.label}: size ${scale.toFixed(2)}x`;
    _refreshModeHub(true);
    xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_resized', {
      surface: panel.surface?.action || '',
      scale: _roundPoseValue(scale),
      source: button.action,
    });
    return true;
  }
  if (button.action === 'panel-layout-curve') {
    const preset = _panelLayoutPreset(current.preset);
    const curveM = current.curveM > 0.001 ? 0 : Math.max(0.05, Number(preset.curveM || 0.055));
    _applyPanelLayout(panel, { ...current, curveM }, { persist: true, moveToPreset: false });
    xrState.modeHubStatus = `${panel.surface.label}: ${curveM > 0 ? 'curved' : 'flat'} panel`;
    _refreshModeHub(true);
    return true;
  }
  const preset = _panelLayoutPreset(button.preset || 'work');
  _applyPanelLayout(panel, preset, { persist: true, moveToPreset: true });
  xrState.modeHubStatus = `${panel.surface.label}: ${preset.label} layout`;
  _refreshModeHub(true);
  return true;
}

function _drawBrowserSurfacePanel(panel) {
  const browser = panel?.browser;
  const ctx = panel?.ctx;
  const canvas = panel?.canvas;
  if (!browser || !ctx || !canvas) return false;
  const w = canvas.width;
  const h = canvas.height;
  const { sx, sy, logicalW, logicalH } = _panelCanvasScale(panel);
  const toolbarH = 116;
  browser.contentRect = { x: 0, y: 0, w, h };
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#070b14';
  ctx.fillRect(0, 0, w, h);

  if (browser.image) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(browser.image, 0, 0, w, h);
  } else {
    ctx.fillStyle = 'rgba(210, 225, 255, 0.80)';
    ctx.font = 'bold 28px ui-monospace, Menlo, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const label = browser.status === 'error'
      ? `Live page failed: ${browser.error || 'unavailable'}`
      : 'Loading live page...';
    ctx.fillText(_fitCanvasText(ctx, label, w - 90), w / 2, h / 2);
  }

  ctx.save();
  ctx.scale(sx, sy);
  const gradient = ctx.createLinearGradient(0, 0, 0, toolbarH);
  gradient.addColorStop(0, 'rgba(7, 10, 18, 0.98)');
  gradient.addColorStop(1, 'rgba(12, 18, 32, 0.90)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, logicalW, toolbarH);
  ctx.strokeStyle = 'rgba(132, 255, 216, 0.50)';
  ctx.lineWidth = 3;
  ctx.strokeRect(1, 1, logicalW - 2, logicalH - 2);

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#f5fff9';
  ctx.font = 'bold 28px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(_fitCanvasText(ctx, `${panel.surface.label} Live`, Math.max(230, Math.min(360, logicalW - 540))), 24, 19);
  if (logicalW > 760) {
    ctx.fillStyle = 'rgba(220, 238, 255, 0.70)';
    ctx.font = '16px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(_fitCanvasText(ctx, browser.url || '', Math.max(120, logicalW - 570)), 382, 26);
  }

  const buttons = _drawPanelLayoutControls(ctx, panel, 24, 72, { compact: true });
  const defs = [
    { action: 'browser-scroll-up', label: 'Up', x: logicalW - 318 },
    { action: 'browser-scroll-down', label: 'Down', x: logicalW - 236 },
    { action: 'browser-refresh', label: 'Refresh', x: logicalW - 138 },
    { action: 'browser-summary', label: 'Cards', x: logicalW - 48 },
  ];
  for (const def of defs) {
    const bw = def.action === 'browser-summary' ? 38 : (def.action === 'browser-refresh' ? 82 : 66);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
    _roundRect(ctx, def.x, 14, bw, 44, 12);
    ctx.fill();
    ctx.strokeStyle = 'rgba(190, 215, 255, 0.46)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#f2f6ff';
    ctx.font = 'bold 15px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(_fitCanvasText(ctx, def.label, bw - 14), def.x + 8, 27);
    buttons.push({ kind: 'browser-control', action: def.action, label: def.label, x: def.x, y: 14, w: bw, h: 44 });
  }
  const moveButton = { kind: 'move', x: 0, y: 0, w: Math.max(220, Math.min(360, logicalW - 360)), h: 68 };
  ctx.restore();
  panel.buttons = _scaleCanvasButtons([...buttons, moveButton], sx, sy);
  panel.texture.needsUpdate = true;
  return true;
}

function _refreshSurfacePanel(panel, force = false) {
  if (!panel?.ctx || !panel?.texture || !panel?.canvas) return;
  if (panel.browser) {
    const key = [
      'browser',
      panel.surface.action,
      panel.browser.status,
      panel.browser.revision,
      panel.browser.error || '',
      _panelLayoutKey(panel),
      panel.dragging ? 'dragging' : '',
    ].join('|');
    if (!force && key === panel.lastKey) return;
    panel.lastKey = key;
    _drawBrowserSurfacePanel(panel);
    return;
  }
  const voiceLabel = _resolveVoiceLabel();
  const key = [
    panel.surface.action,
    panel.surface.label,
    panel.selectedAction,
    voiceLabel,
    _panelLayoutKey(panel),
    panel.dragging ? 'dragging' : '',
  ].join('|');
  const previousKey = panel.lastKey || '';
  const previousContentVersion = panel.lastContentVersion || 0;

  const { canvas, ctx, surface } = panel;
  const fallbackView = describeXrSurface(surface, {
    selectedAction: panel.selectedAction,
    voiceLabel,
  });
  const view = xrState.xrWorkspace?.describeSurface?.(surface.action, {
    selectedAction: panel.selectedAction,
    voiceLabel,
  }) || fallbackView;
  const liveContent = xrState.xrSurfaceData?.snapshot?.(surface.action) || null;
  const canvasW = canvas.width;
  const canvasH = canvas.height;
  const contentVersion = liveContent?.version || 0;
  if (!force && previousContentVersion === contentVersion && key === previousKey) return;
  panel.lastKey = key;
  panel.lastContentVersion = contentVersion;
  const { sx, sy, logicalW, logicalH } = _panelCanvasScale(panel);
  ctx.clearRect(0, 0, canvasW, canvasH);
  ctx.save();
  ctx.scale(sx, sy);
  const w = logicalW;
  const h = logicalH;

  ctx.fillStyle = 'rgba(8, 12, 22, 0.94)';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = panel.dragging ? 'rgba(92, 128, 255, 0.72)' : 'rgba(74, 104, 210, 0.58)';
  ctx.fillRect(0, 0, w, 86);
  ctx.strokeStyle = panel.dragging ? 'rgba(235, 242, 255, 0.88)' : 'rgba(135, 166, 255, 0.68)';
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, w - 4, h - 4);

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 40px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(view.title || surface.label, 38, 24);
  const moveButton = { kind: 'move', x: 0, y: 0, w: w - 104, h: 86 };
  ctx.fillStyle = 'rgba(230, 238, 255, 0.62)';
  ctx.font = '17px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('grab title bar to move', Math.max(280, Math.min(w - 360, ctx.measureText(view.title || surface.label).width + 70)), 34);
  ctx.fillStyle = 'rgba(230, 238, 255, 0.80)';
  ctx.font = '21px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(_formatPanelLabel(surface.panelKind || surface.placement || surface.action), 38, 96);
  const layoutButtons = _drawPanelLayoutControls(ctx, panel, Math.max(350, w - 372), 94, { compact: true });

  const closeButton = { kind: 'close', x: w - 82, y: 20, w: 48, h: 48 };
  ctx.fillStyle = 'rgba(255, 130, 130, 0.18)';
  _roundRect(ctx, closeButton.x, closeButton.y, closeButton.w, closeButton.h, 12);
  ctx.fill();
  ctx.fillStyle = '#ffe4e4';
  ctx.font = 'bold 30px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText('x', closeButton.x + 16, closeButton.y + 8);

  ctx.fillStyle = 'rgba(225, 232, 250, 0.90)';
  ctx.font = '24px ui-monospace, Menlo, Consolas, monospace';
  _wrapCanvasText(ctx, liveContent?.summary || view.summary, 38, 138, w - 76, 30, 2);

  const hasLiveContent = _drawSurfaceLiveContent(ctx, liveContent, 38, 212, w - 76, 176);
  if (!hasLiveContent) {
    const lineY = 210;
    ctx.fillStyle = 'rgba(190, 208, 244, 0.78)';
    ctx.font = '19px ui-monospace, Menlo, Consolas, monospace';
    const lines = Array.isArray(view.lines) ? view.lines.slice(0, 2) : [];
    lines.forEach((line, idx) => {
      _wrapCanvasText(ctx, `- ${line}`, 46, lineY + idx * 29, w - 92, 24, 1);
    });
  }

  const actionButtons = [];
  const actions = Array.isArray(view.actions) && view.actions.length
    ? view.actions.slice(0, 7)
    : ['focus'];
  let ax = 38;
  let ay = hasLiveContent ? 414 : 294;
  const liveLabel = 'Live Page';
  const liveW = 156;
  ctx.fillStyle = 'rgba(120, 255, 214, 0.18)';
  _roundRect(ctx, ax, ay, liveW, 50, 14);
  ctx.fill();
  ctx.strokeStyle = 'rgba(132, 255, 216, 0.62)';
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = '#e8fff8';
  ctx.font = 'bold 20px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(liveLabel, ax + 20, ay + 13);
  actionButtons.push({ kind: 'action', action: 'open-live-page', label: liveLabel, x: ax, y: ay, w: liveW, h: 50 });
  ax += liveW + 18;
  for (const action of actions) {
    const label = _formatPanelLabel(action);
    const bw = Math.min(238, Math.max(132, ctx.measureText(label).width + 42));
    if (ax + bw > w - 38) {
      ax = 38;
      ay += 70;
    }
    const selected = panel.selectedAction === action;
    ctx.fillStyle = selected ? 'rgba(120, 180, 255, 0.58)' : 'rgba(255, 255, 255, 0.12)';
    _roundRect(ctx, ax, ay, bw, 50, 14);
    ctx.fill();
    ctx.strokeStyle = selected ? 'rgba(230, 244, 255, 0.80)' : 'rgba(150, 170, 220, 0.42)';
    ctx.lineWidth = selected ? 3 : 2;
    ctx.stroke();
    ctx.fillStyle = '#f5f8ff';
    ctx.font = 'bold 20px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(label, ax + 20, ay + 13);
    actionButtons.push({ kind: 'action', action, label, x: ax, y: ay, w: bw, h: 50 });
    ax += bw + 18;
  }

  const context = Array.isArray(surface.contextSources) && surface.contextSources.length
    ? surface.contextSources.map(_formatPanelLabel).slice(0, 3).join('  /  ')
    : (Array.isArray(view.next) && view.next.length
      ? `Next: ${view.next.slice(0, 4).join('  /  ')}`
      : _formatPanelLabel(surface.hint || surface.action));
  ctx.fillStyle = 'rgba(176, 196, 235, 0.74)';
  ctx.font = '19px ui-monospace, Menlo, Consolas, monospace';
  _wrapCanvasText(ctx, context, 38, h - 88, w - 76, 25, 2);

  ctx.restore();
  panel.buttons = _scaleCanvasButtons([closeButton, ...layoutButtons, ...actionButtons, moveButton], sx, sy);
  panel.texture.needsUpdate = true;
}

function _createSurfacePanel(surface, pose) {
  const THREE = xrState.THREE;
  if (!THREE || !xrState.xrRig) return null;
  const layout = _normalizePanelLayout(pose?.layout || pose?.panelLayout || {});

  const canvas = document.createElement('canvas');
  canvas.width = layout.canvasW;
  canvas.height = layout.canvasH;
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const geometry = _createSurfacePanelGeometry(layout);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = `AugmentumXRSurfacePanel:${surface.action}`;
  mesh.renderOrder = 45;
  mesh.userData.xrSurfacePanel = surface.action;

  const group = new THREE.Group();
  group.name = `AugmentumXRSurface:${surface.action}`;
  group.add(mesh);
  _applySurfacePanelPose(group, pose || _panelPoseForSurface(surface));
  xrState.xrRig.add(group);

  const panel = {
    surface,
    group,
    mesh,
    canvas,
    ctx,
    texture,
    buttons: [],
    selectedAction: '',
    layout,
    dragging: false,
    lastKey: '',
    lastContentVersion: 0,
  };
  xrState.spatialPanels.set(surface.action, panel);
  _refreshSurfacePanel(panel, true);
  return panel;
}

function _disposeSurfacePanel(panel) {
  if (!panel) return;
  _disposeBrowserSurfacePanel(panel);
  panel.group?.parent?.remove(panel.group);
  panel.mesh?.geometry?.dispose();
  panel.mesh?.material?.map?.dispose();
  panel.mesh?.material?.dispose();
}

function _disposeSpatialPanels() {
  if (!xrState.spatialPanels) return;
  for (const panel of xrState.spatialPanels.values()) {
    _disposeSurfacePanel(panel);
  }
  xrState.spatialPanels.clear();
  xrState.panelDrag = null;
  xrState.hubDrag = null;
}

function _restoreSpatialPanels(panelState = {}) {
  if (!panelState || typeof panelState !== 'object') return;
  const surfaces = xrState.modeHubSurfaces || XR_HUB_SURFACES;
  for (const [action, pose] of Object.entries(panelState)) {
    if (pose?.open === false) continue;
    const surface = surfaces.find((s) => s.action === action || s.id === action);
    if (surface) _summonSurfacePanel(surface, pose, { persist: false });
  }
}

function _hitSurfacePanel(controller) {
  if (!_setRayFromController(controller)) return null;
  const panels = Array.from(xrState.spatialPanels?.values?.() || []);
  const meshes = panels.map((p) => p.mesh).filter(Boolean);
  if (!meshes.length) return null;
  const hits = xrState.raycaster.intersectObjects(meshes, false);
  if (!hits.length) return null;
  const hit = hits[0];
  const action = hit.object?.userData?.xrSurfacePanel;
  const panel = xrState.spatialPanels.get(action);
  if (!panel || !hit.uv) return null;
  const x = hit.uv.x * panel.canvas.width;
  const y = (1 - hit.uv.y) * panel.canvas.height;
  const button = panel.buttons.find((b) => (
    x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h
  ));
  return {
    panel,
    button,
    distance: hit.distance,
    point: hit.point,
    x,
    y,
    uv: hit.uv,
  };
}

function _hitModeHub(controller) {
  if (!controller || !xrState.modeHubPanel || !xrState.raycaster) return null;
  if (xrState.modeHubGroup?.visible === false) return null;
  if (!_setRayFromController(controller)) return null;
  const hits = xrState.raycaster.intersectObject(xrState.modeHubPanel, false);
  if (!hits.length) return null;
  const hit = hits[0];
  if (!hit.uv || !xrState.modeHubCanvas) return { button: null, distance: hit.distance };
  const x = hit.uv.x * xrState.modeHubCanvas.width;
  const y = (1 - hit.uv.y) * xrState.modeHubCanvas.height;
  return {
    button: _buttonAtCanvasPoint(xrState.modeHubButtons, x, y),
    distance: hit.distance,
  };
}

function _hitHandMenu(controller) {
  if (!controller || !xrState.handMenuPanel || xrState.handMenuGroup?.visible === false || !xrState.raycaster) return null;
  if (!_setRayFromController(controller)) return null;
  const hits = xrState.raycaster.intersectObject(xrState.handMenuPanel, false);
  if (!hits.length) return null;
  const hit = hits[0];
  if (!hit.uv || !xrState.handMenuCanvas) return { button: null, distance: hit.distance };
  const x = hit.uv.x * xrState.handMenuCanvas.width;
  const y = (1 - hit.uv.y) * xrState.handMenuCanvas.height;
  return {
    button: _buttonAtCanvasPoint(xrState.handMenuButtons, x, y),
    distance: hit.distance,
    point: hit.point,
  };
}

function _defaultHandMenuSurface() {
  const surfaces = xrState.modeHubSurfaces || XR_HUB_SURFACES;
  return surfaces.find((s) => s.action === xrState.modeHubActiveAction && s.action !== 'voice')
    || surfaces.find((s) => s.action === 'chat')
    || surfaces[0]
    || null;
}

function _activateHandMenuButton(button, source = 'hand-menu') {
  if (!button) return true;
  if (button.disabled) return true;
  if (button.action === 'hide-menu') {
    _hideHandMenu(source);
    return true;
  }
  if (button.action === 'place-avatar') {
    _enterMixedRealityAvatarPlacementMode(source, { hideAvatar: false });
    _refreshHandMenu(true);
    _hideHandMenu(`${source}:place-avatar`);
    return true;
  }
  if (button.action === 'toggle-hub') {
    if (xrState.modeHubGroup?.visible === false) {
      _summonHubNearView(source);
    } else {
      _setModeHubVisible(false, source);
    }
    _hideHandMenu(`${source}:toggle-hub`);
    return true;
  }
  if (button.action === 'close-panels') {
    _closeAllSurfacePanels(source);
    _hideHandMenu(`${source}:close-panels`);
    return true;
  }
  if (button.action === 'open-surface') {
    const surfaces = xrState.modeHubSurfaces || XR_HUB_SURFACES;
    const surface = surfaces.find((s) => s.action === button.surfaceAction || s.id === button.surfaceAction);
    if (surface) {
      _summonSurfacePanel(surface, null, { source });
    } else {
      _summonHubNearView(source);
    }
    _hideHandMenu(`${source}:open-surface`);
    return true;
  }
  if (button.action === 'summon-hub') {
    _summonHubNearView(source);
    _hideHandMenu(`${source}:summon-hub`);
    return true;
  }
  if (button.action === 'open-panel') {
    const surface = _defaultHandMenuSurface();
    if (surface) {
      _summonSurfacePanel(surface, null, { source });
    } else {
      _summonHubNearView(source);
    }
    _hideHandMenu(`${source}:open-panel`);
    return true;
  }
  if (button.action === 'toggle-probe') {
    _toggleHandDebugPanel(source);
    _hideHandMenu(`${source}:toggle-probe`);
    return true;
  }
  return false;
}

function _activateBrowserPanelControl(panel, button, hit = {}) {
  if (!panel?.browser) return false;
  const action = button?.action || '';
  if (action === 'browser-summary') {
    _disposeBrowserSurfacePanel(panel);
    xrState.modeHubStatus = `${panel.surface.label}: summary controls`;
    _refreshSurfacePanel(panel, true);
    _refreshModeHub(true);
    return true;
  }
  if (action === 'browser-refresh') {
    _sendBrowserPanelInput(panel, 0.5, 0.5, 'refresh', { normalized: true });
    xrState.modeHubStatus = `${panel.surface.label}: refreshing live page`;
    _refreshModeHub(true);
    return true;
  }
  if (action === 'browser-scroll-up' || action === 'browser-scroll-down') {
    const rect = panel.browser.contentRect || { x: 0, y: 0, w: panel.canvas.width, h: panel.canvas.height };
    _sendBrowserPanelInput(
      panel,
      rect.x + rect.w * 0.5,
      rect.y + rect.h * 0.5,
      'wheel',
      { deltaY: action === 'browser-scroll-up' ? -520 : 520 },
    );
    xrState.modeHubStatus = `${panel.surface.label}: ${action === 'browser-scroll-up' ? 'scrolling up' : 'scrolling down'}`;
    _refreshModeHub(true);
    return true;
  }
  _sendBrowserPanelInput(panel, hit.x, hit.y, 'click');
  return true;
}

function _activateInteractiveHit(hit, source = 'gaze') {
  if (!hit) return false;
  if (hit.type === 'hand-menu') {
    return _activateHandMenuButton(hit.button, source);
  }
  if (hit.type === 'hub') {
    if (!hit.button) return true;
    if (hit.button.kind === 'move-hub') {
      xrState.modeHubStatus = 'Use Quick Menu Hub to bring this closer, or pinch the title bar to move it.';
      _refreshModeHub(true);
      return true;
    }
    return _activateHubButton(hit.button);
  }
  if (hit.type === 'surface' && hit.panel) {
    if (hit.button?.kind === 'close') {
      _closeSurfacePanel(hit.panel.surface.action);
      return true;
    }
    if (hit.button?.kind === 'panel-layout') {
      return _activatePanelLayoutControl(hit.panel, hit.button);
    }
    if (hit.button?.kind === 'browser-control') {
      return _activateBrowserPanelControl(hit.panel, hit.button, hit);
    }
    if (hit.button?.kind === 'action') {
      _activateSurfacePanelAction(hit.panel, hit.button);
      return true;
    }
    if (hit.button?.kind === 'move') {
      xrState.modeHubStatus = `${hit.panel.surface.label}: pinch the title bar with a hand ray to move`;
      _refreshModeHub(true);
      return true;
    }
    if (hit.panel.browser?.status === 'active') {
      _sendBrowserPanelInput(hit.panel, hit.x, hit.y, 'click');
      xrState.modeHubStatus = `${hit.panel.surface.label}: page click sent`;
      _refreshModeHub(true);
      return true;
    }
    xrState.pendingHubConfirm = null;
    xrState.modeHubActiveAction = hit.panel.surface.action;
    xrState.modeHubStatus = `${hit.panel.surface.label}: aim at a button, then pinch`;
    _refreshModeHub(true);
    return true;
  }
  return false;
}

function _beginPanelDrag(controller, hit) {
  if (!controller || !hit?.panel) return false;
  hit.panel.dragging = true;
  _refreshSurfacePanel(hit.panel, true);
  xrState.panelDrag = {
    input: 'controller',
    controller,
    panel: hit.panel,
    distance: Math.max(0.55, Math.min(2.15, Number(hit.distance || 1.0))),
  };
  xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_grabbed', {
    surface: hit.panel.surface.action,
    handle: 'title-bar',
  });
  return true;
}

function _beginHubDrag(controller, hit) {
  if (!controller || !xrState.modeHubGroup || !hit) return false;
  xrState.hubDrag = {
    controller,
    distance: Math.max(0.68, Math.min(2.35, Number(hit.distance || 1.12))),
  };
  xrState.modeHubStatus = 'Hub grabbed. Move your hand, then release to place it.';
  xrState.modeHubLastKey = '';
  _refreshModeHub(true);
  xrSession.recordEvent(xrState.serverSessionId, 'hub_panel_grabbed', {
    handle: 'title-bar',
  });
  return true;
}

function _startPanelInteraction(controller, inputSource = null) {
  const menuHit = _hitHandMenu(controller);
  if (menuHit) return true;
  const hit = _hitSurfacePanel(controller);
  if (hit) {
    if (hit.button?.kind === 'move') return _beginPanelDrag(controller, hit);
    return true;
  }
  const hubHit = _hitModeHub(controller);
  if (hubHit?.button?.kind === 'move-hub') return _beginHubDrag(controller, hubHit);
  if (hubHit) return true;
  return false;
}

function _selectXrTarget(controller, inputSource = null) {
  const menuHit = _hitHandMenu(controller);
  if (menuHit) {
    return _activateHandMenuButton(menuHit.button, _isHandInputSource(inputSource) ? 'hand-ray' : 'controller-ray');
  }
  const hit = _hitSurfacePanel(controller);
  if (hit) {
    if (hit.button?.kind === 'close') {
      _closeSurfacePanel(hit.panel.surface.action);
      return true;
    }
    if (hit.button?.kind === 'panel-layout') {
      return _activatePanelLayoutControl(hit.panel, hit.button);
    }
    if (hit.button?.kind === 'browser-control') {
      return _activateBrowserPanelControl(hit.panel, hit.button, hit);
    }
    if (hit.button?.kind === 'action') {
      _activateSurfacePanelAction(hit.panel, hit.button);
      return true;
    }
    if (hit.button?.kind === 'move') {
      xrState.modeHubStatus = `${hit.panel.surface.label}: hold the title bar to move`;
      _refreshModeHub(true);
      return true;
    }
    if (hit.panel.browser?.status === 'active') {
      _sendBrowserPanelInput(hit.panel, hit.x, hit.y, 'click');
      xrState.modeHubStatus = `${hit.panel.surface.label}: page click sent`;
      _refreshModeHub(true);
      return true;
    }
    xrState.pendingHubConfirm = null;
    xrState.modeHubActiveAction = hit.panel.surface.action;
    xrState.modeHubStatus = _isHandInputSource(inputSource)
      ? `${hit.panel.surface.label}: aim at a button, then pinch`
      : `${hit.panel.surface.label}: use the title bar to move`;
    _refreshModeHub(true);
    return true;
  }
  if (_selectHubSurface(controller)) return true;
  if (xrState.xrMode === XR_MODE_MR) {
    return _placeMixedRealityAvatarFromController(controller, 'empty-space-select');
  }
  return false;
}

function _endPanelInteraction(controller) {
  const drag = xrState.panelDrag;
  if (!drag) return false;
  if (drag.input === 'hand') {
    if (drag.hand !== controller) return false;
  } else if (drag.controller !== controller) {
    return false;
  }
  drag.panel.dragging = false;
  _refreshSurfacePanel(drag.panel, true);
  xrState.panelDrag = null;
  _persistSpatialPanels(drag.panel.surface.action);
  xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_placed', {
    surface: drag.panel.surface.action,
    pose: _surfacePanelPose(drag.panel),
  });
  return true;
}

function _endHubInteraction(controller) {
  const drag = xrState.hubDrag;
  if (!drag || drag.controller !== controller) return false;
  xrState.hubDrag = null;
  xrState.hubUserPlaced = true;
  xrState.modeHubStatus = 'Hub placed. Select a surface or grab the title bar again.';
  xrState.modeHubLastKey = '';
  _refreshModeHub(true);
  _persistSpatialPanels(xrState.modeHubActiveAction);
  xrSession.recordEvent(xrState.serverSessionId, 'hub_panel_placed', {
    pose: xrState.modeHubGroup ? _surfacePanelPose(xrState.modeHubGroup) : null,
  });
  return true;
}

function _updatePanelDrag() {
  const drag = xrState.panelDrag;
  if (!drag?.panel?.group) return;
  const point = xrState.panelDragWorldPoint;
  if (!point || !xrState.xrRig) return;
  if (!drag.controller || !_setRayFromController(drag.controller)) return;
  point.copy(xrState.rayDirection).multiplyScalar(drag.distance).add(xrState.rayOrigin);
  xrState.xrRig.worldToLocal(point);
  point.y = Math.max(0.72, Math.min(1.72, point.y));
  drag.panel.group.position.copy(point);
  if (xrState.camera && xrState.panelCameraWorld) {
    xrState.camera.getWorldPosition(xrState.panelCameraWorld);
    drag.panel.group.lookAt(xrState.panelCameraWorld);
  }
}

function _updateHubDrag() {
  const drag = xrState.hubDrag;
  if (!drag || !xrState.modeHubGroup) return;
  const point = xrState.panelDragWorldPoint;
  if (!point || !xrState.xrRig) return;
  if (!drag.controller || !_setRayFromController(drag.controller)) return;
  point.copy(xrState.rayDirection).multiplyScalar(drag.distance).add(xrState.rayOrigin);
  xrState.xrRig.worldToLocal(point);
  point.y = Math.max(0.82, Math.min(1.62, point.y));
  xrState.modeHubGroup.position.copy(point);
  if (xrState.camera && xrState.panelCameraWorld) {
    xrState.camera.getWorldPosition(xrState.panelCameraWorld);
    xrState.modeHubGroup.lookAt(xrState.panelCameraWorld);
  }
}

function _summonSurfacePanel(surface, pose = null, { persist = true, source = 'xr-panel' } = {}) {
  if (!surface?.action) return null;
  let panel = xrState.spatialPanels.get(surface.action);
  if (!panel) {
    panel = _createSurfacePanel(surface, pose || _panelPoseForSurface(surface));
  } else if (pose) {
    _applySurfacePanelPose(panel.group, pose);
    if (pose.layout || pose.panelLayout) {
      _applyPanelLayout(panel, pose.layout || pose.panelLayout, { restartBrowser: false });
    }
  }
  if (!panel) return null;
  panel.group.visible = true;
  xrState.modeHubActiveAction = surface.action;
  xrState.modeHubStatus = `${surface.label}: panel summoned`;
  xrState.xrWorkspace?.openSurface?.(surface.action, {
    source: persist ? source : 'restore',
    label: surface.label,
  }, { emit: persist });
  _refreshSurfacePanel(panel, true);
  _refreshModeHub(true);
  _refreshHandMenu(true);
  if (persist) {
    xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_summoned', {
      surface: surface.action,
      label: surface.label,
    });
    _persistSpatialPanels(surface.action);
    if (xrState.renderer?.xr?.isPresenting) {
      _startBrowserSurfacePanel(panel, source);
    }
  }
  return panel;
}

function _closeSurfacePanel(action) {
  const panel = xrState.spatialPanels.get(action);
  if (!panel) return;
  hideXrWebEmbed(action);
  _disposeSurfacePanel(panel);
  xrState.spatialPanels.delete(action);
  if (xrState.modeHubActiveAction === action) {
    const next = xrState.spatialPanels.values().next().value;
    xrState.modeHubActiveAction = next?.surface?.action || 'voice';
  }
  xrState.modeHubStatus = panel.surface.label + ' panel closed';
  xrState.xrWorkspace?.closeSurface?.(action, {
    source: 'xr-panel',
    label: panel.surface.label,
  });
  _refreshModeHub(true);
  _refreshHandMenu(true);
  _persistSpatialPanels(xrState.modeHubActiveAction);
  xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_closed', {
    surface: action,
  });
}

function _closeAllSurfacePanels(source = 'quick-menu') {
  const actions = Array.from(xrState.spatialPanels?.keys?.() || []);
  if (!actions.length) {
    xrState.modeHubStatus = 'No panels are open.';
    _refreshModeHub(true);
    _refreshHandMenu(true);
    return false;
  }
  for (const action of actions) {
    _closeSurfacePanel(action);
  }
  xrState.modeHubActiveAction = 'voice';
  xrState.modeHubStatus = `${actions.length} panel${actions.length === 1 ? '' : 's'} closed.`;
  _refreshModeHub(true);
  _refreshHandMenu(true);
  _persistSpatialPanels('voice');
  xrSession.recordEvent(xrState.serverSessionId, 'surface_panels_closed_all', {
    count: actions.length,
    source,
  });
  return true;
}

function _setModeHubVisible(visible, source = 'quick-menu') {
  if (!xrState.modeHubGroup) return false;
  xrState.modeHubGroup.visible = !!visible;
  if (!visible) xrState.hubDrag = null;
  xrState.modeHubLastKey = '';
  _refreshModeHub(true);
  _refreshHandMenu(true);
  _persistSpatialPanels(xrState.modeHubActiveAction);
  xrSession.recordEvent(xrState.serverSessionId, 'hub_visibility_changed', {
    visible: !!visible,
    source,
  });
  return true;
}

function _showWebEmbedForSurface(surface, source, extra = {}) {
  if (!surface?.action) return '';
  try {
    const presenting = !!xrState.renderer?.xr?.isPresenting;
    if (presenting && !_useDomOverlayPanels()) {
      const marker = `xr-panel:${surface.action}`;
      xrSession.recordEvent(xrState.serverSessionId, 'xr_native_surface_opened', {
        surface: surface.action,
        source,
        marker,
        primary_action: extra.primaryAction || '',
      });
      return marker;
    }
    const href = showXrWebEmbed(surface, {
      source,
      sessionId: xrState.serverSessionId,
      ...extra,
      immersiveOverlay: presenting,
    });
    const embedState = getXrWebEmbedState();
    xrSession.recordEvent(xrState.serverSessionId, 'web_embed_opened', {
      surface: surface.action,
      source,
      embed_url: href,
      primary_action: extra.primaryAction || '',
      immersive_suppressed: !!embedState.immersiveSuppressed,
    });
    return href;
  } catch (err) {
    console.warn('[avatar-xr] web embed open failed', err);
    xrSession.recordEvent(xrState.serverSessionId, 'web_embed_failed', {
      surface: surface.action,
      source,
      message: err?.message || String(err),
    });
    return '';
  }
}

function _surfaceEmbedStatus(label) {
  if (xrState.renderer?.xr?.isPresenting) {
    return `${label}: stereo panel active`;
  }
  const embedState = getXrWebEmbedState();
  if (embedState.immersiveSuppressed) {
    return `${label}: stereo panel active`;
  }
  return `${label}: web embed open`;
}

function _activateSurfacePanelAction(panel, button) {
  xrState.pendingHubConfirm = null;
  panel.selectedAction = button.action;
  xrState.modeHubActiveAction = panel.surface.action;
  xrState.modeHubStatus = `${panel.surface.label}: ${button.label}`;
  if (button.action === 'open-live-page') {
    xrState.xrWorkspace?.invokeSurfaceAction?.(panel.surface.action, button.action, {
      source: 'xr-panel',
      label: panel.surface.label,
      primaryActionLabel: button.label,
    });
    _startBrowserSurfacePanel(panel, 'panel-live-button');
    _persistSpatialPanels(panel.surface.action, button.action);
    xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_live_page_selected', {
      surface: panel.surface.action,
    });
    return;
  }
  xrState.xrWorkspace?.invokeSurfaceAction?.(panel.surface.action, button.action, {
    source: 'xr-panel',
    label: panel.surface.label,
    primaryActionLabel: button.label,
  });
  const embedHref = _showWebEmbedForSurface(panel.surface, 'xr-panel', {
    primaryAction: button.action,
  });
  if (embedHref) {
    xrState.modeHubStatus = _surfaceEmbedStatus(panel.surface.label);
  }
  _refreshSurfacePanel(panel, true);
  _refreshModeHub(true);
  _persistSpatialPanels(panel.surface.action, button.action);
  xrSession.recordEvent(xrState.serverSessionId, 'surface_panel_action_selected', {
    surface: panel.surface.action,
    action: button.action,
  });
  try {
    const detail = {
      source: 'xr-panel',
      action: panel.surface.action,
      label: panel.surface.label,
      primaryAction: button.action,
      primaryActionLabel: button.label,
      sessionId: xrState.serverSessionId,
      immersive: !!xrState.renderer?.xr?.isPresenting,
      presentation: xrState.renderer?.xr?.isPresenting ? 'immersive' : 'browser',
    };
    window.dispatchEvent(new CustomEvent('augmentum:xr-open-surface', {
      detail,
    }));
    window.dispatchEvent(new CustomEvent('augmentum:xr-panel-action', {
      detail,
    }));
  } catch (err) {
    console.warn('[avatar-xr] panel action dispatch failed', err);
  }
}

function _roomStateFromPanels(activeSurface = xrState.modeHubActiveAction, selectedPanelAction = '') {
  const surfacePanels = {};
  const openSurfaces = ['voice'];
  for (const [action, panel] of xrState.spatialPanels.entries()) {
    surfacePanels[action] = {
      ..._surfacePanelPose(panel),
      selectedAction: panel.selectedAction || '',
    };
    openSurfaces.push(action);
  }
  return {
    activeSurface: activeSurface || 'voice',
    selectedPanelAction: selectedPanelAction || '',
    openSurfaces: Array.from(new Set(openSurfaces)),
    surfacePanels,
    hub: {
      enabled: xrState.modeHubGroup?.visible !== false,
      selectedSurface: activeSurface || 'voice',
      selectedAt: new Date().toISOString(),
      layout: 'seat-relative-dock',
      pose: xrState.modeHubGroup ? _surfacePanelPose(xrState.modeHubGroup) : undefined,
    },
  };
}

function _persistSpatialPanels(activeSurface = xrState.modeHubActiveAction, selectedPanelAction = '') {
  xrSession.patchSession(xrState.serverSessionId, {
    room_state: _roomStateFromPanels(activeSurface, selectedPanelAction),
  });
}

function _selectHubSurface(controller) {
  const hit = _hitModeHub(controller);
  if (!hit) return false;
  const button = hit.button;
  if (!button) return true;
  if (button.kind === 'move-hub') {
    xrState.modeHubStatus = 'Hold the title bar, move your hand, then release.';
    _refreshModeHub(true);
    return true;
  }
  _activateHubButton(button);
  return true;
}

function _openHubSurface(surface) {
  xrState.modeHubActiveAction = surface.action;
  xrState.modeHubStatus = `${surface.label}: panel summoned`;
  const panel = _summonSurfacePanel(surface, null, { source: 'xr-hub' });
  if (!xrState.renderer?.xr?.isPresenting && panel) {
    const embedHref = _showWebEmbedForSurface(surface, 'xr-hub');
    if (embedHref) {
      xrState.modeHubStatus = _surfaceEmbedStatus(surface.label);
    }
  }
  _refreshModeHub(true);
  xrSession.recordEvent(xrState.serverSessionId, 'hub_surface_selected', {
    surface: surface.action,
    label: surface.label,
  });

  try {
    window.dispatchEvent(new CustomEvent('augmentum:xr-open-surface', {
      detail: {
        source: 'xr-hub',
        action: surface.action,
        label: surface.label,
        sessionId: xrState.serverSessionId,
        immersive: !!xrState.renderer?.xr?.isPresenting,
        presentation: xrState.renderer?.xr?.isPresenting ? 'immersive' : 'browser',
      },
    }));
  } catch (err) {
    console.warn('[avatar-xr] surface dispatch failed', err);
  }
}

// Reload saved seat coordinates and reapply them to the active rig.
// Cheap form of recentering — assumes the user is physically still on
// their real-world couch; resets virtual placement to the saved good.
async function _recenterRig() {
  if (!xrState.xrRig) return;
  if (xrState.xrMode === XR_MODE_MR) {
    xrState.xrRig.position.set(0, 0, 0);
    xrState.xrRig.rotation.y = 0;
    _restartMixedRealityPlacement('recenter');
    return;
  }
  const seat = await loadSavedSeat();
  if (!xrState.xrRig) return;   // session may have ended during await
  xrState.xrRig.position.set(seat.x, seat.y, seat.z);
  xrState.xrRig.rotation.y = seat.rotY;
}

// ── Spatial audio (TTS from avatar's head) ───────────────────────────
// Re-routes voice.js's TTS analyserNode through a THREE.PositionalAudio
// panner attached to the avatar's head bone. Lipsync continues to work
// because lipsync taps the analyser's INPUT side (ttsGainNode upstream),
// while the panner sits on the OUTPUT side. Pause/duck infrastructure is
// also untouched — it's all upstream of the analyser.
//
// Defensive: every audio operation is guarded. If the setup fails for
// any reason (no analyser, closed context, version mismatch), we leave
// voice.js's original graph intact and the user gets flat audio in VR
// rather than no audio at all.
function _setupSpatialAudio(THREE, camera, vrm) {
  try {
    const analyserNode = voiceAudio.getTtsAnalyser();
    if (!analyserNode) return;
    if (!vrm?.humanoid) return;

    const audioContext = analyserNode.context;
    if (!audioContext || audioContext.state === 'closed') return;

    // Share three.js's audio context with voice.js's so both reference
    // the same AudioContext. Without this, three.js would lazily create
    // a SECOND context and the worlds wouldn't share Web Audio nodes.
    if (THREE.AudioContext?.setContext) {
      THREE.AudioContext.setContext(audioContext);
    }

    const listener = new THREE.AudioListener();
    camera.add(listener);

    const positional = new THREE.PositionalAudio(listener);
    positional.setRefDistance(0.6);
    positional.setRolloffFactor(1.2);
    positional.setDistanceModel('inverse');

    // Attach to head bone if available, else avatar root.
    const headBone = vrm.humanoid.getNormalizedBoneNode('head');
    (headBone || vrm.scene).add(positional);

    // Ask voice.js to route TTS (downstream of the lip-sync analyser) into
    // the panner's gain — voice.js owns the node graph, so we don't rewire
    // it ourselves. If the reroute can't happen we leave voice.js's flat
    // path intact (flat audio beats silence). The panner chain
    // (positional.gain -> positional.panner -> destination) is wired by the
    // PositionalAudio constructor, so analyser -> positional.gain yields
    // HRTF-spatialized output that tracks the headset via the listener pose.
    // Pause/duck infrastructure in voice.js is untouched — it's upstream of
    // the analyser.
    if (!voiceAudio.routeTtsToNode(positional.gain)) {
      if (positional.parent) positional.parent.remove(positional);
      if (listener.parent) listener.parent.remove(listener);
      return;
    }

    xrState.audioListener = listener;
    xrState.positionalAudio = positional;
    xrState.audioContext = audioContext;
    xrState.audioRerouted = true;
  } catch (err) {
    console.warn('[avatar-xr] spatial audio setup failed — falling back to flat audio', err);
    // voice.js owns the graph — ask it to restore the direct-to-speakers
    // route in case we mutated it partway through.
    try { voiceAudio.restoreTtsRoute(); } catch { /* swallow */ }
  }
}

function _teardownSpatialAudio() {
  if (!xrState.audioRerouted) return;
  try {
    voiceAudio.restoreTtsRoute();
    if (xrState.positionalAudio?.parent) {
      xrState.positionalAudio.parent.remove(xrState.positionalAudio);
    }
    if (xrState.audioListener?.parent) {
      xrState.audioListener.parent.remove(xrState.audioListener);
    }
  } catch (err) {
    console.warn('[avatar-xr] spatial audio teardown failed', err);
  }
  xrState.audioRerouted = false;
  xrState.positionalAudio = null;
  xrState.audioListener = null;
  xrState.audioContext = null;
}

// ── Status HUD ────────────────────────────────────────────────────────
// A 0.5m × 0.125m plane anchored to the camera at (0, -0.35, -1.0) —
// below eye-line so it doesn't fight the avatar, close enough to read.
// Updates only on state transitions, not every frame.
function _setupHUD(THREE, camera) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');

  const tex = new THREE.CanvasTexture(canvas);
  _configurePanelTexture(tex);

  const geom = new THREE.PlaneGeometry(0.5, 0.125);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthTest: false,   // always visible, even if avatar is in front
    depthWrite: false,
  });
  const plane = new THREE.Mesh(geom, mat);
  plane.position.set(0, -0.35, -1.0);
  plane.renderOrder = 999;

  camera.add(plane);

  xrState.hudCanvas = canvas;
  xrState.hudCtx = ctx;
  xrState.hudTexture = tex;
  xrState.hudPlane = plane;
  xrState.hudLastLabel = '';

  // Initial paint so the user sees something the first frame.
  _refreshHUD(ctx, tex);
}

function _resolveVoiceLabel() {
  if (!appAvatarState) return 'Ready';
  if (bus.state.is_speaking) return 'Speaking';
  const v = bus.state.voice_state;
  if (v === 'processing') return 'Thinking...';
  if (v === 'listening' || v === 'recording') return 'Listening';
  if (v === 'composing') return 'Composing...';
  return 'Ready';
}

// Bottom-line label — shows the active app mode + character so VR feels
// connected to the rest of Augmentum. Mode lives in avatarState.callMode
// (set on activation), character name in avatarState.avatarName (stashed
// from /api/avatar/for-session). Returns empty string when neither is
// available (e.g., voice call hasn't activated yet).
function _resolveContextLabel() {
  const s = appAvatarState;
  if (!s) return '';
  const mode = s.callMode || '';
  const name = s.avatarName || '';
  const modeLabel = mode ? mode.charAt(0).toUpperCase() + mode.slice(1) : '';
  if (modeLabel && name) return `${modeLabel} · ${name}`;
  return modeLabel || name;
}

function _refreshHUD(ctx, tex) {
  const voiceLabel = _resolveVoiceLabel();
  const ctxLabel = _resolveContextLabel();
  // Combined cache key — invalidates on either change so mode swaps
  // mid-session repaint the bottom line.
  const combo = voiceLabel + '|' + ctxLabel;
  if (combo === xrState.hudLastLabel) return;
  xrState.hudLastLabel = combo;

  ctx.clearRect(0, 0, 512, 128);
  ctx.fillStyle = 'rgba(10, 12, 22, 0.78)';
  ctx.fillRect(0, 0, 512, 128);
  ctx.strokeStyle = 'rgba(96, 128, 255, 0.55)';
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, 510, 126);

  // Top line: voice state — dominant, larger.
  ctx.fillStyle = '#dde';
  ctx.font = 'bold 44px ui-monospace, Menlo, Consolas, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(voiceLabel, 256, ctxLabel ? 44 : 64);

  // Bottom line: mode + character — secondary, dimmer. Only drawn when
  // we have something to say (skips empty space pre-activation).
  if (ctxLabel) {
    ctx.fillStyle = 'rgba(160, 180, 220, 0.75)';
    ctx.font = '24px ui-monospace, Menlo, Consolas, monospace';
    ctx.fillText(ctxLabel, 256, 92);
  }

  tex.needsUpdate = true;
  _refreshModeHub();
}

function _resolveXrAvatarPose(seat = {}, roomManifest = null) {
  const envId = seat?.envId || roomManifest?.id || DEFAULT_SEAT.envId;
  const seatAnchor = seat?.avatar && typeof seat.avatar === 'object' ? seat.avatar : {};
  const manifestAnchor = roomManifest?.anchors?.avatarSeat || {};
  const fallback = DEFAULT_XR_AVATAR_ANCHOR;
  const pose = {
    x: _finiteNumber(seatAnchor.x, _finiteNumber(manifestAnchor.x, fallback.x)),
    y: _finiteNumber(seatAnchor.y, _finiteNumber(manifestAnchor.y, fallback.y)),
    z: _finiteNumber(seatAnchor.z, _finiteNumber(manifestAnchor.z, fallback.z)),
    rotY: _finiteNumber(
      seatAnchor.rotY ?? seatAnchor.rot_y,
      _finiteNumber(manifestAnchor.rotY ?? manifestAnchor.rot_y, fallback.rotY),
    ),
  };
  const unclamped = { ...pose };
  if (envId === 'modern-room') {
    pose.x = _clamp(pose.x, MODERN_ROOM_AVATAR_BOUNDS.minX, MODERN_ROOM_AVATAR_BOUNDS.maxX);
    pose.z = _clamp(pose.z, MODERN_ROOM_AVATAR_BOUNDS.minZ, MODERN_ROOM_AVATAR_BOUNDS.maxZ);
  }
  if (!_nearNumber(pose.x, unclamped.x, 0.001) || !_nearNumber(pose.z, unclamped.z, 0.001)) {
    xrSession.recordEvent(xrState.serverSessionId, 'avatar_pose_clamped', {
      envId,
      requested: unclamped,
      applied: pose,
    });
  }
  return pose;
}

function _applyXrAvatarRootPose(vrm, pose = null) {
  if (!vrm?.scene) return;
  const presetFallback = POSE_PRESETS.sittingEdge?._avatarPosition;
  const root = pose || (presetFallback ? {
    x: presetFallback[0],
    y: presetFallback[1],
    z: presetFallback[2],
    rotY: null,
  } : null);
  if (!root) return;
  vrm.scene.position.set(
    _finiteNumber(root.x, 0),
    _finiteNumber(root.y, 0),
    _finiteNumber(root.z, 0),
  );
  const rotY = _finiteNumber(root.rotY ?? root.rot_y, NaN);
  if (Number.isFinite(rotY)) {
    vrm.scene.rotation.y = rotY;
  }
  vrm.scene.updateMatrixWorld?.(true);
}

function _applySittingPoseOnce(vrm, avatarPose = null) {
  const preset = POSE_PRESETS.sittingEdge;
  if (!vrm.humanoid || !preset) return;
  // applyPosePreset handles the full sequence (resetNormalizedPose,
  // per-bone Euler order, per-bone axis-sign correction, hips
  // translation) so externally imported VRMs in either arm/finger
  // convention sit correctly. Pre-fix this used a raw rotation.set loop
  // with no profile awareness — fingers hyperextended and arms drifted
  // on mirrored-axis exports.
  const THREE = xrState.THREE;
  const profile = vrm.__augmentumCompatibilityProfile;
  applyPosePreset(THREE, vrm, preset, {
    armAxisSign: armAxisSignFromProfile(profile?.armAxisProfile, 'mirrored'),
    fingerAxisSign: fingerAxisSignFromProfile(profile?.fingerAxisProfile),
    reset: true,
  });
  _applyXrAvatarRootPose(vrm, avatarPose);
}

// Per-frame pose lock. Hooked from vrm.scene.onBeforeRender so it fires
// AFTER AvatarAnimator's procedural writes and BEFORE the VRM renders —
// guaranteeing the lower body lands on the couch each frame regardless
// of the animator's sway/breathing. Upper body stays unlocked so chest
// breathing, head gaze, and arm gestures continue to play normally.
function _stampLockedSitPose(vrm, fsmState = null) {
  if (!vrm?.humanoid) return;
  // When voice_xr_proxemics_enabled and the FSM has transitioned out of a
  // seated state, release the per-frame lock so locomotion takes effect.
  // With the flag off, fsmState is null and the lock applies unconditionally.
  if (fsmState && !fsmState.isSeated()) return;
  const preset = POSE_PRESETS.sittingEdge;
  if (!preset) return;
  // Lower-body bone rotations
  for (const bone of SIT_LOCKED_BONE_ROTATIONS) {
    const rot = preset.bones?.[bone];
    if (!rot) continue;
    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (!node) continue;
    const order = BONE_ROTATION_ORDERS[bone] || 'XYZ';
    node.rotation.set(rot[0], rot[1], rot[2], order);
  }
  // Hips position — animator may have perturbed Y via simplex sway.
  const hips = vrm.humanoid.getNormalizedBoneNode('hips');
  if (hips) {
    if (!hips._origPos) hips._origPos = hips.position.clone();
    hips.position.copy(hips._origPos);
    if (preset._hipsTranslation) {
      hips.position.x += preset._hipsTranslation[0];
      hips.position.y += preset._hipsTranslation[1];
      hips.position.z += preset._hipsTranslation[2];
    }
  }
  // VRM root position — animator may translate the whole avatar.
  _applyXrAvatarRootPose(vrm, xrState.avatarXrPose);
}
