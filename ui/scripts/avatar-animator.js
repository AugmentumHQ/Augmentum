/**
 * avatar-animator.js — PD spring-damper procedural animation for VRM avatars
 *
 * Rewritten with Luna reference project's tuned constants.
 * Each bone has a spring-damper (PD) controller — layers set TARGET rotations
 * and the physics interpolate smoothly.
 *
 * Layers (additive targets per bone):
 *   1. Breathing     — chest/upperChest/shoulder sine wave, emotion-modulated
 *   2. Sway          — simplex noise on hips/spine/head with counter-sway
 *   3. Emotion       — posture targets + VRM expression blend weights
 *   4. Gaze          — saccades, blinks, idle look-around, head follow
 *   5. Awareness     — reactive impulses (silence, typing, tool call)
 *
 * THREE is injected via constructor — do not import it here.
 */

import {
  CALL_ACTIONS,
  DEFAULT_CALL_MOTION_TEXTURE,
  HAND_SHAPES,
  CallActionScheduler,
  assessCallActionCompatibility,
  getCallActionContract,
} from './avatar-call-actions.js';
import {
  HAND_POSES,
  FINGER_NAMES,
  JOINT_NAMES,
  HAND_POSE_NAMES,
  fingerBoneName,
} from './avatar-hand-poses.js';
import {
  fingerAxisSignFromProfile,
  armAxisSignFromProfile,
} from './avatar-vrm-profile.js';
import { PoseOrchestrator } from './avatar-pose-orchestrator.js';
import { POSE_PRESETS } from './avatar-pose-presets.js';

// ---------------------------------------------------------------------------
// Lightweight inline simplex noise (2D)
// ---------------------------------------------------------------------------
const _GRAD3 = [
  [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
  [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
  [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
];
const _PERM = new Uint8Array(512);
const _PERM12 = new Uint8Array(512);
(function _seedPerm() {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) {
    _PERM[i] = p[i & 255];
    _PERM12[i] = _PERM[i] % 12;
  }
})();

function _dot2(g, x, y) { return g[0] * x + g[1] * y; }

function simplex2D(xin, yin) {
  const F2 = 0.5 * (Math.sqrt(3) - 1);
  const G2 = (3 - Math.sqrt(3)) / 6;
  const s = (xin + yin) * F2;
  const i = Math.floor(xin + s);
  const j = Math.floor(yin + s);
  const t = (i + j) * G2;
  const X0 = i - t, Y0 = j - t;
  const x0 = xin - X0, y0 = yin - Y0;
  const i1 = x0 > y0 ? 1 : 0, j1 = x0 > y0 ? 0 : 1;
  const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
  const x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
  const ii = i & 255, jj = j & 255;
  const gi0 = _PERM12[ii + _PERM[jj]];
  const gi1 = _PERM12[ii + i1 + _PERM[jj + j1]];
  const gi2 = _PERM12[ii + 1 + _PERM[jj + 1]];
  let n0 = 0, n1 = 0, n2 = 0;
  let t0 = 0.5 - x0*x0 - y0*y0; if (t0 >= 0) { t0 *= t0; n0 = t0*t0*_dot2(_GRAD3[gi0], x0, y0); }
  let t1 = 0.5 - x1*x1 - y1*y1; if (t1 >= 0) { t1 *= t1; n1 = t1*t1*_dot2(_GRAD3[gi1], x1, y1); }
  let t2 = 0.5 - x2*x2 - y2*y2; if (t2 >= 0) { t2 *= t2; n2 = t2*t2*_dot2(_GRAD3[gi2], x2, y2); }
  return 70 * (n0 + n1 + n2);
}

// ---------------------------------------------------------------------------
// Constants — Luna reference project tuned values
// ---------------------------------------------------------------------------
const DEG = Math.PI / 180;

const BREATHING = {
  rate: 0.16,           // Hz
  depth: 1.45,          // degrees
  chestRatio: 0.7,
  shoulderLift: 0.28,   // degrees
  headNod: 0.12,        // degrees
  emotionModifiers: {
    neutral:  { rate: 1.0,  depth: 1.0 },
    happy:    { rate: 1.08, depth: 1.08 },
    sad:      { rate: 0.85, depth: 1.25 },
    excited:  { rate: 1.25, depth: 0.95 },
    angry:    { rate: 1.2,  depth: 1.2 },
    relaxed:  { rate: 0.72, depth: 1.1 },
    thinking: { rate: 1.0,  depth: 1.0 },
    surprised:{ rate: 1.2,  depth: 1.0 },
    curious:  { rate: 1.0,  depth: 1.0 },
  },
};

const SWAY = {
  amount: 0.55,         // degrees
  speed: 0.045,
  hipWeight: 1.0,
  spineWeight: 0.45,
  headWeight: 0.3,
  counterSpine: -0.35,
  counterHead: -0.18,
};

const WEIGHT_SHIFT = {
  // Faster cycle so the shift completes within human attention span.
  // Prior 13-24s spanned multiple breaths and read as imperceptible drift.
  periodMin: 8,
  periodMax: 15,
  hipTilt: 1.0,
  shoulderCompensation: 0.16,
  headCompensation: -0.08,
};

// CCD-IK chains for foot lock — ported from ui/mockups/scene-test.html
// (PoseDirector's _applyFootLock). Two-bone chain per leg keeps the foot
// pinned at its captured rest world position while the hip rotates from
// breathing/sway/weight-shift. Without this, hip motion swings the feet
// through the floor — the brain reads it as "ungrounded" and the avatar
// loses visual weight even with all the other idle layers running.
//
// Per-bone Euler ranges prevent the knee from bending the wrong way:
// lowerLeg X must stay >= 0 so the joint flexes correctly.
//
// iterations:3 is enough for a 2-bone chain with the small hip drift of
// idle breath/sway (sub-degree per frame). Visually indistinguishable
// from iterations:8 — convergence after 3 inner passes leaves residual
// foot-target error well under one pixel at typical camera distances.
// Cuts the per-frame IK cost by ~60% vs. the test-bench-ported
// iterations:8 default. Bumped only if profile reveals visible foot
// drift on a specific bundled VRM.
const _PI = Math.PI;
const IK_LEG_CHAINS = {
  leftFoot: {
    iterations: 3,
    bones: [
      { name: 'leftLowerLeg',  order: 'XYZ', min: [0, 0, 0],          max: [_PI, 0, 0] },
      { name: 'leftUpperLeg',  order: 'XYZ', min: [-_PI, -_PI, -_PI], max: [_PI, _PI, _PI] },
    ],
  },
  rightFoot: {
    iterations: 3,
    bones: [
      { name: 'rightLowerLeg', order: 'XYZ', min: [0, 0, 0],          max: [_PI, 0, 0] },
      { name: 'rightUpperLeg', order: 'XYZ', min: [-_PI, -_PI, -_PI], max: [_PI, _PI, _PI] },
    ],
  },
};

const IDLE_ARM_POSE = {
  // Bundled VRMs load from a T-pose rest, so the neutral call pose needs
  // enough shoulder roll to land arms at the sides before idle layers
  // run. Values aligned with the cinematic mockup's `relaxArms` helper
  // (ui/mockups/voice-mode-cinematic.html, ~line 1658) which is what
  // visually reads as "person standing naturally" on the bundled VRMs:
  //   - Z=±80 brings the upper arms down close to vertical without
  //     touching the torso (T-pose has arms at 90° out, so this is
  //     ~10° gap from the body, the typical relaxed-stance look)
  //   - Y=±2 adds a tiny inward shoulder roll for naturalness
  //   - Lower arm X=6 is a subtle forward elbow bend
  // An earlier iteration tried the avatar-testbench's [15, 0, 64]
  // values which read as A-pose on the bundled cast — bone proportions
  // vary enough between VRMs that the same target angle settles
  // differently per model, and 64° wasn't enough rotation here.
  leftUpperArm:   [0,  2,  80],
  rightUpperArm:  [0, -2, -80],
  leftLowerArm:   [6,  0,  -3],
  rightLowerArm:  [6,  0,   3],
  leftHand:       [0,  0,   0],
  rightHand:      [0,  0,   0],
};

const ARM_TARGET_BONES = new Set([
  'leftUpperArm', 'rightUpperArm',
  'leftLowerArm', 'rightLowerArm',
  'leftHand', 'rightHand',
]);

// Bones the PoseOrchestrator is allowed to override per frame. Body
// only — arms stay with the spring channel (which carries breathing
// /sway/emotion deltas on top of IDLE_ARM_POSE) and fingers stay with
// the hand-pose channel. Idle family drift on these body bones is
// subtle enough (head tilt, hip lean, slight spine curve) that it
// doesn't fight the spring's breathing oscillation.
const _ORCHESTRATOR_OWNED_BONES = new Set([
  'head', 'neck',
  'spine', 'chest', 'upperChest',
  'hips',
]);

const FINGER_BONE_SUFFIXES = [
  'ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal',
  'IndexProximal', 'IndexIntermediate', 'IndexDistal',
  'MiddleProximal', 'MiddleIntermediate', 'MiddleDistal',
  'RingProximal', 'RingIntermediate', 'RingDistal',
  'LittleProximal', 'LittleIntermediate', 'LittleDistal',
];

const FINGER_TARGET_BONES = new Set([
  ...FINGER_BONE_SUFFIXES.map((suffix) => `left${suffix}`),
  ...FINGER_BONE_SUFFIXES.map((suffix) => `right${suffix}`),
]);

const PRODUCTION_CALL_ACTIONS = new Set([
  'call_acknowledge',
  'call_attentive_lean',
  'call_clarify_question',
  'call_key_point',
  'call_thoughtful_pause',
  'call_light_laugh',
  'call_grounding_breath',
  'call_wrap_up',
]);

const EMOTION_POSTURES = {
  neutral:  { spine: [1,0,0],    chest: [1,0,0],     head: [0,0,0],      shoulders: [0,0] },
  happy:    { spine: [-1.8,0,0], chest: [-1.2,0,0],  head: [-2.5,0,0],   shoulders: [-1.5,-1.5] },
  sad:      { spine: [4.5,0,0],  chest: [5.5,0,0],   head: [7,0,0],      shoulders: [2.5,2.5] },
  excited:  { spine: [-3,0,0],   chest: [-3,0,0],    head: [-4,0,0],     shoulders: [-2.5,-2.5] },
  thinking: { spine: [2,0,1],    chest: [2.5,0,0],   head: [4,5,2],      shoulders: [1,0] },
  angry:    { spine: [-1,0,0],   chest: [0,0,0],     head: [2.5,0,0],    shoulders: [3.5,3.5] },
  relaxed:  { spine: [1.6,0,-0.5], chest: [1.2,0,0], head: [1,-2,0],     shoulders: [0.8,0.8] },
  surprised:{ spine: [-1.5,0,0], chest: [-2,0,0],    head: [-3,0,0],     shoulders: [-1.5,-1.5] },
  curious:  { spine: [1.2,0,0.5], chest: [1.6,0,0],  head: [3,4,2],      shoulders: [0.5,0] },
};

const POSTURE_LERP_RATE = 2.0;  // per second — frame-rate independent

const EYE_CONFIG = {
  saccadeRate: 1.15,
  saccadeSpringHalflife: 0.09,
  tremorFreq: 4.5,
  tremorAmpX: 0.12,
  tremorAmpY: 0.08,
  vergenceDeg: 0.5,
  eyeMaxYaw: 24,
  eyeMaxPitch: 16,
  headMaxYaw: 30,
  headMaxPitch: 20,
  headFollowYaw: 0.34,
  headFollowPitch: 0.24,
  headFollowDelay: 0.24,
  contactRatio: 0.65,
  avertDurationMin: 1.5,
  avertDurationMax: 3.4,
  contactDurationMin: 3.0,
  contactDurationMax: 6.5,
  avertIdle: [-4, 2],
  avertThinking: [-10, 8],
  avertRecalling: [8, -6],
  avertProcessing: [5, -4],
  avertSpeaking: [-7, 4],
  blinkCloseMs: 55,
  blinkOpenMs: 105,
  blinkIntervalBase: 4.0,
  blinkIntervalVariance: 2.0,
  gazeEvokedBlinkChance: 0.7,
  gazeEvokedThreshold: 15,
  halfBlinkChance: 0.22,
  halfBlinkMax: 0.65,
  blinkRates: {
    // happy: dropped from 16 → 13. The VRM "happy" blendshape already
    // squints the eyes; a normal-human blink rate (16) on top compounds
    // into visible eye-clamping every ~4s. 13 gives the smile room to
    // settle.
    neutral: 14, happy: 13, sad: 11, excited: 20, angry: 18,
    nervous: 20, thinking: 12, surprised: 10, curious: 14, relaxed: 10,
  },
};

const VRM_EXPRESSION_PRESETS = {
  neutral:   {},
  // happy: dropped from 0.42 → 0.28. Most VRM "happy" blendshapes
  // bake eye-squint + mouth-open as a single morph, so 0.42 plus the
  // valence overlay (up to +0.30) hit 0.72 — a gaping joyful grin.
  // 0.28 lands on a gentle smile.
  happy:     { happy: 0.28, surprised: 0.04 },
  sad:       { sad: 0.45, relaxed: 0.05 },
  angry:     { angry: 0.42, surprised: 0.03 },
  surprised: { surprised: 0.55 },
  relaxed:   { relaxed: 0.18 },
  curious:   { surprised: 0.18, relaxed: 0.05 },
  excited:   { happy: 0.58, surprised: 0.14 },
  thinking:  { relaxed: 0.08 },
};

const EMOTION_TIMING = {
  neutral:   { attackMs: 300,  decayMs: 500  },
  happy:     { attackMs: 200,  decayMs: 1500 },
  sad:       { attackMs: 800,  decayMs: 2500 },
  angry:     { attackMs: 400,  decayMs: 2000 },
  surprised: { attackMs: 50,   decayMs: 500  },
  thinking:  { attackMs: 600,  decayMs: 1000 },
  excited:   { attackMs: 150,  decayMs: 1200 },
  nervous:   { attackMs: 500,  decayMs: 1500 },
  curious:   { attackMs: 400,  decayMs: 1200 },
  relaxed:   { attackMs: 500,  decayMs: 2000 },
};

const MOUTH_EXPRESSIONS = new Set(['happy', 'sad', 'angry', 'surprised']);
// During active TTS, scale mouth-bearing emotion blendshapes hard so
// visemes carry lip-sync without compounding with the emotion's own
// jaw motion.
const SPEECH_MOUTH_SCALE = 0.3;
// At idle (between sentences / when she's just there with positive
// valence), scale the same blendshapes to 0.6 so the smile reads as a
// smile, not a gape. Higher than SPEECH_MOUTH_SCALE because there are
// no competing visemes to drown out, but well under 1.0 because the
// VRM "happy" morph baked-in mouth opening is already overkill.
const SPEECH_IDLE_MOUTH_SCALE = 0.6;
const EXPRESSION_NOISE_AMP = 0.018;
const EXPRESSION_NOISE_FREQ = 0.5;

const VISEME_DECAY = 0.92;  // slower decay — mouth holds shape between syllables

// Half-life in seconds — time for displacement to halve
// Lower = snappier, higher = smoother
const SPRING_CONFIGS = {
  hips:          { halflife: 0.26 },
  spine:         { halflife: 0.20 },
  chest:         { halflife: 0.20 },
  upperChest:    { halflife: 0.20 },
  neck:          { halflife: 0.14 },
  head:          { halflife: 0.11 },
  leftUpperArm:  { halflife: 0.16 },
  rightUpperArm: { halflife: 0.16 },
  leftLowerArm:  { halflife: 0.16 },
  rightLowerArm: { halflife: 0.16 },
  leftShoulder:  { halflife: 0.18 },
  rightShoulder: { halflife: 0.18 },
  leftHand:      { halflife: 0.22 },
  rightHand:     { halflife: 0.22 },
  leftEye:       { halflife: 0.025 },
  rightEye:      { halflife: 0.025 },
};

// ---------------------------------------------------------------------------
// Procedural Gestures — one-shot keyframed bone target sequences
// Each gesture: { duration, keyframes: [{ t (0-1), bones: { boneName: [rx,ry,rz] in degrees } }] }
// The PD controller interpolates between keyframes smoothly.
// ---------------------------------------------------------------------------
const GESTURES = {
  nod: {
    duration: 1.0,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.35, bones: { head: [-7, 0, 0], neck: [-3, 0, 0] } },
      { t: 0.68, bones: { head: [2, 0, 0], neck: [1, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  shake: {
    duration: 1.15,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.22, bones: { head: [0, 8, 0], neck: [0, 3, 0] } },
      { t: 0.44, bones: { head: [0, -8, 0], neck: [0, -3, 0] } },
      { t: 0.66, bones: { head: [0, 6, 0], neck: [0, 2, 0] } },
      { t: 0.82, bones: { head: [0, -5, 0], neck: [0, -2, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  bow: {
    duration: 1.5,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.3, bones: { spine: [-15, 0, 0], chest: [-10, 0, 0], head: [-10, 0, 0] } },
      { t: 0.7, bones: { spine: [-15, 0, 0], chest: [-10, 0, 0], head: [-10, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  shrug: {
    duration: 0.9,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.25, bones: { leftShoulder: [-18, 0, 0], rightShoulder: [-18, 0, 0], head: [0, 0, 5] } },
      { t: 0.55, bones: { leftShoulder: [-18, 0, 0], rightShoulder: [-18, 0, 0], head: [0, 0, 5] } },
      { t: 1.0, bones: {} },
    ],
  },
  wave: {
    duration: 1.6,
    keyframes: [
      { t: 0.0,  bones: {} },
      { t: 0.15, bones: { rightUpperArm: [-60, -15, 0], rightLowerArm: [-40, 20, 0] } },
      { t: 0.35, bones: { rightUpperArm: [-60, -15, 0], rightLowerArm: [-40, 20, 0] } },
      { t: 0.5,  bones: { rightUpperArm: [-65, -20, 0], rightLowerArm: [-30, 25, 0] } },
      { t: 0.65, bones: { rightUpperArm: [-60, -15, 0], rightLowerArm: [-40, 15, 0] } },
      { t: 0.8,  bones: { rightUpperArm: [-65, -20, 0], rightLowerArm: [-30, 25, 0] } },
      { t: 1.0,  bones: {} },
    ],
  },
  laugh: {
    duration: 1.2,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.15, bones: { spine: [-3, 0, 0], head: [-5, 0, 2], chest: [-4, 0, 0] } },
      { t: 0.3, bones: { spine: [2, 0, 0], head: [3, 0, -2], chest: [2, 0, 0] } },
      { t: 0.45, bones: { spine: [-2, 0, 0], head: [-4, 0, 1], chest: [-3, 0, 0] } },
      { t: 0.6, bones: { spine: [1, 0, 0], head: [2, 0, -1], chest: [1, 0, 0] } },
      { t: 0.8, bones: { spine: [-1, 0, 0], head: [-2, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  think: {
    duration: 2.0,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.2, bones: { rightUpperArm: [-30, -30, -10], rightLowerArm: [-90, 20, 0], head: [8, 10, 5] } },
      { t: 0.7, bones: { rightUpperArm: [-30, -30, -10], rightLowerArm: [-90, 20, 0], head: [8, 10, 5] } },
      { t: 1.0, bones: {} },
    ],
  },
  surprise: {
    duration: 0.7,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.15, bones: { head: [-8, 0, 0], spine: [-3, 0, 0], leftShoulder: [-8, 0, 0], rightShoulder: [-8, 0, 0] } },
      { t: 0.5, bones: { head: [-5, 0, 0], spine: [-2, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  lean_forward: {
    duration: 1.4,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.25, bones: { spine: [-4.5, 0, 0], chest: [-3, 0, 0], head: [-2, 0, 0] } },
      { t: 0.7, bones: { spine: [-4.5, 0, 0], chest: [-3, 0, 0], head: [-2, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  lean_back: {
    duration: 1.7,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.3, bones: { spine: [3.5, 0, 0], chest: [2.5, 0, 0], head: [2.5, 0, 0] } },
      { t: 0.7, bones: { spine: [3.5, 0, 0], chest: [2.5, 0, 0], head: [2.5, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  head_tilt: {
    duration: 1.35,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.28, bones: { head: [1.5, 3, 4.5], neck: [0, 1, 2] } },
      { t: 0.68, bones: { head: [1.5, 3, 4.5], neck: [0, 1, 2] } },
      { t: 1.0, bones: {} },
    ],
  },
  point_up: {
    duration: 1.3,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.2, bones: { rightUpperArm: [-40, -20, 0], rightLowerArm: [-60, 10, 0], head: [-5, 5, 0] } },
      { t: 0.6, bones: { rightUpperArm: [-40, -20, 0], rightLowerArm: [-60, 10, 0], head: [-5, 5, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  hand_to_chin: {
    duration: 2.2,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.2, bones: { rightUpperArm: [-35, -25, -15], rightLowerArm: [-100, 15, 0], head: [5, 8, 3] } },
      { t: 0.75, bones: { rightUpperArm: [-35, -25, -15], rightLowerArm: [-100, 15, 0], head: [5, 8, 3] } },
      { t: 1.0, bones: {} },
    ],
  },
  cross_arms: {
    duration: 2.0,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.3, bones: { leftUpperArm: [10, 30, -50], rightUpperArm: [10, -30, 50], leftLowerArm: [-70, -30, 0], rightLowerArm: [-70, 30, 0], head: [5, 0, 0] } },
      { t: 0.7, bones: { leftUpperArm: [10, 30, -50], rightUpperArm: [10, -30, 50], leftLowerArm: [-70, -30, 0], rightLowerArm: [-70, 30, 0], head: [5, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  open_palms: {
    duration: 1.4,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.2, bones: { leftUpperArm: [5, 15, -45], rightUpperArm: [5, -15, 45], leftLowerArm: [-30, -20, 0], rightLowerArm: [-30, 20, 0] } },
      { t: 0.5, bones: { leftUpperArm: [5, 15, -45], rightUpperArm: [5, -15, 45], leftLowerArm: [-30, -20, 0], rightLowerArm: [-30, 20, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  look_away: {
    duration: 1.8,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.25, bones: { head: [2, -8, -1.5], neck: [0, -3.5, 0] } },
      { t: 0.72, bones: { head: [2, -8, -1.5], neck: [0, -3.5, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  weight_shift: {
    duration: 2.4,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.32, bones: { hips: [0, 0, 1.5], spine: [0, 0, -0.7], head: [0, 0, -0.4] } },
      { t: 0.72, bones: { hips: [0, 0, 1.5], spine: [0, 0, -0.7], head: [0, 0, -0.4] } },
      { t: 1.0, bones: {} },
    ],
  },
  look_around: {
    duration: 3.0,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.25, bones: { head: [0, 9, 0], neck: [0, 4, 0] } },
      { t: 0.52, bones: { head: [1, 9, 0], neck: [0, 4, 0] } },
      { t: 0.75, bones: { head: [0, -7, 0], neck: [0, -3, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  blink_slow: {
    duration: 1.2,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.3, bones: { head: [0.8, 0, 0] } },
      { t: 0.5, bones: { head: [0.8, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  deep_breath: {
    duration: 3.4,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.32, bones: { chest: [-2.5, 0, 0], upperChest: [-1.5, 0, 0], leftShoulder: [-1.2, 0, 0], rightShoulder: [-1.2, 0, 0], head: [-1.2, 0, 0] } },
      { t: 0.62, bones: { chest: [-2.5, 0, 0], upperChest: [-1.5, 0, 0], leftShoulder: [-1.2, 0, 0], rightShoulder: [-1.2, 0, 0], head: [-1.2, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  stretch_subtle: {
    duration: 2.8,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.25, bones: { spine: [-1.5, 0, 0], chest: [-2, 0, 0], leftShoulder: [-2.5, 0, -1.2], rightShoulder: [-2.5, 0, 1.2] } },
      { t: 0.58, bones: { spine: [-1.5, 0, 0], chest: [-2, 0, 0], leftShoulder: [-2.5, 0, -1.2], rightShoulder: [-2.5, 0, 1.2] } },
      { t: 1.0, bones: {} },
    ],
  },
  posture_adjust: {
    duration: 1.8,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.32, bones: { spine: [-2, 0, 0], chest: [-1.5, 0, 0], head: [-1, 0, 0] } },
      { t: 0.7, bones: { spine: [-2, 0, 0], chest: [-1.5, 0, 0], head: [-1, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
  sigh: {
    duration: 3.0,
    keyframes: [
      { t: 0.0, bones: {} },
      { t: 0.18, bones: { chest: [-1.5, 0, 0], head: [-1, 0, 0] } },
      { t: 0.45, bones: { chest: [2.5, 0, 0], head: [2.5, 0, 0], leftShoulder: [1.2, 0, 0], rightShoulder: [1.2, 0, 0] } },
      { t: 0.72, bones: { chest: [2.5, 0, 0], head: [2.5, 0, 0], leftShoulder: [1.2, 0, 0], rightShoulder: [1.2, 0, 0] } },
      { t: 1.0, bones: {} },
    ],
  },
};

// VRM bone name mapping
const VRM_BONE_NAMES = {
  hips: 'hips', spine: 'spine', chest: 'chest', upperChest: 'upperChest',
  neck: 'neck', head: 'head',
  leftUpperArm: 'leftUpperArm', rightUpperArm: 'rightUpperArm',
  leftLowerArm: 'leftLowerArm', rightLowerArm: 'rightLowerArm',
  leftShoulder: 'leftShoulder', rightShoulder: 'rightShoulder',
  leftHand: 'leftHand', rightHand: 'rightHand',
  leftEye: 'leftEye', rightEye: 'rightEye',
};

// ---------------------------------------------------------------------------
// CriticalSpring — Holden's critically damped exact solution
// Source: Daniel Holden, "Spring-It-On" (Ubisoft La Forge)
// One parameter (halflife) replaces stiffness+damping. Frame-rate independent.
// ---------------------------------------------------------------------------
class CriticalSpring {
  constructor(THREE, config) {
    this._THREE = THREE;
    this.halflife = config.halflife;

    // Target euler angles (radians)
    this.targetX = 0;
    this.targetY = 0;
    this.targetZ = 0;

    // Current state (radians)
    this.currentX = 0;
    this.currentY = 0;
    this.currentZ = 0;

    // Velocity (rad/s)
    this.velocityX = 0;
    this.velocityY = 0;
    this.velocityZ = 0;

    // Working objects
    this._quat = new THREE.Quaternion();
    this._euler = new THREE.Euler();
  }

  resetTarget() {
    this.targetX = 0;
    this.targetY = 0;
    this.targetZ = 0;
  }

  addTarget(xDeg, yDeg, zDeg) {
    this.targetX += xDeg * DEG;
    this.targetY += yDeg * DEG;
    this.targetZ += zDeg * DEG;
  }

  step(dt) {
    // Clamp dt to prevent numerical instability on lag spikes / tab switches
    const clampedDt = Math.min(dt, 0.1);
    const d = (4.0 * 0.6931472) / this.halflife; // ln(2) = 0.6931472
    const eydt = Math.exp(-d * clampedDt);
    const jdt = d * clampedDt;

    // X axis
    let cx = this.currentX - this.targetX;
    let j0x = this.velocityX + cx * d;
    this.currentX = this.targetX + (cx + j0x * clampedDt) * eydt;
    this.velocityX = (this.velocityX - j0x * jdt) * eydt;

    // Y axis
    let cy = this.currentY - this.targetY;
    let j0y = this.velocityY + cy * d;
    this.currentY = this.targetY + (cy + j0y * clampedDt) * eydt;
    this.velocityY = (this.velocityY - j0y * jdt) * eydt;

    // Z axis
    let cz = this.currentZ - this.targetZ;
    let j0z = this.velocityZ + cz * d;
    this.currentZ = this.targetZ + (cz + j0z * clampedDt) * eydt;
    this.velocityZ = (this.velocityZ - j0z * jdt) * eydt;
  }

  applyToBone(bone, restQuat) {
    if (!bone) return;
    this._euler.set(this.currentX, this.currentY, this.currentZ, 'XYZ');
    this._quat.setFromEuler(this._euler);
    bone.quaternion.copy(restQuat).multiply(this._quat);
  }
}

// ---------------------------------------------------------------------------
// AvatarAnimator — main animation controller
// ---------------------------------------------------------------------------
export class AvatarAnimator {
  /**
   * @param {object} THREE  — Three.js namespace
   * @param {object} vrm    — loaded VRM model
   */
  constructor(THREE, vrm) {
    this._THREE = THREE;
    this._vrm = vrm;
    this._clock = 0;
    this._disposed = false;
    this._callActionEuler = new THREE.Euler();
    this._callActionQuat = new THREE.Quaternion();
    // Scratch objects for the hand-pose channel (avoid per-frame
    // allocs in _applyHandPoses, which writes 30 finger joints per
    // call at rAF rate).
    this._fingerEuler = new THREE.Euler();
    this._fingerQuat = new THREE.Quaternion();

    // --- Cache bones ---
    this._bones = {};
    this._restQuats = {};
    const humanoid = vrm.humanoid;
    if (!humanoid) {
      console.warn('[AvatarAnimator] VRM has no humanoid — animation disabled');
      this._disabled = true;
      return;
    }
    this._disabled = false;

    // Prefer the BIND-rest stash loadVRM captures before applyPosePreset
    // runs. Without it, capturing the bone's current quaternion would
    // include the at-load natural pose (Z=-1.35 on left upper arm), and
    // the spring channel's `rest * delta` composition would add the
    // natural pose's -77° to IDLE_ARM_POSE's +80° → +3° = arms up. The
    // stash holds the post-resetNormalizedPose bind quat (typically
    // identity), so spring deltas land at their authored targets.
    // Falls back to current-state capture when loaded outside the
    // standard loadVRM path (test bench, mockup) where the stash isn't
    // populated — preserves prior behavior in those contexts.
    const stashedBoneRests = vrm.__augmentumBoneRestQuats || null;
    for (const [key, vrmName] of Object.entries(VRM_BONE_NAMES)) {
      const bone = humanoid.getNormalizedBoneNode(vrmName)
                || humanoid.getRawBoneNode(vrmName);
      if (bone) {
        this._bones[key] = bone;
        const stashed = stashedBoneRests?.[vrmName];
        this._restQuats[key] = stashed
          ? stashed.clone()
          : bone.quaternion.clone();
      }
    }

    // --- Foot-lock IK setup ---
    // Cache leg bones separately from this._bones (no springs run on them)
    // and snapshot foot world positions as the rest anchors. Capture has
    // to happen here, after the VRM is fully loaded but before any spring
    // ever runs, so the anchors reflect the canonical rest pose.
    this._ikBones = {};
    for (const name of ['leftUpperLeg', 'leftLowerLeg', 'leftFoot',
                        'rightUpperLeg', 'rightLowerLeg', 'rightFoot']) {
      const bone = humanoid.getNormalizedBoneNode?.(name) || humanoid.getRawBoneNode?.(name);
      if (bone) this._ikBones[name] = bone;
    }
    this._footAnchorLeft = null;
    this._footAnchorRight = null;
    this._footLockReady = false;
    if (this._ikBones.leftFoot && this._ikBones.rightFoot) {
      try {
        vrm.scene.updateMatrixWorld(true);
        this._footAnchorLeft = new THREE.Vector3();
        this._footAnchorRight = new THREE.Vector3();
        this._ikBones.leftFoot.getWorldPosition(this._footAnchorLeft);
        this._ikBones.rightFoot.getWorldPosition(this._footAnchorRight);
        this._footLockReady = true;
      } catch { /* leave disabled */ }
    }
    // Reusable scratch objects for the IK solver (avoid per-frame allocs).
    this._ikScratch = {
      bonePos:  new THREE.Vector3(),
      boneQuat: new THREE.Quaternion(),
      boneScale:new THREE.Vector3(),
      invQuat:  new THREE.Quaternion(),
      effWorld: new THREE.Vector3(),
      goalWorld:new THREE.Vector3(),
      toEff:    new THREE.Vector3(),
      toGoal:   new THREE.Vector3(),
      axis:     new THREE.Vector3(),
      rotQuat:  new THREE.Quaternion(),
      euler:    new THREE.Euler(),
    };

    this._callActionBones = { ...this._bones };
    this._callActionRestQuats = { ...this._restQuats };
    // Finger rests for call actions: prefer the stash captured in
    // loadVRM (clean post-resetNormalizedPose bind), fall back to the
    // bone's current quaternion. Same rationale as the hand-pose
    // channel below — once applyPosePreset(natural) writes to fingers
    // at load, the "current" quat is no longer rest.
    const stashedRestsForCallAction = vrm.__augmentumFingerRestQuats || null;
    for (const side of ['left', 'right']) {
      for (const suffix of FINGER_BONE_SUFFIXES) {
        const boneName = `${side}${suffix}`;
        const bone = humanoid.getNormalizedBoneNode?.(boneName)
                  || humanoid.getRawBoneNode?.(boneName);
        if (bone) {
          this._callActionBones[boneName] = bone;
          this._callActionRestQuats[boneName] = stashedRestsForCallAction?.[boneName]?.clone()
            || bone.quaternion.clone();
        }
      }
    }

    // --- Cache finger bones for the hand-pose channel ---
    // Separate structured cache (not the flat dict above) so the per-frame
    // hand-pose application can do `this._fingerBones[side][finger][joint]`
    // lookups without string concatenation. Finger bones missing on the
    // VRM (low-poly avatars) are simply absent from the structure;
    // _applyHandPoses skips silently per joint.
    //
    // Rest-quat snapshot lives alongside the bone cache. The hand-pose
    // channel writes rest-relative (`bone.quaternion = rest * delta`)
    // matching the call-action handshape path (_applyCallActionBoneTarget),
    // so the curl direction inherits from each bone's bind orientation
    // instead of assuming an axis convention.
    //
    // Prefers vrm.__augmentumFingerRestQuats (captured in loadVRM
    // immediately after resetNormalizedPose, before any pose
    // application). Falls back to the bone's current quaternion if the
    // VRM was loaded outside the standard path (test bench, mockup) —
    // in that case "rest" is whatever the bones currently say, which
    // matches pre-fix behavior on clean rest poses.
    const stashedFingerRests = vrm.__augmentumFingerRestQuats || null;
    this._fingerBones = { left: {}, right: {} };
    this._fingerRestQuats = { left: {}, right: {} };
    for (const side of ['left', 'right']) {
      for (const finger of FINGER_NAMES) {
        this._fingerBones[side][finger] = {};
        this._fingerRestQuats[side][finger] = {};
        for (const joint of JOINT_NAMES) {
          const boneName = fingerBoneName(side, finger, joint);
          const bone = humanoid.getNormalizedBoneNode?.(boneName)
                    || humanoid.getRawBoneNode?.(boneName);
          if (bone) {
            this._fingerBones[side][finger][joint] = bone;
            const stashed = stashedFingerRests?.[boneName];
            this._fingerRestQuats[side][finger][joint] = stashed
              ? stashed.clone()
              : bone.quaternion.clone();
          }
        }
      }
    }

    // Per-VRM finger axis sign. Stamped from the compatibility profile
    // (which the loader attached to vrm.__augmentumCompatibilityProfile)
    // by fingerAxisSignFromProfile — identity for VRMs that match the
    // HAND_POSES author convention, Z-flipped for the opposite profile.
    // Applied INSIDE _applyHandPoses as an additional multiplier on the
    // per-axis delta Euler, on top of the table's intrinsic left/right
    // mirror. Symmetric with _armTargetSign for the spring channel.
    const profile = vrm.__augmentumCompatibilityProfile;
    this._fingerAxisSign = fingerAxisSignFromProfile(profile?.fingerAxisProfile);

    // Pose-orchestrator family-drift. Mirrors scene-test's wiring (which
    // is what gives idle avatars their living-not-statue look between
    // VRMA plays). Activated below in constructor after spring init so
    // the orchestrator's slerp endpoints can read live bone state.
    // Gated behind VRMA / call-action precedence inside update() so it
    // never fights a higher-priority animation.
    this._poseOrchestrator = new PoseOrchestrator({
      three: THREE,
      armAxisSign: armAxisSignFromProfile(profile?.armAxisProfile, 'mirrored'),
      fingerAxisSign: this._fingerAxisSign,
      restPose: POSE_PRESETS.natural?.bones || null,
      // Live-bone provider lets the first slerp start from the live VRM
      // state rather than identity (avoids a T-pose flash at activation).
      boneStateProvider: (boneName) => {
        const node = vrm.humanoid?.getNormalizedBoneNode?.(boneName);
        return node?.quaternion || null;
      },
    });

    // Hand-pose channel: defaults to 'relaxed' so every avatar gets
    // natural-looking fingers without explicit setup. setHandPose()
    // overrides at runtime; state.handPose in update() overrides
    // per-frame (director-friendly). null disables the channel
    // entirely (lets a VRMA's baked finger animation through).
    this._handPoseLeft = 'relaxed';
    this._handPoseRight = 'relaxed';

    // --- Create spring controllers per bone ---
    this._springs = {};
    for (const [key, config] of Object.entries(SPRING_CONFIGS)) {
      if (this._bones[key]) {
        this._springs[key] = new CriticalSpring(THREE, config);
      }
    }

    // --- Validate gesture compatibility ---
    this._gestureCompat = {};
    const foundBones = new Set(Object.keys(this._bones));
    const missingBones = Object.keys(VRM_BONE_NAMES).filter(k => !foundBones.has(k));
    if (missingBones.length) {
      console.warn('[AvatarAnimator] VRM missing optional bones:', missingBones.join(', '));
    }

    for (const [gestureName, gesture] of Object.entries(GESTURES)) {
      // Collect all bones this gesture needs
      const needed = new Set();
      for (const kf of gesture.keyframes) {
        if (kf.bones) for (const b of Object.keys(kf.bones)) needed.add(b);
      }
      // Check which ones are available
      const available = [...needed].filter(b => this._springs[b]);
      const missing = [...needed].filter(b => !this._springs[b]);
      const coverage = needed.size > 0 ? available.length / needed.size : 1;
      this._gestureCompat[gestureName] = { coverage, missing };
      if (missing.length) {
        console.debug(`[AvatarAnimator] Gesture "${gestureName}" partial (${Math.round(coverage * 100)}%): missing ${missing.join(', ')}`);
      }
    }
    console.debug('[AvatarAnimator] Bones found:', [...foundBones].join(', '));
    console.debug('[AvatarAnimator] Gesture compatibility:', Object.fromEntries(
      Object.entries(this._gestureCompat).map(([k, v]) => [k, `${Math.round(v.coverage * 100)}%`])
    ));

    this._armTargetSign = this._detectArmTargetSign();
    this._callActionAvailableBones = new Set([
      ...Object.keys(this._springs),
      ...Object.keys(this._callActionBones).filter((name) => FINGER_TARGET_BONES.has(name)),
    ]);
    this._callActionScheduler = new CallActionScheduler({
      getCompatibility: (name, options = {}) => this._getCallActionCompatibility(name, options),
      now: () => this._clock,
    });
    this._callActionCompat = {};
    for (const name of Object.keys(CALL_ACTIONS)) {
      this._callActionCompat[name] = this._getCallActionCompatibility(name);
    }
    this._idleArmPose = Object.fromEntries(
      Object.entries(IDLE_ARM_POSE).map(([key, angles]) => [
        key,
        this._mapArmTarget(key, angles),
      ]),
    );

    // --- Set idle arm pose as the spring starting position ---
    for (const [key, angles] of Object.entries(this._idleArmPose)) {
      if (this._springs[key]) {
        this._springs[key].currentX = angles[0] * DEG;
        this._springs[key].currentY = angles[1] * DEG;
        this._springs[key].currentZ = angles[2] * DEG;
        this._springs[key].targetX = angles[0] * DEG;
        this._springs[key].targetY = angles[1] * DEG;
        this._springs[key].targetZ = angles[2] * DEG;
      }
    }

    // --- Emotion state ---
    this._emotion = 'neutral';
    this._currentPosture = { spine: [1,0,0], chest: [1,0,0], head: [0,0,0], shoulders: [0,0] };
    this._currentExpressionWeights = {};
    this._expressionTargetWeights = {};
    this._isSpeaking = false;

    // --- Gaze state ---
    this._gazeTarget = { yaw: 0, pitch: 0 };   // where to look (degrees)
    this._headFollow = { yaw: 0, pitch: 0 };    // delayed head target
    this._headFollowTimer = 0;
    this._saccadeTimer = 0;
    this._nextSaccadeAt = 0;

    // --- Gaze aversion state ---
    this._gazeState = 'contact';
    this._gazeStateTimer = 0;
    this._gazeStateDuration = this._randRange(
      EYE_CONFIG.contactDurationMin, EYE_CONFIG.contactDurationMax
    );
    this._avertTarget = { yaw: 0, pitch: 0 };
    this._conversationState = 'idle';

    // --- Saccade spring (for ballistic movement) ---
    this._saccadeSpring = { current: { yaw: 0, pitch: 0 }, velocity: { yaw: 0, pitch: 0 } };
    this._saccadeTarget = { yaw: 0, pitch: 0 };
    this._lastSaccadeMag = 0;

    // Default to InteroceptionEngine setpoints so identity multipliers
    // apply when no physiology source is wired. Must be set before
    // _getNextBlinkInterval() runs — it reads heart_rate.
    this._physiology = { heart_rate: 0.40, muscle_tension: 0.30 };

    // --- Blink state ---
    this._blinkTimer = 0;
    this._blinkDuration = 0;
    this._blinkPhase = 'idle';   // idle | closing | opening
    this._blinkCloseDuration = 0;
    this._blinkOpenDuration = 0;
    this._blinkMaxClose = 1.0;
    this._nextBlinkAt = this._getNextBlinkInterval();

    // --- Viseme state ---
    this._lastViseme = null;
    this._visemeWeight = 0;

    // --- Awareness state ---
    this._awarenessImpulse = null;
    this._awarenessTimer = 0;
    this._activeGesture = null;    // current one-shot gesture playing
    this._activeCallAction = null; // current audited call action playing
    this._callActionFrame = null;
    this._lastCallActionResult = null;
    this._silenceTime = 0;

    // --- Breathing phase ---
    this._breathPhase = 0;
    // Dimensional affect overlay (see setAffectModifier). 0.5 = identity;
    // matches the default InteroceptionEngine output until something
    // pushes physiology around.
    this._affectMod = { arousal: 0.5, valence: 0.5 };
    this._weightShiftPeriod = WEIGHT_SHIFT.periodMin + Math.random() * (WEIGHT_SHIFT.periodMax - WEIGHT_SHIFT.periodMin);
    this._weightShiftPhase = Math.random() * Math.PI * 2;
    this._deepBreathTimer = 45 + Math.random() * 45;
    this._isDeepBreath = false;
    this._deepBreathStartPhase = 0;
    this._breathRateMod = 1.0;
    this._breathDepthMod = 1.0;

    // Activate the orchestrator's idle_standing family so the avatar
    // imperceptibly drifts between natural / contrapposto / leaningIn /
    // leaningBack / handOnHip variants during idle stretches — same
    // behavior scene-test renders. The orchestrator runs after the
    // spring layer in update() and writes to bones the spring touches
    // too; the slow slerp (1-2s) plus the spring's small breath/sway
    // amplitudes don't visibly fight. Gated behind VRMA / call-action
    // precedence so canned animations and explicit-shape beats are
    // never overridden.
    try {
      this._poseOrchestrator.setFamily('idle_standing');
    } catch (err) {
      console.debug('[AvatarAnimator] pose-orchestrator setFamily failed', err);
    }
  }

  _detectArmTargetSign() {
    const leftX = this._getLocalChildX(this._bones.leftUpperArm, this._bones.leftLowerArm);
    const rightX = this._getLocalChildX(this._bones.rightUpperArm, this._bones.rightLowerArm);

    // Older bundled VRMs have left arms extending along local -X and right
    // arms along +X. Some VRM 1.0/VRoid 2 exports, including Vance, expose
    // the opposite local arm axis. In that profile, roll targets must be
    // mirrored or the neutral A-pose correction becomes a raised Y-pose.
    const mirroredArmAxis = leftX > 0.001 && rightX < -0.001;
    return mirroredArmAxis ? { x: 1, y: 1, z: -1 } : { x: 1, y: 1, z: 1 };
  }

  _getLocalChildX(parent, child) {
    if (!parent || !child) return 0;
    parent.updateWorldMatrix?.(true, false);
    child.updateWorldMatrix?.(true, false);
    const p = child.getWorldPosition(new this._THREE.Vector3());
    return parent.worldToLocal(p).x;
  }

  _mapArmTarget(boneName, angles) {
    if (!ARM_TARGET_BONES.has(boneName)) return angles;
    const sign = this._armTargetSign || { x: 1, y: 1, z: 1 };
    return [
      angles[0] * sign.x,
      angles[1] * sign.y,
      angles[2] * sign.z,
    ];
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Main update tick — call every frame.
   * @param {number} delta — seconds since last frame
   * @param {object} state — { speaking, rms, viseme, emotion, userSpeaking }
   */
  update(delta, state = {}) {
    if (this._disabled || this._disposed) return;

    // Clamp delta to prevent large jumps on tab switch / lag spike
    const dt = Math.min(delta, 0.1);
    this._clock += dt;

    // ── VRMA priority gate ──────────────────────────────────────────
    // When a VRMA is actively driving bones (set by avatar.js when
    // playVrma() is active), our procedural spring pipeline would
    // overwrite the canned animation each frame. Yield bone rotations
    // entirely to the VRMA — but keep running the face/expression
    // pipeline since blendshapes are a separate channel that doesn't
    // conflict with bone rotation tracks. Lipsync / blink / emotion
    // expressions still work during a wave/dance.
    if (state.vrmaActive) {
      this._updateViseme(state);
      this._isSpeaking = !!state.speaking;
      this._updateExpressions(dt);
      this._updateBlink(dt);
      // Body physics hook — desktop/XR contact + compliance + rapier tick
      // here, BEFORE vrm.update bakes the skeleton, so any compliance
      // deltas applied to bone quaternions are visible this frame.
      if (typeof state.onPreVrmUpdate === 'function') {
        try { state.onPreVrmUpdate(dt); }
        catch (err) { console.debug('[animator] onPreVrmUpdate threw:', err?.message); }
      }
      if (this._vrm.update) this._vrm.update(dt);
      return;
    }

    // 1. Reset all spring targets to zero
    for (const spring of Object.values(this._springs)) {
      spring.resetTarget();
    }

    // 2. Set idle arm pose targets (base pose)
    for (const [key, angles] of Object.entries(this._idleArmPose || IDLE_ARM_POSE)) {
      if (this._springs[key]) {
        this._springs[key].addTarget(angles[0], angles[1], angles[2]);
      }
    }

    // Liveliness (0..1) from the render loop — <1 when the frame cap is
    // throttled (coder mode under load). The high-frequency layers below
    // (sway, weight-shift, saccades, tremor) scale by it so low fps reads as a
    // calm hold, not jitter. Breathing/posture stay full (low-freq, smooth).
    this._liveliness = (state && state.liveliness != null) ? state.liveliness : 1;

    // 3. Additive layers set targets
    this._updateBreathing(dt, state);
    this._updateSway(dt);
    this._updateWeightShift(dt);
    this._updateEmotionPosture(dt);
    this._updateGaze(dt, state);
    this._updateAwareness(dt, state);
    this._updateGestures(dt);
    this._updateCallAction(dt);

    // 4. Spring step — critically damped exact solution
    for (const spring of Object.values(this._springs)) {
      spring.step(dt);
    }

    // 4b. Pose-orchestrator tick — runs BEFORE spring application so
    //     spring writes can layer their breath/sway delta on top of
    //     the orchestrator's idle-family base pose instead of the
    //     constructor-captured static rest. The orchestrator's slow
    //     slerp (1-2s) provides drifting posture; the springs (~0.2s
    //     halflife) provide breath/sway motion. Composed cleanly, the
    //     avatar reads as "alive, settled in a posture, breathing"
    //     rather than "statue with breath".
    let orchestratorSnapshot = null;
    if (!state.vrmaActive && !this._activeCallAction && this._poseOrchestrator?.isActive()) {
      this._poseOrchestrator.tick(performance.now());
      orchestratorSnapshot = this._poseOrchestrator.getCurrentPose();
    }

    // 5. Apply spring results to bones. For bones the orchestrator owns
    //    AND has a current quat for, use the orchestrator's quat as the
    //    spring's REST baseline (so delta layers on top). Otherwise the
    //    constructor-captured rest applies — same as before this change.
    for (const [key, spring] of Object.entries(this._springs)) {
      let baseQuat = this._restQuats[key];
      if (orchestratorSnapshot && _ORCHESTRATOR_OWNED_BONES.has(key)) {
        const orchState = orchestratorSnapshot[key];
        if (orchState?.quaternion) baseQuat = orchState.quaternion;
      }
      spring.applyToBone(this._bones[key], baseQuat);
    }
    this._applyCallActionDirect();

    // 5b. Foot-lock IK — solve the leg chains so the feet stay at their
    //     captured rest world position despite the hip rotation just
    //     written above. Must run BEFORE vrm.update() so the corrected
    //     leg rotations are visible to the renderer this frame.
    this._applyFootLock();

    // 5c. Hand-pose channel — write finger bones directly. Skipped when
    //     something else owns the fingers:
    //       - VRMA playback (smartphone grip, dance hand poses)
    //       - Active call-action (explanatory hand shape mid-utterance)
    //     Both write to the same finger bones; we step aside so they
    //     don't get clobbered. The animator's spring/IK layers don't
    //     touch finger bones, so applying on top of those is safe.
    //     Per-frame override via state.handPose has the same shape as
    //     setHandPose() — string or { left, right } object.
    if (!state.vrmaActive && !this._activeCallAction) {
      // Apply per-frame override if present, otherwise the persisted
      // _handPoseLeft / _handPoseRight values stand.
      let leftPose = this._handPoseLeft;
      let rightPose = this._handPoseRight;
      const override = state.handPose;
      if (typeof override === 'string') {
        if (HAND_POSES[override]) leftPose = rightPose = override;
      } else if (override && typeof override === 'object') {
        if ('left' in override) {
          leftPose = override.left == null ? null
            : (HAND_POSES[override.left] ? override.left : leftPose);
        }
        if ('right' in override) {
          rightPose = override.right == null ? null
            : (HAND_POSES[override.right] ? override.right : rightPose);
        }
      }
      this._applyHandPoses(leftPose, rightPose);
    }

    // 6. Update visemes with decay
    this._updateViseme(state);

    // 7. Update VRM expressions
    this._isSpeaking = !!state.speaking;
    this._updateExpressions(dt);
    this._applyCallActionExpressions();

    // 8. Update blink
    this._updateBlink(dt);

    // 8b. Body physics hook — desktop/XR contact + compliance + rapier
    // tick here, BEFORE vrm.update bakes the skeleton, so compliance
    // deltas right-multiplied onto bone quaternions and any IK reaches
    // are visible this frame.
    if (typeof state.onPreVrmUpdate === 'function') {
      try { state.onPreVrmUpdate(dt); }
      catch (err) { console.debug('[animator] onPreVrmUpdate threw:', err?.message); }
    }

    // 9. VRM update
    if (this._vrm.update) {
      this._vrm.update(dt);
    }
  }

  /** Set current emotion. */
  setEmotion(emotion) {
    if (EMOTION_POSTURES[emotion]) {
      this._emotion = emotion;
    }
  }

  /** Trigger an awareness reaction. */
  triggerAwareness(type) {
    this._awarenessImpulse = type;
    this._awarenessTimer = 0;
  }

  /**
   * Set breath rate/depth modifier from presence engine.
   * @param {{ rate: number, depth: number }} mod
   */
  setBreathModifier(mod) {
    this._breathRateMod = mod.rate;
    this._breathDepthMod = mod.depth;
  }

  /**
   * Dimensional affect overlay from InteroceptionEngine. Biases the
   * categorical emotion posture/expression with a continuous arousal
   * (energy) and valence (positivity) signal. Both are in [0,1]; 0.5
   * is the identity midpoint, so passing {0.5, 0.5} is a no-op against
   * the existing emotion layer.
   *
   * Kept subtle by design — affect is felt-state, not theatre. The
   * categorical EMOTION_POSTURES / VRM_EXPRESSION_PRESETS tables stay
   * in charge of the dominant pose; this just leans the body and face
   * slightly to match interoception.
   *
   * @param {{ arousal: number, valence: number }} mod
   */
  setAffectModifier(mod) {
    if (!mod) return;
    const a = Number(mod.arousal);
    const v = Number(mod.valence);
    if (Number.isFinite(a)) this._affectMod.arousal = Math.max(0, Math.min(1, a));
    if (Number.isFinite(v)) this._affectMod.valence = Math.max(0, Math.min(1, v));
  }

  /**
   * Snap the breath cycle to the start of an inhale and force one
   * deep-breath pass — reads as the avatar gathering breath to speak.
   * Called by avatar.js right as TTS playback starts so the visible
   * chest rise overlaps the first phoneme. No-op if we're already
   * mid-inhale (don't yank an in-progress breath).
   */
  triggerInhale() {
    const phase = this._breathPhase % (Math.PI * 2);
    if (phase > Math.PI * 0.5) {
      this._breathPhase = Math.PI * 2 * Math.ceil(this._breathPhase / (Math.PI * 2));
    }
    this._isDeepBreath = true;
    this._deepBreathStartPhase = this._breathPhase;
    this._deepBreathTimer = 45 + Math.random() * 45;
  }

  /**
   * Continuous physiology coupling. heart_rate biases blink cadence
   * (faster heart → quicker blinks), muscle_tension biases sway
   * amplitude (stiff body sways less). Both default to interoception
   * setpoints {0.40, 0.30}, where the multipliers resolve to identity
   * so omitting the call leaves behavior unchanged.
   *
   * @param {{heart_rate?: number, muscle_tension?: number}} p
   */
  setPhysiology(p) {
    if (!p) return;
    const hr = Number(p.heart_rate);
    const mt = Number(p.muscle_tension);
    if (Number.isFinite(hr)) this._physiology.heart_rate = Math.max(0, Math.min(1, hr));
    if (Number.isFinite(mt)) this._physiology.muscle_tension = Math.max(0, Math.min(1, mt));
  }

  /**
   * Steer the pose orchestrator's idle family. Caller decides intent
   * (typically from PresenceEngine.flow: speaking → 'idle_standing',
   * listening → 'idle_engaged'); the orchestrator dedups same-family
   * calls internally, so calling this every frame is safe.
   *
   * @param {string} family  one of POSE_FAMILIES keys
   */
  setPoseIntent(family) {
    if (!family || !this._poseOrchestrator) return;
    try {
      this._poseOrchestrator.setFamily(family);
    } catch (err) {
      console.debug('[AvatarAnimator] setPoseIntent failed', err);
    }
  }

  /**
   * Returns the current breath phase (0-1, 0=start inhale, 0.4=peak, 1=end exhale).
   * Used by presence engine to sync gestures to breath cycle.
   */
  getBreathPhase() {
    return (this._breathPhase % (Math.PI * 2)) / (Math.PI * 2);
  }

  /**
   * Set the hand pose. Accepts:
   *   - string: applied to both hands ('relaxed', 'fist', 'open', 'holding',
   *             'point', 'peace', 'pinch', 'waving')
   *   - { left, right }: per-side override; either field may be omitted
   *                      (keeps current) or null (disables that side)
   *   - null: disables the hand-pose channel on both sides — finger bones
   *           are not written, so a VRMA's baked finger animation can
   *           come through unobstructed
   *
   * Unknown pose names are silently ignored (channel keeps prior value).
   */
  setHandPose(value) {
    if (value === null) {
      this._handPoseLeft = null;
      this._handPoseRight = null;
      return;
    }
    if (typeof value === 'string') {
      if (HAND_POSES[value]) {
        this._handPoseLeft = value;
        this._handPoseRight = value;
      }
      return;
    }
    if (value && typeof value === 'object') {
      if ('left' in value) {
        this._handPoseLeft = value.left == null ? null
          : (HAND_POSES[value.left] ? value.left : this._handPoseLeft);
      }
      if ('right' in value) {
        this._handPoseRight = value.right == null ? null
          : (HAND_POSES[value.right] ? value.right : this._handPoseRight);
      }
    }
  }

  /**
   * Release any "I'm the active speaker" hand state so this animator can
   * cleanly transition to listening without a frozen mid-gesture pose.
   *
   * Called from avatar.js's onSpeakerSwitch on the now-secondary
   * animator. In solo this is never invoked. The two things being
   * released:
   *   1. Active call action — long-running, often holds an explicit
   *      hand shape (e.g. 'holding' during a "let me explain" beat).
   *      Cancelled via _clearCallAction, which writes the rest pose
   *      back onto the affected finger bones so they don't freeze.
   *   2. Hand-pose channel — reset to 'relaxed', the baseline. Stops a
   *      lingering 'point' / 'open' / 'fist' from the prior speaking
   *      turn writing finger bones every frame.
   *
   * In-progress GESTURES (the procedural one-shots: nod, wave, surprise)
   * are intentionally left to finish — they're short (~1s) and
   * non-blocking. Aborting them would look more jarring than letting
   * them play out.
   */
  releaseActiveSpeakerState() {
    this._clearCallAction();
    this._handPoseLeft = 'relaxed';
    this._handPoseRight = 'relaxed';
  }

  /**
   * Trigger a one-shot procedural gesture by name.
   * Gestures are keyframed bone target sequences that the PD controller
   * interpolates through. They overlay on top of other animation layers.
   */
  triggerGesture(name, options = {}) {
    if (CALL_ACTIONS[name]) {
      const contract = getCallActionContract(name);
      const fallback = contract?.fallback || 'call_acknowledge';
      const callActionName = PRODUCTION_CALL_ACTIONS.has(name)
        ? name
        : (PRODUCTION_CALL_ACTIONS.has(fallback) ? fallback : 'call_acknowledge');
      return this._triggerCallAction(callActionName, options);
    }

    const gesture = GESTURES[name];
    if (!gesture) return false;

    // Skip gestures that have no compatible bones on this model
    const compat = this._gestureCompat?.[name];
    if (compat && compat.coverage === 0) {
      // Try a universal fallback (head-only gestures work on all
      // VRMs). Spread across nod/head_tilt — three gestures all
      // collapsing to 'nod' turned variety into MORE nodding on
      // models missing arm clips (2026-06-11).
      const fallbacks = { wave: 'nod', open_palms: 'head_tilt', cross_arms: 'head_tilt',
                          point_up: 'head_tilt', hand_to_chin: 'head_tilt', stretch_subtle: 'deep_breath' };
      const fb = fallbacks[name];
      if (fb && fb !== name && GESTURES[fb]) {
        this.triggerGesture(fb, options);
        return true;
      }
      return false; // no fallback, skip entirely
    }

    const highPriority = new Set(['surprise', 'laugh', 'wave']);
    if (this._activeGesture && !highPriority.has(name)) return false;
    this._clearCallAction();
    this._activeGesture = { name, ...gesture, elapsed: 0 };
    return true;
  }

  _getCallActionCompatibility(name, options = {}) {
    return assessCallActionCompatibility(name, this._callActionAvailableBones, {
      phase: options.phase || this._getCallActionPhase(),
    });
  }

  _getCallActionPhase() {
    if (this._conversationState === 'thinking') return 'processing';
    return this._conversationState || 'idle';
  }

  _triggerCallAction(name, options = {}) {
    if (!CALL_ACTIONS[name] || !this._callActionScheduler) return false;

    const result = this._callActionScheduler.request(name, {
      availableBones: this._callActionAvailableBones,
      mode: options.mode || 'auto',
      phase: options.phase || this._getCallActionPhase(),
      now: this._clock,
      ignoreCooldown: options.ignoreCooldown === true,
    });
    this._lastCallActionResult = result;
    if (!result.accepted || !result.action) return false;

    this._activeGesture = null;
    this._clearCallAction();
    this._activeCallAction = {
      name: result.name,
      requestedName: name,
      action: result.action,
      elapsed: 0,
      expressionNames: new Set(Object.keys(result.action.expression || {})),
    };
    return true;
  }

  _clearCallAction() {
    if (this._activeCallAction?.action) {
      this._resetCallActionHands(this._activeCallAction.action);
    }
    this._clearCallActionExpressions();
    this._activeCallAction = null;
    this._callActionFrame = null;
  }

  _clearCallActionExpressions() {
    const expressionManager = this._vrm.expressionManager;
    if (!expressionManager || !this._activeCallAction?.expressionNames) return;
    for (const name of this._activeCallAction.expressionNames) {
      expressionManager.setValue?.(name, 0);
    }
  }

  _resetCallActionHands(action) {
    // Previously this loop wrote `bone.quaternion.copy(rest)` — i.e.
    // restored finger bones to their VRM-rest (identity) quaternion.
    // That used to be the "go back to a known baseline" step. With the
    // hand-pose channel as the fingers' deterministic baseline, that
    // reset actively caused the finger-flicker bug:
    //
    //   - _clearCallAction can fire OUTSIDE the animator update loop
    //     (synchronous external trigger from triggerGesture / new
    //     call-action accept). If a render frame happens between that
    //     reset and the next animator.update, the user sees one frame
    //     of straight (T-pose) fingers before the hand-pose channel
    //     re-applies relaxed.
    //
    // Keeping this as a no-op leaves fingers wherever the prior
    // call-action's slerp left them; the next animator.update writes
    // the hand-pose baseline (relaxed by default) and a new call-action
    // (if triggered) slerps cleanly from that baseline. No flicker.
    void action; // referenced parameter retained for shape compatibility
  }

  /** Dispose the animator. */
  dispose() {
    this._clearCallAction();
    this._callActionScheduler?.clear?.();
    // Drop the orchestrator (it doesn't own any DOM/GL resources, but
    // dropping the reference lets its per-VRM bone-state provider GC
    // alongside the animator instance).
    this._poseOrchestrator = null;
    this._disposed = true;
    this._bones = {};
    this._springs = {};
    this._restQuats = {};
    this._callActionBones = {};
    this._callActionRestQuats = {};
    this._callActionEuler = null;
    this._callActionQuat = null;
  }

  // -------------------------------------------------------------------------
  // Layer: Breathing — emotion-modulated
  // -------------------------------------------------------------------------
  _updateBreathing(dt, state) {
    const emo = this._emotion;
    const mod = BREATHING.emotionModifiers[emo] || BREATHING.emotionModifiers.neutral;
    let rate = BREATHING.rate * mod.rate * this._breathRateMod;
    let depth = BREATHING.depth * mod.depth * this._breathDepthMod;

    // Deep breath check
    this._deepBreathTimer -= dt;
    if (this._deepBreathTimer <= 0 && !this._isDeepBreath) {
      this._isDeepBreath = true;
      this._deepBreathStartPhase = this._breathPhase;
      this._deepBreathTimer = 45 + Math.random() * 45;
    }
    if (this._isDeepBreath) {
      depth *= 1.25;
      rate *= 0.7;
      if (this._breathPhase - this._deepBreathStartPhase > Math.PI * 2) {
        this._isDeepBreath = false;
      }
    }

    this._breathPhase += dt * rate * Math.PI * 2;

    // Asymmetric wave: inhale faster (40%), exhale slower (60%)
    const rawPhase = this._breathPhase % (Math.PI * 2);
    let wave;
    if (rawPhase < Math.PI * 0.8) {
      wave = Math.sin(rawPhase / 0.8 * (Math.PI / 2));
    } else {
      const exhalePhase = (rawPhase - Math.PI * 0.8) / 1.2;
      wave = Math.cos(exhalePhase * (Math.PI / 2));
    }

    const chestDeg = wave * depth * BREATHING.chestRatio;
    const upperChestDeg = wave * depth * (1 - BREATHING.chestRatio);
    const shoulderDeg = wave * BREATHING.shoulderLift;
    const headDeg = wave * BREATHING.headNod;

    // Abdominal component: slight spine forward lean on exhale
    const abdominal = (1 - wave) * depth * 0.15;

    if (this._springs.chest) this._springs.chest.addTarget(chestDeg, 0, 0);
    if (this._springs.upperChest) this._springs.upperChest.addTarget(upperChestDeg, 0, 0);
    if (this._springs.spine) this._springs.spine.addTarget(abdominal, 0, 0);
    if (this._springs.leftShoulder) this._springs.leftShoulder.addTarget(0, 0, -shoulderDeg);
    if (this._springs.rightShoulder) this._springs.rightShoulder.addTarget(0, 0, shoulderDeg);
    if (this._springs.head) this._springs.head.addTarget(headDeg, 0, 0);

    // Hands: subtle wrist sway following breath cycle (relaxed pendulum effect)
    const handSway = wave * 0.35; // slight wrist flex on inhale
    if (this._springs.leftHand) this._springs.leftHand.addTarget(handSway, 0, 0);
    if (this._springs.rightHand) this._springs.rightHand.addTarget(handSway, 0, 0);
  }

  // -------------------------------------------------------------------------
  // Layer: Sway — simplex noise with counter-sway
  // -------------------------------------------------------------------------
  _updateSway(dt) {
    const t = this._clock * SWAY.speed;
    // Muscle-tension multiplier — high tension dampens all sway layers
    // (rigid body holds still); low tension amplifies them slightly
    // (loose body weight-shifts more). 0.30 = setpoint = identity.
    const tens = this._physiology.muscle_tension;
    const tensionMul = 1.0 - (tens - 0.30) * 1.2;
    const swayMul = Math.max(0.2, Math.min(1.6, tensionMul))
      * (this._liveliness == null ? 1 : this._liveliness);
    const hipSway = simplex2D(t, 0) * SWAY.amount * swayMul;

    // Layered periodic sway — simplex noise gives organic randomness, but
    // by itself the period is too long to read as "weight shifting." Add
    // two faster sine waves at different rates (test bench values: 0.4Hz Z
    // roll, 0.27Hz X tilt) to surface deliberate-looking weight motion on
    // top of the noise without overwriting it.
    const periodicZ = Math.sin(this._clock * 0.4 * Math.PI * 2)         * 0.35 * swayMul;
    const periodicX = Math.sin(this._clock * 0.27 * Math.PI * 2 + 1.0)  * 0.23 * swayMul;

    if (this._springs.hips) {
      this._springs.hips.addTarget(periodicX, 0, hipSway * SWAY.hipWeight + periodicZ);
    }
    // Counter-sway on spine
    if (this._springs.spine) {
      this._springs.spine.addTarget(0, 0, hipSway * SWAY.counterSpine);
    }
    // Secondary noise on spine Y
    const spineY = simplex2D(t + 100, 0) * SWAY.amount * SWAY.spineWeight * swayMul;
    if (this._springs.spine) {
      this._springs.spine.addTarget(0, spineY, 0);
    }
    // Counter-sway on head
    if (this._springs.head) {
      this._springs.head.addTarget(0, 0, hipSway * SWAY.counterHead);
    }

    // Micro head movements — subtle noise layer
    const microX = simplex2D(this._clock * 0.55, 100) * 0.12 * swayMul;
    const microY = simplex2D(this._clock * 0.45, 200) * 0.08 * swayMul;
    if (this._springs.head) {
      this._springs.head.addTarget(microX, microY, 0);
    }
  }

  // -------------------------------------------------------------------------
  // Layer: Weight Shift — slow hip/shoulder cycle
  // -------------------------------------------------------------------------
  _updateWeightShift(dt) {
    this._weightShiftPhase += dt * (Math.PI * 2 / this._weightShiftPeriod);
    const shift = Math.sin(this._weightShiftPhase) * WEIGHT_SHIFT.hipTilt
      * (this._liveliness == null ? 1 : this._liveliness);

    if (this._springs.hips) {
      this._springs.hips.addTarget(0, 0, shift);
    }
    if (this._springs.leftShoulder) {
      this._springs.leftShoulder.addTarget(shift * WEIGHT_SHIFT.shoulderCompensation, 0, 0);
    }
    if (this._springs.rightShoulder) {
      this._springs.rightShoulder.addTarget(-shift * WEIGHT_SHIFT.shoulderCompensation, 0, 0);
    }
    if (this._springs.head) {
      this._springs.head.addTarget(0, 0, shift * WEIGHT_SHIFT.headCompensation);
    }
  }

  // -------------------------------------------------------------------------
  // Hand-pose channel — write finger bone rotations from HAND_POSES
  // -------------------------------------------------------------------------
  // Rest-relative write: bone.quaternion = restQuat * deltaQuat.
  //
  // Previously this set bone.rotation absolutely. That worked on the
  // bundled roster (HAND_POSES authored against them, identity normalized
  // rest pose) but produced visible dorsal hyperextension on externally
  // imported VRMs whose finger bones land in the opposite axis convention.
  // The rewrite is two-pronged:
  //   1. Rest-relative composition — same idiom _applyCallActionBoneTarget
  //      uses for call-action handshapes. Identity-rest bones reproduce
  //      the old visual; non-identity-rest bones now compose correctly.
  //   2. Per-VRM finger axis sign — this._fingerAxisSign, derived from
  //      the compatibility profile's fingerAxisProfile detection, layers
  //      on top of the table's intrinsic left/right mirror. Mirrors the
  //      spring channel's _armTargetSign / _mapArmTarget approach.
  //
  // Conventional `null` pose name = skip this side (lets a VRMA's
  // baked finger animation pass through unobstructed).
  _applyHandPoses(leftPoseName, rightPoseName) {
    const sign = this._fingerAxisSign || { x: 1, y: 1, z: 1 };
    const sides = [
      [leftPoseName, this._fingerBones.left, this._fingerRestQuats.left, false],
      [rightPoseName, this._fingerBones.right, this._fingerRestQuats.right, true],
    ];
    for (const [poseName, fingerSet, restSet, isRight] of sides) {
      if (!poseName) continue;
      const pose = HAND_POSES[poseName];
      if (!pose) continue;
      for (const finger of FINGER_NAMES) {
        const segs = pose[finger];
        if (!segs) continue;
        const fingerBones = fingerSet[finger];
        const fingerRests = restSet[finger];
        if (!fingerBones || !fingerRests) continue;
        for (const joint of JOINT_NAMES) {
          const bone = fingerBones[joint];
          const rest = fingerRests[joint];
          if (!bone || !rest) continue;
          const rot = segs[joint];
          if (!rot) continue;
          // Mirror Y/Z for right hand (table is authored for left), then
          // multiply by per-axis sign factor for opposite-convention VRMs.
          const x = rot[0] * sign.x;
          const y = (isRight ? -rot[1] : rot[1]) * sign.y;
          const z = (isRight ? -rot[2] : rot[2]) * sign.z;
          this._fingerEuler.set(x, y, z, 'XYZ');
          this._fingerQuat.setFromEuler(this._fingerEuler);
          bone.quaternion.copy(rest).multiply(this._fingerQuat);
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Foot-lock IK — keep feet planted while hips rotate
  // -------------------------------------------------------------------------
  _applyFootLock() {
    if (!this._footLockReady) return;
    const leftFoot = this._ikBones.leftFoot;
    const rightFoot = this._ikBones.rightFoot;
    if (!leftFoot || !rightFoot) return;
    const hips = this._bones.hips;
    if (!hips) return;
    // Propagate the hip rotation we just wrote down through the leg
    // chain so the foot world positions reflect the pre-IK state.
    hips.updateMatrixWorld(true);
    this._solveIK(IK_LEG_CHAINS.leftFoot,  'leftFoot',  this._footAnchorLeft);
    this._solveIK(IK_LEG_CHAINS.rightFoot, 'rightFoot', this._footAnchorRight);
  }

  // CCD-IK solver — ported from PoseDirector / scene-test.html. For each
  // iteration, walks leaf-to-root, rotates each bone to align (bone→eff)
  // with (bone→goal), clamps the result to the bone's per-axis Euler
  // ranges. updateMatrixWorld(true) propagates each write to descendants
  // so the next bone in the iteration sees a fresh effector position.
  _solveIK(chain, effectorBoneName, goalVec3) {
    const humanoid = this._vrm?.humanoid;
    if (!humanoid || !chain) return;
    const effectorNode = humanoid.getNormalizedBoneNode?.(effectorBoneName);
    if (!effectorNode) return;
    const s = this._ikScratch;
    s.goalWorld.copy(goalVec3);

    const iterations = chain.iterations || 8;
    for (let iter = 0; iter < iterations; iter++) {
      let converged = true;
      for (const cfg of chain.bones) {
        const node = humanoid.getNormalizedBoneNode?.(cfg.name);
        if (!node) continue;

        node.matrixWorld.decompose(s.bonePos, s.boneQuat, s.boneScale);
        s.invQuat.copy(s.boneQuat).invert();

        effectorNode.getWorldPosition(s.effWorld);
        s.toEff.subVectors(s.effWorld, s.bonePos).applyQuaternion(s.invQuat).normalize();
        s.toGoal.subVectors(s.goalWorld, s.bonePos).applyQuaternion(s.invQuat).normalize();

        let cosTheta = s.toGoal.dot(s.toEff);
        if (cosTheta > 1) cosTheta = 1; else if (cosTheta < -1) cosTheta = -1;
        const angle = Math.acos(cosTheta);
        if (angle < 1e-5) continue;  // already aligned

        s.axis.crossVectors(s.toEff, s.toGoal);
        if (s.axis.lengthSq() < 1e-10) continue;  // parallel
        s.axis.normalize();
        s.rotQuat.setFromAxisAngle(s.axis, angle);
        node.quaternion.multiply(s.rotQuat);

        // Clamp to the bone's per-axis Euler range so the knee can't
        // hyperextend or twist unnaturally.
        s.euler.setFromQuaternion(node.quaternion, cfg.order);
        if (s.euler.x < cfg.min[0]) s.euler.x = cfg.min[0];
        else if (s.euler.x > cfg.max[0]) s.euler.x = cfg.max[0];
        if (s.euler.y < cfg.min[1]) s.euler.y = cfg.min[1];
        else if (s.euler.y > cfg.max[1]) s.euler.y = cfg.max[1];
        if (s.euler.z < cfg.min[2]) s.euler.z = cfg.min[2];
        else if (s.euler.z > cfg.max[2]) s.euler.z = cfg.max[2];
        node.rotation.order = cfg.order;
        node.quaternion.setFromEuler(s.euler);
        node.updateMatrixWorld(true);
        converged = false;
      }
      if (converged) break;
    }
  }

  // -------------------------------------------------------------------------
  // Layer: Emotion posture — lerp toward target + VRM expressions
  // -------------------------------------------------------------------------
  _updateEmotionPosture(dt) {
    const target = EMOTION_POSTURES[this._emotion] || EMOTION_POSTURES.neutral;
    const cur = this._currentPosture;

    // Frame-rate independent exponential decay toward target
    const alpha = 1 - Math.exp(-POSTURE_LERP_RATE * dt);
    for (const part of ['spine', 'chest', 'head']) {
      for (let i = 0; i < 3; i++) {
        cur[part][i] += (target[part][i] - cur[part][i]) * alpha;
      }
    }
    for (let i = 0; i < 2; i++) {
      cur.shoulders[i] += (target.shoulders[i] - cur.shoulders[i]) * alpha;
    }

    // Apply to PD targets
    if (this._springs.spine) {
      this._springs.spine.addTarget(cur.spine[0], cur.spine[1], cur.spine[2]);
    }
    if (this._springs.chest) {
      this._springs.chest.addTarget(cur.chest[0], cur.chest[1], cur.chest[2]);
    }
    if (this._springs.head) {
      this._springs.head.addTarget(cur.head[0], cur.head[1], cur.head[2]);
    }
    if (this._springs.leftShoulder) {
      this._springs.leftShoulder.addTarget(cur.shoulders[0], 0, 0);
    }
    if (this._springs.rightShoulder) {
      this._springs.rightShoulder.addTarget(cur.shoulders[1], 0, 0);
    }

    // Dimensional affect overlay — small continuous bias on top of the
    // categorical posture. Arousal lifts the spine and chest (low
    // arousal slumps); valence lifts the shoulders (low valence drops
    // them). Magnitudes kept under the smallest categorical delta so
    // the named-emotion layer keeps clear authority.
    const ad = this._affectMod.arousal - 0.5;
    const vd = this._affectMod.valence - 0.5;
    if (Math.abs(ad) > 0.001 || Math.abs(vd) > 0.001) {
      const spinePitchBias = -ad * 2.5;   // deg, upright when aroused
      const chestPitchBias = -ad * 1.5;
      const shoulderBias   =  vd * 3.0;   // deg, up when positive
      if (this._springs.spine) {
        this._springs.spine.addTarget(spinePitchBias, 0, 0);
      }
      if (this._springs.chest) {
        this._springs.chest.addTarget(chestPitchBias, 0, 0);
      }
      if (this._springs.leftShoulder) {
        this._springs.leftShoulder.addTarget(shoulderBias, 0, 0);
      }
      if (this._springs.rightShoulder) {
        this._springs.rightShoulder.addTarget(shoulderBias, 0, 0);
      }
    }
  }

  // -------------------------------------------------------------------------
  // VRM Expression blending
  // -------------------------------------------------------------------------
  _updateExpressions(dt) {
    const expressionManager = this._vrm.expressionManager;
    if (!expressionManager) return;

    const targetPreset = VRM_EXPRESSION_PRESETS[this._emotion] || {};
    const timing = EMOTION_TIMING[this._emotion] || EMOTION_TIMING.neutral;

    // Dimensional valence overlay — continuous bias on top of the
    // categorical preset. Positive valence adds a small happy weight,
    // negative valence a small sad weight. The categorical preset still
    // dominates; this just nudges baseline warmth/coolness.
    //
    // Multiplier dropped 0.3 → 0.15: combined with the happy preset
    // (0.28) the old 0.3 overlay could push total happy weight to 0.72,
    // which on most VRM "happy" morphs reads as a wide grin + squinted
    // eyes. 0.15 keeps the peak around 0.43 — a smile, not a beam.
    const vd = this._affectMod.valence - 0.5;
    const valenceHappy = Math.max(0,  vd) * 0.15;
    const valenceSad   = Math.max(0, -vd) * 0.15;

    const allNames = new Set([
      'happy', 'sad', 'angry', 'surprised', 'relaxed',
      ...Object.keys(this._currentExpressionWeights),
      ...Object.keys(targetPreset),
    ]);

    for (const name of allNames) {
      let target = targetPreset[name] || 0;
      if (name === 'happy') target = Math.min(1, target + valenceHappy);
      else if (name === 'sad') target = Math.min(1, target + valenceSad);
      const current = this._currentExpressionWeights[name] || 0;

      const isApproaching = Math.abs(target) > Math.abs(current);
      const tau = (isApproaching ? timing.attackMs : timing.decayMs) / 1000;
      const alpha = 1 - Math.exp(-dt / tau);
      let next = current + (target - current) * alpha;

      // Mouth-bearing emotion blendshapes (happy/sad/angry/surprised)
      // open the mouth as part of the morph. We scale them down across
      // ALL states, not just during active TTS:
      //   * Speaking: scale to SPEECH_MOUTH_SCALE so visemes can carry
      //     the lip-sync without compounding mouth motion.
      //   * Not speaking: scale to SPEECH_IDLE_MOUTH_SCALE so the smile
      //     doesn't gape between sentences or while she's just sitting
      //     there with positive valence. Previously this was an
      //     ``_isSpeaking``-gated suppression — between sentences the
      //     mouth would pop back to full preset weight, creating an
      //     "off and on" intensity pulse.
      if (MOUTH_EXPRESSIONS.has(name)) {
        next *= this._isSpeaking ? SPEECH_MOUTH_SCALE : SPEECH_IDLE_MOUTH_SCALE;
      }

      if (next > 0.1) {
        const seed = name.charCodeAt(0) * 7 + name.charCodeAt(1) * 13;
        const noise = simplex2D(this._clock * EXPRESSION_NOISE_FREQ, seed) * EXPRESSION_NOISE_AMP;
        next = Math.max(0, next + noise);
      }

      if (next < 0.001 && target === 0) {
        delete this._currentExpressionWeights[name];
        expressionManager.setValue(name, 0);
      } else {
        next = Math.min(1.0, Math.max(0, next));
        this._currentExpressionWeights[name] = next;
        expressionManager.setValue(name, next);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Viseme — multi-channel from lip sync analyser
  // -------------------------------------------------------------------------
  _updateViseme(state) {
    const expressionManager = this._vrm.expressionManager;
    if (!expressionManager) return;

    const VISEME_NAMES = ['aa', 'ih', 'ou', 'ee', 'oh'];

    if (state.visemes && state.speaking) {
      // Full viseme set from lip sync — set all channels with clamping
      for (const name of VISEME_NAMES) {
        const val = Math.min(1.0, Math.max(0, state.visemes[name] || 0));
        expressionManager.setValue(name, val);
      }
      this._hasActiveVisemes = true;
    } else if (state.viseme && state.speaking) {
      // Legacy single-viseme path
      if (this._lastViseme && this._lastViseme !== state.viseme) {
        expressionManager.setValue(this._lastViseme, 0);
      }
      this._lastViseme = state.viseme;
      this._visemeWeight = Math.min(1.0, (state.rms || 0.5) * 1.5);
      expressionManager.setValue(state.viseme, this._visemeWeight);
      this._hasActiveVisemes = true;
    } else if (this._hasActiveVisemes || this._lastViseme) {
      // Decay all visemes when not speaking
      let anyActive = false;
      for (const name of VISEME_NAMES) {
        const cur = expressionManager.getValue?.(name) ?? 0;
        if (cur > 0.01) {
          const next = cur * VISEME_DECAY;
          expressionManager.setValue(name, next > 0.01 ? next : 0);
          if (next > 0.01) anyActive = true;
        }
      }
      if (this._lastViseme) {
        this._visemeWeight *= VISEME_DECAY;
        if (this._visemeWeight < 0.01) {
          expressionManager.setValue(this._lastViseme, 0);
          this._lastViseme = null;
          this._visemeWeight = 0;
        } else {
          expressionManager.setValue(this._lastViseme, this._visemeWeight);
          anyActive = true;
        }
      }
      this._hasActiveVisemes = anyActive;
    }
  }

  // -------------------------------------------------------------------------
  // Layer: Gaze — aversion state machine, ballistic saccades, tremor, vergence
  // -------------------------------------------------------------------------
  _updateGaze(dt, state) {
    const cfg = EYE_CONFIG;
    const t = this._clock;

    if (state.speaking) this._conversationState = 'speaking';
    else if (state.processing) this._conversationState = 'thinking';
    else if (state.userSpeaking) this._conversationState = 'listening';
    else this._conversationState = 'idle';

    // --- Gaze aversion state machine ---
    // ``_externalGaze`` (optional {yaw,pitch} in degrees) bypasses the
    // contact/avert state machine and pins the gaze base. Used by the
    // companion widget for cursor-tracking. Saccades + tremor still
    // layer on top so the eyes don't look mechanical. Clear it to null
    // to release back to the auto state machine.
    let baseYaw, basePitch;
    if (this._externalGaze) {
      baseYaw = this._externalGaze.yaw;
      basePitch = this._externalGaze.pitch;
      this._gazeStateTimer = 0;  // start fresh when external releases
      this._gazeState = 'contact';
    } else {
      this._gazeStateTimer += dt;
      if (this._gazeStateTimer >= this._gazeStateDuration) {
        this._gazeStateTimer = 0;
        if (this._gazeState === 'contact') {
          this._gazeState = 'avert';
          this._gazeStateDuration = this._randRange(cfg.avertDurationMin, cfg.avertDurationMax);
          const dir = this._conversationState === 'thinking' ? cfg.avertThinking
            : this._conversationState === 'speaking' ? cfg.avertSpeaking
            : this._conversationState === 'listening' ? cfg.avertRecalling
            : this._conversationState === 'idle' ? cfg.avertIdle
            : cfg.avertProcessing;
          this._avertTarget.yaw = dir[0] + (Math.random() - 0.5) * 3;
          this._avertTarget.pitch = dir[1] + (Math.random() - 0.5) * 2;
        } else {
          this._gazeState = 'contact';
          this._gazeStateDuration = this._randRange(cfg.contactDurationMin, cfg.contactDurationMax);
        }
      }
      baseYaw = this._gazeState === 'contact' ? 0 : this._avertTarget.yaw;
      basePitch = this._gazeState === 'contact' ? 0 : this._avertTarget.pitch;
    }

    const pursueLerp = 1 - Math.exp(-3.0 * dt);
    this._gazeTarget.yaw += (baseYaw - this._gazeTarget.yaw) * pursueLerp;
    this._gazeTarget.pitch += (basePitch - this._gazeTarget.pitch) * pursueLerp;

    // --- Saccades (ballistic via spring) ---
    this._saccadeTimer += dt;
    if (this._saccadeTimer >= this._nextSaccadeAt) {
      const _ll = this._liveliness == null ? 1 : this._liveliness;
      const mag = (0.5 + Math.random() * 1.6) * _ll;
      this._saccadeTarget.yaw = (Math.random() - 0.5) * 2 * mag;
      this._saccadeTarget.pitch = (Math.random() - 0.5) * 2 * mag;
      this._lastSaccadeMag = mag;
      this._saccadeTimer = 0;
      // Throttled → smaller darts and longer gaps, so gaze settles toward a
      // calm hold rather than teleporting around at low fps.
      this._nextSaccadeAt = (1 / cfg.saccadeRate + (Math.random() - 0.5) * 0.2) / Math.max(0.15, _ll);

      if (mag > cfg.gazeEvokedThreshold * 0.5 && Math.random() < cfg.gazeEvokedBlinkChance * (mag / cfg.gazeEvokedThreshold)) {
        this._triggerBlink();
      }
    }

    const sdt = Math.min(dt, 0.1);
    const sd = (4.0 * 0.6931472) / cfg.saccadeSpringHalflife;
    const seydt = Math.exp(-sd * sdt);
    const sjdt = sd * sdt;

    let scY = this._saccadeSpring.current.yaw - this._saccadeTarget.yaw;
    let sj0Y = this._saccadeSpring.velocity.yaw + scY * sd;
    this._saccadeSpring.current.yaw = this._saccadeTarget.yaw + (scY + sj0Y * sdt) * seydt;
    this._saccadeSpring.velocity.yaw = (this._saccadeSpring.velocity.yaw - sj0Y * sjdt) * seydt;

    let scP = this._saccadeSpring.current.pitch - this._saccadeTarget.pitch;
    let sj0P = this._saccadeSpring.velocity.pitch + scP * sd;
    this._saccadeSpring.current.pitch = this._saccadeTarget.pitch + (scP + sj0P * sdt) * seydt;
    this._saccadeSpring.velocity.pitch = (this._saccadeSpring.velocity.pitch - sj0P * sjdt) * seydt;

    this._saccadeTarget.yaw *= 0.95;
    this._saccadeTarget.pitch *= 0.95;

    // --- Microsaccade tremor ---
    const _llT = this._liveliness == null ? 1 : this._liveliness;
    const tremorX = simplex2D(t * cfg.tremorFreq, 0) * cfg.tremorAmpX * _llT;
    const tremorY = simplex2D(0, t * cfg.tremorFreq) * cfg.tremorAmpY * _llT;

    // --- Final eye angles ---
    let eyeYaw = this._gazeTarget.yaw + this._saccadeSpring.current.yaw + tremorX;
    let eyePitch = this._gazeTarget.pitch + this._saccadeSpring.current.pitch + tremorY;
    eyeYaw = Math.max(-cfg.eyeMaxYaw, Math.min(cfg.eyeMaxYaw, eyeYaw));
    eyePitch = Math.max(-cfg.eyeMaxPitch, Math.min(cfg.eyeMaxPitch, eyePitch));

    // --- Apply eye rotation via spring system (not direct .rotation.set) ---
    // VRoid eye bones (J_Adj_*_FaceEye) control face mesh vertices,
    // so direct rotation conflicts with the spring system and deforms the face.
    if (this._springs.leftEye) {
      this._springs.leftEye.addTarget(eyePitch, eyeYaw + cfg.vergenceDeg * 0.5, 0);
    }
    if (this._springs.rightEye) {
      this._springs.rightEye.addTarget(eyePitch, eyeYaw - cfg.vergenceDeg * 0.5, 0);
    }

    // --- Head follow ---
    this._headFollowTimer += dt;
    const headTargetYaw = eyeYaw * cfg.headFollowYaw;
    const headTargetPitch = eyePitch * cfg.headFollowPitch;
    const headLerp = 1 - Math.exp(-2.0 * dt);
    if (this._headFollowTimer > cfg.headFollowDelay) {
      this._headFollow.yaw += (headTargetYaw - this._headFollow.yaw) * headLerp;
      this._headFollow.pitch += (headTargetPitch - this._headFollow.pitch) * headLerp;
    }
    this._headFollow.yaw = Math.max(-cfg.headMaxYaw, Math.min(cfg.headMaxYaw, this._headFollow.yaw));
    this._headFollow.pitch = Math.max(-cfg.headMaxPitch, Math.min(cfg.headMaxPitch, this._headFollow.pitch));

    if (this._springs.head) {
      this._springs.head.addTarget(this._headFollow.pitch, this._headFollow.yaw, 0);
    }
    if (this._springs.neck) {
      this._springs.neck.addTarget(this._headFollow.pitch * 0.3, this._headFollow.yaw * 0.3, 0);
    }
  }

  // -------------------------------------------------------------------------
  // Blink — gaze-evoked + half-blinks + emotion-modulated rate
  // -------------------------------------------------------------------------
  _triggerBlink() {
    if (this._blinkPhase !== 'idle') return;
    const isHalf = Math.random() < EYE_CONFIG.halfBlinkChance;
    this._blinkMaxClose = isHalf ? (0.5 + Math.random() * 0.2) : 1.0;
    this._blinkCloseDuration = EYE_CONFIG.blinkCloseMs / 1000;
    this._blinkOpenDuration = EYE_CONFIG.blinkOpenMs / 1000;
    this._blinkTimer = 0;
    this._blinkPhase = 'closing';
  }

  _updateBlink(dt) {
    const expressionManager = this._vrm.expressionManager;
    if (!expressionManager) return;

    switch (this._blinkPhase) {
      case 'idle':
        this._nextBlinkAt -= dt;
        if (this._nextBlinkAt <= 0) {
          this._triggerBlink();
        }
        break;

      case 'closing': {
        this._blinkTimer += dt;
        const t = Math.min(1, this._blinkTimer / this._blinkCloseDuration);
        const eased = t * t;
        expressionManager.setValue('blink', eased * this._blinkMaxClose);
        if (t >= 1) {
          this._blinkTimer = 0;
          this._blinkPhase = 'opening';
        }
        break;
      }

      case 'opening': {
        this._blinkTimer += dt;
        const t = Math.min(1, this._blinkTimer / this._blinkOpenDuration);
        const eased = 1 - (1 - t) * (1 - t);
        expressionManager.setValue('blink', this._blinkMaxClose * (1 - eased));
        if (t >= 1) {
          expressionManager.setValue('blink', 0);
          this._blinkPhase = 'idle';
          this._nextBlinkAt = this._getNextBlinkInterval();
        }
        break;
      }
    }
  }

  _getNextBlinkInterval() {
    const bpm = EYE_CONFIG.blinkRates[this._emotion] || EYE_CONFIG.blinkRates.neutral;
    const avgInterval = 60 / bpm;
    // Heart-rate multiplier — elevated HR shortens the gap between
    // blinks (sympathetic arousal correlates with higher blink rate).
    // 0.40 = setpoint = identity. Clamped so it never goes silly.
    const hr = this._physiology.heart_rate;
    const hrMul = 1.0 / (1.0 + (hr - 0.40) * 0.9);
    const scaled = avgInterval * Math.max(0.55, Math.min(1.5, hrMul));
    return scaled + (Math.random() - 0.5) * 2 * (scaled * 0.3);
  }

  _randRange(min, max) {
    return min + Math.random() * (max - min);
  }

  // -------------------------------------------------------------------------
  // Layer: Awareness — silence detection, reactions
  // -------------------------------------------------------------------------
  _updateAwareness(dt, state) {
    // --- Silence fidget ---
    if (!state.speaking && !state.userSpeaking) {
      this._silenceTime += dt;
      // After 5s of silence, add subtle fidget
      if (this._silenceTime > 5) {
        const fidgetPhase = (this._silenceTime - 5) * 0.3;
        const fidget = simplex2D(fidgetPhase, 50) * 1.5;
        if (this._springs.head) {
          this._springs.head.addTarget(fidget * 0.5, fidget, 0);
        }
        if (this._springs.spine) {
          this._springs.spine.addTarget(0, fidget * 0.3, 0);
        }
      }
    } else {
      this._silenceTime = 0;
    }

    // --- Impulse reactions ---
    if (this._awarenessImpulse) {
      this._awarenessTimer += dt;
      const t = this._awarenessTimer;
      const fade = Math.max(0, 1 - t / 0.6);  // 0.6s impulse duration

      switch (this._awarenessImpulse) {
        case 'nod':
          if (this._springs.head) {
            this._springs.head.addTarget(Math.sin(t * 8) * 6 * fade, 0, 0);
          }
          break;

        case 'headTilt':
          if (this._springs.head) {
            this._springs.head.addTarget(0, 0, 10 * fade);
          }
          if (this._springs.neck) {
            this._springs.neck.addTarget(0, 0, 5 * fade);
          }
          break;

        case 'surprise':
          if (this._springs.head) {
            this._springs.head.addTarget(-8 * fade, 0, 0);
          }
          if (this._springs.chest) {
            this._springs.chest.addTarget(-4 * fade, 0, 0);
          }
          break;

        case 'thinking':
          if (this._springs.head) {
            this._springs.head.addTarget(5 * fade, 8 * fade, 3 * fade);
          }
          break;

        case 'attention':
          // Lean forward slightly
          if (this._springs.spine) {
            this._springs.spine.addTarget(-3 * fade, 0, 0);
          }
          if (this._springs.head) {
            this._springs.head.addTarget(-4 * fade, 0, 0);
          }
          break;

        case 'user_typing':
          // Slight attention shift — lean forward, look toward "source"
          if (this._springs.spine) {
            this._springs.spine.addTarget(-2 * fade, 0, 0);
          }
          if (this._springs.head) {
            this._springs.head.addTarget(-3 * fade, -5 * fade, 0);
          }
          break;

        case 'silence_short':
          // Subtle settle — slight posture relax
          if (this._springs.head) {
            this._springs.head.addTarget(2 * fade, 0, 0);
          }
          break;

        case 'tool_call':
          // Brief "processing" look — gaze down-right
          if (this._springs.head) {
            this._springs.head.addTarget(4 * fade, -6 * fade, 0);
          }
          break;
      }

      if (t >= 0.6) {
        this._awarenessImpulse = null;
        this._awarenessTimer = 0;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Curated Call Actions - audited upper-body conversational motions
  // -------------------------------------------------------------------------
  _updateCallAction(dt) {
    if (!this._activeCallAction) return;

    const current = this._callActionScheduler?.update?.(this._clock);
    if (!current || current.name !== this._activeCallAction.name) {
      this._clearCallAction();
      return;
    }

    const action = this._activeCallAction.action;
    const duration = Math.max(action.duration || current.duration || 1, 0.001);
    this._activeCallAction.elapsed = Math.max(0, this._clock - current.startedAt);

    const progress = Math.min(1, this._activeCallAction.elapsed / duration);
    const envelope = this._callActionEnvelope(progress);
    const pose = this._sampleCallActionPose(action, progress);
    this._applyCallActionMotionTexture(pose, action, progress, envelope);

    this._callActionFrame = { action, pose, envelope };

    for (const [boneName, angles] of Object.entries(pose)) {
      const spring = this._springs[boneName];
      if (!spring) continue;

      const idleArmPose = (this._idleArmPose || IDLE_ARM_POSE)[boneName];
      if (idleArmPose) {
        spring.addTarget(
          angles[0] - idleArmPose[0],
          angles[1] - idleArmPose[1],
          angles[2] - idleArmPose[2],
        );
      } else {
        spring.addTarget(angles[0], angles[1], angles[2]);
      }
    }
  }

  _sampleCallActionPose(action, progress) {
    const keyframes = action?.keyframes || [];
    if (!keyframes.length) return {};

    let kfA = keyframes[0];
    let kfB = keyframes[keyframes.length - 1];
    for (let i = 0; i < keyframes.length - 1; i += 1) {
      if (progress >= keyframes[i].t && progress <= keyframes[i + 1].t) {
        kfA = keyframes[i];
        kfB = keyframes[i + 1];
        break;
      }
    }

    const range = Math.max(0.0001, kfB.t - kfA.t);
    const localT = Math.min(1, Math.max(0, (progress - kfA.t) / range));
    const eased = this._smooth01(localT);
    const allBones = new Set([
      ...Object.keys(kfA.bones || {}),
      ...Object.keys(kfB.bones || {}),
    ]);
    const pose = {};

    for (const boneName of allBones) {
      const neutral = (this._idleArmPose || IDLE_ARM_POSE)[boneName] || [0, 0, 0];
      const rawA = (kfA.bones || {})[boneName];
      const rawB = (kfB.bones || {})[boneName];
      const a = rawA ? this._mapArmTarget(boneName, rawA) : neutral;
      const b = rawB ? this._mapArmTarget(boneName, rawB) : neutral;

      pose[boneName] = [
        a[0] + (b[0] - a[0]) * eased,
        a[1] + (b[1] - a[1]) * eased,
        a[2] + (b[2] - a[2]) * eased,
      ];
    }

    return pose;
  }

  _applyCallActionMotionTexture(pose, action, progress, envelope) {
    const texture = action?.motionTexture || DEFAULT_CALL_MOTION_TEXTURE;
    const entries = Object.entries(texture?.bones || {});
    if (!entries.length) return;

    const holdStart = texture.holdStart ?? 0.16;
    const holdEnd = texture.holdEnd ?? 0.88;
    const attack = this._smooth01((progress - holdStart) / 0.16);
    const release = this._smooth01((holdEnd - progress) / 0.18);
    const holdWeight = Math.min(attack, release);
    if (holdWeight <= 0) return;

    const frequency = texture.frequency || 0.82;
    const phase = (texture.phase || 0.17) * Math.PI * 2;
    const primary = Math.sin((this._activeCallAction.elapsed * frequency * Math.PI * 2) + phase);
    const secondary = Math.sin((this._activeCallAction.elapsed * frequency * 0.47 * Math.PI * 2) + phase * 1.7);
    const drift = ((primary * 0.68) + (secondary * 0.32)) * envelope * holdWeight * (texture.weight || 1);

    for (const [boneName, delta] of entries) {
      const current = pose[boneName] || ((this._idleArmPose || IDLE_ARM_POSE)[boneName] || [0, 0, 0]);
      const mappedDelta = ARM_TARGET_BONES.has(boneName)
        ? this._mapArmTarget(boneName, delta)
        : delta;
      pose[boneName] = [
        current[0] + ((mappedDelta[0] || 0) * drift),
        current[1] + ((mappedDelta[1] || 0) * drift),
        current[2] + ((mappedDelta[2] || 0) * drift),
      ];
    }
  }

  _applyCallActionDirect() {
    const frame = this._callActionFrame;
    if (!frame?.action) return;

    for (const [boneName, angles] of Object.entries(frame.pose || {})) {
      if (this._springs[boneName]) continue;
      this._applyCallActionBoneTarget(boneName, angles, frame.envelope);
    }

    this._applyCallActionHandShapes(frame.action, frame.envelope);
  }

  _applyCallActionHandShapes(action, envelope) {
    for (const [side, shapeName] of Object.entries(action?.handShapes || {})) {
      const shape = HAND_SHAPES[shapeName];
      if (!shape) continue;
      for (const [suffix, angles] of Object.entries(shape.bones || {})) {
        const boneName = `${side}${suffix}`;
        this._applyCallActionBoneTarget(boneName, this._mapFingerTarget(boneName, angles), envelope);
      }
    }
  }

  _applyCallActionBoneTarget(boneName, angles, envelope) {
    const bone = this._callActionBones?.[boneName];
    const rest = this._callActionRestQuats?.[boneName];
    if (!bone || !rest || !this._callActionEuler || !this._callActionQuat) return;

    this._callActionEuler.set(angles[0] * DEG, angles[1] * DEG, angles[2] * DEG, 'XYZ');
    this._callActionQuat.setFromEuler(this._callActionEuler);
    const target = rest.clone().multiply(this._callActionQuat);
    bone.quaternion.slerp(target, envelope);
  }

  _applyCallActionExpressions() {
    const frame = this._callActionFrame;
    const expressionManager = this._vrm.expressionManager;
    if (!frame?.action || !expressionManager) return;

    for (const [name, value] of Object.entries(frame.action.expression || {})) {
      const target = Math.max(0, Math.min(1, value * frame.envelope));
      expressionManager.setValue?.(name, target);
    }
  }

  _callActionEnvelope(progress) {
    const attack = Math.min(1, progress / 0.18);
    const release = Math.min(1, (1 - progress) / 0.2);
    return this._smooth01(Math.min(attack, release));
  }

  _smooth01(value) {
    const t = Math.min(1, Math.max(0, value));
    return t * t * (3 - 2 * t);
  }

  _mapFingerTarget(boneName, angles) {
    if (!FINGER_TARGET_BONES.has(boneName)) return angles;
    const sideSign = boneName.startsWith('right') ? -1 : 1;
    return [
      angles[0],
      angles[1] * sideSign,
      angles[2] * sideSign,
    ];
  }

  // -------------------------------------------------------------------------
  // Procedural Gestures - one-shot keyframed sequences
  // -------------------------------------------------------------------------
  _updateGestures(dt) {
    if (!this._activeGesture) return;

    const g = this._activeGesture;
    g.elapsed += dt;

    const progress = Math.min(1, g.elapsed / g.duration);

    // Find the two keyframes we're between
    const kfs = g.keyframes;
    let kfA = kfs[0], kfB = kfs[kfs.length - 1];
    for (let i = 0; i < kfs.length - 1; i++) {
      if (progress >= kfs[i].t && progress <= kfs[i + 1].t) {
        kfA = kfs[i];
        kfB = kfs[i + 1];
        break;
      }
    }

    // Interpolation factor between kfA and kfB
    const range = kfB.t - kfA.t;
    const localT = range > 0 ? (progress - kfA.t) / range : 1;

    // Smooth ease (cubic)
    const eased = localT * localT * (3 - 2 * localT);

    // Collect all bones mentioned in either keyframe
    const allBones = new Set([...Object.keys(kfA.bones || {}), ...Object.keys(kfB.bones || {})]);

    for (const boneName of allBones) {
      const idleArmPose = (this._idleArmPose || IDLE_ARM_POSE)[boneName];
      const neutral = idleArmPose || [0, 0, 0];
      const rawA = (kfA.bones || {})[boneName];
      const rawB = (kfB.bones || {})[boneName];
      const a = rawA ? this._mapArmTarget(boneName, rawA) : neutral;
      const b = rawB ? this._mapArmTarget(boneName, rawB) : neutral;

      // Lerp between keyframe values
      const rx = a[0] + (b[0] - a[0]) * eased;
      const ry = a[1] + (b[1] - a[1]) * eased;
      const rz = a[2] + (b[2] - a[2]) * eased;

      // Arm gesture values are absolute targets relative to the VRM rest pose;
      // subtract the neutral arm pose because it has already been applied.
      if (this._springs[boneName]) {
        if (idleArmPose) {
          this._springs[boneName].addTarget(
            rx - idleArmPose[0],
            ry - idleArmPose[1],
            rz - idleArmPose[2],
          );
        } else {
          this._springs[boneName].addTarget(rx, ry, rz);
        }
      }
    }

    // End gesture when complete
    if (progress >= 1) {
      this._activeGesture = null;
    }
  }
}
