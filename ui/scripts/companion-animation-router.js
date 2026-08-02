/**
 * companion-animation-router.js
 *
 * Thin adapter from companion-surface events into the existing avatar
 * animation stack. It does not own animation selection; MovementConductor
 * and PresenceEngine keep doing that work.
 */

import { PoseTriggerEngine } from './avatar-pose-trigger.js';
import { POSE_FAMILIES } from './avatar-pose-presets.js';
import { bus } from './activity-bus.js';

const STATE_MAP = {
  recording: 'recording',
  processing: 'processing',
  speaking: 'speaking',
  armed: 'listening',
  idle: 'listening',
  error: 'listening',
};

const EMOTION_KEYS = ['warmth', 'energy', 'openness', 'focus'];
const DEFAULT_SITUATION_COOLDOWN_MS = 6000;
const DEFAULT_POSE_INTENT_MS = 9000;
const MAX_POSE_INTENT_MS = 45000;

const POSE_PRIORITY = {
  idle: 0,
  audio: 10,
  situational: 20,
  companion: 20,
  boundary: 30,
  safety: 30,
  conversation: 40,
  explicit: 50,
};

const POSE_VERBS = {
  listen: { family: 'idle_engaged', durationMs: 10000 },
  listening: { family: 'idle_engaged', durationMs: 10000 },
  attentive: { family: 'idle_engaged', durationMs: 9000 },
  engage: { family: 'idle_engaged', durationMs: 9000 },
  curious: { family: 'idle_engaged', durationMs: 10000 },
  reach_out: { family: 'idle_engaged', durationMs: 12000 },

  think: { family: 'thinking', durationMs: 9000 },
  thinking: { family: 'thinking', durationMs: 9000 },
  reflect: { family: 'thinking', durationMs: 11000 },
  create: { family: 'thinking', durationMs: 11000 },
  concerned: { family: 'thinking', durationMs: 9000 },

  speak: { family: 'talking', durationMs: 6500 },
  speaking: { family: 'talking', durationMs: 6500 },

  present: { family: 'idle_grounded', durationMs: 10000 },
  settle: { family: 'idle_holding', durationMs: 8000 },
  settled: { family: 'idle_holding', durationMs: 10000 },
  host: { family: 'idle_grounded', durationMs: 9000 },

  formal: { family: 'formal', durationMs: 12000 },
  wind_down: { family: 'idle_holding', durationMs: 14000 },
  dormant: { family: 'idle_holding', durationMs: 16000 },
  asleep: { family: 'idle_holding', durationMs: 16000 },

  handoff: { family: 'formal_behind', durationMs: 18000 },
  step_aside: { family: 'formal_behind', durationMs: 18000 },

  closed: { family: 'closed', durationMs: 9000 },
  boundary: { family: 'closed', durationMs: 9000 },
  unsure: { family: 'closed', durationMs: 8000 },
  frustrated: { family: 'closed', durationMs: 8000 },
  melancholy: { family: 'closed', durationMs: 9000 },

  surface_attention: { family: 'idle_lefthip', durationMs: 11000 },
  media_attention: { family: 'idle_lefthip', durationMs: 11000 },
  world_attention: { family: 'idle_lefthip', durationMs: 11000 },
  confident: { family: 'idle_grounded', durationMs: 12000 },
};

const TOPIC_INTENTS = {
  'initiative.surfaced': {
    roles: ['attention-seek', 'reach-out'],
    emotion: { warmth: 0.72, energy: 0.48, openness: 0.75, focus: 0.65 },
    pose: { verb: 'reach_out' },
    cooldownMs: 12000,
  },
  'behavior.reach_out': {
    roles: ['attention-seek', 'reach-out'],
    emotion: { warmth: 0.75, energy: 0.5, openness: 0.8, focus: 0.65 },
    pose: { verb: 'reach_out' },
    cooldownMs: 12000,
  },
  'voice.tool_call': {
    roles: ['think', 'ponder'],
    emotion: { warmth: 0.5, energy: 0.35, openness: 0.45, focus: 0.9 },
    pose: { verb: 'thinking' },
    cooldownMs: 5000,
  },
  'voice.tool_result': {
    roles: ['realization', 'react-positive', 'oh-i-see'],
    emotion: { warmth: 0.62, energy: 0.5, openness: 0.65, focus: 0.75 },
    pose: { verb: 'attentive' },
    cooldownMs: 7000,
  },
  'voice.completed': {
    roles: ['gratitude', 'agree', 'soften'],
    emotion: { warmth: 0.75, energy: 0.42, openness: 0.72, focus: 0.55 },
    pose: { verb: 'settle' },
    cooldownMs: 7000,
  },
  'personality.labeled': {
    roles: ['gratitude', 'agree', 'soften'],
    emotion: { warmth: 0.78, energy: 0.38, openness: 0.72, focus: 0.55 },
    pose: { verb: 'attentive' },
    cooldownMs: 9000,
  },
  'channel.entering': {
    roles: ['posture-shift', 'wind-down'],
    emotion: { warmth: 0.55, energy: 0.25, openness: 0.45, focus: 0.6 },
    pose: { verb: 'handoff' },
    cooldownMs: 12000,
  },
  'channel.user_idle': {
    roles: ['attention-seek', 'reach-out'],
    emotion: { warmth: 0.68, energy: 0.45, openness: 0.7, focus: 0.65 },
    pose: { verb: 'reach_out' },
    cooldownMs: 30000,
  },
  'channel.exiting': {
    roles: ['idle-shift', 'idle-attentive'],
    emotion: { warmth: 0.62, energy: 0.38, openness: 0.65, focus: 0.7 },
    pose: { verb: 'attentive' },
    cooldownMs: 10000,
  },
  'channel.exited': {
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.78, energy: 0.48, openness: 0.75, focus: 0.55 },
    pose: { verb: 'attentive' },
    cooldownMs: 18000,
  },
};

const ACTIVITY_INTENTS = {
  journal: {
    roles: ['think', 'ponder'],
    emotion: { warmth: 0.55, energy: 0.3, openness: 0.5, focus: 0.88 },
    pose: { verb: 'reflect' },
  },
  revisit: {
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.58, energy: 0.45, openness: 0.72, focus: 0.78 },
    pose: { verb: 'curious' },
  },
  revisit_thread: {
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.58, energy: 0.45, openness: 0.72, focus: 0.78 },
    pose: { verb: 'curious' },
  },
  creation: {
    roles: ['think', 'ponder'],
    emotion: { warmth: 0.6, energy: 0.36, openness: 0.55, focus: 0.85 },
    pose: { verb: 'create' },
  },
  observation: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.58, energy: 0.38, openness: 0.62, focus: 0.82 },
    pose: { verb: 'surface_attention' },
  },
  reach_out: {
    roles: ['attention-seek', 'reach-out'],
    emotion: { warmth: 0.75, energy: 0.5, openness: 0.78, focus: 0.65 },
    pose: { verb: 'reach_out' },
  },
  scene_update: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.55, energy: 0.36, openness: 0.58, focus: 0.78 },
    pose: { verb: 'surface_attention' },
  },
  dream: {
    roles: ['posture-shift', 'wind-down'],
    emotion: { warmth: 0.52, energy: 0.22, openness: 0.45, focus: 0.45 },
    pose: { verb: 'wind_down' },
  },
};

const AFFECT_INTENTS = {
  curious: {
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.58, energy: 0.45, openness: 0.72, focus: 0.78 },
    pose: { verb: 'curious' },
  },
  alert: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.5, energy: 0.52, openness: 0.62, focus: 0.85 },
    pose: { verb: 'attentive' },
  },
  warm: {
    roles: ['affection', 'warmth', 'react-positive'],
    emotion: { warmth: 0.88, energy: 0.35, openness: 0.78, focus: 0.45 },
    pose: { verb: 'attentive' },
  },
  delighted: {
    roles: ['celebrate', 'joy', 'react-positive'],
    emotion: { warmth: 0.86, energy: 0.75, openness: 0.82, focus: 0.55 },
    pose: { verb: 'confident' },
  },
  tender: {
    roles: ['affection', 'warmth', 'react-positive'],
    emotion: { warmth: 0.9, energy: 0.28, openness: 0.76, focus: 0.42 },
    pose: { verb: 'attentive' },
  },
  patient: {
    roles: ['idle-shift', 'idle-attentive'],
    emotion: { warmth: 0.62, energy: 0.25, openness: 0.55, focus: 0.65 },
    pose: { verb: 'settled' },
  },
  settled: {
    roles: ['idle-shift', 'idle-attentive'],
    emotion: { warmth: 0.6, energy: 0.22, openness: 0.55, focus: 0.55 },
    pose: { verb: 'settled' },
  },
  weary: {
    roles: ['posture-shift', 'wind-down'],
    emotion: { warmth: 0.48, energy: 0.18, openness: 0.4, focus: 0.45 },
    pose: { verb: 'wind_down' },
  },
  melancholy: {
    roles: ['sympathy', 'mirror-sad', 'soften'],
    emotion: { warmth: 0.5, energy: 0.22, openness: 0.42, focus: 0.45 },
    pose: { verb: 'melancholy' },
  },
  unsure: {
    roles: ['nervousness', 'mirror-nervous', 'hesitant'],
    emotion: { warmth: 0.48, energy: 0.38, openness: 0.42, focus: 0.72 },
    pose: { verb: 'unsure' },
  },
  concerned: {
    roles: ['comfort', 'sympathy', 'soften'],
    emotion: { warmth: 0.65, energy: 0.32, openness: 0.55, focus: 0.7 },
    pose: { verb: 'concerned' },
  },
  frustrated: {
    roles: ['react-negative', 'angry', 'mirror-anger'],
    emotion: { warmth: 0.32, energy: 0.68, openness: 0.3, focus: 0.78 },
    pose: { verb: 'frustrated' },
  },
};

const STATE_INTENTS = {
  present: {
    roles: ['idle-shift', 'idle-attentive'],
    emotion: { warmth: 0.58, energy: 0.38, openness: 0.62, focus: 0.68 },
    pose: { verb: 'present' },
  },
  dormant: {
    roles: ['posture-shift', 'wind-down'],
    emotion: { warmth: 0.48, energy: 0.18, openness: 0.38, focus: 0.42 },
    pose: { verb: 'dormant' },
  },
  asleep: {
    roles: ['posture-shift', 'wind-down'],
    emotion: { warmth: 0.45, energy: 0.12, openness: 0.32, focus: 0.3 },
    pose: { verb: 'asleep' },
  },
};

const FOCUS_INTENTS = {
  user: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.62, energy: 0.4, openness: 0.68, focus: 0.85 },
    pose: { verb: 'listening' },
  },
  conversation: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.62, energy: 0.4, openness: 0.68, focus: 0.85 },
    pose: { verb: 'listening' },
  },
  self: {
    roles: ['think', 'ponder'],
    emotion: { warmth: 0.52, energy: 0.28, openness: 0.5, focus: 0.82 },
    pose: { verb: 'thinking' },
  },
  world: {
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.56, energy: 0.42, openness: 0.7, focus: 0.76 },
    pose: { verb: 'world_attention' },
  },
  media: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.55, energy: 0.35, openness: 0.6, focus: 0.74 },
    pose: { verb: 'media_attention' },
  },
};

const AUDIO_KIND_PRIORITY = ['speech', 'music', 'narration', 'dialogue', 'mixed', 'ambient', 'unknown'];

const AUDIO_KIND_INTENTS = {
  speech: {
    emotion: { warmth: 0.65, energy: 0.48, openness: 0.68, focus: 0.78 },
    pose: { verb: 'speaking', durationMs: 0 },
    cooldownMs: 0,
  },
  music: {
    emotion: { warmth: 0.72, energy: 0.55, openness: 0.72, focus: 0.45 },
    pose: { verb: 'host', durationMs: 0 },
    cooldownMs: 0,
  },
  narration: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.58, energy: 0.28, openness: 0.58, focus: 0.78 },
    pose: { verb: 'media_attention', durationMs: 0 },
    cooldownMs: 12000,
  },
  dialogue: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.56, energy: 0.34, openness: 0.58, focus: 0.76 },
    pose: { verb: 'media_attention', durationMs: 0 },
    cooldownMs: 12000,
  },
  mixed: {
    roles: ['listen', 'attentive', 'idle-shift'],
    emotion: { warmth: 0.56, energy: 0.32, openness: 0.58, focus: 0.72 },
    pose: { verb: 'media_attention', durationMs: 0 },
    cooldownMs: 12000,
  },
  ambient: {
    roles: ['idle-shift', 'idle-relaxed'],
    emotion: { warmth: 0.5, energy: 0.2, openness: 0.48, focus: 0.42 },
    pose: { verb: 'settled', durationMs: 0 },
    cooldownMs: 18000,
  },
  unknown: {
    emotion: { warmth: 0.52, energy: 0.28, openness: 0.5, focus: 0.58 },
    pose: { verb: 'media_attention', durationMs: 0 },
    cooldownMs: 12000,
  },
};

function _clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.min(1, n));
}

function _stringList(value, cap = 6) {
  if (!Array.isArray(value)) return [];
  return value
    .map(v => String(v || '').trim())
    .filter(Boolean)
    .slice(0, cap);
}

function _audioKindList(value) {
  return _stringList(value, 10)
    .map(v => v.toLowerCase())
    .filter(v => v && v !== 'sfx');
}

function _dominantAudioKind(kinds) {
  const set = new Set(_audioKindList(kinds));
  for (const kind of AUDIO_KIND_PRIORITY) {
    if (set.has(kind)) return kind;
  }
  return '';
}

function _emotion(value) {
  if (!value || typeof value !== 'object') return null;
  const out = {};
  for (const key of EMOTION_KEYS) {
    const v = _clamp01(value[key]);
    if (v !== null) out[key] = v;
  }
  return Object.keys(out).length ? out : null;
}

function _poseVerbKey(value) {
  return String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
}

function _knownPoseFamily(value) {
  const family = String(value || '').trim();
  if (!family) return '';
  return Object.prototype.hasOwnProperty.call(POSE_FAMILIES, family) ? family : '';
}

function _poseDurationMs(value, fallback = DEFAULT_POSE_INTENT_MS) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(MAX_POSE_INTENT_MS, Math.round(n)));
}

function _priorityRank(value) {
  const key = String(value || 'situational').trim().toLowerCase();
  const rank = POSE_PRIORITY[key];
  return Number.isFinite(rank) ? rank : POSE_PRIORITY.situational;
}

function _poseIntentPriority(intent) {
  const rank = Number(intent?.priorityRank);
  return Number.isFinite(rank) ? rank : _priorityRank(intent?.priority);
}

function _poseIntentActive(intent, now) {
  if (!intent?.family) return false;
  return !intent.expiresAt || now < intent.expiresAt;
}

export function normalizeCompanionPoseIntent(raw, fallback = {}) {
  const payload = raw?.payload && typeof raw.payload === 'object'
    ? raw.payload
    : raw;
  if (!payload || typeof payload !== 'object') return null;

  const nested = payload.pose && typeof payload.pose === 'object' ? payload.pose : {};
  const verb = _poseVerbKey(
    nested.verb
    || nested.poseVerb
    || nested.pose_verb
    || payload.poseVerb
    || payload.pose_verb
    || payload.verb
    || fallback.verb
  );
  const verbDefaults = POSE_VERBS[verb] || null;
  const family = _knownPoseFamily(
    nested.family
    || nested.poseFamily
    || nested.pose_family
    || payload.family
    || payload.poseFamily
    || payload.pose_family
    || fallback.family
    || verbDefaults?.family
  );
  if (!family) return null;

  const durationMs = _poseDurationMs(
    nested.durationMs
    ?? nested.duration_ms
    ?? payload.durationMs
    ?? payload.duration_ms
    ?? payload.poseDurationMs
    ?? payload.pose_duration_ms
    ?? fallback.durationMs
    ?? verbDefaults?.durationMs
  );

  return {
    family,
    verb: verb || fallback.verb || '',
    durationMs,
    source: String(payload.source || fallback.source || raw?.topic || 'companion').slice(0, 64),
    priority: String(
      nested.priority
      || payload.priority
      || payload.posePriority
      || fallback.priority
      || (payload.explicit === true ? 'explicit' : 'situational'),
    ).slice(0, 32),
  };
}

export function normalizeCompanionAnimationIntent(raw) {
  const payload = raw?.payload && typeof raw.payload === 'object'
    ? raw.payload
    : raw;
  if (!payload || typeof payload !== 'object') return null;

  const id = String(payload.id || payload.anim_id || payload.animation_id || '').trim();
  const roles = _stringList(payload.roles);
  const emotion = _emotion(payload.emotion || payload.mood);
  const pose = normalizeCompanionPoseIntent(raw);
  if (!id && !roles.length && !pose) return null;

  return {
    id: id || '',
    roles,
    emotion,
    pose,
    source: String(payload.source || raw?.topic || 'companion').slice(0, 64),
    priority: String(payload.priority || (payload.explicit === true ? 'explicit' : 'companion')).slice(0, 32),
    explicit: payload.explicit === true || payload.priority === 'explicit',
  };
}

// Model-emitted motion cues (e.g. [motion:happy] in a chat reply) → animation
// roles + a light target "feel". The conductor's select() then picks an actual
// clip BY ROLE from the user's curated pool, so their ratings / disables /
// uploaded clips govern what plays (an all-disliked category → graceful no-op).
// Keep these keys in sync with the _MOTION_CUE_DIRECTIVE vocabulary in
// augmentum/modes/becca_direct/handler.py.
const MOTION_CUE_INTENT = {
  happy:   { roles: ['joy', 'celebrate', 'react-positive'], emotion: { warmth: 0.85, energy: 0.7, openness: 0.75, focus: 0.55 } },
  excited: { roles: ['excitement-peak', 'emphasize', 'celebrate'], emotion: { warmth: 0.8, energy: 0.9, openness: 0.75, focus: 0.6 } },
  dancing: { roles: ['dance', 'show-off'], emotion: { warmth: 0.8, energy: 0.85, openness: 0.8, focus: 0.5 } },
  bow:     { roles: ['bow-deep', 'gratitude'], emotion: { warmth: 0.7, energy: 0.4, openness: 0.6, focus: 0.6 } },
  wave:    { roles: ['greet', 'wave'], emotion: { warmth: 0.85, energy: 0.6, openness: 0.75, focus: 0.55 } },
  shrug:   { roles: ['unsure', 'hesitant'], emotion: { warmth: 0.55, energy: 0.4, openness: 0.5, focus: 0.5 } },
  think:   { roles: ['think', 'ponder'], emotion: { warmth: 0.5, energy: 0.35, openness: 0.45, focus: 0.9 } },
  curious: { roles: ['curiosity', 'curious', 'question'], emotion: { warmth: 0.6, energy: 0.5, openness: 0.7, focus: 0.75 } },
  sad:     { roles: ['sad', 'soften', 'melancholy'], emotion: { warmth: 0.55, energy: 0.3, openness: 0.5, focus: 0.5 } },
  laugh:   { roles: ['amusement', 'micro-laugh', 'playful'], emotion: { warmth: 0.85, energy: 0.7, openness: 0.8, focus: 0.5 } },
  nod:     { roles: ['agree', 'affirm'], emotion: { warmth: 0.7, energy: 0.5, openness: 0.65, focus: 0.7 } },
  tender:  { roles: ['affection', 'comfort', 'tender'], emotion: { warmth: 0.9, energy: 0.4, openness: 0.7, focus: 0.6 } },
  shy:     { roles: ['shy', 'embarrassment'], emotion: { warmth: 0.65, energy: 0.4, openness: 0.4, focus: 0.55 } },
  proud:   { roles: ['pride'], emotion: { warmth: 0.75, energy: 0.6, openness: 0.7, focus: 0.65 } },
};

export class CompanionAnimationRouter {
  constructor(options = {}) {
    if (!options.conductor) {
      throw new Error('CompanionAnimationRouter requires opts.conductor');
    }
    this.conductor = options.conductor;
    this.avatarState = options.avatarState || null;
    this.hooks = options.hooks || {};
    this.logger = options.logger || console;
    this.now = options.now || (() => Date.now());
    this._disposed = false;
    this._lastPttState = 'idle';
    this._lastFinalTranscript = '';
    this._lastFinalTranscriptAt = 0;
    this._lastSituationAt = new Map();
    this._lastAudioKey = '';
    this._lastPoseIntentRejected = false;

    this.pose = new PoseTriggerEngine({
      conductor: this.conductor,
      character: this.avatarState?.avatarProfile || {},
    });
  }

  dispose() {
    this._disposed = true;
    this._releasePoseIntent();
    try { this.pose?.dispose(); } catch (err) { this._warn('pose dispose failed', err); }
  }

  onPttStateChange(state) {
    if (this._disposed) return;
    const mapped = STATE_MAP[state];
    if (mapped) this._call('onStateChange', mapped);

    if (state === 'recording') {
      this._applyPoseIntent({ verb: 'listening' }, { source: 'ptt:recording', priority: 'conversation' });
      this._markUserTurnStarted();
      try { this.pose.onUserStartedSpeaking(); } catch (err) { this._warn('pose user-start failed', err); }
    } else if (state === 'processing') {
      this._applyPoseIntent({ verb: 'thinking' }, { source: 'ptt:processing', priority: 'conversation' });
      try { this.pose.onUserStoppedSpeaking?.(); } catch (err) { this._warn('pose user-stop failed', err); }
    } else if (state === 'speaking') {
      this._applyPoseIntent({ verb: 'speaking' }, { source: 'ptt:speaking', priority: 'conversation' });
    } else if (state === 'idle' || state === 'error') {
      this._releasePoseIntent({ priority: 'conversation' });
    }
    this._lastPttState = state || this._lastPttState;
  }

  onTranscript(text, options = {}) {
    if (this._disposed) return;
    const clean = String(text || '').trim();
    if (!clean) return;
    const final = options.final === true;
    this._call('onUserTranscript', clean, final);
    if (final) {
      const key = clean.toLowerCase();
      const now = this.now();
      if (key === this._lastFinalTranscript && now - this._lastFinalTranscriptAt < 1500) {
        return;
      }
      this._lastFinalTranscript = key;
      this._lastFinalTranscriptAt = now;
      try { this.pose.onUserTranscriptFinal(clean); } catch (err) { this._warn('pose transcript failed', err); }
    }
  }

  onLLMDelta(text) {
    if (this._disposed) return;
    if (!text) return;
    this._call('onLLMDelta', String(text));
  }

  onTtsStart(sentence = '') {
    if (this._disposed) return;
    const clean = String(sentence || '').trim();
    if (!clean) return;
    this._applyPoseIntent({ verb: 'speaking' }, { source: 'tts:start', priority: 'conversation' });
    try { this.pose.onResponseStarted(clean); } catch (err) { this._warn('pose response-start failed', err); }
  }

  onTtsEnd() {
    if (this._disposed) return;
    this._applyPoseIntent({ verb: 'settle', durationMs: 3500 }, { source: 'tts:end', priority: 'conversation' });
    try { this.pose.onResponseEndedSpeaking(); } catch (err) { this._warn('pose response-end failed', err); }
  }

  /**
   * Play an animation from a model-emitted motion cue (e.g. [motion:happy]).
   * Maps the cue word → roles and lets the conductor's select() pick an actual
   * clip from the user's curated pool — so their ratings / disables / uploads
   * govern what plays. Explicit, so it bypasses quietMode (the model asked for
   * it); an unknown or fully-filtered-out cue is a graceful no-op.
   */
  onMotionCue(cue) {
    if (this._disposed || !cue) return;
    const intent = MOTION_CUE_INTENT[String(cue).toLowerCase()];
    if (!intent) return;
    try {
      this.conductor.play(
        { roles: intent.roles, emotion: intent.emotion },
        { explicit: true, source: 'chat:motion_cue' },
      );
    } catch (err) { this._warn('motion cue play failed', err); }
  }

  onRuntimeBusEvent(msg) {
    if (this._disposed) return null;
    if (msg?.topic === 'behavior.animation_intent') return this.dispatchAnimationIntent(msg);
    const situational = this._intentForSituation(msg);
    if (!situational) return null;
    return this._dispatchSituationalIntent(situational);
  }

  onAudioBusState(detail = {}, options = {}) {
    if (this._disposed) return null;
    const kind = _dominantAudioKind(detail.activeKinds || []);
    if (!kind) {
      this._releaseAudioPoseIntent();
      this._lastAudioKey = '';
      return null;
    }

    const audioRole = String(options.audioRole || detail.audioRole || '').trim().toLowerCase();
    const key = `audio:${kind}:${audioRole || 'any'}`;
    if (key === this._lastAudioKey) return null;

    const entry = AUDIO_KIND_INTENTS[kind] || AUDIO_KIND_INTENTS.unknown;
    const poseResult = this._applyPoseIntent(entry.pose, {
      source: key,
      priority: 'audio',
    });
    if (this._lastPoseIntentRejected) return null;
    this._lastAudioKey = key;

    // Music's full-body motion remains owned by the existing host-mode
    // dance loop. Speech remains owned by lipsync. The router still pins
    // the posture so the procedural body is coherent around those systems.
    if (kind === 'music' || kind === 'speech') return poseResult;

    const routed = this._dispatchSituationalIntent({
      ...entry,
      key,
      pose: null,
    });
    return routed || poseResult;
  }

  async dispatchAnimationIntent(raw) {
    if (this._disposed) return null;
    const intent = normalizeCompanionAnimationIntent(raw);
    if (!intent) return null;
    if (!intent.explicit && this._shouldSuppressPassiveBodyAction()) return null;
    const pose = intent.pose && intent.explicit
      ? { ...intent.pose, priority: 'explicit' }
      : intent.pose;
    const poseResult = this._applyPoseIntent(pose, {
      source: intent.source,
      priority: intent.explicit ? 'explicit' : intent.priority,
    });
    if (this._lastPoseIntentRejected) return null;
    if (intent.id) {
      return this.conductor.playById(intent.id, { explicit: intent.explicit });
    }
    if (!intent.roles.length) return poseResult;
    return this.conductor.play(
      { roles: intent.roles, emotion: intent.emotion || undefined },
      { explicit: intent.explicit },
    );
  }

  _intentForSituation(msg) {
    const topic = String(msg?.topic || '');
    const payload = msg?.payload && typeof msg.payload === 'object' ? msg.payload : {};

    if (topic === 'behavior.activity_chosen' || topic === 'behavior.creation_made') {
      const kind = String(payload.kind || '').toLowerCase();
      const mapped = ACTIVITY_INTENTS[kind];
      return mapped ? { ...mapped, key: `${topic}:${kind || 'unknown'}` } : null;
    }

    if (topic === 'affect.changed') {
      const tag = String(payload.tag || '').toLowerCase();
      const mapped = AFFECT_INTENTS[tag];
      return mapped ? { ...mapped, key: `${topic}:${tag}` } : null;
    }

    if (topic === 'state.transition' || topic === 'role.transition') {
      const to = String(payload.to || '').toLowerCase();
      const mapped = STATE_INTENTS[to];
      return mapped ? { ...mapped, key: `${topic}:${to}` } : null;
    }

    if (topic === 'focus.transition') {
      const to = String(payload.to || '').toLowerCase();
      const mapped = FOCUS_INTENTS[to];
      return mapped ? { ...mapped, key: `${topic}:${to}` } : null;
    }

    const mapped = TOPIC_INTENTS[topic];
    return mapped ? { ...mapped, key: topic } : null;
  }

  _dispatchSituationalIntent(entry) {
    if (this._shouldSuppressPassiveBodyAction()) return null;
    const key = entry.key || entry.roles?.join('|') || 'situation';
    const now = this.now();
    const cooldownMs = entry.cooldownMs ?? DEFAULT_SITUATION_COOLDOWN_MS;
    const hasLast = this._lastSituationAt.has(key);
    const lastAt = hasLast ? this._lastSituationAt.get(key) : 0;
    if (hasLast && now - lastAt < cooldownMs) return null;
    const poseResult = this._applyPoseIntent(entry.pose, {
      source: key,
      priority: 'situational',
    });
    if (this._lastPoseIntentRejected) return null;
    this._lastSituationAt.set(key, now);
    if (!entry.roles?.length) return poseResult;
    return this.conductor.play({
      roles: entry.roles,
      emotion: entry.emotion,
    });
  }

  _applyPoseIntent(poseLike, fallback = {}) {
    this._lastPoseIntentRejected = false;
    if (!poseLike) return null;
    const pose = normalizeCompanionPoseIntent(poseLike, fallback);
    if (!pose) return null;
    if (!this.avatarState) return null;

    const now = this.now();
    const current = this.avatarState._companionPoseIntent;
    if (current && !_poseIntentActive(current, now)) {
      this.avatarState._companionPoseIntent = null;
    } else if (current && _priorityRank(pose.priority) < _poseIntentPriority(current)) {
      this._lastPoseIntentRejected = true;
      return null;
    }

    const priorityRank = _priorityRank(pose.priority);
    const expiresAt = pose.durationMs > 0 ? now + pose.durationMs : 0;
    this.avatarState._companionPoseIntent = {
      family: pose.family,
      verb: pose.verb,
      source: pose.source,
      priority: pose.priority,
      priorityRank,
      setAt: now,
      expiresAt,
    };

    const animator = this.avatarState.animator;
    if (animator?.setPoseIntent && this.avatarState._poseIntent !== pose.family) {
      this.avatarState._poseIntent = pose.family;
      try { animator.setPoseIntent(pose.family); } catch (err) { this._warn('set pose intent failed', err); }
    }

    // Host hook so the presence widget can surface posture shifts in the
    // status row even when quiet mode kills the VRMA. Pre-2026-06-10 the
    // pose was applied silently and the user had no signal that a
    // backend topic (voice.tool_call → thinking, voice.completed → settle,
    // etc.) actually reached her. Now we fire onPoseShift with the verb
    // (the human-readable label) and durationMs so the host can show a
    // transient label like "thinking" / "settling" while the pose is
    // active. Best-effort — host may not provide the hook.
    this._call('onPoseShift', {
      verb: pose.verb || '',
      family: pose.family,
      durationMs: pose.durationMs || 0,
      source: pose.source || '',
      priority: pose.priority || '',
    });

    return this.avatarState._companionPoseIntent;
  }

  _releasePoseIntent(filter = {}) {
    const current = this.avatarState?._companionPoseIntent;
    if (!current) return;
    if (filter.priority && _poseIntentPriority(current) !== _priorityRank(filter.priority)) return;
    if (filter.sourcePrefix && !String(current.source || '').startsWith(filter.sourcePrefix)) return;
    this.avatarState._companionPoseIntent = null;
  }

  _releaseAudioPoseIntent() {
    const cur = this.avatarState?._companionPoseIntent;
    if (!cur || !String(cur.source || '').startsWith('audio:')) return;
    this._releasePoseIntent({ sourcePrefix: 'audio:' });
  }

  _shouldSuppressPassiveBodyAction() {
    if (this._lastPttState === 'speaking') return true;
    if (bus.state.voice_state === 'speaking') return true;
    if (this.avatarState?.active && this.avatarState._standalone === false) return true;
    return false;
  }

  _markUserTurnStarted() {
    const now = this.now();
    this.pose.callStartedAt = now;
    this.pose.lastUserActivityAt = now;
    this.pose.lastIdleEscalationAt = now;
    this.pose.callPhase = 'opening';
  }

  _call(name, ...args) {
    const fn = this.hooks?.[name];
    if (typeof fn !== 'function') return;
    try { fn(...args); } catch (err) { this._warn(`${name} hook failed`, err); }
  }

  _warn(message, err) {
    try { this.logger?.warn?.(`[companion-animation] ${message}`, err); } catch (_) {}
  }
}

export const __test = {
  AFFECT_INTENTS,
  ACTIVITY_INTENTS,
  AUDIO_KIND_INTENTS,
  _dominantAudioKind,
  _poseIntentActive,
  _poseIntentPriority,
  _priorityRank,
  FOCUS_INTENTS,
  POSE_VERBS,
  STATE_MAP,
  STATE_INTENTS,
  TOPIC_INTENTS,
  normalizeCompanionAnimationIntent,
  normalizeCompanionPoseIntent,
};
