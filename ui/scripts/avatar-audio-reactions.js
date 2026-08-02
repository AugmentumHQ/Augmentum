/**
 * avatar-audio-reactions.js — procedural WebAudio reaction cues layered
 * on top of ContactReactor events.
 *
 * The contact reactor (`contact-reactor.js`) fires region-tagged events
 * (cheek_L, chest_R, hand_L, hip_R, ...) when a user's hand touches the
 * AI VRM. `avatar-xr-contact.js` already drives expression spikes from
 * those events; this module adds tiny vocal/breath reactions on the
 * same channel to deepen embodied presence — soft "oh!", playful giggle,
 * sharp inhale "ah!", gasp + "hey!".
 *
 * All sounds are synthesized PROCEDURALLY via OscillatorNode +
 * GainNode + BiquadFilterNode envelopes — no audio files are loaded.
 * Each region maps to a parametrized profile.
 *
 * Lifecycle (all functions accept the integration agent's wiring):
 *
 *   initAvatarAudioReactions({ vrm, voiceProfile?, audioContext? })
 *   handleAudioContact(evt)            // public hook for contact events
 *   teardownAvatarAudioReactions()     // cleanup
 *   getAvatarAudioReactions()
 *
 * Gating: reads `body_physics_audio_reactions_enabled` from the server
 * settings (default true) and re-checks every 5s. When disabled,
 * `handleAudioContact` is a no-op.
 *
 * Debouncing: same region cannot retrigger within 1500ms; the global
 * playback cap is 1 — a new contact in a different region while a clip
 * plays fades the current and starts the new; same region drops.
 *
 * Voice profile (optional): if `voiceProfile.pitchShift` is a finite
 * number of semitones, all fundamentals are scaled by 2^(pitchShift/12).
 */

const SAME_REGION_DEBOUNCE_MS = 1500;
const SETTINGS_REFRESH_MS = 5000;
const SETTING_KEY = 'body_physics_audio_reactions_enabled';
const CROSSFADE_MS = 60;

/** Region-pattern → reaction profile dispatch. Order matters — first match wins. */
const REGION_RULES = Object.freeze([
  { pattern: /^(cheek|forehead|chin|mouth|temple|jaw|nose)/, profile: 'face' },
  { pattern: /^shoulder_/,                                   profile: 'shoulder' },
  { pattern: /^hand_/,                                       profile: 'hand' },
  { pattern: /^(chest_|sternum|back_upper)/,                 profile: 'chest' },
  { pattern: /^(hip_|belly|navel|thigh_|side_)/,             profile: 'belly' },
]);

/** Module-singleton state, mirrors the pattern used by sibling XR modules. */
let _state = null;

/* ─── Public API ─────────────────────────────────────────────────────── */

/**
 * Initialize the audio reactions subsystem. Idempotent — calling again
 * tears down any prior state (e.g. after VRM swap).
 *
 * @param {object}  opts
 * @param {object}  opts.vrm               VRM the reactions are voiced "for".
 * @param {object} [opts.voiceProfile]     Optional `{ pitchShift: number }`.
 * @param {AudioContext} [opts.audioContext] Reuse caller's context if present.
 * @returns {AvatarAudioReactions|null}    the live instance, or null on failure
 */
export function initAvatarAudioReactions(opts) {
  const { vrm, voiceProfile = null, audioContext = null } = opts || {};
  teardownAvatarAudioReactions();
  if (!vrm) {
    console.warn('[avatar-audio] init missing vrm');
    return null;
  }
  const instance = new AvatarAudioReactions({ vrm, voiceProfile, audioContext });
  _state = instance;
  return instance;
}

/**
 * Public hook the contact event subscriber calls with every event the
 * reactor emits. Mirrors the shape of `_handleContact(evt, ...)` in
 * `avatar-xr-contact.js` — only `evt.region` and `evt.released` are read.
 *
 * @param {object} evt   { region, released, state, userSide, ... }
 */
export function handleAudioContact(evt) {
  if (!_state) return;
  _state.handleAudioContact(evt);
}

/** Tear down: cancel timers, stop active nodes, drop the audio context if we own it. */
export function teardownAvatarAudioReactions() {
  if (!_state) return;
  try { _state._dispose(); } catch (err) { console.debug('[avatar-audio] teardown error', err?.message); }
  _state = null;
}

/** @returns {AvatarAudioReactions|null} instance for HUD/debug inspection */
export function getAvatarAudioReactions() {
  return _state;
}

/* ─── Class ──────────────────────────────────────────────────────────── */

export class AvatarAudioReactions {
  /**
   * @param {object} opts
   * @param {object} opts.vrm
   * @param {object|null} opts.voiceProfile
   * @param {AudioContext|null} opts.audioContext
   */
  constructor(opts) {
    this.vrm = opts.vrm;
    this.voiceProfile = opts.voiceProfile || null;
    /** Pitch multiplier derived from `voiceProfile.pitchShift` semitones. */
    this.pitchScale = _pitchScaleFromProfile(this.voiceProfile);
    /** Borrowed context (don't close on teardown) vs context we created (do close). */
    this._externalCtx = !!opts.audioContext;
    /** @type {AudioContext|null} created lazily on first reaction if none was passed. */
    this.audioContext = opts.audioContext || null;
    /** Last fire time per region — for the 1500ms same-region debounce. */
    this._lastFireByRegion = new Map();
    /** Currently-playing clip handle (single-channel cap). */
    this._active = null;
    /** Server-fetched gate; defaults to true so first events fire even before /api/config returns. */
    this.enabled = true;
    this._refreshTimer = null;
    this._fetching = false;
    this._disposed = false;
    // Kick off the settings poll immediately + every 5s.
    this._scheduleRefresh(true);
  }

  /* ── settings gate ────────────────────────────────────────────────── */

  _scheduleRefresh(immediate) {
    if (this._disposed) return;
    if (immediate) this._refreshSetting();
    this._refreshTimer = setInterval(() => this._refreshSetting(), SETTINGS_REFRESH_MS);
  }

  async _refreshSetting() {
    if (this._disposed || this._fetching) return;
    this._fetching = true;
    try {
      const resp = await fetch('/api/config/tools', { credentials: 'same-origin' });
      if (!resp.ok) {
        console.debug('[avatar-audio] settings fetch non-ok', resp.status);
        return;
      }
      const data = await resp.json();
      const raw = data?.[SETTING_KEY];
      if (raw === undefined || raw === null) {
        this.enabled = true; // default
      } else if (typeof raw === 'boolean') {
        this.enabled = raw;
      } else if (typeof raw === 'string') {
        this.enabled = raw !== 'false' && raw !== '0' && raw !== '';
      } else {
        this.enabled = !!raw;
      }
    } catch (err) {
      console.debug('[avatar-audio] settings fetch failed', err?.message);
    } finally {
      this._fetching = false;
    }
  }

  /* ── contact dispatch ─────────────────────────────────────────────── */

  /**
   * Map a contact event to a region profile, apply debounce + single-
   * channel cap, then synthesize the cue. Silent no-op when disabled,
   * released, missing-region, or debounced.
   *
   * @param {object} evt
   */
  handleAudioContact(evt) {
    if (!this.enabled || this._disposed) return;
    if (!evt || evt.released) return;
    const region = String(evt.region || '');
    if (!region) return;
    const profile = _profileForRegion(region);
    if (!profile) return;

    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const last = this._lastFireByRegion.get(region) || 0;
    if (now - last < SAME_REGION_DEBOUNCE_MS) return;

    // Global cap: only one clip at a time. Different region → crossfade;
    // same region was already filtered by the debounce above.
    if (this._active) {
      if (this._active.region === region) return;
      this._fadeOutActive();
    }

    const ctx = this._ensureContext();
    if (!ctx) return;
    this._lastFireByRegion.set(region, now);
    try {
      this._playProfile(profile, region);
    } catch (err) {
      console.debug('[avatar-audio] play failed', profile, err?.message);
      this._active = null;
    }
  }

  /* ── audio context ────────────────────────────────────────────────── */

  _ensureContext() {
    if (this.audioContext) return this.audioContext;
    try {
      const Ctor = (typeof window !== 'undefined')
        ? (window.AudioContext || window.webkitAudioContext)
        : null;
      if (!Ctor) {
        console.warn('[avatar-audio] no AudioContext available');
        return null;
      }
      this.audioContext = new Ctor();
      this._externalCtx = false;
      return this.audioContext;
    } catch (err) {
      console.warn('[avatar-audio] AudioContext construct failed', err?.message);
      return null;
    }
  }

  /* ── playback ─────────────────────────────────────────────────────── */

  _playProfile(profile, region) {
    const ctx = this.audioContext;
    const t0 = ctx.currentTime;
    const nodes = []; // every node we create — for disconnect on stop
    let endAt = t0;

    switch (profile) {
      case 'face':     endAt = _playFace(ctx, t0, nodes, this.pitchScale); break;
      case 'shoulder': endAt = _playShoulder(ctx, t0, nodes, this.pitchScale); break;
      case 'hand':     endAt = _playHand(ctx, t0, nodes, this.pitchScale); break;
      case 'chest':    endAt = _playChest(ctx, t0, nodes, this.pitchScale); break;
      case 'belly':    endAt = _playBelly(ctx, t0, nodes, this.pitchScale); break;
      default: return;
    }

    // Master gain owned by us so we can crossfade on interruption.
    const master = ctx.createGain();
    master.gain.setValueAtTime(1.0, t0);
    // Re-route every node's destination through master. We did the
    // local envelopes already; master is purely an interrupt fader.
    for (const n of nodes) {
      if (n._isOutput) {
        try { n.disconnect(ctx.destination); } catch {}
        n.connect(master);
      }
    }
    master.connect(ctx.destination);
    nodes.push(master);

    const handle = {
      region,
      profile,
      endAt,
      master,
      nodes,
      timer: null,
    };
    const lifetimeMs = Math.max(50, (endAt - t0) * 1000 + 80);
    handle.timer = setTimeout(() => this._cleanup(handle), lifetimeMs);
    this._active = handle;
  }

  _fadeOutActive() {
    const handle = this._active;
    if (!handle) return;
    const ctx = this.audioContext;
    const now = ctx.currentTime;
    try {
      const g = handle.master.gain;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      g.linearRampToValueAtTime(0.0001, now + CROSSFADE_MS / 1000);
    } catch (err) { console.debug('[avatar-audio] fade failed', err?.message); }
    if (handle.timer) clearTimeout(handle.timer);
    handle.timer = setTimeout(() => this._cleanup(handle), CROSSFADE_MS + 40);
    if (this._active === handle) this._active = null;
  }

  _cleanup(handle) {
    if (!handle) return;
    if (handle.timer) { clearTimeout(handle.timer); handle.timer = null; }
    for (const n of handle.nodes) {
      try { n.stop?.(); } catch {}
      try { n.disconnect(); } catch {}
    }
    handle.nodes.length = 0;
    if (this._active === handle) this._active = null;
  }

  /* ── teardown ─────────────────────────────────────────────────────── */

  _dispose() {
    this._disposed = true;
    if (this._refreshTimer) { clearInterval(this._refreshTimer); this._refreshTimer = null; }
    if (this._active) this._cleanup(this._active);
    this._active = null;
    this._lastFireByRegion.clear();
    // Only close the context if WE created it; otherwise the caller owns it.
    if (this.audioContext && !this._externalCtx) {
      try { this.audioContext.close(); } catch {}
    }
    this.audioContext = null;
  }
}

/* ─── Profile dispatch ───────────────────────────────────────────────── */

function _profileForRegion(region) {
  for (const rule of REGION_RULES) {
    if (rule.pattern.test(region)) return rule.profile;
  }
  return null;
}

function _pitchScaleFromProfile(profile) {
  const semis = Number(profile?.pitchShift);
  if (!Number.isFinite(semis)) return 1.0;
  return Math.pow(2, semis / 12);
}

/* ─── Procedural voice generators ────────────────────────────────────── */
/* Each generator pushes nodes into `nodes[]`. Output-tier nodes get a
 * truthy `_isOutput` marker so the caller can re-route them through the
 * crossfade master. Returns the AudioContext time the clip ends at. */

/** Soft "oh!" — fundamental 220→280Hz glide, 30/80/200ms ADR, peak 0.18. */
function _playFace(ctx, t0, nodes, pitchScale) {
  const f0 = 220 * pitchScale, f1 = 280 * pitchScale;
  const attack = 0.030, sustain = 0.080, release = 0.200, peak = 0.18;
  const osc = ctx.createOscillator(); osc.type = 'sine';
  osc.frequency.setValueAtTime(f0, t0);
  osc.frequency.linearRampToValueAtTime(f1, t0 + 0.150);
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.0001, t0);
  gain.gain.linearRampToValueAtTime(peak, t0 + attack);
  gain.gain.linearRampToValueAtTime(peak, t0 + attack + sustain);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + sustain + release);
  osc.connect(gain); gain.connect(ctx.destination); gain._isOutput = true;
  osc.start(t0); osc.stop(t0 + attack + sustain + release + 0.02);
  nodes.push(osc, gain);
  return t0 + attack + sustain + release;
}

/** Chuckle — three 110Hz bursts, 80ms each, 60ms gaps, falling pitch, peak 0.15. */
function _playShoulder(ctx, t0, nodes, pitchScale) {
  const base = 110 * pitchScale, peak = 0.15;
  const burst = 0.080, gap = 0.060;
  let cursor = t0;
  for (let i = 0; i < 3; i++) {
    const osc = ctx.createOscillator(); osc.type = 'triangle';
    const f = base * (1 - i * 0.06); // falling pitch
    osc.frequency.setValueAtTime(f, cursor);
    osc.frequency.linearRampToValueAtTime(f * 0.94, cursor + burst);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, cursor);
    gain.gain.linearRampToValueAtTime(peak, cursor + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, cursor + burst);
    osc.connect(gain); gain.connect(ctx.destination); gain._isOutput = true;
    osc.start(cursor); osc.stop(cursor + burst + 0.02);
    nodes.push(osc, gain);
    cursor += burst + gap;
  }
  return cursor;
}

/** Giggle — 5 pulses 280→360Hz, 40ms each + 30ms gaps, peak 0.20. */
function _playHand(ctx, t0, nodes, pitchScale) {
  const fStart = 280 * pitchScale, fEnd = 360 * pitchScale, peak = 0.20;
  const burst = 0.040, gap = 0.030, n = 5;
  let cursor = t0;
  for (let i = 0; i < n; i++) {
    const f = fStart + (fEnd - fStart) * (i / (n - 1));
    const osc = ctx.createOscillator(); osc.type = 'sine';
    osc.frequency.setValueAtTime(f, cursor);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, cursor);
    gain.gain.linearRampToValueAtTime(peak, cursor + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, cursor + burst);
    osc.connect(gain); gain.connect(ctx.destination); gain._isOutput = true;
    osc.start(cursor); osc.stop(cursor + burst + 0.02);
    nodes.push(osc, gain);
    cursor += burst + gap;
  }
  return cursor;
}

/** Sharp inhale "ah!" — 120ms noise inhale → 200ms 180Hz vocal, peak 0.22. */
function _playChest(ctx, t0, nodes, pitchScale) {
  const peak = 0.22;
  // Inhale: short-burst filtered white noise (band-pass around ~1.6kHz).
  const inhaleDur = 0.120;
  const noiseBuf = _makeNoiseBuffer(ctx, inhaleDur);
  const noise = ctx.createBufferSource(); noise.buffer = noiseBuf;
  const noiseFilt = ctx.createBiquadFilter();
  noiseFilt.type = 'bandpass';
  noiseFilt.frequency.setValueAtTime(1600, t0);
  noiseFilt.Q.setValueAtTime(0.9, t0);
  const noiseGain = ctx.createGain();
  noiseGain.gain.setValueAtTime(0.0001, t0);
  noiseGain.gain.linearRampToValueAtTime(peak * 0.55, t0 + 0.030);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, t0 + inhaleDur);
  noise.connect(noiseFilt); noiseFilt.connect(noiseGain); noiseGain.connect(ctx.destination);
  noiseGain._isOutput = true;
  noise.start(t0); noise.stop(t0 + inhaleDur + 0.02);
  nodes.push(noise, noiseFilt, noiseGain);

  // Vocalized "ah": 180Hz with mild upward bend.
  const vStart = t0 + inhaleDur;
  const vDur = 0.200;
  const osc = ctx.createOscillator(); osc.type = 'sawtooth';
  const f = 180 * pitchScale;
  osc.frequency.setValueAtTime(f, vStart);
  osc.frequency.linearRampToValueAtTime(f * 1.06, vStart + vDur * 0.6);
  // Cheap formant-ish shaping for the "ah" vowel.
  const formant = ctx.createBiquadFilter();
  formant.type = 'lowpass';
  formant.frequency.setValueAtTime(1100, vStart);
  formant.Q.setValueAtTime(4, vStart);
  const vGain = ctx.createGain();
  vGain.gain.setValueAtTime(0.0001, vStart);
  vGain.gain.linearRampToValueAtTime(peak, vStart + 0.020);
  vGain.gain.linearRampToValueAtTime(peak * 0.8, vStart + vDur * 0.7);
  vGain.gain.exponentialRampToValueAtTime(0.0001, vStart + vDur);
  osc.connect(formant); formant.connect(vGain); vGain.connect(ctx.destination);
  vGain._isOutput = true;
  osc.start(vStart); osc.stop(vStart + vDur + 0.02);
  nodes.push(osc, formant, vGain);

  return vStart + vDur;
}

/** Gasp + "hey!" — 100ms inhale noise → 300ms 200→240Hz "hey", peak 0.25. */
function _playBelly(ctx, t0, nodes, pitchScale) {
  const peak = 0.25;
  // Gasp inhale: a touch brighter + shorter than chest's.
  const inhaleDur = 0.100;
  const noiseBuf = _makeNoiseBuffer(ctx, inhaleDur);
  const noise = ctx.createBufferSource(); noise.buffer = noiseBuf;
  const noiseFilt = ctx.createBiquadFilter();
  noiseFilt.type = 'bandpass';
  noiseFilt.frequency.setValueAtTime(2200, t0);
  noiseFilt.Q.setValueAtTime(1.1, t0);
  const noiseGain = ctx.createGain();
  noiseGain.gain.setValueAtTime(0.0001, t0);
  noiseGain.gain.linearRampToValueAtTime(peak * 0.6, t0 + 0.025);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, t0 + inhaleDur);
  noise.connect(noiseFilt); noiseFilt.connect(noiseGain); noiseGain.connect(ctx.destination);
  noiseGain._isOutput = true;
  noise.start(t0); noise.stop(t0 + inhaleDur + 0.02);
  nodes.push(noise, noiseFilt, noiseGain);

  // "hey": 200→240Hz with formant character.
  const vStart = t0 + inhaleDur;
  const vDur = 0.300;
  const osc = ctx.createOscillator(); osc.type = 'sawtooth';
  const f0 = 200 * pitchScale, f1 = 240 * pitchScale;
  osc.frequency.setValueAtTime(f0, vStart);
  osc.frequency.linearRampToValueAtTime(f1, vStart + vDur * 0.4);
  osc.frequency.linearRampToValueAtTime(f1 * 0.92, vStart + vDur);
  // Formant pair (rough "eh" → "ay"): two lowpass stages.
  const fmt1 = ctx.createBiquadFilter();
  fmt1.type = 'bandpass';
  fmt1.frequency.setValueAtTime(550, vStart);
  fmt1.Q.setValueAtTime(6, vStart);
  fmt1.frequency.linearRampToValueAtTime(700, vStart + vDur);
  const fmt2 = ctx.createBiquadFilter();
  fmt2.type = 'lowpass';
  fmt2.frequency.setValueAtTime(2000, vStart);
  fmt2.Q.setValueAtTime(2, vStart);
  const vGain = ctx.createGain();
  vGain.gain.setValueAtTime(0.0001, vStart);
  vGain.gain.linearRampToValueAtTime(peak, vStart + 0.025);
  vGain.gain.linearRampToValueAtTime(peak * 0.9, vStart + vDur * 0.65);
  vGain.gain.exponentialRampToValueAtTime(0.0001, vStart + vDur);
  osc.connect(fmt1); fmt1.connect(fmt2); fmt2.connect(vGain); vGain.connect(ctx.destination);
  vGain._isOutput = true;
  osc.start(vStart); osc.stop(vStart + vDur + 0.02);
  nodes.push(osc, fmt1, fmt2, vGain);

  return vStart + vDur;
}

/** Build a small white-noise AudioBuffer for inhale-style bursts. */
function _makeNoiseBuffer(ctx, durationSec) {
  const len = Math.max(1, Math.floor(ctx.sampleRate * durationSec));
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
  return buf;
}
