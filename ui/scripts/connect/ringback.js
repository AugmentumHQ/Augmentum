/* connect/ringback.js — WebAudio-synthesized outbound dial/ring tone.
 *
 * Counterpart to ringtone.js: that module plays the loud, attention-
 * grabbing ringer to the *callee*. This one plays the soft progress
 * tone to the *caller* while they wait for the callee to answer.
 *
 * Without ringback the caller hears total silence between MSG_INVITE
 * sent and EVENT_ACCEPT received — that gap can run several seconds
 * (callee's mediaPromise + SDP exchange) and reads as "the call is
 * broken". Adding ringback puts the caller in the same audible state
 * a phone or Skype would.
 *
 * Pattern: 440+480Hz dual-tone, 2 seconds on / 4 seconds off (NA POTS
 * cadence). Softer than the receive-side ringtone because it's a
 * status cue, not an alert. Same silent-mode honour as ringtone.js.
 */

import { getSettings } from '../settings.js';

const CYCLE_MS = 6_000;       // 2s tone + 4s gap (NA standard)
const TONE_DURATION_MS = 2_000;
const FADE_MS = 40;
const GAIN_PEAK = 0.10;       // softer than ringtone (0.18)

let _audioCtx = null;
let _osc1 = null;
let _osc2 = null;
let _gain = null;
let _loopTimer = null;
let _running = false;

// ── Public API ──────────────────────────────────────────────────

/** Begin ringback. Idempotent — second call while running is a no-op. */
export function startRingback() {
  if (_running) return;
  if (_isSilent()) return;
  _running = true;
  _runOneCycle();
}

/** Stop ringback. Safe to call from anywhere. */
export function stopRingback() {
  if (!_running && !_loopTimer && !_osc1 && !_osc2) return;
  _running = false;
  if (_loopTimer) { window.clearTimeout(_loopTimer); _loopTimer = null; }
  _stopOscs();
}

// ── Internals ────────────────────────────────────────────────────

function _ensureAudioContext() {
  if (_audioCtx) return _audioCtx;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    _audioCtx = new Ctx();
  } catch (err) {
    console.warn('connect: ringback AudioContext init failed', err);
    return null;
  }
  return _audioCtx;
}

function _runOneCycle() {
  if (!_running) return;
  const ctx = _ensureAudioContext();
  if (!ctx) { _running = false; return; }

  if (ctx.state === 'suspended') {
    // The caller initiated the call via a user-gesture click, so this
    // resume() almost always succeeds. Fire-and-forget regardless.
    ctx.resume().catch(() => {});
  }

  try {
    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.value = 440;
    const osc2 = ctx.createOscillator();
    osc2.type = 'sine';
    osc2.frequency.value = 480;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(ctx.destination);

    const now = ctx.currentTime;
    const fade = FADE_MS / 1000;
    const tone = TONE_DURATION_MS / 1000;
    // Click-free envelope.
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(GAIN_PEAK, now + fade);
    gain.gain.setValueAtTime(GAIN_PEAK, now + tone - fade);
    gain.gain.linearRampToValueAtTime(0, now + tone);

    osc1.start(now);
    osc2.start(now);
    osc1.stop(now + tone);
    osc2.stop(now + tone);
    _osc1 = osc1;
    _osc2 = osc2;
    _gain = gain;

    osc1.onended = () => { if (_osc1 === osc1) _osc1 = null; };
    osc2.onended = () => {
      if (_osc2 === osc2) { _osc2 = null; _gain = null; }
    };
  } catch (err) {
    console.warn('connect: ringback oscillator failed', err);
    _running = false;
    return;
  }

  _loopTimer = window.setTimeout(() => {
    _loopTimer = null;
    _runOneCycle();
  }, CYCLE_MS);
}

function _stopOscs() {
  for (const o of [_osc1, _osc2]) {
    if (!o) continue;
    try { o.stop(); }
    catch (_) { /* already stopped */ }
  }
  _osc1 = null;
  _osc2 = null;
  _gain = null;
}

function _isSilent() {
  try {
    const s = getSettings();
    return Boolean(s && s.connectRingtoneSilent);
  } catch (_) {
    return false;
  }
}
