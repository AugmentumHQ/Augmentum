/* connect/ringtone.js — WebAudio-synthesized incoming-call ringtone.
 *
 * No external asset — the tone is generated in-browser so we don't
 * have to ship and license an mp3. Two-pulse pattern at ~440Hz/~520Hz
 * approximating a North-American telephone ring; loops with a 4-second
 * cycle (1s ring, 3s silence) until ``stopRingtone()`` is called.
 *
 * Autoplay rules: AudioContext.resume() requires a user gesture on
 * most browsers. The first time the ringtone tries to start without
 * one, it will silently fail — that's fine, the modal is still
 * visible. Subsequent calls (after the user has clicked anywhere)
 * will succeed.
 *
 * Respects the global "connect notify silent" toggle (settings.js
 * `connect.silentMode`) — when set, this module short-circuits to a
 * no-op so users in a meeting can still see the modal without the
 * audio cue.
 */

import { getSettings } from '../settings.js';

const CYCLE_MS = 4_000;       // 1s ring + 3s gap, matches POTS pattern
const RING_DURATION_MS = 1_000;
const FADE_MS = 30;

let _audioCtx = null;
let _gain = null;
let _osc = null;
let _loopTimer = null;
let _running = false;

// ── Public API ──────────────────────────────────────────────────

/** Begin ringing. Idempotent — second call while ringing is a no-op. */
export function startRingtone() {
  if (_running) return;
  if (_isSilent()) return;
  _running = true;
  _runOneCycle();
}

/** Stop ringing. Safe to call from anywhere. */
export function stopRingtone() {
  if (!_running && !_loopTimer && !_osc) return;
  _running = false;
  if (_loopTimer) { window.clearTimeout(_loopTimer); _loopTimer = null; }
  _stopOsc();
}

// ── Internals ────────────────────────────────────────────────────

function _ensureAudioContext() {
  if (_audioCtx) return _audioCtx;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    _audioCtx = new Ctx();
  } catch (err) {
    console.warn('connect: AudioContext init failed', err);
    return null;
  }
  return _audioCtx;
}

function _runOneCycle() {
  if (!_running) return;
  const ctx = _ensureAudioContext();
  if (!ctx) { _running = false; return; }

  // Re-suspended contexts (after autoplay block) need resume(). Fire
  // and forget — if it fails, the visual cue still works.
  if (ctx.state === 'suspended') {
    ctx.resume().catch(() => { /* expected pre-gesture */ });
  }

  try {
    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.value = 440;

    const gain = ctx.createGain();
    gain.gain.value = 0;

    osc.connect(gain).connect(ctx.destination);

    const now = ctx.currentTime;
    const fade = FADE_MS / 1000;
    const ring = RING_DURATION_MS / 1000;
    // Quick fade in/out so the start/stop doesn't click.
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.18, now + fade);
    // Two-tone effect: shift to a higher pitch mid-ring.
    osc.frequency.setValueAtTime(440, now);
    osc.frequency.setValueAtTime(520, now + ring / 2);
    gain.gain.setValueAtTime(0.18, now + ring - fade);
    gain.gain.linearRampToValueAtTime(0, now + ring);

    osc.start(now);
    osc.stop(now + ring);
    _osc = osc;
    _gain = gain;

    osc.onended = () => {
      if (_osc === osc) { _osc = null; _gain = null; }
    };
  } catch (err) {
    console.warn('connect: ringtone oscillator failed', err);
    _running = false;
    return;
  }

  _loopTimer = window.setTimeout(() => {
    _loopTimer = null;
    _runOneCycle();
  }, CYCLE_MS);
}

function _stopOsc() {
  if (_osc) {
    try { _osc.stop(); }
    catch (_) { /* already stopped */ }
    _osc = null;
  }
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
