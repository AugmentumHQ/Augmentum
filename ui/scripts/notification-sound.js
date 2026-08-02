/* notification-sound.js — short procedural cues for notifications.
 *
 * The notification substrate's ChannelTemplate carries a `default_sound`
 * ("chime"/"ping"/"ring"/"bell"), but nothing on the client ever played
 * it — banners and toasts were silent, so a scheduled briefing landing in
 * an open tab gave no audible heads-up. This module fills that gap.
 *
 * It synthesises the cue with WebAudio (no asset files, no network) and
 * deliberately does NOT go through AudioBus.claim(): a notification cue is
 * a transient blip, not a "source", so it must not stop the user's music.
 * It layers over whatever is playing, like the avatar reaction cues
 * (avatar-audio-reactions.js) it borrows its AudioContext pattern from.
 *
 * Gating (enabled setting, quiet hours, importance) lives in the caller
 * (notifications.js). This module just plays when asked.
 */

let _ctx = null;
// Rate-limit so a burst of catch-up notifications on reconnect doesn't
// fire a machine-gun of overlapping cues.
let _lastPlayedAt = 0;
const _MIN_GAP_MS = 1200;

function _audioContext() {
  if (_ctx) return _ctx;
  const Ctor = (typeof window !== 'undefined')
    && (window.AudioContext || window.webkitAudioContext);
  if (!Ctor) return null;
  try {
    _ctx = new Ctor();
  } catch (_) {
    return null;
  }
  return _ctx;
}

// Each voice: a frequency (Hz) and a start offset (s) from the cue start.
// Soft sine partials, gentle attack/decay — comfortable, not piercing
// (matches the Claude-register softening of the chat surfaces).
const _VOICES = {
  // Rising two-note major third — "here's something."
  chime: [{ f: 660, t: 0.0 }, { f: 880, t: 0.12 }],
  // Single soft note — routine arrival.
  ping: [{ f: 784, t: 0.0 }],
  // Three quick pulses — insistent (timers / reminders).
  ring: [{ f: 880, t: 0.0 }, { f: 880, t: 0.18 }, { f: 880, t: 0.36 }],
  // Low-high pair — a soft bell.
  bell: [{ f: 523, t: 0.0 }, { f: 1046, t: 0.08 }],
  // Ascending major triad — warm and unhurried.
  bloom: [{ f: 523, t: 0.0 }, { f: 659, t: 0.1 }, { f: 784, t: 0.2 }],
  // Very short single blip — minimal, easy to ignore.
  pop: [{ f: 587, t: 0.0 }],
  // Gentle falling pair — a soft "settle".
  drop: [{ f: 880, t: 0.0 }, { f: 587, t: 0.1 }],
};

// Picker catalog — single source of truth for the Settings dropdown.
// 'auto' is handled by the caller (channel/importance logic), so it is
// not a voice here but IS offered as a picker option.
export const NOTIFICATION_SOUNDS = [
  { id: 'auto', label: 'Auto (match the notification type)' },
  { id: 'chime', label: 'Chime' },
  { id: 'bloom', label: 'Bloom' },
  { id: 'ping', label: 'Ping' },
  { id: 'bell', label: 'Bell' },
  { id: 'drop', label: 'Drop' },
  { id: 'ring', label: 'Ring (three pulses)' },
  { id: 'pop', label: 'Pop' },
];

function _playVoice(ctx, freq, startAt) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'sine';
  osc.frequency.value = freq;
  // Short envelope: quick attack, smooth decay to silence (~0.32s).
  const peak = 0.14;
  gain.gain.setValueAtTime(0.0001, startAt);
  gain.gain.exponentialRampToValueAtTime(peak, startAt + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, startAt + 0.32);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(startAt);
  osc.stop(startAt + 0.36);
}

/**
 * Play a notification cue.
 *
 * @param {string} name  One of chime/ping/ring/bell. Unknown → chime.
 * @param {{force?: boolean}} [opts]  force bypasses the rate limiter
 *        (used for the Settings preview so it always sounds).
 */
export function playNotificationSound(name = 'chime', opts = {}) {
  const ctx = _audioContext();
  if (!ctx) return;
  const now = (typeof performance !== 'undefined' && performance.now)
    ? performance.now()
    : 0;
  if (!opts.force && now && (now - _lastPlayedAt) < _MIN_GAP_MS) return;
  _lastPlayedAt = now;

  // Autoplay policy: the context may be suspended until a user gesture.
  // resume() is best-effort; if it stays suspended the cue is simply
  // skipped (the visual banner/toast still shows).
  if (ctx.state === 'suspended') {
    try { ctx.resume(); } catch (_) { /* ignore */ }
  }

  const voices = _VOICES[name] || _VOICES.chime;
  const t0 = ctx.currentTime + 0.01;
  for (const v of voices) {
    try { _playVoice(ctx, v.f, t0 + v.t); } catch (_) { /* per-voice best effort */ }
  }
}

export default { playNotificationSound, NOTIFICATION_SOUNDS };
