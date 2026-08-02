/**
 * AudioBus — central coordinator for everything that makes sound.
 *
 * Problem it solves: we had six audio sources (chat TTS, voice-mode TTS,
 * audiobook, Grove HTML5 radio, Grove YouTube ambient, file-browser
 * previews) and exactly one ad-hoc duck relationship — TTS poking
 * grove.duckSoundscape() directly. Everything else played on top of
 * everything else. Adding a seventh source would have meant N² wiring.
 *
 * Model: sources "claim" the bus at one of three tiers. Claiming at a
 * higher tier ducks every active lower-tier source via its registered
 * duck(level) callback. Releasing the claim unducks them, as long as no
 * other higher-tier claim is still active. Orthogonally, kinds listed in
 * EXCLUSIVE_KINDS ('music') allow only ONE active source at a time:
 * claiming stops same-kind peers via their registered stop() callback.
 *
 *   speech  (TTS, voice-mode speech output)   highest
 *   media   (audiobook, file preview, user-opened YouTube)
 *   ambient (Grove soundscape, lo-fi radio)   lowest
 *
 * Usage:
 *
 *   const claim = AudioBus.claim({
 *     id: 'chat-tts',
 *     tier: 'speech',
 *     duck: (level) => { audioEl.volume = baseline * level; },
 *     unduck: () => { audioEl.volume = baseline; },
 *   });
 *   // ...
 *   claim.release();
 *
 * Register a long-lived source once with `register`, then call the
 * returned claim() / release() helpers on each play/pause cycle. The
 * bus only tracks active claims — sources that aren't currently making
 * sound shouldn't be claimed.
 */

const TIERS = { ambient: 0, media: 1, speech: 2 };

// Content-kind vocabulary — describes WHAT a source is, not its
// ducking priority. The Becca presence widget (and any future
// audio-reactive surface) branches embodiment on this:
//   music     → dance loop
//   narration → listening pose (audiobook, podcast)
//   dialogue  → listening pose (voice call, video w/ heavy dialogue)
//   mixed     → listening pose (video — could be either; safer default)
//   ambient   → idle / very low movement (background lo-fi)
//   speech    → lipsync (TTS — already handled by TTS analyser path)
//   sfx       → no embodiment change (UI sounds, etc.)
//   unknown   → fallback to current behavior (dance under host mode)
//
// Open vocabulary by design — sources declare what they ARE; consumers
// decide what to do. Adding a new kind doesn't require a bus change.
const KNOWN_KINDS = new Set([
  'music', 'narration', 'dialogue', 'mixed', 'ambient', 'speech', 'sfx', 'unknown',
]);

// Kinds that are EXCLUSIVE: only one source of this kind makes sound at
// a time. Claiming the bus with an exclusive kind STOPS (not ducks) every
// other active source of the same kind via its registered stop() callback.
// Ducking resolves cross-tier dominance (speech over music); exclusivity
// resolves same-role competition — Grove radio and the YouTube ambient orb
// are both ambient-tier music, so without this they stack into two
// parallel tracks ("play some jazz" → radio; "actually something else" →
// YouTube on top of the still-playing radio).
const EXCLUSIVE_KINDS = new Set(['music']);

// Duck levels for (lower-tier victim) × (highest active dominator).
// Speech over anything is aggressive (0.15); media over ambient is
// lighter (0.35) because an audiobook lets its bed-track breathe a bit.
const DUCK_LEVEL = {
  // key: tier-of-victim + ':' + tier-of-dominator → level
  'ambient:speech': 0.15,
  'ambient:media':  0.35,
  'media:speech':   0.18,
};

const _sources = new Map();   // id → { tier, duck, unduck, ducked: bool }
const _active = new Set();    // id of sources currently claiming the bus

function _highestActiveTier() {
  let highest = -1;
  for (const id of _active) {
    const src = _sources.get(id);
    if (!src) continue;
    const t = TIERS[src.tier];
    if (t > highest) highest = t;
  }
  return highest;
}

function _tierName(level) {
  for (const [name, n] of Object.entries(TIERS)) if (n === level) return name;
  return null;
}

function _stopExclusivePeers(claimantId) {
  const claimant = _sources.get(claimantId);
  if (!claimant || !EXCLUSIVE_KINDS.has(claimant.kind)) return;
  for (const otherId of Array.from(_active)) {
    if (otherId === claimantId) continue;
    const other = _sources.get(otherId);
    if (!other || other.kind !== claimant.kind) continue;
    try { other.stop?.(); } catch { /* source can't stop — fine */ }
    // Restore the victim's volume so a later manual play doesn't start
    // at a stale ducked level, then drop it from the active set now —
    // its own pause event will call release(), which is idempotent.
    if (other.ducked) {
      try { other.unduck(); } catch { /* source gone — fine */ }
      other.ducked = false;
    }
    _active.delete(otherId);
  }
}

function _applyDucking() {
  const highestTier = _highestActiveTier();
  const highestName = _tierName(highestTier);

  for (const [id, src] of _sources) {
    const myTier = TIERS[src.tier];
    const iAmDominatedBy = myTier < highestTier ? highestName : null;
    // Active sources whose tier is dominated should be ducked (regardless
    // of whether they themselves are actively claiming — Grove that kept
    // playing at user-set volume still wants to duck when TTS starts).
    const isPlaying = _active.has(id);
    if (iAmDominatedBy && isPlaying) {
      const key = `${src.tier}:${iAmDominatedBy}`;
      const level = DUCK_LEVEL[key] ?? 0.2;
      if (!src.ducked) {
        try { src.duck(level); } catch { /* source can't duck — fine */ }
        src.ducked = true;
      }
    } else if (src.ducked) {
      try { src.unduck(); } catch { /* source gone — fine */ }
      src.ducked = false;
    }
  }

  // Broadcast the new state so audio-reactive surfaces (Becca presence
  // widget, future embodiment hosts) can react. Includes the set of
  // active tiers AND the per-source kinds so subscribers can branch
  // on content type without calling debug().
  try {
    const activeTiers = new Set();
    const activeKinds = new Set();
    const activeSources = [];
    for (const id of _active) {
      const src = _sources.get(id);
      if (!src) continue;
      activeTiers.add(src.tier);
      activeKinds.add(src.kind);
      activeSources.push({ id, tier: src.tier, kind: src.kind });
    }
    window.dispatchEvent(new CustomEvent('augmentum:audio-bus-state', {
      detail: {
        highestTier: highestName,
        activeTiers: Array.from(activeTiers),
        activeKinds: Array.from(activeKinds),
        activeSources,
      },
    }));
  } catch (_) { /* document gone — fine */ }
}

function register({ id, tier, kind, duck, unduck, stop }) {
  if (!id || !TIERS.hasOwnProperty(tier)) return null;
  if (typeof duck !== 'function' || typeof unduck !== 'function') return null;
  const normKind = (kind && KNOWN_KINDS.has(kind)) ? kind : 'unknown';
  _sources.set(id, {
    tier, kind: normKind, duck, unduck,
    stop: typeof stop === 'function' ? stop : null,
    ducked: false,
  });

  return {
    claim() {
      if (!_sources.has(id)) return;
      if (_active.has(id)) return;  // idempotent
      _stopExclusivePeers(id);
      _active.add(id);
      _applyDucking();
    },
    // For sources that play more than one content kind through the same
    // element (media-player: audiobooks AND local music files). Call
    // before playback starts so exclusivity + embodiment see the truth.
    setKind(nextKind) {
      const src = _sources.get(id);
      if (!src) return;
      src.kind = (nextKind && KNOWN_KINDS.has(nextKind)) ? nextKind : 'unknown';
    },
    release() {
      if (!_active.has(id)) return;
      _active.delete(id);
      _applyDucking();
    },
    unregister() {
      _active.delete(id);
      _sources.delete(id);
      _applyDucking();
    },
    isActive() { return _active.has(id); },
  };
}

// One-shot helper for short-lived claimers (a single TTS utterance). The
// returned handle is the same shape register+claim produces, but the
// registration auto-unregisters on release.
function claim({ id, tier, kind, duck, unduck }) {
  const handle = register({ id, tier, kind, duck, unduck });
  if (!handle) return { release() {}, isActive() { return false; } };
  handle.claim();
  const originalRelease = handle.release;
  handle.release = () => {
    originalRelease();
    handle.unregister();
  };
  return handle;
}

function debug() {
  return {
    sources: Array.from(_sources.entries()).map(([id, s]) => ({
      id, tier: s.tier, kind: s.kind,
      active: _active.has(id), ducked: s.ducked,
    })),
    active: Array.from(_active),
  };
}

export const AudioBus = { register, claim, debug };
export default AudioBus;

// Expose on window for non-module scripts (media-player, TTS is already
// an ES module, but voice.js and others load as plain scripts).
if (typeof window !== 'undefined') window.AudioBus = AudioBus;
