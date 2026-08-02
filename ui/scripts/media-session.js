/**
 * media-session.js — W3C Media Session bridge.
 *
 * Single owner pattern: at any time, at most one surface drives the
 * lock-screen widget, system media controls, headphone / AirPods buttons,
 * Bluetooth AVRCP, and the car head unit's now-playing tile. Ownership is
 * decided by the AudioBus state event — the highest-tier active source
 * with a registered media-session config wins; ties resolve LIFO so the
 * most recent claim takes the surface.
 *
 * 'speech' tier is intentionally excluded from ownership. A 4-second TTS
 * line shouldn't flicker the lock-screen widget over an in-progress
 * audiobook; the audiobook stays the owner and the TTS just ducks under
 * it via the existing AudioBus rules.
 *
 * Surfaces register independently of AudioBus.register so the two
 * concerns stay decoupled. The bridge subscribes to the existing
 * `augmentum:audio-bus-state` event for activation signals.
 *
 * Artwork strategy: surfaces provide a content cover URL (audiobook
 * cover, podcast art) when available, and the bridge appends the PWA
 * app icons as fallback so the lock-screen widget always shows our
 * brand even before/after content art loads. Same orbital glyph as the
 * chat-UI header, rasterized for the OS.
 *
 * Safari note: navigator.mediaSession exists on iOS Safari 15+, but
 * some action handlers are silently ignored if more than ~5 are set.
 * Setting only what the surface actually supports keeps the lock-screen
 * buttons predictable.
 *
 * Graceful no-op on browsers without navigator.mediaSession — older
 * Safari, Firefox with media.session disabled, embedded WebViews.
 */

// PWA app icons — same Augmentum brand glyph as the chat-UI header.
// Listed largest-first because iOS lock-screen prefers ≥256 and Android
// AVRCP picks the first match it likes. Both same-origin so no CORS.
const APP_ARTWORK = [
  { src: '/ui/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
  { src: '/ui/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
];

const APP_NAME = 'Augmentum';

// Tier ordering matches AudioBus. We exclude 'speech' from ownership
// (TTS utterances shouldn't claim the lock-screen surface) and rank
// 'media' above 'ambient' so an audiobook always wins over Grove radio.
const OWNER_TIER_RANK = { media: 2, ambient: 1 };

const _registered = new Map();   // id → cfg ({ getMetadata, getPosition, handlers })
let _ownerId = null;
let _positionPushTimer = null;
let _busStateSubscribed = false;

function isSupported() {
  return typeof navigator !== 'undefined'
    && 'mediaSession' in navigator
    && typeof window !== 'undefined'
    && typeof window.MediaMetadata === 'function';
}

function _resolveArtwork(artworkUrl) {
  if (!artworkUrl) return APP_ARTWORK.slice();
  // Content art leads. We don't know its sizes precisely; declaring
  // 512x512 is a hint, not a constraint — OS picks based on its own
  // preference and falls through to the app icons if the content URL
  // fails to load. Type left as image/png since covers are commonly
  // PNG; browsers don't reject on mismatch.
  return [
    { src: artworkUrl, sizes: '512x512', type: 'image/png' },
    ...APP_ARTWORK,
  ];
}

function _writeMetadata(cfg) {
  if (!isSupported() || !cfg || typeof cfg.getMetadata !== 'function') return;
  const m = cfg.getMetadata() || {};
  try {
    navigator.mediaSession.metadata = new window.MediaMetadata({
      title:   m.title   || APP_NAME,
      artist:  m.artist  || '',
      album:   m.album   || '',
      artwork: _resolveArtwork(m.artworkUrl),
    });
  } catch (err) {
    console.warn('[media-session] metadata write failed:', err);
  }
}

function _writePositionState(cfg) {
  if (!isSupported() || !cfg) return;
  if (typeof navigator.mediaSession.setPositionState !== 'function') return;
  if (typeof cfg.getPosition !== 'function') return;
  const p = cfg.getPosition() || {};
  const dur = Number(p.duration) || 0;
  const rate = Math.max(0.01, Number(p.playbackRate) || 1);
  // Live streams (Grove radio) report no duration. Clear the position
  // state so the lock-screen scrubber hides instead of showing 0:00/0:00.
  if (!Number.isFinite(dur) || dur <= 0) {
    try { navigator.mediaSession.setPositionState(); } catch { /* noop */ }
    return;
  }
  const pos = Math.max(0, Math.min(dur, Number(p.position) || 0));
  try {
    navigator.mediaSession.setPositionState({ duration: dur, position: pos, playbackRate: rate });
  } catch (err) {
    // Browsers reject pathologically bad values (e.g. dur=0 just clamped).
    // Don't propagate — better to skip one tick than break the widget.
    console.warn('[media-session] setPositionState rejected:', err);
  }
}

// Action keys we expose. Order matters on iOS Safari (only first ~5 land
// reliably); putting transport first means lock-screen play/pause work
// even when the seek/skip handlers are dropped on smaller widgets.
const _ACTION_KEYS = [
  'play', 'pause', 'stop',
  'seekbackward', 'seekforward', 'seekto',
  'previoustrack', 'nexttrack',
];

function _wireHandlers(cfg) {
  if (!isSupported()) return;
  const handlers = (cfg && cfg.handlers) || {};
  for (const key of _ACTION_KEYS) {
    const fn = handlers[key];
    try {
      if (typeof fn === 'function') {
        navigator.mediaSession.setActionHandler(key, (details) => {
          try { fn(details || {}); }
          catch (err) { console.warn(`[media-session] handler ${key} threw:`, err); }
        });
      } else {
        navigator.mediaSession.setActionHandler(key, null);
      }
    } catch (_) {
      // Browser doesn't support this specific action — leave the slot
      // empty. Not all handlers are supported everywhere (Firefox
      // doesn't take 'stop'; iOS rejects unknown keys).
    }
  }
}

function _clearHandlers() {
  if (!isSupported()) return;
  for (const key of _ACTION_KEYS) {
    try { navigator.mediaSession.setActionHandler(key, null); } catch { /* noop */ }
  }
}

function _setPlaybackState(state) {
  if (!isSupported()) return;
  try { navigator.mediaSession.playbackState = state; } catch { /* noop */ }
}

function _adoptOwner(id) {
  const cfg = _registered.get(id);
  if (!cfg) return;
  _ownerId = id;
  _writeMetadata(cfg);
  _wireHandlers(cfg);
  _writePositionState(cfg);
  _setPlaybackState('playing');
  _ensurePositionPolling();
}

function _releaseOwner() {
  if (_ownerId === null) return;
  _ownerId = null;
  _clearHandlers();
  _setPlaybackState('none');
  if (isSupported() && typeof navigator.mediaSession.setPositionState === 'function') {
    try { navigator.mediaSession.setPositionState(); } catch { /* noop */ }
  }
  _stopPositionPolling();
  // We intentionally leave metadata in place — the OS lock-screen
  // widget keeps showing the last title briefly during a pause, which
  // matches Spotify/Apple Music behaviour. The next _adoptOwner will
  // overwrite it.
}

function _ensurePositionPolling() {
  if (_positionPushTimer || !_ownerId) return;
  // 1 Hz scrubber refresh — plenty for the lock-screen UI and well
  // under any internal throttle browsers apply to setPositionState.
  _positionPushTimer = setInterval(() => {
    const cfg = _registered.get(_ownerId);
    if (cfg) _writePositionState(cfg);
  }, 1000);
}

function _stopPositionPolling() {
  if (_positionPushTimer) {
    clearInterval(_positionPushTimer);
    _positionPushTimer = null;
  }
}

function _pickOwnerFromBusEvent(detail) {
  // Detail shape comes from audio-bus.js's `augmentum:audio-bus-state`:
  // { highestTier, activeTiers, activeKinds, activeSources: [{id, tier, kind}] }
  const active = Array.isArray(detail?.activeSources) ? detail.activeSources : [];
  if (active.length === 0) return null;

  // Eligible: active, has registered MS config, and not 'speech' tier
  // (TTS shouldn't grab the lock-screen widget for one utterance).
  let best = null;
  let bestRank = -1;
  // Iterate in order; tie-break by LIFO (later wins) since the audio-
  // bus event preserves _active Set insertion order.
  for (const s of active) {
    if (!_registered.has(s.id)) continue;
    const rank = OWNER_TIER_RANK[s.tier] ?? -1;
    if (rank < 0) continue;          // 'speech' or unknown
    if (rank >= bestRank) {           // >= for LIFO tie-break
      best = s.id;
      bestRank = rank;
    }
  }
  return best;
}

function _handleBusEvent(event) {
  const nextOwner = _pickOwnerFromBusEvent(event?.detail);
  if (nextOwner === _ownerId) {
    // Same owner still active — could be a duck/unduck event that didn't
    // change ownership. Nothing to do; metadata pushes happen via the
    // surface's notifyMetadataChanged when it has news.
    return;
  }
  if (_ownerId !== null) {
    _clearHandlers();
    _stopPositionPolling();
  }
  if (nextOwner === null) {
    _releaseOwner();
    return;
  }
  _adoptOwner(nextOwner);
}

function _ensureBusSubscription() {
  if (_busStateSubscribed || typeof window === 'undefined') return;
  window.addEventListener('augmentum:audio-bus-state', _handleBusEvent);
  _busStateSubscribed = true;
}

/**
 * Register a surface's media-session config.
 *
 * cfg.getMetadata():  () => ({ title, artist, album, artworkUrl? })
 *                     Called on owner adoption + on notifyMetadataChanged.
 * cfg.getPosition():  () => ({ duration, position, playbackRate })
 *                     Optional. Called 1Hz while owned. Return
 *                     duration=0 for live streams (scrubber hides).
 * cfg.handlers:       { play, pause, stop?, seekto?, seekbackward?,
 *                       seekforward?, previoustrack?, nexttrack? }
 *                     Only what the surface supports. The bridge
 *                     forwards null for unset keys so a previous
 *                     owner's handlers don't leak.
 *
 * Idempotent: re-registering the same id replaces the prior config.
 */
function register(id, cfg) {
  if (!isSupported() || !id || !cfg) return;
  _registered.set(id, cfg);
  _ensureBusSubscription();
  // If this id is already the owner (re-register while playing),
  // refresh handlers + metadata so config changes take effect.
  if (_ownerId === id) {
    _wireHandlers(cfg);
    _writeMetadata(cfg);
    _writePositionState(cfg);
  }
}

function unregister(id) {
  _registered.delete(id);
  if (_ownerId === id) _releaseOwner();
}

/**
 * Surface tells the bridge its metadata changed (chapter advance,
 * station switch, sleep-timer state). No-op unless this id is the
 * current owner — non-owners refresh on next adoption.
 */
function notifyMetadataChanged(id) {
  if (id !== _ownerId) return;
  const cfg = _registered.get(id);
  if (cfg) _writeMetadata(cfg);
}

/**
 * Surface tells the bridge position jumped (user seek, chapter skip).
 * The 1Hz poll catches drift; this is for "right now" updates so the
 * lock-screen scrubber doesn't lag a full second after a skip.
 */
function notifyPositionChanged(id) {
  if (id !== _ownerId) return;
  const cfg = _registered.get(id);
  if (cfg) _writePositionState(cfg);
}

/**
 * Set 'playing' / 'paused' / 'none'. Surfaces call this on their
 * audio element's play/pause events so the lock-screen widget shows
 * the correct icon even when the OS asks but our audio takes a
 * moment to respond (Bluetooth round-trip).
 */
function setPlaybackState(id, state) {
  if (id !== _ownerId) return;
  _setPlaybackState(state);
}

export const MediaSessionBridge = {
  isSupported,
  register,
  unregister,
  notifyMetadataChanged,
  notifyPositionChanged,
  setPlaybackState,
};

export default MediaSessionBridge;

// Non-module surfaces can read it off window. The chat composer + a
// few learning games still load as plain scripts.
if (typeof window !== 'undefined') {
  window.AugmentumMediaSession = MediaSessionBridge;
}
