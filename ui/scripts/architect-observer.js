/* ==========================================================================
   Augmentum — Architect Observer Bridge

   Forwards client-side observations the server can't see (AudioBus
   state changes, page focus/blur, surface foreground transitions)
   onto the runtime bus via POST /api/architect/observe.

   Design constraints:
   - Deduplicate identical consecutive states (don't spam the bus
     when AudioBus fires the same kind/tier twice in a row).
   - Throttle to once per N ms minimum between identical emits to
     keep the observer's recent deque (capped at 50) usable.
   - Best-effort: a failed POST is silently dropped. We do not
     retry — observation gaps are acceptable; over-emission is not.
   - No-op when not authenticated (no auth cookie / 401 response).
   ========================================================================== */

const OBSERVE_URL = '/api/architect/observe';

// Minimum gap between identical emits, ms. AudioBus fires whenever
// any source claims/releases; an audiobook play+pause sequence can
// produce 4-6 events in <100ms. We only care about meaningful state.
const DEDUP_WINDOW_MS = 4000;

// Track last fingerprint per topic so we don't re-emit on no-op.
const _lastByTopic = new Map();

let _enabled = true;

/** Disable the bridge (e.g., for tests). */
export function disableArchitectObserver() {
  _enabled = false;
}

async function _post(topic, payload) {
  if (!_enabled) return;
  try {
    const resp = await fetch(OBSERVE_URL, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, payload }),
    });
    if (resp.status === 401) {
      // Anonymous tab — stop trying for the rest of the session.
      _enabled = false;
      return;
    }
  } catch (_) {
    // Network blip — drop. Observations are best-effort.
  }
}

/**
 * Report an attention change from any surface (article opened, media
 * started, comic opened, mode switched, ...). Topics must match the
 * server's allow-listed prefixes (surface.browse./media./comic./audio./
 * attention./narrative./coder./commands.) — the observe route folds
 * them into the companion's AttentionStore so "this page" / "this
 * song" / "this scene" / "this file" deixis has a referent.
 * Dedup-windowed; best-effort; safe to call from any module.
 */
export function reportAttention(topic, payload) {
  _maybeEmit(topic, payload || {});
}

/** Emit a topic if it differs from the last and outside the dedup window. */
function _maybeEmit(topic, payload) {
  const now = Date.now();
  const fingerprint = JSON.stringify(payload || {});
  const prev = _lastByTopic.get(topic);
  if (prev && prev.fp === fingerprint && now - prev.ts < DEDUP_WINDOW_MS) {
    return;
  }
  _lastByTopic.set(topic, { fp: fingerprint, ts: now });
  _post(topic, payload);
}

// --- AudioBus state changes -------------------------------------------------
// AudioBus dispatches `augmentum:audio-bus-state` with {highestTier,
// activeTiers, activeKinds, activeSources} whenever ducking changes.
// We emit one observation per distinct activeKinds set so the architect
// can infer "music is playing" / "audiobook is playing" / "video plays".

function _onAudioBusState(event) {
  const detail = event.detail || {};
  const kinds = Array.isArray(detail.activeKinds) ? detail.activeKinds : [];
  const tiers = Array.isArray(detail.activeTiers) ? detail.activeTiers : [];
  // Filter out 'speech' + 'sfx' — those are companion-own / UI chrome,
  // not user-attention signal.
  const meaningful = kinds.filter(k => k !== 'speech' && k !== 'sfx');
  if (meaningful.length === 0 && _lastByTopic.has('surface.audio.kind_changed')) {
    // Active sources cleared — emit an explicit "silence" event so the
    // architect knows the user's no longer listening to anything.
    _maybeEmit('surface.audio.kind_changed', { kinds: [], tiers: [] });
    return;
  }
  if (meaningful.length === 0) return;
  _maybeEmit('surface.audio.kind_changed', {
    kinds: meaningful,
    tiers,
    highest_tier: detail.highestTier || '',
  });
}

// --- Surface foreground / blur ---------------------------------------------
// The window 'focus' / 'blur' events fire when the tab itself changes
// foreground state. We track this so the architect knows whether the
// user is actively present in Augmentum vs. on another tab/app.

function _onWindowFocus() {
  _maybeEmit('surface.attention.foreground', { focused: true });
}

function _onWindowBlur() {
  _maybeEmit('surface.attention.foreground', { focused: false });
}

// --- Init -------------------------------------------------------------------

let _installed = false;

/** Attach the bridge listeners. Idempotent. */
export function installArchitectObserver() {
  if (_installed) return;
  _installed = true;

  window.addEventListener('augmentum:audio-bus-state', _onAudioBusState);
  window.addEventListener('focus', _onWindowFocus);
  window.addEventListener('blur', _onWindowBlur);
}

/** Tear down — used in tests or on signout. */
export function uninstallArchitectObserver() {
  if (!_installed) return;
  _installed = false;
  window.removeEventListener('augmentum:audio-bus-state', _onAudioBusState);
  window.removeEventListener('focus', _onWindowFocus);
  window.removeEventListener('blur', _onWindowBlur);
}

// Auto-install on module load. Safe — no-ops on missing endpoint /
// unauthenticated tabs.
installArchitectObserver();
