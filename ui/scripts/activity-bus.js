// ActivityBus — single source of truth for "what's happening right now"
// across UI subsystems.
//
// Replaces scattered module-local flags and `window.*` globals that
// different subsystems used to coordinate (chat streaming, voice speaking,
// VRM animating, drawer open, in-call, etc.). Consumers that used to read
// `_chatStreamActive` or `window.__beccaConversationState` now read
// `bus.state.chat_streaming` / `bus.state.becca_conversation`.
//
// Design choices:
//   - Flat state object, NOT a Map. `bus.state.foo` is direct property
//     access — cheap enough to read every animation frame from hot loops
//     like the avatar render loop and lipsync.
//   - Underscore keys (`voice_speaking`, not `voice.speaking`) so the
//     hot-path read stays `bus.state.voice_speaking`, no quoting needed.
//   - Synchronous fan-out. `bus.set('foo', true)` runs subscribers
//     inline before returning. Consumers that read on the same tick see
//     the new value immediately — critical for lipsync/breath coupling
//     in Phase 2. No microtask deferral; no queue.
//   - No-op on unchanged sets. Lets callers `set()` every frame without
//     waking subscribers needlessly.
//   - One singleton. There is exactly one bus for the UI.
//
// Subscribers SHOULD NOT throw. A throwing subscriber is caught and
// logged but does not abort fan-out to siblings — one buggy listener
// cannot deafen the others. State mutation from inside a subscriber is
// allowed but discouraged (re-entrant set() runs nested fan-out).

const state = Object.create(null);
const listeners = new Map();        // key -> Set<fn>
const allListeners = new Set();     // fan-out for every change (dev overlay)

function _notify(key, value, prev) {
  const set = listeners.get(key);
  if (set) {
    for (const fn of set) {
      try { fn(value, prev, key); }
      catch (err) { console.warn(`[activity-bus] subscriber for "${key}" threw:`, err); }
    }
  }
  if (allListeners.size > 0) {
    for (const fn of allListeners) {
      try { fn(key, value, prev); }
      catch (err) { console.warn('[activity-bus] subscribeAll listener threw:', err); }
    }
  }
}

export const bus = {
  /** Hot-path read surface. Subsystems doing per-frame reads should
   *  access `bus.state.<key>` directly — no method call overhead. */
  state,

  /** Publish a new value for ``key``. Synchronous fan-out: subscribers
   *  run inline. No-op when the value hasn't changed. */
  set(key, value) {
    const prev = state[key];
    if (prev === value) return;
    state[key] = value;
    _notify(key, value, prev);
  },

  /** Atomic multi-key update. All values are written FIRST, then
   *  subscribers fire — so a subscriber reading other keys via
   *  ``bus.state.*`` sees the final state, never an in-between mix.
   *  Useful when two logically-paired fields flip together (e.g. voice
   *  state changing AND tts playback ending in the same transition).
   *  Subscribers for unchanged keys are skipped. */
  setBatch(updates) {
    const changed = [];
    for (const key in updates) {
      const value = updates[key];
      const prev = state[key];
      if (prev === value) continue;
      state[key] = value;
      changed.push([key, value, prev]);
    }
    for (const [key, value, prev] of changed) _notify(key, value, prev);
  },

  /** Subscribe to changes for one key. Returns an unsubscribe fn. */
  subscribe(key, fn) {
    let set = listeners.get(key);
    if (!set) { set = new Set(); listeners.set(key, set); }
    set.add(fn);
    return () => set.delete(fn);
  },

  /** Subscribe to EVERY change. Intended for dev overlays / debug
   *  panels — not for production hot paths. Returns an unsubscribe. */
  subscribeAll(fn) {
    allListeners.add(fn);
    return () => allListeners.delete(fn);
  },

  /** Shallow copy of the current state — for snapshot/debug. */
  snapshot() {
    return { ...state };
  },
};
