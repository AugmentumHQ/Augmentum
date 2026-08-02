// ui/scripts/input/input-bus.js
//
// App-wide input dispatcher. Multiple sources (keyboard, gamepad,
// touch, virtual gamepad) push physical events in; consumers subscribe
// to logical actions ("jump", "next_track", "open_menu") via the
// active profile's bindings.
//
// This module is intentionally games-agnostic. It is the foundation
// for the game streaming surface (consumer #1) but is designed to
// also drive: gallery navigation, avatar control via thumbsticks,
// push-to-talk on shoulder buttons, mobile chat keyboard nav, and
// any future controller-driven UI.
//
// Design:
//   * Sources register with `bus.attachSource(name, source)`.
//   * Sources call `bus._dispatchRaw(event)` to publish raw events.
//   * Consumers subscribe via `bus.on('action', handler)` or
//     `bus.onAny(handler)` for everything.
//   * Active ActionProfile resolves raw events into logical action
//     events. Multiple physical bindings can resolve to the same
//     logical action (keyboard + gamepad both fire "jump").
//   * Profile switching is a single call; no listener reattaching.

const RAW_EVENT_KIND = Object.freeze({
  BUTTON_DOWN: 'button_down',
  BUTTON_UP: 'button_up',
  AXIS: 'axis',
});

const ACTION_EVENT_KIND = Object.freeze({
  ACTION_DOWN: 'action_down',
  ACTION_UP: 'action_up',
  AXIS: 'axis',
});

export class InputBus {
  constructor() {
    this._sources = new Map();         // name -> source instance
    this._actionListeners = new Map(); // action -> Set<handler>
    this._anyListeners = new Set();    // handler(event)
    this._rawListeners = new Set();    // handler(rawEvent) — for diagnostics
    this._profile = null;              // active ActionProfile or null
    // Track currently-held actions so we can synthesise action_up on
    // profile switch and mute releases for unbound consumers.
    this._heldActions = new Map();     // actionId -> Set<rawSourceKey>
    this._enabled = true;
  }

  // ── Source management ───────────────────────────────────────────

  attachSource(name, source) {
    if (!name || !source) return;
    if (this._sources.has(name)) {
      this.detachSource(name);
    }
    this._sources.set(name, source);
    if (typeof source.attach === 'function') source.attach(this);
  }

  detachSource(name) {
    const src = this._sources.get(name);
    if (!src) return;
    if (typeof src.detach === 'function') src.detach();
    this._sources.delete(name);
  }

  hasSource(name) { return this._sources.has(name); }

  // ── Profile management ──────────────────────────────────────────

  setProfile(profile) {
    // Releasing held actions before swapping prevents "stuck jump"
    // when you change context mid-press.
    this._releaseAllHeld();
    this._profile = profile || null;
  }

  getProfile() { return this._profile; }

  // ── Subscription API ────────────────────────────────────────────

  on(action, handler) {
    if (!action || typeof handler !== 'function') return () => {};
    let set = this._actionListeners.get(action);
    if (!set) {
      set = new Set();
      this._actionListeners.set(action, set);
    }
    set.add(handler);
    return () => set.delete(handler);
  }

  onAny(handler) {
    if (typeof handler !== 'function') return () => {};
    this._anyListeners.add(handler);
    return () => this._anyListeners.delete(handler);
  }

  onRaw(handler) {
    if (typeof handler !== 'function') return () => {};
    this._rawListeners.add(handler);
    return () => this._rawListeners.delete(handler);
  }

  // ── Source-side dispatch (called BY sources) ────────────────────
  // Sources publish a normalised raw event:
  //   { kind: 'button_down'|'button_up'|'axis', source: 'keyboard',
  //     code: 'Space' | 'gp:0:button:0' | 'gp:0:axis:0',
  //     value: 1.0, ts: performance.now() }

  _dispatchRaw(rawEvent) {
    if (!this._enabled || !rawEvent || !rawEvent.kind) return;
    rawEvent.ts = rawEvent.ts || (typeof performance !== 'undefined' ? performance.now() : Date.now());
    // Diagnostic listeners always see the raw stream.
    for (const fn of this._rawListeners) {
      try { fn(rawEvent); } catch (_) { /* swallow listener errors */ }
    }
    if (!this._profile) return;
    const resolved = this._profile.resolve(rawEvent);
    if (!resolved) return;

    // Track held state so we can release on profile switch / source detach.
    const sourceKey = `${rawEvent.source}:${rawEvent.code}`;
    if (resolved.kind === ACTION_EVENT_KIND.ACTION_DOWN) {
      let held = this._heldActions.get(resolved.action);
      if (!held) {
        held = new Set();
        this._heldActions.set(resolved.action, held);
      }
      // Suppress repeats from the same physical source.
      if (held.has(sourceKey)) return;
      held.add(sourceKey);
    } else if (resolved.kind === ACTION_EVENT_KIND.ACTION_UP) {
      const held = this._heldActions.get(resolved.action);
      if (held) {
        held.delete(sourceKey);
        if (held.size > 0) {
          // Action still held by another physical source -> don't fire up.
          return;
        }
        this._heldActions.delete(resolved.action);
      }
    }

    this._fireAction(resolved);
  }

  _fireAction(actionEvent) {
    const set = this._actionListeners.get(actionEvent.action);
    if (set) {
      for (const fn of set) {
        try { fn(actionEvent); } catch (_) { /* swallow */ }
      }
    }
    for (const fn of this._anyListeners) {
      try { fn(actionEvent); } catch (_) { /* swallow */ }
    }
  }

  _releaseAllHeld() {
    for (const [action, sources] of this._heldActions) {
      for (const sourceKey of sources) {
        const [src, code] = this._splitSourceKey(sourceKey);
        this._fireAction({
          kind: ACTION_EVENT_KIND.ACTION_UP,
          action,
          source: src,
          code,
          value: 0,
          synthetic: true,
          ts: typeof performance !== 'undefined' ? performance.now() : Date.now(),
        });
      }
    }
    this._heldActions.clear();
  }

  _splitSourceKey(key) {
    const i = key.indexOf(':');
    return i === -1 ? [key, ''] : [key.slice(0, i), key.slice(i + 1)];
  }

  // ── Lifecycle / global toggle ───────────────────────────────────

  enable() { this._enabled = true; }
  disable() {
    this._releaseAllHeld();
    this._enabled = false;
  }

  destroy() {
    for (const name of Array.from(this._sources.keys())) this.detachSource(name);
    this._actionListeners.clear();
    this._anyListeners.clear();
    this._rawListeners.clear();
    this._heldActions.clear();
    this._profile = null;
  }
}

// Singleton for the default app-wide bus. Tests construct their own.
export const inputBus = new InputBus();

export const InputEventKind = ACTION_EVENT_KIND;
export const RawInputKind = RAW_EVENT_KIND;
