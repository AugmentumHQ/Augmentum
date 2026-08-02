/**
 * avatar-fsm.js — Avatar locomotion / posture state machine.
 *
 * Tracks the avatar's high-level state for the WebXR session. The single
 * load-bearing interface is `currentState.isSeated()` — consumed by
 * `avatar-xr.js::_stampLockedSitPose()` to decide whether to enforce the
 * seated-pose bone lock this frame. When `isSeated()` returns false the
 * lock is bypassed and the procedural animator (see `avatar-animator.js`)
 * retains hip/leg control.
 *
 * Phase 1 scope (Sprint 1 / Unit B, lineage from Sprint 0 proxemic-schema
 * groundwork): states + transition validation + cooldown. No autonomous
 * transitions; transitions are driven only by external callers — Phase 2+
 * will introduce the ProxemicDirector as that driver. Default state is
 * SeatedDefault, which matches today's behavior exactly when no caller
 * requests a transition, making Phase 1 default-off safe.
 *
 * Style note: vanilla ES module, no framework, no console.log (uses
 * console.debug for trace and console.warn for warnings, matching the
 * rest of `ui/scripts/avatar-*`).
 */

/**
 * Enum of FSM state names. Frozen so callers cannot mutate the set.
 * @type {Readonly<{
 *   SEATED_DEFAULT: 'SeatedDefault',
 *   SEATED_LEANING: 'SeatedLeaning',
 *   SEATED_FORWARD: 'SeatedForward',
 *   SEATED_BACK:    'SeatedBack',
 *   STANDING_IDLE:  'StandingIdle',
 *   LOCOMOTING:     'Locomoting',
 * }>}
 */
export const FSM_STATES = Object.freeze({
  SEATED_DEFAULT: 'SeatedDefault',
  SEATED_LEANING: 'SeatedLeaning',
  SEATED_FORWARD: 'SeatedForward',
  SEATED_BACK:    'SeatedBack',
  STANDING_IDLE:  'StandingIdle',
  LOCOMOTING:     'Locomoting',
});

// Set of seated state names — read at most once per transition.
// Read-path methods (current/name/isSeated) do NOT touch this set; they
// hit the precomputed `_isSeated` field on the active StateDescriptor.
const _SEATED = new Set([
  FSM_STATES.SEATED_DEFAULT,
  FSM_STATES.SEATED_LEANING,
  FSM_STATES.SEATED_FORWARD,
  FSM_STATES.SEATED_BACK,
]);

// Frozen list of legal state values for O(1) membership testing in
// requestTransition() without re-walking Object.values() each call.
const _VALID_STATES = new Set(Object.values(FSM_STATES));

/** Minimum seconds between non-forced transitions. */
const _TRANSITION_COOLDOWN_S = 0.4;

/**
 * Lightweight state descriptor returned by `current()` and consumed by the
 * seated-lock hook. Identity is stable within a state and replaced on
 * transition, so reference equality (`prev !== fsm.current()`) is a valid
 * change-detection signal for consumers.
 *
 * `_isSeated` is precomputed at construction so the read-path `isSeated()`
 * is a single field read with zero allocations and zero hashing.
 */
class StateDescriptor {
  /**
   * @param {string} stateName — one of FSM_STATES values.
   * @param {number} enteredAt — seconds (monotonic clock used by FSM).
   */
  constructor(stateName, enteredAt) {
    /** @type {string} */
    this.name = stateName;
    /** @type {number} */
    this.enteredAt = enteredAt;
    /** @type {boolean} */
    this._isSeated = _SEATED.has(stateName);
  }

  /**
   * Whether this state is a seated posture. Load-bearing for the
   * `avatar-xr.js` seated-pose lock — must remain O(1) and allocation-free.
   * @returns {boolean}
   */
  isSeated() { return this._isSeated; }
}

/**
 * Finite state machine for avatar locomotion / posture.
 *
 * Intentionally inert in Phase 1: it only changes state when a host calls
 * `requestTransition()`. With no caller, it stays in `SeatedDefault` for
 * the entire session, which is exactly today's behavior — the seated lock
 * applies every frame.
 */
export class AvatarFSM {
  /**
   * @param {{
   *   now?: () => number,
   *   presence?: object | null,
   * }} [opts]
   *   - `now`: clock function returning seconds. Defaults to a
   *     `performance.now()`-derived monotonic clock. Injectable for tests
   *     and for callers that want to align the FSM with their own frame
   *     clock (e.g. XRSession time).
   *   - `presence`: optional opaque reference to a PresenceEngine. The FSM
   *     stores it for downstream consumers but does NOT call into it —
   *     coupling stays one-way and is the host's responsibility.
   */
  constructor(opts = {}) {
    this._now = typeof opts.now === 'function'
      ? opts.now
      : (() => (typeof performance !== 'undefined' && performance.now
          ? performance.now() / 1000
          : Date.now() / 1000));

    // Held opaque — never introspected by the FSM. Intentional loose
    // coupling so Phase 2 can swap presence implementations without
    // touching this file.
    this._presence = opts.presence || null;

    /** @type {Map<string, Set<Function>>} */
    this._listeners = new Map();

    /** Seconds of the last applied transition (monotonic clock). */
    this._lastTransition = -Infinity;

    /** @type {StateDescriptor} */
    this._current = new StateDescriptor(FSM_STATES.SEATED_DEFAULT, this._now());
  }

  // -------------------------------------------------------------------------
  // Read path — must remain allocation-free and O(1).
  // -------------------------------------------------------------------------

  /**
   * Current state descriptor. The `.isSeated()` method on this object is
   * the load-bearing interface consumed by the seated-pose lock.
   * Reference identity is stable within a state.
   * @returns {StateDescriptor}
   */
  current() { return this._current; }

  /**
   * Convenience accessor; same as `current().name`.
   * @returns {string}
   */
  name() { return this._current.name; }

  /**
   * Convenience: is the avatar currently in a seated state? Pass-through
   * to `current().isSeated()`. O(1), no allocation.
   * @returns {boolean}
   */
  isSeated() { return this._current._isSeated; }

  // -------------------------------------------------------------------------
  // Write path — transitions and event subscription.
  // -------------------------------------------------------------------------

  /**
   * Request a transition to a new state.
   *
   * Returns true on success, false if rejected (unknown target, illegal
   * transition, or cooldown not yet expired). A request to the current
   * state is a no-op and returns true without firing listeners.
   *
   * @param {string} toState — one of FSM_STATES values.
   * @param {{force?: boolean}} [opts] — `force: true` bypasses the cooldown
   *   gate (for explicit user requests / emergency overrides). It does NOT
   *   bypass legality checks.
   * @returns {boolean} whether the transition was applied.
   */
  requestTransition(toState, opts = {}) {
    if (!_VALID_STATES.has(toState)) {
      console.warn('avatar-fsm: unknown state requested', { toState });
      return false;
    }

    if (toState === this._current.name) return true; // no-op

    const now = this._now();
    if (!opts.force && (now - this._lastTransition) < _TRANSITION_COOLDOWN_S) {
      return false;
    }

    const from = this._current.name;
    if (!this._isLegalTransition(from, toState)) {
      console.debug('avatar-fsm: illegal transition rejected', { from, to: toState });
      return false;
    }

    this._current = new StateDescriptor(toState, now);
    this._lastTransition = now;
    this._emit('transition', { from, to: toState, at: now });
    return true;
  }

  /**
   * Subscribe to FSM events. Currently only `'transition'` is emitted,
   * with payload `{ from: string, to: string, at: number }`.
   *
   * @param {string} event
   * @param {(payload: object) => void} fn
   * @returns {() => void} unsubscribe handle.
   */
  on(event, fn) {
    if (typeof fn !== 'function') return () => {};
    let set = this._listeners.get(event);
    if (!set) {
      set = new Set();
      this._listeners.set(event, set);
    }
    set.add(fn);
    return () => {
      const s = this._listeners.get(event);
      if (s) s.delete(fn);
    };
  }

  /**
   * Diagnostics snapshot for testbench / debug overlays. Not part of the
   * stable runtime API; do not rely on its shape from production code.
   * @returns {{state: string, enteredAt: number, isSeated: boolean, lastTransition: number}}
   */
  _dbg() {
    return {
      state: this._current.name,
      enteredAt: this._current.enteredAt,
      isSeated: this._current._isSeated,
      lastTransition: this._lastTransition,
    };
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Legality matrix. Phase 1 keeps this conservative — the host (or, in
   * Phase 2+, the ProxemicDirector) is expected to upstream-validate
   * higher-level intent (e.g. that an `arrive-at-seat` event preceded a
   * Locomoting→Seated request).
   *
   * @param {string} from
   * @param {string} to
   * @returns {boolean}
   */
  _isLegalTransition(from, to) {
    const fromSeated = _SEATED.has(from);
    const toSeated = _SEATED.has(to);

    // Seated* ↔ Seated* — sub-postures are freely interchangeable.
    if (fromSeated && toSeated) return true;

    // Seated* → StandingIdle (rising). Cooldown is the only gate here in
    // Phase 1; higher-level intent validation happens upstream.
    if (fromSeated && to === FSM_STATES.STANDING_IDLE) return true;

    // StandingIdle ↔ Locomoting — driven by target_velocity in the host.
    if (from === FSM_STATES.STANDING_IDLE && to === FSM_STATES.LOCOMOTING) return true;
    if (from === FSM_STATES.LOCOMOTING && to === FSM_STATES.STANDING_IDLE) return true;

    // StandingIdle → Seated* (returning to a seat anchor without walking).
    if (from === FSM_STATES.STANDING_IDLE && toSeated) return true;

    // Locomoting → Seated* (arriving at a chair). Host determines arrival.
    if (from === FSM_STATES.LOCOMOTING && toSeated) return true;

    return false;
  }

  /**
   * Fire listeners for an event. Defensive: a listener that throws is
   * isolated — its exception is logged via console.warn and the remaining
   * listeners still run. This is critical because the FSM is upstream of
   * the seated-pose lock and must never break the render loop.
   *
   * @param {string} event
   * @param {object} payload
   */
  _emit(event, payload) {
    const fns = this._listeners.get(event);
    if (!fns || fns.size === 0) return;
    // Snapshot to a local array so a listener that unsubscribes itself
    // (or another listener) during dispatch cannot mutate the Set we are
    // iterating over.
    const snapshot = Array.from(fns);
    for (let i = 0; i < snapshot.length; i++) {
      try {
        snapshot[i](payload);
      } catch (e) {
        console.warn('avatar-fsm: listener threw', { event, error: String(e) });
      }
    }
  }
}
