/**
 * body-physics-coordinator.js — blend-weight conductor for the hybrid
 * SDF compliance + Rapier ragdoll body-physics stack.
 *
 * Two channels write into the same VRM skeleton each frame:
 *
 *   1. SDFCompliance         (avatar-xr-compliance.js) — local at-contact
 *                            "give": chest leans back when poked, spring
 *                            damped via `stiffness` + `recoverHz`.
 *   2. RapierRagdoll wrapper (avatar-xr-rapier.js)     — global chain
 *                            secondary motion, blended via `weight`.
 *
 * Both channels expose live-mutable knobs. This coordinator owns the
 * settings sync: it fetches `/api/config/tools` on a 5s cadence, applies
 * server-side values onto the live channel instances, and exposes
 * overrides for the HUD/debug surface. Channels stay alive even when the
 * feature is "disabled" — we just zero their gains so toggle response is
 * instant and we don't pay re-init cost on each flip.
 *
 * Late-binding: either channel may be null when the coordinator first
 * ticks (depending on session init order). We skip that channel's push
 * for the cycle and try again on the next tick. The rapier wrapper file
 * itself may not exist in older builds — the import is wrapped in
 * try/catch and the coordinator gracefully runs with compliance only.
 *
 * Setting keys (resolved server-side, defaults applied here as fallback
 * when the request 401s or the network drops):
 *
 *   body_physics_enabled          bool   (default true)
 *   body_physics_compliance_gain  float  0..2  (default 1.0)
 *   body_physics_rapier_weight    float  0..2  (default 0.6)
 *   body_physics_recover_hz       float  2..20 (default 6.0)
 *
 * Lifecycle:
 *
 *   const c = await initBodyPhysicsCoordinator();   // fetches + first sync
 *   tickBodyPhysicsCoordinator();                   // optional per-frame
 *   teardownBodyPhysicsCoordinator();               // session end
 *   getBodyPhysicsCoordinator();                    // HUD inspector
 */

import { getXRCompliance } from './avatar-xr-compliance.js';
import { getXRContactReactor } from './avatar-xr-contact.js';

const REFRESH_MS = 5000;

const DEFAULTS = Object.freeze({
  body_physics_enabled:         false,  // beta — user opts in via Personalize tab
  body_physics_compliance_gain: 1.0,
  body_physics_rapier_weight:   0.6,
  body_physics_recover_hz:      6.0,
});

const CLAMP = Object.freeze({
  body_physics_compliance_gain: [0, 2],
  body_physics_rapier_weight:   [0, 2],
  body_physics_recover_hz:      [2, 20],
});

/** Late-bound, may stay null if the wrapper file is absent in this build. */
let _getXRRapier = null;
(async () => {
  try {
    const mod = await import('./avatar-xr-rapier.js');
    _getXRRapier = typeof mod?.getXRRapier === 'function' ? mod.getXRRapier : null;
  } catch (err) {
    // Wrapper not present — coordinator runs in compliance-only mode.
    console.debug('[body-physics] rapier wrapper unavailable, compliance-only', err?.message);
  }
})();

/** Module-singleton instance, mirrors the pattern used by the XR wrapper files. */
let _instance = null;

function _clamp(key, value) {
  const range = CLAMP[key];
  if (!range) return value;
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULTS[key];
  return Math.min(range[1], Math.max(range[0], n));
}

function _coerce(key, raw) {
  if (raw === undefined || raw === null) return DEFAULTS[key];
  if (key === 'body_physics_enabled') {
    if (typeof raw === 'boolean') return raw;
    if (typeof raw === 'string') return raw !== 'false' && raw !== '0' && raw !== '';
    return !!raw;
  }
  return _clamp(key, raw);
}

export class BodyPhysicsCoordinator {
  constructor() {
    /** Last values fetched from `/api/config/tools`, post-coerce. */
    this.settings = { ...DEFAULTS };
    /** HUD/debug overrides — null values mean "no override, use server value". */
    this.overrides = Object.create(null);
    this._timer = null;
    this._lastSyncAt = 0;
    this._fetching = false;
  }

  /** Effective value: override wins, else server-side setting. */
  _effective(key) {
    const ov = this.overrides[key];
    if (ov !== undefined && ov !== null) return ov;
    return this.settings[key];
  }

  /**
   * Fetch settings from the server and merge with defaults. 401/network
   * failures preserve last-known values — we only log at debug level so
   * a brief offline blip doesn't spam the console.
   */
  async refresh() {
    if (this._fetching) return;
    this._fetching = true;
    try {
      const resp = await fetch('/api/config/tools', { credentials: 'same-origin' });
      if (!resp.ok) {
        console.debug('[body-physics] settings fetch non-ok', resp.status);
        return;
      }
      const data = await resp.json();
      for (const key of Object.keys(DEFAULTS)) {
        this.settings[key] = _coerce(key, data?.[key]);
      }
      this._lastSyncAt = Date.now();
    } catch (err) {
      console.debug('[body-physics] settings fetch failed', err?.message);
    } finally {
      this._fetching = false;
    }
  }

  /**
   * Push current effective values onto the live channel instances. Each
   * channel is independently late-bound: if its accessor returns null we
   * skip the push for this cycle and try again next tick.
   */
  applyToChannels() {
    const enabled = !!this._effective('body_physics_enabled');
    const gain    = Number(this._effective('body_physics_compliance_gain'));
    const weight  = Number(this._effective('body_physics_rapier_weight'));
    const recover = Number(this._effective('body_physics_recover_hz'));

    const compliance = (() => {
      try { return getXRCompliance(); } catch { return null; }
    })();
    if (compliance) {
      compliance.stiffness = enabled ? gain : 0;
      compliance.recoverHz = recover;
    }

    const rapier = (() => {
      if (typeof _getXRRapier !== 'function') return null;
      try { return _getXRRapier(); } catch { return null; }
    })();
    if (rapier) {
      // The wrapper exposes a mutable `weight` property; zero it when
      // disabled rather than tearing the simulation down so toggles are
      // instant and physics state stays warm.
      rapier.weight = enabled ? weight : 0;
    }

    // Gate the contact reactor too — otherwise expression spikes and audio
    // cues keep firing on pointer events even when body physics is
    // "disabled". `reactor.enabled = false` short-circuits its tick (see
    // contact-reactor.js: `if (!this.enabled || !this.bodyMesh) return`),
    // so user-hand input gets dropped before any contact-state classification
    // runs. State stays warm — flipping the toggle on re-enables instantly.
    const reactor = (() => {
      try { return getXRContactReactor(); } catch { return null; }
    })();
    if (reactor) {
      reactor.enabled = enabled;
    }
  }

  /** Combined refresh + push. Called by the 5s timer and by `tick()`. */
  async sync() {
    await this.refresh();
    this.applyToChannels();
  }

  /**
   * Set an override for a setting key. Pass `null`/`undefined` to clear.
   * Values are coerced/clamped through the same path as server values.
   */
  setOverride(key, value) {
    if (!(key in DEFAULTS)) {
      console.warn('[body-physics] setOverride: unknown key', key);
      return;
    }
    if (value === null || value === undefined) {
      delete this.overrides[key];
    } else {
      this.overrides[key] = _coerce(key, value);
    }
    // Immediate push so the HUD slider feels live, no waiting for the timer.
    this.applyToChannels();
  }

  /** Snapshot for HUD readouts. */
  inspect() {
    return {
      enabled:          !!this._effective('body_physics_enabled'),
      compliance_gain:  Number(this._effective('body_physics_compliance_gain')),
      rapier_weight:    Number(this._effective('body_physics_rapier_weight')),
      recover_hz:       Number(this._effective('body_physics_recover_hz')),
      overrides:        { ...this.overrides },
      lastSyncAt:       this._lastSyncAt,
    };
  }

  /** Start the 5s refresh timer. Safe to call multiple times. */
  start() {
    if (this._timer) return;
    this._timer = setInterval(() => {
      this.sync().catch((err) => {
        console.debug('[body-physics] periodic sync failed', err?.message);
      });
    }, REFRESH_MS);
  }

  /** Stop the timer and clear pending refs. */
  stop() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }
}

/**
 * Initialize the coordinator singleton: fetches settings, kicks the
 * first sync, and starts the 5s refresh timer. Returns the live
 * instance. Idempotent — repeated calls return the existing instance
 * without restarting the timer.
 *
 * @returns {Promise<BodyPhysicsCoordinator>}
 */
export async function initBodyPhysicsCoordinator() {
  if (_instance) return _instance;
  _instance = new BodyPhysicsCoordinator();
  await _instance.sync();
  _instance.start();
  console.debug('[body-physics] coordinator initialized', _instance.inspect());
  return _instance;
}

/**
 * Push current effective values onto the live channel instances. Call
 * from the XR frame loop for instant late-binding (when compliance or
 * rapier comes online mid-session) — the 5s timer also calls this, so
 * per-frame invocation is optional. Cheap: no allocations on the hot
 * path, just a handful of property writes.
 */
export function tickBodyPhysicsCoordinator() {
  if (!_instance) return;
  _instance.applyToChannels();
}

/**
 * Tear down the coordinator: clears the refresh timer and drops the
 * singleton reference. Channel-side state is NOT modified — the
 * compliance/rapier teardown is owned by their respective wrappers.
 */
export function teardownBodyPhysicsCoordinator() {
  if (!_instance) return;
  try { _instance.stop(); } catch (err) {
    console.debug('[body-physics] stop error', err?.message);
  }
  _instance = null;
}

/** @returns {BodyPhysicsCoordinator|null} live instance for HUD inspection. */
export function getBodyPhysicsCoordinator() {
  return _instance;
}
