/**
 * avatar-xr-rapier.js — production XR integration of RapierRagdoll.
 *
 * Parallels avatar-xr-compliance.js / avatar-xr-contact.js. Owns a single
 * RapierRagdoll instance for the active VRM, kicks off its async physics
 * init in the background, ticks it each frame, and tears it down on
 * session end. The underlying ragdoll exposes `enabled=false` if Rapier
 * fails to load — wrapper accepts that gracefully so callers can keep
 * ticking blindly without guarding on init completion.
 *
 * Lifecycle (called from avatar-xr.js):
 *
 *   initXRRapier({ three, vrm })       // after _prepareVrmSceneForXR
 *   tickXRRapier(dtMs)                 // every frame, BEFORE vrm.update
 *   teardownXRRapier()                 // on session end
 *
 * Init is fire-and-forget: the instance is returned synchronously, but
 * its `init()` promise runs in the background. Frames before init
 * resolves no-op via the ragdoll's internal `enabled` gate. Teardown
 * awaits the in-flight init (capped at 500ms) so we never call
 * `dispose()` on a half-constructed Rapier world.
 *
 * Bone deltas (getBoneDeltas) are produced by the ragdoll but consumed
 * elsewhere — wrapper exposes the live instance via getXRRapier() so
 * a coordinator / HUD can read them without going through this module.
 */

import { RapierRagdoll } from './rapier-ragdoll.js';

let _rapier = null;
let _initPromise = null;

/**
 * Initialize Rapier ragdoll physics for an XR session. Idempotent —
 * calling again tears down the prior instance first (e.g. VRM swap
 * mid-session). Returns the live instance synchronously; the
 * underlying async init runs in the background and flips
 * `instance.enabled` once Rapier is loaded.
 *
 * @param {object} opts
 * @param {object} opts.three
 * @param {object} opts.vrm
 * @param {number} [opts.weight]   blend weight (defaults to ragdoll's own default of 0.6)
 * @returns {RapierRagdoll|null}
 */
export function initXRRapier({ three, vrm, weight } = {}) {
  if (!three || !vrm) {
    console.warn('[xr-rapier] init missing required deps (three/vrm)');
    return null;
  }
  // Tear down any prior instance — important when VRM swaps mid-session.
  teardownXRRapier();

  try {
    _rapier = new RapierRagdoll({ three, vrm, weight });
  } catch (err) {
    console.debug('[xr-rapier] construction failed:', err?.message);
    _rapier = null;
    return null;
  }

  console.debug('[xr-rapier] initializing', {
    hasVrm: !!vrm,
    weight: _rapier.weight,
  });

  // Fire-and-forget async init. The ragdoll's `enabled` flag gates its
  // own tick(), so the caller can start calling tickXRRapier
  // immediately — frames before init completes just no-op internally.
  _initPromise = Promise.resolve()
    .then(() => _rapier?.init())
    .then(() => {
      if (!_rapier) return; // teardown raced
      let bonesTracked = 0;
      try { bonesTracked = _rapier.getBoneDeltas()?.size || 0; } catch {}
      console.debug('[xr-rapier] ready', {
        bonesTracked,
        weight: _rapier.weight,
        enabled: _rapier.enabled,
      });
    })
    .catch((err) => {
      console.debug('[xr-rapier] init failed (rapier disabled):', err?.message);
      if (_rapier) {
        try { _rapier.enabled = false; } catch {}
      }
    });

  return _rapier;
}

/**
 * Tick the ragdoll. Safe to call before init completes — the ragdoll's
 * `enabled` gate makes pre-init ticks a no-op. Also safe when no
 * ragdoll is active (session not started, or init failed).
 *
 * @param {number} dtMs   frame delta in milliseconds
 */
export function tickXRRapier(dtMs) {
  if (!_rapier) return;
  try {
    _rapier.tick(dtMs);
  } catch (err) {
    console.debug('[xr-rapier] tick error:', err?.message);
  }
}

/**
 * Tear down on session end. Awaits any in-flight init (capped at 500ms)
 * so we don't dispose a half-constructed Rapier world, then disposes
 * and clears refs. Safe to call when nothing is active.
 */
export async function teardownXRRapier() {
  if (!_rapier && !_initPromise) return;

  // Wait for in-flight init to settle so dispose() sees a consistent
  // state. Cap at 500ms — if init is hung past that, the world is
  // either never coming up or already broken; either way, drop refs.
  if (_initPromise) {
    try {
      await Promise.race([
        _initPromise,
        new Promise((resolve) => setTimeout(resolve, 500)),
      ]);
    } catch { /* in-flight init rejected — dispose path below cleans up anyway */ }
  }

  if (_rapier) {
    try { _rapier.dispose(); } catch (err) {
      console.debug('[xr-rapier] dispose error:', err?.message);
    }
  }
  _rapier = null;
  _initPromise = null;
}

/** Get the live ragdoll instance for inspectors / debug HUDs / coordinator. */
export function getXRRapier() {
  return _rapier;
}
