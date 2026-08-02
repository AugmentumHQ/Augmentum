/**
 * avatar-xr-compliance.js — production XR integration of SDFCompliance.
 *
 * Parallels avatar-xr-contact.js. Initializes the SDF-driven body
 * compliance channel against the active VRM, ticks it each frame, and
 * tears it down on session end. Reads user hand positions through the
 * ContactReactor peer (lookup via getXRContactReactor) so the two
 * features share one source of truth for input state.
 *
 * Lifecycle (called from avatar-xr.js):
 *
 *   initXRCompliance({ three, vrm })       // after _prepareVrmSceneForXR + initXRContact
 *   tickXRCompliance(dtMs)                 // every frame, AFTER pose channels write, BEFORE vrm.update
 *   teardownXRCompliance()                 // on session end
 *
 * If the VRM has no BodyAtlas, init returns null and tick is a no-op.
 * If the contact reactor isn't running, compliance still attaches but
 * has no input to react to — harmless, just dead-quiet.
 */

import { SDFCompliance } from './sdf-compliance.js';
import { getXRContactReactor } from './avatar-xr-contact.js';

let _compliance = null;
let _vrmRef = null;

/**
 * Initialize SDF compliance for an XR session. Idempotent — replaces any
 * prior instance (e.g. VRM swap mid-session).
 *
 * @param {object} opts
 * @param {object} opts.three
 * @param {object} opts.vrm
 * @returns {SDFCompliance|null}
 */
export function initXRCompliance({ three, vrm }) {
  if (!three || !vrm) {
    console.warn('[xr-compliance] init missing required deps (three/vrm)');
    return null;
  }
  if (!vrm.__augmentumBodyAtlas) {
    console.debug('[xr-compliance] vrm has no BodyAtlas — compliance disabled');
    return null;
  }
  teardownXRCompliance();

  const reactor = getXRContactReactor();
  if (!reactor) {
    console.debug('[xr-compliance] no contact reactor yet — compliance will attach but stay idle');
  }

  _compliance = new SDFCompliance({
    three, vrm,
    contactReactor: reactor,    // may be null at init; tick rebinds via late-lookup below
  });
  _vrmRef = vrm;

  console.debug('[xr-compliance] initialized', {
    hasAtlas: !!vrm.__augmentumBodyAtlas,
    hasReactor: !!reactor,
    trackedBones: 7,
  });
  return _compliance;
}

/**
 * Tick the compliance channel. Reads user hand positions via the contact
 * reactor (late-bound if it wasn't ready at init), writes bone rotation
 * deltas. Safe to call when no compliance is active — no-ops.
 */
export function tickXRCompliance(dtMs) {
  if (!_compliance) return;
  // Late-bind reactor: contact reactor may have initialized after compliance
  // (depends on ordering in avatar-xr.js session start). Pick it up the
  // first frame it becomes available.
  if (!_compliance.contactReactor) {
    _compliance.contactReactor = getXRContactReactor();
  }
  _compliance.tick(dtMs);
}

/** Tear down on session end. */
export function teardownXRCompliance() {
  if (!_compliance) return;
  try { _compliance.dispose(); } catch (err) {
    console.debug('[xr-compliance] dispose error:', err?.message);
  }
  _compliance = null;
  _vrmRef = null;
}

/** Get the live compliance instance for inspectors / debug HUDs. */
export function getXRCompliance() {
  return _compliance;
}
