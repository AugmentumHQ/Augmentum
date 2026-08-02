/**
 * avatar-body-physics-desktop.js — desktop + mobile body-physics wiring.
 *
 * Mirrors avatar-xr.js's body-physics stack for the non-XR render path.
 * Mouse drag (desktop) and touch (mobile) feed user-hand positions into
 * the same ContactReactor + SDFCompliance + RapierRagdoll chain used in
 * VR — only the input source changes (controller/hand tracking → pointer
 * events). Tuning settings come from the shared BodyPhysicsCoordinator,
 * so adjusting the sliders in Personalize → Body Physics live-updates
 * both desktop and XR sessions.
 *
 * Lifecycle (called from avatar.js after startAnimationLoop):
 *
 *   initDesktopBodyPhysics({ three, vrm, renderer, camera })
 *   tickDesktopBodyPhysics(dt)            // dt in seconds — called via
 *                                          //  animator's onPreVrmUpdate hook
 *   teardownDesktopBodyPhysics()
 *
 * Init is idempotent — a re-init on VRM swap tears down the prior
 * instance first. If the VRM has no BodyMesh (atypical), init returns
 * null and tick is a no-op. SDFCompliance and Rapier each gracefully
 * no-op when their own prerequisites (BodyAtlas / Rapier library) are
 * unavailable, so the chain stays alive in degraded form.
 */

import {
  initXRContact, tickXRContact, teardownXRContact,
} from './avatar-xr-contact.js';
import {
  initXRCompliance, tickXRCompliance, teardownXRCompliance,
} from './avatar-xr-compliance.js';
import {
  initXRRapier, tickXRRapier, teardownXRRapier,
} from './avatar-xr-rapier.js';
import {
  initContactDesktopFallback, teardownContactDesktopFallback,
} from './contact-desktop-fallback.js';
import { initBodyPhysicsCoordinator } from './body-physics-coordinator.js';

let _state = null;

/**
 * Initialize the desktop/mobile body-physics chain against the active VRM.
 * Safe to call repeatedly — replaces any prior instance.
 *
 * @param {object} opts
 * @param {object} opts.three     THREE namespace
 * @param {object} opts.vrm       VRM with __augmentumBodyMesh (and ideally __augmentumBodyAtlas)
 * @param {object} opts.renderer  WebGLRenderer (provides .domElement for pointer events)
 * @param {object} opts.camera    Camera used for NDC → world raycast in the desktop input module
 * @returns {object|null}         internal state for diagnostics, or null on init failure
 */
export async function initDesktopBodyPhysics({ three, vrm, renderer, camera } = {}) {
  if (!three || !vrm || !renderer || !camera) {
    console.warn('[body-physics-desktop] init missing required deps');
    return null;
  }
  await teardownDesktopBodyPhysics();

  // Contact reactor — feeds embodiment events + audio reactions. Desktop
  // input mode skips controller/hand binding; pointer events drive
  // reactor.setUserHand() directly via contact-desktop-fallback.js.
  //
  // `disableReach: true` opts out of AvatarIK reach-toward-user — when the
  // pointer is over the avatar's own body, IK has no valid solution and
  // the arm chain crunches into a broken silhouette. The soft-response
  // stack (compliance indent + rapier sway + expression spikes + audio)
  // is what other shipped projects (VRChat PhysBones, VRoid, Resonite)
  // rely on for body contact, so we lean on that here. A future Phase 2
  // adds region-keyed canned flinch animations on top.
  const reactor = initXRContact({
    three, vrm, renderer,
    inputMode: 'desktop',
    disableReach: true,
  });
  if (!reactor) {
    console.debug('[body-physics-desktop] reactor init failed — disabled');
    return null;
  }

  // SDF compliance — at-contact "give" for torso/neck/shoulder/head bones.
  // Gracefully no-ops when the VRM has no BodyAtlas (404 for user VRMs).
  initXRCompliance({ three, vrm });

  // Rapier ragdoll — global chain secondary motion. Async init, gated by
  // Rapier library availability; falls back to compliance-only if absent.
  initXRRapier({ three, vrm });

  // Coordinator (5s settings poll) pushes server-side body_physics_*
  // values onto the live channel instances. Singleton — shared with XR.
  initBodyPhysicsCoordinator().catch((err) => {
    console.debug('[body-physics-desktop] coordinator init failed', err?.message);
  });

  // Pointer-event source: mouse + touch + pen unified through Pointer Events.
  initContactDesktopFallback({ three, renderer, camera, vrm, reactor });

  _state = { three, vrm, renderer, camera, reactor };
  console.debug('[body-physics-desktop] initialized');
  return _state;
}

/**
 * Tick the body-physics chain. Wired into avatar-animator.js via the
 * `onPreVrmUpdate` callback so deltas land BEFORE vrm.update bakes the
 * skeleton — same ordering invariant as the XR path.
 *
 * @param {number} dt   frame delta in SECONDS (matches animator's clock)
 */
export function tickDesktopBodyPhysics(dt) {
  if (!_state) return;
  // Convert to ms + clamp: very small dt (paused tab) → 1ms minimum to
  // keep spring math stable; very large dt (long tab freeze) → 50ms cap
  // so springs don't snap on resume.
  const dtMs = Math.max(1, Math.min(50, (Number(dt) || 0) * 1000));
  try { tickXRContact(dtMs); }
  catch (err) { console.debug('[body-physics-desktop] contact tick failed:', err?.message); }
  try { tickXRCompliance(dtMs); }
  catch (err) { console.debug('[body-physics-desktop] compliance tick failed:', err?.message); }
  try { tickXRRapier(dtMs); }
  catch (err) { console.debug('[body-physics-desktop] rapier tick failed:', err?.message); }
}

/**
 * Tear down the desktop body-physics chain. Awaits Rapier's in-flight
 * init (capped) before disposing so we never call dispose() on a half-
 * constructed world. The BodyPhysicsCoordinator singleton is intentionally
 * left alive — it's shared with XR and re-initializes cheaply.
 */
export async function teardownDesktopBodyPhysics() {
  if (!_state) return;
  try { teardownContactDesktopFallback(); }
  catch (err) { console.debug('[body-physics-desktop] desktop input teardown error:', err?.message); }
  try { teardownXRContact(); }
  catch (err) { console.debug('[body-physics-desktop] contact teardown error:', err?.message); }
  try { teardownXRCompliance(); }
  catch (err) { console.debug('[body-physics-desktop] compliance teardown error:', err?.message); }
  try { await teardownXRRapier(); }
  catch (err) { console.debug('[body-physics-desktop] rapier teardown error:', err?.message); }
  _state = null;
}

/** Diagnostic accessor — exposes the live internal state for HUDs/tests. */
export function getDesktopBodyPhysics() {
  return _state;
}
