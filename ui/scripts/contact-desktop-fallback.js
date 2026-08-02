/**
 * contact-desktop-fallback.js — mouse + touch virtual-hand input for the
 * desktop/mobile body-physics path.
 *
 * The avatar's contact / approach / hover state machine and any region-
 * keyed reactions normally fire from XR controller / hand-tracking poses
 * fed by `avatar-xr-contact.js`. This module is the desktop/mobile
 * analogue: a `pointerdown`/`pointermove` drag on the renderer canvas
 * raycasts into a virtual sphere around the VRM torso, and the resulting
 * world point is pushed to `reactor.setUserHand('R', [x, y, z])` exactly
 * as the XR path would. Pointer Events unify mouse + touch + pen, so the
 * same handlers cover desktop and mobile.
 *
 * No visual indicator — the cursor sits on the body where you point,
 * and the avatar's reaction (compliance indent, expression, sway) IS
 * the visual feedback. Earlier dev iterations rendered a wireframe
 * cue sphere; production drops it as a UX nit.
 *
 * Lifecycle (mirrors avatar-xr-contact.js):
 *
 *   initContactDesktopFallback({ three, renderer, camera, vrm, reactor })
 *   tickContactDesktopFallback(dtMs)     // optional; event-driven path
 *                                         //  handles state; tick is
 *                                         //  reserved for future polling
 *   teardownContactDesktopFallback()
 *   getContactDesktopFallback()
 *
 * Etiquette: any pointer drag inside the renderer canvas counts, except
 * when Ctrl/Alt/Meta is held (those modifiers are reserved for future
 * camera gestures / OrbitControls — even though the current avatar
 * viewer doesn't register OrbitControls, leaving them free avoids a
 * surprise conflict later). XR sessions short-circuit so the desktop
 * path never competes with controllers / hand tracking.
 *
 * Single-hand model: every drag feeds the right virtual hand. Two-hand
 * input (Shift modifier or second touch) is a future expansion when the
 * tuning loop demands it.
 *
 * Wired in production by avatar-body-physics-desktop.js when the avatar
 * activates outside of an XR session.
 */

const VIRTUAL_SPHERE_RADIUS_M = 0.40;   // radius around torso for fallback raycast (BodyMesh-less VRMs only)
const WHEEL_DEPTH_SCALE       = 0.02;   // meters per wheel-delta unit (push into body)

let _state = null;

/**
 * Initialize desktop fallback. Idempotent — calling again replaces any
 * prior state (handy for VRM swaps mid-session). Safe to call without
 * an active reactor; the listeners just no-op until one is provided.
 *
 * @param {object} opts
 * @param {object} opts.three      THREE namespace
 * @param {object} opts.renderer   WebGLRenderer whose .domElement we listen on
 * @param {object} opts.camera     Camera used for NDC → world raycast
 * @param {object} opts.vrm        VRM whose torso anchors the virtual sphere
 * @param {object} opts.reactor    ContactReactor instance to feed
 * @returns {object|null}          internal state for diagnostics, or null
 */
export function initContactDesktopFallback(opts) {
  const { three, renderer, camera, vrm, reactor } = opts || {};
  if (!three || !renderer || !camera || !vrm || !reactor) {
    console.warn('[contact-desktop] init missing required deps (three/renderer/camera/vrm/reactor)');
    return null;
  }
  if (!renderer.domElement) {
    console.warn('[contact-desktop] renderer has no domElement');
    return null;
  }
  teardownContactDesktopFallback();

  const ray = new three.Raycaster();
  const ndc = new three.Vector2();
  const sphereCenter = new three.Vector3();
  const tmpHit = new three.Vector3();

  const ctx = {
    three, renderer, camera, vrm, reactor,
    ray, ndc, sphereCenter, tmpHit,
    dragging: false,
    side: 'R',
    depthOffset: 0,      // wheel-scrolled bias along view direction (meters)
    lastWorld: null,     // last [x,y,z] handed to the reactor
    detachers: [],
  };

  const onPointerDown = (ev) => _onPointerDown(ctx, ev);
  const onPointerMove = (ev) => _onPointerMove(ctx, ev);
  const onPointerUp   = (ev) => _onPointerUp(ctx, ev);
  const onPointerLeave = (ev) => _onPointerUp(ctx, ev);
  const onPointerCancel = (ev) => _onPointerUp(ctx, ev);
  const onWheel = (ev) => _onWheel(ctx, ev);

  const el = renderer.domElement;
  el.addEventListener('pointerdown', onPointerDown);
  el.addEventListener('pointermove', onPointerMove);
  el.addEventListener('pointerup', onPointerUp);
  el.addEventListener('pointerleave', onPointerLeave);
  el.addEventListener('pointercancel', onPointerCancel);
  el.addEventListener('wheel', onWheel, { passive: false });
  ctx.detachers.push(
    () => el.removeEventListener('pointerdown', onPointerDown),
    () => el.removeEventListener('pointermove', onPointerMove),
    () => el.removeEventListener('pointerup', onPointerUp),
    () => el.removeEventListener('pointerleave', onPointerLeave),
    () => el.removeEventListener('pointercancel', onPointerCancel),
    () => el.removeEventListener('wheel', onWheel),
  );

  _state = ctx;
  console.debug('[contact-desktop] initialized — drag or touch the avatar to push');
  return ctx;
}

/**
 * Optional tick. The event-driven path keeps state up to date on its
 * own; this hook exists so callers can wire it into their RAF loop if
 * future enhancements (idle drift, replay, smoothing) need polling.
 * Currently a no-op so the integration site can call it unconditionally.
 *
 * @param {number} _dtMs frame delta in milliseconds (unused)
 */
export function tickContactDesktopFallback(_dtMs) {
  // Reserved. All state updates happen in pointer/wheel handlers.
}

/** Tear down on session end / VRM swap / unload. Detaches all listeners
 *  and clears the reactor's user-hand state to avoid a stuck pose. */
export function teardownContactDesktopFallback() {
  if (!_state) return;
  try {
    _state.reactor?.setUserHand?.(_state.side, null);
  } catch {}
  for (const detach of _state.detachers) {
    try { detach(); } catch {}
  }
  _state = null;
}

/** Diagnostic accessor — exposes the live internal state for HUDs/tests. */
export function getContactDesktopFallback() {
  return _state;
}

// ─── Internal handlers ────────────────────────────────────────────────────

function _onPointerDown(ctx, ev) {
  // Touch + mouse + pen all dispatch pointer events; button is 0 for
  // touch/pen and primary mouse, but pointerType is the cleaner gate
  // for mouse-specific paths. Reject middle/right mouse only.
  if (ev.pointerType === 'mouse' && ev.button !== 0) return;
  if (ctx.renderer.xr?.isPresenting) return;        // never compete with XR
  if (ev.ctrlKey || ev.altKey || ev.metaKey) return; // leave room for camera ops

  // Single-hand model: every drag drives the right virtual hand.
  ctx.side = 'R';
  ctx.depthOffset = 0;
  ctx.dragging = true;
  try { ctx.renderer.domElement.setPointerCapture?.(ev.pointerId); } catch {}

  const world = _raycastVirtualHand(ctx, ev);
  if (world) _pushHand(ctx, world);
  ev.preventDefault();
}

function _onPointerMove(ctx, ev) {
  if (!ctx.dragging) return;
  if (ctx.renderer.xr?.isPresenting) { _onPointerUp(ctx, ev); return; }
  const world = _raycastVirtualHand(ctx, ev);
  if (!world) return;
  _pushHand(ctx, world);
}

function _onPointerUp(ctx, ev) {
  if (!ctx.dragging) return;
  ctx.dragging = false;
  try { ctx.reactor?.setUserHand?.(ctx.side, null); } catch {}
  ctx.lastWorld = null;
  try { ctx.renderer.domElement.releasePointerCapture?.(ev.pointerId); } catch {}
}

function _onWheel(ctx, ev) {
  if (!ctx.dragging) return;
  if (ctx.renderer.xr?.isPresenting) return;
  ctx.depthOffset += ev.deltaY * WHEEL_DEPTH_SCALE;
  // Recompute hand with same NDC + new depth offset so the reactor
  // reflects the push.
  const world = _raycastVirtualHand(ctx, ev);
  if (world) {
    _pushHand(ctx, world);
  }
  ev.preventDefault();
}

// ─── Internal helpers ─────────────────────────────────────────────────────

/** Push a hand world position to the reactor (with no-op guard). */
function _pushHand(ctx, world) {
  ctx.lastWorld = world;
  try { ctx.reactor?.setUserHand?.(ctx.side, world); } catch (err) {
    console.warn('[contact-desktop] reactor.setUserHand threw:', err?.message);
  }
}

/** Resolve VRM torso world position. Falls back through the humanoid
 *  bone preference order (chest → upperChest → spine → hips → scene). */
function _getTorsoWorld(ctx, out) {
  const humanoid = ctx.vrm.humanoid;
  const get = humanoid?.getNormalizedBoneNode?.bind(humanoid);
  const node = (get && (get('chest') || get('upperChest') || get('spine') || get('hips')))
    || ctx.vrm.scene;
  node.updateMatrixWorld?.(true);
  node.getWorldPosition(out);
  return out;
}

/** Resolve the virtual-hand world position for the given pointer event.
 *
 *  Strategy: cast the camera→pointer ray, find the point on that ray
 *  closest to the torso center, then snap THAT to the avatar's actual
 *  skinned mesh via BodyMesh's BVH `closestPoint` (millimeter precision,
 *  same substrate the rest of the contact system reads from). Push
 *  slightly inward along the surface normal so the resulting position
 *  is inside ContactReactor's CONTACT_DIST_M (~2cm) threshold —
 *  otherwise we'd hover at "approach" forever and nothing on the body
 *  would actually fire (compliance, expression spikes, audio).
 *
 *  Why not raycast the VRM scene directly: Three.js Raycaster against
 *  a SkinnedMesh hits BIND-POSE triangle positions, not the live posed
 *  mesh. BodyMesh's substrate has a BVH built from the same posed-mesh
 *  extraction used by SDFCompliance and ContactReactor, so we share a
 *  consistent view of "where is Becca's body actually right now".
 *
 *  Wheel-depth (mouse-only) adds extra inward push for harder presses,
 *  which scales SDFCompliance's response (deeper SDF penetration =
 *  larger bone displacement). On touch input the depth stays at 0 and
 *  contact still fires — finger taps register the same as light mouse
 *  drags.
 */
function _raycastVirtualHand(ctx, ev) {
  const rect = ctx.renderer.domElement.getBoundingClientRect();
  ctx.ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  ctx.ndc.y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
  ctx.ray.setFromCamera(ctx.ndc, ctx.camera);

  // 1. Find the point on the camera ray closest to the torso — gives a
  //    stable query seed even when the cursor drifts off Becca's
  //    silhouette (still maps to the nearest body part).
  _getTorsoWorld(ctx, ctx.sphereCenter);
  const o = ctx.ray.ray.origin;
  const d = ctx.ray.ray.direction;
  const cx = ctx.sphereCenter.x - o.x;
  const cy = ctx.sphereCenter.y - o.y;
  const cz = ctx.sphereCenter.z - o.z;
  const tca = d.x * cx + d.y * cy + d.z * cz;
  const queryX = o.x + d.x * tca;
  const queryY = o.y + d.y * tca;
  const queryZ = o.z + d.z * tca;

  // 2. Snap to actual body surface using the BVH from BodyMesh.
  const bodyMesh = ctx.vrm.__augmentumBodyMesh;
  if (bodyMesh?.closestPoint) {
    const hit = bodyMesh.closestPoint([queryX, queryY, queryZ]);
    if (hit?.point) {
      // Push inward along surface normal so distance to body is <
      // CONTACT_DIST_M. 0.015m by default = 1.5cm into the body, well
      // inside the 2cm contact threshold; wheel-scroll adds more.
      const baseDepth = 0.015 + Math.max(0, ctx.depthOffset);
      const nx = hit.normal?.[0] || 0;
      const ny = hit.normal?.[1] || 0;
      const nz = hit.normal?.[2] || 0;
      ctx.tmpHit.set(
        hit.point[0] - nx * baseDepth,
        hit.point[1] - ny * baseDepth,
        hit.point[2] - nz * baseDepth,
      );
      return [ctx.tmpHit.x, ctx.tmpHit.y, ctx.tmpHit.z];
    }
  }

  // 3. Fallback if BodyMesh isn't available (very early during VRM load,
  //    or a substrate-less VRM): use the legacy ray-sphere intersection
  //    so we don't blank out entirely — produces approach-level events.
  const d2 = (cx * cx + cy * cy + cz * cz) - tca * tca;
  const r2 = VIRTUAL_SPHERE_RADIUS_M * VIRTUAL_SPHERE_RADIUS_M;
  let t;
  if (d2 > r2) {
    t = tca;
  } else {
    const thc = Math.sqrt(r2 - d2);
    t = Math.max(0, tca - thc);
  }
  t += ctx.depthOffset;
  ctx.tmpHit.copy(o).addScaledVector(d, t);
  return [ctx.tmpHit.x, ctx.tmpHit.y, ctx.tmpHit.z];
}
