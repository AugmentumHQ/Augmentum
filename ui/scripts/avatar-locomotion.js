/**
 * avatar-locomotion.js — Force/torque → position/yaw integrator for the
 * avatar root. The locomotion primitive sits downstream of `fformation.js`'s
 * reactive controller: that module computes per-frame `{force, yawTorque}`
 * from the F-formation field; this module turns those into motion using
 * critically-damped Euler integration with per-frame multiplicative damping.
 *
 * Gated by the FSM: when `fsmState.isSeated()` returns true the avatar is
 * locked to a seated pose by `avatar-xr.js` and locomotion is a no-op.
 *
 * Algorithm (per `update`):
 *
 *   if fsmState && fsmState.isSeated():  return zero scratch (no mutation)
 *   clamp dt to [0, 0.25]
 *
 *   // linear
 *   body.vel.addScaledVector(force, dt)
 *   body.vel.multiplyScalar(dampingLinear)        // per-frame damping
 *   clamp body.vel length to maxSpeed
 *   posDelta = body.vel * dt
 *   body.pos.add(posDelta)
 *
 *   // angular
 *   body.yawVel = (body.yawVel + yawTorque * dt) * dampingAngular
 *   clamp |body.yawVel| to maxYawSpeed
 *   yawDelta = body.yawVel * dt
 *   body.yaw = wrapAngle(body.yaw + yawDelta)     // [-PI, PI]
 *
 * Mutation contract:
 *   The `body` argument passed to `step()` is *owned by the caller* and is
 *   *mutated in place*. The caller retains the reference and reads back the
 *   updated `{pos, vel, yaw, yawVel}` after the call. The returned scratch
 *   object is reused across calls — callers must not retain it past the
 *   next `step()` invocation.
 *
 * Body shape:
 *   {
 *     pos:    THREE.Vector3,   // world-space position
 *     vel:    THREE.Vector3,   // world-space linear velocity (m/s)
 *     yaw:    number,          // radians, world Y axis
 *     yawVel: number,          // radians/s
 *   }
 *
 * THREE.js dependency:
 *   Imported via the bare `'three'` specifier, resolved by the HTML import
 *   map (`index.html`, `avatar-testbench.html`, etc.). Matches the
 *   convention established in `fformation.js`.
 */

import * as THREE from 'three';
import { DEFAULT_GAINS } from './fformation.js';

// Fail-closed damping defaults if the constants are not present on the
// imported `DEFAULT_GAINS` for any reason (renamed, partial export, etc.).
// Values mirror the Wave 1 defaults documented in the design spec's
// "Open design decisions" section.
// TODO: verify constant name
const _FALLBACK_DAMPING_LINEAR  = 0.85;
const _FALLBACK_DAMPING_ANGULAR = 0.85;

const _DEFAULT_DAMPING_LINEAR = (
  DEFAULT_GAINS && typeof DEFAULT_GAINS.DAMPING_LINEAR === 'number'
    ? DEFAULT_GAINS.DAMPING_LINEAR
    : _FALLBACK_DAMPING_LINEAR
);
const _DEFAULT_DAMPING_ANGULAR = (
  DEFAULT_GAINS && typeof DEFAULT_GAINS.DAMPING_ANGULAR === 'number'
    ? DEFAULT_GAINS.DAMPING_ANGULAR
    : _FALLBACK_DAMPING_ANGULAR
);

// Safety clamps. Mirror the philosophy of `fformation.js`'s MAX_FORCE /
// MAX_YAW_TORQUE — preserve frame-to-frame legibility and prevent the
// avatar from "snapping" under a bad input frame.
const _DEFAULT_MAX_SPEED     = 2.5;  // m/s
const _DEFAULT_MAX_YAW_SPEED = 3.0;  // rad/s

// Frame-time clamp defending against backgrounded-tab multi-second frames.
// Identical to the convention used elsewhere in the codebase.
const _DT_MIN = 0.0;
const _DT_MAX = 0.25;

/**
 * Wrap an angle to `[-PI, PI]`. Allocation-free; trigonometric form is
 * exact for any finite input including very large magnitudes.
 * @param {number} a
 * @returns {number}
 */
function wrapAngle(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/**
 * Critically-damped Euler integrator for the avatar root body.
 *
 * Hot path is allocation-free: the `posDelta` scratch vector and the
 * `_result` scratch object are instance fields, reused across calls.
 * Callers must not retain references returned from `step()`.
 */
export class LocomotionIntegrator {
  /**
   * @param {{
   *   dampingLinear?: number,   // defaults to fformation's DAMPING_LINEAR
   *   dampingAngular?: number,  // defaults to fformation's DAMPING_ANGULAR
   *   maxSpeed?: number,        // safety clamp, default 2.5 m/s
   *   maxYawSpeed?: number,     // safety clamp, default 3.0 rad/s
   * }} [opts]
   */
  constructor(opts) {
    const o = opts || {};
    this.dampingLinear = (
      typeof o.dampingLinear === 'number' ? o.dampingLinear : _DEFAULT_DAMPING_LINEAR
    );
    this.dampingAngular = (
      typeof o.dampingAngular === 'number' ? o.dampingAngular : _DEFAULT_DAMPING_ANGULAR
    );
    this.maxSpeed = (
      typeof o.maxSpeed === 'number' ? o.maxSpeed : _DEFAULT_MAX_SPEED
    );
    this.maxYawSpeed = (
      typeof o.maxYawSpeed === 'number' ? o.maxYawSpeed : _DEFAULT_MAX_YAW_SPEED
    );

    // Scratch reused every frame. Never returned by identity to outside
    // state; only the `_result.posDelta` reference escapes, and we
    // document that callers must not retain it.
    this._posDelta = new THREE.Vector3();
    this._result = { posDelta: this._posDelta, yawDelta: 0 };
  }

  /**
   * Advance the body one tick.
   *
   * Mutates `body` in place. Reads `force` and `yawTorque` as the per-frame
   * acceleration inputs (mass is implicitly 1, so force in N is m/s^2).
   *
   * @param {{pos: THREE.Vector3, vel: THREE.Vector3, yaw: number, yawVel: number}} body
   *   Mutated in place.
   * @param {THREE.Vector3} force
   *   World-space force in N, interpreted as m/s^2 since mass = 1.
   * @param {number} yawTorque
   *   Target angular acceleration in rad/s^2.
   * @param {{isSeated: () => boolean} | null | undefined} fsmState
   *   When `fsmState.isSeated()` returns true this is a no-op and the
   *   returned scratch carries zeros.
   * @param {number} dt
   *   Seconds since previous tick, clamped to `[0, 0.25]`.
   * @returns {{posDelta: THREE.Vector3, yawDelta: number}}
   *   Scratch — do not retain past the next call. `posDelta` is a
   *   reference to an instance-owned `THREE.Vector3`.
   */
  step(body, force, yawTorque, fsmState, dt) {
    // ── Input validation ────────────────────────────────────────────
    if (!body) {
      console.warn('[avatar-locomotion] step() called with missing body');
      this._posDelta.set(0, 0, 0);
      this._result.yawDelta = 0;
      return this._result;
    }
    if (typeof dt !== 'number' || !Number.isFinite(dt)) {
      console.warn('[avatar-locomotion] step() called with non-finite dt');
      this._posDelta.set(0, 0, 0);
      this._result.yawDelta = 0;
      return this._result;
    }

    // ── FSM gate: seated states never move ──────────────────────────
    if (fsmState && typeof fsmState.isSeated === 'function' && fsmState.isSeated()) {
      this._posDelta.set(0, 0, 0);
      this._result.yawDelta = 0;
      return this._result;
    }

    // ── dt clamp ────────────────────────────────────────────────────
    const dtClamped = dt < _DT_MIN ? _DT_MIN : (dt > _DT_MAX ? _DT_MAX : dt);

    // ── Linear integration ──────────────────────────────────────────
    if (force) {
      body.vel.addScaledVector(force, dtClamped);
    }
    body.vel.multiplyScalar(this.dampingLinear);

    // Length-clamp the velocity. `clampLength` is allocation-free in
    // recent THREE builds; we use it directly to avoid manual length
    // computation.
    if (body.vel.lengthSq() > this.maxSpeed * this.maxSpeed) {
      body.vel.setLength(this.maxSpeed);
    }

    // posDelta = vel * dt — written into the persistent scratch field.
    this._posDelta.copy(body.vel).multiplyScalar(dtClamped);
    body.pos.add(this._posDelta);

    // ── Angular integration ─────────────────────────────────────────
    const tq = typeof yawTorque === 'number' && Number.isFinite(yawTorque) ? yawTorque : 0;
    let yawVel = (body.yawVel + tq * dtClamped) * this.dampingAngular;
    if (yawVel > this.maxYawSpeed) yawVel = this.maxYawSpeed;
    else if (yawVel < -this.maxYawSpeed) yawVel = -this.maxYawSpeed;
    body.yawVel = yawVel;

    const yawDelta = yawVel * dtClamped;
    body.yaw = wrapAngle(body.yaw + yawDelta);

    // ── Pack scratch result ─────────────────────────────────────────
    // `posDelta` already lives in `this._posDelta` (same reference as
    // `this._result.posDelta`); only `yawDelta` needs writing.
    this._result.yawDelta = yawDelta;
    return this._result;
  }

  /**
   * Zero the body's linear and angular velocities. Call when transitioning
   * into a seated state or on scene reset so the avatar does not carry
   * residual motion across an FSM transition.
   *
   * Does not touch `pos` or `yaw` — only the rates.
   *
   * @param {{vel: THREE.Vector3, yawVel: number}} body
   */
  reset(body) {
    if (!body) {
      console.warn('[avatar-locomotion] reset() called with missing body');
      return;
    }
    if (body.vel && typeof body.vel.set === 'function') {
      body.vel.set(0, 0, 0);
    }
    body.yawVel = 0;
  }
}
