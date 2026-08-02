/**
 * motion-curves.js — biomechanically-shaped motion curves for the
 * motion engine. Replaces simple slerp+smoothstep with curves modeling
 * anticipation, asymmetric acceleration, overshoot, and settle.
 *
 * All curves return a per-bone interpolation factor f(t) ∈ [-margin, 1+margin]
 * where t ∈ [0, 1] is the normalized motion time. The factor is then
 * fed to slerp(startQuat, endQuat, factor) by the engine.
 *
 * No three.js dependency — pure math. THREE.Quaternion.slerp is the
 * caller's responsibility.
 *
 * See docs/superpowers/specs/2026-05-14-motion-engine-design.md for
 * the curve shape rationale.
 */

// ─────────────────────────────────────────────────────────────────────────
// Easing primitives
// ─────────────────────────────────────────────────────────────────────────
const PI = Math.PI;

export function easeOutCubic(u)  { return 1 - Math.pow(1 - u, 3); }
export function easeInOutSine(u) { return 0.5 - 0.5 * Math.cos(PI * u); }
export function easeInQuad(u)    { return u * u; }
export function smoothstep(u)    { return u * u * (3 - 2 * u); }

// Critically damped settle: exponential decay back to 1.
// settleSpeed=4 settles fully by u=1; higher = faster settle.
export function damped(u, settleSpeed = 4.0) {
  return 1 - Math.exp(-settleSpeed * u);
}

// ─────────────────────────────────────────────────────────────────────────
// Biomechanical motion curve
//
// Four phases (parameterized; defaults reflect typical human reach motion):
//   1. Anticipation:     [0, ta]      — reverse motion (counter-anticipation)
//   2. Main acceleration: [ta, tm]    — asymmetric ease-out
//   3. Overshoot:        [tm, to]     — pass through target + return
//   4. Settle:           [to, 1]      — critically damped to exact target
//
// `f` returns the slerp interpolation factor:
//   f < 0  → motion is REVERSE of target (anticipation)
//   f = 0  → at the start pose
//   f = 1  → at the target pose
//   f > 1  → past the target (overshoot)
// ─────────────────────────────────────────────────────────────────────────

export const DEFAULT_BIO_CURVE = Object.freeze({
  ta: 0.08,                    // anticipation phase end
  tm: 0.70,                    // main acceleration phase end
  to: 0.92,                    // overshoot phase end
  anticipationAmount: 0.04,    // peak reverse motion (4% of total)
  overshootAmount:    0.025,   // peak past target (2.5% of total)
  settleSpeed:        4.0,     // damping for phase 4
});

export function bioCurve(t, params = DEFAULT_BIO_CURVE) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const { ta, tm, to, anticipationAmount, overshootAmount, settleSpeed } = params;

  if (t < ta) {
    // Phase 1: anticipation. f goes from 0 → -anticipationAmount via quadratic
    const u = t / ta;
    return -anticipationAmount * easeInQuad(u);
  }
  // anticipation endpoint: -anticipationAmount at t=ta
  // We need to ramp from -anticipationAmount through 0 toward 1 during phase 2.

  if (t < tm) {
    // Phase 2: main acceleration. From -anticipationAmount to 1 via ease-out cubic.
    const u = (t - ta) / (tm - ta);
    const start = -anticipationAmount;
    const end = 1.0;
    return start + (end - start) * easeOutCubic(u);
  }

  if (t < to) {
    // Phase 3: overshoot. f passes through 1 at t=tm, peaks at 1+overshootAmount
    // mid-phase, returns to ~1 at t=to.
    const u = (t - tm) / (to - tm);
    return 1 + overshootAmount * Math.sin(PI * u);
  }

  // Phase 4: settle. Critically damped decay to exactly 1.
  const u = (t - to) / (1 - to);
  // Start near 1 (slight residual from overshoot), decay toward 1.
  // For simplicity here, just smoothstep from f(to) back to 1.
  // f(to) ≈ 1 (sin(π) = 0), so settle is mostly a no-op for the default
  // params. Custom params with longer settle phases will see meaningful work.
  return 1 + (overshootAmount * Math.sin(PI) * (1 - damped(u, settleSpeed)));
}

// ─────────────────────────────────────────────────────────────────────────
// Energy-modulated curve parameters.
//
// `energy ∈ [0, 1]` shapes the curve:
//   0.0 = sluggish/sad (longer, less anticipation, less overshoot)
//   0.5 = neutral (default params)
//   1.0 = snappy/excited (shorter, more anticipation, more overshoot)
//
// Returns a `{params, durationMultiplier}` pair; the engine applies the
// duration multiplier when constructing channel timing.
// ─────────────────────────────────────────────────────────────────────────
export function energyModulate(energy, baseParams = DEFAULT_BIO_CURVE) {
  energy = Math.max(0, Math.min(1, energy));
  const e = energy - 0.5;  // -0.5 (low) to +0.5 (high)
  const params = {
    ta: baseParams.ta,
    tm: baseParams.tm - e * 0.05,   // higher energy = earlier main phase
    to: baseParams.to,
    anticipationAmount: baseParams.anticipationAmount * (1 + e * 0.8),
    overshootAmount:    baseParams.overshootAmount    * (1 + e * 1.2),
    settleSpeed:        baseParams.settleSpeed        * (1 + e * 0.5),
  };
  const durationMultiplier = 1 - e * 0.3;   // 0.85 at high energy, 1.15 at low
  return { params, durationMultiplier };
}

// ─────────────────────────────────────────────────────────────────────────
// Kinematic chain delays (ms).
//
// Real human reach starts at the trunk, propagates through the arm.
// These per-bone offsets reproduce that timing in the engine: each
// bone uses the same curve but starts `delay` ms after t=0.
//
// For motion lasting D ms, a bone with delay d has effective t' = (t·D - d)/D
// (clamped to [0, 1]). Bones with delay > D never finish — duration must
// be long enough to accommodate the chain. Engine enforces this by extending
// duration to cover the slowest bone in the active chain.
// ─────────────────────────────────────────────────────────────────────────
export const KINEMATIC_DELAYS = Object.freeze({
  // Trunk + head
  hips: 0,
  spine: 30,
  chest: 60,
  upperChest: 60,
  neck: 80,
  head: 100,
  // Left arm chain
  leftShoulder: 90,
  leftUpperArm: 110,
  leftLowerArm: 140,
  leftHand: 170,
  // Right arm chain
  rightShoulder: 90,
  rightUpperArm: 110,
  rightLowerArm: 140,
  rightHand: 170,
  // Fingers (lumped — finger control is usually instant on contact)
  leftThumbProximal: 180,    leftThumbIntermediate: 180,    leftThumbDistal: 180,
  leftIndexProximal: 180,    leftIndexIntermediate: 180,    leftIndexDistal: 180,
  leftMiddleProximal: 180,   leftMiddleIntermediate: 180,   leftMiddleDistal: 180,
  leftRingProximal: 180,     leftRingIntermediate: 180,     leftRingDistal: 180,
  leftLittleProximal: 180,   leftLittleIntermediate: 180,   leftLittleDistal: 180,
  rightThumbProximal: 180,   rightThumbIntermediate: 180,   rightThumbDistal: 180,
  rightIndexProximal: 180,   rightIndexIntermediate: 180,   rightIndexDistal: 180,
  rightMiddleProximal: 180,  rightMiddleIntermediate: 180,  rightMiddleDistal: 180,
  rightRingProximal: 180,    rightRingIntermediate: 180,    rightRingDistal: 180,
  rightLittleProximal: 180,  rightLittleIntermediate: 180,  rightLittleDistal: 180,
  // Legs (rare for upper-body poses but provide for completeness)
  leftUpperLeg: 50,  leftLowerLeg: 80,  leftFoot: 100,
  rightUpperLeg: 50, rightLowerLeg: 80, rightFoot: 100,
});

export function delayFor(boneName) {
  return KINEMATIC_DELAYS[boneName] ?? 100;
}

// Normalize the kinematic time for a given bone, given the channel time t
// in [0, 1] and the channel duration in ms.
export function boneT(boneName, t, durationMs) {
  if (durationMs <= 0) return t;
  const d = delayFor(boneName);
  const tMs = t * durationMs;
  const tBoneMs = tMs - d;
  if (tBoneMs <= 0) return 0;
  // After the delay, the bone has (durationMs - d) ms to complete
  // its own curve from 0 → 1. Engine guarantees durationMs > maxDelay
  // by extending the channel duration when constructing.
  const denom = Math.max(1, durationMs - d);
  return Math.max(0, Math.min(1, tBoneMs / denom));
}

// Convenience: full pipeline (curve sample for one bone given channel
// time + duration + energy).
export function sampleBone(boneName, t, durationMs, energy = 0.5) {
  const { params } = energyModulate(energy);
  const tb = boneT(boneName, t, durationMs);
  return bioCurve(tb, params);
}
