/**
 * fformation.js — Reactive F-formation controller for dyadic conversation.
 *
 * Computes a target avatar position + facing for a single user-avatar
 * dyad in a WebXR scene, given the user's head pose, the avatar's
 * current pose, and (optionally) a 3D attention anchor such as a panel
 * or shared object. Outputs reactive forces consumed by the locomotion
 * primitive; outputs the chosen F-formation enum for downstream
 * behavior (gaze, posture, conversational policy).
 *
 * Algorithmic basis:
 *   - Pedica, C. & Vilhjálmsson, H. (2008). "Social perception and
 *     steering for online avatars." IVA 2008.
 *   - Pedica, C. & Vilhjálmsson, H. (2010). "Spontaneous avatar behavior
 *     for human territoriality." Applied Artificial Intelligence.
 *     [Both papers paywalled — the dyadic case here is distilled from
 *     secondary citations during Sprint 0 research. Constants marked
 *     [INFERRED] are pragmatic defaults pending paper retrieval.]
 *   - Kendon, A. (1990). "Conducting Interaction." (F-formation taxonomy.)
 *   - Hall, E. T. (1966). "The Hidden Dimension." (proxemic zones.)
 *   - Bönsch, A. et al. (2018). "Social VR: How Personal Space Is Affected
 *     by Virtual Agents' Emotions." IEEE VR. (VR-tuned personal-zone
 *     radius — ~20% tighter than Hall's real-world figure.)
 *
 * Phase 1 scope (this file):
 *   - Dyadic detection only (DYAD_VIS_A_VIS, DYAD_L_SHAPE,
 *     DYAD_SIDE_BY_SIDE, NONE). Crowd-N branches deliberately omitted.
 *   - Affect-modulated gain shaping (presence, flow, resonance).
 *   - Intent + intentOverride wired but no-op-safe (LLM intent path
 *     lands in Sprint 2).
 *   - No allocations on the hot `update()` path — scratch vectors
 *     pre-allocated in the constructor; the method runs at 72-90 Hz
 *     in XR and allocation churn there causes audible GC stalls.
 *
 * THREE.js dependency:
 *   The project ships Three via an HTML import map (`index.html`,
 *   `avatar-testbench.html`, etc.) that maps the bare `'three'`
 *   specifier to `ui/lib/three/three.module.min.js`. We use the bare
 *   specifier so this module composes cleanly with every existing
 *   entry-point HTML without forcing a relative-path coupling.
 */

import * as THREE from 'three';

// ─── Enums / constants ────────────────────────────────────────────────

/** F-formation taxonomy values produced by the detector. */
export const FORMATION = Object.freeze({
  NONE: 'NONE',
  VIS_A_VIS: 'DYAD_VIS_A_VIS',
  L_SHAPE: 'DYAD_L_SHAPE',
  SIDE_BY_SIDE: 'DYAD_SIDE_BY_SIDE',
});

/** Symbolic intent strings (Sprint 2 LLM emits these). */
export const INTENT = Object.freeze({
  INTIMATE: 'intimate',
  COLLABORATIVE: 'collaborative',
  COMPANIONABLE: 'companionable',
  PRESENTING: 'presenting',
});

/**
 * Pedica force-field gain defaults + hysteresis + damping.
 * [INFERRED] = pragmatic default pending paper retrieval.
 */
export const DEFAULT_GAINS = Object.freeze({
  K_COHESION:       1.0,   // [INFERRED] pull toward o-space ideal stance
  K_REPULSION:      2.5,   // [INFERRED] push out of intimate zone
  K_EQUALITY:       0.8,   // [INFERRED] balance distance (no-op for dyad)
  K_FACING:         1.5,   // [INFERRED] yaw torque toward o-space center
  SIGMA_REPULSION:  0.30,  // [INFERRED] decay length (m), intimate-zone falloff
  EPS_HYSTERESIS:   0.15,  // m — formation lock band
  TAU_HYSTERESIS:   0.6,   // s — min dwell before formation switch
  DAMPING_LINEAR:   0.85,  // per-frame velocity damping
  DAMPING_ANGULAR:  0.85,  // per-frame yaw-rate damping
  // Hard cap on per-frame yaw torque so the avatar never snaps; mirrors
  // the convention in avatar-ik.js where joint corrections are clamped
  // to keep frame-to-frame motion legible.
  MAX_YAW_TORQUE:   8.0,   // [INFERRED] rad/s, clamp magnitude
  // Hard cap on per-frame linear force magnitude. Same rationale.
  MAX_FORCE:        12.0,  // [INFERRED] m/s^2-equivalent (consumer scales)
});

/** Hall's proxemic zones, with VR-tuned personal radius per Bönsch 2018. */
export const HALL_ZONES = Object.freeze({
  INTIMATE: 0.45,   // [CANONICAL] Hall 1966
  PERSONAL: 1.00,   // [VR-TUNED]  Bönsch 2018 (~20% < Hall's 1.20m)
  SOCIAL:   3.60,   // [CANONICAL] Hall 1966
  PUBLIC:   7.60,   // [CANONICAL] Hall 1966
});

/** Kendon-derived dyad detection thresholds. */
export const DETECTION = Object.freeze({
  ANGLE_VIS_A_VIS_DEG:        30,    // both within this cone -> face-to-face
  ANGLE_L_SHAPE_MIN_DEG:      60,    // one party angled off the other
  ANGLE_L_SHAPE_MAX_DEG:     120,
  ANGLE_SIDE_BY_SIDE_DOT_TOL: 0.2,   // |1 - dot(fwd_a, fwd_b)| < tol -> parallel
});

// Precomputed radian thresholds (avoids deg→rad per frame).
const _ANGLE_VIS_A_VIS_RAD     = DETECTION.ANGLE_VIS_A_VIS_DEG     * Math.PI / 180;
const _ANGLE_L_SHAPE_MIN_RAD   = DETECTION.ANGLE_L_SHAPE_MIN_DEG   * Math.PI / 180;
const _ANGLE_L_SHAPE_MAX_RAD   = DETECTION.ANGLE_L_SHAPE_MAX_DEG   * Math.PI / 180;

// ─── Internal helpers (pure, no allocations) ──────────────────────────

/**
 * Smallest signed angle delta from `from` to `to` in radians, in (-π, π].
 * @param {number} from
 * @param {number} to
 * @returns {number}
 */
function shortestAngleDelta(from, to) {
  let d = to - from;
  // Normalise into (-π, π] without allocating.
  while (d >  Math.PI) d -= 2 * Math.PI;
  while (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

/**
 * Angle between two 3-vectors projected onto the XZ ground plane.
 * Uses scratch.tmpA / scratch.tmpB so it does not allocate.
 * Returns 0 if either vector has near-zero ground length.
 * @param {THREE.Vector3} a
 * @param {THREE.Vector3} b
 * @param {{tmpA: THREE.Vector3, tmpB: THREE.Vector3}} scratch
 * @returns {number}
 */
function groundAngleBetween(a, b, scratch) {
  scratch.tmpA.set(a.x, 0, a.z);
  scratch.tmpB.set(b.x, 0, b.z);
  const la = scratch.tmpA.length();
  const lb = scratch.tmpB.length();
  if (la < 1e-6 || lb < 1e-6) return 0;
  // Avoid divides — dot / (la*lb), clamped for acos safety.
  const c = Math.max(-1, Math.min(1, scratch.tmpA.dot(scratch.tmpB) / (la * lb)));
  return Math.acos(c);
}

/**
 * Clamp the magnitude of `v` to `max`, in place.
 * @param {THREE.Vector3} v
 * @param {number} max
 */
function clampMagnitude(v, max) {
  const lsq = v.lengthSq();
  const mx2 = max * max;
  if (lsq > mx2 && lsq > 1e-12) {
    v.multiplyScalar(max / Math.sqrt(lsq));
  }
}

/**
 * Map a discrete intent string to the formation it implies.
 * Returns FORMATION.NONE for unknown / null intents.
 * @param {string|null|undefined} intent
 * @returns {string}
 */
export function mapIntentToFormation(intent) {
  switch (intent) {
    case INTENT.INTIMATE:      return FORMATION.VIS_A_VIS;
    case INTENT.COLLABORATIVE: return FORMATION.L_SHAPE;
    case INTENT.COMPANIONABLE: return FORMATION.SIDE_BY_SIDE;
    case INTENT.PRESENTING:    return FORMATION.VIS_A_VIS;
    default:                   return FORMATION.NONE;
  }
}

// ─── Controller ───────────────────────────────────────────────────────

/**
 * Per-frame reactive F-formation controller. One instance per dyad
 * (typically one per active avatar). Maintains persistent state across
 * frames (current formation, hysteresis timer, last o-space).
 */
export class FFormationController {
  /**
   * @param {object} [opts]
   * @param {Partial<typeof DEFAULT_GAINS>} [opts.gains]   gain overrides
   * @param {Partial<typeof HALL_ZONES>}    [opts.zones]   proxemic overrides
   * @param {Partial<typeof DETECTION>}     [opts.detection] threshold overrides
   */
  constructor(opts = {}) {
    this.gains = { ...DEFAULT_GAINS, ...(opts.gains || {}) };
    this.zones = { ...HALL_ZONES,    ...(opts.zones || {}) };
    this.detection = { ...DETECTION, ...(opts.detection || {}) };

    // Effective (post-affect-modulation) copies of the gains. Recomputed
    // every frame inside `update()`; never mutate `this.gains` directly.
    this._eff = { ...this.gains };

    /** @type {{
     *   formation: string,
     *   lastFormation: string,
     *   oCenter: THREE.Vector3,
     *   oRadius: number,
     *   hysteresisTimer: number,
     *   idealPos: THREE.Vector3,
     *   idealYaw: number,
     * }} */
    this._state = {
      formation:       FORMATION.NONE,
      lastFormation:   FORMATION.NONE,
      oCenter:         new THREE.Vector3(),
      oRadius:         this.zones.PERSONAL / 2,
      hysteresisTimer: 0,
      idealPos:        new THREE.Vector3(),
      idealYaw:        0,
    };

    // Pre-allocated scratch storage. update() must never `new` anything.
    this._scratch = {
      // ground-projected user head position
      userGround:      new THREE.Vector3(),
      // self -> user vector and user -> self vector (XZ-flattened)
      selfToUser:      new THREE.Vector3(),
      userToSelf:      new THREE.Vector3(),
      // ground-projected user forward
      userFwdGround:   new THREE.Vector3(),
      // ground-projected self forward
      selfFwdGround:   new THREE.Vector3(),
      // midpoint helper / perpendicular helper / averaged forward
      mid:             new THREE.Vector3(),
      perp:            new THREE.Vector3(),
      avgFwd:          new THREE.Vector3(),
      // accumulated force + scratch direction for repulsion
      force:           new THREE.Vector3(),
      dirAway:         new THREE.Vector3(),
      // formation-local ideal position scratch
      idealPos:        new THREE.Vector3(),
      // tmp pair for groundAngleBetween (cannot alias the outputs above)
      tmpA:            new THREE.Vector3(),
      tmpB:            new THREE.Vector3(),
      // returned-result object: same instance every frame to avoid
      // allocations on the caller's side as well.
      result: {
        force:     new THREE.Vector3(),
        yawTorque: 0,
        oCenter:   new THREE.Vector3(),
        oRadius:   0,
        formation: FORMATION.NONE,
        idealPos:  new THREE.Vector3(),
        idealYaw:  0,
      },
    };
  }

  /**
   * Reset persistent state. Safe to call mid-session (e.g. avatar respawn).
   */
  reset() {
    this._state.formation       = FORMATION.NONE;
    this._state.lastFormation   = FORMATION.NONE;
    this._state.oCenter.set(0, 0, 0);
    this._state.oRadius         = this.zones.PERSONAL / 2;
    this._state.hysteresisTimer = 0;
    this._state.idealPos.set(0, 0, 0);
    this._state.idealYaw        = 0;
  }

  /**
   * Per-frame update. Inputs are read but never mutated; outputs are
   * written into a stable scratch result object (do NOT retain the
   * returned reference across frames — copy out what you need).
   *
   * Defensive: missing or malformed inputs cause a zero-result return
   * with the previous formation preserved. The method never throws.
   *
   * @param {object} ctx
   * @param {{pos: THREE.Vector3, yaw: number, fwd: THREE.Vector3}} ctx.self
   *   The avatar's current pose. `fwd` should be its forward direction
   *   in world space (XZ component is used; Y is ignored).
   * @param {{headPos: THREE.Vector3, headFwd: THREE.Vector3}} ctx.user
   *   The user's head pose in world space.
   * @param {{pos: THREE.Vector3}|null} [ctx.anchor]
   *   Optional shared-attention anchor (e.g. a 3D panel). When present,
   *   the o-space center is biased 40% toward it.
   * @param {{presence: number, flow: number, temperature: number, resonance: number}} [ctx.affect]
   *   Affect scalars. `presence` and `resonance` are clamped to [0,1];
   *   `flow` is treated as [-1, +1] (negative dampens cohesion);
   *   `temperature` is reserved for callers (no gain effect here).
   * @param {string|null} [ctx.intent]
   *   Symbolic intent ('intimate' | 'collaborative' | 'companionable' |
   *   'presenting'). Ignored when `intentOverride` is false.
   * @param {boolean} [ctx.intentOverride]
   *   When true, intent dictates the formation and detection/hysteresis
   *   are bypassed for this frame.
   * @param {number} dt   seconds since previous update (clamped to [0, 0.25])
   * @returns {{
   *   force: THREE.Vector3,
   *   yawTorque: number,
   *   oCenter: THREE.Vector3,
   *   oRadius: number,
   *   formation: string,
   *   idealPos: THREE.Vector3,
   *   idealYaw: number,
   * }} reactive control output (scratch — do not retain)
   */
  update(ctx, dt) {
    const out = this._scratch.result;

    // ── Input validation (defensive — never throw out of update()) ──
    if (!ctx || !ctx.self || !ctx.user
        || !ctx.self.pos || !ctx.self.fwd
        || !ctx.user.headPos || !ctx.user.headFwd
        || !Number.isFinite(ctx.self.yaw)) {
      console.warn('[fformation] update(): malformed ctx, returning zero result');
      return this._zeroResult(out);
    }
    // Clamp dt so a paused tab returning a 5-second frame doesn't
    // catapult the hysteresis timer.
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > 0.25) dt = 0.25;

    const s = this._scratch;
    const self = ctx.self;
    const user = ctx.user;

    // Ground-project user head + forwards (XZ plane only).
    s.userGround.set(user.headPos.x, self.pos.y, user.headPos.z);
    s.userFwdGround.set(user.headFwd.x, 0, user.headFwd.z);
    if (s.userFwdGround.lengthSq() < 1e-8) {
      // Fall back to vector from user toward self if user fwd is degenerate.
      s.userFwdGround.set(self.pos.x - user.headPos.x, 0, self.pos.z - user.headPos.z);
      if (s.userFwdGround.lengthSq() < 1e-8) s.userFwdGround.set(0, 0, 1);
    }
    s.userFwdGround.normalize();

    s.selfFwdGround.set(self.fwd.x, 0, self.fwd.z);
    if (s.selfFwdGround.lengthSq() < 1e-8) {
      s.selfFwdGround.set(Math.sin(self.yaw), 0, Math.cos(self.yaw));
    }
    s.selfFwdGround.normalize();

    s.selfToUser.subVectors(s.userGround, self.pos); s.selfToUser.y = 0;
    s.userToSelf.subVectors(self.pos, s.userGround); s.userToSelf.y = 0;

    // ── 1. Affect-modulated gain shaping ──
    this._modulateGains(ctx.affect);

    // ── 2. Detect desired formation (or take intent override) ──
    const detected = this._detectFormation(s);
    const intentOverride = !!ctx.intentOverride && ctx.intent != null;
    const desired = intentOverride
      ? mapIntentToFormation(ctx.intent)
      : detected;

    // ── 3. Hysteresis lock ──
    if (intentOverride) {
      this._state.formation       = desired;
      this._state.lastFormation   = desired;
      this._state.hysteresisTimer = 0;
    } else if (desired !== this._state.lastFormation) {
      this._state.hysteresisTimer += dt;
      if (this._state.hysteresisTimer > this.gains.TAU_HYSTERESIS) {
        this._state.formation       = desired;
        this._state.lastFormation   = desired;
        this._state.hysteresisTimer = 0;
      }
    } else {
      this._state.hysteresisTimer = 0;
    }
    const formation = this._state.formation;

    // ── 4. O-space center + radius ──
    this._computeOSpace(formation, ctx, s);
    if (ctx.anchor && ctx.anchor.pos) {
      // Bias 40% toward shared-attention anchor.
      this._state.oCenter.lerp(ctx.anchor.pos, 0.4);
    }

    // ── 5. Ideal stance for self in this formation ──
    this._computeIdealStance(formation, ctx, s);

    // ── 6. Force summation: cohesion to ideal + repulsion from user ──
    s.force.set(0, 0, 0);
    // Cohesion: pull toward ideal stance.
    s.force.x += (this._state.idealPos.x - self.pos.x) * this._eff.K_COHESION;
    s.force.z += (this._state.idealPos.z - self.pos.z) * this._eff.K_COHESION;
    // Repulsion: exponential falloff inside personal zone.
    const d = Math.sqrt(s.selfToUser.x * s.selfToUser.x + s.selfToUser.z * s.selfToUser.z);
    if (d < this.zones.PERSONAL && d > 1e-6) {
      // dirAway = normalize(userToSelf)
      s.dirAway.set(s.userToSelf.x / d, 0, s.userToSelf.z / d);
      const falloff = Math.exp(-(d - this.zones.INTIMATE) / this._eff.SIGMA_REPULSION);
      const mag = this._eff.K_REPULSION * falloff;
      s.force.x += s.dirAway.x * mag;
      s.force.z += s.dirAway.z * mag;
    }
    // (Equality term is no-op for a dyad — placeholder for future
    // multi-party expansion; intentionally left out of the hot path.)

    clampMagnitude(s.force, this._eff.MAX_FORCE);

    // ── 7. Yaw torque: rotate self to face o-space center ──
    const desiredYaw = Math.atan2(
      this._state.oCenter.x - self.pos.x,
      this._state.oCenter.z - self.pos.z,
    );
    let yawTorque = this._eff.K_FACING * shortestAngleDelta(self.yaw, desiredYaw);
    if (yawTorque >  this._eff.MAX_YAW_TORQUE) yawTorque =  this._eff.MAX_YAW_TORQUE;
    if (yawTorque < -this._eff.MAX_YAW_TORQUE) yawTorque = -this._eff.MAX_YAW_TORQUE;
    this._state.idealYaw = desiredYaw;

    // ── 8. Pack the stable result object ──
    out.force.copy(s.force);
    out.yawTorque = yawTorque;
    out.oCenter.copy(this._state.oCenter);
    out.oRadius   = this._state.oRadius;
    out.formation = formation;
    out.idealPos.copy(this._state.idealPos);
    out.idealYaw  = desiredYaw;
    return out;
  }

  /**
   * Debug introspection for the avatar-pose-harness testbench. Returns
   * a fresh plain object snapshot of the controller's internal state.
   * NOT on the hot path — allocates freely; do not call from update().
   * @returns {object}
   */
  _dbg() {
    return {
      formation:       this._state.formation,
      lastFormation:   this._state.lastFormation,
      oCenter:         this._state.oCenter.toArray(),
      oRadius:         this._state.oRadius,
      hysteresisTimer: this._state.hysteresisTimer,
      idealPos:        this._state.idealPos.toArray(),
      idealYaw:        this._state.idealYaw,
      effectiveGains:  { ...this._eff },
      zones:           { ...this.zones },
    };
  }

  // ─── Private helpers ─────────────────────────────────────────────

  /**
   * Recompute `_eff` (effective gains) from base `gains` and current affect.
   * Always writes every field so stale modulations from a previous frame
   * never bleed through.
   * @param {{presence?: number, flow?: number, temperature?: number, resonance?: number}|undefined} affect
   */
  _modulateGains(affect) {
    const g = this.gains;
    // Default to neutral affect when caller omits the field.
    const presence  = _clamp01(affect?.presence,  0.5);
    const resonance = _clamp01(affect?.resonance, 0.5);
    // flow is [-1, +1]; default 0.
    const flowRaw   = (affect && Number.isFinite(affect.flow)) ? affect.flow : 0;
    const flow      = Math.max(-1, Math.min(1, flowRaw));

    // Start from canonical gains.
    this._eff.K_COHESION      = g.K_COHESION;
    this._eff.K_REPULSION     = g.K_REPULSION;
    this._eff.K_EQUALITY      = g.K_EQUALITY;
    this._eff.K_FACING        = g.K_FACING;
    this._eff.SIGMA_REPULSION = g.SIGMA_REPULSION;
    this._eff.MAX_FORCE       = g.MAX_FORCE;
    this._eff.MAX_YAW_TORQUE  = g.MAX_YAW_TORQUE;

    // Low presence -> stronger personal-space defense.
    this._eff.K_REPULSION    *= 1 + 0.5 * (1 - presence);
    // High flow -> tighter cohesion; negative flow loosens it (floored at 0.3).
    this._eff.K_COHESION     *= Math.max(0.3, 1.0 + 0.4 * flow);

    // Resonance widens o-space (consumed in _computeOSpace).
    this._eff._oRadiusScale   = 1 + 0.3 * (1 - resonance);
    // temperature: reserved (drives external timing/urgency, not gains).
  }

  /**
   * Detect the current dyadic formation from geometric inputs. Returns
   * FORMATION.NONE when no F-formation is plausible (too far, or angles
   * fail every taxonomy bin).
   * @param {object} s   controller scratch
   * @returns {string}
   */
  _detectFormation(s) {
    // Distance on the ground.
    const dx = s.selfToUser.x, dz = s.selfToUser.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist > this.zones.SOCIAL) return FORMATION.NONE;

    // Angle from self's forward to the vector toward the user.
    const angleSelf = groundAngleBetween(s.selfFwdGround, s.selfToUser, s);
    // Angle from user's forward to the vector toward self.
    const angleUser = groundAngleBetween(s.userFwdGround, s.userToSelf, s);

    if (angleSelf < _ANGLE_VIS_A_VIS_RAD && angleUser < _ANGLE_VIS_A_VIS_RAD) {
      return FORMATION.VIS_A_VIS;
    }
    if (angleSelf > _ANGLE_L_SHAPE_MIN_RAD && angleSelf < _ANGLE_L_SHAPE_MAX_RAD) {
      return FORMATION.L_SHAPE;
    }
    // Side-by-side: forwards approximately parallel (dot ≈ 1).
    const fwdDot = s.selfFwdGround.x * s.userFwdGround.x + s.selfFwdGround.z * s.userFwdGround.z;
    if (Math.abs(1 - fwdDot) < this.detection.ANGLE_SIDE_BY_SIDE_DOT_TOL) {
      return FORMATION.SIDE_BY_SIDE;
    }
    return FORMATION.NONE;
  }

  /**
   * Write the active o-space center + radius into `this._state`.
   * @param {string} formation
   * @param {object} ctx
   * @param {object} s   controller scratch
   */
  _computeOSpace(formation, ctx, s) {
    const self = ctx.self;
    const baseRadius = this.zones.PERSONAL / 2;
    const scale = this._eff._oRadiusScale ?? 1;

    switch (formation) {
      case FORMATION.VIS_A_VIS: {
        // Midpoint(self, user_ground)
        this._state.oCenter.set(
          (self.pos.x + s.userGround.x) * 0.5,
          self.pos.y,
          (self.pos.z + s.userGround.z) * 0.5,
        );
        this._state.oRadius = baseRadius * scale;
        return;
      }
      case FORMATION.L_SHAPE: {
        // m = midpoint; n = perpendicular to user->self direction (XZ).
        s.mid.set(
          (self.pos.x + s.userGround.x) * 0.5,
          self.pos.y,
          (self.pos.z + s.userGround.z) * 0.5,
        );
        // perp_xz of (dx, dz) is (-dz, dx).
        const dx = s.userGround.x - self.pos.x;
        const dz = s.userGround.z - self.pos.z;
        const plen = Math.sqrt(dx * dx + dz * dz);
        if (plen > 1e-6) {
          s.perp.set(-dz / plen, 0, dx / plen);
        } else {
          s.perp.set(1, 0, 0);
        }
        // Offset midpoint by 0.45m along the perpendicular. Sprint 1 has
        // no side-preference signal from intent/anchor yet, so we pick
        // the side that puts the o-space CLOSER to the avatar's current
        // forward — feels less like an arbitrary teleport.
        const sgn = (s.perp.x * s.selfFwdGround.x + s.perp.z * s.selfFwdGround.z) >= 0 ? 1 : -1;
        this._state.oCenter.set(
          s.mid.x + s.perp.x * 0.45 * sgn,
          s.mid.y,
          s.mid.z + s.perp.z * 0.45 * sgn,
        );
        this._state.oRadius = baseRadius * scale;
        return;
      }
      case FORMATION.SIDE_BY_SIDE: {
        // Average forwards (already ground-projected + unit-length).
        s.avgFwd.set(
          (s.selfFwdGround.x + s.userFwdGround.x) * 0.5,
          0,
          (s.selfFwdGround.z + s.userFwdGround.z) * 0.5,
        );
        if (s.avgFwd.lengthSq() < 1e-8) {
          s.avgFwd.copy(s.selfFwdGround);
        } else {
          s.avgFwd.normalize();
        }
        const midX = (self.pos.x + s.userGround.x) * 0.5;
        const midZ = (self.pos.z + s.userGround.z) * 0.5;
        this._state.oCenter.set(
          midX + s.avgFwd.x * this.zones.PERSONAL,
          self.pos.y,
          midZ + s.avgFwd.z * this.zones.PERSONAL,
        );
        this._state.oRadius = baseRadius * scale;
        return;
      }
      case FORMATION.NONE:
      default: {
        // No active formation — o-space collapses to "right here," widened
        // so the cohesion term doesn't tug us around.
        this._state.oCenter.set(self.pos.x, self.pos.y, self.pos.z);
        this._state.oRadius = this.zones.PERSONAL * scale;
        return;
      }
    }
  }

  /**
   * Write the avatar's ideal stance position for this formation into
   * `this._state.idealPos`. The ideal sits on the o-space perimeter on
   * the side opposite to the user (or wherever is most natural).
   * @param {string} formation
   * @param {object} ctx
   * @param {object} s   controller scratch
   */
  _computeIdealStance(formation, ctx, s) {
    const self = ctx.self;
    const oc = this._state.oCenter;
    const r  = this._state.oRadius;

    switch (formation) {
      case FORMATION.VIS_A_VIS: {
        // Ideal stance: opposite the user across the o-center, at radius r.
        const dx = oc.x - s.userGround.x;
        const dz = oc.z - s.userGround.z;
        const len = Math.sqrt(dx * dx + dz * dz);
        if (len > 1e-6) {
          this._state.idealPos.set(
            oc.x + (dx / len) * r,
            self.pos.y,
            oc.z + (dz / len) * r,
          );
        } else {
          this._state.idealPos.set(oc.x, self.pos.y, oc.z + r);
        }
        return;
      }
      case FORMATION.L_SHAPE: {
        // Stand at the o-center offset along the line FROM user TO o-center,
        // at radius r. Produces a right-angle stance with the user.
        const dx = oc.x - s.userGround.x;
        const dz = oc.z - s.userGround.z;
        const len = Math.sqrt(dx * dx + dz * dz);
        if (len > 1e-6) {
          this._state.idealPos.set(
            oc.x + (dx / len) * r,
            self.pos.y,
            oc.z + (dz / len) * r,
          );
        } else {
          this._state.idealPos.set(oc.x + r, self.pos.y, oc.z);
        }
        return;
      }
      case FORMATION.SIDE_BY_SIDE: {
        // Stand level with user but offset laterally so both face the
        // o-center forward. Lateral offset = PERSONAL/2 to the LEFT of
        // user (arbitrary side; collaborative wrapper can mirror).
        const fwdX = oc.x - ((self.pos.x + s.userGround.x) * 0.5);
        const fwdZ = oc.z - ((self.pos.z + s.userGround.z) * 0.5);
        const flen = Math.sqrt(fwdX * fwdX + fwdZ * fwdZ);
        let nfx = 1, nfz = 0;
        if (flen > 1e-6) { nfx = fwdX / flen; nfz = fwdZ / flen; }
        // Left-perp of forward (XZ): (-fz, fx).
        const lateralX = -nfz;
        const lateralZ =  nfx;
        const lateralOffset = this.zones.PERSONAL * 0.5;
        this._state.idealPos.set(
          s.userGround.x + lateralX * lateralOffset,
          self.pos.y,
          s.userGround.z + lateralZ * lateralOffset,
        );
        return;
      }
      case FORMATION.NONE:
      default: {
        // No formation -> the ideal IS where we already are (no pull).
        this._state.idealPos.set(self.pos.x, self.pos.y, self.pos.z);
        return;
      }
    }
  }

  /**
   * Populate the result struct with a zero-force / preserved-formation
   * snapshot. Used by the defensive guard path; the formation enum is
   * preserved so a one-frame bad input doesn't drop the F-formation lock.
   * @param {object} out   `_scratch.result`
   * @returns {object}
   */
  _zeroResult(out) {
    out.force.set(0, 0, 0);
    out.yawTorque = 0;
    out.oCenter.copy(this._state.oCenter);
    out.oRadius   = this._state.oRadius;
    out.formation = this._state.formation;
    out.idealPos.copy(this._state.idealPos);
    out.idealYaw  = this._state.idealYaw;
    return out;
  }
}

// ─── Integrator helper ────────────────────────────────────────────────

/**
 * Apply a controller output to a simple kinematic body. Critically-damped
 * semi-implicit Euler — caller passes a body with `{pos, vel, yaw, yawVel}`
 * (THREE.Vector3 for pos / vel) and the controller's `update()` return
 * value.
 *
 * This is the canonical seam for the locomotion primitive; production code
 * may swap in a fuller character-controller while preserving the contract.
 *
 * @param {{pos: THREE.Vector3, vel: THREE.Vector3, yaw: number, yawVel: number}} body
 * @param {{force: THREE.Vector3, yawTorque: number}} control
 * @param {number} dt
 * @param {number} [damping]   per-frame velocity damping (0..1)
 * @param {number} [angularDamping]   per-frame yaw-rate damping (0..1)
 */
export function applyForceToAvatar(
  body,
  control,
  dt,
  damping = DEFAULT_GAINS.DAMPING_LINEAR,
  angularDamping = DEFAULT_GAINS.DAMPING_ANGULAR,
) {
  if (!body || !control || !Number.isFinite(dt) || dt <= 0) return;
  if (!body.pos || !body.vel) return;

  // Linear: vel += F * dt; vel *= damping; pos += vel * dt.
  body.vel.x += control.force.x * dt;
  body.vel.z += control.force.z * dt;
  body.vel.x *= damping;
  body.vel.z *= damping;
  body.pos.x += body.vel.x * dt;
  body.pos.z += body.vel.z * dt;

  // Angular: yawVel += torque * dt; damp; integrate.
  if (Number.isFinite(body.yawVel) && Number.isFinite(body.yaw)) {
    body.yawVel += (control.yawTorque || 0) * dt;
    body.yawVel *= angularDamping;
    body.yaw    += body.yawVel * dt;
    // Wrap yaw into (-π, π] to keep downstream math stable.
    while (body.yaw >  Math.PI) body.yaw -= 2 * Math.PI;
    while (body.yaw <= -Math.PI) body.yaw += 2 * Math.PI;
  }
}

// ─── Local helpers ────────────────────────────────────────────────────

/**
 * Clamp `v` to [0, 1]; substitute `fallback` if `v` is not a finite number.
 * @param {number|undefined|null} v
 * @param {number} fallback
 * @returns {number}
 */
function _clamp01(v, fallback) {
  if (!Number.isFinite(v)) return fallback;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}
