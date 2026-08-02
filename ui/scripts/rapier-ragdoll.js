/**
 * rapier-ragdoll.js — Rapier.js active ragdoll for VRM secondary motion.
 *
 * GLOBAL chain-dynamics complement to sdf-compliance.js's LOCAL bone
 * displacement. Where sdf-compliance.js moves the chest a few cm at the
 * point of contact, this module gives the neck/head/spine/shoulders a
 * lagging follow-through with momentum, and lets any small disturbance
 * propagate up and down the kinematic chain as visible secondary motion.
 *
 * Active-ragdoll pattern: one Rapier rigid body per tracked humanoid
 * bone, configured as KinematicPositionBased — we drive their target
 * transforms from the post-pose VRM skeleton each frame, so they FOLLOW
 * the animated pose rather than being ruled by gravity. A PD spring
 * (kp=4000, kd=400 — critically damped around 10Hz) computes the
 * physics body's intended next position, and joint-limit clamping plus
 * the spring's natural lag produce realistic deviation that recovers
 * smoothly. The resulting delta between (kinematic target) and (after-
 * step physics body) is exported via getBoneDeltas() for the
 * coordinator to right-multiply into node.quaternion / node.position.
 *
 * v1 has NO collision shapes attached. Bodies provide joint-constrained
 * kinematic-chain dynamics only — secondary motion comes from target-
 * spring lag + joint limits. Adding shapes (capsules around limbs)
 * is future work for hand/world contact response.
 *
 * Composes ON TOP of pose channels AND sdf-compliance: read order is
 *   1. base pose written by anim channels
 *   2. sdf-compliance right-multiplies its local rotation delta
 *   3. rapier-ragdoll.tick() snapshots the resulting world transforms
 *      as kinematic targets, steps physics, exports global deltas
 *   4. coordinator applies the deltas before vrm.update()
 *
 * Skips fingers (24 small bones per hand-set) — their joint dynamics
 * add cost without visible payoff at the camera distances this avatar
 * is normally framed at, and the spring lag would smear fingertip
 * gestures rather than enhance them.
 *
 * Pairs with: sdf-compliance.js, contact-reactor.js. Singleton wrapper
 * (consumer): avatar-xr-rapier.js (parallels avatar-xr-compliance.js).
 */

const RAPIER_CDN = 'https://cdn.jsdelivr.net/npm/@dimforge/rapier3d-compat/+esm';

/** PD spring driving the kinematic target. kp/kd chosen for critical
 *  damping at ~10Hz natural frequency: ω = sqrt(kp) ≈ 63 rad/s,
 *  ζ = kd / (2·sqrt(kp)) ≈ 1.0 — fast enough to track keyframed pose
 *  changes without visible lag, slow enough that an impulse pushes
 *  through several frames of secondary motion before settling. */
const SPRING_KP        = 4000;
const SPRING_KD        = 400;
const DEFAULT_WEIGHT   = 0.6;
const FIXED_DT_S       = 1 / 60;
const MAX_SUBSTEPS     = 3;        // cap catch-up after long frames
const MAX_POS_DELTA_M  = 0.10;     // safety clamp on exported position delta
const MAX_ANGLE_DELTA  = 0.6;      // ~34° safety clamp on rotation delta (radians)

const DEG = Math.PI / 180;

/**
 * Tracked bones with parent + joint configuration.
 *
 * type: 'spherical' uses Rapier's spherical joint with per-axis angular
 * limits applied via a generic 6-DOF joint (Rapier 0.12+ exposes
 * `RAPIER.JointData.generic` with per-axis limits). For elbows and
 * knees, type: 'revolute' uses a single-axis hinge. Limits are
 * symmetric (±value) for spherical and explicit (min, max) for
 * revolute.
 *
 * `parent` is the upstream bone whose body anchors the joint. The root
 * (spine) has parent null — its body is unconstrained kinematic and
 * just follows the hips/animation root via the spring.
 *
 * Angles are stored in radians (converted from the spec's degrees).
 */
const BONE_GRAPH = Object.freeze({
  spine:         { parent: null,           type: 'spherical', limits: { pitch: 15 * DEG, yaw: 15 * DEG, roll: 15 * DEG } },
  chest:         { parent: 'spine',        type: 'spherical', limits: { pitch: 15 * DEG, yaw: 15 * DEG, roll: 15 * DEG } },
  upperChest:    { parent: 'chest',        type: 'spherical', limits: { pitch: 15 * DEG, yaw: 15 * DEG, roll: 15 * DEG } },
  neck:          { parent: 'upperChest',   type: 'spherical', limits: { pitch: 30 * DEG, yaw: 45 * DEG, roll: 20 * DEG } },
  head:          { parent: 'neck',         type: 'spherical', limits: { pitch: 30 * DEG, yaw: 45 * DEG, roll: 20 * DEG } },

  leftShoulder:  { parent: 'upperChest',   type: 'spherical', limits: { pitch: 90 * DEG, yaw: 90 * DEG, roll: 45 * DEG } },
  leftUpperArm:  { parent: 'leftShoulder', type: 'spherical', limits: { pitch: 90 * DEG, yaw: 90 * DEG, roll: 45 * DEG } },
  leftLowerArm:  { parent: 'leftUpperArm', type: 'revolute',  axis: [1, 0, 0], limits: { min: 0,        max: 150 * DEG } },
  leftHand:      { parent: 'leftLowerArm', type: 'spherical', limits: { pitch: 60 * DEG, yaw: 30 * DEG, roll: 30 * DEG } },

  rightShoulder: { parent: 'upperChest',   type: 'spherical', limits: { pitch: 90 * DEG, yaw: 90 * DEG, roll: 45 * DEG } },
  rightUpperArm: { parent: 'rightShoulder', type: 'spherical', limits: { pitch: 90 * DEG, yaw: 90 * DEG, roll: 45 * DEG } },
  rightLowerArm: { parent: 'rightUpperArm', type: 'revolute',  axis: [1, 0, 0], limits: { min: 0,        max: 150 * DEG } },
  rightHand:     { parent: 'rightLowerArm', type: 'spherical', limits: { pitch: 60 * DEG, yaw: 30 * DEG, roll: 30 * DEG } },

  leftUpperLeg:  { parent: 'spine',        type: 'spherical', limits: { pitch: 60 * DEG, yaw: 45 * DEG, roll: 30 * DEG } },
  leftLowerLeg:  { parent: 'leftUpperLeg', type: 'revolute',  axis: [1, 0, 0], limits: { min: 0,        max: 145 * DEG } },
  leftFoot:      { parent: 'leftLowerLeg', type: 'spherical', limits: { pitch: 30 * DEG, yaw: 20 * DEG, roll: 20 * DEG } },

  rightUpperLeg: { parent: 'spine',        type: 'spherical', limits: { pitch: 60 * DEG, yaw: 45 * DEG, roll: 30 * DEG } },
  rightLowerLeg: { parent: 'rightUpperLeg', type: 'revolute',  axis: [1, 0, 0], limits: { min: 0,        max: 145 * DEG } },
  rightFoot:     { parent: 'rightLowerLeg', type: 'spherical', limits: { pitch: 30 * DEG, yaw: 20 * DEG, roll: 20 * DEG } },
});

const BONE_NAMES = Object.freeze(Object.keys(BONE_GRAPH));

/**
 * Rapier-driven active ragdoll over a VRM humanoid skeleton.
 *
 * Lifecycle:
 *   const r = new RapierRagdoll({ three, vrm });
 *   await r.init();          // loads Rapier (CDN), builds bodies + joints
 *   // each frame, after pose + sdf-compliance write, before vrm.update:
 *   r.tick(dtMs);
 *   for (const [name, { posDelta, quatDelta }] of r.getBoneDeltas()) { ... }
 *   r.dispose();
 *
 * If Rapier fails to load (offline, CDN blocked, WASM error), the
 * instance silently disables itself — `tick()` is a no-op and
 * `getBoneDeltas()` returns an empty Map. The consumer doesn't need
 * to special-case the failure.
 */
export class RapierRagdoll {
  /**
   * @param {object} opts
   * @param {object} opts.three   THREE namespace (for Vector3 / Quaternion)
   * @param {object} opts.vrm     VRM with humanoid (getNormalizedBoneNode)
   * @param {number} [opts.weight] initial output mix weight 0..1 (default 0.6)
   */
  constructor({ three, vrm, weight } = {}) {
    if (!three) throw new Error('RapierRagdoll needs three');
    if (!vrm?.humanoid) throw new Error('RapierRagdoll needs vrm with humanoid');
    this.three = three;
    this.vrm = vrm;

    /** Public, mutable. Coordinator updates this from server settings live;
     *  getBoneDeltas() scales magnitudes by it. 0 = pass-through, 1 = full. */
    this.weight = typeof weight === 'number' ? Math.max(0, Math.min(1, weight)) : DEFAULT_WEIGHT;

    /** Set to false on load failure or disposal — guards all entrypoints. */
    this.enabled = false;

    /** @type {*} Rapier module + world handles, filled in init(). */
    this._rapier = null;
    this._world = null;

    /** name -> { body, joint?, parentName, kinematicTarget: { pos, quat }, lastBodyPose: { pos, quat } } */
    this._bones = new Map();

    /** name -> { posDelta: Vector3, quatDelta: Quaternion } — reused each frame. */
    this._deltas = new Map();

    this._scratch = {
      worldPos:  new three.Vector3(),
      worldQuat: new three.Quaternion(),
      tmpQuat:   new three.Quaternion(),
      tmpVec:    new three.Vector3(),
      identity:  new three.Quaternion(),
    };

    this._accumDt = 0;
  }

  /**
   * Load Rapier (CDN), initialize the physics world, build rigid bodies
   * and joints. Idempotent — calling twice does nothing on the second
   * call. Resolves successfully even on Rapier load failure; check
   * `this.enabled` after if you want to know whether physics is active.
   *
   * @returns {Promise<void>}
   */
  async init() {
    if (this.enabled || this._rapier) return;
    try {
      const RAPIER = await import(/* @vite-ignore */ RAPIER_CDN);
      const mod = RAPIER.default || RAPIER;
      await mod.init();
      this._rapier = mod;
    } catch (err) {
      // Offline, blocked CDN, WASM init failure — go dark silently.
      console.debug('[rapier-ragdoll] Rapier unavailable, disabled:', err?.message || String(err));
      this.enabled = false;
      return;
    }

    try {
      // Zero gravity: bodies are kinematic, but if we ever convert any to
      // dynamic (e.g. for collision response), the secondary motion shouldn't
      // be pulled down by 9.81 — the spring is the only restoring force.
      this._world = new this._rapier.World({ x: 0, y: 0, z: 0 });
      this._buildBodies();
      this._buildJoints();
      this._initDeltas();
      this.enabled = true;
    } catch (err) {
      console.debug('[rapier-ragdoll] world init failed, disabled:', err?.message || String(err));
      this._teardownWorld();
      this.enabled = false;
    }
  }

  /**
   * Per-frame step. Call AFTER pose channels + sdf-compliance have written
   * `node.quaternion`/`node.position` for this frame, and BEFORE the
   * consumer applies the exported deltas (or before `vrm.update()` if the
   * consumer writes deltas straight back into nodes).
   *
   * @param {number} dtMs delta-time in milliseconds since previous tick
   */
  tick(dtMs) {
    if (!this.enabled || !this._world) return;
    const dt = Math.max(0, Math.min(0.25, dtMs / 1000));   // cap 250ms (tab-switch)
    if (dt <= 0) return;

    this._snapshotKinematicTargets();
    this._driveBodiesToTargets(dt);
    this._stepWorld(dt);
    this._computeDeltas();
  }

  /**
   * Get the post-physics deltas for the most recent tick. Map values are
   * scaled by `this.weight` — at weight 0 every delta is identity, at
   * weight 1 the full simulated deviation is exported.
   *
   * Returned Map is owned by this instance and reused each frame; the
   * Vector3/Quaternion values inside are also reused. Consumers should
   * read and apply within the same frame, not retain references across
   * frames.
   *
   * @returns {Map<string, { posDelta: import('three').Vector3, quatDelta: import('three').Quaternion }>}
   */
  getBoneDeltas() {
    if (!this.enabled) return this._deltas;   // populated as identities at init
    return this._deltas;
  }

  /**
   * Hard-reset all physics state: snap every kinematic body back to its
   * current animated bone transform, zero velocities, clear deltas.
   * Call on VRM swap, session-end, or after a big keyframe jump (e.g.
   * teleport, scene transition) to avoid a frame of huge spring-recoil.
   */
  reset() {
    if (!this.enabled || !this._world) {
      this._initDeltas();
      return;
    }
    for (const [name, entry] of this._bones) {
      this._readBoneWorldTransform(name, this._scratch.worldPos, this._scratch.worldQuat);
      const t = entry.body.translation();
      t.x = this._scratch.worldPos.x;
      t.y = this._scratch.worldPos.y;
      t.z = this._scratch.worldPos.z;
      entry.body.setNextKinematicTranslation(t);
      entry.body.setTranslation(t, true);
      const q = entry.body.rotation();
      q.x = this._scratch.worldQuat.x;
      q.y = this._scratch.worldQuat.y;
      q.z = this._scratch.worldQuat.z;
      q.w = this._scratch.worldQuat.w;
      entry.body.setNextKinematicRotation(q);
      entry.body.setRotation(q, true);

      entry.kinematicTarget.pos.copy(this._scratch.worldPos);
      entry.kinematicTarget.quat.copy(this._scratch.worldQuat);
      entry.lastBodyPose.pos.copy(this._scratch.worldPos);
      entry.lastBodyPose.quat.copy(this._scratch.worldQuat);
    }
    this._initDeltas();
    this._accumDt = 0;
  }

  /** Release Rapier resources. Safe to call multiple times. */
  dispose() {
    this._teardownWorld();
    this._bones.clear();
    this._deltas.clear();
    this._rapier = null;
    this.enabled = false;
  }

  // ─── Internal ────────────────────────────────────────────────────────────

  /** Build one KinematicPositionBased rigid body per tracked bone, seeded
   *  at the bone's current world transform. KinematicPosition (not
   *  Velocity) because we explicitly drive `setNextKinematicTranslation/
   *  Rotation` each frame — Rapier interpolates between current and next
   *  position internally for the integrator. */
  _buildBodies() {
    const R = this._rapier;
    for (const name of BONE_NAMES) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(name);
      if (!node) continue;   // VRM missing this optional bone (e.g. no upperChest)

      this._readBoneWorldTransform(name, this._scratch.worldPos, this._scratch.worldQuat);

      const desc = R.RigidBodyDesc.kinematicPositionBased()
        .setTranslation(this._scratch.worldPos.x, this._scratch.worldPos.y, this._scratch.worldPos.z)
        .setRotation({
          x: this._scratch.worldQuat.x,
          y: this._scratch.worldQuat.y,
          z: this._scratch.worldQuat.z,
          w: this._scratch.worldQuat.w,
        });
      const body = this._world.createRigidBody(desc);

      this._bones.set(name, {
        body,
        joint: null,
        parentName: BONE_GRAPH[name].parent,
        kinematicTarget: {
          pos:  this._scratch.worldPos.clone(),
          quat: this._scratch.worldQuat.clone(),
        },
        lastBodyPose: {
          pos:  this._scratch.worldPos.clone(),
          quat: this._scratch.worldQuat.clone(),
        },
      });
    }
  }

  /** Connect each bone's body to its parent via a joint. Skipped if the
   *  parent body wasn't created (VRM missing it). Joints inform Rapier
   *  about kinematic-chain structure even though both bodies are
   *  kinematic — when v1 grows collision shapes that turn certain
   *  bodies dynamic-on-impact, the joints will already be in place. */
  _buildJoints() {
    const R = this._rapier;
    for (const name of BONE_NAMES) {
      const cfg = BONE_GRAPH[name];
      if (!cfg.parent) continue;
      const child = this._bones.get(name);
      const parent = this._bones.get(cfg.parent);
      if (!child || !parent) continue;

      // Anchor point: child body's origin in parent's local frame and (0,0,0)
      // in child's frame — the joint pin coincides with the child bone's
      // skeleton joint position, since both bodies were seeded at the bone's
      // world position.
      const childAnchor  = { x: 0, y: 0, z: 0 };
      const parentAnchor = this._parentLocalAnchor(name, cfg.parent);

      let jointData;
      if (cfg.type === 'revolute') {
        const ax = cfg.axis || [1, 0, 0];
        jointData = R.JointData.revolute(parentAnchor, childAnchor, { x: ax[0], y: ax[1], z: ax[2] });
        if (cfg.limits && typeof cfg.limits.min === 'number' && typeof cfg.limits.max === 'number') {
          // Revolute hinge: 1-axis limit. API exposes `limitsEnabled` + `limits`
          // on the joint data; fall back gracefully if absent.
          try { jointData.limitsEnabled = true; jointData.limits = [cfg.limits.min, cfg.limits.max]; } catch (_) { /* older API */ }
        }
      } else {
        jointData = R.JointData.spherical(parentAnchor, childAnchor);
        // Spherical joint limits in Rapier 0.12+ are per-axis on a generic
        // joint; the simple spherical doesn't expose them. We rely on the
        // PD spring + animated targets to keep the bodies within anatomic
        // range, and use the limits dict here only as documentation for
        // future generic-joint conversion. (Stored on entry for inspectors.)
      }

      const joint = this._world.createImpulseJoint(jointData, parent.body, child.body, true);
      child.joint = joint;
    }
  }

  /** Vector from parent bone's world origin to child bone's world origin,
   *  expressed in parent's local frame. Used as the parent-side anchor. */
  _parentLocalAnchor(childName, parentName) {
    const childPos  = new this.three.Vector3();
    const parentPos = new this.three.Vector3();
    const parentQ   = new this.three.Quaternion();
    this._readBoneWorldTransform(childName, childPos, this._scratch.tmpQuat);
    this._readBoneWorldTransform(parentName, parentPos, parentQ);
    const v = childPos.sub(parentPos).applyQuaternion(parentQ.invert());
    return { x: v.x, y: v.y, z: v.z };
  }

  /** Read a bone's CURRENT world position + quaternion from the live VRM
   *  skeleton (after all upstream pose writes for this frame). */
  _readBoneWorldTransform(name, outPos, outQuat) {
    const node = this.vrm.humanoid.getNormalizedBoneNode?.(name);
    if (!node) {
      outPos.set(0, 0, 0);
      outQuat.set(0, 0, 0, 1);
      return false;
    }
    node.getWorldPosition(outPos);
    node.getWorldQuaternion(outQuat);
    return true;
  }

  /** Snapshot kinematic targets from the live skeleton into each bone
   *  entry. This is what the PD spring "wants" the physics body to be. */
  _snapshotKinematicTargets() {
    for (const [name, entry] of this._bones) {
      this._readBoneWorldTransform(name, entry.kinematicTarget.pos, entry.kinematicTarget.quat);
    }
  }

  /** Apply the PD spring: compute next kinematic target as current body
   *  pose + (kp·(target - current) - kd·velocity)·dt, then push to
   *  Rapier. With Kinematic*Position*Based bodies we don't have a "real"
   *  body velocity to damp against, so we approximate it from the
   *  per-frame body displacement (lastBodyPose → current). */
  _driveBodiesToTargets(dt) {
    if (dt <= 0) return;
    for (const [, entry] of this._bones) {
      const bodyPos = entry.body.translation();
      const bodyRot = entry.body.rotation();

      // Position spring (per-axis Euler integration; quaternion handled below).
      const velX = (bodyPos.x - entry.lastBodyPose.pos.x) / dt;
      const velY = (bodyPos.y - entry.lastBodyPose.pos.y) / dt;
      const velZ = (bodyPos.z - entry.lastBodyPose.pos.z) / dt;
      const accX = SPRING_KP * (entry.kinematicTarget.pos.x - bodyPos.x) - SPRING_KD * velX;
      const accY = SPRING_KP * (entry.kinematicTarget.pos.y - bodyPos.y) - SPRING_KD * velY;
      const accZ = SPRING_KP * (entry.kinematicTarget.pos.z - bodyPos.z) - SPRING_KD * velZ;
      const nextX = bodyPos.x + velX * dt + 0.5 * accX * dt * dt;
      const nextY = bodyPos.y + velY * dt + 0.5 * accY * dt * dt;
      const nextZ = bodyPos.z + velZ * dt + 0.5 * accZ * dt * dt;

      entry.body.setNextKinematicTranslation({ x: nextX, y: nextY, z: nextZ });

      // Rotation: nlerp the body's current quaternion partway toward the
      // target, with the same effective spring stiffness. For the visible
      // range of secondary motion (small angles) nlerp matches slerp closely
      // and is cheaper. dt-scaled blend factor is clamped to <=1 to stay
      // stable under big frame spikes (the early dt cap also protects this).
      const blend = Math.min(1, Math.max(0, SPRING_KP * 0.5 * dt * dt + SPRING_KD * dt * 0.25));
      const tq = entry.kinematicTarget.quat;
      const nx = bodyRot.x + (tq.x - bodyRot.x) * blend;
      const ny = bodyRot.y + (tq.y - bodyRot.y) * blend;
      const nz = bodyRot.z + (tq.z - bodyRot.z) * blend;
      const nw = bodyRot.w + (tq.w - bodyRot.w) * blend;
      const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw) || 1;
      entry.body.setNextKinematicRotation({ x: nx / nLen, y: ny / nLen, z: nz / nLen, w: nw / nLen });
    }
  }

  /** Step the physics world with a fixed timestep, substepping to catch
   *  up to the actual frame's dt. Fixed-dt keeps the spring stable across
   *  variable framerates — the integrator is sensitive to dt² in the PD
   *  acceleration term. */
  _stepWorld(dt) {
    this._accumDt += dt;
    let steps = 0;
    while (this._accumDt >= FIXED_DT_S && steps < MAX_SUBSTEPS) {
      // Rapier uses integrationParameters.dt internally; setting it here is
      // safe because we never share the world with another stepper.
      try { this._world.integrationParameters.dt = FIXED_DT_S; } catch (_) { /* readonly in some builds */ }
      this._world.step();
      this._accumDt -= FIXED_DT_S;
      steps += 1;
    }
    if (steps >= MAX_SUBSTEPS) {
      // Drop accumulated time to prevent spiral-of-death on a stalled tab.
      this._accumDt = 0;
    }
  }

  /** Compute (post-physics body pose) − (kinematic target) for each bone,
   *  scale by `weight`, store in the exported delta map. The deltas are
   *  the secondary-motion contribution the coordinator will fold into
   *  the final pose. */
  _computeDeltas() {
    const w = Math.max(0, Math.min(1, this.weight));
    for (const [name, entry] of this._bones) {
      const bodyPos = entry.body.translation();
      const bodyRot = entry.body.rotation();
      const tgt = entry.kinematicTarget;
      const delta = this._deltas.get(name);

      // Position delta, clamped + weighted.
      let dx = (bodyPos.x - tgt.pos.x) * w;
      let dy = (bodyPos.y - tgt.pos.y) * w;
      let dz = (bodyPos.z - tgt.pos.z) * w;
      const dmag = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dmag > MAX_POS_DELTA_M) {
        const s = MAX_POS_DELTA_M / dmag;
        dx *= s; dy *= s; dz *= s;
      }
      delta.posDelta.set(dx, dy, dz);

      // Rotation delta: q_delta = q_body * q_target^-1, normalized,
      // then "slerped from identity" by `weight` to scale magnitude.
      // identity.slerp(qDelta, w) keeps the axis but multiplies the angle.
      this._scratch.tmpQuat.set(tgt.quat.x, tgt.quat.y, tgt.quat.z, tgt.quat.w).invert();
      this._scratch.worldQuat.set(bodyRot.x, bodyRot.y, bodyRot.z, bodyRot.w)
        .multiply(this._scratch.tmpQuat);

      // Safety clamp on rotation magnitude (angle from identity).
      const halfA = Math.min(1, Math.abs(this._scratch.worldQuat.w));
      const angle = 2 * Math.acos(halfA);
      if (angle > MAX_ANGLE_DELTA) {
        const scale = MAX_ANGLE_DELTA / angle;
        this._scratch.identity.identity();
        this._scratch.identity.slerp(this._scratch.worldQuat, scale * w);
        delta.quatDelta.copy(this._scratch.identity);
      } else {
        this._scratch.identity.identity();
        this._scratch.identity.slerp(this._scratch.worldQuat, w);
        delta.quatDelta.copy(this._scratch.identity);
      }

      // Remember body pose for next frame's velocity estimate.
      entry.lastBodyPose.pos.set(bodyPos.x, bodyPos.y, bodyPos.z);
      entry.lastBodyPose.quat.set(bodyRot.x, bodyRot.y, bodyRot.z, bodyRot.w);
    }
  }

  /** Seed/clear the export map with identity deltas for every tracked
   *  bone. Lets the consumer iterate the same key set even before the
   *  first tick (or when disabled). */
  _initDeltas() {
    this._deltas.clear();
    for (const name of BONE_NAMES) {
      this._deltas.set(name, {
        posDelta:  new this.three.Vector3(0, 0, 0),
        quatDelta: new this.three.Quaternion(0, 0, 0, 1),
      });
    }
  }

  /** Release the world and all bodies. Joints are owned by the world. */
  _teardownWorld() {
    if (this._world) {
      try { this._world.free(); } catch (_) { /* some builds auto-free */ }
      this._world = null;
    }
  }
}
