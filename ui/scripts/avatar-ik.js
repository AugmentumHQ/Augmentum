/**
 * avatar-ik.js — analytical 2-bone IK + affordance library
 *
 * Drives a VRM's arm chains to land hands at semantic targets ("chin",
 * "hip_left", "behind_back", etc.) without authoring per-pose rotations.
 * The complete authored static pose corpus is replaced by a small library
 * of 3D positions (extracted from those same poses via FK), and an IK
 * solver computes the joint rotations every frame.
 *
 * Why this replaces the orchestrator/family system long-term:
 *   - Authoring a new pose becomes "extract a hand position" not
 *     "record 30+ bone quaternions". O(N) authoring instead of O(N²).
 *   - Body-clipping `via` waypoints become unnecessary because the
 *     solver routes the hand around a torso capsule rather than blindly
 *     slerping joint quaternions through the body.
 *   - Per-character calibration is automatic — VRM bone proportions vary
 *     but `affordance position * skeleton_height` produces a target that
 *     is right for any rig.
 *
 * What it does NOT do (yet):
 *   - Wrist orientation. Right now the wrist follows the bone chain;
 *     authoring poses had explicit wrist rotations (e.g. palm-down on
 *     hip) that we'll need to add as per-affordance hints.
 *   - Body capsule collision. The IK solver routes the hand through
 *     the shortest path; for affordances behind the back this currently
 *     works because the pose is reachable, but for transitions between
 *     chest-front and back-of-hip the hand would still cut through
 *     unless we add capsule routing on the goal path.
 *   - Smoothing across affordance changes. Each call snaps to the new
 *     target. A wrapping interpolation layer (lerp goal positions over
 *     0.5-2s) belongs above this module.
 *   - Finger curl. Stays in HAND_POSES territory; this module is arm-only.
 *
 * Geometry contract:
 *   - Affordance positions are stored as 3-vectors in
 *     skeleton-height-normalized, hip-relative, avatar-local coordinates
 *     (matches what the motion-database extractor outputs).
 *   - Runtime: target_world = hips_world + position * skeleton_height
 *     applied in the avatar's current frame (so the avatar can rotate
 *     and the affordance follows the hip frame correctly).
 *
 * Math:
 *   - Two-bone IK via law of cosines on the triangle
 *     (shoulder, elbow, hand). Pole vector chooses elbow side.
 *   - Bone rotations expressed via setFromUnitVectors mapping rest
 *     child-direction (in bone-local frame) to desired direction
 *     (converted to bone-parent frame). This avoids the bone-axis-
 *     convention guesswork that breaks across VRM rigs.
 *
 * THREE is injected — same pattern as avatar-animator.js. No imports.
 */

export class AvatarIK {
  /**
   * @param {object} opts
   * @param {object} opts.three   THREE namespace
   * @param {object} opts.vrm     loaded VRM (with humanoid)
   * @param {object} opts.affordances  { [name]: { position: [x,y,z] } }
   *                                   — keyed library; positions are in
   *                                   skeleton-units, hip-relative.
   * @param {[number, number, number] | { L, R }} [opts.poleHint]
   *   Direction in AVATAR-LOCAL space the elbow should point toward.
   *   Per-side is the natural form because each hand's elbow swings
   *   to its OWN side (left elbow goes to avatar's left, right elbow
   *   to avatar's right) — a single shared pole forces one of them
   *   to fold through the body on inward reaches like clasping.
   *   Defaults: L = [-1, -0.2, -1] (out-left + forward + down),
   *             R = [+1, -0.2, -1] (out-right + forward + down).
   *   If a single 3-array is passed, the X component is auto-flipped
   *   per side so a positive X reads as "outward" for both hands.
   */
  constructor(opts = {}) {
    if (!opts.three) throw new Error('AvatarIK requires opts.three');
    if (!opts.vrm?.humanoid) throw new Error('AvatarIK requires opts.vrm with humanoid');
    this.three = opts.three;
    this.vrm = opts.vrm;
    this.affordances = opts.affordances || {};
    this.poleHint = opts.poleHint || { L: [-1, -0.2, -1], R: [+1, -0.2, -1] };
    // Optional Body Atlas for body-aware solving:
    //   - Pole derived from "max-clearance candidate" on the elbow swing circle
    //   - Joint push-out for elbow positions inside the body
    //   - Reach-target validation (warns if target is inside body)
    // If absent, falls back to constant pole hint and no collision response.
    this.atlas = opts.bodyAtlas || null;
    this.atlasOpts = {
      circleSamples: opts.atlasOpts?.circleSamples ?? 12,
      elbowClearance: opts.atlasOpts?.elbowClearance ?? 0.02,
      pushIters: opts.atlasOpts?.pushIters ?? 6,
    };
    this._scratch = {
      v: new this.three.Vector3(),
      v2: new this.three.Vector3(),
      v3: new this.three.Vector3(),
      v4: new this.three.Vector3(),
      v5: new this.three.Vector3(),
      q: new this.three.Quaternion(),
      q2: new this.three.Quaternion(),
      fHipsP: new this.three.Vector3(),    // dedicated: BodyFrame hips pose
      fHipsQ: new this.three.Quaternion(),
      fHipsS: new this.three.Vector3(),
    };
    this._initRig();
  }

  /**
   * Capture rest geometry once at construction. We snapshot:
   *   - per-side: upper arm length, lower arm length
   *   - per-side: child position in parent-local frame at REST
   * These are used at runtime to convert "I want the hand at world point P"
   * into bone-local rotations without depending on bone axis conventions.
   */
  _initRig() {
    const h = this.vrm.humanoid;
    const required = {
      L: ['leftUpperArm', 'leftLowerArm', 'leftHand'],
      R: ['rightUpperArm', 'rightLowerArm', 'rightHand'],
    };
    this.bones = { L: {}, R: {} };
    for (const side of ['L', 'R']) {
      const [u, l, h_] = required[side];
      this.bones[side].upper = h.getNormalizedBoneNode(u);
      this.bones[side].lower = h.getNormalizedBoneNode(l);
      this.bones[side].hand  = h.getNormalizedBoneNode(h_);
      if (!this.bones[side].upper || !this.bones[side].lower || !this.bones[side].hand) {
        throw new Error(`Missing arm bone(s) for side ${side}: ${u}/${l}/${h_}`);
      }
    }
    this.bones.L.hips = h.getNormalizedBoneNode('hips');
    this.bones.R.hips = this.bones.L.hips;
    if (!this.bones.L.hips) throw new Error('Missing hips bone');

    // Capture rest geometry. b.lower.position is lowerArm's origin in
    // upperArm's LOCAL frame (THREE bone positions are parent-local).
    // Same for b.hand.position relative to lowerArm. These vectors
    // describe where each child sits when the parent has identity rotation.
    this.rest = { L: {}, R: {} };
    for (const side of ['L', 'R']) {
      const b = this.bones[side];
      const upperToElbow = b.lower.position.clone();
      const elbowToHand = b.hand.position.clone();
      this.rest[side].upperLen = upperToElbow.length();
      this.rest[side].lowerLen = elbowToHand.length();
      this.rest[side].upperRestDir = upperToElbow.normalize();
      this.rest[side].lowerRestDir = elbowToHand.normalize();
    }

    // Skeleton height — head world Y at rest, used to scale hip-relative
    // affordance positions back to world space at runtime.
    const head = h.getNormalizedBoneNode('head');
    this.vrm.scene.updateMatrixWorld(true);
    this.skeletonHeight = head ? head.getWorldPosition(this._scratch.v).y : 1.6;
    if (this.skeletonHeight < 0.5) this.skeletonHeight = 1.6;  // sanity floor

    // Per-side active target. null = released (IK does not write this side
    // and lets whatever animation layer above set the bones).
    this._handTargets = { L: null, R: null };
  }

  // ─── Public API ──────────────────────────────────────────────────────

  /**
   * Aim a hand at a named affordance from the library.
   *
   * @param {'L' | 'R'} side
   * @param {string} affordance   name in this.affordances
   * @param {object} [opts]
   * @param {number} [opts.energy]  0–1 multiplier on engagement (future use)
   */
  setHandTarget(side, affordance, opts = {}) {
    const aff = this.affordances[affordance];
    if (!aff) {
      console.warn(`[avatar-ik] unknown affordance: ${affordance}`);
      this._handTargets[side] = null;
      return;
    }
    this._handTargets[side] = {
      kind: 'affordance',
      name: affordance,
      position: aff.position,
      energy: opts.energy ?? 1,
    };
  }

  /**
   * Aim a hand at a raw 3D position (skeleton-units, hip-relative).
   * Bypasses the affordance library — useful for procedural targets
   * (gaze-aimed pointing, holding an object at a tracked position, etc.).
   */
  setHandPosition(side, position) {
    this._handTargets[side] = { kind: 'position', position, energy: 1 };
  }

  /**
   * Aim a hand at a raw WORLD-space position (meters). Converts to the
   * IK's internal hip-relative skeleton-normalized convention. Use this
   * when your target comes from world-space sources (atlas surface
   * points, world-space markers, etc.) instead of normalized affordance
   * positions.
   */
  setHandPositionWorld(side, worldPos) {
    const hipsWorld = this.bones[side].hips.getWorldPosition(this._scratch.v);
    const norm = [
      (worldPos[0] - hipsWorld.x) / this.skeletonHeight,
      (worldPos[1] - hipsWorld.y) / this.skeletonHeight,
      (worldPos[2] - hipsWorld.z) / this.skeletonHeight,
    ];
    this._handTargets[side] = { kind: 'position', position: norm, energy: 1 };
  }

  /** Stop driving the hand; the IK leaves it where it last was. */
  releaseHand(side) {
    this._handTargets[side] = null;
  }

  /** True if either side is being driven. */
  isActive() {
    return !!(this._handTargets.L || this._handTargets.R);
  }

  /**
   * Per-frame solve. Call from the render loop AFTER any layer that
   * positions the avatar root or rotates the chest/spine, so the IK
   * sees current parent transforms; call BEFORE vrm.update(dt) so the
   * normalized humanoid → raw skeleton push picks up our writes.
   */
  update(_dt) {
    this.vrm.scene.updateMatrixWorld(true);
    if (this._handTargets.L) this._solveArm('L', this._handTargets.L);
    if (this._handTargets.R) this._solveArm('R', this._handTargets.R);
  }

  // ─── Internal: 2-bone solver ─────────────────────────────────────────

  _solveArm(side, target) {
    const b = this.bones[side];
    const r = this.rest[side];
    const T = this.three;
    const sc = this._scratch;

    // 1. Target world position from hip-relative skeleton-units coordinates.
    //    The motion-database extractor stored offsets as `hand_world - hips_world`
    //    AFTER the avatar's face-camera rotation was already applied, so the
    //    affordance values are in WORLD-aligned axes (not hips-local). Adding
    //    them directly to hipsWorldPos gives the correct world target. If we
    //    re-rotated by hipsWorldRot we'd double-rotate (her-left becomes her-
    //    right, in front of becomes behind — exactly the visual bug we hit).
    //    TODO: extract affordances in TRUE hips-local frame so this becomes
    //    rotation-aware (avatar can face any direction; targets follow). For
    //    now the IK assumes runtime orientation matches extraction (face-camera).
    const hipsWorldPos = b.hips.getWorldPosition(sc.v);
    const targetWorld = sc.v3.set(
      hipsWorldPos.x + target.position[0] * this.skeletonHeight,
      hipsWorldPos.y + target.position[1] * this.skeletonHeight,
      hipsWorldPos.z + target.position[2] * this.skeletonHeight,
    );

    // 2. Shoulder = upper arm bone origin in world.
    const shoulderWorld = b.upper.getWorldPosition(sc.v4);

    // 3. Triangle reach (law of cosines is computed in step 6 below).
    const distToTarget = shoulderWorld.distanceTo(targetWorld);
    const maxReach = (r.upperLen + r.lowerLen) * 0.99;
    const minReach = Math.abs(r.upperLen - r.lowerLen) * 1.01;
    const reach = Math.max(minReach, Math.min(distToTarget, maxReach));

    // 4. Direction from shoulder to target (WORLD frame).
    const dirToTargetWorld = sc.v5.subVectors(targetWorld, shoulderWorld).normalize();

    // 5-6. Geometry of the swing circle. Two-bone IK has one degree of
    //   freedom after fixing shoulder + wrist positions: the elbow swings
    //   around the shoulder→target axis on a circle of radius
    //     R_swing = upperLen * sin(shoulderAngle)
    //   centered on the axis at distance
    //     d_axis  = upperLen * cos(shoulderAngle)
    //   from the shoulder. We sample candidate elbow positions around this
    //   circle and either pick by atlas (max-clearance) or by constant pole
    //   hint (legacy behavior).
    const cosShoulderAngle = (r.upperLen ** 2 + reach ** 2 - r.lowerLen ** 2)
      / (2 * r.upperLen * reach);
    const cosClamped = Math.max(-1, Math.min(1, cosShoulderAngle));
    const sinClamped = Math.sqrt(Math.max(0, 1 - cosClamped * cosClamped));
    const axisCenter = new T.Vector3().copy(shoulderWorld)
      .addScaledVector(dirToTargetWorld, r.upperLen * cosClamped);
    const swingRadius = r.upperLen * sinClamped;

    // Build an orthonormal basis (u, v) on the plane perpendicular to dirToTarget.
    // Pick u = any vector ⊥ dirToTarget (use cross with global Y, fall back to X).
    const u = new T.Vector3();
    if (Math.abs(dirToTargetWorld.y) < 0.9) {
      u.set(0, 1, 0).cross(dirToTargetWorld);
    } else {
      u.set(1, 0, 0).cross(dirToTargetWorld);
    }
    u.normalize();
    const v = new T.Vector3().crossVectors(dirToTargetWorld, u).normalize();

    let elbowWorld;
    if (this.atlas) {
      // BodyFrame maps these live world-space elbow candidates into the atlas's
      // bake frame so collision stays correct when the avatar is turned/scaled.
      // The candidate positions below are real current-pose world coords; the
      // frame handles the world→bake similarity (rotation + translation + scale).
      const hpW = b.hips.getWorldPosition(sc.fHipsP);
      const hqW = b.hips.getWorldQuaternion(sc.fHipsQ);
      const hsW = b.hips.getWorldScale(sc.fHipsS);
      const frame = this.atlas.frame(
        [hpW.x, hpW.y, hpW.z],
        [hqW.x, hqW.y, hqW.z, hqW.w],
        hsW.x,
      );
      // ── Atlas-aware: anatomical pole is PRIMARY (chooses the natural
      //   elbow position for the target), atlas SDF is used only for
      //   collision response. This avoids the failure mode where naive
      //   max-clearance selection puts the elbow up over the head because
      //   that's the part of free space with the most clearance.
      //
      //   1. Apply constant pole hint through hips frame (legacy behavior)
      //   2. Place elbow via projection
      //   3. Score the result by SDF: only if elbow is INSIDE body
      //      (sdf < 0) AND a different swing-circle candidate would put
      //      it OUTSIDE, swap to that better candidate.
      //   4. After swap, push out via gradient until clearance met.
      const hipsWorldRot = b.hips.getWorldQuaternion(sc.q);
      let poleLocal;
      if (Array.isArray(this.poleHint)) {
        const sgn = side === 'L' ? -1 : 1;
        poleLocal = [sgn * Math.abs(this.poleHint[0]), this.poleHint[1], this.poleHint[2]];
      } else {
        poleLocal = this.poleHint[side] || [side === 'L' ? -1 : 1, -0.2, -1];
      }
      const poleWorld = new T.Vector3(poleLocal[0], poleLocal[1], poleLocal[2])
        .applyQuaternion(hipsWorldRot)
        .normalize();
      const dirDotPole = dirToTargetWorld.dot(poleWorld);
      const polePerp = new T.Vector3().copy(poleWorld)
        .addScaledVector(dirToTargetWorld, -dirDotPole);
      if (polePerp.lengthSq() < 1e-6) {
        polePerp.set(0, 1, 0).addScaledVector(dirToTargetWorld, -dirToTargetWorld.y);
        if (polePerp.lengthSq() < 1e-6) polePerp.set(1, 0, 0);
      }
      polePerp.normalize();

      // Initial elbow from anatomical pole
      elbowWorld = new T.Vector3().copy(axisCenter)
        .addScaledVector(polePerp, swingRadius);

      // Atlas collision check: if elbow is inside body, try other candidates
      // on the swing circle and pick the one that's least-inside (or outside).
      const elbowSdf = frame.sdf([elbowWorld.x, elbowWorld.y, elbowWorld.z]);
      if (elbowSdf < this.atlasOpts.elbowClearance) {
        // Scan candidates and find the one with best SDF that's still
        // CLOSE to the anatomical pole direction (don't completely abandon
        // anatomy — only deviate when current pick is in conflict).
        const N = this.atlasOpts.circleSamples;
        let bestScore = elbowSdf;
        let bestElbow = elbowWorld.clone();
        const cand = new T.Vector3();
        // Direction we WANT (anatomical) for tie-breaking
        const wantDir = polePerp.clone();
        for (let s = 0; s < N; s++) {
          const theta = (s / N) * Math.PI * 2;
          cand.copy(axisCenter)
            .addScaledVector(u, swingRadius * Math.cos(theta))
            .addScaledVector(v, swingRadius * Math.sin(theta));
          const sdf = frame.sdf([cand.x, cand.y, cand.z]);
          // Score: SDF (clear of body) + anatomical alignment (stay close to wantDir)
          const dirToCand = cand.clone().sub(axisCenter).normalize();
          const align = dirToCand.dot(wantDir);
          // Heavily prefer candidates outside the body, but among those,
          // prefer ones aligned with anatomy.
          const score = (sdf > this.atlasOpts.elbowClearance ? 1.0 : sdf * 5) + align * 0.3;
          if (score > bestScore) {
            bestScore = score;
            bestElbow = cand.clone();
          }
        }
        elbowWorld = bestElbow;
        // Final push-out if needed
        const sdfNow = frame.sdf([elbowWorld.x, elbowWorld.y, elbowWorld.z]);
        if (sdfNow < this.atlasOpts.elbowClearance) {
          const pushed = frame.pushOutsideBody(
            [elbowWorld.x, elbowWorld.y, elbowWorld.z],
            this.atlasOpts.elbowClearance,
            this.atlasOpts.pushIters,
          );
          elbowWorld = new T.Vector3(pushed[0], pushed[1], pushed[2]);
        }
      }
    } else {
      // ── Fallback: constant pole hint (legacy behavior)
      const hipsWorldRot = b.hips.getWorldQuaternion(sc.q);
      let poleLocal;
      if (Array.isArray(this.poleHint)) {
        const sgn = side === 'L' ? -1 : 1;
        poleLocal = [sgn * Math.abs(this.poleHint[0]), this.poleHint[1], this.poleHint[2]];
      } else {
        poleLocal = this.poleHint[side] || [side === 'L' ? -1 : 1, -0.2, -1];
      }
      const poleWorld = new T.Vector3(poleLocal[0], poleLocal[1], poleLocal[2])
        .applyQuaternion(hipsWorldRot)
        .normalize();
      const dirDotPole = dirToTargetWorld.dot(poleWorld);
      const polePerp = new T.Vector3().copy(poleWorld)
        .addScaledVector(dirToTargetWorld, -dirDotPole);
      if (polePerp.lengthSq() < 1e-6) {
        polePerp.set(0, 1, 0).addScaledVector(dirToTargetWorld, -dirToTargetWorld.y);
        if (polePerp.lengthSq() < 1e-6) polePerp.set(1, 0, 0);
      }
      polePerp.normalize();
      elbowWorld = new T.Vector3().copy(shoulderWorld)
        .addScaledVector(dirToTargetWorld, r.upperLen * cosClamped)
        .addScaledVector(polePerp, r.upperLen * sinClamped);
    }
    const upperDirWorld = new T.Vector3()
      .subVectors(elbowWorld, shoulderWorld).normalize();

    // 8. Express upper arm direction in its PARENT's local frame (so we can
    //    map rest direction → desired direction directly as a local rotation).
    const upperParentInvQ = b.upper.parent.getWorldQuaternion(new T.Quaternion()).invert();
    const upperDirInParent = upperDirWorld.clone().applyQuaternion(upperParentInvQ);

    // 9. Upper arm local rotation: rotates rest child-direction to desired.
    //    Bone.quaternion transforms vectors from bone-local to parent-local
    //    (THREE convention); rest dir IS the child position in bone-local
    //    frame, so this is exactly the rotation we want.
    const upperLocal = new T.Quaternion().setFromUnitVectors(r.upperRestDir, upperDirInParent);
    b.upper.quaternion.copy(upperLocal);
    b.upper.updateMatrixWorld(true);  // refresh so lower arm's parent is current

    // 10. Lower arm direction: elbow → hand target in world.
    const lowerDirWorld = new T.Vector3()
      .subVectors(targetWorld, elbowWorld).normalize();

    // 11. Convert to lower arm's parent (upper arm) local frame.
    const lowerParentInvQ = b.lower.parent.getWorldQuaternion(new T.Quaternion()).invert();
    const lowerDirInParent = lowerDirWorld.clone().applyQuaternion(lowerParentInvQ);

    // 12. Lower arm local rotation.
    const lowerLocal = new T.Quaternion().setFromUnitVectors(r.lowerRestDir, lowerDirInParent);
    b.lower.quaternion.copy(lowerLocal);
    b.lower.updateMatrixWorld(true);
  }
}

/**
 * Convenience loader — fetches affordances.json and returns the parsed
 * library ready to pass to AvatarIK constructor.
 */
export async function loadAffordances(url = '/poses/affordances.json') {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load affordances from ${url}: ${res.status}`);
  const data = await res.json();
  return data.affordances || data;
}
