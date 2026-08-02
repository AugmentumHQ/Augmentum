/**
 * sdf-compliance.js — SDF-gradient procedural body compliance.
 *
 * When a user's hand penetrates or approaches the AI VRM's body envelope,
 * sample the per-VRM voxel SDF + its gradient at the contact point, then
 * displace nearby torso/neck/shoulder/head bones along the negative
 * gradient direction. Spring-damp the displacement back to zero when the
 * user retreats. The body visibly "gives" under touch without a physics
 * engine — the SDF substrate already encodes both penetration depth
 * (signed distance) and push-out direction (gradient).
 *
 * Designed to compose ON TOP of base-pose channels: each frame it reads
 * the current `node.quaternion` (whatever the pose channels have just
 * written) and right-multiplies a small rotational delta. No state is
 * carried across base-pose writes — every frame the delta is computed
 * fresh from the spring-damped displacement state.
 *
 * Hand positions are sourced from a ContactReactor peer via
 * `getUserHands()`, so this module and the reactor share the same view
 * of user input without duplicating the controller/hand pickoff logic.
 *
 * Pairs with: contact-reactor.js, eventually a Rapier-based active
 * ragdoll for joint dynamics + secondary motion (this module handles
 * LOCAL compliance at the point of contact; Rapier would handle GLOBAL
 * body chain response).
 */

const ACTIVE_RANGE_M    = 0.08;   // hand within 8cm of (or inside) body → response engages
const FALLOFF_RANGE_M   = 0.30;   // bones within 30cm of contact participate, with falloff
const MAX_DISP_M        = 0.05;   // cap per-bone displacement magnitude at 5cm
const STIFFNESS_DEFAULT = 1.0;    // overall response gain
const RECOVER_HZ        = 6.0;    // critically-damped spring rate (per-second)

/** Tracked bones + their effective "lever arm" length (meters).
 *  This is the rough bone segment length used to convert linear
 *  displacement to a rotation angle (angle ≈ disp / arm). Values
 *  approximate VRM humanoid proportions for a ~1.6m avatar; bones
 *  not in the map are ignored. */
const BONE_ARMS = Object.freeze({
  spine:         0.18,
  chest:         0.16,
  upperChest:    0.12,
  neck:          0.10,
  head:          0.12,
  leftShoulder:  0.10,
  rightShoulder: 0.10,
});

export class SDFCompliance {
  /**
   * @param {object} opts
   * @param {object} opts.three           THREE namespace
   * @param {object} opts.vrm             VRM with __augmentumBodyAtlas attached
   * @param {object} opts.contactReactor  ContactReactor instance with getUserHands()
   * @param {number} [opts.stiffness]     gain on response magnitude (default 1.0)
   * @param {number} [opts.recoverHz]     spring return rate per second (default 6)
   */
  constructor({ three, vrm, contactReactor, stiffness, recoverHz }) {
    if (!three) throw new Error('SDFCompliance needs three');
    if (!vrm?.humanoid) throw new Error('SDFCompliance needs vrm with humanoid');
    this.three = three;
    this.vrm = vrm;
    this.atlas = vrm.__augmentumBodyAtlas || null;
    this.contactReactor = contactReactor || null;
    this.stiffness = stiffness ?? STIFFNESS_DEFAULT;
    this.recoverHz = recoverHz ?? RECOVER_HZ;
    this.enabled = !!this.atlas;

    /** Per-bone spring-damp state: { current, velocity } as Vector3. */
    this._bones = new Map();
    // Probe each tracked bone ONCE at init so per-frame code never has to
    // log absence (would spam at 90Hz). The per-frame guards in
    // `_accumulate` / `_applyBoneDelta` still handle null `getNormalizedBoneNode`
    // returns — this is purely a surfacing hook so missing bones show up in
    // the console exactly once per VRM swap. VRM 0.x rigs frequently omit
    // `upperChest`; flag that case specifically so the absence is legible
    // ("chest will absorb its share") rather than a generic "bone missing".
    const humanoid = vrm.humanoid;
    const hasChest = !!humanoid?.getNormalizedBoneNode?.('chest');
    for (const name of Object.keys(BONE_ARMS)) {
      const node = humanoid?.getNormalizedBoneNode?.(name) || null;
      if (!node) {
        if (name === 'upperChest' && hasChest) {
          console.debug('[sdf-compliance] upperChest absent — chest will absorb its share');
        } else {
          console.debug('[sdf-compliance] bone unavailable on this VRM:', name);
        }
      }
      this._bones.set(name, {
        current:  new three.Vector3(),
        velocity: new three.Vector3(),
        target:   new three.Vector3(),  // reused per-tick scratch
      });
    }

    this._scratch = {
      bonePos:  new three.Vector3(),
      handVec:  new three.Vector3(),
      axisW:    new three.Vector3(),
      axisL:    new three.Vector3(),
      pQuat:    new three.Quaternion(),
      pQuatInv: new three.Quaternion(),
      delta:    new three.Quaternion(),
      boneAxisW: new three.Vector3(),
      hipsPos:   new three.Vector3(),
      hipsQuat:  new three.Quaternion(),
      hipsScale: new three.Vector3(),
    };

    // Per-tick BodyFrame: maps live world-space hand positions into the atlas's
    // bake frame so compliance stays correct when the avatar is turned/scaled.
    // Rebuilt each tick() from the current normalized hips. Defaults to an
    // identity frame (== legacy raw-world behavior) until the first tick.
    this._frame = this.atlas ? this.atlas.frame(this.atlas.bakeHipsPos, this.atlas.bakeHipsQuat, 1) : null;
  }

  /** Refresh the world↔bake frame from the avatar's current normalized hips. */
  _updateFrame() {
    const hips = this.vrm.humanoid?.getNormalizedBoneNode?.('hips');
    if (!hips) return;   // keep last frame; better than snapping to identity
    const s = this._scratch;
    hips.getWorldPosition(s.hipsPos);
    hips.getWorldQuaternion(s.hipsQuat);
    hips.getWorldScale(s.hipsScale);
    this._frame = this.atlas.frame(
      [s.hipsPos.x, s.hipsPos.y, s.hipsPos.z],
      [s.hipsQuat.x, s.hipsQuat.y, s.hipsQuat.z, s.hipsQuat.w],
      s.hipsScale.x,
    );
  }

  /** Per-frame tick. Call AFTER pose channels have written `node.quaternion`
   *  and BEFORE `vrm.update()` so the delta makes it into the skinned mesh. */
  tick(dtMs) {
    if (!this.enabled || !this.atlas || !this.contactReactor) return;
    const dt = Math.max(0, dtMs / 1000);

    // Refresh the world↔bake transform for this frame's avatar pose BEFORE any
    // atlas query, so rotation/scale are accounted for at the contact point.
    this._updateFrame();

    // Reset per-tick targets.
    for (const state of this._bones.values()) {
      state.target.set(0, 0, 0);
    }

    // Accumulate displacement contributions from each user hand.
    const hands = this.contactReactor.getUserHands?.() || { L: null, R: null };
    if (hands.L) this._accumulate(hands.L);
    if (hands.R) this._accumulate(hands.R);

    // Integrate spring per bone, then apply rotational delta.
    const k = this.recoverHz * this.recoverHz;   // spring constant
    const c = 2 * this.recoverHz;                // critical damping
    for (const [name, state] of this._bones) {
      // F = k*(target - current) - c*velocity
      const fx = k * (state.target.x - state.current.x) - c * state.velocity.x;
      const fy = k * (state.target.y - state.current.y) - c * state.velocity.y;
      const fz = k * (state.target.z - state.current.z) - c * state.velocity.z;
      state.velocity.x += fx * dt;
      state.velocity.y += fy * dt;
      state.velocity.z += fz * dt;
      state.current.x  += state.velocity.x * dt;
      state.current.y  += state.velocity.y * dt;
      state.current.z  += state.velocity.z * dt;

      // Clamp to MAX_DISP_M magnitude to avoid blowups if integration overshoots.
      const m = state.current.length();
      if (m > MAX_DISP_M) state.current.multiplyScalar(MAX_DISP_M / m);

      if (m < 1e-4) continue;   // negligible — skip the quaternion work
      this._applyBoneDelta(name, state.current);
    }
  }

  /** Reset all compliance state. Call on VRM swap or session end. */
  reset() {
    for (const state of this._bones.values()) {
      state.current.set(0, 0, 0);
      state.velocity.set(0, 0, 0);
      state.target.set(0, 0, 0);
    }
  }

  dispose() {
    this.reset();
    this.enabled = false;
  }

  // ─── Internal ────────────────────────────────────────────────────────────
  _accumulate(handPos) {
    const [hx, hy, hz] = handPos;
    const sdfVal = this._frame.sdf([hx, hy, hz]);
    // Skip far-field: outside-grid sentinel or hand >ACTIVE_RANGE outside body.
    if (sdfVal > ACTIVE_RANGE_M || !Number.isFinite(sdfVal)) return;

    // Penetration "severity": 0..ACTIVE_RANGE+penetration. sdf<0 inside body
    // contributes more than sdf=ACTIVE_RANGE just outside.
    const severity = Math.max(0, ACTIVE_RANGE_M - sdfVal);

    // Gradient (outward unit-ish vector). Push body opposite of gradient
    // so the surface moves AWAY from the user's hand (chest leans back
    // when poked from the front, etc.).
    const g = this._frame.gradient([hx, hy, hz]);
    const pushX = -g[0], pushY = -g[1], pushZ = -g[2];

    for (const [name, state] of this._bones) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(name);
      if (!node) continue;
      node.getWorldPosition(this._scratch.bonePos);
      const bp = this._scratch.bonePos;
      const dx = bp.x - hx, dy = bp.y - hy, dz = bp.z - hz;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (dist > FALLOFF_RANGE_M) continue;

      // Quadratic falloff: sharper localization than linear, so chest
      // touch primarily moves chest, not also the head.
      const f = 1 - (dist / FALLOFF_RANGE_M);
      const falloff = f * f;
      const mag = MAX_DISP_M * (severity / ACTIVE_RANGE_M) * falloff * this.stiffness;

      state.target.x += pushX * mag;
      state.target.y += pushY * mag;
      state.target.z += pushZ * mag;
    }
  }

  _applyBoneDelta(name, dispWorld) {
    const node = this.vrm.humanoid.getNormalizedBoneNode?.(name);
    if (!node || !node.parent) return;
    const arm = BONE_ARMS[name];
    if (!arm) return;

    // Bone's world-space "tip" direction: take local +Y and rotate by the
    // bone's current world quaternion. VRM humanoid bones use +Y along the
    // bone segment by convention.
    this._scratch.boneAxisW.set(0, 1, 0);
    node.getWorldQuaternion(this._scratch.pQuat);
    this._scratch.boneAxisW.applyQuaternion(this._scratch.pQuat);

    // Rotation axis (world): perpendicular to both bone axis and displacement.
    this._scratch.axisW.crossVectors(this._scratch.boneAxisW, dispWorld);
    const axisLen = this._scratch.axisW.length();
    if (axisLen < 1e-5) return;
    this._scratch.axisW.divideScalar(axisLen);

    // Angle: small-angle approximation = disp / arm, clamped to avoid
    // numerical blowup if displacement somehow exceeds arm length.
    const dispMag = dispWorld.length();
    let angle = dispMag / arm;
    if (angle > 0.5) angle = 0.5;     // ~28°, generous safety cap

    // Convert world axis to parent-local space, since node.quaternion is
    // local-relative-to-parent. axisLocal = parentWorldInv * axisWorld
    node.parent.getWorldQuaternion(this._scratch.pQuat);
    this._scratch.pQuatInv.copy(this._scratch.pQuat).invert();
    this._scratch.axisL.copy(this._scratch.axisW).applyQuaternion(this._scratch.pQuatInv);

    // Build delta in parent-local space and PREMULTIPLY onto the bone's
    // current local quaternion. Right-multiply would apply the rotation
    // in the bone's own local frame (axis would co-rotate with the bone);
    // premultiply applies it in the parent's frame, which is what we want
    // — the world push direction stays world-stable as the bone tilts.
    //   newWorld = parentWorld * (delta_parentLocal * local)
    //            = parentWorld * delta_parentLocal * local
    // i.e. delta acts on the bone's current world orientation from the
    // parent's frame, matching the world-derived axis we built.
    this._scratch.delta.setFromAxisAngle(this._scratch.axisL, angle);
    node.quaternion.premultiply(this._scratch.delta);
  }
}
