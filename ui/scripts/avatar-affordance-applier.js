/**
 * avatar-affordance-applier.js — bone-rotation playback for static affordances
 *
 * Replaces the IK path for authored static affordances. Each affordance
 * carries the arm-chain Euler rotations from its source pose; setHandPose()
 * copies those rotations onto the live VRM's normalized humanoid bones.
 * No IK math, no pole hint, no collision response.
 *
 * Why this exists: the user authored each pose by hand using a recording
 * studio (scene-test, on the bundled-roster VRMs — mirrored axis profile).
 * Those JSONs ARE the correct arm configurations for the author's avatars,
 * but a raw rotation.set on an external VRM in the opposite axis profile
 * reproduces the same dorsal-curl bug that bit _applyHandPoses. So this
 * applier:
 *   - Captures finger rest quaternions at construction
 *   - Looks up the VRM's compatibility profile for armAxisProfile +
 *     fingerAxisProfile
 *   - Applies armAxisSign on arm bones and rest-relative + fingerAxisSign
 *     on finger bones, matching the in-animator hand-pose channel
 *
 * Companion module: avatar-pose-orchestrator.js handles slerp transitions
 * between affordances. Both can drive the same bones — orchestrator wins
 * during transitions, applier provides instant snap-to-pose. Compose by
 * having the orchestrator call applier internally for endpoint targets.
 *
 * THREE is injected via constructor — same pattern as avatar-animator.js.
 */

import { isFingerBoneName, isArmBoneName } from './avatar-pose-presets.js';
import { armAxisSignFromProfile, fingerAxisSignFromProfile } from './avatar-vrm-profile.js';

// VRM normalized humanoid arm-chain bone names (4 arm + 15 finger per side).
const ARM_CHAIN = {
  L: [
    'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
    'leftThumbProximal',  'leftThumbIntermediate',  'leftThumbDistal',
    'leftIndexProximal',  'leftIndexIntermediate',  'leftIndexDistal',
    'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
    'leftRingProximal',   'leftRingIntermediate',   'leftRingDistal',
    'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
  ],
  R: [
    'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
    'rightThumbProximal',  'rightThumbIntermediate',  'rightThumbDistal',
    'rightIndexProximal',  'rightIndexIntermediate',  'rightIndexDistal',
    'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
    'rightRingProximal',   'rightRingIntermediate',   'rightRingDistal',
    'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
  ],
};

// VRM normalized humanoid bones use specific Euler rotation orders for
// the IK chain bones. Setting `node.rotation.set(x, y, z, order)` with
// the wrong order produces wrong rotations. These values mirror the IK
// chain definitions in scene-test.html and avatar-pose-presets.js.
const BONE_ROTATION_ORDERS = {
  leftShoulder:  'ZXY', rightShoulder:  'ZXY',
  leftUpperArm:  'ZXY', rightUpperArm:  'ZXY',
  leftLowerArm:  'YZX', rightLowerArm:  'YZX',
  // Hand and finger bones default to 'XYZ'
};
function getBoneRotationOrder(name) {
  return BONE_ROTATION_ORDERS[name] || 'XYZ';
}

export class AvatarAffordanceApplier {
  /**
   * @param {object} opts
   * @param {object} opts.three        THREE namespace
   * @param {object} opts.vrm          VRM with humanoid
   * @param {object} opts.affordances  { name: { boneName: [x,y,z] } }
   *                                   from affordances-bones.json
   */
  constructor(opts = {}) {
    if (!opts.three) throw new Error('AvatarAffordanceApplier requires opts.three');
    if (!opts.vrm?.humanoid) throw new Error('requires opts.vrm with humanoid');
    this.three = opts.three;
    this.vrm = opts.vrm;
    this.affordances = opts.affordances || {};
    // Track active affordance per side for caller introspection
    this._active = { L: null, R: null };

    // Per-VRM axis signs. Affordance data was authored in scene-test
    // (POSE_PRESETS-mirrored convention), so we use 'mirrored' as the
    // author marker for arms.
    const profile = opts.vrm.__augmentumCompatibilityProfile;
    this._armSign = armAxisSignFromProfile(profile?.armAxisProfile, 'mirrored');
    this._fingerSign = fingerAxisSignFromProfile(profile?.fingerAxisProfile);

    // Capture finger rest quaternions so setHandPose() can compose
    // rest-relative writes on finger bones (matches _applyHandPoses).
    // Body/arm bones stay on absolute writes — they're shoulder-to-hand
    // chain rotations that the affordance pose authoritatively
    // configures, not delta curls.
    //
    // Prefers vrm.__augmentumFingerRestQuats (captured in loadVRM
    // immediately after resetNormalizedPose, before applyPosePreset
    // wrote the at-load natural curl). If that stash is absent (applier
    // built outside the standard loadVRM path), falls back to current
    // bone state — which on a clean rest reproduces the old behavior.
    const humanoid = opts.vrm.humanoid;
    const stashed = opts.vrm.__augmentumFingerRestQuats || null;
    this._fingerRestQuats = {};
    for (const side of ['L', 'R']) {
      for (const boneName of ARM_CHAIN[side]) {
        if (!isFingerBoneName(boneName)) continue;
        const stashedQuat = stashed?.[boneName];
        if (stashedQuat) {
          this._fingerRestQuats[boneName] = stashedQuat.clone();
          continue;
        }
        const node = humanoid.getNormalizedBoneNode?.(boneName);
        if (node) this._fingerRestQuats[boneName] = node.quaternion.clone();
      }
    }
    // Reusable scratch — same shape as the animator's hand-pose path.
    this._tmpEuler = new opts.three.Euler();
    this._tmpQuat = new opts.three.Quaternion();
  }

  /**
   * Apply an affordance's arm-chain rotations to the named hand's bones.
   * Bones not present in the affordance are left untouched (previous
   * pose persists), so calling setHandPose('L', 'chin_L') only writes
   * left-arm bones; the right arm is whatever it was before.
   *
   * @param {'L' | 'R'} side
   * @param {string} affordanceName
   */
  setHandPose(side, affordanceName) {
    const aff = this.affordances[affordanceName];
    if (!aff) {
      console.warn(`[applier] unknown affordance: ${affordanceName}`);
      return false;
    }
    const expectedSet = new Set(ARM_CHAIN[side]);
    const humanoid = this.vrm.humanoid;
    const armSign = this._armSign;
    const fingerSign = this._fingerSign;
    let written = 0;
    for (const [boneName, rot] of Object.entries(aff)) {
      // Defensive: only write bones in the requested side's chain. Some
      // affordances might have bones for both sides if authored that way.
      if (!expectedSet.has(boneName)) continue;
      const node = humanoid.getNormalizedBoneNode(boneName);
      if (!node) continue;
      const order = getBoneRotationOrder(boneName);

      if (isFingerBoneName(boneName)) {
        // Rest-relative + finger axis sign. Same idiom as the animator's
        // _applyHandPoses so cross-VRM behavior matches the procedural
        // hand-pose channel.
        const rest = this._fingerRestQuats[boneName];
        if (!rest) {
          // Bone exists now but didn't at construction (shouldn't happen,
          // but bail gracefully) — fall back to absolute write.
          node.rotation.set(rot[0], rot[1], rot[2], order);
        } else {
          this._tmpEuler.set(
            rot[0] * fingerSign.x,
            rot[1] * fingerSign.y,
            rot[2] * fingerSign.z,
            order,
          );
          this._tmpQuat.setFromEuler(this._tmpEuler);
          node.quaternion.copy(rest).multiply(this._tmpQuat);
        }
      } else if (isArmBoneName(boneName)) {
        node.rotation.set(
          rot[0] * armSign.x,
          rot[1] * armSign.y,
          rot[2] * armSign.z,
          order,
        );
      } else {
        node.rotation.set(rot[0], rot[1], rot[2], order);
      }
      written += 1;
    }
    this._active[side] = affordanceName;
    return written > 0;
  }

  /** Reset one side's arm-chain rotations to rest pose. Finger bones
   *  restore from the captured rest quaternion (not identity) so VRMs
   *  whose normalized humanoid has non-identity finger bind don't get
   *  jammed straight on release. */
  releaseHand(side) {
    const humanoid = this.vrm.humanoid;
    for (const boneName of ARM_CHAIN[side]) {
      const node = humanoid.getNormalizedBoneNode(boneName);
      if (!node) continue;
      if (isFingerBoneName(boneName) && this._fingerRestQuats[boneName]) {
        node.quaternion.copy(this._fingerRestQuats[boneName]);
      } else {
        node.rotation.set(0, 0, 0);
      }
    }
    this._active[side] = null;
  }

  /** Reset both sides + release. Equivalent to humanoid.resetNormalizedPose
   *  scoped to arm chains only — leaves spine/legs/head untouched. */
  releaseAll() {
    this.releaseHand('L');
    this.releaseHand('R');
  }

  /** Active affordance for a side (or null). */
  getActive(side) {
    return this._active[side];
  }

  /** Available affordance names. */
  listAffordances() {
    return Object.keys(this.affordances).sort();
  }

  /** Convenience static loader. Fetches affordances-bones.json from a URL,
   *  constructs an applier with it. */
  static async load(opts) {
    const url = opts.affordancesUrl || '/poses/affordances-bones.json';
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
    const data = await res.json();
    return new AvatarAffordanceApplier({
      three: opts.three,
      vrm: opts.vrm,
      affordances: data.affordances || data,
    });
  }
}
