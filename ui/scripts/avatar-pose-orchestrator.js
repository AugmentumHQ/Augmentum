/**
 * avatar-pose-orchestrator.js — pose family state machine with slerp drift
 *
 * Owns "what static body pose is the avatar currently holding, and how
 * is it transitioning to the next one." Two flavors of transition:
 *
 *   1. Intra-family drift — when a family has multiple members, the
 *      orchestrator picks a different member every dwell window and
 *      slerps to it. This is what makes a 2-pose "head turning"
 *      family read as the avatar looking around naturally, or a
 *      thinking_a ↔ thinking_b drift read as deliberation.
 *
 *   2. Inter-family transition — `setFamily('thinking')` swaps the
 *      active family with a longer slerp, then resumes intra-family
 *      drift in the new family. Drives mode/intent changes.
 *
 * Output: per-bone quaternion + Euler+order via getCurrentPose(). The
 * caller (scene-test or production avatar-animator) decides how to
 * apply — direct rotation set, spring target, additive layer, etc.
 *
 * THREE is injected via constructor — do not import here. Same pattern
 * as avatar-animator.js.
 */

import {
  POSE_PRESETS,
  POSE_FAMILIES,
  getPose,
  getFamily,
  getFamilyMembers,
  getBoneRotationOrder,
  isFingerBoneName,
  isArmBoneName,
} from './avatar-pose-presets.js';

const DEFAULT_DWELL_MS = [4500, 8500];     // intra-family member hold time
const DEFAULT_INTRA_SLERP_MS = 1200;       // member → member transition
const DEFAULT_INTER_SLERP_MS = 1800;       // family → family transition

// Bones whose orientation is meaningful relative to their parent
// (wrist twist, finger curl) and that should CARRY OVER between
// transitions when the destination doesn't author them.
//
// Carry-over works by RELEASING OWNERSHIP: when a bone in this set
// was authored by the previous pose but not the next, the orchestrator
// drops it from _activeBoneKeys entirely. The live VRM value
// (whatever the orchestrator last wrote, or whatever a higher-priority
// system like applyHandPose wrote afterward) then persists undisturbed.
//
// The alternative — keeping the bone in _activeBoneKeys with TO=FROM
// "no slerp movement" — was the original implementation, but it had
// a fatal flaw: the orchestrator still wrote that value to the VRM
// every frame, stomping any external write that came in between
// transitions. Releasing ownership avoids that fight.
const CARRY_OVER_BONES = new Set([
  'leftHand', 'rightHand',
  'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal',
  'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
  'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
  'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
  'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
  'rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal',
  'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
  'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
  'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
  'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
]);

// Cubic ease-in-out — slow start/end, faster middle. Reads more
// natural than linear for body-pose transitions.
function _easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function _randInt(min, max) {
  return Math.floor(min + Math.random() * (max - min));
}

export class PoseOrchestrator {
  /**
   * @param {object} opts
   * @param {object} opts.three  THREE namespace (needs Quaternion, Euler)
   * @param {[number, number]} [opts.dwellRangeMs]  intra-family hold time range
   * @param {number} [opts.intraSlerpMs]            intra-family transition duration
   * @param {number} [opts.interSlerpMs]            inter-family transition duration
   * @param {() => number} [opts.now]               clock fn (defaults to performance.now)
   * @param {(boneName: string) => object|null} [opts.boneStateProvider]
   *   Optional callback returning a THREE.Quaternion for the live bone
   *   state (e.g. read from a VRM). Used as the FROM endpoint on the
   *   first transition so activation slerps from the avatar's current
   *   pose rather than identity (which would cause a brief T-pose flash).
   * @param {[number, number, number]} [opts.defaultAvatarPosition]
   *   Scene-default world position for the avatar root. Used as the
   *   fallback when transitioning OUT of a pose that authored its own
   *   `_avatarPosition` (e.g. sittingEdge → standing) so the avatar
   *   smoothly returns to standing instead of staying in the seat.
   * @param {[number, number, number]} [opts.defaultHipsTranslation]
   *   Same idea for hip translation (rest = [0,0,0]).
   * @param {object} [opts.restPose]
   *   Bone-keyed map of `[x, y, z]` Euler triples giving the avatar's
   *   neutral rest values. Used as the release target for bones the
   *   incoming pose doesn't author and that aren't in CARRY_OVER_BONES.
   *   Without this, unauthored bones release to identity (T-pose),
   *   which yanks shoulder droop / head tilt / hip rotation flat
   *   mid-transition. Pass `POSE_PRESETS.natural.bones` here.
   */
  constructor(opts = {}) {
    if (!opts.three) throw new Error('PoseOrchestrator requires opts.three (THREE namespace)');
    this._THREE = opts.three;
    this._now = opts.now || (() => performance.now());
    this._boneStateProvider = opts.boneStateProvider || null;
    this._defaultAvatarPosition = opts.defaultAvatarPosition || null;
    this._defaultHipsTranslation = opts.defaultHipsTranslation || null;
    this._restPose = opts.restPose || null;

    // Per-VRM axis sign correction. POSE_PRESETS authored against the
    // bundled-roster avatars (mirrored arm/finger profile); VRMs in the
    // opposite convention need Z negation on arms and fingers so the
    // authored Eulers produce the same visual on every VRM.
    // Symmetric with avatar-animator.js _armTargetSign and the rewritten
    // _applyHandPoses fingerAxisSign correction.
    this._armAxisSign = opts.armAxisSign || { x: 1, y: 1, z: 1 };
    this._fingerAxisSign = opts.fingerAxisSign || { x: 1, y: 1, z: 1 };

    this.dwellRange = opts.dwellRangeMs || DEFAULT_DWELL_MS;
    this.intraSlerpMs = opts.intraSlerpMs ?? DEFAULT_INTRA_SLERP_MS;
    this.interSlerpMs = opts.interSlerpMs ?? DEFAULT_INTER_SLERP_MS;

    this.activeFamily = null;
    this.activeMember = null;        // pose name currently held (post-transition)
    this.previousMember = null;      // pose name being slerped from
    this.transitionStart = 0;
    this.transitionDur = 0;          // 0 when at rest at activeMember
    this.dwellUntil = 0;             // when 0 + at-rest, intra-family drift not scheduled

    // Per-bone state. _from / _to are quaternion endpoints of the
    // CURRENT slerp; _current is the latest interpolated value.
    this._fromQuats = {};
    this._toQuats = {};
    this._currentQuats = {};

    // Reusable scratch objects to avoid per-frame allocation.
    this._scratchEuler = new this._THREE.Euler();
    this._scratchQuat = new this._THREE.Quaternion();

    // Bones touched by the active OR previous member, so getCurrentPose
    // returns a stable key set during transitions.
    this._activeBoneKeys = new Set();
  }

  // ─── Public API ──────────────────────────────────────────────────────

  /**
   * Switch to a different family. Picks a starting member at random and
   * begins the inter-family slerp. If already in this family, no-op.
   *
   * @param {string} familyName   one of POSE_FAMILIES keys
   * @param {object} [opts]
   * @param {string} [opts.startMember]   force a specific member to start
   * @param {number} [opts.transitionMs]  override the slerp duration
   */
  setFamily(familyName, opts = {}) {
    if (this.activeFamily === familyName && this.activeMember) return;
    const members = getFamilyMembers(familyName);
    if (!members.length) {
      console.warn('[pose-orchestrator] unknown family:', familyName);
      return;
    }
    this.activeFamily = familyName;
    const start = opts.startMember && members.includes(opts.startMember)
      ? opts.startMember
      : members[Math.floor(Math.random() * members.length)];
    this._enterPose(start, { transitionMs: opts.transitionMs ?? this.interSlerpMs });
  }

  /**
   * Pin a specific pose. The orchestrator switches to the pose's family
   * and holds that exact member; if the family has siblings, drift will
   * resume after the dwell window unless `opts.holdIndefinitely` is set.
   *
   * @param {string} poseName
   * @param {object} [opts]
   * @param {number} [opts.transitionMs]
   * @param {boolean} [opts.holdIndefinitely]  suppress intra-family drift
   */
  setPose(poseName, opts = {}) {
    const family = getFamily(poseName);
    if (!family) {
      console.warn('[pose-orchestrator] unknown pose:', poseName);
      return;
    }
    this.activeFamily = family;
    this._holdIndefinitely = !!opts.holdIndefinitely;
    this._enterPose(poseName, opts);
  }

  /**
   * Internal — begin a transition to `poseName`, routing through a
   * `via` waypoint if either endpoint declares one and the source +
   * destination aren't in the same family. The waypoint queue is
   * drained one step at a time inside tick() (each leg runs the full
   * interSlerpMs so via-routed transitions take roughly 2x a direct
   * slerp — a fair price to keep hands out of the torso).
   */
  _enterPose(poseName, opts = {}) {
    const slerpMs = opts.transitionMs ?? this.interSlerpMs;
    const path = opts.skipVia ? [poseName] : this._resolveTransitionPath(poseName);
    if (path.length > 1) {
      this._viaQueue = path.slice(1);
      this._viaSlerpMs = slerpMs;
      this._beginTransition(path[0], slerpMs);
    } else {
      this._viaQueue = null;
      this._viaSlerpMs = 0;
      this._beginTransition(poseName, slerpMs);
    }
  }

  /**
   * Decide whether a transition needs a via waypoint, and which one.
   * Returns `[toName]` for direct slerp, or `[viaName, toName]` for
   * a routed two-step transition.
   *
   * Routing rules:
   *   - First activation (no activeMember) → direct, since we're
   *     slerping from the live VRM state which is presumably sane.
   *   - Same pose / same family → direct (intra-family drift is
   *     always safe by family-design contract).
   *   - Either endpoint declares `via: [...]` → route through one
   *     of those waypoints (random pick, excluding from/to to avoid
   *     no-op steps).
   */
  _resolveTransitionPath(toName) {
    if (!this.activeMember) return [toName];
    if (this.activeMember === toName) return [toName];
    const fromPose = getPose(this.activeMember);
    const toPose = getPose(toName);
    if (fromPose?.family && fromPose.family === toPose?.family) return [toName];
    // If the current pose is already on the destination's via list, it
    // IS the bridge — direct slerp is safe, no double-bridging needed.
    // Same logic when leaving a via-declared pose toward something on
    // its own via list.
    if (toPose?.via?.includes(this.activeMember)) return [toName];
    if (fromPose?.via?.includes(toName)) return [toName];
    const viaList = (toPose?.via?.length ? toPose.via : fromPose?.via) || [];
    const candidates = viaList.filter((p) => p !== this.activeMember && p !== toName);
    if (candidates.length === 0) return [toName];
    const pick = candidates[Math.floor(Math.random() * candidates.length)];
    return [pick, toName];
  }

  /**
   * Advance the state machine. Call once per frame from the render loop.
   * Returns true if the pose changed (caller may want to re-evaluate
   * whether to apply, e.g. skip while a higher-priority layer like a
   * VRMA is active).
   */
  tick(nowMs) {
    nowMs = nowMs ?? this._now();
    let changed = false;

    // Advance active slerp
    if (this.transitionDur > 0) {
      const elapsed = nowMs - this.transitionStart;
      const t = Math.min(1, elapsed / this.transitionDur);
      const eased = _easeInOutCubic(t);
      for (const bone of this._activeBoneKeys) {
        const from = this._fromQuats[bone];
        const to = this._toQuats[bone];
        if (!from || !to) continue;
        if (!this._currentQuats[bone]) {
          this._currentQuats[bone] = new this._THREE.Quaternion();
        }
        this._currentQuats[bone].copy(from).slerp(to, eased);
      }
      changed = true;
      if (t >= 1) {
        this.transitionDur = 0;
        // Via routing: if a waypoint queue is in progress, immediately
        // start the next leg. Bypasses dwell + sequence + family drift
        // so the via is a transparent "stop on the way" rather than a
        // pose the avatar settles into.
        if (this._viaQueue?.length) {
          const next = this._viaQueue.shift();
          this._beginTransition(next, this._viaSlerpMs);
          return changed;
        }
        // Schedule next step. Sequence mode trumps family drift.
        if (this._sequence) {
          const dwell = this._sequence.dwellMs
            ?? _randInt(this.dwellRange[0], this.dwellRange[1]);
          this.dwellUntil = nowMs + dwell;
        } else {
          const members = getFamilyMembers(this.activeFamily);
          if (members.length > 1 && !this._holdIndefinitely) {
            const dwell = _randInt(this.dwellRange[0], this.dwellRange[1]);
            this.dwellUntil = nowMs + dwell;
          } else {
            this.dwellUntil = 0;
          }
        }
      }
    }

    // Trigger next step when the dwell window expires
    if (this.dwellUntil > 0 && nowMs >= this.dwellUntil && this.transitionDur === 0) {
      if (this._sequence) {
        // Sequence: advance index, looping if configured
        const seq = this._sequence;
        seq.index += 1;
        if (seq.index >= seq.list.length) {
          if (seq.loop) {
            seq.index = 0;
          } else {
            this._sequence = null;
            this.dwellUntil = 0;
            return changed;
          }
        }
        this._beginTransition(seq.list[seq.index], seq.slerpMs);
      } else {
        // Family drift: pick a random non-current member
        const members = getFamilyMembers(this.activeFamily).filter((m) => m !== this.activeMember);
        if (members.length) {
          const next = members[Math.floor(Math.random() * members.length)];
          this._beginTransition(next, this.intraSlerpMs);
        }
      }
      if (!this._sequence) this.dwellUntil = 0;
    }

    return changed;
  }

  /**
   * Snapshot of the current per-bone state. Returns:
   *   {
   *     boneName: { quaternion: THREE.Quaternion, euler: [x,y,z], order: 'XYZ' },
   *     ...
   *     _avatarPosition?: [x,y,z],
   *     _hipsTranslation?: [x,y,z],
   *   }
   *
   * Quaternion is returned by reference (do not mutate). Euler is
   * decomposed in the bone's canonical rotation order so the caller
   * can hand it directly to `node.rotation.set(x, y, z, order)`.
   *
   * Metadata (avatar position / hip translation) follows the active
   * member — during a transition, it snaps at t=0.5 rather than
   * interpolating, since translating a couch isn't meaningful.
   */
  getCurrentPose() {
    const out = {};
    for (const bone of this._activeBoneKeys) {
      const q = this._currentQuats[bone];
      if (!q) continue;
      const order = getBoneRotationOrder(bone);
      this._scratchEuler.setFromQuaternion(q, order);
      out[bone] = {
        quaternion: q,
        euler: [this._scratchEuler.x, this._scratchEuler.y, this._scratchEuler.z],
        order,
      };
    }
    // Metadata interpolation. Each pose may declare _avatarPosition
    // (avatar root world position) and _hipsTranslation (bone-relative
    // hip offset) for sitting / reclined / kneeling poses. To avoid
    // the "stuck in the seat" bug when transitioning out of a pose
    // that authored these, the orchestrator FALLS BACK to constructor-
    // provided defaults (typically scene's standing position + zero
    // hip offset) and lerps between them across the slerp duration.
    const fromPreset = this.previousMember ? getPose(this.previousMember) : null;
    const toPreset = this.activeMember ? getPose(this.activeMember) : null;

    const fromAvatar = fromPreset?._avatarPosition || this._defaultAvatarPosition;
    const toAvatar = toPreset?._avatarPosition || this._defaultAvatarPosition;
    if (fromAvatar && toAvatar) {
      out._avatarPosition = this._lerpVec3(fromAvatar, toAvatar);
    } else if (toAvatar) {
      out._avatarPosition = toAvatar;
    }

    const fromHips = fromPreset?._hipsTranslation || this._defaultHipsTranslation;
    const toHips = toPreset?._hipsTranslation || this._defaultHipsTranslation;
    if (fromHips && toHips) {
      out._hipsTranslation = this._lerpVec3(fromHips, toHips);
    } else if (toHips) {
      out._hipsTranslation = toHips;
    }
    return out;
  }

  // Linear interpolate two 3-vectors using the same eased timeline
  // as the bone slerp, so position + rotation animate as one motion.
  _lerpVec3(a, b) {
    if (this.transitionDur <= 0) return b;
    const t = Math.min(1, (this._now() - this.transitionStart) / this.transitionDur);
    const eased = _easeInOutCubic(t);
    return [
      a[0] + (b[0] - a[0]) * eased,
      a[1] + (b[1] - a[1]) * eased,
      a[2] + (b[2] - a[2]) * eased,
    ];
  }

  // Apply per-VRM axis sign correction to an authored Euler triple.
  // Finger bones use _fingerAxisSign, arm bones use _armAxisSign; body
  // bones pass through unchanged (they don't mirror across the midline).
  // Identity factors are skipped early so the common-case bundled-roster
  // VRM pays no cost.
  _correctEuler(boneName, euler) {
    const sign = isFingerBoneName(boneName)
      ? this._fingerAxisSign
      : (isArmBoneName(boneName) ? this._armAxisSign : null);
    if (!sign || (sign.x === 1 && sign.y === 1 && sign.z === 1)) return euler;
    return [euler[0] * sign.x, euler[1] * sign.y, euler[2] * sign.z];
  }

  /**
   * True while the orchestrator owns bones — either holding a member
   * or slerping (including the release-to-identity slerp). Use this
   * (not `activeMember`) to gate the per-frame tick + apply, otherwise
   * release() will appear to "freeze" the avatar at its last pose.
   */
  isActive() {
    return !!this.activeMember || this.transitionDur > 0;
  }

  /** Diagnostic — current state for debug HUD. */
  getDebugState() {
    const progress = this.transitionDur > 0
      ? Math.min(1, (this._now() - this.transitionStart) / this.transitionDur)
      : 1;
    return {
      activeFamily: this.activeFamily,
      activeMember: this.activeMember,
      previousMember: this.previousMember,
      transitioning: this.transitionDur > 0,
      progress,
      msUntilDrift: this.dwellUntil > 0 ? Math.max(0, this.dwellUntil - this._now()) : null,
    };
  }

  /**
   * Play an explicit sequence of poses in order. Bypasses random member
   * selection — the orchestrator walks the list step by step, slerping
   * to each in turn after the dwell window. Useful for authoring chain
   * loops (e.g. "look around" → straight → left → straight → right →
   * straight) where the order matters.
   *
   * @param {string[]} poseNames    sequence to walk
   * @param {object} [opts]
   * @param {boolean} [opts.loop]   restart after the last member (default true)
   * @param {number}  [opts.slerpMs]  override per-step slerp duration
   * @param {number}  [opts.dwellMs]  override per-step dwell (no random range)
   */
  playSequence(poseNames, opts = {}) {
    if (!Array.isArray(poseNames) || poseNames.length === 0) {
      this._sequence = null;
      return;
    }
    this._sequence = {
      list: poseNames.slice(),
      index: 0,
      loop: opts.loop !== false,
      slerpMs: opts.slerpMs ?? this.intraSlerpMs,
      dwellMs: opts.dwellMs ?? null,  // null => use this.dwellRange
    };
    // Clear any active family (sequence overrides random walk) and
    // start the first step immediately.
    this.activeFamily = null;
    this._holdIndefinitely = false;
    this._beginTransition(poseNames[0], opts.slerpMs ?? this.interSlerpMs);
  }

  /** Stop a running sequence. Holds at the current member. */
  stopSequence() {
    this._sequence = null;
    this.dwellUntil = 0;
  }

  /**
   * Reset to neutral. Active member becomes null; bones drift back to
   * identity over `transitionMs`. Use when an external system (VRMA)
   * takes over and the orchestrator should release control.
   */
  release(opts = {}) {
    if (!this.activeMember) return;
    const transitionMs = opts.transitionMs ?? this.interSlerpMs;
    const identity = new this._THREE.Quaternion();
    for (const bone of this._activeBoneKeys) {
      this._fromQuats[bone] = (this._currentQuats[bone] || new this._THREE.Quaternion()).clone();
      this._toQuats[bone] = identity.clone();
    }
    this.previousMember = this.activeMember;
    this.activeMember = null;
    this.activeFamily = null;
    this.transitionStart = this._now();
    this.transitionDur = transitionMs;
    this.dwellUntil = 0;
  }

  // ─── Internal ────────────────────────────────────────────────────────

  _beginTransition(toName, durMs) {
    const toPose = getPose(toName);
    if (!toPose) return;
    const toBones = toPose.bones || {};

    // _activeBoneKeys has two categories:
    //   1. Bones the new pose AUTHORS — slerped to the authored value.
    //   2. Bones the previous pose authored that the new pose DOESN'T —
    //      released toward rest pose / identity.
    //
    // Carry-over bones (hands, fingers) in category 2 are EXCLUDED:
    // the orchestrator releases ownership rather than holding them at
    // a stale value. This is what lets the user click a body pose +
    // hand pose in sequence without applyHandPose's writes getting
    // overwritten by the orchestrator each frame for the slerp duration.
    const toBoneKeys = new Set(Object.keys(toBones));
    const newKeys = new Set(toBoneKeys);
    const prevBones = this.activeMember ? (getPose(this.activeMember)?.bones || {}) : {};
    for (const k of Object.keys(prevBones)) {
      if (toBoneKeys.has(k)) continue;          // already added (category 1)
      if (CARRY_OVER_BONES.has(k)) continue;    // release ownership
      newKeys.add(k);                            // release toward rest / identity
    }

    // Snapshot current as FROM. Priority order:
    //   1. boneStateProvider(bone) — live VRM bone state. Authoritative,
    //      since other systems (hand-pose presets, IK, breath) may have
    //      written to the bone since our last frame. Reading from the
    //      cache would yank carry-over bones back to a stale value
    //      (visible as: pick body pose → click hand preset → pick body
    //      pose, hand drifts back over the slerp).
    //   2. _currentQuats[bone] — fallback when no provider is wired.
    //   3. identity — final fallback.
    for (const bone of newKeys) {
      if (!this._fromQuats[bone]) this._fromQuats[bone] = new this._THREE.Quaternion();
      const live = this._boneStateProvider?.(bone);
      if (live) {
        this._fromQuats[bone].copy(live);
      } else if (this._currentQuats[bone]) {
        this._fromQuats[bone].copy(this._currentQuats[bone]);
      } else {
        this._fromQuats[bone].identity();
      }
    }

    // Build TO quaternions. Bones in newKeys are either authored by
    // the new pose, or being released (carry-over bones were filtered
    // out above so they can't reach this branch). Eulers are per-bone-
    // class sign-corrected before quaternion construction so VRMs in
    // either arm/finger axis convention get the right visual.
    for (const bone of newKeys) {
      if (!this._toQuats[bone]) this._toQuats[bone] = new this._THREE.Quaternion();
      const euler = toBones[bone];
      if (euler) {
        const order = getBoneRotationOrder(bone);
        const corrected = this._correctEuler(bone, euler);
        this._scratchEuler.set(corrected[0], corrected[1], corrected[2], order);
        this._toQuats[bone].setFromEuler(this._scratchEuler);
      } else {
        // Release toward rest pose if known, else identity. The rest
        // pose target keeps shoulder droop / head tilt / etc. from
        // flattening to T-pose during transitions where neither
        // endpoint authored those bones.
        const rest = this._restPose?.[bone];
        if (rest) {
          const order = getBoneRotationOrder(bone);
          const corrected = this._correctEuler(bone, rest);
          this._scratchEuler.set(corrected[0], corrected[1], corrected[2], order);
          this._toQuats[bone].setFromEuler(this._scratchEuler);
        } else {
          this._toQuats[bone].identity();
        }
      }
      // Seed _currentQuats so the first frame has something to slerp from.
      if (!this._currentQuats[bone]) {
        this._currentQuats[bone] = this._fromQuats[bone].clone();
      }
    }

    this._activeBoneKeys = newKeys;
    this.previousMember = this.activeMember;
    this.activeMember = toName;
    this.transitionStart = this._now();
    this.transitionDur = durMs;
  }
}
