/**
 * motion-engine.js — channel-based motion composition engine for the
 * VRM avatar. Wraps existing motion sources (springs, VRMA, affordances,
 * pose primitives) as priority-ordered channels with masks + blend modes,
 * and adds biomechanically-shaped pose transitions.
 *
 * See docs/superpowers/specs/2026-05-14-motion-engine-design.md for the
 * architecture rationale.
 *
 * Usage pattern:
 *   const engine = new MotionEngine({ three: THREE, vrm });
 *   engine.addChannel(new PoseTransitionChannel({ ... }));
 *   engine.addChannel(new SpringChannel({ animator: existingAvatarAnimator }));
 *   engine.addChannel(new GazeChannel({ vrm }));
 *
 *   // Per-frame:
 *   engine.update(performance.now(), dt);
 *   engine.applyToVRM();
 *
 *   // Send a transition:
 *   engine.channel('pose').setTarget(thoughtfulChinPrimitive, { duration: 1200, energy: 0.5 });
 */

import {
  bioCurve, energyModulate, boneT, KINEMATIC_DELAYS, DEFAULT_BIO_CURVE,
} from './motion-curves.js';

// ─────────────────────────────────────────────────────────────────────────
// Tier vocabulary
//
// Channels declare a tier representing the SCOPE of their writes:
//
//   FACE       — morphs + eye targets. Lives on a separate "always alive"
//                track. Lipsync, mood expression, eye-look-at. Never
//                suppressed by body tiers; face stays animated through
//                VRMA, full-body poses, anything.
//   INDIVIDUAL — single bone writes (head pitch, finger fidget, single
//                eye saccade). Lowest body-tier priority.
//   IK_GROUP   — one limb chain (arm or leg). Per-arm primitives, IK
//                reach controllers.
//   FULL_BODY  — curated pose covering most of the body (idle_natural,
//                thoughtful_chin_touch, etc.).
//   VRMA       — animation clip; takes over the whole body for its
//                duration. Highest priority.
//
// Priority is derived from tier by default — higher tier writes later
// in the mixer pass and so wins composition for overlapping bones.
// Face (priority 400) sits between FULL_BODY (300) and VRMA (500) so
// it composes ON TOP of body work (gaze applies head delta after pose);
// VRMA sits highest because when an animation plays, it owns the body.
// ─────────────────────────────────────────────────────────────────────────
export const TIER = Object.freeze({
  FACE:       'face',
  INDIVIDUAL: 'individual',
  IK_GROUP:   'ik_group',
  FULL_BODY:  'full_body',
  VRMA:       'vrma',
});

const TIER_PRIORITY = Object.freeze({
  face:       400,
  individual: 100,
  ik_group:   200,
  full_body:  300,
  vrma:       500,
});

const TIER_ORDER = Object.freeze({
  face:       0,
  individual: 1,
  ik_group:   2,
  full_body:  3,
  vrma:       4,
});

// ─────────────────────────────────────────────────────────────────────────
// Base channel
// ─────────────────────────────────────────────────────────────────────────
export class MotionChannel {
  /**
   * @param {object} opts
   * @param {string} opts.name
   * @param {Iterable<string>} opts.mask       bones this channel may write
   * @param {number} [opts.priority]           higher = applied later (overrides lower)
   * @param {'override'|'compose-quat'|'weighted'} [opts.blendMode]
   * @param {number} [opts.weight]             0..1 for 'weighted' blend
   * @param {boolean} [opts.enabled]
   */
  constructor({ name, mask, priority, blendMode = 'override', weight = 1.0, enabled = true, tier = TIER.FULL_BODY }) {
    if (!name) throw new Error('MotionChannel requires name');
    this.name = name;
    this.mask = new Set(mask || []);
    this.tier = tier;
    // Priority derived from tier if not explicitly set. Subclasses can
    // override by passing priority directly.
    this.priority = priority !== undefined ? priority : (TIER_PRIORITY[tier] ?? 300);
    this.blendMode = blendMode;
    this.weight = weight;
    this.enabled = enabled;
  }

  /** Subclass override. Return Map<boneName, THREE.Quaternion>.
   *  Third arg is the in-progress mixer result so channels can read
   *  what lower-priority channels already wrote — useful for "compose
   *  on top of pose" semantics where breathing should multiply onto
   *  whatever the pose channel set, falling back to bind rest. */
  evaluate(_timeMs, _dt, _currentResult) {
    return new Map();
  }

  /** Optional: for channels that drive VRM state outside the bone bus
   *  (eye look-at proxies, expression blendshapes, etc.). Default no-op. */
  applyExtras(_vrm) {}
}

// ─────────────────────────────────────────────────────────────────────────
// Mixer
// ─────────────────────────────────────────────────────────────────────────
export class MotionEngine {
  /**
   * @param {object} opts
   * @param {object} opts.three   THREE namespace
   * @param {object} opts.vrm     VRM instance to apply final state to
   * @param {object} [opts.restQuats]  bind-rest quaternions per bone (defaults to vrm.__augmentumBoneRestQuats)
   */
  constructor({ three, vrm, restQuats = null }) {
    if (!three) throw new Error('MotionEngine requires opts.three');
    if (!vrm?.humanoid) throw new Error('MotionEngine requires opts.vrm with humanoid');
    this.three = three;
    this.vrm = vrm;
    this.restQuats = restQuats || vrm.__augmentumBoneRestQuats || null;
    /** @type {Map<string, MotionChannel>} */
    this.channels = new Map();
    this._scratchA = new three.Quaternion();
    this._scratchB = new three.Quaternion();
    this._lastFrame = null;     // boneName → Quaternion (carried across frames for stability)
  }

  /** Register a channel. Replaces any existing channel with the same name. */
  addChannel(channel) {
    this.channels.set(channel.name, channel);
    return channel;
  }

  /** Remove a channel by name. */
  removeChannel(name) {
    this.channels.delete(name);
  }

  /** Look up a channel by name. */
  channel(name) {
    return this.channels.get(name);
  }

  /** Tick all enabled channels and store the merged result internally. */
  update(timeMs, dt) {
    const ordered = [...this.channels.values()]
      .filter(c => c.enabled)
      .sort((a, b) => a.priority - b.priority);

    const result = new Map();    // boneName → Quaternion (allocated lazily)

    for (const ch of ordered) {
      let contrib;
      try {
        contrib = ch.evaluate(timeMs, dt, result);
      } catch (err) {
        console.warn(`[motion-engine] channel '${ch.name}' evaluate threw:`, err);
        continue;
      }
      if (!contrib) continue;
      for (const [boneName, q] of contrib) {
        if (!ch.mask.has(boneName)) continue;
        if (ch.blendMode === 'override') {
          // Replace whatever the lower-priority channel wrote.
          result.set(boneName, q.clone());
        } else if (ch.blendMode === 'compose-quat') {
          // rest × delta semantics: multiply this channel's quat onto the prior.
          const prior = result.get(boneName);
          if (prior) {
            result.set(boneName, prior.clone().multiply(q));
          } else {
            result.set(boneName, q.clone());
          }
        } else if (ch.blendMode === 'weighted') {
          const prior = result.get(boneName);
          const w = Math.max(0, Math.min(1, ch.weight));
          if (prior) {
            const blended = prior.clone().slerp(q, w);
            result.set(boneName, blended);
          } else {
            result.set(boneName, q.clone());
          }
        }
      }
    }

    this._lastFrame = result;

    // Let channels also write to extras (eye look-at proxy, blendshapes, etc.)
    for (const ch of ordered) {
      try {
        if (typeof ch.applyExtras === 'function') ch.applyExtras(this.vrm);
      } catch (err) {
        console.warn(`[motion-engine] channel '${ch.name}' applyExtras threw:`, err);
      }
    }
  }

  /** Push the merged bone state into the VRM. Call after update(). */
  applyToVRM() {
    if (!this._lastFrame) return;
    const humanoid = this.vrm.humanoid;
    for (const [boneName, q] of this._lastFrame) {
      const node = humanoid.getNormalizedBoneNode?.(boneName);
      if (!node) continue;
      node.quaternion.copy(q);
    }
  }

  /** Diagnostic: report channel-mask coverage + active priorities. */
  inspect() {
    const out = [];
    for (const ch of [...this.channels.values()].sort((a, b) => a.priority - b.priority)) {
      out.push({
        name: ch.name,
        priority: ch.priority,
        blendMode: ch.blendMode,
        weight: ch.weight,
        enabled: ch.enabled,
        maskSize: ch.mask.size,
      });
    }
    return out;
  }
}

// Bone groups used by PoseTransitionChannel. Per-side arm chain matches
// AvatarAffordanceApplier's ARM_CHAIN so we can snapshot affordance-derived
// targets accurately.
const FINGER_BONES_L = [
  'leftThumbProximal','leftThumbIntermediate','leftThumbDistal',
  'leftIndexProximal','leftIndexIntermediate','leftIndexDistal',
  'leftMiddleProximal','leftMiddleIntermediate','leftMiddleDistal',
  'leftRingProximal','leftRingIntermediate','leftRingDistal',
  'leftLittleProximal','leftLittleIntermediate','leftLittleDistal',
];
const FINGER_BONES_R = FINGER_BONES_L.map(n => n.replace(/^left/, 'right'));
const ARM_CHAIN_L = ['leftShoulder','leftUpperArm','leftLowerArm','leftHand', ...FINGER_BONES_L];
const ARM_CHAIN_R = ['rightShoulder','rightUpperArm','rightLowerArm','rightHand', ...FINGER_BONES_R];
const BODY_BONES = ['hips','spine','chest','upperChest','neck','head'];
const LEG_BONES = [
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
];
const ALL_TRANSITION_BONES = [...BODY_BONES, ...LEG_BONES, ...ARM_CHAIN_L, ...ARM_CHAIN_R];

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: PoseTransitionChannel
//
// Animates from current bone state to a target pose primitive over a
// duration using biomechanical motion curves + kinematic chain delays.
//
// Key behaviors:
//   - Captures CURRENT bone state as the per-bone start (works mid-flight)
//   - Resolves the primitive's body rotations into target quats
//   - For arms with a fingerStyle: applies the affordance temporarily,
//     snapshots the resulting arm-chain quats, restores original state.
//     The captured quats become tween targets — affordance arm/finger
//     bones now ride the biomechanical curve with kinematic delays
//     instead of snapping at t=0.5.
//   - Optional via-queue: opts.via is a list of intermediate primitives
//     to transition through before reaching the final target. Each
//     segment gets its proportional share of the total duration.
// ─────────────────────────────────────────────────────────────────────────
export class PoseTransitionChannel extends MotionChannel {
  /**
   * @param {object} opts
   * @param {object} opts.three         THREE namespace
   * @param {object} opts.vrm           VRM
   * @param {object} opts.resolver      PoseResolver instance
   * @param {object} [opts.applier]     AvatarAffordanceApplier for finger styles
   * @param {object} [opts.restQuats]   bind-rest quats
   * @param {Iterable<string>} [opts.mask]
   * @param {number} [opts.priority]    default 100
   */
  constructor(opts) {
    const mask = opts.mask || ALL_TRANSITION_BONES;
    super({
      name: opts.name || 'pose_transition',
      mask,
      priority: opts.priority,
      blendMode: 'override',
      tier: opts.tier || TIER.FULL_BODY,
    });
    this.three = opts.three;
    this.vrm = opts.vrm;
    this.resolver = opts.resolver;
    this.applier = opts.applier || null;
    this.restQuats = opts.restQuats || opts.vrm.__augmentumBoneRestQuats || null;
    /** BodyAtlas instance for path validation. Optional; if absent,
     *  transitions are unvalidated (legacy behavior). */
    this.bodyAtlas = opts.bodyAtlas || opts.vrm.__augmentumBodyAtlas || null;
    /** Primitives to try as via waypoints when the direct path is invalid.
     *  Tried in order; first clean route wins. Usually pre-loaded
     *  idle_natural (or any neutral arms-at-sides pose). */
    this.defaultViaCandidates = opts.defaultViaCandidates || [];
    /** SDF threshold for collision: hand world positions with sdf < this
     *  (i.e. that deep inside the body) flag the path as invalid. Defaults
     *  to -0.02 m (2cm inside) to ignore grazing contact while catching
     *  real penetration. */
    this.collisionThreshold = opts.collisionThreshold ?? -0.02;
    /** How many intermediate samples to evaluate per path validation. */
    this.validationSamples = opts.validationSamples ?? 16;
    /** Tracking: was the most recent transition auto-routed and how?
     *  `reason` ∈ {'explicit-via', 'same-family', 'declared-via', 'sdf-clean',
     *  'sdf-routed', 'forced-direct', 'no-atlas'}. */
    this.lastRoute = { via: null, reason: 'idle', collisions: 0 };
    /** The primitive most recently applied — used for family comparison. */
    this._lastPrimitive = null;
    /** Snapshot of bone state at channel-construction time, used as the
     *  "natural" pose target for `rest` anchors. Caller is expected to
     *  apply applyPosePreset(natural) before constructing the channel
     *  (see avatar.js::loadVRM and the bench setup), so this captures
     *  the arms-at-sides bind that idle_natural-style primitives mean. */
    this._naturalBoneState = new Map();
    for (const boneName of this.mask) {
      const node = opts.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) this._naturalBoneState.set(boneName, node.quaternion.clone());
    }
    /** Families that should NOT be trusted as same-family — too coarse
     *  for the family graph to guarantee safety. `per_arm` lumps all
     *  per-arm atoms together (chin / hip / chest / behind / etc.) which
     *  have wildly different end-spaces, so cross-region transitions
     *  inside this family genuinely need validation + via routing. */
    this._untrustedFamilies = new Set(opts.untrustedFamilies || ['per_arm']);

    /** @type {Map<string, THREE.Quaternion>} bone → start state at t0 */
    this._startState = new Map();
    /** @type {Map<string, THREE.Quaternion>} bone → target state */
    this._targetState = new Map();
    this._t0 = 0;
    this._duration = 0;
    this._energy = 0.5;
    this._curveParams = DEFAULT_BIO_CURVE;
    /** Set of bones the current transition is animating. */
    this._activeBones = new Set();
    /** Queue of primitives to transition through (via routing). Last is final target. */
    this._queue = [];
    /** Total duration for the entire queue (when via routing). */
    this._queueTotalDuration = 0;
    /** Persisted opts so dequeued segments inherit energy/etc. */
    this._lastOpts = null;

    this._scratchEuler = new opts.three.Euler();
    this._scratchQuat = new opts.three.Quaternion();
    this._scratchVec = new opts.three.Vector3();
  }

  /**
   * Begin a transition toward the resolved version of `primitive`.
   * @param {object} primitive
   * @param {object} [opts]
   * @param {number} [opts.duration]    transition length in ms (default 1000)
   * @param {number} [opts.energy]      0..1 (default 0.5)
   * @param {number} [opts.now]         override current timestamp (default performance.now())
   * @param {object[]} [opts.via]       primitives to transition THROUGH before
   *                                    reaching `primitive`. Each via segment
   *                                    gets a share of the total duration.
   */
  setTarget(primitive, opts = {}) {
    this._lastOpts = opts;
    const explicitVia = Array.isArray(opts.via) ? opts.via : null;

    // ─── Tier 1: explicit via from caller (always trusted, never validated) ─
    if (explicitVia && explicitVia.length > 0) {
      this._queue = [...explicitVia, primitive];
      this._queueTotalDuration = Math.max(200, opts.duration ?? 1000);
      this.lastRoute = {
        via: explicitVia.map(p => p?.name).filter(Boolean),
        reason: 'explicit-via', collisions: 0,
      };
      this._beginNextSegment();
      this._lastPrimitive = primitive;
      return;
    }

    // Begin direct segment so _startState/_targetState are populated for
    // validation. We may queue+rebegin below if a via is needed.
    this._queue = [];
    this._beginSegment(primitive, opts);

    // ─── Tier 2: same-family transitions are trusted as safe ────────────
    // The family graph in POSE_PRESETS encodes pre-curated knowledge
    // that poses sharing a family share an arms-down basin — slerping
    // between them won't collide. Skip validation for these to save cycles
    // AND respect curated intent. EXCEPT for untrusted families like
    // `per_arm` which lump too many disparate end-spaces together to
    // guarantee safe direct slerps.
    const fromFamily = this._lastPrimitive?.family;
    const toFamily = primitive?.family;
    const sameFamily = fromFamily && toFamily && fromFamily === toFamily
                       && !this._untrustedFamilies.has(fromFamily);
    if (sameFamily) {
      this.lastRoute = { via: null, reason: 'same-family', collisions: 0 };
      this._lastPrimitive = primitive;
      return;
    }

    // ─── Tier 3: primitive declares transitionVia — use those names ─────
    // The pose primitive can declare `transitionVia: [name, ...]` listing
    // waypoint primitives to route through. We resolve those names against
    // the channel's defaultViaCandidates pool (which the caller pre-loaded).
    const declaredViaNames = Array.isArray(primitive?.transitionVia) ? primitive.transitionVia : [];
    if (declaredViaNames.length > 0 && this.defaultViaCandidates.length > 0) {
      for (const name of declaredViaNames) {
        const candidate = this.defaultViaCandidates.find(p => p?.name === name);
        if (!candidate) continue;
        // Skip if we're already in this via's family (avoid pointless detour).
        if (candidate.family && candidate.family === fromFamily) continue;
        this._queue = [candidate, primitive];
        this._queueTotalDuration = Math.max(200, opts.duration ?? 1000);
        this._beginNextSegment();
        this.lastRoute = {
          via: [candidate.name], reason: 'declared-via', collisions: 0,
        };
        this._lastPrimitive = primitive;
        return;
      }
    }

    // ─── Tier 4: atlas SDF validation (safety net for un-curated paths) ─
    const skipValidation = opts.skipValidation === true || !this.bodyAtlas;
    if (skipValidation) {
      this.lastRoute = { via: null, reason: 'no-atlas', collisions: 0 };
      this._lastPrimitive = primitive;
      return;
    }

    const directValidation = this._validatePath();
    if (directValidation.valid) {
      this.lastRoute = { via: null, reason: 'sdf-clean', collisions: 0 };
      this._lastPrimitive = primitive;
      return;
    }

    // Direct collides — try defaultViaCandidates.
    const candidates = (opts.viaCandidates && opts.viaCandidates.length)
      ? opts.viaCandidates
      : this.defaultViaCandidates;
    for (const candidate of candidates) {
      if (!candidate || candidate.name === primitive?.name) continue;
      this._queue = [candidate, primitive];
      this._queueTotalDuration = Math.max(200, opts.duration ?? 1000);
      this._beginNextSegment();
      const viaValidation = this._validatePath();
      if (viaValidation.valid) {
        this.lastRoute = {
          via: [candidate.name],
          reason: 'sdf-routed',
          collisions: directValidation.failedSamples.length,
        };
        console.debug(
          `[pose-transition] direct path had ${directValidation.failedSamples.length} collisions; routing via '${candidate.name}'`
        );
        this._lastPrimitive = primitive;
        return;
      }
    }

    // Nothing worked — direct path with warning.
    console.warn(
      `[pose-transition] no collision-free route found (${directValidation.failedSamples.length} direct collisions); proceeding with direct`
    );
    this._queue = [];
    this._beginSegment(primitive, opts);
    this.lastRoute = {
      via: null,
      reason: 'forced-direct',
      collisions: directValidation.failedSamples.length,
    };
    this._lastPrimitive = primitive;
  }

  /**
   * Validate the currently-committed transition by sampling intermediate
   * t values along the planned bioCurve. At each sample, the channel's
   * bone state is set to the interpolated values, the scene matrix is
   * updated, and the hand world positions are queried against the body
   * atlas SDF. Returns {valid, failedSamples}.
   *
   * Side effects: mutates VRM bone state during sampling, then restores
   * the original state. Within a single setTarget call this is invisible
   * to the renderer (no frame ticks between mutate-and-restore).
   * @private
   */
  _validatePath() {
    if (!this.bodyAtlas) return { valid: true, failedSamples: [] };
    if (this._activeBones.size === 0) return { valid: true, failedSamples: [] };

    // Save the current bone state so we can restore after sampling.
    const saved = new Map();
    for (const boneName of this._activeBones) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) saved.set(boneName, node.quaternion.clone());
    }

    const N = this.validationSamples;
    const failedSamples = [];
    const tmpVec = this._scratchVec;
    const handNames = ['leftHand', 'rightHand'];

    // World↔bake frame for the avatar's CURRENT pose, so the SDF queries below
    // stay correct when she's turned (vrm.scene.rotation.y) or scaled. Hips is
    // not an active (arm) bone, so its world pose is stable across the sampling
    // mutations in this loop — build the frame once up front.
    const frame = (() => {
      const hips = this.vrm.humanoid.getNormalizedBoneNode?.('hips');
      if (!hips) return this.bodyAtlas.frame(this.bodyAtlas.bakeHipsPos, this.bodyAtlas.bakeHipsQuat, 1);
      const hp = hips.getWorldPosition(new this.three.Vector3());
      const hq = hips.getWorldQuaternion(new this.three.Quaternion());
      const hs = hips.getWorldScale(new this.three.Vector3());
      return this.bodyAtlas.frame([hp.x, hp.y, hp.z], [hq.x, hq.y, hq.z, hq.w], hs.x);
    })();

    // Sample t ∈ (0, 1) — strictly interior. Endpoints are the start and
    // target poses which are by definition the user's intended valid states.
    for (let i = 1; i < N; i++) {
      const t = i / N;

      // Set every active bone to its bioCurve-interpolated quaternion at this t.
      for (const boneName of this._activeBones) {
        const start = this._startState.get(boneName);
        const target = this._targetState.get(boneName);
        if (!start || !target) continue;
        const tb = boneT(boneName, t, this._duration);
        const f = bioCurve(tb, this._curveParams);
        const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
        if (!node) continue;
        // Use start.slerp(target, f) which accepts f outside [0,1] via extrapolation.
        node.quaternion.copy(start).slerp(target, f);
      }
      // Force the scene matrix to reflect this sample.
      this.vrm.scene.updateMatrixWorld(true);

      // Query hand positions.
      for (const handName of handNames) {
        const node = this.vrm.humanoid.getNormalizedBoneNode?.(handName);
        if (!node) continue;
        node.getWorldPosition(tmpVec);
        const sdf = frame.sdf([tmpVec.x, tmpVec.y, tmpVec.z]);
        if (sdf < this.collisionThreshold) {
          failedSamples.push({
            t, hand: handName, sdf,
            pos: [tmpVec.x, tmpVec.y, tmpVec.z],
          });
        }
      }
    }

    // Restore original bone state.
    for (const [boneName, quat] of saved) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) node.quaternion.copy(quat);
    }
    this.vrm.scene.updateMatrixWorld(true);

    return { valid: failedSamples.length === 0, failedSamples };
  }

  /** Internal: pull the next segment from the queue and begin it. Each
   *  segment gets a proportional share of `_queueTotalDuration`. */
  _beginNextSegment() {
    if (this._queue.length === 0) return;
    const remaining = this._queue.length;
    const next = this._queue.shift();
    // Earlier segments are FAST waypoints (~40% share each), final is the
    // dominant segment. Approximation: split duration evenly, then bias
    // toward the final segment.
    const segmentShare = this._queueTotalDuration / (remaining * 0.85);
    this._beginSegment(next, {
      ...this._lastOpts,
      duration: segmentShare,
    });
  }

  /** Internal: start a single transition segment toward `primitive`. */
  _beginSegment(primitive, { duration = 1000, energy = 0.5, now } = {}) {
    const t0 = (now ?? performance.now());
    const resolved = this.resolver.resolve(primitive);

    // Snapshot current bone state — these are our start positions.
    this._startState.clear();
    for (const boneName of this.mask) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) this._startState.set(boneName, node.quaternion.clone());
    }

    this._targetState.clear();
    this._activeBones.clear();

    // ─── Body bones (resolver.bones — Eulers relative to rest) ──────────
    for (const boneName of this.mask) {
      const eulerArr = resolved.bones[boneName];
      if (eulerArr) {
        this._scratchEuler.set(eulerArr[0], eulerArr[1], eulerArr[2], 'XYZ');
        this._scratchQuat.setFromEuler(this._scratchEuler);
        const rest = this.restQuats?.[boneName];
        const target = rest
          ? rest.clone().multiply(this._scratchQuat)
          : this._scratchQuat.clone();
        this._targetState.set(boneName, target);
        this._activeBones.add(boneName);
      }
    }

    // ─── Arm-chain bones from affordances (apply-snapshot-restore) ──────
    // For each side with a fingerStyle, we want the affordance's authored
    // arm + finger bone rotations to be TARGETS for our biomechanical
    // tween — NOT applied at t=0.5 as a snap. We accomplish this by
    // temporarily applying the affordance, snapshotting the resulting
    // bone state, then restoring pre-affordance state. The captured quats
    // join the target set and ride the bioCurve like any other bone.
    if (this.applier) {
      for (const side of ['L', 'R']) {
        const style = resolved.fingerStyles[side];
        if (!style) continue;
        const sideChain = side === 'L' ? ARM_CHAIN_L : ARM_CHAIN_R;
        // Pre-snapshot original state for restore.
        const preSnap = new Map();
        for (const boneName of sideChain) {
          const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
          if (node) preSnap.set(boneName, node.quaternion.clone());
        }
        // Apply affordance (writes to bones).
        let appliedOk = false;
        try {
          appliedOk = this.applier.setHandPose(side, style);
        } catch (err) {
          console.warn(`[pose-transition] affordance '${style}' failed:`, err);
        }
        if (appliedOk) {
          // Capture post-write state as targets.
          for (const boneName of sideChain) {
            const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
            if (!node) continue;
            this._targetState.set(boneName, node.quaternion.clone());
            this._activeBones.add(boneName);
          }
        }
        // Restore pre-affordance state — the tween will animate INTO
        // the captured target from the current (pre-affordance) state.
        for (const [boneName, q] of preSnap) {
          const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
          if (node) node.quaternion.copy(q);
        }
      }
    }

    // ─── Arm-chain bones from `rest` anchors (drive arm to natural pose) ─
    // A `rest` anchor on an arm means "release this arm to its natural
    // arms-at-sides position." We tween the arm chain bones to the
    // snapshotted natural state. This is what makes `idle_natural` (with
    // rest anchors on both arms) actually do something when used as a
    // via waypoint — without this, the via segment was a no-op.
    for (const [anchorKey, sideChain] of [
      ['leftHand', ARM_CHAIN_L],
      ['rightHand', ARM_CHAIN_R],
    ]) {
      const anchor = primitive.anchors?.[anchorKey];
      if (anchor?.type !== 'rest') continue;
      for (const boneName of sideChain) {
        // Affordance already set a target for this bone — skip.
        if (this._targetState.has(boneName)) continue;
        const natural = this._naturalBoneState.get(boneName);
        if (natural) {
          this._targetState.set(boneName, natural.clone());
          this._activeBones.add(boneName);
        }
      }
    }

    // ─── Fill in pass-through targets for masked bones not actively
    // tweened (keeps them at their current state instead of identity) ───
    for (const boneName of this.mask) {
      if (!this._targetState.has(boneName)) {
        const start = this._startState.get(boneName);
        if (start) this._targetState.set(boneName, start.clone());
      }
    }

    this._t0 = t0;
    this._duration = Math.max(200, duration);
    this._energy = energy;
    const mod = energyModulate(energy);
    this._curveParams = mod.params;
    this._duration *= mod.durationMultiplier;

    // Ensure duration covers the slowest bone in the active chain.
    let maxDelay = 0;
    for (const b of this._activeBones) {
      const d = KINEMATIC_DELAYS[b] ?? 100;
      if (d > maxDelay) maxDelay = d;
    }
    this._duration = Math.max(this._duration, maxDelay + 300);
  }

  /** Stop the current transition; channel becomes pass-through (no writes). */
  stop() {
    this._t0 = 0;
    this._duration = 0;
    this._queue = [];
    this._activeBones.clear();
  }

  /**
   * Tween EVERY mask bone (body + arms + legs + fingers) back to the
   * snapshotted natural-pose state. The dedicated way to recover from
   * a VRMA that ended in a non-neutral position (legs splayed, arms up,
   * body rotated). After this completes, the avatar is back at the
   * canonical idle pose ready for the next primitive.
   *
   * @param {object} [opts]
   * @param {number} [opts.duration]  default 700ms (quicker than idle transitions)
   * @param {number} [opts.energy]    default 0.5
   * @param {number} [opts.now]       override clock
   */
  restoreToNatural({ duration = 700, energy = 0.5, now } = {}) {
    const t0 = (now ?? performance.now());
    this._lastOpts = { duration, energy };
    this._queue = [];

    // Snapshot start state for everything in mask.
    this._startState.clear();
    this._targetState.clear();
    this._activeBones.clear();
    for (const boneName of this.mask) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (!node) continue;
      this._startState.set(boneName, node.quaternion.clone());
      const natural = this._naturalBoneState.get(boneName);
      if (natural) {
        this._targetState.set(boneName, natural.clone());
        // Only activate bones whose current state actually differs from
        // natural — avoids wasting curve evaluation on bones that are
        // already at rest.
        if (!node.quaternion.equals(natural)) {
          this._activeBones.add(boneName);
        }
      }
    }
    this._t0 = t0;
    this._duration = Math.max(200, duration);
    this._energy = energy;
    const mod = energyModulate(energy);
    this._curveParams = mod.params;
    this._duration *= mod.durationMultiplier;

    // Ensure duration covers the slowest bone delay.
    let maxDelay = 0;
    for (const b of this._activeBones) {
      const d = KINEMATIC_DELAYS[b] ?? 100;
      if (d > maxDelay) maxDelay = d;
    }
    this._duration = Math.max(this._duration, maxDelay + 300);
    this.lastRoute = { via: null, reason: 'restore-to-natural', collisions: 0 };
  }

  evaluate(timeMs, _dt) {
    const result = new Map();
    if (this._duration <= 0) return result;

    const dt = timeMs - this._t0;
    const t = Math.max(0, Math.min(1, dt / this._duration));

    // Per-bone curve sampling
    for (const boneName of this._activeBones) {
      const start = this._startState.get(boneName);
      const target = this._targetState.get(boneName);
      if (!start || !target) continue;
      const tb = boneT(boneName, t, this._duration);
      const f = bioCurve(tb, this._curveParams);
      // f can go negative (anticipation) or >1 (overshoot); three.js slerp
      // extrapolates via its second-arg interpolation factor.
      const blended = start.clone().slerp(target, f);
      result.set(boneName, blended);
    }

    if (t >= 1) {
      // Segment complete — snap to target.
      for (const boneName of this._activeBones) {
        const target = this._targetState.get(boneName);
        if (target) result.set(boneName, target.clone());
      }
      this._duration = 0;
      this._activeBones.clear();
      // Dequeue next via segment if any.
      if (this._queue.length > 0) {
        this._beginNextSegment();
      }
    }

    return result;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: GazeChannel
//
// Drives eye look-at via VRMLookAtQuaternionProxy. Optionally co-rotates
// the head a fraction toward the target.
// ─────────────────────────────────────────────────────────────────────────
export class GazeChannel extends MotionChannel {
  /**
   * @param {object} opts
   * @param {object} opts.three
   * @param {object} opts.vrm
   * @param {number} [opts.headInfluence]  0..1, how much the head co-rotates
   * @param {number} [opts.smoothing]      0..1 frame smoothing factor
   * @param {number} [opts.priority]       default 300
   */
  constructor(opts) {
    super({
      name: opts.name || 'gaze',
      mask: opts.headInfluence > 0 ? ['head'] : [],
      priority: opts.priority,
      blendMode: 'compose-quat',
      tier: opts.tier || TIER.FACE,
    });
    this.three = opts.three;
    this.vrm = opts.vrm;
    this.headInfluence = opts.headInfluence ?? 0.2;
    this.smoothing = opts.smoothing ?? 0.15;
    /** @type {THREE.Vector3 | null} World-space gaze target. */
    this._target = null;
    /** Smoothed direction vector. */
    this._smoothedDir = new opts.three.Vector3(0, 0, 1);
  }

  /** Set the gaze target in world coordinates. Pass null to release. */
  setTargetWorld(pos) {
    if (!pos) { this._target = null; return; }
    this._target = new this.three.Vector3(pos[0], pos[1], pos[2]);
  }

  evaluate(_timeMs, _dt) {
    const result = new Map();
    if (this.headInfluence <= 0 || !this._target) return result;
    // Compute head delta-rotation toward the target
    const head = this.vrm.humanoid.getNormalizedBoneNode?.('head');
    if (!head) return result;
    const headWorld = head.getWorldPosition(new this.three.Vector3());
    const desiredDir = this._target.clone().sub(headWorld);
    if (desiredDir.lengthSq() < 1e-6) return result;
    desiredDir.normalize();
    this._smoothedDir.lerp(desiredDir, this.smoothing);

    // Convert smoothedDir → small head rotation (yaw + pitch toward target)
    const yaw = Math.atan2(this._smoothedDir.x, this._smoothedDir.z);
    const pitch = Math.asin(Math.max(-1, Math.min(1, this._smoothedDir.y)));
    const e = new this.three.Euler(
      -pitch * this.headInfluence,
       yaw   * this.headInfluence,
      0,
      'YXZ'
    );
    const q = new this.three.Quaternion().setFromEuler(e);
    result.set('head', q);
    return result;
  }

  applyExtras(vrm) {
    // Drive the VRMLookAtQuaternionProxy if present, so eyes follow.
    if (!this._target || !vrm.lookAt) return;
    // VRMLookAt expects a target object in world space; we set its
    // position to match our target. The proxy attached at avatar load
    // sources gaze direction from there.
    const proxy = vrm.scene.getObjectByName?.('VRMLookAtQuaternionProxy');
    if (proxy) proxy.position.copy(this._target);
    // Force the lookAt update so the eye bones rotate this frame.
    try { vrm.lookAt.update?.(0); } catch {}
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: ExpressionChannel
//
// Writes VRM blendshape weights (joy, sorrow, angry, surprised, fun,
// lip morphs, etc.). Not a bone-state channel — mask is empty; uses
// applyExtras to write expression weights directly.
// ─────────────────────────────────────────────────────────────────────────
export class ExpressionChannel extends MotionChannel {
  constructor(opts = {}) {
    super({
      name: opts.name || 'expression',
      mask: [],
      priority: opts.priority,
      blendMode: 'override',
      tier: opts.tier || TIER.FACE,
    });
    this.vrm = opts.vrm;
    /** @type {Map<string, number>} expressionName → target weight */
    this._target = new Map();
    /** @type {Map<string, number>} expressionName → current weight (for smoothing) */
    this._current = new Map();
    // Slower smoothing → expressions ease in/out over ~1s rather than
    // snapping to target. Was 0.18 (too twitchy); 0.08 reads more
    // natural for emotional shifts.
    this.smoothing = opts.smoothing ?? 0.08;
    /** Per-expression amplitude scaling — multiplied onto every weight
     *  written for that name. Defaults to 1.0 across the board. Lets
     *  users tune individual expressions (e.g. boost a too-subtle smile). */
    this.amplitudeMul = opts.amplitudeMul || {};
    /** Cached set of names this VRM actually supports. Populated lazily. */
    this._knownNames = null;
    /** Aliases for VRM 0.x compatibility — three-vrm 3.x normalizes most
     *  of these, but a fallback layer here catches edge cases. */
    this._aliases = {
      happy: ['happy', 'joy', 'Happy', 'Joy'],
      sad: ['sad', 'sorrow', 'Sad', 'Sorrow'],
      angry: ['angry', 'Angry'],
      surprised: ['surprised', 'Surprised'],
      relaxed: ['relaxed', 'fun', 'Relaxed', 'Fun'],
    };
    /** Names we've already warned about being unrecognized (don't spam). */
    this._unknownWarned = new Set();
  }

  /** Set expression weights. Pass partial dict; unset expressions retain. */
  setExpressions(weights) {
    for (const [name, w] of Object.entries(weights)) {
      const mul = this.amplitudeMul[name] ?? 1.0;
      this._target.set(name, Math.max(0, Math.min(1, w * mul)));
    }
  }

  /** Clear an expression back to 0 (smoothly). */
  clearExpression(name) {
    this._target.set(name, 0);
  }

  /** Snap to a single expression at full intensity, clearing others. */
  setOnly(name, weight = 1) {
    this._target.clear();
    this._target.set(name, weight);
  }

  /** Lazily-cached list of expression names this VRM accepts. Tries
   *  the VRM 1.0 expressionMap and falls back to VRM 0.x. */
  availableExpressions() {
    if (this._knownNames) return this._knownNames;
    const set = new Set();
    const em = this.vrm?.expressionManager;
    if (em?.expressionMap) {
      for (const name of Object.keys(em.expressionMap)) set.add(name);
    }
    if (em?.expressions) {
      for (const e of em.expressions) {
        if (e?.expressionName) set.add(e.expressionName);
        if (e?.name) set.add(e.name);
      }
    }
    this._knownNames = set;
    return set;
  }

  /** Resolve a requested name through aliases. Returns the actual name
   *  the VRM uses, or null if none of the aliases exist. */
  _resolveName(requested) {
    const known = this.availableExpressions();
    if (known.has(requested)) return requested;
    const aliases = this._aliases[requested] || [requested];
    for (const alias of aliases) {
      if (known.has(alias)) return alias;
    }
    if (!this._unknownWarned.has(requested)) {
      this._unknownWarned.add(requested);
      console.debug(`[expression-channel] VRM has no expression '${requested}' (tried aliases ${JSON.stringify(aliases)})`);
    }
    return null;
  }

  evaluate(_timeMs, _dt) {
    // Smooth toward target
    const all = new Set([...this._target.keys(), ...this._current.keys()]);
    for (const name of all) {
      const tgt = this._target.get(name) ?? 0;
      const cur = this._current.get(name) ?? 0;
      const next = cur + (tgt - cur) * this.smoothing;
      this._current.set(name, next);
    }
    return new Map();
  }

  applyExtras(vrm) {
    if (!vrm.expressionManager) return;
    for (const [name, w] of this._current) {
      const resolved = this._resolveName(name);
      if (!resolved) continue;
      try { vrm.expressionManager.setValue(resolved, w); } catch {}
    }
    try { vrm.expressionManager.update?.(); } catch {}
  }

  /** Diagnostic snapshot for the bench inspector. */
  inspect() {
    const result = { available: [...this.availableExpressions()].sort(), live: {} };
    for (const [name, w] of this._current) {
      const resolved = this._resolveName(name);
      result.live[name] = { weight: +w.toFixed(3), resolved };
    }
    return result;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: LipsyncChannel
//
// Drives the VRM's viseme morphs (aa/ih/ou/ee/oh) — typically from
// audio analysis (RMS energy + formant features) or from a TTS
// synthesizer's phoneme stream.
//
// Tier: FACE — always alive. Body channels (pose, VRMA) don't suppress
// it, so the mouth keeps animating regardless of what the body is doing.
// This is the right composition: speech audio plays, mouth moves, body
// is free to dance/idle/whatever.
//
// Two ways to drive it:
//
//   1. setViseme(name, weight) — write a single viseme weight directly.
//      For TTS-driven lipsync where the synthesizer provides phoneme
//      timing.
//
//   2. setEnergy(rms) — derive viseme from an audio RMS value, useful
//      for simple amplitude-driven lipsync where you don't have phoneme
//      data. Cycles through visemes proportionally to amplitude.
//
// Writes to the standard VRM 1.0 viseme names: aa, ih, ou, ee, oh.
// Alias-resolves through ExpressionChannel-style fallbacks if needed.
// ─────────────────────────────────────────────────────────────────────────
export class LipsyncChannel extends MotionChannel {
  constructor(opts = {}) {
    super({
      name: opts.name || 'lipsync',
      mask: [],
      priority: opts.priority,
      blendMode: 'override',
      tier: opts.tier || TIER.FACE,
      enabled: opts.enabled !== false,
    });
    this.vrm = opts.vrm;
    /** Smoothing factor toward target viseme weights (0..1, per-frame). */
    this.smoothing = opts.smoothing ?? 0.45;
    /** Maximum mouth opening when audio is at peak amplitude. */
    this.maxOpen = opts.maxOpen ?? 0.85;
    /** @type {Map<string, number>} viseme name → target weight */
    this._target = new Map();
    /** @type {Map<string, number>} viseme name → current weight */
    this._current = new Map();
    /** Standard VRM viseme names — written each frame. */
    this.visemeNames = ['aa', 'ih', 'ou', 'ee', 'oh'];
    /** For simulated/synthetic lipsync via setEnergy — cycles slowly
     *  through visemes when the avatar appears to be speaking. */
    this._energyCyclePhase = 0;
  }

  /** Set a single viseme's target weight. */
  setViseme(name, weight) {
    this._target.set(name, Math.max(0, Math.min(1, weight)));
  }

  /** Snap all visemes to zero. Useful for silence / end-of-speech. */
  silence() {
    for (const name of this.visemeNames) this._target.set(name, 0);
  }

  /**
   * Synthetic lipsync from audio RMS (0..1). Slowly cycles which viseme
   * gets the energy so the mouth shape varies — pure amplitude looks
   * robotic (just "aa" pumping). Real lipsync should use phoneme data
   * via setViseme.
   */
  setEnergy(rms, dt = 0.016) {
    const energy = Math.max(0, Math.min(1, rms));
    if (energy < 0.04) {
      this.silence();
      return;
    }
    // Cycle viseme every ~150ms based on phase
    this._energyCyclePhase += dt * 4.0;
    const i = Math.floor(this._energyCyclePhase) % this.visemeNames.length;
    const fade = (this._energyCyclePhase % 1);
    const current = this.visemeNames[i];
    const next = this.visemeNames[(i + 1) % this.visemeNames.length];
    // Crossfade between adjacent visemes
    for (const name of this.visemeNames) {
      let w = 0;
      if (name === current) w = (1 - fade) * energy * this.maxOpen;
      else if (name === next) w = fade * energy * this.maxOpen;
      this._target.set(name, w);
    }
  }

  evaluate(_timeMs, _dt) {
    // Smooth current → target each frame
    for (const name of this.visemeNames) {
      const tgt = this._target.get(name) ?? 0;
      const cur = this._current.get(name) ?? 0;
      this._current.set(name, cur + (tgt - cur) * this.smoothing);
    }
    return new Map();
  }

  applyExtras(vrm) {
    if (!vrm.expressionManager) return;
    for (const name of this.visemeNames) {
      const w = this._current.get(name) ?? 0;
      try { vrm.expressionManager.setValue(name, w); } catch {}
    }
  }

  /** Diagnostic snapshot for the inspector. */
  inspect() {
    const out = {};
    for (const name of this.visemeNames) {
      out[name] = +(this._current.get(name) ?? 0).toFixed(3);
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: VRMAChannel
//
// Wraps a THREE.AnimationMixer driving a VRM Animation (.vrma) clip.
// Owns the mixer, ticks it each frame, snapshots the resulting bone
// state into the channel's output so it composes through the engine's
// priority + mask system instead of fighting other channels via the
// raw bone.quaternion bus.
//
// Higher priority than PoseTransitionChannel (150 vs 100) so a playing
// VRMA wins composition during its lifetime. When the clip finishes,
// the channel disables itself and fires onComplete — the consumer
// (embodiment engine, bench) responds by picking a landing pose and
// kicking off a PoseTransitionChannel.setTarget that smoothly hands
// off from the VRMA's final frame.
// ─────────────────────────────────────────────────────────────────────────
export class VRMAChannel extends MotionChannel {
  constructor(opts) {
    const mask = opts.mask || ALL_TRANSITION_BONES;
    super({
      name: opts.name || 'vrma',
      mask,
      priority: opts.priority,
      blendMode: 'override',
      enabled: false,
      tier: opts.tier || TIER.VRMA,
    });
    this.three = opts.three;
    this.vrm = opts.vrm;
    /** Lazily-built mixer. */
    this.mixer = null;
    this.currentAction = null;
    this.currentClip = null;
    this.currentClipName = null;
    /** Loop mode: 'once' | 'repeat'. */
    this.loopMode = 'once';
    /** Completion handler — invoked once when a `once` clip finishes. */
    this.onComplete = opts.onComplete || null;
    this._completedFired = false;
    this._scratchQuat = new opts.three.Quaternion();
    /** Optional bone freeze — kept as an escape hatch for badly-authored
     *  clips but DEFAULT IS EMPTY. The right fix for "head spinning" /
     *  "drift" is attaching VRMLookAtQuaternionProxy at VRM load and
     *  patching missing GLB specVersion (see fetchTolerantVrmaUrl in
     *  scene-test.html). Those two cure the actual disease; freezing
     *  treats symptoms. */
    this.freezeBones = new Set(opts.freezeBones || []);
    this._frozenQuats = new Map();
    this._frozenPositions = new Map();
  }

  /** Lazy mixer constructor. */
  _ensureMixer() {
    if (!this.mixer) this.mixer = new this.three.AnimationMixer(this.vrm.scene);
    return this.mixer;
  }

  /**
   * Play a VRMAnimationClip. The clip is what `createVRMAnimationClip(animation, vrm)`
   * returns from @pixiv/three-vrm-animation.
   * @param {THREE.AnimationClip} clip
   * @param {object} [opts]
   * @param {string} [opts.name]   diagnostic name surfaced in onComplete
   * @param {'once'|'repeat'} [opts.loop]
   * @param {number} [opts.crossfadeMs]  smoothly fade from current state
   */
  play(clip, {
    name = 'unnamed', loop = 'once', crossfadeMs = 0, freezeBones = null,
    trimStart = 0, trimEnd = 0, defaultSpeed = null,
  } = {}) {
    const mixer = this._ensureMixer();
    if (this.currentAction) {
      this.currentAction.stop();
      mixer.uncacheClip(this.currentClip);
    }
    // trimEnd — shorten the clip's duration BEFORE building the action.
    // The mixer treats the new duration as the loop boundary AND the
    // one-shot end, so glitchy final frames (e.g. spin clip flips upside
    // down in the last ~0.8s) are simply never reached.
    if (trimEnd > 0 && clip.duration > trimEnd) {
      clip.duration = Math.max(0.1, clip.duration - trimEnd);
    }
    this.currentClip = clip;
    this.currentAction = mixer.clipAction(clip);
    this.currentAction.setLoop(
      loop === 'repeat' ? this.three.LoopRepeat : this.three.LoopOnce
    );
    this.currentAction.clampWhenFinished = true;
    if (crossfadeMs > 0) {
      this.currentAction.fadeIn(crossfadeMs / 1000);
    }
    // defaultSpeed — overrides the mixer's effective time scale for this
    // clip. Lets the library declare "VRMA_05 reads better at 0.75x".
    if (defaultSpeed != null && defaultSpeed > 0) {
      this.currentAction.setEffectiveTimeScale(defaultSpeed);
    }
    // trimStart — jump past the opening N seconds (e.g. skip a crouch-
    // and-spring intro). For looping clips, wrap back to trimStart on
    // each loop instead of 0; for one-shots, start at trimStart on play.
    this._trimStart = trimStart;
    if (trimStart > 0) {
      this.currentAction.time = trimStart;
      // Loop boundary handler: when the mixer wraps, reset to trimStart
      // not 0. Stays attached to the mixer for the channel's lifetime;
      // we only fire when our specific action loops.
      if (!this._loopHandler) {
        this._loopHandler = (e) => {
          if (e.action === this.currentAction && this._trimStart > 0) {
            this.currentAction.time = this._trimStart;
          }
        };
        mixer.addEventListener('loop', this._loopHandler);
      }
    }
    this.currentClipName = name;
    this.loopMode = loop;
    this._completedFired = false;
    this.enabled = true;

    // Snapshot frozen bones at play start. Per-play override of the
    // channel default is supported via opts.freezeBones.
    const freeze = freezeBones ? new Set(freezeBones) : this.freezeBones;
    this._frozenQuats.clear();
    this._frozenPositions.clear();
    for (const boneName of freeze) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) {
        this._frozenQuats.set(boneName, node.quaternion.clone());
        this._frozenPositions.set(boneName, node.position.clone());
      }
    }

    this.currentAction.play();
  }

  /** Stop the active clip immediately. Does NOT fire onComplete. */
  stop() {
    if (this.currentAction) this.currentAction.stop();
    this.enabled = false;
  }

  /** True if a clip is currently mid-playback. */
  isPlaying() {
    return this.enabled && this.currentAction && !this.currentAction.paused;
  }

  /** Normalized playback position [0, 1] for diagnostics. */
  progress() {
    if (!this.currentAction || !this.currentClip) return 0;
    return Math.min(1, this.currentAction.time / Math.max(0.001, this.currentClip.duration));
  }

  evaluate(_timeMs, dtSec) {
    const out = new Map();
    if (!this.enabled || !this.mixer) return out;

    // Tick the mixer — writes to bone.quaternion directly.
    this.mixer.update(dtSec);

    // Restore frozen bones BEFORE we snapshot. The mixer may have written
    // hip translation/rotation that would drift each loop; this clamps
    // those bones back to their pre-play state so the avatar stays
    // anchored. The mask exclusion below then prevents those bones from
    // being snapshotted (so engine.applyToVRM doesn't propagate them
    // either — they stay at exactly the values we just restored).
    for (const [boneName, q] of this._frozenQuats) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) node.quaternion.copy(q);
    }
    for (const [boneName, p] of this._frozenPositions) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) node.position.copy(p);
    }

    this.vrm.scene.updateMatrixWorld(true);

    // Snapshot bone state into the channel's output so it composes
    // through the engine's priority system. Frozen bones are excluded
    // so other channels (or the next frame's mixer) can still write to
    // them without our channel claiming ownership.
    for (const boneName of this.mask) {
      if (this._frozenQuats.has(boneName)) continue;
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) out.set(boneName, node.quaternion.clone());
    }

    // Detect completion for one-shot clips.
    if (this.loopMode === 'once' && !this._completedFired && this.currentClip
        && this.currentAction.time >= this.currentClip.duration - 0.005) {
      this._completedFired = true;
      this.enabled = false;
      const cb = this.onComplete;
      const meta = {
        clipName: this.currentClipName,
        duration: this.currentClip.duration,
      };
      // Fire async so engine.update can finish this frame cleanly.
      if (cb) setTimeout(() => { try { cb(meta); } catch (err) { console.warn('[vrma] onComplete threw:', err); } }, 0);
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: BreathingChannel
//
// Sub-degree sinusoidal chest expansion + contraction. Mirrors the
// behavior of AvatarAnimator's spring-driven breathing in production
// but as a focused channel that composes through the engine. Reads the
// in-progress mixer result so it composes on top of whatever the pose
// channel wrote for chest/upperChest, falling back to bind rest if no
// channel claimed those bones this frame.
// ─────────────────────────────────────────────────────────────────────────
export class BreathingChannel extends MotionChannel {
  constructor(opts) {
    super({
      name: opts.name || 'breathing',
      mask: opts.mask || ['chest', 'upperChest'],
      priority: opts.priority,
      blendMode: 'override',
      tier: opts.tier || TIER.INDIVIDUAL,
    });
    this.three = opts.three;
    this.vrm = opts.vrm;
    /** Peak rotation depth at full inhale, in degrees. ~0.5° matches the
     *  subtle production setting. Higher values produce visible chest
     *  pump that reads as heavy breathing or exertion. */
    this.depthDeg = opts.depthDeg ?? 0.6;
    /** Breaths per minute. 14 is calm-resting adult; 18+ reads as energetic. */
    this.rateBpm = opts.rateBpm ?? 14;
    /** Phase accumulator. Initialized at random so multiple avatars in
     *  the same scene don't all breathe in lockstep. */
    this._phase = Math.random() * Math.PI * 2;
    /** Cached bind rest for fallback when no other channel wrote. */
    this._restQuats = new Map();
    for (const boneName of this.mask) {
      const node = opts.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) this._restQuats.set(boneName, node.quaternion.clone());
    }
    this._tmpEuler = new opts.three.Euler();
    this._tmpQuat = new opts.three.Quaternion();
  }

  setRate(bpm) { this.rateBpm = bpm; }
  setDepth(deg) { this.depthDeg = deg; }

  evaluate(_timeMs, dtSec, currentResult) {
    this._phase += (this.rateBpm / 60) * 2 * Math.PI * dtSec;
    const breath = Math.sin(this._phase);          // -1..1
    const angle = breath * this.depthDeg * Math.PI / 180;
    // Slight upper-back arch on inhale: positive X-rotation, larger on
    // upperChest than chest (more visible breath at the top of the rib cage).
    const result = new Map();
    for (const boneName of this.mask) {
      const scale = boneName === 'upperChest' ? 1.0 : 0.5;
      this._tmpEuler.set(angle * scale, 0, 0, 'XYZ');
      this._tmpQuat.setFromEuler(this._tmpEuler);
      // Compose on top of whatever lower-priority channels wrote;
      // fall back to bind rest if nothing else claims this bone.
      const base = currentResult?.get(boneName) || this._restQuats.get(boneName);
      if (!base) continue;
      result.set(boneName, base.clone().multiply(this._tmpQuat));
    }
    return result;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: BlinkChannel
//
// Scheduled blinks via the VRM 'blink' morph. Replicates production's
// AvatarAnimator blink timing: random interval (4±2s default), ~55ms
// close + ~105ms open. Lives in FACE tier so it stays alive through
// pose transitions AND VRMA playback (only the blink morph is touched).
// ─────────────────────────────────────────────────────────────────────────
export class BlinkChannel extends MotionChannel {
  constructor(opts = {}) {
    super({
      name: opts.name || 'blink',
      mask: [],
      priority: opts.priority,
      blendMode: 'override',
      tier: opts.tier || TIER.FACE,
    });
    this.vrm = opts.vrm;
    /** Mean interval between blinks, ms. */
    this.intervalBase = opts.intervalBase ?? 4000;
    /** ± randomization on each schedule, ms. */
    this.intervalVariance = opts.intervalVariance ?? 2000;
    /** Phase 1 (closing) duration, ms. */
    this.closeMs = opts.closeMs ?? 55;
    /** Phase 2 (opening) duration, ms. */
    this.openMs = opts.openMs ?? 105;
    this._nextBlinkAtMs = performance.now() + this._randomInterval();
    this._blinkStartMs = 0;
    this._inBlink = false;
    this._currentWeight = 0;
  }

  _randomInterval() {
    return this.intervalBase + (Math.random() - 0.5) * 2 * this.intervalVariance;
  }

  /** Schedule a blink soon (e.g. on attention shift, emotion spike). */
  triggerBlink({ delayMs = 80 } = {}) {
    this._nextBlinkAtMs = performance.now() + delayMs;
  }

  evaluate(timeMs, _dtSec, _currentResult) {
    if (!this._inBlink && timeMs >= this._nextBlinkAtMs) {
      this._inBlink = true;
      this._blinkStartMs = timeMs;
    }
    if (this._inBlink) {
      const elapsed = timeMs - this._blinkStartMs;
      if (elapsed < this.closeMs) {
        this._currentWeight = elapsed / this.closeMs;
      } else if (elapsed < this.closeMs + this.openMs) {
        this._currentWeight = 1 - (elapsed - this.closeMs) / this.openMs;
      } else {
        this._inBlink = false;
        this._currentWeight = 0;
        this._nextBlinkAtMs = timeMs + this._randomInterval();
      }
    }
    return new Map();
  }

  applyExtras(vrm) {
    if (!vrm.expressionManager) return;
    try { vrm.expressionManager.setValue('blink', this._currentWeight); } catch {}
  }

  inspect() {
    return {
      inBlink: this._inBlink,
      weight: +this._currentWeight.toFixed(3),
      nextInMs: Math.max(0, this._nextBlinkAtMs - performance.now()),
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: IdleSwayChannel
//
// Sub-degree incoherent noise on hips + spine to give "alive standing"
// micro-motion. Each axis uses an independent slow sinusoid at slightly
// different frequencies so the result reads as natural micro-balance
// rather than periodic wobble. Counter-sway on spine relative to hips
// so the upper body partially compensates (real bodies don't pivot
// rigidly around the ankle).
// ─────────────────────────────────────────────────────────────────────────
export class IdleSwayChannel extends MotionChannel {
  constructor(opts) {
    super({
      name: opts.name || 'idle_sway',
      mask: opts.mask || ['hips', 'spine'],
      priority: opts.priority,
      blendMode: 'override',
      tier: opts.tier || TIER.INDIVIDUAL,
    });
    this.three = opts.three;
    this.vrm = opts.vrm;
    this.hipDeg = opts.hipDeg ?? 0.4;
    this.spineDeg = opts.spineDeg ?? 0.3;
    // Phase + frequency per axis. Frequencies are intentionally
    // incommensurate (no simple ratio) so the motion never appears
    // periodic on perceptual timescales.
    this._phaseX = Math.random() * Math.PI * 2;
    this._phaseY = Math.random() * Math.PI * 2;
    this._phaseZ = Math.random() * Math.PI * 2;
    this._freqX = 0.31;  this._freqY = 0.43;  this._freqZ = 0.27;
    this._restQuats = new Map();
    for (const boneName of this.mask) {
      const node = opts.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (node) this._restQuats.set(boneName, node.quaternion.clone());
    }
    this._tmpEuler = new opts.three.Euler();
    this._tmpQuat = new opts.three.Quaternion();
  }

  evaluate(_timeMs, dtSec, currentResult) {
    this._phaseX += dtSec * this._freqX;
    this._phaseY += dtSec * this._freqY;
    this._phaseZ += dtSec * this._freqZ;
    const sX = Math.sin(this._phaseX);
    const sY = Math.sin(this._phaseY);
    const sZ = Math.sin(this._phaseZ);
    const result = new Map();
    const DEG = Math.PI / 180;
    for (const boneName of this.mask) {
      const baseDeg = boneName === 'hips' ? this.hipDeg : this.spineDeg;
      const sign = boneName === 'spine' ? -0.5 : 1.0;   // spine counter-sways
      this._tmpEuler.set(
        sX * baseDeg * 0.3 * sign * DEG,
        sY * baseDeg * 1.0 * sign * DEG,
        sZ * baseDeg * 0.3 * sign * DEG,
        'XYZ',
      );
      this._tmpQuat.setFromEuler(this._tmpEuler);
      const base = currentResult?.get(boneName) || this._restQuats.get(boneName);
      if (!base) continue;
      result.set(boneName, base.clone().multiply(this._tmpQuat));
    }
    return result;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Built-in channel: SpringChannel
//
// Thin wrapper over an existing AvatarAnimator's spring system so it
// composes through the engine instead of being hard-coded in the avatar
// tick loop. The animator does the spring math; we just route its
// outputs through the channel system.
//
// Mask: spine/chest/neck/head — same as the existing spring channel.
// Blend: compose-quat (springs add motion ON TOP of pose state).
// ─────────────────────────────────────────────────────────────────────────
export class SpringChannel extends MotionChannel {
  constructor(opts) {
    super({
      name: opts.name || 'spring',
      mask: opts.mask || ['spine', 'chest', 'upperChest', 'neck', 'head'],
      priority: opts.priority ?? 200,
      blendMode: 'compose-quat',
    });
    /** Function (boneName) → THREE.Quaternion delta (rest-relative). */
    this.deltaProvider = opts.deltaProvider || (() => null);
  }

  evaluate(_timeMs, _dt) {
    const result = new Map();
    for (const boneName of this.mask) {
      const delta = this.deltaProvider(boneName);
      if (delta) result.set(boneName, delta);
    }
    return result;
  }
}
