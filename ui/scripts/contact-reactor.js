/**
 * contact-reactor.js — reactive contact sensor for VR/MR + desktop.
 *
 * Subscribes to user hand world positions (from WebXR controllers, hand
 * tracking, or a desktop mouse-raycast proxy). Each tick:
 *
 *   1. For each user hand, query BodyAtlas + BodyMesh to find:
 *        - distance from user hand to avatar body
 *        - nearest mesh region (chin / chest_L / hand_R / etc.)
 *        - approach/hover/contact state classification
 *
 *   2. Emit lifecycle events on state transitions:
 *        - 'approach'  — user hand entered proximity window
 *        - 'hover'     — user hand sustained within hover band
 *        - 'contact'   — user hand inside body surface (sdf < threshold)
 *        - 'release'   — user hand left proximity entirely
 *
 *   3. While in 'approach' or 'hover' state, drive the avatar's nearest
 *      arm to reach toward the user — interpolated via critically-damped
 *      smoothing so the motion looks deliberate, stopping a safe gap
 *      short of the user's hand (no clipping through).
 *
 * The reactor doesn't write bones directly through the channel system —
 * it CALLS the AvatarIK helper passed at construction. The bench (and
 * eventual production wiring) is responsible for suspending pose_transition
 * for the active arm while reach is engaged, so the channels don't fight.
 *
 * Contact events delegate to the embodiment engine's onContactEvent —
 * that's where region-specific reactions live (cheek touch → blink + tilt,
 * chest touch → recoil, etc.).
 */

const REACH_DIST_M    = 0.50;   // user hand within this → avatar starts reaching
const HOVER_DIST_M    = 0.15;   // user hand within this → "hover band"
const CONTACT_DIST_M  = 0.02;   // user hand inside this → contact event
const STOP_SHORT_M    = 0.08;   // avatar hand stops this far short of user
const REACH_SMOOTHING = 4.0;    // critically-damped spring rate, per-second

// Comfort-gating (touchability-aware contact). Comfort c ∈ [0,1] is the body
// atlas's authored per-region touchability prior (cheek/chest/shoulder high,
// eyes/mouth/neck low). Below GUARD_PIVOT she keeps extra distance (up to
// +GUARD_EXTRA_M at c=0) and reaches more hesitantly; above WELCOME_PIVOT she
// leans in (closes up to WELCOME_CLOSE_FRAC of the base gap). Between, unchanged.
const GUARD_PIVOT        = 0.40;
const GUARD_EXTRA_M      = 0.22;
const WELCOME_PIVOT      = 0.75;
const WELCOME_CLOSE_FRAC = 0.50;

const HAND_BONE_BY_SIDE = { L: 'leftHand', R: 'rightHand' };

export class ContactReactor {
  /**
   * @param {object} opts
   * @param {object} opts.three         THREE namespace
   * @param {object} opts.vrm           VRM (uses humanoid bones + substrates)
   * @param {object} [opts.bodyMesh]    defaults to vrm.__augmentumBodyMesh
   * @param {object} [opts.bodyAtlas]   defaults to vrm.__augmentumBodyAtlas
   * @param {object} [opts.ik]          AvatarIK or compatible; receives
   *                                    setHandPositionWorld(side, worldPos)
   * @param {object} [opts.embodiment]  EmbodimentEngine to receive contact events
   * @param {function} [opts.onLog]
   */
  constructor(opts) {
    if (!opts?.three) throw new Error('ContactReactor needs three');
    if (!opts?.vrm) throw new Error('ContactReactor needs vrm');
    this.three = opts.three;
    this.vrm = opts.vrm;
    this.bodyMesh = opts.bodyMesh || opts.vrm.__augmentumBodyMesh || null;
    this.bodyAtlas = opts.bodyAtlas || opts.vrm.__augmentumBodyAtlas || null;
    this.ik = opts.ik || null;
    this.embodiment = opts.embodiment || null;
    this.onLog = opts.onLog || (() => {});

    /** Per-side user-hand world positions: [x,y,z] or null. */
    this._userHands = { L: null, R: null };
    /** Per-USER-hand state machine: 'idle' | 'approach' | 'hover' | 'contact'. */
    this._userState = { L: 'idle', R: 'idle' };
    /** Per-AVATAR-arm: which user hand is it reaching toward? */
    this._reachingFor = { L: null, R: null };   // 'L' | 'R' | null
    /** Per-avatar-arm smoothed IK target while reaching. */
    this._smoothedTarget = { L: null, R: null };
    /** Pose_transition channel handle so we can suspend per-arm during reach. */
    this._poseChannel = opts.poseChannel || null;
    /** True when reactor is actively driving an avatar arm. The bench
     *  reads this to decide if pose_transition should keep its hands off. */
    this.isActivelyReaching = false;
    /** Diagnostic: most recent contact info per user hand. */
    this.lastContact = { L: null, R: null };
    this.enabled = opts.enabled !== false;

    /** Velocity-aware motion intent classification. Public + mutable so the
     *  coordinator can flip it live (e.g. when the user toggles
     *  body_physics_velocity_aware) without rebuilding the reactor. */
    this.velocityAware = opts.velocityAware !== false;

    /** Comfort-gated contact: the atlas's per-region touchability prior (0..1)
     *  modulates how she meets a hand — welcoming affectionate zones
     *  (cheek/chest/shoulder) by leaning in, guarding sensitive ones
     *  (eyes/mouth/neck) by keeping more distance and reaching hesitantly.
     *  Public + mutable to flip live. Inert (legacy full-welcome reach) when no
     *  atlas is present or the region is unknown, so user VRMs without a baked
     *  atlas are byte-for-byte unchanged. */
    this.comfortGating = opts.comfortGating !== false;
    /** region name → comfort 0..1, from the atlas's authored touchability
     *  defaults (== per-voxel value; the bake fills each region uniformly).
     *  Empty without an atlas → _comfortForRegion returns null → legacy reach. */
    this._comfortByRegion = new Map();
    if (Array.isArray(this.bodyAtlas?.regionTable) && Array.isArray(this.bodyAtlas?.touchabilityDefaults)) {
      const rt = this.bodyAtlas.regionTable;
      const td = this.bodyAtlas.touchabilityDefaults;
      for (let i = 0; i < rt.length; i++) {
        if (typeof td[i] === 'number') this._comfortByRegion.set(rt[i], td[i] / 255);
      }
    }

    /** Per-hand timestamped position history (capped to 5 entries each).
     *  Used by getUserHandVelocity / getUserHandIntent. Cleared whenever a
     *  side goes null (tracking lost) so the next sample starts fresh
     *  rather than synthesizing a teleport velocity. */
    this._handHistory = { L: [], R: [] };

    /** Cached hips bone node — fetched lazily once, reused for the
     *  "toward avatar" dot-product test inside getUserHandIntent. */
    this._hipsBoneCache = null;
    this._hipsWorldScratch = new opts.three.Vector3();

    this._scratchVec = new opts.three.Vector3();
    this._scratchVec2 = new opts.three.Vector3();
  }

  /** Set user hand world positions per frame. Pass null on a side to
   *  clear (user dropped controller, hand left tracking volume). */
  setUserHand(side, worldPos) {
    if (side !== 'L' && side !== 'R') return;
    if (worldPos) {
      this._userHands[side] = [...worldPos];
      const hist = this._handHistory[side];
      hist.push({ pos: [worldPos[0], worldPos[1], worldPos[2]], t: performance.now() });
      while (hist.length > 5) hist.shift();
    } else {
      this._userHands[side] = null;
      this._handHistory[side] = [];
    }
  }

  /** Bulk setter from an array [leftPos, rightPos]. */
  setUserHands(arr) {
    this.setUserHand('L', arr?.[0] || null);
    this.setUserHand('R', arr?.[1] || null);
  }

  /** Per-frame tick. Drives detection, state transitions, reach. */
  tick(dtMs) {
    if (!this.enabled || !this.bodyMesh) return;
    const dtSec = dtMs / 1000;
    let anyReaching = false;
    for (const userSide of ['L', 'R']) {
      if (this._tickUserHand(userSide, dtSec)) anyReaching = true;
    }
    this.isActivelyReaching = anyReaching;
  }

  _tickUserHand(userSide, dtSec) {
    const userHand = this._userHands[userSide];
    if (!userHand) {
      // No tracking — release any state for this user hand.
      this._transitionState(userSide, 'idle');
      return false;
    }
    // Query mesh for nearest body region/distance.
    const hit = this.bodyMesh.closestPoint(userHand);
    if (!hit) {
      this._transitionState(userSide, 'idle');
      return false;
    }
    const distance = hit.distance;
    const region = hit.region;
    const comfort = this._comfortForRegion(region);
    this.lastContact[userSide] = { region, distance, point: hit.point, comfort };

    // State classification by distance.
    let nextState;
    if (distance <= CONTACT_DIST_M) nextState = 'contact';
    else if (distance <= HOVER_DIST_M) nextState = 'hover';
    else if (distance <= REACH_DIST_M) nextState = 'approach';
    else nextState = 'idle';

    if (nextState !== this._userState[userSide]) {
      this._transitionState(userSide, nextState, { region, distance, point: hit.point, comfort });
    }

    // Drive reach motion in approach/hover states.
    if (nextState === 'approach' || nextState === 'hover') {
      const arm = this._pickClosestArm(userHand);
      if (arm) {
        this._driveReach(arm, userSide, userHand, dtSec, comfort);
        return true;
      }
    } else {
      // Released — clear reach state for any arm that was reaching for THIS user hand.
      for (const arm of ['L', 'R']) {
        if (this._reachingFor[arm] === userSide) {
          this._reachingFor[arm] = null;
          this._smoothedTarget[arm] = null;
        }
      }
    }
    return false;
  }

  _transitionState(userSide, nextState, info = {}) {
    const prev = this._userState[userSide];
    this._userState[userSide] = nextState;
    if (prev === nextState) return;
    // Enrich with motion intent so downstream subscribers (audio reactions,
    // visual feedback, SDF compliance severity scaling) can vary intensity.
    // When velocityAware is off these are zeroed/'still' for consistency.
    const velocity = this.getUserHandVelocity(userSide);
    const intent = this.getUserHandIntent(userSide);
    const payload = { userSide, prev, state: nextState, velocity, intent, ...info };
    // Coarse comfort band so embodiment.onContactEvent can pick warm vs guarded
    // reactions without re-deriving thresholds. comfort itself rides in via info.
    payload.comfortBand = this._comfortBand(info.comfort ?? null);
    this.onLog({ kind: 'contact-state', ...payload });
    if (nextState === 'contact') {
      // Fire embodiment event so region-specific reactions can trigger.
      this.embodiment?.onContactEvent?.(payload);
    }
    if (prev === 'contact' && nextState !== 'contact') {
      this.embodiment?.onContactEvent?.({ ...payload, released: true });
    }
  }

  /** Which avatar arm is geometrically closer to the user hand? */
  _pickClosestArm(userHandPos) {
    let bestArm = null;
    let bestDist = Infinity;
    for (const arm of ['L', 'R']) {
      const boneName = HAND_BONE_BY_SIDE[arm];
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (!node) continue;
      node.getWorldPosition(this._scratchVec);
      const dx = this._scratchVec.x - userHandPos[0];
      const dy = this._scratchVec.y - userHandPos[1];
      const dz = this._scratchVec.z - userHandPos[2];
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (d < bestDist) { bestDist = d; bestArm = arm; }
    }
    return bestArm;
  }

  _driveReach(arm, userSide, userHand, dtSec, comfort = null) {
    this._reachingFor[arm] = userSide;
    // Current avatar hand world position.
    const handBone = this.vrm.humanoid.getNormalizedBoneNode?.(HAND_BONE_BY_SIDE[arm]);
    if (!handBone) return;
    handBone.getWorldPosition(this._scratchVec);
    const avatarHand = [this._scratchVec.x, this._scratchVec.y, this._scratchVec.z];

    // Target: along the line from avatar hand → user hand, stopping a comfort-
    // dependent gap short. Guarded (sensitive) zones widen the gap so she keeps
    // distance; welcoming zones narrow it so she leans in. Unknown/legacy →
    // STOP_SHORT_M exactly (no behavior change).
    const gap = this._gapForComfort(comfort);
    const dx = userHand[0] - avatarHand[0];
    const dy = userHand[1] - avatarHand[1];
    const dz = userHand[2] - avatarHand[2];
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const reachDist = Math.max(0, dist - gap);
    const t = dist > 1e-6 ? reachDist / dist : 0;
    const desired = [
      avatarHand[0] + dx * t,
      avatarHand[1] + dy * t,
      avatarHand[2] + dz * t,
    ];

    // Critically-damped smoothing toward the desired position. Initialize
    // smoothed target from current avatar hand so first frame doesn't snap.
    // Low comfort slows the approach (hesitant reach toward a guarded zone).
    if (!this._smoothedTarget[arm]) {
      this._smoothedTarget[arm] = [...avatarHand];
    }
    const hesitancy = comfort == null ? 1 : (0.55 + 0.45 * comfort);   // 0.55..1
    const alpha = Math.min(1, REACH_SMOOTHING * hesitancy * dtSec);
    for (let i = 0; i < 3; i++) {
      this._smoothedTarget[arm][i] += (desired[i] - this._smoothedTarget[arm][i]) * alpha;
    }

    // Apply via IK (caller-provided).
    if (this.ik?.setHandPositionWorld) {
      this.ik.setHandPositionWorld(arm, this._smoothedTarget[arm]);
    }
  }

  /** Per-region comfort 0..1 (atlas touchability prior), or null when gating is
   *  off / no atlas / region unknown — callers then use legacy full-welcome reach. */
  _comfortForRegion(region) {
    if (!this.comfortGating || !region) return null;
    const c = this._comfortByRegion.get(region);
    return c === undefined ? null : c;
  }

  /** Stop-short gap (m) for a comfort value: guarded below GUARD_PIVOT (wider),
   *  lean-in above WELCOME_PIVOT (narrower), STOP_SHORT_M between / when null. */
  _gapForComfort(c) {
    if (c == null) return STOP_SHORT_M;
    if (c < GUARD_PIVOT) {
      const g = (GUARD_PIVOT - c) / GUARD_PIVOT;            // 0..1 as c→0
      return STOP_SHORT_M + g * GUARD_EXTRA_M;
    }
    if (c > WELCOME_PIVOT) {
      const w = (c - WELCOME_PIVOT) / (1 - WELCOME_PIVOT);  // 0..1 as c→1
      return STOP_SHORT_M * (1 - w * WELCOME_CLOSE_FRAC);
    }
    return STOP_SHORT_M;
  }

  /** Coarse band for downstream reactions: 'guarded' | 'neutral' | 'welcome'. */
  _comfortBand(c) {
    if (c == null) return null;
    if (c < GUARD_PIVOT) return 'guarded';
    if (c >= WELCOME_PIVOT) return 'welcome';
    return 'neutral';
  }

  /** Latest user-hand world positions ({L: [x,y,z]|null, R: [x,y,z]|null}).
   *  Peer consumers (e.g. SDFCompliance) read from this to share the same
   *  per-frame hand-tracking state without duplicating the controller/hand
   *  pickoff logic. */
  getUserHands() {
    return { L: this._userHands.L, R: this._userHands.R };
  }

  /** Hand velocity over the recent history window.
   *  Returns {speed: m/s, direction: unit-vec [x,y,z]}.
   *  Returns zero/zero-vec if history is too short, dt is too small, or
   *  velocity-awareness is gated off. */
  getUserHandVelocity(side) {
    const zero = { speed: 0, direction: [0, 0, 0] };
    if (!this.velocityAware) return zero;
    if (side !== 'L' && side !== 'R') return zero;
    const hist = this._handHistory[side];
    if (!hist || hist.length < 2) return zero;
    const oldest = hist[0];
    const newest = hist[hist.length - 1];
    const dtMs = newest.t - oldest.t;
    if (dtMs < 16) return zero;
    const dtSec = dtMs / 1000;
    const dx = newest.pos[0] - oldest.pos[0];
    const dy = newest.pos[1] - oldest.pos[1];
    const dz = newest.pos[2] - oldest.pos[2];
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const speed = dist / dtSec;
    if (dist < 1e-8) return { speed, direction: [0, 0, 0] };
    const inv = 1 / dist;
    return { speed, direction: [dx * inv, dy * inv, dz * inv] };
  }

  /** Resolve and cache the avatar's hips world position into scratch.
   *  Returns true on success, false if the bone is missing. */
  _resolveHipsWorld() {
    if (!this._hipsBoneCache) {
      this._hipsBoneCache = this.vrm?.humanoid?.getNormalizedBoneNode?.('hips') || null;
    }
    if (!this._hipsBoneCache) return false;
    this._hipsBoneCache.getWorldPosition(this._hipsWorldScratch);
    return true;
  }

  /** Classify recent hand motion: 'still' | 'caress' | 'poke' | 'wave'.
   *  See class doc for thresholds. Always returns 'still' when
   *  velocityAware is false. */
  getUserHandIntent(side) {
    if (!this.velocityAware) return 'still';
    if (side !== 'L' && side !== 'R') return 'still';
    const vel = this.getUserHandVelocity(side);
    const speed = vel.speed;
    if (speed < 0.05) return 'still';

    const hist = this._handHistory[side];
    // Smoothness: dot of consecutive step directions.
    let smoothMin = 1.0;
    if (hist && hist.length >= 3) {
      for (let i = 1; i < hist.length - 1; i++) {
        const a = hist[i - 1].pos, b = hist[i].pos, c = hist[i + 1].pos;
        const ax = b[0] - a[0], ay = b[1] - a[1], az = b[2] - a[2];
        const bx = c[0] - b[0], by = c[1] - b[1], bz = c[2] - b[2];
        const la = Math.sqrt(ax*ax + ay*ay + az*az);
        const lb = Math.sqrt(bx*bx + by*by + bz*bz);
        if (la < 1e-6 || lb < 1e-6) continue;
        const d = (ax*bx + ay*by + az*bz) / (la * lb);
        if (d < smoothMin) smoothMin = d;
      }
    }

    // "Toward avatar" check uses the hips position as a proxy for body center.
    const hand = this._userHands[side];
    let towardDot = 0;
    if (hand && this._resolveHipsWorld()) {
      let tx = this._hipsWorldScratch.x - hand[0];
      let ty = this._hipsWorldScratch.y - hand[1];
      let tz = this._hipsWorldScratch.z - hand[2];
      const tl = Math.sqrt(tx*tx + ty*ty + tz*tz);
      if (tl > 1e-6) {
        tx /= tl; ty /= tl; tz /= tl;
        const d = vel.direction;
        towardDot = d[0]*tx + d[1]*ty + d[2]*tz;
      }
    }

    if (speed >= 0.05 && speed <= 0.3 && smoothMin > 0.8) return 'caress';
    if (speed > 0.5 && towardDot > 0.5) return 'poke';
    if (speed > 0.3 && towardDot < 0.3) return 'wave';
    return 'still';
  }

  /** Snapshot for inspectors. */
  inspect() {
    return {
      enabled: this.enabled,
      velocityAware: this.velocityAware,
      comfortGating: this.comfortGating,
      isActivelyReaching: this.isActivelyReaching,
      userState: { ...this._userState },
      reachingFor: { ...this._reachingFor },
      lastContact: { ...this.lastContact },
      velocity: {
        L: this.getUserHandVelocity('L'),
        R: this.getUserHandVelocity('R'),
      },
      intent: {
        L: this.getUserHandIntent('L'),
        R: this.getUserHandIntent('R'),
      },
    };
  }
}
