/**
 * avatar-pose-presets.js — shared static-body-pose catalog
 *
 * Authored in the scene-test studio (ui/mockups/scene-test.html), where
 * the IK gizmo + pose recorder produce per-bone Euler rotations in the
 * VRM normalized humanoid space. This module is the single source of
 * truth — scene-test imports POSE_PRESETS from here, and production
 * (PoseOrchestrator → AvatarAnimator) consumes the same data.
 *
 * Units & conventions:
 *   - All bone rotations are EULER ANGLES in RADIANS (matches the JSON
 *     export format from the studio's pose recorder).
 *   - Rotation order is per-bone (BONE_ROTATION_ORDERS). IK-chain bones
 *     use the chain's declared order so post-IK values stay consistent
 *     across save → reload → slerp. Non-IK bones default to 'XYZ'.
 *   - Finger rotations use the 'finger.joint' nested form when authored
 *     via HAND_POSES, but in pose presets they're flat keys
 *     (`leftIndexProximal`, etc.) because the recorder captures bones
 *     individually. Both forms describe the same VRM normalized bones.
 *   - `_avatarPosition` and `_hipsTranslation` are metadata for sitting/
 *     reclined poses — the pose application code reads them to move the
 *     avatar root + hip bone before applying joint rotations.
 *
 * Family system:
 *   Each pose declares a `family`. The PoseOrchestrator walks between
 *   members of the active family on a dwell+slerp timer, producing
 *   subtle living motion (e.g. drift between thinking_a ↔ thinking_b
 *   reads as natural deliberation). Single-member families are valid
 *   for static "pin this pose" use; the orchestrator just holds them
 *   without intra-family drift.
 *
 *   Adding more variants to a family (e.g. head-turned-left,
 *   head-turned-right, head-center for a `looking_around` family) is
 *   a pure-data change; no orchestrator code change needed.
 */

// ─── Bone rotation order lookup ────────────────────────────────────────
// IK-chain bones must use their chain's declared order so save / apply
// / slerp stay consistent. Non-listed bones default to 'XYZ'.
//
// Source of truth: IK_CHAINS in scene-test.html (~line 1590). Mirrored
// here so this module is self-contained — if the chains change, update
// both places.
export const BONE_ROTATION_ORDERS = {
  // Torso (XYZ)
  chest: 'XYZ',
  spine: 'XYZ',
  hips: 'XYZ',
  // Arm chains
  leftShoulder: 'ZXY',
  rightShoulder: 'ZXY',
  leftUpperArm: 'ZXY',
  rightUpperArm: 'ZXY',
  leftLowerArm: 'YZX',
  rightLowerArm: 'YZX',
  // Leg chains
  leftUpperLeg: 'XYZ',
  rightUpperLeg: 'XYZ',
  leftLowerLeg: 'XYZ',
  rightLowerLeg: 'XYZ',
};

export function getBoneRotationOrder(boneName) {
  return BONE_ROTATION_ORDERS[boneName] || 'XYZ';
}

// ─── Pose presets ──────────────────────────────────────────────────────
// Eleven approved poses across six families. Authored in scene-test on
// bundled VRMs (Becca / Vance / Lise / Danny / Roxanne / Louis), then
// inlined here from poses/*.json after visual approval.
//
// Family conventions:
//   - 'idle_standing'  — relaxed full-body standing variants, suitable
//                        for ambient drift while the avatar is at rest
//   - 'idle_engaged'   — forward-leaning / hip-shifted variants for
//                        active listening / conversation drift
//   - 'thinking'       — hand-on-chin variants for reasoning beats
//   - 'formal'         — contained postures (hands behind / clasped)
//                        for formal / professional read
//   - 'closed'         — defensive / contemplative closed-body
//   - 'seated'         — seated variants (couch edge etc.)
export const POSE_PRESETS = {
  natural: {
    family: 'idle_standing',
    label: 'Natural',
    bones: {
      head:           [0.04, 0,    0],
      spine:          [0,    0,    0],
      chest:          [0,    0,    0],
      leftShoulder:   [0,    0,    0],
      rightShoulder:  [0,    0,    0],
      leftUpperArm:   [0,    0.04, -1.35],
      rightUpperArm:  [0,   -0.04,  1.35],
      leftLowerArm:   [0.10, 0,   -0.05],
      rightLowerArm:  [0.10, 0,    0.05],
      hips:           [0,    0,    0],
      // Soft relaxed-curl fingers — without these, VRoid VRMs render
      // straight splayed "doll hands" since the rest pose is fully open.
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,   -0.3],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,   -0.35],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,   -0.4],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,   -0.45],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  contrapposto: {
    family: 'idle_standing',
    label: 'Contrapposto',
    bones: {
      head:           [0.02,  -0.06, 0.04],
      spine:          [0,      0,    0.05],
      chest:          [0,      0,   -0.03],
      rightShoulder:  [0,      0,    0.04],
      leftUpperArm:   [0,      0.04, -1.30],
      rightUpperArm:  [0,     -0.06,  1.40],
      leftLowerArm:   [0.15,   0,   -0.10],
      rightLowerArm:  [0.05,   0,    0.05],
      hips:           [0,      0,   -0.06],
      // Soft relaxed-curl fingers (see `natural` for rationale).
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,   -0.3],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,   -0.35],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,   -0.4],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,   -0.45],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  // Slight lean back, weight on heels — skeptical / amused / casual.
  leaningBack: {
    family: 'idle_standing',
    label: 'Leaning back',
    bones: {
      head:           [-0.04, 0,    0],
      spine:          [-0.08, 0,    0],
      chest:          [-0.04, 0,    0],
      leftUpperArm:   [0.10,  0.10, -1.30],
      rightUpperArm:  [0.10, -0.10,  1.30],
      leftLowerArm:   [0.10,  0,   -0.05],
      rightLowerArm:  [0.10,  0,    0.05],
      hips:           [-0.03, 0,    0],
      // Soft relaxed-curl fingers (see `natural` for rationale).
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,   -0.3],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,   -0.35],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,   -0.4],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,   -0.45],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  // ─── Talking-gesture family (2 members, forearm-bob drift) ────────
  // Right hand raised forward in a presenting/talking gesture. Both
  // members share `rightShoulder` and `rightUpperArm` (same upper-arm
  // angle holding the position) — only `rightLowerArm` + `rightHand`
  // differ so drift reads as the forearm pivoting up and down at the
  // elbow, like natural conversational gesturing.
  // Authored 2026-05-03 (poses/righthandinfront[down]-talking-pose-...).

  // Right hand raised higher — forearm nearly horizontal.
  talking_high: {
    family: 'talking',
    label: 'Talking (hand high)',
    bones: {
      head:           [0.04,    0,      0],
      rightShoulder:  [-0.111, -0.256, -0.14],
      leftUpperArm:   [0,       0.04,  -1.35],
      rightUpperArm:  [1.475,   2.354, -1.082],
      leftLowerArm:   [0.1,     0,     -0.05],
      rightLowerArm:  [-0.016,  1.747, -1.067],
      rightHand:      [-0.742,  0.314,  0.337],
      // Left hand: relaxed-curl baseline.
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,   -0.3],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,   -0.35],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,   -0.4],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,   -0.45],
      // Right hand: looser open-palm-style curl (presenting).
      rightThumbProximal:     [-0.096, 0.3,  0.2],
      rightThumbDistal:       [0,      0,    0.2],
      rightIndexProximal:     [0,      0,    0.233],
      rightIndexIntermediate: [0.008, -0.002, 0.156],
      rightIndexDistal:       [0,      0,    0.3],
      rightMiddleProximal:    [0,      0,    0.008],
      rightMiddleIntermediate:[0,      0,    0.45],
      rightMiddleDistal:      [0,      0,    0.35],
      rightRingProximal:      [0,      0,    0.006],
      rightRingIntermediate:  [0,      0,    0.389],
      rightRingDistal:        [0,      0,    0.4],
      rightLittleProximal:    [0,      0,    0.45],
      rightLittleIntermediate:[0,      0,   -0.271],
      rightLittleDistal:      [0,      0,    0.45],
    },
  },

  // Right hand lowered — forearm angled down, hand pointing more
  // toward the listener. Drift partner of talking_high.
  talking_low: {
    family: 'talking',
    label: 'Talking (hand low)',
    bones: {
      head:           [-0.029,  0,      0],
      leftShoulder:   [0,      -0.004, -0.019],
      rightShoulder:  [-0.111, -0.256, -0.14],
      leftUpperArm:   [-0.006,  0.016, -1.371],
      rightUpperArm:  [1.475,   2.354, -1.082],
      leftLowerArm:   [0,      -0.197,  0],
      rightLowerArm:  [-0.735,  0.838, -0.817],
      rightHand:      [-0.791,  0.079,  0.247],
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,   -0.3],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,   -0.35],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,   -0.4],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,   -0.45],
      rightThumbProximal:     [-0.096, 0.3,  0.2],
      rightThumbDistal:       [0,      0,    0.2],
      rightIndexProximal:     [0,      0,    0.233],
      rightIndexIntermediate: [0.008, -0.002, 0.156],
      rightIndexDistal:       [0,      0,    0.3],
      rightMiddleProximal:    [0,      0,    0.008],
      rightMiddleIntermediate:[0,      0,    0.45],
      rightMiddleDistal:      [0,      0,    0.35],
      rightRingProximal:      [0,      0,    0.006],
      rightRingIntermediate:  [0,      0,    0.389],
      rightRingDistal:        [0,      0,    0.4],
      rightLittleProximal:    [0,      0,    0.45],
      rightLittleIntermediate:[0,      0,   -0.271],
      rightLittleDistal:      [0,      0,    0.45],
    },
  },

  // Casual idle — left hand resting on stomach, right at side.
  // Authored 2026-05-03 (poses/idle-lefthandonstomach-righthandonside-...).
  //
  // Lives in its own `idle_holding` family rather than `idle_standing`
  // because the left arm wraps to the stomach — drifting between this
  // and natural/contrapposto/leaningBack would visibly swing the arm
  // in and out every dwell cycle. Authoring a sibling variant (e.g.
  // right hand also moves to stomach) would let it drift cleanly.
  handOnStomach: {
    family: 'idle_holding',
    label: 'Hand on stomach',
    bones: {
      head:           [0.04,    0,      0],
      leftShoulder:   [0.227,   0.258,  0.249],
      rightShoulder:  [-0.261, -0.181,  0.253],
      leftUpperArm:   [0.237,   0.141, -1.767],
      rightUpperArm:  [0.022,  -0.243,  1.111],
      leftLowerArm:   [0,      -1.31,   0],
      leftHand:       [0.281,  -0.266, -0.63],
      rightHand:      [0.013,  -0.112,  0.068],
      leftUpperLeg:   [-0.163,  0,      0],
      leftLowerLeg:   [0.169,   0,      0],
      rightLowerLeg:  [-0.053,  0,      0],
      // Left hand: index/middle relaxed-curl, ring/little folded under (rests on stomach)
      leftThumbProximal:      [0.032, -0.314, -0.423],
      leftThumbDistal:        [0.406,  0.245,  0.288],
      leftIndexProximal:      [0,      0,    -0.3],
      leftIndexIntermediate:  [0,      0,    -0.4],
      leftIndexDistal:        [0,      0,     0.122],
      leftMiddleProximal:     [0,      0,    -0.35],
      leftMiddleIntermediate: [0,      0,    -0.45],
      leftMiddleDistal:       [0,      0,     0.075],
      leftRingProximal:       [0,      0,    -0.4],
      leftRingIntermediate:   [0,      0,    -1.696],
      leftRingDistal:         [0,      0,    -1.378],
      leftLittleProximal:     [0,      0,    -0.45],
      leftLittleIntermediate: [0,      0,    -1.495],
      leftLittleDistal:       [0,      0,    -1.464],
      // Right hand: fully relaxed at side
      rightThumbProximal:     [0.1,    0.3,   0.2],
      rightThumbDistal:       [0,      0,     0.2],
      rightIndexProximal:     [0,      0,     0.3],
      rightIndexIntermediate: [0,      0,     0.4],
      rightIndexDistal:       [0,      0,     0.3],
      rightMiddleProximal:    [0,      0,     0.35],
      rightMiddleIntermediate:[0,      0,     0.45],
      rightMiddleDistal:      [0,      0,     0.35],
      rightRingProximal:      [0,      0,     0.4],
      rightRingIntermediate:  [0,      0,     0.5],
      rightRingDistal:        [0,      0,     0.4],
      rightLittleProximal:    [0,      0,     0.45],
      rightLittleIntermediate:[0,      0,     0.55],
      rightLittleDistal:      [0,      0,     0.45],
    },
  },

  // Grounded idle - left arm settled at the side, right hand relaxed
  // into a loose fist, one foot forward. Authored 2026-05-05
  // (poses/lefthandonside-righthandclosedfist-footforwardidlepose-...).
  //
  // Own family because the foot-forward stance and closed right hand read
  // as a distinct "ready/present" basin. Route through neutral waypoints
  // when entering from cross-body families.
  groundedFootForward: {
    family: 'idle_grounded',
    label: 'Grounded (foot forward)',
    via: ['natural', 'leaningIn', 'leaningBack'],
    bones: {
      head:           [0.04,    0,       0],
      chest:          [0.018,   0,       0],
      leftShoulder:   [-0.214,  0.187,   0.041],
      rightShoulder:  [0.262,   0.077,   0.262],
      leftUpperArm:   [0.553,   0.959,  -1.453],
      rightUpperArm:  [0.719,   0.289,   0.888],
      leftLowerArm:   [0,      -1.475,   0],
      leftHand:       [0.923,   0,       0.188],
      hips:           [-0.002,  0,      -0.002],
      leftUpperLeg:   [-0.141,  0.483,   0.089],
      rightUpperLeg:  [0,      -0.361,   0.002],
      leftLowerLeg:   [0.001,   0,       0],
      rightLowerLeg:  [0.005,   0,       0],
      leftFoot:       [0.19,    0,       0],
      leftThumbProximal:      [0.918, -0.248, -1.209],
      leftThumbDistal:        [0,      0,     -0.2],
      leftIndexProximal:      [-0.083, 0.079, -1.061],
      leftIndexIntermediate:  [0,      0,     -0.4],
      leftIndexDistal:        [0,      0,     -0.3],
      leftMiddleProximal:     [0,      0,     -1.112],
      leftMiddleIntermediate: [0,      0,     -0.45],
      leftMiddleDistal:       [0,      0,     -0.35],
      leftRingProximal:       [0,      0,     -1.071],
      leftRingIntermediate:   [0,      0,     -0.5],
      leftRingDistal:         [0,      0,     -0.4],
      leftLittleProximal:     [0,      0,     -0.941],
      leftLittleIntermediate: [0,      0,     -0.55],
      leftLittleDistal:       [0,      0,     -0.45],
      rightThumbProximal:     [-1.898, -1.519, -1.85],
      rightThumbDistal:       [0,      -0.78,  -0.042],
      rightIndexProximal:     [0,       0,      1.576],
      rightIndexIntermediate: [0,       0,      1.417],
      rightMiddleProximal:    [0,       0,      1.576],
      rightMiddleIntermediate:[0,       0,      1.684],
      rightMiddleDistal:      [0,       0,      0.35],
      rightRingProximal:      [0,       0,      1.694],
      rightRingIntermediate:  [0,       0,      1.474],
      rightRingDistal:        [0,       0,      0.4],
      rightLittleProximal:    [0,       0,      1.599],
      rightLittleIntermediate:[0,       0,      1.602],
      rightLittleDistal:      [0,       0,      0.45],
    },
  },

  // Slight lean forward — engaged / listening / curious.
  // Authored 2026-05-03 (poses/leaningin-pose-2026-05-03T02-00-26.json).
  leaningIn: {
    family: 'idle_engaged',
    label: 'Leaning in',
    bones: {
      head:           [0.04,   0,      0],
      spine:          [0.123,  0,      0.003],
      chest:          [-0.088, -0.002, -0.011],
      leftShoulder:   [0,      0,     -0.01],
      rightShoulder:  [0.231,  0.212,  0.068],
      leftUpperArm:   [-0.001, 0.035, -1.361],
      rightUpperArm:  [0.429,  0.072,  1.1],
      leftLowerArm:   [0,     -0.139,  0],
      rightLowerArm:  [0,      0.504,  0],
      hips:           [0.054,  0,     -0.001],
      leftUpperLeg:   [0,      0,      0.002],
      rightUpperLeg:  [0,      0,      0.002],
      leftLowerLeg:   [0.003,  0,      0],
      rightLowerLeg:  [0.003,  0,      0],
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,   -0.3],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,   -0.35],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,   -0.4],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,   -0.45],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  // Right hand on hip; left arm relaxed at side. Confident-relaxed.
  // Refined values from poses/Rhandonhip-pose-2026-04-30T22-48-18.json.
  handOnHip: {
    family: 'idle_engaged',
    label: 'Hand on hip',
    bones: {
      head:           [0.020,  0.040, -0.030],
      spine:          [0,      0,     0.020],
      chest:          [0.017,  0,    -0.020],
      rightShoulder:  [-0.084, -0.044, -0.191],
      leftUpperArm:   [0,      0.040, -1.300],
      rightUpperArm:  [0.027, -0.715,  1.663],
      leftLowerArm:   [0.100,  0,    -0.050],
      rightLowerArm:  [0,      1.391, 0],
      hips:           [0,      0,    -0.037],
    },
  },

  // Thinking variant A — right hand on chin, left hand on waist.
  // Authored 2026-05-03 (poses/thinking-righthandonchin-lefthandonwaistpose-...).
  thinking_a: {
    family: 'thinking',
    label: 'Thinking (waist)',
    bones: {
      head:           [0.04,    0,     0],
      chest:          [0.012,   0,     0],
      leftShoulder:   [-0.246,  0.261, -0.201],
      rightShoulder:  [0.118,  -0.1,    0.248],
      leftUpperArm:   [0.248,   0.797, -1.177],
      rightUpperArm:  [0.079,   1.22,   0.448],
      leftLowerArm:   [-0.042, -1.372, -0.02],
      rightLowerArm:  [0,       2.206,  0],
      rightHand:      [-1.099,  0.221,  0.481],
      hips:           [-0.001,  0,     -0.003],
      leftUpperLeg:   [0,       0,      0.003],
      rightUpperLeg:  [-0.199, -0.097, -0.064],
      leftLowerLeg:   [0.003,   0,      0],
      rightLowerLeg:  [0.003,   0,      0],
      rightFoot:      [0.294,  -0.122,  0.022],
      leftThumbProximal:      [0.2,  -0.4, -0.3],
      leftThumbDistal:        [0,     0,   -0.3],
      leftIndexProximal:      [0,     0,   -0.6],
      leftIndexIntermediate:  [0,     0,   -0.9],
      leftIndexDistal:        [0,     0,   -0.7],
      leftMiddleProximal:     [0,     0,   -0.65],
      leftMiddleIntermediate: [0,     0,   -0.95],
      leftMiddleDistal:       [0,     0,   -0.75],
      leftRingProximal:       [0,     0,   -0.7],
      leftRingIntermediate:   [0,     0,   -1.0],
      leftRingDistal:         [0,     0,   -0.8],
      leftLittleProximal:     [0,     0,   -0.75],
      leftLittleIntermediate: [0,     0,   -1.05],
      leftLittleDistal:       [0,     0,   -0.85],
      rightThumbProximal:     [0.194, 0.064, 0.482],
      rightIndexProximal:     [0,     0,    0.6],
      rightIndexIntermediate: [-0.28, -0.289, 0.118],
      rightIndexDistal:       [0,     0,    0.7],
      rightMiddleProximal:    [0,     0,    0.65],
      rightMiddleIntermediate:[0,     0,    0.95],
      rightMiddleDistal:      [0,     0,    0.75],
      rightRingProximal:      [0,     0,    0.7],
      rightRingIntermediate:  [0,     0,    1.0],
      rightRingDistal:        [0,     0,    0.8],
      rightLittleProximal:    [0,     0,    0.75],
      rightLittleIntermediate:[0,     0,    1.05],
      rightLittleDistal:      [0,     0,    0.85],
    },
  },

  // Thinking variant B — right hand on chin, left hand under chest.
  // Pair with thinking_a for slow drift between members.
  // Authored 2026-05-03 (poses/thinking-righthandonchin-lefthandunderchest-...).
  thinking_b: {
    family: 'thinking',
    label: 'Thinking (chest)',
    bones: {
      head:           [0.04,    0,      0],
      chest:          [0.012,   0,      0],
      leftShoulder:   [-0.26,  -0.063, -0.033],
      rightShoulder:  [0.118,  -0.1,    0.248],
      leftUpperArm:   [1.024,  -0.394, -1.18],
      rightUpperArm:  [0.079,   1.22,   0.448],
      leftLowerArm:   [1.282,  -1.566, -0.016],
      rightLowerArm:  [0,       2.206,  0],
      leftHand:       [-1.313, -0.305, -1.251],
      rightHand:      [-1.099,  0.221,  0.481],
      hips:           [-0.001,  0,     -0.003],
      leftUpperLeg:   [0,       0,      0.003],
      rightUpperLeg:  [-0.143, -0.099, -0.044],
      leftLowerLeg:   [0.003,   0,      0],
      rightFoot:      [0.225,  -0.122,  0.022],
      leftThumbProximal:      [0.047,  0.478, -0.114],
      leftThumbDistal:        [0.872,  0.985, -1.142],
      leftIndexProximal:      [0,      0,    -0.6],
      leftIndexIntermediate:  [0,      0,    -1.164],
      leftIndexDistal:        [0,      0,    -0.7],
      leftMiddleProximal:     [0,      0,    -0.65],
      leftMiddleIntermediate: [0,      0,    -1.169],
      leftMiddleDistal:       [0,      0,    -1.245],
      leftRingProximal:       [0,      0,    -0.7],
      leftRingIntermediate:   [0,      0,    -1.243],
      leftRingDistal:         [0,      0,    -1.297],
      leftLittleProximal:     [0,      0,    -0.75],
      leftLittleIntermediate: [0.176, -0.196, -1.335],
      leftLittleDistal:       [0,      0,    -1.453],
      rightThumbProximal:     [0.194,  0.064, 0.482],
      rightIndexProximal:     [0,      0,     0.6],
      rightIndexIntermediate: [-0.28, -0.289, 0.118],
      rightIndexDistal:       [0,      0,     0.7],
      rightMiddleProximal:    [0,      0,     0.65],
      rightMiddleIntermediate:[0,      0,     0.95],
      rightMiddleDistal:      [0,      0,     0.75],
      rightRingProximal:      [0,      0,     0.7],
      rightRingIntermediate:  [0,      0,     1.0],
      rightRingDistal:        [0,      0,     0.8],
      rightLittleProximal:    [0,      0,     0.75],
      rightLittleIntermediate:[0,      0,     1.05],
      rightLittleDistal:      [0,      0,     0.85],
    },
  },

  // Hands clasped behind the back — formal / parade-rest.
  // Authored 2026-05-03 (poses/handsbehindback-pose-2026-05-03T01-50-27.json).
  //
  // Own family `formal_behind` because hands sit at the lower back —
  // drifting against `clasped` (hands low in front) would swing both
  // arms across the body every dwell cycle. Pair with a future
  // sibling (e.g. weight-shift variant of hands-behind) to drift.
  //
  // `via` — direct slerp from poses with hands forward (clasped,
  // thinking, etc.) drives the elbows through the torso. Routing
  // through an arms-at-sides pose first lets the hands swing out and
  // around the body cleanly. Orchestrator picks one randomly.
  handsBehind: {
    family: 'formal_behind',
    label: 'Hands behind',
    via: ['natural', 'leaningIn', 'leaningBack'],
    bones: {
      head:           [0.04,   0,      0],
      leftShoulder:   [-0.237, 0.243, -0.134],
      rightShoulder:  [-0.215, -0.21, -0.007],
      leftUpperArm:   [0.456,  0.525, -1.718],
      rightUpperArm:  [0.074, -0.597,  1.642],
      leftLowerArm:   [0,     -0.221,  0],
      rightLowerArm:  [-0.002, 0.306,  0.008],
      leftHand:       [1.118, -0.19,  -0.091],
      rightHand:      [1.307,  0.266,  0.039],
      // Soft fists resting against the lower back.
      leftThumbProximal:      [0.2,  -0.5, -0.4],
      leftThumbDistal:        [0,     0,   -0.5],
      leftIndexProximal:      [0,     0,   -0.9],
      leftIndexIntermediate:  [0,     0,   -1.4],
      leftIndexDistal:        [0,     0,   -1.2],
      leftMiddleProximal:     [0,     0,   -0.95],
      leftMiddleIntermediate: [0,     0,   -1.45],
      leftMiddleDistal:       [0,     0,   -1.2],
      leftRingProximal:       [0,     0,   -1.0],
      leftRingIntermediate:   [0,     0,   -1.45],
      leftRingDistal:         [0,     0,   -1.2],
      leftLittleProximal:     [0,     0,   -1.05],
      leftLittleIntermediate: [0,     0,   -1.45],
      leftLittleDistal:       [0,     0,   -1.2],
      rightThumbProximal:     [0.2,   0.5,  0.4],
      rightThumbDistal:       [0,     0,    0.5],
      rightIndexProximal:     [0,     0,    0.9],
      rightIndexIntermediate: [0,     0,    1.4],
      rightIndexDistal:       [0,     0,    1.2],
      rightMiddleProximal:    [0,     0,    0.95],
      rightMiddleIntermediate:[0,     0,    1.45],
      rightMiddleDistal:      [0,     0,    1.2],
      rightRingProximal:      [0,     0,    1.0],
      rightRingIntermediate:  [0,     0,    1.45],
      rightRingDistal:        [0,     0,    1.2],
      rightLittleProximal:    [0,     0,    1.05],
      rightLittleIntermediate:[0,     0,    1.45],
      rightLittleDistal:      [0,     0,    1.2],
    },
  },

  // ─── Clasped family (4 members, head-only drift) ──────────────────
  // All 4 share IDENTICAL body bones (interlocked-fist clasp held low
  // in front of waist). Only the HEAD bone differs — the orchestrator
  // slerp between members reads as the avatar slowly looking around
  // while keeping a contained, formal posture.
  //
  // Body data refined 2026-05-03 from re-recorded variants. Earlier
  // `clasped` pose used slightly different right-arm angles + no
  // finger curl; replaced here so drift between the 4 head variants
  // is purely head-driven (no arm shift).

  // Head straight — gentle forward chin tilt. Use as the "base"
  // resting member; the others are deviations from this.
  clasped: {
    family: 'formal',
    label: 'Clasped (straight)',
    bones: {
      head:           [0.06,    0,      0],
      spine:          [0.04,    0,      0],
      chest:          [0.02,    0,      0],
      leftShoulder:   [0.262,  -0.262, -0.262],
      rightShoulder:  [0.258,   0.232,  0.22],
      leftUpperArm:   [-0.37,  -0.531, -1.18],
      rightUpperArm:  [0.466,   0.608,  0.832],
      leftLowerArm:   [0.592,  -0.06,  -0.614],
      rightLowerArm:  [0,       0.465,  0],
      leftHand:       [0.157,   0.075, -0.092],
      rightHand:      [0.243,   0.721,  0.642],
      // Both hands in interlocked fists (grip pose).
      leftThumbProximal:      [0.772,  1.143, -0.62],
      leftThumbDistal:        [0,      0,    -0.5],
      leftIndexProximal:      [0,      0,    -0.9],
      leftIndexIntermediate:  [0,      0,    -1.4],
      leftIndexDistal:        [0,      0,    -1.2],
      leftMiddleProximal:     [0,      0,    -0.95],
      leftMiddleIntermediate: [0,      0,    -1.45],
      leftMiddleDistal:       [0,      0,    -1.2],
      leftRingProximal:       [0,      0,    -1.0],
      leftRingIntermediate:   [0,      0,    -1.45],
      leftRingDistal:         [0,      0,    -1.2],
      leftLittleProximal:     [0,      0,    -1.05],
      leftLittleIntermediate: [0,      0,    -1.45],
      leftLittleDistal:       [0,      0,    -1.2],
      rightThumbProximal:     [0.2,    0.5,   0.4],
      rightThumbDistal:       [0,      0,     0.5],
      rightIndexProximal:     [0,      0,     0.9],
      rightIndexIntermediate: [0,      0,     1.4],
      rightIndexDistal:       [0,      0,     1.2],
      rightMiddleProximal:    [0,      0,     0.95],
      rightMiddleIntermediate:[0,      0,     1.45],
      rightMiddleDistal:      [0,      0,     1.2],
      rightRingProximal:      [0,      0,     1.0],
      rightRingIntermediate:  [0,      0,     1.45],
      rightRingDistal:        [0,      0,     1.2],
      rightLittleProximal:    [0,      0,     1.05],
      rightLittleIntermediate:[0,      0,     1.45],
      rightLittleDistal:      [0,      0,     1.2],
    },
  },

  // Head turned to her left.
  // Authored 2026-05-03 (handclaspedinfront-pose-headturnedleft-...).
  clasped_left: {
    family: 'formal',
    label: 'Clasped (looking left)',
    bones: {
      head:           [0.108,   0.978, -0.089],
      spine:          [0.04,    0,      0],
      chest:          [0.02,    0,      0],
      leftShoulder:   [0.262,  -0.262, -0.262],
      rightShoulder:  [0.258,   0.232,  0.22],
      leftUpperArm:   [-0.37,  -0.531, -1.18],
      rightUpperArm:  [0.466,   0.608,  0.832],
      leftLowerArm:   [0.592,  -0.06,  -0.614],
      rightLowerArm:  [0,       0.465,  0],
      leftHand:       [0.157,   0.075, -0.092],
      rightHand:      [0.243,   0.721,  0.642],
      leftThumbProximal:      [0.772,  1.143, -0.62],
      leftThumbDistal:        [0,      0,    -0.5],
      leftIndexProximal:      [0,      0,    -0.9],
      leftIndexIntermediate:  [0,      0,    -1.4],
      leftIndexDistal:        [0,      0,    -1.2],
      leftMiddleProximal:     [0,      0,    -0.95],
      leftMiddleIntermediate: [0,      0,    -1.45],
      leftMiddleDistal:       [0,      0,    -1.2],
      leftRingProximal:       [0,      0,    -1.0],
      leftRingIntermediate:   [0,      0,    -1.45],
      leftRingDistal:         [0,      0,    -1.2],
      leftLittleProximal:     [0,      0,    -1.05],
      leftLittleIntermediate: [0,      0,    -1.45],
      leftLittleDistal:       [0,      0,    -1.2],
      rightThumbProximal:     [0.2,    0.5,   0.4],
      rightThumbDistal:       [0,      0,     0.5],
      rightIndexProximal:     [0,      0,     0.9],
      rightIndexIntermediate: [0,      0,     1.4],
      rightIndexDistal:       [0,      0,     1.2],
      rightMiddleProximal:    [0,      0,     0.95],
      rightMiddleIntermediate:[0,      0,     1.45],
      rightMiddleDistal:      [0,      0,     1.2],
      rightRingProximal:      [0,      0,     1.0],
      rightRingIntermediate:  [0,      0,     1.45],
      rightRingDistal:        [0,      0,     1.2],
      rightLittleProximal:    [0,      0,     1.05],
      rightLittleIntermediate:[0,      0,     1.45],
      rightLittleDistal:      [0,      0,     1.2],
    },
  },

  // Head turned to her right.
  // Authored 2026-05-03 (handclaspedinfront-pose-headturnedright-...).
  clasped_right: {
    family: 'formal',
    label: 'Clasped (looking right)',
    bones: {
      head:           [0.094,  -0.875,  0.072],
      spine:          [0.04,    0,      0],
      chest:          [0.02,    0,      0],
      leftShoulder:   [0.262,  -0.262, -0.262],
      rightShoulder:  [0.258,   0.232,  0.22],
      leftUpperArm:   [-0.37,  -0.531, -1.18],
      rightUpperArm:  [0.466,   0.608,  0.832],
      leftLowerArm:   [0.592,  -0.06,  -0.614],
      rightLowerArm:  [0,       0.465,  0],
      leftHand:       [0.157,   0.075, -0.092],
      rightHand:      [0.243,   0.721,  0.642],
      leftThumbProximal:      [0.772,  1.143, -0.62],
      leftThumbDistal:        [0,      0,    -0.5],
      leftIndexProximal:      [0,      0,    -0.9],
      leftIndexIntermediate:  [0,      0,    -1.4],
      leftIndexDistal:        [0,      0,    -1.2],
      leftMiddleProximal:     [0,      0,    -0.95],
      leftMiddleIntermediate: [0,      0,    -1.45],
      leftMiddleDistal:       [0,      0,    -1.2],
      leftRingProximal:       [0,      0,    -1.0],
      leftRingIntermediate:   [0,      0,    -1.45],
      leftRingDistal:         [0,      0,    -1.2],
      leftLittleProximal:     [0,      0,    -1.05],
      leftLittleIntermediate: [0,      0,    -1.45],
      leftLittleDistal:       [0,      0,    -1.2],
      rightThumbProximal:     [0.2,    0.5,   0.4],
      rightThumbDistal:       [0,      0,     0.5],
      rightIndexProximal:     [0,      0,     0.9],
      rightIndexIntermediate: [0,      0,     1.4],
      rightIndexDistal:       [0,      0,     1.2],
      rightMiddleProximal:    [0,      0,     0.95],
      rightMiddleIntermediate:[0,      0,     1.45],
      rightMiddleDistal:      [0,      0,     1.2],
      rightRingProximal:      [0,      0,     1.0],
      rightRingIntermediate:  [0,      0,     1.45],
      rightRingDistal:        [0,      0,     1.2],
      rightLittleProximal:    [0,      0,     1.05],
      rightLittleIntermediate:[0,      0,     1.45],
      rightLittleDistal:      [0,      0,     1.2],
    },
  },

  // Head down + offset to her right (thinking-while-clasped read).
  // Slight chest forward (0.028 vs 0.02) deepens the contemplative
  // posture. Authored 2026-05-03 (handclaspedinfront-pose-headdownofftosidethinking-...).
  clasped_thinking: {
    family: 'formal',
    label: 'Clasped (thinking)',
    bones: {
      head:           [0.324,  -0.564,  0.038],
      spine:          [0.04,    0,      0],
      chest:          [0.028,   0,      0],
      leftShoulder:   [0.262,  -0.262, -0.262],
      rightShoulder:  [0.258,   0.232,  0.22],
      leftUpperArm:   [-0.37,  -0.531, -1.18],
      rightUpperArm:  [0.466,   0.608,  0.832],
      leftLowerArm:   [0.592,  -0.06,  -0.614],
      rightLowerArm:  [0,       0.465,  0],
      leftHand:       [0.157,   0.075, -0.092],
      rightHand:      [0.243,   0.721,  0.642],
      hips:           [0,       0,     -0.002],
      leftUpperLeg:   [0,       0,      0.002],
      rightUpperLeg:  [0,       0,      0.002],
      leftThumbProximal:      [0.772,  1.143, -0.62],
      leftThumbDistal:        [0,      0,    -0.5],
      leftIndexProximal:      [0,      0,    -0.9],
      leftIndexIntermediate:  [0,      0,    -1.4],
      leftIndexDistal:        [0,      0,    -1.2],
      leftMiddleProximal:     [0,      0,    -0.95],
      leftMiddleIntermediate: [0,      0,    -1.45],
      leftMiddleDistal:       [0,      0,    -1.2],
      leftRingProximal:       [0,      0,    -1.0],
      leftRingIntermediate:   [0,      0,    -1.45],
      leftRingDistal:         [0,      0,    -1.2],
      leftLittleProximal:     [0,      0,    -1.05],
      leftLittleIntermediate: [0,      0,    -1.45],
      leftLittleDistal:       [0,      0,    -1.2],
      rightThumbProximal:     [0.2,    0.5,   0.4],
      rightThumbDistal:       [0,      0,     0.5],
      rightIndexProximal:     [0,      0,     0.9],
      rightIndexIntermediate: [0,      0,     1.4],
      rightIndexDistal:       [0,      0,     1.2],
      rightMiddleProximal:    [0,      0,     0.95],
      rightMiddleIntermediate:[0,      0,     1.45],
      rightMiddleDistal:      [0,      0,     1.2],
      rightRingProximal:      [0,      0,     1.0],
      rightRingIntermediate:  [0,      0,     1.45],
      rightRingDistal:        [0,      0,     1.2],
      rightLittleProximal:    [0,      0,     1.05],
      rightLittleIntermediate:[0,      0,     1.45],
      rightLittleDistal:      [0,      0,     1.2],
    },
  },

  // ─── Left-hand-on-hip family (3 members, head-only drift) ─────────
  // Distinct from `handOnHip` (which is right-hand-on-hip). All 3
  // members share identical body bones — only HEAD differs, so drift
  // reads as glancing around while the hip-hand stays planted.
  // Authored 2026-05-03 (idle-turnedlefthandonhip-righthandonsidepose-*).

  // Base — head turned slightly to her right (conversational rest).
  // `via`: left arm has to wrap to the hip — direct slerp from a
  // chest/chin pose drives the wrist through the torso. Bridge first.
  leftHandOnHip: {
    family: 'idle_lefthip',
    label: 'Left hip (right glance)',
    via: ['natural', 'leaningIn', 'leaningBack'],
    bones: {
      head:           [0.049,  -0.634,  0.029],
      chest:          [-0.014,  0,      0],
      leftShoulder:   [-0.216, -0.217,  0.115],
      rightShoulder:  [0,       0.002, -0.12],
      leftUpperArm:   [-0.084,  0.845, -1.647],
      rightUpperArm:  [-0.004, -0.065,  1.55],
      leftLowerArm:   [0,      -1.011,  0],
      rightLowerArm:  [0,       0.122,  0],
      leftHand:       [-0.593,  0.015,  0.238],
      hips:           [0.001,   0,     -0.003],
      leftUpperLeg:   [-0.001,  0,      0.004],
      rightUpperLeg:  [-0.001,  0,      0.004],
      // Soft relaxed-curl fingers — same baseline as `relaxed` hand pose.
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,    0.089],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,    0.028],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,    0.094],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,    0.219],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  // Head straight forward.
  leftHandOnHip_straight: {
    family: 'idle_lefthip',
    label: 'Left hip (straight)',
    via: ['natural', 'leaningIn', 'leaningBack'],
    bones: {
      head:           [0.04,    0.035, -0.002],
      chest:          [-0.014,  0,      0],
      leftShoulder:   [-0.216, -0.217,  0.115],
      rightShoulder:  [0,       0.002, -0.12],
      leftUpperArm:   [-0.084,  0.845, -1.647],
      rightUpperArm:  [-0.004, -0.065,  1.55],
      leftLowerArm:   [0,      -1.011,  0],
      rightLowerArm:  [0,       0.122,  0],
      leftHand:       [-0.593,  0.015,  0.238],
      hips:           [0.001,   0,     -0.003],
      leftUpperLeg:   [-0.001,  0,      0.004],
      rightUpperLeg:  [-0.001,  0,      0.004],
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,    0.089],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,    0.028],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,    0.094],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,    0.219],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  // Head turned to her left (positive Y in normalized humanoid space).
  // Source file is named "headturnedright" reflecting user perspective
  // (mirror view); the bone data turns the avatar's own head left.
  leftHandOnHip_left: {
    family: 'idle_lefthip',
    label: 'Left hip (left glance)',
    via: ['natural', 'leaningIn', 'leaningBack'],
    bones: {
      head:           [0.066,   0.922, -0.053],
      chest:          [-0.014,  0,      0],
      leftShoulder:   [-0.216, -0.217,  0.115],
      rightShoulder:  [0,       0.002, -0.12],
      leftUpperArm:   [-0.084,  0.845, -1.647],
      rightUpperArm:  [-0.004, -0.065,  1.55],
      leftLowerArm:   [0,      -1.011,  0],
      rightLowerArm:  [0,       0.122,  0],
      leftHand:       [-0.593,  0.015,  0.238],
      hips:           [0.001,   0,     -0.003],
      leftUpperLeg:   [-0.001,  0,      0.004],
      rightUpperLeg:  [-0.001,  0,      0.004],
      leftThumbProximal:      [0.1,  -0.3, -0.2],
      leftThumbDistal:        [0,     0,   -0.2],
      leftIndexProximal:      [0,     0,   -0.3],
      leftIndexIntermediate:  [0,     0,   -0.4],
      leftIndexDistal:        [0,     0,    0.089],
      leftMiddleProximal:     [0,     0,   -0.35],
      leftMiddleIntermediate: [0,     0,   -0.45],
      leftMiddleDistal:       [0,     0,    0.028],
      leftRingProximal:       [0,     0,   -0.4],
      leftRingIntermediate:   [0,     0,   -0.5],
      leftRingDistal:         [0,     0,    0.094],
      leftLittleProximal:     [0,     0,   -0.45],
      leftLittleIntermediate: [0,     0,   -0.55],
      leftLittleDistal:       [0,     0,    0.219],
      rightThumbProximal:     [0.1,   0.3,  0.2],
      rightThumbDistal:       [0,     0,    0.2],
      rightIndexProximal:     [0,     0,    0.3],
      rightIndexIntermediate: [0,     0,    0.4],
      rightIndexDistal:       [0,     0,    0.3],
      rightMiddleProximal:    [0,     0,    0.35],
      rightMiddleIntermediate:[0,     0,    0.45],
      rightMiddleDistal:      [0,     0,    0.35],
      rightRingProximal:      [0,     0,    0.4],
      rightRingIntermediate:  [0,     0,    0.5],
      rightRingDistal:        [0,     0,    0.4],
      rightLittleProximal:    [0,     0,    0.45],
      rightLittleIntermediate:[0,     0,    0.55],
      rightLittleDistal:      [0,     0,    0.45],
    },
  },

  // Arms folded across chest. spine/chest/neck are zeroed so per-frame
  // breathing+sway overlays don't compound unbounded.
  armsCrossed: {
    family: 'closed',
    label: 'Arms crossed',
    bones: {
      head:           [0.04,   0,     0],
      neck:           [0,      0,     0],
      spine:          [0,      0,     0],
      chest:          [0,      0,     0],
      leftShoulder:   [-0.041, -0.262, -0.262],
      rightShoulder:  [-0.262,  0.262,  0.262],
      leftUpperArm:   [-0.180, -0.916, -0.981],
      rightUpperArm:  [-0.324,  0.853,  1.349],
      leftLowerArm:   [-2.798, -2.792, -1.407],
      rightLowerArm:  [ 2.407, -2.206,  1.131],
      leftHand:       [ 0.012,  0.020, -0.539],
      rightHand:      [ 0.257, -0.080,  0.562],
      hips:           [0,      0,      0],
    },
  },

  // Sitting on the edge of a couch — right hand on right thigh, left
  // hand resting on the couch surface to her side. Authored via IK
  // gizmo from poses/Sitting-Rhandonleg-Lhandoncouch.json.
  sittingEdge: {
    family: 'seated',
    label: 'Sitting (couch edge)',
    // Root world placement calibrated against modern-room.glb's couch
    // in ui/mockups/scene-test.html (_COUCH_SIT). The drop is fully in
    // root Y — no hip-bone translation — to match how the bench was
    // authored, so legs/feet line up against the couch geometry as
    // tuned in the IK gizmo.
    _avatarPosition: [0.75, -0.54, 2.25],
    bones: {
      head:           [0.050,   0,      0],
      neck:           [0.030,   0,      0],
      spine:          [0.080,   0,      0],
      chest:          [0.040,   0,      0],
      hips:           [-0.050,  0,      0],
      leftShoulder:   [0.241,  -0.257, -0.169],
      rightShoulder:  [-0.078,  0.132,  0.169],
      leftUpperArm:   [0.018,  -0.042, -0.969],
      rightUpperArm:  [0.608,   0.474,  0.960],
      leftLowerArm:   [0,      -0.218,  0],
      rightLowerArm:  [0,       0.594,  0],
      leftHand:       [-0.130,  0.172,  0.749],
      rightHand:      [-0.161, -0.616, -0.422],
      leftUpperLeg:   [-1.428,  0.016,  0.027],
      rightUpperLeg:  [-1.437,  0.116,  0.147],
      leftLowerLeg:   [ 1.569,  0,      0],
      rightLowerLeg:  [ 1.504,  0,      0],
      leftFoot:       [-0.088,  0,      0],
      rightFoot:      [ 0.030,  0,      0],
    },
  },
};

// ─── Family index ──────────────────────────────────────────────────────
// Auto-derived from POSE_PRESETS so adding a new pose just requires
// declaring its `family` — no separate registration step.
export const POSE_FAMILIES = (() => {
  const fams = {};
  for (const [name, preset] of Object.entries(POSE_PRESETS)) {
    const fam = preset.family || 'misc';
    if (!fams[fam]) fams[fam] = [];
    fams[fam].push(name);
  }
  return fams;
})();

// Human-readable family labels for UI selectors. Singleton families
// (one member) still need a label so they appear in the dropdown —
// orchestrator picks them, slerp-pins, no drift fires.
export const FAMILY_LABELS = {
  idle_standing: 'Idle (standing)',
  idle_engaged: 'Idle (engaged)',
  idle_holding: 'Idle (holding)',
  idle_grounded: 'Idle (grounded)',
  thinking: 'Thinking',
  talking: 'Talking (hand bob)',
  formal: 'Clasped (drift)',
  formal_behind: 'Hands behind',
  idle_lefthip: 'Left hand on hip (drift)',
  closed: 'Closed',
  seated: 'Seated',
};

// ─── Helpers ───────────────────────────────────────────────────────────
export function getPose(name) {
  return POSE_PRESETS[name] || null;
}

export function getFamily(poseName) {
  return POSE_PRESETS[poseName]?.family || null;
}

export function getFamilyMembers(familyName) {
  return POSE_FAMILIES[familyName] || [];
}

export function listPoses() {
  return Object.keys(POSE_PRESETS);
}

export function listFamilies() {
  return Object.keys(POSE_FAMILIES);
}

// Returns a flat dict of bone → euler array, dropping metadata keys.
// Convenience for callers that want to iterate joint rotations directly.
export function getPoseBones(name) {
  const preset = POSE_PRESETS[name];
  if (!preset) return null;
  return preset.bones;
}

// Returns metadata (avatar root position / hip translation) for sitting
// or reclined poses. Null if the pose has no metadata overrides.
export function getPoseMetadata(name) {
  const preset = POSE_PRESETS[name];
  if (!preset) return null;
  const meta = {};
  if (preset._avatarPosition) meta.avatarPosition = preset._avatarPosition;
  if (preset._hipsTranslation) meta.hipsTranslation = preset._hipsTranslation;
  return Object.keys(meta).length ? meta : null;
}

// Per-bone classification — drives axis-sign correction when applying a
// preset across VRMs with different arm/finger axis conventions.
const _FINGER_BONE_RE = /^(left|right)(Thumb|Index|Middle|Ring|Little)(Metacarpal|Proximal|Intermediate|Distal)$/;
const _ARM_BONE_RE = /^(left|right)(Shoulder|UpperArm|LowerArm|Hand)$/;

export function isFingerBoneName(name) { return _FINGER_BONE_RE.test(name); }
export function isArmBoneName(name) { return _ARM_BONE_RE.test(name); }

/**
 * Apply a PoseOrchestrator snapshot to a VRM with rest-relative finger
 * composition. Body and arm bones write the snapshot quaternion
 * directly; finger bones compose against captured rest quats so VRMs
 * with non-identity finger bind (rare but real) get correct curls.
 *
 * @param {object} vrm                 Loaded VRM
 * @param {object} snapshot            Output of PoseOrchestrator.getCurrentPose()
 * @param {object} [fingerRestQuats]   Map of boneName → THREE.Quaternion rest
 *                                     (typically vrm.__augmentumFingerRestQuats)
 * @returns {boolean}                  True if applied
 */
export function applyOrchestratorPoseToVRM(vrm, snapshot, fingerRestQuats = null) {
  if (!vrm?.humanoid || !snapshot) return false;
  const rests = fingerRestQuats || vrm.__augmentumFingerRestQuats || null;
  for (const [boneName, state] of Object.entries(snapshot)) {
    if (boneName.startsWith('_')) continue;
    const node = vrm.humanoid.getNormalizedBoneNode?.(boneName);
    if (!node || !state?.quaternion) continue;
    if (isFingerBoneName(boneName) && rests?.[boneName]) {
      node.quaternion.copy(rests[boneName]).multiply(state.quaternion);
    } else {
      node.quaternion.copy(state.quaternion);
    }
  }
  // Hips translation and avatar root position — same handling as
  // applyPosePreset, but only writes when the snapshot carries them.
  const hips = vrm.humanoid.getNormalizedBoneNode?.('hips');
  if (hips && snapshot._hipsTranslation) {
    if (!hips._origPos) hips._origPos = hips.position.clone();
    hips.position.copy(hips._origPos);
    hips.position.x += snapshot._hipsTranslation[0];
    hips.position.y += snapshot._hipsTranslation[1];
    hips.position.z += snapshot._hipsTranslation[2];
  }
  if (snapshot._avatarPosition && vrm.scene) {
    vrm.scene.position.set(
      snapshot._avatarPosition[0],
      snapshot._avatarPosition[1],
      snapshot._avatarPosition[2],
    );
  }
  return true;
}

/**
 * Apply a pose preset to a VRM with axis-aware bone writing.
 *
 * Production callers (loadVRM at-load baseline, XR seated pose, future
 * pose-orchestrator wiring) consume the same authored preset data but
 * must compensate for per-VRM arm and finger axis conventions — same
 * problem that bit the hand-pose channel. This helper is the single
 * source of truth for "apply preset → bones" so the correction lives in
 * exactly one place.
 *
 * Body bones (head/spine/chest/hips/legs): absolute write at per-bone
 *   Euler order. No axis sign — these bones don't mirror across the
 *   midline.
 * Arm bones (shoulder/upper/lower/hand): absolute write at per-bone
 *   Euler order, multiplied by `armAxisSign`. Same correction the spring
 *   channel uses via _armTargetSign.
 * Finger bones: rest-relative write (`bone.quaternion = restQuat *
 *   delta`), with the delta Euler multiplied by `fingerAxisSign`. Matches
 *   the rest-relative path that _applyHandPoses uses for the procedural
 *   relaxed pose. Captures rest from the bone's current quaternion at
 *   call time — caller should `resetNormalizedPose` first (or pass
 *   `reset: true`) so "rest" actually means rest, not whatever the prior
 *   pose left behind.
 *
 * @param {object} THREE              Three.js namespace
 * @param {object} vrm                Loaded VRM
 * @param {string|object} presetOrName  Preset name or preset object
 * @param {object} [options]
 * @param {object} [options.armAxisSign]      { x, y, z } multiplier for arm bones
 * @param {object} [options.fingerAxisSign]   { x, y, z } multiplier for finger bones
 * @param {boolean} [options.reset=true]      Run resetNormalizedPose before applying
 * @returns {boolean}  True if applied
 */
export function applyPosePreset(THREE, vrm, presetOrName, options = {}) {
  const preset = typeof presetOrName === 'string'
    ? POSE_PRESETS[presetOrName]
    : presetOrName;
  if (!preset?.bones || !vrm?.humanoid || !THREE) return false;

  const armSign = options.armAxisSign || { x: 1, y: 1, z: 1 };
  const fingerSign = options.fingerAxisSign || { x: 1, y: 1, z: 1 };
  const reset = options.reset !== false;
  if (reset) vrm.humanoid.resetNormalizedPose?.();

  const tempEuler = new THREE.Euler();
  const tempQuat = new THREE.Quaternion();
  const restQuat = new THREE.Quaternion();

  for (const [boneName, rot] of Object.entries(preset.bones)) {
    const node = vrm.humanoid.getNormalizedBoneNode?.(boneName);
    if (!node || !Array.isArray(rot)) continue;
    const order = BONE_ROTATION_ORDERS[boneName] || 'XYZ';

    if (isFingerBoneName(boneName)) {
      // Rest-relative for fingers — same approach as _applyHandPoses.
      // Capture rest from current quaternion (which, post-reset, IS the
      // normalized bind orientation).
      restQuat.copy(node.quaternion);
      tempEuler.set(
        rot[0] * fingerSign.x,
        rot[1] * fingerSign.y,
        rot[2] * fingerSign.z,
        order,
      );
      tempQuat.setFromEuler(tempEuler);
      node.quaternion.copy(restQuat).multiply(tempQuat);
    } else if (isArmBoneName(boneName)) {
      // Arms: absolute with sign correction.
      node.rotation.set(
        rot[0] * armSign.x,
        rot[1] * armSign.y,
        rot[2] * armSign.z,
        order,
      );
    } else {
      // Body bones — no axis sign; preset values apply directly.
      node.rotation.set(rot[0], rot[1], rot[2], order);
    }
  }

  // Hips translation — separate from rotation, applied off the cached
  // rest position so the avatar can sit / kneel without losing the
  // standing baseline.
  const hips = vrm.humanoid.getNormalizedBoneNode?.('hips');
  if (hips && preset._hipsTranslation) {
    if (!hips._origPos) hips._origPos = hips.position.clone();
    hips.position.copy(hips._origPos);
    hips.position.x += preset._hipsTranslation[0];
    hips.position.y += preset._hipsTranslation[1];
    hips.position.z += preset._hipsTranslation[2];
  }
  return true;
}
