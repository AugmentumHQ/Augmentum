/**
 * Hand pose library — finger-curl Euler values per joint, ported
 * verbatim from the scene-test studio (ui/mockups/scene-test.html, the
 * `HAND_POSES` const around line 1156). Single source of truth so the
 * production avatar animator and the studio render the same hands;
 * scene-test will be migrated to import from this module in a follow-up
 * (left as-is for now to avoid perturbing the studio).
 *
 * Each pose is a map of `finger → { proximal, intermediate, distal }`
 * where each joint value is `[x, y, z]` in radians, applied directly
 * via `bone.rotation.set(x, y, z)` (default XYZ order).
 *
 * The data is authored for the LEFT hand. Mirror to the right by
 * negating Y and Z (including the thumb's Y — author's note in the
 * scene-test apply function specifically calls this out). See
 * `mirrorRotForRight` below; the animator's `_applyHandPoses` uses it.
 */

export const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little'];
export const JOINT_NAMES = ['proximal', 'intermediate', 'distal'];

export const HAND_POSES = {
  // Default — slight progressive curl from index to little. Reads as a
  // hand at rest; without this, fingers sit straight (the "daggers"
  // look on production avatars before the port).
  relaxed: {
    thumb:  { proximal: [0.10, -0.30, -0.20], intermediate: [0, 0, -0.20], distal: [0, 0, -0.20] },
    index:  { proximal: [0,     0,    -0.30], intermediate: [0, 0, -0.40], distal: [0, 0, -0.30] },
    middle: { proximal: [0,     0,    -0.35], intermediate: [0, 0, -0.45], distal: [0, 0, -0.35] },
    ring:   { proximal: [0,     0,    -0.40], intermediate: [0, 0, -0.50], distal: [0, 0, -0.40] },
    little: { proximal: [0,     0,    -0.45], intermediate: [0, 0, -0.55], distal: [0, 0, -0.45] },
  },
  fist: {
    thumb:  { proximal: [0.20, -0.50, -0.40], intermediate: [0, 0, -0.50], distal: [0, 0, -0.50] },
    index:  { proximal: [0,     0,    -0.90], intermediate: [0, 0, -1.40], distal: [0, 0, -1.20] },
    middle: { proximal: [0,     0,    -0.95], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
    ring:   { proximal: [0,     0,    -1.00], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
    little: { proximal: [0,     0,    -1.05], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
  },
  open: {
    thumb:  { proximal: [0,    -0.10, -0.10], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    index:  { proximal: [0,     0,    -0.10], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    middle: { proximal: [0,     0,    -0.10], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    ring:   { proximal: [0,     0,    -0.10], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    little: { proximal: [0,     0,    -0.10], intermediate: [0, 0, 0], distal: [0, 0, 0] },
  },
  // Cupped — fingers curled like wrapping a waist or holding a glass.
  holding: {
    thumb:  { proximal: [0.20, -0.40, -0.30], intermediate: [0, 0, -0.40], distal: [0, 0, -0.30] },
    index:  { proximal: [0,     0,    -0.60], intermediate: [0, 0, -0.90], distal: [0, 0, -0.70] },
    middle: { proximal: [0,     0,    -0.65], intermediate: [0, 0, -0.95], distal: [0, 0, -0.75] },
    ring:   { proximal: [0,     0,    -0.70], intermediate: [0, 0, -1.00], distal: [0, 0, -0.80] },
    little: { proximal: [0,     0,    -0.75], intermediate: [0, 0, -1.05], distal: [0, 0, -0.85] },
  },
  // Index extended, others fist.
  point: {
    thumb:  { proximal: [0.20, -0.50, -0.40], intermediate: [0, 0, -0.50], distal: [0, 0, -0.50] },
    index:  { proximal: [0,     0,    -0.05], intermediate: [0, 0, -0.05], distal: [0, 0,  0] },
    middle: { proximal: [0,     0,    -0.95], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
    ring:   { proximal: [0,     0,    -1.00], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
    little: { proximal: [0,     0,    -1.05], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
  },
  // Index + middle extended in V, others curled.
  peace: {
    thumb:  { proximal: [0.20, -0.50, -0.40], intermediate: [0, 0, -0.50], distal: [0, 0, -0.50] },
    index:  { proximal: [0,    -0.10, -0.05], intermediate: [0, 0, -0.05], distal: [0, 0,  0] },
    middle: { proximal: [0,     0.10, -0.05], intermediate: [0, 0, -0.05], distal: [0, 0,  0] },
    ring:   { proximal: [0,     0,    -1.00], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
    little: { proximal: [0,     0,    -1.05], intermediate: [0, 0, -1.45], distal: [0, 0, -1.20] },
  },
  // Loose pinch — thumb + index pads near each other, others gently
  // curled. The "talking with hands" / explanatory gesture.
  pinch: {
    thumb:  { proximal: [0.30, -0.55, -0.35], intermediate: [0, 0, -0.40], distal: [0, 0, -0.40] },
    index:  { proximal: [0,    -0.20, -0.50], intermediate: [0, 0, -0.50], distal: [0, 0, -0.30] },
    middle: { proximal: [0,     0,    -0.30], intermediate: [0, 0, -0.40], distal: [0, 0, -0.30] },
    ring:   { proximal: [0,     0,    -0.35], intermediate: [0, 0, -0.45], distal: [0, 0, -0.35] },
    little: { proximal: [0,     0,    -0.40], intermediate: [0, 0, -0.50], distal: [0, 0, -0.40] },
  },
  // Open hand, fingers slightly spread, palm-out — waving / "stop" / "hi".
  waving: {
    thumb:  { proximal: [0,    -0.30, -0.20], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    index:  { proximal: [0,    -0.10,  0.05], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    middle: { proximal: [0,     0,     0   ], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    ring:   { proximal: [0,     0.10,  0   ], intermediate: [0, 0, 0], distal: [0, 0, 0] },
    little: { proximal: [0,     0.20, -0.05], intermediate: [0, 0, 0], distal: [0, 0, 0] },
  },
};

/** Names of valid hand poses (for validation / UI dropdowns). */
export const HAND_POSE_NAMES = Object.keys(HAND_POSES);

/**
 * Mirror a left-hand rotation triple for the right hand by negating Y
 * and Z. Author's note in scene-test calls out that the thumb's Y also
 * mirrors — this is consistent across all fingers, so no per-finger
 * special case needed.
 */
export function mirrorRotForRight(rot) {
  return [rot[0], -rot[1], -rot[2]];
}

/** Standard VRM normalized humanoid bone name for a finger joint.
 *  e.g. ('left', 'thumb', 'proximal') → 'leftThumbProximal'. */
export function fingerBoneName(side, finger, joint) {
  const F = finger[0].toUpperCase() + finger.slice(1);
  const J = joint[0].toUpperCase() + joint.slice(1);
  return `${side}${F}${J}`;
}
