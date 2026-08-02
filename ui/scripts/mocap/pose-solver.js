// ui/scripts/mocap/pose-solver.js
// Converts MediaPipe landmarks → VRM bone rotations
// Based on Kalidokit's proven approach (MIT licensed)
//
// Key principles:
// - 2D plane projections (atan2 on ZX, ZY, XY planes) for stable angle extraction
// - Normalized [-1, 1] output scaled by empirical constants
// - Absolute rotations (identity = T-pose on VRM normalized bones)
// - Left/right inversion for mirroring
// - Joint limit clamping
//
// MediaPipe world landmark coordinate system:
//   X: positive = subject's left
//   Y: negative = upward (head has more negative Y)
//   Z: negative = toward camera

const PI = Math.PI;
const TWO_PI = PI * 2;

const MP = {
  NOSE: 0,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
};

const ARKIT_TO_VRM = {
  eyeBlinkLeft: 'blinkLeft', eyeBlinkRight: 'blinkRight',
  jawOpen: 'aa',
  mouthSmileLeft: 'happy', mouthSmileRight: 'happy',
  mouthFunnel: 'ou', mouthPucker: 'ou',
  browDownLeft: 'angry', browDownRight: 'angry',
  browInnerUp: 'surprised', eyeWideLeft: 'surprised', eyeWideRight: 'surprised',
  mouthFrownLeft: 'sad', mouthFrownRight: 'sad',
};

// --- Math utilities (matching Kalidokit's approach) ---

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

/** 2D angle between two points */
function find2DAngle(cx, cy, ex, ey) {
  return Math.atan2(ey - cy, ex - cx);
}

/**
 * Normalize radians to [-1, 1] range (Kalidokit's normalizeRadians / PI)
 */
function normalizeAngle(radians) {
  if (radians >= PI / 2) radians -= TWO_PI;
  if (radians <= -PI / 2) { radians += TWO_PI; radians = PI - radians; }
  return radians / PI;
}

/**
 * Compute rotation from direction between two landmarks.
 * Projects onto three 2D planes (ZX, ZY, XY) independently.
 * Returns normalized [-1, 1] values.
 */
function findRotation(a, b) {
  return {
    x: normalizeAngle(find2DAngle(a.z, a.x, b.z, b.x)),
    y: normalizeAngle(find2DAngle(a.z, a.y, b.z, b.y)),
    z: normalizeAngle(find2DAngle(a.x, a.y, b.x, b.y)),
  };
}

/**
 * Angle at point B between vectors BA and BC.
 * Returns normalized value.
 */
function angleBetween3D(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const len1 = Math.sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
  const len2 = Math.sqrt(v2.x*v2.x + v2.y*v2.y + v2.z*v2.z);
  if (len1 < 1e-8 || len2 < 1e-8) return 0;
  const dot = (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z) / (len1 * len2);
  return normalizeAngle(Math.acos(clamp(dot, -1, 1)));
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

// --- Arm rigging (matching Kalidokit's rigArm pattern) ---

function rigArm(upperRot, lowerRot, side) {
  const invert = side === 'right' ? 1 : -1;

  // Scale and transform to VRM bone space
  const upper = {
    x: upperRot.x,
    y: upperRot.y * PI * invert,
    z: upperRot.z * -2.3 * invert,
  };

  // Compensate upper arm Y for forearm position
  upper.y -= Math.max(lowerRot.x, 0);
  upper.y -= -invert * Math.max(lowerRot.z, 0);
  upper.x -= 0.3 * invert; // Rest pose offset

  const lower = {
    x: lowerRot.x * 2.14 * invert,
    y: lowerRot.y * 2.14 * invert,
    z: lowerRot.z * -2.14 * invert,
  };

  // Clamp to human joint limits
  upper.x = clamp(upper.x, -0.5, PI);
  lower.x = clamp(lower.x, -0.3, 0.3);

  return { upper, lower };
}

// --- Rest pose defaults (when landmarks not visible) ---
const REST = {
  rightUpperArm: { x: 0, y: 0, z: -1.25 },
  leftUpperArm: { x: 0, y: 0, z: 1.25 },
  rightLowerArm: { x: 0, y: 0, z: 0 },
  leftLowerArm: { x: 0, y: 0, z: 0 },
  spine: { x: 0, y: 0, z: 0 },
  chest: { x: 0, y: 0, z: 0 },
  head: { x: 0, y: 0, z: 0 },
};

/**
 * Solve body pose from MediaPipe Pose Landmarker world landmarks.
 * Returns absolute rotations in radians for VRM normalized bones.
 * On VRM normalized bones, identity (0,0,0) = T-pose.
 */
export function solveBody(landmarks) {
  if (!landmarks || landmarks.length < 33) return {};

  const lm = landmarks;
  const bones = {};

  // --- Spine/Chest ---
  const hipsMid = midpoint(lm[MP.LEFT_HIP], lm[MP.RIGHT_HIP]);
  const shouldersMid = midpoint(lm[MP.LEFT_SHOULDER], lm[MP.RIGHT_SHOULDER]);
  const spineRot = findRotation(hipsMid, shouldersMid);
  bones.spine = [spineRot.x * 0.3, spineRot.y * 0.3, spineRot.z * 0.3]; // Subtle spine motion

  const neckApprox = midpoint(shouldersMid, lm[MP.NOSE]);
  const chestRot = findRotation(shouldersMid, neckApprox);
  bones.chest = [chestRot.x * 0.2, chestRot.y * 0.2, chestRot.z * 0.2]; // Even more subtle

  // --- Head (from ear/nose geometry) ---
  const earMid = midpoint(lm[MP.LEFT_EAR], lm[MP.RIGHT_EAR]);
  const headRot = findRotation(earMid, lm[MP.NOSE]);
  bones.head = [
    headRot.x * 0.8,   // pitch (nod)
    headRot.y * 0.6,   // yaw (turn)
    headRot.z * 0.3,   // roll (tilt) — subtle
  ];

  // --- Right Arm ---
  const rUpperRot = findRotation(lm[MP.RIGHT_SHOULDER], lm[MP.RIGHT_ELBOW]);
  rUpperRot.y = angleBetween3D(lm[MP.LEFT_SHOULDER], lm[MP.RIGHT_SHOULDER], lm[MP.RIGHT_ELBOW]);
  const rLowerRot = findRotation(lm[MP.RIGHT_ELBOW], lm[MP.RIGHT_WRIST]);
  rLowerRot.y = angleBetween3D(lm[MP.RIGHT_SHOULDER], lm[MP.RIGHT_ELBOW], lm[MP.RIGHT_WRIST]);

  const rVisible = (lm[MP.RIGHT_SHOULDER].visibility || 0) > 0.3 &&
                   (lm[MP.RIGHT_ELBOW].visibility || 0) > 0.3;
  if (rVisible) {
    const rArm = rigArm(rUpperRot, rLowerRot, 'right');
    bones.rightUpperArm = [rArm.upper.x, rArm.upper.y, rArm.upper.z];
    bones.rightLowerArm = [rArm.lower.x, rArm.lower.y, rArm.lower.z];
  } else {
    bones.rightUpperArm = [REST.rightUpperArm.x, REST.rightUpperArm.y, REST.rightUpperArm.z];
    bones.rightLowerArm = [REST.rightLowerArm.x, REST.rightLowerArm.y, REST.rightLowerArm.z];
  }

  // --- Left Arm ---
  const lUpperRot = findRotation(lm[MP.LEFT_SHOULDER], lm[MP.LEFT_ELBOW]);
  lUpperRot.y = angleBetween3D(lm[MP.RIGHT_SHOULDER], lm[MP.LEFT_SHOULDER], lm[MP.LEFT_ELBOW]);
  const lLowerRot = findRotation(lm[MP.LEFT_ELBOW], lm[MP.LEFT_WRIST]);
  lLowerRot.y = angleBetween3D(lm[MP.LEFT_SHOULDER], lm[MP.LEFT_ELBOW], lm[MP.LEFT_WRIST]);

  const lVisible = (lm[MP.LEFT_SHOULDER].visibility || 0) > 0.3 &&
                   (lm[MP.LEFT_ELBOW].visibility || 0) > 0.3;
  if (lVisible) {
    const lArm = rigArm(lUpperRot, lLowerRot, 'left');
    bones.leftUpperArm = [lArm.upper.x, lArm.upper.y, lArm.upper.z];
    bones.leftLowerArm = [lArm.lower.x, lArm.lower.y, lArm.lower.z];
  } else {
    bones.leftUpperArm = [REST.leftUpperArm.x, REST.leftUpperArm.y, REST.leftUpperArm.z];
    bones.leftLowerArm = [REST.leftLowerArm.x, REST.leftLowerArm.y, REST.leftLowerArm.z];
  }

  return bones;
}

/**
 * Map ARKit face blend shapes to VRM expression weights.
 */
export function solveFace(arkitShapes) {
  if (!arkitShapes) return {};
  const vrmExprs = {};
  for (const [arkit, vrm] of Object.entries(ARKIT_TO_VRM)) {
    const val = arkitShapes[arkit];
    if (val !== undefined && val > 0.01) {
      if (vrmExprs[vrm] !== undefined) {
        vrmExprs[vrm] = (vrmExprs[vrm] + val) / 2;
      } else {
        vrmExprs[vrm] = val;
      }
    }
  }
  return vrmExprs;
}
