/**
 * avatar-vrm-profile.js - Runtime compatibility profiling for loaded VRMs.
 *
 * The profile is intentionally descriptive. Animation code should still make
 * the final safety decision, but this gives presence/UI code a reliable view
 * of what the current avatar can probably do well.
 */
import {
  CALL_ACTION_ORDER,
  assessCallActionCompatibility,
} from './avatar-call-actions.js';

export const REQUIRED_VRM_BONES = [
  'hips', 'spine', 'head',
  'leftUpperArm', 'leftLowerArm', 'leftHand',
  'rightUpperArm', 'rightLowerArm', 'rightHand',
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot',
];

export const OPTIONAL_APP_BONES = [
  'chest', 'upperChest', 'neck',
  'leftShoulder', 'rightShoulder',
  'leftEye', 'rightEye',
];

export const FINGER_BONE_SUFFIXES = [
  'ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal',
  'IndexProximal', 'IndexIntermediate', 'IndexDistal',
  'MiddleProximal', 'MiddleIntermediate', 'MiddleDistal',
  'RingProximal', 'RingIntermediate', 'RingDistal',
  'LittleProximal', 'LittleIntermediate', 'LittleDistal',
];

const CALL_PHASES = ['idle', 'listening', 'processing', 'speaking'];

export function getVRMBone(humanoid, name) {
  return humanoid?.getNormalizedBoneNode?.(name)
    || humanoid?.getRawBoneNode?.(name)
    || null;
}

export function detectVRMGeneration(gltfJson) {
  const extensions = gltfJson?.extensions || {};
  if (extensions.VRMC_vrm) return '1.0';
  if (extensions.VRM) return '0.x';
  return 'unknown';
}

export function createAvatarCompatibilityProfile(THREE, vrm, options = {}) {
  const humanoid = vrm?.humanoid;
  const profile = {
    avatarId: options.avatarId || '',
    label: options.label || '',
    generation: detectVRMGeneration(options.gltfJson),
    facingCorrection: 'none',
    armAxisProfile: 'unknown',
    armLocalX: {
      leftLowerFromUpper: 0,
      rightLowerFromUpper: 0,
    },
    // Per-VRM finger axis convention. Detected by probing
    // {side}IndexProximal → {side}IndexIntermediate local-X direction
    // (same idiom as armAxisProfile). Drives the rest-relative hand-pose
    // channel's per-side sign correction so the canonical HAND_POSES
    // curl values produce palmar curl on every VRM, not dorsal
    // hyperextension on exports with the opposite convention.
    //   - 'legacy'   : left fingers extend along -X, right along +X (older bundled / VRoid 1.x style)
    //   - 'mirrored' : left fingers extend along +X, right along -X (VRoid Studio 2.x exports including Becca)
    //   - 'unknown'  : could not classify (low-poly VRM missing index proximal/intermediate)
    fingerAxisProfile: 'unknown',
    fingerLocalX: {
      leftIntermediateFromProximal: 0,
      rightIntermediateFromProximal: 0,
    },
    height: null,
    hipsHeight: null,
    requiredBonesOk: false,
    missingRequired: [...REQUIRED_VRM_BONES],
    missingOptional: [...OPTIONAL_APP_BONES],
    availableBones: [],
    handProfile: null,
    expressions: [],
    callActions: {},
    actionSummary: {
      autoReady: [],
      playable: [],
      limited: [],
      blocked: [],
    },
    recommendedTier: 'blocked',
    warnings: [],
  };

  if (!THREE || !vrm || !humanoid) {
    profile.warnings.push('VRM humanoid data is unavailable.');
    return profile;
  }

  const available = new Set();
  for (const boneName of [...REQUIRED_VRM_BONES, ...OPTIONAL_APP_BONES]) {
    if (getVRMBone(humanoid, boneName)) available.add(boneName);
  }

  const handProfile = detectHandProfile(humanoid);
  for (const side of ['left', 'right']) {
    for (const boneName of handProfile[side].present) available.add(boneName);
  }

  const leftUpper = getVRMBone(humanoid, 'leftUpperArm');
  const leftLower = getVRMBone(humanoid, 'leftLowerArm');
  const rightUpper = getVRMBone(humanoid, 'rightUpperArm');
  const rightLower = getVRMBone(humanoid, 'rightLowerArm');
  const leftX = getLocalChildX(THREE, leftUpper, leftLower);
  const rightX = getLocalChildX(THREE, rightUpper, rightLower);

  if (leftX < -0.001 && rightX > 0.001) profile.armAxisProfile = 'legacy';
  else if (leftX > 0.001 && rightX < -0.001) profile.armAxisProfile = 'mirrored';
  profile.armLocalX = {
    leftLowerFromUpper: round(leftX),
    rightLowerFromUpper: round(rightX),
  };
  profile.facingCorrection = profile.armAxisProfile === 'mirrored' ? 'rotateY180' : 'none';

  // Finger axis convention — independent probe so we don't assume it
  // correlates with arms (usually does, but VRMs with hand-bones authored
  // off the upstream pipeline can diverge). Probe the index proximal→
  // intermediate vector because index is the bone all five VRoid hand
  // exports we've seen actually emit. Falls back to middle finger when
  // index isn't present.
  const leftFinger = _probeFingerLocalX(THREE, humanoid, 'left');
  const rightFinger = _probeFingerLocalX(THREE, humanoid, 'right');
  if (leftFinger < -0.001 && rightFinger > 0.001) profile.fingerAxisProfile = 'legacy';
  else if (leftFinger > 0.001 && rightFinger < -0.001) profile.fingerAxisProfile = 'mirrored';
  profile.fingerLocalX = {
    leftIntermediateFromProximal: round(leftFinger),
    rightIntermediateFromProximal: round(rightFinger),
  };

  try {
    const box = new THREE.Box3().setFromObject(vrm.scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    profile.height = round(size.y);
    const hips = getVRMBone(humanoid, 'hips');
    if (hips) {
      profile.hipsHeight = round(hips.getWorldPosition(new THREE.Vector3()).y);
    }
  } catch {
    profile.warnings.push('Could not measure avatar height.');
  }

  profile.missingRequired = REQUIRED_VRM_BONES.filter((name) => !available.has(name));
  profile.missingOptional = OPTIONAL_APP_BONES.filter((name) => !available.has(name));
  profile.requiredBonesOk = profile.missingRequired.length === 0;
  profile.availableBones = [...available].sort();
  profile.handProfile = handProfile;
  profile.expressions = getExpressionNames(vrm);

  if (!profile.requiredBonesOk) {
    profile.warnings.push(`Missing required bones: ${profile.missingRequired.join(', ')}`);
  }
  if (profile.armAxisProfile === 'unknown') {
    profile.warnings.push('Could not classify arm axis profile.');
  }
  if (profile.fingerAxisProfile === 'unknown') {
    profile.warnings.push('Could not classify finger axis profile.');
  }
  if (!profile.expressions.length) {
    profile.warnings.push('No expression presets detected.');
  }

  for (const phase of CALL_PHASES) {
    profile.callActions[phase] = {};
    for (const action of CALL_ACTION_ORDER) {
      const compatibility = assessCallActionCompatibility(action, available, { phase });
      profile.callActions[phase][action] = summarizeActionCompatibility(compatibility);
    }
  }

  const speakingActions = profile.callActions.speaking || {};
  for (const [name, compatibility] of Object.entries(speakingActions)) {
    if (compatibility.canAuto) profile.actionSummary.autoReady.push(name);
    else if (compatibility.canPlay) profile.actionSummary.playable.push(name);
    else if (compatibility.status === 'fallback') profile.actionSummary.limited.push(name);
    else profile.actionSummary.blocked.push(name);
  }

  const hasCore = ['call_acknowledge', 'call_thoughtful_pause'].some(
    (name) => speakingActions[name]?.canAuto,
  );
  const hasHands = ['call_clarify_question', 'call_key_point'].some(
    (name) => speakingActions[name]?.canAuto,
  );
  profile.recommendedTier = !profile.requiredBonesOk
    ? 'blocked'
    : hasHands
      ? 'full-call'
      : hasCore
        ? 'core-call'
        : 'limited';

  return profile;
}

function detectHandProfile(humanoid) {
  const sides = {};
  for (const side of ['left', 'right']) {
    const present = [];
    const missing = [];
    for (const suffix of FINGER_BONE_SUFFIXES) {
      const name = `${side}${suffix}`;
      if (getVRMBone(humanoid, name)) present.push(name);
      else missing.push(name);
    }
    const optionalThumbIntermediate = `${side}ThumbIntermediate`;
    const optional = getVRMBone(humanoid, optionalThumbIntermediate)
      ? [optionalThumbIntermediate]
      : [];
    sides[side] = {
      present,
      missing,
      optional,
      coverage: FINGER_BONE_SUFFIXES.length
        ? present.length / FINGER_BONE_SUFFIXES.length
        : 0,
    };
  }
  return {
    left: sides.left,
    right: sides.right,
    summary: `L ${sides.left.present.length}/${FINGER_BONE_SUFFIXES.length}, R ${sides.right.present.length}/${FINGER_BONE_SUFFIXES.length}`,
  };
}

function getExpressionNames(vrm) {
  const manager = vrm?.expressionManager;
  if (!manager) return [];
  if (manager.expressionMap instanceof Map) return [...manager.expressionMap.keys()].sort();
  if (manager._expressionMap instanceof Map) return [...manager._expressionMap.keys()].sort();
  if (manager.expressions && typeof manager.expressions === 'object') {
    return Object.keys(manager.expressions).sort();
  }
  return [];
}

function getLocalChildX(THREE, parent, child) {
  if (!THREE || !parent || !child) return 0;
  parent.updateWorldMatrix?.(true, false);
  child.updateWorldMatrix?.(true, false);
  const point = child.getWorldPosition(new THREE.Vector3());
  return parent.worldToLocal(point).x;
}

// Probe a finger's proximal→intermediate local-X direction. Index first
// (universally present on hand-bearing VRMs); falls back to middle when
// the export omits index intermediate (rare but seen on stylised VRMs).
// Returns 0 when neither chain is available — fingerAxisProfile then
// stays 'unknown' and downstream code uses identity sign.
function _probeFingerLocalX(THREE, humanoid, side) {
  if (!humanoid) return 0;
  for (const finger of ['Index', 'Middle']) {
    const proximal = getVRMBone(humanoid, `${side}${finger}Proximal`);
    const intermediate = getVRMBone(humanoid, `${side}${finger}Intermediate`);
    if (proximal && intermediate) {
      return getLocalChildX(THREE, proximal, intermediate);
    }
  }
  return 0;
}

// The canonical HAND_POSES curl table in avatar-hand-poses.js was
// authored against the bundled-roster avatars (Becca et al. — VRoid
// Studio 2.x exports, finger-axis profile = 'mirrored'). VRMs with the
// opposite ('legacy') convention see -Z curl as dorsal-side rotation
// and the fingers visibly hyperextend. The animator's hand-pose channel
// multiplies its delta Euler by this sign factor (in addition to the
// left/right mirror that's intrinsic to the table) to keep the curl
// palmar on every VRM.
//
// Symmetric with _armTargetSign in avatar-animator.js, which solves the
// same kind of bug for arm spring targets — the two channels disagreed
// previously because hand-pose had no per-VRM correction at all.
export const HAND_POSE_AUTHOR_FINGER_AXIS_PROFILE = 'mirrored';

export function fingerAxisSignFromProfile(profile) {
  if (!profile || profile === 'unknown') return { x: 1, y: 1, z: 1 };
  if (profile === HAND_POSE_AUTHOR_FINGER_AXIS_PROFILE) return { x: 1, y: 1, z: 1 };
  return { x: 1, y: 1, z: -1 };
}

// Arm sign factor for off-animator callers (pose preset application,
// affordance applier, future orchestrator wiring). Takes the consumer's
// `author` convention because authored tables disagree:
//   - The animator's IDLE_ARM_POSE (Z=+80 for left) is authored LEGACY
//     and gets Z negated for mirrored VRMs (handled in-animator via
//     _detectArmTargetSign — this function isn't used there).
//   - POSE_PRESETS arm bones (Z=-1.35 for leftUpperArm) are authored
//     MIRRORED (scene-test authored against Becca et al.) and need Z
//     negated for legacy VRMs.
//
// Pass `'mirrored'` (the default) when applying POSE_PRESETS data; pass
// `'legacy'` if you're consuming an older table.
export function armAxisSignFromProfile(profile, author = 'mirrored') {
  if (!profile || profile === 'unknown') return { x: 1, y: 1, z: 1 };
  if (profile === author) return { x: 1, y: 1, z: 1 };
  return { x: 1, y: 1, z: -1 };
}

function summarizeActionCompatibility(compatibility) {
  return {
    name: compatibility.name,
    label: compatibility.label,
    status: compatibility.status,
    canPlay: compatibility.canPlay,
    canAuto: compatibility.canAuto,
    score: round(compatibility.score),
    poseCoverage: round(compatibility.poseCoverage),
    handCoverage: round(compatibility.handCoverage),
    motionCoverage: round(compatibility.motionCoverage),
    missing: compatibility.missing || [],
    warnings: compatibility.warnings || [],
    fallback: compatibility.fallback,
  };
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : value;
}
