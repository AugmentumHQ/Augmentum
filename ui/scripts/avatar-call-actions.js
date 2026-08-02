// Curated mid-call avatar actions shared by production and the normalization lab.
// These are intentionally upper-body only: useful while speaking/listening,
// restrained enough for desktop calls, and compatible with normalized VRM bones.

import {
  HAND_POSES,
  FINGER_NAMES,
  JOINT_NAMES,
} from './avatar-hand-poses.js';

// HAND_SHAPES — historically the call-action system's own finger-rotation
// catalog (in degrees, with `Suffix`-keyed bones). It collided with the
// idle hand-pose channel: HAND_SHAPES.relaxed was nearly straight while
// the hand-pose channel's relaxed had natural curl, so every call-action
// turn made fingers visibly extend then re-curl.
//
// Resolved by deriving HAND_SHAPES from the canonical HAND_POSES (single
// source of truth for finger curl, in radians + finger.joint structure).
// The derived shape preserves the legacy contract (label + flat bones
// dict in degrees) so getHandShapeBones / assessCallActionCompatibility
// keep working unchanged. Call-action handShapes data was renamed in
// place to use HAND_POSES names ('openPalm' → 'open', 'softPoint' →
// 'point', 'gentleStop' → 'open').
//
// Note: HAND_POSES does not author the ThumbMetacarpal joint, while the
// old HAND_SHAPES did (slight thumb-base spread). VRMs with that bone
// no longer get a metacarpal write from call-actions; if needed, that
// joint can be added to HAND_POSES authoring later.
const _RAD_TO_DEG = 180 / Math.PI;
function _deriveHandShapes() {
  const out = {};
  for (const [name, pose] of Object.entries(HAND_POSES)) {
    const bones = {};
    for (const finger of FINGER_NAMES) {
      const segs = pose[finger];
      if (!segs) continue;
      const F = finger[0].toUpperCase() + finger.slice(1);
      for (const joint of JOINT_NAMES) {
        const rot = segs[joint];
        if (!rot) continue;
        const J = joint[0].toUpperCase() + joint.slice(1);
        bones[`${F}${J}`] = [
          rot[0] * _RAD_TO_DEG,
          rot[1] * _RAD_TO_DEG,
          rot[2] * _RAD_TO_DEG,
        ];
      }
    }
    out[name] = { label: name, bones };
  }
  return out;
}
export const HAND_SHAPES = _deriveHandShapes();

export const DEFAULT_CALL_MOTION_TEXTURE = {
  label: 'Quiet conversational drift',
  frequency: 0.82,
  holdStart: 0.16,
  holdEnd: 0.88,
  bones: {
    chest: [0.2, 0.04, 0],
    neck: [0.08, 0.06, 0.04],
    head: [0.12, 0.1, 0.05],
  },
};

export const CALL_ACTIONS = {
  call_acknowledge: {
    label: 'Acknowledge',
    cue: 'Confirms that the AI is following or agrees.',
    inspiredBy: 'VMagicMirror wait motion with a compact conversational nod.',
    duration: 1.18,
    expression: { relaxed: 0.08, happy: 0.08 },
    handShapes: { left: 'relaxed', right: 'relaxed' },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.22, bones: { head: [-5.2, 0, 0], neck: [-2.2, 0, 0], chest: [-0.6, 0, 0] } },
      { t: 0.46, bones: { head: [1.6, 0, 0], neck: [0.8, 0, 0], chest: [0.2, 0, 0] } },
      { t: 0.68, bones: { head: [-3.8, 0, 0], neck: [-1.4, 0, 0], chest: [-0.4, 0, 0] } },
      { t: 1, bones: {} },
    ],
  },

  call_greeting_wave: {
    label: 'Greeting Wave',
    cue: 'A restrained hello or handoff at the start or end of a call.',
    inspiredBy: 'TalkingHead handup/wave gestures and VMagicMirror hand tracking greetings.',
    duration: 1.82,
    expression: { happy: 0.18, relaxed: 0.08 },
    handShapes: { right: 'open' },
    motionTexture: {
      label: 'Small wrist follow-through',
      frequency: 1.45,
      holdStart: 0.18,
      holdEnd: 0.82,
      bones: {
        chest: [0.16, 0.06, 0],
        head: [0.12, 0.16, 0.08],
        rightHand: [0.8, 0.5, 4.2],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.18, bones: {
        spine: [-0.8, 0, 0],
        chest: [-1.0, 0, 0],
        head: [-2.4, 2.2, 0.8],
        rightUpperArm: [-30, -10, -58],
        rightLowerArm: [-76, 10, 0],
        rightHand: [-8, -6, 16],
      } },
      { t: 0.35, bones: {
        spine: [-0.8, 0, 0],
        chest: [-1.0, 0, 0],
        head: [-2.4, 2.2, 0.8],
        rightUpperArm: [-32, -10, -54],
        rightLowerArm: [-74, 12, 0],
        rightHand: [-8, 6, 8],
      } },
      { t: 0.52, bones: {
        spine: [-0.8, 0, 0],
        chest: [-1.0, 0, 0],
        head: [-2.4, 2.2, 0.8],
        rightUpperArm: [-30, -10, -58],
        rightLowerArm: [-76, 10, 0],
        rightHand: [-8, -6, 16],
      } },
      { t: 0.68, bones: {
        spine: [-0.8, 0, 0],
        chest: [-1.0, 0, 0],
        head: [-2.4, 2.2, 0.8],
        rightUpperArm: [-32, -10, -54],
        rightLowerArm: [-74, 12, 0],
        rightHand: [-8, 6, 10],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_attentive_lean: {
    label: 'Attentive Lean',
    cue: 'Shows active listening without interrupting.',
    inspiredBy: 'VMagicMirror upper-body wait motion and eye-contact behavior.',
    duration: 1.75,
    expression: { relaxed: 0.1, surprised: 0.06 },
    handShapes: { left: 'relaxed', right: 'relaxed' },
    motionTexture: {
      label: 'Held listening breath',
      frequency: 0.58,
      holdStart: 0.22,
      holdEnd: 0.78,
      bones: {
        chest: [0.28, 0.06, 0],
        head: [0.16, 0.16, 0.08],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.22, bones: {
        spine: [-2.6, 0, 0],
        chest: [-2.1, 0, 0],
        head: [2.2, -3.2, 1.4],
        neck: [0.8, -1.3, 0.4],
        leftUpperArm: [14, 3, 70],
        rightUpperArm: [14, -3, -70],
      } },
      { t: 0.72, bones: {
        spine: [-3.2, 0, 0],
        chest: [-2.5, 0, 0],
        head: [1.2, 3.2, -1.2],
        neck: [0.5, 1.2, -0.4],
        leftUpperArm: [14, 3, 70],
        rightUpperArm: [14, -3, -70],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_clarify_question: {
    label: 'Clarify Question',
    cue: 'A small palm-up question gesture.',
    inspiredBy: 'TalkingHead side/handup gestures adapted as a low call-safe question.',
    duration: 1.55,
    expression: { surprised: 0.18, relaxed: 0.06 },
    handShapes: { right: 'open' },
    motionTexture: {
      label: 'Palm-up uncertainty',
      frequency: 0.92,
      holdStart: 0.22,
      holdEnd: 0.76,
      bones: {
        chest: [0.18, 0.04, 0],
        head: [0.12, 0.2, 0.12],
        rightHand: [0.45, 0.35, 1.25],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.25, bones: {
        head: [3.8, 5.4, 4.2],
        neck: [1.2, 2.0, 1.4],
        rightUpperArm: [-16, -14, -58],
        rightLowerArm: [-74, 12, 0],
        rightHand: [-12, 6, 16],
      } },
      { t: 0.62, bones: {
        head: [4.2, 5.4, 4.2],
        neck: [1.3, 2.0, 1.4],
        rightUpperArm: [-16, -14, -58],
        rightLowerArm: [-74, 12, 0],
        rightHand: [-12, 6, 16],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_reassure_open: {
    label: 'Reassure Open',
    cue: 'Soft, low open-palms reassurance.',
    inspiredBy: 'VMagicMirror presentation-like hands with a gentler therapeutic call posture.',
    duration: 1.85,
    expression: { relaxed: 0.16, happy: 0.1 },
    handShapes: { left: 'open', right: 'open' },
    motionTexture: {
      label: 'Slow open-palms breath',
      frequency: 0.66,
      holdStart: 0.24,
      holdEnd: 0.78,
      bones: {
        chest: [0.28, 0.02, 0],
        head: [0.12, 0.08, 0.04],
        leftHand: [0.24, -0.25, -0.5],
        rightHand: [0.24, 0.25, 0.5],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.28, bones: {
        spine: [-0.8, 0, 0],
        chest: [-1.2, 0, 0],
        head: [-1.4, 0, 0],
        leftUpperArm: [8, 8, 62],
        rightUpperArm: [8, -8, -62],
        leftLowerArm: [-24, -10, -5],
        rightLowerArm: [-24, 10, 5],
        leftHand: [-9, -10, -18],
        rightHand: [-9, 10, 18],
      } },
      { t: 0.68, bones: {
        spine: [-0.8, 0, 0],
        chest: [-1.2, 0, 0],
        head: [-1.4, 0, 0],
        leftUpperArm: [8, 8, 62],
        rightUpperArm: [8, -8, -62],
        leftLowerArm: [-24, -10, -5],
        rightLowerArm: [-24, 10, 5],
        leftHand: [-9, -10, -18],
        rightHand: [-9, 10, 18],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_key_point: {
    label: 'Key Point',
    cue: 'A compact presentation hand for important points.',
    inspiredBy: 'TalkingHead index gesture constrained for desktop call framing.',
    duration: 1.35,
    expression: { surprised: 0.08, relaxed: 0.06 },
    handShapes: { right: 'point' },
    motionTexture: {
      label: 'Presentation beat',
      frequency: 1.05,
      holdStart: 0.2,
      holdEnd: 0.72,
      bones: {
        chest: [0.18, 0.04, 0],
        head: [0.08, 0.16, 0],
        rightHand: [0.45, 0.2, 0.8],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.22, bones: {
        chest: [-1.0, 0, 0],
        head: [-2.0, 3.0, 0],
        rightUpperArm: [-22, -12, -58],
        rightLowerArm: [-76, 12, 0],
        rightHand: [-12, 5, 8],
      } },
      { t: 0.58, bones: {
        chest: [-1.0, 0, 0],
        head: [-2.0, 3.0, 0],
        rightUpperArm: [-22, -12, -58],
        rightLowerArm: [-76, 12, 0],
        rightHand: [-12, 5, 8],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_precise_detail: {
    label: 'Precise Detail',
    cue: 'A small OK/pinch shape for a careful distinction or exact detail.',
    inspiredBy: 'TalkingHead ok gesture converted to a close-to-body precision cue.',
    duration: 1.5,
    expression: { relaxed: 0.08, surprised: 0.06 },
    handShapes: { right: 'pinch' },
    motionTexture: {
      label: 'Fine-detail hand drift',
      frequency: 0.96,
      holdStart: 0.22,
      holdEnd: 0.78,
      bones: {
        chest: [0.16, 0.04, 0],
        head: [0.08, 0.14, 0.06],
        rightHand: [0.28, 0.22, 0.65],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.24, bones: {
        chest: [-0.6, 0, 0],
        head: [1.0, 2.4, 0.6],
        neck: [0.4, 0.9, 0.2],
        rightUpperArm: [-24, -10, -66],
        rightLowerArm: [-92, 14, 0],
        rightHand: [-18, 7, 14],
      } },
      { t: 0.64, bones: {
        chest: [-0.6, 0, 0],
        head: [1.0, 2.4, 0.6],
        neck: [0.4, 0.9, 0.2],
        rightUpperArm: [-24, -10, -66],
        rightLowerArm: [-92, 14, 0],
        rightHand: [-18, 7, 14],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_compare_options: {
    label: 'Compare Options',
    cue: 'Alternates attention between two choices.',
    inspiredBy: 'Presentation-like hand alternation from VMagicMirror-style upper-body control.',
    duration: 2.2,
    expression: { relaxed: 0.08, surprised: 0.06 },
    handShapes: { left: 'open', right: 'open' },
    motionTexture: {
      label: 'Two-option weighing',
      frequency: 0.72,
      holdStart: 0.2,
      holdEnd: 0.82,
      bones: {
        chest: [0.18, 0.14, 0],
        head: [0.1, 0.22, 0.08],
        leftHand: [0.32, -0.2, -0.55],
        rightHand: [0.32, 0.2, 0.55],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.24, bones: {
        head: [1.2, -5.0, -1.5],
        leftUpperArm: [8, 7, 60],
        leftLowerArm: [-30, -13, -2],
        leftHand: [-10, -8, -16],
      } },
      { t: 0.5, bones: {
        head: [1.2, 5.0, 1.5],
        rightUpperArm: [8, -7, -60],
        rightLowerArm: [-30, 13, 2],
        rightHand: [-10, 8, 16],
      } },
      { t: 0.74, bones: {
        head: [0.6, 0, 0],
        leftUpperArm: [9, 6, 58],
        rightUpperArm: [9, -6, -58],
        leftLowerArm: [-27, -10, -2],
        rightLowerArm: [-27, 10, 2],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_side_reference: {
    label: 'Side Reference',
    cue: 'Points to an external item, page, tool result, or side topic without hard pointing.',
    inspiredBy: 'TalkingHead side gesture and VMagicMirror presentation-like hand.',
    duration: 1.72,
    expression: { relaxed: 0.08, surprised: 0.04 },
    handShapes: { right: 'open' },
    motionTexture: {
      label: 'Side-presenting hold',
      frequency: 0.8,
      holdStart: 0.2,
      holdEnd: 0.82,
      bones: {
        chest: [0.16, 0.12, 0],
        head: [0.08, 0.2, 0.06],
        rightHand: [0.3, 0.2, 0.9],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.24, bones: {
        spine: [-0.4, 0, 0],
        chest: [-0.8, 1.2, 0],
        head: [-1.0, 4.4, 0.6],
        neck: [-0.3, 1.4, 0.2],
        rightUpperArm: [-12, -20, -56],
        rightLowerArm: [-48, 16, 2],
        rightHand: [-8, 9, 18],
      } },
      { t: 0.68, bones: {
        spine: [-0.4, 0, 0],
        chest: [-0.8, 1.2, 0],
        head: [-1.0, 4.4, 0.6],
        neck: [-0.3, 1.4, 0.2],
        rightUpperArm: [-12, -20, -56],
        rightLowerArm: [-48, 16, 2],
        rightHand: [-8, 9, 18],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_thoughtful_pause: {
    label: 'Thoughtful Pause',
    cue: 'A restrained thinking beat before answering.',
    inspiredBy: 'Kalidokit/VTuber gaze-aversion conventions with VMagicMirror wait motion restraint.',
    duration: 2.15,
    expression: { relaxed: 0.14 },
    handShapes: { left: 'relaxed', right: 'relaxed' },
    motionTexture: {
      label: 'Quiet thinking drift',
      frequency: 0.48,
      holdStart: 0.22,
      holdEnd: 0.82,
      bones: {
        chest: [0.2, 0.04, 0],
        head: [0.12, 0.18, 0.1],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.24, bones: {
        spine: [1.6, 0, 0.4],
        chest: [1.2, 0, 0],
        head: [5.5, 7.0, 3.0],
        neck: [2.0, 2.4, 1.0],
        leftUpperArm: [14, 2, 70],
        rightUpperArm: [14, -2, -70],
      } },
      { t: 0.72, bones: {
        spine: [1.6, 0, 0.4],
        chest: [1.2, 0, 0],
        head: [5.5, 7.0, 3.0],
        neck: [2.0, 2.4, 1.0],
        leftUpperArm: [14, 2, 70],
        rightUpperArm: [14, -2, -70],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_soft_shrug: {
    label: 'Soft Shrug',
    cue: 'A mild uncertainty gesture for “it depends” without feeling dismissive.',
    inspiredBy: 'TalkingHead shrug adapted to low, professional call framing.',
    duration: 1.38,
    expression: { surprised: 0.12, relaxed: 0.08 },
    handShapes: { left: 'open', right: 'open' },
    motionTexture: {
      label: 'Shoulder settle',
      frequency: 0.7,
      holdStart: 0.2,
      holdEnd: 0.74,
      bones: {
        chest: [0.2, 0, 0],
        head: [0.1, 0.08, 0.16],
        leftHand: [0.22, -0.18, -0.35],
        rightHand: [0.22, 0.18, 0.35],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.22, bones: {
        spine: [0.6, 0, 0],
        chest: [0.8, 0, 0],
        head: [1.4, -1.8, 3.2],
        leftShoulder: [-7.5, 0, 0],
        rightShoulder: [-7.5, 0, 0],
        leftUpperArm: [8, 10, 62],
        rightUpperArm: [8, -10, -62],
        leftLowerArm: [-20, -10, -2],
        rightLowerArm: [-20, 10, 2],
        leftHand: [-7, -8, -18],
        rightHand: [-7, 8, 18],
      } },
      { t: 0.56, bones: {
        spine: [0.6, 0, 0],
        chest: [0.8, 0, 0],
        head: [1.4, -1.8, 3.2],
        leftShoulder: [-7.5, 0, 0],
        rightShoulder: [-7.5, 0, 0],
        leftUpperArm: [8, 10, 62],
        rightUpperArm: [8, -10, -62],
        leftLowerArm: [-20, -10, -2],
        rightLowerArm: [-20, 10, 2],
        leftHand: [-7, -8, -18],
        rightHand: [-7, 8, 18],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_light_laugh: {
    label: 'Light Laugh',
    cue: 'Warm, small laugh motion without flailing.',
    inspiredBy: 'TalkingHead emoji/expression timing constrained to subtle call movement.',
    duration: 1.42,
    expression: { happy: 0.34, relaxed: 0.08 },
    handShapes: { left: 'relaxed', right: 'relaxed' },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.18, bones: { spine: [-2.2, 0, 0], chest: [-2.8, 0, 0], head: [-4.2, 0, 1.2], leftShoulder: [-2.0, 0, 0], rightShoulder: [-2.0, 0, 0] } },
      { t: 0.38, bones: { spine: [1.6, 0, 0], chest: [1.8, 0, 0], head: [2.0, 0, -1.0], leftShoulder: [1.0, 0, 0], rightShoulder: [1.0, 0, 0] } },
      { t: 0.58, bones: { spine: [-1.6, 0, 0], chest: [-2.0, 0, 0], head: [-3.0, 0, 0.6], leftShoulder: [-1.2, 0, 0], rightShoulder: [-1.2, 0, 0] } },
      { t: 1, bones: {} },
    ],
  },

  call_gentle_no: {
    label: 'Gentle No',
    cue: 'A correction or boundary without feeling harsh.',
    inspiredBy: 'Conversational head shake plus a TalkingHead-style stop hand at reduced amplitude.',
    duration: 1.45,
    expression: { relaxed: 0.08 },
    handShapes: { right: 'open' },
    motionTexture: {
      label: 'Soft boundary hold',
      frequency: 0.76,
      holdStart: 0.18,
      holdEnd: 0.76,
      bones: {
        chest: [0.14, 0.06, 0],
        rightHand: [0.22, 0.12, 0.65],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.2, bones: {
        head: [0, 7, 0],
        neck: [0, 2.6, 0],
        rightUpperArm: [-10, -12, -58],
        rightLowerArm: [-68, 8, 0],
        rightHand: [-8, 0, 18],
      } },
      { t: 0.42, bones: {
        head: [0, -7, 0],
        neck: [0, -2.6, 0],
        rightUpperArm: [-10, -12, -58],
        rightLowerArm: [-68, 8, 0],
        rightHand: [-8, 0, 18],
      } },
      { t: 0.66, bones: {
        head: [0, 4, 0],
        neck: [0, 1.4, 0],
        rightUpperArm: [-10, -12, -58],
        rightLowerArm: [-68, 8, 0],
        rightHand: [-8, 0, 18],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_grounding_breath: {
    label: 'Grounding Breath',
    cue: 'A visible but quiet inhale/exhale while processing or resetting tone.',
    inspiredBy: 'VMagicMirror wait motion and deep-breath idle behavior.',
    duration: 2.75,
    expression: { relaxed: 0.16 },
    handShapes: { left: 'relaxed', right: 'relaxed' },
    motionTexture: {
      label: 'Long exhale settle',
      frequency: 0.38,
      holdStart: 0.16,
      holdEnd: 0.9,
      bones: {
        chest: [0.36, 0, 0],
        upperChest: [0.24, 0, 0],
        head: [0.16, 0.08, 0.04],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.26, bones: {
        spine: [-0.7, 0, 0],
        chest: [-2.2, 0, 0],
        upperChest: [-1.2, 0, 0],
        head: [-1.0, 0, 0],
        leftShoulder: [-0.8, 0, 0],
        rightShoulder: [-0.8, 0, 0],
      } },
      { t: 0.58, bones: {
        spine: [1.2, 0, 0],
        chest: [1.9, 0, 0],
        upperChest: [0.9, 0, 0],
        head: [1.4, 0, 0],
        leftShoulder: [0.8, 0, 0],
        rightShoulder: [0.8, 0, 0],
      } },
      { t: 0.78, bones: {
        spine: [0.4, 0, 0],
        chest: [0.6, 0, 0],
        upperChest: [0.2, 0, 0],
        head: [0.5, 0, 0],
        leftShoulder: [0.2, 0, 0],
        rightShoulder: [0.2, 0, 0],
      } },
      { t: 1, bones: {} },
    ],
  },

  call_wrap_up: {
    label: 'Wrap Up',
    cue: 'Signals conclusion and settles back to idle.',
    inspiredBy: 'VMagicMirror presentation hand settling back into wait motion.',
    duration: 1.65,
    expression: { relaxed: 0.12, happy: 0.08 },
    handShapes: { left: 'open', right: 'open' },
    motionTexture: {
      label: 'Closing settle',
      frequency: 0.62,
      holdStart: 0.2,
      holdEnd: 0.78,
      bones: {
        chest: [0.22, 0.02, 0],
        head: [0.1, 0.06, 0.04],
        leftHand: [0.2, -0.14, -0.35],
        rightHand: [0.2, 0.14, 0.35],
      },
    },
    keyframes: [
      { t: 0, bones: {} },
      { t: 0.24, bones: {
        spine: [1.2, 0, 0],
        chest: [0.8, 0, 0],
        head: [-4.2, 0, 0],
        leftUpperArm: [10, 6, 68],
        rightUpperArm: [10, -6, -68],
        leftLowerArm: [-18, -6, -3],
        rightLowerArm: [-18, 6, 3],
        leftHand: [-7, -4, -6],
        rightHand: [-7, 4, 6],
      } },
      { t: 0.62, bones: {
        spine: [1.2, 0, 0],
        chest: [0.8, 0, 0],
        head: [-4.2, 0, 0],
        leftUpperArm: [10, 6, 68],
        rightUpperArm: [10, -6, -68],
        leftLowerArm: [-18, -6, -3],
        rightLowerArm: [-18, 6, 3],
        leftHand: [-7, -4, -6],
        rightHand: [-7, 4, 6],
      } },
      { t: 1, bones: {} },
    ],
  },
};

export const CALL_ACTION_ORDER = Object.keys(CALL_ACTIONS);

const DEFAULT_CALL_ACTION_CONTRACT = {
  intent: 'conversational beat',
  phases: ['speaking'],
  safetyTier: 'stable',
  autoSafe: false,
  explicitSafe: true,
  interrupt: 'soft',
  priority: 30,
  cooldown: 1.4,
  cooldownGroup: null,
  minProgressBeforeInterrupt: 0.68,
  minPoseCoverage: 0.72,
  minHandCoverage: 0,
  minScore: 0.7,
  minAutoScore: 0.92,
  handRequirement: 'optional',
  coreBones: ['head'],
  fallback: 'call_acknowledge',
  frameRisk: 'low',
};

export const CALL_ACTION_CONTRACTS = {
  call_acknowledge: {
    intent: 'acknowledgement',
    phases: ['speaking', 'listening'],
    safetyTier: 'core',
    autoSafe: true,
    priority: 20,
    cooldown: 1.0,
    cooldownGroup: 'head-small',
    minAutoScore: 0.82,
    coreBones: ['head'],
    fallback: null,
  },

  call_greeting_wave: {
    intent: 'greeting or handoff',
    phases: ['idle', 'speaking'],
    safetyTier: 'directed',
    autoSafe: false,
    priority: 70,
    cooldown: 4.0,
    cooldownGroup: 'right-arm-high',
    handRequirement: 'important',
    coreBones: ['head', 'rightUpperArm', 'rightLowerArm', 'rightHand'],
    fallback: 'call_acknowledge',
    frameRisk: 'high',
  },

  call_attentive_lean: {
    intent: 'active listening',
    phases: ['listening', 'processing'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 28,
    cooldown: 2.4,
    cooldownGroup: 'torso-listening',
    minAutoScore: 0.82,
    coreBones: ['spine', 'head'],
    fallback: 'call_acknowledge',
  },

  call_clarify_question: {
    intent: 'clarifying question',
    phases: ['speaking', 'listening'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 38,
    cooldown: 2.2,
    cooldownGroup: 'right-arm-low',
    handRequirement: 'optional',
    minAutoScore: 0.86,
    coreBones: ['head', 'rightUpperArm', 'rightLowerArm'],
    fallback: 'call_acknowledge',
  },

  call_reassure_open: {
    intent: 'soft reassurance',
    phases: ['speaking'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 36,
    cooldown: 2.8,
    cooldownGroup: 'two-hand-open',
    handRequirement: 'important',
    coreBones: ['head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm'],
    fallback: 'call_acknowledge',
  },

  call_key_point: {
    intent: 'important point',
    phases: ['speaking'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 44,
    cooldown: 2.4,
    cooldownGroup: 'right-arm-point',
    handRequirement: 'important',
    coreBones: ['head', 'rightUpperArm', 'rightLowerArm'],
    fallback: 'call_acknowledge',
  },

  call_precise_detail: {
    intent: 'precise detail',
    phases: ['speaking'],
    safetyTier: 'directed',
    autoSafe: false,
    priority: 46,
    cooldown: 2.8,
    cooldownGroup: 'right-arm-point',
    handRequirement: 'important',
    minHandCoverage: 0.7,
    minScore: 0.82,
    coreBones: ['head', 'rightUpperArm', 'rightLowerArm', 'rightHand'],
    fallback: 'call_key_point',
    frameRisk: 'medium',
  },

  call_compare_options: {
    intent: 'compare alternatives',
    phases: ['speaking'],
    safetyTier: 'directed',
    autoSafe: false,
    priority: 48,
    cooldown: 3.2,
    cooldownGroup: 'two-hand-open',
    handRequirement: 'important',
    coreBones: ['head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm'],
    fallback: 'call_reassure_open',
    frameRisk: 'medium',
  },

  call_side_reference: {
    intent: 'reference side content',
    phases: ['speaking'],
    safetyTier: 'directed',
    autoSafe: false,
    priority: 42,
    cooldown: 2.4,
    cooldownGroup: 'right-arm-low',
    handRequirement: 'optional',
    coreBones: ['head', 'rightUpperArm', 'rightLowerArm'],
    fallback: 'call_key_point',
    frameRisk: 'medium',
  },

  call_thoughtful_pause: {
    intent: 'thinking pause',
    phases: ['processing', 'speaking'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 24,
    cooldown: 3.2,
    cooldownGroup: 'thinking',
    minAutoScore: 0.82,
    coreBones: ['head', 'spine'],
    fallback: 'call_acknowledge',
  },

  call_soft_shrug: {
    intent: 'uncertainty',
    phases: ['speaking'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 40,
    cooldown: 2.8,
    cooldownGroup: 'two-hand-open',
    handRequirement: 'important',
    coreBones: ['head', 'leftShoulder', 'rightShoulder'],
    fallback: 'call_acknowledge',
  },

  call_light_laugh: {
    intent: 'warm laugh',
    phases: ['speaking'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 58,
    cooldown: 3.8,
    cooldownGroup: 'expression',
    minAutoScore: 0.82,
    coreBones: ['head', 'spine'],
    fallback: 'call_acknowledge',
  },

  call_gentle_no: {
    intent: 'gentle correction',
    phases: ['speaking'],
    safetyTier: 'directed',
    autoSafe: false,
    priority: 54,
    cooldown: 3.0,
    cooldownGroup: 'right-arm-boundary',
    handRequirement: 'optional',
    coreBones: ['head', 'rightUpperArm', 'rightLowerArm'],
    fallback: 'call_acknowledge',
  },

  call_grounding_breath: {
    intent: 'grounding reset',
    phases: ['processing', 'idle', 'listening'],
    safetyTier: 'core',
    autoSafe: true,
    priority: 18,
    cooldown: 4.2,
    cooldownGroup: 'breath',
    minAutoScore: 0.82,
    coreBones: ['head', 'spine'],
    fallback: 'call_acknowledge',
  },

  call_wrap_up: {
    intent: 'conclusion',
    phases: ['speaking'],
    safetyTier: 'stable',
    autoSafe: true,
    priority: 34,
    cooldown: 3.0,
    cooldownGroup: 'two-hand-open',
    handRequirement: 'important',
    coreBones: ['head', 'leftUpperArm', 'rightUpperArm'],
    fallback: 'call_acknowledge',
  },
};

export function getCallAction(name) {
  return CALL_ACTIONS[name] || null;
}

export function getCallActionContract(name) {
  if (!CALL_ACTIONS[name]) return null;
  const contract = { ...DEFAULT_CALL_ACTION_CONTRACT, ...(CALL_ACTION_CONTRACTS[name] || {}) };
  contract.cooldownGroup = contract.cooldownGroup || name;
  return contract;
}

export function getCallActionBones(action) {
  action = typeof action === 'string' ? getCallAction(action) : action;
  const bones = new Set();
  for (const keyframe of action?.keyframes || []) {
    for (const boneName of Object.keys(keyframe.bones || {})) {
      bones.add(boneName);
    }
  }
  return [...bones];
}

export function getHandShapeBones(shapeName, side) {
  const shape = HAND_SHAPES[shapeName];
  if (!shape || !side) return [];
  return Object.keys(shape.bones || {}).map((suffix) => `${side}${suffix}`);
}

export function getCallActionHandShapeBones(action) {
  action = typeof action === 'string' ? getCallAction(action) : action;
  const bones = new Set();
  for (const [side, shapeName] of Object.entries(action?.handShapes || {})) {
    for (const boneName of getHandShapeBones(shapeName, side)) {
      bones.add(boneName);
    }
  }
  return [...bones];
}

export function getCallActionMotionBones(action) {
  action = typeof action === 'string' ? getCallAction(action) : action;
  const texture = action?.motionTexture || DEFAULT_CALL_MOTION_TEXTURE;
  return Object.keys(texture?.bones || {});
}

export function getCallActionRequirements(name) {
  const action = getCallAction(name);
  const contract = getCallActionContract(name);
  if (!action || !contract) return null;
  const poseBones = getCallActionBones(action);
  const handShapeBones = getCallActionHandShapeBones(action);
  const motionBones = getCallActionMotionBones(action);
  return {
    contract,
    poseBones,
    handShapeBones,
    motionBones,
    coreBones: [...new Set(contract.coreBones || [])],
    allBones: [...new Set([...poseBones, ...handShapeBones, ...motionBones, ...(contract.coreBones || [])])],
  };
}

export function assessCallActionCompatibility(name, availableBones, options = {}) {
  const action = getCallAction(name);
  const requirements = getCallActionRequirements(name);
  if (!action || !requirements) {
    return {
      name,
      status: 'unknown',
      canPlay: false,
      canAuto: false,
      score: 0,
      coverage: 0,
      missing: [],
      warnings: [`Unknown call action: ${name}`],
      fallback: null,
    };
  }

  const available = _toAvailableBoneSet(availableBones);
  const { contract, poseBones, handShapeBones, motionBones, coreBones } = requirements;
  const pose = _coverage(poseBones, available);
  const hands = _coverage(handShapeBones, available);
  const motion = _coverage(motionBones, available);
  const core = _coverage(coreBones, available);

  const handWeight = handShapeBones.length
    ? (contract.handRequirement === 'required' ? 0.28 : contract.handRequirement === 'important' ? 0.2 : 0.12)
    : 0;
  const motionWeight = motionBones.length ? 0.08 : 0;
  const poseWeight = Math.max(0, 1 - handWeight - motionWeight);
  const score = (pose.coverage * poseWeight) + (hands.coverage * handWeight) + (motion.coverage * motionWeight);
  const handOk = contract.handRequirement === 'required'
    ? hands.coverage >= Math.max(0.01, contract.minHandCoverage)
    : hands.coverage >= (contract.minHandCoverage || 0);
  const canPlay = core.missing.length === 0
    && pose.coverage >= contract.minPoseCoverage
    && handOk
    && score >= contract.minScore;
  const canAuto = canPlay && !!contract.autoSafe && score >= contract.minAutoScore;

  let status = 'ready';
  if (!canPlay) status = contract.fallback ? 'fallback' : 'blocked';
  else if (score < 0.96 || pose.missing.length || hands.missing.length || motion.missing.length) status = 'degraded';

  const warnings = [];
  if (core.missing.length) warnings.push(`Missing core bones: ${core.missing.join(', ')}`);
  if (pose.missing.length) warnings.push(`Missing action bones: ${pose.missing.join(', ')}`);
  if (hands.missing.length) warnings.push(`Missing hand bones: ${hands.missing.length}`);
  if (motion.missing.length) warnings.push(`Missing motion texture bones: ${motion.missing.join(', ')}`);
  if (contract.frameRisk !== 'low') warnings.push(`Frame risk: ${contract.frameRisk}`);
  if (options.phase && !contract.phases.includes(options.phase)) {
    warnings.push(`Not intended for ${options.phase} phase`);
  }

  return {
    name,
    label: action.label,
    status,
    canPlay,
    canAuto,
    score,
    coverage: score,
    poseCoverage: pose.coverage,
    handCoverage: hands.coverage,
    motionCoverage: motion.coverage,
    coreCoverage: core.coverage,
    missing: [...new Set([...core.missing, ...pose.missing, ...hands.missing, ...motion.missing])],
    missingCore: core.missing,
    missingPose: pose.missing,
    missingHands: hands.missing,
    missingMotion: motion.missing,
    warnings,
    fallback: canPlay ? null : contract.fallback,
    contract,
  };
}

export function resolveCallActionForRig(name, availableBones, options = {}) {
  const visited = [];
  let currentName = name;
  for (let i = 0; i < 5; i += 1) {
    if (!currentName || visited.includes(currentName)) break;
    visited.push(currentName);
    const compatibility = assessCallActionCompatibility(currentName, availableBones, options);
    if (compatibility.canPlay) {
      return {
        requestedName: name,
        name: currentName,
        action: getCallAction(currentName),
        compatibility,
        fallbackChain: visited,
        fallbackUsed: currentName !== name,
      };
    }
    currentName = compatibility.fallback;
  }

  return {
    requestedName: name,
    name: null,
    action: null,
    compatibility: visited.length
      ? assessCallActionCompatibility(visited[visited.length - 1], availableBones, options)
      : assessCallActionCompatibility(name, availableBones, options),
    fallbackChain: visited,
    fallbackUsed: false,
  };
}

export class CallActionScheduler {
  constructor(options = {}) {
    this._now = options.now || _nowSeconds;
    this._getCompatibility = options.getCompatibility || null;
    this.current = null;
    this.cooldowns = new Map();
    this.lastResult = null;
  }

  request(name, options = {}) {
    const now = options.now ?? this._now();
    this.update(now);

    const compatibility = this._getCompatibility?.(name, options);
    const resolved = compatibility?.canPlay
      ? { requestedName: name, name, action: getCallAction(name), compatibility, fallbackChain: [name], fallbackUsed: false }
      : resolveCallActionForRig(name, options.availableBones || null, options);
    if (!resolved.action) {
      return this._remember({ accepted: false, reason: 'no-compatible-action', requestedName: name, resolved });
    }

    const contract = getCallActionContract(resolved.name);
    const mode = options.mode || 'manual';
    if (mode === 'auto' && !resolved.compatibility.canAuto) {
      return this._remember({ accepted: false, reason: 'not-auto-safe', requestedName: name, resolved });
    }

    const cooldownKey = contract.cooldownGroup || resolved.name;
    const cooldownUntil = this.cooldowns.get(cooldownKey) || 0;
    if (!options.ignoreCooldown && now < cooldownUntil) {
      return this._remember({
        accepted: false,
        reason: 'cooldown',
        requestedName: name,
        resolved,
        remaining: cooldownUntil - now,
      });
    }

    if (this.current && !this._canInterrupt(this.current, contract, now)) {
      return this._remember({ accepted: false, reason: 'busy', requestedName: name, resolved, current: this.current });
    }

    this.current = {
      name: resolved.name,
      requestedName: name,
      startedAt: now,
      duration: resolved.action.duration || 1,
      contract,
      compatibility: resolved.compatibility,
    };
    return this._remember({ accepted: true, reason: 'accepted', requestedName: name, ...resolved, current: this.current });
  }

  update(now = this._now()) {
    if (!this.current) return null;
    const elapsed = now - this.current.startedAt;
    if (elapsed >= this.current.duration) {
      this.finish(now);
      return null;
    }
    return this.current;
  }

  finish(now = this._now()) {
    if (!this.current) return;
    const key = this.current.contract.cooldownGroup || this.current.name;
    this.cooldowns.set(key, now + (this.current.contract.cooldown || 0));
    this.current = null;
  }

  clear() {
    this.current = null;
  }

  _canInterrupt(current, incomingContract, now) {
    const progress = Math.max(0, Math.min(1, (now - current.startedAt) / Math.max(current.duration, 0.001)));
    if (progress >= (current.contract.minProgressBeforeInterrupt ?? 0.68)) return true;
    if (incomingContract.interrupt === 'force') return true;
    return (incomingContract.priority || 0) >= ((current.contract.priority || 0) + 24);
  }

  _remember(result) {
    this.lastResult = result;
    return result;
  }
}

export function validateCallActionLibrary(actions = CALL_ACTIONS) {
  const issues = [];
  for (const [name, action] of Object.entries(actions)) {
    const contract = getCallActionContract(name);
    if (!contract) issues.push(`${name}: missing action contract`);
    if (!Number.isFinite(action.duration) || action.duration <= 0) issues.push(`${name}: invalid duration`);
    if (!Array.isArray(action.keyframes) || action.keyframes.length < 2) issues.push(`${name}: needs at least two keyframes`);
    if (action.keyframes?.[0]?.t !== 0) issues.push(`${name}: first keyframe must start at 0`);
    if (action.keyframes?.[action.keyframes.length - 1]?.t !== 1) issues.push(`${name}: final keyframe must end at 1`);
    for (let i = 1; i < (action.keyframes || []).length; i += 1) {
      if (action.keyframes[i].t < action.keyframes[i - 1].t) issues.push(`${name}: keyframes are not sorted`);
    }
    for (const shapeName of Object.values(action.handShapes || {})) {
      if (!HAND_SHAPES[shapeName]) issues.push(`${name}: unknown hand shape ${shapeName}`);
    }
    if (contract?.fallback && !actions[contract.fallback]) issues.push(`${name}: unknown fallback ${contract.fallback}`);
  }
  return { ok: issues.length === 0, issues };
}

function _toAvailableBoneSet(availableBones) {
  if (!availableBones) return null;
  if (availableBones instanceof Set) return availableBones;
  if (Array.isArray(availableBones)) return new Set(availableBones);
  if (availableBones.bones && typeof availableBones.bones === 'object') {
    return new Set(Object.keys(availableBones.bones).filter((name) => availableBones.bones[name]));
  }
  if (typeof availableBones === 'object') {
    return new Set(Object.keys(availableBones).filter((name) => availableBones[name]));
  }
  return null;
}

function _coverage(bones, available) {
  const unique = [...new Set(bones || [])].filter(Boolean);
  if (!available) return { coverage: 1, missing: [] };
  const missing = unique.filter((boneName) => !available.has(boneName));
  return {
    coverage: unique.length ? (unique.length - missing.length) / unique.length : 1,
    missing,
  };
}

function _nowSeconds() {
  if (typeof performance !== 'undefined' && performance.now) return performance.now() / 1000;
  return Date.now() / 1000;
}
