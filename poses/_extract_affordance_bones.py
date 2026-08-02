"""Extract per-affordance arm-chain bone rotations from authored pose JSONs.

For each affordance, we look up the SOURCE POSE that authored the right-hand
or left-hand position. We then read that pose's arm-chain Euler rotations
(shoulder, upperArm, lowerArm, hand, plus all 14 finger joints per side)
and write them into affordances-bones.json keyed by affordance name.

When the source pose only authored one hand at the target affordance position
(e.g. R hand on chin → only R arm chain authored), we MIRROR across the
saggital plane to produce the opposite hand's arm chain. Mirror is the
standard `[x, -y, -z]` Euler transform plus L↔R bone-name swap.

Output is consumed by ui/scripts/avatar-affordance-applier.js: when the user
calls setHandPose('L', 'chin_L'), the module copies the affordance's L-arm
quaternions onto the live VRM bones. No IK math runs. The avatar lands in
exactly the configuration the source pose was authored to produce, and the
affordance position emerges from the bone hierarchy + skeleton proportions.
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

POSES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affordances-bones.json')

# Per affordance: which source pose authored the position, and which hand
# in that source pose. Target hand comes from the _L/_R suffix; mirror is
# applied automatically when source_hand != target_hand.
SOURCE_MAP = {
    # Hand on chin — both _L and _R use the right-hand-on-chin source.
    # _L mirrors to land L hand at the mirrored chin position.
    'chin_L':  ('thinking-righthandonchin-lefthandonwaistpose-2026-05-03T02-39-12', 'R'),
    'chin_R':  ('thinking-righthandonchin-lefthandonwaistpose-2026-05-03T02-39-12', 'R'),

    # Hand at chest (under-chest cross-body source) — L hand authored.
    'chest_L': ('thinking-righthandonchin-lefthandunderchest-pose-2026-05-03T02-51-14', 'L'),
    'chest_R': ('thinking-righthandonchin-lefthandunderchest-pose-2026-05-03T02-51-14', 'L'),

    # Crossed arms — bilateral, both hands authored independently.
    'arms_crossed_L': ('Arms-crossed-pose', 'L'),
    'arms_crossed_R': ('Arms-crossed-pose', 'R'),

    # Hands behind back — bilateral.
    'behind_back_L': ('handsbehindback-pose-2026-05-03T01-50-27', 'L'),
    'behind_back_R': ('handsbehindback-pose-2026-05-03T01-50-27', 'R'),

    # Clasped low front — bilateral.
    'clasp_low_front_L': ('handclaspedinfront-pose-2026-05-01T04-47-40', 'L'),
    'clasp_low_front_R': ('handclaspedinfront-pose-2026-05-01T04-47-40', 'R'),

    # Hand on hip — separate L and R source poses.
    'hip_L': ('idle-turnedlefthandonhip-righthandonsidepose-2026-05-03T03-31-45', 'L'),
    'hip_R': ('Rhandonhip-pose-2026-04-30T22-48-18', 'R'),

    # Hand on stomach — only L authored, mirror for R.
    'stomach_L': ('idle-lefthandonstomach-righthandonside-pose-2026-05-03T03-20-40', 'L'),
    'stomach_R': ('idle-lefthandonstomach-righthandonside-pose-2026-05-03T03-20-40', 'L'),

    # Forward-high talking gesture — R authored.
    'forward_high_L': ('righthandinfront-talking-pose-2026-05-03T04-09-28', 'R'),
    'forward_high_R': ('righthandinfront-talking-pose-2026-05-03T04-09-28', 'R'),

    # Forward-low talking gesture — R authored.
    'forward_low_L': ('righthandinfrontdown-talking-pose-2026-05-03T04-09-28', 'R'),
    'forward_low_R': ('righthandinfrontdown-talking-pose-2026-05-03T04-09-28', 'R'),
}

# side_relaxed has no authored pose source — it's a synthesized "natural
# arm hanging at side" using values from the natural preset in
# avatar-pose-presets.js. Stored as bone rotations directly.
HAND_CODED_BONES = {
    'side_relaxed_L': {
        'leftShoulder':  [0, 0, 0],
        'leftUpperArm':  [0, 0.04, -1.35],
        'leftLowerArm':  [0.10, 0, -0.05],
        'leftHand':      [0, 0, 0],
        # Relaxed-curl fingers (L)
        'leftThumbProximal':      [0.1,  -0.3, -0.2],
        'leftThumbDistal':        [0,     0,   -0.2],
        'leftIndexProximal':      [0,     0,   -0.3],
        'leftIndexIntermediate':  [0,     0,   -0.4],
        'leftIndexDistal':        [0,     0,   -0.3],
        'leftMiddleProximal':     [0,     0,   -0.35],
        'leftMiddleIntermediate': [0,     0,   -0.45],
        'leftMiddleDistal':       [0,     0,   -0.35],
        'leftRingProximal':       [0,     0,   -0.4],
        'leftRingIntermediate':   [0,     0,   -0.5],
        'leftRingDistal':         [0,     0,   -0.4],
        'leftLittleProximal':     [0,     0,   -0.45],
        'leftLittleIntermediate': [0,     0,   -0.55],
        'leftLittleDistal':       [0,     0,   -0.45],
    },
    'side_relaxed_R': {
        'rightShoulder': [0, 0, 0],
        'rightUpperArm': [0, -0.04, 1.35],
        'rightLowerArm': [0.10, 0, 0.05],
        'rightHand':     [0, 0, 0],
        # Relaxed-curl fingers (R)
        'rightThumbProximal':      [0.1,  0.3, 0.2],
        'rightThumbDistal':        [0,    0,   0.2],
        'rightIndexProximal':      [0,    0,   0.3],
        'rightIndexIntermediate':  [0,    0,   0.4],
        'rightIndexDistal':        [0,    0,   0.3],
        'rightMiddleProximal':     [0,    0,   0.35],
        'rightMiddleIntermediate': [0,    0,   0.45],
        'rightMiddleDistal':       [0,    0,   0.35],
        'rightRingProximal':       [0,    0,   0.4],
        'rightRingIntermediate':   [0,    0,   0.5],
        'rightRingDistal':         [0,    0,   0.4],
        'rightLittleProximal':     [0,    0,   0.45],
        'rightLittleIntermediate': [0,    0,   0.55],
        'rightLittleDistal':       [0,    0,   0.45],
    },
}

def arm_chain_bones(side):
    """Returns the 19 bone names making up an arm chain (4 arm + 15 finger)."""
    p = 'left' if side == 'L' else 'right'
    return [
        f'{p}Shoulder', f'{p}UpperArm', f'{p}LowerArm', f'{p}Hand',
        f'{p}ThumbProximal',  f'{p}ThumbIntermediate',  f'{p}ThumbDistal',
        f'{p}IndexProximal',  f'{p}IndexIntermediate',  f'{p}IndexDistal',
        f'{p}MiddleProximal', f'{p}MiddleIntermediate', f'{p}MiddleDistal',
        f'{p}RingProximal',   f'{p}RingIntermediate',   f'{p}RingDistal',
        f'{p}LittleProximal', f'{p}LittleIntermediate', f'{p}LittleDistal',
    ]

def mirror_bone_name(name):
    if name.startswith('left'):  return 'right' + name[4:]
    if name.startswith('right'): return 'left'  + name[5:]
    return name

def mirror_euler(rot):
    """Saggital plane mirror in normalized humanoid frame.
    For VRM bones, the body's vertical center plane is the YZ plane, so
    mirroring across X negates the components that encode left/right
    asymmetry. In Euler form for the rotation orders used by arm bones
    (mostly ZXY/YZX), the mirror is approximately [x, -y, -z]: X is the
    pitch (forward/back) which is symmetric, Y is yaw (which mirrors),
    Z is roll (which also mirrors)."""
    return [rot[0], -rot[1], -rot[2]]

def extract_for_side(pose, side):
    """Extract just the arm-chain bones for one side, dropping anything else."""
    chain = arm_chain_bones(side)
    return {bone: pose[bone] for bone in chain if bone in pose}

def mirror_chain(bones):
    """Map each L-* bone name → R-* (or vice versa) and apply Euler mirror."""
    out = {}
    for name, rot in bones.items():
        out[mirror_bone_name(name)] = mirror_euler(rot)
    return out

def main():
    output = {}
    missing = []
    for aff_name, (pose_basename, source_hand) in SOURCE_MAP.items():
        target_hand = 'L' if aff_name.endswith('_L') else 'R'
        pose_path = os.path.join(POSES_DIR, pose_basename + '.json')
        if not os.path.exists(pose_path):
            missing.append(pose_basename)
            continue
        with open(pose_path, 'r', encoding='utf-8') as f:
            pose = json.load(f)
        bones = extract_for_side(pose, source_hand)
        if target_hand != source_hand:
            bones = mirror_chain(bones)
        output[aff_name] = bones

    # Add hand-coded synthetic affordances
    for name, bones in HAND_CODED_BONES.items():
        output[name] = bones

    if missing:
        print(f'Missing pose files: {missing}')

    out = {
        'schema': 'augmentum.affordance-bones.v1',
        'note': ('Per-affordance arm-chain Euler rotations (radians) extracted '
                 'from authored pose JSONs. Each entry maps bone name → [x,y,z]. '
                 'Apply via avatar-affordance-applier.js: setHandPose(side, name) '
                 'copies these rotations onto the live VRM bones. Bones absent '
                 'from an entry are left as-is on the avatar.'),
        'mirrorPolicy': '[x, -y, -z] saggital mirror applied when source pose authored opposite hand',
        'affordances': output,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f'✓ {len(output)} affordances extracted')
    for name in sorted(output):
        bone_count = len(output[name])
        print(f'  {name:25}  {bone_count} bones')
    print(f'\nWrote {os.path.getsize(OUT_PATH)/1024:.1f} KB to {OUT_PATH}')

if __name__ == '__main__':
    main()
