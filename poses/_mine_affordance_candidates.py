"""Mine candidate affordances from captured motion clips.

Body-relative classification is RECOMPUTED here from each frame's hand
world positions against the rich body-geometry capsule chain (hips/spine/
chest + 4 arm + 4 leg + head sphere). The brL/brR labels saved during
capture used an older minimal body-geometry where every capsule was
labeled `torso/lower`, so they're useless for clustering — we ignore them
and reclassify offline. One source of truth: the rich body geometry.

Pipeline:
  1. Reclassify each frame's handL/handR against the rich capsule chain
     (closest-surface-by-|distFromSurface|, same algorithm as the capture
     tool's classifyAgainstBody but with the better geometry).
  2. Walk frames per side, find STILL WINDOWS — runs of consecutive
     frames where the hand sits at the same capsule with negligible
     body-relative drift (Δh ≤ 0.08, Δ|dr| ≤ 1.5 cm). A still window =
     "hand anchored at this body location" = an affordance candidate.
  3. Bucket windows by (side, capsule), then greedy-cluster within each
     bucket by Frobenius distance on the 12-float arm-chain rotation
     signature (shoulder, upperArm, lowerArm, hand). Fingers vary inside
     a single affordance (open/closed during a hold) so they don't drive
     clustering — but the centroid frame's full chain (including fingers)
     is emitted so finger pose is preserved.
  4. Sum cluster votes (= total frame count across all windows). Drop
     tiny clusters.
  5. For each surviving cluster, compute distance to the closest existing
     affordance of the same side. Below DUPE_THRESH = duplicate of
     existing pose, just rediscovered. Above = novel.
  6. Write affordance-candidates.json sorted by votes, with each entry
     carrying full arm-chain bones (for instant playback via the
     AffordanceApplier), source clip+timecode, and vote breakdown.

Plain-glb clips are skipped this pass — their coordinate frame isn't
aligned to body-geometry. After we add a per-clip facing-axis alignment
step, plain-glb frames re-enter the same pipeline.
"""
import json, math, os, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

POSES = os.path.dirname(os.path.abspath(__file__))

# ─── Tunables ─────────────────────────────────────────────────────────
MIN_WINDOW_SECS = 0.4    # ≥ this many seconds of stillness to qualify
MAX_H_DRIFT     = 0.08   # capsule axial position drift cap across window
MAX_DR_DRIFT    = 0.015  # 1.5 cm radial drift cap (fingertip wobble OK)
CLUSTER_THRESH  = 0.50   # Frobenius dist on 12-float arm-rot vector
DUPE_THRESH     = 0.40   # below this = "same as existing affordance"
MIN_VOTES       = 6      # cluster must have ≥ this many frames total
DR_FAR_FILTER   = 0.30   # skip windows >30cm from body (mid-gesture)

# ─── Arm chains (humanoid normalized bones) ───────────────────────────
ARM_L = ['leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand']
ARM_R = ['rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand']

FULL_CHAIN_L = ARM_L + [
    'leftThumbProximal',  'leftThumbIntermediate',  'leftThumbDistal',
    'leftIndexProximal',  'leftIndexIntermediate',  'leftIndexDistal',
    'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
    'leftRingProximal',   'leftRingIntermediate',   'leftRingDistal',
    'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
]
FULL_CHAIN_R = ARM_R + [
    'rightThumbProximal',  'rightThumbIntermediate',  'rightThumbDistal',
    'rightIndexProximal',  'rightIndexIntermediate',  'rightIndexDistal',
    'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
    'rightRingProximal',   'rightRingIntermediate',   'rightRingDistal',
    'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
]

def arm_signature(bones, side):
    """Flattened 12-float vector of arm-chain Euler rotations."""
    chain = ARM_L if side == 'L' else ARM_R
    out = []
    for b in chain:
        rot = bones.get(b, [0.0, 0.0, 0.0])
        out.extend(rot)
    return out

def vec_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def classify_against_body(world_pos, body_geo):
    """Returns the closest capsule-or-sphere classification for `world_pos`,
    same algorithm as the JS classifyAgainstBody but here so we can ignore
    the captured brL/brR and use the rich body-geometry-vrm (1).json."""
    if not world_pos:
        return None
    best = None
    all_caps = list(body_geo.get('torsoCapsules', [])) + list(body_geo.get('limbCapsules', []))
    for c in all_caps:
        fr = c['from']
        to = c['to']
        seg = [to[i] - fr[i] for i in range(3)]
        seg_len = math.sqrt(sum(x*x for x in seg))
        if seg_len < 1e-9: continue
        seg_dir = [x / seg_len for x in seg]
        v_minus_fr = [world_pos[i] - fr[i] for i in range(3)]
        t = max(0.0, min(1.0, sum(v_minus_fr[i] * seg_dir[i] for i in range(3)) / seg_len))
        proj = [fr[i] + seg_dir[i] * t * seg_len for i in range(3)]
        radial = [world_pos[i] - proj[i] for i in range(3)]
        radial_dist = math.sqrt(sum(x*x for x in radial))
        d = radial_dist - c['radius']
        if best is None or abs(d) < abs(best['dr']):
            best = {
                'capsule': f'{c["region"]}/{c["segment"]}',
                'h': round(t, 3),
                'dr': round(d, 4),
                'dir': [round(radial[i] / radial_dist, 3) for i in range(3)] if radial_dist > 1e-6 else [0.0, 0.0, -1.0],
            }
    sphere = body_geo.get('headSphere')
    if sphere:
        cen = sphere['center']
        radial = [world_pos[i] - cen[i] for i in range(3)]
        r = math.sqrt(sum(x*x for x in radial))
        d = r - sphere['radius']
        if best is None or abs(d) < abs(best['dr']):
            best = {
                'capsule': 'head/sphere',
                'h': 0,
                'dr': round(d, 4),
                'dir': [round(radial[i] / r, 3) for i in range(3)] if r > 1e-6 else [0.0, 0.0, -1.0],
            }
    return best


def find_still_windows(frames, side, fps, body_geo):
    """Walk frames, return windows where the hand stays anchored to one
    capsule with ≤MAX_H_DRIFT axial / ≤MAX_DR_DRIFT radial drift.
    Classification is recomputed per frame from the world hand position
    against the rich body-geometry capsule chain (the captured brL/brR
    used a stripped-down geometry where every capsule was 'torso/lower')."""
    min_frames = max(3, int(fps * MIN_WINDOW_SECS))
    hand_key = f'hand{side}'
    # Pre-compute per-frame classification so we don't re-classify on rewind
    classified = []
    for f in frames:
        pos = f.get('features', {}).get(hand_key)
        classified.append(classify_against_body(pos, body_geo))

    out = []
    n = len(frames)
    i = 0
    while i < n:
        b0 = classified[i]
        if not b0:
            i += 1; continue
        cap = b0['capsule']
        h_min = h_max = b0['h']
        dr_min = dr_max = b0['dr']
        j = i
        while j + 1 < n:
            nb = classified[j + 1]
            if not nb or nb['capsule'] != cap: break
            nh, ndr = nb['h'], nb['dr']
            new_h_min, new_h_max = min(h_min, nh), max(h_max, nh)
            new_dr_min, new_dr_max = min(dr_min, ndr), max(dr_max, ndr)
            if (new_h_max - new_h_min) > MAX_H_DRIFT: break
            if (new_dr_max - new_dr_min) > MAX_DR_DRIFT: break
            h_min, h_max = new_h_min, new_h_max
            dr_min, dr_max = new_dr_min, new_dr_max
            j += 1
        length = j - i + 1
        if length >= min_frames:
            mid = (i + j) // 2
            avg_dr = (dr_min + dr_max) / 2
            if abs(avg_dr) <= DR_FAR_FILTER:
                out.append({
                    'side': side,
                    'mid': mid,
                    'start': i, 'end': j, 'length': length,
                    'capsule': cap,
                    'h': round((h_min + h_max) / 2, 3),
                    'dr': round(avg_dr, 4),
                    'dir': classified[mid]['dir'],
                })
        i = j + 1 if j > i else i + 1
    return out


def main():
    with open(os.path.join(POSES, 'motion-clips-all.json'), 'r', encoding='utf-8') as f:
        motion = json.load(f)
    with open(os.path.join(POSES, 'affordances-bones.json'), 'r', encoding='utf-8') as f:
        existing = json.load(f)
    # Use the rich geometry — torso (3) + limb (10) + head sphere — not the
    # stripped-down one with bone-only torso capsules.
    with open(os.path.join(POSES, 'body-geometry-vrm (1).json'), 'r', encoding='utf-8') as f:
        body_geo = json.load(f)
    existing_aff = existing.get('affordances', {})
    cap_count = len(body_geo.get('torsoCapsules', [])) + len(body_geo.get('limbCapsules', []))

    fps = motion['sampleFps']
    clips = motion['clips']
    print(f"Loaded {len(clips)} clips @ {fps} fps; {cap_count} body capsules + head sphere; existing affordances: {len(existing_aff)}")

    # Stage 1: collect still windows from each vrma-1x clip (reclassifying)
    all_wins = []
    skipped_clips = 0
    for name, clip in clips.items():
        if clip.get('coordFrame') == 'source-scene-native':
            skipped_clips += 1
            continue
        for side in ('L', 'R'):
            for w in find_still_windows(clip['frames'], side, fps, body_geo):
                w['clip_name'] = name
                w['t'] = clip['frames'][w['mid']]['t']
                w['bones'] = clip['frames'][w['mid']]['bones']
                w['features'] = clip['frames'][w['mid']]['features']
                w['arm_sig'] = arm_signature(w['bones'], side)
                all_wins.append(w)
    print(f"Stage 1: {len(all_wins)} still windows  ({skipped_clips} plain-glb clips skipped)")

    # Stage 2: greedy cluster within (side, capsule)
    by_side_cap = defaultdict(list)
    for w in all_wins:
        by_side_cap[(w['side'], w['capsule'])].append(w)

    clusters = []
    for (side, cap), wins in sorted(by_side_cap.items()):
        remaining = list(wins)
        while remaining:
            seed = max(remaining, key=lambda w: w['length'])
            members = [w for w in remaining if vec_dist(w['arm_sig'], seed['arm_sig']) < CLUSTER_THRESH]
            if len(members) > 1:
                centroid = min(members, key=lambda w:
                    sum(vec_dist(w['arm_sig'], m['arm_sig']) for m in members))
            else:
                centroid = members[0]
            clusters.append({
                'side': side, 'capsule': cap,
                'members': members, 'centroid': centroid,
                'votes': sum(m['length'] for m in members),
            })
            remaining = [w for w in remaining if w not in members]
    print(f"Stage 2: {len(clusters)} raw clusters")

    # Stage 3: drop low-vote clusters
    clusters = [c for c in clusters if c['votes'] >= MIN_VOTES]
    print(f"  → {len(clusters)} after MIN_VOTES≥{MIN_VOTES} filter")

    # Stage 4: similarity vs existing affordances
    novel = 0
    for c in clusters:
        side = c['side']
        sig = c['centroid']['arm_sig']
        best_name, best_dist = None, float('inf')
        for ex_name, ex_bones in existing_aff.items():
            if not ex_name.endswith(f'_{side}'):
                continue
            ex_sig = arm_signature(ex_bones, side)
            d = vec_dist(sig, ex_sig)
            if d < best_dist:
                best_dist = d; best_name = ex_name
        c['similarTo']  = best_name if best_dist < DUPE_THRESH else None
        c['similarity'] = round(best_dist, 3) if best_name else None
        c['nearest']    = best_name  # always reported
        if not c['similarTo']:
            novel += 1
    print(f"  → {novel} novel (not within {DUPE_THRESH} of existing)")

    # Stage 5: emit
    clusters.sort(key=lambda c: -c['votes'])
    candidates = []
    for rank, c in enumerate(clusters, 1):
        cent = c['centroid']
        cap_short = c['capsule'].replace('/', '_')
        h_zone = ('low' if cent['h'] < 0.34 else
                  'mid' if cent['h'] < 0.67 else 'high')
        chain = FULL_CHAIN_L if c['side'] == 'L' else FULL_CHAIN_R
        candidates.append({
            'rank': rank,
            'side': c['side'],
            'votes': c['votes'],
            'memberCount': len(c['members']),
            'capsule': c['capsule'],
            'h': cent['h'],
            'dr': cent['dr'],
            'dir': cent.get('dir'),
            'nameHint': f'{cap_short}_{h_zone}_{c["side"]}',
            'source': {'clip': cent['clip_name'], 't': cent['t']},
            'nearestExisting': c['nearest'],
            'similarTo': c['similarTo'],
            'similarity': c['similarity'],
            'bones': {b: cent['bones'][b] for b in chain if b in cent['bones']},
            'memberSources': [
                {'clip': m['clip_name'], 't': m['t'], 'length': m['length']}
                for m in sorted(c['members'], key=lambda m: -m['length'])[:5]
            ],
        })

    out = {
        'schema': 'augmentum.affordance-candidates.v1',
        'generatedFromMotionClips': 'motion-clips-all.json',
        'referenceVRM': motion.get('referenceVRM'),
        'totalCandidates': len(candidates),
        'novelCandidates': novel,
        'tunables': {
            'MIN_WINDOW_SECS': MIN_WINDOW_SECS,
            'MAX_H_DRIFT': MAX_H_DRIFT,
            'MAX_DR_DRIFT': MAX_DR_DRIFT,
            'CLUSTER_THRESH': CLUSTER_THRESH,
            'DUPE_THRESH': DUPE_THRESH,
            'MIN_VOTES': MIN_VOTES,
            'DR_FAR_FILTER': DR_FAR_FILTER,
        },
        'candidates': candidates,
    }
    out_path = os.path.join(POSES, 'affordance-candidates.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(candidates)} candidates ({os.path.getsize(out_path)/1024:.1f} KB) to {out_path}")

    # Summary table
    print(f"\n{'#':<4} {'side':<4} {'votes':<6} {'capsule':<22} {'h':<5} {'dr':<8} {'novel?':<7} {'nearest':<22} {'src clip'}")
    print('-' * 120)
    for c in candidates[:40]:
        novel_flag = '★ NEW' if not c['similarTo'] else f'dup {c["similarity"]}'
        nearest = c['nearestExisting'] or '—'
        clip = c['source']['clip'][:30]
        print(f"{c['rank']:<4} {c['side']:<4} {c['votes']:<6} {c['capsule']:<22} {c['h']:<5} {c['dr']:<8} {novel_flag:<7} {nearest:<22} {clip}")
    if len(candidates) > 40:
        print(f"  … and {len(candidates) - 40} more")


if __name__ == '__main__':
    main()
