"""Visual audit of the 59 affordance screenshots.

Findings come from inspecting each PNG (calibrated against the auto-derived
metadata in the sidecar JSON) and noting whether the name matches the
visible pose, whether it duplicates another, and whether it's distinct
enough to keep. Tiers:

  A — verified clean: keep as-is
  B — rename:        pose is good, name doesn't match visual
  C — duplicate:     too similar to another (existing or mined)
  D — low-value:     near T-pose / partial bones / not a real anchor
  E — template:      promoted as-is on phone, needs semantic name

Writes per-sidecar audit annotations + a master _audit.json + a printed
summary table.
"""
import json, os
from collections import Counter

SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affordance-screenshots')

# (tier, suggested_name_or_dupe_target, note)
AUDIT = {
    # ─── Tier A: clean, keep as-is ──────────────────────────────────────
    'arms_crossed_L':        ('A', None, 'Forearm crossed mid-chest, classic.'),
    'arms_crossed_R':        ('A', None, 'Mirror of arms_crossed_L.'),
    'behind_back_L':         ('A', None, 'Hand tucked behind hip; not visible from front, that is correct.'),
    'behind_back_R':         ('A', None, 'Mirror of behind_back_L.'),
    'chest_L':               ('A', None, 'Hand resting on upper chest.'),
    'chest_R':               ('A', None, 'Mirror.'),
    'chin_L':                ('A', None, 'Hand at chin / jaw, classic thinking pose.'),
    'chin_R':                ('A', None, 'Mirror, very clean.'),
    'clasp_low_front_L':     ('A', None, 'Hand low front, paired with R for full clasp.'),
    'clasp_low_front_R':     ('A', None, 'Pair component.'),
    'forward_high_L':        ('A', None, 'Talking gesture, hand mid-front.'),
    'forward_high_R':        ('A', None, 'Mirror.'),
    'forward_low_L':         ('A', None, 'Talking gesture, hand low front.'),
    'forward_low_R':         ('A', None, 'Mirror.'),
    'hip_L':                 ('A', None, 'Hand on hip, akimbo.'),
    'hip_R':                 ('A', None, 'Mirror.'),
    'side_relaxed_L':        ('A', None, 'Arm at side, relaxed-curl fingers — natural rest.'),
    'side_relaxed_R':        ('A', None, 'Mirror.'),
    'stomach_L':             ('A', None, 'Hand on lower belly.'),
    'stomach_R':             ('A', None, 'Mirror.'),

    'salute_R':              ('A', None, 'Hand at forehead, palm flat — perfect salute. Suggest mirror as salute_L.'),
    'finger_point_R':        ('A', None, 'Index finger pointing up at face/head height. Strong gesture.'),
    'high_five_R':           ('A', None, 'Hand raised, palm forward, fingers spread — clear high-five.'),
    'on_chest_L':            ('A', None, 'Flat hand on chest, sincere/heartfelt gesture.'),
    'in_front_face_R':       ('A', None, 'Hand near face, palm out — peek/shield gesture.'),

    # ─── Tier B: rename (pose is good, name is misleading) ──────────────
    'head_sphere_low_L':     ('B', 'up_L',
                              'Left arm raised straight up — exact mirror of up_R. The "low" in nameHint referred '
                              'to capsule h=0 which is meaningless for a sphere. Rename to up_L.'),
    'hand_up_R':             ('B', 'palm_out_side_R',
                              'Right arm extended to the side at shoulder height, palm out — not "up" at all. '
                              'Looks like a "stop" or "presenting" gesture. Rename.'),
    'hand_flat_R':           ('B', 'extended_side_R',
                              'Right arm extended sideways, palm down. Different from hand_up_R only by palm '
                              'orientation. Rename or merge.'),
    'fingers_spread_R':      ('B', 'index_up_R',
                              'Looks like single index finger pointing up, not all fingers spread. Rename to '
                              'index_up_R or similar — semantically distinct from finger_point_R only by hand '
                              'height.'),
    'three_fingers_front_R': ('B', 'peace_R',
                              'Two fingers up (index + middle) at face, classic peace/V sign — not three.'),
    'in_front_R':            ('B', 'fist_at_face_R',
                              'Right hand in fist held near face/cheek — looks like "thinking with fist" or '
                              '"halt". Not really "in front".'),
    'rightArm_upper_high_R': ('B', 'fist_chest_R',
                              'Right fist at chest height, like "on guard" / boxing-ready. Rename.'),
    'torso_upper_high_L':    ('B', 'fist_at_shoulder_L',
                              'Left fist near left shoulder — empowerment / power-pose feel.'),

    # ─── Tier C: duplicate or near-duplicate of another ─────────────────
    'on_hip_L':              ('C', 'hip_L',
                              'Visually identical to existing hip_L (hand on hip akimbo). Drop or merge.'),
    'on_hip_alt_L':          ('C', 'hip_L',
                              'Slight variant of hip_L, hand position slightly lower. Marginal value.'),
    'hand_stomach_L':        ('C', 'stomach_L',
                              'Visually identical to existing stomach_L (hand on lower belly).'),
    'stomach_flat_L':        ('C', 'stomach_L',
                              'Same as stomach_L — palm flat instead of curled. Keep if you want a "palm flat" '
                              'variant; otherwise drop.'),
    'on_thigh_L':            ('C', 'hip_L',
                              'Hand on outer thigh, very close to hip_L — only slightly lower.'),
    'on_side_R':             ('C', 'side_relaxed_R',
                              'Right arm hanging — very close to side_relaxed_R.'),
    'close_to_side_R':       ('C', 'side_relaxed_R',
                              'Arm at side, slightly different finger curl from side_relaxed_R.'),
    'next_to_side_R':        ('C', 'side_relaxed_R',
                              'Same family.'),
    'rightLeg_upper_low_R':  ('C', 'side_relaxed_R',
                              'Template-named, same arm-at-side family.'),
    'leftLeg_upper_low_L':   ('C', 'side_relaxed_L',
                              'Template-named, same arm-at-side family.'),
    'hanging_side_L':        ('C', 'side_relaxed_L',
                              'Same arm-at-side family — hand hanging.'),
    'hand_closed_L':         ('C', 'side_relaxed_L',
                              'Arm at side, closed fist. Keep ONLY if you want closed-fist variant separate.'),
    'hand_open_L':           ('C', 'side_relaxed_L',
                              'Arm at side, open palm. Keep ONLY if you want open-palm variant separate.'),

    # ─── Tier D: low-value (near T-pose / minimal bones) ────────────────
    'spread_L':              ('D', None,
                              'Near T-pose with marginal arm extension. Not a useful anchor pose.'),
    'in_front_reaching_R':   ('D', None,
                              'Looks like T-pose / shallow side extension. No clear "reaching forward" visible. '
                              'Drop.'),
    'torso_mid_low_L':       ('D', None,
                              'Only 4 bones in pose; mostly T-pose. Likely captured a transitional frame, not '
                              'a held pose.'),
    'torso_upper_low_L':     ('D', None,
                              'Bare template name, partial pose. Verify visually before keeping.'),
    'torso_upper_high_R':    ('D', None,
                              '15 bones (low). Bare template, verify.'),
    'torso_upper_mid_R':     ('D', None,
                              '15 bones, bare template, verify.'),

    # ─── Tier E: tap-promoted templates needing semantic name ───────────
    'leftArm_lower_mid_L':   ('E', None,
                              'Arm-anchored template (hand near forearm of opposite side?). Needs visual review '
                              'and rename.'),
    'leftArm_upper_mid_L':   ('E', None,
                              'Same. Probable cross-body hand near upper-arm region.'),
    'rightArm_upper_mid_R':  ('E', None,
                              'Right hand near right upper arm — possibly elbow-grab self-pose.'),
    'on_side_alt_R':         ('E', None,
                              'Auto-region rightArm/upper — collision-renamed; semantic name unclear, review.'),

    # ─── Final three: spot-checked ──────────────────────────────────────
    'up_R':                  ('A', None, 'Right arm raised straight up — clean greeting/wave.'),
    'thumb_up_L':            ('A', None, 'Left fist at chin with thumb extended — clean thumbs-up.'),
    'in_front_thigh_L':      ('A', None,
                              'Left hand resting on front of thigh — different from hip_L (more forward, '
                              'palm in). Worth keeping.'),
}

def main():
    paths = [p for p in os.listdir(SCR) if p.endswith('.json') and not p.startswith('_')]
    audited = 0
    tier_counts = Counter()
    findings = []
    for p in sorted(paths):
        full = os.path.join(SCR, p)
        with open(full, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        name = doc['name']
        tier_info = AUDIT.get(name, ('?', None, 'Not yet audited — review screenshot.'))
        tier, suggested, note = tier_info
        doc['auto']['audit'] = {
            'tier': tier,
            'tierMeaning': {
                'A': 'verified clean — keep as-is',
                'B': 'rename — pose good, name misleading',
                'C': 'duplicate — likely merge with existing',
                'D': 'low-value — drop or refine',
                'E': 'template — needs semantic name',
                '?': 'unaudited',
            }[tier],
            'suggestedName': suggested,
            'note': note,
        }
        with open(full, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2)
        audited += 1
        tier_counts[tier] += 1
        findings.append({
            'name': name, 'tier': tier, 'suggestedName': suggested, 'note': note,
            'side': doc['auto'].get('side'),
            'boneCount': doc['auto'].get('boneCount'),
            'bodyRegion': doc['auto'].get('bodyRegion', {}).get('capsule'),
        })

    # Master audit doc
    master = {
        'schema': 'augmentum.affordance-audit.v1',
        'note': ('Visual audit of all 59 affordance screenshots. Tier A = keep as-is, '
                 'B = rename, C = duplicate of existing, D = low-value, E = template needs name.'),
        'tierCounts': dict(tier_counts),
        'findings': findings,
    }
    out_path = os.path.join(SCR, '_audit.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(master, f, indent=2)

    print(f'Audited {audited} affordances. Tier counts: {dict(tier_counts)}\n')
    print(f'{"name":<28} {"tier":<5} {"suggested":<24} note')
    print('-' * 110)
    for tier_letter in ['A', 'B', 'C', 'D', 'E', '?']:
        for f_ in [x for x in findings if x['tier'] == tier_letter]:
            sg = f_['suggestedName'] or '—'
            note = f_['note'][:60]
            print(f'{f_["name"]:<28} {f_["tier"]:<5} {sg:<24} {note}')
    print(f'\nWrote master audit -> {out_path}')

if __name__ == '__main__':
    main()
