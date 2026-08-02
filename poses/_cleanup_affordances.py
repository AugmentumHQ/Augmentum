"""Apply the visual-audit cleanup to affordances-bones.json.

  - 8 renames + 1 from Tier E (pose good, name was misleading)
  - Drop 8 low-value (T-pose / partial) entries
  - Drop 12 duplicates (cluster around side_relaxed / hip / stomach already
    in canonical set); keep 2 useful side-arm variants (open vs closed
    palm) for stylistic options.

Backs up the previous file first so this is reversible.
"""
import json, os, shutil, sys
sys.stdout.reconfigure(encoding='utf-8')

POSES = os.path.dirname(os.path.abspath(__file__))
CANON = f'{POSES}/affordances-bones.json'
BACKUP = f'{POSES}/affordances-bones.json.before-cleanup'

# (current_name, new_name, reason)
RENAMES = [
    ('head_sphere_low_L',     'up_L',                'Mirror of up_R; "low" was a capsule param, not pose semantics'),
    ('hand_up_R',             'palm_out_side_R',     'Arm extended sideways at shoulder, palm out — not "up"'),
    ('hand_flat_R',           'extended_side_R',     'Sideways extension, palm down'),
    ('fingers_spread_R',      'index_up_R',          'Single index finger up, not all spread'),
    ('three_fingers_front_R', 'peace_R',             'Two-finger peace/V sign, not three'),
    ('in_front_R',            'fist_at_face_R',      'Closed fist near cheek'),
    ('rightArm_upper_high_R', 'fist_chest_R',        'On-guard fist at chest'),
    ('torso_upper_high_L',    'fist_at_shoulder_L',  'Power-pose fist at shoulder'),
    ('rightArm_upper_mid_R',  'pinch_at_face_R',     'Right hand near face with delicate pinch grip'),
]

# Tier-D drops (T-pose / partial / no clear anchor)
DROP_LOW_VALUE = [
    'in_front_reaching_R',
    'spread_L',
    'torso_mid_low_L',
    'torso_upper_high_R',
    'torso_upper_low_L',
    'torso_upper_mid_R',
    'leftArm_lower_mid_L',  # T-pose, demoted from Tier E
    'on_side_alt_R',        # near T-pose, demoted from Tier E
]

# Tier-C drops (visually clusters into existing canonical)
# Keeping hand_closed_L (closed-fist side variant) + hand_open_L (open-palm
# side variant) for stylistic options vs the curled-fingers side_relaxed_L.
DROP_DUPLICATES = [
    'hanging_side_L',         # → side_relaxed_L
    'leftLeg_upper_low_L',    # → side_relaxed_L
    'close_to_side_R',        # → side_relaxed_R
    'next_to_side_R',         # → side_relaxed_R
    'on_side_R',              # → side_relaxed_R
    'rightLeg_upper_low_R',   # → side_relaxed_R
    'hand_stomach_L',         # → stomach_L
    'stomach_flat_L',         # → stomach_L
    'on_hip_L',               # → hip_L
    'on_hip_alt_L',           # → hip_L
    'on_thigh_L',             # → hip_L
    'leftArm_upper_mid_L',    # → fist_at_shoulder_L (renamed torso_upper_high_L)
]

def main():
    apply = '--apply' in sys.argv
    with open(CANON, 'r', encoding='utf-8') as f:
        canon = json.load(f)
    aff = canon['affordances']
    before_count = len(aff)

    new_aff = dict(aff)

    # 1. Renames
    rename_log = []
    for old, new, reason in RENAMES:
        if old not in new_aff:
            rename_log.append(('MISSING', old, new, reason))
            continue
        if new in new_aff:
            rename_log.append(('COLLISION', old, new, f'Target already exists — skipping. {reason}'))
            continue
        new_aff[new] = new_aff.pop(old)
        rename_log.append(('OK', old, new, reason))

    # 2. Drops
    drop_log = []
    for n in DROP_LOW_VALUE:
        if n in new_aff:
            del new_aff[n]; drop_log.append(('low-value', n))
        else:
            drop_log.append(('missing', n))
    for n in DROP_DUPLICATES:
        if n in new_aff:
            del new_aff[n]; drop_log.append(('duplicate', n))
        else:
            drop_log.append(('missing', n))

    # Print plan
    print(f'Before: {before_count} affordances\n')
    print(f'=== Renames ({len([r for r in rename_log if r[0] == "OK"])}/{len(RENAMES)}) ===')
    for status, old, new, reason in rename_log:
        print(f'  [{status:<9}] {old:<24} -> {new:<22} ({reason})')
    print(f'\n=== Drops ({len([d for d in drop_log if d[0] != "missing"])}/{len(DROP_LOW_VALUE)+len(DROP_DUPLICATES)}) ===')
    for kind, n in drop_log:
        if kind == 'missing':
            print(f'  [MISSING ] {n}')
        else:
            print(f'  [{kind:<9}] {n}')

    after_count = len(new_aff)
    delta = before_count - after_count
    print(f'\nAfter: {after_count} affordances  (-{delta})')

    if not apply:
        print('\nDry run. Pass --apply to write changes.')
        return

    # Backup, then write
    shutil.copy(CANON, BACKUP)
    print(f'\n* Backed up canonical -> {BACKUP}')
    canon['affordances'] = new_aff
    canon['note'] = (canon.get('note', '') +
        f' --- Cleanup pass on 2026-05-03: {len([r for r in rename_log if r[0] == "OK"])} renames, '
        f'{len([d for d in drop_log if d[0] != "missing"])} drops based on visual screenshot audit '
        f'(see _audit.json in poses/affordance-screenshots/).')
    with open(CANON, 'w', encoding='utf-8') as f:
        json.dump(canon, f, indent=2)
    print(f'* Wrote cleaned canonical -> {CANON}')

    # List final names sorted
    print(f'\n=== Final {after_count} affordances ===')
    for n in sorted(new_aff.keys()):
        print(f'  {n}')

if __name__ == '__main__':
    main()
