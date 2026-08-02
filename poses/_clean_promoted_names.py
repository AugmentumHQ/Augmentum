"""Cleans up names in promoted-affordances.json before merging into
affordances-bones.json.

UX bug at promote time: the name input was prefilled with the templated
nameHint (e.g. `head_sphere_low_R`). Typing a semantic suffix produced
combined names like `head_sphere_low_Rfingerpoint_R`. We strip the
template prefix, fix obvious typos, and flag collisions with existing
affordances.

Output: prints the proposed cleanup table. Pass --apply to also write
promoted-affordances-clean.json with renamed entries.
"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

POSES = os.path.dirname(os.path.abspath(__file__))

# nameHint pattern: `{cap_short}_{low|mid|high}_{L|R}`
TEMPLATE_RE = re.compile(r'^([a-zA-Z]+(?:_[a-zA-Z]+)*)_(low|mid|high)_([LR])')

# Common typo fixes the user made on phone
TYPO_FIXES = {
    'fingersspresd': 'fingers_spread',
    'dtomachflat':   'stomach_flat',
    'handclosed':    'hand_closed',
    'handopen':      'hand_open',
    'hangingside':   'hanging_side',
    'closetoside':   'close_to_side',
    'nexttoside':    'next_to_side',
    'onside':        'on_side',
    'onhip':         'on_hip',
    'onhip2':        'on_hip_alt',
    'onthigh':       'on_thigh',
    'onchest':       'on_chest',
    'fingerpoint':   'finger_point',
    'highfive':      'high_five',
    'thumbup':       'thumb_up',
    '3fingersfront': 'three_fingers_front',
    'infrontface':   'in_front_face',
    'infrontthigh':  'in_front_thigh',
    'infrontreaching':'in_front_reaching',
    'infront':       'in_front',
    'handflat':      'hand_flat',
    'handup':        'hand_up',
    'handstomach':   'hand_stomach',
    'salute':        'salute',
    'spread':        'spread',
    'up':            'up',
}

def strip_template(name):
    """Remove the templated prefix `{cap}_{zone}_{side}` from a promoted
    name, returning (user_typed_part, side). The user's typed text was
    appended directly to the prefilled input, so it sits between the
    prefix and the auto-appended trailing `_L` / `_R`."""
    m = TEMPLATE_RE.match(name)
    if not m:
        return None, None
    cap, zone, side = m.group(1), m.group(2), m.group(3)
    rest = name[m.end():]              # everything after the template
    rest = re.sub(rf'_?{side}$', '', rest)  # drop the auto-appended side suffix
    return rest, side

def normalize_user_text(text):
    """Lowercase, fix typos, collapse to snake_case."""
    if not text: return ''
    t = text.lower().strip().lstrip('_')
    if t in TYPO_FIXES:
        return TYPO_FIXES[t]
    # Greedy match longer keys first (e.g. 'infrontreaching' before 'infront')
    for typo in sorted(TYPO_FIXES, key=lambda k: -len(k)):
        if typo in t:
            t = t.replace(typo, TYPO_FIXES[typo])
    return t

def propose_clean_name(promoted_name):
    user_text, side = strip_template(promoted_name)
    if user_text is None:
        return promoted_name, 'unparseable'
    if not user_text:
        # User just hit Promote — keep the template name (it's descriptive enough)
        return promoted_name, 'kept_template'
    clean = normalize_user_text(user_text)
    return f'{clean}_{side}', 'cleaned'

def main():
    apply = '--apply' in sys.argv
    promoted_path = os.path.join(POSES, 'promoted-affordances.json')
    existing_path = os.path.join(POSES, 'affordances-bones.json')
    with open(promoted_path, 'r', encoding='utf-8') as f:
        promoted = json.load(f)
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing = json.load(f)
    existing_names = set(existing.get('affordances', {}).keys())

    aff = promoted.get('affordances', {})
    print(f'Loaded {len(aff)} promoted affordances; {len(existing_names)} already in canonical set\n')

    print(f'{"original":<45} {"→ proposed":<35} {"flag"}')
    print('-' * 105)

    proposals = []
    name_collisions = {}    # newname → list of original names
    for orig in sorted(aff):
        new, kind = propose_clean_name(orig)
        flag = ''
        if new in existing_names:
            flag = '⚠ COLLIDES with existing'
        if new in name_collisions:
            flag = '⚠ COLLIDES with another promoted'
            name_collisions[new].append(orig)
        else:
            name_collisions[new] = [orig]
        if kind == 'kept_template':
            flag = (flag + ' (template)').strip()
        proposals.append({'orig': orig, 'new': new, 'kind': kind, 'flag': flag})
        print(f'  {orig:<43} → {new:<33}  {flag}')

    # Resolve internal collisions by inserting _alt before the side suffix
    # (e.g. on_side_R + on_side_R → on_side_R + on_side_alt_R, not _R_alt)
    seen = {}
    for p in proposals:
        if 'COLLIDES with another' in p['flag']:
            base = p['new']
            seen.setdefault(base, 0)
            seen[base] += 1
            n = seen[base]
            tag = '_alt' if n == 1 else f'_alt{n}'
            # Strip side suffix, append tag, re-add side
            m = re.match(r'^(.*)_([LR])$', base)
            p['new'] = f'{m.group(1)}{tag}_{m.group(2)}' if m else f'{base}{tag}'

    # Summary
    print()
    cleaned = sum(1 for p in proposals if p['kind'] == 'cleaned')
    kept = sum(1 for p in proposals if p['kind'] == 'kept_template')
    collide_existing = sum(1 for p in proposals if 'existing' in p['flag'])
    collide_promoted = sum(1 for p in proposals if 'another' in p['flag'])
    print(f'Cleaned: {cleaned}  · kept-template: {kept}')
    print(f'Collisions: {collide_existing} with existing affordances · {collide_promoted} within promoted set')

    if apply:
        out = dict(promoted)  # shallow copy
        out['affordances'] = {}
        for p in proposals:
            out['affordances'][p['new']] = aff[p['orig']]
        out['cleanupNote'] = 'Names cleaned by _clean_promoted_names.py — original promoted names had template prefix concat bug'
        clean_path = os.path.join(POSES, 'promoted-affordances-clean.json')
        with open(clean_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f'\nWrote {len(out["affordances"])} cleaned affordances to {clean_path}')
    else:
        print('\nDry run. Pass --apply to write promoted-affordances-clean.json')

if __name__ == '__main__':
    main()
