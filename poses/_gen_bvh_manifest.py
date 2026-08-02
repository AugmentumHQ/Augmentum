"""Generate poses/bvh-manifest.json for the scene-test BVH dropdown.

Scans poses/external/sillytavern-pack/animation*/ and produces a JSON file
the scene-test page fetches at load time. Re-run after adding new BVH files.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

POSES = Path(os.path.dirname(os.path.abspath(__file__)))
# BVH files live under poses/external/sillytavern-pack/, which the augmentum
# proxy server exposes at /bvh-library/. The manifest is written INTO that
# same directory so the dropdown can fetch /bvh-library/bvh-manifest.json.
BVH_ROOT = POSES / 'external' / 'sillytavern-pack'
SERVED_PREFIX = '/bvh-library'
SOURCES = [
    BVH_ROOT / 'animation',
    BVH_ROOT / 'animation_nitral-fork',
]

# Regex prefixes → group label. First match wins.
GROUP_RULES = [
    (re.compile(r'^action_'),       'Actions'),
    (re.compile(r'^dance_'),        'Dances'),
    (re.compile(r'^exercise_'),     'Exercises'),
    (re.compile(r'^hitarea_'),      'Hit Reactions'),
    (re.compile(r'^reaction_'),     'Hit Reactions'),
    (re.compile(r'^(kneel|laying|neutral|sit)_idle'), 'Idles'),
]

def categorize(stem: str) -> str:
    for rx, label in GROUP_RULES:
        if rx.match(stem):
            return label
    return 'Emotions'   # default for everything else (admiration, anger, ...)


def pretty(stem: str) -> str:
    """File stem → human-friendly label."""
    s = stem.replace('_', ' ').strip()
    # Title case but preserve numeric suffixes
    return re.sub(r'\b(\w)', lambda m: m.group(1).upper(), s)


def build():
    out = {'version': 1, 'sources': []}
    for src in SOURCES:
        if not src.exists():
            print(f'[skip] {src} not found', file=sys.stderr)
            continue
        bvhs = sorted(src.glob('*.bvh'))
        if not bvhs:
            continue
        # Source label = directory name, made friendly
        src_label = 'Main' if src.name == 'animation' else 'Nitral fork'
        # URL paths use the /bvh-library/ mount → directory name within
        rel_dir = SERVED_PREFIX + '/' + src.relative_to(BVH_ROOT).as_posix()
        groups = {}
        for f in bvhs:
            stem = f.stem
            grp = categorize(stem)
            groups.setdefault(grp, []).append({
                'label': pretty(stem),
                'file': stem + '.bvh',
            })
        # Stable group order
        ordered = []
        for label in ['Actions', 'Dances', 'Exercises', 'Idles', 'Hit Reactions', 'Emotions']:
            if label in groups:
                ordered.append({'label': label, 'items': groups[label]})
        out['sources'].append({
            'label': src_label,
            'baseUrl': rel_dir,
            'count': len(bvhs),
            'groups': ordered,
        })
    # Manifest goes inside the mounted dir so it's reachable at
    # /bvh-library/bvh-manifest.json (alongside the actual .bvh files).
    out_path = BVH_ROOT / 'bvh-manifest.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    total = sum(s['count'] for s in out['sources'])
    print(f'wrote {out_path}  ({total} files across {len(out["sources"])} sources)')


if __name__ == '__main__':
    build()
