#!/usr/bin/env python3
"""
generate_per_arm_primitives.py — wrap every per-arm affordance from
poses/affordances-bones.json as a single-arm pose primitive.

Each generated primitive:
  - Drives one arm via the affordance (matching the affordance's side)
  - Leaves the other arm as free-space (composable with another primitive)
  - Tags the anchor with the cross-VRM consensus region from
    poses/affordance-region-tags.json (when present)
  - Sets no body rotations — these are arm-only atoms

After running, also writes poses/primitives-manifest.json indexing all
primitives in the directory so the render grid can auto-discover.

Usage:
    python scripts/generate_per_arm_primitives.py
    python scripts/generate_per_arm_primitives.py --dry-run
    python scripts/generate_per_arm_primitives.py --rebuild   # nuke + regenerate per-arm files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POSES_DIR = REPO_ROOT / "poses"
PRIMITIVES_DIR = POSES_DIR / "primitives"
AFFORDANCES_FILE = POSES_DIR / "affordances-bones.json"
REGION_TAGS_FILE = POSES_DIR / "affordance-region-tags.json"
MANIFEST_FILE = POSES_DIR / "primitives-manifest.json"

# Crude semantic tagging by affordance name pattern.
SEMANTIC_TAGS = [
    (r'chin_',           ['face', 'self-touch', 'thinking']),
    (r'fist_at_face',    ['face', 'emphatic']),
    (r'in_front_face',   ['face', 'gesture']),
    (r'on_chest',        ['chest', 'emphasis', 'sincere']),
    (r'chest_[LR]$',     ['chest', 'emphasis']),
    (r'fist_chest',      ['chest', 'emphatic', 'sincere']),
    (r'hip_',            ['torso', 'stance']),
    (r'behind_back',     ['torso', 'formal']),
    (r'in_front_thigh',  ['torso', 'casual']),
    (r'stomach_',        ['torso', 'self-soothing']),
    (r'clasp_low_front', ['formal', 'attentive']),
    (r'arms_crossed',    ['closed', 'defensive']),
    (r'forward_high',    ['reaching', 'gesture']),
    (r'forward_low',     ['open', 'inviting']),
    (r'salute_',         ['greeting', 'formal']),
    (r'up_[LR]$',        ['greeting', 'attention']),
    (r'high_five',       ['greeting', 'casual']),
    (r'peace_',          ['casual', 'gesture']),
    (r'finger_point',    ['pointing', 'directing']),
    (r'index_up',        ['pointing', 'teaching']),
    (r'palm_out_side',   ['gesture', 'open']),
    (r'extended_side',   ['reaching']),
    (r'fist_at_shoulder', ['stance', 'shoulder']),
    (r'hand_open',       ['passive', 'rest']),
    (r'hand_closed',     ['passive', 'rest']),
]


def derive_tags(affordance_name: str) -> list[str]:
    side = 'R' if affordance_name.endswith('_R') else ('L' if affordance_name.endswith('_L') else None)
    tags = ['single-arm']
    if side: tags.append(side)
    for pattern, extra in SEMANTIC_TAGS:
        if re.search(pattern, affordance_name):
            tags.extend(extra)
            break
    return tags


def derive_description(affordance_name: str, region: str | None) -> str:
    side = 'right' if affordance_name.endswith('_R') else ('left' if affordance_name.endswith('_L') else 'one')
    base = f"Single-arm primitive: {side} arm in '{affordance_name}' pose."
    if region:
        base += f" Lands at region '{region}' across the bundled roster."
    base += " Other arm left as free-space for composition with another primitive."
    return base


def get_consensus_region(region_tags: dict, affordance_name: str) -> str | None:
    entry = region_tags.get('affordances', {}).get(affordance_name)
    if not entry: return None
    consensus = entry.get('consensus') or {}
    # Find the side that has consensus
    for side in ('L', 'R'):
        c = consensus.get(side)
        if c and c.get('agreement', 0) >= 0.5:
            return c.get('region')
    return None


def make_primitive(affordance_name: str, region_tags: dict) -> dict:
    side = 'R' if affordance_name.endswith('_R') else 'L'
    anchor_key = 'rightHand' if side == 'R' else 'leftHand'
    other_key = 'leftHand' if side == 'R' else 'rightHand'
    region = get_consensus_region(region_tags, affordance_name)
    return {
        "schema": "augmentum.pose-primitive.v1",
        "name": f"arm_{affordance_name}",
        "tags": derive_tags(affordance_name),
        "description": derive_description(affordance_name, region),
        "family": "per_arm",
        "anchors": {
            anchor_key: {
                "type": "region-contact" if region else "landmark-relative",
                **({"region": region} if region else {"landmark": affordance_name}),
                "tolerance": 0.04,
                "fingerStyle": affordance_name,
            },
            other_key: {"type": "free-space"},
        },
        "body": {},
        "provenance": {
            "source": "auto-generated",
            "generator": "scripts/generate_per_arm_primitives.py",
            "wrapsAffordance": affordance_name,
            "consensusRegion": region,
            "createdAt": time.strftime("%Y-%m-%d"),
        }
    }


def build_manifest() -> dict:
    """Walk poses/primitives/, build a manifest of every primitive."""
    primitives = []
    for path in sorted(PRIMITIVES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if data.get('schema') != 'augmentum.pose-primitive.v1':
            continue
        primitives.append({
            "name": data.get('name', path.stem),
            "file": path.name,
            "tags": data.get('tags', []),
            "family": data.get('family', 'misc'),
            "description": data.get('description', ''),
            "source": data.get('provenance', {}).get('source', 'unknown'),
            "kind": "per-arm" if data.get('name', '').startswith('arm_') else "curated",
        })
    return {
        "schema": "augmentum.primitive-manifest.v1",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(primitives),
        "primitives": primitives,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be written without writing.")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing arm_*.json files before generating.")
    parser.add_argument("--manifest-only", action="store_true", help="Skip per-arm generation; just rebuild the manifest.")
    args = parser.parse_args()

    if not AFFORDANCES_FILE.is_file():
        sys.exit(f"Missing {AFFORDANCES_FILE}")
    affordance_data = json.loads(AFFORDANCES_FILE.read_text(encoding='utf-8'))
    affordances = affordance_data.get('affordances', affordance_data)
    region_tags = {}
    if REGION_TAGS_FILE.is_file():
        try:
            region_tags = json.loads(REGION_TAGS_FILE.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f"[warn] couldn't read region tags: {exc}", file=sys.stderr)

    PRIMITIVES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.manifest_only:
        if args.rebuild:
            removed = 0
            for p in PRIMITIVES_DIR.glob("arm_*.json"):
                if not args.dry_run: p.unlink()
                removed += 1
            print(f"[rebuild] removed {removed} existing arm_*.json files")

        written = 0
        skipped = 0
        for affordance_name in sorted(affordances.keys()):
            if not (affordance_name.endswith('_L') or affordance_name.endswith('_R')):
                print(f"[skip] {affordance_name} — no side suffix")
                skipped += 1
                continue
            primitive = make_primitive(affordance_name, region_tags)
            out_path = PRIMITIVES_DIR / f"arm_{affordance_name}.json"
            if args.dry_run:
                print(f"  would write {out_path.name}")
            else:
                out_path.write_text(json.dumps(primitive, indent=2), encoding='utf-8')
            written += 1
        print(f"{'would write' if args.dry_run else 'wrote'} {written} per-arm primitives; skipped {skipped}")

    if not args.dry_run:
        manifest = build_manifest()
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f"wrote {MANIFEST_FILE.name} ({manifest['count']} primitives total)")
    else:
        print("(manifest would be rebuilt)")


if __name__ == "__main__":
    main()
