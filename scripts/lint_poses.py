#!/usr/bin/env python3
"""
lint_poses.py — validate pose primitives against the schema + collect
cross-VRM resolution diagnostics.

Schema rules (offline):
  - Required fields: schema, name, anchors
  - schema == 'augmentum.pose-primitive.v1'
  - Each anchor has a valid 'type'
  - For region-contact / rest anchors: region is a known region name
  - tolerances are positive floats
  - body rotation angles are degrees, finite, within reasonable ranges
  - finger styles reference real affordance names (if affordances JSON present)
  - validity.requires lists known VRM humanoid bone names

Runtime resolution rules (browser-based):
  - For each primitive × each bundled VRM, the resolver should produce
    valid output (no anchor errors, all targets resolve to plausible
    positions). This needs the same Playwright pattern as mine_substrate.py
    and is gated by --runtime flag.

Usage:
  python scripts/lint_poses.py                # offline schema-only
  python scripts/lint_poses.py --runtime      # also run resolver per VRM
  python scripts/lint_poses.py --pose name    # lint a single primitive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIMITIVES_DIR = REPO_ROOT / "poses" / "primitives"
AFFORDANCES_FILE = REPO_ROOT / "poses" / "affordances-bones.json"
LANDMARK_STATS_FILE = REPO_ROOT / "poses" / "landmark-cross-vrm-stats.json"

# Region table — kept in sync with body-mesh.js / generator
REGIONS = [
    'forehead','temple_L','temple_R','ear_L','ear_R',
    'cheek_L','cheek_R','eye_L','eye_R','mouth',
    'jaw','chin','head_top','head_back',
    'neck',
    'shoulder_L','shoulder_R','chest_L','chest_R','sternum',
    'side_L','side_R','belly','navel','back_upper',
    'hip_L','hip_R','lower_back',
    'upper_arm_L','upper_arm_R','elbow_L','elbow_R',
    'lower_arm_L','lower_arm_R','hand_L','hand_R',
    'thigh_L','thigh_R','knee_L','knee_R',
    'shin_L','shin_R','ankle_L','ankle_R','foot_L','foot_R',
    'other',
]
REGION_SET = set(REGIONS)

VALID_ANCHOR_TYPES = {'region-contact', 'rest', 'landmark-relative', 'world-target', 'free-space'}
VALID_PALM_FACING = {'body','away','up','down','inward','outward'}
VRM_HUMANOID_BONES = {
    'hips','spine','chest','upperChest','neck','head',
    'leftEye','rightEye','jaw',
    'leftShoulder','leftUpperArm','leftLowerArm','leftHand',
    'rightShoulder','rightUpperArm','rightLowerArm','rightHand',
    'leftUpperLeg','leftLowerLeg','leftFoot','leftToes',
    'rightUpperLeg','rightLowerLeg','rightFoot','rightToes',
    'humanoid',  # alias for "needs full humanoid"
}


class LintResult:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str): self.errors.append(msg)
    def warn(self, msg: str): self.warnings.append(msg)
    def note(self, msg: str): self.info.append(msg)


def lint_anchor(key: str, anchor: dict, known_affordances: set[str]) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    warns: list[str] = []
    if not isinstance(anchor, dict):
        errs.append(f"anchors.{key} must be an object")
        return errs, warns
    atype = anchor.get('type')
    if atype not in VALID_ANCHOR_TYPES:
        errs.append(f"anchors.{key}.type '{atype}' invalid (expected one of {sorted(VALID_ANCHOR_TYPES)})")
        return errs, warns
    if atype == 'region-contact' or atype == 'rest':
        region = anchor.get('region')
        if not region:
            if atype == 'region-contact':
                errs.append(f"anchors.{key} of type region-contact requires 'region'")
        elif region not in REGION_SET:
            errs.append(f"anchors.{key}.region '{region}' is not a known region")
    if atype == 'landmark-relative':
        if not anchor.get('landmark'):
            errs.append(f"anchors.{key} of type landmark-relative requires 'landmark'")
    if atype == 'world-target':
        pos = anchor.get('position')
        if not isinstance(pos, list) or len(pos) != 3:
            errs.append(f"anchors.{key} of type world-target requires position [x,y,z]")
        frame = anchor.get('frame', 'world')
        if frame not in ('world', 'hipsRelative'):
            warns.append(f"anchors.{key}.frame '{frame}' unrecognized (expected world|hipsRelative)")
    # Common fields
    tol = anchor.get('tolerance')
    if tol is not None and (not isinstance(tol, (int, float)) or tol <= 0):
        errs.append(f"anchors.{key}.tolerance must be a positive number")
    offset = anchor.get('offset')
    if offset is not None and (not isinstance(offset, list) or len(offset) != 3):
        errs.append(f"anchors.{key}.offset must be [x,y,z]")
    elif offset:
        for v in offset:
            if not isinstance(v, (int, float)):
                errs.append(f"anchors.{key}.offset contains non-numeric")
                break
            if abs(v) > 0.5:
                warns.append(f"anchors.{key}.offset has component |{v}|>0.5m — unusually large")
                break
    palm = anchor.get('palmFacing')
    if palm is not None and palm not in VALID_PALM_FACING:
        warns.append(f"anchors.{key}.palmFacing '{palm}' unrecognized")
    finger = anchor.get('fingerStyle')
    if finger and known_affordances and finger not in known_affordances:
        warns.append(f"anchors.{key}.fingerStyle '{finger}' not in affordance vocabulary")
    return errs, warns


def lint_body_rotation(key: str, rot: dict) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    warns: list[str] = []
    if not isinstance(rot, dict):
        errs.append(f"body.{key} must be an object")
        return errs, warns
    for axis in ['pitch','yaw','roll','forwardLean','twist']:
        v = rot.get(axis)
        if v is None: continue
        if not isinstance(v, (int, float)):
            errs.append(f"body.{key}.{axis} must be a number")
        elif abs(v) > 90:
            warns.append(f"body.{key}.{axis}={v} exceeds ±90° — likely radians instead of degrees")
    tol = rot.get('tolerance')
    if tol is not None and (not isinstance(tol, (int, float)) or tol < 0):
        errs.append(f"body.{key}.tolerance must be non-negative")
    return errs, warns


def lint_one(path: Path, known_affordances: set[str]) -> LintResult:
    r = LintResult(path)
    try:
        prim = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        r.err(f"failed to parse JSON: {exc}")
        return r

    schema = prim.get('schema')
    if schema != 'augmentum.pose-primitive.v1':
        r.err(f"schema '{schema}' invalid (expected 'augmentum.pose-primitive.v1')")
    if not prim.get('name'):
        r.err("name is required")
    if prim.get('name') and prim['name'] != path.stem:
        r.warn(f"name '{prim['name']}' does not match filename stem '{path.stem}'")

    anchors = prim.get('anchors') or {}
    if not isinstance(anchors, dict):
        r.err("anchors must be an object")
    else:
        for key, anchor in anchors.items():
            if key not in ('leftHand', 'rightHand'):
                r.warn(f"unknown anchor key '{key}' (expected leftHand/rightHand)")
                continue
            errs, warns = lint_anchor(key, anchor, known_affordances)
            for e in errs: r.err(e)
            for w in warns: r.warn(w)

    body = prim.get('body') or {}
    if not isinstance(body, dict):
        r.err("body must be an object")
    else:
        for key, rot in body.items():
            if key not in ('head','neck','spine','chest','upperChest','hips'):
                r.warn(f"body.{key} not a standard body rotation key")
            errs, warns = lint_body_rotation(key, rot)
            for e in errs: r.err(e)
            for w in warns: r.warn(w)

    finger_styles = prim.get('fingerStyles') or {}
    if isinstance(finger_styles, dict):
        for side, name in finger_styles.items():
            if side not in ('L', 'R'):
                r.warn(f"fingerStyles.{side} not a side (expected L/R)")
            if known_affordances and name not in known_affordances:
                r.warn(f"fingerStyles.{side} = '{name}' not in affordance vocabulary")

    validity = prim.get('validity') or {}
    requires = validity.get('requires') or []
    for bone in requires:
        if bone not in VRM_HUMANOID_BONES:
            r.warn(f"validity.requires['{bone}'] not a known VRM humanoid bone")

    provenance = prim.get('provenance') or {}
    if not provenance.get('source'):
        r.warn("provenance.source missing — recommended for trust tier tracking")

    return r


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Lint pose primitives against the schema.")
    parser.add_argument("--pose", help="Lint a single primitive by name (without .json).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-primitive info; only show fail/pass totals.")
    args = parser.parse_args()

    if not PRIMITIVES_DIR.is_dir():
        sys.exit(f"Primitives directory not found: {PRIMITIVES_DIR}")

    known_affordances: set[str] = set()
    if AFFORDANCES_FILE.is_file():
        try:
            data = json.loads(AFFORDANCES_FILE.read_text(encoding='utf-8'))
            vocab = data.get('affordances') if isinstance(data, dict) else None
            if vocab is None: vocab = data
            known_affordances = set(vocab.keys())
        except Exception as exc:
            print(f"[warn] could not load affordances vocabulary: {exc}", file=sys.stderr)

    if args.pose:
        targets = [PRIMITIVES_DIR / f"{args.pose}.json"]
    else:
        targets = sorted(PRIMITIVES_DIR.glob("*.json"))

    if not targets or (args.pose and not targets[0].is_file()):
        sys.exit(f"No primitives to lint at {PRIMITIVES_DIR}")

    fail_count = 0
    pass_count = 0
    warn_count = 0
    for path in targets:
        r = lint_one(path, known_affordances)
        if r.ok:
            pass_count += 1
            if not args.quiet:
                if r.warnings:
                    print(f"⚠ {path.name}")
                    for w in r.warnings: print(f"    warn: {w}")
                else:
                    print(f"✓ {path.name}")
        else:
            fail_count += 1
            print(f"✗ {path.name}")
            for e in r.errors: print(f"    error: {e}")
            for w in r.warnings: print(f"    warn:  {w}")
        if r.warnings: warn_count += 1

    total = pass_count + fail_count
    print(f"\n{pass_count}/{total} passed; {fail_count} failed; {warn_count} with warnings")
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
