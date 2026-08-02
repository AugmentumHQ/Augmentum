"""Re-extract affordances with explicit L/R hand semantics.

V1 mistake: affordances like `chest` and `chest_right` were ambiguous. The
extracted value depended on which hand the source pose used, but the name
suggested symmetric. Result: picking `chest_right` for the right hand aimed
the right hand at the avatar's LEFT side of chest (because the source pose
was a LEFT hand crossed over to the right, and the mirror put X negative).

V2 convention: every affordance ends in `_L` or `_R` indicating which HAND
it's for, NOT which body side. The IK demo's L dropdown shows `_L` only
and R shows `_R` only, eliminating the cross-side picking footgun.

For symmetric affordances (clasp_low_front, behind_back, arms_crossed),
we now extract per-hand positions separately rather than averaging L+R.
"""
import json, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motion-db-conversation.json')
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affordances.json')

with open(src, 'r', encoding='utf-8') as f:
    db = json.load(f)

# Map authored static poses to (affordance_name_with_HAND_suffix, hand_to_sample) tuples.
# `hand_to_sample` is which hand position to extract from the database record.
POSE_TO_AFFORDANCE = {
    # Right hand on chin (avatar's right side, near face)
    "thinking-righthandonchin-lefthandonwaistpose-2026-05-03T02-39-12":
        [("chin_R", "R"), ("hip_L", "L")],
    # The LEFT hand sample in this pose is wrapping cross-body to the
    # avatar's RIGHT side of chest (X+). By the convention where _L/_R
    # suffix means "which hand's natural territory", the X+ position is
    # `chest_R` (right hand's natural chest reach), and we'll mirror to
    # get `chest_L` at X-. Naming by SOURCE hand was confusing because
    # the position landed on the OPPOSITE body side from what the suffix
    # implied.
    "thinking-righthandonchin-lefthandunderchest-pose-2026-05-03T02-51-14":
        [("chin_R", "R"), ("chest_R", "L")],
    # Left hand on stomach (avatar's left side, low front)
    "idle-lefthandonstomach-righthandonside-pose-2026-05-03T03-20-40":
        [("stomach_L", "L"), ("side_relaxed_R", "R")],
    # Left hand on hip (4 head variants share same hand position)
    "idle-turnedlefthandonhip-righthandonsidepose-2026-05-03T03-31-45":
        [("hip_L", "L"), ("side_relaxed_R", "R")],
    "idle-turnedlefthandonhip-righthandonsidepose-headstraight-2026-05-03T03-31-45":
        [("hip_L", "L"), ("side_relaxed_R", "R")],
    "idle-turnedlefthandonhip-righthandonsidepose-headturnedright-2026-05-03T03-31-45":
        [("hip_L", "L"), ("side_relaxed_R", "R")],
    # Right hand on hip
    "Rhandonhip-pose-2026-04-30T22-48-18":
        [("hip_R", "R"), ("side_relaxed_L", "L")],
    # Both hands clasped low front — extract L and R separately
    "handclaspedinfront-pose-2026-05-01T04-47-40":
        [("clasp_low_front_L", "L"), ("clasp_low_front_R", "R")],
    "handclaspedinfront-pose-headturnedleft-2026-05-01T04-47-40":
        [("clasp_low_front_L", "L"), ("clasp_low_front_R", "R")],
    "handclaspedinfront-pose-headturnedright-2026-05-01T04-47-40":
        [("clasp_low_front_L", "L"), ("clasp_low_front_R", "R")],
    "handclaspedinfront-pose-headdownofftosidethinking-2026-05-01T04-47-40":
        [("clasp_low_front_L", "L"), ("clasp_low_front_R", "R")],
    # Both hands behind back — extract L and R separately
    "handsbehindback-pose-2026-05-03T01-50-27":
        [("behind_back_L", "L"), ("behind_back_R", "R")],
    # Arms crossed — extract L and R separately
    "Arms-crossed-pose":
        [("arms_crossed_L", "L"), ("arms_crossed_R", "R")],
    # Right hand forward (talking gesture)
    "righthandinfront-talking-pose-2026-05-03T04-09-28":
        [("forward_high_R", "R"), ("side_relaxed_L", "L")],
    "righthandinfrontdown-talking-pose-2026-05-03T04-09-28":
        [("forward_low_R", "R"), ("side_relaxed_L", "L")],
    # Forward leaning (mostly hands at sides)
    "leaningin-pose-2026-05-03T02-00-26":
        [("side_relaxed_L", "L"), ("side_relaxed_R", "R")],
    # Sitting on couch edge — asymmetric (L hand on couch beside her, R on thigh)
    "Sitting-Rhandonleg-Lhandoncouch-pose-2026-04-30T23-25-30":
        [("lap_L", "L"), ("thigh_R", "R")],
}
SKIP = {"Discord Dark VRM 2.0"}

# Collect samples
samples = defaultdict(list)
for r in db["records"]:
    if r["format"] != "json": continue
    src_name = r["source"]
    if src_name in SKIP: continue
    if src_name not in POSE_TO_AFFORDANCE:
        continue
    feats = r["features"]
    for affordance, hand_side in POSE_TO_AFFORDANCE[src_name]:
        pos = feats["handL"] if hand_side == "L" else feats["handR"]
        samples[affordance].append({"src": src_name, "pos": pos})

def avg(positions):
    n = len(positions)
    return [sum(p[i] for p in positions) / n for i in range(3)]

def mirror_pos(pos):
    """Mirror across X axis. Used to derive a hand's affordance from the
    other hand's authored pose when the user only authored one side."""
    return [-pos[0], pos[1], pos[2]]

# Build affordance map
affordances = {}
for name, recs in samples.items():
    positions = [r["pos"] for r in recs]
    affordances[name] = {
        "position": [round(v, 4) for v in avg(positions)],
        "sampleCount": len(recs),
        "sources": sorted(set(r["src"] for r in recs)),
    }

# Mirror unilateral affordances to derive the other-hand counterpart by
# flipping X (anatomy is bilaterally symmetric for these hand-position roles).
MIRROR_PAIRS = [
    # (source_affordance, derived_mirror, derived_hand)
    ("chin_R", "chin_L"),
    ("stomach_L", "stomach_R"),
    ("chest_L", "chest_R"),
    ("forward_high_R", "forward_high_L"),
    ("forward_low_R", "forward_low_L"),
    ("lap_L", "lap_R"),
    ("thigh_R", "thigh_L"),
]
for src_aff, mirrored in MIRROR_PAIRS:
    if src_aff in affordances and mirrored not in affordances:
        affordances[mirrored] = {
            "position": [round(v, 4) for v in mirror_pos(affordances[src_aff]["position"])],
            "sampleCount": 0,
            "sources": [],
            "derivedFrom": src_aff,
            "method": "mirror",
        }

# Hand-coded fallback for side_relaxed if not sampled
for name, default_x in [("side_relaxed_L", -0.18), ("side_relaxed_R", 0.18)]:
    if name not in affordances:
        affordances[name] = {
            "position": [default_x, -0.05, 0.02],
            "sampleCount": 0,
            "sources": [],
            "method": "hand-coded",
        }

# Validate naming convention: every affordance MUST end in _L or _R
bad = [n for n in affordances if not (n.endswith("_L") or n.endswith("_R"))]
if bad:
    print(f"WARNING: affordances without _L/_R suffix: {bad}")

out = {
    "schema": "augmentum.affordances.v2",
    "sourceCorpus": "motion-db-conversation.json",
    "units": "skeleton-height-normalized, hip-relative, world-aligned (matches extractor)",
    "naming": "Every affordance ends in _L or _R indicating which HAND it is for. The IK demo filters dropdowns by suffix so picking a left-hand target for the right hand isn't possible.",
    "axisConvention": "X positive = avatar's right side. Y positive = up. Z negative = in front (forward).",
    "affordances": dict(sorted(affordances.items())),
}

with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

# Summarize
l_count = sum(1 for n in affordances if n.endswith("_L"))
r_count = sum(1 for n in affordances if n.endswith("_R"))
print(f"V2 affordances: {len(affordances)} total ({l_count} L, {r_count} R)")
print(f"\nL hand affordances:")
for name in sorted(affordances):
    if not name.endswith("_L"): continue
    info = affordances[name]
    p = info["position"]
    method = info.get("method", "sampled")
    n = info.get("sampleCount", 0)
    print(f"  {name:25} pos=[{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]  n={n} [{method}]")

print(f"\nR hand affordances:")
for name in sorted(affordances):
    if not name.endswith("_R"): continue
    info = affordances[name]
    p = info["position"]
    method = info.get("method", "sampled")
    n = info.get("sampleCount", 0)
    print(f"  {name:25} pos=[{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]  n={n} [{method}]")

print(f"\nWrote {os.path.getsize(dst)/1024:.1f} KB to affordances.json")
