"""Extract affordance positions from authored static poses in the motion DB.

Each authored pose maps to one or more (affordance, hand_side) tuples.
Per affordance, we collect sample hand positions (normalized hip-relative
coords from the database) and average them. The result is the affordance
library that avatar-ik.js targets.
"""
import json, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motion-db-conversation.json')
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affordances.json')

with open(src, 'r', encoding='utf-8') as f:
    db = json.load(f)

# Map authored static poses to (affordance, hand_side) tuples.
POSE_TO_AFFORDANCE = {
    "thinking-righthandonchin-lefthandonwaistpose-2026-05-03T02-39-12": [("chin", "R"), ("hip_left", "L")],
    "thinking-righthandonchin-lefthandunderchest-pose-2026-05-03T02-51-14": [("chin", "R"), ("chest", "L")],
    "idle-lefthandonstomach-righthandonside-pose-2026-05-03T03-20-40": [("stomach", "L"), ("side_relaxed", "R")],
    "idle-turnedlefthandonhip-righthandonsidepose-2026-05-03T03-31-45": [("hip_left", "L"), ("side_relaxed", "R")],
    "idle-turnedlefthandonhip-righthandonsidepose-headstraight-2026-05-03T03-31-45": [("hip_left", "L"), ("side_relaxed", "R")],
    "idle-turnedlefthandonhip-righthandonsidepose-headturnedright-2026-05-03T03-31-45": [("hip_left", "L"), ("side_relaxed", "R")],
    "Rhandonhip-pose-2026-04-30T22-48-18": [("hip_right", "R"), ("side_relaxed", "L")],
    "handclaspedinfront-pose-2026-05-01T04-47-40": [("clasp_low_front", "L"), ("clasp_low_front", "R")],
    "handclaspedinfront-pose-headturnedleft-2026-05-01T04-47-40": [("clasp_low_front", "L"), ("clasp_low_front", "R")],
    "handclaspedinfront-pose-headturnedright-2026-05-01T04-47-40": [("clasp_low_front", "L"), ("clasp_low_front", "R")],
    "handclaspedinfront-pose-headdownofftosidethinking-2026-05-01T04-47-40": [("clasp_low_front", "L"), ("clasp_low_front", "R")],
    "handsbehindback-pose-2026-05-03T01-50-27": [("behind_back", "L"), ("behind_back", "R")],
    "Arms-crossed-pose": [("arms_crossed", "L"), ("arms_crossed", "R")],
    "righthandinfront-talking-pose-2026-05-03T04-09-28": [("forward_high", "R"), ("side_relaxed", "L")],
    "righthandinfrontdown-talking-pose-2026-05-03T04-09-28": [("forward_low", "R"), ("side_relaxed", "L")],
    "leaningin-pose-2026-05-03T02-00-26": [("side_relaxed", "L"), ("side_relaxed", "R")],
    "Sitting-Rhandonleg-Lhandoncouch-pose-2026-04-30T23-25-30": [("lap_left", "L"), ("thigh_right", "R")],
}
SKIP = {"Discord Dark VRM 2.0"}

samples = defaultdict(list)
unmapped = []
for r in db["records"]:
    if r["format"] != "json": continue
    src_name = r["source"]
    if src_name in SKIP: continue
    if src_name not in POSE_TO_AFFORDANCE:
        unmapped.append(src_name)
        continue
    feats = r["features"]
    for affordance, side in POSE_TO_AFFORDANCE[src_name]:
        if side == "L":
            samples[affordance].append({"src": src_name, "pos": feats["handL"]})
        elif side == "R":
            samples[affordance].append({"src": src_name, "pos": feats["handR"]})

def avg(positions):
    n = len(positions)
    return [sum(p[i] for p in positions) / n for i in range(3)]

def mirror_pos(pos):
    return [-pos[0], pos[1], pos[2]]

affordances = {}
for name, recs in samples.items():
    positions = [r["pos"] for r in recs]
    affordances[name] = {
        "position": [round(v, 4) for v in avg(positions)],
        "sampleCount": len(recs),
        "sources": sorted(set(r["src"] for r in recs)),
    }

# Mirror across X for the opposite-hand counterparts
MIRROR_PAIRS = [
    ("hip_left", "hip_right"),
    ("chin", "chin_left"),
    ("stomach", "stomach_right"),
    ("chest", "chest_right"),
    ("forward_high", "forward_high_left"),
    ("forward_low", "forward_low_left"),
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

# Hand-coded sided variants for side_relaxed since the sampled "side_relaxed"
# was collected from both L and R contributions and is ambiguous.
HAND_CODED = {
    "side_relaxed_L": {"position": [-0.18, -0.05, 0.02], "sampleCount": 0, "sources": [], "method": "hand-coded"},
    "side_relaxed_R": {"position": [0.18, -0.05, 0.02], "sampleCount": 0, "sources": [], "method": "hand-coded"},
}
if "side_relaxed" in affordances:
    affordances["side_relaxed_L"] = HAND_CODED["side_relaxed_L"]
    affordances["side_relaxed_R"] = HAND_CODED["side_relaxed_R"]
    del affordances["side_relaxed"]

out = {
    "schema": "augmentum.affordances.v1",
    "sourceCorpus": "motion-db-conversation.json",
    "units": "skeleton-height-normalized, hip-relative, avatar-local frame",
    "note": "X positive = avatar's right side; X negative = avatar's left side. Y positive = up. Z negative = in front (forward).",
    "affordances": dict(sorted(affordances.items())),
}

with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"Extracted {len(affordances)} affordances:")
for name, info in sorted(affordances.items()):
    p = info["position"]
    n = info.get("sampleCount", 0)
    method = info.get("method", "sampled")
    print(f"  {name:25} pos=[{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]  n={n} [{method}]")

if unmapped:
    print(f"\nUnmapped poses (not in POSE_TO_AFFORDANCE):")
    for u in unmapped: print(f"  {u}")

print(f"\nWrote {os.path.getsize(dst)/1024:.1f} KB to affordances.json")
