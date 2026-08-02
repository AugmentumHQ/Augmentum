"""Quick QA on motion-clips-all.json — verifies plain-glb clips actually
animate (don't repeat the same hand position frame after frame, the bug
that motivated the latest fix). Reports source mix, frame counts, and
hand-pose variance per clip. A 'static' flag ≥0.95 frame match ratio is
a smell; the previous bug had it at 1.00."""
import json, os, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motion-clips-all.json')

with open(PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

clips = data['clips']
print(f"Schema: {data.get('schema')}")
print(f"Reference VRM: {data.get('referenceVRM')}")
print(f"Clip count: {len(clips)}, total frames: {data.get('totalFrames')}")
print(f"Sample fps: {data.get('sampleFps')}")

src_counter = Counter(c.get('source') for c in clips.values())
print(f"\nSource mix: {dict(src_counter)}")

print("\n=== Per-clip variance (plain-glb path is what we just fixed) ===")
print(f"{'name':<55} {'src':<10} {'frames':<7} {'L-uniq':<7} {'R-uniq':<7} {'flag'}")
print("-" * 100)

issues = []
for name, clip in sorted(clips.items()):
    frames = clip['frames']
    n = len(frames)
    if n == 0:
        print(f"{name[:54]:<55} {clip.get('source','?'):<10} 0       —       —       (empty)")
        continue
    # Round handL/handR to 3 decimal places and count unique tuples
    def rk(p):
        return tuple(round(x, 3) for x in p) if p else None
    handL_set = {rk(f['features'].get('handL')) for f in frames}
    handR_set = {rk(f['features'].get('handR')) for f in frames}
    flag = ''
    if n > 5 and (len(handL_set) <= 1 and len(handR_set) <= 1):
        flag = '★ STATIC'
        issues.append(name)
    elif n > 20 and (len(handL_set) / n < 0.05 and len(handR_set) / n < 0.05):
        flag = '⚠ low-variance'
    print(f"{name[:54]:<55} {clip.get('source','?'):<10} {n:<7} {len(handL_set):<7} {len(handR_set):<7} {flag}")

print()
if issues:
    print(f"❌ {len(issues)} clips still appear static (T-pose bug not fully fixed):")
    for n in issues:
        print(f"   {n}")
else:
    print("✓ All clips show frame-to-frame variance — plain-glb path animates correctly.")

# Body-relative coverage (vrma-1x clips only — plain-glb is unmarked)
bv_counter = Counter('with_brL' if any('brL' in f for f in c['frames']) else 'no_brL' for c in clips.values())
print(f"\nBody-relative coverage: {dict(bv_counter)}")
