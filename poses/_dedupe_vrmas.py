"""Dedupe .vrma files in poses/ by SHA-256 content hash. Build a canonical
manifest the capture tool can consume — one path per unique animation,
preferring non-external/ copies so the curated set stays canonical.
"""
import hashlib, os, sys, json
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

POSES = os.path.dirname(os.path.abspath(__file__))

hashes = defaultdict(list)
total = 0
total_bytes = 0
for root, _, files in os.walk(POSES):
    for f in files:
        if not f.lower().endswith('.vrma'):
            continue
        p = os.path.join(root, f)
        total += 1
        total_bytes += os.path.getsize(p)
        with open(p, 'rb') as fp:
            h = hashlib.sha256(fp.read()).hexdigest()[:12]
        hashes[h].append(p)

unique = len(hashes)
dup_clusters = {h: paths for h, paths in hashes.items() if len(paths) > 1}
saved = total - unique

print(f"Total .vrma files: {total}  ({total_bytes/1024/1024:.1f} MB)")
print(f"Unique by content: {unique}")
print(f"Duplicate clusters: {len(dup_clusters)}")
print(f"Files saved by dedup: {saved}\n")

def to_rel(p):
    return p.replace(POSES, '').replace('\\', '/').lstrip('/')

if dup_clusters:
    print("=== Duplicate clusters (newest copy preferred) ===")
    for h, paths in sorted(dup_clusters.items(), key=lambda x: -len(x[1])):
        print(f"  [{h}] ({len(paths)} copies):")
        for p in paths:
            print(f"    {to_rel(p)}")
        print()

# Pick canonical path: prefer non-external (curated > scraped)
print("=== Canonical unique set ===")
canonical = []
for h, paths in hashes.items():
    paths_sorted = sorted(paths, key=lambda p: ('external' in p, p))
    canonical.append(paths_sorted[0])
canonical.sort()
for p in canonical:
    size = os.path.getsize(p) / 1024
    print(f"  {to_rel(p):75}  {size:6.1f} KB")

manifest = {
    'totalFound': total,
    'uniqueByContent': unique,
    'duplicateClusters': len(dup_clusters),
    'canonical': [
        {
            'path': to_rel(p),
            'sizeKB': round(os.path.getsize(p) / 1024, 1),
            'hash': next(h for h, ps in hashes.items() if p in ps),
        }
        for p in canonical
    ],
}
out_path = os.path.join(POSES, 'vrma-manifest.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote manifest to {to_rel(out_path)}")
