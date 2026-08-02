#!/usr/bin/env python3
"""One-shot inventory of all BVH/VRMA animation files vs what the
atlas/library actually references.

Run from repo root:
    python .claude/skills/augmentum-dev/scripts/anim_inventory.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
os.chdir(REPO)

JS_SOURCES = [
    "ui/scripts/anim-atlas.js",
    "ui/scripts/avatar-vrma-library.js",
    "ui/scripts/avatar-pose-presets.js",
]

# URL mount → on-disk dir. `/bvh-library/` is mounted from
# `poses/external/sillytavern-pack/` per server.py.
URL_MOUNTS: dict[str, str] = {
    "bvh-library/": "poses/external/sillytavern-pack/",
    "ui/lib/animations/": "ui/lib/animations/",
    "poses/": "poses/",
}


def _url_to_disk(ref: str) -> str:
    """Translate a served URL (e.g. /bvh-library/animation/x.bvh) to
    the on-disk path the static mount resolves against."""
    ref = ref.lstrip("/")
    for mount, disk in URL_MOUNTS.items():
        if ref.startswith(mount):
            return disk + ref[len(mount):]
    return ref


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def collect_disk_files() -> list[tuple[Path, str, int]]:
    files: list[tuple[Path, str, int]] = []
    for ext in ("vrma", "bvh"):
        for p in Path(".").rglob(f"*.{ext}"):
            parts = p.parts
            if ".git" in parts:
                continue
            if "worktrees" in _norm(str(p)):
                continue
            if "__pycache__" in parts:
                continue
            files.append((p, ext, p.stat().st_size))
    return files


def collect_refs() -> set[str]:
    refs: set[str] = set()
    # Allow spaces in filenames (a handful of nitral-fork VRMAs use them:
    # "model pose.vrma", "peace sign.vrma", "show full body.vrma").
    pat = re.compile(
        r"""["'](/(?:ui/lib/animations|bvh-library|poses)/[A-Za-z0-9 _./-]+\.(?:vrma|bvh))["']"""
    )
    for js in JS_SOURCES:
        try:
            text = Path(js).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in pat.finditer(text):
            refs.add(m.group(1).lstrip("/"))
    return refs


def main() -> None:
    files = collect_disk_files()
    refs = collect_refs()

    referenced: list[tuple[Path, str, int]] = []
    orphan: list[tuple[Path, str, int]] = []

    # Translate URL refs to expected on-disk paths via the mount table.
    disk_refs = {_url_to_disk(r) for r in refs}

    for p, ext, sz in files:
        pstr = _norm(str(p))
        hit = False
        for ref in disk_refs:
            if pstr.endswith(ref) or pstr == ref:
                hit = True
                break
        if hit:
            referenced.append((p, ext, sz))
        else:
            orphan.append((p, ext, sz))

    # Dedupe by basename across paths
    by_name: dict[str, list[Path]] = defaultdict(list)
    for p, _, _ in files:
        by_name[p.name].append(p)
    dupes = {n: ps for n, ps in by_name.items() if len(ps) > 1}

    print()
    print("=" * 72)
    print(f"  Animation file inventory")
    print("=" * 72)
    print()
    print(f"  Total files on disk:            {len(files)}")
    print(f"  Unique basenames:               {len(by_name)}")
    print(f"  Files with duplicate basenames: {sum(len(ps) for ps in dupes.values())}")
    print(f"  Atlas/library refs:             {len(refs)}")
    print(f"  Referenced on disk:             {len(referenced)}")
    print(f"  Orphans (unreferenced):         {len(orphan)}")
    print()

    by_dir: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ref": 0, "orphan": 0, "vrma": 0, "bvh": 0, "size": 0}
    )
    for p, ext, sz in files:
        d = _norm(str(p.parent))
        by_dir[d]["total"] += 1
        by_dir[d][ext] += 1
        by_dir[d]["size"] += sz
        if any(p == r[0] for r in referenced):
            by_dir[d]["ref"] += 1
        else:
            by_dir[d]["orphan"] += 1

    print("Per-directory breakdown:")
    print(f"  {'dir':<60} {'total':>6} {'ref':>5} {'orph':>5} {'vrma':>5} {'bvh':>5} {'MB':>6}")
    for d, c in sorted(by_dir.items(), key=lambda kv: -kv[1]["total"]):
        mb = c["size"] / (1024 * 1024)
        print(
            f"  {d[-60:]:<60} {c['total']:>6} {c['ref']:>5} "
            f"{c['orphan']:>5} {c['vrma']:>5} {c['bvh']:>5} {mb:>6.1f}"
        )

    print()
    if dupes:
        print("Duplicate basenames (same file replicated across dirs):")
        for n in sorted(dupes)[:25]:
            paths = sorted(_norm(str(p)) for p in dupes[n])
            print(f"  {n}  ×{len(paths)}")
            for p in paths:
                print(f"      {p}")
        if len(dupes) > 25:
            print(f"  ... and {len(dupes) - 25} more")

    print()
    print("Largest orphans (top 20 by size):")
    orphan_sorted = sorted(orphan, key=lambda x: -x[2])[:20]
    for p, ext, sz in orphan_sorted:
        kb = sz / 1024
        print(f"  {kb:>8.1f} KB  {ext}  {_norm(str(p))}")

    # ── Actionable split ────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  Actionable categories")
    print("=" * 72)

    ref_basenames = {p.name for p, _, _ in referenced}
    orphan_by_basename: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
    for p, ext, sz in orphan:
        orphan_by_basename[p.name].append((p, ext, sz))

    # Bucket 1 — basename also exists as referenced (this is a true dupe;
    # the referenced copy is canonical, the orphan is bytewise redundant).
    redundant_dupes: list[tuple[Path, str, int]] = []
    # Bucket 2 — basename NOT referenced anywhere; this is a UNIQUE asset
    # we have but never wired. Candidates for adoption (atlas registration).
    unwired_unique: list[tuple[Path, str, int]] = []
    # Bucket 3 — same basename appears in multiple orphan paths (no
    # canonical version exists). Keep one, delete the rest.
    unwired_dupes: list[tuple[Path, str, int]] = []

    seen_unwired_basenames: set[str] = set()
    for name, group in orphan_by_basename.items():
        if name in ref_basenames:
            redundant_dupes.extend(group)
        elif len(group) > 1:
            unwired_dupes.extend(group)
        else:
            unwired_unique.extend(group)

    total_mb = lambda items: sum(sz for _, _, sz in items) / (1024 * 1024)

    print()
    print(f"  1. UNWIRED UNIQUE assets (consider adding to atlas):")
    print(f"     {len(unwired_unique)} files, {total_mb(unwired_unique):.1f} MB")
    print()
    print(f"  2. UNWIRED DUPLICATES (same basename in multiple orphan dirs;")
    print(f"     pick one canonical path, delete others):")
    print(f"     {len(unwired_dupes)} files, {total_mb(unwired_dupes):.1f} MB")
    print()
    print(f"  3. REDUNDANT DUPES OF REFERENCED FILES (safe to delete):")
    print(f"     {len(redundant_dupes)} files, {total_mb(redundant_dupes):.1f} MB")
    print()

    print("  Unwired unique — by source dir:")
    by_dir_unwired: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
    for entry in unwired_unique:
        by_dir_unwired[_norm(str(entry[0].parent))].append(entry)
    for d, items in sorted(by_dir_unwired.items(), key=lambda kv: -len(kv[1])):
        vrma = sum(1 for _, e, _ in items if e == "vrma")
        bvh = sum(1 for _, e, _ in items if e == "bvh")
        mb = sum(sz for _, _, sz in items) / (1024 * 1024)
        print(f"    {d[-55:]:<55} {len(items):>4} files  vrma={vrma:>3} bvh={bvh:>3}  {mb:>5.1f} MB")

    print()
    print("  Sample unwired unique filenames (alphabetical, first 25):")
    for p, ext, sz in sorted(unwired_unique, key=lambda x: x[0].name)[:25]:
        kb = sz / 1024
        print(f"    {ext}  {p.name:<45} {kb:>7.1f} KB  ({_norm(str(p.parent))})")
    if len(unwired_unique) > 25:
        print(f"    ... and {len(unwired_unique) - 25} more")


def emit_plan() -> None:
    """Emit two action artifacts: a deletion script and an adoption list.

    Run with ``--plan`` flag. Writes:
      * .claude/skills/augmentum-dev/scripts/anim_cleanup.sh
        — git-rm commands for buckets 2 + 3 (redundant dupes).
      * .claude/skills/augmentum-dev/scripts/anim_adopt.json
        — list of {path, basename, ext, size_kb} for bucket 1
          (unwired unique files we should add to the atlas).
    """
    files = collect_disk_files()
    refs = collect_refs()
    disk_refs = {_url_to_disk(r) for r in refs}

    referenced: list[tuple[Path, str, int]] = []
    orphan: list[tuple[Path, str, int]] = []
    for p, ext, sz in files:
        pstr = _norm(str(p))
        hit = any(pstr.endswith(ref) or pstr == ref for ref in disk_refs)
        (referenced if hit else orphan).append((p, ext, sz))

    ref_basenames = {p.name for p, _, _ in referenced}
    ref_paths = {_norm(str(p)) for p, _, _ in referenced}

    by_name: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
    for p, ext, sz in orphan:
        by_name[p.name].append((p, ext, sz))

    # Files safe to delete:
    # - Any orphan whose basename matches a referenced file (bucket 3 — pure dupe)
    # - Bucket 2: same basename in multiple orphan dirs — keep ONE canonical
    #   (we pick the path that exists at the most "obviously source" location:
    #    ui/lib/animations/ over poses/, poses/external/sillytavern-pack/animation
    #    over animation_nitral-fork)
    to_delete: list[str] = []
    keep_log: list[str] = []

    def _canonical_score(p: Path) -> tuple[int, str]:
        s = _norm(str(p))
        if "ui/lib/animations/" in s:
            return (0, s)
        if "poses/external/sillytavern-pack/animation/" in s:
            return (1, s)
        if "poses/external/sillytavern-pack/animation_nitral-fork/" in s:
            return (2, s)
        if s.startswith("poses/vrma/"):
            return (3, s)
        if "poses/external/vrm-viewer/" in s:
            return (4, s)
        # plain poses/foo.vrma
        return (5, s)

    for name, group in by_name.items():
        if name in ref_basenames:
            # Bucket 3 — all orphan copies are redundant with the referenced one.
            for p, _, _ in group:
                to_delete.append(_norm(str(p)))
        elif len(group) > 1:
            # Bucket 2 — keep canonical, delete the rest.
            sorted_group = sorted(group, key=lambda x: _canonical_score(x[0]))
            keep = sorted_group[0]
            keep_log.append(f"keep    {_norm(str(keep[0]))}")
            for p, _, _ in sorted_group[1:]:
                to_delete.append(_norm(str(p)))

    # Bucket 1 — unwired unique
    adopt: list[dict] = []
    for name, group in by_name.items():
        if name in ref_basenames or len(group) > 1:
            continue
        p, ext, sz = group[0]
        adopt.append({
            "path": _norm(str(p)),
            "basename": p.name,
            "ext": ext,
            "size_kb": round(sz / 1024, 1),
        })
    adopt.sort(key=lambda r: r["path"])

    # Emit deletion script
    out_dir = Path(__file__).resolve().parent
    sh = out_dir / "anim_cleanup.sh"
    with sh.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Auto-generated by anim_inventory.py --plan\n")
        f.write(f"# {len(to_delete)} files to delete.\n")
        f.write("# These are bytewise duplicates of files the atlas already references\n")
        f.write("# (bucket 3) or unwired duplicates where a canonical copy is kept\n")
        f.write("# (bucket 2). Review the list before running.\n")
        f.write("set -euo pipefail\n\n")
        for k in keep_log:
            f.write(f"# {k}\n")
        f.write("\n")
        for path in sorted(to_delete):
            f.write(f"git rm \"{path}\"\n")
    sh.chmod(0o755)

    # Emit adoption JSON
    import json
    adopt_json = out_dir / "anim_adopt.json"
    adopt_json.write_text(json.dumps(adopt, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {sh} — {len(to_delete)} deletions")
    print(f"Wrote {adopt_json} — {len(adopt)} adoption candidates")


if __name__ == "__main__":
    if "--plan" in sys.argv:
        emit_plan()
    else:
        main()
