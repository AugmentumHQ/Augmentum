"""Re-classify emulator ROM artifacts whose system_id was misdetected
at upload time.

Why this exists: ``rom_systems.detect_system`` falls back to extension
matching when no header rule matches. ``.iso`` files maps first to 3DO
in the catalog, so a GameCube or Wii .iso uploaded before the
content-sniffing magic-byte rules were added landed under 3do (libretro
core ``opera``). Clicking such an artifact would launch the wrong
emulator.

This script walks every emulator_rom artifact, reads its first 64
bytes from the blob_store, and re-runs detection. If the detected
system differs from the stored metadata, it patches the artifact's
metadata.system_id / system_label / libretro_core / bios_required
in place.

Run inside the augmentum container:

    docker exec augmentum-augmentum-1 \
        python -m scripts.reclassify_rom_systems --dry-run
    docker exec augmentum-augmentum-1 \
        python -m scripts.reclassify_rom_systems        # actually apply

Idempotent: a second run is a no-op once everything's correctly tagged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiosqlite

from augmentum.titles.rom_systems import detect_system, get_system

DATA_DIR = Path("/data")
# Live db filename varies between deploys; probe the standard candidates.
_DB_CANDIDATES = ("augmentum.db", "augmentum.sqlite", "state.sqlite")
DB_PATH = next(
    (DATA_DIR / n for n in _DB_CANDIDATES if (DATA_DIR / n).exists()),
    DATA_DIR / "augmentum.db",
)
BLOB_ROOT = DATA_DIR / "blobs"


def blob_path_for(sha: str) -> Path:
    """Mirror VfsBlobStore.path_for() — content-addressed sharded layout."""
    return BLOB_ROOT / sha[:2] / sha[2:4] / sha


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would change without writing.",
    )
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: state db not found at {DB_PATH}", file=sys.stderr)
        return 1

    fixed = 0
    skipped_missing_blob = 0
    skipped_no_change = 0
    failed = 0

    async with aiosqlite.connect(DB_PATH) as conn:
        cur = await conn.execute(
            "SELECT id, metadata FROM artifacts "
            "WHERE json_extract(metadata, '$.kind') = 'emulator_rom'"
        )
        rows = await cur.fetchall()
        await cur.close()

        print(f"Scanning {len(rows)} emulator_rom artifacts...")
        for aid, meta_raw in rows:
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, (str, bytes)) else (meta_raw or {})
            except Exception:
                meta = {}
            current_sys = str(meta.get("system_id") or "")
            sha = str(meta.get("rom_sha256") or "")
            filename = str(meta.get("original_filename") or "")
            if not sha:
                print(f"  SKIP {aid[:16]}  no rom_sha256 (filename={filename!r})")
                failed += 1
                continue

            blob = blob_path_for(sha)
            if not blob.exists():
                print(f"  SKIP {aid[:16]}  blob missing ({blob})")
                skipped_missing_blob += 1
                continue

            with blob.open("rb") as fh:
                header = fh.read(64)
            spec = detect_system(filename, header=header)
            if spec is None:
                print(
                    f"  SKIP {aid[:16]}  detector returned no match "
                    f"(current system={current_sys!r}, file={filename!r})"
                )
                skipped_no_change += 1
                continue

            if spec.id == current_sys:
                skipped_no_change += 1
                continue

            current_spec = get_system(current_sys)
            current_label = (
                current_spec.label if current_spec is not None else "?"
            )
            print(
                f"  FIX  {aid[:16]}  {current_sys}/{current_label!r} "
                f"-> {spec.id}/{spec.label!r}  (file={filename!r})"
            )

            if not args.dry_run:
                meta["system_id"] = spec.id
                meta["system_label"] = spec.label
                meta["libretro_core"] = spec.libretro_core
                meta["bios_required"] = bool(spec.bios_required)
                # Also clear runtime_preferred if it was baked in;
                # the resolver will fall through to the right runtime
                # via supports() once the system is correct.
                # (We leave runtime_alternates alone — it's still
                # ("emulator-browser", "agsp-streamed") which covers
                # both possibilities.)
                await conn.execute(
                    "UPDATE artifacts SET metadata = ? WHERE id = ?",
                    (json.dumps(meta), aid),
                )
                fixed += 1
            else:
                fixed += 1  # would-be-fixed count for dry-run summary

        if not args.dry_run:
            await conn.commit()

    print()
    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"{verb}: {fixed}")
    print(f"Skipped (no change needed): {skipped_no_change}")
    print(f"Skipped (blob missing on disk): {skipped_missing_blob}")
    print(f"Skipped (no metadata.rom_sha256): {failed}")
    if args.dry_run:
        print()
        print("Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
