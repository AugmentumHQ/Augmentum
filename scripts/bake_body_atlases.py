#!/usr/bin/env python3
"""
bake_body_atlases.py — batch-bake body atlases for bundled VRMs.

Drives `ui/mockups/body-atlas-generator.html` in headless Chromium via
Playwright. Outputs `poses/body-atlas-<slug>.json` per VRM.

One-time setup:
    pip install playwright
    playwright install chromium

Usage:
    python scripts/bake_body_atlases.py                 # bake everything missing
    python scripts/bake_body_atlases.py --rebake        # force-rebake all
    python scripts/bake_body_atlases.py --vrm Lise.vrm  # bake one
    python scripts/bake_body_atlases.py --show-browser  # watch it run
    python scripts/bake_body_atlases.py --voxel 1.0     # voxel size (cm); default 1.0
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_DIR = REPO_ROOT / "ui" / "lib" / "bundled-avatars"
POSES_DIR = REPO_ROOT / "poses"
GENERATOR_PATH = "/ui/mockups/body-atlas-generator.html"

# Hardcoded bundled-avatar roster (the VRMs that ship in-tree, authored
# in VRoid Studio). Becca is intentionally omitted — already baked at
# poses/body-atlas-becca.json. Add new entries here when a new bundled
# VRM lands.
BUNDLED_VRM_FILES: list[str] = [
    "Danny.vrm",
    "Lise.vrm",
    "Louis.vrm",
    "Luna.vrm",
    "Roxanne.vrm",
    "vance.vrm",
]


def find_vrms() -> list[Path]:
    paths: list[Path] = []
    for name in BUNDLED_VRM_FILES:
        p = BUNDLED_DIR / name
        if not p.is_file():
            print(f"[warn] missing bundled VRM: {p}", file=sys.stderr)
            continue
        paths.append(p)
    return paths


def output_path(vrm: Path) -> Path:
    # Match existing convention: body-atlas-becca.json (lowercase slug).
    return POSES_DIR / f"body-atlas-{vrm.stem.lower()}.json"


def start_static_server(root: Path) -> tuple[socketserver.TCPServer, int]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, fmt, *args):
            return  # quiet

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def bake_one(page, vrm_path: Path, out_path: Path, voxel_cm: float) -> None:
    print(f"[{vrm_path.name}] loading VRM…", flush=True)
    page.set_input_files("#vrmPicker", str(vrm_path))
    # genBtn enables once the VRM is parsed and the scene is updated.
    page.wait_for_function(
        "() => !document.getElementById('genBtn').disabled",
        timeout=60_000,
    )
    page.evaluate(
        f"document.getElementById('voxelSize').value = {voxel_cm}"
    )

    print(f"[{vrm_path.name}] baking at {voxel_cm}cm voxels…", flush=True)
    t0 = time.time()
    # Generator's headless API does the full bake when awaited.
    # Return `true` (not the atlas) so we don't shovel 50MB through CDP.
    page.evaluate(
        """
        async () => {
          const opts = {
            voxelSize: parseFloat(document.getElementById('voxelSize').value) / 100,
            activeBand: parseFloat(document.getElementById('activeBand').value) / 100,
            fwnBeta: parseFloat(document.getElementById('fwnBeta').value),
          };
          await window.augmentumAtlas.generate(opts);
          return true;
        }
        """
    )
    # Stream JSON out as a string (faster than letting CDP serialize the object).
    atlas_json = page.evaluate(
        "() => JSON.stringify(window.augmentumAtlas.getAtlas())"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(atlas_json, encoding="utf-8")
    # Precompressed sibling for the static server: the live avatar pipeline
    # fetches these 50-60 MB atlases on every surface that mounts a VRM, so the
    # /poses mount serves <name>.json.gz with Content-Encoding: gzip when present
    # (~4x smaller wire transfer, zero data change). gzip -9; mtime kept aligned
    # so the gz never looks staler than its source on a re-bake.
    import gzip as _gzip
    gz_path = out_path.with_suffix(out_path.suffix + ".gz")
    gz_path.write_bytes(_gzip.compress(atlas_json.encode("utf-8"), compresslevel=9))
    dt = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    gz_mb = gz_path.stat().st_size / 1e6
    print(
        f"[{vrm_path.name}] ✓ {dt:.1f}s — wrote {out_path.name} "
        f"({size_mb:.1f}MB, +{gz_path.name} {gz_mb:.1f}MB gzip)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--rebake",
        action="store_true",
        help="Re-bake even if the atlas JSON already exists.",
    )
    parser.add_argument(
        "--vrm",
        help="Bake only the named VRM file (e.g. Lise.vrm).",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=1.0,
        help="Voxel size in cm (default 1.0).",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run headed Chromium (default headless).",
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "Playwright not installed. Run:\n"
            "    pip install playwright\n"
            "    playwright install chromium"
        )

    if not BUNDLED_DIR.is_dir():
        sys.exit(f"Bundled VRM directory not found: {BUNDLED_DIR}")

    vrms = find_vrms()
    if args.vrm:
        vrms = [v for v in vrms if v.name == args.vrm]
        if not vrms:
            sys.exit(f"No VRM named {args.vrm!r} in {BUNDLED_DIR}")

    POSES_DIR.mkdir(exist_ok=True)
    work: list[tuple[Path, Path]] = []
    for v in vrms:
        out = output_path(v)
        if out.exists() and not args.rebake:
            print(f"[skip] {v.name} — atlas exists at {out.name}")
            continue
        work.append((v, out))

    if not work:
        print("Nothing to do. Use --rebake to force re-bake.")
        return

    print(f"Baking {len(work)} VRM(s) at {args.voxel}cm voxels.")

    httpd, port = start_static_server(REPO_ROOT)
    url = f"http://127.0.0.1:{port}{GENERATOR_PATH}"
    print(f"Static server on :{port}, generator at {url}")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.show_browser)
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(60_000)

            page.on(
                "console",
                lambda msg: (
                    print(f"  [console.{msg.type}] {msg.text}")
                    if msg.type in ("error", "warning")
                    else None
                ),
            )
            page.on("pageerror", lambda exc: print(f"  [pageerror] {exc}"))

            print("Loading generator…", flush=True)
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.wait_for_selector("#vrmPicker", timeout=30_000)
            # Wait for the headless API to be defined (means the module
            # script finished and the scene is alive).
            page.wait_for_function(
                "() => !!(window.augmentumAtlas && window.augmentumAtlas.isReady)",
                timeout=30_000,
            )

            for vrm, out in work:
                try:
                    bake_one(page, vrm, out, voxel_cm=args.voxel)
                except Exception as exc:
                    print(f"[{vrm.name}] ✗ {exc}", flush=True)
                    # Continue with the rest; one bad VRM shouldn't block the batch.

            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    print("Done.")


if __name__ == "__main__":
    main()
