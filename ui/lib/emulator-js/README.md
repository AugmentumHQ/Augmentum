# EmulatorJS vendor target

This directory holds the **EmulatorJS** distribution -- a libretro-core
WASM bundle that powers the AXF `emulator-browser` runtime. Multi-MB
binaries don't belong in the repo, so we ship a vendor script
(`scripts/vendor_emulator_js.sh`) that fetches a pinned release.

## How to install

```bash
# From the repo root:
bash scripts/vendor_emulator_js.sh

# To pin a specific version:
EJS_VERSION=4.2.3 bash scripts/vendor_emulator_js.sh
```

After running, this directory contains:

```
ui/lib/emulator-js/
├── loader.js       # main entry point
├── data/           # cores (WASM per system), shaders, BIOS stubs
├── VERSION         # pinned version string
└── README.md       # this file
```

## How the runtime uses it

`augmentum/titles/runtimes.py:EmulatorBrowserRuntime` produces a
`LaunchHandle` whose `metadata.emulator_js_path` points here
(`/ui/lib/emulator-js/`). The frontend stage (Phase E1, not yet
shipped) mounts EmulatorJS by:

1. Inserting `<script src="/ui/lib/emulator-js/loader.js">` into the
   player iframe document
2. Setting `EJS_player`, `EJS_core`, `EJS_gameUrl` from the launch
   handle's metadata config
3. Wiring the EmulatorJS save hooks (`EJS_emulator.save_state()`,
   `EJS_emulator.load_state()`, SRAM auto-save events) to the
   `/api/titles/{id}/saves/*` endpoints via a small bridge module

## When to bump

Bump the pin **deliberately** and run the emulator runtime tests
(`pytest tests/test_emulator_runtime.py`) after. EmulatorJS releases
occasionally rename core ids or shift the loader path -- the runtime
adapter has version-aware path logic; tests catch drift.

The audit script (`audit.py`) will surface a stale-vendor warning if
this directory is missing the `VERSION` file. Re-run the vendor
script to fix.

## Why we vendor (vs CDN)

Augmentum is local-first. CDN dependencies break the offline story
and create a privacy regression (every emulator launch leaks the
user's IP to a CDN). Vendoring keeps everything inside the user's
machine.

## License

EmulatorJS is GPL-2.0 (cores) + permissive licensing on the wrapper.
The ROM bytes are user-supplied; we never bundle copyrighted ROMs.
The vendor script downloads only the engine + cores; users provide
their own legally-obtained game files via the upload-rom endpoint.
