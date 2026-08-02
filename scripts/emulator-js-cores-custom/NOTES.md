# Custom-built EmulatorJS mGBA cores — full memory-map export

These four `mgba-*.data` cores are rebuilt from `github.com/EmulatorJS/RetroArch`
with a **full libretro memory-map export** that upstream release cores lack. It
is what lets the game agent read GBA **EWRAM** (and every bus region) by absolute
console-bus address — the agnostic memory primitive (see
`augmentum/game_agent/probes/` + `ui/scripts/emulator-bridge.js` resolver).

## What was added (vs upstream RetroArch)
- `retroarch.c`: `char* get_memory_map(void)` — serialises the core's
  `runloop_state_get_ptr()->system.mmaps` descriptor table (published via
  `RETRO_ENVIRONMENT_SET_MEMORY_MAPS`) as `flags,ptr,offset,start,select,
  disconnect,len` rows.
- `emscripten/emulatorjs.js`: `$EmulatorJSGetMemoryMap` glue (parses → array).
- `Makefile.emulatorjs`: `_get_memory_map` in EXPORTED_FUNCTIONS,
  `EmulatorJSGetMemoryMap` in EXPORTED_RUNTIME_METHODS.
- `build.json`: `minimumEJSVersion` patched 4.3.0 → **4.2.2** so the vendored
  4.2.3 loader accepts these cores (the export needs no 4.3.0 loader feature).

## Verified
`EmulatorJSGetMemoryMap()` returns all 11 GBA bus regions incl. EWRAM
(`start=0x02000000, len=262144`); EWRAM reads live (scripts/probe_verify_map.py:
4769 bytes changed / 1.5s during play).

## Rebuild recipe
See `ui/lib/emulator-js/data/cores/NOTES.md` for the Docker/emsdk pipeline.
CRITICAL: patch `ejs-build/compile/RetroArch/` (NOT the top-level clone — the
build uses `buildPath=$PWD/compile`), skip `build_env.sh` on re-runs (it wipes
emsdk), then re-pack build.json (7z) with minimumEJSVersion 4.2.2.

## Durability
`scripts/vendor_emulator_js.sh` overlays these back onto `ui/lib/emulator-js/
data/cores/` after any upstream re-vendor.
