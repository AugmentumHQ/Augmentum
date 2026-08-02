#!/usr/bin/env bash
# scripts/vendor_emulator_js.sh
#
# Fetch a pinned release of EmulatorJS into ui/lib/emulator-js/.
#
# Why a script: the bundle is multi-MB (cores are WASM compiled per
# system) and shouldn't live in the repo. This script downloads from
# the upstream release tarball, verifies the version, and lays out
# the files where the AXF emulator-browser runtime expects them.
#
# Pinned versions: bump the EJS_VERSION below deliberately and run
# `pytest tests/test_emulator_runtime.py -v` after to catch any path
# mismatches with the runtime adapter.
#
# Usage:
#     bash scripts/vendor_emulator_js.sh
# or, to pin a specific version:
#     EJS_VERSION=4.2.3 bash scripts/vendor_emulator_js.sh
#
# Result:
#     ui/lib/emulator-js/
#       loader.js
#       data/                  (cores, bios stubs, etc.)
#       VERSION

set -euo pipefail

EJS_VERSION="${EJS_VERSION:-4.2.3}"
TARGET_DIR="ui/lib/emulator-js"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Vendoring EmulatorJS ${EJS_VERSION} into ${TARGET_DIR}/"

# EmulatorJS upstream: https://github.com/EmulatorJS/EmulatorJS
# Release tarballs include the prebuilt loader + cores. We fetch the
# tarball rather than cloning so .git/ doesn't bloat the working tree.
URL="https://github.com/EmulatorJS/EmulatorJS/archive/refs/tags/v${EJS_VERSION}.tar.gz"

echo "  Downloading ${URL}..."
if command -v curl >/dev/null 2>&1; then
  curl -sSL -o "${TMP_DIR}/ejs.tar.gz" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${TMP_DIR}/ejs.tar.gz" "${URL}"
else
  echo "ERROR: need curl or wget on PATH" >&2
  exit 1
fi

echo "  Extracting..."
tar -xzf "${TMP_DIR}/ejs.tar.gz" -C "${TMP_DIR}"

# The release archive top-level dir is EmulatorJS-${VERSION}/.
SRC_DIR="${TMP_DIR}/EmulatorJS-${EJS_VERSION}"
if [ ! -d "${SRC_DIR}" ]; then
  echo "ERROR: extracted dir ${SRC_DIR} missing -- archive layout changed?" >&2
  exit 1
fi

# Lay out the bundle. EmulatorJS' standard distribution puts loader.js
# at the top level + a data/ subdir for the cores. We mirror that so
# the runtime adapter's emulator_js_path config Just Works.
echo "  Installing into ${TARGET_DIR}/..."
mkdir -p "${TARGET_DIR}"

# Strip prior install (preserves README.md if you've added one).
find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 \
     ! -name "README.md" -exec rm -rf {} +

# Copy the data/ dir (cores + supporting files) and loader.js.
if [ -d "${SRC_DIR}/data" ]; then
  cp -r "${SRC_DIR}/data" "${TARGET_DIR}/"
fi
if [ -f "${SRC_DIR}/data/loader.js" ]; then
  cp "${SRC_DIR}/data/loader.js" "${TARGET_DIR}/loader.js"
elif [ -f "${SRC_DIR}/loader.js" ]; then
  cp "${SRC_DIR}/loader.js" "${TARGET_DIR}/loader.js"
else
  echo "WARNING: loader.js not found in archive layout -- check upstream" >&2
fi

# Stamp the version so the audit script can verify what's installed.
echo "${EJS_VERSION}" > "${TARGET_DIR}/VERSION"

# ── Custom-built core overlay (durable across re-vendoring) ──────────
# The four mGBA cores are rebuilt from the EmulatorJS/RetroArch source
# with a memory-map export (get_memory_map / EmulatorJSGetMemoryMap) that
# upstream release tarballs lack — it's what lets the game agent read
# EWRAM (and every bus region) by absolute address. Re-vendoring from the
# upstream tarball above wipes data/cores, so we overlay the custom builds
# back on top from a checked-in location. Rebuild recipe + provenance live
# in ${TARGET_DIR}/data/cores/NOTES.md.
CUSTOM_CORES="scripts/emulator-js-cores-custom"
if [ -d "${CUSTOM_CORES}" ] && ls "${CUSTOM_CORES}"/*.data >/dev/null 2>&1; then
  echo "  Overlaying custom-built mGBA cores from ${CUSTOM_CORES}/ ..."
  mkdir -p "${TARGET_DIR}/data/cores"
  cp -f "${CUSTOM_CORES}"/*.data "${TARGET_DIR}/data/cores/"
  [ -f "${CUSTOM_CORES}/NOTES.md" ] && cp -f "${CUSTOM_CORES}/NOTES.md" "${TARGET_DIR}/data/cores/"
  echo "  Overlaid: $(ls "${CUSTOM_CORES}"/*.data | wc -l) core file(s)."
else
  echo "  (No custom cores at ${CUSTOM_CORES}/ — using upstream cores as-is.)"
fi

echo "Vendored EmulatorJS ${EJS_VERSION} into ${TARGET_DIR}/"
echo "  Wired-up files:"
ls -la "${TARGET_DIR}" | sed 's/^/    /'
echo ""
echo "Next: enable in Settings ('emulator_browser_enabled') or via:"
echo "    curl -X PUT http://localhost:6100/api/config/tools \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"emulator_browser_enabled\": true}'"
