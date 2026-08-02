#!/usr/bin/env bash
# Regenerate ui/lib/VENDORED.sha256 from the vendored library entry files
# tracked in ui/lib/VENDORED.md. Run after bumping any vendored library.
#
# Usage:
#   scripts/regen_vendored_hashes.sh
#
# Files tracked here MUST match the "Entry file" lines in VENDORED.md —
# the two are coordinates of the same thing (markdown doc + machine-
# verifiable manifest). If you add a vendored library, add it to BOTH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/ui/lib"

# Keep this list aligned with VENDORED.md's "Entry file" lines.
ENTRY_FILES=(
    "dompurify/purify.min.js"
    "mermaid/mermaid.min.js"
    "hls.js/hls.min.js"
    "highlight.js/highlight.min.js"
    "pdfjs/pdf.min.mjs"
    "pdf-lib/pdf-lib.min.js"
    "three-vrm/three-vrm.module.min.js"
    "silero-vad/bundle.min.js"
    "emulator-js/data/loader.js"
)

OUT="VENDORED.sha256"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

for f in "${ENTRY_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f — skipping" >&2
        continue
    fi
    sha256sum "$f" >> "$TMP"
done

mv "$TMP" "$OUT"
echo "Wrote $OUT ($(wc -l < "$OUT") entries)"
echo
echo "Verify with:  ( cd ui/lib && sha256sum -c VENDORED.sha256 )"
