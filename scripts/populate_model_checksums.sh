#!/usr/bin/env bash
# Populate MODEL_CHECKSUMS.txt with the current sha256 of every pinned
# URL. Streams each URL through sha256sum so nothing lands on disk —
# safe to run on a build host without the model directories writable.
#
# Use case: you cleared the <UNPINNED> placeholders or upstream
# published a new patch under the same URL; re-run this and review the
# diff before committing.
#
# Usage:
#   scripts/populate_model_checksums.sh           # rewrite in place
#   scripts/populate_model_checksums.sh --dry-run # print, don't rewrite
#
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/MODEL_CHECKSUMS.txt"

if [ ! -f "$SRC" ]; then
    echo "MODEL_CHECKSUMS.txt missing at repo root" >&2
    exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# Pass-through every line that isn't a checksum entry. For checksum
# entries (any line where field 1 is a sha256 or a placeholder), fetch
# the URL and re-compute the hash.
while IFS= read -r line; do
    # Skip blanks + comments + section headers verbatim.
    if [ -z "${line// }" ] || [[ "$line" =~ ^# ]]; then
        printf '%s\n' "$line" >> "$TMP"
        continue
    fi

    # Parse: <hash>  <file>  <url>
    read -r hash file url <<< "$line"
    if [ -z "$url" ]; then
        # Not a checksum line — pass through.
        printf '%s\n' "$line" >> "$TMP"
        continue
    fi

    if [ "$hash" = "<FLOATING>" ]; then
        echo "  floating  $file  (kept unpinned by policy)"
        printf '%s\n' "$line" >> "$TMP"
        continue
    fi

    echo "  hashing   $file  ←  $url"
    new_hash="$(curl -fsSL --retry 3 "$url" | sha256sum | awk '{print $1}')"
    if [ -z "$new_hash" ]; then
        echo "    WARN: hash unavailable — leaving as <UNPINNED>" >&2
        printf '<UNPINNED>  %s  %s\n' "$file" "$url" >> "$TMP"
        continue
    fi
    printf '%s  %s  %s\n' "$new_hash" "$file" "$url" >> "$TMP"
done < "$SRC"

if [ "$DRY_RUN" -eq 1 ]; then
    diff -u "$SRC" "$TMP" || true
    echo
    echo "Dry-run complete. No file written."
else
    mv "$TMP" "$SRC"
    echo
    echo "Updated MODEL_CHECKSUMS.txt. Diff:"
    git -C "$REPO_ROOT" diff -- MODEL_CHECKSUMS.txt | head -40 || true
fi
