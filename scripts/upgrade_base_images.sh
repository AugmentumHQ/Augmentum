#!/usr/bin/env bash
# Bump every pinned base-image digest in the Dockerfiles to whatever
# the upstream tag currently resolves to. Auditable: prints the
# old → new sha256 transition for each image before rewriting, so a
# diff review can see exactly what changed.
#
# Why pin to digests at all: a registry compromise (or an intentional
# upstream re-publish under the same tag) can swap the bytes behind
# `python:3.11-slim-bookworm` without changing the version string.
# Pinning to a sha256 makes that swap fail loudly instead of silently
# bringing in new code. The tradeoff: we have to manually rotate
# digests when we want the upstream's latest patch level. That's
# what this script does — once a month or whenever upstream pushes
# a security update we care about.
#
# Requires docker buildx (ships with modern Docker Desktop and modern
# `docker-ce` distributions).
#
# Usage:
#   scripts/upgrade_base_images.sh            # bump in place
#   scripts/upgrade_base_images.sh --dry-run  # print transitions, don't touch files
#
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Each entry is "tag|file1[ file2 ...]". The script looks up the current
# manifest-list digest for <tag> and replaces every occurrence of
# ``<tag>@sha256:<old>`` (or bare ``<tag>``) with ``<tag>@sha256:<new>``
# inside the listed files. Add new pins by appending entries here.
PINS=(
    "python:3.11-slim-bookworm|Dockerfile"
    "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04|Dockerfile.gpu"
    "nvidia/cuda:12.8.1-devel-ubuntu22.04|Dockerfile.llama-server"
    "nvidia/cuda:12.8.1-base-ubuntu22.04|Dockerfile.llama-server"
    "ubuntu:22.04|Dockerfile.llama-server-cpu"
    "ubuntu:24.04|Dockerfile.workspace"
)

resolve_digest() {
    local tag="$1"
    docker buildx imagetools inspect "$tag" 2>/dev/null \
        | awk '/^Digest:/ { print $2; exit }'
}

# Portable in-place sed (GNU vs BSD).
sed_inplace() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

echo "Resolving current digests..."
for pin in "${PINS[@]}"; do
    tag="${pin%%|*}"
    files="${pin#*|}"

    new_digest="$(resolve_digest "$tag")"
    if [ -z "$new_digest" ]; then
        echo "  WARN: could not resolve $tag — skipping" >&2
        continue
    fi

    for f in $files; do
        path="$REPO_ROOT/$f"
        if [ ! -f "$path" ]; then
            echo "  WARN: $f missing — skipping" >&2
            continue
        fi

        # Capture current value (if any) for the diff log.
        old_line="$(grep -nE "^FROM ${tag//\//\\/}(@sha256:[a-f0-9]+)?" "$path" || true)"
        old_digest="$(printf '%s\n' "$old_line" | grep -oE 'sha256:[a-f0-9]+' | head -1 || true)"

        if [ "$old_digest" = "$new_digest" ]; then
            echo "  unchanged  $tag  in $f"
            continue
        fi

        echo "  rotating   $tag  in $f"
        echo "             from ${old_digest:-<unpinned>}"
        echo "             to   $new_digest"

        if [ "$DRY_RUN" -eq 1 ]; then
            continue
        fi

        # Two patterns to rewrite: existing pin (tag@sha256:old) and
        # bare tag (no digest yet). Order matters — rewrite the pinned
        # form first so the bare-form regex can't accidentally double-
        # apply.
        if [ -n "$old_digest" ]; then
            sed_inplace "s|${tag}@${old_digest}|${tag}@${new_digest}|g" "$path"
        else
            # Match ``FROM <tag>`` at start-of-line, optionally followed
            # by `` AS <stage>``, and insert the @digest before the AS.
            sed_inplace -E "s|^(FROM ${tag//\//\\/})( AS [A-Za-z0-9_-]+)?\$|\\1@${new_digest}\\2|" "$path"
        fi
    done
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "Dry-run complete. No files modified."
else
    echo
    echo "Done. Review the diff with: git diff Dockerfile*"
fi
