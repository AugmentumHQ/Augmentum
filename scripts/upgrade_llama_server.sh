#!/usr/bin/env bash
#
# Upgrade the bundled llama-server binary (Engine v2's inference core).
#
# The pinned version lives in LLAMA_SERVER_VERSION at the repo root so the
# Dockerfile and this script share one source of truth.
#
# Usage:
#   ./scripts/upgrade_llama_server.sh              # use current pin
#   ./scripts/upgrade_llama_server.sh b8839        # pin to specific tag
#   ./scripts/upgrade_llama_server.sh --latest     # fetch latest released tag
#
# After this finishes you'll still need to rebuild the main augmentum image
# so it picks up the new llama-server binary from the builder stage:
#
#   docker compose build augmentum
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${REPO_ROOT}/LLAMA_SERVER_VERSION"
DOCKERFILE="${REPO_ROOT}/Dockerfile.llama-server"

target="${1:-}"

if [[ "$target" == "--latest" ]]; then
    echo "Fetching latest llama.cpp tag from GitHub…"
    target="$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
        | grep -oE '"tag_name":\s*"b[0-9]+"' | head -1 | grep -oE 'b[0-9]+')"
    if [[ -z "$target" ]]; then
        echo "ERROR: could not determine latest tag from GitHub API" >&2
        exit 1
    fi
    echo "Latest release tag: $target"
elif [[ -z "$target" ]]; then
    target="$(tr -d '\n' < "$VERSION_FILE")"
    echo "Using currently-pinned version: $target"
fi

# Validate tag shape — llama.cpp uses bNNNN style.
if [[ ! "$target" =~ ^b[0-9]+$ ]]; then
    echo "ERROR: tag '$target' does not look like a llama.cpp release tag (bNNNN)" >&2
    exit 1
fi

# Persist pin — so git tracks the upgrade even if the user forgets to commit.
echo "$target" > "$VERSION_FILE"

echo ""
echo "Building augmentum-llama-server:$target from Dockerfile.llama-server…"
echo "  (this is the long step — expect ~10-25 min on first build)"
echo ""

docker build \
    --build-arg "LLAMA_CPP_VERSION=$target" \
    -t "augmentum-llama-server:$target" \
    -t "augmentum-llama-server:latest" \
    -f "$DOCKERFILE" \
    "$REPO_ROOT"

echo ""
echo "✓ augmentum-llama-server:$target built and tagged :latest"

# ── Behavior-defaults diff guard ─────────────────────────────────────────
# Upstream changes FLAG DEFAULTS between releases, silently changing server
# behavior under an identical Augmentum config. Real incident: b9181
# flipped cache_idle_slots to default-ON, which cleared idle slots on
# every task launch, killed LCP slot routing entirely, and cost every
# narrative turn a full re-prefill (12-15 min at 61k tokens) — invisible
# until a live trace on 2026-07-02. Snapshot the new binary's --help and
# diff against the previous snapshot so default changes surface AT BUMP
# TIME, not months later in production.
DEFAULTS_DIR="${REPO_ROOT}/docs/llama-server-defaults"
mkdir -p "$DEFAULTS_DIR"
new_snapshot="${DEFAULTS_DIR}/help-${target}.txt"
docker run --rm --entrypoint llama-server "augmentum-llama-server:$target" \
    --help > "$new_snapshot" 2>&1 || \
    echo "WARN: could not capture --help from the new image (non-fatal)"
prev_snapshot="$(ls -1 "$DEFAULTS_DIR"/help-b*.txt 2>/dev/null | sort -V | grep -v "help-${target}.txt" | tail -1 || true)"
if [[ -n "$prev_snapshot" && -s "$new_snapshot" ]]; then
    echo ""
    echo "── llama-server flag/default changes since $(basename "$prev_snapshot" | sed 's/help-//;s/.txt//') ──"
    if diff -u "$prev_snapshot" "$new_snapshot" | grep -E '^[+-]' | grep -viE '^\+\+\+|^---' | grep -iE 'default|^[+-]\s*-' | head -60; then
        echo ""
        echo "REVIEW THE LINES ABOVE — especially 'default:' changes. A flipped"
        echo "default is a behavior change Augmentum inherits without any code diff."
    else
        echo "(no flag or default changes detected)"
    fi
fi

echo ""
echo "Next:  docker compose build augmentum && docker compose up -d augmentum"
echo "Verify: docker exec augmentum-augmentum-1 llama-server --version | head -3"
