#!/usr/bin/env bash
# Publish the augmentum app + llama-server images to the Home Main's LAN
# registry so fabric NODES pull prebuilt images instead of compiling. Run on
# the MAIN after building/pulling images (e.g. after ./start.sh build or an
# upgrade_llama_server.sh bump). LAN analog of .github/workflows/build-images.yml.
#
# Prerequisite: the registry service is up (compose.registry.yaml in the main's
# .augmentum.conf) and reachable at $AUGMENTUM_REGISTRY (default localhost:5000).
#
# MUST run on the main BEFORE deploy-nodes.sh fans out — a node that git-pulls a
# commit whose image hasn't been published 404s the registry pull.
#
# Config (env, all optional):
#   AUGMENTUM_REGISTRY   registry host:port   (default: localhost:5000)
#   AUGMENTUM_VARIANT    cpu | gpu            (default: read from .env, else cpu)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pull a value from .env (gitignored per-node config) if not already in the env.
env_get() { [ -f "$REPO_DIR/.env" ] && grep -E "^$1=" "$REPO_DIR/.env" | tail -1 | cut -d= -f2- | tr -d '\r' || true; }

REGISTRY="${AUGMENTUM_REGISTRY:-$(env_get AUGMENTUM_REGISTRY)}"
REGISTRY="${REGISTRY:-localhost:5000}"
VARIANT="${AUGMENTUM_VARIANT:-$(env_get AUGMENTUM_VARIANT)}"
VARIANT="${VARIANT:-cpu}"

LLAMA_VER=""
[ -f "$REPO_DIR/LLAMA_SERVER_VERSION" ] && LLAMA_VER="$(tr -d '[:space:]' < "$REPO_DIR/LLAMA_SERVER_VERSION")"

AB_VER=""
[ -f "$REPO_DIR/AGENT_BROWSER_VERSION" ] && AB_VER="$(tr -d '[:space:]' < "$REPO_DIR/AGENT_BROWSER_VERSION")"

echo "[publish] registry=$REGISTRY  variant=$VARIANT  llama=${LLAMA_VER:-<none>}  agent-browser=${AB_VER:-<none>}"

# $1 = local source image ref, $2 = target repo:tag (without registry prefix).
push_one() {
  local src="$1" dst="$REGISTRY/$2"
  if ! docker image inspect "$src" >/dev/null 2>&1; then
    echo "  [skip] $src not present locally — build/pull it on the main first." >&2
    return 0
  fi
  echo "  tag  $src -> $dst"
  docker tag "$src" "$dst"
  echo "  push $dst"
  docker push "$dst"
}

# The app image nodes pull (compose.dev-bind.yaml). A local dev build and a GHCR
# pull both end up tagged ghcr.io/augmentumhq/augmentum:${VARIANT}-latest.
push_one "ghcr.io/augmentumhq/augmentum:${VARIANT}-latest" "augmentumhq/augmentum:${VARIANT}-latest"

# The llama-server binary image (build stage). Only nodes that still BUILD need
# it; dev-bind nodes get the binary baked into the app image above. Published so
# start.sh's registry leg can grab it instead of compiling.
if [ -n "$LLAMA_VER" ]; then
  push_one "augmentum-llama-server:latest" "augmentumhq/augmentum-llama-server:${LLAMA_VER}"
fi

# Sidecar images (code executor + agent-browser). Published so nodes running
# those overlays pull prebuilt instead of compiling sci-Python wheels /
# Chromium. Sources are the GHCR-tagged refs (present locally after a GHCR
# pull or a compose build tagged to match); push_one skips any not present.
push_one "ghcr.io/augmentumhq/augmentum-executor:latest" "augmentumhq/augmentum-executor:latest"
push_one "ghcr.io/augmentumhq/augmentum-agent-browser:latest" "augmentumhq/augmentum-agent-browser:latest"
if [ -n "$AB_VER" ]; then
  push_one "ghcr.io/augmentumhq/augmentum-agent-browser:${AB_VER}" "augmentumhq/augmentum-agent-browser:${AB_VER}"
fi

echo "[publish] done. On each node set AUGMENTUM_REGISTRY=$REGISTRY in .env and"
echo "          add it to /etc/docker/daemon.json insecure-registries (one-time)."
