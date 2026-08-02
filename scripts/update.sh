#!/usr/bin/env bash
# Update THIS node's Augmentum to the latest code and restart.
#
# Why this is cheap: the augmentum container runs a prebuilt image with the
# host repo overlaid onto /app at boot, so CODE changes apply on a plain
# `restart` — no rebuild. Gitignored per-node config (.env, .augmentum.conf,
# data/) is never touched by `git pull`, so each node keeps its own setup.
#
# Run manually on a node, from cron (set-and-forget), or via deploy-nodes.sh.
#
# Config (env, all optional):
#   AUGMENTUM_UPDATE_REMOTE   git remote to pull from   (default: main)
#   AUGMENTUM_UPDATE_BRANCH   branch to pull            (default: current)
#   AUGMENTUM_REGISTRY        LAN registry host:port to pull prebuilt images
#                             from on an image-relevant change (else .env)
#   AUGMENTUM_AUTO_IMAGE_PULL set to 0 to disable auto image pull+recreate
#                             (default: enabled when AUGMENTUM_REGISTRY is set)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

REMOTE="${AUGMENTUM_UPDATE_REMOTE:-main}"
BRANCH="${AUGMENTUM_UPDATE_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "[update] no git remote '$REMOTE'. Set one up once, e.g.:" >&2
  echo "         git remote add $REMOTE ssh://you@MAIN-HOST$REPO_DIR" >&2
  echo "         (or git://MAIN-HOST/augmentum if the main runs serve-git.sh)" >&2
  exit 1
fi

before="$(git rev-parse HEAD)"
echo "[update] pulling $REMOTE/$BRANCH ..."
if ! git pull --ff-only "$REMOTE" "$BRANCH"; then
  echo "[update] non-fast-forward — this node has diverged from $REMOTE/$BRANCH." >&2
  echo "         Resolve manually (the node shouldn't carry local commits)." >&2
  exit 1
fi
after="$(git rev-parse HEAD)"

if [ "$before" = "$after" ]; then
  echo "[update] already up to date ($(git rev-parse --short HEAD)). No restart."
  exit 0
fi

echo "[update] $(git rev-parse --short "$before") -> $(git rev-parse --short "$after")"
git --no-pager log --oneline "$before..$after" | sed 's/^/    /'

# Image-relevant changes (deps/Dockerfile/binary/sidecar) mean a plain restart
# would run the new code against the OLD image (stale deps/binary).
image_relevant=0
if git --no-pager diff --name-only "$before" "$after" \
     | grep -qE '^(Dockerfile|.*requirements.*|pyproject\.toml|uv\.lock|requirements\.lock|services/|LLAMA_SERVER_VERSION)'; then
  image_relevant=1
fi

# Resolve a LAN image registry (Home Main) — shell env first, then .env (which
# docker compose also reads for the compose.dev-bind.yaml image: substitution).
AUG_REGISTRY="${AUGMENTUM_REGISTRY:-}"
if [ -z "$AUG_REGISTRY" ] && [ -f "$REPO_DIR/.env" ]; then
  AUG_REGISTRY="$(grep -E '^AUGMENTUM_REGISTRY=' "$REPO_DIR/.env" | tail -1 | cut -d= -f2- | tr -d '\r')"
fi

# Restart/recreate only augmentum, honoring this node's own compose overlay list.
COMPOSE_FLAGS=""
if [ -f "$REPO_DIR/.augmentum.conf" ]; then
  for f in $(cat "$REPO_DIR/.augmentum.conf"); do COMPOSE_FLAGS="$COMPOSE_FLAGS -f $f"; done
fi

if [ "$image_relevant" = 1 ] && [ -n "$AUG_REGISTRY" ] && [ "${AUGMENTUM_AUTO_IMAGE_PULL:-1}" != 0 ]; then
  # Image-relevant change + a configured LAN registry: pull the prebuilt image
  # from the Home Main (the trusted source on the home LAN) and RECREATE.
  # NOTE: recreate kills any resident llama-server model slot mid-generation —
  # acceptable here because the image genuinely changed. On an untrusted network
  # front the registry with TLS before relying on this unattended path.
  echo "[update] image-relevant change — pulling augmentum from $AUG_REGISTRY ..."
  docker compose $COMPOSE_FLAGS pull augmentum
  echo "[update] recreating augmentum ..."
  docker compose $COMPOSE_FLAGS up -d --no-deps augmentum
elif [ "$image_relevant" = 1 ]; then
  # Image-relevant but no registry/auto-pull configured: a restart alone runs
  # stale. Apply code-only parts but tell the operator how to refresh the image.
  echo "[update] NOTE: image-relevant files changed (deps/Dockerfile/binary/sidecar)." >&2
  echo "         Restart alone may run stale. On the Home Main run ./scripts/publish-images.sh," >&2
  echo "         set AUGMENTUM_REGISTRY=<main-host>:5000 in this node's .env, and re-run —" >&2
  echo "         or rebuild here with ./start.sh build." >&2
  echo "[update] restarting augmentum (code changes apply; image stays as-is) ..."
  docker compose $COMPOSE_FLAGS restart augmentum
else
  echo "[update] restarting augmentum ..."
  docker compose $COMPOSE_FLAGS restart augmentum
fi
echo "[update] done — now at $(git rev-parse --short HEAD)."
