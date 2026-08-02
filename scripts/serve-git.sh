#!/usr/bin/env bash
# Serve this repo read-only over the LAN so fabric nodes can pull from it —
# run on the MAIN. This is the "always-current source" that replaces bundles.
#
# Nodes then add it as a remote (once):
#   git remote add main git://MAIN-HOST/augmentum
#
# Read-only + anonymous; fine for a trusted home LAN. For untrusted networks,
# prefer SSH remotes instead (ssh://you@MAIN-HOST/path/to/augmentum) and skip
# this script. Runs in the foreground; background it or wrap in systemd/tmux.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AUGMENTUM_GIT_PORT:-9418}"

# Allow anonymous export of this one repo.
touch "$REPO_DIR/.git/git-daemon-export-ok"

echo "Serving $REPO_DIR as git://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/augmentum"
echo "On each node:  git remote add main git://MAIN-HOST/augmentum"
exec git daemon \
  --reuseaddr \
  --port="$PORT" \
  --base-path="$(dirname "$REPO_DIR")" \
  --export-all \
  "$(dirname "$REPO_DIR")"
