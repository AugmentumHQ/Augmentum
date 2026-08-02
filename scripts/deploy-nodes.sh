#!/usr/bin/env bash
# Push the latest code to every fabric node and restart them — run on the MAIN.
#
# Each node pulls from its configured git remote and restarts augmentum (see
# scripts/update.sh). Per-node config is preserved (gitignore), and code
# changes apply on restart — no rebuild. This replaces shuffling bundles
# around: the main repo is the always-current source.
#
# Node list (first match wins):
#   1. AUGMENTUM_NODES env — space-separated "user@host[:/repo/path]"
#   2. scripts/nodes.conf  — one entry per line (# comments allowed)
#
# Requires: SSH access from the main to each node, and each node having a
# git remote that reaches the main (see scripts/FABRIC_UPDATE.md for setup).
#
# Flags:
#   --status   Only report each node's current commit vs the main's (no update)
set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
NODES_FILE="$SELF_DIR/nodes.conf"
STATUS_ONLY=0
[ "${1:-}" = "--status" ] && STATUS_ONLY=1

NODES="${AUGMENTUM_NODES:-}"
if [ -z "$NODES" ] && [ -f "$NODES_FILE" ]; then
  NODES="$(grep -vE '^\s*#|^\s*$' "$NODES_FILE")"
fi
if [ -z "$NODES" ]; then
  echo "No nodes configured. Create scripts/nodes.conf (user@host[:/path] per" >&2
  echo "line) or set AUGMENTUM_NODES. See scripts/FABRIC_UPDATE.md." >&2
  exit 1
fi

REMOTE="${AUGMENTUM_UPDATE_REMOTE:-main}"
BRANCH="${AUGMENTUM_UPDATE_BRANCH:-$(git -C "$SELF_DIR/.." rev-parse --abbrev-ref HEAD)}"
MAIN_SHA="$(git -C "$SELF_DIR/.." rev-parse --short HEAD)"
echo "main is at $MAIN_SHA ($BRANCH)"

fail=0
for entry in $NODES; do
  host="${entry%%:*}"
  path="${entry#*:}"; [ "$path" = "$host" ] && path='~/augmentum'
  if [ "$STATUS_ONLY" = 1 ]; then
    node_sha="$(ssh "$host" "git -C $path rev-parse --short HEAD" 2>/dev/null || echo '??')"
    mark="behind/ahead"; [ "$node_sha" = "$MAIN_SHA" ] && mark="up to date"
    printf '  %-28s %s  (%s)\n' "$host" "$node_sha" "$mark"
    continue
  fi
  echo "=== $host ($path) ==="
  if ssh "$host" "cd $path && AUGMENTUM_UPDATE_REMOTE=$REMOTE AUGMENTUM_UPDATE_BRANCH=$BRANCH ./scripts/update.sh"; then
    :
  else
    echo "  [FAIL] $host"; fail=1
  fi
done

if [ "$STATUS_ONLY" = 1 ]; then exit 0; fi
if [ "$fail" = 0 ]; then echo "All nodes updated."; else echo "One or more nodes failed." >&2; exit 1; fi
