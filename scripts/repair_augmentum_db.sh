#!/usr/bin/env bash
# Repair a corrupted augmentum.db. POSIX equivalent of repair_augmentum_db.ps1.
#
# Stops augmentum, runs the repair logic in a transient container that
# mounts the data volume, then restarts augmentum.
#
# Usage (from repo root):
#   ./scripts/repair_augmentum_db.sh
#
# Optional env:
#   AUGMENTUM_CONTAINER  default: augmentum-augmentum-1
#   AUGMENTUM_VOLUME     default: augmentum_augmentum_data
#   AUGMENTUM_IMAGE      default: resolved from the running container

set -euo pipefail

container="${AUGMENTUM_CONTAINER:-augmentum-augmentum-1}"
volume="${AUGMENTUM_VOLUME:-augmentum_augmentum_data}"

# Resolve the image from the running container so we use the same one
# the user is actually running, not whatever 'augmentum-augmentum'
# happens to point at locally.
image="${AUGMENTUM_IMAGE:-}"
if [ -z "$image" ]; then
    image="$(docker inspect --format '{{.Config.Image}}' "$container" 2>/dev/null || true)"
    image="${image:-augmentum-augmentum}"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_host="$script_dir/repair_augmentum_db.py"
if [ ! -f "$script_host" ]; then
    echo "[wrapper] error: repair script not found: $script_host" >&2
    exit 1
fi

echo "[wrapper] container: $container"
echo "[wrapper] volume:    $volume"
echo "[wrapper] image:     $image"
echo

echo "[wrapper] stopping $container..."
docker stop "$container" >/dev/null

echo "[wrapper] running repair..."
script_in_container=/tmp/repair_augmentum_db.py
repair_cmd="(command -v sqlite3 >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq sqlite3 >/dev/null)) && python3 $script_in_container"

set +e
docker run --rm \
    -v "${volume}:/data" \
    -v "${script_host}:${script_in_container}:ro" \
    --entrypoint sh \
    --user root \
    "$image" \
    -c "$repair_cmd"
repair_exit=$?
set -e

echo
echo "[wrapper] starting $container..."
docker start "$container" >/dev/null

if [ "$repair_exit" -ne 0 ]; then
    echo
    echo "[wrapper] WARNING: repair exited with code $repair_exit" >&2
    echo "[wrapper] DB may still be corrupt; check repair logs above" >&2
    exit "$repair_exit"
fi

echo
echo "[wrapper] done."
echo "  - active DB:    /data/augmentum.db (rebuilt)"
echo "  - retired:      /data/augmentum.db.corrupt-<ts>"
echo "  - safety copy:  /data/augmentum.db.backup-<ts>"
echo
echo "Inspect via: docker exec $container ls -la /data/"
