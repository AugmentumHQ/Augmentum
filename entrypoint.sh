#!/bin/sh
# One-time fix: ensure /data directories are owned by augmentum.
# Handles volumes that were created by an older image running as root.

MARKER="/data/.permissions_fixed"

if [ ! -f "$MARKER" ]; then
    echo "[entrypoint] Fixing /data permissions (one-time)..."
    mkdir -p /data/image_models /data/image_output
    chown -R augmentum:augmentum /data
    touch "$MARKER"
    chown augmentum:augmentum "$MARKER"
    echo "[entrypoint] Done."
fi

# The home dir itself must be owned by augmentum with the search (execute) bit,
# or the app (uid 1000) cannot descend into ~/.cache/* — every fastembed /
# huggingface access then fails with EACCES and the embedder is marked
# permanently unavailable (memory recall + extraction + narrative embedding all
# silently degrade). The base image / volume mounts can leave /home/augmentum as
# root:root 750, so fix the PARENT before the per-subdir chowns below.
chown augmentum:augmentum /home/augmentum 2>/dev/null || true
chmod 755 /home/augmentum 2>/dev/null || true

# Ensure cache directories are writable by augmentum user
# (volumes may be created as root on first mount)
for d in /home/augmentum/.cache \
         /home/augmentum/.wespeaker \
         /home/augmentum/.dtln \
         /home/augmentum/.kokoro \
         /home/augmentum/.smart-turn \
         /home/augmentum/.u2net; do
    if [ -d "$d" ]; then
        chown -R augmentum:augmentum "$d" 2>/dev/null || true
    fi
done

# Docker access now flows through the docker-proxy sidecar (compose.yaml).
# The augmentum container no longer mounts /var/run/docker.sock directly,
# so there is nothing to chmod. DOCKER_HOST=tcp://docker-proxy:2375 in the
# environment routes aiodocker through the proxy's ACL.

# --- self-edit L2 boot parachute (augmentum/selfedit/rollback.py) ----------
# A self-edit "apply" that breaks the backend would crash-loop on Docker's
# respawn. The app resets the boot counter once it reaches healthy startup
# (mark_boot_healthy), and the apply path snapshots known-good augmentum/ first.
# So: bump the counter each boot and, if it crosses the threshold WHILE a
# snapshot exists (= consecutive failed boots), restore the snapshot before
# launching. Pure /data file I/O — no git, no network. Run as the augmentum user
# so the counter file stays app-writable; best-effort, never blocks boot.
if [ -f /app/augmentum/selfedit/rollback.py ]; then
    gosu augmentum python -c '
import sys
sys.path.insert(0, "/app")
from augmentum.selfedit import rollback as r
d = "/data"
n = r.record_boot_attempt(d)
if r.should_rollback(d) and r.has_snapshot(d):
    print("[entrypoint] self-edit crash-loop (boot #%d) -- restoring last-good augmentum/" % n)
    r.restore_tree(d, "/app/augmentum")
    r.clear_boot_counter(d)
' 2>/dev/null || true
fi

exec gosu augmentum "$@"
