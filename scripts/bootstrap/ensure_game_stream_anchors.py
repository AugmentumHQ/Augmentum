#!/usr/bin/env python3
"""Keep the game-stream images referenced so host cleanups can't sweep them.

WHY THIS EXISTS
---------------
The AGSP streaming images (base / luanti / emulator-streamed /
stream-browser) are never referenced by a long-running container.
``GameStreamRuntime`` spawns them per session and removes the container
at teardown, so between sessions the images sit there held by nothing.

To ``docker image prune -a`` -- and to Docker Desktop's "Clean up" --
an image that no container is using is indistinguishable from garbage.
That makes these the FIRST images a routine host cleanup deletes and
the LAST you notice, because nothing checks at boot: the only symptom
is a 404 at launch time, per title, possibly months later.

    Couldn't launch Spider-Man 3 (USA): runtime 'agsp-streamed' failed
    to launch: image 'augmentum-game-stream-emulator-streamed:latest'
    not visible to the augmentum container (HTTP 404).

That is what happened on 2026-07-25. The builder cache still held every
layer (so the images HAD been built), while the images themselves were
gone -- the fingerprint of an image sweep, which spares build cache.

AN ANCHOR is a container that exists only to be that missing
reference. It is **created and never started**: zero processes, zero
RAM, zero CPU, no network namespace, no disk beyond the container
record. It shows up in ``docker ps -a`` and nowhere else.

That a *created* container is enough was verified experimentally, both
directions, with a label-scoped prune::

    docker create ... probe-img          -> `docker image prune -af` reclaims 0B
    (no container)                       -> same prune reclaims the image

So the protection costs nothing to run. ``--running`` upgrades the
anchors to ``sleep infinity`` (measured: ~590KiB RSS, 0.00% CPU each)
for the one case a created anchor does NOT cover: ``docker system
prune -a`` removes stopped containers FIRST and then unused images, so
it defeats a created anchor in a single pass, while a running
container survives both.

Created is the default because the evidence says the sweep that
actually happened here was image-only: the **builder cache survived**,
and ``docker system prune -a`` would have taken that too.

Why this is a script and not compose services: the default start path
is an attached, blocking ``docker compose up``. A compose service whose
image is missing aborts the whole ``up``, which would turn a missing
optional streaming image into a total boot failure -- exactly the
class of regression that stranded augmentum+caddy in Created earlier
this cycle. This runs best-effort, warns, and never blocks boot.

The warning is the second half of the point. Missing images now
surface once at startup instead of per-title at launch.

Anchors deliberately do NOT carry the ``augmentum.game_stream`` label:
``DockerContainerAdapter.list_owned()`` filters on it to enumerate live
sessions, and an anchor appearing there would look like an orphaned
container to the runtime's reconcile path.

Usage::

    python scripts/bootstrap/ensure_game_stream_anchors.py
    python scripts/bootstrap/ensure_game_stream_anchors.py --check   # report only
    python scripts/bootstrap/ensure_game_stream_anchors.py --remove  # tear down

Always exits 0 unless --check is passed (then non-zero if an image is
missing), so callers can wire it in without risking the boot path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_FILE = REPO_ROOT / ".augmentum.conf"
OVERLAY = "compose.game-stream.yaml"

# Label lets a human (or a future `docker ps` filter) tell an anchor
# from something that matters. NOT the augmentum.game_stream label --
# see the module docstring.
ANCHOR_LABEL = "com.augmentum.role=image-anchor"

# (image tag, anchor container name, what breaks without it)
ANCHORS: tuple[tuple[str, str, str], ...] = (
    (
        "augmentum-game-stream-base:latest",
        "augmentum-anchor-gs-base",
        "the FROM parent of every streaming image -- losing the tag "
        "breaks all three rebuilds",
    ),
    (
        "augmentum-game-stream-luanti:latest",
        "augmentum-anchor-gs-luanti",
        "Luanti titles",
    ),
    (
        "augmentum-game-stream-emulator-streamed:latest",
        "augmentum-anchor-gs-emulator",
        "streamed GameCube / Wii / PS2 titles",
    ),
    (
        "augmentum-stream-browser:latest",
        "augmentum-anchor-stream-browser",
        "cast-to-TV of web surfaces (VRM, notebook, comic reader)",
    ),
)


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["docker", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )


def _image_id(tag: str) -> str:
    """Resolve a tag to an image ID, or "" when it isn't present."""
    r = _docker("image", "inspect", tag, "--format", "{{.Id}}")
    return r.stdout.strip() if r.returncode == 0 else ""


def _container_state(name: str) -> tuple[str, str]:
    """Return (status, image_id) for a container, ("", "") if absent."""
    r = _docker("container", "inspect", name)
    if r.returncode != 0:
        return "", ""
    try:
        data = json.loads(r.stdout)[0]
    except (ValueError, IndexError, KeyError):
        return "", ""
    return (data.get("State") or {}).get("Status", ""), data.get("Image", "")


def _create_anchor(tag: str, name: str, *, running: bool) -> bool:
    # `create` (not `run`) is the default: the container never starts,
    # so it costs nothing but still counts as a reference.
    verb = ["run", "--detach"] if running else ["create"]
    extra = ["--restart", "unless-stopped"] if running else []
    r = _docker(
        *verb,
        "--name", name,
        *extra,
        # No network namespace at all: an anchor must never be
        # reachable, resolvable, or occupy a port.
        "--network", "none",
        "--label", ANCHOR_LABEL,
        "--entrypoint", "/bin/sleep",
        tag,
        "infinity",
    )
    if r.returncode != 0:
        print(f"  [warning] could not anchor {tag}: {r.stderr.strip()[:200]}")
        return False
    return True


def _remove_all() -> int:
    for _tag, name, _why in ANCHORS:
        if _container_state(name)[0]:
            _docker("rm", "--force", name)
            print(f"  removed {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="report only; exit non-zero if a streaming image is missing",
    )
    ap.add_argument(
        "--remove", action="store_true",
        help="tear the anchors down (images become prune-eligible again)",
    )
    ap.add_argument(
        "--running", action="store_true",
        help=(
            "keep anchors RUNNING (~590KiB each) instead of created-only. "
            "Only needed to survive `docker system prune -a`, which removes "
            "stopped containers before it prunes images."
        ),
    )
    args = ap.parse_args()

    if args.remove:
        return _remove_all()

    # Game streaming is opt-in via the overlay. Anchoring images for a
    # subsystem the operator hasn't enabled would be pure noise.
    try:
        if OVERLAY not in CONF_FILE.read_text(encoding="utf-8"):
            return 0
    except OSError:
        return 0

    if _docker("version", "--format", "{{.Server.Version}}").returncode != 0:
        # Docker isn't up yet. Not our problem to solve, and definitely
        # not our problem to fail the boot over.
        return 0

    missing: list[tuple[str, str]] = []
    anchored = 0

    for tag, name, why in ANCHORS:
        image_id = _image_id(tag)
        if not image_id:
            missing.append((tag, why))
            # A stale anchor pointing at a deleted image is just a dead
            # container taking up a name we need.
            if _container_state(name)[0]:
                _docker("rm", "--force", name)
            continue

        status, anchored_id = _container_state(name)

        if status and anchored_id != image_id:
            # The image was rebuilt. The anchor still pins the OLD
            # image ID, which means the old one is protected as an
            # untagged leftover and the NEW one is not -- the exact
            # failure this script exists to prevent, quietly inverted.
            _docker("rm", "--force", name)
            status = ""

        if not status:
            anchored += int(_create_anchor(tag, name, running=args.running))
        elif args.running and status != "running":
            # Only meaningful under --running; a created/exited anchor is
            # already a valid reference for `docker image prune -a`.
            if _docker("start", name).returncode == 0:
                anchored += 1
            else:
                _docker("rm", "--force", name)
                anchored += int(_create_anchor(tag, name, running=True))

    if anchored:
        print(f"  Game-stream images anchored against pruning ({anchored} restored).")

    if missing:
        print()
        print("  [warning] game streaming is enabled but these images are NOT built:")
        for tag, why in missing:
            print(f"      {tag}")
            print(f"        without it: {why}")
        print("    Build them with: start.bat build   (or ./start.sh build)")
        print()

    return 1 if (missing and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
