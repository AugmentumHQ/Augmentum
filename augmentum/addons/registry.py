"""Add-on availability, anchoring, and capability resolution.

THE ANCHOR IS THE INSTALL RECORD. There is deliberately no
``installed_addons`` table. An add-on's whole state is "does the image
exist, and is it held against pruning", both of which the Docker daemon
already knows authoritatively. A database row would be a second source of
truth that can disagree with the daemon — and disagreeing with the daemon is
precisely the failure this subsystem exists to prevent (2026-07-25: images
swept by a host cleanup, nothing noticed until a per-title 404 at launch).

An anchor is a container that exists only to be a reference. It is CREATED
AND NEVER STARTED: zero processes, zero RAM, zero CPU, no network namespace,
nothing but a container record. Verified experimentally in both directions
with a label-scoped prune — with a created container ``docker image prune
-af`` reclaims 0B; without one it takes the image.

Anchors must NOT carry the ``augmentum.game_stream`` label:
``DockerContainerAdapter.list_owned()`` filters on it to enumerate live
sessions, and an anchor showing up there would read as an orphaned session
container to the runtime's reconcile path.

The standalone ``scripts/bootstrap/ensure_game_stream_anchors.py`` remains
the host-side equivalent for operators who build with ``start.sh build``;
both converge on the same container names and label, so they are
interchangeable rather than competing.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from augmentum.addons.catalog import (
    ADDONS,
    AddonSpec,
    addon_by_id,
    addon_for_capability,
    dependants_of,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Matches the label written by scripts/bootstrap/ensure_game_stream_anchors.py
# so host-built and in-app-built installs are indistinguishable afterwards.
ANCHOR_LABEL_KEY = "com.augmentum.role"
ANCHOR_LABEL_VALUE = "image-anchor"

# Anchor container names, matching the bootstrap script exactly. Changing one
# without the other would leave two anchors per image and make uninstall
# appear to succeed while the image stayed pinned.
_ANCHOR_NAMES = {
    "game-stream-base": "augmentum-anchor-gs-base",
    "game-stream-luanti": "augmentum-anchor-gs-luanti",
    "game-stream-emulator": "augmentum-anchor-gs-emulator",
    "stream-browser": "augmentum-anchor-stream-browser",
}


def anchor_name_for(addon_id: str) -> str:
    return _ANCHOR_NAMES.get(addon_id, f"augmentum-anchor-{addon_id}")


@dataclass(frozen=True)
class AddonState:
    """Live state of one add-on, derived entirely from the Docker daemon."""

    id: str
    capability: str
    installed: bool          # image present
    anchored: bool           # held against host pruning
    image_id: str = ""
    # True when the image exists but nothing holds it — the exact condition
    # that preceded the 2026-07-25 sweep. Surfaced so the UI can offer to
    # re-anchor instead of waiting for the failure.
    at_risk: bool = False


async def _client(app_state: Any = None):
    """Return an aiodocker client, preferring one already on ``app.state``.

    Falls back to constructing one (DOCKER_HOST points at docker-proxy).
    Returns ``(client, owned)`` — callers close only what they created.
    """
    existing = getattr(app_state, "docker_client", None) if app_state is not None else None
    if existing is not None:
        return existing, False
    import aiodocker

    return aiodocker.Docker(), True


async def _image_id(docker: Any, image: str) -> str:
    try:
        info = await docker.images.inspect(image)
    except Exception:  # noqa: BLE001 — absence is the common, expected case
        return ""
    return str(info.get("Id") or "")


async def _anchor_image_id(docker: Any, name: str) -> tuple[bool, str]:
    """Return ``(exists, image_id)`` for an anchor container."""
    try:
        container = docker.containers.container(name)
        info = await container.show()
    except Exception:  # noqa: BLE001
        return False, ""
    return True, str(info.get("Image") or "")


async def get_state(addon_id: str, *, app_state: Any = None) -> AddonState:
    spec = addon_by_id(addon_id)
    if spec is None:
        raise KeyError(f"unknown add-on {addon_id!r}")
    docker, owned = await _client(app_state)
    try:
        image_id = await _image_id(docker, spec.image)
        has_anchor, anchored_id = await _anchor_image_id(docker, anchor_name_for(addon_id))
        # A stale anchor (pointing at a rebuilt image's predecessor) protects
        # the OLD image while leaving the new one exposed — the failure this
        # subsystem prevents, quietly inverted. Treat it as unanchored.
        fresh = has_anchor and bool(image_id) and anchored_id == image_id
        return AddonState(
            id=spec.id,
            capability=spec.capability,
            installed=bool(image_id),
            anchored=fresh,
            image_id=image_id,
            at_risk=bool(image_id) and not fresh,
        )
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await docker.close()


async def list_states(*, app_state: Any = None) -> dict[str, AddonState]:
    """State for every catalogued add-on, keyed by id.

    One client for the whole sweep — Discover renders the full add-on
    section from a single call.
    """
    docker, owned = await _client(app_state)
    out: dict[str, AddonState] = {}
    try:
        for spec in ADDONS:
            image_id = await _image_id(docker, spec.image)
            has_anchor, anchored_id = await _anchor_image_id(
                docker, anchor_name_for(spec.id),
            )
            fresh = has_anchor and bool(image_id) and anchored_id == image_id
            out[spec.id] = AddonState(
                id=spec.id,
                capability=spec.capability,
                installed=bool(image_id),
                anchored=fresh,
                image_id=image_id,
                at_risk=bool(image_id) and not fresh,
            )
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await docker.close()
    return out


async def ensure_anchor(spec: AddonSpec, *, docker: Any) -> bool:
    """Create (never start) the anchor container holding ``spec.image``.

    Idempotent, and self-correcting across rebuilds: an anchor pinning a
    different image id is destroyed and recreated, because a stale anchor is
    worse than none — it protects the wrong image and reads as "safe".
    """
    name = anchor_name_for(spec.id)
    image_id = await _image_id(docker, spec.image)
    if not image_id:
        return False

    exists, anchored_id = await _anchor_image_id(docker, name)
    if exists and anchored_id == image_id:
        return True
    if exists:
        with contextlib.suppress(Exception):
            await docker.containers.container(name).delete(force=True)

    try:
        await docker.containers.create(
            config={
                "Image": spec.image,
                # `sleep infinity` is never executed — the container is
                # created, not started. It is only here so the config is
                # valid and so `--running` upgrades stay possible.
                "Entrypoint": ["/bin/sleep"],
                "Cmd": ["infinity"],
                "Labels": {ANCHOR_LABEL_KEY: ANCHOR_LABEL_VALUE},
                # No network namespace at all: an anchor must never be
                # reachable, resolvable, or occupy a port.
                "HostConfig": {"NetworkMode": "none"},
            },
            name=name,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("addon_anchor_failed", addon=spec.id, error=str(exc))
        return False
    log.info("addon_anchored", addon=spec.id, image=spec.image)
    return True


async def remove_addon(
    addon_id: str, *, app_state: Any = None, delete_image: bool = True,
) -> dict[str, Any]:
    """Uninstall: drop the anchor, delete the image, release dependencies.

    Uninstall REALLY reclaims the disk — that is the honest counterpart to an
    install that spent 25 minutes and 2.3GB. The shared base is refcounted:
    it goes only when the last add-on that builds FROM it is gone, otherwise
    removing one streaming add-on would break the others' next rebuild.
    """
    spec = addon_by_id(addon_id)
    if spec is None:
        raise KeyError(f"unknown add-on {addon_id!r}")

    docker, owned = await _client(app_state)
    removed: list[str] = []
    kept: list[str] = []
    try:
        async def _drop(target: AddonSpec) -> None:
            with contextlib.suppress(Exception):
                await docker.containers.container(
                    anchor_name_for(target.id),
                ).delete(force=True)
            if delete_image:
                try:
                    await docker.images.delete(target.image, force=False)
                except Exception as exc:  # noqa: BLE001
                    # Most often: another image still builds FROM it, or a
                    # session container is live. Not fatal — the anchor is
                    # gone, so the image is prunable and the capability is
                    # already reported absent.
                    log.warning(
                        "addon_image_delete_failed",
                        addon=target.id, image=target.image, error=str(exc),
                    )
            removed.append(target.id)

        await _drop(spec)

        # Refcount the dependencies this add-on pulled in.
        for dep_id in spec.requires:
            dep = addon_by_id(dep_id)
            if dep is None or dep.user_facing:
                continue
            still_needed = []
            for other in dependants_of(dep_id):
                if other.id == spec.id:
                    continue
                if await _image_id(docker, other.image):
                    still_needed.append(other.id)
            if still_needed:
                kept.append(dep_id)
                log.info(
                    "addon_dependency_retained",
                    dependency=dep_id, needed_by=still_needed,
                )
            else:
                await _drop(dep)
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await docker.close()

    log.info("addon_uninstalled", addon=addon_id, removed=removed, kept=kept)
    return {"removed": removed, "kept": kept}


# ── Capability resolution (what the rest of Augmentum calls) ──────────


def capability_image(capability: str) -> str:
    """Image tag backing a capability, or "" if the capability is unknown."""
    spec = addon_for_capability(capability)
    return spec.image if spec else ""


async def capability_available(capability: str, *, app_state: Any = None) -> bool:
    """True when the add-on providing ``capability`` is installed.

    Callers should prefer this over inspecting an image tag directly: it is
    the seam that lets a missing capability be reported as "not installed,
    here's the add-on" instead of a launch-time HTTP 404.
    """
    spec = addon_for_capability(capability)
    if spec is None:
        return False
    state = await get_state(spec.id, app_state=app_state)
    return state.installed
