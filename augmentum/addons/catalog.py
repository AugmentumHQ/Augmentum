"""The add-on catalog — what exists, what it costs, and what it depends on.

Kept as Python rather than JSON (unlike ``data/marketplace/listings.json``)
because these entries name Dockerfiles that ship *inside this repo*: an
add-on's build recipe and the code that builds it version together, and a
catalog that could drift from the files it references would fail at build
time instead of import time. Discover reads presentation copy from the
listings JSON; the buildable facts live here and the listing points at an
id in this table.

Every ``build_args`` value is PINNED. That is this category's equivalent of
the service standard's "no ``:latest``" rule — an unpinned build arg is an
unreproducible image, which is the same defect wearing different clothes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Where the narrow build context is mounted inside the augmentum container.
# See the mount in compose.yaml: only ./services/game-stream is exposed,
# read-only, because every COPY in the four game-stream Dockerfiles reads
# from services/game-stream/scripts/. An add-on build never ships the repo
# root to the daemon.
BUILD_CONTEXT_ROOT = "/app/addon-build"


@dataclass(frozen=True)
class AddonSpec:
    """One installable capability.

    ``capability`` is what the rest of Augmentum asks for. Runtime code must
    resolve an image through :func:`addons.registry.capability_image` rather
    than hardcoding a tag — that indirection is what turns "image missing" at
    launch time into "this add-on isn't installed" at decision time, which is
    the whole point of the category.
    """

    id: str
    capability: str
    title: str
    image: str
    # Path of the Dockerfile *within the build context tar*, which mirrors
    # the repo layout so the COPY lines in these Dockerfiles resolve
    # unchanged whether built by Docker Compose on the host or by the
    # in-app installer.
    dockerfile: str
    # Directory (host-relative) that becomes the tar, mapped to the same
    # prefix inside it.
    context_subdir: str
    build_args: dict[str, str] = field(default_factory=dict)
    # Other add-on ids that must be built FIRST. base is the FROM parent of
    # all three leaf images, so it is a real dependency, not a nicety.
    requires: tuple[str, ...] = ()
    # False = dependency-only; never shown as its own Discover card. It is
    # installed implicitly and removed only when its last dependant goes.
    user_facing: bool = True
    # Honest, measured costs (2026-07-25 build on Matt's box). build_minutes
    # is wall-clock on a warm cache-less first build.
    disk_mb: int = 0
    build_minutes: int = 0
    # Shown before the build starts when the recipe compiles or installs
    # third-party software whose terms the *builder* accepts. Empty means no
    # acknowledgement step.
    license_notice: str = ""
    # Where "Open" goes once installed — an in-app surface, never a URL.
    surface: str = ""
    # One-line answer to "what stops working if this is missing?", reused by
    # the startup preflight warning and the uninstall confirmation.
    provides: str = ""


ADDONS: tuple[AddonSpec, ...] = (
    AddonSpec(
        id="game-stream-base",
        capability="game_stream.base",
        title="Streaming runtime",
        image="augmentum-game-stream-base:latest",
        dockerfile="game-stream/Dockerfile.base",
        context_subdir="game-stream",
        build_args={
            "SELKIES_VERSION": "1.6.1",
            "SELKIES_WEB_VERSION": "1.6.2",
        },
        user_facing=False,
        disk_mb=1910,
        build_minutes=8,
        provides=(
            "the shared WebRTC streaming layer every other streaming add-on "
            "builds FROM"
        ),
    ),
    AddonSpec(
        id="game-stream-luanti",
        capability="game_stream.luanti",
        title="Luanti worlds",
        image="augmentum-game-stream-luanti:latest",
        dockerfile="game-stream/Dockerfile.luanti",
        context_subdir="game-stream",
        requires=("game-stream-base",),
        disk_mb=1980,
        build_minutes=4,
        # surface = "<overlay>:<selection kind>:<sub-selection>". The third
        # segment is the games source id (ui/scripts/library-game-sources.js);
        # without it the browse pane opens on whichever source sorts first.
        surface="library:browse-games:streamed",
        provides="streamed Luanti worlds, with persistent per-profile saves",
    ),
    AddonSpec(
        id="game-stream-emulator",
        capability="game_stream.emulator",
        title="Console emulation",
        image="augmentum-game-stream-emulator-streamed:latest",
        dockerfile="game-stream/Dockerfile.emulator-streamed",
        context_subdir="game-stream",
        build_args={
            "DOLPHIN_VERSION": "2603a",
            "PCSX2_VERSION": "v2.0.2",
        },
        requires=("game-stream-base",),
        disk_mb=2300,
        build_minutes=25,
        license_notice=(
            "This add-on compiles Dolphin (GPLv2) from pinned source and "
            "installs PCSX2 (GPLv3) release binaries on your machine. Both "
            "are free software; you are building them locally rather than "
            "receiving them from Augmentum. Emulators ship no game data and "
            "no BIOS — you supply your own."
        ),
        surface="library:browse-games:emulator",
        provides="streamed GameCube, Wii and PS2 titles",
    ),
    AddonSpec(
        id="stream-browser",
        capability="cast.browser",
        title="Cast to TV",
        image="augmentum-stream-browser:latest",
        dockerfile="game-stream/Dockerfile.browser",
        context_subdir="game-stream",
        requires=("game-stream-base",),
        disk_mb=2740,
        build_minutes=6,
        license_notice=(
            "This add-on installs Google Chrome, which is proprietary "
            "software governed by Google's Terms of Service. Augmentum "
            "cannot redistribute it, so the build downloads it from Google "
            "onto your machine under your acceptance of those terms."
        ),
        # No verified sub-selection for cast targets, so this opens the
        # Library root rather than guessing a pane that may not exist.
        surface="library",
        provides=(
            "casting Augmentum's own web surfaces (avatar, notebook, comic "
            "reader) to a TV or receiver"
        ),
    ),
)

_BY_ID = {a.id: a for a in ADDONS}
_BY_CAPABILITY = {a.capability: a for a in ADDONS}


def addon_by_id(addon_id: str) -> AddonSpec | None:
    return _BY_ID.get(addon_id)


def addon_for_capability(capability: str) -> AddonSpec | None:
    return _BY_CAPABILITY.get(capability)


def user_facing_addons() -> tuple[AddonSpec, ...]:
    """Add-ons that get their own Discover card (excludes dependencies)."""
    return tuple(a for a in ADDONS if a.user_facing)


def resolve_build_order(addon_id: str) -> list[AddonSpec]:
    """Return ``addon_id`` and its dependencies, dependencies first.

    Depth-first with a visited set. The graph is tiny and shallow (one
    shared base), but resolving it properly means a user installing the
    emulator add-on never has to know the base exists — the dependency is
    implicit in the UI and explicit here.
    """
    ordered: list[AddonSpec] = []
    seen: set[str] = set()

    def _visit(aid: str) -> None:
        if aid in seen:
            return
        seen.add(aid)
        spec = _BY_ID.get(aid)
        if spec is None:
            raise KeyError(f"unknown add-on {aid!r}")
        for dep in spec.requires:
            _visit(dep)
        ordered.append(spec)

    _visit(addon_id)
    return ordered


def dependants_of(addon_id: str) -> tuple[AddonSpec, ...]:
    """Add-ons that declare ``addon_id`` as a requirement.

    Uninstall refcounts against this: the shared base is only removed once
    the last leaf that builds FROM it is gone, otherwise removing one
    streaming add-on would silently break the others' rebuilds.
    """
    return tuple(a for a in ADDONS if addon_id in a.requires)
