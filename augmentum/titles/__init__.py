"""Augmentum Experience Framework (AXF) -- Title abstraction.

A "Title" is anything playable, runnable, or interactively viewable: a
js13k game, an AGSP-streamed Luanti world, an emulator ROM, a bookmarked
web app, a GitHub-cloned indie project. They all reduce to the same
shape: an ``artifacts`` row whose ``metadata.kind`` is one of the
recognised title kinds, plus optional auxiliary data (runs, saves) in
dedicated tables.

This package is the *substrate* layer. It exposes:

* ``TitleManifest`` -- a typed projection over an artifact row
* ``TitleStore``    -- CRUD over manifests + per-launch ``title_runs``
* ``Source`` + ``SourceRegistry`` -- discovery / import protocol
* ``Runtime`` + ``RuntimeRegistry`` -- execution protocol
* ``TitleService`` -- the orchestrator that ties them together

The route layer (``augmentum/proxy/titles_routes.py``) is a thin shell
around ``TitleService``. Adding a new source or runtime is one
registry registration; nothing else needs to change.

See ``docs/superpowers/specs/2026-05-08-axf-design.md`` (forthcoming)
for the full architecture notes. See also the existing
``augmentum/games/`` (catalog discovery for js13k) and
``augmentum/game_stream/`` (AGSP container runtime) which are the two
substrates this layer composes over.
"""

from __future__ import annotations

from augmentum.titles.bios_store import (
    BiosRecord,
    BiosServiceError,
    BiosStatusEntry,
    BiosStore,
)
from augmentum.titles.manifest import (
    KIND_EMULATOR_ROM,
    KIND_GIT_PROJECT,
    KIND_JS13K_GAME,
    KIND_STREAMED_GAME,
    KIND_WEB_APP,
    TITLE_KINDS,
    TitleManifest,
    is_title_kind,
)
from augmentum.titles.runtimes import (
    AgspStreamedRuntime,
    BrowserIframeRuntime,
    EmulatorBrowserRuntime,
    LaunchHandle,
    Runtime,
    RuntimeRegistry,
    runtime_registry,
)
from augmentum.titles.service import (
    TitleNotFound,
    TitleNotPlayable,
    TitleService,
    TitleServiceError,
)
from augmentum.titles.sources import (
    DiscoveryItem,
    InternalSource,
    Source,
    SourceImportError,
    SourceRegistry,
    source_registry,
)
from augmentum.titles.sources_js13k import Js13kSource
from augmentum.titles.sources_rom import InternalRomSource
from augmentum.titles.store import TitleStore

__all__ = [
    "AgspStreamedRuntime",
    "BiosRecord",
    "BiosServiceError",
    "BiosStatusEntry",
    "BiosStore",
    "BrowserIframeRuntime",
    "DiscoveryItem",
    "EmulatorBrowserRuntime",
    "InternalRomSource",
    "InternalSource",
    "Js13kSource",
    "KIND_EMULATOR_ROM",
    "KIND_GIT_PROJECT",
    "KIND_JS13K_GAME",
    "KIND_STREAMED_GAME",
    "KIND_WEB_APP",
    "LaunchHandle",
    "Runtime",
    "RuntimeRegistry",
    "Source",
    "SourceImportError",
    "SourceRegistry",
    "TITLE_KINDS",
    "TitleManifest",
    "TitleNotFound",
    "TitleNotPlayable",
    "TitleService",
    "TitleServiceError",
    "TitleStore",
    "is_title_kind",
    "runtime_registry",
    "source_registry",
]
