"""Source registry -- where Titles come from.

A Source is anything that produces TitleManifests: an external catalog
(js13k GitHub directory), a built-in profile registry (AGSP), a manual
upload flow, a URL bookmark, a GitHub repo clone, the curated
marketplace. Each Source declares two things:

* **Discovery** -- ``discover(query)`` returns ``DiscoveryItem``s the UI
  can render and the user can act on (install / pin / launch).
* **Import** -- ``import_for_user(manifest_data, user_id)`` materialises
  a discovered item into an ``artifacts`` row, returning the new
  artifact id. The route layer wraps this in a 201 and returns the
  manifest projection.

Sources are stateless coordinators. Heavy state (cached catalog hits,
pending downloads) lives in the source's helper layer (e.g.
``augmentum/games/providers/js13k.py``). Adding a new source = one
``source_registry.register(...)`` call.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from augmentum.titles.manifest import TITLE_KINDS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class SourceImportError(Exception):
    """Raised when a Source can't materialise the requested manifest.

    Carriers should map this to 4xx (bad input) or 5xx (source
    upstream failure) at the route layer.
    """


# ── DiscoveryItem ────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiscoveryItem:
    """A title that's discoverable but not yet installed.

    The shape is stable across sources so a single UI card can render
    js13k hits, marketplace listings, AGSP profiles, GitHub releases
    without per-source branches. Items round-trip into
    ``Source.import_for_user`` to become real titles.
    """

    source_id: str                          # 'js13k' / 'marketplace' / 'agsp-profile' / ...
    source_remote_id: str                   # stable id within the source
    kind: str                               # one of TITLE_KINDS
    title: str
    runtime_preferred: str = ""
    runtime_alternates: tuple[str, ...] = ()
    author: str = ""
    tagline: str = ""
    description: str = ""
    thumbnail_url: str = ""
    source_url: str = ""                    # canonical "go look at this elsewhere" URL
    embed_url: str = ""                     # framable URL for browser-iframe
    capabilities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Whether the user already has this one installed. Sources may
    # leave it None and let the service layer decorate it after a
    # cheap library lookup.
    installed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_remote_id": self.source_remote_id,
            "kind": self.kind,
            "title": self.title,
            "runtime_preferred": self.runtime_preferred,
            "runtime_alternates": list(self.runtime_alternates),
            "author": self.author,
            "tagline": self.tagline,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "source_url": self.source_url,
            "embed_url": self.embed_url,
            "capabilities": dict(self.capabilities),
            "metadata": dict(self.metadata),
            "installed": self.installed,
        }


class Source(Protocol):
    """Contract for a Title source.

    Sources are stateless coordinators. State (cached catalog hits,
    pending downloads, etc.) lives in the source's own helper layer.
    """

    id: str
    label: str

    async def discover(
        self, query: dict[str, Any], *, user_id: str = "",
    ) -> list[DiscoveryItem]:
        """Return a list of candidate items for the given query.

        ``query`` is a free-form dict; each source documents its keys
        (``sort``, ``page``, ``q``, etc.). Items round-trip into
        ``import_for_user`` -- the UI can pass them straight back when
        the user clicks Install.
        """

    async def import_for_user(
        self, manifest_data: dict, *, user_id: str,
    ) -> str:
        """Materialise a manifest into an artifact for this user.

        ``manifest_data`` is either a DiscoveryItem.to_dict() (the
        common path: install something the user just discovered) or a
        hand-built manifest dict (the InternalSource path).

        Returns the artifact id. Raises ``SourceImportError`` on
        validation failure -- the route layer translates to 400.
        """


# ── Built-in: InternalSource ──────────────────────────────────────────


class InternalSource:
    """Source for manually-provided manifests.

    The route layer accepts a JSON body that's already a TitleManifest
    description (kind, title, source_remote_id, runtime_preferred,
    metadata, etc.) and persists it as an artifact row. Used for:

    * Emulator ROM imports (Phase C will bridge this through a richer
      uploads-flow that also writes the ROM blob)
    * URL bookmarks (Phase F)
    * Test fixtures and manual seeding

    All policy lives in the route layer: this source just coordinates
    persistence.
    """

    id = "internal"
    label = "Internal"

    def __init__(self, conn, artifact_store: Any | None = None) -> None:
        self._conn = conn
        self._artifacts = artifact_store

    async def discover(
        self, query: dict[str, Any], *, user_id: str = "",
    ) -> list[DiscoveryItem]:
        # InternalSource has no catalog -- it exists for the
        # ``hand-built manifest`` path. Return empty so the protocol
        # stays uniform.
        return []

    async def import_for_user(
        self, manifest_data: dict, *, user_id: str,
    ) -> str:
        if not user_id:
            raise SourceImportError("user_id required")
        kind = str(manifest_data.get("kind", "")).strip()
        if kind not in TITLE_KINDS:
            raise SourceImportError(
                f"unknown title kind: {kind!r} "
                f"(known: {sorted(TITLE_KINDS)})"
            )
        title = str(manifest_data.get("title", "")).strip()
        if not title:
            raise SourceImportError("title is required")

        source_remote_id = str(manifest_data.get("source_remote_id", ""))
        runtime_preferred = str(manifest_data.get("runtime_preferred", ""))

        # Build the metadata blob the artifacts table will store.
        # ``kind`` is the discriminator; everything else is free-form
        # but typed by the manifest layer on read.
        metadata = {
            "kind": kind,
            "source": str(manifest_data.get("source_id") or self.id),
            "source_id": source_remote_id,
            "title": title,
        }
        if runtime_preferred:
            metadata["runtime_preferred"] = runtime_preferred
        alternates = manifest_data.get("runtime_alternates")
        if isinstance(alternates, list):
            metadata["runtime_alternates"] = [str(a) for a in alternates]
        capabilities = manifest_data.get("capabilities")
        if isinstance(capabilities, dict):
            metadata["capabilities"] = capabilities
        # Pass through any metadata the caller supplied (genre, tags,
        # screenshots, year, ...). Don't allow them to overwrite the
        # discriminator keys we just set.
        extra = manifest_data.get("metadata") or {}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k in metadata:
                    continue
                metadata[k] = v

        artifact_id = await self._insert_artifact_row(
            user_id=user_id,
            display_name=title,
            metadata=metadata,
        )
        log.info(
            "title_imported_via_internal",
            user_id=user_id,
            artifact_id=artifact_id,
            kind=kind,
            source_remote_id=source_remote_id,
        )
        return artifact_id

    async def _insert_artifact_row(
        self,
        *,
        user_id: str,
        display_name: str,
        metadata: dict,
    ) -> str:
        """Persist a manifest as an artifact row.

        We bypass ArtifactStore's higher-level ``save`` because that path
        expects a binary payload + filename. Title manifests don't
        always have one (URL bookmark = no payload). Direct INSERT keeps
        the schema as the source of truth without forcing a stub blob.
        """
        artifact_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO artifacts
               (id, task_id, session_id, filename, display_name, format,
                size_bytes, path, metadata, user_id, pinned)
               VALUES (?, '', '', ?, ?, '', 0, '', ?, ?, 1)""",
            (
                artifact_id,
                f"{display_name}.title",
                display_name,
                json.dumps(metadata),
                user_id,
            ),
        )
        await self._conn.commit()
        return artifact_id


# ── Registry ──────────────────────────────────────────────────────────


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}

    def register(self, source: Source) -> None:
        self._sources[source.id] = source

    def get(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def list(self) -> list[Source]:
        return sorted(self._sources.values(), key=lambda s: s.label)

    def has(self, source_id: str) -> bool:
        return source_id in self._sources

    def clear(self) -> None:
        """Reset registry (test-only)."""
        self._sources.clear()


# Module-level singleton. Tests construct their own when isolation matters.
source_registry = SourceRegistry()
