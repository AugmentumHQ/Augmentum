"""Js13kSource -- AXF Source bridge over the existing js13k provider.

Wraps ``augmentum/games/providers/js13k.py`` so the unified
``/api/titles/_/discover?source_id=js13k`` and ``POST /api/titles/``
paths can browse and install js13k games without duplicating the
catalog/parse logic that already lives in the provider.

The js13k_provider returns ``GameBrowseResult`` rows. We map them onto
``DiscoveryItem`` and route an "install" call back through the
existing pin flow shape (writing ``metadata.kind = "js13k_game"`` and
``metadata.source = "js13k"`` -- the manifest layer's legacy bridge
also accepts the older ``metadata.kind = "game"`` value, so coexistence
with previously-pinned artifacts is automatic).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from augmentum.games.providers import js13k as js13k_provider
from augmentum.titles.manifest import KIND_JS13K_GAME
from augmentum.titles.sources import DiscoveryItem, SourceImportError
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient

log = get_logger(__name__)


class Js13kSource:
    """Source bridge for the js13kgames.com catalog."""

    id = "js13k"
    label = "js13k"

    def __init__(
        self,
        conn,
        *,
        safe_client: SafeHttpClient | None = None,
    ) -> None:
        self._conn = conn
        # Reuse a single SafeHttpClient across the source's lifetime.
        # The existing games_routes path keeps its own module-level
        # client; both can coexist (each gets its own connection
        # pool). If we ever consolidate, the route layer would inject
        # its client here.
        self._client = safe_client or SafeHttpClient()

    # ── Discovery ────────────────────────────────────────────────────

    async def discover(
        self, query: dict[str, Any], *, user_id: str = "",
    ) -> list[DiscoveryItem]:
        sort = str(query.get("sort", "newest"))
        page = max(1, int(query.get("page", 1) or 1))
        try:
            hits = await js13k_provider.browse(sort, page, self._client)
        except Exception as exc:
            log.warning("js13k_discover_failed", page=page, error=str(exc))
            return []
        return [_hit_to_discovery_item(h) for h in hits]

    # ── Install ──────────────────────────────────────────────────────

    async def import_for_user(
        self, manifest_data: dict, *, user_id: str,
    ) -> str:
        if not user_id:
            raise SourceImportError("user_id required")

        slug = (
            manifest_data.get("source_remote_id")
            or manifest_data.get("source_id")
            or ""
        )
        slug = str(slug).strip()
        if not slug:
            raise SourceImportError(
                "source_remote_id (the js13k slug) is required",
            )
        title = (
            str(manifest_data.get("title") or "").strip()
            or _humanize_slug(slug)
        )

        # Idempotent: if the user already pinned this slug, return the
        # existing artifact id instead of inserting a duplicate.
        existing = await self._find_existing(slug, user_id=user_id)
        if existing:
            return existing

        # Build the canonical metadata blob. We adopt the new
        # ``kind = "js13k_game"`` value for fresh imports; the manifest
        # layer's legacy bridge keeps older ``kind = "game"`` rows
        # readable.
        metadata: dict[str, Any] = {
            "kind": KIND_JS13K_GAME,
            "source": self.id,
            "source_id": slug,
            "title": title,
            "runtime_preferred": "browser-iframe",
            "embed_url": str(
                manifest_data.get("embed_url")
                or f"{js13k_provider._PLAY_BASE}/{slug}/"
            ),
            "source_url": str(
                manifest_data.get("source_url")
                or js13k_provider._source_page(slug)
            ),
            "thumbnail_url": str(manifest_data.get("thumbnail_url") or ""),
            "tagline": str(manifest_data.get("tagline") or ""),
            "author": str(manifest_data.get("author") or ""),
            "play_mode": "local",
        }
        # Pass through any extra metadata the caller supplied (genre,
        # screenshots, year, ...). We protect the discriminator keys
        # so they can't be overwritten via the API.
        extra = manifest_data.get("metadata") or {}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k in metadata:
                    continue
                metadata[k] = v

        artifact_id = await self._insert_artifact(
            user_id=user_id,
            display_name=title,
            metadata=metadata,
        )
        log.info(
            "title_imported_via_js13k",
            user_id=user_id,
            artifact_id=artifact_id,
            slug=slug,
        )
        return artifact_id

    # ── Internals ────────────────────────────────────────────────────

    async def _find_existing(
        self, slug: str, *, user_id: str,
    ) -> str | None:
        """Look up an already-pinned artifact for this user + slug.

        Matches both the modern ``kind = "js13k_game"`` AND the legacy
        ``kind = "game"`` rows so existing pins aren't double-imported.
        """
        cursor = await self._conn.execute(
            "SELECT id FROM artifacts "
            "WHERE user_id = ? "
            "  AND json_extract(metadata, '$.source') = ? "
            "  AND json_extract(metadata, '$.source_id') = ? "
            "  AND ("
            "    json_extract(metadata, '$.kind') = ? OR "
            "    json_extract(metadata, '$.kind') = 'game'"
            "  ) "
            "LIMIT 1",
            (user_id, self.id, slug, KIND_JS13K_GAME),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def _insert_artifact(
        self,
        *,
        user_id: str,
        display_name: str,
        metadata: dict,
    ) -> str:
        artifact_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO artifacts
               (id, task_id, session_id, filename, display_name, format,
                size_bytes, path, metadata, user_id, pinned)
               VALUES (?, '', '', ?, ?, '', 0, '', ?, ?, 1)""",
            (
                artifact_id,
                f"{display_name}.js13k",
                display_name,
                json.dumps(metadata),
                user_id,
            ),
        )
        await self._conn.commit()
        return artifact_id


# ── helpers ──────────────────────────────────────────────────────────


def _hit_to_discovery_item(hit) -> DiscoveryItem:
    """Map js13k provider's GameBrowseResult onto DiscoveryItem."""
    return DiscoveryItem(
        source_id="js13k",
        source_remote_id=hit.source_id,
        kind=KIND_JS13K_GAME,
        title=hit.name,
        runtime_preferred="browser-iframe",
        author=hit.author or "",
        tagline=hit.tagline or "",
        thumbnail_url=hit.thumbnail_url or "",
        source_url=hit.source_url or "",
        embed_url=hit.embed_url or "",
        capabilities={
            "input_modes": ["keyboard", "mouse"],
            "save_states": False,
            "offline": True,
        },
        metadata={
            "play_mode": hit.play_mode,
            "size_bytes": hit.size_bytes,
            "genre": hit.genre,
            **(hit.extra or {}),
        },
    )


def _humanize_slug(slug: str) -> str:
    """Fallback display name when the caller didn't supply one."""
    return js13k_provider._humanize_slug(slug)
