"""Persistence for discovered media-server libraries/views."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.media.library_classification import (
    classify_library,
)
from augmentum.media.providers.base import LibraryView

if TYPE_CHECKING:
    import aiosqlite


_library_store: MediaLibraryStore | None = None


def set_media_library_store(store: MediaLibraryStore) -> None:
    global _library_store
    _library_store = store


def get_media_library_store() -> MediaLibraryStore | None:
    return _library_store


@dataclass(slots=True)
class MediaLibraryRecord:
    id: str
    user_id: str
    server_id: str
    provider: str
    provider_library_id: str
    provider_name: str
    provider_view_type: str
    provider_collection_type: str
    detected_group: str
    detected_primary_entity: str
    detection_confidence: float
    sample_type_counts: dict[str, int] = field(default_factory=dict)
    sample_notes: dict = field(default_factory=dict)
    display_name_override: str = ""
    surface_group_override: str = ""
    is_hidden: bool = False
    include_in_search: bool = True
    include_in_overview: bool = True
    sort_order: int = 0
    last_seen_at: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def display_name(self) -> str:
        return self.display_name_override or self.provider_name

    @property
    def surface_group(self) -> str:
        return self.surface_group_override or self.detected_group

    @property
    def needs_review(self) -> bool:
        return self.detection_confidence < 0.75

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "server_id": self.server_id,
            "provider": self.provider,
            "provider_library_id": self.provider_library_id,
            "provider_name": self.provider_name,
            "provider_view_type": self.provider_view_type,
            "provider_collection_type": self.provider_collection_type,
            "detected_group": self.detected_group,
            "detected_primary_entity": self.detected_primary_entity,
            "detection_confidence": self.detection_confidence,
            "sample_type_counts": self.sample_type_counts,
            "sample_notes": self.sample_notes,
            "display_name_override": self.display_name_override,
            "surface_group_override": self.surface_group_override,
            "display_name": self.display_name,
            "surface_group": self.surface_group,
            "is_hidden": self.is_hidden,
            "include_in_search": self.include_in_search,
            "include_in_overview": self.include_in_overview,
            "sort_order": self.sort_order,
            "needs_review": self.needs_review,
            "last_seen_at": self.last_seen_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MediaLibraryStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def upsert_discovered(
        self,
        *,
        user_id: str,
        server_id: str,
        provider: str,
        libraries: list[LibraryView],
    ) -> list[MediaLibraryRecord]:
        for idx, library in enumerate(libraries):
            classified = classify_library(
                collection_type=library.collection_type,
                sample_type_counts=library.sample_type_counts,
                view_type=library.view_type,
            )
            existing = await self.get_by_provider_id(
                user_id=user_id,
                server_id=server_id,
                provider_library_id=library.external_id,
            )
            record_id = existing.id if existing else f"mlv_{secrets.token_hex(8)}"
            await self._conn.execute(
                """
                INSERT INTO media_library_views (
                    id, user_id, server_id, provider, provider_library_id,
                    provider_name, provider_view_type, provider_collection_type,
                    detected_group, detected_primary_entity, detection_confidence,
                    sample_type_counts, sample_notes, sort_order, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(user_id, server_id, provider_library_id) DO UPDATE SET
                    provider_name = excluded.provider_name,
                    provider_view_type = excluded.provider_view_type,
                    provider_collection_type = excluded.provider_collection_type,
                    detected_group = excluded.detected_group,
                    detected_primary_entity = excluded.detected_primary_entity,
                    detection_confidence = excluded.detection_confidence,
                    sample_type_counts = excluded.sample_type_counts,
                    sample_notes = excluded.sample_notes,
                    last_seen_at = datetime('now'),
                    updated_at = datetime('now')
                """,
                (
                    record_id,
                    user_id,
                    server_id,
                    provider,
                    library.external_id,
                    library.name,
                    library.view_type,
                    library.collection_type,
                    classified.detected_group,
                    classified.detected_primary_entity,
                    classified.detection_confidence,
                    json.dumps(classified.sample_type_counts),
                    json.dumps(classified.sample_notes),
                    idx,
                ),
            )
        await self._conn.commit()
        return await self.list_for_server(user_id=user_id, server_id=server_id)

    async def get(self, record_id: str, *, user_id: str) -> MediaLibraryRecord | None:
        cursor = await self._conn.execute(
            "SELECT * FROM media_library_views WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_record(row) if row else None

    async def get_by_provider_id(
        self,
        *,
        user_id: str,
        server_id: str,
        provider_library_id: str,
    ) -> MediaLibraryRecord | None:
        cursor = await self._conn.execute(
            "SELECT * FROM media_library_views "
            "WHERE user_id = ? AND server_id = ? AND provider_library_id = ?",
            (user_id, server_id, provider_library_id),
        )
        row = await cursor.fetchone()
        return _row_to_record(row) if row else None

    async def list_for_server(
        self, *, user_id: str, server_id: str,
    ) -> list[MediaLibraryRecord]:
        cursor = await self._conn.execute(
            "SELECT * FROM media_library_views "
            "WHERE user_id = ? AND server_id = ? "
            "ORDER BY sort_order ASC, provider_name COLLATE NOCASE ASC",
            (user_id, server_id),
        )
        return [_row_to_record(row) for row in await cursor.fetchall()]

    async def active_by_provider_id(
        self, *, user_id: str, server_id: str,
    ) -> dict[str, MediaLibraryRecord]:
        rows = await self.list_for_server(user_id=user_id, server_id=server_id)
        return {row.provider_library_id: row for row in rows}

    async def update(
        self,
        record_id: str,
        *,
        user_id: str,
        display_name_override: str | None = None,
        surface_group_override: str | None = None,
        is_hidden: bool | None = None,
        include_in_search: bool | None = None,
        include_in_overview: bool | None = None,
        sort_order: int | None = None,
    ) -> MediaLibraryRecord | None:
        fields: list[str] = []
        params: list[object] = []
        if display_name_override is not None:
            fields.append("display_name_override = ?")
            params.append(display_name_override.strip())
        if surface_group_override is not None:
            fields.append("surface_group_override = ?")
            params.append(surface_group_override.strip())
        if is_hidden is not None:
            fields.append("is_hidden = ?")
            params.append(1 if is_hidden else 0)
        if include_in_search is not None:
            fields.append("include_in_search = ?")
            params.append(1 if include_in_search else 0)
        if include_in_overview is not None:
            fields.append("include_in_overview = ?")
            params.append(1 if include_in_overview else 0)
        if sort_order is not None:
            fields.append("sort_order = ?")
            params.append(int(sort_order))
        if not fields:
            return await self.get(record_id, user_id=user_id)
        fields.append("updated_at = datetime('now')")
        params.extend([record_id, user_id])
        await self._conn.execute(
            f"UPDATE media_library_views SET {', '.join(fields)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get(record_id, user_id=user_id)


def _row_to_record(row) -> MediaLibraryRecord:
    def _json_load(raw: object, fallback):
        try:
            parsed = json.loads(raw or "")
            return parsed if isinstance(parsed, type(fallback)) else fallback
        except (TypeError, ValueError):
            return fallback

    return MediaLibraryRecord(
        id=row[0],
        user_id=row[1],
        server_id=row[2],
        provider=row[3],
        provider_library_id=row[4],
        provider_name=row[5],
        provider_view_type=row[6],
        provider_collection_type=row[7],
        detected_group=row[8],
        detected_primary_entity=row[9],
        detection_confidence=float(row[10] or 0.0),
        sample_type_counts=_json_load(row[11], {}),
        sample_notes=_json_load(row[12], {}),
        display_name_override=row[13] or "",
        surface_group_override=row[14] or "",
        is_hidden=bool(row[15]),
        include_in_search=bool(row[16]),
        include_in_overview=bool(row[17]),
        sort_order=int(row[18] or 0),
        last_seen_at=row[19],
        created_at=row[20] or "",
        updated_at=row[21] or "",
    )
