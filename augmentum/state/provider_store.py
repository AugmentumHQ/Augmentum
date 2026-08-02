"""Persistence layer for runtime provider configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key

if TYPE_CHECKING:
    import aiosqlite


@dataclass
class ProviderConfig:
    """Configuration for a runtime-added model provider."""

    id: str
    name: str
    base_url: str
    api_key: str | None = None
    provider_type: str = "openai"
    is_enabled: bool = True
    is_default: bool = False
    # ID of the built-in ProviderProfile this provider follows (e.g. "nvidia",
    # "deepseek"). Drives provider-specific post-processing — without it,
    # NVIDIA-bound requests bypass the "semi" message normalization and 400.
    # Empty string means "no profile selected"; the load path falls back to
    # URL-pattern matching in that case.
    profile_id: str = ""
    # Ownership + sharing (migration 305). owner_user_id="" means
    # admin/server-global; a user id means privately owned. shared=True
    # exposes it to every user on the instance. See the migration header.
    owner_user_id: str = ""
    shared: bool = False
    created_at: str = ""
    updated_at: str = ""


class ProviderStore:
    """CRUD operations for provider configuration in SQLite."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def list_providers(
        self,
        *,
        enabled_only: bool = False,
        visible_to: str | None = None,
    ) -> list[ProviderConfig]:
        """List providers.

        ``enabled_only`` restricts to enabled rows (used at startup to
        decide which backends to register).

        ``visible_to`` (a user id), when provided, filters to providers
        the user may see: shared, admin/server-global (empty owner), or
        owned by that user. Pass ``None`` (the default) to list every
        row regardless of owner — startup load + admin management use
        this; the user-facing list route passes the requester's id.
        """
        clauses: list[str] = []
        params: list = []
        if enabled_only:
            clauses.append("is_enabled = 1")
        if visible_to is not None:
            # Visible = shared OR owned by this user. Deliberately NO
            # empty-owner escape hatch: pre-305 rows were backfilled with
            # owner='' and shared=1, so global providers pass via shared;
            # an unshared ownerless row (legacy Unshare click) stays hidden
            # until an admin re-toggles it (which stamps ownership).
            clauses.append("(shared = 1 OR (owner_user_id != '' AND owner_user_id = ?))")
            params.append(visible_to)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor = await self._conn.execute(
            f"SELECT * FROM providers{where} ORDER BY created_at", params
        )
        rows = await cursor.fetchall()
        return [self._row_to_config(row) for row in rows]

    async def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Get a provider by ID."""
        cursor = await self._conn.execute(
            "SELECT * FROM providers WHERE id = ?", (provider_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_config(row)

    async def create_provider(self, config: ProviderConfig) -> ProviderConfig:
        """Create a new provider."""
        await self._conn.execute(
            """INSERT INTO providers (id, name, base_url, api_key, provider_type,
               is_enabled, is_default, profile_id, owner_user_id, shared)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                config.id,
                config.name,
                config.base_url,
                encrypt_api_key(config.api_key),
                config.provider_type,
                1 if config.is_enabled else 0,
                1 if config.is_default else 0,
                config.profile_id,
                config.owner_user_id,
                1 if config.shared else 0,
            ),
        )
        await self._conn.commit()
        return await self.get_provider(config.id)  # type: ignore[return-value]

    async def update_provider(
        self, provider_id: str, **fields: str | bool | None
    ) -> ProviderConfig | None:
        """Update provider fields."""
        existing = await self.get_provider(provider_id)
        if existing is None:
            return None

        updates = ["updated_at = datetime('now')"]
        params: list = []

        field_map = {
            "name": "name",
            "base_url": "base_url",
            "api_key": "api_key",
            "provider_type": "provider_type",
            "profile_id": "profile_id",
        }
        for key, col in field_map.items():
            if key in fields:
                updates.append(f"{col} = ?")
                val = fields[key]
                if key == "api_key" and val is not None:
                    val = encrypt_api_key(val)
                params.append(val)

        if "is_enabled" in fields:
            updates.append("is_enabled = ?")
            params.append(1 if fields["is_enabled"] else 0)

        if "is_default" in fields:
            updates.append("is_default = ?")
            params.append(1 if fields["is_default"] else 0)

        if "shared" in fields:
            updates.append("shared = ?")
            params.append(1 if fields["shared"] else 0)

        if "owner_user_id" in fields:
            updates.append("owner_user_id = ?")
            params.append(fields["owner_user_id"])

        params.append(provider_id)
        await self._conn.execute(
            f"UPDATE providers SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get_provider(provider_id)

    async def delete_provider(self, provider_id: str) -> bool:
        """Delete a provider. Returns True if a row was deleted."""
        cursor = await self._conn.execute(
            "DELETE FROM providers WHERE id = ?", (provider_id,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_config(row: aiosqlite.Row) -> ProviderConfig:
        """Convert a database row to a ProviderConfig."""
        d = dict(row)
        return ProviderConfig(
            id=d["id"],
            name=d["name"],
            base_url=d["base_url"],
            api_key=decrypt_api_key(d.get("api_key")),
            provider_type=d.get("provider_type", "openai"),
            is_enabled=bool(d.get("is_enabled", 1)),
            is_default=bool(d.get("is_default", 0)),
            profile_id=d.get("profile_id") or "",
            owner_user_id=d.get("owner_user_id") or "",
            shared=bool(d.get("shared", 0)),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
