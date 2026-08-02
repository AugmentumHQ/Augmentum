"""File source adapters — pluggable backends for the unified file index.

Each adapter owns one `source` slug on `file_index` and knows how to:
  - resolve(source_id) → filesystem path or bytes for that row
  - list_source_ids(user_id) → every live source_id owned by this adapter
  - delete(source_id, user_id) → drop the backing row + release any blobs

Everywhere else — download, render, reconcile, purge — loops over the
registry instead of hand-written switch statements. Adding a new backend
(Dropbox sync, S3, etc.) means writing one adapter + registering at
startup. No other file changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FileSourceAdapter(Protocol):
    name: str  # stable source slug — matches file_index.source

    async def resolve(self, source_id: str, *, user_id: str) -> str | bytes | None:
        """Return a filesystem path (str), raw bytes, or None if not found."""
        ...

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        """Live source_ids for this user — used by reconcile sweeps."""
        ...

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        """Permanently remove this row (and release any shared blob)."""
        ...


# Module-level registry. Populated at startup by server.py.
_registry: dict[str, FileSourceAdapter] = {}


def register_adapter(adapter: FileSourceAdapter) -> None:
    _registry[adapter.name] = adapter


def get_adapter(source: str) -> FileSourceAdapter | None:
    return _registry.get(source)


def all_adapters() -> dict[str, FileSourceAdapter]:
    return dict(_registry)
