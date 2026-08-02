"""Virtual filesystem, file index, and unified search."""

from __future__ import annotations

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_file_index = None
_thumbnail_service = None


def set_file_index(index) -> None:
    global _file_index
    _file_index = index


def set_thumbnail_service(service) -> None:
    """Install the process-wide thumbnail service.

    Mirrors ``set_file_index`` so adapter delete paths can purge cached
    thumbs without having to thread the service through their
    constructors.
    """
    global _thumbnail_service
    _thumbnail_service = service


def purge_thumbnails(source: str, source_id: str) -> int:
    """Drop every cached thumbnail for one source row. Safe no-op if no
    service is installed (tests, stripped-down configs).
    """
    if _thumbnail_service is None:
        return 0
    try:
        return _thumbnail_service.purge(source, source_id)
    except Exception:
        log.warning("thumbnail_purge_failed", source=source, source_id=source_id)
        return 0


def file_index_is_configured() -> bool:
    """Return True if an index has been installed via ``set_file_index``.

    Callers that need to distinguish "no file index configured" (benign,
    nothing to register) from "registration attempted and failed" (bug,
    surface to user) should check this before calling ``register_file``.
    """
    return _file_index is not None


async def register_file(**kwargs) -> str | None:
    """Safe registration — silently skips if file index isn't initialized.

    Returns the registered file id on success, ``None`` either when no
    index is configured OR when ``_file_index.register`` raised. Use
    :func:`file_index_is_configured` to tell those two cases apart.
    """
    if not _file_index:
        return None
    try:
        return await _file_index.register(**kwargs)
    except Exception:
        log.warning("file_register_failed", exc_info=True)
        return None


async def unregister_file(source: str, source_id: str, *, user_id: str) -> bool:
    if not _file_index:
        return False
    try:
        return await _file_index.unregister(source, source_id, user_id=user_id)
    except Exception:
        return False


# Adapter registry — re-export so callers can `from augmentum.vfs import ...`
# without having to know about the adapters subpackage.
from augmentum.vfs.adapters import (  # noqa: E402
    FileSourceAdapter,
    all_adapters,
    get_adapter,
    register_adapter,
)

__all__ = [
    "set_file_index", "register_file", "unregister_file",
    "file_index_is_configured",
    "set_thumbnail_service", "purge_thumbnails",
    "FileSourceAdapter", "register_adapter", "get_adapter", "all_adapters",
]
