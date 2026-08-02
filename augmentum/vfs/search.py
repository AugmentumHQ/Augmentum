"""Unified federated search across file index, memory, and documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.vfs.models import SearchResult

if TYPE_CHECKING:
    from augmentum.vfs.index import FileIndexService

log = get_logger(__name__)


async def unified_search(
    query: str,
    *,
    user_id: str,
    file_index: FileIndexService,
    sources: list[str] | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    """Search across file index. Extensible to memory + documents later."""
    results: list[SearchResult] = []

    if not sources or "files" in sources:
        file_results = await file_index.search(query, user_id=user_id, limit=limit)
        for f in file_results:
            results.append(SearchResult(
                source="file", item=f, score=f.score,
                card=f.to_card("card"),
            ))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
