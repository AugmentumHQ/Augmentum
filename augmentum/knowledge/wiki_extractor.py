"""Backward-compat shim for the old ``wiki_extractor`` module.

The extractor was renamed to ``content_extractor`` in Phase 4.5 because it
now handles agnostic sources (any URL, not just wikis). This module re-
exports the old names so existing imports continue to work.
"""

from __future__ import annotations

from augmentum.knowledge.content_extractor import (
    ContentDoc,
    ContentExtractError,
    Link,
    WikiContext,
    WikiExtractError,
    clear_content_cache,
    clear_wiki_cache,
    fetch_content_doc,
    fetch_path,
    fetch_wiki_context,
)

__all__ = [
    "ContentDoc",
    "ContentExtractError",
    "Link",
    "WikiContext",
    "WikiExtractError",
    "clear_content_cache",
    "clear_wiki_cache",
    "fetch_content_doc",
    "fetch_path",
    "fetch_wiki_context",
]
