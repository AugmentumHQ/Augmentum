"""Discovery Engine — history, knowledge, recommendations."""
from __future__ import annotations

import asyncio


def inject_system_context(messages, content: str) -> None:
    """Add ``content`` to the request's system message in-place.

    Folds into the first existing system message rather than prepending a
    second one. Strict chat templates (Gemma 3, several Mistral variants)
    raise "Conversation roles must alternate" when they see two consecutive
    system messages, so injecting a second system block via
    ``insert(-1, system)`` next to the user message produces a 500 from
    llama-server at template-apply time.

    The downstream ``LlamaCppBackend._normalize_system_messages`` coalescer
    catches this as a safety net, but fixing it at the source means
    requests leave handler.py in the shape every backend can render
    (including remote OpenAI-compat backends that don't go through the
    llama.cpp normalizer at all).
    """
    if not messages or not content:
        return
    from augmentum.models.base import Message
    sys_msg = next((m for m in messages if m.role == "system"), None)
    if sys_msg:
        # Two newlines so the model sees a paragraph break between
        # the original system content and the injected context.
        sys_msg.content = (sys_msg.content or "") + f"\n\n{content}"
    else:
        messages.insert(0, Message(role="system", content=content))


async def retrieve_knowledge_context(
    app_state,
    query: str,
    *,
    max_chunks: int = 5,
    min_score: float = 0.65,
    user_id: str = "",
) -> str | None:
    """Retrieve relevant knowledge library chunks for chat context injection.

    Returns formatted context string, or None if nothing relevant found.
    """
    from augmentum.config import settings

    if not settings.knowledge_library_in_chat or not settings.knowledge_library_enabled:
        return None

    store = getattr(app_state, "discovery_store", None)
    if not store:
        return None

    try:
        from augmentum.memory.embeddings import EmbeddingService
        query_vec = await asyncio.to_thread(EmbeddingService.embed_query, query)
        query_blob = EmbeddingService.to_blob(query_vec)
    except Exception:
        return None

    results = await store.search_library(
        query_blob, limit=max_chunks, min_score=min_score, user_id=user_id,
    )
    if not results:
        return None

    # Increment retrieved counts
    for r in results:
        await store.increment_retrieved(r["chunk_id"], user_id=user_id)

    # Format as context block
    lines = ["[From your reading/viewing history]"]
    for r in results:
        source_label = "article" if r["source_type"] == "article" else "video"
        lines.append(f"\nFrom a {source_label} you consumed: {r['source_title']}")
        lines.append(r["content"])

    return "\n".join(lines)
