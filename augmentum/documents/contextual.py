"""Contextual retrieval — LLM-generated chunk context at ingest time.

Implements Anthropic's contextual retrieval pattern: for each chunk,
an LLM generates a short (50-100 token) preamble that situates the
chunk within the full document. This preamble is prepended to the
chunk text before embedding and FTS indexing, dramatically improving
retrieval precision.

Reference: https://www.anthropic.com/news/contextual-retrieval

Performance (Anthropic benchmarks):
- Contextual embeddings alone: -35% retrieval failure
- + BM25: -49% failure
- + Reranking: -67% failure
"""

from __future__ import annotations

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_CONTEXT_PROMPT = """\
<document>
{document_text}
</document>

Here is the chunk we want to situate within the overall document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this \
chunk within the overall document for the purposes of improving search \
retrieval of this chunk. Answer only with the context, nothing else."""


async def generate_chunk_contexts(
    chunks: list[str],
    document_text: str,
    backend: object | None = None,
) -> list[str]:
    """Generate contextual preambles for a list of chunks.

    Uses the configured LLM backend to generate a short context for
    each chunk that describes how it fits within the full document.

    Returns a list of context strings (same length as chunks).
    Empty strings are returned for chunks that fail.
    """
    if backend is None:
        return [""] * len(chunks)

    # Truncate document to avoid exceeding context window
    max_doc_chars = 30_000  # ~7500 tokens — fits in most context windows
    doc_preview = document_text[:max_doc_chars]
    if len(document_text) > max_doc_chars:
        doc_preview += "\n\n[... document truncated ...]"

    contexts: list[str] = []
    for chunk_text in chunks:
        try:
            prompt = _CONTEXT_PROMPT.format(
                document_text=doc_preview,
                chunk_text=chunk_text,
            )
            # Use backend.chat() with a simple single-message request
            from augmentum.models.base import Message

            response = await backend.chat(
                messages=[Message(role="user", content=prompt)],
                system_prompt="You are a helpful assistant that provides concise context for document chunks.",
                temperature=0.0,
                max_tokens=150,
            )
            context = response.content.strip() if response and response.content else ""
            contexts.append(context)
        except Exception:
            log.debug("contextual_generation_failed", chunk_idx=len(contexts), exc_info=True)
            contexts.append("")

    generated = sum(1 for c in contexts if c)
    log.info("chunk_contexts_generated", total=len(chunks), success=generated)
    return contexts


def prepend_context(chunk_text: str, context: str) -> str:
    """Prepend LLM-generated context to chunk text.

    The combined text is what gets embedded and indexed for search.
    """
    if not context:
        return chunk_text
    return f"{context}\n\n{chunk_text}"
