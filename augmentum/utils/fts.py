"""Shared FTS5 query tokenization for the document and knowledge-pack stores.

Both stores run hybrid retrieval (vector + FTS5 → RRF merge → optional
rerank) over chunks tables that share the same content shape. Keeping the
tokenizer in one place ensures both surfaces apply identical stop-word
handling and AND-first/OR-fallback semantics.
"""
from __future__ import annotations

import re

# English stop words — small list, tuned for technical and encyclopedic
# corpora (Python docs, Wikipedia, medical wikis). Pronouns are included
# because pack queries usually arrive post-condensing, where pronouns
# have already been resolved against the chat history.
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because",
    "but", "and", "or", "if", "while", "what", "which", "who",
    "whom", "this", "that", "these", "those", "i", "me", "my",
    "myself", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
})


def tokenize_fts_query(query: str) -> str | tuple[str, str]:
    """Convert a natural-language query into FTS5 expression(s).

    Returns either a single expression string or a (primary, fallback) tuple.
    Callers should try the primary expression first; if no rows match, retry
    with the fallback. AND queries surface high-precision hits; OR is the
    safety net for queries where strict conjunction is too restrictive.

    Tokenization rules:
    - Strip punctuation, lowercase, remove stop words.
    - 3+ content words → AND-primary, OR-fallback.
    - 1-2 content words → OR only.
    - All stop words → quoted phrase match (handles "to be or not to be"-style
      queries that lose every signal under stop-word removal).
    """
    words = re.findall(r"\w+", query.lower())
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    if not content_words:
        safe = query.replace('"', '""')
        return f'"{safe}"'
    or_expr = " OR ".join(f'"{w}"' for w in content_words)
    if len(content_words) >= 3:
        and_expr = " AND ".join(f'"{w}"' for w in content_words)
        return (and_expr, or_expr)
    return or_expr
