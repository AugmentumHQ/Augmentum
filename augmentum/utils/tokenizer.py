"""Token counting utility — lazy-loaded tiktoken for accurate token budgets.

Uses cl100k_base encoding (GPT-4 family, ~100K vocab).  For Llama/Mistral
models the counts are ~15-20% different, which is acceptable for context
budget management — users set approximate limits anyway.

Falls back to a character-based estimate (~4 chars/token) if tiktoken
fails to load (e.g. missing dependency, offline with no cached encoding).
"""

from __future__ import annotations

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_encoding = None
_fallback = False


def _get_encoding():
    """Lazy-load the tiktoken encoding on first use."""
    global _encoding, _fallback
    if _encoding is not None or _fallback:
        return _encoding
    try:
        import tiktoken
        _encoding = tiktoken.get_encoding("cl100k_base")
        log.info("tiktoken_loaded", encoding="cl100k_base")
    except Exception:
        _fallback = True
        log.warning("tiktoken_unavailable_using_char_estimate")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in a string.  Fast path for empty/short strings."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Fallback: ~4 chars per token
    return len(text) // 4 + 1


def count_tokens_messages(messages: list[dict | object]) -> int:
    """Count total tokens across a list of messages.

    Accepts both dicts (``{"content": "..."}``}) and objects with a
    ``.content`` attribute (``Message`` instances).
    """
    total = 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if content:
            total += count_tokens(content)
        # Per-message overhead (role, separators) — ~4 tokens per message
        total += 4
    return total
