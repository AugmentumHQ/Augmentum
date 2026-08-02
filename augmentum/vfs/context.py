"""File context builder — generates tiered AI-ready file references."""

from __future__ import annotations

from augmentum.vfs.models import FileEntry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Budget limits
MAX_FILE_CARDS = 10
MAX_FILE_TOKENS_CARD = 60       # ~tokens per card-only reference
MAX_FILE_TOKENS_SUMMARY = 350   # ~tokens per card+summary
MAX_FILE_TOKENS_TOTAL = 1500    # Total budget for file context


def detect_tier(context_length: int, has_vision: bool = False) -> str:
    """Determine file context tier based on model capabilities.

    Returns: "card", "card_summary", or "card_content"
    """
    if context_length < 8192:
        return "card"
    elif context_length < 32768:
        return "card_summary"
    else:
        return "card_content"


def build_file_context(entries: list[FileEntry], tier: str, remaining_budget: int = 0) -> str:
    """Build file context string from entries at the specified tier.

    Args:
        entries: File entries to include (already ranked by relevance)
        tier: "card", "card_summary", or "card_content"
        remaining_budget: Approximate tokens remaining in context (0 = unlimited)
    """
    if not entries:
        return ""

    # If context is very tight, force card-only regardless of tier
    if remaining_budget and remaining_budget < 2000:
        tier = "card"

    parts = []
    token_estimate = 0
    max_per_card = MAX_FILE_TOKENS_CARD if tier == "card" else MAX_FILE_TOKENS_SUMMARY

    for entry in entries[:MAX_FILE_CARDS]:
        card = entry.to_card(tier)
        card_tokens = len(card.split()) * 1.3  # rough estimate
        if token_estimate + card_tokens > MAX_FILE_TOKENS_TOTAL:
            break
        parts.append(card)
        token_estimate += card_tokens

    if not parts:
        return ""

    header = f"[{len(parts)} relevant file(s) found]"
    return header + "\n\n" + "\n\n".join(parts)


def build_single_card(entry: FileEntry) -> str:
    """Build a single file card for inline chat reference."""
    return entry.to_card("card")
