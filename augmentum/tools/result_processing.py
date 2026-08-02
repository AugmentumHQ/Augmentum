"""Tool result post-processing — truncation, formatting."""

from __future__ import annotations


def truncate_tool_result(
    text: str,
    max_chars: int = 4000,
    tail_chars: int = 500,
) -> str:
    """Truncate long tool output, keeping head and tail for context.

    Returns the original text if it fits within *max_chars*.  Otherwise
    keeps the first ``max_chars - tail_chars - notice_len`` characters,
    appends a truncation notice, then appends the last *tail_chars*.
    """
    if not text or len(text) <= max_chars:
        return text

    # Ensure tail_chars doesn't exceed half the budget
    tail_chars = min(tail_chars, max_chars // 3)

    notice = "\n\n[... {n} characters truncated ...]\n\n"
    # Calculate how much room the notice takes (with placeholder replaced)
    truncated_middle = len(text) - (max_chars - len(notice) + 10)
    filled_notice = notice.replace("{n}", str(truncated_middle))

    head_budget = max_chars - tail_chars - len(filled_notice)
    if head_budget < 100:
        # Not enough room for meaningful head — just hard truncate
        return text[:max_chars]

    head = text[:head_budget]
    tail = text[-tail_chars:] if tail_chars > 0 else ""

    # Recalculate actual truncated count
    actual_truncated = len(text) - len(head) - len(tail)
    filled_notice = notice.replace("{n}", str(actual_truncated))

    return head + filled_notice + tail
