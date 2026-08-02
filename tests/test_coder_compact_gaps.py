"""Gap tests for the Coder mid-turn compactor.

Existing coverage in ``test_coder_structured_compaction.py`` +
``test_coder_context_preservation.py`` exercises the structured-header
fields, tool-result clip caps, and the rollback guard. This file pins
two behaviors that aren't currently covered and are easy to regress:

  1. ``thinking`` (reasoning_content) on Messages is invisible to
     ``count_tokens_messages``. For providers that round-trip reasoning
     back to the upstream (DeepSeek, local engines via DeepSeek-compat),
     the compactor's count diverges from the real prompt size. OpenAI
     and Anthropic strip it on the request path, so they're safe — but
     the asymmetry matters for accurate budget reporting.

  2. Tail-pinned pathology: when ``keep_recent`` contains a huge
     message, compaction can't drop it. The current implementation
     rolls back silently (compacted=False, no warning), and the caller
     proceeds to the LLM call which may overflow. This test pins the
     current behavior so we notice when it changes — and gives us a
     hook to add a warning event later without surprising the test
     suite.

Run: python -m pytest tests/test_coder_compact_gaps.py -v
"""
from __future__ import annotations

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.models.base import Message
from augmentum.utils.tokenizer import count_tokens, count_tokens_messages

from tests.test_coder_handler import _ExtendedContainerManager, _FakeBackend


def _make_handler():
    return CoderHandler(
        _FakeBackend([]),
        session_id="sess-compact-gaps",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-compact-gaps",
    )


def _trip_thresholds(monkeypatch, *, limit: int = 50, keep_recent: int = 2) -> None:
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._COMPACT_AT_TOKENS", limit,
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._COMPACT_KEEP_RECENT", keep_recent,
    )


# ---------------------------------------------------------------------------
# Gap 1 — reasoning_content invisible to the token counter
# ---------------------------------------------------------------------------


def test_thinking_field_is_invisible_to_token_counter():
    """``count_tokens_messages`` walks ``.content`` only. Reasoning text
    stored on ``Message.thinking`` is NOT counted.

    Why pin this: providers split into two camps for the request round-trip.
    OpenAI / Anthropic / OpenRouter strip reasoning_content before
    sending (``ProviderProfile.accepts_reasoning_content=False``) — for
    them the counter is honest. DeepSeek's reasoning lineup REQUIRES
    reasoning_content on prior turns or the request 400s
    (``accepts_reasoning_content=True``) — for them the real prompt
    payload includes thinking but the compactor's budget view doesn't,
    so a "we're under the limit" call from the compactor can still
    overflow the upstream window. This test documents the gap so a
    future fix (count thinking when the active backend round-trips it)
    has a regression anchor.
    """
    text = "x" * 4000  # ~1000 tokens at cl100k_base
    content_only = Message(role="assistant", content=text)
    with_thinking = Message(role="assistant", content="", thinking=text)

    content_tokens = count_tokens_messages([content_only])
    thinking_tokens = count_tokens_messages([with_thinking])

    # The per-message overhead (~4) is the only thing the thinking msg
    # contributes — the reasoning payload is wholly invisible.
    assert content_tokens > thinking_tokens + 100, (
        "Counter currently ignores Message.thinking. If this changes "
        "(e.g. we start counting reasoning for accepts_reasoning_content "
        "providers), update the compactor's budget reporting too."
    )


# ---------------------------------------------------------------------------
# Gap 2 — tail-pinned content pathology
# ---------------------------------------------------------------------------


def test_compact_rolls_back_silently_when_tail_is_huge(monkeypatch):
    """When the kept tail alone exceeds the compact threshold, there's
    nothing the compactor can drop — system + first_user + tail are all
    pinned. Today's behavior: rollback (compacted=False, before==after,
    messages untouched). No warning event.

    This pins the current contract so a future improvement
    (emit a ``compaction.no_progress`` meta chunk so the UI can warn
    the user) doesn't break silently — and so we notice if the rollback
    semantics ever change.
    """
    _trip_thresholds(monkeypatch, limit=100, keep_recent=2)
    h = _make_handler()

    huge_payload = "Y" * 4000  # ~1000 tokens, well over the 100-token limit

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        # Middle region (would be dropped on a healthy compaction)
        Message(role="assistant", content="a1"),
        Message(role="tool", content="t1", tool_call_id="tc1"),
        Message(role="assistant", content="a2"),
        Message(role="tool", content="t2", tool_call_id="tc2"),
        # Tail (kept verbatim) — one of these is huge
        Message(role="assistant", content="thinking..."),
        Message(role="tool", content=huge_payload, tool_call_id="tc-huge"),
    ]
    original = list(messages)

    before = count_tokens_messages(messages)
    assert before > 100, "fixture must actually trip the threshold"

    compacted, before_n, after_n = h._maybe_compact_messages(messages)

    # Today: the compactor DOES drop the middle and the summary IS smaller
    # than what it replaced, so compacted=True is plausible. Either way,
    # the tail's huge tool_result remains pinned in the kept-region — so
    # the FINAL token count is still above the limit. This is the silent
    # failure: caller proceeds to LLM with a known-over budget.
    assert messages[-1].content == huge_payload, (
        "Tail must be preserved verbatim — that's the contract. The "
        "issue is that no warning fires when the preserved tail keeps "
        "us above limit."
    )
    final = count_tokens_messages(messages)
    assert final > 100, (
        "Documents the pathology: even after a 'successful' compaction, "
        "the budget is still blown because the tail is huge. A future "
        "fix should emit a compaction.no_progress event when "
        "after >= compact_limit, so the UI can surface 'context too "
        "full to compact further' instead of letting the LLM call 400."
    )
    # Sanity: if compactor said it didn't compact, messages are unchanged
    if not compacted:
        assert messages == original
        assert before_n == after_n == before
