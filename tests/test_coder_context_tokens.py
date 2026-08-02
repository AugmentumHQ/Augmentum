from __future__ import annotations

import pytest

from augmentum.coder.context_tokens import (
    compact_conversation_messages,
    coder_context_token_limit,
    coder_digest_token_budget,
    derive_coder_context_token_limit,
    derive_coder_digest_token_budget,
    token_budget_payload,
)
from augmentum.models.base import InternalChatRequest, InternalStreamChunk, Message
from augmentum.modes.coder.handler import CoderHandler


def test_coder_context_limit_unknown_keeps_legacy_default(monkeypatch):
    monkeypatch.delenv("AUGMENTUM_CODER_COMPACT_TOKENS", raising=False)
    assert coder_context_token_limit() == 16_000


def test_coder_context_limit_scales_with_model_window(monkeypatch):
    monkeypatch.delenv("AUGMENTUM_CODER_COMPACT_TOKENS", raising=False)
    limit = derive_coder_context_token_limit(261_376)
    # Policy 2026-05-31: 10% reserve, 100% usable utilization →
    # ~90% of window. For 261_376: reserve ≈ 26_137 (under 32K cap),
    # usable ≈ 235_239. Allow a small range to absorb future minor
    # tuning without churn.
    assert 230_000 <= limit <= 240_000
    assert coder_context_token_limit(261_376) == limit


def test_coder_context_limit_env_override_wins(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_CODER_COMPACT_TOKENS", "12345")
    assert coder_context_token_limit(261_376) == 12_345


def test_coder_digest_budget_scales_below_compaction_limit(monkeypatch):
    monkeypatch.delenv("AUGMENTUM_CODER_DIGEST_BUDGET", raising=False)
    digest = derive_coder_digest_token_budget(261_376)
    compact = derive_coder_context_token_limit(261_376)
    assert 75_000 <= digest <= 85_000
    assert digest < compact
    assert coder_digest_token_budget(261_376) == digest


def test_coder_digest_budget_env_override_wins(monkeypatch):
    monkeypatch.setenv("AUGMENTUM_CODER_DIGEST_BUDGET", "55555")
    assert coder_digest_token_budget(261_376) == 55_555


def test_compact_conversation_preserves_first_user_and_recent_tail():
    messages = [{"role": "user", "content": "start"}]
    for i in range(18):
        messages.append({
            "role": "assistant",
            "content": f"assistant {i} " + ("alpha beta gamma " * 80),
        })
        messages.append({
            "role": "user",
            "content": f"user {i} " + ("delta epsilon zeta " * 80),
        })

    result = compact_conversation_messages(
        messages,
        keep_recent=6,
        force=True,
    )

    assert result.compacted is True
    assert result.messages[0]["content"] == "start"
    assert result.messages[1]["role"] == "assistant"
    assert "<compacted" in result.messages[1]["content"]
    assert result.messages[-1]["content"].startswith("user 17")
    assert len(result.messages) == 8
    assert result.tokens_after < result.tokens_before


def test_compact_conversation_extends_existing_block_append_only():
    """Re-compaction must EXTEND the existing <compacted> block, never
    re-render it. Re-rendering rewrote the head of history (llama-server
    prefix-cache kill, measured 2026-07-02 stable_pct 0.13) and crushed
    the whole prior summary into one one-line entry."""
    messages = [{"role": "user", "content": "start"}]
    for i in range(18):
        messages.append({
            "role": "assistant",
            "content": f"assistant {i} " + ("alpha beta gamma " * 80),
        })
        messages.append({
            "role": "user",
            "content": f"user {i} " + ("delta epsilon zeta " * 80),
        })

    first = compact_conversation_messages(messages, keep_recent=6, force=True)
    assert first.compacted is True
    block_v1 = first.messages[1]["content"]

    # Conversation grows past the block again.
    grown = list(first.messages)
    for i in range(18, 30):
        grown.append({
            "role": "assistant",
            "content": f"assistant {i} " + ("alpha beta gamma " * 80),
        })
        grown.append({
            "role": "user",
            "content": f"user {i} " + ("delta epsilon zeta " * 80),
        })

    second = compact_conversation_messages(grown, keep_recent=6, force=True)
    assert second.compacted is True
    block_v2 = second.messages[1]["content"]

    # Byte-prefix preserved: v2 begins with all of v1 minus the closer.
    v1_prefix = block_v1[: -len("</compacted>")].rstrip("\n")
    assert block_v2.startswith(v1_prefix)
    # Two segments, one wrapper — never a nested <compacted>.
    assert block_v2.count("<compacted") == 1
    assert block_v2.count("## Condensed segment") == 2
    assert block_v2.endswith("</compacted>")


def test_compact_conversation_skips_runtime_carrier_anchor():
    """The preserved 'first user' anchor must be the real task, not a
    runtime carrier riding ahead of it."""
    from augmentum.modes.coder.chat_egress import RUNTIME_CARRIER_HEADER

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"{RUNTIME_CARRIER_HEADER}\nstale state"},
        {"role": "user", "content": "the real task"},
    ]
    for i in range(18):
        messages.append({
            "role": "assistant",
            "content": f"assistant {i} " + ("alpha beta gamma " * 80),
        })
        messages.append({
            "role": "user",
            "content": f"user {i} " + ("delta epsilon zeta " * 80),
        })

    result = compact_conversation_messages(messages, keep_recent=6, force=True)
    assert result.compacted is True
    # Carrier and task both survive verbatim ahead of the block.
    assert result.messages[1]["content"].startswith(RUNTIME_CARRIER_HEADER)
    assert result.messages[2]["content"] == "the real task"
    assert "<compacted" in result.messages[3]["content"]
    # The stale carrier is never condensed into the block body.
    assert "stale state" not in result.messages[3]["content"]


def test_compact_conversation_noops_when_too_short():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = compact_conversation_messages(messages, force=True)

    assert result.compacted is False
    assert result.messages == messages


def test_token_budget_payload_has_stable_shape():
    payload = token_budget_payload(
        [{"role": "user", "content": "hello"}],
        scope="incoming_request",
        limit=1000,
        iteration=3,
    )

    assert payload["scope"] == "incoming_request"
    assert payload["tokens"] > 0
    assert payload["limit"] == 1000
    assert payload["iteration"] == 3
    assert "ratio" in payload


@pytest.mark.asyncio
async def test_handler_budget_metadata_uses_backend_context_window(monkeypatch):
    monkeypatch.delenv("AUGMENTUM_CODER_COMPACT_TOKENS", raising=False)

    class _ContextBackend:
        async def get_context_length(self, model):
            assert model == "big-model"
            return 65_536

        async def chat_stream(self, request):
            yield InternalStreamChunk(
                content_delta="",
                model=request.model,
                done=True,
                finish_reason="stop",
            )

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ContextBackend(),
        session_id="sess-budget-window",
        container_manager=object(),
        workspace_id="ws-budget-window",
    )
    request = InternalChatRequest(
        model="big-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )

    chunks = [chunk async for chunk in handler._handle_stream_body(request)]
    budget = chunks[0].augmentum["tokens"]
    assert budget["context_window"] == 65_536
    assert budget["limit"] == derive_coder_context_token_limit(65_536)


@pytest.mark.asyncio
async def test_backend_compact_command_short_circuits_model():
    class _NoModelCalls:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            raise AssertionError("/compact should not reach the model")

        async def chat(self, request):
            return None

    messages = [Message(role="user", content="start")]
    for i in range(18):
        messages.append(Message(
            role="assistant",
            content=f"assistant {i} " + ("alpha beta gamma " * 80),
        ))
        messages.append(Message(
            role="user",
            content=f"user {i} " + ("delta epsilon zeta " * 80),
        ))
    messages.append(Message(role="user", content="/compact"))

    backend = _NoModelCalls()
    handler = CoderHandler(
        backend,
        session_id="sess-compact",
        container_manager=object(),
        workspace_id="ws-compact",
    )
    request = InternalChatRequest(
        model="test-model",
        messages=messages,
        stream=True,
    )

    chunks = [chunk async for chunk in handler._handle_stream_body(request)]
    statuses = [
        chunk.augmentum.get("status")
        for chunk in chunks
        if chunk.augmentum
    ]

    assert backend.calls == 0
    assert statuses[0] == "budget"
    assert "compaction" in statuses
    assert statuses[-1] == "complete"
    assert chunks[-1].augmentum["manual_compact"] is True
