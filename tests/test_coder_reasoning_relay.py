"""Live reasoning relay + thinking-control wiring (2026-07-02).

Covers the two halves of the coder thinking-toggle fix:

**Visibility** — the act loop used to swallow ``thinking_delta`` into a
server-side buffer; the user saw only a "reasoning…" status label. Now
``_stream_and_parse`` relays coalesced ``reasoning_delta`` chunks through
``progress_out`` (batched, never dropped), ``_stream_and_parse_live``
surfaces them WHILE the model generates, and the turn ledger skips them
so a long reasoning burst never becomes per-chunk SQLite commits.

**Control** — the coder composer's toggle sends
``chat_template_kwargs={"enable_thinking": ...}`` in the ``/api/chat``
body; that field was silently dropped at the ingress boundary (the
explicit-field-list trap from ``models/base.py:60``) and, separately,
ignored by every cloud provider's reasoning emitter (which keyed off
``request.think`` alone). Both routes now map it and
``openai_compat._effective_think`` folds it over ``request.think``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.analytical.tool_calling import ToolCallingTier
from augmentum.modes.coder.chat_egress import ReasoningRelay
from augmentum.modes.coder.handler import CoderHandler
from augmentum.modes.coder.phase_act import _iteration_thinking_kwargs


@dataclass
class _FakeChunk:
    content_delta: str = ""
    thinking_delta: str = ""
    role: str | None = None
    finish_reason: str | None = None
    usage: Any = None
    model: str = ""
    done: bool = False
    augmentum: dict | None = None


class _FakeBackend:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.last_request: InternalChatRequest | None = None

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[_FakeChunk]:
        self.last_request = request
        for c in self._chunks:
            yield c

    async def chat(self, request: InternalChatRequest):
        return None


def _handler(backend: _FakeBackend) -> CoderHandler:
    return CoderHandler(
        backend,
        session_id="ws-test",
        workspace_id="ws-test",
        container_manager=None,
        user_id="alice",
    )


def _req(**kw) -> InternalChatRequest:
    kw.setdefault("model", "test-model")
    return InternalChatRequest(
        messages=[Message(role="user", content="hi")],
        stream=True,
        **kw,
    )


def _thinking_then_prose_chunks() -> list[_FakeChunk]:
    return [
        _FakeChunk(thinking_delta="I should look at "),
        _FakeChunk(thinking_delta="the config first. "),
        _FakeChunk(thinking_delta="Then edit the route."),
        _FakeChunk(content_delta="Reading the config now."),
        _FakeChunk(done=True, finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# ReasoningRelay coalescer
# ---------------------------------------------------------------------------


def test_relay_never_drops_text():
    relay = ReasoningRelay(phase="executing", model="m", min_chars=8)
    pieces = ["alpha ", "beta ", "g", "amma ", "delta"]
    out: list[str] = []
    for p in pieces:
        ev = relay.add(p)
        if ev is not None:
            out.append(ev.thinking_delta)
    final = relay.flush()
    if final is not None:
        out.append(final.thinking_delta)
    assert "".join(out) == "".join(pieces)


def test_relay_batches_below_threshold():
    relay = ReasoningRelay(
        phase="executing", model="m", min_chars=1000, max_latency_s=999,
    )
    assert relay.add("tiny") is None
    assert relay.add("bits") is None
    ev = relay.flush()
    assert ev is not None
    assert ev.thinking_delta == "tinybits"
    assert ev.augmentum["status"] == "reasoning_delta"
    assert ev.augmentum["mode"] == "coder"


def test_relay_flush_empty_returns_none():
    relay = ReasoningRelay(phase="planning", model="m")
    assert relay.flush() is None


# ---------------------------------------------------------------------------
# _stream_and_parse relays reasoning through progress_out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_and_parse_relays_full_reasoning():
    h = _handler(_FakeBackend(_thinking_then_prose_chunks()))
    progress: list = []
    full_content, _, error_kind, full_thinking, _, _ = await h._stream_and_parse(
        _req(), messages=[], tool_schemas=[], tool_map={},
        tier=ToolCallingTier.NATIVE, iteration=1,
        progress_out=progress,
    )
    assert error_kind == ""
    relayed = "".join(
        c.thinking_delta for c in progress
        if (c.augmentum or {}).get("status") == "reasoning_delta"
    )
    # Everything the model thought reaches the client — batched, never cut.
    assert relayed == "I should look at the config first. Then edit the route."
    assert full_thinking == relayed
    assert full_content == "Reading the config now."
    # Transition markers still present (thinking → responding).
    statuses = [(c.augmentum or {}).get("status") for c in progress]
    assert "thinking" in statuses
    assert "responding" in statuses
    # Ordering: all reasoning text lands before the responding transition.
    responding_i = statuses.index("responding")
    assert all(
        i < responding_i
        for i, s in enumerate(statuses) if s == "reasoning_delta"
    )


@pytest.mark.asyncio
async def test_stream_and_parse_no_progress_out_still_works():
    h = _handler(_FakeBackend(_thinking_then_prose_chunks()))
    full_content, _, error_kind, full_thinking, _, _ = await h._stream_and_parse(
        _req(), messages=[], tool_schemas=[], tool_map={},
        tier=ToolCallingTier.NATIVE, iteration=1,
    )
    assert error_kind == ""
    assert full_thinking.startswith("I should look at")
    assert full_content == "Reading the config now."


@pytest.mark.asyncio
async def test_stream_and_parse_forwards_engine_stage_events():
    """Engine stage chunks (prefill / model_load / slot_restore) must
    reach progress_out so coder-stream.js can drive the real prefill
    progress bar. They carry neither content_delta nor thinking_delta —
    before the relay branch they were silently dropped and the user
    stared at a frozen label for the whole 30-330s prefill window
    (measured 2026-07-02, run …284079a9: 328s dead air on ~97k tokens).
    """
    stage_start = {
        "stage_start": {
            "id": "stg_1", "stage": "prefill",
            "label": "Preparing context", "detail": "",
        },
        "status": "tokenizing",
    }
    stage_complete = {
        "stage_complete": {
            "id": "stg_0", "stage": "slot_restore",
            "success": True, "duration_ms": 1200,
        },
    }
    chunks = [
        _FakeChunk(augmentum=stage_complete),
        _FakeChunk(augmentum=stage_start),
        _FakeChunk(content_delta="READY"),
        _FakeChunk(done=True, finish_reason="stop"),
    ]
    h = _handler(_FakeBackend(chunks))
    progress: list = []
    full_content, _, error_kind, _, _, _ = await h._stream_and_parse(
        _req(), messages=[], tool_schemas=[], tool_map={},
        tier=ToolCallingTier.NATIVE, iteration=1,
        progress_out=progress,
    )
    assert error_kind == ""
    assert full_content == "READY"
    stage_chunks = [
        c for c in progress if (c.augmentum or {}).get("status") == "stage"
    ]
    assert len(stage_chunks) == 2
    fwd_complete, fwd_start = stage_chunks
    assert fwd_complete.augmentum["stage_complete"]["stage"] == "slot_restore"
    assert fwd_start.augmentum["stage_start"]["stage"] == "prefill"
    assert fwd_start.augmentum["stage_start"]["label"] == "Preparing context"
    # The legacy engine-side "status" key must NOT leak through — the
    # forwarded chunk is a proper coder emit() with status="stage".
    assert fwd_start.augmentum["status"] == "stage"
    # Ordering preserved: complete (restore) before start (prefill),
    # both before the responding transition.
    statuses = [(c.augmentum or {}).get("status") for c in progress]
    assert statuses.index("stage") < statuses.index("responding")


# ---------------------------------------------------------------------------
# _stream_and_parse_live wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_wrapper_yields_events_and_result():
    h = _handler(_FakeBackend(_thinking_then_prose_chunks()))
    result: list = []
    events = [
        ev async for ev in h._stream_and_parse_live(
            _req(), [], [], {}, ToolCallingTier.NATIVE, 1,
            result_out=result,
        )
    ]
    assert len(result) == 1
    full_content, tool_calls, error_kind, full_thinking, _, _ = result[0]
    assert error_kind == ""
    assert full_content == "Reading the config now."
    relayed = "".join(
        c.thinking_delta for c in events
        if (c.augmentum or {}).get("status") == "reasoning_delta"
    )
    assert relayed == full_thinking


@pytest.mark.asyncio
async def test_live_wrapper_forwards_chat_template_kwargs():
    backend = _FakeBackend(_thinking_then_prose_chunks())
    h = _handler(backend)
    result: list = []
    async for _ in h._stream_and_parse_live(
        _req(), [], [], {}, ToolCallingTier.NATIVE, 1,
        result_out=result,
        chat_template_kwargs={"enable_thinking": True},
    ):
        pass
    assert backend.last_request is not None
    assert backend.last_request.chat_template_kwargs == {
        "enable_thinking": True,
    }


# ---------------------------------------------------------------------------
# Turn ledger skips the high-frequency relay chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ledger_skips_reasoning_delta():
    from augmentum.coder.ledger import CoderTurnLedger

    class _SpyStore:
        def __init__(self):
            self.events: list = []

        async def record_event(self, **kw):
            self.events.append(kw)

    ledger = CoderTurnLedger(
        store=_SpyStore(), run_id="r1", model="m", strategy="native",
        user_id="alice",
    )
    relay = ReasoningRelay(phase="executing", model="m", min_chars=1)
    reasoning_chunk = relay.add("thinking hard about this one")
    assert reasoning_chunk is not None
    await ledger.observe_chunk(reasoning_chunk)
    assert ledger.store.events == []  # no INSERT for reasoning deltas

    # A normal status chunk still records.
    from augmentum.modes.coder.chat_egress import emit
    await ledger.observe_chunk(
        emit(phase="executing", status="tool_call", model="m"),
    )
    assert len(ledger.store.events) == 1


# ---------------------------------------------------------------------------
# Thinking policy — one helper, all strategies
# ---------------------------------------------------------------------------


def test_policy_defaults_off():
    assert _iteration_thinking_kwargs(_req()) == {"enable_thinking": False}


def test_policy_honors_toggle_on():
    req = _req(chat_template_kwargs={"enable_thinking": True})
    assert _iteration_thinking_kwargs(req) == {"enable_thinking": True}


def test_policy_honors_explicit_off():
    req = _req(chat_template_kwargs={"enable_thinking": False})
    assert _iteration_thinking_kwargs(req) == {"enable_thinking": False}


def test_policy_ignores_global_think_flag():
    # Global think=True must NOT re-enable coder thinking — only the
    # explicit per-turn kwarg does.
    req = _req(think=True)
    assert _iteration_thinking_kwargs(req) == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# Ingress mapping — the field survives both API boundaries
# ---------------------------------------------------------------------------


def test_ollama_ingress_maps_chat_template_kwargs():
    from augmentum.proxy.ollama_routes import (
        OllamaChatRequest,
        to_internal_chat_request,
    )
    body = OllamaChatRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"enable_thinking": True},
    )
    internal = to_internal_chat_request(body, think=False)
    assert internal.chat_template_kwargs == {"enable_thinking": True}


def test_ollama_ingress_defaults_to_none():
    from augmentum.proxy.ollama_routes import (
        OllamaChatRequest,
        to_internal_chat_request,
    )
    body = OllamaChatRequest(
        model="m", messages=[{"role": "user", "content": "hi"}],
    )
    internal = to_internal_chat_request(body, think=False)
    assert internal.chat_template_kwargs is None


def test_openai_ingress_maps_chat_template_kwargs():
    from augmentum.proxy.openai_routes import (
        OpenAIChatRequest,
        to_internal_chat_request,
    )
    body = OpenAIChatRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"enable_thinking": False},
    )
    internal = to_internal_chat_request(body, think=True)
    assert internal.chat_template_kwargs == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# openai_compat folding — explicit kwarg beats request.think on cloud
# ---------------------------------------------------------------------------


def test_effective_think_folds_kwarg_over_think():
    from augmentum.models.openai_compat import _effective_think
    assert _effective_think(_req(think=True)) is True
    assert _effective_think(_req(think=False)) is False
    assert _effective_think(
        _req(think=True, chat_template_kwargs={"enable_thinking": False}),
    ) is False
    assert _effective_think(
        _req(think=False, chat_template_kwargs={"enable_thinking": True}),
    ) is True
    # Kimi-style bare ``thinking`` key is honored too.
    assert _effective_think(
        _req(think=True, chat_template_kwargs={"thinking": False}),
    ) is False
    # Unrelated kwargs don't override.
    assert _effective_think(
        _req(think=True, chat_template_kwargs={"add_generation_prompt": False}),
    ) is True


def test_deepseek_payload_honors_coder_kwarg():
    """The coder act loop's {"enable_thinking": False} must reach
    DeepSeek's cloud toggle even when the global think flag is on."""
    import httpx

    from augmentum.models.openai_compat import OpenAIBackend
    from augmentum.models.provider_profiles import PROFILES

    profile = PROFILES["deepseek"]
    backend = OpenAIBackend(
        httpx.AsyncClient(), profile.base_url, None, profile=profile,
    )
    req = _req(
        model="deepseek-chat",
        think=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    payload = backend._build_openai_payload(req)
    assert payload["thinking"] == {"type": "disabled"}

    req_on = _req(
        model="deepseek-chat",
        think=False,
        chat_template_kwargs={"enable_thinking": True},
    )
    payload_on = backend._build_openai_payload(req_on)
    assert payload_on["thinking"] == {"type": "enabled"}
