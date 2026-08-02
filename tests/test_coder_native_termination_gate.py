"""Tests for the Termination Quality Gate as wired into ``_act_native``.

The hybrid loop has used ``coder.termination.evaluate_termination`` since
phase 3.6, but native pre-2026-05-27 broke immediately on the first
prose-with-no-tools response — which caught Qwen-3.6 mid-preamble
("Let me check that for you.", content_len=55, tool_call_count=0) and
terminated the turn before the model ever called a tool. The fix drops
the same gate into ``_act_native`` so it nudges the bail and accepts
the wrap-up.

Test cases cover the five gate verdicts that fire in the native context:

  1. ``REASON_NUDGE_BAILOUT`` — first iter, short single-sentence prose,
     zero progress → loop continues, model gets a nudge message.
  2. ``REASON_SUBSTANTIVE_ACTIVE`` — first iter, substantive prose
     (>=30 chars, 2+ sentences) under an action-shaped request →
     accepts even with zero tool calls.
  3. ``REASON_RECENT_PROGRESS`` — successful tool call (read), then
     bailout-shaped prose → accepts because the gate counts any tool
     use as progress in native (writes-only would over-nudge).
  4. ``REASON_ALREADY_NUDGED`` — bailout once, model responds with
     another bailout → accepts on the second pass; one nudge max.
  5. ``REASON_NUDGE_INSISTENT`` — user said "don't stop until finished",
     model wraps up with zero writes → nudges even though prose may be
     substantive; INSISTENT demand requires actual write work.

Plus one safety case: writes-only progress tracking would over-nudge
the file_list+summary pattern. We verify that pattern accepts in
native via the reads-also-count-as-progress branch.

Run: python -m pytest tests/test_coder_native_termination_gate.py -v
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.models.base import InternalStreamChunk

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _FakeTool,
    _make_request,
    _tc_delta,
)


# These tests were written when the native nudge cap was hard-coded at
# 1 (the gate accepts the second prose-no-tools response via
# REASON_ALREADY_NUDGED). On 2026-05-31 the cap moved to a live
# setting (``coder_native_nudge_max``, default 2). The gate semantics
# tested here — "nudges once, then accepts" — are still valid with
# cap=1; we just need to pin the cap so the tests don't see the new
# default's extra nudge. Module-level autouse so every test inherits.
@pytest.fixture(autouse=True)
def _pin_nudge_cap_to_one(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_native_nudge_max", 1, raising=False)
    yield


def _run_native(backend, *, user_text: str = "list files") -> list[InternalStreamChunk]:
    handler = CoderHandler(
        backend,
        session_id="sess-native-tqg",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-native-tqg",
    )
    chunks: list[InternalStreamChunk] = []

    async def _drain():
        async for c in handler._act_native(
            _make_request(user_text), workspace_context="",
        ):
            chunks.append(c)

    import asyncio as _asyncio
    _asyncio.get_event_loop().run_until_complete(_drain())
    return chunks


def _termination_reason(chunks: list[InternalStreamChunk]) -> str:
    """Find the final 'complete' meta chunk's termination_reason."""
    for c in reversed(chunks):
        if (
            c.augmentum
            and c.augmentum.get("status") == "complete"
            and "termination_reason" in c.augmentum
        ):
            return c.augmentum["termination_reason"]
    return ""


# ---------------------------------------------------------------------------
# Case 1 — Bailout nudges, then accept on substantive retry
# ---------------------------------------------------------------------------


class _BailThenSubstantive:
    """First response is a short single-sentence bail; second is real."""

    def __init__(self):
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            # 21 chars, 1 sentence → BAILOUT under the gate
            yield _FakeChunk(content_delta="Let me check that.")
        else:
            # 65 chars, 2 sentences → SUBSTANTIVE; gate accepts via
            # already_nudged.
            yield _FakeChunk(
                content_delta=(
                    "Listed all three files. No further work needed."
                ),
            )
        yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_native_bailout_prose_nudges_then_accepts(monkeypatch):
    """Iter 1 prose is a bail → gate nudges, model retries, accept."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )
    backend = _BailThenSubstantive()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("list files"), workspace_context="",
    ):
        chunks.append(c)

    # Two backend roundtrips: bail, then substantive retry.
    assert len(backend.requests) == 2
    # The second request's tail must be the nudge we appended.
    last_msg = backend.requests[1].messages[-1]
    assert last_msg.role == "user"
    assert "<nudge>" in last_msg.content
    # Loop emits a continuation_nudge meta chunk on the bail.
    nudge_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "continuation_nudge"
    ]
    assert len(nudge_chunks) == 1
    assert nudge_chunks[0].augmentum.get("strategy") == "native"
    assert nudge_chunks[0].augmentum.get("nudge_kind") in {
        "bailout_short_prose", "no_progress_action_turn",
    }
    # Gate's decision tree (termination.py:357) fires "already_nudged"
    # FIRST — once we've nudged in this turn, we accept the next stop
    # unconditionally. The retry being substantive doesn't override
    # that; the one-nudge bound is what we want.
    final_reason = _termination_reason(chunks)
    assert final_reason == "model_stop:already_nudged"


# ---------------------------------------------------------------------------
# Case 2 — Substantive prose iter 1 accepts immediately
# ---------------------------------------------------------------------------


class _SubstantiveImmediate:
    def __init__(self):
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        # 2026-05-31: bumped past 200 chars so the native preamble
        # override (added alongside the nudge-cap work) accepts this
        # as a real substantive answer rather than treating short
        # 2-sentence prose as a preamble worth nudging.
        yield _FakeChunk(
            content_delta=(
                "Python's GIL (Global Interpreter Lock) serializes "
                "bytecode execution within one interpreter, so only "
                "one thread runs Python at a time. It does not affect "
                "IO-bound concurrency — threads released the GIL while "
                "waiting on syscalls. CPU-bound parallelism needs "
                "multiprocessing or a free-threaded build."
            ),
        )
        yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_native_substantive_prose_accepts_with_no_tools(monkeypatch):
    """A real first-iteration answer accepts even with zero tool calls."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )
    backend = _SubstantiveImmediate()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("explain the GIL"), workspace_context="",
    ):
        chunks.append(c)

    # Only one backend call — no nudge needed.
    assert len(backend.requests) == 1
    # And no continuation_nudge meta chunks.
    nudge_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "continuation_nudge"
    ]
    assert nudge_chunks == []
    final_reason = _termination_reason(chunks)
    assert final_reason == "model_stop:substantive_under_active"


# ---------------------------------------------------------------------------
# Case 3 — Tool call then bailout-shaped wrap accepts via recent progress
# ---------------------------------------------------------------------------


class _ReadThenBail:
    def __init__(self):
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, "tc1", "file_list", {"path": "/workspace"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")
        else:
            # 31 chars, 1 sentence — would be BAILOUT in isolation, but
            # the previous iteration produced a successful read, so the
            # gate's had_recent_progress branch accepts the wrap-up.
            yield _FakeChunk(content_delta="Found index.html in /workspace.")
            yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_native_read_then_short_wrap_accepts_via_progress(monkeypatch):
    """The file_list → 'Found X.' wrap-up MUST accept, not over-nudge.

    This is the regression case from the first attempt at this patch:
    write-only progress tracking forced an unnecessary nudge after every
    read-only turn. Fix is to count reads as progress in the native
    branch (intent classification doesn't happen there, so we don't have
    the INSPECT/RESEARCH path that hybrid uses to skip the nudge).
    """
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list", output="file index.html")],
    )
    backend = _ReadThenBail()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("list files"), workspace_context="",
    ):
        chunks.append(c)

    # Two backend calls — tool, then wrap. No third (no nudge fired).
    assert len(backend.requests) == 2
    final_reason = _termination_reason(chunks)
    assert final_reason == "model_stop:recent_progress"


# ---------------------------------------------------------------------------
# Case 4 — Already-nudged bound prevents infinite nudging
# ---------------------------------------------------------------------------


class _DoubleBail:
    def __init__(self):
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        # Both responses are bail-shaped. Gate nudges once, then
        # already_nudged on the second pass accepts regardless of
        # prose quality — one-nudge bound.
        yield _FakeChunk(content_delta="Let me check.")
        yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_native_already_nudged_caps_at_one_nudge(monkeypatch):
    """Even if the model bails again after a nudge, accept — never loop."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )
    backend = _DoubleBail()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("list files"), workspace_context="",
    ):
        chunks.append(c)

    # Exactly two backend calls — bail + nudge + bail → accept.
    assert len(backend.requests) == 2
    nudge_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "continuation_nudge"
    ]
    assert len(nudge_chunks) == 1
    final_reason = _termination_reason(chunks)
    assert final_reason == "model_stop:already_nudged"


# ---------------------------------------------------------------------------
# Case 5 — INSISTENT user demand nudges even substantive prose
# ---------------------------------------------------------------------------


class _SubstantiveButZeroWrites:
    def __init__(self):
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            # 110 chars, 2 sentences → SUBSTANTIVE. But the user said
            # "don't stop until finished" and zero writes happened, so
            # the gate nudges regardless of prose quality.
            yield _FakeChunk(
                content_delta=(
                    "I've reviewed the structure and identified the "
                    "right approach. Implementation is straightforward."
                ),
            )
        else:
            # On the retry, model still doesn't write — but we've
            # nudged once so the gate caps and accepts.
            yield _FakeChunk(
                content_delta=(
                    "Still blocked on writes. The path is read-only."
                ),
            )
        yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_native_insistent_demand_nudges_zero_writes(monkeypatch):
    """User said 'don't stop until done' + zero writes → nudge."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_write")],
    )
    backend = _SubstantiveButZeroWrites()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        # Insistent phrasing triggers the gate's INSISTENT classification.
        _make_request("implement the feature, don't stop until finished"),
        workspace_context="",
    ):
        chunks.append(c)

    # Two roundtrips — gate nudged on the first stop even though prose
    # was substantive (INSISTENT + zero writes overrides).
    assert len(backend.requests) == 2
    nudge_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "continuation_nudge"
    ]
    assert len(nudge_chunks) == 1
    assert nudge_chunks[0].augmentum.get("nudge_kind") == "user_demanded_completion"
    final_reason = _termination_reason(chunks)
    # Second pass: still zero writes, but already nudged → accept.
    assert final_reason == "model_stop:already_nudged"
