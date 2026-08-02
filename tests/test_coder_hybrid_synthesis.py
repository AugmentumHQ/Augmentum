"""Phase 6 — hybrid loop calls _synthesize_response on termination.

When ``AUGMENTUM_CODER_SYNTHESIZE_HYBRID=1`` is set and the model ended
a turn having done real tool work but narrated little or no prose, the
hybrid loop invokes the backend one more time to synthesize a final
user-facing summary. When the flag is unset, behaviour is unchanged —
the deterministic ``_render_fallback_summary`` fires exactly as before.

Covers:

- Flag off → no synthesis backend call, deterministic fallback as-is.
- Flag on + tool results collected + low prose → synthesis runs, its
  streamed content reaches the user, deterministic fallback does NOT.
- Flag on but synthesis backend raises → silent fall-through to the
  deterministic fallback (coder_synthesis_failed logged).
- Flag on but synthesis emits nothing (silent backend) → fall through
  to deterministic fallback.
- Flag on but no tool work done at all → neither path fires (nothing
  to summarize; the model's own prose or lack of it is the answer).
"""
from __future__ import annotations

import pytest

from augmentum.models.base import InternalStreamChunk
from augmentum.modes.coder.handler import CoderHandler
from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)


class _LowProseBackend:
    """Iter 1 emits one write tool_call; iter 2 stops silently with no prose.

    Write tool (not read) so the continuation judge sees recent
    progress and the loop breaks naturally on iter 2 without firing
    the nudge. This reproduces the production failure mode Phase 6
    targets: model did work, tool_calls_made > 0, but final prose is
    < the 80-char threshold — UI shows a blank "Done" without
    synthesis.
    """

    def __init__(self, synth_output: str = "") -> None:
        self.calls = 0
        self.synth_output = synth_output

    async def chat_stream(self, request):
        self.calls += 1
        if self.calls == 1:
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, "tc-1", "file_write",
                          {"path": "/x.py", "content": "print('x')\n"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")
        elif self.calls == 2:
            # Silent stop — no content_delta, just done. Triggers the
            # no-prose summary path. Loop breaks here because iter 1
            # counted as a write.
            yield _FakeChunk(done=True, finish_reason="stop")
        else:
            # Synthesis call. When synth_output is set, stream it back.
            if self.synth_output:
                yield _FakeChunk(content_delta=self.synth_output)
            yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


def _make_handler(backend, monkeypatch):
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_write", output="written")],
    )
    return CoderHandler(
        backend, session_id="sess-synth",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-synth",
    )


@pytest.mark.asyncio
async def test_flag_off_uses_deterministic_fallback(monkeypatch):
    """Default behaviour: synthesis flag unset, deterministic fallback runs."""
    monkeypatch.delenv("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", raising=False)
    backend = _LowProseBackend()
    handler = _make_handler(backend, monkeypatch)

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # Fallback summary emits with status "fallback_summary"
    fallback_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "fallback_summary"
    ]
    assert len(fallback_chunks) == 1

    # Synthesis did NOT run — backend only saw 2 chat_stream calls
    # (iter 1 tool call, iter 2 stop). A third call would mean the
    # synthesis path fired.
    assert backend.calls == 2


@pytest.mark.asyncio
async def test_flag_on_synthesis_replaces_fallback(monkeypatch):
    """Flag on + synthesis succeeds → synthesis content streams, fallback suppressed."""
    monkeypatch.setenv("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", "1")
    backend = _LowProseBackend(
        synth_output="I read x.py and here is the summary.",
    )
    handler = _make_handler(backend, monkeypatch)

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # Synthesis added a third backend call
    assert backend.calls == 3

    # Fallback did NOT fire
    fallback_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "fallback_summary"
    ]
    assert fallback_chunks == []

    # Synthesis output reached the chat stream
    text = "".join(c.content_delta or "" for c in chunks)
    assert "I read x.py" in text


@pytest.mark.asyncio
async def test_flag_on_synthesis_silent_backend_falls_through(monkeypatch):
    """Flag on but synthesis emits no content → fall through to fallback."""
    monkeypatch.setenv("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", "1")
    # Empty synth_output → synthesis call returns without content_delta.
    backend = _LowProseBackend(synth_output="")
    handler = _make_handler(backend, monkeypatch)

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # Synthesis ran (3 backend calls) but emitted nothing useful
    assert backend.calls == 3

    # Fallback took over as safety net
    fallback_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "fallback_summary"
    ]
    assert len(fallback_chunks) == 1


@pytest.mark.asyncio
async def test_flag_on_synthesis_exception_falls_through(monkeypatch):
    """Flag on but synthesis backend raises → fall through to fallback."""
    monkeypatch.setenv("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", "1")

    class _ExplodingSynth(_LowProseBackend):
        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_write",
                              {"path": "/x.py", "content": "x\n"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            elif self.calls == 2:
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                # Synthesis call — raise mid-stream
                raise RuntimeError("backend 503 during synthesis")

    backend = _ExplodingSynth()
    handler = _make_handler(backend, monkeypatch)

    chunks: list[InternalStreamChunk] = []
    # Must NOT raise — synthesis failure is silent, fallback fires.
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    assert backend.calls == 3  # synthesis attempted
    fallback_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "fallback_summary"
    ]
    assert len(fallback_chunks) == 1


@pytest.mark.asyncio
async def test_flag_on_no_tool_work_skips_both_paths(monkeypatch):
    """When no tool work happened, neither synthesis nor fallback fires.

    The summary path is guarded on ``total_writes > 0 OR
    tool_calls_made > 0``. If the model just chatted without tools
    (which is rare in hybrid but possible on greeting-like prompts
    that escape the conversational short-circuit), leave the output
    alone.
    """
    monkeypatch.setenv("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", "1")

    class _NoToolsBackend:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # No tool calls, model answers inline (but still short).
            yield _FakeChunk(content_delta="hi")
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _NoToolsBackend()
    handler = _make_handler(backend, monkeypatch)

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # "hi" is short (<40 chars) so Innovation #4 nudges, adding one
    # more iteration. But no tool work ever happened, so the final
    # summary guard (total_writes > 0 OR tool_calls_made > 0) fails
    # and neither synthesis nor fallback fires. Backend call count is
    # 2 (nudged iter + break iter), synthesis adds 0.
    assert backend.calls == 2, (
        f"Expected 2 iters (one + nudge) with zero synthesis, "
        f"got {backend.calls}"
    )
    fallback_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "fallback_summary"
    ]
    assert fallback_chunks == []
    # No synthesis content either — chunks should not contain any
    # synthesized final-summary prose.
    synth_content = "".join(
        c.content_delta or "" for c in chunks
        if c.augmentum and c.augmentum.get("status") == "streaming"
        and c.augmentum.get("phase") == "executing"
    )
    # The only streaming content should be the model's own "hi" prose,
    # not a synthesized summary.
    assert "hi" in synth_content


@pytest.mark.asyncio
async def test_synthesis_receives_tool_result_rollup(monkeypatch):
    """Synthesis call's messages include the tool_results the turn produced.

    Regression guard: without the _tap_tool_result collector, synthesis
    would fire with an empty results list and the model would have
    nothing to summarize. This test exercises the tap path end-to-end.
    """
    monkeypatch.setenv("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", "1")

    # Capture the user-message content on the synthesis call.
    synth_user_content: list[str] = []

    class _CapturingBackend:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_write",
                              {"path": "/x.py", "content": "x\n"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            elif self.calls == 2:
                yield _FakeChunk(done=True, finish_reason="stop")
            else:
                # Synthesis call — capture its user message for inspection.
                for m in request.messages:
                    if m.role == "user":
                        synth_user_content.append(m.content)
                yield _FakeChunk(content_delta="ok")
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _CapturingBackend()
    handler = _make_handler(backend, monkeypatch)

    chunks = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # Synthesis user message contains the tool-result rollup shape
    assert synth_user_content, "synthesis call should have a user message"
    combined = "\n".join(synth_user_content)
    assert "file_write" in combined  # the tool that ran in iter 1
