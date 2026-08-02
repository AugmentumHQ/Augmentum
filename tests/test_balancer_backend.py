"""BalancerBackend facade — fallback semantics (step 1)."""
from __future__ import annotations

import types

import pytest

from augmentum.models.backend_errors import (
    BackendError,
    parse_retry_after,
    retry_after_from_body,
)
from augmentum.models.balancer_backend import BalancerBackend, _is_retryable
from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.load_balancer import LoadBalancer
from augmentum.state.balancer_store import BalancerConfig, BalancerMember


class _FakeBackend:
    """Minimal backend: stream yields sentinel chunks; optionally fail."""

    def __init__(self, *, fail=None, when="before", chunks=2):
        self.fail = fail          # exception instance to raise, or None
        self.when = when          # "before" first chunk, or "after" first chunk
        self.chunks = chunks
        self.stream_calls = 0
        self.chat_calls = 0

    async def chat_stream(self, request):
        self.stream_calls += 1
        if self.fail and self.when == "before":
            raise self.fail
        for i in range(self.chunks):
            yield types.SimpleNamespace(content_delta=f"c{i}", _model=request.model)
            if self.fail and self.when == "after" and i == 0:
                raise self.fail

    async def chat(self, request):
        self.chat_calls += 1
        if self.fail:
            raise self.fail
        return types.SimpleNamespace(content=f"ok:{request.model}")


def _rig(*, fallback_enabled=True, a_fail=None, a_when="before", b_fail=None,
         a_chunks=2, b_chunks=2):
    a = _FakeBackend(fail=a_fail, when=a_when, chunks=a_chunks)
    b = _FakeBackend(fail=b_fail, chunks=b_chunks)
    registry = types.SimpleNamespace(_backends={"A": a, "B": b})
    cfg = BalancerConfig(id="lb1", name="pool", strategy="round_robin",
                         fallback_enabled=fallback_enabled)
    members = [
        BalancerMember(id=1, balancer_id="lb1", model_name="m-a", backend_key="A", priority=0),
        BalancerMember(id=2, balancer_id="lb1", model_name="m-b", backend_key="B", priority=1),
    ]
    lb = LoadBalancer(cfg, members)
    return BalancerBackend(lb, registry), a, b, lb


def _req():
    return InternalChatRequest(model="lb/pool", messages=[Message(role="user", content="hi")])


async def _drain(facade, req):
    return [c async for c in facade.chat_stream(req)]


class TestRetryableClassifier:
    def test_cancellation_not_retryable(self):
        import asyncio
        assert _is_retryable(asyncio.CancelledError()) is False
        assert _is_retryable(GeneratorExit()) is False

    def test_rate_limit_retryable_auth_not(self):
        assert _is_retryable(RuntimeError("Backend returned 429: rate limit")) is True
        assert _is_retryable(RuntimeError("503 Service Unavailable")) is True
        assert _is_retryable(RuntimeError("401 unauthorized")) is False
        assert _is_retryable(RuntimeError("maximum context length exceeded")) is False


class TestRetryAfterParser:
    def test_parses_body_hints(self):
        assert retry_after_from_body("Please retry in 56.016388495s.") == pytest.approx(56.016, abs=0.01)
        assert retry_after_from_body('"retryDelay": "14s"') == 14.0
        assert retry_after_from_body("retry after 30 seconds") == 30.0
        assert retry_after_from_body("high demand, try again later") is None

    def test_parses_headers(self):
        # Retry-After as plain seconds (OpenAI-compat convention)
        assert parse_retry_after({"retry-after": "42"}) == 42.0
        # X-RateLimit-Reset as a duration string (some gateways)
        assert parse_retry_after({"x-ratelimit-reset": "6m0s"}) == 360.0
        assert parse_retry_after({"x-ratelimit-reset": "1s"}) == 1.0
        # header beats body when both present
        assert parse_retry_after({"retry-after": "10"}, "retry in 999s") == 10.0
        # no header, no body signal → None (caller uses blind backoff)
        assert parse_retry_after({}, "overloaded") is None
        # falls back to body when headers carry nothing usable
        assert parse_retry_after({"x-foo": "bar"}, "retry in 12s") == 12.0


class TestCooldown:
    @pytest.mark.asyncio
    async def test_failed_member_skipped_next_request(self):
        # A fails with a long retry hint on call 1 → cooled → call 2 skips A.
        facade, a, b, lb = _rig(a_fail=RuntimeError("Backend returned 429: retry in 300s"))
        c1 = await _drain(facade, _req())
        assert [c._model for c in c1] == ["m-b", "m-b"]  # A failed, B served
        assert lb.is_cooling(lb.members[0]) is True        # A cooling
        assert lb.cooldown_remaining(lb.members[0]) == pytest.approx(300, abs=5)
        c2 = await _drain(facade, _req())
        assert [c._model for c in c2] == ["m-b", "m-b"]  # served by B again
        assert a.stream_calls == 1                        # A NOT re-tried on call 2
        assert b.stream_calls == 2

    @pytest.mark.asyncio
    async def test_success_clears_cooldown(self):
        facade, a, b, lb = _rig()
        lb.note_failure(lb.members[0], 300)
        assert lb.is_cooling(lb.members[0]) is True
        # A healthy call to A clears it (round-robin picks B first here since A
        # is cooling, so drive note_success directly to assert the mechanism):
        lb.note_success(lb.members[0])
        assert lb.is_cooling(lb.members[0]) is False

    @pytest.mark.asyncio
    async def test_all_cooling_still_attempts(self):
        # Both cooling → _active_members falls back to ALL so we still try.
        facade, a, b, lb = _rig()
        lb.note_failure(lb.members[0], 300)
        lb.note_failure(lb.members[1], 300)
        chunks = await _drain(facade, _req())  # A serves (round-robin over all)
        assert [c._model for c in chunks] == ["m-a", "m-a"]

    @pytest.mark.asyncio
    async def test_structured_retry_after_honored(self):
        # "429 high demand" has NO body hint — the 120s must come from the
        # adapter-parsed BackendError.retry_after (header-derived).
        facade, a, b, lb = _rig(a_fail=BackendError("429 high demand", retry_after=120))
        await _drain(facade, _req())
        assert lb.cooldown_remaining(lb.members[0]) == pytest.approx(120, abs=5)

    def test_blind_backoff_escalates(self):
        _, _, _, lb = _rig()
        m = lb.members[0]
        lb.note_failure(m, None)          # streak 1
        first = lb.cooldown_remaining(m)
        lb.note_success(m)
        lb.note_failure(m, None)          # streak 1 again
        lb.note_failure(m, None)          # streak 2
        second = lb.cooldown_remaining(m)
        assert second > first  # exponential: streak 2 cools longer than streak 1


class TestFallback:
    @pytest.mark.asyncio
    async def test_falls_over_on_retryable_before_first_token(self):
        facade, a, b, _ = _rig(a_fail=RuntimeError("429 rate limit"))
        req = _req()
        chunks = await _drain(facade, req)
        assert [c._model for c in chunks] == ["m-b", "m-b"]  # served by member B
        assert a.stream_calls == 1 and b.stream_calls == 1

    @pytest.mark.asyncio
    async def test_no_fallover_once_tokens_streamed(self):
        # A yields one chunk THEN fails — can't un-send, must surface.
        facade, a, b, _ = _rig(a_fail=RuntimeError("429"), a_when="after")
        with pytest.raises(RuntimeError, match="429"):
            await _drain(facade, _req())
        assert b.stream_calls == 0  # B never tried

    @pytest.mark.asyncio
    async def test_empty_completion_falls_over(self):
        # A returns a clean but content-less stream (Gemini safety/quota block,
        # swallowed mid-stream error). Must NOT be silent success — fall over.
        facade, a, b, _ = _rig(a_chunks=0)
        chunks = await _drain(facade, _req())
        assert [c._model for c in chunks] == ["m-b", "m-b"]
        assert a.stream_calls == 1 and b.stream_calls == 1

    @pytest.mark.asyncio
    async def test_all_empty_raises_visible_error(self):
        facade, a, b, _ = _rig(a_chunks=0, b_chunks=0)
        with pytest.raises(RuntimeError, match="failed or returned empty"):
            await _drain(facade, _req())

    @pytest.mark.asyncio
    async def test_non_retryable_surfaces_immediately(self):
        facade, a, b, _ = _rig(a_fail=RuntimeError("401 unauthorized"))
        with pytest.raises(RuntimeError, match="unauthorized"):
            await _drain(facade, _req())
        assert b.stream_calls == 0  # no pool burn on auth failure

    @pytest.mark.asyncio
    async def test_fallback_disabled_is_single_shot(self):
        facade, a, b, _ = _rig(fallback_enabled=False, a_fail=RuntimeError("429"))
        with pytest.raises(RuntimeError, match="429"):
            await _drain(facade, _req())
        assert a.stream_calls == 1 and b.stream_calls == 0

    @pytest.mark.asyncio
    async def test_all_members_exhausted_raises_last(self):
        facade, a, b, _ = _rig(a_fail=RuntimeError("429 A"),
                               b_fail=RuntimeError("503 B"))
        # both fail; b_fail is passed to the B fake via _rig? -> set below
        b.fail = RuntimeError("503 B")
        with pytest.raises(RuntimeError, match="503 B"):
            await _drain(facade, _req())
        assert a.stream_calls == 1 and b.stream_calls == 1

    @pytest.mark.asyncio
    async def test_non_stream_chat_falls_over(self):
        facade, a, b, _ = _rig(a_fail=RuntimeError("429"))
        resp = await facade.chat(_req())
        assert resp.content == "ok:m-b"
        assert a.chat_calls == 1 and b.chat_calls == 1

    @pytest.mark.asyncio
    async def test_selects_once_per_call_no_double_advance(self):
        facade, a, b, lb = _rig()
        calls = {"n": 0}
        orig = lb.select
        lb.select = lambda: (calls.__setitem__("n", calls["n"] + 1), orig())[1]
        await _drain(facade, _req())
        assert calls["n"] == 1  # facade selects exactly once (RR not double-advanced)

    @pytest.mark.asyncio
    async def test_does_not_mutate_caller_request_model(self):
        facade, a, b, _ = _rig(a_fail=RuntimeError("429"))
        req = _req()
        await _drain(facade, req)
        assert req.model == "lb/pool"  # caller's request untouched
