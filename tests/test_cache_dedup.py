"""Tests for cache/dedup.py — in-flight request deduplication."""

from __future__ import annotations

import asyncio

import pytest

from augmentum.cache.dedup import RequestDeduplicator
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
    Usage,
)


def _make_request(content: str = "hello") -> InternalChatRequest:
    return InternalChatRequest(
        model="llama3.1:8b",
        messages=[Message(role="user", content=content)],
    )


def _make_response(content: str = "Hi there!") -> InternalChatResponse:
    return InternalChatResponse(
        message=Message(role="assistant", content=content),
        model="llama3.1:8b",
        finish_reason="stop",
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestRequestDeduplicator:
    def test_construct(self):
        dedup = RequestDeduplicator()
        assert dedup.dedup_count == 0
        assert dedup.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_acquire_first_request_returns_none(self):
        dedup = RequestDeduplicator()
        key, future = await dedup.acquire(_make_request())
        assert key
        assert future is None
        assert dedup.in_flight_count == 1

    @pytest.mark.asyncio
    async def test_acquire_duplicate_returns_future(self):
        dedup = RequestDeduplicator()
        req = _make_request()
        key1, fut1 = await dedup.acquire(req)
        assert fut1 is None

        key2, fut2 = await dedup.acquire(req)
        assert key1 == key2
        assert fut2 is not None
        assert dedup.dedup_count == 1

    @pytest.mark.asyncio
    async def test_complete_resolves_waiters(self):
        dedup = RequestDeduplicator()
        req = _make_request()
        key, _ = await dedup.acquire(req)
        _, future = await dedup.acquire(req)

        resp = _make_response()
        await dedup.complete(key, resp)

        result = await future
        assert result.message.content == "Hi there!"
        assert dedup.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_fail_propagates_exception(self):
        dedup = RequestDeduplicator()
        req = _make_request()
        key, _ = await dedup.acquire(req)
        _, future = await dedup.acquire(req)

        await dedup.fail(key, RuntimeError("backend down"))

        with pytest.raises(RuntimeError, match="backend down"):
            await future

    @pytest.mark.asyncio
    async def test_different_requests_not_deduped(self):
        dedup = RequestDeduplicator()
        key1, _ = await dedup.acquire(_make_request("hello"))
        key2, _ = await dedup.acquire(_make_request("goodbye"))
        assert key1 != key2
        assert dedup.dedup_count == 0
        assert dedup.in_flight_count == 2

    @pytest.mark.asyncio
    async def test_complete_unknown_key_no_error(self):
        dedup = RequestDeduplicator()
        await dedup.complete("nonexistent", _make_response())

    @pytest.mark.asyncio
    async def test_fail_unknown_key_no_error(self):
        dedup = RequestDeduplicator()
        await dedup.fail("nonexistent", RuntimeError("test"))

    def test_get_stats(self):
        dedup = RequestDeduplicator()
        stats = dedup.get_stats()
        assert stats["dedup_count"] == 0
        assert stats["in_flight_count"] == 0
