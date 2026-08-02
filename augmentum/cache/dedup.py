"""In-flight request deduplication.

If two identical requests arrive simultaneously, the second one waits
for the first to complete rather than making a duplicate LLM call.
Uses asyncio.Future for coordination and properly propagates exceptions.
"""

from __future__ import annotations

import asyncio

from augmentum.cache.prompt_cache import PromptCache
from augmentum.models.base import InternalChatRequest, InternalChatResponse
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class RequestDeduplicator:
    """Deduplicates identical in-flight LLM requests.

    When a request arrives that matches an already in-flight request,
    the second caller waits for the first to complete and receives
    the same response. Exceptions from the first request are propagated
    to all waiters.
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Future[InternalChatResponse]] = {}
        self._lock = asyncio.Lock()
        self._dedup_count = 0

    @property
    def dedup_count(self) -> int:
        """Number of requests that were deduplicated (avoided duplicate calls)."""
        return self._dedup_count

    @property
    def in_flight_count(self) -> int:
        """Number of currently in-flight requests."""
        return len(self._in_flight)

    async def acquire(
        self, request: InternalChatRequest
    ) -> tuple[str, asyncio.Future[InternalChatResponse] | None]:
        """Try to acquire the right to execute a request.

        Returns (key, future_or_none):
        - If future_or_none is None: this caller should execute the request
          and then call ``complete()`` or ``fail()`` with the key.
        - If future_or_none is a Future: another caller is already executing
          this request. Await the future to get the result.
        """
        key = PromptCache.make_key(request)

        async with self._lock:
            if key in self._in_flight:
                self._dedup_count += 1
                log.debug("request_dedup", key=key[:12])
                return key, self._in_flight[key]

            future: asyncio.Future[InternalChatResponse] = asyncio.get_event_loop().create_future()
            self._in_flight[key] = future
            return key, None

    async def complete(self, key: str, response: InternalChatResponse) -> None:
        """Mark a request as completed with a successful response.

        Sets the result on the future so all waiters receive it.
        """
        async with self._lock:
            future = self._in_flight.pop(key, None)
            if future and not future.done():
                future.set_result(response)

    async def fail(self, key: str, exception: BaseException) -> None:
        """Mark a request as failed with an exception.

        Sets the exception on the future so all waiters receive it.
        """
        async with self._lock:
            future = self._in_flight.pop(key, None)
            if future and not future.done():
                future.set_exception(exception)

    def get_stats(self) -> dict:
        """Return deduplicator statistics."""
        return {
            "dedup_count": self._dedup_count,
            "in_flight_count": len(self._in_flight),
        }
