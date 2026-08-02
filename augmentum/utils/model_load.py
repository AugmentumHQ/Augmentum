"""Off-loop model-load helper — runs the load on a worker thread with
bounded global concurrency.

Why: every model load (ONNX session creation, torch weight load,
embedding/reranker init, llama.cpp tokenizer probe) is hundreds of
milliseconds to several seconds of CPU + disk burst. Wrapping with
``asyncio.to_thread`` keeps the event loop responsive, but without
bounded concurrency, N simultaneous voice sessions race-load N copies
of the same models — peg CPU, thrash disk, can OOM on cold cache.

A small global semaphore gates concurrent loads to
``AUGMENTUM_MODEL_LOAD_CONCURRENCY`` (default 2). Loads queue rather
than serialize the event loop. The idempotent ``load_model`` /
``get_model`` convention across the codebase means a queued waiter
typically becomes a no-op once the first load finishes.

Usage::

    from augmentum.utils.model_load import load_model_off_loop
    if not eng.is_available:
        await load_model_off_loop(eng.load_model)

When the semaphore is fully held, queued callers emit
``model_load_queued`` so operators can spot saturation and bump the
concurrency knob.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

T = TypeVar("T")

log = structlog.get_logger(__name__)

_DEFAULT_CONCURRENCY = 2


def _read_concurrency() -> int:
    raw = os.environ.get("AUGMENTUM_MODEL_LOAD_CONCURRENCY", "").strip()
    if not raw:
        return _DEFAULT_CONCURRENCY
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_CONCURRENCY


_CONCURRENCY = _read_concurrency()

# One semaphore per event loop. The dict avoids the pytest footgun
# where a module-level Semaphore() created at import time binds to
# the first loop that touches it and breaks subsequent test loops.
# In prod there's exactly one loop, so this dict has exactly one entry.
_semaphores: dict[int, asyncio.Semaphore] = {}


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _semaphores.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_CONCURRENCY)
        _semaphores[key] = sem
    return sem


async def load_model_off_loop(
    fn: Callable[..., T], /, *args: Any, **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` on a worker thread, gated by the global
    model-load semaphore.

    Drop-in replacement for ``asyncio.to_thread(fn, ...)`` at any site
    that loads model weights. Returns whatever ``fn`` returns.
    """
    sem = _get_semaphore()
    if sem.locked():
        log.info(
            "model_load_queued",
            target=getattr(fn, "__qualname__", repr(fn)),
            concurrency=_CONCURRENCY,
        )
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)
