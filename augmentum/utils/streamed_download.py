"""Shared streamed-download primitive.

One place for "fetch a (possibly large) file from a URL to disk, atomically,
with byte progress" — used by both the knowledge-pack ``.augpack`` download
endpoint and the language-pack install job.

Implemented as an async generator that yields ``(bytes_written, total_or_None)``
after each chunk (and once up front with ``bytes_written == 0`` so the caller
learns the total before any data arrives). The download lands at a ``.part``
temp file and is renamed onto ``dest`` only on success; the ``.part`` is
removed if the stream fails or the caller stops early. SSRF-checked unless the
caller opts out (e.g. when it has already validated the URL).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import check_ssrf

log = get_logger(__name__)

_DEFAULT_CHUNK = 256 * 1024


async def streamed_download(
    client: Any,                       # httpx.AsyncClient-like (has .stream)
    url: str,
    dest: Path,
    *,
    chunk_size: int = _DEFAULT_CHUNK,
    timeout: float = 180.0,
    ssrf_check: bool = True,
) -> AsyncIterator[tuple[int, int | None]]:
    """Stream ``url`` to ``dest`` (atomic via a ``.part`` temp file).

    Yields ``(written, total)`` — first with ``written == 0`` right after
    the response headers (so the caller can show the total), then after
    every chunk. ``total`` is ``None`` when the server omits a usable
    ``Content-Length``. Raises on HTTP error or SSRF rejection — the caller
    decides what to do with that.
    """
    if ssrf_check:
        await check_ssrf(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    ok = False
    try:
        written = 0
        async with client.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
            resp.raise_for_status()
            total_hdr = str(resp.headers.get("content-length", ""))
            total = int(total_hdr) if total_hdr.isdigit() and int(total_hdr) > 0 else None
            yield written, total
            with open(tmp, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    yield written, total
        tmp.replace(dest)
        ok = True
        log.info("streamed_download_complete", url=url, dest=str(dest), bytes=written)
    finally:
        if not ok and tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
