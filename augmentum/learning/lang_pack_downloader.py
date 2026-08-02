"""Language-pack source downloads.

A thin wrapper over :func:`augmentum.utils.streamed_download.streamed_download`
that fits the install job's needs: a sync ``on_progress(done, total)``
callback (the job bridges it to ``ctx.update_progress``) and a
URL→filename helper for naming the downloaded corpus files. The HTTP
client is passed in (``app.state.http_client``) so this module has no
global I/O and is easy to drive from a test with a fake.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from augmentum.utils.logging import get_logger
from augmentum.utils.streamed_download import streamed_download

log = get_logger(__name__)


def filename_for(url: str) -> str:
    """A safe local filename derived from a URL's last path segment.

    Preserves multi-suffix names like ``sentences.tar.bz2`` and
    ``JMdict_e.gz``; falls back to ``download`` if the path is empty.
    Never contains a path separator.
    """
    path = urlsplit(url).path
    name = unquote(path.rsplit("/", 1)[-1]) if path else ""
    name = name.strip().strip(".").replace("\\", "_").replace("/", "_")
    return name or "download"


async def download_to(
    client: Any,                       # httpx.AsyncClient-like (has .stream)
    url: str,
    dest: Path,
    *,
    on_progress: Callable[[int, int | None], None] | None = None,
    timeout: float = 180.0,
) -> int:
    """Stream ``url`` to ``dest`` (atomically). Returns bytes written.

    ``on_progress(done, total)`` is invoked after each chunk; ``total`` is
    ``None`` when the server omits a usable ``Content-Length``. Raises on
    any HTTP error or SSRF rejection — the caller decides whether the
    source is required.
    """
    written = 0
    async for done, total in streamed_download(client, url, dest, timeout=timeout):
        written = done
        if on_progress:
            on_progress(done, total)
    log.info("lang_pack_source_downloaded", url=url, dest=str(dest), bytes=written)
    return written
