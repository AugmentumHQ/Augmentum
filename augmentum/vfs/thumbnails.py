"""Unified thumbnail service for anything surfaced via file_index.

Gallery cards, Files panel tiles, chat inline previews, comic series
covers — every surface that renders a small preview of a file goes
through this one endpoint so we have a single cache, single eviction
policy, and single URL shape on the frontend.

Per-source rendering is adapter-dispatched. An adapter opts in by
implementing an optional ``produce_thumbnail`` coroutine::

    async def produce_thumbnail(
        self, source_id: str, size: int, *, user_id: str,
    ) -> bytes | None:
        ...

Adapters that don't implement it get a default: if the file's mime_type
starts with ``image/``, we resolve the source and downscale it here.
That default covers the ``images`` and ``chat_images`` sources without
adapter changes. PDF/video/other producers plug in on their adapter.

Cache layout — ``{data_dir}/thumbs/{source}/{source_id}_{size}.webp`` —
is clustered by source so per-source purge on delete is a simple
recursive drop of one subtree.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.vfs.models import FileEntry

log = get_logger(__name__)

# Fixed size tokens. Anything outside this set 400s at the route layer.
# Keeping the set small caps cache blow-up at N_files × len(ALLOWED_SIZES).
ALLOWED_SIZES: frozenset[int] = frozenset({150, 300, 800})

# WebP quality tuned for thumbnails: q=82 gives ~15-30 KB for a 300² tile,
# ~60-150 KB for 800². Lower than full-fidelity JPEGs on purpose — these
# are decorative previews, not viewing copies.
_WEBP_QUALITY = 82

# Raise PIL's decompression-bomb ceiling — but not to None. PIL's
# MAX_IMAGE_PIXELS is a module-global; setting it to None disables the
# zip-bomb defense for every Image.open in the worker, including
# untrusted upload paths in files_routes / image_search / enrichment /
# epub_extractor. A crafted tiny PNG with billion-pixel dimensions
# would then OOM the worker on decode.
#
# 500M pixels comfortably fits the largest legitimate generation
# (16384² = 268M) while still bombing on egregious uploads. ALLOWED_SIZES
# + thumbnail() cap memory during the final downscale step.
_DECOMPRESSION_CAP_PIXELS = 500_000_000
try:
    from PIL import Image as _PILImage  # noqa: N813 — private alias
    if (_PILImage.MAX_IMAGE_PIXELS or 0) < _DECOMPRESSION_CAP_PIXELS:
        _PILImage.MAX_IMAGE_PIXELS = _DECOMPRESSION_CAP_PIXELS
except ImportError:
    pass


class ThumbnailService:
    """Resolve a FileEntry → cached thumbnail bytes on disk.

    Constructed once at server startup with the thumbnail cache root;
    callers invoke :meth:`get` with a FileEntry + size and receive
    ``(path, mime_type)`` pointing at a ready-to-stream WebP.
    """

    def __init__(self, cache_root: str) -> None:
        self._root = Path(cache_root)

    async def get(
        self, entry: FileEntry, size: int, *, user_id: str,
    ) -> tuple[str, str] | None:
        """Return (cache_path, mime_type) for a thumb, producing if needed.

        Returns ``None`` when the source can't be thumbnailed — either the
        adapter has no producer and the mime isn't an image, the source
        row has vanished, or the producer raised. The caller 404s.
        """
        if size not in ALLOWED_SIZES:
            return None

        cache_path = self._cache_path(entry.source, entry.source_id, size)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return str(cache_path), "image/webp"

        data = await self._produce(entry, size, user_id=user_id)
        if not data:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, cache_path)
        except OSError:
            # Concurrent producer won; another request already wrote the
            # canonical file. Our tmp is orphaned — clean up and proceed.
            with _suppress_oserror():
                tmp.unlink()
            if not cache_path.exists():
                log.warning(
                    "thumbnail_write_failed",
                    source=entry.source, source_id=entry.source_id, size=size,
                )
                return None
        return str(cache_path), "image/webp"

    def purge(self, source: str, source_id: str) -> int:
        """Delete every cached size for one source row. Idempotent.

        Called from adapter delete paths so a purged image doesn't leave
        stale thumbs behind.
        """
        src_dir = self._root / source
        if not src_dir.is_dir():
            return 0
        removed = 0
        for f in src_dir.glob(f"{source_id}_*.webp"):
            with _suppress_oserror():
                f.unlink()
                removed += 1
        return removed

    def _cache_path(self, source: str, source_id: str, size: int) -> Path:
        return self._root / source / f"{source_id}_{size}.webp"

    async def _produce(
        self, entry: FileEntry, size: int, *, user_id: str,
    ) -> bytes | None:
        """Dispatch to an adapter's producer or the default image path."""
        from augmentum.vfs import get_adapter

        adapter = get_adapter(entry.source)
        if adapter is not None:
            producer = getattr(adapter, "produce_thumbnail", None)
            if producer is not None:
                try:
                    data = await producer(entry.source_id, size, user_id=user_id)
                    if data:
                        return data
                except Exception:
                    log.warning(
                        "thumbnail_producer_failed",
                        source=entry.source, source_id=entry.source_id,
                        exc_info=True,
                    )
                    return None

        # Default: if the file is an image, resolve the source and
        # downscale here. Covers `images` and `chat_images` without
        # adapter-level code.
        if not (entry.mime_type or "").startswith("image/"):
            return None
        if adapter is None:
            return None
        try:
            resolved = await adapter.resolve(entry.source_id, user_id=user_id)
        except Exception:
            log.warning(
                "thumbnail_resolve_failed",
                source=entry.source, source_id=entry.source_id, exc_info=True,
            )
            return None

        if isinstance(resolved, str):
            if not resolved or not os.path.exists(resolved):
                return None
            return await asyncio.to_thread(_downscale_path, resolved, size)
        if isinstance(resolved, (bytes, bytearray)):
            return await asyncio.to_thread(_downscale_bytes, bytes(resolved), size)
        return None


# ---------------------------------------------------------------------------
# PIL helpers — sync, run via asyncio.to_thread so CPU work doesn't stall
# the event loop when many tiles request thumbs at once.
# ---------------------------------------------------------------------------

def _downscale_path(path: str, size: int) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        log.warning("thumbnail_pillow_missing")
        return None
    try:
        with Image.open(path) as img:
            return _encode_webp(img, size)
    except Exception:
        log.warning("thumbnail_downscale_failed", path=path, exc_info=True)
        return None


def _downscale_bytes(raw: bytes, size: int) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        log.warning("thumbnail_pillow_missing")
        return None
    try:
        import io as _io
        with Image.open(_io.BytesIO(raw)) as img:
            return _encode_webp(img, size)
    except Exception:
        log.warning("thumbnail_downscale_bytes_failed", exc_info=True)
        return None


def _encode_webp(img, size: int) -> bytes:
    """Fit inside a size×size box, preserve aspect, encode as WebP."""
    import io as _io

    from PIL import Image

    # Honor EXIF orientation so portrait phone photos don't land sideways.
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception as exc:
        # Missing/corrupt EXIF — render the raw orientation; rarely
        # ideal but always safe.
        log.debug("thumbnail_exif_transpose_failed", error=str(exc))

    # Drop alpha into white — WebP handles alpha fine but small tiles look
    # cleaner against app backgrounds when we bake a neutral matte.
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode == "P" or img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((size, size), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="WEBP", quality=_WEBP_QUALITY, method=4)
    return buf.getvalue()


class _suppress_oserror:
    def __enter__(self): return self
    def __exit__(self, et, ev, tb): return isinstance(ev, OSError)
