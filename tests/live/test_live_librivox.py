"""Live test for LibriVox provider — hits the real LibriVox + archive.org APIs.

Run: pytest tests/live/test_live_librivox.py --run-live -v

Unlike the other live tests in this directory, this one doesn't require a
running Augmentum server. It only needs outbound HTTPS to librivox.org
and archive.org. Skipped when either upstream is unreachable so CI on
an offline runner doesn't flake.

The pinned book id ``pride_and_prejudice_librivox`` is the canonical
archive.org identifier (verified live 2026-04-20). It's been on
archive.org for ~15 years — if it ever disappears, every LibriVox app
breaks simultaneously, so the coupling is acceptable.
"""

from __future__ import annotations

import httpx
import pytest

from augmentum.media.providers.base import BrowseResult
from augmentum.media.providers.librivox import (
    LibrivoxProvider,
    normalise_details_to_catalog,
    normalise_librivox_sections,
)

pytestmark = pytest.mark.live

STABLE_ID = "pride_and_prejudice_librivox"
STABLE_LIBRIVOX_ID = "253"


async def _reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
            a = await c.get("https://librivox.org/api/feed/audiobooks",
                            params={"format": "json", "limit": 1})
            b = await c.get(f"https://archive.org/metadata/{STABLE_ID}")
            return a.status_code == 200 and b.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
async def reachable():
    if not await _reachable():
        pytest.skip("LibriVox or archive.org unreachable from this runner")


async def test_browse_returns_real_books(reachable):
    """Smoke: the feed API returns at least one book with an archive id."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        results = await p.browse(query="austen", page_size=5)
    assert results, "Expected at least one LibriVox result for 'austen'"
    for r in results:
        assert r.external_id
        assert r.name
        assert r.cover_url.startswith("https://archive.org/services/img/")


async def test_browse_empty_query_returns_first_page(reachable):
    """No query = default first page of the catalog (used for 'browse all')."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        results = await p.browse(query="", page_size=5)
    assert results, "Default feed page should return books"


# Note: no live test for "empty results" — LibriVox's `search` param is
# lenient and returns the default page for nonsense input instead of a
# true empty result. The 404-handling path is covered by the offline
# test (test_browse_404_returns_empty_list) via mocked httpx.


async def test_fetch_book_by_id_returns_sections(reachable):
    """LibriVox feed returns the full book record including sections + readers."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        book = await p.fetch_book_by_id(STABLE_LIBRIVOX_ID)
    assert book is not None
    sections = book.get("sections") or []
    assert len(sections) >= 30, "P&P has ~37 sections — expected lots"
    # At least one section must have a reader populated, else the
    # "per-chapter narrator" enrichment is dead weight.
    any_with_reader = any(
        s.get("readers") for s in sections if isinstance(s, dict)
    )
    assert any_with_reader, "Expected at least one section with readers[] populated"


async def test_sections_path_produces_playable_chapters(reachable):
    """Primary pin path: LibriVox feed → sections → chapters with narrators."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        book = await p.fetch_book_by_id(STABLE_LIBRIVOX_ID)
    assert book is not None

    br = BrowseResult(
        external_id=STABLE_ID,
        name="Pride and Prejudice",
        author="Jane Austen",
        license="public-domain",
        extra={"language": "English", "librivox_id": STABLE_LIBRIVOX_ID},
    )
    detail = normalise_librivox_sections(librivox_book=book, browse_result=br)
    assert detail["audio_files"], "Expected at least one chapter MP3"
    assert detail["chapters"], "Expected computed chapter offsets"
    # Contiguous, monotonically-increasing offsets.
    starts = [ch["start"] for ch in detail["chapters"]]
    assert starts == sorted(starts)
    # At least one chapter must have per-section narrators — that's the
    # primary win of this code path over archive.org /metadata.
    assert any(ch.get("narrators") for ch in detail["chapters"])
    # Total duration should be > 8 hours (typical LibriVox P&P runtime).
    assert detail["duration_ms"] > 8 * 3600 * 1000


async def test_archive_fallback_path_still_works(reachable):
    """Fallback pin path: archive.org /metadata → chapters (no readers)."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        meta = await p.fetch_item_details("", "", external_id=STABLE_ID)
    assert meta is not None
    assert "files" in meta
    br = BrowseResult(
        external_id=STABLE_ID, name="P&P", license="public-domain",
    )
    detail = normalise_details_to_catalog(archive_meta=meta, browse_result=br)
    assert detail["audio_files"]
    assert detail["chapters"]


async def test_stream_url_is_reachable(reachable):
    """The URL build_stream_url produces must actually serve bytes with Range."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        book = await p.fetch_book_by_id(STABLE_LIBRIVOX_ID)
    assert book is not None
    br = BrowseResult(external_id=STABLE_ID, name="P&P")
    detail = normalise_librivox_sections(librivox_book=book, browse_result=br)

    first_file = detail["audio_files"][0]["name"]
    stream_path = f"{STABLE_ID}/{first_file}"
    url = LibrivoxProvider(httpx.AsyncClient()).build_stream_url("", stream_path, "")

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
        resp = await c.get(url, headers={"Range": "bytes=0-1023"})
    # 206 Partial Content expected; 200 acceptable if CDN strips Range;
    # 503 happens under archive.org load — retry once.
    if resp.status_code == 503:
        import asyncio
        await asyncio.sleep(2)
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            resp = await c.get(url, headers={"Range": "bytes=0-1023"})
    assert resp.status_code in (200, 206), f"Unexpected {resp.status_code} from {url}"


async def test_cover_url_resolves(reachable):
    """Archive.org generates a default cover for every item."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
        p = LibrivoxProvider(c)
        url = p.build_cover_url("", STABLE_ID, "")
        resp = await c.get(url)
    assert resp.status_code < 400, f"Cover URL returned {resp.status_code}"
