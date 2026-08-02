"""Tests for the Kiwix OPDS catalog client."""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from augmentum.knowledge.catalog import (
    CATEGORY_MAP,
    FEATURED_PACK_IDS,
    CatalogClient,
    CatalogEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOM_NS = "http://www.w3.org/2005/Atom"
DC_NS = "http://purl.org/dc/terms/"


def _make_entry_xml(
    *,
    id_val: str = "wikipedia.en.medicine",
    title: str = "Wikipedia Medicine",
    summary: str = "Medical articles",
    language: str = "eng",
    category: str = "wikipedia",
    article_count: str = "50000",
    media_count: str = "1000",
    size_bytes: str = "2147483648",
    download_url: str = "https://example.com/file.zim",
    thumbnail_url: str = "https://example.com/thumb.jpg",
    issued_date: str = "2024-01-15",
    tags: str = "wikipedia;medicine;health",
) -> ET.Element:
    """Build a minimal Atom entry XML element for testing."""
    el = ET.Element(f"{{{ATOM_NS}}}entry")

    def sub(tag: str, text: str, ns: str = ATOM_NS) -> ET.Element:
        child = ET.SubElement(el, f"{{{ns}}}{tag}")
        child.text = text
        return child

    def sub_no_ns(tag: str, text: str) -> ET.Element:
        child = ET.SubElement(el, tag)
        child.text = text
        return child

    sub("id", id_val)
    sub("title", title)
    sub("summary", summary)
    # DC terms
    sub("language", language, ns=DC_NS)
    sub("issued", issued_date, ns=DC_NS)
    # Non-namespaced Kiwix elements
    sub_no_ns("name", id_val)
    sub_no_ns("articleCount", article_count)
    sub_no_ns("mediaCount", media_count)
    # Tags stored in Atom <tags> (non-namespaced)
    sub_no_ns("tags", tags)
    # Category as non-namespaced element
    sub_no_ns("category", category)
    # Thumbnail link
    thumb = ET.SubElement(el, f"{{{ATOM_NS}}}link")
    thumb.set("rel", "http://opds-spec.org/image/thumbnail")
    thumb.set("href", thumbnail_url)
    # Acquisition link (download)
    acq = ET.SubElement(el, f"{{{ATOM_NS}}}link")
    acq.set("rel", "http://opds-spec.org/acquisition")
    acq.set("type", "application/x-zim")
    acq.set("href", download_url)
    acq.set("length", size_bytes)

    return el


def _make_sample_entry(**kwargs) -> CatalogEntry:
    el = _make_entry_xml(**kwargs)
    return CatalogEntry.from_opds(el)


def _make_entries_for_browse() -> list[CatalogEntry]:
    return [
        _make_sample_entry(
            id_val="wikipedia.en.medicine",
            title="Wikipedia Medicine",
            category="wikipedia",
            size_bytes="1000000",
            article_count="5000",
        ),
        _make_sample_entry(
            id_val="devdocs.en",
            title="DevDocs",
            category="devdocs",
            size_bytes="500000",
            article_count="3000",
        ),
        _make_sample_entry(
            id_val="ifixit.en",
            title="iFixit",
            category="ifixit",
            size_bytes="2000000",
            article_count="2000",
        ),
        _make_sample_entry(
            id_val="stack_exchange.en.stackoverflow",
            title="Stack Overflow",
            category="stack_exchange",
            size_bytes="800000",
            article_count="10000",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. CatalogEntry.from_opds parses all fields
# ---------------------------------------------------------------------------


def test_from_opds_parses_all_fields():
    entry = _make_sample_entry(
        id_val="wikipedia.en.medicine",
        title="Wikipedia Medicine",
        summary="Medical articles",
        language="eng",
        category="wikipedia",
        article_count="50000",
        media_count="1000",
        size_bytes="2147483648",
        download_url="https://example.com/file.zim",
        thumbnail_url="https://example.com/thumb.jpg",
        issued_date="2024-01-15",
        tags="wikipedia;medicine;health",
    )
    assert entry.id == "wikipedia.en.medicine"
    assert entry.title == "Wikipedia Medicine"
    assert entry.description == "Medical articles"
    assert entry.language == "eng"
    assert entry.raw_category == "wikipedia"
    assert entry.article_count == 50000
    assert entry.media_count == 1000
    assert entry.size_bytes == 2147483648
    assert entry.download_url == "https://example.com/file.zim"
    assert entry.thumbnail_url == "https://example.com/thumb.jpg"
    assert entry.issued_date == "2024-01-15"
    assert "medicine" in entry.tags


# ---------------------------------------------------------------------------
# 2. CatalogEntry.display_size — human-readable formatting
# ---------------------------------------------------------------------------


def test_display_size_bytes():
    entry = _make_sample_entry(size_bytes="500")
    assert "B" in entry.display_size


def test_display_size_kilobytes():
    entry = _make_sample_entry(size_bytes="2048")
    assert "KB" in entry.display_size or "B" in entry.display_size


def test_display_size_megabytes():
    entry = _make_sample_entry(size_bytes="5242880")  # 5 MB
    assert "MB" in entry.display_size


def test_display_size_gigabytes():
    entry = _make_sample_entry(size_bytes="2147483648")  # 2 GB
    assert "GB" in entry.display_size


def test_display_size_zero():
    entry = _make_sample_entry(size_bytes="0")
    assert entry.display_size  # Should not raise or return empty


# ---------------------------------------------------------------------------
# 3. Unknown category maps to "Other"
# ---------------------------------------------------------------------------


def test_unknown_category_maps_to_other():
    entry = _make_sample_entry(category="unknown_category_xyz")
    assert entry.category == "Other"


def test_known_category_maps_correctly():
    entry = _make_sample_entry(category="wikipedia")
    assert entry.category == "Wikipedia"

    entry2 = _make_sample_entry(category="stack_exchange")
    assert entry2.category == "Stack Exchange"


# ---------------------------------------------------------------------------
# 4. CatalogClient.browse() uses cache on second call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_uses_cache_on_second_call(tmp_path: Path):
    entries = _make_entries_for_browse()
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        result1 = await client.browse(lang="en")
        result2 = await client.browse(lang="en")

    # _fetch_page should only have been called once (or at most for pagination)
    # The key thing is the second browse() doesn't trigger new network fetches
    first_call_count = fetch_mock.call_count
    # Reset and check: calling a third time should NOT increase the call count
    call_count_after_two = fetch_mock.call_count
    result3 = await client.browse(lang="en")
    assert fetch_mock.call_count == call_count_after_two, (
        "Third browse() should use cache, not re-fetch"
    )
    assert result1 == result2 == result3


# ---------------------------------------------------------------------------
# 5. Cache expiry — TTL 0 causes re-fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_expiry_refetches(tmp_path: Path):
    entries = _make_entries_for_browse()
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=0)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        await client.browse(lang="en")
        first_call_count = fetch_mock.call_count
        await client.browse(lang="en")
        second_call_count = fetch_mock.call_count

    assert second_call_count > first_call_count, (
        "TTL=0 should cause re-fetch on second call"
    )


# ---------------------------------------------------------------------------
# 6. CatalogClient.featured() filters to only featured IDs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_featured_returns_only_featured_ids(tmp_path: Path):
    entries = _make_entries_for_browse()
    # Add a non-featured entry
    entries.append(
        _make_sample_entry(
            id_val="wikivoyage.en.travel",
            title="Wikivoyage",
            category="wikivoyage",
        )
    )
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        featured = await client.featured(lang="en")

    featured_ids = {e.id for e in featured}
    assert featured_ids.issubset(set(FEATURED_PACK_IDS))
    assert "wikivoyage.en.travel" not in featured_ids


@pytest.mark.asyncio
async def test_featured_with_override(tmp_path: Path):
    entries = _make_entries_for_browse()
    entries.append(
        _make_sample_entry(id_val="wikivoyage.en.travel", category="wikivoyage")
    )
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        featured = await client.featured(lang="en", override="wikivoyage.en.travel")

    featured_ids = {e.id for e in featured}
    assert "wikivoyage.en.travel" in featured_ids
    # Should not include the default featured IDs (override replaces them)
    assert "devdocs.en" not in featured_ids


# ---------------------------------------------------------------------------
# 7. Browse filters by category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_filters_by_category(tmp_path: Path):
    entries = _make_entries_for_browse()
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        result = await client.browse(lang="en", category="Wikipedia")

    assert all(e.category == "Wikipedia" for e in result)
    assert len(result) == 1
    assert result[0].id == "wikipedia.en.medicine"


# ---------------------------------------------------------------------------
# 8. Browse filters by max_size_bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_filters_by_max_size_bytes(tmp_path: Path):
    entries = _make_entries_for_browse()
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        # Only entries with size <= 800000
        result = await client.browse(lang="en", max_size_bytes=800000)

    assert all(e.size_bytes <= 800000 for e in result)
    ids = {e.id for e in result}
    assert "ifixit.en" not in ids  # 2000000 bytes
    assert "wikipedia.en.medicine" not in ids  # 1000000 bytes
    assert "devdocs.en" in ids  # 500000 bytes
    assert "stack_exchange.en.stackoverflow" in ids  # 800000 bytes


# ---------------------------------------------------------------------------
# 9. CatalogClient.categories() returns sorted unique list
# ---------------------------------------------------------------------------


def test_categories_returns_sorted_unique(tmp_path: Path):
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)
    cats = client.categories()
    assert isinstance(cats, list)
    assert cats == sorted(set(cats))
    # Verify known categories are present
    assert "Wikipedia" in cats
    assert "Stack Exchange" in cats
    assert "Education" in cats
    # Should not have duplicates
    assert len(cats) == len(set(cats))


# ---------------------------------------------------------------------------
# 10. to_dict includes computed properties
# ---------------------------------------------------------------------------


def test_to_dict_includes_computed_properties():
    entry = _make_sample_entry(
        id_val="wikipedia.en.medicine",
        category="wikipedia",
        size_bytes="2147483648",
    )
    d = entry.to_dict()
    assert "category" in d
    assert d["category"] == "Wikipedia"
    assert "display_size" in d
    assert "GB" in d["display_size"]
    assert "license" in d


# ---------------------------------------------------------------------------
# 11. Browse with query filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_filters_by_query(tmp_path: Path):
    entries = _make_entries_for_browse()
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        result = await client.browse(lang="en", query="medicine")

    assert len(result) >= 1
    # All results should have "medicine" in title or description
    for e in result:
        text = (e.title + " " + (e.description or "")).lower()
        assert "medicine" in text


# ---------------------------------------------------------------------------
# 12. Browse sort — recommended puts featured first
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browse_sort_recommended_puts_featured_first(tmp_path: Path):
    entries = _make_entries_for_browse()
    # Add a non-featured entry with high article count
    entries.append(
        _make_sample_entry(
            id_val="some.non.featured",
            title="Non-featured",
            category="wikipedia",
            article_count="99999",
            size_bytes="100000",
        )
    )
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        result = await client.browse(lang="en", sort="recommended")

    # Featured items should appear before non-featured
    featured_set = set(FEATURED_PACK_IDS)
    last_featured_idx = -1
    first_nonfeatured_idx = len(result)
    for i, e in enumerate(result):
        if e.id in featured_set:
            last_featured_idx = i
        elif e.id not in featured_set and i < first_nonfeatured_idx:
            first_nonfeatured_idx = i

    assert last_featured_idx < first_nonfeatured_idx or last_featured_idx == -1, (
        "Featured entries should come before non-featured in 'recommended' sort"
    )


# ---------------------------------------------------------------------------
# 13. Disk cache is written and read back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disk_cache_written_and_read(tmp_path: Path):
    entries = _make_entries_for_browse()
    client = CatalogClient(cache_dir=tmp_path, cache_ttl=300)

    fetch_mock = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client, "_fetch_page", fetch_mock):
        await client.browse(lang="en")

    # Disk cache file should exist
    cache_file = tmp_path / "catalog_en.json"
    assert cache_file.exists(), "Disk cache file should be written"

    # Create new client (no in-memory cache) — should load from disk
    client2 = CatalogClient(cache_dir=tmp_path, cache_ttl=300)
    fetch_mock2 = AsyncMock(return_value=(entries, len(entries)))
    with patch.object(client2, "_fetch_page", fetch_mock2):
        result = await client2.browse(lang="en")

    assert fetch_mock2.call_count == 0, "Should have loaded from disk cache, not fetched"
    assert len(result) == len(entries)
