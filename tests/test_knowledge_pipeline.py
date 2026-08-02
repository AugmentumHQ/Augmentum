"""Tests for knowledge pipeline — packs, catalog, importer."""

from __future__ import annotations

from pathlib import Path

from augmentum.knowledge.catalog import (
    CATEGORY_MAP,
    CatalogClient,
    CatalogEntry,
    _entry_from_dict,
    _sort_entries,
)
from augmentum.knowledge.importer import (
    ALL_SUPPORTED,
    IMPORTABLE_EXTS,
    ImportChunk,
    _extract_csv,
    _extract_json,
    detect_format,
)
from augmentum.knowledge.packs import PackManager, PackMeta, PackResult


class TestPackManager:
    """PackManager — knowledge pack lifecycle."""

    def test_pack_meta_defaults(self):
        meta = PackMeta()
        assert meta.name == ""
        assert meta.chunk_count == 0
        assert meta.embedding_dim == 0

    def test_pack_result_fields(self):
        r = PackResult(
            content="text", title="Title", section="S1",
            url="http://x", score=0.5, pack_id="p1", source="zim",
        )
        assert r.score == 0.5
        assert r.pack_id == "p1"

    async def test_scan_nonexistent_directory(self, tmp_path):
        mgr = PackManager(tmp_path / "nonexistent")
        count = await mgr.scan()
        assert count == 0

    async def test_scan_empty_directory(self, tmp_path):
        mgr = PackManager(tmp_path)
        count = await mgr.scan()
        assert count == 0

    def test_installed_empty(self):
        import tempfile
        mgr = PackManager(Path(tempfile.mkdtemp()))
        assert mgr.installed == []

    def test_active_count_empty(self):
        import tempfile
        mgr = PackManager(Path(tempfile.mkdtemp()))
        assert mgr.active_count == 0

    async def test_activate_nonexistent(self):
        import tempfile
        mgr = PackManager(Path(tempfile.mkdtemp()))
        assert await mgr.activate("nope") is False

    async def test_deactivate_nonexistent(self):
        import tempfile
        mgr = PackManager(Path(tempfile.mkdtemp()))
        assert await mgr.deactivate("nope") is False

    async def test_delete_nonexistent(self):
        import tempfile
        mgr = PackManager(Path(tempfile.mkdtemp()))
        assert await mgr.delete("nope") is False


class TestCatalog:
    """Kiwix OPDS catalog client."""

    def test_category_map_has_entries(self):
        assert len(CATEGORY_MAP) > 5

    def test_catalog_entry_display_size(self):
        entry = CatalogEntry(
            id="test", title="Test", description="",
            language="en", raw_category="wikipedia",
            article_count=1000, media_count=0,
            size_bytes=1024 * 1024 * 50,  # 50 MB
            download_url="", thumbnail_url="",
            issued_date="2026-01-01",
        )
        assert "MB" in entry.display_size

    def test_catalog_entry_category(self):
        entry = CatalogEntry(
            id="test", title="Test", description="",
            language="en", raw_category="wikipedia",
            article_count=0, media_count=0,
            size_bytes=0, download_url="",
            thumbnail_url="", issued_date="",
        )
        assert entry.category == "Wikipedia"

    def test_catalog_entry_license_wikipedia(self):
        entry = CatalogEntry(
            id="wikipedia_en", title="Wikipedia", description="",
            language="en", raw_category="wikipedia",
            article_count=0, media_count=0,
            size_bytes=0, download_url="",
            thumbnail_url="", issued_date="",
        )
        assert entry.license == "CC BY-SA"

    def test_catalog_entry_to_dict(self):
        entry = CatalogEntry(
            id="test", title="Test", description="desc",
            language="en", raw_category="wikipedia",
            article_count=100, media_count=0,
            size_bytes=1000, download_url="http://dl",
            thumbnail_url="http://thumb", issued_date="2026-01-01",
        )
        d = entry.to_dict()
        assert d["id"] == "test"
        assert "category" in d
        assert "display_size" in d

    def test_entry_from_dict_round_trip(self):
        original = CatalogEntry(
            id="rt", title="Round Trip", description="test",
            language="en", raw_category="devdocs",
            article_count=50, media_count=0,
            size_bytes=2000, download_url="http://x",
            thumbnail_url="", issued_date="2026-01-01",
            flavour="nopic",
        )
        d = original.to_dict()
        restored = _entry_from_dict(d)
        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.flavour == original.flavour

    def test_sort_entries_smallest(self):
        entries = [
            CatalogEntry(id="big", title="Big", description="", language="en",
                         raw_category="", article_count=0, media_count=0,
                         size_bytes=1000, download_url="", thumbnail_url="", issued_date=""),
            CatalogEntry(id="small", title="Small", description="", language="en",
                         raw_category="", article_count=0, media_count=0,
                         size_bytes=100, download_url="", thumbnail_url="", issued_date=""),
        ]
        sorted_entries = _sort_entries(entries, "smallest")
        assert sorted_entries[0].id == "small"

    def test_catalog_client_categories(self):
        client = CatalogClient()
        cats = client.categories()
        assert isinstance(cats, list)
        assert len(cats) > 0


class TestImporter:
    """Knowledge pack import from various file formats."""

    def test_detect_format_csv(self):
        assert detect_format("data.csv") == ".csv"

    def test_detect_format_json(self):
        assert detect_format("data.json") == ".json"

    def test_detect_format_unknown(self):
        assert detect_format("file.xyz") is None

    def test_detect_format_augpack(self):
        assert detect_format("pack.augpack") == ".augpack"

    def test_all_supported_includes_native(self):
        assert ".augpack" in ALL_SUPPORTED

    def test_importable_exts_includes_pdf(self):
        assert ".pdf" in IMPORTABLE_EXTS

    def test_extract_csv_basic(self):
        csv_data = b"title,content\nFirst,This is the first row content here\nSecond,Second row content text"
        chunks = _extract_csv(csv_data, "test.csv", "imported")
        assert len(chunks) >= 1
        assert isinstance(chunks[0], ImportChunk)

    def test_extract_json_array(self):
        json_data = b'[{"title": "A", "content": "Content of item A which is long enough"}]'
        chunks = _extract_json(json_data, "test.json", "imported")
        assert len(chunks) == 1
        assert chunks[0].title == "A"

    def test_extract_json_empty(self):
        json_data = b"[]"
        chunks = _extract_json(json_data, "test.json", "imported")
        assert len(chunks) == 0

    def test_import_chunk_defaults(self):
        chunk = ImportChunk(content="test content", title="Test")
        assert chunk.section == ""
        assert chunk.source == ""
        assert chunk.chunk_index == 0


