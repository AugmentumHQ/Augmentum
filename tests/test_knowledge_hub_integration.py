"""Integration tests for Knowledge Hub feature."""
from __future__ import annotations

import pytest


class TestSettingsWiring:
    """Verify all knowledge settings exist in all 4 layers."""

    def test_config_has_all_settings(self):
        from augmentum.config import settings
        assert hasattr(settings, "knowledge_packs_enabled")
        assert hasattr(settings, "knowledge_embedding_use_gpu")
        assert hasattr(settings, "knowledge_embedding_batch_size")
        assert hasattr(settings, "knowledge_catalog_cache_ttl")
        assert hasattr(settings, "knowledge_packs_custom_dir")
        assert hasattr(settings, "knowledge_featured_packs")

    def test_config_defaults(self):
        from augmentum.config import settings
        assert settings.knowledge_embedding_batch_size == 512
        assert settings.knowledge_catalog_cache_ttl == 86400

    def test_tool_settings_has_knowledge(self):
        from augmentum.proxy.config_routes import _TOOL_SETTINGS
        assert "knowledge_packs_enabled" in _TOOL_SETTINGS
        assert "knowledge_embedding_use_gpu" in _TOOL_SETTINGS

    def test_zim_embed_threshold_removed(self):
        """Auto-embed-on-install was removed 2026-05-07. The setting
        and its code path must stay gone — embedding is opt-in only
        via POST /api/knowledge/packs/{pack_id}/embed."""
        from augmentum.config import settings
        from augmentum.proxy.config_routes import _TOOL_SETTINGS
        assert not hasattr(settings, "knowledge_zim_embed_threshold")
        assert "knowledge_zim_embed_threshold" not in _TOOL_SETTINGS

    def test_string_settings_has_knowledge(self):
        from augmentum.proxy.config_routes import _STRING_SETTINGS
        assert "knowledge_packs_custom_dir" in _STRING_SETTINGS
        assert "knowledge_featured_packs" in _STRING_SETTINGS


class TestCatalogIntegration:
    @pytest.mark.asyncio
    async def test_browse_returns_dicts(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from augmentum.knowledge.catalog import CatalogClient, CatalogEntry

        client = CatalogClient(cache_dir=tmp_path, cache_ttl=3600)
        entries = [
            CatalogEntry(
                id="test.pack",
                title="Test",
                description="",
                language="eng",
                raw_category="wikipedia",
                article_count=500,
                media_count=0,
                size_bytes=1_000_000,
                download_url="https://example.com/test.zim",
                thumbnail_url="",
                issued_date="2026-01-01",
            )
        ]
        with patch.object(client, '_fetch_page', new_callable=AsyncMock, return_value=(entries, 1)):
            result = await client.browse(lang="eng")

        assert len(result) == 1
        d = result[0].to_dict()
        assert d["id"] == "test.pack"
        assert "display_size" in d


class TestZimReaderIntegration:
    def test_zim_reader_imports(self):
        from augmentum.knowledge.zim_reader import ZimReader  # noqa: F401
