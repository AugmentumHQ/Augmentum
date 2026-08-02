"""Tests for VFS bridge resolution."""

from __future__ import annotations


class TestFilesRoutes:
    def test_import(self):
        from augmentum.proxy.files_routes import router
        assert router is not None

    def test_router_prefix(self):
        from augmentum.proxy.files_routes import router
        assert router.prefix == "/api/files"

    def test_has_expected_routes(self):
        from augmentum.proxy.files_routes import router
        paths = {r.path for r in router.routes}
        assert "/api/files/search" in paths
        assert "/api/files/stats" in paths
        assert "/api/files/browse" in paths


class TestSearchFilesTool:
    def test_import(self):
        from augmentum.tools.search_files import SearchFilesTool
        tool = SearchFilesTool()
        assert tool.name == "search_files"


class TestWebDAVModule:
    def test_import(self):
        try:
            from augmentum.vfs.webdav import create_webdav_app
            assert callable(create_webdav_app)
        except ImportError:
            pass  # wsgidav may not be installed


class TestVFSModels:
    def test_file_entry_card_with_metadata(self):
        from augmentum.vfs.models import FileEntry
        e = FileEntry(
            id="fi_1", user_id="usr_1", source="images", source_id="img_1",
            name="sunset.png", mime_type="image/png", size_bytes=3_200_000,
            description="Beautiful sunset", created_at="2026-03-15",
            source_metadata={"prompt": "sunset over ocean", "model": "sdxl"},
        )
        card = e.to_card()
        assert "sunset.png" in card
        assert "Beautiful sunset" in card
        assert "Prompt:" in card
        assert "Model: sdxl" in card


class TestVFSRouter:
    def test_resolve_root(self):
        from augmentum.vfs.bridges import VFS
        vfs = VFS()
        # No bridges registered — root listing is empty
        import asyncio
        nodes = asyncio.get_event_loop().run_until_complete(
            vfs.list("/", user_id="usr_1")
        )
        assert nodes == []

    def test_resolve_bridge_prefix(self):
        from augmentum.vfs.bridges import VFS, VFSBridge
        vfs = VFS()

        class FakeBridge(VFSBridge):
            prefix = "/Test"
            source = "test"

        vfs.register_bridge(FakeBridge(None))
        bridge, subpath = vfs._resolve("/Test/file.txt")
        assert bridge is not None
        assert bridge.source == "test"
        assert subpath == "/file.txt"

    def test_resolve_unknown_path(self):
        from augmentum.vfs.bridges import VFS
        vfs = VFS()
        bridge, subpath = vfs._resolve("/Unknown/path")
        assert bridge is None

    def test_root_lists_bridges(self):
        import asyncio

        from augmentum.vfs.bridges import VFS, VFSBridge

        class FakeA(VFSBridge):
            prefix = "/Alpha"
            source = "alpha"

        class FakeB(VFSBridge):
            prefix = "/Beta"
            source = "beta"

        vfs = VFS()
        vfs.register_bridge(FakeA(None))
        vfs.register_bridge(FakeB(None))
        nodes = asyncio.get_event_loop().run_until_complete(
            vfs.list("/", user_id="usr_1")
        )
        assert len(nodes) == 2
        assert all(n.is_dir for n in nodes)
        names = {n.name for n in nodes}
        assert "Alpha" in names
        assert "Beta" in names
