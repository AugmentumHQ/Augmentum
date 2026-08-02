"""Phase 1 unified primitive layer — surface declaration tests.

Pins ``Tool.surfaces`` semantics, ``ToolRegistry.get_for_surface``
filtering, voice-manifest derivation from the registry, and the
auto-route dispatcher. See ``docs/superpowers/specs/2026-06-01-unified-
primitive-layer-design.md``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTool(Tool):
    """Configurable Tool used to exercise surface filtering."""

    def __init__(self, name: str, surfaces: SurfaceExposure | None = None) -> None:
        self._name = name
        self._surfaces = surfaces or SurfaceExposure()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy tool {self._name}"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return self._surfaces

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"ran {self._name}", metadata=dict(kwargs))


@pytest.fixture
def registry_with_dummies():
    """Registry pre-populated with one Tool per surface combination."""
    reg = ToolRegistry()
    reg.register(_DummyTool("default_chat_coder"))  # default surfaces
    reg.register(_DummyTool("voice_core", SurfaceExposure(voice="core")))
    reg.register(_DummyTool("voice_costly", SurfaceExposure(voice="costly")))
    reg.register(
        _DummyTool(
            "studio_only",
            SurfaceExposure(chat=False, coder=False, artifact_studio=True),
        )
    )
    reg.register(
        _DummyTool(
            "with_route",
            SurfaceExposure(http_route="/api/tools/with_route"),
        )
    )
    reg.register(
        _DummyTool(
            "companion_only",
            SurfaceExposure(chat=False, coder=False, companion=True),
        )
    )
    return reg


@pytest.fixture
def unbind_voice_registry():
    """Ensure voice manifest registry binding is reset around each test."""
    import augmentum.intent.manifest as manifest

    orig = manifest._registry
    yield
    manifest._registry = orig


# ---------------------------------------------------------------------------
# SurfaceExposure dataclass
# ---------------------------------------------------------------------------


class TestSurfaceExposure:
    def test_default_is_chat_and_coder(self):
        s = SurfaceExposure()
        assert s.chat is True
        assert s.coder is True
        assert s.voice is None
        assert s.companion is False
        assert s.artifact_studio is False
        assert s.http_route is None
        assert s.file_context_menu == ()

    def test_is_frozen(self):
        s = SurfaceExposure()
        with pytest.raises(Exception):
            s.chat = False  # type: ignore[misc]

    def test_tool_default_surfaces_matches_status_quo(self):
        # A Tool that doesn't override surfaces gets the default —
        # matches existing behavior of all already-registered Tools.
        t = _DummyTool("foo")
        assert t.surfaces == SurfaceExposure()


# ---------------------------------------------------------------------------
# get_for_surface
# ---------------------------------------------------------------------------


class TestGetForSurface:
    def test_chat_surface(self, registry_with_dummies):
        names = {t.name for t in registry_with_dummies.get_for_surface("chat")}
        assert "default_chat_coder" in names
        assert "voice_core" in names  # voice tools default chat=True too
        assert "studio_only" not in names  # explicitly chat=False
        assert "companion_only" not in names

    def test_voice_surface_all_levels(self, registry_with_dummies):
        names = {t.name for t in registry_with_dummies.get_for_surface("voice")}
        assert names == {"voice_core", "voice_costly"}

    def test_voice_surface_level_filter(self, registry_with_dummies):
        names = {
            t.name
            for t in registry_with_dummies.get_for_surface("voice", voice_level="core")
        }
        assert names == {"voice_core"}

    def test_coder_surface(self, registry_with_dummies):
        names = {t.name for t in registry_with_dummies.get_for_surface("coder")}
        assert "default_chat_coder" in names
        assert "studio_only" not in names

    def test_artifact_studio_surface(self, registry_with_dummies):
        names = {t.name for t in registry_with_dummies.get_for_surface("artifact_studio")}
        assert names == {"studio_only"}

    def test_companion_surface(self, registry_with_dummies):
        names = {t.name for t in registry_with_dummies.get_for_surface("companion")}
        assert names == {"companion_only"}

    def test_http_surface(self, registry_with_dummies):
        names = {t.name for t in registry_with_dummies.get_for_surface("http")}
        assert names == {"with_route"}

    def test_unknown_surface_returns_empty(self, registry_with_dummies):
        assert registry_with_dummies.get_for_surface("nonexistent") == []


# ---------------------------------------------------------------------------
# Voice manifest derivation
# ---------------------------------------------------------------------------


class TestVoiceManifestDerivation:
    def test_static_buckets_preserved_when_unbound(self, unbind_voice_registry):
        from augmentum.intent.manifest import (
            VOICE_TOOLS_CORE,
            VOICE_TOOLS_INTERACTIVE,
            all_voice_tools,
        )
        import augmentum.intent.manifest as manifest

        manifest._registry = None
        all_set = all_voice_tools()
        # All static-set members still present.
        assert "note.create" in all_set
        assert "navigate.open_surface" in all_set
        # Verb sets themselves are unchanged.
        assert "note.create" in VOICE_TOOLS_CORE
        assert "web_search" in VOICE_TOOLS_INTERACTIVE

    def test_runtime_tools_unioned_into_all_voice_tools(
        self, registry_with_dummies, unbind_voice_registry
    ):
        from augmentum.intent.manifest import all_voice_tools, bind_registry

        bind_registry(registry_with_dummies)
        union = all_voice_tools()
        assert "voice_core" in union
        assert "voice_costly" in union
        # Tools that opted out of voice stay absent.
        assert "default_chat_coder" not in union
        assert "studio_only" not in union

    def test_safe_policy_includes_runtime_core_and_interactive(
        self, registry_with_dummies, unbind_voice_registry
    ):
        from augmentum.intent.manifest import bind_registry, voice_tools_for

        bind_registry(registry_with_dummies)
        safe = voice_tools_for(ambient=True, policy="safe")
        assert "voice_core" in safe
        # No registered interactive tools in this fixture; static set still present.
        assert "web_search" in safe
        # Costly is filtered.
        assert "voice_costly" not in safe

    def test_minimal_policy_is_core_only(
        self, registry_with_dummies, unbind_voice_registry
    ):
        from augmentum.intent.manifest import bind_registry, voice_tools_for

        bind_registry(registry_with_dummies)
        minimal = voice_tools_for(ambient=True, policy="minimal")
        assert "voice_core" in minimal
        assert "voice_costly" not in minimal
        # Interactive is also out.
        assert "web_search" not in minimal

    def test_capability_line_falls_back_to_tool_declaration(
        self, registry_with_dummies, unbind_voice_registry
    ):
        from augmentum.intent.manifest import bind_registry, capability_line

        registry_with_dummies.register(
            _DummyTool(
                "fancy_tool",
                SurfaceExposure(voice="core", voice_capability_line="do a fancy thing"),
            )
        )
        bind_registry(registry_with_dummies)
        assert capability_line("fancy_tool") == "do a fancy thing"


# ---------------------------------------------------------------------------
# Migrated tools — surfaces declarations are wired
# ---------------------------------------------------------------------------


class TestMigratedToolSurfaces:
    def test_hash_tool_is_voice_core(self):
        from augmentum.tools.hash_tool import HashTool

        tool = HashTool()
        assert tool.surfaces.chat is True
        assert tool.surfaces.voice == "core"
        assert tool.surfaces.coder is True
        assert tool.surfaces.voice_capability_line

    def test_image_convert_declares_full_surface(self):
        from augmentum.tools.image_convert import ImageConvertTool

        store = MagicMock()
        tool = ImageConvertTool(store)
        s = tool.surfaces
        assert s.chat and s.voice == "core" and s.coder and s.artifact_studio
        assert "image/*" in s.file_context_menu
        assert s.http_route == "/api/tools/convert_image"

    def test_background_remove_is_costly(self):
        from augmentum.tools.background_remove import BackgroundRemoveTool

        store = MagicMock()
        tool = BackgroundRemoveTool(store)
        assert tool.surfaces.voice == "costly"
        assert tool.surfaces.artifact_studio is True

    def test_document_convert_declares_extensions(self):
        from augmentum.tools.document_convert import DocumentConvertTool

        store = MagicMock()
        tool = DocumentConvertTool(store)
        s = tool.surfaces
        assert ".pdf" in s.file_context_menu
        assert ".docx" in s.file_context_menu
        assert s.http_route == "/api/tools/convert_document"

    def test_csv_export_artifact_studio(self):
        from augmentum.tools.export_tools import CsvExportTool

        store = MagicMock()
        tool = CsvExportTool(store)
        assert tool.surfaces.artifact_studio is True
        assert ".csv" in tool.surfaces.file_context_menu


# ---------------------------------------------------------------------------
# HTTP delegation — _convert_image / _convert_document still work
# ---------------------------------------------------------------------------


class TestHttpDelegation:
    def test_convert_image_delegates_to_tool(self, tmp_path):
        from PIL import Image

        from augmentum.proxy.artifact_routes import _convert_image

        # Make a 4x4 RGBA PNG so the JPG path exercises the alpha
        # flattening branch.
        src = tmp_path / "src.png"
        Image.new("RGBA", (4, 4), (255, 0, 0, 128)).save(src)

        data, ext = asyncio.run(_convert_image(src, "jpg"))
        assert ext == "jpg"
        assert data.startswith(b"\xff\xd8")  # JPEG magic

    def test_convert_image_rejects_unknown_target(self, tmp_path):
        from PIL import Image

        from augmentum.proxy.artifact_routes import _convert_image

        src = tmp_path / "src.png"
        Image.new("RGB", (2, 2)).save(src)

        with pytest.raises(ValueError):
            asyncio.run(_convert_image(src, "bmp"))

    def test_convert_document_requires_source_json(self):
        from fastapi import HTTPException

        from augmentum.proxy.artifact_routes import _convert_document

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_convert_document({"display_name": "no source"}, "pdf"))
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Auto-route registration
# ---------------------------------------------------------------------------


class TestAutoRoutes:
    def test_register_tool_routes_binds_only_http_route_tools(
        self, registry_with_dummies
    ):
        from fastapi import FastAPI

        from augmentum.tools.auto_routes import register_tool_routes

        app = FastAPI()
        bound = register_tool_routes(app, registry_with_dummies)
        assert bound == ["/api/tools/with_route"]

        # Idempotent: second call binds nothing new.
        bound2 = register_tool_routes(app, registry_with_dummies)
        assert bound2 == []

    def test_dispatcher_runs_tool_and_returns_metadata(self, registry_with_dummies):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from augmentum.tools.auto_routes import register_tool_routes

        app = FastAPI()
        register_tool_routes(app, registry_with_dummies)

        client = TestClient(app)
        resp = client.post("/api/tools/with_route", json={"foo": "bar"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        # _context.user_id is injected by the dispatcher.
        assert body["metadata"]["foo"] == "bar"
        assert body["metadata"]["_context"]["user_id"] == ""

    def test_dispatcher_rejects_non_object_body(self, registry_with_dummies):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from augmentum.tools.auto_routes import register_tool_routes

        app = FastAPI()
        register_tool_routes(app, registry_with_dummies)

        client = TestClient(app)
        resp = client.post("/api/tools/with_route", json=["not", "an", "object"])
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Pinning test — every Tool has a surfaces property that returns SurfaceExposure
# ---------------------------------------------------------------------------


class TestSurfacesPinning:
    def test_default_surface_is_chat_and_coder_only(self):
        # Sanity check the documented default — guards against a refactor
        # silently widening default exposure.
        s = SurfaceExposure()
        assert s.chat is True
        assert s.coder is True
        # Default does NOT expose to voice, companion, artifact_studio,
        # or auto-bind an HTTP route.
        assert s.voice is None
        assert s.companion is False
        assert s.artifact_studio is False
        assert s.http_route is None

    def test_phase1_migrated_tools_have_explicit_surfaces(self):
        """The 5 Phase-1 migration targets all declare surfaces explicitly
        (i.e. not the inherited default). Regression guard for the spec."""
        from augmentum.tools.hash_tool import HashTool

        store = MagicMock()
        from augmentum.tools.background_remove import BackgroundRemoveTool
        from augmentum.tools.document_convert import DocumentConvertTool
        from augmentum.tools.export_tools import CsvExportTool
        from augmentum.tools.image_convert import ImageConvertTool

        tools = [
            HashTool(),
            ImageConvertTool(store),
            BackgroundRemoveTool(store),
            DocumentConvertTool(store),
            CsvExportTool(store),
        ]
        default = SurfaceExposure()
        for tool in tools:
            assert tool.surfaces != default, (
                f"{tool.name} still uses the default SurfaceExposure — Phase-1 "
                "migration targets must declare an explicit surface set."
            )
