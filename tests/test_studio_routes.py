"""Tests for studio_routes.py — artifact studio editing endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def _mock_artifact_store():
    store = MagicMock()
    store.get = AsyncMock(return_value={
        "id": "art_1",
        "filename": "report.pdf",
        "display_name": "Report",
        "format": "pdf",
        "size_bytes": 1024,
        "source_json": json.dumps({"type": "document", "title": "Report", "sections": []}),
        "metadata": {},
        "download_url": "/api/artifacts/art_1/download",
    })
    store.update_source = AsyncMock()
    store.update_file = AsyncMock()
    return store


class TestThemes:
    def test_list_themes(self, client):
        resp = client.get("/api/studio/themes/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "themes" in data
        assert isinstance(data["themes"], list)


class TestGetArtifact:
    def test_get_artifact_no_store(self, client):
        resp = client.get("/api/studio/art_1")
        assert resp.status_code == 503

    def test_get_artifact_not_found(self, app, client):
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        app.state.artifact_store = store
        resp = client.get("/api/studio/nonexistent")
        assert resp.status_code == 404

    def test_get_artifact_success(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.get("/api/studio/art_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "art_1"
        assert data["format"] == "pdf"
        assert "source" in data


class TestSaveArtifact:
    def test_save_no_store(self, client):
        resp = client.post("/api/studio/art_1/save", json={"source": {}})
        assert resp.status_code == 503

    def test_save_not_found(self, app, client):
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/save", json={"source": {}})
        assert resp.status_code == 404

    def test_save_missing_source(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post("/api/studio/art_1/save", json={})
        assert resp.status_code == 400

    def test_save_unknown_type(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post(
            "/api/studio/art_1/save",
            json={"source": {"type": "unknown_type"}},
        )
        assert resp.status_code == 400
        assert "Unknown source type" in resp.json()["error"]

    def test_save_ebook_rerenders_epub(self, app, client):
        store = _mock_artifact_store()
        app.state.artifact_store = store

        resp = client.post(
            "/api/studio/art_1/save",
            json={
                "source": {
                    "type": "ebook",
                    "title": "Edited Book",
                    "author": "A. Writer",
                    "chapters": [
                        {"heading": "Opening", "body": "Once upon a test."},
                    ],
                },
            },
        )

        assert resp.status_code == 200
        assert resp.json()["size_bytes"] > 0
        store.update_source.assert_awaited_once()
        saved_source = json.loads(store.update_source.await_args.args[1])
        assert saved_source["type"] == "ebook"
        assert saved_source["chapters"][0]["heading"] == "Opening"
        store.update_file.assert_awaited_once()
        assert store.update_file.await_args.args[1].startswith(b"PK")

    def test_save_ebook_resolves_artifact_images_with_user_scope(self, app, client, tmp_path):
        from PIL import Image as PILImage

        cover_path = tmp_path / "cover.png"
        scene_path = tmp_path / "scene.png"
        PILImage.new("RGB", (80, 120), color="blue").save(cover_path)
        PILImage.new("RGB", (80, 80), color="green").save(scene_path)

        store = MagicMock()
        store.get = AsyncMock(side_effect=[
            {
                "id": "art_1",
                "filename": "Story.epub",
                "display_name": "Story",
                "format": "epub",
                "size_bytes": 1024,
                "source_json": json.dumps({"type": "ebook", "title": "Story", "chapters": []}),
                "metadata": {},
                "download_url": "/api/artifacts/art_1/download",
            },
            {"path": "standalone/scene.png"},
            {"path": "standalone/cover.png"},
        ])
        store.get_file_path = MagicMock(
            side_effect=lambda rel: scene_path if "scene" in rel else cover_path
        )
        store.update_source = AsyncMock()
        store.update_file = AsyncMock()
        app.state.artifact_store = store

        resp = client.post(
            "/api/studio/art_1/save",
            json={
                "source": {
                    "type": "ebook",
                    "title": "Illustrated",
                    "author": "A. Writer",
                    "cover_image_url": "/api/artifacts/cover_art/download",
                    "chapters": [
                        {
                            "heading": "Opening",
                            "body": "Once upon a scoped test.",
                            "image_url": "/api/artifacts/scene_art/download",
                        },
                    ],
                },
            },
        )

        assert resp.status_code == 200
        assert store.get.await_args_list[1].args == ("scene_art",)
        assert store.get.await_args_list[1].kwargs == {"user_id": "usr_test"}
        assert store.get.await_args_list[2].args == ("cover_art",)
        assert store.get.await_args_list[2].kwargs == {"user_id": "usr_test"}


class TestPreviewArtifact:
    def test_preview_no_store(self, client):
        resp = client.post("/api/studio/art_1/preview", json={"source": {}})
        assert resp.status_code == 503

    def test_preview_not_found(self, app, client):
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/preview", json={"source": {}})
        assert resp.status_code == 404

    def test_preview_missing_source(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post("/api/studio/art_1/preview", json={})
        assert resp.status_code == 400

    def test_preview_ebook_returns_epub(self, app, client):
        store = _mock_artifact_store()
        store.get = AsyncMock(return_value={
            "id": "art_1",
            "filename": "Story.epub",
            "display_name": "Story",
            "format": "epub",
            "size_bytes": 1024,
            "source_json": json.dumps({"type": "ebook", "title": "Story", "chapters": []}),
            "metadata": {},
            "download_url": "/api/artifacts/art_1/download",
        })
        app.state.artifact_store = store

        resp = client.post(
            "/api/studio/art_1/preview",
            json={
                "source": {
                    "type": "ebook",
                    "format": "epub",
                    "title": "Preview Book",
                    "chapters": [{"heading": "Opening", "body": "Preview text."}],
                },
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/epub+zip")
        assert resp.content.startswith(b"PK")


# ===========================================================================
# Phase 1 Tool Palette substrate — Image tool endpoints
# ===========================================================================


def _mock_tool_result(success=True, output="", metadata=None, error=""):
    """Build a minimal ToolResult-shaped object the routes can read."""
    res = MagicMock()
    res.success = success
    res.output = output
    res.metadata = metadata or {}
    res.error = error
    return res


class TestSearchImages:
    def test_search_no_store(self, client):
        resp = client.post("/api/studio/art_1/search-images", json={"query": "solar"})
        assert resp.status_code == 503

    def test_search_artifact_not_found(self, app, client):
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        app.state.artifact_store = store
        resp = client.post("/api/studio/missing/search-images", json={"query": "x"})
        assert resp.status_code == 404

    def test_search_requires_query(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post("/api/studio/art_1/search-images", json={})
        assert resp.status_code == 400

    def test_search_no_tool_registered(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post(
            "/api/studio/art_1/search-images", json={"query": "x"},
        )
        assert resp.status_code == 503

    def test_search_returns_candidates(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        image_search = MagicMock()
        image_search.execute = AsyncMock(return_value=_mock_tool_result(
            success=True,
            metadata={
                "images": [
                    {
                        "embed_url": "/api/artifacts/img_1/download",
                        "thumb_url": "/api/artifacts/img_1/thumb",
                        "source": "irena.org",
                        "title": "Solar cost decline",
                    },
                    {
                        "url": "http://example.com/photo.jpg",
                        "source": "wikipedia",
                        "title": "Solar farm",
                    },
                    {"embed_url": "", "title": "skipped-empty-url"},
                    {"not_a_dict": True},
                ],
            },
        ))
        app.state.tool_registry.resolve = MagicMock(return_value=image_search)
        resp = client.post(
            "/api/studio/art_1/search-images",
            json={"query": "solar cost chart", "count": 4, "prefer_charts": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["candidates"]) == 2
        assert data["candidates"][0]["source"] == "irena.org"
        assert data["candidates"][0]["embed_url"].startswith("/api/artifacts/")
        assert "candidate_id" in data["candidates"][0]
        assert data["query"] == "solar cost chart"
        image_search.execute.assert_awaited_once()
        kwargs = image_search.execute.await_args.kwargs
        assert kwargs["query"] == "solar cost chart"
        assert kwargs["count"] == 4
        assert kwargs["prefer_charts"] is True

    def test_search_count_clamped(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        image_search = MagicMock()
        image_search.execute = AsyncMock(return_value=_mock_tool_result(
            success=True, metadata={"images": []},
        ))
        app.state.tool_registry.resolve = MagicMock(return_value=image_search)
        client.post(
            "/api/studio/art_1/search-images",
            json={"query": "x", "count": 999},
        )
        assert image_search.execute.await_args.kwargs["count"] == 6

    def test_search_tool_failure_returns_empty_candidates(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        image_search = MagicMock()
        image_search.execute = AsyncMock(return_value=_mock_tool_result(
            success=False, error="searxng unreachable",
        ))
        app.state.tool_registry.resolve = MagicMock(return_value=image_search)
        resp = client.post(
            "/api/studio/art_1/search-images", json={"query": "x"},
        )
        assert resp.status_code == 200
        assert resp.json()["candidates"] == []


class TestGenerateImage:
    def test_generate_no_store(self, client):
        resp = client.post(
            "/api/studio/art_1/generate-image", json={"prompt": "a cat"},
        )
        assert resp.status_code == 503

    def test_generate_requires_prompt(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post(
            "/api/studio/art_1/generate-image", json={"style": "realism"},
        )
        assert resp.status_code == 400

    def test_generate_no_tool_registered(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        resp = client.post(
            "/api/studio/art_1/generate-image", json={"prompt": "x"},
        )
        assert resp.status_code == 503

    def test_generate_stages_image_and_returns_id(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        image_gen = MagicMock()
        image_gen.execute = AsyncMock(return_value=_mock_tool_result(
            success=True,
            metadata={
                "image_id": "img_abc123",
                "url": "/api/image/img_abc123",
                "prompt": "a solar panel at sunset",
            },
        ))
        app.state.tool_registry.resolve = MagicMock(return_value=image_gen)
        resp = client.post(
            "/api/studio/art_1/generate-image",
            json={"prompt": "a solar panel", "style": "realism", "aspect": "landscape"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["gen_id"] == "img_abc123"
        assert data["embed_url"] == "/api/image/img_abc123"
        assert "staged_until" in data
        assert "img_abc123" in app.state.studio_staging
        entry = app.state.studio_staging["img_abc123"]
        assert entry["artifact_id"] == "art_1"
        assert entry["prompt"] == "a solar panel"

    def test_generate_tool_failure_returns_502(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        image_gen = MagicMock()
        image_gen.execute = AsyncMock(return_value=_mock_tool_result(
            success=False, error="model unavailable",
        ))
        app.state.tool_registry.resolve = MagicMock(return_value=image_gen)
        resp = client.post(
            "/api/studio/art_1/generate-image", json={"prompt": "x"},
        )
        assert resp.status_code == 502

    def test_generate_missing_image_id_returns_502(self, app, client):
        """Tool succeeded but didn't return an image_id. Reject — no orphan stage."""
        app.state.artifact_store = _mock_artifact_store()
        image_gen = MagicMock()
        image_gen.execute = AsyncMock(return_value=_mock_tool_result(
            success=True, metadata={"url": "/api/image/orphan"},
        ))
        app.state.tool_registry.resolve = MagicMock(return_value=image_gen)
        resp = client.post(
            "/api/studio/art_1/generate-image", json={"prompt": "x"},
        )
        assert resp.status_code == 502
        assert not getattr(app.state, "studio_staging", {})


class TestStagingCommit:
    def test_commit_removes_from_registry(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        app.state.studio_staging = {
            "img_x": {
                "user_id": "usr_test", "artifact_id": "art_1",
                "prompt": "x", "style": "", "aspect": "square",
                "created_at": 1000.0,
            },
        }
        resp = client.post("/api/studio/art_1/staging/img_x/commit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["committed"] is True
        assert data["url"] == "/api/image/img_x"
        assert "img_x" not in app.state.studio_staging

    def test_commit_idempotent_when_already_swept(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        app.state.studio_staging = {}
        resp = client.post("/api/studio/art_1/staging/ghost/commit")
        assert resp.status_code == 200
        assert resp.json()["committed"] is True


class TestStagingDiscard:
    def test_discard_calls_delete_generation(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        app.state.studio_staging = {
            "img_d": {
                "user_id": "usr_test", "artifact_id": "art_1",
                "prompt": "x", "style": "", "aspect": "square",
                "created_at": 1000.0,
            },
        }
        image_store = MagicMock()
        image_store.delete_generation = AsyncMock(return_value=None)
        app.state.image_persistence = image_store

        resp = client.delete("/api/studio/art_1/staging/img_d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert "img_d" not in app.state.studio_staging
        image_store.delete_generation.assert_awaited_once_with(
            "img_d", user_id="usr_test",
        )

    def test_discard_cross_tenant_blocked(self, app, client):
        """Staging entry owned by a different user must 404."""
        app.state.artifact_store = _mock_artifact_store()
        app.state.studio_staging = {
            "img_other": {
                "user_id": "someone_else", "artifact_id": "art_1",
                "prompt": "x", "style": "", "aspect": "square",
                "created_at": 1000.0,
            },
        }
        image_store = MagicMock()
        image_store.delete_generation = AsyncMock()
        app.state.image_persistence = image_store

        resp = client.delete("/api/studio/art_1/staging/img_other")
        assert resp.status_code == 404
        assert "img_other" in app.state.studio_staging
        image_store.delete_generation.assert_not_called()

    def test_discard_idempotent_when_already_gone(self, app, client):
        app.state.artifact_store = _mock_artifact_store()
        app.state.studio_staging = {}
        image_store = MagicMock()
        image_store.delete_generation = AsyncMock(return_value=None)
        app.state.image_persistence = image_store

        resp = client.delete("/api/studio/art_1/staging/ghost")
        assert resp.status_code == 200


class TestStagingSweep:
    """Direct unit-tests of studio_staging_sweep — no HTTP round-trip."""

    def _run(self, app_obj, now=None):
        import asyncio
        from augmentum.proxy.studio_routes import studio_staging_sweep
        return asyncio.run(studio_staging_sweep(app_obj, now=now))

    def test_sweep_empty_registry_is_noop(self, app):
        app.state.studio_staging = {}
        assert self._run(app) == 0

    def test_sweep_keeps_fresh_entries(self, app):
        import time as _time
        app.state.studio_staging = {
            "fresh": {
                "user_id": "u1", "artifact_id": "a1",
                "prompt": "x", "style": "", "aspect": "square",
                "created_at": _time.time(),
            },
        }
        image_store = MagicMock()
        image_store.delete_generation = AsyncMock(return_value="/tmp/img.png")
        app.state.image_persistence = image_store

        swept = self._run(app)
        assert swept == 0
        assert "fresh" in app.state.studio_staging
        image_store.delete_generation.assert_not_called()

    def test_sweep_deletes_expired_entries(self, app):
        app.state.studio_staging = {
            "stale": {
                "user_id": "u1", "artifact_id": "a1",
                "prompt": "x", "style": "", "aspect": "square",
                "created_at": 0.0,
            },
            "fresh": {
                "user_id": "u1", "artifact_id": "a1",
                "prompt": "y", "style": "", "aspect": "square",
                "created_at": 10**12,
            },
        }
        image_store = MagicMock()
        image_store.delete_generation = AsyncMock(return_value=None)
        app.state.image_persistence = image_store

        swept = self._run(app)
        assert swept == 1
        assert "stale" not in app.state.studio_staging
        assert "fresh" in app.state.studio_staging
        image_store.delete_generation.assert_awaited_once_with(
            "stale", user_id="u1",
        )

    def test_sweep_skips_orphans_without_user_id(self, app):
        """An entry missing user_id can't be deleted safely. Remove from registry but don't call delete."""
        app.state.studio_staging = {
            "orphan": {
                "user_id": "", "artifact_id": "a1",
                "prompt": "x", "style": "", "aspect": "square",
                "created_at": 0.0,
            },
        }
        image_store = MagicMock()
        image_store.delete_generation = AsyncMock()
        app.state.image_persistence = image_store

        swept = self._run(app)
        assert "orphan" not in app.state.studio_staging
        assert swept == 0
        image_store.delete_generation.assert_not_called()


class TestDesignBlockNormalization:
    """Coverage for artifact_theme.normalize_design + apply_design + Studio _resolve_design.

    These are the Phase 2 substrate. The Design tool writes through the
    JSON shape these helpers parse, so the round-trip needs to survive
    bad inputs without breaking the existing per-format renderers.
    """

    def test_normalize_design_full_input(self):
        from augmentum.tools.artifact_theme import normalize_design
        d = normalize_design({
            "theme": "emerald",
            "font_family": "serif",
            "font_size_scale": 1.15,
            "line_height": "airy",
            "density": "spacious",
            "accent_override": "#FF6600",
        }, fallback_theme="slate")
        assert d == {
            "theme": "emerald",
            "font_family": "serif",
            "font_size_scale": 1.15,
            "line_height": "airy",
            "density": "spacious",
            "accent_override": "#FF6600",
        }

    def test_normalize_design_clamps_bad_inputs(self):
        from augmentum.tools.artifact_theme import normalize_design
        # Out-of-range scale, unknown enum, bad hex → all fall back.
        d = normalize_design({
            "font_family": "wingdings",
            "font_size_scale": 99.0,
            "line_height": "loose",
            "density": "supercompact",
            "accent_override": "NOTHEX",
        }, fallback_theme="slate")
        assert d["theme"] == "slate"
        assert d["font_family"] == "system"
        assert d["font_size_scale"] == 1.0
        assert d["line_height"] == "comfortable"
        assert d["density"] == "default"
        assert d["accent_override"] is None

    def test_normalize_design_3digit_hex_expands(self):
        from augmentum.tools.artifact_theme import normalize_design
        d = normalize_design({"accent_override": "#f60"}, fallback_theme="slate")
        assert d["accent_override"] == "#FF6600"

    def test_normalize_design_none_returns_defaults(self):
        from augmentum.tools.artifact_theme import normalize_design
        d = normalize_design(None, fallback_theme="emerald")
        assert d["theme"] == "emerald"
        assert d["font_family"] == "system"
        assert d["font_size_scale"] == 1.0

    def test_apply_design_default_is_identity(self):
        from augmentum.tools.artifact_theme import DEFAULT_DESIGN, apply_design, get_theme
        theme = get_theme("slate")
        assert apply_design(theme, DEFAULT_DESIGN) is theme
        assert apply_design(theme, None) is theme

    def test_apply_design_scales_font_sizes(self):
        from augmentum.tools.artifact_theme import apply_design, get_theme
        theme = get_theme("slate")
        scaled = apply_design(theme, {"font_size_scale": 1.3})
        assert scaled.body_size == round(theme.body_size * 1.3, 2)
        assert scaled.title_size == round(theme.title_size * 1.3, 2)
        assert scaled.slide_title_size == round(theme.slide_title_size * 1.3, 2)
        # Identity on non-size fields
        assert scaled.accent == theme.accent
        assert scaled.margin_left == theme.margin_left

    def test_apply_design_scales_line_height(self):
        from augmentum.tools.artifact_theme import apply_design, get_theme
        theme = get_theme("slate")
        assert apply_design(theme, {"line_height": "tight"}).line_height < theme.line_height
        assert apply_design(theme, {"line_height": "airy"}).line_height > theme.line_height

    def test_apply_design_scales_margins(self):
        from augmentum.tools.artifact_theme import apply_design, get_theme
        theme = get_theme("slate")
        assert apply_design(theme, {"density": "compact"}).margin_left < theme.margin_left
        assert apply_design(theme, {"density": "spacious"}).margin_left > theme.margin_left

    def test_apply_design_accent_override_recomputes_dark_light(self):
        from augmentum.tools.artifact_theme import apply_design, get_theme
        theme = get_theme("slate")
        out = apply_design(theme, {"accent_override": "#FF6600"})
        assert out.accent == "#FF6600"
        # accent_dark should be a darker variant; accent_light lighter.
        assert out.accent_dark != theme.accent_dark
        assert out.accent_light != theme.accent_light

    def test_resolve_design_prefers_explicit_block(self):
        from augmentum.proxy.studio_routes import _resolve_design
        source = {
            "type": "document",
            "design": {"theme": "emerald", "font_size_scale": 1.15},
            "theme": "slate",   # legacy — should be overridden by design
        }
        d = _resolve_design(source)
        assert d["theme"] == "emerald"
        assert d["font_size_scale"] == 1.15

    def test_resolve_design_synthesizes_from_legacy_theme(self):
        """Old artifacts with only source.theme should still get a design block."""
        from augmentum.proxy.studio_routes import _resolve_design
        source = {"type": "document", "theme": "modern"}
        d = _resolve_design(source)
        assert d["theme"] == "modern"
        assert d["font_family"] == "system"
        assert d["font_size_scale"] == 1.0

    def test_resolve_design_synthesizes_from_theme_dict(self):
        from augmentum.proxy.studio_routes import _resolve_design
        source = {"type": "presentation", "theme": {"preset": "rose"}}
        assert _resolve_design(source)["theme"] == "rose"

    def test_resolve_design_synthesizes_from_ebook_reading(self):
        """Old ebooks have source.reading instead of design. Migrate on read."""
        from augmentum.proxy.studio_routes import _resolve_design
        source = {
            "type": "ebook",
            "theme": "sepia",
            "reading": {"font": "serif", "size": "lg", "leading": "relaxed"},
        }
        d = _resolve_design(source)
        assert d["theme"] == "sepia"
        assert d["font_family"] == "serif"
        assert d["font_size_scale"] == 1.15
        assert d["line_height"] == "airy"

    def test_project_design_to_reading_round_trip(self):
        from augmentum.proxy.studio_routes import (
            _project_design_to_reading,
            _resolve_design,
        )
        # source.reading → design → reading should preserve semantics
        source = {
            "type": "ebook",
            "reading": {"font": "sans", "size": "sm", "leading": "compact"},
        }
        design = _resolve_design(source)
        reading_back = _project_design_to_reading(design)
        assert reading_back["font"] == "sans"
        assert reading_back["size"] == "sm"
        assert reading_back["leading"] == "compact"

    def test_resolve_design_missing_returns_defaults(self):
        from augmentum.proxy.studio_routes import _resolve_design
        d = _resolve_design({"type": "document"})
        assert d["theme"] == ""
        assert d["font_family"] == "system"
        assert d["font_size_scale"] == 1.0
        assert d["line_height"] == "comfortable"
