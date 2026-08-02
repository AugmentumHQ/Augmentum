"""Tests for the per-modality fabric endpoints:

  - /api/fabric/knowledge/search
  - /api/fabric/render
  - /api/fabric/image/generate

Mirror the shape of test_fabric_inference_endpoint.py — pin the
invariants that the architecture review called out as load-bearing:

  * Verified-peer-only auth gate (no auth-bypass surface).
  * 503 when fabric is disabled.
  * Local-only resolution (no cross-peer fan-out, no recursion).
  * Body validation (clean 400s, not silent misdispatch).
  * Happy-path returns the expected shape.

Tests stub the upstream dependencies (pack_manager, executor, image
pipeline) so we're unit-testing the endpoint logic, not the heavy
subsystems themselves.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _make_request(*, fabric_peer: dict | None, body: dict | None):
    request = MagicMock()
    request.scope = {"fabric_peer": fabric_peer} if fabric_peer is not None else {}
    request.app = MagicMock()
    request.app.state = MagicMock()

    async def _json_body():
        if body is None:
            raise ValueError("no body")
        return body

    request.json = _json_body
    return request


# ── Knowledge search endpoint ─────────────────────────────────────


@pytest.mark.asyncio
async def test_knowledge_search_403_when_not_peer():
    """Single-source-of-truth: same auth gate as inference. Non-peer
    callers get 403."""
    from augmentum.proxy.fabric_routes import fabric_knowledge_search

    request = _make_request(fabric_peer=None, body={"q": "test", "pack_ids": ["p"]})
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_knowledge_search(request)
        assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_knowledge_search_503_when_disabled():
    from augmentum.proxy.fabric_routes import fabric_knowledge_search

    request = _make_request(fabric_peer={"sender_node_id": "p"}, body={"q": "t"})
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = False
        with pytest.raises(HTTPException) as excinfo:
            await fabric_knowledge_search(request)
        assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_knowledge_search_400_on_empty_query():
    from augmentum.proxy.fabric_routes import fabric_knowledge_search

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"q": "", "pack_ids": ["p1"]},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_knowledge_search(request)
        assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_knowledge_search_returns_empty_when_no_local_packs_match():
    """Critical recursion guard: requested pack_ids that don't exist
    locally are silently skipped — we MUST NOT fan out to other peers
    or we'd open A→B→C→A loops with no hop-count protection.
    """
    from augmentum.proxy.fabric_routes import fabric_knowledge_search

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"q": "search", "pack_ids": ["pack_not_local"]},
    )
    pack_mgr = MagicMock()
    pack_mgr.installed = MagicMock(return_value=[{"pack_id": "other_pack"}])
    pack_mgr.search = AsyncMock()
    request.app.state.pack_manager = pack_mgr

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        result = await fabric_knowledge_search(request)

    assert result == {"query": "search", "pack_ids": ["pack_not_local"], "results": []}
    # CRITICAL: pack_mgr.search must NOT be called (no local packs match).
    pack_mgr.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_knowledge_search_calls_pack_manager_for_local_packs():
    """Happy path: pack_ids that exist locally get searched, results
    returned in the expected shape."""
    from augmentum.proxy.fabric_routes import fabric_knowledge_search

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"q": "embedding", "pack_ids": ["wikipedia"], "limit": 5},
    )
    pack_mgr = MagicMock()
    pack_mgr.installed = MagicMock(return_value=[{"pack_id": "wikipedia"}])
    pack_mgr.search = AsyncMock(return_value=[
        {"text": "hit 1", "score": 0.9},
        {"text": "hit 2", "score": 0.8},
    ])
    request.app.state.pack_manager = pack_mgr

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        result = await fabric_knowledge_search(request)

    pack_mgr.search.assert_awaited_once()
    call_kwargs = pack_mgr.search.await_args.kwargs
    assert call_kwargs["pack_ids"] == ["wikipedia"]
    assert call_kwargs["limit"] == 5
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_knowledge_search_no_pack_manager_returns_empty():
    """Defensive: nodes without a pack_manager (e.g. compute-only peer
    config) return empty results cleanly, not 500."""
    from augmentum.proxy.fabric_routes import fabric_knowledge_search

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"q": "test", "pack_ids": ["p1"]},
    )
    request.app.state.pack_manager = None

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        result = await fabric_knowledge_search(request)

    assert result["results"] == []


# ── Render endpoint ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_render_403_when_not_peer():
    from augmentum.proxy.fabric_routes import fabric_render

    request = _make_request(fabric_peer=None, body={"kind": "html"})
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_render(request)
        assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_render_400_on_missing_kind():
    from augmentum.proxy.fabric_routes import fabric_render

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"target_device_id": "device"},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_render(request)
        assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_render_returns_renderresult_dict():
    """Happy path: dispatches to execute_local_render, returns the
    serialized RenderResult."""
    from augmentum.cast.render import RenderResult
    from augmentum.proxy.fabric_routes import fabric_render

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"kind": "html", "target_device_id": "device-1", "payload": {"url": "x"}},
    )
    request.app.state.html_renderer = MagicMock()
    request.app.state.render_output_store = MagicMock()
    request.app.state.fabric_coordinator = None
    request.scope["user"] = None

    fake_result = RenderResult(
        ok=True, location="peer", node_id="n", output_url="/x", code="", message="",
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings, \
         patch("augmentum.cast.executors.execute_local_render", new_callable=AsyncMock) as mock_run:
        mock_settings.fabric_enabled = True
        mock_run.return_value = fake_result
        result = await fabric_render(request)

    assert result["ok"] is True
    assert result["output_url"] == "/x"
    mock_run.assert_awaited_once()


# ── Image endpoint ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_image_generate_403_when_not_peer():
    from augmentum.proxy.fabric_routes import fabric_image_generate

    request = _make_request(
        fabric_peer=None,
        body={"model": "m", "prompt": "a cat"},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_image_generate(request)
        assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_image_generate_400_on_missing_model_or_prompt():
    from augmentum.proxy.fabric_routes import fabric_image_generate

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"model": "", "prompt": "a cat"},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_image_generate(request)
        assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_image_generate_503_when_no_pipeline_registry():
    from augmentum.proxy.fabric_routes import fabric_image_generate

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"model": "m", "prompt": "a cat"},
    )
    request.app.state.image_pipeline_registry = None
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_image_generate(request)
        assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_image_generate_503_when_no_persistence():
    """Cannot resolve model name → path without ImagePersistence."""
    from augmentum.proxy.fabric_routes import fabric_image_generate

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"model": "m", "prompt": "a cat"},
    )
    request.app.state.image_pipeline_registry = MagicMock()
    request.app.state.image_persistence = None
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_image_generate(request)
        assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_image_generate_404_when_model_unknown_on_peer():
    """If the requested model isn't in this peer's image_models table,
    generate_for_fabric raises ValueError — the endpoint maps that to
    404 so the sender's capability invalidation path can drop the
    stale advertisement."""
    from augmentum.proxy.fabric_routes import fabric_image_generate

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"model": "ghost-model", "prompt": "a cat"},
    )
    registry = MagicMock()
    registry.generate_for_fabric = AsyncMock(
        side_effect=ValueError("model 'ghost-model' not in image_models table"),
    )
    request.app.state.image_pipeline_registry = registry
    request.app.state.image_persistence = MagicMock()
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_image_generate(request)
        assert excinfo.value.status_code == 404
        assert "ghost-model" in excinfo.value.detail


@pytest.mark.asyncio
async def test_image_generate_returns_multipart_with_bytes_and_metadata():
    """Happy path: returns multipart/form-data with the JSON metadata
    part + raw image bytes part. Single round-trip — no separate fetch.
    """
    from augmentum.proxy.fabric_routes import fabric_image_generate

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"model": "m", "prompt": "a cat", "width": 512, "height": 512},
    )
    registry = MagicMock()
    fake_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes-here"
    fake_meta = {"width": 512, "height": 512, "seed": 42}
    registry.generate_for_fabric = AsyncMock(return_value=(fake_bytes, fake_meta))
    request.app.state.image_pipeline_registry = registry
    request.app.state.image_persistence = MagicMock()

    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        resp = await fabric_image_generate(request)

    assert resp.status_code == 200
    assert "multipart/form-data" in resp.media_type
    # The raw image bytes must appear somewhere in the multipart body
    assert fake_bytes in resp.body
    # And the metadata JSON should be present too
    assert b'"seed":42' in resp.body or b'"seed": 42' in resp.body


# ── TTS endpoint ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tts_403_when_not_peer():
    from augmentum.proxy.fabric_routes import fabric_tts

    request = _make_request(
        fabric_peer=None,
        body={"input": "hello world", "voice": "af_heart"},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_tts(request)
        assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_tts_400_on_missing_input():
    from augmentum.proxy.fabric_routes import fabric_tts

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"input": "", "voice": "af_heart"},
    )
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_tts(request)
        assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_tts_404_when_resolved_provider_is_fabric():
    """Critical recursion guard: if the receiver's voice resolution
    returns a fabric provider, refuse — we'd otherwise A→B→C loop.
    Same shape as knowledge_search's local-only gate."""
    from augmentum.proxy.fabric_routes import fabric_tts

    request = _make_request(
        fabric_peer={"sender_node_id": "p"},
        body={"input": "hi", "voice": "remote_voice"},
    )
    request.app.state.state_manager = MagicMock()
    # _get_conn returns a non-None conn; resolve_voice_provider returns a fabric provider.
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings, \
         patch("augmentum.proxy.audio_routes._get_conn", return_value=MagicMock()), \
         patch(
             "augmentum.proxy.audio_routes.resolve_voice_provider",
             new_callable=AsyncMock,
         ) as mock_resolve, \
         patch(
             "augmentum.proxy.audio_routes._FABRIC_PROVIDER_PREFIX", "fabric:",
         ):
        mock_settings.fabric_enabled = True
        mock_resolve.return_value = ({"id": "fabric:remote-node"}, "remote_voice")

        with pytest.raises(HTTPException) as excinfo:
            await fabric_tts(request)
        assert excinfo.value.status_code == 404
        assert "cross-peer recursion" in excinfo.value.detail


# ── STT endpoint ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stt_403_when_not_peer():
    from augmentum.proxy.fabric_routes import fabric_stt

    request = _make_request(fabric_peer=None, body=None)

    async def _empty_form():
        return MagicMock()

    request.form = _empty_form
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_stt(request)
        assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_stt_400_on_missing_file_field():
    from augmentum.proxy.fabric_routes import fabric_stt

    request = _make_request(fabric_peer={"sender_node_id": "p"}, body=None)

    # Empty form (no "file" field)
    async def _empty_form():
        form = MagicMock()
        form.get = MagicMock(return_value=None)
        return form

    request.form = _empty_form
    with patch("augmentum.proxy.fabric_routes.settings") as mock_settings:
        mock_settings.fabric_enabled = True
        with pytest.raises(HTTPException) as excinfo:
            await fabric_stt(request)
        assert excinfo.value.status_code == 400
