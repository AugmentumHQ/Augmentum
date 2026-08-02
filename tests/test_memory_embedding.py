"""Tests for augmentum/memory/embeddings.py — EmbeddingService serialization."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from augmentum.memory.embeddings import EmbeddingService


class TestBlobSerialization:
    """to_blob/from_blob round-trip tests."""

    def test_to_blob_correct_size(self):
        vec = [0.1] * 768
        blob = EmbeddingService.to_blob(vec)
        assert len(blob) == 768 * 4  # float32 = 4 bytes each

    def test_from_blob_correct_length(self):
        vec = [0.5] * 768
        blob = EmbeddingService.to_blob(vec)
        restored = EmbeddingService.from_blob(blob)
        assert len(restored) == 768

    def test_round_trip_preserves_values(self):
        vec = [float(i) / 768 for i in range(768)]
        blob = EmbeddingService.to_blob(vec)
        restored = EmbeddingService.from_blob(blob)
        for orig, rest in zip(vec, restored):
            assert abs(orig - rest) < 1e-6

    def test_to_blob_empty_vector(self):
        blob = EmbeddingService.to_blob([])
        assert blob == b""

    def test_from_blob_empty(self):
        restored = EmbeddingService.from_blob(b"")
        assert restored == []

    def test_to_blob_single_value(self):
        blob = EmbeddingService.to_blob([3.14])
        assert len(blob) == 4
        restored = EmbeddingService.from_blob(blob)
        assert abs(restored[0] - 3.14) < 1e-5

    def test_blob_is_little_endian_float32(self):
        vec = [1.0]
        blob = EmbeddingService.to_blob(vec)
        value = struct.unpack("<f", blob)[0]
        assert abs(value - 1.0) < 1e-7

    def test_negative_values_round_trip(self):
        vec = [-0.5, -1.0, -0.001]
        blob = EmbeddingService.to_blob(vec)
        restored = EmbeddingService.from_blob(blob)
        for orig, rest in zip(vec, restored):
            assert abs(orig - rest) < 1e-5


class TestEmbeddingPrefixes:
    """Verify document vs query prefix handling."""

    def test_embed_adds_document_prefix(self):
        """embed() should add 'search_document: ' prefix."""
        mock_model = MagicMock()
        mock_model.embed.return_value = [MagicMock(tolist=MagicMock(return_value=[0.1] * 768))]

        with patch.object(EmbeddingService, "get_model", return_value=mock_model):
            EmbeddingService.embed(["test text"])
            args = mock_model.embed.call_args[0][0]
            assert args[0].startswith("search_document: ")

    def test_embed_query_adds_query_prefix(self):
        """embed_query() should add 'search_query: ' prefix."""
        mock_model = MagicMock()
        result_array = MagicMock()
        result_array.tolist.return_value = [0.1] * 768
        mock_model.embed.return_value = iter([result_array])

        with patch.object(EmbeddingService, "get_model", return_value=mock_model):
            EmbeddingService.embed_query("test query")
            args = mock_model.embed.call_args[0][0]
            assert args[0].startswith("search_query: ")

    def test_dimension_constant(self):
        assert EmbeddingService.DIMENSION == 768

    def test_model_name_constant(self):
        assert "nomic" in EmbeddingService.MODEL_NAME.lower()

    def test_reset_clears_model(self):
        """reset() should set model back to unloaded sentinel."""
        EmbeddingService.reset()
        # After reset, _model should be the _UNLOADED sentinel
        from augmentum.memory.embeddings import _UNLOADED
        assert EmbeddingService._model is _UNLOADED


class TestProviderSelection:
    """Verify provider override path (env > setting > library default).

    Regression guard: every call to EmbeddingService.get_model() must
    honor the same three-tier resolution. The default (no env, setting
    False) MUST pin CPUExecutionProvider so a stock deployment never
    silently consumes ~500 MiB of VRAM for an INT8 model that ORT can't
    keep resident on-device anyway.
    """

    def _patch_textembedding(self, monkeypatch):
        """Replace fastembed.TextEmbedding with a recording mock so we
        can inspect the kwargs passed by get_model() without actually
        downloading or loading the ONNX session."""
        import fastembed

        ctor_calls: list[dict] = []

        def _fake_ctor(name, **kwargs):
            ctor_calls.append(kwargs)
            return MagicMock(name=f"TextEmbedding({name})")

        monkeypatch.setattr(fastembed, "TextEmbedding", _fake_ctor)
        return ctor_calls

    def test_default_setting_forces_cpu(self, monkeypatch):
        """embedding_use_gpu=False (default) → providers=[CPU]."""
        calls = self._patch_textembedding(monkeypatch)
        monkeypatch.delenv("ORT_PROVIDERS", raising=False)
        EmbeddingService.reset()
        from augmentum.config import settings
        monkeypatch.setattr(settings, "embedding_use_gpu", False)
        EmbeddingService.get_model()
        assert calls, "TextEmbedding constructor was never called"
        assert calls[0].get("providers") == ["CPUExecutionProvider"]

    def test_setting_gpu_omits_provider_override(self, monkeypatch):
        """embedding_use_gpu=True → no providers kwarg → fastembed default."""
        calls = self._patch_textembedding(monkeypatch)
        monkeypatch.delenv("ORT_PROVIDERS", raising=False)
        EmbeddingService.reset()
        from augmentum.config import settings
        monkeypatch.setattr(settings, "embedding_use_gpu", True)
        EmbeddingService.get_model()
        assert calls
        assert "providers" not in calls[0]

    def test_env_override_wins_over_setting(self, monkeypatch):
        """ORT_PROVIDERS env var wins regardless of setting value."""
        calls = self._patch_textembedding(monkeypatch)
        monkeypatch.setenv("ORT_PROVIDERS", "CUDAExecutionProvider")
        EmbeddingService.reset()
        from augmentum.config import settings
        monkeypatch.setattr(settings, "embedding_use_gpu", False)
        EmbeddingService.get_model()
        assert calls
        assert calls[0].get("providers") == ["CUDAExecutionProvider"]


class TestLoadFailureGracefulDegrade:
    """When the embedder model file is missing/corrupted, callers should
    get a typed ``EmbeddingUnavailable`` instead of a chat-killing raw
    ONNX traceback. Sticky failure (subsequent calls bail immediately)
    so a busy chat path doesn't spew tracebacks per-call.

    Reproduces the 2026-05-23 production incident: fastembed cache for
    nomic-embed-text-v1.5 was missing model_quantized.onnx, every
    chat path that touched memory recall / knowledge pack search
    crashed mid-stream until the cache was wiped + re-downloaded.
    """

    def setup_method(self):
        EmbeddingService.reset()

    def teardown_method(self):
        EmbeddingService.reset()

    def test_load_failure_raises_typed_unavailable(self):
        """A persistent load failure (both initial + post-wipe retry)
        raises ``EmbeddingUnavailable``, not the raw ONNX exception.
        Memory recall / knowledge pack callers already ``except
        Exception`` broadly so they degrade gracefully — but typed
        new exception lets future code distinguish ``embedder broken
        on this node`` from other failures.
        """
        from augmentum.memory.embeddings import EmbeddingUnavailable

        with patch("fastembed.TextEmbedding") as mock_te:
            mock_te.side_effect = RuntimeError(
                "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model failed"
            )
            with pytest.raises(EmbeddingUnavailable) as excinfo:
                EmbeddingService.get_model()
            # Should be wrapped from the underlying ONNX error
            assert "failed to load" in str(excinfo.value)

    def test_subsequent_calls_bail_immediately(self):
        """After load failure, subsequent get_model() calls raise the
        typed error without re-attempting the ONNX load. This prevents
        a busy chat handler from spewing the full download+wipe+retry
        cycle on every embedding call.
        """
        from augmentum.memory.embeddings import EmbeddingUnavailable

        with patch("fastembed.TextEmbedding") as mock_te:
            mock_te.side_effect = RuntimeError("NO_SUCHFILE")
            # First call: triggers wipe-and-retry path, both fail, sticky.
            with pytest.raises(EmbeddingUnavailable):
                EmbeddingService.get_model()
            initial_call_count = mock_te.call_count
            # Second call: should NOT re-attempt the constructor at all.
            with pytest.raises(EmbeddingUnavailable):
                EmbeddingService.get_model()
            assert mock_te.call_count == initial_call_count, (
                "subsequent get_model() must bail without re-calling fastembed"
            )

    def test_non_file_error_also_marks_unavailable(self):
        """Non-NO_SUCHFILE errors (e.g. ONNX runtime init failure on a
        machine without the right libs) should also mark unavailable
        rather than just propagating raw.
        """
        from augmentum.memory.embeddings import EmbeddingUnavailable

        with patch("fastembed.TextEmbedding") as mock_te:
            mock_te.side_effect = RuntimeError("ONNX init failed: missing libcuda")
            with pytest.raises(EmbeddingUnavailable):
                EmbeddingService.get_model()

    def test_unavailable_subclasses_runtime_error(self):
        """``EmbeddingUnavailable`` MUST subclass RuntimeError so existing
        ``except Exception`` paths in memory/knowledge code continue to
        catch it. Without this property the rollout breaks every caller.
        """
        from augmentum.memory.embeddings import EmbeddingUnavailable
        assert issubclass(EmbeddingUnavailable, RuntimeError)
        assert issubclass(EmbeddingUnavailable, Exception)
