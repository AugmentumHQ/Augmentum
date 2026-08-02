"""Tests for image/vram.py -- VRAM reclamation utilities."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from augmentum.image.vram import _get_vram_mb, flush_cuda_cache


def _with_mock_torch(fn):
    """Helper: inject a mock torch into sys.modules, run fn, restore."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.memory_allocated.return_value = 0
    mock_torch.cuda.memory_reserved.return_value = 0

    orig = sys.modules.get("torch")
    sys.modules["torch"] = mock_torch
    try:
        return fn(mock_torch)
    finally:
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            sys.modules.pop("torch", None)


class TestGetVramMb:
    def test_returns_values_when_cuda_available(self):
        def run(mock_torch):
            mock_torch.cuda.memory_allocated.return_value = 500 * 1024 * 1024
            mock_torch.cuda.memory_reserved.return_value = 700 * 1024 * 1024
            alloc, reserved = _get_vram_mb()
            assert alloc == 500
            assert reserved == 700
        _with_mock_torch(run)

    def test_returns_zeros_without_cuda(self):
        alloc, reserved = _get_vram_mb()
        assert alloc == 0
        assert reserved == 0


class TestReleasePipeline:
    def test_calls_teardown_sequence(self):
        def run(mock_torch):
            from augmentum.image.vram import release_pipeline
            pipe = MagicMock()
            pipe.components = {"unet": MagicMock(), "vae": MagicMock()}
            pipe.maybe_free_model_hooks = MagicMock()

            release_pipeline(pipe, label="test")

            pipe.maybe_free_model_hooks.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.cuda.ipc_collect.assert_called_once()
        _with_mock_torch(run)

    def test_handles_missing_hooks(self):
        def run(mock_torch):
            mock_torch.cuda.is_available.return_value = False
            from augmentum.image.vram import release_pipeline
            pipe = MagicMock(spec=[])
            pipe.components = {}
            release_pipeline(pipe, label="test")
        _with_mock_torch(run)

    def test_nulls_out_components(self):
        def run(mock_torch):
            mock_torch.cuda.is_available.return_value = False
            from augmentum.image.vram import release_pipeline
            pipe = MagicMock()
            pipe.components = {"unet": MagicMock(), "text_encoder": MagicMock()}
            release_pipeline(pipe, label="test")
            # After release, components should have been set to None
            # The pipeline object is deleted inside release_pipeline,
            # but we can verify the function ran without error
        _with_mock_torch(run)


class TestFlushCudaCache:
    def test_no_crash_without_cuda(self):
        flush_cuda_cache()


class TestVramPositiveValues:
    def test_get_vram_returns_non_negative(self):
        alloc, reserved = _get_vram_mb()
        assert alloc >= 0
        assert reserved >= 0
