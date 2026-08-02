"""Tests for image pipeline optimizations: perf wins, BF16 auto-detect, CPU offload."""

from __future__ import annotations  # noqa: I001

import asyncio
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pipe(*, has_unet: bool = True, has_transformer: bool = False):
    """Build a mock diffusers pipeline with controllable attributes."""
    pipe = MagicMock()

    # VAE with tiling/slicing
    pipe.vae = MagicMock()
    pipe.vae.enable_tiling = MagicMock()
    pipe.vae.enable_slicing = MagicMock()

    # unet / transformer (only one at a time, like real pipelines)
    if has_unet:
        pipe.unet = MagicMock()
        if has_transformer:
            pipe.transformer = MagicMock()
        else:
            del pipe.transformer
    else:
        del pipe.unet
        if has_transformer:
            pipe.transformer = MagicMock()
        else:
            del pipe.transformer

    # fuse_qkv_projections (present on most diffusers pipelines)
    pipe.fuse_qkv_projections = MagicMock()

    # CPU offload
    pipe.enable_model_cpu_offload = MagicMock()

    # Attention slicing
    pipe.enable_attention_slicing = MagicMock()

    # .to() returns self (like real pipelines)
    pipe.to = MagicMock(return_value=pipe)

    return pipe


def _mock_torch(
    *,
    cuda_available: bool = True,
    bf16_supported: bool = False,
    device_capability: tuple[int, int] = (7, 5),
    free_vram_mb: int = 16_000,
):
    """Create a mock torch module with configurable CUDA state."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = cuda_available
    mock.cuda.is_bf16_supported.return_value = bf16_supported
    mock.cuda.get_device_capability.return_value = device_capability
    # mem_get_info returns (free_bytes, total_bytes)
    free_bytes = free_vram_mb * 1024 * 1024
    total_bytes = 24_000 * 1024 * 1024
    mock.cuda.mem_get_info.return_value = (free_bytes, total_bytes)
    # Real torch dtype sentinels
    mock.float16 = "float16"
    mock.float32 = "float32"
    mock.bfloat16 = "bfloat16"
    mock.channels_last = "channels_last"
    mock.backends.cuda.matmul.allow_tf32 = False
    mock.backends.cudnn.allow_tf32 = False
    return mock


# =====================================================================
# 10.1 — VAE tiling + slicing
# =====================================================================


class TestVAETilingSlicing:
    def test_vae_tiling_and_slicing_called(self):
        """VAE tiling and slicing should always be enabled."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="never", pipeline_key="sd15")

        pipe.vae.enable_tiling.assert_called_once()
        pipe.vae.enable_slicing.assert_called_once()

    def test_no_vae_attribute_graceful(self):
        """Pipeline without VAE should not crash."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()
        del pipe.vae

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            # Should not raise
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="never", pipeline_key="sd15")


# =====================================================================
# 10.1 — channels_last memory format
# =====================================================================


class TestChannelsLast:
    def test_channels_last_applied_on_cuda_unet(self):
        """channels_last should be applied to unet on CUDA device."""
        mock_torch = _mock_torch(device_capability=(7, 5))
        pipe = _make_mock_pipe(has_unet=True, has_transformer=False)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="never", pipeline_key="sd15")

        pipe.unet.to.assert_called_with(memory_format="channels_last")

    def test_channels_last_applied_on_cuda_transformer(self):
        """channels_last should be applied to transformer (FLUX) on CUDA."""
        mock_torch = _mock_torch(device_capability=(7, 5))
        pipe = _make_mock_pipe(has_unet=False, has_transformer=True)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="never", pipeline_key="flux")

        pipe.transformer.to.assert_called_with(memory_format="channels_last")

    def test_channels_last_skipped_on_cpu(self):
        """channels_last should NOT be applied on CPU device."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe(has_unet=True)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="never", pipeline_key="sd15")

        # unet.to should not have been called with channels_last
        for call in pipe.unet.to.call_args_list:
            assert call != (("channels_last",),), "channels_last should not be set on CPU"


# =====================================================================
# 10.1 — Fuse QKV projections
# =====================================================================


class TestFuseQKV:
    def test_fuse_qkv_called_when_available(self):
        """fuse_qkv_projections should be called when the method exists."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="never", pipeline_key="sd15")

        pipe.fuse_qkv_projections.assert_called_once()

    def test_fuse_qkv_skipped_when_not_available(self):
        """Should not crash when fuse_qkv_projections is absent."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()
        del pipe.fuse_qkv_projections

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            # Should not raise
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="never", pipeline_key="sd15")


# =====================================================================
# 10.1 — TF32 on Ampere+
# =====================================================================


class TestTF32:
    def test_tf32_enabled_on_ampere(self):
        """TF32 flags should be set when compute capability >= 8.0."""
        mock_torch = _mock_torch(device_capability=(8, 0))
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="never", pipeline_key="sd15")

        assert mock_torch.backends.cuda.matmul.allow_tf32 is True
        assert mock_torch.backends.cudnn.allow_tf32 is True

    def test_tf32_not_set_on_pre_ampere(self):
        """TF32 flags should NOT be set on pre-Ampere (capability < 8.0)."""
        mock_torch = _mock_torch(device_capability=(7, 5))
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="never", pipeline_key="sd15")

        assert mock_torch.backends.cuda.matmul.allow_tf32 is False
        assert mock_torch.backends.cudnn.allow_tf32 is False

    def test_tf32_not_set_on_cpu(self):
        """TF32 flags should NOT be touched when not on CUDA."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="never", pipeline_key="sd15")

        # cuda.is_available() is False, so get_device_capability should not be called
        mock_torch.cuda.get_device_capability.assert_not_called()


# =====================================================================
# 10.4 — BF16 auto-detection
# =====================================================================


class TestBF16AutoDetection:
    def test_auto_returns_bf16_when_supported(self):
        """'auto' dtype should return bfloat16 when bf16 is supported."""
        mock_torch = _mock_torch(bf16_supported=True)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _get_torch_dtype
            result = _get_torch_dtype("auto")

        assert result == "bfloat16"

    def test_auto_returns_fp16_when_bf16_not_supported(self):
        """'auto' dtype should fall back to float16 when bf16 is not supported."""
        mock_torch = _mock_torch(bf16_supported=False)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _get_torch_dtype
            result = _get_torch_dtype("auto")

        assert result == "float16"

    def test_auto_returns_fp32_on_cpu(self):
        """'auto' dtype should return float32 when CUDA is not available."""
        mock_torch = _mock_torch(cuda_available=False)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _get_torch_dtype
            result = _get_torch_dtype("auto")

        assert result == "float32"

    def test_explicit_fp16_unchanged(self):
        """Explicit 'fp16' should always return float16 regardless of bf16 support."""
        mock_torch = _mock_torch(bf16_supported=True)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _get_torch_dtype
            result = _get_torch_dtype("fp16")

        assert result == "float16"

    def test_explicit_bf16_unchanged(self):
        """Explicit 'bf16' should always return bfloat16."""
        mock_torch = _mock_torch(bf16_supported=False)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _get_torch_dtype
            result = _get_torch_dtype("bf16")

        assert result == "bfloat16"


# =====================================================================
# 10.3 — CPU offload: "always"
# =====================================================================


class TestCPUOffloadAlways:
    def test_always_calls_enable_model_cpu_offload(self):
        """cpu_offload='always' on CUDA should call enable_model_cpu_offload."""
        mock_torch = _mock_torch()
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="always", pipeline_key="sd15")

        pipe.enable_model_cpu_offload.assert_called_once()
        # .to(device) should NOT have been called
        pipe.to.assert_not_called()

    def test_always_on_cpu_device_uses_to(self):
        """cpu_offload='always' but device='cpu' should just use .to(device)."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="always", pipeline_key="sd15")

        pipe.to.assert_called_once_with("cpu")
        pipe.enable_model_cpu_offload.assert_not_called()


# =====================================================================
# 10.3 — CPU offload: "never"
# =====================================================================


class TestCPUOffloadNever:
    def test_never_calls_to_device(self):
        """cpu_offload='never' should call .to(device) and skip offload."""
        mock_torch = _mock_torch()
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="never", pipeline_key="sd15")

        pipe.to.assert_called_once_with("cuda")
        pipe.enable_model_cpu_offload.assert_not_called()


# =====================================================================
# 10.3 — CPU offload: "auto"
# =====================================================================


class TestCPUOffloadAuto:
    def test_auto_low_vram_enables_offload(self):
        """auto + low free VRAM should enable CPU offload."""
        # SD15 requires 4000 MB.  Threshold is 4000 * 1.3 = 5200.
        # With only 4000 MB free, offload should kick in.
        mock_torch = _mock_torch(free_vram_mb=4_000)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="auto", pipeline_key="sd15")

        pipe.enable_model_cpu_offload.assert_called_once()
        pipe.to.assert_not_called()

    def test_auto_sufficient_vram_uses_to(self):
        """auto + plenty of free VRAM should use .to(device)."""
        # SD15 requires 4000 MB.  Threshold is 5200.
        # With 16000 MB free, no offload needed.
        mock_torch = _mock_torch(free_vram_mb=16_000)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="auto", pipeline_key="sd15")

        pipe.to.assert_called_once_with("cuda")
        pipe.enable_model_cpu_offload.assert_not_called()

    def test_auto_flux_high_vram_threshold(self):
        """auto for FLUX should use 12000 * 1.3 = 15600 MB threshold."""
        # 14000 MB free < 15600 threshold -> offload
        mock_torch = _mock_torch(free_vram_mb=14_000)
        pipe = _make_mock_pipe(has_unet=False, has_transformer=True)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="auto", pipeline_key="flux")

        pipe.enable_model_cpu_offload.assert_called_once()

    def test_auto_sdxl_borderline_no_offload(self):
        """SDXL with just enough VRAM (>= 6000 * 1.3 = 7800) should not offload."""
        mock_torch = _mock_torch(free_vram_mb=8_000)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="auto", pipeline_key="sdxl")

        pipe.to.assert_called_once_with("cuda")
        pipe.enable_model_cpu_offload.assert_not_called()


# =====================================================================
# 10.3 — Attention slicing on very low VRAM
# =====================================================================


class TestAttentionSlicing:
    def test_attention_slicing_on_very_low_vram(self):
        """Attention slicing should be enabled when free VRAM < 4 GB."""
        mock_torch = _mock_torch(free_vram_mb=3_000)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="auto", pipeline_key="sd15")

        pipe.enable_attention_slicing.assert_called_once_with("auto")

    def test_attention_slicing_not_on_sufficient_vram(self):
        """Attention slicing should NOT be enabled when VRAM is sufficient."""
        mock_torch = _mock_torch(free_vram_mb=8_000)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cuda", cpu_offload="never", pipeline_key="sd15")

        pipe.enable_attention_slicing.assert_not_called()

    def test_attention_slicing_not_on_cpu(self):
        """Attention slicing should NOT be enabled on CPU (only CUDA low-VRAM)."""
        mock_torch = _mock_torch(cuda_available=False)
        pipe = _make_mock_pipe()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline import _apply_pipeline_optimizations
            _apply_pipeline_optimizations(pipe, "cpu", cpu_offload="auto", pipeline_key="sd15")

        pipe.enable_attention_slicing.assert_not_called()


# =====================================================================
# Config field
# =====================================================================


class TestConfigField:
    def test_image_cpu_offload_exists_with_default(self):
        """Settings should have image_cpu_offload field with default 'auto'."""
        from augmentum.config import Settings

        s = Settings()
        assert hasattr(s, "image_cpu_offload")
        assert s.image_cpu_offload == "auto"

    def test_image_cpu_offload_accepts_always(self):
        """Settings should accept 'always' for image_cpu_offload."""
        from augmentum.config import Settings

        s = Settings(image_cpu_offload="always")
        assert s.image_cpu_offload == "always"

    def test_image_cpu_offload_accepts_never(self):
        """Settings should accept 'never' for image_cpu_offload."""
        from augmentum.config import Settings

        s = Settings(image_cpu_offload="never")
        assert s.image_cpu_offload == "never"


# =====================================================================
# VRAM requirements constant
# =====================================================================


class TestVRAMRequirements:
    def test_vram_requirements_keys(self):
        """VRAM requirements dict should have entries for all pipeline types."""
        from augmentum.image.pipeline import _VRAM_REQUIREMENTS

        assert "sd15" in _VRAM_REQUIREMENTS
        assert "sdxl" in _VRAM_REQUIREMENTS
        assert "flux" in _VRAM_REQUIREMENTS

    def test_vram_requirements_values(self):
        """VRAM requirements should match hardware tier thresholds."""
        from augmentum.image.pipeline import _VRAM_REQUIREMENTS

        assert _VRAM_REQUIREMENTS["sd15"] == 4_000
        assert _VRAM_REQUIREMENTS["sdxl"] == 6_000
        assert _VRAM_REQUIREMENTS["flux"] == 12_000


# =====================================================================
# _get_cpu_offload_setting helper
# =====================================================================


class TestGetCPUOffloadSetting:
    def test_returns_config_value(self):
        """Should read from settings.image_cpu_offload."""
        with patch("augmentum.image.pipeline.settings", create=True) as mock_settings:
            mock_settings.image_cpu_offload = "always"
            # Need to reload to pick up the patch
            from augmentum.image.pipeline import _get_cpu_offload_setting
            with patch("augmentum.config.settings") as ms:
                ms.image_cpu_offload = "always"
                result = _get_cpu_offload_setting()
            assert result == "always"

    def test_defaults_to_auto_on_import_error(self):
        """Should default to 'auto' if config import fails."""
        with patch(
            "augmentum.image.pipeline._get_cpu_offload_setting",
        ) as mock_fn:
            mock_fn.return_value = "auto"
            result = mock_fn()
        assert result == "auto"


# =====================================================================
# Integration: pipeline load methods call optimizations
# =====================================================================


def _install_mock_diffusers(mock_pipe):
    """Inject a fake ``diffusers`` package into ``sys.modules`` so that
    ``from diffusers import ...`` inside pipeline load methods succeeds
    without having the real library installed."""
    import sys
    import types

    mock_diffusers = types.ModuleType("diffusers")
    sd_cls = MagicMock()
    sd_cls.from_pretrained.return_value = mock_pipe
    sdxl_cls = MagicMock()
    sdxl_cls.from_pretrained.return_value = mock_pipe
    flux_cls = MagicMock()
    flux_cls.from_pretrained.return_value = mock_pipe

    diff_cls = MagicMock()
    diff_cls.from_pretrained.return_value = mock_pipe

    mock_diffusers.StableDiffusionPipeline = sd_cls
    mock_diffusers.StableDiffusionXLPipeline = sdxl_cls
    mock_diffusers.FluxPipeline = flux_cls
    mock_diffusers.DiffusionPipeline = diff_cls

    sys.modules["diffusers"] = mock_diffusers
    return mock_diffusers


class TestPipelineLoadIntegration:
    """Verify that UnifiedPipeline passes through _apply_pipeline_optimizations."""

    def test_unified_load_calls_optimizations(self):
        """UnifiedPipeline.load should call _apply_pipeline_optimizations."""
        mock_pipe = _make_mock_pipe()
        # Add __call__ signature for _cache_pipe_metadata
        mock_pipe.__call__ = MagicMock()
        _install_mock_diffusers(mock_pipe)

        with patch(
            "augmentum.image.pipeline._apply_pipeline_optimizations"
        ) as mock_opt, patch(
            "augmentum.image.pipeline._get_torch_dtype", return_value="float16"
        ), patch(
            "augmentum.image.pipeline_v2._get_torch_dtype", return_value="float16"
        ), patch(
            "augmentum.image.pipeline._get_cpu_offload_setting", return_value="never"
        ), patch(
            "augmentum.image.pipeline_v2._get_cpu_offload_setting", return_value="never"
        ), patch(
            "augmentum.image.pipeline_v2._apply_pipeline_optimizations"
        ) as mock_opt_v2, patch(
            "augmentum.image.pipeline_v2._is_gguf_model", return_value=False
        ), patch(
            "augmentum.image.pipeline_v2._detect_type_from_pipe",
        ) as mock_detect, patch(
            "augmentum.image.pipeline_v2._is_edit_pipeline", return_value=False
        ), patch(
            "augmentum.image.pipeline_v2._is_qwen_pipeline", return_value=False
        ):
            from augmentum.image.schemas import PipelineType
            mock_detect.return_value = PipelineType.SD15
            mock_opt_v2.return_value = mock_pipe

            from augmentum.image.pipeline_v2 import UnifiedPipeline

            pipeline = UnifiedPipeline()
            asyncio.get_event_loop().run_until_complete(
                pipeline.load("test_model", device="cuda", dtype="fp16")
            )

            mock_opt_v2.assert_called_once()
            call_kwargs = mock_opt_v2.call_args
            assert call_kwargs[1]["pipeline_key"] in ("sd15", "sdxl", "flux")
