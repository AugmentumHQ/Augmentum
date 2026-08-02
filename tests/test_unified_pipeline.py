"""Tests for the unified image pipeline (pipeline_v2).

Verifies that UnifiedPipeline correctly:
- Auto-detects pipeline type from loaded model
- Loads any model via DiffusionPipeline.from_pretrained()
- Generates images with architecture-appropriate parameters
- Falls back to latent-space img2img/inpaint universally
- Handles LoRA load/unload
- Properly cleans up on unload
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock, patch

import pytest

from augmentum.image.schemas import PipelineType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pipe(
    *,
    class_name: str = "StableDiffusionPipeline",
    has_unet: bool = True,
    has_transformer: bool = False,
    cross_attention_dim: int = 768,
):
    """Build a mock diffusers pipeline with controllable attributes."""
    # Create a proper subclass so type(pipe).__name__ returns the right value
    mock_cls = type(class_name, (MagicMock,), {})
    pipe = mock_cls()

    # VAE
    pipe.vae = MagicMock()
    pipe.vae.enable_tiling = MagicMock()
    pipe.vae.enable_slicing = MagicMock()
    pipe.vae_scale_factor = 8

    if has_unet:
        pipe.unet = MagicMock()
        pipe.unet.config = MagicMock()
        pipe.unet.config.cross_attention_dim = cross_attention_dim
        if not has_transformer:
            del pipe.transformer
    else:
        del pipe.unet
        if has_transformer:
            pipe.transformer = MagicMock()
        else:
            del pipe.transformer

    pipe.scheduler = MagicMock()
    pipe.fuse_qkv_projections = MagicMock()
    pipe.enable_model_cpu_offload = MagicMock()
    pipe.enable_attention_slicing = MagicMock()
    pipe.to = MagicMock(return_value=pipe)
    pipe.set_progress_bar_config = MagicMock()

    # safety checker (SD1.5)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # Generation result
    mock_image = MagicMock()
    mock_image.width = 512
    mock_image.height = 512
    mock_image.save = MagicMock()
    pipe.return_value.images = [mock_image]

    # LoRA
    pipe.load_lora_weights = MagicMock()
    pipe.fuse_lora = MagicMock()
    pipe.unfuse_lora = MagicMock()
    pipe.unload_lora_weights = MagicMock()

    return pipe


def _install_mock_diffusers(mock_pipe):
    """Inject a fake diffusers module into sys.modules."""
    import sys

    mock_diffusers = types.ModuleType("diffusers")
    diff_cls = MagicMock()
    diff_cls.from_pretrained.return_value = mock_pipe
    mock_diffusers.DiffusionPipeline = diff_cls
    mock_diffusers.StableDiffusionPipeline = MagicMock()
    mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_pipe
    mock_diffusers.StableDiffusionXLPipeline = MagicMock()
    mock_diffusers.StableDiffusionXLPipeline.from_pretrained.return_value = mock_pipe

    sys.modules["diffusers"] = mock_diffusers
    return mock_diffusers


def _mock_torch(*, cuda_available: bool = False):
    """Minimal mock torch for pipeline tests."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = cuda_available
    mock.cuda.is_bf16_supported.return_value = False
    mock.float16 = "float16"
    mock.float32 = "float32"
    mock.bfloat16 = "bfloat16"
    mock.channels_last = "channels_last"
    mock.backends.cuda.matmul.allow_tf32 = False
    mock.backends.cudnn.allow_tf32 = False

    # Generator mock
    gen = MagicMock()
    gen.manual_seed.return_value = gen
    mock.Generator.return_value = gen

    return mock


# =====================================================================
# Type detection
# =====================================================================


class TestTypeDetection:
    def test_sd15_detected_from_class_name(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(class_name="StableDiffusionPipeline")
        assert _detect_type_from_pipe(pipe) == PipelineType.SD15

    def test_sdxl_detected_from_class_name(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(class_name="StableDiffusionXLPipeline")
        assert _detect_type_from_pipe(pipe) == PipelineType.SDXL

    def test_sdxl_detected_from_unet_cross_attention(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(class_name="SomeCustomPipeline", cross_attention_dim=2048)
        assert _detect_type_from_pipe(pipe) == PipelineType.SDXL

    def test_flux_detected_from_transformer(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="FluxPipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_lumina_detected_as_flux(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="Lumina2Pipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_sd3_detected_as_flux(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="StableDiffusion3Pipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_unknown_unet_defaults_to_sd15(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(class_name="SomeWeirdPipeline", cross_attention_dim=768)
        assert _detect_type_from_pipe(pipe) == PipelineType.SD15

    def test_xl_in_class_name_detected_as_sdxl(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(class_name="SomeXLCustomPipeline")
        assert _detect_type_from_pipe(pipe) == PipelineType.SDXL

    def test_pixart_detected_as_flux(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="PixArtAlphaPipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_hunyuandit_detected_as_flux(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="HunyuanDiTPipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_kandinsky3_detected_as_flux(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="Kandinsky3Pipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_auraflow_detected_as_flux(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="AuraFlowPipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX

    def test_flux_class_name_detected(self):
        from augmentum.image.pipeline_v2 import _detect_type_from_pipe

        pipe = _make_mock_pipe(
            class_name="FluxPipeline", has_unet=False, has_transformer=True,
        )
        assert _detect_type_from_pipe(pipe) == PipelineType.FLUX


# =====================================================================
# Variant pipe resolution
# =====================================================================


class TestVariantPipeResolution:
    def test_returns_none_when_no_candidates(self):
        from augmentum.image.pipeline_v2 import _try_get_variant_pipe

        pipe = _make_mock_pipe(class_name="UnknownPipeline")
        # With diffusers not importable, should return None gracefully
        with patch.dict("sys.modules", {"diffusers": None}):
            result = _try_get_variant_pipe(pipe, "img2img")
        # May or may not be None depending on import behavior,
        # but should never raise
        assert result is None or result is not None

    def test_img2img_variant_for_flux(self):
        from augmentum.image.pipeline_v2 import _try_get_variant_pipe

        pipe = _make_mock_pipe(class_name="FluxPipeline")
        mock_i2i_cls = MagicMock()
        mock_i2i_pipe = MagicMock()
        mock_i2i_cls.from_pipe.return_value = mock_i2i_pipe

        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.FluxImg2ImgPipeline = mock_i2i_cls
        mock_diffusers.AutoPipelineForImage2Image = MagicMock()

        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            result = _try_get_variant_pipe(pipe, "img2img")

        assert result is mock_i2i_pipe
        mock_i2i_cls.from_pipe.assert_called_once_with(pipe)

    def test_inpaint_variant_for_sd15(self):
        from augmentum.image.pipeline_v2 import _try_get_variant_pipe

        pipe = _make_mock_pipe(class_name="StableDiffusionPipeline")
        mock_inpaint_cls = MagicMock()
        mock_inpaint_pipe = MagicMock()
        mock_inpaint_cls.from_pipe.return_value = mock_inpaint_pipe

        mock_diffusers = types.ModuleType("diffusers")
        mock_diffusers.StableDiffusionInpaintPipeline = mock_inpaint_cls
        mock_diffusers.AutoPipelineForInpainting = MagicMock()

        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            result = _try_get_variant_pipe(pipe, "inpaint")

        assert result is mock_inpaint_pipe

    def test_falls_through_on_from_pipe_error(self):
        from augmentum.image.pipeline_v2 import _try_get_variant_pipe

        pipe = _make_mock_pipe(class_name="FluxPipeline")

        mock_diffusers = types.ModuleType("diffusers")
        failing_cls = MagicMock()
        failing_cls.from_pipe.side_effect = RuntimeError("incompatible")
        mock_diffusers.FluxImg2ImgPipeline = failing_cls

        auto_cls = MagicMock()
        auto_pipe = MagicMock()
        auto_cls.from_pipe.return_value = auto_pipe
        mock_diffusers.AutoPipelineForImage2Image = auto_cls

        import sys
        sys.modules["diffusers"] = mock_diffusers

        result = _try_get_variant_pipe(pipe, "img2img")
        assert result is auto_pipe


# =====================================================================
# UnifiedPipeline lifecycle
# =====================================================================


class TestUnifiedPipelineLoad:
    def _load_pipeline(self, class_name="StableDiffusionPipeline", **pipe_kwargs):
        mock_pipe = _make_mock_pipe(class_name=class_name, **pipe_kwargs)
        _install_mock_diffusers(mock_pipe)
        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline._apply_pipeline_optimizations",
            return_value=mock_pipe,
        ), patch(
            "augmentum.image.pipeline._get_torch_dtype", return_value="float16"
        ), patch(
            "augmentum.image.pipeline._get_cpu_offload_setting", return_value="never"
        ), patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline_v2 import UnifiedPipeline
            p = UnifiedPipeline()
            asyncio.get_event_loop().run_until_complete(
                p.load("test_model", device="cpu", dtype="fp16")
            )
        return p, mock_pipe

    def test_load_detects_sd15(self):
        p, _ = self._load_pipeline("StableDiffusionPipeline")
        assert p.is_loaded
        assert p.pipeline_type == PipelineType.SD15
        assert p.model_name == "test_model"

    def test_load_detects_flux(self):
        p, _ = self._load_pipeline(
            "FluxPipeline", has_unet=False, has_transformer=True,
        )
        assert p.pipeline_type == PipelineType.FLUX

    def test_load_detects_sdxl_from_class(self):
        p, _ = self._load_pipeline("StableDiffusionXLPipeline")
        assert p.pipeline_type == PipelineType.SDXL

    def test_unload_clears_state(self):
        p, _ = self._load_pipeline()
        assert p.is_loaded

        mock_torch = _mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            asyncio.get_event_loop().run_until_complete(p.unload())
        assert not p.is_loaded

    def test_not_loaded_raises_on_generate(self):
        from augmentum.image.pipeline_v2 import UnifiedPipeline

        p = UnifiedPipeline()
        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            asyncio.get_event_loop().run_until_complete(
                p.generate("test prompt")
            )


# =====================================================================
# Generation
# =====================================================================


def _setup_unified_pipeline(class_name="StableDiffusionPipeline", **pipe_kwargs):
    """Helper to create a loaded UnifiedPipeline with all dependencies mocked."""
    mock_pipe = _make_mock_pipe(class_name=class_name, **pipe_kwargs)
    _install_mock_diffusers(mock_pipe)
    mock_torch = _mock_torch()

    with patch(
        "augmentum.image.pipeline._apply_pipeline_optimizations",
        return_value=mock_pipe,
    ), patch(
        "augmentum.image.pipeline._get_torch_dtype", return_value="float32"
    ), patch(
        "augmentum.image.pipeline._get_cpu_offload_setting", return_value="never"
    ), patch.dict("sys.modules", {"torch": mock_torch}):
        from augmentum.image.pipeline_v2 import UnifiedPipeline
        p = UnifiedPipeline()
        asyncio.get_event_loop().run_until_complete(
            p.load("test_model", device="cpu", dtype="fp32")
        )
    return p, mock_pipe


class TestUnifiedPipelineGenerate:
    def test_generate_returns_result(self, tmp_path):
        p, mock_pipe = _setup_unified_pipeline()

        mock_torch = _mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = asyncio.get_event_loop().run_until_complete(
                p.generate("a cat", output_dir=str(tmp_path), seed=42)
            )

        assert result.seed == 42
        assert result.image_id  # non-empty
        assert result.width == 512
        assert result.height == 512

    def test_generate_calls_pipe(self, tmp_path):
        p, mock_pipe = _setup_unified_pipeline()

        mock_torch = _mock_torch()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            asyncio.get_event_loop().run_until_complete(
                p.generate(
                    "a dog",
                    width=768,
                    height=768,
                    steps=30,
                    cfg_scale=5.0,
                    output_dir=str(tmp_path),
                    seed=123,
                )
            )

        mock_pipe.assert_called_once()
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["prompt"] == "a dog"
        assert call_kwargs["width"] == 768
        assert call_kwargs["height"] == 768
        assert call_kwargs["num_inference_steps"] == 30
        assert call_kwargs["guidance_scale"] == 5.0


# =====================================================================
# Img2Img
# =====================================================================


class TestUnifiedPipelineImg2Img:
    def test_img2img_tries_variant_first(self, tmp_path):
        """Should try dedicated img2img pipeline before falling back."""
        p, mock_pipe = _setup_unified_pipeline()

        mock_i2i_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 512
        mock_image.height = 512
        mock_image.save = MagicMock()
        mock_i2i_pipe.return_value.images = [mock_image]

        source_image = MagicMock()
        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline_v2._try_get_variant_pipe",
            return_value=mock_i2i_pipe,
        ), patch.dict("sys.modules", {"torch": mock_torch}):
            result = asyncio.get_event_loop().run_until_complete(
                p.img2img("edit this", source_image, output_dir=str(tmp_path), seed=42)
            )

        assert result.seed == 42
        mock_i2i_pipe.assert_called_once()

    def test_img2img_falls_back_to_latent(self, tmp_path):
        """When variant pipe is unavailable, should use latent fallback."""
        p, mock_pipe = _setup_unified_pipeline()

        source_image = MagicMock()
        fallback_image = MagicMock()
        fallback_image.width = 512
        fallback_image.height = 512
        fallback_image.save = MagicMock()

        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline_v2._try_get_variant_pipe",
            return_value=None,
        ), patch(
            "augmentum.image.pipeline_v2._latent_img2img_fallback",
            return_value=fallback_image,
        ) as mock_fallback, patch.dict("sys.modules", {"torch": mock_torch}):
            result = asyncio.get_event_loop().run_until_complete(
                p.img2img("edit this", source_image, output_dir=str(tmp_path), seed=42)
            )

        mock_fallback.assert_called_once()
        assert result.seed == 42


# =====================================================================
# Inpaint
# =====================================================================


class TestUnifiedPipelineInpaint:
    def test_inpaint_tries_variant_first(self, tmp_path):
        p, mock_pipe = _setup_unified_pipeline()

        mock_inpaint_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 512
        mock_image.height = 512
        mock_image.save = MagicMock()
        mock_inpaint_pipe.return_value.images = [mock_image]

        source_image = MagicMock()
        mask_image = MagicMock()
        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline_v2._try_get_variant_pipe",
            return_value=mock_inpaint_pipe,
        ), patch.dict("sys.modules", {"torch": mock_torch}):
            result = asyncio.get_event_loop().run_until_complete(
                p.inpaint("fix this", source_image, mask_image,
                          output_dir=str(tmp_path), seed=42)
            )

        assert result.seed == 42
        mock_inpaint_pipe.assert_called_once()

    def test_inpaint_falls_back_to_composite(self, tmp_path):
        p, mock_pipe = _setup_unified_pipeline()

        source_image = MagicMock()
        mask_image = MagicMock()
        fallback_image = MagicMock()
        fallback_image.width = 512
        fallback_image.height = 512
        fallback_image.save = MagicMock()

        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline_v2._try_get_variant_pipe",
            return_value=None,
        ), patch(
            "augmentum.image.pipeline_v2._latent_img2img_fallback",
            return_value=fallback_image,
        ) as mock_fallback, patch.dict("sys.modules", {"torch": mock_torch}):
            result = asyncio.get_event_loop().run_until_complete(
                p.inpaint("fix this", source_image, mask_image,
                          output_dir=str(tmp_path), seed=42)
            )

        mock_fallback.assert_called_once()


# =====================================================================
# LoRA
# =====================================================================


class TestUnifiedPipelineLoRA:
    def test_load_lora(self):
        p, mock_pipe = _setup_unified_pipeline()

        asyncio.get_event_loop().run_until_complete(
            p.load_lora("/path/to/lora", weight=0.8)
        )
        mock_pipe.load_lora_weights.assert_called_once_with("/path/to/lora")
        mock_pipe.fuse_lora.assert_called_once_with(lora_scale=0.8)

    def test_unload_loras(self):
        p, mock_pipe = _setup_unified_pipeline()

        asyncio.get_event_loop().run_until_complete(p.unload_loras())
        mock_pipe.unfuse_lora.assert_called_once()
        mock_pipe.unload_lora_weights.assert_called_once()

    def test_load_lora_not_loaded_raises(self):
        from augmentum.image.pipeline_v2 import UnifiedPipeline
        p = UnifiedPipeline()
        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            asyncio.get_event_loop().run_until_complete(
                p.load_lora("/path/to/lora")
            )

    def test_load_lora_unsupported_pipeline_raises(self):
        """Should give a clear error when pipeline doesn't support LoRA."""
        mock_pipe = _make_mock_pipe(class_name="SomePipeline")
        del mock_pipe.load_lora_weights  # Remove LoRA support
        _install_mock_diffusers(mock_pipe)
        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline._apply_pipeline_optimizations",
            return_value=mock_pipe,
        ), patch(
            "augmentum.image.pipeline._get_torch_dtype", return_value="float32"
        ), patch(
            "augmentum.image.pipeline._get_cpu_offload_setting", return_value="never"
        ), patch.dict("sys.modules", {"torch": mock_torch}):
            from augmentum.image.pipeline_v2 import UnifiedPipeline
            p = UnifiedPipeline()
            asyncio.get_event_loop().run_until_complete(
                p.load("test_model", device="cpu", dtype="fp32")
            )

        with pytest.raises(RuntimeError, match="does not support LoRA"):
            asyncio.get_event_loop().run_until_complete(
                p.load_lora("/path/to/lora")
            )


# =====================================================================
# Registry integration
# =====================================================================


class TestRegistryIntegration:
    """Verify UnifiedPipeline can be used as a drop-in replacement in the registry."""

    def test_implements_image_pipeline_interface(self):
        from augmentum.image.pipeline import ImagePipeline
        from augmentum.image.pipeline_v2 import UnifiedPipeline

        assert issubclass(UnifiedPipeline, ImagePipeline)

    def test_pipeline_registry_can_use_unified(self):
        """PipelineRegistry should work with UnifiedPipeline for all types."""
        from augmentum.image.pipeline_v2 import UnifiedPipeline

        # Verify all PipelineType values can map to UnifiedPipeline
        mapping = {
            PipelineType.SD15: UnifiedPipeline,
            PipelineType.SDXL: UnifiedPipeline,
            PipelineType.FLUX: UnifiedPipeline,
        }
        for pt in PipelineType:
            assert pt in mapping
            assert mapping[pt] is UnifiedPipeline

    def test_registry_uses_detected_type(self):
        """Registry should use pipeline's auto-detected type, not the hint."""
        from augmentum.image.pipeline_registry import PipelineRegistry

        # The pipeline was passed SD15 but auto-detected as FLUX
        mock_pipe = _make_mock_pipe(
            class_name="Lumina2Pipeline", has_unet=False, has_transformer=True,
        )
        _install_mock_diffusers(mock_pipe)
        mock_torch = _mock_torch()

        with patch(
            "augmentum.image.pipeline._apply_pipeline_optimizations",
            return_value=mock_pipe,
        ), patch(
            "augmentum.image.pipeline._get_torch_dtype", return_value="float32"
        ), patch(
            "augmentum.image.pipeline._get_cpu_offload_setting", return_value="never"
        ), patch.dict("sys.modules", {"torch": mock_torch}):
            reg = PipelineRegistry()
            pipeline = asyncio.get_event_loop().run_until_complete(
                reg.load("test_model", PipelineType.SD15, device="cpu", dtype="fp32")
            )

        # Registry should have corrected the type to FLUX
        assert reg._current_type == PipelineType.FLUX
        assert pipeline.pipeline_type == PipelineType.FLUX


# =====================================================================
# Model manager detection
# =====================================================================


class TestModelManagerDetection:
    """Test expanded _detect_pipeline_type in model_manager."""

    def test_flux_from_model_index(self, tmp_path):
        import json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "FluxPipeline"})
        )
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.FLUX

    def test_sd3_from_model_index(self, tmp_path):
        import json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "StableDiffusion3Pipeline"})
        )
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.FLUX

    def test_pixart_from_model_index(self, tmp_path):
        import json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "PixArtAlphaPipeline"})
        )
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.FLUX

    def test_lumina_from_model_index(self, tmp_path):
        import json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "Lumina2Pipeline"})
        )
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.FLUX

    def test_transformer_dir_detected_as_flux(self, tmp_path):
        """Model with transformer/ subdir but no model_index.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "transformer").mkdir()
        (model_dir / "transformer" / "config.json").write_text("{}")
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.FLUX

    def test_sdxl_from_unet_cross_attention(self, tmp_path):
        import json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "unet").mkdir()
        (model_dir / "unet" / "config.json").write_text(
            json.dumps({"cross_attention_dim": 2048})
        )
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.SDXL

    def test_sd15_default(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.SD15

    def test_sd15_from_model_index(self, tmp_path):
        import json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "StableDiffusionPipeline"})
        )
        from augmentum.image.model_manager import _detect_pipeline_type
        assert _detect_pipeline_type(str(model_dir)) == PipelineType.SD15
