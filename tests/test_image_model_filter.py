"""Tests for image ModelManager download filtering.

Ensures pull_from_huggingface only downloads inference-required files
(configs, tokenizers, single variant of safetensors) and skips training
artifacts, redundant weight formats, and extra variants.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

from augmentum.image.model_manager import ModelManager


def _make_sibling(name: str, size: int = 1000) -> MagicMock:
    s = MagicMock()
    s.rfilename = name
    s.size = size
    return s


class TestSelectVariant:
    """Tests for ModelManager._select_variant()."""

    def test_prefers_fp16(self):
        siblings = [
            _make_sibling("model.fp16.safetensors"),
            _make_sibling("model.bf16.safetensors"),
            _make_sibling("model.safetensors"),
        ]
        assert ModelManager._select_variant(siblings) == "fp16"

    def test_falls_back_to_bf16(self):
        siblings = [
            _make_sibling("model.bf16.safetensors"),
            _make_sibling("model.safetensors"),
        ]
        assert ModelManager._select_variant(siblings) == "bf16"

    def test_falls_back_to_fp8(self):
        siblings = [
            _make_sibling("model.fp8.safetensors"),
            _make_sibling("model.safetensors"),
        ]
        assert ModelManager._select_variant(siblings) == "fp8"

    def test_no_variant_returns_none(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("config.json"),
        ]
        assert ModelManager._select_variant(siblings) is None

    def test_ignores_non_safetensors(self):
        siblings = [
            _make_sibling("model.fp16.bin"),
            _make_sibling("model.safetensors"),
        ]
        assert ModelManager._select_variant(siblings) is None

    def test_multiple_fp16_shards(self):
        siblings = [
            _make_sibling("model-00001-of-00003.fp16.safetensors"),
            _make_sibling("model-00002-of-00003.fp16.safetensors"),
            _make_sibling("model-00003-of-00003.fp16.safetensors"),
            _make_sibling("model-00001-of-00003.safetensors"),
            _make_sibling("model-00002-of-00003.safetensors"),
            _make_sibling("model-00003-of-00003.safetensors"),
        ]
        assert ModelManager._select_variant(siblings) == "fp16"


class TestFilterInferenceFiles:
    """Tests for ModelManager._filter_inference_files()."""

    def setup_method(self):
        with tempfile.TemporaryDirectory() as d:
            self.mgr = ModelManager(d)

    def test_skips_ckpt_files(self):
        siblings = [
            _make_sibling("model.safetensors", 5_000_000_000),
            _make_sibling("model.ckpt", 5_000_000_000),
            _make_sibling("config.json", 500),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "model.ckpt" not in names
        assert "model.safetensors" in names
        assert "config.json" in names

    def test_skips_bin_files(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("pytorch_model.bin"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "pytorch_model.bin" not in names
        assert "model.safetensors" in names

    def test_skips_optimizer_states(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("optimizer.pt"),
            _make_sibling("optimizer_state.bin"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert not any("optimizer" in n for n in names)

    def test_skips_training_args(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("training_args.bin"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "training_args.bin" not in names

    def test_skips_pth_files(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("consolidated.00-of-01.pth", 10_000_000_000),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "consolidated.00-of-01.pth" not in names

    def test_skips_images(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("Example/Demo_v2.png", 6_000_000),
            _make_sibling("preview.jpg"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert not any(n.endswith((".png", ".jpg")) for n in names)

    def test_skips_python_scripts(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("Script_Lora_Convert/Convert_lora.py"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert not any(n.endswith(".py") for n in names)

    def test_skips_readme(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("README.md"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "README.md" not in names

    def test_skips_gitattributes(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling(".gitattributes"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert ".gitattributes" not in names

    def test_keeps_all_json_configs(self):
        siblings = [
            _make_sibling("model_index.json"),
            _make_sibling("unet/config.json"),
            _make_sibling("vae/config.json"),
            _make_sibling("text_encoder/config.json"),
            _make_sibling("scheduler/scheduler_config.json"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        assert len(filtered) == len(siblings)

    def test_keeps_tokenizer_files(self):
        siblings = [
            _make_sibling("tokenizer/tokenizer.json"),
            _make_sibling("tokenizer/vocab.txt"),
            _make_sibling("tokenizer/merges.txt"),
            _make_sibling("tokenizer/spiece.model"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        assert len(filtered) == len(siblings)

    def test_variant_filtering_keeps_fp16_only(self):
        """When fp16 variant exists, skip full-precision safetensors."""
        siblings = [
            _make_sibling("unet/model.fp16.safetensors", 2_000_000_000),
            _make_sibling("unet/model.safetensors", 4_000_000_000),
            _make_sibling("vae/model.fp16.safetensors", 300_000_000),
            _make_sibling("vae/model.safetensors", 600_000_000),
            _make_sibling("model_index.json", 500),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "unet/model.fp16.safetensors" in names
        assert "unet/model.safetensors" not in names
        assert "vae/model.fp16.safetensors" in names
        assert "vae/model.safetensors" not in names
        assert "model_index.json" in names

    def test_variant_filtering_skips_wrong_variant(self):
        """When fp16 is chosen, bf16 variant is skipped."""
        siblings = [
            _make_sibling("model.fp16.safetensors"),
            _make_sibling("model.bf16.safetensors"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "model.fp16.safetensors" in names
        assert "model.bf16.safetensors" not in names

    def test_no_variant_keeps_all_safetensors(self):
        """When no variant tags exist, keep all safetensors."""
        siblings = [
            _make_sibling("unet/diffusion_pytorch_model-00001-of-00003.safetensors"),
            _make_sibling("unet/diffusion_pytorch_model-00002-of-00003.safetensors"),
            _make_sibling("unet/diffusion_pytorch_model-00003-of-00003.safetensors"),
            _make_sibling("model_index.json"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        assert len(filtered) == 4

    def test_keeps_untagged_safetensors_when_no_variant_in_subdir(self):
        """Safetensors without variant tag kept if no variant-specific file in same subdir."""
        siblings = [
            _make_sibling("unet/model.fp16.safetensors"),  # unet has fp16
            _make_sibling("unet/model.safetensors"),        # unet full-precision — skip
            _make_sibling("text_encoder/model.safetensors"),  # no variant in text_encoder — keep
            _make_sibling("model_index.json"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "unet/model.fp16.safetensors" in names
        assert "unet/model.safetensors" not in names
        assert "text_encoder/model.safetensors" in names

    def test_realistic_lumina_repo(self):
        """Simulate a repo like NetaYume-Lumina with sharded weights + fp16."""
        siblings = [
            # Full-precision shards (~10GB each = 70GB total)
            _make_sibling("diffusion_pytorch_model-00001-of-00007.safetensors", 10_000_000_000),
            _make_sibling("diffusion_pytorch_model-00002-of-00007.safetensors", 10_000_000_000),
            _make_sibling("diffusion_pytorch_model-00003-of-00007.safetensors", 10_000_000_000),
            _make_sibling("diffusion_pytorch_model-00004-of-00007.safetensors", 10_000_000_000),
            _make_sibling("diffusion_pytorch_model-00005-of-00007.safetensors", 10_000_000_000),
            _make_sibling("diffusion_pytorch_model-00006-of-00007.safetensors", 10_000_000_000),
            _make_sibling("diffusion_pytorch_model-00007-of-00007.safetensors", 10_000_000_000),
            # fp16 shards (~5GB each = 35GB total)
            _make_sibling("diffusion_pytorch_model-00001-of-00007.fp16.safetensors", 5_000_000_000),
            _make_sibling("diffusion_pytorch_model-00002-of-00007.fp16.safetensors", 5_000_000_000),
            _make_sibling("diffusion_pytorch_model-00003-of-00007.fp16.safetensors", 5_000_000_000),
            _make_sibling("diffusion_pytorch_model-00004-of-00007.fp16.safetensors", 5_000_000_000),
            _make_sibling("diffusion_pytorch_model-00005-of-00007.fp16.safetensors", 5_000_000_000),
            _make_sibling("diffusion_pytorch_model-00006-of-00007.fp16.safetensors", 5_000_000_000),
            _make_sibling("diffusion_pytorch_model-00007-of-00007.fp16.safetensors", 5_000_000_000),
            # Configs
            _make_sibling("config.json", 1000),
            _make_sibling("model_index.json", 500),
            # Training artifacts
            _make_sibling("optimizer.bin", 20_000_000_000),
            _make_sibling("training_args.bin", 5000),
            _make_sibling("README.md", 3000),
            _make_sibling(".gitattributes", 200),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        total_size = sum(s.size for s in filtered)

        # Should keep: 7 fp16 shards + 2 configs = 9 files
        assert len(filtered) == 9
        # Should be ~35GB not ~105GB+
        assert total_size == 35_000_001_500
        # No full-precision shards
        assert not any("safetensors" in n and "fp16" not in n for n in names
                       if n.endswith(".safetensors"))
        # No training artifacts
        assert "optimizer.bin" not in names
        assert "training_args.bin" not in names
        assert "README.md" not in names

    def test_skips_logs_subdirectory(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("logs/events.out.tfevents.12345"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert not any("logs/" in n for n in names)

    def test_skips_pt_files(self):
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("model.pt"),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "model.pt" not in names

    def test_realistic_multi_version_repo(self):
        """Simulate a repo like NetaYume-Lumina with multiple versions."""
        siblings = [
            # 5 all-in-one bundles (~10GB each)
            _make_sibling("NetaYume_Lumina_v2_all_in_one.safetensors", 10_000_000_000),
            _make_sibling("NetaYume_v2_plus_all_in_one.safetensors", 10_000_000_000),
            _make_sibling("NetaYume_v3_all_in_one.safetensors", 10_000_000_000),
            _make_sibling("NetaYume_v4_all_in_one.safetensors", 10_000_000_000),
            _make_sibling("NetaYumev35_pretrained_all_in_one.safetensors", 10_000_000_000),
            # Separate unets (~5GB each)
            _make_sibling("Unet/v2/NetaYume_Lumina_v2_unet.safetensors", 5_000_000_000),
            _make_sibling("Unet/v3/NetaYumev3_unet.safetensors", 5_000_000_000),
            _make_sibling("Unet/v4/NetaYumev4_unet.safetensors", 5_000_000_000),
            # Shared components
            _make_sibling("Text_Encoder/gemma_2_2b_fp16.safetensors", 5_000_000_000),
            _make_sibling("Vae/vae.safetensors", 320_000_000),
            # Training artifacts
            _make_sibling("Trained_weights_and_config/v2/consolidated.00-of-01.pth", 10_000_000_000),
            _make_sibling("Trained_weights_and_config/v2/model_args.pth", 5000),
            _make_sibling("Trained_weights_and_config/v2_plus/consolidated.00-of-01.pth", 10_000_000_000),
            # Junk
            _make_sibling("README.md", 3000),
            _make_sibling("Example/Demo_v2.png", 6_000_000),
            _make_sibling(".gitattributes", 200),
            _make_sibling("Script_Lora_Convert/Convert_lora.py", 5000),
            _make_sibling("Lumina_image_v2_tensorart_workflow.json", 1000),
        ]

        # Default filter: skips .pth, .md, .png, .py, .gitattributes
        # but still downloads all safetensors (no variant tags)
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert not any(n.endswith((".pth", ".md", ".png", ".py")) for n in names)
        assert ".gitattributes" not in names

        # With allow_patterns: user picks only v4 + shared components
        filtered_v4 = self.mgr._filter_inference_files(
            siblings,
            allow_patterns=[
                "NetaYume_v4_all_in_one.safetensors",
                "Text_Encoder/*",
                "Vae/*",
                "*.json",
            ],
        )
        names_v4 = [s.rfilename for s in filtered_v4]
        total_v4 = sum(s.size for s in filtered_v4)

        assert "NetaYume_v4_all_in_one.safetensors" in names_v4
        assert "Text_Encoder/gemma_2_2b_fp16.safetensors" in names_v4
        assert "Vae/vae.safetensors" in names_v4
        assert "Lumina_image_v2_tensorart_workflow.json" in names_v4
        # Should NOT include other versions
        assert "NetaYume_v3_all_in_one.safetensors" not in names_v4
        assert "NetaYume_Lumina_v2_all_in_one.safetensors" not in names_v4
        # ~15.3GB instead of 103GB
        assert total_v4 < 16_000_000_000

    def test_realistic_sdxl_repo(self):
        """Simulate the real stabilityai/stable-diffusion-xl-base-1.0 repo (71.6GB).
        Should filter down to ~6.6GB (fp16 component files only)."""
        siblings = [
            # Root-level single-file checkpoints (redundant with components)
            _make_sibling("sd_xl_base_1.0.safetensors", 6_938_246_474),
            _make_sibling("sd_xl_base_1.0_0.9vae.safetensors", 6_938_246_474),
            _make_sibling("sd_xl_offset_example-lora_1.0.safetensors", 49_590_178),
            # model_index.json
            _make_sibling("model_index.json", 542),
            # scheduler
            _make_sibling("scheduler/scheduler_config.json", 292),
            # text_encoder (4 formats)
            _make_sibling("text_encoder/config.json", 617),
            _make_sibling("text_encoder/flax_model.msgpack", 492_280_832),
            _make_sibling("text_encoder/model.fp16.safetensors", 246_144_152),
            _make_sibling("text_encoder/model.onnx", 492_563_974),
            _make_sibling("text_encoder/model.safetensors", 492_398_696),
            _make_sibling("text_encoder/openvino_model.bin", 492_282_630),
            _make_sibling("text_encoder/openvino_model.xml", 1_069_862),
            # text_encoder_2 (4 formats)
            _make_sibling("text_encoder_2/config.json", 782),
            _make_sibling("text_encoder_2/flax_model.msgpack", 2_778_316_800),
            _make_sibling("text_encoder_2/model.fp16.safetensors", 1_389_382_128),
            _make_sibling("text_encoder_2/model.onnx", 1_048_898),
            _make_sibling("text_encoder_2/model.onnx_data", 2_778_132_228),
            _make_sibling("text_encoder_2/model.safetensors", 2_778_505_080),
            _make_sibling("text_encoder_2/openvino_model.bin", 2_778_316_506),
            _make_sibling("text_encoder_2/openvino_model.xml", 2_832_466),
            # tokenizer
            _make_sibling("tokenizer/merges.txt", 524_619),
            _make_sibling("tokenizer/special_tokens_map.json", 588),
            _make_sibling("tokenizer/tokenizer_config.json", 680),
            _make_sibling("tokenizer/vocab.json", 1_064_915),
            # tokenizer_2
            _make_sibling("tokenizer_2/merges.txt", 524_619),
            _make_sibling("tokenizer_2/special_tokens_map.json", 588),
            _make_sibling("tokenizer_2/tokenizer_config.json", 680),
            _make_sibling("tokenizer_2/vocab.json", 1_064_915),
            # unet (4 formats)
            _make_sibling("unet/config.json", 1_519),
            _make_sibling("unet/diffusion_flax_model.msgpack", 10_271_308_800),
            _make_sibling("unet/diffusion_pytorch_model.fp16.safetensors", 5_135_149_760),
            _make_sibling("unet/diffusion_pytorch_model.safetensors", 10_270_295_892),
            _make_sibling("unet/model.onnx", 7_340_254),
            _make_sibling("unet/model.onnx_data", 10_270_301_752),
            _make_sibling("unet/openvino_model.bin", 10_270_293_676),
            _make_sibling("unet/openvino_model.xml", 22_530_932),
            # vae (2 formats)
            _make_sibling("vae/config.json", 820),
            _make_sibling("vae/diffusion_flax_model.msgpack", 334_643_200),
            _make_sibling("vae/diffusion_pytorch_model.fp16.safetensors", 167_335_342),
            _make_sibling("vae/diffusion_pytorch_model.safetensors", 334_643_468),
            # vae_1_0 (alternate VAE, redundant)
            _make_sibling("vae_1_0/config.json", 820),
            _make_sibling("vae_1_0/diffusion_pytorch_model.fp16.safetensors", 167_335_342),
            _make_sibling("vae_1_0/diffusion_pytorch_model.safetensors", 334_643_468),
            # vae_decoder (ONNX only)
            _make_sibling("vae_decoder/config.json", 820),
            _make_sibling("vae_decoder/model.onnx", 198_056_542),
            _make_sibling("vae_decoder/openvino_model.bin", 198_054_618),
            _make_sibling("vae_decoder/openvino_model.xml", 950_234),
            # vae_encoder (ONNX only)
            _make_sibling("vae_encoder/config.json", 820),
            _make_sibling("vae_encoder/model.onnx", 136_745_062),
            _make_sibling("vae_encoder/openvino_model.bin", 136_743_466),
            _make_sibling("vae_encoder/openvino_model.xml", 892_774),
            # Junk
            _make_sibling(".gitattributes", 1_519),
            _make_sibling("01.png", 4_604_693),
            _make_sibling("LICENSE.md", 16_154),
            _make_sibling("README.md", 6_850),
            _make_sibling("comparison.png", 88_414),
            _make_sibling("pipeline.png", 67_648),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        total = sum(s.size for s in filtered)
        total_gb = total / 1024 / 1024 / 1024

        # Should include: fp16 safetensors for each component + configs + tokenizers
        assert "model_index.json" in names
        assert "scheduler/scheduler_config.json" in names
        assert "text_encoder/model.fp16.safetensors" in names
        assert "text_encoder/config.json" in names
        assert "text_encoder_2/model.fp16.safetensors" in names
        assert "unet/diffusion_pytorch_model.fp16.safetensors" in names
        assert "unet/config.json" in names
        assert "vae/diffusion_pytorch_model.fp16.safetensors" in names
        assert "vae/config.json" in names
        assert "tokenizer/vocab.json" in names
        assert "tokenizer_2/vocab.json" in names

        # Should NOT include redundant formats
        assert "text_encoder/model.safetensors" not in names  # fp32
        assert "text_encoder/flax_model.msgpack" not in names  # Flax
        assert "text_encoder/model.onnx" not in names  # ONNX
        assert "text_encoder/openvino_model.bin" not in names  # OpenVINO
        assert "unet/diffusion_pytorch_model.safetensors" not in names  # fp32
        assert "unet/model.onnx" not in names
        assert "vae/diffusion_pytorch_model.safetensors" not in names  # fp32

        # Should NOT include root-level single-file checkpoints
        assert "sd_xl_base_1.0.safetensors" not in names
        assert "sd_xl_base_1.0_0.9vae.safetensors" not in names
        assert "sd_xl_offset_example-lora_1.0.safetensors" not in names

        # Should NOT include ONNX-only subdirs
        assert not any("vae_decoder/" in n for n in names)
        assert not any("vae_encoder/" in n for n in names)

        # Should NOT include junk
        assert "README.md" not in names
        assert "01.png" not in names

        # Total should be ~6.6 GB, not 71.6 GB
        assert total_gb < 7.0, f"Expected <7GB, got {total_gb:.1f}GB"
        assert total_gb > 6.0, f"Expected >6GB, got {total_gb:.1f}GB"

    def test_skips_onnx_only_subdirs(self):
        siblings = [
            _make_sibling("vae/config.json"),
            _make_sibling("vae/model.safetensors", 300_000_000),
            _make_sibling("vae_decoder/config.json"),
            _make_sibling("vae_decoder/model.onnx", 200_000_000),
            _make_sibling("vae_encoder/config.json"),
            _make_sibling("vae_encoder/model.onnx", 130_000_000),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "vae/model.safetensors" in names
        assert "vae/config.json" in names
        assert not any("vae_decoder" in n for n in names)
        assert not any("vae_encoder" in n for n in names)

    def test_root_safetensors_kept_when_no_component_layout(self):
        """For single-file repos, root safetensors should NOT be skipped."""
        siblings = [
            _make_sibling("model.safetensors", 5_000_000_000),
            _make_sibling("config.json", 500),
        ]
        filtered = self.mgr._filter_inference_files(siblings)
        names = [s.rfilename for s in filtered]
        assert "model.safetensors" in names

    def test_allow_patterns_basic(self):
        siblings = [
            _make_sibling("model_v1.safetensors", 5_000_000_000),
            _make_sibling("model_v2.safetensors", 5_000_000_000),
            _make_sibling("config.json", 500),
        ]
        filtered = self.mgr._filter_inference_files(
            siblings, allow_patterns=["model_v2*", "*.json"],
        )
        names = [s.rfilename for s in filtered]
        assert "model_v2.safetensors" in names
        assert "config.json" in names
        assert "model_v1.safetensors" not in names

    def test_allow_patterns_still_skips_training_artifacts(self):
        """allow_patterns narrows first, then ignore patterns still apply."""
        siblings = [
            _make_sibling("model.safetensors"),
            _make_sibling("optimizer.bin", 1_000_000_000),
            _make_sibling("README.md"),
        ]
        filtered = self.mgr._filter_inference_files(
            siblings, allow_patterns=["*"],
        )
        names = [s.rfilename for s in filtered]
        assert "model.safetensors" in names
        assert "optimizer.bin" not in names
        assert "README.md" not in names

    def test_empty_siblings(self):
        filtered = self.mgr._filter_inference_files([])
        assert filtered == []

    def test_narrow_allow_patterns_still_keeps_configs(self):
        """A narrow allow_patterns (e.g. ['*Q4_K_M*'] for a GGUF catalog
        entry) must still keep root-level config.json + tokenizer files.

        Regression for installs that left the model dir with only the .gguf
        blob — at inference time the loader hit JSONDecodeError / "missing
        config.json" because the always-include patterns were never applied
        on the already-pre-filtered sibling list.
        """
        siblings = [
            _make_sibling("model-Q4_K_M.gguf", 3_000_000_000),
            _make_sibling("model-Q5_K_M.gguf", 4_000_000_000),
            _make_sibling("config.json", 1000),
            _make_sibling("model_index.json", 500),
            _make_sibling("tokenizer.json", 5000),
            _make_sibling("tokenizer/merges.txt", 524_619),
            _make_sibling("scheduler/scheduler_config.json", 300),
            _make_sibling("README.md", 3000),
        ]
        filtered = self.mgr._filter_inference_files(
            siblings, allow_patterns=["*Q4_K_M*"],
        )
        names = [s.rfilename for s in filtered]
        assert "model-Q4_K_M.gguf" in names
        assert "model-Q5_K_M.gguf" not in names
        # _ALWAYS_INCLUDE files survive the narrow allow_patterns filter
        assert "config.json" in names
        assert "model_index.json" in names
        assert "tokenizer.json" in names
        assert "tokenizer/merges.txt" in names
        assert "scheduler/scheduler_config.json" in names
        # _IGNORE_PATTERNS still apply
        assert "README.md" not in names
