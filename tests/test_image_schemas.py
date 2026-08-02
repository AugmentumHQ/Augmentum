"""Tests for image/schemas.py -- request/response model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from augmentum.image.schemas import (
    AspectRatio,
    BatchDeleteRequest,
    GenerateRequest,
    GenerateResponse,
    HistoryEntry,
    Img2ImgRequest,
    InpaintRequest,
    JobStatus,
    JobType,
    LoraWeight,
    ModelInfo,
    OpenAIImageRequest,
    OpenAIImageResponse,
    PipelineType,
)


class TestEnums:
    def test_aspect_ratio_values(self):
        assert AspectRatio.PORTRAIT == "portrait"
        assert AspectRatio.LANDSCAPE == "landscape"
        assert AspectRatio.SQUARE == "square"

    def test_job_type_values(self):
        assert JobType.TXT2IMG == "txt2img"
        assert JobType.IMG2IMG == "img2img"

    def test_pipeline_type_values(self):
        assert PipelineType.SD15 == "sd15"
        assert PipelineType.SDXL == "sdxl"
        assert PipelineType.FLUX == "flux"


class TestGenerateRequest:
    def test_minimal_valid(self):
        req = GenerateRequest(prompt="a cat")
        assert req.prompt == "a cat"
        assert req.seed == -1

    def test_full_input(self):
        req = GenerateRequest(
            prompt="a cat",
            negative_prompt="ugly",
            model="sd15",
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.0,
            seed=42,
            aspect=AspectRatio.SQUARE,
        )
        assert req.width == 512
        assert req.seed == 42

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_lora_weight_validation(self):
        lora = LoraWeight(name="my_lora", weight=1.5)
        assert lora.weight == 1.5


class TestImg2ImgRequest:
    def test_minimal_valid(self):
        req = Img2ImgRequest(prompt="a cat", source_image="base64data")
        assert req.strength == 0.75

    def test_strength_range(self):
        req = Img2ImgRequest(prompt="a cat", source_image="data", strength=0.5)
        assert req.strength == 0.5

    def test_strength_out_of_range(self):
        with pytest.raises(ValidationError):
            Img2ImgRequest(prompt="a cat", source_image="data", strength=1.5)


class TestInpaintRequest:
    def test_minimal_valid(self):
        req = InpaintRequest(prompt="fill sky", source_image="img", mask_image="mask")
        assert req.strength == 1.0
        assert req.mask_blur == 4

    def test_inpaint_mode_validation(self):
        req = InpaintRequest(prompt="fix", source_image="img", mask_image="mask",
                             inpaint_mode="improve")
        assert req.inpaint_mode == "improve"


class TestGenerateResponse:
    def test_construct(self):
        resp = GenerateResponse(image_id="img123", job_id="job456",
                                status=JobStatus.COMPLETED, seed=42)
        assert resp.image_id == "img123"
        assert resp.status == JobStatus.COMPLETED


class TestOpenAIImageRequest:
    def test_minimal_valid(self):
        req = OpenAIImageRequest(prompt="a sunset")
        assert req.size == "1024x1024"
        assert req.n == 1

    def test_size_validation_valid(self):
        req = OpenAIImageRequest(prompt="test", size="512x768")
        assert req.size == "512x768"

    def test_size_validation_auto(self):
        req = OpenAIImageRequest(prompt="test", size="auto")
        assert req.size == "auto"

    def test_size_validation_invalid(self):
        with pytest.raises(ValidationError):
            OpenAIImageRequest(prompt="test", size="100x100")

    def test_quality_validation(self):
        req = OpenAIImageRequest(prompt="test", quality="hd")
        assert req.quality == "hd"

    def test_quality_invalid(self):
        with pytest.raises(ValidationError):
            OpenAIImageRequest(prompt="test", quality="super")


class TestModelInfo:
    def test_construct(self):
        info = ModelInfo(name="test_model", pipeline_type=PipelineType.SDXL)
        assert info.name == "test_model"
        assert info.is_loaded is False


class TestHistoryEntry:
    def test_construct_minimal(self):
        entry = HistoryEntry(image_id="img1", prompt="test")
        assert entry.job_type == "txt2img"
        assert entry.is_private is False
