"""Tests for the capability schema + serialisation.

Pins:
  - dataclass constructors accept their typed fields
  - serialise/deserialise roundtrip is lossless
  - unknown kinds in deserialise return None (forward compat)
  - bulk deserialise drops unknown kinds silently
  - schema mismatch in deserialise returns None (doesn't raise)
"""
from __future__ import annotations

from augmentum.fabric.capabilities import (
    KIND_IMAGE_GENERATION,
    KIND_LLM_INFERENCE,
    ImageGenerationCapability,
    KnowledgeSearchCapability,
    LLMInferenceCapability,
    deserialise,
    deserialise_list,
    serialise,
)


def test_llm_capability_roundtrip():
    cap = LLMInferenceCapability(
        backend="engine",
        model_id="Qwen3.5-72B-A10B-q4",
        model_family="qwen3",
        params_b=72.0,
        active_params_b=10.0,
        ctx_max=32768,
        loaded=True,
        free_slots=2,
        device={"gpu_name": "GPU-A", "vram_free_mb": 8200, "vram_total_mb": 24576},
    )
    raw = serialise(cap)
    assert raw["kind"] == KIND_LLM_INFERENCE
    assert raw["model_id"] == "Qwen3.5-72B-A10B-q4"

    parsed = deserialise(raw)
    assert isinstance(parsed, LLMInferenceCapability)
    assert parsed == cap


def test_image_capability_roundtrip():
    cap = ImageGenerationCapability(
        backend="diffusers",
        model_id="flux-schnell-fp8",
        family="flux",
        loaded=True,
        max_resolution="1024x1024",
    )
    parsed = deserialise(serialise(cap))
    assert isinstance(parsed, ImageGenerationCapability)
    assert parsed == cap


def test_knowledge_capability_roundtrip():
    cap = KnowledgeSearchCapability(
        pack_id="wikipedia_en_simple_2026-02",
        pack_name="Wikipedia (Simple English)",
        chunk_count=12345,
        embedding_dim=768,
        active=True,
        pack_format="augpack+zim",
    )
    parsed = deserialise(serialise(cap))
    assert isinstance(parsed, KnowledgeSearchCapability)
    assert parsed == cap


def test_deserialise_unknown_kind_returns_none():
    raw = {"kind": "some.future.kind", "schema_version": 1, "foo": "bar"}
    assert deserialise(raw) is None


def test_deserialise_missing_kind_returns_none():
    assert deserialise({}) is None
    assert deserialise({"schema_version": 1}) is None


def test_deserialise_extra_fields_are_ignored():
    """A peer running a newer Augmentum may add optional fields. We
    should accept the known fields and drop the rest, not crash.
    """
    raw = {
        "kind": KIND_LLM_INFERENCE,
        "schema_version": 1,
        "model_id": "test-model",
        "future_field_we_dont_know_about": "ignored",
    }
    parsed = deserialise(raw)
    assert isinstance(parsed, LLMInferenceCapability)
    assert parsed.model_id == "test-model"


def test_deserialise_list_drops_unknowns():
    items = [
        {"kind": KIND_LLM_INFERENCE, "model_id": "a"},
        {"kind": "unknown.kind", "model_id": "b"},  # dropped
        {"kind": KIND_IMAGE_GENERATION, "model_id": "c"},
        "not_even_a_dict",  # dropped
    ]
    out = deserialise_list(items)
    assert len(out) == 2
    assert out[0].kind == KIND_LLM_INFERENCE
    assert out[1].kind == KIND_IMAGE_GENERATION


def test_deserialise_list_handles_non_list_input():
    assert deserialise_list({}) == []
    assert deserialise_list("string") == []
    assert deserialise_list(None) == []
