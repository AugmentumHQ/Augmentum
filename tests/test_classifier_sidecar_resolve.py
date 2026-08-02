"""Classifier role prefers the local classifier sidecar when present.

The dedicated sidecar (SmolLM2-135M by default; compose.classifier.yaml)
auto-registers as the ``classifier`` backend; the role resolver routes the
classification hop to it with zero config, while an explicit
``classifier_model`` still overrides.

SCOPING: the onboard sidecar serves the ``classifier`` AND ``utility`` roles
(2026-06-17 — memory consolidation/compaction/reflection and other recurring
sub-turn reasoning), but NEVER primary chat. Precedence guarantees the user
stays in control: a per-feature ``override`` or a manual ``utility_model``
wins over the sidecar, and ``primary_chat_model`` catches the no-sidecar case
so these tasks keep working when no dedicated model is running.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from augmentum.models.provider_registry import ProviderRegistry


def _registry(backends: dict) -> ProviderRegistry:
    # Bypass the heavy __init__ — the sidecar branch only touches _backends.
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg._backends = backends
    return reg


def test_sidecar_used_when_role_on_auto():
    sidecar = SimpleNamespace(name="sidecar")
    reg = _registry({"classifier": sidecar})
    settings = SimpleNamespace(
        classifier_model="",
        classifier_sidecar_model="smollm2-135m-instruct",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("classifier", settings=settings)
    )
    assert backend is sidecar
    assert model == "smollm2-135m-instruct"


def test_utility_model_overrides_sidecar():
    # A manual ``utility_model`` wins over the onboard sidecar — a user who
    # runs their OWN model and sets it is never silently overridden.
    sidecar = SimpleNamespace(name="sidecar")

    async def _fake_resolve(name):
        return (SimpleNamespace(name="util-backend"), name)

    reg = _registry({"classifier": sidecar})
    reg.resolve_backend_with_fabric = _fake_resolve  # type: ignore[method-assign]
    settings = SimpleNamespace(
        classifier_model="",
        utility_model="qwen3-utility",
        classifier_sidecar_model="smollm2-135m-instruct",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("utility", settings=settings)
    )
    assert backend is not sidecar
    assert model == "qwen3-utility"


def test_utility_uses_sidecar_on_auto():
    # On AUTO (no utility_model), the utility role now prefers the onboard
    # sidecar — so memory consolidation/compaction/reflection run on the
    # resident small-reasoner (e.g. Gemma 4 E2B) instead of contending the
    # primary chat model.
    sidecar = SimpleNamespace(name="sidecar")
    reg = _registry({"classifier": sidecar})
    settings = SimpleNamespace(
        classifier_model="",
        utility_model="",
        classifier_sidecar_model="gemma-4-e2b",
        primary_chat_model="deepseek-v4-pro",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("utility", settings=settings)
    )
    assert backend is sidecar
    assert model == "gemma-4-e2b"


def test_utility_falls_back_to_primary_without_sidecar():
    # No sidecar running + no utility_model → the utility role still works,
    # falling back to the primary chat model. This is the "user didn't run
    # the dedicated classifier model" path — nothing breaks.
    async def _fake_resolve(name):
        return (SimpleNamespace(name="primary-backend"), name)

    reg = _registry({})  # no sidecar registered
    reg.resolve_backend_with_fabric = _fake_resolve  # type: ignore[method-assign]
    settings = SimpleNamespace(
        classifier_model="",
        utility_model="",
        primary_chat_model="deepseek-v4-pro",
        classifier_sidecar_model="gemma-4-e2b",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("utility", settings=settings)
    )
    assert model == "deepseek-v4-pro"


def test_sidecar_never_used_for_primary_role():
    # A non-classifier/utility role falls straight through to
    # primary_chat_model; the classifier backend is never consulted.
    sidecar = SimpleNamespace(name="sidecar")

    async def _fake_resolve(name):
        return (SimpleNamespace(name="primary-backend"), name)

    reg = _registry({"classifier": sidecar})
    reg.resolve_backend_with_fabric = _fake_resolve  # type: ignore[method-assign]
    settings = SimpleNamespace(
        classifier_model="",
        utility_model="",
        primary_chat_model="deepseek-v4-pro",
        classifier_sidecar_model="smollm2-135m-instruct",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("primary_chat", settings=settings)
    )
    assert backend is not sidecar
    assert model == "deepseek-v4-pro"


def test_no_sidecar_falls_through():
    # Without the sidecar registered, the classifier branch must NOT crash
    # and must fall through (here to utility_model).
    captured = {}

    async def _fake_resolve(name):
        captured["name"] = name
        return (SimpleNamespace(name="util"), name)

    reg = _registry({})
    reg.resolve_backend_with_fabric = _fake_resolve  # type: ignore[method-assign]
    settings = SimpleNamespace(classifier_model="", utility_model="qwen3-utility")
    backend, model = asyncio.run(
        reg.resolve_model_for_role("classifier", settings=settings)
    )
    assert model == "qwen3-utility"
    assert captured["name"] == "qwen3-utility"


def test_explicit_classifier_model_overrides_sidecar():
    sidecar = SimpleNamespace(name="sidecar")

    async def _fake_resolve(name):
        return (SimpleNamespace(name="explicit"), name)

    reg = _registry({"classifier": sidecar})
    reg.resolve_backend_with_fabric = _fake_resolve  # type: ignore[method-assign]
    settings = SimpleNamespace(
        classifier_model="my-chosen-model",
        classifier_sidecar_model="smollm2-135m-instruct",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("classifier", settings=settings)
    )
    # Explicit choice wins; the sidecar is ignored.
    assert model == "my-chosen-model"
    assert backend.name == "explicit"


# --- HF-id ↔ filename-stem normalization (the classifier drop bug) ---------
#
# A model saved into a role setting in its canonical HF form
# ("unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL") never exact-matches the SAME
# GGUF served by a local llama-server, which advertises the filename STEM
# ("unsloth_gemma-4-E2B-it-qat-GGUF_UD-Q4_K_XL", from Path(model_path).stem).
# The strict map lookup missed, the request fell through to the default
# backend / fabric, and the always-loaded classifier silently dropped every
# voice turn. resolve_backend_for_model now retries under the canonical
# alnum-lowercase normalizer.

_HF_ID = "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"
_STEM = "unsloth_gemma-4-E2B-it-qat-GGUF_UD-Q4_K_XL"


def _map_registry(model_map: dict, backends: dict) -> ProviderRegistry:
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg._backends = backends
    reg._model_map = model_map
    reg._model_pins = {}
    reg._lb_registry = None
    reg._default = ""
    reg._model_unverified = set()
    reg._unverified_served_logged = set()

    async def _noop_refresh(*a, **k):
        return reg._model_map

    reg.refresh_model_map = _noop_refresh  # type: ignore[method-assign]
    return reg


def test_resolve_normalizes_hf_id_to_served_stem():
    # The exact bug: setting holds the HF id, the local backend serves the
    # filename stem. Resolution must find it and return the ACTUAL served
    # name (so downstream dispatch uses the name llama-server knows).
    sidecar = SimpleNamespace(name="sidecar")
    reg = _map_registry({_STEM: "classifier"}, {"classifier": sidecar})

    backend, model = asyncio.run(reg.resolve_backend_for_model(_HF_ID))
    assert backend is sidecar
    assert model == _STEM


def test_classifier_model_hf_id_resolves_to_local_sidecar_end_to_end():
    # Full role path: classifier_model is the HF id, the sidecar serves the
    # stem, fabric disabled (director None). resolve_model_for_role ->
    # resolve_backend_with_fabric -> resolve_backend_for_model must land on
    # the local sidecar instead of raising ModelUnavailableError.
    sidecar = SimpleNamespace(name="sidecar")
    reg = _map_registry({_STEM: "classifier"}, {"classifier": sidecar})
    reg._fabric_director = None
    settings = SimpleNamespace(
        classifier_model=_HF_ID,
        classifier_sidecar_model="smollm2-135m-instruct",
    )
    backend, model = asyncio.run(
        reg.resolve_model_for_role("classifier", settings=settings)
    )
    assert backend is sidecar
    assert model == _STEM


def test_ambiguous_normalized_match_does_not_cross_route():
    # Safety guard: two DIFFERENT backends whose names collapse to the same
    # normalized key must NOT resolve via the fuzzy fallback — better to fall
    # through than to silently route to the wrong backend.
    default_be = SimpleNamespace(name="default")
    reg = _map_registry(
        {_STEM: "a", _HF_ID.replace("/", "_").replace(":", "_") + "x": "b"},
        {"a": SimpleNamespace(name="a"), "b": SimpleNamespace(name="b")},
    )
    # Make both entries collide under the normalizer.
    reg._model_map = {_STEM: "a", "unsloth/gemma-4-E2B-it-qat-GGUF:UD/Q4-K-XL": "b"}
    reg.get_backend = lambda *a, **k: default_be  # type: ignore[method-assign]

    backend, model = asyncio.run(reg.resolve_backend_for_model(_HF_ID))
    # Ambiguous -> fell through to default, original name preserved.
    assert backend is default_be
    assert model == _HF_ID
