"""Unit tests for the managed classifier slot ("Slot C").

Slot C is the augmentum-managed, runtime-switchable counterpart to the
external compose.classifier.yaml container. It serves the classifier/utility
roles (registered under the "classifier" backend key) and exposes a
vision-capability bit so the vision substrate can route captioning to it.

These tests pin the contract WITHOUT spawning a subprocess: construction is
cheap, the manager is resident (idle_timeout=0), vision capability tracks the
loaded model's mmproj, and the backend resolves both the classifier and
utility roles.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from augmentum.models.base import ModelInfo


def _slot():
    from augmentum.models.classifier_slot import ClassifierSlot, ClassifierSlotConfig

    return ClassifierSlot(
        ClassifierSlotConfig(model_dirs=["/models"]), http_client=MagicMock(),
    )


def _fake_manager(state="READY", model_id="gemma-4-e2b", mmproj=""):
    m = MagicMock()
    m.state.name = state
    m.model_id = model_id
    m.current_mmproj_path = mmproj
    m.status.return_value = {"state": state, "model_id": model_id}
    return m


def test_construction_is_cheap_no_subprocess():
    """Constructing the slot does NOT build a manager or spawn anything."""
    s = _slot()
    assert s.manager is None
    assert s.base_url == ""


def test_backend_builds_resident_manager():
    """Accessing .backend lazily builds the manager + LlamaCppBackend, and the
    manager is RESIDENT (idle_timeout=0) so it never cold-reloads mid-turn."""
    s = _slot()
    with patch(
        "augmentum.models.llama_server_manager.LlamaServerManager",
    ) as MgrCls, patch("augmentum.models.llama_cpp.LlamaCppBackend") as BeCls:
        mgr = MagicMock()
        mgr.base_url = "http://localhost:8093"
        MgrCls.return_value = mgr
        BeCls.return_value = MagicMock()

        be = s.backend
        assert be is not None
        MgrCls.assert_called_once()
        # Resident: never auto-unload.
        assert mgr.idle_timeout == 0.0
        # LlamaCppBackend (preserves chat_template_kwargs / enable_thinking).
        BeCls.assert_called_once()


def test_vision_capable_false_when_not_loaded():
    s = _slot()
    assert s.is_vision_capable() is False


def test_vision_capable_true_only_with_mmproj():
    """Vision capability == VL model launched WITH its mmproj projector."""
    s = _slot()
    s._manager = _fake_manager(mmproj="/models/mmproj-gemma.gguf")
    assert s.is_vision_capable() is True

    s._manager = _fake_manager(mmproj="")  # text-only model
    assert s.is_vision_capable() is False

    s._manager = _fake_manager(state="LOADING", mmproj="/m/p.gguf")  # not ready
    assert s.is_vision_capable() is False


def test_status_shape_includes_vision_bit():
    s = _slot()
    # Empty (built but no model) — still reports the bit.
    s._manager = None
    empty = s.status()
    assert empty["enabled"] is True
    assert empty["loaded"] is False
    assert empty["vision_capable"] is False

    s._manager = _fake_manager(model_id="gemma-4-e2b", mmproj="/m/p.gguf")
    st = s.status()
    assert st["enabled"] is True
    assert st["loaded"] is True
    assert st["vision_capable"] is True


def test_resolve_abs_path_passthrough():
    s = _slot()
    s._manager = _fake_manager()  # so _build() is a no-op
    assert s.resolve_model_path("/abs/model.gguf") == "/abs/model.gguf"


# -- role resolution: the slot backend serves classifier AND utility ----

class _FakeBackend:
    def __init__(self, model_names: list[str], loaded: str = "") -> None:
        self._names = list(model_names)
        self._manager = MagicMock()
        self._manager.model_id = loaded

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=n, model=n) for n in self._names]


def _registry():
    with patch("augmentum.models.provider_registry.settings") as mock_settings:
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"
        from augmentum.models.provider_registry import ProviderRegistry

        return ProviderRegistry(MagicMock())


@pytest.mark.asyncio
async def test_slot_backend_serves_classifier_and_utility_roles():
    """Registering the slot under the "classifier" key wires BOTH the
    classifier and utility roles to it (resolve_model_for_role unchanged),
    returning the synced classifier_sidecar_model id."""
    reg = _registry()
    slot_backend = _FakeBackend([], loaded="gemma-4-e2b")
    reg.register_backend("classifier", slot_backend)

    ms = MagicMock()
    ms.classifier_model = ""
    ms.utility_model = ""
    ms.classifier_sidecar_model = "gemma-4-e2b"

    b, name = await reg.resolve_model_for_role("classifier", settings=ms)
    assert b is slot_backend
    assert name == "gemma-4-e2b"

    b2, name2 = await reg.resolve_model_for_role("utility", settings=ms)
    assert b2 is slot_backend
    assert name2 == "gemma-4-e2b"


@pytest.mark.asyncio
async def test_explicit_classifier_model_still_wins_over_slot():
    """An explicit classifier_model setting overrides the slot (user choice
    is never second-guessed)."""
    reg = _registry()
    slot_backend = _FakeBackend([], loaded="gemma-4-e2b")
    primary = _FakeBackend(["my-model"])
    reg.register_backend("classifier", slot_backend)
    reg.register_backend("engine", primary)

    ms = MagicMock()
    ms.classifier_model = "my-model"
    ms.utility_model = ""
    ms.classifier_sidecar_model = "gemma-4-e2b"

    b, name = await reg.resolve_model_for_role("classifier", settings=ms)
    assert name == "my-model"
    assert b is not slot_backend

# -- load(): per-model saved options vs caller options -------------------

def _slot_with_async_manager(saved: dict | None):
    """Slot with a manager stub whose _load_saved_options returns ``saved``."""
    from unittest.mock import AsyncMock

    s = _slot()
    mgr = MagicMock()
    mgr._load_saved_options = AsyncMock(return_value=saved)
    mgr.swap = AsyncMock()
    mgr.start = AsyncMock()
    mgr.persist_load_options = AsyncMock()
    mgr.model_id = "gemma-4-e2b"
    s._manager = mgr  # _build() no-ops
    return s, mgr


@pytest.mark.asyncio
async def test_load_empty_options_replays_saved():
    """No caller options → the per-model saved profile is replayed."""
    from augmentum.models.llama_server_manager import ProcessState

    s, mgr = _slot_with_async_manager({"ctx_size": 96000, "mmproj_path": "/m/p.gguf"})
    mgr.state = ProcessState.READY
    await s.load("/models/gemma-4-e2b.gguf")
    _, kwargs = mgr.swap.call_args
    assert kwargs["load_options"]["ctx_size"] == 96000
    assert kwargs["load_options"]["mmproj_path"] == "/m/p.gguf"


@pytest.mark.asyncio
async def test_load_merge_saved_lets_profile_override_boot_globals():
    """Boot path (merge_saved=True): globals are DEFAULTS, the user's saved
    per-model ctx/mmproj win — restart no longer reverts slot config. The
    resident invariant (idle_timeout=0) survives even if the saved profile
    (written by a primary-engine load) carries a nonzero idle_timeout."""
    from augmentum.models.llama_server_manager import ProcessState

    s, mgr = _slot_with_async_manager(
        {"ctx_size": 96000, "mmproj_path": "/m/p.gguf", "idle_timeout": 300},
    )
    mgr.state = ProcessState.READY
    await s.load(
        "/models/gemma-4-e2b.gguf",
        load_options={"idle_timeout": 0, "ctx_size": 4096, "gpu_layers": 99},
        merge_saved=True,
    )
    _, kwargs = mgr.swap.call_args
    opts = kwargs["load_options"]
    assert opts["ctx_size"] == 96000        # saved beats global default
    assert opts["mmproj_path"] == "/m/p.gguf"
    assert opts["gpu_layers"] == 99         # global fills the gap
    assert opts["idle_timeout"] == 0        # resident invariant forced


@pytest.mark.asyncio
async def test_load_explicit_options_keep_caller_wins():
    """Explicit (UI/API) loads: caller options win untouched; saved profile
    is NOT merged over them (merge_saved defaults False)."""
    from augmentum.models.llama_server_manager import ProcessState

    s, mgr = _slot_with_async_manager({"ctx_size": 96000})
    mgr.state = ProcessState.READY
    await s.load(
        "/models/gemma-4-e2b.gguf", load_options={"ctx_size": 8192},
    )
    _, kwargs = mgr.swap.call_args
    assert kwargs["load_options"]["ctx_size"] == 8192
