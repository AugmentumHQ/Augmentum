"""Post-install provider bridge — register a managed service as a usable
provider across every category.

The gap this closes: marketplace/Discover install started the container
but never registered it as a provider, so a UI-installed service (Matt's
speaches) was invisible forever. These tests pin the catalog-driven
register-for-every-category behavior + the post-install next-steps.
"""

from __future__ import annotations

import pytest

from augmentum.providers.models import ServiceCategory, ServiceDefinition
from augmentum.providers.provider_bridge import (
    extract_default_model,
    register_provider_for_service,
    resolve_service_url,
)


def _sd(**kw) -> ServiceDefinition:
    base = dict(
        id="speaches-stt", name="Speaches", description="",
        category=ServiceCategory.STT, image="img",
        internal_port=8000, host_port=6200,
        env={"PRELOAD_MODELS": '["Systran/faster-whisper-small.en"]'},
        api_type="openai_stt",
        augmentum_env={"AUGMENTUM_STT_PROVIDER_URL": "http://{container_name}:{internal_port}"},
    )
    base.update(kw)
    return ServiceDefinition(**base)


# ----------------------------------------------------------------------
# URL + model resolution
# ----------------------------------------------------------------------

def test_resolve_service_url_strips_prefix_and_substitutes():
    key, url = resolve_service_url(_sd())
    assert key == "stt_provider_url"
    assert url == "http://augmentum-speaches-stt:8000"


def test_resolve_service_url_empty_when_no_template():
    key, url = resolve_service_url(_sd(augmentum_env={}))
    assert key == "" and url == ""


def test_extract_model_from_preload_json_list():
    assert extract_default_model(_sd()) == "Systran/faster-whisper-small.en"


def test_extract_model_from_plain_env():
    assert extract_default_model(_sd(env={"MODEL": "whisper-base"})) == "whisper-base"


def test_extract_model_api_type_fallback():
    sd = _sd(env={}, api_type="openai_tts", category=ServiceCategory.TTS)
    assert extract_default_model(sd) == "kokoro"


def test_extract_model_tolerates_malformed_preload():
    assert extract_default_model(_sd(env={"PRELOAD_MODELS": "not-json"})) == "not-json"


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class _Cur:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class _FakeConn:
    """Simulates the audio_providers / image_providers tables."""

    def __init__(self, existing_type_count=0, existing_row=None):
        self.existing_type_count = existing_type_count
        self.existing_row = existing_row
        self.inserts: list[tuple] = []
        self.updates: list[tuple] = []

    async def execute(self, q, params=()):
        ql = " ".join(q.split())
        if ql.startswith("SELECT is_default FROM"):
            return _Cur(self.existing_row)
        if "COUNT(*)" in ql:
            return _Cur((self.existing_type_count,))
        if ql.startswith("INSERT"):
            self.inserts.append(params)
            return _Cur(None)
        if ql.startswith("UPDATE"):
            self.updates.append(params)
            return _Cur(None)
        return _Cur(None)

    async def commit(self):
        pass


class _FakeRegistry:
    def __init__(self):
        self.registered: dict[str, object] = {}
        self.invalidated = False

    def register_backend(self, key, backend):
        self.registered[key] = backend

    def invalidate_model_map(self):
        self.invalidated = True


class _FakeSettingsStore:
    def __init__(self):
        self.values: dict[str, str] = {}

    async def set(self, key, value):
        self.values[key] = value


class _FakeHttp:
    pass


# ----------------------------------------------------------------------
# Registration per category
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stt_registers_as_audio_provider_default_when_first():
    conn = _FakeConn(existing_type_count=0)
    reg = await register_provider_for_service(_sd(), conn=conn)
    assert reg.registered is True
    assert reg.provider_type == "stt"
    assert reg.target == "audio_providers"
    assert reg.is_default is True            # first STT
    assert len(conn.inserts) == 1
    # next-steps: no set_default (already default), but open_webui + test + model
    actions = [s.action for s in reg.next_steps]
    assert "set_default" not in actions
    assert "test" in actions and "open_webui" in actions


@pytest.mark.asyncio
async def test_stt_not_default_when_other_exists_offers_set_default():
    conn = _FakeConn(existing_type_count=1)  # a TTS/STT already present
    reg = await register_provider_for_service(_sd(), conn=conn)
    assert reg.is_default is False
    assert "set_default" in [s.action for s in reg.next_steps]


@pytest.mark.asyncio
async def test_existing_row_updates_not_inserts():
    conn = _FakeConn(existing_row=(1,))  # already registered, is_default=1
    reg = await register_provider_for_service(_sd(), conn=conn)
    assert reg.registered is True
    assert reg.is_default is True
    assert len(conn.updates) == 1 and len(conn.inserts) == 0


@pytest.mark.asyncio
async def test_image_registers_as_image_provider():
    sd = _sd(id="sd-webui", name="SD", category=ServiceCategory.IMAGE,
             api_type="openai_image", env={},
             augmentum_env={"AUGMENTUM_IMAGE_URL": "http://{container_name}:{internal_port}"})
    conn = _FakeConn(existing_type_count=0)
    reg = await register_provider_for_service(sd, conn=conn)
    assert reg.registered is True
    assert reg.target == "image_providers"
    assert len(conn.inserts) == 1


@pytest.mark.asyncio
async def test_llm_persists_setting_and_hot_registers_backend():
    sd = _sd(id="ollama-gpu", name="Ollama", category=ServiceCategory.LLM,
             api_type="ollama", env={}, internal_port=11434, host_port=11435,
             augmentum_env={"AUGMENTUM_OLLAMA_BASE_URL": "http://{container_name}:{internal_port}"})
    store = _FakeSettingsStore()
    registry = _FakeRegistry()
    reg = await register_provider_for_service(
        sd, conn=None, settings_store=store, registry=registry, http_client=_FakeHttp(),
    )
    assert reg.registered is True
    assert reg.target == "settings"
    # Setting persisted under the derived key.
    assert store.values.get("ollama_base_url") == "http://augmentum-ollama-gpu:11434"
    # Backend hot-registered + model map invalidated.
    assert "ollama" in registry.registered
    assert registry.invalidated is True


@pytest.mark.asyncio
async def test_no_template_yields_unregistered_with_retry_step():
    reg = await register_provider_for_service(_sd(augmentum_env={}), conn=_FakeConn())
    assert reg.registered is False
    assert "no augmentum_env" in reg.detail
    assert "retry" in [s.action for s in reg.next_steps]


@pytest.mark.asyncio
async def test_audio_without_conn_fails_gracefully_not_raises():
    reg = await register_provider_for_service(_sd(), conn=None)
    assert reg.registered is False
    assert reg.detail  # carries the reason
    assert "retry" in [s.action for s in reg.next_steps]
