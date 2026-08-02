"""Smoke tests — import and construct every module in augmentum/models/."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestBaseDataModels:
    """Verify all base data models can be instantiated with defaults."""

    def test_construct_message(self):
        from augmentum.models.base import Message

        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.images is None
        assert msg.tool_calls is None
        assert msg.thinking is None

    def test_construct_usage(self):
        from augmentum.models.base import Usage

        u = Usage()
        assert u.prompt_tokens == 0
        assert u.total_tokens == 0

    def test_construct_model_info(self):
        from augmentum.models.base import ModelInfo

        mi = ModelInfo(name="test", model="test")
        assert mi.name == "test"
        assert mi.size == 0
        assert mi.vision is False
        assert mi.context_length == 0

    def test_construct_model_details(self):
        from augmentum.models.base import ModelDetails

        md = ModelDetails()
        assert md.modelfile == ""
        assert md.format == ""

    def test_construct_internal_chat_request(self):
        from augmentum.models.base import InternalChatRequest, Message

        req = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
        )
        assert req.model == "test-model"
        assert req.stream is False
        assert req.think is False
        assert req.voice_input is False

    def test_construct_internal_chat_response(self):
        from augmentum.models.base import InternalChatResponse, Message

        resp = InternalChatResponse(
            message=Message(role="assistant", content="hey"),
            model="test",
        )
        assert resp.finish_reason is None
        assert resp.usage.total_tokens == 0

    def test_construct_internal_stream_chunk(self):
        from augmentum.models.base import InternalStreamChunk

        chunk = InternalStreamChunk()
        assert chunk.content_delta == ""
        assert chunk.thinking_delta == ""
        assert chunk.done is False


class TestVisionDetection:
    """Test the is_vision_model_name heuristic."""

    def test_detect_llava(self):
        from augmentum.models.base import is_vision_model_name

        assert is_vision_model_name("llava:7b") is True

    def test_detect_gpt4o(self):
        from augmentum.models.base import is_vision_model_name

        assert is_vision_model_name("gpt-4o-mini") is True

    def test_detect_gemini_flash(self):
        from augmentum.models.base import is_vision_model_name

        assert is_vision_model_name("gemini-2.0-flash") is True

    def test_detect_claude3(self):
        from augmentum.models.base import is_vision_model_name

        assert is_vision_model_name("claude-3-opus-20240229") is True

    def test_reject_llama_text(self):
        from augmentum.models.base import is_vision_model_name

        assert is_vision_model_name("llama3.1:8b") is False


class TestVisionPromptInjection:
    """Test inject_vision_prompt on edge cases."""

    def test_inject_empty_text_with_images(self):
        from augmentum.models.base import InternalChatRequest, Message, inject_vision_prompt

        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="", images=["data:image/png;base64,abc"])],
        )
        inject_vision_prompt(req)
        assert "Describe" in req.messages[0].content

    def test_no_inject_when_no_images(self):
        from augmentum.models.base import InternalChatRequest, Message, inject_vision_prompt

        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )
        inject_vision_prompt(req)
        assert req.messages[0].content == "hi"


class TestApplyVisionPipeline:
    """The canonical image-handling sequence reused by non-chat surfaces
    (voice turn, future live camera) so they match the chat routes."""

    async def test_runs_three_steps_in_order(self, monkeypatch):
        import augmentum.models.base as base

        calls: list[str] = []

        async def _resolve(req, state):
            calls.append("resolve")

        async def _caption(req, state, backend, **kwargs):
            calls.append("caption")

        def _inject(req):
            calls.append("inject")

        monkeypatch.setattr(base, "resolve_chat_image_urls", _resolve)
        monkeypatch.setattr(base, "caption_via_router_fallback", _caption)
        monkeypatch.setattr(base, "inject_vision_prompt", _inject)

        req = base.InternalChatRequest(
            model="m",
            messages=[base.Message(role="user", content="what is this",
                                   images=["data:image/png;base64,abc"])],
        )
        await base.apply_vision_pipeline(req, object(), object())
        # resolve refs → caption-fallback (text-only) → ensure prompt.
        assert calls == ["resolve", "caption", "inject"]

    async def test_passes_backend_to_caption_fallback(self, monkeypatch):
        import augmentum.models.base as base

        seen = {}

        async def _resolve(req, state):
            pass

        async def _caption(req, state, backend, **kwargs):
            seen["backend"] = backend

        def _inject(req):
            pass

        monkeypatch.setattr(base, "resolve_chat_image_urls", _resolve)
        monkeypatch.setattr(base, "caption_via_router_fallback", _caption)
        monkeypatch.setattr(base, "inject_vision_prompt", _inject)

        backend = object()
        req = base.InternalChatRequest(model="m", messages=[base.Message(role="user", content="hi")])
        await base.apply_vision_pipeline(req, object(), backend)
        # The resolved backend must reach the fallback so its
        # can_read_images_natively() check decides direct-vs-caption.
        assert seen["backend"] is backend


class _FakeRouter:
    """Records the caption call; pretends a provider is available."""

    def __init__(self, available=True, text="a wooden chair"):
        self._available = available
        self.text = text
        self.calls = []

    async def is_available(self):
        return self._available

    async def caption(self, image_bytes, *, prompt, max_tokens, timeout_s, workload, frames=None):
        self.calls.append({"prompt": prompt, "frames": frames, "max_tokens": max_tokens})
        return self.text


class _TextOnlyBackend:
    def is_vision_paired(self):
        return False


class TestCaptionFallbackQuality:
    """Query-conditioned + multi-frame behavior of caption_via_router_fallback."""

    async def test_gates_on_router_availability_not_smolvlm(self, monkeypatch):
        import augmentum.models.base as base
        # No provider available → no-op (returns 0), images untouched path.
        state = type("S", (), {"vision_router": _FakeRouter(available=False)})()
        req = base.InternalChatRequest(
            model="m",
            messages=[base.Message(role="user", content="what is this?",
                                   images=["data:image/png;base64,Zm9v"])],
        )
        n = await base.caption_via_router_fallback(req, state, _TextOnlyBackend())
        assert n == 0

    async def test_query_conditioned_single_image(self):
        import augmentum.models.base as base
        router = _FakeRouter()
        state = type("S", (), {"vision_router": router})()
        req = base.InternalChatRequest(
            model="m",
            messages=[base.Message(role="user", content="what plant is this?",
                                   images=["data:image/png;base64,Zm9v"])],
        )
        n = await base.caption_via_router_fallback(req, state, _TextOnlyBackend())
        assert n == 1
        assert "what plant is this?" in router.calls[0]["prompt"]
        assert router.calls[0]["frames"] is None
        # Caption inlined as [Image: ...], images stripped.
        assert "[Image: a wooden chair]" in req.messages[0].content
        assert req.messages[0].images is None

    async def test_multiframe_one_clip_call(self):
        import augmentum.models.base as base
        router = _FakeRouter()
        state = type("S", (), {"vision_router": router})()
        req = base.InternalChatRequest(
            model="m",
            messages=[base.Message(role="user", content="what am I doing?",
                                   images=[
                                       "data:image/png;base64,Zm9v",
                                       "data:image/png;base64,YmFy",
                                       "data:image/png;base64,YmF6",
                                   ])],
        )
        n = await base.caption_via_router_fallback(req, state, _TextOnlyBackend())
        assert n == 3
        # ONE caption call for the whole clip, extra frames passed through.
        assert len(router.calls) == 1
        assert router.calls[0]["frames"] is not None
        assert len(router.calls[0]["frames"]) == 2
        assert "[Scene: a wooden chair]" in req.messages[0].content


class TestOllamaBackendConstruct:
    """Verify OllamaBackend can be constructed."""

    def test_construct(self):
        from augmentum.models.ollama import OllamaBackend

        client = MagicMock()
        backend = OllamaBackend(client, "http://localhost:11434")
        assert backend._base_url == "http://localhost:11434"

    def test_build_payload(self):
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.models.ollama import OllamaBackend

        backend = OllamaBackend(MagicMock(), "http://localhost:11434")
        req = InternalChatRequest(
            model="llama3:8b",
            messages=[Message(role="user", content="hi")],
            temperature=0.7,
        )
        payload = backend._build_ollama_payload(req)
        assert payload["model"] == "llama3:8b"
        assert payload["options"]["temperature"] == 0.7


class TestOpenAIBackendConstruct:
    """Verify OpenAIBackend can be constructed."""

    def test_construct_with_key(self):
        from augmentum.models.openai_compat import OpenAIBackend

        backend = OpenAIBackend(MagicMock(), "https://api.openai.com/v1", "sk-test")
        assert backend._api_key == "sk-test"

    def test_headers_include_bearer(self):
        from augmentum.models.openai_compat import OpenAIBackend

        backend = OpenAIBackend(MagicMock(), "https://api.openai.com/v1", "sk-test")
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer sk-test"

    def test_construct_without_key(self):
        from augmentum.models.openai_compat import OpenAIBackend

        backend = OpenAIBackend(MagicMock(), "http://localhost:1234/v1")
        headers = backend._headers()
        assert "Authorization" not in headers


class TestLlamaCppBackendConstruct:
    """Verify LlamaCppBackend can be constructed."""

    def test_construct(self):
        from augmentum.models.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(MagicMock(), "http://localhost:8080")
        assert backend._base_url == "http://localhost:8080"

    def test_strips_v1_suffix(self):
        from augmentum.models.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(MagicMock(), "http://localhost:8080/v1")
        assert backend._base_url == "http://localhost:8080"


class TestClaudeBackendConstruct:
    """Verify ClaudeBackend can be constructed."""

    def test_construct(self):
        from augmentum.models.adapters.claude import ClaudeBackend

        backend = ClaudeBackend(MagicMock(), "sk-ant-test")
        assert backend._api_key == "sk-ant-test"
        assert "api.anthropic.com" in backend._base_url

    def test_headers_include_api_key(self):
        from augmentum.models.adapters.claude import ClaudeBackend

        backend = ClaudeBackend(MagicMock(), "sk-ant-test")
        headers = backend._headers()
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"


class TestGeminiBackendConstruct:
    """Verify GeminiBackend can be constructed."""

    def test_construct_ai_studio(self):
        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(MagicMock(), "test-key")
        assert backend._vertex is False
        assert "googleapis.com" in backend._base_url

    def test_construct_vertex(self):
        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(
            MagicMock(), "test-key", vertex=True,
            vertex_project="my-project", vertex_region="us-east1",
        )
        assert backend._vertex is True


class TestConverters:
    """Verify all converter modules import and construct."""

    def test_import_base_converter(self):
        from augmentum.models.converters.base import PostProcessMode

        assert PostProcessMode.NONE.value == "none"

    def test_construct_claude_converter(self):
        from augmentum.models.converters.claude import ClaudeConverter

        c = ClaudeConverter()
        assert c is not None

    def test_construct_gemini_converter(self):
        from augmentum.models.converters.gemini import GeminiConverter

        c = GeminiConverter()
        assert c is not None

    def test_construct_cohere_converter(self):
        from augmentum.models.converters.cohere import CohereConverter

        c = CohereConverter()
        assert c is not None

    def test_construct_mistral_converter(self):
        from augmentum.models.converters.mistral import MistralConverter

        c = MistralConverter()
        assert c is not None

    def test_import_utils(self):
        from augmentum.models.converters.utils import ZWS

        assert ZWS == "\u200b"


class TestEngineBackendConstruct:
    """Verify AugmentumEngineBackend can be constructed."""

    def test_construct(self):
        from augmentum.models.engine import AugmentumEngineBackend

        backend = AugmentumEngineBackend(MagicMock(), "http://localhost:9000")
        assert backend._base_url == "http://localhost:9000"


class TestLoadBalancer:
    """Verify LoadBalancer and registry can be constructed."""

    def test_construct_registry(self):
        from augmentum.models.load_balancer import LoadBalancerRegistry

        reg = LoadBalancerRegistry()
        assert reg.all_balancers() == []

    def test_is_balancer_model(self):
        from augmentum.models.load_balancer import LoadBalancerRegistry

        reg = LoadBalancerRegistry()
        assert reg.is_balancer_model("lb/my-pool") is True
        assert reg.is_balancer_model("llama3:8b") is False

    def test_import_strategies(self):
        from augmentum.models.load_balancer import (
            ROUND_ROBIN,
            STRATEGIES,
        )

        assert ROUND_ROBIN in STRATEGIES


class TestModelManager:
    """Verify ModelManager can be constructed."""

    def test_construct(self):
        from augmentum.models.model_manager import ModelManager

        registry = MagicMock()
        mm = ModelManager(registry)
        assert mm._registry is registry


class TestProviderProfiles:
    """Verify provider profiles load."""

    def test_profiles_dict(self):
        from augmentum.models.provider_profiles import PROFILES

        assert "openai" in PROFILES
        assert "openrouter" in PROFILES

    def test_get_profile(self):
        from augmentum.models.provider_profiles import get_profile

        profile = get_profile("openai")
        assert profile is not None
        assert profile.base_url == "https://api.openai.com/v1"

    def test_get_unknown_profile(self):
        from augmentum.models.provider_profiles import get_profile

        assert get_profile("nonexistent") is None
