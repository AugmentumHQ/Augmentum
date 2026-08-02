"""Tests for shared /v command detection and direct image generation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
)
from augmentum.modes.v_command import extract_v_command, generate_direct_image

# ---------------------------------------------------------------------------
# extract_v_command tests
# ---------------------------------------------------------------------------


class TestExtractVCommand:
    def _make_request(self, *messages: tuple[str, str]) -> InternalChatRequest:
        return InternalChatRequest(
            model="test-model",
            messages=[Message(role=r, content=c) for r, c in messages],
        )

    def test_extract_v_with_instruction(self):
        req = self._make_request(("user", "/v sunset over mountains"))
        has_v, instruction, cleaned = extract_v_command(req)

        assert has_v is True
        assert instruction == "sunset over mountains"
        assert cleaned.messages[0].content == "sunset over mountains"

    def test_extract_v_no_instruction_default_fallback(self):
        req = self._make_request(("user", "/v"))
        has_v, instruction, cleaned = extract_v_command(req)

        assert has_v is True
        assert instruction == ""
        assert cleaned.messages[0].content == "Continue the scene."

    def test_extract_v_custom_fallback(self):
        req = self._make_request(("user", "/v"))
        has_v, instruction, cleaned = extract_v_command(
            req, fallback_text="Generate something."
        )

        assert has_v is True
        assert instruction == ""
        assert cleaned.messages[0].content == "Generate something."

    def test_no_v_command(self):
        req = self._make_request(("user", "Hello world"))
        has_v, instruction, cleaned = extract_v_command(req)

        assert has_v is False
        assert instruction == ""
        assert cleaned is req  # same object, unmodified

    def test_v_only_in_last_user_message(self):
        """Older /v in history should not be detected — only the latest user msg."""
        req = self._make_request(
            ("user", "/v old command"),
            ("assistant", "Sure!"),
            ("user", "Just a normal message"),
        )
        has_v, instruction, cleaned = extract_v_command(req)

        assert has_v is False
        assert instruction == ""

    def test_v_with_assistant_messages_after(self):
        """Scans backward for *user* role, skipping assistant messages."""
        req = self._make_request(
            ("user", "/v cat sitting on a rock"),
            ("assistant", "Here is a cat."),
        )
        has_v, instruction, cleaned = extract_v_command(req)

        assert has_v is True
        assert instruction == "cat sitting on a rock"
        assert cleaned.messages[0].content == "cat sitting on a rock"

    def test_cleaned_request_preserves_model(self):
        req = self._make_request(("user", "/v test"))
        _, _, cleaned = extract_v_command(req)

        assert cleaned.model == "test-model"

    def test_v_with_leading_whitespace(self):
        req = self._make_request(("user", "  /v a prompt"))
        has_v, instruction, _ = extract_v_command(req)

        assert has_v is True
        assert instruction == "a prompt"

    def test_empty_messages(self):
        req = InternalChatRequest(model="test-model", messages=[])
        has_v, instruction, cleaned = extract_v_command(req)

        assert has_v is False
        assert cleaned is req


# ---------------------------------------------------------------------------
# generate_direct_image tests
# ---------------------------------------------------------------------------


class TestGenerateDirectImage:
    @pytest.mark.asyncio
    async def test_generate_direct_image(self):
        mock_queue = AsyncMock()
        mock_queue.submit.return_value = MagicMock(
            future=asyncio.get_event_loop().create_future()
        )
        mock_queue.wait_for_result.return_value = {"image_id": "abc123"}

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.image_default_model = "sd-model"
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0

            result = await generate_direct_image(
                "sunset over mountains", mock_queue, "ses_test"
            )

        assert result == "/api/image/abc123"
        # Verify the job was submitted with correct prompt
        submitted_job = mock_queue.submit.call_args[0][0]
        assert submitted_job.prompt == "sunset over mountains"
        assert submitted_job.session_id == "ses_test"
        assert "blurry" in submitted_job.negative_prompt

    @pytest.mark.asyncio
    async def test_generate_direct_image_empty_instruction(self):
        mock_queue = AsyncMock()
        mock_queue.submit.return_value = MagicMock(
            future=asyncio.get_event_loop().create_future()
        )
        mock_queue.wait_for_result.return_value = {"image_id": "def456"}

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.image_default_model = "sd-model"
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0

            result = await generate_direct_image("", mock_queue, "ses_test")

        assert result == "/api/image/def456"
        submitted_job = mock_queue.submit.call_args[0][0]
        assert submitted_job.prompt == "a scene"

    @pytest.mark.asyncio
    async def test_generate_direct_image_failure(self):
        mock_queue = AsyncMock()
        mock_queue.submit.side_effect = RuntimeError("Queue full")

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.image_default_model = "sd-model"
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0

            result = await generate_direct_image("test", mock_queue, "ses_test")

        assert result is None


# ---------------------------------------------------------------------------
# Integration tests with handlers
# ---------------------------------------------------------------------------


class TestPassthroughVCommand:
    @pytest.mark.asyncio
    async def test_passthrough_v_command(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from tests.conftest import MockOllamaBackend

        mock_queue = AsyncMock()
        mock_queue.submit.return_value = MagicMock()
        mock_queue.wait_for_result.return_value = {"image_id": "img001"}

        handler = PassthroughHandler(
            backend=MockOllamaBackend(),
            image_queue=mock_queue,
            image_enabled=True,
            session_id="ses_test",
        )

        req = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="/v cat sitting on a rock")],
        )

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.image_default_model = "sd-model"
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0

            response = await handler.handle(req)

        assert "![Generated Image]" in response.message.content
        assert "/api/image/img001" in response.message.content

    @pytest.mark.asyncio
    async def test_passthrough_no_v(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from tests.conftest import MockOllamaBackend

        handler = PassthroughHandler(
            backend=MockOllamaBackend(),
            image_queue=AsyncMock(),
            image_enabled=True,
            session_id="ses_test",
        )

        req = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello world")],
        )
        response = await handler.handle(req)

        assert "![Generated Image]" not in response.message.content

    @pytest.mark.asyncio
    async def test_passthrough_v_stream(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        from tests.conftest import MockOllamaBackend

        mock_queue = AsyncMock()
        mock_queue.submit.return_value = MagicMock()
        mock_queue.wait_for_result.return_value = {"image_id": "img002"}

        handler = PassthroughHandler(
            backend=MockOllamaBackend(),
            image_queue=mock_queue,
            image_enabled=True,
            session_id="ses_test",
        )

        req = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="/v sunset")],
            stream=True,
        )

        chunks = []
        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.image_default_model = "sd-model"
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0

            async for chunk in handler.handle_stream(req):
                chunks.append(chunk)

        # Last non-done chunk should be the image
        all_content = "".join(c.content_delta for c in chunks)
        assert "![Generated Image](/api/image/img002)" in all_content


class TestAnalyticalVCommand:
    @pytest.mark.asyncio
    async def test_analytical_v_handle(self):
        from augmentum.modes.analytical.handler import AnalyticalHandler

        mock_queue = AsyncMock()
        mock_queue.submit.return_value = MagicMock()
        mock_queue.wait_for_result.return_value = {"image_id": "img003"}

        # Mock the engine.process to avoid full UARF pipeline
        mock_backend = AsyncMock()
        mock_backend.chat.return_value = InternalChatResponse(
            message=Message(role="assistant", content="Analysis result"),
            model="test-model",
            finish_reason="stop",
        )

        handler = AnalyticalHandler(
            backend=mock_backend,
            image_queue=mock_queue,
            image_enabled=True,
            session_id="ses_test",
        )

        req = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="/v a dragon")],
        )

        with (
            patch("augmentum.config.settings") as mock_settings,
            patch.object(handler, "handle", wraps=handler.handle) as _,
            patch(
                "augmentum.modes.analytical.handler.AnalyticalEngine"
            ) as MockEngine,
        ):
            mock_settings.image_default_model = "sd-model"
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0

            mock_engine_inst = MockEngine.return_value
            mock_result = MagicMock()
            mock_result.conclusion = "Here is the analysis."
            mock_result.total_tokens = 100
            mock_engine_inst.process = AsyncMock(return_value=mock_result)

            response = await handler.handle(req)

        assert "![Generated Image](/api/image/img003)" in response.message.content
        assert "Here is the analysis." in response.message.content


class TestImageToolRemoved:
    def test_image_not_in_apply_phase(self):
        from augmentum.tools.base import ToolCategory
        from augmentum.tools.registry import _PHASE_CATEGORIES

        assert ToolCategory.IMAGE not in _PHASE_CATEGORIES["apply"]
