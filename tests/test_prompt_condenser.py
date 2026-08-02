"""Tests for image prompt condensation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from augmentum.image.prompt_condenser import (
    condense_prompt,
    detect_token_limit,
    estimate_tokens,
    needs_condensing,
)

# =====================================================================
# Token limit detection
# =====================================================================


class TestDetectTokenLimit:
    def test_clip_tokenizer_limit(self):
        """Should detect CLIP's 77 token limit from tokenizer config."""
        pipe = MagicMock()
        pipe.tokenizer = MagicMock()
        pipe.tokenizer.model_max_length = 77
        del pipe.tokenizer_2
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 75  # 77 - 2 for BOS/EOS

    def test_t5_tokenizer_limit(self):
        """Should detect T5's 512 token limit from tokenizer config."""
        pipe = MagicMock()
        pipe.tokenizer = MagicMock()
        pipe.tokenizer.model_max_length = 512
        del pipe.tokenizer_2
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 510

    def test_sdxl_uses_minimum_of_two_tokenizers(self):
        """SDXL has two tokenizers — should use the shorter limit."""
        pipe = MagicMock()
        pipe.tokenizer = MagicMock()
        pipe.tokenizer.model_max_length = 77
        pipe.tokenizer_2 = MagicMock()
        pipe.tokenizer_2.model_max_length = 77
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 75

    def test_mixed_tokenizer_limits_uses_minimum(self):
        """When tokenizers have different limits, use the shortest."""
        pipe = MagicMock()
        pipe.tokenizer = MagicMock()
        pipe.tokenizer.model_max_length = 512  # T5
        pipe.tokenizer_2 = MagicMock()
        pipe.tokenizer_2.model_max_length = 77  # CLIP
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 75  # min(510, 75)

    def test_fallback_to_default(self):
        """Should default to 75 when no tokenizer or encoder detected."""
        pipe = MagicMock()
        del pipe.tokenizer
        del pipe.tokenizer_2
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 75

    def test_ignores_sentinel_max_length(self):
        """Tokenizers with model_max_length=1e30 (sentinels) should be ignored."""
        pipe = MagicMock()
        pipe.tokenizer = MagicMock()
        pipe.tokenizer.model_max_length = int(1e30)
        del pipe.tokenizer_2
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 75

    def test_detects_from_encoder_max_position_embeddings(self):
        """Should read max_position_embeddings from encoder config."""
        pipe = MagicMock()
        del pipe.tokenizer
        del pipe.tokenizer_2
        del pipe.tokenizer_3

        encoder = MagicMock()
        encoder.config = MagicMock()
        encoder.config.max_position_embeddings = 512
        pipe.text_encoder = encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 510

    def test_uses_tokenizer_2_if_primary_missing(self):
        """Should check tokenizer_2 (SDXL has two tokenizers)."""
        pipe = MagicMock()
        del pipe.tokenizer
        pipe.tokenizer_2 = MagicMock()
        pipe.tokenizer_2.model_max_length = 77
        del pipe.tokenizer_3
        del pipe.text_encoder
        del pipe.text_encoder_2
        del pipe.text_encoder_3

        assert detect_token_limit(pipe) == 75


# =====================================================================
# Token estimation
# =====================================================================


class TestTokenEstimation:
    def test_short_prompt(self):
        assert estimate_tokens("a cat") == 1

    def test_medium_prompt(self):
        # 100 chars ≈ 25 tokens
        prompt = "a" * 100
        assert estimate_tokens(prompt) == 25

    def test_empty_prompt(self):
        assert estimate_tokens("") == 1  # min 1


# =====================================================================
# Needs condensing check
# =====================================================================


class TestNeedsCondensing:
    def test_short_prompt_no_condense(self):
        assert not needs_condensing("a beautiful sunset", 75)

    def test_long_prompt_needs_condense(self):
        # 500 chars ≈ 125 tokens, over CLIP's 75 limit
        long_prompt = "masterpiece, best quality, " * 20
        assert needs_condensing(long_prompt, 75)

    def test_prompt_at_limit(self):
        # Exactly at limit should not condense
        prompt = "a" * (75 * 4)  # 75 tokens
        assert not needs_condensing(prompt, 75)

    def test_long_prompt_fine_for_t5(self):
        # Same prompt, but T5 has 512 limit
        long_prompt = "masterpiece, best quality, " * 20  # ~135 tokens
        assert not needs_condensing(long_prompt, 500)


# =====================================================================
# Prompt condensation
# =====================================================================


class TestCondensePrompt:
    def test_condense_returns_shortened_prompt(self):
        from augmentum.models.base import InternalChatResponse, Message

        backend = MagicMock()
        condensed = "elven woman, silver hair, forest, dragon companion, fantasy concept art"
        backend.chat = AsyncMock(
            return_value=InternalChatResponse(
                message=Message(role="assistant", content=condensed),
                model="test",
            )
        )

        long_prompt = (
            "Create an evocative image capturing a mystical forest scene "
            "with an elven woman named Kaelani who has long silver hair "
            "and a small dragon-like creature on her shoulder..."
        )

        result = asyncio.get_event_loop().run_until_complete(
            condense_prompt(long_prompt, 75, backend)
        )

        assert result == condensed
        backend.chat.assert_called_once()

    def test_condense_preserves_original_on_failure(self):
        backend = MagicMock()
        backend.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        original = "a very long prompt " * 50
        result = asyncio.get_event_loop().run_until_complete(
            condense_prompt(original, 75, backend)
        )

        assert result == original

    def test_condense_preserves_original_on_empty_response(self):
        from augmentum.models.base import InternalChatResponse, Message

        backend = MagicMock()
        backend.chat = AsyncMock(
            return_value=InternalChatResponse(
                message=Message(role="assistant", content=""),
                model="test",
            )
        )

        original = "a very long prompt " * 50
        result = asyncio.get_event_loop().run_until_complete(
            condense_prompt(original, 75, backend)
        )

        assert result == original

    def test_condense_passes_correct_model(self):
        from augmentum.models.base import InternalChatResponse, Message

        backend = MagicMock()
        backend.chat = AsyncMock(
            return_value=InternalChatResponse(
                message=Message(role="assistant", content="condensed"),
                model="test",
            )
        )

        asyncio.get_event_loop().run_until_complete(
            condense_prompt("long prompt", 75, backend, model="my-model")
        )

        call_args = backend.chat.call_args[0][0]
        assert call_args.model == "my-model"

    def test_condense_system_prompt_includes_limits(self):
        from augmentum.models.base import InternalChatResponse, Message

        backend = MagicMock()
        backend.chat = AsyncMock(
            return_value=InternalChatResponse(
                message=Message(role="assistant", content="condensed"),
                model="test",
            )
        )

        asyncio.get_event_loop().run_until_complete(
            condense_prompt("long prompt", 75, backend)
        )

        call_args = backend.chat.call_args[0][0]
        system_msg = call_args.messages[0].content
        assert "75" in system_msg  # token limit
        assert "300" in system_msg  # char limit (75 * 4)


# ---------------------------------------------------------------------------
# derive_image_capabilities — job-aware model tagging
# ---------------------------------------------------------------------------


class TestDeriveImageCapabilities:
    """Coarse capability tags derived from the family classifier.

    These drive the agentic build flow's choice between a real-photo search,
    a synthetic diagram, and a stylised illustration — and, critically, keep a
    stylised (anime) model from being picked for a real-world how-to photo.
    """

    def _cap(self, name):
        from augmentum.image.prompt_condenser import derive_image_capabilities

        return derive_image_capabilities(name)

    def test_anime_models_are_stylized_not_photoreal(self):
        # The reported failure: Lumina (anime default) used for a tire guide.
        for name in ("lumina2", "neta-lumina", "pony-diffusion-v6", "animagine-xl"):
            cap = self._cap(name)
            assert cap["stylized"] is True, name
            assert cap["photoreal"] is False, name
            assert cap["diagram"] is False, name

    def test_flux_is_photoreal_and_diagram_capable(self):
        cap = self._cap("flux.1-schnell")
        assert cap["photoreal"] is True
        assert cap["diagram"] is True
        assert cap["stylized"] is False

    def test_realistic_finetunes_are_photoreal(self):
        assert self._cap("realistic-vision-v5")["photoreal"] is True
        assert self._cap("juggernaut-xl")["photoreal"] is True

    def test_bare_sdxl_does_diagrams_but_not_assumed_photoreal(self):
        cap = self._cap("sdxl-base-1.0")
        assert cap["diagram"] is True
        # A bare base isn't assumed to be a believable-photo model.
        assert cap["photoreal"] is False

    def test_anime_sdxl_finetune_is_not_a_diagram_pick(self):
        cap = self._cap("AnythingXL")
        assert cap["stylized"] is True
        assert cap["diagram"] is False

    def test_cloud_api_is_photoreal(self):
        assert self._cap("dall-e-3")["photoreal"] is True

    def test_unknown_model_is_conservative(self):
        cap = self._cap("somerandommodel")
        assert cap["photoreal"] is False
        assert cap["stylized"] is False

    def test_summary_is_human_readable(self):
        assert "photographs" in self._cap("flux.1-schnell")["summary"]
        assert "anime" in self._cap("lumina2")["summary"]
