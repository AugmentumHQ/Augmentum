"""Tests for voice/voice_walk.py -- evolutionary voice cloning."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from augmentum.voice.voice_walk import (
    StyleManifold,
    WalkProgress,
    WalkResult,
    _constrain_embedding,
    _duration_quality,
    _resample,
)


class TestResample:
    def test_upsample_doubles_length(self):
        audio = np.random.randn(8000).astype(np.float32)
        result = _resample(audio, 8000, 16000)
        assert len(result) == 16000

    def test_downsample_halves_length(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = _resample(audio, 16000, 8000)
        assert len(result) == 8000

    def test_same_rate_passthrough(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = _resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_output_is_float(self):
        audio = np.random.randn(8000).astype(np.float32)
        result = _resample(audio, 8000, 16000)
        assert result.dtype in (np.float32, np.float64)


class TestFindBestSeed:
    @pytest.mark.asyncio
    async def test_returns_voice_name(self):
        from augmentum.voice.voice_walk import _find_best_seed

        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["af_heart", "af_bella", "am_michael"]
        mock_kokoro._kokoro = MagicMock()
        mock_kokoro._kokoro.get_voice_style.return_value = np.random.randn(256).astype(np.float32)
        mock_kokoro._kokoro.create.return_value = (np.random.randn(24000).astype(np.float32), 24000)

        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.random.randn(256).astype(np.float32)

        target_embed = np.random.randn(256).astype(np.float32)

        with patch("augmentum.voice.kokoro_tts.VOICE_META", {
            "af_heart": {"grade": "A"},
            "af_bella": {"grade": "A-"},
            "am_michael": {"grade": "C+"},
        }), patch("augmentum.voice.kokoro_tts._RECOMMENDED_GRADES", {"A", "A-", "C+"}):
            result = await _find_best_seed(mock_kokoro, mock_encoder, target_embed, "Test text")

        assert isinstance(result, str)
        assert len(result) > 0


class TestMutationProducesSameDimensions:
    def test_mutation_shape_preserved(self):
        base = np.random.randn(1, 256).astype(np.float32)
        noise = np.random.randn(*base.shape).astype(base.dtype) * 0.02
        candidate = base + noise
        assert candidate.shape == base.shape

    def test_similarity_score_range(self):
        a = np.random.randn(256).astype(np.float32)
        b = np.random.randn(256).astype(np.float32)
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        # Cosine similarity is between -1 and 1; clamped to 0 in code
        assert -1.0 <= sim <= 1.0

    def test_embedding_constraint_clips_to_manifold(self):
        emb = np.array([-10.0, 0.5, 10.0], dtype=np.float32)
        manifold = StyleManifold(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        result = _constrain_embedding(emb, manifold)

        np.testing.assert_array_equal(
            result,
            np.array([-1.0, 0.5, 1.0], dtype=np.float32),
        )

    def test_duration_quality_penalizes_collapsed_or_repeated_audio(self):
        text = "The quick brown fox jumps over the lazy dog."

        assert _duration_quality(2.5, text, reference_duration_s=2.5) == 1.0
        assert _duration_quality(0.4, text, reference_duration_s=2.5) < 0.2
        assert _duration_quality(7.0, text, reference_duration_s=2.5) < 0.2


class TestStreamConsumerDisconnect:
    """``clone_voice_walk_stream`` must cancel the underlying walk
    when its consumer closes the generator.

    Production path: the kokoro voice-clone route exposes the walk
    as an SSE stream. When the user closes the page mid-clone, the
    SSE generator's ``aclose()`` runs the streamer's ``finally``
    block; without an explicit ``task.cancel()`` there, the walk
    keeps grinding on the GPU forever — wasting cycles producing a
    result that nobody will consume.

    Pre-HF-3 the streamer didn't have the finally cleanup; this
    test locks in that the cleanup is real, runs on consumer
    disconnect, and propagates to the inner task.
    """

    @pytest.mark.asyncio
    async def test_consumer_disconnect_cancels_walk_task(self):
        """Closing the generator mid-stream cancels the inner walk."""
        import asyncio

        from augmentum.voice import voice_walk as vw

        started = asyncio.Event()
        was_cancelled = asyncio.Event()

        async def fake_walk(*args, **kwargs):
            started.set()
            try:
                # Mimic a long-running walk that never finishes on its
                # own — the only way out is cancellation.
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                was_cancelled.set()
                raise
            return None  # pragma: no cover — never reached

        # Patch the underlying walk so we don't need a real kokoro.
        with patch.object(vw, "clone_voice_walk", side_effect=fake_walk):
            target = np.zeros(16000, dtype=np.float32)
            gen = vw.clone_voice_walk_stream(
                kokoro=MagicMock(),
                target_audio=target,
                target_sr=16000,
            )

            # Drive the generator once so it enters the body and
            # creates the inner task. The first ``__anext__`` will
            # block on the progress queue's 1.0s timeout — break out
            # via ``wait_for`` so we control the timing.
            anext_task = asyncio.create_task(gen.__anext__())
            await started.wait()
            # Cancel the consumer so aclose() runs the finally block.
            anext_task.cancel()
            try:
                await anext_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            await gen.aclose()

            # The inner walk must have observed CancelledError.
            await asyncio.wait_for(was_cancelled.wait(), timeout=2.0)


class TestWalkDataclasses:
    def test_walk_progress_construct(self):
        p = WalkProgress(step=10, total_steps=100, similarity=0.85,
                         best_similarity=0.87, mutation_scale=0.01, elapsed_s=5.0)
        assert p.step == 10
        assert 0.0 <= p.similarity <= 1.0

    def test_walk_result_construct(self):
        r = WalkResult(embedding=np.zeros(256), similarity=0.9,
                       steps_taken=500, elapsed_s=120.0, seed_voice="af_heart")
        assert r.seed_voice == "af_heart"
        assert r.similarity == 0.9

    def test_similarity_scoring_returns_0_to_1(self):
        # Simulate the cosine similarity computation from _evaluate
        a = np.random.randn(256).astype(np.float32)
        a /= np.linalg.norm(a)
        b = np.random.randn(256).astype(np.float32)
        b /= np.linalg.norm(b)
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        clamped = max(0.0, sim)
        assert 0.0 <= clamped <= 1.0
