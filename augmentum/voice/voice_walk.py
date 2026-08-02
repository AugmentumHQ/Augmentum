"""Voice Walk — evolutionary voice cloning for Kokoro TTS.

Inspired by RobViren/kvoicewalk. Optimizes a Kokoro voice embedding to
match a target audio sample using speaker similarity scoring.

The approach:
  1. Extract a speaker embedding from the target audio (resemblyzer)
  2. Start from a seed embedding (best-matching stock voice or random)
  3. Iteratively mutate the embedding via CMA-ES-like strategy:
     - Generate speech with the current embedding
     - Compare speaker similarity to the target
     - Keep mutations that increase similarity
  4. Output: a numpy embedding array that can be used as a Kokoro voice

This is NOT fine-tuning — the Kokoro model is untouched. We're finding
the point in Kokoro's existing voice space that best reproduces the
target speaker's characteristics.

Typical results: ~85-93% speaker similarity after 500-2000 steps.
Runtime: 3-10 minutes on CPU depending on steps and text length.

Requires: pip install resemblyzer (speaker verification embeddings)
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Optimization defaults
_DEFAULT_STEPS = 1000
_DEFAULT_POPULATION = 8      # mutations per step
_DEFAULT_MUTATION_SCALE = 0.02  # initial mutation magnitude
_DEFAULT_DECAY = 0.999        # mutation scale decay per step
_DEFAULT_MIN_SCALE = 0.002    # minimum mutation magnitude
_DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore."
_MANIFOLD_STD_MARGIN = 0.5    # allow some exploration beyond stock voice bounds


@dataclass
class WalkProgress:
    """Progress update from the optimization loop."""
    step: int
    total_steps: int
    similarity: float
    best_similarity: float
    mutation_scale: float
    elapsed_s: float


@dataclass
class WalkResult:
    """Final result of the voice walk optimization."""
    embedding: np.ndarray
    similarity: float
    steps_taken: int
    elapsed_s: float
    seed_voice: str


@dataclass
class StyleManifold:
    """Element-wise bounds for Kokoro's known style embedding space."""

    low: np.ndarray
    high: np.ndarray


async def clone_voice_walk(
    kokoro,
    target_audio: np.ndarray,
    target_sr: int = 16000,
    *,
    steps: int = _DEFAULT_STEPS,
    population: int = _DEFAULT_POPULATION,
    seed_voice: str = "",
    text: str = _DEFAULT_TEXT,
    progress_callback: Callable[[WalkProgress], None] | None = None,
) -> WalkResult:
    """Run evolutionary voice cloning optimization.

    Args:
        kokoro: KokoroTTS instance (must be loaded)
        target_audio: mono float32 audio of the target speaker
        target_sr: sample rate of target audio (will resample to 16kHz)
        steps: number of optimization steps (more = better but slower)
        population: mutations evaluated per step
        seed_voice: starting voice name (empty = auto-select best match)
        text: reference text for generating comparison audio
        progress_callback: called every 10 steps with WalkProgress

    Returns:
        WalkResult with optimized embedding and metadata
    """
    # Load resemblyzer for speaker embeddings
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        raise RuntimeError(
            "resemblyzer is required for voice cloning. "
            "Install it: pip install resemblyzer"
        )

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("voice_walk_encoder_device", device=_device)
    encoder = await asyncio.to_thread(VoiceEncoder, _device)

    # Preprocess target audio to 16kHz mono
    if target_sr != 16000:
        target_audio = _resample(target_audio, target_sr, 16000)
    target_audio = preprocess_wav(target_audio, source_sr=16000)
    target_embed = await asyncio.to_thread(encoder.embed_utterance, target_audio)

    # Find the best seed voice (or use specified one)
    if not seed_voice:
        seed_voice = await _find_best_seed(kokoro, encoder, target_embed, text)
        log.info("voice_walk_seed", voice=seed_voice)

    # Get the seed embedding. Voice walks mutate Kokoro's style vector
    # directly; unconstrained random walks can drift outside the stock voice
    # manifold and produce intelligibility failures (skipped/repeated words)
    # that still score well on speaker similarity.
    current = kokoro._kokoro.get_voice_style(seed_voice).copy()
    manifold = _build_style_manifold(kokoro, current.shape)
    best = current.copy()
    reference_duration_s = await _generated_duration(kokoro, current, text)
    best_sim = await _evaluate(
        kokoro,
        encoder,
        current,
        target_embed,
        text,
        reference_duration_s=reference_duration_s,
    )

    t0 = time.monotonic()
    scale = _DEFAULT_MUTATION_SCALE

    for step in range(steps):
        # Generate population of mutations
        mutations = []
        for _ in range(population):
            noise = np.random.randn(*current.shape).astype(current.dtype) * scale
            candidate = _constrain_embedding(current + noise, manifold)
            mutations.append(candidate)

        # Evaluate all mutations (sequential to avoid overloading Kokoro)
        sims = []
        for candidate in mutations:
            sim = await _evaluate(
                kokoro,
                encoder,
                candidate,
                target_embed,
                text,
                reference_duration_s=reference_duration_s,
            )
            sims.append(sim)

        # Select the best mutation
        best_idx = int(np.argmax(sims))
        if sims[best_idx] > best_sim:
            best_sim = sims[best_idx]
            best = mutations[best_idx].copy()
            current = mutations[best_idx].copy()
        else:
            # No improvement — shrink search slightly toward best
            current = _constrain_embedding(current * 0.95 + best * 0.05, manifold)

        # Decay mutation scale
        scale = max(_DEFAULT_MIN_SCALE, scale * _DEFAULT_DECAY)

        # Progress reporting — every step
        if progress_callback:
            elapsed = time.monotonic() - t0
            progress_callback(WalkProgress(
                step=step,
                total_steps=steps,
                similarity=sims[best_idx],
                best_similarity=best_sim,
                mutation_scale=scale,
                elapsed_s=elapsed,
            ))

        # Early termination if similarity is very high
        if best_sim > 0.95:
            log.info("voice_walk_early_stop", step=step, similarity=best_sim)
            break

    elapsed = time.monotonic() - t0
    log.info("voice_walk_complete",
             steps=step + 1, similarity=best_sim, elapsed_s=round(elapsed, 1))

    return WalkResult(
        embedding=best,
        similarity=best_sim,
        steps_taken=step + 1,
        elapsed_s=elapsed,
        seed_voice=seed_voice,
    )


# Yield a progress update every N steps. 1500-step runs at every-step
# cadence emit ~1500 SSE messages and saturate any consumer that polls
# slowly; ~150 events/run is enough resolution for a progress bar AND
# leaves headroom for unrelated UI traffic that competes for the
# event-loop.
_PROGRESS_YIELD_INTERVAL = 10

# Cap the in-flight progress queue so a slow/blocked consumer can't
# grow it without bound. Hitting this means the consumer dropped a
# few intermediate progress updates — the FINAL result is always
# delivered regardless.
_PROGRESS_QUEUE_MAX = 64


async def clone_voice_walk_stream(
    kokoro,
    target_audio: np.ndarray,
    target_sr: int = 16000,
    **kwargs,
) -> AsyncGenerator[dict, None]:
    """Streaming version that yields progress dicts for SSE/NDJSON.

    Compartmentalization guarantees:

    * Progress yields are throttled to every ``_PROGRESS_YIELD_INTERVAL``
      steps so the SSE channel doesn't dominate the event loop on
      long runs.
    * The progress queue is bounded; a slow consumer drops intermediate
      updates rather than back-pressuring the walk.
    * If the caller's generator is closed (consumer disconnected, e.g.
      browser tab closed mid-clone), the underlying walk task is
      cancelled in the ``finally`` block so we don't keep burning GPU
      cycles on output nobody will read.
    """
    progress_queue: asyncio.Queue[WalkProgress | None] = asyncio.Queue(
        maxsize=_PROGRESS_QUEUE_MAX,
    )

    def _on_progress(p: WalkProgress):
        # Throttle: keep first + every Nth + final. The walk's main loop
        # always emits the final step regardless of throttle.
        if p.step != 0 and p.step != p.total_steps - 1 and p.step % _PROGRESS_YIELD_INTERVAL != 0:
            return
        try:
            progress_queue.put_nowait(p)
        except asyncio.QueueFull:
            # Consumer is too slow — drop this intermediate update.
            # Final result still delivers via the post-task path below.
            pass

    kwargs["progress_callback"] = _on_progress

    # Run the walk in background.
    task = asyncio.create_task(
        clone_voice_walk(kokoro, target_audio, target_sr, **kwargs)
    )

    try:
        # Yield progress updates.
        while not task.done():
            try:
                p = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield {
                    "status": "running",
                    "step": p.step,
                    "total_steps": p.total_steps,
                    "similarity": round(p.similarity, 4),
                    "best_similarity": round(p.best_similarity, 4),
                    "mutation_scale": round(p.mutation_scale, 6),
                    "elapsed_s": round(p.elapsed_s, 1),
                }
            except asyncio.TimeoutError:
                continue

        # Drain remaining progress.
        while not progress_queue.empty():
            p = progress_queue.get_nowait()
            yield {
                "status": "running",
                "step": p.step,
                "total_steps": p.total_steps,
                "similarity": round(p.similarity, 4),
                "best_similarity": round(p.best_similarity, 4),
            }

        # Final result.
        result = await task
        yield {
            "status": "complete",
            "similarity": round(result.similarity, 4),
            "steps_taken": result.steps_taken,
            "elapsed_s": round(result.elapsed_s, 1),
            "seed_voice": result.seed_voice,
            "_embedding": result.embedding,  # numpy array, consumed by route for saving
        }
    finally:
        # Generator close (consumer disconnected or completed) — cancel
        # the underlying walk if it's still running so we don't burn
        # GPU cycles on results no one will consume. asyncio.CancelledError
        # propagates into the walk's await points and unwinds cleanly;
        # the callback closures finish naturally.
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _evaluate(
    kokoro,
    encoder,
    embedding: np.ndarray,
    target_embed: np.ndarray,
    text: str,
    *,
    reference_duration_s: float = 0.0,
) -> float:
    """Generate speech with embedding and return cosine similarity to target."""
    try:
        samples, sr = await asyncio.to_thread(
            kokoro._kokoro.create,
            text,
            voice=embedding,
            speed=1.0,
            lang="en-us",
        )
        duration_s = len(samples) / sr if sr else 0.0
        duration_quality = _duration_quality(
            duration_s,
            text,
            reference_duration_s=reference_duration_s,
        )
        if duration_quality <= 0.0:
            return 0.0

        # Resample to 16kHz for resemblyzer
        if sr != 16000:
            samples = _resample(samples, sr, 16000)

        from resemblyzer import preprocess_wav
        samples = preprocess_wav(samples, source_sr=16000)
        gen_embed = await asyncio.to_thread(encoder.embed_utterance, samples)

        # Cosine similarity
        sim = float(np.dot(target_embed, gen_embed) /
                     (np.linalg.norm(target_embed) * np.linalg.norm(gen_embed) + 1e-8))
        return max(0.0, sim) * duration_quality
    except Exception as exc:
        log.debug("voice_walk_eval_error", error=str(exc))
        return 0.0


async def _find_best_seed(
    kokoro,
    encoder,
    target_embed: np.ndarray,
    text: str,
) -> str:
    """Find the stock Kokoro voice most similar to the target speaker."""
    voices = kokoro.get_voices()
    if not voices:
        return "af_heart"

    # Only check high-quality voices to avoid wasting time on D-grade seeds
    from augmentum.voice.kokoro_tts import VOICE_META, _RECOMMENDED_GRADES
    priority_voices = [v for v in voices if VOICE_META.get(v, {}).get("grade", "") in _RECOMMENDED_GRADES]
    if not priority_voices:
        priority_voices = voices[:10]

    best_voice = "af_heart"
    best_sim = 0.0

    for name in priority_voices:
        try:
            embedding = kokoro._kokoro.get_voice_style(name)
            sim = await _evaluate(kokoro, encoder, embedding, target_embed, text)
            if sim > best_sim:
                best_sim = sim
                best_voice = name
        except Exception as exc:
            log.debug("voice_walk_candidate_failed", voice=name, error=str(exc))
            continue

    log.info("voice_walk_seed_selected", voice=best_voice, similarity=round(best_sim, 3))
    return best_voice


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple linear interpolation resampling (no scipy needed)."""
    if from_sr == to_sr:
        return audio
    if len(audio) == 0 or from_sr <= 0 or to_sr <= 0:
        return audio
    ratio = to_sr / from_sr
    n_out = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_out)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
    frac = indices - idx_floor
    return audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac


async def _generated_duration(kokoro, embedding: np.ndarray, text: str) -> float:
    """Return generated duration for a reference style embedding."""
    try:
        samples, sr = await asyncio.to_thread(
            kokoro._kokoro.create,
            text,
            voice=embedding,
            speed=1.0,
            lang="en-us",
        )
        return len(samples) / sr if sr else 0.0
    except Exception as exc:
        log.debug("voice_walk_reference_duration_failed", error=str(exc))
        return 0.0


def _duration_quality(
    duration_s: float,
    text: str,
    *,
    reference_duration_s: float = 0.0,
) -> float:
    """Score whether generated duration looks plausible for the text.

    Speaker encoders care about timbre, not whether every word was spoken.
    Collapsed embeddings can skip text while keeping a high speaker score;
    unstable embeddings can loop or drag phonemes out. This lightweight
    duration gate makes those candidates less attractive without requiring
    ASR in the optimization loop.
    """
    if duration_s <= 0.0:
        return 0.0

    quality = 1.0

    if reference_duration_s > 0.0:
        ratio = duration_s / reference_duration_s
        if ratio < 0.5 or ratio > 2.0:
            quality *= 0.1
        elif ratio < 0.75:
            quality *= max(0.25, (ratio - 0.5) / 0.25)
        elif ratio > 1.4:
            quality *= max(0.25, (2.0 - ratio) / 0.6)

    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    if words:
        seconds_per_word = duration_s / len(words)
        if seconds_per_word < 0.10 or seconds_per_word > 0.75:
            quality *= 0.1
        elif seconds_per_word < 0.14:
            quality *= max(0.35, (seconds_per_word - 0.10) / 0.04)
        elif seconds_per_word > 0.55:
            quality *= max(0.35, (0.75 - seconds_per_word) / 0.20)

    return float(max(0.0, min(1.0, quality)))


def _build_style_manifold(kokoro, expected_shape: tuple[int, ...]) -> StyleManifold | None:
    """Build loose element-wise bounds from stock Kokoro voice styles."""
    try:
        styles = []
        for name in kokoro.get_voices():
            try:
                style = kokoro._kokoro.get_voice_style(name)
            except Exception as exc:
                log.debug("voice_style_read_failed", voice=name, error=str(exc))
                continue
            if getattr(style, "shape", None) == expected_shape:
                styles.append(style.astype(np.float32, copy=False))
        if len(styles) < 2:
            return None
        stacked = np.stack(styles, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        low = np.min(stacked, axis=0) - std * _MANIFOLD_STD_MARGIN
        high = np.max(stacked, axis=0) + std * _MANIFOLD_STD_MARGIN
        low = np.minimum(low, mean - np.maximum(std, 1e-4) * 4.0)
        high = np.maximum(high, mean + np.maximum(std, 1e-4) * 4.0)
        return StyleManifold(low=low.astype(np.float32), high=high.astype(np.float32))
    except Exception as exc:
        log.debug("voice_walk_manifold_unavailable", error=str(exc))
        return None


def _constrain_embedding(
    embedding: np.ndarray,
    manifold: StyleManifold | None,
) -> np.ndarray:
    """Keep a mutated style embedding finite and near the Kokoro voice space."""
    if manifold is None:
        return np.nan_to_num(embedding, copy=False)
    constrained = np.nan_to_num(
        embedding,
        nan=0.0,
        posinf=float(np.max(manifold.high)),
        neginf=float(np.min(manifold.low)),
    )
    return np.clip(constrained, manifold.low, manifold.high).astype(
        embedding.dtype,
        copy=False,
    )
