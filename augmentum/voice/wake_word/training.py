"""Wake-word training: Kokoro-synthesized data → small CRNN → ONNX.

The pipeline is self-contained: no external dataset downloads, no
network access beyond what Kokoro needs (model already cached). For
each target phrase we synthesize positive samples across Kokoro
voices + speeds, mix them with augmented common-phrase negatives,
and train a small mel-spectrogram CRNN. The exported ONNX model is
~200 KB and runs in <5ms on CPU per 1-second window.

Quality bar for the first cut: false-accept rate <1/hour in quiet
conditions, false-reject rate <10% on the trained user's voice. A
real-world deployment with users beyond the operator would want to
retrain with their voices added as positives, OR pull a public
non-trigger corpus (LibriSpeech) for richer negatives. Both are
extensions the contract already supports — just point ``negatives_dir``
at audio files instead of relying on synthesis.

The training entrypoint is ``train_wake_word_model`` — designed to
be called from the JobRunner handler with a :class:`JobContext` for
progress reporting and cancellation.
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from augmentum.utils.logging import get_logger
from augmentum.voice.wake_word import negatives_corpus

if TYPE_CHECKING:
    from augmentum.jobs.context import JobContext

log = get_logger(__name__)


# ── Audio constants ──────────────────────────────────────────────────

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
N_MELS = 40
N_FFT = 400          # 25 ms @ 16 kHz
HOP_LENGTH = 160     # 10 ms @ 16 kHz — gives 101 frames per 1 s window


# ── Synthesis config ─────────────────────────────────────────────────

# Voices spanning Kokoro's American/British × F/M space. Adding more
# voices is cheap (one row each) and improves model robustness.
_DEFAULT_VOICES = [
    "af_heart", "af_nicole", "af_sky",
    "am_michael", "am_adam",
    "bf_emma", "bf_isabella",
]

# Common short phrases used as negative samples — chosen so they don't
# rhyme with or alliterate to typical wake words. Synthesized via the
# same Kokoro voices so distributional artifacts match the positives.
_NEGATIVE_PHRASES = [
    "hello there", "good morning", "the weather", "what time is it",
    "open the door", "turn on the light", "play some music",
    "set a timer", "tell me a story", "how are you", "thank you",
    "see you later", "where is it", "computer", "assistant",
    "alright then", "go ahead", "never mind", "wait a moment",
    "right now", "later today", "this evening", "yesterday",
    "tomorrow morning", "next week", "show me", "find a recipe",
    "book a flight", "call my mom", "send a message", "what's the news",
    "remind me", "schedule a meeting", "cancel that", "try again",
    "one more time", "the quick brown fox", "lorem ipsum",
    "good evening", "good night", "see ya", "of course",
    "absolutely", "indeed it is", "right away", "in a minute",
    "later tonight", "soon enough",
]

# Per-voice generation budget (multiplied by speeds for total per voice).
# Tuned for usable val_acc: the initial 8/12 budget produced models that
# only matched majority-class baseline (~60%). 24/50 brings the budget
# into the range where the model actually learns a discriminator.
# With 7 voices × 3 speeds: positives = 504, negatives = 1050.
POSITIVES_PER_VOICE_PER_SPEED = 24
NEGATIVES_PER_VOICE_PER_SPEED = 50

# How many copies of each personal recording to add to the positives
# pool. The unsampled default (1×) drowns 21 personal recordings in
# ~500 Kokoro positives — gradient signal from personal voice is <5 %
# of the per-epoch budget, and the resulting model has high FRR on the
# user's real voice (eval harness 2026-05-28 measured 35 % FRR on real
# recordings for v5 of bundled_f_becca with 21 personal samples included
# at 1×).
#
# Each copy gets independent random augmentation via ``_augment`` in
# ``_WakeWordDataset.__getitem__`` (gain ±6 dB, ±50 ms shift, noise
# floor -45..-30 dBFS), so 5 copies of one recording train as 5 distinct
# datapoints, not 5 identical ones. The upper limit is "don't overwhelm
# the Kokoro positives that provide cross-speaker generalization" —
# at 5× a typical 20-sample personal pool contributes ~17 % of the
# positive budget, roughly matching one Kokoro voice's contribution.
PERSONAL_OVERSAMPLE_FACTOR = 5

# Three speed variants per voice — covers natural rate variation.
_SPEEDS = (0.85, 1.0, 1.15)


# ── Phrase-derived hard negatives ────────────────────────────────────


def _derive_hard_negatives(phrase: str) -> list[str]:
    """Derive negatives that share structure with the wake phrase.

    The default ``_NEGATIVE_PHRASES`` are generic conversational filler
    that doesn't share words with the wake word at all. Without contrastive
    examples like "the substantive word said alone" or "the same word with
    a different prefix", the model learns the substring of the phrase as
    the discriminative feature instead of the full phrase.

    Concretely for ``phrase="hey becca"`` (the canonical case the eval
    harness caught firing at 70% on bare "becca"): this generator yields
    ``["becca", "hey rebecca", "okay becca", ...]`` so the model sees
    those constructions labeled negative during training.

    The output is deterministic and capped at a small set so it doesn't
    drown out the generic negatives — the cycling shuffle in the train
    loop interleaves both pools.
    """
    words = phrase.lower().split()
    if not words:
        return []

    hard: set[str] = set()

    # Each substantive word standalone (≥3 chars to skip "a", "of", etc.).
    # This is the most important class — it directly teaches "the keyword
    # alone is not a wake".
    for w in words:
        if len(w) >= 3:
            hard.add(w)

    # Conversational prefixes attached to each substantive word. Teaches
    # that "okay becca" / "umm becca" should not trigger even though
    # they contain the keyword.
    prefixes = ("okay", "ok", "umm", "well", "yes", "yeah", "no", "and", "so")
    for w in words:
        if len(w) >= 3:
            for pre in prefixes:
                hard.add(f"{pre} {w}")

    # Conversational suffixes — same idea, trailing context.
    suffixes = ("please", "right", "now", "though")
    for w in words:
        if len(w) >= 3:
            for suf in suffixes:
                hard.add(f"{w} {suf}")

    # Reversed-order phrase. Teaches that word order matters: "becca hey"
    # contains both words but isn't the wake.
    if len(words) > 1:
        hard.add(" ".join(reversed(words)))

    # If the original phrase happens to be one of the derivations, drop it.
    hard.discard(phrase.lower())
    return sorted(hard)


# ── Helpers ──────────────────────────────────────────────────────────

def _decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to mono float32 samples at 16 kHz."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sr != SAMPLE_RATE:
        # Use torchaudio's resampler for accurate band-limited resampling.
        t = torch.from_numpy(samples).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        samples = t.squeeze(0).numpy()

    return samples.astype(np.float32)


def _pad_or_clip(samples: np.ndarray, length: int) -> np.ndarray:
    """Center-pad or center-clip samples to ``length``."""
    n = len(samples)
    if n == length:
        return samples
    if n < length:
        pad = length - n
        before = pad // 2
        after = pad - before
        return np.pad(samples, (before, after), mode="constant")
    # Clip — center
    start = (n - length) // 2
    return samples[start:start + length]


def _augment(samples: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply random gain + small temporal shift + noise injection.

    Pitch shift / time stretch are skipped here for speed; the Kokoro
    speed variants already give us prosodic diversity. This pass adds
    acoustic robustness (gain, shift, noise floor).
    """
    out = samples.copy()

    # Random gain ±6 dB
    gain_db = rng.uniform(-6.0, 6.0)
    out = out * (10.0 ** (gain_db / 20.0))

    # Random circular-shift up to ±50 ms
    max_shift = SAMPLE_RATE // 20
    shift = rng.randint(-max_shift, max_shift)
    if shift != 0:
        out = np.roll(out, shift)

    # White-noise floor at -30 to -45 dBFS
    noise_db = rng.uniform(-45.0, -30.0)
    noise_amp = 10.0 ** (noise_db / 20.0)
    out = out + np.random.normal(0.0, noise_amp, size=out.shape).astype(np.float32)

    # Clip to [-1, 1]
    return np.clip(out, -1.0, 1.0)


# ── Synthesis (Kokoro → 16 kHz mono float32 windows) ─────────────────

async def synthesize_samples(
    kokoro: Any,
    phrase: str,
    voices: list[str],
    speeds: tuple[float, ...],
    samples_per_voice_per_speed: int,
    ctx: "JobContext | None" = None,
    label_prefix: str = "synth",
) -> list[np.ndarray]:
    """Generate raw 1-second 16 kHz audio windows for a phrase.

    Multiple variants come from voice × speed × Kokoro's internal
    sampling randomness. The augmentation pass at training time adds
    further diversity per sample.
    """
    results: list[np.ndarray] = []
    total = len(voices) * len(speeds) * samples_per_voice_per_speed
    done = 0

    for voice in voices:
        for speed in speeds:
            for _ in range(samples_per_voice_per_speed):
                if ctx is not None:
                    await ctx.check_cancel()
                try:
                    wav_bytes = await kokoro.generate(
                        phrase, voice=voice, speed=speed, response_format="wav",
                    )
                except Exception as exc:
                    log.warning("wake_word_synth_failed",
                                phrase=phrase, voice=voice, error=str(exc))
                    continue
                if not wav_bytes:
                    continue
                samples = _decode_wav_bytes(wav_bytes)
                samples = _pad_or_clip(samples, WINDOW_SAMPLES)
                results.append(samples)
                done += 1
                if ctx is not None and done % 25 == 0:
                    progress = done / total if total else 1.0
                    await ctx.update_progress(progress, stage=label_prefix)
                # Yield to the event loop so the curl-based healthcheck
                # gets a turn between TTS rounds. Without this, 500+ back-
                # to-back kokoro.generate() calls starve the main loop
                # long enough for autoheal to SIGTERM the container —
                # see [[kokoro-synth-autoheal-trip]] in memory.
                await asyncio.sleep(0)

    log.info("wake_word_synth_complete", phrase=phrase, count=len(results), target=total)
    return results


# ── Dataset ──────────────────────────────────────────────────────────

class _WakeWordDataset(Dataset):
    """In-memory dataset of (audio_window, label) pairs.

    Augmentation runs each ``__getitem__`` so every epoch sees fresh
    perturbations of the same base samples — implicit oversampling.
    """

    def __init__(
        self,
        positives: list[np.ndarray],
        negatives: list[np.ndarray],
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self._items: list[tuple[np.ndarray, int]] = (
            [(s, 1) for s in positives] + [(s, 0) for s in negatives]
        )
        self._augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        samples, label = self._items[idx]
        if self._augment:
            samples = _augment(samples, self._rng)
        return torch.from_numpy(samples.astype(np.float32)), torch.tensor(label, dtype=torch.float32)


# ── Model: small CRNN over mel-spectrogram ───────────────────────────

class MelFrontend(nn.Module):
    """Log-mel spectrogram extraction. Pure feature engineering, no
    learnable parameters. Lives outside the exported model graph
    because torchaudio's MelSpectrogram uses STFT with complex tensors
    that ONNX cannot serialize. Inference-time mel computation happens
    in Python before feeding the ONNX model.
    """

    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=n_mels, power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, samples) → (B, 1, n_mels, T) log-mel
        mel = self.mel(audio)
        mel = self.amplitude_to_db(mel)
        return mel.unsqueeze(1)


class WakeWordCRNN(nn.Module):
    """2 conv blocks → GRU → sigmoid, over a precomputed log-mel.

    Input: (B, 1, n_mels, n_frames) log-mel features.
    Output: (B, 1) probability that the trigger phrase was said.

    The mel transform lives in :class:`MelFrontend` (not a submodule of
    this class) so this graph stays ONNX-exportable. Training pipeline
    composes ``crnn(mel_frontend(audio))``; inference computes mel in
    Python before feeding the ONNX session.
    """

    def __init__(self, n_mels: int = N_MELS, hidden: int = 64) -> None:
        super().__init__()
        # 2D conv stack — F dim shrinks, T dim preserved (no pooling on T).
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))   # halve F, keep T
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))   # halve F again

        # After two pools the freq dim is n_mels // 4 = 10. Flatten across
        # freq+channel = 320 per timestep, feed to GRU.
        flat_feats = 32 * (n_mels // 4)
        self.gru = nn.GRU(input_size=flat_feats, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, n_mels, T)
        x = F.relu(self.bn1(self.conv1(mel)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # x: (B, 32, n_mels/4, T) → (B, T, 32 * n_mels/4)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.gru(x)
        # Take last timestep — full window has been observed by then.
        logits = self.head(out[:, -1, :])   # (B, 1)
        return logits


# ── Train loop ───────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Output of a training run, persisted as ``train_metrics`` JSON."""
    epochs_run: int
    best_val_loss: float
    best_val_acc: float
    final_train_loss: float
    positives_count: int
    negatives_count: int
    device: str
    # Per-wake recommended inference threshold (sigmoid cutoff). Swept
    # against the validation set after training to maximize F1. The
    # widget-side detector (Slice 2) reads this instead of defaulting
    # to 0.5 — typically cuts false-accepts dramatically.
    best_threshold: float = 0.5
    val_precision_at_best: float = 0.0
    val_recall_at_best: float = 0.0
    val_f1_at_best: float = 0.0
    # Which negatives pipeline produced this model. ``real_audio`` means
    # the LibriSpeech corpus was installed and mixed into the negatives
    # pool; ``synthetic_only`` means Kokoro-only (the legacy degraded
    # path). load_models_from_db inspects this to decide how strictly to
    # gate the runtime threshold floor.
    negatives_pipeline: str = "synthetic_only"
    # How many personal voice recordings the user contributed to this
    # bake. >0 means the model was anchored on the actual speaker, not
    # only on synthetic Kokoro voices — typically a large quality win
    # for that specific user. UI surfaces this so the user can see the
    # difference a re-bake with their voice made.
    personal_samples_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


def _sweep_threshold(model: nn.Module, frontend: nn.Module, val_loader, device):
    """After training, find the sigmoid threshold that maximizes F1 on val.
    Returns (best_threshold, precision, recall, f1)."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for audio, labels in val_loader:
            audio = audio.to(device)
            mel = frontend(audio)
            logits = model(mel)
            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
    if not all_probs:
        return 0.5, 0.0, 0.0, 0.0
    best = (0.5, 0.0, 0.0, 0.0)  # threshold, p, r, f1
    # Sweep 0.25..0.85 in 0.025 steps — fine enough for tuning, cheap.
    t = 0.25
    while t <= 0.85 + 1e-6:
        tp = fp = fn = 0
        for p, y in zip(all_probs, all_labels):
            pred = 1 if p >= t else 0
            if pred == 1 and y >= 0.5: tp += 1
            elif pred == 1 and y < 0.5: fp += 1
            elif pred == 0 and y >= 0.5: fn += 1
        if tp + fp == 0 or tp + fn == 0:
            t += 0.025
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best[3]:
            best = (round(t, 3), round(precision, 3), round(recall, 3), round(f1, 3))
        t += 0.025
    return best


def _load_personal_positives(samples_dir: Path) -> list[np.ndarray]:
    """Load every WAV under ``samples_dir`` as 1-second 16 kHz mono windows.

    Files are produced by ``POST /api/wake_word/personal_samples`` and are
    already normalized to 16 kHz mono PCM16 + length-validated to 0.5-2s,
    so the load path is small. We center-pad/clip each one to the canonical
    16,000-sample window the CRNN expects.

    Missing or malformed files are skipped with a warning rather than
    aborting the whole bake — one bad recording shouldn't lose 14 minutes
    of training work.
    """
    if not samples_dir.exists():
        return []
    out: list[np.ndarray] = []
    for wav_path in sorted(samples_dir.glob("*.wav")):
        try:
            samples = _decode_wav_bytes(wav_path.read_bytes())
            samples = _pad_or_clip(samples, WINDOW_SAMPLES)
            out.append(samples)
        except Exception as exc:
            log.warning(
                "wake_word_personal_sample_skipped",
                path=str(wav_path), error=str(exc),
            )
    return out


async def train_wake_word_model(
    phrase: str,
    output_path: Path,
    kokoro: Any,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    voices: list[str] | None = None,
    personal_samples_dir: Path | None = None,
    ctx: "JobContext | None" = None,
) -> TrainResult:
    """Train a wake-word model for ``phrase`` and export ONNX.

    Pipeline: synth positives → (optional) load personal recordings →
    synth negatives → build dataset → train → ONNX export. The handler
    caller is responsible for creating ``output_path``'s parent directory.

    When ``personal_samples_dir`` is provided, the user's recordings of
    the wake phrase are loaded and added to the positives pool. This is
    the structural fix for the FRR-on-out-of-training-voices problem —
    one real recording of the actual user's voice anchors the model
    far better than another synthetic Kokoro voice would.

    Returns the final :class:`TrainResult` for storage in
    ``wake_word_models.train_metrics``.
    """
    voices = voices or _DEFAULT_VOICES
    notes: list[str] = []

    if ctx is not None:
        await ctx.update_progress(0.0, stage="synth_positives")

    log.info("wake_word_train_starting", phrase=phrase, voices=voices, epochs=epochs)

    # Positive samples — repeat the same phrase across all voices/speeds.
    positives = await synthesize_samples(
        kokoro=kokoro, phrase=phrase, voices=voices, speeds=_SPEEDS,
        samples_per_voice_per_speed=POSITIVES_PER_VOICE_PER_SPEED,
        ctx=ctx, label_prefix="synth_positives",
    )
    if len(positives) < 20:
        notes.append(f"low positive count: {len(positives)} — model quality will suffer")

    # Personal voice recordings (optional). Loaded BEFORE negatives so
    # the dataset class-weighting + split arithmetic below sees the full
    # positive pool.
    #
    # Each personal recording is replicated ``PERSONAL_OVERSAMPLE_FACTOR``
    # times before being added to the positives pool. Independent random
    # augmentation in ``_WakeWordDataset.__getitem__`` makes each copy a
    # distinct datapoint. Without oversampling, ~20 personal recordings
    # contribute <5 % of gradient signal vs. ~500 Kokoro positives and
    # the model fails to anchor on the real user's voice. See
    # PERSONAL_OVERSAMPLE_FACTOR for the bake-cost vs. anchor-quality
    # trade-off.
    personal_count = 0
    if personal_samples_dir is not None:
        if ctx is not None:
            await ctx.update_progress(0.33, stage="loading personal recordings")
        personal_samples = _load_personal_positives(personal_samples_dir)
        personal_count = len(personal_samples)
        if personal_samples:
            oversampled = personal_samples * PERSONAL_OVERSAMPLE_FACTOR
            positives.extend(oversampled)
            log.info(
                "wake_word_train_personal_loaded",
                count=personal_count,
                oversample_factor=PERSONAL_OVERSAMPLE_FACTOR,
                effective_count=len(oversampled),
                positives_total_after=len(positives),
            )
            notes.append(
                f"personal recordings: {personal_count} × "
                f"{PERSONAL_OVERSAMPLE_FACTOR}× oversample = "
                f"{len(oversampled)} effective (positives now "
                f"{len(positives)})"
            )

    # Negative samples. Two pipelines depending on what's installed:
    #
    #   Real-audio pipeline (LibriSpeech corpus present):
    #     ~315 Kokoro phrases  — keeps TTS-distribution coverage so the
    #                             model still learns to reject "phrases
    #                             spoken by the same voice family as the
    #                             positives but with different words"
    #     ~1000 LibriSpeech crops — real human speech in real acoustics,
    #                             the dominant fix for self-trigger +
    #                             room-tone false positives
    #     ~200 synthetic silence/noise — quiet-room class that the
    #                             synthetic-only path never sees
    #
    #   Synthetic-only fallback (no corpus): the original 1050-window
    #     Kokoro-phrases-only pool, preserved as a graceful degrade for
    #     dev installs where the operator hasn't downloaded the corpus.
    use_real_audio_negatives = negatives_corpus.is_installed()
    if use_real_audio_negatives:
        kokoro_per_vs = 15        # 7 voices × 3 speeds × 15 = 315
        real_speech_target = 1000
        silence_target = 200
        log.info(
            "wake_word_train_negatives_pipeline",
            mode="real_audio",
            kokoro_per_voice_speed=kokoro_per_vs,
            real_speech_target=real_speech_target,
            silence_target=silence_target,
        )
    else:
        kokoro_per_vs = NEGATIVES_PER_VOICE_PER_SPEED  # 50 → 1050 total
        real_speech_target = 0
        silence_target = 0
        log.info(
            "wake_word_train_negatives_pipeline",
            mode="synthetic_only_fallback",
            note=(
                "LibriSpeech corpus not installed — install via "
                "wake_word_corpus_download job for higher quality"
            ),
        )

    if ctx is not None:
        await ctx.update_progress(0.35, stage="synth_negatives")

    negatives: list[np.ndarray] = []
    neg_target = kokoro_per_vs * len(voices) * len(_SPEEDS)
    rng = random.Random(0)
    # Mix in phrase-derived hard negatives — see _derive_hard_negatives
    # docstring for why. The shuffle interleaves them with the generic
    # filler so the model sees the full mix across the training loop.
    hard_neg_phrases = _derive_hard_negatives(phrase)
    neg_phrases = list(_NEGATIVE_PHRASES) + hard_neg_phrases
    rng.shuffle(neg_phrases)
    log.info(
        "wake_word_train_hard_negatives_derived",
        phrase=phrase,
        hard_count=len(hard_neg_phrases),
        generic_count=len(_NEGATIVE_PHRASES),
        examples=hard_neg_phrases[:6],
    )
    pi = 0
    total_done = 0
    for voice in voices:
        for speed in _SPEEDS:
            for _ in range(kokoro_per_vs):
                if ctx is not None:
                    await ctx.check_cancel()
                p = neg_phrases[pi % len(neg_phrases)]
                pi += 1
                try:
                    wav_bytes = await kokoro.generate(
                        p, voice=voice, speed=speed, response_format="wav",
                    )
                except Exception as exc:
                    log.debug("wake_word_neg_gen_failed", voice=voice, speed=speed, error=str(exc))
                    continue
                if not wav_bytes:
                    continue
                samples = _decode_wav_bytes(wav_bytes)
                samples = _pad_or_clip(samples, WINDOW_SAMPLES)
                negatives.append(samples)
                total_done += 1
                if total_done % 50 == 0 and ctx is not None:
                    base = 0.35
                    p_done = total_done / neg_target if neg_target else 1.0
                    # Reserve 0.35-0.55 for kokoro synthesis when the real-
                    # audio pipeline is in play (sampling takes another 0.05).
                    range_hi = 0.55 if use_real_audio_negatives else 0.60
                    await ctx.update_progress(
                        base + p_done * (range_hi - base), stage="synth_negatives",
                    )
                # Yield to event loop — see synthesize_samples for why.
                await asyncio.sleep(0)

    if use_real_audio_negatives:
        # FLAC decode happens off-loop — even a few hundred files would
        # otherwise serialize behind the event loop and starve chat/voice.
        if ctx is not None:
            await ctx.check_cancel()
            await ctx.update_progress(0.56, stage="sampling LibriSpeech")
        try:
            real_speech = await asyncio.to_thread(
                negatives_corpus.sample_real_speech_windows,
                real_speech_target, random.Random(42),
            )
            negatives.extend(real_speech)
        except Exception as exc:
            log.warning(
                "wake_word_real_speech_sampling_failed",
                error=str(exc),
                note="continuing with kokoro+silence negatives only",
            )
            real_speech = []

        # Silence/noise synthesis is in-process numpy, no I/O — cheap.
        if ctx is not None:
            await ctx.check_cancel()
            await ctx.update_progress(0.58, stage="synthesizing silence negatives")
        silence_windows = negatives_corpus.sample_silence_windows(
            silence_target, random.Random(43),
        )
        negatives.extend(silence_windows)

        notes.append(
            f"negatives pool: kokoro={total_done} real_speech={len(real_speech)} "
            f"silence={len(silence_windows)} total={len(negatives)}"
        )

    log.info("wake_word_synth_summary",
             positives=len(positives), negatives=len(negatives),
             pipeline=("real_audio" if use_real_audio_negatives else "synthetic_only"))

    if not positives or not negatives:
        raise RuntimeError(
            f"insufficient training data: positives={len(positives)} "
            f"negatives={len(negatives)} — Kokoro likely unavailable"
        )

    # Split 90/10 for train/val.
    pos_split = max(1, int(len(positives) * 0.9))
    neg_split = max(1, int(len(negatives) * 0.9))
    train_ds = _WakeWordDataset(positives[:pos_split], negatives[:neg_split],
                                 augment=True, seed=0)
    val_ds = _WakeWordDataset(positives[pos_split:], negatives[neg_split:],
                               augment=False, seed=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("wake_word_train_device", device=device_str)

    # Frontend (mel + log) lives outside the exported model — see
    # MelFrontend docstring for why. Both training and inference compute
    # the same mel transform in Python; the ONNX model accepts mel as input.
    frontend = MelFrontend().to(device)
    frontend.eval()  # no learnable params; eval is just a safety habit
    model = WakeWordCRNN().to(device)

    # Class weighting to compensate for negative:positive ratio.
    n_pos = pos_split
    n_neg = neg_split
    pos_weight = torch.tensor([float(n_neg) / max(1, n_pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    final_train_loss = 0.0
    epochs_run = 0

    for epoch in range(epochs):
        if ctx is not None:
            await ctx.check_cancel()

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for audio, labels in train_loader:
            audio = audio.to(device)
            labels = labels.to(device).unsqueeze(1)
            with torch.no_grad():
                mel = frontend(audio)
            optim.zero_grad()
            logits = model(mel)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_train_loss = epoch_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for audio, labels in val_loader:
                audio = audio.to(device)
                labels = labels.to(device).unsqueeze(1)
                mel = frontend(audio)
                logits = model(mel)
                val_loss += loss_fn(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            # Save best so far as the ONNX. Re-export each improvement.
            _export_onnx(model, output_path, device)

        epochs_run = epoch + 1
        log.info("wake_word_train_epoch",
                 epoch=epochs_run, train_loss=round(final_train_loss, 4),
                 val_loss=round(val_loss, 4), val_acc=round(val_acc, 4))

        if ctx is not None:
            base = 0.60
            p = (epoch + 1) / epochs
            await ctx.update_progress(base + p * 0.40, stage=f"train_epoch_{epochs_run}")

    # Threshold tuning — sweep over sigmoid cutoffs on the validation set
    # to maximize F1. Stored in result for the inference layer to read at
    # detection time instead of defaulting to 0.5. Free quality bump on
    # the same trained weights — no extra training.
    best_threshold, t_precision, t_recall, t_f1 = _sweep_threshold(
        model, frontend, val_loader, device,
    )
    log.info(
        "wake_word_threshold_swept",
        threshold=best_threshold, precision=t_precision,
        recall=t_recall, f1=t_f1,
    )

    return TrainResult(
        epochs_run=epochs_run,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        final_train_loss=final_train_loss,
        positives_count=len(positives),
        negatives_count=len(negatives),
        device=device_str,
        best_threshold=best_threshold,
        val_precision_at_best=t_precision,
        val_recall_at_best=t_recall,
        val_f1_at_best=t_f1,
        negatives_pipeline=(
            "real_audio" if use_real_audio_negatives else "synthetic_only"
        ),
        personal_samples_count=personal_count,
        notes=notes,
    )


def _export_onnx(model: nn.Module, path: Path, device: torch.device) -> None:
    """Export ``model`` to ONNX at ``path``. Dynamic batch axis only.

    Input is the pre-computed log-mel: (B, 1, N_MELS, T) where T is the
    number of frames in a 1-second window (~101 at our 10ms hop). The
    mel transform runs in Python at inference time — see MelFrontend.
    """
    model.eval()
    # T = floor((WINDOW_SAMPLES - N_FFT) / HOP_LENGTH) + 1 once windowed
    # by torchaudio's mel transform with center=True (default).
    # For 16000 samples, 400 fft, 160 hop, center=True: T = 101.
    n_frames = WINDOW_SAMPLES // HOP_LENGTH + 1
    dummy = torch.randn(1, 1, N_MELS, n_frames, device=device)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("wake_word_onnx_exported", path=str(path))
