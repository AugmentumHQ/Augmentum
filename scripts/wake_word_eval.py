#!/usr/bin/env python3
"""Measure FA/FR for a trained wake-word ONNX model.

This is the canary signal for whether the new real-audio negatives
pipeline actually improves quality over the old synthetic-only path.
Runs the trained model against three in-memory test sets and reports
score distributions + FA/FRR across a sweep of thresholds.

Usage (inside the augmentum container):

    docker exec augmentum-augmentum-1 \\
        python3 /app/scripts/wake_word_eval.py bundled_f_becca

Test sets:

  * positives — Kokoro synth of the wake phrase across voices NOT in
    the training voice list (proxy for "different speaker than the
    model trained on" without requiring user recordings). Healthy
    model should fire on most of these.

  * hard_negatives — Kokoro synth of near-trigger words: ``becca``
    alone (legacy single-word that was the original feedback-loop
    source), ``hey beck``, ``hey jessica``, etc. A model that fires
    here has the same defect as the old model the service.py quality
    gate excludes.

  * librispeech_speech — held-out LibriSpeech utterances. The model
    was trained against the full speaker pool, but eval samples a
    deterministic subset (first 4 speakers by id) so re-runs are
    reproducible. This is the closest in-repo proxy for "real
    foreground speech that isn't the wake phrase."

Output is a per-set score histogram + an FA/FRR table across the
threshold sweep so we can read the calibration curve at a glance.
JSON also saved to /data/wake_word_eval/{avatar_id}-{timestamp}.json
for archival comparison between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Container layout — script lives at /app/scripts/, augmentum at /app/.
sys.path.insert(0, "/app")

from augmentum.config import settings  # noqa: E402
from augmentum.voice.kokoro_tts import KokoroTTS  # noqa: E402
from augmentum.voice.wake_word import negatives_corpus  # noqa: E402
from augmentum.voice.wake_word.service import _compute_log_mel  # noqa: E402
from augmentum.voice.wake_word.training import (  # noqa: E402
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    _decode_wav_bytes,
    _pad_or_clip,
)


# ── Configuration ──────────────────────────────────────────────────


# Kokoro voices that the default training pool does NOT use. Picking
# from these for synthetic positives gives the model novel timbres to
# generalize to, instead of letting it score against the same voices
# it overfit to during training.
#
# ``af_bella`` is split out into ``TEMPLATE_VOICE`` below — when running
# with --with-channel-b it acts as the "user-voice templates" source,
# and a held-out bucket of additional af_bella generations stands in
# for "the user re-saying the phrase in a new session." Keeping it out
# of EVAL_VOICES keeps the held-out positive pool disjoint from the
# template source. With --no-with-channel-b the harness behaves exactly
# as before (af_bella stayed in pre-Phase-0; we accept that small shift
# in the baseline positives pool for the cleanliness of the comparison).
EVAL_VOICES = ["af_jessica", "am_eric", "am_liam", "bf_alice"]

# Channel B (hybrid Phase 0) — synthetic-templates proxy for a single
# enrolled user. Templates are five takes of the wake phrase at five
# speeds from one voice; held-out same-voice positives test "this user
# in another session." Different-voice positives (EVAL_VOICES) test the
# permissive-mode "guest in the kitchen" path.
TEMPLATE_VOICE = "af_bella"
TEMPLATE_SPEEDS = (0.85, 0.925, 1.0, 1.075, 1.15)
SAME_VOICE_HOLDOUT_COUNT = 8

# Near-trigger phrases. ``becca`` alone is the legacy single-word that
# the service.py quality gate explicitly excludes — if the new model
# still fires on it, the feedback-loop defect is back.
HARD_NEG_PHRASES = [
    "becca",
    "hey beck",
    "hey jessica",
    "hey becky",
    "okay becca",
    "hey bella",
    "hey rebecca",
]

# Per-voice generations for the synthetic test sets. Small because each
# generation is ~2s of Kokoro work and the eval is meant to be a quick
# canary signal, not an exhaustive benchmark.
POSITIVES_PER_VOICE = 6   # 5 voices × 6 = 30 positives
HARD_NEG_PER_VOICE = 2    # 7 phrases × 5 voices × 2 = 70 hard negatives

# Held-out LibriSpeech speakers. The catalog is keyed by speaker_id at
# the first path level (e.g. ``3853/163249/utt.flac``). We take the
# first 4 by sorted id — deterministic so re-runs against the same
# model produce the same numbers (modulo cropping RNG).
EVAL_SPEAKER_COUNT = 4
LIBRISPEECH_WINDOWS = 300

# Threshold sweep. The service.py runtime floor is 0.65; this sweep
# brackets that, plus the F1-optimal 0.25-0.50 range the synthetic-only
# pipeline used to pick.
THRESHOLDS = [0.25, 0.40, 0.50, 0.60, 0.65, 0.75, 0.85]


# ── Inference ──────────────────────────────────────────────────────


def _load_session(model_path: str):
    """Load the ONNX model via onnxruntime. Mirrors service.py exactly."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _score(session, samples: np.ndarray) -> float:
    """Single-shot inference. Returns sigmoid probability in [0, 1]."""
    if samples.shape[0] != WINDOW_SAMPLES:
        samples = _pad_or_clip(samples, WINDOW_SAMPLES)
    mel = _compute_log_mel(samples)
    outputs = session.run(None, {"mel": mel.astype(np.float32)})
    logits = outputs[0]
    return float(1.0 / (1.0 + np.exp(-logits[0, 0])))


# ── Channel B (Phase 0) — template-matching via DTW on log-mel ─────────
#
# Implements the Channel B principle from
# docs/superpowers/specs/2026-05-27-wake-word-hybrid-design.md without
# committing to the 30 MB Google speech-embedding ONNX (Phase 1
# decision). Phase 0 uses log-mel features the harness already
# computes — if even classical DTW shows independence from the CRNN's
# failure surface, the embedding upgrade in Phase 1 is justified.
#
# DTW config:
#   - Cosine distance per L2-normalized 40-dim log-mel frame
#   - Sakoe-Chiba band radius = 10 frames (≈10 % of the 101-frame window)
#   - Path-length-normalized cost (returned as "distance"; lower = better)
#
# Score in [0, 1] is derived via ``_dtw_to_score`` using the calibration
# from ``calibrate_channel_b``. Both go to None when calibration fails
# (templates don't separate from the negative pool — bail rather than
# fire spuriously).


def _samples_to_mel_frames(samples: np.ndarray) -> np.ndarray:
    """Return (T, n_mels) float32 log-mel frames for a 1-second window.

    Shares the exact mel front-end the CRNN saw at training time —
    same n_fft / hop / power / db-conversion. We just transpose into
    (T, D) layout because DTW iterates along time.
    """
    mel = _compute_log_mel(samples)        # (1, 1, n_mels, T)
    frames = mel[0, 0]                     # (n_mels, T)
    return frames.T.astype(np.float32)     # (T, n_mels)


# ── Phase 1 A/B: pluggable feature extractors ──────────────────────────
#
# Phase 0 with real recordings showed log-mel + DTW does not separate
# a single speaker's intra-session variance from LibriSpeech speakers. The Phase 1
# answer is to feed DTW a feature space that IS speaker-discriminative
# instead. Three candidates via torchaudio.pipelines (no HF download
# needed; checkpoints cache to ~/.cache/torch/hub on first call):
#
#   wav2vec2 — facebook's self-supervised ASR features. 95 MB checkpoint.
#              Trained for phonetic discrimination, not speaker ID, but
#              the lower layers are known to carry speaker info.
#
#   hubert   — facebook HuBERT. Similar architecture, different training.
#              Often slightly stronger than wav2vec2 on speaker tasks.
#
#   wavlm    — Microsoft WavLM. Designed explicitly for speaker
#              verification + diarization. Should give the cleanest
#              speaker separation of the three.
#
# All three output frame-level features at ~50 Hz (49 frames per 1-second
# window), vs. mel's 101 frames. DTW band radius scales with T so the
# proportional band stays constant.


_emb_model = None
_emb_kind: str | None = None


def _load_embedding_model(kind: str):
    """Lazily load and cache a torchaudio embedding pipeline.

    First call downloads ~95 MB to the torch hub cache. Subsequent runs
    in the same container are instant. Sent to CUDA when available
    because per-window embedding on CPU is ~50 ms while on a consumer
    NVIDIA GPU it's ~1 ms — matters when scoring 400+ windows.
    """
    global _emb_model, _emb_kind
    if _emb_model is not None and _emb_kind == kind:
        return _emb_model
    import torch
    import torchaudio
    bundles = {
        "wav2vec2": torchaudio.pipelines.WAV2VEC2_BASE,
        "hubert":   torchaudio.pipelines.HUBERT_BASE,
        "wavlm":    torchaudio.pipelines.WAVLM_BASE,
    }
    if kind not in bundles:
        raise ValueError(f"unknown feature extractor: {kind}")
    bundle = bundles[kind]
    model = bundle.get_model()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    _emb_model = model
    _emb_kind = kind
    return model


def _samples_to_embedding_frames(
    samples: np.ndarray, kind: str,
) -> np.ndarray:
    """Run one 1-second window through the selected embedding model,
    return (T, D) frame features ready for DTW.

    The embedding models all consume raw 16 kHz PCM as a (1, samples)
    tensor and return (1, T, D) features. T ~= 49 at 1 second (≈20 ms
    hop). D varies by model (768 for base variants).
    """
    import torch
    model = _load_embedding_model(kind)
    audio = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)  # (1, samples)
    if torch.cuda.is_available():
        audio = audio.cuda()
    with torch.no_grad():
        # torchaudio bundle models return (features, lengths) for ASR
        # convenience. We only want the features.
        out = model.extract_features(audio)
        if isinstance(out, tuple):
            features = out[0]
        else:
            features = out
        # extract_features may return a list (one per layer) — take last.
        if isinstance(features, list):
            features = features[-1]
    # features: (B, T, D) → (T, D)
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def features_for(samples: np.ndarray, extractor: str) -> np.ndarray:
    """Single entry point used by Channel B regardless of which feature
    space we're running in. ``extractor='mel'`` uses the existing
    log-mel path; anything else dispatches to an embedding model.
    """
    if extractor == "mel":
        return _samples_to_mel_frames(samples)
    return _samples_to_embedding_frames(samples, extractor)


def _dtw_cosine(a: np.ndarray, b: np.ndarray, radius: int | None = None) -> float:
    """Band-limited DTW with cosine distance per frame.

    Args:
        a: (T_a, D) query frames
        b: (T_b, D) template frames
        radius: Sakoe-Chiba band — max |i - j| in frame indices. When
            None, scales to ~10 % of max(T_a, T_b) so mel (T≈101) and
            embedding (T≈49) feature spaces both get a proportional
            band. Minimum 5 frames to avoid pathological starvation.

    Returns:
        DTW cost normalized by approximate path length. Lower = more
        similar. Costs above ~0.7 indicate "different content"
        (cosine distance saturates at 2 for opposite vectors, but
        typical speech frames live in a much narrower cone).
    """
    T_a, _ = a.shape
    T_b, _ = b.shape
    if radius is None:
        radius = max(5, max(T_a, T_b) // 10)
    # L2-normalize once so the inner loop is a single dot product.
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    INF = np.float32(1e9)
    dp = np.full((T_a + 1, T_b + 1), INF, dtype=np.float32)
    dp[0, 0] = 0.0
    # We use index-diagonal banding (|i - j| <= radius) rather than the
    # scaled-diagonal variant because both sequences are the same
    # canonical length (101 frames per 1-second window); they don't
    # need rescaling.
    for i in range(1, T_a + 1):
        j_lo = max(1, i - radius)
        j_hi = min(T_b, i + radius)
        ai = a_n[i - 1]
        for j in range(j_lo, j_hi + 1):
            cost = 1.0 - float(ai @ b_n[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]
            )
    final = float(dp[T_a, T_b])
    if final >= INF / 2:
        return 1.0  # band starvation — treat as max distance
    # Approximate path length is around max(T_a, T_b) for diagonal-
    # dominated alignment.
    return final / max(T_a, T_b)


def best_template_distance(
    query_frames: np.ndarray,
    template_frames: list[np.ndarray],
) -> float:
    """Return the minimum DTW distance across all templates.

    Matches the runtime contract from the spec: a query fires Channel B
    iff it matches *any* template well. The user's own variants form a
    disjunction.
    """
    if not template_frames:
        return 1.0
    return min(_dtw_cosine(query_frames, tf) for tf in template_frames)


@dataclass
class ChannelBCalibration:
    """Output of :func:`calibrate_channel_b`. Captures separation
    quality and the per-template threshold the runtime would use.
    """
    d_self_max: float       # worst case template-vs-template
    d_neg_min: float        # best case template-vs-non-trigger
    tau: float | None       # midpoint threshold (None = no separation)
    separated: bool         # d_self_max < d_neg_min
    n_templates: int
    n_neg_calibration: int


def calibrate_channel_b(
    template_frames: list[np.ndarray],
    neg_calibration_frames: list[np.ndarray],
) -> ChannelBCalibration:
    """Compute the per-bake Channel B threshold from templates + a
    sample of non-trigger speech.

    Mirrors the threshold-calibration math in the spec:
      d_self_max = worst pairwise DTW among templates themselves
      d_neg_min  = best DTW from any template to any non-trigger frame
      τ = midpoint if d_self_max < d_neg_min, else None (insufficient)

    ``neg_calibration_frames`` should be a small sample drawn from the
    same distribution the user's room will see — LibriSpeech crops are
    the in-repo proxy. ~30-50 is plenty for a stable estimate; more
    just slows calibration without changing the answer.
    """
    n = len(template_frames)
    if n < 2:
        return ChannelBCalibration(1.0, 0.0, None, False, n, len(neg_calibration_frames))
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(_dtw_cosine(template_frames[i], template_frames[j]))
    d_self_max = max(pairwise) if pairwise else 1.0

    neg_distances: list[float] = []
    for nf in neg_calibration_frames:
        d = best_template_distance(nf, template_frames)
        neg_distances.append(d)
    d_neg_min = min(neg_distances) if neg_distances else 0.0

    separated = d_self_max < d_neg_min
    tau = ((d_self_max + d_neg_min) / 2.0) if separated else None
    return ChannelBCalibration(
        d_self_max=d_self_max,
        d_neg_min=d_neg_min,
        tau=tau,
        separated=separated,
        n_templates=n,
        n_neg_calibration=len(neg_calibration_frames),
    )


# ── Test-set builders ──────────────────────────────────────────────


async def _kokoro_window(
    kokoro: Any, phrase: str, voice: str, speed: float,
) -> np.ndarray | None:
    """Generate one 1-second 16 kHz window. Returns None on failure so
    the caller can skip without inflating the test set with zeros.
    """
    try:
        wav_bytes = await kokoro.generate(
            phrase, voice=voice, speed=speed, response_format="wav",
        )
    except Exception as exc:
        print(f"  Kokoro fail ({voice}, '{phrase}'): {exc}", flush=True)
        return None
    if not wav_bytes:
        return None
    samples = _decode_wav_bytes(wav_bytes)
    return _pad_or_clip(samples, WINDOW_SAMPLES)


async def build_positives(kokoro: Any, phrase: str) -> list[np.ndarray]:
    """Synth the wake phrase across out-of-training voices. The harness
    intentionally uses voices ``EVAL_VOICES`` that the default training
    pool excludes — that pushes the model to generalize across timbres
    rather than score against the same voices it saw.
    """
    speeds = (0.85, 1.0, 1.15)
    out: list[np.ndarray] = []
    for voice in EVAL_VOICES:
        for _ in range(POSITIVES_PER_VOICE):
            speed = random.choice(speeds)
            w = await _kokoro_window(kokoro, phrase, voice, speed)
            if w is not None:
                out.append(w)
    return out


async def build_templates(kokoro: Any, phrase: str) -> list[np.ndarray]:
    """Synthesize five templates from ``TEMPLATE_VOICE`` across speeds.

    Phase 0 proxy for "the user recorded five takes of their wake
    phrase at enrolment." Real templates would be raw user mic input;
    the harness uses Kokoro generation at a single voice so the test
    is reproducible and runs in CI without needing live recordings.
    """
    out: list[np.ndarray] = []
    for speed in TEMPLATE_SPEEDS:
        w = await _kokoro_window(kokoro, phrase, TEMPLATE_VOICE, speed)
        if w is not None:
            out.append(w)
    return out


async def build_same_voice_holdout(
    kokoro: Any, phrase: str,
) -> list[np.ndarray]:
    """Held-out positives from the template voice — proxies "the user
    re-says the phrase in another session, slightly different prosody."

    These are what strict mode needs to keep firing on. If Channel B
    rejects too many of these, the per-bake calibration was too tight
    or the templates underrepresent the user's prosodic range.
    """
    out: list[np.ndarray] = []
    speeds = (0.85, 0.95, 1.0, 1.05, 1.15)
    for _ in range(SAME_VOICE_HOLDOUT_COUNT):
        speed = random.choice(speeds)
        w = await _kokoro_window(kokoro, phrase, TEMPLATE_VOICE, speed)
        if w is not None:
            out.append(w)
    return out


def load_real_recordings(
    directory: Path,
    *,
    n_templates: int,
    n_holdout: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Load real user wake-word recordings from ``directory`` and split
    deterministically into (templates, same-voice-holdout, remaining).

    The directory layout is the one produced by
    ``POST /api/wake_word/personal_samples`` —
    ``/data/wake_word_personal/{user_id}/{avatar_id}/*.wav``. Files are
    already 16 kHz mono PCM16 + length-validated to 0.5-2 s; we just
    center-pad/clip each to the canonical 1-second window the harness
    operates on.

    Shuffle is deterministic via ``seed`` so re-runs are reproducible.
    Returns (templates, holdout, total_loaded) so the caller can report
    how many real recordings were available vs. how many were assigned
    to each pool.
    """
    wavs = sorted(directory.glob("*.wav"))
    rng = random.Random(seed)
    rng.shuffle(wavs)

    loaded: list[np.ndarray] = []
    for path in wavs:
        try:
            samples = _decode_wav_bytes(path.read_bytes())
            samples = _pad_or_clip(samples, WINDOW_SAMPLES)
            loaded.append(samples)
        except Exception as exc:
            print(f"  skipped {path.name}: {exc}", flush=True)

    if len(loaded) < n_templates + n_holdout:
        print(
            f"  WARNING: only {len(loaded)} recordings; "
            f"{n_templates} templates + {n_holdout} holdout requested",
            flush=True,
        )

    templates = loaded[:n_templates]
    holdout = loaded[n_templates:n_templates + n_holdout]
    return templates, holdout, len(loaded)


async def build_hard_negatives(kokoro: Any) -> dict[str, list[np.ndarray]]:
    """One bucket per near-trigger phrase so the report can break out
    score-by-phrase. Lets us spot specific failure modes (e.g., the
    model fires on ``becca`` alone but not the others — that's a
    legacy-defect tell).
    """
    speeds = (0.85, 1.0, 1.15)
    by_phrase: dict[str, list[np.ndarray]] = {p: [] for p in HARD_NEG_PHRASES}
    for phrase in HARD_NEG_PHRASES:
        for voice in EVAL_VOICES:
            for _ in range(HARD_NEG_PER_VOICE):
                speed = random.choice(speeds)
                w = await _kokoro_window(kokoro, phrase, voice, speed)
                if w is not None:
                    by_phrase[phrase].append(w)
    return by_phrase


def build_librispeech(rng: random.Random) -> list[np.ndarray]:
    """Sample windows from held-out LibriSpeech speakers.

    Catalog entries are ``{'path': 'spk/chapter/utt.flac', ...}``. We
    deterministically pick the first ``EVAL_SPEAKER_COUNT`` speakers
    by sorted speaker_id so eval runs are reproducible. ``sample_real_
    speech_windows`` would sample uniformly across all speakers — fine
    for training but bad for eval.
    """
    if not negatives_corpus.is_installed():
        print(
            "WARNING: LibriSpeech corpus not installed — skipping librispeech eval",
            flush=True,
        )
        return []

    import torchaudio
    catalog = negatives_corpus._get_flac_catalog()
    if not catalog:
        return []

    # Group by speaker_id (first directory level).
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog:
        spk = entry["path"].split("/", 1)[0]
        by_spk.setdefault(spk, []).append(entry)

    held_out_spks = sorted(by_spk.keys())[:EVAL_SPEAKER_COUNT]
    held_out_pool: list[dict[str, Any]] = []
    for spk in held_out_spks:
        held_out_pool.extend(by_spk[spk])
    print(
        f"  LibriSpeech held-out: {len(held_out_spks)} speakers, "
        f"{len(held_out_pool)} utterances",
        flush=True,
    )

    flac_root = negatives_corpus._flac_root()
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < LIBRISPEECH_WINDOWS and attempts < LIBRISPEECH_WINDOWS * 4:
        attempts += 1
        entry = rng.choice(held_out_pool)
        frames = int(entry.get("frames", 0))
        if frames < WINDOW_SAMPLES:
            continue
        path = flac_root / entry["path"]
        offset = rng.randint(0, frames - WINDOW_SAMPLES)
        try:
            wav, sr = torchaudio.load(
                str(path), frame_offset=offset, num_frames=WINDOW_SAMPLES,
            )
        except Exception:
            continue
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        samples = wav.squeeze(0).numpy().astype(np.float32)
        if samples.shape[0] != WINDOW_SAMPLES:
            samples = _pad_or_clip(samples, WINDOW_SAMPLES)
        out.append(samples)
    return out


# ── Reporting ──────────────────────────────────────────────────────


def _describe(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"n": 0}
    arr = np.array(scores, dtype=np.float32)
    return {
        "n": len(scores),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _trigger_rate(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    triggered = sum(1 for s in scores if s >= threshold)
    return triggered / len(scores)


def _print_table(
    pos: list[float],
    hard_neg: dict[str, list[float]],
    speech: list[float],
) -> None:
    print()
    print("=" * 80)
    print("SCORE DISTRIBUTIONS")
    print("=" * 80)
    print(f"  positives        : {_describe(pos)}")
    for phrase, scores in hard_neg.items():
        print(f"  hard_neg ({phrase:<14}): {_describe(scores)}")
    print(f"  librispeech      : {_describe(speech)}")

    print()
    print("=" * 80)
    print("FA / FRR TABLE  (higher threshold = fewer FAs, more FRRs)")
    print("=" * 80)
    cols = ["threshold", "FRR(pos)", "FA(hard)", "FA(speech)"]
    print(f"{cols[0]:>10s}  {cols[1]:>10s}  {cols[2]:>10s}  {cols[3]:>12s}")
    print("-" * 50)
    all_hard = [s for scores in hard_neg.values() for s in scores]
    for t in THRESHOLDS:
        frr = 1.0 - _trigger_rate(pos, t)
        fa_hard = _trigger_rate(all_hard, t)
        fa_speech = _trigger_rate(speech, t)
        print(
            f"{t:>10.2f}  {frr*100:>9.1f}%  {fa_hard*100:>9.1f}%  {fa_speech*100:>11.1f}%"
        )

    # Per-phrase hard-negative breakdown at the runtime-floor threshold.
    print()
    print("=" * 80)
    print(f"HARD-NEGATIVE TRIGGERS AT THRESHOLD {THRESHOLDS[4]:.2f} (runtime floor)")
    print("=" * 80)
    for phrase, scores in hard_neg.items():
        rate = _trigger_rate(scores, THRESHOLDS[4])
        n_trig = sum(1 for s in scores if s >= THRESHOLDS[4])
        marker = " ← LEGACY FEEDBACK-LOOP WORD" if phrase == "becca" else ""
        print(
            f"  {phrase:<14}  {n_trig:>2d}/{len(scores):<2d} triggered  "
            f"({rate*100:>5.1f}%){marker}"
        )


# ── Channel B reporting ────────────────────────────────────────────


# Channel B threshold sweep (distance space, lower = better). Picked to
# bracket the typical calibrated τ values we expect: ~0.20 for highly
# similar templates, ~0.50 around the separation boundary.
CHANNEL_B_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def _b_trigger_rate(distances: list[float], threshold: float) -> float:
    """Fraction of windows where DTW distance ≤ threshold (fires)."""
    if not distances:
        return 0.0
    triggered = sum(1 for d in distances if d <= threshold)
    return triggered / len(distances)


def _describe_distances(distances: list[float]) -> dict[str, float]:
    """Same shape as _describe but for distance values (lower = better)."""
    if not distances:
        return {"n": 0}
    arr = np.array(distances, dtype=np.float32)
    return {
        "n": len(distances),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _print_channel_b_table(
    cal: ChannelBCalibration,
    pos_same: list[float],
    pos_other: list[float],
    hard_neg: dict[str, list[float]],
    speech: list[float],
) -> None:
    """Mirror of _print_table for Channel B distances."""
    print()
    print("=" * 80)
    print("CHANNEL B (template DTW) — CALIBRATION")
    print("=" * 80)
    print(f"  n_templates        : {cal.n_templates}")
    print(f"  d_self_max         : {cal.d_self_max:.4f}  (worst template-vs-template)")
    print(f"  d_neg_min          : {cal.d_neg_min:.4f}  (best template-vs-LibriSpeech)")
    print(f"  separated          : {cal.separated}")
    print(f"  calibrated τ       : {cal.tau:.4f}" if cal.tau is not None
          else "  calibrated τ       : INSUFFICIENT — templates do not separate")

    print()
    print("=" * 80)
    print("CHANNEL B — DISTANCE DISTRIBUTIONS  (lower = more similar)")
    print("=" * 80)
    print(f"  positives (same-voice holdout)  : {_describe_distances(pos_same)}")
    print(f"  positives (other voices)        : {_describe_distances(pos_other)}")
    for phrase, ds in hard_neg.items():
        print(f"  hard_neg ({phrase:<14}): {_describe_distances(ds)}")
    print(f"  librispeech                     : {_describe_distances(speech)}")

    print()
    print("=" * 80)
    print("CHANNEL B — FA / FRR TABLE  (threshold is DTW distance; lower = fire)")
    print("=" * 80)
    cols = ["threshold", "FRR(same)", "FRR(other)", "FA(hard)", "FA(speech)"]
    print(f"{cols[0]:>10s}  {cols[1]:>10s}  {cols[2]:>11s}  {cols[3]:>10s}  {cols[4]:>11s}")
    print("-" * 60)
    all_hard = [d for ds in hard_neg.values() for d in ds]
    for t in CHANNEL_B_THRESHOLDS:
        frr_same = 1.0 - _b_trigger_rate(pos_same, t)
        frr_other = 1.0 - _b_trigger_rate(pos_other, t)
        fa_hard = _b_trigger_rate(all_hard, t)
        fa_speech = _b_trigger_rate(speech, t)
        marker = " ← calibrated τ" if cal.tau is not None and abs(t - cal.tau) < 0.025 else ""
        print(
            f"{t:>10.3f}  {frr_same*100:>9.1f}%  {frr_other*100:>10.1f}%  "
            f"{fa_hard*100:>9.1f}%  {fa_speech*100:>10.1f}%{marker}"
        )


def _combine_strict(crnn_scores: list[float], b_distances: list[float],
                    t_crnn: float, t_b: float) -> list[bool]:
    """AND: both channels must fire. CRNN ≥ t_crnn AND DTW ≤ t_b."""
    return [(c >= t_crnn) and (d <= t_b) for c, d in zip(crnn_scores, b_distances)]


def _combine_permissive(crnn_scores: list[float], b_distances: list[float],
                        t_crnn: float, t_b: float) -> list[bool]:
    """OR: either channel fires. CRNN ≥ t_crnn OR DTW ≤ t_b."""
    return [(c >= t_crnn) or (d <= t_b) for c, d in zip(crnn_scores, b_distances)]


def _rate(fires: list[bool]) -> float:
    return (sum(fires) / len(fires)) if fires else 0.0


def _print_combinator_table(
    label: str,
    pairs: dict[str, tuple[list[float], list[float]]],
    t_crnn: float, t_b: float,
    combiner,
) -> None:
    """Render one combinator at one operating point against every test
    bucket. ``pairs`` is {bucket_name: (crnn_scores, b_distances)}.
    """
    print()
    print("=" * 80)
    print(f"{label}   |   τ_crnn = {t_crnn:.2f}   τ_b = {t_b:.3f}")
    print("=" * 80)
    print(f"  {'bucket':<32s}  {'n':>4s}  {'trigger rate':>14s}")
    print("-" * 60)
    for name, (c, d) in pairs.items():
        if not c:
            print(f"  {name:<32s}  {0:>4d}  {'—':>14s}")
            continue
        fires = combiner(c, d, t_crnn, t_b)
        rate = _rate(fires)
        print(f"  {name:<32s}  {len(c):>4d}  {rate*100:>12.1f}%")


def _find_threshold_at_frr(
    pos_scores: list[float],
    target_frr: float,
    candidates: list[float],
    *,
    direction: str = "sigmoid",
) -> tuple[float, float]:
    """Pick the strictest threshold from ``candidates`` whose FRR on
    ``pos_scores`` is at or below ``target_frr``.

    ``direction='sigmoid'`` — higher threshold = stricter (CRNN).
    ``direction='distance'`` — lower threshold = stricter (Channel B).

    If no candidate meets the target, returns the candidate with the
    lowest FRR (most generous threshold).

    Returns (threshold, achieved_frr).
    """
    meeting: list[tuple[float, float]] = []
    fallback: tuple[float, float] = (candidates[0], 1.0)
    for t in candidates:
        if direction == "sigmoid":
            frr = 1.0 - _trigger_rate(pos_scores, t)
        else:
            frr = 1.0 - _b_trigger_rate(pos_scores, t)
        if frr < fallback[1]:
            fallback = (t, frr)
        if frr <= target_frr + 1e-6:
            meeting.append((t, frr))
    if not meeting:
        return fallback
    if direction == "sigmoid":
        meeting.sort(key=lambda x: x[0], reverse=True)   # strictest = highest
    else:
        meeting.sort(key=lambda x: x[0])                 # strictest = lowest
    return meeting[0]


def _print_consequences_of_strict(
    pos_other_scores: list[float],
    pos_same_scores: list[float],
    hard_scores: list[float],
    speech_scores: list[float],
    pos_other_dist: list[float],
    pos_same_dist: list[float],
    hard_dist: list[float],
    speech_dist: list[float],
    cal: ChannelBCalibration,
) -> None:
    """The decision-gate table promised in the spec.

    Pin FRR on the same-voice holdout pool (the user's own voice — the
    one strict mode promises to keep firing on) at ~10 % across all
    four operating modes:
      1. CRNN-only legacy
      2. Channel B-only
      3. Strict (AND)
      4. Permissive (OR)

    Then report FA on hard-negatives and on LibriSpeech at that pinned
    FRR. The mode whose FA stays lowest while preserving same-voice
    recall is the operating point the spec was aiming for.

    Out-of-template-voice FRR is reported as a separate column so the
    "guest in the kitchen" cost is visible — that's where strict mode
    legitimately gives up recall in exchange for FA reduction.
    """
    target_frr = 0.10  # 10 % FRR on the same-voice holdout

    # CRNN-only operating point.
    t_crnn, frr_same_crnn = _find_threshold_at_frr(
        pos_same_scores, target_frr, THRESHOLDS, direction="sigmoid"
    )
    fa_hard_crnn = _trigger_rate(hard_scores, t_crnn)
    fa_speech_crnn = _trigger_rate(speech_scores, t_crnn)
    frr_other_crnn = 1.0 - _trigger_rate(pos_other_scores, t_crnn)

    # Channel B-only operating point. We always run the picker against
    # the actual same-voice distance distribution even when calibration
    # reports separated=False — the calibration's "insufficient" verdict
    # gates whether to enable Channel B in *production*, but for the
    # eval report we want to see the best operating point the data
    # supports regardless. Skipping it would let a bad fallback like
    # τ=0.15 mask a usable τ=0.50 in the verdict table.
    if pos_same_dist:
        t_b, frr_same_b = _find_threshold_at_frr(
            pos_same_dist, target_frr, CHANNEL_B_THRESHOLDS,
            direction="distance",
        )
    else:
        t_b, frr_same_b = (CHANNEL_B_THRESHOLDS[0], 1.0)
    fa_hard_b = _b_trigger_rate(hard_dist, t_b)
    fa_speech_b = _b_trigger_rate(speech_dist, t_b)
    frr_other_b = 1.0 - _b_trigger_rate(pos_other_dist, t_b)

    # Strict (AND) — use the same t_crnn + t_b chosen independently.
    # Note: combined FRR may exceed the per-channel target because both
    # must agree. This is the explicit tradeoff strict mode promises.
    strict_same = _combine_strict(pos_same_scores, pos_same_dist, t_crnn, t_b)
    strict_other = _combine_strict(pos_other_scores, pos_other_dist, t_crnn, t_b)
    strict_hard = _combine_strict(hard_scores, hard_dist, t_crnn, t_b)
    strict_speech = _combine_strict(speech_scores, speech_dist, t_crnn, t_b)

    # Permissive (OR).
    perm_same = _combine_permissive(pos_same_scores, pos_same_dist, t_crnn, t_b)
    perm_other = _combine_permissive(pos_other_scores, pos_other_dist, t_crnn, t_b)
    perm_hard = _combine_permissive(hard_scores, hard_dist, t_crnn, t_b)
    perm_speech = _combine_permissive(speech_scores, speech_dist, t_crnn, t_b)

    print()
    print("=" * 80)
    print(f"CONSEQUENCES OF STRICT MODE  (each mode targets ~{target_frr*100:.0f}% FRR on same-voice)")
    print("=" * 80)
    print(
        f"  {'mode':<22s}  {'FRR(same)':>10s}  {'FRR(other)':>11s}  "
        f"{'FA(hard)':>9s}  {'FA(speech)':>11s}"
    )
    print("-" * 80)
    print(
        f"  {'CRNN-only (legacy)':<22s}  "
        f"{frr_same_crnn*100:>9.1f}%  {frr_other_crnn*100:>10.1f}%  "
        f"{fa_hard_crnn*100:>8.1f}%  {fa_speech_crnn*100:>10.1f}%"
    )
    print(
        f"  {'Channel B-only':<22s}  "
        f"{frr_same_b*100:>9.1f}%  {frr_other_b*100:>10.1f}%  "
        f"{fa_hard_b*100:>8.1f}%  {fa_speech_b*100:>10.1f}%"
    )
    print(
        f"  {'Strict (AND)':<22s}  "
        f"{(1.0 - _rate(strict_same))*100:>9.1f}%  "
        f"{(1.0 - _rate(strict_other))*100:>10.1f}%  "
        f"{_rate(strict_hard)*100:>8.1f}%  "
        f"{_rate(strict_speech)*100:>10.1f}%"
    )
    print(
        f"  {'Permissive (OR)':<22s}  "
        f"{(1.0 - _rate(perm_same))*100:>9.1f}%  "
        f"{(1.0 - _rate(perm_other))*100:>10.1f}%  "
        f"{_rate(perm_hard)*100:>8.1f}%  "
        f"{_rate(perm_speech)*100:>10.1f}%"
    )

    print()
    print("=" * 80)
    print("DECISION GATE — Phase 0 verdict")
    print("=" * 80)
    # The signal: does strict-AND reduce FA on hard negatives below
    # CRNN-alone without making same-voice FRR catastrophic?
    fa_hard_strict = _rate(strict_hard)
    fa_speech_strict = _rate(strict_speech)
    same_recall_strict = _rate(strict_same)
    if same_recall_strict < 0.5:
        print("  STRICT MODE LOSES TOO MUCH RECALL ON OWN-VOICE — "
              "Phase 0 hypothesis weak. Increase template count or "
              "loosen calibration before Phase 1.")
    elif fa_hard_strict < fa_hard_crnn * 0.5 and \
         fa_speech_strict < fa_speech_crnn * 0.5:
        print("  STRICT MODE HALVES FA WITH RETAINED RECALL — "
              "failure-mode-independence hypothesis supported. "
              "Phase 1 (embedding upgrade) is justified.")
    elif fa_hard_strict < fa_hard_crnn:
        print("  STRICT MODE REDUCES FA MODESTLY. "
              "Phase 1 justified only if the embedding model widens "
              "the gap further. Consider running with real personal "
              "templates before committing.")
    else:
        print("  STRICT MODE DOES NOT REDUCE FA. "
              "Phase 0 hypothesis NOT supported with classical DTW. "
              "Either templates don't add information beyond CRNN, "
              "or the eval setup is masking the gap. Investigate "
              "before Phase 1.")


# ── Main ───────────────────────────────────────────────────────────


def _load_wake_word_row(slug: str) -> dict[str, Any]:
    db_path = Path(settings.data_dir) / "augmentum.db"
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT avatar_id, phrase, model_path, version, train_metrics, trained_at "
        "FROM wake_word_models WHERE avatar_id = ?",
        (slug,),
    ).fetchone()
    con.close()
    if not row:
        raise SystemExit(f"no wake_word_models row for avatar_id={slug!r}")
    metrics = json.loads(row["train_metrics"]) if row["train_metrics"] else {}
    return {
        "avatar_id": row["avatar_id"],
        "phrase": row["phrase"],
        "model_path": row["model_path"],
        "version": row["version"],
        "trained_at": row["trained_at"],
        "metrics": metrics,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("avatar_id", help="e.g. bundled_f_becca")
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG seed for LibriSpeech sampling (default: 7)")
    ap.add_argument(
        "--with-channel-b",
        action="store_true",
        help=(
            "Phase 0 hybrid eval: also score a Channel B (template-DTW) "
            "leg and report strict/permissive combinators. See "
            "docs/superpowers/specs/2026-05-27-wake-word-hybrid-design.md."
        ),
    )
    ap.add_argument(
        "--personal-templates-dir",
        type=Path,
        default=None,
        help=(
            "If set, load real WAV recordings from this directory as "
            "Channel B templates + same-voice holdout instead of "
            "synthesizing them from Kokoro. Layout is "
            "/data/wake_word_personal/{user}/{avatar}/*.wav. The split "
            "is 8 templates + 10 holdout by default; the rest of the "
            "WAVs are unused. Implies --with-channel-b."
        ),
    )
    ap.add_argument(
        "--n-real-templates",
        type=int,
        default=8,
        help="How many real recordings to use as templates (default 8).",
    )
    ap.add_argument(
        "--n-real-holdout",
        type=int,
        default=10,
        help="How many real recordings to hold out as same-voice positives (default 10).",
    )
    ap.add_argument(
        "--feature-extractor",
        choices=["mel", "wav2vec2", "hubert", "wavlm"],
        default="mel",
        help=(
            "Channel B feature space: 'mel' (Phase 0 baseline) or one "
            "of three torchaudio embedding pipelines (Phase 1 A/B). "
            "First call to an embedding downloads ~95 MB to torch hub "
            "cache. WAVLM is designed for speaker-aware tasks and is "
            "the most likely candidate for the production system."
        ),
    )
    ap.add_argument(
        "--model-path-override",
        type=Path,
        default=None,
        help=(
            "Score against this ONNX instead of the one recorded in "
            "wake_word_models for the avatar_id. Used by Phase 0.5 "
            "to evaluate a sidecar rebake without touching the prod row."
        ),
    )
    args = ap.parse_args()
    if args.personal_templates_dir is not None:
        args.with_channel_b = True

    print("=" * 80)
    print(f"wake-word eval — {args.avatar_id}")
    print("=" * 80)
    row = _load_wake_word_row(args.avatar_id)
    print(f"  phrase             : {row['phrase']!r}")
    print(f"  version            : v{row['version']}")
    print(f"  trained_at         : {row['trained_at']}")
    print(f"  model_path         : {row['model_path']}")
    m = row["metrics"]
    print(f"  negatives_pipeline : {m.get('negatives_pipeline', 'synthetic_only')}")
    print(f"  recorded val_acc   : {m.get('best_val_acc', 0):.3f}")
    print(f"  recorded threshold : {m.get('best_threshold', 0.5):.3f}")
    print(f"  positives_count    : {m.get('positives_count', '?')}")
    print(f"  negatives_count    : {m.get('negatives_count', '?')}")

    if args.model_path_override is not None:
        if not args.model_path_override.exists():
            raise SystemExit(
                f"override model file missing: {args.model_path_override}"
            )
        row["model_path"] = str(args.model_path_override)
        print(f"  ** OVERRIDE MODEL: {args.model_path_override} **")
    elif not Path(row["model_path"]).exists():
        raise SystemExit(f"model file missing: {row['model_path']}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    print()
    print("loading Kokoro …", flush=True)
    kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
    await asyncio.to_thread(kokoro.load_model)
    if not kokoro.is_available:
        raise SystemExit("Kokoro unavailable — cannot build eval sets")

    print(f"building positives (×{POSITIVES_PER_VOICE} per voice × {len(EVAL_VOICES)} voices)…", flush=True)
    t0 = time.monotonic()
    pos_samples = await build_positives(kokoro, row["phrase"])
    print(f"  {len(pos_samples)} positives in {time.monotonic()-t0:.1f}s")

    print(f"building hard negatives ({len(HARD_NEG_PHRASES)} phrases)…", flush=True)
    t0 = time.monotonic()
    hard_neg_samples = await build_hard_negatives(kokoro)
    n_hard = sum(len(v) for v in hard_neg_samples.values())
    print(f"  {n_hard} hard negatives in {time.monotonic()-t0:.1f}s")

    print(f"sampling LibriSpeech windows (target {LIBRISPEECH_WINDOWS})…", flush=True)
    t0 = time.monotonic()
    rng = random.Random(args.seed)
    speech_samples = build_librispeech(rng)
    print(f"  {len(speech_samples)} LibriSpeech windows in {time.monotonic()-t0:.1f}s")

    # Channel B (--with-channel-b) — additional test pools.
    template_samples: list[np.ndarray] = []
    same_voice_samples: list[np.ndarray] = []
    real_recordings_total = 0
    if args.with_channel_b:
        if args.personal_templates_dir is not None:
            print(
                f"loading real recordings from {args.personal_templates_dir}…",
                flush=True,
            )
            t0 = time.monotonic()
            template_samples, same_voice_samples, real_recordings_total = (
                load_real_recordings(
                    args.personal_templates_dir,
                    n_templates=args.n_real_templates,
                    n_holdout=args.n_real_holdout,
                    seed=args.seed,
                )
            )
            print(
                f"  loaded {real_recordings_total} WAV(s); "
                f"using {len(template_samples)} as templates + "
                f"{len(same_voice_samples)} as same-voice holdout "
                f"in {time.monotonic()-t0:.1f}s"
            )
        else:
            print(
                f"building templates from {TEMPLATE_VOICE!r} "
                f"(×{len(TEMPLATE_SPEEDS)} speeds)…",
                flush=True,
            )
            t0 = time.monotonic()
            template_samples = await build_templates(kokoro, row["phrase"])
            print(f"  {len(template_samples)} templates in {time.monotonic()-t0:.1f}s")

            print(
                f"building same-voice held-out positives "
                f"(×{SAME_VOICE_HOLDOUT_COUNT})…",
                flush=True,
            )
            t0 = time.monotonic()
            same_voice_samples = await build_same_voice_holdout(kokoro, row["phrase"])
            print(f"  {len(same_voice_samples)} same-voice holdout in {time.monotonic()-t0:.1f}s")

    print(f"loading ONNX model …", flush=True)
    session = _load_session(row["model_path"])

    print("scoring …", flush=True)
    t0 = time.monotonic()
    pos_scores = [_score(session, s) for s in pos_samples]
    hard_neg_scores: dict[str, list[float]] = {
        phrase: [_score(session, s) for s in samples]
        for phrase, samples in hard_neg_samples.items()
    }
    speech_scores = [_score(session, s) for s in speech_samples]
    same_voice_scores: list[float] = (
        [_score(session, s) for s in same_voice_samples] if args.with_channel_b else []
    )
    print(
        f"  scored {len(pos_scores) + n_hard + len(speech_scores) + len(same_voice_scores)} "
        f"windows in {time.monotonic()-t0:.1f}s"
    )

    _print_table(pos_scores, hard_neg_scores, speech_scores)

    # ── Channel B pass ─────────────────────────────────────────────
    channel_b_payload: dict[str, Any] | None = None
    if args.with_channel_b:
        print()
        print("computing Channel B (DTW) distances …", flush=True)
        t0 = time.monotonic()
        if not template_samples or len(template_samples) < 2:
            print("  WARNING: insufficient templates — skipping Channel B leg")
        else:
            print(f"  feature extractor: {args.feature_extractor}")
            if args.feature_extractor != "mel":
                print(
                    "  (first call downloads checkpoint to "
                    "~/.cache/torch/hub — subsequent runs are instant)"
                )

            def feats(s):
                return features_for(s, args.feature_extractor)

            template_frames = [feats(s) for s in template_samples]
            # Calibrate against a small LibriSpeech slice (held back from
            # the eval pool — first 30 windows). This is the in-repo
            # proxy for "negative speech the user's room will see."
            calib_neg = [feats(s) for s in speech_samples[:30]]
            cal = calibrate_channel_b(template_frames, calib_neg)

            # Score every test bucket through Channel B.
            pos_other_dist = [
                best_template_distance(feats(s), template_frames)
                for s in pos_samples
            ]
            pos_same_dist = [
                best_template_distance(feats(s), template_frames)
                for s in same_voice_samples
            ]
            hard_dist_by_phrase: dict[str, list[float]] = {
                phrase: [
                    best_template_distance(feats(s), template_frames)
                    for s in samples
                ]
                for phrase, samples in hard_neg_samples.items()
            }
            # Use only the held-back portion of speech for FA reporting
            # so calibration set ≠ evaluation set.
            eval_speech_samples = speech_samples[30:]
            eval_speech_scores = speech_scores[30:]
            speech_dist = [
                best_template_distance(feats(s), template_frames)
                for s in eval_speech_samples
            ]
            print(f"  Channel B scoring done in {time.monotonic()-t0:.1f}s")

            _print_channel_b_table(
                cal,
                pos_same=pos_same_dist,
                pos_other=pos_other_dist,
                hard_neg=hard_dist_by_phrase,
                speech=speech_dist,
            )

            # Combinator tables at two operating points: the runtime
            # floor τ_crnn=0.65 + calibrated τ_b, and a looser τ_crnn=0.50.
            all_hard_dist = [d for ds in hard_dist_by_phrase.values() for d in ds]
            all_hard_scores = [s for v in hard_neg_scores.values() for s in v]
            buckets = {
                "positives (same-voice)": (same_voice_scores, pos_same_dist),
                "positives (other voices)": (pos_scores, pos_other_dist),
                "hard negatives (combined)": (all_hard_scores, all_hard_dist),
                "librispeech": (eval_speech_scores, speech_dist),
            }
            t_b = cal.tau if cal.tau is not None else CHANNEL_B_THRESHOLDS[2]
            _print_combinator_table(
                "STRICT (AND) — runtime floor",
                buckets, t_crnn=0.65, t_b=t_b,
                combiner=_combine_strict,
            )
            _print_combinator_table(
                "PERMISSIVE (OR) — runtime floor",
                buckets, t_crnn=0.65, t_b=t_b,
                combiner=_combine_permissive,
            )

            # Decision-gate report — the actual Phase 0 verdict.
            _print_consequences_of_strict(
                pos_other_scores=pos_scores,
                pos_same_scores=same_voice_scores,
                hard_scores=all_hard_scores,
                speech_scores=eval_speech_scores,
                pos_other_dist=pos_other_dist,
                pos_same_dist=pos_same_dist,
                hard_dist=all_hard_dist,
                speech_dist=speech_dist,
                cal=cal,
            )

            channel_b_payload = {
                "calibration": asdict(cal),
                "template_voice": TEMPLATE_VOICE,
                "template_speeds": list(TEMPLATE_SPEEDS),
                "n_templates": len(template_samples),
                "n_same_voice_holdout": len(same_voice_samples),
                "distances": {
                    "positives_same_voice": pos_same_dist,
                    "positives_other_voices": pos_other_dist,
                    "hard_negatives": hard_dist_by_phrase,
                    "librispeech": speech_dist,
                },
                "thresholds": CHANNEL_B_THRESHOLDS,
            }

    # Archive JSON for cross-run comparison.
    out_dir = Path(settings.data_dir) / "wake_word_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{args.avatar_id}-v{row['version']}-{stamp}.json"
    archive: dict[str, Any] = {
        "avatar_id": row["avatar_id"],
        "phrase": row["phrase"],
        "version": row["version"],
        "model_path": row["model_path"],
        "train_metrics": m,
        "eval_params": {
            "seed": args.seed,
            "positives_per_voice": POSITIVES_PER_VOICE,
            "hard_neg_per_voice": HARD_NEG_PER_VOICE,
            "eval_voices": EVAL_VOICES,
            "hard_neg_phrases": HARD_NEG_PHRASES,
            "librispeech_windows_target": LIBRISPEECH_WINDOWS,
            "thresholds": THRESHOLDS,
            "with_channel_b": args.with_channel_b,
        },
        "scores": {
            "positives": pos_scores,
            "hard_negatives": hard_neg_scores,
            "librispeech": speech_scores,
            "positives_same_voice": same_voice_scores,
        },
        "summary": {
            "positives": _describe(pos_scores),
            "hard_negatives_combined": _describe(
                [s for v in hard_neg_scores.values() for s in v]
            ),
            "hard_negatives_per_phrase": {
                phrase: _describe(scores)
                for phrase, scores in hard_neg_scores.items()
            },
            "librispeech": _describe(speech_scores),
            "positives_same_voice": _describe(same_voice_scores),
        },
    }
    if channel_b_payload is not None:
        archive["channel_b"] = channel_b_payload
    out_path.write_text(json.dumps(archive, indent=2))
    print()
    print(f"archived to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
