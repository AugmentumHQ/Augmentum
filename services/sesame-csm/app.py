"""Sesame CSM TTS sidecar — OpenAI-compatible streaming TTS for Augmentum.

Wraps the davidbrowne17/csm-streaming fork (which adds incremental Mimi
decoding to Sesame's CSM-1B) behind the same ``POST /v1/audio/speech``
contract every other Augmentum TTS provider speaks, so the main app
registers it as a drop-in provider with zero special-casing.

Why a sidecar (not in-process): CSM pins torch==2.4.0 + CUDA and needs a
GPU; isolating it in its own container keeps it off the main image's
dependency graph and — on a multi-GPU or dedicated-box setup — off the
LLM's VRAM entirely. This is the fabric-friendly shape: an isolated
instance on a port.

What CSM gives that Pocket/Kokoro can't: prosody conditioned on the
*conversation so far*. The OpenAI speech contract has no slot for
conversation history, so we carry it statefully here — keyed by an
``X-Augmentum-Session`` header, the sidecar keeps a short rolling buffer
of what IT just said and feeds it back as CSM context. That's the
conversational-continuity essence, delivered without changing the wire
contract. Cross-speaker context (priming the *user's* audio) is a later
phase; v1 is self-context + voice cloning.

Endpoints (OpenAI-compatible, mirrors services/chatterbox-turbo/app.py):
  GET  /health              — Docker healthcheck
  GET  /v1/models           — model listing for the UI picker
  GET  /v1/voices           — list cloned-voice references
  POST /v1/voices           — upload a reference clip (+ transcript) to clone
  POST /v1/audio/speech     — synthesize; streams WAV/PCM, buffers mp3/opus

Streaming: ``generate_stream()`` yields CPU float32 tensors at 24 kHz in
~1.6s chunks (no watermark on this path — see README). For wav/pcm we
emit a sentinel-header WAV and stream PCM frames as they're produced
(first audio in ~one chunk instead of after full synthesis). For
compressed formats we buffer + ffmpeg-encode once, since you can't
re-encode a chunk you've already flushed.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import threading
import gc
import time
import wave
from collections import deque
from typing import Any

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

# The streaming fork is cloned to /opt/csm by the Dockerfile and added to
# PYTHONPATH. It exposes load_csm_1b(), Segment, and Generator.generate_stream.
from generator import Segment, load_csm_1b  # type: ignore[import-not-found]
from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

# torch.compile (inductor) is how the streaming fork hits real-time, but it
# fails hard on marginal setups — no C compiler, or bf16 kernels on a Turing
# GPU (RTX 20-series, e.g. 2080 Super, which has no native bf16). Letting
# dynamo fall back to eager turns those fatal crashes into "slower but works"
# (the fix the crash message itself recommends). On capable GPUs compilation
# still succeeds, so this only changes behavior where it would otherwise die.
try:
    import torch._dynamo  # noqa: E402
    torch._dynamo.config.suppress_errors = True
except Exception:  # noqa: BLE001 — older torch without _dynamo config
    pass

# The fork compiles with mode='reduce-overhead' (CUDA graphs). CUDA-graph
# *trees* keep per-thread state in TLS, but the sidecar warms up on the load
# thread and serves each generation on a different (threadpool) thread — so
# the request thread trips `assert _is_key_in_tls(...)` and the stream dies
# after the WAV header. Disabling cudagraph_trees keeps inductor's kernel
# fusion (the bulk of the speedup) while making generation thread-safe.
try:
    import torch._inductor.config as _ind_cfg  # noqa: E402
    _ind_cfg.triton.cudagraph_trees = False
except Exception:  # noqa: BLE001 — older torch without this knob
    pass

DEVICE = os.environ.get("DEVICE", "cuda")
PORT = int(os.environ.get("PORT", "8920"))
# Force the model to fp16 when set ("fp16"). CSM loads in bf16, which Turing
# GPUs don't support natively — fp16 is the supported 16-bit path there and
# avoids slow bf16 emulation. Opt-in (default keeps bf16) since it can't be
# verified per-card from here. Set CSM_DTYPE=fp16 on an RTX 20-series box.
CSM_DTYPE = os.environ.get("CSM_DTYPE", "").lower()
VOICE_DIR = os.environ.get("VOICE_LIBRARY_DIR", "/voices")
# Load a FINE-TUNED voice (Sesame-format, patched by the training harness'
# bridge_to_sesame.py) instead of base CSM-1B. The dir must contain
# model.safetensors + config.json. We redirect the fork's hardcoded
# Model.from_pretrained("sesame/csm-1b") to this dir, reusing all of
# load_csm_1b's real-time machinery (compile, Mimi, warmup) unchanged.
CSM_MODEL_DIR = os.environ.get("CSM_MODEL_DIR", "").strip()
# A fine-tuned voice is single-speaker by default → skip dialogue context (it
# wasn't trained on it and runs away). Set CSM_DIALOGUE=1 for a voice trained
# dialogue-aware (emotion-paired two-speaker data) so it KEEPS cross-speaker
# context and reacts to how the user sounded.
CSM_DIALOGUE = os.environ.get("CSM_DIALOGUE", "").strip().lower() not in ("", "0", "false", "no")
SAMPLE_RATE = 24_000


def _redirect_model_to_local() -> None:
    """Point the fork's base-weights load at CSM_MODEL_DIR (a fine-tuned voice).
    Only the 'sesame/csm-1b' load is redirected; the Mimi codec (kyutai/mimi,
    loaded separately by Generator) is untouched. No-op if CSM_MODEL_DIR unset."""
    if not CSM_MODEL_DIR:
        return
    try:
        import models as _csm_models  # the fork's module (on PYTHONPATH=/opt/csm)
        _orig = _csm_models.Model.from_pretrained.__func__
        def _local(cls, repo_id, *a, **k):  # noqa: ANN001
            target = CSM_MODEL_DIR if repo_id == "sesame/csm-1b" else repo_id
            return _orig(cls, target, *a, **k)
        _csm_models.Model.from_pretrained = classmethod(_local)
        print(f"[csm] fine-tuned voice: loading weights from {CSM_MODEL_DIR}", flush=True)
    except Exception as exc:  # noqa: BLE001 — fall back to base on any import drift
        print(f"[csm] CSM_MODEL_DIR redirect failed ({exc}); using base CSM-1B", flush=True)
# CSM is a dialogue model: it conditions prosody on BOTH speakers' recent
# turns. We keep two budgeted buffers per session — her own sentences
# (speaker 0) and the user's turns (speaker 1, pushed via
# /v1/context/user_turn). Two buffers, not one, so her multi-sentence
# replies can't evict the single user turn, and so the identity balance
# stays safe (mostly her anchor + sentences, one fresh user reference).
# Bounded to stay well under CSM's 923-token context cap; each segment's
# audio is trimmed before it enters context.
DEFAULT_SPEAKER = 0          # her voice
USER_SPEAKER = 1             # the person she's talking to
MAX_SELF_TURNS = int(os.environ.get("CSM_SELF_TURNS", "2"))
MAX_USER_TURNS = int(os.environ.get("CSM_USER_TURNS", "1"))
MAX_CTX_AUDIO_S = 8.0        # her sentences (anchored, can be longer)
MAX_USER_CTX_AUDIO_S = 6.0   # user turns (kept shorter — less identity bleed)
# Re-inject the clone anchor on every turn (default). CSM drifts off a
# cloned identity without it, and that re-anchoring is what keeps her
# voice locked now that the user's audio shares the context. Inverse
# (CSM_REANCHOR=0): anchor only at the start of a session and let her own
# recent sentences carry identity thereafter — cheaper on context budget.
REANCHOR_EVERY_TURN = os.environ.get("CSM_REANCHOR", "1") not in ("0", "false", "False")
# Release GPU VRAM after this many idle seconds; the model lazy-reloads
# from the local HF cache on the next request (or a /warmup ping). An idle
# CSM otherwise pins ~3-4GB 24/7 for a few seconds of real use per
# conversation. 0 = never unload (trade VRAM for zero cold-start).
IDLE_UNLOAD_S = float(os.environ.get("CSM_IDLE_UNLOAD_S", "90"))

os.makedirs(VOICE_DIR, exist_ok=True)


def _force_fp16_load() -> None:
    """Opt-in (CSM_DTYPE=fp16) patch so the fork LOADS in fp16 on Turing.

    ``load_csm_1b`` picks its dtype as ``bfloat16 if
    torch.cuda.is_bf16_supported() else float16``. On Turing (RTX 20-series)
    that probe returns True — CUDA advertises bf16 via slow emulation — so the
    model AND the KV cache it allocates in ``setup_caches`` come up bf16.
    Pinning the probe False makes the fork pick fp16 for both weights and
    cache at the source, which is cleaner than converting after the fact (and
    leaves Mimi, a separate module, on its own dtype). ``_coerce_dtype``
    remains a backstop; the autocast redirect below is the other half."""
    if CSM_DTYPE not in ("fp16", "float16", "half"):
        return
    torch.cuda.is_bf16_supported = lambda *a, **k: False  # type: ignore[assignment]
    print("[csm] CSM_DTYPE=fp16: forcing fp16 load (Turing path)", flush=True)


# Install before any load_csm_1b call (lazy load happens in _get_generator).
_force_fp16_load()


_autocast_patched = False


def _match_autocast_to_model(gen: Any) -> None:
    """Redirect the fork's HARDCODED bf16 autocast to fp16 when the model
    actually loaded in fp16. This is the real fix for the
    "Index put ... got Half for the destination and BFloat16 for the source"
    crash, and it must NOT be gated on CSM_DTYPE: a genuine Turing card can
    report ``is_bf16_supported() == False`` on its own, so the fork loads
    fp16 weights + fp16 KV cache without any env var — then its hardcoded
    ``torch.autocast(dtype=torch.bfloat16)`` in ``generate_stream`` casts the
    attention k/v projections back to bf16, which can't be index-put into the
    fp16 cache. Keying off the real param dtype covers both the env-forced and
    the auto-detected fp16 paths, and leaves pure-bf16 boxes untouched.

    Idempotent and process-global (single-purpose sidecar). Wrapped, not
    subclassed, so we don't depend on torch.autocast's C-level internals; the
    return value is still the context manager used as ``with ...:``."""
    global _autocast_patched
    if _autocast_patched:
        return
    try:
        model = getattr(gen, "_model", None)
        model_dtype = next(model.parameters()).dtype if model is not None else None
    except Exception:  # noqa: BLE001 — best-effort; never break a working load
        model_dtype = None
    if model_dtype != torch.float16:
        return  # bf16 model — its hardcoded bf16 autocast is already correct

    _orig_autocast = torch.autocast

    def _fp16_autocast(*args, **kwargs):  # noqa: ANN002,ANN003
        if kwargs.get("dtype") == torch.bfloat16:
            kwargs["dtype"] = torch.float16
        elif len(args) >= 2 and args[1] == torch.bfloat16:
            args = (args[0], torch.float16, *args[2:])
        return _orig_autocast(*args, **kwargs)

    torch.autocast = _fp16_autocast  # type: ignore[assignment]
    _autocast_patched = True
    print("[csm] model loaded in fp16 — redirecting fork's bf16 autocast to fp16", flush=True)


# CSM ships no built-in named voices (it clones from references), so a
# fresh install would otherwise show an EMPTY voice picker and be
# unusable until the user uploads a clone. We always expose CSM's two
# bundled example voices (downloaded on first use from sesame/csm-1b) so
# there's something selectable out of the box, alongside any uploads.
# Transcripts are the canonical prompt texts from the CSM repo.
_BUNDLED_VOICES: dict[str, dict] = {
    "conversational_a": {
        "file": "prompts/conversational_a.wav",
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because "
            "I'd start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
    },
    "conversational_b": {
        "file": "prompts/conversational_b.wav",
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
    },
}

app = FastAPI(title="Sesame CSM TTS (Augmentum sidecar)")

# CSM inference is batch=1 and not thread-safe — serialize like the
# in-process Pocket engine does. A request queue would be the move if
# throughput ever matters; for a personal companion, a lock is correct.
_gen_lock = threading.Lock()
_generator: Any = None
_load_error: str = ""
_load_failed_at: float = 0.0  # monotonic time of last failed load
_last_activity: float = 0.0  # monotonic time of last synth — drives idle unload

# After a failed load, don't re-attempt for this long. The fork's
# "maximum-intensity warmup" puts several GB of weights + audio contexts
# on the GPU BEFORE it can crash (e.g. missing C compiler for inductor),
# and CUDA's caching allocator holds that memory for the life of the
# process. Re-attempting on every request would strand more VRAM each
# time. A deterministic failure (bad image) stays broken until rebuild,
# so retrying sooner buys nothing — surface the cached error instead.
LOAD_RETRY_COOLDOWN_S = 60.0

# session_id -> deque[Segment]. Two buffers: her own recent generations
# (speaker 0) and the user's recent turns (speaker 1). In-memory only; a
# fresh container starts every session cold, which is fine — continuity
# is a within-conversation property, and keeping the user's voice off disk
# is the right privacy posture.
_self_turns: dict[str, deque] = {}
_user_turns: dict[str, deque] = {}
_sessions_lock = threading.Lock()

# ── performance caches + observability (Phase 1) ──────────────────────
# Mimi-encode cache for context audio. The fork re-encodes EVERY context
# segment (clone anchor + user turn + her recent sentences) through Mimi on
# EVERY /v1/audio/speech call — and the main app calls us once per SENTENCE,
# so a 4-sentence reply re-encodes the same ~30s of anchor+context audio 4×.
# This is the XTTS `get_conditioning_latents` pattern (compute the speaker
# conditioning once, cache it, reuse across generations) ported to CSM:
# encode once, key by audio content, reuse. Encoded codes are tiny (frames ×
# codebooks of ints), so holding them costs ~nothing. Cleared on unload (the
# codes belong to that model instance's Mimi).
_CTX_CACHE_MAX = 64
_ctx_token_cache: dict[str, Any] = {}
_ctx_cache_hits = 0
_ctx_cache_misses = 0

# Loaded/resampled clone-anchor Segments, keyed by base voice name, so we
# skip torchaudio.load + resample + trim on every turn. Bonus: the anchor's
# audio tensor becomes a STABLE object, so its Mimi codes stay cache-hot in
# _ctx_token_cache above.
_anchor_cache: dict[str, Any] = {}

# Last-generation + warmup performance, surfaced on /health so we can SEE
# whether the fp16 compile / CUDA-graph fast path actually engaged on this
# GPU (RTF < 1 == faster than real-time). The dtype fix only *unblocks* the
# fast path; these numbers confirm it's live.
_compiled_wrapper = False        # is backbone a torch.compile OptimizedModule
_warmup_rtf: float | None = None  # steady-state wall_s / audio_s from warm-compile
_last_rtf: float | None = None    # most recent real generation (end-to-end)
_last_gen_audio_s: float | None = None


def _audio_key(audio: torch.Tensor) -> str:
    """Content key for a context-audio tensor. blake2b over the raw bytes is
    sub-millisecond for our ≤8s clips and collision-safe — far cheaper than
    the Mimi encode it guards. Shape folded in to disambiguate trims."""
    a = audio.detach().to("cpu", torch.float32).contiguous()
    h = hashlib.blake2b(a.numpy().tobytes(), digest_size=16)
    h.update(repr(tuple(a.shape)).encode())
    return h.hexdigest()


def _clone_codes(x: Any) -> Any:
    """Return a private copy of cached Mimi codes so a downstream in-place op
    can't corrupt the cache. Codes are tiny, so this clone is ~free."""
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, tuple):
        return tuple(_clone_codes(e) for e in x)
    return x


def _install_context_cache(gen: Any) -> None:
    """Wrap the generator's ``_tokenize_audio`` with a content-keyed cache so
    unchanged context segments aren't re-encoded through Mimi every call.

    The single biggest per-sentence latency win: the clone anchor is identical
    every turn (re-anchor mode) and her recent sentences repeat across the
    sentences of one reply, yet the fork re-encodes all of them each time.
    Best-effort — if the fork's internals drift (no ``_tokenize_audio``), we
    log and leave generation untouched (slower, still correct)."""
    orig = getattr(gen, "_tokenize_audio", None)
    if orig is None:
        print("[csm] context cache: _tokenize_audio not found — skipping "
              "(slower, still correct)", flush=True)
        return
    if getattr(orig, "_aug_cached", False):
        return  # idempotent across reloads

    def cached(audio: torch.Tensor):
        global _ctx_cache_hits, _ctx_cache_misses
        try:
            key = _audio_key(audio)
        except Exception:  # noqa: BLE001 — never break synth over a cache key
            return orig(audio)
        hit = _ctx_token_cache.get(key)
        if hit is not None:
            _ctx_cache_hits += 1
            return _clone_codes(hit)
        _ctx_cache_misses += 1
        out = orig(audio)
        if len(_ctx_token_cache) < _CTX_CACHE_MAX:
            _ctx_token_cache[key] = out
        return out

    cached._aug_cached = True  # type: ignore[attr-defined]
    gen._tokenize_audio = cached
    print("[csm] context cache installed (Mimi-encode memoized by audio content)", flush=True)


def _coerce_dtype(gen: Any) -> None:
    """Backstop to ``_install_fp16_patches`` (which already forces the fork to
    load in fp16): if anything still came back bf16, retarget ONLY those bf16
    tensors to fp16 — that's the Llama backbone, the part Turing can't run in
    bf16. With the load-time patch in place this normally finds nothing.

    Critically we leave fp32 modules alone: the Mimi audio codec runs in fp32
    on fp32 audio input, and a blanket ``.half()`` makes its conv weights fp16
    while the audio stays fp32 → "Input type (float) and bias type (Half)".
    The backbone and Mimi only exchange integer codes, so mixed precision
    across them is safe. Best-effort; never break a working load."""
    if CSM_DTYPE not in ("fp16", "float16", "half"):
        return
    import torch.nn as nn
    n = 0
    for val in list(vars(gen).values()):
        if not isinstance(val, nn.Module):
            continue
        for p in val.parameters(recurse=True):
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
                n += 1
        for b in val.buffers(recurse=True):
            if b.dtype == torch.bfloat16:
                b.data = b.data.to(torch.float16)
                n += 1
    print(f"[csm] CSM_DTYPE=fp16: retargeted {n} bf16 tensors to fp16 (fp32 left as-is)", flush=True)


def _get_generator() -> Any:
    """Lazy-load CSM-1B on first request (or after an idle unload).
    ~3-4GB VRAM in bf16; reload reads the local HF cache (no network)."""
    global _generator, _load_error, _last_activity, _load_failed_at, _compiled_wrapper
    if _generator is not None:
        return _generator
    with _gen_lock:
        if _generator is not None:
            return _generator
        # Short-circuit a recently-failed load (see LOAD_RETRY_COOLDOWN_S)
        # so a broken image doesn't re-leak GPU memory on every request.
        if _load_error and (time.monotonic() - _load_failed_at) < LOAD_RETRY_COOLDOWN_S:
            raise RuntimeError(f"CSM load failed recently: {_load_error}")
        err = ""
        try:
            t = time.monotonic()
            _redirect_model_to_local()   # fine-tuned voice if CSM_MODEL_DIR set
            _generator = load_csm_1b(device=DEVICE)
            _coerce_dtype(_generator)
            # If the model is fp16 (env-forced or a Turing card that probed
            # bf16-unsupported on its own), match the fork's hardcoded bf16
            # autocast to fp16 so attention keys fit the fp16 KV cache.
            _match_autocast_to_model(_generator)
            # Memoize Mimi-encoding of context audio (XTTS-style) — biggest
            # per-sentence latency win once a conversation has context.
            _install_context_cache(_generator)
            # Record whether the backbone is a torch.compile wrapper (the
            # fast path). This only proves it's *wrapped*; the warm-compile
            # RTF on /health is the real "did it actually go fast" signal.
            try:
                bb = getattr(getattr(_generator, "_model", None), "backbone", None)
                _compiled_wrapper = type(bb).__name__ == "OptimizedModule"
            except Exception:  # noqa: BLE001
                _compiled_wrapper = False
            _load_error = ""
            _last_activity = time.monotonic()
            print(f"[csm] model loaded in {time.monotonic() - t:.1f}s on {DEVICE}", flush=True)
        except Exception as exc:  # noqa: BLE001 — surface at /health + request time
            err = str(exc)
        # The except block has ended, so ``exc`` (and the traceback frames
        # pinning the partially-loaded GPU tensors) are released — only now
        # can empty_cache actually reclaim that VRAM.
        if err:
            _generator = None
            _load_error = err
            _load_failed_at = time.monotonic()
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — empty_cache is best-effort
                pass
            print(f"[csm] load failed (GPU freed): {err}", flush=True)
            raise RuntimeError(f"CSM load failed: {err}")
    return _generator


def _touch() -> None:
    """Mark synthesis activity so the idle watcher doesn't unload us."""
    global _last_activity
    _last_activity = time.monotonic()


def _unload_generator() -> None:
    """Drop the model and release GPU VRAM. An in-flight generation holds
    its own local ref (and _gen_lock), so this can't free a model mid-use —
    it only clears the global; the next request lazy-reloads."""
    global _generator, _compiled_wrapper
    with _gen_lock:
        if _generator is None:
            return
        _generator = None
    # The cached Mimi codes + anchor segments belong to THIS model instance;
    # a reload rebuilds Mimi, so stale codes would be invalid. Drop them.
    _ctx_token_cache.clear()
    _anchor_cache.clear()
    _compiled_wrapper = False
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — empty_cache is best-effort
        pass
    print("[csm] idle — model unloaded, GPU VRAM released", flush=True)


# When pinned, the idle watcher won't unload — hold the (slow-to-reload)
# model resident while actively testing, then unpin to let it idle out again.
_pinned: bool = False


def _idle_watcher() -> None:
    """Background: unload after IDLE_UNLOAD_S of no synthesis (unless pinned)."""
    while True:
        time.sleep(15)
        if IDLE_UNLOAD_S <= 0 or _generator is None or _pinned:
            continue
        if time.monotonic() - _last_activity > IDLE_UNLOAD_S:
            _unload_generator()


def _warm_compile() -> None:
    """Load, then run two short generations so torch.compile + CUDA-graph
    capture AND the Mimi DECODE path all happen here, off the request path.

    Orpheus's production lesson: the codec decoder (SNAC for them, Mimi for
    us) is THE latency bottleneck to warm — warming only the backbone leaves
    the first real sentence paying decode-path capture. So we run a real
    generate that produces a decoded chunk. The first pass eats the (slow,
    one-time) inductor compile; the SECOND pass measures steady-state RTF,
    which we surface on /health to confirm the fp16 fast path engaged."""
    global _warmup_rtf
    gen = _get_generator()
    last_rtf: float | None = None
    for i in range(2):  # pass 0 compiles/captures; pass 1 measures steady state
        samples = 0
        t = time.monotonic()
        try:
            with _gen_lock:
                for chunk in gen.generate_stream(
                    text="Warming up the voice engine now.",
                    speaker=DEFAULT_SPEAKER, context=[],
                    temperature=0.7, topk=30,
                ):
                    samples += int(chunk.detach().reshape(-1).shape[0])
                    if samples >= SAMPLE_RATE:  # ~1s decoded is enough to measure
                        break
        except Exception as exc:  # noqa: BLE001 — surfaced via /health
            print(f"[csm] warm-compile pass {i} failed: {exc}", flush=True)
            return
        audio_s = samples / SAMPLE_RATE
        if audio_s > 0:
            last_rtf = round((time.monotonic() - t) / audio_s, 3)
    if last_rtf:
        _warmup_rtf = last_rtf
        print(f"[csm] warm-compile done: RTF {last_rtf} "
              f"({1 / last_rtf:.2f}× real-time)", flush=True)
    _touch()


def _safe_warm() -> None:
    try:
        _warm_compile()
    except Exception as exc:  # noqa: BLE001 — surfaced via /health
        print(f"[csm] warmup failed: {exc}", flush=True)


# ── voice library (cloning references) ────────────────────────────────
# A cloned voice = a reference WAV + its transcript. CSM needs the text
# (unlike Chatterbox) because the clone anchor is a (text, audio) Segment.


# Every voice CSM exposes is tagged with this suffix so the unified
# Augmentum picker distinguishes "matt" (the Chatterbox/Pocket clone of a
# shared reference) from "matt-csm" (CSM cloning the same source). The tag
# is stripped before resolving the underlying file.
CSM_TAG = "-csm"


def _base_name(voice: str) -> str:
    """Strip the -csm tag the UI round-trips back to us."""
    v = (voice or "").strip()
    return v[: -len(CSM_TAG)] if v.endswith(CSM_TAG) else v


def _voice_paths(name: str) -> tuple[str, str]:
    name = _base_name(name)
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "voice"
    return os.path.join(VOICE_DIR, f"{safe}.wav"), os.path.join(VOICE_DIR, f"{safe}.txt")


# The shared /data/voices store holds clones saved by the main app, which
# preserves the original extension (.wav/.mp3/...). Match all of them.
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")


def _find_voice_file(base: str) -> str | None:
    """Locate a shared clone audio file for a base name, any extension."""
    for ext in _AUDIO_EXTS:
        p = os.path.join(VOICE_DIR, f"{base}{ext}")
        if os.path.isfile(p):
            return p
    return None


def _voice_name() -> str:
    """Display name for the loaded fine-tuned voice (CSM_VOICE_NAME, else the
    model dir basename with -sesame stripped)."""
    n = os.environ.get("CSM_VOICE_NAME", "").strip()
    if n:
        return n
    if CSM_MODEL_DIR:
        b = os.path.basename(CSM_MODEL_DIR.rstrip("/\\")).replace("-sesame", "")
        return b if b and b != "model" else "csm"
    return "csm"


def _list_voices() -> list[dict]:
    """Advertise voices WITH capability metadata — the handshake that lets
    Augmentum tailor handling per voice (kind/emotion/dialogue) instead of
    treating CSM like a generic provider.

    A FINE-TUNED model (CSM_MODEL_DIR) IS one voice — advertise exactly that
    one (no gated bundled clips, which 401 on preview). Its `kind` says whether
    it keeps dialogue context (CSM_DIALOGUE) or is single-speaker."""
    if CSM_MODEL_DIR:
        nm = _voice_name()
        return [{
            "id": nm + CSM_TAG, "name": nm + CSM_TAG,
            "finetuned": True,
            "kind": "dialogue" if CSM_DIALOGUE else "single",
            "emotion_tags": True,        # fine-tuned voices respond to (emotion) prefixes
            "speakers": [0],
            "stream": "frame",           # streams audio frames (gapless playback fits)
        }]
    # Base CSM: bundled examples + shared clones, all clone-conditioned (dialogue-capable).
    seen: set[str] = set()
    out: list[dict] = []
    for name in _BUNDLED_VOICES:
        out.append({"id": name + CSM_TAG, "name": name + CSM_TAG, "bundled": True,
                    "kind": "base-clone", "emotion_tags": False, "stream": "frame"})
        seen.add(name)
    for fn in sorted(os.listdir(VOICE_DIR)):
        base, ext = os.path.splitext(fn)
        if ext.lower() in _AUDIO_EXTS and base not in seen:
            out.append({"id": base + CSM_TAG, "name": base + CSM_TAG, "cloned": True,
                        "kind": "base-clone", "emotion_tags": False, "stream": "frame"})
            seen.add(base)
    return out


def _load_clone_segment(voice: str) -> Segment | None:
    """Build the cloning anchor Segment for a named voice, or None.

    Resolution order: an uploaded clone in /voices wins; otherwise a
    bundled CSM example voice (downloaded from sesame/csm-1b on first use
    and cached). Returns None for an unknown name (generation then runs
    with no clone anchor — CSM's default speaker)."""
    base = _base_name(voice)
    if not base:
        return None
    cached = _anchor_cache.get(base)
    if cached is not None:
        return cached
    transcript = ""
    # Shared clone in /data/voices (any extension) wins; its transcript —
    # written by the main app's clone flow as <name>.txt — makes CSM clone
    # well. Falls back to a bundled example voice from sesame/csm-1b.
    wav_path = _find_voice_file(base)
    if wav_path is not None:
        txt_path = os.path.join(VOICE_DIR, f"{base}.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                transcript = f.read().strip()
    else:
        spec = _BUNDLED_VOICES.get(base)
        if spec is None:
            return None
        wav_path = hf_hub_download(repo_id="sesame/csm-1b", filename=spec["file"])
        transcript = spec["text"]
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(dim=0)  # mono
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = _trim_audio(audio, MAX_CTX_AUDIO_S)
    # An empty transcript still clones (weaker); CSM tolerates it.
    seg = Segment(speaker=DEFAULT_SPEAKER, text=transcript, audio=audio)
    _anchor_cache[base] = seg  # stable object → stays Mimi-encode cache-hot
    return seg


def _trim_audio(audio: torch.Tensor, max_s: float) -> torch.Tensor:
    max_n = int(max_s * SAMPLE_RATE)
    return audio[-max_n:] if audio.shape[-1] > max_n else audio


def _decode_to_tensor(raw: bytes) -> torch.Tensor:
    """Decode arbitrary uploaded audio (WAV/PCM/webm/…) to a 24 kHz mono
    float tensor via ffmpeg, matching the clone-anchor decode path. Used
    for the user's STT clip on the cross-speaker context channel."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", "pipe:1"],
        input=raw, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(400, "could not decode user-turn audio")
    audio, sr = torchaudio.load(io.BytesIO(proc.stdout))
    audio = audio.mean(dim=0)  # mono
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio


# ── request model (OpenAI-compatible) ─────────────────────────────────


class TTSRequest(BaseModel):
    model: str = ""
    input: str = ""
    voice: str = ""
    response_format: str = "wav"
    speed: float = 1.0
    instructions: str = ""  # accepted for parity; CSM has no instruct slot


# ── context assembly ──────────────────────────────────────────────────


def _build_context(session_id: str, voice: str) -> list[Segment]:
    """Assemble CSM's conditioning context for this turn:
    ``[clone_anchor?, user_turn(s) (spk1), her_recent_sentences (spk0)]``.

    Chronological within an exchange — the user spoke, then she responds —
    so prosody conditions on how the user just sounded *and* on her own
    flow. The anchor leads so it re-anchors identity (see REANCHOR_EVERY_TURN)."""
    # A FINE-TUNED voice (CSM_MODEL_DIR) is single-speaker — it was NOT trained
    # on dialogue/cross-speaker context, so feeding it the user's audio (or a
    # clone anchor, or accumulated turns) pushes it off-distribution and it runs
    # away into grinding artifacts. Generate with no context — the model IS the
    # voice. (Base CSM, and a CSM_DIALOGUE-flagged dialogue-aware fine-tune, keep
    # the full context below.)
    if CSM_MODEL_DIR and not CSM_DIALOGUE:
        return []
    ctx: list[Segment] = []
    self_buf: list[Segment] = []
    user_buf: list[Segment] = []
    if session_id:
        with _sessions_lock:
            sb = _self_turns.get(session_id)
            ub = _user_turns.get(session_id)
            self_buf = list(sb) if sb else []
            user_buf = list(ub) if ub else []

    # A dialogue-aware FINE-TUNE was trained on exactly [one user turn, her
    # reply] — no self-turns, no anchor. Feed ONLY the most recent user turn so
    # we stay in-distribution; accumulating her own prior replies (self-turns)
    # over a conversation pushes her off-distribution and she degrades into
    # trailing artifacts on later messages. (Base CSM keeps the full context.)
    if CSM_MODEL_DIR and CSM_DIALOGUE:
        return user_buf[-1:]

    # Anchor: every turn by default; "once" mode injects it only at the
    # very start of a session (no self-turns banked yet) and lets her own
    # sentences carry identity from there.
    if REANCHOR_EVERY_TURN or not self_buf:
        anchor = _load_clone_segment(voice)
        if anchor is not None:
            ctx.append(anchor)

    ctx.extend(user_buf)
    ctx.extend(self_buf)
    return ctx


def _remember_turn(session_id: str, text: str, audio: torch.Tensor) -> None:
    """Bank one of HER synthesized sentences (speaker 0) as context."""
    if not session_id:
        return
    seg = Segment(speaker=DEFAULT_SPEAKER, text=text,
                  audio=_trim_audio(audio.detach().cpu(), MAX_CTX_AUDIO_S))
    with _sessions_lock:
        buf = _self_turns.setdefault(session_id, deque(maxlen=MAX_SELF_TURNS))
        buf.append(seg)


def _remember_user_turn(session_id: str, text: str, audio: torch.Tensor) -> None:
    """Bank the USER's spoken turn (speaker 1) as cross-speaker context, so
    her next reply's prosody reacts to how they actually sounded — not just
    the words. Audio is the resampled 24 kHz mono clip from STT."""
    if not session_id:
        return
    seg = Segment(speaker=USER_SPEAKER, text=text,
                  audio=_trim_audio(audio.detach().cpu(), MAX_USER_CTX_AUDIO_S))
    with _sessions_lock:
        buf = _user_turns.setdefault(session_id, deque(maxlen=MAX_USER_TURNS))
        buf.append(seg)


# ── audio encoding helpers ────────────────────────────────────────────


def _pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _wav_header_streaming() -> bytes:
    """44-byte WAV header with sentinel sizes for live streaming — the
    client reads PCM until the stream closes (mirrors Augmentum's own
    _wav_header_streaming pattern)."""
    import struct
    sentinel = 0x7FFFFFFF
    byte_rate = SAMPLE_RATE * 2
    return (
        b"RIFF" + struct.pack("<I", sentinel) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16)
        + b"data" + struct.pack("<I", sentinel)
    )


def _encode_compressed(pcm: bytes, fmt: str) -> tuple[bytes, str]:
    """Buffer→ffmpeg for mp3/opus/aac/flac (no streaming for these)."""
    codec_args = {
        "mp3": (["-f", "mp3", "-b:a", "128k"], "audio/mpeg"),
        "opus": (["-f", "opus", "-b:a", "96k"], "audio/opus"),
        "aac": (["-f", "adts", "-b:a", "128k"], "audio/aac"),
        "flac": (["-f", "flac"], "audio/flac"),
    }
    args, mime = codec_args.get(fmt, codec_args["mp3"])
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         *args, "pipe:1"],
        input=pcm, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg encode failed: {proc.stderr[:200]!r}")
    return proc.stdout, mime


# ── streaming generation ──────────────────────────────────────────────


def _stream_pcm_chunks(text: str, context: list[Segment], session_id: str):
    """Yield PCM16 bytes per generated chunk, holding the inference lock
    for the whole stream (batch=1, not thread-safe). Accumulates the full
    audio so we can remember the turn for next-turn context."""
    global _last_rtf, _last_gen_audio_s
    gen = _get_generator()
    collected: list[torch.Tensor] = []
    t0 = time.monotonic()
    with _gen_lock:
        for chunk in gen.generate_stream(
            text=text,
            speaker=DEFAULT_SPEAKER,
            context=context,
            temperature=0.7,
            topk=30,
        ):
            t = chunk.detach().cpu().reshape(-1)
            collected.append(t)
            _touch()  # keep activity fresh so a long stream isn't unloaded
            yield _pcm16(t.numpy())
    if collected:
        full = torch.cat(collected)
        _remember_turn(session_id, text, full)
        # End-to-end RTF (includes client drain between yields) — a rough live
        # health signal; the warm-compile RTF is the pure-generation number.
        audio_s = full.shape[-1] / SAMPLE_RATE
        if audio_s > 0:
            _last_rtf = round((time.monotonic() - t0) / audio_s, 3)
            _last_gen_audio_s = round(audio_s, 2)


# ── endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    ok = _generator is not None
    return {"status": "ok" if (ok or not _load_error) else "error",
            "model": "sesame-csm-1b", "device": DEVICE,
            "loaded": ok,
            "idle_unload_s": IDLE_UNLOAD_S,
            "idle_for_s": round(time.monotonic() - _last_activity, 1) if _last_activity else None,
            "error": _load_error or None,
            # Phase 1 observability — is the fp16 fast path live, and how fast?
            "dtype": CSM_DTYPE or "bf16",
            "compiled": _compiled_wrapper,
            "warmup_rtf": _warmup_rtf,
            "warmup_x_realtime": round(1 / _warmup_rtf, 2) if _warmup_rtf else None,
            "last_rtf": _last_rtf,
            "last_gen_audio_s": _last_gen_audio_s,
            "pinned": _pinned,
            "ctx_cache": {"size": len(_ctx_token_cache),
                          "hits": _ctx_cache_hits, "misses": _ctx_cache_misses}}


@app.post("/warmup")
async def warmup():
    """Pre-load the model so the first utterance after an idle unload
    doesn't pay the cold reload. Ping this at voice-session start. Returns
    immediately; loading proceeds in the background."""
    _touch()
    threading.Thread(target=_safe_warm, daemon=True).start()
    return JSONResponse({"status": "warming"}, status_code=202)


@app.post("/pin")
async def pin():
    """Hold the model resident — the idle watcher won't unload while pinned.
    Use while actively testing so you don't re-pay the slow reload+compile.
    Warms the model now (background) if it isn't loaded. Call /unpin when done."""
    global _pinned
    _pinned = True
    _touch()
    if _generator is None:
        threading.Thread(target=_safe_warm, daemon=True).start()
        return JSONResponse({"status": "pinned", "loading": True})
    return JSONResponse({"status": "pinned", "loading": False})


@app.post("/unpin")
async def unpin():
    """Release the pin — the model can idle-unload again after IDLE_UNLOAD_S.
    Does NOT unload immediately; the idle timer resumes from now."""
    global _pinned
    _pinned = False
    _touch()
    return JSONResponse({"status": "unpinned"})


@app.post("/unload")
async def unload(session: str = ""):
    """Release GPU VRAM now (e.g. the voice WS closed). Drives the
    conversation-scoped residency mode: warm on session open, unload on
    close, instead of a blind idle timer. Optionally clears that session's
    cross-speaker context so a new conversation starts clean. Clears the pin."""
    global _pinned
    _pinned = False
    if session:
        with _sessions_lock:
            _self_turns.pop(session, None)
            _user_turns.pop(session, None)
    _unload_generator()
    return JSONResponse({"status": "unloaded"})


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "sesame-csm-1b", "object": "model", "owned_by": "sesame"}]}


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def list_voices():
    return JSONResponse(content=_list_voices())


@app.post("/v1/voices")
async def upload_voice(
    voice_file: UploadFile = File(...),
    voice_name: str = Form(""),
    transcript: str = Form(""),
):
    """Register a cloning reference: a short clip + (ideally) its transcript.
    CSM clones from a (text, audio) Segment, so the transcript materially
    helps — if omitted, cloning still works but weaker."""
    name = voice_name or os.path.splitext(voice_file.filename or "voice")[0]
    wav_path, txt_path = _voice_paths(name)
    raw = await voice_file.read()
    # Decode whatever was uploaded → 24kHz mono wav via ffmpeg.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", "pipe:1"],
        input=raw, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(400, "could not decode uploaded audio")
    with open(wav_path, "wb") as f:
        f.write(proc.stdout)
    if transcript.strip():
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.strip())
    # Drop any cached anchor Segment for this name so the new clip is picked
    # up next turn (the content-keyed Mimi cache self-invalidates — new audio,
    # new key).
    _anchor_cache.pop(_base_name(name), None)
    return {"id": name, "name": name, "has_transcript": bool(transcript.strip())}


@app.post("/v1/context/user_turn")
async def user_turn(
    audio: UploadFile = File(...),
    transcript: str = Form(""),
    x_augmentum_session: str = Header(default=""),
):
    """Cross-speaker context: record the USER's spoken turn for a session so
    her next reply's prosody reacts to how they sounded (pace, energy, mood)
    — the dialogue conditioning that's the whole point of CSM over a plain
    one-shot TTS. The clip + transcript become a speaker-1 Segment in this
    session's context buffer. RAM-only; never written to disk.

    Fire-and-forget from the caller's view: returns immediately after
    banking the turn; the LLM's generation latency covers the gap before
    her first sentence synthesizes."""
    if not x_augmentum_session:
        # Without a session key there's nothing to attach the turn to —
        # accept silently so the caller's voice loop never blocks on this.
        return JSONResponse({"status": "skipped", "reason": "no session"})
    raw = await audio.read()
    if not raw:
        return JSONResponse({"status": "skipped", "reason": "empty audio"})
    try:
        tensor = _decode_to_tensor(raw)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — never break the turn over context
        print(f"[csm] user_turn decode failed: {exc}", flush=True)
        return JSONResponse({"status": "skipped", "reason": "decode failed"})
    _remember_user_turn(x_augmentum_session, transcript.strip(), tensor)
    return JSONResponse({"status": "ok"})


@app.post("/v1/audio/speech")
async def speech(
    body: TTSRequest,
    x_augmentum_session: str = Header(default=""),
):
    text = (body.input or "").strip()
    if not text:
        raise HTTPException(400, "input is required")
    _touch()
    try:
        _get_generator()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"CSM model unavailable: {exc}") from exc

    fmt = (body.response_format or "wav").lower()
    context = _build_context(x_augmentum_session, body.voice)

    if fmt in ("wav", "pcm"):
        # Low-latency path: stream chunks as they decode.
        def _gen():
            if fmt == "wav":
                yield _wav_header_streaming()
            yield from _stream_pcm_chunks(text, context, x_augmentum_session)
        media = "audio/wav" if fmt == "wav" else "audio/pcm"
        return StreamingResponse(_gen(), media_type=media)

    # Compressed formats: buffer then encode once (no streaming).
    pcm = b"".join(_stream_pcm_chunks(text, context, x_augmentum_session))
    if not pcm:
        raise HTTPException(500, "no audio generated")
    data, mime = _encode_compressed(pcm, fmt)
    return Response(content=data, media_type=mime)


# Idle-unload watcher (daemon — dies with the process). Started on import
# so it runs under uvicorn, not just the __main__ path.
if IDLE_UNLOAD_S > 0:
    threading.Thread(target=_idle_watcher, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
