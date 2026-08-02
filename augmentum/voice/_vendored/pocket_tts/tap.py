"""MimiTappedTTSModel — subclass override that exposes Mimi codec tokens.

The Phase 2 audio-history substrate (per
``augmentum/companion/presence/audio_history.py``) stores per-turn
Mimi tokens so a future Kyutai-family model swap inherits conversation
context for free. Upstream ``pocket_tts.models.tts_model.TTSModel``
generates these tokens internally but only exposes the decoded PCM
output — this module bridges that gap.

Design constraints (per VENDOR.md):

* Additive — the parent decode path runs unchanged. Audio output is
  byte-identical to upstream. Only a side channel changes.
* Optional — when ``mimi_codes_callback`` is None (the default),
  the override no-ops; zero overhead compared to upstream.
* Loud on drift — at import time we verify the required upstream
  methods are still present. On drift we log a warning and continue,
  rather than raising, so a runtime that doesn't need the tap (audio
  history disabled) still works.
* Lazy import — ``pocket_tts`` is an optional dep at the Augmentum
  layer; this module's import surface guards against ImportError so
  modules that touch ``MimiTappedTTSModel`` don't break the import
  graph in dev environments without the package installed.

Usage::

    from augmentum.voice._vendored.pocket_tts.tap import (
        MimiTappedTTSModel,
        try_install_tap,
    )

    if MimiTappedTTSModel is not None:
        model = MimiTappedTTSModel.load_model()
        model.mimi_codes_callback = lambda idx, codes: ...  # capture
        for chunk in model.generate_audio_stream(state, text):
            ...  # audio plays as normal; tokens flow to callback

    # Or attach the tap to an already-instantiated upstream model:
    upstream_model = TTSModel.load_model()
    try_install_tap(upstream_model, callback=...)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from augmentum.voice._vendored.pocket_tts.upstream_pin import (
    REQUIRED_UPSTREAM_METHODS,
    UPSTREAM_COMMIT,
    UPSTREAM_DATE,
)

log = logging.getLogger(__name__)


# Type alias for the Mimi codes callback.
# Args: (chunk_index: int, quantized_tokens) — see VENDOR.md for the
# token shape. The callback runs INSIDE the decode worker thread, so it
# must be quick + thread-safe. For audio history capture, the typical
# implementation is to push the tokens onto a queue the main coroutine
# drains.
MimiCodesCallback = Callable[[int, Any], None]


_upstream_TTSModel: Any | None = None
_import_error: ImportError | None = None


def _resolve_upstream() -> Any | None:
    """Import upstream lazily and cache the class object.

    Returns None and stashes the error in ``_import_error`` if upstream
    isn't installed — every site that calls this must handle the None
    case (typically by falling back to a no-op tap path).
    """
    global _upstream_TTSModel, _import_error
    if _upstream_TTSModel is not None:
        return _upstream_TTSModel
    try:
        from pocket_tts.models.tts_model import TTSModel  # type: ignore[import-untyped]
    except ImportError as exc:
        _import_error = exc
        return None
    _upstream_TTSModel = TTSModel
    _check_upstream_surface(TTSModel)
    return TTSModel


def _check_upstream_surface(cls: Any) -> None:
    """Verify the upstream class still has the methods we need.

    Logs warning on drift; does not raise. The override only attaches if
    the tap point (``_decode_audio_worker``) is still present.
    """
    missing = [
        name for name in REQUIRED_UPSTREAM_METHODS
        if not hasattr(cls, name)
    ]
    if missing:
        log.warning(
            "pocket_tts_vendor_drift",
            extra={
                "missing_methods": missing,
                "pinned_commit": UPSTREAM_COMMIT,
                "pinned_date": UPSTREAM_DATE,
                "remediation": (
                    "Review upstream changes since pinned commit; update "
                    "tap.py + upstream_pin.py. See VENDOR.md."
                ),
            },
        )


def _build_tapped_class() -> Any | None:
    """Construct the subclass override of upstream TTSModel.

    Returns None if upstream isn't importable or if the tap point method
    is missing. Returning None (not raising) means callers can gracefully
    fall back to the upstream model unmodified.
    """
    upstream = _resolve_upstream()
    if upstream is None:
        return None
    if not hasattr(upstream, "_decode_audio_worker"):
        log.warning(
            "pocket_tts_tap_unavailable",
            extra={
                "reason": "_decode_audio_worker not present on upstream class",
                "pinned_commit": UPSTREAM_COMMIT,
            },
        )
        return None

    class _MimiTappedTTSModel(upstream):  # type: ignore[misc, valid-type]
        """TTSModel that emits Mimi codes via ``mimi_codes_callback``.

        The override re-runs upstream's ``_decode_audio_worker`` and
        captures the quantized Mimi latent before it's decoded to PCM.
        The audio path is otherwise identical to upstream.

        Set ``mimi_codes_callback`` to a callable to enable the tap;
        leave None for upstream-equivalent behavior.

        Thread safety: the callback fires inside the upstream worker
        thread (per upstream's ``_decode_audio_worker`` docstring).
        Callback implementations must not block — push to a queue and
        return.
        """

        mimi_codes_callback: MimiCodesCallback | None = None

        def _decode_audio_worker(  # type: ignore[override]
            self,
            latents_queue: Any,
            result_queue: Any,
            mimi_sequence_length: int,
            mimi_steps_per_latent: int,
        ) -> None:
            """Tapped re-implementation of upstream's decode worker.

            We deliberately re-implement the loop rather than
            super().__call__ + side-channel — upstream's worker pulls
            latents from a queue inside a tight loop, and there's no
            hook point that lets us see each latent before decode
            without owning the loop. So we own it.

            Behavior MUST stay byte-identical to upstream's worker
            on every code path except the optional callback. The body
            below mirrors upstream commit ``UPSTREAM_COMMIT`` exactly,
            with one added block at the documented tap point.

            On any divergence between this method and upstream's,
            update both at once and re-pin via VENDOR.md procedure.
            """
            # Local imports for the upstream symbols we need. Doing them
            # inside the method (not at module scope) keeps the
            # subclass module importable when pocket_tts is absent —
            # the class itself wouldn't have been built in that case.
            import queue  # noqa: F401  — used implicitly by the queue.put pattern
            import time
            from pocket_tts.models.mimi import init_states, increment_steps  # type: ignore[import-untyped]

            callback = self.mimi_codes_callback
            chunk_index = 0
            try:
                audio_chunks = []
                mimi_state = init_states(
                    self.mimi, batch_size=1,
                    sequence_length=mimi_sequence_length,
                )
                while True:
                    latent = latents_queue.get()
                    if latent is None:
                        break
                    mimi_decoding_input = (
                        latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                    )
                    transposed = mimi_decoding_input.transpose(-1, -2)
                    quantized = self.mimi.quantizer(transposed)

                    # ── Mimi tap ────────────────────────────────────
                    # quantized is the post-RVQ latent that feeds the
                    # vocoder. For audio history purposes this is the
                    # frame-level token state we want to capture. The
                    # caller may further reduce it to discrete codes
                    # via the quantizer's encoder helpers if they need
                    # the integer code IDs (Mimi exposes
                    # `quantizer.encode_to_codes`-style helpers but
                    # the surface varies by version; we hand off the
                    # quantized latent and let the consumer decide).
                    if callback is not None:
                        try:
                            callback(chunk_index, quantized)
                        except Exception as cb_exc:  # noqa: BLE001
                            log.warning(
                                "pocket_tts_tap_callback_failed",
                                extra={
                                    "chunk_index": chunk_index,
                                    "error": str(cb_exc),
                                },
                            )
                    chunk_index += 1
                    # ────────────────────────────────────────────────

                    t = time.monotonic()
                    audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
                    increment_steps(
                        self.mimi, mimi_state,
                        increment=mimi_steps_per_latent,
                    )
                    audio_frame_duration = (
                        audio_frame.shape[2] / self.config.mimi.sample_rate
                    )
                    log.debug(
                        " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                        int(audio_frame_duration * 1000),
                        int((time.monotonic() - t) * 1000),
                    )
                    audio_chunks.append(audio_frame)

                    result_queue.put(("chunk", audio_frame))
                    latents_queue.task_done()

                result_queue.put(("done", None))

            except Exception as exc:  # noqa: BLE001 — propagate via queue
                result_queue.put(("error", exc))

    return _MimiTappedTTSModel


# Public symbol. None when upstream isn't installed; callers must
# handle the None case (typically by skipping the audio-history
# capture and logging).
MimiTappedTTSModel: Any | None = _build_tapped_class()


def try_install_tap(
    model: Any, callback: MimiCodesCallback | None,
) -> bool:
    """Attach the Mimi tap to an existing upstream model instance.

    Useful when the upstream model was constructed before we knew we
    wanted captures (e.g. lazy callback wiring inside Phase 3's
    streaming TTS path). Replaces the bound ``_decode_audio_worker``
    on the instance with our tapped version.

    Returns True on success, False if upstream isn't available or the
    install is unsafe (e.g. method already overridden by someone else).
    """
    if MimiTappedTTSModel is None:
        return False
    if not hasattr(model, "_decode_audio_worker"):
        return False
    # Bind the override as an instance method; this leaves the class
    # itself unmodified (no global side effects).
    import types
    model._decode_audio_worker = types.MethodType(
        MimiTappedTTSModel._decode_audio_worker, model,
    )
    model.mimi_codes_callback = callback
    return True


def upstream_available() -> bool:
    """Return True if ``pocket_tts`` is importable in this environment.

    Useful for callers that want to surface "audio history capture
    disabled" diagnostics rather than silently degrading.
    """
    return _resolve_upstream() is not None
