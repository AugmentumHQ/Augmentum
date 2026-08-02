"""``wake_word_training`` job handler — train a per-avatar wake-word ONNX.

Orchestrates: load Kokoro → synth positives + negatives → train CRNN →
export ONNX → record row in ``wake_word_models``. Idempotent on
retry: re-running for the same avatar_id replaces the model in place
and bumps ``version``. Training time on a 24 GB consumer GPU is
~5 minutes for the default sample budget.

Payload schema::

    {
        "avatar_id": "bundled_f_becca",   # required, table primary key
        "phrase":    "becca",              # required, what to listen for
        "voices":    ["af_heart", ...],    # optional, default voice mix
        "epochs":    20                     # optional
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop

log = get_logger(__name__)


def _wake_word_dir(data_dir: str | Path, avatar_id: str) -> Path:
    """Resolve the directory holding {avatar_id}/model.onnx."""
    base = Path(data_dir) / "wake_word_models" / avatar_id
    base.mkdir(parents=True, exist_ok=True)
    return base


async def _next_version(conn, avatar_id: str) -> int:
    """Read the prior version for this avatar; bump by one on each train."""
    cur = await conn.execute(
        "SELECT version FROM wake_word_models WHERE avatar_id = ?",
        (avatar_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) + 1 if row else 1


async def _upsert_model_row(
    conn,
    *,
    avatar_id: str,
    phrase: str,
    model_path: str,
    version: int,
    train_metrics: str,
) -> None:
    """Insert or replace the wake_word_models row for this avatar."""
    await conn.execute(
        """
        INSERT INTO wake_word_models
            (avatar_id, phrase, model_path, version, train_metrics)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(avatar_id) DO UPDATE SET
            phrase = excluded.phrase,
            model_path = excluded.model_path,
            version = excluded.version,
            trained_at = datetime('now'),
            train_metrics = excluded.train_metrics
        """,
        (avatar_id, phrase, model_path, version, train_metrics),
    )
    await conn.commit()


def make_wake_word_training_handler(app):
    """Build the ``wake_word_training`` handler bound to runtime services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        from augmentum.config import settings
        from augmentum.voice.kokoro_tts import KokoroTTS
        from augmentum.voice.wake_word.training import train_wake_word_model

        avatar_id = str(ctx.payload.get("avatar_id") or "").strip()
        phrase = str(ctx.payload.get("phrase") or "").strip()
        if not avatar_id or not phrase:
            raise RuntimeError(
                "wake_word_training: payload must include avatar_id + phrase"
            )

        voices = ctx.payload.get("voices") or None
        epochs = int(ctx.payload.get("epochs") or 20)

        # Personal recordings come from the user's own captures via
        # POST /api/wake_word/personal_samples. They're stored per-user
        # so the path includes ctx.user_id; routed into training when
        # the payload sets ``use_personal_samples=true``.
        use_personal = bool(ctx.payload.get("use_personal_samples", False))
        personal_dir = None
        if use_personal:
            personal_dir = (
                Path(getattr(settings, "data_dir", "/data"))
                / "wake_word_personal"
                / ctx.user_id
                / avatar_id
            )

        sm = getattr(app.state, "state_manager", None)
        backend = getattr(sm, "backend", None) if sm else None
        if backend is None:
            raise RuntimeError(
                "wake_word_training: state backend not initialized"
            )
        conn = backend.conn

        kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
        if not kokoro.is_available:
            # Model load is blocking — keep it off the event loop.
            await load_model_off_loop(kokoro.load_model)
        if not kokoro.is_available:
            raise RuntimeError(
                "wake_word_training: Kokoro TTS unavailable, cannot synthesize "
                "training data"
            )

        out_dir = _wake_word_dir(getattr(settings, "data_dir", "/data"), avatar_id)
        out_path = out_dir / "model.onnx"

        log.info("wake_word_training_starting",
                 avatar_id=avatar_id, phrase=phrase, out_path=str(out_path))

        result = await train_wake_word_model(
            phrase=phrase,
            output_path=out_path,
            kokoro=kokoro,
            epochs=epochs,
            voices=voices,
            personal_samples_dir=personal_dir,
            ctx=ctx,
        )

        if not out_path.exists():
            raise RuntimeError(
                f"wake_word_training: ONNX export failed — {out_path} missing"
            )

        version = await _next_version(conn, avatar_id)
        await _upsert_model_row(
            conn,
            avatar_id=avatar_id,
            phrase=phrase,
            model_path=str(out_path),
            version=version,
            train_metrics=result.to_json(),
        )

        log.info("wake_word_training_complete",
                 avatar_id=avatar_id,
                 best_val_acc=round(result.best_val_acc, 4),
                 epochs_run=result.epochs_run,
                 model_path=str(out_path))

        # Proactively reload the model in the running service so any
        # live /ws/voice/wake session picks up the fresh ONNX on its
        # next inference window — instead of the user having to toggle
        # wake off + on. Safe to call even when the detector isn't
        # currently loaded; load_models_from_db is a no-op for missing
        # subscriptions. mtime-aware enable_model in service.py is what
        # makes this actually swap the in-memory session.
        try:
            from augmentum.voice.wake_word.service import load_models_from_db
            await load_models_from_db(app.state, avatar_ids=[avatar_id])
        except Exception as exc:
            log.warning(
                "wake_word_training_reload_failed",
                avatar_id=avatar_id, error=str(exc),
                note="live wake session will pick up the new model on next reconnect",
            )

        return {
            "avatar_id": avatar_id,
            "phrase": phrase,
            "model_path": str(out_path),
            "version": version,
            "metrics": json.loads(result.to_json()),
        }

    return handler
