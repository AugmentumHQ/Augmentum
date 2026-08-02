#!/usr/bin/env python3
"""train_builtin_wake_words.py — bake all 15 default wake-word ONNX models.

Run overnight. Resumable across restarts: re-running the script skips
phrases that already have an acceptable-quality model on disk and
picks up from the next un-baked one. Logs progress to stdout AND to
``/data/wake_word_training.log`` so morning-checkup is easy.

Each model is ~314 KB. Total training time at the bumped budget
(504 positives / 1050 negatives / 20 epochs) is ~15 min per phrase
on a 24 GB consumer GPU; 15 phrases ≈ 4 hours wall time end-to-end.

Output:
    /data/wake_word_models/{slug}/model.onnx       per-phrase ONNX
    /data/wake_word_training.log                    progress log
    wake_word_models DB rows with is_builtin=1     registry update

Invocation:
    # Attached (watch live):
    docker exec -it augmentum-augmentum-1 \\
        python3 /app/scripts/train_builtin_wake_words.py

    # Overnight detached:
    docker exec augmentum-augmentum-1 sh -c \\
        'cd /app && setsid nohup python3 -u scripts/train_builtin_wake_words.py \\
            > /data/wake_word_training.log 2>&1 < /dev/null &'

    # Then watch later:
    docker exec augmentum-augmentum-1 tail -f /data/wake_word_training.log

If the container restarts mid-training, the script also dies — just
re-run it. Already-baked phrases are skipped; training resumes from
the in-flight one.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

# Container layout: this script lives at /app/scripts/, augmentum module
# lives at /app/augmentum/. Add /app to sys.path so imports resolve when
# the script is invoked as ``python3 /app/scripts/...`` (not as a module).
sys.path.insert(0, "/app")

from augmentum.config import settings  # noqa: E402
from augmentum.voice.kokoro_tts import KokoroTTS  # noqa: E402
from augmentum.voice.wake_word.training import train_wake_word_model  # noqa: E402


# ── Configuration ───────────────────────────────────────────────────

# Each entry: (slug, phrase-to-say, friendly-label).
# The slug is the wake_word_models primary key. For wake words that
# correspond to a bundled VRM avatar (becca, danny, etc.) we use the
# avatar's bundled_*_id so the runtime can resolve avatar ↔ wake-word
# without a second lookup table. Generic pop-culture wake words use
# ``wake-<slug>`` prefixed ids.
WAKE_WORDS = [
    # Pop culture
    ("wake-hey-jarvis",    "hey jarvis",     "Hey Jarvis"),
    ("wake-hey-samantha",  "hey samantha",   "Hey Samantha"),
    ("wake-hey-cortana",   "hey cortana",    "Hey Cortana"),
    ("wake-hey-baymax",    "hey baymax",     "Hey Baymax"),
    ("wake-hey-wheatley",  "hey wheatley",   "Hey Wheatley"),
    ("wake-hey-codsworth", "hey codsworth",  "Hey Codsworth"),
    ("wake-hey-glados",    "hey glados",     "Hey Glados"),
    ("wake-hey-friday",    "hey friday",     "Hey Friday"),
    ("wake-computer",      "computer",       "Computer"),
    # Bundled VRM cast
    ("bundled_f_becca",    "hey becca",      "Hey Becca"),
    ("bundled_m_danny",    "hey danny",      "Hey Danny"),
    ("bundled_f_lise",     "hey lise",       "Hey Lise"),
    ("bundled_f_roxanne",  "hey roxanne",    "Hey Roxanne"),
    ("bundled_m_vance",    "hey vance",      "Hey Vance"),
    ("bundled_m_louis",    "hey louis",      "Hey Louis"),
]

EPOCHS = 20
# Skip phrases that already have a model AND its val_acc is at least this.
# Below this we retrain (the first low-budget canary produced 60% which is
# basically chance — those should retrain at the bumped budget).
MIN_KEEP_VAL_ACC = 0.75


# ── Helpers ─────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    """Stamp + flush so the file follow is real-time."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _open_db() -> sqlite3.Connection:
    db_path = Path(getattr(settings, "data_dir", "/data")) / "augmentum.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _existing_metrics(conn: sqlite3.Connection, slug: str) -> dict | None:
    cur = conn.execute(
        "SELECT model_path, train_metrics, version FROM wake_word_models "
        "WHERE avatar_id = ?",
        (slug,),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        metrics = json.loads(row["train_metrics"]) if row["train_metrics"] else {}
    except Exception:
        metrics = {}
    return {
        "model_path": row["model_path"],
        "version": row["version"],
        "metrics": metrics,
    }


def _should_skip(conn: sqlite3.Connection, slug: str) -> tuple[bool, str]:
    """Return (skip?, reason). Skip when model file exists AND val_acc is OK."""
    rec = _existing_metrics(conn, slug)
    if not rec:
        return False, "no prior model"
    model_path = Path(rec["model_path"])
    if not model_path.exists():
        return False, f"DB row points to missing file {model_path}"
    val_acc = rec["metrics"].get("best_val_acc")
    if val_acc is None:
        return False, "no val_acc recorded — retrain"
    if val_acc < MIN_KEEP_VAL_ACC:
        return False, f"val_acc={val_acc:.3f} < {MIN_KEEP_VAL_ACC} — retrain"
    return True, f"already trained: val_acc={val_acc:.3f} v{rec['version']}"


def _next_version(conn: sqlite3.Connection, slug: str) -> int:
    cur = conn.execute(
        "SELECT version FROM wake_word_models WHERE avatar_id = ?", (slug,),
    )
    row = cur.fetchone()
    return int(row["version"]) + 1 if row else 1


def _upsert_row(
    conn: sqlite3.Connection,
    *,
    slug: str,
    phrase: str,
    model_path: str,
    version: int,
    train_metrics: str,
) -> None:
    conn.execute(
        """
        INSERT INTO wake_word_models
            (avatar_id, phrase, model_path, version, train_metrics, is_builtin)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(avatar_id) DO UPDATE SET
            phrase = excluded.phrase,
            model_path = excluded.model_path,
            version = excluded.version,
            trained_at = datetime('now'),
            train_metrics = excluded.train_metrics,
            is_builtin = 1
        """,
        (slug, phrase, model_path, version, train_metrics),
    )
    conn.commit()


# ── Main loop ───────────────────────────────────────────────────────


async def main() -> int:
    data_dir = Path(getattr(settings, "data_dir", "/data"))
    out_root = data_dir / "wake_word_models"
    out_root.mkdir(parents=True, exist_ok=True)

    _log(f"baking {len(WAKE_WORDS)} wake words → {out_root}")
    _log(f"min keep val_acc: {MIN_KEEP_VAL_ACC} (below this triggers retrain)")
    _log(f"epochs per phrase: {EPOCHS}")

    # Load Kokoro once for the whole session — model load is ~3-5s.
    _log("loading Kokoro TTS …")
    kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
    await asyncio.to_thread(kokoro.load_model)
    if not kokoro.is_available:
        _log("FATAL: Kokoro failed to load. Aborting.")
        return 1
    _log(f"Kokoro ready ({len(kokoro.voices) if hasattr(kokoro, 'voices') else '?'} voices)")

    conn = _open_db()
    done = []
    skipped = []
    failed = []
    overall_start = time.time()

    for idx, (slug, phrase, label) in enumerate(WAKE_WORDS, 1):
        _log("─" * 60)
        _log(f"[{idx}/{len(WAKE_WORDS)}] {label} (slug={slug}, phrase='{phrase}')")
        skip, reason = _should_skip(conn, slug)
        if skip:
            _log(f"  SKIP — {reason}")
            skipped.append(slug)
            continue
        _log(f"  TRAIN — {reason}")

        out_dir = out_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "model.onnx"

        t0 = time.time()
        try:
            result = await train_wake_word_model(
                phrase=phrase,
                output_path=out_path,
                kokoro=kokoro,
                epochs=EPOCHS,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            _log(f"  FAILED after {elapsed:.0f}s — {type(exc).__name__}: {exc}")
            failed.append((slug, str(exc)))
            continue

        if not out_path.exists():
            _log("  FAILED — ONNX file not written")
            failed.append((slug, "no onnx written"))
            continue

        version = _next_version(conn, slug)
        _upsert_row(
            conn,
            slug=slug,
            phrase=phrase,
            model_path=str(out_path),
            version=version,
            train_metrics=result.to_json(),
        )

        elapsed = time.time() - t0
        size_kb = out_path.stat().st_size / 1024
        _log(
            f"  DONE in {elapsed:.0f}s — val_acc={result.best_val_acc:.3f} "
            f"val_loss={result.best_val_loss:.3f} pos={result.positives_count} "
            f"neg={result.negatives_count} epochs={result.epochs_run} "
            f"size={size_kb:.0f}KB v{version}"
        )
        done.append(slug)

    total = time.time() - overall_start
    _log("─" * 60)
    _log(f"COMPLETE in {total/60:.1f} min")
    _log(f"  trained:  {len(done)}  → {done}")
    _log(f"  skipped:  {len(skipped)}  → {skipped}")
    _log(f"  failed:   {len(failed)}")
    for slug, err in failed:
        _log(f"    {slug}: {err}")

    conn.close()
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
