"""Atomic JSON file writes — kill-9-survivable persistence.

A plain ``open(path, "w") + json.dump`` truncates the file at the
moment of open. Any process death between truncate and successful
write produces a zero-byte or partially-written file on disk. For
volatile sidecar state (image metadata, model session pins, journal
entries, transcription progress) the failure mode is silent data
loss — the file exists at the expected path, but it's corrupt.

The atomic-write pattern is the standard defense:

  1. Write to a sibling temporary file (with PID suffix to avoid
     collision between concurrent callers).
  2. ``flush`` + ``fsync`` to push the bytes into the kernel's page
     cache and out to durable storage.
  3. ``os.replace`` from temp to the real path — atomic on POSIX
     when src and dst live on the same filesystem.

On Windows ``os.replace`` is also atomic since Python 3.3, with the
same same-filesystem caveat.

Use this everywhere Augmentum writes JSON outside the SQLite paths.
SQLite writes go through aiosqlite + WAL and have their own atomicity
guarantees; sidecar JSON does not.

This module is intentionally synchronous (no asyncio): atomic writes
are short and the ``os.replace`` is the same syscall in either
flavor. Callers that need non-blocking semantics should defer via
``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: str | Path,
    data: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = False,
) -> None:
    """Persist ``data`` as JSON at ``path`` via tmp + fsync + replace.

    Args:
        path: target path (str or Path).
        data: any JSON-serialisable Python value (the same shape
            ``json.dump`` accepts).
        indent: pretty-print indent. ``None`` (default) writes compact
            output. Match the existing call site's format when
            migrating from a plain ``json.dump``.
        ensure_ascii: same as the stdlib argument. Default False so
            non-ASCII characters (names, emoji, foreign text) write
            verbatim instead of \\uXXXX escapes — matches Augmentum's
            in-repo style and keeps DB compares straightforward.

    Raises:
        OSError if the parent directory can't be created or the
        write fails. The temp file is cleaned up best-effort on
        failure; if cleanup itself fails, we let the exception
        propagate (the operator should see both signals).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # PID in the suffix means concurrent writers from different
    # processes (e.g. unit tests + a running container) don't collide
    # on the rename target. Within a single process, this is still safe
    # because os.replace is atomic.
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            # Push the buffer to the kernel...
            f.flush()
            # ...and force the kernel to push the bytes to durable
            # storage. Without this, a kernel panic between flush and
            # replace can produce a stale renamed file. fsync is
            # expensive — single-digit ms — but state writes are not
            # on the hot path.
            os.fsync(f.fileno())
        # Atomic on POSIX + Windows (Python 3.3+) when source + dest
        # are on the same filesystem.
        os.replace(tmp, target)
    except Exception:
        # Best-effort cleanup so we don't leave a tmp turd if the
        # write failed mid-flight.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        except OSError:
            # Cleanup itself failed — let it pass; the primary
            # exception is the load-bearing signal.
            pass
        raise


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically write a plain-text payload at ``path``.

    Same guarantees as :func:`atomic_write_json` but for text bodies
    that don't go through JSON encoding (config snippets, generated
    asset manifests, etc.). Prefer the JSON variant for structured
    data so the schema stays auditable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise
