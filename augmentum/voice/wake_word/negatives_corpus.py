"""Real-audio negatives corpus for wake-word training.

Synthetic-only training (Kokoro for both positives and negatives) leaves
the model blind to real-world acoustics:

  * idle rooms score 0.3-0.5 because the model never saw silence,
  * the avatar's own TTS triggers it because TTS playback is in-
    distribution for TTS-trained negatives,
  * the F1-optimal threshold picked against synthetic val is mis-
    calibrated for real audio (often 0.25).

This module supplies the missing piece — real-world speech that isn't
the wake phrase, plus runtime-synthesized silence/noise windows. The
``training.py`` pipeline mixes these into the negatives pool when the
corpus is installed; it falls back to the legacy synthetic-only path
when the corpus is absent, so the contract degrades gracefully.

Source: LibriSpeech dev-clean — ~337 MB, ~5.4 hours of public-domain
read speech across 40 speakers, native 16 kHz mono FLAC. Distributed by
OpenSLR under CC-BY-4.0. No noise corpus is downloaded — the
``sample_silence_windows`` helper synthesizes the quiet-room class at
training time (gaussian + low-pass-filtered noise), mirroring
microWakeWord's RIR/SNR augmentation approach instead of pulling a
multi-GB noise tarball.

Layout under ``{data_dir}/wake_word_corpora/``::

    librispeech-dev-clean/
        LibriSpeech/dev-clean/{spk}/{chapter}/{utt}.flac
    manifest.json                    # corpus metadata + flac catalog

Public API:

  * :func:`ensure_downloaded` — idempotent download + extract + verify
  * :func:`is_installed` — quick boolean for the training path
  * :func:`sample_real_speech_windows` — random 1-second LibriSpeech crops
  * :func:`sample_silence_windows` — synthetic silence/noise windows

The training pipeline never calls ``ensure_downloaded`` itself — that's
the corpus-download job's responsibility. Training observes
``is_installed`` and routes accordingly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tarfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import numpy as np
import torchaudio

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000  # must match training.py

# LibriSpeech dev-clean. Public-domain read speech, CC-BY-4.0.
# MD5 is published by the Kaldi LibriSpeech recipe
# (egs/librispeech/s5/local/download_and_untar.sh upstream); if upstream
# rotates the tarball the recipe is the first place to find the new hash.
_LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
_LIBRISPEECH_MD5 = "42e2234ba48799c1f50f24a7926300a1"

# Approximate tarball size, used for progress denominators when the HTTP
# response doesn't include Content-Length. Off by a few percent is fine —
# the value only drives the progress bar, not correctness.
_LIBRISPEECH_BYTES_APPROX = 337_926_624

# Streaming download chunk. 1 MB is small enough that the progress bar
# moves visibly on a fast link, large enough that per-chunk overhead
# (httpx aiter_bytes, hash update, file write) doesn't dominate.
_DL_CHUNK_BYTES = 1024 * 1024

# Maximum number of FLAC files to remember in the manifest catalog. The
# dev-clean split has ~2,700 utterances which is well under any sane cap;
# the limit just bounds memory if a future corpus is added that's larger.
_CATALOG_MAX_FILES = 50_000


# ── Paths ────────────────────────────────────────────────────────────


def _corpora_root() -> Path:
    """Return (creating if needed) the on-disk root for all wake-word corpora."""
    from augmentum.config import settings  # lazy — settings imports are heavy
    p = Path(settings.data_dir) / "wake_word_corpora"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _librispeech_dir() -> Path:
    return _corpora_root() / "librispeech-dev-clean"


def _flac_root() -> Path:
    """Where LibriSpeech's tarball extracts to."""
    return _librispeech_dir() / "LibriSpeech" / "dev-clean"


def _manifest_path() -> Path:
    return _corpora_root() / "manifest.json"


# ── Manifest ─────────────────────────────────────────────────────────


def _read_manifest() -> dict[str, Any]:
    p = _manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        log.warning("negatives_corpus_manifest_unreadable", path=str(p))
        return {}


def _write_manifest(data: dict[str, Any]) -> None:
    tmp = _manifest_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, _manifest_path())


# ── Status ───────────────────────────────────────────────────────────


def is_installed() -> bool:
    """True if LibriSpeech dev-clean is on disk and the manifest looks complete.

    Cheap (one file stat + JSON parse + one rglob hit). Safe to call from
    the training hot path to decide which negatives pool to use.
    """
    m = _read_manifest()
    info = m.get("librispeech_dev_clean") or {}
    if not info.get("complete"):
        return False
    if not _flac_root().exists():
        return False
    # Sanity-check: at least one .flac actually exists. A truncated extract
    # could leave the manifest claiming completion with an empty tree.
    for _ in _flac_root().rglob("*.flac"):
        return True
    return False


def installed_summary() -> dict[str, Any]:
    """Return a small dict describing the installed corpus, or empty if absent.

    Used by the UI to show "Installed: 337 MB, 2,703 utterances" after the
    download job completes. Cheap (manifest read only).
    """
    if not is_installed():
        return {}
    info = (_read_manifest().get("librispeech_dev_clean") or {}).copy()
    info["path"] = str(_librispeech_dir())
    return info


# ── Download + extract ───────────────────────────────────────────────


async def ensure_downloaded(
    progress_cb: Callable[[float, str], Awaitable[None]] | None = None,
    *,
    cancel_cb: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Download + extract LibriSpeech dev-clean if not already present.

    Idempotent. Safe to call from a JobContext-driven handler — pass
    ``ctx.update_progress`` as ``progress_cb`` and ``ctx.check_cancel`` as
    ``cancel_cb`` to get progress reporting + cooperative cancellation.

    Returns a small dict mirroring :func:`installed_summary` on success.
    Raises :class:`RuntimeError` on download/verify failures so the job
    runner can mark the row failed.
    """
    if is_installed():
        log.info("negatives_corpus_already_installed")
        return installed_summary()

    target_dir = _librispeech_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    tarball = target_dir / "dev-clean.tar.gz"

    if progress_cb:
        await progress_cb(0.0, "downloading LibriSpeech dev-clean")

    # Resume support — if a prior interrupted download left bytes on
    # disk, skip the HTTP fetch and re-verify what's there. We use the
    # MD5 as the source of truth: if it matches, we extract; if it
    # doesn't, we wipe and re-download. No partial-resume of the
    # tarball itself — at 337 MB on a residential link it's not worth
    # the complexity of Range-resume sidecars.
    md5_ok = False
    if tarball.exists() and tarball.stat().st_size > 0:
        log.info("negatives_corpus_resume_check", existing_bytes=tarball.stat().st_size)
        if progress_cb:
            await progress_cb(0.05, "verifying existing tarball")
        md5_ok = await asyncio.to_thread(_verify_md5, tarball, _LIBRISPEECH_MD5)
        if not md5_ok:
            log.info("negatives_corpus_resume_md5_mismatch_redownloading")
            tarball.unlink(missing_ok=True)

    if not md5_ok:
        await _download_streaming(
            _LIBRISPEECH_URL, tarball,
            expected_bytes=_LIBRISPEECH_BYTES_APPROX,
            progress_cb=progress_cb,
            progress_lo=0.0, progress_hi=0.85,
            cancel_cb=cancel_cb,
        )
        if progress_cb:
            await progress_cb(0.85, "verifying download")
        md5_ok = await asyncio.to_thread(_verify_md5, tarball, _LIBRISPEECH_MD5)
        if not md5_ok:
            tarball.unlink(missing_ok=True)
            raise RuntimeError(
                "LibriSpeech tarball MD5 mismatch — the file did not match "
                f"the expected hash ({_LIBRISPEECH_MD5}). The upstream URL "
                "may have rotated. Check the Kaldi recipe at "
                "egs/librispeech/s5/local/download_and_untar.sh for the "
                "current hash."
            )

    if cancel_cb:
        await cancel_cb()
    if progress_cb:
        await progress_cb(0.88, "extracting tarball")
    await asyncio.to_thread(_extract_tar, tarball, target_dir)

    if cancel_cb:
        await cancel_cb()
    if progress_cb:
        await progress_cb(0.96, "cataloging files")
    catalog = await asyncio.to_thread(_scan_flac_catalog, _flac_root())
    if not catalog:
        raise RuntimeError(
            f"LibriSpeech extract appeared to succeed but no FLAC files "
            f"were found under {_flac_root()} — corpus is unusable."
        )

    # Now that everything's on disk, free the tarball — it's ~337 MB of
    # redundant data once the FLACs are extracted.
    tarball.unlink(missing_ok=True)

    manifest = _read_manifest()
    manifest["librispeech_dev_clean"] = {
        "complete": True,
        "downloaded_at": int(time.time()),
        "url": _LIBRISPEECH_URL,
        "md5": _LIBRISPEECH_MD5,
        "num_files": len(catalog),
        "total_bytes": sum(
            entry["size"] for entry in catalog if "size" in entry
        ),
        "catalog": catalog[:_CATALOG_MAX_FILES],
    }
    _write_manifest(manifest)

    if progress_cb:
        await progress_cb(1.0, "done")

    log.info(
        "negatives_corpus_installed",
        path=str(_librispeech_dir()),
        flac_count=len(catalog),
    )
    return installed_summary()


async def _download_streaming(
    url: str,
    dest: Path,
    *,
    expected_bytes: int,
    progress_cb: Callable[[float, str], Awaitable[None]] | None,
    progress_lo: float,
    progress_hi: float,
    cancel_cb: Callable[[], Awaitable[None]] | None,
) -> None:
    """Stream ``url`` to ``dest``. Maps download bytes to [lo, hi] in the
    parent progress range so a multi-stage handler can carve up the bar.

    No Range-resume — a stalled or interrupted download discards bytes
    and starts over. At 337 MB this is fine; for multi-GB corpora the
    multi-part scheme in ``gguf_download.py`` would be a better template.
    """
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.unlink(missing_ok=True)

    timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or expected_bytes or 0)
            done = 0
            last_emit = 0.0
            with open(tmp, "wb") as fh:
                async for chunk in resp.aiter_bytes(_DL_CHUNK_BYTES):
                    if cancel_cb:
                        await cancel_cb()
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if progress_cb and now - last_emit >= 0.5:
                        last_emit = now
                        frac = done / total if total else 0.0
                        frac = max(0.0, min(1.0, frac))
                        scaled = progress_lo + frac * (progress_hi - progress_lo)
                        await progress_cb(
                            scaled,
                            f"downloading {done // (1024 * 1024)} / "
                            f"{(total // (1024 * 1024)) if total else '?'} MB",
                        )
    os.replace(tmp, dest)


def _verify_md5(path: Path, expected: str) -> bool:
    """Return True iff ``path`` has the expected MD5. Reads in 4 MB chunks."""
    if not path.exists():
        return False
    hasher = hashlib.md5(usedforsecurity=False)  # checksum, not authentication
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(4 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected:
        log.warning(
            "negatives_corpus_md5_mismatch",
            path=str(path), expected=expected, actual=actual,
        )
        return False
    return True


def _extract_tar(tarball: Path, target_dir: Path) -> None:
    """Extract ``tarball`` into ``target_dir``. Refuses absolute paths or
    parent-traversal members so a malicious archive can't escape the
    corpus directory.

    Members are extracted one-by-one inside the validation loop. An
    earlier version of this function called ``extractall`` after the
    loop, which ignored the per-member checks entirely — and the
    ``except TypeError`` fallback for Python <3.12 stripped the
    ``filter='data'`` safety net too. Per-member extraction makes the
    validation actually load-bearing on every supported Python.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_abs = target_dir.resolve()
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            # Refuse absolute paths and ``..`` traversal.
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                log.warning(
                    "negatives_corpus_skip_unsafe_member", member=name,
                )
                continue
            resolved = (target_abs / name).resolve()
            try:
                resolved.relative_to(target_abs)
            except ValueError:
                log.warning(
                    "negatives_corpus_skip_escape_member", member=name,
                )
                continue
            # Extract this single validated member. ``filter='data'``
            # adds belt-and-braces safety (perm/UID stripping, refuses
            # device / link entries) on Python >= 3.12; older Pythons
            # fall back to plain extraction of just the safe payload.
            try:
                tf.extract(member, target_dir, filter="data")
            except TypeError:
                tf.extract(member, target_dir)


def _scan_flac_catalog(root: Path) -> list[dict[str, Any]]:
    """Build a list of (path, frames, size) entries for every .flac under root.

    Frames are reported by ``torchaudio.info`` — cheap (header read only,
    no decode). Files that fail to open are skipped with a warning.
    """
    catalog: list[dict[str, Any]] = []
    for path in root.rglob("*.flac"):
        try:
            info = torchaudio.info(str(path))
        except Exception:
            log.warning(
                "negatives_corpus_flac_info_failed", path=str(path),
            )
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        catalog.append({
            "path": str(path.relative_to(_flac_root())),
            "frames": int(info.num_frames),
            "sample_rate": int(info.sample_rate),
            "size": size,
        })
    return catalog


# ── Sampling ─────────────────────────────────────────────────────────


_catalog_cache: list[dict[str, Any]] | None = None


def _get_flac_catalog() -> list[dict[str, Any]]:
    """Lazy-load the catalog from the manifest. Cached in module scope."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache
    catalog = (_read_manifest().get("librispeech_dev_clean") or {}).get("catalog")
    if not isinstance(catalog, list):
        catalog = []
    _catalog_cache = catalog
    return _catalog_cache


def sample_real_speech_windows(
    count: int,
    rng: random.Random | None = None,
) -> list[np.ndarray]:
    """Return ``count`` random 1-second windows from LibriSpeech dev-clean.

    Each window is a 16,000-sample float32 array in [-1, 1] at 16 kHz mono.
    Files are decoded on-demand via torchaudio with frame_offset/num_frames
    so memory stays flat regardless of corpus size — we never load a full
    multi-minute utterance.

    Raises :class:`RuntimeError` if the corpus isn't installed. Caller
    must check :func:`is_installed` first (cheap) or be ready to fall
    back to synthetic negatives.
    """
    if not is_installed():
        raise RuntimeError(
            "negatives corpus not installed — call ensure_downloaded() "
            "first or check is_installed() before sampling"
        )
    catalog = _get_flac_catalog()
    if not catalog:
        raise RuntimeError(
            "LibriSpeech catalog empty — the manifest may be corrupt; "
            "delete /data/wake_word_corpora/manifest.json and re-run "
            "ensure_downloaded() to rescan."
        )
    rng = rng or random.Random()

    out: list[np.ndarray] = []
    attempts = 0
    max_attempts = count * 4   # bail out if too many files fail to decode
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        entry = rng.choice(catalog)
        frames = int(entry.get("frames", 0))
        if frames < WINDOW_SAMPLES:
            continue
        rel = entry["path"]
        path = _flac_root() / rel
        offset = rng.randint(0, frames - WINDOW_SAMPLES)
        try:
            wav, sr = torchaudio.load(
                str(path),
                frame_offset=offset,
                num_frames=WINDOW_SAMPLES,
            )
        except Exception:
            log.warning(
                "negatives_corpus_flac_decode_failed", path=str(path),
            )
            continue
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        samples = wav.squeeze(0).numpy().astype(np.float32)
        if samples.shape[0] != WINDOW_SAMPLES:
            # torchaudio.load can return a frame off when frame_offset
            # lands near EOF — pad or clip to the canonical length.
            if samples.shape[0] < WINDOW_SAMPLES:
                samples = np.pad(samples, (0, WINDOW_SAMPLES - samples.shape[0]))
            else:
                samples = samples[:WINDOW_SAMPLES]
        out.append(samples)

    if len(out) < count:
        log.warning(
            "negatives_corpus_sample_underflow",
            requested=count, got=len(out), attempts=attempts,
        )
    return out


def sample_silence_windows(
    count: int,
    rng: random.Random | None = None,
) -> list[np.ndarray]:
    """Synthesize ``count`` 1-second windows that approximate the quiet-
    room class missing from synthetic-only training.

    Mixed by class so the model sees enough variety in "no wake here":

      * 5%  — pure digital zeros (sanity baseline)
      * 80% — Gaussian noise at -50 to -30 dBFS (mic floor / room tone)
      * 15% — Low-pass-filtered noise at -45 to -25 dBFS (rumble / HVAC)

    None of this is meant to model adversarial near-trigger acoustics —
    it just teaches the model that "audio with no speech in it" is not
    a wake. Real speech that isn't the wake phrase comes from
    :func:`sample_real_speech_windows`.
    """
    rng = rng or random.Random()
    # Seed numpy from the user-provided rng so the same Random instance
    # reproduces noise the same way batch-to-batch.
    npr = np.random.default_rng(rng.randint(0, 2**31 - 1))

    out: list[np.ndarray] = []
    for _ in range(count):
        r = rng.random()
        if r < 0.05:
            out.append(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
            continue
        if r < 0.85:
            db = rng.uniform(-50.0, -30.0)
            amp = 10.0 ** (db / 20.0)
            samples = npr.normal(0.0, amp, WINDOW_SAMPLES).astype(np.float32)
        else:
            db = rng.uniform(-45.0, -25.0)
            amp = 10.0 ** (db / 20.0)
            raw = npr.normal(0.0, amp, WINDOW_SAMPLES).astype(np.float32)
            # First-order low-pass IIR — approximates rumble. Cutoff
            # ~200 Hz at our 16 kHz rate with alpha=0.92.
            alpha = 0.92
            samples = np.empty_like(raw)
            samples[0] = raw[0]
            for i in range(1, len(raw)):
                samples[i] = alpha * samples[i - 1] + (1.0 - alpha) * raw[i]
        out.append(np.clip(samples, -1.0, 1.0))
    return out


# ── Module-level invalidation ────────────────────────────────────────


def invalidate_cache() -> None:
    """Drop the in-memory catalog cache. Call after a re-download so the
    sampler picks up the new manifest without a process restart.
    """
    global _catalog_cache
    _catalog_cache = None
    log.debug("negatives_corpus_cache_invalidated")


__all__ = [
    "SAMPLE_RATE",
    "WINDOW_SAMPLES",
    "ensure_downloaded",
    "installed_summary",
    "invalidate_cache",
    "is_installed",
    "sample_real_speech_windows",
    "sample_silence_windows",
]
