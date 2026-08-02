"""``narration_synth`` job handler — synthesize a full EPUB to an audiobook.

Builds the "audio partner" for an EPUB: reads the spine as plain text,
splits into provider-sized chunks, synthesizes each with a built-in TTS
engine (Kokoro / Pocket TTS), and **streams the PCM straight to a WAV file
on disk** as it goes — so a multi-hour book never holds the whole audio in
memory. The partial file is the checkpoint: ``processed_chunks`` on the
``epub_narrations`` row records how far we got, the partial path is
deterministic from (user, epub), so a crash/restart **resumes** from the
last finished chunk instead of from chapter one. On completion the WAV is
(optionally) transcoded to MP3 file→file via ffmpeg and saved as an audio
artifact (which lands in Files → Audio).

Built-in engines only: they yield well-formed per-segment WAV from
``stream_speech(response_format="wav")``; external HTTP providers chunk a
single stream arbitrarily, which we can't merge — the routes reject them.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import wave
from typing import Any

from augmentum.jobs import JobCancelled
from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop
from augmentum.vfs import epub_extractor
from augmentum.voice.wav_writer import WavWriter, wav_to_mp3

log = get_logger(__name__)

_MAX_CHUNK_CHARS = 2500
_TTS_SAMPLE_RATE = 24_000   # Kokoro emits 24 kHz mono; Pocket resamples to match


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split prose into ≤max_chars pieces on paragraph/sentence boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    out: list[str] = []
    buf = ""
    # Paragraph-first, then sentence-level for overlong paragraphs.
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        units = [para] if len(para) <= max_chars else re.findall(r"[^.!?]+[.!?]+|\S[^.!?]*$", para)
        for u in units:
            u = u.strip()
            if not u:
                continue
            if len(u) > max_chars:
                # Hard wrap a runaway sentence.
                for i in range(0, len(u), max_chars):
                    if buf:
                        out.append(buf)
                        buf = ""
                    out.append(u[i:i + max_chars])
                continue
            if buf and len(buf) + 1 + len(u) > max_chars:
                out.append(buf)
                buf = u
            else:
                buf = f"{buf}\n{u}" if buf else u
        if buf:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _chapters_to_text(chapters: list[dict]) -> str:
    parts: list[str] = []
    for ch in chapters or []:
        if not ch:
            continue
        h = (ch.get("heading") or "").strip()
        t = (ch.get("text") or "").strip()
        if h and not t.lower().startswith(h.lower()):
            parts.append(h)
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def _concat_wav(wav_blobs: list[bytes]) -> bytes:
    """Concatenate well-formed mono WAV blobs into one WAV file."""
    out = io.BytesIO()
    writer: wave.Wave_write | None = None
    for blob in wav_blobs:
        if not blob:
            continue
        with wave.open(io.BytesIO(blob), "rb") as r:
            if writer is None:
                writer = wave.open(out, "wb")
                writer.setparams(r.getparams())
            writer.writeframes(r.readframes(r.getnframes()))
    if writer is None:
        raise RuntimeError("TTS produced no audio")
    writer.close()
    return out.getvalue()


def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", (s or "").strip()) or "ebook"
    return s[:80]


def _partial_path(data_dir: str, user_id: str, epub_kind: str, epub_ref: str) -> str:
    """Deterministic on-disk path for this (user, epub)'s in-progress WAV."""
    key = hashlib.sha1(f"{user_id}|{epub_kind}|{epub_ref}".encode()).hexdigest()[:24]
    return os.path.join(data_dir or "/data", "narration_work", f"{key}.wav")


async def _resolve_epub_path(app, backend, epub_kind: str, epub_ref: str, user_id: str) -> str | None:
    import os

    if epub_kind == "artifact":
        store = getattr(app.state, "artifact_store", None)
        if not store:
            return None
        info = await store.get(epub_ref, user_id=user_id)
        if not info or info.get("format") != "epub":
            return None
        fp = store.get_file_path(info.get("path", ""))
        return str(fp) if fp and fp.is_file() else None
    if epub_kind == "file":
        idx = getattr(app.state, "file_index", None)
        if not idx:
            return None
        entry = await idx.get(epub_ref, user_id=user_id)
        if not entry or not entry.name.lower().endswith(".epub"):
            return None
        if entry.real_path and os.path.exists(entry.real_path):
            return entry.real_path
        return None
    return None


def _resolve_synth_engine(engine_id: str):
    """Return the loaded in-process TTS engine for ``engine_id``, or None.

    Covers ``kokoro-builtin`` and ``pockettts-builtin`` — both yield
    well-formed per-segment WAV from ``stream_speech(response_format="wav")``,
    which :func:`_concat_wav` stitches into the final audiobook.
    """
    from augmentum.config import settings
    if engine_id == "pockettts-builtin":
        from augmentum.voice.pocket_tts import PocketTTS
        return PocketTTS.instance(
            model_dir=settings.tts_pocket_model_dir,
            language=settings.tts_pocket_language,
        )
    from augmentum.voice.kokoro_tts import KokoroTTS
    return KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)


async def _engine_wav_blobs(engine, text: str, voice: str) -> list[bytes]:
    """Synthesize one text chunk with a built-in engine → list of WAV blobs.

    Kokoro and Pocket both emit ONE buffered WAV per response, split across
    the stream as a 44-byte header chunk followed by raw PCM body chunks
    (~16KB strides for cancellation-friendly delivery). Reassemble the
    stream into a single WAV blob so the caller's WAV parser sees a
    well-formed file — the previous "treat each yield as its own WAV"
    shape broke on the first PCM-only chunk with "file does not start
    with RIFF id".

    Defensive prefix strip: callers that resolve through
    ``resolve_voice_provider`` already pass a bare voice name, but
    in-flight jobs queued before the narration_common fix carry the
    ``provider_id::voice`` form in their payload. Strip here so those
    jobs synthesize with the chosen voice instead of silently falling
    back to the engine default.
    """
    from augmentum.voice.text_cleaning import clean_for_tts

    if not engine.is_available:
        # Model load is blocking — keep it off the event loop.
        await load_model_off_loop(engine.load_model)
    if not engine.is_available:
        raise RuntimeError("Built-in TTS engine is not available (model failed to load)")
    bare_voice = voice or ""
    if "::" in bare_voice:
        _, bare_voice = bare_voice.split("::", 1)
    cleaned = clean_for_tts(text) or text
    buf = bytearray()
    async for blob in engine.stream_speech(cleaned, voice=bare_voice, speed=1.0, response_format="wav"):
        if blob:
            buf += blob
    return [bytes(buf)] if buf else []


def make_narration_synth_handler(app):
    """Build the ``narration_synth`` handler bound to runtime services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:

        from augmentum.config import settings
        from augmentum.state.narration_store import NarrationStore
        from augmentum.tools.artifact_storage import ArtifactStore

        sm = getattr(app.state, "state_manager", None)
        backend = getattr(sm, "backend", None) if sm else None
        if backend is None:
            raise RuntimeError("narration_synth: state backend not initialized")
        conn = backend.conn

        epub_kind = str(ctx.payload.get("epub_kind") or "")
        epub_ref = str(ctx.payload.get("epub_ref") or "")
        voice = str(ctx.payload.get("voice") or "")
        title = str(ctx.payload.get("title") or "Ebook")
        # `engine_id` was added later — older queued jobs default to Kokoro.
        engine_id = str(ctx.payload.get("engine_id") or "kokoro-builtin")
        out_format = str(ctx.payload.get("format") or "mp3").lower()
        if out_format not in ("wav", "mp3"):
            out_format = "mp3"
        bitrate = int(ctx.payload.get("bitrate") or 128)
        if epub_kind not in ("artifact", "file") or not epub_ref:
            raise RuntimeError("narration_synth: malformed payload")
        engine = _resolve_synth_engine(engine_id)

        nstore = NarrationStore(conn)
        row = await nstore.get(epub_kind, epub_ref, user_id=ctx.user_id)
        if not row:
            # Routes create the row before enqueuing; tolerate a missing row.
            row_id = await nstore.begin(epub_kind, epub_ref, voice, ctx.job_id, user_id=ctx.user_id)
            done_so_far, prev_total = 0, 0
        else:
            row_id = row["id"]
            done_so_far = int(row.get("processed_chunks") or 0)
            prev_total = int(row.get("total_chunks") or 0)
        await nstore.mark_running(row_id)

        partial = _partial_path(getattr(settings, "data_dir", "/data"), ctx.user_id, epub_kind, epub_ref)

        try:
            path = await _resolve_epub_path(app, backend, epub_kind, epub_ref, ctx.user_id)
            if not path:
                raise RuntimeError("EPUB file not found on disk")

            await ctx.update_progress(0.01, stage="Reading EPUB")
            chapters = await ctx.run_in_thread(lambda: epub_extractor.chapters_text(path))
            chunks = _chunk_text(_chapters_to_text(chapters))
            if not chunks:
                raise RuntimeError("No readable text in this EPUB")
            total = len(chunks)

            # Resume only when the prior run matches this exact chunk plan and
            # the partial file is still there; otherwise start fresh (the
            # WavWriter in non-resume mode truncates any stale partial).
            resume = (done_so_far > 0 and prev_total == total and os.path.exists(partial))
            start = done_so_far if resume else 0
            if not resume:
                await nstore.set_progress(row_id, 0, total)
            else:
                log.info("narration_resume", job_id=ctx.job_id, from_chunk=start, total=total)

            # Load the engine once (blocking model load → thread).
            if not engine.is_available:
                await load_model_off_loop(engine.load_model)
            if not engine.is_available:
                raise RuntimeError(f"Built-in TTS '{engine_id}' is not available (model failed to load)")

            writer = WavWriter(partial, sample_rate=_TTS_SAMPLE_RATE, resume=resume)
            try:
                from augmentum.voice import lexicon_store
                for i in range(start, total):
                    await ctx.check_cancel()
                    await ctx.update_progress(0.02 + 0.92 * (i / total), stage=f"Synthesizing {i + 1}/{total}")
                    # Per-voice pronunciation lexicon (migration 261),
                    # applied after chunking so the chunk plan (and thus
                    # resume's total comparison) is unaffected. Long-form
                    # narration is where a mispronounced name hurts most —
                    # it repeats for the whole book.
                    chunk_text = await lexicon_store.apply(
                        conn, chunks[i], user_id=ctx.user_id, voice=voice,
                    )
                    for blob in await _engine_wav_blobs(engine, chunk_text, voice):
                        writer.append_wav(blob)
                    writer.flush_header()
                    await nstore.set_progress(row_id, i + 1, total)
                writer.close()
                if writer.empty:
                    raise RuntimeError("Synthesis produced no audio")

                await ctx.update_progress(0.97, stage="Encoding")
                out_path, fmt = partial, "wav"
                if out_format == "mp3":
                    mp3_path = os.path.splitext(partial)[0] + ".mp3"
                    if await ctx.run_in_thread(lambda: wav_to_mp3(partial, mp3_path, bitrate_kbps=bitrate)):
                        out_path, fmt = mp3_path, "mp3"

                await ctx.update_progress(0.99, stage="Saving")
                store: ArtifactStore = getattr(app.state, "artifact_store", None) or ArtifactStore(conn)
                saved = await store.save_from_path(
                    out_path,
                    f"{_safe_name(title)} narration.{fmt}",
                    fmt,
                    display_name=f"{title} (narration)",
                    user_id=ctx.user_id,
                    metadata={
                        "narration_for": epub_ref,
                        "narration_kind": epub_kind,
                        "voice": voice,
                        "chunks": total,
                    },
                )
                # save_from_path moved out_path away; clean up the leftover WAV
                # if we transcoded (out_path was the mp3, partial still exists).
                if out_path != partial:
                    try:
                        os.remove(partial)
                    except OSError:
                        pass
            finally:
                writer.close()   # idempotent — guarantees a valid partial WAV if we bailed mid-loop

            await nstore.mark_done(row_id, saved["id"])
            await ctx.update_progress(1.0, stage="Done")
            return {"narration_artifact_id": saved["id"], "chunks": total, "format": fmt, "bytes": saved.get("size_bytes")}
        except JobCancelled:
            # Partial WAV is left intact (a valid file, with processed_chunks
            # recorded) — a retry/restart resumes from there.
            raise
        except Exception as exc:  # noqa: BLE001 — record + re-raise so the job is marked failed
            try:
                await nstore.mark_failed(row_id, str(exc))
            except Exception:  # noqa: BLE001
                log.warning("narration_synth_mark_failed_errored", job_id=ctx.job_id)
            raise

    return handler
