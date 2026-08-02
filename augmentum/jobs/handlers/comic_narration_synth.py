"""``comic_narration_synth`` job — voice a comic into a motion-comic.

Per chapter: page through the comic (external provider images), OCR each page
with boxes, run the LLM script-assembly pass to turn rough lettering into a
clean reading-ordered script, TTS each line with a built-in engine, and stream
the audio to a WAV on disk while recording a **per-bubble timeline**
(page / bbox / text / kind / audio offset+duration) that drives the cast
pan-and-scan player.

Mirror of ``narration_synth`` (same built-in-engine WAV-concat + checkpoint +
artifact-save spine — reused directly), with two additions: the OCR→assembly
front-end (``augmentum/ocr``) and the timeline with per-line audio offsets.

Restart-survivable: the checkpoint is ``processed_pages`` on the
``comic_narrations`` row + the partial WAV + the persisted partial timeline —
re-entry resumes from the next page.
"""

from __future__ import annotations

import hashlib
import io
import os
import wave
from typing import Any

from augmentum.jobs import JobCancelled
from augmentum.jobs.context import JobContext
from augmentum.jobs.handlers.narration_synth import (
    _TTS_SAMPLE_RATE,
    _engine_wav_blobs,
    _resolve_synth_engine,
    _safe_name,
)
from augmentum.ocr.reading_order import (
    default_reading_direction,
    normalize_reading_direction,
)
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop
from augmentum.voice.wav_writer import WavWriter, wav_to_mp3

log = get_logger(__name__)


def _page_wav_path(data_dir: str, user_id: str, comic_kind: str, comic_ref: str, page: int) -> str:
    key = hashlib.sha1(f"comic|{user_id}|{comic_kind}|{comic_ref}".encode()).hexdigest()[:24]
    return os.path.join(data_dir or "/data", "comic_narration_work", f"{key}_p{page}.wav")


def _wav_ms(blob: bytes) -> int:
    """Duration in ms of a well-formed mono WAV blob (0 on parse failure)."""
    if not blob:
        return 0
    try:
        with wave.open(io.BytesIO(blob), "rb") as r:
            rate = r.getframerate() or _TTS_SAMPLE_RATE
            return int(1000 * r.getnframes() / rate)
    except Exception:  # noqa: BLE001 — bad blob contributes 0 ms
        return 0


async def _prune_old_narrations(store, astore, user_id: str, settings) -> None:
    """Drop finished narrations beyond ``comic_narration_cache_max`` (newest
    kept), deleting each pruned chapter's per-page audio artifacts. 0 disables
    pruning. Best-effort — a prune failure never fails the synth job."""
    import json as _json

    try:
        cap = int(getattr(settings, "comic_narration_cache_max", 12) or 0)
        if cap <= 0:
            return
        rows = await store.list_done(user_id=user_id)
        for row in rows[cap:]:
            try:
                pages = _json.loads(row.get("pages") or "[]")
            except Exception:  # noqa: BLE001
                pages = []
            for p in pages:
                aid = p.get("artifact_id")
                if not aid:
                    continue
                try:
                    await astore.delete(aid, user_id=user_id)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    log.warning("comic_narration_prune_artifact_failed", artifact_id=aid)
            await store.delete(
                row.get("comic_kind") or "file", row.get("comic_ref") or "",
                user_id=user_id,
            )
            log.info(
                "comic_narration_pruned",
                comic_ref=row.get("comic_ref"), pages=len(pages), cap=cap,
            )
    except Exception:  # noqa: BLE001
        log.warning("comic_narration_prune_failed", exc_info=True)


def make_comic_narration_synth_handler(app):
    """Build the ``comic_narration_synth`` handler bound to runtime services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        from augmentum.config import settings
        from augmentum.media.comic_pages import fetch_page_bytes, open_comic_source
        from augmentum.ocr import extract_page_script
        from augmentum.ocr.glossary import build_glossary, merge_page_terms, render_glossary
        from augmentum.state.comic_narration_store import ComicNarrationStore
        from augmentum.tools.artifact_storage import ArtifactStore

        sm = getattr(app.state, "state_manager", None)
        backend = getattr(sm, "backend", None) if sm else None
        if backend is None:
            raise RuntimeError("comic_narration_synth: state backend not initialized")
        conn = backend.conn

        comic_kind = str(ctx.payload.get("comic_kind") or "file")
        comic_ref = str(ctx.payload.get("comic_ref") or "")
        voice = str(ctx.payload.get("voice") or "")
        # Voice Cast — bare voices per register bucket, on the SAME engine as
        # ``voice`` (resolved by the begin route). Any unset bucket falls back to
        # the narrator voice, so a chapter with no casting reads in one voice
        # exactly as before. Legacy payloads carry voice_male/voice_female only.
        cast = ctx.payload.get("voice_cast")
        cast = cast if isinstance(cast, dict) else {}
        legacy_male = str(ctx.payload.get("voice_male") or "")
        legacy_female = str(ctx.payload.get("voice_female") or "")
        voice_by_speaker = {
            "m_low": str(cast.get("m_low") or legacy_male or voice),
            "m_high": str(cast.get("m_high") or cast.get("m_low") or legacy_male or voice),
            "f_low": str(cast.get("f_low") or legacy_female or voice),
            "f_high": str(cast.get("f_high") or cast.get("f_low") or legacy_female or voice),
            "narrator": voice,
        }
        title = str(ctx.payload.get("title") or "Comic")
        engine_id = str(ctx.payload.get("engine_id") or "kokoro-builtin")
        reading_direction = normalize_reading_direction(
            ctx.payload.get("reading_direction"), fallback=default_reading_direction(),
        )
        out_format = str(ctx.payload.get("format") or "mp3").lower()
        if out_format not in ("wav", "mp3"):
            out_format = "mp3"
        bitrate = int(ctx.payload.get("bitrate") or 128)
        if not comic_ref:
            raise RuntimeError("comic_narration_synth: malformed payload (no comic_ref)")

        # Resolve the transcription model ONCE for the whole chapter, so a run
        # can't be read by two different models partway through. The begin
        # route already preflighted this; re-resolving covers a resumed or
        # requeued job whose model changed since. Failure is terminal and its
        # message is written for the user, so it lands on the narration row
        # as-is.
        vlm_reader = None
        refine_reader = None
        if (getattr(settings, "ocr_engine", "docling") or "docling").lower() == "vlm":
            from augmentum.ocr.vlm_reader import resolve_reader

            vlm_reader = await resolve_reader(app)
            # Pass 2 proof-reads pass 1 against the same image, so it is also a
            # vision call and gets the same up-front capability check — finding
            # out the primary can't see on page 30 would waste the whole run.
            # Resolved to the SAME (base_url, model) is a valid outcome and the
            # simplest configuration; nothing here assumes they differ.
            if bool(getattr(settings, "ocr_vlm_second_pass_enabled", True)):
                refine_reader = await resolve_reader(
                    app,
                    role=getattr(settings, "ocr_vlm_second_pass_role", "primary") or "primary",
                    override=getattr(settings, "ocr_vlm_second_pass_model", "") or "",
                )

        store = ComicNarrationStore(conn)
        row = await store.get(comic_kind, comic_ref, user_id=ctx.user_id)
        if not row:
            row_id = await store.begin(
                comic_kind, comic_ref, voice, ctx.job_id,
                engine_id=engine_id, reading_direction=reading_direction, user_id=ctx.user_id,
            )
            prev_total = 0
            pages: list[dict] = []
        else:
            row_id = row["id"]
            prev_total = int(row.get("total_pages") or 0)
            try:
                import json as _json
                pages = _json.loads(row.get("pages") or "[]")
            except Exception:  # noqa: BLE001
                pages = []
        await store.mark_running(row_id)

        engine = _resolve_synth_engine(engine_id)
        data_dir = getattr(settings, "data_dir", "/data")
        os.makedirs(os.path.join(data_dir or "/data", "comic_narration_work"), exist_ok=True)

        try:
            idx = getattr(app.state, "file_index", None)
            if idx is None:
                raise RuntimeError("file index unavailable")
            entry = await idx.get(comic_ref, user_id=ctx.user_id)
            if not entry:
                raise RuntimeError("Comic not found")

            await ctx.update_progress(0.01, stage="Opening comic")
            src = await open_comic_source(app, entry, user_id=ctx.user_id)
            if not src:
                raise RuntimeError("Comic source unavailable (only Komga/Suwayomi support per-page narration)")
            total = int(src["page_count"] or 0)
            if total <= 0:
                raise RuntimeError("Comic has no pages")

            # Reading order: start where the READER is and wrap around. Someone
            # who opens to page 17 and presses Listen should not wait out
            # sixteen pages they've already read — but the chapter still has to
            # complete, so the sweep wraps to 1..16 afterwards rather than
            # skipping them.
            start_page = int(ctx.payload.get("start_page") or 0)
            if not (0 <= start_page < total):
                start_page = 0

            # Resume only when the page plan matches. Which pages are done is
            # read from ``pages`` itself, not from a count: a wrapped sweep
            # isn't a prefix, so "17 processed" no longer identifies WHICH 17.
            resume = (prev_total == total and len(pages) > 0)
            if not resume:
                pages = []
                await store.set_pages(row_id, [])
                await store.set_progress(row_id, 0, total)
            done: set[int] = {
                int(p.get("page", -1)) for p in pages if isinstance(p, dict)
            } if resume else set()
            page_order = [(start_page + i) % total for i in range(total)]

            # Chapter glossary, seeded from whatever previous runs already
            # transcribed. Streaming (per-page) rather than built from a full
            # cheap sweep first: a pre-sweep would mean no audio until every
            # page had been read once, and starting mid-chapter to hear THIS
            # page immediately is the point. The cost is that early pages see a
            # thinner glossary than late ones — which is exactly what seeding
            # from the persisted pages repairs on a re-listen.
            glossary_counts = build_glossary(pages if resume else [])

            # Load the built-in engine once (blocking load → thread).
            if not engine.is_available:
                await load_model_off_loop(engine.load_model)
            if not engine.is_available:
                raise RuntimeError(f"Built-in TTS '{engine_id}' is not available (model failed to load)")

            astore: ArtifactStore = getattr(app.state, "artifact_store", None) or ArtifactStore(conn)

            for page in page_order:
                if page in done:
                    continue
                await ctx.check_cancel()
                # Progress counts pages FINISHED, not the page index — under a
                # wrap those diverge, and a bar that jumps to 90% because the
                # user started near the end would be lying.
                await ctx.update_progress(
                    0.02 + 0.92 * (len(done) / total),
                    stage=f"Page {page + 1}/{total}",
                )
                img = await fetch_page_bytes(
                    src["http_client"], src["server"], src["provider"],
                    src["external_id"], src["extra"], page + 1,
                )
                if not img:
                    log.warning("comic_narration_page_missing", comic_ref=comic_ref, page=page + 1)
                    await _record_silent(store, row_id, pages, done, page, total)
                    continue

                # exclude_page: a term this page's own draft just invented must
                # not vouch for itself when the same page is proof-read.
                glossary = render_glossary(
                    glossary_counts,
                    min_sightings=int(getattr(settings, "ocr_vlm_glossary_min_sightings", 2) or 2),
                    max_terms=int(getattr(settings, "ocr_vlm_glossary_max_terms", 40) or 40),
                    exclude_page=page,
                ) if bool(getattr(settings, "ocr_vlm_glossary_enabled", True)) else []

                lines = await extract_page_script(
                    app, img, reading_direction=reading_direction, filename=f"page{page + 1}.jpg",
                    vlm_reader=vlm_reader, refine_reader=refine_reader, glossary=glossary,
                )
                merge_page_terms(glossary_counts, page, lines)

                # Synthesize THIS page into its own WAV (its own audio clock).
                page_wav = _page_wav_path(data_dir, ctx.user_id, comic_kind, comic_ref, page)
                writer = WavWriter(page_wav, sample_rate=_TTS_SAMPLE_RATE, resume=False)
                page_lines: list[dict] = []
                running_ms = 0
                try:
                    for ln in lines:
                        text = (ln.get("text") or "").strip()
                        if not text:
                            continue
                        # Swap voices by the reader's per-line speaker tag; an
                        # untagged line (or a slot the user left unset) reads in
                        # the narrator voice.
                        speaker = ln.get("speaker") or "narrator"
                        line_voice = voice_by_speaker.get(speaker, voice)
                        blobs = await _engine_wav_blobs(engine, text, line_voice)
                        dur = 0
                        for blob in blobs:
                            writer.append_wav(blob)
                            dur += _wav_ms(blob)
                        page_lines.append({
                            "order": len(page_lines),
                            "kind": ln.get("kind", "speech"),
                            "speaker": speaker,
                            "bbox": ln.get("bbox"),
                            "text": text,
                            "audio_start_ms": running_ms,
                            "audio_end_ms": running_ms + dur,
                        })
                        running_ms += dur
                    writer.close()
                finally:
                    writer.close()  # idempotent

                # Textless page (splash art) — no audio; advance the checkpoint.
                if writer.empty or not page_lines:
                    _safe_remove(page_wav)
                    await _record_silent(store, row_id, pages, done, page, total)
                    continue

                # Encode this page's audio and save it as its own artifact, so
                # the player can start on it immediately.
                out_path, fmt = page_wav, "wav"
                if out_format == "mp3":
                    mp3_path = os.path.splitext(page_wav)[0] + ".mp3"
                    if await ctx.run_in_thread(lambda s=page_wav, d=mp3_path: wav_to_mp3(s, d, bitrate_kbps=bitrate)):
                        out_path, fmt = mp3_path, "mp3"

                # transient=True: per-page audio is regenerable playback
                # cache (30+ per chapter) — keep it out of the Files/library
                # surfaces instead of flooding them one artifact per page.
                saved = await astore.save_from_path(
                    out_path,
                    f"{_safe_name(title)} p{page + 1}.{fmt}",
                    fmt,
                    display_name=f"{title} — page {page + 1}",
                    user_id=ctx.user_id,
                    transient=True,
                    metadata={
                        "comic_narration_for": comic_ref,
                        "comic_kind": comic_kind,
                        "voice": voice,
                        "page": page,
                    },
                )
                _safe_remove(page_wav)
                if out_path != page_wav:
                    _safe_remove(out_path)

                pages.append({
                    "page": page,
                    "artifact_id": saved["id"],
                    "duration_ms": running_ms,
                    "lines": page_lines,
                })
                pages.sort(key=lambda p: p.get("page", 0))
                done.add(page)
                await store.set_pages(row_id, pages)
                await store.set_progress(row_id, len(done), total)

            if not any((p.get("lines") or []) for p in pages):
                raise RuntimeError("No readable text in this comic")

            await store.mark_done(row_id, pages=pages)
            # Retention: narration audio is regenerable cache and accumulates
            # ~30 artifacts per chapter — without a cap a binge-read session
            # leaves hundreds of files behind. Keep the newest N finished
            # chapters per user; prune the rest (row + page artifacts).
            await _prune_old_narrations(store, astore, ctx.user_id, settings)
            await ctx.update_progress(1.0, stage="Done")
            return {
                "pages": len(pages),
                "bubbles": sum(len(p["lines"]) for p in pages),
                "format": out_format,
            }
        except JobCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — record + re-raise → job marked failed
            try:
                await store.mark_failed(row_id, str(exc))
            except Exception:  # noqa: BLE001
                log.warning("comic_narration_synth_mark_failed_errored", job_id=ctx.job_id)
            raise

    return handler


async def _record_silent(store, row_id, pages, done, page: int, total: int) -> None:
    """Checkpoint a page that produced no audio (splash art, missing image).

    Recorded as a real entry with no ``artifact_id`` rather than as a bare
    counter bump, for two reasons. Resume needs to know this page was ATTEMPTED
    — under a wrapped sweep the checkpoint is a set of page numbers, and a
    silent page absent from it would be re-read on every resume, forever. And
    the player can then say "no narration for this page" instead of holding on
    "synthesizing…" for a page that will never arrive.
    """
    if page in done:
        return
    pages.append({"page": page, "artifact_id": "", "duration_ms": 0, "lines": []})
    pages.sort(key=lambda p: p.get("page", 0))
    done.add(page)
    await store.set_pages(row_id, pages)
    await store.set_progress(row_id, len(done), total)


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
