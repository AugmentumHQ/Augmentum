"""Subprocess worker for ZIM-to-augpack conversion.

Runs in a separate process so CPU-intensive embedding doesn't starve
the main server.  Communicates progress via a JSON file that the
parent process polls.

Pipeline shape: a producer thread iterates ZIM entries, strips HTML,
filters by length, and pushes accepted chunks onto a bounded queue.
The main thread consumes the queue in batches, embeds, and writes to
SQLite. Producer + consumer run concurrently so the embedder starts
work as soon as the first chunk lands instead of after the whole ZIM
is extracted into RAM. Caps memory at queue_maxsize × chunk_size and
cuts wallclock by overlapping I/O with compute.

Usage:
    python -m augmentum.knowledge.convert_worker \\
        --zim /path/to/file.zim \\
        --output /path/to/file.augpack \\
        --pack-name my_pack \\
        --progress /tmp/progress.json \\
        [--batch-size 32] [--use-gpu]
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import sqlite3
import struct
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Any

_MAX_CONTENT = 4000
_MIN_CONTENT = 30
# Bounded queue between producer (extract) and consumer (embed). Tuned to
# cap memory near (1024 chunks × ~4KB) ≈ 4MB headroom — enough to keep the
# embedder fed without buffering an entire Wikipedia in RAM.
_EXTRACT_QUEUE_MAX = 1024
# Sentinel pushed by the producer when extraction completes successfully.
_END_OF_STREAM = object()

# Per-call token budget for the embedder. Fastembed pads each batch to its
# longest member, so the real memory cost is ``count × max_seq``, not just
# count. Capping the *product* gives a hard memory ceiling regardless of
# input shape: worst-case attention scratch =
# (budget / max_seq) × heads × max_seq² × 4 = budget × heads × max_seq × 4.
# With budget=32K, max_seq=1024 (set in EmbeddingService.MAX_SEQ_TOKENS),
# heads=12, the cap is ~1.5 GB on GPU — safe on any consumer card. The
# user's --batch-size argument still applies as an upper bound on count.
_MAX_TOKENS_PER_BATCH = 32_768


def _would_exceed_budget(
    cur_count: int, cur_max_seq: int, candidate_seq: int,
    max_count: int, token_budget: int,
) -> bool:
    """Return True if adding a candidate would push the batch past either
    the count cap or the token-product budget. Empty batch never overflows."""
    if cur_count == 0:
        return False
    if cur_count + 1 > max_count:
        return True
    new_max_seq = max(cur_max_seq, candidate_seq)
    return (cur_count + 1) * new_max_seq > token_budget


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _write_progress(path: Path, stage: str, current: int, total: int, error: str = "") -> None:
    data = {"stage": stage, "current": current, "total": total, "error": error}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # atomic on POSIX


def _extract_one(archive: Any, idx: int) -> dict | None:
    """Read entry ``idx`` from the archive, return a chunk dict or None.

    Returns None for any entry we want to skip: non-text mimetype, content
    shorter than _MIN_CONTENT, or a libzim read error. The filter rules are
    deterministic against a given .zim file so the chunk-index → entry-index
    mapping is stable across runs (load-bearing for resume — see the
    resume_from comment in main()).
    """
    try:
        entry = archive._get_entry_by_id(idx)
        item = entry.get_item()
        if "text" not in item.mimetype.lower():
            return None
        raw = item.content.tobytes().decode("utf-8", errors="replace")
        text = _strip_html(raw)
        if len(text) < _MIN_CONTENT:
            return None
        return {
            "title": entry.title,
            "content": text[:_MAX_CONTENT],
            "url": getattr(entry, "path", ""),
        }
    except Exception:
        return None


def _extract_producer(
    archive: Any,
    total_entries: int,
    out_queue: queue.Queue,
    skip_count: int,
    stop_event: threading.Event,
    stats: list,
) -> None:
    """Stream chunks from the ZIM into ``out_queue``.

    ``skip_count`` chunks are silently dropped at the head (used for
    resume — chunks 0..skip_count-1 are already in the .augpack, no need
    to re-emit). ``stop_event`` lets the consumer halt the producer if
    embedding fails. On any unhandled error, pushes the exception to the
    queue so the consumer can re-raise with traceback.

    ``stats`` is a 1-element list used as a writeback channel for the
    total number of valid chunks observed (whether skipped or emitted).
    The consumer reads ``stats[0]`` after joining the thread to detect
    resume-against-shrunken-.zim — see the sanity check in main().
    """
    try:
        observed = 0
        for idx in range(total_entries):
            if stop_event.is_set():
                return
            chunk = _extract_one(archive, idx)
            if chunk is None:
                continue
            observed += 1
            if observed <= skip_count:
                continue
            out_queue.put(chunk)
        stats[0] = observed
        out_queue.put(_END_OF_STREAM)
    except BaseException as exc:  # noqa: BLE001 — propagate to consumer
        out_queue.put(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zim", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pack-name", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--description", default="")
    parser.add_argument("--license", default="CC BY-SA")
    # Resume mode: open the existing .augpack instead of creating a fresh one,
    # detect how many chunks already have embeddings, and skip re-embedding
    # them. Used after a prior run died mid-embed (OOM, CUDA error, server
    # restart) — saves the embedding cost on huge packs (Wikipedia = days).
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    zim_path = Path(args.zim)
    output_path = Path(args.output)
    progress_path = Path(args.progress)

    # DB handle hoisted out of the try so the outer except can close it
    # cleanly. Leaving it open on a crash leaves a partial .augpack in WAL/
    # journal limbo and turns the next resume into a recovery problem.
    db: sqlite3.Connection | None = None

    # Tracking for the producer thread so the outer except can shut it
    # down cleanly on consumer-side failures.
    producer_thread: threading.Thread | None = None
    producer_stop = threading.Event()

    try:
        _write_progress(progress_path, "extracting", 0, 0)

        import libzim  # type: ignore[import-untyped]
        archive = libzim.Archive(str(zim_path))
        total_entries = archive.entry_count

        # Force CPU execution provider unless explicitly told to use GPU.
        # When onnxruntime-gpu is installed, ONNX defaults to CUDA which
        # causes OOM when embedding 1000+ chunks — the FusedMatMul node
        # tries to allocate a single huge buffer on GPU for the batch.
        if not args.use_gpu:
            import os
            os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"
            # Also tell fastembed to use CPU providers explicitly
            os.environ["ORT_PROVIDERS"] = "CPUExecutionProvider"

        from augmentum.memory.embeddings import EmbeddingService

        has_gpu = False
        if args.use_gpu:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    has_gpu = True
                else:
                    print(f"GPU not available, using CPU. Providers: {providers}", flush=True)
            except ImportError:
                pass

        # batch_size is the upper bound on chunks per embed() call. Memory is
        # ALSO bounded independently by _MAX_TOKENS_PER_BATCH (count × longest
        # sequence) — the consumer flushes on whichever fires first. With
        # EmbeddingService.MAX_SEQ_TOKENS=1024 and budget=32K, worst-case
        # attention scratch is ~1.5GB on GPU regardless of batch_size or
        # corpus density. So the user's batch_size acts as a soft preference
        # rather than the load-bearing safety knob it used to be.
        batch_size = args.batch_size

        # Probe dim by embedding a constant sentinel. This also forces the
        # ONNX model to load NOW (the load is ~5-10s of cold start) so we
        # know the schema width before opening the DB. Doing it ahead of
        # the producer means model-load time overlaps with extraction once
        # we start the thread. The result is discarded.
        sample_emb = EmbeddingService.embed(["dim probe"])[0]
        dim = len(sample_emb)

        # Grab the tokenizer for length-aware batching. Best-effort — if
        # fastembed's internals shift we fall back to count-only batching
        # (the tokenizer cap inside EmbeddingService still bounds per-seq
        # length, just without the per-batch token product check).
        _embed_model = EmbeddingService.get_model()
        embed_tokenizer = getattr(getattr(_embed_model, "model", None),
                                  "tokenizer", None)

        # Resume detection. If --resume was passed and the augpack exists
        # with at least one committed chunk, pick up from MAX(id)+1. Otherwise
        # behave like a fresh run (delete any leftover and create schema).
        #
        # Why MAX(id) is the resume point: the embed loop commits per-batch
        # (db.commit() at end of each batch). If the worker dies mid-batch,
        # SQLite rolls back the uncommitted writes for that batch on next
        # open, so MAX(id) reflects the last fully-committed chunk. The
        # extraction is deterministic against a given .zim file (range(
        # total_entries) iteration + deterministic mimetype/length filters),
        # so chunk N from this run equals chunk N from the original run —
        # the producer's skip_count drops chunks 0..MAX(id), the consumer
        # picks up at MAX(id)+1. If the user swaps the .zim between runs
        # this assumption breaks and the resume corrupts the pack; out of
        # scope to detect.
        resume_from = -1
        is_resume = bool(args.resume) and output_path.exists() and output_path.stat().st_size > 0

        if is_resume:
            # Verify the existing augpack is compatible before opening for
            # write. A dim mismatch (user upgraded the embedding model since
            # the failed run) would silently produce a hybrid pack with
            # half-768-dim, half-other-dim vectors that vec0 can't query.
            try:
                _probe = sqlite3.connect(str(output_path))
                _probe.enable_load_extension(True)
                import sqlite_vec
                sqlite_vec.load(_probe)
                _probe.enable_load_extension(False)
                cur = _probe.execute("SELECT embedding FROM chunks_vec LIMIT 1")
                row = cur.fetchone()
                if row is not None:
                    existing_dim = len(row[0]) // 4  # FLOAT32 = 4 bytes
                    if existing_dim != dim:
                        _probe.close()
                        raise RuntimeError(
                            f"Cannot resume: existing augpack has embedding dim "
                            f"{existing_dim} but current model produces dim {dim}. "
                            f"Discard and re-convert from scratch."
                        )
                cur = _probe.execute("SELECT COALESCE(MAX(id), -1) FROM chunks")
                resume_from = int(cur.fetchone()[0])
                _probe.close()
            except sqlite3.Error as exc:
                # Schema missing or corrupt — can't safely resume. Fall back
                # to fresh by clearing the resume flag; the user-visible
                # error path is still informative because we re-create.
                print(f"Resume probe failed ({exc}); starting fresh", flush=True)
                is_resume = False
                resume_from = -1

        if is_resume and resume_from >= 0:
            # Open existing pack for append.
            db = sqlite3.connect(str(output_path))
            db.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            print(
                f"Resuming from chunk {resume_from + 1} "
                f"(skipping {resume_from + 1} already-embedded)",
                flush=True,
            )
        else:
            # Fresh build: drop any leftover and create schema.
            if output_path.exists():
                output_path.unlink()

            db = sqlite3.connect(str(output_path))
            db.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(db)
            db.enable_load_extension(False)

            db.execute("""
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT,
                    source TEXT NOT NULL,
                    url TEXT,
                    chunk_index INTEGER DEFAULT 0
                )
            """)
            db.execute(f"""
                CREATE VIRTUAL TABLE chunks_vec USING vec0(
                    id INTEGER PRIMARY KEY,
                    embedding FLOAT32[{dim}]
                )
            """)
            db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

        # ``written`` is the global chunk-id of the next slot to write —
        # equals resume_from+1 (so 0 on fresh runs, MAX(id)+1 on resumes).
        # Used both as the SQL primary key and as the progress numerator.
        written = resume_from + 1
        skip_count = written  # producer drops the first ``skip_count`` chunks

        # Start the extraction producer. The bounded queue caps memory; the
        # producer pushes chunks as it finds them and a sentinel when done.
        # On any error, the producer pushes the exception object so the
        # consumer can re-raise with traceback. ``producer_stats`` is a
        # writeback channel (1-element list) used to detect resume-against-
        # shrunken-.zim after the producer joins.
        chunk_queue: queue.Queue = queue.Queue(maxsize=_EXTRACT_QUEUE_MAX)
        producer_stats: list[int] = [0]
        producer_thread = threading.Thread(
            target=_extract_producer,
            args=(archive, total_entries, chunk_queue, skip_count,
                  producer_stop, producer_stats),
            name="zim-extract",
            daemon=True,
        )
        producer_thread.start()

        # Initial progress write — total_entries is an upper bound on chunks
        # since not every entry passes the mimetype/length filters. The bar
        # under-shoots a little until the producer finishes and we know the
        # real total, at which point we update with the actual chunk count.
        _write_progress(progress_path, "embedding", written, total_entries)

        # Drain the queue. Each chunk is pre-tokenized so the consumer can
        # honor _MAX_TOKENS_PER_BATCH (count × longest-seq budget) in
        # addition to the user's batch_size count cap. Flushing earlier on
        # token pressure keeps GPU memory predictable when a corpus mixes
        # short prose with code-heavy outliers.
        batch: list[dict] = []
        batch_max_seq = 0
        producer_done = False

        def _flush() -> None:
            nonlocal batch, batch_max_seq, written
            if not batch:
                return
            batch_texts = [c["content"] for c in batch]
            batch_embeddings = EmbeddingService.embed(batch_texts)
            for j, (chunk, emb) in enumerate(zip(batch, batch_embeddings)):
                idx = written + j
                db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                    (idx, chunk["content"], chunk["title"], "", "zim",
                     chunk["url"], 0),
                )
                blob = struct.pack(f"<{dim}f", *emb)
                db.execute("INSERT INTO chunks_vec VALUES (?,?)", (idx, blob))
            db.commit()
            written += len(batch)
            _write_progress(progress_path, "embedding", written, total_entries)
            batch, batch_max_seq = [], 0

        while not producer_done:
            item = chunk_queue.get()
            if item is _END_OF_STREAM:
                producer_done = True
            elif isinstance(item, BaseException):
                # Producer hit an unrecoverable error; re-raise here so the
                # outer except handles cleanup uniformly.
                raise item
            else:
                # Tokenize before deciding to flush so we know whether THIS
                # chunk would push us over budget. The "search_document: "
                # prefix added inside EmbeddingService is short (≤4 tokens);
                # we ignore it for sizing — the safety margin in the budget
                # absorbs that rounding.
                if embed_tokenizer is not None:
                    candidate_seq = len(
                        embed_tokenizer.encode(item["content"]).ids
                    )
                else:
                    # Fallback: assume worst-case capped by tokenizer hard
                    # cap. Triggers count-only batching (still safe because
                    # MAX_SEQ_TOKENS bounds per-seq).
                    candidate_seq = EmbeddingService.MAX_SEQ_TOKENS
                # Flush BEFORE adding if this candidate would overflow.
                if _would_exceed_budget(
                    len(batch), batch_max_seq, candidate_seq,
                    batch_size, _MAX_TOKENS_PER_BATCH,
                ):
                    _flush()
                batch.append(item)
                batch_max_seq = max(batch_max_seq, candidate_seq)

            # Flush the trailing partial batch when the producer has
            # signalled end-of-stream.
            if producer_done:
                _flush()

        producer_thread.join()
        producer_thread = None
        total_chunks = written
        observed_in_zim = producer_stats[0]

        # Sanity check resume against the .zim's actual chunk yield. If the
        # user replaced the source file with a smaller one, the producer
        # observed fewer valid chunks than the existing pack already holds —
        # we'd silently write meta with a chunk_count that doesn't match
        # the underlying rows. Refuse rather than corrupt.
        if is_resume and observed_in_zim < skip_count:
            raise RuntimeError(
                f"Resume impossible: existing augpack has {skip_count} chunks "
                f"but extraction from this .zim produced only {observed_in_zim}. "
                f"The .zim may have been replaced. Discard and re-convert."
            )

        action = "Resumed" if is_resume else "Embedded"
        print(f"{action} + packaged {total_chunks} chunks (dim={dim})", flush=True)

        today = date.today().isoformat()
        meta = {
            "name": args.pack_name,
            "version": today,
            "description": args.description,
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
            "embedding_dim": str(dim),
            "chunk_count": str(total_chunks),
            "source_license": args.license,
            "build_date": today,
        }
        # Use INSERT OR REPLACE because resume runs that originally created
        # the schema may have an empty meta table; either path needs the
        # final values written cleanly.
        for k, v in meta.items():
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))

        # Same idempotency for the index — resume on a partial pack that
        # somehow already created the index (shouldn't happen with current
        # ordering but harmless to guard) won't crash.
        db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(title)")
        db.commit()
        db.execute("VACUUM")
        db.close()
        db = None

        _write_progress(progress_path, "done", total_chunks, total_chunks)
        print(f"Packaged to {output_path} ({output_path.stat().st_size} bytes)", flush=True)
        return 0

    except Exception as exc:
        # Signal the extraction thread to stop as soon as it checks the
        # event (producer reads stop_event between entries). Also drain the
        # queue once so the producer doesn't block on a full queue while
        # we're trying to shut down. Daemon=True means the thread won't
        # prevent process exit even if something hangs, but we try to be
        # cooperative.
        producer_stop.set()
        if producer_thread is not None and producer_thread.is_alive():
            try:
                while True:
                    chunk_queue.get_nowait()
            except queue.Empty:
                pass
            except NameError:
                pass
            producer_thread.join(timeout=2.0)

        # Close the DB cleanly so the partial augpack is in a coherent
        # state for the next resume attempt. Critically, we ROLLBACK rather
        # than commit: a failure mid-batch (embed raised after some inserts
        # but before the batch commit) would otherwise leave a chunks row
        # without a matching chunks_vec row — resume's MAX(id) would then
        # think those chunks were complete and skip re-embedding them. The
        # batch boundary is the unit of consistency; rollback drops the
        # in-flight batch and leaves MAX(id) at the last clean batch.
        if db is not None:
            try:
                db.rollback()
            except sqlite3.Error:
                pass
            try:
                db.close()
            except sqlite3.Error:
                pass
        _write_progress(progress_path, "error", 0, 0, error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
