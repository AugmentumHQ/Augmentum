"""Codebase indexer — persistent semantic search for workspace files.

Builds and maintains a sqlite-vec index of workspace file contents.
Runs as a background task on workspace enter, updates incrementally
on file changes. The agent queries this index for semantic code search
instead of (or alongside) grep-based keyword matching.

Architecture:
- Index stored as `/workspace/.augmentum/index.db` (inside the persistent volume)
- Chunks: ~200 line blocks per file, with file path + line range metadata
- Embeddings: same EmbeddingService (nomic-embed-text) used by memory system
- Incremental: tracks file mtimes, only re-indexes changed files
- Query: vector similarity search returns relevant code chunks with file locations
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import shlex
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Max lines per chunk (overlapping by 20 lines for context continuity)
_CHUNK_LINES = 100
_CHUNK_OVERLAP = 20

# Extensions to index
_INDEXABLE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".rb",
    ".java", ".c", ".cpp", ".h", ".cs", ".php", ".swift", ".kt",
    ".sh", ".bash", ".yaml", ".yml", ".toml", ".json",
    ".html", ".css", ".scss", ".sql", ".md", ".txt",
    ".dockerfile", ".env", ".gitignore", ".cfg", ".ini",
}

# Directories to skip
_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".tox", ".mypy_cache", ".pytest_cache", "coverage",
    ".cargo", "pkg", "bin", "obj", ".augmentum",
}

# Max file size to index (skip huge generated files)
_MAX_FILE_SIZE = 100_000  # 100KB

# Read this many files per container exec (one base64 batch) instead of one
# `cat` exec per file. Bounds per-batch memory ≈ N × _MAX_FILE_SIZE × 1.33.
_READ_BATCH_FILES = 40

# Embed in small batches to keep ONNX activation memory bounded.
# One-shot embedding of thousands of chunks has caused OOM-kills (exit 137)
# when combined with concurrent narrative loads / chats sync. The streaming
# per-file pipeline below holds ≤ one file's chunks at a time, so this only
# caps one forward pass's activation memory.
_EMBED_BATCH_SIZE = 16

# Per-workspace lock so a double-POST to /api/coder/index doesn't stack
# two concurrent index builds (which both hold the full chunk list in RAM).
_index_locks: dict[str, asyncio.Lock] = {}

# Per-workspace index-build progress, polled by the file UI so the long
# first index (or a mass re-index) shows a determinate bar instead of
# silent idle time. Keyed by workspace_id; value carries state +
# done/total counts. Bounded by workspace count — overwritten each build.
_index_progress: dict[str, dict] = {}


def get_index_progress(workspace_id: str) -> dict | None:
    """Snapshot of the current/last index build for a workspace, or None.

    Returned by ``GET /api/coder/index/{id}/progress`` so the UI can render
    a live "Indexing N/total" strip and stop polling once ``state`` is
    ``done``.
    """
    p = _index_progress.get(workspace_id)
    return dict(p) if p else None


async def _batch_read_files(container_manager, workspace_id, abs_paths):
    """Read multiple workspace files in ONE container exec.

    The old indexer ran one ``docker exec cat`` per file — and each exec is
    ~3 Docker round-trips (inspect + exec-create + exec-start), so a 450-file
    workspace meant ~1,500 Docker calls hammering the daemon during open.

    This emits, per file, a sentinel header line then the file's base64.
    base64's alphabet ([A-Za-z0-9+/=] + newlines) can't contain the
    ``@@AUGFILE:`` sentinel, and base64 output is ASCII so it survives
    ``_run_command``'s utf-8 decode (a raw ``cat`` of a binary file would
    not). Returns ``{abs_path: text}``; files that error are simply absent.
    """
    if not abs_paths:
        return {}
    quoted = " ".join(shlex.quote(p) for p in abs_paths)
    script = (
        "for f in " + quoted + "; do "
        'printf "\\n@@AUGFILE:%s@@\\n" "$f"; '
        'base64 "$f" 2>/dev/null || true; '
        "done"
    )
    try:
        out = await container_manager._run_command(
            workspace_id, ["bash", "-c", script], timeout=60.0,
        )
    except Exception:
        return {}
    result: dict[str, str] = {}
    for part in out.split("\n@@AUGFILE:")[1:]:
        head, sep, body = part.partition("@@\n")
        if not sep:
            continue
        try:
            result[head] = base64.b64decode(body).decode("utf-8", errors="replace")
        except Exception:
            continue
    return result


@dataclass
class CodeChunk:
    """A chunk of code with its location metadata."""
    file_path: str       # Relative to /workspace
    start_line: int
    end_line: int
    content: str
    file_hash: str       # MD5 of full file content (for change detection)


@dataclass
class SearchResult:
    """A search result from the codebase index."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float


async def build_index(
    container_manager,
    workspace_id: str,
    force: bool = False,
) -> dict:
    """Build or update the codebase index for a workspace.

    Runs inside the Augmentum server, not inside the container.
    Reads files via container_manager, embeds via EmbeddingService,
    stores vectors in a local SQLite file.

    Serialized per workspace: if an index build is already running for
    this workspace, returns an already-running stub instead of kicking
    off a second concurrent build.

    Args:
        container_manager: ContainerManager instance
        workspace_id: Workspace to index
        force: If True, rebuild from scratch

    Returns:
        Stats dict: {indexed, skipped, total_chunks, duration_ms}
    """
    lock = _index_locks.setdefault(workspace_id, asyncio.Lock())
    if lock.locked():
        log.info("build_index_already_running", workspace=workspace_id)
        return {
            "indexed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "duration_ms": 0,
            "status": "already_running",
        }
    async with lock:
        return await _build_index_impl(container_manager, workspace_id, force=force)


async def _build_index_impl(
    container_manager,
    workspace_id: str,
    force: bool = False,
) -> dict:
    import sqlite3

    import sqlite_vec

    from augmentum.memory.embeddings import EmbeddingService

    t0 = time.monotonic()

    # Get the index DB path (stored server-side, keyed by workspace ID)
    from augmentum.config import settings
    index_dir = Path(settings.data_dir) / "coder_indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / f"{workspace_id[:12]}.db"

    # Init DB. All sqlite work is offloaded to a worker thread: synchronous
    # sqlite3 commits (fsync) on the event loop froze every other request
    # while a large-workspace index built (2026-06-13 loop-stall audit).
    # check_same_thread=False lets the one connection move between to_thread
    # worker threads; access stays serialized by the per-workspace lock, so
    # no two threads ever touch it at once.
    def _open_db():
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            content TEXT NOT NULL,
            file_hash TEXT NOT NULL
        )""")
        return conn

    db = await asyncio.to_thread(_open_db)

    # Get embedding dimension from the model. Offloaded: the first call
    # also triggers the model load, which would block the event loop for
    # seconds on a cold start during workspace index build (2026-06-13
    # loop-stall audit). The batch embeds below already use to_thread.
    dim = len(await asyncio.to_thread(EmbeddingService.embed_one, "test"))

    def _init_schema():
        # Create vec table if needed.
        try:
            db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding FLOAT32[{dim}]
            )""")
        except Exception as exc:
            # IF NOT EXISTS would be ideal but sqlite-vec rejects it. Errors
            # other than "table already exists" still need surfacing — a
            # silent-swallow here would mean the indexer runs on a missing
            # vec table and every insert fails.
            log.debug("coder_indexer_vec_table_create_skipped", error=str(exc))
        db.execute("""CREATE TABLE IF NOT EXISTS file_index (
            path TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            indexed_at REAL NOT NULL
        )""")
        # mtime + size let us skip UNCHANGED files without reading them at all
        # (the old path cat'd every file just to hash it — a full re-read of
        # the workspace on every open). Added via ALTER so index DBs created
        # before this change pick the columns up in place.
        cols = {r[1] for r in db.execute("PRAGMA table_info(file_index)").fetchall()}
        for col, decl in (("mtime", "TEXT"), ("size", "INTEGER")):
            if col not in cols:
                db.execute(f"ALTER TABLE file_index ADD COLUMN {col} {decl}")
        db.commit()
        # Existing per-file metadata for change detection + the chunk-id
        # counter seed, read together so the per-file loop never round-trips
        # sqlite on the event loop. Value: (hash, mtime_str, size_int).
        existing = {}
        if not force:
            for row in db.execute(
                "SELECT path, hash, mtime, size FROM file_index"
            ).fetchall():
                existing[row[0]] = (row[1], row[2], row[3])
        max_row = db.execute("SELECT MAX(id) FROM chunks").fetchone()
        return existing, (max_row[0] or 0) + 1

    existing_meta, next_id = await asyncio.to_thread(_init_schema)

    # List all files WITH their mtime + size in one find (no per-file stat).
    # The "<mtime>\t<size>\t<path>" rows let the loop below skip unchanged
    # files without a single `cat`.
    try:
        file_list_output = await container_manager._run_command(
            workspace_id,
            ["bash", "-c",
             "find /workspace -type f "
             + " ".join(f"-not -path '*/{d}/*'" for d in _SKIP_DIRS)
             + r" -printf '%T@\t%s\t%p\n' 2>/dev/null | head -500"],
            timeout=15.0,
        )
    except Exception:
        log.warning("index_file_list_failed", workspace=workspace_id)
        await asyncio.to_thread(db.close)
        return {"indexed": 0, "skipped": 0, "total_chunks": 0, "duration_ms": 0}

    # Parse "<mtime>\t<size>\t<path>" rows; keep only indexable extensions.
    # Each entry: (rel_path, abs_path, mtime_str, size_int).
    files_meta: list[tuple[str, str, str, int]] = []
    for line in file_list_output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        mtime_str, size_str, abs_path = parts
        if not any(abs_path.endswith(ext) for ext in _INDEXABLE_EXTS):
            continue
        try:
            size_int = int(size_str)
        except ValueError:
            size_int = -1
        files_meta.append(
            (abs_path.replace("/workspace/", ""), abs_path, mtime_str, size_int)
        )

    indexed = 0
    skipped = 0
    unchanged = 0

    def _touch_meta(rel_path, file_hash, mtime_str, size_int):
        # File was touched (mtime/size drifted) but content is identical —
        # refresh stored mtime/size so the NEXT open hits the no-read fast
        # path, without re-embedding now.
        try:
            db.execute(
                "UPDATE file_index SET mtime=?, size=?, indexed_at=? WHERE path=?",
                (mtime_str, size_int, time.time(), rel_path),
            )
            db.commit()
        except Exception:
            db.rollback()

    # Publish progress so the UI can render a determinate bar over the one
    # piece of idle time we can't trim (a genuine first index / mass change).
    progress = {
        "state": "running", "total": len(files_meta), "done": 0,
        "indexed": 0, "unchanged": 0, "skipped": 0,
    }
    _index_progress[workspace_id] = progress

    # Partition using ONLY the cheap find metadata — zero reads. Unchanged
    # files (mtime AND size match the last index) and over-size files are
    # decided here without a single `cat`.
    to_read: list[tuple[str, str, str, int]] = []
    for rel_path, file_path, mtime_str, size_int in files_meta:
        prev = existing_meta.get(rel_path)
        if prev is not None and prev[1] == mtime_str and prev[2] == size_int:
            unchanged += 1
            continue
        if size_int > _MAX_FILE_SIZE:  # size from find — skip big files unread
            skipped += 1
            continue
        to_read.append((rel_path, file_path, mtime_str, size_int))
    progress["done"] = unchanged + skipped

    async def _index_one_file(rel_path, file_path, mtime_str, size_int, content):
        """Chunk → embed → persist one already-read file. Mutates the running
        indexed/unchanged/skipped counters + the chunk-id seed; kept nested so
        it shares db/dim without threading them through every call."""
        nonlocal next_id, indexed, unchanged, skipped
        if len(content) > _MAX_FILE_SIZE:  # belt-and-suspenders for unknown sizes
            skipped += 1
            return
        file_hash = hashlib.md5(content.encode()).hexdigest()
        prev = existing_meta.get(rel_path)
        if prev is not None and prev[0] == file_hash:
            # Touched but content identical — refresh meta, don't re-embed.
            await asyncio.to_thread(
                _touch_meta, rel_path, file_hash, mtime_str, size_int,
            )
            unchanged += 1
            return

        # Build this file's chunks — held in memory only for this call.
        file_chunks: list[CodeChunk] = []
        lines = content.splitlines()
        for start in range(0, len(lines), _CHUNK_LINES - _CHUNK_OVERLAP):
            end = min(start + _CHUNK_LINES, len(lines))
            chunk_text = "\n".join(lines[start:end])
            if len(chunk_text.strip()) < 20:
                continue
            enriched = f"File: {rel_path} (lines {start + 1}-{end})\n{chunk_text}"
            file_chunks.append(CodeChunk(
                file_path=rel_path,
                start_line=start + 1,
                end_line=end,
                content=enriched,
                file_hash=file_hash,
            ))

        # Embed in bounded batches (off the loop) and accumulate this file's
        # rows. Peak memory ≈ one file's chunks + embeddings.
        rows: list[tuple] = []
        embed_failed = False
        for batch_start in range(0, len(file_chunks), _EMBED_BATCH_SIZE):
            batch = file_chunks[batch_start:batch_start + _EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            try:
                embeddings = await asyncio.to_thread(EmbeddingService.embed, texts)
            except Exception as exc:
                log.warning(
                    "index_file_embed_failed",
                    workspace=workspace_id, file=rel_path, error=str(exc),
                )
                embed_failed = True
                break
            for chunk, emb in zip(batch, embeddings, strict=True):
                chunk_id = next_id
                next_id += 1
                rows.append((
                    chunk_id, chunk.file_path, chunk.start_line,
                    chunk.end_line, chunk.content, chunk.file_hash,
                    struct.pack(f"<{dim}f", *emb),
                ))
            # Release the batch's text + embedding arrays before the next
            # pass so ONNX activation memory doesn't stack.
            del texts, embeddings, batch

        if embed_failed:
            skipped += 1
            return

        # Replace old rows + write new ones in one transaction (off the loop)
        # so a mid-file failure can't leave stale deletes but missing inserts.
        def _persist(rel_path=rel_path, file_hash=file_hash, rows=rows,
                     mtime_str=mtime_str, size_int=size_int):
            try:
                db.execute(
                    "DELETE FROM chunks_vec WHERE id IN "
                    "(SELECT id FROM chunks WHERE file_path = ?)",
                    (rel_path,),
                )
                db.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
                for r in rows:
                    db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", r[:6])
                    db.execute(
                        "INSERT OR REPLACE INTO chunks_vec VALUES (?, ?)",
                        (r[0], r[6]),
                    )
                db.execute(
                    "INSERT OR REPLACE INTO file_index "
                    "(path, hash, indexed_at, mtime, size) VALUES (?, ?, ?, ?, ?)",
                    (rel_path, file_hash, time.time(), mtime_str, size_int),
                )
                db.commit()
                return True
            except Exception as exc:
                db.rollback()
                log.warning(
                    "index_file_failed",
                    workspace=workspace_id, file=rel_path, error=str(exc),
                )
                return False

        if await asyncio.to_thread(_persist):
            indexed += 1
        else:
            skipped += 1

    # Read the changed/new files in BATCHES — one container exec per batch
    # (base64) instead of one `cat` per file. Each batch's contents are
    # processed and freed before the next read, so peak memory stays bounded.
    for bstart in range(0, len(to_read), _READ_BATCH_FILES):
        read_batch = to_read[bstart:bstart + _READ_BATCH_FILES]
        contents = await _batch_read_files(
            container_manager, workspace_id, [b[1] for b in read_batch],
        )
        for rel_path, file_path, mtime_str, size_int in read_batch:
            progress["done"] += 1
            progress["indexed"] = indexed
            progress["unchanged"] = unchanged
            progress["skipped"] = skipped
            content = contents.get(file_path)
            if content is None:
                skipped += 1
                continue
            await _index_one_file(rel_path, file_path, mtime_str, size_int, content)

    # All files processed — flip progress to done so the UI strip can show a
    # brief "ready" and stop polling. (chunk count isn't part of progress.)
    progress.update(
        state="done", done=len(files_meta),
        indexed=indexed, unchanged=unchanged, skipped=skipped,
    )

    # Reconcile deleted files — drop chunks_vec + chunks + file_index rows
    # together so no orphans linger behind a missing parent row. Offloaded
    # with the final count + close so the last commit doesn't block the loop.
    def _finalize():
        current_rel_paths = {m[0] for m in files_meta}
        for path in list(existing_meta.keys()):
            if path not in current_rel_paths:
                db.execute(
                    "DELETE FROM chunks_vec WHERE id IN "
                    "(SELECT id FROM chunks WHERE file_path = ?)",
                    (path,),
                )
                db.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
                db.execute("DELETE FROM file_index WHERE path = ?", (path,))
        db.commit()
        total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db.close()
        return total

    total_chunks = await asyncio.to_thread(_finalize)

    duration_ms = int((time.monotonic() - t0) * 1000)
    log.info("codebase_index_built",
             workspace=workspace_id,
             indexed=indexed, skipped=skipped, unchanged=unchanged,
             files=len(files_meta),
             chunks=total_chunks, duration_ms=duration_ms)

    return {
        "indexed": indexed,
        "skipped": skipped,
        "unchanged": unchanged,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }


async def search_index(
    workspace_id: str,
    query: str,
    limit: int = 10,
) -> list[SearchResult]:
    """Search the codebase index for code relevant to a query.

    Args:
        workspace_id: Workspace to search
        query: Natural language or code query
        limit: Max results to return

    Returns:
        List of SearchResult with file paths, line ranges, and content.
    """
    import sqlite3

    import sqlite_vec

    from augmentum.config import settings
    from augmentum.memory.embeddings import EmbeddingService

    index_dir = Path(settings.data_dir) / "coder_indexes"
    db_path = index_dir / f"{workspace_id[:12]}.db"

    if not db_path.exists():
        return []

    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    # Embed the query
    import asyncio
    query_vec = await asyncio.to_thread(EmbeddingService.embed_query, query)
    query_blob = EmbeddingService.to_blob(query_vec)

    try:
        cursor = db.execute(
            "SELECT v.id, v.distance FROM chunks_vec v "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance",
            (query_blob, limit),
        )
        vec_rows = cursor.fetchall()
    except Exception:
        db.close()
        return []

    if not vec_rows:
        db.close()
        return []

    # Fetch chunk content
    chunk_ids = [r[0] for r in vec_rows]
    dist_map = {r[0]: r[1] for r in vec_rows}

    placeholders = ",".join("?" * len(chunk_ids))
    cursor = db.execute(
        f"SELECT id, file_path, start_line, end_line, content "
        f"FROM chunks WHERE id IN ({placeholders})",
        chunk_ids,
    )
    rows = cursor.fetchall()
    db.close()

    results = []
    for row in rows:
        dist = dist_map.get(row[0], 100.0)
        # Convert L2 distance to 0-1 similarity score
        # Using exponential decay: score = exp(-dist/scale)
        import math
        score = math.exp(-dist / 20.0)
        results.append(SearchResult(
            file_path=row[1],
            start_line=row[2],
            end_line=row[3],
            content=row[4],
            score=round(score, 3),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
