"""Knowledge pack importer — convert various file formats into .augpack.

Accepts: .augpack (direct copy), .csv, .jsonl, .json, .sqlite/.db,
.md, .txt, .pdf, .docx, .html, .epub, .zim, and folder archives (.zip).

Uses the existing document chunker for text extraction (PDF, DOCX, HTML)
and EmbeddingService for vectorization. Output is a standard .augpack
SQLite file with sqlite-vec index.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import struct
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augmentum.utils.logging import get_logger
from augmentum.utils.safe_archive import ensure_zip_sane
from augmentum.utils.sql import quote_ident

log = get_logger(__name__)

# Formats we handle natively (no conversion needed)
NATIVE_EXT = {".augpack"}

# Formats we can import (text extraction + embedding)
IMPORTABLE_EXTS = {
    ".csv", ".tsv",
    ".json", ".jsonl", ".ndjson",
    ".sqlite", ".db", ".sqlite3",
    ".md", ".txt", ".rst",
    ".pdf",
    ".docx", ".doc",
    ".html", ".htm",
    ".epub",
    ".zip",
}

ALL_SUPPORTED = NATIVE_EXT | IMPORTABLE_EXTS


@dataclass
class ImportChunk:
    """A chunk of text ready for embedding."""
    content: str
    title: str
    section: str = ""
    source: str = ""
    url: str = ""
    chunk_index: int = 0


def detect_format(filename: str) -> str | None:
    """Detect import format from filename extension."""
    ext = Path(filename).suffix.lower()
    if ext in ALL_SUPPORTED:
        return ext
    return None


async def import_to_augpack(
    file_data: bytes,
    filename: str,
    output_path: Path,
    pack_name: str = "",
    description: str = "",
    source: str = "",
    progress_cb: Callable[[str, int, int], Any] | None = None,
) -> dict:
    """Import a file into .augpack format.

    Args:
        file_data: Raw file bytes
        filename: Original filename (for format detection)
        output_path: Where to write the .augpack file
        pack_name: Display name for the pack
        description: Pack description
        source: Source label (e.g., "wikipedia", "local")
        progress_cb: Optional callback(stage, current, total)

    Returns:
        Dict with import stats (chunk_count, file_size, etc.)
    """
    ext = Path(filename).suffix.lower()

    # Native .augpack — just copy directly
    if ext == ".augpack":
        output_path.write_bytes(file_data)
        return {"format": "augpack", "chunk_count": 0, "file_size": len(file_data), "copied": True}

    # Extract chunks from the source format
    if progress_cb:
        progress_cb("extracting", 0, 0)

    chunks = _extract_chunks(file_data, filename, ext, source or "imported")

    if not chunks:
        raise ValueError(f"No content extracted from {filename}")

    log.info("import_extracted", filename=filename, format=ext, chunks=len(chunks))

    # Embed all chunks
    if progress_cb:
        progress_cb("embedding", 0, len(chunks))

    import asyncio

    from augmentum.memory.embeddings import EmbeddingService

    texts = [c.content for c in chunks]
    # Batch embed in thread to avoid blocking the event loop
    embeddings = await asyncio.to_thread(EmbeddingService.embed, texts)
    dim = len(embeddings[0])

    if progress_cb:
        progress_cb("embedding", len(chunks), len(chunks))

    # Build the .augpack SQLite file
    if progress_cb:
        progress_cb("packaging", 0, len(chunks))

    _build_augpack(
        output_path=output_path,
        chunks=chunks,
        embeddings=embeddings,
        dim=dim,
        pack_name=pack_name or Path(filename).stem.replace("_", " ").replace("-", " ").title(),
        description=description,
        source=source or "imported",
    )

    stats = {
        "format": ext,
        "chunk_count": len(chunks),
        "embedding_dim": dim,
        "file_size": output_path.stat().st_size,
    }
    log.info("import_complete", filename=filename, **stats)

    if progress_cb:
        progress_cb("done", len(chunks), len(chunks))

    return stats


# ---------------------------------------------------------------------------
# Format-specific extractors
# ---------------------------------------------------------------------------

def _extract_chunks(data: bytes, filename: str, ext: str, source: str) -> list[ImportChunk]:
    """Route to the appropriate extractor based on format."""
    if ext in (".csv", ".tsv"):
        return _extract_csv(data, filename, source, delimiter="\t" if ext == ".tsv" else ",")
    if ext in (".json", ".jsonl", ".ndjson"):
        return _extract_json(data, filename, source)
    if ext in (".sqlite", ".db", ".sqlite3"):
        return _extract_sqlite(data, filename, source)
    if ext == ".zip":
        return _extract_zip(data, filename, source)
    if ext == ".epub":
        return _extract_epub(data, filename, source)
    # Everything else: use the document chunker (PDF, DOCX, HTML, MD, TXT)
    return _extract_document(data, filename, source)


def _extract_csv(data: bytes, filename: str, source: str, delimiter: str = ",") -> list[ImportChunk]:
    """Extract from CSV/TSV. Combines all text columns per row into a chunk."""
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    # Detect which columns have text content
    chunks = []
    title_col = None
    content_cols = []

    for i, row in enumerate(reader):
        if i == 0:
            # Heuristic: find title and content columns
            for col in row:
                col_lower = col.lower()
                if col_lower in ("title", "name", "heading", "subject", "question"):
                    title_col = col
                elif col_lower in ("content", "text", "body", "description", "answer", "abstract", "summary"):
                    content_cols.append(col)
            # If no explicit content columns, use all non-title text columns
            if not content_cols:
                content_cols = [c for c in row if c != title_col and len(str(row.get(c, ""))) > 20]

        title = str(row.get(title_col, "")) if title_col else f"Row {i + 1}"
        content_parts = [str(row.get(c, "")) for c in content_cols if row.get(c)]
        content = "\n\n".join(content_parts)

        if content.strip() and len(content.strip()) > 10:
            chunks.append(ImportChunk(
                content=content.strip(),
                title=title.strip() or f"Row {i + 1}",
                source=source,
                chunk_index=0,
            ))

    return chunks


def _extract_json(data: bytes, filename: str, source: str) -> list[ImportChunk]:
    """Extract from JSON (array of objects) or JSONL (one object per line)."""
    text = data.decode("utf-8", errors="replace").strip()
    records = []

    if text.startswith("["):
        # JSON array
        records = json.loads(text)
    else:
        # JSONL / NDJSON
        for line in text.split("\n"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    chunks = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        # Find title and content fields
        title = (
            rec.get("title") or rec.get("name") or rec.get("heading")
            or rec.get("subject") or rec.get("question") or f"Item {i + 1}"
        )
        content = (
            rec.get("content") or rec.get("text") or rec.get("body")
            or rec.get("description") or rec.get("answer") or rec.get("abstract")
        )

        if not content:
            # Fallback: concatenate all string values
            parts = [f"{k}: {v}" for k, v in rec.items() if isinstance(v, str) and len(v) > 10]
            content = "\n".join(parts)

        if content and len(str(content).strip()) > 10:
            chunks.append(ImportChunk(
                content=str(content).strip(),
                title=str(title).strip(),
                section=str(rec.get("section", "")),
                url=str(rec.get("url", "")),
                source=source,
                chunk_index=0,
            ))

    return chunks


def _extract_sqlite(data: bytes, filename: str, source: str) -> list[ImportChunk]:
    """Extract text from a SQLite database. Finds tables with text columns."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        db = sqlite3.connect(tmp_path)
        # Find tables with text columns
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()

        chunks = []
        for (table_name,) in tables:
            # Get column info. quote_ident: table_name comes from the
            # UPLOADED file's sqlite_master, so it's attacker-controlled —
            # quote it (audit 2026-06-17). The connection is to the
            # throwaway uploaded DB (not the main DB), so the blast radius
            # was small, but bare f-string identifiers are exactly the
            # fragility worth removing at an untrusted-input surface.
            cols = db.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
            text_cols = [c[1] for c in cols if c[2].upper() in ("TEXT", "VARCHAR", "CLOB", "")]

            if not text_cols:
                continue

            # Heuristic: find title and content columns
            title_col = None
            content_cols = []
            for col in text_cols:
                col_lower = col.lower()
                if col_lower in ("title", "name", "heading", "subject"):
                    title_col = col
                elif col_lower in ("content", "text", "body", "description", "abstract"):
                    content_cols.append(col)

            if not content_cols:
                content_cols = [c for c in text_cols if c != title_col]

            if not content_cols:
                continue

            select_cols = ([title_col] if title_col else []) + content_cols
            # Both the column list and the table name are attacker-derived
            # (uploaded sqlite_master/PRAGMA) — quote every identifier.
            select_sql = ", ".join(quote_ident(c) for c in select_cols)
            rows = db.execute(
                f"SELECT {select_sql} FROM {quote_ident(table_name)} LIMIT 100000"
            ).fetchall()

            for i, row in enumerate(rows):
                offset = 1 if title_col else 0
                title = str(row[0]) if title_col else f"{table_name} #{i + 1}"
                content = "\n\n".join(str(v) for v in row[offset:] if v)

                if content.strip() and len(content.strip()) > 10:
                    chunks.append(ImportChunk(
                        content=content.strip(),
                        title=title.strip(),
                        source=source,
                        chunk_index=0,
                    ))

        db.close()
        return chunks
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_zip(data: bytes, filename: str, source: str) -> list[ImportChunk]:
    """Extract from a ZIP archive — process each file inside."""
    chunks = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        ensure_zip_sane(zf, source=f"knowledge_import:{filename}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in IMPORTABLE_EXTS and ext not in (".md", ".txt", ".html", ".pdf", ".csv", ".json", ".jsonl"):
                continue
            try:
                file_data = zf.read(info.filename)
                file_chunks = _extract_chunks(file_data, info.filename, ext, source)
                chunks.extend(file_chunks)
            except Exception:
                log.debug("zip_entry_extract_failed", entry=info.filename, exc_info=True)
    return chunks


def _extract_epub(data: bytes, filename: str, source: str) -> list[ImportChunk]:
    """Extract text from EPUB. EPUB is a ZIP of HTML files."""
    chunks = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ensure_zip_sane(zf, source=f"knowledge_epub:{filename}")
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                if info.filename.endswith((".xhtml", ".html", ".htm")):
                    try:
                        html_data = zf.read(info.filename)
                        html_chunks = _extract_document(html_data, info.filename, source)
                        chunks.extend(html_chunks)
                    except Exception as exc:
                        log.debug("epub_chunk_failed", member=info.filename, error=str(exc))
                        continue
    except Exception:
        log.warning("epub_extract_failed", filename=filename, exc_info=True)
    return chunks


def _extract_document(data: bytes, filename: str, source: str) -> list[ImportChunk]:
    """Extract using the existing document chunker (PDF, DOCX, HTML, MD, TXT)."""
    import mimetypes

    from augmentum.documents.chunker import chunk_sections, extract_text, section_split

    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        mime = "text/plain"

    pages = extract_text(data, mime, filename)
    if not pages:
        return []

    full_text = "\n\n".join(text for text, _ in pages if text)
    if not full_text.strip():
        return []

    # Use the existing section-aware chunker
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    sections = section_split(full_text)
    doc_chunks = chunk_sections(sections, filename=filename)

    return [
        ImportChunk(
            content=c.enriched_text or c.text,
            title=title,
            section=c.section,
            source=source,
            chunk_index=c.index,
        )
        for c in doc_chunks
        if (c.enriched_text or c.text).strip()
    ]


# ---------------------------------------------------------------------------
# Pack builder
# ---------------------------------------------------------------------------

def _build_augpack(
    output_path: Path,
    chunks: list[ImportChunk],
    embeddings: list[list[float]],
    dim: int,
    pack_name: str,
    description: str,
    source: str,
) -> None:
    """Write chunks + embeddings into an .augpack SQLite file."""
    if output_path.exists():
        output_path.unlink()

    db = sqlite3.connect(str(output_path))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)

    db.execute("""CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        content TEXT NOT NULL,
        title TEXT NOT NULL,
        section TEXT,
        source TEXT NOT NULL,
        url TEXT,
        chunk_index INTEGER DEFAULT 0
    )""")
    db.execute(f"""CREATE VIRTUAL TABLE chunks_vec USING vec0(
        id INTEGER PRIMARY KEY,
        embedding FLOAT32[{dim}]
    )""")
    # FTS5 mirror for hybrid retrieval — see converter.py for the rationale
    # and augmentum/knowledge/packs.py for the RRF merge consumer.
    db.execute("""CREATE VIRTUAL TABLE chunks_fts USING fts5(
        content, content=chunks, content_rowid=id
    )""")
    db.execute("""CREATE TRIGGER trg_pack_chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
    END""")
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        db.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
            (i, chunk.content, chunk.title, chunk.section,
             chunk.source, chunk.url, chunk.chunk_index),
        )
        blob = struct.pack(f"<{dim}f", *emb)
        db.execute("INSERT INTO chunks_vec VALUES (?,?)", (i, blob))

    from datetime import date
    meta = {
        "name": pack_name,
        "version": date.today().isoformat(),
        "description": description,
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
        "embedding_dim": str(dim),
        "chunk_count": str(len(chunks)),
        "source_license": "imported",
        "build_date": date.today().isoformat(),
    }
    for k, v in meta.items():
        db.execute("INSERT INTO meta VALUES (?,?)", (k, v))

    db.execute("CREATE INDEX idx_chunks_title ON chunks(title)")
    db.commit()
    db.execute("VACUUM")
    db.close()
