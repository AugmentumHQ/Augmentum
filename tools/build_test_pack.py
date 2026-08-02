#!/usr/bin/env python3
"""Build a .augpack knowledge pack from a directory of text files.

Usage:
    python tools/build_test_pack.py --input docs/ --output test.augpack --name "Test Docs"
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def chunk_text(text: str, title: str, source: str = "local",
               max_chars: int = 2000, overlap: int = 200) -> list[dict]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append({
                "content": current.strip(),
                "title": title,
                "section": "",
                "source": source,
                "url": "",
                "chunk_index": idx,
            })
            current = current[-overlap:] + "\n\n" + para if overlap else para
            idx += 1
        else:
            current += ("\n\n" if current else "") + para

    if current.strip():
        chunks.append({
            "content": current.strip(),
            "title": title,
            "section": "",
            "source": source,
            "url": "",
            "chunk_index": idx,
        })

    return chunks


def build_pack(input_dir: Path, output_path: Path, name: str,
               description: str = "", source: str = "local") -> None:
    from augmentum.memory.embeddings import EmbeddingService

    files = sorted(input_dir.glob("**/*.txt")) + sorted(input_dir.glob("**/*.md"))
    if not files:
        print(f"No .txt or .md files found in {input_dir}")
        return

    all_chunks: list[dict] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        title = f.stem.replace("_", " ").replace("-", " ").title()
        file_chunks = chunk_text(text, title=title, source=source)
        all_chunks.extend(file_chunks)
        print(f"  {f.name}: {len(file_chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(files)} files")

    print("Embedding chunks...")
    texts = [c["content"] for c in all_chunks]
    embeddings = EmbeddingService.embed(texts)
    dim = len(embeddings[0])
    print(f"  Dimension: {dim}")

    if output_path.exists():
        output_path.unlink()

    db = sqlite3.connect(str(output_path))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)

    db.execute("""CREATE TABLE chunks (
        id INTEGER PRIMARY KEY, content TEXT NOT NULL, title TEXT NOT NULL,
        section TEXT, source TEXT NOT NULL, url TEXT, chunk_index INTEGER DEFAULT 0
    )""")
    db.execute(f"""CREATE VIRTUAL TABLE chunks_vec USING vec0(
        id INTEGER PRIMARY KEY, embedding FLOAT32[{dim}]
    )""")
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
        db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                   (i, chunk["content"], chunk["title"], chunk["section"],
                    chunk["source"], chunk["url"], chunk["chunk_index"]))
        blob = struct.pack(f"<{dim}f", *emb)
        db.execute("INSERT INTO chunks_vec VALUES (?,?)", (i, blob))

    from datetime import date
    meta = {
        "name": name, "version": date.today().isoformat(),
        "description": description,
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
        "embedding_dim": str(dim), "chunk_count": str(len(all_chunks)),
        "source_license": "local", "build_date": date.today().isoformat(),
    }
    for k, v in meta.items():
        db.execute("INSERT INTO meta VALUES (?,?)", (k, v))

    db.execute("CREATE INDEX idx_chunks_title ON chunks(title)")
    db.commit()
    db.execute("VACUUM")
    db.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nPack built: {output_path} ({size_mb:.1f} MB, {len(all_chunks)} chunks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a .augpack from text files")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output .augpack path")
    parser.add_argument("--name", "-n", required=True, help="Pack name")
    parser.add_argument("--description", "-d", default="", help="Pack description")
    parser.add_argument("--source", "-s", default="local", help="Source label")
    args = parser.parse_args()

    build_pack(Path(args.input), Path(args.output), args.name, args.description, args.source)
