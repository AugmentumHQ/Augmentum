"""Tests for the file-analyzer substrate.

Targets the registry contract (extension dispatch, magic-byte dispatch,
graceful failure) and one concrete builtin (SQLite — stdlib-only, so
the test runs without optional deps installed in the test env). The
other builtins are smoke-tested at import time by the substrate's
``__init__`` running their ``register_analyzer`` calls; a missing or
broken handler would raise at module import.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from augmentum.coder.analyzers import (
    AnalysisReport,
    analyze_file,
    is_analyzable,
    register_analyzer,
)
from augmentum.coder.analyzers.registry import _BY_EXT, _BY_MAGIC


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_registry_loaded_with_builtins():
    """Importing the analyzers package should auto-register builtins."""
    expected_extensions = {"gguf", "glb", "vrm", "gltf", "safetensors",
                            "mp3", "flac", "wav", "db", "sqlite",
                            "zip", "tar"}
    missing = expected_extensions - set(_BY_EXT.keys())
    assert not missing, f"Builtin handlers didn't register for: {missing}"


def test_is_analyzable_by_extension():
    assert is_analyzable("/workspace/model.gguf")
    assert is_analyzable("/workspace/avatar.vrm")
    assert is_analyzable("/workspace/song.MP3")  # case-insensitive
    assert not is_analyzable("/workspace/src/main.py")
    assert not is_analyzable("/workspace/README.md")


def test_is_analyzable_by_magic_bytes():
    # No extension on the path but bytes look like GGUF
    assert is_analyzable("/workspace/blob", raw=b"GGUF\x00\x00\x00\x03data")
    # No extension and unknown bytes
    assert not is_analyzable("/workspace/blob", raw=b"hello world")


@pytest.mark.asyncio
async def test_analyze_file_returns_none_for_unmatched():
    report = await analyze_file("/workspace/src/main.py", b"def foo(): pass")
    assert report is None


@pytest.mark.asyncio
async def test_analyzer_failure_logs_and_returns_none(monkeypatch):
    """When a registered analyzer raises, dispatch returns None rather
    than bubbling — caller (file_read) falls through to the raw path."""

    class _Broken:
        name = "broken"
        extensions = ("brokenfmt",)
        magic_bytes = ()
        async def analyze(self, path, raw):
            raise RuntimeError("boom")

    register_analyzer(_Broken())
    result = await analyze_file("/workspace/x.brokenfmt", b"data")
    assert result is None


# ---------------------------------------------------------------------------
# SQLite analyzer (stdlib-only — runnable everywhere)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_analyzer_reports_tables_and_counts(tmp_path: Path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, body TEXT);
        CREATE INDEX idx_posts_user ON posts(user_id);
        INSERT INTO users (name) VALUES ('alice'), ('bob'), ('carol');
        INSERT INTO posts (user_id, body) VALUES (1, 'hi'), (2, 'hey');
        """
    )
    conn.commit()
    conn.close()

    raw = db_path.read_bytes()
    report = await analyze_file(str(db_path), raw)

    assert report is not None
    assert report.format == "SQLite database"
    assert "users" in report.summary
    assert "posts" in report.summary
    # Row counts surface
    assert "3" in report.summary  # 3 users
    assert "2" in report.summary  # 2 posts (or appears among other 2s)
    # Details should be machine-readable
    assert report.details["table_count"] == 2
    assert report.details["index_count"] == 1
    assert "users" in report.details["tables"]
    assert report.details["tables"]["users"]["row_count"] == 3


@pytest.mark.asyncio
async def test_sqlite_analyzer_recognises_magic_bytes(tmp_path: Path):
    """Even without a .db extension, the magic prefix should match."""
    db_path = tmp_path / "no_extension"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE x (a INT)")
    conn.commit()
    conn.close()

    raw = db_path.read_bytes()
    assert raw.startswith(b"SQLite format 3\x00")
    report = await analyze_file(str(db_path), raw)
    assert report is not None
    assert report.format == "SQLite database"
