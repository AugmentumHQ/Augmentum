"""Tests — augmentum.utils.atomic_io tmp + fsync + replace persistence.

Covers:
  * Basic write produces the expected JSON on disk
  * Existing file is replaced atomically (old content gone)
  * Parent directories created if missing
  * Unicode round-trips with ensure_ascii=False
  * Indent kwarg respected
  * On write failure the temp file is cleaned up (no .tmp.<pid> turds)
  * Text variant works for non-JSON payloads
  * Concurrent-call simulation: two pids → two tmp names → no collision
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from augmentum.utils.atomic_io import atomic_write_json, atomic_write_text


class TestAtomicWriteJson:
    def test_basic_write(self, tmp_path):
        target = tmp_path / "out.json"
        atomic_write_json(target, {"hello": "world"})
        assert json.loads(target.read_text()) == {"hello": "world"}

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "out.json"
        target.write_text(json.dumps({"old": "data"}))
        atomic_write_json(target, {"new": "data"})
        assert json.loads(target.read_text()) == {"new": "data"}

    def test_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "path" / "file.json"
        atomic_write_json(nested, [1, 2, 3])
        assert json.loads(nested.read_text()) == [1, 2, 3]

    def test_unicode_round_trip(self, tmp_path):
        target = tmp_path / "unicode.json"
        payload = {"name": "Becca 🌙", "japanese": "こんにちは"}
        atomic_write_json(target, payload)
        assert json.loads(target.read_text(encoding="utf-8")) == payload

    def test_indent_respected(self, tmp_path):
        target = tmp_path / "pretty.json"
        atomic_write_json(target, {"k": 1}, indent=2)
        text = target.read_text()
        assert "\n  " in text  # 2-space indent
        atomic_write_json(target, {"k": 1})  # compact
        text = target.read_text()
        assert "\n" not in text

    def test_temp_file_cleaned_up_on_failure(self, tmp_path):
        """If the write raises mid-flight, the .tmp.<pid> sibling
        must not be left on disk."""
        target = tmp_path / "fail.json"

        # Inject a TypeError by passing a non-serialisable value;
        # json.dump raises inside the with-block, before os.replace.
        class NotSerialisable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(target, {"obj": NotSerialisable()})

        # No tmp turds.
        leftover = list(tmp_path.glob("*.tmp.*"))
        assert leftover == [], f"leftover tmp files: {leftover}"
        # Target also wasn't created (atomic — partial state never visible).
        assert not target.exists()

    def test_existing_file_preserved_on_failure(self, tmp_path):
        """If the write fails, the prior content of the target must
        remain on disk (atomicity guarantee)."""
        target = tmp_path / "existing.json"
        target.write_text(json.dumps({"prior": "content"}))

        class NotSerialisable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(target, {"obj": NotSerialisable()})

        # Prior content intact.
        assert json.loads(target.read_text()) == {"prior": "content"}

    def test_concurrent_pids_use_distinct_tmp(self, tmp_path):
        """Simulate two pids writing concurrently: each must use a
        distinct .tmp.<pid> name so they don't collide on the rename
        target."""
        target = tmp_path / "concurrent.json"

        # Capture the temp filenames each call generates by intercepting
        # os.getpid() to return different values per call.
        pids = iter([11111, 22222])
        # Snapshot of tmp files seen during writes.
        snapshots: list[set[str]] = []

        original_replace = os.replace

        def _capturing_replace(src, dst):
            # Capture tmp names that exist at the moment of replace.
            snapshots.append({p.name for p in tmp_path.glob("*.tmp.*")})
            return original_replace(src, dst)

        with patch("augmentum.utils.atomic_io.os.getpid", side_effect=lambda: next(pids)):
            with patch("augmentum.utils.atomic_io.os.replace", side_effect=_capturing_replace):
                atomic_write_json(target, {"first": True})
                atomic_write_json(target, {"second": True})

        # The two writes saw distinct tmp filenames.
        all_tmps = set()
        for snap in snapshots:
            all_tmps |= snap
        assert any(".tmp.11111" in name for name in all_tmps)
        assert any(".tmp.22222" in name for name in all_tmps)


class TestAtomicWriteText:
    def test_basic_write(self, tmp_path):
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello world\n")
        assert target.read_text() == "hello world\n"

    def test_unicode(self, tmp_path):
        target = tmp_path / "unicode.txt"
        atomic_write_text(target, "🌙 こんにちは")
        assert target.read_text(encoding="utf-8") == "🌙 こんにちは"
