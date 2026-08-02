"""Tests for the UI-shell content digest that backs /api/ui-version.

The digest is the native-app bundle handshake: the Android shell serves its
baked copy of the SPA from the APK only when this digest matches, so the
digest's stability and its include/exclude rules are correctness-critical.
The include set + excludes here MUST mirror the Android Gradle ``syncWebAssets``
task; if they drift the digests stop matching and the app network-loads
(fail-safe), but these tests pin the server side.
"""

from __future__ import annotations

import time
from pathlib import Path

from augmentum.proxy.server import (
    _UI_SHELL_EXCLUDE_TOP,
    _ui_shell_digest,
    _ui_shell_files,
)


def _make_tree(root: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "styles").mkdir()
    (root / "fonts").mkdir()
    (root / "mockups" / "sub").mkdir(parents=True)
    (root / "lib" / "three").mkdir(parents=True)
    # Shell files (should be included)
    (root / "index.html").write_text("<html>hi</html>", encoding="utf-8")
    (root / "scripts" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (root / "styles" / "main.css").write_text("body{}", encoding="utf-8")
    (root / "fonts" / "reader.woff2").write_bytes(b"\x00\x01woff2")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    # Excluded: dirs
    (root / "mockups" / "sub" / "demo.js").write_text("// mock", encoding="utf-8")
    (root / "lib" / "three" / "three.module.min.js").write_text("// THREE", encoding="utf-8")
    # Excluded: extensions
    (root / "scripts" / "app.js.map").write_text("{}", encoding="utf-8")
    (root / "big.wasm").write_bytes(b"\x00asm")
    (root / "core.data").write_bytes(b"binary")


def test_shell_files_includes_only_shell(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    rels = {rel for rel, _ in _ui_shell_files(tmp_path)}
    assert rels == {
        "index.html",
        "scripts/app.js",
        "styles/main.css",
        "fonts/reader.woff2",
        "manifest.json",
    }
    # No excluded dir leaked
    assert not any(r.startswith(("mockups/", "lib/")) for r in rels)
    # No excluded extension leaked
    assert not any(r.endswith((".map", ".wasm", ".data")) for r in rels)


def test_excluded_dirs_are_the_documented_set() -> None:
    # Guards against silent drift from the Gradle bundle's excludes.
    assert _UI_SHELL_EXCLUDE_TOP == frozenset({"mockups", "lib"})


def test_digest_is_stable_hex(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    d1 = _ui_shell_digest(tmp_path)
    assert len(d1) == 64 and all(c in "0123456789abcdef" for c in d1)
    # Deterministic on a re-call (also exercises the TTL cache hit).
    assert _ui_shell_digest(tmp_path) == d1


def test_digest_changes_when_shell_content_changes(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    d1 = _ui_shell_digest(tmp_path)
    # Mutate a shell file; bust the TTL cache by waiting out the signature path.
    time.sleep(0.01)
    (tmp_path / "scripts" / "app.js").write_text("console.log(2)", encoding="utf-8")
    # Force a recompute by clearing the in-process cache (TTL would otherwise
    # mask the change for up to 60s — the cache is keyed by absolute path).
    from augmentum.proxy.server import _ui_shell_digest_cache

    _ui_shell_digest_cache.pop(str(tmp_path), None)
    d2 = _ui_shell_digest(tmp_path)
    assert d2 != d1


def test_digest_ignores_excluded_changes(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    from augmentum.proxy.server import _ui_shell_digest_cache

    d1 = _ui_shell_digest(tmp_path)
    # Change only excluded content (a mockup + a wasm); digest must not move.
    (tmp_path / "mockups" / "sub" / "demo.js").write_text("// changed", encoding="utf-8")
    (tmp_path / "big.wasm").write_bytes(b"\x00asm-different-bytes")
    _ui_shell_digest_cache.pop(str(tmp_path), None)
    d2 = _ui_shell_digest(tmp_path)
    assert d2 == d1


def test_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert _ui_shell_digest(tmp_path / "does-not-exist") == ""
