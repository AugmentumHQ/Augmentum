"""Static regression guard for the chat renderer → avatar streaming wire.

The chat renderer's ``appendToStreaming`` is the per-chunk hook during
text streaming. Without forwarding deltas to the avatar module's
``onLLMDelta``, the companion's presence engine sees no signal during
text chat → the VRMA picker never re-picks, breathing stays at the
idle baseline, gesture cadence doesn't fire. Looks "laggy and low
quality" even when the library content is fine.

This test doesn't run the JS — it pins that the source still calls
``_avatarOnDelta(text)`` inside ``appendToStreaming`` so a future
refactor doesn't quietly remove the bridge.
"""
from __future__ import annotations

import pathlib


_RENDERER_PATH = pathlib.Path("ui/scripts/chat/renderer.js")


def _renderer_src() -> str:
    return _RENDERER_PATH.read_text(encoding="utf-8")


# ── Helper presence ─────────────────────────────────────────────


class TestAvatarHelperPresent:
    def test_avatar_module_cache_declared(self):
        """Module-scoped cache + loader exists. Without the cache, every
        chunk allocates a promise for the dynamic import which would
        make streaming costly."""
        src = _renderer_src()
        assert "let _avatarMod = null" in src
        assert "let _avatarModLoading = false" in src

    def test_helper_signature(self):
        """`_avatarOnDelta` function declared with the documented shape."""
        src = _renderer_src()
        assert "function _avatarOnDelta(text)" in src

    def test_helper_imports_avatar_module_dynamically(self):
        """Dynamic import keeps Three.js out of the chat bundle when no
        avatar surface is mounted."""
        src = _renderer_src()
        assert "import('../avatar.js')" in src

    def test_helper_calls_onllmdelta(self):
        src = _renderer_src()
        # The helper passes the text through to avatar.onLLMDelta.
        assert ".onLLMDelta(text)" in src


# ── Wire-up at the streaming site ───────────────────────────────


class TestWireUpAtAppendSite:
    def test_append_to_streaming_calls_avatar_helper(self):
        """The per-chunk site invokes the avatar bridge."""
        src = _renderer_src()
        # The call must appear inside (or near) appendToStreaming. We
        # use a substring window check: find appendToStreaming, then
        # look for _avatarOnDelta(text) within ~30 lines.
        idx = src.find("appendToStreaming(text)")
        assert idx > 0, "appendToStreaming(text) not found in renderer"
        window = src[idx:idx + 2000]
        assert "_avatarOnDelta(text)" in window, (
            "_avatarOnDelta(text) must be called inside appendToStreaming "
            "so text-mode chat keeps the companion presence engine in sync"
        )

    def test_avatar_call_after_raw_content_accumulation(self):
        """Avatar forward should be after the rawContent accumulation
        so the dataset reflects the full content even if the avatar
        path throws. Order matters for abort/finalize observability."""
        src = _renderer_src()
        idx = src.find("appendToStreaming(text)")
        window = src[idx:idx + 2000]
        raw_idx = window.find("el.dataset.rawContent = current + text")
        avatar_idx = window.find("_avatarOnDelta(text)")
        assert 0 < raw_idx < avatar_idx, (
            "Avatar forward must come after rawContent accumulation so "
            "the dataset is always coherent even if the avatar throws"
        )


# ── Defensive shape: errors don't break chat ────────────────────


class TestDefensiveShape:
    def test_helper_wraps_call_in_try_catch(self):
        """A presence engine bug must not break chat streaming."""
        src = _renderer_src()
        # Locate the _avatarOnDelta function body and check for try/catch.
        idx = src.find("function _avatarOnDelta(text)")
        assert idx > 0
        body = src[idx:idx + 2000]
        assert "try {" in body
        assert "catch" in body

    def test_failure_path_uses_sentinel(self):
        """A failed load sets _avatarMod = false so we don't retry on
        every subsequent chunk — bounded work even when avatar is
        unavailable."""
        src = _renderer_src()
        assert "_avatarMod = false" in src

    def test_error_logging_is_one_shot(self):
        """If the presence engine throws repeatedly, we log once and
        suppress — no console.warn per chunk during a stream."""
        src = _renderer_src()
        assert "_chatOnLLMDeltaErrorLogged" in src or "one-shot" in src.lower()
