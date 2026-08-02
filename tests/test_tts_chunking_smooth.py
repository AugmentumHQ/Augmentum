"""Smooth TTS chunking — no comma seams on fast local providers.

2026-06-11: "sentence" mode's clause tier split short utterances at
commas ("Sure," [pause] "let me check") and the 30-char word-boundary
fallback chopped short sentences mid-phrase. For Pocket TTS — local,
near-instant, natively handles ~8KB — every split is a prosody reset
with no TTFA win. "smooth" emits whole sentences from chunk 0; the
default "sentence" setting auto-upgrades to it per provider.
"""

from __future__ import annotations

from augmentum.voice.pipeline import (
    SMOOTH_CHUNKING_PROVIDERS,
    SentenceBuffer,
    effective_chunking_mode,
)


def _feed(buf: SentenceBuffer, text: str) -> list[str]:
    out = []
    for tok in text.split(" "):
        chunk = buf.add_token(tok + " ")
        if chunk:
            out.append(chunk)
    tail = buf.flush()
    if tail:
        out.append(tail)
    return out


class TestSmoothMode:
    def test_short_utterance_is_one_chunk(self):
        # The exact complaint: "sentence" mode split this at the comma.
        buf = SentenceBuffer(min_chars=10, mode="smooth")
        chunks = _feed(buf, "Sure thing, let me check that for you now.")
        assert chunks == ["Sure thing, let me check that for you now."]

    def test_sentence_mode_still_clause_splits(self):
        # Pin the old behavior so the contrast is explicit — slow
        # network providers still want the fast first chunk.
        buf = SentenceBuffer(min_chars=10, mode="sentence")
        chunks = _feed(buf, "Sure thing, let me check that for you now.")
        assert len(chunks) > 1

    def test_multi_sentence_emits_per_sentence(self):
        buf = SentenceBuffer(min_chars=10, mode="smooth")
        chunks = _feed(buf, "First sentence here. Second one follows. Third!")
        assert chunks == [
            "First sentence here.", "Second one follows.", "Third!",
        ]

    def test_runaway_unpunctuated_text_still_caps(self):
        buf = SentenceBuffer(min_chars=10, mode="smooth")
        chunks = _feed(buf, "word " * 120)  # 600 chars, no punctuation
        assert len(chunks) >= 2  # the relaxed schedule still fires


class TestEffectiveChunkingMode:
    def test_default_upgrades_for_pocket(self):
        assert "pockettts-builtin" in SMOOTH_CHUNKING_PROVIDERS
        assert effective_chunking_mode("sentence", "pockettts-builtin") == "smooth"
        assert effective_chunking_mode("", "pockettts-builtin") == "smooth"

    def test_default_stays_for_network_providers(self):
        assert effective_chunking_mode("sentence", "kokoro-builtin") == "sentence"
        assert effective_chunking_mode("sentence", "") == "sentence"

    def test_explicit_setting_always_honored(self):
        assert effective_chunking_mode("clause", "pockettts-builtin") == "clause"
        assert effective_chunking_mode("full", "pockettts-builtin") == "full"
        assert effective_chunking_mode("paragraph", "pockettts-builtin") == "paragraph"


class TestSetModeGuard:
    def test_set_mode_before_streaming_applies(self):
        buf = SentenceBuffer(min_chars=10, mode="sentence")
        buf.set_mode("smooth")
        chunks = _feed(buf, "Sure thing, let me check that for you now.")
        assert chunks == ["Sure thing, let me check that for you now."]

    def test_set_mode_after_tokens_is_noop(self):
        buf = SentenceBuffer(min_chars=10, mode="sentence")
        buf.add_token("Hello there ")
        buf.set_mode("smooth")
        assert buf.mode == "sentence"

    def test_set_mode_unknown_is_noop(self):
        buf = SentenceBuffer(min_chars=10, mode="sentence")
        buf.set_mode("warp-speed")
        assert buf.mode == "sentence"
