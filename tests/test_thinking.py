"""Tests for thinking block awareness — normalization, streaming, backends, and integration."""

from __future__ import annotations  # noqa: I001

import json

import httpx
import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    Usage,
)
from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.models.ollama import OllamaBackend
from augmentum.models.openai_compat import (
    OpenAIBackend,
    to_openai_chat_response,
)
from augmentum.proxy.streaming import (
    _chunk_to_ollama_chat_ndjson,
    _chunk_to_openai_sse,
)
from augmentum.utils.thinking import (
    ThinkingStreamBuffer,
    detect_reasoning_family,
    normalize_thinking,
)


# ===========================================================================
# normalize_thinking()
# ===========================================================================


class TestNormalizeThinking:
    """Tests for the normalize_thinking utility."""

    def test_no_tags(self):
        clean, thinking = normalize_thinking("Hello world")
        assert clean == "Hello world"
        assert thinking == ""

    def test_single_block(self):
        clean, thinking = normalize_thinking(
            "<think>reasoning here</think>The answer is 42."
        )
        assert clean == "The answer is 42."
        assert thinking == "reasoning here"

    def test_multiple_blocks(self):
        clean, thinking = normalize_thinking(
            "<think>first</think>middle<think>second</think>end"
        )
        assert clean == "middleend"
        assert "first" in thinking
        assert "second" in thinking

    def test_multiline_block(self):
        content = "<think>line1\nline2\nline3</think>answer"
        clean, thinking = normalize_thinking(content)
        assert clean == "answer"
        assert "line1\nline2\nline3" in thinking

    def test_native_field_only(self):
        clean, thinking = normalize_thinking("Hello", thinking_field="native reasoning")
        assert clean == "Hello"
        assert thinking == "native reasoning"

    def test_native_field_plus_inline(self):
        clean, thinking = normalize_thinking(
            "<think>inline</think>answer",
            thinking_field="native",
        )
        assert clean == "answer"
        assert "native" in thinking
        assert "inline" in thinking

    def test_empty_tags(self):
        clean, thinking = normalize_thinking("<think></think>content")
        assert clean == "content"
        assert thinking == ""

    def test_content_is_only_thinking(self):
        clean, thinking = normalize_thinking("<think>all thinking</think>")
        assert clean == ""
        assert thinking == "all thinking"

    def test_whitespace_around_tags(self):
        clean, thinking = normalize_thinking("  <think>thought</think>  answer  ")
        assert clean == "answer"
        assert thinking == "thought"

    def test_empty_content_no_field(self):
        clean, thinking = normalize_thinking("")
        assert clean == ""
        assert thinking == ""

    def test_empty_content_with_field(self):
        clean, thinking = normalize_thinking("", thinking_field="reasoning")
        assert clean == ""
        assert thinking == "reasoning"

    def test_nested_angle_brackets_in_thinking(self):
        clean, thinking = normalize_thinking(
            "<think>if x > 0 and y < 10</think>result"
        )
        assert clean == "result"
        assert "x > 0" in thinking

    def test_think_tag_not_confused_with_other_tags(self):
        clean, thinking = normalize_thinking("<b>bold</b> text")
        assert clean == "<b>bold</b> text"
        assert thinking == ""

    def test_native_field_empty_string(self):
        # Empty string native field should not be treated as "present"
        clean, thinking = normalize_thinking("hello", thinking_field="")
        assert clean == "hello"
        assert thinking == ""

    def test_gemma_analysis_and_final_channels(self):
        clean, thinking = normalize_thinking(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello, world!"
        )
        assert clean == "Hello, world!"
        assert thinking == "I'm\nthinking"

    def test_gemma_commentary_channel_treated_as_content(self):
        clean, thinking = normalize_thinking(
            "<|channel|>commentary<|message|>Hello from commentary<|end|>"
        )
        assert clean == "Hello from commentary"
        assert thinking == ""

    # -- asymmetric orphan-closer fallback (custom-named finetunes) ----------

    def test_orphan_closer_unknown_family(self):
        # A response with a </think> but NO <think> means the opener was in
        # the prompt prefix (asymmetric model). Even with an unknown family
        # (custom name matches no needle), route text before the closer to
        # reasoning rather than leaking it + the stray tag into content.
        clean, thinking = normalize_thinking(
            "Let me reason about this.</think>The answer is 42."
        )
        assert clean == "The answer is 42."
        assert thinking == "Let me reason about this."

    def test_orphan_closer_qwythos_name(self):
        # Real-world case: Qwythos is a qwen35 model but its name matches no
        # needle, so name-only detection returns no family. The orphan-closer
        # fallback still strips the leaked CoT.
        clean, thinking = normalize_thinking(
            "reasoning prose here</think>final answer",
            model="Qwythos-9B-Claude-Mythos-5-1M-MTP-Q8_0",
        )
        assert clean == "final answer"
        assert thinking == "reasoning prose here"

    def test_orphan_closer_authoritative_arch_family(self):
        # When the backend plumbs the GGUF arch through (engine Fix B),
        # family='qwen35' is a known starts-thinking family — same result via
        # the authoritative path.
        clean, thinking = normalize_thinking(
            "thinking...</think>answer", family="qwen35",
        )
        assert clean == "answer"
        assert thinking == "thinking..."

    def test_no_closer_passes_through_unchanged(self):
        # The fallback fires ONLY when a closer is present. A plain answer
        # (no </think>) with an unknown family must be untouched — never
        # routed wholesale into reasoning.
        text = "This is a long, ordinary answer with no thinking tags at all."
        clean, thinking = normalize_thinking(text)
        assert clean == text
        assert thinking == ""

    def test_orphan_closer_suppressed_when_thinking_off(self):
        # With thinking explicitly disabled, a stray closer is not treated as
        # a reasoning boundary (gating parity with starts-thinking families).
        clean, thinking = normalize_thinking(
            "answer body", thinking_enabled=False,
        )
        assert clean == "answer body"
        assert thinking == ""


# ===========================================================================
# ThinkingStreamBuffer
# ===========================================================================


class TestThinkingStreamBuffer:
    """Tests for the streaming thinking block buffer."""

    def test_no_tags(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("hello world")
        assert clean == "hello world"
        assert thinking == ""

    def test_complete_tag_in_one_delta(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("<think>reasoning</think>answer")
        assert "answer" in clean
        assert "reasoning" in thinking
        assert "<think>" not in clean

    def test_tag_split_across_deltas(self):
        buf = ThinkingStreamBuffer()

        c1, t1 = buf.process("before<thi")
        c2, t2 = buf.process("nk>inside thought")
        c3, t3 = buf.process("</think>after")

        # "before" should be in clean content
        full_clean = c1 + c2 + c3
        full_think = t1 + t2 + t3
        assert "before" in full_clean
        assert "after" in full_clean
        assert "inside thought" in full_think

    def test_native_passthrough(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("content", "native thinking")
        assert clean == "content"
        assert thinking == "native thinking"

    def test_native_plus_inline(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("<think>inline</think>text", "native")
        assert "text" in clean
        assert "native" in thinking
        assert "inline" in thinking

    def test_empty_delta(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("")
        assert clean == ""
        assert thinking == ""

    def test_multiple_blocks_streaming(self):
        buf = ThinkingStreamBuffer()
        c1, t1 = buf.process("<think>first</think>mid")
        c2, t2 = buf.process("dle<think>second</think>end")
        full_clean = c1 + c2
        full_think = t1 + t2
        assert "mid" in full_clean
        assert "end" in full_clean
        assert "first" in full_think
        assert "second" in full_think

    def test_flush_with_no_buffer(self):
        buf = ThinkingStreamBuffer()
        buf.process("clean text")
        clean, thinking = buf.flush()
        assert clean == ""
        assert thinking == ""

    def test_flush_partial_tag_outside_think(self):
        buf = ThinkingStreamBuffer()
        buf.process("text<thi")
        clean, thinking = buf.flush()
        # Partial tag should be flushed as content since not inside think
        assert "<thi" in clean
        assert thinking == ""

    def test_flush_partial_content_inside_think(self):
        buf = ThinkingStreamBuffer()
        buf.process("<think>partial")
        clean, thinking = buf.flush()
        # We're inside think, remaining buffer goes to thinking
        # The actual implementation: the text "partial" was already emitted as thinking
        # flush only emits what's in the tag buffer
        assert thinking == "" or "partial" not in clean

    def test_only_thinking_content(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("<think>all thinking</think>")
        assert clean == ""
        assert "all thinking" in thinking

    def test_interleaved_content_and_thinking(self):
        buf = ThinkingStreamBuffer()
        results = []
        for delta in ["He", "llo", "<think>", "hmm", "</think>", " world"]:
            c, t = buf.process(delta)
            results.append((c, t))
        full_clean = "".join(r[0] for r in results)
        full_think = "".join(r[1] for r in results)
        assert "Hello" in full_clean
        assert "world" in full_clean
        assert "hmm" in full_think

    def test_gemma_streaming_analysis_then_final(self):
        buf = ThinkingStreamBuffer()
        deltas = [
            "<|channel|>analysis<|message|>I'm\nthinking",
            "<|end|><|start|>assistant<|channel|>final<|message|>Hello, ",
            "world!",
        ]
        full_clean = ""
        full_think = ""
        for delta in deltas:
            clean, thinking = buf.process(delta)
            full_clean += clean
            full_think += thinking

        assert full_clean == "Hello, world!"
        assert full_think == "I'm\nthinking"

    def test_gemma_streaming_partial_analysis_header(self):
        buf = ThinkingStreamBuffer()
        c1, t1 = buf.process("<|channel|>anal")
        c2, t2 = buf.process("ysis<|message|>thinking")
        c3, t3 = buf.flush()

        assert c1 == ""
        assert t1 == ""
        assert c2 == ""
        assert t2 == "thinking"
        assert c3 == ""
        assert t3 == ""

    def test_buffer_bounded_under_consecutive_triggers(self):
        """Regression: ``<`` chars piling up past max_buf must not leak.

        Pre-fix bug: when ``len(_tag_buffer) >= max_buf`` AND the trigger
        char (``<``) is somewhere in the buffer, neither flush branch
        fired — every subsequent character grew the buffer without
        bound. Realistic trigger: code blocks with git conflict markers
        (``<<<<<<< HEAD``), ASCII art with rows of ``<``, or any model
        degeneration that emits the trigger char repeatedly.

        Verifies (a) the buffer size stays bounded across the stream,
        and (b) every input character is eventually emitted (nothing
        gets trapped in the buffer).
        """
        # Use a SYMMETRIC family (deepseek3 — not in _STARTS_THINKING_FAMILIES)
        # so the buffer-bound test exercises the "outside think" path.
        # Qwen3 was added to _STARTS_THINKING_FAMILIES 2026-05-10, so it
        # starts inside-think and would route everything to thinking.
        buf = ThinkingStreamBuffer(family="deepseek3")
        # 50 angle brackets followed by 50 letters — well past the
        # max_buf of 10 for ``<think>`` / ``</think>``.
        deltas = ["<"] * 50 + list("abcdefghij" * 5)
        emitted_clean: list[str] = []
        emitted_think: list[str] = []

        for delta in deltas:
            clean, think = buf.process(delta)
            emitted_clean.append(clean)
            emitted_think.append(think)
            # Buffer must never exceed the longest delimiter — otherwise
            # the leak is back. ``</think>`` is 8 chars; allow some
            # headroom for partial-tag tracking but bound it.
            assert len(buf._tag_buffer) <= 16, (
                f"buffer leaked to {len(buf._tag_buffer)} chars: "
                f"{buf._tag_buffer!r}"
            )

        # Final flush picks up any trailing partial.
        flush_clean, flush_think = buf.flush()
        emitted_clean.append(flush_clean)
        emitted_think.append(flush_think)

        # All input characters must round-trip out as content (we never
        # entered a thinking block, so nothing should land in thinking).
        round_trip = "".join(emitted_clean) + "".join(emitted_think)
        assert round_trip == "".join(deltas), (
            f"input lost during streaming. expected {''.join(deltas)!r}, "
            f"got {round_trip!r}"
        )
        # And nothing wrongly routed to thinking.
        assert "".join(emitted_think) == ""

    def test_buffer_bounded_under_consecutive_triggers_magistral(self):
        """Same regression for Magistral's [THINK]/[/THINK] markers."""
        buf = ThinkingStreamBuffer(family="magistral")
        # Repeated ``[`` (Magistral's trigger) before content.
        deltas = ["["] * 50 + list("xyzpqr" * 6)
        emitted: list[tuple[str, str]] = []

        for delta in deltas:
            emitted.append(buf.process(delta))
            assert len(buf._tag_buffer) <= 16, (
                f"buffer leaked to {len(buf._tag_buffer)} chars"
            )

        emitted.append(buf.flush())
        round_trip = "".join(c for c, _ in emitted) + "".join(t for _, t in emitted)
        assert round_trip == "".join(deltas)


# ===========================================================================
# Buffer fuzz — randomized round-trip + bound invariants
# ===========================================================================


class TestThinkingStreamBufferFuzz:
    """Property-style fuzz tests for the streaming buffer.

    What this catches that the targeted regression tests don't:

    * Asymmetric-family buffer leak: the symmetric ``<<<<<<...`` test
      proved the bound holds for ``family='qwen3'`` but never exercised
      ``family='glm47'`` — different init state (``_inside_think=True``)
      can mask leaks because content routes to thinking instead of clean.
    * Random-chunking edge cases: production streams arrive in chunks of
      varying width (single tokens, batched bursts, mid-tag splits).
      Targeted tests choose chunk boundaries on purpose; fuzz hits the
      cases nobody thought of.
    * Round-trip drift: characters that get silently swallowed under
      certain delimiter sequences. The bound check alone wouldn't notice.

    Hypothesis isn't installed in this venv, so this is hand-rolled
    with ``random.Random`` seeded for determinism. Each test does ~200
    trials of varied input lengths.
    """

    # The "interesting" alphabet: every character that participates in
    # any delimiter we care about (``<``, ``>``, ``/``, ``[``, ``]``,
    # ``|``) plus a few innocuous letters/spaces so the fuzz also
    # generates ordinary content. Sampling weights matter — spamming
    # ``<`` matches what a code-block-with-conflict-markers stream
    # actually looks like.
    _ALPHABET = (
        list("<<<<<")        # heavily weighted trigger char
        + list(">>>")
        + list("///")
        + list("[[[")
        + list("]]")
        + list("||")
        + list("think")      # tag-name-ish letters (case match)
        + list("THINK")
        + list("abcdefghij")
        + [" "] * 4
        + ["\n"]
    )

    # Buffer cap headroom. The longest delimiter the parser tracks is
    # ``</think>`` at 8 chars (or ``[/THINK]`` at 8 chars for Magistral).
    # The bound-and-trim invariant is "buffer ≤ len(longest_marker) - 1"
    # plus a small carry. 16 is loose enough not to false-positive but
    # tight enough to catch real regressions.
    _BUFFER_CAP = 16

    def _generate_deltas(
        self, rng: object, length: int, max_chunk: int,
    ) -> list[str]:
        """Generate a list of streaming deltas of total ``length`` chars.

        Each delta is 1..max_chunk chars sampled from _ALPHABET. Mixed
        widths are critical — an all-1-char fuzz misses bugs that only
        appear when a partial tag spans the chunk boundary in just the
        wrong place.
        """
        deltas: list[str] = []
        produced = 0
        while produced < length:
            chunk_len = rng.randint(1, max_chunk)
            chunk_len = min(chunk_len, length - produced)
            chunk = "".join(rng.choices(self._ALPHABET, k=chunk_len))
            deltas.append(chunk)
            produced += chunk_len
        return deltas

    def _run_trial(
        self,
        family: str,
        deltas: list[str],
        *,
        is_asymmetric: bool,
    ) -> None:
        """Process a sequence of deltas; assert bound + round-trip."""
        buf = ThinkingStreamBuffer(family=family)
        emitted_clean: list[str] = []
        emitted_think: list[str] = []

        for delta in deltas:
            clean, think = buf.process(delta)
            emitted_clean.append(clean)
            emitted_think.append(think)
            assert len(buf._tag_buffer) <= self._BUFFER_CAP, (
                f"family={family} buffer leaked to {len(buf._tag_buffer)} "
                f"chars: {buf._tag_buffer!r} after delta {delta!r}"
            )

        flush_clean, flush_think = buf.flush()
        emitted_clean.append(flush_clean)
        emitted_think.append(flush_think)

        original = "".join(deltas)
        round_trip = "".join(emitted_clean) + "".join(emitted_think)
        # Round-trip MUST preserve every byte. If a character disappears,
        # that's a content-loss bug regardless of which channel it ended
        # up in.
        assert round_trip == original, (
            f"family={family} round-trip mismatch.\n"
            f"  expected: {original!r}\n"
            f"  got:      {round_trip!r}\n"
            f"  diff at idx {next((i for i, (a, b) in enumerate(zip(original, round_trip)) if a != b), -1)}"
        )

        # For symmetric families with no complete ``<think>`` opener in
        # the random input, nothing should land in thinking. This is a
        # strong invariant — if the parser starts hallucinating thinking
        # content from random text, we catch it here.
        if not is_asymmetric:
            think_text = "".join(emitted_think)
            # Allow only chars from a complete ``<think>...`` block.
            # If no opener appeared in the input, think_text MUST be empty.
            if "<think>" not in original and "[THINK]" not in original:
                assert think_text == "", (
                    f"family={family} routed {think_text!r} to thinking "
                    f"despite no opener in input"
                )

    def test_fuzz_asymmetric_qwen3(self):
        """Qwen3 is in ``_STARTS_THINKING_FAMILIES`` since 2026-05-10 —
        starts inside-think, so it IS an asymmetric family for fuzz purposes
        (content before ``</think>`` routes to thinking, not content)."""
        import random
        rng = random.Random(0xCAFEBABE)
        for trial in range(200):
            length = rng.randint(20, 400)
            max_chunk = rng.randint(1, 8)
            deltas = self._generate_deltas(rng, length, max_chunk)
            try:
                self._run_trial("qwen3", deltas, is_asymmetric=True)
            except AssertionError as exc:
                raise AssertionError(
                    f"trial {trial} (len={length}, max_chunk={max_chunk}, "
                    f"input={''.join(deltas)!r}): {exc}"
                ) from exc

    def test_fuzz_symmetric_magistral(self):
        """Symmetric ``[THINK]...[/THINK]`` family — Magistral.

        Critical because Magistral's trigger char ``[`` is much more
        common in real content (markdown links, code, JSON arrays) than
        ``<`` is — so a Magistral leak would manifest sooner in the
        wild than the qwen3 one we already test.
        """
        import random
        rng = random.Random(0xDEADC0DE)
        for trial in range(200):
            length = rng.randint(20, 400)
            max_chunk = rng.randint(1, 8)
            deltas = self._generate_deltas(rng, length, max_chunk)
            try:
                self._run_trial("magistral", deltas, is_asymmetric=False)
            except AssertionError as exc:
                raise AssertionError(
                    f"trial {trial} (len={length}, max_chunk={max_chunk}, "
                    f"input={''.join(deltas)!r}): {exc}"
                ) from exc

    def test_fuzz_asymmetric_glm47(self):
        """Asymmetric family — GLM-4.7 starts inside ``<think>``.

        Different init state from the symmetric families
        (``_inside_think=True``) — content routes to thinking until
        ``</think>`` arrives. We assert the same round-trip + bound
        invariants; a leak here would have looked fine to the
        symmetric-family fuzz because the asymmetric init was never
        exercised.
        """
        import random
        rng = random.Random(0xFEEDFACE)
        for trial in range(200):
            length = rng.randint(20, 400)
            max_chunk = rng.randint(1, 8)
            deltas = self._generate_deltas(rng, length, max_chunk)
            try:
                self._run_trial("glm47", deltas, is_asymmetric=True)
            except AssertionError as exc:
                raise AssertionError(
                    f"trial {trial} (len={length}, max_chunk={max_chunk}, "
                    f"input={''.join(deltas)!r}): {exc}"
                ) from exc

    def test_fuzz_single_char_chunks(self):
        """Worst-case streaming: every char arrives as its own delta.

        Single-char chunks are the hardest case for partial-marker
        tracking because every boundary is a potential split point.
        Dedicated trial because the random generator's chunk-size
        distribution under-samples single-char streams.
        """
        import random
        rng = random.Random(0x12345678)
        for trial in range(50):
            length = rng.randint(20, 200)
            deltas = list("".join(rng.choices(self._ALPHABET, k=length)))
            for family, asym in (
                ("qwen3", True), ("magistral", False), ("glm47", True),
            ):
                try:
                    self._run_trial(family, list(deltas), is_asymmetric=asym)
                except AssertionError as exc:
                    raise AssertionError(
                        f"family={family} trial {trial} "
                        f"input={''.join(deltas)!r}: {exc}"
                    ) from exc


# ===========================================================================
# Base types
# ===========================================================================


class TestBaseTypes:
    """Test that thinking fields exist with correct defaults."""

    def test_message_thinking_default(self):
        msg = Message(role="assistant", content="hi")
        assert msg.thinking is None

    def test_message_thinking_set(self):
        msg = Message(role="assistant", content="hi", thinking="reasoning")
        assert msg.thinking == "reasoning"

    def test_stream_chunk_thinking_delta_default(self):
        chunk = InternalStreamChunk()
        assert chunk.thinking_delta == ""

    def test_stream_chunk_thinking_delta_set(self):
        chunk = InternalStreamChunk(thinking_delta="thought")
        assert chunk.thinking_delta == "thought"

    def test_request_think_default(self):
        req = InternalChatRequest(
            model="test", messages=[Message(role="user", content="hi")]
        )
        assert req.think is False

    def test_request_think_set(self):
        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            think=True,
        )
        assert req.think is True


# ===========================================================================
# Ollama backend integration
# ===========================================================================


class MockTransport(httpx.AsyncBaseTransport):
    """Programmable async transport for httpx tests."""

    def __init__(self) -> None:
        self.responses: dict[str, httpx.Response] = {}
        self.requests: list[httpx.Request] = []

    def add_response(
        self, method: str, url: str, *, status: int = 200, json_data: dict | None = None
    ) -> None:
        key = f"{method.upper()} {url}"
        body = json.dumps(json_data or {}).encode()
        self.responses[key] = httpx.Response(
            status_code=status,
            content=body,
            headers={"content-type": "application/json"},
            request=httpx.Request(method, url),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {str(request.url)}"
        if key in self.responses:
            return self.responses[key]
        return httpx.Response(404, request=request, text="Not found")


class TestOllamaThinking:
    """Test Ollama backend thinking capture."""

    def _make_backend(self) -> tuple[OllamaBackend, MockTransport]:
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        return OllamaBackend(client, "http://ollama:11434"), t

    @pytest.mark.asyncio
    async def test_chat_captures_native_thinking(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://ollama:11434/api/chat",
            json_data={
                "model": "qwq:32b",
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "thinking": "Let me reason about this...",
                },
                "done": True,
            },
        )
        req = InternalChatRequest(
            model="qwq:32b",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "The answer is 42."
        assert result.message.thinking == "Let me reason about this..."

    @pytest.mark.asyncio
    async def test_chat_strips_inline_thinking(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://ollama:11434/api/chat",
            json_data={
                "model": "qwq:32b",
                "message": {
                    "role": "assistant",
                    "content": "<think>reasoning</think>The answer.",
                },
                "done": True,
            },
        )
        req = InternalChatRequest(
            model="qwq:32b",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "The answer."
        assert result.message.thinking == "reasoning"

    @pytest.mark.asyncio
    async def test_chat_no_thinking(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://ollama:11434/api/chat",
            json_data={
                "model": "llama3.1:8b",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
            },
        )
        req = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "Hello!"
        assert result.message.thinking is None

    @pytest.mark.asyncio
    async def test_think_true_in_payload(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://ollama:11434/api/chat",
            json_data={
                "model": "qwq:32b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            },
        )
        req = InternalChatRequest(
            model="qwq:32b",
            messages=[Message(role="user", content="test")],
            think=True,
        )
        await backend.chat(req)
        sent = transport.requests[-1]
        body = json.loads(sent.content)
        assert body["think"] is True

    @pytest.mark.asyncio
    async def test_think_false_not_in_payload(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://ollama:11434/api/chat",
            json_data={
                "model": "llama3.1:8b",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            },
        )
        req = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="test")],
            think=False,
        )
        await backend.chat(req)
        sent = transport.requests[-1]
        body = json.loads(sent.content)
        assert "think" not in body


# ===========================================================================
# OpenAI backend integration
# ===========================================================================


class TestOpenAIThinking:
    """Test OpenAI backend thinking capture."""

    def _make_backend(self) -> tuple[OpenAIBackend, MockTransport]:
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        return OpenAIBackend(client, "http://openai:8080/v1"), t

    @pytest.mark.asyncio
    async def test_chat_captures_reasoning_content(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://openai:8080/v1/chat/completions",
            json_data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "42",
                            "reasoning_content": "Let me think...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "deepseek-r1",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
        req = InternalChatRequest(
            model="deepseek-r1",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "42"
        assert result.message.thinking == "Let me think..."

    @pytest.mark.asyncio
    async def test_chat_strips_inline_tags(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://openai:8080/v1/chat/completions",
            json_data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>hmm</think>answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "test",
                "usage": {},
            },
        )
        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "answer"
        assert result.message.thinking == "hmm"


# ===========================================================================
# LlamaCpp backend integration
# ===========================================================================


class TestLlamaCppThinking:
    """Test LlamaCpp backend thinking capture."""

    def _make_backend(self) -> tuple[LlamaCppBackend, MockTransport]:
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        return LlamaCppBackend(client, "http://llamacpp:8080"), t

    @pytest.mark.asyncio
    async def test_chat_captures_reasoning_content(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://llamacpp:8080/v1/chat/completions",
            json_data={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "result",
                            "reasoning_content": "deep thought",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "qwq",
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )
        req = InternalChatRequest(
            model="qwq",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "result"
        assert result.message.thinking == "deep thought"

    @pytest.mark.asyncio
    async def test_chat_no_thinking(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://llamacpp:8080/v1/chat/completions",
            json_data={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "llama",
                "usage": {},
            },
        )
        req = InternalChatRequest(
            model="llama",
            messages=[Message(role="user", content="test")],
        )
        result = await backend.chat(req)
        assert result.message.content == "hello"
        assert result.message.thinking is None

    @pytest.mark.asyncio
    async def test_chat_payload_enables_qwen_reasoning(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://llamacpp:8080/v1/chat/completions",
            json_data={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "Qwen3.5-27B",
                "usage": {},
            },
        )
        req = InternalChatRequest(
            model="Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS",
            messages=[Message(role="user", content="test")],
            think=True,
        )
        await backend.chat(req)

        sent = json.loads(transport.requests[0].content.decode("utf-8"))
        assert sent["chat_template_kwargs"] == {"enable_thinking": True}
        assert sent["reasoning_format"] == "deepseek"

    @pytest.mark.asyncio
    async def test_apply_template_forwards_qwen_thinking_kwargs(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://llamacpp:8080/apply-template",
            json_data={"prompt": "<think>reason</think>answer"},
        )

        await backend.apply_template(
            [{"role": "user", "content": "test"}],
            chat_template_kwargs={"enable_thinking": True},
        )

        sent = json.loads(transport.requests[0].content.decode("utf-8"))
        assert sent["chat_template_kwargs"] == {"enable_thinking": True}

    def test_qwen_thinking_disables_completion_fast_path(self):
        backend, _ = self._make_backend()

        req = InternalChatRequest(
            model="Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS",
            messages=[Message(role="user", content="test")],
            think=True,
        )

        assert backend._can_use_completion(req) is False

    def test_qwen_non_thinking_keeps_completion_fast_path(self):
        backend, _ = self._make_backend()

        req = InternalChatRequest(
            model="Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS",
            messages=[Message(role="user", content="test")],
            think=False,
        )

        assert backend._can_use_completion(req) is True


# ===========================================================================
# Streaming format conversion
# ===========================================================================


class TestStreamingConversion:
    """Test that thinking_delta is routed to augmentum metadata in stream chunks."""

    def test_ndjson_includes_thinking_delta(self):
        chunk = InternalStreamChunk(
            content_delta="hello",
            thinking_delta="reasoning",
            model="test",
        )
        result = _chunk_to_ollama_chat_ndjson(chunk, "test")
        assert result["augmentum"]["model_thinking_delta"] == "reasoning"
        assert result["message"]["content"] == "hello"

    def test_ndjson_no_thinking_no_augmentum(self):
        chunk = InternalStreamChunk(content_delta="hello", model="test")
        result = _chunk_to_ollama_chat_ndjson(chunk, "test")
        assert "augmentum" not in result

    def test_ndjson_thinking_merges_with_existing_augmentum(self):
        chunk = InternalStreamChunk(
            content_delta="text",
            thinking_delta="thought",
            model="test",
            augmentum={"mode": "passthrough"},
        )
        result = _chunk_to_ollama_chat_ndjson(chunk, "test")
        assert result["augmentum"]["model_thinking_delta"] == "thought"
        assert result["augmentum"]["mode"] == "passthrough"

    def test_sse_includes_thinking_delta(self):
        chunk = InternalStreamChunk(
            content_delta="hello",
            thinking_delta="reasoning",
            model="test",
        )
        result = _chunk_to_openai_sse(chunk, "chunk-id")
        assert result["augmentum"]["model_thinking_delta"] == "reasoning"

    def test_sse_no_thinking_no_augmentum(self):
        chunk = InternalStreamChunk(content_delta="hello", model="test")
        result = _chunk_to_openai_sse(chunk, "chunk-id")
        assert "augmentum" not in result

    def test_sse_thinking_merges_with_existing_augmentum(self):
        chunk = InternalStreamChunk(
            content_delta="text",
            thinking_delta="thought",
            model="test",
            augmentum={"phases": [{"name": "ASSESS"}]},
        )
        result = _chunk_to_openai_sse(chunk, "chunk-id")
        assert result["augmentum"]["model_thinking_delta"] == "thought"
        assert result["augmentum"]["phases"] == [{"name": "ASSESS"}]


# ===========================================================================
# Tool parsing safety
# ===========================================================================


class TestToolParsingSafety:
    """Verify TOOL_CALL: inside <think> blocks is stripped before tool parsing."""

    def test_tool_call_inside_think_stripped(self):
        content = "<think>TOOL_CALL: web_search({\"query\": \"test\"})</think>The answer."
        clean, thinking = normalize_thinking(content)
        assert "TOOL_CALL:" not in clean
        assert "TOOL_CALL:" in thinking
        assert clean == "The answer."

    def test_tool_call_outside_think_preserved(self):
        content = "TOOL_CALL: calculator({\"expression\": \"2+2\"})"
        clean, thinking = normalize_thinking(content)
        assert "TOOL_CALL:" in clean
        assert thinking == ""

    def test_tool_call_mixed_locations(self):
        content = (
            "<think>TOOL_CALL: bad_call({})</think>"
            "TOOL_CALL: good_call({\"x\": 1})"
        )
        clean, thinking = normalize_thinking(content)
        assert "good_call" in clean
        assert "bad_call" not in clean
        assert "bad_call" in thinking

    def test_streaming_tool_call_inside_think(self):
        buf = ThinkingStreamBuffer()
        deltas = ["<think>", "TOOL_CALL: search", "({})", "</think>", "result"]
        full_clean = ""
        full_think = ""
        for d in deltas:
            c, t = buf.process(d)
            full_clean += c
            full_think += t
        assert "TOOL_CALL:" not in full_clean
        assert "TOOL_CALL:" in full_think

    def test_streaming_tool_call_outside_think(self):
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("TOOL_CALL: calc({\"x\": 1})")
        assert "TOOL_CALL:" in clean


# ===========================================================================
# OpenAI response helper
# ===========================================================================


class TestOpenAIResponseHelper:
    """Test to_openai_chat_response includes reasoning_content."""

    def test_includes_reasoning_content_when_thinking(self):
        resp = InternalChatResponse(
            message=Message(
                role="assistant", content="answer", thinking="my reasoning"
            ),
            model="test",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        result = to_openai_chat_response(resp)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "answer"
        assert msg["reasoning_content"] == "my reasoning"

    def test_no_reasoning_content_when_no_thinking(self):
        resp = InternalChatResponse(
            message=Message(role="assistant", content="answer"),
            model="test",
            usage=Usage(),
        )
        result = to_openai_chat_response(resp)
        msg = result["choices"][0]["message"]
        assert "reasoning_content" not in msg


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    """Test thinking config defaults."""

    def test_think_enabled_default(self):
        from augmentum.config import Settings

        s = Settings()
        assert s.think_enabled is True

    def test_think_enabled_disabled(self):
        from augmentum.config import Settings

        s = Settings(think_enabled=False)
        assert s.think_enabled is False


# ===========================================================================
# Gemma 4 asymmetric channel (<|channel>thought…<channel|>)
# ===========================================================================


class TestGemma4NormalizeThinking:
    """Post-hoc extraction of Gemma 4's asymmetric channel format."""

    def test_basic_thought_and_answer(self):
        clean, thinking = normalize_thinking(
            "<|channel>thought\nI need to add 2+2.<channel|>The answer is 4.",
            family="gemma4",
        )
        assert clean == "The answer is 4."
        assert thinking == "I need to add 2+2."

    def test_no_newline_after_thought(self):
        clean, thinking = normalize_thinking(
            "<|channel>thoughtreasoning here<channel|>final",
            family="gemma4",
        )
        assert clean == "final"
        assert thinking == "reasoning here"

    def test_truncated_stream_no_closer(self):
        # Gemma 4 doesn't guarantee a closer on max_tokens truncation.
        clean, thinking = normalize_thinking(
            "<|channel>thought\nStill thinking when cutoff happened",
            family="gemma4",
        )
        assert clean == ""
        assert thinking == "Still thinking when cutoff happened"

    def test_answer_before_opener_is_content(self):
        clean, thinking = normalize_thinking(
            "Prefix text<|channel>thought\nreasoning<channel|>Answer.",
            family="gemma4",
        )
        assert clean == "Prefix textAnswer."
        assert thinking == "reasoning"

    def test_gemma4_hint_ignores_gemma3_markers(self):
        # With gemma4 hint, Gemma 3 parser is not active — so symmetric
        # channel markers should NOT be stripped.
        clean, _ = normalize_thinking(
            "<|channel|>analysis<|message|>x<|end|>answer",
            family="gemma4",
        )
        assert "<|channel|>" in clean

    def test_native_field_prevails(self):
        clean, thinking = normalize_thinking(
            "<|channel>thought\ninline<channel|>answer",
            thinking_field="native reasoning text",
            family="gemma4",
        )
        assert clean == "answer"
        assert "native reasoning text" in thinking
        assert "inline" in thinking


class TestGemma4StreamBuffer:
    """Streaming parse of Gemma 4 asymmetric channel format."""

    def test_single_chunk(self):
        buf = ThinkingStreamBuffer(family="gemma4")
        clean, thinking = buf.process(
            "<|channel>thought\nreason here<channel|>final answer"
        )
        assert clean == "final answer"
        assert thinking == "reason here"

    def test_opener_split_across_chunks(self):
        buf = ThinkingStreamBuffer(family="gemma4")
        deltas = ["<|chan", "nel>th", "ought\nrea", "soning", "<channel|>", "answer"]
        full_clean, full_think = "", ""
        for d in deltas:
            c, t = buf.process(d)
            full_clean += c
            full_think += t
        fc, ft = buf.flush()
        full_clean += fc
        full_think += ft
        assert full_clean == "answer"
        assert full_think == "reasoning"

    def test_closer_split_across_chunks(self):
        buf = ThinkingStreamBuffer(family="gemma4")
        deltas = ["<|channel>thought\nthinking", "<chan", "nel|>visible"]
        full_clean, full_think = "", ""
        for d in deltas:
            c, t = buf.process(d)
            full_clean += c
            full_think += t
        fc, ft = buf.flush()
        full_clean += fc
        full_think += ft
        assert full_clean == "visible"
        assert full_think == "thinking"

    def test_truncated_stream_routes_to_thinking(self):
        buf = ThinkingStreamBuffer(family="gemma4")
        c, t = buf.process("<|channel>thought\nstill going")
        fc, ft = buf.flush()
        assert (c + fc) == ""
        assert (t + ft) == "still going"

    def test_no_channel_passthrough(self):
        buf = ThinkingStreamBuffer(family="gemma4")
        c, t = buf.process("plain answer, no reasoning markers.")
        fc, _ = buf.flush()
        assert (c + fc) == "plain answer, no reasoning markers."
        assert t == ""

    def test_gemma3_content_untouched_under_gemma4_hint(self):
        # Caller said this is gemma4 — don't engage gemma3 parser on gemma3 markers.
        buf = ThinkingStreamBuffer(family="gemma4")
        c, _ = buf.process("<|channel|>analysis<|message|>x<|end|>done")
        fc, _ = buf.flush()
        assert "<|channel|>" in (c + fc)

    def test_no_hint_still_detects_gemma4(self):
        # Backward-compat path: no family given, but Gemma 4 markers present.
        buf = ThinkingStreamBuffer()
        c, t = buf.process("<|channel>thought\nreason<channel|>ans")
        fc, ft = buf.flush()
        assert (c + fc) == "ans"
        assert (t + ft) == "reason"

    def test_no_hint_preserves_gemma3_behavior(self):
        # Gemma 3 in a no-hint stream must still parse correctly (regression
        # guard against the Gemma-4-trigger-steals-Gemma-3 bug).
        buf = ThinkingStreamBuffer()
        c, t = buf.process(
            "<|channel|>analysis<|message|>reason<|end|>"
            "<|start|>assistant<|channel|>final<|message|>ans"
        )
        fc, ft = buf.flush()
        assert (c + fc) == "ans"
        assert (t + ft) == "reason"


# ===========================================================================
# Family detection
# ===========================================================================


class TestDetectReasoningFamily:
    """Heuristic family detection for parser dispatch."""

    def test_by_arch_gemma4(self):
        assert detect_reasoning_family(arch="gemma4") == "gemma4"

    def test_by_arch_qwen35moe(self):
        assert detect_reasoning_family(arch="qwen35moe") == "qwen35moe"

    def test_by_arch_case_insensitive(self):
        assert detect_reasoning_family(arch="GEMMA4") == "gemma4"

    def test_by_model_name_gemma4(self):
        assert (
            detect_reasoning_family(model="unsloth/gemma-4-E4B-it-GGUF") == "gemma4"
        )

    def test_by_model_name_gemma3_not_gemma4(self):
        assert detect_reasoning_family(model="google/gemma-3-4b-it") == "gemma3"

    def test_qwen36_falls_back_to_qwen35(self):
        # Qwen 3.6 reuses the qwen35 arch — name-based lookup confirms.
        assert detect_reasoning_family(model="Qwen3.6-35B-A3B") == "qwen35"

    def test_arch_beats_model(self):
        # When both provided, arch (authoritative) wins over model name.
        assert (
            detect_reasoning_family(model="gemma-4-whatever", arch="qwen35")
            == "qwen35"
        )

    def test_unknown_returns_none(self):
        assert detect_reasoning_family(model="random-model-xyz") is None
        assert detect_reasoning_family() is None

    # -- Needle-ordering invariant (T2-3) ---------------------------------------

    def test_needles_sorted_longest_first(self):
        """Sort invariant: the public _NAME_NEEDLES tuple is ordered by
        descending needle length so longer / more-specific patterns
        match before shorter / more-generic ones.

        Pre-T2-3 ordering was hand-maintained — easy to break by
        inserting a new entry at the wrong position. The sort makes
        adding a new family a single-line change AND eliminates a
        whole class of regression where a short generic needle (e.g.
        ``qwen3``) silently swallowed a more-specific one (e.g.
        ``qwen-3.7``) inserted later in the tuple.
        """
        from augmentum.utils.thinking import _NAME_NEEDLES

        lengths = [len(needle) for needle, _family in _NAME_NEEDLES]
        assert lengths == sorted(lengths, reverse=True), (
            f"_NAME_NEEDLES not sorted longest-first: lengths={lengths}"
        )

    def test_specific_needle_beats_generic_via_sort(self):
        """Concrete check: ``glm-4.7-foo`` matches glm47 even though
        ``glm`` and ``glm-4`` are both also substrings.

        Pre-T2-3 this worked only because the tuple was hand-ordered;
        the test guards against a future entry break the invariant.
        """
        assert detect_reasoning_family(model="zai/glm-4.7-flash") == "glm47"
        assert detect_reasoning_family(model="zai/glm-4.6-air") == "glm46"
        assert detect_reasoning_family(model="zai/glm-4-base") == "glm4"
        # Plain ``chatglm-6b`` should still resolve to chatglm, not glm.
        assert detect_reasoning_family(model="thudm/chatglm-6b") == "chatglm"

    def test_specific_qwen_versions(self):
        """Same invariant for the Qwen family."""
        assert detect_reasoning_family(model="Qwen3.5-Coder-32B") == "qwen35"
        assert detect_reasoning_family(model="Qwen3.6-35B-A3B") == "qwen35"
        assert detect_reasoning_family(model="qwen3-7b-instruct") == "qwen3"

    def test_specific_deepseek_versions(self):
        """Asymmetric V3.2/V4 must match before generic deepseek."""
        assert detect_reasoning_family(model="deepseek-v4-pro") == "deepseek4"
        assert detect_reasoning_family(model="deepseek-v3.2-flash") == "deepseek32"
        assert detect_reasoning_family(model="deepseek-r1-distill") == "deepseek3"
        assert detect_reasoning_family(model="DeepSeek-V2-Chat") == "deepseek2"

    def test_unresolved_arch_logs_at_debug(self, monkeypatch):
        """A non-empty arch that doesn't resolve should emit a debug
        log so future-unknown architectures surface in diagnostics
        rather than routing silently to the all-parsers fallback.

        Monkey-patches the module's structlog logger directly because
        structlog routes through its own handler chain that doesn't
        always show up in pytest's caplog capture.
        """
        import augmentum.utils.thinking as th

        captured: list[tuple[str, dict]] = []

        class _LogStub:
            def debug(self, event: str, **kwargs: object) -> None:
                captured.append((event, dict(kwargs)))

            def __getattr__(self, _name: str):
                return lambda *_a, **_kw: None

        monkeypatch.setattr(th, "log", _LogStub())

        result = th.detect_reasoning_family(arch="brand_new_arch_v9", model="")
        assert result is None
        assert any(
            event == "reasoning_family_unresolved"
            and kwargs.get("arch") == "brand_new_arch_v9"
            for event, kwargs in captured
        ), f"expected reasoning_family_unresolved debug log; got {captured}"

    def test_resolved_arch_does_not_log_unresolved(self, monkeypatch):
        """Symmetric: when arch DOES resolve, no diagnostic log fires."""
        import augmentum.utils.thinking as th

        captured: list[str] = []

        class _LogStub:
            def debug(self, event: str, **_kw: object) -> None:
                captured.append(event)

            def __getattr__(self, _name: str):
                return lambda *_a, **_kw: None

        monkeypatch.setattr(th, "log", _LogStub())

        assert th.detect_reasoning_family(arch="qwen3") == "qwen3"
        assert "reasoning_family_unresolved" not in captured

    # -- GLM family detection (Z.AI thinking models) ---------------------------

    def test_by_arch_glm4(self):
        assert detect_reasoning_family(arch="glm4") == "glm4"
        assert detect_reasoning_family(arch="chatglm") == "chatglm"

    def test_by_model_name_glm47(self):
        # Catalog entry "unsloth/GLM-4.7-Flash-GGUF" must match glm47, not glm4.
        assert detect_reasoning_family(model="unsloth/GLM-4.7-Flash-GGUF") == "glm47"

    def test_by_model_name_glm46(self):
        assert detect_reasoning_family(model="zai-org/glm-4.6-air") == "glm46"

    def test_glm_resolves_to_think_parser(self):
        # GLM models must dispatch to the <think> parser, not the catch-all.
        from augmentum.utils.thinking import _resolve_active_parsers
        assert _resolve_active_parsers("glm47") == ("think",)
        assert _resolve_active_parsers("glm4") == ("think",)
        assert _resolve_active_parsers("chatglm") == ("think",)

    def test_glm_round_trip_strips_think_tags(self):
        # End-to-end: GLM emits <think>...</think>, normalize_thinking strips it.
        from augmentum.utils.thinking import normalize_thinking
        raw = "<think>analyzing the input...</think>Hey Alex, here's my reply."
        clean, thinking = normalize_thinking(raw, family="glm47")
        assert clean == "Hey Alex, here's my reply."
        assert thinking == "analyzing the input..."


# --- Asymmetric closer (GLM-style: <think> in prompt prefix) ----------------


class TestAsymmetricCloser:
    """GLM's chat template puts <think> in the prompt prefix, so the response
    stream starts directly with reasoning content and only </think> arrives
    in the visible stream. Mirror Ollama's GLM47Parser approach: parser
    initializes in 'inside thinking' state, routes everything before </think>
    to reasoning_content. If </think> never appears, the whole response is
    reasoning (visible content empty)."""

    def test_post_hoc_glm_starts_thinking(self):
        from augmentum.utils.thinking import normalize_thinking
        raw = "Analyzing the user's casual greeting...</think>Hey Alex!"
        clean, thinking = normalize_thinking(raw, family="glm47")
        assert clean == "Hey Alex!"
        assert thinking == "Analyzing the user's casual greeting..."

    def test_post_hoc_glm_no_closer_routes_all_to_thinking(self):
        """If the model never emits </think>, route everything to thinking
        rather than leaking reasoning into the visible response."""
        from augmentum.utils.thinking import normalize_thinking
        raw = "Analyzing... reasoning step 1... step 2... no closing tag here"
        clean, thinking = normalize_thinking(raw, family="glm47")
        assert clean == ""
        assert thinking == raw

    def test_post_hoc_qwen_starts_thinking(self):
        """Qwen3 IS in ``_STARTS_THINKING_FAMILIES`` since 2026-05-10.
        Post-hoc normalize_thinking routes tagless content to thinking
        (same asymmetric behavior as GLM)."""
        from augmentum.utils.thinking import normalize_thinking
        raw = "regular content without any tags"
        clean, thinking = normalize_thinking(raw, family="qwen3")
        assert clean == ""
        assert thinking == "regular content without any tags"

    def test_streaming_glm_routes_first_chunk_to_thinking(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47")
        # First chunk arrives — model is reasoning. No <think> tag in stream
        # because GLM's template put it in the prompt prefix.
        clean, thinking = buf.process("Analyzing the user's input...")
        assert clean == ""
        assert thinking == "Analyzing the user's input..."

    def test_streaming_glm_split_at_close_tag(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47")
        # Reasoning chunks accumulate, then </think> arrives, then final answer.
        clean1, think1 = buf.process("Reasoning step 1. ")
        clean2, think2 = buf.process("Step 2.</think>Hey Alex!")
        assert clean1 == "" and "Reasoning step 1." in think1
        assert clean2 == "Hey Alex!" and "Step 2." in think2

    def test_streaming_glm_no_closer_routes_all_to_thinking(self):
        """GLM-4.7-Flash sometimes never emits </think>. Across the whole
        stream, every byte must land in thinking — none should leak to
        the visible response."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47")
        # Send the reasoning in two chunks to mimic streaming.
        c1, t1 = buf.process("Some reasoning that ")
        c2, t2 = buf.process("never closed")
        cf, tf = buf.flush()
        assert c1 + c2 + cf == ""
        assert t1 + t2 + tf == "Some reasoning that never closed"

    def test_streaming_qwen_starts_inside_thinking(self):
        """Qwen3 was added to ``_STARTS_THINKING_FAMILIES`` on 2026-05-10.
        Plain content (no tags) now routes to thinking (the prompt prefix
        injected the opener). Same asymmetric behavior as GLM."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="qwen3")
        clean, thinking = buf.process("plain content with no tags")
        assert clean == ""
        assert thinking == "plain content with no tags"

    # --- thinking_enabled=False suppresses GLM asymmetric init ----------

    def test_glm_thinking_disabled_skips_inside_init_streaming(self):
        """When the user toggles thinking OFF for GLM, the chat template puts
        </think> in the prompt prefix instead. The response is plain content
        with no reasoning. Parser must NOT initialize inside-think — that
        would route the user's actual response to thinking and leave the
        visible content empty."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47", thinking_enabled=False)
        clean, thinking = buf.process("Hey Alex, here's a direct answer.")
        assert clean == "Hey Alex, here's a direct answer."
        assert thinking == ""

    def test_glm_thinking_disabled_skips_inside_init_post_hoc(self):
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "Hey Alex, direct answer.",
            family="glm47",
            thinking_enabled=False,
        )
        assert clean == "Hey Alex, direct answer."
        assert thinking == ""

    def test_glm_thinking_enabled_explicit_uses_inside_init(self):
        """thinking_enabled=True is equivalent to None (the default) for GLM —
        both initialize inside-think."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47", thinking_enabled=True)
        clean, thinking = buf.process("Reasoning step 1.</think>Final answer.")
        assert clean == "Final answer."
        assert "Reasoning step 1." in thinking

    # --- local_engine gate: cloud OpenAI-compat hosts (#17) -------------
    #
    # The asymmetric "starts inside think" assumption is valid ONLY for a
    # local llama-server whose --jinja template injects the bare opener into
    # the prompt prefix. A cloud host (NVIDIA NIM, Fireworks, Together, Z.AI,
    # OpenRouter, …) templates server-side and returns a clean content stream
    # (or native reasoning_content). Applying the assumption there empties the
    # visible answer. local_engine=False must defuse it.

    def test_glm_cloud_local_engine_false_keeps_content_visible(self):
        """#17: GLM served by a CLOUD host returns a plain content stream (no
        leading </think>, no native reasoning). With local_engine=False the
        parser must NOT start inside-think — the whole answer is the visible
        response, not reasoning."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47", local_engine=False)
        clean, thinking = buf.process("Hey Alex, here's the answer.")
        cf, tf = buf.flush()
        assert clean + cf == "Hey Alex, here's the answer."
        assert thinking + tf == ""

    def test_deepseek_v4_cloud_local_engine_false_keeps_content_visible(self):
        """Same class, different family — proves the fix is on the CLASS
        (_STARTS_THINKING_FAMILIES), not patched for GLM specifically. A cloud
        DeepSeek-V4 plain reply stays visible."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="deepseek4", local_engine=False)
        c1, t1 = buf.process("The capital of France ")
        c2, t2 = buf.process("is Paris.")
        cf, tf = buf.flush()
        assert c1 + c2 + cf == "The capital of France is Paris."
        assert t1 + t2 + tf == ""

    def test_glm_cloud_local_engine_false_native_reasoning_passthrough(self):
        """A cloud host that DOES return native reasoning_content still routes
        reasoning to thinking and content to content — local_engine=False only
        disables the prompt-prefix assumption, not native side-channel
        handling."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47", local_engine=False)
        clean, thinking = buf.process("Visible answer.", "hidden reasoning")
        assert clean == "Visible answer."
        assert thinking == "hidden reasoning"

    def test_glm_local_engine_true_default_still_inside_think(self):
        """Regression guard: the default (local llama-server) behavior is
        unchanged — local_engine defaults True so the first chunk still routes
        to thinking, exactly as before the #17 fix."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47", local_engine=True)
        clean, thinking = buf.process("Analyzing...")
        assert clean == ""
        assert thinking == "Analyzing..."


# --- End-of-stream salvage: promote reasoning → content when content is empty


class TestSalvageEmptyContent:
    """The asymmetric-closer flake: a hybrid reasoning model sometimes routes
    100% of its output into reasoning_content (or into <think>… without ever
    emitting </think>) and never writes visible content. Without salvage the
    user sees a mute "Thought for Ns" bubble and no answer.

    Observed on GLM-5.2 through NVIDIA NIM under CC's tool-heavy prompt shape,
    and documented for GLM-4.7-Flash upstream. Salvage promotes the reasoning
    to content so the user sees an answer."""

    def test_normalize_thinking_salvage_native_field(self):
        """Cloud path — reasoning_content populated, content empty. With
        salvage on, reasoning becomes the visible answer."""
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "", "The answer is 42.", model="z-ai/glm-5.2",
            salvage_empty_content=True,
        )
        assert clean == "The answer is 42."
        assert thinking == ""

    def test_normalize_thinking_salvage_off_preserves_legacy(self):
        """Without salvage (default) the legacy behavior stands — content
        stays empty, reasoning stays in the thinking field."""
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "", "The answer is 42.", model="z-ai/glm-5.2",
        )
        assert clean == ""
        assert thinking == "The answer is 42."

    def test_normalize_thinking_salvage_inline_think_tags(self):
        """Local path — model emitted <think>…</think> and nothing after.
        Salvage promotes the reasoning."""
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "<think>The answer is 42.</think>", family="qwen3",
            salvage_empty_content=True,
        )
        assert clean == "The answer is 42."
        assert thinking == ""

    def test_normalize_thinking_salvage_noop_when_content_present(self):
        """Salvage only fires when content is empty — a normal response
        with both fields populated is unaffected."""
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "Visible answer.", "reasoning trace", model="z-ai/glm-5.2",
            salvage_empty_content=True,
        )
        assert clean == "Visible answer."
        assert thinking == "reasoning trace"

    def test_normalize_thinking_salvage_noop_when_both_empty(self):
        """Nothing to salvage — return empty pair, don't fabricate."""
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking("", None, salvage_empty_content=True)
        assert clean == ""
        assert thinking == ""

    def test_stream_buffer_salvage_native_reasoning(self):
        """NVIDIA GLM-5.2 case — reasoning_content chunks arrive, content
        chunks never do. Salvage on flush returns the accumulated reasoning
        for the caller to emit as content."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(
            model="z-ai/glm-5.2", local_engine=False,
            salvage_empty_content=True,
        )
        buf.process("", "The answer ")
        buf.process("", "is 42.")
        cf, tf = buf.flush()
        salvaged = buf.salvage()
        assert cf == "" and tf == ""
        assert salvaged == "The answer is 42."

    def test_stream_buffer_salvage_noop_when_content_arrived(self):
        """Normal streaming response — content arrived, salvage returns ""
        so the caller doesn't double-emit."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(
            model="z-ai/glm-5.2", local_engine=False,
            salvage_empty_content=True,
        )
        buf.process("Hey there.", "some reasoning")
        buf.flush()
        assert buf.salvage() == ""

    def test_stream_buffer_salvage_disabled_by_default(self):
        """Regression guard for the existing GLM-4.7-Flash tests — without
        opting in, ``salvage()`` always returns "" and the legacy
        "route-to-thinking" contract stands."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="glm47")  # local, salvage off
        buf.process("Reasoning that ")
        buf.process("never closed")
        buf.flush()
        assert buf.salvage() == ""

    def test_stream_buffer_salvage_idempotent(self):
        """Double-flush safety — calling salvage twice returns "" the
        second time. Prevents the openai_compat retry path from re-emitting
        the same salvaged text after a normal flush + salvage."""
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(
            model="z-ai/glm-5.2", local_engine=False,
            salvage_empty_content=True,
        )
        buf.process("", "reasoning only")
        buf.flush()
        assert buf.salvage() == "reasoning only"
        assert buf.salvage() == ""


# --- _template_thinking_override: per-turn enable_thinking for hybrid models -


class TestTemplateThinkingOverride:
    """The function that decides whether to inject ``enable_thinking`` into
    the OpenAI payload for hybrid reasoning models. Drives the chat-composer
    thinking button's effect on Qwen3.x and GLM-4.x."""

    def _override(self, model, think):
        from augmentum.models.openai_compat import _template_thinking_override
        return _template_thinking_override(model, think)

    def test_qwen3_hybrid_follows_request_think(self):
        assert self._override("qwen3-7b", True) is True
        assert self._override("qwen3-7b", False) is False

    def test_qwen35_thinking_variant_locked_on(self):
        # "Thinking" suffix means the variant is hard-locked into thinking mode.
        assert self._override("Qwen3.5-14B-Thinking", False) is True

    def test_qwen3_instruct_variant_locked_off(self):
        assert self._override("Qwen3-7B-Instruct", True) is False

    def test_glm_47_flash_follows_request_think(self):
        # The user's actual model. Was previously returning None — no override
        # at all — which is why thinking leaked.
        assert self._override("GLM-4.7-Flash-Q4_K_M", True) is True
        assert self._override("GLM-4.7-Flash-Q4_K_M", False) is False

    def test_glm_46_air_follows_request_think(self):
        assert self._override("zai-org/GLM-4.6-Air", True) is True
        assert self._override("zai-org/GLM-4.6-Air", False) is False

    def test_chatglm_recognized(self):
        assert self._override("chatglm-6b", True) is True

    def test_glm_5_follows_request_think(self):
        # GLM-5.x (5.0/5.1/5.2, Jun 2026) shares the GLM-4.5+ hybrid
        # ``enable_thinking`` chat-template kwarg. Prior to the glm5 needle
        # this returned None, so the composer toggle silently no-op'd on
        # local llama-server GLM-5.
        assert self._override("z-ai/glm-5.2", True) is True
        assert self._override("z-ai/glm-5.2", False) is False
        assert self._override("GLM-5-Air", True) is True
        assert self._override("glm-5.1-flash", False) is False

    def test_non_reasoning_model_returns_none(self):
        # Mistral, Llama 3, etc. — no thinking kwarg should be added.
        assert self._override("mistral-7b-instruct", True) is None
        assert self._override("llama-3.1-8b", False) is None

    def test_legacy_alias_still_imports(self):
        # Brief BC: external code that imported the old name shouldn't break.
        from augmentum.models.openai_compat import (
            _qwen_thinking_override,
            _template_thinking_override,
        )
        assert _qwen_thinking_override is _template_thinking_override

    def test_exaone_4_follows_request_think(self):
        # LG AI EXAONE 4.0 / EXAONE-Deep — README documents enable_thinking
        # kwarg gating reasoning mode.
        assert self._override("LGAI-EXAONE/EXAONE-4.0-32B", True) is True
        assert self._override("LGAI-EXAONE/EXAONE-4.0-32B", False) is False
        assert self._override("EXAONE-Deep-32B", True) is True

    def test_nemotron_follows_request_think(self):
        # NVIDIA Nemotron 3 Nano Reasoning — enable_thinking defaults to True
        # but UI toggle should still flip it.
        assert self._override("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", True) is True
        assert self._override("nvidia/Nemotron-Nano-9B-v2", False) is False


# --- Family detection for newly-added reasoning families ------------------


class TestNewReasoningFamilyDetection:
    """Verify detect_reasoning_family resolves the families surfaced by the
    HuggingFace top-100 survey: DeepSeek V3.2/V4 (asymmetric), MiniMax M-series
    (asymmetric), EXAONE 4.x (asymmetric), Nemotron + Hunyuan (symmetric)."""

    def _detect(self, model):
        from augmentum.utils.thinking import detect_reasoning_family
        return detect_reasoning_family(model=model)

    def test_deepseek_v4_resolves_to_deepseek4(self):
        assert self._detect("deepseek-ai/DeepSeek-V4-Pro") == "deepseek4"
        assert self._detect("deepseek-ai/DeepSeek-V4-Flash") == "deepseek4"

    def test_deepseek_v32_resolves_to_deepseek32(self):
        assert self._detect("deepseek-ai/DeepSeek-V3.2") == "deepseek32"

    def test_deepseek_v3_still_symmetric(self):
        # Regression: V3 (no .2 suffix) and R1 must NOT route to the asymmetric
        # family — they emit symmetric <think>...</think>.
        assert self._detect("deepseek-ai/DeepSeek-V3") == "deepseek3"
        assert self._detect("deepseek-ai/DeepSeek-R1") == "deepseek3"

    def test_minimax_m2_resolves(self):
        assert self._detect("MiniMaxAI/MiniMax-M2.7") == "minimaxm2"
        assert self._detect("MiniMaxAI/MiniMax-M2.5") == "minimaxm2"

    def test_exaone_4_resolves(self):
        assert self._detect("LGAI-EXAONE/EXAONE-4.0-32B") == "exaone4"
        assert self._detect("LGAI-EXAONE/EXAONE-Deep-32B") == "exaone4"

    def test_nemotron_resolves_from_model_name(self):
        # Path-based detection lands on the generic "nemotron" family. When a
        # GGUF is loaded, arch="nemotron_h" overrides via the arch path. Both
        # entries are wired in _FAMILY_PARSERS to the same symmetric <think>
        # parser, so either resolves correctly downstream.
        assert self._detect("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16") == "nemotron"
        assert self._detect("nvidia/Nemotron-Nano-9B-v2") == "nemotron"

    def test_nemotron_h_arch_resolves(self):
        # GGUF general.architecture path — arch field takes priority and
        # lands on the more specific entry.
        from augmentum.utils.thinking import detect_reasoning_family
        assert detect_reasoning_family(arch="nemotron_h") == "nemotron_h"

    def test_nemotron_h_moe_arch_resolves(self):
        # Nemotron 3 Nano Omni 30B-A3B — Mamba2-Transformer hybrid MoE.
        # GGUF reports general.architecture = "nemotron_h_moe"; must resolve
        # to its own family key (not fall through to the all-parsers default),
        # otherwise the symmetric <think> parser runs alongside the Gemma
        # parsers and false-positive checks slow the streaming buffer.
        from augmentum.utils.thinking import detect_reasoning_family
        assert detect_reasoning_family(arch="nemotron_h_moe") == "nemotron_h_moe"

    def test_nemotron_omni_name_resolves(self):
        # Model-name path: longest-needle-first ordering means the omni
        # variant resolves to the MoE family key even when the file name
        # also matches the shorter "nemotron" needle.
        assert self._detect("unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF") == "nemotron_h_moe"
        assert self._detect("nvidia/Nemotron-3-Omni-Reasoning") == "nemotron_h_moe"

    def test_hunyuan_hy3_resolves(self):
        assert self._detect("tencent/Hy3-preview") == "hy3"

    def test_lfm2_resolves(self):
        """Liquid AI LFM2 family — symmetric <think>. The MoE variant
        LFM2-8B-A1B and dense LFM2-2.6B both ship with symmetric think
        markers per the LFM2 technical report (Dec 2025).

        Regression context (2026-06-10): LFM-routed deliver calls were
        leaking <think>...</think> reasoning into TTS because the
        parser dispatcher had no LFM family entry. Pin the resolution
        so the streaming buffer runs the symmetric think parser.
        """
        assert self._detect("LiquidAI/LFM2-2.6B-GGUF") == "lfm2"
        assert self._detect("LiquidAI/LFM2-8B-A1B-GGUF") == "lfm2"

    def test_lfm25_resolves(self):
        """LFM2.5 — longer needle wins via the longest-first sort so
        ``LFM2.5-1.2B`` doesn't fall through to the plain ``lfm2`` key.
        """
        assert self._detect("LFM2.5-8B-A1B-Q4_0") == "lfm25"
        assert self._detect("LiquidAI/LFM2.5-1.2B-Thinking") == "lfm25"

    def test_lfm2_arch_resolves(self):
        """GGUF general.architecture field path."""
        from augmentum.utils.thinking import detect_reasoning_family
        assert detect_reasoning_family(arch="lfm2") == "lfm2"


# --- Asymmetric parsing for newly-added GLM-style families ----------------


class TestAsymmetricNewFamilies:
    """All four asymmetric families (GLM, DeepSeek V3.2/V4, MiniMax, EXAONE 4)
    share the same parser path: response stream starts INSIDE a think block,
    only </think> arrives in visible stream."""

    def test_deepseek_v4_streaming_routes_first_chunk_to_thinking(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="deepseek4")
        clean, thinking = buf.process("Reasoning step 1.</think>Final answer.")
        assert clean == "Final answer."
        assert "Reasoning step 1." in thinking

    def test_deepseek_v32_post_hoc_extracts_correctly(self):
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "Analyzing the input...</think>Here's the answer.",
            family="deepseek32",
        )
        assert clean == "Here's the answer."
        assert thinking == "Analyzing the input..."

    def test_minimax_streaming_routes_first_chunk_to_thinking(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="minimaxm2")
        clean, thinking = buf.process("Thinking carefully...</think>Done.")
        assert clean == "Done."
        assert "Thinking carefully..." in thinking

    def test_exaone_4_post_hoc_extracts_correctly(self):
        from augmentum.utils.thinking import normalize_thinking
        clean, thinking = normalize_thinking(
            "Step 1 of reasoning.</think>The result is 42.",
            family="exaone4",
        )
        assert clean == "The result is 42."
        assert thinking == "Step 1 of reasoning."

    def test_deepseek_v3_does_not_use_asymmetric_init(self):
        # Regression: original V3/R1 are symmetric. Bare prose without an opening
        # <think> tag must pass through untouched, NOT route to thinking.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="deepseek3")
        clean, thinking = buf.process("plain content with no tags")
        assert clean == "plain content with no tags"
        assert thinking == ""

    def test_thinking_disabled_skips_inside_init_for_new_families(self):
        # When user toggles thinking OFF for these families, the template puts
        # </think> in the prompt prefix → response is plain content. Parser must
        # NOT init inside-think.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        for family in ("deepseek4", "deepseek32", "minimaxm2", "exaone4"):
            buf = ThinkingStreamBuffer(family=family, thinking_enabled=False)
            clean, thinking = buf.process("Direct answer with no reasoning.")
            assert clean == "Direct answer with no reasoning.", f"{family} leaked"
            assert thinking == "", f"{family} false-positive thinking"


class TestCloudAsymmetricStreamingLeak:
    """Regression for the cloud DeepSeek-V4 CoT leak (2026-08-01).

    A cloud OpenAI-compat endpoint (``local_engine=False``) for an asymmetric
    family streamed ~16 KB of reasoning inline in ``content``, terminated by a
    lone ``</think>``. Before the fix the streaming buffer refused to start
    inside-think on cloud (issue #17 guard) AND had no orphan-closer branch, so
    the whole reasoning block leaked into visible content and only the tag was
    stripped. The fix lets known asymmetric families start inside-think on
    cloud WHEN the salvage safety net is enabled, and adds a data-driven
    orphan-closer branch for unknown finetunes."""

    def test_cloud_deepseek_v4_with_salvage_routes_cot_to_thinking(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(
            family="deepseek4", local_engine=False, salvage_empty_content=True,
        )
        # Reasoning arrives across several chunks BEFORE the orphan closer.
        c1, t1 = buf.process("The user wants a weather report. Plan: ")
        c2, t2 = buf.process("fetch Open-Meteo, cross-check NWS.")
        c3, t3 = buf.process("</think>Here is your report.")
        assert (c1 + c2) == "", "reasoning leaked into visible content"
        assert "Plan:" in (t1 + t2)
        assert c3 == "Here is your report."

    def test_cloud_without_salvage_keeps_strict_local_only(self):
        # The soft-trigger probe builds the buffer WITHOUT salvage; it must keep
        # the #17-safe behavior and not assume inside-think on cloud.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="deepseek4", local_engine=False)
        clean, thinking = buf.process("Just a plain cloud answer.")
        assert clean == "Just a plain cloud answer."
        assert thinking == ""

    def test_cloud_side_channel_disarms_inside_think(self):
        # A cloud endpoint that DOES stream reasoning via reasoning_content
        # (thinking_delta) must flip inside-think off so its clean content
        # streams normally rather than routing into reasoning.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(
            family="deepseek4", local_engine=False, salvage_empty_content=True,
        )
        c1, t1 = buf.process("", "native reasoning here")
        c2, t2 = buf.process("The visible answer.")
        assert t1 == "native reasoning here"
        assert c2 == "The visible answer."

    def test_unknown_family_orphan_closer_routes_before_to_thinking(self):
        # A finetune that slipped family detection: no family hint, a lone
        # </think> with no opener. The orphan-closer branch routes the text
        # before it to reasoning (parity with non-streaming normalize_thinking).
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("hidden reasoning</think>visible answer")
        assert clean == "visible answer"
        assert thinking == "hidden reasoning"

    def test_symmetric_pair_after_orphan_still_works(self):
        # After an orphan closer is consumed, a later genuine <think>..</think>
        # pair is still extracted normally.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process(
            "first reasoning</think>answer <think>more</think> tail"
        )
        assert clean == "answer  tail"
        assert "first reasoning" in thinking
        assert "more" in thinking

    def test_plain_content_no_closer_passes_through_default_family(self):
        # No family, plain content, no tags at all — must not be touched.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer()
        clean, thinking = buf.process("totally normal answer with no markers")
        assert clean == "totally normal answer with no markers"
        assert thinking == ""


# --- Mistral Magistral parser ---------------------------------------------


class TestMagistralFamily:
    """Magistral uses `[THINK]...[/THINK]` SPECIAL TOKENS (single token IDs)
    as symmetric delimiters. Both tokens appear in the response stream;
    controlled by the system prompt, not a chat template kwarg."""

    def test_family_detection(self):
        from augmentum.utils.thinking import detect_reasoning_family
        assert detect_reasoning_family(model="mistralai/Magistral-Small-2509") == "magistral"
        assert detect_reasoning_family(model="mistralai/Magistral-Small-2506") == "magistral"
        assert detect_reasoning_family(model="bartowski/Magistral-Small-2509-GGUF") == "magistral"

    def test_vanilla_mistral_does_not_resolve_to_magistral(self):
        # Regression: vanilla Mistral 7B / Small / Large do NOT emit
        # [THINK]/[/THINK]. Only Magistral does. The needle "magistral" must
        # not match plain "mistral".
        from augmentum.utils.thinking import detect_reasoning_family
        assert detect_reasoning_family(model="mistralai/Mistral-7B-Instruct-v0.3") is None
        assert detect_reasoning_family(model="mistralai/Mistral-Small-Instruct-2501") is None

    def test_post_hoc_extracts_bracketed_thinking(self):
        from augmentum.utils.thinking import normalize_thinking
        raw = "[THINK]Working through the problem step by step.[/THINK]The answer is 42."
        clean, thinking = normalize_thinking(raw, family="magistral")
        assert clean == "The answer is 42."
        assert thinking == "Working through the problem step by step."

    def test_post_hoc_multiple_blocks(self):
        from augmentum.utils.thinking import normalize_thinking
        raw = "[THINK]First.[/THINK]Mid.[THINK]Second.[/THINK]End."
        clean, thinking = normalize_thinking(raw, family="magistral")
        # Both reasoning blocks stripped; visible content (between + after) preserved.
        assert clean == "Mid.End."
        assert "First." in thinking
        assert "Second." in thinking

    def test_post_hoc_no_thinking_passes_through(self):
        from augmentum.utils.thinking import normalize_thinking
        raw = "Just a direct answer with no reasoning markers."
        clean, thinking = normalize_thinking(raw, family="magistral")
        assert clean == "Just a direct answer with no reasoning markers."
        assert thinking == ""

    def test_streaming_basic(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="magistral")
        clean, thinking = buf.process("[THINK]Reasoning.[/THINK]Answer.")
        assert clean == "Answer."
        assert thinking == "Reasoning."

    def test_streaming_split_across_chunks(self):
        # Markers may arrive split across deltas — state machine must buffer.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="magistral")
        c1, t1 = buf.process("[THI")
        c2, t2 = buf.process("NK]Mid-")
        c3, t3 = buf.process("reasoning here[/TH")
        c4, t4 = buf.process("INK]Final answer.")
        assert c1 + c2 + c3 + c4 == "Final answer."
        assert (t1 + t2 + t3 + t4) == "Mid-reasoning here"

    def test_streaming_bracket_in_content_does_not_false_trigger(self):
        # `[` is much more common in regular text than `<`. The state machine
        # must flush quickly when the bracket isn't part of a [THINK] tag.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="magistral")
        clean, thinking = buf.process("The result is [42] and the list is [a, b, c].")
        assert clean == "The result is [42] and the list is [a, b, c]."
        assert thinking == ""

    def test_streaming_real_world_with_brackets_in_thinking(self):
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="magistral")
        raw = "[THINK]Let me check items [1] and [2].[/THINK]Items 1 and 2 match."
        clean, thinking = buf.process(raw)
        assert clean == "Items 1 and 2 match."
        assert thinking == "Let me check items [1] and [2]."

    def test_streaming_unfinished_thinking_routes_to_thinking_on_flush(self):
        # If the model never closes the [THINK] block (truncated stream), the
        # in-progress reasoning must route to the thinking channel rather
        # than leak as visible content.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="magistral")
        c1, t1 = buf.process("[THINK]Started reasoning but never closed")
        cf, tf = buf.flush()
        assert (c1 + cf) == ""
        assert "Started reasoning but never closed" in (t1 + tf)

    def test_default_family_still_uses_angle_brackets(self):
        # Regression: Qwen / DeepSeek / etc. must keep using <think>/</think>
        # markers. The Magistral refactor must not change their behavior.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="qwen3")
        clean, thinking = buf.process("<think>Reasoning.</think>Answer.")
        assert clean == "Answer."
        assert thinking == "Reasoning."

    def test_default_family_ignores_bracketed_markers(self):
        # Qwen3 starts inside-think (asymmetric since 2026-05-10), so tagless
        # content routes to thinking. The `[THINK]` brackets must NOT be parsed
        # as Magistral markers — they should pass through verbatim as thinking
        # text, not flip the parser state.
        from augmentum.utils.thinking import ThinkingStreamBuffer
        buf = ThinkingStreamBuffer(family="qwen3")
        clean, thinking = buf.process("Here's a snippet: [THINK]example[/THINK] for reference.")
        # All content routes to thinking (qwen3 asymmetric, no </think>)
        assert clean == ""
        assert thinking == "Here's a snippet: [THINK]example[/THINK] for reference."
