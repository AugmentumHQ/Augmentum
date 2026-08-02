"""Native <tool_call> JSON block handling in the TagSieve.

Regression for 2026-06-11: Qwen-family models fall back to their
trained native format when tools arrive via prompt text (the voice
path). The strict tag grammar and the attr-pair salvage both miss
JSON bodies, so the block was SPOKEN in TTS and nothing executed.
"""
from __future__ import annotations

from augmentum.companion_runtime.tool_protocol import TagSieve

KNOWN = {"note.append", "note.create", "web.search", "grove.play_matching"}


def _run(sieve, text, chunk_size=7):
    """Feed text in small chunks (streaming shape), then drain."""
    events = []
    for i in range(0, len(text), chunk_size):
        events.extend(sieve.feed(text[i:i + chunk_size]))
    events.extend(sieve.drain())
    return events


def _tags(events):
    return [t for _, t in events if t is not None]


def _spoken(events):
    return "".join(c for c, _ in events if c)


def test_native_block_parses_and_executes():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = (
        'Let me add that. <tool_call>{"name": "note.append", '
        '"arguments": {"content": "CPI rose 2.9% in 2026"}}</tool_call>'
    )
    events = _run(sieve, text)
    tags = _tags(events)
    assert len(tags) == 1
    assert tags[0].name == "note.append"
    assert tags[0].args["content"] == "CPI rose 2.9% in 2026"
    spoken = _spoken(events)
    assert "tool_call" not in spoken
    assert "arguments" not in spoken
    assert "Let me add that." in spoken


def test_native_block_long_body_never_leaks_midstream():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    body = "x" * 1200  # far past the 256-char tail window
    text = (
        f'On it. <tool_call>{{"name": "note.append", '
        f'"arguments": {{"content": "{body}"}}}}</tool_call> done.'
    )
    leaked_midstream = []
    events = []
    for i in range(0, len(text), 16):
        for clean, tag in sieve.feed(text[i:i + 16]):
            events.append((clean, tag))
            if clean and ("tool_call" in clean or '"content"' in clean):
                leaked_midstream.append(clean)
    events.extend(sieve.drain())
    assert not leaked_midstream
    tags = _tags(events)
    assert len(tags) == 1
    assert tags[0].args["content"] == body


def test_native_block_unknown_name_stripped_not_spoken():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = (
        'Hm. <tool_call>{"name": "rm.everything", '
        '"arguments": {"path": "/"}}</tool_call> Anyway.'
    )
    events = _run(sieve, text)
    assert _tags(events) == []
    spoken = _spoken(events)
    # The block is debris — neither executed nor spoken.
    assert "rm.everything" not in spoken
    assert "Hm." in spoken and "Anyway." in spoken


def test_native_block_dotted_suffix_resolves():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = '<tool_call>{"name": "play_matching", "arguments": {"query": "jazz"}}</tool_call>'
    tags = _tags(_run(sieve, text))
    assert len(tags) == 1
    assert tags[0].name == "grove.play_matching"


def test_unclosed_native_block_stripped_at_drain():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = 'Sure thing. <tool_call>{"name": "note.append", "arguments": {"content": "trunca'
    events = _run(sieve, text)
    assert _tags(events) == []
    spoken = _spoken(events)
    assert "tool_call" not in spoken
    assert "Sure thing." in spoken


def test_nested_json_arguments_coerced():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = (
        '<tool_call>{"name": "web.search", "arguments": '
        '{"query": "inflation 2026", "filters": {"region": "US"}}}</tool_call>'
    )
    tags = _tags(_run(sieve, text))
    assert len(tags) == 1
    assert tags[0].args["query"] == "inflation 2026"
    assert '"region"' in tags[0].args["filters"]  # dict coerced to JSON string


def test_strict_grammar_still_first():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = 'One sec. <tool:note.append content="hello" /> after.'
    tags = _tags(_run(sieve, text))
    assert len(tags) == 1
    assert tags[0].args["content"] == "hello"


def test_flush_compat_drops_tags_but_returns_text():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    sieve._buf = (  # noqa: SLF001 — simulating end-of-stream state
        'Text. <tool_call>{"name": "note.append", '
        '"arguments": {"content": "x"}}</tool_call>'
    )
    out = sieve.flush()
    assert "Text." in out
    assert "tool_call" not in out


def test_prose_angle_brackets_unaffected():
    sieve = TagSieve(known_tools=lambda: KNOWN)
    text = "Math: 3 < 5 and 7 > 2, and <b>bold</b> stays."
    events = _run(sieve, text)
    assert _tags(events) == []
    assert _spoken(events) == text
