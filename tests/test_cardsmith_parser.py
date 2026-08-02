"""Unit tests for the Cardsmith streaming field-emission parser.

Covers:
- ``<commit>{json}</commit>`` block parsing (primary protocol)
- Legacy ``<set path="...">value</set>`` tags (backward compat)
- Mixed protocols within a single reply
- Streaming chunk-boundary edge cases
- ``[CARDSMITH_DONE]`` sentinel detection
- Code-fence-wrapped JSON inside commit blocks
- Case-insensitive tag matching
- Array-key (``foo[]``) expansion to multiple emissions
- Unterminated blocks at end of stream
- Multi-line content
"""

from __future__ import annotations

import json

import pytest

from augmentum.modes.narrative.cardsmith.parser import (
    DONE_SENTINEL,
    StreamingFieldParser,
    parse_field_emissions,
)

# ── <commit> block protocol (primary) ─────────────────────────────────────

def test_commit_block_basic():
    visible, emissions, done = parse_field_emissions(
        'Here you go!\n\n<commit>\n{"name": "Lyra", "imageStyle": "scifi"}\n</commit>\n\nWhat next?'
    )
    assert "Here you go!" in visible
    assert "What next?" in visible
    assert "<commit>" not in visible
    assert "{" not in visible  # raw JSON should be stripped
    paths = {e.path: e.value for e in emissions}
    assert paths == {"name": "Lyra", "imageStyle": "scifi"}
    assert done is False


def test_commit_array_path_expands_to_multiple_emissions():
    _, emissions, _ = parse_field_emissions(
        '<commit>{"tags[]": ["cyberpunk", "stoic", "noir"]}</commit>'
    )
    paths = [(e.path, e.value) for e in emissions]
    assert paths == [
        ("tags[]", "cyberpunk"),
        ("tags[]", "stoic"),
        ("tags[]", "noir"),
    ]


def test_commit_array_path_with_object_items_serializes_to_json():
    _, emissions, _ = parse_field_emissions(
        '<commit>{"lorebook[]": [{"keys": ["X"], "content": "Y"}]}</commit>'
    )
    assert len(emissions) == 1
    assert emissions[0].path == "lorebook[]"
    parsed = json.loads(emissions[0].value)
    assert parsed == {"keys": ["X"], "content": "Y"}


def test_commit_with_code_fence_wrapping_strips_fences():
    """Models sometimes wrap JSON in ```json fences despite instructions."""
    _, emissions, _ = parse_field_emissions(
        '<commit>```json\n{"name": "Lyra"}\n```</commit>'
    )
    assert len(emissions) == 1
    assert emissions[0].path == "name"
    assert emissions[0].value == "Lyra"


def test_commit_malformed_json_drops_silently():
    visible, emissions, _ = parse_field_emissions(
        'Hello <commit>{ this is not json }</commit> world'
    )
    assert "Hello" in visible
    assert "world" in visible
    assert emissions == []


def test_commit_empty_body_emits_nothing():
    _, emissions, _ = parse_field_emissions('<commit></commit>')
    assert emissions == []


def test_commit_non_object_top_level_drops_silently():
    """JSON arrays at top level aren't supported — only objects."""
    _, emissions, _ = parse_field_emissions(
        '<commit>["not", "an", "object"]</commit>'
    )
    assert emissions == []


def test_commit_null_value_is_skipped():
    _, emissions, _ = parse_field_emissions(
        '<commit>{"name": "Lyra", "imageStyle": null}</commit>'
    )
    paths = {e.path for e in emissions}
    assert paths == {"name"}


def test_multiple_commit_blocks_in_one_reply_both_processed():
    _, emissions, _ = parse_field_emissions(
        '<commit>{"name": "A"}</commit> middle <commit>{"name": "B"}</commit>'
    )
    # Both processed; "name" is committed twice (state layer handles overwrite)
    paths = [e.path for e in emissions]
    assert paths == ["name", "name"]
    assert emissions[0].value == "A"
    assert emissions[1].value == "B"


# ── Legacy <set> tag protocol ─────────────────────────────────────────────

def test_legacy_set_tag_double_quotes():
    _, emissions, _ = parse_field_emissions('<set path="name">Lyra</set>')
    assert emissions == [type(emissions[0])(path="name", value="Lyra")]


def test_legacy_set_tag_single_quotes():
    _, emissions, _ = parse_field_emissions("<set path='name'>Lyra</set>")
    assert len(emissions) == 1
    assert emissions[0].path == "name"


def test_legacy_set_tag_whitespace_around_equals():
    _, emissions, _ = parse_field_emissions('<set path = "name" >Lyra</set>')
    assert len(emissions) == 1
    assert emissions[0].path == "name"


def test_legacy_set_tag_case_insensitive():
    _, emissions, _ = parse_field_emissions('<SET PATH="name">Lyra</SET>')
    assert len(emissions) == 1


def test_legacy_set_tag_strips_from_visible():
    visible, _, _ = parse_field_emissions(
        'Hello <set path="name">Lyra</set> world'
    )
    assert visible.replace("  ", " ").strip() == "Hello world"


def test_legacy_set_tag_multiline_value():
    multiline = "Line 1\nLine 2\nLine 3"
    _, emissions, _ = parse_field_emissions(
        f'<set path="description">{multiline}</set>'
    )
    assert emissions[0].value == multiline


# ── Mixed protocols ───────────────────────────────────────────────────────

def test_mixed_set_and_commit_in_one_reply():
    _, emissions, _ = parse_field_emissions(
        '<set path="name">Lyra</set> Now: <commit>{"desc_physical": "..."}</commit>'
    )
    paths = sorted(e.path for e in emissions)
    assert paths == ["desc_physical", "name"]


# ── Streaming / chunk boundaries ──────────────────────────────────────────

def test_streaming_split_across_commit_opener():
    parser = StreamingFieldParser()
    chunks = ["Hi! <com", "mit>", '{"name":', ' "Lyra"}', "</commit>", " bye"]
    all_visible = []
    all_emissions = []
    for c in chunks:
        step = parser.feed(c)
        all_visible.append(step.visible)
        all_emissions.extend(step.emissions)
    tail = parser.flush()
    all_visible.append(tail.visible)
    visible = "".join(all_visible)
    assert "Hi!" in visible
    assert "bye" in visible
    assert "<commit" not in visible
    assert len(all_emissions) == 1
    assert all_emissions[0].path == "name"


def test_streaming_split_across_set_closer():
    parser = StreamingFieldParser()
    chunks = ['<set path="name">Lyra</se', "t>"]
    emissions = []
    for c in chunks:
        emissions.extend(parser.feed(c).emissions)
    tail = parser.flush()
    emissions.extend(tail.emissions)
    assert len(emissions) == 1
    assert emissions[0].path == "name"
    assert emissions[0].value == "Lyra"


def test_streaming_split_across_done_sentinel():
    parser = StreamingFieldParser()
    chunks = ["bye [CARD", "SMITH", "_DONE]"]
    visible = []
    done = False
    for c in chunks:
        step = parser.feed(c)
        visible.append(step.visible)
        if step.done:
            done = True
    assert done is True
    assert "[CARDSMITH_DONE]" not in "".join(visible)
    assert "bye" in "".join(visible)


def test_streaming_holds_back_potential_opener_prefix():
    parser = StreamingFieldParser()
    # A chunk ending in "<co" could be the start of "<commit>" — should be held back.
    step = parser.feed("hello <co")
    assert "<co" not in step.visible
    # Next chunk completes it
    step2 = parser.feed("mmit>{}</commit> done")
    # Empty commit emits nothing but should clear state
    final = parser.flush()
    full_visible = step.visible + step2.visible + final.visible
    assert "<commit" not in full_visible
    assert "hello" in full_visible
    assert "done" in full_visible


def test_streaming_long_set_opener_split_mid_attribute_does_not_leak():
    """Regression: a `<set path="postHistoryInstructions">` opener split
    mid-attribute used to leak protocol bytes (`<set path="postHist`) into
    the visible stream and lose the field emission. The earlier fix used a
    fixed 16-char holdback that wasn't large enough for path attributes
    longer than the cap; the parser now scans for partial-marker prefixes
    instead.
    """
    parser = StreamingFieldParser()
    step1 = parser.feed('Hello world. <set path="postHistoryInstructions"')
    step2 = parser.feed('>my val</set>')
    final = parser.flush()
    visible = step1.visible + step2.visible + final.visible
    assert "<set" not in visible
    assert visible.strip() == "Hello world."
    emissions = step1.emissions + step2.emissions + final.emissions
    assert len(emissions) == 1
    assert emissions[0].path == "postHistoryInstructions"
    assert emissions[0].value == "my val"


def test_streaming_non_opener_lessthan_flushes_normally():
    """A bare `<` followed by content that can't become an opener (e.g. `<3`,
    `<setiquette` — diverges from `<set ` because no whitespace follows) must
    not be held back forever. Otherwise prose containing `<` would stall.
    """
    parser = StreamingFieldParser()
    step = parser.feed("Less than <3 sign and <setiquette is a typo")
    final = parser.flush()
    visible = step.visible + final.visible
    assert visible == "Less than <3 sign and <setiquette is a typo"


def test_streaming_partial_opener_at_eos_is_stripped():
    """If the model abruptly ends with an unfinished `<set path="...` opener,
    the partial bytes must not leak to the user at flush time.
    """
    parser = StreamingFieldParser()
    step = parser.feed('cut off mid-tag <set path="hu')
    final = parser.flush()
    visible = step.visible + final.visible
    assert "<set" not in visible
    assert visible.strip() == "cut off mid-tag"


# ── Sentinel handling ─────────────────────────────────────────────────────

def test_sentinel_at_end_sets_done():
    visible, _, done = parse_field_emissions(f'all done {DONE_SENTINEL}')
    assert done is True
    assert "all done" in visible
    assert DONE_SENTINEL not in visible


def test_sentinel_before_commit_block_still_works():
    visible, emissions, done = parse_field_emissions(
        f'{DONE_SENTINEL}\n<commit>{{"name": "Lyra"}}</commit>'
    )
    # Once the parser sees DONE, it stops processing further chunks in a real
    # streaming context. parse_field_emissions is one-shot via feed+flush, so
    # done flips and remaining text after DONE is discarded.
    assert done is True


def test_done_after_first_commit_short_circuits_subsequent_input():
    parser = StreamingFieldParser()
    parser.feed(f"hi {DONE_SENTINEL}")
    # Subsequent feed should be a no-op
    step = parser.feed("more text after done")
    assert step.visible == ""
    assert step.done is True


# ── Unterminated / malformed ──────────────────────────────────────────────

def test_unterminated_set_block_dropped_on_flush():
    """A <set> tag without a closing </set> at end-of-stream is silently dropped."""
    parser = StreamingFieldParser()
    parser.feed('<set path="name">Lyra without closer')
    tail = parser.flush()
    assert tail.emissions == []


def test_unterminated_commit_block_dropped_on_flush():
    parser = StreamingFieldParser()
    parser.feed('<commit>{"name": "Lyra"')  # no </commit>
    tail = parser.flush()
    assert tail.emissions == []


# ── Convenience function consistency ──────────────────────────────────────

def test_one_shot_matches_streamed_chunked_result():
    """parse_field_emissions should match feed-then-flush behavior."""
    text = 'Hello <commit>{"name": "Lyra"}</commit> world'

    one_shot = parse_field_emissions(text)

    parser = StreamingFieldParser()
    chunks_visible = []
    chunks_emissions = []
    for chunk in [text[i:i + 5] for i in range(0, len(text), 5)]:
        step = parser.feed(chunk)
        chunks_visible.append(step.visible)
        chunks_emissions.extend(step.emissions)
    tail = parser.flush()
    chunks_visible.append(tail.visible)
    chunks_emissions.extend(tail.emissions)

    assert "".join(chunks_visible) == one_shot[0]
    assert chunks_emissions == one_shot[1]


# ── Regression: name-only-persists bug ────────────────────────────────────

def test_long_conversation_with_multiple_commits_all_land():
    """Regression: earlier versions only persisted the first emitted field."""
    parser = StreamingFieldParser()
    turns = [
        '<commit>{"name": "Lyra"}</commit>',
        'OK so: <commit>{"desc_physical": "para 1+2", "visualTraits": "tags"}</commit>',
        'And now <commit>{"desc_personality": "para 3", "personality": "short"}</commit>',
        '<commit>{"desc_depth": "para 4+5"}</commit>',
        '<commit>{"scenario": "scene", "greeting": "hi"}</commit>',
    ]
    all_emissions = []
    for t in turns:
        all_emissions.extend(parser.feed(t).emissions)
    parser.flush()

    paths = sorted(e.path for e in all_emissions)
    assert "name" in paths
    assert "desc_physical" in paths
    assert "desc_personality" in paths
    assert "desc_depth" in paths
    assert "personality" in paths
    assert "visualTraits" in paths
    assert "scenario" in paths
    assert "greeting" in paths


# ── pytest entry sanity ───────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
