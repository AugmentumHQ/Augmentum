"""Tag grammar — dotted verb names MUST parse.

Pins the 2026-06-10 act-gap root cause: TAG_RE's name charset was
``[a-z_]+`` so every dotted intent-registry verb (grove.play_matching,
media.play, navigate.open_surface) was unparseable. The model emitted
exactly what the roster taught it and the sieve dropped the tag as
plain text; the TTS scrubber then ate the debris, so the failure was
invisible end-to-end.
"""

from __future__ import annotations

from augmentum.companion_runtime.tool_protocol import TagSieve, scan


def test_dotted_tool_name_parses():
    # The literal turn-1 shape from the 2026-06-10 voice session.
    tags = scan('<tool:grove.play_matching query="yuzu" />')
    assert len(tags) == 1
    assert tags[0].name == "grove.play_matching"
    assert tags[0].args == {"query": "yuzu"}


def test_dotted_names_across_registry_verbs():
    for name in (
        "media.play", "navigate.open_surface", "note.create",
        "companion.today_recap", "image.generate_with_defaults",
        "search.local",
    ):
        tags = scan(f"<tool:{name} query='x' />")
        assert len(tags) == 1, name
        assert tags[0].name == name


def test_single_and_double_quotes_both_parse():
    # The roster teaches single quotes for registry verbs and double
    # quotes for catalog tools — the sieve must accept both.
    for raw in (
        '<tool:media.play query="dune" />',
        "<tool:media.play query='dune' />",
    ):
        tags = scan(raw)
        assert len(tags) == 1
        assert tags[0].args == {"query": "dune"}


def test_legacy_underscore_names_still_parse():
    tags = scan('<tool:files_read path="/tmp/x" />')
    assert len(tags) == 1
    assert tags[0].name == "files_read"


def test_handoff_tags_unaffected():
    tags = scan('<handoff:coder reason="refactor" brief="cleanup" />')
    assert len(tags) == 1
    assert tags[0].kind == "handoff"
    assert tags[0].name == "coder"


def test_sieve_catches_dotted_tag_mid_stream():
    sieve = TagSieve()
    chunks = ["Let me put that on. ", "<tool:grove.play_", 'matching query="yuzu" />']
    seen_tag = None
    clean = []
    for chunk in chunks:
        for text, tag in sieve.feed(chunk):
            if text:
                clean.append(text)
            if tag is not None:
                seen_tag = tag
    assert seen_tag is not None
    assert seen_tag.name == "grove.play_matching"
    assert seen_tag.args == {"query": "yuzu"}
    # Pre-tag prose is preserved for the promise capture.
    assert "Let me put that on." in "".join(clean) + sieve.flush()


def test_numeric_arg_names_parse():
    tags = scan('<tool:media.play query="dune" k1="v1" />')
    assert len(tags) == 1
    assert tags[0].args == {"query": "dune", "k1": "v1"}


def test_plain_prose_with_angle_bracket_is_not_a_tag():
    assert scan("I think 3 < 5 is true, and x->y too") == []


# ── Salvage pass — registry-validated mangled-prefix recovery ─────────
# Observed live 2026-06-10: Qwen 3.6 emitted ``<j:...`` instead of
# ``<tool:...`` (likely <tool colliding with its native <tool_call>
# special tokens). Right verb, clean args, garbled prefix.

KNOWN = {
    "grove.play_matching", "media.play", "web.search", "note.create",
    "recall", "browse",
}


def _salvage_sieve():
    return TagSieve(known_tools=lambda: KNOWN)


def test_salvage_literal_jazz_tag_from_prod():
    sieve = _salvage_sieve()
    tag = None
    for _text, t in sieve.feed("<j:play_matching query='smooth jazz' />"):
        if t:
            tag = t
    assert tag is not None
    assert tag.name == "grove.play_matching"   # unique suffix resolution
    assert tag.args == {"query": "smooth jazz"}


def test_salvage_literal_news_tag_from_prod():
    sieve = _salvage_sieve()
    tag = None
    for _text, t in sieve.feed("<j:web.search query='latest US news headlines' />"):
        if t:
            tag = t
    assert tag is not None
    assert tag.name == "web.search"
    assert tag.args == {"query": "latest US news headlines"}


def test_salvage_requires_registry_match():
    sieve = _salvage_sieve()
    tags = [t for _x, t in sieve.feed("<j:rm_rf path='/' />") if t]
    assert tags == []          # unknown verb → stays text
    assert "<j:rm_rf" in sieve.flush()


def test_salvage_ambiguous_suffix_stays_text():
    sieve = TagSieve(known_tools=lambda: {"a.play", "b.play"})
    tags = [t for _x, t in sieve.feed("<j:play query='x' />") if t]
    assert tags == []          # two suffix candidates → refuse to guess


def test_salvage_ignores_html_shaped_text():
    sieve = _salvage_sieve()
    tags = [t for _x, t in sieve.feed("use <br /> and <i>this</i> like 3 < 5") if t]
    assert tags == []


def test_salvage_disabled_without_known_tools():
    sieve = TagSieve()
    tags = [t for _x, t in sieve.feed("<j:play_matching query='jazz' />") if t]
    assert tags == []


def test_strict_grammar_still_wins_over_salvage():
    sieve = _salvage_sieve()
    tag = None
    for _text, t in sieve.feed('<tool:media.play query="dune" />'):
        if t:
            tag = t
    assert tag is not None
    assert tag.kind == "tool"
    assert tag.name == "media.play"


# ── Tier 3: fuzzy call recovery (act-gated, format-tolerant) ──────────

from augmentum.companion_runtime.tool_protocol import recover_loose_call


def test_recover_missing_self_close():
    # Strict + salvage both need '/>'; recovery doesn't.
    c = recover_loose_call('<tool: grove.play_matching query="jazz">', KNOWN)
    assert c is not None and c.name == "grove.play_matching"
    assert c.args == {"query": "jazz"}


def test_recover_bare_no_brackets():
    c = recover_loose_call("tool:grove.play_matching query='smooth jazz'", KNOWN)
    assert c is not None and c.name == "grove.play_matching"
    assert c.args["query"] == "smooth jazz"


def test_recover_typo_in_name():
    # "play_maching" — 92% similar to play_matching.
    c = recover_loose_call("<j:play_maching query='jazz' />", KNOWN)
    assert c is not None and c.name == "grove.play_matching"


def test_recover_unquoted_args():
    c = recover_loose_call("web.search query=jazz limit=5", KNOWN)
    assert c is not None and c.name == "web.search"
    assert c.args == {"query": "jazz", "limit": "5"}


def test_recover_prose_mention_without_args_stays_none():
    assert recover_loose_call("let me web.search it for you", KNOWN) is None


def test_recover_single_words_never_fuzzy():
    # 'recall' is a known single-word tool; 'recalls' in prose must NOT
    # fuzzy-resolve, and even exact 'recall' needs args to fire.
    assert recover_loose_call("she recalls the query=thing fondly", KNOWN) is None
    assert recover_loose_call("I recall that day", KNOWN) is None
    c = recover_loose_call("recall query='our trip plans'", KNOWN)
    assert c is not None and c.name == "recall"


def test_recover_unknown_verb_stays_none():
    assert recover_loose_call("rm_rf path='/' now=yes", KNOWN) is None


def test_recover_empty_inputs():
    assert recover_loose_call("", KNOWN) is None
    assert recover_loose_call("web.search query=x", set()) is None


# ── Loose-call quote/window fixes + sieve hold (2026-06-11) ──────────
# Live failure: model emitted a BARE "note.append content='Becca's
# capabilities include: …'" — no wrapper, so the whole call streamed to
# TTS, and the old quote regex closed at the apostrophe in "Becca's",
# appending exactly one word to the note.

_KNOWN_NOTES = KNOWN | {"note.append"}

_LIVE_EMISSION = (
    "note.append content='Becca's capabilities include:\n"
    "- Technical tasks: Running Python code, managing files, searching memory\n"
    "- Creative tasks: Generating images, creating notes'"
)


def test_recover_apostrophe_inside_single_quoted_value():
    c = recover_loose_call(_LIVE_EMISSION, _KNOWN_NOTES)
    assert c is not None and c.name == "note.append"
    assert c.args["content"].startswith("Becca's capabilities include:")
    assert c.args["content"].endswith("creating notes")
    assert "Running Python code" in c.args["content"]


def test_recover_unquoted_multiword_value_runs_to_end():
    c = recover_loose_call("note.append content=grocery list with milk and eggs",
                           _KNOWN_NOTES)
    assert c is not None
    assert c.args["content"] == "grocery list with milk and eggs"


def test_recover_long_value_not_window_truncated():
    body = "line of useful note content. " * 30  # ~900 chars, > old 240 window
    c = recover_loose_call(f"note.append content='{body.strip()}'", _KNOWN_NOTES)
    assert c is not None
    assert c.args["content"] == body.strip()


def test_recover_multiple_args_after_quoted_value():
    c = recover_loose_call(
        "note.append note_id='n1' content='Becca's list: a, b' ",
        _KNOWN_NOTES,
    )
    assert c is not None
    assert c.args["note_id"] == "n1"
    assert c.args["content"] == "Becca's list: a, b"


def test_recover_span_covers_full_call():
    text = "Sure, adding that now. " + _LIVE_EMISSION
    c = recover_loose_call(text, _KNOWN_NOTES)
    assert c is not None
    start, end = c.span
    assert text[start:end].startswith("note.append")
    assert text[start:end].endswith("creating notes'")


def _loose_sieve(enabled=True):
    return TagSieve(known_tools=lambda: _KNOWN_NOTES,
                    allow_loose=lambda: enabled)


def _run_sieve(sieve, text, chunk=7):
    spoken, tags = [], []
    for i in range(0, len(text), chunk):
        for clean, tag in sieve.feed(text[i:i + chunk]):
            if clean:
                spoken.append(clean)
            if tag is not None:
                tags.append(tag)
    for clean, tag in sieve.drain():
        if clean:
            spoken.append(clean)
        if tag is not None:
            tags.append(tag)
    return "".join(spoken), tags


def test_sieve_holds_bare_call_from_tts_and_recovers_at_drain():
    text = "On it. " + _LIVE_EMISSION
    spoken, tags = _run_sieve(_loose_sieve(), text)
    assert len(tags) == 1
    assert tags[0].name == "note.append"
    assert tags[0].args["content"].endswith("creating notes")
    # The call body must never reach the speakable stream.
    assert "note.append" not in spoken
    assert "content=" not in spoken
    assert "Running Python" not in spoken
    assert spoken.strip() == "On it."


def test_sieve_loose_disabled_passes_text_through():
    # Conversational turn (not act-classified): prose mentioning a verb
    # id stays prose — spoken, never dispatched.
    text = "you could try note.append content=something to do it"
    spoken, tags = _run_sieve(_loose_sieve(enabled=False), text)
    assert tags == []
    assert spoken == text


def test_sieve_loose_prose_mention_without_args_stays_spoken():
    text = "the note.append verb is how I write things down for you"
    spoken, tags = _run_sieve(_loose_sieve(), text)
    assert tags == []
    assert spoken == text


def test_sieve_strict_tags_still_win_with_loose_enabled():
    spoken, tags = _run_sieve(
        _loose_sieve(), "sure <tool:media.play query=\"dune\" /> done",
    )
    assert len(tags) == 1
    assert tags[0].name == "media.play"
    assert tags[0].args == {"query": "dune"}
