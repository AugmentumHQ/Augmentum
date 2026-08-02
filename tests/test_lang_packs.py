"""lang_packs lookup tests — build a tiny pack, exercise the query path."""

from __future__ import annotations

import json

import aiosqlite
import pytest

from augmentum.knowledge.lang_pack_builder import build_pack, build_pack_wiktionary
from augmentum.learning import lang_packs

_JMDICT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [
<!ENTITY v1 "Ichidan verb">
<!ENTITY vt "transitive verb">
<!ENTITY n "noun">
<!ENTITY int "interjection">
]>
<JMdict>
<entry>
<ent_seq>1358280</ent_seq>
<k_ele><keb>食べる</keb></k_ele>
<r_ele><reb>たべる</reb></r_ele>
<sense><pos>&v1;</pos><pos>&vt;</pos><gloss>to eat</gloss><gloss>to live on</gloss></sense>
</entry>
<entry>
<ent_seq>1202450</ent_seq>
<k_ele><keb>食</keb></k_ele>
<r_ele><reb>しょく</reb></r_ele>
<sense><pos>&n;</pos><gloss>meal</gloss><gloss>food</gloss></sense>
</entry>
<entry>
<ent_seq>1578850</ent_seq>
<k_ele><keb>朝ごはん</keb></k_ele>
<r_ele><reb>あさごはん</reb></r_ele>
<sense><pos>&n;</pos><gloss>breakfast</gloss></sense>
</entry>
<entry>
<ent_seq>1000000</ent_seq>
<r_ele><reb>ありがとう</reb></r_ele>
<sense><pos>&int;</pos><gloss>thank you</gloss></sense>
</entry>
</JMdict>
"""

_SENTENCES_TSV = "\n".join([
    "1\tjpn\t彼は朝ごはんを食べる。",
    "2\teng\tHe eats breakfast.",
    "3\tjpn\t毎日ご飯を食べる。",
    "4\tjpn\tもっと食べる時間がない長い文章をここに書いておく。",
])
_LINKS_TSV = "1\t2\n2\t1\n"


@pytest.fixture
async def pack_conn(tmp_path):
    jm = tmp_path / "JMdict_e"
    jm.write_text(_JMDICT_XML, encoding="utf-8")
    s = tmp_path / "sentences.tsv"
    s.write_text(_SENTENCES_TSV, encoding="utf-8")
    ln = tmp_path / "links.tsv"
    ln.write_text(_LINKS_TSV, encoding="utf-8")
    out = tmp_path / "ja.augpack"
    build_pack(out_path=out, lang_code="ja", jmdict_xml=jm,
               tatoeba_sentences=s, tatoeba_links=ln)
    conn = await aiosqlite.connect(str(out))
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_longest_prefix_picks_the_verb(pack_conn):
    # Clicking 食 in "食べる" must resolve to 食べる, not 食.
    hits = await lang_packs.lookup_at(pack_conn, "食べるのが好き", 0)
    assert [h["word_id"] for h in hits] == ["1358280"]
    assert hits[0]["surface"] == "食べる"
    assert hits[0]["pos"] == "v1,vt"
    assert hits[0]["glosses"] == ["to eat", "to live on"]


@pytest.mark.asyncio
async def test_longest_prefix_mid_string(pack_conn):
    text = "彼は朝ごはんを食べる。"
    # Click on 朝 (index 2) → 朝ごはん.
    hits = await lang_packs.lookup_at(pack_conn, text, 2)
    assert [h["word_id"] for h in hits] == ["1578850"]
    # Click on 食 (index 7) → 食べる.
    hits = await lang_packs.lookup_at(pack_conn, text, 7)
    assert [h["word_id"] for h in hits] == ["1358280"]


@pytest.mark.asyncio
async def test_lookup_at_kana_reading(pack_conn):
    # Kana-only headword resolved via the reading column.
    hits = await lang_packs.lookup_at(pack_conn, "ありがとうございます", 0)
    assert [h["word_id"] for h in hits] == ["1000000"]
    assert hits[0]["surface"] == "ありがとう"


@pytest.mark.asyncio
async def test_lookup_at_no_match_and_bounds(pack_conn):
    assert await lang_packs.lookup_at(pack_conn, "XYZ", 0) == []
    assert await lang_packs.lookup_at(pack_conn, "食べる", -1) == []
    assert await lang_packs.lookup_at(pack_conn, "食べる", 99) == []
    assert await lang_packs.lookup_at(pack_conn, "", 0) == []


@pytest.mark.asyncio
async def test_lookup_text_exact_and_fts(pack_conn):
    # Exact surface.
    hits = await lang_packs.lookup_text(pack_conn, "食べる")
    assert "1358280" in {h["word_id"] for h in hits}
    # English gloss via FTS.
    hits = await lang_packs.lookup_text(pack_conn, "breakfast")
    assert "1578850" in {h["word_id"] for h in hits}
    # Reading.
    hits = await lang_packs.lookup_text(pack_conn, "たべる")
    assert "1358280" in {h["word_id"] for h in hits}
    assert await lang_packs.lookup_text(pack_conn, "") == []


@pytest.mark.asyncio
async def test_get_entry(pack_conn):
    e = await lang_packs.get_entry(pack_conn, "1358280")
    assert e is not None and e["surface"] == "食べる"
    assert await lang_packs.get_entry(pack_conn, "0000000") is None
    assert await lang_packs.get_entry(pack_conn, "") is None


@pytest.mark.asyncio
async def test_get_example(pack_conn):
    ex = await lang_packs.get_example(pack_conn, "食べる")
    assert ex is not None
    # Easiest (shortest) sentence containing 食べる wins.
    assert ex["lang_text"] == "毎日ご飯を食べる。"
    assert ex["en_text"] is None
    # Require an English translation: falls through to the linked one.
    ex = await lang_packs.get_example(pack_conn, "朝ごはん", en_required=True)
    assert ex is not None
    assert ex["lang_text"] == "彼は朝ごはんを食べる。"
    assert ex["en_text"] == "He eats breakfast."
    assert await lang_packs.get_example(pack_conn, "存在しない") is None
    assert await lang_packs.get_example(pack_conn, "") is None


@pytest.mark.asyncio
async def test_latin_examples_require_word_boundaries():
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            """CREATE TABLE sentences (
                sent_id INTEGER PRIMARY KEY,
                lang_text TEXT NOT NULL,
                en_text TEXT,
                difficulty INTEGER
            )"""
        )
        await conn.executemany(
            "INSERT INTO sentences (lang_text, en_text, difficulty) VALUES (?, ?, ?)",
            [
                ("La casa es grande.", "The house is big.", 1),
                ("Voy a casa.", "I am going home.", 2),
            ],
        )
        await conn.commit()

        ex = await lang_packs.get_example(conn, "a")
        assert ex is not None
        assert ex["lang_text"] == "Voy a casa."

        narrowed = await lang_packs.read_sentences(
            conn, n=10, max_difficulty=4, contains="a", require_translation=False,
        )
        assert [s["lang_text"] for s in narrowed] == ["Voy a casa."]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_latin_examples_do_not_fallback_to_substrings():
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute(
            """CREATE TABLE sentences (
                sent_id INTEGER PRIMARY KEY,
                lang_text TEXT NOT NULL,
                en_text TEXT,
                difficulty INTEGER
            )"""
        )
        await conn.execute(
            "INSERT INTO sentences (lang_text, en_text, difficulty) VALUES (?, ?, ?)",
            ("La casa es grande.", "The house is big.", 1),
        )
        await conn.commit()

        assert await lang_packs.get_example(conn, "a") is None
        narrowed = await lang_packs.read_sentences(
            conn, n=10, max_difficulty=4, contains="a", require_translation=False,
        )
        assert narrowed == []
    finally:
        await conn.close()


def test_gameworthy_surface_filters_single_latin_letters():
    assert not lang_packs.is_gameworthy_surface("a")
    assert not lang_packs.is_gameworthy_surface("é")
    assert lang_packs.is_gameworthy_surface("casa")
    assert lang_packs.is_gameworthy_surface("食")


@pytest.mark.asyncio
async def test_pack_meta(pack_conn):
    meta = await lang_packs.pack_meta(pack_conn)
    assert meta["pack_kind"] == "language"
    assert meta["lang_code"] == "ja"


@pytest.mark.asyncio
async def test_pack_pos_labels(pack_conn):
    labels = await lang_packs.pack_pos_labels(pack_conn)
    # Built pack ships the JMdict map. Sample a few core entries.
    assert labels.get("v1") == "Ichidan verb"
    assert labels.get("n") == "noun"
    assert labels.get("adj-i") == "i-adjective"


@pytest.mark.asyncio
async def test_pack_tokenization(pack_conn):
    # ja packs use longest-prefix segmentation (no word spaces in JP).
    assert await lang_packs.pack_tokenization(pack_conn) == "longest_prefix"


@pytest.mark.asyncio
async def test_tokenize_segment(pack_conn):
    toks = await lang_packs.tokenize_segment(pack_conn, "朝ごはんを食べる")
    matched = [t for t in toks if t.get("matched")]
    assert [t["word_id"] for t in matched] == ["1578850", "1358280"]
    assert [t["text"] for t in matched] == ["朝ごはん", "食べる"]
    raws = [t for t in toks if not t.get("matched")]
    assert [t["text"] for t in raws] == ["を"]
    # all-unknown span → all raw, full text preserved.
    junk = await lang_packs.tokenize_segment(pack_conn, "XYZ")
    assert all(not t.get("matched") for t in junk)
    assert "".join(t["text"] for t in junk) == "XYZ"
    assert await lang_packs.tokenize_segment(pack_conn, "") == []


# ── Whitespace tokenizer (space-delimited languages) ─────────────────

_KAIKKI_ES_FIXTURE = "\n".join([
    json.dumps({"word": "casa", "pos": "noun", "senses": [{"glosses": ["house"]}]}),
    json.dumps({"word": "es", "pos": "verb", "senses": [{"glosses": ["is"]}]}),
    json.dumps({"word": "grande", "pos": "adj", "senses": [{"glosses": ["big"]}]}),
])


@pytest.fixture
async def es_pack_conn(tmp_path):
    jsonl = tmp_path / "es.jsonl"
    jsonl.write_text(_KAIKKI_ES_FIXTURE, encoding="utf-8")
    out = tmp_path / "es.augpack"
    build_pack_wiktionary(out_path=out, lang_code="es", wiktionary_jsonl=jsonl)
    conn = await aiosqlite.connect(str(out))
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_es_pack_tokenization_is_whitespace(es_pack_conn):
    # The whitespace mode is recorded in pack meta so tokenize_segment
    # dispatches correctly without the caller needing to know.
    assert await lang_packs.pack_tokenization(es_pack_conn) == "whitespace"


@pytest.mark.asyncio
async def test_tokenize_segment_whitespace_basic(es_pack_conn):
    toks = await lang_packs.tokenize_segment(es_pack_conn, "La casa es grande.")
    # Words land as matched (when in dict) or raw (e.g. "La"); spaces
    # and punctuation are preserved as raw tokens so the UI can re-render.
    matched_surfaces = [t["text"] for t in toks if t.get("matched")]
    assert "casa" in matched_surfaces
    assert "es" in matched_surfaces
    assert "grande" in matched_surfaces
    # The original text is preserved verbatim when reconstructed.
    assert "".join(t["text"] for t in toks) == "La casa es grande."


@pytest.mark.asyncio
async def test_tokenize_segment_whitespace_case_fallback(es_pack_conn):
    # Spanish dictionaries are lower-case; an uppercased clicked word
    # must still resolve via the lowercase fallback path.
    toks = await lang_packs.tokenize_segment(es_pack_conn, "Casa.")
    casa = [t for t in toks if t["text"] == "Casa"]
    assert casa and casa[0].get("matched") is True


@pytest.mark.asyncio
async def test_lookup_surfaces_case_fallback(es_pack_conn):
    # The path-seeder feeds in lowercase path surfaces; if the pack
    # stored an uppercased variant they'd silently drop without the
    # second-pass lowercase retry. Mirrors the click-time fallback.
    hits = await lang_packs.lookup_surfaces(
        es_pack_conn, ["Casa", "ES", "Grande", "missing"],
    )
    # Order-preserving: every input that resolves (case-insensitively)
    # returns its word_id in input order; misses (`missing`) drop.
    assert hits == ["casa", "es", "grande"]


@pytest.mark.asyncio
async def test_lookup_surfaces_exact_case_still_works(es_pack_conn):
    # Lowercase surfaces hit on the first (exact-case) pass — no
    # second-pass round-trip needed. Asserts the happy path didn't
    # regress when we added the case-fallback.
    hits = await lang_packs.lookup_surfaces(es_pack_conn, ["casa", "grande"])
    assert hits == ["casa", "grande"]


@pytest.mark.asyncio
async def test_tokenize_segment_mode_override(pack_conn):
    # Caller can explicitly request a tokenisation mode. Whitespace mode
    # on JP text (no spaces) treats the whole \w+ run as one token —
    # which only matches if the entire string is a single dictionary
    # surface. Demonstrates the dispatch is honoured: a multi-word JP
    # phrase that longest-prefix segments into 3 tokens collapses to
    # one unmatched token under whitespace mode.
    phrase = "朝ごはんを食べる"
    ws_toks = await lang_packs.tokenize_segment(pack_conn, phrase, mode="whitespace")
    lp_toks = await lang_packs.tokenize_segment(pack_conn, phrase, mode="longest_prefix")
    assert len(ws_toks) == 1 and not ws_toks[0].get("matched")
    assert len([t for t in lp_toks if t.get("matched")]) == 2


@pytest.mark.asyncio
async def test_read_sentences(pack_conn):
    batch = await lang_packs.read_sentences(pack_conn, n=10, max_difficulty=4)
    # Default require_translation=True → every returned sentence has en_text.
    assert all(s.get("en_text") for s in batch)
    assert all({"sent_id", "lang_text", "en_text", "difficulty"} <= set(s) for s in batch)
    # contains-filter narrows to sentences holding that substring.
    narrowed = await lang_packs.read_sentences(pack_conn, n=10, max_difficulty=4, contains="食べる")
    assert all("食べる" in s["lang_text"] for s in narrowed)
    # require_translation=False also surfaces the untranslated JP sentences.
    all_jp = await lang_packs.read_sentences(pack_conn, n=10, max_difficulty=9, require_translation=False)
    assert len(all_jp) >= len(batch)
    # difficulty cap filters out long sentences.
    short = await lang_packs.read_sentences(pack_conn, n=10, max_difficulty=1, require_translation=False)
    assert all(s["difficulty"] <= 1 for s in short)


@pytest.mark.asyncio
async def test_top_frequency(pack_conn):
    top = await lang_packs.top_frequency(pack_conn, 30)
    # Only 食べる and 朝ごはん occur in the test corpus → only they carry
    # a freq_rank; 食べる is the most frequent so it ranks first.
    assert [e["word_id"] for e in top] == ["1358280", "1578850"]
    assert top[0]["surface"] == "食べる"
    # The limit is honoured.
    assert len(await lang_packs.top_frequency(pack_conn, 1)) == 1
    assert [e["word_id"] for e in await lang_packs.top_frequency(pack_conn, 1)] == ["1358280"]
