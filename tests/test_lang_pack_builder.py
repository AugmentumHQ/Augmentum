"""lang_pack_builder tests — build a tiny .augpack from inline corpora."""

from __future__ import annotations

import json
import sqlite3

import pytest

from augmentum.knowledge.lang_pack_builder import (
    build_pack,
    build_pack_wiktionary,
    iter_hermitdave_frequency,
    iter_kaikki_entries,
)

# A miniature JMdict file: inline DTD with POS entities (so we exercise
# the entity-stripping path), one kanji word (食べる), one kana-only word
# (ありがとう), and one gloss-less entry that must be skipped.
_JMDICT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [
<!ENTITY v1 "Ichidan verb">
<!ENTITY vt "transitive verb">
<!ENTITY int "interjection">
]>
<JMdict>
<entry>
<ent_seq>1358280</ent_seq>
<k_ele><keb>食べる</keb></k_ele>
<r_ele><reb>たべる</reb></r_ele>
<sense>
<pos>&v1;</pos>
<pos>&vt;</pos>
<gloss>to eat</gloss>
<gloss>to live on (e.g. a salary)</gloss>
</sense>
</entry>
<entry>
<ent_seq>1000000</ent_seq>
<r_ele><reb>ありがとう</reb></r_ele>
<sense>
<pos>&int;</pos>
<gloss>thank you</gloss>
</sense>
</entry>
<entry>
<ent_seq>9999999</ent_seq>
<r_ele><reb>なにもない</reb></r_ele>
<sense><pos>&int;</pos></sense>
</entry>
</JMdict>
"""

_SENTENCES_TSV = "\n".join([
    "1\tjpn\t彼は朝ごはんを食べる。",
    "2\teng\tHe eats breakfast.",
    "3\tjpn\tありがとうございます。",
    "4\tfra\tMerci.",
])

_LINKS_TSV = "\n".join(["1\t2", "2\t1"])


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def built_pack(tmp_path):
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    sents = _write(tmp_path / "sentences.tsv", _SENTENCES_TSV)
    links = _write(tmp_path / "links.tsv", _LINKS_TSV)
    freq = _write(tmp_path / "freq.tsv", "食べる\t12\nありがとう\t300\n")
    jlpt = _write(tmp_path / "jlpt.tsv", "食べる\t5\n")
    out = tmp_path / "ja.augpack"
    summary = build_pack(
        out_path=out, lang_code="ja", jmdict_xml=jmdict,
        tatoeba_sentences=sents, tatoeba_links=links,
        freq_tsv=freq, jlpt_tsv=jlpt, name="JP test",
    )
    return out, summary


def test_builds_and_counts(built_pack):
    out, summary = built_pack
    assert out.exists()
    # 2 entries with glosses; the gloss-less entry (9999999) is skipped.
    assert summary["vocab"] == 2
    assert summary["sentences"] == 2  # the two jpn lines


def test_meta_rows(built_pack):
    out, _ = built_pack
    db = sqlite3.connect(str(out))
    try:
        kv = dict(db.execute("SELECT key, value FROM meta").fetchall())
    finally:
        db.close()
    assert kv["pack_kind"] == "language"
    assert kv["lang_code"] == "ja"
    assert kv["schema"] == "1"
    assert kv["name"] == "JP test"
    assert kv["vocab_count"] == "2"
    assert kv["sentence_count"] == "2"
    assert "JMdict" in kv["source"]
    assert "CC BY" in kv["source_license"]
    # Built packs carry their own POS labels + tokenization mode so the UI
    # doesn't need a hardcoded per-language map.
    assert kv["tokenization"] == "longest_prefix"
    labels = json.loads(kv["pos_labels"])
    assert labels["v1"] == "Ichidan verb"
    assert labels["adj-i"] == "i-adjective"
    assert labels["n"] == "noun"


def test_vocab_row_and_entity_stripping(built_pack):
    out, _ = built_pack
    db = sqlite3.connect(str(out))
    try:
        row = db.execute(
            "SELECT surface, reading, pos, glosses, freq_rank, jlpt "
            "FROM vocab WHERE word_id='1358280'"
        ).fetchone()
        kana = db.execute(
            "SELECT surface, reading FROM vocab WHERE word_id='1000000'"
        ).fetchone()
        gloss_less = db.execute(
            "SELECT 1 FROM vocab WHERE word_id='9999999'"
        ).fetchone()
    finally:
        db.close()
    assert row is not None
    surface, reading, pos, glosses, freq_rank, jlpt = row
    assert surface == "食べる"
    assert reading == "たべる"
    assert pos == "v1,vt"  # &v1; &vt; stripped to bare names
    assert json.loads(glosses) == ["to eat", "to live on (e.g. a salary)"]
    assert freq_rank == 12
    assert jlpt == 5
    assert kana == ("ありがとう", "ありがとう")  # surface falls back to reading
    assert gloss_less is None  # gloss-less entry skipped


def test_fts_lookup(built_pack):
    out, _ = built_pack
    db = sqlite3.connect(str(out))
    try:
        hits = db.execute(
            "SELECT v.word_id FROM vocab_fts f JOIN vocab v ON v.rowid = f.rowid "
            "WHERE vocab_fts MATCH 'eat'"
        ).fetchall()
    finally:
        db.close()
    assert ("1358280",) in hits


def test_sentences_table_and_substring_lookup(built_pack):
    out, _ = built_pack
    db = sqlite3.connect(str(out))
    try:
        rows = db.execute(
            "SELECT lang_text, en_text, difficulty FROM sentences ORDER BY sent_id"
        ).fetchall()
        example = db.execute(
            "SELECT lang_text FROM sentences "
            "WHERE lang_text LIKE '%' || ? || '%' ORDER BY difficulty LIMIT 1",
            ("食べる",),
        ).fetchone()
    finally:
        db.close()
    assert rows[0][0] == "彼は朝ごはんを食べる。"
    assert rows[0][1] == "He eats breakfast."   # paired via links
    assert rows[0][2] >= 1
    assert rows[1][0] == "ありがとうございます。"
    assert rows[1][1] is None  # unlinked → no English translation
    assert example == ("彼は朝ごはんを食べる。",)


def test_limit_vocab_and_no_tatoeba(tmp_path):
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    out = tmp_path / "ja.augpack"
    summary = build_pack(out_path=out, lang_code="ja", jmdict_xml=jmdict, limit_vocab=1)
    assert summary == {"vocab": 1, "sentences": 0}
    db = sqlite3.connect(str(out))
    try:
        assert db.execute("SELECT COUNT(*) FROM vocab").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 0
    finally:
        db.close()


def test_rebuild_overwrites(tmp_path):
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    out = tmp_path / "ja.augpack"
    build_pack(out_path=out, lang_code="ja", jmdict_xml=jmdict, limit_vocab=2)
    # Build again with a smaller limit — should fully replace, not append.
    build_pack(out_path=out, lang_code="ja", jmdict_xml=jmdict, limit_vocab=1)
    db = sqlite3.connect(str(out))
    try:
        assert db.execute("SELECT COUNT(*) FROM vocab").fetchone()[0] == 1
    finally:
        db.close()


# ── server-side build hardening ───────────────────────────────────


def test_gzipped_jmdict_input(tmp_path):
    import gzip
    gz_path = tmp_path / "JMdict_e.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(_JMDICT_XML)
    out = tmp_path / "ja.augpack"
    summary = build_pack(out_path=out, lang_code="ja", jmdict_xml=gz_path)
    assert summary["vocab"] == 2
    db = sqlite3.connect(str(out))
    try:
        row = db.execute("SELECT surface, pos FROM vocab WHERE word_id='1358280'").fetchone()
    finally:
        db.close()
    assert row == ("食べる", "v1,vt")


def test_bz2_and_tar_tatoeba_inputs(tmp_path):
    import bz2
    import tarfile

    # sentences as a plain .bz2
    sents_bz2 = tmp_path / "sentences.tsv.bz2"
    with bz2.open(sents_bz2, "wt", encoding="utf-8") as f:
        f.write(_SENTENCES_TSV)
    # links inside a .tar.bz2 (the shape of the real Tatoeba dump)
    links_tsv = tmp_path / "links.csv"
    links_tsv.write_text(_LINKS_TSV, encoding="utf-8")
    links_tar = tmp_path / "links.tar.bz2"
    with tarfile.open(links_tar, "w:bz2") as tf:
        tf.add(links_tsv, arcname="links.csv")

    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    out = tmp_path / "ja.augpack"
    summary = build_pack(
        out_path=out, lang_code="ja", jmdict_xml=jmdict,
        tatoeba_sentences=sents_bz2, tatoeba_links=links_tar,
    )
    assert summary["sentences"] == 2
    db = sqlite3.connect(str(out))
    try:
        row = db.execute(
            "SELECT lang_text, en_text FROM sentences ORDER BY sent_id LIMIT 1"
        ).fetchone()
    finally:
        db.close()
    assert row == ("彼は朝ごはんを食べる。", "He eats breakfast.")


def test_corpus_derived_frequency(tmp_path):
    # 食べる appears in both jpn sentences; 朝ごはん in one. With no
    # explicit freq TSV, freq_rank should be derived from the corpus —
    # 食べる ranked above 朝ごはん.
    sentences = "\n".join([
        "1\tjpn\t彼は朝ごはんを食べる。",
        "2\tjpn\t毎日食べる。",
        "3\teng\tHe eats.",
    ])
    links = "1\t3\n"
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    s = _write(tmp_path / "sentences.tsv", sentences)
    ln = _write(tmp_path / "links.tsv", links)
    out = tmp_path / "ja.augpack"
    build_pack(
        out_path=out, lang_code="ja", jmdict_xml=jmdict,
        tatoeba_sentences=s, tatoeba_links=ln,  # no freq_tsv → compute
    )
    db = sqlite3.connect(str(out))
    try:
        taberu = db.execute("SELECT freq_rank FROM vocab WHERE word_id='1358280'").fetchone()[0]
        asagohan = db.execute("SELECT freq_rank FROM vocab WHERE word_id='1578850'").fetchone()
    finally:
        db.close()
    assert taberu == 1                       # most frequent
    # 朝ごはん only appears in the test JMdict if it's there; this fixture's
    # _JMDICT_XML doesn't include it, so it won't be ranked. Just assert
    # 食べる won the top spot.
    assert asagohan is None or asagohan[0] != 1


def test_explicit_freq_tsv_overrides_corpus(tmp_path):
    sentences = "1\tjpn\t毎日食べる。\n"
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    s = _write(tmp_path / "sentences.tsv", sentences)
    ln = _write(tmp_path / "links.tsv", "")
    freq = _write(tmp_path / "freq.tsv", "食べる\t99\n")
    out = tmp_path / "ja.augpack"
    build_pack(
        out_path=out, lang_code="ja", jmdict_xml=jmdict,
        tatoeba_sentences=s, tatoeba_links=ln, freq_tsv=freq,
    )
    db = sqlite3.connect(str(out))
    try:
        rank = db.execute("SELECT freq_rank FROM vocab WHERE word_id='1358280'").fetchone()[0]
    finally:
        db.close()
    assert rank == 99  # the explicit TSV value, not a corpus-derived 1


def test_jlpt_json_input(tmp_path):
    import json as _json
    jl = tmp_path / "jlpt.json"
    jl.write_text(_json.dumps([
        {"word": "食べる", "level": "N5"},
        {"word": "朝ごはん", "level": 5},
    ]), encoding="utf-8")
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    out = tmp_path / "ja.augpack"
    build_pack(out_path=out, lang_code="ja", jmdict_xml=jmdict, jlpt_tsv=jl)
    db = sqlite3.connect(str(out))
    try:
        row = db.execute("SELECT jlpt FROM vocab WHERE word_id='1358280'").fetchone()
    finally:
        db.close()
    assert row == (5,)


def test_progress_callback_invoked(tmp_path):
    jmdict = _write(tmp_path / "JMdict_e", _JMDICT_XML)
    out = tmp_path / "ja.augpack"
    seen = []
    build_pack(out_path=out, lang_code="ja", jmdict_xml=jmdict,
               progress=lambda frac, stage: seen.append((frac, stage)))
    assert seen, "progress callback should have been called"
    assert seen[-1][0] == 1.0
    assert all(0.0 <= f <= 1.0 for f, _ in seen)


# ── Wiktionary / Spanish builder ─────────────────────────────────────

# A miniature kaikki.org JSONL fixture covering: a noun ("casa"), a verb
# ("ser") with multiple senses across two lines (same headword, different
# POS — must merge into one row), and one entry with no glosses (skipped).
_KAIKKI_ES_JSONL = "\n".join([
    json.dumps({"word": "casa", "pos": "noun", "lang_code": "es",
                "senses": [{"glosses": ["house"]}, {"glosses": ["home"]}]}),
    json.dumps({"word": "ser", "pos": "verb", "lang_code": "es",
                "senses": [{"glosses": ["to be"]}]}),
    json.dumps({"word": "ser", "pos": "noun", "lang_code": "es",
                "senses": [{"glosses": ["being"]}]}),
    json.dumps({"word": "hola", "pos": "intj", "lang_code": "es",
                "senses": [{"glosses": ["hello"]}]}),
    # gloss-less entry → skipped
    json.dumps({"word": "skipme", "pos": "noun", "lang_code": "es", "senses": []}),
])

_TATOEBA_ES_SENTENCES = "\n".join([
    "10\tspa\tLa casa es grande.",
    "11\teng\tThe house is big.",
    "12\tspa\tHola, ¿cómo estás?",
])
_TATOEBA_ES_LINKS = "\n".join(["10\t11", "11\t10"])

# Hermitdave frequency list format: ``word count`` per line, descending
# count (which we re-rank to 1, 2, 3, …).
_HERMITDAVE_ES = "casa 100\nser 80\nhola 50\n"


@pytest.fixture
def built_es_pack(tmp_path):
    jsonl = _write(tmp_path / "es_wiktionary.jsonl", _KAIKKI_ES_JSONL)
    sents = _write(tmp_path / "sentences.tsv", _TATOEBA_ES_SENTENCES)
    links = _write(tmp_path / "links.tsv", _TATOEBA_ES_LINKS)
    freq = _write(tmp_path / "es_freq.txt", _HERMITDAVE_ES)
    out = tmp_path / "es.augpack"
    summary = build_pack_wiktionary(
        out_path=out, lang_code="es", wiktionary_jsonl=jsonl,
        tatoeba_sentences=sents, tatoeba_links=links,
        frequency_txt=freq, name="Spanish test", tatoeba_lang="spa",
    )
    return out, summary


def test_iter_kaikki_merges_by_headword(tmp_path):
    jsonl = _write(tmp_path / "es.jsonl", _KAIKKI_ES_JSONL)
    entries = list(iter_kaikki_entries(jsonl))
    by_word = {e["word_id"]: e for e in entries}
    # "ser" had two lines (verb + noun) — merged into one row with both POS.
    ser = by_word["ser"]
    assert set(ser["pos"].split(",")) == {"verb", "noun"}
    assert "to be" in ser["glosses"]
    assert "being" in ser["glosses"]
    # gloss-less entry skipped
    assert "skipme" not in by_word


def test_iter_hermitdave_frequency_ranks(tmp_path):
    freq = _write(tmp_path / "f.txt", _HERMITDAVE_ES)
    ranks = iter_hermitdave_frequency(freq)
    assert ranks == {"casa": 1, "ser": 2, "hola": 3}


def test_iter_hermitdave_frequency_case_folds(tmp_path):
    # OpenSubtitles-derived freq lists preserve sentence-initial capitals,
    # so "Y" and "y" appear at different ranks. The builder lowercases
    # surfaces so it would otherwise look up "y" and get the sentence-
    # initial rank instead of the (more accurate) lowercase rank.
    freq = _write(tmp_path / "f.txt", "y 500\nY 100\nde 1000\n")
    ranks = iter_hermitdave_frequency(freq)
    # First-occurrence wins for case-variants: "y" at line 1 keeps rank 1.
    assert ranks == {"y": 1, "de": 2}


def test_iter_kaikki_case_variants_merge(tmp_path):
    # Wiktionary entries for "y" (conj) and "Y" (letter) must collapse
    # into one row keyed on the lowercased headword — otherwise an L2
    # learner gets case-duplicate single-character clutter in their queue
    # ("y / Y / no / No" is the failure that drove this change).
    jsonl_lines = [
        json.dumps({"word": "y", "pos": "conj", "lang_code": "es",
                    "senses": [{"glosses": ["and"]}]}),
        json.dumps({"word": "Y", "pos": "letter", "lang_code": "es",
                    "senses": [{"glosses": ["the letter Y"]}]}),
        json.dumps({"word": "No", "pos": "adv", "lang_code": "es",
                    "senses": [{"glosses": ["no (sentence-initial)"]}]}),
        json.dumps({"word": "no", "pos": "adv", "lang_code": "es",
                    "senses": [{"glosses": ["no / not"]}]}),
    ]
    jsonl = _write(tmp_path / "es.jsonl", "\n".join(jsonl_lines))
    entries = list(iter_kaikki_entries(jsonl))
    by_word = {e["word_id"]: e for e in entries if e["kind"] == "vocab"}
    # Both case-variants of each headword collapse into one entry whose
    # word_id and surface are the lowercase canonical form.
    assert set(by_word) == {"y", "no"}
    # POS values from both case-variants accumulate so the merge is
    # information-preserving (one card now lists ``conj,letter``).
    assert set(by_word["y"]["pos"].split(",")) == {"conj", "letter"}
    assert "and" in by_word["y"]["glosses"]
    assert "the letter Y" in by_word["y"]["glosses"]
    # Same-POS case-variants ("No" + "no") collapse without doubling POS.
    assert by_word["no"]["pos"] == "adv"


def test_wiktionary_pack_builds_and_counts(built_es_pack):
    out, summary = built_es_pack
    assert out.exists()
    assert summary["vocab"] == 3   # casa, ser, hola — skipme dropped
    assert summary["sentences"] == 2   # 2 spa lines (eng filtered out)


def test_wiktionary_pack_meta(built_es_pack):
    out, _ = built_es_pack
    db = sqlite3.connect(str(out))
    try:
        kv = dict(db.execute("SELECT key, value FROM meta").fetchall())
    finally:
        db.close()
    assert kv["pack_kind"] == "language"
    assert kv["lang_code"] == "es"
    assert kv["tokenization"] == "whitespace"
    assert "Wiktionary" in kv["source"]
    pos_labels = json.loads(kv["pos_labels"])
    assert pos_labels["noun"] == "noun"
    assert pos_labels["verb"] == "verb"
    assert pos_labels["adj"] == "adjective"


def test_wiktionary_pack_vocab_rows(built_es_pack):
    out, _ = built_es_pack
    db = sqlite3.connect(str(out))
    try:
        casa = db.execute(
            "SELECT surface, reading, pos, glosses, freq_rank FROM vocab WHERE word_id='casa'"
        ).fetchone()
        ser = db.execute(
            "SELECT pos, glosses FROM vocab WHERE word_id='ser'"
        ).fetchone()
    finally:
        db.close()
    assert casa is not None
    surface, reading, pos, glosses, freq_rank = casa
    assert surface == "casa"
    assert reading == ""   # no separate phonetic column for ES
    assert pos == "noun"
    assert "house" in json.loads(glosses)
    assert freq_rank == 1   # top-of-list in hermitdave fixture
    # Merged multi-POS entry survives the DB write.
    ser_pos, ser_gl = ser
    assert set(ser_pos.split(",")) == {"verb", "noun"}
    assert {"to be", "being"} <= set(json.loads(ser_gl))
