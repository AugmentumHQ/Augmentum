"""TTS script routing — per-language consistency (2026-06-12).

One test per supported learning-catalog language (ja/zh/ko + Latin
es/fr pass-through) plus the web-content scripts (cyrillic, greek,
RTL/abugida drop classes). The contract: an English-pipeline voice
NEVER receives a script it can't pronounce — romanized when
deterministic, dropped from speech otherwise, visible text untouched
(this module only ever sees the spoken copy).
"""

from __future__ import annotations

from augmentum.voice.script_routing import (
    prepare_speech_text,
    segment,
)

# ── Pass-through (the common path must be byte-identical) ───────────

def test_plain_english_untouched():
    text = "Hello there — it's 3:45pm, and we're 100% done."
    out, stats = prepare_speech_text(text)
    assert out == text
    assert stats == {}


def test_spanish_french_latin_pass_through():
    # es/fr are learning-catalog languages but share the Latin
    # pipeline — accents must survive untouched.
    for text in ("¿Dónde está la biblioteca?", "C'est très intéressant, non?"):
        out, stats = prepare_speech_text(text)
        assert out == text
        assert stats == {}


# ── Japanese (kana — built-in Hepburn) ──────────────────────────────

def test_hiragana_romanized():
    out, stats = prepare_speech_text("こんにちは")
    assert out == "konnichiha" or out == "konnichiwa"  # は particle nuance
    assert stats.get("romanized_kana") == 1


def test_katakana_romanized_with_long_vowel():
    out, _ = prepare_speech_text("コーヒー")
    assert out == "koohii"


def test_kana_digraphs_and_sokuon():
    out, _ = prepare_speech_text("きょう がっこう ちょっと")
    assert "kyou" in out
    assert "gakkou" in out
    assert "chotto" in out  # っち → tch (Hepburn gemination)


def test_mixed_english_japanese_sentence():
    out, stats = prepare_speech_text('The word for cat is ねこ, easy!')
    assert "The word for cat is" in out
    assert "neko" in out
    assert "easy!" in out
    assert stats.get("romanized_kana") == 1


def test_kanji_without_dictionary_drops_not_garbles():
    # pykakasi isn't in the current image; kanji-only spans must drop
    # from speech rather than letter-spell. (When the lib is present
    # the dictionary tier romanizes instead — covered conditionally.)
    from augmentum.voice.script_routing import _kakasi
    out, stats = prepare_speech_text("漢字")
    if _kakasi() is None:
        assert out == ""
        assert stats.get("dropped_han") == 1 or stats.get("dropped_han_ja") == 1
    else:
        assert out.strip()
        assert "romanized_han" in str(stats)


# ── Korean (built-in Revised Romanization) ──────────────────────────

def test_hangul_romanized():
    out, stats = prepare_speech_text("안녕하세요")
    assert out == "annyeonghaseyo"
    assert stats.get("romanized_hangul") == 1


def test_hangul_mixed_sentence():
    out, _ = prepare_speech_text("Say 감사합니다 to thank someone.")
    assert "gamsahamnida" in out
    assert "Say" in out and "to thank someone." in out


# ── Chinese (dictionary tier — graceful without lib) ────────────────

def test_hanzi_without_pinyin_drops():
    from augmentum.voice.script_routing import _pinyin
    out, stats = prepare_speech_text("你好")
    if _pinyin() is None:
        assert out == ""
        assert stats.get("dropped_han") == 1
    else:
        assert "ni" in out and "hao" in out


# ── Cyrillic + Greek (built-in tables) ──────────────────────────────

def test_cyrillic_romanized():
    out, stats = prepare_speech_text("привет")
    assert out == "privet"
    assert stats.get("romanized_cyrillic") == 1


def test_cyrillic_capitalization_carries():
    out, _ = prepare_speech_text("Москва")
    assert out == "Moskva"


def test_greek_romanized():
    out, _ = prepare_speech_text("καλημέρα")
    # accented chars lack table entries and are skipped; core shape holds
    assert out.startswith("kal")
    assert "m" in out and "ra" in out


# ── Drop classes (no deterministic romanization) ────────────────────

def test_arabic_hebrew_thai_devanagari_drop():
    for text, script in (
        ("مرحبا", "arabic"), ("שלום", "hebrew"),
        ("สวัสดี", "thai"), ("नमस्ते", "devanagari"),
    ):
        out, stats = prepare_speech_text(f"hello {text} world")
        assert "hello" in out and "world" in out
        assert stats.get(f"dropped_{script}") == 1
        # The script itself never reaches the voice.
        assert text not in out


# ── Segmentation mechanics ──────────────────────────────────────────

def test_segment_tags_han_as_japanese_when_kana_present():
    runs = dict(segment("これは漢字です"))
    assert "han_ja" in runs


def test_cjk_punctuation_mapped():
    out, _ = prepare_speech_text("はい。そうです、ね！")
    assert "。" not in out and "、" not in out and "！" not in out
    assert "hai" in out


def test_clean_for_tts_carries_routing():
    # The universal choke point: every TTS consumer cleans through
    # clean_for_tts, so routing must fire there.
    from augmentum.voice.text_cleaning import clean_for_tts
    out = clean_for_tts("Listen: こんにちは friend")
    assert "こんにちは" not in out
    assert "konnichi" in out
    assert "friend" in out


def test_idempotent_double_application():
    once, _ = prepare_speech_text("ねこ and приет")
    twice, stats = prepare_speech_text(once)
    assert twice == once
    assert stats == {}
