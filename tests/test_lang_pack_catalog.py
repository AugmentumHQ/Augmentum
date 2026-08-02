"""Language-pack catalog tests - pure data, no app stack."""

from __future__ import annotations

from augmentum.learning import lang_pack_catalog as catalog

SUPPORTED = {"ja", "es", "zh", "fr", "ko"}
EXPECTED_BUILDERS = {
    "ja": ("jmdict_tatoeba", "longest_prefix"),
    "es": ("wiktionary_tatoeba", "spaces"),
    "zh": ("cedict_tatoeba", "longest_prefix"),
    "fr": ("wiktionary_tatoeba", "spaces"),
    "ko": ("wiktionary_tatoeba", "spaces"),
}


def test_japanese_is_available_and_complete():
    spec = catalog.get("ja")
    assert spec is not None
    assert spec.status == "available"
    assert spec.builder == "jmdict_tatoeba"
    assert spec.word_segmentation == "longest_prefix"
    keys = {s.key for s in spec.sources}
    assert {"jmdict", "tatoeba_sentences", "tatoeba_links"} <= keys
    # Only JMdict is required. Tatoeba is large and best-effort, so a slow
    # download can still yield a vocab-only pack.
    assert spec.source("jmdict").required is True
    assert spec.source("tatoeba_sentences").required is False
    assert spec.source("tatoeba_links").required is False
    # EDRDG's FTP host serves over plain HTTP (HTTPS cert is misconfigured).
    assert spec.source("jmdict").url.startswith("http://")
    assert "edrdg.org" in spec.source("jmdict").url
    assert spec.total_download_mb > 0


def test_supported_languages_are_installable():
    for code in SUPPORTED:
        assert catalog.is_installable(code) is True, code
    assert catalog.is_installable("xx") is False


def test_all_packs_and_available_packs():
    all_codes = {s.lang_code for s in catalog.all_packs()}
    assert all_codes == SUPPORTED
    assert {s.lang_code for s in catalog.available_packs()} == SUPPORTED


def test_builders_and_segmentation_match_language_families():
    for code, (builder, segmentation) in EXPECTED_BUILDERS.items():
        spec = catalog.get(code)
        assert spec is not None
        assert spec.status == "available"
        assert spec.builder == builder
        assert spec.word_segmentation == segmentation


def test_spanish_uses_wiktionary_builder():
    spec = catalog.get("es")
    assert spec is not None
    assert spec.status == "available"
    assert spec.builder == "wiktionary_tatoeba"
    assert spec.word_segmentation == "spaces"
    keys = {s.key for s in spec.sources}
    assert {"wiktionary", "tatoeba_sentences", "tatoeba_links"} <= keys
    assert spec.source("wiktionary").required is True
    # Hermitdave frequency list is optional; pack still builds without it.
    assert spec.source("frequency").required is False


def test_chinese_uses_cedict_builder():
    spec = catalog.get("zh")
    assert spec is not None
    assert spec.status == "available"
    assert spec.builder == "cedict_tatoeba"
    assert spec.word_segmentation == "longest_prefix"
    keys = {s.key for s in spec.sources}
    assert {"cedict", "tatoeba_sentences", "tatoeba_links", "hsk"} <= keys
    assert spec.source("cedict").required is True
    assert spec.source("hsk").required is False


def test_french_and_korean_use_shared_wiktionary_builder():
    for code in ("fr", "ko"):
        spec = catalog.get(code)
        assert spec is not None
        assert spec.status == "available"
        assert spec.builder == "wiktionary_tatoeba"
        assert spec.word_segmentation == "spaces"
        keys = {s.key for s in spec.sources}
        assert {"wiktionary", "tatoeba_sentences", "tatoeba_links", "frequency"} <= keys
        assert spec.source("wiktionary").required is True
        assert spec.source("frequency").required is False


def test_to_public_dict_shape():
    d = catalog.get("ja").to_public_dict()
    assert d["lang_code"] == "ja"
    assert d["status"] == "available"
    assert d["approx_download_mb"] > 0
    assert isinstance(d["sources"], list) and d["sources"]
    for s in d["sources"]:
        assert {"key", "description", "required", "approx_mb"} <= set(s)


def test_tatoeba_lang_code_mapping():
    assert catalog.TATOEBA_LANG_CODE["ja"] == "jpn"
    assert catalog.TATOEBA_LANG_CODE["es"] == "spa"
    assert catalog.TATOEBA_LANG_CODE["zh"] == "cmn"
    assert catalog.TATOEBA_LANG_CODE["fr"] == "fra"
    assert catalog.TATOEBA_LANG_CODE["ko"] == "kor"
    assert catalog.TATOEBA_LANG_CODE["en"] == "eng"


def test_get_unknown_returns_none():
    assert catalog.get("klingon") is None
