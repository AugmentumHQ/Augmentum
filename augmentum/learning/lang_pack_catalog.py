"""Catalog of installable language packs and the open-source corpora that
build them.

A language pack is **built on demand**: when the user clicks "install
Japanese", a background job downloads the sources listed here, runs
:func:`augmentum.knowledge.lang_pack_builder.build_pack` server-side, and
writes ``<lang>.augpack`` into the knowledge-packs directory. Augmentum
redistributes nothing — every corpus is fetched directly from its
canonical host, so provenance and licensing are unambiguous and the data
is always current.

Curation principle: the *smallest set of high-quality, permissively-
licensed* corpora that makes the core learning loop work — a comprehensive
bilingual dictionary plus a large bank of natural example sentences.
Frequency ranking is *computed from the sentence corpus at build time*
(no extra download), so it reflects the same text the learner reads.
Optional extras (JLPT/HSK bands, kanji data, conjugation tables) layer in as
available. ``status="available"`` packs can be installed; planned entries may
exist later as catalog previews, but the current supported learner set is the
five available packs below."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """One downloadable input for a language-pack build."""

    key: str             # role: jmdict | tatoeba_sentences | tatoeba_links | jlpt | wiktionary | cedict | frequency
    url: str
    description: str = ""
    required: bool = True
    approx_mb: int = 0   # rough download size, for the progress UI
    sha256: str = ""     # optional integrity pin ("" = skip check)


@dataclass(frozen=True)
class LanguagePackSpec:
    """A user-installable language pack."""

    lang_code: str                 # ISO 639 code, e.g. "ja"
    name: str                      # display name
    builder: str                   # build pipeline id, dispatched in lang_pack_builder
    native_for: tuple[str, ...]    # native languages this pack targets ("en")
    sources: tuple[Source, ...]
    status: str = "available"      # "available" | "planned"
    word_segmentation: str = "spaces"   # "spaces" | "longest_prefix" (CJK)
    notes: str = ""
    approx_pack_mb: int = 0        # rough size of the built .augpack

    @property
    def total_download_mb(self) -> int:
        return sum(s.approx_mb for s in self.sources)

    def source(self, key: str) -> Source | None:
        for s in self.sources:
            if s.key == key:
                return s
        return None

    def to_public_dict(self) -> dict:
        """Shape returned by the catalog API endpoint."""
        return {
            "lang_code": self.lang_code,
            "name": self.name,
            "status": self.status,
            "approx_download_mb": self.total_download_mb,
            "approx_pack_mb": self.approx_pack_mb,
            "sources": [
                {"key": s.key, "description": s.description,
                 "required": s.required, "approx_mb": s.approx_mb}
                for s in self.sources
            ],
            "notes": self.notes,
        }


# Shared Tatoeba sources (the full multilingual dumps; the builder streams
# them and filters to the relevant language pair). Per-language sentence
# exports are smaller but there is no per-pair links export, so the build
# job needs the full links file regardless — keeping both as the full
# archive keeps the source list simple. Marked optional: example sentences
# are a nice-to-have on each SRS card, not load-bearing — if the (large)
# downloads fail or time out, the build still produces a vocab-only pack
# and the user gets dictionary + click-to-define + recognition cards
# without examples.
#   (Sizes are the live Content-Length as of 2026-05-11; verify before a
#    release. A future optimisation: per-language sentence files
#    (`per_language/{lang}/{lang}_sentences.tsv.bz2`, ~MBs) merged with the
#    English file — needs the builder to accept multiple sentence files.)
def _tatoeba_sources() -> tuple[Source, ...]:
    return (
        Source(
            key="tatoeba_sentences",
            url="https://downloads.tatoeba.org/exports/sentences.tar.bz2",
            description="Tatoeba sentences (all languages; filtered at build time) — CC BY 2.0 FR",
            required=False,
            approx_mb=206,
        ),
        Source(
            key="tatoeba_links",
            url="https://downloads.tatoeba.org/exports/links.tar.bz2",
            description="Tatoeba translation links — CC BY 2.0 FR",
            required=False,
            approx_mb=142,
        ),
    )


# -- Japanese ---------------------------------------------------------------
#
# What's inside the built ja.augpack:
#   vocab     — JMdict English edition (~200k entries). ``word_id`` is the
#               JMdict <ent_seq> (stable across releases → rebuilds never
#               orphan a learner's progress). surface = first kanji form
#               (or kana for kana-only words), reading = first kana form,
#               pos = comma-joined POS codes, glosses = JSON array of EN
#               meanings.
#   vocab_fts — FTS5 mirror of (surface, reading, glosses) for free-text
#               lookup (e.g. /lookup?q=breakfast → 朝ごはん via gloss).
#   sentences — Tatoeba jpn sentences with their English translations and
#               a crude difficulty bucket (length). Powers the example on
#               each SRS card and the "comprehensible input" surface.
#   freq_rank — computed at build time by greedily segmenting the Tatoeba
#               jpn corpus against the JMdict surface/reading forms and
#               ranking by occurrence count. No extra download; gives a
#               sensible vocabulary-introduction order.
#   jlpt      — (best-effort) N5..N1 band tags from a community word list;
#               build proceeds without it if the source 404s.

_JA = LanguagePackSpec(
    lang_code="ja",
    name="Japanese",
    builder="jmdict_tatoeba",
    native_for=("en",),
    status="available",
    word_segmentation="longest_prefix",
    approx_pack_mb=160,
    sources=(
        Source(
            # EDRDG's FTP host serves over plain HTTP — its HTTPS cert
            # doesn't match `ftp.edrdg.org`. ~10 MB gzipped XML.
            key="jmdict",
            url="http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz",
            description="JMdict English edition (gzipped XML) — EDRDG, CC BY-SA 4.0",
            approx_mb=10,
        ),
        *_tatoeba_sources(),
    ),
    notes=(
        "JMdict over HTTP (the EDRDG FTP host's HTTPS cert is misconfigured). "
        "Only JMdict is required — if the (large) Tatoeba downloads fail, the "
        "build still produces a working dictionary pack without example "
        "sentences. A JLPT-band tagging source was dropped for MVP (the "
        "community list URL had moved); JLPT tags aren't load-bearing."
    ),
)


# -- Additional supported packs --------------------------------------------

_ES = LanguagePackSpec(
    lang_code="es",
    name="Spanish",
    builder="wiktionary_tatoeba",
    native_for=("en",),
    status="available",
    word_segmentation="spaces",
    approx_pack_mb=120,
    sources=(
        Source(
            key="wiktionary",
            url="https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl",
            description="Wiktionary Spanish→English (kaikki.org JSONL extract) — CC BY-SA",
            approx_mb=40,
        ),
        *_tatoeba_sources(),
        Source(
            key="frequency",
            url="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_full.txt",
            description="OpenSubtitles-derived Spanish frequency list (hermitdave) — MIT",
            required=False,
            approx_mb=3,
        ),
    ),
    notes=(
        "First Wiktionary-sourced pack — exercises the kaikki + whitespace "
        "code path. JMdict-style stable IDs aren't available; surface "
        "form is the word_id (rebuilds preserve user progress as long as "
        "the headword spelling doesn't change). Verb conjugation tables "
        "(verbecc) deferred to the grammar phase."
    ),
)

_ZH = LanguagePackSpec(
    lang_code="zh",
    name="Chinese (Mandarin)",
    builder="cedict_tatoeba",
    native_for=("en",),
    status="available",
    word_segmentation="longest_prefix",
    approx_pack_mb=80,
    sources=(
        Source(
            key="cedict",
            url="https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz",
            description="CC-CEDICT Chinese-English dictionary (gzipped) — CC BY-SA 4.0",
            approx_mb=4,
        ),
        *_tatoeba_sources(),
        Source(
            key="hsk",
            url="https://raw.githubusercontent.com/elkmovie/hsk30/main/wordlist.txt",
            description="HSK 3.0 wordlist (levels 1-7 grouped, ~11k words) — MIT, OCR'd from official PDF by elkmovie/hsk30",
            required=False,
            approx_mb=1,
        ),
    ),
    notes=(
        "Like Japanese: longest-prefix segmentation against the dictionary at click time. "
        "Numbered pinyin in CEDICT is converted to tone-marked pinyin at build time. "
        "HSK 3.0 word list tags vocab with proficiency level (HSK1..HSK7) for level-aware "
        "drilling and progress estimation; the build still succeeds if the HSK download fails."
    ),
)

_FR = LanguagePackSpec(
    lang_code="fr",
    name="French",
    builder="wiktionary_tatoeba",
    native_for=("en",),
    status="available",
    word_segmentation="spaces",
    approx_pack_mb=120,
    sources=(
        Source(
            key="wiktionary",
            url="https://kaikki.org/dictionary/French/kaikki.org-dictionary-French.jsonl",
            description="Wiktionary French→English (kaikki.org JSONL extract) — CC BY-SA",
            approx_mb=50,
        ),
        *_tatoeba_sources(),
        Source(
            # Aligned with the Spanish path — same hermitdave format, so
            # the shared `wiktionary_tatoeba` builder ingests it without
            # a French-specific parser. Lexique 3.83 (richer but different
            # schema) is a future upgrade.
            key="frequency",
            url="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/fr/fr_full.txt",
            description="OpenSubtitles-derived French frequency list (hermitdave) — MIT",
            required=False,
            approx_mb=10,
        ),
    ),
)

_KO = LanguagePackSpec(
    lang_code="ko",
    name="Korean",
    builder="wiktionary_tatoeba",
    native_for=("en",),
    status="available",
    word_segmentation="spaces",
    approx_pack_mb=90,
    sources=(
        Source(
            key="wiktionary",
            url="https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl",
            description="Wiktionary Korean→English (kaikki.org JSONL extract) — CC BY-SA",
            approx_mb=20,
        ),
        *_tatoeba_sources(),
        Source(
            key="frequency",
            url="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ko/ko_full.txt",
            description="OpenSubtitles-derived Korean frequency list (hermitdave) — MIT",
            required=False,
            approx_mb=10,
        ),
    ),
    notes=(
        "Korean is written with spaces between words (eojeol), so whitespace "
        "segmentation works at a coarse level; finer morpheme analysis is a "
        "later phase. Note: Kokoro TTS ships zero Korean voices — the games' "
        "browser-system-voice fallback handles ko audio via OS-level TTS "
        "(macOS Yuna / Windows Heami / etc.)."
    ),
)


_CATALOG: dict[str, LanguagePackSpec] = {
    s.lang_code: s for s in (_JA, _ES, _ZH, _FR, _KO)
}

# Tatoeba uses ISO 639-3 codes in its dumps; map our pack lang_code to it.
TATOEBA_LANG_CODE: dict[str, str] = {
    "ja": "jpn", "es": "spa", "zh": "cmn", "fr": "fra", "ko": "kor", "en": "eng",
}


def all_packs() -> list[LanguagePackSpec]:
    """Every catalog entry (available + planned), in display order."""
    return list(_CATALOG.values())


def available_packs() -> list[LanguagePackSpec]:
    return [s for s in _CATALOG.values() if s.status == "available"]


def get(lang_code: str) -> LanguagePackSpec | None:
    return _CATALOG.get(lang_code)


def is_installable(lang_code: str) -> bool:
    spec = _CATALOG.get(lang_code)
    return spec is not None and spec.status == "available"
