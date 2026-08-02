"""Build a language-learning ``.augpack`` from open-source corpora.

Builds a self-contained SQLite ``.augpack`` (``pack_kind=language``) from
a JMdict XML dictionary and (optionally) Tatoeba sentence dumps. Runs in
two contexts:

* **On install** — invoked by the ``lang_pack_install`` background job
  (see :mod:`augmentum.learning.lang_pack_catalog` for the source list).
* **Offline** — ``scripts/build_lang_pack.py`` for dev / pre-building.

Because it can run inside the server it is written to be memory-frugal:
JMdict is parsed with :func:`xml.etree.ElementTree.iterparse` (clearing
each entry as it's consumed rather than building the whole tree), and the
Tatoeba dumps are streamed line-by-line straight out of their ``.bz2`` /
``.tar.bz2`` archives.

Pack schema
-----------
- ``meta``       key/value: ``pack_kind=language``, ``lang_code``,
                 ``schema``, ``name``, ``source``, ``source_license``,
                 ``build_date``, ``vocab_count``, ``sentence_count``.
- ``vocab``      one row per dictionary entry. ``word_id`` is the JMdict
                 ``<ent_seq>`` (numeric, stable across releases).
                 ``surface`` = first kanji form (or kana for kana-only
                 entries), ``reading`` = first kana form, ``pos`` =
                 comma-joined POS codes, ``glosses`` = JSON array of
                 English meanings, ``freq_rank`` / ``jlpt`` nullable.
- ``vocab_fts``  FTS5 external-content mirror of (surface, reading,
                 glosses) for the free-text ``/lookup`` fallback.
- ``sentences``  Tatoeba target-language sentences with their English
                 translations and a crude difficulty bucket.

``freq_rank`` is, by default, *computed from the sentence corpus*: the
builder greedily segments each target-language sentence against the
dictionary surface/reading forms and ranks entries by occurrence count.
Pass an explicit ``freq_tsv`` to override.

JMdict ships with an inline DTD defining part-of-speech entities
(``&v1;`` etc.). stdlib ``xml.etree`` doesn't resolve those, so we strip
the non-standard entity refs to their bare names before parsing
(``&v1;`` -> ``v1``) — which is exactly the value we want to keep.

Source data:
  * JMdict (English edition): https://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz
  * Tatoeba dumps: https://tatoeba.org/downloads
"""

from __future__ import annotations

import bz2
import gzip
import io
import json
import re
import sqlite3
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Standard XML entities ElementTree DOES understand — leave them alone.
_KEEP_ENTITIES = ("amp", "lt", "gt", "quot", "apos")
_ENTITY_RE = re.compile(
    r"&(?!" + "|".join(e + ";" for e in _KEEP_ENTITIES) + r")([A-Za-z0-9_-]+);"
)
_DOCTYPE_RE = re.compile(r"<!DOCTYPE.*?\]>", re.DOTALL)
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# Longest dictionary form to try when greedily segmenting a sentence for
# the corpus-frequency pass (Japanese headwords are short).
_MAX_TOKEN_CHARS = 16

SCHEMA_VERSION = 2


# ── Part-of-speech label maps ────────────────────────────────────────
# Written into each pack's ``meta.pos_labels`` JSON blob at build time so
# the UI doesn't have to ship a hardcoded per-language map. Each pack
# becomes self-describing: install a new language → the pack carries its
# own labels → frontend renders correct labels with no JS deploy.

WIKTIONARY_POS_LABELS: dict[str, str] = {
    # kaikki.org uses Wiktionary's POS taxonomy — same shape across every
    # language it dumps. Adding a language → no new POS map needed.
    "noun": "noun", "verb": "verb", "adj": "adjective", "adv": "adverb",
    "pron": "pronoun", "det": "determiner", "conj": "conjunction",
    "intj": "interjection", "prep": "preposition", "particle": "particle",
    "num": "numeral", "art": "article", "name": "proper noun",
    "phrase": "phrase", "abbrev": "abbreviation", "prefix": "prefix",
    "suffix": "suffix", "infix": "infix", "contraction": "contraction",
    "character": "character", "symbol": "symbol",
    "romanization": "romanization", "punct": "punctuation",
    "interfix": "interfix", "circumfix": "circumfix",
    "postp": "postposition", "adv_phrase": "adverbial phrase",
    "adj_phrase": "adjective phrase", "verb_phrase": "verb phrase",
    "noun_phrase": "noun phrase",
}


JMDICT_POS_LABELS: dict[str, str] = {
    "n": "noun", "n-adv": "adverbial noun", "n-suf": "noun suffix",
    "n-pref": "noun prefix", "n-t": "temporal noun", "pn": "pronoun",
    "adj": "adjective", "adj-i": "i-adjective", "adj-na": "na-adjective",
    "adj-no": "adjectival noun", "adj-pn": "pre-noun adjectival",
    "adj-t": "taru-adjective", "adj-f": "prenominal noun/verb",
    "adv": "adverb", "adv-to": "adverb (と)",
    "aux": "auxiliary", "aux-v": "auxiliary verb", "aux-adj": "auxiliary adjective",
    "conj": "conjunction", "cop": "copula", "ctr": "counter",
    "exp": "expression", "int": "interjection", "num": "numeric",
    "prt": "particle", "pref": "prefix", "suf": "suffix", "unc": "unclassified",
    "v1": "Ichidan verb", "v1-s": "Ichidan verb (kureru)",
    "vz": "Ichidan verb (zuru)",
    "v5": "Godan verb", "v5u": "Godan verb (-う)",
    "v5u-s": "Godan verb (-う special)", "v5k": "Godan verb (-く)",
    "v5k-s": "Godan verb (iku/yuku)", "v5g": "Godan verb (-ぐ)",
    "v5s": "Godan verb (-す)", "v5t": "Godan verb (-つ)",
    "v5n": "Godan verb (-ぬ)", "v5b": "Godan verb (-ぶ)",
    "v5m": "Godan verb (-む)", "v5r": "Godan verb (-る)",
    "v5r-i": "Godan verb (-る irregular)", "v5aru": "Godan verb (-aru special)",
    "vk": "kuru verb (irregular)",
    "vs": "suru verb", "vs-s": "suru verb (-する)",
    "vs-i": "suru verb (irregular)", "vs-c": "su verb",
    "vt": "transitive verb", "vi": "intransitive verb",
    "vn": "irregular nu verb", "vr": "irregular ru verb",
    "v2a-s": "Nidan verb (-う)", "v-unspec": "verb",
}


# ── Compressed-file readers ──────────────────────────────────────────


def _read_text(path: Path) -> str:
    """Read a (possibly gzip/bz2-compressed) text file fully into memory.

    Used only for JMdict, which we then strip + iter-parse. ~60 MB of
    XML → ~120 MB of Python str at the peak; acceptable for a one-shot
    build job.
    """
    name = path.name.lower()
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    if name.endswith(".bz2"):
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _open_text_lines(path: Path) -> Iterator[str]:
    """Stream a text file's lines, transparently handling ``.gz``,
    ``.bz2`` and single-member ``.tar.{gz,bz2}`` archives (the shape of
    the Tatoeba dumps: ``sentences.tar.bz2`` → one ``sentences.csv``).
    """
    name = path.name.lower()
    if name.endswith((".tar.bz2", ".tbz2")):
        yield from _open_tar_lines(path, "r:bz2")
        return
    if name.endswith((".tar.gz", ".tgz")):
        yield from _open_tar_lines(path, "r:gz")
        return
    if name.endswith(".bz2"):
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")
        return
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def _open_tar_lines(path: Path, mode: str) -> Iterator[str]:
    with tarfile.open(path, mode) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            for raw in handle:
                yield raw.decode("utf-8", errors="replace").rstrip("\n")
            return  # first regular file only — Tatoeba archives carry one csv


# ── Schema ───────────────────────────────────────────────────────────


def _build_db_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE vocab (
            word_id   TEXT PRIMARY KEY,
            surface   TEXT NOT NULL,
            reading   TEXT NOT NULL,
            pos       TEXT,
            glosses   TEXT NOT NULL,
            freq_rank INTEGER,
            jlpt      INTEGER,
            -- Generic proficiency band: "A1".."C2" for derived/CEFR, or
            -- "N5".."N1" for JLPT (ja), or "HSK1".."HSK7" for zh. NULL
            -- when the source has no signal for this word.
            level     TEXT
        );
        CREATE INDEX idx_vocab_surface ON vocab(surface);
        CREATE INDEX idx_vocab_reading ON vocab(reading);
        CREATE INDEX idx_vocab_freq    ON vocab(freq_rank);
        CREATE INDEX idx_vocab_level   ON vocab(level);
        CREATE VIRTUAL TABLE vocab_fts USING fts5(
            surface, reading, glosses,
            content='vocab', content_rowid='rowid'
        );
        CREATE TABLE sentences (
            sent_id    INTEGER PRIMARY KEY,
            lang_text  TEXT NOT NULL,
            en_text    TEXT,
            difficulty INTEGER
        );
        -- Inflected-form → lemma mapping. Powers click-define on
        -- conjugated verbs and pluralised nouns: "hablé" → "hablar",
        -- "pies" → "pie". Populated for langs whose dictionary source
        -- ships form-of metadata (kaikki Wiktionary extracts do).
        CREATE TABLE inflections (
            form  TEXT NOT NULL,
            lemma TEXT NOT NULL,
            tags  TEXT,           -- JSON array, e.g. ["plural","masculine"]
            PRIMARY KEY (form, lemma)
        );
        CREATE INDEX idx_inflections_form ON inflections(form);
        """
    )


# ── JMdict ───────────────────────────────────────────────────────────


def _strip_jmdict_entities(xml_text: str) -> str:
    """Remove the inline DTD and rewrite ``&v1;`` -> ``v1`` so stdlib
    ElementTree can parse the JMdict body."""
    xml_text = _DOCTYPE_RE.sub("", xml_text, count=1)
    return _ENTITY_RE.sub(r"\1", xml_text)


def _parse_jmdict_entry(entry: ET.Element) -> dict | None:
    seq_el = entry.find("ent_seq")
    if seq_el is None or not (seq_el.text or "").strip():
        return None
    word_id = seq_el.text.strip()
    kebs = [k.text.strip() for k in entry.findall("k_ele/keb") if k.text and k.text.strip()]
    rebs = [r.text.strip() for r in entry.findall("r_ele/reb") if r.text and r.text.strip()]
    if not rebs:
        return None
    surface = kebs[0] if kebs else rebs[0]
    reading = rebs[0]
    pos: list[str] = []
    glosses: list[str] = []
    for sense in entry.findall("sense"):
        for p in sense.findall("pos"):
            code = (p.text or "").strip()
            if code and code not in pos:
                pos.append(code)
        for g in sense.findall("gloss"):
            if g.attrib.get(_XML_LANG, "eng") != "eng":
                continue
            txt = (g.text or "").strip()
            if txt:
                glosses.append(txt)
    if not glosses:
        return None
    return {
        "word_id": word_id,
        "surface": surface,
        "reading": reading,
        "pos": ",".join(pos),
        "glosses": glosses,
    }


def iter_jmdict_entries(jmdict_xml: Path) -> Iterator[dict]:
    """Yield ``{word_id, surface, reading, pos, glosses}`` per JMdict
    entry. Accepts a plain or gzipped XML file. Memory-frugal — uses
    incremental parsing and clears each entry's subtree once consumed
    (leaves only an empty ``<entry/>`` shell attached to the root, which
    is negligible)."""
    text = _strip_jmdict_entities(_read_text(jmdict_xml))
    for _event, elem in ET.iterparse(io.StringIO(text), events=("end",)):
        if elem.tag != "entry":
            continue
        parsed = _parse_jmdict_entry(elem)
        if parsed is not None:
            yield parsed
        elem.clear()


# ── Wiktionary (kaikki.org JSONL) ────────────────────────────────────


def iter_kaikki_entries(jsonl_path: Path) -> Iterator[dict]:
    """Yield merged-by-headword vocab dicts from a kaikki.org JSONL dump.

    Kaikki dumps one Wiktionary entry per line. The same headword can
    appear in multiple lines (one per part-of-speech section, AND once
    per case-variant — Wiktionary lists ``y`` as a conjunction and
    ``Y`` as the letter as two distinct entries), so we accumulate
    them in memory keyed by **lowercased** surface form and yield a
    single merged row per unique word.

    Case-folding the merge key is deliberate: the L2-learner SRS has
    one job — teach the user *one* card per logical word. Letting
    sentence-initial capitals like ``Y``/``y``, ``No``/``no``,
    ``Si``/``si`` survive as separate rows fills starter queues with
    duplicate single-character function words (the exact failure that
    drove this change — Spanish learners were getting queues like
    ``y / Y / no / si / Si``). The semantic trade — that "Y the
    letter" stops being its own dictionary entry — is acceptable for
    learner use cases. The canonical surface stored is the lowercase
    form. POS values from all case-variants are concatenated; if a
    case-variant carries a niche POS (``letter``, etc.) it surfaces
    in the merged POS string alongside the primary one.

    Memory: ~50k unique words for Spanish → ~25-40 MB peak. Fine for the
    one-shot build job. Yields:
    ``{word_id, surface, reading, pos, glosses, inflections}`` where
    ``inflections`` is a list of ``{form, lemma, tags}`` — populated
    when the entry is itself an inflected form (``form_of`` tag). The
    ``word_id`` is the lowercased surface (kaikki has no stable
    cross-release ID; the lowercased headword *is* the stable key for
    non-CJK languages — kanji forms are what break that invariant for
    ja, hence JMdict's ``ent_seq``).
    """
    merged: dict[str, dict] = {}
    inflections: list[dict] = []
    for raw in _open_text_lines(jsonl_path):
        line = raw.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        key = word.lower()
        pos = (entry.get("pos") or "").strip()
        # Flatten every sense's glosses into one list, deduped while
        # preserving order (first-mention wins). Also harvest form-of
        # lemma links per sense — a kaikki entry can carry multiple
        # senses, only some of which are inflectional, so the scan is
        # per-sense.
        glosses: list[str] = []
        for sense in entry.get("senses") or ():
            if not isinstance(sense, dict):
                continue
            for g in sense.get("glosses") or ():
                g = (g or "").strip()
                if g and g not in glosses:
                    glosses.append(g)
            sense_tags = [t for t in (sense.get("tags") or []) if isinstance(t, str)]
            for fo in sense.get("form_of") or ():
                if not isinstance(fo, dict):
                    continue
                lemma = (fo.get("word") or "").strip().lower()
                if not lemma or lemma == key:
                    continue
                inflections.append({
                    "form": key,
                    "lemma": lemma,
                    "tags": sense_tags,
                })
        if not glosses:
            continue
        bucket = merged.get(key)
        if bucket is None:
            merged[key] = {
                "word_id": key,
                "surface": key,
                "reading": "",
                "pos": [pos] if pos else [],
                "glosses": glosses,
            }
        else:
            if pos and pos not in bucket["pos"]:
                bucket["pos"].append(pos)
            for g in glosses:
                if g not in bucket["glosses"]:
                    bucket["glosses"].append(g)
    # Yield headwords first, then inflections — the builder needs the
    # vocab table populated before it can index forms.
    for _key, bucket in merged.items():
        yield {
            "kind": "vocab",
            "word_id": bucket["word_id"],
            "surface": bucket["surface"],
            "reading": bucket["reading"],
            "pos": ",".join(bucket["pos"]),
            "glosses": bucket["glosses"],
        }
    # Dedupe (form, lemma) pairs — kaikki has duplicates across senses.
    seen_pairs: set[tuple[str, str]] = set()
    for inf in inflections:
        key = (inf["form"], inf["lemma"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        yield {"kind": "inflection", **inf}


# Frequency-rank → CEFR band, calibrated against Davies' Routledge
# frequency dictionaries for major Romance languages. A1 covers the
# bedrock 750 words that handle ~80% of conversational tokens; bands
# widen toward the tail. None of these are gospel — they're a
# defensible coarse-grained signal so the progress dashboard can say
# "you're approaching B1" instead of nothing.
_CEFR_FREQ_BANDS: tuple[tuple[int, str], ...] = (
    (750,   "A1"),
    (1500,  "A2"),
    (3000,  "B1"),
    (5000,  "B2"),
    (8000,  "C1"),
    (16000, "C2"),
)


def _cefr_band_for_rank(rank: int | None) -> str | None:
    if rank is None or rank <= 0:
        return None
    for threshold, label in _CEFR_FREQ_BANDS:
        if rank <= threshold:
            return label
    return None   # tail beyond C2 — leave unlevelled rather than mislabelling


# ── CC-CEDICT (Chinese) ──────────────────────────────────────────────

# Tone-marked vowels by tone (1..4) for pinyin display. Index 0 unused.
_PINYIN_TONE_MARKS: dict[str, tuple[str, ...]] = {
    "a": ("a", "ā", "á", "ǎ", "à"),
    "e": ("e", "ē", "é", "ě", "è"),
    "i": ("i", "ī", "í", "ǐ", "ì"),
    "o": ("o", "ō", "ó", "ǒ", "ò"),
    "u": ("u", "ū", "ú", "ǔ", "ù"),
    "ü": ("ü", "ǖ", "ǘ", "ǚ", "ǜ"),
}


def _convert_pinyin_syllable(syl: str) -> str:
    """Convert a single numbered pinyin syllable (``zhao1``, ``lu:3``) to
    a tone-marked one (``zhāo``, ``lǚ``). Tone 5 / no tone returns the
    unmarked base. ``u:`` (CEDICT's ASCII spelling for ü) is normalised.

    Tone-mark placement follows the standard pinyin rule:
      1. If 'a' is in the syllable, mark goes on 'a'.
      2. Else if 'e' or 'o' is present, mark goes on whichever is present.
      3. Else mark goes on the *last* vowel (handles 'iu' → 'iú', 'ui' → 'uí').
    """
    if not syl:
        return ""
    # CEDICT writes ü as u:
    syl = syl.replace("u:", "ü").replace("U:", "Ü")
    tone = 0
    if syl[-1].isdigit():
        tone = int(syl[-1])
        syl = syl[:-1]
    if tone not in (1, 2, 3, 4) or not syl:
        return syl   # neutral tone / non-pinyin token
    lowered = syl.lower()
    if "a" in lowered:
        target = lowered.index("a")
    elif "e" in lowered:
        target = lowered.index("e")
    elif "o" in lowered:
        target = lowered.index("o")
    else:
        # Last vowel in the syllable.
        target = -1
        for i, ch in enumerate(lowered):
            if ch in "iouüü":
                target = i
        if target < 0:
            return syl
    base = lowered[target]
    marks = _PINYIN_TONE_MARKS.get(base)
    if not marks:
        return syl
    marked = marks[tone]
    # Preserve original case at that position.
    if syl[target].isupper():
        marked = marked.upper()
    return syl[:target] + marked + syl[target + 1:]


def _convert_pinyin(reading: str) -> str:
    """Convert space-separated numbered pinyin syllables to tone-marked
    pinyin. Non-syllable tokens (English words, digits) pass through."""
    return " ".join(_convert_pinyin_syllable(s) for s in reading.split())


def iter_cedict_entries(cedict_path: Path) -> Iterator[dict]:
    """Yield merged-by-simplified-form vocab dicts from a CC-CEDICT dump.

    CEDICT row format::

        TRAD SIMP [pin1 yin2 ...] /gloss/gloss/.../

    Header lines start with ``#`` or ``%`` and are skipped. The same
    simplified form can appear with different traditional variants or
    different pinyin readings — we keep distinct readings but merge
    their gloss lists, matching the JMdict-on-kanji merge behaviour.

    Yields ``{kind, word_id, surface, reading, pos, glosses, traditional}``.
    ``word_id`` is the simplified form (modern reading-oriented stable key).
    """
    merged: dict[str, dict] = {}
    line_re = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/\s*$")
    for raw in _open_text_lines(cedict_path):
        line = raw.rstrip("\n").rstrip("\r")
        if not line or line[0] in "#%":
            continue
        m = line_re.match(line)
        if not m:
            continue
        trad, simp, pinyin_raw, gloss_blob = m.groups()
        reading = _convert_pinyin(pinyin_raw)
        glosses = [g.strip() for g in gloss_blob.split("/") if g.strip()]
        if not glosses:
            continue
        bucket = merged.get(simp)
        if bucket is None:
            merged[simp] = {
                "word_id": simp,
                "surface": simp,
                "reading": reading,
                "pos": "",
                "glosses": list(glosses),
                "traditional": trad if trad != simp else "",
            }
        else:
            for g in glosses:
                if g not in bucket["glosses"]:
                    bucket["glosses"].append(g)
            # If we already have a reading, only append a new variant
            # if it's different — keeps the bucket compact.
            if reading and reading not in bucket["reading"].split(" / "):
                bucket["reading"] = bucket["reading"] + " / " + reading if bucket["reading"] else reading
            if trad and trad != simp and bucket.get("traditional") and trad not in bucket["traditional"].split(","):
                bucket["traditional"] = bucket["traditional"] + "," + trad
    for word, bucket in merged.items():
        yield {
            "kind": "vocab",
            "word_id": bucket["word_id"],
            "surface": bucket["surface"],
            "reading": bucket["reading"],
            "pos": bucket["pos"],
            "glosses": bucket["glosses"],
            "traditional": bucket.get("traditional", ""),
        }


# ── HSK 3.0 level list ───────────────────────────────────────────────

# Section markers in elkmovie/hsk30/wordlist.txt indicate level boundaries.
# HSK 3.0 has levels 1-6 individually + a combined 7-9 advanced band; we
# map them to "HSK1".."HSK7" for a clean ordinal progression.
_HSK_SECTION_LEVEL: dict[str, str] = {
    "一级词汇表":     "HSK1",
    "二级词汇表":     "HSK2",
    "三级词汇表":     "HSK3",
    "四级词汇表":     "HSK4",
    "五级词汇表":     "HSK5",
    "六级词汇表":     "HSK6",
    "七一九级词汇表": "HSK7",
}


def iter_hsk_levels(hsk_path: Path) -> dict[str, str]:
    """Parse the elkmovie/hsk30 wordlist into ``{surface: level}``.

    The file is a flat text dump with section headers (``一级词汇表``…)
    delineating HSK levels, then numbered rows like ``1 爱`` or
    ``4 爸爸｜爸`` (pipe-alternate forms count for both surfaces) or
    ``6 白（形）`` (parenthetical POS / disambiguation hints stripped
    for the surface key).
    """
    out: dict[str, str] = {}
    current_level: str | None = None
    paren_re = re.compile(r"（[^）]*）")
    for raw in _open_text_lines(hsk_path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in _HSK_SECTION_LEVEL:
            current_level = _HSK_SECTION_LEVEL[line]
            continue
        if current_level is None:
            continue
        # Row: "<index> <surface>" — strip the leading numeric id.
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        surface_raw = parts[1].strip()
        # Drop parenthetical hints, e.g. "白（形）" → "白"
        surface_raw = paren_re.sub("", surface_raw).strip()
        # Pipe alternates: "爸爸｜爸" → both "爸爸" and "爸" map to the
        # same level (each is a real word the learner needs).
        for variant in surface_raw.split("｜"):
            v = variant.strip()
            if v and v not in out:
                out[v] = current_level
    return out


def iter_hermitdave_frequency(path: Path) -> dict[str, int]:
    """Parse a hermitdave/FrequencyWords ``word count`` file → ``{word: rank}``.
    Lines are descending-frequency; rank = 1-based line number after the
    first valid line. Lines that don't parse are skipped silently.

    Words are case-folded before keying — the source list (OpenSubtitles-
    derived) preserves sentence-initial capitals, so ``y`` (rank 6 in
    Spanish) and ``Y`` (much later, sentence-initial only) appear as
    separate ranks. Without folding, vocab lookup against the kaikki
    pack (which now case-folds its surfaces too) would miss the more
    informative rank, biasing the CEFR band assignment.
    The earliest (lowest-numbered) rank for any case-variant wins.
    """
    out: dict[str, int] = {}
    rank = 0
    for raw in _open_text_lines(path):
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        try:
            int(parts[1])
        except ValueError:
            continue
        word = parts[0].lower() if parts[0] else ""
        if word and word not in out:
            rank += 1
            out[word] = rank
    return out


# ── Tatoeba ──────────────────────────────────────────────────────────


def iter_tatoeba_pairs(
    sentences_path: Path,
    links_path: Path,
    *,
    lang: str = "jpn",
    trans_lang: str = "eng",
) -> Iterator[tuple[str, str | None]]:
    """Yield ``(target_sentence, english_translation_or_None)`` pairs.

    Both files may be plain, ``.bz2``, or single-member ``.tar.bz2``.
    Tatoeba ``sentences.csv`` columns: ``id<TAB>lang<TAB>text``;
    ``links.csv`` columns: ``sentence_id<TAB>translation_id``.
    """
    target: dict[int, str] = {}
    english: dict[int, str] = {}
    for raw in _open_text_lines(sentences_path):
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        try:
            sid = int(parts[0])
        except ValueError:
            continue
        if parts[1] == lang:
            target[sid] = parts[2]
        elif parts[1] == trans_lang:
            english[sid] = parts[2]
    trans_of: dict[int, int] = {}
    for raw in _open_text_lines(links_path):
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if a in target and b in english and a not in trans_of:
            trans_of[a] = b
    for sid, txt in target.items():
        en = english.get(trans_of[sid]) if sid in trans_of else None
        yield txt, en


def _difficulty_bucket(text: str) -> int:
    """Crude difficulty by character length (no tokenizer at build time)."""
    n = len(text)
    if n <= 10:
        return 1
    if n <= 20:
        return 2
    if n <= 35:
        return 3
    return 4


# ── Optional side-tables ─────────────────────────────────────────────


def _load_kv_tsv(path: Path | None) -> dict[str, int]:
    """Load a ``key<TAB>int`` file (frequency rank / JLPT level), or {}."""
    out: dict[str, int] = {}
    if not path or not path.exists():
        return out
    for raw in _open_text_lines(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out[parts[0].strip()] = int(parts[1].strip())
        except ValueError:
            continue
    return out


def _load_word_levels(path: Path | None) -> dict[str, int]:
    """Load a word→level map from JSON (``[{"word":..,"level":..}, ...]``
    or ``{"N5": [..], ...}``) or a ``word<TAB>level`` TSV. Best-effort —
    unrecognised shapes yield ``{}``."""
    if not path or not path.exists():
        return {}
    name = path.name.lower()
    if name.endswith((".json", ".jsonl")):
        out: dict[str, int] = {}
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return {}

        def _coerce_level(v) -> int | None:
            # "N5" / "n5" / 5 / "5" -> 5
            if isinstance(v, int):
                return v if 1 <= v <= 5 else None
            s = str(v).strip().lower().lstrip("n")
            return int(s) if s.isdigit() and 1 <= int(s) <= 5 else None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # JSONL
            for ln in raw.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                w = obj.get("word") or obj.get("kanji") or obj.get("expression")
                lvl = _coerce_level(obj.get("level") or obj.get("jlpt"))
                if w and lvl:
                    out[str(w)] = lvl
            return out
        if isinstance(data, dict):
            for k, words in data.items():
                lvl = _coerce_level(k)
                if lvl and isinstance(words, list):
                    for w in words:
                        if w:
                            out[str(w)] = lvl
        elif isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                w = obj.get("word") or obj.get("kanji") or obj.get("expression")
                lvl = _coerce_level(obj.get("level") or obj.get("jlpt"))
                if w and lvl:
                    out[str(w)] = lvl
        return out
    return _load_kv_tsv(path)


# ── Corpus-derived frequency ─────────────────────────────────────────


def _build_form_index(db: sqlite3.Connection) -> dict[str, list[str]]:
    """Map every surface / reading form to the word_ids that carry it."""
    idx: dict[str, list[str]] = {}
    for word_id, surface, reading in db.execute("SELECT word_id, surface, reading FROM vocab"):
        for form in (surface, reading):
            if form:
                idx.setdefault(form, []).append(word_id)
    return idx


def _count_forms_in_sentence(text: str, form_index: dict, counts: dict[str, int]) -> None:
    """Greedy maximum-matching segmentation: walk the string, consume the
    longest form present in the index, bump the count for every word_id
    that carries it; advance by 1 on no match."""
    n = len(text)
    i = 0
    while i < n:
        matched: str | None = None
        for k in range(min(_MAX_TOKEN_CHARS, n - i), 0, -1):
            cand = text[i : i + k]
            if cand in form_index:
                matched = cand
                break
        if matched is None:
            i += 1
            continue
        for wid in form_index[matched]:
            counts[wid] = counts.get(wid, 0) + 1
        i += len(matched)


def _ranks_from_counts(counts: dict[str, int]) -> dict[str, int]:
    """word_id -> 1-based rank, most frequent first. Unseen words absent."""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {wid: rank for rank, (wid, _) in enumerate(ordered, start=1)}


# ── Build ────────────────────────────────────────────────────────────


def build_pack(
    *,
    out_path: Path,
    lang_code: str,
    jmdict_xml: Path,
    tatoeba_sentences: Path | None = None,
    tatoeba_links: Path | None = None,
    freq_tsv: Path | None = None,
    jlpt_tsv: Path | None = None,
    name: str = "",
    source_license: str = "JMdict CC BY-SA 4.0; Tatoeba CC BY 2.0 FR",
    compute_frequency: bool = True,
    tatoeba_lang: str = "jpn",
    limit_vocab: int | None = None,
    limit_sentences: int | None = None,
    progress=None,
) -> dict[str, int]:
    """Build ``out_path`` from the supplied corpora. Overwrites any
    existing file. Returns ``{"vocab": n, "sentences": m}``.

    ``progress`` — optional ``callable(fraction: float, stage: str)`` for
    the install-job progress UI.
    """
    def _p(frac: float, stage: str) -> None:
        if progress:
            try:
                progress(max(0.0, min(1.0, frac)), stage)
            except Exception as exc:
                # Progress callback failure shouldn't abort the build —
                # debug-log so a faulty install-job UI is findable.
                log.debug("lang_pack_progress_callback_failed", error=str(exc))

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    explicit_freq = _load_kv_tsv(freq_tsv)
    word_levels = _load_word_levels(jlpt_tsv)

    db = sqlite3.connect(str(out_path))
    try:
        _build_db_schema(db)

        # 1. Dictionary entries.
        _p(0.05, "parsing dictionary")
        vocab_n = 0
        for entry in iter_jmdict_entries(jmdict_xml):
            if limit_vocab is not None and vocab_n >= limit_vocab:
                break
            jlpt = word_levels.get(entry["surface"]) or word_levels.get(entry["reading"])
            # `level` mirrors `jlpt` as a string token ("N5".."N1") so the
            # progress dashboard's per-card level logic works uniformly
            # across packs — ja reads N-labels, es/fr reads A1..C2, zh
            # reads HSK1..HSK7.
            level = f"N{int(jlpt)}" if jlpt else None
            db.execute(
                """INSERT OR IGNORE INTO vocab
                       (word_id, surface, reading, pos, glosses, freq_rank, jlpt, level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry["word_id"], entry["surface"], entry["reading"], entry["pos"],
                    json.dumps(entry["glosses"], ensure_ascii=False),
                    explicit_freq.get(entry["surface"]) or explicit_freq.get(entry["reading"]),
                    jlpt, level,
                ),
            )
            vocab_n += 1
            if vocab_n % 20000 == 0:
                _p(0.05 + 0.25 * min(1.0, vocab_n / 200000), f"parsing dictionary ({vocab_n})")
        db.commit()

        # 2. Sentences (+ accumulate corpus word counts for frequency).
        _p(0.35, "parsing sentences")
        form_index = _build_form_index(db) if (compute_frequency and not explicit_freq) else {}
        counts: dict[str, int] = {}
        sent_n = 0
        if tatoeba_sentences and tatoeba_links and tatoeba_sentences.exists() and tatoeba_links.exists():
            for jp, en in iter_tatoeba_pairs(tatoeba_sentences, tatoeba_links, lang=tatoeba_lang):
                if limit_sentences is not None and sent_n >= limit_sentences:
                    break
                db.execute(
                    "INSERT INTO sentences (lang_text, en_text, difficulty) VALUES (?, ?, ?)",
                    (jp, en, _difficulty_bucket(jp)),
                )
                if form_index:
                    _count_forms_in_sentence(jp, form_index, counts)
                sent_n += 1
                if sent_n % 50000 == 0:
                    _p(0.35 + 0.35 * min(1.0, sent_n / 250000), f"parsing sentences ({sent_n})")
            db.commit()

        # 3. Frequency ranks (corpus-derived unless an explicit TSV was given).
        if counts:
            _p(0.78, "ranking vocabulary")
            for wid, rank in _ranks_from_counts(counts).items():
                db.execute("UPDATE vocab SET freq_rank = ? WHERE word_id = ?", (rank, wid))
            db.commit()

        # 4. FTS index.
        _p(0.88, "indexing")
        db.execute("INSERT INTO vocab_fts(vocab_fts) VALUES('rebuild')")

        # 5. Metadata.
        meta = {
            "pack_kind": "language",
            "lang_code": lang_code,
            "schema": str(SCHEMA_VERSION),
            "name": name or f"{lang_code.upper()} language pack",
            "source": "JMdict + Tatoeba",
            "source_license": source_license,
            "build_date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "vocab_count": str(vocab_n),
            "sentence_count": str(sent_n),
            "tokenization": "longest_prefix",
            "pos_labels": json.dumps(JMDICT_POS_LABELS, ensure_ascii=False),
        }
        db.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            list(meta.items()),
        )
        db.commit()
        _p(0.95, "finalizing")
        db.execute("VACUUM")
        _p(1.0, "done")
        return {"vocab": vocab_n, "sentences": sent_n}
    finally:
        db.close()


# ── Wiktionary-sourced builder (Spanish, French, Korean, …) ──────────


def build_pack_wiktionary(
    *,
    out_path: Path,
    lang_code: str,
    wiktionary_jsonl: Path,
    tatoeba_sentences: Path | None = None,
    tatoeba_links: Path | None = None,
    frequency_txt: Path | None = None,
    name: str = "",
    source_license: str = "Wiktionary CC BY-SA; Tatoeba CC BY 2.0 FR",
    tatoeba_lang: str = "spa",
    limit_vocab: int | None = None,
    limit_sentences: int | None = None,
    progress=None,
) -> dict[str, int]:
    """Build a language pack from a kaikki.org Wiktionary JSONL dump plus
    (optionally) the Tatoeba sentence corpus and a hermitdave-style
    frequency list.

    Pack shape is identical to :func:`build_pack` — same SQLite schema,
    same query API — but the source/format of the vocab and frequency
    inputs differs, and the resulting pack records ``tokenization =
    "whitespace"`` so the runtime tokenizer dispatches to the word-
    boundary path instead of the CJK longest-prefix path.
    """
    def _p(frac: float, stage: str) -> None:
        if progress:
            try:
                progress(max(0.0, min(1.0, frac)), stage)
            except Exception as exc:
                # Progress callback failure shouldn't abort the build —
                # debug-log so a faulty install-job UI is findable.
                log.debug("lang_pack_progress_callback_failed", error=str(exc))

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    explicit_freq = iter_hermitdave_frequency(frequency_txt) if frequency_txt else {}

    db = sqlite3.connect(str(out_path))
    try:
        _build_db_schema(db)

        # 1. Dictionary entries + form-of relations from kaikki JSONL.
        #    The iterator yields vocab rows first, then inflection rows,
        #    so the FK-style ordering is implicit.
        _p(0.05, "parsing dictionary")
        vocab_n = 0
        infl_n = 0
        for entry in iter_kaikki_entries(wiktionary_jsonl):
            kind = entry.get("kind")
            if kind == "vocab":
                if limit_vocab is not None and vocab_n >= limit_vocab:
                    continue
                rank = (
                    explicit_freq.get(entry["surface"])
                    or explicit_freq.get(entry["surface"].lower())
                )
                level = _cefr_band_for_rank(rank)
                db.execute(
                    """INSERT OR IGNORE INTO vocab
                           (word_id, surface, reading, pos, glosses, freq_rank, jlpt, level)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, ?)""",
                    (
                        entry["word_id"], entry["surface"], entry["reading"], entry["pos"],
                        json.dumps(entry["glosses"], ensure_ascii=False),
                        rank, level,
                    ),
                )
                vocab_n += 1
                if vocab_n % 10000 == 0:
                    _p(0.05 + 0.25 * min(1.0, vocab_n / 80000), f"parsing dictionary ({vocab_n})")
            elif kind == "inflection":
                db.execute(
                    """INSERT OR IGNORE INTO inflections (form, lemma, tags)
                       VALUES (?, ?, ?)""",
                    (
                        entry["form"], entry["lemma"],
                        json.dumps(entry.get("tags") or [], ensure_ascii=False),
                    ),
                )
                infl_n += 1
        db.commit()

        # 2. Sentences (the Tatoeba pair iterator handles language filtering).
        _p(0.35, "parsing sentences")
        sent_n = 0
        if (tatoeba_sentences and tatoeba_links
                and tatoeba_sentences.exists() and tatoeba_links.exists()):
            for tgt, en in iter_tatoeba_pairs(
                tatoeba_sentences, tatoeba_links, lang=tatoeba_lang,
            ):
                if limit_sentences is not None and sent_n >= limit_sentences:
                    break
                db.execute(
                    "INSERT INTO sentences (lang_text, en_text, difficulty) VALUES (?, ?, ?)",
                    (tgt, en, _difficulty_bucket(tgt)),
                )
                sent_n += 1
                if sent_n % 50000 == 0:
                    _p(0.35 + 0.35 * min(1.0, sent_n / 250000), f"parsing sentences ({sent_n})")
            db.commit()

        # 3. FTS index.
        _p(0.88, "indexing")
        db.execute("INSERT INTO vocab_fts(vocab_fts) VALUES('rebuild')")

        # 4. Metadata. ``tokenization = "whitespace"`` flips the runtime
        # tokenizer to the word-boundary path; ``pos_labels`` ships the
        # Wiktionary POS map so the UI renders human labels with no
        # hardcoded per-language JS table.
        meta = {
            "pack_kind": "language",
            "lang_code": lang_code,
            "schema": str(SCHEMA_VERSION),
            "name": name or f"{lang_code.upper()} language pack",
            "source": "Wiktionary (kaikki.org) + Tatoeba",
            "source_license": source_license,
            "build_date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "vocab_count": str(vocab_n),
            "sentence_count": str(sent_n),
            "inflection_count": str(infl_n),
            "tokenization": "whitespace",
            "pos_labels": json.dumps(WIKTIONARY_POS_LABELS, ensure_ascii=False),
            "level_system": "cefr",   # consumers can disambiguate ja/zh/cefr
        }
        db.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            list(meta.items()),
        )
        db.commit()
        _p(0.95, "finalizing")
        db.execute("VACUUM")
        _p(1.0, "done")
        return {"vocab": vocab_n, "sentences": sent_n, "inflections": infl_n}
    finally:
        db.close()


# ── CEDICT-sourced builder (Chinese) ─────────────────────────────────


def build_pack_cedict(
    *,
    out_path: Path,
    lang_code: str,
    cedict_txt: Path,
    tatoeba_sentences: Path | None = None,
    tatoeba_links: Path | None = None,
    hsk_txt: Path | None = None,
    name: str = "",
    source_license: str = "CC-CEDICT CC BY-SA 4.0; Tatoeba CC BY 2.0 FR; HSK 3.0 list MIT (elkmovie/hsk30)",
    tatoeba_lang: str = "cmn",
    limit_vocab: int | None = None,
    limit_sentences: int | None = None,
    progress=None,
) -> dict[str, int]:
    """Build a Chinese language pack from CC-CEDICT (text dump) plus,
    optionally, the Tatoeba ``cmn`` sentence corpus and the elkmovie HSK
    3.0 wordlist for level tagging.

    Shape parity with :func:`build_pack` (JMdict) — same SQLite schema,
    same query API, ``tokenization = "longest_prefix"`` so the runtime
    tokeniser uses the CJK click-into-the-character path instead of
    splitting on whitespace.

    Frequency: CEDICT itself doesn't ship frequency data. If we want
    rank-derived signals later we'd add a Chinese OpenSubtitles list
    (hermitdave has zh_cn / zh_tw). For now ``freq_rank`` stays NULL
    for everything except the HSK-leveled subset, whose level *is* the
    practical proxy for "where in the curriculum to drill this word".
    """
    def _p(frac: float, stage: str) -> None:
        if progress:
            try:
                progress(max(0.0, min(1.0, frac)), stage)
            except Exception as exc:
                # Progress callback failure shouldn't abort the build —
                # debug-log so a faulty install-job UI is findable.
                log.debug("lang_pack_progress_callback_failed", error=str(exc))

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hsk_levels = iter_hsk_levels(hsk_txt) if hsk_txt and hsk_txt.exists() else {}

    db = sqlite3.connect(str(out_path))
    try:
        _build_db_schema(db)

        # 1. Dictionary entries.
        _p(0.05, "parsing dictionary")
        vocab_n = 0
        for entry in iter_cedict_entries(cedict_txt):
            if limit_vocab is not None and vocab_n >= limit_vocab:
                break
            surface = entry["surface"]
            level = hsk_levels.get(surface)
            db.execute(
                """INSERT OR IGNORE INTO vocab
                       (word_id, surface, reading, pos, glosses, freq_rank, jlpt, level)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)""",
                (
                    entry["word_id"], surface, entry["reading"], entry["pos"],
                    json.dumps(entry["glosses"], ensure_ascii=False),
                    level,
                ),
            )
            vocab_n += 1
            if vocab_n % 20000 == 0:
                _p(0.05 + 0.4 * min(1.0, vocab_n / 120000), f"parsing dictionary ({vocab_n})")
        db.commit()

        # Count HSK-tagged entries so the pack metadata can advertise how
        # much of the curriculum was actually matched against the dict.
        hsk_tagged = 0
        if hsk_levels:
            cur = db.execute("SELECT COUNT(*) FROM vocab WHERE level IS NOT NULL")
            hsk_tagged = int(cur.fetchone()[0])

        # 2. Sentences.
        _p(0.50, "parsing sentences")
        sent_n = 0
        if (tatoeba_sentences and tatoeba_links
                and tatoeba_sentences.exists() and tatoeba_links.exists()):
            for tgt, en in iter_tatoeba_pairs(
                tatoeba_sentences, tatoeba_links, lang=tatoeba_lang,
            ):
                if limit_sentences is not None and sent_n >= limit_sentences:
                    break
                db.execute(
                    "INSERT INTO sentences (lang_text, en_text, difficulty) VALUES (?, ?, ?)",
                    (tgt, en, _difficulty_bucket(tgt)),
                )
                sent_n += 1
                if sent_n % 50000 == 0:
                    _p(0.50 + 0.35 * min(1.0, sent_n / 250000), f"parsing sentences ({sent_n})")
            db.commit()

        # 3. FTS index.
        _p(0.88, "indexing")
        db.execute("INSERT INTO vocab_fts(vocab_fts) VALUES('rebuild')")

        # 4. Metadata. Tokenization mirrors the JMdict builder — CJK
        # needs longest-prefix click-into-the-character resolution.
        meta = {
            "pack_kind": "language",
            "lang_code": lang_code,
            "schema": str(SCHEMA_VERSION),
            "name": name or f"{lang_code.upper()} language pack",
            "source": "CC-CEDICT + Tatoeba" + (" + HSK 3.0" if hsk_levels else ""),
            "source_license": source_license,
            "build_date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "vocab_count": str(vocab_n),
            "sentence_count": str(sent_n),
            "hsk_tagged_count": str(hsk_tagged),
            "tokenization": "longest_prefix",
            # CEDICT doesn't ship per-entry POS; the UI's POS label table
            # is JMdict-specific, so we don't supply one here.
            "level_system": "hsk",
        }
        db.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            list(meta.items()),
        )
        db.commit()
        _p(0.95, "finalizing")
        db.execute("VACUUM")
        _p(1.0, "done")
        return {"vocab": vocab_n, "sentences": sent_n, "hsk_tagged": hsk_tagged}
    finally:
        db.close()
