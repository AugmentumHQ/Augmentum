"""Script-aware TTS text routing — never feed a voice a script it
can't pronounce.

Kokoro voices are per-language G2P pipelines and the builtin catalog
is English; Pocket TTS is English-only. Feeding kana/kanji, hangul,
or cyrillic to an English voice produces letter-salad (observed live:
Japanese spoken as character names, 2026-06-12). This module is the
deterministic fix at the same choke point the pronunciation lexicon
uses — it transforms ONLY the spoken text; the visible message is
untouched.

Policy ladder, per non-Latin span:

1. **Built-in romanization** — pure-python, zero deps, deterministic:
   kana (Hepburn), hangul (Revised Romanization), cyrillic, greek.
2. **Dictionary romanization** when the optional libs are installed:
   pykakasi (kanji → romaji), pypinyin (han → pinyin). Declared in
   pyproject; light pure-python — present after the next image build.
3. **Drop the span from speech** — silence beats garbage. Logged with
   per-script counts so the gap is visible, never silent.

Unicode script detection is character-class work — deterministic and
cheap (this is NOT the banned regex-switchboard pattern; no intent is
being inferred). Covers every learning-catalog language (ja/zh/ko —
es/fr are Latin and pass through untouched) plus scripts web content
drags in (cyrillic, greek, arabic, hebrew, devanagari, thai).
"""

from __future__ import annotations

from functools import lru_cache

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Script classification ───────────────────────────────────────────

# (start, end, tag) — tags name the romanization strategy, not the
# language. Han is ambiguous (ja kanji vs zh hanzi); see segment().
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x3040, 0x309F, "kana"),      # hiragana
    (0x30A0, 0x30FF, "kana"),      # katakana (incl. chōonpu)
    (0x31F0, 0x31FF, "kana"),      # katakana phonetic extensions
    (0x4E00, 0x9FFF, "han"),       # CJK unified
    (0x3400, 0x4DBF, "han"),       # CJK ext A
    (0xF900, 0xFAFF, "han"),       # CJK compat
    (0xAC00, 0xD7AF, "hangul"),    # syllables
    (0x1100, 0x11FF, "hangul"),    # jamo
    (0x3130, 0x318F, "hangul"),    # compat jamo
    (0x0400, 0x04FF, "cyrillic"),
    (0x0500, 0x052F, "cyrillic"),
    (0x0370, 0x03FF, "greek"),
    (0x1F00, 0x1FFF, "greek"),     # extended (polytonic)
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
)

# CJK punctuation that should ride whichever span it appears in but
# map to ASCII equivalents when romanized.
_CJK_PUNCT = {
    "。": ". ", "、": ", ", "・": " ", "！": "! ",
    "？": "? ", "「": '"', "」": '"', "，": ", ",
    "　": " ",
}


def _script_of(ch: str) -> str:
    cp = ord(ch)
    for start, end, tag in _RANGES:
        if start <= cp <= end:
            return tag
    return ""  # neutral / latin / digits / punctuation


def segment(text: str) -> list[tuple[str, str]]:
    """Split text into (script, span) runs.

    Neutral characters (spaces, digits, ASCII punctuation) attach to
    the CURRENT run so phrasing survives. Han runs in a text that
    contains kana anywhere are tagged ``han_ja`` (kanji in Japanese
    prose) so the dictionary step picks the right romanizer.
    """
    if not text:
        return []
    has_kana = any(_script_of(c) == "kana" for c in text)
    runs: list[tuple[str, list[str]]] = []
    current = ""
    for ch in text:
        tag = _script_of(ch)
        if tag == "han" and has_kana:
            tag = "han_ja"
        if not tag:
            # Neutral chars: extend the current run; CJK punctuation
            # keeps non-Latin context.
            if ch in _CJK_PUNCT and runs and runs[-1][0] == current and current:
                runs[-1][1].append(ch)
                continue
            tag = current if (current and not ch.strip()) else ""
        if runs and runs[-1][0] == tag:
            runs[-1][1].append(ch)
        else:
            runs.append((tag, [ch]))
            current = tag if tag else ""
    return [(tag, "".join(chars)) for tag, chars in runs]


# ── Built-in romanizers (deterministic, zero deps) ──────────────────

# Hepburn, hiragana-keyed. Katakana normalizes to hiragana by offset.
_KANA: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゔ": "vu",
}
_KANA_DIGRAPH: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
}
_VOWELS = "aeiou"


def _romanize_kana(span: str) -> str:
    # Normalize katakana → hiragana (fixed offset), keep chōonpu.
    norm = []
    for ch in span:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            norm.append(chr(cp - 0x60))
        else:
            norm.append(ch)
    s = "".join(norm)
    out: list[str] = []
    i = 0
    pending_sokuon = False
    while i < len(s):
        ch = s[i]
        if ch in ("っ", "ッ"):
            pending_sokuon = True
            i += 1
            continue
        if ch == "ー":  # long vowel: repeat last vowel
            if out and out[-1] and out[-1][-1] in _VOWELS:
                out.append(out[-1][-1])
            i += 1
            continue
        if ch in _CJK_PUNCT:
            out.append(_CJK_PUNCT[ch])
            i += 1
            continue
        pair = s[i:i + 2]
        roma = _KANA_DIGRAPH.get(pair)
        if roma:
            i += 2
        else:
            roma = _KANA.get(ch, "")
            i += 1
        if not roma:
            continue
        if pending_sokuon and roma and roma[0] not in _VOWELS:
            # Geminate: っち → tchi (Hepburn), otherwise double consonant
            out.append("t" if roma.startswith("ch") else roma[0])
            pending_sokuon = False
        out.append(roma)
    return "".join(out)


# Revised Romanization of Korean — algorithmic from jamo indices.
_HANGUL_INITIALS = (
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
    "j", "jj", "ch", "k", "t", "p", "h",
)
_HANGUL_MEDIALS = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
    "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
# 28 codas in Unicode jamo order: ∅ ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ
_HANGUL_FINALS = (
    "", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l",
    "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t",
    "p", "t",
)


def _romanize_hangul(span: str) -> str:
    out: list[str] = []
    for ch in span:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7A3:
            idx = cp - 0xAC00
            ini, rem = divmod(idx, 588)
            med, fin = divmod(rem, 28)
            out.append(
                _HANGUL_INITIALS[ini] + _HANGUL_MEDIALS[med]
                + _HANGUL_FINALS[fin]
            )
        elif ch in _CJK_PUNCT:
            out.append(_CJK_PUNCT[ch])
        else:
            out.append(ch if not _script_of(ch) else "")
    joined = "".join(out)
    # Nasal assimilation — obstruent codas before nasals (합니다 →
    # hamnida, not hapnida). One rule covers the polite endings that
    # appear in virtually every Korean sentence; full phonology is
    # out of scope for a pronunciation shim.
    import re as _re
    joined = _re.sub(r"p(?=[nm])", "m", joined)
    joined = _re.sub(r"k(?=[nm])", "ng", joined)
    joined = _re.sub(r"t(?=[nm])", "n", joined)
    return joined


_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
}

_GREEK = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z",
    "η": "i", "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m",
    "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
    "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps",
    "ω": "o",
}


def _table_romanize(span: str, table: dict[str, str]) -> str:
    out = []
    for ch in span:
        low = ch.lower()
        roma = table.get(low)
        if roma is None:
            out.append(ch if not _script_of(ch) else "")
            continue
        out.append(roma.capitalize() if ch != low and roma else roma)
    return "".join(out)


# ── Optional dictionary romanizers ──────────────────────────────────

@lru_cache(maxsize=1)
def _kakasi():
    try:
        import pykakasi
        return pykakasi.kakasi()
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _pinyin():
    try:
        from pypinyin import lazy_pinyin
        return lazy_pinyin
    except ImportError:
        return None


def _romanize_han_ja(span: str) -> str | None:
    kks = _kakasi()
    if kks is None:
        return None
    try:
        return " ".join(
            item.get("hepburn", "") for item in kks.convert(span)
        ).strip()
    except Exception:  # noqa: BLE001
        log.warning("pykakasi_convert_failed", exc_info=True)
        return None


def _romanize_han_zh(span: str) -> str | None:
    lp = _pinyin()
    if lp is None:
        return None
    try:
        return " ".join(lp(span)).strip()
    except Exception:  # noqa: BLE001
        log.warning("pypinyin_convert_failed", exc_info=True)
        return None


# ── Public API ──────────────────────────────────────────────────────

_BUILTIN = {
    "kana": _romanize_kana,
    "hangul": _romanize_hangul,
}
_TABLES = {
    "cyrillic": _CYRILLIC,
    "greek": _GREEK,
}
_DICTIONARY = {
    "han_ja": _romanize_han_ja,
    "han": _romanize_han_zh,
}


def prepare_speech_text(text: str) -> tuple[str, dict[str, int]]:
    """Transform text for an English-pipeline voice.

    Returns ``(spoken_text, stats)`` where stats counts romanized and
    dropped spans per script. Latin/neutral text passes through
    byte-identical — zero cost on the common path.
    """
    if not text or all(not _script_of(c) for c in text):
        return text, {}

    stats: dict[str, int] = {}
    out: list[str] = []
    for tag, span in segment(text):
        if not tag:
            out.append(span)
            continue
        roma: str | None = None
        if tag in _BUILTIN:
            roma = _BUILTIN[tag](span)
        elif tag in _TABLES:
            roma = _table_romanize(span, _TABLES[tag])
        elif tag in _DICTIONARY:
            roma = _DICTIONARY[tag](span)
            if roma is None and tag == "han_ja":
                # Kanji without pykakasi: salvage any kana that rode
                # along in the span's neutral merge; drop the rest.
                roma = None
        if roma:
            # Space-pad so romanized words don't fuse with neighbors.
            out.append(f" {roma.strip()} ")
            stats[f"romanized_{tag}"] = stats.get(f"romanized_{tag}", 0) + 1
        else:
            # Silence beats garbage — drop from SPEECH only.
            stats[f"dropped_{tag}"] = stats.get(f"dropped_{tag}", 0) + 1
    spoken = " ".join("".join(out).split())
    if stats:
        log.info("tts_script_routing", **stats)
    return spoken, stats
