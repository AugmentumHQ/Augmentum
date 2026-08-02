"""Clean text for TTS — strip markdown, RP formatting, normalize numbers.

Extracted from pipeline.py for modularity.
"""

from __future__ import annotations

import re

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def clean_for_tts(text: str, is_narrative: bool = False, preserve_brackets: bool = False) -> str:
    """Strip markdown, RP formatting, LitRPG notation, and decorative symbols.

    Covers the full range of symbols found in roleplay, narrative fiction,
    LitRPG, and character card content that cause TTS engines to read
    symbols aloud, produce garbled output, or create unnatural pauses.
    """
    clean = text

    # --- Block-level removal (entire sections) ---

    # Fenced code blocks — remove entirely (stat blocks, code samples)
    clean = re.sub(r"```[\s\S]*?```", "", clean)
    # Indented code blocks (4+ spaces or tab at line start)
    clean = re.sub(r"^(?:    |\t).+$", "", clean, flags=re.MULTILINE)
    # HTML code/pre tags
    clean = re.sub(r"<(?:code|pre)[^>]*>[\s\S]*?</(?:code|pre)>", "", clean, flags=re.IGNORECASE)

    # LitRPG stat blocks and system notifications — remove entire lines
    # Matches: [Skill Acquired: Fireball], [[DUNGEON CLEARED]], <<Level Up!>>
    clean = re.sub(r"\[\[.*?\]\]", "", clean)
    clean = re.sub(r"<<.*?>>", "", clean)
    # Stat lines: HP: 150/200, EXP: 1250/2000, MP: 80/100 | STM: 45/50
    clean = re.sub(
        r"^[A-Z]{2,5}:\s*\d+\s*/\s*\d+.*$", "", clean, flags=re.MULTILINE,
    )
    # Stat modifiers: +5 STR, -3 DEF (standalone on a line)
    clean = re.sub(r"^[+-]\d+\s+[A-Z]{2,5}\b.*$", "", clean, flags=re.MULTILINE)
    # Progress bars (block elements, braille, circles)
    clean = re.sub(r"[█▓▒░▂▃▄▅▆▇⣿⣀●○◯⬛⬜🟩🟥🟢🔴❚═]{3,}", "", clean)

    # Decorative dividers and scene separators
    # Box-drawing lines: ─── ━━━ ═══ and mixed ornamental dividers
    clean = re.sub(r"[─━═┈┉╌╍╔╗╚╝╭╮╰╯┏┓┗┛┃│╿╽]{3,}", "", clean)
    # Sparkle/aesthetic dividers: ✧･ﾟ patterns, ⋇⋆✦⋆⋇, ☆○o。etc.
    clean = re.sub(r"[✧✦⋆⋇☆○。･ﾟ゜‧͙⁺˚꒦꒷꒰꒱︶₊⊹༶⛧♛]{3,}", "", clean)
    # Tibetan/decorative ornament separators: ༺ ༻ etc.
    clean = re.sub(r"[༺༻༶]+", "", clean)
    # Diamond/star pattern dividers: ◆◇◆◇◆
    clean = re.sub(r"[◆◇◈★☆✦✧⚬]{2,}", "", clean)
    # Heart/decorative dividers mixed with dashes: ━━ ♡ ━━
    clean = re.sub(r"[━═─]{2,}\s*[♡♥♫♪✦❖]\s*[━═─]{2,}", "", clean)

    # --- Inline code ---
    def _inline_code_sub(m: re.Match) -> str:
        content = m.group(1)
        if re.search(r"[_./\\{}()<>=;]|^\d+$|^[A-Z_]+$", content):
            return ""
        return content
    clean = re.sub(r"`([^`]+)`", _inline_code_sub, clean)

    # --- Citations, footnotes, and source references ---

    # Citation markers: [1], [2,3], [1-5], [n], [citation needed], [source]
    clean = re.sub(r"\[\d+(?:[,\s-]+\d+)*\]", "", clean)
    clean = re.sub(r"\[citation[^\]]*\]", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\[source[^\]]*\]", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\[ref[^\]]*\]", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\[note[^\]]*\]", "", clean, flags=re.IGNORECASE)
    # Footnote references: [^1], [^note]
    clean = re.sub(r"\[\^[^\]]+\]", "", clean)

    # Parenthesized citation numbers: (1), (2,3), (1-5)
    clean = re.sub(r"\(\d+(?:[,\s-]+\d+)*\)(?=[\s.,;:]|$)", "", clean)
    # Superscript numbers used as citations: ¹ ² ³ etc.
    clean = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", clean)
    # CJK bracket citations: 【1】, 〖source〗, 〔ref〕
    clean = re.sub(r"[【〖〔][^】〗〕]*[】〗〕]", "", clean)

    # Inline source attributions: (Source: ...), (via ...), (from ...)
    clean = re.sub(r"\(\s*(?:source|via|from|per|according to)[^)]{0,200}\)", "", clean, flags=re.IGNORECASE)
    # "According to [Source]" or "According to Source," — strip the attribution phrase
    clean = re.sub(r"(?:according to|as (?:reported|noted|stated|mentioned) (?:by|in|on))\s+(?:\[[^\]]*\]|[A-Z][\w\s]{1,40}(?:,|\.|\s—))", "", clean, flags=re.IGNORECASE)

    # Source/reference lines — single or multi-line sections at end of text
    # Matches: "Sources:", "References:", "Further reading:", "See also:" etc.
    clean = re.sub(
        r"^(?:Sources?|References?|Citations?|Further\s+reading|See\s+also|Bibliography|Works?\s+cited):\s*.*$",
        "", clean, flags=re.MULTILINE | re.IGNORECASE,
    )
    # Bulleted/numbered source lists: "- Source: ...", "1. https://...", "• Wikipedia"
    clean = re.sub(r"^[\s]*[-•*]\s*(?:Source|Reference|Citation|https?://)\S*.*$", "", clean, flags=re.MULTILINE | re.IGNORECASE)
    clean = re.sub(r"^\s*\d+\.\s*https?://\S+.*$", "", clean, flags=re.MULTILINE)

    # --- Markdown elements ---

    # Images
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", clean)
    # Links → text only
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    # URLs (with and without protocol)
    clean = re.sub(r"https?://\S+", "", clean)
    # Bare domain names: wikipedia.org, example.com/path, www.site.net
    clean = re.sub(r"\b(?:www\.)\S+", "", clean)
    clean = re.sub(r"\b\w+\.(?:com|org|net|edu|gov|io|co|info|wiki)\b(?:/\S*)?", "", clean)
    # Tables
    clean = re.sub(r"\|[^\n]+\|", "", clean)
    clean = re.sub(r"[-|:]{3,}", "", clean)
    # Horizontal rules
    clean = re.sub(r"^[-*_]{3,}\s*$", "", clean, flags=re.MULTILINE)
    # HTML entities
    clean = clean.replace("&amp;", "and").replace("&lt;", "less than").replace("&gt;", "greater than")
    clean = re.sub(r"&#?\w+;", "", clean)
    # HTML/XML tags: <br>, <i>, <b>, <thinking>, card wrapper tags
    clean = re.sub(r"<[^>]+>", " ", clean)
    # Blockquotes
    clean = re.sub(r"^>\s*", "", clean, flags=re.MULTILINE)
    # Headers
    clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)
    # Single-asterisk spans can be emphasis or RP actions; speak the content either way.
    def _handle_asterisk_content(m: re.Match) -> str:
        content = m.group(1).strip()
        return content

    # Bold/strong (**text** or ***text***) — always unwrap, never remove.
    # Bold is emphasis, not an action — the text should be spoken.
    clean = re.sub(r"\*{2,3}([^*]+?)\*{2,3}", r"\1", clean)
    # Single asterisks (*text*) are emphasis/actions; unwrap and speak them.
    clean = re.sub(r"\*([^*]+?)\*", _handle_asterisk_content, clean)
    clean = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", clean)
    clean = re.sub(r"~~([^~]+)~~", r"\1", clean)
    # List markers
    clean = re.sub(r"^\s*[-*+]\s+", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"^\s*\d+\.\s+", "", clean, flags=re.MULTILINE)

    # --- RP-specific formatting ---

    # OOC markers: ((out of character)), {ooc}, // ooc comment
    clean = re.sub(r"\(\(.*?\)\)", "", clean)
    clean = re.sub(r"^\s*//\s.*$", "", clean, flags=re.MULTILINE)
    # Unresolved macros: {{char}}, {{user}}, {{time}} etc.
    clean = re.sub(r"\{\{[^}]+\}\}", "", clean)
    # RP action hyphens: -sighs deeply- (but preserve normal hyphens in words)
    clean = re.sub(r"(?<!\w)-([^-\n]+)-(?!\w)", r"\1", clean)
    # Slash-style italics: /whispered words/
    clean = re.sub(r"(?<!\w)/([^/\n]+)/(?!\w)", r"\1", clean)

    # --- Japanese/CJK quote styles ---
    # 「brackets」→ content, 『double』→ content, «guillemets» → content
    clean = re.sub(r"[「」『』]", "", clean)
    clean = re.sub(r"[«»]", "", clean)

    # --- Musical notes (singing/humming) ---
    clean = re.sub(r"[♪♫♬♩]+", "", clean)

    # --- Tildes (playful tone markers, separators) ---
    clean = clean.replace("~", "")

    # [7] Text normalization — expand numbers, abbreviations, symbols BEFORE
    # special character stripping removes %, $, etc.
    clean = _normalize_for_speech(clean)

    # --- Remaining special symbols ---

    # Brackets, braces, angle brackets, and misc symbols (% removed — handled above)
    if preserve_brackets:
        # Keep [tag] brackets (e.g. [laugh], [cough] for Chatterbox Turbo)
        clean = re.sub(r"[(){}<>@^#]", "", clean)
    else:
        clean = re.sub(r"[\[\](){}<>@^#]", "", clean)
    # Leftover asterisks or underscores (unmatched formatting, trailing, etc.)
    clean = re.sub(r"\*+", "", clean)
    clean = re.sub(r"(?<!\w)_+(?!\w)", "", clean)
    # Decorative Unicode symbols that TTS reads aloud or garbles
    clean = re.sub(r"[◆◇◈★☆✦✧✨⚔🗡🔮⚡💀🔥❄☾☽⊹❖✄♡♥⚬]", "", clean)
    # Fullwidth text: Ｈｅｌｌｏ → Hello
    clean = re.sub(
        r"[\uFF01-\uFF5E]",
        lambda m: chr(ord(m.group()) - 0xFEE0),
        clean,
    )
    # Fullwidth space
    clean = clean.replace("\u3000", " ")
    # Halfwidth katakana, CJK decorative punctuation
    clean = re.sub(r"[\uFF61-\uFF9F\u309B-\u309E\u30FB]", "", clean)

    # Emoji (comprehensive Unicode emoji ranges)
    clean = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        r"\U0000FE00-\U0000FE0F\U0000200D\U0000200B-\U0000200F"
        r"\U00002028-\U00002029\U0000FEFF]+", "", clean,
    )

    # --- Smart punctuation normalization ---
    # Misaki's phonemizer (used by Kokoro) emits "words count mismatch" warnings
    # and silently drops/misaligns words when it sees smart quotes, unicode
    # ellipsis, or em/en dashes. Normalize to ASCII equivalents.
    clean = clean.translate(str.maketrans({
        "\u201c": '"', "\u201d": '"',  # curly double quotes
        "\u2018": "'", "\u2019": "'",  # curly single quotes / apostrophe
        "\u201a": ",", "\u201e": ",",  # low-9 quotes
        "\u2032": "'", "\u2033": '"',  # primes
        "\u2026": "...",               # horizontal ellipsis → three dots
        "\u2014": ", ", "\u2013": ", ",  # em/en dash → comma (natural pause)
        "\u2010": "-", "\u2011": "-", "\u2012": "-",  # other hyphens
    }))
    # Straight double quotes themselves can also confuse misaki when they hug
    # words — strip them (dialogue is conveyed by prosody, not "quote" readout).
    clean = clean.replace('"', '')

    # --- Whitespace cleanup ---

    # Collapse newlines into sentence breaks
    clean = re.sub(r"\n{2,}", ". ", clean)
    clean = re.sub(r"\n", " ", clean)
    clean = re.sub(r"\s{2,}", " ", clean)
    # Clean up punctuation artifacts from stripped blocks
    # Preserve ellipsis (...) — TTS engines use it for natural pauses
    clean = re.sub(r"\.{4,}", "...", clean)  # 4+ dots → ellipsis
    clean = re.sub(r"\.\s+\.", ".", clean)  # spaced dots (artifact) → single dot
    clean = re.sub(r"[,]{2,}", ",", clean)
    clean = re.sub(r"^\s*[.,;:]\s*", "", clean)  # leading punctuation

    if is_narrative:
        # Strip "CharacterName:" prefix at start of output
        clean = re.sub(r"^\w[\w\s]{0,30}:\s*", "", clean)

    # Script-aware routing (2026-06-12) — the universal choke point:
    # every TTS consumer (HTTP speech endpoints, voice WS sentence
    # chunks, previews, read-aloud) cleans through here. English-
    # pipeline voices letter-spell kana/han/hangul/cyrillic; romanize
    # deterministically where possible, drop from SPEECH otherwise.
    # Idempotent — romanized output is Latin and passes byte-identical
    # if a path cleans twice. See augmentum/voice/script_routing.py.
    try:
        from augmentum.voice.script_routing import prepare_speech_text
        clean, _ = prepare_speech_text(clean)
    except Exception:  # noqa: BLE001 — routing must never block speech
        pass

    return clean.strip()


# ---------------------------------------------------------------------------
# [7] Text Normalization for TTS
# ---------------------------------------------------------------------------

# Small numbers that TTS should speak as words (up to 100)
_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]

# Common abbreviations → spoken forms (case-INSENSITIVE — safe because
# every key here is unambiguous regardless of casing).
#
# Unit abbreviations (in., mm, kg, …) were REMOVED from this table
# 2026-06-11: they only expand when attached to a number now (see the
# number-unit rule in _normalize_for_speech). Standalone they collide
# with plain English — "Come on in." became "Come on inches", and the
# conversational filler "mm" / "mm-hmm" became "millimeters-hmm".
_ABBREVIATIONS: dict[str, str] = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Prof.": "Professor",
    "Sr.": "Senior",
    "Jr.": "Junior",
    "St.": "Saint",
    "vs.": "versus",
    "vs": "versus",
    "etc.": "etcetera",
    "approx.": "approximately",
    "dept.": "department",
    "govt.": "government",
    "est.": "established",
    "vol.": "volume",
    "avg.": "average",
    "qty.": "quantity",
}

# Gaming/RPG stat abbreviations — case-SENSITIVE (caps only). The old
# IGNORECASE matching expanded ordinary words: "convert to int" →
# "convert to intelligence", "con artist" → "constitution artist",
# "str" in code talk → "strength" (2026-06-11).
_ABBREVIATIONS_CASED: dict[str, str] = {
    "HP": "hit points",
    "MP": "mana points",
    "XP": "experience points",
    "EXP": "experience",
    "ATK": "attack",
    "DEF": "defense",
    "STR": "strength",
    "DEX": "dexterity",
    "INT": "intelligence",
    "WIS": "wisdom",
    "CHA": "charisma",
    "CON": "constitution",
    "LVL": "level",
    "LV": "level",
    "Lv.": "level",
}

# Build regexes for abbreviation matching (longest first to avoid
# partial matches). Use (?<!\w) instead of \b since \b doesn't work
# well with trailing periods.
_ABBR_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
_ABBR_CASED_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS_CASED, key=len, reverse=True)) + r")(?!\w)",
)


def _number_to_words(n: int) -> str:
    """Convert integer to English words (handles 0–999,999)."""
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _number_to_words(-n)

    parts: list[str] = []

    if n >= 1_000_000:
        return str(n)  # Too large — let TTS handle it

    if n >= 1000:
        parts.append(_number_to_words(n // 1000) + " thousand")
        n %= 1000

    if n >= 100:
        parts.append(_ONES[n // 100] + " hundred")
        n %= 100

    if n >= 20:
        word = _TENS[n // 10]
        if n % 10:
            word += "-" + _ONES[n % 10]
        parts.append(word)
    elif n > 0:
        parts.append(_ONES[n])

    return " ".join(parts)


def _expand_number(match: re.Match) -> str:
    """Expand a standalone number to words."""
    text = match.group(0)

    # Percentages: 42% → forty-two percent
    if text.endswith("%"):
        num_str = text[:-1].replace(",", "")
        try:
            n = float(num_str)
            if n == int(n) and 0 <= n <= 999_999:
                return _number_to_words(int(n)) + " percent"
            return num_str + " percent"
        except ValueError:
            return text

    # Currency: $42 → forty-two dollars, $3.50 → three dollars fifty
    if text.startswith("$"):
        num_str = text[1:].replace(",", "")
        try:
            n = float(num_str)
            if "." in num_str:
                dollars = int(n)
                cents = round((n - dollars) * 100)
                result = _number_to_words(dollars) + " dollar" + ("s" if dollars != 1 else "")
                if cents:
                    result += " " + _number_to_words(cents) + " cent" + ("s" if cents != 1 else "")
                return result
            if 0 <= n <= 999_999:
                word = _number_to_words(int(n))
                return word + " dollar" + ("s" if int(n) != 1 else "")
        except ValueError:
            pass
        return text

    # Plain numbers
    num_str = text.replace(",", "")
    try:
        n = float(num_str)
        if "." in num_str:
            # Decimal: read as "three point five"
            int_part, dec_part = num_str.split(".", 1)
            int_val = int(int_part) if int_part else 0
            if 0 <= int_val <= 999_999:
                result = _number_to_words(int_val) + " point"
                for digit in dec_part:
                    if digit.isdigit():
                        result += " " + (_ONES[int(digit)] if int(digit) > 0 else "zero")
                return result
        elif 0 <= n <= 999_999 and n == int(n):
            return _number_to_words(int(n))
    except ValueError:
        pass
    return text


# Units that expand ONLY next to a number ("5 mm", "60mph"). None of
# these are common English words, so a space between number and unit
# is safe. "in" is deliberately ABSENT — "at 12 in the morning" is
# real English; "in" expands only when glued to the number ("5in").
_UNIT_MAP = {
    "mph": "miles per hour", "kph": "kilometers per hour",
    "km": "kilometers", "kg": "kilograms", "cm": "centimeters",
    "mm": "millimeters", "ft": "feet", "lb": "pounds", "lbs": "pounds",
    "oz": "ounces", "in": "inches",
}
_NUMBER_UNIT_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*(mph|kph|km|kg|cm|mm|ft|lbs|lb|oz)(?!\w)",
)
_NUMBER_IN_GLUED_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)(in)(?!\w)")


# Operator lexicon — voice_tts_lexicon setting (JSON object). Keys are
# terms (matched case-insensitively on word boundaries); values are
# the spoken form. An EMPTY value means "never touch this term" — it's
# shielded from all built-in normalization. Compiled patterns cached
# against the raw setting string so per-sentence cost is one compare.
_lexicon_cache: tuple[str, list[tuple[re.Pattern, str]]] = ("", [])


def _user_lexicon() -> list[tuple[re.Pattern, str]]:
    try:
        from augmentum.config import settings
        raw = (getattr(settings, "voice_tts_lexicon", "") or "").strip()
    except Exception:  # noqa: BLE001 — config unavailable in some tests
        return []
    global _lexicon_cache
    if raw == _lexicon_cache[0]:
        return _lexicon_cache[1]
    compiled: list[tuple[re.Pattern, str]] = []
    if raw:
        try:
            import json
            data = json.loads(raw)
            if isinstance(data, dict):
                for term, spoken in data.items():
                    term = str(term).strip()
                    if not term:
                        continue
                    compiled.append((
                        re.compile(
                            rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE,
                        ),
                        str(spoken),
                    ))
        except Exception:  # noqa: BLE001 — bad JSON = no lexicon
            log.warning("voice_tts_lexicon_invalid_json")
    _lexicon_cache = (raw, compiled)
    return compiled


def _normalize_for_speech(text: str) -> str:
    """Expand numbers, abbreviations, and symbols for natural TTS output."""
    result = text

    # [0] Operator lexicon — highest priority. Replacements apply
    # before any built-in rule; empty-value terms are shielded behind
    # sentinels so no built-in pass can touch them, then restored.
    protected: list[str] = []
    for pattern, spoken in _user_lexicon():
        if spoken == "":
            # Sentinel must contain NO digits/words — every later pass
            # (number expansion, abbreviations) ignores \x00 + letters
            # wrapped in \x00. A digit-bearing sentinel got its index
            # number-expanded ("\x000\x00" → "zero") and the restore
            # lookup missed.
            def _shield(m: re.Match, _p: list = protected) -> str:
                _p.append(m.group(0))
                return "\x00" + "q" * len(_p) + "\x00"
            result = pattern.sub(_shield, result)
        else:
            result = pattern.sub(spoken, result)

    # Expand abbreviations (case-preserving match)
    def _abbr_replace(m: re.Match) -> str:
        key = m.group(0)
        # Try exact match first, then case-insensitive
        replacement = _ABBREVIATIONS.get(key)
        if replacement is None:
            for k, v in _ABBREVIATIONS.items():
                if k.lower() == key.lower():
                    replacement = v
                    break
        return replacement or key

    result = _ABBR_PATTERN.sub(_abbr_replace, result)
    # Gaming stats — exact case only ("INT 18" speaks; "convert to
    # int" doesn't).
    result = _ABBR_CASED_PATTERN.sub(
        lambda m: _ABBREVIATIONS_CASED.get(m.group(0), m.group(0)), result,
    )

    # Expand number-attached units: "60mph" / "5 mm" → spoken unit.
    # The digits are left alone — the generic number pass right below
    # wordifies them, so this rule only owns the unit token.
    def _expand_number_unit(m: re.Match) -> str:
        return m.group(1) + " " + _UNIT_MAP.get(m.group(2), m.group(2))

    result = _NUMBER_UNIT_RE.sub(_expand_number_unit, result)
    result = _NUMBER_IN_GLUED_RE.sub(_expand_number_unit, result)

    # Expand numbers: $42, 42%, 3.14, 1,000 (but not dates, times, IDs)
    # Only match standalone numbers, not part of timestamps or version strings
    result = re.sub(
        r"\$\d[\d,.]*|\b\d[\d,.]*%|\b\d{1,6}(?:,\d{3})*(?:\.\d+)?\b(?![\d:/.-])",
        _expand_number,
        result,
    )

    # Common symbols → words
    result = result.replace(" & ", " and ")
    result = result.replace(" + ", " plus ")
    result = result.replace(" = ", " equals ")
    result = result.replace(" > ", " greater than ")
    result = result.replace(" < ", " less than ")
    result = result.replace(" >= ", " greater than or equal to ")
    result = result.replace(" <= ", " less than or equal to ")
    result = result.replace(" != ", " not equal to ")
    result = result.replace(" @ ", " at ")
    # Keep ellipsis as-is — most TTS engines (including Kokoro) treat ... as a natural pause.
    # Converting to comma removes the dramatic pause effect.
    # result = result.replace("...", ", ")  # DISABLED — ellipses are useful for TTS pacing

    # Ordinals: 1st, 2nd, 3rd, 4th etc.
    # Ordinals: 1st→first, 2nd→second, 3rd→third, 4th→fourth, 21st→twenty-first
    _ORDINAL_MAP = {1: "first", 2: "second", 3: "third", 5: "fifth",
                    8: "eighth", 9: "ninth", 12: "twelfth"}

    def _to_ordinal(m: re.Match) -> str:
        n = int(m.group(1))
        if n in _ORDINAL_MAP:
            return _ORDINAL_MAP[n]
        word = _number_to_words(n)
        # Compound: twenty-one → twenty-first
        if "-" in word:
            base, last = word.rsplit("-", 1)
            last_n = n % 10
            if last_n in _ORDINAL_MAP:
                return base + "-" + _ORDINAL_MAP[last_n]
            if last.endswith("y"):
                return base + "-" + last[:-1] + "ieth"
            return base + "-" + last + "th"
        if word.endswith("y"):
            return word[:-1] + "ieth"
        if word.endswith("e"):
            return word[:-1] + "th"  # nine→ninth handled above
        return word + "th"

    result = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _to_ordinal, result)

    # Restore lexicon-shielded terms (verbatim, normalization-proof).
    # Longest sentinel first so "qq" doesn't partially match inside "qqq".
    for i in range(len(protected) - 1, -1, -1):
        result = result.replace(
            "\x00" + "q" * (i + 1) + "\x00", protected[i],
        )

    return result
