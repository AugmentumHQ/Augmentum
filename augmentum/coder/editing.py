"""5-tier SEARCH/REPLACE edit matching engine for Coder mode.

Tiers (tried in order):
  1. exact         — character-for-character substring match
  2. whitespace    — strip trailing whitespace per line, collapse consecutive
                     blank lines; match after normalisation but splice at the
                     original position in the file
  3. indentation   — strip common leading indent from every line of the search
                     block, compare trimmed content, then re-apply the file's
                     actual indentation to the replacement
  4. unicode       — fold smart quotes / en-em dashes / NBSP to their ASCII
                     equivalents on both sides, then retry exact. Catches
                     "LLM wrote plain ASCII, file has curly quotes" — a real
                     failure mode when files are touched by docs editors.
                     (Ported from Codex's apply_patch seek_sequence.rs.)
  5. fuzzy         — sliding-window difflib.SequenceMatcher; returns a result
                     but the tier name signals the caller to confirm

Public API
----------
apply_edit(content, search, replace, *, fuzzy_threshold=0.6)
    -> (new_content | None, tier_name)

parse_search_replace_blocks(text)
    -> list[tuple[search, replace, filename | None]]
"""
from __future__ import annotations

import difflib
import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _common_indent(lines: list[str]) -> str:
    """Return the longest common leading whitespace shared by all non-empty lines."""
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return ""
    indent = re.match(r"^(\s*)", non_empty[0]).group(1)
    for line in non_empty[1:]:
        m = re.match(r"^(\s*)", line)
        candidate = m.group(1) if m else ""
        # Shrink to common prefix
        max_len = min(len(indent), len(candidate))
        i = 0
        while i < max_len and indent[i] == candidate[i]:
            i += 1
        indent = indent[:i]
        if not indent:
            break
    return indent


def _strip_trailing(lines: list[str]) -> list[str]:
    """Strip trailing whitespace from each line (right-strip)."""
    return [ln.rstrip() for ln in lines]


def _collapse_blank_runs(lines: list[str]) -> list[str]:
    """Collapse runs of 2+ consecutive blank lines to a single blank line."""
    result: list[str] = []
    prev_blank = False
    for ln in lines:
        is_blank = ln.strip() == ""
        if is_blank and prev_blank:
            continue
        result.append(ln)
        prev_blank = is_blank
    return result


def _normalise_ws(lines: list[str]) -> list[str]:
    """Strip trailing whitespace and collapse blank-line runs."""
    return _collapse_blank_runs(_strip_trailing(lines))


# ---------------------------------------------------------------------------
# Tier 1: Exact
# ---------------------------------------------------------------------------

def _try_exact(content: str, search: str, replace: str) -> str | None:
    """Return the updated content if *search* appears verbatim in *content*."""
    if search in content:
        return content.replace(search, replace, 1)
    return None


# ---------------------------------------------------------------------------
# Tier 2: Whitespace-normalised
# ---------------------------------------------------------------------------

def _try_whitespace(content: str, search: str, replace: str) -> str | None:
    """Match after normalising trailing whitespace and blank-line runs.

    Strategy:
      - Normalise both content and search.
      - Find the normalised search inside the normalised content (as a
        contiguous block of lines).
      - Map the match back to the original content line indices and splice.
    """
    c_lines = content.splitlines(keepends=True)
    s_lines = search.splitlines(keepends=True)

    c_norm = _normalise_ws([ln.rstrip("\n\r") for ln in c_lines])
    s_norm = _normalise_ws([ln.rstrip("\n\r") for ln in s_lines])

    ns = len(s_norm)
    nc = len(c_norm)

    if ns == 0 or nc < ns:
        return None

    for i in range(nc - ns + 1):
        if c_norm[i : i + ns] == s_norm:
            # Verify this isn't just an indentation difference (tabs vs spaces).
            # We only claim the whitespace tier when the *indent* of corresponding
            # lines is identical (same chars, same length); the difference must
            # be trailing-only or blank-line count.
            indent_ok = True
            for j, sl in enumerate(s_lines[:ns]):
                cl = c_lines[i + j] if i + j < len(c_lines) else ""
                # Leading whitespace comparison (characters, not just existence)
                s_lead = re.match(r"^(\s*)", sl.rstrip("\n\r")).group(1)
                c_lead = re.match(r"^(\s*)", cl.rstrip("\n\r")).group(1)
                if sl.strip() and cl.strip() and s_lead != c_lead:
                    indent_ok = False
                    break
            if not indent_ok:
                continue

            # Determine line ending used in content
            ending = "\n"
            if c_lines and c_lines[0].endswith("\r\n"):
                ending = "\r\n"
            elif c_lines and c_lines[0].endswith("\r"):
                ending = "\r"

            r_lines = replace.splitlines(keepends=False)
            # Rebuild replacement preserving the content's line ending
            replaced_block = [rl + ending for rl in r_lines]
            # If the matched region ended without a newline, strip last ending
            last_orig = c_lines[i + ns - 1] if i + ns - 1 < len(c_lines) else ""
            if not last_orig.endswith(("\n", "\r")) and replaced_block:
                replaced_block[-1] = replaced_block[-1].rstrip("\r\n")

            new_lines = c_lines[:i] + replaced_block + c_lines[i + ns:]
            return "".join(new_lines)

    return None


# ---------------------------------------------------------------------------
# Tier 3: Indentation-preserving
# ---------------------------------------------------------------------------

def _try_indentation(content: str, search: str, replace: str) -> str | None:
    """Match ignoring absolute indentation; preserve relative indent on output.

    Algorithm:
      1. Strip the *common* leading indent from all non-empty lines in
         *search*, giving a "de-indented" search pattern.
      2. Slide through the content lines; for each candidate window strip its
         common indent and compare de-indented forms.
      3. On match, take the *content's* base indent for the first matched line
         and rebase the replacement relative to that indent.
    """
    c_lines_raw = content.splitlines(keepends=True)
    s_lines_raw = search.splitlines(keepends=False)

    s_common = _common_indent(s_lines_raw)
    s_deindent = [
        ln[len(s_common):] if ln.startswith(s_common) else ln.lstrip()
        for ln in s_lines_raw
    ]

    ns = len(s_deindent)
    nc = len(c_lines_raw)

    if ns == 0 or nc < ns:
        return None

    for i in range(nc - ns + 1):
        window = [c_lines_raw[i + j].rstrip("\n\r") for j in range(ns)]
        w_common = _common_indent(window)
        w_deindent = [
            ln[len(w_common):] if ln.startswith(w_common) else ln.lstrip()
            for ln in window
        ]
        if w_deindent != s_deindent:
            continue

        # Match found — rebase replacement to the content's indent
        r_lines_raw = replace.splitlines(keepends=False)
        r_common = _common_indent(r_lines_raw)

        ending = "\n"
        if c_lines_raw and c_lines_raw[0].endswith("\r\n"):
            ending = "\r\n"
        elif c_lines_raw and c_lines_raw[0].endswith("\r"):
            ending = "\r"

        rebased: list[str] = []
        for rl in r_lines_raw:
            if not rl.strip():
                rebased.append(rl + ending)
            else:
                # Strip the replace block's own common indent, then add the
                # file's base indent for this match location.
                stripped = rl[len(r_common):] if rl.startswith(r_common) else rl.lstrip()
                rebased.append(w_common + stripped + ending)

        last_orig = c_lines_raw[i + ns - 1]
        if not last_orig.endswith(("\n", "\r")) and rebased:
            rebased[-1] = rebased[-1].rstrip("\r\n")

        new_lines = c_lines_raw[:i] + rebased + c_lines_raw[i + ns:]
        return "".join(new_lines)

    return None


# ---------------------------------------------------------------------------
# Tier 4: Unicode normalization
# ---------------------------------------------------------------------------
#
# Port of Codex's 4th seek-sequence tier (codex-rs/apply-patch/src/
# seek_sequence.rs). Catches the "ASCII patch vs smart-quoted source"
# failure: the LLM emits a search block with straight quotes / hyphens,
# but the actual file was run through a smart-quote-replacing editor or
# copied from a Word doc. A deterministic Unicode fold is cheaper and
# more predictable than sending it through fuzzy, so it lives *between*
# indentation and fuzzy — still a last-resort-before-probabilistic tier.
#
# Kept in sync with Codex's table; expand here when we see a new failure
# mode (the test suite is the source of truth for what's in-scope).

_UNICODE_FOLD_MAP = {
    # En dash, em dash, horizontal bar, minus sign, bullet operators → hyphen
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    # Smart double quotes → straight
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    # Smart single quotes / apostrophes → straight
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    # Non-breaking space, thin space, zero-width space, BOM → regular space / empty
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\u200b": "", "\ufeff": "",
    # Ellipsis → three dots (so the ... elision syntax keeps working)
    "\u2026": "...",
}


def _unicode_fold(text: str) -> str:
    """Replace every character in _UNICODE_FOLD_MAP with its ASCII form.

    Kept deliberately narrow — we only fold characters LLMs get wrong,
    not a full NFKC normalisation. NFKC would also fold 'ﬁ' → 'fi' which
    can silently break source code that intentionally contains ligatures
    (rare, but this matcher shouldn't paper over that).
    """
    if not any(ch in text for ch in _UNICODE_FOLD_MAP):
        return text
    return "".join(_UNICODE_FOLD_MAP.get(ch, ch) for ch in text)


def _try_unicode(content: str, search: str, replace: str) -> str | None:
    """Fold Unicode curly/dash punctuation on both sides, then retry exact.

    The file's actual bytes are preserved in the output — we only use the
    folded form for *locating* the match; the replacement splice happens
    against the original content at the original offset.
    """
    if not search:
        return None
    folded_search = _unicode_fold(search)
    folded_content = _unicode_fold(content)
    # No-op if neither side changed under fold.
    if folded_search == search and folded_content == content:
        return None
    idx = folded_content.find(folded_search)
    if idx < 0:
        return None
    # The folded string has the same char *count* as the original (every
    # entry in _UNICODE_FOLD_MAP is 1:1 except '\u2026' → "..." and
    # zero-width strips). Handle length changes safely by counting
    # characters in the original that correspond to the folded span.
    # Fast path: most items are 1:1, zero-width strips are rare.
    end = idx + len(folded_search)
    # If any non-1:1 folds happened, recompute the actual span in content
    # by walking character-by-character.
    if any(
        ch in _UNICODE_FOLD_MAP and len(_UNICODE_FOLD_MAP[ch]) != 1
        for ch in content
    ):
        cursor = 0
        folded_cursor = 0
        start_in_content: int | None = None
        end_in_content: int | None = None
        while cursor < len(content) and folded_cursor <= end:
            if folded_cursor == idx and start_in_content is None:
                start_in_content = cursor
            if folded_cursor == end:
                end_in_content = cursor
                break
            ch = content[cursor]
            folded_cursor += len(_UNICODE_FOLD_MAP.get(ch, ch))
            cursor += 1
        if start_in_content is None or end_in_content is None:
            return None
        return content[:start_in_content] + replace + content[end_in_content:]
    # Simple 1:1 path
    return content[:idx] + replace + content[end:]


# ---------------------------------------------------------------------------
# Tier 5: Fuzzy
# ---------------------------------------------------------------------------

def _try_fuzzy(
    content: str,
    search: str,
    replace: str,
    threshold: float,
) -> str | None:
    """Sliding-window SequenceMatcher over content lines.

    Finds the window of *len(search_lines)* content lines whose ratio against
    the search block exceeds *threshold*; replaces that window.
    """
    c_lines = content.splitlines(keepends=True)
    s_lines = search.splitlines(keepends=True)
    ns = len(s_lines)
    nc = len(c_lines)

    if ns == 0 or nc == 0:
        return None

    search_str = "".join(s_lines)
    best_ratio = 0.0
    best_i = -1

    for i in range(nc - ns + 1):
        window = "".join(c_lines[i : i + ns])
        ratio = difflib.SequenceMatcher(None, search_str, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_i = i

    if best_ratio < threshold or best_i == -1:
        return None

    ending = "\n"
    if c_lines and c_lines[0].endswith("\r\n"):
        ending = "\r\n"
    elif c_lines and c_lines[0].endswith("\r"):
        ending = "\r"

    r_lines = replace.splitlines(keepends=False)
    replaced_block = [rl + ending for rl in r_lines]
    last_orig = c_lines[best_i + ns - 1] if best_i + ns - 1 < len(c_lines) else ""
    if not last_orig.endswith(("\n", "\r")) and replaced_block:
        replaced_block[-1] = replaced_block[-1].rstrip("\r\n")

    new_lines = c_lines[:best_i] + replaced_block + c_lines[best_i + ns:]
    return "".join(new_lines)


# ---------------------------------------------------------------------------
# Public: apply_edit
# ---------------------------------------------------------------------------

def apply_edit(
    content: str,
    search: str,
    replace: str,
    *,
    fuzzy_threshold: float = 0.6,
) -> tuple[str | None, str]:
    """Apply a SEARCH/REPLACE edit to *content* using a 4-tier matching chain.

    Parameters
    ----------
    content:
        The full text of the file to edit.
    search:
        The block of text to locate in *content*.
    replace:
        The text to substitute in place of *search*.
    fuzzy_threshold:
        Minimum SequenceMatcher ratio for the fuzzy tier (default 0.6).

    Returns
    -------
    (new_content, tier_name)
        *new_content* is the edited file text, or ``None`` if no match was
        found.  *tier_name* is one of ``"exact"``, ``"whitespace"``,
        ``"indentation"``, ``"fuzzy"``, or ``"none"``.
    """
    # Guard: empty search string is meaningless
    if not search:
        return None, "none"

    # Guard: nothing to search in
    if not content:
        return None, "none"

    result = _try_exact(content, search, replace)
    if result is not None:
        return result, "exact"

    result = _try_whitespace(content, search, replace)
    if result is not None:
        return result, "whitespace"

    result = _try_indentation(content, search, replace)
    if result is not None:
        return result, "indentation"

    # Deterministic Unicode fold — catches smart-quote / en-dash mismatches
    # before falling back to probabilistic fuzzy match.
    result = _try_unicode(content, search, replace)
    if result is not None:
        return result, "unicode"

    result = _try_fuzzy(content, search, replace, fuzzy_threshold)
    if result is not None:
        return result, "fuzzy"

    return None, "none"


# ---------------------------------------------------------------------------
# Public: parse_search_replace_blocks
# ---------------------------------------------------------------------------

# Regex fragments ----------------------------------------------------------------
# Opening marker: 3+ of '<' optionally followed by whitespace and SEARCH (case-insensitive)
_OPEN = r"<{3,}\.?\s*SEARCH"
# Separator: 3+ of '=' on its own line
_SEP = r"={3,}"
# Closing marker: 3+ of '>' optionally followed by whitespace and REPLACE
_CLOSE = r">{3,}\.?\s*REPLACE"

# FILE section header: === FILE: path ===  (flexible = count)
_FILE_HEADER = re.compile(r"={3,}\s*FILE:\s*(.+?)\s*={3,}", re.IGNORECASE)

# Single SEARCH/REPLACE block (content groups may be empty)
_BLOCK_RE = re.compile(
    _OPEN + r"\s*\n([\s\S]*?)\n?" + _SEP + r"\s*\n([\s\S]*?)\n?" + _CLOSE,
    re.IGNORECASE,
)


class _Block(NamedTuple):
    search: str
    replace: str
    filename: str | None


def parse_search_replace_blocks(text: str) -> list[tuple[str, str, str | None]]:
    """Parse SEARCH/REPLACE blocks from LLM output.

    Supports two formats:

    **Bare blocks** (no file header)::

        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE

    **FILE-wrapped blocks**::

        === FILE: src/main.py ===
        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE

    Flexible delimiters (any number ≥ 3 of ``<``, ``=``, ``>``) are accepted.
    Keywords ``SEARCH`` and ``REPLACE`` are matched case-insensitively.

    Returns
    -------
    list of (search, replace, filename | None)
        *filename* is ``None`` for bare blocks.
    """
    results: list[tuple[str, str, str | None]] = []

    # Split text into FILE sections and a leading "no-file" section.
    # We find all FILE header positions, then process each region.
    file_header_positions: list[tuple[int, str]] = [
        (m.start(), m.group(1).strip())
        for m in _FILE_HEADER.finditer(text)
    ]

    if not file_header_positions:
        # No file headers — all blocks are bare
        for m in _BLOCK_RE.finditer(text):
            results.append((m.group(1), m.group(2), None))
        return results

    # Process text in sections delimited by FILE headers.
    # Section boundaries: [0, pos0), [pos0, pos1), ..., [posN, end)
    section_bounds: list[tuple[int, int, str | None]] = []

    # Leading section before first FILE header (no filename)
    first_file_pos = file_header_positions[0][0]
    if first_file_pos > 0:
        section_bounds.append((0, first_file_pos, None))

    for idx, (pos, fname) in enumerate(file_header_positions):
        end = file_header_positions[idx + 1][0] if idx + 1 < len(file_header_positions) else len(text)
        section_bounds.append((pos, end, fname))

    for start, end, fname in section_bounds:
        section = text[start:end]
        for m in _BLOCK_RE.finditer(section):
            results.append((m.group(1), m.group(2), fname))

    return results
