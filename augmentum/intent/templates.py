"""Hassil-style template matcher — Tier 1.5 between strict regex and LLM.

The strict regex patterns in `Action.patterns` are exact matches —
fast and deterministic but brittle to natural phrasing variation.
The Tier 3 LLM classifier handles the long tail but costs 1-3s per
call. The middle is a template syntax that compiles to regex at
registration time but is much friendlier to author by hand.

Inspired by Home Assistant's Hassil
(``github.com/home-assistant/hassil``). We implement the subset of
the syntax that matters for our primitives:

  - ``[optional]``    — phrase that may or may not be present
  - ``(a|b|c)``       — alternatives within a phrase
  - ``{slot_name}``   — named slot capture (non-greedy, non-empty)
  - whitespace tolerant — collapse runs of spaces; optional commas

Example: ``[hey] [please] (play|put on|throw on) [some] {query}``
matches:
  - "play jazz"
  - "hey, put on some lofi"
  - "please throw on some rock"
  - "throw on miles davis"

Compilation flattens the template to a token list, then emits one
regex with word-boundary anchors on literal tokens and optional
whitespace separators. Inner content (within ``[...]`` and
``(...)``) is recursively compiled to a non-anchored *inner*
fragment so we don't pile up redundant ``\b`` / terminator
wrappers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledTemplate:
    """A pre-compiled template — regex + the original source string."""

    source: str
    pattern: re.Pattern[str]
    slots: tuple[str, ...]


# ── Tokenizer ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Token:
    """One template token. ``kind`` is 'word' | 'optional' | 'group'
    | 'slot'. ``text`` carries the literal (word) or raw inner
    content (optional/group/slot). ``branches`` is only set for
    group tokens and holds the pre-split alternatives.
    """

    kind: str
    text: str = ""
    branches: tuple[str, ...] = ()


def _find_matching(text: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    """Return the index of the matching close bracket, respecting
    nested groups. -1 if unmatched."""
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j
    return -1


def _split_alternatives(text: str) -> list[str]:
    """Split ``a|b|(c|d)|e`` on top-level pipes only."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch in ("(", "["):
            depth += 1
            buf.append(ch)
        elif ch in (")", "]"):
            depth -= 1
            buf.append(ch)
        elif ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _tokenize(template: str) -> list[_Token]:
    """Walk the template, emit one token per literal word / optional
    group / alternative group / slot. Whitespace is the separator —
    it's not represented in the token list; the assembler inserts
    ``\\s*`` between adjacent tokens.
    """
    tokens: list[_Token] = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch.isspace() or ch == ",":
            i += 1
            continue
        if ch == "[":
            close = _find_matching(template, i, "[", "]")
            if close < 0:
                raise ValueError(f"unmatched '[' at {i} in {template!r}")
            tokens.append(_Token("optional", template[i + 1 : close]))
            i = close + 1
            continue
        if ch == "(":
            close = _find_matching(template, i, "(", ")")
            if close < 0:
                raise ValueError(f"unmatched '(' at {i} in {template!r}")
            inner = template[i + 1 : close]
            branches = tuple(b.strip() for b in _split_alternatives(inner))
            if len(branches) < 2 or any(not b for b in branches):
                raise ValueError(
                    f"group with bad branches at {i} in {template!r}: {branches}"
                )
            tokens.append(_Token("group", inner, branches))
            i = close + 1
            continue
        if ch == "{":
            close = template.find("}", i)
            if close < 0:
                raise ValueError(f"unmatched '{{' at {i} in {template!r}")
            name = template[i + 1 : close].strip()
            if not name.replace("_", "").isalnum():
                raise ValueError(
                    f"slot name {name!r} must be alphanumeric/underscore"
                )
            tokens.append(_Token("slot", name))
            i = close + 1
            continue
        # Literal word — accumulate until whitespace or special char.
        # A stray pipe at top level (``[by|in]`` author mistake —
        # `|` belongs inside `(...)`) would otherwise infinite-loop
        # because the special-char list rejects it but no case
        # handler consumes it. We treat it as a hard error so the
        # author sees the issue at registration time, not at runtime.
        j = i
        while j < n and not template[j].isspace() and template[j] not in "[](){}|,":
            j += 1
        if j == i:
            # No progress — must be a stray special char. ``|`` outside
            # a group is the most common author error; report it
            # explicitly so register_action raises clearly.
            raise ValueError(
                f"unexpected {template[i]!r} at position {i} in "
                f"{template!r}: pipes belong inside ``(...)`` groups, "
                f"not ``[...]`` optionals; use ``(by|in)`` or "
                f"``[by] [in]`` instead"
            )
        word = template[i:j]
        if word:
            tokens.append(_Token("word", word))
        i = j

    return tokens


# ── Compiler ───────────────────────────────────────────────────────


def _compile_inner(template: str, slot_names: list[str]) -> str:
    """Compile an inner template (within ``[...]`` or ``(...)``) to
    a regex *fragment* — no leading ``\\b`` or trailing terminator.
    Inner whitespace is allowed but not required, mirroring the
    top-level behavior.
    """
    tokens = _tokenize(template)
    return _assemble(tokens, slot_names)


def _assemble(tokens: list[_Token], slot_names: list[str]) -> str:
    parts: list[str] = []
    for tok in tokens:
        if tok.kind == "word":
            # Word-boundary anchors so "play" doesn't match "splayed"
            parts.append(rf"\b{re.escape(tok.text)}\b")
        elif tok.kind == "optional":
            inner = _compile_inner(tok.text, slot_names)
            # Possessive `?+` (Python 3.11+) — match the optional
            # when present and DON'T backtrack to give up the match.
            # Plain `?` lets the regex engine skip the optional in
            # favor of a greedy slot eating its content, which is
            # the wrong tradeoff: an optional that's literally
            # present should be consumed by the optional, not the
            # slot. Without possessive, "make me a picture of cats"
            # captures slot="a picture of cats" instead of "cats".
            parts.append(f"(?:{inner})?+")
        elif tok.kind == "group":
            branches = [_compile_inner(b, slot_names) for b in tok.branches]
            parts.append("(?:" + "|".join(branches) + ")")
        elif tok.kind == "slot":
            if tok.text in slot_names:
                raise ValueError(f"slot {tok.text!r} used twice")
            slot_names.append(tok.text)
            # Greedy, non-empty, no sentence terminators. Combined
            # with the final ``\s*[?.!,]*\s*$`` anchor in
            # ``compile_template``, this captures everything from
            # the slot's start to the end of the utterance (or
            # the next sentence terminator).
            parts.append(rf"(?P<{tok.text}>[^.!?,]+)")
    # Optional whitespace / comma between adjacent tokens. Optional
    # rather than required so "playjazz" still matches "play jazz"
    # template (sloppy STT joins words).
    return r"[\s,]*".join(parts)


def compile_template(template: str) -> CompiledTemplate:
    """Compile a hassil-subset template into a regex pattern."""
    if not template or not template.strip():
        raise ValueError("template is empty")

    slot_names: list[str] = []
    body = _compile_inner(template.strip(), slot_names)

    # Final pattern: word-boundary lead-in, body, tolerant
    # terminator + end-of-string anchor. The end anchor forces the
    # (greedy) slot to consume the rest of the utterance after the
    # last required token — without it, ``re.search`` would settle
    # for a 1-char match and drop the actual content.
    #
    # ``re.search`` against this finds matches anywhere in the
    # utterance (STT often joins multiple clauses; we want the
    # question-half to match even when preceded by a greeting).
    final = rf"\b{body}\s*[?.!,]*\s*$"
    try:
        compiled = re.compile(final, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(
            f"template {template!r} compiled to invalid regex "
            f"{final!r}: {exc}"
        ) from exc

    return CompiledTemplate(
        source=template,
        pattern=compiled,
        slots=tuple(slot_names),
    )
