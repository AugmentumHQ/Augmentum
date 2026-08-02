"""Robust JSON recovery for subagent output.

Every bug_finder stage (comprehender, investigator, lead, detector,
orchestrator) ends its turn by emitting a fenced JSON block that the
pipeline parses. The naive parser — "last ```json fence, or output
starting with '{'" — loses the whole result in the failure mode that
actually dominates on large codebases: the model hits its token budget
and gets cut off **mid-JSON**, so there's no closing fence and the
output starts with prose. A lost map/finding-set there is the
"comprehender couldn't produce parseable JSON → zero findings" class
(audit 2026-06-17).

This module recovers JSON from real model output:
  * fenced blocks (the happy path);
  * an opening fence with NO closing fence (truncated final block);
  * JSON embedded in prose without a fence;
  * truncated objects/arrays — closed + repaired down to whatever
    complete elements landed before the cut.

It deliberately does NOT strip ``//`` comments — finding text and the
comprehender ``brief`` routinely contain ``http://`` URLs. A repaired
candidate is always re-validated by ``json.loads``, so a bad repair can
only fail closed, never inject wrong data.
"""
from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _scan_object(text: str, start: int) -> tuple[str, int, bool]:
    """Scan a ``{...}`` (or ``[...]``) starting at ``text[start]``,
    string/escape aware. Returns ``(substring, end_index, complete)``."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : j + 1], j + 1, True
    return text[start:], len(text), False


def _json_candidates(text: str, *, opener: str = "{") -> list[str]:
    """All plausible JSON candidates (objects when opener='{', arrays when
    '['), in document order — fenced, unfenced, and truncated."""
    cands: list[str] = []
    for m in _JSON_BLOCK_RE.finditer(text):
        cands.append(m.group(1).strip())
    i = 0
    n = len(text)
    while i < n:
        if text[i] == opener:
            obj, end, complete = _scan_object(text, i)
            cands.append(obj.strip())
            i = end if complete else n
        else:
            i += 1
    return cands


def _open_structure(s: str) -> bool:
    """True when ``s`` ends with an unclosed object/array/string."""
    stack = 0
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack += 1
        elif ch in "}]":
            stack = max(0, stack - 1)
    return stack > 0 or in_str


def _close_stack(s: str) -> str:
    """Close any open string + brackets at the end of ``s`` (string-aware),
    trimming a dangling trailing comma/colon first."""
    closers: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            closers.append("}")
        elif ch == "[":
            closers.append("]")
        elif ch in "}]" and closers:
            closers.pop()
    out = s
    if in_str:
        out += '"'
    out = out.rstrip()
    while out and out[-1] in ",:":
        out = out[:-1].rstrip()
    return out + "".join(reversed(closers))


def _last_struct_delim(s: str) -> tuple[int, str] | None:
    """Index + char of the last ``,`` / ``{`` / ``[`` not inside a string."""
    in_str = False
    esc = False
    last: tuple[int, str] | None = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in ",{[":
            last = (i, ch)
    return last


def _repair_truncated(s: str) -> str | None:
    """Iteratively close + parse a truncated JSON value; on failure chop
    the last incomplete element and retry. Returns ``None`` when the input
    wasn't truncated (don't mask a real error) or nothing salvageable
    remains."""
    if not _open_structure(s):
        return None
    candidate = s
    for _ in range(64):
        closed = _close_stack(candidate)
        try:
            json.loads(closed)
            return closed
        except json.JSONDecodeError:
            pass
        delim = _last_struct_delim(candidate)
        if delim is None:
            return None
        idx, ch = delim
        candidate = candidate[:idx] if ch == "," else candidate[: idx + 1]
        if not candidate.strip():
            return None
    return None


def _loads_lenient(blk: str) -> Any | None:
    """``json.loads`` with a single truncation-repair fallback."""
    blk = (blk or "").strip()
    if not blk:
        return None
    try:
        return json.loads(blk)
    except json.JSONDecodeError:
        pass
    repaired = _repair_truncated(blk)
    if repaired is None:
        return None
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def salvage_json_object(output: str) -> dict | None:
    """Return the last usable JSON *object* from model ``output``.

    Tries every candidate in reverse document order (so the final, most
    refined emit wins), with truncation repair recovering a budget-cut
    final block instead of losing it.
    """
    if not output:
        return None
    for blk in reversed(_json_candidates(output, opener="{")):
        parsed = _loads_lenient(blk)
        if isinstance(parsed, dict):
            return parsed
    return None


def salvage_json_array(output: str) -> list | None:
    """Return the last usable JSON *array* from model ``output`` (for
    stages that emit a bare list of findings)."""
    if not output:
        return None
    for blk in reversed(_json_candidates(output, opener="[")):
        parsed = _loads_lenient(blk)
        if isinstance(parsed, list):
            return parsed
    return None


__all__ = ["salvage_json_array", "salvage_json_object"]
