"""Streaming parser for Cardsmith field emissions.

The Cardsmith model emits one of two block formats per reply:

1. ``<commit>{json}</commit>`` — preferred. End-of-turn structured block
   with all fields established this turn. JSON keys map to ``FieldEmission``
   paths. Reliable because models are heavily trained on JSON output.
   Array paths use ``"name[]": [...]`` and each item appends.

2. ``<set path="x">value</set>`` — legacy inline form. Each tag emits its
   own field as soon as the closer arrives. Useful for mid-stream commits
   but harder for models to emit consistently across long replies.

Both formats are supported simultaneously — fields from either path merge
into the session. The parser hides both block types from the user-visible
stream so chat reads clean.

Also watches for the literal sentinel ``[CARDSMITH_DONE]`` which the model
places on its own line at the end of its final reply once the user has
confirmed the card. The sentinel is hidden and sets a ``done`` flag.

Edge cases handled:
- Block opener / value / closer split across chunks
- Sentinel split across chunks
- Multiple blocks in one chunk
- Unterminated blocks at end of stream (silently dropped)
- Mixed protocols within one reply
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

DONE_SENTINEL = "[CARDSMITH_DONE]"

# Accept single or double quotes and optional whitespace around `=`. Models
# vary, and we'd rather forgive minor formatting drift than silently lose a field.
_SET_OPEN_RE = re.compile(r'''<set\s+path\s*=\s*["']([^"']+)["']\s*>''', re.IGNORECASE)
_SET_CLOSE = "</set>"
_COMMIT_OPEN_RE = re.compile(r'<commit\s*>', re.IGNORECASE)
_COMMIT_CLOSE = "</commit>"


@dataclass
class FieldEmission:
    """A single field commit (path, value)."""

    path: str
    value: str


@dataclass
class ParseStep:
    visible: str
    emissions: list[FieldEmission]
    done: bool


class StreamingFieldParser:
    """Stateful parser. Feed chunks; receive visible text + emissions.

    Internal state machine has three modes:
      - NORMAL: scanning for any opener (<set or <commit) or sentinel
      - INSIDE_SET: collecting <set> value until </set>
      - INSIDE_COMMIT: collecting <commit> body until </commit>, then JSON-parse
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "normal"  # "normal" | "set" | "commit"
        self._set_path = ""
        self._set_content = ""
        self._commit_content = ""
        self._done = False

    def feed(self, chunk: str) -> ParseStep:
        if self._done:
            return ParseStep(visible="", emissions=[], done=True)

        self._buffer += chunk
        visible_parts: list[str] = []
        emissions: list[FieldEmission] = []

        while True:
            if self._mode == "set":
                close_idx = self._find_close(self._buffer, _SET_CLOSE)
                if close_idx >= 0:
                    self._set_content += self._buffer[:close_idx]
                    emissions.append(
                        FieldEmission(
                            path=self._set_path,
                            value=self._set_content.strip(),
                        )
                    )
                    self._buffer = self._buffer[close_idx + len(_SET_CLOSE):]
                    self._mode = "normal"
                    self._set_path = ""
                    self._set_content = ""
                    continue
                hold = len(_SET_CLOSE) - 1
                if len(self._buffer) > hold:
                    self._set_content += self._buffer[:-hold]
                    self._buffer = self._buffer[-hold:]
                break

            if self._mode == "commit":
                close_idx = self._find_close(self._buffer, _COMMIT_CLOSE)
                if close_idx >= 0:
                    self._commit_content += self._buffer[:close_idx]
                    emissions.extend(_parse_commit_block(self._commit_content))
                    self._buffer = self._buffer[close_idx + len(_COMMIT_CLOSE):]
                    self._mode = "normal"
                    self._commit_content = ""
                    continue
                hold = len(_COMMIT_CLOSE) - 1
                if len(self._buffer) > hold:
                    self._commit_content += self._buffer[:-hold]
                    self._buffer = self._buffer[-hold:]
                break

            # NORMAL mode: scan for any of (<set, <commit, [CARDSMITH_DONE])
            set_m = _SET_OPEN_RE.search(self._buffer)
            commit_m = _COMMIT_OPEN_RE.search(self._buffer)
            sentinel_idx = self._buffer.find(DONE_SENTINEL)

            candidates = []
            if set_m:
                candidates.append((set_m.start(), "set", set_m))
            if commit_m:
                candidates.append((commit_m.start(), "commit", commit_m))
            if sentinel_idx >= 0:
                candidates.append((sentinel_idx, "done", None))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                idx, kind, m = candidates[0]
                visible_parts.append(self._buffer[:idx])
                if kind == "set":
                    self._mode = "set"
                    self._set_path = m.group(1)
                    self._set_content = ""
                    self._buffer = self._buffer[m.end():]
                    continue
                if kind == "commit":
                    self._mode = "commit"
                    self._commit_content = ""
                    self._buffer = self._buffer[m.end():]
                    continue
                # done sentinel
                self._buffer = self._buffer[idx + len(DONE_SENTINEL):]
                self._done = True
                break

            # No complete opener / sentinel — find the earliest position
            # whose tail could still grow into one and hold back from there.
            # Anything before that position is safe to flush. The earlier
            # fixed-length holdback was wrong: a `<set path="long_name`
            # opener split mid-attribute exceeds the cap and would leak
            # protocol bytes (and the field) to the user-visible stream.
            hold_at = self._find_partial_marker_start(self._buffer)
            if hold_at < 0:
                visible_parts.append(self._buffer)
                self._buffer = ""
            elif hold_at > 0:
                visible_parts.append(self._buffer[:hold_at])
                self._buffer = self._buffer[hold_at:]
            break

        return ParseStep(
            visible="".join(visible_parts),
            emissions=emissions,
            done=self._done,
        )

    def flush(self) -> ParseStep:
        """Emit any held-back text at end-of-stream."""
        if self._mode == "set" or self._mode == "commit":
            # Unterminated block — drop silently rather than leak partial markup.
            self._mode = "normal"
            self._set_content = ""
            self._set_path = ""
            self._commit_content = ""
            self._buffer = ""
            return ParseStep(visible="", emissions=[], done=self._done)
        out = self._buffer
        self._buffer = ""
        if DONE_SENTINEL in out:
            idx = out.find(DONE_SENTINEL)
            return ParseStep(visible=out[:idx], emissions=[], done=True)
        # Strip any trailing partial opener/sentinel — the model never
        # finished it, and showing protocol bytes to the user reads as garbage.
        partial_at = self._find_partial_marker_start(out)
        if partial_at >= 0:
            out = out[:partial_at]
        return ParseStep(visible=out, emissions=[], done=self._done)

    @staticmethod
    def _find_close(text: str, marker: str) -> int:
        """Case-insensitive find for closing tags."""
        return text.lower().find(marker.lower())

    @staticmethod
    def _find_partial_marker_start(buffer: str) -> int:
        """Return the index of the earliest in-progress marker prefix, or -1.

        Position ``i`` is a partial-marker start when ``buffer[i:]`` could
        still grow into a complete ``<set ...>``, ``<commit>``, or
        ``[CARDSMITH_DONE]`` marker. Caller must already have verified that
        no complete marker exists in the buffer.
        """
        earliest = -1
        for ch in ('<', '['):
            start = 0
            while True:
                i = buffer.find(ch, start)
                if i < 0:
                    break
                if StreamingFieldParser._is_partial_marker(buffer[i:]):
                    if earliest == -1 or i < earliest:
                        earliest = i
                    break
                start = i + 1
        return earliest

    @staticmethod
    def _is_partial_marker(tail: str) -> bool:
        """Could ``tail`` still grow into a complete opener or sentinel?"""
        if not tail:
            return False
        if tail[0] == '<':
            tlo = tail.lower()
            # `<commit>` opener (or any prefix of it).
            if '<commit>'.startswith(tlo):
                return True
            # Prefix of literal `<set` (covers `<`, `<s`, `<se`, `<set`).
            if '<set'.startswith(tlo):
                return True
            # `<set` followed by content — only viable if char 4 is whitespace
            # (the regex requires `\s+` after `<set`); `<setiquette` diverges.
            if tlo.startswith('<set'):
                return len(tlo) == 4 or tlo[4].isspace()
            return False
        if tail[0] == '[':
            # Sentinel is case-sensitive (matches `_buffer.find(DONE_SENTINEL)`).
            return DONE_SENTINEL.startswith(tail)
        return False


def _parse_commit_block(body: str) -> list[FieldEmission]:
    """Turn a JSON commit block body into FieldEmission objects.

    Accepts:
      - Bare JSON object: ``{"name": "Lyra", ...}``
      - JSON wrapped in code fences: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\```
      - Array values for ``name[]`` keys: each item becomes a separate
        emission so the state layer can append.
    """
    s = body.strip()
    if not s:
        return []
    # Strip code fences if present
    if s.startswith("```"):
        # Drop first line and trailing fence
        first_nl = s.find("\n")
        last_fence = s.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            s = s[first_nl + 1:last_fence].strip()

    try:
        data = json.loads(s)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []

    out: list[FieldEmission] = []
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(k, str) and k.endswith("[]") and isinstance(v, list):
            # Each item in the array becomes a separate append-emission.
            for item in v:
                if isinstance(item, dict | list):
                    out.append(FieldEmission(path=k, value=json.dumps(item)))
                elif item is not None:
                    out.append(FieldEmission(path=k, value=str(item)))
        elif isinstance(v, dict | list):
            out.append(FieldEmission(path=k, value=json.dumps(v)))
        else:
            out.append(FieldEmission(path=k, value=str(v)))
    return out


def parse_field_emissions(text: str) -> tuple[str, list[FieldEmission], bool]:
    """Convenience one-shot parser for non-streaming text.

    Returns ``(visible_text, emissions, done)``.
    """
    parser = StreamingFieldParser()
    step1 = parser.feed(text)
    step2 = parser.flush()
    return (
        step1.visible + step2.visible,
        step1.emissions + step2.emissions,
        step1.done or step2.done,
    )
