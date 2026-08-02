"""Tool-call tag protocol for Becca's mid-stream tool invocation (Lane 1 §3).

Becca emits one tag per turn that the sieve catches mid-stream:

    <tool:NAME k1="v1" k2="v2" />
    <handoff:CHANNEL reason="..." brief="..." />

Self-closing only. One per emission. Values are single- or double-quoted.
The sieve buffers a tail window so partial tags don't leak to the user.

Lane 3 owns the ``ToolResult`` envelope contract; this module owns the
syntactic protocol + the streaming sieve that detects complete tags
and routes around them.
"""
from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Tag grammar ──────────────────────────────────────────────────────

# Match: <tool:name args /> or <handoff:name args />
# args is a series of k="v" or k='v' pairs, whitespace-separated.
#
# Name charset MUST include dots: intent-registry verbs are dotted
# (grove.play_matching, media.play, navigate.open_surface). The
# original ``[a-z_]+`` silently rejected every dotted tag — the model
# emitted exactly what the roster taught it and the sieve dropped the
# tag as plain text (then the TTS scrubber ate the debris, so nothing
# was visible anywhere). Found 2026-06-10 via becca_act_gap telemetry:
# a 12-token response to "put on some yuzu" that produced no dispatch.
TAG_RE = re.compile(
    r"<(?P<kind>tool|handoff):(?P<name>[a-z][a-z0-9_.]*)"
    r"(?P<args>(?:\s+[a-z_][a-z0-9_]*=(?:\"[^\"]*\"|'[^']*'))*)\s*/>",
    re.IGNORECASE,
)

ARG_RE = re.compile(r'([a-z_][a-z0-9_]*)=(?:"([^"]*)"|\'([^\']*)\')', re.IGNORECASE)

# Salvage grammar — registry-validated error correction for mangled tag
# prefixes. Observed live 2026-06-10: Qwen 3.6 emitted
# ``<j:play_matching query='smooth jazz' />`` and
# ``<j:web.search query='...' />`` — right verb, clean args, garbled
# prefix (likely ``<tool`` colliding with the model's native
# <tool_call> special-token vocabulary). The salvage pass accepts ANY
# tag-shaped self-closing emission, then requires the name to resolve
# against the KNOWN TOOL SET (exact id, or unique dotted-suffix match:
# ``play_matching`` → ``grove.play_matching``). Registry validation is
# the safety: ``<br />`` or prose-shaped angle brackets never resolve,
# so they pass through as text. This is wire-format tolerance on a
# structured channel — not transcript pattern-matching.
SALVAGE_RE = re.compile(
    r"<(?:[a-z][a-z0-9_]*:)?(?P<name>[a-z][a-z0-9_.]*)"
    r"(?P<args>(?:\s+[a-z_][a-z0-9_]*=(?:\"[^\"]*\"|'[^']*'))+)\s*/>",
    re.IGNORECASE,
)


def _resolve_salvage_name(name: str, known: set[str]) -> str | None:
    """Resolve a salvaged tag name against the known tool set.

    Exact match wins; otherwise a UNIQUE dotted-suffix match (the model
    dropped the namespace). Ambiguous or unknown → None (stay text).
    """
    if name in known:
        return name
    suffix_hits = [k for k in known if k.endswith("." + name)]
    if len(suffix_hits) == 1:
        return suffix_hits[0]
    return None


# ── Tier 3: fuzzy call recovery ──────────────────────────────────────
#
# The loosest layer of the three (strict grammar → tag-shaped salvage →
# this). Used by act-classified turns that produced ZERO parseable
# tags: the address classifier already established the user asked for
# an action, so a response containing a known verb name plus
# harvestable args IS that action — even with the format mostly gone
# ("85% correct still continues and matches", Matt 2026-06-10).
#
# Two gates keep prose safe:
#   1. The name must resolve against the registry: exact, unique
#      dotted-suffix, or ≥ FUZZY_NAME_THRESHOLD similarity (catches
#      typos like "play_maching"). Single common words ("recall",
#      "browse") resolve only exactly — never fuzzily.
#   2. At least one k=v arg must follow within the harvest window —
#      call-shape evidence. "let me web.search it" stays prose;
#      "web.search query=jazz" dispatches.

FUZZY_NAME_THRESHOLD = 0.85
# Window was 240 — a note.append body is a paragraph, and anything past
# the window silently vanished. Matches the native hold cap.
_ARG_HARVEST_WINDOW = 4096

# A key at the current parse position: optional separators (whitespace,
# an opening paren for function-call-shaped emissions, a comma between
# pairs) then ``name=``.
_LOOSE_KEY_RE = re.compile(r"[\s(,]*([a-z_][a-z0-9_]*)=", re.IGNORECASE)
# Where an UNQUOTED value ends: the next key boundary.
_UNQUOTED_BOUNDARY_RE = re.compile(r"[\s,]+[a-z_][a-z0-9_]*=", re.IGNORECASE)


def _harvest_loose_args(window: str) -> tuple[dict[str, str], int]:
    """Sequentially parse ``k=v`` pairs from the START of ``window``.

    Replaces the old anywhere-in-window regex, which had two live
    failure modes (observed 2026-06-11, "append wrote one word"):

      * a single-quoted value containing an apostrophe
        (``content='Becca's capabilities…'``) closed one word in under
        the ``'[^']*'`` arm — quoted values now scan forward to the
        first closing quote whose TAIL is parseable (another ``k=``
        pair, tag debris, or end-of-text), so internal apostrophes stay
        inside the value;
      * an unquoted multi-word value (``content=grocery list``) stopped
        at the first space — unquoted values now run to the next key
        boundary or end of window.

    Returns ``(args, end_offset)``: end_offset is how much of the
    window the call consumed, so callers can strip exactly the call
    from speakable text.
    """
    args: dict[str, str] = {}
    pos = 0
    while True:
        km = _LOOSE_KEY_RE.match(window, pos)
        if km is None:
            break
        key = km.group(1).lower()
        vstart = km.end()
        if vstart < len(window) and window[vstart] in "\"'":
            q = window[vstart]
            vend = -1
            scan = vstart + 1
            while True:
                cand = window.find(q, scan)
                if cand < 0:
                    break
                tail = window[cand + 1:].lstrip()
                if (
                    not tail
                    or tail.startswith(("/>", ">", ")"))
                    or _LOOSE_KEY_RE.match(window, cand + 1) is not None
                ):
                    vend = cand
                    break
                scan = cand + 1
            if vend > vstart:
                args[key] = window[vstart + 1: vend]
                pos = vend + 1
            else:
                # No closing quote with a parseable tail anywhere —
                # treat the quote as unterminated (truncated stream)
                # and take the rest. Losing trailing debris beats
                # losing everything past the first apostrophe.
                value = window[vstart + 1:].rstrip().rstrip("/>)").rstrip()
                if value:
                    args[key] = value
                pos = len(window)
        else:
            bm = _UNQUOTED_BOUNDARY_RE.search(window, vstart)
            vend = bm.start() if bm is not None else len(window)
            value = window[vstart:vend].strip().rstrip("/>)").strip()
            if not value:
                break
            args[key] = value
            pos = vend
    return args, pos

# Candidate verb tokens: dotted ids (grove.play_matching) or bare
# words of 4+ chars (play_matching, search). Shorter words are too
# collision-prone with prose to even consider.
_CANDIDATE_RE = re.compile(
    r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+|[a-z][a-z0-9_]{3,}",
    re.IGNORECASE,
)


def _fuzzy_resolve_name(cand: str, known: set[str]) -> str | None:
    """Resolve with typo tolerance. Dotted/underscored candidates may
    match fuzzily; plain single words must match exactly (they appear
    in normal prose far too often to gamble on similarity)."""
    exact = _resolve_salvage_name(cand, known)
    if exact is not None:
        return exact
    if "." not in cand and "_" not in cand:
        return None
    best: str | None = None
    best_ratio = 0.0
    for k in known:
        ratio = max(
            SequenceMatcher(None, cand, k).ratio(),
            SequenceMatcher(None, cand, k.rsplit(".", 1)[-1]).ratio(),
        )
        if ratio > best_ratio:
            best_ratio = ratio
            best = k
    if best is not None and best_ratio >= FUZZY_NAME_THRESHOLD:
        return best
    return None


def recover_loose_call(text: str, known: set[str]) -> ToolCall | None:
    """Last-resort call recovery over a full response.

    Returns the FIRST known-verb-plus-args shape found, or None.
    Callers gate this on "act-classified turn, zero tags parsed" —
    never run it on conversational turns.
    """
    if not text or not known:
        return None
    for m in _CANDIDATE_RE.finditer(text):
        resolved = _fuzzy_resolve_name(m.group(0).lower(), known)
        if resolved is None:
            continue
        window = text[m.end(): m.end() + _ARG_HARVEST_WINDOW]
        args, consumed = _harvest_loose_args(window)
        if not args:
            continue  # name without args = prose mention, not a call
        end = m.end() + consumed
        return ToolCall(
            kind="tool",
            name=resolved,
            args=args,
            raw=text[m.start(): end].strip(),
            span=(m.start(), end),
        )
    return None

# ── Native <tool_call> JSON blocks ───────────────────────────────────
#
# Reasoning-trained families (Qwen 3.x foremost) carry a strongly
# trained native tool-call vocabulary: ``<tool_call>{"name": ...,
# "arguments": {...}}</tool_call>``. When tools arrive via prompt text
# (the voice path) instead of the API tools param, the model falls
# back to this habit under pressure. Observed live 2026-06-11: a
# "look up inflation numbers and add to the note" turn produced ~19s
# of TTS that SPOKE the JSON block — the strict grammar and the
# attr-pair salvage both miss JSON bodies, so it sailed through as
# prose and nothing executed. Same registry-validation safety as
# salvage: an unknown name stays text... except an unparseable block
# is STRIPPED, never spoken — JSON read aloud is worse than silence.
_NATIVE_BLOCK_RE = re.compile(
    r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_NATIVE_OPEN_RE = re.compile(r"<tool_call>", re.IGNORECASE)
# Start of a BARE loose call forming mid-stream: a dotted verb id
# followed by a first ``k=`` pair ("note.append content='…"). Observed
# live 2026-06-11: the model skipped all three trained formats and the
# whole call went to TTS while post-stream recovery harvested one word.
# Registry validation happens at the hold site — this regex alone never
# triggers anything.
_LOOSE_CALL_START_RE = re.compile(
    r"\b(?P<name>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)[\s(]+[a-z_][a-z0-9_]*=",
    re.IGNORECASE,
)
# Hold-back cap while waiting for </tool_call>: past this we assume
# the closer is never coming and release the buffer (debris-stripped
# at drain) rather than stalling TTS for the rest of the stream.
_NATIVE_HOLD_CAP = 4096


def _parse_native_block(json_text: str, known: set[str]) -> ToolCall | None:
    """Decode a native JSON tool call, registry-validated."""
    if not known:
        return None
    try:
        data = _json.loads(json_text)
    except Exception:  # noqa: BLE001 — malformed JSON stays debris
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name") or "").strip().lower()
    resolved = _resolve_salvage_name(name, known)
    if resolved is None:
        return None
    raw_args = data.get("arguments") or data.get("parameters") or {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    args: dict[str, str] = {}
    for k, v in raw_args.items():
        if isinstance(v, dict | list):
            args[str(k).lower()] = _json.dumps(v)
        else:
            args[str(k).lower()] = "" if v is None else str(v)
    return ToolCall(kind="tool", name=resolved, args=args, raw="", span=(0, 0))


# The longest plausible tag prefix we need to hold in the tail buffer
# while waiting to disambiguate. A tag emission is at most ~256 chars
# in practice; we use 256 as the buffer size for safety.
_TAIL_BUFFER_SIZE = 256


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A parsed tool/handoff tag.

    ``kind`` is "tool" or "handoff". ``name`` is the canonical id (matches
    SubagentRegistry/PrimitiveRegistry name, or channel name for
    handoffs). ``args`` is the decoded keyword dict. ``raw`` is the
    original tag text for stream-replacement bookkeeping.
    """
    kind: str
    name: str
    args: dict[str, str]
    raw: str
    span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Promise:
    """Captured commitment Becca made before a tool tag.

    ``pre_text`` is everything she streamed before the tag — the
    "promise" the deliver step has to honor. ``tag`` is the verb she
    chose. ``started_at`` is monotonic seconds when the tag was emitted.

    Used by the primary-tier deliver step to produce a confirmation that
    closes the loop opened by ``pre_text``: "let me put that on" →
    "alright, Dune's playing in the living room".
    """
    pre_text: str
    tag: "ToolCall"
    started_at: float


@dataclass(frozen=True, slots=True)
class ToolError:
    """Categorical error returned by a tool invocation (Lane 3 §2.3)."""
    category: str         # 'timeout' | 'unauthorized' | 'content_policy' | 'model_unavailable' | 'invalid_args' | 'upstream_error' | 'cancelled' | 'tool_self_error'
    message: str          # human-readable, NOT shown to user verbatim
    retryable: bool = False
    fallback_hint: str = ""


@dataclass(frozen=True, slots=True)
class UIEffect:
    """A side effect declared by a tool — fanned out by the runtime to
    the bus so existing UI surfaces mount (image viewer, workspace, etc.)
    without modification. Lane 3 §7."""
    kind: str             # bus topic to emit (e.g., "image.generated")
    target: str           # surface id or "_inline"
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Envelope every tool returns. Lane 3 §2.1."""
    ok: bool
    tool: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    ui_effects: tuple[UIEffect, ...] = ()
    duration_ms: int = 0
    cancelled: bool = False
    error: ToolError | None = None


# ── Tag parser ───────────────────────────────────────────────────────

def scan(text: str) -> list[ToolCall]:
    """Return all complete tags in ``text``, in order.

    Used in tests and offline analysis; the streaming sieve below uses
    its own incremental logic to avoid re-scanning the entire buffer
    every token.
    """
    out: list[ToolCall] = []
    for m in TAG_RE.finditer(text):
        args: dict[str, str] = {}
        for a in ARG_RE.finditer(m.group("args") or ""):
            k = a.group(1)
            v = a.group(2) if a.group(2) is not None else a.group(3)
            args[k] = v
        out.append(ToolCall(
            kind=m.group("kind").lower(),
            name=m.group("name").lower(),
            args=args,
            raw=m.group(0),
            span=m.span(),
        ))
    return out


# ── Streaming sieve ──────────────────────────────────────────────────

class TagSieve:
    """Incremental scanner over a token stream.

    Usage:

        sieve = TagSieve()
        async for chunk in primary_stream:
            for clean_text, tag in sieve.feed(chunk):
                if clean_text:
                    await out.write(clean_text)
                if tag is not None:
                    # cancel primary, invoke tool, splice result back
                    ...

    The sieve holds the last ``_TAIL_BUFFER_SIZE`` chars un-emitted while
    deciding whether they're part of a tag. When a complete tag is
    detected, it yields (pre_tag_clean, tag); otherwise it yields
    (clean, None) chunks as text accumulates past the tail buffer.
    """

    def __init__(self, known_tools=None, allow_loose=False) -> None:
        """``known_tools`` enables the salvage pass: a set of canonical
        tool ids, or a zero-arg callable returning one (lazy — lets
        callers avoid import cycles). None disables salvage; the strict
        TAG_RE grammar still applies.

        ``allow_loose`` (bool or zero-arg callable, evaluated lazily)
        extends the sieve to BARE call shapes ("note.append content=…"
        with no tag wrapper at all): mid-stream they're held back from
        the emit path so TTS never speaks them, and at drain() they're
        recovered as tags via the registry-validated loose parser.
        Callers gate this on act-classified turns — the same gate the
        post-stream recover_loose_call fallback uses — so prose that
        merely MENTIONS a verb id on a conversational turn is never
        held or dispatched."""
        self._buf: str = ""
        self._known_tools = known_tools
        self._allow_loose = allow_loose

    def _known(self) -> set[str]:
        kt = self._known_tools
        if kt is None:
            return set()
        if callable(kt):
            try:
                return set(kt() or ())
            except Exception:
                return set()
        return set(kt)

    def _loose_enabled(self) -> bool:
        al = self._allow_loose
        if callable(al):
            try:
                return bool(al())
            except Exception:
                return False
        return bool(al)

    def _find_loose_start(self) -> int | None:
        """Position of a registry-validated bare call forming in the
        buffer, or None. Exact/unique-suffix resolution only (no fuzzy
        — too hot for the per-chunk path; a typo'd name falls through
        to the post-stream recovery as before)."""
        known = self._known()
        if not known:
            return None
        for m in _LOOSE_CALL_START_RE.finditer(self._buf):
            if _resolve_salvage_name(m.group("name").lower(), known) is not None:
                return m.start()
        return None

    def _try_salvage(self):
        """Registry-validated salvage on the current buffer. Returns a
        (match, ToolCall) pair or None. Only fires when the mangled
        name resolves against the known tool set."""
        known = self._known()
        if not known:
            return None
        for m in SALVAGE_RE.finditer(self._buf):
            resolved = _resolve_salvage_name(m.group("name").lower(), known)
            if resolved is None:
                continue
            args: dict[str, str] = {}
            for a in ARG_RE.finditer(m.group("args") or ""):
                k = a.group(1)
                v = a.group(2) if a.group(2) is not None else a.group(3)
                args[k] = v
            return m, ToolCall(
                kind="tool",
                name=resolved,
                args=args,
                raw=m.group(0),
                span=m.span(),
            )
        return None

    def _scan_buffer(self):
        """One parse attempt over the buffer, strictest grammar first.

        Returns ``(pre_text, tag, rest)`` on a hit, or None. Unparseable
        native ``<tool_call>`` blocks (bad JSON / unknown name) are CUT
        from the buffer and logged — spoken JSON is the one outcome
        worse than a dropped call.
        """
        m = TAG_RE.search(self._buf)
        if m is not None:
            args: dict[str, str] = {}
            for a in ARG_RE.finditer(m.group("args") or ""):
                k = a.group(1)
                v = a.group(2) if a.group(2) is not None else a.group(3)
                args[k] = v
            tag = ToolCall(
                kind=m.group("kind").lower(),
                name=m.group("name").lower(),
                args=args,
                raw=m.group(0),
                span=m.span(),
            )
            return self._buf[: m.start()], tag, self._buf[m.end():]

        # Strict grammar missed — salvage pass for mangled prefixes
        # (``<j:play_matching …/>``). Registry-validated, so prose and
        # HTML-shaped text can't fire it.
        salvaged = self._try_salvage()
        if salvaged is not None:
            m2, tag = salvaged
            return self._buf[: m2.start()], tag, self._buf[m2.end():]

        # Native <tool_call> JSON blocks (Qwen-family habit). Parse
        # failures don't fall through to prose — strip and log.
        m3 = _NATIVE_BLOCK_RE.search(self._buf)
        while m3 is not None:
            tag = _parse_native_block(m3.group("json"), self._known())
            if tag is not None:
                return self._buf[: m3.start()], tag, self._buf[m3.end():]
            log.warning(
                "becca_native_block_unresolvable",
                preview=m3.group(0)[:160],
            )
            self._buf = self._buf[: m3.start()] + self._buf[m3.end():]
            m3 = _NATIVE_BLOCK_RE.search(self._buf)
        return None

    def feed(self, chunk: str):
        """Yield (clean_text_to_emit, ToolCall_or_None) pairs.

        On a tag detection: yields one ``(pre_text, tag)`` and resets
        the buffer to whatever followed the tag close. The caller is
        expected to cancel the primary stream at that point (we have
        what we wanted; rest of primary output is discarded).
        """
        self._buf += chunk

        hit = self._scan_buffer()
        if hit is not None:
            pre, tag, rest = hit
            self._buf = rest
            yield pre, tag
            return

        # No complete tag — emit anything older than the tail-buffer
        # window so we don't hold the entire stream in memory.
        if len(self._buf) > _TAIL_BUFFER_SIZE:
            # Be careful not to emit chars that could be the start of a
            # tag. The safest rule: only emit up to the last newline OR
            # the last char that cannot start a tag (i.e., not '<').
            cutoff = len(self._buf) - _TAIL_BUFFER_SIZE
            # An OPEN native block can far exceed the tail window (a
            # note.append body is a paragraph of JSON) — hold from its
            # opener so no fragment of it reaches TTS, up to the cap.
            open_m = _NATIVE_OPEN_RE.search(self._buf)
            if (
                open_m is not None
                and open_m.start() < cutoff
                and len(self._buf) - open_m.start() <= _NATIVE_HOLD_CAP
            ):
                cutoff = open_m.start()
            # A bare loose call has NO closing delimiter, so it can only
            # be parsed at drain — but it must be held NOW or the body
            # streams to TTS word by word (the "she reads the tool call"
            # bug). Holding is safe even on a false positive: the text
            # is merely delayed to drain, where a failed parse releases
            # it as normal speakable text.
            if self._loose_enabled():
                loose_pos = self._find_loose_start()
                if (
                    loose_pos is not None
                    and loose_pos < cutoff
                    and len(self._buf) - loose_pos <= _NATIVE_HOLD_CAP
                ):
                    cutoff = loose_pos
            # If the cutoff is mid-potential-tag (a '<' appears in
            # [cutoff, end)), keep that '<' and everything after it.
            lt_pos = self._buf.find("<", cutoff)
            if lt_pos >= 0:
                cutoff = lt_pos
            emit = self._buf[:cutoff]
            self._buf = self._buf[cutoff:]
            if emit:
                yield emit, None

    def drain(self):
        """End-of-stream: yield remaining (clean, tag) pairs, then the
        final text residue. Unclosed native-block debris is stripped
        from the residue (logged) — it is never speakable text. After
        ``drain()`` the sieve is reset.
        """
        while True:
            hit = self._scan_buffer()
            if hit is None:
                break
            pre, tag, rest = hit
            self._buf = rest
            yield pre, tag
        out = self._buf
        self._buf = ""
        open_m = _NATIVE_OPEN_RE.search(out)
        if open_m is not None:
            log.warning(
                "becca_tool_call_debris_stripped",
                preview=out[open_m.start(): open_m.start() + 160],
            )
            out = out[: open_m.start()]
        # Bare loose call held by feed() — recover it as a tag here so
        # it executes through the normal pending/native-loop path with
        # FULL args, instead of being spoken and scraped post-stream.
        if out and self._loose_enabled():
            loose = recover_loose_call(out, self._known())
            if loose is not None:
                log.info(
                    "becca_loose_call_sieved",
                    tool=loose.name,
                    args_keys=sorted(loose.args.keys()),
                )
                pre = out[: loose.span[0]]
                # Strip mangled-wrapper debris the name match excludes
                # ("<tool: " / "<j:" prefixes) so it isn't spoken.
                pre = re.sub(r"<[a-z_:]*\s*$", "", pre, flags=re.IGNORECASE)
                rest = out[loose.span[1]:].lstrip()
                while rest[:2] == "/>" or rest[:1] in (">", ")"):
                    rest = rest[2:].lstrip() if rest[:2] == "/>" else rest[1:].lstrip()
                yield pre, loose
                out = rest
        if out:
            yield out, None

    def flush(self) -> str:
        """Text-only end-of-stream drain, for callers with no post-stream
        execution path (the no-runtime fallback in becca_direct — tools
        can't run there anyway). Parsed-but-undeliverable tags are
        logged, not silently eaten."""
        parts: list[str] = []
        dropped = 0
        for clean, tag in self.drain():
            if clean:
                parts.append(clean)
            if tag is not None:
                dropped += 1
        if dropped:
            log.warning("becca_sieve_flush_dropped_tags", count=dropped)
        return "".join(parts)


__all__ = [
    "TAG_RE",
    "ARG_RE",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "UIEffect",
    "TagSieve",
    "Promise",
    "scan",
]
