"""Cardsmith scratchpad + reference index.

The scratchpad accumulates fetched ``ContentDoc`` instances across the
conversation so the Cardsmith model has persistent memory of every
external source the user has surfaced. Designed to scale to long
iterative refinement (~30 articles) without bloating the system prompt.

# Three zones

  active   — recently fetched, not yet used. Full content visible to model.
  indexed  — older fetches, not consumed. Title + 1-line digest only.
  consumed — already used in a lorebook entry. One-line ref + reverse pointer.

The model sees all three but with progressively compact rendering as
zones get older / more used. The active set is bounded; the rest can grow
within reason.

# Reference index

Each fetched doc is indexed at insertion time on its title, aliases,
section headings, and category names. At each turn, the user's last
message + recent conversation tail is scanned against the index. Hits
surface in a ``<recalled>`` block in the system prompt for that turn —
even if the doc is zoned to indexed/consumed, its full content is
re-injected when something it knows about gets mentioned.

This solves the "30 turns later, user says 'Tessia's grandfather' and
the model has no recall" problem. The index is rebuilt from the
scratchpad on demand (no separate persistence — it's derived state).
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from augmentum.knowledge.content_extractor import ContentDoc


# Zone classifications.
ZONE_ACTIVE = "active"
ZONE_INDEXED = "indexed"
ZONE_CONSUMED = "consumed"

# Cap how many docs stay in the "active" zone (full content rendered).
# Beyond this, oldest active docs get demoted to "indexed".
_ACTIVE_CAP = 10

# Hard cap on total scratchpad size. Beyond this, oldest non-consumed
# entries get LRU-evicted entirely.
_TOTAL_CAP = 30


@dataclass
class ScratchEntry:
    """One fetched document, plus its zone state.

    Stored on ``CardsmithSession.meta["scratchpad"]`` as a list of dicts
    (round-tripped through ``asdict()`` since the session meta is
    JSON-serializable).
    """

    url: str
    path: str  # /wiki/X for MediaWiki, full URL otherwise — used for dedup
    title: str
    summary: str
    sections: dict[str, str] = field(default_factory=dict)
    infobox: dict[str, str] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    # Internal links extracted from this doc — surfaced in the scratchpad
    # render so the model picks REAL article paths via fetch_targets[]
    # instead of hallucinating slugs. Stored as list of {title, path, is_internal}.
    extracted_links: list[dict[str, Any]] = field(default_factory=list)
    source_kind: str = "generic"

    zone: str = ZONE_ACTIVE
    consumed_by: str = ""  # if zone=consumed: the lorebook entry name/ref
    fetched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScratchEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_content_doc(cls, doc: ContentDoc, *, path: str = "") -> ScratchEntry:
        return cls(
            url=doc.url,
            path=path or doc.url,
            title=doc.title,
            summary=doc.summary,
            sections=dict(doc.sections),
            infobox=dict(doc.infobox),
            aliases=list(doc.aliases),
            extracted_links=[link.to_dict() for link in doc.extracted_links],
            source_kind=doc.source_kind,
        )


# ── Scratchpad operations ─────────────────────────────────────────────────

def add_to_scratchpad(
    scratchpad: list[ScratchEntry],
    entry: ScratchEntry,
) -> bool:
    """Append a new entry, dedup by path. Demote stale active → indexed.

    Returns True if added, False if it was a duplicate.
    """
    if any(e.path == entry.path for e in scratchpad):
        return False

    scratchpad.append(entry)

    # Demote oldest active entries past the cap to indexed.
    actives = [e for e in scratchpad if e.zone == ZONE_ACTIVE]
    if len(actives) > _ACTIVE_CAP:
        actives.sort(key=lambda e: e.fetched_at)
        excess = len(actives) - _ACTIVE_CAP
        for old in actives[:excess]:
            old.zone = ZONE_INDEXED

    # Hard-evict oldest non-consumed entries past total cap.
    non_consumed = [e for e in scratchpad if e.zone != ZONE_CONSUMED]
    if len(non_consumed) > _TOTAL_CAP:
        non_consumed.sort(key=lambda e: e.fetched_at)
        excess = len(non_consumed) - _TOTAL_CAP
        to_evict = {id(e) for e in non_consumed[:excess]}
        scratchpad[:] = [e for e in scratchpad if id(e) not in to_evict]

    return True


def mark_consumed(
    scratchpad: list[ScratchEntry],
    paths: list[str],
    *,
    consumer: str,
) -> int:
    """Mark scratchpad entries as consumed by a lorebook entry name.

    Returns the count of entries successfully marked.
    """
    if not paths:
        return 0
    path_set = set(paths)
    n = 0
    for entry in scratchpad:
        if entry.path in path_set and entry.zone != ZONE_CONSUMED:
            entry.zone = ZONE_CONSUMED
            entry.consumed_by = consumer
            n += 1
    return n


def serialize_scratchpad(scratchpad: list[ScratchEntry]) -> list[dict[str, Any]]:
    """Round-trip helper for session.meta JSON storage."""
    return [e.to_dict() for e in scratchpad]


def deserialize_scratchpad(raw: list[dict[str, Any]]) -> list[ScratchEntry]:
    if not isinstance(raw, list):
        return []
    return [ScratchEntry.from_dict(d) for d in raw if isinstance(d, dict)]


# ── Reference index ───────────────────────────────────────────────────────

# Term-weight tiers. When multiple sources of evidence point at the same
# entry, the highest weight wins.
_W_PRIMARY = 1.0  # title
_W_ALIAS = 0.9  # canonical aliases
_W_LINK = 0.5  # entry's outbound links (cross-references)
_W_SECTION = 0.3  # section headings
_W_INFOBOX = 0.4  # infobox key/value tokens


@dataclass(frozen=True)
class IndexEntry:
    path: str
    weight: float


def build_reference_index(
    scratchpad: list[ScratchEntry],
) -> dict[str, list[IndexEntry]]:
    """Build the per-session reference index from current scratchpad state.

    Returns a dict mapping ``lowercase_term → list[IndexEntry]``. Built
    fresh on each turn (cheap — ~30 entries × few terms each).
    """
    index: dict[str, list[IndexEntry]] = defaultdict(list)

    def _add(term: str, path: str, weight: float) -> None:
        if not term or len(term) < 2:
            return
        # Normalize: lowercase, collapse internal whitespace.
        key = " ".join(term.lower().split())
        if not key or len(key) > 80:
            return
        index[key].append(IndexEntry(path=path, weight=weight))

    for entry in scratchpad:
        path = entry.path
        # Title
        _add(entry.title, path, _W_PRIMARY)
        # Title variations: drop parenthetical, drop common articles
        for var in _title_variations(entry.title):
            _add(var, path, _W_ALIAS)
        # Aliases
        for alias in entry.aliases:
            _add(alias, path, _W_ALIAS)
        # Section headings
        for heading in entry.sections:
            _add(heading, path, _W_SECTION)
        # Infobox keys + short values (people/place names tend to appear here)
        for k, v in entry.infobox.items():
            if len(k) <= 30:
                _add(k, path, _W_INFOBOX)
            # Mine multi-token values for capitalized phrases (likely names)
            for token in _proper_noun_tokens(v):
                _add(token, path, _W_INFOBOX)

    return dict(index)


def _title_variations(title: str) -> list[str]:
    """Common variants — strip parenthetical, drop a leading article."""
    out: list[str] = []
    base = title.strip()
    # Drop parenthetical: "Sapin (Kingdom)" → "Sapin"
    if "(" in base:
        no_paren = base.split("(", 1)[0].strip()
        if no_paren and no_paren != base:
            out.append(no_paren)
    # Drop leading article
    for article in ("The ", "A ", "An "):
        if base.startswith(article):
            out.append(base[len(article):])
            break
    return out


# Crude proper-noun token finder: capitalized words >= 3 chars, joined when
# adjacent. Works well enough on English text without NLP dependencies.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")


def _proper_noun_tokens(text: str) -> list[str]:
    if not text or len(text) > 500:
        return []
    return _PROPER_NOUN_RE.findall(text)[:8]


# ── Recall ────────────────────────────────────────────────────────────────

def recall_for_turn(
    user_message: str,
    conversation_tail: str,
    index: dict[str, list[IndexEntry]],
    scratchpad: list[ScratchEntry],
    *,
    max_recalls: int = 3,
) -> list[ScratchEntry]:
    """Find scratchpad entries the user may have referenced in their message.

    Scans both the user's last message AND the recent conversation tail
    (so the model can recall things its earlier replies established but
    that the user echoes). Returns up to ``max_recalls`` highest-scoring
    entries — typically zone=indexed or zone=consumed (active is already
    in the prompt by default).
    """
    if not index or not scratchpad:
        return []

    text = (user_message + " " + conversation_tail).lower()
    if not text.strip():
        return []

    # Score: max-weight per matched path (so a primary hit beats two section hits).
    scores: dict[str, float] = {}
    for term, entries in index.items():
        if term in text:
            for e in entries:
                scores[e.path] = max(scores.get(e.path, 0.0), e.weight)

    if not scores:
        return []

    # Resolve paths back to scratchpad entries — but only entries whose
    # zone is NOT active (those are already in the prompt baseline).
    by_path = {e.path: e for e in scratchpad}
    candidates: list[tuple[float, ScratchEntry]] = []
    for path, score in scores.items():
        entry = by_path.get(path)
        if entry is None:
            continue
        if entry.zone == ZONE_ACTIVE:
            continue
        candidates.append((score, entry))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [e for _, e in candidates[:max_recalls]]


# ── Prompt rendering ──────────────────────────────────────────────────────

# Treat any scratchpad entry fetched within this many seconds as "recent" —
# surfaced in a dedicated <recently_fetched> hint so the model doesn't redundantly
# emit fetch_targets[] for paths it already pulled.
_RECENT_WINDOW_SECONDS = 90.0


def render_scratchpad_block(
    scratchpad: list[ScratchEntry],
    *,
    recalled: list[ScratchEntry] | None = None,
) -> str:
    """Render scratchpad + recall as XML-ish blocks for the system prompt.

    ``recalled`` is the list returned by ``recall_for_turn`` for this turn —
    those entries are surfaced in a ``<recalled>`` block in addition to
    the standard scratchpad block.
    """
    if not scratchpad:
        return ""

    by_zone = {ZONE_ACTIVE: [], ZONE_INDEXED: [], ZONE_CONSUMED: []}
    for e in scratchpad:
        by_zone.setdefault(e.zone, []).append(e)

    lines = [
        f'<scratchpad active="{len(by_zone[ZONE_ACTIVE])}" '
        f'indexed="{len(by_zone[ZONE_INDEXED])}" '
        f'consumed="{len(by_zone[ZONE_CONSUMED])}">',
    ]

    # Active: full content.
    for entry in by_zone[ZONE_ACTIVE]:
        lines.append(f'  <doc path="{_esc(entry.path)}" status="active">')
        lines.append(f"    <title>{_esc(entry.title)}</title>")
        if entry.summary:
            lines.append(f"    <summary>{_esc(_clip(entry.summary, 600))}</summary>")
        if entry.aliases:
            lines.append(
                f"    <aliases>{_esc(', '.join(entry.aliases[:6]))}</aliases>"
            )
        if entry.sections:
            for heading, content in list(entry.sections.items())[:5]:
                lines.append(
                    f'    <section h="{_esc(heading)}">{_esc(_clip(content, 400))}</section>'
                )
        # Internal links — these are the REAL article paths the model can
        # request via fetch_targets[]. Without this surfaced, the model
        # hallucinates slugs (audit caught Luminous/Stockholm/QuSense for
        # an Infinite Dendrogram session — none of which exist on the wiki).
        if entry.extracted_links:
            lines.append('    <links note="real paths — pick from these for fetch_targets[]">')
            for link in entry.extracted_links[:30]:
                title = link.get("title") if isinstance(link, dict) else getattr(link, "title", "")
                path = link.get("path") if isinstance(link, dict) else getattr(link, "path", "")
                if title and path:
                    lines.append(f'      <link path="{_esc(path)}">{_esc(title)}</link>')
            lines.append("    </links>")
        lines.append("  </doc>")

    # Indexed: digest only.
    if by_zone[ZONE_INDEXED]:
        lines.append('  <indexed_digest>')
        for entry in by_zone[ZONE_INDEXED]:
            digest = _clip(entry.summary, 120) if entry.summary else ""
            lines.append(
                f'    <doc path="{_esc(entry.path)}">{_esc(entry.title)}'
                + (f" — {_esc(digest)}" if digest else "")
                + "</doc>"
            )
        lines.append('  </indexed_digest>')

    # Consumed: one-line ref.
    if by_zone[ZONE_CONSUMED]:
        lines.append('  <consumed_refs>')
        for entry in by_zone[ZONE_CONSUMED]:
            label = entry.consumed_by or "lorebook"
            lines.append(
                f'    <doc path="{_esc(entry.path)}">{_esc(entry.title)}'
                f" — used in {_esc(label)}</doc>"
            )
        lines.append('  </consumed_refs>')

    lines.append("</scratchpad>")

    # Recently-fetched hint — paths fetched within the last window. Helps the
    # model avoid re-emitting fetch_targets[] for things it already pulled.
    cutoff = time.time() - _RECENT_WINDOW_SECONDS
    recent = sorted(
        (e for e in scratchpad if e.fetched_at >= cutoff),
        key=lambda e: e.fetched_at,
    )
    if recent:
        lines.append("")
        lines.append("<recently_fetched note=\"already in scratchpad — do not re-fetch\">")
        for entry in recent[-12:]:
            lines.append(
                f'  <doc path="{_esc(entry.path)}">{_esc(entry.title)}</doc>'
            )
        lines.append("</recently_fetched>")

    # Recall block — full content of any non-active entry the user mentioned.
    if recalled:
        lines.append("")
        lines.append("<recalled>")
        for entry in recalled:
            lines.append(
                f'  <doc path="{_esc(entry.path)}" was="{entry.zone}">'
            )
            lines.append(f"    <title>{_esc(entry.title)}</title>")
            if entry.summary:
                lines.append(f"    <summary>{_esc(_clip(entry.summary, 600))}</summary>")
            if entry.sections:
                for heading, content in list(entry.sections.items())[:5]:
                    lines.append(
                        f'    <section h="{_esc(heading)}">{_esc(_clip(content, 400))}</section>'
                    )
            lines.append("  </doc>")
        lines.append("</recalled>")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────

def _clip(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + "…"


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
