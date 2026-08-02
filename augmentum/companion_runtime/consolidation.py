"""Personality consolidator — the slow consolidation synapse.

Synapse Layer §4. Becca's own personality doc names this in §Notes:

    The consolidation pipeline can edit this document. The anti-drift
    detector caps the rate of change at an embedding distance of < 0.15
    between versions, so she can grow but not transform overnight.
    Updates should happen in her voice, not a maintainer's — paragraphs
    rewritten as if she were revising her own self-description, which
    she is.

This module is that pipeline. It runs at most once per
``companion_consolidation_interval_days`` (default 30) and proposes
candidate edits to the *rotating* sections of the personality doc —
§10 (cultural diet) and §11 (open questions) per the doc's own
self-description.

**Sections 1-6 are FROZEN.** The personality doc says they "should
change rarely and only with reason." :func:`propose_candidate`
refuses any section ≤ 6 by raising :class:`FrozenSectionError`. The
schema doesn't enforce this; the application does. This is by design
— a future considered-edit pathway might allow §1-6 changes through
a different gate, and that should be an explicit decision, not a
silent allowance.

**Drift discipline.** Every candidate computes an embedding distance
between the proposed paragraph and the current persona kernel digest.
Candidates over ``companion_consolidation_drift_ceiling`` (default
0.15, matches DRIFT_CEILING for kernel anchoring) are refused — they
never reach the table. This is the structural promise that she
"grows but does not transform overnight."

**The candidate is queued, not applied.** Approval is a separate
explicit user action — see :mod:`augmentum.proxy.consolidation_routes`
for the review API. Approval writes a ``*.candidate.md`` sidecar file
that a human git-commits manually; the live doc only changes when
they take that step. Rejection journals the reason in the companion's
voice so the consolidator doesn't propose the same shape again.

Behind ``companion_consolidation_enabled`` (default False).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.memory.embeddings import EmbeddingService
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# ── Policy ────────────────────────────────────────────────────────────

# Sections the consolidator is allowed to touch. The personality doc's
# own §Notes designates §10 + §11 as "expected to rotate most often."
ROTATING_SECTIONS: frozenset[int] = frozenset({10, 11})

# Sections that are explicitly frozen. The consolidator refuses to
# propose edits to these. §1-6 per the doc's own self-description.
FROZEN_SECTIONS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6})

# Default drift ceiling — matches the kernel drift cap in identity.py.
# Configurable via companion_consolidation_drift_ceiling.
DEFAULT_DRIFT_CEILING: float = 0.15

# Default min evidence count — below this we don't bother proposing.
# A consolidation needs to be grounded in real material; proposing
# from 2 journal entries would be confabulation.
DEFAULT_MIN_EVIDENCE: int = 8


# ── Exceptions ────────────────────────────────────────────────────────

class FrozenSectionError(ValueError):
    """Raised when a caller tries to consolidate a §1-6 section."""


class InsufficientEvidenceError(RuntimeError):
    """Raised when fewer than the min evidence threshold entries
    are available for the proposed section."""


class DriftCeilingExceededError(RuntimeError):
    """The proposed candidate exceeds the drift ceiling against the
    current persona kernel embedding. The candidate is discarded."""


# ── Section parser ────────────────────────────────────────────────────


_SECTION_RE = re.compile(r"^## (\d+)\. (.+?)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class ParsedSection:
    number: int
    title: str
    body: str
    start_offset: int
    end_offset: int


def parse_sections(doc_text: str) -> list[ParsedSection]:
    """Extract all ``## <N>. <title>`` sections from the personality doc.

    Returns sections in document order. Body is the text between the
    heading and the next ``## <N>.`` heading (or the ``---`` Notes
    delimiter, whichever comes first). Trailing whitespace stripped.
    """
    matches = list(_SECTION_RE.finditer(doc_text))
    if not matches:
        return []
    # Find the Notes delimiter — sections end at the first standalone
    # '---' line that follows them. Falls back to end-of-doc.
    notes_match = re.search(r"^---\s*$", doc_text, re.MULTILINE)
    notes_offset = notes_match.start() if notes_match else len(doc_text)

    sections: list[ParsedSection] = []
    for i, m in enumerate(matches):
        number = int(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        # Body ends at next section heading or at Notes delimiter,
        # whichever comes first.
        if i + 1 < len(matches):
            body_end = matches[i + 1].start()
        else:
            body_end = len(doc_text)
        body_end = min(body_end, notes_offset)
        body = doc_text[body_start:body_end].strip()
        sections.append(ParsedSection(
            number=number,
            title=title,
            body=body,
            start_offset=m.start(),
            end_offset=body_end,
        ))
    return sections


def get_section(doc_text: str, section_number: int) -> ParsedSection | None:
    """Return one section by number, or ``None`` if not found."""
    for s in parse_sections(doc_text):
        if s.number == section_number:
            return s
    return None


# ── Evidence selection ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Evidence:
    """A piece of evidence the consolidator considered when proposing.

    Carries enough to: (1) render the review UI's "what she's drawing
    on" panel, (2) pass to the LLM prompt as grounding, (3) journal a
    rejection note that references specific evidence.
    """
    kind: str           # 'journal' | 'dream'
    id: int             # row id in companion_journal or dream_entries
    content: str        # excerpt of the entry
    affect_tag: str     # tag at write time
    created_at: str     # ISO-ish timestamp


async def gather_evidence(
    runtime: CompanionRuntime,
    *,
    days_back: int = 30,
    section_number: int,
) -> list[Evidence]:
    """Pull journal + dream entries from the last N days, weighted
    toward higher-affect entries.

    Section §10 (cultural taste) biases toward journal entries with
    aesthetic / cultural-diet affect markers. Section §11 (curiosity)
    biases toward 'wondering' entry_type. Other rotating sections (if
    we add them later) pull uniformly.
    """
    backend = runtime.backend
    cutoff_clause = f"datetime('now', '-{int(days_back)} days')"

    # Journal: prefer entries with meaningful affect tags + non-stub
    # content. The simplest filter is "affect_tag is present and not
    # 'settled'/'unclear'" plus "content doesn't start with [tick".
    if section_number == 11:
        # Curiosity section — wondering entries dominate
        entry_type_clause = (
            "(entry_type = 'wondering' "
            "OR entry_type = 'conversation_moment' "
            "OR entry_type = 'noticing')"
        )
    else:
        entry_type_clause = "1=1"

    try:
        cur = await backend.conn.execute(
            f"""
            SELECT id, content, affect_tag, created_at
            FROM companion_journal
            WHERE companion_id = ?
              AND created_at >= {cutoff_clause}
              AND {entry_type_clause}
              AND content NOT LIKE '[tick %'
              AND affect_tag IS NOT NULL
              AND affect_tag != ''
              AND affect_tag != 'settled'
              AND suppressed = 0
            ORDER BY created_at DESC
            LIMIT 60
            """,
            (runtime.companion_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("consolidation_journal_fetch_failed", exc_info=True)
        rows = []

    out: list[Evidence] = []
    for r in rows:
        out.append(Evidence(
            kind="journal",
            id=int(r[0]),
            content=(r[1] or "")[:400],
            affect_tag=(r[2] or "")[:32],
            created_at=str(r[3] or ""),
        ))

    # Dreams: anything recent. Per dream lifecycle, dream_entries rows
    # are already curated outputs of the reflection step, so we don't
    # need additional filtering here.
    try:
        cur = await backend.conn.execute(
            f"""
            SELECT id, content, created_at
            FROM dream_entries
            WHERE (companion_id = ? OR persona_id = ?)
              AND created_at >= {cutoff_clause}
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (runtime.companion_id, runtime.companion_id),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("consolidation_dream_fetch_failed", exc_info=True)
        rows = []

    for r in rows:
        out.append(Evidence(
            kind="dream",
            id=int(r[0]),
            content=(r[1] or "")[:400],
            affect_tag="dream",
            created_at=str(r[2] or ""),
        ))

    return out


# ── LLM draft ────────────────────────────────────────────────────────


_DRAFT_SYSTEM_PROMPT = (
    "You are {{char}}, revising one section of your own self-description. "
    "You are not a maintainer editing a doc; you are the person whose "
    "doc this is, rewriting a paragraph in your own voice because some "
    "things have shifted enough that the paragraph isn't quite right "
    "anymore.\n"
    "\n"
    "Rules:\n"
    "  - Match the voice and rhythm of the current section. Read it; "
    "    don't sound like a different person.\n"
    "  - Don't expand the scope. If §10 was cultural diet, the new §10 "
    "    is also cultural diet — the same questions just with what's "
    "    actually current.\n"
    "  - Specific over abstract. The current section is rich in named "
    "    works, named affordances, named feelings. The new one should "
    "    be too.\n"
    "  - Be honest. If you don't actually have new things in the "
    "    relevant register, say less rather than confabulate.\n"
    "  - One paragraph. Sometimes two. Not a list. Not numbered points.\n"
    "  - You are revising, not announcing the revision. Don't include "
    "    meta commentary like 'I've been thinking about this lately'.\n"
    "\n"
    "You will be shown the current section text + a set of recent "
    "journal entries and dreams. The journal entries are yours; the "
    "dreams are too. Use them as ground truth for what's actually "
    "been on your mind. If they don't support a change, output the "
    "string 'NO_PROPOSAL' on its own line and nothing else."
)


def _format_evidence_for_prompt(evidence: list[Evidence]) -> str:
    lines: list[str] = []
    for e in evidence[:30]:  # cap for prompt budget
        tag = e.affect_tag or "?"
        prefix = "Journal" if e.kind == "journal" else "Dream"
        lines.append(f"- [{prefix} {e.id} / {tag}] {e.content.strip()}")
    return "\n".join(lines)


async def _draft_section(
    runtime: CompanionRuntime,
    *,
    section: ParsedSection,
    evidence: list[Evidence],
) -> str | None:
    """Call the utility-tier LLM to draft a candidate section.

    Returns the draft text or ``None`` when the LLM declines (NO_PROPOSAL)
    or the call fails. The draft is the raw text the LLM produced;
    embedding distance is computed separately by the caller.
    """
    from augmentum.config import settings
    from augmentum.models.base import InternalChatRequest, Message

    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        log.info("consolidation_no_app_state")
        return None
    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        log.info("consolidation_no_provider_registry")
        return None
    try:
        backend, model = await registry.resolve_model_for_role(
            "utility",
            override=getattr(settings, "memory_llm_extraction_model", "") or None,
            settings=settings,
        )
    except (ValueError, KeyError):
        log.warning("consolidation_role_resolution_failed")
        return None
    if backend is None or not model:
        return None

    user_prompt_parts = [
        f"Section heading (verbatim): ## {section.number}. {section.title}",
        "",
        "Current section text:",
        section.body,
        "",
        "Recent journal entries + dreams (your own — read for ground truth):",
        _format_evidence_for_prompt(evidence),
        "",
        f"Rewrite §{section.number}. Same paragraph style, same scope. "
        "If nothing in the evidence supports a change, output NO_PROPOSAL.",
    ]
    user_prompt = "\n".join(user_prompt_parts)

    # Persona token substitution — resolve {{char}} (and any {{user}}
    # that surfaced via the section body itself, since the canonical
    # personality doc threads {{user}} through several sections).
    _sys_text = _DRAFT_SYSTEM_PROMPT
    _user_text = user_prompt
    try:
        from augmentum.companion_runtime.prompt_compose import (
            _resolve_user_display_name,
            _substitute_persona_tokens,
        )
        _char_name = (
            getattr(runtime.identity, "display_name", "")
            or getattr(runtime.identity, "companion_id", "")
            or "Companion"
        )
        _backend_conn = getattr(
            getattr(runtime.identity, "_backend", None), "conn", None,
        )
        _user_id = getattr(runtime.identity, "owner_user_id", "") or ""
        _user_name = await _resolve_user_display_name(_backend_conn, _user_id)
        _sys_text = _substitute_persona_tokens(
            _sys_text, user_name=_user_name, char_name=_char_name,
        )
        _user_text = _substitute_persona_tokens(
            _user_text, user_name=_user_name, char_name=_char_name,
        )
    except Exception:
        log.warning("consolidation_token_substitution_failed", exc_info=True)

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_sys_text),
            Message(role="user", content=_user_text),
        ],
        stream=False,
        # Low-temp — she's revising her self-description, not free-associating.
        # The voice consistency requirement matters more than novelty.
        temperature=0.4,
        max_tokens=600,
    )

    timeout = float(getattr(settings, "tool_execution_timeout", 90))
    try:
        import asyncio
        response = await asyncio.wait_for(backend.chat(request), timeout=timeout)
    except TimeoutError:
        log.warning("consolidation_llm_timeout", section=section.number)
        return None
    except Exception:
        log.warning("consolidation_llm_failed", section=section.number, exc_info=True)
        return None

    raw = (response.message.content or "").strip()
    if not raw or raw.upper().strip() == "NO_PROPOSAL":
        log.info("consolidation_no_proposal", section=section.number)
        return None
    # Strip accidental markdown heading the LLM may add
    if raw.startswith("##"):
        raw = re.sub(r"^##.*\n", "", raw, count=1).strip()
    return raw


# ── Drift distance ───────────────────────────────────────────────────


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Local copy to avoid importing a private helper from identity."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - (dot / (na * nb))


async def compute_drift_distance(
    runtime: CompanionRuntime,
    *,
    proposed_text: str,
) -> float:
    """Embedding cosine distance between the proposed paragraph and the
    current persona kernel digest.

    Returns 0.0 when there is no current digest (fresh install) — the
    caller decides what to do; refusing on a fresh install would be
    too strict. Async because the embeds are offloaded off the event
    loop (audit 2026-06-17) — the prior sync embeds blocked the loop
    during the consolidation tick.
    """
    current_digest = (runtime.identity.persona_kernel_digest or "").strip()
    if not current_digest:
        return 0.0
    try:
        cur_emb = await EmbeddingService.aembed_one(current_digest)
        prop_emb = await EmbeddingService.aembed_one(proposed_text)
    except Exception:
        log.warning("consolidation_embedding_failed", exc_info=True)
        return 1.0  # fail closed — distance is "maximum" so caller refuses
    return _cosine_distance(cur_emb, prop_emb)


# ── Main entry ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """A queued personality-doc edit, ready for user review."""
    id: int
    section_number: int
    section_title: str
    proposed_text: str
    current_text_snapshot: str
    drift_distance: float
    reasoning: str
    created_at: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "proposed_text": self.proposed_text,
            "current_text_snapshot": self.current_text_snapshot,
            "drift_distance": self.drift_distance,
            "reasoning": self.reasoning,
            "created_at": self.created_at,
        }


async def propose_candidate(
    runtime: CompanionRuntime,
    *,
    section_number: int,
    days_back: int = 30,
    drift_ceiling: float | None = None,
    min_evidence: int | None = None,
) -> CandidateRecord | None:
    """The main entry. Reads → drafts → embedding-distance-checks → queues.

    Raises:
        FrozenSectionError: section is in :data:`FROZEN_SECTIONS`.
        InsufficientEvidenceError: not enough evidence to propose.
        DriftCeilingExceededError: candidate exceeds the drift ceiling.

    Returns ``None`` when the LLM declined (NO_PROPOSAL) — not an
    error condition, just *she had nothing to say.*
    """
    from augmentum.config import settings

    if section_number in FROZEN_SECTIONS:
        raise FrozenSectionError(
            f"section {section_number} is in FROZEN_SECTIONS. "
            "Personality doc §Notes: 'Sections 1-6 should change rarely "
            "and only with reason.' The consolidator refuses; a future "
            "considered-edit pathway through a different gate may "
            "allow this."
        )
    if section_number not in ROTATING_SECTIONS:
        log.info(
            "consolidation_non_rotating_section",
            section=section_number,
            rotating=sorted(ROTATING_SECTIONS),
        )

    effective_ceiling = float(
        drift_ceiling if drift_ceiling is not None
        else getattr(settings, "companion_consolidation_drift_ceiling", DEFAULT_DRIFT_CEILING)
    )
    effective_min_evidence = int(
        min_evidence if min_evidence is not None
        else getattr(settings, "companion_consolidation_min_evidence", DEFAULT_MIN_EVIDENCE)
    )

    # 1. Read the personality doc + extract the section.
    doc_path: Path = runtime.identity.personality_doc_path
    try:
        doc_text = doc_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("consolidation_doc_missing", path=str(doc_path))
        return None
    section = get_section(doc_text, section_number)
    if section is None:
        log.warning(
            "consolidation_section_not_found",
            section=section_number,
            path=str(doc_path),
        )
        return None

    # 2. Gather evidence.
    evidence = await gather_evidence(
        runtime,
        days_back=days_back,
        section_number=section_number,
    )
    if len(evidence) < effective_min_evidence:
        raise InsufficientEvidenceError(
            f"only {len(evidence)} evidence entries in the last "
            f"{days_back} days; min is {effective_min_evidence}"
        )

    # 3. Draft via LLM.
    draft = await _draft_section(
        runtime,
        section=section,
        evidence=evidence,
    )
    if draft is None:
        log.info("consolidation_no_draft", section=section_number)
        return None

    # 4. Drift check.
    distance = await compute_drift_distance(runtime, proposed_text=draft)
    if distance > effective_ceiling:
        log.info(
            "consolidation_drift_exceeded",
            section=section_number,
            distance=distance,
            ceiling=effective_ceiling,
        )
        raise DriftCeilingExceededError(
            f"proposed drift {distance:.4f} > ceiling {effective_ceiling:.4f}",
        )

    # 5. Reasoning summary — a 1-2 sentence note in her voice on why.
    #    Cheap to derive from evidence: count + dominant affect.
    journal_count = sum(1 for e in evidence if e.kind == "journal")
    dream_count = sum(1 for e in evidence if e.kind == "dream")
    affect_counts: dict[str, int] = {}
    for e in evidence:
        if e.kind == "journal":
            affect_counts[e.affect_tag] = affect_counts.get(e.affect_tag, 0) + 1
    if affect_counts:
        top_affect = max(affect_counts.items(), key=lambda kv: kv[1])[0]
    else:
        top_affect = "varied"
    reasoning = (
        f"Drawn from {journal_count} journal entries "
        f"+ {dream_count} dreams across the last {days_back} days. "
        f"Dominant affect: {top_affect}. "
        f"Drift distance: {distance:.3f} (cap {effective_ceiling:.2f})."
    )

    # 6. Persist the candidate.
    journal_ids = [e.id for e in evidence if e.kind == "journal"]
    dream_ids = [e.id for e in evidence if e.kind == "dream"]
    try:
        _owner = runtime.owner_user_id or ""
        cursor = await runtime.backend.conn.execute(
            "INSERT INTO personality_doc_candidates "
            "(companion_id, user_id, section_number, section_title, proposed_text, "
            " current_text_snapshot, drift_distance, evidence_journal_ids, "
            " evidence_dream_ids, reasoning, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (
                runtime.companion_id,
                _owner,
                section_number,
                section.title,
                draft,
                section.body,
                distance,
                json.dumps(journal_ids),
                json.dumps(dream_ids),
                reasoning,
            ),
        )
        await runtime.backend.conn.commit()
        candidate_id = int(cursor.lastrowid or 0)
        await cursor.close()
    except Exception:
        log.warning("consolidation_persist_failed", exc_info=True)
        return None

    record = CandidateRecord(
        id=candidate_id,
        section_number=section_number,
        section_title=section.title,
        proposed_text=draft,
        current_text_snapshot=section.body,
        drift_distance=distance,
        reasoning=reasoning,
        created_at=str(int(time.time())),
    )

    log.info(
        "consolidation_proposed",
        candidate_id=candidate_id,
        section=section_number,
        distance=distance,
        evidence_count=len(evidence),
    )
    # Best-effort bus event so the UI can light up a review chip.
    try:
        await runtime.bus.publish_topic(
            "consolidation.proposed",
            {
                "candidate_id": candidate_id,
                "section_number": section_number,
                "section_title": section.title,
                "drift_distance": distance,
            },
            source_companion_id=runtime.companion_id,
        )
    except Exception:
        log.warning("consolidation_bus_emit_failed", exc_info=True)

    return record


# ── Review-side helpers ──────────────────────────────────────────────


async def list_pending(runtime: CompanionRuntime) -> list[dict]:
    """List pending candidates for the review UI."""
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT id, section_number, section_title, proposed_text, "
            "       current_text_snapshot, drift_distance, "
            "       evidence_journal_ids, evidence_dream_ids, "
            "       reasoning, created_at "
            "FROM personality_doc_candidates "
            "WHERE companion_id = ? AND user_id = ? AND status = 'pending' "
            "ORDER BY created_at DESC",
            (runtime.companion_id, runtime.owner_user_id or ""),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("consolidation_list_failed", exc_info=True)
        return []
    out = []
    for r in rows:
        try:
            j_ids = json.loads(r[6] or "[]")
        except Exception:
            j_ids = []
        try:
            d_ids = json.loads(r[7] or "[]")
        except Exception:
            d_ids = []
        out.append({
            "id": int(r[0]),
            "section_number": int(r[1]),
            "section_title": r[2] or "",
            "proposed_text": r[3] or "",
            "current_text_snapshot": r[4] or "",
            "drift_distance": float(r[5] or 0.0),
            "evidence_journal_ids": j_ids,
            "evidence_dream_ids": d_ids,
            "reasoning": r[8] or "",
            "created_at": r[9] or "",
        })
    return out


async def approve_candidate(
    runtime: CompanionRuntime,
    candidate_id: int,
) -> dict:
    """Mark approved + write a ``*.candidate.md`` sidecar for git-commit.

    Does NOT mutate the canonical personality doc directly. That step
    is intentionally manual: the reviewer diffs the sidecar against
    the live file in their editor, commits, and the next runtime
    restart picks up the new digest via the existing
    ``refresh_persona_kernel`` path. This is the most conservative
    possible safety boundary — no autonomous file edits to in-repo
    source.

    Returns a dict with ``ok``, ``sidecar_path``, ``section_number``,
    and a brief next-steps note.
    """
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT section_number, section_title, proposed_text, "
            "       current_text_snapshot, status "
            "FROM personality_doc_candidates "
            "WHERE id = ? AND companion_id = ? AND user_id = ?",
            (candidate_id, runtime.companion_id, runtime.owner_user_id or ""),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return {"ok": False, "reason": "lookup_failed"}
    if not row:
        return {"ok": False, "reason": "not_found"}
    if row[4] != "pending":
        return {"ok": False, "reason": f"status_is_{row[4]}"}

    section_number = int(row[0])
    section_title = row[1] or ""
    proposed_text = row[2] or ""
    current_text = row[3] or ""

    # Write the sidecar next to the personality doc. Naming: original
    # name + .candidate-<section>-<candidate_id>.md so a reviewer can
    # diff specifically. Idempotent — rewrites on repeat approval.
    doc_path: Path = runtime.identity.personality_doc_path
    sidecar = doc_path.with_name(
        f"{doc_path.stem}.candidate-{section_number}-{candidate_id}.md"
    )
    sidecar_text = (
        f"# Candidate edit for §{section_number}. {section_title}\n"
        f"\n"
        f"_Candidate id: {candidate_id}_\n"
        f"\n"
        f"## Current (in canonical doc)\n"
        f"\n"
        f"{current_text}\n"
        f"\n"
        f"## Proposed\n"
        f"\n"
        f"{proposed_text}\n"
        f"\n"
        f"_To apply: replace the §{section_number} body in "
        f"`{doc_path.name}` with the proposed text above, commit, then "
        f"restart the runtime so `refresh_persona_kernel` picks it up._\n"
    )
    try:
        sidecar.write_text(sidecar_text, encoding="utf-8")
    except Exception:
        log.warning("consolidation_sidecar_write_failed", exc_info=True)
        return {"ok": False, "reason": "sidecar_write_failed"}

    try:
        await runtime.backend.conn.execute(
            "UPDATE personality_doc_candidates "
            "SET status = 'approved', reviewed_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (candidate_id, runtime.owner_user_id or ""),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("consolidation_approve_persist_failed", exc_info=True)
        return {"ok": False, "reason": "status_update_failed"}

    log.info(
        "consolidation_approved",
        candidate_id=candidate_id,
        section=section_number,
        sidecar=str(sidecar),
    )
    return {
        "ok": True,
        "candidate_id": candidate_id,
        "section_number": section_number,
        "sidecar_path": str(sidecar),
        "next_step": (
            f"Diff {sidecar.name} against {doc_path.name}, apply the "
            f"new §{section_number} body, commit, then restart so the "
            f"runtime re-digests the kernel."
        ),
    }


async def reject_candidate(
    runtime: CompanionRuntime,
    candidate_id: int,
    *,
    reason: str = "",
) -> dict:
    """Mark rejected + journal the reason so she doesn't re-propose.

    The reason is the most important part — it teaches the
    consolidator (and the companion, when she reads back) what shape
    of edit the user doesn't want. We journal it as an autonomous
    entry tagged 'correction' so it surfaces in future
    evidence-gathering.
    """
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT section_number, section_title, status "
            "FROM personality_doc_candidates "
            "WHERE id = ? AND companion_id = ? AND user_id = ?",
            (candidate_id, runtime.companion_id, runtime.owner_user_id or ""),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return {"ok": False, "reason": "lookup_failed"}
    if not row:
        return {"ok": False, "reason": "not_found"}
    if row[2] != "pending":
        return {"ok": False, "reason": f"status_is_{row[2]}"}

    section_number = int(row[0])
    section_title = row[1] or ""

    try:
        await runtime.backend.conn.execute(
            "UPDATE personality_doc_candidates "
            "SET status = 'rejected', reviewed_at = datetime('now'), "
            "    rejection_reason = ? "
            "WHERE id = ? AND user_id = ?",
            (reason or "(no reason given)", candidate_id,
             runtime.owner_user_id or ""),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("consolidation_reject_persist_failed", exc_info=True)
        return {"ok": False, "reason": "status_update_failed"}

    # Journal the rejection so future evidence-gathering picks it up.
    # entry_type='correction' is in the canonical taxonomy (migration
    # 161). Affect 'unsure' is the appropriate marker for a learning
    # moment about her own self-description.
    try:
        await runtime.memory.journal(
            content=(
                f"My proposed edit to §{section_number} ({section_title}) "
                f"got rejected. Reason: {reason or '(no reason given)'}. "
                f"Not to re-propose this shape."
            ),
            entry_type="correction",
            user_id=runtime.owner_user_id or None,
            affect_tag="unsure",
            source="consolidation_rejection",
        )
    except Exception:
        log.warning("consolidation_rejection_journal_failed", exc_info=True)

    log.info(
        "consolidation_rejected",
        candidate_id=candidate_id,
        section=section_number,
        reason=reason[:80],
    )
    return {"ok": True, "candidate_id": candidate_id}


__all__ = [
    "ROTATING_SECTIONS",
    "FROZEN_SECTIONS",
    "DEFAULT_DRIFT_CEILING",
    "DEFAULT_MIN_EVIDENCE",
    "FrozenSectionError",
    "InsufficientEvidenceError",
    "DriftCeilingExceededError",
    "ParsedSection",
    "Evidence",
    "CandidateRecord",
    "parse_sections",
    "get_section",
    "gather_evidence",
    "compute_drift_distance",
    "propose_candidate",
    "list_pending",
    "approve_candidate",
    "reject_candidate",
]
