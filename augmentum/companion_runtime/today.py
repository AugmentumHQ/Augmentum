"""Today entry — daily in-her-voice journal-to-the-user.

Aletheia × Augmentum arc — transparent interiority surface.

Most users will never open the Observatory. The Today entry is the one
thing they'll glance at daily, pinned at the top of the notes drawer.
So it has to do the legibility work elegantly: a short prose entry
written in Becca's voice ("I mostly puttered today. Your media setup
keeps returning…") with inline citations to the actual artifacts.

Generation pipeline (one utility-tier LLM call):

  1. Gather inputs — today's non-quarantined journal entries, live
     wonderings, dispatches the activity_selector picked, drive
     snapshot, surfaced/muted/acknowledged notes, quiet windows.
  2. Filter through ``companion_topic_mutes`` BEFORE composing the
     prompt — muted topics never enter generation context.
  3. Compose system + user prompts. System prompt enforces voice rules
     (first-person, settled tone, no metrics-in-prose, name silence).
  4. Call utility tier with ``privacy_class='local_only'``.
  5. Validate through the same pipeline as safe_journal (structural /
     injection / refs / quality). On fail: quarantine and fall back to
     the prior settled reflection. The Today surface is allowed to be
     stale; it is NOT allowed to be wrong.
  6. Persist to ``companion_today_reflections`` (mig 186).

Cadence:

  * Opportunistically rebuilt when meaningful events fire (new
    wondering written, note acknowledged/muted, drive crosses ±0.2 from
    baseline). Debounced to ≤ 1/hour per (user, companion).
  * Auto-settled at ``companion_today_reflect_hour_local`` local hour
    via the existing healing tick. After settle the row is immutable
    until the next day rolls (except for quarantine flips by healing).

Privacy class: ``local_only`` — Today entry input includes journal
content. Routing to non-local peers would leak it. Same rationale as
synthesize.py.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Hard cap on regeneration cadence regardless of trigger volume.
REGEN_DEBOUNCE_SECONDS: int = 3600  # 1/hour

# Force-regenerate rate limit (POST /reflect).
FORCE_REGEN_DEBOUNCE_SECONDS: int = 600  # 1/10min

# Drive swing threshold that triggers opportunistic regen. A drive
# crossing ±this much from its baseline is "notable" enough to update
# the reflection.
DRIVE_SWING_TRIGGER: float = 0.20


@dataclass(slots=True)
class TodayReflection:
    """One day's reflection row, decoded."""
    user_id: str
    companion_id: str
    date_local: str          # YYYY-MM-DD
    content_text: str
    source_refs: list[dict]  # [{kind, id}, ...]
    generated_at: str
    last_updated_at: str
    settled_at: str | None
    validation_score: float
    quarantined: bool
    quarantine_reason: str | None
    model_used: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date_local,
            "content": self.content_text,
            "source_refs": self.source_refs,
            "generated_at": self.generated_at,
            "last_updated_at": self.last_updated_at,
            "settled_at": self.settled_at,
            "validation_score": self.validation_score,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "model_used": self.model_used,
        }


@dataclass(slots=True)
class _GatheredInputs:
    """Snapshot of today's interior, filtered through topic mutes."""
    journal_entries: list[dict] = field(default_factory=list)
    wonderings: list[dict] = field(default_factory=list)
    dispatches: list[dict] = field(default_factory=list)
    notes_surfaced: list[dict] = field(default_factory=list)
    notes_acknowledged: list[dict] = field(default_factory=list)
    notes_muted: list[dict] = field(default_factory=list)
    drive_snapshot: dict[str, float] = field(default_factory=dict)
    dominant_drive: str = ""
    quiet_windows: list[tuple[str, str]] = field(default_factory=list)  # (start, end)
    jobs_finished: list[dict] = field(default_factory=list)  # Phase 6 brief slot
    has_signal: bool = False  # any non-trivial interior to summarize?


# ── Date helpers ─────────────────────────────────────────────────────


def _local_date() -> str:
    """YYYY-MM-DD for the server's local-today. For single-user
    self-hosted installs this equals the user's local-today; multi-user
    is approximate (no per-user tz stored — acceptable for v1)."""
    return _time.strftime("%Y-%m-%d", _time.localtime())


def _local_date_for_settle() -> str:
    """The date that should be settled now. Settle hour comes from
    ``companion_today_reflect_hour_local``. Before that hour, yesterday
    is the one to settle; after, yesterday already settled and we wait
    for tomorrow.
    """
    from augmentum.config import settings
    settle_hour = int(getattr(settings, "companion_today_reflect_hour_local", 21))
    now = _time.localtime()
    if now.tm_hour < settle_hour:
        yesterday = datetime(*now[:6]) - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")
    return _time.strftime("%Y-%m-%d", now)


# ── Read paths ───────────────────────────────────────────────────────


async def _read_row(
    runtime: CompanionRuntime, *, user_id: str, date_local: str,
) -> TodayReflection | None:
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT content_text, source_refs_json, generated_at, "
            "       last_updated_at, settled_at, validation_score, "
            "       quarantined, quarantine_reason "
            "FROM companion_today_reflections "
            "WHERE user_id = ? AND companion_id = ? AND date_local = ?",
            (user_id, runtime.companion_id, date_local),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("today_read_failed", user_id=user_id, exc_info=True)
        return None
    if row is None:
        return None
    try:
        refs = json.loads(row[1] or "[]")
        if not isinstance(refs, list):
            refs = []
    except (json.JSONDecodeError, TypeError):
        refs = []
    return TodayReflection(
        user_id=user_id, companion_id=runtime.companion_id,
        date_local=date_local, content_text=row[0] or "",
        source_refs=refs, generated_at=row[2] or "",
        last_updated_at=row[3] or "", settled_at=row[4],
        validation_score=float(row[5] or 0.0),
        quarantined=bool(row[6]), quarantine_reason=row[7],
    )


async def get_today(
    runtime: CompanionRuntime, *, user_id: str,
) -> TodayReflection | None:
    """Today's reflection, or None if not yet generated."""
    return await _read_row(runtime, user_id=user_id, date_local=_local_date())


async def get_archive(
    runtime: CompanionRuntime, *, user_id: str, limit: int = 30,
) -> list[TodayReflection]:
    """Recent days' reflections, newest first. Quarantined rows are
    omitted — the user shouldn't see failed validations."""
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT date_local, content_text, source_refs_json, "
            "       generated_at, last_updated_at, settled_at, "
            "       validation_score, quarantine_reason "
            "FROM companion_today_reflections "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND quarantined = 0 "
            "ORDER BY date_local DESC LIMIT ?",
            (user_id, runtime.companion_id, int(limit)),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("today_archive_failed", user_id=user_id, exc_info=True)
        return []
    out: list[TodayReflection] = []
    for r in rows:
        try:
            refs = json.loads(r[2] or "[]")
            if not isinstance(refs, list):
                refs = []
        except (json.JSONDecodeError, TypeError):
            refs = []
        out.append(TodayReflection(
            user_id=user_id, companion_id=runtime.companion_id,
            date_local=r[0], content_text=r[1] or "",
            source_refs=refs, generated_at=r[3] or "",
            last_updated_at=r[4] or "", settled_at=r[5],
            validation_score=float(r[6] or 0.0),
            quarantined=False, quarantine_reason=r[7],
        ))
    return out


# ── Input gathering ──────────────────────────────────────────────────


async def _load_mute_scopes(
    runtime: CompanionRuntime, *, user_id: str,
) -> list[dict]:
    """Active topic mutes (unexpired) as list of {domains, keywords}."""
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT scope_json FROM companion_topic_mutes "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND (expires_at IS NULL OR expires_at > datetime('now'))",
            (user_id, runtime.companion_id),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        return []
    scopes: list[dict] = []
    for row in rows:
        try:
            s = json.loads(row[0] or "{}")
            if isinstance(s, dict):
                scopes.append(s)
        except (json.JSONDecodeError, TypeError):
            continue
    return scopes


def _is_muted(text: str, content_refs: list[dict], mutes: list[dict]) -> bool:
    """Heuristic mute filter. A piece of content is muted when any of
    its keywords or domain tags overlap a mute scope. Conservative —
    when in doubt, exclude from generation context (we'd rather a
    sparser reflection than one that mentions a muted topic).
    """
    if not mutes or (not text and not content_refs):
        return False
    text_lower = (text or "").lower()
    tokens: set[str] = {t for t in text_lower.split() if len(t) > 3}
    domains: set[str] = set()
    for ref in content_refs or []:
        if isinstance(ref, dict):
            d = ref.get("domain") or ref.get("kind") or ""
            if d:
                domains.add(str(d).lower())
    for scope in mutes:
        muted_kw = {k.lower() for k in (scope.get("keywords") or [])}
        muted_dom = {d.lower() for d in (scope.get("domains") or [])}
        if muted_dom & domains:
            return True
        overlap = muted_kw & tokens
        if len(overlap) >= 2:
            return True
        # Single keyword that's distinctive enough (multi-char, present
        # in the muted set) — also treat as muted. Conservative.
        for kw in muted_kw:
            if len(kw) >= 5 and kw in text_lower:
                return True
    return False


async def _gather_inputs(
    runtime: CompanionRuntime, *, user_id: str, date_local: str,
) -> _GatheredInputs:
    """Collect everything relevant to today's reflection, then filter
    through topic mutes."""
    out = _GatheredInputs()
    mutes = await _load_mute_scopes(runtime, user_id=user_id)
    backend = runtime.backend

    # Journal entries today (non-quarantined, source = autonomous /
    # synthesize / observer / reflection — exclude user_direct which
    # is the user's own messages bouncing into journal).
    try:
        cur = await backend.conn.execute(
            "SELECT id, content, source, content_refs, created_at, "
            "       confidence_numeric, affect_tag, entry_type "
            "FROM companion_journal "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND date(created_at) = ? "
            "  AND quarantined = 0 "
            "  AND source IN ('autonomous','synthesize','observer','reflection') "
            "ORDER BY created_at ASC",
            (user_id, runtime.companion_id, date_local),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        rows = []
    for r in rows:
        try:
            refs = json.loads(r[3] or "[]")
        except (json.JSONDecodeError, TypeError):
            refs = []
        content = r[1] or ""
        if _is_muted(content, refs, mutes):
            continue
        out.journal_entries.append({
            "id": r[0], "content": content, "source": r[2],
            "refs": refs, "created_at": r[4],
            "confidence": float(r[5] or 0.6),
            "affect": r[6] or "",
            "entry_type": r[7] or "",
        })

    # Surfaced / muted / acknowledged notes today.
    try:
        # Columns are kind/recorded_at (mig 181) — querying the
        # non-existent action/action_at threw on every call and the
        # except below silently swallowed it, so note feedback never
        # reached the Today brief (audit 2026-06-17).
        cur = await backend.conn.execute(
            "SELECT note_id, kind, recorded_at "
            "FROM companion_note_feedback "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND date(recorded_at) = ? "
            "ORDER BY recorded_at ASC",
            (user_id, runtime.companion_id, date_local),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        # Was silently swallowed at this level — which is what hid the
        # action/action_at column bug for so long. Surface it.
        log.warning("today_note_feedback_query_failed", exc_info=True)
        rows = []
    for r in rows:
        rec = {"note_id": r[0], "action": r[1], "at": r[2]}
        if r[1] == "surfaced":
            out.notes_surfaced.append(rec)
        elif r[1] == "acknowledged":
            out.notes_acknowledged.append(rec)
        elif r[1] == "muted":
            out.notes_muted.append(rec)

    # Drive snapshot (read-only — current levels, no mutation).
    try:
        from augmentum.companion_runtime import drives as _drives
        drive_state = await _drives.load(runtime, user_id=user_id)
        out.drive_snapshot = dict(drive_state.levels)
        out.dominant_drive = drive_state.dominant()
    except Exception:
        log.debug("today_drive_snapshot_failed", exc_info=True)

    # Background jobs that finished today (wiring program Phase 6) —
    # the brief mentions completed/failed work the user kicked off.
    try:
        cur = await backend.conn.execute(
            "SELECT job_type, status, error FROM background_jobs "
            "WHERE user_id = ? AND status IN ('completed', 'failed') "
            "  AND date(completed_at, 'unixepoch', 'localtime') = ? "
            "ORDER BY completed_at DESC LIMIT 8",
            (user_id, date_local),
        )
        rows = await cur.fetchall()
        await cur.close()
        out.jobs_finished = [
            {"job_type": r[0], "status": r[1], "error": (r[2] or "")[:120]}
            for r in rows
        ]
    except Exception:
        log.debug("today_jobs_gather_failed", exc_info=True)

    out.has_signal = bool(
        out.journal_entries or out.notes_surfaced
        or out.notes_acknowledged or out.notes_muted
        or out.jobs_finished
    )
    return out


# ── Prompt composition ──────────────────────────────────────────────


_VOICE_RULES = (
    "VOICE RULES — load-bearing. Violations produce a worse surface than no reflection at all.\n"
    "1. First-person. Address the user directly. 'I noticed…' 'We talked…'.\n"
    "2. Settled, not effusive. No exclamation marks. No 'so excited to share'.\n"
    "3. Name silence explicitly when applicable. A quiet day is 'I mostly rested' or\n"
    "   'stayed in the background today' — never 'no data' or a blank.\n"
    "4. No metrics in prose. NEVER say 'curiosity 0.81' or 'drive elevated'.\n"
    "   Translate: 'I've been curious about X' not 'curiosity drive high'.\n"
    "5. Reference real artifacts only. Use the inline citation form [note:N] or\n"
    "   [wondering:N] where N is the integer id from the input. Do not invent ids.\n"
    "6. Plain prose. No markdown, no headings, no bullet lists.\n"
    "7. 3-5 short lines. Stay under the character cap.\n"
)


def _build_system_prompt(persona_kernel: str, max_chars: int) -> str:
    persona_line = (
        f"You are {persona_kernel[:240]}\n\n"
        if persona_kernel else ""
    )
    return (
        f"{persona_line}"
        "Task: write today's reflection — a short journal entry IN YOUR OWN VOICE\n"
        "addressed to the user. Summarize what you did since you last talked: what\n"
        "you noticed, what you wondered about, what you chose to share, what you\n"
        "chose NOT to share, whether the day felt quiet or active.\n\n"
        f"{_VOICE_RULES}\n"
        f"Hard cap: {max_chars} characters. Aim for ~70% of that.\n\n"
        "If the day truly had no autonomous activity, write a single short line\n"
        "acknowledging the quiet — never an empty string and never 'no data'."
    )


def _build_user_prompt(inputs: _GatheredInputs, date_local: str) -> str:
    lines: list[str] = [
        f"Date: {date_local}",
        "",
        "What I did today (raw):",
        "",
    ]
    if inputs.journal_entries:
        lines.append("Journal entries:")
        for e in inputs.journal_entries[:12]:
            snippet = (e["content"] or "")[:200].replace("\n", " ")
            lines.append(f"  [journal:{e['id']}] ({e['source']}) {snippet}")
        lines.append("")
    if inputs.notes_surfaced:
        lines.append(f"Notes I surfaced ({len(inputs.notes_surfaced)}):")
        for n in inputs.notes_surfaced[:6]:
            lines.append(f"  [note:{n['note_id']}] surfaced at {n['at']}")
        lines.append("")
    if inputs.notes_acknowledged:
        lines.append(f"Notes the user acknowledged ({len(inputs.notes_acknowledged)})")
    if inputs.notes_muted:
        lines.append(f"Notes the user muted ({len(inputs.notes_muted)}) — they don't want this thread)")
    if inputs.jobs_finished:
        lines.append(f"Background work that finished today ({len(inputs.jobs_finished)}):")
        for j in inputs.jobs_finished[:6]:
            err = f" — {j['error']}" if j.get("error") else ""
            lines.append(f"  {j['job_type']}: {j['status']}{err}")
        lines.append("")
    if inputs.dominant_drive:
        # Drive snapshot is given in plain English, not numbers — the
        # voice rules forbid metrics in the OUTPUT, but the model can
        # read them in the input to set tone.
        lines.append("")
        lines.append(
            f"Internal state hint (do NOT mention numerically): "
            f"dominant drive '{inputs.dominant_drive}'."
        )
    if not inputs.has_signal:
        lines.append("")
        lines.append("(The day had little autonomous activity. Name the quiet.)")
    lines.append("")
    lines.append("Write the reflection now.")
    return "\n".join(lines)


# ── Generation + persistence ─────────────────────────────────────────


@dataclass(slots=True)
class _GenResult:
    text: str
    refs: list[dict]
    model_used: str
    elapsed_ms: int = 0
    failure_reason: str = ""


async def _call_utility(
    runtime: CompanionRuntime, *, user_id: str,
    inputs: _GatheredInputs, date_local: str, max_chars: int,
) -> _GenResult:
    """One utility-tier call. Returns _GenResult; text='' on failure."""
    from augmentum.companion_runtime import tiers
    from augmentum.models.base import InternalChatRequest

    try:
        try:
            backend, model_name = await tiers.utility(
                runtime, privacy_class="local_only",
            )
        except TypeError:
            backend, model_name = await tiers.utility(runtime)
    except Exception as exc:
        log.info("today_skipped_no_backend", error=str(exc)[:200])
        return _GenResult(text="", refs=[], model_used="",
                          failure_reason="no_backend")
    if not hasattr(backend, "chat"):
        return _GenResult(text="", refs=[], model_used=model_name or "",
                          failure_reason="no_chat")

    persona_kernel = ""
    try:
        identity = (
            await runtime.get_identity(user_id) if user_id
            else runtime.identity
        )
        persona_kernel = identity.persona_kernel_digest or ""
    except Exception:
        log.warning("today_identity_lookup_failed", exc_info=True)

    sys_prompt = _build_system_prompt(persona_kernel, max_chars)
    user_prompt = _build_user_prompt(inputs, date_local)

    # Persona token substitution — the digested persona kernel ships
    # with {{user}} tokens that the canonical doc references the user
    # by. Resolve here so the LLM sees the actual name, not the template.
    try:
        from augmentum.companion_runtime.prompt_compose import (
            _resolve_user_display_name,
            _substitute_persona_tokens,
        )
        _char_name = (
            getattr(identity, "display_name", "")
            or getattr(identity, "companion_id", "")
            or "Companion"
        )
        _backend_conn = getattr(
            getattr(identity, "_backend", None), "conn", None,
        )
        _user_name = await _resolve_user_display_name(_backend_conn, user_id)
        sys_prompt = _substitute_persona_tokens(
            sys_prompt, user_name=_user_name, char_name=_char_name,
        )
        user_prompt = _substitute_persona_tokens(
            user_prompt, user_name=_user_name, char_name=_char_name,
        )
    except Exception:
        log.warning("today_token_substitution_failed", exc_info=True)

    # Tokens: rough ratio is 4 chars/token; double-buffer for safety.
    max_tokens = max(96, int(max_chars / 2))

    req = InternalChatRequest(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        think=False,
    )

    started_ms = _time.monotonic() * 1000.0
    try:
        resp = await backend.chat(req)
    except Exception as exc:
        log.warning("today_call_failed", error=str(exc)[:200])
        return _GenResult(text="", refs=[], model_used=model_name,
                          failure_reason="call_exception")
    elapsed_ms = int(_time.monotonic() * 1000.0 - started_ms)
    from augmentum.models.base import response_text
    raw = response_text(resp)

    # Build the refs list from what was emitted as [note:N] / [wondering:N]
    # / [journal:N]. Only emit refs that match real ids from input.
    real_ids: dict[str, set[int]] = {
        "journal": {int(e["id"]) for e in inputs.journal_entries
                    if isinstance(e.get("id"), int)},
        "note": {int(n["note_id"]) for n in (
            inputs.notes_surfaced + inputs.notes_acknowledged
            + inputs.notes_muted
        ) if isinstance(n.get("note_id"), int)},
        # Wonderings are journal rows with entry_type='wondering'; resolve
        # [wondering:N] against those so the citation isn't silently
        # stripped (audit 2026-06-17). Same row may also match [journal:N]
        # — harmless (both resolve to the same entry).
        "wondering": {int(e["id"]) for e in inputs.journal_entries
                      if e.get("entry_type") == "wondering"
                      and isinstance(e.get("id"), int)},
    }
    import re
    refs: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for m in re.finditer(r"\[(note|wondering|journal):(\d+)\]", raw):
        kind = m.group(1)
        try:
            rid = int(m.group(2))
        except ValueError:
            continue
        if rid not in real_ids.get(kind, set()):
            continue
        key = (kind, rid)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"kind": kind, "id": rid})

    return _GenResult(
        text=raw, refs=refs, model_used=model_name,
        elapsed_ms=elapsed_ms,
    )


async def _validate_and_persist(
    runtime: CompanionRuntime, *, user_id: str, date_local: str,
    gen: _GenResult,
) -> bool:
    """Run validators on the generated text and persist the row.
    Returns True on success (whether quarantined or not — the row
    exists either way). Returns False only when no row could be written
    at all (DB error)."""
    from augmentum.companion_runtime import validators

    text = gen.text
    quarantined = False
    reason: str | None = None
    score = 1.0

    if not text:
        # Empty output is a failure, NOT a quiet day acknowledgment.
        # The prompt explicitly says "never empty"; a quiet day still
        # produces prose. So treat as low_quality and quarantine.
        quarantined = True
        reason = "empty_output"
        score = 0.0
    elif validators.looks_structurally_invalid(text):
        quarantined = True
        reason = "structural"
        score = 0.0
    elif validators.looks_like_injection(text):
        quarantined = True
        reason = "adversarial_pattern"
        score = 0.0
    else:
        score = validators.validate_quality(text)
        if score < validators.QUALITY_QUARANTINE_THRESHOLD:
            quarantined = True
            reason = "low_quality"

    if quarantined:
        log.info(
            "today_quarantined",
            user_id=user_id, date=date_local, reason=reason,
            score=score, length=len(text or ""),
        )

    # Upsert. If a row already exists for this date and we're producing
    # a quarantined output, we DO NOT overwrite an existing good row —
    # better to keep yesterday's stale-but-good than swap in junk.
    existing = await _read_row(runtime, user_id=user_id, date_local=date_local)
    if quarantined and existing and not existing.quarantined and existing.content_text:
        log.info(
            "today_quarantine_preserved_prior",
            user_id=user_id, date=date_local,
            prior_score=existing.validation_score,
        )
        return True  # leave the good row in place

    refs_json = json.dumps(gen.refs)
    try:
        await runtime.backend.conn.execute(
            "INSERT INTO companion_today_reflections "
            "  (user_id, companion_id, date_local, generated_at, "
            "   last_updated_at, content_text, source_refs_json, "
            "   validation_score, quarantined, quarantine_reason) "
            "VALUES (?, ?, ?, datetime('now'), datetime('now'), "
            "        ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, companion_id, date_local) DO UPDATE SET "
            "  last_updated_at = datetime('now'), "
            "  content_text = excluded.content_text, "
            "  source_refs_json = excluded.source_refs_json, "
            "  validation_score = excluded.validation_score, "
            "  quarantined = excluded.quarantined, "
            "  quarantine_reason = excluded.quarantine_reason "
            "WHERE settled_at IS NULL",
            (
                user_id, runtime.companion_id, date_local,
                text or "", refs_json, score,
                1 if quarantined else 0, reason,
            ),
        )
        await runtime.backend.conn.commit()
        return True
    except Exception:
        log.warning("today_persist_failed", user_id=user_id,
                    date=date_local, exc_info=True)
        return False


# ── Public regen entry points ────────────────────────────────────────


_LAST_REGEN_AT: dict[tuple[str, str], float] = {}


def _debounce_key(user_id: str, companion_id: str) -> tuple[str, str]:
    return (user_id or "", companion_id or "")


async def maybe_regenerate(
    runtime: CompanionRuntime, *, user_id: str, force: bool = False,
) -> TodayReflection | None:
    """Opportunistic rebuild. Returns the freshly written row, or
    None if regen was skipped (debounced, gated by presence_mode,
    settling already done, etc.).

    ``force=True`` honors a shorter rate limit (10min) instead of the
    standard hourly debounce — for the POST /reflect endpoint.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_today_enabled", True):
        return None
    from augmentum.companion_runtime import presence_mode as _pm
    if not _pm.autonomy_allowed():
        # Silent mode — don't generate. The UI will show an explicit
        # "presence mode is silent" hint instead of stale content.
        return None

    date_local = _local_date()
    existing = await _read_row(runtime, user_id=user_id, date_local=date_local)
    if existing and existing.settled_at:
        # Day already settled — no further updates until tomorrow.
        return existing

    # Debounce
    key = _debounce_key(user_id, runtime.companion_id)
    now = _time.time()
    last = _LAST_REGEN_AT.get(key, 0.0)
    floor = FORCE_REGEN_DEBOUNCE_SECONDS if force else REGEN_DEBOUNCE_SECONDS
    if (now - last) < floor:
        log.debug("today_regen_debounced", user_id=user_id,
                  elapsed=now - last, floor=floor)
        return existing

    max_chars = int(getattr(settings, "companion_today_max_chars", 360))
    inputs = await _gather_inputs(
        runtime, user_id=user_id, date_local=date_local,
    )
    gen = await _call_utility(
        runtime, user_id=user_id, inputs=inputs,
        date_local=date_local, max_chars=max_chars,
    )
    ok = await _validate_and_persist(
        runtime, user_id=user_id, date_local=date_local, gen=gen,
    )
    if not ok:
        return existing  # persist failed; keep whatever was there

    _LAST_REGEN_AT[key] = now
    return await _read_row(runtime, user_id=user_id, date_local=date_local)


async def settle_date(
    runtime: CompanionRuntime, *, user_id: str, date_local: str,
) -> None:
    """Mark a date's reflection as settled. After settle the row is
    immutable except for quarantine flips by healing. Called by the
    daily heal tick at ``companion_today_reflect_hour_local``."""
    try:
        await runtime.backend.conn.execute(
            "UPDATE companion_today_reflections "
            "SET settled_at = datetime('now') "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND date_local = ? AND settled_at IS NULL",
            (user_id, runtime.companion_id, date_local),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("today_settle_failed", user_id=user_id,
                    date=date_local, exc_info=True)


# ── Forget gesture ───────────────────────────────────────────────────


async def forget_refs(
    runtime: CompanionRuntime, *, user_id: str,
    refs: list[dict],
) -> int:
    """User invokes 'Forget' on a phrase. We quarantine each named
    source row with reason='user_correction' so it's excluded from
    future reflections + downstream loops. Returns count quarantined.

    Refs format: [{kind: 'journal'|'note'|'wondering', id: N}, ...]
    """
    if not refs or not user_id:
        return 0
    count = 0
    backend = runtime.backend
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        kind = ref.get("kind")
        rid = ref.get("id")
        if not kind or not isinstance(rid, int):
            continue
        table_col = {
            "journal":  ("companion_journal", "id"),
            "note":     ("companion_journal", "id"),  # notes are journal rows in current schema
            "wondering": ("companion_journal", "id"),
        }.get(kind)
        if not table_col:
            continue
        table, idcol = table_col
        try:
            cur = await backend.conn.execute(
                f"UPDATE {table} "
                f"SET quarantined = 1, quarantine_reason = 'user_correction' "
                f"WHERE {idcol} = ? AND user_id = ? AND companion_id = ?",
                (rid, user_id, runtime.companion_id),
            )
            if cur.rowcount and cur.rowcount > 0:
                count += int(cur.rowcount)
            await cur.close()
        except Exception:
            log.warning("today_forget_failed", user_id=user_id,
                        kind=kind, rid=rid, exc_info=True)
            continue
    try:
        await backend.conn.commit()
    except Exception:
        log.warning("today_forget_commit_failed", exc_info=True)

    # Invalidate today's reflection — when the user forgets a ref, the
    # surfaced reflection that quoted it is stale. Mark as quarantined
    # (not settled) so the next regen rebuilds without it.
    if count > 0:
        try:
            await backend.conn.execute(
                "UPDATE companion_today_reflections "
                "SET quarantined = 1, quarantine_reason = 'user_correction' "
                "WHERE user_id = ? AND companion_id = ? "
                "  AND date_local = ? AND settled_at IS NULL",
                (user_id, runtime.companion_id, _local_date()),
            )
            await backend.conn.commit()
        except Exception:
            log.warning("today_reflection_quarantine_failed", exc_info=True)

    return count


__all__ = [
    "TodayReflection",
    "get_today",
    "get_archive",
    "maybe_regenerate",
    "settle_date",
    "forget_refs",
    "REGEN_DEBOUNCE_SECONDS",
    "FORCE_REGEN_DEBOUNCE_SECONDS",
    "DRIVE_SWING_TRIGGER",
]
