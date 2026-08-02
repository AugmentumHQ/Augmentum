"""Lesson capture — turn observed corrections into held lessons.

The judgment half of the lesson registry (the substrate is
:mod:`augmentum.companion.lessons`). Reads recent text — her own
end-of-day reflections for the MVP — and extracts the places where the
user corrected her or stated a clear preference about how she should
behave, storing each as a :class:`~augmentum.companion.lessons.Lesson`.

**Why nightly + reflection text (MVP boundary).** The richest capture
source is the day's actual conversation turns, but becca_direct turns
live in ``ui_sessions`` JSON trees, not a clean table. So this slice
mirrors :mod:`augmentum.companion_runtime.reflection` (``maybe_apply_
nudge``): it sources from ``companion_journal`` — confirmed schema,
already swept by ``healing.daily_heal``. The extractor takes *text*, so
swapping in a real-conversation source later is a drop-in change with no
touch to the store, retrieval, or injection. Realtime capture is the
documented phase-2 upgrade.

**No regex switchboard.** Per project guidance, the "was this a
correction?" judgment is the model's, not a pattern table's. The
extractor asks her utility tier for structured JSON; :func:`_parse_lessons`
is the only deterministic piece and it just parses, it doesn't classify.

Gated by ``companion_lessons_capture_enabled`` (default OFF).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Max lessons stored from a single capture pass — bounds writes even if
# the model over-produces. A day rarely contains more than a couple of
# real corrections; the cap is a guard, not a target.
MAX_LESSONS_PER_PASS: int = 5

# Recent-journal window the nightly pass reads for capture text.
CAPTURE_WINDOW_HOURS: int = 26  # ~a day, with slack for tick jitter


_EXTRACT_SYSTEM = (
    "You review an AI companion's recent private reflections about her day "
    "with one person. Your job: find the moments where that person CORRECTED "
    "her or stated a clear preference about how she should behave — the kind "
    "of moment she should learn from so she doesn't repeat the mistake.\n\n"
    "Output ONLY a JSON array. Each element is an object with exactly:\n"
    '  "situation": when this lesson applies, short ("when he\'s debugging")\n'
    '  "trap": the mistake to avoid, short ("jumping to a fix before he\'s '
    'finished describing it")\n'
    '  "better": what to do instead, in her own first-person voice ("let him '
    'finish, then ask one question")\n\n'
    "Only include CLEAR corrections or stated preferences — not vague moods, "
    "not things that merely went well, not your own speculation. If there are "
    "no clear corrections, output []. Output nothing but the JSON array."
)


def _parse_lessons(raw: str) -> list[dict]:
    """Parse the extractor's response into a list of lesson dicts.

    Tolerant of code fences and a ``{"lessons": [...]}`` wrapper. Drops
    any element missing ``situation`` or ``better`` (``trap`` may be
    empty). Never raises — returns ``[]`` on anything unparseable.
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    # Strip a ```json ... ``` fence if present.
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else ""
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Narrow to the outermost array/object if there's surrounding prose.
    for opener, closer in (("[", "]"), ("{", "}")):
        if opener in text and closer in text:
            start = text.index(opener)
            end = text.rindex(closer) + 1
            candidate = text[start:end]
            try:
                data = json.loads(candidate)
                break
            except (json.JSONDecodeError, ValueError):
                continue
    else:
        return []

    if isinstance(data, dict):
        data = data.get("lessons") or data.get("items") or []
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        situation = str(item.get("situation") or "").strip()
        better = str(item.get("better") or "").strip()
        trap = str(item.get("trap") or "").strip()
        if not situation or not better:
            continue
        out.append({"situation": situation, "trap": trap, "better": better})
        if len(out) >= MAX_LESSONS_PER_PASS:
            break
    return out


async def extract_lessons(runtime: CompanionRuntime, text: str) -> list[dict]:
    """Ask her utility tier to extract correction-lessons from ``text``.

    Returns a list of ``{situation, trap, better}`` dicts (possibly
    empty). On any model/backend failure returns ``[]`` — capture is
    best-effort; a missing model means no lessons today, not a crash.
    """
    if not text or not text.strip():
        return []
    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import InternalChatRequest, response_text
    except Exception:
        return []
    try:
        backend, model_name = await tiers.utility(runtime)
    except Exception:
        return []
    if not hasattr(backend, "chat") or not model_name:
        return []

    req = InternalChatRequest(
        model=model_name,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text[:6000]},
        ],
        max_tokens=500,
        temperature=0.2,
    )
    try:
        resp = await backend.chat(req)
    except Exception as exc:
        log.warning("lesson_extract_failed", error=str(exc)[:200])
        return []
    return _parse_lessons(response_text(resp))


def _lesson_graph(runtime: CompanionRuntime):
    """Build a LessonGraph over the runtime's backend. Returns None when
    the backend isn't available (degraded runtime / test fixtures)."""
    backend = getattr(runtime, "backend", None)
    if backend is None:
        return None
    try:
        from augmentum.companion.lessons import LessonGraph
        return LessonGraph(
            backend, bus=getattr(runtime, "bus", None),
            companion_id=runtime.companion_id,
        )
    except Exception:
        log.debug("lesson_graph_init_failed", exc_info=True)
        return None


async def capture_from_text(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    text: str,
    source: str = "reflection",
    evidence: str = "",
) -> dict:
    """Extract correction-lessons from ``text`` and store them.

    Returns a summary dict ``{"captured": [...], "extracted": N}`` for
    the nightly log + the Observatory. Always returns a dict — never
    raises. Gated by ``companion_lessons_capture_enabled`` (default OFF).
    """
    from augmentum.config import settings
    result: dict = {"captured": [], "extracted": 0, "skipped": ""}
    if not getattr(settings, "companion_lessons_capture_enabled", False):
        result["skipped"] = "feature_disabled"
        return result
    if not user_id:
        result["skipped"] = "no_user_id"
        return result

    graph = _lesson_graph(runtime)
    if graph is None:
        result["skipped"] = "no_lesson_graph"
        return result

    extracted = await extract_lessons(runtime, text)
    result["extracted"] = len(extracted)
    for item in extracted:
        try:
            lesson = await graph.capture(
                situation=item["situation"],
                trap=item["trap"],
                better=item["better"],
                user_id=user_id,
                source=source,
                evidence=evidence,
            )
            result["captured"].append(
                {"id": lesson.id, "situation": lesson.situation},
            )
        except Exception:
            log.debug("lesson_store_failed", exc_info=True)

    if result["captured"]:
        log.info(
            "lessons_captured",
            user_id=user_id, companion_id=runtime.companion_id,
            count=len(result["captured"]), source=source,
        )
    return result


async def capture_recent(runtime: CompanionRuntime, *, user_id: str) -> dict:
    """Nightly hook: pull the user's recent reflections and capture
    lessons from them. Called from ``healing.daily_heal`` per active user.

    Sources from ``companion_journal`` (her own reflective writing) over
    the last ``CAPTURE_WINDOW_HOURS``. Empty / no-store degrades to a
    no-op summary.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_lessons_capture_enabled", False):
        return {"skipped": "feature_disabled"}
    if not user_id:
        return {"skipped": "no_user_id"}

    backend = getattr(runtime, "backend", None)
    if backend is None:
        return {"skipped": "no_backend"}

    try:
        cur = await backend.conn.execute(
            "SELECT content FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND created_at > datetime('now', ?) "
            "  AND COALESCE(quarantined, 0) = 0 "
            "ORDER BY created_at DESC LIMIT 25",
            (runtime.companion_id, user_id, f"-{CAPTURE_WINDOW_HOURS} hours"),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.debug("lesson_capture_journal_query_failed", exc_info=True)
        return {"skipped": "journal_query_failed"}

    entries = [str(r[0]).strip() for r in rows if r and r[0]]
    if not entries:
        return {"skipped": "no_recent_reflections", "extracted": 0}

    text = "\n\n".join(entries)
    return await capture_from_text(
        runtime, user_id=user_id, text=text,
        source="reflection", evidence="nightly journal sweep",
    )


__all__ = [
    "extract_lessons",
    "capture_from_text",
    "capture_recent",
    "MAX_LESSONS_PER_PASS",
]
