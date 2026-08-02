"""Ask-about-your-own-data read verbs — wiring program Phase 3.

The app collects taste signal, playtime, job status, call history,
resource telemetry, and health signals for its UI surfaces; until
2026-06-12 the companion could reach NONE of it — "what should I
watch?" got a generic answer over a database that knows the answer.

All six are headless SELECT-wrappers: they return grounded material
as a ``prompt_addendum`` (the model composes the spoken reply from
it, same shape as memory.recall) and never touch a surface. Digests
into the results ring are indexical. Individual verbs with sharp
first sentences beat one ``my.stats(domain)`` — the verb-family
disambiguation lesson — with roster cost handled by relevance
ranking.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


def _conn(session: SessionContext):
    sm = getattr(session.app_state, "state_manager", None) if session.app_state else None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


def _no_user() -> ActionResult:
    return ActionResult(
        short_circuit=True,
        speak="I can't see personal data for a signed-out session.",
    )


def _addendum(tag: str, lines: list[str], guidance: str = "") -> ActionResult:
    body = "\n".join(lines) if lines else "(nothing recorded yet)"
    extra = f"\n{guidance}" if guidance else ""
    return ActionResult(
        short_circuit=False,
        prompt_addendum=f"<{tag}>\n{body}{extra}\n</{tag}>",
        digest=f"{tag.replace('_', ' ')} fetched — details available",
    )


# ---------------------------------------------------------------------------
# my.taste — the recommendation gap
# ---------------------------------------------------------------------------

async def _my_taste(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return _no_user()
    lines: list[str] = []

    # Interest clusters — what they keep coming back to.
    try:
        store = getattr(session.app_state, "discovery_store", None)
        if store is not None:
            clusters = await store.list_clusters(
                include_dampened=False, user_id=session.user_id,
            )
            for c in (clusters or [])[:6]:
                name = c.get("name") or c.get("cluster_id") or ""
                if name:
                    lines.append(f"interest: {name}")
    except Exception:  # noqa: BLE001
        log.warning("my_taste_clusters_failed", exc_info=True)

    # Recent + favorite plays — what they actually consume.
    try:
        reg = getattr(session.app_state, "device_registry", None)
        hist = getattr(reg, "_history_store", None) if reg else None
        if hist is not None:
            recent = await hist.recent_for_kind(
                user_id=session.user_id, content_kind="", limit=8,
            )
            for r in recent or []:
                label = r.get("content_label") or ""
                kind = r.get("content_kind") or "media"
                if label:
                    lines.append(f"recently played: {label} ({kind})")
            favs = await hist.favorites_for_kind(
                user_id=session.user_id, content_kind="", limit=5,
            )
            for r in favs or []:
                label = r.get("content_label") or ""
                if label:
                    lines.append(f"favorite: {label}")
    except Exception:  # noqa: BLE001
        log.warning("my_taste_history_failed", exc_info=True)

    return _addendum(
        "taste_profile", lines,
        guidance=(
            "Recommend from THIS — their real interests and plays, not "
            "generic picks. If they want one started, media.play / "
            "grove.play_matching take it from here."
        ),
    )


register_action(
    id="my.taste",
    summary=(
        "Silently fetch the user's real taste profile — interest "
        "clusters, recent plays, favorites — so 'what should I watch/"
        "read/listen to?' gets a grounded recommendation instead of a "
        "generic one. Read-only; chain into media.play or "
        "grove.play_matching to actually start something. Siblings: "
        "hours-played stats are my.playtime."
    ),
    # Examples are QUESTION-shaped on purpose: my.taste answers "what
    # should I…?"; COMMAND-shaped asks ("put on some jazz") belong to
    # grove.play_matching, and sharing its lexical surface here once
    # clipped grove out of the roster on exactly those turns
    # (2026-06-12 roster audit).
    examples=[
        "what should I watch tonight", "recommend me something to read",
        "any suggestions for me", "what do I usually like",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_my_taste,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# my.playtime
# ---------------------------------------------------------------------------

async def _my_playtime(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return _no_user()
    conn = _conn(session)
    if conn is None:
        return ActionResult(short_circuit=True, speak="I can't reach play history right now.")
    lines: list[str] = []
    try:
        cur = await conn.execute(
            """SELECT tr.artifact_id, SUM(tr.duration_s) AS total_s,
                      COUNT(*) AS runs, a.display_name
               FROM title_runs tr
               LEFT JOIN artifacts a ON a.id = tr.artifact_id
               WHERE tr.user_id = ? AND tr.duration_s IS NOT NULL
               GROUP BY tr.artifact_id
               ORDER BY total_s DESC LIMIT 6""",
            (session.user_id,),
        )
        for row in await cur.fetchall():
            name = row[3] or row[0] or "unknown title"
            hours = (row[1] or 0) / 3600
            stamp = f"{hours:.1f}h" if hours >= 1 else f"{int((row[1] or 0) / 60)}m"
            lines.append(f"game: {name} — {stamp} across {row[2]} sessions")
    except Exception:  # noqa: BLE001
        log.warning("my_playtime_titles_failed", exc_info=True)
    try:
        cur = await conn.execute(
            """SELECT game_id, MAX(score), COUNT(*)
               FROM game_results WHERE user_id = ?
               GROUP BY game_id ORDER BY COUNT(*) DESC LIMIT 5""",
            (session.user_id,),
        )
        for row in await cur.fetchall():
            lines.append(
                f"word game: {row[0]} — best score {row[1]}, {row[2]} plays"
            )
    except Exception:  # noqa: BLE001
        log.warning("my_playtime_results_failed", exc_info=True)
    return _addendum("playtime_stats", lines)


register_action(
    id="my.playtime",
    summary=(
        "Silently fetch the user's play statistics — hours per game, "
        "session counts, best scores — for 'how long have I played X' "
        "or 'what's my high score'. Read-only stats; picking something "
        "NEW to play is my.taste."
    ),
    examples=[
        "how many hours have I put into that game",
        "what's my high score in bubble pop",
        "which game do I play the most",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_my_playtime,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# my.jobs
# ---------------------------------------------------------------------------

async def _my_jobs(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return _no_user()
    store = getattr(session.app_state, "jobs_store", None) if session.app_state else None
    if store is None:
        return ActionResult(short_circuit=True, speak="I can't reach the job queue right now.")
    try:
        jobs = await store.list_for_user(user_id=session.user_id, limit=10)
    except Exception:  # noqa: BLE001
        log.warning("my_jobs_failed", exc_info=True)
        return ActionResult(short_circuit=True, speak="I couldn't read the job queue.")
    lines = []
    for j in jobs or []:
        status = j.get("status") or "?"
        stage = j.get("stage") or ""
        prog = j.get("progress")
        pct = f" {int(float(prog) * 100)}%" if status == "running" and prog else ""
        bits = f"{j.get('job_type') or 'job'}: {status}{pct}"
        if stage and status == "running":
            bits += f" ({stage})"
        if status == "failed" and j.get("error"):
            bits += f" — {str(j['error'])[:80]}"
        lines.append(bits)

    # Image queue snapshot — a different queue from background_jobs
    # (in-memory GPU queue), but "is my image done?" is the same ask.
    # Without this leg the verb was blind to exactly the long-horizon
    # work she most often starts herself (2026-06-12 audit).
    try:
        queue = getattr(session.app_state, "image_queue", None)
        if queue is not None:
            current_id = getattr(queue, "_current_job_id", None)
            job = queue.get_job(current_id) if current_id else None
            if job is not None:
                stage = getattr(job, "stage", "") or "generating"
                steps = ""
                if getattr(job, "steps_total", 0):
                    steps = f" ({job.steps_done}/{job.steps_total})"
                lines.append(f"image generation: running — {stage}{steps}")
            pending = int(getattr(queue, "queue_size", 0) or 0)
            if pending:
                lines.append(f"image generation: {pending} more queued")
    except Exception:  # noqa: BLE001
        log.warning("my_jobs_image_queue_failed", exc_info=True)

    return _addendum(
        "background_jobs", lines,
        guidance="Most recent first. Speak plainly about failures.",
    )


register_action(
    id="my.jobs",
    summary=(
        "Silently check the user's background jobs — transcriptions, "
        "downloads, conversions — with status, progress, and errors. "
        "Call for 'is my transcription done?', 'did the download "
        "finish?', 'why did that import fail?'."
    ),
    examples=[
        "is my transcription done yet", "did the model download finish",
        "what happened to that book import",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_my_jobs,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# my.calls
# ---------------------------------------------------------------------------

async def _my_calls(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return _no_user()
    conn = _conn(session)
    if conn is None:
        return ActionResult(short_circuit=True, speak="I can't reach call history right now.")
    try:
        from augmentum.connect.call_store import list_calls_for_user
        calls = await list_calls_for_user(
            conn, user_id=session.user_id, limit=10,
        )
    except Exception:  # noqa: BLE001
        log.warning("my_calls_failed", exc_info=True)
        return ActionResult(short_circuit=True, speak="I couldn't read call history.")
    lines = []
    for c in calls or []:
        other = getattr(c, "receiver_did", "") or getattr(c, "initiator_did", "")
        state = getattr(c, "state", "")
        when = getattr(c, "initiated_at", "")
        lines.append(f"call with {other or 'unknown'}: {state}, {when}")
    return _addendum("call_history", lines)


register_action(
    id="my.calls",
    summary=(
        "Silently check the user's call history — who, when, missed or "
        "connected. Call for 'when did I last call X?', 'did I miss "
        "any calls?'."
    ),
    examples=[
        "when did I last talk to dad", "did I miss any calls today",
        "how long was my last call",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_my_calls,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# system.health
# ---------------------------------------------------------------------------

async def _system_health(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    conn = _conn(session)
    lines: list[str] = []
    degraded = bool(getattr(session.app_state, "persistence_degraded", False)) \
        if session.app_state else False
    if degraded:
        lines.append(
            "PERSISTENCE DEGRADED — the database is running in a "
            "fallback mode; saves may not survive a restart."
        )
    if conn is not None:
        try:
            cur = await conn.execute(
                """SELECT timestamp, gpu_used_mb, gpu_total_mb,
                          ram_used_mb, ram_total_mb, loaded_models_json
                   FROM resource_snapshots
                   ORDER BY timestamp DESC LIMIT 1""",
            )
            row = await cur.fetchone()
            if row:
                if row[2]:
                    lines.append(f"GPU: {row[1]} / {row[2]} MB used")
                if row[4]:
                    lines.append(f"RAM: {row[3]} / {row[4]} MB used")
                if row[5]:
                    lines.append(f"loaded models: {row[5]}")
                lines.append(f"snapshot taken: {row[0]}")
        except Exception:  # noqa: BLE001
            log.warning("system_health_snapshot_failed", exc_info=True)
    return _addendum(
        "system_health", lines,
        guidance=(
            "Answer in your own voice — this is how loaded the machine "
            "you run on is, not a dashboard dump."
        ),
    )


register_action(
    id="system.health",
    summary=(
        "Silently check how loaded the machine is — GPU/RAM headroom, "
        "loaded models, persistence health. Call for 'how are you "
        "doing, load-wise?', 'is the server struggling?', 'can we fit "
        "a bigger model?'. Sibling: things that are BROKEN in the "
        "setup are system.signals."
    ),
    examples=[
        "how are you doing load-wise", "is the gpu maxed out",
        "is the server healthy right now",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_system_health,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# system.signals
# ---------------------------------------------------------------------------

async def _system_signals(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return _no_user()
    conn = _conn(session)
    if conn is None:
        return ActionResult(short_circuit=True, speak="I can't reach the signal ledger right now.")
    lines: list[str] = []
    try:
        cur = await conn.execute(
            """SELECT source, category, summary, occurrence_count
               FROM signal_events
               WHERE user_id = ? AND status = 'open'
               ORDER BY last_seen_at DESC LIMIT 10""",
            (session.user_id,),
        )
        for row in await cur.fetchall():
            count = f" (seen {row[3]}x)" if (row[3] or 0) > 1 else ""
            lines.append(f"[{row[1]}] {row[2]}{count} — via {row[0]}")
    except Exception:  # noqa: BLE001
        log.warning("system_signals_failed", exc_info=True)
        return ActionResult(short_circuit=True, speak="I couldn't read the signal ledger.")
    return _addendum(
        "open_signals", lines,
        guidance="These are open issues the system noticed. Plain talk, no alarmism.",
    )


register_action(
    id="system.signals",
    summary=(
        "Silently list open issues the system has noticed about "
        "itself — confirmed bugs, drift, gaps from the signal ledger. "
        "Call for 'what's broken in my setup?', 'any known issues?'. "
        "Sibling: machine load/headroom is system.health."
    ),
    examples=[
        "what's broken in my setup", "any known issues right now",
        "did the bug finder turn anything up",
    ],
    arg_schema={},
    fanout=_TIER3_ONLY,
    handler=_system_signals,
    delivery="verbal",
)
