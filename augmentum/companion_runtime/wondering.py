"""Wondering generator — write journal entries when a topical thread
crosses the corroboration bar.

Sprint 2, Aletheia × Augmentum arc Piece 7.

The wondering generator is the *fuel for revisit_thread*. Without it,
the autonomous-research loop starves: revisit_thread filters for
``entry_type IN ('wondering', 'unfinished')`` but only this generator
+ the dream cycle write those types.

Cross-layer corroboration is the load-bearing design choice. A
topical thread *alone* is suggestive; combine it with elevated
curiosity facet activation and you have signal worth committing to
journal. Only one signal → write at ``confidence='early'`` so it
doesn't feed revisit_thread (which filters confidence ≥ 'normal').

Gates (all must pass before write):

1. Topical aggregator returns at least one thread
2. Not in hush window (``companion_journal_hushed_until``)
3. User cooldown clear (no chat in last N seconds — observer signal)
4. Daily cap not exceeded (default 3 per user per day)
5. Topic mute list (Sprint 3 Piece 12) — checked but tolerant when
   the table doesn't exist yet

Uses ``safe_journal`` so every write is validated + provenance-tagged.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.perception.topical import Thread
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Default daily cap. Tuning surface: ``companion_wondering_daily_cap``.
DEFAULT_DAILY_CAP: int = 3

# Per-topic dedup window. Same topic within this many hours doesn't
# merit a fresh wondering — the drawer was accumulating four identical
# "spent attention on cnn.com" rows per evening. 4h ~= "one media
# session cycle." Tunable via ``companion_wondering_dedup_hours``.
DEDUP_WINDOW_HOURS: int = 4


def _compose_wondering_prose(*, topic: str, event_count: int, keywords: list[str]) -> str:
    """Compose the wondering body as a short field-note sentence.

    Replaces the prior template "The user spent attention on X — N
    touches in the last few hours. Themes: A, B, C." which read like a
    log line. The new shape is plain prose that still mentions the
    topic (so the dedup LIKE query can find it) and the relevant
    themes (when they add information beyond the topic itself).

    Themes that just restate the topic ("cnn" for cnn.com, "pornhub"
    for pornhub.com) are dropped — they're noise.
    """
    topic = (topic or "").strip()
    topic_lower = topic.lower()
    topic_stem = topic_lower.split(".")[0]
    real_themes: list[str] = []
    for kw in keywords[:4]:
        kw_clean = (kw or "").strip()
        if not kw_clean:
            continue
        kw_lower = kw_clean.lower()
        if kw_lower == topic_lower or kw_lower == topic_stem:
            continue
        real_themes.append(kw_clean)
        if len(real_themes) >= 2:
            break
    theme_str = " and ".join(real_themes) if real_themes else ""
    n = max(int(event_count or 0), 1)

    # Light variation so the drawer doesn't read like a Mad Lib.
    if theme_str:
        if n >= 5:
            return f"Back to {topic} a few more times — {theme_str}."
        if n >= 3:
            return f"A few returns to {topic} today, mostly {theme_str}."
        return f"Caught {topic} again — {theme_str}."
    if n >= 5:
        return f"Back to {topic} a handful of times today."
    if n >= 3:
        return f"A few returns to {topic} this afternoon."
    return f"Caught {topic} again."


async def maybe_write_wondering(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    threads: list[Thread] | None = None,
    now: float | None = None,
) -> int | None:
    """Maybe write one wondering entry. Returns journal_id on write,
    None on skip.

    Caller is the tick loop or observer. Caller supplies threads when
    cached aggregator output is available; when ``threads`` is None,
    this function pulls and aggregates from ``observer.observed_state``.

    Idempotent guards: daily cap, hush, user-idle. All checks fail-open
    on missing infrastructure (e.g. empty observed_state, missing
    settings) so this never raises in normal operation.
    """
    if not user_id:
        return None

    from augmentum.config import settings

    # Master kill switch — Sprint 2 defaults OFF until tuning.
    if not getattr(settings, "companion_topical_aggregator_enabled", False):
        return None

    # Sprint 5 — presence_mode gate. Silent suppresses all autonomous
    # writes. This is the user-facing dial; admin flags above are
    # operator escape hatches.
    from augmentum.companion_runtime import presence_mode as _pm
    if not _pm.autonomy_allowed():
        return None

    # Hush gate — explicit user silence.
    from augmentum.companion_runtime import gates
    if gates.is_hushed_now():
        return None

    # User-active gate — don't write a wondering while user is mid-chat.
    if gates.is_user_recently_active(runtime):
        return None

    # Daily cap — counted via journal entries written today by source.
    daily_cap = int(getattr(settings, "companion_wondering_daily_cap", DEFAULT_DAILY_CAP))
    today_count = await _count_wonderings_today(runtime, user_id=user_id)
    if today_count >= daily_cap:
        log.debug(
            "wondering_skip_daily_cap",
            user_id=user_id, count=today_count, cap=daily_cap,
        )
        return None

    # Pull threads if not provided
    if threads is None:
        threads = await _aggregate_for_user(runtime, user_id=user_id, now=now)
    if not threads:
        return None

    # Pick the most recent thread that isn't muted or NSFW. The
    # observer captures every browse signal — including visits to porn
    # tubes — and aggregates them into threads. Without this gate the
    # wondering composer would emit "spent attention on pornhub.com"
    # notes from the same machinery that watches HN. safe_journal also
    # catches it downstream, but skipping here saves the LLM round-trip
    # and the quarantine row.
    try:
        from augmentum.discovery.safety import is_nsfw_text
    except Exception:
        is_nsfw_text = None
    selected: Thread | None = None
    for thread in threads:
        if await _is_topic_muted(runtime, user_id=user_id, thread=thread):
            continue
        if is_nsfw_text is not None:
            topic = (getattr(thread, "topic", "") or "")
            keywords = " ".join(getattr(thread, "keywords", []) or [])
            try:
                if is_nsfw_text(topic) or is_nsfw_text(keywords):
                    log.info(
                        "wondering_skip_nsfw_thread",
                        user_id=user_id, topic=topic[:60],
                    )
                    continue
            except Exception:
                log.debug("wondering_nsfw_check_failed", exc_info=True)
        # Dedup window — same topic already noticed within the last N
        # hours doesn't merit a fresh wondering. The visible drawer was
        # showing four identical "spent attention on cnn.com — 5 touches"
        # entries spaced an hour apart; same content_refs, same prose,
        # different rows. The fix is upstream: if there's a recent
        # wondering whose content already mentions this topic, skip.
        existing = await _existing_wondering_for_topic(
            runtime, user_id=user_id, topic=thread.topic,
            within_hours=DEDUP_WINDOW_HOURS,
        )
        if existing:
            log.info(
                "wondering_skip_dedup",
                user_id=user_id, topic=thread.topic[:60],
                prior_journal_id=existing,
                window_hours=DEDUP_WINDOW_HOURS,
            )
            continue
        selected = thread
        break
    if selected is None:
        return None

    # Cross-layer corroboration — check curiosity facet recent activation
    has_curiosity_signal = await _curiosity_elevated(runtime, user_id=user_id)

    # Build the journal entry
    confidence_numeric = 0.6 if has_curiosity_signal else 0.3
    confidence_label = "normal" if has_curiosity_signal else "early"

    content = _compose_wondering_prose(
        topic=selected.topic,
        event_count=selected.event_count,
        keywords=selected.keywords or [],
    )

    # content_refs: synthetic refs for each surface event (kind=surface
    # is a future-compat kind; the resolver tolerates it). Sprint 3+
    # may upgrade to real file_index refs when surface events carry them.
    content_refs = [
        {"kind": "surface", "id": event_id}
        for event_id in selected.event_ids[:5]  # cap for size
    ]

    journal_id = await runtime.memory.safe_journal(
        content,
        source="autonomous",
        user_id=user_id,
        entry_type="wondering",
        affect_tag="curious",
        content_refs=content_refs,
        confidence_numeric=confidence_numeric,
        origin=_thread_origin(selected),
    )
    if journal_id:
        log.info(
            "wondering_written",
            user_id=user_id, journal_id=journal_id,
            topic=selected.topic, event_count=selected.event_count,
            confidence=confidence_label,
        )
        # Best-effort Today regen — a new wondering is a meaningful
        # event surface. Debounced inside maybe_regenerate; never blocks.
        try:
            from augmentum.companion_runtime import today as _today
            await _today.maybe_regenerate(runtime, user_id=user_id)
        except Exception:
            log.debug("today_regen_from_wondering_failed", exc_info=True)
    return journal_id


# ── Internal helpers ─────────────────────────────────────────────────


def _thread_origin(thread: Thread) -> dict:
    """Provenance record for a wondering born from an attention thread.

    Persisted to companion_journal.origin_json (mig 257); the drawer's
    "why am I seeing this" chip renders it — e.g. "from browsing (web)
    · 3 visits". ``client`` is the dominant auth-session source among
    the thread's events (the emit sites stamp it server-side).
    """
    from datetime import UTC, datetime

    def _minute(ts: float) -> str:
        try:
            return datetime.fromtimestamp(
                float(ts), tz=UTC,
            ).strftime("%Y-%m-%dT%H:%M")
        except (OSError, OverflowError, TypeError, ValueError):
            return ""

    start = _minute(thread.first_seen)
    end = _minute(thread.last_seen)
    # Compact "2026-06-08T06:34/06:40" when both ends share the day.
    window = (
        f"{start}/{end[11:]}" if start and end and end[:10] == start[:10]
        else f"{start}/{end}".strip("/")
    )
    clients = getattr(thread, "clients", ()) or ()
    return {
        "source": "attention",
        "client": clients[0] if clients else "web",
        "signal_count": int(thread.event_count or 0),
        "window": window,
        "detail": f"browse: {thread.topic} x{thread.event_count}",
    }


async def _existing_wondering_for_topic(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    topic: str,
    within_hours: int,
) -> int | None:
    """Return the id of a recent non-quarantined wondering whose content
    mentions ``topic``, or None when no match.

    Cheap LIKE-based scan against companion_journal — the topic name is
    embedded into the wondering's prose by the composer, so substring
    match is a sufficient signal. Quarantined rows don't count (they
    were never visible to the user anyway).
    """
    needle = (topic or "").strip()
    if not needle:
        return None
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT id FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND entry_type = 'wondering' "
            "  AND COALESCE(quarantined, 0) = 0 "
            "  AND content LIKE ? "
            f"  AND created_at > datetime('now', '-{int(within_hours)} hours') "
            "ORDER BY created_at DESC LIMIT 1",
            (runtime.companion_id, user_id, f"%{needle}%"),
        )
        row = await cur.fetchone()
        await cur.close()
        return int(row[0]) if row else None
    except Exception:
        log.warning("wondering_dedup_query_failed", exc_info=True)
        return None


async def _count_wonderings_today(
    runtime: CompanionRuntime, *, user_id: str,
) -> int:
    """Count wondering entries written today (local-day, server tz).
    Excludes quarantined entries — those don't count against the cap
    since they wouldn't surface to revisit_thread anyway.

    Uses a local calendar-day match (not a rolling 24h window) so the
    cap actually resets at local midnight — the rolling window let a
    23:00 write keep counting at 22:00 the next day (audit 2026-06-17)."""
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND entry_type = 'wondering' "
            "  AND COALESCE(quarantined, 0) = 0 "
            "  AND date(created_at, 'localtime') = date('now', 'localtime')",
            (runtime.companion_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        return int(row[0] if row else 0)
    except Exception:
        log.warning("wondering_count_query_failed", exc_info=True)
        return 0


async def _aggregate_for_user(
    runtime: CompanionRuntime, *, user_id: str, now: float | None = None,
) -> list[Thread]:
    """Pull observer.recent and run the topical aggregator."""
    observed = getattr(runtime, "observed_state", None)
    if not observed:
        return []
    recent = observed.get("recent")
    if not recent:
        return []
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    from augmentum.config import settings

    min_events = int(getattr(settings, "companion_topical_min_events", 3))
    window_hours = float(getattr(settings, "companion_topical_window_hours", 4.0))
    return aggregate_threads(
        recent,
        user_id=user_id,
        min_events=min_events,
        window_seconds=window_hours * 3600.0,
        now=now,
        allowed_clients=allowed_attention_clients(),
    )


def allowed_attention_clients() -> frozenset[str]:
    """Parse ``companion_attention_sources`` into the client allow-set.

    Comma-separated auth-session sources (web / android / ...) whose
    surface events may form attention threads. cast_receiver is excluded
    by default — a shared TV logged in as you must not write your
    attention stream. A blank setting falls back to the default rather
    than allow-all: clearing the field shouldn't silently widen the
    privacy boundary.
    """
    from augmentum.config import settings

    raw = str(getattr(settings, "companion_attention_sources", "") or "")
    parsed = frozenset(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )
    return parsed or frozenset({"web", "android"})


async def _curiosity_elevated(
    runtime: CompanionRuntime, *, user_id: str,
) -> bool:
    """Has the 'curious' (or related) facet activated recently?

    Sprint 2 stub: returns True when there's been at least one facet
    activation tagged 'curious' or 'exploratory' in the last
    ``window_seconds``. Sprint 6 (PAD/drives) will replace this with
    a richer signal from the PAD projector.
    """
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT COUNT(*) FROM personality_facet_activations "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND facet IN ('curious', 'exploratory') "
            "  AND activated_at > datetime('now', '-1 hour')",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
        return int(row[0] if row else 0) > 0
    except Exception:
        # personality_facet_activations is mig 160; if missing, treat
        # as no signal (writes at confidence='early').
        return False


async def _is_topic_muted(
    runtime: CompanionRuntime, *, user_id: str, thread: Thread,
) -> bool:
    """Check companion_topic_mutes for an unexpired mute matching the thread.

    Mute scope is stored as JSON ``{domains: [...], keywords: [...]}``.
    A thread is muted when ANY of its domains overlaps any muted domain
    OR ≥2 of its keywords overlap a muted keyword set.

    Sprint 3 Piece 12 creates the mute table. Until then this returns
    False (no mutes in effect).
    """
    import json
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
        # Table not yet created (mig 180 lands Sprint 3) — no mutes.
        return False

    thread_domains = set(thread.domains)
    thread_keywords = set(thread.keywords)
    for row in rows:
        try:
            scope = json.loads(row[0] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        muted_domains = set(scope.get("domains") or [])
        if thread_domains & muted_domains:
            return True
        muted_keywords = set(scope.get("keywords") or [])
        if len(thread_keywords & muted_keywords) >= 2:
            return True
    return False


__all__ = [
    "maybe_write_wondering",
    "allowed_attention_clients",
    "DEFAULT_DAILY_CAP",
]
