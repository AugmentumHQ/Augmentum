"""``watch_for`` — the conversational front door for ALL watches.

Originally URL-only (companion verbs architecture, Phase 4); extended
per the scheduled-requests-and-watches spec (2026-06-11) to a closed
kind enum over the standing-task watch kinds:

    url    → url_watch          "tell me when this page changes"
    search → recurring_search   "watch for new results about X"
    repo   → github_releases    "tell me when llama.cpp releases"
    topic  → feed_digest        "keep a daily digest on X"
    metric → metric_watch       "tell me when it's above 85°F"

Two verification layers ride along:
  * ``intent`` — a natural-language importance condition stored in
    params; the watch judge evaluates each detected change against it
    at fire time (suppressed-but-logged when it doesn't match).
  * ``condition`` — compiled quantitative rule ({op, value, unit}) for
    number-shaped asks; evaluated by CODE at fire time, never the LLM.

Creation probe: the watch runs once synchronously at creation (via
``run_now``, capped by companion_watch_probe_timeout_s) and the
observed baseline is reflected back — wrong URLs and unbindable
intents surface while the user is present to correct them, not three
weeks later. URL fetch failure refuses creation outright.
"""

from __future__ import annotations

from typing import Any

from augmentum.tools._standing_common import (
    CONFIRM_REPLACE_SCHEMA_PROPERTY,
    DELIVERY_SCHEMA_PROPERTY,
    duplicate_review,
    parse_delivery_param,
    standing_gate,
)
from augmentum.tools.base import (
    CoreVerbAutonomyClass,
    CoreVerbMetadata,
    CoreVerbSafetyClass,
    CostEnvelope,
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class WatchForTool(Tool):
    """Subscribe to URL change detection."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "watch_for"

    @property
    def description(self) -> str:
        return (
            "Watch something and tell the user when it changes. Kinds: "
            "url (page change), search (new results for a query), repo "
            "(new GitHub releases), feed (a creator/publication — "
            "YouTube channel, podcast, blog, subreddit — new posts via "
            "RSS/Atom), topic (periodic digest), metric (numeric "
            "threshold via a data provider). For 'let me know when X "
            "updates / posts / releases / drops below Y.' NOT for "
            "one-time fetches (use web_fetch)."
        )

    @property
    def model_hint(self) -> str:
        return (
            "kind=url needs target=full http(s) URL; kind=search "
            "target=the query; kind=repo target=owner/name; kind=feed "
            "target=the creator/source as the user said it (a YouTube "
            "channel URL or @handle, r/subreddit, a blog/podcast URL, "
            "or a feed URL — it gets resolved); kind=topic "
            "target=the subject. intent = the user's own filter in "
            "plain words ('only price or availability changes', 'only "
            "security releases') — include it whenever the user said "
            "WHAT they care about, not just WHERE to look. For "
            "number-shaped asks ('below $500', 'above 85') ALSO pass "
            "condition={op,value,unit}. The creation result reflects "
            "what the watch sees right now — relay it to the user."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="watch something and alert when it changes (watch_for)",
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short label for the watch (shown in "
                                   "list_briefings / notification title).",
                },
                "kind": {
                    "type": "string",
                    "enum": ["url", "search", "repo", "feed", "topic",
                             "metric"],
                    "description": "What to watch. url=page change, "
                                   "search=new results, repo=GitHub "
                                   "releases, feed=creator/publication "
                                   "new posts (YouTube/podcast/blog/"
                                   "subreddit), topic=periodic digest, "
                                   "metric=numeric provider value. "
                                   "Default url.",
                },
                "target": {
                    "type": "string",
                    "description": "The thing to watch: full http(s) URL "
                                   "(url), query text (search), owner/name "
                                   "(repo), creator as the user said it "
                                   "(feed), subject (topic).",
                },
                "url": {
                    "type": "string",
                    "description": "Legacy alias for target when kind=url.",
                },
                "intent": {
                    "type": "string",
                    "description": "The user's filter in their own words — "
                                   "'only price or availability changes', "
                                   "'only security releases'. Changes that "
                                   "don't match are logged, not delivered.",
                },
                "condition": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string",
                               "enum": ["<", ">", "<=", ">=", "=="]},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "description": "Compiled quantitative rule for "
                                   "number-shaped asks ('below $500' → "
                                   "{op:'<', value:500, unit:'USD'}). "
                                   "Evaluated by code at fire time.",
                },
                "interval_hours": {
                    "type": "number",
                    "description": "Poll cadence in hours. Default 1 for "
                                   "url, 24 otherwise. Minimum 0.5.",
                },
                "delivery": DELIVERY_SCHEMA_PROPERTY,
                "confirm_replace": CONFIRM_REPLACE_SCHEMA_PROPERTY,
            },
            "required": ["title"],
        }

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.WRITE_SELF,
            autonomy_class=CoreVerbAutonomyClass.EXPLICIT,
            cost_envelope=CostEnvelope(max_wallclock_ms=3_000, max_db_ops=6),
            cite_self_required=True,
        )

    async def execute(self, **kwargs) -> ToolResult:
        ok, err, runtime = standing_gate(self._app_state)
        if not ok:
            return err
        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False, error="user_id missing",
                metadata={"ok": False, "reason": "missing_user"},
            )

        title = str(kwargs.get("title") or "").strip()
        kind = str(kwargs.get("kind") or "url").strip().lower()
        # Legacy callers pass url=; new callers pass target=.
        target = str(
            kwargs.get("target") or kwargs.get("url") or "",
        ).strip()
        if not title or not target:
            return ToolResult(
                success=False, error="title and target required",
                metadata={"ok": False, "reason": "missing_args"},
            )
        if kind not in _KIND_MAP:
            return ToolResult(
                success=False,
                error=f"unknown watch kind: {kind}",
                metadata={"ok": False, "reason": "bad_kind"},
            )
        task_kind, target_key, default_hours = _KIND_MAP[kind]
        if kind == "url" and not target.startswith(("http://", "https://")):
            return ToolResult(
                success=False, error="url must be http(s)",
                metadata={"ok": False, "reason": "bad_url"},
            )

        # Creator/feed watches resolve the user's words into a concrete
        # feed BEFORE the row exists — a watch that can't find its feed
        # is refused while the user is still here to correct it.
        source_label = ""
        if kind == "feed":
            from augmentum.companion_runtime.curator import _resolve_http_client
            from augmentum.companion_runtime.feed_resolve import (
                resolve_feed_source,
            )
            http_client = _resolve_http_client(runtime)
            if http_client is None:
                return ToolResult(
                    success=False, error="no http client available",
                    metadata={"ok": False, "reason": "no_http_client"},
                )
            resolved = await resolve_feed_source(http_client, target)
            if not resolved.ok:
                return ToolResult(
                    success=False,
                    error=f"couldn't resolve that source: {resolved.error}",
                    metadata={"ok": False, "reason": "feed_unresolved"},
                )
            target = resolved.feed_url
            source_label = resolved.label

        from augmentum.companion_runtime import standing_tasks
        if task_kind not in standing_tasks.known_kinds():
            return ToolResult(
                success=False,
                error=f"{kind} watches aren't available in this install",
                metadata={"ok": False, "reason": "kind_unavailable"},
            )

        try:
            interval_hours = float(
                kwargs.get("interval_hours") or default_hours,
            )
        except (TypeError, ValueError):
            interval_hours = default_hours
        interval_hours = max(0.5, interval_hours)

        params: dict[str, Any] = {target_key: target}
        if source_label:
            params["source_label"] = source_label
        intent = str(kwargs.get("intent") or "").strip()
        if intent:
            params["intent"] = intent[:300]
        condition = _validate_condition(kwargs.get("condition"))
        if condition:
            params["condition"] = condition
        delivery, delivery_err = parse_delivery_param(kwargs.get("delivery"))
        if delivery_err:
            return ToolResult(
                success=False, error=delivery_err, validation_error=True,
            )
        if delivery:
            params["delivery"] = delivery

        # Read-before-create: deterministic duplicate review against
        # the user's existing schedule (same discipline as
        # schedule_briefing — same-target watches are the same watch).
        dup = await duplicate_review(
            runtime, user_id=user_id, kind=task_kind, title=title,
            params=params,
            confirm_replace=bool(kwargs.get("confirm_replace")),
        )
        if dup is not None:
            return dup

        try:
            task = await standing_tasks.add_task(
                runtime.backend.conn,
                user_id=user_id,
                companion_id=runtime.companion_id,
                title=title,
                kind=task_kind,
                params=params,
                interval_seconds=int(interval_hours * 3600),
                user_timezone=await standing_tasks._resolve_user_timezone(
                    self._app_state, user_id,
                ),
            )
        except ValueError as e:
            return ToolResult(
                success=False, error=str(e),
                metadata={"ok": False, "reason": "validation"},
            )

        if task is None:
            return ToolResult(
                success=False, error="watch creation failed",
                metadata={"ok": False, "reason": "persist_failed"},
            )

        # Creation probe: run once now (no notification — the result
        # comes back through this reply) so the user hears what the
        # watch actually sees. Catches wrong URLs and unbindable asks
        # while they're still here to correct them.
        probe_line = await self._probe(runtime, task=task, user_id=user_id)
        if probe_line is None and kind == "url":
            # URL fetch failed outright — a watch that can't see its
            # page is a dead watch. Refuse rather than silently watch
            # nothing; the engine row is removed.
            await standing_tasks.remove_task(
                runtime.backend.conn, task_id=task.id,
                user_id=user_id, companion_id=runtime.companion_id,
            )
            return ToolResult(
                success=False,
                error="couldn't fetch that URL just now — watch not "
                      "created. Double-check the address?",
                metadata={"ok": False, "reason": "probe_fetch_failed"},
            )

        cadence = (
            f"every {interval_hours:.0f}h" if interval_hours < 24
            else f"every {interval_hours / 24:.0f}d"
        )
        out = f"Watching '{title}' ({kind}) {cadence}."
        if intent:
            out += f" Only flagging: {intent}."
        if probe_line:
            out += f" Right now: {probe_line}"
        return ToolResult(
            success=True,
            output=out,
            metadata={
                "ok": True,
                "task_id": task.id,
                "kind": task_kind,
                "next_run_at": task.next_run_at,
                "probe": probe_line or "",
            },
        )

    async def _probe(self, runtime: Any, *, task: Any, user_id: str) -> str | None:
        """One synchronous fire at creation, never notifying. Returns the
        observed baseline line, '' when the probe couldn't run (service
        down — watch stays, honestly labeled), or None when the fetch
        itself failed (caller decides whether that refuses creation)."""
        import asyncio

        from augmentum.companion_runtime import standing_tasks
        from augmentum.config import settings

        timeout_s = float(
            getattr(settings, "companion_watch_probe_timeout_s", 10.0) or 10.0,
        )
        try:
            result = await asyncio.wait_for(
                standing_tasks.run_now(
                    runtime, task_id=task.id, user_id=user_id, surface=False,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            log.warning("watch_probe_timeout", task_id=task.id)
            return ""
        except Exception:
            log.warning("watch_probe_failed", task_id=task.id, exc_info=True)
            return ""
        if not result:
            return ""
        summary = str(result.get("summary") or "")
        if summary.startswith("error:"):
            return None
        return summary


# kind → (standing-task kind, params key for the target, default cadence h)
_KIND_MAP: dict[str, tuple[str, str, float]] = {
    "url": ("url_watch", "url", 1.0),
    "search": ("recurring_search", "query", 24.0),
    "repo": ("github_releases", "repo", 24.0),
    "topic": ("feed_digest", "topic", 24.0),
    "metric": ("metric_watch", "metric", 1.0),
    # target here is whatever the user said — a YouTube channel/@handle,
    # r/subreddit, blog URL, or a feed URL. Resolved + validated at
    # creation by feed_resolve; the row stores the concrete feed_url.
    "feed": ("feed_watch", "feed_url", 4.0),
}


def _validate_condition(raw: Any) -> dict[str, Any] | None:
    """Light shape-check at creation; evaluation is fire-time code."""
    if not isinstance(raw, dict):
        return None
    op = str(raw.get("op") or "").strip()
    if op not in {"<", ">", "<=", ">=", "=="}:
        return None
    try:
        value = float(raw.get("value"))
    except (TypeError, ValueError):
        return None
    out: dict[str, Any] = {"op": op, "value": value}
    unit = str(raw.get("unit") or "").strip()
    if unit:
        out["unit"] = unit[:16]
    return out
