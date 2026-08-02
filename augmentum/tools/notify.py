"""``notify`` — companion verbs architecture, Phase 4 linchpin tool.

The notify tool surfaces a short user-visible notification via the
existing :mod:`augmentum.notifications.hub` substrate. Per the verbs
architecture spec, this is the *linchpin* core verb: management
verbs invoke it for runtime→user surfacings without an LLM hop,
and the chat model can also call it directly when it wants to leave
a note rather than reply.

Two invocation paths:

* **Model-invoked** — Tool registry dispatch. The LLM calls
  ``notify`` like any other tool; the result is a one-line confirmation
  the model can chain on.
* **Management-verb-invoked** — :mod:`narrate_state_to_user` and
  related Phase 3c verbs call :func:`notify_user` directly,
  bypassing the LLM. The Tool wrapper delegates to the same helper.

The split lets the same code path serve both halves of the verb
taxonomy. ``core_verb`` metadata declares the safety/autonomy/cost
envelope so the dispatcher can gate management-verb invocations
against the presence-mode autonomy floor.
"""

from __future__ import annotations

from typing import Any

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


async def notify_user(
    app_state: Any, *,
    user_id: str,
    title: str,
    body: str = "",
    channel_id: str = "companion.notify",
    source: str = "companion.notify",
    importance: int | None = None,
    dedupe_key: str = "",
    transient: bool = False,
) -> str | None:
    """Publish-and-dispatch helper. Returns the notification_id on
    success, None on failure.

    Used by both the Tool's ``execute`` path and by management verbs
    (narrate_state_to_user) so neither side has to re-thread the
    hub-resolution dance.
    """
    if not user_id or not title:
        log.debug("notify_skipped_empty_args", user_id=user_id, title=title)
        return None
    hub = getattr(app_state, "notification_hub", None)
    if hub is None:
        try:
            from augmentum.notifications.hub import NotificationHub
            hub = NotificationHub()
            app_state.notification_hub = hub
        except Exception:
            log.warning("notify_hub_create_failed", exc_info=True)
            return None

    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None:
        log.debug("notify_no_runtime")
        return None

    try:
        from augmentum.notifications.hub import publish_and_dispatch
        notification_id = await publish_and_dispatch(
            runtime.backend.conn,
            hub=hub,
            user_id=user_id,
            channel_id=channel_id,
            source=source,
            title=title,
            body=body,
            importance=importance,
            dedupe_key=dedupe_key,
            transient=transient,
        )
        return notification_id
    except Exception:
        log.warning("notify_publish_failed", exc_info=True)
        return None


class NotifyTool(Tool):
    """Leave a short user-visible note in the notifications feed.

    Companion verbs architecture — core verb (linchpin). Both
    management verbs and the chat model can invoke this; the runtime
    routes the result through the same hub.
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "notify"

    @property
    def description(self) -> str:
        return (
            "Leave the user a brief notification — a short title + "
            "optional body. Use when you want to mention something "
            "small without replying in the chat thread (e.g. "
            "background work finished, a reminder fired). NOT for "
            "long-form responses — that's what the chat reply is "
            "for. The user can see notifications in their feed even "
            "when the chat is closed."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=True, coder=False, companion=True, flow=False)

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
                    "description": "Short notification title (1-60 chars).",
                },
                "body": {
                    "type": "string",
                    "description": "Optional longer body (0-300 chars).",
                },
                "transient": {
                    "type": "boolean",
                    "description": (
                        "If true, the notification auto-fades from the "
                        "feed after display. Use for ambient mood "
                        "narrations; default false for substantive "
                        "notes."
                    ),
                },
            },
            "required": ["title"],
        }

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.WRITE_USER,
            autonomy_class=CoreVerbAutonomyClass.BACKGROUND,
            cost_envelope=CostEnvelope(max_wallclock_ms=2_000, max_db_ops=4),
            cite_self_required=True,
            counts_in_chain_depth=False,  # Terminal surface verb.
        )

    async def execute(self, **kwargs) -> ToolResult:
        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False, error="user_id missing",
                metadata={"ok": False, "reason": "missing_user"},
            )

        title = str(kwargs.get("title") or "").strip()
        body = str(kwargs.get("body") or "").strip()
        transient = bool(kwargs.get("transient") or False)
        if not title:
            return ToolResult(
                success=False, error="title required",
                metadata={"ok": False, "reason": "missing_title"},
            )

        # Cap the title and body lengths defensively — the LLM
        # occasionally drops a whole paragraph in here when it
        # confuses notify with reply.
        if len(title) > 80:
            title = title[:77] + "..."
        if len(body) > 400:
            body = body[:397] + "..."

        notification_id = await notify_user(
            self._app_state,
            user_id=user_id,
            title=title,
            body=body,
            transient=transient,
        )
        if not notification_id:
            return ToolResult(
                success=False, error="notification publish failed",
                metadata={"ok": False, "reason": "publish_failed"},
            )

        return ToolResult(
            success=True,
            output=f"Notification posted: {title}",
            metadata={"ok": True, "notification_id": notification_id},
        )
