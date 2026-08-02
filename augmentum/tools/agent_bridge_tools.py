"""ATP agent-bridge tools — presence + ask/reply for external agents.

Thin adapters over ``augmentum/proxy/agent_bridge.py``. Harness/project
identity comes from the ATP route's ``_context`` (header-derived, never
from tool arguments). ATP-only surfaces.
"""

from __future__ import annotations

from augmentum.proxy import agent_bridge as bridge
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class _BridgeToolBase(Tool):
    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    def health_check(self) -> bool:
        return bridge._conn(self._app_state) is not None

    def _identity(self, kwargs: dict) -> tuple[str, str, str]:
        user_id = self.extract_user_id(kwargs)
        ctx = kwargs.get("_context") or {}
        harness = str(ctx.get("harness") or "") if isinstance(ctx, dict) else ""
        project = str(ctx.get("project") or "") if isinstance(ctx, dict) else ""
        return user_id, harness, project


class AgentCheckinTool(_BridgeToolBase):
    @property
    def name(self) -> str:
        return "agent_checkin"

    @property
    def description(self) -> str:
        return (
            "Register/update this agent session so the user can see active "
            "agents from any device. Call at task start (returns your "
            "agent_id — reuse it), on major progress, and when finishing "
            "(status='done' with a summary). The response also delivers "
            "any replies the user has sent you since your last check-in."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "From a previous check-in; omit on first call"},
                "title": {"type": "string", "description": "One line: what you are working on"},
                "status": {"type": "string", "enum": list(bridge.AGENT_STATUSES), "default": "working"},
                "summary": {"type": "string", "description": "Short progress note"},
            },
        }

    async def execute(self, **kwargs) -> ToolResult:
        user_id, harness, project = self._identity(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        res = await bridge.checkin(
            self._app_state, user_id=user_id, harness=harness, project=project,
            agent_id=str(kwargs.get("agent_id") or ""),
            title=str(kwargs.get("title") or ""),
            status=str(kwargs.get("status") or "working"),
            summary=str(kwargs.get("summary") or ""),
        )
        if res is None:
            return ToolResult(success=False, error="agent bridge unavailable")
        answered = res.get("answered_requests") or []
        assignments = res.get("assignments") or []
        lines = [f"Checked in as {res['agent_id']} (status {res['status']})."]
        if assignments:
            lines.append("NEW TASK(S) assigned to you by the user — work on these:")
            for a in assignments:
                lines.append(f"- {a.get('task') or a.get('title') or ''}")
        if answered:
            lines.append("Replies from the user since your last check-in:")
            for a in answered:
                what = a.get("reply_action") or ""
                txt = a.get("reply_text") or ""
                lines.append(f"- [{a['request_id']}] {a['title']}: "
                             + " ".join(x for x in (what, txt) if x))
        return ToolResult(success=True, output="\n".join(lines), metadata=res)


class AskUserTool(_BridgeToolBase):
    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Reach the user through an Augmentum notification on whatever "
            "device they're on — they do NOT need to be at this terminal. "
            "kind='approve' shows Approve/Deny buttons (permission gates); "
            "'question' and 'review' invite a free-text reply (use "
            "'review' when work is done and you're offering results + "
            "what to do next); 'notify' is fire-and-forget status. For "
            "approve/question/review, poll check_reply with the returned "
            "request_id (every 10-30s) or pick replies up at your next "
            "agent_checkin."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(bridge.REQUEST_KINDS)},
                "title": {"type": "string", "description": "Short headline (shows on the notification)"},
                "body": {"type": "string", "description": "Detail: what you need approved / are asking / the review summary"},
                "agent_id": {"type": "string", "description": "From agent_checkin, so the reply routes back to you"},
            },
            "required": ["kind", "title"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        user_id, harness, project = self._identity(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        kind = str(kwargs.get("kind") or "question")
        res = await bridge.create_request(
            self._app_state, user_id=user_id,
            agent_session_id=str(kwargs.get("agent_id") or ""),
            kind=kind,
            title=str(kwargs.get("title") or "").strip(),
            body=str(kwargs.get("body") or ""),
            harness=harness, project=project,
        )
        if res is None:
            return ToolResult(success=False, error="agent bridge unavailable or empty title")
        if kind == "notify":
            return ToolResult(success=True, output="Notification sent.", metadata=res)
        return ToolResult(
            success=True,
            output=(
                f"Sent. Poll check_reply with request_id {res['request_id']} "
                "(the user may take a while — keep working if you can, or "
                "poll every 10-30s if blocked)."
            ),
            metadata=res,
        )


class CheckReplyTool(_BridgeToolBase):
    @property
    def name(self) -> str:
        return "check_reply"

    @property
    def description(self) -> str:
        return (
            "Poll an ask_user request for the user's answer. Returns "
            "pending, or the Approve/Deny action and/or free-text reply."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"request_id": {"type": "string"}},
            "required": ["request_id"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        user_id, _, _ = self._identity(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        req = await bridge.get_request(
            self._app_state, user_id=user_id,
            request_id=str(kwargs.get("request_id") or "").strip(),
        )
        if req is None:
            return ToolResult(success=False, error="request not found (or not yours)")
        if req["status"] == "pending":
            return ToolResult(success=True, output="Still pending — no answer yet.",
                              metadata=req)
        parts = []
        if req["reply_action"]:
            parts.append(f"action: {req['reply_action']}")
        if req["reply_text"]:
            parts.append(f"reply: {req['reply_text']}")
        return ToolResult(
            success=True,
            output=f"Answered ({req['status']}). " + "; ".join(parts),
            metadata=req,
        )


ATP_BRIDGE_TOOL_CLASSES = (AgentCheckinTool, AskUserTool, CheckReplyTool)
