"""``propose_offer`` tool — model-facing entrypoint to the offer substrate.

The chat LLM calls this when it notices the user would benefit from an
install / setting change / memory save / mode switch. The tool result
tells the model whether the offer surfaced or was suppressed/rate-limited
so it can adjust prose ("I'd usually offer to set this up, but you've
dismissed it before — here's the manual route").

The actual surfacing happens in
``augmentum/offers/dispatcher.py::propose_offer`` — this file is just
the Tool subclass + JSON schema. Keeping the two split means the
dispatcher is also reachable from non-tool callers (companion-init
phase 5, scripted tests, etc.).

See ``docs/superpowers/specs/2026-06-02-offer-substrate-design.md``.
"""

from __future__ import annotations

from typing import Any

from augmentum.config import settings
from augmentum.offers.catalog.base import list_kinds, list_targets
from augmentum.offers.dispatcher import propose_offer as _dispatch
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.tools.base import (
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _format_catalog_for_description() -> str:
    """Render the live catalog as a one-line summary the LLM can read.

    Tools' descriptions are baked into the system prompt; we don't want
    a stale list. The string lists each registered kind plus its
    target_ids so the model knows what's offerable without guessing.
    """

    lines: list[str] = []
    for kind in list_kinds():
        targets = list_targets(kind)
        if not targets:
            continue
        targets_str = ", ".join(targets)
        lines.append(f"{kind}: {targets_str}")
    if not lines:
        return "(no offers registered yet)"
    return "; ".join(lines)


class ProposeOfferTool(Tool):
    """Propose an install / change / save the user can Accept inline.

    The chip surfaces in the chat stream with [Install] / [Not now] /
    [Never] buttons; the user's click triggers the catalog entry's
    accept handler. Nothing changes until the user explicitly accepts.
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "propose_offer"

    @property
    def description(self) -> str:
        return (
            "Propose an action the user can Accept inline — install an "
            "MCP server, save a memory, switch mode, etc. Renders as a "
            "chip in the chat with Install / Not now / Never buttons. "
            "Nothing happens until the user explicitly accepts. Use "
            "when the user's message suggests they'd benefit from "
            "something installable or changeable that isn't already "
            "in place. Do NOT use to perform an action — only to "
            "propose it. The user already dismissed an offer? You'll "
            "get ok=False with suppressed=True; mention the manual "
            "route instead.\n\n"
            f"Available kinds/targets: {_format_catalog_for_description()}"
        )

    @property
    def category(self) -> ToolCategory:
        # EXECUTE so it's available in apply/respond/gather phases —
        # matches the phase visibility of memory_recall and other
        # state-touching tools.
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        # Chat + coder + companion. Not voice — offers don't make
        # sense when the user can't see the chip.
        return SurfaceExposure(chat=True, coder=True, companion=True, flow=False)

    @property
    def cacheable(self) -> bool:
        # Always live — same inputs may differ in outcome because of
        # rate limits / suppression state.
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": (
                        "Offer category. One of the registered kinds "
                        "(e.g. 'mcp_server'). See the description for "
                        "the live catalog."
                    ),
                },
                "target_id": {
                    "type": "string",
                    "description": (
                        "Specific target within the kind (e.g. 'gmail' "
                        "for kind='mcp_server'). Must be a registered "
                        "target_id — the tool will fail with "
                        "reason='unknown_target' otherwise."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One-sentence 'why' shown in the chip body. "
                        "Reference what the user just said (e.g. 'you "
                        "asked about your email')."
                    ),
                },
                "extra": {
                    "type": "object",
                    "description": (
                        "Optional per-kind extras. For mcp_server, may "
                        "include 'url' and 'headers' if the user "
                        "supplied them."
                    ),
                },
            },
            "required": ["kind", "target_id", "reason"],
        }

    def _resolve_conn(self):
        sm = getattr(self._app_state, "state_manager", None)
        if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend):
            return sm.backend.conn
        return None

    async def execute(self, **kwargs) -> ToolResult:
        if not bool(getattr(settings, "offers_enabled", True)):
            return ToolResult(
                success=False,
                error="offers_disabled",
                metadata={"ok": False, "reason": "offers_disabled"},
            )

        kind = str(kwargs.get("kind") or "").strip()
        target_id = str(kwargs.get("target_id") or "").strip()
        reason = str(kwargs.get("reason") or "").strip()
        extra = kwargs.get("extra")
        if not isinstance(extra, dict):
            extra = None

        if not kind or not target_id:
            return ToolResult(
                success=False,
                error="kind and target_id are required",
                validation_error=True,
            )

        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="user_id missing — offers are per-user",
                metadata={"ok": False, "reason": "missing_user"},
            )

        # Session / turn / mode context. The handler stamps these onto
        # the tool context when the chat LLM dispatches the call. They're
        # optional — missing thread_id / turn_id just disables the
        # corresponding rate-limit slices; missing mode falls open on
        # the catalog's mode gate (any registered target succeeds).
        ctx = kwargs.get("_context") or {}
        thread_id = str(ctx.get("session_id") or ctx.get("thread_id") or "")
        turn_id = str(ctx.get("turn_id") or "")
        mode = str(ctx.get("mode") or "")
        workspace_id = str(ctx.get("workspace_id") or "")

        # Substrate-stashed extras travel under leading-underscore keys
        # so they don't collide with model-supplied ``extra`` payload
        # fields (which the prompt explicitly lists as "URL, headers,
        # etc."). The accept handler reads back from the same key.
        if workspace_id:
            if extra is None:
                extra = {}
            extra.setdefault("_workspace_id", workspace_id)

        conn = self._resolve_conn()
        if conn is None:
            return ToolResult(
                success=False,
                error="offers require a SQLite backend",
                metadata={"ok": False, "reason": "no_backend"},
            )

        hub = getattr(self._app_state, "notification_hub", None)

        result = await _dispatch(
            conn,
            hub=hub,
            user_id=user_id,
            kind=kind,
            target_id=target_id,
            reason=reason,
            extra=extra,
            thread_id=thread_id,
            turn_id=turn_id,
            mode=mode,
            max_per_turn=int(getattr(settings, "offers_max_per_turn", 2)),
            max_pending_per_session=int(
                getattr(settings, "offers_max_pending_per_session", 5),
            ),
            max_per_day=int(getattr(settings, "offers_max_per_day", 20)),
            expiry_days=int(getattr(settings, "offers_default_expiry_days", 7)),
        )

        # The model reads ``metadata`` to decide how to phrase its
        # follow-up message; ``output`` is the human-readable summary.
        out_parts: list[str] = []
        if result.ok:
            out_parts.append(f"Offer surfaced: {kind}/{target_id}.")
        elif result.suppressed:
            out_parts.append(
                f"User has previously dismissed offers for {kind}/{target_id}. "
                "Skip the offer; reference the manual route instead.",
            )
        else:
            out_parts.append(
                f"Offer not surfaced ({result.reason}). "
                "Do not mention this to the user; respond as if the "
                "offer tool wasn't called.",
            )

        return ToolResult(
            success=True,
            output=" ".join(out_parts),
            metadata=result.to_dict(),
        )
