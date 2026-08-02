"""Narrative lorebook grounding tools — ``lorebook.check`` / ``lorebook.create``.

These are the F1/F5 tools from ``docs/companion-model-training-design.md``,
exposed as first-class :class:`Tool` objects so they EXIST in the tool
registry (the catalog the training/inference plumbing references) and so a
direct ``execute`` works for any surface that drives the registry.

The LIVE narrative-mode path, however, does not run tools through the
registry — narrative mode dispatches OpenAI tool schemas through the recall
loop (``modes/narrative/recall_loop.py``) against a string dispatcher
(``modes/narrative/lorebook_native_schemas.py``). Both these Tool objects
and that dispatcher share ONE implementation: ``execute`` resolves the live
per-session ``NarrativeEngine`` from ``app_state.narrative_engines`` and
calls ``dispatch_lorebook_native_tool``. So registry-driven calls and
narrative-loop calls are guaranteed to behave identically.

Surfaces: narrative-only. We set ``chat=False, coder=False`` (and leave
companion/voice off) so these never leak onto the general chat/coder tool
trees — they only make sense inside a narrative session that has a
``LoreEngine``. There is no ``narrative`` SurfaceExposure flag (narrative
exposes tools through its own loop, not the surface registry), so the
SurfaceExposure here is purely a "don't widen reach" declaration.
"""

from __future__ import annotations

from typing import Any

from augmentum.modes.narrative.lorebook_native_schemas import (
    LOREBOOK_CATEGORIES,
    dispatch_lorebook_native_tool,
)
from augmentum.tools.base import (
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _resolve_engine(app_state: Any, user_id: str, session_id: str):
    """Resolve the live NarrativeEngine for (user_id, session_id), or None.

    Mirrors the cache key built in ``proxy/handler_factory._get_narrative_engine``
    — ``(user_id, session_id)`` when a user is present, else bare session_id.
    Returns None when no engine is cached (no active narrative session) so
    callers can return a clean error instead of raising.
    """
    if app_state is None:
        return None
    engines = getattr(app_state, "narrative_engines", None)
    if not engines:
        return None
    key: str | tuple[str, str] = (user_id, session_id) if user_id else session_id
    return engines.get(key)


class _LorebookToolBase(Tool):
    """Shared resolution + dispatch for the two narrative lorebook verbs."""

    def __init__(self, app_state: Any = None) -> None:
        self._app_state = app_state

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def surfaces(self) -> SurfaceExposure:
        # Narrative-only. Keep it off chat/coder/voice/companion — these
        # verbs only function inside a narrative session with a LoreEngine.
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        # Lore changes mid-session (create writes, check reflects writes),
        # so never serve a stale cached result.
        return False

    def _session_id(self, kwargs: dict) -> str:
        ctx = kwargs.get("_context") or {}
        if isinstance(ctx, dict):
            return str(ctx.get("session_id") or "")
        return ""

    async def _dispatch(self, tool_name: str, kwargs: dict) -> ToolResult:
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="No user context — narrative lorebook tools are user-scoped.",
            )
        session_id = self._session_id(kwargs)
        if not session_id:
            return ToolResult(
                success=False,
                error="No session context — lorebook tools need a narrative session.",
            )

        engine = _resolve_engine(self._app_state, user_id, session_id)
        if engine is None:
            return ToolResult(
                success=False,
                error=(
                    "No active narrative session for this session_id — open "
                    "the chat in narrative mode first."
                ),
            )

        branch_id = getattr(engine.state, "branch_id", "main") or "main"
        try:
            result_text, mutations = dispatch_lorebook_native_tool(
                engine._lore_engine,
                session_id,
                user_id=user_id,
                branch_id=branch_id,
                tool_name=tool_name,
                raw_arguments=dict(kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("lorebook_tool_dispatch_failed", tool=tool_name, error=str(exc))
            return ToolResult(success=False, error=f"{tool_name} failed: {exc}")

        return ToolResult(
            success=True,
            output=result_text,
            metadata={"mutations": mutations or []},
        )


class LorebookCheckTool(_LorebookToolBase):
    """``lorebook.check`` — grounded mid-scene lore retrieval (read-only)."""

    @property
    def name(self) -> str:
        return "lorebook.check"

    @property
    def description(self) -> str:
        return (
            "Query the session's world knowledge base for established lore "
            "relevant to what you're about to describe (a location, "
            "character, item, or event). Use it to ground descriptions "
            "instead of confabulating. An empty result means nothing is "
            "established yet — you're free to invent, and should record "
            "anything significant with lorebook.create."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What you need to know — a location, character "
                        "name, event, item, or concept."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Brief context for why you're checking (helps "
                        "retrieval relevance). Optional."
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return await self._dispatch("lorebook.check", kwargs)


class LorebookCreateTool(_LorebookToolBase):
    """``lorebook.create`` — record newly-established detail as session lore."""

    @property
    def name(self) -> str:
        return "lorebook.create"

    @property
    def description(self) -> str:
        return (
            "Record a newly established detail as session world lore — a "
            "new character, location feature, rule, faction, or event that "
            "should stay consistent for the rest of this session. Records "
            "to THIS session only (source=narrative_established); never "
            "modifies the source character card. Check first with "
            "lorebook.check to avoid duplicating established lore."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Keywords that should trigger this entry in future "
                        "turns (names, places, concepts)."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The established detail to record.",
                },
                "category": {
                    "type": "string",
                    "enum": list(LOREBOOK_CATEGORIES),
                    "description": "What kind of world detail this is.",
                },
            },
            "required": ["keywords", "content"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return await self._dispatch("lorebook.create", kwargs)
