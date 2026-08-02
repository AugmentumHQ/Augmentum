"""ATP ``memory_store`` — explicit, human-gated memory writes from any harness.

The counterpart to ``memory_recall`` for external coding agents: "remember
this convention" works identically from Claude Code, pi, cursor, etc.

It does NOT write live memory. Every call STAGES a candidate through the
same harness-harvest pipeline as background capture
(``proxy/harness.py`` / ``training/capture.py``); the baseline only grows
when the user promotes the candidate in the review UI
(``/api/harness/harvest/view``). Scope identity ({harness}:{project}) is
derived server-side from the request headers by the ATP route — never
from tool arguments.
"""

from __future__ import annotations

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_KINDS = ("convention", "fact", "preference")


class HarnessMemoryStoreTool(Tool):
    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "memory_store"

    @property
    def description(self) -> str:
        return (
            "Stage a durable memory (convention, fact, or preference) for "
            "this project's memory scope. The memory is NOT active "
            "immediately: it enters a human review queue and only becomes "
            "part of the injected baseline once the user promotes it. Use "
            "when the user says something worth remembering across "
            "sessions (a rule, a decision, a correction, a project fact)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The memory, phrased as a standalone durable "
                        "statement (commands/paths verbatim)"
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": list(_KINDS),
                    "default": "fact",
                },
            },
            "required": ["text"],
        }

    def health_check(self) -> bool:
        from augmentum.config import settings

        return bool(getattr(settings, "harness_capture_enabled", True))

    async def execute(self, **kwargs) -> ToolResult:
        from augmentum.proxy.harness import _SECRET, harness_memory_scope
        from augmentum.training.capture import capture_harness_observation

        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        text = " ".join(str(kwargs.get("text") or "").split())
        if len(text) < 4:
            return ToolResult(success=False, validation_error=True,
                              error="'text' is required (min 4 chars)")
        if _SECRET.search(text):
            return ToolResult(
                success=False, validation_error=True,
                error="refusing to store: the text looks like it contains a credential/secret",
            )
        kind = str(kwargs.get("kind") or "fact").strip().lower()
        if kind not in _KINDS:
            kind = "fact"
        ctx = kwargs.get("_context") or {}
        harness = str(ctx.get("harness") or "") if isinstance(ctx, dict) else ""
        project = str(ctx.get("project") or "") if isinstance(ctx, dict) else ""
        target_scope = harness_memory_scope(harness, project)
        obs_id = capture_harness_observation(
            user_id=user_id, harness=harness or "atp", model="",
            source_message=f"[memory_store tool] {text}"[:2000],
            candidates=[{
                "kind": kind,
                "text": text,
                "durable": True,
                "supersedes_baseline_id": None,
                "supersedes_baseline_text": None,
                "supersedes_baseline_scope": None,
                "target_scope": target_scope,
            }],
        )
        if not obs_id:
            return ToolResult(
                success=False,
                error="staging is disabled (harness_capture_enabled is off) or the write failed",
            )
        log.info("atp_memory_store_staged", user_id=user_id,
                 scope=target_scope, obs_id=obs_id)
        return ToolResult(
            success=True,
            output=(
                f"Staged for review (scope {target_scope}, id {obs_id}). "
                "It becomes part of the injected memory once promoted in "
                "the review queue at /api/harness/harvest/view."
            ),
            metadata={"obs_id": obs_id, "scope": target_scope, "kind": kind},
        )
