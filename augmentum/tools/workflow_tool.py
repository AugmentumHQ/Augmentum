"""ATP workflow tool — self-minted, self-improving soft procedural memory.

The model mints a workflow when something worked ("save what worked as a
reusable playbook"), refines it over time (each save bumps the version), and
prunes freely. A workflow is natural-language guidance — a ``when_to_use``
trigger + numbered steps — NOT an executable macro (that's ``atp_recipe``).
The matching workflow is auto-surfaced into the harness briefing by FTS on
its trigger (see harness.py), so recall costs no per-turn tool call; this
tool is the mint/edit/prune/inspect surface.

Governance (Matt's call): auto-mint into the model's OWN per-user +
harness:project scope, with list/get/delete for easy pruning. No staging
gate — procedural how-to is lower-risk than facts-about-the-user, and the
model owning its own skill library is the point.
"""

from __future__ import annotations

from augmentum.proxy.harness import harness_memory_scope
from augmentum.tools import workflow_store
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SHARED_SCOPE = "harness:default"


class WorkflowTool(Tool):
    """Save, search, list, inspect, prune, and rate soft workflows."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "workflow"

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
    def description(self) -> str:
        return (
            "Your self-improving playbook library — save a procedure that "
            "worked so you don't re-derive it next time, and refine it over "
            "runs. A workflow is natural-language GUIDANCE (when_to_use + "
            "numbered steps), not an executable macro (that's atp_recipe). "
            "The right workflow is auto-surfaced into your context by its "
            "when_to_use trigger, so you rarely search by hand. "
            "action='save' {name, when_to_use, steps, description?} — mint or "
            "refine (each save bumps the version). "
            "action='search' {query} — find workflows whose trigger matches. "
            "action='list' / action='get' {name} / action='delete' {name} — "
            "manage/prune. action='record_outcome' {name, success} — log "
            "whether it worked, so weak workflows can be spotted and rewritten."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["save", "search", "list", "get", "delete",
                                    "record_outcome"]},
                "name": {"type": "string"},
                "when_to_use": {"type": "string",
                                "description": "for save: the semantic trigger — "
                                "'when the user wants X' (this is what retrieval matches)"},
                "steps": {"type": "string",
                          "description": "for save: numbered steps as free text/markdown"},
                "description": {"type": "string"},
                "query": {"type": "string", "description": "for search"},
                "success": {"type": "boolean", "description": "for record_outcome"},
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------

    def _scopes(self, kwargs: dict) -> tuple[str, list[str]]:
        """Return (own_scope, search_scopes). Mint targets own scope; search
        spans own + the shared default pool."""
        ctx = kwargs.get("_context") if isinstance(kwargs.get("_context"), dict) else {}
        own = harness_memory_scope(str(ctx.get("harness") or ""), str(ctx.get("project") or ""))
        scopes = [own] if own == _SHARED_SCOPE else [own, _SHARED_SCOPE]
        return own, scopes

    async def execute(self, **kwargs) -> ToolResult:
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        action = str(kwargs.get("action") or "").strip().lower()
        name = str(kwargs.get("name") or "").strip()
        own_scope, search_scopes = self._scopes(kwargs)
        ctx = kwargs.get("_context") if isinstance(kwargs.get("_context"), dict) else {}

        if action == "save":
            when = str(kwargs.get("when_to_use") or "").strip()
            steps = str(kwargs.get("steps") or "").strip()
            if not name or not when or not steps:
                return ToolResult(success=False, validation_error=True,
                                  error="save needs name, when_to_use, and steps")
            saved = await workflow_store.save_workflow(
                self._app_state, user_id=user_id, scope=own_scope, name=name,
                when_to_use=when, steps=steps,
                description=str(kwargs.get("description") or ""),
                harness=str(ctx.get("harness") or ""),
            )
            if saved is None:
                return ToolResult(success=False, error="could not save workflow")
            verb = "Updated" if saved["version"] > 1 else "Saved"
            return ToolResult(
                success=True,
                output=f"{verb} workflow {saved['name']!r} (v{saved['version']}). "
                       f"It'll auto-surface when a task matches: {saved['when_to_use']!r}",
                metadata=saved,
            )

        if action == "search":
            rows = await workflow_store.search_workflows(
                self._app_state, user_id=user_id, scopes=search_scopes,
                query=str(kwargs.get("query") or ""), limit=3,
            )
            if not rows:
                return ToolResult(success=True, output="No matching workflows.",
                                  metadata={"workflows": []})
            out = "\n\n".join(
                f"### {r['name']} (v{r['version']}, used {r['times_used']}× / "
                f"{r['times_succeeded']}✓)\nwhen: {r['when_to_use']}\n{r['steps']}"
                for r in rows
            )
            return ToolResult(success=True, output=out, metadata={"workflows": rows})

        if action == "list":
            rows = await workflow_store.list_workflows(
                self._app_state, user_id=user_id, scopes=search_scopes)
            if not rows:
                return ToolResult(success=True, output="No saved workflows yet.",
                                  metadata={"workflows": []})
            lines = [f"- {r['name']} (v{r['version']}, {r['times_succeeded']}/{r['times_used']}✓) "
                     f"— {r['when_to_use']}" for r in rows]
            return ToolResult(success=True, output="Your workflows:\n" + "\n".join(lines),
                              metadata={"workflows": rows})

        if action == "get":
            rec = await workflow_store.get_workflow(
                self._app_state, user_id=user_id, scope=own_scope, name=name)
            if rec is None:
                # fall back to the shared pool
                rec = await workflow_store.get_workflow(
                    self._app_state, user_id=user_id, scope=_SHARED_SCOPE, name=name)
            if rec is None:
                return ToolResult(success=False, error=f"no workflow named {name!r}")
            return ToolResult(
                success=True,
                output=f"# {rec['name']} (v{rec['version']})\nwhen: {rec['when_to_use']}\n"
                       f"{rec['description']}\n\n{rec['steps']}",
                metadata=rec,
            )

        if action == "delete":
            ok = await workflow_store.delete_workflow(
                self._app_state, user_id=user_id, scope=own_scope, name=name)
            return ToolResult(success=ok,
                              output=f"Deleted workflow {name!r}." if ok else "",
                              error="" if ok else f"could not delete {name!r}")

        if action == "record_outcome":
            ok = await workflow_store.record_outcome(
                self._app_state, user_id=user_id, scope=own_scope, name=name,
                success=bool(kwargs.get("success")))
            return ToolResult(success=ok,
                              output="Recorded." if ok else "",
                              error="" if ok else f"no workflow named {name!r}")

        return ToolResult(success=False, validation_error=True,
                          error=f"unknown action {action!r}")
