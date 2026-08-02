"""ATP recipes — one call replays a saved sequence of ATP tool calls.

The problem this solves: reaching any signed-in surface (or any repeated
multi-tool task) costs a fixed choreography every session — e.g. reviewing
the image gallery is ensure_auth -> navigate -> screenshot, 3+ round trips
the model re-derives each time. A recipe lets a harness (or the user) save
that sequence once under a name and replay it in a single ``atp_recipe``
call, with ``{{placeholders}}`` filled at run time.

Isolation: recipes are per-user (recipe_store enforces user_id on every
row), and each replayed step is executed through the SAME registry with the
SAME force-injected ``_context`` the ATP ``/call`` route uses — a recipe can
only reach tools its owner could already call, as their own user. Steps may
only invoke ATP-whitelisted or discoverable tools, and never ``atp_recipe``
itself (no recursion).
"""

from __future__ import annotations

import re
from typing import Any

from augmentum.tools import recipe_store
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _substitute(value: Any, params: dict) -> Any:
    """Fill ``{{key}}`` tokens from params, recursively. A string that is
    EXACTLY one placeholder yields the raw param value (so numbers/bools/
    objects survive); mixed strings are interpolated as text."""
    if isinstance(value, str):
        m = _PLACEHOLDER.fullmatch(value.strip())
        if m:
            return params.get(m.group(1), value)
        return _PLACEHOLDER.sub(lambda mm: str(params.get(mm.group(1), mm.group(0))), value)
    if isinstance(value, dict):
        return {k: _substitute(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, params) for v in value]
    return value


class AtpRecipeTool(Tool):
    """Save, list, delete, and run named per-user ATP macros."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "atp_recipe"

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
    def timeout(self) -> float:
        return 180.0

    @property
    def description(self) -> str:
        return (
            "Save and replay named macros over ATP tools — crystallize a "
            "repeated tool sequence into one call. "
            "action='save' {name, steps:[{tool, arguments}], description?} — "
            "arguments may hold {{placeholder}} tokens. "
            "action='run' {name, params?} — replays the steps, filling "
            "placeholders from params, and returns each step's output. "
            "action='list' — your saved recipes. "
            "action='get' {name} / action='delete' {name}. "
            "Example: save 'review_gallery' = "
            "[{tool:'browser_ensure_auth'}, "
            "{tool:'browser_screenshot', arguments:{url:'{{ui_base}}/ui/#gallery', full_page:true}}], "
            "then run it with params={ui_base:'https://host.docker.internal:6443'}."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["save", "run", "list", "get", "delete"]},
                "name": {"type": "string", "description": "recipe name (all but 'list')"},
                "steps": {
                    "type": "array",
                    "description": "for save: [{tool, arguments}] in order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["tool"],
                    },
                },
                "params": {"type": "object",
                           "description": "for run: values for {{placeholders}}"},
                "description": {"type": "string"},
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="no authenticated user")
        action = str(kwargs.get("action") or "").strip().lower()
        name = str(kwargs.get("name") or "").strip()

        if action == "list":
            rows = await recipe_store.list_recipes(self._app_state, user_id=user_id)
            if not rows:
                return ToolResult(success=True, output="No saved recipes yet.",
                                  metadata={"recipes": []})
            lines = [f"- {r['name']} ({r['steps']} steps)"
                     + (f" — {r['description']}" if r['description'] else "")
                     for r in rows]
            return ToolResult(success=True, output="Saved recipes:\n" + "\n".join(lines),
                              metadata={"recipes": rows})

        if action == "get":
            rec = await recipe_store.get_recipe(self._app_state, user_id=user_id, name=name)
            if rec is None:
                return ToolResult(success=False, error=f"no recipe named {name!r}")
            steps_desc = "\n".join(
                f"  {i+1}. {s.get('tool')}({', '.join(sorted((s.get('arguments') or {}).keys()))})"
                for i, s in enumerate(rec["steps"])
            )
            return ToolResult(
                success=True,
                output=f"{rec['name']}: {rec['description'] or '(no description)'}\n{steps_desc}",
                metadata=rec,
            )

        if action == "delete":
            ok = await recipe_store.delete_recipe(self._app_state, user_id=user_id, name=name)
            return ToolResult(success=ok,
                              output=f"Deleted recipe {name!r}." if ok else "",
                              error="" if ok else f"could not delete {name!r}")

        if action == "save":
            steps, err = recipe_store.validate_steps(kwargs.get("steps"))
            if err:
                return ToolResult(success=False, validation_error=True, error=err)
            # Reject steps that name tools this user can't reach via ATP.
            bad = self._ungated_step_tools(steps)
            if bad:
                return ToolResult(
                    success=False, validation_error=True,
                    error=(f"these step tools are not ATP-callable: {', '.join(bad)}. "
                           "Use tools from /v1/tools/list or discoverable via /discover."),
                )
            ctx = kwargs.get("_context") if isinstance(kwargs.get("_context"), dict) else {}
            saved = await recipe_store.save_recipe(
                self._app_state, user_id=user_id, name=name, steps=steps,
                description=str(kwargs.get("description") or ""),
                harness=str(ctx.get("harness") or ""),
            )
            if saved is None:
                return ToolResult(success=False, error="could not save recipe (name required)")
            return ToolResult(
                success=True,
                output=f"Saved recipe {saved['name']!r} ({len(steps)} steps). "
                       f"Run it with action='run', name={saved['name']!r}.",
                metadata=saved,
            )

        if action == "run":
            return await self._run(user_id, name, kwargs)

        return ToolResult(success=False, validation_error=True,
                          error=f"unknown action {action!r}")

    # ------------------------------------------------------------------

    def _registry(self):
        return getattr(self._app_state, "tool_registry", None)

    def _gate(self):
        """Return (ATP_TOOLS, _discoverable) mirroring the /call route's
        reachability rule. Lazy import avoids an atp_routes<->tools cycle."""
        from augmentum.proxy.atp_routes import ATP_TOOLS, _discoverable
        return ATP_TOOLS, _discoverable

    def _ungated_step_tools(self, steps: list[dict]) -> list[str]:
        registry = self._registry()
        if registry is None:
            return []  # can't check now; run-time gate still applies
        atp_tools, discoverable = self._gate()
        bad = []
        for s in steps:
            tname = s.get("tool")
            tool = registry.resolve(tname)
            if tname not in atp_tools and (tool is None or not discoverable(tool)):
                bad.append(tname)
        return bad

    async def _run(self, user_id: str, name: str, kwargs: dict) -> ToolResult:
        rec = await recipe_store.get_recipe(self._app_state, user_id=user_id, name=name)
        if rec is None:
            return ToolResult(success=False, error=f"no recipe named {name!r}")
        registry = self._registry()
        if registry is None:
            return ToolResult(success=False, error="tool registry unavailable")
        atp_tools, discoverable = self._gate()
        params = kwargs.get("params") if isinstance(kwargs.get("params"), dict) else {}
        # Reuse the caller's force-injected context for every sub-call so
        # user isolation and harness/project scope carry through unchanged.
        ctx = kwargs.get("_context") if isinstance(kwargs.get("_context"), dict) else {"user_id": user_id}

        segments: list[str] = []
        for i, step in enumerate(rec["steps"]):
            tname = step.get("tool")
            tool = registry.resolve(tname)
            if tname not in atp_tools and (tool is None or not discoverable(tool)):
                return self._partial(segments, i, tname,
                                     f"tool {tname!r} is not ATP-callable")
            if tool is None:
                return self._partial(segments, i, tname, "tool not found")
            args = _substitute(step.get("arguments") or {}, params)
            args["_context"] = ctx
            try:
                result: ToolResult = await tool.execute(**args)
            except TypeError:
                args.pop("_context", None)
                args.pop("_user_id", None)
                try:
                    result = await tool.execute(**args)
                except Exception as exc:  # noqa: BLE001
                    return self._partial(segments, i, tname, f"bad arguments: {exc}")
            except Exception as exc:  # noqa: BLE001
                return self._partial(segments, i, tname, str(exc))
            if not result.success:
                segments.append(f"[{i+1}/{len(rec['steps'])}] {tname}: FAILED — {result.error}")
                return ToolResult(
                    success=False,
                    error=f"recipe {name!r} stopped at step {i+1} ({tname}): {result.error}",
                    output="\n\n".join(segments),
                )
            segments.append(f"[{i+1}/{len(rec['steps'])}] {tname}:\n{result.output}")

        return ToolResult(
            success=True,
            output=f"Recipe {name!r} completed ({len(rec['steps'])} steps).\n\n"
                   + "\n\n".join(segments),
            metadata={"recipe": name, "steps_run": len(rec["steps"])},
        )

    @staticmethod
    def _partial(segments: list[str], idx: int, tname: str, err: str) -> ToolResult:
        segments.append(f"[step {idx+1}] {tname}: ERROR — {err}")
        return ToolResult(success=False, error=f"step {idx+1} ({tname}): {err}",
                          output="\n\n".join(segments))
