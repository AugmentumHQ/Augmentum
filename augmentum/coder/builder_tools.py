"""Builder resource tools — the app-builder toolkit as a *pulled-on resource*.

The legacy app builder force-fed its design system, references, and API
signatures into every generation prompt. The toolkit's own code admits small
models silently drop instructions past ~500 tokens, so force-feeding fights
the context budget and produces recipe-following instead of reasoning.

Under the Builder-on-the-coder-harness model (spec:
docs/superpowers/specs/2026-06-15-builder-profiles-system-synthesizer-design.md)
the same knowledge becomes tools the build agent *calls when it needs them*:

  builder_reference        — working code exemplars for a kind of app/game
  builder_design_system    — a concrete, WCAG-checked palette + typography
  builder_api_refs         — verified API signatures for the relevant surface

These are pure wrappers over functions in ``augmentum.tools.application_*``;
they carry no workspace/container state. The Builder facade appends them to
the coder tool set for a build session (they are NOT added to the default
coder tool list, so ordinary coder sessions are unaffected).
"""

from __future__ import annotations

import asyncio

from augmentum.tools.application_api_refs import api_refs_for_categories
from augmentum.tools.application_design_system import compute_design_system
from augmentum.tools.application_references import select_references
from augmentum.tools.application_scaffolds import SCAFFOLDS, _detect_categories
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult

# The kinds a build profile maps onto for toolkit lookups. Mirrors the
# scaffold vocabulary the pure functions key on; kept as a tuple so the
# tool input schemas can advertise it as an enum.
_KINDS = tuple(SCAFFOLDS.keys())  # ("static", "dashboard", "game", "form")
_DEFAULT_KIND = "static"


def _coerce_kind(kind: str) -> str:
    k = (kind or "").strip().lower()
    return k if k in SCAFFOLDS else _DEFAULT_KIND


class _BuilderResourceTool(Tool):
    """Base for stateless builder resource tools.

    No container/workspace state — these are pure lookups the agent pulls
    on. Coder-surface only; not exposed as chat function-calls.
    """

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def surfaces(self) -> SurfaceExposure:
        # Builder-internal: reachable by the build agent, not chat.
        return SurfaceExposure(chat=False, coder=True)

    @property
    def cacheable(self) -> bool:
        # Cheap + deterministic; results are stable for identical args.
        return True


class BuilderReferenceTool(_BuilderResourceTool):
    """Pull working code exemplars for the kind of thing being built."""

    @property
    def name(self) -> str:
        return "builder_reference"

    @property
    def description(self) -> str:
        return (
            "Pull working reference code for the kind of app/game you're "
            "building (the canonical patterns: game loop, Chart.js setup, "
            "form validation, state+render, etc.). Models copy a working "
            "pattern far more reliably than they follow prose — call this "
            "before writing a file in unfamiliar territory. Returns 1-N "
            "tagged code skeletons selected for your description."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": list(_KINDS),
                    "default": _DEFAULT_KIND,
                    "description": "Project kind. Picks the reference family.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "What you're building (e.g. 'snake game with score', "
                        "'tip calculator'). Used to rank references by relevance."
                    ),
                },
                "max_refs": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 7,
                    "description": "How many reference skeletons to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, *, kind: str = _DEFAULT_KIND, query: str = "", max_refs: int = 3, **_kw) -> ToolResult:
        scaffold = _coerce_kind(kind)
        n = max(1, min(int(max_refs or 3), 7))
        # select_references may run fastembed (sync) for semantic ranking —
        # offload so it never blocks the event loop. Falls back to keyword
        # scoring internally when embeddings aren't available.
        try:
            block = await asyncio.to_thread(select_references, query or scaffold, scaffold, n)
        except Exception as exc:  # noqa: BLE001 — resource lookup must not break the build
            return ToolResult(success=False, error=f"reference lookup failed: {exc}")
        if not (block or "").strip():
            return ToolResult(
                success=True,
                output=f"(no specific references matched '{query}' for kind '{scaffold}')",
            )
        return ToolResult(success=True, output=block)


class BuilderDesignSystemTool(_BuilderResourceTool):
    """Get a concrete, accessible design system to build against."""

    @property
    def name(self) -> str:
        return "builder_design_system"

    @property
    def description(self) -> str:
        return (
            "Get a concrete design system (CSS custom-property palette, "
            "typography, radii) tailored to the app's mood and guaranteed "
            "WCAG-AA contrast. Reference these variables instead of "
            "hardcoding colors — hardcoded ad-hoc palettes are the tell "
            "that makes generated UIs look generated. Call once up front."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The app description; drives the mood/palette.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(_KINDS),
                    "default": _DEFAULT_KIND,
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        }

    async def execute(self, *, description: str = "", kind: str = _DEFAULT_KIND, **_kw) -> ToolResult:
        scaffold = _coerce_kind(kind)
        try:
            ds = compute_design_system(description or "", scaffold)
            guidance = ds.guidance_for_prompt()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"design system computation failed: {exc}")
        return ToolResult(success=True, output=guidance, metadata={"mood": getattr(ds, "mood", "")})


class BuilderApiRefsTool(_BuilderResourceTool):
    """Verified API signatures for the surfaces this build touches."""

    @property
    def name(self) -> str:
        return "builder_api_refs"

    @property
    def description(self) -> str:
        return (
            "Get verified API signatures for the surfaces your app uses "
            "(Canvas 2D for games, Chart.js for dashboards, form handling, "
            "Intl number/date formatting). Anchors you to real APIs so you "
            "don't invent methods that don't exist (e.g. ctx.fillCircle). "
            "Call before writing code against an API you're unsure of."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The app description; used to detect which API surfaces apply.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(_KINDS),
                    "default": _DEFAULT_KIND,
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["canvas_game", "charts_dashboard", "interactive_form", "data_visualization"],
                    },
                    "description": (
                        "Optional explicit API categories. When omitted, "
                        "categories are auto-detected from description + kind."
                    ),
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        }

    async def execute(self, *, description: str = "", kind: str = _DEFAULT_KIND, categories: list | None = None, **_kw) -> ToolResult:
        scaffold = _coerce_kind(kind)
        try:
            cats = list(categories) if categories else _detect_categories(description or "", scaffold)
            block = api_refs_for_categories(cats)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"api-refs lookup failed: {exc}")
        if not (block or "").strip():
            return ToolResult(
                success=True,
                output="(no specialized API surface detected — standard DOM/JS applies)",
                metadata={"categories": cats},
            )
        return ToolResult(success=True, output=block, metadata={"categories": cats})


# Canonical order; the Builder facade appends these to the coder tool set.
BUILDER_TOOLS: list[type[_BuilderResourceTool]] = [
    BuilderReferenceTool,
    BuilderDesignSystemTool,
    BuilderApiRefsTool,
]


def create_builder_tools() -> list[Tool]:
    """Instantiate the builder resource tools (stateless).

    The Builder facade calls this and appends the result to
    ``create_coder_tools(...)`` for a build session, so the build agent can
    pull on the toolkit on demand without these tools widening the surface
    of ordinary coder sessions.
    """
    return [cls() for cls in BUILDER_TOOLS]
