"""Native-tool proxies for gated capabilities.

Gated capabilities (``image_generation``, ``build_application``, the structured
creators) are heavy: the model PROPOSES them and the user confirms via an offer
chip before they run. For the model to propose one in Auto mode it has to SEE
the capability — but Auto mode hands the model native function-call schemas. The
old ``[[tool:NAME]]`` text marker that the gated branch once relied on has no
production caller anymore (modern function-calling models reliably miss it), so
each gated capability is exposed here as a minimal single-arg NATIVE tool.

The proxy is never executed. The passthrough resolvers intercept a parsed gated
call BEFORE execution and surface the confirmation offer instead (see
``PassthroughHandler._first_gated`` / ``_surface_gated_offer``). The schema is
deliberately a single string — the capability's ``primary_arg`` — so the model's
whole job is "describe what you want"; structuring (ebook chapters, deck slides)
is the planner's job AFTER the user confirms the outline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.tools.base import Tool, ToolCategory, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    from augmentum.modes.passthrough.orchestrator import ModelCapability
    from augmentum.tools.registry import ToolRegistry


# Per-arg guidance so the single-field schema reads well to the model.
_ARG_DESCRIPTIONS: dict[str, str] = {
    "prompt": "A vivid, detailed description of the image to generate.",
    "description": "What the application should do and how it should look.",
    "brief": (
        "A one-line description of what to create — you'll confirm a full "
        "outline with the user before anything is generated."
    ),
}


class GatedProxyTool(Tool):
    """Minimal single-arg stand-in so the model can REQUEST a gated capability
    via native function-calling. Never executed — the handler intercepts gated
    calls and turns them into a confirmation offer."""

    # Marker the handler / chain paths check to treat this as propose-only.
    is_gated_proxy = True

    def __init__(
        self, cap: ModelCapability, *, description: str, model_hint: str = "",
    ) -> None:
        self._cap = cap
        self._description = description
        self._model_hint = model_hint

    @property
    def name(self) -> str:
        return self._cap.tool

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> ToolCategory:
        return (
            ToolCategory.IMAGE
            if self._cap.tool == "image_generation"
            else ToolCategory.ARTIFACT
        )

    @property
    def input_schema(self) -> dict:
        arg = self._cap.primary_arg
        return {
            "type": "object",
            "properties": {
                arg: {
                    "type": "string",
                    "description": _ARG_DESCRIPTIONS.get(arg, "What to create."),
                },
            },
            "required": [arg],
        }

    @property
    def model_hint(self) -> str:
        return self._model_hint

    @property
    def cacheable(self) -> bool:
        # Side-effecting proposal — never a cache target.
        return False

    async def execute(self, **kwargs: Any) -> ToolResult:
        # Never reached: gated calls are intercepted before execution. Defensive
        # — surface a clear error rather than silently doing nothing.
        return ToolResult(
            success=False,
            error=(
                "gated capability must be confirmed via an offer chip, not "
                "executed directly"
            ),
        )


def build_gated_proxy_tools(
    gated_caps: Iterable[ModelCapability], registry: ToolRegistry | None,
) -> list[GatedProxyTool]:
    """One proxy per gated capability whose real tool is registered on THIS
    install. A capability whose tool is absent (e.g. ``image_generation`` with
    no image provider configured) is skipped — we don't advertise something the
    user can't actually run. Borrows the real tool's description / model_hint so
    the model gets accurate guidance."""
    out: list[GatedProxyTool] = []
    if registry is None:
        return out
    resolve = getattr(registry, "resolve", None) or getattr(registry, "get", None)
    for cap in gated_caps:
        real = resolve(cap.tool) if resolve else None
        if real is None:
            continue  # capability unavailable here — don't offer it
        desc = (getattr(real, "description", "") or cap.fallback_hint).strip()
        hint = (getattr(real, "model_hint", "") or "").strip()
        # Routing contract rides the DESCRIPTION, not just model_hint:
        # model_hint only reaches the wire for small models (handler tier
        # gate), but choosing proxy-vs-inline is a routing decision every
        # model size gets wrong when it only sees the sales pitch — the
        # observed failure was a large model escalating an explicit
        # "single html file" request into the build pipeline.
        if hint and hint not in desc:
            desc = f"{desc} {hint}"
        out.append(GatedProxyTool(cap, description=desc, model_hint=hint))
    return out
