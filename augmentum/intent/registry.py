"""Action registry — singleton store + register_action decorator.

Importing a builtin module (``augmentum.intent.builtin.control``) runs
its ``@register_action(...)`` calls, which populate ``REGISTRY``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from augmentum.intent.action import (
    Action,
    ActionFanout,
    ArgInferrer,
    ArgTransformer,
    HandlerFn,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Module-level store. Single registry per process; tests can monkey-
# patch by directly mutating ``REGISTRY._actions`` if needed.
class _ActionStore:
    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def add(self, action: Action) -> None:
        if action.id in self._actions:
            log.warning("intent_action_redefined", id=action.id)
        self._actions[action.id] = action

    def get(self, action_id: str) -> Action | None:
        return self._actions.get(action_id)

    def all(self) -> list[Action]:
        return list(self._actions.values())

    def __len__(self) -> int:
        return len(self._actions)


REGISTRY = _ActionStore()

# Stakes tiers a DEFERRED, user-stated action may carry — timer
# then-actions ("in 20 minutes pause the music") and scheduled
# verb_fire tasks. Wider than app.act's trivial_reversible-only cap
# because the user explicitly asked for the action; narrower than
# everything because nobody should wake up to an unattended
# "costly"/"irrevocable" having fired.
DEFERRED_ACTION_STAKES = frozenset({"trivial_reversible", "disruptive"})


def _compile_patterns(
    patterns: list[str],
    examples: list[str],
) -> list[re.Pattern[str]]:
    """Compile explicit patterns + auto-derive from examples.

    Explicit ``patterns`` take precedence; ``examples`` only seed a
    pattern when no explicit ones were given (so a quick action
    definition with just examples still works). Auto-derived patterns
    are word-boundary-wrapped and treat punctuation/whitespace
    permissively.
    """
    compiled: list[re.Pattern[str]] = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as exc:
            log.warning("intent_pattern_invalid", pattern=p, error=str(exc))
            continue

    if not compiled and examples:
        # Auto-seed — escape each example and allow flexible whitespace.
        for ex in examples:
            tokens = ex.strip().split()
            if not tokens:
                continue
            joined = r"\s+".join(re.escape(t) for t in tokens)
            try:
                compiled.append(re.compile(rf"\b{joined}\b", re.IGNORECASE))
            except re.error:
                continue

    return compiled


def register_action(
    *,
    id: str,
    summary: str,
    examples: list[str],
    handler: HandlerFn,
    patterns: list[str] | None = None,
    templates: list[str] | None = None,
    arg_schema: dict | None = None,
    required: list[str] | None = None,
    modes: list[str] | None = None,
    fanout: ActionFanout | None = None,
    surfaces: list[str] | None = None,
    arg_inferrer: ArgInferrer | None = None,
    arg_transformer: ArgTransformer | None = None,
    companion_initiatable: bool = False,
    stakes: str = "trivial_reversible",
    delivery: str = "verbal",
) -> Callable[[HandlerFn], HandlerFn]:
    """Register an action.

    Usable as a decorator (``@register_action(id=..., handler=fn)``)
    or as a direct call. Both forms return the original handler so
    callers can decorate without losing access to the function.

    ``required`` is the JSON Schema ``required`` list — names of args
    the LLM MUST provide. The tool adapter surfaces it to native
    function-calling so the model gets validation errors before the
    handler runs.

    Architect extensions:

      * ``surfaces`` filters which CLIENT surface(s) expose this action
        ('voice'/'chat'/'cast'/'xr'). Empty = all surfaces.
      * ``arg_inferrer`` runs between matcher hit and handler call, filling
        missing args from observation history (device_play_history,
        image_generations, browse_history, ReferentCache).
      * ``companion_initiatable`` allows the runtime to dispatch this
        action without an explicit user command. Default False; flip
        per-primitive only after a careful design pass.
    """
    # Compile hassil-style templates at registration time so the
    # matcher loop stays branch-free at runtime. Compilation errors
    # bubble up loudly — a malformed template is a developer bug,
    # not a runtime soft-fail.
    #
    # Lint discipline: primitives are for ACTIONS, not questions.
    # Templates that start with a WH-question word will fire on
    # genuinely-conversational utterances like "what time is it"
    # and produce canned responses where a thoughtful LLM reply
    # belongs. Warn at registration so the design rule is visible.
    compiled_templates: list = []
    if templates:
        import re as _re

        from augmentum.intent.templates import compile_template
        _question_prefix = _re.compile(
            r"^\s*(?:\[[^\]]*\]\s*)*"  # leading optional groups
            r"\(?(?:what|how|where|when|why|who|which|what's|how's|where's)\b",
            _re.IGNORECASE,
        )
        for tmpl in templates:
            if _question_prefix.match(tmpl):
                # Startup-only design hint — not a runtime failure. Was
                # firing as ``warning`` at every cold start (~80 per boot)
                # which crowded the warning floor for active issues.
                # Debug keeps the signal for intent-registry refactors
                # without making every container restart look unhealthy.
                log.debug(
                    "intent_template_question_form",
                    id=id, template=tmpl[:80],
                    note=(
                        "templates starting with a WH-word match "
                        "questions, which should usually route to the "
                        "LLM (conversational path) not a primitive. "
                        "Consider whether this is the right design."
                    ),
                )
            compiled_templates.append(compile_template(tmpl))

    action = Action(
        id=id,
        summary=summary,
        examples=examples,
        handler=handler,
        patterns=_compile_patterns(patterns or [], examples),
        compiled_templates=compiled_templates,
        arg_schema=arg_schema or {},
        required_args=list(required) if required else [],
        modes=modes or [],
        fanout=fanout or ActionFanout(),
        surfaces=list(surfaces) if surfaces else [],
        arg_inferrer=arg_inferrer,
        arg_transformer=arg_transformer,
        companion_initiatable=companion_initiatable,
        stakes=stakes,
        delivery=delivery,
    )
    # Reject unknown stakes values loudly — typos here silently default
    # actions to trivial_reversible, which is the wrong direction for a
    # primitive someone meant to annotate as irrevocable.
    _ALLOWED_STAKES = {
        "trivial_reversible", "disruptive", "costly",
        "personal", "irrevocable", "safety_critical",
    }
    if stakes not in _ALLOWED_STAKES:
        raise ValueError(
            f"register_action({id!r}): stakes={stakes!r} is not one of "
            f"{sorted(_ALLOWED_STAKES)}. See "
            f"docs/superpowers/specs/2026-05-28-confidence-tier-dispatch-design.md"
        )
    if delivery not in ("verbal", "artifact"):
        raise ValueError(
            f"register_action({id!r}): delivery={delivery!r} must be "
            f"'verbal' or 'artifact'"
        )
    REGISTRY.add(action)

    def _identity(fn: HandlerFn) -> HandlerFn:
        return fn

    return _identity


def list_actions(mode: str | None = None) -> list[Action]:
    """Return registered actions, optionally filtered by mode."""
    actions = REGISTRY.all()
    if mode is None:
        return actions
    return [a for a in actions if a.available_in(mode)]
