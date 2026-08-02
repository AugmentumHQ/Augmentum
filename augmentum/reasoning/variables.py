"""Variable substitution for reasoning flow prompts.

Resolves template variables like {query}, {previous_output}, {step:Name}
in system prompts and user templates at runtime.
"""

from __future__ import annotations

import re

from augmentum.utils.datetime_context import get_datetime_context


class StepContext:
    """Accumulates step outputs during flow execution for variable resolution."""

    def __init__(self, query: str, model: str = "") -> None:
        self.query = query
        self.model = model
        self.complexity = ""
        self.conversation = ""
        self.search_results = ""
        self.plan = ""  # Agentic mode: running plan markdown (attention anchor)
        self._step_outputs: dict[str, str] = {}
        self._previous_output = ""

    def record_step(self, step_name: str, output: str) -> None:
        """Record the output of a completed step."""
        self._step_outputs[step_name] = output
        self._previous_output = output

    @property
    def previous_output(self) -> str:
        return self._previous_output

    # Steps whose output is internal pipeline metadata (not useful for synthesis)
    _INTERNAL_STEPS: frozenset[str] = frozenset({
        "Classify",       # TYPE/DOMAIN/COMPLEXITY markers
    })

    # Per-step character budget for the bulk {all_outputs} concatenation.
    # Whole step outputs are stored uncapped in _step_outputs (so {step:Name}
    # and {previous_output} stay full-fidelity); only the bulk concat that
    # feeds late synthesis/respond steps is bounded, by keeping leading whole
    # paragraphs up to this budget. Matches the agentic deliver-context budget.
    _ALL_OUTPUTS_PER_STEP_BUDGET: int = 2500

    @classmethod
    def _budget_truncate(cls, text: str, budget: int) -> str:
        """Keep leading whole paragraphs up to ``budget`` chars."""
        if len(text) <= budget:
            return text
        paras = [p for p in text.split("\n\n") if p.strip()]
        kept: list[str] = []
        used = 0
        for p in paras:
            if used and used + len(p) > budget:
                break
            kept.append(p)
            used += len(p) + 2
        body = "\n\n".join(kept) if kept else text[:budget]
        # Guard against a single mega-paragraph blowing the budget wide open.
        hard = budget * 2
        if len(body) > hard:
            body = body[:hard].rstrip()
        if len(body) < len("\n\n".join(paras)):
            omitted = max(0, len(paras) - len(kept))
            tail = f"[... {omitted} more paragraph(s) omitted]" if omitted else "[... truncated]"
            body += f"\n\n{tail}"
        return body

    @property
    def all_outputs(self) -> str:
        if not self._step_outputs:
            return ""
        parts = []
        for name, output in self._step_outputs.items():
            # Skip internal pipeline metadata — no value for the final response.
            # Underscore-prefixed keys (_delivery_context, _user_message, ...) are
            # scratch slots, not real steps — including them would duplicate the
            # whole context back into itself.
            if name in self._INTERNAL_STEPS or name.startswith("_"):
                continue
            # Strip raw verification markers that small models tend to echo
            cleaned = output
            if name == "Fact-Check" or name.endswith("_backtrack"):
                # Keep only the substantive content, not VERIFIED:/CONFIDENCE: markers
                lines = [
                    ln for ln in cleaned.splitlines()
                    if not ln.strip().startswith(("VERIFIED:", "CONFIDENCE:", "<output>", "</output>"))
                ]
                cleaned = "\n".join(lines).strip()
                if not cleaned:
                    continue
            cleaned = self._budget_truncate(cleaned, self._ALL_OUTPUTS_PER_STEP_BUDGET)
            parts.append(f"## {name}\n{cleaned}")
        return "\n\n".join(parts)

    def get_step_output(self, step_name: str) -> str:
        return self._step_outputs.get(step_name, "")


def resolve_variables(template: str, ctx: StepContext, tools_section: str = "") -> str:
    """Substitute template variables with runtime values.

    Supported variables:
        {query}            - user's original question
        {conversation}     - recent conversation history
        {search_results}   - web search results
        {previous_output}  - output from preceding step
        {step:Name}        - output from a specific named step
        {all_outputs}      - concatenated outputs from all prior steps
        {complexity}       - detected complexity level
        {model}            - current model name
        {tools}            - auto-generated tool description section
        {plan}             - agentic mode running plan (attention anchor)
    """
    if not template:
        return template

    result = template

    # Simple replacements
    replacements = {
        "{query}": ctx.query,
        "{conversation}": ctx.conversation,
        "{search_results}": ctx.search_results,
        "{previous_output}": ctx.previous_output,
        "{all_outputs}": ctx.all_outputs,
        "{complexity}": ctx.complexity,
        "{model}": ctx.model,
        "{tools}": tools_section,
        "{plan}": ctx.plan,
        "{current_date}": get_datetime_context(),
    }

    for var, value in replacements.items():
        if var in result:
            result = result.replace(var, value)

    # Step-specific: {step:Name}
    step_refs = re.findall(r"\{step:([^}]+)\}", result)
    for step_name in step_refs:
        output = ctx.get_step_output(step_name)
        result = result.replace(f"{{step:{step_name}}}", output)

    return result


# Default user template used when a step doesn't specify one.
DEFAULT_USER_TEMPLATE = """\
## Query
{query}

{conversation}
{previous_output}
{search_results}
{tools}"""

# Default user template for steps inside a reasoning *flow* that don't specify
# one. Unlike DEFAULT_USER_TEMPLATE this carries ALL prior step outputs, not
# just the immediately-preceding one — flow steps such as "Synthesize" /
# "Cross-Reference" / "Verify Fixes" exist precisely to combine several earlier
# steps and were previously starved of everything but step N-1.
FLOW_STEP_USER_TEMPLATE = """\
## Query
{query}

{conversation}
{all_outputs}
{search_results}
{tools}"""


def build_user_message(
    user_template: str,
    ctx: StepContext,
    tools_section: str = "",
) -> str:
    """Build the user message for a step.

    Uses the step's user_template if provided, otherwise falls back
    to DEFAULT_USER_TEMPLATE. Strips excessive blank lines.
    """
    template = user_template.strip() if user_template else DEFAULT_USER_TEMPLATE
    result = resolve_variables(template, ctx, tools_section)

    # Clean up: collapse 3+ consecutive newlines to 2
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")

    return result.strip()
