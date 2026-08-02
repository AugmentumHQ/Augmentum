"""Plan decomposition and todo.md management for agentic mode.

The plan serves as the attention anchor — it's re-injected at the end of
every subsequent step's context to keep the model focused during long chains.
"""

from __future__ import annotations

import re

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# System prompt for the planning step
PLAN_SYSTEM_PROMPT = """\
You are a task planner. Given the user's request, create a structured execution plan.

Output a numbered checklist in markdown. Each item should be a clear, actionable step.
Mark all items as unchecked ([ ]). Keep the plan between 3-8 steps.

Format:
## Task: <brief task title>

- [ ] 1. <step description>
- [ ] 2. <step description>
...

After the checklist, add a brief "Notes:" section capturing any user constraints \
or preferences mentioned in the request."""


def parse_plan(raw_plan: str) -> tuple[str, list[str]]:
    """Parse LLM plan output into a title and list of step descriptions.

    Returns (title, [step_descriptions]).
    """
    title = ""
    steps: list[str] = []

    # Extract title from "## Task: ..." line
    title_match = re.search(r"##\s*Task:\s*(.+)", raw_plan)
    if title_match:
        title = title_match.group(1).strip()

    # Extract checklist items
    for match in re.finditer(r"-\s*\[[ x]\]\s*\d*\.?\s*(.+)", raw_plan):
        step_text = match.group(1).strip()
        if step_text:
            steps.append(step_text)

    # Fallback: if no checklist found, try numbered list
    if not steps:
        for match in re.finditer(r"^\d+[.)]\s*(.+)", raw_plan, re.MULTILINE):
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)

    # Fallback title
    if not title and steps:
        title = steps[0][:60]

    return title, steps


def update_plan_step(plan_md: str, step_index: int, note: str = "") -> str:
    """Mark a step as completed in the plan markdown.

    Args:
        plan_md: Current plan markdown.
        step_index: 0-based index of the step to mark complete.
        note: Optional note to append after the step.
    """
    lines = plan_md.split("\n")
    checklist_idx = 0
    result_lines: list[str] = []

    for line in lines:
        if re.match(r"\s*-\s*\[[ x]\]", line):
            if checklist_idx == step_index and "[ ]" in line:
                # Mark as complete
                line = line.replace("[ ]", "[x]", 1)
                if note:
                    line = f"{line} ({note})"
            checklist_idx += 1
        result_lines.append(line)

    return "\n".join(result_lines)


def mark_current_step(plan_md: str, step_index: int) -> str:
    """Add a CURRENT marker to the active step for attention anchoring."""
    lines = plan_md.split("\n")
    checklist_idx = 0
    result_lines: list[str] = []

    for line in lines:
        # Remove existing CURRENT markers
        cleaned = re.sub(r"\s*← CURRENT\s*$", "", line)
        if re.match(r"\s*-\s*\[[ x]\]", cleaned):
            if checklist_idx == step_index and "[ ]" in cleaned:
                cleaned = f"{cleaned} ← CURRENT"
            checklist_idx += 1
        result_lines.append(cleaned)

    return "\n".join(result_lines)


def plan_to_context(plan_md: str) -> str:
    """Format the plan as a context injection block for step prompts."""
    if not plan_md.strip():
        return ""
    return f"\n\n---\n## Current Task Plan\n{plan_md}\n---"
