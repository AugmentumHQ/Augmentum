"""Conduct contract for narrative-internal tools (recall + lorebook).

Why this exists (2026-07-15, Matt's live report): the recall/lorebook
schemas were attached BARE — no directive anywhere told the model that
these tools are silent bookkeeping machinery. Models with a strong
agentic register (DeepSeek especially) fill that vacuum by announcing
their tool plans in VISIBLE content: "Let me check whether there's a
lorebook entry about X…" lands in the story pane as a meta paragraph,
sometimes without any call following, forcing a cancel-and-regenerate.

This is also a flywheel-integrity fix, not just UX: narrative sessions
are captured as training exemplars for the mid-story bookkeeping
behavior (the lorebook tools are the biggest shown-but-never-trained
bucket in the registry coverage report). Serve must demonstrate the
correct shape — call first, prose after, tools never mentioned — or
every capture is an anti-exemplar.

Two levers, both applied at the handler's tool-attach site:

  * :data:`TOOL_CONDUCT_DIRECTIVE` — appended once to the system prompt
    whenever internal tools ride the request.
  * :func:`with_silent_suffix` — appends a one-line reminder to every
    tool's own description (models weigh the description at the moment
    they decide to call), on deep copies so the module-level schema
    constants are never mutated across turns.

The mechanical guarantee for models that ignore instructions lives in
``recall_loop.py`` (the pre-call gate); this module is the behavioral
half.

Relation to ``context_builder._build_tool_guidance``: that block teaches
WHEN to use the tools (check before writing, create when established) —
it is a priority-16 context block and can be BUDGET-CLIPPED on tight
turns while the tools stay attached. The conduct contract here is the
budget-immune complement: appended unconditionally at the handler's
tool-attach site, ~60 tokens, never dropped. Usage guidance and conduct
contract are deliberately single-sourced in their two homes — don't
duplicate one into the other.
"""

from __future__ import annotations

import copy
from typing import Any

from augmentum.models.base import Message

TOOL_CONDUCT_DIRECTIVE = (
    "[Tool conduct] The recall and lorebook tools are silent background "
    "bookkeeping. When you need one, call it FIRST, before writing any "
    "prose for the turn. Never mention the tools, the lorebook, notes, "
    "entries, plans to check or record anything, or tool results in the "
    "visible reply. The reader sees only the story itself. After results "
    "arrive, continue the story directly."
)

SILENT_SUFFIX = (
    "Silent background tool: never mention this tool, its use, or its "
    "results in the visible story."
)


def with_silent_suffix(schemas: list[dict]) -> list[dict]:
    """Deep-copied schemas with :data:`SILENT_SUFFIX` on each description.

    Deep copies are load-bearing: the inputs are module-level constants
    shared across every session — in-place mutation would stack the
    suffix once per turn forever.
    """
    out: list[dict] = []
    for schema in schemas:
        s2 = copy.deepcopy(schema)
        fn = s2.get("function")
        if isinstance(fn, dict):
            desc = str(fn.get("description") or "").rstrip()
            if SILENT_SUFFIX not in desc:
                fn["description"] = (desc + " " + SILENT_SUFFIX).strip()
        out.append(s2)
    return out


def append_conduct_directive(request: Any) -> bool:
    """Append the conduct directive to the request's system prompt.

    Appends to the FIRST system message (narrative composes exactly one),
    or inserts a new system message at index 0 when none exists (non-UI
    clients can send bare turns). Idempotent — a substring check keeps
    regenerate/retry paths from stacking copies. Returns True when the
    request was modified.
    """
    messages = getattr(request, "messages", None)
    if messages is None:
        return False
    for i, msg in enumerate(messages):
        if getattr(msg, "role", "") == "system":
            content = getattr(msg, "content", "") or ""
            if TOOL_CONDUCT_DIRECTIVE in content:
                return False
            messages[i] = Message(
                role="system",
                content=(content.rstrip() + "\n\n" + TOOL_CONDUCT_DIRECTIVE).strip(),
                images=getattr(msg, "images", None),
                tool_calls=getattr(msg, "tool_calls", None),
                thinking=getattr(msg, "thinking", None),
                tool_call_id=getattr(msg, "tool_call_id", None),
            )
            return True
    messages.insert(0, Message(role="system", content=TOOL_CONDUCT_DIRECTIVE))
    return True
