"""Parent → child context inheritance for task_dispatch.

Three modes match the role-file ``context.mode`` field:

* **slim**     — minimal orientation only: the `<orientation>` anchor
  (objective + project-shape line, ≤ ~240 chars). Even a slim role
  needs to know what the session is FOR and what stack it's on, or a
  ``research`` answer drifts generic and an ``audit_zone`` reviewer
  judges code without knowing what the system is.
* **workspace** — orientation + the full `<workspace_facts>` block
  (kernel facts + observations + objective) prepended to the prompt.
  Most subagents want this — keeps continuity without dragging the
  parent's turn-by-turn chatter into the child loop.
* **hot**      — workspace + the parent's last N tool-result digests
  (truncated to keep token cost bounded). Use for "continue what I was
  investigating" roles.

The ``<orientation>`` anchor is included for EVERY mode (it's the
cheapest, highest-leverage signal); ``<workspace_facts>`` is added on
top for workspace/hot. Both are pulled from the live CoderState
(``orientation_text`` / ``kernel_facts_text``), set each turn by the
coder handler's ``_refresh_kernel_facts``.

The kernel facts block is the same one the coder's _refresh_kernel_facts
already produces; we pull it from the live CoderState when available
and otherwise emit an empty block. Decoupled via duck-typing rather
than a hard import so the agents package doesn't take a coder-mode
dependency.
"""

from __future__ import annotations

from typing import Any


def _safe_str(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _render_bullets(tag: str, items: list[str] | tuple[str, ...] | None) -> str:
    """Wrap a list of short strings in a ``<tag>`` block, one bullet each.

    Returns ``""`` when there's nothing to render. Each item is truncated
    so a verbose lead can't blow the child's prompt budget.
    """
    if not items:
        return ""
    bullets = [
        "- " + _safe_str(str(it).strip(), 240)
        for it in items
        if it and str(it).strip()
    ]
    if not bullets:
        return ""
    return f"<{tag}>\n" + "\n".join(bullets) + f"\n</{tag}>"


def build_initial_user_message(
    *,
    prompt: str,
    context_mode: str,
    orientation: str = "",
    workspace_facts: str = "",
    recent_tool_digests: list[str] | None = None,
    success_criteria: list[str] | tuple[str, ...] | None = None,
    constraints: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Compose the subagent's first user message per the context-bridge mode.

    The shape is intentionally compact so it doesn't dominate the
    child's prompt budget. Orientation is ≤ ~240 chars (all modes);
    workspace block is ≤ ~2K chars; hot block adds ≤ ~1K more.

    ``orientation`` is the already-wrapped ``<orientation>`` block from
    the kernel; ``workspace_facts`` is the inner text of the
    ``<workspace_facts>`` block (wrapped here). ``success_criteria`` and
    ``constraints`` are the lead's definition-of-done / hard limits,
    rendered as bullet blocks for EVERY mode (they're cheap and they're
    the whole point of a focused delegation). Empty values are skipped,
    so callers can pass them unconditionally.
    """
    mode = (context_mode or "workspace").strip().lower()
    if mode not in {"slim", "workspace", "hot"}:
        mode = "workspace"

    sections: list[str] = []

    # Orientation anchor — included for EVERY mode, including slim. It's
    # the cheapest, highest-leverage signal (objective + project shape).
    anchor = _safe_str(orientation, 400).strip()
    if anchor:
        sections.append(anchor)

    # Definition-of-done + hard limits — included for EVERY mode. These
    # carry the lead's actual intent so the child self-checks against it
    # rather than guessing from the freehand prompt alone.
    criteria_block = _render_bullets("success_criteria", success_criteria)
    if criteria_block:
        sections.append(criteria_block)
    constraints_block = _render_bullets("constraints", constraints)
    if constraints_block:
        sections.append(constraints_block)

    if mode != "slim":
        facts = _safe_str(workspace_facts, 2_000).strip()
        if facts:
            # render_facts_block() already wraps in <workspace_facts>; a
            # bare string (e.g. a test fixture or legacy caller) does not.
            # Wrap idempotently so we never double-tag.
            if facts.startswith("<workspace_facts>"):
                sections.append(facts)
            else:
                sections.append("<workspace_facts>\n" + facts + "\n</workspace_facts>")

        if mode == "hot" and recent_tool_digests:
            # One line per tool digest, last 8 entries, truncated.
            tail = recent_tool_digests[-8:]
            body = "\n".join("- " + _safe_str(line, 200) for line in tail if line)
            if body:
                sections.append("<recent_activity>\n" + body + "\n</recent_activity>")

    sections.append(prompt.strip())

    # Definition-of-done self-report. When the lead handed down criteria,
    # ask the subagent to close with a structured <criteria_check> block:
    # one line per criterion → met/unmet + the evidence. This does double
    # duty — it forces the subagent to actually confront each criterion
    # before claiming done, and it gives the return-path verification judge
    # (agents/verify.py) concrete evidence to grade instead of free prose.
    # Co-located with the criteria so it covers every role (built-in +
    # user-defined) without editing each system prompt.
    if success_criteria and any(str(c).strip() for c in success_criteria):
        sections.append(
            "Before you finish, confirm each success criterion above is "
            "satisfied. End your final message with a `<criteria_check>` "
            "block — one line per criterion: the criterion, then `met` or "
            "`unmet`, then the concrete evidence (the file you wrote, the "
            "command you ran and its result). If a criterion is genuinely "
            "impossible or out of scope, mark it `unmet` and say why — never "
            "claim done on work you didn't do."
        )

    return "\n\n".join(sections)


def extract_workspace_facts(coder_state: Any) -> str:
    """Pull the cached <workspace_facts> block from a live CoderState.

    Returns ``""`` when the state has no kernel facts attached (legacy
    sessions, kernel disabled, fresh workspace). Duck-typed so this
    module doesn't depend on coder mode.
    """
    if coder_state is None:
        return ""
    facts = getattr(coder_state, "kernel_facts_text", None)
    if isinstance(facts, str) and facts.strip():
        return facts
    cache = getattr(coder_state, "_kernel_facts_cache", None)
    if isinstance(cache, str) and cache.strip():
        return cache
    return ""


def extract_orientation(coder_state: Any) -> str:
    """Pull the compact ``<orientation>`` anchor from a live CoderState.

    Returns ``""`` when the state has no orientation attached (legacy
    sessions, kernel disabled, fresh workspace). Handed to every
    context mode — including ``slim`` — so even a minimal-context role
    knows the session objective + project shape. Duck-typed so this
    module doesn't depend on coder mode.
    """
    if coder_state is None:
        return ""
    text = getattr(coder_state, "orientation_text", None)
    if isinstance(text, str) and text.strip():
        return text
    return ""


def extract_recent_tool_digests(coder_state: Any, *, limit: int = 8) -> list[str]:
    """Pull the most recent tool-call digests from a CoderState.

    Falls back to turn_summaries when no explicit digest log exists.
    Duck-typed; no coder mode import required.
    """
    if coder_state is None:
        return []
    log = getattr(coder_state, "recent_tool_digests", None)
    if isinstance(log, list):
        return [str(x) for x in log[-limit:] if x]
    summaries = getattr(coder_state, "turn_summaries", None)
    if isinstance(summaries, list):
        return [str(x) for x in summaries[-limit:] if x]
    return []
