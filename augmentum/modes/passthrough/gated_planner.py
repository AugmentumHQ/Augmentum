"""Brief → structure planner for gated artifact tools.

The structured creators (``create_ebook``, ``create_presentation``,
``create_document``) need ``title`` + a list of sections — they can't ride the
single-string ``[[tool:NAME]] brief`` marker. This planner is the missing
layer: one structured LLM call expands a one-line brief into the tool's full
input, so the confirmation chip can show the OUTLINE and Accept runs the real
tool with the plan.

Design: ONE call returns the whole draft (a short ebook/deck/doc is well within
a single generation). The offer surfaces the outline (titles); the tool's own
progress UI handles assembly + illustration. Returns ``None`` on any failure —
the caller falls back to a plain brief-confirm rather than erroring.

See docs/superpowers/specs/2026-06-02-offer-substrate-design.md and the gated
capabilities in orchestrator.py.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_MAX_SECTIONS = 12


@dataclass(frozen=True)
class PlanSpec:
    instruction: str
    schema_hint: str
    required: tuple[str, ...]
    list_key: str                      # the field holding the sections list
    section_required: tuple[str, ...]  # required keys within each section
    summarize: Callable[[dict], str]   # outline line for the confirm chip


_PLAN_SPECS: dict[str, PlanSpec] = {
    "create_ebook": PlanSpec(
        instruction=(
            "Write a short, complete illustrated ebook from the brief. 3-6 "
            "chapters. Each chapter needs a heading, 2-4 short paragraphs of "
            "real body text (not a summary), and a one-line illustration "
            "caption describing the picture for that chapter."
        ),
        schema_hint=(
            '{"title": "...", "author": "...", "chapters": '
            '[{"heading": "...", "body": "...", "illustration": "..."}]}'
        ),
        required=("title", "chapters"),
        list_key="chapters",
        section_required=("heading", "body"),
        summarize=lambda d: (
            f'"{d.get("title", "Untitled")}" — '
            f'{len(d.get("chapters") or [])} chapters: '
            + ", ".join(
                str(c.get("heading", "")) for c in (d.get("chapters") or [])[:5]
            )
        ),
    ),
    "create_presentation": PlanSpec(
        instruction=(
            "Build a concise slide deck from the brief. 4-8 slides. Each slide "
            "needs a title and 2-4 bullet points of real content."
        ),
        schema_hint=(
            '{"title": "...", "slides": [{"title": "...", '
            '"bullets": ["...", "..."]}]}'
        ),
        required=("title", "slides"),
        list_key="slides",
        section_required=("title",),
        summarize=lambda d: (
            f'"{d.get("title", "Untitled")}" — '
            f'{len(d.get("slides") or [])} slides: '
            + ", ".join(
                str(s.get("title", "")) for s in (d.get("slides") or [])[:6]
            )
        ),
    ),
    "create_document": PlanSpec(
        instruction=(
            "Write a structured document from the brief. 3-6 sections. Each "
            "section needs a heading and real body text (markdown allowed)."
        ),
        schema_hint=(
            '{"title": "...", "sections": [{"heading": "...", "body": "..."}]}'
        ),
        required=("title", "sections"),
        list_key="sections",
        section_required=("heading", "body"),
        summarize=lambda d: (
            f'"{d.get("title", "Untitled")}" — '
            f'{len(d.get("sections") or [])} sections: '
            + ", ".join(
                str(s.get("heading", "")) for s in (d.get("sections") or [])[:6]
            )
        ),
    ),
}


def is_planned_tool(tool_name: str) -> bool:
    return tool_name in _PLAN_SPECS


def outline_summary(tool_name: str, structured: dict) -> str:
    spec = _PLAN_SPECS.get(tool_name)
    if spec is None:
        return ""
    try:
        return spec.summarize(structured)[:240]
    except Exception:
        return ""


def _coerce(structured: dict, spec: PlanSpec) -> dict | None:
    """Validate + trim the planner output against the tool's contract."""
    if not isinstance(structured, dict):
        return None
    if any(k not in structured for k in spec.required):
        return None
    sections = structured.get(spec.list_key)
    if not isinstance(sections, list) or not sections:
        return None
    clean: list[dict] = []
    for sec in sections[:_MAX_SECTIONS]:
        if not isinstance(sec, dict):
            continue
        if any(not str(sec.get(k, "")).strip() for k in spec.section_required):
            continue
        clean.append(sec)
    if not clean:
        return None
    structured[spec.list_key] = clean
    return structured


async def expand_brief(
    tool_name: str, brief: str, backend: ModelBackend, model: str = "",
) -> dict | None:
    """One LLM call: brief → the tool's structured input. None on any failure."""
    spec = _PLAN_SPECS.get(tool_name)
    if spec is None or not brief.strip() or backend is None:
        return None
    from augmentum.models.base import InternalChatRequest, Message

    prompt = (
        f"{spec.instruction}\n\nBrief: {brief.strip()}\n\n"
        f"Return ONLY valid JSON in exactly this shape, nothing else:\n"
        f"{spec.schema_hint}"
    )
    try:
        resp = await backend.chat(InternalChatRequest(
            model=model or "",
            messages=[Message(role="user", content=prompt)],
            max_tokens=2400,
        ))
    except Exception as exc:
        log.warning("gated_plan_llm_failed", tool=tool_name, error=str(exc)[:200])
        return None

    raw = (resp.message.content if resp.message else "") or ""
    raw = _FENCE_RE.sub("", raw.strip())
    # Tolerate prose around the JSON object.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        log.warning("gated_plan_no_json", tool=tool_name)
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        log.warning("gated_plan_bad_json", tool=tool_name)
        return None
    return _coerce(parsed, spec)
