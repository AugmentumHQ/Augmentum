"""LLM-agency advisor over the debt list — versatile recommendations, not a map.

The deterministic catalog (``debt.py``) is the SAFETY FLOOR: it decides what's
*mechanical* (auto-lane eligible) vs *structural*/red-tier, and that classification
must never be left to a model — you don't let an LLM decide a security finding is
safe to auto-fix. On TOP of that floor, this advisor uses model agency to do what
a fixed map cannot: read the actual flagged situation and recommend the **best
choices** — what to tackle first and why, how to approach each, what to group,
what to skip — versatile across whatever the audit surfaces (per the
no-switchboard principle: use the model, don't hardcode).

Safety by construction:
* the chat call is INJECTED (pure + testable);
* the model's picks are FILTERED against the real flagged ids — it can't invent a
  finding;
* ``kind`` (mechanical/structural) is carried from the deterministic triage, NOT
  the model — the advisor annotates + orders, it never reclassifies or widens
  autonomy.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.selfedit.debt import DebtTriage
from augmentum.selfedit.prompts import register_prompt
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Injected: a chat callable that takes a prompt and returns the model's text.
Chat = Callable[[str], Awaitable[str]]


@dataclass
class DebtRecommendation:
    target_id: str          # "scanner.metric" — MUST match a real flagged finding
    rank: int               # 1 = do first
    kind: str               # carried from the deterministic triage (not the model)
    title: str
    rationale: str          # the model's reasoning for the priority
    approach: str           # how to tackle it
    effort: str = "M"       # S | M | L (the model's estimate)
    group: str = ""         # optional cluster label

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id, "rank": self.rank, "kind": self.kind,
            "title": self.title, "rationale": self.rationale, "approach": self.approach,
            "effort": self.effort, "group": self.group,
        }


@dataclass
class DebtAdvice:
    summary: str = ""
    recommendations: list[DebtRecommendation] = field(default_factory=list)
    note: str = ""
    available: bool = True

    def to_dict(self) -> dict:
        return {
            "summary": self.summary, "note": self.note, "available": self.available,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


_PROMPT = """\
You are advising on which code-health findings to tackle. Below is what an audit \
flagged, already classified by a safety policy into:
  - auto-lane (mechanical): a fix the audit can confirm; safe to attempt automatically
  - needs-you (structural): taste/security/schema — a human decides; never auto-fixed

DO NOT reclassify anything. DO NOT invent findings. Only reference the ids listed.
Reason about the ACTUAL situation: severity, blast radius, effort, what unblocks
other work, what's worth your attention vs noise. Recommend the best order to work
through them and how to approach each.

FINDINGS:
{findings}

Return ONLY a JSON object:
{"summary": "<2-3 sentence read on the overall debt + strategy>",
  "recommendations": [
    {"id": "<scanner.metric from the list>", "rationale": "<why this priority>",
      "approach": "<how to tackle it>", "effort": "S|M|L", "group": "<optional cluster>"}
  ]}
Order the recommendations best-first. Include the ones genuinely worth doing \
(you may omit pure noise), up to {max_targets}."""

# Registered as an overridable prompt so Evolve can improve the advisor itself —
# the first, deliberately self-contained, low-blast-radius target.
register_prompt("debt_advisor", _PROMPT, label="Debt advisor",
                description="how the agent prioritizes the flagged debt list",
                user_facing=False)


def _render_findings(triage: DebtTriage) -> tuple[str, dict]:
    """The findings block for the prompt + an id→target index (the safety allow-list)."""
    index: dict = {}
    lines: list[str] = []
    for kind_label, targets in (("auto-lane (mechanical)", triage.mechanical),
                                ("needs-you (structural)", triage.structural)):
        for t in targets:
            tid = f"{t.scanner}.{t.metric}"
            index[tid] = t
            note = f" — {t.note}" if t.note else ""
            lines.append(f"- [{kind_label}] id={tid} | {t.title} (count {t.count}) | "
                         f"{t.objective}{note}")
    return "\n".join(lines), index


def _json_obj(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _fill(template: str, *, findings: str, max_targets: int) -> str:
    """Inject the findings into the prompt template via replace (not .format), so
    an evolved/override prompt with stray braces can't crash it. If an override
    dropped the {findings} slot, append the block so the model still sees them."""
    out = template.replace("{max_targets}", str(max_targets))
    if "{findings}" in out:
        return out.replace("{findings}", findings)
    return f"{out}\n\nFINDINGS:\n{findings}"


def _render_activation(activation: dict | None) -> str:
    """A compact 'regions the system has learned about' block — the verified
    skill-graph signal (top_regions: positive = a region where edits have shipped
    and stuck, negative = a region of repeated rollback/rejection). The model leans
    toward fixing in trusted regions and treats failure-prone regions with extra
    caution. Empty when the graph has no verdict evidence yet."""
    regions = [(a, w) for a, w in (activation or {}).get("top_regions", []) if abs(w) > 0.01]
    if not regions:
        return ""
    trusted = [f"{a} ({w:+.2f})" for a, w in regions if w > 0][:6]
    risky = [f"{a} ({w:+.2f})" for a, w in regions if w < 0][:6]
    parts = []
    if trusted:
        parts.append("verified-success regions (edits here tend to stick): "
                     + ", ".join(trusted))
    if risky:
        parts.append("repeated-failure regions (be cautious, prefer escalation): "
                     + ", ".join(risky))
    if not parts:
        return ""
    return ("\n\nWHAT THE SYSTEM HAS LEARNED WORKS (verified skill graph — lean "
            "toward trusted regions, escalate in failure-prone ones):\n- "
            + "\n- ".join(parts))


def _render_prefs(preferences: list[dict] | None) -> str:
    """A compact 'what the user tends to keep/revert' block — the learning-loop
    signal the model weighs when ordering (lean toward kept shapes, be cautious on
    reverted ones). Empty when there's no history yet."""
    rows = [p for p in (preferences or []) if (p.get("kept", 0) or p.get("reverted", 0))]
    if not rows:
        return ""
    lines = [f"- {p['shape']}: kept {p.get('kept', 0)}, reverted {p.get('reverted', 0)}"
             f"{' (trusted)' if p.get('trusted') else ''}" for p in rows[:12]]
    return ("\n\nWHAT THE USER TENDS TO KEEP/REVERT (lean toward kept shapes, be "
            "cautious on reverted ones):\n" + "\n".join(lines))


async def advise(triage: DebtTriage, *, chat: Chat, max_targets: int = 8,
                 prompt: str = "", preferences: list[dict] | None = None,
                 activation: dict | None = None) -> DebtAdvice:
    """Ask the model to recommend the best choices over the flagged debt. Pure
    (chat injected). The result is filtered to real flagged ids and the kind is
    carried from the deterministic triage — the model orders + reasons, never
    reclassifies. ``prompt`` overrides the template (an evolved advisor prompt);
    ``preferences`` is the learning-loop signal (kept/reverted per shape) and
    ``activation`` is the verified skill-graph signal (regions that ship vs roll
    back) the model leans on. On any failure returns ``available=False`` so the
    caller falls back to the deterministic list."""
    findings, index = _render_findings(triage)
    if not index:
        return DebtAdvice(summary="No actionable debt flagged.", available=True)
    template = prompt if (prompt and prompt.strip()) else _PROMPT
    context = findings + _render_prefs(preferences) + _render_activation(activation)
    try:
        raw = await chat(_fill(template, findings=context, max_targets=max_targets))
    except Exception as exc:  # noqa: BLE001 — advisor is best-effort; floor still stands
        log.warning("debt_advise_failed", error=repr(exc))
        return DebtAdvice(available=False, note=f"advisor unavailable: {exc!r}")

    obj = _json_obj(raw)
    recs: list[DebtRecommendation] = []
    for rank, item in enumerate(obj.get("recommendations") or [], start=1):
        tid = str(item.get("id", "")).strip()
        target = index.get(tid)
        if target is None:  # the model referenced something not flagged → drop it
            continue
        recs.append(DebtRecommendation(
            target_id=tid, rank=rank, kind=target.kind, title=target.title,
            rationale=str(item.get("rationale", "")).strip(),
            approach=str(item.get("approach", "")).strip(),
            effort=str(item.get("effort", "M")).strip().upper()[:1] or "M",
            group=str(item.get("group", "")).strip(),
        ))
        if len(recs) >= max_targets:
            break
    return DebtAdvice(summary=str(obj.get("summary", "")).strip(), recommendations=recs,
                      available=True)
