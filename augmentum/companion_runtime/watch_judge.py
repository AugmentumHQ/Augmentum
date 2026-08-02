"""Watch importance judge — does this detected change MATTER to the user?

The pipeline position is fixed (scheduled-requests spec §7.2-7.3):

    runner diff says "changed" ─┬─ no params.intent ──────→ deliver
                                └─ intent → judge(diff, intent)
                                      ├─ important      → deliver
                                      ├─ not important  → suppress (logged)
                                      └─ judge failed   → deliver (fail OPEN)

Hard rules:
  * Downgrade-only. The judge can silence a detected change, never
    invent one — noteworthy=False from the runner is never escalated.
  * Fail open. Timeout, parse failure, no model → deliver. A broken
    judge degrades to today's behavior, not to silence (the
    Gemini-silent-miss anti-pattern).
  * Evidence-bound. The verdict must quote the content it judged; the
    quote is substring-checked mechanically. A verdict whose evidence
    isn't in the source keeps its suppress/deliver decision ONLY in the
    deliver direction — an unverified "not important" cannot suppress
    (P7: LLM claims only count when they point at bytes that exist).

Visualping's platform stat — their AI marks 83% of detected changes
unimportant — is why this layer exists; local inference is why we can
afford it on every fire.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_PROMPT = """You judge whether a detected change matters to a user, given their stated intent.

User's intent for this watch: {intent}

What the watch detected:
{diff}

Reply with ONLY a JSON object, no other text:
{{"important": true/false, "reason": "<one short sentence>", "evidence": "<short verbatim quote from the detected content above>"}}"""


@dataclass(slots=True)
class JudgeVerdict:
    important: bool              # the delivery decision (post-rules)
    reason: str = ""
    evidence: str = ""
    evidence_verified: bool = False
    consulted: bool = False      # False = judge skipped/unavailable
    raw_important: bool | None = None  # model's own answer, pre-rules

    def as_dict(self) -> dict[str, Any]:
        return {
            "important": self.important,
            "reason": self.reason[:300],
            "evidence": self.evidence[:300],
            "evidence_verified": self.evidence_verified,
            "consulted": self.consulted,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def parse_verdict(raw_text: str, source_content: str) -> JudgeVerdict | None:
    """Parse + verify a judge reply. Pure (tested directly).

    Returns None when the reply isn't a usable verdict (caller fails
    open). Applies the evidence rule: an unverified verdict may only
    deliver, never suppress."""
    m = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "important" not in obj:
        return None

    raw_important = bool(obj.get("important"))
    evidence = str(obj.get("evidence") or "").strip()
    verified = bool(
        evidence and _normalize(evidence) in _normalize(source_content)
    )
    important = raw_important
    if not verified and not raw_important:
        # Unverified suppression → deliver. The judge claimed the change
        # doesn't match the intent while quoting something that isn't in
        # the content it was shown; that claim doesn't get to silence
        # the user's watch.
        important = True
    return JudgeVerdict(
        important=important,
        reason=str(obj.get("reason") or "").strip(),
        evidence=evidence,
        evidence_verified=verified,
        consulted=True,
        raw_important=raw_important,
    )


async def judge_change(
    runtime: Any,
    *,
    intent: str,
    diff_content: str,
) -> JudgeVerdict:
    """One short local completion. Every failure path returns an
    important=True verdict (fail open) with consulted=False so the run
    row records that the judge never actually weighed in."""
    from augmentum.config import settings

    fail_open = JudgeVerdict(important=True, consulted=False)
    if not bool(getattr(settings, "companion_watch_judge_enabled", True)):
        return fail_open
    intent = (intent or "").strip()
    diff_content = (diff_content or "").strip()[:1500]
    if not intent or not diff_content:
        return fail_open

    timeout_s = float(
        getattr(settings, "companion_watch_judge_timeout_s", 10.0) or 10.0,
    )
    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import (
            InternalChatRequest,
            Message,
            response_text,
        )

        backend, model_name = await tiers.primary(runtime)
        req = InternalChatRequest(
            model=model_name,
            messages=[Message(
                role="user",
                content=_PROMPT.format(intent=intent, diff=diff_content),
            )],
            stream=False,
            temperature=0.1,
            max_tokens=200,
        )
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
        verdict = parse_verdict(response_text(resp) or "", diff_content)
    except TimeoutError:
        log.warning("watch_judge_timeout", timeout_s=timeout_s)
        return fail_open
    except Exception:
        log.warning("watch_judge_failed", exc_info=True)
        return fail_open
    if verdict is None:
        log.warning("watch_judge_unparseable")
        return fail_open
    return verdict
