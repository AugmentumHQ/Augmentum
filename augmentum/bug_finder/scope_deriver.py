"""Make bug_finder purposeful on general input.

Field data (06-14 run bfr_f80deb251ce5): the user gave general input —
``mode=explore``, no threat model, no focus paths, ``severity_floor=info``
— and the detectors hunted with EMPTY threat framing across the whole
codebase. Anthropic's bug-finder research names a missing/mismatched
threat model as the #1 cause of valid-but-rejected findings: a detector
with no sense of what's valuable or where untrusted input enters flags
noise and misses the real bugs.

This module synthesizes a serviceable threat model from whatever signal
the run already has — the comprehension brief's structural map and the
detected stack — so general input becomes a *grounded* hunt instead of a
blind spray. It is prompt-framing only (the derived string is prepended
to detector + verifier + planner prompts exactly as a user-supplied
threat model would be); it changes no control flow and adds no tokens
beyond the slightly longer prompt prefix.

When even comprehension failed (skeleton-only / no brief), the deriver
still emits a minimal stack-grounded default rather than nothing — a
generic-but-purposeful frame always beats an empty one.
"""
from __future__ import annotations

_DEFAULT_IN_SCOPE = (
    "Correctness: off-by-one, unhandled None/error returns, incorrect "
    "control flow, state-machine gaps, idempotency bugs.",
    "Security & data isolation: missing input validation, injection, "
    "cross-user/tenant data leaks, auth bypass, unbounded queries, "
    "secrets exposure.",
    "Concurrency: shared-state mutation without synchronization, "
    "held resources across awaits, fire-and-forget tasks swallowing "
    "errors, races.",
    "Resource & error handling: leaks (file/socket/connection), "
    "swallowed exceptions, missing rollback on failure, unbounded "
    "growth.",
)

_DEFAULT_OUT_OF_SCOPE = (
    "Pure style / formatting / naming with no behavioral impact.",
    "Intentional design choices (sandboxed eval, debug-only endpoints) "
    "unless they cross a real trust boundary.",
    "Speculative bugs with no concrete trigger grounded in the code.",
)


def derive_threat_model(
    *,
    existing_threat_model: str = "",
    knowledge_brief: str = "",
    detected_language: str = "",
    user_goal_description: str = "",
) -> tuple[str, bool]:
    """Return ``(threat_model, derived)``.

    If the user supplied a threat model, it's returned verbatim with
    ``derived=False`` — never override an explicit one. Otherwise a
    grounded default is synthesized from the available signal and
    returned with ``derived=True``.

    The synthesized model is deliberately concise (a focused 1-page frame
    beats a 10-page checklist, per the bug-finder research) and points the
    detector back at the comprehension brief's structural map when one
    exists, so the hunt is anchored to THIS codebase rather than a generic
    taxonomy.
    """
    if existing_threat_model and existing_threat_model.strip():
        return existing_threat_model, False

    lines: list[str] = ["## Threat model (auto-derived — no explicit one was supplied)"]
    lines.append("")

    stack = (detected_language or "").strip()
    if stack:
        lines.append(
            f"This is a {stack} codebase. Reason about the failure modes "
            f"that stack actually exhibits — not a generic checklist."
        )
        lines.append("")

    goal = (user_goal_description or "").strip()
    if goal:
        lines.append(f"### Run focus\n{goal}")
        lines.append("")

    if knowledge_brief and knowledge_brief.strip():
        lines.append(
            "### Anchor to the structural map\n"
            "A comprehension brief for this workspace is available to you. "
            "Prioritize the risk surfaces, entry points, and pillars it "
            "identifies — that map IS this codebase's trust-boundary "
            "picture. Where untrusted input enters those surfaces is where "
            "the real bugs are."
        )
        lines.append("")

    lines.append("### In scope")
    lines.extend(f"- {item}" for item in _DEFAULT_IN_SCOPE)
    lines.append("")
    lines.append("### Out of scope")
    lines.extend(f"- {item}" for item in _DEFAULT_OUT_OF_SCOPE)
    lines.append("")
    lines.append(
        "Disproof discipline: treat each suspicion as a hypothesis to "
        "DISPROVE before reporting it. A grounded, triggerable bug beats "
        "ten speculations."
    )

    return "\n".join(lines), True


__all__ = ["derive_threat_model"]
