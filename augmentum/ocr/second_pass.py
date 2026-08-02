"""Second read of the same page — image again, plus what the chapter knows.

Pass 1 (:mod:`vlm_reader`) reads a page cold: no idea who the characters are,
no idea what sentence was running when the page turned. It resolves ambiguous
lettering locally and gets it *nearly* right — the residual failure is
mis-transcription, not blindness. The model is reading the words.

This pass hands the SAME image back to a model together with pass 1's draft and
the chapter glossary, and asks it to correct the draft against the page. Two
things make it strictly more informed than pass 1 was:

* **A prior.** The glossary supplies proper nouns the chapter has already
  established, so ``GOJP`` resolves to ``GOJO`` because GOJO is a known word
  and GOJP is not. No amount of prompt-tuning on an isolated page can supply
  that, because the information genuinely isn't on the page.
* **A second sample.** Even with identical inputs a second look is another draw
  from a noisy process; with a draft to react to, the model spends its budget
  checking rather than transcribing from scratch.

**The image is not optional here, and that is the whole point.** A text-only
repair pass is free to invent whatever makes the draft read smoothly, and it
will — fluent and wrong is harder to catch than garbled and obviously wrong.
Keeping the page in front of the model means every proposed correction still
has to be consistent with what is actually printed. Context proposes, pixels
veto.

Model selection is by ROLE, and the two passes use different ones: classifier
for the literal read, primary for the repair. The cheap model is adequate at
"what glyphs are these" and the strong one earns its cost at "which reading is
coherent". Pointing both roles at one model is a valid configuration (and the
first thing to test), not a special case.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import httpx

from augmentum.ocr.vlm_reader import _parse_lines, build_prompt
from augmentum.utils.logging import get_logger
from augmentum.vision.provider import _caption_via_openai_endpoint

log = get_logger(__name__)

__all__ = ["refine_page_script", "build_refine_prompt", "REFINE_PROMPT"]


REFINE_PROMPT = """You are proof-reading a draft transcript of ONE comic page against the page \
itself. Another model already transcribed it; your job is to correct that draft, not to \
start over.

{order_rules}

THE DRAFT MAY BE WRONG IN THESE WAYS
- Mis-read letters and words (the usual case: the shapes were ambiguous).
- Lines in the wrong order, so the dialogue does not follow.
- A bubble read twice, or two bubbles merged into one line.
- A bubble missed entirely.

HOW TO CORRECT IT
- Look at the page. For each draft line, find the bubble it came from and check it.
- Fix misread words so the line matches what is printed AND reads as English.
- Reorder lines if the reading order was wrong.
- Delete a line that duplicates another. Add a line only if you can SEE a bubble the \
draft missed.
- Keep every line that is correct exactly as it is. Most of the draft is usually right.

SPEAKER TAGS
- Each draft line begins with a register tag: [ML] male low, [MH] male high, \
[FL] female low, [FH] female high, [N] narration box / off-panel narrator.
- Keep the tag on every line. Fix it only when the art clearly shows the draft \
cast the wrong speaker (wrong gender) or an obviously wrong register.
- A line you ADD must also start with a tag; a bubble you cannot attribute is [N].

{glossary_block}
THE LIMIT ON CORRECTION
- The page is the authority. Never change a line into something that is not printed on \
this page, however much better it would read.
- Never continue the story, invent dialogue, or fill in a bubble you cannot make out.
- If a draft line looks wrong but you cannot read the bubble well enough to fix it, \
leave the draft line alone.

OUTPUT
The corrected transcript: one tagged line per bubble, in reading order, and nothing \
else. Every line starts with [ML], [MH], [FL], [FH], or [N]. No numbering, no speaker \
names, no commentary, no notes about what you changed. If the page genuinely has no \
readable bubble text, output nothing at all.

DRAFT TRANSCRIPT
{draft}"""

_GLOSSARY_TEMPLATE = """NAMES AND TERMS ALREADY ESTABLISHED IN THIS CHAPTER
These spellings are confirmed from other pages. If a draft word is a near-miss for one \
of them, it is almost certainly that word misread — fix it. Use this list ONLY for \
spelling; it is not dialogue and nothing from it should appear in your output unless it \
is printed on this page.
{terms}
"""


def build_refine_prompt(draft: list[str], *, reading_direction: str, glossary: list[str]) -> str:
    """The proof-reading prompt for one page.

    The draft is numbered purely so the model can hold its place while checking;
    the numbering is stripped back out of the response by
    :func:`~augmentum.ocr.vlm_reader._parse_lines`, which already handles the
    enumeration models add unbidden.
    """
    from augmentum.ocr.reading_order import (
        default_reading_direction,
        normalize_reading_direction,
    )
    from augmentum.ocr.vlm_reader import _ORDER_RULES

    # Same seed as pass 1 — a proof-reader told to sweep the other way would
    # "correct" a right read into a wrong one.
    rules = _ORDER_RULES[normalize_reading_direction(
        reading_direction, fallback=default_reading_direction(),
    )]
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(draft))
    block = ""
    if glossary:
        block = _GLOSSARY_TEMPLATE.format(terms="\n".join(f"- {t}" for t in glossary))
    return REFINE_PROMPT.format(
        order_rules=f"READING ORDER\n{rules}",
        glossary_block=block,
        draft=numbered,
    )


def _accept(draft: list[str], refined: list[str], *, max_drift: float) -> tuple[bool, str]:
    """Should the refined transcript replace the draft?

    A proof-reading pass that returns something wildly different in SIZE has
    stopped proof-reading — it has either summarized the page, started
    describing it, or hallucinated a conversation. Line count is a crude proxy
    for that, but it is the one signal available without a second opinion, and
    the failure it catches is the one that would silently poison both the audio
    and the glossary that later pages depend on.

    Rejecting means keeping pass 1's output, which is a known-usable result.
    The bar is therefore "is this plausibly the same page", not "is this
    better" — we cannot measure better, so we only guard against obvious harm.
    """
    if not refined:
        # Empty from a non-empty draft is the classic refusal/timeout shape.
        # Trusting it would narrate a page with dialogue as silent art.
        return (False, "empty") if draft else (True, "both_empty")
    if not draft:
        # Pass 1 found nothing and pass 2 found text. Possible (a faint page the
        # stronger model could read), but it had no draft to check against, so
        # it was transcribing blind — exactly the mode we do not trust here.
        return False, "draft_empty"
    # Asymmetric on purpose. The two directions are not equally suspicious:
    #
    # * FEWER lines is the dangerous direction — a proof-reader that gives up on
    #   bubbles it can't re-verify silently deletes dialogue, and the result
    #   still looks like a clean page. Held to ``max_drift``.
    # * MORE lines means it found bubbles the draft lacked, which is the recall
    #   we want. It is also the normal case after a pass-1 repetition loop
    #   collapses to a single line: a symmetric cap would reject the correct
    #   4-line reading for drifting +300% from a 1-line draft, keeping the
    #   broken output precisely when the fix arrived.
    #
    # Additions are still bounded — an order-of-magnitude explosion is a
    # hallucinated conversation, not a dense page.
    if len(refined) < len(draft):
        drift = (len(draft) - len(refined)) / max(1, len(draft))
        if drift > max_drift:
            return False, f"lost_{drift:.2f}"
    elif len(refined) > max(8, len(draft) * 6):
        return False, f"exploded_{len(refined)}"
    return True, "ok"


async def refine_page_script(
    image_bytes: bytes,
    lines: list[dict],
    *,
    reader: tuple[str, str],
    reading_direction: str = "",
    glossary: list[str] | None = None,
    prompt: str = "",
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
    max_drift: float = 0.5,
    rescue_empty: bool = True,
) -> list[dict]:
    """Correct ``lines`` against ``image_bytes``; returns ``lines`` if unsure.

    Never raises on a bad or missing second opinion — pass 1's output is always
    a usable answer, so every failure here degrades to it rather than failing
    the page. That is deliberate: this pass is an improvement, and an
    improvement that can take down a chapter is a bad trade.

    ``bbox`` is carried over positionally where the line count is unchanged, and
    dropped otherwise. Reordered or edited lines can no longer be matched to the
    boxes pass 1 produced, and a confidently WRONG box sends the pan-and-scan
    player to the wrong corner of the page — worse than no box, which the player
    already handles by holding on the full page.
    """
    draft = [(ln.get("text") or "").strip() for ln in lines or []]
    draft = [t for t in draft if t]
    base_url, model = reader

    if not draft:
        # Pass 1 saw nothing. That is either a genuinely textless page (splash
        # art, very common in manga) or a page the cheap model simply missed —
        # and pass 1 must NOT be the final authority on which, because it is
        # the component that just failed. Left alone, a missed page is recorded
        # as silent art and is indistinguishable from the real thing.
        #
        # So the second model reads it from scratch. This is not the blind
        # confabulation the proof-reading path guards against: there is no draft
        # to anchor to and no story context in play, so it is exactly the plain
        # single-pass read, just performed by the stronger model.
        #
        # The glossary is deliberately withheld here. Everywhere else it is
        # checked against a draft that came from the page; on a blind read there
        # is nothing to contradict it, which is precisely the setup where a
        # familiar name gets "found" in artwork that never contained it.
        if not rescue_empty:
            return lines or []
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as http:
                raw = await _caption_via_openai_endpoint(
                    http, base_url, image_bytes,
                    prompt=build_prompt(reading_direction), max_tokens=max_tokens,
                    timeout_s=timeout_s, model=model,
                    # See read_page_script: thinking is charged to the same
                    # max_tokens budget as the answer, so on a dense page the
                    # response comes back empty. A rescue read that truncates
                    # itself into silence defeats the entire point of rescuing.
                    enable_thinking=False,
                )
        except Exception as exc:  # noqa: BLE001 — a missed rescue is still silent art
            log.warning("ocr_rescue_failed", model=model, error=str(exc)[:200])
            return lines or []

        rescued = _parse_lines(raw or "")
        log.info("ocr_rescue", model=model, direction=reading_direction, out_lines=len(rescued))
        if not rescued:
            return lines or []
        return [
            {"order": i, "kind": "speech", "text": t, "bbox": None}
            for i, t in enumerate(rescued)
        ]
    instructions = (prompt or "").strip() or build_refine_prompt(
        draft, reading_direction=reading_direction, glossary=glossary or [],
    )
    if prompt:
        # An admin override supplies its own wording but still needs the draft
        # and glossary appended, or it is proof-reading nothing.
        instructions = (
            f"{instructions}\n\n"
            + build_refine_prompt(draft, reading_direction=reading_direction,
                                  glossary=glossary or [])
        )

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as http:
            raw = await _caption_via_openai_endpoint(
                http, base_url, image_bytes,
                prompt=instructions, max_tokens=max_tokens,
                timeout_s=timeout_s, model=model,
                # Same budget trap as pass 1 — and worse here, because an empty
                # refined transcript is rejected as 'empty' and silently keeps
                # the draft, so the truncation never even surfaces as a failure.
                enable_thinking=False,
            )
    except Exception as exc:  # noqa: BLE001 — degrade to pass 1, never fail the page
        log.warning("ocr_refine_failed", model=model, error=str(exc)[:200])
        return lines

    refined = _parse_lines(raw or "")
    ok, why = _accept(draft, refined, max_drift=max_drift)
    if not ok:
        log.info(
            "ocr_refine_rejected",
            model=model, reason=why, draft_lines=len(draft), refined_lines=len(refined),
        )
        return lines

    # Separate the two things this pass can do, because they fail for opposite
    # reasons and the fixes are opposite too. `changed` is repair — it can only
    # improve lines the draft already had. `added` is RECALL, and it is the
    # number to watch: a pass 2 that never adds is anchored on the draft and
    # needs a re-sweep instruction, whereas one that adds but still misses
    # bubbles is hitting the same resolution ceiling pass 1 did, which no
    # prompt can lift. Without this split, "still missing things" is
    # undiagnosable from the logs.
    changed = sum(1 for a, b in zip(draft, refined) if a != b)
    log.info(
        "ocr_refine",
        model=model, direction=reading_direction, glossary_terms=len(glossary or []),
        draft_lines=len(draft), out_lines=len(refined), changed=changed,
        added=max(0, len(refined) - len(draft)), removed=max(0, len(draft) - len(refined)),
    )

    # Deletions are the suspicious direction. Adding a line means it found a
    # bubble; dropping several usually means it failed to re-verify them and
    # quietly gave up on them — which shows up to the user as "still missing
    # things" and is indistinguishable from a pass 1 miss unless the dropped
    # text is named here. Compared case/punctuation-insensitively so a mere
    # repair of a line isn't miscounted as a deletion plus an addition.
    # Matched by SIMILARITY, not equality. Exact matching is useless here: this
    # pass exists to rewrite lines, so a well-repaired line ("Doncha ever go
    # time in the morning?" -> the real sentence) shares almost no characters
    # with its draft and would be reported as a deletion. That would make a
    # working repair look like data loss — the exact wrong conclusion, and one
    # that would send the next fix at the wrong stage.
    def _key(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    remaining = [_key(t) for t in refined]
    dropped: list[str] = []
    for text in draft:
        k = _key(text)
        best, best_at = 0.0, -1
        for i, cand in enumerate(remaining):
            if cand is None:
                continue
            ratio = SequenceMatcher(None, k, cand).ratio()
            if ratio > best:
                best, best_at = ratio, i
        # 0.55 admits a substantial rewrite as "the same line, repaired" while
        # still catching a line that simply vanished.
        if best >= 0.55 and best_at >= 0:
            remaining[best_at] = None  # consumed — one draft line per output line
        else:
            dropped.append(text)
    if dropped:
        log.info("ocr_refine_dropped", model=model, count=len(dropped), texts=dropped[:8])

    keep_bbox = len(refined) == len(lines or [])
    return [
        {
            "order": i,
            "kind": (lines[i].get("kind", "speech") if keep_bbox else "speech"),
            "text": t,
            "bbox": (lines[i].get("bbox") if keep_bbox else None),
        }
        for i, t in enumerate(refined)
    ]
