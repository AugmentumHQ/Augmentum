"""Read a comic/manga page directly with a vision LLM — no docling.

The docling path (:mod:`docling_client` + :mod:`reading_order` + :mod:`assembly`)
extracts *fragments with true geometry* and then has a text-only model repair
them. This module is the other end of the trade: hand the whole page image to a
vision-capable model and ask it for the spoken transcript in reading order.

What you gain: the model SEES the page, so bubble-tail attribution, panel
sequence and vertical RTL flow are judged from the art rather than from
band-row heuristics on (x,y) numbers. Manga in particular is where the
geometric sort is weakest — right-to-left panels with vertically-set text.

What you lose: **bounding boxes**. VLMs confabulate coordinates, so this path
returns ``bbox=None`` for every line rather than inventing numbers that would
send the pan-and-scan player to the wrong corner. The player already handles
``None`` by holding on the full page — that degradation is honest and visible.

Model selection goes through the **classifier role** in the normal registry
hierarchy (``override -> classifier_model -> sidecar -> utility_model ->
primary_chat_model``), so the user picks the transcription model from the model
manager like any other role. An earlier version of this module bypassed that and
bound to :class:`VisionRouter`'s classifier *provider object* — one hardcoded
container URL, one model, no selectability and no fallback. When that container
died the whole path died with it, which is what a URL pin buys you.

The role is resolved ONCE per chapter (see :func:`resolve_reader`) and the caller
records the result, so a run can't be transcribed by two different models
mid-chapter — the stability the old pin was reaching for, without freezing the
transport to get it.

This path does NOT fall back to a smaller captioner. Narrating a comic is an
explicit vision request: if the classifier role points at a model that can't
see, the user is told which knob to turn. Silently handing the job to a 500M CPU
captioner would return quietly-worse text and teach them nothing. (The implicit
paths — a drive-by image pasted into chat — keep their own fallback ladder in
:mod:`augmentum.vision`; that chain is deliberate and is not this module's
business.)

Output is plain numbered lines, not JSON. A small VL model spends its reliability
budget on either seeing or on grammar-constrained formatting — not both — and
one-line-per-bubble degrades gracefully where a truncated JSON object yields
nothing at all.
"""

from __future__ import annotations

import re

import httpx

from augmentum.utils.logging import get_logger
from augmentum.vision.provider import _caption_via_openai_endpoint

log = get_logger(__name__)

__all__ = [
    "read_page_script",
    "resolve_reader",
    "VisionNotConfigured",
    "build_prompt",
    "DEFAULT_MANGA_PROMPT",
    "split_speaker",
    "VOICE_CAST_SLOTS",
]


# The Voice Cast — five register buckets a line can be cast into. This is the
# v1 implementation of the "casting" concept: coarse and stateless (register
# judged from the art per line), with room to upgrade to per-character identity
# later without changing this contract. The narrator slot doubles as the
# fallback voice for any line the caster can't place. Order is display order.
VOICE_CAST_SLOTS: tuple[str, ...] = ("m_low", "m_high", "f_low", "f_high", "narrator")

# Leading register tag the reader/refiner prefixes each line with:
#   [ML] male low   [MH] male high   [FL] female low   [FH] female high   [N] narration
# The opening bracket is REQUIRED — matching a bare leading letter would eat the
# 'n' of "No, wait!" — but the closing bracket and casing are tolerated, since a
# small VL model drops the ']' or lowercases occasionally. Legacy bare [M]/[F]
# (from the 3-bucket version) map to the low register. Anything unmatched falls
# through to the narrator voice.
_SPEAKER_TAG = re.compile(r"^\s*\[\s*(FL|FH|ML|MH|N|F|M)\s*\]?\s*", re.IGNORECASE)
_SPEAKER_BY_TAG = {
    "ml": "m_low", "mh": "m_high", "fl": "f_low", "fh": "f_high", "n": "narrator",
    # Legacy 3-bucket tags → low register.
    "m": "m_low", "f": "f_low",
}


def split_speaker(text: str) -> tuple[str | None, str]:
    """``"[FH] Hello"`` -> ``("f_high", "Hello")``.

    Returns ``(None, text)`` when no tag is present, so the caller can tell an
    UNTAGGED line (leave its kind alone; read in the narrator voice) apart from
    an explicitly narrator-tagged ``[N]`` line (mark it narration). The narrator
    voice reads both without casting a register that isn't there.
    """
    text = (text or "").strip()
    m = _SPEAKER_TAG.match(text)
    if not m:
        return None, text
    speaker = _SPEAKER_BY_TAG.get(m.group(1).lower(), "narrator")
    return speaker, text[m.end():].strip()


class VisionNotConfigured(RuntimeError):
    """The classifier role can't do vision — the user has a knob to turn.

    Carries a message written FOR the user, not for the log: it names both
    remedies (the per-model vision toggle that's easy to forget at load time,
    and picking a vision-capable model for the role) because those are the two
    ways this actually happens in practice.
    """


# The panel/bubble sweep is the ONLY part that differs by reading direction,
# so it's the only part that varies. Everything below it — joining, coherence,
# what to include — is identical for manga and western comics.
# Reading order is ROW-MAJOR, not column-major, and that distinction is the
# whole game. "Right to left, then top to bottom" reads as "take the rightmost
# thing first" — so a low bubble on the right beats a high bubble on the left
# and two speakers swap turns. Height wins first; side only breaks ties within
# the same horizontal band. Stated explicitly because the failure it prevents
# produces fluent, plausible, wrongly-ordered dialogue rather than an obvious
# error.
_ORDER_RULES = {
    "rtl": """- Work in horizontal bands, from the TOP of the page to the BOTTOM.
- Panels: within a band of panels, right to left. Finish a whole panel before \
moving to the next one.
- Bubbles within a panel: HEIGHT COMES FIRST. Take the highest bubble first. \
Only when two bubbles sit at roughly the same height does the RIGHT one come \
first.
- A bubble that is clearly higher is read BEFORE a bubble that is further \
right but lower. Do not jump down the page to reach something on the right.
- Inside a single bubble: always top to bottom. Never read upward.
- This page is RIGHT-TO-LEFT. Within a row, the rightmost panel comes FIRST. \
If you find yourself starting at the top-left, you are reading it backwards.""",
    "ltr": """- Work in horizontal bands, from the TOP of the page to the BOTTOM.
- Panels: within a band of panels, left to right. Finish a whole panel before \
moving to the next one.
- Bubbles within a panel: HEIGHT COMES FIRST. Take the highest bubble first. \
Only when two bubbles sit at roughly the same height does the LEFT one come \
first.
- A bubble that is clearly higher is read BEFORE a bubble that is further \
left but lower. Do not jump down the page to reach something on the left.
- Inside a single bubble: always top to bottom. Never read upward.
- This page is LEFT-TO-RIGHT. Within a row, the leftmost panel comes FIRST.""",
}

_PROMPT_TEMPLATE = """You are transcribing one page of a comic into speech-ready English for a \
text-to-speech reading. Extract the text in the order it is spoken.

READING ORDER
{order_rules}

ONE BUBBLE = ONE LINE
- Output exactly one line per speech bubble.
- Text that wraps across several lines inside a bubble is ONE sentence. Join \
it, repairing any word split by the line break.
- Never split one bubble across two output lines. Never merge two bubbles into \
one line.

WHO IS SPEAKING — TAG EVERY LINE
- Begin every line with a speaker tag in square brackets, then the text. Judge \
BOTH the speaker's apparent gender AND their vocal register (pitch/age/build):
  [ML] male, low register — grown men, deep or heavy voices,
  [MH] male, high register — boys, young or light-voiced males,
  [FL] female, low register — grown women, mature or alto voices,
  [FH] female, high register — girls, young or bright-voiced females,
  [N]  narration boxes, captions, and any off-panel narrator voice.
- Decide from the ART: follow the bubble's tail to the character drawn \
speaking it, and read their apparent gender and age/build. Narration boxes \
(usually rectangular, tailless, at a panel edge) are always [N].
- Register matters most when two same-gender characters share a scene — split \
them by pitch so they don't sound identical. When unsure of register, pick low.
- If you genuinely cannot tell who is speaking, use [N]. Never leave a line \
untagged.
- The tag is the FIRST thing on the line: `[FL] I told you it wasn't safe.`

COHERENT ENGLISH — THIS IS THE PRIORITY
- Every line must read aloud as a grammatical English sentence.
- If a joined line comes out garbled, backwards, or starts mid-sentence, you \
read that bubble in the wrong order. Re-read it top to bottom and correct it.
- Sentence structure wins over literal fragment order.
- Write in normal sentence case, not ALL CAPS. Keep ? and !, but collapse \
decorative repetition (AAAAHHH -> Aah).

WHAT TO INCLUDE
- Only text inside bubble-like shapes: speech balloons, thought bubbles, and \
narration boxes.
- Skip sound effects drawn into the artwork, signs, labels, background text, \
page numbers, and credits.
- Transcribe only what is legibly printed on this page. Never invent, guess, or \
continue dialogue that is not there. Skip any bubble you cannot read.

OUTPUT
One tagged line per bubble, in reading order, and nothing else. No numbering, \
no speaker names, no panel headings, no commentary, no explanation. Every line \
starts with [ML], [MH], [FL], [FH], or [N]. If the page has no readable bubble \
text, output nothing at all."""


# Same prior the second pass used, worded for a cold read. The difference
# matters: pass 2 compares the list against a draft that demonstrably came from
# the page, so a bad suggestion has something to contradict it. Here there is no
# draft, so the "only if printed on this page" clause is the ONLY thing standing
# between a familiar name and artwork that never contained it. Hence the
# repetition of that constraint and the explicit textless-page reminder.
_GLOSSARY_TEMPLATE = """NAMES AND TERMS ALREADY ESTABLISHED IN THIS CHAPTER
These spellings are confirmed from earlier pages. When a word you are reading is \
hard to make out and is a near-miss for one of them, prefer the spelling below.
Use this list ONLY for spelling. It is not dialogue. Never output a term from \
this list unless you can actually read it printed on THIS page — if the page \
has no readable bubble text, output nothing at all, no matter what is listed here.
{terms}
"""


def build_prompt(reading_direction: str = "", glossary: list[str] | None = None) -> str:
    """The transcription prompt for a page read in ``reading_direction``.

    Getting this wrong is not cosmetic: an RTL prompt on a western comic makes
    the model sweep panels backwards, which reads as scrambled dialogue rather
    than as an obvious failure. The reader already knows the direction — it's
    a per-series user setting — so it should never be guessed here.

    ``glossary`` is the chapter's confirmed proper nouns. It rides along as a
    spelling prior so a cold read doesn't have to recover 'Gojo' from blurry
    lettering it has never seen spelled out. Costs a few dozen prompt tokens and
    no extra request, which is the whole reason it belongs on this pass too.
    """
    from augmentum.ocr.reading_order import (
        default_reading_direction,
        normalize_reading_direction,
    )

    # No silent "rtl" floor here any more. This used to disagree with the
    # entry point's "ltr" floor, so which way a page got read depended on which
    # of the two happened to fill in the blank.
    resolved = normalize_reading_direction(
        reading_direction, fallback=default_reading_direction(),
    )
    rules = _ORDER_RULES[resolved]
    base = _PROMPT_TEMPLATE.format(order_rules=rules)
    if not glossary:
        return base
    terms = "\n".join(f"- {t}" for t in glossary)
    # The list lands last, which is why its own block re-states the output
    # contract ("nothing unless printed", "textless page -> output nothing"):
    # ending a prompt on a bare list of names is an invitation to emit them.
    return f"{base}\n\n{_GLOSSARY_TEMPLATE.format(terms=terms)}"


# Back-compat / default for callers with no direction to hand (manga is the
# case the VLM path exists for).
DEFAULT_MANGA_PROMPT = build_prompt("rtl")

# Leading "1.", "1)", "- ", "* " the model adds despite being told not to.
_ENUM_PREFIX = re.compile(r"^\s*(?:[-*•]|\(?\d{1,3}[.)])\s+")
# Lines that are the model talking ABOUT the page instead of transcribing it.
# Every alternative must stay narrow enough that real dialogue survives: comics
# are full of "I can't believe it!", "Okay!", "There is no time!" — matching on
# a bare "i can"/"okay"/"there is no" silently deletes speech, which is a worse
# failure than letting one stray meta line through to the TTS.
#
# Every alternative below is ANCHORED at the start of the line, which is what
# keeps it safe: real dialogue that happens to mention an image ("So our image
# would also be very important") doesn't OPEN with "the provided image
# contains", and a character does not begin a sentence with "**Analyze the
# Image:**".
_META_LINE = re.compile(
    r"^\s*(?:"
    r"here(?:'s| is| are)? (?:the|a) (?:transcript|transcription|text|dialogue|lines)"
    # Subject phrase: the model describing the artwork instead of reading it.
    # The old version required a bare "the page|image|panel", so every refusal
    # of the form "The PROVIDED image contains no speech bubbles" walked
    # straight through and got read aloud by TTS. Observed on 6 pages of one
    # chapter — the single most jarring failure this feature has, because the
    # narrator audibly stops narrating and starts filing a report.
    # Split in two on purpose. A bare copula is NOT enough on its own: "The
    # panel is heavy, help me lift it!" is dialogue, and an earlier draft of
    # this pattern deleted it. So a copula only counts when the noun carries a
    # "provided/attached/given" qualifier — which no character ever says — and
    # without the qualifier the verb itself has to be descriptive.
    r"|(?:the |this |that )?"
    r"(?:provided|attached|given|uploaded|supplied|current) "
    r"(?:image|page|panel|picture|strip|crop|scan)s? "
    r"|(?:the |this |that )?"
    r"(?:image|page|panel|picture|strip|crop|scan)s? "
    r"(?:shows?|contains?|depicts?|doesn't|does not|do not"
    r"|appears?|consists?|seems?|has no|have no)"
    r"|i (?:cannot|can't|am unable to|was unable to) (?:read|see|make out|transcribe|discern)"
    r"|i (?:must|will|should|am going to) (?:output|return|provide) nothing"
    r"|there (?:is|are) (?:absolutely )?no (?:readable|legible|visible|english|bubble|speech|dialogue|text)"
    r"|no (?:readable|legible|visible|english|bubble|speech) text"
    r"|since there (?:is|are) no text"
    # Reasoning that arrived as ANSWER content rather than in reasoning_content.
    # `enable_thinking=False` is a chat-template kwarg and not every model
    # honors it; when it's ignored, the scratchpad lands in `content` and is
    # indistinguishable from a transcript to everything downstream. These are
    # the shapes it takes — markdown headers and step labels, neither of which
    # occurs in lettering.
    r"|\*\*"
    r"|(?:analy[sz]e|analysis|search for text|apply constraints|conclusion"
    r"|result|reasoning|step \d+)\s*[:.]"
    r"|the (?:user|instruction|instructions|prompt|task) (?:has|have|is|are|state|states|said|says|provided|asked|specify|specifies)"
    r"|the reading order"
    r"|\[no text"
    r"|(?:panel|page) \d+\s*[:.—-]"
    r"|note\s*:"
    r"|transcript(?:ion)?\s*:"
    r")",
    re.IGNORECASE,
)


def _parse_lines(raw: str) -> list[str]:
    """Plain model output → one spoken line per entry.

    Strips enumeration the model bolted on, drops meta/refusal chatter and
    fenced-block scaffolding, and collapses runs of blank lines. Everything
    surviving is treated as dialogue.

    Two artifacts of small VL models are removed here rather than left to the
    prompt, because a prompt instruction is a request and this is a guarantee:

    * **Repetition loops.** A page comes back as the same sentence four times.
      It survives every size-based sanity check (four lines is a plausible page)
      and is the single most audible failure there is — TTS reads it aloud four
      times. Exact repeats collapse to the first occurrence.
    * **Contentless lines.** ``!`` or ``...`` alone: nothing for TTS to say, and
      they inflate the line count that the acceptance guard trusts.

    The repeat collapse can in principle lose a genuinely repeated identical
    line (a character shouting the same word twice). That trade is deliberate:
    hearing a real line once is a small loss, hearing a phantom line four times
    is the failure people notice. Near-duplicates are left alone — only exact
    matches after normalization collapse, so real dialogue that merely echoes
    survives intact.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        text = line.strip()
        if not text or text.startswith("```"):
            continue
        text = _ENUM_PREFIX.sub("", text).strip()
        # Bare quoting is common ("HELLO" -> HELLO); balanced only.
        if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        if not text or _META_LINE.match(text):
            continue
        # Needs at least one letter or digit to be worth speaking.
        if not any(ch.isalnum() for ch in text):
            continue
        key = "".join(ch for ch in text.lower() if ch.isalnum())
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


async def resolve_reader(app, *, role: str = "classifier", override: str | None = None) -> tuple[str, str]:
    """Resolve ``role`` to ``(base_url, model)`` for page reading.

    Raises :class:`VisionNotConfigured` with a user-facing message when the
    role resolves to something that can't read an image. Callers run this
    ONCE, up front — before any synthesis work — so the user finds out in
    seconds rather than after a chapter of empty pages.

    ``role`` is a parameter because the two-pass read deliberately uses two
    different ones: the cheap classifier does the literal transcription, the
    primary does the contextual repair. Both are vision calls on the same image,
    so both go through this same capability check — a primary that can't see is
    exactly as fatal as a classifier that can't, and finding out per-page would
    mean discovering it thirty pages in.

    ``override`` of ``None`` means "use the OCR model override setting"; pass an
    explicit ``""`` to follow the role with no override at all.
    """
    from augmentum.config import settings

    registry = (getattr(app.state, "provider_registry", None)
                or getattr(app.state, "registry", None))
    if registry is None:
        raise VisionNotConfigured("Model registry unavailable — is the server still starting?")

    if override is None:
        override = getattr(settings, "ocr_vlm_model", "") or ""

    backend, model = await registry.resolve_model_for_role(
        role,
        override=override,
        settings=settings,
    )
    if backend is None or not model:
        raise VisionNotConfigured(
            f"No model is assigned to the {role} role. Pick a vision-capable "
            "model for it in the model manager.",
        )

    base_url = (getattr(backend, "base_url", "")
                or getattr(backend, "_base_url", "") or "")
    if not base_url:
        raise VisionNotConfigured(
            f"The {role} role resolves to '{model}', which doesn't expose an "
            "OpenAI-compatible endpoint for image input. Assign a locally served "
            f"vision model to the {role} role.",
        )

    # Capability comes from the backend's own /v1/models claim (mmproj paired →
    # supports_vision), which is the same signal every other vision consumer
    # keys off. The name heuristic alone misses it: 'gemma-4-E2B-it-qat-GGUF'
    # carries no 'VL' token but sees fine with its projector loaded.
    vision = None
    try:
        for info in await backend.list_models():
            if getattr(info, "name", "") == model:
                vision = bool(getattr(info, "vision", False))
                break
    except Exception as exc:  # noqa: BLE001 — unreachable backend is its own error
        raise VisionNotConfigured(
            f"Couldn't reach the backend serving '{model}' to check for vision "
            f"support ({str(exc)[:120]}). Is that service running?",
        ) from exc

    if vision is None:
        raise VisionNotConfigured(
            f"'{model}' isn't currently loaded on its backend, so its vision "
            "support can't be confirmed. Load it, then try again.",
        )
    if not vision:
        raise VisionNotConfigured(
            f"'{model}' is assigned to the {role} role but reports no vision "
            "support. Either enable vision for it when loading the model (its "
            "projector/mmproj must be loaded too), or assign a vision-capable "
            f"model to the {role} role in the model manager.",
        )

    log.info("ocr_vlm_resolved", role=role, model=model, base_url=base_url)
    return base_url, model


async def read_page_script(
    app,
    image_bytes: bytes,
    *,
    prompt: str = "",
    reading_direction: str = "",
    glossary: list[str] | None = None,
    reader: tuple[str, str] | None = None,
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
) -> list[dict]:
    """One page image → ``[{order, kind, text, bbox}]`` via a vision LLM.

    ``bbox`` is always ``None`` — see the module docstring. ``kind`` is always
    ``"speech"``: the prompt deliberately asks only for bubble text, so there is
    no narration/sfx distinction to make, and guessing one would be noise.

    ``reader`` is the ``(base_url, model)`` from :func:`resolve_reader`, resolved
    once for the whole chapter. Passing it per page is what keeps one run on one
    model without freezing the transport.

    Returns ``[]`` only for a page the model read as textless (splash art) — a
    legitimate empty result. Transport failure raises, so "the endpoint is down"
    can never be mistaken for "this page has no dialogue".
    """
    if reader is None:
        reader = await resolve_reader(app)
    base_url, model = reader

    # An admin override wins outright — including over the glossary, since a
    # hand-written prompt is the one case where appending our own blocks would
    # fight whatever the author was trying to test.
    prompt = (prompt or "").strip() or build_prompt(reading_direction, glossary)

    # The shared helper does the image transcode + OpenAI vision request that
    # every llama-server-backed captioner already uses. It reports failure by
    # returning '' (its other callers are best-effort and want that), which on
    # this path is indistinguishable from a splash page — so an empty read is
    # disambiguated below rather than by changing that shared contract.
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        meta: dict = {}
        raw = await _caption_via_openai_endpoint(
            http,
            base_url,
            image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            model=model,
            # Transcription is not a reasoning task, and on a reasoning model
            # the thinking block is charged to the SAME max_tokens budget as
            # the answer. A dense page can spend the entire budget thinking and
            # return finish_reason='length' with empty content — which this
            # pipeline then records as "no text on this page". Measured on
            # Gemma-4-E4B: 824 completion tokens with thinking, 32 without,
            # for the same page. Off is both correct and ~25x cheaper here.
            enable_thinking=False,
            out_meta=meta,
        )

        # Truncation is NOT an empty page. If the budget ran out before any
        # content arrived, retry once with room rather than let a full page be
        # narrated as silent art. Bounded at one doubling: this should be
        # unreachable with thinking off, and a page that still can't answer in
        # 2x the budget is a real failure, not something to keep paying for.
        if not (raw or "").strip() and meta.get("finish_reason") == "length":
            log.warning(
                "ocr_vlm_truncated_retry",
                model=model,
                max_tokens=max_tokens,
                completion_tokens=meta.get("completion_tokens"),
                reasoning_chars=meta.get("reasoning_chars"),
            )
            raw = await _caption_via_openai_endpoint(
                http,
                base_url,
                image_bytes,
                prompt=prompt,
                max_tokens=max_tokens * 2,
                timeout_s=timeout_s,
                model=model,
                enable_thinking=False,
                out_meta=meta,
            )

        if not (raw or "").strip():
            # Empty means either "textless page" or "the endpoint just failed".
            # One cheap probe tells them apart; guessing wrong is how a dead
            # service turns into a 300-page comic that reports "no text".
            root = base_url.rstrip("/")
            root = root[:-3].rstrip("/") if root.endswith("/v1") else root
            try:
                resp = await http.get(f"{root}/v1/models", timeout=10.0)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001 — surfaced, not swallowed
                raise VisionNotConfigured(
                    f"The service serving '{model}' stopped responding "
                    f"({str(exc)[:120]}). Narration stopped rather than "
                    "transcribing the rest of the comic as blank pages.",
                ) from exc

    texts = _parse_lines(raw)
    if (raw or "").strip() and not texts:
        # The model read something and NOTHING survived the parse. That is a
        # third outcome, distinct from both a truly textless page and a dead
        # endpoint, and until now it looked identical to the first: the page
        # went by silent, with no error, while the log said only out_lines=0.
        #
        # It is either the model refusing/narrating instead of transcribing
        # (correctly dropped by _META_LINE) or real dialogue being eaten by the
        # filters. Those need opposite fixes, and they are indistinguishable
        # without the text, so the text goes in the log — in full, because it is
        # by definition short and a truncated sample is exactly the part that
        # would be missing when it matters.
        log.warning(
            "ocr_vlm_read_all_filtered",
            model=model,
            direction=reading_direction,
            chars=len(raw),
            raw=raw.strip(),
        )
    log.info(
        "ocr_vlm_read",
        model=model,
        direction=reading_direction,
        glossary_terms=len(glossary or []),
        chars=len(raw or ""),
        out_lines=len(texts),
        finish=meta.get("finish_reason") or "",
        # 0 here does NOT mean the model didn't think — it means no reasoning
        # arrived in `reasoning_content`. Combined with a content blob full of
        # "**Analyze the Image:**", 0 is the signal that `enable_thinking=False`
        # was ignored and the scratchpad came back as the answer.
        reasoning_chars=meta.get("reasoning_chars") or 0,
    )
    return [
        {"order": i, "kind": "speech", "text": t, "bbox": None}
        for i, t in enumerate(texts)
    ]
