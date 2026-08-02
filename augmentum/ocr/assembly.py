"""LLM script-assembly pass — the keystone that makes rough OCR meaningless.

Validated 2026-06-17: handing geometry-ordered OCR fragments (text + position)
to a model Augmentum already runs — with a fixed "clean + merge + reorder by
meaning + tag" prompt — turned ``ANDWETHOUGHTTHERE'DBE PEACE`` into
``AND WE THOUGHT THERE'D BE PEACE`` and assembled a coherent page script from
24 garbage fragments. The small classifier (Gemma-E2B) does ~90%; the primary
model nails it. This collapses BOTH OCR caveats (rough lettering + panel
intermix) into one cheap pass, schema-constrained so the model can't ramble.

Resolution goes through ``resolve_model_for_role`` so a configured classifier
sidecar shields the pass from whatever heavy chat model is selected; set
``ocr_assembly_role='primary'`` (or an explicit ``ocr_assembly_model``) for
maximum fidelity on a faithful comic narration.
"""

from __future__ import annotations

import asyncio
import json

from augmentum.models.base import InternalChatRequest, Message
from augmentum.ocr.docling_client import Region
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Object-rooted schema (array-rooted grammars are rejected by some backends).
ASSEMBLY_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["narration", "speech", "sfx"]},
                    "text": {"type": "string"},
                    "src": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["order", "kind", "text", "src"],
            },
        }
    },
    "required": ["lines"],
}

_SYSTEM = (
    "You assemble a clean spoken script from rough OCR fragments of one comic "
    "page. The fragments arrive in approximate reading order, each with an "
    "(x,y) position (0..1, origin top-left). Your job: "
    "(1) fix OCR spacing and letter errors (e.g. YHE->THE, "
    "THECOPPERWIRES->THE COPPER WIRES, DONIMISS->DON'T MISS); "
    "(2) MERGE fragments that form one sentence; "
    "(3) fix local order where a sentence was split across rows, using meaning; "
    "(4) tag each line narration | speech | sfx; "
    "(5) for each output line, list in 'src' the fragment NUMBERS you merged "
    "into it (1-based, from the input list) so it can be located on the page. "
    "Stay faithful — do NOT invent words or dialogue. Output only the JSON object."
)


def _user_prompt(regions: list[Region]) -> str:
    lines = []
    for i, r in enumerate(regions):
        lines.append(f"{i + 1}. (x={r.cx:.2f},y={r.cy:.2f}) {r.text}")
    return "OCR fragments:\n" + "\n".join(lines)


def _parse(content: str) -> list[dict]:
    content = (content or "").strip()
    if not content:
        return []
    # Tolerate fenced output / leading prose — grab the first {...} object.
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return []
    lines = obj.get("lines") if isinstance(obj, dict) else None
    if not isinstance(lines, list):
        return []
    out = []
    for i, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        kind = (ln.get("kind") or "speech").lower()
        if kind not in ("narration", "speech", "sfx"):
            kind = "speech"
        src = ln.get("src")
        src_ids = [int(s) for s in src if isinstance(s, (int, float))] if isinstance(src, list) else []
        out.append({"order": i, "kind": kind, "text": text, "src": src_ids})
    return out


async def assemble_script(
    registry,
    settings,
    regions: list[Region],
    *,
    role: str = "classifier",
    override_model: str = "",
    timeout_s: float = 60.0,
) -> list[dict] | None:
    """Clean + merge + reorder ``regions`` into ``[{order,kind,text}]``.

    Returns ``None`` when the model is unavailable or produces nothing usable —
    the caller falls back to the raw geometric order (still narratable, just
    rougher). Never raises into the synth loop.
    """
    if not regions:
        return []
    try:
        backend, resolved = await registry.resolve_model_for_role(
            role or "classifier", override=override_model or "", settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to raw order
        log.warning("ocr_assembly_resolve_failed", error=str(exc)[:160])
        return None
    if backend is None:
        return None

    # Match the classifier sampling profile (Gemma-E2B needs 1.0/0.95/64).
    cs_temp = float(getattr(settings, "classifier_sampling_temperature", 0.0) or 0.0)
    cs_top_p = float(getattr(settings, "classifier_sampling_top_p", 1.0) or 1.0)
    cs_top_k = int(getattr(settings, "classifier_sampling_top_k", 0) or 0)

    req = InternalChatRequest(
        model=resolved or override_model or "",
        messages=[
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=_user_prompt(regions)),
        ],
        stream=False,
        temperature=cs_temp,
        top_p=(cs_top_p if cs_top_p < 1.0 else None),
        top_k=(cs_top_k if cs_top_k > 0 else None),
        chat_template_kwargs={"enable_thinking": False},
        raw_options={
            "json_schema": ASSEMBLY_SCHEMA,
            "json_schema_name": "comic_page_script",
        },
        # ~30 fragments * (clean line + json overhead). Headroom for a page.
        max_tokens=1200,
    )
    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except (TimeoutError, Exception) as exc:  # noqa: BLE001
        log.warning("ocr_assembly_failed", model=resolved, error=str(exc)[:200])
        return None

    message = getattr(resp, "message", None)
    content = (getattr(message, "content", "") or "").strip()
    parsed = _parse(content)
    if not parsed:
        # Reasoning models sometimes leave the JSON in the thinking trace.
        thinking = (getattr(message, "thinking", "") or "").strip()
        parsed = _parse(thinking)
    log.info("ocr_assembly", model=resolved, in_regions=len(regions), out_lines=len(parsed))
    return parsed or None
