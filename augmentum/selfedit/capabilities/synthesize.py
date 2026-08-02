"""LLM step — a natural-language capability request → a validated CapabilitySpec.

The ONLY model-authored part of the whole pipeline, and it's deliberately
hemmed in: the model picks a behavior from the safe palette and fills data slots,
emitting JSON that must pass ``validate_spec`` before anything is rendered. One
repair retry on validation failure; otherwise it returns ``(None, errors)`` so
the caller can fall the request back to human authoring rather than mint junk.

``model_invoke`` is injected (``async (prompt) -> text``) so this is unit-testable
with a canned responder and engine-agnostic in production (native loop, role
model, or a frontier model all satisfy the same signature).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.selfedit.capabilities.registry_grounding import (
    MatchFn,
    default_match_fn,
    describe_known_verbs,
    find_exact_duplicate,
    find_example_collisions,
    known_verbs_from_registry,
)
from augmentum.selfedit.capabilities.router_catalog import (
    describe_for_prompt,
    validate_declared_args,
    validate_emit_target,
)
from augmentum.selfedit.capabilities.spec import (
    BEHAVIORS,
    SAFE_STAKES,
    CapabilitySpec,
    validate_spec,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ModelInvoke = Callable[[str], Awaitable[str]]

_SCHEMA_DOC = """\
Return ONE JSON object (no prose, no code fence) describing the verb:

{
  "id": "surface.action",          // lowercase, e.g. "navigate.open_browse"
  "summary": "what it does + when to use it (becomes the tool description)",
  "examples": ["open the browse panel", "take me to browse"],
  "behavior": "surface_emit" | "speak",
  // behavior=surface_emit — a UI action the frontend performs:
  "channel": "navigate.open_surface",   // a frontend WS channel
  "payload": {"surface": "browse"},     // static data for the channel
  "toast": "Opening Browse",             // optional small confirmation chip
  // behavior=speak — a canned line it says:
  "speak": "",
  "arg_schema": {"name": {"type": "string", "description": "..."}},  // {} if none
  "required": [],
  "surfaces": ["voice", "chat", "becca"],
  "stakes": "trivial_reversible"
}

Hard rules (the request is REJECTED if broken):
- behavior MUST be one of: surface_emit, speak. You do NOT write code — you only
  choose a behavior and fill data. If the ask needs server work, a network call,
  a database write, sending/posting/paying, or touching the user's data, you
  CANNOT build it here: respond {"unsupported": "<one-line reason>"}.
- stakes MUST be one of: trivial_reversible, disruptive.
- arg types MUST be one of: string, integer, number, boolean.
- The channel and (for navigate.open_surface) the payload.surface MUST come from
  the GROUNDING list below. A channel or surface not in that list registers but
  does NOTHING when dispatched, so it is REJECTED.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    # tolerate a ```json fence or surrounding prose — grab the outermost braces
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _parse_and_validate(
    raw: str, known: list[tuple[str, str]] | None = None,
    match_fn: MatchFn | None = None,
) -> tuple[CapabilitySpec | None, list[str]]:
    obj = _extract_json(raw)
    if obj is None:
        return None, ["model did not return a JSON object"]
    if obj.get("unsupported"):
        return None, [f"unsupported: {str(obj['unsupported'])[:160]}"]
    if obj.get("duplicate"):
        # The model recognized an existing verb already covers the request — the
        # right outcome (reuse, don't mint). Not an error: a no-spec with a clear,
        # non-"invalid" reason so the caller routes to the existing verb.
        return None, [f"duplicate: already covered by {str(obj['duplicate'])[:80]}"]
    spec = CapabilitySpec.from_dict(obj)
    errs = validate_spec(spec)
    # Exact-id gate: never let a synthesized verb reuse an existing id —
    # registry.add() overwrites on duplicate, so this would silently SHADOW a
    # (possibly hand-authored) verb. Reject with the existing summary so the
    # repair retry either picks a new id or declines as a duplicate.
    if not errs and known:
        existing = find_exact_duplicate(spec.id, known)
        if existing:
            errs = [
                f"id {spec.id!r} already exists ({existing[:90]}). Choose a "
                'different id, or respond {"duplicate": "' + spec.id + '"} to reuse it.'
            ]
    # Live-target gate: a structurally-valid surface_emit spec can still name a
    # channel/surface the frontend never handles — registers, passes its oracle,
    # does nothing. Reject it here (with the real catalog in the reason) so the
    # repair retry can fix it rather than minting a dead verb. Tolerant when the
    # router source can't be read (validate_emit_target returns "").
    if not errs and spec.behavior == "surface_emit":
        problem = validate_emit_target(spec.channel, spec.payload)
        if problem:
            errs = [problem]
    # Arg-drop gate: a declared arg that the channel never reads merges into the
    # payload and is silently dropped — "verified but inert". Reject so the model
    # removes it or picks a channel that uses it.
    if not errs and spec.behavior == "surface_emit":
        problem = validate_declared_args(spec.channel, list(spec.arg_schema.keys()))
        if problem:
            errs = [problem]
    # Collision gate: if a trigger phrasing already fires an EXISTING verb, the new
    # verb (registered last) loses tier-1 dispatch on it. Reject so the model picks
    # distinct phrasings or declares the duplicate. Only runs when a matcher is
    # supplied (production passes the live one; unit tests can disable it).
    if not errs and match_fn is not None:
        collisions = find_example_collisions(list(spec.examples), match_fn)
        if collisions:
            ex0, vid0 = collisions[0]
            errs = [
                f"trigger phrasing {ex0!r} already fires the existing verb {vid0!r}, "
                "so the new verb would lose tier-1 dispatch on it. Use distinct "
                f'phrasings, or respond {{"duplicate": "{vid0}"}} if {vid0!r} already '
                "does what the user wants."
            ]
    return (None if errs else spec), errs


def _prompt(request: str, known: list[tuple[str, str]] | None = None) -> str:
    blocks = [describe_for_prompt(), describe_known_verbs(known or [])]
    grounding_block = "".join(f"\n\n{b}" for b in blocks if b)
    return (
        "You author a new primitive VERB for Augmentum's action registry — a "
        "permanent new thing the assistant can do on command.\n\n"
        f"The user wants this capability:\n{request.strip()}\n\n"
        f"{_SCHEMA_DOC}{grounding_block}"
    )


def _repair_prompt(
    request: str, raw: str, errs: list[str], known: list[tuple[str, str]] | None = None,
) -> str:
    return (
        f"{_prompt(request, known)}\n\n"
        f"Your previous answer was invalid:\n{raw[:800]}\n\n"
        "Problems:\n" + "\n".join(f"- {e}" for e in errs) + "\n\n"
        "Return a corrected JSON object only."
    )


async def synthesize_capability_spec(
    request: str, *, model_invoke: ModelInvoke,
    known_verbs: list[tuple[str, str]] | None = None,
    match_fn: MatchFn | None = None,
) -> tuple[CapabilitySpec | None, list[str]]:
    """Synthesize + validate a CapabilitySpec from a request. Returns
    ``(spec, [])`` on success, or ``(None, errors)`` (the request couldn't be
    expressed as a safe verb — fall back to human authoring).

    ``known_verbs`` grounds the model in the verbs that already exist so it reuses
    or declines instead of minting duplicates; defaults to the live registry.
    ``match_fn`` is the tier-1 matcher used for the collision gate; defaults to the
    live matcher. Pass ``lambda _t: None`` to disable collision checks (tests)."""
    known = known_verbs if known_verbs is not None else known_verbs_from_registry()
    matcher = match_fn if match_fn is not None else default_match_fn()
    try:
        raw = await model_invoke(_prompt(request, known))
    except Exception as exc:  # noqa: BLE001 — model hiccup = no spec, not a crash
        log.warning("capability_synthesize_model_failed", error=repr(exc))
        return None, [f"model call failed: {exc!r}"]

    spec, errs = _parse_and_validate(raw, known, matcher)
    if spec is not None:
        log.info("capability_spec_synthesized", id=spec.id, behavior=spec.behavior)
        return spec, []

    # A clean "duplicate"/"unsupported" decline isn't a retryable error — the
    # request is already covered or out of scope; don't burn a repair call on it.
    if errs and errs[0].startswith(("duplicate:", "unsupported:")):
        log.info("capability_spec_declined", reason=errs[0])
        return None, errs

    # one repair retry
    try:
        raw2 = await model_invoke(_repair_prompt(request, raw, errs, known))
    except Exception as exc:  # noqa: BLE001
        log.warning("capability_synthesize_repair_failed", error=repr(exc))
        return None, errs
    spec2, errs2 = _parse_and_validate(raw2, known, matcher)
    if spec2 is not None:
        log.info("capability_spec_synthesized_after_repair", id=spec2.id)
        return spec2, []
    log.info("capability_spec_rejected", errors=errs2[:4])
    return None, errs2


# Re-exported for callers that want the palette without importing spec directly.
SAFE_BEHAVIORS = BEHAVIORS
SAFE_STAKES_SET = SAFE_STAKES
