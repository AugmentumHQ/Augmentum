"""Tool catalogue + execution bridge (Lane 3 §1, §2).

This module classifies each existing subagent/primitive as a TOOL (inline
single-shot) or CHANNEL (multi-turn handoff), declares the tool grammar
Becca emits to call them, and converts the existing SubagentResult /
PrimitiveResult shapes into the unified ``ToolResult`` envelope the voice
pipeline expects.

The classification is data, not configuration — the existing adapter
files stay unchanged. Adding a new tool means adding an entry here.

Lane 3's table mapped:
  TOOL:     passthrough, analytical, image_gen, web_search, browse,
            files (search/read), code_exec, memory_recall, tts/stt
  CHANNEL:  coder, agentic, narrative (RP), bug_finder, game_agent, cardsmith
  BOTH:     narrative (one-shot scene), browse (interactive — v2)
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from typing import TYPE_CHECKING, Any

from augmentum.companion_runtime.tool_protocol import (
    ToolCall,
    ToolError,
    ToolResult,
    UIEffect,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Per-tool timeout. The 5-tool cap + 60s budget in voice.py is the
# higher-level guard; this is the per-call safety net.
_PER_TOOL_TIMEOUT_S = 120.0


# ── Tool catalogue ───────────────────────────────────────────────────

# Each entry's keys:
#   registry      "primitive" | "subagent"
#   name          canonical id matching the registry entry
#   description   one line for the prompt's tool roster
#   args_hint     shown in the prompt as <tool:NAME args_hint />
#   ui_effect     optional bus event to fan out on success (Lane 3 §7)

TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "recall": {
        "registry": "primitive",
        "name": "memory_recall",
        "description": "Look in your own memory for relevant context.",
        "args_hint": 'query="..."',
    },
    "browse": {
        "registry": "primitive",
        "name": "browse",
        # Headless READ tool — matches the underlying `web` tool's own contract
        # ("Nothing opens on the user's screen"). First sentence carries the
        # delivery keyword ("silently") per the roster convention so the model
        # treats this as a gather tool, not a show-me one.
        #
        # NO ui_effect: a lookup must NOT auto-mount a panel. Until 2026-06-25
        # this entry fired `browse.snapshot_ready` on EVERY successful call, so a
        # plain "what's the news" popped the browse panel and she briefed thinly
        # off the open page instead of gathering and answering in words —
        # contradicting both this tool's contract and the headless-first doctrine.
        # The snapshot event had no consumer (orphaned). Showing a page is now
        # opt-in only, via the explicit screen verbs (browse.open_url /
        # browse.search / navigate.open_surface) the model reaches for when the
        # user actually asks to SEE something.
        "description": "Look something up and READ it silently — nothing opens on screen. Pass url= or query=.",
        "args_hint": 'url="..."',
    },
    "image": {
        "registry": "primitive",
        "name": "image_gen",
        "description": "Generate an image or sketch from a prompt.",
        "args_hint": 'prompt="..."',
        "ui_effect": "image.generated",
    },
    "files_read": {
        "registry": "primitive",
        # The handler is search-only (index lookup), not path-read — the
        # advertised path="..." made the model fill an arg the handler
        # never read, so every call silently returned "empty query"
        # (audit 2026-06-17). Advertise the real contract.
        "name": "files",
        "description": "Search your indexed files by name or keyword.",
        "args_hint": 'query="..."',
    },
    "code_run": {
        "registry": "primitive",
        # Handler reads `code`; the old hint advertised body="..." (+ an
        # ignored lang=), so the model's body= arg was dropped and the
        # call returned "empty code" (audit 2026-06-17).
        "name": "code_exec",
        "description": "Run a small bit of Python code in a sandbox.",
        "args_hint": 'code="..."',
    },
    "analytical": {
        "registry": "subagent",
        "name": "analytical",
        "description": (
            "Rigorous reasoning for hard analytical or factual questions — "
            "use when the answer needs care, not when chatting."
        ),
        "args_hint": 'question="..."',
    },
    "passthrough": {
        "registry": "subagent",
        "name": "passthrough",
        "description": (
            "Direct single-shot ask of the base model with web tools "
            "available. Use for general lookups that aren't memory or browse."
        ),
        "args_hint": 'question="..."',
    },
}


# ── Core capability roster (prompt-visibility only) ──────────────────
#
# These are backend function-calling tools carried in
# ``native_loop.CORE_TOOL_NAMES`` — already RESOLVABLE and EXECUTABLE via
# the native loop's PassthroughHandler (``_execute_and_append``) in BOTH
# the native-FC and TEXT tool-calling tiers, so a big or a small model
# dispatches them identically. What they LACKED was a line in the prompt
# roster: ``enumerate_tools`` only drew from TOOL_CATALOG + the intent
# registry, so the model was never TOLD youtube/image_search existed and
# defaulted to media_recommendations (local library) or a raw browse
# search for "find me a youtube video" (companion_eval gap 2026-06-10,
# re-observed live 2026-07-16 → Spartacus from the user's own files).
#
# Kept SEPARATE from TOOL_CATALOG on purpose: a TOOL_CATALOG entry routes
# execute_tool → _execute_primitive(entry["name"]), but these are not
# companion primitives — they'd 'unknown tool' on the execute_tool path
# (timers, intent dispatch). Surfacing them here changes only what she's
# told; execution stays on the native-loop registry path it already uses.
# Descriptions + required args are verbatim from the backend tool defs.
_CORE_CAPABILITY_ROSTER: dict[str, dict[str, str]] = {
    "youtube": {
        "description": "Find and watch YouTube videos with transcripts.",
        "args_hint": 'query="..."',
    },
    "image_search": {
        "description": "Search the web for an existing image and save it.",
        "args_hint": 'query="..."',
    },
    "remove_background": {
        "description": "Remove the background from an image (transparent PNG).",
        "args_hint": 'artifact_id="..."',
    },
}


CHANNEL_CATALOG: dict[str, dict[str, Any]] = {
    "coder": {
        "description": "Open the coder workspace for multi-turn code editing.",
    },
    "narrative": {
        "description": "Enter narrative mode for fiction or RP sessions.",
    },
    "agentic": {
        "description": "Plan and run a multi-step task with approval gates.",
    },
    "bug_finder": {
        "description": "Run the bug-finder over a codebase.",
    },
}


# Maximum total characters consumed by the per-verb lines rendered
# in :func:`augmentum.companion_runtime.prompt_compose._tool_roster_block`.
# The static grammar header in prompt_compose.py is ~470 chars (~118
# tokens). The Sprint-B design allocated 280 tokens / ~1120 chars to
# the entire tools layer, but that was set when typical local models
# topped out at 4-8K context. Today's local 30B+ MoE models have
# 32K+ context, so a slightly richer roster is affordable and worth it
# for capability coverage.
#
# 1200 chars (~300 tokens) for per-verb lines + ~120 tokens header
# = ~420 tokens for the whole tools layer. Comfortably under the
# 1800-token total prompt ceiling once other layers (digest, memory,
# relationship, transcript) take their share.
#
# Without this cap, every Tier-3-eligible action in REGISTRY.all() —
# architect primitives + Phase 0 verbs + control verbs — gets enumerated
# verbatim, producing a 1714-token tools layer that crashes the
# 1800-token prompt ceiling and forces small/medium models into a
# text-format tool-call fallback that leaks raw "tool:NAME args=..."
# syntax into TTS.
_TOOL_ROSTER_CHAR_BUDGET = 1200


# ── Roster relevance scoring ─────────────────────────────────────────
#
# Selection, not dispatch: these scores decide which verbs make the
# inline roster under the char budget — the LLM still chooses whether
# and what to call. (Distinct from the banned regex-switchboard
# pattern, which extracted intent + args from raw text.)

_ROSTER_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "for", "from", "get", "had", "has", "have", "he", "her", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "let", "me", "my",
    "no", "not", "of", "on", "or", "our", "she", "so", "some", "than",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "up", "us", "was", "we", "what", "when", "which", "who", "will",
    "with", "would", "you", "your", "please",
})

# Per-action embedding cache — id → unit-normalized vector of the
# verb's summary + examples. Populated lazily, only when the embedder
# is already warm (we never trigger the ~130MB first-load from here).
_roster_doc_vecs: dict[str, list[float]] = {}


def _roster_tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9']+", text.lower())
        if len(t) > 1 and t not in _ROSTER_STOPWORDS
    }


def _roster_lexical(turn_tokens: set[str], action) -> float:
    """Weighted token overlap between the turn and the verb's surface.

    Examples weigh heaviest — they're literal phrasings of the verb's
    use. The id contributes its dotted segments ("media", "play").
    Normalized by turn size so long turns don't dilute every verb
    equally.
    """
    if not turn_tokens:
        return 0.0
    id_tokens = _roster_tokens(action.id.replace(".", " "))
    summary_tokens = _roster_tokens(action.summary or "")
    example_tokens: set[str] = set()
    for ex in (action.examples or [])[:6]:
        example_tokens |= _roster_tokens(ex)
    score = (
        2.0 * len(turn_tokens & example_tokens)
        + 1.5 * len(turn_tokens & id_tokens)
        + 1.0 * len(turn_tokens & summary_tokens)
    )
    return score / max(3.0, float(len(turn_tokens)))


def _embedder_if_warm():
    """Return the EmbeddingService class iff its model is already
    loaded. The roster path must never pay (or block on) the first
    ~130MB model load — memory recall warms it in practice."""
    try:
        from augmentum.memory.embeddings import (
            _LOAD_FAILED,
            _UNLOADED,
            EmbeddingService,
        )
        model = getattr(EmbeddingService, "_model", None)
        if model is None or model is _UNLOADED or model is _LOAD_FAILED:
            return None
        return EmbeddingService
    except Exception:
        return None


def _roster_query_vec(turn_text: str, svc) -> list[float] | None:
    """Embed + normalize the turn text ONCE per roster build.

    Previously ``_roster_cosine`` re-embedded ``turn_text`` for every
    eligible verb, so a single turn ran N synchronous ONNX embeds on the
    event loop — the source of the ``event_loop_stall`` (lag_s up to 2.2)
    the watchdog logged immediately after ``tool_roster_budget_clipped
    relevance_ranked=True``. Computing it once and sharing the vector
    across verbs collapses that to a single embed per turn.
    """
    try:
        q = svc.embed_query(turn_text)
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        return [x / qn for x in q]
    except Exception:
        return None


def _roster_cosine(action, qvec_normed, svc) -> float:
    """Cosine between the (pre-normalized) turn vector and the verb's
    cached doc vector. Both vectors are unit-normalized, so the dot
    product is the cosine. Returns 0.0 on any failure — relevance must
    never break a roster."""
    try:
        doc_vec = _roster_doc_vecs.get(action.id)
        if doc_vec is None:
            doc = " ".join(
                [action.summary or "", *list(action.examples or [])[:6]],
            ).strip()
            if not doc:
                return 0.0
            doc_vec = svc.embed([doc])[0]
            norm = math.sqrt(sum(x * x for x in doc_vec)) or 1.0
            doc_vec = [x / norm for x in doc_vec]
            _roster_doc_vecs[action.id] = doc_vec
        return max(0.0, sum(a * b for a, b in zip(qvec_normed, doc_vec, strict=False)))
    except Exception:
        return 0.0


def _roster_relevance(turn_text: str, action, qvec_normed=None, svc=None) -> float:
    """Blend lexical overlap with embedding cosine (when warm).

    ``qvec_normed`` is the turn's normalized embedding, computed ONCE by
    the caller (:func:`_roster_query_vec`) and shared across every verb
    this turn. When it's None — embedder cold, or the caller didn't
    precompute — we fall back to pure lexical and never embed per verb.
    """
    lex = _roster_lexical(_roster_tokens(turn_text), action)
    if svc is None or qvec_normed is None:
        return lex
    cos = _roster_cosine(action, qvec_normed, svc)
    return 0.5 * min(1.0, lex) + 0.5 * cos


def _approx_line_chars(entry: dict[str, Any]) -> int:
    """Estimate the chars ``_tool_roster_block`` will render for
    ``entry``. Mirrors the rendering shape so the budget tracks reality
    (rather than counting raw description chars, which over-states by
    the description-trim factor)."""
    name = entry.get("name", "?")
    args_hint = entry.get("args_hint", "")
    base = (
        len(f"  <tool:{name} {args_hint} />")
        if args_hint else len(f"  <tool:{name} />")
    )
    desc = entry.get("description", "").strip()
    if desc:
        # Same trim _tool_roster_block applies.
        short = desc.split(".")[0].strip()[:60].rstrip()
        if short:
            base += len(f"  -- {short}")
    return base + 1  # +1 for the join newline


def pending_pin(app_state: Any, user_id: str, session_id: str) -> tuple[str, ...]:
    """Verbs that MUST survive roster clipping this turn.

    A parked clarification ("what city should I use?") means the next
    coherent utterance is probably the answer — but the answer text
    ("Springfield, Illinois") shares no vocabulary with the waiting
    verb, so relevance ranking defers it out of the roster and the
    model literally cannot fill the slot. Found by companion_eval's
    clarify-weather scenario (2026-06-11). Returns the parked verb's
    id, freshness-gated by the pending-intent TTL.
    """
    try:
        from augmentum.intent.dispatch import (
            get_fresh_pending_intent,
            get_referent_cache,
        )
        refs = get_referent_cache(app_state, user_id, session_id)
        pi = get_fresh_pending_intent(refs)
        if pi and pi.get("action_id"):
            return (str(pi["action_id"]),)
    except Exception:  # noqa: BLE001 — pin is best-effort
        log.debug("pending_pin_failed", exc_info=True)
    return ()


def enumerate_tools(
    turn_text: str = "", pin: tuple[str, ...] = (),
    *, context_budget_chars: int | None = None,
) -> list[dict[str, str]]:
    """Snapshot of the tool catalogue for ``prompt_compose``'s roster.

    ``context_budget_chars`` overrides the static
    :data:`_TOOL_ROSTER_CHAR_BUDGET` with a value scaled to the loaded
    model's context window (see ``context_budget.derive_roster_char_budget``).
    When ``None`` or non-positive, the fixed budget is used — so callers that
    don't (yet) know the window get legacy behaviour with zero regression.

    Merges two sources so Becca's prompt sees ALL primitives she can
    dispatch, not just the legacy TOOL_CATALOG entries:

      1. **TOOL_CATALOG** — handler-style tools (recall, browse, image,
         narrative, coder, agentic, …) that the companion runtime
         executes via :func:`execute_tool`.
      2. **Intent registry** — architect-callable primitives (every
         Action exposed on 'becca' or 'chat' surfaces) registered via
         ``@register_action`` in ``augmentum/intent/`` and
         ``augmentum/architect/primitives/``.

    Without the merge, the architect ships primitives Becca technically
    CAN dispatch but doesn't KNOW about — she keeps refusing real
    requests with "I'm text-only" because her prompt enumerates only
    half the catalogue. With it, every primitive she can call is in
    the roster, with consistent description + args grammar.

    Budget enforcement: total per-verb char count is capped at
    :data:`_TOOL_ROSTER_CHAR_BUDGET`. TOOL_CATALOG entries are pinned
    (cheap, universal); registry verbs fill the remaining budget in
    **relevance order** when ``turn_text`` is given — scored against
    the verb's id/summary/examples (lexical overlap, blended with
    embedding cosine when the embedder is already warm). Registration
    order is the fallback when no turn text is supplied. Before
    relevance ranking, registration order silently starved late-
    registered verbs (media.play never made the cut → "I can't play
    audio files directly", 2026-06-10). Deferred verbs are still
    callable — :func:`execute_tool` resolves the full registry — the
    inline roster is just what she's TOLD about.
    """
    out: list[dict[str, str]] = []
    chars_used = 0
    # Context-scaled budget when the caller supplied one; else the legacy fixed
    # cap. All budget checks below use this local, not the module constant.
    _budget = (
        int(context_budget_chars)
        if context_budget_chars and context_budget_chars > 0
        else _TOOL_ROSTER_CHAR_BUDGET
    )

    # TOOL_CATALOG entries — added first. These are the canonical
    # companion-runtime-executable handlers and have curated short
    # descriptions, so they're cheap.
    for canon, entry in TOOL_CATALOG.items():
        candidate = {
            "name": canon,
            "description": entry["description"],
            "args_hint": entry["args_hint"],
        }
        cost = _approx_line_chars(candidate)
        if chars_used + cost > _budget:
            # Catalogue overflowed — log so the operator sees it.
            log.info(
                "tool_roster_budget_clipped",
                stage="catalogue",
                added=len(out),
                deferred=len(TOOL_CATALOG) - len(out),
                budget=_budget,
            )
            return out
        out.append(candidate)
        chars_used += cost

    # Core capability tools (youtube, image_search, remove_background):
    # backend FC tools she can already call via the native loop but was
    # never TOLD about. Pinned like TOOL_CATALOG (few, universal, cheap)
    # and added before the relevance-ranked registry verbs so a "find me
    # a video" turn actually sees the video tool instead of falling back
    # to media_recommendations / browse. Budget-checked + deduped.
    for canon, entry in _CORE_CAPABILITY_ROSTER.items():
        if any(t["name"] == canon for t in out):
            continue
        candidate = {
            "name": canon,
            "description": entry["description"],
            "args_hint": entry["args_hint"],
        }
        cost = _approx_line_chars(candidate)
        if chars_used + cost > _budget:
            log.info(
                "tool_roster_budget_clipped",
                stage="core_capability",
                added=len(out),
                budget=_budget,
            )
            break
        out.append(candidate)
        chars_used += cost

    # Append architect-callable intent-registry actions. Filter to
    # ones that target Becca's surfaces ('becca' or 'chat') — XR-only
    # or voice-call-only verbs (e.g., "stop", "repeat") aren't useful
    # in the tool roster.
    try:
        from augmentum.intent.registry import REGISTRY

        eligible = []
        for action in REGISTRY.all():
            if not action.fanout.tier3:
                continue  # opted out of LLM tool exposure
            # Surfaces: empty = all, else require becca/chat overlap.
            if action.surfaces and not (
                "becca" in action.surfaces or "chat" in action.surfaces
            ):
                continue
            # Skip duplicates if the same id already came from TOOL_CATALOG
            if any(t["name"] == action.id for t in out):
                continue
            eligible.append(action)

        # Load-bearing verbs (fanout.always_offer) are the sole path to a major
        # capability — never let the char budget or the family cap silently clip
        # them out of the roster (a clipped tool is invisible AND uncallable).
        _always_ids = {
            a.id for a in eligible if getattr(a.fanout, "always_offer", False)
        }

        if turn_text.strip():
            # Relevance order — stable sort keeps registration order
            # for ties, so an all-zero-score turn degrades to legacy.
            # Embed the turn ONCE (only when the embedder is already warm)
            # and share the vector across every verb — see
            # _roster_query_vec. Previously each verb re-embedded the turn,
            # so this loop ran N synchronous embeds and stalled the loop.
            _svc = _embedder_if_warm()
            _qvec = _roster_query_vec(turn_text, _svc) if _svc is not None else None
            scored = [
                (_roster_relevance(turn_text, a, _qvec, _svc), i, a)
                for i, a in enumerate(eligible)
            ]
            scored.sort(key=lambda t: (-t[0], t[1]))
            ordered = [a for _, _, a in scored]
            # Family diversity — at most N verbs per dotted-id family
            # in the first pass; overflow re-queues behind everything
            # else (still reachable when budget allows). Raw top-K
            # clusters on one domain: "throw in some music and open a
            # note" scored five note.* verbs above ANY music verb, the
            # 7-slot budget filled with notes, and she told the user
            # she can't play music (2026-06-11). Multi-intent turns
            # need breadth across families more than depth within one.
            _FAMILY_CAP = 3
            fam_counts: dict[str, int] = {}
            first_pass: list = []
            overflow: list = []
            for a in ordered:
                fam = a.id.split(".", 1)[0]
                if a.id not in _always_ids and fam_counts.get(fam, 0) >= _FAMILY_CAP:
                    overflow.append(a)
                    continue
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
                first_pass.append(a)
            ordered = first_pass + overflow
        else:
            ordered = eligible

        # Pinned verbs jump the queue and are exempt from the family
        # cap — a parked clarification's verb must be in the roster on
        # the answer turn regardless of how the answer text scores.
        if pin:
            pin_set = {p for p in pin if p}
            pinned = [a for a in ordered if a.id in pin_set]
            if pinned:
                ordered = pinned + [a for a in ordered if a.id not in pin_set]

        registry_added = 0
        deferred_names: list[str] = []
        for action in ordered:
            # Render args_hint from the JSON schema — pick the required
            # keys first, then a couple of optionals, so the grammar
            # line stays under ~60 chars.
            required = list(action.required_args or [])
            schema = action.arg_schema or {}
            # Confirm-flow slots (memory_id/confirm on memory.forget,
            # playlist_id/confirm on playlist.delete) are filled by
            # the parked-clarify machinery, never by the model on a
            # first call. Showing them in the grammar hint teaches
            # small models to stuff junk into them — which at best
            # produces "couldn't find that one" misses and at worst
            # races the confirm gate. Convention: their schema
            # description starts with "Internal".
            optionals = [
                k for k, v in schema.items()
                if k not in required
                and not str((v or {}).get("description", "")).startswith("Internal")
            ]
            hint_keys = required + optionals[: max(0, 3 - len(required))]
            args_hint = ""
            if hint_keys:
                # "query='...' source='...'" — single-quoted ellipsis
                # so the grammar matches TOOL_CATALOG's existing shape.
                args_hint = " ".join(f"{k}='...'" for k in hint_keys)
            candidate = {
                "name": action.id,
                "description": (action.summary or "").strip(),
                "args_hint": args_hint,
            }
            cost = _approx_line_chars(candidate)
            # Load-bearing verbs are budget-EXEMPT — added regardless of
            # remaining budget so a "launch a coder run" is never met with a
            # tool the model can't see or call. (Keep the always_offer set small.)
            if action.id not in _always_ids and chars_used + cost > _budget:
                deferred_names.append(action.id)
                continue  # try remaining (cheaper lines may still fit)
            out.append(candidate)
            chars_used += cost
            registry_added += 1
        if deferred_names:
            log.info(
                "tool_roster_budget_clipped",
                stage="registry",
                catalogue_added=len(TOOL_CATALOG),
                registry_added=registry_added,
                # The verbs that DID make the roster — without this the
                # clip log only showed what was dropped, and "did
                # grove.play_matching make the cut for this turn?" was
                # unanswerable from logs (2026-06-10 act-gap debugging).
                added=[
                    t["name"] for t in out[len(TOOL_CATALOG):]
                ],
                registry_seen=len(eligible),
                registry_deferred=len(deferred_names),
                deferred=deferred_names[:12],
                relevance_ranked=bool(turn_text.strip()),
                budget=_budget,
                used=chars_used,
            )
    except Exception as exc:  # noqa: BLE001 — never break the roster
        # If the intent registry fails to import (test isolation) we
        # still ship the legacy TOOL_CATALOG so the prompt isn't empty.
        from augmentum.utils.logging import get_logger as _gl
        _gl(__name__).debug(
            "enumerate_tools_intent_merge_failed", error=str(exc)[:200],
        )

    return out


def enumerate_channels() -> list[dict[str, str]]:
    return [
        {"name": canon, "description": entry["description"]}
        for canon, entry in CHANNEL_CATALOG.items()
    ]


def known_tool_names() -> set[str]:
    """Canonical ids the TagSieve salvage pass may resolve against:
    TOOL_CATALOG handlers plus every tier-3 intent-registry verb.
    Used as the registry gate for mangled-prefix tag recovery
    (``<j:play_matching …/>`` → grove.play_matching)."""
    names = set(TOOL_CATALOG.keys())
    try:
        from augmentum.intent.registry import REGISTRY
        for action in REGISTRY.all():
            if action.fanout.tier3:
                names.add(action.id)
    except Exception:  # noqa: BLE001 — salvage is optional
        pass
    return names


# ── Execution ────────────────────────────────────────────────────────

async def execute_tool(
    call: ToolCall,
    runtime: CompanionRuntime,
    *,
    cancel: asyncio.Event | None = None,
    user_id: str = "",
    session_id: str = "",
) -> ToolResult:
    """Invoke a tool by canonical name. Returns a ``ToolResult`` envelope.

    Error categories (Lane 3 §2.3): timeout / unauthorized /
    content_policy / model_unavailable / invalid_args / upstream_error /
    cancelled / tool_self_error.

    Resolution order: TOOL_CATALOG (handler-style tools the companion
    runtime executes directly), then the intent-action REGISTRY
    (architect verbs — media.play, navigate.open_surface, note.create,
    …). ``enumerate_tools`` advertises BOTH sources in Becca's roster,
    so without the registry fallback she's promised verbs the executor
    rejects — the 2026-06-10 "unknown tool: navigate.open_surface" bug.

    The runtime fan-out of ``ui_effects`` happens here, AFTER successful
    invocation — so e.g. ``image.generated`` fires identically whether
    image_gen was called from Becca or from the legacy /image route.
    """
    entry = TOOL_CATALOG.get(call.name)
    registry_kind = entry["registry"] if entry else "intent"
    if entry is None and _lookup_registry_action(call.name) is None:
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(
                category="invalid_args",
                message=f"unknown tool: {call.name}",
            ),
        )

    if cancel is not None and cancel.is_set():
        return ToolResult(
            ok=False, tool=call.name, payload=None, cancelled=True,
            error=ToolError(category="cancelled", message="cancelled before dispatch"),
        )

    t_start = time.monotonic()
    if runtime is not None:
        try:
            await runtime.bus.publish_topic(
                "tool.invoked",
                {
                    "name": call.name,
                    "registry": registry_kind,
                    "user_id": user_id,
                    "args_keys": sorted(call.args.keys()) if call.args else [],
                },
                source_companion_id=runtime.companion_id,
            )
        except Exception:
            log.warning("tool_invoked_emit_failed", tool=call.name, exc_info=True)

    if entry is None:
        result = await _execute_registry_action(
            call, runtime, user_id=user_id, session_id=session_id,
        )
    elif entry["registry"] == "primitive":
        result = await _execute_primitive(entry["name"], call.args, runtime, user_id=user_id)
    elif entry["registry"] == "subagent":
        result = await _execute_subagent(entry["name"], call.args, runtime, user_id=user_id)
    else:
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(
                category="invalid_args",
                message=f"unknown registry kind: {entry['registry']}",
            ),
        )

    duration_ms = int((time.monotonic() - t_start) * 1000)
    result = _replace_duration(result, duration_ms)

    if runtime is not None:
        try:
            await runtime.bus.publish_topic(
                "tool.completed",
                {
                    "name": call.name,
                    "registry": registry_kind,
                    "user_id": user_id,
                    "ok": result.ok,
                    "duration_ms": duration_ms,
                    "error_category": result.error.category if result.error else None,
                },
                source_companion_id=runtime.companion_id,
            )
        except Exception:
            log.warning("tool_completed_emit_failed", tool=call.name, exc_info=True)

    # Side-effect fan-out (Lane 3 §7): emit the declared bus event so the
    # existing UI surfaces mount as they do from the legacy routes.
    # Registry actions handle their own surface_emit→ui_effects mapping
    # inside _execute_registry_action, so this block is catalog-only.
    if result.ok and entry is not None and "ui_effect" in entry:
        effect = UIEffect(
            kind=entry["ui_effect"],
            target="_inline",
            payload=_payload_for_effect(result.payload),
        )
        try:
            await runtime.bus.publish_topic(
                effect.kind,
                {
                    "target": effect.target,
                    "payload": effect.payload,
                    "source": "companion_tool_call",
                    "tool": call.name,
                },
                source_companion_id=runtime.companion_id,
            )
        except Exception:
            log.warning("ui_effect_publish_failed", tool=call.name, kind=effect.kind, exc_info=True)
        result = ToolResult(
            ok=result.ok, tool=result.tool, payload=result.payload,
            metadata=result.metadata, ui_effects=(effect,),
            duration_ms=result.duration_ms, cancelled=result.cancelled,
            error=result.error,
        )

    return result


# ── Private helpers ──────────────────────────────────────────────────

# Defensive arg-name normalization: map the names a model is likely to
# emit (from the catalog hint or common synonyms) onto the kwarg the
# primitive actually reads. Belt-and-suspenders alongside the catalog
# args_hint fix — a model that says body=/path= still works (audit
# 2026-06-17). Keep this small; the durable fix is a shared param schema.
_PRIMITIVE_ARG_ALIASES: dict[str, dict[str, str]] = {
    "code_exec": {"body": "code", "source": "code", "snippet": "code"},
    "files": {"path": "query", "file": "query", "name": "query", "q": "query"},
}


async def _execute_primitive(
    primitive_name: str,
    args: dict[str, str],
    runtime: CompanionRuntime,
    *,
    user_id: str,
) -> ToolResult:
    from augmentum.companion_runtime.primitives.base import PrimitiveContext
    from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry

    aliases = _PRIMITIVE_ARG_ALIASES.get(primitive_name)
    if aliases and args:
        # Only remap a synonym when the canonical key isn't already present.
        args = {**{aliases.get(k, k): v for k, v in args.items()}}

    prim = PrimitiveRegistry.get(primitive_name)
    if prim is None:
        return ToolResult(
            ok=False, tool=primitive_name, payload=None,
            error=ToolError(
                category="tool_self_error",
                message=f"primitive not registered: {primitive_name}",
            ),
        )

    ctx = PrimitiveContext(
        runtime=runtime,
        bus=runtime.bus,
        companion_id=runtime.companion_id,
        user_id=user_id,
        request_id="",
    )

    try:
        result = await asyncio.wait_for(prim.call(ctx, **args), timeout=_PER_TOOL_TIMEOUT_S)
    except TimeoutError:
        return ToolResult(
            ok=False, tool=primitive_name, payload=None,
            error=ToolError(category="timeout", message="primitive timeout", retryable=True),
        )
    except Exception as exc:
        log.warning("primitive_call_failed", primitive=primitive_name, error=str(exc)[:200])
        return ToolResult(
            ok=False, tool=primitive_name, payload=None,
            error=ToolError(category="upstream_error", message=str(exc)[:200]),
        )

    return ToolResult(
        ok=result.ok,
        tool=primitive_name,
        payload=result.payload,
        metadata=dict(result.metadata or {}),
        error=None if result.ok else ToolError(
            category="upstream_error", message=result.error or "primitive failure",
        ),
    )


async def _execute_subagent(
    subagent_name: str,
    args: dict[str, str],
    runtime: CompanionRuntime,
    *,
    user_id: str,
) -> ToolResult:
    from augmentum.companion_runtime.runtime import Intent
    from augmentum.companion_runtime.subagents.base import SubagentContext
    from augmentum.companion_runtime.subagents.registry import SubagentRegistry

    sub = SubagentRegistry.get(subagent_name)
    if sub is None:
        return ToolResult(
            ok=False, tool=subagent_name, payload=None,
            error=ToolError(
                category="tool_self_error",
                message=f"subagent not registered: {subagent_name}",
            ),
        )

    # Translate the tool args into a single-question Intent the subagent
    # adapter expects. Each entry in TOOL_CATALOG documents which arg name
    # carries the prompt — we accept question / prompt / body / text for
    # forgiveness in what Becca emits.
    text = (
        args.get("question") or args.get("prompt")
        or args.get("body") or args.get("text") or ""
    )
    intent = Intent(
        text=text,
        user_id=user_id,
        source="becca_tool_call",
        metadata={"as_tool": True},
    )
    ctx = SubagentContext(
        intent=intent,
        runtime=runtime,
        bus=runtime.bus,
        companion_id=runtime.companion_id,
        invocation_id="",
    )

    try:
        result = await asyncio.wait_for(sub.invoke(ctx), timeout=_PER_TOOL_TIMEOUT_S)
    except TimeoutError:
        return ToolResult(
            ok=False, tool=subagent_name, payload=None,
            error=ToolError(category="timeout", message="subagent timeout", retryable=True),
        )
    except Exception as exc:
        log.warning("subagent_call_failed", subagent=subagent_name, error=str(exc)[:200])
        return ToolResult(
            ok=False, tool=subagent_name, payload=None,
            error=ToolError(category="upstream_error", message=str(exc)[:200]),
        )

    ok = not result.error
    return ToolResult(
        ok=ok,
        tool=subagent_name,
        payload={"content": result.content},
        metadata=dict(result.metadata or {}),
        error=ToolError(category="upstream_error", message=result.error) if not ok else None,
    )


def _lookup_registry_action(name: str):
    """Return the tier-3 intent Action for ``name``, or None.

    None covers: registry import failure (test isolation), unknown id,
    and actions that opted out of LLM exposure (fanout.tier3=False) —
    those are surface-control verbs (stop/repeat) Becca shouldn't fire.
    """
    try:
        from augmentum.intent.registry import REGISTRY
    except Exception:
        return None
    action = REGISTRY.get(name)
    if action is None or not action.fanout.tier3:
        return None
    return action


async def _execute_registry_action(
    call: ToolCall,
    runtime: CompanionRuntime,
    *,
    user_id: str,
    session_id: str = "",
) -> ToolResult:
    """Execute an intent-registry verb through the architect invocation
    pipeline: referent-cache binding → arg inference → translation →
    required-args validation → handler → dispatch anchors.

    Mirrors ``architect/dispatch.py::dispatch_architect_command`` minus
    the matching stage (the LLM already chose the verb and args) so a
    verb behaves identically whether it fired from a Tier-1 voice match
    or a companion tool tag. ``ActionResult.surface_emit`` becomes a
    ``ToolResult.ui_effects`` entry keyed by the emit channel — the
    chat/voice consumers route those to ``intent-action-router.js``.
    """
    from augmentum.intent.action import SessionContext

    action = _lookup_registry_action(call.name)
    if action is None:
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(
                category="invalid_args",
                message=f"unknown tool: {call.name}",
            ),
        )

    app_state = getattr(runtime, "_app_state", None) if runtime is not None else None
    session = SessionContext(
        user_id=user_id,
        session_id=session_id,
        mode=None,
        app_state=app_state,
    )
    # Bind the persistent per-session ReferentCache so anchor writes
    # ("the one you just played") survive past this call.
    if app_state is not None and user_id:
        try:
            from augmentum.intent.dispatch import get_referent_cache
            session.referents = get_referent_cache(app_state, user_id, session_id)
            # Continuity: a fresh cache (new session id / post-restart)
            # rehydrates the working set (active note, trail) from the
            # per-user settings store before the handler reads it.
            from augmentum.companion_runtime.working_state import (
                hydrate_working_state,
            )
            await hydrate_working_state(app_state, user_id, session.referents)
        except Exception:
            log.warning("registry_action_referent_bind_failed", tool=call.name, exc_info=True)

    args: dict[str, Any] = dict(call.args or {})

    # Inference — fill missing args from observation history. Failures
    # degrade to the partial args; never block a dispatch.
    try:
        from augmentum.architect.inference import infer_args
        args = await infer_args(action, dict(args), session, runtime)
    except Exception:
        log.warning("registry_action_inference_failed", tool=call.name, exc_info=True)

    # Translation — reshape raw user phrasing into well-formed tool input
    # (e.g. image prompt enrichment). Same degrade-gracefully contract.
    if action.arg_transformer is not None:
        try:
            transformed = await action.arg_transformer(dict(args), session, runtime)
            if isinstance(transformed, dict):
                args = transformed
        except Exception:
            log.warning("registry_action_transformer_failed", tool=call.name, exc_info=True)

    missing = [
        a for a in action.required_args
        if a not in args or args[a] in (None, "", [])
    ]
    if missing:
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(
                category="invalid_args",
                message=f"missing required arg: {missing[0]}",
            ),
        )

    _pi_before = getattr(getattr(session, "referents", None), "pending_intent", None)
    try:
        result = await asyncio.wait_for(
            action.handler("", session, args), timeout=_PER_TOOL_TIMEOUT_S,
        )
    except TimeoutError:
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(category="timeout", message="action timeout", retryable=True),
        )
    except Exception as exc:
        log.warning("registry_action_failed", tool=call.name, error=str(exc)[:200])
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(category="upstream_error", message=str(exc)[:200]),
        )

    if result is None:
        # Handler opted out (no referent, empty index, …) — tell the
        # model so it can try a different approach instead of pretending.
        return ToolResult(
            ok=False, tool=call.name, payload=None,
            error=ToolError(
                category="tool_self_error",
                message=f"{call.name} could not run (missing referent or precondition)",
            ),
        )

    if result.clarify:
        # Handler-level ask — park so next turn's answer fills the slot
        # (survives the _pi_before identity-check clear below: fresh dict).
        from augmentum.intent.dispatch import park_clarify
        park_clarify(
            getattr(session, "referents", None),
            action_id=call.name,
            args=args,
            clarify=result.clarify,
            question=result.speak,
        )
    else:
        # Results ring — verb outcomes decay to digests, not to nothing.
        from augmentum.companion_runtime import ring as _ring
        _ring.record_action_result(
            getattr(session, "referents", None),
            action_id=call.name, args=args, result=result,
        )

    # Compose what the model sees from speak/toast/addendum — same
    # folding as intent/tool_adapter.py so both Tier-3 paths read alike.
    bits: list[str] = []
    if result.speak:
        bits.append(result.speak)
    if result.toast and result.toast not in (result.speak or ""):
        bits.append(f"[{result.toast}]")
    if result.prompt_addendum:
        bits.append(result.prompt_addendum)
    payload: dict[str, Any] = {"content": "\n".join(b for b in bits if b) or "Done."}

    ui_effects: tuple[UIEffect, ...] = ()
    if result.surface_emit:
        channel = str(result.surface_emit.get("channel") or "")
        effect = UIEffect(
            kind=channel or "intent_action",
            target="_inline",
            payload=_payload_for_effect(result.surface_emit.get("payload") or {}),
        )
        ui_effects = (effect,)
        if runtime is not None and channel:
            try:
                await runtime.bus.publish_topic(
                    channel,
                    {
                        "target": effect.target,
                        "payload": effect.payload,
                        "source": "companion_tool_call",
                        "tool": call.name,
                        # Empty session_id marks a HEADLESS fire (scheduled
                        # verb_fire / timer then-action). The widget's bus
                        # forwarder dispatches ONLY those — live calls
                        # already reach the client via the per-session
                        # intent_action queue, and forwarding them too
                        # would double-dispatch every surface effect.
                        "session_id": session_id or "",
                    },
                    source_companion_id=runtime.companion_id,
                )
            except Exception:
                log.warning(
                    "registry_action_effect_publish_failed",
                    tool=call.name, kind=channel, exc_info=True,
                )
        # CLIENT DELIVERY — park the intent_action payload on the
        # per-session queue that the voice route drains into WS
        # ``intent_action`` events at turn end (same mechanism
        # intent/tool_adapter.py uses). Without this, voice-path
        # registry verbs executed server-side but their surface
        # effects never reached the browser: grove "played" with no
        # audio, the deliver pass narrated the phantom success, and
        # her false claims compounded through the transcript window
        # ("I just played 'Midnight Blue' by Chet Baker", 2026-06-10
        # — a track that never existed). The chat path doesn't need
        # this queue (ui_effects ride becca_tool_result chunks); a
        # duplicate dispatch can't happen because chat sessions never
        # reach the voice route's drain.
        refs = getattr(session, "referents", None)
        if refs is not None and channel:
            ws_payload: dict[str, Any] = {
                "type": "intent_action",
                "v": 1,
                "action": call.name,
                "tier": 3,
                "short_circuit": result.short_circuit,
                "surface": result.surface_emit,
            }
            if result.speak:
                ws_payload["speak"] = result.speak
            if result.toast:
                ws_payload["toast"] = result.toast
            refs.pending_surface_events.append(ws_payload)

    # Dispatch anchors — so "the thing you just did" resolves next turn.
    refs = getattr(session, "referents", None)
    if refs is not None:
        refs.last_dispatch_action = call.name
        refs.last_dispatch_args = dict(args)
        refs.last_dispatch_summary = (result.speak or result.toast or "")[:200]
        refs.last_dispatch_ts = time.time()
        # Successful dispatch resolves any parked clarification —
        # UNLESS this very handler just parked a new one (clarify
        # paths write a fresh dict; identity check distinguishes).
        if refs.pending_intent is _pi_before:
            refs.pending_intent = None
        # Continuity write-through — the working set (active note,
        # dispatch anchors) survives restarts and session-id churn.
        try:
            from augmentum.companion_runtime.working_state import (
                save_working_state,
            )
            await save_working_state(
                getattr(session, "app_state", None), user_id, refs,
            )
        except Exception:  # noqa: BLE001
            log.debug("working_state_save_failed", exc_info=True)

    return ToolResult(
        ok=True,
        tool=call.name,
        payload=payload,
        ui_effects=ui_effects,
        # Delivery class + raw speak line ride the metadata so the
        # voice consumer can honor artifact-delivery verbs (speak the
        # handler's own line verbatim, skip the synthesize pass)
        # without a second registry lookup.
        metadata={
            "delivery": action.delivery,
            "speak": result.speak or "",
        },
    )


def delivery_for_tool(name: str) -> str:
    """Delivery class for a tool name: ``"artifact"`` when the registry
    verb declares its feedback is the on-screen artifact, ``"verbal"``
    for everything else (including non-registry tools like web_search).
    """
    action = _lookup_registry_action(name)
    return getattr(action, "delivery", "verbal") if action is not None else "verbal"


def _replace_duration(result: ToolResult, duration_ms: int) -> ToolResult:
    return ToolResult(
        ok=result.ok, tool=result.tool, payload=result.payload,
        metadata=result.metadata, ui_effects=result.ui_effects,
        duration_ms=duration_ms, cancelled=result.cancelled, error=result.error,
    )


def _payload_for_effect(payload: Any) -> dict[str, Any]:
    """Coerce arbitrary payload into a JSON-serializable dict for bus."""
    if isinstance(payload, dict):
        return payload
    return {"raw": str(payload)[:512]}


def summarize_payload(payload: Any, *, cap: int = 1200) -> str:
    """Cap payload at ``cap`` chars and stringify for the synthesizer."""
    if payload is None:
        return "(no result)"
    if isinstance(payload, str):
        return payload[:cap]
    try:
        return json.dumps(payload, default=str, ensure_ascii=False)[:cap]
    except Exception:
        return str(payload)[:cap]


def map_error_to_failure_deck(category: str) -> str:
    """Translate a tool-error category to the ``affordances.FAILURE_DECKS``
    key for Becca's failure narration."""
    return {
        "timeout": "tool_timeout",
        "unauthorized": "primary_unreachable",
        "content_policy": "primary_unreachable",
        "model_unavailable": "primary_unreachable",
        "upstream_error": "primary_unreachable",
        "tool_self_error": "tool_self_error",
        "cancelled": "primary_unreachable",
        "invalid_args": "tool_self_error",
    }.get(category, "primary_unreachable")


__all__ = [
    "TOOL_CATALOG",
    "CHANNEL_CATALOG",
    "enumerate_tools",
    "enumerate_channels",
    "execute_tool",
    "summarize_payload",
    "map_error_to_failure_deck",
]
