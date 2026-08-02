"""Action shape — Action definition, IntentMatch result, SessionContext.

An ``Action`` is a single voice/text addressable verb. The registry
holds one of these per registered action; the matcher returns an
``IntentMatch`` that names which action fired and carries the args
the handler should receive. The handler runs against a
``SessionContext`` so it can read the per-session referent cache and
emit surface events without importing FastAPI / WebSocket plumbing.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


# Tier 3 fan-out — which dispatch surfaces an action participates in.
# An action can opt out of LLM tool exposure (e.g., conversation-repair
# actions like ``stop`` are pure surface controls, not capabilities the
# model should reason about). It can also opt out of Tier 1/2 (rare —
# only useful for actions whose disambiguation genuinely needs the LLM).
@dataclass(frozen=True)
class ActionFanout:
    tier1: bool = True      # Regex match on raw transcript
    tier2: bool = True      # Embedding similarity fallback
    tier3: bool = True      # Exposed as an LLM tool via UARF
    fast_path: bool = False # Frontend fast-path mirror eligible
    # Load-bearing: this verb is the SOLE path to a major capability, so it must
    # NEVER be silently clipped from the companion tool roster by the char
    # budget. The model can't choose a tool it was never shown — a budget-clipped
    # coder.delegate is why "launch a coder run" produced a chat reply and no run
    # (2026-07-27). Verbs with this set are budget- AND family-cap-exempt in
    # enumerate_tools. Keep the set SMALL — every always_offer verb permanently
    # spends roster budget.
    always_offer: bool = False


# Where the handler runs. Most actions emit a surface event; some need
# server-side work (memory writes, tool calls). The result tells the
# dispatcher what to do next.
@dataclass
class ActionResult:
    # If True, skip the LLM call entirely. The dispatch layer turns the
    # ``surface_emit`` payload into a WS message and (optionally) a
    # TTS-spoken ``speak`` line.
    short_circuit: bool = False

    # If set, dispatcher attaches this XML-ish tag to the user message
    # before mode routing. Soft-augmentation path — the LLM still
    # composes the reply but knows the intent + referent.
    prompt_addendum: str = ""

    # Surface-side action to emit (frontend handles via the WS
    # ``intent_action`` event router). Schema is per-action; see
    # ``ui/scripts/intent-action-router.js`` for known channels.
    surface_emit: dict[str, Any] | None = None

    # Text to speak via TTS when ``short_circuit`` is True. Empty =
    # silent action (e.g., ``stop``, ``nevermind``).
    speak: str = ""

    # Optional inline notification — surfaces a small chip in the
    # widget telling the user what happened without a TTS line. Useful
    # for fast-path control actions where speaking would add noise.
    toast: str = ""

    # Clarify marker — set when ``speak`` IS a question the handler
    # needs answered before it can act (e.g. weather.today's "what
    # city should I use?"). The dispatch layers park it as
    # ``ReferentCache.pending_intent`` so the user's ANSWER fills the
    # slot next turn instead of arriving as a bare, context-free
    # utterance the router has to re-derive (and often drops — a
    # one-word place name classifies as noise). Shape:
    # ``{"missing": ["location"], "args": {...overrides...}}`` —
    # ``missing`` names the slots the answer should fill; ``args``
    # (optional) overrides the parked partial args. Dispatcher-level
    # missing-required-args clarifies park without this field; it
    # exists for handler-level asks the dispatcher can't see into.
    clarify: dict[str, Any] | None = None

    # One-line digest for the results ring (companion_runtime/ring.py)
    # — what survives in her context for a few turns after this result.
    # INDEXICAL, not informational: name what exists ("weather fetched
    # for Springfield"), never half-enumerate specifics she didn't say
    # aloud (that's what gets confabulated). Empty = the dispatcher
    # falls back to ``speak``/``toast`` truncation.
    digest: str = ""

    # Whether the handler actually ACTUATED the user's intent. The
    # architect router pre-composes an optimistic spoken line
    # (``decision.response_text`` — "Playing X, say cancel if you'd
    # rather not") BEFORE the handler runs, then overrides the handler's
    # speak with it. That override is correct only when the action DID
    # the thing. A miss / empty result / parked clarification must set
    # this False so the router keeps THIS honest ``speak`` instead of
    # voicing a confirmation for something that never happened (the
    # companion lying: "Playing your favorites" over a library miss).
    # Default True: the action succeeded, or its ``speak`` IS the message.
    fulfilled: bool = True


@dataclass
class IntentMatch:
    """Result of a successful Tier 1 / Tier 2 match.

    ``args`` are extracted via named capture groups in the patterns
    (Tier 1) or by a small extractor model (Tier 2 — TBD). They flow
    into the action handler as kwargs.
    """

    action_id: str
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    tier: int = 1


# A handler receives the transcript + per-session context + extracted
# args, and returns an ActionResult. Async so handlers can touch the DB
# / emit through the event bus without spawning their own loop.
HandlerFn = Callable[
    [str, "SessionContext", dict[str, Any]],
    Awaitable[ActionResult | None],
]


# Architect inference — async callable that fills missing args from
# observation history (device_play_history, image_generations,
# browse_history, ReferentCache, etc.) BEFORE the handler runs. Lets
# "play jazz" land on the user's favourite Miles Davis track instead
# of asking which track. Returns the filled args dict.
ArgInferrer = Callable[
    [dict[str, Any], "SessionContext", Any],
    Awaitable[dict[str, Any]],
]

# Architect translation — async callable that REFORMS args between
# inference and handler dispatch. Where inference fills missing
# defaults, translation reshapes the raw user-derived values into
# well-formed tool input. The canonical example: turn "a dog" into
# a scene-rich image prompt ("a golden retriever puppy in soft
# sunlight, professional photography, shallow depth of field") so
# the downstream image model gets something it can actually paint.
#
# Receives args (post-inference) + session + runtime; returns the
# transformed args. Failures degrade gracefully — caller falls back
# to the untransformed args so a translator hiccup never blocks a
# dispatch.
ArgTransformer = Callable[
    [dict[str, Any], "SessionContext", Any],
    Awaitable[dict[str, Any]],
]


@dataclass
class Action:
    """One registered action.

    ``patterns`` are raw regex strings; the matcher compiles them once
    at registration. ``examples`` are the natural-language phrasings
    used both for embedding pre-compute (Tier 2) and for Discovery UI
    display ("try saying…"). Examples can ALSO seed auto-generated
    patterns if ``patterns`` is empty — the matcher escapes them and
    wraps with word boundaries so a low-effort action definition still
    catches the common case.

    ``modes`` filters by REQUEST MODE (passthrough/coder/narrative/...).
    ``surfaces`` filters by CLIENT SURFACE (voice/chat/cast/xr). The
    two are independent — an action might be available on voice +
    chat across every mode, or scoped to coder mode only on chat.
    Empty list = available everywhere on that axis.
    """

    id: str
    summary: str
    examples: list[str]
    handler: HandlerFn

    patterns: list[re.Pattern[str]] = field(default_factory=list)
    # Tier 1b — compiled hassil-style templates. Filled by
    # ``register_action`` from the ``templates`` kwarg. Each entry
    # is a ``CompiledTemplate`` (see augmentum/intent/templates.py)
    # whose regex slot names align with ``arg_schema`` keys.
    compiled_templates: list[Any] = field(default_factory=list)
    arg_schema: dict[str, Any] = field(default_factory=dict)
    # JSON Schema ``required`` array — names of args the LLM MUST
    # provide when calling this primitive. Empty = all args optional.
    required_args: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    fanout: ActionFanout = field(default_factory=ActionFanout)

    # Architect extensions ----------------------------------------------
    # Which CLIENT SURFACES expose this action. Common values:
    # 'voice', 'chat', 'cast', 'xr'. Empty = all surfaces.
    surfaces: list[str] = field(default_factory=list)

    # Optional async inferrer run between matcher hit and handler call.
    # Fills missing args from observation history. None = no inference,
    # partial args passed through unchanged.
    arg_inferrer: ArgInferrer | None = None

    # Optional async transformer run AFTER inference, BEFORE handler.
    # Reshapes raw user-derived args into well-formed tool input —
    # e.g. expanding "a dog" into a scene-rich image prompt. None =
    # no translation, args pass through unchanged.
    arg_transformer: ArgTransformer | None = None

    # If True, Becca's runtime may dispatch this primitive on her own
    # initiative (without an explicit user command). Default False —
    # most actions require an explicit verbal/text trigger.
    companion_initiatable: bool = False

    # Stakes class — drives confidence-tier dispatch (see
    # docs/superpowers/specs/2026-05-28-confidence-tier-dispatch-design.md).
    # The architect router uses this to decide whether high-confidence
    # intent gets just-do-it (Tier A) vs. confirm-then-act (Tier B) vs.
    # explicit yes/no (Tier C). The default is the safest assumption:
    # actions are reversible until proven otherwise. Annotate every new
    # primitive at registration. Allowed values:
    #   "trivial_reversible" — easy to undo, no resource cost
    #   "disruptive"         — interrupts current state (media, ambient)
    #   "costly"             — meaningful resource consumption (image gen)
    #   "personal"           — touches the user's data (gates on speaker
    #                          verification)
    #   "irrevocable"        — cannot be cleanly undone (send, post, share)
    #   "safety_critical"    — real-world effect (smart home, calls, pay)
    stakes: str = "trivial_reversible"

    # Delivery class — how the COMPANION VOICE path should treat this
    # verb's feedback (2026-06-10 co-author register fix):
    #   "verbal"   — default. Voice covers latency with an affordance
    #                line and confirms the outcome through the
    #                synthesize pass. Right for slow gather tools and
    #                anything with no visible artifact.
    #   "artifact" — the screen artifact (sticky note, opened surface)
    #                IS the feedback. Voice skips the latency affordance
    #                and the synthesize confirmation; only the handler's
    #                own ``speak`` line (if any) is voiced. Right for
    #                sub-second writes the user watches land — the
    #                Obsidian co-author register: keep talking about
    #                the content, not the mechanics.
    delivery: str = "verbal"

    def available_in(self, mode: str | None) -> bool:
        if not self.modes:
            return True
        return mode in self.modes

    def surfaces_for(self, surface: str | None) -> bool:
        if not self.surfaces:
            return True
        return surface in self.surfaces


# Minimal session context — the dispatch layer constructs one per turn
# and hands it to handlers. The referent cache fields stay None until
# Phase 8 wires their writers; handlers should always None-check.
@dataclass
class ReferentCache:
    """Per-session "what we were just talking about" anchors.

    Populated by tool/result emitters (image_generation writes
    ``last_image``, web_fetch writes ``last_url``, etc.) so the model
    and the user can both reference them by demonstrative ("show that",
    "open it again") without naming the resource explicitly.
    """

    last_image_id: str | None = None
    last_image_title: str | None = None
    last_url: str | None = None
    last_quote: str | None = None
    last_file_id: str | None = None
    last_entity: str | None = None
    # Active note for the current conversation. ``create_note`` writes
    # here; ``append_to_note`` defaults to this when ``note_id`` is
    # omitted; ``show_sticky`` surfaces it.
    active_note_id: str | None = None
    active_note_title: str | None = None
    # Capture mode — when True, normal turns are intercepted and the
    # transcript is appended to ``active_note_id`` after a "shape this
    # into notes" LLM pass rather than running as a free-form
    # conversation. Toggled by ``start_note_capture`` /
    # ``end_note_capture`` primitives.
    note_capture_mode: bool = False
    # Auto-exit timestamp — monotonic seconds when capture mode should
    # auto-exit if no activity. Set by start_note_capture, refreshed
    # on each capture-append, cleared on end_note_capture.
    note_capture_deadline: float = 0.0
    # Pending intent_action WS payloads from LLM-invoked actions
    # waiting to flush to the active voice session. The chain layer
    # has no WS handle, so ActionTool.execute appends here and the
    # voice route drains at turn boundaries. List discipline is FIFO.
    pending_surface_events: list[dict] = field(default_factory=list)
    # Idempotency: track recently-created note ids by their
    # (title, content) fingerprint so a model retry doesn't duplicate.
    # FIFO-bounded to 32 entries (older entries fall off).
    recent_note_fingerprints: dict[str, str] = field(default_factory=dict)
    # Architect dispatch anchors — set by the dispatcher (generic
    # last_dispatch_*) and per-primitive (e.g. last_played_track,
    # last_image_prompt) so follow-up references ("the one I played",
    # "make another like that") have something to resolve against.
    last_dispatch_action: str | None = None
    last_dispatch_args: dict[str, Any] | None = None
    last_dispatch_summary: str | None = None
    last_dispatch_ts: float = 0.0
    last_played_track: str | None = None     # grove.play_matching writes
    last_played_query: str | None = None
    last_image_prompt: str | None = None     # image.generate_with_defaults writes
    # Media disambiguation parking — media.play's "offer" decision
    # stores its candidate payloads here so a follow-up turn ("the
    # second one", "the Johansson one") can resolve against them
    # without re-searching. Replaced wholesale on each offer; cleared
    # when a play dispatch succeeds.
    pending_candidates: list[dict] = field(default_factory=list)
    # When pending_candidates were offered (time.time()). The architect
    # router only re-presents an offer within a freshness TTL — a stale
    # offer (minutes old) must NOT be silently reused for a later, often
    # UNRELATED request (e.g. an audiobook recommend getting replayed for
    # "throw in some music"). 0.0 = no live offer.
    pending_candidates_at: float = 0.0
    # Generic offer metadata so a spoken/typed accept ("the second one")
    # re-dispatches the RIGHT verb with the RIGHT arg — not the hard-coded
    # media.play/file_id the router used to assume. Set by the offering verb
    # alongside pending_candidates; empty = legacy media.play/file_id default
    # (so existing offerers — media/livetv/games — are unchanged). coder.delegate
    # sets ("coder.delegate", "workspace_id"); each candidate payload carries
    # that id field.
    pending_candidates_intent: str = ""
    pending_candidates_id_field: str = ""
    # Parked intent — a clarifying question PARKS the verb it was
    # asked for, so the user's answer FILLS the slot instead of
    # re-deriving (and possibly re-gambling) the whole intent.
    # Shape: {"action_id", "args" (partial), "missing" (list[str]),
    # "question" (what was asked), "asked_at" (time.time())}.
    # Consumed by the architect router via its confidence stack;
    # cleared on any successful dispatch or freshness TTL.
    pending_intent: dict | None = None
    # Her trail — positions she visited while working headlessly
    # (searches run, pages read). "Take me there" jumps the user to
    # the latest entry. Appended + capped (20) by
    # companion_runtime/native_loop.py::_append_trail.
    # Entry shape: {kind, label, ref, ts}.
    trail: list[dict] = field(default_factory=list)
    # Results ring — recent tool/perception results with turn-based
    # decay (full on the turn they ran, digest for a few turns,
    # pull-only after). Written via companion_runtime/ring.py only.
    # Entry shape: {kind, slot, label, digest, refetch, born_turn,
    # touch_turn}. Capped (4) with same-slot supersede.
    results_ring: list[dict] = field(default_factory=list)
    # Per-conversation turn counter — bumped once per USER turn by the
    # prompt-compose path (shared by voice + becca_direct). The ring's
    # decay clock; distinct from wall-clock TTLs in the AttentionStore.
    turn_seq: int = 0
    # Eviction watermark — monotonic seconds of the last get_referent_cache
    # touch. Used by the lazy TTL sweeper to identify stale per-session
    # caches that can be safely dropped. Initialised to 0.0 so a freshly-
    # constructed cache is touched on first get; production paths always
    # go through get_referent_cache so the field is current by the time
    # any handler reads it.
    last_touched: float = 0.0
    # Capture-mode baseline — char offset into the active note's content
    # at the moment note_capture_mode was switched on. The end-of-capture
    # cleanup pass slices ``content[note_capture_baseline_chars:]`` so it
    # only rewrites what was dictated, leaving pre-capture content alone.
    # 0 (default) means "no baseline recorded" and the cleanup pass is
    # skipped, which is the legacy-safe fallback.
    note_capture_baseline_chars: int = 0


@dataclass
class SessionContext:
    user_id: str
    session_id: str
    mode: str | None = None
    referents: ReferentCache = field(default_factory=ReferentCache)
    # Set when the call site has access to the FastAPI app; handlers
    # that need the event bus / DB use this. Avoiding a hard import so
    # this module stays tree-shakeable for tests.
    app_state: Any = None
