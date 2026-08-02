"""Internal canonical types and abstract model backend."""

from __future__ import annotations

import base64 as _b64
import re as _re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Message:
    role: str
    content: str
    images: list[str] | None = None
    tool_calls: list[dict] | None = None
    thinking: str | None = None
    tool_call_id: str | None = None


# Field names ``Message`` knows about. Used by the dict-coercion in
# ``InternalChatRequest.__post_init__`` to tolerate extra keys on incoming
# dict-shaped messages without TypeError-ing the construction.
_MESSAGE_FIELDS = frozenset(("role", "content", "images", "tool_calls",
                             "thinking", "tool_call_id"))


def _coerce_message(m: object) -> Message:
    """Convert ``m`` into a ``Message`` if it isn't one already.

    Accepts: an existing ``Message`` (passthrough) or a dict with at
    minimum ``role`` and ``content``. Unknown dict keys are dropped — the
    Message dataclass has a fixed shape and silently-passing-through
    unrelated keys would mask schema drift.

    Anything else passes through unchanged; the backend adapter will raise
    on first attribute access, which gives a clearer error than a
    construction-time TypeError that swallows the original message data.
    """
    if isinstance(m, Message):
        return m
    if isinstance(m, dict):
        return Message(**{k: v for k, v in m.items() if k in _MESSAGE_FIELDS})
    return m  # type: ignore[return-value]


@dataclass
class InternalChatRequest:
    """Internal LLM request shape, shared across modes and backends.

    Field-addition warning
    ----------------------
    When you add a field that should propagate from the user's incoming
    request through to the backend (e.g. routing flags, KV-cache hints,
    feature toggles), audit every transform site BEFORE merging. Sites
    that take a request and produce a modified one MUST use
    ``dataclasses.replace(request, ...)`` rather than constructing a
    fresh ``InternalChatRequest(...)`` with an explicit field list —
    otherwise your new field will be silently dropped on every
    transform path.

    The known transform sites (these all use ``dataclass_replace``):
    - ``augmentum/modes/narrative/engine.py::_augment_request``
    - ``augmentum/modes/narrative/prompt_presets.py::apply_preset``
    - ``augmentum/models/llama_cpp.py::_checkpoint_request_from_messages``

    Sites that construct fresh internal requests (memory extraction,
    dream summary, image distiller, agentic phases, analytical UARF
    phases, coder plan/act phases, group-chat director, etc.) are
    intentionally NOT carrying user context — those stay as explicit
    constructions because the fields shouldn't propagate through them.

    Bug class this comment guards against: 731a96d (KV checkpoint
    silently no-ops because ``apply_preset`` dropped
    ``kv_stable_messages``). Took multiple investigation sessions to
    find because the silent-drop was indistinguishable from "the field
    was never set" at the consumer site.
    """
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    # llama.cpp / local sampling knob; not a standard OpenAI field. Left None
    # for cloud calls (real OpenAI 400s on it) — only the local classifier
    # path sets it, where the backend is a llama-server that accepts it.
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    tools: list[dict] | None = None
    # OpenAI tool_choice: "auto" (default — model decides), "required"
    # (force any tool call), "none" (disable tools), or
    # ``{"type":"function", "function":{"name":"..."}}`` to pin a
    # specific tool. Native-strategy coder normally leaves this unset so
    # the model can decide when to call tools and when to stop with
    # visible prose.
    tool_choice: str | dict | None = None
    # Per-request chat-template kwargs forwarded to llama-server's
    # ``chat_template_kwargs`` payload field. Models like Qwen 3.x and
    # GLM-4.x use this to toggle reasoning ("enable_thinking": false
    # makes Qwen go straight to tool_calls without burning tokens on
    # chain-of-thought). Native-strategy coder uses this to disable
    # thinking on tool-call iterations.
    chat_template_kwargs: dict | None = None
    format: str | None = None
    keep_alive: str | None = None
    # Ollama-specific options passthrough
    raw_options: dict | None = None
    # Enable thinking/reasoning mode (Ollama think=true, etc.)
    think: bool = False
    # Reasoning-effort budget for OpenAI-family models (GPT-5.x / o1 /
    # o3 / xAI Grok). Valid: "minimal" | "low" | "medium" | "high" |
    # "xhigh". None = let the backend's default apply (medium on
    # OpenAI). The openai_compat adapter gates transmission via the
    # provider's ``supports_reasoning_effort`` flag + the OpenAI-family
    # model-id catch-all, so passing a value never reaches a provider
    # that would 400 on the unknown field.
    reasoning_effort: str | None = None
    # Preserve <think> traces across multi-turn history. Per-request override
    # for the Qwen 3.6 ``preserve_thinking`` chat-template kwarg. None falls
    # back to the per-user UI setting (or off if neither is set). Other
    # families' templates ignore this kwarg.
    preserve_thinking: bool | None = None
    # Lightweight memory availability hint for analytical mode
    memory_hint: str | None = None
    # True when the last user message came from speech-to-text transcription
    voice_input: bool = False
    # Explicit flow selection (from /flow command or header)
    explicit_flow_name: str = ""
    # Lorebook entries sourced from the UI session (narrative mode). Raw dicts;
    # the narrative handler feeds them into LoreEngine.replace_entries_preserving_state
    # each turn. None means "no UI lorebook" — falls back to card-embedded character_book.
    lorebook: list[dict] | None = None
    # Group chat id from X-Augmentum-Group-Id header. When present, the narrative
    # handler activates GroupTurnManager + per-turn speaker card swap. Empty =
    # single-character chat.
    group_id: str = ""
    # One-shot speaker pin from X-Augmentum-Speaker header. When present in a
    # group turn, overrides rotation / random / llm_decide for this single
    # request. Releases after the turn completes (no permanent state change).
    speaker_override: str = ""
    # Stable Augmentum session key used by the built-in engine's KV persistence.
    # Prefer this over prompt-derived hashes whenever the route layer knows the
    # real UI chat/session identity.
    kv_session_key: str = ""
    # Stable, pre-augmentation history for checkpoint-oriented KV reuse.
    # Narrative mode uses this to rebuild a prefix checkpoint from the base
    # chat history while treating dynamic lore/archive injections as a tail.
    kv_stable_messages: list[Message] | None = None
    # High-level mode hint used for retention policies (for example, letting
    # narrative sessions live longer than lightweight helper chats).
    kv_mode: str = ""
    # Knowledge-pack injection metadata. Set by inject_pack_context() when
    # retrieval was *attempted* (pack bound + mode enabled + query present).
    # The streaming layer surfaces this on the final SSE chunk so the UI can
    # render a "📚 Searched X — N sources" chip in the message footer.
    # None when retrieval was not attempted (no binding, mode off, etc.).
    pack_injection: dict | None = None
    # True when this request is fired by a background task (memory refresh,
    # ledger compaction, narrative extraction, dream cycle, etc.) rather than
    # the user-facing chat. Backends that support multi-slot routing (e.g.
    # LlamaCppBackend with ``engine_multislot_enabled``) use this hint to
    # route the request to a slot OTHER than slot 0, so the background work
    # doesn't queue behind the user's chat or block the next turn waiting
    # for slot 0's lock. Backends without slot routing ignore this flag —
    # behavior is identical to ``False``.
    is_background_task: bool = False
    # Training data generation mode. When True, tool pre-filtering is
    # bypassed so the model sees every available tool regardless of query
    # relevance. Set from `X-Augmentum-Training: true` header.
    training_mode: bool = False
    # True when the request is a "continue the trailing assistant message"
    # turn (Continue button in the chat UI). The last message in
    # ``messages`` is a partial assistant turn the model should extend
    # verbatim — no re-intro, no preamble, no fresh turn marker. Each
    # backend implements this differently:
    #   - LlamaCppBackend: ``chat_template_kwargs={"add_generation_prompt":
    #     False, "enable_thinking": False}`` so the chat template formats
    #     the assistant message without the usual "now-generate" suffix.
    #   - OpenAICompatBackend: providers with ``supports_assistant_prefix``
    #     (DeepSeek's ``/beta`` endpoint) get the profile's
    #     ``assistant_prefix_marker`` merged onto the last assistant dict
    #     (``{"prefix": True}`` for DeepSeek). Providers without prefix
    #     support fall back to a synthetic user "continue from where you
    #     left off" message appended after the partial.
    #   - ClaudeBackend: Anthropic's Messages API natively continues a
    #     trailing assistant turn. Models flagged by ``is_no_prefill_model``
    #     fall back to the synthetic-user path. Trailing whitespace on the
    #     partial is stripped per Anthropic's API constraint.
    continue_last_assistant: bool = False

    def __post_init__(self) -> None:
        """Coerce dict-shaped messages into ``Message`` instances.

        Many call sites historically pass raw dicts to ``messages=`` — the
        ergonomic pattern is ``{"role": "user", "content": "..."}``, and
        plenty of behavior modules (activity_selector, today, honest_gap,
        dispatch, every subagent) do it that way. The dataclass typing
        promises ``list[Message]`` but doesn't coerce at construction the
        way Pydantic would; without this hook the dicts leak through to
        backend adapters that do ``msg.tool_call_id`` and surface as
        ``AttributeError: 'dict' object has no attribute 'tool_call_id'``,
        which is opaque about its actual cause.

        Coerces both ``messages`` and ``kv_stable_messages`` for the same
        reason — the narrative checkpoint path could trip the same bug if
        a caller ever passes dicts there.
        """
        if self.messages:
            self.messages = [_coerce_message(m) for m in self.messages]
        if self.kv_stable_messages:
            self.kv_stable_messages = [
                _coerce_message(m) for m in self.kv_stable_messages
            ]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Cache telemetry. Populated by providers that report per-request KV
    # cache stats: DeepSeek (``prompt_cache_hit_tokens`` /
    # ``prompt_cache_miss_tokens`` in the response usage block — cache
    # hits bill 1/10 of misses, so this is load-bearing for cost math)
    # and our local llama-server (``cache_n`` from the timings block —
    # already surfaced as ``augmentum.prompt_tokens_cached``; this is
    # the typed sibling). 0 means "provider didn't report" (most cloud
    # OAI-compat backends), not "no cache hits".
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    # Cache WRITES — prompt tokens the provider ingested to CREATE a cache
    # entry, as opposed to reading one. Anthropic reports these as
    # ``cache_creation_input_tokens`` and bills them at ~1.25× fresh input;
    # Bedrock calls them ``cacheWriteInputTokens``. Folding writes into
    # ``cache_miss_tokens`` (the old behaviour) makes a thrashing cache —
    # one that re-creates every turn and never reads — look identical to
    # having no cache at all, when it is in fact 25% MORE expensive. Kept
    # separate so that failure mode is visible. 0 means "provider doesn't
    # report writes" (OpenAI, DeepSeek, llama-server); those providers
    # create cache entries for free, so there is nothing to report.
    cache_write_tokens: int = 0
    # Reasoning/CoT token count, sibling of completion_tokens. OpenAI
    # ships this as ``completion_tokens_details.reasoning_tokens`` on
    # reasoning models (o1, o3, gpt-5.x); DeepSeek V4 mirrors the same
    # shape. Useful for cost math (some providers bill reasoning at the
    # output rate but separately track it) and for UI displays that want
    # to show "model thought for N tokens".
    reasoning_tokens: int = 0
    # Authoritative server-side DECODE wall-time in milliseconds, when the
    # backend reports it (llama-server ``timings.predicted_ms``). 0 means
    # "not reported". The proxy's _StreamTimer measures gen rate from the
    # wall-clock gap between the first and last token it observes; that
    # collapses to ~microseconds when a backend delivers the whole
    # completion in a single burst (e.g. a CPU llama-server that finishes
    # prefill, then emits every token at once), producing nonsense rates
    # (131 tok / 6µs ≈ 20M tok/s). When this field is present, stats()
    # divides by the real decode time instead, so the reported tok/s is
    # the true server-side rate regardless of delivery shape.
    eval_duration_ms: float = 0.0


@dataclass
class InternalChatResponse:
    message: Message
    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    # Ollama timing stats passthrough
    timing: dict | None = None


def response_text(
    resp: InternalChatResponse | None,
    *,
    thinking_fallback: bool = True,
) -> str:
    """Read the visible text content from a chat response, defensively.

    Background: ``InternalChatResponse.message`` is where ``content``
    lives — NOT ``resp.content`` (which doesn't exist). Multiple call
    sites historically used ``getattr(resp, "content", "")`` and got
    empty strings forever, silently discarding every LLM response.
    This helper centralises the right access so the bug can't recur.

    Args:
        resp: an InternalChatResponse, or None (e.g. when the call
            failed before construction). None returns ''.
        thinking_fallback: when True (default) AND the model emitted
            only reasoning tokens (chat-template ignored
            ``enable_thinking=false``), the reasoning text is returned
            so the work isn't lost. Set False for paths where you
            specifically need post-think output and want empty on
            think-only emissions.

    Returns the stripped content (or thinking fallback) as a string.
    Never raises — defensive against the response shape varying
    across backends.
    """
    if resp is None:
        return ""
    msg = getattr(resp, "message", None)
    if msg is None:
        return ""
    raw = getattr(msg, "content", "")
    content = raw.strip() if isinstance(raw, str) else ""
    if content:
        return content
    if not thinking_fallback:
        return ""
    raw_think = getattr(msg, "thinking", "")
    if isinstance(raw_think, str):
        return raw_think.strip()
    return ""


@dataclass
class InternalStreamChunk:
    content_delta: str = ""
    thinking_delta: str = ""
    role: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str = ""
    done: bool = False
    # Augmentum metadata (UARF phase indicators, mode info)
    augmentum: dict | None = None


@dataclass
class ModelInfo:
    name: str
    model: str
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    details: dict | None = None
    vision: bool = False
    context_length: int = 0  # Model's max context window (0 = unknown)
    max_output: int = 0  # Model's max completion/output tokens (0 = unknown)
    mtp: bool = False  # GGUF advertises built-in MTP / next-N predict heads


@dataclass
class ModelDetails:
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: dict | None = None
    model_info: dict | None = None
    # Extended fields for llama.cpp and multi-backend support
    format: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""
    system_prompt: str = ""


# --- Vision prompt injection ---

# Short/empty user text that doesn't meaningfully describe what to do with an image.
_TRIVIAL_TEXT_MAX_LEN = 12

_VISION_DEFAULT_PROMPT = (
    "Describe and analyze the attached image in detail. "
    "Note key subjects, context, text, colors, and anything notable."
)

_VISION_ENHANCE_NOTE = (
    "\n\n[An image is attached. Include observations about the image in your response.]"
)


def inject_vision_prompt(req: InternalChatRequest) -> None:
    """Ensure the last user message has adequate prompting when images are attached.

    - If the user sent images with no text or trivial text (e.g. "hey", "hi"),
      replace with a descriptive analysis prompt.
    - If the user sent images with short text that looks like a greeting rather
      than an instruction, append a gentle nudge to also address the image.
    - If the user wrote a real question/instruction, leave it alone entirely.
    """
    for msg in reversed(req.messages):
        if msg.role != "user":
            continue
        if not msg.images:
            return  # No images on last user message — nothing to do

        text = (msg.content or "").strip()

        if not text:
            # Empty text + image → supply a default analysis prompt
            msg.content = _VISION_DEFAULT_PROMPT
        elif len(text) <= _TRIVIAL_TEXT_MAX_LEN and not _looks_like_instruction(text):
            # Very short greeting-like text + image → append a hint
            msg.content = text + _VISION_ENHANCE_NOTE
        # Otherwise: user wrote a real prompt, leave it as-is
        return


def _looks_like_instruction(text: str) -> bool:
    """Return True if short text looks like it's directing the model about the image."""
    lower = text.lower().rstrip("?!.")
    # Question words and action verbs that suggest the user wants something specific
    instruction_starts = (
        "what", "who", "where", "when", "why", "how",
        "describe", "explain", "analyze", "read", "translate",
        "extract", "identify", "compare", "summarize", "tell",
        "list", "find", "count", "ocr", "transcribe", "solve",
    )
    return any(lower.startswith(w) for w in instruction_starts)


# --- Chat image URL resolution ---

_CHAT_IMAGE_PREFIX = "/api/chat-images/"


async def resolve_chat_image_urls(req: InternalChatRequest, app_state: object) -> None:
    """Replace /api/chat-images/<id> URLs with inline base64 data URLs.

    Called in the route layer before the request reaches model backends,
    so both Ollama and OpenAI backends receive data URLs they already handle.
    """
    # Collect all image IDs that need resolving
    ids_to_resolve: dict[str, list[tuple[Message, int]]] = {}
    for msg in req.messages:
        if not msg.images:
            continue
        for idx, img in enumerate(msg.images):
            if isinstance(img, str) and img.startswith(_CHAT_IMAGE_PREFIX):
                image_id = img[len(_CHAT_IMAGE_PREFIX):]
                ids_to_resolve.setdefault(image_id, []).append((msg, idx))

    if not ids_to_resolve:
        return

    # Get DB connection
    backend = getattr(app_state, "state_manager", None)
    conn = getattr(backend, "_backend", None) if backend else None
    if not conn:
        conn = getattr(app_state, "sqlite_backend", None)
    if not conn:
        return

    db = getattr(conn, "_conn", conn)

    for image_id, locations in ids_to_resolve.items():
        try:
            async with db.execute(
                "SELECT mime_type, data FROM chat_images WHERE id = ?", (image_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                continue
            mime_type, raw_data = row
            b64 = _b64.b64encode(raw_data).decode("ascii")
            data_url = f"data:{mime_type};base64,{b64}"
            for msg, idx in locations:
                msg.images[idx] = data_url
        except Exception as exc:
            # Leave the URL as-is — backend will get a 404-like string.
            log.debug("chat_image_url_resolve_failed", error=str(exc))


# --- Vision caption fallback ---

_CAPTION_FALLBACK_PROMPT = "Describe this image in one or two short sentences."
# Raised from 96: the structured SEES/MAIN caption needs room to emit the
# inventory line even when there's no user question (bake-off 2026-06-18).
_CAPTION_FALLBACK_MAX_TOKENS = 160
# Query-conditioned captions carry more (the reasoning primary needs enough
# concrete detail to answer the user's actual question, not a one-liner).
_CAPTION_QUERY_MAX_TOKENS = 220
_CAPTION_FALLBACK_TIMEOUT_S = 20.0


def _caption_prompt_for(question: str, *, multi: bool, live_camera: bool = False) -> str:
    """Build the captioner instruction — one structured, grounded prompt
    shared by every captioner model (SmolVLM sibling + Gemma classifier)
    so their outputs stay consistent and resist confabulation.

    Empirically tuned in the 2026-06-18 captioner consistency bake-off:
    the structured ``SEES:``/``MAIN:`` shape plus an explicit "report only
    what you can see, never invent" instruction roughly doubled run-to-run
    consistency and stopped Gemma embellishing settings/brands/backstory
    that then poison downstream context and memory. Paired with the low
    caption sampling in ``provider.py::_CAPTION_SAMPLING``. The ``SEES``
    inventory preserves completeness (nothing dropped when captioning for a
    text-only model); ``MAIN`` keeps the answer focused on what the user
    asked — both halves of the relevance-vs-completeness tradeoff at once.

    ``live_camera`` asserts the frames are a REAL object/scene the user is
    physically showing right now — anti-narrative grounding (a held-up
    bottle was once waved off as "fake"). ``question`` is folded in so MAIN
    answers what the user actually asked when the answer is visible.
    ``multi`` (a live clip) needs no wording change: the frames ride in as
    one sequence and the structure is identical.
    """
    q = (question or "").strip()
    if live_camera:
        opener = (
            "You are describing what the user is showing on their live camera. "
            "It is a REAL object or scene physically present right now — never "
            "fiction, a prop, or a story."
        )
        main_hint = "the main subject the user is showing"
    else:
        opener = "Describe this image for someone who cannot see it."
        main_hint = "the main subject of the image"
    lines = [
        opener,
        "Report ONLY what you can actually see. Read any visible text, brand, "
        "or label word-for-word. If you cannot tell what something is, write "
        '"unclear" rather than guessing — do NOT invent objects, brands, '
        "settings, or backstory.",
    ]
    if q:
        lines.append(
            f'The user asked: "{q[:300]}" — make sure MAIN answers it if the '
            "answer is visible in the frame."
        )
    lines.append(
        "Answer in exactly two lines:\n"
        "SEES: <comma-separated list of every distinct object / person / "
        "visible text>\n"
        f"MAIN: <one or two sentences on {main_hint}, with its key visible "
        "details>"
    )
    return "\n".join(lines)


async def caption_via_router_fallback(
    req: InternalChatRequest,
    app_state: object,
    backend: object | None,
    *,
    live_camera: bool = False,
) -> int:
    """When the resolved backend can't natively read images, caption the
    attachment(s) via the vision router and inline the result into the
    message text — then strip the image attachments so the request becomes
    a pure text chat completion.

    No-op when:
      - The request has no images.
      - ``backend.is_vision_paired()`` returns True (backend handles
        vision natively — cloud VL, llama-server with mmproj).
      - ``app_state.vision_router`` is missing or has no available provider.

    Two quality behaviors for the live-camera path:
      - **Query-conditioned**: the user's turn text is handed to the
        captioner so it describes what's relevant to the question.
      - **Multi-frame**: when a message carries several frames (a live
        clip), they're captioned as ONE sequence (video understanding on
        Gemma) → a single combined description, not N independent stills.

    Returns the number of images that were captioned + stripped. Workload
    hint is ``INTERACTIVE`` — the user is watching, latency matters.
    """
    has_images = any(getattr(m, "images", None) for m in req.messages)
    if not has_images:
        return 0
    if backend is not None:
        paired = getattr(backend, "is_vision_paired", None)
        if paired is not None:
            try:
                native = paired(req.model)
            except TypeError:
                native = paired()  # legacy no-arg backends
            if native:
                return 0

    router = getattr(app_state, "vision_router", None)
    # Any available provider can caption (classifier/Gemma OR SmolVLM OR a
    # fallback primary) — gating on SmolVLM specifically would skip the
    # Gemma-as-captioner path entirely when the sibling is off.
    if router is None or not await router.is_available():
        return 0

    from augmentum.vision.router import Workload

    captioned = 0
    for msg in req.messages:
        if not msg.images:
            continue
        # Decode every attached frame up front (skip the unreadable ones).
        raws: list[bytes] = []
        for img in msg.images:
            if not isinstance(img, str) or not img.startswith("data:"):
                continue
            try:
                _, b64_data = img.split(",", 1)
                raws.append(_b64.b64decode(b64_data))
            except Exception:
                continue

        original = (msg.content or "").strip()
        if not raws:
            # All attachments unreadable — leave a marker, drop the images.
            block = "[image attachment unavailable]"
            msg.content = f"{block}\n\n{original}" if original else block
            msg.images = None
            continue

        multi = len(raws) > 1
        text = await router.caption(
            raws[0],
            prompt=_caption_prompt_for(original, multi=multi, live_camera=live_camera),
            max_tokens=_CAPTION_QUERY_MAX_TOKENS if original else _CAPTION_FALLBACK_MAX_TOKENS,
            timeout_s=_CAPTION_FALLBACK_TIMEOUT_S,
            workload=Workload.INTERACTIVE,
            frames=raws[1:] or None,
        )
        captioned += len(raws)
        caption = text.strip() if text else "[no caption available]"
        # Live-camera framing asserts REALITY: the model otherwise reads a
        # bare "[Image: …]"/"[Scene: …]" prefix as an incidental or fictional
        # prop and can dismiss a real object as "fake" / narrative (seen live
        # 2026-06-18 — a held-up hot-sauce bottle called fake). Naming it as
        # the user's live camera, right now, grounds the turn in the real
        # world. Non-camera attachments (pasted chat images) keep the plain
        # label — they genuinely may be illustrative, not live.
        if live_camera:
            label = (
                "User's live camera right now (a few frames)"
                if multi
                else "User's live camera right now"
            )
        else:
            label = "Scene" if multi else "Image"
        block = f"[{label}: {caption}]"
        # Prepend the visual context so the model sees it before the question.
        msg.content = f"{block}\n\n{original}" if original else block
        msg.images = None

    if captioned:
        log.info("vision_caption_fallback_applied", images=captioned)
    return captioned


# Single source of truth for the "you can see live, and it's REAL" grounding.
# Used two ways: (1) prompt_compose Layer 8.6 folds it into the companion's
# composed system prompt (the becca surfaces, gated on
# intent.metadata['live_camera']); (2) ``ensure_live_camera_framing`` below
# injects it as a system message on the surfaces that do NOT compose a
# companion prompt (passthrough / narrative voice calls). Keeping ONE string
# means the anchor can't drift between paths.
LIVE_CAMERA_SYSTEM_NOTE = (
    "RIGHT NOW you can see through the user's live camera — they're pointing "
    "it at something or holding it up to show you, in the real world, in real "
    "time. Anything from their live camera is a REAL object or scene physically "
    "in front of them, never fiction, a prop, or part of a story. React to what "
    "they're actually showing you."
)


def ensure_live_camera_framing(req: InternalChatRequest) -> None:
    """Prepend the live-camera grounding to the request's system message.

    The captioned (text-only-primary) path already carries this reality
    anchor inside the ``[User's live camera right now: …]`` label, but a
    VISION-CAPABLE primary reads the frames DIRECTLY — no caption, no label —
    so on surfaces that don't compose a companion prompt (passthrough /
    narrative voice calls) it would otherwise get raw pixels with zero
    framing and can dismiss a real object as fiction (the 2026-06-18 "that
    hot sauce is fake" class). Companion surfaces get the same anchor from
    prompt_compose Layer 8.6 instead. Idempotent — safe to call on every
    live-camera turn regardless of VL-vs-text-only.
    """
    note = LIVE_CAMERA_SYSTEM_NOTE
    for m in req.messages:
        if m.role == "system":
            if note in (m.content or ""):
                return
            m.content = f"{note}\n\n{m.content}" if m.content else note
            return
    req.messages.insert(0, Message(role="system", content=note))


async def apply_vision_pipeline(
    req: InternalChatRequest,
    app_state: object,
    backend: object | None,
    *,
    reason_on_vision: bool = False,
    live_camera: bool = False,
) -> None:
    """Canonical image-handling sequence for an outbound chat request.

    Runs the three steps the chat routes (``openai_routes`` /
    ``ollama_routes``) apply to every turn, in order:

      1. :func:`resolve_chat_image_urls` — expand stored ``/api/chat-images``
         refs to inline base64 the backend can read.
      2. :func:`caption_via_router_fallback` — when the resolved backend
         can't natively read images, caption each via the vision sibling
         and inline the result as text (so a text-only primary still
         responds usefully). A VL primary is left untouched → image direct.
      3. :func:`inject_vision_prompt` — ensure an image-bearing turn has a
         usable instruction when the user supplied none.

    Call this from any surface that lets the companion RECEIVE an image —
    the voice turn, and the live camera frame loop — so the
    VL-primary-direct vs sibling-caption behavior is identical everywhere
    instead of re-implemented per surface. No-op for image-free requests.

    ``reason_on_vision``: when True and this turn actually carried frames,
    unlock the responding brain's reasoning (``req.think = True``) — the
    "what do you think of this?" moment deserves real judgment. Captioning
    already ran instruct (the low-latency, high-repetition role), so this
    only lifts the ANSWER's reasoning, and it's a no-op on non-thinking
    models (``think`` is ignored there). Off by default so non-live
    surfaces keep their own thinking policy.
    """
    had_images = any(getattr(m, "images", None) for m in req.messages)
    await resolve_chat_image_urls(req, app_state)
    await caption_via_router_fallback(req, app_state, backend, live_camera=live_camera)
    inject_vision_prompt(req)
    if reason_on_vision and had_images:
        req.think = True


# --- Vision model detection ---

# Patterns that identify VL/multimodal models by name.
# Checked case-insensitively against the model name string.
_VL_NAME_PATTERNS: list[_re.Pattern] = [
    _re.compile(p, _re.IGNORECASE) for p in (
        # --- Local model names ---
        r"llava",
        r"llama.*vision",
        r"qwen2?[._-](?:5[._-])?vl",
        r"minicpm[._-]?v",
        r"cogvlm",
        r"internvl",
        r"pixtral",
        r"molmo",
        r"gemma[._-]3",           # Gemma 3+ are multimodal
        r"phi[._-](?:3|4).*(?:vision|multimodal)",
        r"deepseek[._-]vl",
        r"yi[._-]vl",
        r"bunny",
        r"moondream",
        r"bakllava",
        r"obsidian",
        r"granite.*vision",
        r"llama[._-]?3[._-]2.*(?:11b|90b)",  # Llama 3.2 11B/90B are VL
        # --- Cloud API model names ---
        r"gemini[._-](?:1\.5|2|2\.0|2\.5|pro-vision|flash)",  # Gemini 1.5+ all multimodal
        r"gpt[._-]4o",                        # GPT-4o / 4o-mini are VL
        r"gpt[._-]4[._-]turbo",               # GPT-4 Turbo is VL
        r"gpt[._-]4[._-]vision",
        r"gpt[._-]image",
        r"claude[._-](?:3|3\.5|4)",            # Claude 3+ are multimodal
        r"gemma[._-]3",                        # Gemma 3 (cloud variant names too)
    )
]


def is_vision_model_name(name: str) -> bool:
    """Heuristic: detect VL models by name patterns."""
    return any(p.search(name) for p in _VL_NAME_PATTERNS)


def v1_entry_is_vision(entry: dict) -> bool:
    """Decide whether a /v1/models entry is vision-capable.

    A remote llama-server (e.g. the external classifier sidecar serving a
    Gemma multimodal GGUF with a paired mmproj) advertises this in its
    model entry, but the name heuristic alone misses it — names like
    ``unsloth/gemma-4-E2B-it-qat-GGUF`` carry no "VL"/"vision" token. The
    game-agent frame-attach path keys off ``ModelInfo.vision``; without
    honoring the server's own capability claim it silently drops every
    frame and the model plays blind. Check, in order:

      1. ``capabilities`` as a list  → "multimodal"/"vision" present
      2. ``capabilities`` as a dict  → ``{"vision": true}`` (Mistral shape)
      3. ``modalities``/``input_modalities`` containing "image"
      4. fallback: the name-pattern heuristic

    Shared by every OpenAI-compatible backend (``LlamaCppBackend``,
    ``AugmentumEngineBackend``) so the capability-detection class is fixed
    in one place rather than re-derived per backend.
    """
    caps = entry.get("capabilities")
    if isinstance(caps, list) and any(
        str(c).lower() in ("multimodal", "vision", "image") for c in caps
    ):
        return True
    if isinstance(caps, dict) and caps.get("vision"):
        return True
    for key in ("input_modalities", "modalities"):
        mods = entry.get(key)
        if isinstance(mods, list) and "image" in [str(x).lower() for x in mods]:
            return True
    return is_vision_model_name(entry.get("id", ""))


class ModelBackend(ABC):
    """Abstract interface for LLM backends (Ollama, OpenAI, etc.)."""

    # Whether this backend benefits from system messages injected mid-conversation
    # (right before the latest user turn) vs. folded into the leading system block.
    #
    # llama-server slot reuse and Ollama slot reuse both prefix-match at the
    # token level, so a stable "system at fixed position before last user"
    # injection lets the cache hit on every turn. Cloud OpenAI-compat APIs
    # have no such cache to preserve, and many of them (NVIDIA NIM, DeepSeek,
    # Mistral, Cohere) reject system messages after position 0 with a 400.
    #
    # Default False is the fail-safe: an unknown backend gets the universally
    # accepted shape (single leading system block). Backends that benefit
    # from mid-conversation injection opt in by overriding this attribute.
    supports_mid_conversation_system: bool = False

    @abstractmethod
    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        """Send a chat request and return the full response."""
        ...

    @abstractmethod
    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        """Send a chat request and yield streaming chunks."""
        ...

    def pre_stream_validate(self, request: InternalChatRequest) -> None:
        """Refuse a request before any tokens are generated.

        Why:
          The streaming wire format commits ``http.response.start`` the
          moment Starlette begins iterating the body — so anything that
          raises *after* that point becomes a truncated 200 response,
          not a 4xx. Backends use this gate to surface requirements that
          would otherwise show up mid-stream (image attached to a
          text-only model, future: context-length overflow, unsupported
          tool combinations). The streaming generator in
          ``augmentum/proxy/streaming.py`` catches ``ValueError`` and
          turns it into a typed error chunk, so the user gets an
          actionable message even though the HTTP status is already 200.

        Raises:
          ``ValueError`` with a human-readable message describing why
          the request was refused. The phrase should be specific enough
          that ``_classify_backend_error`` can route it to a stable
          ``error_kind``.
        """
        return

    def is_local_engine(self) -> bool:
        """Return True iff this backend is a local inference engine whose
        chat template (our ``--jinja`` llama-server, vLLM, sglang) injects
        the bare reasoning opener into the prompt PREFIX.

        Load-bearing for reasoning extraction: the asymmetric "response
        stream starts INSIDE a think block" assumption used by GLM-4.x /
        DeepSeek-V4 / Qwen3 / MiniMax / EXAONE (``_STARTS_THINKING_FAMILIES``
        in ``utils/thinking.py``) is ONLY valid when we control the prompt
        prefix. A cloud OpenAI-compat host (NVIDIA NIM, Fireworks, Together,
        Z.AI, OpenRouter, …) templates server-side and returns either a
        native ``reasoning_content`` side-channel or a clean content stream;
        applying the assumption there routes a matched model's entire visible
        answer into the thinking channel and empties the response (the #17
        bug). Callers constructing a ``ThinkingStreamBuffer`` against this
        backend MUST pass ``local_engine=backend.is_local_engine()``.

        Default False is the fail-safe — an unknown backend does NOT get the
        asymmetric assumption (a plain reply survives). Local llama-server
        backends override to True; the OpenAI-compat backend resolves it per
        configured base URL via ``is_local_engine_url``.
        """
        return False

    #: Every sampling-editor field id — the full knob set a local llama.cpp
    #: engine forwards. The single source of truth for both the base
    #: ``supported_sampler_params`` default and the UI's field list.
    SAMPLER_PARAM_KEYS: frozenset[str] = frozenset({
        "temperature", "top_p", "top_k", "min_p",
        "repeat_penalty", "presence_penalty",
    })

    def supported_sampler_params(self, model: str = "") -> set[str]:
        """Which sampling-editor knobs THIS backend actually honors for ``model``.

        The sampling analogue of ``is_local_engine`` — a capability the UI reads
        so the Tuning editor shows exactly the controls that reach the model,
        instead of offering (e.g.) ``min_p`` on an OpenAI model that silently
        drops it. Mirrors the payload-build emission so the wire and the UI
        agree (the same discipline ``detectThinkingSupport`` keeps for
        reasoning). Keys are the editor's field ids.

        Default = ALL of them: local inference engines (llama-server, vLLM,
        sglang, Ollama) forward the full llama.cpp sampler set.
        ``OpenAIBackend`` narrows this per provider (cloud APIs 400 on, or
        silently ignore, non-OpenAI knobs); ``BalancerBackend`` delegates.
        """
        return set(self.SAMPLER_PARAM_KEYS)

    def is_vision_paired(self, model: str = "") -> bool:
        """Return True iff this backend can natively read image attachments
        on the next chat request.

        ``model`` is the model id the request targets — multi-model
        backends (OpenAI-compat clouds, proxies) decide per-model; single-
        loaded-model backends (llama-server) ignore it.

        Used by the route layer (``caption_via_router_fallback``) to decide
        whether to caption images through the SmolVLM sibling and inline
        the result as text instead of letting an image-bearing request hit
        a backend that would either reject it or hallucinate around the
        unread image marker.

        Default True — backends that handle vision natively (cloud VL
        models, llama-server with mmproj) or that we can't introspect
        (Ollama against an arbitrary model) opt out of the fallback. The
        local llama-server path overrides this when no mmproj is paired.
        """
        return True

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return available models."""
        ...

    @abstractmethod
    async def show_model(self, name: str) -> ModelDetails:
        """Return details for a specific model."""
        ...

    async def get_context_length(self, model: str) -> int:
        """Return the model's context window size in tokens.  0 = unknown.

        Default implementation calls show_model and inspects common fields.
        Backends can override for more accurate detection.
        """
        try:
            details = await self.show_model(model)
            # Ollama: model_info contains architecture-specific keys like
            # "llama.context_length", "qwen2.context_length", etc.
            if details.model_info and isinstance(details.model_info, dict):
                for key, val in details.model_info.items():
                    if key.endswith(".context_length") and isinstance(val, int | float):
                        return int(val)
            # llama.cpp / generic: check details dict
            if details.details and isinstance(details.details, dict):
                for key in ("context_length", "context_window", "max_model_len"):
                    val = details.details.get(key)
                    if val and isinstance(val, int | float):
                        return int(val)
        except Exception as exc:
            log.debug("context_length_probe_failed", model=model, error=str(exc))
        return 0
