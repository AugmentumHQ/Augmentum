"""Provider profile catalog.

Each ProviderProfile captures connection details, auth configuration,
and capability flags for a known LLM API provider.  The built-in
``PROFILES`` dict ships entries for 20+ providers so that users only
need to supply an API key and optional model name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class ProviderProfile:
    """Immutable descriptor for an LLM API provider."""

    id: str
    name: str
    base_url: str
    auth_type: str = "bearer"  # "bearer", "api-key", "x-api-key", "query", "none"
    auth_header: str = "Authorization"
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    post_process: str = "none"  # "none", "merge", "semi", "strict"
    converter_id: str = ""  # References a MessageConverter implementation
    model_list_url: str = ""  # Override model listing endpoint
    model_list_params: dict[str, str] = field(default_factory=dict)
    supports_tools: bool = True
    supports_vision: bool = True
    supports_thinking: bool = False
    thinking_param: str = ""  # "reasoning_effort", "thinking", etc.
    # Provider-default context window in tokens, used as a fallback when
    # ``show_model(name)`` doesn't carry per-model metadata (most OpenAI-
    # compat providers only expose ``{id, object}`` from /v1/models/{name}).
    # Read by ``OpenAICompatBackend.get_context_length`` so the Coder
    # mode auto-compactor + UI context indicators stop clamping every
    # cloud model to ``DEFAULT_CODER_COMPACT_TOKENS`` (16K) when the
    # real window is much larger. Default 128K is the modal value across
    # the catalog; override per-profile where it differs meaningfully.
    max_context: int = 128_000
    # Provider-default MAX OUTPUT (completion) tokens for a single response —
    # the ``max_tokens`` / ``max_completion_tokens`` ceiling, which is a
    # SEPARATE limit from ``max_context`` (input window) and often far smaller.
    # 0 = unknown/unmodeled → callers fall back to status-quo behavior and
    # never raise the budget for that provider (safe default). Set ONLY to a
    # value verified from the provider's official docs, and ONLY for providers
    # where a single per-provider number is accurate (single-family first-party
    # APIs). For multi-model hosts / aggregators where the cap varies per model
    # (Cohere, Groq, Fireworks, OpenRouter, NanoGPT, …) leave this 0 and read
    # the per-model limit from their /models metadata instead — a flat value
    # would be wrong for some model they serve. Read by the coder output-budget
    # logic (OpenAIBackend._coder_output_budget) to give capable models real
    # room to write a file in one response while clamping small-cap models so
    # we never send a value their API rejects. Verified June 2026.
    max_output: int = 0
    # Assistant-prefix continuation support. When True, the backend can
    # send a request whose last message is ``role: "assistant"`` and the
    # model will continue that message verbatim instead of starting a
    # fresh turn. Used by the Chat UI's Continue button to extend a
    # response that was cut off (output limit, network hiccup, etc.).
    # DeepSeek requires a sentinel on the last message dict
    # (``{"prefix": True}``); Anthropic continues natively with no
    # marker. Providers without prefix support fall back to a synthetic
    # "continue from where you left off" user message inserted by the
    # backend layer.
    supports_assistant_prefix: bool = False
    assistant_prefix_marker: dict[str, Any] = field(default_factory=dict)
    # Per-request endpoint override for prefix-completion requests.
    # DeepSeek's prefix feature is only enabled on the ``/beta`` path
    # but the user's stored provider may point at ``/v1`` (the standard
    # OpenAI-compat endpoint). When a continue-last-assistant request
    # fires AND ``supports_assistant_prefix`` is True, the backend posts
    # to this URL instead of ``self._base_url`` for that one call. Empty
    # means "no override; use the standard base_url".
    prefix_endpoint_override: str = ""
    # Whether this provider accepts ``reasoning_content`` on assistant
    # messages in the request body. DeepSeek's reasoning lineup REQUIRES
    # it on prior turns (400s otherwise on the round-trip mid-tool-loop).
    # OpenAI / OpenRouter / Cerebras / Mistral / Groq strictly validate
    # request bodies and 400 on unknown top-level keys — feeding them
    # reasoning_content rejects the whole request:
    #   {"message":"messages.2.assistant.reasoning_content: property
    #    'messages.2.assistant.reasoning_content' is unsupported",
    #    "type":"invalid_request_error", "code":"wrong_api_format"}
    # Local engines (llama-server, vLLM, etc.) accept the field for the
    # DeepSeek-compat path. Default False (strict-OAI shape); flip True
    # only for backends that need it on the round-trip.
    accepts_reasoning_content: bool = False

    # ── GPT-5.x / reasoning-model capability flags (2026-05-31) ─────────
    # These gate transmission of OpenAI-family modern parameters added
    # for the GPT 5.x lineup. Default False to preserve strict-compat
    # behavior on every other provider — DeepSeek / Mistral / Groq /
    # Cerebras strict-validate request bodies and 400 on unknown keys,
    # so silent "send to everyone" would break working flows.
    #
    # Set True ONLY when the provider's API explicitly documents
    # support for the field. For OpenAI-family models served by re-
    # routers (OpenRouter / Azure proxying gpt-5), the model-id-based
    # override in ``is_openai_family_target`` catches them even when
    # the upstream profile doesn't have the flag set — the catch-all
    # avoids needing per-router profile maintenance.

    # Send ``max_completion_tokens`` instead of ``max_tokens``.
    # GPT-5.x / o1 / o3 silently IGNORE ``max_tokens`` — user's output
    # cap has no effect without this flag set.
    supports_max_completion_tokens: bool = False

    # Send ``reasoning_effort`` (minimal / low / medium / high / xhigh).
    # When False, the reasoning budget is controlled by other means
    # (engine_reasoning_budget for local; nothing for cloud non-OAI).
    supports_reasoning_effort: bool = False

    # Whether this profile accepts the ``minimal`` tier specifically.
    # OpenAI's direct API supports it on GPT-5.x. Some OpenAI-compat
    # re-routers (codex-proxy/ChatGPT bridge confirmed 2026-05-31)
    # haven't updated their enum and 400 with
    # ``Invalid enum value. Expected 'low' | 'medium' | 'high' |
    # 'xhigh', received 'minimal'``. When False, the adapter demotes
    # an outbound ``minimal`` to ``low`` automatically so user
    # selections + mode-hint defaults keep working without per-target
    # surgery. Only meaningful when ``supports_reasoning_effort`` is
    # also True.
    supports_reasoning_effort_minimal: bool = True

    # Send ``reasoning.summary`` so the response carries a short
    # human-readable thinking preview. OpenAI Responses API only.
    # Chat Completions ignores it silently — safe to default False.
    supports_reasoning_summary: bool = False

    # Use ``role: "developer"`` instead of ``role: "system"`` on
    # outbound messages. OpenAI's spec for reasoning models — both
    # still work but mixing them in one request raises a warning. Most
    # other providers reject "developer" with 400.
    supports_developer_role: bool = False

    # Send ``prompt_cache_key`` (sticky-routing hash) and
    # ``prompt_cache_retention`` (in_memory / 24h). Caching IS
    # automatic on OpenAI for 1024+ token prefixes regardless of these
    # fields, but a stable key sharply increases hit rate (10× cheaper
    # cached input + 80% TTFT win).
    supports_prompt_cache_key: bool = False

    # Send ``service_tier`` (flex / default / priority / scale).
    # Latency-cost tradeoff knob. OpenAI only — silently ignored or
    # 400-rejected elsewhere.
    supports_service_tier: bool = False

    # Send the structured ``thinking`` toggle ({"type": "enabled" |
    # "disabled"}) as a top-level request field — the modern way several
    # providers control reasoning per request instead of via a dedicated
    # "reasoner" model id. Three providers share this EXACT shape and all
    # default reasoning ENABLED, so without the toggle our ``think=False``
    # is silently ignored (reasoning burns latency + empties ``content``):
    #   * DeepSeek V4 (flash/pro) — retired deepseek-chat/deepseek-reasoner;
    #     verified api-docs.deepseek.com/guides/thinking_mode.
    #   * Moonshot Kimi (K2.6 / K2-Thinking) — platform.kimi.ai.
    #   * Z.AI GLM (4.6/4.7) — accepts enabled/disabled/auto; docs.z.ai.
    # Opt-in per profile because strict OpenAI-compat providers 400 on the
    # unknown top-level key. The field name is provider-neutral on purpose
    # — the wire shape, not the vendor, is what gates emission.
    # Verified 2026-06-15.
    supports_thinking_type_toggle: bool = False

    # Gate reasoning via a NESTED ``chat_template_kwargs`` object (NVIDIA NIM
    # / hosted-vLLM convention) instead of the top-level ``enable_thinking``
    # (llama-server) or ``thinking:{type}`` (DeepSeek/Z.AI/Moonshot). The
    # per-model key name (``thinking`` vs ``enable_thinking``) is resolved by
    # ``_nim_chat_template_kwargs`` from the model family. **Load-bearing**:
    # NIM strictly requires ``chat_template_kwargs:{thinking:true}`` to stream
    # reasoning for DeepSeek-V4 — without it the request HANGS indefinitely.
    # Opt-in per profile because strict OpenAI-compat providers 400 on the
    # unknown key. Verified 2026-06-25 (build.nvidia.com).
    reasoning_via_chat_template_kwargs: bool = False

    # Emit Groq's per-model ``reasoning_effort`` (gpt-oss → low/medium/high;
    # qwen3 → none/default). Groq strict-400s on out-of-set enum values, so
    # the per-model mapping + clamp lives in ``_groq_reasoning_params``. Bare
    # Groq profile otherwise never sends reasoning control → the thinking
    # toggle is a no-op on qwen3 and effort is uncontrolled on gpt-oss.
    # Verified 2026-06-25 (console.groq.com/docs/reasoning).
    reasoning_via_groq_params: bool = False

    # Emit OpenRouter's unified ``reasoning`` object ({effort} / {enabled}).
    # OpenRouter normalizes it across every underlying provider, so this one
    # field reaches Anthropic/DeepSeek/Qwen/GLM/Gemini routed via OR — which
    # otherwise get NO reasoning control (only OpenAI-family ids do). Shape +
    # effort-clamp in ``_openrouter_reasoning``. Verified 2026-06-25
    # (openrouter.ai/docs/.../reasoning-tokens).
    supports_openrouter_reasoning: bool = False

    # Emit SiliconFlow's ``enable_thinking`` (bool) + ``thinking_budget``
    # (128–32768) at the request root. Per-model reasoning-model gate +
    # effort→budget mapping in ``_siliconflow_thinking_params``. Verified
    # docs.siliconflow.com 2026-06-25.
    reasoning_via_siliconflow_params: bool = False

    # Accept ``response_format: {"type": "json_schema", ...}`` (OpenAI-style
    # strict structured outputs, schema-validated server-side). Defaults
    # True because OpenAI, local llama-server / sglang / vLLM, and the
    # SmolLM-135M classifier sidecar all honor it — that's what lets a tiny
    # non-reasoning model emit a guaranteed-valid verdict on the voice hop.
    # Flip to False for providers whose API rejects the json_schema TYPE
    # (as opposed to ignoring it): they 400 the whole request. DeepSeek's
    # own API is the confirmed case — it supports ``json_object`` JSON mode
    # but returns ``400 "This response_format type is unavailable now"`` on
    # json_schema (verified api-docs.deepseek.com 2026-06-16). When False
    # the adapter demotes to ``json_object`` (valid-JSON mode); the caller's
    # prompt must name the shape, which the voice-router prompt does.
    supports_response_format_json_schema: bool = True

    # Per-provider sampling ranges. OpenAI's chat API accepts temperature
    # 0–2 and top_p 0–1, but several OpenAI-compat providers use a NARROWER
    # valid range and 400 (or silently reject) an out-of-range value:
    #   * Moonshot Kimi — temperature [0, 1] (platform.kimi.ai).
    #   * Z.AI GLM — temperature [0, 1], top_p [0.01, 1] (docs.z.ai).
    #   * Cohere v2 — temperature [0, 1] (docs.cohere.com).
    # ``None`` means "no provider-specific bound" → the value is passed
    # through untouched (the default for every provider, so local engines
    # that legitimately use temperature > 2 are never clamped). Set a
    # ``(lo, hi)`` tuple ONLY where the provider documents a tighter range;
    # the emit site clamps into it. Verified 2026-06-25.
    temperature_range: tuple[float, float] | None = None
    top_p_range: tuple[float, float] | None = None

    # Non-OpenAI sampler knobs this provider's API DOCUMENTS accepting, each
    # mapped to its documented ``(lo, hi)`` clamp range (``None`` = unbounded
    # on that side). This is the sampling parallel of the ``reasoning_via_*``
    # flags: the model-card recommended sampling (min_p / top_k /
    # repetition_penalty resolved by ``sampling_profiles``) reaches LOCAL
    # engines via the full llama.cpp param set, but is DROPPED for cloud unless
    # the provider is listed here — otherwise strict providers 400 on the
    # unknown key. Keys use the OpenAI-COMPAT WIRE names (``repetition_penalty``
    # — NOT llama.cpp's ``repeat_penalty``; the emit site maps the source key).
    # Ranges verified against each provider's official API reference (2026-08):
    #   * Fireworks   — top_k 0–100, min_p 0–1, repetition_penalty 0–2
    #     (docs.fireworks.ai/api-reference/post-chatcompletions)
    #   * Together    — top_k, min_p 0–1, repetition_penalty
    #     (docs.together.ai/reference/chat-completions-1)
    #   * OpenRouter  — top_k 0+, min_p 0–1, repetition_penalty 0–2 (+ top_a)
    #     (openrouter.ai/docs/api-reference/parameters)
    #   * SiliconFlow — top_k, min_p 0–1 (min_p is Qwen3-only server-side); NO
    #     repetition_penalty (docs.siliconflow.com)
    # Strict providers (OpenAI/DeepSeek/Z.AI/Mistral/Groq/Cerebras/Moonshot/
    # xAI) document NONE of these — they stay ABSENT here, which also gates
    # ``top_k`` OFF for them (it was previously sent to every cloud provider, a
    # latent 400 once sampling profiles set a recommended top_k for a model
    # they serve, e.g. Qwen3/QwQ/MiniMax).
    sampler_extras: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )

    # Whether the provider accepts ``tool_choice="required"``. OpenAI /
    # DeepSeek / most providers do; Moonshot Kimi does NOT and 400s on it
    # (platform.kimi.ai). When False, the emit site demotes ``required``
    # to ``auto`` so a forced-tool turn (e.g. the math reasoning flow's
    # Compute step) degrades gracefully instead of hard-failing. Verified
    # 2026-06-25.
    supports_tool_choice_required: bool = True

    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in catalog
# ---------------------------------------------------------------------------

PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        supports_thinking=True,
        thinking_param="reasoning_effort",
        # GPT-5.x feature set (2026 platform). All True — these are
        # OpenAI's native fields and they expect them on reasoning
        # models in particular.
        supports_max_completion_tokens=True,
        supports_reasoning_effort=True,
        supports_reasoning_summary=True,
        supports_developer_role=True,
        supports_prompt_cache_key=True,
        supports_service_tier=True,
        max_context=400_000,  # GPT-5.5 advertises ~1M; 5.5-pro >272K input is surcharged.
        # GPT-5.x API models publish a 128K output ceiling (the ChatGPT
        # snapshot gpt-5-chat-latest is the exception at 16K, but the API
        # family is 128K). Verified developers.openai.com 2026-06.
        max_output=128_000,
    ),
    "chatgpt_bridge": ProviderProfile(
        id="chatgpt_bridge",
        name="ChatGPT (via codex-proxy)",
        # User-provided; bridge runs on a user-chosen host:port and there's
        # no canonical URL to auto-detect from.
        base_url="",
        supports_thinking=True,
        thinking_param="reasoning_effort",
        # Bridge passes through to real OpenAI — same capabilities.
        supports_max_completion_tokens=True,
        supports_reasoning_effort=True,
        # Confirmed 2026-05-31: codex-proxy bridge 400s on
        # ``reasoning_effort="minimal"`` with
        # ``Invalid enum value. Expected 'low' | 'medium' | 'high' | 'xhigh'``.
        # Bridge's enum hasn't been updated for GPT-5.x's minimal tier.
        # Adapter demotes minimal → low transparently. Re-enable if
        # the upstream bridge gains support.
        supports_reasoning_effort_minimal=False,
        supports_reasoning_summary=True,
        supports_developer_role=True,
        supports_prompt_cache_key=True,
        supports_service_tier=True,
        # GPT-5 family advertises 400K input; the Codex Desktop Responses
        # path (which this bridge wraps) hasn't published a hard ceiling,
        # so settle at a conservative 256K. Stops the Coder compactor from
        # clamping to DEFAULT_CODER_COMPACT_TOKENS (16K) when the bridge's
        # /v1/models doesn't ship a context_length field.
        max_context=256_000,
        notes="Pairs with a local codex-proxy instance that exposes a "
              "ChatGPT Plus/Pro account as OpenAI-compatible endpoints.",
    ),
    "openrouter": ProviderProfile(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        extra_headers={"HTTP-Referer": "https://augmentum.dev", "X-Title": "Augmentum"},
        supports_thinking=True,
        # Unified ``reasoning`` object reaches every underlying provider via
        # OR (Anthropic/DeepSeek/Qwen/GLM/…) — see the flag docstring + #4.
        supports_openrouter_reasoning=True,
        # OpenRouter passes sampler extras through to whichever provider it
        # routes to (omitting ones the target doesn't support). Verified
        # openrouter.ai/docs/api-reference/parameters 2026-08.
        sampler_extras={
            "top_k": (0, None),
            "min_p": (0.0, 1.0),
            "repetition_penalty": (0.0, 2.0),
        },
    ),
    "mistral": ProviderProfile(
        id="mistral",
        name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        converter_id="mistral",
        # Mistral Large 3 is 256K context. Output isn't documented as a
        # separate ceiling (bounded by the shared context budget), so leave
        # max_output=0. Verified docs.mistral.ai 2026-06.
        max_context=256_000,
    ),
    "deepseek": ProviderProfile(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/beta",
        post_process="semi",
        model_list_url="https://api.deepseek.com/models",
        # DeepSeek's Chat Prefix Completion (beta endpoint): set
        # ``prefix: true`` on the trailing assistant message and the
        # model continues it verbatim. The prefix feature is ONLY
        # exposed on the ``/beta`` path — users whose stored base_url
        # is ``/v1`` (the standard OpenAI-compat endpoint) need the
        # request rerouted to ``/beta`` for prefix completion. Standard
        # /v1 requests are unaffected.
        supports_assistant_prefix=True,
        assistant_prefix_marker={"prefix": True},
        prefix_endpoint_override="https://api.deepseek.com/beta",
        # DeepSeek's reasoning models REQUIRE ``reasoning_content`` on
        # prior assistant turns in mid-tool-loop replays — 400s with
        # "reasoning_content in thinking mode must be passed back"
        # otherwise. Only flip this for providers that need it.
        accepts_reasoning_content=True,
        # V4 (flash/pro) gates reasoning per-request via the ``thinking``
        # field instead of a dedicated reasoner model ID. Lets the
        # latency-sensitive voice classifier hop (think=False) actually
        # turn reasoning OFF on the API instead of timing out.
        supports_thinking_type_toggle=True,
        # DeepSeek's API supports ``json_object`` JSON mode but NOT the
        # ``json_schema`` response_format type — it 400s with "This
        # response_format type is unavailable now" (verified
        # api-docs.deepseek.com 2026-06-16). The voice-router classifier
        # attaches a json_schema; without this flag every classifier call
        # that falls back to a DeepSeek primary 400s → regex-drops the
        # utterance. Demote to json_object instead.
        supports_response_format_json_schema=False,
        # DeepSeek V4 (flash/pro) is a 1M-token context window (verified
        # api-docs.deepseek.com 2026-06-15). Without this the profile
        # inherits the 128K default and the Coder auto-compactor wastes
        # ~87% of the window.
        max_context=1_000_000,
        # DeepSeek V4 (flash/pro) max OUTPUT is 384,000 tokens — verified
        # api-docs.deepseek.com/quick_start/pricing 2026-06. We were capping
        # coder writes at the flat 8192 mode-hint (47× under), so a large
        # file_write ran out of output budget mid-JSON and the trailing
        # ``path`` field got chopped → "called without a 'path' argument".
        max_output=384_000,
        # DeepSeek ships NO vision-capable models — deepseek-chat,
        # deepseek-reasoner, V3.2 and V4 (flash/pro) are all text-only,
        # and the API 400s on ``image_url`` parts (verified live 2026-06-18
        # via the live-camera path). Without this the profile inherits the
        # default ``supports_vision=True``, so ``is_vision_paired`` claims
        # native vision, the SmolVLM caption fallback is skipped, and the
        # image is silently dropped at the API → the companion answers
        # blind. Flag it text-only so attachments get captioned-to-text.
        supports_vision=False,
    ),
    "ai21": ProviderProfile(
        id="ai21",
        name="AI21 Labs",
        base_url="https://api.ai21.com/studio/v1",
        converter_id="ai21",
        max_context=256_000,  # Jamba — 256K context. Verified docs.ai21.com 2026-06.
        # Jamba hard-caps a single response at 4096 tokens — model the ceiling
        # so the coder output-budget logic clamps instead of over-requesting.
        # (CORRECTIONS #7)
        max_output=4_096,
        # Jamba supports json_object JSON mode but NOT the json_schema
        # response_format type → demote to json_object. (CORRECTIONS #9)
        supports_response_format_json_schema=False,
    ),
    "xai": ProviderProfile(
        id="xai",
        name="xAI (Grok)",
        base_url="https://api.x.ai/v1",
        supports_thinking=True,
        thinking_param="reasoning_effort",
        # xAI accepts reasoning_effort on Grok-4. Other GPT-5.x-specific
        # fields (max_completion_tokens, prompt_cache_key, service_tier)
        # are NOT documented for xAI — leave False.
        supports_reasoning_effort=True,
        # No Grok model accepts the ``minimal`` tier — grok-4.3 is
        # none/low/medium/high, the multi-agent line is low/medium/high/
        # xhigh — so an outbound ``minimal`` 400s. Demote it to ``low``
        # (same handling as chatgpt_bridge/fireworks). Verified
        # docs.x.ai 2026-06-25 (CORRECTIONS #10).
        supports_reasoning_effort_minimal=False,
        # Current Grok (4.3 / 4.x) is a 1M context window (grok-4-fast is 2M);
        # the old 256K was the original grok-4-0709. Verified docs.x.ai 2026-06.
        max_context=1_000_000,
        # xAI publishes NO per-model max-output ceiling — max_output_tokens
        # defaults to null (uncapped, generates to natural stop / context).
        # Leave max_output=0 (unknown) so we don't invent a cap. Verified
        # docs.x.ai 2026-06.
    ),
    "azure": ProviderProfile(
        id="azure",
        name="Azure OpenAI",
        base_url="",  # Instance-specific
        auth_type="api-key",
        auth_header="api-key",
        notes="base_url is deployment-specific; user must provide it.",
    ),
    "moonshot": ProviderProfile(
        id="moonshot",
        name="Moonshot AI",
        base_url="https://api.moonshot.ai/v1",
        supports_thinking=True,
        # Kimi K2.6 / K2-Thinking gate reasoning via the same structured
        # ``thinking:{type:enabled|disabled}`` field as DeepSeek, default
        # ENABLED. Without this our think=False never reached the wire.
        supports_thinking_type_toggle=True,
        max_context=256_000,  # Kimi K2.6 — 256K context
        # Kimi K2.6 documents a separate 32,768 output ceiling (well below the
        # 256K context). Verified platform.kimi.ai 2026-06.
        max_output=32_768,
        # Kimi temperature range is [0, 1] (not OpenAI's [0, 2]); an
        # out-of-range value is rejected. Clamp at emit. (CORRECTIONS #13)
        temperature_range=(0.0, 1.0),
        # Kimi does not support tool_choice="required" → 400. Demote to
        # "auto" at emit. (CORRECTIONS #12)
        supports_tool_choice_required=False,
    ),
    "groq": ProviderProfile(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        # gpt-oss → low/medium/high · qwen3 → none/default. Per-model enum
        # mapping in ``_groq_reasoning_params`` (Groq strict-400s otherwise).
        reasoning_via_groq_params=True,
    ),
    "perplexity": ProviderProfile(
        id="perplexity",
        name="Perplexity",
        base_url="https://api.perplexity.ai",
        post_process="strict",
        supports_thinking=True,
        max_context=200_000,  # sonar-pro 200K (sonar 128K)
        # Sonar models cap output at 8,000 tokens. Verified
        # docs.perplexity.ai 2026-06 (corroborated via pricing docs).
        max_output=8_000,
        # sonar-reasoning-pro + sonar-deep-research accept
        # minimal/low/medium/high. Search-only models silently ignore it.
        # Verified docs.perplexity.ai 2026-06.
        supports_reasoning_effort=True,
    ),
    "fireworks": ProviderProfile(
        id="fireworks",
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        # Fireworks accepts low/medium/high/xhigh/max/none/adaptive/int.
        # The generic path clamps minimal→low (Fireworks has no minimal).
        # Verified docs.fireworks.ai 2026-06.
        supports_reasoning_effort=True,
        supports_reasoning_effort_minimal=False,
        # Fireworks documents the full sampler set with explicit ranges.
        # Verified docs.fireworks.ai/api-reference/post-chatcompletions 2026-08.
        sampler_extras={
            "top_k": (0, 100),
            "min_p": (0.0, 1.0),
            "repetition_penalty": (0.0, 2.0),
        },
    ),
    "pollinations": ProviderProfile(
        id="pollinations",
        name="Pollinations",
        base_url="https://gen.pollinations.ai/v1",
        model_list_url="https://gen.pollinations.ai/text",
        supports_thinking=True,
        # minimal/low/medium/high. Verified pollinations APIDOCS.md 2026-06.
        supports_reasoning_effort=True,
    ),
    "aimlapi": ProviderProfile(
        id="aimlapi",
        name="AIML API",
        base_url="https://api.aimlapi.com/v1",
        extra_headers={"Content-Type": "application/json"},
    ),
    "electronhub": ProviderProfile(
        id="electronhub",
        name="ElectronHub",
        base_url="https://api.electronhub.ai/v1",
    ),
    "chutes": ProviderProfile(
        id="chutes",
        name="Chutes AI",
        base_url="https://llm.chutes.ai/v1",
        supports_thinking=True,
    ),
    "nanogpt": ProviderProfile(
        id="nanogpt",
        name="NanoGPT",
        base_url="https://nano-gpt.com/api/v1",
        # NanoGPT authenticates with ``Authorization: Bearer <key>`` across
        # all official examples — the bearer default. A prior ``x-api-key``
        # override produced silent 401s. Verified docs.nano-gpt.com 2026-06-15.
    ),
    "zai": ProviderProfile(
        id="zai",
        name="Z.AI",
        base_url="https://api.z.ai/api/paas/v4",
        extra_headers={"Accept-Language": "en-US"},
        supports_thinking=True,
        max_context=200_000,  # GLM 4.6/4.7 ~200K (202,752)
        # GLM 4.6/4.7 document a 131,072 output ceiling (unusually high).
        # Some Coding-plan deployments silently cap at 98,304. Verified
        # docs.z.ai 2026-06.
        max_output=131_072,
        # GLM 4.6/4.7 use the same ``thinking:{type:...}`` field (accepts
        # enabled/disabled/auto; we send enabled/disabled), default
        # reasoning ON. Same never-sent bug as DeepSeek/Kimi otherwise.
        supports_thinking_type_toggle=True,
        # GLM sampling: temperature [0, 1] and top_p [0.01, 1] (not
        # OpenAI's [0, 2] / [0, 1]); out-of-range values are rejected.
        # Clamp at emit. (CORRECTIONS #26)
        temperature_range=(0.0, 1.0),
        top_p_range=(0.01, 1.0),
    ),
    "siliconflow": ProviderProfile(
        id="siliconflow",
        name="SiliconFlow",
        base_url="https://api.siliconflow.com/v1",
        model_list_params={"type": "text"},
        # enable_thinking bool + thinking_budget int (128–32768). Per-model
        # reasoning-model gate + budget mapping in
        # ``_siliconflow_thinking_params``. Verified docs.siliconflow.com
        # 2026-06-25.
        reasoning_via_siliconflow_params=True,
        # SiliconFlow documents top_k + min_p (0–1; min_p applies to Qwen3
        # server-side, harmlessly ignored elsewhere) but NOT repetition_penalty
        # (it exposes frequency_penalty instead). Verified docs.siliconflow.com
        # 2026-08.
        sampler_extras={
            "top_k": (0, None),
            "min_p": (0.0, 1.0),
        },
    ),
    "cohere": ProviderProfile(
        id="cohere",
        name="Cohere",
        base_url="https://api.cohere.com/v2",
        converter_id="cohere",
    ),
    "together": ProviderProfile(
        id="together",
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        # Together documents temperature as a decimal 0–1 (NOT OpenAI's 0–2);
        # an out-of-range value is rejected. Verified docs.together.ai 2026-08.
        temperature_range=(0.0, 1.0),
        # Together exposes the full sampler set. top_k has no documented upper
        # bound; min_p is 0–1; repetition_penalty documented without an
        # explicit range (pass through). Verified
        # docs.together.ai/reference/chat-completions-1 2026-08.
        sampler_extras={
            "top_k": (0, None),
            "min_p": (0.0, 1.0),
            "repetition_penalty": (0.0, None),
        },
    ),
    "nvidia": ProviderProfile(
        id="nvidia",
        name="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1",
        # NVIDIA's NIM rejects any system message after position 0 with
        # "System message must be at the beginning.". Narrative mode injects
        # dynamic STATE/MEMORY as a system message just before the latest user
        # turn (for llama-server prefix-cache reasons), which violates that
        # rule. "semi" converts non-leading system messages to user role and
        # merges consecutive same-role messages — matches SillyTavern's
        # NVIDIA handling.
        post_process="semi",
        # NIM gates reasoning via nested ``chat_template_kwargs`` and
        # DeepSeek-V4 HANGS without it — see the flag docstring + the
        # ``nvidia`` provider card.
        reasoning_via_chat_template_kwargs=True,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_profile(profile_id: str) -> ProviderProfile | None:
    """Look up a provider profile by ID.  Returns ``None`` if not found."""
    return PROFILES.get(profile_id)


def list_profiles() -> list[ProviderProfile]:
    """Return all built-in profiles sorted by ID."""
    return sorted(PROFILES.values(), key=lambda p: p.id)


def _host(url: str) -> str:
    """Return the lowercase host of ``url``, or ``""`` if unparseable."""
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.hostname or "").lower()


def get_profile_for_url(base_url: str) -> ProviderProfile | None:
    """Match ``base_url`` to a built-in profile by host.

    Used as a fallback when a stored provider has no ``profile_id`` —
    either because it predates the column (rows from migration 111 or
    earlier) or because the user added it without picking a profile.
    Matching is host-only so trailing slashes, ``/v1`` vs ``/v1/``, and
    other path variations don't affect the result.

    Returns the first profile (by sorted ID) whose ``base_url`` shares
    a host with the input. Profiles with an empty ``base_url`` (Azure,
    where the deployment URL is per-instance) cannot match.
    """
    target = _host(base_url)
    if not target:
        return None
    for profile in sorted(PROFILES.values(), key=lambda p: p.id):
        if profile.base_url and _host(profile.base_url) == target:
            return profile
    return None


# Model-id patterns that identify OpenAI-family reasoning/chat models.
# When the user routes a gpt-5-style id through a profile WITHOUT the
# capability flags (e.g. OpenRouter / Azure / a custom proxy), this
# catch-all upgrades the request to the OpenAI-family feature set.
# Without it, profile maintainers would need to flip flags on every
# re-router profile any time OpenAI ships a new family.
#
# Kept tight — only matches model ids whose semantics are documented
# OpenAI's. Mistral/Anthropic/Gemini ids are left to their respective
# provider profiles. ``gpt-3.5`` excluded deliberately: legacy chat
# completions, no reasoning, ``max_completion_tokens`` isn't expected.
_OPENAI_FAMILY_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-5",       # 5, 5.5, 5.4, 5.3-instant, 5.2-codex, mini/nano/pro
    "gpt-4.1",     # 4.1 / 4.1-mini / 4.1-nano
    "gpt-4o",      # 4o / 4o-mini / 4o-realtime
    "o1",          # o1-preview, o1-mini
    "o3",          # o3, o3-mini
    "o4",          # o4-mini and beyond
    "chatgpt",     # chatgpt-4o-latest etc.
    "codex-",      # codex-mini-latest, codex-2.x
)


def is_openai_family_model(model_id: str) -> bool:
    """True iff ``model_id`` matches an OpenAI-family reasoning/chat
    prefix. Case-insensitive, leading/trailing whitespace tolerated.
    Doesn't check the provider — see ``effective_capability`` for the
    composed check that combines profile + model-id."""
    if not model_id:
        return False
    mid = model_id.strip().lower()
    return mid.startswith(_OPENAI_FAMILY_MODEL_PREFIXES)


def effective_capability(
    profile: ProviderProfile | None,
    model_id: str,
    attr: str,
) -> bool:
    """Combine profile-level capability with the model-id catch-all.

    Returns True iff EITHER the profile explicitly declares support
    for ``attr`` OR the model id matches the OpenAI family. Used at
    payload-build time so every new field has a single decision point.

    ``attr`` examples: ``"supports_max_completion_tokens"``,
    ``"supports_reasoning_effort"``, ``"supports_developer_role"``,
    ``"supports_prompt_cache_key"``, ``"supports_service_tier"``.

    Safe with ``profile=None`` (anonymous endpoint, no catalog entry):
    falls through to the model-id check only."""
    profile_says_yes = bool(getattr(profile, attr, False)) if profile else False
    if profile_says_yes:
        return True
    return is_openai_family_model(model_id)
