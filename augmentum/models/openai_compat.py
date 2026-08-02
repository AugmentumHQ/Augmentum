"""OpenAI-compatible backend implementation."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from augmentum.models.backend_errors import BackendError, parse_retry_after
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
    is_vision_model_name,
)
from augmentum.models.converters.base import PostProcessMode
from augmentum.models.converters.utils import post_process as _post_process_messages
from augmentum.models.kv_reuse_audit import KvReuseAuditMixin
from augmentum.models.provider_profiles import ProviderProfile
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail
from augmentum.utils.thinking import ThinkingStreamBuffer, normalize_thinking

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


def _normalized_model_name(model: str) -> str:
    return "".join(ch for ch in (model or "").lower() if ch.isalnum())


def _effective_think(request: InternalChatRequest) -> bool:
    """Per-request thinking flag with the explicit kwarg override folded in.

    Higher-level loops (coder act/plan iterations, companion voice,
    architect routers) express their thinking policy as
    ``request.chat_template_kwargs = {"enable_thinking": bool}`` — the
    shape llama-server consumes directly. This backend historically keyed
    every provider-specific reasoning toggle (DeepSeek/Moonshot/Z.AI
    ``thinking:{type}``, NIM nested kwargs, Groq/OpenRouter/SiliconFlow
    params) off ``request.think`` alone, so those explicit policies were
    silently ignored on cloud providers: coder turns reasoned (or not)
    per the global think setting instead of the coder policy/toggle.

    Precedence mirrors llama_cpp's merge order: an explicit
    ``enable_thinking`` (or Kimi-style ``thinking``) in
    ``chat_template_kwargs`` wins over ``request.think``.
    """
    ctk = request.chat_template_kwargs or {}
    for key in ("enable_thinking", "thinking"):
        if key in ctk:
            return bool(ctk[key])
    return bool(request.think)


def _template_thinking_override(model: str, think: bool) -> bool | None:
    """Return a per-turn ``enable_thinking`` kwarg for hybrid reasoning models.

    Several families branch on the ``enable_thinking`` chat-template kwarg.
    llama-server / vLLM / sglang accept it at the top level of the OpenAI
    request payload (forwarded into the template). Mapping:

    * Qwen 3 / 3.5 / 3.6 hybrid → driven by the UI's per-turn ``request.think``.
      Fixed ``Thinking`` / ``Instruct`` variants are locked on/off respectively.
    * GLM-4.x (incl. GLM-4.5, 4.6, 4.7, Air, Flash) → all hybrid; driven by
      the same per-turn flag.
    * LG AI EXAONE 4.0 / EXAONE-Deep → README documents ``enable_thinking=True``
      opening a reasoning block via the chat template. Same per-turn flag.
    * NVIDIA Nemotron 3 Nano (Reasoning variants) → ``enable_thinking`` kwarg
      defaulting to True; same per-turn flag.
    * Google Gemma 4 (April 2026) — ``enable_thinking`` abstracts the
      ``<|think|>`` system-prompt control token. With it set, the model
      emits the asymmetric ``<|channel>thought\\n…<channel|>`` block;
      without it the channel block is emitted empty (non-Edge variants)
      or skipped. Llama.cpp's jinja path handles the token swap from
      the kwarg.

    The UI's ``detectThinkingSupport`` exposes the thinking button for these
    families, so flipping it here flows end-to-end.

    Returns ``None`` for non-reasoning models (no kwarg added).
    """
    normalized = _normalized_model_name(model)
    is_qwen3 = (
        "qwen3" in normalized
        or "qwen35" in normalized
        or "qwen36" in normalized
    )
    # GLM-4.x (4.5/4.6/4.7) and GLM-5.x (5.0/5.1/5.2) use the same hybrid
    # ``enable_thinking`` chat-template kwarg. Local llama-server / vLLM
    # honor it; cloud NIM does not confidently accept a nested key for GLM
    # (see ``_nim_chat_template_kwargs`` GLM exclusion), so this override
    # only reaches templates on the local-engine path — see the top-level
    # ``payload["enable_thinking"]`` gate on ``is_local_engine_url``.
    is_glm4 = "glm4" in normalized or "chatglm" in normalized
    is_glm5 = "glm5" in normalized  # matches glm5 / glm52 / glm51
    is_exaone = "exaone" in normalized
    is_nemotron = "nemotron" in normalized
    is_gemma4 = "gemma4" in normalized
    # Moonshot Kimi K2.6 / K2.6-Thinking (Apr 2026). Per Moonshot's
    # platform docs the toggle name is bare ``thinking`` (bool), NOT
    # ``enable_thinking``; the llama_cpp adapter handles the name
    # remap based on family detection. This function still returns
    # the boolean value the caller wants to send.
    is_kimi = "kimi" in normalized and "k2" in normalized
    # Xiaomi MiMo-V2.5 / V2.5-Pro (Apr-May 2026). Standard
    # ``enable_thinking`` kwarg; also accepts a ``reasoning_effort``
    # string (low/med/high) which we currently surface only via the
    # OpenAI-family reasoning-effort UI for paid endpoints.
    is_mimo = "mimo" in normalized
    if not (is_qwen3 or is_glm4 or is_glm5 or is_exaone or is_nemotron or is_gemma4 or is_kimi or is_mimo):
        return None

    # Qwen-specific locked variants.
    if is_qwen3:
        if "thinking" in normalized:
            return True
        if "instruct" in normalized:
            return False
        # Qwen3-Coder (incl. Qwen3-Coder-Next) is non-thinking by design
        # even without an Instruct suffix. Skip emitting the kwarg —
        # the model ignores it and the UI hides the toggle to match.
        if "coder" in normalized:
            return None

    # Kimi-K2.6-Thinking is the locked thinking-on variant (mirrors
    # Qwen3-Thinking). Bare K2.6 is hybrid — the kwarg drives it.
    if is_kimi and "thinking" in normalized:
        return True

    return bool(think)


def _supports_preserve_thinking(model: str) -> bool:
    """Return True when the model family consumes ``preserve_thinking``.

    Qwen 3.6 trained the chat template to keep ``<think>`` traces across
    multi-turn history when the kwarg is set. Other hybrid families
    (Qwen 3 / 3.5, GLM-4.x, EXAONE 4, Nemotron 3) do not document this
    behavior — forwarding the kwarg there is a no-op, but the UI gates
    its preserve popover to the same set to avoid pretending it works.
    """
    normalized = _normalized_model_name(model)
    return "qwen36" in normalized


def _nim_chat_template_kwargs(model: str, think: bool) -> dict[str, bool] | None:
    """Reasoning toggle for NVIDIA NIM, nested under ``chat_template_kwargs``.

    NIM (NVIDIA's hosted vLLM) gates reasoning via a NESTED
    ``chat_template_kwargs`` object — NOT the top-level ``enable_thinking``
    that llama-server accepts, and NOT ``reasoning_effort``.

    **Load-bearing for DeepSeek-V4**: NIM *strictly requires*
    ``chat_template_kwargs:{thinking:bool}`` to stream reasoning. WITHOUT it,
    ``deepseek-v4-flash`` / ``deepseek-v4-pro`` **HANG indefinitely** instead
    of returning (build.nvidia.com/deepseek-ai/deepseek-v4-flash;
    docs.nvidia.com/nim/.../qwen/api.html). For Qwen/Nemotron it controls
    thinking; reasoning still returns in ``reasoning_content`` either way.

    Families handled (confident NIM conventions only):

      * **DeepSeek** V4/V3.2 → ``{"thinking": bool}`` — the HANG case;
        handled explicitly because ``_template_thinking_override`` does NOT
        recognize DeepSeek (cloud DeepSeek uses the top-level ``thinking:
        {type}`` toggle instead). Always emitted so the toggle reaches NIM.
      * **Qwen3 / Nemotron / EXAONE / Gemma 4 / MiMo** →
        ``{"enable_thinking": bool}`` — value + locked-variant + reasoning-
        model gate reused from ``_template_thinking_override`` (None for
        non-reasoning → no kwarg, so a base Llama on NIM is untouched).

    GLM / Kimi are intentionally EXCLUDED: their nested NIM key is not
    confidently documented, and only DeepSeek-V4 hangs without the kwarg, so
    guessing a key risks breaking them for no safety gain. (They still get the
    top-level ``thinking:{type}`` toggle if a future NVIDIA profile sets
    ``supports_thinking_type_toggle``.)
    """
    normalized = _normalized_model_name(model)
    # DeepSeek V4 (flash/pro) / V3.2 — the documented hang case. The thinking
    # template var is ignored by non-reasoning deepseek templates, so emitting
    # it broadly is harmless; NIM only serves the reasoning variants anyway.
    if "deepseek" in normalized:
        return {"thinking": bool(think)}
    value = _template_thinking_override(model, think)
    if value is None:
        return None
    if "glm" in normalized or "kimi" in normalized:
        return None  # uncertain NIM convention — don't guess
    return {"enable_thinking": value}


def _siliconflow_thinking_params(
    model: str, think: bool, reasoning_effort: str | None
) -> dict[str, object] | None:
    """SiliconFlow reasoning control: ``enable_thinking`` + ``thinking_budget``.

    SiliconFlow defaults ``enable_thinking=True`` for supported models, so
    without our toggle the ``think=False`` path is a no-op. Budget (128–32768,
    default 4096) lets the user trade latency for reasoning depth.

    Only reasoning-capable models (DeepSeek/Qwen3/GLM) get the toggle; non-
    reasoning Llama/etc. emit nothing. ``min_p`` is Qwen3-only on SiliconFlow
    but that's a sampler concern, not handled here.

    Verified docs.siliconflow.com/en/api-reference/chat-completions 2026-06-25.
    """
    normalized = _normalized_model_name(model)
    is_reasoning = (
        "deepseek" in normalized
        or "qwen3" in normalized
        or "glm" in normalized
        or "minimax" in normalized
        or "kimi" in normalized
    )
    if not is_reasoning:
        return None
    params: dict[str, object] = {"enable_thinking": bool(think)}
    if think and reasoning_effort:
        eff = str(reasoning_effort).strip().lower()
        budget_map = {
            "minimal": 512, "low": 1024, "medium": 4096,
            "high": 16384, "xhigh": 24576, "max": 32768,
        }
        budget = budget_map.get(eff)
        if budget is not None:
            budget = max(128, min(budget, 32768))
            params["thinking_budget"] = budget
    return params


def _groq_reasoning_params(
    model: str, think: bool, reasoning_effort: str | None
) -> dict[str, str] | None:
    """Groq reasoning control — per-model ``reasoning_effort`` enums.

    Groq strict-validates the body: each reasoning model accepts a DIFFERENT
    effort enum and 400s on out-of-set values (console.groq.com/docs/reasoning):

      * **GPT-OSS** 20B/120B → ``low`` / ``medium`` / ``high`` (the model
        always reasons — no disable value). ``reasoning_format`` is
        UNSUPPORTED on gpt-oss.
      * **Qwen3** → ``none`` (disable) / ``default`` (enable). Here the
        per-turn ``think`` toggle finally has effect (it was a no-op before).

    ``reasoning_format`` is intentionally NOT sent: ``raw`` + JSON-mode/tools
    → 400, and downstream we already read BOTH the ``reasoning`` field and
    inline ``<think>`` tags, so Groq's own default is safe to leave alone.

    Returns ``None`` for non-reasoning Groq models (Llama, Whisper, …) and
    when gpt-oss has no UI-selected effort → nothing emitted, Groq default
    stands. The UI's finer tiers are clamped into Groq's set so a user's
    ``minimal``/``xhigh`` selection can never 400.
    """
    normalized = _normalized_model_name(model)
    if "gptoss" in normalized:
        eff = (reasoning_effort or "").strip().lower()
        mapped = {
            "minimal": "low", "low": "low", "medium": "medium",
            "high": "high", "xhigh": "high", "max": "high",
        }.get(eff)
        # Only emit when the UI actually picked an effort; otherwise leave
        # Groq's own default (medium) untouched.
        return {"reasoning_effort": mapped} if mapped else None
    if "qwen3" in normalized:
        return {"reasoning_effort": "default" if think else "none"}
    return None


def _openrouter_reasoning(
    think: bool, reasoning_effort: str | None
) -> dict[str, object]:
    """OpenRouter unified ``reasoning`` control object.

    OpenRouter normalizes one ``reasoning`` object across every underlying
    provider (Anthropic/DeepSeek/Qwen/GLM/Gemini/Grok), so a single emit
    point reaches them all — without it, reasoning control only reaches
    OpenAI-family model ids and everything else routed via OR is
    uncontrolled. Schema (openrouter.ai/docs/.../reasoning-tokens):

      * ``effort``: ``"low"`` / ``"medium"`` / ``"high"`` (OpenAI-style;
        mutually exclusive with ``max_tokens``). OR maps to the nearest
        supported level per model, so the UI's wider tiers are clamped here.
      * ``enabled``: ``true`` to enable with defaults / ``false`` to disable.

    ``think=False`` → ``{"enabled": False}`` (lets the user turn OFF a
    default-on reasoning model). ``think=True`` → ``{"effort": …}`` when the
    UI picked one, else ``{"enabled": True}``.
    """
    if not think:
        return {"enabled": False}
    eff = (reasoning_effort or "").strip().lower()
    mapped = {
        "minimal": "low", "low": "low", "medium": "medium",
        "high": "high", "xhigh": "high", "max": "high",
    }.get(eff)
    if mapped:
        return {"effort": mapped}
    return {"enabled": True}


# Legacy alias retained briefly so any external imports don't break. New
# callers should import ``_template_thinking_override``.
_qwen_thinking_override = _template_thinking_override


# Local docker-compose hostnames that bundle a chat-template-aware engine
# (llama-server, vLLM, sglang, Ollama). These accept ``enable_thinking``
# and the matching budget keys at the top level of the OpenAI payload.
_LOCAL_ENGINE_HOSTS = frozenset({
    "engine", "llama-server", "llama_server", "llamacpp", "llama_cpp",
    "ollama", "vllm", "sglang", "speaches",
})


# OpenAI's tools/function-calling spec enforces this regex on every tool
# name (definitions in ``tools[]`` and references in
# assistant.tool_calls[].function.name + Responses-API input items).
# chatgpt-bridge / Codex forward this validation literally with a 400
# "Invalid 'input[N].name': string does not match pattern" and the
# whole turn fails permanently. Some local models emit names like
# ``shell.exec``, ``code:write``, ``mcp:fs:read``, or accidentally
# include a unicode quote — we rewrite at the egress boundary so they
# don't kill the turn. Same map applied to defs AND calls so the model
# can still match by name.
_TOOL_NAME_VALID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_TOOL_NAME_INVALID_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    """Coerce a tool name to OpenAI's ``^[a-zA-Z0-9_-]+$`` shape.

    Invalid chars (``.`` ``:`` whitespace, unicode, etc.) become ``_``.
    Names that collapse to empty become ``invalid_tool``. The result is
    deterministic so a sanitised def matches the sanitised call.
    """
    if not name:
        return "invalid_tool"
    if _TOOL_NAME_VALID_RE.match(name):
        return name
    cleaned = _TOOL_NAME_INVALID_CHARS_RE.sub("_", name)
    return cleaned or "invalid_tool"


def _scrub_orphan_tool_messages(messages: list[dict]) -> list[dict]:
    """Drop ``role=tool`` messages whose ``tool_call_id`` has no parent.

    The OpenAI Chat Completions API and the Codex Responses API both 400
    when a tool result has no matching tool_call earlier in the array:

        "Messages with role 'tool' must be a response to a preceding
         message with 'tool_calls'"
        "No tool call found for function call output with call_id ..."

    Compaction and the rewind tool can leave orphans behind (parent
    assistant turn dropped, child result kept). Walking the array in
    order and keeping a live set of valid ids is enough — assistant
    tool_calls must precede their matching tool result by spec.
    """
    valid_ids: set[str] = set()
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                tc_id = tc.get("id")
                if tc_id:
                    valid_ids.add(tc_id)
            out.append(m)
        elif role == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id and tc_id in valid_ids:
                out.append(m)
            else:
                log.warning(
                    "openai_compat.orphan_tool_message_dropped",
                    tool_call_id=tc_id or "<missing>",
                )
        else:
            out.append(m)
    return out


def is_local_engine_url(base_url: str) -> bool:
    """Return True if ``base_url`` points to a local inference engine.

    ``enable_thinking`` and the matching ``reasoning_budget`` /
    ``thinking_token_budget`` / ``grace_period`` keys are chat-template
    kwargs forwarded by llama-server, vLLM, sglang and Ollama. Cloud
    OpenAI-compatible providers (Cerebras, OpenAI, OpenRouter, Anthropic
    via proxy, etc.) strictly validate the request body and 400 on
    unknown top-level keys — Cerebras for instance returns::

        {"message": "enable_thinking: property 'enable_thinking' is
                     unsupported", "code": "wrong_api_format"}

    To stay correct on both sides we only forward those keys when the
    request lands on a local engine. "Local" means loopback, RFC 1918,
    or a docker-compose hostname we ship ourselves.
    """
    if not base_url:
        return False
    try:
        host = urlparse(base_url if "://" in base_url else f"http://{base_url}").hostname
    except Exception:
        return False
    if not host:
        return False
    host = host.lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    if host in _LOCAL_ENGINE_HOSTS:
        return True
    # RFC 1918 private ranges — 10/8, 172.16/12, 192.168/16.
    if host.startswith(("10.", "192.168.")):
        return True
    if host.startswith("172."):
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return True
        except (ValueError, IndexError):
            pass
    return False


def _clamp_to_range(
    value: float, rng: tuple[float | None, float | None]
) -> float:
    """Clamp ``value`` into ``(lo, hi)``; ``None`` bound = open on that side.

    Used to fit a model-card recommended sampler value (min_p / top_k /
    repetition_penalty) into the range a specific provider documents, so a
    profile that is right for one host can never send an out-of-range value
    to another. Preserves ``int`` inputs (top_k) so the wire type is unchanged.
    """
    lo, hi = rng
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


class OpenAIBackend(KvReuseAuditMixin, ModelBackend):
    """Communicates with an OpenAI-compatible API."""

    # TTL for cached ``list_models`` results. Two independent callers
    # (provider_registry.refresh_model_map + model_manager.list_models)
    # fire on every UI refresh; without the cache each refresh sends 2
    # HTTP calls per provider (12+ providers = 24+ outbound requests
    # per tick). 30s is short enough that adding / removing a model on
    # the upstream side surfaces inside one refresh cycle.
    _LIST_MODELS_TTL_S = 30.0

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str | None = None,
        *,
        profile: ProviderProfile | None = None,
        chat_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client
        self._chat_client = chat_client or client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._profile = profile
        # Prefix-stability contract + response-side reuse audit (shared with
        # the bundled llama-server backend via KvReuseAuditMixin). Lets a
        # remote provider that silently re-charges a cacheable prefix surface
        # as ``server_void`` instead of being invisible. Default-on here — the
        # mixin's ``_kv_reuse_trackable`` returns True — because remote usage
        # now carries normalised cache_hit/miss telemetry to judge against.
        self._init_kv_audit()
        # Single-flight + TTL state for ``list_models``. The lock
        # serialises concurrent calls so the second arrival waits for
        # the first's HTTP round-trip instead of firing its own. After
        # the first call completes, both arrivals see the cached
        # result for up to ``_LIST_MODELS_TTL_S`` seconds.
        self._list_models_cache: list[ModelInfo] | None = None
        self._list_models_cached_at: float = 0.0
        self._list_models_lock = asyncio.Lock()
        # Tool-call ``extra_content`` cache, keyed by tool_call id. Gemini 3.x
        # (via Google's OpenAI-compat endpoint) returns a per-call
        # ``extra_content.google.thought_signature`` on each tool_call and
        # REQUIRES it echoed back on every subsequent turn that includes that
        # call, or it 400s ("Function call is missing a thought_signature").
        # The coder's stream accumulator rebuilds tool_calls from id/name/args
        # only, dropping extra_content — so we cache it here at capture time and
        # re-attach it at egress by id (the id round-trips intact). Bounded FIFO.
        # Verified 2026-07-04 against gemini-3.1-flash-lite: drop → 400, keep →
        # 200. Harmless for providers that don't send extra_content.
        self._tool_call_extra: dict[str, dict] = {}

    # Cap on the extra_content cache so a long-lived backend can't grow it
    # unbounded. A conversation rarely holds more than a few dozen live tool
    # calls; 512 leaves generous headroom while bounding memory.
    _TOOL_CALL_EXTRA_CAP = 512

    def _capture_tool_call_extra(self, tool_calls: list | None) -> None:
        """Cache any ``extra_content`` carried on response tool_calls, keyed by id.

        Called on both the streaming and non-streaming response paths. Only
        stores when a tool_call has both an ``id`` and a non-empty
        ``extra_content`` — i.e. Gemini's thought_signature envelope. FIFO-
        evicts the oldest entry past the cap. Never raises.
        """
        if not tool_calls:
            return
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            extra = tc.get("extra_content")
            if tc_id and extra:
                if (
                    tc_id not in self._tool_call_extra
                    and len(self._tool_call_extra) >= self._TOOL_CALL_EXTRA_CAP
                ):
                    # Drop the oldest inserted key (dict preserves insertion order).
                    try:
                        oldest = next(iter(self._tool_call_extra))
                        del self._tool_call_extra[oldest]
                    except StopIteration:
                        pass
                self._tool_call_extra[tc_id] = extra

    def _reattach_tool_call_extra(self, tc: dict) -> dict:
        """Re-attach cached ``extra_content`` to an outgoing tool_call by id.

        No-op when the tool_call already carries ``extra_content`` (the coder
        preserved it) or when nothing is cached for its id. Returns the (possibly
        augmented) dict; callers use the return value.
        """
        if not isinstance(tc, dict) or tc.get("extra_content"):
            return tc
        cached = self._tool_call_extra.get(tc.get("id", ""))
        if cached:
            return {**tc, "extra_content": cached}
        return tc

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._profile:
            auth_type = self._profile.auth_type
            if auth_type == "bearer" and self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            elif auth_type in ("api-key", "x-api-key") and self._api_key:
                headers[self._profile.auth_header] = self._api_key
            headers.update(self._profile.extra_headers)
        elif self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _sanitize_url_for_log(url: str) -> str:
        """Return URL context safe for logs and user-facing diagnostics."""
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            if not host:
                return ""
            netloc = host
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            path = parsed.path or ""
            return urlunparse((parsed.scheme, netloc, path, "", "", ""))
        except Exception:
            return ""

    @staticmethod
    def _transport_error_message(exc: BaseException, *, model: str, url: str) -> str:
        """Human-readable transport error that never collapses to blank.

        httpx timeout exceptions often stringify to ``""``. Include the class
        name plus sanitized destination so Anthropic/Coder clients show an
        actionable failure instead of ``Backend error:``.
        """
        exc_name = type(exc).__name__
        detail = str(exc).strip()
        target = OpenAIBackend._sanitize_url_for_log(url)
        parts = [f"Provider transport error ({exc_name})"]
        if model:
            parts.append(f"for model {model}")
        if target:
            parts.append(f"at {target}")
        if detail:
            parts.append(f": {detail}")
        return " ".join(parts)

    @staticmethod
    def _payload_diagnostics(payload: dict) -> dict:
        """Diagnostic byte-shape of an outbound chat payload.

        Used on backend-error logs so context-window failures stop being
        opaque 502s — the log line itself surfaces total bytes, message
        count, and per-image-byte breakdown. Cheap (single JSON serialize
        + a linear scan), safe to call on any payload shape.

        Returns keys: ``payload_bytes``, ``message_count``,
        ``instruction_bytes`` (sum of system/developer message text),
        ``image_count``, ``total_image_bytes`` (sum of byte lengths of
        ``image_url.url`` strings — base64 data URIs included).
        """
        try:
            payload_bytes = len(json.dumps(payload, ensure_ascii=False))
        except Exception:
            payload_bytes = -1

        messages = payload.get("messages") or []
        instruction_bytes = 0
        image_count = 0
        total_image_bytes = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content")
            if role in ("system", "developer") and isinstance(content, str):
                instruction_bytes += len(content)
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        image_count += 1
                        url = (part.get("image_url") or {}).get("url", "")
                        if isinstance(url, str):
                            total_image_bytes += len(url)
        return {
            "payload_bytes": payload_bytes,
            "message_count": len(messages),
            "instruction_bytes": instruction_bytes,
            "image_count": image_count,
            "total_image_bytes": total_image_bytes,
        }

    @staticmethod
    def _build_vision_content(msg: Message) -> str | list[dict]:
        """Build OpenAI content field, using vision array format when images present."""
        if not msg.images:
            return msg.content
        parts: list[dict] = []
        if msg.content:
            parts.append({"type": "text", "text": msg.content})
        for img in msg.images:
            parts.append({"type": "image_url", "image_url": {"url": img}})
        return parts

    def _coder_output_budget(self, requested: int) -> int:
        """Model-aware output-token budget — the cloud mirror of
        ``LlamaCppBackend._floor_max_tokens_for_coder``.

        On a LARGE-OUTPUT request (>= 4096, i.e. the coder mode-hint; tight
        modes like analytical's 512-tok UARF stay untouched) raise the budget
        toward ``coder_cloud_max_tokens_floor`` so a capable cloud model has
        real room to emit a file in one response — then CLAMP to the model's
        documented ``ProviderProfile.max_output`` so a small-cap model (Cohere
        R+ 4K, Perplexity 8K) never receives a value its API rejects.

        Only acts when ``max_output`` is KNOWN (>0). For unknown providers
        (aggregators, undocumented ceilings) it returns ``requested`` unchanged
        — we don't invent a ceiling or risk raising a request a provider can't
        honor. The ``finish_reason="length"`` retry backstops those.
        """
        max_out = int(getattr(self._profile, "max_output", 0) or 0) if self._profile else 0
        if max_out <= 0:
            return requested  # unknown ceiling — status quo, never guess
        if requested >= 4096:  # large-output (coder) mode
            from augmentum.config import settings as _settings
            floor = int(getattr(_settings, "coder_cloud_max_tokens_floor", 0) or 0)
            if floor > 0:
                requested = max(requested, min(floor, max_out))
        return min(requested, max_out)  # never exceed the documented ceiling

    def _build_openai_payload(self, request: InternalChatRequest, *, strip_images: bool = False) -> dict:
        """Convert internal request to OpenAI API format."""
        # Role rewrite: OpenAI's reasoning models (GPT-5.x, o1, o3,
        # o4-mini) expect ``role: "developer"`` instead of
        # ``role: "system"``. Both still work, but the docs say not
        # to mix them in one request. Every other provider rejects
        # "developer" with a 400, so gate strictly via the OpenAI-
        # family check. Applied once per payload-build at the top
        # so the alternation/post_process logic below sees the final
        # role names.
        # Gate STRICTLY on the model being OpenAI-family — never on the
        # profile flag alone. ``developer`` is a property of the OpenAI
        # MODEL, not the endpoint: genuine OpenAI reasoning models accept
        # both ``system`` and ``developer`` (keeping ``system`` costs
        # nothing), while every other backend HARD-400s on ``developer``
        # (observed live: DeepSeek's beta endpoint returns "unknown variant
        # `developer`, expected one of `system`, `user`, `assistant`,
        # `tool`, `latest_reminder`"). The profile flag (supports_developer_
        # role on openai / chatgpt_bridge) leaks onto ANY provider a user or
        # re-router attached that profile to — so a DeepSeek/custom backend
        # mis-classed as OpenAI-compatible would otherwise get a fatal
        # rewrite for zero benefit. The model-id check is the real
        # precondition; the rewrite is a no-loss optimization for true
        # OpenAI models only.
        # ...AND the request must actually be bound for OpenAI's endpoint.
        # ``developer`` is an OpenAI *API* feature, not merely a model
        # property: an OpenAI-family model re-routed to another provider
        # (e.g. a ``coder_model`` pointed at ``api.deepseek.com``) HARD-400s
        # on ``developer`` ("unknown variant `developer`"). ``system`` is
        # universally accepted (OpenAI included), so requiring the genuine
        # OpenAI host here is strictly loss-free.
        from urllib.parse import urlparse as _urlparse

        from augmentum.models.provider_profiles import is_openai_family_model
        _host = (_urlparse(self._base_url).hostname or "").lower()
        _is_openai_host = _host == "api.openai.com" or _host.endswith(".openai.com")
        rewrite_system_to_developer = (
            is_openai_family_model(request.model) and _is_openai_host
        )
        messages = []
        for msg in request.messages:
            outbound_role = msg.role
            if rewrite_system_to_developer and msg.role == "system":
                outbound_role = "developer"
            if strip_images:
                m: dict = {"role": outbound_role, "content": msg.content}
            else:
                m = {"role": outbound_role, "content": self._build_vision_content(msg)}
            if msg.tool_calls:
                # Egress hygiene: sanitize every tool-call name to the
                # OpenAI ``^[a-zA-Z0-9_-]+$`` shape. Local models
                # occasionally emit dotted/colon-prefixed names that
                # chatgpt-bridge / Codex reject with a 400 mid-loop.
                # See ``_sanitize_tool_name`` for the mapping.
                m["tool_calls"] = [
                    self._reattach_tool_call_extra({
                        **tc,
                        "function": {
                            **(tc.get("function") or {}),
                            "name": _sanitize_tool_name(
                                (tc.get("function") or {}).get("name", "")
                            ),
                        },
                    })
                    if isinstance(tc, dict) and "function" in tc
                    else tc
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            # ``reasoning_content`` round-trip: emit only when the
            # destination actually accepts it. The field is load-bearing
            # for DeepSeek's reasoning lineup (400s otherwise mid-tool-
            # loop with "reasoning_content in thinking mode must be
            # passed back") AND tolerated by local engines (llama-server,
            # vLLM). Strict cloud OAI-compat (Cerebras, OpenAI proper,
            # OpenRouter, Mistral, Groq, NVIDIA NIM) reject unknown
            # top-level keys with 400:
            #   "messages.N.assistant.reasoning_content: property
            #    'messages.N.assistant.reasoning_content' is unsupported"
            # Gate emission on profile.accepts_reasoning_content (True
            # only for backends that need it) OR locality (local engines
            # tolerate it).
            if msg.role == "assistant" and msg.thinking:
                accepts_reasoning = (
                    (self._profile is not None
                     and getattr(self._profile, "accepts_reasoning_content", False))
                    or is_local_engine_url(self._base_url)
                )
                if accepts_reasoning:
                    m["reasoning_content"] = msg.thinking
            messages.append(m)

        # Egress hygiene: drop orphan role=tool messages whose parent
        # tool_call was removed by compaction or rewind. Otherwise the
        # provider returns 400 "Messages with role 'tool' must be a
        # response to a preceding message with 'tool_calls'" or, on the
        # Codex Responses API, "No tool call found for function call
        # output with call_id ...". The scrub keeps the message order
        # intact so continue-last-assistant and post_process below still
        # operate on a coherent shape.
        messages = _scrub_orphan_tool_messages(messages)

        # Continue-last-assistant handling. Runs BEFORE post_process so the
        # alternation pipeline sees the final shape.
        #
        # Strategy (mirrors Open WebUI's approach in
        # backend/open_webui/utils/middleware.py:2302-2313): send the
        # messages with the partial assistant as the trailing turn. No
        # synthetic "continue from where you left off" user message —
        # that prompts the model to start a fresh composition with
        # re-introduction. With trailing assistant + nothing after,
        # OpenAI-compatible endpoints naturally complete that turn.
        #
        # Two variants:
        #
        # 1. Provider supports native assistant-prefix AND the model
        #    routes prefix completion to the visible content channel
        #    (DeepSeek's /beta on `deepseek-chat`): merge the profile's
        #    ``assistant_prefix_marker`` (``prefix: true``) onto the
        #    trailing assistant message so DeepSeek's prefix-completion
        #    pipeline kicks in. Strip reasoning_content so the model
        #    sees only the visible partial.
        # 2. Otherwise (DeepSeek reasoning lineup that emits
        #    reasoning_content for prefix requests, plus every other
        #    provider): send the messages as-is with the trailing
        #    assistant, no prefix marker, reasoning_content stripped.
        #    Models complete the assistant turn without re-introducing.
        #    Confirmed via Open WebUI users that this pattern works
        #    against DeepSeek, OpenAI, and Anthropic in practice.
        if request.continue_last_assistant and messages and messages[-1].get("role") == "assistant":
            use_native_prefix = (
                self._profile is not None
                and self._profile.supports_assistant_prefix
                and self._reasoning_safe_for_prefix(request.model)
            )
            if use_native_prefix and self._profile.assistant_prefix_marker:
                messages[-1].update(self._profile.assistant_prefix_marker)
            # Strip ``reasoning_content`` from the trailing message in
            # both paths. For reasoning models, leaving prior reasoning
            # in the trailing message primes the reasoner to keep
            # reasoning instead of producing visible content (observed
            # on DeepSeek V4 Pro / V4 Flash: 24KB reasoning, 0 content
            # bytes for prefix completion requests with reasoning
            # carried forward).
            messages[-1].pop("reasoning_content", None)

        # Provider-specific normalization — e.g. NVIDIA's NIM rejects any
        # system message after position 0, DeepSeek expects merged/semi
        # alternation, Perplexity requires strict user/assistant alternation.
        # The engine deliberately injects system messages mid-array for
        # llama-server prefix-cache reasons; this is where we reconcile that
        # with strict OpenAI-compat APIs that can't accept it.
        #
        # Skipped when message list has tool_calls/tool_call_id because the
        # post-processing pipeline doesn't model the tool-call/tool-result
        # pairing and would corrupt it.
        #
        # Also skipped on continue-last-assistant requests with a
        # trailing assistant: the trailing assistant is the prefix the
        # model continues from, and alternation rewriters (e.g.
        # DeepSeek's "semi" mode) would either reject the trailing
        # assistant or rewrite it as a user turn, breaking continuation.
        is_continue_with_assistant_tail = (
            request.continue_last_assistant
            and messages
            and messages[-1].get("role") == "assistant"
        )
        if (
            self._profile
            and self._profile.post_process != "none"
            and not is_continue_with_assistant_tail
        ):
            has_tool_flow = any("tool_calls" in m or "tool_call_id" in m for m in messages)
            if not has_tool_flow:
                mode_str = self._profile.post_process
                try:
                    mode = PostProcessMode(mode_str)
                except ValueError:
                    mode = PostProcessMode.NONE
                # Vision content is a list[dict]; post_process works on string
                # content. Preserve vision messages by routing them through
                # unchanged — post_process only runs when every message has
                # string content.
                if all(isinstance(m.get("content"), str) for m in messages):
                    messages = _post_process_messages(messages, mode)

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }

        if request.temperature is not None:
            # Clamp into the provider's documented range when it declares a
            # narrower one (Moonshot/Z.AI/Cohere use [0, 1], not OpenAI's
            # [0, 2]); ``None`` range = pass through untouched so local
            # engines that use temperature > 2 are never clamped.
            _t_range = self._profile.temperature_range if self._profile else None
            payload["temperature"] = (
                max(_t_range[0], min(_t_range[1], request.temperature))
                if _t_range else request.temperature
            )
        if request.top_p is not None:
            _p_range = self._profile.top_p_range if self._profile else None
            payload["top_p"] = (
                max(_p_range[0], min(_p_range[1], request.top_p))
                if _p_range else request.top_p
            )
        # Non-OpenAI sampler knobs — top_k / min_p / repetition_penalty. These
        # carry the model-card recommended sampling (sampling_profiles) that
        # reaches LOCAL engines via the full llama.cpp param set. On CLOUD they
        # are emitted ONLY to providers whose official API documents them
        # (profile.sampler_extras), clamped to the documented range — strict
        # providers (OpenAI/DeepSeek/Z.AI/Mistral/Groq/Cerebras/…) 400 on the
        # unknown key. Local engines (llama-server/vLLM/sglang/Ollama) keep
        # top_k exactly as before; min_p / repetition_penalty on the
        # openai-compat LOCAL path are left to that engine's own defaults here
        # (the bundled-engine backends forward the full set themselves, with
        # the correct llama.cpp spelling — we don't guess it at this layer).
        _sampler_extras = self._profile.sampler_extras if self._profile else {}
        _local_engine = is_local_engine_url(self._base_url)
        # top_k is a first-class request field; local keeps it unconditionally,
        # cloud only when the provider documents it (clamped to its range).
        if request.top_k is not None and (_local_engine or "top_k" in _sampler_extras):
            top_k_val = request.top_k
            if "top_k" in _sampler_extras:
                top_k_val = int(_clamp_to_range(top_k_val, _sampler_extras["top_k"]))
            payload["top_k"] = top_k_val
        # min_p / repeat_penalty ride in raw_options (not dataclass fields).
        # Source key ``repeat_penalty`` (llama.cpp) → wire ``repetition_penalty``
        # (the OpenAI-compat spelling every listed cloud provider expects).
        _raw_opts = request.raw_options if isinstance(request.raw_options, dict) else {}
        if "min_p" in _sampler_extras:
            _mp = _raw_opts.get("min_p")
            if _mp is not None:
                payload["min_p"] = _clamp_to_range(_mp, _sampler_extras["min_p"])
        if "repetition_penalty" in _sampler_extras:
            _rp = _raw_opts.get("repeat_penalty")
            if _rp is not None:
                payload["repetition_penalty"] = _clamp_to_range(
                    _rp, _sampler_extras["repetition_penalty"]
                )

        # Output-cap routing: GPT-5.x / o1 / o3 silently IGNORE the
        # legacy ``max_tokens`` field — must use ``max_completion_tokens``
        # per OpenAI's spec for reasoning models. Every other provider
        # (DeepSeek, Mistral, Groq, Cerebras, local llama-server) still
        # honors ``max_tokens`` and either ignores or 400s on
        # ``max_completion_tokens``. So the gate is twofold:
        #   1. profile says supports_max_completion_tokens, OR
        #   2. model id matches OpenAI-family pattern (gpt-5*, o1*, o3*,
        #      etc.) — catches OpenRouter / Azure routing to gpt-5 even
        #      when the upstream profile doesn't have the flag set.
        # See ``effective_capability`` in provider_profiles.py.
        if request.max_tokens is not None:
            from augmentum.models.provider_profiles import effective_capability
            budget = self._coder_output_budget(request.max_tokens)
            if effective_capability(
                self._profile, request.model,
                "supports_max_completion_tokens",
            ):
                payload["max_completion_tokens"] = budget
            else:
                payload["max_tokens"] = budget

        if request.stop:
            payload["stop"] = request.stop
        # DeepSeek DEPRECATED frequency_penalty / presence_penalty — its API
        # documents that a passed value "will not take effect" (verified
        # api-docs.deepseek.com 2026-08). Skip them so we don't ship dead
        # fields that imply a control the model won't honor; every other
        # provider still honors them.
        _penalties_ignored = bool(self._profile and self._profile.id == "deepseek")
        if request.frequency_penalty is not None and not _penalties_ignored:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None and not _penalties_ignored:
            payload["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            # Match the sanitization applied to assistant.tool_calls so
            # the names in defs and calls still line up after rewriting.
            # Dedupe by the SANITIZED name: distinct raw names can collapse to
            # one here (``web.search`` and ``web_search`` both -> ``web_search``),
            # and providers (DeepSeek/OpenAI) HARD-400 on "Tool names must be
            # unique." This is the definitive choke point — every outbound
            # payload's tool names are finalized here.
            sanitized_tools = []
            _seen_tool_names: set[str] = set()
            for t in request.tools:
                if isinstance(t, dict) and isinstance(t.get("function"), dict):
                    fn = dict(t["function"])
                    sname = _sanitize_tool_name(fn.get("name", ""))
                    if sname in _seen_tool_names:
                        continue
                    _seen_tool_names.add(sname)
                    fn["name"] = sname
                    sanitized_tools.append({**t, "function": fn})
                else:
                    sanitized_tools.append(t)
            payload["tools"] = sanitized_tools
        if request.tool_choice is not None:
            # Moonshot Kimi 400s on tool_choice="required"; demote to "auto"
            # for any provider that declares it unsupported. Every other
            # value (and provider) passes through unchanged.
            _tc = request.tool_choice
            if (
                _tc == "required"
                and self._profile
                and not self._profile.supports_tool_choice_required
            ):
                _tc = "auto"
            payload["tool_choice"] = _tc

        if request.format == "json":
            payload["response_format"] = {"type": "json_object"}

        # Schema-constrained output. A higher layer (the voice router) puts
        # a JSON Schema in ``raw_options["json_schema"]`` to force the model
        # to emit EXACTLY the verdict shape — no prose, no chain-of-thought,
        # no truncated objects. Both llama-server and OpenAI honor the
        # standard ``response_format`` json_schema form, so a constrained
        # tiny model (the SmolLM-135M classifier sidecar) literally cannot
        # emit anything but a valid verdict. Overrides the json_object form
        # above when present. The local engine path forwards the same
        # ``json_schema`` key via its raw_options passthrough allowlist.
        _raw = request.raw_options or {}
        _schema = _raw.get("json_schema")
        if _schema:
            # Some OpenAI-compat providers accept the json_schema TYPE and
            # validate against it server-side (OpenAI, local llama-server /
            # sglang / vLLM, the SmolLM classifier sidecar). Others — DeepSeek
            # most notably — reject the type outright with a 400 ("This
            # response_format type is unavailable now") rather than ignoring
            # it, which 400s the whole classifier call and silently drops the
            # utterance. For those, demote to ``json_object`` (valid-JSON
            # mode): the model still can't emit prose, and the caller's prompt
            # carries the exact shape. Profiles flag rejecters with
            # ``supports_response_format_json_schema=False``; default True
            # preserves behavior for every provider that already worked.
            _supports_schema = (
                getattr(self._profile, "supports_response_format_json_schema", True)
                if self._profile is not None
                else True
            )
            if _supports_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _raw.get("json_schema_name", "structured_output"),
                        "schema": _schema,
                        "strict": True,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        # Reasoning effort routing for OpenAI-family + xAI Grok. Both
        # accept ``reasoning_effort`` at the top level; other cloud
        # providers reject unknown fields. Validity: minimal / low /
        # medium / high / xhigh.
        #
        # ``minimal`` is a 2026 GPT-5.x tier some OpenAI-compat
        # re-routers (codex-proxy bridge confirmed 2026-05-31) haven't
        # added to their enum yet — they 400 with
        # ``Invalid enum value. Expected 'low' | 'medium' | 'high' | 'xhigh'``.
        # Profiles flag this with ``supports_reasoning_effort_minimal``;
        # when False, demote silently to ``low``. User choice is
        # preserved on profiles that DO accept minimal (real OpenAI).
        from augmentum.models.provider_profiles import effective_capability
        if (
            request.reasoning_effort
            and effective_capability(
                self._profile, request.model,
                "supports_reasoning_effort",
            )
        ):
            effort_value = str(request.reasoning_effort).strip().lower()
            if effort_value == "minimal" and not effective_capability(
                self._profile, request.model,
                "supports_reasoning_effort_minimal",
            ):
                effort_value = "low"
            payload["reasoning_effort"] = effort_value

        # Prompt-cache discipline. OpenAI caches every prefix ≥1024
        # tokens automatically, but routing through a stable cache key
        # sharply increases hit rate (10× cheaper input + ~80% TTFT
        # reduction). The key MUST be stable across turns for the
        # same conceptual session — we tag it with workspace + user
        # so per-workspace prefixes (system prompt, tool defs, repo
        # manifest) reuse the cached prefill. Pulled from raw_options
        # because openai_routes is the highest layer that knows the
        # session/workspace identity.
        if effective_capability(
            self._profile, request.model, "supports_prompt_cache_key",
        ):
            raw = request.raw_options or {}
            cache_key = str(raw.get("prompt_cache_key") or "").strip()
            if cache_key:
                payload["prompt_cache_key"] = cache_key[:128]
            # 24h retention is GPT-5.5's default but explicit is safer
            # — older 5.x families default in_memory (5-10min) and a
            # long agentic run benefits from the persistent variant.
            retention = str(raw.get("prompt_cache_retention") or "").strip()
            if retention in ("in_memory", "24h"):
                payload["prompt_cache_retention"] = retention

        # Service tier passthrough (flex / default / priority / scale).
        # Per-request latency-vs-cost knob. OpenAI only — silently
        # ignored or 400-rejected elsewhere.
        if effective_capability(
            self._profile, request.model, "supports_service_tier",
        ):
            raw = request.raw_options or {}
            tier = str(raw.get("service_tier") or "").strip()
            if tier in ("flex", "default", "priority", "scale", "auto"):
                payload["service_tier"] = tier

        # Effective per-turn thinking flag: request.think with any explicit
        # ``chat_template_kwargs`` policy folded over it (coder loops send
        # {"enable_thinking": bool} — see _effective_think). Every
        # provider-specific reasoning emitter below keys off THIS, so the
        # coder default-OFF policy + composer toggle reach cloud providers,
        # not just llama-server.
        think = _effective_think(request)

        # Structured ``thinking`` toggle (cloud). DeepSeek V4 (flash/pro),
        # Moonshot Kimi, and Z.AI GLM all gate reasoning with a top-level
        # ``thinking: {"type": "enabled"|"disabled"}`` field, default
        # ENABLED (verified 2026-06-15). Without this, every cloud call to
        # these providers reasons regardless of ``request.think`` — which
        # times out the voice classifier hop (think=False) and empties
        # ``content`` on act turns (reasoning eats the budget). The
        # local-engine ``enable_thinking`` path below can't cover this:
        # it's gated to is_local_engine_url and the field name/shape
        # differ. Capability-gated because strict cloud providers 400 on
        # the unknown top-level key.
        if effective_capability(
            self._profile, request.model, "supports_thinking_type_toggle",
        ):
            thinking_obj: dict[str, str] = {
                "type": "enabled" if think else "disabled"
            }
            # DeepSeek V4 (flash/pro) accepts ``reasoning_effort`` NESTED
            # inside ``thinking:{type, reasoning_effort}`` (high/max). The
            # other toggle providers (Moonshot/Z.AI) don't document this
            # field and may strict-400 on it, so only DeepSeek gets it.
            if (
                think
                and request.reasoning_effort
                and self._profile
                and self._profile.id == "deepseek"
            ):
                eff = str(request.reasoning_effort).strip().lower()
                if eff in ("high", "max"):
                    thinking_obj["reasoning_effort"] = eff
            payload["thinking"] = thinking_obj

        thinking_override = _template_thinking_override(request.model, think)
        if thinking_override is not None and is_local_engine_url(self._base_url):
            # Only forward to local engines (llama-server / vLLM / sglang /
            # Ollama). Cloud providers like Cerebras strict-validate the
            # payload and return 400 on unknown top-level keys — see
            # ``is_local_engine_url`` for the full rationale.
            payload["enable_thinking"] = thinking_override
            # ALSO nest it under ``chat_template_kwargs`` — llama-server
            # only forwards template kwargs from the nested object (the
            # top-level key is a vLLM/sglang convention it ignores).
            # Live-verified on llama-server b9664 with Gemma-4-E2B: the
            # top-level form left thinking ON; the nested form is what
            # actually reaches the jinja template. Both are sent because
            # vLLM/sglang honor top-level while older llama-server builds
            # predate chat_template_kwargs; each engine ignores the
            # other's spelling. Merge-not-replace so the NIM block below
            # composes.
            merged_tpl = dict(payload.get("chat_template_kwargs") or {})
            merged_tpl["enable_thinking"] = thinking_override
            payload["chat_template_kwargs"] = merged_tpl
            # Mirror llama_cpp.py's per-turn budget forwarding so the same
            # cap applies whether the request lands on llama-server or a
            # vLLM/SGLang endpoint.
            if thinking_override:
                from augmentum.config import settings as _settings
                budget = int(getattr(_settings, "engine_reasoning_budget", 0) or 0)
                grace = int(getattr(_settings, "engine_reasoning_grace_period", 0) or 0)
                if budget > 0:
                    payload["reasoning_budget"] = budget
                    payload["thinking_token_budget"] = budget + grace
                if grace > 0:
                    payload["grace_period"] = grace
                if request.preserve_thinking and _supports_preserve_thinking(request.model):
                    payload["preserve_thinking"] = True

        # Explicit per-request template kwargs (coder loops, external /v1
        # callers) — merged over the automatic think-mapping for LOCAL
        # engines only, mirroring llama_cpp's precedence (explicit wins).
        # llama-server / vLLM / sglang forward the nested object into the
        # chat template; strict cloud validators 400 on the unknown key,
        # and cloud reasoning is already covered by the _effective_think
        # folding above.
        if request.chat_template_kwargs and is_local_engine_url(self._base_url):
            merged_explicit = dict(payload.get("chat_template_kwargs") or {})
            merged_explicit.update(request.chat_template_kwargs)
            payload["chat_template_kwargs"] = merged_explicit

        # NVIDIA NIM reasoning: nested ``chat_template_kwargs`` (hosted-vLLM
        # convention), NOT the top-level ``enable_thinking`` above (which is
        # gated to local engines and strict-cloud-400s elsewhere). REQUIRED
        # for DeepSeek-V4 on NIM — the request HANGS without it. Per-model key
        # + reasoning-model gate live in ``_nim_chat_template_kwargs``.
        if self._profile and self._profile.reasoning_via_chat_template_kwargs:
            nim_kwargs = _nim_chat_template_kwargs(request.model, think)
            if nim_kwargs is not None:
                merged = dict(payload.get("chat_template_kwargs") or {})
                merged.update(nim_kwargs)
                payload["chat_template_kwargs"] = merged

        # Groq reasoning: per-model ``reasoning_effort`` (gpt-oss low/med/high,
        # qwen3 none/default). Groq strict-400s on out-of-set enums, so the
        # mapping + clamp is in ``_groq_reasoning_params``. The generic
        # supports_reasoning_effort path doesn't fire here (bare Groq profile,
        # non-OpenAI-family ids), so this is the sole emitter — no double-send.
        if self._profile and self._profile.reasoning_via_groq_params:
            groq_params = _groq_reasoning_params(
                request.model, think, request.reasoning_effort
            )
            if groq_params:
                payload.update(groq_params)

        # OpenRouter unified ``reasoning`` object — normalizes reasoning
        # control across every underlying provider (Anthropic/DeepSeek/Qwen/
        # GLM/…) that OR proxies. Shape + effort-clamp in
        # ``_openrouter_reasoning``.
        if self._profile and self._profile.supports_openrouter_reasoning:
            payload["reasoning"] = _openrouter_reasoning(
                think, request.reasoning_effort
            )

        # SiliconFlow: ``enable_thinking`` bool + ``thinking_budget`` int.
        # Per-model reasoning-model gate + effort→budget mapping in
        # ``_siliconflow_thinking_params``.
        if self._profile and self._profile.reasoning_via_siliconflow_params:
            sf_params = _siliconflow_thinking_params(
                request.model, think, request.reasoning_effort
            )
            if sf_params:
                payload.update(sf_params)

        # Continue-with-trailing-assistant requests strip tool fields.
        # DeepSeek 400s on "Function call should not be used with
        # prefix" when prefix:true is set with tools, and any
        # provider continuing a trailing-assistant turn shouldn't be
        # making fresh tool decisions — the model is extending a
        # prior response verbatim.
        if is_continue_with_assistant_tail:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        return payload

    def _parse_openai_response(
        self, data: dict, *, model: str | None = None,
        thinking_enabled: bool | None = None,
        preserve_thinking: bool = False,
    ) -> InternalChatResponse:
        """Convert OpenAI response JSON to internal format."""
        choice = data.get("choices", [{}])[0]
        msg_data = choice.get("message", {})

        raw_content = msg_data.get("content") or ""
        # OpenRouter (and some unified gateways) return reasoning text in a
        # bare ``reasoning`` field rather than ``reasoning_content``; fall
        # back to it so OpenRouter reasoning isn't silently discarded.
        native_thinking = msg_data.get("reasoning_content") or msg_data.get("reasoning")
        clean_content, thinking_text = normalize_thinking(
            raw_content, native_thinking,
            model=model or data.get("model"),
            thinking_enabled=thinking_enabled,
            preserve_thinking=preserve_thinking,
            # End-of-response safety net: if the model routed 100% of its
            # output into reasoning (empty content with populated thinking),
            # promote reasoning → content so callers see the answer instead
            # of a mute "Thought for Ns" bubble. Observed on GLM-5.2 through
            # NVIDIA NIM under CC's tool-heavy prompt shape, and documented
            # for GLM-4.7-Flash upstream. Uses ``salvage_empty_content`` in
            # normalize_thinking rather than a per-provider workaround so
            # the fix covers every asymmetric-closer flake regardless of
            # backend / model / family.
            salvage_empty_content=True,
        )
        if not raw_content and native_thinking and clean_content:
            # Salvage fired — log so the flake is visible in ops without
            # hiding the underlying "empty content" event.
            log.warning(
                "reasoning_promoted_to_content",
                model=model or data.get("model"),
                thinking_bytes=len(native_thinking or ""),
                content_bytes=len(clean_content),
            )
        # Cache any per-call extra_content (Gemini thought_signature) before the
        # tool_calls flow downstream and lose it — re-attached at egress by id.
        self._capture_tool_call_extra(msg_data.get("tool_calls"))
        message = Message(
            role=msg_data.get("role", "assistant"),
            content=clean_content,
            tool_calls=msg_data.get("tool_calls"),
            thinking=thinking_text or None,
        )

        usage_data = data.get("usage", {}) or {}
        completion_details = usage_data.get("completion_tokens_details") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            # DeepSeek surfaces these at the top of ``usage`` (no
            # nesting). OpenAI nests cached tokens under
            # ``prompt_tokens_details.cached_tokens`` — read both. Miss
            # tokens are DeepSeek-only; OpenAI doesn't report a sibling.
            cache_hit_tokens=int(
                usage_data.get("prompt_cache_hit_tokens")
                or (usage_data.get("prompt_tokens_details") or {}).get("cached_tokens")
                or 0
            ),
            cache_miss_tokens=_cache_miss_tokens(usage_data),
            # OpenAI + DeepSeek V4 both nest reasoning under
            # ``completion_tokens_details.reasoning_tokens``.
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        )

        return InternalChatResponse(
            message=message,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )

    @staticmethod
    def _has_images(request: InternalChatRequest) -> bool:
        return any(m.images for m in request.messages)

    @staticmethod
    def _is_vision_rejected(status: int, body: str) -> bool:
        """Check if a backend error indicates vision/image_url is unsupported."""
        return status == 400 and "image_url" in body

    def _reasoning_safe_for_prefix(self, model: str) -> bool:
        """Return True iff the model can be safely used with native prefix completion.

        DeepSeek's reasoning lineup (V4 Pro, V4 Flash, V3.2, R1,
        Reasoner) accepts prefix-completion requests but routes 100%
        of generated tokens into ``reasoning_content`` — confirmed
        live with ``deepseek-v4-flash`` emitting 24KB of reasoning
        and 0 content bytes for a continue request. The visible
        bubble never updates. Only ``deepseek-chat`` (the explicit
        non-reasoning variant) routes prefix completion to visible
        content on DeepSeek today.

        Other prefix-supporting providers (none today; Anthropic
        and llama-server handle reasoning + prefix correctly via
        different code paths) default to True so we don't break
        them when they're added.
        """
        if not self._profile:
            return True
        # DeepSeek is the only currently-shipping provider with a
        # known reasoning/prefix incompatibility. Other providers
        # added later can layer in their own checks here.
        if self._profile.id == "deepseek":
            return "chat" in (model or "").lower()
        return True

    def _chat_url(self, request: InternalChatRequest) -> str:
        """Resolve the chat-completions URL for this request.

        Most requests go to ``{base_url}/chat/completions``. Prefix
        continuation requests against a provider with
        ``prefix_endpoint_override`` set (DeepSeek's ``/beta``) route to
        that override instead — DeepSeek's prefix feature is gated to
        the ``/beta`` path, but a user's stored provider often points
        at ``/v1``. The profile carries the canonical prefix URL so the
        Continue button works regardless of how the user originally
        configured the provider.
        """
        if (
            request.continue_last_assistant
            and self._profile
            and self._profile.supports_assistant_prefix
            and self._profile.prefix_endpoint_override
            # If the model fell back to synthetic-user (e.g. DeepSeek
            # reasoning lineup), don't route to /beta — the request
            # has no ``prefix: true`` marker, so the standard endpoint
            # works and matches what the user actually configured.
            and self._reasoning_safe_for_prefix(request.model)
        ):
            base = self._profile.prefix_endpoint_override.rstrip("/")
        else:
            base = self._base_url
        return self._apply_azure_api_version(f"{base}/chat/completions")

    def _apply_azure_api_version(self, url: str) -> str:
        """Append the mandatory ``?api-version=`` query param for Azure OpenAI.

        Azure's endpoint is ``{resource}/openai/deployments/{deployment}/
        chat/completions?api-version=YYYY-MM-DD`` — the deployment PATH comes
        from the user's per-provider ``base_url`` (the profile note says it's
        deployment-specific), but the version query param is required on every
        call and no other layer injects it, so a bare request 400s. Non-Azure
        profiles are untouched. If the user already encoded ``api-version`` in
        their base_url (e.g. a per-deployment version), that wins — we don't
        override it. Version comes from the ``azure_api_version`` setting
        (default ``2024-02-01``)."""
        if not (self._profile and self._profile.id == "azure"):
            return url
        if "api-version=" in url:
            return url
        from augmentum.config import settings
        version = (getattr(settings, "azure_api_version", "") or "2024-02-01").strip()
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}api-version={version}"

    @staticmethod
    def _is_transport_exception(exc: BaseException) -> bool:
        """True for httpx/httpcore transport failures without importing httpx.

        Some lightweight test environments import this module without httpx
        installed. Production always has httpx, but class/module-name detection
        keeps helper tests importable while still narrowing diagnostics to the
        provider transport family.
        """
        names = {cls.__name__ for cls in type(exc).mro()}
        if names.intersection({"TimeoutException", "TransportError"}):
            return True
        module = type(exc).__module__
        return module.startswith(("httpx", "httpcore")) and (
            "Timeout" in type(exc).__name__ or "Error" in type(exc).__name__
        )

    @staticmethod
    def _is_connect_retryable(exc: BaseException) -> bool:
        """True for connection-ESTABLISHMENT failures that are safe to retry.

        Narrower than ``_is_transport_exception``: only connect/pool timeouts
        and connect errors, which occur before any response bytes arrive, so a
        blind retry can't duplicate a partial stream. Read/stream-mid failures
        are deliberately excluded — those may have already emitted tokens.
        """
        name = type(exc).__name__
        return name in {"ConnectTimeout", "ConnectError", "PoolTimeout"}

    def _raise_transport_error(
        self, exc: BaseException, *, request: InternalChatRequest, url: str,
        payload: dict, stream: bool, retry: str = "",
    ) -> None:
        log.warning(
            "backend_chat_transport_error",
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
            model=request.model,
            url=self._sanitize_url_for_log(url),
            profile=getattr(self._profile, "id", "") if self._profile else "",
            stream=stream,
            retry=retry,
            **self._payload_diagnostics(payload),
        )
        raise RuntimeError(
            self._transport_error_message(exc, model=request.model, url=url)
        ) from exc

    async def _open_stream_response(
        self, request: InternalChatRequest, *, url: str, payload: dict,
        retry: str = "",
    ):
        # One retry on connection-establishment failure: a single slow TLS
        # handshake on an internet-remote provider used to kill the whole turn
        # even though a fresh connection succeeds in milliseconds. No bytes have
        # arrived at __aenter__, so re-opening can't duplicate a partial stream.
        for attempt in range(2):
            stream_ctx = self._chat_client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(),
            )
            try:
                resp = await stream_ctx.__aenter__()
            except Exception as exc:
                if not self._is_transport_exception(exc):
                    raise
                if attempt == 0 and self._is_connect_retryable(exc):
                    log.warning(
                        "backend_chat_connect_retry",
                        error_type=type(exc).__name__,
                        model=request.model,
                        url=url,
                    )
                    await asyncio.sleep(0.25)
                    continue
                self._raise_transport_error(
                    exc, request=request, url=url, payload=payload,
                    stream=True, retry=retry,
                )
            return stream_ctx, resp

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        payload = self._build_openai_payload(request)
        payload["stream"] = False
        url = self._chat_url(request)

        try:
            resp = await self._chat_client.post(
                url,
                json=payload,
                headers=self._headers(),
            )
        except Exception as exc:
            if not self._is_transport_exception(exc):
                raise
            # Transport failures (ConnectTimeout/PoolTimeout/etc.) often have
            # an empty ``str(exc)``. Log the class + safe destination here, at
            # the provider boundary where the payload/model context still exists.
            self._raise_transport_error(
                exc, request=request, url=url, payload=payload, stream=False,
            )
        if resp.status_code >= 400:
            body = sanitize_error_detail(resp.text[:500])
            # Retry without images if backend doesn't support vision
            if self._is_vision_rejected(resp.status_code, body) and self._has_images(request):
                log.info("vision_not_supported_retrying_text_only", url=url)
                payload = self._build_openai_payload(request, strip_images=True)
                payload["stream"] = False
                try:
                    resp = await self._chat_client.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                    )
                except Exception as exc:
                    if not self._is_transport_exception(exc):
                        raise
                    self._raise_transport_error(
                        exc, request=request, url=url, payload=payload,
                        stream=False, retry="text_only",
                    )
                if resp.status_code < 400:
                    return self._parse_openai_response(
                        resp.json(), model=request.model,
                        thinking_enabled=_effective_think(request),
                        preserve_thinking=bool(request.preserve_thinking),
                    )
                body = sanitize_error_detail(resp.text[:500])
            log.error(
                "backend_chat_error",
                status=resp.status_code,
                url=url,
                body=body,
                **self._payload_diagnostics(payload),
            )
            raise BackendError(
                f"Backend returned {resp.status_code}: {body}",
                retry_after=parse_retry_after(resp.headers, body),
                status=resp.status_code,
            )
        return self._parse_openai_response(
            resp.json(), model=request.model, thinking_enabled=_effective_think(request),
            preserve_thinking=bool(request.preserve_thinking),
        )

    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        # Record this turn's payload against the previous one so the
        # response-side audit can tell a genuinely-new prompt from a
        # cacheable prefix the provider re-charged anyway. No-ops without a
        # kv_session_key (external API clients), so only the in-app UI —
        # which sends X-Augmentum-Session — is tracked.
        self.track_prefix_stability(request)
        payload = self._build_openai_payload(request)
        payload["stream"] = True
        # Ask upstream to emit per-request token counts in the final
        # SSE chunk. OpenAI itself returns NO usage in streams without
        # this flag — our _iter_stream already reads it (line ~650),
        # but with no usage chunk to read the timer's eval_tokens stays
        # at its chunk-count approximation. Most non-OpenAI providers
        # send usage regardless; this flag is the polite way to request
        # it and any provider that doesn't recognise it ignores it.
        existing_stream_opts = payload.get("stream_options") or {}
        existing_stream_opts["include_usage"] = True
        payload["stream_options"] = existing_stream_opts
        url = self._chat_url(request)

        # local_engine gates the asymmetric "starts inside think" assumption:
        # valid only for a local llama-server (prompt-prefix opener injection),
        # WRONG for cloud hosts (NVIDIA NIM etc.) which would empty the visible
        # answer for GLM/DeepSeek-V4/Qwen3 plain-content replies. See #17.
        local_engine = self.is_local_engine()
        thinking_buf = ThinkingStreamBuffer(
            model=request.model, thinking_enabled=_effective_think(request),
            preserve_thinking=bool(request.preserve_thinking),
            local_engine=local_engine,
            # End-of-stream safety net for asymmetric-closer flake (see the
            # matching non-streaming ``salvage_empty_content`` note in
            # ``_parse_openai_response``). Emits accumulated reasoning as
            # content on flush when the visible stream stayed empty.
            salvage_empty_content=True,
        )

        stream_ctx, resp = await self._open_stream_response(
            request, url=url, payload=payload,
        )
        try:
            if resp.status_code >= 400:
                body = sanitize_error_detail(
                    (await resp.aread()).decode("utf-8", errors="replace")[:500]
                )
                # Retry without images if backend doesn't support vision
                if self._is_vision_rejected(resp.status_code, body) and self._has_images(request):
                    log.info("vision_not_supported_retrying_text_only", url=url)
                    payload = self._build_openai_payload(request, strip_images=True)
                    payload["stream"] = True
                    # Same stream_options as the initial request — without
                    # this the retry would silently drop the usage chunk.
                    payload["stream_options"] = {"include_usage": True}
                    # Fall through to retry below
                else:
                    log.error(
                        "backend_stream_error",
                        status=resp.status_code,
                        url=url,
                        body=body,
                        **self._payload_diagnostics(payload),
                    )
                    raise BackendError(
                        f"Backend returned {resp.status_code}: {body}",
                        retry_after=parse_retry_after(resp.headers, body),
                        status=resp.status_code,
                    )
            else:
                # Success — stream the response
                try:
                    async for chunk in self._iter_stream(resp, thinking_buf, request):
                        yield chunk
                finally:
                    # Flush any pending thinking content on normal exit or exception
                    flush_content, flush_thinking = thinking_buf.flush()
                    if flush_content or flush_thinking:
                        yield InternalStreamChunk(
                            content_delta=flush_content,
                            thinking_delta=flush_thinking,
                            done=True,
                        )
                    # Salvage: if the visible content stream stayed empty
                    # across the whole response but reasoning accumulated,
                    # promote it to content so the caller sees an answer.
                    salvaged = thinking_buf.salvage()
                    if salvaged:
                        log.warning(
                            "reasoning_promoted_to_content_stream",
                            model=request.model,
                            thinking_bytes=len(salvaged),
                        )
                        yield InternalStreamChunk(
                            content_delta=salvaged, done=True,
                        )
                return
        finally:
            await stream_ctx.__aexit__(None, None, None)

        # Retry path (text-only fallback after vision rejection)
        # Flush first buffer before creating a new one to avoid discarding content
        flush_content, flush_thinking = thinking_buf.flush()
        if flush_content or flush_thinking:
            yield InternalStreamChunk(
                content_delta=flush_content,
                thinking_delta=flush_thinking,
            )
        thinking_buf = ThinkingStreamBuffer(
            model=request.model, thinking_enabled=_effective_think(request),
            preserve_thinking=bool(request.preserve_thinking),
            local_engine=local_engine,
            # End-of-stream safety net for asymmetric-closer flake (see the
            # matching non-streaming ``salvage_empty_content`` note in
            # ``_parse_openai_response``). Emits accumulated reasoning as
            # content on flush when the visible stream stayed empty.
            salvage_empty_content=True,
        )
        stream_ctx, resp = await self._open_stream_response(
            request, url=url, payload=payload, retry="text_only",
        )
        try:
            if resp.status_code >= 400:
                body = sanitize_error_detail(
                    (await resp.aread()).decode("utf-8", errors="replace")[:500]
                )
                log.error(
                    "backend_stream_error",
                    status=resp.status_code,
                    url=url,
                    body=body,
                )
                raise BackendError(
                    f"Backend returned {resp.status_code}: {body}",
                    retry_after=parse_retry_after(resp.headers, body),
                    status=resp.status_code,
                )
            try:
                async for chunk in self._iter_stream(resp, thinking_buf, request):
                    yield chunk
            finally:
                # Flush any pending thinking content on normal exit or exception
                flush_content, flush_thinking = thinking_buf.flush()
                if flush_content or flush_thinking:
                    yield InternalStreamChunk(
                        content_delta=flush_content,
                        thinking_delta=flush_thinking,
                        done=True,
                    )
                # Salvage mirror of the success path — see the note there.
                salvaged = thinking_buf.salvage()
                if salvaged:
                    log.warning(
                        "reasoning_promoted_to_content_stream",
                        model=request.model,
                        thinking_bytes=len(salvaged),
                    )
                    yield InternalStreamChunk(
                        content_delta=salvaged, done=True,
                    )
        finally:
            await stream_ctx.__aexit__(None, None, None)

    async def _iter_stream(
        self,
        resp: httpx.Response,
        thinking_buf: ThinkingStreamBuffer,
        request: InternalChatRequest | None = None,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Iterate SSE lines from an OpenAI-compatible streaming response.

        Instance method (not static) so it can cache per-tool-call
        ``extra_content`` on ``self`` — Gemini's thought_signature must be
        re-attached at egress. See ``_capture_tool_call_extra``.

        ``request`` is threaded through only so the terminal usage chunk can
        run ``_audit_kv_reuse`` against the request-side contract; it stays
        optional so any other caller keeps working.
        """
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]  # Strip "data: " prefix
            if data_str == "[DONE]":
                flush_content, flush_thinking = thinking_buf.flush()
                if flush_content or flush_thinking:
                    yield InternalStreamChunk(
                        content_delta=flush_content,
                        thinking_delta=flush_thinking,
                        done=True,
                    )
                else:
                    yield InternalStreamChunk(done=True)
                return

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                log.warning("invalid_sse_data", data=data_str[:200])
                continue

            choices = data.get("choices") or [{}]
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            raw_content = delta.get("content", "")
            # OpenRouter streams reasoning in ``reasoning`` not
            # ``reasoning_content`` — fall back so it isn't dropped.
            native_thinking = (
                delta.get("reasoning_content") or delta.get("reasoning") or ""
            )
            clean_content, thinking = thinking_buf.process(
                raw_content, native_thinking
            )

            chunk = InternalStreamChunk(
                content_delta=clean_content,
                thinking_delta=thinking,
                role=delta.get("role"),
                model=data.get("model", ""),
                finish_reason=finish_reason,
                done=finish_reason is not None,
            )

            # Capture native tool calls from the delta
            if "tool_calls" in delta:
                tc_deltas = delta["tool_calls"]
                if tc_deltas:
                    chunk.augmentum = chunk.augmentum or {}
                    chunk.augmentum["tool_calls"] = tc_deltas
                    # Stash any per-call extra_content (Gemini thought_signature)
                    # so it can be re-attached when this call is echoed back —
                    # the downstream stream accumulator drops it. See
                    # _capture_tool_call_extra.
                    self._capture_tool_call_extra(tc_deltas)

            if "usage" in data and data["usage"]:
                u = data["usage"]
                completion_details = u.get("completion_tokens_details") or {}
                # llama-server (and proxies in front of it) attach a
                # ``timings`` block with the authoritative decode time.
                timings = data.get("timings") or {}
                chunk.usage = Usage(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    total_tokens=u.get("total_tokens", 0),
                    cache_hit_tokens=int(
                        u.get("prompt_cache_hit_tokens")
                        or (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                        or 0
                    ),
                    cache_miss_tokens=_cache_miss_tokens(u),
                    reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
                    eval_duration_ms=float(timings.get("predicted_ms") or 0.0),
                )
                # Join the response-side cache counts with the request-side
                # prefix contract. Reuses the bundled engine's audit verbatim —
                # here the token counts come from normalised usage rather than
                # llama-server ``timings``. Rides the ``augmentum`` block to the
                # UI exactly like the local path. Only fires when the provider
                # actually reported caching (cache_miss_tokens is derived only
                # then), so a no-telemetry provider stays silent.
                if request is not None:
                    kv_aug = self._audit_kv_reuse(
                        request,
                        evaluated_n=chunk.usage.cache_miss_tokens,
                        cache_n=chunk.usage.cache_hit_tokens,
                        endpoint="chat_completions_stream",
                    )
                    if kv_aug:
                        chunk.augmentum = {**(chunk.augmentum or {}), **kv_aug}

            yield chunk

    async def list_models(self) -> list[ModelInfo]:
        # Fast path: cached result still valid. Read happens outside
        # the lock so the hot path (single refresh tick, all backends
        # in parallel) doesn't serialise on a stale-check.
        cached = self._list_models_cache
        if cached is not None and (
            time.monotonic() - self._list_models_cached_at
        ) < self._LIST_MODELS_TTL_S:
            return cached

        # Single-flight: the second concurrent caller waits here and
        # picks up the first caller's result on lock release. Without
        # this the log showed every provider getting two simultaneous
        # /v1/models requests + two simultaneous /api/v0/models probes
        # on each UI refresh.
        async with self._list_models_lock:
            # Re-check under the lock — another coroutine may have
            # populated the cache while we were waiting.
            cached = self._list_models_cache
            if cached is not None and (
                time.monotonic() - self._list_models_cached_at
            ) < self._LIST_MODELS_TTL_S:
                return cached

            resp = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            # Try LM Studio native API for rich metadata (type: "vlm")
            lms_types = await self._probe_lmstudio_types()

            models: list[ModelInfo] = []
            for m in data.get("data", []):
                name = m.get("id", "")
                vision = self._detect_vision(m, name, lms_types)
                models.append(
                    ModelInfo(
                        name=name,
                        model=name,
                        modified_at=str(m.get("created", "")),
                        vision=vision,
                        context_length=self._pick_limit_int(
                            m, ("max_input_tokens", "context_length",
                                "context_window", "context_size", "max_model_len"),
                        ),
                        max_output=self._pick_limit_int(
                            m, ("max_completion_tokens", "max_output_tokens",
                                "max_tokens"),
                        ),
                    )
                )
            # Cache on success only. A failed upstream call shouldn't
            # poison the TTL window — the next caller should retry.
            self._list_models_cache = models
            self._list_models_cached_at = time.monotonic()
            return models

    @staticmethod
    def _detect_vision(raw: dict, name: str, lms_types: dict[str, str]) -> bool:
        """Detect VL capability from provider-specific metadata, then name fallback.

        Detection order (first match wins):
          1. OpenRouter:  architecture.input_modalities contains "image"
          2. Mistral:     capabilities.vision == true
          3. LM Studio:   native API type == "vlm"
          4. Fallback:    name-pattern heuristic
        """
        # OpenRouter — architecture.input_modalities: ["text", "image", ...]
        arch = raw.get("architecture")
        if isinstance(arch, dict):
            modalities = arch.get("input_modalities") or []
            return "image" in modalities

        # Mistral — capabilities: { vision: true, ... }
        caps = raw.get("capabilities")
        if isinstance(caps, dict) and "vision" in caps:
            return bool(caps["vision"])

        # LM Studio native API probe
        if lms_types and name in lms_types:
            return lms_types[name] == "vlm"

        # Last resort: name heuristic
        return is_vision_model_name(name)

    @staticmethod
    def _pick_limit_int(raw: dict, fields: tuple[str, ...]) -> int:
        """First positive int among ``fields`` in a provider /v1/models entry,
        checking the top level and the common nested containers (top_provider /
        capabilities / limits). Captures a real per-model context/output window
        at list time so /v1/models can report it instead of the per-PROVIDER
        profile fallback — exact for aggregators (OpenRouter) where the profile
        ceiling is meaningless. Mirrors the field list in get_context_length()."""
        if not isinstance(raw, dict):
            return 0
        for key in fields:
            v = raw.get(key)
            if isinstance(v, int) and v > 0:
                return v
            for ck in ("top_provider", "capabilities", "limits"):
                c = raw.get(ck)
                if isinstance(c, dict):
                    nv = c.get(key)
                    if isinstance(nv, int) and nv > 0:
                        return nv
        return 0

    def is_local_engine(self) -> bool:
        """Resolve from the configured base URL: a loopback / RFC-1918 /
        docker-compose host is one of our own local llama-server / vLLM /
        sglang instances (prompt-prefix opener injection applies); a cloud
        provider host is not. See ``ModelBackend.is_local_engine`` and #17.
        """
        return is_local_engine_url(self._base_url)

    def supported_sampler_params(self, model: str = "") -> set[str]:
        """Which sampler knobs actually reach the model — mirrors
        ``_build_openai_payload`` exactly so the Tuning UI never offers a dead
        control. ``temperature`` / ``top_p`` are universal on OpenAI-compat.
        ``presence_penalty`` is sent to every provider EXCEPT DeepSeek (which
        deprecated it — see the payload gate). ``top_k`` reaches local engines
        (vLLM/sglang/llama-server) or any cloud provider that documents it;
        ``min_p`` / ``repeat_penalty`` (→ wire ``repetition_penalty``) reach only
        cloud providers whose ``sampler_extras`` list them. Keep in lockstep
        with the payload builder — the two are one contract.
        """
        supported = {"temperature", "top_p"}
        if not (self._profile and self._profile.id == "deepseek"):
            supported.add("presence_penalty")
        extras = self._profile.sampler_extras if self._profile else {}
        if self.is_local_engine() or "top_k" in extras:
            supported.add("top_k")
        if "min_p" in extras:
            supported.add("min_p")
        if "repetition_penalty" in extras:
            supported.add("repeat_penalty")  # editor/source key
        return supported

    def is_vision_paired(self, model: str = "") -> bool:
        """True iff this provider can natively read image attachments for
        ``model`` — otherwise the route-layer caption fallback rewrites the
        images to text before the request reaches us.

        Unlike the local llama-server backend (one loaded model, one
        projector), an OpenAI-compat endpoint serves many models with mixed
        modality, so the decision is per-model:

          1. Profile flag / OpenAI-family catch-all (``effective_capability``)
             — the primary signal. Profiles default ``supports_vision=True``,
             so every currently-working VL path (OpenAI, OpenRouter, Mistral,
             Gemini …) is unchanged; a profile that ships zero vision models
             (DeepSeek) sets it False.
          2. Positive per-model catalog hit — a multi-model endpoint flagged
             text-only at the profile level can still host an individual VL
             model the catalog detected. Trust a confirmed ``vision=True``
             before forcing the caption fallback.

        Returns False only when there's no evidence of native vision, so a
        text-only cloud model gets captioned-to-text instead of having its
        image silently dropped at the API.
        """
        from augmentum.models.provider_profiles import effective_capability

        if effective_capability(self._profile, model, "supports_vision"):
            return True
        if model and self._list_models_cache:
            norm = _normalized_model_name(model)
            for mi in self._list_models_cache:
                if mi.vision and (
                    mi.name == model or _normalized_model_name(mi.name) == norm
                ):
                    return True
        return False

    # Cloud API domains that will never respond to /api/v0/models.
    # Skipping the probe at this layer kills the 404-noise visible in
    # the runtime logs for Cerebras / NVIDIA / etc. on every refresh
    # tick. Match by exact hostname (urlparse-derived); the suffix
    # check happens inline in _probe_lmstudio_types for the rarer
    # subdomain variants (e.g. `inference-time-cluster-X.aws.together`).
    _CLOUD_DOMAINS = frozenset((
        "api.openai.com", "api.deepseek.com", "openrouter.ai",
        "api.mistral.ai", "api.anthropic.com", "api.together.xyz",
        "api.groq.com", "api.fireworks.ai", "api.perplexity.ai",
        "generativelanguage.googleapis.com", "api.cohere.com",
        # Added 2026-06-08 after runtime logs showed these providers
        # 404'ing on /api/v0/models every refresh cycle. The probe is
        # an LM-Studio detection path; cloud providers will never
        # serve LM-Studio's native API.
        "api.cerebras.ai", "integrate.api.nvidia.com",
        "api.x.ai", "api.sambanova.ai",
    ))

    async def _probe_lmstudio_types(self) -> dict[str, str]:
        """Try LM Studio's native /api/v0/models for model type metadata.

        Returns {model_id: type_string} or empty dict if not LM Studio.
        Skips the probe entirely for known cloud API domains.
        """
        try:
            # Skip probe for known cloud APIs (they'll always 404)
            from urllib.parse import urlparse
            host = urlparse(self._base_url).hostname or ""
            if host in self._CLOUD_DOMAINS:
                return {}

            # Strip /v1 suffix to get the base origin for native API
            base = self._base_url
            if base.endswith("/v1"):
                base = base[:-3]
            elif base.endswith("/v1/"):
                base = base[:-4]
            resp = await self._client.get(
                f"{base}/api/v0/models",
                headers=self._headers(),
                timeout=3.0,
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
            return {
                m.get("id", ""): m.get("type", "")
                for m in data.get("data", [])
                if m.get("id")
            }
        except Exception:
            return {}

    async def show_model(self, name: str) -> ModelDetails:
        resp = await self._client.get(
            f"{self._base_url}/models/{name}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return ModelDetails(
            details=data,
        )

    async def get_context_length(self, model: str) -> int:
        """Resolve the context window for ``model`` in tokens.

        Two-tier lookup: per-model metadata first (Anthropic + OpenRouter
        + some forks return real numbers via /v1/models/{name}), profile-
        level ``max_context`` second (the conservative fallback shipped
        per-provider in ``provider_profiles.py``).

        Why this exists: the base implementation in ``ModelBackend``
        relies on ``model_info`` which OpenAI-compat providers don't
        populate. That made every cloud model report context_window=0
        upstream, which collapsed the Coder auto-compactor's threshold
        to ``DEFAULT_CODER_COMPACT_TOKENS`` (16K) — see
        ``coder/context_tokens.py:13``. The result: a turn on DeepSeek
        v4 Pro (128K) compacted as if it were a 16K-window model,
        burning the provider's real headroom for no reason.

        Field-name search covers the major variants seen in the wild:

          * Anthropic: ``max_input_tokens`` (only field on /v1/models/{name})
          * OpenRouter: ``context_length`` (also on /v1/models)
          * Together / Fireworks / some local proxies: ``context_window``
            or ``context_size``
          * vLLM-shaped: ``max_model_len`` (also used by some LMI variants)
          * Anything else: fall back to profile default.

        Never raises — failures here just yield 0 / fallback and let the
        caller use its conservative default. Tracked at warning-level
        only if telemetry is needed; chatty INFOs would spam on every
        turn-start probe.
        """
        try:
            details = await self.show_model(model)
        except Exception as exc:
            log.debug(
                "show_model_failed_for_context_length",
                model=model, error=str(exc)[:160],
            )
            details = None

        candidate_fields = (
            "max_input_tokens",
            "context_length",
            "context_window",
            "context_size",
            "max_model_len",
        )
        if details and isinstance(details.details, dict):
            for key in candidate_fields:
                value = details.details.get(key)
                if isinstance(value, int) and value > 0:
                    return value
                # Some providers nest under `top_provider` / `capabilities`
                # — quick second pass on common containers without going
                # full-recursive (which would be brittle).
                for container_key in ("top_provider", "capabilities", "limits"):
                    container = details.details.get(container_key)
                    if isinstance(container, dict):
                        nested = container.get(key)
                        if isinstance(nested, int) and nested > 0:
                            return nested

        # Profile fallback. ``max_context`` defaults to 128K on
        # ProviderProfile; profiles that ship a meaningful override
        # (xai=256K, moonshot=256K, etc.) win here.
        if self._profile is not None:
            profile_max = int(getattr(self._profile, "max_context", 0) or 0)
            if profile_max > 0:
                return profile_max
        return 0


def to_openai_chat_response(response: InternalChatResponse) -> dict:
    """Convert internal response to OpenAI chat completion format."""
    msg: dict = {
        "role": response.message.role,
        "content": response.message.content,
    }
    if response.message.thinking:
        msg["reasoning_content"] = response.message.thinking
    # Surface tool_calls so external agent clients (CC, Cursor agent mode,
    # SillyTavern function-calling) can drive their own tool loops via
    # the /v1/chat/completions surface. Without this the field is silently
    # dropped, which makes Augmentum look like a non-tool-capable backend
    # to OpenAI-compat tool callers.
    if response.message.tool_calls:
        msg["tool_calls"] = response.message.tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": _usage_to_openai_dict(response.usage),
    }


def _cache_miss_tokens(usage_data: dict) -> int:
    """Derive freshly-evaluated prompt tokens from an OpenAI-shaped usage block.

    DeepSeek reports ``prompt_cache_miss_tokens`` directly. OpenAI (and every
    proxy in front of it, including codex-proxy bridges) reports ONLY
    ``prompt_tokens_details.cached_tokens`` with no sibling miss count, so
    misses were previously hard-zero for the entire OpenAI family.

    That made a total cache failure indistinguishable from a provider that
    doesn't report caching: hits 0, misses 0, nothing emitted, and the UI's
    cache chip — which requires a non-zero hit count — renders nothing. A
    provider silently re-charging the full prompt every turn looked exactly
    like a provider with no cache telemetry at all.

    So when the provider proves it IS cache-aware by sending the details
    block at all, derive misses from the prompt total. Presence of the key
    is the signal, NOT its value: ``{"cached_tokens": 0}`` is a real report
    of a real 0% hit rate and must survive to the UI. Providers that send no
    details block at all still get 0, preserving "not reported".
    """
    explicit = usage_data.get("prompt_cache_miss_tokens")
    if explicit is not None:
        return int(explicit or 0)
    details = usage_data.get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        return 0
    prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
    cached = int(details.get("cached_tokens") or 0)
    return max(0, prompt_tokens - cached)


def _usage_to_openai_dict(usage: Usage) -> dict:
    """Serialize Usage to an OpenAI-shaped usage dict.

    Mirrors the request-side parsing: cache fields go top-level
    (DeepSeek shape — strict OpenAI nests under ``prompt_tokens_details``,
    but the DeepSeek shape round-trips cleanly through any OAI-compat
    client and avoids a second representation), reasoning tokens nest
    under ``completion_tokens_details`` per OpenAI's spec.

    Conditional emission: zero-valued cache/reasoning fields are omitted
    so non-reporting providers don't grow a synthetic ``"cache_hit_tokens": 0``
    line in every response.
    """
    out: dict = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cache_hit_tokens:
        out["prompt_cache_hit_tokens"] = usage.cache_hit_tokens
    if usage.cache_miss_tokens:
        out["prompt_cache_miss_tokens"] = usage.cache_miss_tokens
    if usage.cache_write_tokens:
        out["prompt_cache_write_tokens"] = usage.cache_write_tokens
    if usage.reasoning_tokens:
        out["completion_tokens_details"] = {"reasoning_tokens": usage.reasoning_tokens}
    return out


def to_openai_stream_chunk(
    chunk: InternalStreamChunk, chunk_id: str | None = None
) -> dict:
    """Convert internal stream chunk to OpenAI SSE chunk format."""
    delta: dict = {}
    if chunk.role:
        delta["role"] = chunk.role
    if chunk.content_delta:
        delta["content"] = chunk.content_delta

    return {
        "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": chunk.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": chunk.finish_reason,
            }
        ],
    }


def to_openai_models_response(models: list[ModelInfo]) -> dict:
    """Convert internal model list to OpenAI models response format."""
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "augmentum",
            }
            for m in models
        ],
    }
