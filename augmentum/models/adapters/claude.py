"""Claude (Anthropic) Messages API backend adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

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
)
from augmentum.models.converters.claude import (
    ClaudeConverter,
    apply_prompt_caching,
    convert_response,
    get_thinking_config,
    is_adaptive_model,
    is_no_prefill_model,
    is_no_sampling_model,
    is_thinking_model,
)
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known Claude models (no listing endpoint)
# ---------------------------------------------------------------------------

# Anthropic has no public model-listing endpoint, so this catalog is the
# single source of truth for both the picker and get_context_length.
# Ordered MOST-SPECIFIC first so prefix matching ("claude-opus-4-8" before
# the generic "claude-opus-4") resolves correctly. Retired 3.5/3.7 ids
# (404 as of 2026) removed. Frontier (4.6+/Fable-5) = 1M context.
_KNOWN_MODELS: list[dict[str, Any]] = [
    {"name": "claude-fable-5", "ctx": 1_000_000, "vision": True},
    {"name": "claude-opus-4-8", "ctx": 1_000_000, "vision": True},
    {"name": "claude-opus-4-7", "ctx": 1_000_000, "vision": True},
    {"name": "claude-opus-4-6", "ctx": 1_000_000, "vision": True},
    {"name": "claude-sonnet-4-6", "ctx": 1_000_000, "vision": True},
    {"name": "claude-opus-4-5", "ctx": 200_000, "vision": True},
    {"name": "claude-opus-4", "ctx": 200_000, "vision": True},
    {"name": "claude-sonnet-4", "ctx": 200_000, "vision": True},
    {"name": "claude-haiku-4-5", "ctx": 200_000, "vision": True},
    {"name": "claude-3-opus-latest", "ctx": 200_000, "vision": True},
]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class ClaudeBackend(ModelBackend):
    """Communicates with the Anthropic Claude Messages API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1",
        cache_enabled: bool = True,
        cache_ttl: str = "5m",
        thinking_effort: str = "medium",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl
        self._thinking_effort = thinking_effort
        self._converter = ClaudeConverter()

    # ---- Headers ----------------------------------------------------------

    def _headers(
        self, *, tools: bool = False, thinking: bool = False, model: str = ""
    ) -> dict[str, str]:
        """Build request headers with appropriate beta flags."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        beta_flags: list[str] = []
        if tools:
            beta_flags.append("tools-2024-04-04")
        # ``interleaved-thinking-2025-05-14`` is GA on the adaptive
        # (4.6+/Fable-5) generation and the migration guide says to remove
        # it there — sending it to an adaptive model is meaningless and a
        # documented don't. Keep it only for the older thinking models
        # (3.7 / 4.0) where it still applies.
        if thinking and not is_adaptive_model(model):
            beta_flags.append("interleaved-thinking-2025-05-14")
        if self._cache_enabled:
            beta_flags.append("prompt-caching-2024-07-31")
        if beta_flags:
            headers["anthropic-beta"] = ",".join(beta_flags)
        return headers

    # ---- Tool conversion --------------------------------------------------

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI tool format to Claude format.

        OpenAI: ``{type: "function", function: {name, description, parameters}}``
        Claude: ``{name, description, input_schema}``
        """
        if not tools:
            return None
        claude_tools: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function", tool)
            claude_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        return claude_tools

    # ---- Request building -------------------------------------------------

    def _build_request_body(
        self, request: InternalChatRequest
    ) -> dict[str, Any]:
        """Convert an InternalChatRequest to a Claude Messages API body."""
        # Convert messages from internal Message objects to dicts
        msg_dicts: list[dict[str, Any]] = []
        for msg in request.messages:
            d: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content or "",
            }
            if msg.tool_calls:
                d["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
                d["role"] = "tool"
            if msg.images:
                # Convert to OpenAI-style content array for the converter
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"type": "text", "text": msg.content})
                for img in msg.images:
                    parts.append({"type": "image_url", "image_url": {"url": img}})
                d["content"] = parts
            msg_dicts.append(d)

        # Continue-last-assistant handling for Claude. Anthropic's Messages
        # API natively continues a trailing ``role: assistant`` message —
        # no special flag, the model just extends it. Two constraints:
        #
        # 1. Trailing whitespace on the final assistant content causes a
        #    400 "final assistant content cannot end with trailing
        #    whitespace". Rstrip the string-content case here; vision
        #    content (list of blocks) is left alone since the converter
        #    handles its own normalization.
        # 2. Some Claude variants (Haiku 3.5, etc.) don't support
        #    assistant prefill. ``is_no_prefill_model`` flags those —
        #    fall back to inserting a synthetic user "continue from
        #    where you left off" message after the partial, matching
        #    the OpenAICompatBackend fallback path.
        if request.continue_last_assistant and msg_dicts and msg_dicts[-1].get("role") == "assistant":
            if is_no_prefill_model(request.model):
                msg_dicts.append({
                    "role": "user",
                    "content": (
                        "Continue from exactly where you left off in your previous "
                        "message. Do not re-introduce yourself, summarize, or repeat "
                        "anything. Pick up mid-sentence if that is where you stopped."
                    ),
                })
            else:
                last = msg_dicts[-1]
                if isinstance(last.get("content"), str):
                    last["content"] = last["content"].rstrip()

        # Use converter to get system + messages
        converted = self._converter.convert_messages(msg_dicts)
        system_blocks: list[dict[str, Any]] = converted["system"]
        claude_messages: list[dict[str, Any]] = converted["messages"]

        # Convert tools
        claude_tools = self._convert_tools(request.tools)

        # Apply prompt caching
        if self._cache_enabled:
            apply_prompt_caching(
                system_blocks, claude_tools, self._cache_ttl
            )

        # Build body
        max_tokens = request.max_tokens or 4096
        body: dict[str, Any] = {
            "model": request.model,
            "messages": claude_messages,
            "max_tokens": max_tokens,
        }

        if system_blocks:
            body["system"] = system_blocks

        if claude_tools:
            body["tools"] = claude_tools

        # Opus 4.7/4.8 + Fable 5 reject sampling params unconditionally
        # (400) — even with thinking off. Older models accept them; 4.6 /
        # Sonnet 4.6 accept them when thinking is off (the thinking block
        # below pops them in that case). Gate at the source for the
        # always-reject set.
        allow_sampling = not is_no_sampling_model(request.model)
        if allow_sampling and request.temperature is not None:
            body["temperature"] = request.temperature
        if allow_sampling and request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop_sequences"] = request.stop

        # Thinking
        use_thinking = request.think and is_thinking_model(request.model)
        if use_thinking:
            thinking_cfg = get_thinking_config(
                request.model, self._thinking_effort, max_tokens
            )
            body.update(thinking_cfg)
            # Claude disallows temperature with thinking
            body.pop("temperature", None)
            body.pop("top_p", None)

        return body

    # ---- Non-streaming chat -----------------------------------------------

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        """Send a non-streaming chat request to Claude."""
        body = self._build_request_body(request)
        use_thinking = request.think and is_thinking_model(request.model)

        resp = await self._client.post(
            f"{self._base_url}/messages",
            json=body,
            headers=self._headers(
                tools=bool(request.tools),
                thinking=use_thinking,
                model=request.model,
            ),
        )
        if resp.status_code >= 400:
            err = sanitize_error_detail(resp.text[:500])
            log.error(
                "claude_chat_error",
                status=resp.status_code,
                body=err,
            )
            raise BackendError(
                f"Claude API returned {resp.status_code}: {err}",
                retry_after=parse_retry_after(resp.headers, err),
                status=resp.status_code,
            )

        data = resp.json()
        normalised = convert_response(data)

        message = Message(
            role="assistant",
            content=normalised.get("content", ""),
            thinking=normalised.get("thinking"),
            tool_calls=normalised.get("tool_calls"),
        )

        usage_data = normalised.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cache_hit_tokens=int(usage_data.get("prompt_cache_hit_tokens") or 0),
            cache_miss_tokens=int(usage_data.get("prompt_cache_miss_tokens") or 0),
            cache_write_tokens=int(usage_data.get("prompt_cache_write_tokens") or 0),
        )

        return InternalChatResponse(
            message=message,
            model=normalised.get("model", request.model),
            finish_reason=normalised.get("finish_reason"),
            usage=usage,
        )

    # ---- Streaming chat ---------------------------------------------------

    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        """Send a streaming chat request and yield chunks."""
        body = self._build_request_body(request)
        body["stream"] = True
        use_thinking = request.think and is_thinking_model(request.model)

        async with self._client.stream(
            "POST",
            f"{self._base_url}/messages",
            json=body,
            headers=self._headers(
                tools=bool(request.tools),
                thinking=use_thinking,
                model=request.model,
            ),
        ) as resp:
            if resp.status_code >= 400:
                err_bytes = await resp.aread()
                err = sanitize_error_detail(
                    err_bytes.decode("utf-8", errors="replace")[:500]
                )
                log.error(
                    "claude_stream_error",
                    status=resp.status_code,
                    body=err,
                )
                raise BackendError(
                    f"Claude API returned {resp.status_code}: {err}",
                    retry_after=parse_retry_after(resp.headers, err),
                    status=resp.status_code,
                )

            # Track current content block type for routing deltas
            current_block_type: str = ""
            tool_json_buf: str = ""
            model_name: str = request.model

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    # Also handle event: lines (skip them)
                    continue

                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    log.warning("claude_invalid_sse", data=data_str[:200])
                    continue

                etype = event.get("type", "")

                if etype == "message_start":
                    msg = event.get("message", {})
                    model_name = msg.get("model", model_name)
                    yield InternalStreamChunk(
                        role="assistant",
                        model=model_name,
                    )

                elif etype == "content_block_start":
                    block = event.get("content_block", {})
                    current_block_type = block.get("type", "")
                    if current_block_type == "tool_use":
                        tool_json_buf = ""

                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")

                    if delta_type == "text_delta":
                        yield InternalStreamChunk(
                            content_delta=delta.get("text", ""),
                            model=model_name,
                        )

                    elif delta_type == "thinking_delta":
                        yield InternalStreamChunk(
                            thinking_delta=delta.get("thinking", ""),
                            model=model_name,
                        )

                    elif delta_type == "input_json_delta":
                        tool_json_buf += delta.get("partial_json", "")

                elif etype == "content_block_stop":
                    current_block_type = ""

                elif etype == "message_delta":
                    delta = event.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    usage_data = event.get("usage", {})

                    finish_map = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                        "stop_sequence": "stop",
                    }
                    finish = finish_map.get(stop_reason, stop_reason)

                    chunk = InternalStreamChunk(
                        model=model_name,
                        finish_reason=finish,
                        done=True,
                    )
                    if usage_data:
                        # Same reconstruction as the non-streaming path in
                        # converters/claude.py — Anthropic's ``input_tokens``
                        # is the fresh remainder, NOT the prompt total.
                        cache_read = int(usage_data.get("cache_read_input_tokens") or 0)
                        cache_write = int(
                            usage_data.get("cache_creation_input_tokens") or 0
                        )
                        raw_input = int(usage_data.get("input_tokens", 0) or 0)
                        prompt_tokens = raw_input + cache_read + cache_write
                        output_tokens = usage_data.get("output_tokens", 0)
                        chunk.usage = Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=prompt_tokens + output_tokens,
                            cache_hit_tokens=cache_read,
                            cache_miss_tokens=raw_input,
                            cache_write_tokens=cache_write,
                        )
                    yield chunk

    # ---- Model listing ----------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """Return hardcoded known Claude models."""
        return [
            ModelInfo(
                name=m["name"],
                model=m["name"],
                vision=m.get("vision", False),
                context_length=m.get("ctx", 200_000),
            )
            for m in _KNOWN_MODELS
        ]

    async def show_model(self, name: str) -> ModelDetails:
        """Return basic details for a Claude model."""
        return ModelDetails(
            details={"id": name, "owned_by": "anthropic"},
            family="claude",
        )

    async def get_context_length(self, model: str) -> int:
        """Return context length for a Claude model from the known catalog.

        Frontier models (Opus 4.6/4.7/4.8, Sonnet 4.6, Fable 5) are 1M;
        the rest are 200K. Driven off ``_KNOWN_MODELS`` (ordered
        most-specific first) so the catalog is the single source of truth
        and tolerates dated id suffixes (e.g. ``claude-opus-4-8-2026…``).
        """
        for entry in _KNOWN_MODELS:
            if entry["name"] in model:
                return int(entry["ctx"])
        return 200_000
