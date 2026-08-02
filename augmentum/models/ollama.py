"""Ollama backend implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

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
from augmentum.utils.thinking import detect_reasoning_family
from augmentum.utils.logging import get_logger
from augmentum.utils.thinking import ThinkingStreamBuffer, normalize_thinking

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


class OllamaBackend(ModelBackend):
    """Communicates with an Ollama instance."""

    # Ollama's runner slot reuse benefits from the same stable
    # mid-conversation system injection pattern that llama-server uses;
    # both prefix-match at the token level.
    supports_mid_conversation_system = True

    def is_local_engine(self) -> bool:
        """Resolve from the configured host. Ollama returns reasoning in a
        native ``thinking`` field so the asymmetric assumption is largely
        moot here, but report locality honestly for callers that introspect
        the active backend (e.g. the passthrough soft-trigger probe). See
        ``ModelBackend.is_local_engine``.
        """
        from augmentum.models.openai_compat import is_local_engine_url

        return is_local_engine_url(self._base_url)

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    def _build_ollama_payload(self, request: InternalChatRequest) -> dict:
        """Convert internal request to Ollama API format."""
        messages = []
        for msg in request.messages:
            m: dict = {"role": msg.role, "content": msg.content}
            if msg.images:
                m["images"] = msg.images
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            messages.append(m)

        payload: dict = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
        }

        # Build options dict from individual params + raw passthrough
        options: dict = {}
        if request.raw_options:
            options.update(request.raw_options)
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop
        if request.frequency_penalty is not None:
            options["repeat_penalty"] = request.frequency_penalty
        if request.seed is not None:
            options["seed"] = request.seed
        if options:
            payload["options"] = options

        if request.format:
            payload["format"] = request.format
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        if request.tools:
            payload["tools"] = request.tools
        if request.think or detect_reasoning_family(model=request.model):
            payload["think"] = bool(request.think)

        return payload

    def _parse_ollama_response(
        self, data: dict, *, model: str | None = None
    ) -> InternalChatResponse:
        """Convert Ollama response JSON to internal format."""
        msg_data = data.get("message", {})
        raw_content = msg_data.get("content", "")
        native_thinking = msg_data.get("thinking")
        clean_content, thinking_text = normalize_thinking(
            raw_content, native_thinking, model=model or data.get("model"),
        )
        message = Message(
            role=msg_data.get("role", "assistant"),
            content=clean_content,
            tool_calls=msg_data.get("tool_calls"),
            thinking=thinking_text or None,
        )

        # Extract timing stats
        timing = {}
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            if key in data:
                timing[key] = data[key]

        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )

        done_reason = data.get("done_reason")

        return InternalChatResponse(
            message=message,
            model=data.get("model", ""),
            finish_reason=done_reason,
            usage=usage,
            timing=timing if timing else None,
        )

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        payload = self._build_ollama_payload(request)
        payload["stream"] = False

        resp = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        return self._parse_ollama_response(resp.json(), model=request.model)

    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        payload = self._build_ollama_payload(request)
        payload["stream"] = True

        thinking_buf = ThinkingStreamBuffer(model=request.model)

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("invalid_ndjson_line", line=line[:200])
                    continue

                msg = data.get("message", {})
                done = data.get("done", False)

                raw_content = msg.get("content", "")
                native_thinking = msg.get("thinking", "")
                clean_content, thinking = thinking_buf.process(
                    raw_content, native_thinking
                )

                chunk = InternalStreamChunk(
                    content_delta=clean_content,
                    thinking_delta=thinking,
                    role=msg.get("role") if msg.get("role") else None,
                    model=data.get("model", request.model),
                    done=done,
                )

                if done:
                    chunk.finish_reason = data.get("done_reason")
                    chunk.usage = Usage(
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                        total_tokens=(
                            data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        ),
                    )
                    # Flush any remaining buffer
                    flush_content, flush_thinking = thinking_buf.flush()
                    if flush_content:
                        chunk.content_delta += flush_content
                    if flush_thinking:
                        chunk.thinking_delta += flush_thinking

                yield chunk

    async def list_models(self) -> list[ModelInfo]:
        resp = await self._client.get(f"{self._base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            details = m.get("details") or {}
            # Ollama VL models have "clip" in the families list
            families = details.get("families") or []
            vision = "clip" in families or is_vision_model_name(name)
            models.append(
                ModelInfo(
                    name=name,
                    model=m.get("model", name),
                    size=m.get("size", 0),
                    digest=m.get("digest", ""),
                    modified_at=m.get("modified_at", ""),
                    details=m.get("details"),
                    vision=vision,
                )
            )
        return models

    async def show_model(self, name: str) -> ModelDetails:
        resp = await self._client.post(
            f"{self._base_url}/api/show",
            json={"name": name},
        )
        resp.raise_for_status()
        data = resp.json()
        return ModelDetails(
            modelfile=data.get("modelfile", ""),
            parameters=data.get("parameters", ""),
            template=data.get("template", ""),
            details=data.get("details"),
            model_info=data.get("model_info"),
        )
