"""Google Gemini API backend adapter.

Supports both AI Studio (API-key auth) and Vertex AI (Bearer token).
"""

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
    is_vision_model_name,
)
from augmentum.models.converters.gemini import (
    GeminiConverter,
    convert_response,
    get_safety_settings,
    get_thinking_config,
    usage_from_metadata,
)
from augmentum.models.thinking_control import resolve_thinking
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


def _usage_from_meta(usage_meta: dict) -> Usage:
    """Build a :class:`Usage` from Gemini's raw ``usageMetadata`` block.

    Bridges the wire-shaped dict from ``converters.gemini.usage_from_metadata``
    (``prompt_cache_hit_tokens``, matching the Anthropic converter's naming)
    onto ChatUsage's field names. Both streaming sites and the non-streaming
    path go through here so context-cache reporting can't drift between them
    again.
    """
    fields = usage_from_metadata(usage_meta or {})
    return Usage(
        prompt_tokens=fields["prompt_tokens"],
        completion_tokens=fields["completion_tokens"],
        total_tokens=fields["total_tokens"],
        cache_hit_tokens=fields["prompt_cache_hit_tokens"],
        cache_miss_tokens=fields["prompt_cache_miss_tokens"],
        reasoning_tokens=fields["reasoning_tokens"],
    )


class GeminiBackend(ModelBackend):
    """Communicates with the Google Gemini API (AI Studio or Vertex)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com",
        api_version: str = "v1beta",
        vertex: bool = False,
        vertex_project: str = "",
        vertex_region: str = "us-central1",
        thinking_effort: str = "medium",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._vertex = vertex
        self._vertex_project = vertex_project
        self._vertex_region = vertex_region
        self._thinking_effort = thinking_effort
        self._converter = GeminiConverter()

    # ---- URL / headers ----------------------------------------------------

    def _endpoint(
        self, model: str, method: str, *, stream: bool = False
    ) -> str:
        """Build the full endpoint URL for a Gemini API call."""
        # Model ids surfaced by the OpenAI-compat listing (and stored in
        # settings) carry a ``models/`` prefix; the native REST path below
        # already injects ``/models/``, so a prefixed id would double into
        # ``/models/models/<id>`` → 404. Normalise to the bare id.
        if model.startswith("models/"):
            model = model[len("models/"):]
        # Defense-in-depth against the load-balancer disambiguation suffix
        # ("<model>@<backend>", e.g. gemini-2.5-flash@google-gemini-3). The
        # resolver strips it, but a caller that discards the resolved clean
        # name (e.g. chat_routes reusing the raw request model) could still
        # leak it here, and the composite 404s. Gemini model ids never
        # contain "@", so dropping everything from the first "@" is safe.
        if "@" in model:
            model = model.split("@", 1)[0]
        if self._vertex:
            return (
                f"https://{self._vertex_region}-aiplatform.googleapis.com"
                f"/v1/projects/{self._vertex_project}"
                f"/locations/{self._vertex_region}"
                f"/publishers/google/models/{model}:{method}"
            )
        # AI Studio
        url = (
            f"{self._base_url}/{self._api_version}"
            f"/models/{model}:{method}?key={self._api_key}"
        )
        if stream:
            url += "&alt=sse"
        return url

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._vertex:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ---- Tool conversion --------------------------------------------------

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI tool format to Gemini function_declarations."""
        if not tools:
            return None
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function", tool)
            declarations.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        return [{"function_declarations": declarations}]

    # ---- Request building -------------------------------------------------

    def _build_body(self, request: InternalChatRequest) -> dict[str, Any]:
        """Convert an InternalChatRequest to a Gemini API request body."""
        # Convert messages from internal Message objects to dicts.
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
                d["name"] = msg.tool_call_id
            if msg.images:
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"type": "text", "text": msg.content})
                for img in msg.images:
                    parts.append({"type": "image_url", "image_url": {"url": img}})
                d["content"] = parts
            msg_dicts.append(d)

        converted = self._converter.convert_messages(msg_dicts)

        body: dict[str, Any] = {
            "contents": converted["contents"],
        }

        if converted["systemInstruction"]:
            body["systemInstruction"] = converted["systemInstruction"]

        # Generation config.
        gen_config: dict[str, Any] = {}
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
        if request.stop:
            gen_config["stopSequences"] = request.stop
        if request.seed is not None:
            gen_config["seed"] = request.seed

        # Thinking config. Resolve via the shared policy and ALWAYS send an
        # explicit directive (enable OR disable) when the model is thinking-
        # capable — Gemini reasons by DEFAULT, so omitting it on an off setting
        # left the model thinking (which silently broke the tiny-budget
        # goal-judge one-shot). Not sent for non-thinking Gemini models.
        _td = resolve_thinking(
            request.model,
            think=bool(request.think),
            effort=self._thinking_effort,
        )
        if _td.capable:
            thinking = get_thinking_config(
                request.model,
                self._thinking_effort,
                max_tokens=request.max_tokens or 8192,
                enabled=_td.enabled,
            )
            if thinking:
                gen_config.update(thinking)

        if gen_config:
            body["generationConfig"] = gen_config

        # Safety settings.
        body["safetySettings"] = get_safety_settings(vertex=self._vertex)

        # Tools.
        gemini_tools = self._convert_tools(request.tools)
        if gemini_tools:
            body["tools"] = gemini_tools

        return body

    # ---- Non-streaming chat -----------------------------------------------

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        """Send a non-streaming chat request to Gemini."""
        body = self._build_body(request)
        url = self._endpoint(request.model, "generateContent")

        resp = await self._client.post(url, json=body, headers=self._headers())
        if resp.status_code >= 400:
            err = sanitize_error_detail(resp.text[:500])
            log.error("gemini_chat_error", status=resp.status_code, body=err)
            raise BackendError(
                f"Gemini API returned {resp.status_code}: {err}",
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
            reasoning_tokens=int(usage_data.get("reasoning_tokens") or 0),
        )

        return InternalChatResponse(
            message=message,
            model=normalised.get("model") or request.model,
            finish_reason=normalised.get("finish_reason"),
            usage=usage,
        )

    # ---- Streaming chat ---------------------------------------------------

    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        """Send a streaming request and yield chunks."""
        body = self._build_body(request)
        url = self._endpoint(
            request.model, "streamGenerateContent", stream=True
        )

        async with self._client.stream(
            "POST", url, json=body, headers=self._headers()
        ) as resp:
            if resp.status_code >= 400:
                err_bytes = await resp.aread()
                err = sanitize_error_detail(
                    err_bytes.decode("utf-8", errors="replace")[:500]
                )
                log.error("gemini_stream_error", status=resp.status_code, body=err)
                raise BackendError(
                    f"Gemini API returned {resp.status_code}: {err}",
                    retry_after=parse_retry_after(resp.headers, err),
                    status=resp.status_code,
                )

            # Yield initial role chunk.
            yield InternalStreamChunk(role="assistant", model=request.model)

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    log.warning("gemini_invalid_sse", data=data_str[:200])
                    continue

                # Each SSE event is a full generateContent response chunk.
                candidates = event.get("candidates", [])
                if not candidates:
                    # May be a usage-only final event.
                    usage_meta = event.get("usageMetadata")
                    if usage_meta:
                        yield InternalStreamChunk(
                            model=request.model,
                            done=True,
                            usage=_usage_from_meta(usage_meta),
                        )
                    continue

                candidate = candidates[0]
                content_obj = candidate.get("content", {})
                parts = content_obj.get("parts", [])

                for part in parts:
                    if part.get("thought"):
                        # Thinking part.
                        yield InternalStreamChunk(
                            thinking_delta=part.get("text", ""),
                            model=request.model,
                        )
                    elif "text" in part:
                        yield InternalStreamChunk(
                            content_delta=part["text"],
                            model=request.model,
                        )
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        # Emit function call as content (JSON) for downstream
                        # to parse, matching the pattern from other backends.
                        yield InternalStreamChunk(
                            content_delta=json.dumps({
                                "function_call": {
                                    "name": fc.get("name", ""),
                                    "arguments": fc.get("args", {}),
                                }
                            }),
                            model=request.model,
                        )

                # Check finish reason.
                finish_reason = candidate.get("finishReason")
                if finish_reason:
                    from augmentum.models.converters.gemini import (
                        _FINISH_REASON_MAP,
                    )

                    mapped = _FINISH_REASON_MAP.get(finish_reason, finish_reason)
                    usage_meta = event.get("usageMetadata", {})
                    chunk = InternalStreamChunk(
                        model=request.model,
                        finish_reason=mapped,
                        done=True,
                    )
                    if usage_meta:
                        chunk.usage = _usage_from_meta(usage_meta)
                    yield chunk

    # ---- Model listing ----------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """List available Gemini models from the API."""
        url = (
            f"{self._base_url}/{self._api_version}/models"
            f"?key={self._api_key}"
        )
        try:
            resp = await self._client.get(url, headers=self._headers())
            if resp.status_code >= 400:
                log.warning("gemini_list_models_error", status=resp.status_code)
                return []

            data = resp.json()
            models: list[ModelInfo] = []
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue
                name = m.get("name", "")
                # Strip "models/" prefix.
                if name.startswith("models/"):
                    name = name[7:]
                models.append(ModelInfo(
                    name=name,
                    model=name,
                    vision=is_vision_model_name(name),
                    context_length=m.get("inputTokenLimit", 0),
                ))
            return models
        except Exception:
            log.warning("gemini_list_models_failed", exc_info=True)
            return []

    async def show_model(self, name: str) -> ModelDetails:
        """Return basic details for a Gemini model."""
        return ModelDetails(
            details={"id": name, "owned_by": "google"},
            family="gemini",
        )

    async def get_context_length(self, model: str) -> int:
        """Return context length for a Gemini model.

        Flash models: 1M, Pro models: 2M, default: 128K.
        """
        lower = model.lower()
        if "pro" in lower:
            return 2_000_000
        if "flash" in lower:
            return 1_000_000
        return 128_000
