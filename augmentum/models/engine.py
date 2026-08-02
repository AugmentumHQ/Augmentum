"""Augmentum Engine backend — native integration with prefix cache, session persistence, and VRAM coordination.

The engine exposes an OpenAI-compatible /v1/chat/completions API plus
Augmentum-native endpoints for prefix registration, session save/restore,
and VRAM budget queries.
"""

from __future__ import annotations

import asyncio
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
    v1_entry_is_vision,
)
from augmentum.utils.logging import get_logger
from augmentum.utils.thinking import (
    ThinkingStreamBuffer,
    detect_reasoning_family,
    normalize_thinking,
)

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

# llama.cpp sampling params carried in ``raw_options`` that map directly onto
# llama-server's /v1/chat/completions payload (mirrors llama_cpp.py's
# ``_LLAMACPP_PASSTHROUGH_PARAMS``). The per-model sampling profiles
# (sampling_profiles.py) write top_k/min_p/repeat_penalty/etc. into
# ``raw_options``; without forwarding them the Slot-A engine silently drops
# them and the model runs on llama-server's generic defaults (e.g. Qwen3's
# recommended top_k=20 → server default 40). grammar/json_schema are handled
# via ``response_format`` and deliberately excluded here.
_ENGINE_SAMPLING_PASSTHROUGH = frozenset({
    "top_k", "min_p", "typical_p", "repeat_penalty", "repeat_last_n",
    "mirostat", "mirostat_tau", "mirostat_eta",
    "dynatemp_range", "dynatemp_exponent",
    "dry_multiplier", "dry_base", "dry_allowed_length",
    "dry_penalty_last_n", "dry_sequence_breakers",
    "xtc_probability", "xtc_threshold",
})


class AugmentumEngineBackend(ModelBackend):
    """Communicates with the Augmentum Engine's OpenAI-compatible API
    and exposes engine-native features (prefix cache, sessions, VRAM)."""

    # The Augmentum Engine talks to the bundled llama-server subprocess
    # via ``llama_server_manager``. It inherits llama-server's slot
    # reuse, which prefix-matches at the token level — the same reason
    # ``LlamaCppBackend`` opts in.
    supports_mid_conversation_system = True

    def is_local_engine(self) -> bool:
        """Always True — Engine v2 drives the bundled local llama-server,
        whose ``--jinja`` template injects the reasoning opener into the
        prompt prefix. See ``ModelBackend.is_local_engine``.
        """
        return True

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        # Prefix cache: hash of system messages → registered prefix_id
        self._prefix_cache: dict[str, str] = {}
        # model name → resolved reasoning family (from GGUF arch). Detection
        # by model NAME alone misses custom-named finetunes whose arch is
        # authoritative; we resolve once per model and cache.
        self._family_cache: dict[str, str | None] = {}

    async def _resolve_family(self, model: str | None) -> str | None:
        """Resolve the reasoning family for the loaded GGUF, authoritatively.

        Name-only detection (``detect_reasoning_family(model=...)``) fails for
        custom-named finetunes — e.g. ``Qwythos-9B-Claude-Mythos-5-1M`` is a
        ``qwen35`` model (an asymmetric-closer family) but matches no name
        needle, so its chain-of-thought leaks into content. The GGUF's
        ``general.architecture`` is the model author's declared identity, so
        read it from the path llama-server reports at ``/props`` and feed it
        to ``detect_reasoning_family``. Cached per model name; on any failure
        we fall back to name-only detection (no regression).

        Must be called when the model is actually loaded (after a chat
        response or after the stream opens) so ``/props`` reflects it.
        """
        key = model or ""
        if key in self._family_cache:
            return self._family_cache[key]
        arch: str | None = None
        try:
            resp = await self._client.get(f"{self._base_url}/props")
            if resp.status_code < 400:
                props = resp.json()
                model_path = (
                    props.get("model_path")
                    or (props.get("default_generation_settings") or {}).get("model")
                )
                if model_path:
                    from augmentum.models.model_profile_cache import (
                        peek_gguf_string_keys,
                    )
                    peeked = await asyncio.to_thread(
                        peek_gguf_string_keys, model_path, {"general.architecture"}
                    )
                    arch = (peeked.get("general.architecture") or "").strip() or None
        except Exception as exc:
            log.debug("engine_family_resolve_failed", model=key[:80], error=str(exc)[:200])
        family = detect_reasoning_family(model=model, arch=arch)
        # Only cache once we had a working /props round-trip (arch resolved);
        # otherwise retry next call so a cold-load race doesn't pin us to the
        # name-only fallback for the life of the process.
        if arch is not None:
            self._family_cache[key] = family
        return family

    def _compute_prefix_hash(self, messages: list) -> str:
        """Hash the system/injection messages that form the stable prefix.

        The prefix = all messages before the first user message that isn't
        a system prompt. This typically includes: system prompt, character
        card, memory injections, datetime context.
        """
        import hashlib
        prefix_parts = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                prefix_parts.append(msg.get("content", ""))
            else:
                break  # first non-system message = end of prefix
        if not prefix_parts:
            return ""
        combined = "\n".join(prefix_parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def _ensure_prefix(self, messages: list) -> str:
        """Register the stable prefix if not already cached. Returns prefix_id."""
        prefix_hash = self._compute_prefix_hash(messages)
        if not prefix_hash:
            return ""

        # Already registered?
        if prefix_hash in self._prefix_cache:
            return self._prefix_cache[prefix_hash]

        # Extract system messages for registration
        system_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                break

        if not system_msgs:
            return ""

        # Register with engine
        prefix_id = await self.register_prefix(f"pfx_{prefix_hash}", system_msgs)
        if prefix_id:
            self._prefix_cache[prefix_hash] = prefix_id
        return prefix_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _build_payload(self, request: InternalChatRequest) -> dict:
        """Convert internal request to OpenAI API format."""
        messages = []
        for msg in request.messages:
            m: dict = {"role": msg.role, "content": self._build_vision_content(msg)}
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

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            payload["seed"] = request.seed
        # Forward llama.cpp sampling params (top_k/min_p/repeat_penalty/…) that
        # the per-model sampling profiles wrote into raw_options — llama-server
        # accepts them as extensions on /v1/chat/completions. Without this the
        # profile's top_k/min_p never reach the model. See llama_cpp.py.
        if request.raw_options:
            for key in _ENGINE_SAMPLING_PASSTHROUGH:
                if key in request.raw_options:
                    payload[key] = request.raw_options[key]
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = request.chat_template_kwargs

        if request.format == "json":
            payload["response_format"] = {"type": "json_object"}

        return payload

    @staticmethod
    def _parse_response(
        data: dict, *, model: str | None = None,
        thinking_enabled: bool | None = None,
        family: str | None = None,
    ) -> InternalChatResponse:
        """Convert OpenAI response JSON to internal format."""
        choice = data.get("choices", [{}])[0]
        msg_data = choice.get("message", {})

        raw_content = msg_data.get("content", "")
        native_thinking = msg_data.get("reasoning_content")
        clean_content, thinking_text = normalize_thinking(
            raw_content, native_thinking,
            family=family,
            model=model or data.get("model"),
            thinking_enabled=thinking_enabled,
        )
        message = Message(
            role=msg_data.get("role", "assistant"),
            content=clean_content,
            tool_calls=msg_data.get("tool_calls"),
            thinking=thinking_text or None,
        )

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return InternalChatResponse(
            message=message,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    async def chat(self, request: InternalChatRequest, prefix_id: str | None = None) -> InternalChatResponse:
        payload = self._build_payload(request)
        payload["stream"] = False

        # Auto-register prefix for KV cache reuse (skip re-eval of system prompt)
        if not prefix_id:
            prefix_id = await self._ensure_prefix(payload.get("messages", []))
        if prefix_id:
            payload["prefix_id"] = prefix_id

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
        except Exception as exc:
            log.error("engine_chat_connection_error", url=self._base_url, error=str(exc))
            raise RuntimeError(f"Engine connection failed: {exc}") from exc

        if resp.status_code >= 400:
            body = resp.text[:500]
            log.error(
                "engine_chat_error",
                status=resp.status_code,
                url=f"{self._base_url}/v1/chat/completions",
                body=body,
            )
            raise RuntimeError(f"Engine returned {resp.status_code}: {body}")

        # Resolve the reasoning family from the loaded GGUF arch now that the
        # model has responded (so /props reflects it). Authoritative for
        # custom-named finetunes that name-only detection would miss.
        family = await self._resolve_family(request.model)
        return self._parse_response(
            resp.json(), model=request.model, thinking_enabled=request.think,
            family=family,
        )

    async def chat_stream(
        self, request: InternalChatRequest
    ) -> AsyncIterator[InternalStreamChunk]:
        payload = self._build_payload(request)
        payload["stream"] = True

        # Auto-register prefix for KV cache reuse
        prefix_id = await self._ensure_prefix(payload.get("messages", []))
        if prefix_id:
            payload["prefix_id"] = prefix_id

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    log.error(
                        "engine_stream_error",
                        status=resp.status_code,
                        url=f"{self._base_url}/v1/chat/completions",
                        body=body,
                    )
                    raise RuntimeError(f"Engine returned {resp.status_code}: {body}")

                # Build the buffer AFTER the stream opens: the model is loaded
                # by now, so /props reflects it and we can resolve the family
                # from the GGUF arch (authoritative — name-only detection
                # misses custom-named finetunes and leaks their CoT into
                # content during streaming, which can't be fixed retroactively
                # once a delta is emitted).
                family = await self._resolve_family(request.model)
                thinking_buf = ThinkingStreamBuffer(
                    family=family,
                    model=request.model, thinking_enabled=request.think,
                    preserve_thinking=bool(request.preserve_thinking),
                )

                try:
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
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
                            log.warning("engine_invalid_sse_data", data=data_str[:200])
                            continue

                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        raw_content = delta.get("content", "")
                        native_thinking = delta.get("reasoning_content", "")
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

                        # Capture native tool_calls deltas from the stream.
                        # Without this, the streaming path drops every
                        # tool_call llama-server emits — fatal for
                        # native-strategy coder mode and any other
                        # streaming agent flow. Mirrors the OpenAI-compat
                        # backend's pattern at openai_compat.py:438.
                        if "tool_calls" in delta:
                            tc_deltas = delta["tool_calls"]
                            if tc_deltas:
                                chunk.augmentum = chunk.augmentum or {}
                                chunk.augmentum["tool_calls"] = tc_deltas

                        if "usage" in data and data["usage"]:
                            u = data["usage"]
                            chunk.usage = Usage(
                                prompt_tokens=u.get("prompt_tokens", 0),
                                completion_tokens=u.get("completion_tokens", 0),
                                total_tokens=u.get("total_tokens", 0),
                            )

                        yield chunk
                finally:
                    flush_content, flush_thinking = thinking_buf.flush()
                    if flush_content or flush_thinking:
                        yield InternalStreamChunk(
                            content_delta=flush_content,
                            thinking_delta=flush_thinking,
                            done=True,
                        )
        except Exception as exc:
            if "Engine returned" in str(exc):
                raise
            log.error("engine_stream_connection_error", url=self._base_url, error=str(exc))
            raise RuntimeError(f"Engine connection failed: {exc}") from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models")
        except Exception as exc:
            log.warning("engine_list_models_failed", url=self._base_url, error=str(exc))
            return []

        if resp.status_code >= 400:
            log.warning("engine_list_models_error", status=resp.status_code)
            return []

        data = resp.json()
        models = []
        for m in data.get("data", []):
            name = m.get("id", "")
            models.append(
                ModelInfo(
                    name=name,
                    model=name,
                    modified_at=str(m.get("created", "")),
                    # Honor the server's own capability claim (capabilities/
                    # modalities), not just the name heuristic — same fix as
                    # LlamaCppBackend, so a VL GGUF served via the engine slot
                    # isn't silently flagged text-only and made to play blind.
                    vision=v1_entry_is_vision(m),
                )
            )
        return models

    async def show_model(self, name: str) -> ModelDetails:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/engine/status")
        except Exception as exc:
            log.warning("engine_show_model_failed", error=str(exc))
            return ModelDetails()

        if resp.status_code >= 400:
            return ModelDetails()

        data = resp.json()
        return ModelDetails(
            details=data,
            model_info=data.get("model_info"),
        )

    # ------------------------------------------------------------------
    # Augmentum-native features
    # ------------------------------------------------------------------

    async def register_prefix(self, name: str, messages: list) -> str:
        """Register a prefix (system prompt + few-shot) for KV cache reuse.

        POST /v1/prefix/register
        Returns the prefix ID on success, empty string on failure.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/prefix/register",
                json={"name": name, "messages": messages},
            )
        except Exception as exc:
            log.warning("engine_register_prefix_failed", name=name, error=str(exc))
            return ""

        if resp.status_code >= 400:
            log.warning("engine_register_prefix_error", name=name, status=resp.status_code)
            return ""

        data = resp.json()
        prefix_id = data.get("prefix_id", data.get("id", ""))
        log.info("engine_prefix_registered", name=name, prefix_id=prefix_id)
        return prefix_id

    async def save_session(self, session_id: str) -> bool:
        """Persist the engine's KV cache state for a session.

        POST /v1/session/save
        Returns True on success.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/session/save",
                json={"session_id": session_id},
            )
        except Exception as exc:
            log.warning("engine_save_session_failed", session_id=session_id, error=str(exc))
            return False

        if resp.status_code >= 400:
            log.warning("engine_save_session_error", session_id=session_id, status=resp.status_code)
            return False

        log.info("engine_session_saved", session_id=session_id)
        return True

    async def restore_session(self, session_id: str) -> bool:
        """Restore a previously-saved KV cache state for a session.

        POST /v1/session/restore
        Returns True on success.
        """
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/session/restore",
                json={"session_id": session_id},
            )
        except Exception as exc:
            log.warning("engine_restore_session_failed", session_id=session_id, error=str(exc))
            return False

        if resp.status_code >= 400:
            log.warning("engine_restore_session_error", session_id=session_id, status=resp.status_code)
            return False

        log.info("engine_session_restored", session_id=session_id)
        return True

    async def get_vram_budget(self) -> dict:
        """Query the engine's current VRAM allocation and availability.

        GET /v1/vram/budget
        Returns budget dict or empty dict on failure.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/v1/vram/budget")
        except Exception as exc:
            log.warning("engine_vram_budget_failed", error=str(exc))
            return {}

        if resp.status_code >= 400:
            log.warning("engine_vram_budget_error", status=resp.status_code)
            return {}

        return resp.json()

    async def kv_stats(self) -> dict:
        """Query KV cache stats: sequence utilization, prefix cache, host cache.

        GET /v1/engine/kv-stats
        Returns stats dict or empty dict on failure.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/v1/engine/kv-stats")
        except Exception as exc:
            log.warning("engine_kv_stats_failed", error=str(exc))
            return {}

        if resp.status_code >= 400:
            log.warning("engine_kv_stats_error", status=resp.status_code)
            return {}

        return resp.json()

    async def engine_status(self) -> dict:
        """Query the engine's overall status.

        GET /v1/engine/status
        Returns status dict or empty dict on failure.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/v1/engine/status")
        except Exception as exc:
            log.warning("engine_status_failed", error=str(exc))
            return {}

        if resp.status_code >= 400:
            log.warning("engine_status_error", status=resp.status_code)
            return {}

        return resp.json()

    async def load_model(self, name: str) -> bool:
        """Load a model on the engine (uses warm pool for fast swap)."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/models/load",
                json={"model": name},
                timeout=600,
            )
            return resp.status_code == 200
        except Exception as exc:
            log.warning("engine_load_model_failed", model=name, error=str(exc))
            return False

    async def unload_model(self, name: str = "") -> bool:
        """Unload the current model from the engine."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/models/unload",
                json={},
            )
            return resp.status_code == 200
        except Exception as exc:
            log.warning("engine_unload_model_failed", error=str(exc))
            return False

    async def get_loaded_model(self) -> dict | None:
        """Get info about the currently loaded model on the engine."""
        try:
            resp = await self._client.get(f"{self._base_url}/v1/engine/status")
            if resp.status_code == 200:
                data = resp.json()
                model = data.get("model", {})
                if model.get("loaded"):
                    return {
                        "name": model.get("name", ""),
                        "state": model.get("state", ""),
                        "n_ctx": model.get("n_ctx", 0),
                        "backend": "engine",
                    }
            return None
        except Exception:
            return None
