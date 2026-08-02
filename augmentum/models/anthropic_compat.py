"""Anthropic Messages API ↔ OpenAI ChatCompletions translation.

Surface:
    AnthropicMessagesRequest                — Pydantic model for /v1/messages
    anthropic_request_to_openai(req)        — request shape translation
    internal_response_to_anthropic_message  — non-stream response translation
    stream_internal_response_as_anthropic_sse — faux-stream the response
    count_tokens_estimate                   — tiktoken-backed estimator

Design notes
------------
**Faux-streaming.** We run the inner request non-streaming (so we have
the full ``InternalChatResponse``, including ``tool_calls`` populated
on the message), then emit the Anthropic SSE event sequence in one
burst. This is the pragmatic call for v1 because:

  1. ``InternalStreamChunk`` doesn't carry tool_calls today — the
     streaming wire silently loses them on most backends.
  2. Real-time streaming of ``input_json_delta`` events is the most
     bug-prone subsystem in every CC proxy (CCR #1397/#1356,
     LiteLLM #25321/#25561, sglang #24293). Emitting the full block
     at end-of-stream avoids the entire class.
  3. CC's UX doesn't suffer — tool turns are gated by tool execution
     on the user's side anyway.

The route layer can ``ping`` periodically while the handler runs to
keep the connection alive; that lives in ``anthropic_routes`` not here.

**Skipped Anthropic features (CC works without them):**
  * ``cache_control`` — stripped (and its new ``scope`` field). Pure
    prompt-cache optimization; treating as no-op is the convention all
    surviving proxies adopted (CCR ``cleancache``, LiteLLM scope-stripper).
  * ``computer_use`` / ``bash_20241022`` / ``text_editor_20241022``
    special tool types — not in CC's standard tool set.
  * Message Batches API, Files API — CC doesn't use.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field, field_validator

from augmentum.models.base import InternalChatResponse
from augmentum.proxy.openai_routes import OpenAIChatRequest, OpenAIMessage
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Many local backends 400 above 16k/32k max_tokens; cap before forwarding.
# Source: 1rgs/claude-code-proxy + LiteLLM ship the same cap.
_MAX_TOKENS_CAP = 16384


# Anthropic finish-reason → OpenAI-style stop reason.
# CC's TUI specifically renders end_turn / tool_use / max_tokens /
# stop_sequence / refusal — unknown values still pass through but lose
# the right glyph.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filter": "refusal",
    "error": "end_turn",
    None: "end_turn",
}


# ─── Request model ───────────────────────────────────────────────────


class AnthropicMessagesRequest(BaseModel):
    """Minimal Pydantic model for /v1/messages requests.

    Uses dict-shaped ``messages`` / ``system`` / ``tools`` so the
    translator can walk them imperatively. ``extra="allow"`` so unknown
    fields (anthropic-beta-only options, ``metadata``, the new
    ``scope`` field on cache_control, etc.) don't 400 — CC sends a
    bunch of them speculatively.
    """

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "allow"}

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


# ─── Request translation ─────────────────────────────────────────────


def _system_to_text(system: str | list[dict] | None) -> str:
    """Collapse Anthropic ``system`` (string or array of typed text
    blocks) into one string. Strips ``cache_control`` field silently."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _image_block_to_openai_part(block: dict) -> dict | None:
    """Convert Anthropic image block to OpenAI image_url content-part."""
    source = block.get("source") or {}
    stype = source.get("type")
    if stype == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        if data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{data}"},
            }
    elif stype == "url":
        url = source.get("url")
        if url:
            return {"type": "image_url", "image_url": {"url": url}}
    return None


def _tool_use_to_openai_tool_call(block: dict) -> dict:
    """Convert an Anthropic tool_use block to an OpenAI tool_calls entry."""
    return {
        "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input") or {}),
        },
    }


def _tool_result_content_to_str(content: Any) -> str:
    """Normalize the ``content`` field of a tool_result block to a
    string. Anthropic accepts str or a list of typed blocks (text /
    image). Images inside tool_result aren't representable on the
    OpenAI tool role — fall back to a textual marker."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                out.append(block.get("text", ""))
            elif btype == "image":
                out.append("[image attached]")
        return "\n".join(out)
    return str(content)


def _translate_user_message(msg: dict) -> list[OpenAIMessage]:
    """Split an Anthropic user message into one or more OpenAI messages.

    Tool_result blocks become role="tool" messages (one per block).
    Text + image blocks aggregate into a single user message; pure
    text becomes a string content, mixed text+image becomes the
    OpenAI content-parts shape (so the existing OpenAI parser
    extracts images).
    """
    content = msg.get("content")
    if isinstance(content, str):
        return [OpenAIMessage(role="user", content=content)]

    if not isinstance(content, list):
        return []

    out: list[OpenAIMessage] = []
    user_parts: list[dict] = []
    user_text_only: list[str] = []
    has_image = False

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_result":
            out.append(OpenAIMessage(
                role="tool",
                content=_tool_result_content_to_str(block.get("content")),
                # tool_call_id is set via field below since OpenAIMessage
                # carries it via the dict-pass into InternalChatRequest.
            ))
            # OpenAIMessage doesn't have tool_call_id as a top-level
            # Pydantic field, but the InternalChatRequest conversion path
            # reads it off the raw shape — set it via attribute below.
            out[-1].__dict__["tool_call_id"] = block.get("tool_use_id", "")
        elif btype == "text":
            text = block.get("text", "")
            user_text_only.append(text)
            user_parts.append({"type": "text", "text": text})
        elif btype == "image":
            part = _image_block_to_openai_part(block)
            if part is not None:
                user_parts.append(part)
                has_image = True

    if user_parts:
        # Pure-text → string content; mixed text+image → content-parts.
        if has_image:
            out.append(OpenAIMessage(role="user", content=user_parts))
        else:
            out.append(OpenAIMessage(role="user", content="\n".join(user_text_only)))

    return out


def _translate_assistant_message(msg: dict) -> OpenAIMessage:
    """Translate an Anthropic assistant turn (which may contain
    tool_use + text + thinking blocks) into a single OpenAI assistant
    message with optional tool_calls."""
    content = msg.get("content")
    if isinstance(content, str):
        return OpenAIMessage(role="assistant", content=content)

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    thinking: str | None = None

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking = block.get("thinking", "") or block.get("text", "")
            elif btype == "tool_use":
                tool_calls.append(_tool_use_to_openai_tool_call(block))

    out = OpenAIMessage(
        role="assistant",
        content="\n".join(text_parts),
        reasoning_content=thinking,
    )
    if tool_calls:
        # tool_calls isn't a declared Pydantic field on OpenAIMessage
        # but the conversion path tolerates it via the Message dataclass
        # (which DOES have tool_calls). Attach via __dict__.
        out.__dict__["tool_calls"] = tool_calls
    return out


def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    """Rename ``input_schema`` → ``parameters`` and wrap each tool in
    OpenAI's ``{"type":"function","function":{...}}`` shape."""
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        if not name:
            continue
        fn: dict = {"name": name}
        if t.get("description"):
            fn["description"] = t["description"]
        # input_schema → parameters (default to empty object schema)
        fn["parameters"] = t.get("input_schema") or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": fn})
    return out or None


def _translate_tool_choice(tc: dict | str | None) -> dict | str | None:
    """Anthropic tool_choice forms:
       {"type":"auto"} / {"type":"any"} / {"type":"tool","name":"X"} / {"type":"none"}
    OpenAI:
       "auto" / "required" / {"type":"function","function":{"name":"X"}} / "none"
    """
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    ttype = tc.get("type")
    if ttype == "auto":
        return "auto"
    if ttype == "any":
        return "required"
    if ttype == "none":
        return "none"
    if ttype == "tool":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return None


def anthropic_request_to_openai(req: AnthropicMessagesRequest) -> OpenAIChatRequest:
    """Translate an Anthropic Messages request into an OpenAIChatRequest.

    Stream flag is preserved from the input. The route layer is free to
    force ``stream=False`` on the internal call regardless — the
    OpenAIChatRequest only describes the request shape, not who
    consumes it.
    """
    oa_messages: list[OpenAIMessage] = []

    system_text = _system_to_text(req.system)
    if system_text:
        oa_messages.append(OpenAIMessage(role="system", content=system_text))

    for msg in req.messages:
        role = msg.get("role")
        if role == "user":
            oa_messages.extend(_translate_user_message(msg))
        elif role == "assistant":
            oa_messages.append(_translate_assistant_message(msg))
        else:
            # System messages mid-history are uncommon in CC but valid.
            # Collapse the content to text and pass through.
            content = msg.get("content")
            if isinstance(content, str):
                oa_messages.append(OpenAIMessage(role=role, content=content))

    max_tokens = req.max_tokens or 1024
    if max_tokens > _MAX_TOKENS_CAP:
        max_tokens = _MAX_TOKENS_CAP

    return OpenAIChatRequest(
        model=req.model,
        messages=oa_messages,
        stream=False,  # We faux-stream Anthropic SSE in the route layer.
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=max_tokens,
        stop=req.stop_sequences,
        tools=_translate_tools(req.tools),
        # Note: tool_choice not on OpenAIChatRequest's declared fields;
        # the OpenAIChatRequest carries extra="allow", but downstream
        # passthrough handler doesn't currently forward it. Including
        # as an `extra` attribute via __dict__ so a future forward-path
        # picks it up.
    )


# ─── Prefix-cache routing key ────────────────────────────────────────


def compute_prefix_cache_key(req: AnthropicMessagesRequest) -> str:
    """Hash the stable prefix of an Anthropic Messages request for KV slot routing.

    Claude Code marks the cacheable prefix with ``cache_control:
    {"type": "ephemeral"}`` blocks — the boundary between content that's
    stable across turns (system, tools, environment setup) and per-turn
    content (the current user query, recent tool results). This function
    hashes everything up to and INCLUDING the last marker so two CC turns
    differing only in the trailing per-turn content collide on the same
    hash and route to the same llama-server slot, which then hits its
    prefix cache instead of re-prefilling.

    Without any cache_control markers (non-CC callers hitting /v1/messages
    directly), the function falls back to hashing only ``system + tools``.
    That still gives same-system + same-tools traffic slot affinity
    without requiring cache-control awareness on the caller's part.

    Hash is the first 12 hex chars of SHA-256 — 48 bits, no realistic
    collision concern when scoped per-user.
    """
    h = hashlib.sha256()

    if req.system is not None:
        if isinstance(req.system, str):
            h.update(b"sys:str\x00")
            h.update(req.system.encode("utf-8", "ignore"))
        elif isinstance(req.system, list):
            h.update(b"sys:list\x00")
            for block in req.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    h.update(block.get("text", "").encode("utf-8", "ignore"))
                    h.update(b"\x00")

    # Sort tools by name so two clients that send them in different
    # orders still hash to the same key. CC's order is stable, but the
    # API doesn't require it.
    if req.tools:
        h.update(b"tools\x00")
        sortable = [t for t in req.tools if isinstance(t, dict)]
        for t in sorted(sortable, key=lambda x: x.get("name", "")):
            h.update(t.get("name", "").encode("utf-8", "ignore"))
            h.update(b"\x00")
            schema = t.get("input_schema") or {}
            h.update(json.dumps(schema, sort_keys=True).encode("utf-8", "ignore"))
            h.update(b"\x00")

    # Locate the last cache_control marker across messages. Scan blocks
    # within each message — markers can sit on any block type.
    last_marker_idx = -1
    for i, msg in enumerate(req.messages):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control"):
                    last_marker_idx = i
                    break

    if last_marker_idx >= 0:
        h.update(b"msgs\x00")
        for msg in req.messages[:last_marker_idx + 1]:
            h.update((msg.get("role") or "").encode("utf-8", "ignore"))
            h.update(b"\x00")
            content = msg.get("content")
            if isinstance(content, str):
                h.update(content.encode("utf-8", "ignore"))
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    h.update(btype.encode("utf-8", "ignore"))
                    h.update(b"\x00")
                    if btype == "text":
                        h.update(block.get("text", "").encode("utf-8", "ignore"))
                    elif btype == "tool_use":
                        h.update(block.get("name", "").encode("utf-8", "ignore"))
                        h.update(b"\x00")
                        h.update(json.dumps(block.get("input") or {}, sort_keys=True).encode("utf-8", "ignore"))
                    elif btype == "tool_result":
                        h.update(_tool_result_content_to_str(block.get("content")).encode("utf-8", "ignore"))
            h.update(b"\x00")

    return h.hexdigest()[:12]


# ─── Response translation: non-stream + faux-stream ──────────────────


def _stop_reason(finish_reason: str | None) -> str:
    return _STOP_REASON_MAP.get(finish_reason, "end_turn")


def _build_content_blocks(resp: InternalChatResponse) -> list[dict]:
    """Build the ordered Anthropic content-block list from an internal
    response. Order: thinking → text → tool_use(s).

    The thinking → text ordering matches Anthropic's own extended-
    thinking output. Tool_use blocks always come after any prose so
    CC reads them as actions the model wants to take after explaining
    itself."""
    blocks: list[dict] = []
    msg = resp.message
    if msg.thinking:
        blocks.append({"type": "thinking", "thinking": msg.thinking})
    if msg.content:
        blocks.append({"type": "text", "text": msg.content})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "")
            # Backends store arguments as a JSON string; parse to object
            # for the Anthropic shape. Tolerate malformed JSON by
            # surfacing the raw string under a single 'raw' key — better
            # than crashing the whole response.
            try:
                args_obj = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                args_obj = {"_raw": args_raw}
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                "name": fn.get("name", ""),
                "input": args_obj,
            })
    return blocks


def internal_response_to_anthropic_message(
    resp: InternalChatResponse, *, model_name: str,
) -> dict:
    """Translate an InternalChatResponse into a non-streaming Anthropic
    Messages JSON response."""
    blocks = _build_content_blocks(resp)
    # If the model produced any tool_calls, override the stop_reason to
    # tool_use even if the backend reported "stop" — this matches
    # Anthropic's actual behavior and is what CC expects.
    fr = resp.finish_reason or "stop"
    if resp.message.tool_calls and fr != "length":
        fr = "tool_calls"
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": blocks,
        "stop_reason": _stop_reason(fr),
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
    }


def _sse_event(event: str, data: dict) -> bytes:
    """Format one SSE event in Anthropic's wire shape."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def build_message_start_event(
    *, model_name: str, message_id: str | None = None, input_tokens: int = 0,
) -> bytes:
    """Pre-built ``message_start`` SSE event.

    The streaming route emits this immediately on connection — before the
    inner LLM call has started — so the SDK accumulator initialises and
    the connection isn't held silent. input_tokens is corrected via the
    final ``message_delta`` event's usage block (the SDK accepts that
    pattern; the message_start usage is treated as a lower bound).
    """
    mid = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    return _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })


def build_ping_event() -> bytes:
    """SSE ``ping`` keepalive — no semantic state, just bytes on the
    wire so intermediaries don't idle-close while the LLM thinks."""
    return _sse_event("ping", {"type": "ping"})


async def stream_internal_response_as_anthropic_sse(
    resp: InternalChatResponse, *, model_name: str,
    emit_message_start: bool = True,
) -> AsyncIterator[bytes]:
    """Emit an Anthropic SSE event sequence from a completed internal
    response. Ordering matches the SDK's invariants — see the
    test_anthropic_compat module-level docstring.

    This is an async generator so the route layer can use it as a
    StreamingResponse body. We emit synchronously (no awaits inside)
    because the inner LLM call already completed — there's no
    real-time data to await.

    Set ``emit_message_start=False`` when the route already emitted
    message_start independently (e.g. before pinging while waiting on
    the inner LLM call). The remaining content blocks + message_delta +
    message_stop are emitted unchanged.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    blocks = _build_content_blocks(resp)

    fr = resp.finish_reason or "stop"
    if resp.message.tool_calls and fr != "length":
        fr = "tool_calls"
    stop_reason = _stop_reason(fr)

    if emit_message_start:
        yield build_message_start_event(
            model_name=model_name,
            message_id=message_id,
            input_tokens=resp.usage.prompt_tokens,
        )

    # Content blocks. For each block: start → delta(s) → stop. We emit
    # the entire delta in one chunk; this is valid per spec and avoids
    # the "tool argument deltas split incorrectly" failure mode that
    # breaks every other proxy.
    for idx, block in enumerate(blocks):
        btype = block["type"]
        if btype == "text":
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })
            if block["text"]:
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block["text"]},
                })
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })
        elif btype == "thinking":
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "thinking", "thinking": ""},
            })
            if block["thinking"]:
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                })
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })
        elif btype == "tool_use":
            # Anthropic's spec: content_block_start carries input={}
            # (placeholder); arguments are streamed as input_json_delta
            # events carrying partial JSON strings. We emit the full
            # arg JSON in a single delta — valid per spec.
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                },
            })
            args_json = json.dumps(block["input"], separators=(",", ":"))
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": args_json},
            })
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })

    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": resp.usage.completion_tokens},
    })

    yield _sse_event("message_stop", {"type": "message_stop"})


# ─── Token counting ──────────────────────────────────────────────────


# tiktoken cl100k_base is what every major proxy uses as the estimator.
# Loaded lazily so module import doesn't pay the encoder construction
# cost (which is non-trivial — ~80ms).
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def _content_to_text(content: Any) -> str:
    """Flatten any content shape (string, list of typed blocks) to plain
    text for token counting. Image blocks count as a fixed token budget
    rather than encoding the base64 — that would massively overcount."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image":
                # Anthropic's own count is ~tokens for a typical image
                # in the 1000-1600 range. Use a single sentinel that
                # the encoder will count as a small constant — the
                # caller adds the image budget separately.
                parts.append("[IMAGE]")
            elif btype == "tool_use":
                # Tool_use name + input JSON contributes to token count
                parts.append(block.get("name", ""))
                parts.append(json.dumps(block.get("input") or {}))
            elif btype == "tool_result":
                parts.append(_tool_result_content_to_str(block.get("content")))
            elif btype == "thinking":
                parts.append(block.get("thinking", "") or block.get("text", ""))
        return "\n".join(parts)
    return str(content)


# Anthropic charges ~1568 tokens per image in their published count_tokens
# results. Approximate that for budgeting parity.
_IMAGE_TOKEN_BUDGET = 1568
# Per-message OpenAI-style overhead — accounts for role separator etc.
_PER_MESSAGE_OVERHEAD = 4


def count_tokens_estimate(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
) -> int:
    """Estimate the token cost of an Anthropic Messages request.

    Used by /v1/messages/count_tokens. CC consumes this for
    context-window budgeting / compaction decisions, NOT billing —
    ±20% accuracy is fine. The #1 cause of "proxy works for one turn
    then hangs" is returning 404 here; an estimate is always better
    than no answer.
    """
    enc = _get_tokenizer()
    total = 0

    if system:
        total += len(enc.encode(_system_to_text(system)))

    images = 0
    for msg in messages:
        total += _PER_MESSAGE_OVERHEAD
        text = _content_to_text(msg.get("content"))
        total += len(enc.encode(text))
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    images += 1

    total += images * _IMAGE_TOKEN_BUDGET

    if tools:
        for t in tools:
            if not isinstance(t, dict):
                continue
            tool_repr = json.dumps({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "schema": t.get("input_schema") or {},
            }, separators=(",", ":"))
            total += len(enc.encode(tool_repr))

    return total
