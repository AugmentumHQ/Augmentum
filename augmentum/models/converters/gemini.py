"""Google Gemini API message converter.

Converts between Augmentum's internal message format and the
Google Gemini generateContent API format.  Supports both AI Studio
and Vertex AI safety-setting variants.
"""

from __future__ import annotations

import re
from typing import Any

from augmentum.models.converters.utils import extract_system_prefix

# ---------------------------------------------------------------------------
# Thinking-config model detection
# ---------------------------------------------------------------------------

_GEMINI_3_FLASH_RE = re.compile(r"gemini-3.*flash", re.IGNORECASE)
_GEMINI_3_PRO_RE = re.compile(r"gemini-3.*pro", re.IGNORECASE)
_GEMINI_25_FLASH_RE = re.compile(r"gemini-2\.5.*flash", re.IGNORECASE)
_GEMINI_25_PRO_RE = re.compile(r"gemini-2\.5.*pro", re.IGNORECASE)


# ---------------------------------------------------------------------------
# GeminiConverter
# ---------------------------------------------------------------------------


class GeminiConverter:
    """Convert internal messages to Gemini ``contents`` format."""

    # Gemini only has ``user`` and ``model`` roles.
    _ROLE_MAP: dict[str, str] = {
        "user": "user",
        "assistant": "model",
        "system": "user",
        "tool": "user",
    }

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        prefill: str = "",
    ) -> dict[str, Any]:
        """Convert internal messages to Gemini payload components.

        Returns a dict with:
          - ``systemInstruction`` (dict or None)
          - ``contents`` (list of Gemini content objects)
        """
        system_msgs, remaining = extract_system_prefix(messages)

        # Build systemInstruction from leading system messages.
        system_instruction: dict[str, Any] | None = None
        if system_msgs:
            parts = [{"text": m["content"]} for m in system_msgs]
            system_instruction = {"parts": parts}

        # Convert remaining messages to Gemini contents.
        contents: list[dict[str, Any]] = []
        for msg in remaining:
            role = self._ROLE_MAP.get(msg.get("role", "user"), "user")
            parts = self._message_to_parts(msg)
            if not parts:
                parts = [{"text": ""}]

            if contents and contents[-1]["role"] == role:
                # Merge consecutive same-role messages.
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        return {
            "systemInstruction": system_instruction,
            "contents": contents,
        }

    # ---- Part conversion helpers ------------------------------------------

    def _message_to_parts(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a single message to a list of Gemini parts."""
        parts: list[dict[str, Any]] = []
        role = msg.get("role", "user")

        # Handle tool results (tool role -> functionResponse).
        if role == "tool":
            return self._tool_result_parts(msg)

        content = msg.get("content", "")

        # Handle multimodal content arrays (OpenAI format).
        if isinstance(content, list):
            parts.extend(self._multimodal_parts(content, msg.get("name")))
        elif content:
            text = content
            name = msg.get("name")
            if name:
                text = f"{name}: {text}"
            parts.append({"text": text})

        # Handle assistant tool_calls -> functionCall parts.
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    import json

                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                parts.append({
                    "functionCall": {
                        "name": fn.get("name", ""),
                        "args": args,
                    }
                })

        return parts

    def _multimodal_parts(
        self, content: list[dict[str, Any]], name: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style content array to Gemini parts."""
        parts: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                if name and not parts:
                    text = f"{name}: {text}"
                parts.append({"text": text})
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                inline = self._parse_data_uri(url)
                if inline:
                    parts.append({"inlineData": inline})
        return parts

    @staticmethod
    def _parse_data_uri(url: str) -> dict[str, str] | None:
        """Parse ``data:<mime>;base64,<data>`` into inlineData dict."""
        if not url.startswith("data:"):
            return None
        # data:image/png;base64,iVBOR...
        try:
            header, data = url.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
            return {"mimeType": mime, "data": data}
        except (IndexError, ValueError):
            return None

    def _tool_result_parts(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a tool result message to functionResponse parts."""
        # Try to get tool name from the message; fall back to tool_call_id.
        name = msg.get("name", msg.get("tool_call_id", "tool"))
        content = msg.get("content", "")
        return [{
            "functionResponse": {
                "name": name,
                "response": {
                    "name": name,
                    "content": content,
                },
            }
        }]


# ---------------------------------------------------------------------------
# Safety settings
# ---------------------------------------------------------------------------

_BASE_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
]

_VERTEX_EXTRA_CATEGORIES = [
    "HARM_CATEGORY_IMAGE_HATE",
    "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
    "HARM_CATEGORY_IMAGE_HARASSMENT",
    "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_JAILBREAK",
]


def get_safety_settings(*, vertex: bool = False) -> list[dict[str, str]]:
    """Return safety settings with all categories disabled (OFF)."""
    cats = list(_BASE_CATEGORIES)
    if vertex:
        cats.extend(_VERTEX_EXTRA_CATEGORIES)
    return [
        {"category": cat, "threshold": "OFF"}
        for cat in cats
    ]


# ---------------------------------------------------------------------------
# Thinking configuration
# ---------------------------------------------------------------------------

# Gemini 3 Flash thinking-level mappings.
_G3_FLASH_LEVELS: dict[str, str] = {
    "min": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
}

# Gemini 3 Pro thinking-level mappings.
_G3_PRO_LEVELS: dict[str, str] = {
    "min": "low",
    "low": "low",
    "medium": "low",
    "high": "high",
    "max": "high",
}


def get_thinking_config(
    model: str,
    effort: str,
    *,
    max_tokens: int = 8192,
    enabled: bool = True,
) -> dict[str, Any] | None:
    """Return Gemini thinkingConfig for the given model and effort.

    ``enabled=False`` returns an EXPLICIT disable config (``thinkingBudget`` 0 /
    the floor ``thinkingLevel``) rather than None — Gemini 2.5+/3.x reason by
    DEFAULT, so omitting the config leaves the model thinking anyway (this is
    what silently broke the tiny-budget goal-judge one-shot). Returns None only
    when the model has no thinking control at all.
    """
    effort = (effort or "medium").lower()

    # Flash-Lite (2.5 or 3.x) — takes thinkingBudget and DEFAULTS to thinking
    # off; matched before the generic Flash branch because thinkingLevel
    # "minimal" still reasons, which defeats an off setting on small one-shots.
    # 0 fully disables.
    if re.search(r"gemini.*flash.*lite|gemini.*lite.*flash", model, re.IGNORECASE):
        budget = max(0, min(_effort_to_budget(effort, max_tokens), 24576)) if enabled else 0
        return {"thinkingConfig": {"thinkingBudget": budget, "includeThoughts": enabled}}

    # Gemini 3 Flash — thinkingLevel string ("minimal" is the floor).
    if _GEMINI_3_FLASH_RE.search(model):
        level = _G3_FLASH_LEVELS.get(effort, "medium") if enabled else "minimal"
        return {"thinkingConfig": {"thinkingLevel": level, "includeThoughts": enabled}}

    # Gemini 3 Pro — thinkingLevel string ("low" is the floor).
    if _GEMINI_3_PRO_RE.search(model):
        level = _G3_PRO_LEVELS.get(effort, "low") if enabled else "low"
        return {"thinkingConfig": {"thinkingLevel": level, "includeThoughts": enabled}}

    # Gemini 2.5 Flash — thinkingBudget int, capped [0, 24576] (0 = off).
    if _GEMINI_25_FLASH_RE.search(model):
        budget = max(0, min(_effort_to_budget(effort, max_tokens), 24576)) if enabled else 0
        return {"thinkingConfig": {"thinkingBudget": budget, "includeThoughts": enabled}}

    # Gemini 2.5 Pro — thinkingBudget int, capped [128, 32768] (128 = floor).
    if _GEMINI_25_PRO_RE.search(model):
        budget = max(128, min(_effort_to_budget(effort, max_tokens), 32768)) if enabled else 128
        return {"thinkingConfig": {"thinkingBudget": budget, "includeThoughts": enabled}}

    # Fallback: a Gemini thinking family not matched above (new point release,
    # alias). Prefer thinkingBudget (broadly accepted; 0 = off) so control
    # works regardless of the exact model id.
    from augmentum.models.thinking_control import detect_thinking_family
    if detect_thinking_family(model) in ("gemini_25", "gemini_3"):
        budget = max(0, min(_effort_to_budget(effort, max_tokens), 24576)) if enabled else 0
        return {"thinkingConfig": {"thinkingBudget": budget, "includeThoughts": enabled}}

    return None


def _effort_to_budget(effort: str, max_tokens: int) -> int:
    """Map effort string to a thinking budget integer."""
    pct_map = {
        "min": 0.0,
        "low": 0.10,
        "medium": 0.25,
        "high": 0.50,
        "max": 0.95,
    }
    pct = pct_map.get(effort, 0.25)
    return int(max_tokens * pct)


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}


def usage_from_metadata(usage_meta: dict[str, Any]) -> dict[str, Any]:
    """Normalise Gemini's ``usageMetadata`` block onto our usage surface.

    Shared by the non-streaming converter and BOTH streaming sites in
    ``adapters/gemini.py``; those three used to build the dict inline and
    only the non-streaming one was ever updated, so context-cache hits were
    reported on some paths and silently dropped on others.

    Unlike Anthropic, Gemini's ``promptTokenCount`` ALREADY includes the
    cached prefix, so misses are the remainder rather than a reconstructed
    sum. Gemini has no write-token concept — cache creation is a separate
    explicit CachedContent API call, not a per-request side effect — so
    ``cache_write`` stays 0 here.
    """
    prompt = int(usage_meta.get("promptTokenCount") or 0)
    completion = int(usage_meta.get("candidatesTokenCount") or 0)
    cached = int(usage_meta.get("cachedContentTokenCount") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage_meta.get("totalTokenCount") or 0)
        or prompt + completion,
        "prompt_cache_hit_tokens": cached,
        "prompt_cache_miss_tokens": max(0, prompt - cached),
        # Gemini bills thinking separately from candidates and reports it
        # in the same block; ChatUsage already has a home for it.
        "reasoning_tokens": int(usage_meta.get("thoughtsTokenCount") or 0),
    }


def convert_response(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a Gemini generateContent response to internal format.

    Extracts text, tool calls, and usage from the response.
    """
    result: dict[str, Any] = {
        "content": "",
        "thinking": None,
        "tool_calls": None,
        "finish_reason": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model": data.get("modelVersion", ""),
    }

    candidates = data.get("candidates", [])
    if not candidates:
        return result

    candidate = candidates[0]

    # Finish reason.
    raw_reason = candidate.get("finishReason", "")
    result["finish_reason"] = _FINISH_REASON_MAP.get(raw_reason, raw_reason)

    # Extract parts.
    content_obj = candidate.get("content", {})
    parts = content_obj.get("parts", [])

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        # Skip thought parts.
        if part.get("thought"):
            continue

        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": fc.get("args", {}),
                },
            })

    result["content"] = "".join(text_parts)
    if tool_calls:
        result["tool_calls"] = tool_calls
        if not result["finish_reason"] or result["finish_reason"] == "stop":
            result["finish_reason"] = "tool_calls"

    # Usage metadata.
    usage_meta = data.get("usageMetadata", {})
    if usage_meta:
        result["usage"] = usage_from_metadata(usage_meta)

    return result
