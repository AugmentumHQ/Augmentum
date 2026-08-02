"""response_format json_schema downgrade for non-supporting providers (2026-06-16).

The voice-router classifier attaches a JSON Schema via
``raw_options["json_schema"]`` so a tiny non-reasoning model emits a
guaranteed-valid verdict. The openai_compat adapter turned that into
``response_format: {"type": "json_schema", ...}`` for EVERY provider —
but DeepSeek's API rejects that type outright (``400 "This response_format
type is unavailable now"``) rather than ignoring it. Every classifier call
that fell back to a DeepSeek primary 400'd → regex-dropped the utterance
("Sorry, I didn't quite hear that").

Fix: gate emission on ``supports_response_format_json_schema`` (default
True — OpenAI, local llama-server/sglang/vLLM, and the SmolLM sidecar all
honor json_schema). DeepSeek's profile sets it False → demote to
``json_object`` (valid-JSON mode; the prompt names the shape).

These tests pin: DeepSeek demotes, supporting providers and the no-profile
default keep json_schema, and the schema is only touched when present.
"""

from __future__ import annotations

import httpx

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import PROFILES

_SCHEMA = {
    "type": "object",
    "properties": {"goal": {"type": "string"}},
    "required": ["goal"],
}


def _backend(profile_id: str | None) -> OpenAIBackend:
    profile = PROFILES[profile_id] if profile_id else None
    base = profile.base_url if profile else "http://localhost:8090/v1"
    return OpenAIBackend(httpx.AsyncClient(), base, None, profile=profile)


def _req(model: str = "test-model", *, with_schema: bool = True) -> InternalChatRequest:
    raw = {"json_schema": _SCHEMA, "json_schema_name": "voice_router_verdict"} if with_schema else {}
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="reply with ONLY the JSON object")],
        raw_options=raw,
    )


def test_top_k_forwarded_only_when_set():
    # The classifier path (Gemma 4) sets top_k; cloud calls leave it None.
    # When None it must NOT appear in the payload (real OpenAI 400s on it).
    payload = _backend("openai")._build_openai_payload(
        InternalChatRequest(model="gpt-5.5", messages=[Message(role="user", content="hi")])
    )
    assert "top_k" not in payload
    # When set (local classifier), it rides along for the llama-server.
    payload = _backend(None)._build_openai_payload(
        InternalChatRequest(
            model="classifier", messages=[Message(role="user", content="hi")],
            temperature=1.0, top_p=0.95, top_k=64,
        )
    )
    assert payload["top_k"] == 64
    assert payload["top_p"] == 0.95
    assert payload["temperature"] == 1.0


def test_deepseek_profile_flags_json_schema_unsupported():
    # Guard against the flag silently flipping back in a future edit.
    assert PROFILES["deepseek"].supports_response_format_json_schema is False


def test_deepseek_demotes_json_schema_to_json_object():
    # The reported bug: DeepSeek 400s on json_schema. Must demote.
    payload = _backend("deepseek")._build_openai_payload(_req(model="deepseek-v4-pro"))
    assert payload["response_format"] == {"type": "json_object"}


def test_openai_keeps_strict_json_schema():
    payload = _backend("openai")._build_openai_payload(_req(model="gpt-5.5"))
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "voice_router_verdict"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == _SCHEMA


def test_no_profile_defaults_to_json_schema():
    # Anonymous endpoints (local llama-server / sglang / SmolLM sidecar with
    # no catalog entry) must keep the strict form — no regression.
    payload = _backend(None)._build_openai_payload(_req())
    assert payload["response_format"]["type"] == "json_schema"


def test_no_response_format_without_schema():
    # The downgrade path must only trigger when a schema is actually attached.
    payload = _backend("deepseek")._build_openai_payload(
        _req(model="deepseek-v4-pro", with_schema=False)
    )
    assert "response_format" not in payload
