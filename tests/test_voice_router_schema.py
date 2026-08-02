"""Schema-constrained voice-router classifier output (2026-06-15).

The voice router attaches a JSON Schema via ``raw_options["json_schema"]``
so the classifier backend constrains generation to the exact verdict
shape — the prerequisite that makes a tiny non-reasoning model
(SmolLM-135M) viable on this latency-critical hop. Both transport paths
honor the same key:

* OpenAI-compat (the SmolLM sidecar / any llama-server over HTTP) →
  ``response_format`` json_schema.
* Local engine (llama_cpp) → forwarded verbatim through its raw_options
  passthrough allowlist.
"""

from __future__ import annotations

import httpx

from augmentum.architect.voice_router import _VOICE_ROUTER_SCHEMA
from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.llama_cpp import _LLAMACPP_PASSTHROUGH_PARAMS
from augmentum.models.openai_compat import OpenAIBackend


def _backend() -> OpenAIBackend:
    return OpenAIBackend(httpx.AsyncClient(), "http://localhost:8090/v1", None)


def _req(**kw) -> InternalChatRequest:
    return InternalChatRequest(
        model="smollm2-135m-instruct",
        messages=[Message(role="user", content="play some jazz")],
        **kw,
    )


def test_schema_shape_is_the_verdict_contract():
    props = _VOICE_ROUTER_SCHEMA["properties"]
    assert set(_VOICE_ROUTER_SCHEMA["required"]) == {
        "coherent", "addressed", "confidence", "goal",
    }
    # The schema is EXACTLY the four decision fields — no free-text
    # ``reasoning`` field. It was telemetry-only (emitted last, never
    # informed the verdict) but a variable-length string balloons
    # generation on a CPU-served tiny model; with additionalProperties
    # False the json_schema grammar forces the model to stop after
    # ``goal`` (~25 tokens), keeping the hop inside the 2.5s budget.
    # Don't re-add it — see voice_router._VOICE_ROUTER_SCHEMA.
    assert set(props) == {"coherent", "addressed", "confidence", "goal"}
    assert "reasoning" not in props
    assert props["goal"]["enum"] == ["act", "clarify", "converse", "drop", "idle"]
    assert _VOICE_ROUTER_SCHEMA["additionalProperties"] is False


def test_openai_payload_translates_json_schema():
    payload = _backend()._build_openai_payload(_req(raw_options={
        "json_schema": _VOICE_ROUTER_SCHEMA,
        "json_schema_name": "voice_router_verdict",
    }))
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "voice_router_verdict"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] is _VOICE_ROUTER_SCHEMA


def test_json_schema_overrides_plain_json_object():
    payload = _backend()._build_openai_payload(_req(
        format="json",
        raw_options={"json_schema": _VOICE_ROUTER_SCHEMA},
    ))
    assert payload["response_format"]["type"] == "json_schema"


def test_default_name_when_unspecified():
    payload = _backend()._build_openai_payload(_req(
        raw_options={"json_schema": _VOICE_ROUTER_SCHEMA},
    ))
    assert payload["response_format"]["json_schema"]["name"] == "structured_output"


def test_no_schema_leaves_json_object_untouched():
    payload = _backend()._build_openai_payload(_req(format="json"))
    assert payload["response_format"] == {"type": "json_object"}


def test_local_engine_forwards_the_same_key():
    # Same raw_options key constrains the in-process engine path.
    assert "json_schema" in _LLAMACPP_PASSTHROUGH_PARAMS
