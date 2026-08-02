"""Egress-hygiene fixes for the OpenAI-compatible backend.

Pins three independent fixes for failure modes observed in production
when routing tool-using sessions through chatgpt-bridge / Codex (any
strict OpenAI-spec backend will reject the same payloads):

1. Tool-name sanitisation — outbound tool defs and assistant tool_calls
   are forced to match ``^[a-zA-Z0-9_-]+$``. Local models sometimes
   emit dotted or colon-prefixed names that 400 the request mid-loop.
2. Orphan tool-message scrub — ``role=tool`` messages whose
   ``tool_call_id`` has no preceding parent are dropped. Compaction and
   rewind both leave these behind.
3. (Coder-side) Token-quota 429s are classified as ``quota`` and bail
   without burning the 17s retry budget; covered in
   ``test_coder_quota_classification.py``.
"""
from __future__ import annotations

from augmentum.models.openai_compat import (
    OpenAIBackend,
    _sanitize_tool_name,
    _scrub_orphan_tool_messages,
)


# ---------------------------------------------------------------------------
# Tool-name sanitisation
# ---------------------------------------------------------------------------


def test_valid_names_pass_through_unchanged():
    for name in ("shell_exec", "file-read", "task_dispatch", "verify", "X1_y2"):
        assert _sanitize_tool_name(name) == name


def test_dotted_name_becomes_underscored():
    # The exact failure shape from the 2026-05-31 logs: a local model
    # emitted ``shell.exec`` and chatgpt-bridge 400'd with
    # "Invalid 'input[8].name'".
    assert _sanitize_tool_name("shell.exec") == "shell_exec"


def test_colon_prefix_namespace_becomes_underscored():
    # task_dispatch can produce ``@deepseek:explore``-style names when
    # peer routing leaks into the model's tool registry. The original
    # `@` and `:` are both rejected by the OpenAI regex.
    assert _sanitize_tool_name("@deepseek:explore") == "_deepseek_explore"


def test_unicode_becomes_underscored():
    assert _sanitize_tool_name("file—read") == "file_read"


def test_empty_falls_back_to_placeholder():
    # An empty name shouldn't crash payload construction. The placeholder
    # is invalid as a real tool so the model gets a quick `function not
    # found` instead of a silent payload-construction failure.
    assert _sanitize_tool_name("") == "invalid_tool"
    # Each invalid char maps independently → "***" → "___". Match must
    # still pass the regex so the request flies.
    assert _sanitize_tool_name("***") == "___"
    import re
    assert re.match(r"^[a-zA-Z0-9_-]+$", _sanitize_tool_name("***"))


# ---------------------------------------------------------------------------
# Orphan tool-message scrub
# ---------------------------------------------------------------------------


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content, *, tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


def _tool(call_id, content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_well_formed_message_list_unchanged():
    msgs = [
        _user("read /a"),
        _assistant("", tool_calls=[{"id": "call_1", "function": {"name": "file_read"}}]),
        _tool("call_1"),
        _assistant("done"),
    ]
    out = _scrub_orphan_tool_messages(msgs)
    assert out == msgs


def test_orphan_tool_message_at_start_is_dropped():
    # Exact compaction-aftermath shape: parent assistant turn got
    # truncated, child tool result survives. Provider would 400 with
    # "Messages with role 'tool' must follow tool_calls".
    msgs = [
        _user("hi"),
        _tool("call_dangling"),
        _assistant("hi back"),
    ]
    out = _scrub_orphan_tool_messages(msgs)
    assert out == [_user("hi"), _assistant("hi back")]


def test_orphan_after_rewind_is_dropped():
    # Rewind cut the assistant turn at index 1; the tool result at
    # index 2 has no parent now. Without the scrub, the next call
    # returns the Codex "No tool call found for function call output
    # with call_id ..." 400.
    msgs = [
        _user("first"),
        _tool("call_dropped"),
        _user("second"),
    ]
    out = _scrub_orphan_tool_messages(msgs)
    assert out == [_user("first"), _user("second")]


def test_matched_pair_after_orphan_still_kept():
    # Orphan at index 1, valid pair at indices 2-3. The valid pair must
    # survive the scrub regardless of what was dropped earlier.
    msgs = [
        _user("first"),
        _tool("orphan"),
        _assistant("", tool_calls=[{"id": "call_real", "function": {"name": "shell_exec"}}]),
        _tool("call_real"),
    ]
    out = _scrub_orphan_tool_messages(msgs)
    assert out == [
        _user("first"),
        _assistant("", tool_calls=[{"id": "call_real", "function": {"name": "shell_exec"}}]),
        _tool("call_real"),
    ]


def test_tool_message_without_call_id_is_dropped():
    # Defensive: even a tool message with no tool_call_id at all gets
    # dropped — there's no way the provider can match it back.
    msgs = [
        _user("hi"),
        {"role": "tool", "content": "lost"},
    ]
    out = _scrub_orphan_tool_messages(msgs)
    assert out == [_user("hi")]


# ---------------------------------------------------------------------------
# Payload diagnostics — surfaces byte shape on backend error logs so
# context-window failures stop being opaque 502s. See ``_payload_diagnostics``
# in augmentum/models/openai_compat.py.
# ---------------------------------------------------------------------------


def test_payload_diagnostics_counts_text_messages():
    payload = {
        "messages": [
            {"role": "system", "content": "you are helpful"},      # 15 bytes
            {"role": "user", "content": "hi"},
        ],
    }
    diag = OpenAIBackend._payload_diagnostics(payload)
    assert diag["message_count"] == 2
    assert diag["instruction_bytes"] == 15
    assert diag["image_count"] == 0
    assert diag["total_image_bytes"] == 0
    assert diag["payload_bytes"] > 0


def test_payload_diagnostics_counts_developer_role_as_instruction():
    # OpenAI reasoning models rewrite system→developer. Both should
    # contribute to the instruction byte total — the operator cares
    # about the prefix size regardless of which role name carries it.
    payload = {
        "messages": [
            {"role": "developer", "content": "system-style guidance"},  # 21 bytes
            {"role": "user", "content": "go"},
        ],
    }
    diag = OpenAIBackend._payload_diagnostics(payload)
    assert diag["instruction_bytes"] == 21


def test_payload_diagnostics_counts_images():
    # Two image parts across two messages — both base64 URL lengths add
    # to total_image_bytes. The synthetic URLs are 32 + 64 = 96 chars.
    a = "data:image/png;base64," + "A" * 10  # 32 chars
    b = "data:image/jpeg;base64," + "B" * 41  # 64 chars
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": a}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b}},
                ],
            },
        ],
    }
    diag = OpenAIBackend._payload_diagnostics(payload)
    assert diag["image_count"] == 2
    assert diag["total_image_bytes"] == 32 + 64
    assert diag["message_count"] == 2


def test_payload_diagnostics_handles_malformed_parts_gracefully():
    # A non-dict content list item shouldn't crash diagnostics.
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url"}, "stray-string", None]},
        ],
    }
    diag = OpenAIBackend._payload_diagnostics(payload)
    assert diag["image_count"] == 1
    assert diag["total_image_bytes"] == 0
    assert diag["payload_bytes"] > 0


def test_payload_diagnostics_returns_minus_one_on_unserializable():
    # An object that json.dumps can't handle shouldn't crash the error
    # log path — the helper must always return a dict.
    class _BadObj:
        pass
    payload = {"messages": [{"role": "user", "content": _BadObj()}]}
    diag = OpenAIBackend._payload_diagnostics(payload)
    assert diag["payload_bytes"] == -1
    assert diag["message_count"] == 1


# ---------------------------------------------------------------------------
# Transport diagnostics — ConnectTimeout stringifies to "", which used to
# surface to Claude Code as a blank ``Backend error:``.
# ---------------------------------------------------------------------------


def test_sanitize_url_for_log_drops_query_credentials():
    url = "https://user:secret@example.com:8443/v1/chat/completions?api_key=SECRET"
    assert OpenAIBackend._sanitize_url_for_log(url) == "https://example.com:8443/v1/chat/completions"


def test_transport_error_message_includes_exception_type_when_str_empty():
    class _BlankTimeout(Exception):
        def __str__(self):
            return ""

    msg = OpenAIBackend._transport_error_message(
        _BlankTimeout(), model="gpt-5.5", url="http://chatgpt-bridge:8080/v1/chat/completions"
    )
    assert "Provider transport error (_BlankTimeout)" in msg
    assert "gpt-5.5" in msg
    assert "http://chatgpt-bridge:8080/v1/chat/completions" in msg
    assert msg.strip().endswith(":") is False


# ---------------------------------------------------------------------------
# Gemini thought_signature (extra_content) round-trip
#
# Google's OpenAI-compat endpoint returns a per-call
# ``extra_content.google.thought_signature`` on Gemini 3.x tool_calls and
# 400s any later turn that echoes the call WITHOUT it. The coder's stream
# accumulator rebuilds tool_calls from id/name/args only, dropping
# extra_content — so the backend caches it at capture time and re-attaches it
# at egress by id. Verified live 2026-07-04 against gemini-3.1-flash-lite
# (drop → 400, keep → 200); these pin the cache mechanics.
# ---------------------------------------------------------------------------


def _fresh_backend() -> OpenAIBackend:
    # No HTTP is issued by _build_openai_payload, so a None client is fine.
    return OpenAIBackend(None, "https://generativelanguage.googleapis.com/v1beta/openai")


def test_capture_stores_extra_content_by_id():
    be = _fresh_backend()
    be._capture_tool_call_extra([
        {"id": "abc", "extra_content": {"google": {"thought_signature": "SIG"}}},
    ])
    assert be._tool_call_extra["abc"] == {"google": {"thought_signature": "SIG"}}


def test_capture_ignores_tool_calls_without_extra_or_id():
    be = _fresh_backend()
    be._capture_tool_call_extra([
        {"id": "no_extra", "function": {"name": "x"}},          # no extra_content
        {"extra_content": {"google": {}}},                        # no id
        "not-a-dict",
    ])
    assert be._tool_call_extra == {}


def test_capture_fifo_evicts_oldest_past_cap():
    be = _fresh_backend()
    be._TOOL_CALL_EXTRA_CAP = 2
    for i in range(4):
        be._capture_tool_call_extra([{"id": f"id{i}", "extra_content": {"n": i}}])
    assert list(be._tool_call_extra.keys()) == ["id2", "id3"]


def test_reattach_injects_cached_signature_by_id():
    be = _fresh_backend()
    be._tool_call_extra["abc"] = {"google": {"thought_signature": "SIG"}}
    # The coder rebuilds the call in {id,type,function} shape WITHOUT extra_content.
    out = be._reattach_tool_call_extra(
        {"id": "abc", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    )
    assert out["extra_content"] == {"google": {"thought_signature": "SIG"}}


def test_reattach_does_not_overwrite_existing_extra_content():
    be = _fresh_backend()
    be._tool_call_extra["abc"] = {"google": {"thought_signature": "CACHED"}}
    out = be._reattach_tool_call_extra({"id": "abc", "extra_content": {"kept": 1}})
    assert out["extra_content"] == {"kept": 1}


def test_reattach_noop_for_uncached_id():
    be = _fresh_backend()
    out = be._reattach_tool_call_extra({"id": "unknown", "function": {}})
    assert "extra_content" not in out


def test_egress_payload_reattaches_signature_end_to_end():
    # The full outgoing path: capture a signature from a "response", then build
    # the NEXT request's payload with the coder-formatted assistant tool_call
    # (no extra_content) and assert the serialized payload carries it back.
    from augmentum.models.base import InternalChatRequest, Message

    be = _fresh_backend()
    be._capture_tool_call_extra([
        {"id": "call1", "extra_content": {"google": {"thought_signature": "SIG"}}},
    ])
    request = InternalChatRequest(
        model="gemini-3.1-flash-lite",
        messages=[
            Message(role="user", content="weather?"),
            Message(role="assistant", content="", tool_calls=[
                {"id": "call1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
            ]),
            Message(role="tool", tool_call_id="call1", content="18C"),
        ],
    )
    payload = be._build_openai_payload(request)
    assistant = next(m for m in payload["messages"] if m.get("tool_calls"))
    tc = assistant["tool_calls"][0]
    assert tc["extra_content"] == {"google": {"thought_signature": "SIG"}}
    # Sanity: the function name still round-trips through egress sanitisation.
    assert tc["function"]["name"] == "get_weather"
