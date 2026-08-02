"""KV reuse audit — the request-side contract verdict joined with the
response-side timings (cache_n / prompt_n) and the kv_tier decision.

Covers the classification matrix of ``LlamaCppBackend._audit_kv_reuse``:
a byte-stable payload that the server cold-prefills anyway must surface
as ``server_void`` (previously invisible — only showed up as latency),
while an Augmentum-side prefix mutation must surface as
``payload_divergence`` with the culprit carried over from
``track_prefix_stability``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.proxy.status_bus import bind_kv_tier


@pytest.fixture()
def backend() -> LlamaCppBackend:
    # Manager presence gates the audit (remote llama.cpp backends skip it);
    # a bare MagicMock is enough — the audit never calls into it.
    return LlamaCppBackend(
        http_client=MagicMock(), base_url="http://x", server_manager=MagicMock(),
    )


def _req(messages: list[tuple[str, str]], key: str = "u1:sess") -> InternalChatRequest:
    return InternalChatRequest(
        model="m",
        messages=[Message(role=r, content=c) for r, c in messages],
        kv_session_key=key,
        kv_mode="chat",
    )


BASE = [
    ("system", "You are helpful." * 40),
    ("user", "First question, reasonably long to dominate the char count." * 20),
    ("assistant", "First answer." * 40),
]


def _classify(backend, *, evaluated_n, cache_n):
    out = backend._audit_kv_reuse(
        _req(BASE), evaluated_n=evaluated_n, cache_n=cache_n, endpoint="test",
    )
    assert out is not None
    return out


def test_first_turn_is_cold_expected(backend):
    backend.track_prefix_stability(_req(BASE))
    assert backend._kv_contract["u1:sess"]["contract"] == "first_turn"
    out = _classify(backend, evaluated_n=1000, cache_n=0)
    assert out["kv_reuse"] == "cold_expected"


def test_stable_payload_reused_is_hot(backend):
    backend.track_prefix_stability(_req(BASE))
    turn2 = BASE + [("user", "Second question")]
    backend.track_prefix_stability(_req(turn2))
    v = backend._kv_contract["u1:sess"]
    assert v["contract"] == "ok"
    assert v["expected_pct"] > 0.9
    out = _classify(backend, evaluated_n=30, cache_n=970)
    assert out["kv_reuse"] == "hot"


def test_stable_payload_cold_prefill_is_server_void(backend):
    backend.track_prefix_stability(_req(BASE))
    backend.track_prefix_stability(_req(BASE + [("user", "Second question")]))
    token = bind_kv_tier("cold_with_checkpoint")
    try:
        out = _classify(backend, evaluated_n=1000, cache_n=0)
    finally:
        from augmentum.proxy.status_bus import kv_tier_var
        kv_tier_var.reset(token)
    assert out["kv_reuse"] == "server_void"
    assert out["kv_void_cause"] == "restore_ineffective"


def test_server_void_cause_tracks_tier(backend):
    backend.track_prefix_stability(_req(BASE))
    backend.track_prefix_stability(_req(BASE + [("user", "q2")]))
    token = bind_kv_tier("hot")
    try:
        out = _classify(backend, evaluated_n=1000, cache_n=0)
    finally:
        from augmentum.proxy.status_bus import kv_tier_var
        kv_tier_var.reset(token)
    assert out["kv_void_cause"] == "slot_kv_mismatch"


def test_mid_prefix_mutation_is_payload_divergence(backend):
    backend.track_prefix_stability(_req(BASE))
    mutated = [
        (BASE[0][0], BASE[0][1] + " NOW: 12:34"),  # system rewritten mid-prefix
        *BASE[1:],
        ("user", "Second question"),
    ]
    backend.track_prefix_stability(_req(mutated))
    assert backend._kv_contract["u1:sess"]["contract"] == "violated"
    out = _classify(backend, evaluated_n=1000, cache_n=0)
    assert out["kv_reuse"] == "payload_divergence"


def test_partial_reuse_between_floor_and_expectation(backend):
    backend.track_prefix_stability(_req(BASE))
    backend.track_prefix_stability(_req(BASE + [("user", "Second question")]))
    out = _classify(backend, evaluated_n=600, cache_n=400)
    assert out["kv_reuse"] == "partial_reuse"


def test_no_session_key_or_no_manager_returns_none(backend):
    assert backend._audit_kv_reuse(
        _req(BASE, key=""), evaluated_n=100, cache_n=0, endpoint="test",
    ) is None
    backend._manager = None
    assert backend._audit_kv_reuse(
        _req(BASE), evaluated_n=100, cache_n=0, endpoint="test",
    ) is None
