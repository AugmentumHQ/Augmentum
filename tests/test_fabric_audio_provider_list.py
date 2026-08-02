"""Cross-peer audio provider visibility — peer-hosted TTS/STT shows up
selectable in `/api/audio/providers`, not just as a silent default.

The gap: a service installed on a peer (Matt's speaches) was registered +
advertised by that peer, and usable as a fabric default fallback, but the
provider LIST endpoint read only the local table — so it never appeared in
the picker. This pins the merge that brings audio to parity with the image
`/models` peer-merge.
"""

from __future__ import annotations

from types import SimpleNamespace

from augmentum.fabric.capabilities import (
    KIND_STT_TRANSCRIBE,
    KIND_TTS_SYNTHESIZE,
    STTTranscribeCapability,
    TTSSynthesizeCapability,
)
from augmentum.proxy.audio_routes import _fabric_audio_provider_entries


class _FakeCoord:
    def __init__(self, by_kind, peers):
        self._by_kind = by_kind
        self._peers = peers

    def find_peers_with_capability(self, kind):
        return self._by_kind.get(kind, [])

    def peer_state(self, node_id):
        return self._peers.get(node_id)


def _peer(hostname="tower", icon="🖥"):
    return SimpleNamespace(paired=SimpleNamespace(hostname=hostname, icon=icon))


def test_no_coordinator_returns_empty():
    assert _fabric_audio_provider_entries(None) == []


def test_tts_peer_listed_selectable():
    cap = TTSSynthesizeCapability(provider_id="kokoro-builtin",
                                  provider_name="Kokoro", default_voice="af_heart")
    coord = _FakeCoord(
        {KIND_TTS_SYNTHESIZE: [("nodeB", cap)]},
        {"nodeB": _peer("tower")},
    )
    out = _fabric_audio_provider_entries(coord)
    assert len(out) == 1
    e = out[0]
    assert e["id"] == "fabric:nodeB:kokoro-builtin"
    assert e["provider_type"] == "tts"
    assert e["fabric"] is True
    assert "tower" in e["name"] and "Kokoro" in e["name"]
    assert e["default_voice"] == "af_heart"
    assert e["fabric_node_hostname"] == "tower"
    assert e["is_enabled"] is True and e["is_default"] is False


def test_stt_peer_listed_with_speaches_shape():
    cap = STTTranscribeCapability(provider_id="speaches-stt",
                                  provider_name="Speaches",
                                  default_model="Systran/faster-whisper-small.en")
    coord = _FakeCoord(
        {KIND_STT_TRANSCRIBE: [("peer1", cap)]},
        {"peer1": _peer("gpu-rig")},
    )
    out = _fabric_audio_provider_entries(coord)
    assert len(out) == 1
    e = out[0]
    assert e["id"] == "fabric:peer1:speaches-stt"
    assert e["provider_type"] == "stt"
    assert e["default_model"] == "Systran/faster-whisper-small.en"
    assert "gpu-rig" in e["name"]


def test_both_kinds_merged():
    tts = TTSSynthesizeCapability(provider_id="kokoro", provider_name="Kokoro")
    stt = STTTranscribeCapability(provider_id="speaches-stt", provider_name="Speaches")
    coord = _FakeCoord(
        {KIND_TTS_SYNTHESIZE: [("b", tts)], KIND_STT_TRANSCRIBE: [("b", stt)]},
        {"b": _peer("tower")},
    )
    out = _fabric_audio_provider_entries(coord)
    types = sorted(e["provider_type"] for e in out)
    assert types == ["stt", "tts"]


def test_capability_without_provider_id_skipped():
    cap = TTSSynthesizeCapability(provider_id="", provider_name="Anon")
    coord = _FakeCoord({KIND_TTS_SYNTHESIZE: [("b", cap)]}, {"b": _peer()})
    assert _fabric_audio_provider_entries(coord) == []


def test_lookup_failure_is_graceful():
    class _Boom:
        def find_peers_with_capability(self, kind):
            raise RuntimeError("coordinator down")

        def peer_state(self, node_id):
            return None

    assert _fabric_audio_provider_entries(_Boom()) == []


def test_missing_peer_state_falls_back_to_node_id():
    cap = TTSSynthesizeCapability(provider_id="kokoro", provider_name="Kokoro")
    coord = _FakeCoord({KIND_TTS_SYNTHESIZE: [("abcdef1234567890", cap)]}, {})
    out = _fabric_audio_provider_entries(coord)
    assert len(out) == 1
    # No paired peer → hostname falls back to a truncated node id.
    assert out[0]["fabric_node_hostname"] == "abcdef123456"
