"""Voice responsiveness + input integrity (2026-06-13).

Two opposite failures of the same "should she engage?" machinery, both
from live logs:

  - She IGNORED a real turn: "It was about the narrative mode chat" —
    coherent, one-on-one, no other speaker — got dropped because the
    LLM address-router TIMED OUT (35B, 2.5s) and the regex fallback
    misfiled the declarative continuation as self-talk. Fix: the
    fallback is precision-tiered — it releases the LOW-precision
    self_talk guess on an engaged-LLM failure, keeps the HIGH-precision
    third_person drop.

  - She SPOKE to no one: Moonshine hallucinated "Thank you." from a mic
    bump and it became a turn. Fix: a phrase-aware STT hallucination
    gate, ambient-only for the ambiguous fillers.
"""

from __future__ import annotations

from augmentum.architect.voice_router import _regex_fallback
from augmentum.proxy.voice_routes import _is_stt_hallucination

# ── Address-router fallback: precision-tiered, miss-averse ────────────


def test_the_exact_dropped_sentence_is_answered_on_timeout():
    # The live failure (2026-06-13): "it"-led continuation, coherent,
    # one-on-one. After de-greedying third_person ("it" removed) this is
    # no_signal, which the fallback leans addressed on an engaged-LLM
    # timeout. The regression this whole pass exists to kill.
    d = _regex_fallback(
        "It was about the narrative mode chat",
        model="m", latency_ms=2508, parsed_from="timeout_fallback",
    )
    assert d.addressed is True
    assert d.goal != "drop"


def test_it_led_sharing_no_longer_third_person():
    # "it was/is …" is a topic continuation, not narration about a
    # person — must not be classified as third_person ambient.
    from augmentum.architect.address import is_addressed
    assert is_addressed("it was a long day").signal != "third_person"
    assert is_addressed("it's raining outside").signal != "third_person"


def test_self_talk_released_on_parse_fallback():
    # Genuine self_talk ("I was just thinking…") released on an
    # engaged-LLM parse failure.
    d = _regex_fallback(
        "I was just thinking about the weather earlier",
        model="m", latency_ms=10, parsed_from="parse_fallback",
    )
    assert d.addressed is True


def test_real_third_person_still_drops_on_timeout():
    # Narration about an actual person (he/she/they/named) stays a drop —
    # this is the genuinely high-precision negative.
    d = _regex_fallback(
        "He told her to leave the room",
        model="m", latency_ms=2508, parsed_from="timeout_fallback",
    )
    assert d.addressed is False
    assert d.goal == "drop"


def test_self_talk_still_drops_on_hard_backend_error():
    # On a genuine backend failure (no engaged-LLM evidence) the
    # low-precision release does NOT apply — conservative drop holds.
    d = _regex_fallback(
        "I think it's probably fine either way",
        model="m", latency_ms=5, parsed_from="error_fallback",
    )
    assert d.addressed is False


def test_imperative_still_acts_in_fallback():
    d = _regex_fallback(
        "play some jazz",
        model="m", latency_ms=5, parsed_from="timeout_fallback",
    )
    assert d.addressed is True
    assert d.goal == "act"


# ── STT hallucination gate ────────────────────────────────────────────


def test_caption_artifact_always_dropped():
    assert _is_stt_hallucination("Thanks for watching!", explicit=True)
    assert _is_stt_hallucination("Please subscribe", explicit=False)
    assert _is_stt_hallucination("Subtitles by the Amara.org community", explicit=True)


def test_lone_filler_dropped_only_when_ambient():
    # Ambient capture → phantom from a bump → drop.
    assert _is_stt_hallucination("Thank you.", explicit=False) is True
    assert _is_stt_hallucination("okay", explicit=False) is True
    # Explicit ptt/wake → the user deliberately spoke → keep it.
    assert _is_stt_hallucination("Thank you.", explicit=False) is True
    assert _is_stt_hallucination("Thank you.", explicit=True) is False
    assert _is_stt_hallucination("okay", explicit=True) is False


def test_phantom_embedded_in_real_speech_is_kept():
    # Whole-transcript match only — a phantom phrase inside real speech
    # must NOT be dropped.
    assert _is_stt_hallucination(
        "thank you, can you play some music", explicit=False,
    ) is False


def test_real_sentence_never_dropped():
    assert _is_stt_hallucination(
        "what's the weather like today", explicit=False,
    ) is False
    assert _is_stt_hallucination(
        "it was about the narrative mode chat", explicit=False,
    ) is False
