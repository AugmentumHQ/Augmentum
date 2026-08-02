"""conversation.strike — manual context scrub for mangled STT (2026-06-13).

When misheard speech ("play sum jazz") already got a reply and is now
sitting in the model's working context, the user needs a way to say
"scratch that" and have the poisoned exchange removed — distinct from
"never mind" (which only aborts the in-flight turn). Two halves:

  - the verb (conversation.strike) is a fast-path control verb, so the
    pre-router pass fires it before the always-listening converse-skip
    can swallow it;
  - session.strike_last_exchange() pops the trailing assistant+user
    pair from in-call history so the next turn reasons from clean state.
"""

from __future__ import annotations

from augmentum.intent import REGISTRY  # noqa: F401 — triggers builtin registration
from augmentum.intent.matcher import match_intent
from augmentum.voice.pipeline import VoiceSession

# ── The verb: fast-path, matches natural phrasings, distinct from nevermind ──


def test_strike_is_fast_path():
    action = REGISTRY.get("conversation.strike")
    assert action is not None
    # Fast-path so the pre-router pass picks it up; never an LLM tool.
    assert action.fanout.fast_path is True
    assert action.fanout.tier3 is False


def test_strike_matches_natural_phrasings():
    phrasings = [
        "scratch that",
        "strike that",
        "disregard that last recording",
        "disregard what I just said",
        "ignore my last message",
        "delete that last one",
        "that wasn't meant for you",
        "pretend I didn't say that",
        "strike it from context",
    ]
    for text in phrasings:
        m = match_intent(text, fast_path_only=True)
        assert m is not None, f"no match: {text!r}"
        assert m.action_id == "conversation.strike", f"{text!r} -> {m.action_id}"


def test_strike_distinct_from_nevermind():
    # nevermind aborts the in-flight turn; strike removes a committed
    # exchange. They must not collide.
    assert match_intent("never mind", fast_path_only=True).action_id == "control.nevermind"
    assert match_intent("forget it", fast_path_only=True).action_id == "control.nevermind"
    assert match_intent("scratch that", fast_path_only=True).action_id == "conversation.strike"


def test_fast_path_only_excludes_normal_verbs():
    # The pre-router pass must NOT greedily grab conversational asks or
    # action verbs — those belong to the router / architect.
    for text in ["play some jazz", "what's the weather", "tell me about the chat"]:
        assert match_intent(text, fast_path_only=True) is None, text


def test_fast_path_only_still_finds_control_verbs():
    # The other conversation-control verbs remain reachable in the
    # restricted pass (this is what lets "stop" beat the router too).
    assert match_intent("stop", fast_path_only=True).action_id == "control.stop"
    assert match_intent("say that again", fast_path_only=True).action_id == "control.repeat"


# ── The history pop ──────────────────────────────────────────────────


def _session() -> VoiceSession:
    return VoiceSession(session_id="s1")


def test_strike_pops_trailing_exchange():
    s = _session()
    s.add_user_message("tell me about cats")
    s.add_assistant_message("Cats are great.")
    s.add_user_message("play sum jazz")          # mangled STT
    s.add_assistant_message("I'm not sure what you mean.")  # poisoned reply
    removed = s.strike_last_exchange()
    assert removed == 2
    # The clean earlier exchange survives.
    assert [m["content"] for m in s.messages] == [
        "tell me about cats", "Cats are great.",
    ]


def test_strike_bare_user_turn_removes_only_user():
    # She hasn't replied yet (still thinking) — strike just the user msg.
    s = _session()
    s.add_user_message("good morning")
    s.add_assistant_message("Morning.")
    s.add_user_message("scratchy mangled thing")
    removed = s.strike_last_exchange()
    assert removed == 1
    assert [m["content"] for m in s.messages] == ["good morning", "Morning."]


def test_strike_empty_history_is_noop():
    s = _session()
    assert s.strike_last_exchange() == 0
    assert s.messages == []


def test_strike_twice_walks_back_two_exchanges():
    s = _session()
    s.add_user_message("one")
    s.add_assistant_message("first")
    s.add_user_message("two")
    s.add_assistant_message("second")
    assert s.strike_last_exchange() == 2
    assert s.strike_last_exchange() == 2
    assert s.messages == []
