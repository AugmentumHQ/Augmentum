"""Memory extraction grounding — facts about the USER must come from the
user's own self-disclosure, never from a topic they asked about or content
the assistant looked up.

Regression for 2026-06-17: asking about a looked-up fictional character
caused the article's contents to be stored as personal memories ("I am
<character>…"). The hole was twofold: the loose first-person regex matched
"tell ME", and a first-person-phrased fact bypassed the question/request
grounding gate.
"""

from __future__ import annotations

from augmentum.memory.extractor import _validate_fact
from augmentum.memory.models import ExtractedFact, MemoryType


def _fact(content: str, *, importance: float = 0.8) -> ExtractedFact:
    return ExtractedFact(
        content=content,
        type=MemoryType.FACT,
        importance=importance,
        confidence=0.9,
        is_explicit=False,
        source_context={},
    )


class TestRejectsLookedUpTopicAsUserFact:
    def test_tell_me_about_character_is_not_a_user_fact(self):
        # The model read a looked-up article and phrased it first-person.
        f = _fact("I am Lyra, a sky-pirate who found the lost city")
        assert _validate_fact(f, ["Tell me about the characters Lyra and Finch."]) is False

    def test_who_is_question_is_not_a_user_fact(self):
        f = _fact("I invented the first algorithm, born in 1815 as Ada Lovelace")
        assert _validate_fact(f, ["Who is Ada Lovelace and what did she do?"]) is False

    def test_look_up_request_is_not_a_user_fact(self):
        f = _fact("My homeworld is the planet Krypton")
        assert _validate_fact(f, ["Look up Superman's backstory for me"]) is False


class TestKeepsGenuineSelfDisclosure:
    def test_first_person_disclosure_in_a_question_is_kept(self):
        f = _fact("Lives in Seattle")
        assert _validate_fact(f, ["I live in Seattle, what are good restaurants?"]) is True

    def test_preference_disclosure_is_kept(self):
        f = _fact("Loves spicy food")
        assert _validate_fact(f, ["I love spicy food, any recommendations?"]) is True

    def test_skill_disclosure_is_kept(self):
        f = _fact("Works with Go and PostgreSQL")
        assert _validate_fact(f, ["I'm a backend dev, mostly Go and PostgreSQL"]) is True
