"""Tests for browse/note AI prompt construction."""

from augmentum.proxy.browse_routes import _build_ai_messages


def test_ask_prompt_interprets_intent_instead_of_keyword_matching():
    system, user = _build_ai_messages(
        "ask",
        "The article discusses cache invalidation and stale UI state.",
        "summarize",
        "technical",
    )

    assert "interpret the request by intent" in system
    assert "exact keyword matching" in system
    assert "not in the article" in system
    assert "ONLY" not in system
    assert "User request:\nsummarize" in user
    assert "Provided content:" in user
    assert "Page content" not in user


def test_translate_question_is_target_language():
    _, user = _build_ai_messages("translate", "Hello world.", "Spanish")

    assert "Target language:\nSpanish" in user
    assert "Provided content:\nHello world." in user


def test_non_ask_question_is_preserved_as_focus():
    _, user = _build_ai_messages(
        "explain",
        "Whole note body.",
        "This selected paragraph",
    )

    assert "User focus or request:\nThis selected paragraph" in user
    assert "Provided content:\nWhole note body." in user
