"""SFW keyword backstop for character-import search results.

The chub.ai / RisuRealm import search trusts the upstream ``nsfw=false``
flag but does not verify it. These tests pin the post-fetch keyword
filter that drops mistagged-SFW (or explicitly-titled) cards in English
and Korean before they reach the client.
"""

from __future__ import annotations

from augmentum.discovery.safety import (
    is_korean_nsfw_text,
    is_nsfw_text,
    is_unsafe_card_text,
)
from augmentum.proxy.ui_routes import _card_text_unsafe, _filter_chub_sfw_nodes


class TestEnglishWholeWord:
    def test_explicit_token_trips(self):
        assert is_unsafe_card_text("cute hentai waifu") is True

    def test_neutral_text_passes(self):
        assert is_unsafe_card_text("a wholesome adventure story") is False

    def test_substring_does_not_false_positive(self):
        # "anal" inside "analysis" must NOT match (whole-word semantics).
        assert is_unsafe_card_text("analysis of the situation") is False

    def test_neutral_phrase_with_soft_word_passes(self):
        # "sex" is intentionally not a token; "sex education" stays clean.
        assert is_unsafe_card_text("sex education class") is False


class TestKoreanSubstring:
    def test_adult_stem_and_derivative(self):
        # Listing the stem 성인 catches derivatives like 성인용 / 성인물.
        assert is_korean_nsfw_text("성인") is True
        assert is_korean_nsfw_text("성인용 캐릭터") is True
        assert is_korean_nsfw_text("성인물 모음") is True

    def test_age_restriction_marker(self):
        assert is_korean_nsfw_text("19금 콘텐츠") is True

    def test_neutral_korean_passes(self):
        assert is_korean_nsfw_text("따뜻한 일상 이야기") is False
        assert is_korean_nsfw_text("평범한 학교 생활") is False

    def test_combined_helper_routes_korean(self):
        assert is_unsafe_card_text("야설 작가") is True

    def test_english_helper_is_korean_blind(self):
        # is_nsfw_text is English whole-word only; Korean must not trip it.
        assert is_nsfw_text("야동 모음") is False


class TestEmptyInput:
    def test_none_and_empty_are_safe(self):
        assert is_unsafe_card_text(None) is False
        assert is_unsafe_card_text("") is False
        assert is_korean_nsfw_text(None) is False


class TestRisuCardFilter:
    def test_unsafe_via_name(self):
        assert _card_text_unsafe({"name": "성인 전용", "description": "", "tags": []}) is True

    def test_unsafe_via_tags(self):
        card = {"name": "Knight", "description": "a brave hero", "tags": ["hentai", "rpg"]}
        assert _card_text_unsafe(card) is True

    def test_unsafe_via_description_korean(self):
        card = {"name": "선생님", "description": "야한 이야기입니다", "tags": ["일상"]}
        assert _card_text_unsafe(card) is True

    def test_clean_card_passes(self):
        card = {"name": "Knight", "description": "a brave hero", "tags": ["rpg", "fantasy"]}
        assert _card_text_unsafe(card) is False

    def test_missing_fields_tolerated(self):
        assert _card_text_unsafe({}) is False

    def test_non_list_tags_tolerated(self):
        assert _card_text_unsafe({"name": "ok", "tags": "fantasy"}) is False


class TestChubNodeFilter:
    def test_drops_unsafe_node_nested_data(self):
        payload = {
            "data": {
                "nodes": [
                    {"name": "Hero", "topics": ["fantasy"]},
                    {"name": "Lewd Girl", "topics": ["nsfw"]},
                ],
                "count": 2,
            }
        }
        out = _filter_chub_sfw_nodes(payload)
        names = [n["name"] for n in out["data"]["nodes"]]
        assert names == ["Hero"]

    def test_drops_via_korean_topic(self):
        payload = {
            "data": {
                "nodes": [
                    {"name": "선생님", "topics": ["성인"]},
                    {"name": "학생", "topics": ["일상"]},
                ]
            }
        }
        out = _filter_chub_sfw_nodes(payload)
        names = [n["name"] for n in out["data"]["nodes"]]
        assert names == ["학생"]

    def test_flat_nodes_shape(self):
        payload = {"nodes": [{"name": "porn star"}, {"name": "Wizard"}]}
        out = _filter_chub_sfw_nodes(payload)
        assert [n["name"] for n in out["nodes"]] == ["Wizard"]

    def test_all_clean_left_untouched(self):
        payload = {"data": {"nodes": [{"name": "Wizard"}, {"name": "Knight"}]}}
        out = _filter_chub_sfw_nodes(payload)
        assert len(out["data"]["nodes"]) == 2

    def test_non_search_payload_untouched(self):
        # A non-dict / non-node payload (e.g. a single card import) passes through.
        assert _filter_chub_sfw_nodes({"foo": "bar"}) == {"foo": "bar"}
        assert _filter_chub_sfw_nodes([1, 2, 3]) == [1, 2, 3]
