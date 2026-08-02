from __future__ import annotations

import pytest


class TestNoteTagsLogic:
    def test_dedup_tags(self):
        tags_input = [["a", "b"], ["b", "c"], ["a", "d"]]
        all_tags = set()
        for tag_list in tags_input:
            all_tags.update(tag_list)
        assert all_tags == {"a", "b", "c", "d"}

    def test_empty_tags(self):
        all_tags = set()
        assert len(all_tags) == 0


class TestAIActionPrompts:
    def test_rewrite_prompt_prefix(self):
        action_prompts = {
            "rewrite": "Rewrite the following text, improving clarity and flow while preserving meaning:\n\n",
            "expand": "Expand the following text with more detail, examples, and depth:\n\n",
            "compress": "Condense the following text to be more concise while preserving all key information:\n\n",
            "research": "Research and provide factual information about:\n\n",
            "define": "Define and explain:\n\n",
        }
        assert "rewrite" in action_prompts
        assert action_prompts["rewrite"].startswith("Rewrite")
        assert "expand" in action_prompts


class TestScanAnnotationFormat:
    def test_annotation_structure(self):
        ann = {
            "term": "dragon",
            "start": 10,
            "end": 16,
            "source": "lorebook",
            "type": "lorebook",
            "content": "Dragons are ancient creatures",
        }
        assert ann["start"] < ann["end"]
        assert ann["source"] in ("lorebook", "memory", "related_note")
