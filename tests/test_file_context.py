"""Tests for file context builder."""

from __future__ import annotations


class TestTierDetection:
    def test_small_model(self):
        from augmentum.vfs.context import detect_tier
        assert detect_tier(4096) == "card"

    def test_medium_model(self):
        from augmentum.vfs.context import detect_tier
        assert detect_tier(16384) == "card_summary"

    def test_large_model(self):
        from augmentum.vfs.context import detect_tier
        assert detect_tier(131072) == "card_content"

    def test_vision_flag(self):
        from augmentum.vfs.context import detect_tier
        # Vision doesn't change tier (it changes what content is sent)
        assert detect_tier(4096, has_vision=True) == "card"


class TestBuildContext:
    def test_empty_entries(self):
        from augmentum.vfs.context import build_file_context
        assert build_file_context([], "card") == ""

    def test_single_card(self):
        from augmentum.vfs.context import build_file_context
        from augmentum.vfs.models import FileEntry
        entries = [FileEntry(
            id="fi_1", user_id="usr_1", source="images", source_id="img_1",
            name="sunset.png", mime_type="image/png", size_bytes=3_200_000,
            description="Sunset over ocean", created_at="2026-03-15",
        )]
        result = build_file_context(entries, "card")
        assert "1 relevant file" in result
        assert "sunset.png" in result
        assert "Sunset over ocean" in result

    def test_budget_forces_card_tier(self):
        from augmentum.vfs.context import build_file_context
        from augmentum.vfs.models import FileEntry
        entries = [FileEntry(
            id="fi_1", user_id="usr_1", source="documents", source_id="doc_1",
            name="report.pdf", mime_type="application/pdf", size_bytes=1024,
            created_at="2026-01-01",
        )]
        # Very tight budget should force card-only even if tier is card_content
        result = build_file_context(entries, "card_content", remaining_budget=500)
        assert "report.pdf" in result

    def test_max_cards_limit(self):
        from augmentum.vfs.context import MAX_FILE_CARDS, build_file_context
        from augmentum.vfs.models import FileEntry
        entries = [
            FileEntry(id=f"fi_{i}", user_id="usr_1", source="images",
                      source_id=f"img_{i}", name=f"img_{i}.png",
                      created_at="2026-01-01")
            for i in range(20)
        ]
        result = build_file_context(entries, "card")
        # Should not exceed MAX_FILE_CARDS
        assert result.count("[File:") <= MAX_FILE_CARDS
