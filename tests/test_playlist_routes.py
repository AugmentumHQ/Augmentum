"""Playlist item sanitisation — esp. the content-category (entityKind)
signal that the family-based playlist boundary groups on (2026-06-17)."""

from __future__ import annotations

from augmentum.proxy.playlist_routes import _sanitize_item


def test_file_item_preserves_entity_kind():
    item = _sanitize_item({
        "type": "file", "fileId": "f1", "name": "Movie X",
        "kind": "video", "entityKind": "movie",
    })
    assert item is not None
    assert item["entityKind"] == "movie"
    assert item["kind"] == "video"


def test_file_item_without_entity_kind_defaults_blank():
    item = _sanitize_item({"type": "file", "fileId": "f1", "name": "x", "kind": "audio"})
    assert item is not None
    assert item["entityKind"] == ""


def test_entity_kind_lowercased_and_bounded():
    item = _sanitize_item({
        "type": "file", "fileId": "f1", "name": "x", "kind": "audio",
        "entityKind": "  BOOK  ",
    })
    assert item is not None and item["entityKind"] == "book"


def test_file_item_rejects_bad_kind():
    assert _sanitize_item({"type": "file", "fileId": "f1", "name": "x", "kind": "comic"}) is None


def test_youtube_item_has_no_entity_kind_field():
    item = _sanitize_item({"type": "youtube", "videoId": "abc", "title": "t"})
    assert item is not None
    assert "entityKind" not in item  # youtube is flexible — no family key needed


def test_non_dict_rejected():
    assert _sanitize_item("nope") is None
    assert _sanitize_item(None) is None
