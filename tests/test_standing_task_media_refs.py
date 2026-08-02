"""Standing-task media refs — briefings gather a hero image + video, and
those must survive into the journal note's content_refs (kind image/video)
rather than being dropped before render. Pure-function tests for
``_media_refs_from_details``."""
from __future__ import annotations

from augmentum.companion_runtime.standing_tasks import (
    _citation_refs_from_details,
    _media_refs_from_details,
    _open_target_from_refs,
    _speak_payload,
    _task_note_origin,
)


def test_briefing_details_yield_image_and_video_refs():
    refs = _media_refs_from_details({
        "content": "the brief body",
        "hero_image_url": "https://img.example/hero.jpg",
        "video_url": "https://youtube.com/watch?v=abc",
        "video_transcript_summary": "what the clip adds",
    })
    by_kind = {r["kind"]: r for r in refs}
    assert set(by_kind) == {"image", "video"}
    assert by_kind["image"]["url"] == "https://img.example/hero.jpg"
    assert by_kind["video"]["url"] == "https://youtube.com/watch?v=abc"
    assert by_kind["video"]["summary"] == "what the clip adds"
    # Every ref carries a stable id (dedup + validator future-compat).
    assert all(r.get("id") for r in refs)


def test_local_artifact_image_path_is_kept():
    # image_search stores downloaded images as local artifact embed paths.
    refs = _media_refs_from_details({"hero_image_url": "/artifacts/x.png"})
    assert refs == [r for r in refs if r["kind"] == "image"]
    assert len(refs) == 1
    assert refs[0]["url"] == "/artifacts/x.png"


def test_non_http_or_empty_media_is_skipped():
    assert _media_refs_from_details(None) == []
    assert _media_refs_from_details({}) == []
    # A non-fetchable scheme on the hero is dropped; a relative (non-"/")
    # video url is dropped (video must be an external link).
    refs = _media_refs_from_details({
        "hero_image_url": "javascript:alert(1)",
        "video_url": "watch?v=abc",
    })
    assert refs == []


def test_video_summary_is_truncated():
    refs = _media_refs_from_details({
        "video_url": "https://youtube.com/watch?v=z",
        "video_transcript_summary": "x" * 500,
    })
    assert len(refs) == 1
    assert len(refs[0]["summary"]) == 300


def test_text_only_briefing_yields_no_media_refs():
    # Fallback briefings (synthesis unavailable) carry only content + url
    # refs; no media keys → nothing added, url refs untouched by this helper.
    assert _media_refs_from_details({"content": "sources: a, b"}) == []


def test_citations_become_titled_citation_refs():
    refs = _citation_refs_from_details({
        "citations": [
            {"title": "Reactor restart timeline", "url": "https://a.example/x"},
            {"title": "Grid impact analysis", "url": "https://b.example/y"},
        ],
    })
    assert [r["kind"] for r in refs] == ["citation", "citation"]
    assert refs[0]["title"] == "Reactor restart timeline"
    assert refs[0]["url"] == "https://a.example/x"
    assert all(r.get("id") for r in refs)


def test_citations_dedupe_by_url_and_skip_non_http():
    refs = _citation_refs_from_details({
        "citations": [
            {"title": "A", "url": "https://a.example/x"},
            {"title": "A dup", "url": "https://a.example/x"},   # same url
            {"title": "bad", "url": "ftp://nope"},               # non-http
            {"title": "no url"},                                  # missing url
            "not a dict",                                         # malformed
        ],
    })
    assert len(refs) == 1
    assert refs[0]["url"] == "https://a.example/x"


def test_no_citations_yields_empty():
    assert _citation_refs_from_details(None) == []
    assert _citation_refs_from_details({}) == []
    assert _citation_refs_from_details({"content": "x"}) == []


def test_citation_title_truncated():
    refs = _citation_refs_from_details({
        "citations": [{"title": "t" * 400, "url": "https://a.example/x"}],
    })
    assert len(refs[0]["title"]) == 200


def test_origin_carries_read_aloud_toggle():
    origin = _task_note_origin("briefing", "Morning brief", {"read_aloud": True})
    assert origin["read_aloud"] is True
    assert origin["source"] == "task"
    assert origin["detail"].startswith("briefing: Morning brief")


def test_origin_omits_read_aloud_when_off():
    assert "read_aloud" not in _task_note_origin("briefing", "X", {})
    assert "read_aloud" not in _task_note_origin("url_watch", "X", None)
    # Falsy values don't set the flag.
    assert "read_aloud" not in _task_note_origin("briefing", "X", {"read_aloud": False})


def test_origin_detail_truncates_long_title():
    origin = _task_note_origin("briefing", "T" * 200, {})
    # kind prefix + at most 80 chars of title.
    assert origin["detail"] == "briefing: " + ("T" * 80)


def test_speak_payload_when_read_aloud_on():
    p = _speak_payload({"read_aloud": True}, "the spoken briefing body", 42)
    assert p == {
        "read_aloud": True,
        "speak_text": "the spoken briefing body",
        "note_id": 42,
    }


def test_speak_payload_none_when_off_or_empty():
    assert _speak_payload({}, "body", 1) is None
    assert _speak_payload(None, "body", 1) is None
    assert _speak_payload({"read_aloud": False}, "body", 1) is None
    # read-aloud on but nothing to say → None
    assert _speak_payload({"read_aloud": True}, "   ", 1) is None


def test_speak_payload_omits_note_id_when_unknown():
    p = _speak_payload({"read_aloud": True}, "body", None)
    assert p is not None
    assert "note_id" not in p


def test_speak_payload_truncates_long_text():
    p = _speak_payload({"read_aloud": True}, "x" * 9000, 1)
    assert len(p["speak_text"]) == 6000


# ── _open_target_from_refs (notification deep-link) ──────────────────────


def test_open_target_prefers_first_url_and_flags_video_host():
    """Combined refs are assembled runner-refs → video → citations; the first
    http(s) url wins and a video host is flagged so the client plays it
    in-app rather than opening a generic tab."""
    refs = [
        {"kind": "url", "url": "https://www.youtube.com/watch?v=ID1"},
        {"kind": "video", "url": "https://youtu.be/ID2"},
        {"kind": "citation", "url": "https://example.com/post", "title": "Post"},
    ]
    assert _open_target_from_refs(refs) == {
        "url": "https://www.youtube.com/watch?v=ID1", "kind": "video",
    }


def test_open_target_non_video_link_is_kind_link_and_keeps_title():
    refs = [{"kind": "citation", "url": "https://news.example/a", "title": "Headline"}]
    assert _open_target_from_refs(refs) == {
        "url": "https://news.example/a", "kind": "link", "title": "Headline",
    }


def test_open_target_none_for_text_only_or_empty():
    # Text-only briefing (no http refs) → None, so Open keeps the drawer path.
    assert _open_target_from_refs([]) is None
    assert _open_target_from_refs(None) is None
    assert _open_target_from_refs([{"kind": "citation", "url": "not-a-url"}]) is None
    assert _open_target_from_refs([{"kind": "image", "url": "/artifacts/x.png"}]) is None
