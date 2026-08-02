"""Capability self-model — the digest must reflect what's actually registered
(so the assistant never confabulates a capability it lacks, nor denies one it
has), and drop groups whose subsystem is absent."""
from __future__ import annotations

from augmentum.companion_runtime.capability_digest import build_capability_digest


def test_digest_lists_registered_groups():
    d = build_capability_digest({
        "schedule_briefing", "schedule_deadline", "note.create",
        "research", "image_generation",
    })
    assert "scheduled briefings, reminders, deadline countdowns" in d
    assert "notes" in d
    assert "research and search the web" in d
    assert "generate images" in d
    # The anti-denial instruction must be present.
    assert "never claim you lack a capability listed here" in d


def test_digest_drops_absent_subsystems():
    # Only notes registered → only the notes line, nothing about scheduling.
    d = build_capability_digest({"note.create"})
    assert "notes" in d
    assert "scheduled briefings" not in d
    assert "generate images" not in d


def test_digest_empty_when_nothing_registered():
    assert build_capability_digest(set()) == ""
    assert build_capability_digest(None) == ""


def test_digest_matches_on_any_probe():
    # A group is included if ANY of its probes is present (e.g. watch_for
    # alone surfaces the scheduling line; web_search alone the research line).
    assert "price/page/feed watches" in build_capability_digest({"watch_for"})
    assert "research and search the web" in build_capability_digest({"web_search"})
    assert "check the weather" in build_capability_digest({"weather.today"})
