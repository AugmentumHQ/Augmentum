"""CompanionJournal tests — load / merge / save / cap enforcement."""

from __future__ import annotations

import json
from pathlib import Path

from augmentum.game_agent.journal import (
    CompanionJournal,
    JournalSections,
)
from augmentum.game_agent.prompt import build_slow_path_inputs
from augmentum.game_agent.schema import PlanPayload, SurfaceCapsPayload


def _caps() -> SurfaceCapsPayload:
    return SurfaceCapsPayload(
        semantic_inputs=["a", "b"],
        log_schema="pokemon_rs.v1",
        observation_modalities=["log", "frame"],
    )


# ── Path safety ───────────────────────────────────────────────────────


def test_journal_path_is_sanitized(tmp_path: Path) -> None:
    """@example: dangerous ids never escape the journal root.

    ROOT CAUSE:
      user_id and title_id are externally controlled inputs. Without
      sanitization, a malicious id like "../../etc" could write
      outside the journal root. The path-safe regex caps both inputs
      to [A-Za-z0-9._-] which makes traversal impossible.
    """

    j = CompanionJournal(
        root_dir=tmp_path,
        user_id="../../etc/passwd",
        title_id="bad/title\\name",
    )
    # The real safety property: the resolved path stays under the root.
    # ``..`` and ``.`` as substrings of filename segments are OS-safe
    # (parent-dir resolution only triggers at path-separator
    # boundaries); the sanitizer strips slashes which is what matters.
    resolved = j.path.resolve()
    assert resolved.is_relative_to(tmp_path.resolve())
    rel = j.path.relative_to(tmp_path)
    assert len(rel.parts) == 2  # exactly <user_dir>/<title>.json


def test_journal_load_or_create_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """@example: a fresh user/title has no file; we get an empty journal."""

    j = CompanionJournal.load_or_create(
        root_dir=tmp_path, user_id="u1", title_id="t1",
    )
    assert j.sections.status == ""
    assert j.sections.progress == ""
    assert j.sections.objectives == ""
    assert j.sections.notes == []
    assert j.to_prompt_dict() is None


def test_journal_load_or_create_round_trips(tmp_path: Path) -> None:
    """@example: write a journal, reload it, get the same content back."""

    j1 = CompanionJournal.load_or_create(
        root_dir=tmp_path, user_id="u1", title_id="t1",
    )
    j1.sections.status = "On Route 102, just left Petalburg."
    j1.sections.progress = "Beat Brendan's first rival fight."
    j1.sections.objectives = "1) Reach Petalburg Gym; 2) Catch a Zigzagoon."
    j1.sections.notes = ["Treecko is at level 9.", "Wally borrowed our Pokemon."]
    assert j1.save() is True

    j2 = CompanionJournal.load_or_create(
        root_dir=tmp_path, user_id="u1", title_id="t1",
    )
    assert j2.sections.status == j1.sections.status
    assert j2.sections.progress == j1.sections.progress
    assert j2.sections.objectives == j1.sections.objectives
    assert j2.sections.notes == j1.sections.notes


def test_journal_load_tolerates_corrupted_file(tmp_path: Path) -> None:
    """@example: corrupt JSON degrades to an empty journal, not a crash."""

    bogus_dir = tmp_path / "u1"
    bogus_dir.mkdir()
    (bogus_dir / "t1.json").write_text("{this is not json", encoding="utf-8")
    j = CompanionJournal.load_or_create(
        root_dir=tmp_path, user_id="u1", title_id="t1",
    )
    assert j.sections.status == ""


# ── Merge semantics ───────────────────────────────────────────────────


def test_apply_update_replaces_string_sections() -> None:
    """@example: a section in the patch replaces the existing value verbatim."""

    s = JournalSections(status="old", progress="kept")
    changed = s.merge({"status": "new"})
    assert changed is True
    assert s.status == "new"
    assert s.progress == "kept"


def test_apply_update_no_op_when_unchanged() -> None:
    """@example: merging the same value returns False; orchestrator skips save."""

    s = JournalSections(status="same")
    assert s.merge({"status": "same"}) is False


def test_apply_update_appends_notes() -> None:
    """@example: notes_append adds entries without dropping existing ones."""

    s = JournalSections(notes=["first"])
    s.merge({"notes_append": ["second", "third"]})
    assert s.notes == ["first", "second", "third"]


def test_apply_update_dedups_appended_notes() -> None:
    """@example: re-appending a note already present is a no-op for that note."""

    s = JournalSections(notes=["already"])
    s.merge({"notes_append": ["already", "new"]})
    assert s.notes == ["already", "new"]


def test_apply_update_caps_notes_list() -> None:
    """@example: notes_append cap kicks in after 20 entries; latest survives.

    ROOT CAUSE:
      A runaway model would otherwise pad the journal indefinitely,
      blowing the prompt budget across many turns. Capping at 20
      keeps the journal size bounded; FIFO eviction so the latest
      observations survive.
    """

    s = JournalSections(notes=[f"old-{i}" for i in range(20)])
    s.merge({"notes_append": ["fresh"]})
    assert "fresh" in s.notes
    assert "old-0" not in s.notes
    assert len(s.notes) == 20


def test_apply_update_clips_section_strings() -> None:
    """@example: oversized strings are clipped to 2000 chars + ellipsis."""

    s = JournalSections()
    huge = "x" * 5000
    s.merge({"status": huge})
    assert len(s.status) <= 2001  # 2000 chars + "…"
    assert s.status.endswith("…")


def test_apply_update_tolerates_garbage_patch_shape() -> None:
    """@example: non-dict patches are silently rejected, not crashing."""

    s = JournalSections(status="ok")
    assert s.merge("not a dict") is False  # type: ignore[arg-type]
    assert s.status == "ok"


# ── Atomic save ───────────────────────────────────────────────────────


def test_save_uses_atomic_rename(tmp_path: Path) -> None:
    """@example: no .tmp file lingers after a successful save.

    ROOT CAUSE:
      A crash mid-write must never leave a half-written journal
      on disk. Writing to a NamedTemporaryFile + os.replace is the
      atomic-rename pattern; we verify no .tmp file remains after a
      clean save (proving the rename happened).
    """

    j = CompanionJournal.load_or_create(
        root_dir=tmp_path, user_id="u1", title_id="t1",
    )
    j.sections.status = "ok"
    j.save()
    parent = j.path.parent
    leftovers = list(parent.glob(".journal.*.tmp"))
    assert leftovers == []
    assert j.path.exists()


def test_save_persists_only_known_fields(tmp_path: Path) -> None:
    """@example: the on-disk schema is fixed, not influenced by attacker keys."""

    j = CompanionJournal.load_or_create(
        root_dir=tmp_path, user_id="u1", title_id="t1",
    )
    j.sections.status = "test"
    j.save()
    raw = json.loads(j.path.read_text(encoding="utf-8"))
    assert set(raw["sections"].keys()) == {"status", "progress", "objectives", "notes"}
    assert raw["schema"] == "companion_journal.v1"


# ── Prompt integration ───────────────────────────────────────────────


def test_inputs_block_omits_journal_when_none() -> None:
    """@example: no journal -> no JOURNAL line in the prompt."""

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
        journal=None,
    )
    assert "JOURNAL" not in block


def test_inputs_block_renders_journal_above_overlay() -> None:
    """@example: JOURNAL sits above OVERLAY (stable section first).

    ROOT CAUSE:
      llama-server's KV-cache prefix-matches at the token level.
      Sections that change less frequently (JOURNAL: a few times per
      session) belong earlier in the prompt than OVERLAY (every probe
      tick) so the cache stays warm longer.
    """

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
        journal={"status": "in Petalburg", "progress": "got Treecko"},
        overlay={"player_x": 10, "player_y": 5},
    )
    assert "JOURNAL:" in block
    assert "OVERLAY:" in block
    j_idx = block.find("JOURNAL:")
    o_idx = block.find("OVERLAY:")
    log_idx = block.find("LIVE_LOG_TAIL:")
    assert 0 <= j_idx < o_idx < log_idx


# ── PlanPayload accepts the new field ────────────────────────────────


def test_plan_payload_accepts_journal_update() -> None:
    """@example: a plan can emit a journal_update dict that round-trips."""

    plan = PlanPayload.model_validate({
        "observations": [], "state_update": "", "actions": [],
        "confidence": 0.5, "next_check_in_ms": 500,
        "journal_update": {"status": "in Petalburg", "notes_append": ["got Treecko"]},
    })
    assert plan.journal_update is not None
    assert plan.journal_update["status"] == "in Petalburg"


def test_plan_payload_journal_update_defaults_to_none() -> None:
    """@example: omitting journal_update leaves it None (no journal change)."""

    plan = PlanPayload(confidence=0.5, next_check_in_ms=500)
    assert plan.journal_update is None
