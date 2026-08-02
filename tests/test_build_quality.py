"""Unit tests for the inline build quality gate (augmentum/builds/quality.py)."""

from __future__ import annotations

from augmentum.builds.quality import (
    behavior_quality_summary,
    behavior_verdict,
    floor_for_kind,
    judge_tool_names,
    quality_summary,
)


def test_weak_build_completes_but_is_unverified():
    """A build that wrote files + finished + published but never served/drove/
    asserted must NOT be reported clean — it's the unusable-app case."""
    v = judge_tool_names(
        ["file_write", "file_write", "finish_task"],
        status="completed", artifact_ok=True, kind="calculator",
    )
    assert v["passed"] is False
    assert set(v["failed_checks"]) == {
        "ran_server", "opened_browser", "drove_ui", "asserted_behavior",
    }
    q = quality_summary(v, final_status="completed")
    assert q["qualityStatus"] == "unverified"
    # Every unproven check yields a human-readable warning.
    assert len(q["warnings"]) == 4
    assert any("dev server" in w for w in q["warnings"])
    assert any("asserted" in w for w in q["warnings"])


def test_full_build_is_clean_and_passes_depth():
    v = judge_tool_names(
        [
            "builder_design_system", "file_write", "service_start", "browser_open",
            "browser_click", "browser_click", "browser_click",
            "browser_evaluate", "browser_evaluate", "browser_evaluate",
            "finish_task",
        ],
        status="completed", artifact_ok=True, kind="calculator",
    )
    assert v["passed"] is True
    assert v["depth"] == {"enough_drives": True, "enough_asserts": True}
    assert v["soft"]["pulled_resources"] is True
    assert quality_summary(v, final_status="completed")["qualityStatus"] == "clean"


def test_verify_only_resume_is_clean_via_has_files():
    """A resume that only verifies an already-built app writes no new files but
    drives + asserts thoroughly. has_files (the published artifact has source)
    must satisfy wrote_code so this isn't false-flagged unverified — the exact
    case the live DeepSeek resume surfaced (6 drives, 32 asserts, 3 files)."""
    trail = (
        ["dir_tree", "file_read", "file_read", "service_start", "browser_open"]
        + ["browser_type"] * 6
        + ["browser_evaluate"] * 32
        + ["finish_task"]
    )
    v = judge_tool_names(trail, status="completed", artifact_ok=True,
                         kind="calculator", has_files=True)
    assert v["hard"]["wrote_code"] is True
    assert v["passed"] is True
    assert quality_summary(v, final_status="completed")["qualityStatus"] == "clean"
    # Without has_files it would have been dinged on wrote_code.
    v2 = judge_tool_names(trail, status="completed", artifact_ok=True,
                          kind="calculator", has_files=False)
    assert v2["hard"]["wrote_code"] is False
    assert v2["passed"] is False


def test_failed_build_is_not_routed_through_quality_lane():
    """A build that didn't finish cleanly surfaces as a failure on its own —
    quality_summary must not also tag it 'unverified' (double signal)."""
    v = judge_tool_names(["file_write"], status="failed", artifact_ok=False, kind="game")
    q = quality_summary(v, final_status="failed")
    assert q["qualityStatus"] == "clean"
    assert q["warnings"] == []


def test_no_artifact_fails_published_check():
    v = judge_tool_names(
        ["file_write", "service_start", "browser_open", "browser_click",
         "browser_evaluate", "finish_task"],
        status="completed", artifact_ok=False, kind="form",
    )
    assert "published_artifact" in v["failed_checks"]
    assert v["passed"] is False


def test_floor_for_kind_defaults():
    assert floor_for_kind("calculator") == (3, 3)
    assert floor_for_kind("game") == (1, 2)
    assert floor_for_kind("totally-unknown") == (1, 1)


def test_empty_verdict_summary_is_clean():
    assert quality_summary({}, final_status="completed")["qualityStatus"] == "clean"


# --- outcome-based verdict (behaviors actually passing in a browser) --------

def test_behavior_verdict_all_pass_is_clean():
    behaviors = [
        {"id": "a", "description": "shows tip", "status": "pass", "evidence": "ok"},
        {"id": "b", "description": "splits total", "status": "pass", "evidence": "ok"},
    ]
    v = behavior_verdict(behaviors, status="completed", artifact_ok=True)
    assert v["mode"] == "outcome"
    assert v["passed"] is True
    assert v["score"] == 1.0
    assert v["behaviors_passed"] == 2 and v["behaviors_failed"] == 0
    assert behavior_quality_summary(v)["qualityStatus"] == "clean"


def test_behavior_verdict_a_failure_blocks_and_names_it():
    behaviors = [
        {"id": "a", "description": "shows tip", "status": "pass", "evidence": "ok"},
        {"id": "b", "description": "divide by zero shows an error",
         "status": "fail", "evidence": "assertion was false"},
    ]
    v = behavior_verdict(behaviors, status="completed", artifact_ok=True)
    assert v["passed"] is False
    assert v["score"] == 0.5
    assert v["failed"][0]["id"] == "b"
    q = behavior_quality_summary(v)
    assert q["qualityStatus"] == "unverified"
    assert any("divide by zero" in w for w in q["warnings"])
    assert any("assertion was false" in w for w in q["warnings"])


def test_behavior_verdict_nothing_checked_cannot_claim_verified():
    behaviors = [{"id": "a", "description": "x", "status": "untested", "evidence": ""}]
    v = behavior_verdict(behaviors, status="completed", artifact_ok=True)
    assert v["passed"] is False  # can't claim verified with zero checks
    assert behavior_quality_summary(v)["qualityStatus"] == "unverified"


def test_behavior_verdict_unfinished_build_not_clean():
    behaviors = [{"id": "a", "description": "x", "status": "pass", "evidence": "ok"}]
    v = behavior_verdict(behaviors, status="failed", artifact_ok=True)
    assert v["passed"] is False
