"""Tests for proxy.status_bus — Stage payload shape + request_id ContextVar."""

from __future__ import annotations

import asyncio
import time

import pytest

from augmentum.proxy.status_bus import (
    Stage,
    bind_request_id,
    make_request_id,
    request_id_var,
    reset_request_id,
)


class TestRequestId:
    """``request_id_var`` is the foundation for log correlation. Future
    code along the request path will read it without changing function
    signatures, so the binding lifecycle has to be airtight.
    """

    def test_default_is_empty_string(self):
        """Default value is empty string, not None — so .get() never
        returns None and downstream code can stamp it in log payloads
        without an explicit ``or ""`` guard at every site.
        """
        assert request_id_var.get() == ""

    def test_make_request_id_is_short_and_unique(self):
        """8 hex chars — long enough to disambiguate inside a process,
        short enough to not visually clutter logs.
        """
        ids = {make_request_id() for _ in range(1000)}
        assert len(ids) == 1000, "uuid prefix collision in 1000 trials"
        for rid in ids:
            assert len(rid) == 8
            assert all(c in "0123456789abcdef" for c in rid)

    def test_bind_and_reset_round_trip(self):
        token = bind_request_id("abc12345")
        try:
            assert request_id_var.get() == "abc12345"
        finally:
            reset_request_id(token)
        assert request_id_var.get() == ""

    @pytest.mark.asyncio
    async def test_binding_isolated_between_concurrent_tasks(self):
        """ContextVar isolation: two concurrent ``asyncio.create_task``
        calls each get their own request_id, and a ``set`` in one task
        doesn't leak into the other.

        Without this, request ids would smear across tasks and log
        correlation would silently lie. Worth proving once.
        """
        observed: dict[str, str] = {}

        async def stamp(name: str, rid: str) -> None:
            tok = bind_request_id(rid)
            try:
                # yield control so both tasks interleave
                await asyncio.sleep(0)
                observed[name] = request_id_var.get()
            finally:
                reset_request_id(tok)

        await asyncio.gather(
            stamp("a", "11111111"),
            stamp("b", "22222222"),
        )
        assert observed == {"a": "11111111", "b": "22222222"}


class TestStagePayloads:
    """The wire shape of stage_start / stage_complete is contract — the
    frontend dispatches on these keys. If any of them rename, the
    indicator silently stops rendering.
    """

    def test_start_payload_shape(self):
        s = Stage("model_swap", label="Loading model", detail="deepseek-v3")
        payload = s.start_payload()

        assert set(payload.keys()) == {"stage_start"}
        body = payload["stage_start"]
        assert body["stage"] == "model_swap"
        assert body["label"] == "Loading model"
        assert body["detail"] == "deepseek-v3"
        assert body["id"].startswith("stg_")
        assert isinstance(body["started_at"], float)
        # request_id default is empty string when no binding active.
        assert body["request_id"] == ""

    def test_label_falls_back_to_name(self):
        """Untyped emit sites can pass just the stage name and get
        a reasonable label out the other side. Frontend's
        ``_STATUS_LABELS`` table handles the friendly mapping.
        """
        s = Stage("prefill")
        assert s.start_payload()["stage_start"]["label"] == "prefill"

    def test_complete_payload_shape_and_duration(self):
        s = Stage("slot_restore", label="Restoring", detail="32k tokens")
        # Force a measurable duration without relying on real clock drift.
        s._started_mono = time.monotonic() - 0.5

        payload = s.complete_payload(success=True, detail="restored from RAM")

        assert set(payload.keys()) == {"stage_complete"}
        body = payload["stage_complete"]
        assert body["stage"] == "slot_restore"
        assert body["success"] is True
        assert body["error"] == ""
        assert body["detail"] == "restored from RAM"
        # Duration computed from monotonic clock; tolerate 100ms slack
        # for slow CI but assert it's at least the synthetic 500ms.
        assert 450 <= body["duration_ms"] <= 600

    def test_complete_with_failure_carries_error_text(self):
        s = Stage("model_load", label="Loading", detail="x.gguf")
        payload = s.complete_payload(
            success=False, error_text="OOM after 3 retries",
        )
        body = payload["stage_complete"]
        assert body["success"] is False
        assert body["error"] == "OOM after 3 retries"

    def test_complete_detail_defaults_to_start_detail(self):
        """If the caller doesn't pass a fresh detail string, the start
        descriptor carries through — frontend can keep the same row.
        """
        s = Stage("slot_restore", label="Restoring", detail="32k tokens")
        payload = s.complete_payload(success=True)
        assert payload["stage_complete"]["detail"] == "32k tokens"

    def test_start_complete_share_id(self):
        """Frontend correlates start↔complete by id. Same instance =
        same id; different instances = different ids.
        """
        s = Stage("prefill")
        sid = s.start_payload()["stage_start"]["id"]
        cid = s.complete_payload()["stage_complete"]["id"]
        assert sid == cid

        other = Stage("prefill")
        assert other.start_payload()["stage_start"]["id"] != sid

    def test_request_id_propagates_into_payloads(self):
        """When a request_id is bound, both start and complete carry it
        — that's how the frontend can later say "show me everything
        about request abc12345".
        """
        token = bind_request_id("req_abcd12")
        try:
            s = Stage("prefill", label="Preparing context")
            assert s.start_payload()["stage_start"]["request_id"] == "req_abcd12"
            assert s.complete_payload()["stage_complete"]["request_id"] == "req_abcd12"
        finally:
            reset_request_id(token)

    def test_progress_payload_shape(self):
        s = Stage("model_load")
        payload = s.progress_payload(percent=42, message="loading layer 12/32")

        assert set(payload.keys()) == {"stage_progress"}
        body = payload["stage_progress"]
        assert body["stage"] == "model_load"
        assert body["id"] == s.id
        assert body["percent"] == 42
        assert body["message"] == "loading layer 12/32"

    def test_progress_omits_unset_optional_fields(self):
        """Empty progress payload doesn't carry meaningless 0% / "" —
        the frontend can't tell those apart from "no info".
        """
        s = Stage("prefill")
        body = s.progress_payload()["stage_progress"]
        assert "percent" not in body
        assert "message" not in body
