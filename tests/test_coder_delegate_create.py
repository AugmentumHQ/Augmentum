"""Regression tests for coder.delegate workspace handling.

Covers the two 2026-07-28 fixes:
  1. A model-supplied ``workspace_id`` that isn't a REAL owned workspace must
     NOT be enqueued (it queued a run against ``workspace_id="inference"`` and
     died with ``KeyError: Workspace inference not found``). It falls through to
     the resolver as a naming hint instead.
  2. ``workspace_id="__new__"`` provisions a real workspace with a chosen
     (editable) profile + derived name and enqueues — not a route to the empty
     creator.
"""
from __future__ import annotations

import asyncio

import augmentum.intent.builtin.coder as coder


class _WS:
    def __init__(self, ws_id: str, name: str = "") -> None:
        self.id = ws_id
        self.name = name or ws_id


class _Mgr:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    async def create_workspace(self, *, name, tooling_profile, user_id, kind):
        self.calls.append(
            {"name": name, "profile": tooling_profile, "user_id": user_id, "kind": kind}
        )
        if self.fail:
            raise RuntimeError("boom")
        return _WS(f"ws_real_{name[:6]}", name)


class _Jobs:
    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.enqueued: list[dict] = []

    async def create(self, *, user_id, job_type, payload, priority, max_attempts):
        self.enqueued.append(payload)
        return "job_1" if self.ok else None


class _Runner:
    def wake(self) -> None:
        pass


class _App:
    def __init__(self, *, create_fail: bool = False, jobs_ok: bool = True) -> None:
        self.container_manager = _Mgr(fail=create_fail)
        self.jobs_store = _Jobs(ok=jobs_ok)
        self.job_runner = _Runner()


class _Sess:
    def __init__(self, app: _App) -> None:
        self.user_id = "usr_x"
        self.app_state = app
        self.referents = None


def _run(coro):
    return asyncio.run(coro)


# ── name derivation ─────────────────────────────────────────────────────────

def test_derive_ws_name_strips_glue():
    assert coder._derive_ws_name("build me a dark-mode toggle for my ui") == "dark-mode-toggle-ui"


def test_derive_ws_name_research_task():
    assert coder._derive_ws_name("begin a research task about inference on raspberry pi") == "inference-raspberry-pi"


def test_derive_ws_name_never_empty():
    assert coder._derive_ws_name("please build the app") == "new-project"


# ── profile arg schema stays in sync with the catalog ───────────────────────

def test_profile_arg_schema_enumerates_catalog():
    from augmentum.coder.profiles import all_profiles

    schema = coder._profile_arg_schema()
    assert schema["enum"] == [p.id for p in all_profiles()]
    # descriptions carry per-profile labels so the model can choose deliberately
    assert "Pentest" in schema["description"]
    assert "lock-in" in schema["description"]


# ── create-and-delegate paths ───────────────────────────────────────────────

def test_create_valid_profile_enqueues_against_real_id():
    app = _App()
    r = _run(coder._create_and_delegate(
        app, _Sess(app), prompt="build an inference server", model="m",
        tier="primary", profile="power", workspace_name="",
    ))
    assert r.fulfilled is True
    assert app.container_manager.calls[0]["profile"] == "power"
    # enqueued against the REAL id the manager returned, never a guessed name
    assert app.jobs_store.enqueued[0]["workspace_id"].startswith("ws_real_")


def test_create_unknown_profile_falls_through_to_default():
    app = _App()
    r = _run(coder._create_and_delegate(
        app, _Sess(app), prompt="research inference", model="m",
        tier="primary", profile="research", workspace_name="",
    ))
    # unknown profile is blanked so create_workspace uses its configured default
    assert app.container_manager.calls[0]["profile"] == ""
    assert r.fulfilled is True


def test_create_failure_is_graceful():
    app = _App(create_fail=True)
    r = _run(coder._create_and_delegate(
        app, _Sess(app), prompt="build x", model="m",
        tier="primary", profile="power", workspace_name="",
    ))
    assert r.fulfilled is False
    assert not app.jobs_store.enqueued  # never enqueue when creation failed


def test_create_ok_but_jobs_unavailable_is_graceful():
    app = _App(jobs_ok=False)
    r = _run(coder._create_and_delegate(
        app, _Sess(app), prompt="build x", model="m",
        tier="primary", profile="power", workspace_name="",
    ))
    assert r.fulfilled is False
    # the workspace WAS created; the message must say so, not pretend nothing happened
    assert "set up" in r.speak.lower()


def test_custom_workspace_name_wins_over_derivation():
    app = _App()
    _run(coder._create_and_delegate(
        app, _Sess(app), prompt="build an inference server", model="m",
        tier="primary", profile="standard", workspace_name="my-thing",
    ))
    assert app.container_manager.calls[0]["name"] == "my-thing"
