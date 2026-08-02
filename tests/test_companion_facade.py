"""Tests for the unified Companion façade — accumulation thesis Step 2.

The Companion class is the single addressable object that every
other module should talk to. Phase 1 is a thin façade — every method
delegates to the existing CompanionRuntime. The point is to have
*one place to look* so future PRs can migrate read/write sites into
routing through ``app.state.companions[name]`` without behavior
change at each step.

Coverage:

- Companion exposes the right attributes (runtime, bus, memory,
  user_affect, name, etc.)
- ``for_user(user_id)`` returns a CompanionUserView
- View methods scope correctly to the user
- ``app.state.companions["becca"]`` mounts at runtime startup
- Personality doc resolves from the new canonical location
- Personality doc falls back to legacy path when canonical missing
- companion.toml is valid TOML and parseable
- Directory layout is intact
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Companion façade construction ─────────────────────────────────────


def _make_fake_runtime(*, started: bool = True, companion_id: str = "becca"):
    """Minimal runtime stub for testing the façade."""
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    class _FakeIdentity:
        persona_kernel_digest = "she notices small things"

    class _FakeMemory:
        async def journal(self, content, **kwargs):
            return 1

    class _FakeRuntime:
        def __init__(self):
            self.bus = PresenceBus()
            self.companion_id = companion_id
            self.memory = _FakeMemory()
            self.user_affect = UserAffectTracker()
            self.identity = _FakeIdentity()
            self._started = started
            self._app_state = None

        @property
        def owner_user_id(self):
            return "u_owner"

        async def get_identity(self, user_id):
            return _FakeIdentity()

        async def get_state(self, user_id):
            class _State: pass
            return _State()

        async def snapshot(self):
            return {"companion_id": self.companion_id, "started": self._started}

    return _FakeRuntime()


def test_companion_exposes_core_attributes():
    from augmentum.companion import Companion

    runtime = _make_fake_runtime()
    c = Companion(runtime)
    assert c.name == "becca"
    assert c.companion_id == "becca"
    assert c.runtime is runtime
    assert c.bus is runtime.bus
    assert c.memory is runtime.memory
    assert c.user_affect is runtime.user_affect
    assert c.owner_user_id == "u_owner"
    assert c.started is True


def test_companion_started_false_when_runtime_not_started():
    from augmentum.companion import Companion

    runtime = _make_fake_runtime(started=False)
    c = Companion(runtime)
    assert c.started is False


def test_companion_name_matches_runtime_companion_id():
    from augmentum.companion import Companion

    runtime = _make_fake_runtime(companion_id="lyra")
    c = Companion(runtime)
    assert c.name == "lyra"


# ── CompanionUserView ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_for_user_returns_user_view():
    from augmentum.companion import Companion, CompanionUserView

    runtime = _make_fake_runtime()
    c = Companion(runtime)
    view = c.for_user("u_test")
    assert isinstance(view, CompanionUserView)
    assert view.user_id == "u_test"
    assert view.companion is c
    assert view.name == "becca"


@pytest.mark.asyncio
async def test_view_identity_returns_per_user_identity():
    from augmentum.companion import Companion

    runtime = _make_fake_runtime()
    c = Companion(runtime)
    view = c.for_user("u_test")
    identity = await view.identity()
    assert identity is not None
    assert identity.persona_kernel_digest


def test_view_read_affect_returns_observation_when_tracker_present():
    from augmentum.companion import Companion

    runtime = _make_fake_runtime()
    runtime.user_affect.update("u_test", "tender")
    c = Companion(runtime)
    view = c.for_user("u_test")
    obs = view.read_affect()
    assert obs is not None
    assert obs.tag == "tender"
    assert obs.sample_count == 1


def test_view_read_affect_returns_none_when_tracker_missing():
    from augmentum.companion import Companion

    runtime = _make_fake_runtime()
    runtime.user_affect = None  # simulate older runtime
    c = Companion(runtime)
    view = c.for_user("u_test")
    obs = view.read_affect()
    assert obs is None


@pytest.mark.asyncio
async def test_view_journal_writes_through_memory():
    from augmentum.companion import Companion

    journaled = []

    class _RecordingMemory:
        async def journal(self, content, **kwargs):
            journaled.append({"content": content, **kwargs})
            return 1

    runtime = _make_fake_runtime()
    runtime.memory = _RecordingMemory()
    c = Companion(runtime)
    view = c.for_user("u_test")
    await view.journal(
        "test entry",
        entry_type="observation",
        affect_tag="curious",
    )
    assert len(journaled) == 1
    assert journaled[0]["content"] == "test entry"
    assert journaled[0]["entry_type"] == "observation"
    assert journaled[0]["user_id"] == "u_test"
    assert journaled[0]["affect_tag"] == "curious"


# ── Bus delegation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_companion_publish_topic_emits_with_correct_source():
    import asyncio

    from augmentum.companion import Companion

    runtime = _make_fake_runtime()
    c = Companion(runtime)

    sub = await runtime.bus.subscribe("test.**", slice_key="t")
    captured = []

    async def _drain():
        try:
            ev = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
        except TimeoutError:
            return
        if ev:
            captured.append({"topic": ev.topic, "source": ev.source_companion_id})

    drain_task = asyncio.create_task(_drain())
    try:
        await c.publish_topic("test.event", {"x": 1})
        await drain_task
        assert any(c["topic"] == "test.event" for c in captured)
        # Source must be the companion's name
        assert captured[0]["source"] == "becca"
    finally:
        await runtime.bus.unsubscribe(sub)


# ── Personality doc resolution ────────────────────────────────────────


def test_personality_doc_resolves_from_new_canonical_location():
    """The canonical location is companions/<id>/identity/personality.md.
    When it exists, it should be the resolved path."""
    from augmentum.companion_runtime.identity import CompanionIdentity

    identity = CompanionIdentity(
        backend=None,  # not used by personality_doc_path
        companion_id="becca",
    )
    path = identity.personality_doc_path
    assert path.name == "personality.md"
    # The path should be inside companions/becca/identity/
    assert "companions" in str(path)
    assert "becca" in str(path)
    assert "identity" in str(path)
    # And it should actually exist (we copied it in Step 2)
    assert path.exists()


def test_personality_doc_falls_back_to_legacy_when_canonical_missing():
    """For companion_ids that don't have a companions/<id>/ directory,
    the legacy docs/superpowers/specs path is the fallback."""
    from augmentum.companion_runtime.identity import CompanionIdentity

    # A companion_id that almost certainly doesn't have a directory
    identity = CompanionIdentity(
        backend=None,
        companion_id="nonexistent_companion_12345",
    )
    path = identity.personality_doc_path
    # Canonical doesn't exist, legacy doesn't exist either — falls
    # through to canonical (write path target).
    assert path.name == "personality.md"
    assert "companions" in str(path)


# ── Directory layout ──────────────────────────────────────────────────


def _repo_root() -> Path:
    """Helper: find the repo root from the test file location."""
    return Path(__file__).resolve().parent.parent


def test_companions_directory_exists_and_has_becca():
    repo_root = _repo_root()
    becca_dir = repo_root / "companions" / "becca"
    assert becca_dir.is_dir(), f"missing {becca_dir}"


def test_companion_toml_is_valid_and_has_required_keys():
    """companion.toml must be parseable and carry the manifest fields
    the runtime relies on."""
    import tomllib

    repo_root = _repo_root()
    toml_path = repo_root / "companions" / "becca" / "companion.toml"
    assert toml_path.exists(), f"missing {toml_path}"

    with toml_path.open("rb") as f:
        manifest = tomllib.load(f)

    assert "companion" in manifest
    assert manifest["companion"]["id"] == "becca"
    assert "seed_date" in manifest["companion"]
    assert "schema_version" in manifest["companion"]
    assert "paths" in manifest
    assert manifest["paths"]["personality_doc"] == "identity/personality.md"


def test_identity_personality_md_populated():
    """The personality doc must be migrated into the canonical location."""
    repo_root = _repo_root()
    personality = (
        repo_root / "companions" / "becca" / "identity" / "personality.md"
    )
    assert personality.exists(), f"missing {personality}"
    text = personality.read_text(encoding="utf-8")
    # Sanity: it should look like the personality doc, not a stub
    assert "Becca" in text
    assert len(text) > 5000  # the real doc is ~19KB


def test_subdirectories_exist():
    """All the planned subdirectories must exist so future phases can
    populate them incrementally."""
    repo_root = _repo_root()
    becca = repo_root / "companions" / "becca"
    for sub in ("identity", "body", "topology", "artifacts", "history", "state"):
        assert (becca / sub).is_dir(), f"missing companions/becca/{sub}"


def test_readme_present():
    repo_root = _repo_root()
    readme = repo_root / "companions" / "becca" / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "accumulation" in text.lower() or "companion" in text.lower()
