"""Tests for the workspace tooling-profile catalog and resolver.

Covers the data model in ``augmentum/coder/profiles.py`` and ensures the
v1 → v2 refactor preserves the exact install-line emission shape for the
existing standard/power/browser profiles. New profile additions should
add corresponding cases here.

Spec: docs/superpowers/specs/2026-06-02-tooling-profile-system-v2.md
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder import profiles
from augmentum.coder.containers import (
    ContainerManager,
    _profile_image_is_prebaked,
)
from augmentum.coder.profiles import (
    ResolvedProfile,
    ToolingProfile,
    all_profiles,
    chain,
    emit_install_lines,
    has_profile,
    metadata,
    resolve,
)

# ----------------------------------------------------------------------
# Catalog shape
# ----------------------------------------------------------------------


def test_catalog_contains_v1_profiles():
    ids = [p.id for p in all_profiles()]
    assert ids[:3] == ["standard", "power", "browser"], (
        "v1 profiles must appear first in declaration order so existing UI "
        "selectors and defaults keep working"
    )


def test_has_profile_known_and_unknown():
    assert has_profile("standard")
    assert has_profile("power")
    assert has_profile("browser")
    assert not has_profile("kali")
    assert not has_profile("")


def test_every_profile_has_required_fields():
    for prof in all_profiles():
        assert prof.id
        assert prof.label
        assert prof.description
        assert prof.est_size_mb > 0, (
            f"{prof.id} missing est_size_mb — UI size hint will read 0"
        )
        assert prof.network_policy in {"default", "raw_sockets"}


# ----------------------------------------------------------------------
# Inheritance chain
# ----------------------------------------------------------------------


def test_chain_root_only_for_standard():
    walked = chain("standard")
    assert [p.id for p in walked] == ["standard"]


def test_chain_walks_parent_first_for_browser():
    walked = chain("browser")
    assert [p.id for p in walked] == ["standard", "power", "browser"]


def test_chain_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown tooling profile"):
        chain("kali")


def test_chain_rejects_cycle(monkeypatch):
    cyclic = ToolingProfile(
        id="loopy", label="Loopy", description="cycle test",
        inherits="loopy",
    )
    monkeypatch.setitem(profiles._PROFILES, "loopy", cyclic)
    with pytest.raises(ValueError, match="cycle"):
        chain("loopy")


# ----------------------------------------------------------------------
# Resolve — flattening + leaf-owned fields
# ----------------------------------------------------------------------


def test_resolve_standard_is_empty_install():
    r = resolve("standard")
    assert isinstance(r, ResolvedProfile)
    assert r.apt_packages == ()
    assert r.pip_packages == ()
    assert r.npm_packages == ()
    assert r.post_install == ()
    assert r.extra_caps == ()


def test_resolve_power_carries_power_packages():
    r = resolve("power")
    assert "procps" in r.apt_packages
    assert "rsync" in r.apt_packages
    assert "uv" in r.pip_packages
    assert "ipython" in r.pip_packages
    assert "pnpm" in r.npm_packages
    assert "yarn" in r.npm_packages


def test_resolve_browser_is_power_plus_nothing():
    """Since 2026-07-17 browser automation lives on the shared sidecar
    service — the browser profile installs NO Playwright/Chromium."""
    r = resolve("browser")
    power = resolve("power")
    assert r.apt_packages == power.apt_packages
    assert r.pip_packages == power.pip_packages
    assert "playwright" not in r.pip_packages
    assert not any("playwright" in line for line in r.post_install)


def test_resolve_leaf_owns_identity():
    r = resolve("browser")
    assert r.id == "browser"
    assert r.label == "Browser/Test"
    # Image tag is leaf-declared (will diverge in PR 2 when tags split).
    assert r.image_tag


# ----------------------------------------------------------------------
# emit_install_lines — v1 parity
# ----------------------------------------------------------------------


def test_emit_standard_is_empty():
    assert emit_install_lines("standard") == []


def test_emit_power_matches_v1_shape():
    lines = emit_install_lines("power")
    # Expected line count: header, apt, pip, corepack, npm = 5
    assert len(lines) == 5
    assert lines[0] == "export DEBIAN_FRONTEND=noninteractive"
    assert lines[1].startswith("apt-get update -qq 2>/dev/null; apt-get install")
    assert "procps" in lines[1]
    assert "rsync" in lines[1]
    assert lines[2].startswith("python3 -m pip install --no-cache-dir --ignore-installed")
    assert "uv" in lines[2]
    assert "ipython" in lines[2]
    assert lines[3] == "corepack enable 2>/dev/null || true"
    assert lines[4].startswith("npm install -g")
    assert "pnpm" in lines[4]
    assert "yarn" in lines[4]


def test_emit_browser_matches_power_no_playwright():
    # Browser profile emits exactly the power install set — the old
    # per-workspace Playwright/Chromium install moved to the sidecar.
    assert emit_install_lines("browser") == emit_install_lines("power")
    assert not any("playwright" in line for line in emit_install_lines("browser"))


def test_emit_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown tooling profile"):
        emit_install_lines("kali")


# ----------------------------------------------------------------------
# metadata() — compatibility with v1 tooling-profile.json shape
# ----------------------------------------------------------------------


def test_metadata_standard_shape():
    md = metadata("standard")
    assert md == {
        "tooling_profile": "standard",
        "includes": ["standard Augmentum workspace baseline"],
    }


def test_metadata_power_shape():
    md = metadata("power")
    assert md["tooling_profile"] == "power"
    assert "process/network inspection" in md["includes"]


def test_metadata_browser_shape():
    md = metadata("browser")
    assert md["tooling_profile"] == "browser"
    assert "power profile" in md["includes"]
    assert any("sidecar" in s for s in md["includes"])
    assert not any("Playwright" in s for s in md["includes"])


def test_metadata_unknown_profile_raises():
    with pytest.raises(ValueError):
        metadata("kali")


# ----------------------------------------------------------------------
# Pentest profile (PR 3)
# ----------------------------------------------------------------------


def test_pentest_in_catalog():
    assert has_profile("pentest")
    # Stable ordering — pentest follows the v1 trio so existing index-
    # based callers don't shift.
    ids = [p.id for p in all_profiles()]
    assert ids.index("pentest") > ids.index("browser")


def test_pentest_inherits_power():
    walked = chain("pentest")
    assert [p.id for p in walked] == ["standard", "power", "pentest"]


def test_pentest_resolve_accumulates_power_and_pentest_packages():
    power = resolve("power")
    pent = resolve("pentest")
    # Pentest packages must come AFTER power's so apt sees the inheritance
    # order (matches FROM power AS pentest in the Dockerfile).
    assert pent.apt_packages[: len(power.apt_packages)] == power.apt_packages
    assert pent.pip_packages[: len(power.pip_packages)] == power.pip_packages
    # Signature pentest tools are present in the flattened list.
    for tool in ("nmap", "sqlmap", "hashcat", "zaproxy"):
        assert tool in pent.apt_packages
    for lib in ("impacket", "scapy"):
        assert lib in pent.pip_packages


def test_pentest_carries_net_raw_extra_cap():
    pent = resolve("pentest")
    assert "NET_RAW" in pent.extra_caps
    # Earlier profiles must NOT have inadvertently inherited a cap.
    assert resolve("power").extra_caps == ()
    assert resolve("browser").extra_caps == ()


def test_pentest_metadata_includes_signature_buckets():
    md = metadata("pentest")
    assert md["tooling_profile"] == "pentest"
    includes = md["includes"]
    assert "power profile" in includes
    assert any("nmap" in s for s in includes)
    assert any("SecLists" in s for s in includes)
    assert any("NET_RAW" in s for s in includes)


def test_pentest_emit_walks_power_then_pentest_install_layers():
    lines = emit_install_lines("pentest")
    joined = "\n".join(lines)
    # Power layer's pip set must precede pentest's apt set in emit order.
    power_pip_idx = next(
        i for i, line in enumerate(lines)
        if "pre-commit" in line and "pip install" in line
    )
    pentest_apt_idx = next(
        i for i, line in enumerate(lines)
        if "nmap" in line and "apt-get install" in line
    )
    assert power_pip_idx < pentest_apt_idx
    # SecLists clone and gobuster vendor come from post_install — must
    # appear after the pentest apt/pip lines.
    assert "SecLists" in joined
    assert "gobuster" in joined


def test_pentest_has_authorized_use_notice():
    pent = resolve("pentest")
    # The inline notice surfaces in the UI dropdown — non-empty is the
    # contract, exact wording is allowed to evolve.
    assert pent.notice
    assert "authorized" in pent.notice.lower()


def test_pentest_image_tag_matches_dockerfile_target():
    pent = resolve("pentest")
    assert pent.image_tag == "augmentum-workspace:pentest"


def test_pentest_workspace_guide_addendum_present():
    """The pentest workspace-guide addendum must (a) exist, (b) include the
    scope rules the agent is supposed to read at turn start, and (c) leave
    the base guide unmodified for profiles without an addendum."""
    from augmentum.coder.prompts import WORKSPACE_GUIDE, workspace_guide

    pent_guide = workspace_guide("pentest")
    assert pent_guide.startswith(WORKSPACE_GUIDE)
    addendum = pent_guide[len(WORKSPACE_GUIDE):]
    assert "Pentest profile" in addendum
    assert "Scope rules" in addendum
    # Profiles without an addendum return the base guide unchanged so
    # the existing workspace_guide consumers aren't impacted.
    assert workspace_guide("power") == WORKSPACE_GUIDE
    assert workspace_guide("standard") == WORKSPACE_GUIDE
    assert workspace_guide(None) == WORKSPACE_GUIDE


# ----------------------------------------------------------------------
# Image-tag resolution + prebake detection (PR 2)
# ----------------------------------------------------------------------


def test_each_profile_declares_image_tag():
    for prof in all_profiles():
        assert prof.image_tag.startswith("augmentum-workspace:"), (
            f"{prof.id} image_tag {prof.image_tag!r} should be a "
            f"versioned augmentum-workspace tag (PR 2 contract)"
        )


def test_profile_image_is_prebaked_exact_match():
    assert _profile_image_is_prebaked("augmentum-workspace:browser", "browser")
    assert _profile_image_is_prebaked("augmentum-workspace:power", "power")
    assert _profile_image_is_prebaked("augmentum-workspace:standard", "standard")


def test_profile_image_is_prebaked_mismatches_return_false():
    # v1 generic tag is NOT a prebake match — fall through to install.
    assert not _profile_image_is_prebaked("augmentum-workspace", "browser")
    # Standard image with a power profile is a partial match, treat as not prebaked.
    assert not _profile_image_is_prebaked("augmentum-workspace:standard", "power")
    # Raw ubuntu obviously isn't prebaked.
    assert not _profile_image_is_prebaked("ubuntu:24.04", "browser")


def test_profile_image_is_prebaked_unknown_profile_returns_false():
    # Unknown profile = always install (better safe than half-baked).
    assert not _profile_image_is_prebaked("augmentum-workspace:browser", "kali")


def _make_manager_with_inspect(*resolves: str):
    """Build a ContainerManager whose docker.images.inspect resolves ONLY
    for image names in ``resolves``. Anything else raises."""
    available = set(resolves)

    async def fake_inspect(name: str):
        if name in available:
            return {"Id": f"sha256:{hash(name) & 0xffff:04x}"}
        raise RuntimeError(f"image {name!r} not present")

    docker = MagicMock()
    docker.images = MagicMock()
    docker.images.inspect = AsyncMock(side_effect=fake_inspect)
    return ContainerManager(docker=docker, db=None)


@pytest.mark.asyncio
async def test_resolve_image_uses_profile_tag_when_caller_passes_generic():
    """When the caller passes the v1 generic name, the resolver should
    upgrade to the profile's preferred tag before falling back."""
    mgr = _make_manager_with_inspect(
        "augmentum-workspace:browser",
        "augmentum-workspace:power",
        "augmentum-workspace:standard",
        "augmentum-workspace",
    )
    img = await mgr._resolve_workspace_image("augmentum-workspace", "browser")
    assert img == "augmentum-workspace:browser"


@pytest.mark.asyncio
async def test_resolve_image_falls_back_to_generic_when_profile_tag_missing():
    """No :browser yet? Fall back to the v1 generic tag and let the
    install-line path layer extras on top."""
    mgr = _make_manager_with_inspect("augmentum-workspace")
    img = await mgr._resolve_workspace_image("augmentum-workspace", "browser")
    assert img == "augmentum-workspace"


@pytest.mark.asyncio
async def test_resolve_image_falls_back_to_ubuntu_when_nothing_else():
    """Final stop: ubuntu:24.04 + the runtime bootstrap install path."""
    mgr = _make_manager_with_inspect("ubuntu:24.04")
    img = await mgr._resolve_workspace_image("augmentum-workspace", "browser")
    assert img == "ubuntu:24.04"


@pytest.mark.asyncio
async def test_resolve_image_honors_explicit_non_default_base_image():
    """If the caller passes a custom tag (legacy / pinned), don't override
    it — they want what they asked for."""
    mgr = _make_manager_with_inspect("my-custom:tag")
    img = await mgr._resolve_workspace_image("my-custom:tag", "browser")
    assert img == "my-custom:tag"


@pytest.mark.asyncio
async def test_resolve_image_falls_through_when_explicit_tag_missing():
    """Explicit tag missing → try the profile tag, then fall back."""
    mgr = _make_manager_with_inspect("augmentum-workspace:power")
    img = await mgr._resolve_workspace_image("my-missing:tag", "power")
    assert img == "augmentum-workspace:power"
