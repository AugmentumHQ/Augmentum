"""Source-level invariants for the body physics stack.

These tests don't execute JavaScript — they validate the JS source files
under ``ui/scripts/`` via Read + regex/string checks, following the
established Augmentum pattern (see ``.claude/skills/augmentum-dev/scripts/
audit.py``).

The body physics stack lives across several sibling modules:

  * ``sdf-compliance.js``           — SDF-gradient body compliance channel
  * ``contact-reactor.js``          — proxemic contact + reach state machine
  * ``avatar-xr-compliance.js``     — XR lifecycle for SDFCompliance
  * ``avatar-xr-rapier.js``         — XR lifecycle for the Rapier ragdoll
  * ``rapier-ragdoll.js``           — Rapier-based active ragdoll
  * ``body-physics-coordinator.js`` — settings → live-instance push-down

The last two modules may not exist yet (sibling agents in flight); tests
for those use ``pytest.skip`` so the suite stays green either way.

Invariants verified here are the ones that have actually regressed in
the past (e.g. the SDFCompliance ``premultiply`` vs ``multiply`` bug
where the bone's rotation axis co-rotated with the bone, producing a
non-world-stable response) or that downstream peers depend on contractually
(e.g. ``ContactReactor.getUserHands()`` is the shared input source for
``SDFCompliance`` — renaming it silently would break compliance).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_SCRIPTS = REPO_ROOT / "ui" / "scripts"


def _read(name: str) -> str:
    """Read a ui/scripts/<name> module, skipping if absent."""
    path = UI_SCRIPTS / name
    if not path.exists():
        pytest.skip(f"{name} not present yet (sibling module in flight)")
    return path.read_text(encoding="utf-8")


def _exists(name: str) -> bool:
    return (UI_SCRIPTS / name).exists()


# ---------------------------------------------------------------------------
# 1. sdf-compliance.js
# ---------------------------------------------------------------------------


class TestSDFCompliance:
    MODULE = "sdf-compliance.js"

    EXPECTED_BONES = (
        "spine",
        "chest",
        "upperChest",
        "neck",
        "head",
        "leftShoulder",
        "rightShoulder",
    )

    def test_module_exists(self):
        assert _exists(self.MODULE), f"{self.MODULE} missing — body compliance not wired"

    def test_bone_arms_contains_all_seven_bones(self):
        src = _read(self.MODULE)
        bone_arms_match = re.search(
            r"const\s+BONE_ARMS\s*=\s*Object\.freeze\(\{(.*?)\}\)",
            src,
            re.DOTALL,
        )
        assert bone_arms_match, "BONE_ARMS table not found"
        body = bone_arms_match.group(1)
        for bone in self.EXPECTED_BONES:
            assert re.search(rf"\b{bone}\s*:", body), f"BONE_ARMS missing bone: {bone}"

    def test_bone_arm_values_within_reasonable_range(self):
        """Per-bone lever arm lengths should be between 8cm and 20cm."""
        src = _read(self.MODULE)
        bone_arms_match = re.search(
            r"const\s+BONE_ARMS\s*=\s*Object\.freeze\(\{(.*?)\}\)",
            src,
            re.DOTALL,
        )
        assert bone_arms_match
        body = bone_arms_match.group(1)
        entries = re.findall(r"(\w+)\s*:\s*([\d.]+)", body)
        assert entries, "no key:number entries parsed from BONE_ARMS"
        for name, raw in entries:
            value = float(raw)
            assert 0.08 <= value <= 0.20, (
                f"BONE_ARMS.{name}={value} outside 0.08..0.20 range"
            )

    def test_spring_constants_declared(self):
        src = _read(self.MODULE)
        for const in ("ACTIVE_RANGE_M", "FALLOFF_RANGE_M", "MAX_DISP_M", "RECOVER_HZ"):
            assert re.search(rf"\bconst\s+{const}\s*=", src), (
                f"{const} not declared at module scope"
            )

    def test_active_range_smaller_than_falloff_range(self):
        """ACTIVE_RANGE_M < FALLOFF_RANGE_M — engagement zone is tighter
        than the bone-participation falloff."""
        src = _read(self.MODULE)
        active = float(re.search(r"ACTIVE_RANGE_M\s*=\s*([\d.]+)", src).group(1))
        falloff = float(re.search(r"FALLOFF_RANGE_M\s*=\s*([\d.]+)", src).group(1))
        assert active < falloff, (
            f"ACTIVE_RANGE_M ({active}) must be < FALLOFF_RANGE_M ({falloff})"
        )

    def test_max_displacement_within_sane_bound(self):
        """MAX_DISP_M must be a non-trivial but bounded value (1cm..10cm)."""
        src = _read(self.MODULE)
        max_disp = float(re.search(r"MAX_DISP_M\s*=\s*([\d.]+)", src).group(1))
        assert 0.01 <= max_disp <= 0.10, f"MAX_DISP_M={max_disp} outside 0.01..0.10"

    def test_magnitude_clamp_present(self):
        """The spring-integrate path clamps |state.current| ≤ MAX_DISP_M."""
        src = _read(self.MODULE)
        assert re.search(r"if\s*\(\s*m\s*>\s*MAX_DISP_M\s*\)", src), (
            "Magnitude clamp `if (m > MAX_DISP_M)` not found — integration overshoot "
            "is no longer bounded."
        )

    def test_quaternion_uses_premultiply_not_multiply(self):
        """Critical bugfix invariant. ``premultiply`` applies the delta in the
        parent's frame so the world push direction stays world-stable as the
        bone tilts. If somebody regresses this to ``multiply``/right-multiply,
        the rotation axis co-rotates with the bone and compliance becomes
        unstable.
        """
        src = _read(self.MODULE)
        assert "node.quaternion.premultiply" in src, (
            "node.quaternion.premultiply(...) regressed — compliance world axis broken"
        )
        # The right-multiply form must NOT appear on node.quaternion.
        assert not re.search(r"node\.quaternion\.multiply\s*\(", src), (
            "node.quaternion.multiply(...) detected — should be premultiply()"
        )

    def test_public_stiffness_and_recover_hz_mutable(self):
        """Coordinator needs to push gain settings down at runtime."""
        src = _read(self.MODULE)
        assert re.search(r"this\.stiffness\s*=", src), "this.stiffness not assigned"
        assert re.search(r"this\.recoverHz\s*=", src), "this.recoverHz not assigned"

    def test_exports_sdf_compliance_class(self):
        src = _read(self.MODULE)
        assert re.search(r"export\s+class\s+SDFCompliance\b", src)

    def test_reset_and_dispose_present(self):
        src = _read(self.MODULE)
        assert re.search(r"\breset\s*\(\s*\)\s*\{", src), "reset() missing"
        assert re.search(r"\bdispose\s*\(\s*\)\s*\{", src), "dispose() missing"


# ---------------------------------------------------------------------------
# 2. contact-reactor.js
# ---------------------------------------------------------------------------


class TestContactReactor:
    MODULE = "contact-reactor.js"

    def test_module_exists(self):
        assert _exists(self.MODULE)

    def test_exports_contact_reactor_class(self):
        src = _read(self.MODULE)
        assert re.search(r"export\s+class\s+ContactReactor\b", src)

    def test_get_user_hands_public_accessor(self):
        """SDFCompliance reads hand positions through this — renaming breaks
        the peer contract silently."""
        src = _read(self.MODULE)
        assert re.search(r"\bgetUserHands\s*\(\s*\)\s*\{", src), (
            "getUserHands() accessor missing — SDFCompliance peer contract broken"
        )

    def test_state_machine_values_present(self):
        src = _read(self.MODULE)
        for state in ("idle", "approach", "hover", "contact"):
            assert f"'{state}'" in src, f"state '{state}' missing from transition code"

    def test_distance_thresholds_declared(self):
        src = _read(self.MODULE)
        for const in ("REACH_DIST_M", "HOVER_DIST_M", "CONTACT_DIST_M"):
            assert re.search(rf"\bconst\s+{const}\s*=", src), f"{const} missing"

    def test_distance_thresholds_ordered_correctly(self):
        """CONTACT < HOVER < REACH — closer states have tighter thresholds."""
        src = _read(self.MODULE)
        reach = float(re.search(r"REACH_DIST_M\s*=\s*([\d.]+)", src).group(1))
        hover = float(re.search(r"HOVER_DIST_M\s*=\s*([\d.]+)", src).group(1))
        contact = float(re.search(r"CONTACT_DIST_M\s*=\s*([\d.]+)", src).group(1))
        assert contact < hover < reach, (
            f"distance threshold ordering broken: "
            f"CONTACT={contact}, HOVER={hover}, REACH={reach}"
        )

    def test_distance_thresholds_in_meters(self):
        """Sanity: all three values should be small (< 1m) meter-scale."""
        src = _read(self.MODULE)
        for const in ("REACH_DIST_M", "HOVER_DIST_M", "CONTACT_DIST_M"):
            value = float(re.search(rf"{const}\s*=\s*([\d.]+)", src).group(1))
            assert 0.0 < value < 1.0, f"{const}={value} is not a meter-scale value"

    def test_set_user_hand_and_tick_present(self):
        src = _read(self.MODULE)
        assert re.search(r"\bsetUserHand\s*\(", src)
        assert re.search(r"\btick\s*\(\s*dtMs", src)


# ---------------------------------------------------------------------------
# 3. avatar-xr-compliance.js
# ---------------------------------------------------------------------------


class TestAvatarXRCompliance:
    MODULE = "avatar-xr-compliance.js"

    REQUIRED_EXPORTS = (
        "initXRCompliance",
        "tickXRCompliance",
        "teardownXRCompliance",
        "getXRCompliance",
    )

    def test_module_exists(self):
        assert _exists(self.MODULE)

    def test_required_exports(self):
        src = _read(self.MODULE)
        for name in self.REQUIRED_EXPORTS:
            assert re.search(rf"export\s+function\s+{name}\b", src), (
                f"missing export: {name}"
            )

    def test_imports_sdf_compliance(self):
        src = _read(self.MODULE)
        assert "from './sdf-compliance.js'" in src
        assert "SDFCompliance" in src

    def test_late_binds_contact_reactor_inside_tick(self):
        """Init may run before the contact reactor is constructed (depends on
        avatar-xr.js ordering). tick() must rebind the reactor the first
        frame it becomes available — without this, compliance is dead-quiet
        for the whole session.
        """
        src = _read(self.MODULE)
        tick_match = re.search(
            r"export\s+function\s+tickXRCompliance\s*\([^)]*\)\s*\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert tick_match, "tickXRCompliance body not found"
        body = tick_match.group(1)
        assert "getXRContactReactor()" in body, (
            "tickXRCompliance does not late-bind via getXRContactReactor() — "
            "reactor will never attach if reactor init followed compliance init"
        )

    def test_init_log_shape(self):
        """The init debug log should expose hasAtlas, hasReactor, trackedBones
        — these fields drive a downstream HUD/inspector and removing them
        would break the diagnostic surface."""
        src = _read(self.MODULE)
        for key in ("hasAtlas", "hasReactor", "trackedBones"):
            assert key in src, f"init log missing '{key}'"

    def test_imports_get_xr_contact_reactor(self):
        src = _read(self.MODULE)
        assert "getXRContactReactor" in src
        assert "from './avatar-xr-contact.js'" in src


# ---------------------------------------------------------------------------
# 4. avatar-xr-rapier.js
# ---------------------------------------------------------------------------


class TestAvatarXRRapier:
    MODULE = "avatar-xr-rapier.js"

    REQUIRED_EXPORTS = (
        "initXRRapier",
        "tickXRRapier",
        "teardownXRRapier",
        "getXRRapier",
    )

    def test_module_exists_or_skip(self):
        if not _exists(self.MODULE):
            pytest.skip(f"{self.MODULE} not present yet")
        # Smoke read so subsequent tests can fail with useful detail
        _read(self.MODULE)

    def test_required_exports(self):
        src = _read(self.MODULE)
        for name in self.REQUIRED_EXPORTS:
            assert re.search(rf"export\s+(?:async\s+)?function\s+{name}\b", src), (
                f"missing export: {name}"
            )

    def test_handles_async_init(self):
        """Rapier loads asynchronously (WASM). The wrapper must track an
        in-flight init promise so teardown can wait on it before disposing
        a half-constructed world."""
        src = _read(self.MODULE)
        assert "_initPromise" in src, (
            "no _initPromise tracking — teardown can race a half-constructed world"
        )

    def test_imports_rapier_ragdoll(self):
        src = _read(self.MODULE)
        assert "RapierRagdoll" in src
        assert "from './rapier-ragdoll.js'" in src

    def test_teardown_awaits_init_promise(self):
        """teardown should await the in-flight init (cap'd) so dispose sees
        a settled state."""
        src = _read(self.MODULE)
        teardown_match = re.search(
            r"export\s+async\s+function\s+teardownXRRapier\s*\([^)]*\)\s*\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert teardown_match, "teardownXRRapier body not found"
        body = teardown_match.group(1)
        assert "await" in body, "teardownXRRapier does not await — race risk"


# ---------------------------------------------------------------------------
# 5. rapier-ragdoll.js
# ---------------------------------------------------------------------------


class TestRapierRagdoll:
    MODULE = "rapier-ragdoll.js"

    REQUIRED_METHODS = ("init", "tick", "getBoneDeltas", "reset", "dispose")

    def test_module_exists_or_skip(self):
        if not _exists(self.MODULE):
            pytest.skip(f"{self.MODULE} not present yet (sibling agent in flight)")
        _read(self.MODULE)

    def test_exports_rapier_ragdoll_class(self):
        src = _read(self.MODULE)
        assert re.search(r"export\s+class\s+RapierRagdoll\b", src), (
            "RapierRagdoll class export missing"
        )

    def test_required_methods_present(self):
        src = _read(self.MODULE)
        for method in self.REQUIRED_METHODS:
            # Match either "method(" or "async method("
            assert re.search(rf"(?:async\s+)?\b{method}\s*\(", src), (
                f"RapierRagdoll missing method: {method}"
            )

    def test_joint_limits_documented(self):
        """Joint limits should be documented somewhere in the file (comments
        or constants). These are biomechanical safety bounds — silent drift
        produces a contortionist.

        Accepts either same-line anatomical naming (``spine ... 15``) or the
        VRM humanoid lowerArm/lowerLeg convention used for the elbow/knee
        hinge joints. We just need the numeric limit and the body part
        named somewhere in the file.
        """
        src = _read(self.MODULE)
        # Spine ±15°
        assert re.search(r"spine[^\n]*15", src, re.IGNORECASE), (
            "spine ±15° joint limit not documented"
        )
        # Neck ±30°
        assert re.search(r"neck[^\n]*30", src, re.IGNORECASE), (
            "neck ±30° pitch limit not documented"
        )
        # Shoulders ±90°
        assert re.search(r"shoulder[^\n]*90", src, re.IGNORECASE), (
            "shoulder ±90° limit not documented"
        )
        # Elbows 0-150° — anatomically the elbow joint lives on the
        # lower-arm bone in VRM humanoid naming. Either the word "elbow"
        # or "LowerArm" must appear near a 150-valued limit.
        assert re.search(r"(?:elbow|LowerArm)[^\n]*150", src, re.IGNORECASE) or (
            "elbow" in src.lower() and re.search(r"LowerArm[^\n]*150", src)
        ), "elbow 0-150° limit not documented (looked for elbow|LowerArm near 150)"
        # Knees 0-145° — same anatomical convention as elbows above.
        assert re.search(r"(?:knee|LowerLeg)[^\n]*145", src, re.IGNORECASE) or (
            "knee" in src.lower() and re.search(r"LowerLeg[^\n]*145", src)
        ), "knee 0-145° limit not documented (looked for knee|LowerLeg near 145)"

    def test_pd_spring_constants_present(self):
        """PD controller gains kp ≈ 4000, kd ≈ 400 (or similar) — tuned for
        stable ragdoll under typical frame rates."""
        src = _read(self.MODULE)
        assert "4000" in src, "PD spring constant kp=4000 not found"
        assert "400" in src, "PD damping constant kd=400 not found"

    def test_public_weight_property(self):
        src = _read(self.MODULE)
        assert re.search(r"this\.weight\s*=", src), (
            "RapierRagdoll.weight must be a public mutable property "
            "for coordinator push-down"
        )

    def test_rapier_import_failure_handled(self):
        """If Rapier WASM fails to load, the ragdoll should not crash the
        XR session — there must be try/catch around the import or init."""
        src = _read(self.MODULE)
        has_try_catch = "try" in src and "catch" in src
        assert has_try_catch, (
            "no try/catch detected — Rapier load failure will crash the session"
        )
        # And there should be SOME reference to RAPIER or rapier load path
        # inside a try-guarded region. Cheap heuristic: search for the import
        # specifier near a try block.
        assert re.search(r"RAPIER|@dimforge/rapier|rapier", src, re.IGNORECASE), (
            "no reference to Rapier — file looks empty/stub"
        )


# ---------------------------------------------------------------------------
# 6. body-physics-coordinator.js
# ---------------------------------------------------------------------------


class TestBodyPhysicsCoordinator:
    MODULE = "body-physics-coordinator.js"

    SETTINGS_KEYS = (
        "body_physics_enabled",
        "body_physics_compliance_gain",
        "body_physics_rapier_weight",
        "body_physics_recover_hz",
    )

    def test_module_exists_or_skip(self):
        if not _exists(self.MODULE):
            pytest.skip(f"{self.MODULE} not present yet (sibling agent in flight)")
        _read(self.MODULE)

    def test_known_settings_keys_referenced(self):
        src = _read(self.MODULE)
        for key in self.SETTINGS_KEYS:
            assert key in src, f"coordinator does not reference setting: {key}"

    def test_pushes_down_to_live_instances(self):
        """Push-down means writing settings values into the live instance
        fields the coordinator owns: SDFCompliance.stiffness,
        SDFCompliance.recoverHz, RapierRagdoll.weight."""
        src = _read(self.MODULE)
        # Must touch each live-instance property somewhere.
        assert re.search(r"\.stiffness\s*=", src), (
            "coordinator does not write to .stiffness on a live instance"
        )
        assert re.search(r"\.recoverHz\s*=", src), (
            "coordinator does not write to .recoverHz on a live instance"
        )
        assert re.search(r"\.weight\s*=", src), (
            "coordinator does not write to .weight on a live instance"
        )

    def test_exports_set_override_and_inspect(self):
        src = _read(self.MODULE)
        # Accept either named-function exports or method-style exports on
        # an exported object/class — match either pattern.
        for name in ("setOverride", "inspect"):
            patterns = [
                rf"export\s+function\s+{name}\b",
                rf"export\s+const\s+{name}\b",
                rf"\b{name}\s*\(",
            ]
            assert any(re.search(p, src) for p in patterns), (
                f"coordinator does not expose '{name}'"
            )
