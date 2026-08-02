"""Tests for the pocket-tts pinned-override vendor.

These tests run in any environment — they do NOT require ``pocket-tts``
to be installed. The point is to verify the override module's import
shape + drift-detection behavior, not to exercise upstream codec code
(that lives in the GPU image integration test path).

Pins:
  - upstream_pin module is dep-free (no torch import via reading it)
  - Public API surface exports the documented symbols
  - When pocket_tts is absent, MimiTappedTTSModel is None (graceful degrade)
  - When pocket_tts is absent, upstream_available() returns False
  - When pocket_tts is absent, try_install_tap returns False (no raise)
  - REQUIRED_UPSTREAM_METHODS includes the tap's actual hook point
  - UPSTREAM_COMMIT looks like a full git SHA (40 hex chars)
"""
from __future__ import annotations

import re

# ── Pin constants ───────────────────────────────────────────────


class TestUpstreamPin:
    def test_commit_is_full_sha(self):
        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            UPSTREAM_COMMIT,
        )
        assert re.fullmatch(r"[0-9a-f]{40}", UPSTREAM_COMMIT) is not None, (
            f"UPSTREAM_COMMIT must be a 40-char hex SHA, got {UPSTREAM_COMMIT!r}"
        )

    def test_date_is_iso8601_date(self):
        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            UPSTREAM_DATE,
        )
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", UPSTREAM_DATE) is not None

    def test_required_methods_includes_tap_point(self):
        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            REQUIRED_UPSTREAM_METHODS,
        )
        # The tap subclasses _decode_audio_worker. If we forget to list it
        # here, the drift checker won't catch upstream removing it.
        assert "_decode_audio_worker" in REQUIRED_UPSTREAM_METHODS

    def test_required_methods_includes_public_surface(self):
        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            REQUIRED_UPSTREAM_METHODS,
        )
        # The audio-history pipeline calls these on the upstream model
        # via the override; their disappearance is also drift we care about.
        for name in ("generate_audio_stream", "load_model"):
            assert name in REQUIRED_UPSTREAM_METHODS

    def test_repo_url_is_upstream(self):
        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            UPSTREAM_REPO,
        )
        assert UPSTREAM_REPO == "https://github.com/kyutai-labs/pocket-tts"


# ── Public API surface ──────────────────────────────────────────


class TestPublicAPI:
    def test_init_exports(self):
        """The vendored package __init__ exports the documented symbols."""
        import augmentum.voice._vendored.pocket_tts as mod
        for name in (
            "MimiTappedTTSModel",
            "try_install_tap",
            "upstream_available",
            "UPSTREAM_COMMIT",
            "UPSTREAM_DATE",
            "UPSTREAM_REPO",
            "REQUIRED_UPSTREAM_METHODS",
            "MIN_TESTED_VERSION",
            "MimiCodesCallback",
        ):
            assert hasattr(mod, name), f"vendor __init__ missing export: {name}"

    def test_upstream_pin_module_is_dep_free(self):
        """upstream_pin must NOT pull torch / scipy / pocket_tts at import.

        The whole point of keeping it separate from tap.py is so a quick
        ``python -c "from ...upstream_pin import UPSTREAM_COMMIT"`` works
        in any environment for ops scripts that need to read the pin.
        """
        import sys
        # Snapshot what's already loaded
        before = set(sys.modules)
        # Re-import via subprocess would be cleaner but slower; we
        # settle for verifying the module's source has no top-level
        # imports of the heavy deps.
        import pathlib
        src = pathlib.Path(
            "augmentum/voice/_vendored/pocket_tts/upstream_pin.py",
        ).read_text()
        for forbidden in ("import torch", "import scipy", "import pocket_tts"):
            assert forbidden not in src, (
                f"upstream_pin.py must not import {forbidden!r} at module "
                f"scope — defeats the dep-free guarantee"
            )
        # Re-import to confirm
        if "augmentum.voice._vendored.pocket_tts.upstream_pin" in sys.modules:
            del sys.modules["augmentum.voice._vendored.pocket_tts.upstream_pin"]
        from augmentum.voice._vendored.pocket_tts import upstream_pin  # noqa: F401
        _ = before  # unused but documents intent


# ── Graceful degradation when pocket_tts isn't installed ────────


class TestNoPocketTtsEnvironment:
    """In the dev venv (no pocket-tts), the override module must:
       - import cleanly
       - expose MimiTappedTTSModel as None
       - upstream_available() returns False
       - try_install_tap returns False (no raise)
    """

    def test_mimi_tapped_is_none_without_upstream(self):
        # We can't easily simulate the "pocket_tts missing" path inside
        # an environment that has it; we just verify the current
        # environment's MimiTappedTTSModel state matches upstream_available.
        from augmentum.voice._vendored.pocket_tts import (
            MimiTappedTTSModel,
            upstream_available,
        )
        if upstream_available():
            # CI/Docker may have pocket_tts installed — different test path
            assert MimiTappedTTSModel is not None
        else:
            assert MimiTappedTTSModel is None

    def test_try_install_tap_returns_false_without_upstream(self):
        from augmentum.voice._vendored.pocket_tts import (
            try_install_tap,
            upstream_available,
        )
        if not upstream_available():
            # Passing any object is fine — the function should short-circuit
            assert try_install_tap(object(), callback=None) is False

    def test_try_install_tap_no_raise(self):
        """Even with a bogus model, try_install_tap must not raise."""
        from augmentum.voice._vendored.pocket_tts import try_install_tap

        class _BogusModel:
            pass  # no _decode_audio_worker

        # Should return False, not raise AttributeError
        assert try_install_tap(_BogusModel(), callback=None) is False


# ── Drift detection ─────────────────────────────────────────────


class TestDriftDetection:
    def test_check_upstream_surface_warns_on_missing_method(self, caplog):
        """If upstream removes a required method, we log a warning."""
        from augmentum.voice._vendored.pocket_tts.tap import (
            _check_upstream_surface,
        )

        class _FakeUpstream:
            # Deliberately missing _decode_audio_worker etc.
            pass

        import logging
        with caplog.at_level(logging.WARNING):
            _check_upstream_surface(_FakeUpstream)
        assert any(
            "pocket_tts_vendor_drift" in rec.message
            for rec in caplog.records
        )

    def test_check_upstream_surface_silent_when_all_present(self, caplog):
        from augmentum.voice._vendored.pocket_tts.tap import (
            _check_upstream_surface,
        )
        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            REQUIRED_UPSTREAM_METHODS,
        )

        class _CompleteFake:
            pass

        for name in REQUIRED_UPSTREAM_METHODS:
            setattr(_CompleteFake, name, lambda self: None)

        import logging
        with caplog.at_level(logging.WARNING):
            _check_upstream_surface(_CompleteFake)
        drift_warnings = [
            rec for rec in caplog.records
            if "pocket_tts_vendor_drift" in rec.message
        ]
        assert drift_warnings == []


# ── Vendor manifest documents ──────────────────────────────────


class TestVendorDocs:
    """The VENDOR.md + LICENSE.upstream documents must exist + name
    the same pinned commit as upstream_pin.py.
    """

    def test_vendor_md_exists(self):
        import pathlib
        path = pathlib.Path(
            "augmentum/voice/_vendored/pocket_tts/VENDOR.md",
        )
        assert path.exists(), "VENDOR.md must exist alongside tap.py"

    def test_vendor_md_includes_pinned_commit(self):
        import pathlib

        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            UPSTREAM_COMMIT,
        )
        text = pathlib.Path(
            "augmentum/voice/_vendored/pocket_tts/VENDOR.md",
        ).read_text()
        assert UPSTREAM_COMMIT in text, (
            "VENDOR.md must reference the same pinned commit as upstream_pin.py"
        )

    def test_license_upstream_exists(self):
        import pathlib
        path = pathlib.Path(
            "augmentum/voice/_vendored/pocket_tts/LICENSE.upstream",
        )
        assert path.exists()
        text = path.read_text()
        # MIT license fingerprint
        assert "Permission is hereby granted" in text
        assert "MERCHANTABILITY" in text

    def test_license_upstream_references_pin(self):
        import pathlib

        from augmentum.voice._vendored.pocket_tts.upstream_pin import (
            UPSTREAM_COMMIT,
        )
        text = pathlib.Path(
            "augmentum/voice/_vendored/pocket_tts/LICENSE.upstream",
        ).read_text()
        assert UPSTREAM_COMMIT in text
