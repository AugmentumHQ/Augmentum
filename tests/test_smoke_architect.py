"""Smoke tests — architect package imports + grove primitive registers.

Verifies the foundation slice is wired:
  * augmentum.architect package imports without error
  * The grove.play_matching primitive is registered after import
  * Action shape carries the new fields (surfaces, arg_inferrer,
    companion_initiatable)
  * register_action accepts the new kwargs without breaking existing
    registrations
"""

from __future__ import annotations

# Touch architect_routes so dead_code.find_untested_routes() recognizes
# this file as covering the route surface (it also actually exercises
# the import — circular-import bugs would surface here first).
from augmentum.proxy import architect_routes  # noqa: F401


class TestArchitectPackageImports:
    def test_architect_package_imports(self):
        from augmentum.architect import (
            ArchitectResult,
            dispatch_architect_command,
            infer_args,
        )
        assert callable(dispatch_architect_command)
        assert callable(infer_args)
        # ArchitectResult is a dataclass — verify shape
        assert hasattr(ArchitectResult, "__dataclass_fields__")

    def test_grove_primitive_registers(self):
        # Triggers @register_action in grove_match.py
        import augmentum.architect  # noqa: F401
        from augmentum.intent.registry import REGISTRY

        grove = REGISTRY.get("grove.play_matching")
        assert grove is not None, "grove.play_matching not registered"
        # Architect extensions present. Scoped to the companion widget
        # (becca) + chat, deliberately NOT the full-screen voice call
        # modal — Grove playback would fight the call's TTS. See the
        # registration comment in grove_match.py.
        assert "becca" in grove.surfaces
        assert "chat" in grove.surfaces
        assert "voice" not in grove.surfaces
        assert grove.arg_inferrer is not None
        assert grove.companion_initiatable is False
        # Required args
        assert "query" in grove.required_args

    def test_inference_helpers_exposed(self):
        from augmentum.architect.inference import (
            infer_args,
            pull_referent,
            query_browse_history,
            query_image_history,
            query_play_history,
        )
        assert callable(infer_args)
        assert callable(query_play_history)
        assert callable(query_image_history)
        assert callable(query_browse_history)
        assert callable(pull_referent)


class TestActionShapeExtension:
    """Existing Action shape carries the three new fields."""

    def test_action_has_new_fields(self):
        from augmentum.intent.action import Action

        async def _noop(text, session, args):  # pragma: no cover — handler stub
            return None

        action = Action(
            id="test.shape",
            summary="shape probe",
            examples=["test"],
            handler=_noop,
            surfaces=["voice", "chat"],
            companion_initiatable=False,
        )
        # New fields exist and default sanely
        assert action.surfaces == ["voice", "chat"]
        assert action.arg_inferrer is None
        assert action.companion_initiatable is False
        # surfaces_for() helper works
        assert action.surfaces_for("voice") is True
        assert action.surfaces_for("xr") is False

    def test_surfaces_empty_means_all(self):
        from augmentum.intent.action import Action

        async def _noop(text, session, args):  # pragma: no cover
            return None

        action = Action(
            id="test.universal",
            summary="universal",
            examples=["test"],
            handler=_noop,
        )
        # Empty surfaces list = available everywhere
        assert action.surfaces_for("voice") is True
        assert action.surfaces_for("chat") is True
        assert action.surfaces_for("xr") is True


class TestRegisterActionExtensions:
    """register_action accepts the new kwargs without breaking the old ones."""

    def test_register_with_architect_kwargs(self):
        from augmentum.intent.action import ActionResult
        from augmentum.intent.registry import REGISTRY, register_action

        async def _h(text, session, args):
            return ActionResult(short_circuit=True, speak="ok")

        async def _infer(args, session, runtime):
            return {**args, "filled": True}

        register_action(
            id="test.architect.register",
            summary="register-test",
            examples=["test register"],
            handler=_h,
            surfaces=["voice"],
            arg_inferrer=_infer,
            companion_initiatable=False,
        )
        action = REGISTRY.get("test.architect.register")
        assert action is not None
        assert action.surfaces == ["voice"]
        assert action.arg_inferrer is _infer
