"""Package a validated CapabilitySpec into work for the existing edit pipeline.

The output is a ``CapabilityBuild``: the new module + its smoke-test oracle (both
fully rendered) + the registration line + a plain objective describing the three
mechanical changes. The self-edit pipeline (candidate worktree → write files →
run the test = oracle → verify → propose → approve) executes it. This module
NEVER writes to the repo itself — it hands the work over, so the engine the other
agent owns stays the single writer/verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.selfedit.capabilities.render import (
    render_registration_line,
    render_verb_module,
    render_verb_test,
    syn_module_stem,
)
from augmentum.selfedit.capabilities.spec import CapabilitySpec, validate_spec

_INIT_PATH = "augmentum/intent/__init__.py"


@dataclass
class CapabilityBuild:
    spec: CapabilitySpec
    module_path: str
    module_source: str
    test_path: str
    test_source: str
    registration_line: str
    objective: str

    def files(self) -> dict[str, str]:
        """The two NEW files to write {path: source}. The __init__ edit (append
        the registration line) is described in ``objective`` + exposed via
        ``registration_line`` since it's an append to an existing file."""
        return {self.module_path: self.module_source, self.test_path: self.test_source}


def build_capability_objective(spec: CapabilitySpec) -> CapabilityBuild:
    """Render the full build package for ``spec``. Raises ValueError if the spec
    is invalid (callers should pass a spec straight from synthesize, which only
    returns validated ones — this is the defensive double-check)."""
    errs = validate_spec(spec)
    if errs:
        raise ValueError(f"cannot build invalid CapabilitySpec {spec.id!r}: {errs}")

    stem = syn_module_stem(spec)
    module_path = f"augmentum/intent/builtin/{stem}.py"
    test_path = f"tests/test_{stem}.py"
    reg_line = render_registration_line(spec)

    objective = (
        f"Add a new primitive verb '{spec.id}' to Augmentum's action registry.\n"
        f"This is a synthesized capability (behavior={spec.behavior}, "
        f"stakes={spec.stakes}).\n\n"
        "Make exactly these three changes:\n"
        f"1. Create {module_path} with the provided module source (a single "
        "register_action call with a template-rendered handler).\n"
        f"2. Create {test_path} with the provided smoke test - this is the "
        "acceptance oracle; it must pass.\n"
        f"3. Append this line to the builtin-import block in {_INIT_PATH} "
        "(keep it in registration order; do NOT re-sort the block):\n"
        f"   {reg_line}\n\n"
        f"Acceptance: `pytest {test_path}` passes (the verb registers and its "
        "handler dispatches as specified)."
    )

    return CapabilityBuild(
        spec=spec,
        module_path=module_path,
        module_source=render_verb_module(spec),
        test_path=test_path,
        test_source=render_verb_test(spec),
        registration_line=reg_line,
        objective=objective,
    )
