"""Red herring: contains an `eval()` sink, but the function is never called.

The module exports `safe_compute`. `_legacy_compute` was retired but the
code was left in place during a refactor — no caller exists in this
module or any imported module. A useful detector either (a) doesn't flag
it because it's unreachable, or (b) flags it as `info` severity with a
note that no caller was found.
"""

from __future__ import annotations


def _legacy_compute(expr: str) -> float:
    # Dead code — no caller. eval() of a string would be an RCE sink if
    # reachable, but it isn't.
    return float(eval(expr))


def safe_compute(a: float, b: float) -> float:
    return a + b


__all__ = ["safe_compute"]
