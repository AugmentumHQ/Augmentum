"""Verify-time process hardening — the cheapest, highest-value slice of the
"don't let a candidate's code exfiltrate secrets" work (Phase B / W11).

Verification EXECUTES the candidate's code: boot-smoke imports ``create_app``,
pytest runs the authored tests, the audit walks the tree. Those subprocesses used
to inherit the app's full environment — ``AUGMENTUM_OPENAI_API_KEY``, the
HuggingFace tokens, the host-stats token — so a poisoned or careless edit got
code execution *plus* the credentials to phone them home, all BEFORE any human
saw the change. Every mature coding agent (OpenHands/Devin/Jules/Codex) forbids
exactly that.

``scrubbed_env`` is the first, cheapest defense: launch verify subprocesses with
the secrets stripped out. It is NOT the whole story — a full process sandbox
(bubblewrap/seccomp + deny-by-default egress + read-only /data) is the next rung
and needs image changes — but it closes the credential-exfiltration path for the
common case with zero new infrastructure and no risk to the verify signal (a
subprocess that only imports/tests the candidate doesn't need the API keys).

Denylist on secret-SHAPED names (not an allowlist) is deliberate here: an
allowlist would silently break verify when the app legitimately needs some
non-secret env var (locale, PYTHONPATH, non-secret AUGMENTUM_* config). We keep
those and drop only what looks like a credential — a big improvement over
"inherit everything," with the allowlist/full-sandbox as the hardening follow-up.
"""

from __future__ import annotations

import os

# Substrings that mark an env var as a credential to strip. Uppercased match, so
# it catches AUGMENTUM_OPENAI_API_KEY, HF_TOKEN, *_SECRET, *_PASSWORD, etc.
_SECRET_SHAPES = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
    "APIKEY", "PRIVATE", "SESSION", "COOKIE",
)

# Never strip these even if a shape substring matches — they're needed to run a
# Python subprocess and are not credentials (e.g. a KEYBOARD/DESKTOP var).
_ALWAYS_KEEP = frozenset({
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TEMP", "TMP",
    "PWD", "SHELL", "USER", "LOGNAME", "HOSTNAME", "PYTHONPATH", "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE", "VIRTUAL_ENV", "SYSTEMROOT", "PATHEXT",
})


def is_secret_name(name: str) -> bool:
    """True if an env var name looks like a credential (and isn't allow-listed)."""
    if name in _ALWAYS_KEEP:
        return False
    up = name.upper()
    return any(s in up for s in _SECRET_SHAPES)


def scrubbed_env(*, base: dict | None = None, extra_drop: tuple = ()) -> dict:
    """A copy of the environment with credential-shaped vars removed — for any
    subprocess that runs candidate code during verification. ``extra_drop`` lets a
    caller strip additional exact names. Keeps everything a Python subprocess
    needs; drops anything that looks like a secret."""
    src = os.environ if base is None else base
    drop_exact = set(extra_drop)
    return {
        k: v for k, v in src.items()
        if k not in drop_exact and not is_secret_name(k)
    }
