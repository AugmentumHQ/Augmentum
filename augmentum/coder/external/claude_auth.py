"""Claude subscription login via the SANCTIONED browser OAuth flow.

The legitimate way to use a Claude Pro/Max subscription headlessly is
``claude setup-token``: it walks the user through the OFFICIAL browser OAuth
authorization (as Claude Code — the blessed client) and prints a 1-year
long-lived token ``sk-ant-oat01-…`` that the Agent SDK consumes via the
``CLAUDE_CODE_OAUTH_TOKEN`` env var. Usage counts against the subscription plan,
not a per-token API invoice. Since Claude Code v2.1 the flow accepts the OAuth
code pasted back when the browser callback can't reach localhost (containers).

This is deliberately NOT the impersonation/PKCE-spoofing path (mint a
third-party OAuth token by faking Claude Code wire traffic — ToS-gray, fragile,
an arms race). We drive the REAL CLI's official login and capture its output.
The token authenticates against Claude Code only — exactly what the driver runs.

Built + tested here: the pure pieces — the spawn command, the auth-URL and
token extractors, and the credential→env mapping. The interactive spawn +
browser hand-off + encrypted per-user storage + the ``/api`` route + UI button
are the live, on-device slice (need a real ``claude`` CLI + a browser to verify).
"""

from __future__ import annotations

import re

# argv to launch the official browser-OAuth flow. Prints an authorization URL,
# waits for the user to authorize in their browser (or paste the code), then
# prints the long-lived token.
SETUP_TOKEN_CMD: tuple[str, ...] = ("claude", "setup-token")

_OAUTH_TOKEN_RE = re.compile(r"sk-ant-oat01-[A-Za-z0-9_\-]+")
# The authorize URL setup-token prints — Anthropic-hosted OAuth.
_AUTH_URL_RE = re.compile(
    r"https://(?:claude\.ai|console\.anthropic\.com|[A-Za-z0-9.-]*anthropic\.com)/[^\s\"'<>]+"
)


def parse_setup_token(text: str) -> str | None:
    """Extract the ``sk-ant-oat01-…`` long-lived token from setup-token output."""
    m = _OAUTH_TOKEN_RE.search(text or "")
    return m.group(0) if m else None


def parse_auth_url(text: str) -> str | None:
    """Extract the OAuth authorize URL to open in the user's browser."""
    m = _AUTH_URL_RE.search(text or "")
    return m.group(0) if m else None


def is_oauth_token(token: str) -> bool:
    """True for a subscription OAuth token (vs a console API key)."""
    return (token or "").startswith("sk-ant-oat01-")


def auth_env(*, oauth_token: str = "", api_key: str = "") -> dict[str, str]:
    """Map a credential to the env var the Agent SDK reads. OAuth (subscription)
    takes precedence; an API key (per-token billing) is the fallback. Empty dict
    when neither is set (driver then relies on a logged-in CLI credential)."""
    if oauth_token:
        # A token that looks like an API key was passed in the oauth slot →
        # route it correctly rather than mislabel it.
        if is_oauth_token(oauth_token):
            return {"CLAUDE_CODE_OAUTH_TOKEN": oauth_token}
        return {"ANTHROPIC_API_KEY": oauth_token}
    if api_key:
        return {"ANTHROPIC_API_KEY": api_key}
    return {}
