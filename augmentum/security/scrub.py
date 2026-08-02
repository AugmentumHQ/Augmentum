"""Secret scrubbing for pre-auth / non-admin response surfaces.

Used wherever response payloads leave the trust boundary on a path the
auth middleware can't (or doesn't) gate by session: pre-login config
reads, public endpoints, error responses that may have echoed the
request body back, audit logs that capture handler input/output, etc.

The companion helper to :mod:`augmentum.security.untrusted`. Where
``untrusted`` protects content GOING INTO the model, ``scrub`` protects
content COMING OUT to the wire.

Design contract:

* **Stateless** — pure string / dict transformation, no DB, no config.
* **Suffix-keyed** — match on field-name suffixes (``_api_key``,
  ``_password``, ``_token``) rather than substrings. Substring matches
  produce false positives on words like ``monkey`` (contains ``key``)
  and ``broker`` (contains ``ker``).
* **Recursive** — nested dicts and lists are traversed so a secret
  stored under a non-secret parent key still gets caught.
* **Allowlist** — public identifiers that follow the secret naming
  pattern but are not actually secret (``google_pse_cx`` is a public
  CSE identifier).
* **Exact-match sensitive list** — keys that don't fit the secret
  naming pattern but are still capability handles we don't expose to
  pre-auth callers (e.g. ``reminder_webhook_integration_id``).
* **Replacement marker** — secret leaves become :data:`REDACTED`. Keep
  it a string so JSON consumers expecting strings don't error; keep it
  recognizable so accidental shipping to logs is obvious.

When you find a leak, add a test in
``tests/test_security_scrub.py`` so the regression is pinned.
"""

from __future__ import annotations

from typing import Any

# Replacement marker for scrubbed values. String (not None) so any
# downstream consumer expecting a string type doesn't blow up; the
# square brackets make accidental log appearances visually obvious.
REDACTED = "[REDACTED]"

# Secret-shaped suffixes. A field whose name ENDS WITH any of these
# (or EQUALS the stripped form) is treated as a secret.
_SECRET_SUFFIXES = (
    "_api_key", "_apikey",
    "_password", "_passwd", "_pass", "_pwd",
    "_secret", "_client_secret",
    "_token", "_access_token", "_refresh_token", "_bearer_token",
    "_credential", "_credentials",
    "_key",
    "_signing_key", "_encryption_key",
    "_private_key",
    "_otp_secret", "_totp_secret",
    "_session_id",
    "_signature",
    "_cookie",
)

# Public identifiers that LOOK like secrets (suffix match) but are not.
# Anything here is allowed through scrubbing unchanged.
_SECRET_KEY_ALLOWLIST = frozenset({
    # Google Programmable Search Engine: cx parameter is a public ID.
    "google_pse_cx",
})

# Field names that are NOT secret-shaped but are still sensitive in
# pre-auth contexts (capability handles for routes that can do things).
# Exact-match — not suffix-matched.
_SENSITIVE_EXACT_KEYS = frozenset({
    # Webhook integration ID is a stable handle for routes that can
    # trigger outbound webhook sends; even though it's not a secret in
    # the cryptographic sense, exposing it to non-admin callers gives
    # them a way to reference (and potentially poke at) the integration.
    "reminder_webhook_integration_id",
})


def is_secret_key(name: str | None) -> bool:
    """Return True if ``name`` denotes a secret-shaped field.

    Decision order:
      1. None / empty → not secret
      2. Exact allowlist hit → not secret
      3. Exact sensitive list hit → secret
      4. Suffix match → secret
      5. Otherwise → not secret

    The function is case-insensitive on the field name; callers don't
    have to normalise.
    """
    if not name:
        return False
    n = name.lower()
    if n in _SECRET_KEY_ALLOWLIST:
        return False
    if n in _SENSITIVE_EXACT_KEYS:
        return True
    # Suffix match — includes the stripped-leading-underscore form so
    # both ``api_key`` and ``provider_api_key`` are caught (the former
    # via ``_api_key`` lstripped, the latter via raw suffix).
    for suffix in _SECRET_SUFFIXES:
        if n.endswith(suffix):
            return True
        if n == suffix.lstrip("_"):
            return True
    return False


def _scrub_value(key: str | None, value: Any, *, deep: bool) -> Any:
    """Apply the secret-decision to one (key, value) pair.

    Containers (dict/list/tuple) recurse when ``deep`` is True regardless
    of whether the parent key itself looks secret, so a non-secret parent
    holding secret children still gets caught.
    """
    if key is not None and is_secret_key(key):
        return REDACTED
    if not deep:
        return value
    if isinstance(value, dict):
        return {k: _scrub_value(k, v, deep=True) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(None, item, deep=True) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(None, item, deep=True) for item in value)
    return value


def scrub_dict(data: dict[str, Any] | None, *, deep: bool = True) -> dict[str, Any]:
    """Return a copy of ``data`` with secret-shaped leaves redacted.

    Args:
        data: The dict to scrub. ``None`` returns an empty dict so
            callers can chain safely.
        deep: When True (default), recurse into nested dicts/lists.
            When False, only the top-level keys are inspected — useful
            for shallow path validation where deeper structure has
            already been validated.

    Returns:
        A new dict with the same shape; secret leaves replaced with
        :data:`REDACTED`. The input is NOT mutated.
    """
    if data is None:
        return {}
    if not isinstance(data, dict):
        # Defensive — callers occasionally pass response objects of
        # unexpected shape. Return empty rather than raising; the wire
        # layer will then send an empty payload, which is the
        # fail-closed answer.
        return {}
    return {k: _scrub_value(k, v, deep=deep) for k, v in data.items()}


def scrub_response(payload: Any, *, deep: bool = True) -> Any:
    """Scrub any value that might enter a response body.

    Convenience wrapper that handles dict / list / scalar uniformly so
    handlers can scrub without first inspecting the response shape.
    Scalars and unknown types pass through unchanged.
    """
    if isinstance(payload, dict):
        return scrub_dict(payload, deep=deep)
    if isinstance(payload, list):
        return [scrub_response(item, deep=deep) for item in payload]
    if isinstance(payload, tuple):
        return tuple(scrub_response(item, deep=deep) for item in payload)
    return payload
