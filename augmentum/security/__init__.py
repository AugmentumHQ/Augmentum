"""Security primitives — prompt-injection defense, secret scrubbing.

Stateless helpers shared across subsystems. No DB access, no config
dependencies (except optional logging). Modules:

* ``untrusted`` — wrap external/user content in clearly-bounded
  markers so the LLM treats it as data, not instructions. Plus the
  policy preamble that tells the model what those markers mean.
* ``scrub`` — redact secret-shaped fields from response payloads
  before they leave the trust boundary on a path the auth middleware
  can't (or doesn't) gate.

See ``docs/security_model.md`` for the trust boundary, the named
untrusted surfaces, and the current known gaps.
"""

from augmentum.security.scrub import (
    REDACTED,
    is_secret_key,
    scrub_dict,
    scrub_response,
)
from augmentum.security.untrusted import (
    UNTRUSTED_CONTEXT_POLICY,
    ensure_policy_in_system,
    wrap_untrusted,
)

__all__ = [
    "REDACTED",
    "UNTRUSTED_CONTEXT_POLICY",
    "ensure_policy_in_system",
    "is_secret_key",
    "scrub_dict",
    "scrub_response",
    "wrap_untrusted",
]
