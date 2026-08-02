"""PII scrubbing for memory ingest.

Redacts common secret / personally-identifying patterns from content
before it lands in the memory store. Runs pre-embedding so the vector
keys on the scrubbed form (otherwise two scrubbings of the same
email-containing fact would have different embeddings and dedup would
miss them).

Generic across user domains — keys on regex-detectable patterns
(API key shapes, email/IP RFC formats, JWT structure), not on any
curated word list. A user whose work is "I write security audits at
firm@example.com" will see the email redacted; that's the right
default for a memory store the LLM will reference frequently. The
user can disable globally via ``memory_pii_scrub_enabled = False``
or force-save specific PII with the explicit "remember X" path,
which is intentionally exempt from the scrub.

Tradeoff intentionally not handled:
- Two distinct emails scrub to the same ``[email]`` token. Text-dedup
  on the scrubbed form will merge them. For a memory store this is
  acceptable; the value of an email is the *fact* the user has one,
  not the literal address. If a user wants to differentiate, they
  use explicit phrasing which bypasses the scrub.
"""
from __future__ import annotations

import re

# Each pattern: (regex, replacement). Replacements use bracketed labels
# rather than fixed-length redactions so the model can still tell what
# *kind* of PII was here. Ordering matters: JWT must come before generic
# "long base64" patterns; API keys must come before generic alphanumeric.
_PII_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # JWT: three base64 segments separated by dots, opening "eyJ" header.
    # Match before generic API-key patterns since JWTs can look like keys.
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "[jwt]"),
    # Anthropic API key: sk-ant-...
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "[api_key]"),
    # OpenAI-style API key: sk-... (broad, must come after sk-ant-)
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[api_key]"),
    # GitHub tokens: ghp_, gho_, ghu_, ghs_, ghr_, github_pat_
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b"), "[api_key]"),
    # AWS access key ID: AKIA + 16 alphanumeric
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[api_key]"),
    # Google API key: AIza + 30-50 alphanumeric/-/_ (real keys = 35,
    # but tolerate ±5 so near-misses still get scrubbed)
    (re.compile(r"\bAIza[A-Za-z0-9_-]{30,50}\b"), "[api_key]"),
    # Email — RFC-ish, conservative on tld length to avoid matching
    # "foo@bar." prefixes mid-sentence.
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"), "[email]"),
    # IPv4 — strict octet range to skip version strings like "1.2.3.4"
    # is the same regex shape, but most version strings are 3-segment
    # not 4-segment. False positives here are acceptable: a literal IP
    # in a memory is rarely useful and the user can disable the scrub.
    (re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ), "[ip]"),
)


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Redact PII patterns. Returns ``(scrubbed_text, kinds_redacted)``.

    ``kinds_redacted`` is the unique set of labels that fired (one of
    ``"api_key"``, ``"email"``, ``"ip"``, ``"jwt"``). Empty list means
    nothing matched and ``scrubbed_text == text``. Order in the list
    reflects detection order (mostly stable but not load-bearing).

    The function is pure (no I/O, no logging) so the store can decide
    whether to log a scrub event based on its own context.
    """
    if not text:
        return text, []
    out = text
    fired: list[str] = []
    seen: set[str] = set()
    for pattern, replacement in _PII_PATTERNS:
        new_out, n = pattern.subn(replacement, out)
        if n > 0:
            kind = replacement.strip("[]")
            if kind not in seen:
                seen.add(kind)
                fired.append(kind)
            out = new_out
    return out, fired
