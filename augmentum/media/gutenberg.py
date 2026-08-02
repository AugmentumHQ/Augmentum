"""Project Gutenberg plaintext fetcher.

LibriVox recordings are readings of Project Gutenberg public-domain texts.
Every LibriVox book carries a ``url_text_source`` field pointing back to
its Gutenberg source page. This module turns that URL into clean plaintext
we can store alongside the audio for read-along.

Responsibilities:
  * Parse a LibriVox ``url_text_source`` into a Gutenberg eBook id.
  * Download the plaintext (with sensible fallbacks across the several
    URL shapes Gutenberg serves).
  * Strip the boilerplate header/footer so only the actual book body
    remains.

Deliberately does *not* touch the filesystem, the file_index, or the
job system. The fetch job handler in ``augmentum/jobs/handlers/`` is
the integration point.
"""

from __future__ import annotations

import re

import httpx

# LibriVox's ``url_text_source`` field points at any of:
#   https://www.gutenberg.org/ebooks/12345
#   http://www.gutenberg.org/etext/12345
#   https://www.gutenberg.org/files/12345/12345-h/12345-h.htm
#   https://www.gutenberg.org/cache/epub/12345/pg12345.txt
# The integer after the last-matching segment is always the eBook id.
_EBOOK_ID_RE = re.compile(
    r"gutenberg\.org/(?:ebooks|etext|files|cache/epub)/(\d+)",
    re.IGNORECASE,
)

# Gutenberg wraps every plaintext with a marker pair like:
#   *** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***
#   <book body>
#   *** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***
# Older files use "*** START OF THIS PROJECT GUTENBERG ..." and other
# trivial variants. Match loosely on the common stem.
_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


class GutenbergError(Exception):
    """Raised when a fetch cannot be resolved to usable plaintext."""


def resolve_ebook_id(url_text_source: str) -> str:
    """Pull the numeric eBook id out of a Gutenberg URL.

    Returns the id as a string (preserving leading zeros is irrelevant —
    Gutenberg ids are dense integers). Raises ``GutenbergError`` when
    the URL doesn't look like Gutenberg or doesn't carry an id.
    """
    if not url_text_source:
        raise GutenbergError("no url_text_source provided")
    match = _EBOOK_ID_RE.search(url_text_source)
    if not match:
        raise GutenbergError(f"not a Gutenberg URL: {url_text_source}")
    return match.group(1)


def _candidate_urls(ebook_id: str) -> list[str]:
    """Ordered list of plaintext URLs to try for an eBook id.

    Gutenberg doesn't guarantee any single shape works for every book —
    older entries live under ``/files/``, newer ones under
    ``/cache/epub/``. Try the UTF-8 cache URL first (most modern, no
    encoding surprises), then the plain ``-0.txt`` and bare ``.txt``
    variants under ``/files/``.
    """
    return [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]


async def fetch_plaintext(
    http_client: httpx.AsyncClient,
    ebook_id: str,
    *,
    timeout_s: float = 30.0,
) -> str:
    """Download the plaintext for an eBook id, trying fallback URLs.

    Returns the raw body as a decoded string. Raises ``GutenbergError``
    when every candidate URL fails. A non-200 response is treated as a
    miss, not a hard error — we silently try the next candidate.

    The first 404/403 for the ``-0.txt`` variant is common and expected;
    suppress at debug level only (we don't want warn-spam for the
    happy-path fallback).
    """
    last_err: str = ""
    for url in _candidate_urls(ebook_id):
        try:
            resp = await http_client.get(url, timeout=timeout_s, follow_redirects=True)
        except httpx.HTTPError as exc:
            last_err = f"{url}: {exc!r}"
            continue
        if resp.status_code != 200:
            last_err = f"{url}: HTTP {resp.status_code}"
            continue
        # Gutenberg declares encoding in a charset= header; httpx's
        # resp.text honours it. Fall back to utf-8 with errors='replace'
        # when the header is missing so we never crash on a stray byte.
        try:
            return resp.text
        except UnicodeDecodeError:
            return resp.content.decode("utf-8", errors="replace")
    raise GutenbergError(
        f"no plaintext URL reachable for eBook {ebook_id} (last: {last_err})",
    )


def strip_boilerplate(raw: str) -> str:
    """Trim the Gutenberg header + license footer from a plaintext body.

    Returns only the book content between the START/END markers. If
    either marker is missing (rare but possible with pre-2006 files),
    returns the input unchanged — we'd rather ship slightly-noisy text
    than empty-string the book.
    """
    if not raw:
        return raw
    start_match = _START_RE.search(raw)
    end_match = _END_RE.search(raw)
    if start_match and end_match and end_match.start() > start_match.end():
        body = raw[start_match.end():end_match.start()]
    elif start_match:
        body = raw[start_match.end():]
    elif end_match:
        body = raw[:end_match.start()]
    else:
        body = raw
    # Gutenberg plaintext uses CRLF; normalise to LF so downstream
    # word-count and rendering are platform-independent.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    return body.strip() + "\n"
