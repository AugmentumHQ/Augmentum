"""Static-vs-dynamic preview classifier.

A coder preview can be either:

* **Static** — the served files are the artifact. ``python -m http.server``,
  ``npx serve``, ``vite preview``, ``caddy file-server``. Save = snapshot
  the served dir.
* **Dynamic** — content depends on a running process. ``uvicorn``,
  ``node server.js`` with API routes, anything stateful. Save = won't
  work in v1; needs the v2 ephemeral-container launch path.

The classifier looks at the served URL's response and returns one of:

* ``"static"`` — content looks fully self-contained
* ``"dynamic"`` — content references API / WebSocket / EventSource
* ``"unknown"`` — couldn't fetch, non-HTML response, or insufficient signal

Used by the save-preflight route to gate the save button. False
positives (a static-looking page that actually polls an API) get caught
at launch when API calls 404 — acceptable v1 trade-off per the spec.
"""

from __future__ import annotations

import re
from typing import Literal

PreviewKind = Literal["static", "dynamic", "unknown"]


# Patterns matched against the response HTML/text. These are intentionally
# loose — a single hit is enough to flip "dynamic". We accept some false
# positives (a static page that mentions "fetch" in a comment) in exchange
# for catching the obvious cases.
_DYNAMIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfetch\s*\(", re.IGNORECASE),
    re.compile(r"\bXMLHttpRequest\b"),
    re.compile(r"\bnew\s+WebSocket\s*\(", re.IGNORECASE),
    re.compile(r"\bnew\s+EventSource\s*\(", re.IGNORECASE),
    # Same-origin API-shaped paths. Match `/api/` or `'/api/'` as a
    # substring; covers most REST conventions.
    re.compile(r"['\"`]/(api|graphql|ws|sse)/", re.IGNORECASE),
    re.compile(r"\baxios\.(?:get|post|put|delete|patch)\b", re.IGNORECASE),
)

# Extensions that we'll never treat as "static playable" even if the
# response is HTML-ish. Save would just confuse users.
_NON_PLAYABLE_CONTENT_TYPES: frozenset[str] = frozenset({
    "application/json",
    "application/xml",
    "text/xml",
    "text/plain",
    "text/csv",
})


def classify_response(
    *,
    status_code: int,
    content_type: str,
    body: str | bytes,
) -> PreviewKind:
    """Pure classifier — given a single response, return the kind.

    ``body`` is decoded as UTF-8 (errors replaced) if bytes. Only the
    first ~64KB are inspected; large SPA bundles won't be fully scanned
    but the dynamic markers are typically near the top of the HTML or
    inline in early scripts. The boundary is a guardrail against
    pathological inputs.
    """
    if status_code < 200 or status_code >= 400:
        return "unknown"

    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if ct in _NON_PLAYABLE_CONTENT_TYPES:
        return "unknown"

    # Non-HTML / non-JS response = no signal we can use. Static binary
    # assets (images etc.) aren't a "preview" target either.
    if ct and not (
        ct.startswith("text/")
        or ct in {"application/xhtml+xml", "application/javascript", "application/ecmascript"}
    ):
        return "unknown"

    if isinstance(body, bytes):
        text = body[:65536].decode("utf-8", errors="replace")
    else:
        text = body[:65536]

    if not text.strip():
        # Empty body — can't tell. Probably an empty static page.
        return "unknown"

    for pattern in _DYNAMIC_PATTERNS:
        if pattern.search(text):
            return "dynamic"

    # No dynamic indicators + HTML-ish body = call it static. Even
    # plain JS files get this treatment since the coder preview is
    # almost always an HTML entry point.
    return "static"


async def probe_preview(primary_url: str, *, timeout_seconds: float = 5.0) -> PreviewKind:
    """Async version that fetches ``primary_url`` and classifies.

    Returns ``"unknown"`` on any network failure or non-OK status. The
    fetch uses an internal aiohttp / httpx client so the route doesn't
    have to wire one up; pick whichever the project already depends on.

    Kept separate from :func:`classify_response` so unit tests can
    feed canned bodies without needing to spin up a server.

    Failures log at INFO so dogfooders can diagnose unknown-preview
    classifications (most common cause: the preview is reachable from
    the browser but not from this process's network namespace).
    """
    # Imports kept lazy so the pure function is usable in test
    # environments without an HTTP client dep installed and so the
    # logger import doesn't trigger at module load.
    import httpx  # already a project dep

    from augmentum.utils.logging import get_logger
    log = get_logger(__name__)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.get(primary_url)
            kind = classify_response(
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                body=resp.content,
            )
            log.info(
                "library_preview_probed",
                url=primary_url,
                status=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                body_bytes=len(resp.content or b""),
                kind=kind,
            )
            return kind
    except (httpx.HTTPError, OSError) as exc:
        log.info(
            "library_preview_probe_failed",
            url=primary_url,
            error=type(exc).__name__,
            detail=str(exc)[:200],
        )
        return "unknown"
