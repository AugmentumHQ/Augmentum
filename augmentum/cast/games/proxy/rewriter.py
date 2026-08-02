"""HTML/CSS URL rewriter + adapter loader injection.

The cross-origin gap closes here: every reference in the proxied
HTML/CSS that targets the source origin is rewritten to ride through
``/api/cast/game-proxy/{token}/...`` so the browser fetches it from
*our* origin. The universal input adapter loader is injected at the
top of ``<head>`` so the gamepad/keyboard/touch/pointer chain reaches
the inner realm before any game code runs.

URL handling falls into three buckets:

  - **same-origin** (relative + absolute matching source_origin) →
    rewritten to the proxy path.
  - **cross-origin allowlist** (cdnjs, jsdelivr, unpkg, fonts.gstatic,
    raw.githubusercontent.com) → passed through unchanged; CSP is
    rewritten to permit them.
  - **everything else** → left untouched but logged. The browser's
    CSP will likely block them unless the game also imports from a
    known CDN.

JS is NOT rewritten — too brittle, too many false positives, and
modern bundlers' dynamic imports usually reach into the same-origin
asset paths anyway (which our HTML rewrite already covered via
``<script src=...>`` entries).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urljoin, urlparse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Cross-origin asset CDNs we permit pass-through. Each is a host
# suffix match (so subdomains under each are allowed too).
DEFAULT_CDN_ALLOWLIST: frozenset[str] = frozenset({
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "raw.githubusercontent.com",
    "github.io",
})


# HTML attributes per tag that carry URLs we should rewrite. Only the
# top hits — anything not in here passes through untouched.
_URL_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href"}),
    "area": frozenset({"href"}),
    "base": frozenset({"href"}),
    "link": frozenset({"href"}),
    "img": frozenset({"src", "srcset"}),
    "image": frozenset({"href", "xlink:href"}),
    "input": frozenset({"src", "formaction"}),
    "script": frozenset({"src"}),
    "iframe": frozenset({"src"}),
    "frame": frozenset({"src"}),
    "embed": frozenset({"src"}),
    "object": frozenset({"data"}),
    "source": frozenset({"src", "srcset"}),
    "track": frozenset({"src"}),
    "video": frozenset({"src", "poster"}),
    "audio": frozenset({"src"}),
    "form": frozenset({"action"}),
    "use": frozenset({"href", "xlink:href"}),
}


# CSS ``url(...)`` matcher — captures the inner URL, tolerating
# optional quotes + whitespace.
_CSS_URL_RE = re.compile(
    r"""url\(\s*(?:(['"])(?P<q>[^'"]*?)\1|(?P<u>[^)\s'"]+))\s*\)""",
    re.IGNORECASE,
)


# Where to splice the adapter loader. We look for the first ``<head>``
# tag (with or without attributes) and inject immediately after.
# Falls back to ``<html>`` if no head exists; absolute last resort is
# prepending to the body.
_HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)
_HTML_OPEN_RE = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_BODY_OPEN_RE = re.compile(r"<body\b[^>]*>", re.IGNORECASE)


# Boot script injected on every proxied page. Loads the loader as a
# module so its imports resolve from our origin. The service-worker
# override happens inline so it lands BEFORE any page script can
# register one.
_BOOT_SCRIPT = """\
<script>
(function() {
  // Service workers register on the SOURCE origin's scope, not ours
  // — they bypass the proxy entirely and break the adapter
  // injection. Override the API to a no-op so games' "best-effort
  // sw register" calls don't break + don't escape.
  try {
    if (navigator.serviceWorker && typeof navigator.serviceWorker.register === 'function') {
      const _origRegister = navigator.serviceWorker.register.bind(navigator.serviceWorker);
      navigator.serviceWorker.register = function() {
        try { console.warn('[cast-proxy] serviceWorker.register suppressed'); } catch (_) {}
        // Return a rejected-but-quiet promise so the game's .catch handler runs.
        return Promise.reject(new Error('serviceWorker disabled by cast proxy'));
      };
      // Keep the original handle on a private name in case Phase 4's
      // probe wants to inspect it without re-installing.
      navigator.serviceWorker._augmentumOriginalRegister = _origRegister;
    }
  } catch (_) {}
})();
</script>
<script type="module" src="__ADAPTER_LOADER_SRC__"></script>
"""


# ── helpers ──────────────────────────────────────────────────────


URLRewriter = Callable[[str], str]


def _is_data_or_anchor(url: str) -> bool:
    if not url:
        return True
    u = url.strip()
    if not u:
        return True
    return u.startswith(("data:", "blob:", "javascript:", "mailto:", "tel:", "#"))


def _host_matches_allowlist(host: str, allowlist: frozenset[str]) -> bool:
    if not host:
        return False
    host = host.lower()
    for allow in allowlist:
        if host == allow or host.endswith("." + allow):
            return True
    return False


def make_url_rewriter(
    *,
    proxy_base: str,
    source_origin: str,
    page_url: str,
    cdn_allowlist: frozenset[str] = DEFAULT_CDN_ALLOWLIST,
) -> URLRewriter:
    """Build a per-page rewriter callable.

    Args:
      proxy_base:    e.g. ``/api/cast/game-proxy/<token>``
      source_origin: e.g. ``https://example.com``
      page_url:      the absolute URL of the page being rewritten
                     (resolves relative references inside it).
      cdn_allowlist: cross-origin hosts whose assets pass through.
    """
    proxy_base = proxy_base.rstrip("/")
    source_origin = source_origin.rstrip("/")
    src_parsed = urlparse(source_origin)
    page_url = page_url or source_origin

    def _rewrite(raw: str) -> str:
        if _is_data_or_anchor(raw):
            return raw

        # Resolve to absolute against the page URL.
        try:
            absolute = urljoin(page_url, raw)
            parsed = urlparse(absolute)
        except ValueError:
            return raw

        if parsed.scheme not in ("http", "https"):
            return raw

        # Same-origin → proxy. We preserve path + query so deep links
        # round-trip correctly.
        if (
            parsed.scheme.lower() == src_parsed.scheme.lower()
            and (parsed.hostname or "").lower() == (src_parsed.hostname or "").lower()
            and parsed.port == src_parsed.port
        ):
            tail = parsed.path or "/"
            if parsed.query:
                tail = f"{tail}?{parsed.query}"
            if parsed.fragment:
                tail = f"{tail}#{parsed.fragment}"
            # Strip any leading slash on the tail so the join is clean.
            return f"{proxy_base}{tail if tail.startswith('/') else '/' + tail}"

        # Cross-origin CDN → pass through. CSP will be rewritten to
        # whitelist these.
        if _host_matches_allowlist(parsed.hostname or "", cdn_allowlist):
            return absolute

        # Else: leave as-is + log. The browser's CSP usually blocks
        # this; we surface a record so the quirks table can grow.
        log.info(
            "cast_proxy_unrewritten_cross_origin_url",
            page=page_url, url=absolute,
        )
        return raw

    return _rewrite


def _rewrite_srcset(value: str, rewrite: URLRewriter) -> str:
    """``srcset`` is a comma-separated list of ``url descriptor`` pairs.
    Rewrite each url; preserve descriptors verbatim."""
    out: list[str] = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        bits = entry.split(None, 1)
        url = bits[0]
        descriptor = bits[1] if len(bits) > 1 else ""
        rewritten = rewrite(url)
        out.append(rewritten + (f" {descriptor}" if descriptor else ""))
    return ", ".join(out)


# ── HTML rewriter ────────────────────────────────────────────────


class _RewritingParser(HTMLParser):
    """html.parser subclass that emits a rewritten HTML stream."""

    def __init__(self, rewrite: URLRewriter) -> None:
        super().__init__(convert_charrefs=False)
        self._rewrite = rewrite
        self._chunks: list[str] = []

    # We need raw text + tags preserved (the default HTMLParser
    # _doesn't_ give us the original tag text, so we reconstruct).

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._chunks.append(self._render_tag(tag, attrs, self_close=False))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._chunks.append(self._render_tag(tag, attrs, self_close=True))

    def handle_endtag(self, tag: str) -> None:
        self._chunks.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        self._chunks.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._chunks.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._chunks.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._chunks.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self._chunks.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        self._chunks.append(f"<![{data}]>")

    def _render_tag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        self_close: bool,
    ) -> str:
        url_attrs = _URL_ATTRS.get(tag.lower(), frozenset())
        parts: list[str] = [f"<{tag}"]
        for name, value in attrs:
            if value is None:
                parts.append(f" {name}")
                continue
            attr_lower = name.lower()
            if attr_lower in url_attrs:
                if attr_lower == "srcset":
                    value = _rewrite_srcset(value, self._rewrite)
                else:
                    value = self._rewrite(value)
            parts.append(f' {name}="{_escape_attr(value)}"')
        parts.append(" />" if self_close else ">")
        return "".join(parts)

    def output(self) -> str:
        return "".join(self._chunks)


def _escape_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
             .replace('"', "&quot;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
    )


def rewrite_html(
    html: str,
    *,
    proxy_base: str,
    source_origin: str,
    page_url: str,
    cdn_allowlist: frozenset[str] = DEFAULT_CDN_ALLOWLIST,
) -> str:
    """Rewrite every URL-bearing attribute in ``html`` through the
    proxy. Returns the rewritten HTML."""
    rewrite = make_url_rewriter(
        proxy_base=proxy_base,
        source_origin=source_origin,
        page_url=page_url,
        cdn_allowlist=cdn_allowlist,
    )
    parser = _RewritingParser(rewrite)
    parser.feed(html)
    parser.close()
    return parser.output()


# ── CSS rewriter ─────────────────────────────────────────────────


def rewrite_css(
    css: str,
    *,
    proxy_base: str,
    source_origin: str,
    page_url: str,
    cdn_allowlist: frozenset[str] = DEFAULT_CDN_ALLOWLIST,
) -> str:
    """Rewrite ``url(...)`` references inside a CSS payload."""
    rewrite = make_url_rewriter(
        proxy_base=proxy_base,
        source_origin=source_origin,
        page_url=page_url,
        cdn_allowlist=cdn_allowlist,
    )

    def _sub(match: re.Match) -> str:
        quoted = match.group("q") if match.group(1) else None
        raw_url = quoted if quoted is not None else match.group("u")
        rewritten = rewrite(raw_url or "")
        if match.group(1):
            quote = match.group(1)
            return f"url({quote}{rewritten}{quote})"
        return f"url({rewritten})"

    return _CSS_URL_RE.sub(_sub, css)


# ── Adapter loader injection ─────────────────────────────────────


def inject_adapter_loader(
    html: str,
    *,
    adapter_loader_src: str = "/ui/scripts/cast-input/universal-input-adapter.js",
) -> str:
    """Inject the universal input adapter loader (plus the SW-disable
    shim) at the top of ``<head>``. Falls back to right after ``<html>``
    or ``<body>`` when no head exists.

    Idempotent — calling twice doesn't duplicate the script tag.
    """
    if "__augmentum_cast_loader_marker__" in html:
        return html

    boot = _BOOT_SCRIPT.replace("__ADAPTER_LOADER_SRC__", adapter_loader_src)
    boot = boot.replace(
        '<script type="module" src=',
        '<script data-marker="__augmentum_cast_loader_marker__" type="module" src=',
    )

    for pattern in (_HEAD_OPEN_RE, _HTML_OPEN_RE, _BODY_OPEN_RE):
        m = pattern.search(html)
        if m is not None:
            insert_at = m.end()
            return html[:insert_at] + boot + html[insert_at:]
    # Worst case: prepend.
    return boot + html


# ── CSP rewriting ────────────────────────────────────────────────


def rewrite_csp(
    csp_header: str,
    *,
    proxy_base: str,
    cdn_allowlist: frozenset[str] = DEFAULT_CDN_ALLOWLIST,
) -> str:
    """Rewrite a CSP header so the proxied page allows:
      - 'self' (our origin)
      - the proxy-base path (script-src / style-src)
      - CDN allowlist entries (script-src / style-src / font-src)
      - data: and blob: for inline images / fonts
      - 'unsafe-inline' for the adapter loader injection

    This is intentionally permissive — the source's CSP was written
    against its own origin + dependencies; we relax it so our adapter
    + CDN pass-throughs work, but we KEEP the directives that block
    arbitrary external script/connect targets.
    """
    if not csp_header:
        return ""

    # Split on `;` and re-emit each directive with our allowances merged.
    cdn_hosts = " ".join(f"https://{host}" for host in sorted(cdn_allowlist))
    additions = {
        "script-src":  f"'self' 'unsafe-inline' 'unsafe-eval' {cdn_hosts}",
        "style-src":   f"'self' 'unsafe-inline' {cdn_hosts}",
        "font-src":    f"'self' data: {cdn_hosts}",
        "img-src":     f"'self' data: blob: {cdn_hosts}",
        "connect-src": f"'self' {cdn_hosts}",
        "frame-src":   "'self'",
        "frame-ancestors": "'self'",
    }
    out: list[str] = []
    seen: set[str] = set()
    for directive in csp_header.split(";"):
        d = directive.strip()
        if not d:
            continue
        key = d.split(None, 1)[0].lower()
        seen.add(key)
        if key in additions:
            out.append(f"{key} {additions[key]}")
        else:
            out.append(d)
    for key, value in additions.items():
        if key not in seen:
            out.append(f"{key} {value}")
    return "; ".join(out)
