"""API endpoints for knowledge pack management."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.proxy import system_events
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpError, check_ssrf
from augmentum.utils.streamed_download import streamed_download

log = get_logger(__name__)

# Rewrite relative ZIM links so iframe navigation stays inside our route
# instead of escaping to the live web. Matches href= and src= attributes
# whose value is NOT already absolute (http/https), protocol-relative (//),
# data:, anchor (#), or root-relative (/). The capture preserves the full
# original path including ZIM namespace prefixes (A/, I/, M/, etc.) which
# the route handler resolves via libzim.get_entry_by_path.
_ZIM_LINK_RE = re.compile(
    r'(href|src)\s*=\s*"(?!https?://|//|data:|#|/|mailto:|javascript:)([^"]+)"',
    re.IGNORECASE,
)
# Same shape, single-quoted variant — MediaWiki-derived ZIMs usually
# emit double quotes but some templates and edge entries use singles.
_ZIM_LINK_RE_SQ = re.compile(
    r"(href|src)\s*=\s*'(?!https?://|//|data:|#|/|mailto:|javascript:)([^']+)'",
    re.IGNORECASE,
)
# Match <a href="https://...">, <a href='https://...'>. We match anchors
# whose href is absolute (http or https) and prepend target="_blank" plus
# rel="noopener noreferrer" so external links open in a new browser tab
# instead of trying to navigate the iframe (which Augmentum's parent CSP
# blocks for non-allowlisted origins, leaving the user staring at a
# chrome-error://chromewebdata/ frame).
_EXTERNAL_ANCHOR_RE = re.compile(
    r'(<a\b[^>]*\bhref\s*=\s*["\']https?://[^"\']*["\'][^>]*?)>',
    re.IGNORECASE,
)
# Strip ``target="_blank"`` from anchors that point back at our own
# ZIM-serving route. Mirrors the kiwix-tools #591/#678 fix: many ZIMs
# (Wikipedia portals, FANDOM cross-wiki links, Stack Exchange "more"
# nav) embed ``target="_blank"`` on internal links. Without this strip,
# a click opens a bare ``/api/knowledge/zim/{pack}/...`` URL in a new
# tab — outside the browse panel, no chrome, no back-stack. Internal
# navigation should stay inside the iframe; external anchors keep their
# ``target="_blank"`` (added by ``_EXTERNAL_ANCHOR_RE`` above).
#
# The lookahead anchors the rule to "anchors with an internal href"
# regardless of attribute order: ``<a target="_blank" href="…">`` and
# ``<a href="…" target="_blank">`` both match. Run AFTER the link
# rewriter so the href has already been normalized to the route path.
_INTERNAL_TARGET_BLANK_RE = re.compile(
    r'(<a\b(?=[^>]*\bhref\s*=\s*["\']/api/knowledge/zim/)[^>]*?)'
    r'\s+target\s*=\s*["\']_blank["\']'
    r'([^>]*>)',
    re.IGNORECASE,
)
# Strip any pre-existing ``<base>`` element from the article HTML.
# mwoffliner / DevDocs scrapes routinely ship a relative-URL base
# (``<base href="../../">``) that's intended for the original site
# layout. Chrome appears to honor ANY base in the document over our
# injected one — likely because the article's base sits later in the
# parsed tree and some browser builds ignore the "first base wins"
# spec rule under multi-base conditions. Without this strip, asset
# URLs (scripts, stylesheets, images) resolve against the wrong base
# and the network panel fills with ``/api/knowledge/zim/{wrong_pack}/...``
# 404s that surface as MIME-mismatch errors (our 404 returns JSON, the
# browser was expecting JS or CSS). Self-closing variant rare but legal.
_EXISTING_BASE_RE = re.compile(r"<base\b[^>]*/?>", re.IGNORECASE)
# Rewrite ``srcset`` values so retina/responsive image candidates
# resolve through our route. The injected ``<base>`` handles this in
# theory (per the HTML spec, srcset URLs are base-relative), but
# Safari has had historical quirks parsing srcset against a remote
# base in iframes — explicit rewriting eliminates the dependency.
# A srcset value is a comma-separated list of "URL descriptor" pairs
# (e.g. ``foo.png 1x, bar.png 2x`` or ``a.png 320w, b.png 640w``).
# Negative lookbehind ``(?<![-\w])`` prevents matching ``data-srcset``
# (``-`` is non-word, so ``\b`` would mistakenly match between them).
# We want to rewrite the real ``srcset`` attribute only — lazy-load
# attributes like ``data-srcset`` get swapped into ``srcset`` at runtime
# by JS bundles, and our rewriter handles them there.
_SRCSET_RE = re.compile(r'(?<![-\w])srcset\s*=\s*"([^"]*)"', re.IGNORECASE)
_SRCSET_RE_SQ = re.compile(r"(?<![-\w])srcset\s*=\s*'([^']*)'", re.IGNORECASE)
# Pre-compiled prefix recognizer for srcset URLs — anything matching
# is already absolute / protocol-relative / inert and shouldn't be
# rewritten. Same predicate used by ``_ZIM_LINK_RE``'s negative
# lookahead, lifted to a function-friendly form.
_SRCSET_ABSOLUTE_PREFIX_RE = re.compile(
    r"^(?:https?://|//|data:|#|/|mailto:|javascript:)",
    re.IGNORECASE,
)


def _rewrite_srcset_value(value: str, pack_id: str) -> str:
    """Rewrite each candidate URL in a ``srcset`` value through our
    route. Descriptors (``2x`` / ``320w``) pass through verbatim;
    absolute / data / fragment URLs pass through unchanged.

    Robust to:
        * Empty/whitespace candidates (skip).
        * Missing descriptors (URL only — valid per the spec).
        * Comma + space oddity (``foo.png  ,  bar.png 2x``) — split
          and re-emit with normalized ", " separator.
    """
    out_parts: list[str] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        bits = chunk.split(None, 1)
        url = bits[0]
        rest = bits[1] if len(bits) > 1 else ""
        if url and not _SRCSET_ABSOLUTE_PREFIX_RE.match(url):
            url = f"/api/knowledge/zim/{pack_id}/{url}"
        out_parts.append(f"{url} {rest}".rstrip())
    return ", ".join(out_parts)
# Nested <iframe> blocks inside ZIM articles — common for embedded NCBI
# refs, OWID charts, YouTube clips. The parent app's CSP frame-src
# allowlist only covers a few media providers, so these frames get
# rejected with a chrome-error and pollute the console. Replace with a
# small placeholder that links out to the original.
_NESTED_IFRAME_RE = re.compile(
    r'<iframe\b([^>]*)>.*?</iframe>',
    re.DOTALL | re.IGNORECASE,
)
_IFRAME_SRC_RE = re.compile(r'src\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

# Inline-style neutralizer (Layer 3). Strips hostile ``color`` /
# ``background`` / ``background-color`` declarations from ``style=""``
# attributes so source HTML with hardcoded original-site palettes (e.g.
# ``style="color:#222; background:#fff"``) can't fight the reader theme.
#
# Conservative on purpose:
#   - Only the three color-bearing properties are removed. Layout
#     properties (``margin``, ``padding``, ``width``, ``display``,
#     ``float``, etc.) pass through untouched.
#   - Removal is whole-declaration: ``color: rgb(34, 34, 34) !important``
#     is stripped along with its trailing semicolon.
#   - If the resulting ``style="…"`` is empty, the whole attribute is
#     dropped (avoid leaving empty attributes that some validators trip
#     on).
#
# Why not just blanket-override at CSS level: a body-wide
# ``* { color: var(--augz-text) !important }`` would crush all semantic
# accent colors (red for errors, green for diffs, syntax highlighting).
# Stripping inline only — where the original site author hard-coded a
# specific value for the site's palette — leaves legitimate semantic
# colors that ride on CSS classes intact.
_STYLE_ATTR_RE = re.compile(
    r'(\s)style\s*=\s*(["\'])([^"\']*)\2',
    re.IGNORECASE,
)
_NEUTRALIZE_DECL_RE = re.compile(
    # property : value (anything up to the next ``;`` or end of string).
    # Strips:
    #   - background / background-color / color — already-known palette clash
    #   - box-shadow / text-shadow — decorative, almost always clashes with
    #     reader themes (white glows on dark backgrounds, gray drop-shadows
    #     that float disconnected from the surface)
    #   - filter — same; ``filter: drop-shadow(...)`` or ``filter: brightness()``
    #     applied inline assumes the source-site palette
    # Includes ``-webkit-`` prefixed variants — older MediaWiki templates
    # and FANDOM mirrors emit them.
    r'(?:^|;)\s*(?:'
    r'background-color|background|color|'
    r'box-shadow|-webkit-box-shadow|'
    r'text-shadow|'
    r'filter|-webkit-filter'
    r')\s*:[^;]*',
    re.IGNORECASE,
)
# Border-color longhand → ``var(--augz-border)``. Preserves the structural
# property so infoboxes / navboxes / citation cards keep their frame, but
# the frame color tracks the theme. Shorthand ``border: 1px solid #ccc``
# is intentionally left intact — width and style ARE structural info, and
# parsing shorthand to surgically replace just the color token is brittle
# (varies by token order, named colors, multi-keyword shadows). A border
# whose color is half a stop off-theme is readable; the absent frame
# version (full shorthand strip) loses MediaWiki's structural cues.
_BORDER_COLOR_REMAP_RE = re.compile(
    r'(border(?:-(?:top|right|bottom|left))?-color)\s*:\s*[^;]*',
    re.IGNORECASE,
)


# Photo opt-out tagger for image-invert. On dark/midnight themes the
# baseline CSS inverts non-photo images so diagrams / math / schematic
# SVGs (which assume a white background) don't glare against the dark
# page. Photos should NOT be inverted — they look surreal. Tag them
# server-side with ``data-aug-no-invert`` so the CSS selector skips them.
#
# Heuristics (cheap, conservative — false positives are fine because
# they just leave a photo uninverted, which is the safe default):
#   - File extension in {jpg, jpeg, webp}. PNGs / GIFs / SVGs in
#     Wikipedia are almost always diagrams or icons, not photos.
#   - ``alt`` attribute containing "photograph" or "photo" as a token.
#     MediaWiki's File:Photograph_of_X templates emit these alts.
#
# Idempotent: skip if the attribute is already present (handles the rare
# case where both regexes would tag the same tag).
_PHOTO_IMG_SRC_RE = re.compile(
    r'(<img\b[^>]*?\bsrc\s*=\s*["\'][^"\']*?\.(?:jpe?g|webp)(?:[?#][^"\']*)?["\'][^>]*?)(\s*/?>)',
    re.IGNORECASE,
)
_PHOTO_IMG_ALT_RE = re.compile(
    r'(<img\b[^>]*?\balt\s*=\s*["\'][^"\']*?\b(?:photograph|photo)\b[^"\']*?["\'][^>]*?)(\s*/?>)',
    re.IGNORECASE,
)


def _tag_photo_images(html: str) -> str:
    """Append ``data-aug-no-invert`` to ``<img>`` tags whose src or alt
    indicates a photograph rather than a diagram.

    See ``_PHOTO_IMG_SRC_RE`` / ``_PHOTO_IMG_ALT_RE`` for the heuristic
    rationale. Idempotent — running twice has no additional effect.
    """
    def _tag(m: re.Match) -> str:
        attrs, closer = m.group(1), m.group(2)
        if "data-aug-no-invert" in attrs.lower():
            return m.group(0)
        return f'{attrs} data-aug-no-invert=""{closer}'

    html = _PHOTO_IMG_SRC_RE.sub(_tag, html)
    html = _PHOTO_IMG_ALT_RE.sub(_tag, html)
    return html


def _neutralize_inline_styles(html: str) -> str:
    """Strip palette-clashing declarations from inline ``style=""`` attrs,
    and remap ``border-color`` longhand to the theme variable.

    See module-level ``_STYLE_ATTR_RE`` / ``_NEUTRALIZE_DECL_RE`` comments
    for the rationale. Idempotent — running twice produces the same
    output as running once (already-clean styles match the prefix regex
    but the inner sub finds nothing to remove; already-remapped
    ``border-color: var(--augz-border)`` matches the remap pattern but
    the sub yields the same string).
    """

    def _scrub(m: re.Match) -> str:
        leading_ws = m.group(1)
        quote = m.group(2)
        decls = m.group(3)
        cleaned = _NEUTRALIZE_DECL_RE.sub("", decls)
        cleaned = _BORDER_COLOR_REMAP_RE.sub(
            lambda bm: f"{bm.group(1)}: var(--augz-border)",
            cleaned,
        )
        # Trim stray leading/trailing whitespace + semicolons after removal.
        cleaned = cleaned.strip().strip(";").strip()
        if not cleaned:
            # Drop the empty attribute entirely along with its leading ws.
            return ""
        return f"{leading_ws}style={quote}{cleaned}{quote}"

    return _STYLE_ATTR_RE.sub(_scrub, html)

# CSS ``url(...)`` rewriter for stylesheets served from inside the ZIM.
# Applies the same exclusion list as ``_ZIM_LINK_RE`` (skip absolute,
# protocol-relative, data:, fragment, root, mailto:, javascript:). The
# HTML ``<base>`` tag does NOT affect CSS ``url()`` resolution — those
# URLs resolve relative to the stylesheet's own URL — so without this
# rewrite, a stylesheet at ``-/style.css`` referencing ``url(I/foo.png)``
# resolves to ``-/I/foo.png`` (wrong) instead of the ZIM-root
# ``I/foo.png``. Mirroring the HTML rewriter's namespace-prefix-friendly
# behavior keeps stylesheet output consistent with anchor/img output.
#
# Root-relative paths starting with ``/`` are intentionally excluded for
# parity with the HTML rewriter — fixing those (so they target the ZIM
# root rather than the host root) is a separate, broader change that
# would need to land for both rewriters at once.
_CSS_URL_RE = re.compile(
    r'url\(\s*(["\']?)'
    r'(?!https?://|//|data:|#|/|mailto:|javascript:)'
    r'([^"\')\s]+)\1\s*\)',
    re.IGNORECASE,
)


def _rewrite_zim_css(css: str, pack_id: str) -> str:
    """Rewrite ``url(...)`` references in a ZIM-served stylesheet so they
    point back at our route. Returns the CSS verbatim if no matches.

    Idempotent: already-rewritten paths start with ``/`` and are excluded
    by the regex's negative lookahead.
    """
    def _replace(m: re.Match) -> str:
        quote = m.group(1)
        path = m.group(2)
        return f'url({quote}/api/knowledge/zim/{pack_id}/{path}{quote})'
    return _CSS_URL_RE.sub(_replace, css)


# Theme presets mirror the four themes in ui/scripts/grove.js (dark, light,
# midnight, sepia). Hardcoded here rather than parsed from CSS files so the
# server doesn't have to keep up with stylesheet refactors. The iframe is
# isolated (sandbox + cross-document), so it can't read parent CSS variables
# directly — the parent passes theme via ?theme= and we inject a matching
# palette into the served HTML's <head>.
# ``scheme`` declares the intended UA color scheme via the ``color-scheme``
# CSS property. It lets the browser pick matching defaults for native form
# controls and scrollbars, and — more importantly here — neutralizes the
# ``@media (prefers-color-scheme: dark)`` rules that 2024+ Wikipedia ZIMs
# ship (kiwix-js#1376). Without this, a user on dark-mode OS who picks our
# light or sepia theme gets the article's dark CSS forced on top.
# Skin families we ship per-pack CSS for. ``generic`` is the catch-all
# fallback when a pack id matches none of the known prefixes (user-imported
# ZIMs with arbitrary names) — those still get the universal baseline so
# text never goes invisible, just no chrome-hiding or family-specific
# layout. Order matters in ``_skin_family_for_pack``: more specific
# prefixes (``stack_exchange``, ``stackoverflow``) must come before
# generic substrings.
_SKIN_FAMILIES = (
    "mediawiki", "devdocs", "stackexchange",
    "freecodecamp", "gutenberg", "generic",
)


def _skin_family_for_pack(pack_id: str) -> str:
    """Map a pack id to its skin family.

    Detection is pure-string against the pack id; the catalog populates
    these tokens but user-imported ZIMs can be named anything. Unknown
    names fall through to ``generic`` so they still receive the baseline
    palette + inline-style neutralizer (the chrome-hiding rules just
    don't apply).

    The mediawiki family covers every MediaWiki-shaped pack — Wikipedia,
    MDWiki, Wikibooks, Wikivoyage, Wiktionary, Wikiquote, Wikiversity,
    Wikinews, libretexts (uses MediaWiki backend). Anything else with
    the same ``.mw-parser-output`` body wrapper benefits from being
    labelled mediawiki; misclassification only loses the chrome hide,
    not the readable text.

    Wikisource is intentionally NOT in the mediawiki bucket: it's a
    MediaWiki backend but the user-facing intent is "read a book", and
    the gutenberg family's serif typography + wider measure beats the
    mediawiki chrome treatment for long-form prose.
    """
    pid = (pack_id or "").lower()
    # Books first — Wikisource is technically MediaWiki-shaped, but the
    # user-facing intent is reading prose, not browsing reference. Match
    # before the wiki bucket so ``wikisource_*`` doesn't get caught by it.
    if "gutenberg" in pid or "wikisource" in pid:
        return "gutenberg"
    if any(t in pid for t in (
        "wikipedia", "mdwiki", "wikibooks", "wikivoyage", "wiktionary",
        "wikiquote", "wikiversity", "wikinews", "wikimed",
        "libretexts", "rationalwiki", "appropedia",
    )):
        return "mediawiki"
    if "devdocs" in pid:
        return "devdocs"
    # Order: long token first so ``stack_exchange`` matches before
    # the bare ``stack`` substring would.
    if "stack_exchange" in pid or "stackoverflow" in pid or pid.startswith("sotoki"):
        return "stackexchange"
    if "freecodecamp" in pid or pid.startswith("fcc_"):
        return "freecodecamp"
    return "generic"


# Syntax-highlight palettes per theme. Kept in lock-step with the
# ``--syntax-*`` tokens defined in ``ui/styles/themes.css`` so chat code
# blocks and ZIM reader code blocks look identical for the same theme.
# Update both when adjusting one.
_ZIM_SYNTAX_PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "keyword": "#c678dd", "string": "#98c379", "number": "#d19a66",
        "comment": "#6b6b80", "function": "#61afef", "variable": "#e06c75",
        "type": "#56b6c2", "attr": "#d19a66", "built-in": "#56b6c2",
    },
    "light": {
        "keyword": "#a626a4", "string": "#50a14f", "number": "#986801",
        "comment": "#a0a0a0", "function": "#4078f2", "variable": "#e45649",
        "type": "#c18401", "attr": "#986801", "built-in": "#c18401",
    },
    "midnight": {
        "keyword": "#b392f0", "string": "#9ecbff", "number": "#ffab70",
        "comment": "#6e7681", "function": "#79b8ff", "variable": "#f97583",
        "type": "#b8e0ff", "attr": "#ffab70", "built-in": "#b8e0ff",
    },
    "sepia": {
        "keyword": "#e8a35a", "string": "#9bb87c", "number": "#c97a3c",
        "comment": "#7a6c58", "function": "#5f9ea0", "variable": "#c97a3c",
        "type": "#c8a868", "attr": "#d4a55a", "built-in": "#5f9ea0",
    },
}


_ZIM_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#1a1a1d", "surface": "#222226", "text": "#e8e8ec",
        "muted": "#9a9aa3", "accent": "#6c8aff", "border": "#2c2c30",
        "code-bg": "#2c2c30", "scheme": "only dark",
    },
    "light": {
        "bg": "#f4f1ec", "surface": "#fbfaf7", "text": "#1f1d1a",
        "muted": "#6a6862", "accent": "#5b73d9", "border": "#e3dfd7",
        "code-bg": "#eeeae3", "scheme": "only light",
    },
    "midnight": {
        "bg": "#0a1020", "surface": "#101830", "text": "#e2eaf8",
        "muted": "#94a3c4", "accent": "#38bdf8", "border": "#1c2745",
        "code-bg": "#162038", "scheme": "only dark",
    },
    "sepia": {
        "bg": "#f1e7d0", "surface": "#f8f0d9", "text": "#3a2c18",
        "muted": "#7a6648", "accent": "#b88746", "border": "#d9c8a2",
        "code-bg": "#ede1c4", "scheme": "only light",
    },
}
# ``only dark`` / ``only light`` is the strongest form of color-scheme. It
# tells the UA to forbid mixed-scheme rendering entirely, defeating nested
# ``@media (prefers-color-scheme: …)`` blocks in the source ZIM's own CSS
# (Wikipedia 2024+ Vector skin, MDWiki, modern DevDocs). Without ``only``,
# a user on a dark-mode OS who picks our light/sepia theme sees infoboxes
# darken anyway because the source CSS's prefers-color-scheme:dark block
# fires off OS state rather than our declaration.


def _zim_reader_baseline_css(theme: str) -> str:
    """Universal Layer 1 baseline injected for every ZIM regardless of skin.

    Two jobs:
      1. Declare the ``--augz-*`` palette + ``color-scheme`` so per-family
         packs (Layer 2) and the inline-style neutralizer (Layer 3) all
         share a single palette source.
      2. Set sane page-level defaults (bg, text, font, line-height) so a
         pack we've never seen still produces readable text. This is the
         floor — without it, an unknown ZIM whose CSS hardcodes
         ``color: #222`` against our dark background goes invisible.

    Body-level ``color`` uses ``!important`` to win against the article's
    ``body { color: … }`` rule (every site declares one). Per-element
    overrides further down the cascade (paragraphs, headings, links) do
    NOT use ``!important`` here — that lets the article's own accent
    colors on highlights / warnings / syntax-highlighted code survive,
    which the inline-style neutralizer also preserves.

    Theme defaults to dark on unknown values; never raises.
    """
    palette = _ZIM_THEMES.get(theme, _ZIM_THEMES["dark"])
    return f"""
    :root {{
        color-scheme: {palette["scheme"]};
        --augz-bg: {palette["bg"]};
        --augz-surface: {palette["surface"]};
        --augz-text: {palette["text"]};
        --augz-muted: {palette["muted"]};
        --augz-accent: {palette["accent"]};
        --augz-border: {palette["border"]};
        --augz-code-bg: {palette["code-bg"]};
    }}

    /* Page-level palette + typography. CJK fallbacks (Noto/PingFang/
       YaHei/Hiragino/Yu Gothic/Malgun) included before ``sans-serif``
       so Wikipedia zh/ja/ko ZIMs render proper glyphs on stripped-down
       systems. Browsers fall through per-glyph, so listing all four
       families is cheap. */
    html, body {{
        background: var(--augz-bg) !important;
        color: var(--augz-text) !important;
        margin: 0 !important;
        padding: 0 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', Arial,
                     'Noto Sans CJK SC', 'Noto Sans CJK TC',
                     'Noto Sans CJK JP', 'Noto Sans CJK KR',
                     'PingFang SC', 'Microsoft YaHei',
                     'Hiragino Sans', 'Yu Gothic', 'Malgun Gothic',
                     sans-serif !important;
        font-size: 16px !important;
        line-height: 1.65 !important;
        -webkit-font-smoothing: antialiased;
    }}

    /* Text-bearing elements inherit ``--augz-text`` BY DEFAULT (no
       ``!important``) so inline ``color:`` on a span — preserved by the
       neutralizer when it's a real accent — can still take effect. The
       neutralizer strips only fully-opaque colors that would clash with
       our backgrounds; what survives is intentional. */
    body p, body li, body td, body th, body dd, body dt,
    body h1, body h2, body h3, body h4, body h5, body h6,
    body article, body section, body main, body aside,
    body figure, body figcaption, body blockquote {{
        color: var(--augz-text);
    }}

    /* Links — accent everywhere; per-family rules can refine if the
       family ships its own decorations. */
    body a, body a:visited {{
        color: var(--augz-accent);
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-color 0.15s ease;
    }}
    body a:hover {{
        border-bottom-color: var(--augz-accent);
    }}

    /* Inline + block code — universal so even unknown packs read cleanly.
       SF Mono / Menlo / Consolas covers macOS / Linux / Windows defaults. */
    body code, body pre, body tt, body kbd, body samp {{
        background: var(--augz-code-bg);
        color: var(--augz-text);
        border: 1px solid var(--augz-border);
        border-radius: 4px;
        padding: 0.1em 0.35em;
        font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
        font-size: 0.92em;
    }}
    body pre {{
        padding: 12px 16px;
        overflow-x: auto;
    }}
    body pre code {{
        background: transparent;
        border: 0;
        padding: 0;
    }}

    /* Tables — borders themed everywhere. Per-family packs can override
       layout (full-width, scroll-wrap, etc.). */
    body table {{
        border-collapse: collapse;
        max-width: 100%;
        color: var(--augz-text);
        background: var(--augz-surface);
    }}
    body th, body td {{
        border: 1px solid var(--augz-border);
        padding: 6px 10px;
    }}
    body th {{
        background: var(--augz-bg);
        font-weight: 600;
    }}

    /* Blockquotes — themed accent bar */
    body blockquote {{
        border-left: 3px solid var(--augz-accent);
        background: var(--augz-surface);
        margin: 1em 0;
        padding: 8px 16px;
    }}

    /* Horizontal rule */
    body hr {{
        border: 0;
        border-top: 1px solid var(--augz-border);
        margin: 2em 0;
    }}

    /* Images stay inside the column on every family */
    body img {{
        max-width: 100%;
        height: auto;
    }}

    /* Selection — match accent */
    ::selection {{
        background: var(--augz-accent);
        color: var(--augz-bg);
    }}
    {_zim_reader_invert_rules(theme)}
    {_zim_reader_syntax_css(theme)}
    """


def _zim_reader_invert_rules(theme: str) -> str:
    """Image / SVG invert rules for dark-family themes.

    Diagrams, math equations, schematic SVGs, and chart PNGs almost always
    ship with a white background — they were authored for a light page.
    On our dark/midnight themes they glare against the surrounding text
    column. ``filter: invert(1) hue-rotate(180deg)`` flips lightness while
    preserving hue identity, so a blue line in a schematic stays blue
    against a dark canvas instead of becoming a near-black thread.

    Photos opt out via ``data-aug-no-invert`` (set in ``_tag_photo_images``)
    because inverting a photograph produces an X-ray-like surreality. Inline
    ``<svg>`` always inverts in dark themes — Wikipedia math (MathML→SVG),
    chemistry structures, and circuit diagrams all benefit.

    Light + sepia themes return an empty rule set — light pages don't need
    inversion, and inverting a photo on a sepia background would look broken.
    """
    if theme not in ("dark", "midnight"):
        return ""
    return """
    /* Dark themes: invert non-photo images so light-background diagrams
       and math SVGs sit naturally on the dark page. */
    body img:not([data-aug-no-invert]),
    body svg {
        filter: invert(1) hue-rotate(180deg);
        transition: filter 150ms ease;
    }
    """


def _zim_reader_mediawiki_css() -> str:
    """Layer 2 pack for MediaWiki-shaped ZIMs (Wikipedia, MDWiki,
    Wikibooks, Wikivoyage, Wiktionary, Wikiquote, Wikiversity, libretexts,
    RationalWiki, Appropedia, Wikimed).

    Hides Vector skin chrome, resets the nested grid wrappers, and
    centers the article on a 720px column. All rules scoped under
    ``.mw-parser-output`` or Vector-skin selectors so they only fire on
    pages that actually carry that markup — safe to inject even if the
    family detection is slightly off.
    """
    return """
    /* Hide Vector skin chrome — sidebar, top header, footer, edit links,
       table of contents (we have section navigation in the article), and
       category links. Anything that says "you're on Wikipedia" goes. */
    html.augz-reader body > header,
    .vector-header-container, #mw-head, #mw-panel,
    .vector-page-tools, .vector-page-tools-pinned, .vector-column-end,
    .mw-jump-link, .vector-sticky-pinned-container,
    .mw-indicators, .mw-editsection, .mw-redirectedfrom,
    #footer, #footer-info, #footer-places, .printfooter,
    .catlinks, #catlinks, .vector-toc, .toc, #toc,
    .vector-search-box, #p-search, #siteSub,
    .navbox, .sistersitebox, .metadata, .ambox,
    .hatnote.navigation-not-searchable {
        display: none !important;
    }

    /* Vector wraps content in nested grids/flexes — reset them. */
    .vector-body, .mw-page-container, .mw-page-container-inner,
    .mw-content-container, .mw-body, #content, #mw-content-text,
    .mw-body-content, .mw-parser-output {
        background: transparent !important;
        max-width: none !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    /* The reading column. Sits inside whatever Vector wrapper survived. */
    .mw-parser-output {
        max-width: 720px !important;
        margin: 32px auto 64px !important;
        padding: 0 24px !important;
    }

    /* Title — hide Vector's heavyweight hero, the article's first <h1> in
       parser output is enough. */
    h1#firstHeading, .mw-first-heading {
        max-width: 720px !important;
        margin: 32px auto 8px !important;
        padding: 0 24px !important;
        background: transparent !important;
        color: var(--augz-text) !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        border: 0 !important;
    }

    /* Headings — match Augmentum's typographic rhythm. */
    .mw-parser-output h2 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin: 2em 0 0.6em !important;
        padding-bottom: 0.3em !important;
        border-bottom: 1px solid var(--augz-border) !important;
    }
    .mw-parser-output h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin: 1.6em 0 0.4em !important;
    }
    .mw-parser-output h4, .mw-parser-output h5, .mw-parser-output h6 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin: 1.4em 0 0.3em !important;
        color: var(--augz-muted) !important;
    }

    .mw-parser-output p { margin: 0 0 1em !important; }
    .mw-parser-output a.new { color: var(--augz-muted) !important; }

    /* Wikipedia-style figures (right-floated thumbnail boxes) */
    .mw-parser-output .thumb, .mw-parser-output .thumbinner {
        background: var(--augz-surface) !important;
        border: 1px solid var(--augz-border) !important;
        border-radius: 6px !important;
        padding: 6px !important;
        max-width: 320px !important;
    }
    .mw-parser-output .thumbcaption {
        color: var(--augz-muted) !important;
        font-size: 0.88rem !important;
        padding: 4px 6px !important;
    }
    .mw-parser-output img { border-radius: 4px; }

    /* References / footnotes */
    .mw-parser-output .reference, .mw-parser-output sup.reference {
        font-size: 0.75em !important;
        color: var(--augz-muted) !important;
    }
    .mw-parser-output .references, .mw-parser-output ol.references {
        font-size: 0.88rem !important;
        color: var(--augz-muted) !important;
        border-top: 1px solid var(--augz-border) !important;
        padding-top: 1em !important;
        margin-top: 2em !important;
    }
    """


def _zim_reader_devdocs_css() -> str:
    """Layer 2 pack for DevDocs ZIMs (devdocs_en_python, devdocs_en_*).

    DevDocs ships as an SPA with a class hierarchy rooted at ``_app``,
    ``_page``, ``_sidebar``, ``_search``. The sidebar is the dominant
    chrome; hide it and stretch the content area. Code blocks are the
    main signal so they get extra padding + horizontal scroll.

    DevDocs body classes also carry the original theme (``_theme-default``,
    ``_theme-dark``); we don't strip them since neutralizing inline
    color/bg via Layer 3 is enough — the cascading classes that fight us
    are all in their own stylesheet (which the iframe still loads), and
    those are scoped to descendants we override here.
    """
    return """
    /* Hide DevDocs chrome — sidebar, search bar, theme toggle, nav. */
    ._sidebar, ._search, ._toolbar, ._settings, ._notif,
    ._news, ._notice, ._intro, ._notice-link,
    body > nav, ._mobile-search {
        display: none !important;
    }

    /* Stretch the content frame — DevDocs uses fixed left padding to
       make room for the sidebar; reset it. */
    body, ._app, ._container, ._main {
        background: var(--augz-bg) !important;
        color: var(--augz-text) !important;
        margin: 0 !important;
        padding: 0 !important;
        max-width: none !important;
        width: 100% !important;
    }

    /* The reading column for DevDocs API pages. */
    ._page, ._content, ._static, ._reference {
        background: transparent !important;
        max-width: 820px !important;
        margin: 32px auto 64px !important;
        padding: 0 24px !important;
        color: var(--augz-text) !important;
    }

    ._page h1, ._page h2, ._page h3,
    ._content h1, ._content h2, ._content h3 {
        color: var(--augz-text) !important;
        border-color: var(--augz-border) !important;
    }
    ._page h2, ._content h2 {
        border-bottom: 1px solid var(--augz-border) !important;
        padding-bottom: 0.3em !important;
    }

    /* DevDocs syntax highlighting uses many small color classes. Leave
       them visible (they're semantic) but ensure the surrounding pre/code
       picks up our theme background so the highlight stays readable. */
    ._page pre, ._content pre {
        background: var(--augz-code-bg) !important;
        border: 1px solid var(--augz-border) !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
    }

    /* Tables (Python's parameter tables, JS's MDN-style refs). */
    ._page table, ._content table {
        background: var(--augz-surface) !important;
        color: var(--augz-text) !important;
    }
    """


def _zim_reader_stackexchange_css() -> str:
    """Layer 2 pack for Stack Exchange / sotoki ZIMs.

    SE markup centers on ``.s-card`` (question/answer card), ``.s-prose``
    (post body), ``.user-info``, ``.vote-cell``. Hide site chrome (top
    bar, footer, sidebar with related questions) and theme the cards
    against the reader palette.
    """
    return """
    /* Hide Stack Exchange site chrome. */
    .top-bar, .so-top-bar, #header, header[role=banner],
    .footer, footer[role=contentinfo], #footer,
    .sidebar, #sidebar, #left-sidebar, #right-sidebar,
    .js-consent-banner, .js-dismissable-notice,
    .js-search-form, #search, .top-notice {
        display: none !important;
    }

    body, .content-page, #content, .container, .mainbar, .question-page {
        background: var(--augz-bg) !important;
        color: var(--augz-text) !important;
        margin: 0 !important;
        max-width: none !important;
        width: 100% !important;
    }

    /* Card layout — question + answers. */
    .question, .answer, .s-card, .post-layout {
        background: var(--augz-surface) !important;
        border: 1px solid var(--augz-border) !important;
        border-radius: 8px !important;
        margin: 16px auto !important;
        padding: 16px 20px !important;
        max-width: 820px !important;
        color: var(--augz-text) !important;
    }

    /* Post body prose — links, lists, paragraphs all themed. */
    .s-prose, .post-text, .answercell {
        color: var(--augz-text) !important;
    }

    /* User info card (asker/answerer). */
    .user-info, .user-details, .user-card-row {
        background: var(--augz-bg) !important;
        border: 1px solid var(--augz-border) !important;
        border-radius: 6px !important;
        padding: 8px !important;
        color: var(--augz-muted) !important;
    }

    /* Tags — accent pill style. */
    .post-tag, .s-tag {
        background: var(--augz-code-bg) !important;
        color: var(--augz-accent) !important;
        border: 1px solid var(--augz-border) !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 0.85em !important;
        text-decoration: none !important;
    }

    /* Vote/score cells. */
    .vote-cell, .js-vote-count, .vote-count-post {
        color: var(--augz-muted) !important;
    }

    /* SE wraps code in pre.lang-* — palette the wrappers without
       overriding the syntax highlighter colors inside. */
    .question pre, .answer pre, .s-prose pre {
        background: var(--augz-code-bg) !important;
        border: 1px solid var(--augz-border) !important;
        border-radius: 6px !important;
    }
    """


def _zim_reader_freecodecamp_css() -> str:
    """Layer 2 pack for freeCodeCamp ZIMs.

    fCC is an SPA whose curriculum pages render under a layout shell.
    Wikiclass-style mirrors use ``.challenge-`` and ``.fcc-`` prefixes;
    sidebar nav + top bar are the dominant chrome. Code editors are
    interactive — leave them alone (just frame the surrounding container).
    """
    return """
    /* Hide fCC site chrome. */
    nav, header, .universal-nav, .nav-skeleton,
    .donate-page-bar, .donate-banner,
    footer, .footer, .global-footer,
    .challenge-sidebar, .map-nav {
        display: none !important;
    }

    body, #root, .layout-container, .challenge-container,
    .article-container, .lesson-container {
        background: var(--augz-bg) !important;
        color: var(--augz-text) !important;
        margin: 0 !important;
        padding: 0 !important;
        max-width: none !important;
    }

    /* Challenge / curriculum content column. */
    .challenge-content, .challenge-instructions,
    .article-content, .lesson-content, .news-article {
        max-width: 820px !important;
        margin: 24px auto !important;
        padding: 0 24px !important;
        color: var(--augz-text) !important;
        background: transparent !important;
    }

    /* Code-editor wrapper — frame but don't restyle the editor itself
       (Monaco has its own theme; let it ride). */
    .challenge-editor, .react-monaco-editor-container {
        border: 1px solid var(--augz-border) !important;
        border-radius: 6px !important;
        overflow: hidden;
    }

    /* Article body for News / Forum mirrors. */
    .article-content h1, .article-content h2,
    .article-content h3, .news-article h1, .news-article h2 {
        color: var(--augz-text) !important;
        border-color: var(--augz-border) !important;
    }
    """


def _zim_reader_gutenberg_css() -> str:
    """Layer 2 pack for Gutenberg / Wikisource book ZIMs.

    Books are mostly bare HTML — ``<body><h1>…</h1><p>…</p></body>`` or
    wrapped in ``<div class="chapter">``. Serif typography fits the
    reading context better than the system sans-serif baseline. Wider
    measure (760px) because long-form fiction reads better with slightly
    longer lines.
    """
    return """
    body {
        font-family: Georgia, 'Iowan Old Style', 'Palatino Linotype',
                     'Book Antiqua', Palatino, serif !important;
        font-size: 17px !important;
        line-height: 1.7 !important;
    }

    /* Inset reading column — bare-HTML books usually have a top-level
       body or single container holding everything. */
    body > * {
        max-width: 760px;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 24px;
        padding-right: 24px;
    }

    body > h1, body > h2, body > h3 {
        margin-top: 1.6em;
        color: var(--augz-text) !important;
    }

    /* Wikisource book chrome. */
    .navfields, .header-section, .header_notes,
    .indicator, .editsection {
        display: none !important;
    }

    /* Verse / poetry blocks (common in Wikisource). */
    .poem, .verse {
        font-style: italic;
        margin: 1.5em auto !important;
        color: var(--augz-text) !important;
    }

    /* Drop caps look gimmicky against dark backgrounds; flatten them. */
    .dropinitial, .initial {
        font-size: inherit !important;
        float: none !important;
        color: var(--augz-text) !important;
    }
    """


def _zim_reader_syntax_css(theme: str) -> str:
    """hljs + Prism token CSS for ZIM code blocks.

    Mirrors the rules in ``ui/styles/chat.css`` so chat and ZIM reader
    code blocks look identical for the same theme. Selectors use
    ``!important`` to defeat the highlighter's own bundled stylesheet,
    which is loaded from the ZIM's ``<link rel="stylesheet">`` and would
    otherwise win source-order tie-breaks.
    """
    pal = _ZIM_SYNTAX_PALETTES.get(theme, _ZIM_SYNTAX_PALETTES["dark"])
    return f"""
    /* Code blocks (hljs + Prism) — per-theme token palette. */
    body .hljs {{ color: var(--augz-text) !important; background: transparent !important; }}
    body .hljs-keyword, body .hljs-selector-tag, body .hljs-literal,
    body .hljs-section, body .hljs-link {{ color: {pal["keyword"]} !important; }}
    body .hljs-string, body .hljs-doctag, body .hljs-regexp {{ color: {pal["string"]} !important; }}
    body .hljs-number, body .hljs-symbol, body .hljs-bullet,
    body .hljs-meta {{ color: {pal["number"]} !important; }}
    body .hljs-comment, body .hljs-quote {{
        color: {pal["comment"]} !important;
        font-style: italic;
    }}
    body .hljs-function, body .hljs-title, body .hljs-title.function_,
    body .hljs-name {{ color: {pal["function"]} !important; }}
    body .hljs-variable, body .hljs-template-variable,
    body .hljs-attribute, body .hljs-deletion {{ color: {pal["variable"]} !important; }}
    body .hljs-type, body .hljs-class .hljs-title,
    body .hljs-tag, body .hljs-addition {{ color: {pal["type"]} !important; }}
    body .hljs-attr, body .hljs-selector-attr,
    body .hljs-selector-class, body .hljs-selector-id {{ color: {pal["attr"]} !important; }}
    body .hljs-built_in, body .hljs-builtin-name {{ color: {pal["built-in"]} !important; }}
    body .hljs-emphasis {{ font-style: italic; }}
    body .hljs-strong {{ font-weight: 600; }}

    body .token.keyword, body .token.atrule,
    body .token.important {{ color: {pal["keyword"]} !important; }}
    body .token.string, body .token.char,
    body .token.regex {{ color: {pal["string"]} !important; }}
    body .token.number, body .token.boolean,
    body .token.constant {{ color: {pal["number"]} !important; }}
    body .token.comment, body .token.prolog,
    body .token.cdata {{ color: {pal["comment"]} !important; font-style: italic; }}
    body .token.function, body .token.class-name {{ color: {pal["function"]} !important; }}
    body .token.variable, body .token.deleted {{ color: {pal["variable"]} !important; }}
    body .token.tag, body .token.builtin,
    body .token.inserted {{ color: {pal["type"]} !important; }}
    body .token.attr-name, body .token.selector {{ color: {pal["attr"]} !important; }}
    body .token.operator, body .token.punctuation {{
        color: var(--augz-text) !important;
        opacity: 0.75;
    }}
    """


_ZIM_FIND_BAR_CSS = """
#aug-find-bar {
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translate(-50%, calc(-100% - 16px));
    z-index: 2147483647;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 9999px;
    background: color-mix(in srgb, var(--augz-surface) 92%, transparent);
    border: 1px solid var(--augz-border);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    color: var(--augz-text);
    transition: transform 250ms cubic-bezier(0.25, 0.46, 0.45, 0.94),
                opacity 150ms ease;
    opacity: 0;
    pointer-events: none;
}
#aug-find-bar.aug-find-open {
    transform: translate(-50%, 0);
    opacity: 1;
    pointer-events: auto;
}
#aug-find-bar input {
    border: 0;
    outline: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    width: 200px;
    padding: 4px 6px;
}
#aug-find-bar input::placeholder {
    color: var(--augz-muted);
}
#aug-find-bar .aug-find-count {
    font-variant-numeric: tabular-nums;
    color: var(--augz-muted);
    padding: 0 4px;
    white-space: nowrap;
}
#aug-find-bar button {
    width: 24px;
    height: 24px;
    border: 0;
    border-radius: 9999px;
    background: transparent;
    color: var(--augz-muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    transition: background 150ms ease, color 150ms ease;
}
#aug-find-bar button:hover {
    background: var(--augz-surface);
    color: var(--augz-text);
}
#aug-find-bar button:focus-visible {
    outline: 2px solid var(--augz-accent);
    outline-offset: 1px;
}
mark.aug-find-hit {
    background: var(--augz-accent) !important;
    color: var(--augz-bg) !important;
    opacity: 0.45;
    padding: 0 1px;
    border-radius: 2px;
}
mark.aug-find-hit.aug-find-current {
    opacity: 1;
    box-shadow: 0 0 0 2px var(--augz-accent);
}
@media print {
    #aug-find-bar { display: none !important; }
    mark.aug-find-hit { background: transparent !important; color: inherit !important; box-shadow: none !important; }
}
"""

# Find-bar JS. Self-contained vanilla DOM manipulation — runs inside the
# sandboxed iframe and never reaches the parent. Idempotent: checks
# ``window.__augFindBarReady`` so a re-injection (e.g. SPA route change)
# doesn't double-bind. The TreeWalker excludes our own bar plus script /
# style blocks so the search doesn't match injected CSS / JS source.
_ZIM_FIND_BAR_JS = r"""
(function () {
    if (window.__augFindBarReady) return;
    window.__augFindBarReady = true;

    var bar = document.getElementById('aug-find-bar');
    if (!bar) return;
    var input = bar.querySelector('input');
    var countEl = bar.querySelector('.aug-find-count');
    var prevBtn = bar.querySelector('[data-aug-prev]');
    var nextBtn = bar.querySelector('[data-aug-next]');
    var closeBtn = bar.querySelector('[data-aug-close]');

    var matches = [];
    var currentIdx = -1;

    function clearMarks() {
        var hits = document.querySelectorAll('mark.aug-find-hit');
        hits.forEach(function (m) {
            var parent = m.parentNode;
            if (!parent) return;
            while (m.firstChild) parent.insertBefore(m.firstChild, m);
            parent.removeChild(m);
            parent.normalize();
        });
    }

    function highlight(query) {
        clearMarks();
        matches = [];
        currentIdx = -1;
        if (!query) {
            updateCounter();
            return;
        }
        var lower = query.toLowerCase();
        var walker = document.createTreeWalker(
            document.body, NodeFilter.SHOW_TEXT,
            {
                acceptNode: function (node) {
                    if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
                    var p = node.parentNode;
                    while (p && p !== document.body) {
                        var tag = p.nodeName;
                        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT') return NodeFilter.FILTER_REJECT;
                        if (p.id === 'aug-find-bar') return NodeFilter.FILTER_REJECT;
                        if (p.nodeName === 'MARK' && p.classList.contains('aug-find-hit')) return NodeFilter.FILTER_REJECT;
                        p = p.parentNode;
                    }
                    return node.nodeValue.toLowerCase().indexOf(lower) !== -1
                        ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                }
            }
        );
        var pending = [];
        while (walker.nextNode()) pending.push(walker.currentNode);
        for (var i = 0; i < pending.length; i++) {
            var node = pending[i];
            var text = node.nodeValue;
            var lc = text.toLowerCase();
            var idx = 0, found, frag = document.createDocumentFragment();
            while ((found = lc.indexOf(lower, idx)) !== -1) {
                if (found > idx) frag.appendChild(document.createTextNode(text.slice(idx, found)));
                var mark = document.createElement('mark');
                mark.className = 'aug-find-hit';
                mark.textContent = text.slice(found, found + query.length);
                frag.appendChild(mark);
                matches.push(mark);
                idx = found + query.length;
            }
            if (idx < text.length) frag.appendChild(document.createTextNode(text.slice(idx)));
            node.parentNode.replaceChild(frag, node);
        }
        if (matches.length) {
            currentIdx = 0;
            focusCurrent();
        }
        updateCounter();
    }

    function focusCurrent() {
        matches.forEach(function (m) { m.classList.remove('aug-find-current'); });
        if (currentIdx < 0 || currentIdx >= matches.length) return;
        var node = matches[currentIdx];
        node.classList.add('aug-find-current');
        try { node.scrollIntoView({ block: 'center', behavior: 'smooth' }); } catch (e) {
            node.scrollIntoView();
        }
    }

    function updateCounter() {
        if (!countEl) return;
        if (!matches.length) {
            countEl.textContent = input.value ? '0' : '';
            countEl.setAttribute('aria-label', input.value ? 'No matches' : '');
            return;
        }
        var n = currentIdx + 1;
        countEl.textContent = n + ' of ' + matches.length;
        countEl.setAttribute('aria-label', n + ' of ' + matches.length + ' matches');
    }

    function step(delta) {
        if (!matches.length) return;
        currentIdx = (currentIdx + delta + matches.length) % matches.length;
        focusCurrent();
        updateCounter();
    }

    function open() {
        bar.classList.add('aug-find-open');
        setTimeout(function () { input.focus(); input.select(); }, 50);
    }
    function close() {
        bar.classList.remove('aug-find-open');
        clearMarks();
        matches = [];
        currentIdx = -1;
        input.value = '';
        updateCounter();
    }

    var debounce = null;
    input.addEventListener('input', function () {
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(function () { highlight(input.value); }, 120);
    });
    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            if (e.shiftKey) step(-1); else step(1);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            close();
        }
    });
    if (prevBtn) prevBtn.addEventListener('click', function () { step(-1); });
    if (nextBtn) nextBtn.addEventListener('click', function () { step(1); });
    if (closeBtn) closeBtn.addEventListener('click', close);

    document.addEventListener('keydown', function (e) {
        var isOpen = bar.classList.contains('aug-find-open');
        // Cmd/Ctrl+F focuses the bar (overrides browser find, intentionally).
        if ((e.ctrlKey || e.metaKey) && (e.key === 'f' || e.key === 'F')) {
            e.preventDefault();
            if (!isOpen) open(); else { input.focus(); input.select(); }
            return;
        }
        // ``/`` opens the bar when no input has focus — common docs-style
        // shortcut. Skip if the user is typing in any input/textarea.
        if (e.key === '/' && !isOpen) {
            var t = e.target;
            if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
            e.preventDefault();
            open();
        }
    });
})();
"""


def _zim_reader_find_bar_markup() -> str:
    """HTML/CSS/JS bundle for the in-article find bar.

    Injected at the start of ``<body>`` in reader mode. The bar slides
    down from the top on ``/`` or ``Ctrl+F`` (intercepting the browser's
    native Find — by design, our bar matches the reader theme and
    persists across SPA-style ZIM nav, neither of which the browser
    find does).

    Self-contained: no external deps, runs entirely inside the iframe.
    Safe under the iframe's CSP because ``allow-scripts`` is enabled on
    the sandbox and our injected JS uses only same-origin DOM APIs.
    """
    return (
        f'<style data-augmentum-find="1">{_ZIM_FIND_BAR_CSS}</style>'
        '<div id="aug-find-bar" role="search" aria-label="Find in article">'
        '<input type="search" placeholder="Find in article…" '
        'autocomplete="off" autocapitalize="off" spellcheck="false" '
        'aria-label="Find in article">'
        '<span class="aug-find-count" aria-live="polite"></span>'
        '<button type="button" data-aug-prev aria-label="Previous match" title="Previous (Shift+Enter)">'
        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" width="12" height="12" aria-hidden="true"><polyline points="3 10 8 5 13 10"/></svg>'
        '</button>'
        '<button type="button" data-aug-next aria-label="Next match" title="Next (Enter)">'
        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" width="12" height="12" aria-hidden="true"><polyline points="3 6 8 11 13 6"/></svg>'
        '</button>'
        '<button type="button" data-aug-close aria-label="Close find bar" title="Close (Esc)">'
        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" width="12" height="12" aria-hidden="true"><line x1="4" y1="4" x2="12" y2="12"/><line x1="12" y1="4" x2="4" y2="12"/></svg>'
        '</button>'
        '</div>'
        f'<script data-augmentum-find="1">{_ZIM_FIND_BAR_JS}</script>'
    )


def _zim_reader_minimal_baseline(theme: str) -> str:
    """Minimal palette injection used in raw (reader-off) mode.

    Only forces ``html, body`` background + foreground + color-scheme to
    the active theme. Everything else — typography, per-element layout,
    inline styles, infobox/navbox chrome — survives untouched because
    "reader off" means *don't restyle content*, not *clash visually with
    the surrounding app chrome*. Without this, the iframe shows the ZIM's
    native (almost always light-only) palette next to Augmentum's dark
    chrome, creating a stark visual seam.

    Trade-off: ZIM text colors that were calibrated against a light
    background may have lower contrast against our forced dark bg. The
    fix is for the user to flip back to reader mode (which neutralizes
    those inline colors). The "Raw view" badge button in the UI surface
    is the discoverability hook for that escape valve.
    """
    palette = _ZIM_THEMES.get(theme, _ZIM_THEMES["dark"])
    return (
        f":root {{ color-scheme: {palette['scheme']}; }}"
        f"html, body {{ "
        f"background: {palette['bg']} !important; "
        f"color: {palette['text']} !important; "
        f"}}"
    )


def _zim_reader_styles(family: str, theme: str) -> str:
    """Compose Layer 1 (baseline) + Layer 2 (per-family) reader CSS.

    Family is one of ``_SKIN_FAMILIES``. Unknown values are treated as
    ``generic`` — they get the baseline only, no chrome-hiding or
    family-specific layout. The baseline is enough to keep text readable
    against the theme; the family pack adds chrome hide + structural
    layout for known ZIM shapes.

    Why CSS injection instead of an HTML rewrite or upstream replace:
      - Iframe is sandboxed (no JS), so we can't manipulate the DOM
        client-side to swap the skin.
      - Stripping the chrome HTML server-side is brittle — markup varies
        between Vector versions, DevDocs builds, SE eras, etc.
      - CSS hides what we don't want, leaves the structural HTML intact
        so section anchors / fragment links keep working.
    """
    baseline = _zim_reader_baseline_css(theme)
    family_css = ""
    if family == "mediawiki":
        family_css = _zim_reader_mediawiki_css()
    elif family == "devdocs":
        family_css = _zim_reader_devdocs_css()
    elif family == "stackexchange":
        family_css = _zim_reader_stackexchange_css()
    elif family == "freecodecamp":
        family_css = _zim_reader_freecodecamp_css()
    elif family == "gutenberg":
        family_css = _zim_reader_gutenberg_css()
    # ``generic`` and unknown → baseline only.
    return baseline + family_css


def _rewrite_zim_html(
    html: str,
    pack_id: str,
    theme: str = "dark",
    family: str = "generic",
    reader_mode: bool = True,
) -> str:
    """Rewrite relative href/src to absolute /api/knowledge/zim/{pack_id}/... URLs.

    Internal ZIM links use prefix conventions like ``A/Article_Name``,
    ``I/image.png``, ``M/metadata``. After rewrite they point back at this
    route, so the iframe can follow them transparently. External links
    (http/https) are untouched and will open as normal — the iframe sandbox
    decides whether the navigation is allowed.

    Scripts and inline event handlers are intentionally NOT stripped — modern
    ZIMs (freeCodeCamp, DevDocs, Stack Exchange mirrors) ship as SPAs whose
    content only renders after their bundle runs. The iframe sandbox + CSP
    in ``serve_zim_entry`` is what actually decides whether scripts execute;
    stripping here would just break SPAs even when scripts are allowed.

    Also injects a ``<base>`` so any link the regex misses still resolves
    to a sane base instead of the iframe's own URL.

    Reader mode (default on) injects the Layer 1 baseline + Layer 2 family
    pack CSS and runs the Layer 3 inline-style neutralizer so source
    HTML's hardcoded colors don't fight the theme. ``reader_mode=False``
    is the escape hatch: link rewriting + base injection still run
    (functional, not cosmetic), but no CSS or style neutralization.
    Used when an SPA fights our reader and the user wants to see the
    original presentation.
    """

    # Replace nested iframes (embedded YouTube clips, NCBI references,
    # OWID charts, etc.) with a static placeholder that links out. The
    # parent CSP's frame-src allowlist would reject most of these and
    # they end up as chrome-error frames in the user's reader.
    def _iframe_placeholder(m: re.Match) -> str:
        attrs = m.group(1)
        src_match = _IFRAME_SRC_RE.search(attrs)
        if not src_match:
            return ""
        src = src_match.group(1)
        # Skip relative or data-URI sources — those still work in-iframe.
        if not (src.startswith("http://") or src.startswith("https://")):
            return m.group(0)
        try:
            from urllib.parse import urlparse
            host = urlparse(src).netloc or src
        except Exception:
            host = src
        # Inline-style placeholder so it picks up reader theme tokens
        # without needing an extra CSS rule. Stays compact in the body
        # flow, doesn't break around it.
        return (
            f'<a href="{src}" target="_blank" rel="noopener noreferrer" '
            f'style="display:inline-block;padding:8px 12px;margin:8px 0;'
            f'border:1px solid var(--augz-border);border-radius:6px;'
            f'background:var(--augz-surface);color:var(--augz-accent);'
            f'text-decoration:none;font-size:0.9em;">'
            f'🔗 Embedded content from {host} (open in new tab)'
            f'</a>'
        )
    html = _NESTED_IFRAME_RE.sub(_iframe_placeholder, html)

    # External anchors → new tab so they don't try to navigate the iframe
    # itself (which the parent CSP blocks for non-allowlisted origins).
    html = _EXTERNAL_ANCHOR_RE.sub(
        r'\1 target="_blank" rel="noopener noreferrer">',
        html,
    )

    def _replace(m: re.Match) -> str:
        attr = m.group(1)
        path = m.group(2)
        return f'{attr}="/api/knowledge/zim/{pack_id}/{path}"'

    rewritten = _ZIM_LINK_RE.sub(_replace, html)
    rewritten = _ZIM_LINK_RE_SQ.sub(
        lambda m: f"{m.group(1)}='/api/knowledge/zim/{pack_id}/{m.group(2)}'",
        rewritten,
    )
    # srcset rewriter — retina image candidates on Wikipedia/MDWiki
    # ZIMs. ``<base>`` covers this in theory but Safari iframes drop
    # base-relative srcset resolution intermittently. Explicit rewrite
    # is cheap and removes the browser-quirk dependency.
    #
    # The other two rewriters from the wider audit (inline-style
    # ``url(...)`` and SVG ``xlink:href``) are intentionally NOT
    # implemented: the injected ``<base>`` handles both reliably
    # across all browsers we target. Adding regex rewriters for them
    # would duplicate the spec'd browser behavior with attendant
    # maintenance cost. Revisit only on a concrete failing pack.
    rewritten = _SRCSET_RE.sub(
        lambda m: f'srcset="{_rewrite_srcset_value(m.group(1), pack_id)}"',
        rewritten,
    )
    rewritten = _SRCSET_RE_SQ.sub(
        lambda m: f"srcset='{_rewrite_srcset_value(m.group(1), pack_id)}'",
        rewritten,
    )
    # Strip ``target="_blank"`` from anchors that point at our route.
    # Must run AFTER the two rewrites above so internal hrefs are already
    # in the ``/api/knowledge/zim/...`` shape the regex anchors against.
    # External anchors are untouched — they keep the ``target="_blank"``
    # added by ``_EXTERNAL_ANCHOR_RE`` and open in a new tab as intended.
    rewritten = _INTERNAL_TARGET_BLANK_RE.sub(r"\1\2", rewritten)
    # Strip any pre-existing ``<base>`` from the article so our injection
    # below has sole authority over relative URL resolution. See the
    # comment on ``_EXISTING_BASE_RE`` for the failure mode this prevents
    # (asset URL 404s from articles that ship ``<base href="../../">``).
    rewritten = _EXISTING_BASE_RE.sub("", rewritten)
    # Inject <base> at the START of <head> so it applies to every link in
    # the document; inject the reader stylesheet at the END of <head> so it
    # wins source-order tie-breaks against the article's own CSS. Without
    # this split, 2024+ Wikipedia ZIMs (kiwix-js#1376) ship a
    # ``@media (prefers-color-scheme: dark)`` block that fights our theme
    # when the user's OS is in dark mode and they've selected our light or
    # sepia theme. Putting our overrides last in <head> gives same-specificity
    # `!important` rules to us by source order.
    # Layer 3: inline-style neutralizer + photo opt-out tagger. Skip both
    # in raw mode — the user opted out of reader theming, so their original
    # inline colors should ride AND image-invert isn't running anyway.
    if reader_mode:
        rewritten = _neutralize_inline_styles(rewritten)
        rewritten = _tag_photo_images(rewritten)

    # ``<meta name="color-scheme">`` paired with the in-CSS ``color-scheme:
    # only <theme>;`` declaration is belt-and-braces against ZIM source CSS
    # that overrides via ``@media (prefers-color-scheme: …)``. The meta tag
    # is read at parse time before any stylesheet runs, so even if the ZIM's
    # own CSS gets to declare scheme later, the browser already knows the
    # canonical color-scheme for this document. Injected even in raw
    # (reader-off) mode so the iframe surface still picks up the right
    # form-control / scrollbar styling — those are UA-rendered and key off
    # color-scheme, not our reader CSS.
    palette = _ZIM_THEMES.get(theme, _ZIM_THEMES["dark"])
    head_prelude = (
        f'<base href="/api/knowledge/zim/{pack_id}/">'
        f'<meta name="color-scheme" content="{palette["scheme"]}">'
    )
    base_tag = head_prelude
    if reader_mode:
        style_tag = (
            f'<style data-augmentum-reader="1" data-augmentum-family="{family}">'
            f'{_zim_reader_styles(family, theme)}</style>'
        )
    else:
        # Raw mode: minimal palette only so the iframe bg/text track the
        # surrounding Augmentum chrome (no visual seam). Content layout,
        # typography, inline colors all survive — that's what "reader off"
        # is for.
        style_tag = (
            f'<style data-augmentum-reader="raw">'
            f'{_zim_reader_minimal_baseline(theme)}</style>'
        )
    # Mark the html element so reader-only selectors can scope safely.
    # Always mark — the class is also a handy DOM hook for the parent
    # frame to confirm rewrite ran.
    if "<html" in rewritten:
        rewritten = re.sub(
            r"<html\b([^>]*)>",
            r'<html\1 class="augz-reader">',
            rewritten,
            count=1,
        )

    if "</head>" in rewritten:
        # Well-formed: base at head start, style at head end.
        if "<head>" in rewritten:
            rewritten = rewritten.replace("<head>", f"<head>{base_tag}", 1)
        rewritten = rewritten.replace("</head>", f"{style_tag}</head>", 1)
    elif "<head>" in rewritten:
        # <head> with no closing tag — malformed but seen in fragment
        # entries. Inject both right after <head>; lose source-order
        # priority but still better than no reader CSS.
        rewritten = rewritten.replace(
            "<head>", f"<head>{base_tag}{style_tag}", 1,
        )
    elif "<html" in rewritten:
        rewritten = re.sub(
            r"(<html[^>]*>)",
            rf"\1<head>{base_tag}{style_tag}</head>",
            rewritten,
            count=1,
        )
    else:
        rewritten = base_tag + style_tag + rewritten

    # Find-bar: in-article search affordance. Only useful in reader mode
    # — raw mode users have the browser's native Find and don't want our
    # chrome layered on top. The bar lives in <body> (not <head>) so its
    # <div> renders correctly; CSS + JS travel alongside it as a single
    # self-contained bundle.
    if reader_mode:
        find_bar = _zim_reader_find_bar_markup()
        if "<body" in rewritten:
            rewritten = re.sub(
                r"(<body\b[^>]*>)",
                lambda m: f"{m.group(1)}{find_bar}",
                rewritten,
                count=1,
            )
        else:
            # No <body> tag (fragment entries) — prepend at document start.
            rewritten = find_bar + rewritten

    return rewritten

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_pack_manager(request: Request):
    mgr = getattr(request.app.state, "pack_manager", None)
    if not mgr:
        raise HTTPException(503, "Knowledge packs not initialized")
    return mgr


def _get_catalog_client(request: Request):
    client = getattr(request.app.state, "catalog_client", None)
    if not client:
        raise HTTPException(503, "Knowledge catalog not initialized")
    return client


@dataclass
class InstallJob:
    """Tracks a background ZIM install/convert task."""

    job_id: str
    catalog_id: str
    status: str = "pending"
    stage: str = ""
    current: int = 0
    total: int = 0
    error: str | None = None
    started_at: float = 0.0
    task: asyncio.Task | None = field(default=None, repr=False)


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------


class DownloadRequest(BaseModel):
    url: str
    filename: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/packs")
async def list_packs(request: Request):
    """List all installed knowledge packs.

    Returns ``{packs: [...], failed_conversions: [...]}``. The failed
    conversions array is empty in the common case; when populated it
    surfaces stuck install jobs so the UI can show a "Conversion
    incomplete" card with Discard / Retry actions.
    """
    mgr = _get_pack_manager(request)
    return {
        "packs": mgr.installed,
        "failed_conversions": list(mgr.failed_conversions),
    }


@router.post("/discard-failed/{pack_id}")
async def discard_failed_conversion(pack_id: str, request: Request):
    """Remove the .progress.json + empty .augpack shell for a stuck
    conversion. Original .zim file is preserved. Admin only — affects
    the shared pack library.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)
    removed = await mgr.discard_failed_conversion(pack_id)
    if not removed:
        raise HTTPException(404, f"No failed conversion found for: {pack_id}")
    system_events.publish("knowledge.changed", {"pack_id": pack_id, "reason": "discard"})
    return {"ok": True, "pack_id": pack_id}


class ResumeFailedRequest(BaseModel):
    # Optional override — defaults to the value used by the original install
    # (settings.knowledge_embedding_batch_size). Most failures are OOM, so
    # the UI surfaces this as a small input on the failed-conversion card so
    # the user can dial it down before retrying.
    batch_size: int | None = None


@router.post("/resume-failed/{pack_id}")
async def resume_failed_conversion(
    pack_id: str,
    request: Request,
    body: ResumeFailedRequest | None = None,
):
    """Resume a previously-failed conversion from the last committed batch.

    Spins up a convert_worker subprocess in --resume mode against the paired
    .zim + partial .augpack on disk. Tracked via the same install_jobs dict
    as a fresh install, so the existing SSE progress stream
    (/install/{job_id}/progress) works transparently.

    Returns 404 if no failed conversion matches, 409 if the pack isn't
    resumable (no .zim, dim mismatch, etc.) or a job is already running for
    this pack_id, and 202 with a job_id on success.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)
    fc = next((f for f in mgr.failed_conversions if f["pack_id"] == pack_id), None)
    if fc is None:
        raise HTTPException(404, f"No failed conversion found for: {pack_id}")
    if not fc.get("resumable"):
        reason = fc.get("not_resumable_reason") or "Not resumable"
        raise HTTPException(409, f"Cannot resume {pack_id}: {reason}")

    jobs: dict = getattr(request.app.state, "install_jobs", None) or {}
    # Refuse if a job already runs for this pack_id (prevents double-start
    # from an impatient user clicking Resume twice).
    for existing in jobs.values():
        if existing.catalog_id == pack_id and existing.status in ("running", "started", "pending"):
            raise HTTPException(409, f"Resume already running for {pack_id}")

    job_id = uuid.uuid4().hex[:12]
    job = InstallJob(
        job_id=job_id,
        catalog_id=pack_id,
        status="started",
        stage="resuming",
        # Start the visible progress at the chunk count we already embedded
        # so the UI doesn't drop back to 0% for a few seconds before stage 1
        # re-extraction completes.
        current=int(fc.get("chunks_committed", 0)),
        total=int(fc.get("last_total", 0)),
        started_at=time.time(),
    )

    batch_size = (body.batch_size if body and body.batch_size else
                  settings.knowledge_embedding_batch_size)
    # Sanity-clamp: 1 is too small to be useful and >2048 risks OOM on most
    # hosts. The UI input limits this too, but defense in depth.
    batch_size = max(1, min(2048, batch_size))

    task = asyncio.create_task(
        _run_resume(
            job=job,
            zim_path=Path(fc["zim_path"]),
            augpack_path=Path(fc["augpack_path"]),
            pack_mgr=mgr,
            batch_size=batch_size,
        )
    )
    job.task = task
    jobs[job_id] = job
    request.app.state.install_jobs = jobs

    return {"job_id": job_id, "status": "started", "batch_size": batch_size}


class EmbedPackRequest(BaseModel):
    # Optional override on the global default. UI offers it as an inline input
    # for users who hit OOM and want to dial down before re-trying.
    batch_size: int | None = None


@router.post("/packs/{pack_id}/embed")
async def embed_pack(
    pack_id: str,
    request: Request,
    body: EmbedPackRequest | None = None,
):
    """Kick off vector-index conversion for an installed ZIM pack.

    Install always lands ZIM-only (auto-embed-on-install was removed
    in the 2026-05-07 cleanup — see ``_run_install`` Stage 3). This
    endpoint is the sole opt-in path for adding a vector sidecar:
    fired by the per-pack "Embed for vector search" icon on the
    Browse landing card. Useful for frequently-queried packs where
    vector recall meaningfully complements the ZIM's built-in
    Xapian keyword index; not worth running for every pack.

    Returns 404 if the pack isn't installed or has no .zim source, 409 if
    it already has a vector index (use Discard + re-embed if you want to
    rebuild) or a job is already running for this pack, and 202 with a
    job_id on success. Track progress via /install/{job_id}/progress.

    Admin only — affects the shared pack library.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden

    mgr = _get_pack_manager(request)
    zp = mgr._zim_packs.get(pack_id)
    if zp is None:
        raise HTTPException(404, f"ZIM pack not installed: {pack_id}")
    # Refuse if an augpack already exists for this pack — the listing's
    # has_vector_index flag is the source of truth, but check the pack
    # connection too in case scan() hasn't run since the augpack landed.
    if pack_id in mgr._packs:
        raise HTTPException(409, f"Pack {pack_id} is already embedded")

    zim_path = Path(zp.path)
    if not zim_path.exists():
        raise HTTPException(404, f"ZIM file missing on disk: {zim_path}")

    augpack_path = zim_path.with_suffix(".augpack")
    if augpack_path.exists():
        # Stale .augpack on disk that PackManager hasn't picked up yet —
        # could be a partial from a crashed prior embed. Refuse rather
        # than silently overwriting; the user can Discard via the failed-
        # conversion endpoint and retry.
        raise HTTPException(
            409,
            f"Stale .augpack already on disk for {pack_id} — discard via "
            f"/api/knowledge/discard-failed/{pack_id} first.",
        )

    jobs: dict = getattr(request.app.state, "install_jobs", None) or {}
    for existing in jobs.values():
        if existing.catalog_id == pack_id and existing.status in ("running", "started", "pending"):
            raise HTTPException(409, f"Embed already running for {pack_id}")

    job_id = uuid.uuid4().hex[:12]
    job = InstallJob(
        job_id=job_id,
        catalog_id=pack_id,
        status="started",
        stage="embedding",
        current=0,
        total=zp.meta.chunk_count or 0,
        started_at=time.time(),
    )

    batch_size = (body.batch_size if body and body.batch_size else
                  settings.knowledge_embedding_batch_size)
    batch_size = max(1, min(2048, batch_size))

    task = asyncio.create_task(
        _run_embed_zim(
            job=job,
            zim_path=zim_path,
            augpack_path=augpack_path,
            pack_mgr=mgr,
            batch_size=batch_size,
        )
    )
    job.task = task
    jobs[job_id] = job
    request.app.state.install_jobs = jobs

    return {"job_id": job_id, "status": "started", "batch_size": batch_size}


@router.get("/registry")
async def get_registry():
    """Fetch available packs from the registry CDN."""
    url = settings.knowledge_registry_url
    if not url:
        return {"registry_version": 1, "packs": []}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        log.warning("knowledge_registry_fetch_failed", url=url, exc_info=True)
        return {"registry_version": 1, "packs": []}


@router.post("/download")
async def download_pack(body: DownloadRequest, request: Request):
    """Download a .augpack file from a URL with SSE progress streaming.

    Admin only — knowledge packs install to a shared directory and
    become visible to every tenant.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)

    # Basename the user-supplied name before it's joined onto pack_dir below —
    # a value like "../../etc/cron.d/x" would otherwise escape the pack dir on
    # write (admin-gated, but still a traversal). basename strips every
    # directory component; a name that was pure traversal collapses to "".
    import os.path as _osp
    filename = _osp.basename(str(body.filename or "").replace("\\", "/")).strip()
    if not filename or set(filename) <= {"."}:
        raise HTTPException(400, "Invalid pack filename")
    if not filename.endswith(".augpack"):
        filename += ".augpack"

    # SSRF guard: user-supplied URL must not target internal/private IPs.
    try:
        await check_ssrf(body.url)
    except SafeHttpError as exc:
        raise HTTPException(400, f"Invalid download URL: {exc}") from exc

    dest = Path(mgr.pack_dir) / filename
    if dest.exists():
        raise HTTPException(409, f"Pack file already exists: {filename}")

    # Pre-check disk space — HEAD request for Content-Length
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as head_client:
            head_resp = await head_client.head(body.url)
            expected_size = int(head_resp.headers.get("content-length", 0))
            if expected_size > 0:
                import shutil
                free = shutil.disk_usage(str(dest.parent)).free
                # Need ~1.1x the download size (file + temp overhead)
                needed = int(expected_size * 1.1)
                if free < needed:
                    free_gb = free / (1024 ** 3)
                    needed_gb = needed / (1024 ** 3)
                    raise HTTPException(
                        507,
                        f"Insufficient disk space: {free_gb:.1f}GB free, "
                        f"~{needed_gb:.1f}GB needed",
                    )
                log.info("disk_space_ok", free_gb=f"{free / (1024**3):.1f}",
                         needed_gb=f"{needed / (1024**3):.1f}")
    except httpx.HTTPError:
        pass  # HEAD failed — proceed without check

    async def _stream():
        # streamed_download writes to a `<dest>.part` temp and renames onto
        # `dest` only on success, cleaning the `.part` up on failure — so
        # `dest` itself never holds a partial file (and PackManager.scan only
        # picks up real `.augpack` files, never the `.part`).
        try:
            timeout = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=30.0)
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                downloaded, total, last_report, started = 0, None, 0, False
                # SSRF already checked above for this user-supplied URL.
                async for downloaded, total in streamed_download(
                    client, body.url, dest, chunk_size=64 * 1024, ssrf_check=False,
                ):
                    if not started:
                        started = True
                        yield f"data: {json.dumps({'type': 'start', 'total': total or 0})}\n\n"
                    elif downloaded - last_report >= 1_048_576:
                        last_report = downloaded
                        yield f"data: {json.dumps({'type': 'progress', 'downloaded': downloaded, 'total': total or 0})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'downloaded': downloaded})}\n\n"
                await mgr.scan()  # pick up the new pack
        except Exception as exc:
            log.warning("knowledge_download_failed", url=body.url, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/supported-formats")
async def supported_formats():
    """List file formats accepted for import."""
    from augmentum.knowledge.importer import ALL_SUPPORTED
    return {"formats": sorted(ALL_SUPPORTED)}


@router.post("/import")
async def import_pack(request: Request, file: UploadFile):
    """Import a file into the knowledge pack library.

    Accepts: .augpack (direct), .csv, .jsonl, .json, .sqlite, .db,
    .md, .txt, .pdf, .docx, .html, .epub, .zip (archive of any above).

    Admin only — imported packs are shared across all tenants.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden

    from augmentum.knowledge.importer import detect_format, import_to_augpack

    mgr = _get_pack_manager(request)
    filename = file.filename or "imported.augpack"
    fmt = detect_format(filename)
    if not fmt:
        raise HTTPException(
            400,
            f"Unsupported format: {Path(filename).suffix}. "
            f"Use GET /api/knowledge/supported-formats for accepted types.",
        )

    content = await file.read()

    # Output always uses .augpack extension
    out_name = Path(filename).stem + ".augpack"
    dest = Path(mgr.pack_dir) / out_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        raise HTTPException(409, f"Pack '{out_name}' already exists")

    try:
        stats = await import_to_augpack(
            file_data=content,
            filename=filename,
            output_path=dest,
            source="imported",
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        # Clean up partial file on failure
        dest.unlink(missing_ok=True)
        log.warning("knowledge_import_failed", filename=filename, exc_info=True)
        raise HTTPException(500, "Import failed — check server logs")

    await mgr.scan()
    log.info("knowledge_pack_imported", filename=out_name, **stats)
    return {"ok": True, "filename": out_name, **stats}


@router.post("/activate/{pack_id}")
async def activate_pack(pack_id: str, request: Request):
    """Activate a knowledge pack install-wide. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)
    if not await mgr.activate(pack_id):
        raise HTTPException(404, f"Pack not found: {pack_id}")
    system_events.publish("knowledge.changed", {"pack_id": pack_id, "reason": "activate"})
    return {"ok": True, "pack_id": pack_id, "active": True}


@router.post("/deactivate/{pack_id}")
async def deactivate_pack(pack_id: str, request: Request):
    """Deactivate a knowledge pack install-wide. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)
    if not await mgr.deactivate(pack_id):
        raise HTTPException(404, f"Pack not found: {pack_id}")
    system_events.publish("knowledge.changed", {"pack_id": pack_id, "reason": "deactivate"})
    return {"ok": True, "pack_id": pack_id, "active": False}


@router.delete("/{pack_id}")
async def delete_pack(pack_id: str, request: Request):
    """Delete a knowledge pack from the shared library. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)
    if not await mgr.delete(pack_id):
        raise HTTPException(404, f"Pack not found: {pack_id}")
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "knowledge", disk=True)
    system_events.publish("knowledge.changed", {"pack_id": pack_id, "reason": "delete"})
    return {"ok": True, "pack_id": pack_id}


@router.get("/search")
async def search_packs(
    request: Request,
    q: str,
    limit: int = 5,
    pack_ids: str = "",
):
    """Search active knowledge packs.

    Honors the same hybrid pipeline as in-chat injection: per-pack
    vector + FTS5 + ZIM keyword, RRF merge, optional rerank.

    Args:
        q: Free-text query.
        limit: Result cap (post-rerank).
        pack_ids: Comma-separated pack IDs to scope the search to. Empty
            value = search across all loaded packs (the original debug
            behavior). The Browse panel's per-pack home uses the filter
        so a search inside MDWiki doesn't pull in DevDocs results.
    """
    return await _do_pack_search(
        request,
        query=q,
        limit=limit,
        pack_ids_csv=pack_ids,
    )


@router.post("/search")
async def search_packs_post(request: Request):
    """POST variant of /search used by the fabric peer protocol.

    Accepts JSON ``{q, pack_ids: [str, ...], limit}``. Identical
    semantics to the GET endpoint above but with two operational
    advantages for peer-to-peer use:

      - The query never appears in the URL (and therefore not in
        access logs), which matters when a peer's search query may
        reflect private chat content.
      - The body bytes are covered by the fabric signed-request body
        SHA-256 (see peer_middleware.py), so the receiver can detect
        body tampering — impossible with the GET form where the
        query lives in the path.

    Same auth as the GET endpoint (user-auth, or fabric-peer auth
    when FabricPeerMiddleware has pre-populated ``scope["user"]``).
    """
    body = await request.json()
    query = str(body.get("q", "") or "").strip()
    if not query:
        raise HTTPException(400, "q is required")
    pack_ids_raw = body.get("pack_ids", [])
    if isinstance(pack_ids_raw, str):
        pack_ids_csv = pack_ids_raw
    elif isinstance(pack_ids_raw, list):
        pack_ids_csv = ",".join(str(p) for p in pack_ids_raw if p)
    else:
        pack_ids_csv = ""
    try:
        limit = int(body.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    return await _do_pack_search(
        request, query=query, limit=limit, pack_ids_csv=pack_ids_csv,
    )


async def _do_pack_search(
    request: Request, *, query: str, limit: int, pack_ids_csv: str,
) -> dict:
    """Shared search execution used by both GET and POST endpoints.

    Factored to keep the two route surfaces in sync as the pack-scope
    filtering logic evolves (e.g. when we add hybrid pack-format
    handling, only one place to update).

    Fabric fan-out: when ``pack_ids_csv`` names packs we don't have
    locally, we look up which connected peers advertise them and
    dispatch one signed search per peer (grouping packs by peer). Each
    peer only searches packs it actually has installed — the
    extractor's ``installed()`` listing is the source of truth. Local
    + peer results merge into one ordered list before return.
    """
    mgr = _get_pack_manager(request)
    available = list(mgr._packs.keys()) + list(mgr._zim_packs.keys())
    wanted_set: set[str] = set()
    if pack_ids_csv:
        wanted_set = {pid.strip() for pid in pack_ids_csv.split(",") if pid.strip()}
        # Silently drop unknown pack IDs rather than 400 — the UI may
        # have cached a list that's slightly out of date with what's
        # loaded, and a peer may request packs we don't actually have
        # (we advertise periodically; a pack uninstalled between
        # heartbeats would 404 here otherwise).
        target_ids = [pid for pid in available if pid in wanted_set]
    else:
        target_ids = available

    results = await mgr.search(
        query=query,
        pack_ids=target_ids,
        limit=limit,
        rerank=settings.reranker_enabled,
    )

    # Cross-peer fan-out: any wanted pack we don't have locally gets
    # routed to a peer that does. Skip entirely when the caller didn't
    # specify ``pack_ids`` (the "search everything local" path) — we
    # don't want to broadcast every unscoped search to every peer.
    peer_pack_ids: list[str] = []
    if wanted_set:
        missing = wanted_set - set(available)
        peer_results = await _fanout_remote_packs(
            request, query=query, missing_pack_ids=missing, limit=limit,
        )
        if peer_results:
            results.extend(peer_results)
            peer_pack_ids = sorted({
                str(r.get("pack_id", "")) for r in peer_results
                if r.get("pack_id")
            })

    return {
        "query": query,
        "pack_ids": target_ids + [p for p in peer_pack_ids if p not in target_ids],
        "results": [asdict(r) if not isinstance(r, dict) else r for r in results],
    }


async def _fanout_remote_packs(
    request: Request, *, query: str, missing_pack_ids: set[str], limit: int,
) -> list[dict]:
    """Dispatch a knowledge search to each connected peer that advertises
    the missing pack(s). Returns a flat list of result dicts (PackResult
    shape) — no merging beyond extension; the caller decides ordering.

    Failures are logged + dropped, never raised. Knowledge injection is
    background context; a slow peer shouldn't break the chat turn that
    triggered the search.
    """
    if not missing_pack_ids:
        return []

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    fabric_http = getattr(request.app.state, "fabric_http_client", None)
    if coordinator is None or fabric_http is None:
        return []

    from augmentum.fabric.capabilities import KIND_KNOWLEDGE_SEARCH

    try:
        matches = coordinator.find_peers_with_capability(KIND_KNOWLEDGE_SEARCH)
    except Exception:
        log.debug("fabric_knowledge_fanout_match_failed", exc_info=True)
        return []

    # Group requested packs by the peer that has them. First match wins
    # for any given pack (deterministic by coordinator iteration order);
    # future scoring could distribute load.
    peer_packs: dict[str, list[str]] = {}
    pack_seen: set[str] = set()
    for node_id, cap in matches:
        pack_id = getattr(cap, "pack_id", "") or ""
        if not pack_id or pack_id in pack_seen:
            continue
        if pack_id not in missing_pack_ids:
            continue
        pack_seen.add(pack_id)
        peer_packs.setdefault(node_id, []).append(pack_id)

    if not peer_packs:
        return []

    user = request.scope.get("user")
    user_id = user.id if user else ""
    if not user_id:
        log.debug("fabric_knowledge_fanout_no_user")
        return []

    identity = getattr(coordinator, "_identity", None)
    if identity is None:
        return []

    from augmentum.fabric.knowledge_client import (
        RemoteSearchError,
        search_remote_packs,
    )

    out: list[dict] = []
    for node_id, pack_ids in peer_packs.items():
        state = coordinator.peer_state(node_id)
        if state is None or state.paired is None or not state.connected:
            continue
        peer_addr = state.paired.addr
        try:
            peer_results = await search_remote_packs(
                http_client=fabric_http,
                identity=identity,
                user_id=user_id,
                peer_addr=peer_addr,
                query=query,
                pack_ids=pack_ids,
                limit=limit,
            )
        except RemoteSearchError as exc:
            log.info(
                "fabric_knowledge_fanout_peer_failed",
                peer_node_id=node_id, pack_ids=pack_ids,
                error=str(exc)[:200],
            )
            continue
        except Exception:
            log.warning(
                "fabric_knowledge_fanout_peer_unexpected",
                peer_node_id=node_id, exc_info=True,
            )
            continue
        # Annotate peer source on each result so downstream UI can show
        # the badge ("from peer X") and so dedup/merge logic can prefer
        # local on ties.
        for r in peer_results:
            if isinstance(r, dict):
                r.setdefault("fabric_peer_node_id", node_id)
                out.append(r)

    if out:
        log.info(
            "fabric_knowledge_fanout_completed",
            peer_count=len(peer_packs), result_count=len(out),
        )
    return out


# ------------------------------------------------------------------
# ZIM path resolution helpers
# ------------------------------------------------------------------
#
# libzim's ``Archive.get_entry_by_path`` is a strict lookup — no
# namespace fallback, no redirect following, no fallback for empty/"/"
# paths. The two helpers below wrap it with the resolution semantics
# every mature reader (libkiwix, kiwix-js, kiwix-android) implements,
# so a single call from the route handler covers:
#
#   * Empty / "/" paths   → archive's main entry (libkiwix
#                           ``archiveTools.cpp::getEntryFromPath``).
#   * Type-0 vs Type-1 namespace split: try the path as-is, then
#                           ``A/<path>`` (legacy article namespace),
#                           then ``C/<path>`` (Type-1 unified). Only
#                           kicks in for paths that don't already
#                           carry a known namespace prefix, so
#                           ``A/Foo`` doesn't become ``A/A/Foo``.
#   * Redirects: follow up to ``_ZIM_REDIRECT_MAX_HOPS`` hops with
#                cycle detection. libzim exposes ``is_redirect`` +
#                ``get_redirect_entry`` per-step rather than a
#                follow-on flag, so we walk the chain ourselves.
#
# Single character ZIM namespaces (legacy spec). Type-1 archives
# usually expose entries directly under no namespace prefix at all,
# but mwoffliner-built Type-1 ZIMs still emit ``A/`` for back-compat.
# The set covers everything legal: A (articles), C (content, Type-1),
# H (aliases), I (images), J (scripts), M (metadata), U/V (unknown
# legacy), W (well-known, Type-1), X (search index), and ``-`` (raw
# assets in some scrapers).
_NAMESPACE_PREFIX_RE = re.compile(r"^[ABCHIJMUVWX-]/")
_ZIM_REDIRECT_MAX_HOPS = 5
# Hard cap on entry-path length. Wikipedia article paths run ~200
# chars at the long tail; nothing legitimate gets close to 4 KiB.
# Anything longer is fuzzing or a malformed citation.
_ZIM_PATH_MAX_LEN = 4096


def _validate_zim_path(path: str) -> str | None:
    """Sanitize a user-supplied ZIM entry path before resolution.

    libzim treats paths as opaque archive keys (it never walks the host
    filesystem), so the real risk isn't escape — it's probe traffic
    flooding our resolver and 404 logs at scale. Reject obviously
    hostile shapes early so we can return 400 cheaply, log once at
    INFO level, and never enter the resolver.

    Rejects:
        * Length > _ZIM_PATH_MAX_LEN (DoS guard).
        * NUL or any C0 control character (0x00-0x1f) or DEL (0x7f).
        * Leading ``/`` or ``\\`` (absolute-path probes).
        * Any ``..`` path segment, in either separator orientation
          (``foo/../bar``, ``foo\\..\\bar``).

    Returns the path unchanged on success, ``None`` on rejection.
    Empty / ``/`` paths pass through — the resolver maps those to
    ``archive.main_entry``.
    """
    if not path:
        return path
    if len(path) > _ZIM_PATH_MAX_LEN:
        return None
    # Single linear scan: control-char + NUL check. Visible printables
    # only; tab (0x09) and below all rejected because legitimate ZIM
    # entry names don't contain them and a passing test would let
    # newline-injection through to the (server-side) resolver logs.
    for ch in path:
        if ord(ch) < 0x20 or ord(ch) == 0x7f:
            return None
    if path[0] in ("/", "\\"):
        return None
    # Normalize separators before splitting so ``foo\\..\\bar`` is
    # caught even on POSIX (libzim accepts forward slashes only, but
    # an attacker may try the back-slash variant).
    if any(seg == ".." for seg in path.replace("\\", "/").split("/")):
        return None
    return path


# Stream chunk size for non-HTML entries. 64 KiB matches what
# Starlette's FileResponse uses internally and gives ~16 chunks/sec
# at LTE speeds — fast enough that a paused video shows progress, big
# enough that the per-chunk overhead is amortized.
_ZIM_STREAM_CHUNK = 64 * 1024


# When a ZIM (or its scraper) misreports an entry's mimetype as
# ``application/octet-stream`` or omits it entirely, fall back to
# extension-based detection. Real cases:
#   * Gutenberg books — book.pdf / book.epub stored as octet-stream
#   * FANDOM scrapes — image entries occasionally typed as octet-stream
#   * Legacy ZIMs — SVG icons served as octet-stream
# Without this override the browser surfaces a download prompt instead
# of inline-rendering. Table lifted from kiwix-js's ``uiUtil.js`` file
# extension table; entries without a clear MIME map fall back to the
# original ``application/octet-stream``.
_MIME_BY_EXT: dict[str, str] = {
    "html": "text/html",
    "htm": "text/html",
    "css": "text/css",
    "js": "application/javascript",
    "mjs": "application/javascript",
    "json": "application/json",
    "xml": "application/xml",
    "txt": "text/plain",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "avif": "image/avif",
    "ico": "image/x-icon",
    "pdf": "application/pdf",
    "epub": "application/epub+zip",
    "mobi": "application/x-mobipocket-ebook",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp4": "video/mp4",
    "m4v": "video/x-m4v",
    "webm": "video/webm",
    "ogv": "video/ogg",
    "mov": "video/quicktime",
    "wasm": "application/wasm",
    "woff": "font/woff",
    "woff2": "font/woff2",
    "ttf": "font/ttf",
    "otf": "font/otf",
}


def _refine_mimetype(mimetype: str, path: str) -> str:
    """Override ``application/octet-stream`` (or empty) with an
    extension-based MIME guess. Other mimetypes pass through unchanged.

    Only triggers on the catch-all case so a legitimate
    ``application/x-foo`` from a niche scraper isn't second-guessed.
    Case-insensitive on the file extension; unknown extensions return
    the original ``mimetype`` (or ``application/octet-stream`` if the
    input was empty).
    """
    if mimetype and mimetype != "application/octet-stream":
        return mimetype
    dot = path.rfind(".")
    if dot < 0:
        return mimetype or "application/octet-stream"
    ext = path[dot + 1:].lower()
    return _MIME_BY_EXT.get(ext, mimetype or "application/octet-stream")


# Cache-Control windows. HTML responses get the shorter window so a
# pack reinstall reflects fast in active browsers; long-lived assets
# (images, CSS, fonts, video) get the longer window because their
# pack-internal URLs are stable for the install's lifetime.
_CACHE_HTML = "public, max-age=300"      # 5 minutes
_CACHE_ASSET = "public, max-age=86400"   # 24 hours


def _zim_cache_control(path: str) -> str:
    """Predict Cache-Control before reading the item, so the ETag
    short-circuit (304) can serve a freshness header that matches
    what a 200 would carry. Conservative: extension-less paths
    (Wikipedia article shape ``A/Foo``) and ``.html``/``.htm`` get
    the short HTML window; everything else gets the asset window.
    Aligns with the post-read mimetype branch — refinement on
    ``application/octet-stream`` doesn't change the cache decision
    because both branches produce the same window for HTML.
    """
    last_seg = path.rsplit("/", 1)[-1] if "/" in path else path
    if "." not in last_seg:
        return _CACHE_HTML
    ext = last_seg.rsplit(".", 1)[-1].lower()
    if ext in ("html", "htm"):
        return _CACHE_HTML
    return _CACHE_ASSET


def _zim_etag(
    pack_id: str,
    canonical_path: str,
    build_date: str,
    mtime_ns: int,
    theme: str,
    family: str = "generic",
    reader_mode: bool = True,
) -> str:
    """Compute a strong ETag for a ZIM entry response.

    Seed components, all of which can change response bytes:
        * ``pack_id`` + ``canonical_path`` — identifies the entry.
        * ``build_date`` + ``mtime_ns`` — changes on pack reinstall.
        * ``theme`` — HTML responses are theme-templated by the
          rewriter, so the same path with ``?theme=dark`` vs
          ``?theme=light`` produces different bytes.
        * ``family`` — per-family CSS pack is part of the response
          body, so a re-classification (or detector tweak) must bust
          the cache.
        * ``reader_mode`` — raw-mode responses skip CSS + neutralizer,
          producing a different byte stream than themed responses.

    Returned as an RFC 7232 strong entity-tag (quoted hex). 16 hex
    chars = 64 bits of collision space, more than enough — at our
    scale collision risk is negligible vs the chance of a false
    cache hit, which the seed components rule out.
    """
    seed = (
        f"{pack_id}\x00{canonical_path}\x00{build_date}\x00{mtime_ns}\x00"
        f"{theme}\x00{family}\x00{int(bool(reader_mode))}"
    ).encode()
    return '"' + hashlib.sha256(seed).hexdigest()[:16] + '"'


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Test an ``If-None-Match`` header against our computed ETag.

    Handles RFC 7232 forms:
        * ``*`` — always matches when the resource exists.
        * Single tag — direct comparison.
        * Comma-separated list — match any.
        * Weak prefix ``W/`` — stripped before comparison; we issue
          strong tags but accept weak inputs (some proxies downgrade).
    """
    if not if_none_match:
        return False
    s = if_none_match.strip()
    if s == "*":
        return True
    for token in s.split(","):
        t = token.strip()
        if t.startswith("W/"):
            t = t[2:].strip()
        if t == etag:
            return True
    return False


def _parse_range_header(header: str | None, total: int) -> tuple[int, int, int]:
    """Parse an HTTP ``Range`` header against a known total length.

    RFC 7233 byte-range support, single-range only (multipart range
    responses are optional and rare in practice; browsers re-issue
    single ranges instead). Returns ``(start, end_exclusive, status)``
    where status is:

        * ``200`` — no header / unparseable / non-bytes unit. Caller
          should serve the full body. Honoring "best-effort" here is
          per the RFC: an unparseable Range MUST be ignored.
        * ``206`` — valid range, in bounds. Caller serves
          ``[start:end_exclusive)`` and emits ``Content-Range``.
        * ``416`` — syntactically valid but out of bounds. Caller
          replies 416 with ``Content-Range: bytes */{total}``.

    Handles the three RFC 7233 forms: ``bytes=N-M`` (closed),
    ``bytes=N-`` (open-ended), and ``bytes=-N`` (suffix length).
    """
    if not header:
        return 0, total, 200
    if not header.lower().startswith("bytes="):
        return 0, total, 200
    # Take the first range only — the rest of the spec ranges (if any)
    # are dropped. Browsers rarely emit multi-range; servers serving a
    # single range are RFC-compliant.
    spec = header[6:].split(",", 1)[0].strip()
    if "-" not in spec:
        return 0, total, 200
    s_part, _, e_part = spec.partition("-")
    s_part, e_part = s_part.strip(), e_part.strip()
    try:
        if not s_part and e_part:
            # Suffix form: -500 → last 500 bytes. If suffix > total,
            # serve the whole thing (RFC 7233 §2.1: a suffix length
            # larger than the representation length serves the entire
            # representation).
            suffix = int(e_part)
            if suffix <= 0:
                return 0, 0, 416
            start = max(0, total - suffix)
            end = total
        elif s_part and not e_part:
            # Open-ended: 500- → from byte 500 to EOF
            start = int(s_part)
            end = total
        elif s_part and e_part:
            # Closed range: 0-499 → bytes [0,500). The end byte is
            # inclusive in the header; convert to Python exclusive.
            start = int(s_part)
            end = int(e_part) + 1
        else:
            return 0, total, 200
    except ValueError:
        return 0, total, 200

    if start < 0 or start >= total or end > total or start >= end:
        return 0, 0, 416
    return start, end, 206


def _follow_zim_redirects(entry):
    """Walk a ZIM redirect chain to its terminal entry.

    Caps at ``_ZIM_REDIRECT_MAX_HOPS`` and breaks on cycles so a
    malformed ZIM (or one with a self-referential redirect) can't
    stall the request. Returns the terminal entry on success, or
    ``None`` if the chain is broken / too deep / cyclic.

    Mirrors the semantics of libkiwix's ``getFinalItem`` (which calls
    ``Entry::getItem(true)`` on the C++ side) — we walk explicitly
    because the libzim Python binding doesn't expose that flag.
    """
    seen: set[str] = set()
    for _ in range(_ZIM_REDIRECT_MAX_HOPS + 1):
        if not getattr(entry, "is_redirect", False):
            return entry
        try:
            current = entry.path
        except Exception:
            return None
        if current in seen:
            log.warning("zim_redirect_cycle", path=current)
            return None
        seen.add(current)
        try:
            entry = entry.get_redirect_entry()
        except Exception:
            log.warning("zim_redirect_chain_broken", path=current, exc_info=True)
            return None
    log.warning("zim_redirect_too_deep", max_hops=_ZIM_REDIRECT_MAX_HOPS)
    return None


def _resolve_zim_path(archive, path: str):
    """Resolve ``path`` against ``archive``, with namespace + redirect fallback.

    Resolution order:
        1. Empty / "/" path → ``archive.main_entry``.
        2. ``path`` as-is (Type-1 unified namespace, or already-prefixed
           Type-0 like ``A/Foo`` / ``I/img.png``).
        3. ``A/<path>`` — Type-0 article namespace fallback. Skipped
           when ``path`` already starts with a known namespace, so
           ``A/Foo`` does not become ``A/A/Foo``.
        4. ``C/<path>`` — Type-1 unified namespace fallback. Same skip
           rule as above.

    Each candidate is run through ``_follow_zim_redirects`` before being
    accepted, so we never return a redirect-stub entry. Returns the
    terminal entry on success, ``None`` if no candidate resolves.
    """
    if not path or path == "/":
        try:
            return _follow_zim_redirects(archive.main_entry)
        except Exception:
            log.warning("zim_main_entry_unavailable", exc_info=True)
            return None

    candidates: list[str] = [path]
    if not _NAMESPACE_PREFIX_RE.match(path):
        candidates.append(f"A/{path}")
        candidates.append(f"C/{path}")

    for cand in candidates:
        try:
            entry = archive.get_entry_by_path(cand)
        except Exception as exc:
            log.debug("zim_entry_resolve_failed", path=cand, error=str(exc))
            continue
        resolved = _follow_zim_redirects(entry)
        if resolved is not None:
            return resolved
    return None


# ------------------------------------------------------------------
# ZIM article serving (Browse integration)
# ------------------------------------------------------------------
#
# Serves individual ZIM entries (HTML articles, images, CSS) so the
# Browse panel can render them in a sandboxed iframe. Citations from
# chat link directly here. The Browse subsystem treats these like any
# other URL — fetch, render in reader, history-track — except the
# extraction pipeline is skipped because the content is already clean.
#
# Auth: standard user auth required. Knowledge packs are admin-installed
# and shared across all tenants today (per the existing model in
# activate_pack/delete_pack), so any logged-in user can read entries
# from any active pack.
#
# Safety: the response carries a strict CSP that bans script execution
# inside the iframe and the iframe itself runs sandbox="allow-same-origin"
# only — no script, no top-level navigation. ZIM articles can contain
# arbitrary HTML/JS from upstream (Wikipedia, MDWiki, etc.); we trust the
# upstream sources but not at the level of letting their JS run inside
# the user's logged-in session.


@router.get("/zim/{pack_id}/_suggest")
async def suggest_zim_entries(
    pack_id: str,
    request: Request,
    q: str = "",
    limit: int = 8,
):
    """Typeahead suggestions for a single ZIM pack.

    Drives the pack-scoped search input in the browse panel: each
    keystroke fires this endpoint debounced and renders matching
    titles in a dropdown. Powered by libzim's ``SuggestionSearcher``
    when available, with a full-text-search fallback.

    Anti-abuse / scaling constraints:

        * **Min 2 characters.** Single-char queries return empty
          to prevent enumeration via repeated 1-char probes and
          to avoid burning suggester cycles on noise keystrokes.
        * **Max 64 characters.** The suggester returns garbage on
          long phrases; truncation keeps the work bounded.
        * **Limit clamped to [1, 20].** UI default is 8 (fits the
          dropdown); 20 is the cap for power-user contexts.

    The handler never raises on suggester failure — empty list is
    the universal fallback so the typeahead silently degrades
    rather than breaking the search input.

    The path ``_suggest`` is intentionally underscore-prefixed so
    it can never collide with a real ZIM entry name (entries don't
    start with ``_`` in practice) and so the more-specific literal
    route wins matching against ``/zim/{pack_id}/{path:path}``.
    """
    q = (q or "").strip()
    if len(q) < 2:
        return {"pack_id": pack_id, "query": q, "suggestions": []}
    if len(q) > 64:
        q = q[:64]
    limit = max(1, min(limit, 20))

    mgr = _get_pack_manager(request)
    zp = mgr._zim_packs.get(pack_id)
    if zp is None or not zp.active:
        raise HTTPException(404, f"ZIM pack not found: {pack_id}")

    try:
        suggestions = await zp.reader.suggest(q, limit=limit)
    except Exception:
        log.warning(
            "zim_suggest_route_failed", pack_id=pack_id, q=q, exc_info=True,
        )
        suggestions = []

    return {
        "pack_id": pack_id,
        "query": q,
        "suggestions": [
            {"title": s.title, "path": s.path} for s in suggestions
        ],
    }


def _zim_pack_or_404(request: Request, pack_id: str):
    """Resolve a ZIM pack record or raise 404. Shared by the underscore
    sub-routes (_meta, _illustration, _random) so each one carries the
    same three-step guard the main entry route uses.
    """
    mgr = _get_pack_manager(request)
    zp = mgr._zim_packs.get(pack_id)
    if zp is None:
        raise HTTPException(404, f"ZIM pack not found: {pack_id}")
    if not zp.active:
        raise HTTPException(404, f"ZIM pack inactive: {pack_id}")
    if getattr(zp.reader, "_archive", None) is None:
        raise HTTPException(404, f"ZIM archive unavailable: {pack_id}")
    return zp


# Minimal ISO 639-1 → display-name table for ZIM ``Language`` metadata.
# ZIMs ship ISO 639-3 codes (3-letter) most of the time, occasionally 1
# (2-letter), sometimes locale tags ("en-US"). This covers the common
# packs Augmentum onboards — Wikipedia, MDWiki, Wikivoyage, Wiktionary,
# Stack Exchange, DevDocs, Project Gutenberg — without dragging in a
# pycountry-scale dependency. Unknown codes fall through with the code
# echoed as the display name.
_LANG_NAMES = {
    "en": "English", "eng": "English",
    "es": "Spanish", "spa": "Spanish",
    "fr": "French",  "fra": "French", "fre": "French",
    "de": "German",  "deu": "German", "ger": "German",
    "it": "Italian", "ita": "Italian",
    "pt": "Portuguese", "por": "Portuguese",
    "ru": "Russian", "rus": "Russian",
    "zh": "Chinese", "zho": "Chinese", "chi": "Chinese",
    "ja": "Japanese", "jpn": "Japanese",
    "ko": "Korean",  "kor": "Korean",
    "ar": "Arabic",  "ara": "Arabic",
    "hi": "Hindi",   "hin": "Hindi",
    "nl": "Dutch",   "nld": "Dutch", "dut": "Dutch",
    "pl": "Polish",  "pol": "Polish",
    "tr": "Turkish", "tur": "Turkish",
    "sv": "Swedish", "swe": "Swedish",
    "uk": "Ukrainian", "ukr": "Ukrainian",
    "vi": "Vietnamese", "vie": "Vietnamese",
    "id": "Indonesian", "ind": "Indonesian",
    "fa": "Persian", "fas": "Persian", "per": "Persian",
    "he": "Hebrew",  "heb": "Hebrew",
    "el": "Greek",   "ell": "Greek", "gre": "Greek",
    "cs": "Czech",   "ces": "Czech", "cze": "Czech",
    "ro": "Romanian", "ron": "Romanian", "rum": "Romanian",
    "hu": "Hungarian", "hun": "Hungarian",
    "th": "Thai",    "tha": "Thai",
    "da": "Danish",  "dan": "Danish",
    "fi": "Finnish", "fin": "Finnish",
    "no": "Norwegian", "nor": "Norwegian",
    "ca": "Catalan", "cat": "Catalan",
    "bg": "Bulgarian", "bul": "Bulgarian",
    "sr": "Serbian", "srp": "Serbian",
    "hr": "Croatian", "hrv": "Croatian",
    "sk": "Slovak",  "slk": "Slovak", "slo": "Slovak",
    "ms": "Malay",   "msa": "Malay", "may": "Malay",
}


def _resolve_language(value: str) -> dict[str, str]:
    """ZIM ``Language`` metadata → ``{code, name}`` for UI badge rendering.

    ZIMs sometimes ship multiple codes joined by commas (multilingual packs).
    We surface the first non-empty token so the badge stays terse; the full
    string is still available on the raw metadata field for callers that
    need it.
    """
    if not value:
        return {"code": "", "name": ""}
    primary = value.split(",")[0].strip().split(";")[0].strip()
    if not primary:
        return {"code": "", "name": ""}
    key = primary.lower().replace("_", "-").split("-")[0]
    return {"code": primary, "name": _LANG_NAMES.get(key, primary)}


@router.get("/zim/{pack_id}/_meta")
async def serve_zim_metadata(pack_id: str, request: Request):
    """Full ZIM-archive metadata for the browse panel sidebar.

    Returns the raw libzim metadata keys (Title, Description, Creator,
    Publisher, Date, Tags, Name, Flavour, etc.) plus two derived fields:

    * ``language`` — ``{code, name}`` resolved from the raw ``Language``
      string against the in-process ISO table. Powers the UI's language
      badge without forcing the frontend to ship its own lookup.
    * ``tags`` — array form of the semicolon-joined ``Tags`` string.
      Easier to render as chips than splitting client-side.

    Cached aggressively via ETag (pack mtime + build date) — ZIM metadata
    is immutable for the life of a given pack file, so revalidation is
    pure win.
    """
    zp = _zim_pack_or_404(request, pack_id)

    try:
        mtime_ns = zp.path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    build_date = getattr(zp.meta, "build_date", "") or ""
    etag = '"' + hashlib.sha1(
        f"meta:{pack_id}:{build_date}:{mtime_ns}".encode("utf-8"),
    ).hexdigest()[:16] + '"'
    cache_control = "private, max-age=3600"

    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache_control})

    raw = await asyncio.to_thread(zp.reader.get_metadata_full)

    # Derived fields for UI ergonomics. Don't shadow the raw values — the
    # sidebar may want both ("Language: en (English)" + ISO badge).
    payload: dict[str, object] = dict(raw)
    payload["language"] = _resolve_language(str(raw.get("Language", "")))
    tags_raw = str(raw.get("Tags", ""))
    payload["tags"] = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []

    return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": cache_control})


@router.get("/zim/{pack_id}/_illustration")
async def serve_zim_illustration(pack_id: str, request: Request, size: int = 48):
    """Per-pack favicon / illustration bytes, sized at the libzim layer.

    libzim picks the closest-rendered illustration for the requested size;
    we don't resize ourselves. Sizes Augmentum uses: 48 (pack-card list)
    and 96 (active-pack header, retina).

    Returns 404 when the pack has no illustration entry — common on older
    ZIMs. Callers MUST render a fallback (letter glyph) rather than show
    a broken-image icon.
    """
    zp = _zim_pack_or_404(request, pack_id)

    # Clamp to a sane range. libzim will accept anything but typical
    # ZIM illustrations ship 48/96 only, and very-large requests just
    # waste decode work on a pixelated upscale.
    size = max(16, min(size, 512))

    try:
        mtime_ns = zp.path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    etag = '"' + hashlib.sha1(
        f"illu:{pack_id}:{size}:{mtime_ns}".encode("utf-8"),
    ).hexdigest()[:16] + '"'
    # Illustrations are baked into the ZIM and never change for a given
    # pack mtime — immutable cache is safe and shaves a round-trip on
    # every browse-panel pack-list re-render.
    cache_control = "public, max-age=2592000, immutable"

    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache_control})

    result = await asyncio.to_thread(zp.reader.get_illustration, size)
    if result is None:
        raise HTTPException(404, f"ZIM pack has no illustration: {pack_id}")
    content, mimetype = result
    return Response(
        content=content,
        media_type=mimetype or "image/png",
        headers={"ETag": etag, "Cache-Control": cache_control},
    )


@router.get("/zim/{pack_id}/_random")
async def serve_zim_random(
    pack_id: str,
    request: Request,
    theme: str = "dark",
    reader: str = "on",
):
    """302-redirect to a random HTML article in this pack.

    ``no-store`` cache header is non-negotiable: caching defeats the whole
    point of "random". The redirect carries ``theme`` (and ``reader=off``
    when explicitly raw) through so the article loads with the user's
    chosen palette on first paint.

    503 with ``Retry-After: 1`` rather than 404 when libzim returns no
    valid candidate — packs with HTML articles can momentarily fail the
    filter loop (rare; ``get_random_article`` retries internally) and
    returning a retry-friendly status lets a UI dice button "try again"
    cleanly without surfacing a hard error.
    """
    zp = _zim_pack_or_404(request, pack_id)

    target_path = await asyncio.to_thread(zp.reader.get_random_article)
    if target_path is None:
        # 503 vs 404: 404 implies "this pack has no random possible ever",
        # which we can't actually prove from one sample. 503 + Retry-After
        # reflects the truth: try again, the next draw might succeed.
        return Response(
            status_code=503,
            headers={"Retry-After": "1", "Cache-Control": "no-store"},
            content=f"No random article available for pack {pack_id}",
        )

    target = f"/api/knowledge/zim/{pack_id}/{quote(target_path, safe='/')}"
    params = []
    if theme:
        params.append(f"theme={quote(theme, safe='')}")
    if (reader or "on").lower() in ("off", "raw", "0", "false", "no"):
        params.append("reader=off")
    if params:
        target += "?" + "&".join(params)

    return RedirectResponse(
        url=target,
        status_code=302,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/zim/{pack_id}")
async def serve_zim_main_entry(
    pack_id: str,
    request: Request,
    theme: str = "dark",
    reader: str = "on",
):
    """Resolve a pack's main entry and 302 to its canonical URL.

    Mirrors libkiwix's ``/content/{book}`` → ``/content/{book}/{mainPath}``
    redirect (``internalServer.cpp:1490+``). Lets users — and chat
    citations that don't know the main page path — deep-link to a pack
    by ID alone. The redirect carries the ``theme`` query through so
    the iframe's reader-mode palette stays consistent on first load.

    404 cases mirror ``serve_zim_entry``: missing pack, inactive pack,
    archive failed to open, or pack has no resolvable main entry
    (rare, but legal per the ZIM spec).
    """
    # Content isolation: when this request arrived on the isolated
    # origin, redeem the preview token / validate the preview-session
    # cookie before serving. On the main origin this is a no-op (the
    # helper returns None) so existing same-origin loads stay unchanged.
    from augmentum.proxy.content_isolation_routes import check_content_isolated_auth
    auth_response = await check_content_isolated_auth(
        request, kind="knowledge_pack", resource_id=pack_id,
    )
    if auth_response is not None:
        return auth_response

    mgr = _get_pack_manager(request)
    zp = mgr._zim_packs.get(pack_id)
    if zp is None:
        raise HTTPException(404, f"ZIM pack not found: {pack_id}")
    if not zp.active:
        raise HTTPException(404, f"ZIM pack inactive: {pack_id}")

    archive = getattr(zp.reader, "_archive", None)
    if archive is None:
        raise HTTPException(404, f"ZIM archive unavailable: {pack_id}")

    main = await asyncio.to_thread(_resolve_zim_path, archive, "")
    if main is None or not getattr(main, "path", None):
        raise HTTPException(404, f"ZIM pack has no main entry: {pack_id}")

    target = f"/api/knowledge/zim/{pack_id}/{quote(main.path, safe='/')}"
    # ``safe=''`` so the value is fully encoded — theme names are
    # short ASCII slugs today but defensive against future palette
    # additions that might use non-ASCII labels. Reader flag pinned
    # to ``off`` when explicitly raw-requested so the redirect stays
    # raw; otherwise the param is dropped (the entry handler defaults
    # to on, so omitting it keeps the URL shorter).
    params = []
    if theme:
        params.append(f"theme={quote(theme, safe='')}")
    if (reader or "on").lower() in ("off", "raw", "0", "false", "no"):
        params.append("reader=off")
    if params:
        target += "?" + "&".join(params)
    return RedirectResponse(url=target, status_code=302)


@router.get("/zim/{pack_id}/{path:path}")
async def serve_zim_entry(
    pack_id: str,
    path: str,
    request: Request,
    theme: str = "dark",
    reader: str = "on",
):
    """Serve a single entry from a ZIM-backed knowledge pack.

    HTML responses go through ``_rewrite_zim_html`` so internal links
    point back at this route — iframe navigation stays inside the ZIM
    instead of escaping to the live web. Other mimetypes (images, CSS,
    fonts) pass through with their declared Content-Type.

    404 cases:
        - Pack not loaded (deleted, never installed)
        - Pack inactive
        - Entry path doesn't exist in the ZIM
        - Pack is not a ZIM (augpacks have no equivalent browseable shape)
    """
    # Content isolation handoff: on the isolated origin, validate the
    # preview-session cookie (or redeem the one-shot ?_pvt token on
    # first request). On the main origin this is a no-op.
    from augmentum.proxy.content_isolation_routes import check_content_isolated_auth
    auth_response = await check_content_isolated_auth(
        request, kind="knowledge_pack", resource_id=pack_id,
    )
    if auth_response is not None:
        return auth_response

    mgr = _get_pack_manager(request)
    zp = mgr._zim_packs.get(pack_id)
    if zp is None:
        raise HTTPException(404, f"ZIM pack not found: {pack_id}")
    if not zp.active:
        raise HTTPException(404, f"ZIM pack inactive: {pack_id}")

    archive = getattr(zp.reader, "_archive", None)
    if archive is None:
        raise HTTPException(404, f"ZIM archive unavailable: {pack_id}")

    # Defense in depth: filter obviously hostile path shapes BEFORE the
    # resolver burns work on them. libzim is opaque-path-safe, but at
    # scale we'd rather log probes once and reject with 400 than 404
    # silently and lose the signal. INFO so it shows in operational
    # dashboards without spamming WARN on benign typos.
    safe_path = _validate_zim_path(path)
    if safe_path is None:
        log.info(
            "zim_path_rejected",
            pack_id=pack_id,
            client=request.client.host if request.client else None,
        )
        raise HTTPException(400, "Invalid ZIM entry path")
    path = safe_path

    # Resolution and content read are split into two thread hops so the
    # canonical-path 302 short-circuits before we materialize a (possibly
    # large) item body that's about to be discarded. Each call to libzim
    # may touch disk on cache miss, so neither belongs on the event loop.
    resolved = await asyncio.to_thread(_resolve_zim_path, archive, path)
    if resolved is None:
        raise HTTPException(404, f"Entry not found in pack: {path}")

    # 302 to the canonical path when resolution rewrote the URL — namespace
    # fallback (``Foo`` → ``A/Foo``), main-entry fallback (``""`` → real
    # path), and redirect resolution all surface here. Mirrors libkiwix's
    # ``internalServer.cpp:1490-1500``: the iframe address bar tracks the
    # entry's true path, so the in-page ``<base>`` and any ``url(...)``
    # references in the served HTML resolve against the right URL. Without
    # this, relative links inside the article silently break.
    canonical_path = getattr(resolved, "path", path)
    if canonical_path and canonical_path != path:
        return RedirectResponse(
            url=f"/api/knowledge/zim/{pack_id}/{quote(canonical_path, safe='/')}",
            status_code=302,
        )

    # ETag / If-None-Match short-circuit. Computing this BEFORE
    # ``_read_item`` lets a 304 skip the libzim entry materialization
    # entirely — at scale, the 304 fast-path is the difference between
    # serving a cached browser its tiny revalidation reply and re-
    # decompressing a 50 KB Wikipedia article on every back/forward.
    # The seed pulls in ``build_date`` (pack metadata) + ``mtime_ns``
    # (file stat) so reinstalls bust the cache without manual purges.
    try:
        mtime_ns = zp.path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    build_date = getattr(zp.meta, "build_date", "") or ""
    # Reader mode is on by default; users can opt into raw HTML via
    # ``?reader=off`` (or ``raw``, ``0``, ``false`` — same intent, varied
    # spelling so the URL is forgiving). Anything else = on.
    reader_mode = (reader or "on").lower() not in ("off", "raw", "0", "false", "no")
    family = _skin_family_for_pack(pack_id)
    etag = _zim_etag(
        pack_id, canonical_path, build_date, mtime_ns, theme,
        family=family, reader_mode=reader_mode,
    )
    cache_control = _zim_cache_control(canonical_path)

    if _etag_matches(request.headers.get("if-none-match"), etag):
        # RFC 7232 §4.1: 304 MUST carry the same Cache-Control, ETag,
        # Date, Vary, Expires headers a 200 would. Date/Server are
        # added by Starlette; we contribute Cache-Control + ETag.
        # No body, no Content-Type — 304 has no payload.
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": cache_control},
        )

    def _read_item():
        """Resolve the item lazily — keep the Item live so its buffer
        backs the memoryview we return. ``content_view`` is a libzim
        memoryview (zero-copy view into the item's decompressed buffer);
        the streaming path slices it without materializing the whole
        entry. ``bytes()`` is only called on per-chunk slices.
        """
        try:
            item = resolved.get_item()
            return item, item.mimetype, item.content
        except Exception:
            log.warning(
                "zim_item_read_error",
                pack_id=pack_id,
                path=path,
                exc_info=True,
            )
            return None, None, None

    item, mimetype, content_view = await asyncio.to_thread(_read_item)
    if item is None or content_view is None or mimetype is None:
        raise HTTPException(404, f"Entry content unreadable: {path}")

    # Refine misreported octet-stream from extension (Gutenberg books,
    # FANDOM image typings, legacy SVG entries). Pure passthrough for
    # any legitimate mimetype — see _refine_mimetype docstring.
    mimetype = _refine_mimetype(mimetype, path)

    # CSP: scoped to this origin + data:/blob: for inline + dynamic content.
    # Scripts ARE allowed because modern ZIMs ship as SPAs (freeCodeCamp,
    # Stack Exchange, DevDocs use Vite/React/Webpack bundles that render
    # into a #app root at runtime — without scripts the user sees a blank
    # page or a "JS is disabled" noscript fallback). 'unsafe-inline' covers
    # the inline <script> blocks ZIMs use for boot/config; 'unsafe-eval'
    # covers SPAs that use new Function() or dynamic import. worker-src
    # blob: covers SPAs that spawn web workers from in-bundle code.
    #
    # Threat model: ZIM content is user-trusted (catalog packs come from
    # kiwix.org; user-imported packs were explicitly downloaded). The
    # iframe is same-origin so a malicious ZIM script could read the
    # user's session cookies — this is the same trust the user already
    # placed on the original site (wikipedia.org, freecodecamp.org, etc.)
    # when they chose to mirror it. frame-ancestors 'self' still blocks
    # outside pages from embedding our content.
    #
    # Cache-Control is ``public`` (not ``private``) deliberately:
    # knowledge packs are server-scoped, NOT user-scoped (deliberately
    # absent from CLAUDE.md's user_id-scoped table list), and pack
    # content is content-addressed-immutable until reinstall. A shared
    # cache or CDN serving the same bytes to two users is correct.
    safety_headers = {
        "Content-Security-Policy": (
            "default-src 'self' data: blob:; "
            "img-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "worker-src 'self' blob:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'self'"
        ),
        "X-Content-Type-Options": "nosniff",
        # Cache-Control predicted from path before _read_item so 200
        # and 304 responses agree on freshness. ETag carried alongside
        # so browsers can revalidate cheaply on next visit.
        "Cache-Control": cache_control,
        "ETag": etag,
    }

    # HTML branch: buffer + rewrite. The link rewriter needs the whole
    # document, real ZIM articles run sub-megabyte, and browsers don't
    # seek with Range into an HTML response — so streaming buys nothing
    # here. Materialize once, rewrite, return.
    if mimetype.startswith("text/html"):
        raw = bytes(content_view)
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            html = raw.decode("latin-1", errors="replace")
        rewritten = _rewrite_zim_html(
            html, pack_id, theme=theme,
            family=family, reader_mode=reader_mode,
        )
        return Response(
            content=rewritten,
            media_type="text/html; charset=utf-8",
            headers=safety_headers,
        )

    # CSS branch: same buffer-then-rewrite shape as HTML. ``url(...)``
    # references in CSS resolve against the stylesheet's URL (HTML
    # ``<base>`` doesn't help), so without rewriting, a stylesheet at
    # ``-/style.css`` referencing ``url(I/foo.png)`` 404s in the network
    # panel. Stylesheets are typically <100 KB; buffering is fine.
    if mimetype.startswith("text/css"):
        raw = bytes(content_view)
        try:
            css = raw.decode("utf-8", errors="replace")
        except Exception:
            css = raw.decode("latin-1", errors="replace")
        rewritten_css = _rewrite_zim_css(css, pack_id)
        return Response(
            content=rewritten_css,
            media_type="text/css; charset=utf-8",
            headers=safety_headers,
        )

    # Non-HTML, non-CSS branch: stream from the libzim memoryview, with
    # HTTP Range support. Two scaling concerns motivate this:
    #   1. RAM — a 100 MB video entry buffered as ``bytes()`` adds a
    #      full second copy on top of libzim's own decompressed buffer.
    #      At 100k DAU that's real GC pressure on every concurrent
    #      stream. Slicing the memoryview yields per-chunk copies of
    #      ``_ZIM_STREAM_CHUNK`` bytes only.
    #   2. Range — without ``Accept-Ranges: bytes`` and 206 support,
    #      Safari and iOS refuse to play <audio>/<video> at all and
    #      fall into a retry loop that hammers the server. Range is
    #      load-bearing for media-rich packs (TED, Gutenberg audio,
    #      LibriVox-shaped ZIMs).
    total = len(content_view)
    range_header = request.headers.get("range")
    start, end, status_code = _parse_range_header(range_header, total)

    response_headers = dict(safety_headers)
    response_headers["Accept-Ranges"] = "bytes"

    if status_code == 416:
        # RFC 7233 §4.4: a 416 response SHOULD include ``Content-Range``
        # with the unsatisfied-range form ``*/<total>`` so clients can
        # learn the resource length and re-request.
        response_headers["Content-Range"] = f"bytes */{total}"
        return Response(status_code=416, headers=response_headers)

    response_headers["Content-Length"] = str(end - start)
    if status_code == 206:
        response_headers["Content-Range"] = f"bytes {start}-{end - 1}/{total}"

    async def _stream_zim_chunks():
        # ``_hold_item`` is a closure capture, not dead code: it pins
        # the libzim Item alive for the streaming lifetime so the
        # underlying buffer that ``content_view`` references can't be
        # collected mid-stream. memoryview's buffer protocol normally
        # pins its source, but libzim's Python binding has shifted
        # ownership semantics across versions — explicit pin is safer.
        _hold_item = item  # noqa: F841
        pos = start
        while pos < end:
            next_pos = min(pos + _ZIM_STREAM_CHUNK, end)
            # bytes() copies only the slice — never the whole entry.
            yield bytes(content_view[pos:next_pos])
            pos = next_pos

    return StreamingResponse(
        _stream_zim_chunks(),
        status_code=status_code,
        media_type=mimetype,
        headers=response_headers,
    )


# ------------------------------------------------------------------
# Catalog endpoints
# ------------------------------------------------------------------


@router.get("/catalog")
async def browse_catalog(
    request: Request,
    lang: str = "en",
    category: str | None = None,
    size_max: int | None = None,
    sort: str = "recommended",
    q: str | None = None,
    offset: int = 0,
    limit: int = 50,
):
    """Browse the Kiwix ZIM catalog with filtering and pagination."""
    client = _get_catalog_client(request)
    mgr = getattr(request.app.state, "pack_manager", None)

    entries = await client.browse(
        lang=lang,
        category=category,
        max_size_bytes=size_max,
        sort=sort,
        query=q,
        offset=offset,
        limit=limit,
    )

    # Get total count by fetching all matching (unfiltered by offset/limit)
    all_entries = await client.browse(
        lang=lang,
        category=category,
        max_size_bytes=size_max,
        sort=sort,
        query=q,
        offset=0,
        limit=999999,
    )
    total = len(all_entries)

    # Mark which entries are already installed
    installed_ids = set()
    if mgr:
        for pack in mgr.installed:
            pack_id = pack.get("pack_id", pack.get("id", "")) if isinstance(pack, dict) else getattr(pack, "id", "")
            if pack_id:
                installed_ids.add(pack_id)

    result = []
    for e in entries:
        d = e.to_dict()
        d["installed"] = e.id in installed_ids
        result.append(d)

    return {"entries": result, "total": total, "offset": offset, "limit": limit}


@router.get("/catalog/featured")
async def featured_catalog(request: Request, lang: str = "eng"):
    """Return featured / recommended packs."""
    client = _get_catalog_client(request)
    mgr = getattr(request.app.state, "pack_manager", None)

    entries = await client.featured(
        lang=lang,
        override=settings.knowledge_featured_packs or None,
    )

    installed_ids = set()
    if mgr:
        for pack in mgr.installed:
            pack_id = pack.get("pack_id", pack.get("id", "")) if isinstance(pack, dict) else getattr(pack, "id", "")
            if pack_id:
                installed_ids.add(pack_id)

    result = []
    for e in entries:
        d = e.to_dict()
        d["installed"] = e.id in installed_ids
        result.append(d)

    return {"featured": result}


@router.get("/catalog/categories")
async def catalog_categories(request: Request):
    """Return available catalog categories."""
    client = _get_catalog_client(request)
    return {"categories": client.categories()}


# ------------------------------------------------------------------
# Install endpoints
# ------------------------------------------------------------------


class InstallRequest(BaseModel):
    catalog_id: str
    download_url: str
    custom_dir: str = ""


@router.post("/install")
async def start_install(body: InstallRequest, request: Request):
    """Start a background ZIM install job. Admin only — shared library."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    mgr = _get_pack_manager(request)
    jobs: dict = getattr(request.app.state, "install_jobs", None)

    job_id = uuid.uuid4().hex[:12]
    job = InstallJob(
        job_id=job_id,
        catalog_id=body.catalog_id,
        status="started",
        stage="downloading",
        started_at=time.time(),
    )

    install_dir = Path(body.custom_dir) if body.custom_dir else Path(mgr.pack_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    # Stash a ledger-invalidate closure on the job so the background
    # task (no request handle) can refresh the resource panel on
    # completion. The closure binds to this request's app, which is
    # the long-lived FastAPI app object — safe to retain across the
    # job's lifetime.
    _app = request.app
    def _invalidate_ledger() -> None:
        from augmentum.resource.ledger import invalidate as _invalidate_resource
        _invalidate_resource(_app.state, "knowledge", disk=True)
    job._ledger_invalidate = _invalidate_ledger  # type: ignore[attr-defined]

    task = asyncio.create_task(
        _run_install(job, body.download_url, install_dir, mgr)
    )
    job.task = task
    jobs[job_id] = job

    return {"job_id": job_id, "status": "started"}


@router.get("/install/{job_id}/progress")
async def install_progress(job_id: str, request: Request):
    """SSE stream of install progress."""
    jobs: dict = getattr(request.app.state, "install_jobs", None)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Install job not found: {job_id}")

    async def _stream():
        while True:
            payload = {
                "status": job.status,
                "stage": job.stage,
                "current": job.current,
                "total": job.total,
            }
            if job.error:
                payload["error"] = job.error
            yield f"data: {json.dumps(payload)}\n\n"

            if job.status in ("complete", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/install/{job_id}/cancel")
async def cancel_install(job_id: str, request: Request):
    """Cancel a running install job. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    jobs: dict = getattr(request.app.state, "install_jobs", None)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Install job not found: {job_id}")

    if job.task and not job.task.done():
        job.task.cancel()
        job.status = "cancelled"
        job.stage = "cancelled"
    return {"job_id": job_id, "status": job.status}


# ------------------------------------------------------------------
# Storage location
# ------------------------------------------------------------------


@router.put("/storage-location")
async def update_storage_location(request: Request):
    """Update the knowledge packs storage directory. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    body = await request.json()
    new_path = body.get("path", "")
    if not new_path:
        raise HTTPException(400, "Missing 'path' in request body")

    target = Path(new_path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(400, f"Cannot create directory: {exc}")

    jobs: dict = getattr(request.app.state, "install_jobs", {}) or {}
    active_jobs = [
        job_id for job_id, job in jobs.items()
        if getattr(job, "status", "") not in ("complete", "error", "failed", "cancelled")
    ]
    if active_jobs:
        raise HTTPException(409, "Cannot change storage while installs are running")

    settings.knowledge_packs_dir = str(target)

    # Persist to settings store if available
    store = getattr(request.app.state, "settings_store", None)
    if store:
        await store.set("knowledge_packs_dir", str(target))

    loaded = 0
    old_mgr = getattr(request.app.state, "pack_manager", None)
    if old_mgr:
        await old_mgr.close()

    from augmentum.knowledge.packs import PackManager
    pack_mgr = PackManager(target)
    loaded = await pack_mgr.scan()
    request.app.state.pack_manager = pack_mgr

    from augmentum.knowledge.catalog import CatalogClient
    cache_dir = target / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    request.app.state.catalog_client = CatalogClient(
        cache_dir=cache_dir,
        cache_ttl=settings.knowledge_catalog_cache_ttl,
    )

    return {"ok": True, "path": str(target), "loaded": loaded}


# ------------------------------------------------------------------
# Background install runner
# ------------------------------------------------------------------


async def _run_install(
    job: InstallJob,
    download_url: str,
    install_dir: Path,
    pack_mgr: object,
) -> None:
    """Download a ZIM file and optionally convert to augpack."""
    zim_path: Path | None = None
    # Set True after Stage 2 (ZimReader.open) succeeds — the .zim is then
    # known to be a complete, valid Kiwix archive. Used by the error handler
    # to distinguish a partial-download failure (cleanup the .zim) from any
    # later failure (preserve it). Can't rely on job.stage alone because the
    # convert subprocess overwrites stage to "error" on its own failures.
    download_completed = False
    try:
        # ----------------------------------------------------------
        # Stage 1: Download
        # ----------------------------------------------------------
        job.stage = "downloading"
        job.status = "running"

        # Kiwix OPDS catalog provides .meta4 (Metalink XML) URLs instead
        # of direct .zim links.  Strip the suffix to get the real file URL.
        if download_url.endswith(".meta4"):
            download_url = download_url[: -len(".meta4")]
        elif download_url.endswith(".metalink"):
            download_url = download_url[: -len(".metalink")]

        # SSRF guard: catalog-derived URL must not target internal/private IPs.
        try:
            await check_ssrf(download_url)
        except SafeHttpError as exc:
            job.status = "failed"
            job.error = f"Invalid download URL: {exc}"
            log.warning("knowledge_install_ssrf_blocked", url=download_url, error=str(exc))
            return

        filename = download_url.rstrip("/").split("/")[-1]
        if not filename.endswith(".zim"):
            filename = f"{job.catalog_id}.zim"
        zim_path = install_dir / filename

        # Pre-check disk space — HEAD request for Content-Length
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as head_client:
                head_resp = await head_client.head(download_url)
                expected_size = int(head_resp.headers.get("content-length", 0))
                if expected_size > 0:
                    import shutil
                    free = shutil.disk_usage(str(install_dir)).free
                    # Need ~2.5x the download size (ZIM + augpack + temp)
                    needed = int(expected_size * 2.5)
                    if free < needed:
                        free_gb = free / (1024 ** 3)
                        needed_gb = needed / (1024 ** 3)
                        raise RuntimeError(
                            f"Insufficient disk space: {free_gb:.1f}GB free, "
                            f"~{needed_gb:.1f}GB needed (download + conversion)"
                        )
                    log.info("disk_space_ok", free_gb=f"{free / (1024**3):.1f}",
                             needed_gb=f"{needed / (1024**3):.1f}")
        except httpx.HTTPError:
            pass  # HEAD failed — proceed without check (download will fail later if needed)

        # Resume support: if partial file exists, continue from where we left off
        start_byte = 0
        if zim_path.exists():
            start_byte = zim_path.stat().st_size
            log.info("knowledge_download_resume", path=str(zim_path), from_byte=start_byte)

        dl_headers = {}
        if start_byte > 0:
            dl_headers["Range"] = f"bytes={start_byte}-"

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0),
        ) as client:
            async with client.stream("GET", download_url, headers=dl_headers) as resp:
                # 206 = partial content (resume), 200 = full content (no resume support)
                if resp.status_code == 206:
                    # Server supports resume — content-range tells us the total
                    cr = resp.headers.get("content-range", "")
                    if "/" in cr:
                        job.total = int(cr.split("/")[-1])
                    job.current = start_byte
                elif resp.status_code == 200:
                    # Server doesn't support resume — start over
                    start_byte = 0
                    job.total = int(resp.headers.get("content-length", 0))
                    job.current = 0
                else:
                    resp.raise_for_status()

                mode = "ab" if start_byte > 0 else "wb"
                with open(zim_path, mode) as f:
                    async for chunk in resp.aiter_bytes(64 * 1024):
                        f.write(chunk)
                        job.current += len(chunk)

        log.info("knowledge_install_downloaded", path=str(zim_path), bytes=job.current)

        # ----------------------------------------------------------
        # Stage 2: Inspect (quick — just read entry count)
        # ----------------------------------------------------------
        job.stage = "inspecting"
        from augmentum.knowledge.zim_reader import ZimReader

        reader = ZimReader(zim_path)
        article_count = reader.article_count
        reader.close()  # release before subprocess takes over
        # Successful open == the download produced a valid Kiwix archive.
        download_completed = True

        # ----------------------------------------------------------
        # Stage 3: Keep as ZIM
        # ----------------------------------------------------------
        # Auto-embed-on-install was removed deliberately (2026-05-07).
        # Vector indexing is opt-in only via the per-pack "Embed for
        # vector search" icon on the Browse landing card, which fires
        # ``POST /api/knowledge/packs/{pack_id}/embed``. Reasons:
        #   * Embedding is expensive (minutes-to-days) and starts a
        #     CUDA subprocess; no install path should kick that off
        #     without an explicit click.
        #   * The prior threshold knob (``knowledge_zim_embed_threshold``)
        #     had a settings-API floor of 1000, so any user who ever
        #     touched it ended up auto-embedding sub-1k-article packs
        #     forever. Removing the code path is the only durable fix.
        #   * ZIMs ship with their own Xapian keyword index — search
        #     works fine without a vector sidecar. Vector recall is a
        #     power-user complement, not a default need.
        log.info("knowledge_install_kept_zim", articles=article_count)

        # ----------------------------------------------------------
        # Stage 4: Rescan
        # ----------------------------------------------------------
        job.stage = "scanning"
        await pack_mgr.scan()  # type: ignore[attr-defined]

        job.status = "complete"
        job.stage = "done"
        # New pack landed on disk + in knowledge_packs table. Refresh the
        # resource ledger's cached inventory so /api/resources/status
        # shows it without waiting for the next dir-mtime tick. The
        # ledger reference is stashed on the InstallJob at start_install
        # time — see start_install below.
        try:
            ledger_invalidate = getattr(job, "_ledger_invalidate", None)
            if callable(ledger_invalidate):
                ledger_invalidate()
        except Exception:
            log.warning("knowledge_ledger_invalidate_failed", catalog_id=job.catalog_id, exc_info=True)
        log.info("knowledge_install_complete", catalog_id=job.catalog_id)
        # Pack list membership changed — tell every client to refetch
        # /api/knowledge/packs (server-scoped shared library → broadcast).
        system_events.publish("knowledge.changed", {"pack_id": job.catalog_id, "reason": "install"})

    except asyncio.CancelledError:
        job.status = "cancelled"
        job.stage = "cancelled"
        # Clean up partial download
        if zim_path and zim_path.exists():
            try:
                zim_path.unlink()
            except OSError:
                pass
        log.info("knowledge_install_cancelled", catalog_id=job.catalog_id)

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        # Preserve the .zim whenever the download stage completed (the
        # ZimReader open at Stage 2 succeeded). Any later failure (embedder
        # OOM, sqlite error, conversion abort) is a config/data issue — the
        # file itself is intact. Keeping it lets the user retry conversion
        # without re-paying the download cost AND keeps the catalog item
        # browseable via the .zim sidecar even if no augpack materializes.
        # Only true download failures (partial/corrupt .zim) get cleaned up.
        if zim_path and zim_path.exists():
            if download_completed:
                log.info(
                    "knowledge_install_zim_preserved",
                    path=str(zim_path),
                    stage=job.stage,
                )
            else:
                try:
                    zim_path.unlink()
                    log.info(
                        "knowledge_install_cleanup",
                        path=str(zim_path),
                        stage=job.stage,
                    )
                except OSError:
                    pass
        log.warning(
            "knowledge_install_failed",
            catalog_id=job.catalog_id,
            error=str(exc),
            exc_info=True,
        )


async def _run_resume(
    job: InstallJob,
    zim_path: Path,
    augpack_path: Path,
    pack_mgr: object,
    batch_size: int,
) -> None:
    """Resume a failed conversion from the last committed batch.

    Mirrors the convert subprocess block from ``_run_install`` but skips
    download + inspect (the .zim is already on disk and known-good — it
    survived the original install). Spawns convert_worker with --resume
    so the worker opens the existing .augpack instead of creating fresh.
    """
    progress_path = zim_path.with_suffix(".progress.json")
    try:
        job.stage = "resuming"
        job.status = "running"

        convert_args = [
            sys.executable, "-m", "augmentum.knowledge.convert_worker",
            "--zim", str(zim_path),
            "--output", str(augpack_path),
            "--pack-name", job.catalog_id,
            "--progress", str(progress_path),
            "--batch-size", str(batch_size),
            "--resume",
        ]
        if settings.knowledge_embedding_use_gpu:
            convert_args.append("--use-gpu")

        proc = await asyncio.create_subprocess_exec(
            *convert_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Same poll-progress loop as _run_install. Worker re-runs Stage 1
        # extraction (cheap, deterministic) then jumps straight to embedding
        # at the resume point.
        while proc.returncode is None:
            await asyncio.sleep(1)
            try:
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    job.stage = progress.get("stage", job.stage)
                    job.current = progress.get("current", job.current)
                    job.total = progress.get("total", job.total)
                    if progress.get("error"):
                        break
            except (json.JSONDecodeError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        stdout, stderr = await proc.communicate()
        if stdout:
            log.info("convert_worker_stdout", output=stdout.decode(errors="replace").strip())
        if stderr:
            log.warning("convert_worker_stderr", output=stderr.decode(errors="replace").strip())

        # Read worker error BEFORE unlink (same fix as _run_install).
        error_msg = ""
        if proc.returncode != 0:
            try:
                if progress_path.exists():
                    prog = json.loads(progress_path.read_text())
                    error_msg = prog.get("error", "")
            except (json.JSONDecodeError, OSError):
                pass

        try:
            progress_path.unlink(missing_ok=True)
        except OSError:
            pass

        if proc.returncode != 0:
            if not error_msg:
                rc = proc.returncode
                if rc == -9 or rc == 137:
                    error_msg = "Resume killed — out of memory. Try reducing batch size further."
                elif rc == -6 or rc == 134:
                    error_msg = "Resume aborted — likely out of memory. Try reducing batch size further."
                else:
                    error_msg = f"Resume failed (exit {rc})"
            raise RuntimeError(error_msg)

        # Keep the .zim alongside the now-complete .augpack so the pack
        # stays browseable in the Browse panel (mirrors _run_install's
        # post-success behavior since commit a7ad247). The resume path
        # used to unlink here, which broke browseability and made the
        # converted pack non-re-embeddable if a future model upgrade
        # required a fresh build.
        job.stage = "scanning"
        await pack_mgr.scan()  # type: ignore[attr-defined]

        job.status = "complete"
        job.stage = "done"
        log.info("knowledge_resume_complete", pack_id=job.catalog_id,
                 zim_kept=str(zim_path))
        system_events.publish("knowledge.changed", {"pack_id": job.catalog_id, "reason": "resume"})

    except asyncio.CancelledError:
        job.status = "cancelled"
        job.stage = "cancelled"
        log.info("knowledge_resume_cancelled", pack_id=job.catalog_id)
        raise

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        # NB: do NOT unlink the .zim or .augpack here — both are needed for
        # the user's NEXT resume attempt. The whole point of resume is that
        # partial state is recoverable; if the resume itself fails we want
        # to try again later, not nuke the work-so-far.
        log.warning(
            "knowledge_resume_failed",
            pack_id=job.catalog_id,
            error=str(exc),
            exc_info=True,
        )


async def _run_embed_zim(
    job: InstallJob,
    zim_path: Path,
    augpack_path: Path,
    pack_mgr: object,
    batch_size: int,
) -> None:
    """Embed a previously-installed ZIM pack into a fresh .augpack sidecar.

    Used by the per-pack opt-in "Embed for vector search" flow. Distinct
    from _run_install (which downloads first) and _run_resume (which picks
    up a partial .augpack with --resume): here the .zim is already on disk
    AND no prior .augpack exists, so we run the convert subprocess with no
    --resume flag against a clean target path. On success the .zim is kept
    so the pack remains browseable; on failure the .zim is left untouched.
    """
    progress_path = zim_path.with_suffix(".progress.json")
    try:
        job.stage = "embedding"
        job.status = "running"

        convert_args = [
            sys.executable, "-m", "augmentum.knowledge.convert_worker",
            "--zim", str(zim_path),
            "--output", str(augpack_path),
            "--pack-name", job.catalog_id,
            "--progress", str(progress_path),
            "--batch-size", str(batch_size),
        ]
        if settings.knowledge_embedding_use_gpu:
            convert_args.append("--use-gpu")

        proc = await asyncio.create_subprocess_exec(
            *convert_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Same poll-progress loop as _run_install / _run_resume.
        while proc.returncode is None:
            await asyncio.sleep(1)
            try:
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                    job.stage = progress.get("stage", job.stage)
                    job.current = progress.get("current", job.current)
                    job.total = progress.get("total", job.total)
                    if progress.get("error"):
                        break
            except (json.JSONDecodeError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        stdout, stderr = await proc.communicate()
        if stdout:
            log.info("convert_worker_stdout", output=stdout.decode(errors="replace").strip())
        if stderr:
            log.warning("convert_worker_stderr", output=stderr.decode(errors="replace").strip())

        # Read worker error BEFORE unlink (mirrors _run_install).
        error_msg = ""
        if proc.returncode != 0:
            try:
                if progress_path.exists():
                    prog = json.loads(progress_path.read_text())
                    error_msg = prog.get("error", "")
            except (json.JSONDecodeError, OSError):
                pass

        try:
            progress_path.unlink(missing_ok=True)
        except OSError:
            pass

        if proc.returncode != 0:
            if not error_msg:
                rc = proc.returncode
                if rc == -9 or rc == 137:
                    error_msg = "Embed killed — out of memory. Try reducing batch size."
                elif rc == -6 or rc == 134:
                    error_msg = "Embed aborted — likely out of memory. Try reducing batch size."
                else:
                    error_msg = f"Embed failed (exit {rc})"
            raise RuntimeError(error_msg)

        # Keep the .zim alongside the new .augpack — pack stays browseable.
        job.stage = "scanning"
        await pack_mgr.scan()  # type: ignore[attr-defined]

        job.status = "complete"
        job.stage = "done"
        log.info(
            "knowledge_embed_complete",
            pack_id=job.catalog_id,
            augpack=str(augpack_path),
            zim_kept=str(zim_path),
        )
        system_events.publish("knowledge.changed", {"pack_id": job.catalog_id, "reason": "embed"})

    except asyncio.CancelledError:
        job.status = "cancelled"
        job.stage = "cancelled"
        log.info("knowledge_embed_cancelled", pack_id=job.catalog_id)
        raise

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        # Leave .zim and any partial .augpack on disk. The user can retry
        # via the same button (which would resume via the failed-conversion
        # surface) or discard via the existing discard endpoint.
        log.warning(
            "knowledge_embed_failed",
            pack_id=job.catalog_id,
            error=str(exc),
            exc_info=True,
        )
