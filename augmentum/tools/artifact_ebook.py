"""EPUB storybook renderer — generates illustrated EPUB3 ebooks from structured chapter data."""

from __future__ import annotations

import io
import json
import os
import re
import uuid
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape as xml_escape

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.artifact_storage import ArtifactStore

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chapter label helpers
# ---------------------------------------------------------------------------

_NUMBER_WORDS = [
    "",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
    "Fourteen",
    "Fifteen",
    "Sixteen",
    "Seventeen",
    "Eighteen",
    "Nineteen",
    "Twenty",
]


def _chapter_label(index: int) -> str:
    """Return a chapter label like 'Chapter One' (word form up to 20, numeric after)."""
    n = index + 1
    if 1 <= n <= 20:
        return f"Chapter {_NUMBER_WORDS[n]}"
    return f"Chapter {n}"


# ---------------------------------------------------------------------------
# Markdown to HTML
# ---------------------------------------------------------------------------


def _md_to_html(text: str) -> str:
    """Convert markdown to HTML using markdown-it-py with regex fallback."""
    try:
        from markdown_it import MarkdownIt

        md = MarkdownIt("commonmark", {"typographer": True})
        return md.render(text)
    except ImportError:
        html = text
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", html, flags=re.DOTALL)
        # Wrap remaining text blocks in paragraphs
        parts = html.split("\n\n")
        wrapped = []
        for part in parts:
            part = part.strip()
            if part and not part.startswith("<"):
                wrapped.append(f"<p>{part}</p>")
            elif part:
                wrapped.append(part)
        html = "\n".join(wrapped)
        return html


def _body_to_xhtml_paragraphs(body: str) -> str:
    """Convert body text to XHTML paragraphs with drop cap on first paragraph."""
    html = _md_to_html(body)
    # Mark the first <p> tag with class="first" for drop cap styling
    html = re.sub(r"<p>", '<p class="first">', html, count=1)
    return html


# ---------------------------------------------------------------------------
# EPUB structural templates
# ---------------------------------------------------------------------------

_CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


def _build_opf(
    title: str,
    author: str,
    book_id: str,
    chapters: list[dict],
    has_cover_image: bool,
    image_ids: list[tuple[str, str]],
) -> str:
    """Build the OPF package document (manifest + spine + DC metadata)."""
    manifest_items = [
        '<item id="toc" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="nav" href="toc.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="style" href="style.css" media-type="text/css"/>',
        '<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>',
        '<item id="titlepage" href="title.xhtml" media-type="application/xhtml+xml"/>',
    ]
    spine_refs = [
        '<itemref idref="cover"/>',
        '<itemref idref="titlepage"/>',
    ]

    if has_cover_image:
        manifest_items.append(
            '<item id="cover-image" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>'
        )

    for i in range(len(chapters)):
        ch_id = f"chapter-{i + 1:02d}"
        manifest_items.append(
            f'<item id="{ch_id}" href="{ch_id}.xhtml" media-type="application/xhtml+xml"/>'
        )
        spine_refs.append(f'<itemref idref="{ch_id}"/>')

    for img_id, img_file in image_ids:
        manifest_items.append(
            f'<item id="{img_id}" href="images/{img_file}" media-type="image/jpeg"/>'
        )

    manifest_str = "\n    ".join(manifest_items)
    spine_str = "\n    ".join(spine_refs)

    author_meta = ""
    if author:
        author_meta = f"\n    <dc:creator>{xml_escape(author)}</dc:creator>"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">urn:uuid:{book_id}</dc:identifier>
    <dc:title>{xml_escape(title)}</dc:title>{author_meta}
    <dc:language>en</dc:language>
    <dc:date>{datetime.now().strftime("%Y-%m-%d")}</dc:date>
    <dc:publisher>Augmentum</dc:publisher>
    <meta property="dcterms:modified">{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}</meta>
  </metadata>
  <manifest>
    {manifest_str}
  </manifest>
  <spine toc="toc">
    {spine_str}
  </spine>
</package>"""


def _build_ncx(title: str, book_id: str, chapters: list[dict]) -> str:
    """Build the EPUB2-compatible NCX table of contents."""
    nav_points = []
    # Cover
    nav_points.append(
        '    <navPoint id="navpoint-cover" playOrder="1">\n'
        '      <navLabel><text>Cover</text></navLabel>\n'
        '      <content src="cover.xhtml"/>\n'
        "    </navPoint>"
    )
    # Title page
    nav_points.append(
        '    <navPoint id="navpoint-title" playOrder="2">\n'
        '      <navLabel><text>Title Page</text></navLabel>\n'
        '      <content src="title.xhtml"/>\n'
        "    </navPoint>"
    )
    for i, ch in enumerate(chapters):
        order = i + 3
        heading = xml_escape(ch.get("heading", f"Chapter {i + 1}"))
        ch_file = f"chapter-{i + 1:02d}.xhtml"
        nav_points.append(
            f'    <navPoint id="navpoint-{i + 1}" playOrder="{order}">\n'
            f"      <navLabel><text>{heading}</text></navLabel>\n"
            f'      <content src="{ch_file}"/>\n'
            f"    </navPoint>"
        )

    nav_str = "\n".join(nav_points)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="urn:uuid:{book_id}"/>
  </head>
  <docTitle><text>{xml_escape(title)}</text></docTitle>
  <navMap>
{nav_str}
  </navMap>
</ncx>"""


def _build_nav(title: str, chapters: list[dict]) -> str:
    """Build the EPUB3 nav (toc.xhtml) document."""
    items = ['        <li><a href="cover.xhtml">Cover</a></li>']
    items.append('        <li><a href="title.xhtml">Title Page</a></li>')
    for i, ch in enumerate(chapters):
        heading = xml_escape(ch.get("heading", f"Chapter {i + 1}"))
        ch_file = f"chapter-{i + 1:02d}.xhtml"
        items.append(f'        <li><a href="{ch_file}">{heading}</a></li>')

    items_str = "\n".join(items)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
  <meta charset="UTF-8"/>
  <title>{xml_escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <nav epub:type="toc">
    <h1>Contents</h1>
    <ol>
{items_str}
    </ol>
  </nav>
</body>
</html>"""


def _build_cover_xhtml(title: str, author: str, has_image: bool) -> str:
    """Build the cover page XHTML.

    With image: full-bleed image + gradient overlay + title text.
    Without image: CSS gradient fallback with title/author.
    """
    if has_image:
        body = f"""  <div class="cover-wrap">
    <img src="images/cover.jpg" alt="Cover" class="cover-image"/>
    <div class="cover-overlay">
      <h1 class="cover-title">{xml_escape(title)}</h1>
      {f'<p class="cover-author">{xml_escape(author)}</p>' if author else ''}
    </div>
  </div>"""
    else:
        body = f"""  <div class="cover-fallback">
    <h1 class="cover-title">{xml_escape(title)}</h1>
    {f'<p class="cover-author">{xml_escape(author)}</p>' if author else ''}
  </div>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="UTF-8"/>
  <title>Cover</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
{body}
</body>
</html>"""


def _build_title_xhtml(title: str, author: str) -> str:
    """Build the title page XHTML."""
    date_str = datetime.now().strftime("%B %d, %Y")
    author_line = f'<p class="title-author">{xml_escape(author)}</p>' if author else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="UTF-8"/>
  <title>{xml_escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <div class="title-page">
    <h1 class="title-heading">{xml_escape(title)}</h1>
    {author_line}
    <p class="title-date">{date_str}</p>
    <p class="title-generator">Generated by Augmentum</p>
  </div>
</body>
</html>"""


def _build_chapter_xhtml(
    title: str,
    heading: str,
    body: str,
    chapter_index: int,
    image_filename: str | None = None,
    image_caption: str | None = None,
) -> str:
    """Build a single chapter XHTML document.

    Features: chapter label (word form), heading, optional illustration figure
    with alternating left/right placement, drop cap on first paragraph.
    """
    label = _chapter_label(chapter_index)
    body_html = _body_to_xhtml_paragraphs(body) if body else ""

    figure_html = ""
    if image_filename:
        side = "illustration-right" if chapter_index % 2 == 0 else "illustration-left"
        cap = f"<figcaption>{xml_escape(image_caption)}</figcaption>" if image_caption else ""
        figure_html = f"""
    <figure class="{side}">
      <img src="images/{xml_escape(image_filename)}" alt="{xml_escape(image_caption or heading)}"/>
      {cap}
    </figure>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="UTF-8"/>
  <title>{xml_escape(title)} &mdash; {xml_escape(heading)}</title>
  <link rel="stylesheet" type="text/css" href="style.css"/>
</head>
<body>
  <div class="chapter">
    <p class="chapter-label">{xml_escape(label)}</p>
    <h1 class="chapter-heading">{xml_escape(heading)}</h1>{figure_html}
    {body_html}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Storybook CSS
# ---------------------------------------------------------------------------

_STORYBOOK_CSS = """/* Augmentum Storybook Theme — warm parchment with serif typography */

body {
  background: #faf9f6;
  color: #2c1810;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.1em;
  line-height: 1.7;
  margin: 1.5em;
  padding: 0;
}

/* --- Cover page --- */

.cover-wrap {
  position: relative;
  text-align: center;
  page-break-after: always;
}

.cover-image {
  width: 100%;
  max-height: 100%;
  object-fit: cover;
}

.cover-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 2em 1.5em;
  background: linear-gradient(transparent, rgba(44, 24, 16, 0.85));
  color: #faf9f6;
  text-align: center;
}

.cover-fallback {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 90vh;
  background: linear-gradient(135deg, #8b7355 0%, #5c4a32 50%, #2c1810 100%);
  color: #faf9f6;
  text-align: center;
  padding: 2em;
  page-break-after: always;
}

.cover-title {
  font-size: 2.4em;
  font-weight: normal;
  margin: 0 0 0.3em;
  letter-spacing: 1px;
}

.cover-author {
  font-size: 1.2em;
  font-style: italic;
  margin: 0;
  opacity: 0.85;
}

/* --- Title page --- */

.title-page {
  text-align: center;
  padding-top: 30%;
  page-break-after: always;
}

.title-heading {
  font-size: 2em;
  font-weight: normal;
  color: #2c1810;
  margin-bottom: 0.5em;
}

.title-author {
  font-size: 1.3em;
  color: #8b7355;
  margin-bottom: 1em;
}

.title-date {
  font-size: 0.9em;
  color: #8b7355;
}

.title-generator {
  font-size: 0.8em;
  color: #b0a090;
  font-style: italic;
  margin-top: 3em;
}

/* --- Chapter --- */

.chapter {
  page-break-before: always;
}

.chapter-label {
  font-variant: small-caps;
  letter-spacing: 3px;
  color: #8b7355;
  text-align: center;
  margin-bottom: 0.2em;
  font-size: 0.9em;
}

.chapter-heading {
  font-size: 1.6em;
  font-weight: normal;
  text-align: center;
  color: #2c1810;
  border-bottom: 1px solid #d4c5a9;
  padding-bottom: 0.4em;
  margin-bottom: 1.2em;
}

/* --- Drop cap --- */

p.first::first-letter {
  float: left;
  font-size: 3.2em;
  line-height: 0.8;
  padding-right: 0.08em;
  color: #8b7355;
  font-weight: normal;
}

p {
  text-indent: 1.5em;
  margin: 0.4em 0;
}

p.first {
  text-indent: 0;
}

/* --- Illustrations --- */

figure {
  margin: 0.8em 0;
  padding: 0;
}

.illustration-right {
  float: right;
  width: 45%;
  margin: 0 0 1em 1.2em;
}

.illustration-left {
  float: left;
  width: 45%;
  margin: 0 1.2em 1em 0;
}

figure img {
  width: 100%;
  border-radius: 3px;
  box-shadow: 0 2px 8px rgba(44, 24, 16, 0.15);
}

figcaption {
  font-size: 0.8em;
  color: #8b7355;
  font-style: italic;
  text-align: center;
  margin-top: 0.3em;
}

/* --- Navigation / TOC --- */

nav h1 {
  font-size: 1.4em;
  color: #2c1810;
  border-bottom: 1px solid #d4c5a9;
  padding-bottom: 0.3em;
}

nav ol {
  list-style: none;
  padding-left: 0;
}

nav li {
  margin: 0.4em 0;
}

nav a {
  color: #8b7355;
  text-decoration: none;
}

nav a:hover {
  text-decoration: underline;
}

/* --- Mobile / narrow-screen reflow ---------------------------------------
 * Floated illustrations at 45% width become unreadable on phone-class
 * screens, so stack them full-width under the text when the viewport
 * is narrower than typical tablet portrait. */
@media (max-width: 480px) {
  body { margin: 1em; font-size: 1em; }
  .illustration-left,
  .illustration-right {
    float: none;
    width: 100%;
    margin: 0.8em 0;
  }
  .chapter-heading { font-size: 1.4em; }
  .cover-title    { font-size: 1.9em; }
}

/* --- Dark-mode adaptation -------------------------------------------------
 * Kobo, Kindle, Apple Books and most modern readers respect
 * prefers-color-scheme. Without this block the storybook's hardcoded
 * parchment renders as black text on a near-white background even when
 * the reader is set to dark — unreadable in many cases. We keep the
 * same warm palette and just invert foreground/background pairs. */
@media (prefers-color-scheme: dark) {
  body {
    background: #1a140d;
    color: #e8d9be;
  }
  .chapter-heading {
    color: #e8d9be;
    border-bottom-color: #3a2e20;
  }
  .title-heading {
    color: #e8d9be;
  }
  .chapter-label,
  .title-author,
  .title-date,
  figcaption,
  nav a {
    color: #c0a67a;
  }
  .title-generator {
    color: #8a7452;
  }
  p.first::first-letter {
    color: #c0a67a;
  }
  nav h1 {
    color: #e8d9be;
    border-bottom-color: #3a2e20;
  }
  figure img {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
  }
}
"""


# EPUB reading themes — *book* themes (real page background + typography a
# reader would expect from "sepia" / "night" / etc.), deliberately separate
# from the white-paper business palettes in augmentum.tools.artifact_theme.
# Keep the names in sync with _EPUB_THEME_OPTIONS in ui/scripts/studio.js
# (the front-end only needs the swatch colours; the actual render happens
# here). 'storybook' is the default and is special-cased to the verbatim
# warm-parchment CSS above.
#
# Each entry: (bg, fg, accent, accent_dark, muted, border, font, dark).
# `font` is "serif" (keep the storybook's Georgia) or "sans"; `dark` true
# means the base palette already is dark, so the prefers-color-scheme
# override is dropped to stop it fighting the chosen theme.
_EPUB_THEMES: dict[str, dict[str, object]] = {
    "paper": {
        "bg": "#ffffff", "fg": "#1a1a1a", "accent": "#2563eb",
        "accent_dark": "#1e3a8a", "muted": "#6b7280", "border": "#e5e7eb",
        "font": "serif", "dark": False,
    },
    "sepia": {
        "bg": "#f4ecd8", "fg": "#5b4636", "accent": "#9a6b3f",
        "accent_dark": "#6b4a26", "muted": "#8a7a5c", "border": "#ddccab",
        "font": "serif", "dark": False,
    },
    "slate": {
        "bg": "#f5f6f8", "fg": "#1f2933", "accent": "#3b6ea5",
        "accent_dark": "#234a73", "muted": "#6b7785", "border": "#d7dce2",
        "font": "sans", "dark": False,
    },
    "night": {
        "bg": "#16181c", "fg": "#d4d6da", "accent": "#7ea6d8",
        "accent_dark": "#5e84b5", "muted": "#8a8f97", "border": "#2c3036",
        "font": "sans", "dark": True,
    },
    "midnight": {
        "bg": "#0e1320", "fg": "#c4ccda", "accent": "#8aa6d6",
        "accent_dark": "#6b88bb", "muted": "#7d8597", "border": "#232b3d",
        "font": "serif", "dark": True,
    },
}

# Marks the start of the prefers-color-scheme block at the tail of
# _STORYBOOK_CSS — used to lop it off for themed (esp. dark) variants.
_STORYBOOK_DARK_BLOCK_MARKER = "\n/* --- Dark-mode adaptation"


# Reading-comfort overrides layered on top of the theme CSS. Defaults emit
# nothing, so an untouched book is byte-identical to before. Keep the keys in
# sync with the option lists in ui/scripts/studio.js.
#
# Note on selectors: the in-app preview (artifact_routes._epub_to_html) inlines
# the book's CSS scoped under `.epub-content` AND adds its own
# `.serif p { font-size:16px; line-height:1.75 }` shell rule. A plain
# `body { font-size }` (→ `.epub-content { font-size }`) is inherited by <p>
# but loses to that more-targeted `.serif p`. So size/leading are emitted on
# `p`/`li`/`blockquote` (→ `.epub-content p` etc., equal specificity, later in
# the cascade → wins); font-family stays on `body` since nothing in the shell
# targets `p`'s font-family. In a real e-reader (no shell, body == root) these
# behave the same.
_EPUB_FONT_STACKS = {
    "serif": 'Georgia, "Iowan Old Style", "Palatino Linotype", "Times New Roman", serif',
    "sans": '"Helvetica Neue", "Segoe UI", Helvetica, Arial, sans-serif',
    "dyslexic": '"OpenDyslexic", "Comic Sans MS", "Trebuchet MS", Verdana, sans-serif',
}
# 'md' deliberately absent — it's the default, emits nothing. `rem` (not `em`)
# so the step doesn't compound with the theme's `body { font-size: 1.1em }`.
_EPUB_SIZE_REM = {"xs": "0.8rem", "sm": "0.9rem", "lg": "1.2rem", "xl": "1.45rem"}
# 'normal' deliberately absent — default, emits nothing.
_EPUB_LEADING = {"compact": "1.45", "relaxed": "2.05"}


def _epub_reading_css(reading: dict | None) -> str:
    """Extra rules from a per-artifact ``reading`` block — ``font`` (serif /
    sans / dyslexic), ``size`` (xs|sm|md|lg|xl), ``leading``
    (compact|normal|relaxed). Returns ``""`` when every key is at its default
    so a book nobody has tweaked renders exactly as before."""
    if not isinstance(reading, dict):
        return ""
    out = "\n\n/* reading settings */\n"
    emitted = False

    font_key = str(reading.get("font") or "").strip().lower()
    if font_key in _EPUB_FONT_STACKS:
        out += "body{font-family:" + _EPUB_FONT_STACKS[font_key] + "}\n"
        emitted = True

    para_decls: list[str] = []
    size_key = str(reading.get("size") or "").strip().lower()
    if size_key in _EPUB_SIZE_REM:
        para_decls.append("font-size:" + _EPUB_SIZE_REM[size_key])
    lead_key = str(reading.get("leading") or "").strip().lower()
    if lead_key in _EPUB_LEADING:
        para_decls.append("line-height:" + _EPUB_LEADING[lead_key])
    if para_decls:
        out += "body p,body li,body blockquote{" + ";".join(para_decls) + "}\n"
        emitted = True

    return out if emitted else ""


def _build_epub_css(theme_name: str = "", reading: dict | None = None) -> str:
    """Return the stylesheet embedded in the EPUB.

    An empty/blank name — or the ``"storybook"`` sentinel, or any name not
    in ``_EPUB_THEMES`` — keeps the warm parchment serif default
    (``_STORYBOOK_CSS``). A named reading theme reuses the storybook *layout*
    (cover, drop caps, small-caps chapter labels, TOC) but recolours it from
    that theme's palette, swaps the body face when the theme is sans, and —
    for dark themes — drops the device-dark-mode override so the chosen look
    wins regardless of reader settings. ``reading`` (optional) layers
    font / size / line-height tweaks on top of whichever theme is active.

    Done by substituting the storybook's literal palette values inside the
    existing CSS rather than templating, so the structure can't be broken by
    a bad theme name (unknown names just fall back to the default).
    """
    name = (theme_name or "").strip().lower()
    t = _EPUB_THEMES.get(name)
    if not name or name == "storybook" or t is None:
        return _STORYBOOK_CSS + _epub_reading_css(reading)

    css = _STORYBOOK_CSS
    palette = {
        "#faf9f6": t["bg"],          # page background
        "#2c1810": t["fg"],          # body text / heading colour / gradient end
        "#8b7355": t["accent"],      # drop caps, chapter labels, links
        "#5c4a32": t["accent_dark"], # cover-fallback gradient mid-stop
        "#d4c5a9": t["border"],      # heading rules, TOC divider
        "#b0a090": t["muted"],       # generator credit
    }
    if t["dark"]:
        # Base palette is already dark — strip the prefers-color-scheme block.
        idx = css.find(_STORYBOOK_DARK_BLOCK_MARKER)
        if idx != -1:
            css = css[:idx] + "\n"
    else:
        # Light theme: keep the override but recolour it to neutral darks so
        # the book still reads on a device forced to dark mode.
        palette.update({
            "#1a140d": "#15171a", "#e8d9be": "#e7e8ea", "#3a2e20": "#2a2d33",
            "#c0a67a": t["accent"], "#8a7452": t["muted"],
        })
    for old, new in palette.items():
        css = css.replace(old, str(new))
    if t["font"] == "sans":
        css = css.replace(
            'font-family: Georgia, "Times New Roman", serif;',
            'font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;',
        )
    return f"/* Augmentum EPUB — {name} theme */\n" + css + _epub_reading_css(reading)


# ---------------------------------------------------------------------------
# Image conversion helper
# ---------------------------------------------------------------------------


# Max pixel dimension for embedded chapter illustrations. A 4K cover
# at quality 85 JPEG is ~300-400 KB per image, which bloats EPUBs past
# mobile-reader memory limits when multiplied across chapters. 1600px
# on the long side matches the print-equivalent resolution a typical
# e-ink/tablet reader can display without visible downsampling.
_EPUB_MAX_IMAGE_PX = 1600


def _image_to_jpeg_bytes(path: str, *, max_dim: int = _EPUB_MAX_IMAGE_PX) -> bytes:
    """Load an image from disk, resize it to fit within ``max_dim`` on
    the longest side, and return JPEG bytes.

    Downsampling is skipped when the source already fits, so existing
    small covers/illustrations pass through untouched. Only downsizes —
    never upscales. Addresses the audit finding that EPUBs could balloon
    past reader size limits because large source images were embedded
    at full resolution.
    """
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > max_dim:
            ratio = max_dim / longest
            img = img.resize(
                (max(1, int(w * ratio)), max(1, int(h * ratio))),
                resample=PILImage.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# EPUB renderer (pure function: structured data -> bytes)
# ---------------------------------------------------------------------------


def _render_epub(
    title: str,
    author: str,
    chapters: list[dict],
    *,
    cover_image_path: str | None = None,
    theme_name: str = "",
    reading: dict | None = None,
) -> bytes:
    """Render structured chapter data to an EPUB3 file (returned as bytes).

    This is a pure function — no side effects, no database access.
    The mimetype entry is stored first and uncompressed per EPUB spec.
    """
    book_id = str(uuid.uuid4())
    buf = io.BytesIO()

    has_cover_image = bool(cover_image_path and os.path.exists(cover_image_path))

    # Collect chapter images
    image_ids: list[tuple[str, str]] = []  # (manifest_id, filename)
    chapter_images: dict[int, str] = {}  # chapter_index -> filename

    for i, ch in enumerate(chapters):
        img_path = ch.get("_image_path", "")
        if img_path and os.path.exists(img_path):
            fname = f"ch{i + 1:02d}.jpg"
            mid = f"img-ch{i + 1:02d}"
            image_ids.append((mid, fname))
            chapter_images[i] = fname

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # mimetype MUST be first and uncompressed
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )

        # META-INF
        zf.writestr("META-INF/container.xml", _CONTAINER_XML)

        # OEBPS content
        zf.writestr(
            "OEBPS/content.opf",
            _build_opf(title, author, book_id, chapters, has_cover_image, image_ids),
        )
        zf.writestr("OEBPS/toc.ncx", _build_ncx(title, book_id, chapters))
        zf.writestr("OEBPS/toc.xhtml", _build_nav(title, chapters))
        zf.writestr("OEBPS/style.css", _build_epub_css(theme_name, reading))

        # Cover page
        zf.writestr(
            "OEBPS/cover.xhtml",
            _build_cover_xhtml(title, author, has_cover_image),
        )

        # Cover image
        if has_cover_image:
            try:
                cover_data = _image_to_jpeg_bytes(cover_image_path)
                zf.writestr("OEBPS/images/cover.jpg", cover_data)
            except Exception as e:
                log.warning("cover_image_failed", error=str(e))

        # Title page
        zf.writestr("OEBPS/title.xhtml", _build_title_xhtml(title, author))

        # Chapters
        for i, ch in enumerate(chapters):
            heading = ch.get("heading", f"Chapter {i + 1}")
            body = ch.get("body", "")
            if isinstance(body, list):
                body = "\n\n".join(str(item) for item in body)

            img_fname = chapter_images.get(i)
            img_caption = ch.get("image_caption", "") if img_fname else None

            zf.writestr(
                f"OEBPS/chapter-{i + 1:02d}.xhtml",
                _build_chapter_xhtml(title, heading, body, i, img_fname, img_caption),
            )

            # Embed chapter illustration
            if img_fname and ch.get("_image_path"):
                try:
                    img_data = _image_to_jpeg_bytes(ch["_image_path"])
                    zf.writestr(f"OEBPS/images/{img_fname}", img_data)
                except Exception as e:
                    log.warning(
                        "chapter_image_failed", chapter=i + 1, error=str(e)
                    )

    return buf.getvalue()


# ---------------------------------------------------------------------------
# EbookTool — ArtifactStore integration
# ---------------------------------------------------------------------------


class EbookTool(Tool):
    """Generate illustrated EPUB3 storybooks from structured chapter data."""

    def __init__(self, artifact_store: ArtifactStore, app_state=None) -> None:
        self._store = artifact_store
        self._app_state = app_state
        # Per-call scratch slot for the auto-generated cover URL — reset
        # at the top of every execute() call. Initializing here keeps
        # _auto_illustrate() callable in isolation (e.g. in unit tests).
        self._auto_cover_url: str | None = None

    @property
    def name(self) -> str:
        return "create_ebook"

    @property
    def timeout(self) -> float:
        # Auto-illustration generates cover + per-chapter images sequentially
        return 600.0

    @property
    def description(self) -> str:
        return (
            "Create an illustrated EPUB3 ebook from structured chapters. "
            "Illustrations are generated automatically for each chapter — "
            "you do NOT need to call image_generation first. "
            "Each chapter needs: heading (title) and body (full prose text — write "
            "complete paragraphs, not summaries, at least 3-4 paragraphs each). "
            "Returns a download link for the generated .epub file."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def consumes(self) -> list[str]:
        return ["image_url"]

    @property
    def produces(self) -> list[str]:
        return ["artifact_url"]

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No chapters provided": "The chapters parameter must be a JSON array of objects, each with 'heading' and 'body' fields.",
            "No content after cleanup": "Chapter body text was empty or contained only whitespace. Write substantial prose for each chapter.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Book title",
                },
                "author": {
                    "type": "string",
                    "description": "Author name (optional)",
                    "default": "",
                },
                "cover_image_url": {
                    "type": "string",
                    "description": (
                        "Cover image URL. Leave empty to auto-generate. "
                        "If supplied, MUST be a real URL returned by a "
                        "prior image_generation tool call in this same "
                        "conversation — never invent or example-fill IDs."
                    ),
                    "default": "",
                },
                "chapters": {
                    "type": "array",
                    "description": "Ordered list of book chapters",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {
                                "type": "string",
                                "description": "Chapter heading/title",
                            },
                            "body": {
                                "type": "string",
                                "description": "Chapter body text (markdown supported)",
                            },
                            "image_url": {
                                "type": "string",
                                "description": (
                                    "Per-chapter illustration URL. LEAVE THIS "
                                    "EMPTY in almost every case — illustrations "
                                    "are auto-generated for any chapter without "
                                    "an image_url. Only supply a value here when "
                                    "you have a real URL returned by a prior "
                                    "image_generation tool call. Never invent, "
                                    "example-fill, or copy URLs from this schema."
                                ),
                                "default": "",
                            },
                            "image_caption": {
                                "type": "string",
                                "description": "Caption for the chapter illustration (optional)",
                                "default": "",
                            },
                        },
                        "required": ["heading", "body"],
                    },
                },
            },
            "required": ["title", "chapters"],
        }

    async def execute(
        self,
        *,
        title: str = "Untitled",
        author: str = "",
        cover_image_url: str = "",
        chapters: list | None = None,
        task_id: str = "",
        session_id: str = "",
        _context: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        from augmentum.tools.artifact_normalize import normalize_sections, normalize_str

        _user_id = Tool.extract_user_id({"_context": _context, **kwargs})
        title = normalize_str(title, "Untitled")
        author = normalize_str(author)
        chapters = normalize_sections(chapters)
        if not chapters:
            return ToolResult(success=False, error="No chapters provided")

        from augmentum.tools.artifact_sanitize import sanitize_sections

        chapters = sanitize_sections(chapters)
        if not chapters:
            return ToolResult(success=False, error="No content after cleanup")

        # LLMs frequently echo example IDs from the schema description as
        # if they were real URLs (e.g. ``/api/image/abc123``). Validate any
        # supplied cover URL up front and drop it if it doesn't resolve to
        # a real image — the auto-illustrate path will then generate one.
        if cover_image_url:
            cover_resolved = await self._resolve_image_path(
                cover_image_url,
                user_id=_user_id,
            )
            if not cover_resolved:
                log.info(
                    "ebook_dropping_invalid_cover_url",
                    url=cover_image_url[:80],
                )
                cover_image_url = ""

        # Auto-generate illustrations for chapters missing image_url
        self._auto_cover_url = None
        chapters, illus_report = await self._auto_illustrate(
            chapters, title, cover_image_url,
            author=author,
            _context=_context,
            task_id=task_id,
            session_id=session_id,
            user_id=_user_id,
        )
        if not cover_image_url and self._auto_cover_url:
            cover_image_url = self._auto_cover_url

        # Build human-readable warnings from the illustration report so
        # the agent can mention them in chat and the user can see them
        # on the artifact card. Silent partial failures here are exactly
        # the "5 illustrations promised, 4 delivered" UX bug surfaced
        # by the artifact pipeline audit.
        warnings: list[str] = []
        failed_chapters = illus_report.get("failed_chapter_numbers") or []
        if failed_chapters:
            chs = ", ".join(str(n) for n in failed_chapters)
            warnings.append(
                f"Illustration generation failed for chapter{'s' if len(failed_chapters) != 1 else ''} {chs}; "
                "those chapters ship without an illustration."
            )
        if illus_report.get("cover_failed"):
            warnings.append(
                "Cover-image generation failed; the EPUB ships without a "
                "custom cover (a fallback gradient cover is used instead)."
            )

        # Resolve images
        resolved = await self._resolve_images(chapters, user_id=_user_id)

        # Resolve cover image
        cover_path = None
        if cover_image_url:
            cover_path = await self._resolve_image_path(
                cover_image_url,
                user_id=_user_id,
            )

        try:
            data = _render_epub(title, author, resolved, cover_image_path=cover_path)

            safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
            filename = f"{safe_title}.epub"

            image_count = sum(1 for ch in resolved if ch.get("_image_path"))

            requested = illus_report.get("requested_chapters", 0)
            ebook_metadata = {
                "page_type": "ebook",
                "chapter_count": len(chapters),
                "image_count": image_count,
                "image_count_requested": requested,
                "author": author,
            }
            if warnings:
                ebook_metadata["warnings"] = warnings
            info = await self._store.save(
                data=data,
                filename=filename,
                fmt="epub",
                task_id=task_id,
                session_id=session_id,
                display_name=f"{title}.epub",
                user_id=_user_id,
                metadata=ebook_metadata,
                source_json=json.dumps({
                    "type": "ebook",
                    "title": title,
                    "author": author,
                    "cover_image_url": cover_image_url,
                    "chapters": chapters,
                }),
            )

            from augmentum.tools.base import make_artifact_card

            # Summary string: report N of M when some illustrations failed,
            # otherwise the simple count.
            if requested and image_count < requested:
                illus_str = (
                    f", {image_count} of {requested} illustrations"
                )
            elif image_count:
                illus_str = f", {image_count} illustration{'s' if image_count != 1 else ''}"
            else:
                illus_str = ""
            summary = (
                f"Ebook '{title}' is ready — {len(chapters)} chapter"
                f"{'s' if len(chapters) != 1 else ''}{illus_str}. "
                "Available in the artifact library."
            )
            card = make_artifact_card(
                info,
                kind="artifact",
                title=title,
                subtitle=f"by {author}" if author else f"{len(chapters)} chapters",
                summary=summary,
                preview={
                    "artifact_kind": "ebook",
                    "format": "epub",
                    "size_bytes": info.get("size_bytes", 0),
                    "chapters": [
                        {"heading": ch.get("heading", f"Chapter {i+1}")}
                        for i, ch in enumerate(chapters)
                    ],
                    "image_count": image_count,
                    "image_count_requested": requested,
                    "cover_url": cover_image_url or "",
                    "warnings": warnings,
                },
            )
            from augmentum.tools.base import format_output_with_warnings

            return ToolResult(
                success=True,
                output=format_output_with_warnings(summary, warnings),
                metadata=info,
                warnings=warnings,
                card=card,
            )
        except Exception as e:
            log.error("ebook_creation_failed", error=str(e), exc_info=True)
            return ToolResult(success=False, error=f"Ebook creation failed: {e}")

    async def _plan_illustrations(
        self,
        title: str,
        author: str,
        chapters: list,
    ) -> dict | None:
        """Ask the active LLM to author an illustration plan for the book.

        Produces structured per-character + per-chapter visual prompts so
        every image carries (a) a consistent art style, (b) accurate
        appearance for every named character, and (c) a focused, visual,
        scene-specific composition — instead of the heuristic anchor +
        "first 200 chars of body" prompts which lack visual specifics.

        Returns a dict with keys:
            style:         one-sentence art style + palette + framing language
            characters:    {name: physical description}
            cover_prompt:  full prompt for the cover image
            chapter_prompts: list[str], one per chapter (same order)

        Returns ``None`` on any failure — caller falls back to heuristic.
        """
        if not self._app_state:
            return None
        registry = getattr(self._app_state, "provider_registry", None)
        if not registry:
            return None

        # Route to whichever backend hosts the user's selected chat model —
        # default_backend is the internal llama-server, which raises "No model
        # selected" when primary_chat_model is an LM Studio / cloud name that
        # doesn't map to a local GGUF file.
        try:
            from augmentum.config import settings as _settings
            requested_model = _settings.primary_chat_model or ""
        except Exception:
            requested_model = ""
        try:
            backend, model = await registry.resolve_backend_with_fabric(requested_model)
        except Exception:
            return None
        if not backend:
            return None

        # Lazy imports to keep the tool importable in lightweight contexts.
        from augmentum.models.base import InternalChatRequest, Message

        # Compose context: full chapter bodies (capped) so the planner sees
        # appearance details, settings, and key scenes.
        ch_blocks = []
        per_ch_cap = 1200  # chars per chapter — keep prompt budget bounded
        for i, ch in enumerate(chapters):
            heading = (ch.get("heading") or f"Chapter {i + 1}").strip()
            body = (ch.get("body") or "").strip()
            if len(body) > per_ch_cap:
                body = body[:per_ch_cap].rstrip() + " […]"
            ch_blocks.append(f"## Chapter {i + 1}: {heading}\n{body}")
        chapters_text = "\n\n".join(ch_blocks)

        system = (
            "You are an art director planning illustrations for an "
            "illustrated book. Read the entire manuscript below and "
            "output a single JSON object — no prose, no markdown fences "
            "— that an image generator can use to produce visually "
            "consistent, scene-specific art across every chapter.\n\n"
            "REQUIRED SHAPE:\n"
            "{\n"
            '  "style": "<one sentence: medium + palette + lighting + framing — concrete, not generic>",\n'
            '  "characters": {"<Name>": "<exhaustive physical description: species, build, fur/skin/hair color, eye color, distinctive features, clothing/accessories — every detail an artist needs to redraw them identically>", ...},\n'
            '  "cover_prompt": "<focused image prompt for the front cover: hero composition, primary character(s), mood, central setting>",\n'
            '  "chapter_prompts": ["<prompt for chapter 1>", "<prompt for chapter 2>", ...]\n'
            "}\n\n"
            "RULES:\n"
            "- Length of ``chapter_prompts`` MUST equal the number of "
            "chapters in the manuscript.\n"
            "- Each chapter prompt describes ONLY the characters "
            "physically present in that chapter's scene. Do NOT carry "
            "characters forward from other chapters, and do NOT list "
            "the whole cast — only who is actually in the frame.\n"
            "- Each chapter prompt is concrete and visual — pick the "
            "single most striking moment of the chapter and describe "
            "it as a frame: subject, action, expression, setting, "
            "lighting, secondary props. Avoid plot exposition.\n"
            "- Style description applies to EVERY image: be specific "
            "about medium, palette, lighting, and framing (e.g. 'soft "
            "watercolor with pencil outlines, warm dawn palette, "
            "full-bleed cinematic framing') rather than generic "
            "category labels.\n"
            "- Do NOT name characters in chapter prompts — describe "
            "them by their physical traits so the image generator "
            "(which has no character memory) renders them correctly.\n"
            "- Characters dict: ONE entry per named character that "
            "appears visually. Skip incidental mentions.\n"
            "- Output JSON only. No commentary."
        )
        user = (
            f"Title: {title}\n"
            f"Author: {author or '(none)'}\n"
            f"Chapter count: {len(chapters)}\n\n"
            f"Manuscript:\n\n{chapters_text}"
        )
        try:
            req = InternalChatRequest(
                model=model,
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=user),
                ],
                stream=False,
                think=False,
                format="json",
                max_tokens=2400,
            )
            resp = await backend.chat(req)
        except Exception:
            # Backends like LM Studio reject ``format="json"``; retry without.
            try:
                req = InternalChatRequest(
                    model=model,
                    messages=[
                        Message(role="system", content=system),
                        Message(role="user", content=user),
                    ],
                    stream=False,
                    think=False,
                    max_tokens=2400,
                )
                resp = await backend.chat(req)
            except Exception:
                log.warning("ebook_illustration_plan_request_failed", exc_info=True)
                return None

        text = (resp.message.content or "").strip() if resp.message else ""
        if not text:
            return None
        # Strip optional ```json fences
        if text.startswith("```"):
            nl = text.find("\n")
            if nl >= 0:
                text = text[nl + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        # Pull out the first {...} block — many models prepend prose.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            log.warning("ebook_illustration_plan_no_json")
            return None
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            log.warning("ebook_illustration_plan_parse_failed", snippet=text[:200])
            return None

        # Validate shape
        chapter_prompts = data.get("chapter_prompts") or []
        if not isinstance(chapter_prompts, list) or not chapter_prompts:
            return None
        if len(chapter_prompts) < len(chapters):
            # Pad with empty strings; caller will fall back to heuristic for those.
            chapter_prompts = list(chapter_prompts) + [""] * (
                len(chapters) - len(chapter_prompts)
            )

        return {
            "style": str(data.get("style", "")).strip(),
            "characters": data.get("characters") or {},
            "cover_prompt": str(data.get("cover_prompt", "")).strip(),
            "chapter_prompts": [str(p).strip() for p in chapter_prompts],
        }

    def _build_visual_anchor(self, title: str, chapters: list) -> str:
        """Neutral fallback used only when the LLM planner fails.

        Emits NO art direction — no medium, palette, lighting, or
        "storybook/children's" vocabulary, because image models read
        those as subject hints. All style direction is the planner's
        job; when it is missing we prefer to pass chapter content
        through raw rather than poison it with hardcoded descriptors.
        The only signal we add is a recognized-character list so
        recurring subjects stay consistent across chapters.
        """
        snippets = []
        for ch in chapters:
            body = ch.get("body", "")
            if body:
                snippets.append(body[:300])

        combined = " ".join(snippets).lower()

        characters = []
        import re as _re
        for m in _re.finditer(r"(?:named|called)\s+(\w+)", combined):
            characters.append(m.group(1).title())
        for m in _re.finditer(
            r"((?:orange|golden|silver|white|black|grey|brown|red|blue|green)\s+"
            r"(?:tabby|cat|fox|dog|rabbit|owl|deer|bear|wolf|bird|dragon|unicorn))",
            combined,
        ):
            characters.append(m.group(1))

        seen: set[str] = set()
        unique_chars = []
        for c in characters:
            cl = c.lower()
            if cl not in seen:
                seen.add(cl)
                unique_chars.append(c)

        if unique_chars:
            return f"Recurring subjects (render consistently): {', '.join(unique_chars[:5])}."
        return ""

    async def _auto_illustrate(
        self,
        chapters: list,
        title: str,
        cover_image_url: str,
        *,
        author: str = "",
        _context: dict | None = None,
        task_id: str = "",
        session_id: str = "",
        user_id: str = "",
    ) -> tuple[list, dict]:
        """Generate illustrations for chapters that don't already have image_url.

        Also generates a cover image if none was provided. Uses the
        ImageGenerationTool from the registry so all user settings (model,
        preset, quality) are respected. A visual anchor is prepended to
        every prompt for character/style consistency across images.

        Returns ``(chapters, report)`` where ``report`` tracks the
        outcome of each generation attempt:

            {
              "skipped": bool,            # tool unavailable, no attempts made
              "skip_reason": str,         # populated when skipped
              "requested_chapters": int,  # chapters that needed a new image
              "succeeded_chapters": int,
              "failed_chapter_numbers": list[int],
              "cover_attempted": bool,
              "cover_failed": bool,
              "planner_used": bool,       # LLM-driven plan succeeded
            }
        """
        report: dict = {
            "skipped": False,
            "skip_reason": "",
            "requested_chapters": 0,
            "succeeded_chapters": 0,
            "failed_chapter_numbers": [],
            "cover_attempted": False,
            "cover_failed": False,
            "planner_used": False,
        }

        # Get image generation tool from registry
        if not self._app_state:
            report["skipped"] = True
            report["skip_reason"] = "app_state unavailable"
            return chapters, report
        registry = getattr(self._app_state, "tool_registry", None)
        if not registry:
            report["skipped"] = True
            report["skip_reason"] = "tool registry unavailable"
            return chapters, report
        img_tool = registry.get("image_generation")
        if not img_tool:
            log.info("ebook_auto_illustrate_skipped", reason="no image_generation tool")
            report["skipped"] = True
            report["skip_reason"] = "image_generation tool not registered"
            return chapters, report

        # Forward the full execution context to every image_generation call.
        # ``task_id`` makes ``is_agentic`` True inside image_generation, which
        # activates the ``agentic_image_model`` override. ``session_id`` is
        # required for the generation-persistence FK; without it the image
        # renders but can't be saved. ``_context`` carries user_id for the
        # same persistence row.
        _kw: dict = {}
        if _context:
            _kw["_context"] = _context
        if task_id:
            _kw["task_id"] = task_id
        if session_id:
            _kw["session_id"] = session_id

        # Try the LLM-driven illustration planner first — it produces a
        # specific style sentence, full character bible, and per-chapter
        # visual prompts. Falls back to the heuristic anchor when the
        # planner can't run (no backend / no model / parse failure).
        plan = None
        try:
            plan = await self._plan_illustrations(
                title=title,
                author=author,
                chapters=chapters,
            )
        except Exception:
            log.warning("ebook_illustration_plan_unexpected_error", exc_info=True)
            plan = None

        if plan:
            log.info(
                "ebook_illustration_plan_ready",
                style_chars=len(plan.get("style") or ""),
                characters=len(plan.get("characters") or {}),
                chapter_prompts=len(plan.get("chapter_prompts") or []),
            )
            style_line = plan.get("style") or ""
            char_bible = plan.get("characters") or {}
            cover_prompt_text = plan.get("cover_prompt") or ""
            chapter_prompts = plan.get("chapter_prompts") or []
            report["planner_used"] = True
        else:
            style_line = ""
            char_bible = {}
            cover_prompt_text = ""
            chapter_prompts = []

        # Publish the illustration plan to the inspector so the storybook
        # renderer can show the style sentence + character bible up-front,
        # before any image actually renders. Best-effort — no-op outside
        # an agentic context.
        from augmentum.modes.agentic.progress_bus import emit_progress

        await emit_progress({
            "ebook_plan": {
                "title": title,
                "author": author,
                "style": style_line,
                "characters": char_bible,
                "chapters": [
                    {
                        "index": i,
                        "heading": ch.get("heading", f"Chapter {i + 1}"),
                        "prompt": chapter_prompts[i] if i < len(chapter_prompts) else "",
                    }
                    for i, ch in enumerate(chapters)
                ],
            },
        })

        # Heuristic anchor still used as a fallback when the planner output
        # is missing for a particular chapter.
        anchor = self._build_visual_anchor(title, chapters)
        # Negatives are mechanical junk only. Style/subject negatives
        # ("different art style", "no children", etc.) are unreliable —
        # FLUX ignores negatives entirely and SDXL weakly honors them,
        # so the positive prompt is the real control surface. Accuracy
        # comes from the LLM-distilled positive, not from wish-listing
        # here.
        neg = "text, watermark, signature, deformed, extra limbs, extra fingers"

        def _char_block_for(prompt_text: str) -> str:
            """Return per-image character descriptions for any character
            mentioned (by name) in this prompt, joined into one line."""
            if not char_bible or not prompt_text:
                return ""
            mentioned = []
            lower = prompt_text.lower()
            for name, desc in char_bible.items():
                if name and name.lower() in lower and desc:
                    mentioned.append(f"{name} ({desc})")
            return " Characters: " + "; ".join(mentioned) + "." if mentioned else ""

        def _compose(scene_prompt: str, *, is_cover: bool = False) -> str:
            # Style direction is ONLY whatever the planner produced. The
            # heuristic anchor is a neutral fallback (recurring-subject
            # list or empty) and never injects art direction — we do not
            # want to poison a chapter-specific, LLM-distilled prompt
            # with hardcoded medium/palette words.
            parts = []
            if style_line:
                parts.append(style_line)
            elif anchor:
                parts.append(anchor)
            # Only include characters the scene prompt actually names.
            # Dumping the full cast into every image (especially the
            # cover) crowds compositions and produces the disfigured
            # "everyone crammed into one frame" output the user flagged.
            cb = _char_block_for(scene_prompt)
            if cb:
                parts.append(cb.strip())
            if scene_prompt:
                parts.append(scene_prompt if is_cover else f"Scene: {scene_prompt}")
            return " ".join(p for p in parts if p)

        # Generate cover if missing
        if not cover_image_url:
            report["cover_attempted"] = True
            log.info("ebook_generating_cover", title=title)
            cover_scene = cover_prompt_text or (
                f"Book cover: dramatic hero shot of the main character of "
                f"'{title}', vibrant colors"
            )
            cover_result = await img_tool.execute(
                prompt=_compose(cover_scene, is_cover=True),
                negative_prompt=neg,
                style="fantasy_rpg",
                aspect="portrait",
                **_kw,
            )
            if cover_result.success and cover_result.metadata.get("url"):
                self._auto_cover_url = cover_result.metadata["url"]
            else:
                report["cover_failed"] = True
                log.warning(
                    "ebook_cover_generation_failed",
                    error=getattr(cover_result, "error", "")[:120],
                )

        # Use cover image as IP-Adapter reference for chapter consistency
        cover_ref = ""
        if self._auto_cover_url:
            cover_ref = self._auto_cover_url
        elif cover_image_url:
            cover_ref = cover_image_url

        # Generate chapter illustrations
        updated = []
        for i, ch in enumerate(chapters):
            ch = dict(ch)
            existing_url = ch.get("image_url", "")
            # Validate any pre-supplied URL by trying to resolve it. LLMs
            # frequently hallucinate placeholder IDs (abc123, def456) by
            # echoing the schema's example text; if the URL doesn't point
            # at a real image we treat it as missing and regenerate.
            if existing_url:
                if user_id:
                    resolved = await self._resolve_image_path(
                        existing_url,
                        user_id=user_id,
                    )
                else:
                    resolved = await self._resolve_image_path(existing_url)
                if not resolved:
                    log.info(
                        "ebook_dropping_invalid_image_url",
                        chapter=i + 1,
                        url=existing_url[:80],
                    )
                    existing_url = ""
                    ch.pop("image_url", None)
            if not existing_url:
                report["requested_chapters"] += 1
                heading = ch.get("heading", f"Chapter {i + 1}")
                body = ch.get("body", "")
                planned = chapter_prompts[i] if i < len(chapter_prompts) else ""
                if planned:
                    scene_prompt = planned
                else:
                    # Fallback: first 200 chars of body (legacy behaviour).
                    scene_prompt = (body[:200].replace("\n", " ") if body else heading)
                prompt = _compose(scene_prompt)
                log.info(
                    "ebook_generating_illustration",
                    chapter=i + 1,
                    heading=heading,
                    used_planner=bool(planned),
                )

                # Tell the inspector we're about to render this chapter so
                # the storybook panel can flip the chapter card to a
                # "rendering…" state immediately, before the image queue
                # actually starts (pipeline warm-up can take seconds).
                await emit_progress({
                    "chapter_illustration": {
                        "index": i,
                        "heading": heading,
                        "status": "rendering",
                        "prompt": prompt,
                    },
                })

                gen_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": neg,
                    "style": "watercolor",
                    "aspect": "landscape",
                    **_kw,
                }
                # IP-Adapter: use cover as reference for consistent characters/style
                if cover_ref:
                    gen_kwargs["ip_adapter_image"] = cover_ref
                    gen_kwargs["ip_adapter_scale"] = 0.55

                result = await img_tool.execute(**gen_kwargs)

                # If IP-Adapter caused the failure, retry without it
                if not result.success and cover_ref:
                    log.info("ebook_illustration_retry_no_ip", chapter=i + 1)
                    gen_kwargs.pop("ip_adapter_image", None)
                    gen_kwargs.pop("ip_adapter_scale", None)
                    result = await img_tool.execute(**gen_kwargs)
                    if result.success:
                        # Don't try IP-Adapter on remaining chapters
                        cover_ref = ""

                if result.success and result.metadata.get("url"):
                    ch["image_url"] = result.metadata["url"]
                    ch["image_caption"] = heading
                    report["succeeded_chapters"] += 1
                    await emit_progress({
                        "chapter_illustration": {
                            "index": i,
                            "heading": heading,
                            "status": "complete",
                            "url": result.metadata["url"],
                        },
                    })
                else:
                    report["failed_chapter_numbers"].append(i + 1)
                    log.warning(
                        "ebook_chapter_illustration_failed",
                        chapter=i + 1,
                        heading=heading,
                        error=getattr(result, "error", "")[:120],
                    )
                    await emit_progress({
                        "chapter_illustration": {
                            "index": i,
                            "heading": heading,
                            "status": "failed",
                            "error": getattr(result, "error", "")[:200],
                        },
                    })
            updated.append(ch)
        return updated, report

    async def _resolve_images(self, chapters: list, *, user_id: str = "") -> list:
        """Resolve image_url references to filesystem paths."""
        resolved = []
        for ch in chapters:
            c = dict(ch)
            image_url = c.get("image_url", "")
            if image_url:
                path = await self._resolve_image_path(image_url, user_id=user_id)
                if path:
                    c["_image_path"] = str(path)
                else:
                    log.warning("image_resolve_failed", url=image_url)
            resolved.append(c)
        return resolved

    async def _resolve_image_path(self, url: str, *, user_id: str = "") -> str | None:
        """Resolve an image URL to a local filesystem path."""
        m = re.match(r"/api/image/([a-zA-Z0-9_-]+)", url)
        if m:
            image_id = m.group(1)
            return await self._resolve_image_id(image_id, user_id=user_id)

        m = re.match(r"/api/artifacts/([a-zA-Z0-9_-]+)/download", url)
        if m:
            artifact_id = m.group(1)
            info = await self._store.get(artifact_id, user_id=user_id)
            if info:
                file_path = self._store.get_file_path(info["path"])
                if file_path:
                    return str(file_path)

        return None

    async def _resolve_image_id(self, image_id: str, *, user_id: str = "") -> str | None:
        """Look up an image_id via the ImagePersistence store."""
        try:
            db = self._store._db
            try:
                from unittest.mock import Mock
                if isinstance(db, Mock):
                    return None
            except ImportError:
                # unittest.mock unavailable on slimmed builds — fall
                # through to the real DB path.
                pass
            if user_id:
                query = (
                    "SELECT file_path FROM image_generations "
                    "WHERE image_id = ? AND user_id = ?"
                )
                params = (image_id, user_id)
            else:
                log.warning("ebook_image_id_lookup.empty_user_id", image_id=image_id)
                query = "SELECT file_path FROM image_generations WHERE image_id = ?"
                params = (image_id,)
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if row and row[0] and os.path.exists(row[0]):
                    return row[0]
        except Exception as e:
            log.debug("image_id_lookup_failed", image_id=image_id, error=str(e))
        return None
