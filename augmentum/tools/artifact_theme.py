"""Artifact theme system — consistent visual identity across PDF, PPTX, XLSX.

Provides a single source of truth for colors, typography, and spacing
that all artifact renderers draw from.  Users select a theme in settings;
all generated documents, presentations, and spreadsheets match.

Color system based on Tailwind CSS v3 Slate neutrals with accent variants.
Typography follows the Major Third (1.25) scale from a 10pt body base.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ArtifactTheme:
    """Visual theme applied to all generated artifacts."""

    name: str

    # --- Accent colors (the ONE brand color + variants) ---
    accent: str           # Primary accent — headings, links, chart primary
    accent_light: str     # Light variant — callout backgrounds, highlights
    accent_dark: str      # Dark variant — cover page bar, PPTX title slide

    # --- Neutral palette (Slate-based blue-grey, never yellow-grey) ---
    text: str             # Body text — NOT pure black (#0F172A not #000000)
    text_secondary: str   # Subheadings, metadata — slightly lighter
    text_muted: str       # Captions, footers, page numbers
    background: str       # Page/slide background
    surface: str          # Callout/card backgrounds, zebra stripe rows
    border: str           # Rules, table borders, dividers

    # --- Typography ---
    title_size: float = 28.0    # Cover page title (pt)
    h1_size: float = 20.0       # Section heading
    h2_size: float = 16.0       # Subsection
    h3_size: float = 13.0       # Sub-subsection
    h4_size: float = 11.0       # Minor heading
    body_size: float = 10.0     # Body text
    caption_size: float = 8.0   # Captions, footnotes
    line_height: float = 6.0    # Body line height (mm in fpdf2)

    # --- PDF layout (mm) ---
    margin_top: float = 20.0
    margin_bottom: float = 20.0
    margin_left: float = 25.0
    margin_right: float = 25.0
    cover_bar_height: float = 18.0   # Accent bar on cover page (mm)

    # --- PPTX ---
    slide_bar_height: float = 0.15   # Accent bar at bottom (inches)
    slide_title_size: float = 28.0   # Slide heading (pt)
    slide_body_size: float = 16.0    # Slide body (pt)

    # --- XLSX ---
    header_fill: str = ""       # Falls back to accent if empty
    zebra_fill: str = ""        # Falls back to surface if empty

    def rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to (R, G, B) tuple for fpdf2/openpyxl."""
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def hex_no_hash(self, hex_color: str) -> str:
        """Strip # for openpyxl PatternFill which expects bare hex."""
        return hex_color.lstrip("#").upper()

    @property
    def xlsx_header_fill(self) -> str:
        return self.header_fill or self.accent_dark

    @property
    def xlsx_zebra_fill(self) -> str:
        return self.zebra_fill or self.surface


# ---------------------------------------------------------------------------
# Preset themes
# ---------------------------------------------------------------------------


THEME_SLATE = ArtifactTheme(
    name="slate",
    # Blue-600 accent on Slate neutrals — clean, modern, universally professional
    accent="#2563EB",
    accent_light="#EFF6FF",
    accent_dark="#1E3A8A",
    text="#0F172A",           # Slate-900
    text_secondary="#475569",  # Slate-600
    text_muted="#64748B",      # Slate-500
    background="#FFFFFF",
    surface="#F8FAFC",         # Slate-50
    border="#E2E8F0",          # Slate-200
)

THEME_CORPORATE = ArtifactTheme(
    name="corporate",
    # Navy + Amber accent — traditional consulting/business
    accent="#1E3A8A",          # Blue-900
    accent_light="#DBEAFE",    # Blue-100
    accent_dark="#172554",     # Blue-950
    text="#1E293B",            # Slate-800
    text_secondary="#475569",
    text_muted="#64748B",
    background="#FFFFFF",
    surface="#F8FAFC",
    border="#E2E8F0",
)

THEME_MODERN = ArtifactTheme(
    name="modern",
    # Indigo accent — Stripe-inspired tech look
    accent="#4F46E5",          # Indigo-600
    accent_light="#EEF2FF",    # Indigo-50
    accent_dark="#312E81",     # Indigo-900
    text="#0F172A",
    text_secondary="#475569",
    text_muted="#64748B",
    background="#FFFFFF",
    surface="#F8FAFC",
    border="#E2E8F0",
)

THEME_EMERALD = ArtifactTheme(
    name="emerald",
    # Emerald accent — finance, sustainability, positive
    accent="#059669",          # Emerald-600
    accent_light="#ECFDF5",   # Emerald-50
    accent_dark="#064E3B",    # Emerald-900
    text="#0F172A",
    text_secondary="#475569",
    text_muted="#64748B",
    background="#FFFFFF",
    surface="#F8FAFC",
    border="#E2E8F0",
)

THEME_ROSE = ArtifactTheme(
    name="rose",
    # Rose accent — warm, creative, editorial
    accent="#E11D48",          # Rose-600
    accent_light="#FFF1F2",    # Rose-50
    accent_dark="#881337",     # Rose-900
    text="#0F172A",
    text_secondary="#475569",
    text_muted="#64748B",
    background="#FFFFFF",
    surface="#F8FAFC",
    border="#E2E8F0",
)

# Registry
THEMES: dict[str, ArtifactTheme] = {
    "slate": THEME_SLATE,
    "corporate": THEME_CORPORATE,
    "modern": THEME_MODERN,
    "emerald": THEME_EMERALD,
    "rose": THEME_ROSE,
}

DEFAULT_THEME = "slate"


def get_theme(name: str = "") -> ArtifactTheme:
    """Get a theme by name, falling back to the configured default."""
    if not name:
        try:
            from augmentum.config import settings
            name = getattr(settings, "agentic_artifact_theme", DEFAULT_THEME)
        except ImportError:
            name = DEFAULT_THEME

    theme = THEMES.get(name)
    if not theme:
        log.warning("unknown_theme", name=name, fallback=DEFAULT_THEME)
        theme = THEMES[DEFAULT_THEME]
    return theme


# ---------------------------------------------------------------------------
# Design block — Studio palette canonical surface
# ---------------------------------------------------------------------------
# A `source.design` block layers typography + density + accent on top of the
# theme palette. Studio writes it; renderers consume it via `apply_design()`
# which returns a `dataclasses.replace`-style theme copy with scaled fields.
# All five renderers go through this single helper so font-size/spacing
# behavior is identical across PDF / DOCX / PPTX / XLSX / EPUB.

VALID_FONT_FAMILIES = ("system", "sans", "serif", "mono", "dyslexic")
VALID_LINE_HEIGHTS = ("tight", "comfortable", "airy")
VALID_DENSITIES = ("compact", "default", "spacious")
# Discrete UI presets — anything inside the [0.6, 2.0] guardrail validates
# but the picker only exposes these four for muscle-memory consistency.
FONT_SIZE_PRESETS = (0.85, 1.0, 1.15, 1.3)

# Multipliers — line_height + density scale theme baseline values.
_LINE_HEIGHT_SCALE = {"tight": 0.85, "comfortable": 1.0, "airy": 1.2}
_DENSITY_SCALE = {"compact": 0.8, "default": 1.0, "spacious": 1.25}

DEFAULT_DESIGN: dict = {
    "theme": "",
    "font_family": "system",
    "font_size_scale": 1.0,
    "line_height": "comfortable",
    "density": "default",
    "accent_override": None,
}

# Built-in font families per renderer. fpdf2 ships Helvetica / Times /
# Courier built-in; openpyxl + python-pptx + EPUB stacks pick equivalents
# from native OS fonts. Mapping is per-renderer because each library has its
# own font-resolution path.
FONT_FAMILY_PDF = {
    "system": "Helvetica",
    "sans": "Helvetica",
    "serif": "Times",
    "mono": "Courier",
    "dyslexic": "Helvetica",   # no bundled OpenDyslexic; fall back silently
}
FONT_FAMILY_DOCX = {
    "system": "Calibri",
    "sans": "Arial",
    "serif": "Georgia",
    "mono": "Consolas",
    "dyslexic": "Comic Sans MS",   # widely available proxy
}
FONT_FAMILY_PPTX = {
    "system": "Calibri",
    "sans": "Arial",
    "serif": "Georgia",
    "mono": "Consolas",
    "dyslexic": "Comic Sans MS",
}
FONT_FAMILY_XLSX = {
    "system": "Calibri",
    "sans": "Arial",
    "serif": "Times New Roman",
    "mono": "Consolas",
    "dyslexic": "Comic Sans MS",
}


def _is_hex_color(s: object) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) not in (4, 7) or not s.startswith("#"):
        return False
    return all(c in "0123456789abcdefABCDEF" for c in s[1:])


def _expand_hex(hex_color: str) -> str:
    """#abc → #aabbcc; full-length passes through."""
    s = hex_color.strip()
    if len(s) == 4:
        return "#" + "".join(c * 2 for c in s[1:])
    return s


def _darken_hex(hex_color: str, factor: float) -> str:
    """Multiply each channel by `factor` (0..1 darkens)."""
    h = _expand_hex(hex_color).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"


def _lighten_hex(hex_color: str, factor: float) -> str:
    """Move each channel `factor` of the way toward 255 (0..1)."""
    h = _expand_hex(hex_color).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def normalize_design(design: dict | None, fallback_theme: str = "") -> dict:
    """Coerce a partial design dict into a fully-populated normalized one.

    - Unknown keys are silently dropped (forward-compat for v2 fields).
    - Out-of-range values fall back to defaults (never raises).
    - `fallback_theme` seeds the theme slot when the input lacks one.

    Always returns a dict containing every key in `DEFAULT_DESIGN`, so
    renderers can index without `.get()` defensiveness.
    """
    base = dict(DEFAULT_DESIGN)
    if fallback_theme:
        base["theme"] = fallback_theme
    if not isinstance(design, dict):
        return base

    theme = design.get("theme")
    if isinstance(theme, str) and theme:
        base["theme"] = theme

    ff = design.get("font_family")
    if isinstance(ff, str) and ff.lower() in VALID_FONT_FAMILIES:
        base["font_family"] = ff.lower()

    scale = design.get("font_size_scale")
    if isinstance(scale, (int, float)) and 0.6 <= float(scale) <= 2.0:
        base["font_size_scale"] = float(scale)

    lh = design.get("line_height")
    if isinstance(lh, str) and lh.lower() in VALID_LINE_HEIGHTS:
        base["line_height"] = lh.lower()

    dens = design.get("density")
    if isinstance(dens, str) and dens.lower() in VALID_DENSITIES:
        base["density"] = dens.lower()

    accent = design.get("accent_override")
    if _is_hex_color(accent):
        base["accent_override"] = _expand_hex(accent).upper()

    return base


def apply_design(theme: ArtifactTheme, design: dict | None) -> ArtifactTheme:
    """Return a new theme with font sizes / spacing / accent overridden by `design`.

    Scales theme sizes by font_size_scale, expands/contracts line_height by
    the line_height enum, applies density to page margins, and (when set)
    replaces accent + recomputes accent_dark/light. Idempotent — passing
    a normalized default design returns the input theme unchanged.
    """
    from dataclasses import replace

    if not isinstance(design, dict):
        return theme

    d = normalize_design(design, fallback_theme=theme.name)
    overrides: dict = {}

    scale = d["font_size_scale"]
    if scale != 1.0:
        overrides["title_size"] = round(theme.title_size * scale, 2)
        overrides["h1_size"] = round(theme.h1_size * scale, 2)
        overrides["h2_size"] = round(theme.h2_size * scale, 2)
        overrides["h3_size"] = round(theme.h3_size * scale, 2)
        overrides["h4_size"] = round(theme.h4_size * scale, 2)
        overrides["body_size"] = round(theme.body_size * scale, 2)
        overrides["caption_size"] = round(theme.caption_size * scale, 2)
        overrides["slide_title_size"] = round(theme.slide_title_size * scale, 2)
        overrides["slide_body_size"] = round(theme.slide_body_size * scale, 2)

    lh_mult = _LINE_HEIGHT_SCALE[d["line_height"]]
    if lh_mult != 1.0:
        overrides["line_height"] = round(theme.line_height * lh_mult, 3)

    dens_mult = _DENSITY_SCALE[d["density"]]
    if dens_mult != 1.0:
        overrides["margin_top"] = round(theme.margin_top * dens_mult, 2)
        overrides["margin_bottom"] = round(theme.margin_bottom * dens_mult, 2)
        overrides["margin_left"] = round(theme.margin_left * dens_mult, 2)
        overrides["margin_right"] = round(theme.margin_right * dens_mult, 2)

    accent = d["accent_override"]
    if accent:
        overrides["accent"] = accent
        overrides["accent_dark"] = _darken_hex(accent, 0.55)
        overrides["accent_light"] = _lighten_hex(accent, 0.92)

    if not overrides:
        return theme
    return replace(theme, **overrides)
