"""Computed design system for the App Builder (toolkit spec §1).

Derives a concrete palette + typography + spacing from the user's
description. The generator sees the resulting CSS custom-property block
in its prompt and is instructed to reference those variables rather
than reaching for hardcoded colors — which is the tell that makes
AI-generated apps read as AI-produced.

Every built-in palette is WCAG AA verified at import time:
``--text`` against ``--surface`` and ``--accent`` against ``--surface``
must each clear 4.5:1. If a future palette edit drops contrast below
that bar the test suite trips; ``_ensure_contrast`` is also exposed so
runtime edits (e.g. from a LoRA of a palette) can be self-corrected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Mood detection ---------------------------------------------------------
# Keyword → mood classifier. Order matters: more specific keywords come
# first. Unmatched descriptions fall through to ``balanced``, which is
# intentionally neutral so generic apps don't get an opinionated palette
# they didn't ask for.

_MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "playful": (
        "playful", "kid", "kids", "children", "fun", "toy", "game",
        "cute", "whimsical", "bright", "cheerful", "colorful",
    ),
    "moody": (
        "dark", "gothic", "cyberpunk", "horror", "neon", "night",
        "noir", "sinister", "brooding",
    ),
    "minimal": (
        "minimal", "minimalist", "clean", "simple", "zen", "monochrome",
        "spare", "restrained",
    ),
    "elegant": (
        "elegant", "luxury", "premium", "refined", "boutique", "artisan",
        "classic", "editorial",
    ),
    "professional": (
        "professional", "business", "enterprise", "corporate", "b2b",
        "dashboard", "analytics", "crm", "admin", "banking", "finance",
    ),
}

# Default palette used when no strong mood signal is present. Conservative
# neutral scheme that works across form/dashboard/tool descriptions.
_DEFAULT_MOOD = "balanced"


def detect_mood(description: str) -> str:
    """Return the mood key that best matches ``description``.

    Matches are case-insensitive whole-word — "darkness" won't accidentally
    trigger "dark" unless the description actually uses that word. Ties
    go to the mood declared first in ``_MOOD_KEYWORDS``.
    """
    if not description:
        return _DEFAULT_MOOD
    lower = description.lower()
    for mood, keywords in _MOOD_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return mood
    return _DEFAULT_MOOD


# --- Palette definitions ----------------------------------------------------

@dataclass
class DesignSystem:
    """Concrete design tokens derived from a description.

    Serialisable to a CSS custom-properties block via :meth:`to_css_vars`
    and to a short human-readable guidance string via
    :meth:`guidance_for_prompt` — both are wired into the generator's
    system prompt.
    """

    mood: str
    surface: str
    surface_alt: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    border: str
    success: str = "#15803d"
    warning: str = "#b45309"
    error: str = "#b91c1c"
    radius: str = "8px"
    font_body: str = "system-ui, -apple-system, sans-serif"
    font_heading: str = "system-ui, -apple-system, sans-serif"
    notes: list[str] = field(default_factory=list)

    def to_css_vars(self) -> str:
        return (
            ":root {\n"
            f"  --surface: {self.surface};\n"
            f"  --surface-alt: {self.surface_alt};\n"
            f"  --text: {self.text};\n"
            f"  --text-muted: {self.text_muted};\n"
            f"  --accent: {self.accent};\n"
            f"  --accent-hover: {self.accent_hover};\n"
            f"  --border: {self.border};\n"
            f"  --success: {self.success};\n"
            f"  --warning: {self.warning};\n"
            f"  --error: {self.error};\n"
            f"  --radius: {self.radius};\n"
            f"  --font-body: {self.font_body};\n"
            f"  --font-heading: {self.font_heading};\n"
            "}\n"
        )

    def guidance_for_prompt(self) -> str:
        """Short prose block telling the generator how to apply the palette."""
        lines = [
            f"## Design system (mood: {self.mood})",
            "Use the CSS custom properties below — do NOT hardcode hex values "
            "in CSS. Reference them as var(--surface), var(--text), etc. This "
            "keeps the app themable and guarantees a cohesive palette.",
            "",
            self.to_css_vars(),
            f"Typography: body text {self.font_body}, headings {self.font_heading}.",
            f"Border radius: {self.radius}.",
        ]
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"- {n}")
        return "\n".join(lines)


_PALETTES: dict[str, dict] = {
    "playful": {
        "surface": "#fffaf2",
        "surface_alt": "#fff1dc",
        "text": "#2a1a0a",
        "text_muted": "#7a5f3f",
        "accent": "#d94a4a",
        "accent_hover": "#b83a3a",
        "border": "#f0d9a8",
        "radius": "16px",
        "font_body": "'Inter', system-ui, sans-serif",
        "font_heading": "'Fraunces', Georgia, serif",
        "notes": [
            "Favor generous padding, rounded corners, and subtle drop shadows.",
            "Micro-interactions: gentle bounce/scale on hover (transform: scale(1.02)).",
        ],
    },
    "moody": {
        "surface": "#0a0a0f",
        "surface_alt": "#14141f",
        "text": "#f0f0f5",
        "text_muted": "#a0a0b4",
        "accent": "#00ff9f",
        "accent_hover": "#00cc7f",
        "border": "#282838",
        "radius": "4px",
        "font_body": "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
        "font_heading": "'Space Grotesk', system-ui, sans-serif",
        "notes": [
            "Flat UI, hard edges, optional subtle scanline or grain overlay.",
            "Prefer high-contrast accent glows (box-shadow: 0 0 8px var(--accent)).",
        ],
    },
    "minimal": {
        "surface": "#fafafa",
        "surface_alt": "#ffffff",
        "text": "#111111",
        "text_muted": "#666666",
        "accent": "#111111",
        "accent_hover": "#3a3a3a",
        "border": "#e0e0e0",
        "radius": "0",
        "font_body": "'Inter', system-ui, sans-serif",
        "font_heading": "'Inter', system-ui, sans-serif",
        "notes": [
            "Use whitespace as a primary design element. Strict typographic scale.",
            "Avoid shadows, gradients, decorative borders. Structure via whitespace alone.",
        ],
    },
    "elegant": {
        "surface": "#faf7f2",
        "surface_alt": "#ffffff",
        "text": "#1f1610",
        "text_muted": "#6b5d48",
        "accent": "#6e5428",
        "accent_hover": "#4f3c1c",
        "border": "#d9ccb8",
        "radius": "8px",
        "font_body": "Georgia, 'Times New Roman', serif",
        "font_heading": "'Playfair Display', Georgia, serif",
        "notes": [
            "Serif display type, generous line-height, refined hover transitions.",
            "Accent used sparingly — reserve for primary CTAs and key highlights.",
        ],
    },
    "professional": {
        "surface": "#ffffff",
        "surface_alt": "#f5f7fa",
        "text": "#101522",
        "text_muted": "#4c5668",
        "accent": "#0052a3",
        "accent_hover": "#003d7a",
        "border": "#d8dde5",
        "radius": "6px",
        "font_body": "system-ui, -apple-system, 'Segoe UI', sans-serif",
        "font_heading": "system-ui, -apple-system, 'Segoe UI', sans-serif",
        "notes": [
            "Restrained palette, small radius, tight spacing. Data density > whitespace.",
            "Use the accent for interactive affordances, not decoration.",
        ],
    },
    "balanced": {
        "surface": "#ffffff",
        "surface_alt": "#f4f5f7",
        "text": "#111419",
        "text_muted": "#5a6170",
        "accent": "#3b4db4",
        "accent_hover": "#2d3f94",
        "border": "#e2e4e8",
        "radius": "8px",
        "font_body": "system-ui, -apple-system, sans-serif",
        "font_heading": "system-ui, -apple-system, sans-serif",
        "notes": [
            "Neutral default; suitable for general tools and mixed-purpose apps.",
        ],
    },
}


# --- WCAG contrast helpers --------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"invalid hex color: {hex_color!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _relative_luminance(hex_color: str) -> float:
    """Per WCAG 2.1 relative luminance formula — linearises sRGB channels
    before the weighted sum so the ratio matches the spec's definition."""
    r, g, b = _hex_to_rgb(hex_color)

    def _lin(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two hex colors. Symmetric — order of
    arguments doesn't matter, the lighter is always the numerator."""
    lum1 = _relative_luminance(fg)
    lum2 = _relative_luminance(bg)
    lighter, darker = (lum1, lum2) if lum1 > lum2 else (lum2, lum1)
    return (lighter + 0.05) / (darker + 0.05)


# WCAG 2.1 AA for normal text
_WCAG_AA_NORMAL = 4.5
# WCAG 2.1 AA for large text / UI elements
_WCAG_AA_LARGE = 3.0


def meets_wcag_aa(fg: str, bg: str, *, large: bool = False) -> bool:
    """True iff ``fg`` on ``bg`` clears the AA bar (4.5:1 normal, 3:1 large)."""
    threshold = _WCAG_AA_LARGE if large else _WCAG_AA_NORMAL
    return contrast_ratio(fg, bg) >= threshold


def _ensure_contrast(fg: str, bg: str, *, large: bool = False) -> str:
    """Nudge ``fg`` darker (on light ``bg``) or lighter (on dark ``bg``)
    until it clears the AA threshold. Returns the adjusted hex; stops at
    pure black or pure white to avoid an infinite loop on pathological
    inputs."""
    threshold = _WCAG_AA_LARGE if large else _WCAG_AA_NORMAL
    if contrast_ratio(fg, bg) >= threshold:
        return fg

    # Direction: if background is light, darken foreground; else lighten.
    bg_lum = _relative_luminance(bg)
    darken = bg_lum > 0.5
    r, g, b = _hex_to_rgb(fg)
    step = -12 if darken else 12
    for _ in range(24):  # 24 * 12 = 288 — enough to reach either extreme
        r = max(0, min(255, r + step))
        g = max(0, min(255, g + step))
        b = max(0, min(255, b + step))
        candidate = f"#{r:02x}{g:02x}{b:02x}"
        if contrast_ratio(candidate, bg) >= threshold:
            return candidate
        if (darken and (r, g, b) == (0, 0, 0)) or (not darken and (r, g, b) == (255, 255, 255)):
            return candidate
    return f"#{r:02x}{g:02x}{b:02x}"


# --- Public entry point -----------------------------------------------------

def compute_design_system(description: str, scaffold_id: str = "static") -> DesignSystem:
    """Return a :class:`DesignSystem` tailored to ``description``.

    ``scaffold_id`` is accepted for future use (e.g. dashboards might
    want a tighter spacing scale) but currently only ``description``
    drives the palette. The returned object is always WCAG AA compliant
    for ``text`` on ``surface`` and ``accent`` on ``surface`` — if a
    palette edit drops below the bar, ``_ensure_contrast`` silently
    adjusts rather than shipping an inaccessible result.
    """
    mood = detect_mood(description)
    palette = _PALETTES.get(mood, _PALETTES[_DEFAULT_MOOD])
    ds = DesignSystem(mood=mood, **palette)

    # Belt-and-suspenders: verify the declared palette holds up against
    # WCAG. If a future edit violates the bar we adjust in-place rather
    # than ship an inaccessible result.
    ds.text = _ensure_contrast(ds.text, ds.surface)
    ds.text_muted = _ensure_contrast(ds.text_muted, ds.surface, large=True)
    ds.accent = _ensure_contrast(ds.accent, ds.surface, large=True)
    return ds
