"""Artifact content sanitizer — strips AI meta-commentary and placeholders.

LLMs often include meta-commentary ("Here is your document:"), placeholder
text ("[Insert data here]"), and filler phrases ("It is worth noting that")
in generated content. This module strips these before rendering, ensuring
clean professional output.
"""

from __future__ import annotations

import re

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Preamble patterns (AI wrapper text before actual content)
# ---------------------------------------------------------------------------

_PREAMBLE_PATTERNS = [
    re.compile(r"^(?:Here(?:'s| is| are) (?:the |a |your |my )?)", re.IGNORECASE),
    re.compile(r"^(?:Below (?:is|are|you'll find) )", re.IGNORECASE),
    re.compile(r"^(?:I(?:'ve| have) (?:created|prepared|written|drafted|generated|compiled|put together) )", re.IGNORECASE),
    re.compile(r"^(?:(?:The|This) (?:following|document|presentation|report|spreadsheet|chart) (?:is|shows|contains|covers|provides|presents|outlines|summarizes) )", re.IGNORECASE),
    re.compile(r"^(?:Let me (?:share|present|show|provide|create) )", re.IGNORECASE),
    re.compile(r"^(?:As requested,? )", re.IGNORECASE),
    re.compile(r"^(?:Based on (?:the |your )?(?:information|data|request|requirements),? )", re.IGNORECASE),
    re.compile(r"^(?:Please find (?:below |attached )?)", re.IGNORECASE),
    re.compile(r"^(?:I'm providing )", re.IGNORECASE),
    re.compile(r"^(?:Certainly[!.]? )", re.IGNORECASE),
    re.compile(r"^(?:Sure[!.]? )", re.IGNORECASE),
    re.compile(r"^(?:Of course[!.]? )", re.IGNORECASE),
    re.compile(r"^(?:Absolutely[!.]? )", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Placeholder patterns (unfilled template markers)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r"\[(?:Insert|Add|Fill in|Enter|Replace with|Placeholder|Your|TBD|TBA|TODO|FIXME|XXX)[^\]]{0,60}\]",
    re.IGNORECASE,
)

_CURLY_PLACEHOLDER_RE = re.compile(
    r"\{(?:Insert|Add|Fill in|Enter|Replace|Your|company|name|data|value)[^\}]{0,60}\}",
    re.IGNORECASE,
)

_GENERIC_PLACEHOLDER_RE = re.compile(
    r"\b(?:TBD|TBA|TK|N/A|TODO|FIXME|XXX|PLACEHOLDER)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Filler phrase patterns (verbose padding that adds no information)
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    re.compile(r"(?:It is (?:important|worth|interesting) to note that )", re.IGNORECASE),
    re.compile(r"(?:In this section,? we (?:will|shall) (?:explore|examine|discuss|look at|consider|analyze) )", re.IGNORECASE),
    re.compile(r"(?:As (?:we )?(?:can|will) see,? )", re.IGNORECASE),
    re.compile(r"(?:(?:Moving|Let(?:'s| us)) (?:on|forward),? )", re.IGNORECASE),
    re.compile(r"(?:Without further (?:ado|delay),? )", re.IGNORECASE),
    re.compile(r"(?:It goes without saying (?:that )?)", re.IGNORECASE),
    re.compile(r"(?:Needless to say,? )", re.IGNORECASE),
    re.compile(r"(?:As (?:an AI|a language model|mentioned (?:earlier|above|previously)),? )", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# AI disclaimer patterns (self-referential caveats)
# ---------------------------------------------------------------------------

_DISCLAIMER_RE = re.compile(
    r"(?:Note:? )?(?:This (?:is|was) (?:a |an )?(?:generated|AI-generated|sample|template|example|draft|mock|placeholder|estimated|approximate) )"
    r"|(?:(?:Please )?(?:note|remember|keep in mind) (?:that )?(?:this|these|the) (?:data|figures|numbers|values|statistics) (?:are|is) (?:for |)"
    r"(?:illustration|demonstration|example|sample|placeholder|estimated|hypothetical))",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_text(text: str) -> str:
    """Clean a single text block (section body, slide bullet, cell value).

    Strips preambles, placeholders, filler phrases, and disclaimers.
    Returns cleaned text, or empty string if nothing remains.
    """
    if not text:
        return text

    result = text.strip()

    # Strip preamble from the start of text.
    # Two strategies applied in sequence:
    # 1. If a preamble pattern matches, remove the prefix AND the rest of
    #    that introductory sentence (up to first sentence-ending punctuation).
    # 2. Then try again (up to 3 passes) to handle chained preambles like
    #    "Certainly! Below is the analysis." -> "the analysis."
    for _pass in range(3):
        matched = False
        for pattern in _PREAMBLE_PATTERNS:
            m = pattern.match(result)
            if m:
                after = result[m.end():]
                # If the remainder has a sentence boundary followed by more
                # text, consume through it (the preamble's trailing clause).
                sentence_end = re.match(r"^[^.!:;\n]*[.!:;]\s+", after)
                if sentence_end and sentence_end.end() < len(after):
                    result = after[sentence_end.end():]
                else:
                    # No further content after the sentence — just strip
                    # the matched preamble prefix.
                    result = after
                matched = True
                break
        if not matched:
            break

    # Remove placeholder brackets
    result = _PLACEHOLDER_RE.sub("", result)
    result = _CURLY_PLACEHOLDER_RE.sub("", result)

    # Remove standalone TBD/TBA/etc. (but not inside words)
    result = _GENERIC_PLACEHOLDER_RE.sub("", result)

    # Remove filler phrases
    for pattern in _FILLER_PATTERNS:
        result = pattern.sub("", result)

    # Remove AI disclaimers
    result = _DISCLAIMER_RE.sub("", result)

    # Collapse whitespace
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def sanitize_heading(heading: str) -> str:
    """Clean a section heading or slide title.

    Strips numbering prefixes, generic labels, and normalizes.
    """
    if not heading:
        return heading

    result = heading.strip()

    # Strip common numbering prefixes: "1.", "1)", "Section 1:", "Part I:"
    result = re.sub(r"^(?:\d+[.)]\s*|Section\s+\d+[:.]\s*|Part\s+[IVX\d]+[:.]\s*)", "", result, flags=re.IGNORECASE)

    # Strip "Slide N:" prefix
    result = re.sub(r"^Slide\s+\d+[:.]\s*", "", result, flags=re.IGNORECASE)

    # Remove placeholder brackets from headings
    result = _PLACEHOLDER_RE.sub("", result)

    return result.strip()


def sanitize_sections(sections: list[dict]) -> list[dict]:
    """Clean all sections for a document artifact.

    Sanitizes headings and bodies. Removes empty sections.
    """
    cleaned = []
    for section in sections:
        heading = sanitize_heading(section.get("heading", ""))
        body = sanitize_text(section.get("body", ""))

        # Skip sections with no content
        if not heading and not body:
            continue

        cleaned.append({
            **section,
            "heading": heading or "Untitled",
            "body": body,
        })

    return cleaned


def sanitize_slides(slides: list[dict]) -> list[dict]:
    """Clean all slides for a presentation artifact.

    Sanitizes titles, bodies, and speaker notes. Removes empty slides.
    """
    cleaned = []
    for slide in slides:
        title = sanitize_heading(slide.get("title", ""))
        body = sanitize_text(slide.get("body", ""))
        notes = sanitize_text(slide.get("notes", ""))

        # Skip slides with no title and no body
        if not title and not body:
            continue

        cleaned.append({
            **slide,
            "title": title or "Untitled",
            "body": body,
            "notes": notes,
        })

    return cleaned


def sanitize_sheets(sheets: list[dict]) -> list[dict]:
    """Clean spreadsheet data.

    Sanitizes sheet names, headers, and cell values.
    Removes placeholder values from cells.
    """
    cleaned = []
    for sheet in sheets:
        name = sheet.get("name", "Sheet1").strip()
        headers = [sanitize_heading(h) if isinstance(h, str) else h for h in sheet.get("headers", [])]

        rows = []
        for row in sheet.get("rows", []):
            cleaned_row = []
            for cell in row:
                if isinstance(cell, str):
                    val = sanitize_text(cell)
                    # Replace fully-placeholder cells with empty string
                    if not val or _GENERIC_PLACEHOLDER_RE.fullmatch(val.strip()):
                        cleaned_row.append("")
                    else:
                        cleaned_row.append(val)
                else:
                    cleaned_row.append(cell)
            rows.append(cleaned_row)

        cleaned.append({
            **sheet,
            "name": name,
            "headers": headers,
            "rows": rows,
        })

    return cleaned


def sanitize_chart_labels(labels: list) -> list:
    """Clean chart labels — remove placeholders."""
    return [
        sanitize_heading(str(l)) if isinstance(l, str) else l
        for l in labels
    ]
