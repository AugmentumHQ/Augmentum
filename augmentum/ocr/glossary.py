"""Chapter glossary — the spelling prior a per-page read structurally lacks.

A vision model reading page 12 in isolation has no way to know the character is
called GOJO. It sees ambiguous glyphs and resolves them locally, so ``GOJP`` /
``COJO`` / ``GOJO`` all look equally defensible and it picks a different one
each page. That is the dominant residual error once ordering is fixed: not
"couldn't read it" but "read it slightly differently every time".

The fix is to hand the second pass the proper nouns the chapter has already
established. What we deliberately do NOT hand it is the prior prose:

* **Error propagation.** A name misread once on page 3 would become
  authoritative for pages 4-30. The output then gets *more* self-consistent
  while being uniformly wrong, which is worse than noisy — noise is visible,
  uniform error is not. The ``min_sightings`` floor is the whole defence: a term
  has to be read the same way on two DIFFERENT pages before it can influence
  anything. Independent agreement is the only cheap evidence available that a
  reading was real rather than a one-off glyph slip.
* **Anchoring.** Given prior dialogue, a model will echo it — re-emitting a line
  from the previous page, or bending new dialogue toward old phrasing. A bare
  name list is not something you can accidentally copy sentences out of.
* **Context growth.** Chapter-so-far prose is unbounded and by page 40 is both
  crowding the window and diluting attention. A capped term list is flat.

Extraction is intentionally dumb — capitalized and ALL-CAPS tokens minus a
stoplist. It does not need precision: a wrong term that survives the sightings
floor costs a few tokens of prompt, while a missed one just leaves the second
pass no worse off than the first. Under-reaching is the safe direction.
"""

from __future__ import annotations

import re
from collections import defaultdict

__all__ = ["build_glossary", "render_glossary", "merge_page_terms"]

# Tokens worth tracking: Capitalized words and ALL-CAPS runs, apostrophes kept
# so "Gojo's" contributes "Gojo". Length floors drop "A", "Ok", "Mr".
_TERM = re.compile(r"\b([A-Z][a-z']{2,}|[A-Z]{3,}[A-Z']*)\b")

# Words that are capitalized for grammar or convention rather than because they
# name something. Comic dialogue is short, exclamatory and sentence-initial-
# heavy, so without this the list fills up with "What", "Don't", "Never".
_STOP = frozenset("""
about after again against all almost also always and another any are around
because been before being both but came can cannot come could
damn didn did does doing done don down
each else even ever every everyone everything
first for from
get getting going gone good got
had has have having hell her here hers herself him himself his how
into its itself
just
know knew
let like little look looking
made make many maybe more most much must
never new next nothing now
off once one only other our out over own
please
really right
same say see seen shall she should since some someone something still stop such sure
take tell than that the their them then there these they thing think this those though
through time too took
under until
very
wait want was way well went were what when where which while who why will with would
yeah yes yet you your yours yourself
aah agh argh ahh gah grr hah heh hmm huh oof ooh tch ugh
""".split())


def _terms_in(text: str) -> set[str]:
    """Distinct candidate terms in one line, normalized to a canonical form.

    Returns a SET, not a count: a term repeated five times in one line is one
    sighting, not five. Sightings are meant to measure independent agreement,
    and a name shouted three times in the same balloon is a single reading.
    """
    found: set[str] = set()
    for raw in _TERM.findall(text or ""):
        term = raw.strip("'").strip()
        if len(term) < 3 or term.lower() in _STOP:
            continue
        found.add(term)
    return found


def merge_page_terms(counts: dict[str, set[int]], page: int, lines: list[dict]) -> None:
    """Fold one page's lines into ``counts`` (term → set of pages seen on).

    Keyed by page rather than by occurrence so a chatty page can't single-
    handedly promote its own misreading — the sightings floor means "seen on N
    different pages", which is the property that makes it evidence.
    """
    for ln in lines or []:
        for term in _terms_in((ln or {}).get("text") or ""):
            counts[term].add(page)


def build_glossary(pages: list[dict]) -> dict[str, set[int]]:
    """Rebuild the term→pages map from persisted narration pages.

    Called at job start so a resumed or re-run chapter begins with everything
    the previous run learned, instead of relearning it page by page. This is
    also why a second listen transcribes better than the first: page 1 gets the
    glossary that only existed by page 30 last time.
    """
    counts: dict[str, set[int]] = defaultdict(set)
    for p in pages or []:
        if not isinstance(p, dict):
            continue
        merge_page_terms(counts, int(p.get("page", -1)), p.get("lines") or [])
    return counts


def render_glossary(
    counts: dict[str, set[int]],
    *,
    min_sightings: int = 2,
    max_terms: int = 40,
    exclude_page: int | None = None,
) -> list[str]:
    """Terms established well enough to be used as a spelling reference.

    ``exclude_page`` drops sightings from the page currently being refined, so a
    term the draft just invented on THIS page can't vouch for itself — the
    glossary must be evidence from elsewhere in the chapter, or it is only
    telling the model what it already said.

    Ordered by how widely attested a term is, then alphabetically so the prompt
    prefix is stable across pages (a reordered list is a needlessly cold cache).
    """
    scored: list[tuple[int, str]] = []
    for term, pages in counts.items():
        seen = pages - {exclude_page} if exclude_page is not None else pages
        if len(seen) >= max(1, min_sightings):
            scored.append((len(seen), term))
    scored.sort(key=lambda t: (-t[0], t[1].lower()))
    return [term for _, term in scored[: max(0, max_terms)]]
