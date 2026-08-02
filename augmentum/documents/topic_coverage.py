"""Topic coverage mapping — active negative detection for RAG.

Builds a lightweight topic fingerprint per document at ingest time.
At query time, checks if the query's topic is covered by ANY document.
If not, returns an explicit "not covered" signal instead of searching
and injecting irrelevant noise.

Novel technique: standard RAG handles negatives passively (nothing scored
high, so inject nothing). This actively detects what ISN'T in the corpus
and gives the LLM explicit absence information.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Module-level document topic maps
_doc_topics: dict[str, set[str]] = {}  # doc_id -> set of topic terms
_doc_names: dict[str, str] = {}  # doc_id -> filename
_coverage_ready = False

# Words to exclude from topic extraction. Three layers:
# 1. Document structure words (section, paragraph, etc.)
# 2. Legal/formal filler (pursuant, herein, thereof)
# 3. Generic English words an LLM can fill in without context —
#    these tell you nothing about what a document is ABOUT
_TOPIC_STOP = frozenset({
    # Structure
    "section", "article", "paragraph", "page", "table", "figure",
    "appendix", "note", "example", "item", "list", "part", "chapter",
    # Legal filler
    "following", "above", "below", "pursuant", "herein", "thereof",
    "shall", "may", "must", "will", "would", "could", "should",
    "provided", "including", "without", "within", "under", "upon",
    "between", "through", "during", "before", "after",
    # Generic English — an LLM knows these without any document
    "information", "business", "time", "date", "notice", "prior",
    "written", "parties", "reason", "company", "based", "provide",
    "required", "include", "related", "applicable", "respect",
    "period", "services", "agreement", "right", "rights", "terms",
    "conditions", "general", "specific", "certain", "subject",
    "event", "case", "extent", "purpose", "manner", "regard",
    "order", "form", "nature", "person", "place", "state",
    "action", "result", "effect", "process", "system", "level",
    "point", "number", "value", "type", "name", "role", "description",
    "true", "false", "null", "string", "integer", "boolean",
    "request", "response", "data", "status", "code", "error",
    "method", "field", "parameter", "option", "default", "required",
    "used", "using", "make", "made", "also", "well", "however",
    "addition", "accordance", "connection", "respect",
})


def _extract_section_headers(text: str) -> list[str]:
    """Extract section/heading text from markdown-style headers."""
    headers = []
    for line in text.split("\n"):
        line = line.strip()
        # Markdown headers: # Title, ## Section, ### Subsection
        if line.startswith("#"):
            header = re.sub(r'^#+\s*', '', line).strip()
            if header and len(header) > 2:
                headers.append(header.lower())
        # Numbered sections: 1. Title, 1.1 Title
        elif re.match(r'^\d+(?:\.\d+)*\.?\s+[A-Z]', line):
            header = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', line).strip()
            if header and len(header) > 2:
                headers.append(header.lower())
    return headers


def _extract_topic_terms(text: str, max_terms: int = 80) -> set[str]:
    """Extract high-signal topic terms from document text.

    Combines section headers (strongest signal) with TF-IDF-like
    frequency analysis of content words.
    """
    from augmentum.documents.store import _STOP_WORDS

    topics: set[str] = set()

    # Section headers are the strongest topic signal
    headers = _extract_section_headers(text)
    for header in headers:
        # Extract meaningful words from headers
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', header)]
        for w in words:
            if w not in _STOP_WORDS and w not in _TOPIC_STOP:
                topics.add(w)

    # Top frequency content words as supplementary topics
    all_words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered = [w for w in all_words if w not in _STOP_WORDS and w not in _TOPIC_STOP]
    freq = Counter(filtered)

    # Take top N most frequent terms
    for term, _ in freq.most_common(max_terms):
        topics.add(term)

    return topics


def build_topic_map(documents: list[dict]) -> int:
    """Build topic fingerprints for all documents.

    documents: list of {"id": str, "filename": str, "content": str}

    Call once after ingesting all documents.
    Returns number of documents mapped.
    """
    global _doc_topics, _doc_names, _coverage_ready  # noqa: PLW0603

    _doc_topics.clear()
    _doc_names.clear()

    # First pass: extract raw topics per document
    raw_topics: dict[str, set[str]] = {}
    for doc in documents:
        doc_id = doc["id"]
        _doc_names[doc_id] = doc["filename"]
        raw_topics[doc_id] = _extract_topic_terms(doc["content"])

    # IDF filter: remove terms that appear in >60% of documents
    # (they're too generic to be distinctive)
    n_docs = len(documents)
    if n_docs >= 3:
        term_doc_count: dict[str, int] = {}
        for topics in raw_topics.values():
            for t in topics:
                term_doc_count[t] = term_doc_count.get(t, 0) + 1

        idf_threshold = n_docs * 0.6
        noise_terms = {t for t, c in term_doc_count.items() if c > idf_threshold}
    else:
        noise_terms = set()

    # Second pass: apply IDF filter and store
    for doc_id, topics in raw_topics.items():
        filtered = topics - noise_terms
        _doc_topics[doc_id] = filtered

        log.debug("topic_map_built",
                  doc=_doc_names[doc_id],
                  raw=len(topics),
                  filtered=len(filtered),
                  idf_removed=len(topics - filtered),
                  sample=sorted(filtered)[:10])

    _coverage_ready = True
    return len(documents)


def check_topic_coverage(query: str, threshold: float = 0.1) -> dict:
    """Check if the query's topic is covered by any document.

    Returns:
        {
            "covered": bool,
            "best_match_doc": str | None,  # filename of best matching doc
            "best_match_score": float,      # topic overlap score
            "signal": str | None,           # negative signal text if not covered
        }
    """
    if not _coverage_ready or not _doc_topics:
        return {"covered": True, "best_match_doc": None,
                "best_match_score": 0.0, "signal": None}

    from augmentum.documents.store import _STOP_WORDS

    # Extract query terms
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    query_terms = query_words - _STOP_WORDS - _TOPIC_STOP

    if not query_terms:
        return {"covered": True, "best_match_doc": None,
                "best_match_score": 0.0, "signal": None}

    # Score each document by topic overlap
    best_score = 0.0
    best_doc = None

    for doc_id, topics in _doc_topics.items():
        if not topics:
            continue
        overlap = query_terms & topics
        # Jaccard-like: overlap / query terms
        score = len(overlap) / len(query_terms) if query_terms else 0.0
        if score > best_score:
            best_score = score
            best_doc = _doc_names.get(doc_id)

    covered = best_score >= threshold

    signal = None
    if not covered:
        topic_desc = ", ".join(sorted(query_terms)[:5])
        signal = (
            f"[context_note: The available documents do not appear to cover "
            f"the topic of this query ({topic_desc}). The response should "
            f"indicate that this information is not available in the "
            f"provided documents.]"
        )

    return {
        "covered": covered,
        "best_match_doc": best_doc,
        "best_match_score": round(best_score, 3),
        "signal": signal,
    }


def reset() -> None:
    """Reset topic maps (for testing)."""
    global _doc_topics, _doc_names, _coverage_ready  # noqa: PLW0603
    _doc_topics = {}
    _doc_names = {}
    _coverage_ready = False
