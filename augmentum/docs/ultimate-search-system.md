# Ultimate Search System Design

**Date:** March 2026
**Status:** Design — ready for implementation
**Goal:** Make a 4B model competitive with a 671B model through scaffolding alone

## Problem Statement

Raw search is the weakest link in small-model reasoning. A 671B model can:
1. Decompose a complex query into sub-questions
2. Generate diverse, well-targeted search queries
3. Read 10+ pages and extract relevant facts
4. Cross-reference sources for accuracy
5. Synthesize a coherent answer with citations

A 4B model fails at every one of these steps when given the same tools. The solution: **move the intelligence from the model to the pipeline.** Each stage must be so constrained that even a 4B model executes it reliably, while the orchestration provides the reasoning depth.

## Core Principle: Structured I/O at Every Stage

The key insight from BFCL benchmarks: small models degrade rapidly with open-ended tasks but perform well with constrained output formats. Every stage in this pipeline uses a rigid output schema that the model fills in like a form. No free-text reasoning required until the final synthesis.

## Architecture: 7-Stage Search Pipeline

```
Query → [1. Decompose] → [2. Expand] → [3. Search] → [4. Triage] → [5. Extract] → [6. Cross-Ref] → [7. Brief]
              ↓                ↓             ↓              ↓             ↓              ↓              ↓
         sub-questions    query variants   raw results    ranked URLs   fact sheets   confidence     research brief
```

### Stage 1: Query Decomposition (system-driven + LLM assist)

**Purpose:** Break complex queries into atomic sub-questions that each need one search.

**Why this helps small models:** A 4B model can't hold "compare the economic policies of the last 3 US presidents and their impact on GDP growth" in working memory. But it can answer "what were Obama's key economic policies?" reliably.

**Implementation:**

```python
# System prompt — forces structured output
DECOMPOSE_PROMPT = """\
Break this question into independent sub-questions that can each be answered
with a single web search. Output ONLY this JSON format:

{
  "sub_questions": [
    {"q": "sub-question text", "type": "factual|comparative|temporal|definitional"},
    ...
  ],
  "original_is_simple": true/false
}

Rules:
- Maximum 5 sub-questions
- Each must be self-contained (no pronouns referencing other sub-questions)
- If the question is already simple, set original_is_simple to true and return it as the only sub-question
- "temporal" = needs current/recent data
- "comparative" = needs data from multiple entities to compare
"""
```

**Fallback chain:**
1. LLM decomposition (structured JSON output via tier 1/2)
2. Regex heuristics: split on "and", "vs", "compare", "difference between"
3. Pass-through: treat original query as single sub-question

**FlowStep definition:**
```python
_step(
    "Decompose",
    DECOMPOSE_PROMPT,
    role="classify",        # parsed like classify — no streaming
    output_cap=500,
    sort_order=0,
)
```

### Stage 2: Query Expansion (zero-LLM-cost)

**Purpose:** Generate multiple search query variants per sub-question to cast a wider net. This is the stage that most improves small-model search quality because it compensates for the model's inability to generate good search terms.

**Implementation — entirely system-driven, no LLM call:**

```python
def expand_queries(sub_question: str, q_type: str) -> list[str]:
    """Generate 3-5 search variants per sub-question. No LLM needed."""
    variants = [sub_question]  # original always included

    # 1. Synonym expansion (static word map)
    variants.append(apply_synonym_map(sub_question))

    # 2. Type-specific reformulation
    if q_type == "temporal":
        variants.append(f"{sub_question} 2025 2026")
        variants.append(f"{sub_question} latest")
    elif q_type == "definitional":
        variants.append(f"what is {sub_question}")
        variants.append(f"{sub_question} explained")
    elif q_type == "comparative":
        # Split entities and search each separately
        entities = extract_entities(sub_question)
        for entity in entities[:2]:
            variants.append(f"{entity} {extract_comparison_axis(sub_question)}")

    # 3. Site-scoped variant for authoritative sources
    domain_hint = DOMAIN_AUTHORITY_MAP.get(detect_domain(sub_question))
    if domain_hint:
        variants.append(f"site:{domain_hint} {sub_question}")

    return deduplicate(variants)[:5]

# Domain → authoritative site mapping
DOMAIN_AUTHORITY_MAP = {
    "programming": "stackoverflow.com",
    "science": "scholar.google.com",
    "medical": "pubmed.ncbi.nlm.nih.gov",
    "legal": "law.cornell.edu",
    "math": "mathworld.wolfram.com",
    "history": "britannica.com",
    "news": None,  # no site restriction for news
}

# Synonym map for common search-hindering words
SYNONYM_MAP = {
    "best": ["top", "recommended"],
    "difference": ["vs", "comparison"],
    "how to": ["guide", "tutorial"],
    "why does": ["reason", "cause"],
    "problem": ["issue", "error", "bug"],
}
```

**This stage has zero LLM cost.** It runs in Python, expands query coverage by 3-5x, and is the single biggest ROI improvement for small models.

### Stage 3: Parallel Search Execution

**Purpose:** Execute all expanded queries in parallel via SearXNG, deduplicate results.

**Implementation — extends existing auto-search:**

```python
async def parallel_search(
    query_groups: dict[str, list[str]],  # sub_question -> [expanded queries]
    searxng_client: httpx.AsyncClient,
    *,
    results_per_query: int = 5,
    max_total_results: int = 30,
    category_override: str | None = None,
) -> dict[str, list[SearchResult]]:
    """Execute all queries in parallel, grouped by sub-question."""

    tasks = []
    for sub_q, queries in query_groups.items():
        for query in queries:
            # Use existing WebSearchTool internally
            tasks.append((sub_q, search_one(query, results_per_query, category_override)))

    raw_results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    # Group by sub-question
    grouped: dict[str, list[SearchResult]] = defaultdict(list)
    for (sub_q, _), result in zip(tasks, raw_results):
        if isinstance(result, Exception):
            continue
        grouped[sub_q].extend(result)

    # Deduplicate within each group by URL
    for sub_q in grouped:
        seen_urls = set()
        deduped = []
        for r in grouped[sub_q]:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                deduped.append(r)
        grouped[sub_q] = deduped[:max_total_results // len(query_groups)]

    return grouped
```

**SearXNG category routing** (system-driven, no LLM):
```python
CATEGORY_ROUTING = {
    "temporal": "news",        # current events → news engines
    "factual": "general",
    "definitional": "general",
    "comparative": "general",
    "code": "it",              # programming → IT category
    "science": "science",
}
```

**FlowStep definition:**
```python
_step(
    "Search",
    "Searching across {search_query_count} queries...",
    role="search",          # triggers auto-search machinery
    tool_categories=["search"],
    output_cap=0,           # no cap — raw results stored in context
    sort_order=1,
)
```

### Stage 4: Result Triage (LLM — highly constrained)

**Purpose:** Score each search result's relevance to its sub-question. This is where we pay for one LLM call but make it dead simple.

**Why this matters:** Without triage, a 4B model drowns in irrelevant snippets. With triage, it only sees the 3-5 best results per sub-question.

**Implementation:**

```python
TRIAGE_PROMPT = """\
You are scoring search results for relevance. For each result, output a score 0-3.

Question: {sub_question}

Results:
{numbered_results}

Output ONLY this format — one line per result, nothing else:
1: <score>
2: <score>
3: <score>
...

Scoring:
3 = Directly answers the question
2 = Contains relevant information
1 = Tangentially related
0 = Irrelevant or spam
"""
```

**Parsing — regex, no LLM interpretation needed:**
```python
def parse_triage_scores(output: str) -> dict[int, int]:
    """Parse '1: 3\n2: 1\n...' format."""
    scores = {}
    for match in re.finditer(r"(\d+):\s*(\d)", output):
        idx, score = int(match.group(1)), int(match.group(2))
        scores[idx] = min(3, max(0, score))
    return scores
```

**Batch processing:** Send all results for one sub-question in a single call. With 5 sub-questions × 1 call each = 5 total LLM calls at this stage, each with ~200 tokens input and ~20 tokens output. Trivial cost.

**Cutoff:** Keep results scoring ≥2. If fewer than 2 results score ≥2, keep the top 2 regardless (never zero results).

**FlowStep definition:**
```python
_step(
    "Triage",
    TRIAGE_PROMPT,
    role="analyze",
    output_cap=200,         # just scores
    sort_order=2,
)
```

### Stage 5: Deep Extraction (LLM — structured fact sheets)

**Purpose:** For each surviving result, fetch the full page and extract structured facts. This is the stage that transforms raw web content into model-digestible knowledge.

**Implementation — uses existing WebFetchTool + structured extraction:**

```python
EXTRACT_PROMPT = """\
Extract facts from this web page that answer the question.

Question: {sub_question}
Page content:
{page_content_truncated}

Output ONLY this format:

FACTS:
- <fact 1>
- <fact 2>
- <fact 3>

SOURCE_DATE: <date if found, or "unknown">
SOURCE_TYPE: <news|academic|official|blog|forum|wiki|other>
CONFIDENCE: <high|medium|low>

Rules:
- Maximum 5 facts per page
- Each fact must be a single complete sentence
- Include numbers, dates, and proper nouns
- Do not include opinions unless labeled as such
- If the page doesn't answer the question, output: FACTS: none
"""
```

**Selective fetching:** Only fetch pages that scored ≥2 in triage AND whose snippet alone doesn't contain enough information. Heuristic: if the snippet contains a number or date that matches the question type, skip the full fetch.

```python
def needs_deep_fetch(result: SearchResult, sub_question: str) -> bool:
    """Decide if we need the full page or if the snippet suffices."""
    snippet = result.snippet

    # Snippet has a direct answer (contains numbers for quantitative questions)
    if re.search(r"\d+\.?\d*\s*(%|million|billion|kg|km|USD|\$)", snippet):
        if len(snippet) > 100:
            return False  # snippet is rich enough

    # Short snippets always need deep fetch
    if len(snippet) < 50:
        return True

    return True  # default: fetch
```

**Parallelism:** All deep fetches run concurrently via `asyncio.gather`. Rate-limited to 5 concurrent fetches to avoid overwhelming SearXNG/targets.

**FlowStep definition:**
```python
_step(
    "Extract",
    EXTRACT_PROMPT,
    role="analyze",
    tool_categories=["fetch"],
    output_cap=600,
    sort_order=3,
)
```

### Stage 6: Cross-Reference (LLM — confidence scoring)

**Purpose:** Compare extracted facts across sources. Flag contradictions, identify consensus, assign confidence levels. This is what makes a 4B model's output trustworthy.

**Implementation:**

```python
CROSSREF_PROMPT = """\
Compare these facts from different sources about: {sub_question}

{fact_sheets}

Output ONLY this format:

CONSENSUS:
- <fact agreed by 2+ sources>: [source_1, source_2]
- ...

CONTRADICTIONS:
- <topic>: source_1 says X, source_2 says Y
- ...

SINGLE_SOURCE:
- <fact from only one source>: [source]
- ...

BEST_ANSWER: <the most supported answer to the question, 1-2 sentences>
CONFIDENCE: <high|medium|low>
"""
```

**Key insight:** This prompt requires zero reasoning from the model. It's pure comparison — "does fact A from source 1 match fact B from source 2?" Even a 4B model can do pattern matching across short texts.

**Automatic confidence rules (system-driven, applied after LLM):**
```python
def compute_confidence(crossref_output: str) -> float:
    """System-driven confidence from cross-reference results."""
    consensus_count = crossref_output.count("CONSENSUS:") + len(re.findall(r"^- .+: \[.+,.+\]", crossref_output, re.M))
    contradiction_count = len(re.findall(r"^- .+: source_\d+ says", crossref_output, re.M))
    single_source_count = len(re.findall(r"^- .+: \[\w+\]$", crossref_output, re.M))

    if contradiction_count > 0:
        return max(0.3, 0.7 - (contradiction_count * 0.15))
    if consensus_count >= 3:
        return 0.95
    if consensus_count >= 1:
        return 0.8
    if single_source_count > 0:
        return 0.5
    return 0.4
```

**FlowStep definition:**
```python
_step(
    "Cross-Reference",
    CROSSREF_PROMPT,
    role="verify",          # triggers confidence parsing
    output_cap=800,
    complexity_gate=["moderate", "complex"],
    sort_order=4,
)
```

### Stage 7: Research Brief Assembly (LLM — final synthesis)

**Purpose:** Combine all cross-referenced findings into a pre-digested research brief that gets injected into the main reasoning flow as context. The model that reads this brief doesn't need to search — it just needs to use the facts.

**Implementation:**

```python
BRIEF_PROMPT = """\
Write a research brief answering the original question using ONLY the verified findings below.

Original question: {query}

Verified findings:
{all_crossref_outputs}

Format the brief as:

## Key Findings
- <finding 1> (confidence: high/medium/low, sources: N)
- <finding 2> ...

## Data Points
- <specific numbers, dates, statistics>

## Contradictions & Caveats
- <any unresolved disagreements between sources>

## Sources
- [1] <title> — <url>
- [2] ...

Rules:
- Only include facts that appeared in the findings
- Never invent or extrapolate beyond what sources say
- Flag low-confidence items explicitly
- Maximum 400 words
"""
```

**FlowStep definition:**
```python
_step(
    "Brief",
    BRIEF_PROMPT,
    role="respond",
    stream_to_user=False,   # brief is context, not the final answer
    output_cap=1500,
    sort_order=5,
)
```

## Integration with Flow Builder

### New Built-in Template: "Deep Research"

```python
def deep_research_flow() -> ReasoningFlow:
    """7-stage search pipeline with cross-referencing."""
    return ReasoningFlow(
        id=_id(),
        name="Deep Research",
        description="Multi-source research with cross-referencing and fact-checking. "
                    "Optimized for small models — each stage is simple enough for 4B parameters.",
        icon="",
        is_builtin=True,
        auto_search=False,      # we handle search ourselves in Stage 3
        trigger_domains=["research", "news", "current_events", "science", "history"],
        trigger_keywords=["research", "investigate", "find out", "look up", "sources"],
        max_tool_calls_per_step=10,  # deep fetch needs more
        steps=[
            _step("Decompose", DECOMPOSE_PROMPT, role="classify", output_cap=500, sort_order=0),
            _step("Search", "...", role="search", tool_categories=["search"], output_cap=0, sort_order=1),
            _step("Triage", TRIAGE_PROMPT, role="analyze", output_cap=200, sort_order=2),
            _step("Extract", EXTRACT_PROMPT, role="analyze", tool_categories=["fetch"], output_cap=600, sort_order=3),
            _step("Cross-Reference", CROSSREF_PROMPT, role="verify", output_cap=800,
                  complexity_gate=["moderate", "complex"], sort_order=4),
            _step("Brief", BRIEF_PROMPT, role="transform", output_cap=1500, sort_order=5),
            _step("Respond", RESPOND_PROMPT, role="respond", stream_to_user=True, output_cap=0, sort_order=6),
        ],
    )
```

### Modified "Standard" Template with Search Enhancement

The existing Standard flow gets an upgraded Research step:

```python
# Before (current):
_step("Research", _STANDARD_RESEARCH, role="search", tool_categories=["search", "fetch"], ...)

# After (enhanced):
# The Research step now uses the query expansion system internally.
# When auto_search triggers, instead of passing raw queries to SearXNG,
# it runs them through expand_queries() first.
```

This means **every flow that uses auto_search benefits from query expansion** without any template changes.

### Template Variable Extensions

New variables available in step prompts:

| Variable | Content |
|---|---|
| `{sub_questions}` | JSON list from decompose stage |
| `{search_results_grouped}` | Results organized by sub-question |
| `{triage_scores}` | Scored results with relevance ratings |
| `{fact_sheets}` | Extracted facts from deep fetch |
| `{crossref_summary}` | Cross-reference consensus/contradictions |
| `{research_brief}` | Final assembled brief |
| `{source_count}` | Number of unique sources found |
| `{confidence_level}` | Overall confidence (high/medium/low) |

## Adaptive Depth

Not every query needs 7 stages. The system adapts based on the decompose output:

```python
DEPTH_RULES = {
    # original_is_simple=True, 1 sub-question → shallow
    "shallow": {
        "stages": ["Search", "Brief", "Respond"],
        "max_results": 5,
        "deep_fetch": False,
    },
    # 2-3 sub-questions → standard
    "standard": {
        "stages": ["Decompose", "Search", "Triage", "Extract", "Brief", "Respond"],
        "max_results": 15,
        "deep_fetch": True,
    },
    # 4-5 sub-questions OR comparative type → deep
    "deep": {
        "stages": ["Decompose", "Search", "Triage", "Extract", "Cross-Reference", "Brief", "Respond"],
        "max_results": 30,
        "deep_fetch": True,
    },
}

def select_depth(decompose_output: dict) -> str:
    sub_qs = decompose_output.get("sub_questions", [])
    if decompose_output.get("original_is_simple") and len(sub_qs) <= 1:
        return "shallow"
    if len(sub_qs) >= 4 or any(q["type"] == "comparative" for q in sub_qs):
        return "deep"
    return "standard"
```

**This maps directly to complexity_gate** — shallow queries skip gated steps automatically.

## What Makes This Beat a 671B Model

| Capability | 671B Model (raw) | 4B + This Pipeline |
|---|---|---|
| Query decomposition | Good but inconsistent | Forced structure → reliable |
| Search query quality | Good single query | 3-5x variants via expansion (no LLM cost) |
| Result volume | 1 search, 5 results | 5 sub-Qs × 5 variants × 5 results = up to 125 candidates |
| Relevance filtering | Reads everything | Triage scores → only reads top results |
| Deep reading | Can read pages | Same — WebFetchTool + Trafilatura |
| Cross-referencing | Sometimes, inconsistently | Always, with structured consensus/contradiction |
| Confidence calibration | Hallucinated confidence | System-computed from source agreement |
| Source attribution | Sometimes forgets | Mandatory in brief format |
| Total LLM calls | 1 (monolithic) | 5-8 (each trivially simple) |

The 671B model has to do everything in one pass. This pipeline breaks the problem into steps so simple that even a 4B model can't get them wrong:
- **Decompose:** Fill in a JSON template
- **Triage:** Output "1: 3, 2: 1, ..." (10 tokens)
- **Extract:** Copy facts from text (extractive, not generative)
- **Cross-ref:** Compare short bullet lists
- **Brief:** Reorganize existing text

## Implementation Plan

### Phase 1: Query Expansion (zero LLM cost, immediate benefit)

**Files:**
- `augmentum/search/expander.py` — `expand_queries()`, synonym map, domain authority map, category routing
- Wire into existing auto-search path in `executor.py` and `engine.py`

**Effort:** ~150 lines. Immediately improves every flow that uses auto-search.

### Phase 2: Result Triage

**Files:**
- `augmentum/search/triage.py` — triage prompt, score parser, batch executor

**Effort:** ~100 lines. Reduces noise in search context by ~60%.

### Phase 3: Deep Extraction

**Files:**
- `augmentum/search/extractor.py` — fact extraction prompt, selective fetch logic

**Effort:** ~120 lines. Requires existing WebFetchTool.

### Phase 4: Cross-Reference + Brief Assembly

**Files:**
- `augmentum/search/crossref.py` — cross-reference prompt, confidence computation
- `augmentum/search/brief.py` — brief assembly prompt

**Effort:** ~150 lines.

### Phase 5: Deep Research Flow Template

**Files:**
- `augmentum/reasoning/templates.py` — add `deep_research_flow()`
- `augmentum/reasoning/variables.py` — add new template variables

**Effort:** ~80 lines.

### Phase 6: Adaptive Depth

**Files:**
- `augmentum/search/depth.py` — depth selection rules, stage filtering

**Effort:** ~60 lines.

**Total:** ~660 lines of new code across 6 files + template additions.

## Config Settings

```python
# Search pipeline
search_expansion_enabled: bool = True           # Stage 2: query expansion
search_expansion_max_variants: int = 5          # variants per sub-question
search_triage_enabled: bool = True              # Stage 4: result scoring
search_triage_min_score: int = 2                # minimum score to keep
search_deep_fetch_enabled: bool = True          # Stage 5: full page extraction
search_deep_fetch_max_concurrent: int = 5       # parallel fetch limit
search_crossref_enabled: bool = True            # Stage 6: cross-referencing
search_brief_max_words: int = 400               # Stage 7: brief length cap
search_adaptive_depth: bool = True              # auto-select shallow/standard/deep
```

## LLM Cost Analysis (per query)

| Stage | LLM Calls | Input Tokens (est.) | Output Tokens (est.) |
|---|---|---|---|
| Decompose | 1 | ~200 | ~100 |
| Expand | 0 | 0 | 0 |
| Search | 0 | 0 | 0 |
| Triage | 1-5 | ~200 each | ~20 each |
| Extract | 3-10 | ~500 each | ~100 each |
| Cross-Ref | 1-5 | ~400 each | ~200 each |
| Brief | 1 | ~800 | ~400 |
| **Total** | **7-22** | **~5-10K** | **~1-3K** |

For a 4B local model, this is ~15 seconds total inference. For an API model at $0.15/1M tokens, this is ~$0.002 per query. The infrastructure cost (SearXNG + fetching) dominates, not the LLM cost.

## Comparison to Existing System

| Feature | Current Auto-Search | This Design |
|---|---|---|
| Query generation | LLM generates 2-4 queries | LLM decomposes → system expands 3-5x per sub-Q |
| Result handling | Raw snippets dumped into context | Triage → selective deep fetch → structured facts |
| Cross-referencing | None (model does it ad-hoc) | Explicit stage with consensus/contradiction tracking |
| Confidence | Model self-reports (unreliable) | System-computed from source agreement |
| Retry strategy | VERIFY flags SEARCH_NEEDED → refine | Built into pipeline: low-triage scores trigger expansion |
| Context injection | Phase-scoped (full/brief/URLs) | Pre-digested brief — same format regardless of phase |
| Small-model support | Degrades significantly | Each stage designed for 4B capability |
| LLM calls for search | 1 (query gen) + tools | 7-22 (but each is trivially simple) |
