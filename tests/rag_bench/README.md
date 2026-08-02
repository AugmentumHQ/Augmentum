# RAG Pipeline Benchmark Corpus

Synthetic test documents for measuring retrieval accuracy.

## Corpus Provenance
All documents are synthetic — authored specifically for this benchmark.
Not copied from copyrighted sources. Modeled on real document patterns.

## Documents
| File | Type | Words | Topics |
|------|------|-------|--------|
| employment_agreement.md | Legal/formal | ~5000 | Compensation, IP, non-compete, termination, confidentiality, dispute resolution |
| api_reference.md | Technical | ~3000 | Auth, endpoints (users/orders/products), pagination, errors, rate limiting, webhooks |
| research_methods.md | Academic | ~3000 | Quantitative/qualitative, sampling, data collection, statistical analysis, limitations |
| product_catalog.csv | Tabular | ~2000 | Electronics products with specs, pricing, categories |
| meeting_notes.md | Informal | ~2000 | Attendees, pricing discussion, action items, tech debt, feature proposals |

## Labeling Methodology
Ground truth queries (`queries.json`) are hand-labeled with:
- `expected_chunks_contain`: keywords that MUST appear in retrieved chunks
- `expected_chunks_not_contain`: keywords that should NOT appear
- `expected_strategy`: what the query analyzer should classify as
- `should_retrieve`: whether retrieval should trigger

Keyword matching uses word-boundary-aware case-insensitive regex.

## Vocabulary Gaps (by design)
These documents deliberately use different vocabulary than casual queries:
- "compensation" not "salary" or "pay"
- "indemnification" not "protection"
- "pagination" parameters are "offset"/"limit" not "page"/"size"
- Meeting notes use first names only (Sarah, David) not full names
