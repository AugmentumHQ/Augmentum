# Embeddings / Rerank (cloud) — Combined Reference Card

> **Verbatim reference** for cloud **embeddings + rerank** providers. ⚠️ **Coverage gap:**
> Augmentum's `reranker`/`embeddings` are currently **local-only** — there is no cloud
> embeddings/rerank backend abstraction yet. These cards are forward-looking reference for
> the planned cloud backend. **Sourced:** 2026-06-25 · Sources per §.

---

## 1. Voyage AI — embeddings + rerank

| | |
|---|---|
| **Base URL** | `https://api.voyageai.com/v1` · `POST /embeddings` · `POST /rerank` |
| **Auth** | `Authorization: Bearer <KEY>` |

**Embedding models:** `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, **`voyage-code-3`**, `voyage-finance-2`, `voyage-law-2`.
**Params:** `input` (str or list, max 1000), `model`, **`input_type`** (`query`/`document` — adds retrieval prompts; **must match at index + query time**), `output_dimension` (256/512/1024/2048 — Matryoshka), `output_dtype` (`float`/`int8`/`uint8`/`binary`/`ubinary` — quantization), `truncation` (def true), `encoding_format` (base64).
**Per-request token caps:** lite 1M · v4/3.5 320K · large/code-3 120K.
**Rerank:** `rerank-2.5` — `query`, `documents`, `top_k`, `return_documents`. Context-embeddings variant available.
**Source:** docs.voyageai.com/reference/embeddings-api

---

## 2. Jina AI — embeddings + rerank

| | |
|---|---|
| **Base URL** | `https://api.jina.ai/v1` · `POST /embeddings` · `POST /rerank` |
| **Auth** | `Authorization: Bearer <KEY>` |

**Models:** `jina-embeddings-v5-omni` (latest — text/image/audio/video, 1.6B/0.9B), `jina-embeddings-v4` (3.8B multimodal, 32K ctx, dense + **late-interaction**), `jina-embeddings-v3` (text, 89 langs, 8192 ctx).
**Params:** `input` (text or image URL/base64), `model`, **`task`** (`retrieval.query`/`retrieval.passage`/`text-matching`/`classification`), `dimensions`, **`late_chunking`**, `normalized` (L2), `truncate`.
**Reranker:** `jina-reranker-v2` / `jina-reranker-m0` (multimodal).
**Pricing:** free 100 RPM/100K TPM · paid **$0.02 / 1M input tokens** (500 RPM / 2M TPM) · premium 5000 RPM/50M TPM. Images tile-billed.
**Source:** jina.ai/embeddings

---

## 3. Cohere — embeddings + rerank

| | |
|---|---|
| **Base URL** | `https://api.cohere.com/v2` · `POST /embed` · `POST /rerank` |
| **Auth** | `Authorization: Bearer <KEY>` (same key as [cohere.md](cohere.md)) |

**Embed models:** `embed-v4.0` (multimodal, Matryoshka dims), `embed-english-v3.0`, `embed-multilingual-v3.0`.
**Params:** `texts` / `images`, `model`, **`input_type`** (`search_document`/`search_query`/`classification`/`clustering` — **required**), **`embedding_types`** (`float`/`int8`/`uint8`/`binary`/`ubinary`), `output_dimension` (v4: 256/512/1024/1536), `truncate`.
**Rerank:** `rerank-v3.5` — `query`, `documents`, `top_n`, `max_tokens_per_doc`.
**Source:** docs.cohere.com/reference/embed · /rerank

---

## 4. Mistral — embeddings (+ OCR)

| | |
|---|---|
| **Base URL** | `https://api.mistral.ai/v1` · `POST /embeddings` |
| **Auth** | `Authorization: Bearer <KEY>` (same key as [mistral.md](mistral.md)) |

**Models:** `mistral-embed` (general text, 1024 dims), `codestral-embed` (code retrieval, configurable `output_dimension` + `output_dtype`).
**Params:** `input` (str or list), `model`, `output_dimension`, `output_dtype`.
**Also:** Mistral **OCR** (`mistral-ocr-latest`, `POST /v1/ocr`) — document→markdown w/ layout, for RAG ingest.
**Source:** docs.mistral.ai/capabilities/embeddings

---

## Known drift / gaps (embeddings/rerank)

- 🟡 **No cloud backend** — Augmentum has no cloud embeddings/rerank abstraction; retrieval (knowledge packs, memory) uses **local** models only. Wiring any of the above requires a new backend type + provider rows. Tracked as the headline gap.
- 🟢 **`input_type` asymmetry** — Voyage/Cohere/Jina all require **matching** query-vs-document/passage typing at index time and query time; a cloud backend must thread the role through both `embed` calls or retrieval quality silently degrades.
- 🟢 **Quantized dtypes** (`int8`/`binary`/Matryoshka dims) cut storage ~4–32× — worth exposing if a cloud embeddings backend lands (sqlite-vec stores them directly).
