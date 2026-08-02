# Azure OpenAI — Provider Reference Card

> **Verbatim reference** from Microsoft Learn (Azure Foundry / Azure OpenAI REST reference).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `azure` |
| **`provider_type`** | `openai` |
| **Base URL** | **""** (instance-specific — user provides `https://{resource}.openai.azure.com`) |
| **Auth** | `api-key: <KEY>` header (API-Key) **or** `Authorization: Bearer <token>` (Microsoft Entra ID) |
| **Models** | OpenAI family, addressed by **deployment id** (not raw model id) |

---

## 1. Augmentum wiring (what WE send)

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `base_url` | `""` | user must supply full resource URL |
| `auth_type` / `auth_header` | `api-key` / `api-key` | ✅ matches Azure API-Key auth |
| _(no OpenAI-family flags)_ | — | relies on `is_openai_family_model` catch-all so gpt-5/o-series deployments still get `max_completion_tokens` / `reasoning_effort` / `developer` |

⚠️ **Mandatory `api-version`** (see §2/§5) — the profile provides no injection helper.

---

## 2. Endpoint, versioning & auth (verbatim)

**URL structure:**
```http
POST https://{your-resource-name}.openai.azure.com/openai/deployments/{deployment-id}/chat/completions?api-version=2024-06-01
```
- `endpoint` (path, **required**): `https://{your-resource-name}.openai.azure.com`
- `deployment-id` (path, **required**): your model deployment name
- **`api-version` (query, REQUIRED)** — "All versions follow the `YYYY-MM-DD` date structure." Examples: `2024-06-01`, `2024-10-21`. Latest **GA = `v1`**, latest preview = `v1 preview` (new preview inference API).

**Auth (verbatim):**
- **API Key:** "all API requests must include the API Key in the `api-key` HTTP header."
- **Entra ID:** token in `Authorization` header, prefixed `Bearer`.

**API surfaces:** Control plane · Data plane authoring · Data plane inference (each with its own preview/GA cadence; preview ≈ monthly).

---

## 3. Request parameters

Chat-completions body **mirrors OpenAI** (Azure points to the OpenAI REST schema for chat/embeddings/completions). See [`openai.md` §3](openai.md#3-request-body-parameters-verbatim--post-v1chatcompletions): `messages` (system/developer/user/assistant/tool), `max_completion_tokens`, `reasoning_effort`, `temperature`, `top_p`, `response_format`, `tools`, etc.

**Azure-specific response fields:** `prompt_filter_results` + `content_filter_results` — per-category (`hate`/`sexual`/`violence`/`self_harm`) severity (`safe`/`very_low`/`low`/`medium`/`high`), plus `jailbreak`, `profanity`, customer `blocklist`. Returned alongside choices.

---

## 4. Pricing

Per the Azure pricing page (region-dependent; mirrors OpenAI list prices for the same models).
Verbatim numbers **not captured** (JS/region-gated page) — **TODO**.

---

## 5. Known drift / gaps

- 🟡 **`api-version` mandatory** — every Azure call needs `?api-version=YYYY-MM-DD` (or `v1`) **and** the `/openai/deployments/{id}/chat/completions` path. Profile `base_url=""` offers no injection → works only if the user encodes both into their stored base_url. **Verify** the backend's URL-join preserves the query string when it appends `/chat/completions`.
- 🟢 **OpenAI-only fields via catch-all** — gpt-5/o-series Azure deployments get the full OpenAI-family set via `is_openai_family_model`; confirm Azure accepts `service_tier`/`prompt_cache_key` (these are OpenAI-direct features and may 400 on Azure).
- ☐ Verbatim pricing + per-deployment context/output limits not captured.

---

## 6. Sources

- Azure OpenAI REST reference (Foundry): https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
- Chat schema: https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/chat
