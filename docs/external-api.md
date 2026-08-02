# Augmentum External API (OpenAI-compatible)

Augmentum exposes an OpenAI-compatible surface so any OpenAI client can use the
box as a drop-in endpoint — for chat **and** the other modalities — just by
changing the base URL. This doc covers the endpoints, authentication, and
(most importantly) how to choose **how much of Augmentum's machinery runs on a
request**: raw passthrough, passthrough + memory, or the full in-app pipeline.

## Endpoints

| Modality | Endpoint | Notes |
|---|---|---|
| Chat | `POST /v1/chat/completions` | streaming + non-streaming |
| Embeddings | `POST /v1/embeddings` | |
| Models | `GET /v1/models` | lists available models |
| Text-to-speech | `POST /v1/audio/speech` | gated by `audio_tts_enabled` |
| Speech-to-text | `POST /v1/audio/transcriptions` | gated by `audio_stt_enabled` |
| Image generation | `POST /v1/images/generations` | local engine or cloud provider |
| Image edits | `POST /v1/images/edits` | OpenAI image-edit shape |
| Image models | `GET /v1/image-models` | lists image models |

Notes:
- There is **no `/v1/completions`** (legacy text-completion) — use
  `/v1/chat/completions`.
- **Voice listing is on the native surface** (`GET /api/audio/voices`); OpenAI
  defines no `/v1` voices route, so neither does Augmentum.
- Augmentum-specific extras also live under `/v1/`: `/v1/memory`,
  `/v1/memory/search`, `/v1/memory/store`, `/v1/mcp`, `/v1/coder/permissions`.

There is also a richer **native** surface under `/api/` (e.g. `/api/chat`,
`/api/audio/voices`, `/api/image/models`) that the web UI uses. The `/v1/`
routes are the external-client-facing OpenAI-compat layer; `/api/` is the
internal one. Same split as everything else: `/api/chat` internal,
`/v1/chat/completions` external-compat.

Beyond OpenAI compatibility, Augmentum speaks **MCP both ways**:
- As an MCP **server** at `/mcp/` (Streamable HTTP), exposing its tools + your
  private memory to MCP clients (Claude Desktop/Code, Cursor) —
  [Connect to Claude via MCP](connect-to-claude-mcp.md).
- As an MCP **client**, pulling other MCP servers' tools *into* Augmentum so
  your chat/companion/coder can use them —
  [Connect external MCP servers](connect-external-mcp-servers.md).

## Authentication

**Every `/v1/*` endpoint requires authentication** — `/v1` is not in the auth
middleware's public-path allowlist (`augmentum/auth/middleware.py`). External
callers authenticate with an **Augmentum API key** (`sk-aug-…`, SHA-256 hashed
in `augmentum_api_keys`, mapped to a `user_id`):

```
Authorization: Bearer sk-aug-xxxxxxxx
# or
x-api-key: sk-aug-xxxxxxxx
```

All data is scoped by the key's `user_id`, so one external caller never sees
another's sessions, memories, or artifacts.

## Limits

These guard a box that's exposed beyond a trusted LAN. All are admin-tunable
(0 disables a cap); the request-body and per-endpoint caps are **on by
default**, the rate limiter is **opt-in**.

| Limit | Setting | Default |
|---|---|---|
| Request body (general) | `max_request_body_bytes` | 50 MB |
| Request body (uploads) | `files_upload_max_request_bytes` | 500 MB |
| Embeddings batch size | `api_embeddings_max_items` | 2048 items |
| Embeddings total input | `api_embeddings_max_chars` | 1,000,000 chars |
| TTS input length | `api_tts_max_chars` | 50,000 chars |
| STT audio size | `api_stt_max_bytes` | 25 MB |
| Image jobs in flight per user | `image_max_inflight_per_user` | 6 (queue holds 10) |
| Rate limiting (opt-in) | `rate_limit_enabled` (+ `rate_limit_*_rpm`) | off; chat 30/min when on |

Over-limit requests return `413` (size) or `400` (count). The body cap is
enforced before the body is parsed (`_MaxBodySizeMiddleware`), including
chunked/header-less uploads.

## Request modes — how much machinery runs

A chat request runs through a **mode**. The mode decides which context the
route layer injects before the model is called. Pick a mode explicitly with
**either** a header **or** a model-name prefix (header wins):

| Mode | Header value | Model prefix | What it does |
|---|---|---|---|
| **Direct** | `direct` | `d/` | **Raw passthrough — verbatim, zero injection.** |
| **Passthrough** | `passthrough` | `p/` | Proxy to the model + the standard context bundle (memory, etc.). |
| Analytical | `analytical` | `a/` | Research/verification pipeline (UARF). |
| Narrative | `narrative` | `n/` | Roleplay/story memory + entity state. |
| Agentic | `agentic` | `g/` | Autonomous planning loop. |
| Coder | `coder` | `c/` | Containerized coding agent (no personal memory). |

Header: `X-Augmentum-Mode: <value>`. Prefix: set `"model": "d/<model-name>"`.
With **neither**, a classifier picks the mode heuristically and defaults to
**Passthrough** (`augmentum/classifier/router.py`).

### (A) Raw passthrough — no Augmentum features

Use **Direct mode**. It short-circuits *every* injector
(`augmentum/proxy/openai_routes.py` — the `Mode.DIRECT` branch) and forwards
your messages to the backend verbatim: **no memory, no knowledge packs, no
dream context, no media context, no file-token expansion, no tools, no
inference-hint rewriting, no prompt-cache pinning.** The classifier never
selects Direct on its own — you must opt in.

```bash
curl https://YOUR_HOST/v1/chat/completions \
  -H "Authorization: Bearer sk-aug-…" \
  -H "Content-Type: application/json" \
  -H "X-Augmentum-Mode: direct" \
  -d '{"model": "your-model", "messages": [{"role":"user","content":"hi"}]}'
# equivalently: "model": "d/your-model"  (and drop the header)
```

This is the path to use when you want the box to behave as a **plain
OpenAI-compatible proxy** and nothing else.

### (B) Passthrough (+ memory and standard context)

Use **Passthrough mode** (or send nothing and let it default there). Passthrough
proxies to the model but the route layer adds the standard context bundle:

- **Memory recall + injection** (unless globally disabled — see below).
- **Dream context** (only if the user enabled it).
- **Knowledge-pack context** — only when the request carries a `session_id`
  (an in-app concept; bare external calls usually have none, so packs are a
  no-op for them).
- Media-now context, image-URL resolution, vision-caption fallback, file-token
  expansion, mode-aware inference defaults, prompt-cache key.

Tools are **off** in passthrough unless you opt in with
`X-Augmentum-Tools: <selector>`.

```bash
curl https://YOUR_HOST/v1/chat/completions \
  -H "Authorization: Bearer sk-aug-…" -H "Content-Type: application/json" \
  -H "X-Augmentum-Mode: passthrough" \
  -d '{"model": "p/your-model", "messages": [...]}'
```

### (C) Full — let Augmentum decide

Send a **bare model name** with no mode header. The classifier routes by
content: simple → Passthrough, research-y → Analytical, story signals →
Narrative, prior-session continuity, or a user-pinned `default_mode`. This is
the in-app experience.

## The memory axis (important nuance)

Memory injection is **bound to the mode**, not a per-request switch. There is
**no `X-Augmentum-Memory: off` header** today. Your levers:

- **Off for this request:** use **Direct mode** (turns off *everything*,
  including memory) or **Coder mode** (memory is explicitly skipped for the
  coding agent).
- **Steer the recall query without disabling it:** `X-Augmentum-Memory-Query:
  <text>` (Passthrough/Analytical/Narrative).
- **Off globally:** `memory_enabled: false` (server setting), or per-mode
  `memory_inject_analytical` / `memory_inject_agentic` (both default off).

So the clean recipes are:

| Goal | How |
|---|---|
| **No features at all** (raw proxy) | `X-Augmentum-Mode: direct` (or `d/<model>`) |
| **Proxy + memory + standard context** | `X-Augmentum-Mode: passthrough` (or default) |
| **Proxy + memory only, nothing else** | Passthrough, with dream opt-in off and no `session_id`. There is no single flag for "memory only" — memory rides the route-layer bundle. If you want a first-class `memory-only` toggle, it would need a new per-request header (e.g. `X-Augmentum-Memory: off` to subtract it from Passthrough). |
| **Full heuristic routing** | bare model name, no mode header |

## Other modalities

TTS / STT / image generation are the same OpenAI shape and route through the
configured provider (`audio_providers` / `image_providers`), gated by their
enable flags:

```bash
# TTS
curl https://YOUR_HOST/v1/audio/speech -H "Authorization: Bearer sk-aug-…" \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"hello","voice":"alloy"}' --output out.mp3

# STT
curl https://YOUR_HOST/v1/audio/transcriptions -H "Authorization: Bearer sk-aug-…" \
  -F file=@clip.wav -F model=whisper-1

# Image
curl https://YOUR_HOST/v1/images/generations -H "Authorization: Bearer sk-aug-…" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red bicycle","n":1,"size":"1024x1024"}'
```

These do **not** run the chat mode pipeline — there's no memory/narrative layer
on audio/image; they're direct provider proxies regardless of mode.
