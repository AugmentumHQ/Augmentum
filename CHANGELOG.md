# Changelog

All notable changes to Augmentum are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Augmentum is pre-1.0, so until a stable line exists, breaking changes can land
in any release; they will always be called out under **Changed** or **Removed**.

## [Unreleased]

First public release. The heading is renamed to `[0.1.0] — YYYY-MM-DD` and a
fresh `[Unreleased]` is opened above it when the `v0.1.0` tag is cut.

### Added

- **Five processing modes**, auto-selected by a built-in classifier or via
  `a/`, `n/`, `p/`, `g/`, `c/` model-name prefixes:
  - **Passthrough** — forwards requests untouched (the default), with an
    optional SSOS auto-tools layer (calculator, datetime, unit conversion,
    web search/fetch, app builder).
  - **Analytical (UARF)** — a 6-phase reasoning pipeline (ASSESS → IDENTIFY →
    RELEVANT → APPLY → VERIFY → CONCLUDE, plus DECOMPOSE for hard problems)
    with 3-tier tool calling (native → structured → text).
  - **Narrative** — character cards, group chats, lorebook entries, macro
    expansion, regex scripts, prompt presets, and a three-layer memory model
    (STATE snapshot + LEDGER + embedded ARCHIVE) with branching conversation
    trees.
  - **Agentic** — plan-as-attention-anchor execution, a 4-level autonomy dial,
    and artifact tools (docx / pptx / xlsx / charts).
  - **Coder** — containerised per-workspace coding agent with project digest,
    workspace snapshots, a semantic indexer, and structured plan verifiers.
- **Drop-in API compatibility** with both Ollama (`/api/generate`, `/api/chat`,
  `/api/tags`, `/api/show`, `/api/ps`) and OpenAI (`/v1/chat/completions`,
  `/v1/models`) clients, with full streaming support.
- **Bundled inference engine** — a managed `llama-server` subprocess that
  auto-discovers GGUF models, with a family-aware reasoning parser covering
  multiple thinking-tag wire formats (DeepSeek, Qwen 3.x, GLM-4.x, EXAONE 4.x,
  Mistral Magistral, Gemma 3/4, GPT-OSS, Nemotron). External Ollama /
  LM Studio / OpenAI-compatible / Anthropic backends are configurable too.
- **Multi-tenant auth & data isolation** — Argon2id passwords, opaque session
  tokens, rate-limited login, fail-closed ASGI middleware; every user-scoped
  table carries a `user_id`.
- **Persistent state in SQLite** — sessions, character cards, narrative state,
  facts/assumptions/entities, memories, and settings, with a migrations system.
- **Memory** — sqlite-vec embeddings, LLM-driven extraction, a core persona
  profile, consolidation/compaction, and per-mode injection gates. Memory is
  shared across chat / voice / narrative / dream / coder.
- **Voice** — server-side VAD (Silero), streaming STT, sentence-buffered TTS,
  speaker verification, graceful barge-in, and voice cloning via evolutionary
  search over Kokoro embeddings (no fine-tuning).
- **3D avatars** — VRM characters in voice calls with procedural animation,
  lip sync, IK posing, a hand-pose affordance vocabulary, and a per-VRM body
  atlas (SDF + region + touchability voxel grid).
- **Knowledge packs** — offline reference corpora (`.augpack` SQLite + sqlite-vec
  + FTS5, and `.zim` Kiwix archives), hybrid retrieval with Reciprocal Rank
  Fusion + optional cross-encoder rerank, and a browseable sandboxed-iframe
  reader for ZIM articles.
- **Browse** — a web reader with AI tools, Milkdown notes, an extraction
  pipeline (JSON-LD / AMP / RSS / curl_cffi / Wayback), and domain reputation.
- **Image generation** — local (SD / GGUF, DreamShaper 8 baked into the GPU
  variant) and cloud (OpenAI, Together, Stability, BFL, Fal).
- **Media library** — local media plus media-server auto-detection, with the
  LibriVox public-domain audiobook library wired in as a built-in source.
- **Artifact Studio** — edit PDF / DOCX / PPTX / XLSX / charts in-app with a
  theme picker, live preview, and a present mode.
- **Dream system** — periodic AI introspection that writes back into the
  persona (journal entries + an evolved portrait).
- **Background job queue** — a restart-survivable job primitive (first
  consumers: GGUF downloads, LibriVox fetches, ZIM→augpack conversion).
- **Sandboxed Python executor** — a separate container with a read-only
  filesystem, dropped capabilities, a 64-PID limit, a 512 MB memory cap, and
  its own network.
- **Web UI** — vanilla-JS, multi-surface, mobile orb navigation, 4 themes ×
  8 typography presets, served alongside the API on the same port.
- **Docker Compose deployment** — pre-built CPU and GPU images on GHCR, an
  interactive `setup.sh` / `setup.bat` wizard, overlay compose files for
  optional services (GPU/image gen, Kokoro/Qwen/Chatterbox TTS, Speaches STT,
  game streaming), and a `compose.dev.yaml` for live-edit local builds.
- **Security posture** — localhost-only by default, no telemetry, a published
  threat model (`SECURITY.md`), and a multi-tenant isolation audit script in
  the dev skill.

### Known limitations

See [`SECURITY.md`](SECURITY.md) for the current list of documented gaps
(CSRF coverage, SSRF allowlisting, API-key-at-rest encryption, workspace-
container hardening, prompt-injection → tool-abuse, dependency pinning).
