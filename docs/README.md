# Augmentum documentation

Guides for installing, using, and extending Augmentum. Start with the
[project README](../README.md) for the overview and install; the pages below go
deeper on each area.

> **Using an AI coding agent?** Point it at **[`AGENTS.md`](../AGENTS.md)** (or
> [`CLAUDE.md`](../CLAUDE.md)) — it has the install/run flow, a codebase map, and
> navigation pointers so an agent can set Augmentum up and answer questions
> about it.

## Getting started

- **[Install & setup](../README.md#quick-start)** — installer script, manual
  pull, or cloned-repo (`setup.sh`).
- **[Configuration](../README.md#configuration)** — `AUGMENTUM_*` environment
  variables and tuning.
- **[Model Manager](model-manager.md)** — pick and manage models; power-user
  settings (expert offload, KV-cache quantization, keeping models warm,
  downloading GGUF quants in-app).

## Using Augmentum

- **[Modes](modes.md)** — how the classifier routes each request (passthrough ·
  analytical/UARF · narrative · agentic · coder) and how to force one with a
  `p/` `a/` `n/` `g/` `c/` prefix.
- **[Coder mode](coder.md)** — the containerized IDE agent: workspaces, tools,
  the browser, live preview, permissions, sub-agents, and driving it with Claude
  Code / Cursor.
- **[Companion](companion.md)** — enabling and living with the autonomous
  companion (off by default): what it does, how to configure it, and its safety
  posture.
- **[Voice & wake word](voice.md)** — voice input/output, training your own wake
  word, and tuning the pipeline.
- **[Narrative & roleplay](narrative.md)** — character cards, lorebook, world
  state, and consistent long-form storytelling.
- **[Image generation](image.md)** — self-hosted text-to-image, img2img,
  inpaint, upscale; local pipeline, cloud providers, or Fabric peers.
- **[Discover & installing services](discover.md)** — one-click install of
  ~50 companion services (Jellyfin, Suwayomi, vLLM-swap, …) with automatic HTTPS
  and mDNS, using Augmentum as the gateway.
- **[Powers](powers.md)** — using and creating your own capability packs that
  shape coder mode (including the "turn this into a power" shortcut).
- **[Knowledge packs](../README.md#knowledge-packs)** — offline reference corpora
  (Wikipedia/ZIM) for grounded, no-internet answers.

## Connect & your network

- **[Fabric](fabric.md)** — share capabilities (borrow a GPU, TTS, STT, …) across
  your own machines; pairing and trust.
- **[Connect federation](connect-federation.md)** — calls and threads between
  your own instances and other people's.
- **[External MCP servers](connect-external-mcp-servers.md)** — bring outside MCP
  tools into chat and the coder.
- **[Connect to Claude via MCP](connect-to-claude-mcp.md)** — use Augmentum *as*
  a tool from Claude Desktop / Cursor.
- **[Cast to Android TV](cast-android-tv.md)** — the living-room cast receiver.
- **[Web Push setup](web-push-setup.md)** — notifications with the tab closed.

## Reference

- **[External API](external-api.md)** — the Ollama / OpenAI / Anthropic-compatible
  surfaces and the `/v1/tools` (ATP) open tools.
- **[Architecture](ARCHITECTURE.md)** · **[Subsystems](subsystems.md)** —
  how it's built.
- **[Model settings map](model-setting-map.md)** ·
  **[Provider matrix](provider-integration-matrix.md)** — where each setting lands.

## Security & privacy

- **[Security model](security_model.md)** — threat model and isolation.
- **[Privacy](PRIVACY.md)** — what stays local, what leaves only when you say so.
- **[macOS hardening](MAC_HARDENING.md)** — host-side hardening notes.
