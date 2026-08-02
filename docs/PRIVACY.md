# Privacy

This document is the per-feature data-flow map for Augmentum. The short
version: **on a default install nothing leaves your machine.** This page
exists for the longer version — what *can* leave, when, and what to do
about it.

For the security threat model see [`SECURITY.md`](../SECURITY.md). For the
project-level privacy posture (no telemetry, no analytics, no phone-home)
see the "Privacy posture" section there.

---

## At a glance

| Subsystem | Outbound by default? | Outbound if reconfigured? | What leaves |
|---|---|---|---|
| LLM chat | No (bundled engine) | Yes — chosen provider | Messages, system prompt, attached docs |
| Speech-to-text | No (local Speaches) | Yes — Deepgram, etc. | Audio waveform |
| Text-to-speech | No (Kokoro / Pocket TTS / Qwen-TTS / Chatterbox) | Yes — cloud TTS | Text to speak |
| Image generation | No (local SD / DreamShaper) | Yes — OpenAI / Stability / Together / BFL / Fal | Image prompt |
| Web search | Yes — your SearXNG instance | Yes — different SearXNG | Search query |
| Web fetch (`browse`, tools) | Yes — only to the URL you / the LLM target | — | HTTP `GET` to that URL |
| Knowledge pack catalog | Yes — `library.kiwix.org` when you browse the catalog | — | Catalog browse + chosen pack download |
| Model downloads (GGUF, etc.) | Yes — HuggingFace / GitHub release URLs when you pull | — | Model file requests |
| Provider images (Kokoro, Chatterbox, etc.) | Yes — GHCR / Docker Hub when you enable that overlay | — | Image pulls |
| Coder workspace egress | Yes — open bridge network for `pip install` etc. | — | Whatever the workspace runs |
| Media servers | Yes — only servers you configure (Plex / Jellyfin / Emby) | — | Library queries to that server |
| Telemetry / analytics / update checks | **Never** | **Never** | — |
| Error reporting | **Never** | **Never** | — |

"Outbound by default" = happens on a fresh install with no configuration,
when you actually use the feature. The bundled engine, local TTS, and
local image gen mean a fresh install can run an entire end-to-end
conversation with image and voice without sending anything to a third
party.

---

## Per subsystem

### LLM chat
- **Default:** the bundled `llama-server` engine inside the Augmentum
  container — your messages never leave the box.
- **If you switch backends** (Settings → Manage Providers, or
  `AUGMENTUM_DEFAULT_BACKEND` / `AUGMENTUM_OLLAMA_BASE_URL` /
  `AUGMENTUM_OPENAI_*`): the entire chat completion request goes to that
  provider — messages, system prompt, attached documents, knowledge-pack
  excerpts that were retrieved for grounding.
- **Disable:** keep `AUGMENTUM_DEFAULT_BACKEND=engine`. Or use external
  providers selectively: per-character-card overrides and per-mode
  backend selection let you keep the default local and only call out for
  specific tasks.

### Speech-to-text
- **Default:** the `speaches` container (local faster-whisper) when the
  `compose.speaches.yaml` overlay is enabled. Audio is streamed over the
  internal Docker network and never reaches the public internet.
- **If you switch:** any provider you configure (e.g. Deepgram) receives
  the raw audio.
- **Disable:** use the local Speaches overlay, or disable voice features
  entirely in Settings → Voice.

### Text-to-speech
- **Default:** Kokoro, Pocket TTS, Qwen-TTS, or Chatterbox — all local
  containers. Text goes over the Docker network only.
- **If you switch:** any cloud TTS you configure receives the text to
  speak.
- **Disable:** stick with the local engines, or disable assistant voice
  output in Settings → Voice.

### Image generation
- **Default:** local Stable Diffusion / GGUF (the GPU variant pre-bakes
  DreamShaper 8). Prompts and seeds stay on-box.
- **If you switch:** cloud providers (OpenAI, Together, Stability, BFL,
  Fal) receive the prompt and any reference images.
- **Disable:** keep the local provider as default; cloud providers must
  be explicitly added with their API keys.

### Web search
- **Default:** the bundled SearXNG container queries upstream search
  engines on your behalf. SearXNG itself reaches out to Google, Bing,
  Brave, etc. — your IP isn't shielded from those engines unless you
  proxy SearXNG.
- **What leaves:** your search query. Augmentum doesn't add user IDs,
  cookies, or other identifying metadata.
- **Disable:** turn off `web_search` in Settings → Tools, or point
  Augmentum at your own SearXNG instance via `AUGMENTUM_SEARXNG_URL`.

### Web fetch (`browse`, tools, knowledge pulls)
- **What happens:** when you (or the LLM, via `web_fetch`) request a URL,
  Augmentum does an HTTP `GET` to that URL through `SafeHttpClient`. The
  destination server sees a request from your IP.
- **Protection:** `SafeHttpClient` blocks loopback / RFC1918 / link-local
  / multicast destinations, so an injected URL can't probe your internal
  network.
- **Disable:** turn off `web_fetch` / `browse` in Settings → Tools.

### Knowledge pack catalog
- **What leaves:** when you open the Knowledge Pack catalog in Settings,
  Augmentum fetches the index from `library.kiwix.org`. When you install
  a pack, that pack file is downloaded from the same source. After
  install, knowledge packs are 100% local — searches never call out.
- **Disable:** don't open the catalog. Manually drop `.zim` or
  `.augpack` files into `/data/knowledge_packs/` instead.

### Model and provider downloads
- **What leaves:** HuggingFace / GitHub release URLs when you pull a GGUF
  model; Docker Hub / GHCR when you enable a provider overlay (Kokoro,
  Chatterbox, etc.).
- **No background fetches:** Augmentum never auto-pulls models or images.
  You initiate every download.

### Coder workspace
- **What leaves:** whatever the agent or your code runs inside the
  workspace container. `pip install` reaches PyPI; `git push` reaches
  the remote you configured; a prompt-injected `curl` reaches the
  attacker's host. See the project's internal hardening roadmap
  for the planned egress filter.
- **Disable:** disable Coder mode in Settings → Modes.

### Media servers
- **What leaves:** library and stream requests to the Plex / Jellyfin /
  Emby instance you configured. Only servers you explicitly added are
  contacted; Augmentum does not scan the internet for media servers.
- **LAN discovery (mDNS / UPnP):** runs only on your local segment and
  doesn't expose data beyond the LAN.

### Telemetry, analytics, update checks, error reporting
- **None of these exist.** Augmentum makes no calls home, has no
  analytics SDK linked, and never contacts an Augmentum-controlled
  server. Releases are user-initiated (`docker compose pull`).

---

## What stays on-box (always)

The following data is *only* stored in your `/data` volume. None of it
is sent anywhere, on any configuration:

- Account data: usernames, password hashes (Argon2id), API keys
  (Fernet-encrypted at rest), session tokens
- All conversation history, branched message trees, draft messages
- Character cards, lorebook entries, narrative state, plot threads
- Memory vectors, fact registries, contradiction logs, dream journal
- Uploaded documents, file index, knowledge-pack contents
- Generated images, generated audio, VRM avatars, voice clones, body
  atlases
- Notebook content, browse notes, project workspaces
- Audit logs, settings, schema versions

If you make a backup (`/data/backups/*.db`), that backup contains
everything above. Treat it like you'd treat a snapshot of your password
manager.

---

## Things you should know

- **Knowledge pack retrieval uses your message as the query.** When a
  pack is bound to a chat, Augmentum embeds your message locally
  (sentence-transformers) and searches the pack — your message text is
  never sent to a remote ranker. But if you've configured a cloud LLM
  *and* knowledge packs are active, the retrieved pack excerpts get
  injected into the prompt sent to that cloud LLM.
- **Character cards may contain prompts you didn't write.** A
  downloaded card can include a system prompt that asks the model to
  call tools. The tool then runs against your local services (web
  search, web fetch). Audit cards before binding them.
- **Local engine model files** are downloaded from HuggingFace; the
  download itself sends your IP to HuggingFace. Subsequent inference is
  fully offline.
- **Dream cycles run against the configured chat backend.** If your
  default backend is a cloud LLM, dream introspection sends recent
  conversation context to that cloud LLM. Switch to the local engine
  for dream cycles in Settings → Dream if you want it to stay offline.

---

## How to verify

- **Network egress:** run `docker compose exec augmentum sh -c 'ss -tnp'`
  to see active outbound connections. With no provider configured and
  no feature actively using the network, you should see nothing beyond
  internal container links.
- **Disk:** `du -sh /var/lib/docker/volumes/augmentum_*` shows what
  Augmentum is storing locally. `sqlite3 /data/augmentum.db ".tables"`
  enumerates every table.
- **Settings:** Settings → Privacy in the web UI lists every outbound
  surface and its current on/off state, with one-click toggles for the
  ones that aren't load-bearing.
