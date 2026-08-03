# Connect — a self-hosted communications platform

Connect is described in the README in one paragraph — "WebRTC calls and encrypted
text threads." Every noun in that sentence is real, but the sentence badly
undersells what's there. Connect is a **complete, federated communications
platform**: ~11,500 lines of Python, ~30 REST + WebSocket endpoints, 22 JavaScript
UI modules, and a real cryptographic stack — voice/video calls, a modern
messaging system, decentralized identity, **cross-instance federation with no
central server**, device-authenticated guests, and an adaptive reachability
planner.

The one thing to take away: **two completely independent Augmentum instances,
owned by different people, can call and message each other end-to-end — through a
key-based trust ceremony, with no platform in the middle.** Same-instance calling
is table stakes; sovereign federation is the point.

> Setting it up? See the operator guide: [Connect Federation](connect-federation.md).

---

## Calls

A full WebRTC calling stack, not just "voice and video":

- **Signaling protocol** — invite → ringing → accept/decline → offer/answer → ICE
  → hangup, over a single persistent WebSocket
- **Call state machine** — ringing / invited / negotiating / connected / ended /
  missed, with missed-call timers and auto-expiry
- **Multi-device routing** — a `party_id` per connection (Matrix MSC2746-style),
  so a call rings all your devices and the first to answer wins
- **Mid-call renegotiation** — promote audio → video (or back) without dropping
- **LiveKit SFU** integration (JWT minting, reachability probe) + **your own TURN**
  credential server
- **Call history** with duration, modalities, and end reason; optional **1–5
  quality rating** with notes
- Per-user isolation (initiator vs receiver perspective), and a `becca_present`
  flag so the companion can join a call when you want it to

---

## Messaging

A modern messaging stack, not a 2012 chat box. Text threads are 1:1 and
server-persisted in SQLite, with:

- **Send / deliver / read receipts** (bulk)
- **Message editing** and **soft-delete with tombstones** (Signal/Matrix style)
- **Emoji reactions** and **typing indicators**
- **File / image attachments** (upload, serve, HEAD probe)
- **Thread management** — mute, pin, archive, clear
- **Thread-pair dedup** (one thread per user-pair) and **denormalized tails**
  (`last_message_at`, preview, unread count via DB trigger) for a fast inbox
- **Idempotent inserts** (a re-sent message is a no-op) and strict per-user
  isolation (no cross-user read path)
- A **fabric outbox** for durable cross-instance delivery with retry

---

## Identity & contacts

- **Decentralized identity** — a `user@instance` DID; no central account system
- **Contacts** — add / remove / block / tag, with discovery tracking (how you
  found them) and profile management (display name, avatar, handle)
- **Presence** — online/offline per user, fanned out across their devices
- **Directory search** — with per-user discoverability controls (you choose
  whether you're findable)

---

## Federation — the actual story

Federation is the differentiator, and it's where Connect stops resembling a
"calling feature." Two independent instances establish trust and then exchange
calls and messages directly:

- **Cross-instance fabric transport** with **Ed25519-signed envelopes**
- **Contact-card exchange** (link or QR) — no homeserver federation config to
  edit; you trade cards
- **SAS verification ceremony** — read four words / scan a QR to confirm you're
  talking to who you think (defeats machine-in-the-middle)
- **Admission posture** — `private` / `allowlist` / `knock` / `open`, with an
  intro-withheld **knock queue**
- **Federation gate** — deny-by-default for strangers, known-contact bypass
- **Identity-key backup** (24-word phrase) and **safety-number-changed detection**
- A durable **fabric outbox** and inbound dispatch with per-verb handlers
- A **documented, opt-in security model** (host-trusted vs end-to-end) — see below

Compared to a Matrix homeserver, deployment is simpler (contact cards instead of
server-to-server federation config). Compared to a Jitsi room, the trust is
stronger (per-identity keys and SAS, not room-level).

---

## Guests — device-authenticated, end-to-end, revocable

"Scoped guest grants" sounds like a permission checkbox. It's a small
cryptographic system for letting a temporary visitor in without giving them an
account:

- **Scoped grants** — text, call, and/or video, per invite
- **Durable guest-pass tokens** — SHA-256 hashed at rest, the raw token returned
  exactly once
- **Device-authenticated portal** — Ed25519 device-key binding; a guest's device
  proves itself
- **End-to-end envelopes** — ECDH + HKDF → AES-256-GCM, Ed25519-signed, with
  **per-device nonce replay windows** and an X25519 **instance seal key** (itself
  signed by the fabric identity)
- **Registration flow** — pending → admin confirm → IP allowlist, with a per-IP
  throttle
- **PWA-persisted, revocable sessions** — a guest keeps a working session across
  reloads; you can revoke it at any time

That's Signal-grade envelope crypto applied to *temporary visitors* — something
neither Matrix nor Jitsi offers.

---

## Reachability — "no ports to open," in detail

"Reachable over Tailscale" hides an adaptive **4-tier planner** that picks the
*least-exposing* path that actually reaches the recipient:

| Tier | Path | Exposure |
| --- | --- | --- |
| 1 | **LAN** — same network | Zero external exposure |
| 2 | **Tailnet** — private `ts.net` | Private mesh only |
| 3 | **Tailscale Funnel** — public, stable URL | App-managed public ingress |
| 4 | **Cloudflared quick-tunnel** — throwaway | Auto-teardown, no account |

Plus: **live tunnel ref-counting** (a tunnel is shared across invites and torn
down when the last invite expires), **path-scoped public ingress** (only the
invite door is exposed — never `/login` or the API), **capability detection**
(what tiers are actually available on this host), and plan-returned flags when a
tier would exceed the operator's stated preference. The planner is unit-testable
without the tunnel binaries (injected runners and clocks).

---

## Security model

Connect is honest about its trust boundaries, and the model is opt-in:

- **Text and guest envelopes** are end-to-end encrypted (ECDH + AES-256-GCM,
  Ed25519-signed).
- **Calls** are host-trusted through the SFU/TURN you run (you own the media
  path; there's no third-party server), with the option to keep media on your own
  infrastructure.
- **Federation** is deny-by-default: strangers can't reach you without a contact
  card and (optionally) a completed SAS ceremony.

The [federation operator guide](connect-federation.md) documents the host-trusted
vs end-to-end distinction in full, so you can decide what posture fits your
threat model.

---

## Operational maturity

Not visible in the UI, but it's why it holds up:

- **Per-user rate limiting** on signaling verbs (five categories)
- **In-memory presence registry** (`ConnectHub`) with per-user fan-out
- **Process-local invite timers** with clean restart semantics and orphan-
  connection detection
- **Protocol versioning** for forward compatibility; a 64 KB max signaling frame
- **Notification integration** — incoming call/message banners with
  accept/decline
- **Stale-write guards** on profile updates; guest IP throttling

---

## The client

The Connect UI is 22 JavaScript modules plus standalone guest/join PWAs — a
complete client application:

- **People** panel (contacts + directory + search)
- **Threads** panel (message list, compose, attachments, reactions, emoji picker,
  voice recorder)
- **Dialer** with incoming-call modal + ringtone, ringback state, mute/hangup, and
  a call-quality indicator
- **Guests** dashboard (mint invites with link/QR, manage and revoke grants)
- **Federation** panel (contact cards, verification ceremonies, peer management)
  and a **fabric outbox** monitor
- Standalone **connect-join** onboarding and **connect-guest** PWA pages, so a
  visitor can join without an account

---

## How it compares

| Capability | Matrix (homeserver) | Jitsi | Augmentum Connect |
| --- | --- | --- | --- |
| Deployment | Homeserver + federation config | Server + prosody/jicofo | Single install; contact cards, no federation config |
| Voice / video | Via Element/calls | ✅ Core | ✅ WebRTC + LiveKit SFU + your TURN |
| Text messaging | ✅ Rich | — | ✅ Reactions, edits, soft-delete, receipts, attachments |
| Identity | Homeserver account | Room-scoped | **DID (`user@instance`), key-based** |
| Federation trust | Server-to-server keys | Room-level | **Per-identity keys + SAS ceremony** |
| Guest access | Limited | Room link | **Device-authenticated, E2E, scoped, revocable** |
| Reachability | Manual reverse proxy | Manual | **Adaptive 4-tier planner (LAN → tunnel)** |
| Media ownership | Depends | Your server | **Your SFU + TURN, host-trusted path** |

Connect trades Matrix's mature ecosystem and Jitsi's polish for **sovereign
simplicity**: no central server, key-based identity, richer messaging than either,
a guest system neither has, and networking that adapts to your host instead of
asking you to open ports.

---

## Where this lives

For contributors: the core is in `augmentum/connect/` (signaling hub, call
lifecycle, messaging store, guest gateway, reachability/funnel manager, fabric
transport + inbound dispatch, federation gate) and `augmentum/calling/`, with the
REST + WebSocket surface in `augmentum/proxy/connect_routes.py`. The federation
ceremony and identity crypto live under `augmentum/fabric/`. The client is
`ui/scripts/connect/` plus the `ui/connect-join/` and `ui/connect-guest/` pages.
Operator setup: [Connect Federation](connect-federation.md).
