# Fabric — share capabilities across your machines

Fabric lets one Augmentum instance borrow another's hardware. A tablet with no GPU
can run image generation on your tower; a laptop can transcribe on the desktop in
the next room. Each node advertises what it can serve and asks a peer for what it
can't — over one signed, encrypted channel.

> **Off by default.** A solo install never runs a line of Fabric. Turn it on with
> `fabric_enabled` on every node you want to federate.

## What can be shared — seven capability kinds

A peer can serve any of:

1. **LLM inference** — run a chat/completion on the peer's model.
2. **Image generation** — offload diffusion to a GPU peer.
3. **TTS** — text-to-speech.
4. **STT** — speech-to-text.
5. **Knowledge search** — query a peer's knowledge packs.
6. **Code execution** — run code in the peer's executor.
7. **Cast rendering** — render a cast surface remotely.

Each node only offers what it actually has, and routing is **cost-aware** — it
prefers the cheapest capable node.

## Pairing two nodes

Trust is established the way SSH host keys are — deliberately, out of band. There
are two ways to do it, both ending in an explicit **approve** on the receiving
node.

### A) Connect code (easiest)

1. On node A, generate a **connect code** in the Fabric UI — a short, shareable
   string like `K7P2-9QX4` (also shown as a QR).
2. On node B, enter or scan that code. B resolves it to A's contact card and
   sends a pair request.
3. Approve the request on A. The nodes are now paired.

Codes are short-lived and expire, so a leaked code isn't a standing door.

### B) Fingerprint paste (most explicit)

1. Each node shows its own **fingerprint** in the Fabric UI.
2. On the initiating node, paste the *other* node's fingerprint into the pair
   form and send.
3. Approve on the receiving node.

The fingerprint is the out-of-band secret: the pair request itself is
**Ed25519-signed**, carries a replay-blocking timestamp, and must address the
right node's fingerprint — so only a request from the holder of the matching
private key, aimed at the right box, is accepted. There's no admin cookie in this
path (the caller is another node, not a browser), which is why the signature *is*
the authentication.

### Extra assurance — the verification ceremony

For higher-trust links you can run a **verification ceremony**
(`/peers/ceremony` → `/peers/verify`) that confirms both sides see the same keys —
the equivalent of comparing safety numbers. Verified peers show as such in the
list.

## Living with it

- See paired peers and their advertised capabilities in the Fabric UI
  (`/api/fabric/peers`, verified ones at `/peers/verified`).
- **Revoke** a peer at any time (unpair) — trust is not permanent.
- Reachable across your own network and, over Tailscale, from anywhere — no ports
  to open.
- Every federated request rides a signed envelope and is end-to-end encrypted;
  a peer cannot mint API keys, change your settings, or act as you — it can only
  serve the capabilities it advertised. See [Security model](security_model.md).

## Fabric vs Connect

Fabric shares **capabilities** between *your* machines (borrow a GPU). If you want
to share **communication** with people — voice/video calls and encrypted text
threads — that's [Connect](connect-federation.md). They're separate subsystems
that both reach past a single box.

## Where this lives (for contributors)

`augmentum/fabric/` (capabilities, signed envelopes, connect codes, peer
identities) and `augmentum/proxy/fabric_routes.py`. Peer identities and connect
codes are user-scoped (`fabric_peer_identities`, `fabric_connect_codes`).
