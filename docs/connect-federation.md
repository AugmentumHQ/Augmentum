# Connect Federation (the federated-PBX) — Operator Guide

Connect Federation lets two self-hosted Augmentum instances talk to each other —
messages and calls between *their* users — **without any central platform, account
provider, or directory.** You share a signed contact card (a link or QR), verify each
other once, and you're connected. It's the Skype-easy face on Augmentum's existing
server-to-server fabric, built sovereign-first.

This guide is for the person **running** an instance. It is deliberately blunt about what
is and isn't protected — read the Security Model before enabling it for anyone who depends
on it.

---

## What it is (and isn't)

- **It is:** sovereign, federated, key-based identity; first contact by signed contact
  card; an out-of-band verification ceremony (read 4 words on a call, or scan a QR); a
  deny-by-default inbound posture so strangers can't spam you; optional end-to-end
  encryption for direct messages.
- **It is not:** anonymous, metadata-free, or (by default) zero-knowledge to the hosts.
  See the Security Model.

---

## Enabling it

Federation is **OFF by default.** Turn it on only when you intend to federate.

1. Enable the fabric and federation flags (Settings, or `app_settings`):
   - `fabric_enabled = true` — the server-to-server transport.
   - `fabric_federation_enabled = true` — the contact-card / ceremony / knock surface.
2. Choose your stranger **admission posture** (`fabric_admission_posture`):
   | posture | inbound from someone you haven't pinned |
   |---|---|
   | `private` | refused entirely |
   | `allowlist` | only keys you pre-approved |
   | `knock` *(default)* | queued as a **non-ringing, intro-withheld** request you can accept or reject |
   | `open` | auto-surfaced (use only for a public tip line / intake) |
3. Leave `fabric_relay_sealed_only = true` (default). This refuses to forward any
   cross-instance payload that isn't sealed — a relay never sees your cleartext.
4. (Optional) `fabric_e2e_dm_enabled = true` for end-to-end device-to-device DMs so even
   the hosts can't read message content. The default host-trusted path is left on for
   compatibility; E2E is opt-in.

**Back up your identity key immediately after enabling.** It IS your federated identity
and cannot be recovered if lost:
`GET /api/fabric/identity/backup` returns a 24-word phrase — write it down offline. Restore
with `POST /api/fabric/identity/restore`. The loader is fail-closed: if the stored key is
ever corrupt, the server refuses to silently mint a new one (which would break everyone who
pinned you) and tells you to restore from this backup.

---

## First contact, in plain terms

1. You generate a **contact card** (`POST /api/fabric/contact-card`) and send the link/QR
   to someone — over any channel.
2. They accept it (`POST /api/fabric/contact-card/accept`). Their instance pins your
   identity key, marked **"pinned, not verified."**
3. You verify each other **out of band**: on a call, both read the 4 SAS words
   (`POST /api/fabric/peers/ceremony`) — they must match; or in person, scan the QR. Then
   each marks the other verified (`POST /api/fabric/peers/verify`).
4. From then on the UI shows a **verified** badge. Before that it shows **"Unverified
   caller — hosted chat readable by both servers."** Do not skip the ceremony for anyone
   whose authenticity matters to you.

If a contact's key ever changes for a handle you already knew, you'll see a **safety-number
-changed** warning and must re-verify. That's either a legitimate key rotation or an
impersonation attempt — treat it seriously.

---

## Security model — read this

**By default, federation is *host-trusted*, not end-to-end.** Both your server and your
contact's server can read the content of messages and calls that pass between them. The
guarantee is **sovereignty, not secrecy**: your data lives only on the two instances
involved — there is no central provider aggregating it. That is a real and deliberate
difference from Skype/WhatsApp/etc., but it is **not** the same as "private."

What each layer protects:

| Concern | Protection |
|---|---|
| Is this really who they say? | did:key identity, byte-compared; the **ceremony** is what upgrades a pin to trusted. A fresh card is TOFU — a malicious host could mint it, so verify. |
| Can a stranger spam/ring me? | No — deny-by-default `knock` posture; strangers are queued, non-ringing, intro-withheld, rate-limited. |
| Can a relay read forwarded content? | No — `sealed_only` (sign-then-seal: X25519+ChaCha20Poly1305 with the origin signature *inside* the seal). |
| Can someone forge who a message is from? | No — caller-ID is the envelope-verified signer; a forged body claim is rejected. |
| Can the hosts read my DMs? | **Only if you leave E2E off.** Enable `fabric_e2e_dm_enabled` for device-to-device sealing. |
| Metadata (who talks to whom, when)? | **Not hidden.** A relay sees timing; full metadata privacy (mixnet) is out of scope. |

**Honest residuals** (also in the spec): a contact card from your *own* malicious host is
TOFU until the ceremony defeats it (a Signal-class limit); losing your identity key with no
backup is unrecoverable; under key *compromise* a self-signed revocation can name a
successor, so never auto-trust a revocation's successor — re-run the ceremony.

---

## For developers extending the live path (the one rule that matters)

Every inbound federated frame MUST go through the admission choke-point before you act on
it — and you MUST feed it the **envelope-verified** signer key, never the message body's
claimed source:

```python
from augmentum.fabric.admission import authenticate_and_admit, ADMIT, KNOCK

decision = await authenticate_and_admit(
    conn,
    verified_pubkey=scope["fabric_peer"]["verified_pubkey"],  # from the middleware, NOT the body
    claimed_source_did=frame.get("source_did"),               # body claim — checked, not trusted
    to_user_id=recipient_user_id,
    recipient_posture=settings.fabric_admission_posture,
    seq=frame.get("seq"),                                     # enables the durable replay guard
)
if decision.action == ADMIT:
    deliver(decision.source_did, ...)   # source_did is now authenticated
elif decision.action == KNOCK:
    await submit_knock(conn, to_user_id=recipient_user_id, from_did_key=decision.source_did, ...)
# else: denied / forged / replay → drop
```

This single function runs caller-ID authentication, denylist/revocation, the durable replay
guard, pin lookup, and posture — in the right order. Sourcing any auth input from the
untrusted body instead re-opens forgery (SEC-11). PoW-gated intake must use the **signed**
challenge helpers (`pow.sign_challenge` / `verify_signed_solution`), never a client-supplied
challenge.

### End-to-end encryption & the sovereign-AI participant

E2E moves the encryption boundary from instance↔instance to device↔device, so the servers
become blind relays. The conversation model is simple: a message is sealed to a **list of
recipient devices** (`fabric/e2e_session.py`). For a direct 1:1 chat that list is just the
other person's device(s) — nothing more.

The same list is what would let a user's **own sovereign AI** join a conversation as an
extra endpoint (it runs on your hardware, so including it hands nothing to a third party).
That capability is fully wired but deliberately **on standby behind a hard code gate**,
`COMPANION_E2E_SECURITY_CONFIRMED` (currently `False`). While that constant is closed, the
companion can NEVER be sealed into a conversation — *even if* a user turns on the
`companion_e2e_participant_enabled` setting. The setting is only a request; the gate is the
real switch, and lifting it is a deliberate, reviewed act, not a toggle. This is intentional:
a single setting flip must never be able to silently add a third party (even your own AI) to
an end-to-end chat before its threat model is signed off.

True host-blindness additionally requires the sealing to run **client-side** (in the
browser/app) — the recipient/policy model and wire shape are built; the client crypto port is
the remaining work.

### What's wired into the live path today

The existing Connect-over-fabric transport addresses peers as `user@hostname` and verifies
the *sending instance* by its pinned Ed25519 key. The federation trust layer is bridged onto
it by `augmentum/connect/federation_gate.py`, called from the inbound dispatcher
(`fabric_inbound.py`) on every frame:

- **Default-off.** With `fabric_federation_enabled = false` (the default) the gate allows
  everything — existing installs are byte-for-byte unchanged (36 existing roundtrip tests
  confirm this).
- **When enabled:** the sending instance's did:key is derived from its pinned key and checked
  against the denylist/revocation; relationship-creating frames (first message / call invite)
  from a sender the recipient has no contact with are gated by `fabric_admission_posture`
  (`open` delivers, `private`/`allowlist` drop, `knock` queues an intro-withheld request).
  Known `connect_contacts` flow through.
- **Fail-open on a gate bug:** a gate exception preserves delivery (logged loudly) so a bug
  can never silently break the working transport.

The did:key contact-card / ceremony / per-user knock surface (the `/api/fabric/...` routes
and the "Connect: Federation" UI) is the richer parallel model; unifying the `user@hostname`
wire address with per-user did:key identity is the remaining protocol step.

---

## Reference

- Design + decisions: `docs/superpowers/specs/2026-06-23-connect-federated-pbx-design.md`
  (+ `-decisions`, `-redteam`, `-redteam-2`).
- Build record: `…-build-record.md`. Security review: `…-security-review.md`.
- Code: `augmentum/fabric/` (identity, didkey, contact_card, ceremony, peer_identity_store,
  caller_id, knock, author_binding, relay_seal, revocation, directory, pow, receipts, e2e,
  at_rest, admission, durable_guards).
