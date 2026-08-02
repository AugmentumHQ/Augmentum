"""Connect peer-DID resolver — same-instance + fabric-peer parsing.

DID format: ``<local-part>@<instance>`` where ``<instance>`` is the
**instance handle** of the owning Augmentum (a real, externally
addressable name — see :func:`instance_handle`), and ``<local-part>``
is the recipient's ``user_id`` (the routing key the ConnectHub uses).

Two instance forms resolve as *local* (this Augmentum):

* the configured :func:`instance_handle` (derived from
  ``connect_instance_handle`` / ``AUGMENTUM_PUBLIC_HOST``), and
* the legacy ``this-instance`` sentinel — back-compat so DIDs minted
  before instance identity existed (stored in old ``connect_contacts``
  rows) keep resolving.

Any other instance part routes to a paired fabric peer. Forward-
compatible with ``did:augmentum:<keyfp>`` form once the minimum-viable
DID layer lands.

Note: the local-part is the ``user_id``, NOT the username — routing
keys on it. Human ``@handle`` discovery is exposed at the directory /
search / profile layer, not in the wire DID (deferred; a username→
user_id resolver on the routing hot path is the upgrade that unlocks
human wire-DIDs).

The resolver returns either:

* ``("local", user_id)``  — same-instance peer, route via the local
  ConnectHub directly.
* ``("fabric", host)``    — fabric peer, dispatch through the
  fabric session pipe (deferred — not yet wired here).
* ``None``                — unrecognised / malformed DID.

Same-instance routing in Phase 1 is the simplest case: the DID
encodes the recipient's user_id literally. We don't validate that
the user_id exists in the auth store from this module — that's the
caller's responsibility (a missing user is detected when the
ConnectHub finds no attached WS sessions for that user_id).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


THIS_INSTANCE_SENTINEL = "this-instance"
log = logging.getLogger(__name__)


def _sanitize_handle(raw: str) -> str:
    """Reduce a raw host string to a DNS-safe instance handle.

    Strips scheme (``https://``), any path/query/fragment, and the port,
    lowercases, and keeps only ``[a-z0-9.-]``. Returns ``""`` when nothing
    usable remains (the caller falls back to the sentinel). Examples:
    ``https://myhost.ts.net:6443/`` → ``myhost.ts.net``;
    ``192.168.1.10:6443`` → ``192.168.1.10``.
    """
    h = (raw or "").strip().lower()
    if not h:
        return ""
    if "://" in h:
        h = h.split("://", 1)[1]
    for sep in ("/", "?", "#"):
        if sep in h:
            h = h.split(sep, 1)[0]
    if ":" in h:  # strip :port
        h = h.split(":", 1)[0]
    h = "".join(c for c in h if c.isalnum() or c in ".-")
    return h.strip(".-")


def instance_handle() -> str:
    """Return this Augmentum's public instance handle for DID addressing.

    Resolution order: the operator-set ``connect_instance_handle`` setting
    → the ``AUGMENTUM_PUBLIC_HOST`` (sanitized) → the legacy
    ``this-instance`` sentinel. Read live (cheap) so a runtime settings
    change via /api/config/tools takes effect without a restart. Safe in
    sync hot paths (no DB, no I/O).
    """
    raw = ""
    try:
        from augmentum.config import settings as _settings

        raw = (_settings.connect_instance_handle or "").strip() or (
            _settings.augmentum_public_host or ""
        ).strip()
    except Exception:  # pragma: no cover - config import must never break routing
        log.warning("Failed to read connect instance handle; using sentinel", exc_info=True)
        raw = ""
    return _sanitize_handle(raw) or THIS_INSTANCE_SENTINEL


def is_local_instance(instance_part: str) -> bool:
    """True when ``instance_part`` addresses THIS Augmentum.

    Matches both the configured :func:`instance_handle` (case-insensitive)
    and the legacy ``this-instance`` sentinel (back-compat for DIDs minted
    before instance identity existed).
    """
    inst = (instance_part or "").strip()
    if not inst:
        return False
    return inst == THIS_INSTANCE_SENTINEL or inst.lower() == instance_handle()


@dataclass(frozen=True)
class ResolvedPeer:
    """Routing target for a parsed peer DID.

    ``kind`` is the route to take: ``"local"`` for same-instance
    routing, ``"fabric"`` for cross-instance dispatch through a
    paired fabric peer.

    For ``kind == "local"``, ``address`` is the recipient user_id;
    for ``kind == "fabric"``, it's the peer host.
    """

    kind: str
    address: str


def resolve_peer_did(peer_did: str) -> ResolvedPeer | None:
    """Parse a Phase 1 ``<user>@<instance>`` DID into a routing target.

    Returns ``None`` if the DID is malformed (missing ``@``,
    empty user part, empty instance part). Forward-compat note:
    when ``did:augmentum:`` form lands, add a branch above the
    @-split.
    """

    if not peer_did or not isinstance(peer_did, str):
        return None

    # Forward-compat: did:* form lands here later.
    if peer_did.startswith("did:"):
        return None

    if "@" not in peer_did:
        return None

    user_part, _, instance_part = peer_did.rpartition("@")
    user_part = user_part.strip()
    instance_part = instance_part.strip()
    if not user_part or not instance_part:
        return None

    if is_local_instance(instance_part):
        return ResolvedPeer(kind="local", address=user_part)
    return ResolvedPeer(kind="fabric", address=instance_part)


def local_did_for(user_id: str) -> str:
    """Compose the local DID surface form for a user_id.

    Uses the live :func:`instance_handle` so freshly minted DIDs carry the
    instance's real public name. Pre-configuration (handle unset) this is
    identical to the historical ``user_id@this-instance`` form, so nothing
    changes until the operator names the instance.
    """

    return f"{user_id}@{instance_handle()}"


async def display_name_for_did(conn: object, did: str) -> str:
    """Best-effort human name for a Connect DID.

    For same-instance peers, looks up ``users.display_name`` (falling
    back to ``users.username`` when display_name is empty). For fabric
    peers or unknown users, returns the DID's local-part stripped of
    the instance suffix. Never returns an empty string for a non-empty
    DID — used directly in notification titles so the user sees
    "Call from bench" instead of "Call from usr_a8377…@this-instance".

    Defensive: any DB failure falls back to the local-part. Notification
    publish must never fail because of a name lookup.
    """
    raw = (did or "").strip()
    if not raw:
        return ""
    user_part, _, instance = raw.rpartition("@")
    user_part = user_part.strip()
    if not user_part:
        return raw
    # Same-instance: look up the canonical display name.
    if is_local_instance(instance):
        try:
            cur = await conn.execute(
                "SELECT COALESCE(NULLIF(display_name, ''), username) "
                "FROM users WHERE id = ?",
                (user_part,),
            )
            row = await cur.fetchone()
            await cur.close()
            if row and row[0]:
                return str(row[0])
        except Exception:
            log.warning("Failed to resolve local Connect DID display name; falling back", exc_info=True)
    return user_part
