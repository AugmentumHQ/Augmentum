"""Published device bundles for E2E key distribution (P2).

A user publishes the PUBLIC half of their device keys so others can seal
to them: the master author key + one entry per device (signing-subkey
did:key, X25519 sealing pubkey, and the master-signed binding). The
server's job here is integrity, not trust: it validates that every
binding actually chains to the bundle's ``master_did`` before storing, so
a malformed or forged bundle never lands. Whether that master is the one
a recipient *verified in the ceremony* is the client's call against its
own pin — the server can't know that.

Private keys never appear here; this is public material only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.fabric.author_binding import AuthorBindingError, verify_binding
from augmentum.fabric.contact_card import normalize_did
from augmentum.fabric.didkey import did_equal
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class DeviceBundleError(ValueError):
    """Raised when a device bundle is malformed or a binding doesn't chain."""


@dataclass(frozen=True)
class DeviceBundle:
    master_did: str
    devices: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"master_did": self.master_did, "devices": self.devices}


def validate_bundle(master_did: str, devices: list[dict[str, Any]]) -> DeviceBundle:
    """Validate a bundle: well-formed master, ≥1 device, and EVERY device's
    binding verifies against ``master_did`` and vouches for that device's
    own subkey. Raises :class:`DeviceBundleError` on any failure.

    This is the gate: a device whose binding is forged, points at a
    different master, or vouches for a different subkey is rejected — the
    whole bundle is refused rather than silently storing a bad device.
    """
    master = normalize_did(master_did)  # raises on malformed
    if not isinstance(devices, list) or not devices:
        raise DeviceBundleError("bundle must list at least one device")

    clean: list[dict[str, Any]] = []
    for d in devices:
        if not isinstance(d, dict):
            raise DeviceBundleError("each device must be an object")
        subkey = str(d.get("subkey_did", ""))
        sealing = str(d.get("sealing_pub_b64", ""))
        binding = d.get("binding")
        if not subkey or not sealing or not isinstance(binding, dict):
            raise DeviceBundleError("device missing subkey_did / sealing_pub_b64 / binding")
        try:
            bound = verify_binding(binding, expected_master_did=master)
        except AuthorBindingError as exc:
            raise DeviceBundleError(f"device binding does not chain to master: {exc}") from None
        if not did_equal(bound, subkey):
            raise DeviceBundleError("binding vouches for a different subkey than the device")
        clean.append({
            "subkey_did": normalize_did(subkey),
            "sealing_pub_b64": sealing,
            "binding": binding,
            "label": str(d.get("label", "")),
        })
    return DeviceBundle(master_did=master, devices=clean)


async def put_bundle(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    master_did: str,
    devices: list[dict[str, Any]],
) -> DeviceBundle:
    """Validate then store ``user_id``'s device bundle (upsert). Refuses the
    anon row and any bundle that fails :func:`validate_bundle`."""
    if not user_id:
        raise DeviceBundleError("put_bundle requires a non-empty user_id")
    bundle = validate_bundle(master_did, devices)
    await conn.execute(
        "INSERT INTO fabric_device_bundles (user_id, master_did, bundle_json, updated_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "master_did=excluded.master_did, bundle_json=excluded.bundle_json, "
        "updated_at=datetime('now')",
        (user_id, bundle.master_did, json.dumps({"devices": bundle.devices}, separators=(",", ":"))),
    )
    await conn.commit()
    log.info("fabric_device_bundle_published", user_id=user_id, devices=len(bundle.devices))
    return bundle


async def get_bundle(
    conn: aiosqlite.Connection, *, user_id: str,
) -> DeviceBundle | None:
    """Return a user's published bundle, or None if they haven't published."""
    cur = await conn.execute(
        "SELECT master_did, bundle_json FROM fabric_device_bundles WHERE user_id=?",
        (user_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    try:
        devices = json.loads(row[1]).get("devices", [])
    except Exception:
        return None
    return DeviceBundle(master_did=row[0], devices=devices)
