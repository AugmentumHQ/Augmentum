"""Bundled avatar manifest and DB seeding."""
from __future__ import annotations

import json
import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Valid mannerism keys (for validation / documentation)
VALID_MANNERISM_KEYS = {
    "gesture_frequency",
    "eye_contact_tendency",
    "sway_amplitude",
    "breathing_rate",
    "blink_interval",
    "idle_weight_profile",
    "speaking_energy",
}

BUNDLED_AVATARS: list[dict[str, Any]] = [
    # Project-owned VRoid defaults. Keep this list intentionally small so the
    # shipped app only includes avatars with clear project licensing.
    {
        "id": "bundled_m_vance",
        "name": "Vance",
        "vrm_filename": "vance.vrm",
        "source_model": "Vance (creator-licensed CC0, contributed by project author)",
        "mannerisms": {
            "gesture_frequency": 0.45,
            "eye_contact_tendency": 0.8,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
            # Vance ships rotated relative to the other bundled VRMs.
            # face_rotation_y is an explicit override that REPLACES the
            # arm-axis heuristic when present (radians). π lands him
            # front-facing the camera in both the thumbnail render and
            # the voice-call scene.
            "face_rotation_y": 3.141592653589793,
        },
    },
    {
        "id": "bundled_f_becca",
        "name": "Becca",
        "vrm_filename": "Becca.vrm",
        "source_model": "Becca (project-owned custom VRoid Studio avatar)",
        "mannerisms": {
            "gesture_frequency": 0.5,
            "eye_contact_tendency": 0.82,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
        },
    },
    {
        "id": "bundled_f_lise",
        "name": "Lise",
        "vrm_filename": "Lise.vrm",
        "source_model": "Lise (project-owned custom VRoid Studio avatar)",
        "mannerisms": {
            "gesture_frequency": 0.5,
            "eye_contact_tendency": 0.80,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
        },
    },
    {
        "id": "bundled_m_danny",
        "name": "Danny",
        "vrm_filename": "Danny.vrm",
        "source_model": "Danny (project-owned custom VRoid Studio avatar)",
        "mannerisms": {
            "gesture_frequency": 0.45,
            "eye_contact_tendency": 0.78,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
        },
    },
    {
        "id": "bundled_f_roxanne",
        "name": "Roxanne",
        "vrm_filename": "Roxanne.vrm",
        "source_model": "Roxanne (project-owned custom VRoid Studio avatar)",
        "mannerisms": {
            "gesture_frequency": 0.5,
            "eye_contact_tendency": 0.82,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
        },
    },
    {
        "id": "bundled_f_luna",
        "name": "Luna",
        "vrm_filename": "Luna.vrm",
        "source_model": "Luna (project-owned custom VRoid Studio avatar)",
        "mannerisms": {
            "gesture_frequency": 0.5,
            "eye_contact_tendency": 0.82,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
        },
    },
    {
        "id": "bundled_m_louis",
        "name": "Louis",
        "vrm_filename": "Louis.vrm",
        "source_model": "Louis (project-owned custom VRoid Studio avatar)",
        "mannerisms": {
            "gesture_frequency": 0.45,
            "eye_contact_tendency": 0.78,
            "sway_amplitude": 0.18,
            "breathing_rate": 0.95,
            "blink_interval": [3.0, 5.0],
            "idle_weight_profile": "calm",
            "speaking_energy": 0.55,
        },
    },
]


async def seed_bundled_avatars(store: Any, bundled_dir: str) -> int:
    """Seed bundled avatar records into the DB.

    Prunes bundled records that are no longer in the shipped manifest, then
    inserts records only for VRM files that exist on disk. Safe to call
    multiple times: calling a second time always returns 0.

    Args:
        store: An ``AvatarStore`` instance (needs ``._conn``).
        bundled_dir: Directory that contains the bundled ``.vrm`` files.

    Returns:
        Number of avatars newly inserted.
    """
    conn = store._conn
    manifest_ids = {avatar["id"] for avatar in BUNDLED_AVATARS}
    pruned = 0

    if manifest_ids:
        placeholders = ",".join("?" for _ in manifest_ids)
        cursor = await conn.execute(
            f"DELETE FROM avatars WHERE is_bundled = 1 AND id NOT IN ({placeholders})",
            tuple(sorted(manifest_ids)),
        )
        pruned = max(cursor.rowcount or 0, 0)
    else:
        cursor = await conn.execute("DELETE FROM avatars WHERE is_bundled = 1")
        pruned = max(cursor.rowcount or 0, 0)
    if pruned:
        log.info("stale_bundled_avatars_pruned", count=pruned)

    cursor = await conn.execute("SELECT id FROM avatars WHERE is_bundled = 1")
    existing_ids: set[str] = {row[0] for row in await cursor.fetchall()}

    seeded = 0
    for avatar in BUNDLED_AVATARS:
        if avatar["id"] in existing_ids:
            continue

        vrm_path = os.path.join(bundled_dir, avatar["vrm_filename"])
        if not os.path.isfile(vrm_path):
            log.debug(
                "bundled_vrm_not_found",
                avatar_id=avatar["id"],
                path=vrm_path,
            )
            continue

        await conn.execute(
            """INSERT INTO avatars
               (id, vrm_path, mannerisms, is_bundled, type,
                created_at, updated_at)
               VALUES (?, ?, ?, 1, 'vrm', datetime('now'), datetime('now'))""",
            (
                avatar["id"],
                vrm_path,
                json.dumps(avatar["mannerisms"]),
            ),
        )
        seeded += 1
        log.debug("bundled_avatar_seeded", avatar_id=avatar["id"], name=avatar["name"])

    if seeded or pruned:
        await conn.commit()

    return seeded
