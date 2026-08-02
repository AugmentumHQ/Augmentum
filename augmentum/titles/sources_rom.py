"""InternalRomSource -- emulator ROM upload bridge.

ROM uploads are a two-step flow because ROMs are binary blobs that
don't fit the "POST a JSON manifest" pattern InternalSource uses:

  1. ``POST /api/titles/upload-rom`` (multipart) writes the ROM to
     the blob store + detects the system from filename/header,
     returns a manifest_data dict ready to be passed back to step 2.
  2. ``POST /api/titles/`` with ``source_id="internal-rom"`` and that
     manifest_data creates the artifact row + returns the title.

The upload route lives in ``titles_routes.py`` (it handles multipart
parsing). This module is the Source half: it takes the manifest_data
the upload route produced and turns it into a title artifact row.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from augmentum.titles.manifest import KIND_EMULATOR_ROM
from augmentum.titles.rom_systems import get_system
from augmentum.titles.sources import DiscoveryItem, SourceImportError
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class InternalRomSource:
    """Source for user-uploaded emulator ROMs."""

    id = "internal-rom"
    label = "Uploaded ROM"

    def __init__(self, conn) -> None:
        self._conn = conn

    async def discover(
        self, query: dict[str, Any], *, user_id: str = "",
    ) -> list[DiscoveryItem]:
        # ROM uploads have no public catalog by design -- the user
        # supplies their own ROMs. Return empty so the protocol stays
        # uniform; the upload route is the entry point.
        return []

    async def import_for_user(
        self, manifest_data: dict, *, user_id: str,
    ) -> tuple[str, bool]:
        """Return (artifact_id, created). created=False when the same
        ROM sha already exists for this user; the existing title's id
        is returned and no new row is inserted. The bulk import flow
        depends on this distinction to bucket "imported" vs "duplicate"
        in its summary.
        """
        if not user_id:
            raise SourceImportError("user_id required")

        # The upload route produces a manifest_data dict with these
        # required keys. Validate them defensively in case a caller
        # crafts the request by hand.
        rom_sha = str(manifest_data.get("rom_sha256") or "").strip()
        rom_size = int(manifest_data.get("rom_size_bytes") or 0)
        system_id = str(manifest_data.get("system_id") or "").strip()
        title = str(manifest_data.get("title") or "").strip()
        original_filename = str(
            manifest_data.get("original_filename") or "",
        ).strip()

        if not rom_sha:
            raise SourceImportError("rom_sha256 required (upload first)")
        if not system_id:
            raise SourceImportError("system_id required")
        if not title:
            raise SourceImportError("title required")
        spec = get_system(system_id)
        if spec is None:
            raise SourceImportError(f"unknown system {system_id!r}")

        # Build the title metadata. ``source_remote_id`` doubles as
        # the de-dupe key (sha256 of the ROM bytes); re-importing the
        # same ROM is idempotent.
        existing = await self._find_existing(rom_sha, user_id=user_id)
        if existing:
            return existing, False

        metadata: dict[str, Any] = {
            "kind": KIND_EMULATOR_ROM,
            "source": self.id,
            "source_id": rom_sha,                       # ROM hash = stable id
            "title": title,
            "runtime_preferred": "emulator-browser",
            "runtime_alternates": ["emulator-browser", "emulator-streamed"],
            "system_id": spec.id,
            "system_label": spec.label,
            "libretro_core": spec.libretro_core,
            "bios_required": spec.bios_required,
            "rom_sha256": rom_sha,
            "rom_size_bytes": rom_size,
            "original_filename": original_filename,
            "capabilities": {
                "input_modes": ["keyboard", "gamepad", "touch"],
                "save_states": True,
                "offline": True,
                "multiplayer": _multiplayer_for_system(spec.id),
            },
        }

        # Pass through extra metadata (genre, year, screenshots, ...) —
        # protect discriminator keys.
        extra = manifest_data.get("metadata") or {}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k in metadata:
                    continue
                metadata[k] = v

        artifact_id = await self._insert_artifact(
            user_id=user_id,
            display_name=title,
            metadata=metadata,
        )
        log.info(
            "title_imported_via_internal_rom",
            user_id=user_id, artifact_id=artifact_id,
            system=spec.id, sha=rom_sha, size_bytes=rom_size,
        )
        return artifact_id

    async def _find_existing(
        self, rom_sha: str, *, user_id: str,
    ) -> str | None:
        cursor = await self._conn.execute(
            "SELECT id FROM artifacts "
            "WHERE user_id = ? "
            "  AND json_extract(metadata, '$.kind') = ? "
            "  AND json_extract(metadata, '$.rom_sha256') = ? "
            "LIMIT 1",
            (user_id, KIND_EMULATOR_ROM, rom_sha),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def _insert_artifact(
        self,
        *,
        user_id: str,
        display_name: str,
        metadata: dict,
    ) -> str:
        artifact_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO artifacts
               (id, task_id, session_id, filename, display_name, format,
                size_bytes, path, metadata, user_id, pinned)
               VALUES (?, '', '', ?, ?, '', ?, '', ?, ?, 1)""",
            (
                artifact_id,
                f"{display_name}.rom",
                display_name,
                int(metadata.get("rom_size_bytes") or 0),
                json.dumps(metadata),
                user_id,
            ),
        )
        await self._conn.commit()
        return artifact_id


# ── helpers ──────────────────────────────────────────────────────────


# Most retro systems supported 2 controllers (NES Battle, SNES Bomberman).
# A small number supported 4+ via multitap; we keep it conservative and
# expose a soft cap, not an enforcement mechanism.
_FOUR_PLAYER = frozenset({"snes", "n64", "saturn", "psx"})


def _multiplayer_for_system(system_id: str) -> int:
    if system_id in _FOUR_PLAYER:
        return 4
    return 2
