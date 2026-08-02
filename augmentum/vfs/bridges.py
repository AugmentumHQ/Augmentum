"""VFS bridge providers — map virtual paths to real storage locations."""

from __future__ import annotations

import mimetypes
import os
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.vfs.models import VFSNode

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class VFSBridge:
    """Abstract bridge from virtual path to real storage."""

    prefix: str = ""
    source: str = ""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        raise NotImplementedError

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        raise NotImplementedError

    async def read_bytes(self, subpath: str, *, user_id: str) -> bytes | None:
        """Read full file content. For streaming, override read_stream."""
        return None

    async def exists(self, subpath: str, *, user_id: str) -> bool:
        return (await self.stat(subpath, user_id=user_id)) is not None


class ArtifactBridge(VFSBridge):
    """Bridge to artifacts stored on disk with SQLite metadata."""

    prefix = "/Artifacts"
    source = "artifacts"

    def __init__(self, db, artifact_dir: str) -> None:
        super().__init__(db)
        self._dir = artifact_dir

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        nodes = []
        cursor = await self._db.execute(
            "SELECT id, filename, display_name, format, size_bytes, path, "
            "created_at, task_id FROM artifacts WHERE user_id = ? "
            "ORDER BY created_at DESC",
            (user_id,),
        )
        for row in await cursor.fetchall():
            nodes.append(VFSNode(
                path=f"{self.prefix}/{row[1]}",
                name=row[1],
                is_dir=False,
                size=row[4] or 0,
                mime_type=mimetypes.guess_type(row[1])[0] or "",
                modified_at=row[6] or "",
                source=self.source,
                source_id=row[0],
                real_path=os.path.join(self._dir, row[5]) if row[5] else None,
            ))
        return nodes

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        name = subpath.lstrip("/")
        if not name:
            return VFSNode(path=self.prefix, name="Artifacts", is_dir=True)
        cursor = await self._db.execute(
            "SELECT id, filename, size_bytes, path, created_at FROM artifacts "
            "WHERE filename = ? AND user_id = ?",
            (name, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return VFSNode(
            path=f"{self.prefix}/{row[1]}",
            name=row[1],
            size=row[2] or 0,
            mime_type=mimetypes.guess_type(row[1])[0] or "",
            modified_at=row[4] or "",
            source=self.source,
            source_id=row[0],
            real_path=os.path.join(self._dir, row[3]) if row[3] else None,
        )

    async def read_bytes(self, subpath: str, *, user_id: str) -> bytes | None:
        node = await self.stat(subpath, user_id=user_id)
        if not node or not node.real_path or not os.path.exists(node.real_path):
            return None
        with open(node.real_path, "rb") as f:
            return f.read()


class ImageBridge(VFSBridge):
    """Bridge to generated images."""

    prefix = "/Images"
    source = "images"

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        nodes = []
        cursor = await self._db.execute(
            "SELECT image_id, file_path, prompt, model, width, height, created_at "
            "FROM image_generations WHERE user_id = ? AND is_private = 0 "
            "ORDER BY created_at DESC LIMIT 200",
            (user_id,),
        )
        for row in await cursor.fetchall():
            fname = os.path.basename(row[1]) if row[1] else f"{row[0]}.png"
            nodes.append(VFSNode(
                path=f"{self.prefix}/{fname}",
                name=fname,
                size=0,
                mime_type="image/png",
                modified_at=row[6] or "",
                source=self.source,
                source_id=row[0],
                real_path=row[1],
            ))
        return nodes

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        name = subpath.lstrip("/")
        if not name:
            return VFSNode(path=self.prefix, name="Images", is_dir=True)
        # Try by filename or image_id
        cursor = await self._db.execute(
            "SELECT image_id, file_path, created_at FROM image_generations "
            "WHERE (file_path LIKE ? OR image_id = ?) AND user_id = ?",
            (f"%{name}", name.replace(".png", ""), user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return VFSNode(
            path=f"{self.prefix}/{name}", name=name,
            mime_type="image/png", modified_at=row[2] or "",
            source=self.source, source_id=row[0], real_path=row[1],
        )

    async def read_bytes(self, subpath: str, *, user_id: str) -> bytes | None:
        node = await self.stat(subpath, user_id=user_id)
        if not node or not node.real_path or not os.path.exists(node.real_path):
            return None
        with open(node.real_path, "rb") as f:
            return f.read()


class DocumentBridge(VFSBridge):
    """Bridge to uploaded documents (metadata only — content is chunked in DB)."""

    prefix = "/Documents"
    source = "documents"

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        nodes = []
        cursor = await self._db.execute(
            "SELECT id, filename, mime_type, file_size, chunk_count, created_at "
            "FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        for row in await cursor.fetchall():
            nodes.append(VFSNode(
                path=f"{self.prefix}/{row[1]}",
                name=row[1],
                size=row[3] or 0,
                mime_type=row[2] or "",
                modified_at=row[5] or "",
                source=self.source,
                source_id=row[0],
            ))
        return nodes

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        name = subpath.lstrip("/")
        if not name:
            return VFSNode(path=self.prefix, name="Documents", is_dir=True)
        cursor = await self._db.execute(
            "SELECT id, filename, mime_type, file_size, created_at FROM documents "
            "WHERE filename = ? AND user_id = ?",
            (name, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return VFSNode(
            path=f"{self.prefix}/{row[1]}", name=row[1],
            size=row[3] or 0, mime_type=row[2] or "",
            modified_at=row[4] or "", source=self.source, source_id=row[0],
        )


class KnowledgeBridge(VFSBridge):
    """Bridge to knowledge packs on disk."""

    prefix = "/Knowledge"
    source = "knowledge"

    def __init__(self, db, knowledge_dir: str) -> None:
        super().__init__(db)
        self._dir = knowledge_dir

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        nodes = []
        if not os.path.isdir(self._dir):
            return nodes
        for fname in os.listdir(self._dir):
            fpath = os.path.join(self._dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.endswith((".augpack", ".zim")):
                continue
            stat = os.stat(fpath)
            nodes.append(VFSNode(
                path=f"{self.prefix}/{fname}",
                name=fname,
                size=stat.st_size,
                mime_type="application/x-augpack" if fname.endswith(".augpack") else "application/x-zim",
                modified_at=str(stat.st_mtime),
                source=self.source,
                source_id=fname,
                real_path=fpath,
            ))
        return nodes

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        name = subpath.lstrip("/")
        if not name:
            return VFSNode(path=self.prefix, name="Knowledge", is_dir=True)
        fpath = os.path.join(self._dir, name)
        if not os.path.isfile(fpath):
            return None
        stat = os.stat(fpath)
        return VFSNode(
            path=f"{self.prefix}/{name}", name=name,
            size=stat.st_size, source=self.source, source_id=name,
            real_path=fpath,
        )

    async def read_bytes(self, subpath: str, *, user_id: str) -> bytes | None:
        node = await self.stat(subpath, user_id=user_id)
        if not node or not node.real_path:
            return None
        with open(node.real_path, "rb") as f:
            return f.read()


class VoiceBridge(VFSBridge):
    """Bridge to voice samples and mixes."""

    prefix = "/Voices"
    source = "voices"

    def __init__(self, db, voices_dir: str) -> None:
        super().__init__(db)
        self._dir = voices_dir

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        nodes = []
        if os.path.isdir(self._dir):
            for fname in os.listdir(self._dir):
                fpath = os.path.join(self._dir, fname)
                if not os.path.isfile(fpath):
                    continue
                stat = os.stat(fpath)
                nodes.append(VFSNode(
                    path=f"{self.prefix}/{fname}", name=fname,
                    size=stat.st_size, mime_type="audio/wav",
                    source=self.source, source_id=fname, real_path=fpath,
                ))
        return nodes

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        name = subpath.lstrip("/")
        if not name:
            return VFSNode(path=self.prefix, name="Voices", is_dir=True)
        fpath = os.path.join(self._dir, name)
        if not os.path.isfile(fpath):
            return None
        stat = os.stat(fpath)
        return VFSNode(
            path=f"{self.prefix}/{name}", name=name,
            size=stat.st_size, mime_type="audio/wav",
            source=self.source, source_id=name, real_path=fpath,
        )


class ChatImageBridge(VFSBridge):
    """Bridge to inline chat images stored as BLOBs."""

    prefix = "/Chat Images"
    source = "chat_images"

    async def list(self, subpath: str, *, user_id: str) -> list[VFSNode]:
        nodes = []
        cursor = await self._db.execute(
            "SELECT id, mime_type, length(data), created_at FROM chat_images "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
            (user_id,),
        )
        for row in await cursor.fetchall():
            ext = (row[1] or "image/png").split("/")[-1]
            fname = f"{row[0][:12]}.{ext}"
            nodes.append(VFSNode(
                path=f"{self.prefix}/{fname}", name=fname,
                size=row[2] or 0, mime_type=row[1] or "image/png",
                modified_at=row[3] or "", source=self.source, source_id=row[0],
            ))
        return nodes

    async def stat(self, subpath: str, *, user_id: str) -> VFSNode | None:
        name = subpath.lstrip("/")
        if not name:
            return VFSNode(path=self.prefix, name="Chat Images", is_dir=True)
        img_id = name.split(".")[0]
        cursor = await self._db.execute(
            "SELECT id, mime_type, length(data), created_at FROM chat_images "
            "WHERE id LIKE ? AND user_id = ?",
            (f"{img_id}%", user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return VFSNode(
            path=f"{self.prefix}/{name}", name=name,
            size=row[2] or 0, mime_type=row[1] or "image/png",
            modified_at=row[3] or "", source=self.source, source_id=row[0],
        )

    async def read_bytes(self, subpath: str, *, user_id: str) -> bytes | None:
        name = subpath.lstrip("/")
        img_id = name.split(".")[0]
        cursor = await self._db.execute(
            "SELECT data FROM chat_images WHERE id LIKE ? AND user_id = ?",
            (f"{img_id}%", user_id),
        )
        row = await cursor.fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# VFS Router
# ---------------------------------------------------------------------------

class VFS:
    """Routes virtual paths to the appropriate bridge provider."""

    def __init__(self) -> None:
        self._bridges: dict[str, VFSBridge] = {}

    def register_bridge(self, bridge: VFSBridge) -> None:
        self._bridges[bridge.prefix] = bridge

    def _resolve(self, path: str) -> tuple[VFSBridge | None, str]:
        """Resolve a virtual path to (bridge, subpath)."""
        path = "/" + path.strip("/")
        for prefix, bridge in self._bridges.items():
            if path == prefix or path.startswith(prefix + "/"):
                subpath = path[len(prefix):]
                return bridge, subpath
        return None, path

    async def list(self, path: str, *, user_id: str) -> list[VFSNode]:
        """List contents at a virtual path."""
        path = "/" + path.strip("/")
        if path == "/":
            return [
                VFSNode(path=b.prefix, name=b.prefix.strip("/"), is_dir=True, source=b.source)
                for b in self._bridges.values()
            ]
        bridge, subpath = self._resolve(path)
        if not bridge:
            return []
        return await bridge.list(subpath, user_id=user_id)

    async def stat(self, path: str, *, user_id: str) -> VFSNode | None:
        path = "/" + path.strip("/")
        if path == "/":
            return VFSNode(path="/", name="", is_dir=True)
        bridge, subpath = self._resolve(path)
        if not bridge:
            return None
        return await bridge.stat(subpath, user_id=user_id)

    async def read(self, path: str, *, user_id: str) -> bytes | None:
        bridge, subpath = self._resolve(path)
        if not bridge:
            return None
        return await bridge.read_bytes(subpath, user_id=user_id)

    async def exists(self, path: str, *, user_id: str) -> bool:
        return (await self.stat(path, user_id=user_id)) is not None
