"""Server-side resolver for [file:fi_xxx] tokens in chat messages.

Users insert tokens via the Files panel "Reference in Chat" action or by
clicking a chip in the related-files strip. This resolver runs in the chat
route layer (after image-URL resolution, before the request reaches the
backend) and expands tokens inline so the LLM actually sees the file:

  - image      -> base64 data URL appended to msg.images (vision-ready)
  - text/code  -> fenced snippet appended to msg.content (first 4000 chars)
  - binary     -> compact "Attached: name (size) - description" line

The token text itself is left in place (it's user-visible, so removing it
would surprise them); the resolved content is appended to the same message.

Caps prevent a single chat turn from blowing the context window:
  * MAX_INLINE_FILES = 5 distinct files per turn
  * MAX_TEXT_CHARS   = 4000 chars per text/code file
  * MAX_IMAGE_BYTES  = 8 MB per image
"""

from __future__ import annotations

import base64
import os
import re

from augmentum.models.base import InternalChatRequest
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"\[file:(fi_[a-zA-Z0-9]+)\]")

_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "ico"}
_TEXT_EXTS = {
    "js", "ts", "jsx", "tsx", "py", "rb", "go", "rs", "c", "cpp", "h", "hpp",
    "java", "kt", "swift", "cs", "php", "sh", "bash", "zsh", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "json", "xml", "html", "htm", "css", "scss",
    "less", "sql", "md", "txt", "log", "csv", "env", "dockerfile", "makefile",
}

_SOURCE_PREFIX = {
    "artifacts":   "/Artifacts",
    "images":      "/Images",
    "chat_images": "/Chat Images",
    "knowledge":   "/Knowledge",
    "voices":      "/Voices",
    "documents":   "/Documents",
}

MAX_INLINE_FILES = 5
MAX_TEXT_CHARS = 4000
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _is_image(entry) -> bool:
    if entry.mime_type and entry.mime_type.startswith("image/"):
        return True
    return _ext(entry.name) in _IMAGE_EXTS


def _is_text(entry) -> bool:
    mt = (entry.mime_type or "").lower()
    if mt.startswith("text/") or mt in ("application/json", "application/xml", "application/yaml"):
        return True
    return _ext(entry.name) in _TEXT_EXTS


def _human_size(n: int) -> str:
    if not n or n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    f, i = float(n), 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{int(f)} {units[i]}" if i == 0 else f"{f:.1f} {units[i]}"


async def _read_bytes(entry, vfs) -> bytes | None:
    if entry.real_path:
        try:
            if os.path.exists(entry.real_path):
                with open(entry.real_path, "rb") as f:
                    return f.read()
        except OSError:
            pass
    if vfs:
        prefix = _SOURCE_PREFIX.get(entry.source, "")
        if prefix:
            try:
                return await vfs.read(f"{prefix}/{entry.name}", user_id=entry.user_id)
            except Exception:
                return None
    return None


async def resolve_file_tokens(
    req: InternalChatRequest,
    *,
    user_id: str,
    file_index,
    vfs,
) -> None:
    """Expand [file:fi_xxx] tokens in the latest user message. Mutates in place."""
    if not user_id or not file_index:
        return

    target = None
    for msg in reversed(req.messages):
        if msg.role == "user":
            target = msg
            break
    if target is None or not isinstance(target.content, str):
        return
    if "[file:" not in target.content:
        return

    ids_seen: list[str] = []
    for m in _TOKEN_RE.finditer(target.content):
        fid = m.group(1)
        if fid not in ids_seen:
            ids_seen.append(fid)
        if len(ids_seen) >= MAX_INLINE_FILES:
            break
    if not ids_seen:
        return

    blocks: list[str] = []
    for fid in ids_seen:
        try:
            entry = await file_index.get(fid, user_id=user_id)
        except Exception:
            log.warning("file_token_lookup_failed", file_id=fid, exc_info=True)
            entry = None
        if not entry:
            blocks.append(f"[Referenced file {fid} not found or access denied.]")
            continue

        size_str = _human_size(entry.size_bytes or 0)

        if _is_image(entry):
            data = await _read_bytes(entry, vfs)
            if data and len(data) <= MAX_IMAGE_BYTES:
                mime = entry.mime_type or "image/png"
                data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
                if target.images is None:
                    target.images = []
                target.images.append(data_url)
                blocks.append(f"[Attached image: {entry.name} ({size_str})]")
            else:
                desc = f" - {entry.description}" if entry.description else ""
                blocks.append(f"[Attached image (too large or unavailable): {entry.name} ({size_str}){desc}]")
            continue

        if _is_text(entry):
            data = await _read_bytes(entry, vfs)
            if data:
                text = data.decode("utf-8", errors="replace")
                snippet = text[:MAX_TEXT_CHARS]
                truncated = f"\n... [truncated, full size {size_str}]" if len(text) > MAX_TEXT_CHARS else ""
                fence = _ext(entry.name)
                blocks.append(
                    f"--- Attached file: {entry.name} ({size_str}) ---\n"
                    f"```{fence}\n{snippet}{truncated}\n```"
                )
            else:
                blocks.append(f"[Attached file (content unavailable): {entry.name} ({size_str})]")
            continue

        desc = f" - {entry.description}" if entry.description else ""
        blocks.append(
            f"[Attached file: {entry.name} ({entry.mime_type or 'binary'}, {size_str}){desc}]"
        )

    if blocks:
        target.content = target.content.rstrip() + "\n\n" + "\n\n".join(blocks)
