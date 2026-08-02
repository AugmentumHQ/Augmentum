"""WebDAV server integration via WsgiDAV."""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

from wsgidav.dav_provider import DAVCollection, DAVNonCollection, DAVProvider
from wsgidav.dav_error import HTTP_FORBIDDEN, HTTP_NOT_FOUND, DAVError

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.vfs.bridges import VFS

log = get_logger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync WsgiDAV context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return loop.run_until_complete(coro)
    except Exception:
        return None


def _get_user_id(environ: dict) -> str:
    """Extract user_id from WSGI environ (set by auth middleware via ASGI scope)."""
    scope = environ.get("asgi.scope", {})
    user = scope.get("user")
    if user:
        return user.id
    return ""


class VFSDAVProvider(DAVProvider):
    """WsgiDAV provider backed by Augmentum's VFS."""

    def __init__(self, vfs: VFS):
        super().__init__()
        self._vfs = vfs

    def get_resource_inst(self, path, environ):
        user_id = _get_user_id(environ)
        if not user_id:
            raise DAVError(HTTP_FORBIDDEN)

        path = path.rstrip("/") or "/"
        node = _run_async(self._vfs.stat(path, user_id=user_id))

        if node is None:
            return None

        if node.is_dir:
            return VFSDAVCollection(path, environ, node, self._vfs, user_id)
        return VFSDAVFile(path, environ, node, self._vfs, user_id)


class VFSDAVCollection(DAVCollection):
    """A directory in the VFS (e.g., /Artifacts/, /Images/)."""

    def __init__(self, path, environ, node, vfs, user_id):
        super().__init__(path, environ)
        self._node = node
        self._vfs = vfs
        self._user_id = user_id

    def get_display_info(self):
        return {"type": "Directory"}

    def get_member_names(self):
        nodes = _run_async(self._vfs.list(self._node.path, user_id=self._user_id))
        if not nodes:
            return []
        return [n.name for n in nodes]

    def get_member(self, name):
        child_path = f"{self._node.path.rstrip('/')}/{name}"
        child_node = _run_async(self._vfs.stat(child_path, user_id=self._user_id))
        if not child_node:
            raise DAVError(HTTP_NOT_FOUND)
        if child_node.is_dir:
            return VFSDAVCollection(child_path, self.environ, child_node, self._vfs, self._user_id)
        return VFSDAVFile(child_path, self.environ, child_node, self._vfs, self._user_id)


class VFSDAVFile(DAVNonCollection):
    """A file in the VFS."""

    def __init__(self, path, environ, node, vfs, user_id):
        super().__init__(path, environ)
        self._node = node
        self._vfs = vfs
        self._user_id = user_id

    def get_content_length(self):
        return self._node.size

    def get_content_type(self):
        return self._node.mime_type or "application/octet-stream"

    def get_display_info(self):
        return {"type": self._node.mime_type or "File"}

    def get_content(self):
        data = _run_async(self._vfs.read(self._node.path, user_id=self._user_id))
        if data is None:
            raise DAVError(HTTP_NOT_FOUND)
        return io.BytesIO(data)

    def support_etag(self):
        return False

    def support_ranges(self):
        return False


def create_webdav_app(vfs: VFS):
    """Create and return the WsgiDAV WSGI application."""
    from wsgidav.wsgidav_app import WsgiDAVApp

    config = {
        "provider_mapping": {"/": VFSDAVProvider(vfs)},
        "verbose": 0,
        "logging": {"enable": False},
        "simple_dc": {"user_mapping": {"*": True}},  # Auth handled by our ASGI middleware
        "http_authenticator": {
            "accept_basic": False,
            "accept_digest": False,
            "default_to_digest": False,
        },
    }
    return WsgiDAVApp(config)
