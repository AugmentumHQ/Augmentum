"""Session B attachment route tests.

Covers the HTTP surface added to ``connect_routes.py``:

* ``GET  /api/connect/threads/{tid}/messages/{mid}/attachment``  → bytes
* ``HEAD /api/connect/threads/{tid}/messages/{mid}/attachment``  → metadata

Cross-user access enforcement is tested at the access-grant primitive
layer (``get_message`` returns None for non-participants — see
``tests/test_connect_message_store.py``). The route's gate is a thin
wrapper around that call, so the route's 404 case for non-participants
is implied by the store-level tests rather than re-exercised here.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from pathlib import Path

import pytest

from augmentum.config import settings


@pytest.fixture(autouse=True)
def _enable_connect():
    """Force connect_enabled=True for the duration of each test."""

    orig = settings.connect_enabled
    object.__setattr__(settings, "connect_enabled", True)
    yield
    object.__setattr__(settings, "connect_enabled", orig)


def _upload_one_file(client, body: bytes, filename: str, mime: str) -> str:
    """Helper: bypass the upload route and write directly to the blob
    store + uploads table. The upload route needs an UploadsAdapter
    on app.state that the basic sqlite_client fixture doesn't
    bootstrap; for these tests we only need the blob bytes + uploads
    row to exist — the attachment route's job is to resolve them, not
    to create them.

    Returns the upload_id ready to pass as ``attachment_ref``.
    """

    from augmentum.proxy.server import _SETTINGS_RESTORE_MAP  # noqa: F401 — ensures app initialized
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.vfs.blobs import BlobStore

    sm = client.app.state.state_manager
    backend = sm.backend
    assert isinstance(backend, SQLiteBackend), "test expects sqlite backend"
    conn = backend.conn

    # Resolve current user_id the same way the routes do — auth
    # middleware writes user_id into the scope. The sqlite_client
    # fixture uses Bearer test-token; the actual user_id is whatever
    # auth resolves that to. Easiest path: send a dummy message and
    # inspect what the server thinks the user_id is.
    probe = client.post(
        "/api/connect/threads/auto/send",
        json={"peer_did": "bob@this-instance", "body": "_probe"},
    )
    assert probe.status_code == 200, probe.text
    probe_thread = probe.json()["thread_id"]
    # Read back the message to find the sender's user_id.
    msgs = client.get(
        f"/api/connect/threads/{probe_thread}/messages"
    ).json()["messages"]
    user_id = msgs[0]["user_id"]

    sha = hashlib.sha256(body).hexdigest()

    async def _store():
        # Write blob bytes to disk via BlobStore (handles dir layout +
        # refcount + the blobs row), then add an uploads row pointing
        # at it. We skip the VFS register_file step; the attachment
        # route doesn't consult file_index.
        bs = BlobStore(conn)
        blob = await bs.write(body, mime_type=mime)
        upload_id = f"ul_{secrets.token_hex(8)}"
        await conn.execute(
            "INSERT INTO uploads (id, user_id, filename, blob_sha, "
            "size_bytes, mime_type, mime_sniffed, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, '{}')",
            (upload_id, user_id, filename, blob["sha256"],
             blob["size_bytes"], mime, mime),
        )
        await conn.commit()
        return upload_id

    return asyncio.get_event_loop().run_until_complete(_store())


def _send_with_attachment(
    client, *, peer_did: str, body: str, attachment_ref: str,
    attachment_name: str = "", attachment_mime: str = "", attachment_size: int = 0,
):
    payload = {
        "peer_did": peer_did,
        "body": body,
        "format": "plain",
        "attachment_ref": attachment_ref,
    }
    if attachment_name:
        payload["attachment_name"] = attachment_name
    if attachment_mime:
        payload["attachment_mime"] = attachment_mime
    if attachment_size:
        payload["attachment_size"] = attachment_size
    resp = client.post("/api/connect/threads/auto/send", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestAttachmentRoute:
    """End-to-end: upload → send-with-attachment → fetch bytes."""

    def test_get_attachment_returns_bytes(self, sqlite_client) -> None:
        payload_bytes = b"hello attachment world\n"
        upload_id = _upload_one_file(
            sqlite_client, payload_bytes, "note.txt", "text/plain",
        )
        sent = _send_with_attachment(
            sqlite_client,
            peer_did="bob@this-instance",
            body="see attached",
            attachment_ref=upload_id,
            attachment_name="note.txt",
            attachment_mime="text/plain",
            attachment_size=len(payload_bytes),
        )

        resp = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment"
        )
        assert resp.status_code == 200, resp.text
        assert resp.content == payload_bytes
        # MIME comes from the sniffed value (text/plain for plain text bytes).
        assert resp.headers.get("content-type", "").startswith("text/")
        # Default inline disposition (no ?download).
        cd = resp.headers.get("content-disposition", "")
        assert cd.startswith("inline")
        assert "note.txt" in cd

    def test_download_query_flips_disposition(self, sqlite_client) -> None:
        upload_id = _upload_one_file(
            sqlite_client, b"xx", "x.bin", "application/octet-stream",
        )
        sent = _send_with_attachment(
            sqlite_client, peer_did="bob@this-instance",
            body="", attachment_ref=upload_id,
        )
        resp = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment",
            params={"download": "1"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-disposition", "").startswith("attachment")

    def test_head_returns_metadata(self, sqlite_client) -> None:
        upload_id = _upload_one_file(
            sqlite_client, b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
            "pic.png", "image/png",
        )
        sent = _send_with_attachment(
            sqlite_client, peer_did="bob@this-instance",
            body="", attachment_ref=upload_id,
        )
        resp = sqlite_client.head(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment"
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("image/")
        assert resp.headers.get("x-attachment-filename") == "pic.png"
        assert int(resp.headers.get("content-length", "0")) > 0

    def test_unknown_message_404(self, sqlite_client) -> None:
        resp = sqlite_client.get(
            "/api/connect/threads/any/messages/does-not-exist/attachment"
        )
        assert resp.status_code == 404

    def test_message_without_attachment_404(self, sqlite_client) -> None:
        # Send a regular message (no attachment_ref).
        resp = sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={"peer_did": "bob@this-instance", "body": "hi"},
        )
        assert resp.status_code == 200
        sent = resp.json()
        resp = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment"
        )
        assert resp.status_code == 404

    def test_wrong_thread_id_404(self, sqlite_client) -> None:
        upload_id = _upload_one_file(
            sqlite_client, b"hi", "h.txt", "text/plain",
        )
        sent = _send_with_attachment(
            sqlite_client, peer_did="bob@this-instance",
            body="", attachment_ref=upload_id,
        )
        # Right message_id, wrong thread_id — must NOT serve. The
        # check that the message belongs to the supplied thread is
        # what prevents thread_id from being decorative.
        resp = sqlite_client.get(
            f"/api/connect/threads/wrong-thread/messages/{sent['message_id']}/attachment"
        )
        assert resp.status_code == 404

    def test_503_when_connect_disabled(self, sqlite_client) -> None:
        object.__setattr__(settings, "connect_enabled", False)
        resp = sqlite_client.get(
            "/api/connect/threads/t/messages/m/attachment"
        )
        assert resp.status_code == 503

    def test_send_with_attachment_metadata_round_trips_in_message(self, sqlite_client) -> None:
        """Sender's attachment_name/mime/size hint should be readable
        from the messages list — even though it's not persisted on
        the server, the same user (sender) sees their own row with
        attachment_ref set."""

        upload_id = _upload_one_file(
            sqlite_client, b"binary-bytes", "doc.bin", "application/octet-stream",
        )
        sent = _send_with_attachment(
            sqlite_client,
            peer_did="bob@this-instance",
            body="here you go",
            attachment_ref=upload_id,
        )
        resp = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages"
        )
        assert resp.status_code == 200
        msgs = resp.json()["messages"]
        # The helper sends a probe message first (to discover the
        # auth-resolved user_id) so multiple messages are expected in
        # the thread. Find the one that actually carries the
        # attachment_ref we set.
        target = next(
            (m for m in msgs if m["attachment_ref"] == upload_id), None,
        )
        assert target is not None, f"no message with attachment_ref={upload_id}"
        assert target["message_id"] == sent["message_id"]


class TestAttachmentResilience:
    """Edge cases that can crop up in real use: the sender clears
    their upload between send and recipient fetch, the blob bytes
    vanish from disk, the recipient probes a message they don't
    have a row for. All resolve to 404 — never a 5xx."""

    def test_404_when_upload_row_deleted(self, sqlite_client) -> None:
        """Sender cleared their upload after sending. Recipient's
        ``connect_messages`` row still references attachment_ref,
        but the ``uploads`` row is gone. Route returns 404 cleanly
        with ``detail='attachment expired'`` — never a 500."""

        import asyncio

        payload_bytes = b"will be orphaned"
        upload_id = _upload_one_file(
            sqlite_client, payload_bytes, "ephemeral.txt", "text/plain",
        )
        sent = _send_with_attachment(
            sqlite_client,
            peer_did="bob@this-instance",
            body="briefly available",
            attachment_ref=upload_id,
        )

        # Verify it works first (sanity check).
        ok = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment"
        )
        assert ok.status_code == 200

        # Delete the upload row — simulating sender clearing it.
        backend = sqlite_client.app.state.state_manager.backend
        conn = backend.conn

        async def _drop():
            await conn.execute(
                "DELETE FROM uploads WHERE id = ?", (upload_id,),
            )
            await conn.commit()
        asyncio.get_event_loop().run_until_complete(_drop())

        # Now the fetch should 404 with 'attachment expired'.
        resp = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment"
        )
        assert resp.status_code == 404
        assert "expired" in resp.json().get("detail", "")

    def test_404_when_blob_bytes_missing(self, sqlite_client) -> None:
        """Upload row exists but the blob has been hand-deleted from
        disk (administrative cleanup). Route returns 404 'attachment
        bytes missing' — never a 500 unwrapping a missing-file
        exception."""

        import asyncio

        from pathlib import Path

        upload_id = _upload_one_file(
            sqlite_client, b"some bytes", "doomed.txt", "text/plain",
        )
        sent = _send_with_attachment(
            sqlite_client,
            peer_did="bob@this-instance",
            body="", attachment_ref=upload_id,
        )

        # Resolve the blob's on-disk path through BlobStore, then nuke it.
        backend = sqlite_client.app.state.state_manager.backend
        conn = backend.conn

        async def _resolve_and_delete():
            cur = await conn.execute(
                "SELECT blob_sha FROM uploads WHERE id = ?", (upload_id,),
            )
            (sha,) = await cur.fetchone()
            from augmentum.vfs.blobs import BlobStore
            blob = await BlobStore(conn).get(sha)
            real = Path(blob["real_path"])
            real.unlink(missing_ok=True)

        asyncio.get_event_loop().run_until_complete(_resolve_and_delete())

        resp = sqlite_client.get(
            f"/api/connect/threads/{sent['thread_id']}/messages/{sent['message_id']}/attachment"
        )
        assert resp.status_code == 404
        # We accept either 'bytes missing' (file unlinked) or
        # 'expired' (defensive) — the contract is no 500.
        detail = resp.json().get("detail", "")
        assert "missing" in detail or "expired" in detail
