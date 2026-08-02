"""Stdio ACP entrypoint — the command an editor (Zed) spawns to reach Augmentum.

Run as::

    python -m augmentum.coder.acp_stdio          # smoke loop (default)

Zed's "custom agent" launches a subprocess and speaks ACP to it over stdio, so
this module is the ``Command`` you paste into that form. It stands up an
:class:`AugmentumACPAgent` on stdin/stdout via :func:`acp.run_agent`.

WHY A SMOKE LOOP FIRST — this entrypoint ships with a *self-contained* smoke
``loop_runner`` (``_smoke_loop_runner``) that exercises the full editor op set
through the SAME ``RemoteEditorExecutor`` the real coder loop uses:

    * ``list_files``  -> terminal/run (ls)          — terminal round-trip
    * ``write_file``  -> fs/write_text_file          — the edit->approve->apply spine
    * ``read_file``   -> fs/read_text_file           — read-back verification

That proves the Zed<->agent PROTOCOL end-to-end (transport, session lifecycle,
the fs/terminal handshakes, Zed's approval UX) WITHOUT the heavy app_state
assembly. Swapping ``_smoke_loop_runner`` for the production
``make_coder_loop_runner`` (wired to the live backend/registries the same way
``handler_factory`` builds a CoderHandler) turns this into the real coder loop —
that is the remaining P2.3d integration seam.

CRITICAL — stdio is the ACP wire: **nothing may write to stdout** except the
JSON-RPC frames the SDK emits. :func:`_configure_stderr_logging` forces both
stdlib logging and structlog to stderr so a stray log line can't corrupt the
protocol stream.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from augmentum.coder.acp_agent import LoopEvent, run_stdio
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.acp_agent import EditorSession

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Smoke loop — self-contained protocol proof (no backend/app_state required).
# ---------------------------------------------------------------------------

_SMOKE_FILENAME = "AUGMENTUM_SMOKE.md"
_SMOKE_BODY = (
    "# Augmentum smoke test\n\n"
    "This file was written by Augmentum's coder path through the Agent Client "
    "Protocol.\n\n"
    "If you can see it in your editor, the read -> edit -> approve -> apply "
    "spine works.\n"
)


def _join(cwd: str, name: str) -> str:
    """Absolute-join ``name`` onto ``cwd``, preserving the cwd's separator.

    ACP fs ops require absolute paths. Zed may hand us a POSIX path
    (``/home/me/proj``) or a Windows one (``C:\\Users\\me\\proj``); match whichever
    the editor used rather than forcing one style.
    """
    if "\\" in cwd and "/" not in cwd:
        return cwd.rstrip("\\/") + "\\" + name
    return cwd.rstrip("/") + "/" + name


async def _smoke_loop_runner(
    session: EditorSession, prompt_text: str,
) -> AsyncIterator[LoopEvent]:
    """Exercise the editor op set through ``session.executor`` and narrate it.

    Every step is wrapped so one unsupported op (e.g. ``ls`` missing on a
    Windows editor terminal) reports and continues instead of aborting the turn —
    the fs read/write spine is the load-bearing part and is OS-agnostic.
    """
    ex = session.executor
    yield ("thought", {"text": "Augmentum smoke agent connected over ACP."})
    yield ("text", {"text": f"You said: {prompt_text!r}\n\n"})

    # 1) terminal round-trip -------------------------------------------------
    try:
        entries = await ex.list_files(session.cwd)
        names = ", ".join(getattr(e, "name", str(e)) for e in entries[:20])
        more = "" if len(entries) <= 20 else f" (+{len(entries) - 20} more)"
        yield ("text", {
            "text": f"1. Listed `{session.cwd}` -> {len(entries)} entries: "
                    f"{names}{more}\n",
        })
    except Exception as exc:  # noqa: BLE001 — a failed leg reports, doesn't abort
        yield ("text", {"text": f"1. list_files unavailable ({exc})\n"})

    # 2) write round-trip (edit -> approve -> apply) -------------------------
    marker = _join(session.cwd, _SMOKE_FILENAME)
    wrote = False
    try:
        await ex.write_file(marker, _SMOKE_BODY)
        wrote = True
        yield ("text", {"text": f"2. Wrote `{marker}` (approve it in the editor).\n"})
    except Exception as exc:  # noqa: BLE001
        yield ("text", {"text": f"2. write_file failed: {exc}\n"})

    # 3) read-back verification ---------------------------------------------
    if wrote:
        try:
            got = await ex.read_file(marker)
            ok = "Augmentum smoke test" in got
            verdict = "OK" if ok else "content mismatch"
            yield ("text", {
                "text": f"3. Read it back -> {verdict} ({len(got)} chars).\n",
            })
        except Exception as exc:  # noqa: BLE001
            yield ("text", {"text": f"3. read_file failed: {exc}\n"})

    yield ("text", {"text": "\nSmoke test complete — the ACP editor path is live."})


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

def _debug_log_path() -> str:
    """Where the startup/crash breadcrumb goes (editors swallow stderr)."""
    import tempfile

    return f"{tempfile.gettempdir()}/augmentum_acp_stdio.log"


def _breadcrumb(msg: str) -> None:
    """Append a line to the debug file — the ONE channel an editor can't hide.

    Best-effort and totally isolated from the JSON-RPC stdout stream, so it is
    safe to call before/around the ACP connection. Silently ignored if the file
    can't be written.
    """
    with contextlib.suppress(Exception), open(
        _debug_log_path(), "a", encoding="utf-8",
    ) as fh:
        fh.write(msg.rstrip() + "\n")


def _configure_stderr_logging() -> None:
    """Pin ALL logging to stderr so stdout stays a clean JSON-RPC channel."""
    import structlog

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )


async def _smoke_main() -> None:
    _configure_stderr_logging()
    log.info("acp_stdio_start", mode="smoke")
    _breadcrumb("run_agent: entering listen loop (smoke)")
    await run_stdio(_smoke_loop_runner)
    _breadcrumb("run_agent: listen loop returned (client closed the pipe)")


# ---------------------------------------------------------------------------
# Bridge mode — the PRODUCTION path: relay ACP frames stdio <-> in-process WSS.
# ---------------------------------------------------------------------------

async def _bridge_main(url: str, *, api_key: str = "", model: str = "") -> None:
    """Tunnel Zed's stdio ACP frames to the in-process ``/ws/coder/acp`` endpoint.

    A pure byte relay: stdin -> WS and WS -> stdout. The REAL coder loop runs
    server-side (shared model residency); this process is a dumb pipe so it
    stays tiny and stateless. Reuses the SDK's ``stdio_streams`` for correct
    Windows stdin/stdout handling.
    """
    import os
    import urllib.parse

    import websockets
    from acp.stdio import stdio_streams

    _configure_stderr_logging()
    if model and "model=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}model={urllib.parse.quote(model)}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # Self-signed TLS on the HTTPS front-door (6443) — a same-machine bridge to
    # a local instance can't verify a self-signed cert. AUGMENTUM_ACP_INSECURE=1
    # skips verification for wss:// (localhost trust); plain ws:// needs nothing.
    ssl_ctx = None
    if url.startswith("wss://") and os.environ.get("AUGMENTUM_ACP_INSECURE"):
        import ssl

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    reader, writer = await stdio_streams()
    _breadcrumb(f"bridge: connecting url={url} insecure={ssl_ctx is not None}")
    async with websockets.connect(
        url, additional_headers=headers, max_size=None, ssl=ssl_ctx,
    ) as ws:
        _breadcrumb("bridge: connected")

        async def _stdin_to_ws() -> None:
            while True:
                data = await reader.read(65536)
                if not data:  # stdin EOF — editor closed
                    break
                await ws.send(data)

        async def _ws_to_stdout() -> None:
            async for msg in ws:
                writer.write(msg.encode("utf-8") if isinstance(msg, str) else msg)
                await writer.drain()

        t_in = asyncio.create_task(_stdin_to_ws())
        t_out = asyncio.create_task(_ws_to_stdout())
        _done, pending = await asyncio.wait(
            {t_in, t_out}, return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    _breadcrumb("bridge: closed")


def _resolve_bridge_url(argv: list[str]) -> str:
    """``--bridge <url>`` arg wins; else ``AUGMENTUM_ACP_URL`` env; else ''."""
    import os

    if "--bridge" in argv:
        i = argv.index("--bridge")
        if i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("AUGMENTUM_ACP_URL", "")


def main() -> None:
    import os
    import traceback

    argv = sys.argv[1:]
    url = _resolve_bridge_url(argv)
    mode = "bridge" if url else "smoke"
    _breadcrumb(
        f"--- start mode={mode} pid={os.getpid()} cwd={os.getcwd()!r} "
        f"python={sys.executable!r} argv={sys.argv!r}",
    )
    try:
        if url:
            asyncio.run(
                _bridge_main(
                    url,
                    api_key=os.environ.get("AUGMENTUM_API_KEY", ""),
                    model=os.environ.get("AUGMENTUM_CODER_MODEL", ""),
                ),
            )
        else:
            asyncio.run(_smoke_main())
    except KeyboardInterrupt:  # editor closed the pipe
        _breadcrumb("KeyboardInterrupt")
    except BaseException:  # noqa: BLE001 — capture ANY crash for diagnosis
        _breadcrumb("CRASH:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
