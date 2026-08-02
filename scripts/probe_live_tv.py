"""Probe Live TV on the user's configured Emby / Jellyfin servers.

Reads ``user_media_servers`` from the live SQLite DB, instantiates the
matching provider, and calls the new ``list_live_channels()`` /
``list_live_programs()`` methods. Prints what comes back so we can
verify the integration design against real data before writing the
full LiveTV provider + UI.

Run from inside the container (Python deps already there):

    docker exec augmentum-augmentum-1 python /app/scripts/probe_live_tv.py
    docker exec augmentum-augmentum-1 python /app/scripts/probe_live_tv.py --user shadow
    docker exec augmentum-augmentum-1 python /app/scripts/probe_live_tv.py --provider emby

Read-only. No writes to the DB, no edits to the upstream server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Bind-mounted at /app/augmentum, so the import works in-container.
sys.path.insert(0, "/app")

import httpx

from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider  # noqa: F401  (registers)
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.emby_compat import EmbyCompatBase
from augmentum.media.providers.jellyfin import JellyfinProvider

PROVIDER_CLASSES: dict[str, type[EmbyCompatBase]] = {
    "emby": EmbyProvider,
    "jellyfin": JellyfinProvider,
}


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _trim(value: Any, n: int = 80) -> str:
    s = str(value or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _load_servers(db_path: Path, *, username: str = "", provider: str = "") -> list[dict]:
    """Pull user_media_servers rows the probe should hit.

    Filters to Emby/Jellyfin since those are the only providers that
    expose Live TV today.
    """
    if not db_path.exists():
        print(f"DB not found at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = """
            SELECT s.id, s.user_id, s.provider, s.name, s.base_url,
                   s.access_token, s.status, s.status_detail,
                   u.username
              FROM user_media_servers s
              LEFT JOIN users u ON u.id = s.user_id
             WHERE s.provider IN ('emby', 'jellyfin')
        """
        params: list[Any] = []
        if username:
            sql += " AND u.username = ?"
            params.append(username)
        if provider:
            sql += " AND s.provider = ?"
            params.append(provider)
        sql += " ORDER BY u.username, s.name"
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


async def _probe_server(row: dict, *, http: httpx.AsyncClient, sample: int = 10) -> None:
    cls = PROVIDER_CLASSES.get(row["provider"])
    if cls is None:
        print(f"  (no provider class for {row['provider']!r})")
        return
    provider = cls(http)
    base_url = row["base_url"]
    token = row["access_token"]

    _print_section(
        f"{row['provider'].upper()} — {row['name']!r} ({base_url})  "
        f"user={row.get('username') or row['user_id']!r}  status={row['status']}"
    )

    if not token:
        print("  no access_token — skipping (server not connected yet)")
        return

    # ---------------- ping ----------------
    info = await provider.ping(base_url)
    if info is None:
        print("  ping: FAILED (server unreachable / not Emby-compatible)")
        return
    print(f"  ping ok: {info.server_name!r} v{info.version}")

    # ---------------- discover libraries (for Live TV library) ----------------
    try:
        libs = await provider.discover_libraries(base_url, token)
    except Exception as exc:
        libs = []
        print(f"  discover_libraries: FAILED — {exc}")
    livetv_libs = [
        lib for lib in libs
        if (lib.collection_type or "").lower() in ("livetv", "live_tv")
        or (lib.extra.get("detected_group") if isinstance(lib.extra, dict) else None) == "live_tv"
    ]
    print(f"  libraries: {len(libs)} total, {len(livetv_libs)} Live TV")
    for lib in livetv_libs:
        print(f"    - {lib.name!r}  collection_type={lib.collection_type!r}  "
              f"sample={dict(lib.sample_type_counts)}")

    # ---------------- list channels ----------------
    channels = await provider.list_live_channels(base_url, token)
    print(f"  list_live_channels: {len(channels)} returned")
    if channels:
        print("    first sample channel (raw keys):")
        sample_ch = channels[0]
        for k in sorted(sample_ch.keys()):
            v = sample_ch[k]
            if isinstance(v, (dict, list)):
                v = json.dumps(v, default=str)[:120]
            print(f"      {k}: {_trim(v, 100)}")
        print()
        print(f"    summary of first {min(sample, len(channels))} channels:")
        for ch in channels[:sample]:
            cur_prog = ch.get("CurrentProgram") or {}
            cur_name = cur_prog.get("Name") if isinstance(cur_prog, dict) else ""
            print(
                f"      #{ch.get('ChannelNumber') or '?':>6}  "
                f"id={_trim(ch.get('Id'), 18):18}  "
                f"name={_trim(ch.get('Name'), 28):28}  "
                f"type={ch.get('ChannelType') or '?':4}  "
                f"now={_trim(cur_name, 40)}"
            )

    # ---------------- playback info for first channel ----------------
    # Tells us what playback URL Emby/Jellyfin hands us for a live channel:
    # HLS? Transcoded MP4? Raw MPEG-TS? Drives our player codec wiring.
    if channels:
        first_ch = channels[0]
        first_id = str(first_ch.get("Id") or "").strip()
        if first_id:
            print()
            print(f"  fetch_playback_info({first_id}) — channel {first_ch.get('Name')!r}:")
            playback = await provider.fetch_playback_info(
                base_url, token, external_id=first_id,
            )
            if not playback:
                print("    PlaybackInfo: returned None / failed")
            else:
                play_sess = playback.get("PlaySessionId", "")
                sources = playback.get("MediaSources", []) or []
                print(f"    PlaySessionId: {play_sess}")
                print(f"    MediaSources: {len(sources)}")
                for i, src in enumerate(sources[:3]):
                    if not isinstance(src, dict):
                        continue
                    container = src.get("Container") or src.get("TranscodingContainer", "")
                    proto    = src.get("Protocol", "")
                    direct   = src.get("DirectStreamUrl") or ""
                    transc   = src.get("TranscodingUrl") or ""
                    path     = src.get("Path") or ""
                    supports_ts = src.get("SupportsTranscoding")
                    supports_ds = src.get("SupportsDirectStream")
                    print(f"    [{i}] container={container} protocol={proto}")
                    print(f"        SupportsDirectStream={supports_ds} SupportsTranscoding={supports_ts}")
                    if direct:
                        print(f"        DirectStreamUrl: {_trim(direct, 100)}")
                    if transc:
                        print(f"        TranscodingUrl:  {_trim(transc, 100)}")
                    if path:
                        print(f"        Path:            {_trim(path, 100)}")
                    streams = src.get("MediaStreams", []) or []
                    if streams:
                        print(f"        MediaStreams: {len(streams)} —", end=" ")
                        kinds = [s.get("Type", "?") for s in streams if isinstance(s, dict)]
                        codecs = [s.get("Codec", "?") for s in streams if isinstance(s, dict)]
                        print(", ".join(f"{k}({c})" for k, c in zip(kinds, codecs)))

    # ---------------- list programs (EPG) ----------------
    channel_ids = tuple(
        str(c.get("Id") or "").strip()
        for c in channels[:10] if c.get("Id")
    )
    programs = await provider.list_live_programs(
        base_url, token,
        channel_external_ids=channel_ids,
        max_results=50,
        hours_ahead=6,
    )
    print(f"  list_live_programs (next 6h, first 10 channels): {len(programs)} returned")
    if programs:
        print("    first sample program (raw keys):")
        sample_p = programs[0]
        for k in sorted(sample_p.keys()):
            v = sample_p[k]
            if isinstance(v, (dict, list)):
                v = json.dumps(v, default=str)[:120]
            print(f"      {k}: {_trim(v, 100)}")
        print()
        print(f"    upcoming sample ({min(sample, len(programs))}):")
        for p in programs[:sample]:
            print(
                f"      {p.get('StartDate', '')[:16]:16}  "
                f"ch={_trim(p.get('ChannelId'), 18):18}  "
                f"name={_trim(p.get('Name'), 36):36}  "
                f"news={p.get('IsNews') or False}  "
                f"movie={p.get('IsMovie') or False}"
            )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="/data/augmentum.db",
        help="Path to augmentum.db (default: /data/augmentum.db inside container)",
    )
    parser.add_argument("--user", default="", help="Filter to one username (e.g. shadow)")
    parser.add_argument(
        "--provider", default="",
        choices=("", "emby", "jellyfin"),
        help="Filter to one provider type",
    )
    parser.add_argument(
        "--sample", type=int, default=10,
        help="How many channels/programs to print per server (default 10)",
    )
    args = parser.parse_args()

    servers = _load_servers(
        Path(args.db), username=args.user, provider=args.provider,
    )
    if not servers:
        print(
            "No Emby/Jellyfin servers found in user_media_servers. "
            "Configure one first (Settings → Media Servers) and retry.",
            file=sys.stderr,
        )
        return 1

    print(f"Probing {len(servers)} server(s)…")
    async with httpx.AsyncClient(timeout=15.0) as http:
        for row in servers:
            try:
                await _probe_server(row, http=http, sample=args.sample)
            except Exception as exc:
                print(f"  PROBE FAILED for {row['name']!r}: {exc}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
