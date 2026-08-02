"""Live RisuRealm import audit — pulls real cards through the same
``/api/ui/fetch-url`` proxy the browser uses, then runs the same
client-side dispatch (json/binary/text → PNG / charx-zip / JSON sniff)
the UI does in ``ui/scripts/narrative/index.js``.

For each card we capture:

* wire format (json / charx-zip / png-tEXt / text-json / failed)
* spec / spec_version (TavernCard V1 / V2 / V3, RisuAI native, etc.)
* which canonical fields landed (name, description, personality,
  scenario, first_mes, mes_example, creator_notes)
* extension surfaces (visual_traits, extensions.risuai.*)

Then we print a single readable audit table and assert a minimum
success rate so regressions are loud.

Why live: we explicitly want to confirm that the three RisuRealm fixes
(forward caller headers, treat unknown content-types as binary, drop
HTML-biased Accept default) actually unblock real cards from real
RisuRealm. Stubs would tell us nothing about the wire shape we
mis-classified.

Run::

    pytest tests/live/test_live_risurealm_import.py --run-live -v -s

Skip when the Augmentum server isn't running on ``localhost:6100`` or
when RisuRealm/auth isn't reachable. ``-s`` shows the audit table.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json as _json
import struct
import zipfile
from dataclasses import dataclass, field

import httpx
import pytest

pytestmark = pytest.mark.live

BASE = "http://localhost:6100"
SAMPLE_SIZE = 8  # how many top cards to audit per run


# ---------------------------------------------------------------------------
# Auth fixture (shared pattern with test_live_auth.py)
# ---------------------------------------------------------------------------


async def _probe() -> bool:
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=5) as c:
            r = await c.get("/api/version")
            return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def auth_client():
    if not await _probe():
        pytest.skip(f"Augmentum not reachable at {BASE}")

    async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
        status = (await client.get("/api/auth/status")).json()
        if status.get("setup_required"):
            pytest.skip("Server requires setup")
        if not status.get("authenticated"):
            login = await client.post("/api/auth/login", json={
                "username": "admin", "password": "testpassword123",
            })
            if login.status_code != 200:
                pytest.skip("Cannot authenticate (no test creds on this server)")
            client.cookies = login.cookies
        yield client


# ---------------------------------------------------------------------------
# Mirror of ui/scripts/narrative/index.js helpers
# ---------------------------------------------------------------------------


def _extract_chara_from_png(data: bytes) -> dict | None:
    """Port of `extractCharaFromPng` (index.js:2919). Reads tEXt chunks,
    prefers 'ccv3' over 'chara', base64-decodes to UTF-8 JSON."""
    PNG_SIG = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(PNG_SIG):
        return None
    text_chunks: dict[str, str] = {}
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        offset += 4
        ctype = data[offset:offset + 4].decode("latin-1", errors="ignore")
        offset += 4
        if length > len(data) - offset:
            break
        if ctype == "tEXt" and length > 0:
            chunk = data[offset:offset + length]
            null_idx = chunk.find(b"\x00")
            if null_idx > 0:
                keyword = chunk[:null_idx].decode("latin-1", errors="ignore").lower()
                text_chunks[keyword] = chunk[null_idx + 1:].decode("latin-1", errors="ignore")
        offset += length + 4
        if ctype == "IEND":
            break
    raw = text_chunks.get("ccv3") or text_chunks.get("chara")
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        return _json.loads(decoded)
    except Exception:
        return None


def _extract_card_from_charx(data: bytes) -> tuple[dict | None, dict | None]:
    """Port of the new `_extractCardFromBinaryCharx` (index.js:4224).

    Returns ``(cardData, avatarInfo)`` where ``avatarInfo`` is
    ``{source, size}`` or None. Avatar source is one of: ``png_body``
    (the response IS a character PNG), ``charx_asset:<path>``
    (resolved via ``data.assets[type=icon].uri``), or
    ``charx_filename:<path>`` (filename heuristic fallback)."""
    import re as _re
    if not data:
        return None, None
    if data[0:1] == b"{":
        try:
            return _json.loads(data.decode("utf-8")), None
        except Exception:
            pass
    if data[0:2] == b"\x89P":
        chara = _extract_chara_from_png(data)
        if chara:
            return chara, {"source": "png_body", "size": len(data)}
    sig = b"PK\x03\x04"
    zip_offset = data.find(sig)
    if zip_offset < 0:
        return None, None
    try:
        zf = zipfile.ZipFile(io.BytesIO(data[zip_offset:]))
        card: dict | None = None
        for name in zf.namelist():
            if name == "card.json" or name.endswith("/card.json"):
                with zf.open(name) as fh:
                    card = _json.loads(fh.read().decode("utf-8"))
                break
        avatar = None
        if card:
            assets = (card.get("data") or {}).get("assets") or card.get("assets") or []
            icon = next((a for a in assets if a.get("type") == "icon"), None) \
                 or next((a for a in assets if a.get("name") == "main"), None)
            if icon and icon.get("uri"):
                path = _re.sub(r"^embedded?://", "", icon["uri"])
                path = _re.sub(r"^__asset:", "", path)
                names = set(zf.namelist())
                target = path if path in names else (
                    "assets/" + path if "assets/" + path in names else None
                )
                if target:
                    avatar = {"source": f"charx_asset:{target}",
                              "size": zf.getinfo(target).file_size}
        if not avatar:
            for name in zf.namelist():
                if _re.search(r"icon|avatar|main\.(png|jpe?g|webp)$", name, _re.I):
                    avatar = {"source": f"charx_filename:{name}",
                              "size": zf.getinfo(name).file_size}
                    break
        return card, avatar
    except Exception:
        return None, None


def _normalize_card_data(j: dict) -> dict | None:
    """Port of `normalizeCardData` (index.js:2983)."""
    if not isinstance(j, dict):
        return None
    if j.get("spec") and j.get("data"):
        data = j["data"]
        if isinstance(j.get("assets"), list) and "assets" not in data:
            data["assets"] = j["assets"]
        return data
    if isinstance(j.get("data"), dict) and (j["data"].get("name") or j["data"].get("char_name")):
        return j["data"]
    if "definition" in j:
        return j["definition"]
    if j.get("char_name"):
        return {
            "name": j["char_name"],
            "description": j.get("char_persona", ""),
            "personality": "",
            "scenario": j.get("world_scenario", ""),
            "first_mes": j.get("char_greeting", ""),
            "mes_example": j.get("example_dialogue", ""),
        }
    if j.get("name") or j.get("ch_name"):
        return j
    return None


# Mirrors the field mapping in `applyCardDataToCharacter` (index.js:3077),
# minus the cleanCardText HTML-strip pass (cosmetic — doesn't affect import
# success). This is what lands on the character record.
_CANONICAL_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("name",         ("name", "ch_name", "char_name")),
    ("description",  ("description", "char_persona")),
    ("personality",  ("personality", "tavern_personality")),
    ("scenario",     ("scenario", "world_scenario")),
    ("greeting",     ("first_mes", "first_message", "greeting_message", "char_greeting")),
    ("examples",     ("mes_example", "example_dialogue", "example_dialogs")),
    ("creator_notes",("creator_notes",)),
)


def _apply_canonical(data: dict) -> dict[str, str]:
    """Return {canonical_field: value} for every populated field."""
    result: dict[str, str] = {}
    for canon, sources in _CANONICAL_FIELDS:
        for src in sources:
            v = data.get(src)
            if isinstance(v, str) and v.strip():
                result[canon] = v
                break
    # RisuAI-native `extensions.risuai.additionalText` / `newGenData` (mirrors
    # the rescue path in `_importFromRisu` index.js:4193).
    risuai = (data.get("extensions") or {}).get("risuai") or {}
    if "description" not in result and isinstance(risuai.get("additionalText"), str):
        result["description"] = risuai["additionalText"]
    ngd = risuai.get("newGenData") or {}
    if isinstance(ngd, dict):
        for src, canon in (("description", "description"),
                           ("personality", "personality"),
                           ("scenario", "scenario")):
            if canon not in result and isinstance(ngd.get(src), str):
                result[canon] = ngd[src]
    return result


# ---------------------------------------------------------------------------
# Audit core
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    card_id: str
    name: str
    ok: bool = False
    wire_type: str = ""        # json / binary / text / error
    detected_format: str = ""  # tavern_v3 / tavern_v2 / risuai_native / pygmalion / charx_zip / png_text / json_text / unknown
    spec: str = ""             # raw `spec` field if present
    spec_version: str = ""     # raw `spec_version`
    fields_populated: dict[str, int] = field(default_factory=dict)
    avatar: dict | None = None  # {source, size} or None
    error: str = ""


async def _fetch_via_proxy(client: httpx.AsyncClient, url: str,
                            extra_headers: dict | None = None) -> dict:
    """POST /api/ui/fetch-url exactly like the UI does."""
    resp = await client.post("/api/ui/fetch-url", json={
        "url": url,
        "headers": extra_headers or {},
    })
    resp.raise_for_status()
    return resp.json()


def _classify_format(card_obj: dict) -> str:
    spec = (card_obj.get("spec") or "").lower()
    if spec.startswith("chara_card_v3"):
        return "tavern_v3"
    if spec.startswith("chara_card_v2") or spec == "chara_card_v2":
        return "tavern_v2"
    if isinstance(card_obj.get("data"), dict):
        if (card_obj["data"].get("extensions") or {}).get("risuai"):
            return "risuai_native"
        return "tavern_v2_nested"
    if card_obj.get("char_name"):
        return "pygmalion"
    if card_obj.get("name") or card_obj.get("ch_name"):
        return "tavern_v1_flat"
    return "unknown"


async def _import_one(client: httpx.AsyncClient, card_id: str, name: str) -> ImportResult:
    """Mirror `_importFromRisu` (index.js:4152) end-to-end."""
    out = ImportResult(card_id=card_id, name=name)
    url = f"https://realm.risuai.net/api/v1/download/dynamic/{card_id}?cors=true"
    try:
        result = await _fetch_via_proxy(client, url, {"x-risu-api-version": "4"})
    except Exception as exc:
        out.wire_type = "error"
        out.error = f"fetch-url failed: {exc}"
        return out

    rtype = result.get("type", "")
    out.wire_type = rtype
    card_obj: dict | None = None

    if rtype == "json":
        card_obj = result.get("data")
        out.detected_format = _classify_format(card_obj or {}) if isinstance(card_obj, dict) else "unknown"
    elif rtype == "binary":
        try:
            raw = base64.b64decode(result.get("data") or "")
        except Exception as exc:
            out.error = f"base64 decode failed: {exc}"
            return out
        if not raw:
            out.error = "binary payload was empty"
            return out
        card_obj, out.avatar = _extract_card_from_charx(raw)
        if out.avatar and out.avatar["source"] == "png_body":
            out.detected_format = "png_text"
        elif raw[:2] == b"PK" or b"PK\x03\x04" in raw[:64]:
            out.detected_format = "charx_zip"
        else:
            out.detected_format = "binary_sniffed" if card_obj else "binary_unrecognized"
    elif rtype == "text":
        text = result.get("data") or ""
        try:
            card_obj = _json.loads(text)
            out.detected_format = "json_text"
        except Exception as exc:
            out.error = f"text payload not JSON: {exc}; first 80 bytes: {text[:80]!r}"
            return out
    else:
        out.error = f"unexpected response type: {rtype!r}"
        return out

    if not isinstance(card_obj, dict):
        out.error = "no card object extracted from payload"
        return out

    out.spec = card_obj.get("spec") or ""
    out.spec_version = card_obj.get("spec_version") or ""
    if not out.detected_format or out.detected_format in ("binary_sniffed",):
        out.detected_format = _classify_format(card_obj)

    definition = card_obj.get("data") if isinstance(card_obj.get("data"), dict) else card_obj
    normalized = _normalize_card_data(card_obj) or definition or {}
    canon = _apply_canonical(normalized)
    out.fields_populated = {k: len(v) for k, v in canon.items() if v}
    # URL-based avatar fallback (mirrors applyCardDataToCharacter:_pendingAvatarUrl).
    if not out.avatar:
        url = (normalized.get("avatar") or normalized.get("photo")
               or normalized.get("profile_image") or normalized.get("avatar_url"))
        if isinstance(url, str) and url.startswith("http"):
            out.avatar = {"source": "url_field", "size": 0}
        else:
            assets = normalized.get("assets") or []
            icon = next((a for a in assets if a.get("type") == "icon"
                         and isinstance(a.get("uri"), str)
                         and a["uri"].startswith("http")), None)
            if icon:
                out.avatar = {"source": "v3_asset_url", "size": 0}
    out.ok = "name" in out.fields_populated  # same threshold as the UI
    if not out.ok and not out.error:
        keys = list((normalized or {}).keys())[:10]
        out.error = f"no usable name in normalized card; keys={keys}"
    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_risurealm_import_audit(auth_client):
    """Audit the top N RisuRealm cards through the real proxy and report."""
    search = await auth_client.get("/api/ui/risurealm/search",
                                    params={"q": "", "sort": "recommended", "nsfw": True})
    if search.status_code != 200:
        pytest.skip(f"risurealm search returned {search.status_code}")
    cards = (search.json() or {}).get("cards") or []
    if not cards:
        pytest.skip("RisuRealm returned no cards (upstream rate-limit or markup change)")

    sample = cards[:SAMPLE_SIZE]
    print(f"\n=== RisuRealm import audit — {len(sample)} cards ===\n")

    # Throttle to 2 concurrent fetches to be polite to RisuRealm.
    sem = asyncio.Semaphore(2)

    async def _bounded(c):
        async with sem:
            return await _import_one(auth_client, c["id"], c.get("name") or "?")

    results = await asyncio.gather(*[_bounded(c) for c in sample])

    # ------------------ Audit table ------------------
    header = f"{'#':>2} {'OK':>3} {'WIRE':<7} {'FORMAT':<14} {'AVATAR':<22} {'NAME':<24} FIELDS"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        ok = "OK " if r.ok else "FAIL"
        if r.avatar:
            sz = r.avatar.get("size") or 0
            sz_str = f"{sz//1024}KB" if sz else "url"
            avatar_col = f"{r.avatar['source'].split(':',1)[0]}/{sz_str}"[:22]
        else:
            avatar_col = "MISSING"
        name = (r.name or "")[:24]
        fields = ",".join(f"{k}({v})" for k, v in r.fields_populated.items())
        print(f"{i:>2} {ok:>3} {r.wire_type:<7} {r.detected_format:<14} {avatar_col:<22} {name:<24} {fields}")
        if r.error:
            print(f"     -> error: {r.error}")
    print()

    # ------------------ Format breakdown ------------------
    by_fmt: dict[str, int] = {}
    by_wire: dict[str, int] = {}
    for r in results:
        by_fmt[r.detected_format or "?"] = by_fmt.get(r.detected_format or "?", 0) + 1
        by_wire[r.wire_type or "?"] = by_wire.get(r.wire_type or "?", 0) + 1
    print("Wire types:  ", ", ".join(f"{k}={v}" for k, v in sorted(by_wire.items())))
    print("Card formats:", ", ".join(f"{k}={v}" for k, v in sorted(by_fmt.items())))

    # ------------------ Field landing rate ------------------
    canon_keys = [k for k, _ in _CANONICAL_FIELDS]
    total = len(results)
    print("\nField landing rate (populated / total):")
    for k in canon_keys:
        n = sum(1 for r in results if k in r.fields_populated)
        print(f"  {k:<14} {n}/{total}")

    by_av: dict[str, int] = {}
    for r in results:
        key = (r.avatar or {}).get("source", "MISSING").split(":", 1)[0]
        by_av[key] = by_av.get(key, 0) + 1
    print("\nAvatar source breakdown:")
    for k, v in sorted(by_av.items()):
        print(f"  {k:<18} {v}/{total}")

    successes = sum(1 for r in results if r.ok)
    rate = successes / total
    avs = sum(1 for r in results if r.avatar)
    av_rate = avs / total
    print(f"\nOverall: {successes}/{total} imports succeeded ({rate:.0%}); "
          f"avatars: {avs}/{total} ({av_rate:.0%})")

    # Threshold: we don't expect 100% (some cards on RisuRealm are known
    # malformed) but a regression in the proxy should flip most of them.
    # 60% is conservative — a serious break drops well below this.
    assert rate >= 0.60, (
        f"RisuRealm import success rate fell to {rate:.0%}. "
        "Check the audit table above — likely a proxy classification "
        "regression in /api/ui/fetch-url."
    )
    # Avatar threshold: most RisuRealm cards ship a PNG body or a charx
    # icon asset. A regression in the avatar-extraction path drops this
    # rate hard. URL-only cards count toward the threshold.
    assert av_rate >= 0.60, (
        f"Avatar extraction rate fell to {av_rate:.0%}. "
        "Check that _extractCardFromBinaryCharx still returns "
        "{cardData, avatarDataUrl} and that _importFromRisu applies it."
    )
    # Every successful import must at least have a name. Anything else is
    # the format-coverage frontier and is reported, not asserted.
    for r in results:
        if r.ok:
            assert "name" in r.fields_populated
