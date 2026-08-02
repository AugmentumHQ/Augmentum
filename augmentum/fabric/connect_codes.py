"""Short, shareable connect codes for contact cards (UX).

Turns the long signed card into a friendly code people can scan or type:
``K7P2-9QX4``. The alphabet drops visually ambiguous characters (no 0/O,
1/I/L) so a code read off a screen or said aloud doesn't get mistyped.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Crockford-ish: no 0/O/1/I/L/U — unambiguous when read aloud or typed.
_ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
_CODE_LEN = 8  # rendered as XXXX-XXXX


def _derive_code(seed: bytes) -> str:
    """Deterministic 8-char code from a seed (the card bytes + a salt).

    Deterministic so the same card maps to the same code on retry; the
    PK conflict path below regenerates with an extra salt on the rare
    collision."""
    digest = hashlib.sha256(seed).digest()
    n = int.from_bytes(digest, "big")
    out = []
    for _ in range(_CODE_LEN):
        n, rem = divmod(n, len(_ALPHABET))
        out.append(_ALPHABET[rem])
    return "".join(out)


def format_code(code: str) -> str:
    """Group a raw 8-char code as ``XXXX-XXXX`` for display."""
    c = normalize_code(code)
    return f"{c[:4]}-{c[4:]}" if len(c) == _CODE_LEN else c


def normalize_code(code: str) -> str:
    """Strip grouping/whitespace and upper-case for lookup. Tolerant of
    how a human typed it."""
    return "".join(ch for ch in (code or "").upper() if ch in _ALPHABET)


async def create_code(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    card: dict[str, Any],
    expires_at: int = 0,
) -> str:
    """Mint (or reuse) a short code for ``card``. Returns the raw code."""
    card_json = json.dumps(card, separators=(",", ":"))
    seed = card_json.encode("utf-8")
    for salt in range(5):  # handle the astronomically-rare collision
        code = _derive_code(seed + bytes([salt]))
        cur = await conn.execute(
            "SELECT card_json FROM fabric_connect_codes WHERE code=?", (code,)
        )
        row = await cur.fetchone()
        if row is None:
            await conn.execute(
                "INSERT INTO fabric_connect_codes (code, user_id, card_json, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (code, user_id, card_json, int(expires_at)),
            )
            await conn.commit()
            return code
        if row[0] == card_json:
            return code  # same card already coded — reuse
    raise RuntimeError("could not allocate a connect code")


async def resolve_code(
    conn: aiosqlite.Connection, *, code: str, now: int = 0,
) -> dict[str, Any] | None:
    """Return the card for a code, or None if unknown/expired."""
    norm = normalize_code(code)
    if not norm:
        return None
    cur = await conn.execute(
        "SELECT card_json, expires_at FROM fabric_connect_codes WHERE code=?",
        (norm,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    expires_at = int(row[1] or 0)
    if expires_at and now and now > expires_at:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None
