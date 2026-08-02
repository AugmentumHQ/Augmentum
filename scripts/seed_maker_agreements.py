#!/usr/bin/env python
"""Seed a user's Working Agreements (mig 273) from a JSON file.

The ``maker_agreements`` table ships EMPTY — each user accrues their own.
This optional utility loads a starter set so the feature is useful on day
one. Idempotent: the store dedup-reinforces, so re-running is safe and
just firms existing agreements.

Goes through ``SQLiteBackend`` (not a raw sqlite3 connect) so WAL +
pragmas are applied consistently — the project rule for the main DB.

Usage:
    python scripts/seed_maker_agreements.py --user <user_id>
    python scripts/seed_maker_agreements.py --user u1 --db /data/augmentum.db
    python scripts/seed_maker_agreements.py --user u1 --file my_agreements.json

The default ``--file`` is ``scripts/maker_agreements.seed.json`` (a small
set of universal good-practice agreements). Point ``--file`` at your own
JSON to seed personal ones — a list of objects with at least
``principle`` and optionally ``rationale`` / ``category`` / ``source``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running from anywhere in the repo.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


async def _seed(*, user_id: str, db_path: str, file_path: Path) -> int:
    from augmentum.coder.maker_agreements import MakerAgreements
    from augmentum.state.backends.sqlite import SQLiteBackend

    entries = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"{file_path}: expected a JSON list of agreement objects")

    backend = SQLiteBackend(db_path)
    await backend.connect()
    try:
        store = MakerAgreements(backend.conn)
        added = 0
        for e in entries:
            principle = (e.get("principle") or "").strip()
            if not principle:
                continue
            await store.add(
                principle=principle,
                rationale=(e.get("rationale") or ""),
                category=(e.get("category") or "general"),
                source=(e.get("source") or "imported"),
                user_id=user_id,
            )
            added += 1
        active = await store.list_active(user_id=user_id, limit=500)
        print(
            f"Seeded {added} agreement(s) for user '{user_id}'. "
            f"Now holding {len(active)} active."
        )
        for a in active:
            print(f"  [{a.category}] {a.principle}")
    finally:
        await backend.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed a user's Working Agreements.")
    ap.add_argument("--user", required=True, help="user_id to seed")
    ap.add_argument(
        "--db", default="data/augmentum.db",
        help="path to augmentum.db (default: data/augmentum.db)",
    )
    ap.add_argument(
        "--file", default=str(_REPO_ROOT / "scripts" / "maker_agreements.seed.json"),
        help="JSON file of agreements to load",
    )
    args = ap.parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"seed file not found: {file_path}")
    return asyncio.run(_seed(user_id=args.user, db_path=args.db, file_path=file_path))


if __name__ == "__main__":
    raise SystemExit(main())
