#!/usr/bin/env python3
"""Generate the next numbered migration file for Augmentum.

Usage:
    python gen_migration.py "add voice profiles table"
    → Creates augmentum/state/migrations/054_add_voice_profiles_table.sql

The file is pre-populated with a commented template.  Edit the SQL and
the migration will be picked up automatically on next startup.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "state" / "migrations").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_migration.py <description>")
        print('Example: gen_migration.py "add voice profiles table"')
        sys.exit(1)

    desc = " ".join(sys.argv[1:])
    slug = re.sub(r"[^a-z0-9]+", "_", desc.lower()).strip("_")

    root = _find_root()
    mig_dir = root / "augmentum" / "state" / "migrations"

    # Find next number
    existing = sorted(mig_dir.glob("*.sql"))
    max_num = 0
    for f in existing:
        m = re.match(r"(\d+)_", f.name)
        if m:
            max_num = max(max_num, int(m.group(1)))

    next_num = max_num + 1
    filename = f"{next_num:03d}_{slug}.sql"
    filepath = mig_dir / filename

    template = f"""\
-- {filename}
-- {desc}
--
-- Guidelines:
--   - Use IF NOT EXISTS for new tables
--   - Use ALTER TABLE for adding columns to existing tables
--   - Foreign keys to ui_sessions need get_or_create_session() called first
--   - Test: data survives server restart

-- Example: new table
-- CREATE TABLE IF NOT EXISTS my_table (
--     id TEXT PRIMARY KEY,
--     session_id TEXT NOT NULL,
--     data TEXT NOT NULL DEFAULT '{{}}'  ,
--     created_at TEXT NOT NULL DEFAULT (datetime('now'))
-- );

-- Example: add column to existing table
-- ALTER TABLE existing_table ADD COLUMN new_field TEXT NOT NULL DEFAULT '';
"""

    filepath.write_text(template, encoding="utf-8")
    print(f"Created: {filepath.relative_to(root)}")
    print(f"  Number: {next_num:03d}")
    print(f"  Edit the file and add your SQL statements.")


if __name__ == "__main__":
    main()
