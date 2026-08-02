"""Tests for KVSessionManifest — schema, restore-skip gate, self-recovery."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from augmentum.models.kv_session_manifest import KVSessionManifest

# Snapshot of the pre-T2-2 CREATE TABLE statement. Hand-copied from the
# git history at commit 7eaba7a^ — locking it into the test means any
# future schema-narrowing change (dropping a column from the canonical
# schema) won't silently break compatibility with users on older builds.
_PRE_T2_2_SCHEMA_SQL = """
CREATE TABLE kv_sessions (
    model_key TEXT NOT NULL,
    session_key TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT '',
    slot_dir TEXT NOT NULL DEFAULT '',
    slot_filename TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    model_path TEXT NOT NULL DEFAULT '',
    model_mtime REAL NOT NULL DEFAULT 0,
    ctx_size INTEGER NOT NULL DEFAULT 0,
    kv_cache_type TEXT NOT NULL DEFAULT '',
    template_fingerprint TEXT NOT NULL DEFAULT '',
    system_prompt_hash TEXT NOT NULL DEFAULT '',
    prompt_fingerprint TEXT NOT NULL DEFAULT '',
    prompt_message_count INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL NOT NULL DEFAULT 0,
    last_saved REAL NOT NULL DEFAULT 0,
    expires_at REAL NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    last_restore_result TEXT NOT NULL DEFAULT '',
    last_skip_reason TEXT NOT NULL DEFAULT '',
    flash_attn INTEGER NOT NULL DEFAULT 0,
    gpu_layers INTEGER NOT NULL DEFAULT 0,
    gpu_layers_mode TEXT NOT NULL DEFAULT '',
    batch_size INTEGER NOT NULL DEFAULT 0,
    draft_model TEXT NOT NULL DEFAULT '',
    draft_max INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model_key, session_key)
)
"""

# Insert one realistic pre-T2-2 row directly via sql so the migration
# test starts from "we have a real user's old manifest".
_PRE_T2_2_INSERT_SQL = """
INSERT INTO kv_sessions (
    model_key, session_key, mode, slot_dir, slot_filename,
    model_id, model_path, model_mtime, ctx_size, kv_cache_type,
    template_fingerprint, system_prompt_hash, prompt_fingerprint,
    prompt_message_count, last_accessed, last_saved, expires_at,
    pinned, last_restore_result, last_skip_reason,
    flash_attn, gpu_layers, gpu_layers_mode, batch_size,
    draft_model, draft_max
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(manifest: KVSessionManifest, session_key: str = "sess-1") -> None:
    """Write a representative row so corruption tests have something to lose."""
    manifest.record_save(
        model_key="test-model",
        session_key=session_key,
        mode="narrative",
        slot_dir="/slots/test-model",
        slot_filename=f"slot_{session_key}.bin",
        model_id="test-model",
        model_path="/models/test-model.gguf",
        model_mtime=111.0,
        ctx_size=8192,
        kv_cache_type="q8_0",
        template_fingerprint="tpl",
        system_prompt_hash="sys",
        prompt_fingerprint="prompt",
        prompt_message_count=4,
        ttl_days=2.0,
        pinned=False,
        flash_attn=True,
        gpu_layers=32,
        gpu_layers_mode="auto",
        batch_size=512,
        draft_model="",
        draft_max=0,
        n_embed=4096,
        n_layers_total=32,
        n_heads_kv=8,
    )


# ---------------------------------------------------------------------------
# Self-recovery
# ---------------------------------------------------------------------------


class TestSelfRecovery:
    """Corrupted manifest must back up + rebuild rather than disable warm-resume.

    Tests the recovery MECHANISM directly rather than triggering real
    file-system corruption — Windows holds WAL/SHM sidecar locks even
    after sqlite connections close, making in-process corruption tests
    flaky. Injecting recoverable sqlite errors via monkey-patch
    deterministically exercises the same code path.
    """

    def test_is_recoverable_error_matches_known_strings(self):
        """Lock the recoverable-error matcher against the sqlite messages
        we expect to encounter on real-world corruption / I/O failure.
        """
        match = KVSessionManifest._is_recoverable_error
        assert match(sqlite3.DatabaseError("file is not a database"))
        assert match(sqlite3.DatabaseError("database disk image is malformed"))
        assert match(sqlite3.OperationalError("disk i/o error"))
        assert match(sqlite3.OperationalError("unable to open database file"))
        assert match(sqlite3.OperationalError("database or disk is full"))
        # Non-sqlite errors don't match.
        assert not match(ValueError("oops"))
        # sqlite errors that AREN'T in the recoverable list (programming
        # bugs, schema mismatches from a real bug) propagate normally.
        assert not match(sqlite3.OperationalError("near 'SELEC': syntax error"))

    def test_reset_db_creates_backup_and_rebuilds(self, tmp_path: Path):
        """``_reset_db`` renames the corrupted file with a timestamped
        suffix and reinitializes the schema in place. After it returns
        True the manifest is fresh — old data is gone but the manifest
        is writeable again.
        """
        db_path = tmp_path / "manifest.db"
        manifest = KVSessionManifest(str(db_path))
        _seed(manifest, "doomed-row")
        assert manifest.get_session("test-model", "doomed-row") is not None

        # Drive the reset directly with a synthetic recoverable error.
        # In production this would fire from inside _handle_error after
        # an actual sqlite raise.
        success = manifest._reset_db(
            "test_trigger",
            sqlite3.DatabaseError("file is not a database"),
        )

        assert success is True
        assert manifest._healthy is True
        # Original row is gone (rebuild lost data — that's the price).
        assert manifest.get_session("test-model", "doomed-row") is None

        # A timestamped backup of the pre-reset DB lives alongside.
        backups = list(tmp_path.glob("manifest_corrupt_*.db"))
        assert len(backups) == 1, (
            f"expected one corruption backup; got {[p.name for p in backups]}"
        )

        # And we can write to the rebuilt manifest.
        _seed(manifest, "after-reset")
        recovered = manifest.get_session("test-model", "after-reset")
        assert recovered is not None
        assert recovered["session_key"] == "after-reset"

    def test_recoverable_sqlite_error_invokes_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """End-to-end: a recoverable error from sqlite triggers reset
        and the next call works.

        Patches ``_connect`` to raise once with a recoverable message;
        the manifest's error handler should detect it, call
        ``_reset_db``, and the second connection (during init's
        rebuild) should succeed normally.
        """
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed(manifest)

        real_connect = manifest._connect
        call_count = {"n": 0}

        def flaky_connect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise sqlite3.DatabaseError("disk i/o error")
            return real_connect()

        monkeypatch.setattr(manifest, "_connect", flaky_connect)

        # First read trips the flaky connect → recovery fires.
        result = manifest.get_session("test-model", "sess-1")

        # flaky_connect was called at least twice: once to fail, once
        # (via _reset_db → _init_db) to succeed.
        assert call_count["n"] >= 2
        assert manifest._healthy is True
        # Returned None safely (the original row is in the corrupted
        # backup; the rebuilt DB is empty).
        assert result is None
        # Backup file exists with the right naming.
        backups = list(tmp_path.glob("manifest_corrupt_*.db"))
        assert len(backups) == 1

    def test_recovery_failure_marks_unhealthy_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """If even the rebuild fails, mark unhealthy so subsequent calls no-op.

        Belt-and-suspenders: a truly broken environment (read-only
        disk, file permissions clobbered) shouldn't crash the
        backend's request loop. Each subsequent call returns the
        safe-default and silently no-ops.
        """
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        _seed(manifest)

        # Sabotage rebuild by patching _init_db to fail.
        def _broken_init():
            raise sqlite3.OperationalError("simulated read-only filesystem")

        monkeypatch.setattr(manifest, "_init_db", _broken_init)

        # Force a recoverable error on the next read.
        def _broken_connect():
            raise sqlite3.DatabaseError("file is not a database")

        monkeypatch.setattr(manifest, "_connect", _broken_connect)

        # First read after corruption: triggers reset → init fails →
        # marks unhealthy. Returns None safely.
        assert manifest.get_session("test-model", "sess-1") is None
        assert manifest._healthy is False

        # Subsequent calls short-circuit on the unhealthy gate without
        # even attempting sqlite — they return safe defaults.
        assert manifest.list_sessions("/slots/test-model") == []
        assert manifest.list_model_sessions("test-model") == []
        # Mutators no-op silently (no raise).
        manifest.set_pinned("sess-1", True)
        manifest.delete_session("test-model", "sess-1")

    def test_unhealthy_short_circuits_before_sqlite_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """``_healthy = False`` short-circuits ALL public methods.

        Verifies the gate is at the entry of every public method, not
        deep in the SQL path. Saves us from another shell-out to a
        broken DB on every retry attempt.
        """
        manifest = KVSessionManifest(str(tmp_path / "manifest.db"))
        manifest._healthy = False

        # Patch _connect to record any attempt to open the DB. Pure
        # short-circuit guarantee: zero connect calls when unhealthy.
        connect_calls = {"n": 0}

        def counting_connect():
            connect_calls["n"] += 1
            raise AssertionError("connect should not be called when unhealthy")

        monkeypatch.setattr(manifest, "_connect", counting_connect)

        assert manifest.get_session("m", "s") is None
        assert manifest.list_sessions("/slots/m") == []
        assert manifest.list_model_sessions("m") == []
        assert manifest.list_expired_sessions("/slots/m") == []
        manifest.record_save(
            model_key="m", session_key="s", mode="", slot_dir="", slot_filename="",
            model_id="", model_path="", model_mtime=0.0, ctx_size=0,
            kv_cache_type="", template_fingerprint="", system_prompt_hash="",
            prompt_fingerprint="", prompt_message_count=0,
            ttl_days=0.0, pinned=False,
        )
        manifest.touch_session(model_key="m", session_key="s", ttl_days=0.0)
        manifest.mark_restore_skip("m", "s", "reason")
        manifest.set_pinned("s", True)
        manifest.delete_session("m", "s")

        assert connect_calls["n"] == 0, (
            "unhealthy gate failed: at least one method tried to connect"
        )


# ---------------------------------------------------------------------------
# Schema migration (T2-2 — n_embed / n_layers_total / n_heads_kv)
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """Existing manifests from before T2-2 must accept the new columns
    via ALTER TABLE on first open, and existing rows must survive the
    migration intact.

    Without this test, a user running an older build who upgrades would
    get a silent ``no such column: n_embed`` on every record_save — the
    sqlite error path swallows it via _handle_error, but warm-resume
    would mysteriously stop working until the next manifest reset.
    """

    def _columns(self, db_path: Path) -> set[str]:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("PRAGMA table_info(kv_sessions)")
            return {row[1] for row in cursor.fetchall()}

    def test_pre_t2_2_db_gets_new_columns_added(self, tmp_path: Path):
        """Open an older-shape DB → new columns appear after init."""
        db_path = tmp_path / "manifest.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_PRE_T2_2_SCHEMA_SQL)

        # Sanity: the seeded DB starts WITHOUT the T2-2 columns.
        before = self._columns(db_path)
        assert "n_embed" not in before
        assert "n_layers_total" not in before
        assert "n_heads_kv" not in before

        # Opening the manifest runs _init_db → ALTER TABLE for each
        # missing migration column.
        KVSessionManifest(str(db_path))

        after = self._columns(db_path)
        assert "n_embed" in after
        assert "n_layers_total" in after
        assert "n_heads_kv" in after

    def test_existing_rows_survive_migration(self, tmp_path: Path):
        """A pre-T2-2 row stays readable + its old columns are preserved."""
        db_path = tmp_path / "manifest.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_PRE_T2_2_SCHEMA_SQL)
            conn.execute(
                _PRE_T2_2_INSERT_SQL,
                (
                    "old-model", "old-sess", "narrative",
                    "/slots/old-model", "slot_old.bin",
                    "old-model", "/models/old.gguf", 100.0, 4096, "q8_0",
                    "tpl-old", "sys-old", "prompt-old",
                    3, 1000.0, 1000.0, 0.0,
                    0, "saved", "",
                    1, 24, "auto", 256,
                    "", 0,
                ),
            )

        manifest = KVSessionManifest(str(db_path))
        row = manifest.get_session("old-model", "old-sess")
        assert row is not None
        assert row["model_key"] == "old-model"
        assert row["session_key"] == "old-sess"
        assert row["ctx_size"] == 4096
        assert row["gpu_layers"] == 24
        # New columns exist and default to 0 for the legacy row.
        assert row["n_embed"] == 0
        assert row["n_layers_total"] == 0
        assert row["n_heads_kv"] == 0

    def test_migrated_db_accepts_new_record_save(self, tmp_path: Path):
        """After migration, record_save with new fields persists them."""
        db_path = tmp_path / "manifest.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_PRE_T2_2_SCHEMA_SQL)

        manifest = KVSessionManifest(str(db_path))
        _seed(manifest, "post-migration")

        row = manifest.get_session("test-model", "post-migration")
        assert row is not None
        assert row["n_embed"] == 4096
        assert row["n_layers_total"] == 32
        assert row["n_heads_kv"] == 8
        assert manifest._healthy is True

    def test_migration_is_idempotent(self, tmp_path: Path):
        """Reopening a migrated DB doesn't crash on duplicate-column ALTER.

        sqlite raises ``OperationalError: duplicate column`` on
        ALTER TABLE for an existing column. The migration loop must
        swallow that — otherwise every restart after the first
        upgrade would mark the manifest unhealthy.
        """
        db_path = tmp_path / "manifest.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_PRE_T2_2_SCHEMA_SQL)

        # First open: runs the ALTER TABLEs.
        m1 = KVSessionManifest(str(db_path))
        assert m1._healthy is True

        # Second open: ALTER TABLE on already-present columns must not
        # raise. If the swallow logic regresses, the constructor will
        # propagate sqlite3.OperationalError.
        m2 = KVSessionManifest(str(db_path))
        assert m2._healthy is True

        # Schema should still match what we expect.
        cols = self._columns(db_path)
        for col in ("n_embed", "n_layers_total", "n_heads_kv",
                    "flash_attn", "gpu_layers", "draft_model"):
            assert col in cols, f"missing column after re-open: {col}"

    def test_canonical_schema_columns_match_migration_list(self):
        """Belt-and-suspenders: every _MIGRATION_COLUMNS entry must
        also be present in the canonical CREATE TABLE.

        If someone adds a migration column but forgets to add it to
        the canonical schema, fresh DBs (no existing manifest) would
        be missing the column entirely — the ALTER TABLE only runs
        when a pre-existing table is present.

        The CREATE TABLE statement uses ``IF NOT EXISTS`` so the path
        we exercise here also runs on cold-start; this guarantees the
        two stay in lock-step.
        """
        # Read the source for the canonical CREATE TABLE so we don't
        # have to introspect a live DB.
        from augmentum.models import kv_session_manifest as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")

        # Find the CREATE TABLE block. It's a single multi-line
        # triple-quoted string in _init_db.
        create_start = src.index("CREATE TABLE IF NOT EXISTS kv_sessions")
        create_end = src.index(")", create_start)
        create_block = src[create_start:create_end]

        for col_name, _ in KVSessionManifest._MIGRATION_COLUMNS:
            assert col_name in create_block, (
                f"_MIGRATION_COLUMNS contains {col_name!r} but the "
                "canonical CREATE TABLE doesn't — fresh DBs would be "
                "missing this column"
            )
