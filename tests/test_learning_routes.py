"""learning_routes integration tests.

Wires the real `learning_router` against a fresh aiosqlite vocab_state
table, a real PackManager pointing at a freshly-built tiny JP pack, and
an in-memory settings stub. No app-wide migration machinery; the schema
mirrors migration 145 directly.
"""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from augmentum.knowledge.lang_pack_builder import build_pack
from augmentum.knowledge.packs import PackManager
from augmentum.learning import lang_packs
from augmentum.state.vocab_store import VocabStore

# Schema mirrors migration 145 (no FK on users — keeps the fixture lean).
_MINI_SCHEMA = """
CREATE TABLE vocab_state (
    user_id          TEXT NOT NULL,
    lang_code        TEXT NOT NULL,
    word_id          TEXT NOT NULL,
    fsrs_difficulty  REAL NOT NULL DEFAULT 5.0,
    fsrs_stability   REAL NOT NULL DEFAULT 0.0,
    fsrs_due_at      TEXT NOT NULL,
    fsrs_reps        INTEGER NOT NULL DEFAULT 0,
    fsrs_lapses      INTEGER NOT NULL DEFAULT 0,
    fsrs_last_grade  INTEGER,
    mastery_state    TEXT NOT NULL DEFAULT 'new',
    first_seen_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_reviewed_at TEXT,
    source_surface   TEXT NOT NULL,
    source_ref       TEXT,
    exposure_input   INTEGER NOT NULL DEFAULT 0,
    exposure_output  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, lang_code, word_id)
);
"""


class _DictSettingsStore:
    """Minimal in-memory get_user/set_user for routes that only need those."""

    def __init__(self):
        self._d: dict[tuple[str, str], str | None] = {}

    async def get_user(self, user_id: str, key: str):
        return self._d.get((user_id, key))

    async def set_user(self, user_id: str, key: str, value):
        self._d[(user_id, key)] = value


class _FakeJobsStore:
    """Just enough of JobsStore for the /packs install endpoints."""

    def __init__(self):
        self.jobs: list[dict] = []
        self._n = 0

    async def create(self, *, user_id, job_type, payload=None, priority=0, max_attempts=3):
        self._n += 1
        job_id = f"job_{self._n}"
        self.jobs.append({
            "id": job_id, "user_id": user_id, "job_type": job_type,
            "payload": payload or {}, "status": "pending",
        })
        return job_id

    async def list_for_user(self, *, user_id, status=None, job_type=None, limit=100):
        return [
            j for j in self.jobs
            if j["user_id"] == user_id
            and (job_type is None or j["job_type"] == job_type)
            and (status is None or j["status"] == status)
        ][:limit]


_JMDICT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [
<!ENTITY v1 "Ichidan verb">
<!ENTITY vt "transitive verb">
<!ENTITY n "noun">
]>
<JMdict>
<entry>
<ent_seq>1358280</ent_seq>
<k_ele><keb>食べる</keb></k_ele>
<r_ele><reb>たべる</reb></r_ele>
<sense><pos>&v1;</pos><pos>&vt;</pos><gloss>to eat</gloss></sense>
</entry>
<entry>
<ent_seq>1578850</ent_seq>
<k_ele><keb>朝ごはん</keb></k_ele>
<r_ele><reb>あさごはん</reb></r_ele>
<sense><pos>&n;</pos><gloss>breakfast</gloss></sense>
</entry>
</JMdict>
"""
_SENT_TSV = "1\tjpn\t彼は朝ごはんを食べる。\n2\teng\tHe eats breakfast.\n"
_LINK_TSV = "1\t2\n2\t1\n"


@pytest.fixture
def learning_client(app, tmp_path):
    """TestClient with vocab_store + settings_store + a tiny JP pack loaded."""
    pack_dir = tmp_path / "packs"
    pack_dir.mkdir()
    (tmp_path / "JMdict_e").write_text(_JMDICT_XML, encoding="utf-8")
    (tmp_path / "sentences.tsv").write_text(_SENT_TSV, encoding="utf-8")
    (tmp_path / "links.tsv").write_text(_LINK_TSV, encoding="utf-8")
    build_pack(
        out_path=pack_dir / "ja.augpack",
        lang_code="ja",
        jmdict_xml=tmp_path / "JMdict_e",
        tatoeba_sentences=tmp_path / "sentences.tsv",
        tatoeba_links=tmp_path / "links.tsv",
    )

    loop = asyncio.get_event_loop()

    async def _setup():
        c = await aiosqlite.connect(":memory:")
        await c.executescript(_MINI_SCHEMA)
        pm = PackManager(pack_dir)
        await pm.scan()
        return c, pm

    conn, pack_mgr = loop.run_until_complete(_setup())

    app.state.pack_manager = pack_mgr
    app.state.vocab_store = VocabStore(conn)
    app.state.settings_store = _DictSettingsStore()
    app.state.jobs_store = _FakeJobsStore()

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc

    async def _teardown():
        await pack_mgr.close()
        await conn.close()

    loop.run_until_complete(_teardown())


# ── /state ───────────────────────────────────────────────────────────


def test_get_state_default_off(learning_client):
    r = learning_client.get("/api/learning/state")
    assert r.status_code == 200
    j = r.json()
    assert j["toggle"] == "off"
    assert j["native_lang"] == ""
    assert j["target_langs"] == []
    assert j["levels"] == {}
    assert len(j["packs"]) == 1
    assert j["packs"][0]["lang_code"] == "ja"
    assert j["packs"][0]["vocab_count"] == 2


def test_post_state_persists_profile(learning_client):
    r = learning_client.post("/api/learning/state", json={
        "toggle": "on",
        "native_lang": "en",
        "target_langs": ["ja"],
        "levels": {"ja": "basics"},
    })
    assert r.status_code == 200
    j = r.json()
    assert j["toggle"] == "on"
    assert j["native_lang"] == "en"
    assert j["target_langs"] == ["ja"]
    assert j["levels"] == {"ja": "basics"}

    # Persists across a separate request.
    j2 = learning_client.get("/api/learning/state").json()
    assert j2["toggle"] == "on"
    assert j2["levels"] == {"ja": "basics"}


def test_post_state_partial_patch(learning_client):
    learning_client.post("/api/learning/state", json={"native_lang": "en"})
    learning_client.post("/api/learning/state", json={"toggle": "on"})
    j = learning_client.get("/api/learning/state").json()
    assert j["native_lang"] == "en"
    assert j["toggle"] == "on"


def test_post_state_invalid_toggle(learning_client):
    r = learning_client.post("/api/learning/state", json={"toggle": "maybe"})
    assert r.status_code == 400


def test_state_tts_voice(learning_client):
    # Default is empty (off) — the onboarding modal picks a real voice for
    # the user's selected target lang(s). Picking a JP voice by default
    # would silently mispronounce Spanish for a Spanish learner.
    assert learning_client.get("/api/learning/state").json()["tts_voice"] == ""
    # Settable + persisted.
    r = learning_client.post("/api/learning/state", json={"tts_voice": "jm_kumo"})
    assert r.status_code == 200 and r.json()["tts_voice"] == "jm_kumo"
    assert learning_client.get("/api/learning/state").json()["tts_voice"] == "jm_kumo"
    # "off" allowed; blank normalises to "off".
    assert learning_client.post("/api/learning/state", json={"tts_voice": "off"}).json()["tts_voice"] == "off"
    assert learning_client.post("/api/learning/state", json={"tts_voice": "   "}).json()["tts_voice"] == "off"


def test_state_includes_pos_labels_by_lang(learning_client):
    """/state ships the POS code → label map per installed pack so the UI
    can render grammatical info without a hardcoded per-language JS map.
    Packs predating the meta field fall back to JMDICT_POS_LABELS for ja."""
    j = learning_client.get("/api/learning/state").json()
    assert "pos_labels_by_lang" in j
    assert isinstance(j["pos_labels_by_lang"], dict)
    # Fixture installs a ja pack; we must at least know how to label
    # JMdict's core verb codes (v1, v5) — either from pack meta or the
    # built-in fallback table.
    ja_labels = j["pos_labels_by_lang"].get("ja", {})
    assert ja_labels.get("v1") == "Ichidan verb"
    assert ja_labels.get("vt") == "transitive verb"


# ── /lookup ──────────────────────────────────────────────────────────


def test_lookup_longest_prefix_picks_verb(learning_client):
    r = learning_client.get(
        "/api/learning/lookup", params={"lang": "ja", "q": "食べるのが好き", "pos": 0}
    )
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert [e["word_id"] for e in entries] == ["1358280"]
    assert entries[0]["surface"] == "食べる"
    assert entries[0]["glosses"] == ["to eat"]


def test_lookup_freetext_english_gloss(learning_client):
    r = learning_client.get("/api/learning/lookup", params={"lang": "ja", "q": "breakfast"})
    assert r.status_code == 200
    ids = {e["word_id"] for e in r.json()["entries"]}
    assert "1578850" in ids


def test_lookup_unknown_lang_404(learning_client):
    r = learning_client.get("/api/learning/lookup", params={"lang": "xx", "q": "x"})
    assert r.status_code == 404


def test_lookup_empty_query(learning_client):
    r = learning_client.get("/api/learning/lookup", params={"lang": "ja", "q": ""})
    assert r.status_code == 200
    assert r.json()["entries"] == []


# ── /vocab/add ───────────────────────────────────────────────────────


def test_add_vocab_idempotent(learning_client):
    r = learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    assert r.status_code == 200
    assert r.json()["added"] is True

    r2 = learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    assert r2.status_code == 200
    assert r2.json()["added"] is False


def test_add_vocab_unknown_word_404(learning_client):
    r = learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "0000000", "source_surface": "browse",
    })
    assert r.status_code == 404


# ── /srs/due ─────────────────────────────────────────────────────────


def test_srs_due_empty(learning_client):
    r = learning_client.get("/api/learning/srs/due", params={"lang": "ja"})
    assert r.status_code == 200
    assert r.json() == {"due": [], "total": 0}


def test_srs_due_fresh_word_not_yet_due(learning_client):
    learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    j = learning_client.get("/api/learning/srs/due", params={"lang": "ja"}).json()
    assert j["total"] == 0   # fresh-add is due tomorrow, not now


def test_srs_due_returns_enriched_card(learning_client):
    learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    # Backdate so the card counts as due now. The route doesn't expose a
    # "now=" cutoff (production never needs one); we reach through to the
    # same connection the store wraps.
    conn = learning_client.app.state.vocab_store._conn
    loop = asyncio.get_event_loop()
    loop.run_until_complete(conn.execute(
        "UPDATE vocab_state SET fsrs_due_at='2000-01-01 00:00:00' WHERE word_id=?",
        ("1358280",),
    ))
    loop.run_until_complete(conn.commit())

    j = learning_client.get("/api/learning/srs/due", params={"lang": "ja"}).json()
    assert j["total"] == 1
    assert len(j["due"]) == 1
    card = j["due"][0]
    assert card["word_id"] == "1358280"
    assert card["surface"] == "食べる"
    assert card["reading"] == "たべる"
    assert card["glosses"] == ["to eat"]
    assert card["example"] is not None
    assert "食べる" in card["example"]["lang_text"]
    # Predicted next-interval per grade (JSON keys are strings).
    iv = card["preview_intervals"]
    assert set(iv.keys()) == {"1", "2", "3", "4"}
    assert all(int(v) >= 1 for v in iv.values())
    assert int(iv["1"]) <= int(iv["2"]) <= int(iv["3"]) <= int(iv["4"])


# ── /srs/grade ───────────────────────────────────────────────────────


# ?? /games ??????????????????????????????????????????????????????????


def test_games_pool_does_not_backfill_practice_without_discovery(learning_client):
    r = learning_client.get(
        "/api/learning/games/pool",
        params={"lang": "ja", "mode": "drill", "count": 4},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["pool"] == []
    assert j["owned_count"] == 0
    assert j["discovery_count"] == 0

    r2 = learning_client.get(
        "/api/learning/games/pool",
        params={"lang": "ja", "mode": "drill", "count": 4, "allow_discovery": True},
    )
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["pool"]
    assert j2["allow_discovery"] is True
    assert j2["discovery_count"] == len(j2["pool"])
    assert all(c["in_queue"] is False for c in j2["pool"])


def test_games_pool_hides_single_latin_letter_cards(learning_client, monkeypatch):
    original_get_entry = lang_packs.get_entry

    async def fake_get_entry(conn, word_id):
        if word_id == "letter-a":
            return {
                "word_id": "letter-a",
                "surface": "a",
                "reading": "a",
                "pos": "letter",
                "glosses": ["letter a"],
            }
        return await original_get_entry(conn, word_id)

    monkeypatch.setattr(lang_packs, "get_entry", fake_get_entry)

    async def _seed_queue():
        await learning_client.app.state.vocab_store.seed_words(
            user_id="usr_test",
            lang_code="ja",
            word_ids=["letter-a", "1358280"],
        )

    asyncio.get_event_loop().run_until_complete(_seed_queue())

    r = learning_client.get(
        "/api/learning/games/pool",
        params={"lang": "ja", "mode": "garden", "count": 10},
    )
    assert r.status_code == 200
    ids = [c["word_id"] for c in r.json()["pool"]]
    assert "1358280" in ids
    assert "letter-a" not in ids


def test_games_readiness_and_owned_pool_track_real_queue(learning_client):
    learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })

    ready = learning_client.get("/api/learning/games/readiness", params={"lang": "ja"})
    assert ready.status_code == 200
    rj = ready.json()
    assert rj["total"] == 1
    assert rj["due"] == 0
    assert rj["counts"]["new"] == 1
    assert rj["sentences"]["translated_easy"] == 1
    assert rj["path"]["level_system"] == "jlpt"
    assert "games" in rj
    assert rj["games"]["word_garden"]["ready"] is True
    assert rj["games"]["bubble_pop"]["ready"] is False
    assert rj["games"]["mirror"]["ready"] is False

    pool = learning_client.get(
        "/api/learning/games/pool",
        params={"lang": "ja", "mode": "drill", "count": 4},
    )
    assert pool.status_code == 200
    pj = pool.json()
    assert [c["word_id"] for c in pj["pool"]] == ["1358280"]
    assert pj["discovery_count"] == 0
    assert pj["pool"][0]["in_queue"] is True


def test_srs_grade_advances_fsrs_state(learning_client):
    learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    r = learning_client.post("/api/learning/srs/grade", json={
        "lang": "ja", "word_id": "1358280", "grade": 3,
    })
    assert r.status_code == 200
    j = r.json()
    assert j["mastery_state"] in ("learning", "mature")
    assert j["fsrs_due_at"]
    assert j["interval_days"] >= 1
    # JSON object keys are strings.
    ni = j["next_intervals"]
    assert set(ni.keys()) == {"1", "2", "3", "4"}
    assert int(ni["1"]) <= int(ni["2"]) <= int(ni["3"]) <= int(ni["4"])


def test_srs_grade_again_records_lapse_after_recall(learning_client):
    """Add → grade Good (so reps=1) → grade Again → lapses should increment."""
    learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    learning_client.post("/api/learning/srs/grade", json={
        "lang": "ja", "word_id": "1358280", "grade": 3,
    })
    learning_client.post("/api/learning/srs/grade", json={
        "lang": "ja", "word_id": "1358280", "grade": 1,
    })
    # Verify via direct read.
    conn = learning_client.app.state.vocab_store._conn

    async def _read():
        cur = await conn.execute(
            "SELECT fsrs_reps, fsrs_lapses FROM vocab_state WHERE word_id=?",
            ("1358280",),
        )
        return await cur.fetchone()

    reps, lapses = asyncio.get_event_loop().run_until_complete(_read())
    assert reps == 2
    assert lapses == 1


def test_srs_grade_unknown_word_404(learning_client):
    r = learning_client.post("/api/learning/srs/grade", json={
        "lang": "ja", "word_id": "0000000", "grade": 3,
    })
    assert r.status_code == 404


def test_srs_grade_invalid_grade_400(learning_client):
    learning_client.post("/api/learning/vocab/add", json={
        "lang": "ja", "word_id": "1358280", "source_surface": "browse",
    })
    r = learning_client.post("/api/learning/srs/grade", json={
        "lang": "ja", "word_id": "1358280", "grade": 9,
    })
    assert r.status_code == 400


# ── /packs (catalog + install) ───────────────────────────────────────


def test_packs_catalog_lists_all_with_status(learning_client):
    j = learning_client.get("/api/learning/packs/catalog").json()
    by_code = {p["lang_code"]: p for p in j["packs"]}
    assert set(by_code) == {"ja", "es", "zh", "fr", "ko"}
    # The fixture installed a ja pack.
    assert by_code["ja"]["installed"] is True
    assert by_code["ja"]["installable"] is False  # already installed
    assert by_code["ja"]["status"] == "available"
    for code in ("es", "zh", "fr", "ko"):
        assert by_code[code]["status"] == "available"
        assert by_code[code]["installed"] is False
        assert by_code[code]["installable"] is True
    assert by_code["ja"]["approx_download_mb"] > 0
    assert isinstance(by_code["ja"]["sources"], list) and by_code["ja"]["sources"]


def test_install_already_installed_409(learning_client):
    r = learning_client.post("/api/learning/packs/ja/install")
    assert r.status_code == 409


def test_install_available_noninstalled_lang_enqueues_job(learning_client):
    r = learning_client.post("/api/learning/packs/zh/install")
    assert r.status_code == 200
    j = r.json()
    assert j["lang"] == "zh"
    assert j["job_id"].startswith("job_")
    assert j["status"] == "pending"


def test_install_unknown_lang_404(learning_client):
    r = learning_client.post("/api/learning/packs/klingon/install")
    assert r.status_code == 404


def test_seed_pack_walks_curated_path_no_freq_fallback(learning_client):
    # JA ships a curated path (ja.json). The seeder walks path units in
    # order and resolves surfaces against the pack — for langs with a
    # path it MUST NOT fall through to top_frequency (which would seed
    # raw articles/pronouns and re-create the failure mode this fix
    # removed for Spanish learners).
    #
    # The tiny test pack has 食べる + 朝ごはん. Only 食べる appears in
    # the JA path; 朝ごはん is dictionary-only. So path-walk yields one
    # word_id and surfaces source="path_partial" (path exists but
    # couldn't supply the requested n=30) — NOT source="frequency".
    r = learning_client.post("/api/learning/packs/ja/seed", json={"count": 30})
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "path_partial"
    assert j["seeded"] == 1
    assert j["due"] == 1
    due = learning_client.get("/api/learning/srs/due", params={"lang": "ja"}).json()
    assert {c["word_id"] for c in due["due"]} == {"1358280"}
    # Idempotent.
    r2 = learning_client.post("/api/learning/packs/ja/seed", json={"count": 30})
    assert r2.json()["seeded"] == 0
    assert r2.json()["due"] == 1


def test_seed_pack_unknown_lang_404(learning_client):
    r = learning_client.post("/api/learning/packs/xx/seed", json={"count": 5})
    assert r.status_code == 404


def test_read_pack_returns_sentences(learning_client):
    r = learning_client.get("/api/learning/read/ja", params={"count": 5})
    assert r.status_code == 200
    j = r.json()
    assert "sentences" in j
    for s in j["sentences"]:
        assert "sent_id" in s and "lang_text" in s
    # q-filter restricts to sentences containing that text.
    r2 = learning_client.get("/api/learning/read/ja", params={"count": 5, "q": "食べる"})
    assert r2.status_code == 200
    for s in r2.json()["sentences"]:
        assert "食べる" in s["lang_text"]


def test_read_pack_unknown_lang_404(learning_client):
    r = learning_client.get("/api/learning/read/xx")
    assert r.status_code == 404


def test_breakdown_tokenizes_span(learning_client):
    r = learning_client.get("/api/learning/breakdown/ja", params={"q": "朝ごはんを食べる"})
    assert r.status_code == 200
    j = r.json()
    assert j["text"] == "朝ごはんを食べる"
    matched = [t for t in j["tokens"] if t.get("matched")]
    assert {t["word_id"] for t in matched} == {"1578850", "1358280"}
    assert any(t["text"] == "を" and not t.get("matched") for t in j["tokens"])
    assert learning_client.get("/api/learning/breakdown/ja", params={"q": ""}).json() == {"text": "", "tokens": []}


def test_breakdown_unknown_lang_404(learning_client):
    assert learning_client.get("/api/learning/breakdown/xx", params={"q": "x"}).status_code == 404


def test_install_enqueues_job_when_not_installed(app, tmp_path):
    """Separate from `learning_client` (which pre-installs ja): point the
    pack manager at an empty dir so install is allowed."""
    empty_dir = tmp_path / "empty_packs"
    empty_dir.mkdir()
    loop = asyncio.get_event_loop()
    pm = PackManager(empty_dir)
    loop.run_until_complete(pm.scan())
    app.state.pack_manager = pm
    app.state.jobs_store = _FakeJobsStore()
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})

    r = tc.post("/api/learning/packs/ja/install")
    assert r.status_code == 200
    j = r.json()
    assert j["lang"] == "ja"
    assert j["job_id"].startswith("job_")
    assert j["status"] == "pending"

    # Second call coalesces onto the same pending job.
    r2 = tc.post("/api/learning/packs/ja/install")
    assert r2.status_code == 200
    assert r2.json()["job_id"] == j["job_id"]

    # Catalog reflects the in-flight install.
    cat = {p["lang_code"]: p for p in tc.get("/api/learning/packs/catalog").json()["packs"]}
    assert cat["ja"]["install_job_id"] == j["job_id"]
    assert cat["ja"]["installed"] is False

    loop.run_until_complete(pm.close())


# ── unauthorized ─────────────────────────────────────────────────────


def test_unauthorized_state_get(learning_client):
    tc = TestClient(learning_client.app)
    # No Authorization header — auth middleware rejects with 401.
    r = tc.get("/api/learning/state")
    assert r.status_code in (401, 403)
