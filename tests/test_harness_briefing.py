"""Harness briefing — detection, scope-isolated injection, tool-safety,
fail-open, and one-time seeding."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.memory.models import MemoryType
from augmentum.models.base import InternalChatRequest, Message
from augmentum.proxy import harness as H

# --- fakes -----------------------------------------------------------------

class _Hdrs:
    def __init__(self, d: dict): self._d = {k.lower(): v for k, v in d.items()}
    def get(self, k, default=None): return self._d.get(k.lower(), default)


class _Req:
    def __init__(self, headers: dict): self.headers = _Hdrs(headers)


class _Mem:
    def __init__(self, content: str, mid: str = "m"):
        self.content = content
        self.id = mid


def _store(*, raise_on_recall: bool = False, empty: bool = False) -> MagicMock:
    s = MagicMock()

    async def recall(**kw):
        if raise_on_recall:
            raise RuntimeError("boom")
        if empty:
            return []
        if MemoryType.PROCEDURAL in (kw.get("memory_types") or []):
            return [_Mem("Always run the tests before claiming done.")]
        return [_Mem("The rate limiter lives in middleware/throttle.py.")]

    s.recall = AsyncMock(side_effect=recall)
    s.store = AsyncMock(return_value="mem_1")
    return s


def _app(store) -> MagicMock:
    a = MagicMock()
    a.memory_store = store
    return a


def _ireq(user_msg: str = "add rate limiting to the upload endpoint", tools=None):
    return InternalChatRequest(
        model="d/Qwen3.6-27B",
        messages=[Message(role="system", content="sys"), Message(role="user", content=user_msg)],
        tools=tools,
    )


# --- detection -------------------------------------------------------------

def test_detect_header_wins():
    assert H.detect_harness(_Req({"X-Augmentum-Harness": "OpenCode"})) == "opencode"


def test_detect_user_agent_agnostic():
    assert H.detect_harness(_Req({"User-Agent": "aider/0.42"})) == "aider"
    assert H.detect_harness(_Req({"User-Agent": "Cursor-IDE/2.1"})) == "cursor"
    assert H.detect_harness(_Req({"User-Agent": "claude-cli/1.0"})) == "claude_code"


def test_detect_browser_and_empty_are_not_harness():
    assert H.detect_harness(_Req({"User-Agent": "Mozilla/5.0 (Windows) Chrome/120"})) == ""
    assert H.detect_harness(_Req({})) == ""


def test_budget_lines_respects_cap():
    lines = ["a" * 40, "b" * 40, "c" * 40]  # ~10 tokens each
    assert H._budget_lines(lines, 15) == ["a" * 40]  # 2nd would exceed 15


# --- injection -------------------------------------------------------------

@pytest.mark.asyncio
async def test_inject_adds_block_keeps_tools_and_original_message():
    store = _store()
    req = _ireq(tools=[{"type": "function", "function": {"name": "read_file"}}])
    orig_tools = req.tools
    ok = await H.inject_harness_context(req, _app(store), user_id="u1", harness="opencode")
    assert ok is True
    last_user = [m for m in req.messages if m.role == "user"][-1]
    assert "Augmentum memory" in last_user.content          # block injected
    assert "add rate limiting" in last_user.content          # original preserved
    assert "Always run the tests" in last_user.content       # procedural surfaced
    assert "rate limiter lives in middleware" in last_user.content  # fact surfaced
    assert req.tools is orig_tools                           # tools untouched


@pytest.mark.asyncio
async def test_injection_is_scope_isolated():
    store = _store()
    await H.inject_harness_context(req := _ireq(), _app(store), user_id="u1", harness="opencode")
    assert req.messages  # sanity
    # EVERY recall (seed-check + procedural + facts) must be STRICTLY scoped to
    # the harness family (global seeds or a harness:* project scope) — not even
    # unscoped/universal memories may surface here (C1 fix).
    assert store.recall.await_count >= 2
    for call in store.recall.call_args_list:
        scope = call.kwargs.get("scope")
        assert scope == H.HARNESS_SCOPE or scope.startswith(H.HARNESS_SCOPE + ":")
        assert call.kwargs.get("scope_strict") is True


@pytest.mark.asyncio
async def test_injection_reads_project_scope():
    """Facts are read from the {harness}:{project} scope only; conventions merge
    global seeds + the project scope. No other project's scope is touched."""
    store = _store()
    await H.inject_harness_context(
        _ireq(), _app(store), user_id="u1", harness="claude_code", project="augmentum",
    )
    scopes = [c.kwargs.get("scope") for c in store.recall.call_args_list]
    assert "harness:claude_code:augmentum" in scopes
    assert all(
        s in ("harness", "harness:claude_code:augmentum") for s in scopes
    )


@pytest.mark.asyncio
async def test_conventions_first_turn_only(monkeypatch):
    """harness_conventions_mode="first_turn" (default): the conventions block is
    injected on the session's first turn, facts-only once an assistant message
    exists in the transcript."""
    monkeypatch.setattr(H.settings, "harness_conventions_mode", "first_turn", raising=False)
    store = _store()
    req = _ireq()  # no assistant message → first turn
    await H.inject_harness_context(req, _app(store), user_id="u1", harness="opencode")
    assert "Working conventions" in [m for m in req.messages if m.role == "user"][-1].content

    store2 = _store()
    req2 = _ireq()
    req2.messages.insert(1, Message(role="assistant", content="done"))
    await H.inject_harness_context(req2, _app(store2), user_id="u1", harness="opencode")
    content = [m for m in req2.messages if m.role == "user"][-1].content
    assert "Working conventions" not in content
    assert "Relevant project memory" in content

    # "always" restores per-turn conventions.
    monkeypatch.setattr(H.settings, "harness_conventions_mode", "always", raising=False)
    store3 = _store()
    req3 = _ireq()
    req3.messages.insert(1, Message(role="assistant", content="done"))
    await H.inject_harness_context(req3, _app(store3), user_id="u1", harness="opencode")
    assert "Working conventions" in [m for m in req3.messages if m.role == "user"][-1].content


@pytest.mark.asyncio
async def test_fail_open_on_recall_error():
    store = _store(raise_on_recall=True)
    req = _ireq()
    before = req.messages[-1].content
    ok = await H.inject_harness_context(req, _app(store), user_id="u1", harness="opencode")
    assert ok is False
    assert req.messages[-1].content == before  # request unchanged


@pytest.mark.asyncio
async def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(H.settings, "harness_enrich_enabled", False, raising=False)
    req = _ireq()
    before = req.messages[-1].content
    ok = await H.inject_harness_context(req, _app(_store()), user_id="u1", harness="opencode")
    assert ok is False
    assert req.messages[-1].content == before


@pytest.mark.asyncio
async def test_anon_user_noop():
    req = _ireq()
    ok = await H.inject_harness_context(req, _app(_store()), user_id="", harness="opencode")
    assert ok is False


# --- seeding ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_seed_when_empty_then_idempotent():
    s = _store(empty=True)
    await H.ensure_harness_seed(_app(s), "u1")
    assert s.store.await_count == len(H._DEFAULT_CONVENTIONS)
    for call in s.store.call_args_list:
        assert call.args[1] == MemoryType.PROCEDURAL
        assert call.kwargs.get("scope") == H.HARNESS_SCOPE

    # Now non-empty → no re-seed.
    s2 = _store()  # recall returns existing procedural
    await H.ensure_harness_seed(_app(s2), "u1")
    assert s2.store.await_count == 0


# --- capture (learn-out) ---------------------------------------------------

def test_parse_memories_extracts_drops_secrets_and_reads_supersedes():
    raw = (
        '```json\n{"memories":['
        '{"kind":"fact","text":"Deploy via ./deploy.sh prod","durable":true,"supersedes":0},'
        '{"kind":"fact","text":"the key is sk-abcdefghijklmnopqrstuvwxyz","durable":true}'
        ']}\n```'
    )
    out = H._parse_memories(raw)
    assert len(out) == 1  # secret-bearing candidate dropped
    assert out[0]["text"] == "Deploy via ./deploy.sh prod"  # command preserved
    assert out[0]["supersedes"] == 0  # reconcile id parsed


def test_parse_memories_malformed_is_empty():
    assert H._parse_memories("not json at all") == []
    assert H._parse_memories("") == []


@pytest.mark.asyncio
async def test_capture_stages_candidates_without_mutating_memory(monkeypatch):
    """Harness turns STAGE harvest candidates; they must NOT write/supersede live
    memory (no auto-accumulation — the baseline only grows on a deliberate pass)."""
    async def fake_extract(_app, _msg, _existing):
        return [
            {"kind": "fact", "text": "Deploy via ./start.sh restart augmentum", "durable": True, "supersedes": None},
            {"kind": "convention", "text": "Never roll your own rate limiter", "durable": True, "supersedes": None},
            {"kind": "fact", "text": "loosely uses middleware", "durable": False, "supersedes": None},
        ]

    staged = {}

    def fake_stage(**kw):
        staged.update(kw)
        return "hh_test"

    monkeypatch.setattr(H, "_llm_extract", fake_extract)
    monkeypatch.setattr(H, "capture_harness_observation", fake_stage)
    store = MagicMock()
    store.recall = AsyncMock(return_value=[])      # no baseline to flag against
    store.store = AsyncMock(return_value="m")
    store.supersede = AsyncMock(return_value="m")
    await H._capture(
        _app(store), "u1", "remember: deploy via ./start.sh restart",
        harness="opencode", model="d/Qwen3.6-27B",
    )
    # NOTHING written to live memory
    store.store.assert_not_awaited()
    store.supersede.assert_not_awaited()
    # ...but a harvest record WAS staged, with all 3 candidates + provenance
    assert staged["user_id"] == "u1"
    assert staged["harness"] == "opencode"
    cands = staged["candidates"]
    assert len(cands) == 3
    by_text = {c["text"]: c for c in cands}
    assert by_text["Never roll your own rate limiter"]["kind"] == "convention"
    assert by_text["Deploy via ./start.sh restart augmentum"]["durable"] is True
    assert by_text["loosely uses middleware"]["durable"] is False


@pytest.mark.asyncio
async def test_capture_flags_baseline_contradiction_without_superseding(monkeypatch):
    """A candidate that would CHANGE the baseline is FLAGGED on the staged record
    (read-only) — it must not actually supersede anything."""
    async def fake_extract(_app, _msg, existing):
        assert existing == [(0, "Deploy via ./start.sh restart augmentum")]  # int-id context
        return [{"kind": "fact", "text": "Deploy via ./deploy.sh prod", "durable": True, "supersedes": 0}]

    staged = {}

    def fake_stage(**kw):
        staged.update(kw)
        return "hh_test"

    monkeypatch.setattr(H, "_llm_extract", fake_extract)
    monkeypatch.setattr(H, "capture_harness_observation", fake_stage)
    store = MagicMock()
    store.recall = AsyncMock(return_value=[_Mem("Deploy via ./start.sh restart augmentum", "old-uuid")])
    store.store = AsyncMock(return_value="new")
    store.supersede = AsyncMock(return_value="new")
    await H._capture(_app(store), "u1", "correction: deploy via ./deploy.sh prod now")
    # no mutation at all
    store.supersede.assert_not_awaited()
    store.store.assert_not_awaited()
    # the staged candidate carries the read-only baseline-contradiction flag
    cand = staged["candidates"][0]
    assert cand["text"] == "Deploy via ./deploy.sh prod"
    assert cand["supersedes_baseline_id"] == "old-uuid"
    assert cand["supersedes_baseline_text"] == "Deploy via ./start.sh restart augmentum"


@pytest.mark.asyncio
async def test_capture_skips_without_teaching_signal(monkeypatch):
    calls = {"n": 0}

    async def fake_extract(_a, _m, _e):
        calls["n"] += 1
        return []

    monkeypatch.setattr(H, "_llm_extract", fake_extract)
    store = MagicMock()
    store.recall = AsyncMock(return_value=[])
    store.store = AsyncMock()
    await H._capture(_app(store), "u1", "fix the bug in upload.py")  # no teaching signal
    assert calls["n"] == 0  # LLM never invoked
    store.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_skips_pure_question(monkeypatch):
    """A READ must never write/mutate memory. 'How do we deploy?' trips the broad
    teaching prefilter ('we deploy') but is a question — capture must skip it
    (regression: a deploy *question* once superseded the real deploy memory)."""
    calls = {"n": 0}

    async def fake_extract(_a, _m, _e):
        calls["n"] += 1
        return []

    monkeypatch.setattr(H, "_llm_extract", fake_extract)
    store = MagicMock()
    store.recall = AsyncMock(return_value=[])
    store.store = AsyncMock()
    store.supersede = AsyncMock()
    await H._capture(_app(store), "u1", "How do we deploy Augmentum?")
    assert calls["n"] == 0          # never reached the LLM
    store.store.assert_not_awaited()
    store.supersede.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_fail_open(monkeypatch):
    async def boom(_a, _m, _e):
        raise RuntimeError("extract exploded")

    monkeypatch.setattr(H, "_llm_extract", boom)
    store = MagicMock()
    store.recall = AsyncMock(return_value=[])
    store.store = AsyncMock()
    await H._capture(_app(store), "u1", "remember to always run tests")  # must not raise
    store.store.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_capture_fires_on_last_user_message(monkeypatch):
    seen = {}

    async def fake(_app, uid, msg, **_k):
        seen["uid"], seen["msg"] = uid, msg

    monkeypatch.setattr(H, "_capture", fake)
    req = _ireq(user_msg="remember: always use get_user")
    H.schedule_harness_capture(req, _app(_store()), user_id="u1")
    await asyncio.sleep(0)  # let the fire-and-forget task run
    assert seen == {"uid": "u1", "msg": "remember: always use get_user"}


@pytest.mark.asyncio
async def test_schedule_capture_disabled(monkeypatch):
    monkeypatch.setattr(H.settings, "harness_capture_enabled", False, raising=False)
    fired = {}

    async def fake(*_a, **_k):
        fired["x"] = 1

    monkeypatch.setattr(H, "_capture", fake)
    H.schedule_harness_capture(_ireq(), _app(_store()), user_id="u1")
    await asyncio.sleep(0)
    assert "x" not in fired


# --- harvest staging + promotion (Brick 3) --------------------------------

@pytest.mark.asyncio
async def test_harvest_stage_read_promote_dismiss(monkeypatch, tmp_path):
    """Stage candidates → read them as pending → promote one into the baseline →
    dismiss the other. Promotion is the ONLY baseline write, and it's gated here."""
    from augmentum.training import capture as C

    monkeypatch.setattr(C, "_harness_harvest_dir", lambda: tmp_path)

    obs_id = C.capture_harness_observation(
        user_id="u1", harness="opencode", model="m",
        source_message="remember: use ruff; port is 6100",
        candidates=[
            {"kind": "convention", "text": "Use ruff", "durable": True,
             "supersedes_baseline_id": None, "supersedes_baseline_text": None},
            {"kind": "fact", "text": "Port is 6100", "durable": False,
             "supersedes_baseline_id": None, "supersedes_baseline_text": None},
        ],
    )
    assert obs_id
    recs = C.read_harness_harvest(user_id="u1")
    assert len(recs) == 1
    assert recs[0]["pending_count"] == 2
    assert recs[0]["candidates"][0]["status"] == "pending"

    # promote candidate 0 — writes to the baseline (store mocked)
    store = MagicMock()
    store.store = AsyncMock(return_value="base-1")
    store.supersede = AsyncMock()
    res = await H.promote_candidate(_app(store), "u1", obs_id, 0)
    assert res["status"] == "promoted"
    assert res["baseline_id"] == "base-1"
    store.store.assert_awaited_once()
    # Legacy record (no target_scope) → falls back to the shared default
    # project scope, never the seeds-only global scope.
    assert store.store.call_args.kwargs.get("scope") == "harness:default"
    assert store.store.call_args.kwargs.get("scope_strict") is True
    assert store.store.call_args.args[1] == MemoryType.PROCEDURAL

    # ledger now reflects: 0 promoted, 1 pending
    recs2 = C.read_harness_harvest(user_id="u1", include_harvested=True)
    cands = recs2[0]["candidates"]
    assert cands[0]["status"] == "promote" and cands[0]["baseline_id"] == "base-1"
    assert cands[1]["status"] == "pending"

    # dismiss candidate 1 → record fully decided, drops from the pending view
    H.dismiss_candidate("u1", obs_id, 1)
    assert C.read_harness_harvest(user_id="u1") == []
    tr = C.harness_harvest_trends(user_id="u1")
    assert tr["promoted"] == 1 and tr["dismissed"] == 1 and tr["pending"] == 0


@pytest.mark.asyncio
async def test_promote_supersedes_when_flagged(monkeypatch, tmp_path):
    """A candidate flagged as contradicting the baseline supersedes that entry
    (invalidate-not-delete) instead of storing fresh."""
    from augmentum.training import capture as C

    monkeypatch.setattr(C, "_harness_harvest_dir", lambda: tmp_path)
    obs_id = C.capture_harness_observation(
        user_id="u1", harness="claude_code", model="m", source_message="x",
        candidates=[{
            "kind": "fact", "text": "Deploy via ./deploy.sh", "durable": True,
            "supersedes_baseline_id": "old-1",
            "supersedes_baseline_text": "Deploy via ./start.sh",
        }],
    )
    store = MagicMock()
    store.store = AsyncMock()
    store.supersede = AsyncMock(return_value="new-1")
    res = await H.promote_candidate(_app(store), "u1", obs_id, 0)
    assert res["status"] == "promoted" and res["superseded"] is True
    store.supersede.assert_awaited_once()
    assert store.supersede.call_args.args[0] == "old-1"
    store.store.assert_not_awaited()


# --- isolation regression gate (C1) ---------------------------------------

@pytest.mark.asyncio
async def test_scope_strict_isolation_gate():
    """C1 regression gate: a STRICT-scope (harness) operation must NEVER reach an
    unscoped/universal memory. Uses _text_dedup_check (pure SQL — deterministic,
    no embedder) as the proxy for the scope_strict filter shared by recall and
    the dedup/supersede write path. This is the test that should have caught the
    personal-facts-in-the-coding-briefing leak."""
    from augmentum.memory.store import MemoryStore
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    try:
        store = MemoryStore(backend)
        await backend.conn.execute(
            "INSERT INTO memories (id, user_id, content, memory_type, scope) VALUES (?,?,?,?,?)",
            ("m_unscoped", "u1", "lives in rochester ny name is matt", "fact", None),
        )
        await backend.conn.execute(
            "INSERT INTO memories (id, user_id, content, memory_type, scope) VALUES (?,?,?,?,?)",
            ("m_harness", "u1", "deploy augmentum via deploy.sh prod", "fact", "harness"),
        )
        await backend.conn.commit()

        # STRICT harness must NOT reach the unscoped/personal memory (the leak).
        assert await store._text_dedup_check(
            "lives in rochester ny name is matt", "u1", scope="harness", scope_strict=True,
        ) is None
        # Non-strict default DOES (the universal-share behavior other scopes use).
        assert await store._text_dedup_check(
            "lives in rochester ny name is matt", "u1", scope="harness", scope_strict=False,
        ) == "m_unscoped"
        # Strict harness still sees its OWN scope.
        assert await store._text_dedup_check(
            "deploy augmentum via deploy.sh prod", "u1", scope="harness", scope_strict=True,
        ) == "m_harness"
    finally:
        await backend.close()
