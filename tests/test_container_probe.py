"""Tests for container_probe — sidecar classification, stats parsing, and the
never-raise/bounded Docker call wrapper (spec §4.5/§4.6, Phase 0 slice 1)."""

from __future__ import annotations

import asyncio

import pytest

from augmentum.resource import container_probe as cp


@pytest.fixture(autouse=True)
def _reset_probe_module_state():
    """Reset container_probe's module-global caches before AND after each test.

    The probe caches results + last-known/VRAM by container id in module
    globals; without this, a test that populates them leaks the fake sidecar
    rows into other files' /status tests (aiodocker isn't installed in CI, so
    those fall back to the cached list).
    """
    def _clear():
        cp._cache["at"] = 0.0
        cp._cache["data"] = []
        cp._last_by_cid.clear()
        cp._vram_by_cid.clear()
    _clear()
    yield
    _clear()


# ── _classify: pattern wins, hard-infra overrides, siblings always surface ──

class TestClassify:
    def test_classifier_sibling_surfaces(self):
        # The classifier is a llama-server; it must NOT be dropped.
        assert cp._classify("augmentum-classifier-1") == ("llm", "Classifier")

    def test_vision_sibling_surfaces(self):
        assert cp._classify("augmentum-vision-1") == ("llm", "Vision sibling")

    def test_tts_and_stt_sidecars(self):
        assert cp._classify("augmentum-sesame-csm-1") == ("tts", "Sesame CSM")
        assert cp._classify("augmentum-speaches-1") == ("stt", "Speaches STT")

    def test_hard_infra_is_skipped(self):
        for name in ("augmentum-caddy-1", "augmentum-searxng-1",
                     "augmentum-executor-1", "augmentum-augmentum-1",
                     "augmentum-game-stream-abc", "ws-relay-1"):
            assert cp._classify(name) is None, name

    def test_primary_engines_are_not_sidecars(self):
        # Engine tokens match no sidecar pattern → dropped, but they must not
        # block siblings (the reason they were removed from the skip list).
        assert cp._classify("augmentum-ollama-1") is None
        assert cp._classify("augmentum-llamacpp-1") is None

    def test_hard_infra_overrides_a_coincidental_substring(self):
        # A real infra container that happens to contain a pattern substring
        # ("ocr" inside a hypothetical name) must still be skipped when it also
        # matches hard infra.
        assert cp._classify("ws-ocr-bridge") is None


# ── _parse_stats: working-set RAM + CPU% from cpu/precpu delta ──────────────

class TestParseStats:
    def _stats(self, usage, inactive, cur_total, pre_total, cur_sys, pre_sys, online):
        return {
            "memory_stats": {"usage": usage, "stats": {"inactive_file": inactive}},
            "cpu_stats": {
                "cpu_usage": {"total_usage": cur_total},
                "system_cpu_usage": cur_sys,
                "online_cpus": online,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": pre_total},
                "system_cpu_usage": pre_sys,
            },
        }

    def test_working_set_subtracts_inactive_file(self):
        # usage 600 MiB, inactive_file 100 MiB → working-set 500 MiB.
        raw = self._stats(600 * 1024 * 1024, 100 * 1024 * 1024,
                          2000, 1000, 20000, 10000, 4)
        ram_mb, cpu_pct = cp._parse_stats(raw)
        assert ram_mb == 500
        # cpu_delta=1000, sys_delta=10000, online=4 → 0.1*4*100 = 40.0%
        assert cpu_pct == 40.0

    def test_accepts_single_element_list(self):
        raw = self._stats(100 * 1024 * 1024, 0, 1, 1, 1, 1, 1)
        ram_mb, _ = cp._parse_stats([raw])
        assert ram_mb == 100

    def test_zeroed_precpu_yields_unknown_cpu_not_bogus_zero(self):
        # First read with no prior sample (deltas not positive) → cpu unknown.
        raw = self._stats(100 * 1024 * 1024, 0, 1000, 1000, 10000, 10000, 4)
        ram_mb, cpu_pct = cp._parse_stats(raw)
        assert ram_mb == 100
        assert cpu_pct is None

    def test_garbage_payload_degrades(self):
        assert cp._parse_stats(None) == (None, None)
        assert cp._parse_stats({}) == (None, None)
        assert cp._parse_stats([]) == (None, None)

    def test_missing_usage_is_none_not_zero(self):
        ram_mb, _ = cp._parse_stats({"memory_stats": {}})
        assert ram_mb is None


# ── _bounded: never raises, times out to the default ───────────────────────

class TestBounded:
    def test_returns_value_on_success(self):
        async def _go():
            async def ok():
                return 42
            return await cp._bounded(ok())
        assert asyncio.run(_go()) == 42

    def test_returns_default_on_raise(self):
        async def _go():
            async def boom():
                raise RuntimeError("docker wedged")
            return await cp._bounded(boom(), default="fallback")
        assert asyncio.run(_go()) == "fallback"

    def test_returns_default_on_timeout(self, monkeypatch):
        # Shrink the deadline so the test is fast.
        monkeypatch.setattr(cp, "_DOCKER_CALL_TIMEOUT_S", 0.05)

        async def _go():
            async def slow():
                await asyncio.sleep(5)
                return "never"
            return await cp._bounded(slow(), default="timed_out")
        assert asyncio.run(_go()) == "timed_out"


# ── last-known carry-forward: a transient stats timeout must not blink the row ─

class _FakeContainer:
    def __init__(self, cid, name, stats_seq, running=True, gpu=True,
                 log_lines=None, started="2026-06-19T12:00:00.000000Z", labels=None):
        self.id = cid
        self._name = name
        self._stats_seq = list(stats_seq)
        self._running = running
        self._gpu = gpu
        self._log_lines = log_lines or []
        self._started = started
        self._labels = labels or {}

    async def show(self):
        return {
            "Name": "/" + self._name,
            "State": {"Running": self._running, "StartedAt": self._started},
            "HostConfig": {"DeviceRequests": [{}] if self._gpu else []},
            "Config": {"Labels": self._labels},
        }

    async def stats(self, stream=False):
        v = self._stats_seq.pop(0) if self._stats_seq else None
        if v is None:
            raise RuntimeError("stats timeout")
        return v

    async def log(self, *, stdout=False, stderr=False, tail=None,
                  since=None, until=None):
        return list(self._log_lines)


class _FakeContainers:
    def __init__(self, conts):
        self._c = conts

    async def list(self, all=False):  # noqa: A002 — mirror aiodocker signature
        return self._c


class _FakeClient:
    def __init__(self, conts):
        self.containers = _FakeContainers(conts)

    async def close(self):
        pass


def _good_stats(usage_mib=500, cur=2000, pre=1000, cur_sys=20000, pre_sys=10000, online=4):
    return {
        "memory_stats": {"usage": usage_mib * 1024 * 1024, "stats": {"inactive_file": 0}},
        "cpu_stats": {
            "cpu_usage": {"total_usage": cur},
            "system_cpu_usage": cur_sys,
            "online_cpus": online,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": pre},
            "system_cpu_usage": pre_sys,
        },
    }


class TestLastKnownCarryForward:
    def _reset(self):
        cp._cache["at"] = 0.0
        cp._cache["data"] = []
        cp._last_by_cid.clear()

    def test_stats_timeout_keeps_row_and_carries_value(self, monkeypatch):
        self._reset()
        cont = _FakeContainer("cid1", "augmentum-sesame-csm-1",
                              [_good_stats(500), None])  # good, then timeout
        client = _FakeClient([cont])
        monkeypatch.setattr(cp, "_get_docker", lambda s: (client, True))

        e1 = asyncio.run(cp.probe_sidecar_containers(None))
        assert len(e1) == 1
        assert e1[0]["ram_mb"] == 500
        assert e1[0]["confidence"] == "measured"

        cp._cache["at"] = 0.0  # bust the 6s TTL to force a 2nd live probe
        e2 = asyncio.run(cp.probe_sidecar_containers(None))
        assert len(e2) == 1, "row must not disappear on a transient stats timeout"
        assert e2[0]["ram_mb"] == 500, "value carried forward, not blinked to 0"
        assert e2[0]["confidence"] == "stale"

    def test_vanished_container_is_pruned(self, monkeypatch):
        self._reset()
        cont = _FakeContainer("cid1", "augmentum-sesame-csm-1", [_good_stats(500)])
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        asyncio.run(cp.probe_sidecar_containers(None))
        assert "cid1" in cp._last_by_cid

        # Container genuinely gone from the list → pruned, row drops.
        cp._cache["at"] = 0.0
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([]), True))
        e2 = asyncio.run(cp.probe_sidecar_containers(None))
        assert e2 == []
        assert "cid1" not in cp._last_by_cid


# ── sibling VRAM: parse the llama-server log banner ────────────────────────

_BANNER = [
    "load_tensors: CUDA0 model buffer size =  4096.00 MiB",
    "load_tensors:   CPU_Mapped model buffer size =   512.00 MiB",
    "llama_kv_cache_unified: CUDA0 KV buffer size =  1024.00 MiB",
    "llama_context: CUDA0 compute buffer size =   300.00 MiB",
    "some unrelated log line that should be ignored",
]


class TestParseLlamaMemory:
    def test_sums_vram_and_ram_by_scope(self):
        from augmentum.models.llama_server_manager import parse_llama_memory_from_lines
        vram, ram = parse_llama_memory_from_lines(_BANNER)
        # CUDA0: 4096 (model) + 1024 (KV) + 300 (compute) = 5420 VRAM
        assert vram == 5420
        # CPU_Mapped: 512 → RAM
        assert ram == 512

    def test_no_banner_yields_zero(self):
        from augmentum.models.llama_server_manager import parse_llama_memory_from_lines
        assert parse_llama_memory_from_lines(["nothing here", ""]) == (0, 0)


class TestSiblingVram:
    def _reset(self):
        cp._cache["at"] = 0.0
        cp._cache["data"] = []
        cp._last_by_cid.clear()
        cp._vram_by_cid.clear()

    def test_classifier_sibling_gets_measured_vram_from_logs(self, monkeypatch):
        self._reset()
        cont = _FakeContainer("cid-cls", "augmentum-classifier-1",
                              [_good_stats(600)], log_lines=_BANNER)
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        entries = asyncio.run(cp.probe_sidecar_containers(None))
        assert len(entries) == 1
        e = entries[0]
        assert e["name"] == "Classifier"
        assert e["vram_mb"] == 5420   # measured from the banner
        assert e["ram_mb"] == 600     # measured from container stats
        assert "cid-cls" in cp._vram_by_cid

    def test_vram_parsed_once_then_cached(self, monkeypatch):
        self._reset()
        calls = {"n": 0}

        class _CountingContainer(_FakeContainer):
            async def log(self, **kw):
                calls["n"] += 1
                return list(self._log_lines)

        cont = _CountingContainer("cid-cls", "augmentum-classifier-1",
                                  [_good_stats(600), _good_stats(600)], log_lines=_BANNER)
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        asyncio.run(cp.probe_sidecar_containers(None))
        cp._cache["at"] = 0.0
        asyncio.run(cp.probe_sidecar_containers(None))
        # Banner parsed on first sight only; 2nd probe reuses the cached VRAM.
        assert calls["n"] == 1

    def test_non_llm_sidecar_has_no_vram(self, monkeypatch):
        self._reset()
        cont = _FakeContainer("cid-stt", "augmentum-speaches-1",
                              [_good_stats(200)], gpu=False, log_lines=_BANNER)
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        entries = asyncio.run(cp.probe_sidecar_containers(None))
        assert entries[0]["vram_mb"] == 0   # STT sidecar: no banner parse
        assert entries[0]["ram_mb"] == 200


# ── _classify_container: ephemeral (label) vs sidecar (pattern) ────────────

class TestClassifyContainer:
    def test_coder_workspace_is_ephemeral_noncontrollable(self):
        r = cp._classify_container("augmentum-ws-abc123", {"augmentum.workspace": "true"})
        assert r is not None
        kind, subsystem, label, controllable = r
        assert kind == "ephemeral"
        assert subsystem == "coder"
        assert controllable is False
        assert "Coder" in label

    def test_game_stream_label_beats_skip_substring(self):
        # The name contains "game-stream" (a skip token) but the label wins.
        r = cp._classify_container("ws-game-stream-xyz", {"augmentum.game_stream": "true"})
        assert r is not None
        assert r[0] == "ephemeral" and r[1] == "game" and r[3] is False

    def test_sidecar_is_controllable(self):
        r = cp._classify_container("augmentum-sesame-csm-1", {})
        assert r == ("sidecar", "tts", "Sesame CSM", True)

    def test_untracked_returns_none(self):
        assert cp._classify_container("augmentum-caddy-1", {}) is None


class TestEphemeralProbe:
    def _reset(self):
        cp._cache["at"] = 0.0
        cp._cache["data"] = []
        cp._last_by_cid.clear()
        cp._vram_by_cid.clear()

    def test_running_workspace_surfaces_with_measured_ram(self, monkeypatch):
        self._reset()
        cont = _FakeContainer("cid-ws", "augmentum-ws-abc", [_good_stats(700)],
                              gpu=False, labels={"augmentum.workspace": "true"})
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        entries = asyncio.run(cp.probe_sidecar_containers(None))
        assert len(entries) == 1
        e = entries[0]
        assert e["kind"] == "ephemeral"
        assert e["controllable"] is False
        assert e["ram_mb"] == 700

    def test_stopped_ephemeral_is_hidden(self, monkeypatch):
        self._reset()
        cont = _FakeContainer("cid-ws", "augmentum-ws-dead", [],
                              running=False, labels={"augmentum.workspace": "true"})
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        entries = asyncio.run(cp.probe_sidecar_containers(None))
        assert entries == []


# ── cache_only read contract: /status must never run the live probe inline ──

class TestCacheOnlyReadPath:
    def _reset(self):
        cp._cache["at"] = 0.0
        cp._cache["data"] = []
        cp._last_by_cid.clear()
        cp._probe_lock = None

    def test_cache_only_serves_stale_without_probing(self, monkeypatch):
        # Populate the cache, then expire its TTL. A cache_only read must
        # return the stale data WITHOUT touching Docker (the ~2s slow_request
        # this fix removes). _get_docker raising proves no live probe ran.
        self._reset()
        cp._cache["data"] = [{"name": "Classifier", "ram_mb": 123}]
        cp._cache["at"] = 0.0  # already stale (TTL elapsed)

        def _boom(_s):
            raise AssertionError("read path must not open a Docker client")
        monkeypatch.setattr(cp, "_get_docker", _boom)

        out = asyncio.run(cp.probe_sidecar_containers(None, cache_only=True))
        assert out == [{"name": "Classifier", "ram_mb": 123}]

    def test_cache_only_cold_start_falls_through_to_live_probe(self, monkeypatch):
        # Nothing cached yet → cache_only must still do ONE live probe so the
        # first poll after boot isn't blank.
        self._reset()
        cont = _FakeContainer("cid1", "augmentum-sesame-csm-1", [_good_stats(400)])
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        out = asyncio.run(cp.probe_sidecar_containers(None, cache_only=True))
        assert len(out) == 1 and out[0]["ram_mb"] == 400

    def test_sampler_path_refreshes_live(self, monkeypatch):
        # cache_only=False (the background sampler) always does the live work
        # when the cache is stale, even if data is present.
        self._reset()
        cp._cache["data"] = [{"name": "old", "ram_mb": 1}]
        cp._cache["at"] = 0.0
        cont = _FakeContainer("cid1", "augmentum-sesame-csm-1", [_good_stats(900)])
        monkeypatch.setattr(cp, "_get_docker", lambda s: (_FakeClient([cont]), True))
        out = asyncio.run(cp.probe_sidecar_containers(None, cache_only=False))
        assert len(out) == 1 and out[0]["ram_mb"] == 900

    def test_fresh_cache_short_circuits_regardless_of_flag(self, monkeypatch):
        # Within TTL, both read and sampler callers get the cache with no probe.
        self._reset()
        cp._cache["data"] = [{"name": "warm", "ram_mb": 7}]
        import time as _t
        cp._cache["at"] = _t.monotonic()  # fresh

        def _boom(_s):
            raise AssertionError("fresh cache must not probe")
        monkeypatch.setattr(cp, "_get_docker", _boom)
        assert asyncio.run(cp.probe_sidecar_containers(None, cache_only=False)) == \
            [{"name": "warm", "ram_mb": 7}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
