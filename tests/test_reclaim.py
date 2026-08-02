"""Manual memory reclamation (spec §7.1).

The reclaim path's whole value is that its numbers can be trusted, so these
tests are mostly about honesty rather than plumbing: that it refuses busy
slots, that it re-checks between preview and execute, that it reports the
measured delta rather than the sum of what it hoped to free, and that it says
out loud when memory is unreclaimable instead of quietly freeing nothing.
"""

from __future__ import annotations

import time
import types
from unittest.mock import patch

import pytest

from augmentum.resource import reclaim
from augmentum.resource.hostmem import MemoryInfo


def _mgr(*, model_id="m", pid=4242, idle_s=999.0, pinned=False):
    """A stand-in for LlamaServerManager with just the surface reclaim reads."""
    m = types.SimpleNamespace()
    m.model_id = model_id
    m.process = types.SimpleNamespace(pid=pid) if pid else None
    m._last_request_time = (time.monotonic() - idle_s) if idle_s >= 0 else 0.0
    m.is_pinned = lambda _mid: pinned
    m.stopped = False

    async def _stop():
        m.stopped = True

    m.stop = _stop
    return m


def _state(**slots):
    st = types.SimpleNamespace()
    st.llama_manager = slots.get("a")
    st.secondary_slot = types.SimpleNamespace(manager=slots["b"]) if "b" in slots else None
    st.classifier_slot = types.SimpleNamespace(manager=slots["c"]) if "c" in slots else None
    return st


class TestCandidateClassification:
    def test_pinned_slot_is_never_reclaimable(self):
        c = reclaim._slot_candidate("slot_a", "Slot A", _mgr(pinned=True), 120.0)
        assert c.reclaimable is False
        assert "pinned" in c.reason

    def test_recently_active_slot_is_refused_with_the_numbers(self):
        c = reclaim._slot_candidate("slot_a", "Slot A", _mgr(idle_s=10.0), 120.0)
        assert c.reclaimable is False
        # §7 requires refusals carry real numbers, not just "busy".
        assert "10s ago" in c.reason and "120s" in c.reason

    def test_unanswerable_pin_check_is_treated_as_busy(self):
        """Failing safe matters more than reclaiming: guessing 'free' here
        would unload a model out from under a live request."""
        m = _mgr()
        m.is_pinned = lambda _mid: (_ for _ in ()).throw(RuntimeError("no"))
        c = reclaim._slot_candidate("slot_a", "Slot A", m, 120.0)
        assert c.reclaimable is False

    def test_slot_with_no_completed_request_is_refused(self):
        """A slot mid-load has no last-request stamp; it is not idle."""
        c = reclaim._slot_candidate("slot_a", "Slot A", _mgr(idle_s=-1), 120.0)
        assert c.reclaimable is False
        assert "mid-load" in c.reason

    def test_empty_slot_is_not_a_candidate(self):
        assert reclaim._slot_candidate("slot_a", "A", _mgr(model_id=""), 120.0) is None
        assert reclaim._slot_candidate("slot_a", "A", _mgr(pid=None), 120.0) is None

    def test_mlocked_slot_is_reclaimable_but_says_why_it_is_expensive(self):
        """H3 made visible: mlocked memory returns only via a full unload, so
        a user seeing a small delta gets an explanation rather than a mystery."""
        with patch.object(reclaim, "_proc_locked_mib", return_value=15_360), \
                patch.object(reclaim, "_proc_rss_mib", return_value=16_000):
            c = reclaim._slot_candidate("slot_a", "Slot A", _mgr(), 120.0)
        assert c.reclaimable is True
        assert "mlocked" in c.reason and "15.0 GB" in c.reason


class TestPreview:
    @pytest.mark.asyncio
    async def test_splits_reclaimable_from_blocked(self):
        st = _state(a=_mgr(model_id="busy", pinned=True), b=_mgr(model_id="idle"))
        with patch.object(reclaim, "_proc_rss_mib", return_value=2048), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0):
            plan = await reclaim.preview(st)
        assert [c["key"] for c in plan["blocked"]] == ["slot_a"]
        assert "slot_b" in [c["key"] for c in plan["candidates"]]

    @pytest.mark.asyncio
    async def test_allocator_is_always_offered_and_flagged_as_unknowable(self):
        plan = await reclaim.preview(_state())
        alloc = next(c for c in plan["candidates"] if c["key"] == "allocator")
        assert alloc["est"] is True and alloc["mib"] == 0
        # The UI must render "at least X", never a promise.
        assert plan["estimate_is_partial"] is True

    @pytest.mark.asyncio
    async def test_estimate_excludes_unknowable_items(self):
        st = _state(b=_mgr(model_id="idle"))
        with patch.object(reclaim, "_proc_rss_mib", return_value=3072), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0):
            plan = await reclaim.preview(st)
        # 3072 from the slot; the allocator contributes 0, not a guess.
        assert plan["estimated_mib"] == 3072


class TestRun:
    @pytest.mark.asyncio
    async def test_reports_measured_delta_not_declared(self):
        """H2: a component that 'freed' memory into an allocator arena
        returned nothing to the kernel. Only the working-set delta is real."""
        st = _state(b=_mgr(model_id="idle"))
        before = MemoryInfo(32768, 8192, 24576, "test", True)
        after = MemoryInfo(32768, 12288, 20480, "test", True)
        with patch.object(reclaim, "_proc_rss_mib", return_value=6000), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0), \
                patch.object(reclaim, "trim_allocator", return_value=0), \
                patch.object(reclaim.hostmem, "memory_info",
                             side_effect=[before, before, after]):
            res = await reclaim.run(st, keys=["slot_b"])
        # The slot claimed 6000 MiB; the container only moved 4096.
        assert res["measured_freed_mib"] == 4096

    @pytest.mark.asyncio
    async def test_slot_that_went_busy_after_preview_is_skipped_not_evicted(self):
        """Preview → click → execute is easily tens of seconds. A slot the
        user started using in between must win over the stale plan."""
        mgr = _mgr(model_id="idle")
        st = _state(b=mgr)
        real = reclaim._slot_candidate
        calls = {"n": 0}

        def flaky(key, name, m, min_idle):
            calls["n"] += 1
            if calls["n"] > 1:          # the execute-time re-check
                m._last_request_time = time.monotonic()
            return real(key, name, m, min_idle)

        with patch.object(reclaim, "_proc_rss_mib", return_value=4096), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0), \
                patch.object(reclaim, "_slot_candidate", side_effect=flaky):
            res = await reclaim.run(st, keys=["slot_b"])
        assert mgr.stopped is False
        assert [s["key"] for s in res["skipped"]] == ["slot_b"]

    @pytest.mark.asyncio
    async def test_pinned_slot_requested_explicitly_is_refused_with_reason(self):
        st = _state(a=_mgr(model_id="busy", pinned=True))
        with patch.object(reclaim, "_proc_rss_mib", return_value=4096), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0):
            res = await reclaim.run(st, keys=["slot_a"])
        assert st.llama_manager.stopped is False
        assert res["skipped"] and "pinned" in res["skipped"][0]["reason"]

    @pytest.mark.asyncio
    async def test_failed_unload_is_reported_not_swallowed(self):
        mgr = _mgr(model_id="idle")

        async def _boom():
            raise RuntimeError("container gone")

        mgr.stop = _boom
        with patch.object(reclaim, "_proc_rss_mib", return_value=4096), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0), \
                patch.object(reclaim, "trim_allocator", return_value=0):
            res = await reclaim.run(_state(b=mgr), keys=["slot_b"])
        assert res["freed"] == []
        assert "container gone" in res["skipped"][0]["reason"]

    @pytest.mark.asyncio
    async def test_no_keys_means_everything_reclaimable(self):
        st = _state(a=_mgr(model_id="busy", pinned=True), b=_mgr(model_id="idle"))
        with patch.object(reclaim, "_proc_rss_mib", return_value=4096), \
                patch.object(reclaim, "_proc_locked_mib", return_value=0), \
                patch.object(reclaim, "trim_allocator", return_value=12):
            res = await reclaim.run(st)
        assert st.secondary_slot.manager.stopped is True
        assert st.llama_manager.stopped is False   # pinned, still untouched
        # "Everything" means everything RECLAIMABLE — the pinned slot is not
        # silently retried and is not reported as skipped, because the user
        # never asked for it.
        assert "slot_a" not in [f["key"] for f in res["freed"]]
        assert res["skipped"] == []

    @pytest.mark.asyncio
    async def test_delta_never_goes_negative(self):
        """Memory can grow during a reclaim (background work). Reporting a
        negative "freed" figure would be nonsense; report zero."""
        grew = MemoryInfo(32768, 4096, 28672, "test", True)
        start = MemoryInfo(32768, 8192, 24576, "test", True)
        with patch.object(reclaim, "trim_allocator", return_value=0), \
                patch.object(reclaim.hostmem, "memory_info",
                             side_effect=[start, start, grew]):
            res = await reclaim.run(_state(), keys=["allocator"])
        assert res["measured_freed_mib"] == 0


class TestServiceMemLimits:
    """Managed services are capped ONLY when an operator says so.

    These tests exist to keep a default ceiling from creeping back in. An
    earlier pass gave every catalog service a per-category default derived
    from ``resources.ram_mb``; it was reverted because the memory incident it
    responded to came from the Augmentum stack, with no catalog service
    running, and because a static per-service number cannot encode how many
    people are using the thing.
    """

    @staticmethod
    def _sd(**kw):
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        base = dict(
            id="svc", name="Svc", description="", category=ServiceCategory.SERVICE,
            image="img", internal_port=80, host_port=8080,
        )
        base.update(kw)
        return ServiceDefinition(**base)

    def test_no_ceiling_without_an_explicit_one(self):
        """No category default, for any category. A service Augmentum did not
        author, running a workload Augmentum did not choose, gets no ceiling
        Augmentum invented."""
        from augmentum.providers.manager import _resolve_mem_limit
        from augmentum.providers.models import ServiceCategory

        for cat in ServiceCategory:
            assert _resolve_mem_limit(self._sd(category=cat)) == 0

    def test_declared_minimum_never_becomes_a_ceiling(self):
        """The regression this class exists to prevent. ram_mb answers "can
        this host start it", not "how much will it ever want"."""
        from augmentum.providers.manager import _resolve_mem_limit

        assert _resolve_mem_limit(self._sd(min_ram_mb=256)) == 0
        assert _resolve_mem_limit(self._sd(min_ram_mb=8000)) == 0

    def test_explicit_operator_limit_is_honoured_exactly(self):
        """Not scaled, not doubled, not floored — an operator who measured
        their own deployment gets the number they asked for."""
        from augmentum.providers.manager import _resolve_mem_limit

        assert _resolve_mem_limit(self._sd(mem_limit="512m")) == 512 * 1024**2
        assert _resolve_mem_limit(self._sd(mem_limit="512m", min_ram_mb=8000)) == 512 * 1024**2

    def test_unparseable_limit_leaves_the_service_unbounded(self):
        """Falling back to a guessed number on a typo would kill a service for
        a config error. Unbounded is the safe reading, and it is logged."""
        from augmentum.providers.manager import _resolve_mem_limit

        assert _resolve_mem_limit(self._sd(mem_limit="lots")) == 0

    def test_uncapped_service_gets_no_memory_keys_at_all(self):
        from augmentum.providers.manager import ServiceManager

        hc = ServiceManager._build_container_config(self._sd(), "net")["HostConfig"]
        assert "Memory" not in hc and "MemorySwap" not in hc

    def test_operator_limit_is_a_valid_docker_size(self):
        """Whatever the user types has to survive the round-trip into a Docker
        byte count — a value we accept but Docker rejects fails at container
        create, long after the UI said "saved"."""
        from augmentum.providers.manager import _parse_size

        for text, want in (("2g", 2 * 1024**3), ("512m", 512 * 1024**2),
                           ("2gb", 2 * 1024**3), ("1073741824", 1024**3)):
            assert _parse_size(text) == want

    def test_capped_service_pins_swap_to_the_limit(self):
        """Leaving MemorySwap unset lets Docker grant swap equal to the limit,
        so a runaway gets 2x the ceiling and thrashes the disk instead of
        stopping."""
        from augmentum.providers.manager import ServiceManager

        sd = self._sd(mem_limit="2g")
        hc = ServiceManager._build_container_config(sd, "net")["HostConfig"]
        assert hc["Memory"] == 2 * 1024**3
        assert hc["MemorySwap"] == hc["Memory"]


class TestUserSetMemLimit:
    """The opt-in ceiling: Augmentum never picks a number, the operator does.

    The value is user input that cannot be derived, so it has to survive every
    path that recreates a container — the same contract env/volume overrides
    already have, and the reason ``enable_service`` is the single place it is
    applied rather than each call site.
    """

    @staticmethod
    def _mgr(*, persisted="", running=True):
        from augmentum.providers.manager import ServiceManager

        m = ServiceManager.__new__(ServiceManager)
        m._db = object()
        m._saved: dict = {}
        m._recreated: list = []

        async def read_config_json(_sid):
            return dict(m._saved) if m._saved else ({"mem_limit": persisted}
                                                    if persisted else {})

        async def update_config_json(_sid, patch):
            m._saved.update(patch)

        async def find(_sid):
            return object() if running else None

        async def recreate(sid):
            m._recreated.append(sid)
            return "managed"

        m.read_config_json = read_config_json
        m.update_config_json = update_config_json
        m._find_container = find
        m.recreate_with_new_credential = recreate
        return m

    @staticmethod
    def _sd():
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        return ServiceDefinition(
            id="svc", name="Svc", description="", category=ServiceCategory.SERVICE,
            image="img", internal_port=80, host_port=8080,
        )

    @pytest.mark.asyncio
    async def test_persisted_limit_is_applied_on_every_provision(self):
        """The regression guard for a restart silently dropping the ceiling."""
        from augmentum.providers.manager import _resolve_mem_limit

        mgr = self._mgr(persisted="2g")
        sd = await mgr._with_mem_limit_override("svc", self._sd())
        assert sd.mem_limit == "2g"
        assert _resolve_mem_limit(sd) == 2 * 1024**3

    @pytest.mark.asyncio
    async def test_no_override_leaves_the_definition_untouched(self):
        mgr = self._mgr()
        sd_in = self._sd()
        assert (await mgr._with_mem_limit_override("svc", sd_in)) is sd_in

    @pytest.mark.asyncio
    async def test_setting_a_limit_persists_then_recreates(self):
        mgr = self._mgr()
        await mgr.set_mem_limit("svc", "2g")
        assert mgr._saved["mem_limit"] == "2g"
        assert mgr._recreated == ["svc"]     # applied, not merely stored

    @pytest.mark.asyncio
    async def test_clearing_the_limit_is_supported(self):
        """Getting back to unbounded must be as easy as setting a limit —
        otherwise a bad guess is a trap."""
        mgr = self._mgr(persisted="2g")
        await mgr.set_mem_limit("svc", "")
        assert mgr._saved["mem_limit"] == ""
        assert mgr._recreated == ["svc"]

    @pytest.mark.asyncio
    async def test_stopped_service_saves_without_a_pointless_restart(self):
        mgr = self._mgr(running=False)
        assert await mgr.set_mem_limit("svc", "2g") is None
        assert mgr._saved["mem_limit"] == "2g"
        assert mgr._recreated == []

    @pytest.mark.asyncio
    async def test_a_typo_is_rejected_and_nothing_is_persisted(self):
        """Saving an unparseable value would resolve to 0 (unlimited) while
        the UI reported success — silent no-op is the worst outcome."""
        mgr = self._mgr()
        with pytest.raises(ValueError, match="memory size"):
            await mgr.set_mem_limit("svc", "2 gigs")
        assert mgr._saved == {} and mgr._recreated == []

    @pytest.mark.asyncio
    async def test_absurdly_small_limit_is_refused_up_front(self):
        """A 1m ceiling crash-loops the container; the user would blame the
        app, not the number they typed."""
        mgr = self._mgr()
        with pytest.raises(ValueError, match="at least"):
            await mgr.set_mem_limit("svc", "1m")
        assert mgr._saved == {}

    @pytest.mark.asyncio
    async def test_install_time_limit_survives_the_row_not_existing_yet(self):
        """The managed_services row is INSERTed by _persist() at the END of
        enable_service, so a config_json write before it matches zero rows.
        An install-time limit must therefore ride in as an argument and be
        persisted afterwards, or it applies once and vanishes on restart."""
        mgr = self._mgr()
        sd = await mgr._with_mem_limit_override("svc", self._sd(), explicit="2g")
        assert sd.mem_limit == "2g"          # applied to the first container
        # ...and the persisted read path picks it up on every later provision.
        await mgr.update_config_json("svc", {"mem_limit": "2g"})
        assert (await mgr._with_mem_limit_override("svc", self._sd())).mem_limit == "2g"

    @pytest.mark.asyncio
    async def test_explicit_empty_string_means_no_limit_not_fall_back(self):
        """Installing with the box left blank must not inherit a stale
        persisted value from a previous install of the same service."""
        mgr = self._mgr(persisted="8g")
        sd = await mgr._with_mem_limit_override("svc", self._sd(), explicit="")
        assert sd.mem_limit == ""
