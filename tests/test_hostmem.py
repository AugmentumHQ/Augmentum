"""Container-aware memory accounting + host-RAM admission.

Regression suite for the 2026-07-25 incident, in which Augmentum enforced a
VRAM budget, had no host-RAM budget at all, and drove a 128 GB machine into
a forced restart. Spec:
``docs/superpowers/specs/2026-07-25-resource-governance-design.md``.

The bugs these tests pin:

* **B1** — sizing read ``psutil.virtual_memory().total``, which reports the
  host/WSL-VM total inside a container, so ``--cache-ram`` was auto-sized to
  23.6 GiB of anonymous host memory the container never owned.
* **B2** — the OOM-retry ladder relieved VRAM by moving weights into host
  RAM without ever asking whether host RAM had room.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from augmentum.resource import hostmem
from augmentum.resource.hostmem import MemoryInfo


@pytest.fixture(autouse=True)
def _clear_override():
    """The env override is process-global; never leak it between tests."""
    saved = os.environ.pop("AUGMENTUM_MEMORY_LIMIT_MIB", None)
    yield
    os.environ.pop("AUGMENTUM_MEMORY_LIMIT_MIB", None)
    if saved is not None:
        os.environ["AUGMENTUM_MEMORY_LIMIT_MIB"] = saved


class TestSentinels:
    """Both cgroup versions spell "unlimited" differently, and getting
    either wrong silently reintroduces B1."""

    def test_v2_max_keyword_is_unlimited(self, tmp_path: Path):
        f = tmp_path / "memory.max"
        f.write_text("max")
        assert hostmem._read_int(f) is None

    def test_v1_huge_int_is_unlimited(self, tmp_path: Path):
        # The exact value Docker reports on this project's own box.
        f = tmp_path / "memory.limit_in_bytes"
        f.write_text("9223372036854771712")
        assert hostmem._read_int(f) is None

    def test_real_limit_is_read(self, tmp_path: Path):
        f = tmp_path / "memory.max"
        f.write_text(str(32 * 1024**3))
        assert hostmem._read_int(f) == 32 * 1024**3

    def test_garbage_and_missing_do_not_raise(self, tmp_path: Path):
        bad = tmp_path / "bad"
        bad.write_text("not-a-number")
        assert hostmem._read_int(bad) is None
        assert hostmem._read_int(tmp_path / "nope") is None


class TestWorkingSet:
    """Working set is ``usage - inactive_file``, as kubelet does.

    On the box that crashed, 17 GB of a 26 GB footprint was reclaimable
    page cache. Counting it as pressure would evict live models to make
    room for memory the kernel would have handed back for free.
    """

    def test_inactive_file_is_excluded(self, tmp_path: Path):
        (tmp_path / "memory.current").write_text(str(3 * 1024**3))
        (tmp_path / "memory.max").write_text(str(8 * 1024**3))
        (tmp_path / "memory.stat").write_text(
            f"anon {1024**3}\ninactive_file {2 * 1024**3}\n"
        )
        with patch.object(hostmem, "_CG_V2_ROOT", tmp_path):
            info = hostmem.memory_info()
        assert info.source == "cgroup_v2"
        assert info.limited is True
        assert info.used_mib == 1024          # 3 GiB - 2 GiB cache
        assert info.available_mib == 7 * 1024  # 8 GiB limit - 1 GiB working

    def test_v1_uses_total_inactive_file(self, tmp_path: Path):
        (tmp_path / "memory.usage_in_bytes").write_text(str(3 * 1024**3))
        (tmp_path / "memory.limit_in_bytes").write_text(str(4 * 1024**3))
        (tmp_path / "memory.stat").write_text(
            f"total_inactive_file {2 * 1024**3}\n"
        )
        with patch.object(hostmem, "_CG_V2_ROOT", tmp_path / "absent"), \
                patch.object(hostmem, "_CG_V1_ROOT", tmp_path):
            info = hostmem.memory_info()
        assert info.source == "cgroup_v1"
        assert info.used_mib == 1024


class TestUnlimitedCgroup:
    """An unlimited cgroup still accounts US accurately.

    Usage discovery must be independent of limit discovery — conflating
    them paired a configured limit with host-wide usage and reported zero
    available memory on a mostly-idle container.
    """

    def test_reports_our_usage_but_not_limited(self, tmp_path: Path):
        (tmp_path / "memory.current").write_text(str(2 * 1024**3))
        (tmp_path / "memory.max").write_text("max")
        (tmp_path / "memory.stat").write_text("inactive_file 0\n")
        with patch.object(hostmem, "_CG_V2_ROOT", tmp_path):
            info = hostmem.memory_info()
        assert info.limited is False       # nothing is bounding us
        assert info.used_mib == 2048       # but this is still OUR number


class TestEnvOverride:
    """Docker mounts the cgroup read-only and unlimited-by-default, so the
    operator's compose limit must be conveyable out-of-band."""

    def test_override_wins_over_unlimited_cgroup(self, tmp_path: Path):
        (tmp_path / "memory.current").write_text(str(1024**3))
        (tmp_path / "memory.max").write_text("max")
        (tmp_path / "memory.stat").write_text("inactive_file 0\n")
        os.environ["AUGMENTUM_MEMORY_LIMIT_MIB"] = "8192"
        with patch.object(hostmem, "_CG_V2_ROOT", tmp_path):
            info = hostmem.memory_info()
        assert info.source == "env_override"
        assert info.limited is True
        assert info.total_mib == 8192
        # Available must be the limit minus OUR working set (1 GiB), not
        # the host's used memory.
        assert info.available_mib == 7168

    def test_bad_override_is_ignored_not_fatal(self):
        os.environ["AUGMENTUM_MEMORY_LIMIT_MIB"] = "garbage"
        assert hostmem.memory_info().source != "env_override"

    def test_never_raises(self):
        with patch.object(hostmem, "_probe_cgroup", side_effect=OSError("boom")):
            assert hostmem.memory_info().total_mib > 0


class TestBudget:
    def test_fraction_of_available_when_limited(self):
        info = MemoryInfo(32768, 4096, 28672, "test", True)
        with patch.object(hostmem, "memory_info", return_value=info):
            # 25% of what's FREE, not of the total we already spent.
            assert hostmem.budget_mib(0.25) == 1024

    def test_ceiling_and_floor_apply(self):
        info = MemoryInfo(1_000_000, 1_000_000, 0, "test", False)
        with patch.object(hostmem, "memory_info", return_value=info):
            assert hostmem.budget_mib(0.25, ceiling_mib=8192) == 8192
        tiny = MemoryInfo(256, 256, 0, "test", False)
        with patch.object(hostmem, "memory_info", return_value=tiny):
            assert hostmem.budget_mib(0.25, floor_mib=1024) == 1024


class TestCacheRamRegression:
    """B1, pinned directly: the incident's exact numbers."""

    def test_incident_configuration_is_now_bounded(self):
        from augmentum.models.llama_server_manager import LlamaServerManager

        mgr = LlamaServerManager.__new__(LlamaServerManager)
        # What psutil reported inside the container that day: the whole
        # 94 GiB WSL VM, unbounded.
        info = MemoryInfo(96562, 89680, 1131, "psutil", False)
        with patch.object(hostmem, "memory_info", return_value=info):
            got = mgr._auto_cache_ram_mib()
        # Old behaviour: 25% of 94 GiB = 24140 MiB of anonymous host RAM.
        assert got <= LlamaServerManager._CACHE_RAM_ABSOLUTE_CAP_MIB
        assert got < 24140


class TestSpillAdmission:
    """B2: reducing GPU layers moves weights to host RAM. Price the move."""

    @staticmethod
    def _backend():
        from augmentum.models.llama_cpp import LlamaCppBackend

        return LlamaCppBackend.__new__(LlamaCppBackend)

    @staticmethod
    def _profile(total_gib=20, n_layers=40):
        from augmentum.models.llama_server_manager import ModelProfile

        return ModelProfile(
            model_path="m.gguf", model_name="m", n_layers=n_layers,
            n_embed=4096, n_heads=32, n_heads_kv=8,
            total_size_bytes=total_gib * 1024**3,
        )

    def test_spill_is_proportional_to_cpu_layers(self):
        b = self._backend()
        p = self._profile(total_gib=20, n_layers=40)
        assert b._spill_bytes(p, 40) == 0                      # full offload
        assert b._spill_bytes(p, 20) == 10 * 1024**3           # half
        assert b._spill_bytes(p, 0) == 20 * 1024**3            # all on CPU

    @pytest.mark.asyncio
    async def test_refuses_spill_that_does_not_fit(self):
        b = self._backend()
        p = self._profile(total_gib=20, n_layers=40)
        info = MemoryInfo(8192, 4096, 4096, "test", True)
        with patch.object(hostmem, "memory_info", return_value=info):
            ok, reason = await b._host_ram_can_absorb_spill(profile=p, gpu_layers=0)
        assert ok is False
        assert "20.0 GB" in reason and "4.0 GB" in reason  # real numbers, §7

    @pytest.mark.asyncio
    async def test_admits_spill_that_fits(self):
        b = self._backend()
        p = self._profile(total_gib=20, n_layers=40)
        info = MemoryInfo(65536, 40000, 25536, "test", True)
        with patch.object(hostmem, "memory_info", return_value=info):
            ok, _ = await b._host_ram_can_absorb_spill(profile=p, gpu_layers=20)
        assert ok is True

    @pytest.mark.asyncio
    async def test_unknown_profile_admits(self):
        """First load of an unprofiled model must not be blocked."""
        b = self._backend()
        ok, _ = await b._host_ram_can_absorb_spill(profile=None, gpu_layers=0)
        assert ok is True


class TestCheckRamFit:
    @pytest.mark.asyncio
    async def test_refuses_when_over_budget_and_explains(self):
        from augmentum.resource.ledger import ResourceLedger

        led = ResourceLedger.__new__(ResourceLedger)
        info = MemoryInfo(32768, 4096, 28672, "test", True)
        with patch.object(hostmem, "memory_info", return_value=info):
            ok, reason, needed, avail = await led.check_ram_fit(
                needed_mb=8192, label="Slot B",
            )
        assert ok is False
        assert "Slot B" in reason and "container limit" in reason
        assert (needed, avail) == (8192, 4096)

    @pytest.mark.asyncio
    async def test_admits_within_budget(self):
        from augmentum.resource.ledger import ResourceLedger

        led = ResourceLedger.__new__(ResourceLedger)
        info = MemoryInfo(32768, 16384, 16384, "test", True)
        with patch.object(hostmem, "memory_info", return_value=info):
            ok, reason, _, _ = await led.check_ram_fit(needed_mb=4096)
        assert ok is True and reason == ""

    @pytest.mark.asyncio
    async def test_unknown_need_admits(self):
        from augmentum.resource.ledger import ResourceLedger

        led = ResourceLedger.__new__(ResourceLedger)
        ok, _, _, _ = await led.check_ram_fit(needed_mb=0)
        assert ok is True
