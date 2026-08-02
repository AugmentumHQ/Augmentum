"""llama-server memory-line capture robustness.

The resource ledger reports a model's EXACT VRAM from these parsed totals;
when capture fails it falls back to a (often wildly wrong) plan estimate —
the bug behind two resident models summing past the physical device total.
These pin the tolerant parser: standard MiB lines, novel buffer names, and
non-MiB units all parse, host buffers count as RAM, and unparseable buffer
lines are stashed for the diagnostic warning.
"""

from __future__ import annotations

import pytest

from augmentum.models.llama_server_manager import LlamaServerManager


def _mgr(tmp_path) -> LlamaServerManager:
    return LlamaServerManager(model_dir=str(tmp_path / "models"))


def test_captures_standard_mib_lines(tmp_path):
    m = _mgr(tmp_path)
    for line in (
        "load_tensors:        CUDA0 model buffer size =  4096.00 MiB",
        "llama_kv_cache:      CUDA0 KV buffer size =  1024.00 MiB",
        "sched_reserve:       CUDA0 compute buffer size =   512.00 MiB",
    ):
        m._capture_actual_memory_from_line(line)
    snap = m._actual_memory_snapshot()
    assert snap is not None
    assert snap["vram_total_mib"] == pytest.approx(4096 + 1024 + 512, abs=1)
    # "compute_buffer" key must survive (workspace calibration depends on it).
    assert "compute_buffer" in snap["components"]


def test_captures_novel_buffer_name_and_gib_unit(tmp_path):
    """A buffer name the old fixed list didn't enumerate + a GiB unit — the
    exact shape that used to capture NOTHING and force an estimate."""
    m = _mgr(tmp_path)
    m._capture_actual_memory_from_line("alloc: CUDA0 ssm state buffer size = 2.00 GiB")
    snap = m._actual_memory_snapshot()
    assert snap is not None
    assert snap["vram_total_mib"] == pytest.approx(2048, abs=1)  # 2 GiB → MiB


def test_host_buffer_counts_as_ram_not_vram(tmp_path):
    m = _mgr(tmp_path)
    m._capture_actual_memory_from_line("load_tensors: CPU_Mapped model buffer size = 500.00 MiB")
    snap = m._actual_memory_snapshot()
    assert snap is not None
    assert snap["ram_total_mib"] == pytest.approx(500, abs=1)
    assert snap["vram_total_mib"] == 0


def test_unparseable_buffer_line_is_stashed_for_diagnostic(tmp_path):
    m = _mgr(tmp_path)
    # Contains "buffer" + "size" but not the parseable "<words> buffer size = N unit" shape.
    m._capture_actual_memory_from_line("memory: allocated buffer for weights, size unknown")
    assert m._actual_memory_snapshot() is None
    assert any("buffer" in s for s in m._mem_capture_misses)


def test_misses_are_capped(tmp_path):
    m = _mgr(tmp_path)
    for i in range(50):
        m._capture_actual_memory_from_line(f"x: weird buffer blob size ?? line {i}")
    assert len(m._mem_capture_misses) <= 12


def test_reset_clears_misses_and_components(tmp_path):
    m = _mgr(tmp_path)
    m._capture_actual_memory_from_line("load_tensors: CUDA0 model buffer size = 100.00 MiB")
    m._capture_actual_memory_from_line("junk buffer size nope")
    m._reset_actual_memory()
    assert m._actual_memory_snapshot() is None
    assert m._mem_capture_misses == []
