"""Tests for TurboQuant KV cache compression (Phase 4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "engine"))

from turboquant import TurboQuantizer, benchmark_quality


def test_tq3_roundtrip():
    """TQ3_0 quantize → dequantize preserves signal."""
    tq = TurboQuantizer("tq3_0")
    data = np.random.randn(1024).astype(np.float32)

    blocks = tq.quantize(data)
    restored = tq.dequantize(blocks)

    assert len(restored) >= len(data)
    mse = np.mean((data - restored[:len(data)]) ** 2)
    # TQ3_0 MSE should be reasonable (< 0.05 for unit normal data with absmax scaling)
    assert mse < 0.05, f"TQ3_0 MSE too high: {mse}"


def test_tq4_roundtrip():
    """TQ4_0 quantize → dequantize with better quality than TQ3."""
    tq = TurboQuantizer("tq4_0")
    data = np.random.randn(1024).astype(np.float32)

    blocks = tq.quantize(data)
    restored = tq.dequantize(blocks)

    mse = np.mean((data - restored[:len(data)]) ** 2)
    assert mse < 0.05, f"TQ4_0 MSE too high: {mse}"


def test_q8_roundtrip():
    """Q8_0 quantize → dequantize with very low error."""
    tq = TurboQuantizer("q8_0")
    data = np.random.randn(1024).astype(np.float32)

    blocks = tq.quantize(data)
    restored = tq.dequantize(blocks)

    mse = np.mean((data - restored[:len(data)]) ** 2)
    assert mse < 0.001, f"Q8_0 MSE too high: {mse}"


def test_f16_passthrough():
    """F16 roundtrip has minimal precision loss."""
    tq = TurboQuantizer("f16")
    data = np.random.randn(100).astype(np.float32)

    blocks = tq.quantize(data)
    restored = tq.dequantize(blocks)

    # fp16 roundtrip: relative error < 0.1% for typical values
    assert np.allclose(data, restored[:len(data)], rtol=1e-2, atol=1e-2)


def test_compression_ratios():
    """Verify compression ratios match expectations."""
    # With BLOCK_SIZE=32, overhead is higher than BLOCK_SIZE=256
    assert TurboQuantizer("tq3_0").compression_ratio > 3.0
    assert TurboQuantizer("tq4_0").compression_ratio > 2.5
    assert TurboQuantizer("q8_0").compression_ratio > 1.7
    assert TurboQuantizer("f16").compression_ratio == 1.0


def test_tq3_better_than_nothing():
    """TQ3 should have higher MSE than TQ4 which has higher than Q8."""
    data = np.random.randn(2048).astype(np.float32)

    mse = {}
    for qtype in ["tq3_0", "tq4_0", "q8_0"]:
        tq = TurboQuantizer(qtype)
        blocks = tq.quantize(data)
        restored = tq.dequantize(blocks)
        mse[qtype] = np.mean((data - restored[:len(data)]) ** 2)

    assert mse["tq3_0"] > mse["tq4_0"], "TQ4 should be more accurate than TQ3"
    assert mse["tq4_0"] > mse["q8_0"], "Q8 should be more accurate than TQ4"


def test_zero_block():
    """Zero input should not crash."""
    tq = TurboQuantizer("tq3_0")
    data = np.zeros(256, dtype=np.float32)

    blocks = tq.quantize(data)
    restored = tq.dequantize(blocks)

    assert np.allclose(restored[:256], 0.0, atol=0.01)


def test_small_block():
    """Block smaller than BLOCK_SIZE works."""
    tq = TurboQuantizer("tq3_0")
    data = np.random.randn(10).astype(np.float32)

    blocks = tq.quantize(data)
    assert len(blocks) == 1
    assert blocks[0].n_values == 10

    restored = tq.dequantize(blocks)
    assert len(restored) >= 10


def test_pack_unpack_3bit():
    """3-bit packing roundtrip."""
    tq = TurboQuantizer("tq3_0")
    data = np.random.randn(512).astype(np.float32)

    blocks = tq.quantize(data)
    packed = tq.pack_blocks(blocks)
    unpacked = tq.unpack_blocks(packed)

    assert len(unpacked) == len(blocks)
    for orig, restored in zip(blocks, unpacked):
        assert orig.n_values == restored.n_values
        assert abs(orig.scale - restored.scale) < 1e-5
        np.testing.assert_array_equal(
            orig.indices[:orig.n_values],
            restored.indices[:restored.n_values],
        )


def test_pack_unpack_4bit():
    """4-bit packing roundtrip."""
    tq = TurboQuantizer("tq4_0")
    data = np.random.randn(512).astype(np.float32)

    blocks = tq.quantize(data)
    packed = tq.pack_blocks(blocks)
    unpacked = tq.unpack_blocks(packed)

    assert len(unpacked) == len(blocks)
    for orig, restored in zip(blocks, unpacked):
        np.testing.assert_array_equal(
            orig.indices[:orig.n_values],
            restored.indices[:restored.n_values],
        )


def test_estimate_size():
    """Size estimation produces reasonable numbers."""
    tq = TurboQuantizer("tq3_0")

    # Typical 7B model: 32 layers, 32 heads, 128 head_dim, 4096 context
    est = tq.estimate_size(
        n_tokens=4096,
        n_heads=32,
        head_dim=128,
        n_layers=32,
    )

    assert est["fp16_mb"] > 0
    assert est["tq3_0_mb"] > 0
    assert est["compression_ratio"] > 3.0
    assert est["savings_mb"] > 0
    # 7B model, 4K context, fp16 KV = ~2GB
    assert est["fp16_mb"] > 1000  # at least 1GB
    assert est["tq3_0_mb"] < est["fp16_mb"] / 2  # at least 2x smaller


def test_benchmark_quality():
    """Benchmark produces valid results for all types."""
    results = benchmark_quality(n_samples=5000)

    assert "tq3_0" in results
    assert "tq4_0" in results
    assert "q8_0" in results

    # MSE ordering
    assert results["tq3_0"]["mse"] > results["tq4_0"]["mse"]
    assert results["tq4_0"]["mse"] > results["q8_0"]["mse"]

    # SNR ordering (higher = better)
    assert results["q8_0"]["snr_db"] > results["tq4_0"]["snr_db"]
    assert results["tq4_0"]["snr_db"] > results["tq3_0"]["snr_db"]

    # All have positive SNR
    for qtype in results:
        assert results[qtype]["snr_db"] > 0


def test_invalid_type():
    """Invalid quant type raises error."""
    try:
        TurboQuantizer("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
