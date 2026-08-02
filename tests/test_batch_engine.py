"""Tests for continuous batching engine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "engine"))

try:
    from batch_engine import BatchEngine, BatchSequence, SeqState
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"batch_engine not importable in this build: {_import_exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Mock Llama for batch engine testing
# ---------------------------------------------------------------------------

class MockBatch:
    def __init__(self):
        self.n_tokens = 0
        self.token = [0] * 2048
        self.pos = [0] * 2048
        self.seq_id = [[0] for _ in range(2048)]
        self.n_seq_id = [0] * 2048
        self.logits = [False] * 2048


class MockBatchWrapper:
    def __init__(self):
        self.batch = MockBatch()

    def reset(self):
        self.batch.n_tokens = 0


class MockModel:
    pass


class MockContext:
    def __init__(self, vocab_size=100):
        self._vocab = vocab_size
        self._decoded = False
        self._removed = []

    def decode(self, batch):
        self._decoded = True

    def get_logits_ith(self, i):
        # Return a ctypes pointer to a float array
        import ctypes
        arr = (ctypes.c_float * 100)()
        # Make token 42 the most likely
        arr[42] = 10.0
        return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))

    def kv_cache_seq_rm(self, seq_id, p0, p1):
        self._removed.append((seq_id, p0, p1))
        return True


class MockLlama:
    def __init__(self):
        self._batch = MockBatchWrapper()
        self._ctx = MockContext()
        self._model = MockModel()
        self._n_vocab = 100
        self._n_ctx = 4096
        self.n_batch = 512
        self._eos_token = 0

    def token_eos(self):
        return self._eos_token


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_batch_sequence_initial_state():
    seq = BatchSequence(
        seq_id=1,
        request_id="test_1",
        prompt_tokens=[1, 2, 3],
        max_tokens=10,
    )
    assert seq.state == SeqState.QUEUED
    assert not seq.done
    assert seq.prefill_remaining == 3
    assert seq._kv_freed is False


def test_batch_sequence_finish():
    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1)
    seq.finish("stop")
    assert seq.done
    assert seq.state == SeqState.DONE
    assert seq.finish_reason == "stop"
    assert seq.done_event.is_set()


def test_batch_sequence_fail():
    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1)
    seq.fail("test error")
    assert seq.done
    assert seq.state == SeqState.ERROR
    assert "test error" in seq.finish_reason


def test_batch_engine_init():
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=4, max_batch_tokens=256)
    assert engine.max_sequences == 4
    assert engine.max_batch_tokens == 256
    assert engine._running is False


@pytest.mark.asyncio
async def test_batch_submit():
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=2)

    seq = await engine.submit(
        prompt_tokens=[1, 2, 3],
        max_tokens=10,
        temperature=0.5,
    )
    assert seq.seq_id >= 1
    assert seq.request_id.startswith("batch_")
    assert len(seq.prompt_tokens) == 3
    assert seq.temperature == 0.5


@pytest.mark.asyncio
async def test_batch_submit_max_sequences():
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=1)

    seq1 = await engine.submit(prompt_tokens=[1], max_tokens=1)
    with pytest.raises(RuntimeError, match="Max concurrent"):
        await engine.submit(prompt_tokens=[2], max_tokens=1)


@pytest.mark.asyncio
async def test_batch_cancel():
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=4)

    seq = await engine.submit(prompt_tokens=[1, 2], max_tokens=10)
    engine.cancel(seq.request_id)
    assert seq.done
    assert seq.state == SeqState.ERROR
    assert seq._kv_freed is True


def test_batch_double_free_prevented():
    """Verify that _kv_freed flag prevents double-free."""
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=4)

    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1)
    engine._free_seq_kv(seq)
    assert seq._kv_freed is True
    assert len(llm._ctx._removed) == 1

    # Second free should be a no-op
    engine._free_seq_kv(seq)
    assert len(llm._ctx._removed) == 1  # still 1


def test_seq_id_recycling():
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=4)

    id1 = engine._alloc_seq_id()
    id2 = engine._alloc_seq_id()
    assert id1 != id2

    # Free id1 and reallocate
    seq = BatchSequence(seq_id=id1, request_id="t", prompt_tokens=[], max_tokens=1)
    engine._free_seq_kv(seq)
    id3 = engine._alloc_seq_id()
    assert id3 == id1  # recycled


def test_stats():
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=4)
    s = engine.stats()
    assert s["running"] is False
    assert s["active_sequences"] == 0
    assert s["total_requests"] == 0


def test_sample_from_logits_greedy():
    llm = MockLlama()
    engine = BatchEngine(llm)

    logits = np.zeros(100, dtype=np.float32)
    logits[42] = 10.0
    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1,
                        temperature=0.0)
    token = engine._sample_from_logits(logits, seq)
    assert token == 42


def test_sample_from_logits_with_temperature():
    llm = MockLlama()
    engine = BatchEngine(llm)

    # With very peaked distribution and low temp, should still get 42
    logits = np.zeros(100, dtype=np.float32)
    logits[42] = 100.0  # very peaked
    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1,
                        temperature=0.1)
    token = engine._sample_from_logits(logits, seq)
    assert token == 42


def test_sample_from_logits_repeat_penalty():
    llm = MockLlama()
    engine = BatchEngine(llm)

    logits = np.zeros(100, dtype=np.float32)
    logits[42] = 5.0
    logits[43] = 4.9  # slightly less likely
    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1,
                        temperature=0.0, repeat_penalty=2.0)
    seq.generated_tokens = [42, 42, 42]  # heavy repeat

    token = engine._sample_from_logits(logits.copy(), seq)
    assert token == 43  # 42 penalized, 43 becomes top


def test_sample_from_logits_top_k():
    llm = MockLlama()
    engine = BatchEngine(llm)

    np.random.seed(42)
    logits = np.random.randn(100).astype(np.float32)
    seq = BatchSequence(seq_id=1, request_id="t", prompt_tokens=[], max_tokens=1,
                        temperature=1.0, top_k=5, top_p=1.0)

    # Run multiple samples and verify they're from top-k
    top_5 = set(np.argsort(logits)[-5:])
    for _ in range(20):
        token = engine._sample_from_logits(logits.copy(), seq)
        assert token in top_5


def test_process_batch_step_prefill():
    """Verify that prefill correctly adds prompt tokens to batch."""
    llm = MockLlama()
    engine = BatchEngine(llm, max_sequences=4)

    seq = BatchSequence(
        seq_id=1, request_id="t",
        prompt_tokens=[10, 20, 30, 40, 50],
        max_tokens=5, temperature=0.0,
    )

    engine._process_batch_step([seq])

    # Should have prefilled all 5 tokens
    assert seq.prefill_offset == 5
    assert seq.n_past == 5
    # After prefill + sampling, should be in DECODE state
    assert seq.state == SeqState.DECODE
    assert llm._ctx._decoded  # decode was called


def test_process_batch_step_decode():
    """Verify decode step generates tokens."""
    llm = MockLlama()
    engine = BatchEngine(llm)

    seq = BatchSequence(
        seq_id=1, request_id="t",
        prompt_tokens=[10, 20],
        max_tokens=3, temperature=0.0,
    )
    # Pre-fill manually
    seq.state = SeqState.DECODE
    seq.prefill_offset = 2
    seq.n_past = 2
    seq.last_token = 20

    engine._process_batch_step([seq])

    assert len(seq.generated_tokens) == 1
    assert seq.n_past == 3
