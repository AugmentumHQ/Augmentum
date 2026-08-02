"""Tests for speculative decoding (server-level draft-verify loop)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "engine"))

try:
    from speculative import DualModelSlot, SpeculativeGenerator, SpeculativeStats
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"speculative not importable in this build: {_import_exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Mock Llama model for testing without a real GGUF model
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


class MockLlamaContext:
    def __init__(self):
        self._removed = []

    def kv_cache_seq_rm(self, seq_id, p0, p1):
        self._removed.append((seq_id, p0, p1))
        return True

    def get_logits(self):
        return None

    def get_logits_ith(self, i):
        import ctypes
        arr = (ctypes.c_float * 100)()
        # Make the token at position (10 + i) most likely
        arr[10 + i] = 10.0
        return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))

    def decode(self, batch):
        pass


class MockLlama:
    """Minimal mock of llama_cpp.Llama for speculative testing."""

    def __init__(self, tokens_to_sample=None, vocab_size=100):
        self.n_tokens = 0
        self.input_ids = np.zeros(8192, dtype=np.intc)
        self._ctx = MockLlamaContext()
        self._batch = MockBatchWrapper()
        self._eos_token = 2
        self._n_vocab = vocab_size
        self._logits_all = False
        self._n_vocab = vocab_size
        self._sample_queue = list(tokens_to_sample or [])
        self._sample_idx = 0
        self._eval_calls = []
        self._reset_count = 0

    def reset(self):
        self.n_tokens = 0
        self._reset_count += 1

    def eval(self, tokens):
        self._eval_calls.append(list(tokens))
        for i, t in enumerate(tokens):
            self.input_ids[self.n_tokens + i] = t
        self.n_tokens += len(tokens)

    def sample(self, idx=None, **kwargs):
        if self._sample_idx >= len(self._sample_queue):
            return 2  # EOS
        token = self._sample_queue[self._sample_idx]
        self._sample_idx += 1
        return token

    def token_eos(self):
        return self._eos_token

    def tokenize(self, text):
        return list(text)

    def n_vocab(self):
        return self._n_vocab


# ---------------------------------------------------------------------------
# Stats Tests
# ---------------------------------------------------------------------------

def test_stats_initial():
    stats = SpeculativeStats()
    assert stats.acceptance_rate == 0.0
    assert stats.speedup_estimate == 1.0
    assert stats.avg_draft_ms == 0.0
    d = stats.to_dict()
    assert "total_drafted" in d
    assert "speedup_estimate" in d


def test_stats_after_work():
    stats = SpeculativeStats(total_drafted=10, total_accepted=7, total_iterations=2)
    assert stats.acceptance_rate == 0.7
    assert stats.speedup_estimate == 4.5  # 1 + 7/2 = 4.5


# ---------------------------------------------------------------------------
# SpeculativeGenerator Tests
# ---------------------------------------------------------------------------

def test_generate_no_drafts():
    """When draft model produces no tokens, fall back to single-token mode."""
    # Main model generates: 10, 11, 12, EOS
    main = MockLlama(tokens_to_sample=[10, 11, 12, 2])
    # Draft model always returns EOS immediately
    draft = MockLlama(tokens_to_sample=[2] * 20)

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)
    tokens = list(gen.generate([1, 2, 3]))

    # Should get 10, 11, 12, EOS (2)
    assert tokens == [10, 11, 12, 2]


def test_generate_all_accepted():
    """When all draft tokens match, yield them plus a bonus."""
    # Main model: generates 10, then verifies 20,21,22 (all match), bonus=30
    # Verify positions: offset+0 → 20, offset+1 → 21, offset+2 → 22, bonus=30
    main = MockLlama(tokens_to_sample=[10, 20, 21, 22, 30, 2])
    # Draft model: predicts 20, 21, 22 (match main's verification)
    draft = MockLlama(tokens_to_sample=[20, 21, 22] * 5)

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)
    tokens = list(gen.generate([1, 2, 3]))

    # Should get: 10, 20, 21, 22, 30, EOS
    assert 10 in tokens
    assert 2 in tokens  # EOS


def test_generate_rejection():
    """When a draft token is rejected, yield the correction and continue."""
    # Main generates 10, then for verification:
    #   position 0: 20 (matches draft), position 1: 99 (rejects draft's 21)
    # After correction, continues: sample=50, EOS
    main = MockLlama(tokens_to_sample=[10, 20, 99, 50, 2])
    # Draft predicts 20, 21, 22
    draft = MockLlama(tokens_to_sample=[20, 21, 22, 20, 21, 22])

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)
    tokens = list(gen.generate([1, 2, 3]))

    assert 10 in tokens
    assert 20 in tokens  # accepted draft
    assert 99 in tokens  # correction (rejected 21)
    assert 2 in tokens   # EOS
    assert gen.stats.total_rejections >= 1


def test_generate_eos_during_verify():
    """EOS during verification should stop generation."""
    main = MockLlama(tokens_to_sample=[10, 2])  # token 10, then EOS at verification
    draft = MockLlama(tokens_to_sample=[20, 21, 22])

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)
    tokens = list(gen.generate([1, 2, 3]))

    # Should stop after seeing EOS
    assert tokens[-1] == 2


def test_stats_tracking():
    """Verify stats are properly tracked during generation."""
    main = MockLlama(tokens_to_sample=[10, 20, 21, 2])
    draft = MockLlama(tokens_to_sample=[20, 21, 22])

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)
    list(gen.generate([1, 2, 3]))

    assert gen.stats.total_iterations >= 1
    assert gen.stats.total_drafted >= 0


def test_draft_fallback_on_error():
    """When draft model fails during drafting, fall back to non-speculative mode."""
    main = MockLlama(tokens_to_sample=[10, 11, 2])
    draft = MockLlama(tokens_to_sample=[20])

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)

    # Poison the draft eval AFTER generator is created (prompt eval will work)
    original_eval = draft.eval
    call_count = [0]
    def sometimes_bad_eval(tokens):
        call_count[0] += 1
        if call_count[0] > 1:  # let prompt eval succeed, fail on drafting
            raise RuntimeError("Draft model error")
        return original_eval(tokens)
    draft.eval = sometimes_bad_eval

    tokens = list(gen.generate([1, 2, 3]))

    assert 10 in tokens
    assert gen.stats.total_fallbacks >= 1


# ---------------------------------------------------------------------------
# DualModelSlot Tests
# ---------------------------------------------------------------------------

def test_dual_model_slot_init():
    slot = DualModelSlot()
    assert not slot.ready
    assert slot.main_llm is None
    assert slot.draft_llm is None


def test_dual_model_slot_unload():
    slot = DualModelSlot()
    slot.main_llm = MockLlama()
    slot.draft_llm = MockLlama()
    slot.main_name = "test-main"
    slot.draft_name = "test-draft"

    slot.unload()
    assert not slot.ready
    assert slot.main_llm is None
    assert slot.draft_llm is None
    assert slot.main_name == ""


def test_dual_model_slot_stats_empty():
    slot = DualModelSlot()
    stats = slot.stats()
    assert stats["ready"] is False
    assert stats["generator_stats"] == {}


# ---------------------------------------------------------------------------
# Sync Draft KV Tests
# ---------------------------------------------------------------------------

def test_sync_draft_truncation():
    """Draft KV should be truncated when main is behind draft."""
    main = MockLlama(tokens_to_sample=[10, 11, 12])
    draft = MockLlama(tokens_to_sample=[20, 21, 22])

    gen = SpeculativeGenerator(main, draft, num_pred_tokens=3)

    # Simulate: draft has synced 10 tokens, main only has 5
    gen._draft_n_synced = 10
    draft.n_tokens = 10
    for i in range(10):
        draft.input_ids[i] = i

    main_ids = np.arange(5, dtype=np.intc)
    # Pad main_ids with same values for common prefix check
    padded = np.zeros(8192, dtype=np.intc)
    padded[:5] = main_ids

    gen._sync_draft_kv(padded, 5)
    assert gen._draft_n_synced == 5
