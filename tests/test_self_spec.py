"""Test no-rollback self-speculation approach."""
from __future__ import annotations

import time


def test_no_rollback():
    from llama_cpp import Llama
    from llama_cpp import llama_cpp as lcpp

    model_path = "/models/Qwen3.5-0.8B-Q8_0.gguf"
    print(f"Loading {model_path}...")
    llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=99, verbose=False)
    print(f"Loaded, n_vocab={llm.n_vocab()}")

    prompt = list(llm.tokenize(b"The quick brown fox jumps over"))
    K = 3  # draft tokens per iteration

    # Generate 30 tokens using no-rollback self-spec
    print(f"\n=== No-Rollback Self-Spec (K={K}) ===")
    llm.reset()
    llm.eval(prompt)
    tokens_out = []
    t0 = time.time()

    for iteration in range(10):
        # Full model sample
        tok = llm.sample(temp=0.01, idx=llm.n_tokens - 1)
        if tok == llm.token_eos():
            print("EOS reached")
            break
        tokens_out.append(tok)
        llm.eval([tok])

        # Draft K tokens
        lcpp.llama_set_shared_expert_draft(llm._ctx.ctx, True)
        draft_toks = []
        for _ in range(K):
            dt = llm.sample(temp=0.01, idx=llm.n_tokens - 1)
            if dt == llm.token_eos():
                break
            draft_toks.append(dt)
            llm.eval([dt])
        lcpp.llama_set_shared_expert_draft(llm._ctx.ctx, False)

        tokens_out.extend(draft_toks)

    elapsed = time.time() - t0
    text = llm.detokenize(tokens_out).decode('utf-8', errors='replace')
    print(f"Generated {len(tokens_out)} tokens in {elapsed:.2f}s = {len(tokens_out)/elapsed:.1f} tok/s")
    print(f"Text: {text}")

    # Baseline: same but ALL full model (no draft)
    print("\n=== Baseline (full model only) ===")
    llm.reset()
    llm.eval(prompt)
    tokens_baseline = []
    t0 = time.time()

    for _ in range(len(tokens_out)):
        tok = llm.sample(temp=0.01, idx=llm.n_tokens - 1)
        if tok == llm.token_eos():
            break
        tokens_baseline.append(tok)
        llm.eval([tok])

    elapsed = time.time() - t0
    text_baseline = llm.detokenize(tokens_baseline).decode('utf-8', errors='replace')
    print(f"Generated {len(tokens_baseline)} tokens in {elapsed:.2f}s = {len(tokens_baseline)/elapsed:.1f} tok/s")
    print(f"Text: {text_baseline}")

    # Compare
    print("\n=== Comparison ===")
    matching = sum(1 for a, b in zip(tokens_out, tokens_baseline) if a == b)
    print(f"Token match: {matching}/{min(len(tokens_out), len(tokens_baseline))}")

    del llm


if __name__ == "__main__":
    test_no_rollback()
