"""The KV-cache paths (prefill, batched decode) give the same logits as a full recompute."""

from __future__ import annotations

import pytest
import torch

from llm_inference_engine.model import GPT, KVCache, ModelConfig


def test_prefill_then_decode_matches_full_forward(model: GPT) -> None:
    gen = torch.Generator().manual_seed(1)
    seq = torch.randint(0, 257, (40,), generator=gen)
    cache = KVCache(model.config, 2, 64, dtype=torch.float64)
    full = model.forward_full(seq[None])[0]
    last = model.prefill(seq[:10], cache, slot=1)
    torch.testing.assert_close(last, full[9])
    for pos in range(10, 40):
        logits = model.decode(seq[pos : pos + 1], torch.tensor([1]), torch.tensor([pos]), cache)
        torch.testing.assert_close(logits[0], full[pos])


def test_batched_decode_with_ragged_lengths_matches_each_sequence(model: GPT) -> None:
    gen = torch.Generator().manual_seed(2)
    lengths = [3, 17, 9]
    seqs = [torch.randint(0, 257, (n + 1,), generator=gen) for n in lengths]
    cache = KVCache(model.config, 4, 32, dtype=torch.float64)
    slots = [2, 0, 3]
    for s, slot, n in zip(seqs, slots, lengths, strict=True):
        model.prefill(s[:n], cache, slot)
    tokens = torch.stack([s[n] for s, n in zip(seqs, lengths, strict=True)])
    logits = model.decode(tokens, torch.tensor(slots), torch.tensor(lengths), cache)
    for i, s in enumerate(seqs):
        torch.testing.assert_close(logits[i], model.forward_full(s[None])[0, -1])


def test_slots_are_isolated(model: GPT) -> None:
    cache = KVCache(model.config, 2, 16, dtype=torch.float64)
    a = model.prefill(torch.tensor([5, 6, 7]), cache, 0)
    model.prefill(torch.tensor([200, 1, 2, 3, 4]), cache, 1)
    again = model.prefill(torch.tensor([5, 6, 7]), cache, 0)
    torch.testing.assert_close(a, again)
    assert cache.nbytes == 2 * 2 * 2 * 4 * 16 * 16 * 8


def test_config_and_cache_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=10, n_head=3)
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(n_layer=0)
    cfg = ModelConfig(vocab_size=10, n_positions=8, n_embd=8, n_layer=1, n_head=2)
    with pytest.raises(ValueError, match="exceeds"):
        KVCache(cfg, 1, 9)
    with pytest.raises(ValueError, match="positive"):
        KVCache(cfg, 0, 4)


def test_prefill_rejects_empty_or_too_long_prompt(model: GPT) -> None:
    cache = KVCache(model.config, 1, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside"):
        model.prefill(torch.tensor([], dtype=torch.long), cache, 0)
    with pytest.raises(ValueError, match="outside"):
        model.prefill(torch.arange(9), cache, 0)


def test_random_init_is_seeded() -> None:
    cfg = ModelConfig(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2)
    a, b = GPT(cfg).init_random(3), GPT(cfg).init_random(3)
    for (_, p), (_, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(p, q)
    assert float(a.ln_f.weight.detach().mean()) == 1.0
