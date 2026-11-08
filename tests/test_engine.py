"""Scheduling: continuous batching gives the same tokens as the naive baseline, with fewer passes."""

from __future__ import annotations

import random

import pytest
import torch

from llm_inference_engine.engine import Engine, Policy, Request
from llm_inference_engine.model import GPT
from llm_inference_engine.sampling import SamplingParams, sample_rows


def _requests(seed: int, n: int, sampled: bool = False) -> list[Request]:
    """Random prompts and output lengths; EOS ignored so lengths are exact."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        prompt = [rng.randrange(256) for _ in range(rng.randint(1, 30))]
        params = SamplingParams(
            max_new_tokens=rng.randint(1, 25),
            temperature=0.9 if sampled else 0.0,
            top_k=20 if sampled else 0,
            seed=rng.randrange(10**6),
            stop_at_eos=False,
        )
        out.append(Request(f"q{i}", prompt, params))
    return out


def _run_staggered(engine: Engine, reqs: list[Request], seed: int) -> dict[str, list[int]]:
    """Add requests at random steps while others are running."""
    rng = random.Random(seed)
    pending = list(reqs)
    out: dict[str, list[int]] = {r.request_id: [] for r in reqs}
    while pending or engine.has_work():
        for _ in range(rng.randint(0, 2)):
            if pending:
                engine.add_request(pending.pop(0))
        for e in engine.step():
            out[e.request_id].append(e.token_id)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_continuous_matches_naive_greedy_with_random_arrivals(model: GPT, seed: int) -> None:
    reqs = _requests(seed, 12)
    naive = Engine(model, Policy.NAIVE, max_seq_len=64).generate(reqs)
    cont = _run_staggered(Engine(model, Policy.CONTINUOUS, max_batch=4, max_seq_len=64), reqs, seed)
    assert cont == naive
    assert all(len(naive[r.request_id]) == r.params.max_new_tokens for r in reqs)


def test_sampling_is_per_request_and_independent_of_batching(model: GPT) -> None:
    reqs = _requests(7, 10, sampled=True)
    naive = Engine(model, Policy.NAIVE, max_seq_len=64).generate(reqs)
    static = Engine(model, Policy.STATIC, max_batch=3, max_seq_len=64).generate(reqs)
    cont = Engine(model, Policy.CONTINUOUS, max_batch=5, max_seq_len=64).generate(reqs)
    assert naive == static == cont
    greedy = Engine(model, Policy.NAIVE, max_seq_len=64).generate(_requests(7, 10))
    assert greedy != naive


def test_requests_join_and_leave_at_every_step(model: GPT) -> None:
    eng = Engine(model, Policy.CONTINUOUS, max_batch=2, max_seq_len=64)
    p = SamplingParams(max_new_tokens=3, stop_at_eos=False)
    long = SamplingParams(max_new_tokens=8, stop_at_eos=False)
    eng.add_request(Request("short", [1, 2], p))
    eng.add_request(Request("long", [3, 4], long))
    eng.add_request(Request("late", [5], p))
    trace = []
    while eng.has_work():
        events = eng.step()
        trace.append(({e.request_id for e in events}, len(eng.running), len(eng.waiting)))
    # step 1: both prefill; step 3: "short" finishes and frees its slot; step 4: "late" joins.
    assert trace[0] == ({"short", "long"}, 2, 1)
    assert trace[2] == ({"short", "long"}, 1, 1)
    assert trace[3] == ({"late", "long"}, 2, 0)
    assert eng.stats.max_running == 2
    assert eng.stats.finished == 3
    assert sorted(eng.free_slots) == [0, 1]


def test_static_batch_waits_until_the_whole_batch_drains(model: GPT) -> None:
    eng = Engine(model, Policy.STATIC, max_batch=2, max_seq_len=64)
    eng.add_request(Request("a", [1], SamplingParams(max_new_tokens=2, stop_at_eos=False)))
    eng.add_request(Request("b", [2], SamplingParams(max_new_tokens=6, stop_at_eos=False)))
    eng.add_request(Request("c", [3], SamplingParams(max_new_tokens=2, stop_at_eos=False)))
    first_c = None
    step = 0
    while eng.has_work():
        step += 1
        if any(e.request_id == "c" for e in eng.step()) and first_c is None:
            first_c = step
    assert first_c == 7  # "b" occupies the batch for 6 steps although a slot is free from step 2


def test_decode_pass_counts_per_policy(model: GPT) -> None:
    reqs = _requests(3, 16)
    passes = {}
    for pol in Policy:
        eng = Engine(model, pol, max_batch=4, max_seq_len=64)
        eng.generate(reqs)
        passes[pol] = eng.stats.decode_passes
        assert eng.stats.prefills == len(reqs)
    assert passes[Policy.NAIVE] == sum(r.params.max_new_tokens - 1 for r in reqs)
    assert passes[Policy.CONTINUOUS] < passes[Policy.STATIC] < passes[Policy.NAIVE]


def test_eos_stops_a_request(model: GPT) -> None:
    probe = Engine(model, Policy.NAIVE, max_seq_len=64)
    first = probe.generate([Request("p", [10, 20], SamplingParams(max_new_tokens=2))])["p"][0]
    eng = Engine(model, Policy.CONTINUOUS, max_seq_len=64, eos_token_id=first)
    eng.add_request(Request("x", [10, 20], SamplingParams(max_new_tokens=10)))
    events = eng.step()
    assert [(e.token_id, e.finish_reason) for e in events] == [(first, "stop")]
    assert not eng.has_work()


def test_request_validation(model: GPT) -> None:
    eng = Engine(model, Policy.CONTINUOUS, max_seq_len=32, max_waiting=2)
    with pytest.raises(ValueError, match="empty"):
        eng.add_request(Request("e", []))
    with pytest.raises(ValueError, match="exceeds"):
        eng.add_request(Request("l", [1] * 26, SamplingParams(max_new_tokens=7)))
    with pytest.raises(ValueError, match="token ids"):
        eng.add_request(Request("v", [1, 999]))
    eng.add_request(Request("a", [1]))
    with pytest.raises(ValueError, match="duplicate"):
        eng.add_request(Request("a", [2]))
    eng.add_request(Request("b", [1]))
    with pytest.raises(OverflowError):
        eng.add_request(Request("c", [1]))
    with pytest.raises(ValueError, match="max_batch"):
        Engine(model, Policy.CONTINUOUS, max_batch=0)


def test_abort_frees_the_slot(model: GPT) -> None:
    eng = Engine(model, Policy.CONTINUOUS, max_batch=1, max_seq_len=64)
    eng.add_request(Request("a", [1], SamplingParams(max_new_tokens=50, stop_at_eos=False)))
    eng.add_request(Request("b", [2], SamplingParams(max_new_tokens=2, stop_at_eos=False)))
    eng.step()
    assert eng.abort("a")
    assert eng.abort("b")
    assert not eng.abort("zzz")
    assert not eng.has_work()
    assert eng.free_slots == [0]
    assert eng.stats.aborted == 2


def test_sampling_params_validation() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        SamplingParams(max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-1)
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(top_k=-1)


def test_request_validation_boundaries(model: GPT) -> None:
    eng = Engine(model, Policy.CONTINUOUS, max_seq_len=32)
    eng.add_request(Request("fits", [1] * 25, SamplingParams(max_new_tokens=7)))  # exactly 32
    eng.add_request(Request("top", [model.config.vocab_size - 1]))
    with pytest.raises(ValueError, match="token ids"):
        eng.add_request(Request("over", [model.config.vocab_size]))
    with pytest.raises(ValueError, match="token ids"):
        eng.add_request(Request("neg", [-1]))


def test_top_k_one_samples_the_greedy_token(model: GPT) -> None:
    reqs = _requests(5, 4)
    greedy = Engine(model, Policy.CONTINUOUS, max_seq_len=64).generate(reqs)
    top1 = [
        Request(
            r.request_id,
            r.prompt_ids,
            SamplingParams(
                max_new_tokens=r.params.max_new_tokens,
                temperature=1.5,
                top_k=1,
                seed=r.params.seed,
                stop_at_eos=False,
            ),
        )
        for r in reqs
    ]
    assert Engine(model, Policy.CONTINUOUS, max_seq_len=64).generate(top1) == greedy


def test_temperature_scales_the_logits() -> None:
    logits = torch.tensor([[0.0, 1.0]])

    def draws(t: float) -> list[int]:
        """Token picked under 200 seeds at temperature t."""
        p = SamplingParams(temperature=t, seed=0)
        return [sample_rows(logits, [p], [torch.Generator().manual_seed(s)])[0] for s in range(200)]

    assert draws(0.05) == [1] * 200  # e^-20: practically never token 0
    assert 0 < draws(1.0).count(0) < 200
