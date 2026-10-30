"""The load generator: seeded workloads, correct timing and per-policy results."""

from __future__ import annotations

import itertools

import pytest

from llm_inference_engine.bench import Workload, percentile, replay, run_policy, table
from llm_inference_engine.engine import Engine, Policy
from llm_inference_engine.model import GPT


def test_workload_is_seeded_and_within_ranges() -> None:
    w = Workload(n_requests=20, prompt_len=(2, 5), output_len=(3, 4), rate=10.0, seed=4)
    a, b = w.requests(257), w.requests(257)
    assert a == b
    assert a[0][0] == 0.0
    assert all(x[0] < y[0] for x, y in itertools.pairwise(a))
    for _, r in a:
        assert 2 <= len(r.prompt_ids) <= 5
        assert 3 <= r.params.max_new_tokens <= 4
        assert not r.params.stop_at_eos
    assert all(t == 0.0 for t, _ in Workload(n_requests=5).requests(257))


def test_workload_validation() -> None:
    with pytest.raises(ValueError, match="n_requests"):
        Workload(n_requests=0)
    with pytest.raises(ValueError, match="ranges"):
        Workload(prompt_len=(5, 2))
    with pytest.raises(ValueError, match="rate"):
        Workload(rate=-1)


def test_percentile_nearest_rank() -> None:
    vals = [float(v) for v in range(1, 101)]
    assert percentile(vals, 50) == 50.0
    assert percentile(vals, 95) == 95.0
    assert percentile([3.0], 95) == 3.0
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


class FakeClock:
    """Time advances 1 ms per engine step and jumps on sleep."""

    def __init__(self) -> None:
        """Start at zero."""
        self.t = 0.0

    def __call__(self) -> float:
        """Read the clock (each read is 1 ms later)."""
        self.t += 0.001
        return self.t

    def sleep(self, s: float) -> None:
        """Jump forward."""
        self.t += s


def test_replay_measures_from_arrival_and_idles_between_requests(model: GPT) -> None:
    timed = Workload(n_requests=3, prompt_len=(2, 2), output_len=(4, 4), rate=0.5).requests(257)
    clock = FakeClock()
    res = replay(Engine(model, Policy.CONTINUOUS, max_seq_len=32), timed, clock, clock.sleep)
    assert res.requests == 3
    assert res.generated_tokens == 12
    assert res.ttft_p50_ms < 10  # first token right after each arrival, not after the gap
    assert res.seconds > timed[-1][0]


def test_policies_generate_the_same_tokens_and_continuous_needs_fewest_passes(model: GPT) -> None:
    w = Workload(n_requests=12, prompt_len=(4, 16), output_len=(2, 30), seed=1)
    results = {p: run_policy(model, p, w, max_batch=4, max_seq_len=64) for p in Policy}
    expected = sum(r.params.max_new_tokens for _, r in w.requests(257))
    assert {r.generated_tokens for r in results.values()} == {expected}
    assert results[Policy.NAIVE].mean_decode_batch == 1.0
    assert (
        results[Policy.CONTINUOUS].decode_passes
        < results[Policy.STATIC].decode_passes
        < results[Policy.NAIVE].decode_passes
    )
    assert results[Policy.CONTINUOUS].peak_mem_mib is None  # CPU run
    md = table(list(results.values()))
    assert md.count("\n") == 4
    assert "| continuous |" in md
