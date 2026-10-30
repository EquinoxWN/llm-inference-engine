"""Offline load generator: replays one seeded workload through each scheduling policy."""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from .engine import Engine, Policy, Request
from .model import GPT
from .sampling import SamplingParams


@dataclass(frozen=True)
class Workload:
    """A seeded set of requests: prompt and output lengths drawn uniformly, Poisson arrivals."""

    n_requests: int = 64
    prompt_len: tuple[int, int] = (16, 128)
    output_len: tuple[int, int] = (16, 256)
    rate: float = 0.0  # requests per second; 0 means every request arrives at t=0
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject empty or inverted ranges."""
        if self.n_requests < 1:
            raise ValueError("n_requests must be >= 1")
        for lo, hi in (self.prompt_len, self.output_len):
            if not 1 <= lo <= hi:
                raise ValueError("length ranges must satisfy 1 <= min <= max")
        if self.rate < 0:
            raise ValueError("rate must be >= 0")

    def requests(self, vocab_size: int) -> list[tuple[float, Request]]:
        """(arrival offset in seconds, request) pairs; EOS is ignored so lengths are exact."""
        rng = random.Random(self.seed)  # noqa: S311 - workload shape, not security
        out: list[tuple[float, Request]] = []
        t = 0.0
        for i in range(self.n_requests):
            if self.rate > 0 and i > 0:
                t += rng.expovariate(self.rate)
            prompt = [rng.randrange(vocab_size) for _ in range(rng.randint(*self.prompt_len))]
            params = SamplingParams(max_new_tokens=rng.randint(*self.output_len), stop_at_eos=False)
            out.append((t, Request(f"r{i}", prompt, params)))
        return out


@dataclass(frozen=True)
class BenchResult:
    """Aggregate numbers for one policy on one workload."""

    policy: str
    requests: int
    generated_tokens: int
    seconds: float
    tokens_per_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    steps: int
    decode_passes: int
    mean_decode_batch: float
    peak_mem_mib: float | None

    def as_dict(self) -> dict[str, object]:
        """Plain dict for JSON output."""
        return asdict(self)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = min(len(ordered), max(1, math.ceil(q / 100 * len(ordered))))
    return ordered[rank - 1]


def replay(
    engine: Engine,
    timed: list[tuple[float, Request]],
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> BenchResult:
    """Feed requests at their arrival times and time every token."""
    pending = sorted(timed, key=lambda x: x[0])
    start = clock()
    arrival: dict[str, float] = {}
    first: dict[str, float] = {}
    done: dict[str, float] = {}
    tokens = 0
    i = 0
    while i < len(pending) or engine.has_work():
        now = clock() - start
        while i < len(pending) and pending[i][0] <= now:
            offset, req = pending[i]
            engine.add_request(req)
            arrival[req.request_id] = offset
            i += 1
        if not engine.has_work():
            sleep(max(0.0, pending[i][0] - now))
            continue
        events = engine.step()
        t = clock() - start
        for e in events:
            tokens += 1
            first.setdefault(e.request_id, t)
            if e.finish_reason is not None:
                done[e.request_id] = t
    makespan = max(done.values()) - min(arrival.values())
    ttft = [(first[r] - arrival[r]) * 1000 for r in arrival]
    lat = [(done[r] - arrival[r]) * 1000 for r in arrival]
    peak = None
    if engine.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(engine.device) / 2**20
    return BenchResult(
        policy=engine.policy.value,
        requests=len(arrival),
        generated_tokens=tokens,
        seconds=makespan,
        tokens_per_s=tokens / makespan if makespan > 0 else float("inf"),
        ttft_p50_ms=statistics.median(ttft),
        ttft_p95_ms=percentile(ttft, 95),
        latency_p50_ms=statistics.median(lat),
        latency_p95_ms=percentile(lat, 95),
        steps=engine.stats.steps,
        decode_passes=engine.stats.decode_passes,
        mean_decode_batch=engine.stats.mean_decode_batch,
        peak_mem_mib=peak,
    )


def run_policy(
    model: GPT, policy: Policy, workload: Workload, max_batch: int, max_seq_len: int
) -> BenchResult:
    """Warm up, then replay the workload through a fresh engine with the given policy."""
    warm = Engine(model, policy, max_batch=max_batch, max_seq_len=max_seq_len)
    warm.generate(
        [Request(f"w{i}", [1, 2, 3, 4], SamplingParams(max_new_tokens=8)) for i in range(2)]
    )
    del warm
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    engine = Engine(model, policy, max_batch=max_batch, max_seq_len=max_seq_len)
    return replay(engine, workload.requests(model.config.vocab_size))


def table(results: list[BenchResult]) -> str:
    """Markdown table of the results, one row per policy."""
    head = (
        "| policy | tokens/s | TTFT p50 ms | TTFT p95 ms | latency p50 ms | latency p95 ms "
        "| decode passes | mean batch | peak MiB |\n|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r.policy} | {r.tokens_per_s:.1f} | {r.ttft_p50_ms:.1f} | {r.ttft_p95_ms:.1f} "
        f"| {r.latency_p50_ms:.1f} | {r.latency_p95_ms:.1f} | {r.decode_passes} "
        f"| {r.mean_decode_batch:.2f} | {'-' if r.peak_mem_mib is None else f'{r.peak_mem_mib:.0f}'} |"
        for r in results
    ]
    return "\n".join([head, *rows])
