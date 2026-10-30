"""Sampling parameters and next-token selection with a per-request random generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

MAX_NEW_TOKENS = 4096


@dataclass(frozen=True)
class SamplingParams:
    """How one request picks its tokens and when it stops."""

    max_new_tokens: int = 16
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 0
    stop_at_eos: bool = True

    def __post_init__(self) -> None:
        """Reject values that would loop forever or break sampling."""
        if not 1 <= self.max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be in 1..{MAX_NEW_TOKENS}")
        if not 0.0 <= self.temperature <= 10.0:
            raise ValueError("temperature must be in 0..10")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

    @property
    def greedy(self) -> bool:
        """True when the request always takes the most likely token."""
        return self.temperature == 0.0


def make_generator(params: SamplingParams) -> torch.Generator | None:
    """A CPU generator seeded for this request, or None for greedy decoding."""
    return None if params.greedy else torch.Generator().manual_seed(params.seed)


def sample_rows(
    logits: Tensor, params: list[SamplingParams], gens: list[torch.Generator | None]
) -> list[int]:
    """Pick one token per row of logits [B, V]; each row uses its own params and generator."""
    picked = logits.argmax(dim=-1).tolist()
    for i, (p, gen) in enumerate(zip(params, gens, strict=True)):
        if p.greedy:
            continue
        row = logits[i].float().cpu() / p.temperature
        if 0 < p.top_k < row.shape[0]:
            kth = torch.topk(row, p.top_k).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))
        picked[i] = int(torch.multinomial(torch.softmax(row, dim=-1), 1, generator=gen).item())
    return [int(t) for t in picked]
