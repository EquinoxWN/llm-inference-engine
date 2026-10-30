"""A GPT-2 style decoder whose attention reads and writes an explicit, slot-based KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a GPT-2 style model."""

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        """Reject shapes the attention code cannot run."""
        if min(self.vocab_size, self.n_positions, self.n_embd, self.n_layer, self.n_head) < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.n_embd // self.n_head


class KVCache:
    """Preallocated keys and values: one contiguous row of max_len positions per slot."""

    def __init__(
        self,
        config: ModelConfig,
        max_slots: int,
        max_len: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Allocate the cache tensors once, up front."""
        if max_slots < 1 or max_len < 1:
            raise ValueError("max_slots and max_len must be positive")
        if max_len > config.n_positions:
            raise ValueError(
                f"max_len {max_len} exceeds the model's {config.n_positions} positions"
            )
        shape = (config.n_layer, max_slots, config.n_head, max_len, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_slots = max_slots
        self.max_len = max_len

    @property
    def nbytes(self) -> int:
        """Bytes held by the cache tensors."""
        return self.k.element_size() * self.k.nelement() * 2


class Attention(nn.Module):
    """Multi-head causal self-attention with GPT-2 parameter names."""

    def __init__(self, config: ModelConfig, layer: int) -> None:
        """Create the fused QKV and output projections."""
        super().__init__()
        self.layer = layer
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project to queries, keys and values shaped [B, H, T, D]."""
        b, t, _ = x.shape
        q, k, v = self.c_attn(x).split(self.n_head * self.head_dim, dim=2)
        shape = (b, t, self.n_head, self.head_dim)
        return (
            q.view(shape).transpose(1, 2),
            k.view(shape).transpose(1, 2),
            v.view(shape).transpose(1, 2),
        )

    def _out(self, y: Tensor) -> Tensor:
        """Merge heads and project back to the residual width."""
        b, _, t, _ = y.shape
        out: Tensor = self.c_proj(y.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim))
        return out

    def prefill(self, x: Tensor, cache: KVCache, slot: int) -> Tensor:
        """Causal attention over a whole prompt [1, T, C], storing its K/V in one slot."""
        q, k, v = self._qkv(x)
        t = x.shape[1]
        cache.k[self.layer, slot, :, :t] = k[0]
        cache.v[self.layer, slot, :, :t] = v[0]
        return self._out(F.scaled_dot_product_attention(q, k, v, is_causal=True))

    def decode(
        self, x: Tensor, cache: KVCache, slots: Tensor, positions: Tensor, span: int
    ) -> Tensor:
        """One new token per sequence [B, 1, C], attending to the first `span` cached positions."""
        q, k, v = self._qkv(x)
        cache.k[self.layer, slots, :, positions] = k[:, :, 0]
        cache.v[self.layer, slots, :, positions] = v[:, :, 0]
        keys = cache.k[self.layer, slots, :, :span]
        values = cache.v[self.layer, slots, :, :span]
        mask = torch.arange(span, device=x.device)[None, :] <= positions[:, None]
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask[:, None, None, :])
        return self._out(y)


class MLP(nn.Module):
    """GPT-2 feed-forward block with the tanh GELU."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the expand and project layers."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward block."""
        out: Tensor = self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))
        return out


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelConfig, layer: int) -> None:
        """Create norms, attention and MLP."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, layer)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def prefill(self, x: Tensor, cache: KVCache, slot: int) -> Tensor:
        """Block forward for a prompt."""
        x = x + self.attn.prefill(self.ln_1(x), cache, slot)
        out: Tensor = x + self.mlp(self.ln_2(x))
        return out

    def decode(
        self, x: Tensor, cache: KVCache, slots: Tensor, positions: Tensor, span: int
    ) -> Tensor:
        """Block forward for one decoding step."""
        x = x + self.attn.decode(self.ln_1(x), cache, slots, positions, span)
        out: Tensor = x + self.mlp(self.ln_2(x))
        return out


class GPT(nn.Module):
    """GPT-2 language model; the output head is tied to the token embedding."""

    def __init__(self, config: ModelConfig) -> None:
        """Create embeddings, blocks and the final norm."""
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(Block(config, i) for i in range(config.n_layer))
        self.blocks: list[Block] = [b for b in self.h if isinstance(b, Block)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def init_random(self, seed: int) -> GPT:
        """Fill weights with GPT-2's initialisation from a fixed seed (for tests and CI)."""
        gen = torch.Generator().manual_seed(seed)
        proj_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith("bias"):
                    p.zero_()
                elif ".ln_" in name or name.startswith("ln_f"):
                    p.fill_(1.0)
                else:
                    std = proj_std if name.endswith("c_proj.weight") else 0.02
                    p.copy_(torch.randn(p.shape, generator=gen) * std)
        return self

    def _logits(self, x: Tensor) -> Tensor:
        """Final norm and tied output projection."""
        return F.linear(self.ln_f(x), self.wte.weight)

    @torch.inference_mode()
    def prefill(self, tokens: Tensor, cache: KVCache, slot: int) -> Tensor:
        """Run a prompt [T] into a cache slot; return the next-token logits [V]."""
        t = tokens.shape[0]
        if not 0 < t <= cache.max_len:
            raise ValueError(f"prompt length {t} outside 1..{cache.max_len}")
        pos = torch.arange(t, device=tokens.device)
        x = self.wte(tokens)[None] + self.wpe(pos)[None]
        for block in self.blocks:
            x = block.prefill(x, cache, slot)
        return self._logits(x[0, -1])

    @torch.inference_mode()
    def decode(
        self,
        tokens: Tensor,
        slots: Tensor,
        positions: Tensor,
        cache: KVCache,
        span: int | None = None,
    ) -> Tensor:
        """Advance B sequences by one token each; return logits [B, V]."""
        if span is None:  # callers that know the lengths pass span and avoid a device sync
            span = int(positions.max().item()) + 1
        x = (self.wte(tokens) + self.wpe(positions))[:, None, :]
        for block in self.blocks:
            x = block.decode(x, cache, slots, positions, span)
        return self._logits(x[:, 0])

    @torch.inference_mode()
    def forward_full(self, tokens: Tensor) -> Tensor:
        """Logits for every position of [B, T] without any cache (reference path for tests)."""
        _b, t = tokens.shape
        x = self.wte(tokens) + self.wpe(torch.arange(t, device=tokens.device))[None]
        for block in self.blocks:
            a = block.attn
            q, k, v = a._qkv(block.ln_1(x))
            x = x + a._out(F.scaled_dot_product_attention(q, k, v, is_causal=True))
            x = x + block.mlp(block.ln_2(x))
        return self._logits(x)
