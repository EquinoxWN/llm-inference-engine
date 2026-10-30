"""The scheduler: naive (one request at a time), static batching and continuous batching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import torch

from .model import GPT, KVCache
from .sampling import SamplingParams, make_generator, sample_rows


class Policy(StrEnum):
    """When waiting requests may join the running batch."""

    NAIVE = "naive"  # one request at a time, start to finish
    STATIC = "static"  # a batch is formed only when the previous one has fully drained
    CONTINUOUS = "continuous"  # requests join and leave at every decoding step


@dataclass(frozen=True)
class Request:
    """A tokenised prompt and how to sample its continuation."""

    request_id: str
    prompt_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)


@dataclass(frozen=True)
class TokenEvent:
    """One generated token; `finish_reason` is set on the request's last token."""

    request_id: str
    token_id: int
    index: int
    finish_reason: str | None = None


@dataclass
class _Sequence:
    """A running request bound to a KV-cache slot."""

    request: Request
    slot: int
    generator: torch.Generator | None
    generated: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Tokens in the cache once the last generated token is written."""
        return len(self.request.prompt_ids) + len(self.generated)


@dataclass
class EngineStats:
    """Counters that show how much work each forward pass did."""

    steps: int = 0
    prefills: int = 0
    decode_passes: int = 0
    decode_tokens: int = 0
    finished: int = 0
    aborted: int = 0
    max_running: int = 0

    @property
    def mean_decode_batch(self) -> float:
        """Average number of sequences per decode pass."""
        return self.decode_tokens / self.decode_passes if self.decode_passes else 0.0


class Engine:
    """Owns the model and KV cache and advances all running requests one step at a time."""

    def __init__(
        self,
        model: GPT,
        policy: Policy = Policy.CONTINUOUS,
        max_batch: int = 8,
        max_seq_len: int = 512,
        eos_token_id: int | None = None,
        max_waiting: int = 1024,
    ) -> None:
        """Allocate the KV cache for max_batch slots of max_seq_len tokens."""
        self.model = model
        self.policy = Policy(policy)
        self.max_batch = 1 if self.policy is Policy.NAIVE else max_batch
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if max_waiting < 1:
            raise ValueError("max_waiting must be >= 1")
        param = next(model.parameters())
        self.device = param.device
        self.cache = KVCache(model.config, self.max_batch, max_seq_len, param.device, param.dtype)
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.max_waiting = max_waiting
        self.waiting: deque[Request] = deque()
        self.running: list[_Sequence] = []
        self.free_slots = list(range(self.max_batch - 1, -1, -1))
        self.stats = EngineStats()

    def add_request(self, request: Request) -> None:
        """Validate a request and queue it."""
        n = len(request.prompt_ids)
        if n == 0:
            raise ValueError("prompt is empty")
        if n + request.params.max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({n}) + max_new_tokens ({request.params.max_new_tokens}) "
                f"exceeds max_seq_len {self.max_seq_len}"
            )
        vocab = self.model.config.vocab_size
        if min(request.prompt_ids) < 0 or max(request.prompt_ids) >= vocab:
            raise ValueError(f"token ids must be in 0..{vocab - 1}")
        if self._known(request.request_id):
            raise ValueError(f"duplicate request id {request.request_id!r}")
        if len(self.waiting) >= self.max_waiting:
            raise OverflowError("waiting queue is full")
        self.waiting.append(request)

    def _known(self, request_id: str) -> bool:
        """True if the id is waiting or running."""
        return any(r.request_id == request_id for r in self.waiting) or any(
            s.request.request_id == request_id for s in self.running
        )

    def abort(self, request_id: str) -> bool:
        """Drop a waiting or running request and free its slot."""
        for r in self.waiting:
            if r.request_id == request_id:
                self.waiting.remove(r)
                self.stats.aborted += 1
                return True
        for s in self.running:
            if s.request.request_id == request_id:
                self._release(s)
                self.stats.aborted += 1
                return True
        return False

    def has_work(self) -> bool:
        """True while any request is waiting or running."""
        return bool(self.waiting or self.running)

    def _release(self, seq: _Sequence) -> None:
        """Remove a sequence from the batch and return its slot."""
        self.running.remove(seq)
        self.free_slots.append(seq.slot)

    def _admit(self) -> list[_Sequence]:
        """Move waiting requests into free slots as the policy allows."""
        if self.policy is Policy.STATIC and self.running:
            return []
        admitted: list[_Sequence] = []
        while self.waiting and self.free_slots:
            req = self.waiting.popleft()
            seq = _Sequence(req, self.free_slots.pop(), make_generator(req.params))
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    def _emit(self, seq: _Sequence, token: int) -> TokenEvent:
        """Record a sampled token and decide whether the sequence is done."""
        seq.generated.append(token)
        p = seq.request.params
        reason: str | None = None
        if p.stop_at_eos and self.eos_token_id is not None and token == self.eos_token_id:
            reason = "stop"
        elif len(seq.generated) >= p.max_new_tokens:
            reason = "length"
        return TokenEvent(seq.request.request_id, token, len(seq.generated) - 1, reason)

    def step(self) -> list[TokenEvent]:
        """One scheduler iteration: prefill newly admitted requests, then one batched decode."""
        self.stats.steps += 1
        decoding = list(self.running)
        events: list[TokenEvent] = []
        for seq in self._admit():
            ids = torch.tensor(seq.request.prompt_ids, device=self.device)
            logits = self.model.prefill(ids, self.cache, seq.slot)
            token = sample_rows(logits[None], [seq.request.params], [seq.generator])[0]
            events.append(self._emit(seq, token))
            self.stats.prefills += 1
        self.stats.max_running = max(self.stats.max_running, len(self.running))
        if decoding:
            tokens = torch.tensor([s.generated[-1] for s in decoding], device=self.device)
            slots = torch.tensor([s.slot for s in decoding], device=self.device)
            positions = torch.tensor([s.length - 1 for s in decoding], device=self.device)
            span = max(s.length for s in decoding)
            logits = self.model.decode(tokens, slots, positions, self.cache, span)
            picked = sample_rows(
                logits, [s.request.params for s in decoding], [s.generator for s in decoding]
            )
            events.extend(self._emit(s, t) for s, t in zip(decoding, picked, strict=True))
            self.stats.decode_passes += 1
            self.stats.decode_tokens += len(decoding)
        done = {e.request_id for e in events if e.finish_reason is not None}
        for seq in [s for s in self.running if s.request.request_id in done]:
            self._release(seq)
            self.stats.finished += 1
        return events

    def generate(self, requests: list[Request]) -> dict[str, list[int]]:
        """Run requests to completion and return each one's generated ids (offline helper)."""
        for r in requests:
            self.add_request(r)
        out: dict[str, list[int]] = {r.request_id: [] for r in requests}
        while self.has_work():
            for e in self.step():
                out[e.request_id].append(e.token_id)
        return out
