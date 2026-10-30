# ADR 0002: Slot-based KV cache with iteration-level scheduling for M1

- **Status:** Accepted

## Context

Continuous batching needs each running sequence to keep its own keys and values while other
sequences join and leave the batch at every step. The cache layout decides what a join or a
leave costs, and how much memory sits unused. M1 has to show the scheduling effect; M2 is
reserved for memory efficiency (paged blocks).

## Decision

The KV cache is one preallocated tensor per K and V, shaped
`[layers, max_batch, heads, max_seq_len, head_dim]`. A request is bound to a free slot when it is
admitted and keeps it until it finishes. A decode pass gathers the running slots, writes each new
key and value at that sequence's own position, and masks each row by its length. The scheduler
refills free slots at every step (`continuous`), and the same engine also runs the `naive` and
`static` policies so all three are measured with identical model code.

## Consequences

- Joining or leaving costs nothing beyond a list update: no cache copy, no re-padding, no
  recompute of other sequences.
- Memory is reserved for the worst case: every slot holds `max_seq_len` positions even when the
  request is short. With GPT-2 in float16, 16 slots x 512 positions take 288 MiB of KV cache
  whether or not they are used. This is the waste that M2's paged blocks remove.
- The decode pass gathers each running slot's prefix (an index copy per layer per step), and
  attention spans the longest running prefix for every row. Acceptable for M1; paged attention
  changes this in M2.
- Batch size is capped by the number of slots fixed at start-up, not by actual memory use, so the
  engine cannot admit more short requests when memory would allow it (M2/M3).
