# RFC 0001: llm-inference-engine design

- **Status:** Accepted for M1 (M2 and M3 open)
- **Author:** EquinoxWN
- **Created:** 2026-10-23

## Problem

A language model generates one token per forward pass. Served one request at a time, each pass
moves the full set of weights through the GPU to produce a single token for a single user, so the
GPU is mostly waiting on memory and the throughput per GPU is low. Batching requests together
shares that weight traffic, but a fixed (static) batch has a second problem: requests finish at
different lengths, and a batch keeps its slots until the longest request is done. Short requests
wait behind long ones, and new requests wait for the whole batch.

This repository builds the serving loop from first principles in PyTorch to show where the cost
comes from and how much each scheduling step removes, measured on one open model on one laptop GPU.

**Who it is for:** engineers who run or evaluate LLM serving stacks (vLLM, SGLang, TGI) and want to
understand, and measure, what iteration-level scheduling does before tuning one of them.

## Critical-to-quality measures (Define)

| CTQ | Target for M1 | How it is checked |
|---|---|---|
| Correctness of the engine | Same logits as the reference implementation (max abs difference below 1e-3 in float32) and identical greedy tokens | `lie verify` against `transformers` on GPT-2; `test_weights.py` on a random GPT-2 |
| Batching never changes output | Continuous and static batching produce exactly the tokens of the naive server, for greedy and for seeded sampling | `test_engine.py` with random arrival schedules (float64) |
| Throughput | Continuous batching at least 3x the naive server's tokens/s on GPT-2 with 64 queued requests | `make bench-gpu` |
| Scheduling | Continuous batching uses fewer decode passes than static batching for the same work | decode-pass counters, `test_bench.py` |
| Bounded inputs | Oversized prompts or outputs rejected with HTTP 400/422; full queue returns 503 | `test_server.py`, `test_engine.py` |

## Goals

- M1: a naive server that handles one request at a time, as the baseline.
- M1: continuous batching that adds and removes requests at every decoding step.
- M1: an OpenAI-style completions API with token streaming, and a load generator that measures
  tokens/s, time to first token (TTFT), end-to-end latency and peak GPU memory per policy.
- M2: a paged KV cache (fixed-size blocks and a block table) and prefix sharing.
- M3: preemption when memory runs out, and a comparison with vLLM on the same model.

## Non-goals

- Running as a hosted production service (no authentication, no multi-GPU, no quantisation).
- Custom CUDA kernels: attention uses PyTorch's `scaled_dot_product_attention`.
- Models other than the GPT-2 family in M1.

## Design

```
HTTP (FastAPI)  ->  AsyncEngine (asyncio loop)  ->  Engine.step() in a worker thread
                       inbox / aborts              admit -> prefill new -> one batched decode -> retire
                                                        |
                                                   KVCache: [layers, slots, heads, max_len, head_dim]
```

- **Model (`model.py`).** A GPT-2 decoder written against an explicit KV cache. `prefill` runs a
  whole prompt with causal attention and writes its keys and values into one cache slot. `decode`
  takes one token for each of B running sequences, writes their keys and values at each sequence's
  own position, and attends over each slot's prefix with a per-row length mask, so sequences of
  different lengths share one forward pass. Weights load from the Hugging Face safetensors
  checkpoint (Conv1D weights transposed, strict load).
- **KV cache.** Allocated once: `max_batch` slots of `max_seq_len` positions. A sequence owns a
  slot from admission to finish. This reserves memory for the maximum length per slot; M2 replaces
  it with paged blocks.
- **Scheduler (`engine.py`).** One `step()` is one iteration: admit waiting requests into free
  slots (policy-dependent), prefill each admitted request (which yields its first token), run one
  batched decode for every sequence that was already running, then retire finished sequences and
  free their slots. Policies:
  - `naive`: one slot; a request runs from start to finish before the next one starts.
  - `static`: requests are admitted only when the batch is empty, and the batch drains together.
  - `continuous`: free slots are refilled at every step (iteration-level scheduling, as in Orca).
- **Sampling (`sampling.py`).** Greedy, or temperature plus top-k with a CPU generator seeded per
  request, so a request's tokens do not depend on what else is in the batch.
- **API (`server.py`).** `POST /v1/completions` (OpenAI text-completion shape, optional SSE
  streaming), `GET /healthz`, `GET /stats`. Only the asyncio loop task touches the engine; handlers
  put requests and aborts in an inbox that is drained between steps, so no locks are needed. A
  client that disconnects mid-stream aborts its request and frees its slot.
- **Load generator (`bench.py`).** A seeded workload (uniform prompt and output lengths, all at
  t=0 or Poisson arrivals) replayed through each policy on the same model, EOS ignored so every
  policy generates exactly the same number of tokens.

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Wrap `transformers` `generate()` and batch at the request level | `generate()` runs a fixed batch to completion, which is static batching; joining a request mid-batch needs control of each sequence's cache and position, which is the point of this project. `transformers` is kept as the correctness reference. |
| Pad every running sequence into one dense `[B, T]` cache and recompute the batch on each change | Simple, but each join or leave reshapes or copies the cache, and padding grows with the longest sequence. A fixed slot per sequence with a length mask costs no copies on join or leave. |
| Paged KV cache from the start (PagedAttention) | The bigger memory win, but it needs a block allocator, block tables and gather-based attention; M1 isolates the scheduling effect first so M2 can measure the paging effect on its own. |
| Batched prefill of all newly admitted prompts | Fewer passes for many short prompts, but needs padding or varlen kernels; M1 prefills one prompt at a time and records the cost in TTFT. Chunked prefill is an M2/M3 candidate. |

## Measurement plan

- Correctness first: `lie verify` (logits and 32 greedy tokens against `transformers`), then the
  engine tests (batched output equals naive output).
- Performance: `lie bench` runs the same seeded workload through `naive`, `static` and
  `continuous` on GPT-2 (124M) in float16 on the RTX 4050 Laptop GPU (6 GB), with all 64 requests
  queued at t=0 (throughput) and with Poisson arrivals at 4 requests/s (latency under load).
  Recorded: generated tokens/s, TTFT p50/p95, end-to-end latency p50/p95, decode passes, mean decode
  batch size, peak allocated GPU memory. Each policy gets a warm-up pass first.
- M3 adds vLLM on the same model and workload.

## Milestones

- **M1:** naive baseline server; continuous batching; API, verification and benchmark.
- **M2:** paged KV cache with a block table; prefix sharing.
- **M3:** preemption under memory pressure; comparison with the naive server and with vLLM.

## Risks and open questions

- Python overhead per step is significant for a 124M model; the measured speed-up reflects the
  scheduler plus PyTorch eager overhead, not kernel quality. CUDA graphs are out of scope for M1.
- One-prompt-at-a-time prefill stalls the decode batch when many requests arrive together; TTFT
  numbers show this cost.
- The slot cache reserves `max_seq_len` positions for every slot; with 16 slots of 512 positions
  that is most of the KV memory, whatever the real lengths are (the M2 problem).
