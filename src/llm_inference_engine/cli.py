"""Command line: serve, generate, bench and verify."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .bench import Workload, run_policy, table
from .engine import Engine, Policy, Request
from .model import GPT, ModelConfig
from .sampling import SamplingParams
from .tokenizer import ByteTokenizer, Tokenizer
from .weights import DEFAULT_MODEL, DEFAULT_REVISION, download

TINY = "tiny-random"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def tiny_model(seed: int = 0) -> GPT:
    """A 2-layer randomly initialised model over bytes: runs anywhere, needs no download."""
    cfg = ModelConfig(vocab_size=257, n_positions=512, n_embd=64, n_layer=2, n_head=4)
    return GPT(cfg).init_random(seed).eval()


def load(name: str, revision: str, device: str, dtype: str) -> tuple[GPT, Tokenizer, str]:
    """Model, tokenizer and display name for --model."""
    if name == TINY:
        return tiny_model().to(device=device, dtype=DTYPES[dtype]), ByteTokenizer(), TINY
    ckpt = download(name, revision)
    return ckpt.model(device, DTYPES[dtype]), ckpt.tokenizer(), f"{name}@{revision[:7]}"


def _common(p: argparse.ArgumentParser) -> None:
    """Model and device options shared by every subcommand."""
    p.add_argument("--model", default=TINY, help=f"'{TINY}' or a GPT-2 repo id ({DEFAULT_MODEL})")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="full commit hash of the model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=512)


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    ap = argparse.ArgumentParser(prog="lie", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP API")
    _common(s)
    s.add_argument("--policy", choices=[p.value for p in Policy], default="continuous")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    g = sub.add_parser("generate", help="complete one prompt")
    _common(g)
    g.add_argument("prompt")
    g.add_argument("--max-tokens", type=int, default=32)
    b = sub.add_parser("bench", help="replay one workload through each policy")
    _common(b)
    b.add_argument("--policies", default="naive,static,continuous")
    b.add_argument("--requests", type=int, default=64)
    b.add_argument("--prompt-len", type=int, nargs=2, default=(16, 128), metavar=("MIN", "MAX"))
    b.add_argument("--output-len", type=int, nargs=2, default=(16, 256), metavar=("MIN", "MAX"))
    b.add_argument("--rate", type=float, default=0.0, help="requests/s, 0 = all at once")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--json", type=Path, help="also write the results here")
    v = sub.add_parser("verify", help="compare logits and greedy output with transformers")
    _common(v)
    v.add_argument("--tokens", type=int, default=32)
    return ap


PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns",
    "def fibonacci(n):",
    "Continuous batching lets a server",
]


def verify(model: GPT, tokenizer: Tokenizer, directory: Path, tokens: int, device: str) -> int:
    """Check this engine against the transformers implementation of the same checkpoint."""
    from transformers import GPT2LMHeadModel

    ref: Any = GPT2LMHeadModel.from_pretrained(str(directory))
    ref = ref.to(device).eval()
    dtype = next(model.parameters()).dtype
    ref = ref.to(dtype)
    worst = 0.0
    ids = [tokenizer.encode(p) for p in PROMPTS]
    for x in ids:
        t = torch.tensor([x], device=device)
        with torch.inference_mode():
            diff = (model.forward_full(t) - ref(t).logits).abs().max().item()
        worst = max(worst, diff)
    engine = Engine(model, Policy.CONTINUOUS, max_batch=len(ids), max_seq_len=256)
    params = SamplingParams(max_new_tokens=tokens, stop_at_eos=False)
    ours = engine.generate([Request(str(i), x, params) for i, x in enumerate(ids)])
    same = 0
    for i, x in enumerate(ids):
        t = torch.tensor([x], device=device)
        with torch.inference_mode():
            out = ref.generate(
                t,
                attention_mask=torch.ones_like(t),
                max_new_tokens=tokens,
                min_new_tokens=tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        theirs = out[0, len(x) :].tolist()
        ok = theirs == ours[str(i)]
        same += ok
        print(f"{'MATCH' if ok else 'DIFF '} {PROMPTS[i]!r} -> {tokenizer.decode(ours[str(i)])!r}")
    print(f"max |logit difference| over prompt positions: {worst:.2e}")
    print(
        f"greedy continuations identical to transformers: {same}/{len(ids)} ({tokens} tokens each)"
    )
    return 0 if same == len(ids) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `lie`."""
    args = _parser().parse_args(argv)
    if args.max_batch < 1 or not 1 <= args.max_seq_len <= 1024:
        print("error: --max-batch must be >= 1 and --max-seq-len in 1..1024", file=sys.stderr)
        return 2
    model, tok, name = load(args.model, args.revision, args.device, args.dtype)
    if args.cmd == "serve":
        import uvicorn

        from .server import create_app

        def factory() -> Engine:
            """Engine for the server's lifetime."""
            return Engine(
                model, Policy(args.policy), args.max_batch, args.max_seq_len, tok.eos_token_id
            )

        uvicorn.run(create_app(factory, tok, name), host=args.host, port=args.port)
        return 0
    if args.cmd == "generate":
        engine = Engine(model, Policy.NAIVE, 1, args.max_seq_len, tok.eos_token_id)
        params = SamplingParams(max_new_tokens=args.max_tokens)
        out = engine.generate([Request("0", tok.encode(args.prompt), params)])
        print(args.prompt + tok.decode(out["0"]))
        return 0
    if args.cmd == "verify":
        if args.model == TINY:
            print(
                "error: verify needs a real checkpoint (--model openai-community/gpt2)",
                file=sys.stderr,
            )
            return 2
        return verify(
            model, tok, download(args.model, args.revision).directory, args.tokens, args.device
        )
    workload = Workload(
        args.requests, tuple(args.prompt_len), tuple(args.output_len), args.rate, args.seed
    )
    if workload.prompt_len[1] + workload.output_len[1] > args.max_seq_len:
        print("error: max prompt + max output length exceeds --max-seq-len", file=sys.stderr)
        return 2
    print(
        f"model {name} on {args.device} ({args.dtype}), max batch {args.max_batch}; "
        f"{workload.n_requests} requests, prompt {workload.prompt_len[0]}-{workload.prompt_len[1]}"
        f" tokens, output {workload.output_len[0]}-{workload.output_len[1]} tokens, "
        f"rate {'all at t=0' if workload.rate == 0 else f'{workload.rate}/s'}, seed {args.seed}"
    )
    results = []
    for pol in args.policies.split(","):
        results.append(run_policy(model, Policy(pol), workload, args.max_batch, args.max_seq_len))
        r = results[-1]
        print(f"  {r.policy:<10} {r.generated_tokens} tokens in {r.seconds:.2f} s", flush=True)
    print(table(results))
    if args.json:
        args.json.write_text(json.dumps([r.as_dict() for r in results], indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
