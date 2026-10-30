"""The HTTP API: completions, SSE streaming, validation, capacity and batching across clients."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_inference_engine.engine import Engine, Policy, Request
from llm_inference_engine.model import GPT
from llm_inference_engine.server import AsyncEngine, _TextStream, create_app
from llm_inference_engine.tokenizer import ByteTokenizer


def _app(model: GPT, policy: Policy = Policy.CONTINUOUS, max_seq_len: int = 128) -> FastAPI:
    """App over the tiny model."""
    tok = ByteTokenizer()
    return create_app(
        lambda: Engine(model, policy, max_batch=4, max_seq_len=max_seq_len, eos_token_id=256),
        tok,
        "tiny",
    )


def test_completion_and_stream_return_the_same_text(model: GPT) -> None:
    with TestClient(_app(model)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        body = {"prompt": "hello", "max_tokens": 12, "ignore_eos": True}
        r = client.post("/v1/completions", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["usage"] == {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}
        assert data["choices"][0]["finish_reason"] == "length"
        with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as s:
            assert s.headers["content-type"].startswith("text/event-stream")
            lines = [ln for ln in s.iter_lines() if ln.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
        assert "".join(c["choices"][0]["text"] for c in chunks) == data["choices"][0]["text"]
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_validation_errors(model: GPT) -> None:
    with TestClient(_app(model, max_seq_len=32)) as client:
        assert (
            client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0}).status_code == 422
        )
        assert client.post("/v1/completions", json={"prompt": ""}).status_code == 422
        assert (
            client.post("/v1/completions", json={"prompt": "x", "temperature": 5}).status_code
            == 422
        )
        too_long = {"prompt": "y" * 30, "max_tokens": 8}
        r = client.post("/v1/completions", json=too_long)
        assert r.status_code == 400
        assert "exceeds" in r.json()["detail"]
        r = client.post("/v1/completions", json={**too_long, "stream": True})
        assert r.status_code == 400
        # the server keeps working after rejected requests
        assert (
            client.post("/v1/completions", json={"prompt": "ok", "max_tokens": 2}).status_code
            == 200
        )


async def _concurrent(app: FastAPI, n: int) -> dict[str, object]:
    """Send n requests at once and return /stats afterwards."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            body = {"prompt": "abc", "max_tokens": 20, "ignore_eos": True}
            rs = await asyncio.gather(*[c.post("/v1/completions", json=body) for _ in range(n)])
            assert all(r.status_code == 200 for r in rs)
            texts = {r.json()["choices"][0]["text"] for r in rs}
            assert len(texts) == 1  # same prompt, greedy: identical output in any batch
            stats: dict[str, object] = (await c.get("/stats")).json()
            return stats


@pytest.mark.parametrize(("policy", "batched"), [(Policy.NAIVE, False), (Policy.CONTINUOUS, True)])
def test_concurrent_clients_are_batched_only_by_continuous(
    model: GPT, policy: Policy, batched: bool
) -> None:
    stats = asyncio.run(_concurrent(_app(model, policy), 4))
    assert stats["finished"] == 4
    mean = stats["mean_decode_batch"]
    assert isinstance(mean, float | int)
    assert (mean > 1.5) is batched
    assert stats["policy"] == policy.value


def test_submit_refuses_when_queue_is_full(model: GPT) -> None:
    async def run() -> None:
        """Fill the inbox without starting the loop."""
        runner = AsyncEngine(Engine(model, Policy.CONTINUOUS, max_seq_len=32, max_waiting=1))
        runner.submit(Request("a", [1]))
        with pytest.raises(OverflowError):
            runner.submit(Request("b", [1]))

    asyncio.run(run())


def test_text_stream_holds_back_partial_utf8() -> None:
    s = _TextStream(ByteTokenizer())
    e_acute = list("é".encode())
    assert s.push(ord("a")) == "a"
    assert s.push(e_acute[0]) == ""
    assert s.push(e_acute[1]) == "é"
    assert s.flush() == ""
