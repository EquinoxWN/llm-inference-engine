"""An OpenAI-style completions API in front of the engine, with token streaming over SSE."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .engine import Engine, Request, TokenEvent
from .sampling import SamplingParams
from .tokenizer import Tokenizer

MAX_PROMPT_CHARS = 16_384


class CompletionRequest(BaseModel):
    """Request body for POST /v1/completions (a subset of the OpenAI fields)."""

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_tokens: int = Field(default=16, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0, le=1000)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    stream: bool = False
    ignore_eos: bool = False


@dataclass
class _Pending:
    """A submitted request and the queue its tokens are delivered to."""

    request: Request
    queue: asyncio.Queue[TokenEvent | Exception] = field(default_factory=asyncio.Queue)


class AsyncEngine:
    """Runs engine steps in a worker thread; only the loop task ever touches the engine."""

    def __init__(self, engine: Engine) -> None:
        """Wrap an engine; call start() inside a running event loop."""
        self.engine = engine
        self._inbox: list[_Pending] = []
        self._aborts: list[str] = []
        self._live: dict[str, _Pending] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ids = itertools.count()

    def start(self) -> None:
        """Start the scheduling loop."""
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the scheduling loop and wait for it to end."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def next_id(self) -> str:
        """A unique request id."""
        return f"cmpl-{next(self._ids)}"

    def submit(self, request: Request) -> asyncio.Queue[TokenEvent | Exception]:
        """Queue a request for the next step; tokens arrive on the returned queue."""
        backlog = len(self.engine.waiting) + len(self._inbox)
        if backlog >= self.engine.max_waiting:
            raise OverflowError("server is at capacity")
        pending = _Pending(request)
        self._inbox.append(pending)
        self._wake.set()
        return pending.queue

    def abort(self, request_id: str) -> None:
        """Ask the loop to drop a request (client went away)."""
        self._aborts.append(request_id)
        self._wake.set()

    def _drain_inbox(self) -> None:
        """Hand new requests and aborts to the engine between steps."""
        for pending in self._inbox:
            try:
                self.engine.add_request(pending.request)
                self._live[pending.request.request_id] = pending
            except (ValueError, OverflowError) as exc:
                pending.queue.put_nowait(exc)
        self._inbox.clear()
        for rid in self._aborts:
            self.engine.abort(rid)
            self._live.pop(rid, None)
        self._aborts.clear()

    async def _loop(self) -> None:
        """Step the engine while there is work, sleep otherwise."""
        while True:
            self._drain_inbox()
            if not self.engine.has_work():
                self._wake.clear()
                await self._wake.wait()
                continue
            try:
                events = await asyncio.to_thread(self.engine.step)
            except Exception as exc:
                for pending in self._live.values():
                    pending.queue.put_nowait(exc)
                for seq in list(self.engine.running):
                    self.engine.abort(seq.request.request_id)
                self._live.clear()
                continue
            for e in events:
                live = self._live.get(e.request_id)
                if live is not None:
                    live.queue.put_nowait(e)
                    if e.finish_reason is not None:
                        del self._live[e.request_id]


class _TextStream:
    """Turns token ids into text increments, holding back incomplete UTF-8 sequences."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Start with no tokens."""
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.sent = 0

    def push(self, token: int) -> str:
        """Add a token and return any newly completed text."""
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent :], len(text)
        return delta

    def flush(self) -> str:
        """Return whatever text is left at the end."""
        text = self.tokenizer.decode(self.ids)
        delta, self.sent = text[self.sent :], len(text)
        return delta


def create_app(
    engine_factory: Callable[[], Engine], tokenizer: Tokenizer, model_name: str
) -> FastAPI:
    """Build the FastAPI app; the engine is created when the app starts."""
    state: dict[str, AsyncEngine] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Start and stop the scheduling loop with the app."""
        runner = AsyncEngine(engine_factory())
        runner.start()
        state["runner"] = runner
        yield
        await runner.stop()

    app = FastAPI(title="llm-inference-engine", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict[str, object]:
        """Scheduler counters and queue depth."""
        eng = state["runner"].engine
        s = eng.stats
        return {
            "policy": eng.policy.value,
            "max_batch": eng.max_batch,
            "waiting": len(eng.waiting),
            "running": len(eng.running),
            "steps": s.steps,
            "prefills": s.prefills,
            "decode_passes": s.decode_passes,
            "mean_decode_batch": round(s.mean_decode_batch, 3),
            "finished": s.finished,
            "aborted": s.aborted,
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest) -> dict[str, object] | StreamingResponse:
        """Generate a completion, streamed as server-sent events when stream is true."""
        runner = state["runner"]
        prompt_ids = tokenizer.encode(body.prompt)
        params = SamplingParams(
            max_new_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            seed=body.seed,
            stop_at_eos=not body.ignore_eos,
        )
        rid = runner.next_id()
        try:
            queue = runner.submit(Request(rid, prompt_ids, params))
        except OverflowError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def events() -> AsyncGenerator[TokenEvent, None]:
            """Yield this request's tokens; abort it if the consumer stops early."""
            finished = False
            try:
                while not finished:
                    item = await queue.get()
                    if isinstance(item, Exception):
                        raise item
                    finished = item.finish_reason is not None
                    yield item
            finally:
                if not finished:
                    runner.abort(rid)

        def chunk(text: str, reason: str | None) -> str:
            """One SSE data line in the OpenAI text_completion shape."""
            payload = {
                "id": rid,
                "object": "text_completion",
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        if body.stream:
            first = events()
            try:
                head = await anext(first)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            async def sse() -> AsyncIterator[str]:
                """Stream text increments, then [DONE]."""
                text = _TextStream(tokenizer)
                item: TokenEvent | None = head
                try:
                    while item is not None:
                        delta = text.push(item.token_id)
                        if item.finish_reason is not None:
                            yield chunk(delta + text.flush(), item.finish_reason)
                            break
                        if delta:
                            yield chunk(delta, None)
                        item = await anext(first, None)
                    yield "data: [DONE]\n\n"
                finally:
                    await first.aclose()

            return StreamingResponse(sse(), media_type="text/event-stream")

        ids: list[int] = []
        reason: str | None = None
        try:
            async for e in events():
                ids.append(e.token_id)
                reason = e.finish_reason
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": rid,
            "object": "text_completion",
            "model": model_name,
            "choices": [{"index": 0, "text": tokenizer.decode(ids), "finish_reason": reason}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(ids),
                "total_tokens": len(prompt_ids) + len(ids),
            },
        }

    return app
