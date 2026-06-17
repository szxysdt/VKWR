from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vkwr.engine.prompts import apply_chat_template, get_default_stop_tokens
from vkwr.engine.request import RequestOutputKind, SamplingParams
from vkwr.entrypoints.llm import LLM
from vkwr.utils import generate_request_id

if TYPE_CHECKING:
    from vkwr.config.vkwr import EngineArgs

logger = logging.getLogger(__name__)


# ─── Request models ────────────────────────────────────────────────


class CompletionRequest(BaseModel):
    """OpenAI-compatible /v1/completions request body."""

    model: str
    prompt: str | list[int]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    stream: bool = False
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request body."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    stream: bool = False
    seed: int | None = None


# ─── Response helpers ──────────────────────────────────────────────


def _make_completion_response(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    created: int | None = None,
) -> dict:
    """Build an OpenAI text_completion-formatted response."""
    if created is None:
        created = int(time.time())
    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_chat_completion_response(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    created: int | None = None,
) -> dict:
    """Build an OpenAI chat.completion-formatted response."""
    if created is None:
        created = int(time.time())
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_completion_chunk(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    created: int | None = None,
) -> str:
    """Build an SSE streaming delta chunk (text_completion)."""
    if created is None:
        created = int(time.time())
    chunk = {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_chat_completion_chunk(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    created: int | None = None,
    is_first: bool = False,
) -> str:
    """Build an SSE streaming delta chunk (chat.completion)."""
    if created is None:
        created = int(time.time())
    chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant" if is_first else None,
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ─── Sampling params helper ────────────────────────────────────────


def _make_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    seed: int | None = None,
    stream: bool = False,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        stop=stop,
        stop_token_ids=stop_token_ids,
        seed=seed,
        output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
    )


# ─── Streaming generators ──────────────────────────────────────────


async def _stream_completion(
    llm: LLM,
    prompt: str | list[int],
    sampling_params: SamplingParams,
    model: str,
) -> StreamingResponse:
    """Stream a completion generation (text_completion)."""
    created = int(time.time())
    req_id: str | None = None

    async def event_generator():
        nonlocal req_id
        llm._lazy_init()
        engine = llm.llm_engine
        loop = asyncio.get_event_loop()
        req_id = generate_request_id("stream")
        await loop.run_in_executor(
            None,
            engine.add_request,
            req_id,
            prompt,
            sampling_params,
        )

        # Run engine.step() in executor to avoid blocking the event loop.
        # Each step triggers a CUDA forward pass that must not run on the
        # asyncio event loop thread.
        while engine.has_unfinished_requests():
            step_outputs = await loop.run_in_executor(None, engine.step)
            for out in step_outputs:
                text = out.outputs[0].text if out.outputs else ""
                delta = _make_completion_chunk(req_id or "", text, out.finish_reason, model, created)
                yield delta
                if out.finished:
                    engine.remove_request(out.request_id)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _stream_chat_completion(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    model: str,
) -> StreamingResponse:
    """Stream a chat completion generation (chat.completion)."""
    created = int(time.time())
    req_id: str | None = None

    async def event_generator():
        nonlocal req_id
        llm._lazy_init()
        engine = llm.llm_engine
        loop = asyncio.get_event_loop()
        req_id = generate_request_id("stream-chat")
        await loop.run_in_executor(
            None,
            engine.add_request,
            req_id,
            prompt,
            sampling_params,
        )

        # Send initial chunk with role
        yield _make_chat_completion_chunk(req_id or "", "", None, model, created, is_first=True)

        while engine.has_unfinished_requests():
            step_outputs = await loop.run_in_executor(None, engine.step)
            for out in step_outputs:
                text = out.outputs[0].text if out.outputs else ""
                delta = _make_chat_completion_chunk(req_id or "", text, out.finish_reason, model, created)
                yield delta
                if out.finished:
                    engine.remove_request(out.request_id)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ─── App factory ───────────────────────────────────────────────────


def create_app(engine_args: EngineArgs) -> FastAPI:
    """Create an OpenAI-compatible FastAPI application.

    Args:
        engine_args: Engine configuration parameters.

    Returns:
        A FastAPI application instance.
    """
    app = FastAPI(title="VKWR API", version="0.1.0")

    # LLM instance is shared at module level (lazy-initialized)
    llm = LLM(**vars(engine_args))

    @app.get("/v1/models")
    async def models():
        """Get the list of available models."""
        return {"data": [{"id": engine_args.model, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        """OpenAI-compatible text completion endpoint.

        Supports both synchronous and streaming modes.
        """
        sampling_params = _make_sampling_params(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
            stop_token_ids=req.stop_token_ids,
            seed=req.seed,
            stream=req.stream,
        )

        if req.stream:
            return await _stream_completion(llm, req.prompt, sampling_params, req.model)

        try:
            outputs = llm.generate(req.prompt, sampling_params)
        except Exception as e:
            logger.error("Completion failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        if not outputs:
            raise HTTPException(status_code=500, detail="No output generated")

        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ""
        prompt_len = len(out.prompt_token_ids)
        completion_len = len(out.outputs[0].token_ids) if out.outputs else 0
        return _make_completion_response(
            out.request_id,
            text,
            out.finish_reason,
            req.model,
            prompt_len,
            completion_len,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        """OpenAI-compatible chat completion endpoint.

        Supports both synchronous and streaming modes.
        """
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        prompt = apply_chat_template(
            [{"role": m.role, "content": m.content} for m in req.messages],
            add_generation_prompt=True,
        )
        stop = req.stop if req.stop is not None else get_default_stop_tokens()
        sampling_params = _make_sampling_params(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            max_tokens=req.max_tokens,
            stop=stop,
            stop_token_ids=req.stop_token_ids,
            seed=req.seed,
            stream=req.stream,
        )

        if req.stream:
            return await _stream_chat_completion(llm, prompt, sampling_params, req.model)

        try:
            outputs = llm.generate(prompt, sampling_params)
        except Exception as e:
            logger.error("Chat completion failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        if not outputs:
            raise HTTPException(status_code=500, detail="No output generated")

        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ""
        prompt_len = len(out.prompt_token_ids)
        completion_len = len(out.outputs[0].token_ids) if out.outputs else 0
        return _make_chat_completion_response(
            out.request_id,
            text,
            out.finish_reason,
            req.model,
            prompt_len,
            completion_len,
        )

    return app
