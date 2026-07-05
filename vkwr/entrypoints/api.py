from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vkwr.engine import AsyncLLMEngine
from vkwr.engine.exceptions import EngineDeadError, EngineGenerateError
from vkwr.engine.outputs import RequestOutput
from vkwr.engine.prompts import apply_chat_template, get_default_stop_tokens
from vkwr.engine.request import RequestOutputKind, SamplingParams
from vkwr.utils import generate_request_id

if TYPE_CHECKING:
    from vkwr.config.vkwr import EngineArgs

logger = logging.getLogger(__name__)


# ─── Request models ────────────────────────────────────────────────


class StreamOptions(BaseModel):
    include_usage: bool = False


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
    stream_options: StreamOptions | None = None
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
    stream_options: StreamOptions | None = None
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
    system_fingerprint: str | None = None,
    num_cached_tokens: int | None = None,
) -> dict:
    """Build an OpenAI text_completion-formatted response."""
    if created is None:
        created = int(time.time())
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if num_cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": num_cached_tokens}
    resp: dict = {
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
        "usage": usage,
    }
    if system_fingerprint is not None:
        resp["system_fingerprint"] = system_fingerprint
    return resp


def _make_chat_completion_response(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    created: int | None = None,
    system_fingerprint: str | None = None,
    num_cached_tokens: int | None = None,
) -> dict:
    """Build an OpenAI chat.completion-formatted response."""
    if created is None:
        created = int(time.time())
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if num_cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": num_cached_tokens}
    resp: dict = {
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
        "usage": usage,
    }
    if system_fingerprint is not None:
        resp["system_fingerprint"] = system_fingerprint
    return resp


def _make_completion_chunk(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    created: int | None = None,
    system_fingerprint: str | None = None,
    usage: dict | None = None,
) -> str:
    """Build an SSE streaming delta chunk (text_completion)."""
    if created is None:
        created = int(time.time())
    chunk: dict = {
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
    if system_fingerprint is not None:
        chunk["system_fingerprint"] = system_fingerprint
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def _make_chat_completion_chunk(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
    created: int | None = None,
    is_first: bool = False,
    system_fingerprint: str | None = None,
    usage: dict | None = None,
) -> str:
    """Build an SSE streaming delta chunk (chat.completion)."""
    if created is None:
        created = int(time.time())
    chunk: dict = {
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
    if system_fingerprint is not None:
        chunk["system_fingerprint"] = system_fingerprint
    if usage is not None:
        chunk["usage"] = usage
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
    async_engine: AsyncLLMEngine,
    prompt: str | list[int],
    sampling_params: SamplingParams,
    model: str,
    *,
    system_fingerprint: str | None = None,
    stream_options: StreamOptions | None = None,
) -> StreamingResponse:
    """Stream a completion generation (text_completion)."""
    created = int(time.time())
    req_id = generate_request_id("stream")
    include_usage = stream_options is not None and stream_options.include_usage

    async def event_generator():
        last_out = None
        total_completion_tokens = 0
        try:
            async for out in async_engine.generate(prompt, sampling_params, req_id):
                text = out.outputs[0].text if out.outputs else ""
                if out.outputs:
                    total_completion_tokens += len(out.outputs[0].token_ids)
                last_out = out

                if out.finished and include_usage:
                    prompt_len = len(out.prompt_token_ids)
                    usage = {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": prompt_len + total_completion_tokens,
                    }
                    yield _make_completion_chunk(
                        req_id,
                        text,
                        out.finish_reason,
                        model,
                        created,
                        system_fingerprint=system_fingerprint,
                        usage=usage,
                    )
                elif out.finished:
                    yield _make_completion_chunk(
                        req_id,
                        text,
                        out.finish_reason,
                        model,
                        created,
                        system_fingerprint=system_fingerprint,
                    )
                else:
                    yield _make_completion_chunk(req_id, text, out.finish_reason, model, created)
        except EngineDeadError:
            yield _make_completion_chunk(req_id, "", "error", model, created)
            return
        except EngineGenerateError:
            yield _make_completion_chunk(req_id, "", "error", model, created)
            return

        if last_out is not None and not include_usage:
            prompt_len = len(last_out.prompt_token_ids)
            usage_chunk = {
                "id": f"cmpl-{req_id}",
                "object": "text_completion",
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": prompt_len + total_completion_tokens,
                },
            }
            if system_fingerprint is not None:
                usage_chunk["system_fingerprint"] = system_fingerprint
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _stream_chat_completion(
    async_engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    model: str,
    *,
    system_fingerprint: str | None = None,
    stream_options: StreamOptions | None = None,
) -> StreamingResponse:
    """Stream a chat completion generation (chat.completion)."""
    created = int(time.time())
    req_id = generate_request_id("stream-chat")
    include_usage = stream_options is not None and stream_options.include_usage

    async def event_generator():
        yield _make_chat_completion_chunk(
            req_id,
            "",
            None,
            model,
            created,
            is_first=True,
        )
        last_out = None
        total_completion_tokens = 0
        try:
            async for out in async_engine.generate(prompt, sampling_params, req_id):
                text = out.outputs[0].text if out.outputs else ""
                if out.outputs:
                    total_completion_tokens += len(out.outputs[0].token_ids)
                last_out = out

                if out.finished and include_usage:
                    prompt_len = len(out.prompt_token_ids)
                    usage = {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": prompt_len + total_completion_tokens,
                    }
                    yield _make_chat_completion_chunk(
                        req_id,
                        text,
                        out.finish_reason,
                        model,
                        created,
                        system_fingerprint=system_fingerprint,
                        usage=usage,
                    )
                elif out.finished:
                    yield _make_chat_completion_chunk(
                        req_id,
                        text,
                        out.finish_reason,
                        model,
                        created,
                        system_fingerprint=system_fingerprint,
                    )
                else:
                    yield _make_chat_completion_chunk(
                        req_id,
                        text,
                        out.finish_reason,
                        model,
                        created,
                    )
        except EngineDeadError:
            yield _make_chat_completion_chunk(
                req_id,
                "",
                "error",
                model,
                created,
                system_fingerprint=system_fingerprint,
            )
            return
        except EngineGenerateError:
            yield _make_chat_completion_chunk(
                req_id,
                "",
                "error",
                model,
                created,
                system_fingerprint=system_fingerprint,
            )
            return

        if last_out is not None and not include_usage:
            prompt_len = len(last_out.prompt_token_ids)
            trailing_usage_chunk = {
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": prompt_len + total_completion_tokens,
                },
            }
            if system_fingerprint is not None:
                trailing_usage_chunk["system_fingerprint"] = system_fingerprint
            yield f"data: {json.dumps(trailing_usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _collect_completion(
    async_engine: AsyncLLMEngine,
    prompt: str | list[int],
    sampling_params: SamplingParams,
) -> RequestOutput | None:
    """Run a non-streaming generation, collecting FINAL_ONLY output."""
    req_id = generate_request_id("req")
    out = None
    try:
        async for o in async_engine.generate(prompt, sampling_params, req_id):
            out = o
    except EngineDeadError:
        raise
    except EngineGenerateError:
        raise
    return out


# ─── App factory ───────────────────────────────────────────────────


def create_app(engine_args: EngineArgs) -> FastAPI:
    """Create an OpenAI-compatible FastAPI application.

    Args:
        engine_args: Engine configuration parameters.

    Returns:
        A FastAPI application instance.
    """
    async_engine = AsyncLLMEngine.from_engine_args(engine_args)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await async_engine.shutdown()

    system_fingerprint = f"vkwr-{engine_args.model.split('/')[-1]}"
    app = FastAPI(title="VKWR API", version="0.1.0", lifespan=lifespan)

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
            return await _stream_completion(
                async_engine,
                req.prompt,
                sampling_params,
                req.model,
                system_fingerprint=system_fingerprint,
                stream_options=req.stream_options,
            )

        try:
            out = await _collect_completion(async_engine, req.prompt, sampling_params)
        except EngineDeadError as e:
            raise HTTPException(status_code=503, detail="Engine is down") from e
        except EngineGenerateError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        if out is None:
            raise HTTPException(status_code=500, detail="No output generated")

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
            system_fingerprint=system_fingerprint,
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
            return await _stream_chat_completion(
                async_engine,
                prompt,
                sampling_params,
                req.model,
                system_fingerprint=system_fingerprint,
                stream_options=req.stream_options,
            )

        try:
            out = await _collect_completion(async_engine, prompt, sampling_params)
        except EngineDeadError as e:
            raise HTTPException(status_code=503, detail="Engine is down") from e
        except EngineGenerateError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        if out is None:
            raise HTTPException(status_code=500, detail="No output generated")

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
            system_fingerprint=system_fingerprint,
        )

    return app
