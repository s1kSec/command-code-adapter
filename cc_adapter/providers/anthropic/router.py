from __future__ import annotations

import json
import time

import httpx
import structlog
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cc_adapter.providers.anthropic.models import AnthropicRequest, AnthropicResponse, AnthropicUsage
from cc_adapter.providers.anthropic.response import (
    translate_anthropic_stream,
    collect_and_translate_anthropic_nonstream,
)
from cc_adapter.command_code.client import CommandCodeClient
from cc_adapter.core.retry import retry_on_empty, stream_with_retry
from cc_adapter.core.runtime import get_config, get_anthropic_translator
from cc_adapter.core.constants import STREAMING_HEADERS
from cc_adapter.core.errors import AdapterError
from cc_adapter.core.utils import format_sse, generate_id

from cc_adapter.core.request_logger import RequestLogger

logger = structlog.get_logger(__name__)

_request_logger = RequestLogger()

router = APIRouter()


def _get_client() -> CommandCodeClient:
    from cc_adapter.core.runtime import get_or_create_client

    return get_or_create_client()


def _anthropic_sse_error(message: str) -> str:
    return format_sse("error", {"type": "error", "error": {"type": "api_error", "message": message}})


def _extract_system_text(system):
    if system is None:
        return None
    if isinstance(system, str):
        return system
    texts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
    return " ".join(texts) if texts else None


def _build_deepseek_body(req: AnthropicRequest) -> dict:
    body: dict = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
        "messages": [m.model_dump() for m in req.messages],
    }

    system_text = _extract_system_text(req.system)
    if system_text:
        body["system"] = system_text

    if req.tools:
        body["tools"] = [t.model_dump() for t in req.tools]

    if req.thinking and req.thinking.type in ("enabled", "adaptive"):
        thinking_config: dict = {"type": req.thinking.type}
        if req.thinking.budget_tokens is not None:
            thinking_config["budget_tokens"] = req.thinking.budget_tokens
        body["thinking"] = thinking_config

    if req.temperature is not None:
        body["temperature"] = req.temperature

    return body


async def _stream_from_deepseek(req: AnthropicRequest) -> AsyncGenerator[str, None]:
    config = get_config()
    url = f"{config.deepseek_anthropic_url}/v1/messages"
    headers = {
        "x-api-key": config.deepseek_api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    body = _build_deepseek_body(req)
    body["stream"] = True

    body_json = json.dumps(body, ensure_ascii=False, indent=2)
    system_info = _extract_system_text(req.system) or ""
    tools_count = len(req.tools) if req.tools else 0

    request_id, log_file = _request_logger.start_request(
        model=req.model,
        stream=True,
        messages_count=len(req.messages),
        body_json=body_json,
        system_info=system_info[:200] + ("..." if len(system_info) > 200 else ""),
        tools_count=tools_count,
    )
    start_time = time.monotonic()

    stop_reason = ""
    input_tokens = 0
    output_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_str = str(error_text)
                    logger.warning("deepseek.forward.error", status_code=resp.status_code, error=error_str)
                    _request_logger.end_request_error(log_file, request_id, f"HTTP {resp.status_code}: {error_str[:500]}", start_time)
                    yield _anthropic_sse_error(f"DeepSeek API error {resp.status_code}")
                    return
                async for line in resp.aiter_lines():
                    _request_logger.log_raw_sse(log_file, line)
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            event_type = data.get("type", "?")
                            elapsed = time.monotonic() - start_time
                            _request_logger.log_sse_event(log_file, event_type, data, elapsed)
                            if event_type == "message_delta":
                                usage = data.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                output_tokens = usage.get("output_tokens", 0)
                                delta_data = data.get("delta", {})
                                stop_reason = delta_data.get("stop_reason", "")
                        except (json.JSONDecodeError, KeyError):
                            pass
                    yield line + "\n"
        _request_logger.end_request(
            log_file, request_id, start_time,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except httpx.RequestError as e:
        logger.warning("deepseek.forward.request_error", error=str(e))
        _request_logger.end_request_error(log_file, request_id, str(e), start_time)
        yield _anthropic_sse_error(f"DeepSeek API connection error: {e}")
    except Exception as e:
        logger.warning("deepseek.forward.unexpected_error", error=str(e))
        _request_logger.end_request_error(log_file, request_id, str(e), start_time)
        yield _anthropic_sse_error(str(e))


async def _deepseek_nonstream(req: AnthropicRequest) -> AnthropicResponse:
    config = get_config()
    url = f"{config.deepseek_anthropic_url}/v1/messages"
    headers = {
        "x-api-key": config.deepseek_api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    body = _build_deepseek_body(req)
    body["stream"] = False

    body_json = json.dumps(body, ensure_ascii=False, indent=2)
    system_info = _extract_system_text(req.system) or ""
    tools_count = len(req.tools) if req.tools else 0

    request_id, log_file = _request_logger.start_request(
        model=req.model,
        stream=False,
        messages_count=len(req.messages),
        body_json=body_json,
        system_info=system_info[:200] + ("..." if len(system_info) > 200 else ""),
        tools_count=tools_count,
    )
    start_time = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                logger.warning("deepseek.forward.nonstream_error", status_code=resp.status_code, error=resp.text)
                _request_logger.end_request_error(log_file, request_id, f"HTTP {resp.status_code}: {resp.text[:500]}", start_time)
                raise AdapterError(message=f"DeepSeek API returned {resp.status_code}", status_code=502)
            data = resp.json()

        response_json = json.dumps(data, ensure_ascii=False, indent=2)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"--- RAW RESPONSE ---\n{response_json}\n")

        usage = data.get("usage", {})
        stop_reason = data.get("stop_reason", "")

        _request_logger.end_request(
            log_file, request_id, start_time,
            stop_reason=stop_reason,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        return AnthropicResponse(
            id=data.get("id", generate_id("msg_", 16)),
            content=data.get("content", []),
            model=data.get("model", req.model),
            stop_reason=data.get("stop_reason"),
            usage=AnthropicUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
        )
    except httpx.RequestError as e:
        logger.warning("deepseek.forward.nonstream_request_error", error=str(e))
        _request_logger.end_request_error(log_file, request_id, str(e), start_time)
        raise AdapterError(message=f"DeepSeek API connection error: {e}", status_code=502)


@router.post("/v1/messages")
async def anthropic_chat(req: AnthropicRequest, request: Request):
    structlog.contextvars.bind_contextvars(protocol="anthropic")

    logger.info(
        "anthropic.request",
        model=req.model,
        stream=str(req.stream),
        message_count=len(req.messages),
        tools="yes" if req.tools else "no",
    )

    from cc_adapter.providers.shared.web_search import is_web_search_enabled

    config = get_config()

    if is_web_search_enabled(config):
        logger.info("anthropic.web_search.forward", model=req.model)
        try:
            if req.stream:
                return StreamingResponse(
                    _stream_from_deepseek(req),
                    media_type="text/event-stream",
                    headers=STREAMING_HEADERS,
                )
            return await _deepseek_nonstream(req)
        except AdapterError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"type": "api_error", "message": e.message}},
            )

    translator = get_anthropic_translator()
    cc_body, cc_headers = translator.translate(req)
    cc_body["params"]["stream"] = True

    current_client = _get_client()

    try:
        if req.stream:
            return StreamingResponse(
                stream_with_retry(
                    lambda: current_client.generate(cc_body, cc_headers),
                    lambda stream: translate_anthropic_stream(stream, req.model),
                    logger,
                    "anthropic.stream",
                    error_fn=lambda msg: _anthropic_sse_error(msg),
                ),
                media_type="text/event-stream",
                headers=STREAMING_HEADERS,
            )
        return await retry_on_empty(
            lambda: current_client.generate(cc_body, cc_headers),
            lambda stream: collect_and_translate_anthropic_nonstream(stream, req.model),
            logger,
            "anthropic.nonstream",
        )
    except AdapterError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"type": "api_error", "message": e.message}},
        )
