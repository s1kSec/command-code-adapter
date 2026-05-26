# Passthrough Logging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detailed request/response logging to the DeepSeek passthrough forwarding path for web_search research.

**Architecture:** New `RequestLogger` module writes structured logs to both files (`logs/`) and console (structlog). Hooks added only to `_stream_from_deepseek` and `_deepseek_nonstream` in the Anthropic router. CC translation pipeline untouched.

**Tech Stack:** Python, structlog, httpx (existing)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `cc_adapter/core/request_logger.py` | Create | RequestLogger class: file+console logging, SSE parsing |
| `cc_adapter/providers/anthropic/router.py` | Modify | Add logging hooks to DeepSeek forwarding path |
| `.gitignore` | Modify | Add `logs/` |

---

### Task 1: Setup — branch and gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Create branch from master**

```bash
git checkout master && git checkout -b feature/passthrough-logging
```

- [ ] **Step 2: Add logs/ to .gitignore**

Append to `.gitignore`:
```
logs/
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore && git commit -m "chore: add logs/ to gitignore"
```

---

### Task 2: Create RequestLogger module

**Files:**
- Create: `cc_adapter/core/request_logger.py`

- [ ] **Step 1: Write `cc_adapter/core/request_logger.py`**

```python
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class RequestLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_request(self, model: str, stream: bool, messages_count: int,
                      body_json: str, system_info: str = "", tools_count: int = 0) -> tuple[str, Path]:
        request_id = uuid.uuid4().hex[:16]
        timestamp = datetime.now(timezone.utc)
        ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts_str}_{request_id[:8]}.log"
        filepath = self.log_dir / filename

        header = (
            f"=== REQUEST {timestamp.isoformat()} ===\n"
            f"Request ID: {request_id}\n"
            f"Model: {model}\n"
            f"Stream: {stream}\n"
            f"Messages: {messages_count}\n"
            f"Tools: {tools_count}\n"
            f"System: {system_info}\n"
            f"--- RAW BODY ---\n"
            f"{body_json}\n\n"
            f"=== RESPONSE STREAM ===\n"
        )
        filepath.write_text(header, encoding="utf-8")

        logger.info("passthrough.request.start",
                    request_id=request_id,
                    file=str(filepath),
                    model=model,
                    stream=stream,
                    messages_count=messages_count,
                    tools_count=tools_count)

        return request_id, filepath

    def log_raw_sse(self, filepath: Path, line: str):
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"RAW: {line}\n")

    def log_sse_event(self, filepath: Path, event_type: str, data: dict, elapsed_s: float):
        ts = f"+{elapsed_s:.1f}s"
        entry = f"[{ts}] {event_type:<24s}"

        if event_type == "content_block_delta" and isinstance(data, dict):
            delta = data.get("delta", {})
            idx = data.get("index", "?")
            text = delta.get("text", "")
            entry += f" index={idx} text={json.dumps(text)}"
        elif event_type == "message_start" and isinstance(data, dict):
            msg = data.get("message", {})
            entry += f" id={msg.get('id', '?')}"
        elif event_type == "message_delta" and isinstance(data, dict):
            delta = data.get("delta", {})
            entry += f" stop={delta.get('stop_reason', '?')}"
            usage = data.get("usage", {})
            if usage:
                entry += f" in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}"
        elif event_type == "ping":
            entry += "(heartbeat)"
        elif event_type == "error":
            entry += f" error={json.dumps(data)}"

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"{entry}\n")

    def end_request(self, filepath: Path, request_id: str, start_time: float,
                    stop_reason: str = "", input_tokens: int = 0, output_tokens: int = 0):
        duration = time.monotonic() - start_time
        summary = (
            f"\n=== SUMMARY ===\n"
            f"Duration: {duration:.1f}s\n"
            f"Stop reason: {stop_reason or 'N/A'}\n"
            f"Usage: input={input_tokens} output={output_tokens}\n"
            f"Status: OK\n"
        )
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(summary)

        logger.info("passthrough.request.end",
                    request_id=request_id,
                    duration=f"{duration:.1f}s",
                    stop_reason=stop_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens)

    def end_request_error(self, filepath: Path, request_id: str, error: str, start_time: float):
        duration = time.monotonic() - start_time
        summary = (
            f"\n=== SUMMARY ===\n"
            f"Duration: {duration:.1f}s\n"
            f"Status: ERROR\n"
            f"Error: {error}\n"
        )
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(summary)

        logger.warning("passthrough.request.error",
                       request_id=request_id,
                       duration=f"{duration:.1f}s",
                       error=error)
```

- [ ] **Step 2: Verify syntax**

```bash
poetry run python -c "from cc_adapter.core.request_logger import RequestLogger; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add cc_adapter/core/request_logger.py && git commit -m "feat: add RequestLogger for passthrough logging"
```

---

### Task 3: Add logging hooks to router.py

**Files:**
- Modify: `cc_adapter/providers/anthropic/router.py`

- [ ] **Step 1: Add import for RequestLogger**

In `router.py`, add to imports (after line 20 `logger = structlog.get_logger(__name__)`):

```python
from cc_adapter.core.request_logger import RequestLogger

_request_logger = RequestLogger()
```

- [ ] **Step 2: Modify `_stream_from_deepseek` to add logging**

Replace the function body at `router.py:73-101`:

```python
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
                                delta = data.get("delta", {})
                                stop_reason = delta.get("stop_reason", "")
                        except json.JSONDecodeError:
                            pass
                    yield line + "\n"
        _request_logger.end_request(log_file, request_id, start_time,
                                    stop_reason=stop_reason,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens)
    except httpx.RequestError as e:
        logger.warning("deepseek.forward.request_error", error=str(e))
        _request_logger.end_request_error(log_file, request_id, str(e), start_time)
        yield _anthropic_sse_error(f"DeepSeek API connection error: {e}")
    except Exception as e:
        logger.warning("deepseek.forward.unexpected_error", error=str(e))
        _request_logger.end_request_error(log_file, request_id, str(e), start_time)
        yield _anthropic_sse_error(str(e))
```

Also add `import json` and `import time` to the imports at the top of `router.py`.

- [ ] **Step 3: Modify `_deepseek_nonstream` to add logging**

Replace the function body at `router.py:104-136`:

```python
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
```

- [ ] **Step 4: Verify syntax and imports**

```bash
poetry run python -c "from cc_adapter.providers.anthropic.router import router; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add cc_adapter/providers/anthropic/router.py && git commit -m "feat: add passthrough logging hooks to DeepSeek forward path"
```

---

### Task 4: Build and test

- [ ] **Step 1: Run all existing tests**

```bash
poetry run pytest tests/ -v
```
Expected: all 412 tests pass

- [ ] **Step 2: Build Docker image**

```bash
docker build -t dgqyushen/command-code-proxy:latest .
```
Expected: builds successfully

- [ ] **Step 3: Restart container**

```bash
docker compose down && docker compose up -d
```

- [ ] **Step 4: Verify container is running**

```bash
docker compose ps
```

- [ ] **Step 5: Commit**

```bash
git commit --allow-empty -m "chore: verify passthrough logging build and tests"
```

---

### Task 5: Test with Claude Code and collect logs

- [ ] **Step 1: Switch Claude Code config to adapter**

```bash
cp ~/.claude/settings.json ~/.claude/settings.json.ds && cp ~/.claude/settings.json.cc ~/.claude/settings.json
```

- [ ] **Step 2: Send a web_search request and collect logs**

```bash
echo "帮我搜索一下python最新版本是什么" | timeout 30 claude -p --output-format stream-json 2>&1 | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        t = d.get('type','')
        if t == 'stream_event':
            e = d.get('event',{})
            if e.get('type') == 'content_block_delta':
                print(e.get('delta',{}).get('text',''), end='', flush=True)
    except: pass
"
```

- [ ] **Step 3: Check log files**

```bash
ls -la logs/ && echo "---" && cat logs/*.log | head -50
```

- [ ] **Step 4: Verify log contents are detailed enough**

Check that log file contains: request body JSON, SSE events with parsed text, summary with token usage

- [ ] **Step 5: Switch Claude Code back to direct DeepSeek**

```bash
cp ~/.claude/settings.json.cc ~/.claude/settings.json.current_adapter && cp ~/.claude/settings.json.ds ~/.claude/settings.json
```

- [ ] **Step 6: Commit log results**

```bash
git add logs/ && git commit -m "results: passthrough logging samples"
```
