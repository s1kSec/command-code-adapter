from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, TypeVar

from cc_adapter.core.errors import AdapterError

T = TypeVar("T")


async def retry_on_empty(
    generate_fn: Callable[[], AsyncGenerator[dict, None]],
    translate_fn: Callable[[AsyncGenerator[dict, None]], Awaitable[T]],
    logger: logging.Logger,
    label: str = "",
) -> T:
    for attempt in range(2):
        cc_stream = generate_fn()
        try:
            return await translate_fn(cc_stream)
        except AdapterError as e:
            if attempt == 0 and "empty response" in e.message.lower():
                logger.warning("%s: Empty upstream response (attempt 1/2), retrying...", label)
                continue
            raise
