from __future__ import annotations

import asyncio
import time

import httpx
import structlog

from cc_adapter.command_code.headers import make_cc_headers
from cc_adapter.core.constants import KEY_CREDITS_CACHE_TTL, KEY_CREDITS_ERROR_BACKOFF

logger = structlog.get_logger(__name__)


class KeyPool:
    def __init__(self, keys: list[str], base_url: str):
        self._keys = list(keys)
        self._base_url = base_url.rstrip("/")
        self._credits: dict[str, int] = {}
        self._unavailable: set[str] = set()
        self._last_fetch: float | None = None
        self._last_error: str | None = None
        self._fetch_task: asyncio.Task[None] | None = None

    async def select_key(self) -> str | None:
        if self._is_stale():
            self._trigger_refresh()
        for key in self._keys:
            if key in self._unavailable:
                continue
            credits = self._credits.get(key)
            if credits is None or credits > 0:
                return key
        return self._keys[0] if self._keys else None

    def mark_unavailable(self, key: str) -> None:
        self._unavailable.add(key)

    def clear_unavailable(self) -> None:
        self._unavailable.clear()

    def get_credits(self, key: str) -> int | None:
        return self._credits.get(key)

    def _is_stale(self) -> bool:
        if self._last_fetch is None:
            return True
        ttl = KEY_CREDITS_ERROR_BACKOFF if self._last_error else KEY_CREDITS_CACHE_TTL
        return time.monotonic() - self._last_fetch > ttl

    def _trigger_refresh(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            if not self._fetch_task or self._fetch_task.done():
                self._fetch_task = loop.create_task(self._refresh())
        except RuntimeError:
            pass

    async def refresh(self) -> None:
        await self._refresh()

    async def _refresh(self) -> None:
        self._last_error = None
        try:
            tasks = [self._fetch_credits(key) for key in self._keys]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for key, result in zip(self._keys, results):
                if isinstance(result, Exception):
                    logger.warning("credits_fetch_failed", key=key[:8] + "...", error=str(result))
                elif isinstance(result, int):
                    self._credits[key] = result
            self._last_fetch = time.monotonic()
        except Exception as e:
            self._last_error = str(e)
            self._last_fetch = time.monotonic()
            logger.warning("credits_refresh_failed", error=str(e))

    async def _fetch_credits(self, api_key: str) -> int | None:
        headers = make_cc_headers(api_key)
        async with httpx.AsyncClient(timeout=10.0, base_url=self._base_url) as client:
            r = await client.get("/alpha/billing/credits", headers=headers)
            r.raise_for_status()
            data = r.json()
            if "credits" in data:
                c = data["credits"]
                return c.get("monthlyCredits", 0) + c.get("purchasedCredits", 0) + c.get("freeCredits", 0)
            return 0

    @property
    def last_fetch_time(self) -> float | None:
        return self._last_fetch

    @property
    def last_error(self) -> str | None:
        return self._last_error
