"""
RedisRepository — live-mode transport for inference.

Orchestrator pushes InferenceTask dicts onto queue:inference:{session_id}
during live mode. Worker scans active sessions and pops the next task.

(Batch mode goes through Kafka; this is the live-only path.)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import redis.asyncio as redis

from app.Config import LIVE_BLPOP_TIMEOUT_SECONDS, REDIS_URL

logger = logging.getLogger("inference-worker.redis")

_QUEUE_PREFIX = "queue:inference"


class RedisRepository:
    def __init__(self) -> None:
        self._r: redis.Redis | None = None

    async def start(self) -> None:
        self._r = redis.from_url(REDIS_URL, decode_responses=False)

    async def stop(self) -> None:
        if self._r:
            await self._r.close()

    async def scan_active_sessions(self) -> list[str]:
        if not self._r:
            raise RuntimeError("Call start() first")
        sessions: list[str] = []
        async for key in self._r.scan_iter(match=f"{_QUEUE_PREFIX}:*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            sessions.append(key_str[len(_QUEUE_PREFIX) + 1:])
        return sessions

    async def dequeue_inference_task(self, session_id: str) -> Optional[dict]:
        if not self._r:
            raise RuntimeError("Call start() first")
        raw = await self._r.blpop(
            self._queue_key(session_id), timeout=LIVE_BLPOP_TIMEOUT_SECONDS
        )
        if raw is None:
            return None
        _, payload = raw
        return json.loads(payload)

    @staticmethod
    def _queue_key(session_id: str) -> str:
        return f"{_QUEUE_PREFIX}:{session_id}"