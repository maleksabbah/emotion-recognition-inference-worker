"""
KafkaConsumer — pulls InferenceTask messages off inference_tasks topic.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from aiokafka import AIOKafkaConsumer

from app.Config import (
    INFERENCE_GROUP_ID,
    INFERENCE_TASKS_TOPIC,
    KAFKA_BOOTSTRAP_SERVERS,
)

logger = logging.getLogger("inference-worker.kafka-consumer")


class KafkaConsumer:
    def __init__(self) -> None:
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            INFERENCE_TASKS_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=INFERENCE_GROUP_ID,
            enable_auto_commit=True,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()

    async def consume(self) -> AsyncIterator[dict]:
        if not self._consumer:
            raise RuntimeError("Call start() before consume()")
        async for record in self._consumer:
            try:
                yield json.loads(record.value.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error("Bad JSON from %s: %s", INFERENCE_TASKS_TOPIC, e)
                continue