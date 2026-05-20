"""
Inference worker entry point.

Two concurrent loops, both feeding into InferenceService.process_task:
  - kafka_loop: pull InferenceTask from inference_tasks topic   (batch path)
  - redis_loop: scan active sessions, blpop their queue         (live path)

Results always publish to Kafka (inference_results), so orchestrator's
PipelineService handles both modes uniformly.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from app.Config import LIVE_SCAN_INTERVAL_SECONDS, WORKER_ID
from app.Dtos.TaskDto.InferenceTask import InferenceTask
from app.Repositories.KafkaConsumer import KafkaConsumer
from app.Repositories.KafkaProducer import KafkaProducer
from app.Repositories.RedisRepository import RedisRepository
from app.Services.InferenceService import InferenceService
from app.Services.Predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("inference-worker")


# ─── Loop bodies ──────────────────────────────────────────────────────

async def kafka_loop(
    consumer: KafkaConsumer,
    producer: KafkaProducer,
    inference: InferenceService,
) -> None:
    async for raw in consumer.consume():
        try:
            task = InferenceTask.model_validate(raw)
        except Exception as e:
            logger.error("Bad InferenceTask from Kafka: %s", e)
            continue
        await _handle(task, inference, producer)


async def redis_loop(
    redis_repo: RedisRepository,
    producer: KafkaProducer,
    inference: InferenceService,
) -> None:
    while True:
        sessions = await redis_repo.scan_active_sessions()
        if not sessions:
            await asyncio.sleep(LIVE_SCAN_INTERVAL_SECONDS)
            continue

        for sid in sessions:
            raw = await redis_repo.dequeue_inference_task(sid)
            if raw is None:
                continue
            try:
                task = InferenceTask.model_validate(raw)
            except Exception as e:
                logger.error("Bad InferenceTask from Redis: %s", e)
                continue
            await _handle(task, inference, producer)


async def _handle(
    task: InferenceTask,
    inference: InferenceService,
    producer: KafkaProducer,
) -> None:
    try:
        result = await inference.process_task(task)
    except Exception as e:
        logger.exception("process_task failed: %s", e)
        return
    if result is None:
        return
    await producer.publish_inference_result(result.model_dump())


# ─── Entry ────────────────────────────────────────────────────────────

async def run() -> None:
    logger.info("Inference worker starting (id=%s)", WORKER_ID)

    predictor = Predictor()
    predictor.load()

    redis_repo = RedisRepository()
    consumer = KafkaConsumer()
    producer = KafkaProducer()

    await redis_repo.start()
    await consumer.start()
    await producer.start()
    logger.info("Worker ready")

    inference = InferenceService(predictor=predictor)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    kafka_task = asyncio.create_task(kafka_loop(consumer, producer, inference))
    redis_task = asyncio.create_task(redis_loop(redis_repo, producer, inference))

    await stop.wait()
    logger.info("Shutting down...")

    kafka_task.cancel()
    redis_task.cancel()
    for t in (kafka_task, redis_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    await consumer.stop()
    await producer.stop()
    await redis_repo.stop()
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(run())