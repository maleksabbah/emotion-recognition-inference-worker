"""
Inference Worker — stateless Kafka consumer (GPU-bound).

Consumes crops from inference_tasks, runs MultiStreamEmotionNet,
publishes predictions to inference_results.

Task schema (from orchestrator):
{
    "session_id": str,
    "frame_index": int,
    "detection_index": int,
    "crops": {
        "face": str,       # base64 JPEG
        "eyes": str,
        "mouth": str,
        "cheeks": str,
        "forehead": str
    }
}

Result schema (published to inference_results):
{
    "session_id": str,
    "frame_index": int,
    "detection_index": int,
    "worker_id": str,
    "status": "success" | "error",
    "predictions": {
        "emotion":    { "label": str, "confidence": float, "probabilities": {...} },
        "intensity":  { ... },
        "valence":    { ... },
        "arousal":    { ... }
    },
    "processing_ms": float,
    "error": str | null
}
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any

from app.Config import WORKER_ID
from app.Kafka import create_consumer, create_producer, publish_result
from app.Predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("inference-worker")


class InferenceWorker:
    def __init__(self):
        self.predictor = Predictor()
        self.consumer = None
        self.producer = None
        self._running = False

    async def start(self):
        # Load model (synchronous — runs once on startup)
        self.predictor.load()

        self.consumer = await create_consumer()
        self.producer = await create_producer()
        self._running = True
        logger.info("Inference worker %s started", WORKER_ID)

    async def stop(self):
        self._running = False
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logger.info("Inference worker %s stopped", WORKER_ID)

    async def _handle_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a single inference task."""
        session_id = task.get("session_id", "unknown")
        frame_index = task.get("frame_index", -1)
        detection_index = task.get("detection_index", 0)

        start = time.perf_counter()

        try:
            # Run inference in executor (model forward pass is CPU/GPU bound)
            predictions = await asyncio.get_event_loop().run_in_executor(
                None, self.predictor.predict, task["crops"]
            )
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            return {
                "session_id": session_id,
                "frame_index": frame_index,
                "detection_index": detection_index,
                "worker_id": WORKER_ID,
                "status": "success",
                "predictions": predictions,
                "processing_ms": elapsed,
                "error": None,
            }

        except Exception as e:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.error("Error on session=%s frame=%d det=%d: %s",
                         session_id, frame_index, detection_index, e)
            return {
                "session_id": session_id,
                "frame_index": frame_index,
                "detection_index": detection_index,
                "worker_id": WORKER_ID,
                "status": "error",
                "predictions": None,
                "processing_ms": elapsed,
                "error": str(e),
            }

    async def run(self):
        """Main consumer loop."""
        await self.start()

        try:
            async for message in self.consumer:
                if not self._running:
                    break

                task = message.value
                logger.info(
                    "Received task: session=%s frame=%s det=%s",
                    task.get("session_id"),
                    task.get("frame_index"),
                    task.get("detection_index"),
                )

                result = await self._handle_task(task)
                await publish_result(self.producer, result)
                await self.consumer.commit()

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
        finally:
            await self.stop()


async def main():
    worker = InferenceWorker()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(worker)))

    await worker.run()


async def _shutdown(worker: InferenceWorker):
    logger.info("Shutdown signal received")
    worker._running = False


if __name__ == "__main__":
    asyncio.run(main())