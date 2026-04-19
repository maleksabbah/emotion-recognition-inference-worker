"""
Inference Worker — stateless Kafka consumer (GPU-bound).

Consumes crops from inference_tasks, runs MultiStreamEmotionNet,
publishes predictions to inference_results.

InferenceTask schema (from orchestrator):
{
    "task_id": str,
    "session_id": str,
    "detection_id": str,
    "frame_number": int,
    "timestamp_ms": float,
    "face_index": int,
    "track_id": str | null,
    "face_crop": str,           # base64 JPEG
    "region_crops": {
        "eyes": str,
        "mouth": str,
        "cheeks": str,
        "forehead": str
    },
    "priority": int
}

InferenceResult schema (published to inference_results):
{
    "task_id": str,
    "session_id": str,
    "detection_id": str,
    "frame_number": int,
    "timestamp_ms": float,
    "face_index": int,
    "track_id": str | null,
    "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
    "emotions": { "happy": float, "sad": float, ... },
    "top_emotion": str,
    "top_confidence": float,
    "valence": float,
    "arousal": float,
    "intensity": float,
    "inference_time_ms": float,
    "worker_id": str
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

    def _build_crops(self, task: dict[str, Any]) -> dict[str, str]:
        """
        Convert InferenceTask fields to the crops dict the Predictor expects.
        Predictor wants: {"face": b64, "eyes": b64, "mouth": b64, "cheeks": b64, "forehead": b64}
        """
        region_crops = task.get("region_crops", {})
        return {
            "face": task["face_crop"],
            "eyes": region_crops["eyes"],
            "mouth": region_crops["mouth"],
            "cheeks": region_crops["cheeks"],
            "forehead": region_crops["forehead"],
        }

    def _map_predictions_to_result(self, predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Convert Predictor output to InferenceResult fields.
        Predictor returns: {"emotion": {label, confidence, probabilities}, "intensity": ..., "valence": ..., "arousal": ...}
        InferenceResult needs: emotions, top_emotion, top_confidence, valence, arousal, intensity
        """
        emotion_pred = predictions.get("emotion", {})
        intensity_pred = predictions.get("intensity", {})
        valence_pred = predictions.get("valence", {})
        arousal_pred = predictions.get("arousal", {})

        return {
            "emotions": emotion_pred.get("probabilities", {}),
            "top_emotion": emotion_pred.get("label", "unknown"),
            "top_confidence": emotion_pred.get("confidence", 0.0),
            "valence": valence_pred.get("confidence", 0.0),
            "arousal": arousal_pred.get("confidence", 0.0),
            "intensity": intensity_pred.get("confidence", 0.0),
        }

    async def _handle_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a single inference task."""
        session_id = task.get("session_id", "unknown")
        task_id = task.get("task_id", "unknown")
        detection_id = task.get("detection_id", "unknown")
        frame_number = task.get("frame_number", 0)
        timestamp_ms = task.get("timestamp_ms", 0.0)
        face_index = task.get("face_index", 0)
        track_id = task.get("track_id")

        start = time.perf_counter()

        try:
            # Build crops dict from InferenceTask fields
            crops = self._build_crops(task)

            # Run inference in executor (model forward pass is CPU/GPU bound)
            predictions = await asyncio.get_event_loop().run_in_executor(
                None, self.predictor.predict, crops
            )
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            # Map predictor output to InferenceResult schema
            mapped = self._map_predictions_to_result(predictions)

            return {
                "task_id": task_id,
                "session_id": session_id,
                "detection_id": detection_id,
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "face_index": face_index,
                "track_id": track_id,
                "bbox": task.get("bbox", {"x": 0, "y": 0, "w": 0, "h": 0}),
                "emotions": mapped["emotions"],
                "top_emotion": mapped["top_emotion"],
                "top_confidence": mapped["top_confidence"],
                "valence": mapped["valence"],
                "arousal": mapped["arousal"],
                "intensity": mapped["intensity"],
                "inference_time_ms": elapsed,
                "worker_id": WORKER_ID,
            }

        except Exception as e:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.error("Error on session=%s frame=%d det=%s: %s",
                         session_id, frame_number, detection_id, e)
            return {
                "task_id": task_id,
                "session_id": session_id,
                "detection_id": detection_id,
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "face_index": face_index,
                "track_id": track_id,
                "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                "emotions": {},
                "top_emotion": "error",
                "top_confidence": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "intensity": 0.0,
                "inference_time_ms": elapsed,
                "worker_id": WORKER_ID,
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
                    task.get("frame_number"),
                    task.get("detection_id"),
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
