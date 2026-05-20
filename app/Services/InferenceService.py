"""
InferenceService — wraps Predictor into the typed task/result protocol.

InferenceTask (with 5 base64 crops) → run model → InferenceResult.
"""
from __future__ import annotations

import logging
import time

from app.Config import WORKER_ID
from app.Dtos.TaskDto.InferenceResult import EmotionScores, InferenceResult
from app.Dtos.TaskDto.InferenceTask import InferenceTask
from app.Services.Predictor import Predictor

logger = logging.getLogger("inference-worker.service")


class InferenceService:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    async def process_task(self, task: InferenceTask) -> InferenceResult:
        start = time.perf_counter()

        crops = {
            "face": task.face_crop,
            "eyes": task.region_crops.eyes,
            "mouth": task.region_crops.mouth,
            "cheeks": task.region_crops.cheeks,
            "forehead": task.region_crops.forehead,
        }

        # Run model — produces 4 heads (emotion, intensity, valence, arousal)
        output = self.predictor.predict(crops)

        # Pull emotion probabilities into typed scores
        emo_probs = output["emotion"]["probabilities"]
        scores = EmotionScores(
            angry=emo_probs.get("angry", 0.0),
            disgust=emo_probs.get("disgust", 0.0),
            fear=emo_probs.get("fear", 0.0),
            happy=emo_probs.get("happy", 0.0),
            neutral=emo_probs.get("neutral", 0.0),
            sad=emo_probs.get("sad", 0.0),
            surprise=emo_probs.get("surprise", 0.0),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResult(
            task_id=task.task_id,
            session_id=task.session_id,
            frame_number=task.frame_number,
            face_index=task.face_index,
            emotions=scores,
            top_emotion=output["emotion"]["label"],
            top_confidence=output["emotion"]["confidence"],
            valence=output["valence"]["label"],
            arousal=output["arousal"]["label"],
            intensity=output["intensity"]["label"],
            inference_time_ms=elapsed_ms,
            worker_id=WORKER_ID,
        )