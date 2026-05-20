"""
Inference worker DTOs.

  TaskDto/
    InferenceTask     (+ Bbox, RegionCrops)    orchestrator → worker
    InferenceResult   (+ EmotionScores)        worker → orchestrator
"""
from app.Dtos.TaskDto import (
    InferenceTask, Bbox, RegionCrops,
    InferenceResult, EmotionScores,
)

__all__ = [
    "InferenceTask", "Bbox", "RegionCrops",
    "InferenceResult", "EmotionScores",
]