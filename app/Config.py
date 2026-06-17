"""
Inference worker configuration — env vars + model constants.
"""
from __future__ import annotations

import os
import uuid
import torch


# ─── Identity ──────────────────────────────────────────────────────────

WORKER_ID = os.getenv("WORKER_ID", f"inference-{uuid.uuid4().hex[:8]}")


# ─── Kafka ─────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
INFERENCE_TASKS_TOPIC = os.getenv("INFERENCE_TASKS_TOPIC", "inference_tasks")
INFERENCE_RESULTS_TOPIC = os.getenv("INFERENCE_RESULTS_TOPIC", "inference_results")
INFERENCE_GROUP_ID = os.getenv("INFERENCE_GROUP_ID", "inference-workers")


# ─── Redis (live mode) ─────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}")

LIVE_BLPOP_TIMEOUT_SECONDS = float(os.getenv("LIVE_BLPOP_TIMEOUT_SECONDS", "0.5"))
LIVE_SCAN_INTERVAL_SECONDS = float(os.getenv("LIVE_SCAN_INTERVAL_SECONDS", "1.0"))


# ─── Model ─────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "/models/multi_stream_emotion.pt")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "1"))

EMOTION_LABELS = [
    "angry", "happy", "sad", "surprise", "neutral", "fear", "disgust",
]
INTENSITY_LABELS = ["low", "medium", "high"]
VALENCE_LABELS = ["negative", "neutral", "positive"]
AROUSAL_LABELS = ["low", "high"]
