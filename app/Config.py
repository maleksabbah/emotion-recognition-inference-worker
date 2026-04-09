"""
Inference Worker configuration — loaded from environment variables.
"""
import os

# ── Kafka ──────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "inference-worker-group")

TOPIC_INFERENCE_TASKS = os.getenv("TOPIC_INFERENCE_TASKS", "inference_tasks")
TOPIC_INFERENCE_RESULTS = os.getenv("TOPIC_INFERENCE_RESULTS", "inference_results")

# ── Model ──────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"

# ── Labels ─────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
INTENSITY_LABELS = ["low", "medium", "high"]
VALENCE_LABELS = ["negative", "neutral", "positive"]
AROUSAL_LABELS = ["low", "high"]

# ── Worker ─────────────────────────────────────────────
WORKER_ID = os.getenv("WORKER_ID", "inference-worker-1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))