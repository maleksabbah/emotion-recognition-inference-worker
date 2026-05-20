"""
Inference integration tests — single file.
Run from inference-worker/ root: `pytest Test.py -v`

No containers needed — Predictor runs in-process with random weights
(or your real checkpoint if MODEL_PATH is set).

Requires:
  pytest.ini with session-scoped loops (see storage)
"""
from __future__ import annotations

import base64
import io
import uuid

import pytest
import pytest_asyncio
from PIL import Image


# ══════════════════════════════════════════════
# Session-scoped predictor (loading is slow)
# ══════════════════════════════════════════════

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def predictor():
    from app.Services.Predictor import Predictor
    p = Predictor()
    p.load()  # random weights if MODEL_PATH not set
    yield p


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _real_jpeg_b64(size: int = 64) -> str:
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _inference_task() -> dict:
    region = _real_jpeg_b64(64)
    return {
        "task_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "frame_number": 0,
        "face_index": 0,
        "track_id": 0,
        "bbox": {"x": 0, "y": 0, "w": 100, "h": 100},
        "face_crop": _real_jpeg_b64(224),
        "region_crops": {"eyes": region, "mouth": region, "cheeks": region, "forehead": region},
    }


# ══════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════

@pytest.mark.asyncio(loop_scope="session")
async def test_inference_task_produces_typed_result(predictor):
    from app.Dtos.TaskDto.InferenceTask import InferenceTask
    from app.Services.InferenceService import InferenceService

    inference = InferenceService(predictor=predictor)
    task = InferenceTask.model_validate(_inference_task())
    result = await inference.process_task(task)

    assert result is not None
    assert result.task_id == task.task_id


@pytest.mark.asyncio(loop_scope="session")
async def test_inference_output_labels_are_valid(predictor):
    from app.Dtos.TaskDto.InferenceTask import InferenceTask
    from app.Services.InferenceService import InferenceService

    inference = InferenceService(predictor=predictor)
    result = await inference.process_task(
        InferenceTask.model_validate(_inference_task())
    )

    assert result.top_emotion in {
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
    }
    assert 0.0 <= result.top_confidence <= 1.0
    assert result.valence in {"negative", "neutral", "positive"}
    assert result.arousal in {"low", "medium", "high"}
    assert result.intensity in {"low", "medium", "high"}


@pytest.mark.asyncio(loop_scope="session")
async def test_inference_probabilities_sum_to_one(predictor):
    from app.Dtos.TaskDto.InferenceTask import InferenceTask
    from app.Services.InferenceService import InferenceService

    inference = InferenceService(predictor=predictor)
    result = await inference.process_task(
        InferenceTask.model_validate(_inference_task())
    )

    e = result.emotions
    total = e.angry + e.disgust + e.fear + e.happy + e.neutral + e.sad + e.surprise
    assert 0.99 < total < 1.01