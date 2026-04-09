"""
Inference Worker Test Suite
Run: pytest Test.py -v -o asyncio_mode=auto -o python_files=Test.py -o python_classes=Test
"""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
import torch


# ══════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════

@pytest.fixture
def dummy_crops_b64():
    """Base64-encoded dummy crops matching model input sizes."""
    crops = {}
    for name, size in [("face", 224), ("eyes", 64), ("mouth", 64),
                        ("cheeks", 64), ("forehead", 64)]:
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        crops[name] = base64.b64encode(buf.tobytes()).decode("utf-8")
    return crops


@pytest.fixture
def model():
    from app.Model import MultiStreamEmotionNet
    return MultiStreamEmotionNet(num_classes=7)


@pytest.fixture
def predictor():
    """Predictor with random weights (no checkpoint file needed)."""
    from app.Predictor import Predictor
    pred = Predictor()
    pred.load()  # Will use random weights since no file exists
    return pred


@pytest.fixture
def mock_producer():
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


# ══════════════════════════════════════════════
# Model architecture tests
# ══════════════════════════════════════════════

class TestModel:
    def test_model_forward_shapes(self, model):
        batch = 2
        face = torch.randn(batch, 3, 224, 224)
        eyes = torch.randn(batch, 3, 64, 64)
        mouth = torch.randn(batch, 3, 64, 64)
        cheek = torch.randn(batch, 3, 64, 64)
        forehead = torch.randn(batch, 3, 64, 64)

        outputs = model(face, eyes, mouth, cheek, forehead)

        assert outputs["emotion"].shape == (batch, 7)
        assert outputs["intensity"].shape == (batch, 3)
        assert outputs["valence"].shape == (batch, 3)
        assert outputs["arousal"].shape == (batch, 2)

    def test_model_single_sample(self, model):
        face = torch.randn(1, 3, 224, 224)
        eyes = torch.randn(1, 3, 64, 64)
        mouth = torch.randn(1, 3, 64, 64)
        cheek = torch.randn(1, 3, 64, 64)
        forehead = torch.randn(1, 3, 64, 64)

        outputs = model(face, eyes, mouth, cheek, forehead)
        assert outputs["emotion"].shape == (1, 7)

    def test_model_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        # ResNet18 ~11M + 4 RegionCNNs ~1M each + attention + heads
        assert total > 10_000_000  # at least 10M params
        assert total < 25_000_000  # less than 25M

    def test_region_cnn_output_size(self):
        from app.Model import RegionCNN
        cnn = RegionCNN(out_features=256)
        x = torch.randn(4, 3, 64, 64)
        out = cnn(x)
        assert out.shape == (4, 256)

    def test_feature_attention_preserves_shape(self):
        from app.Model import FeatureAttention
        attn = FeatureAttention(in_features=1536)
        x = torch.randn(4, 1536)
        out = attn(x)
        assert out.shape == (4, 1536)

    def test_attention_output_bounded(self):
        """Attention uses sigmoid, so weights should be 0-1."""
        from app.Model import FeatureAttention
        attn = FeatureAttention(in_features=512)
        x = torch.randn(2, 512)
        out = attn(x)
        # Output is x * sigmoid(weights), so values depend on x
        # But sigmoid weights themselves are 0-1
        weights = attn.features(x)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0


# ══════════════════════════════════════════════
# Predictor tests
# ══════════════════════════════════════════════

class TestPredictor:
    def test_predictor_loads_without_checkpoint(self, predictor):
        assert predictor.model is not None
        assert predictor.device is not None

    def test_predict_returns_all_heads(self, predictor, dummy_crops_b64):
        result = predictor.predict(dummy_crops_b64)
        assert "emotion" in result
        assert "intensity" in result
        assert "valence" in result
        assert "arousal" in result

    def test_predict_emotion_format(self, predictor, dummy_crops_b64):
        result = predictor.predict(dummy_crops_b64)
        emotion = result["emotion"]
        assert "label" in emotion
        assert "confidence" in emotion
        assert "probabilities" in emotion
        assert emotion["label"] in [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ]
        assert 0 <= emotion["confidence"] <= 1

    def test_predict_probabilities_sum_to_one(self, predictor, dummy_crops_b64):
        result = predictor.predict(dummy_crops_b64)
        for head_name in ("emotion", "intensity", "valence", "arousal"):
            probs = list(result[head_name]["probabilities"].values())
            assert abs(sum(probs) - 1.0) < 0.01, f"{head_name} probs don't sum to 1"

    def test_predict_intensity_labels(self, predictor, dummy_crops_b64):
        result = predictor.predict(dummy_crops_b64)
        assert result["intensity"]["label"] in ["low", "medium", "high"]
        assert set(result["intensity"]["probabilities"].keys()) == {"low", "medium", "high"}

    def test_predict_valence_labels(self, predictor, dummy_crops_b64):
        result = predictor.predict(dummy_crops_b64)
        assert result["valence"]["label"] in ["negative", "neutral", "positive"]

    def test_predict_arousal_labels(self, predictor, dummy_crops_b64):
        result = predictor.predict(dummy_crops_b64)
        assert result["arousal"]["label"] in ["low", "high"]

    def test_decode_crop(self, dummy_crops_b64):
        from app.Predictor import _decode_crop
        rgb = _decode_crop(dummy_crops_b64["face"])
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3  # RGB

    def test_preprocess_shapes(self, predictor, dummy_crops_b64):
        tensors = predictor._preprocess(dummy_crops_b64)
        assert tensors["face"].shape == (1, 3, 224, 224)
        assert tensors["eyes"].shape == (1, 3, 64, 64)
        assert tensors["mouth"].shape == (1, 3, 64, 64)
        assert tensors["cheek"].shape == (1, 3, 64, 64)
        assert tensors["forehead"].shape == (1, 3, 64, 64)


# ══════════════════════════════════════════════
# Worker tests
# ══════════════════════════════════════════════

class TestInferenceWorker:
    @pytest.mark.asyncio
    async def test_handle_task_success(self, dummy_crops_b64):
        from app.Worker import InferenceWorker
        worker = InferenceWorker()
        worker.predictor.load()

        task = {
            "session_id": "test-session",
            "frame_index": 0,
            "detection_index": 0,
            "crops": dummy_crops_b64,
        }
        result = await worker._handle_task(task)
        assert result["session_id"] == "test-session"
        assert result["frame_index"] == 0
        assert result["detection_index"] == 0
        assert result["status"] == "success"
        assert result["error"] is None
        assert result["processing_ms"] > 0
        assert "emotion" in result["predictions"]

    @pytest.mark.asyncio
    async def test_handle_task_error(self):
        from app.Worker import InferenceWorker
        worker = InferenceWorker()
        worker.predictor.load()

        task = {
            "session_id": "test-session",
            "frame_index": 0,
            "detection_index": 0,
            "crops": {"face": "bad", "eyes": "bad", "mouth": "bad",
                      "cheeks": "bad", "forehead": "bad"},
        }
        result = await worker._handle_task(task)
        assert result["status"] == "error"
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_handle_task_preserves_ids(self, dummy_crops_b64):
        from app.Worker import InferenceWorker
        worker = InferenceWorker()
        worker.predictor.load()

        task = {
            "session_id": "sess-abc",
            "frame_index": 42,
            "detection_index": 3,
            "crops": dummy_crops_b64,
        }
        result = await worker._handle_task(task)
        assert result["session_id"] == "sess-abc"
        assert result["frame_index"] == 42
        assert result["detection_index"] == 3


# ══════════════════════════════════════════════
# Kafka integration tests
# ══════════════════════════════════════════════

class TestKafkaIntegration:
    @pytest.mark.asyncio
    async def test_publish_result(self, mock_producer):
        from app.Kafka import publish_result
        result = {
            "session_id": "s1",
            "frame_index": 0,
            "detection_index": 0,
            "worker_id": "test-worker",
            "status": "success",
        }
        await publish_result(mock_producer, result)
        mock_producer.send_and_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_result_includes_worker_header(self, mock_producer):
        from app.Kafka import publish_result
        result = {"worker_id": "w1", "session_id": "s1",
                  "frame_index": 0, "detection_index": 0}
        await publish_result(mock_producer, result)
        headers = mock_producer.send_and_wait.call_args[1]["headers"]
        assert headers == [("worker_id", b"w1")]


# ══════════════════════════════════════════════
# End-to-end: crops → prediction
# ══════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline_crops_to_prediction(self, predictor, dummy_crops_b64):
        """Full path: base64 crops → preprocess → model → formatted output."""
        result = predictor.predict(dummy_crops_b64)

        # All heads present
        assert set(result.keys()) == {"emotion", "intensity", "valence", "arousal"}

        # Emotion has correct structure
        assert result["emotion"]["label"] in [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ]
        assert 0 <= result["emotion"]["confidence"] <= 1
        assert len(result["emotion"]["probabilities"]) == 7

        # All probs sum to ~1
        for head in result.values():
            total = sum(head["probabilities"].values())
            assert abs(total - 1.0) < 0.01