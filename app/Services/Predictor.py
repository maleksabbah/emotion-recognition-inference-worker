"""
Predictor — loads MultiStreamEmotionNet and runs inference on crops.

Preprocessing is PIL-based, identical to the training pipeline. The crops
arrive base64-encoded; we decode them through PIL (not cv2) so every step
matches what the model saw during training:

    PIL.Image.open(bytes).convert("RGB") → Resize → ToTensor → Normalize

Using cv2 here introduces a BGR/RGB hop that drifts from training and was
producing wrong emotion predictions on real faces.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.Config import (
    AROUSAL_LABELS,
    DEVICE,
    EMOTION_LABELS,
    INTENSITY_LABELS,
    MODEL_PATH,
    NUM_CLASSES,
    VALENCE_LABELS,
)
from app.Services.Model import MultiStreamEmotionNet

logger = logging.getLogger("inference-worker.predictor")


# ── Preprocessing — PIL-based, matches training exactly ──────

FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

REGION_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _decode_crop(b64: str) -> Image.Image:
    """Decode base64 JPEG bytes to a PIL RGB image (training-style)."""
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


class Predictor:
    def __init__(self):
        self.model: Optional[MultiStreamEmotionNet] = None
        self.device: Optional[torch.device] = None

    def load(self):
        """Load model weights from checkpoint."""
        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE)
        logger.info("Loading model from %s on %s", MODEL_PATH, self.device)

        self.model = MultiStreamEmotionNet(num_classes=NUM_CLASSES)

        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(
                    "Loaded from checkpoint (epoch %s, acc %.2f%%)",
                    checkpoint.get("epoch", "?"),
                    checkpoint.get("val_accuracy", 0) * 100,
                )
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded raw state dict")
        except FileNotFoundError:
            logger.warning("No checkpoint at %s — using random weights (dev mode)", MODEL_PATH)

        self.model.to(self.device)
        self.model.eval()
        logger.info("Model ready on %s", self.device)

    def _preprocess(self, crops: dict[str, str]) -> dict[str, torch.Tensor]:
        """
        Convert base64 crops to batched tensors.
        Input:  {"face": b64, "eyes": b64, "mouth": b64, "cheeks": b64, "forehead": b64}
        Output: {"face": (1,3,224,224), "eyes"/"mouth"/"cheek"/"forehead": (1,3,64,64)}
        """
        face_pil = _decode_crop(crops["face"])
        face_t = FACE_TRANSFORM(face_pil).unsqueeze(0).to(self.device)

        regions = {}
        for name in ("eyes", "mouth", "cheeks", "forehead"):
            pil = _decode_crop(crops[name])
            regions[name] = REGION_TRANSFORM(pil).unsqueeze(0).to(self.device)

        return {
            "face":     face_t,
            "eyes":     regions["eyes"],
            "mouth":    regions["mouth"],
            "cheek":    regions["cheeks"],  # model param is "cheek" (singular)
            "forehead": regions["forehead"],
        }

    def _format_head(self, logits: torch.Tensor, labels: list[str]) -> dict[str, Any]:
        """Convert logits to label + probabilities."""
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        top_idx = int(torch.argmax(logits, dim=1).item())
        # Guard against label list being longer than output dim
        if top_idx >= len(labels):
            top_idx = top_idx % len(labels)
        return {
            "label": labels[top_idx],
            "confidence": round(probs[top_idx], 4),
            "probabilities": {
                label: round(prob, 4) for label, prob in zip(labels, probs)
            },
        }

    @torch.no_grad()
    def predict(self, crops: dict[str, str]) -> dict[str, Any]:
        """
        Run inference on a set of crops.
        Input:  {"face": b64, "eyes": b64, "mouth": b64, "cheeks": b64, "forehead": b64}
        Output: {"emotion": {...}, "intensity": {...}, "valence": {...}, "arousal": {...}}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded — call load() first")

        tensors = self._preprocess(crops)
        outputs = self.model(
            tensors["face"],
            tensors["eyes"],
            tensors["mouth"],
            tensors["cheek"],
            tensors["forehead"],
        )
        return {
            "emotion":   self._format_head(outputs["emotion"],   EMOTION_LABELS),
            "intensity": self._format_head(outputs["intensity"], INTENSITY_LABELS),
            "valence":   self._format_head(outputs["valence"],   VALENCE_LABELS),
            "arousal":   self._format_head(outputs["arousal"],   AROUSAL_LABELS),
        }
