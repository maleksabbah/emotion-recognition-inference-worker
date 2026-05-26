"""
MultiStreamEmotionNet — V2 (matches the 82.46% baseline checkpoint).

Architecture:
  - Face stream: ResNet18 (ImageNet pretrained), 512-d output
  - 4 region streams: simple 3-layer RegionCNN, 256-d each
  - Concat -> 1536-d
  - FeatureAttention: sigmoid gating on the 1536-d concat
  - shared_fc: 1536 -> 512 -> ReLU -> Dropout
  - 4 heads (each share the 512-d bottleneck via alias):
        emotion (num_classes), intensity (3), valence (3), arousal (2)

Inputs (all in [0,1] range, ImageNet-normalized):
    face:     [B, 3, 224, 224]
    eyes:     [B, 3,  64,  64]
    mouth:    [B, 3,  64,  64]
    cheek:    [B, 3,  64,  64]
    forehead: [B, 3,  64,  64]

Predictor.py calls forward() with keyword `cheek=` (singular), so the
parameter name must stay `cheek`, not `cheeks`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# ══════════════════════════════════════════════════════════
# Region stream — simple 3-layer CNN with BatchNorm
# ══════════════════════════════════════════════════════════

class RegionCNN(nn.Module):
    """Three stride-2 convs (64→32→16→8), GAP, FC to out_features."""

    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.fc(x)


# ══════════════════════════════════════════════════════════
# Feature attention — sigmoid-gated reweighting of the concat
# ══════════════════════════════════════════════════════════

class FeatureAttention(nn.Module):
    """Bottleneck FC -> sigmoid mask -> elementwise multiply."""

    def __init__(self, in_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.features(x)


# ══════════════════════════════════════════════════════════
# Main model
# ══════════════════════════════════════════════════════════

class MultiStreamEmotionNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Face stream — ResNet18, drop final FC, output 512-d
        resnet = models.resnet18(weights="DEFAULT")
        resnet.fc = nn.Identity()
        self.face_stream = resnet

        # Region streams — 4 simple CNNs, 256-d each
        self.eye_stream = RegionCNN(256)
        self.mouth_stream = RegionCNN(256)
        self.cheek_stream = RegionCNN(256)
        self.forehead_stream = RegionCNN(256)

        # Attention on concat (512 + 4*256 = 1536)
        self.attention = FeatureAttention(1536)

        # Shared bottleneck
        self.shared_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Heads — emit independent logits but share the 512-d bottleneck
        self.emotion_head = nn.Linear(512, num_classes)
        self.intensity_head = nn.Linear(512, 3)
        self.valence_head = nn.Linear(512, 3)
        self.arousal_head = nn.Linear(512, 2)

        # Aliases — the training script (UNetTraining.py) referenced these by
        # name in its optimizer param-group lookup. The checkpoint includes
        # weights under these alias keys; pointing them at shared_fc keeps
        # load_state_dict(strict=True) happy.
        self.emotion_fc = self.shared_fc
        self.valence_fc = self.shared_fc
        self.arousal_fc = self.shared_fc
        self.intensity_fc = self.shared_fc

    def forward(
        self,
        face: torch.Tensor,
        eyes: torch.Tensor,
        mouth: torch.Tensor,
        cheek: torch.Tensor,
        forehead: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        face_feat = self.face_stream(face)              # [B, 512]
        eye_feat = self.eye_stream(eyes)                # [B, 256]
        mouth_feat = self.mouth_stream(mouth)           # [B, 256]
        cheek_feat = self.cheek_stream(cheek)           # [B, 256]
        forehead_feat = self.forehead_stream(forehead)  # [B, 256]

        combined = torch.cat(
            [face_feat, eye_feat, mouth_feat, cheek_feat, forehead_feat],
            dim=1,
        )  # [B, 1536]

        attended = self.attention(combined)
        shared = self.shared_fc(attended)

        return {
            "emotion": self.emotion_head(shared),
            "intensity": self.intensity_head(shared),
            "valence": self.valence_head(shared),
            "arousal": self.arousal_head(shared),
        }