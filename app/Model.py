"""
MultiStreamEmotionNet — your custom architecture.

ResNet18 face stream + 4 regional CNNs (eyes, mouth, cheeks, forehead)
with feature attention and multi-head output.
"""
import torch
import torch.nn as nn
from torchvision import models


class RegionCNN(nn.Module):
    def __init__(self, out_features=256):
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

            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


class FeatureAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.features(x)
        return x * weights


class MultiStreamEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        resnet = models.resnet18(weights='DEFAULT')
        resnet.fc = nn.Identity()
        self.face_stream = resnet

        self.eye_stream = RegionCNN(256)
        self.mouth_stream = RegionCNN(256)
        self.cheek_stream = RegionCNN(256)
        self.forehead_stream = RegionCNN(256)

        self.attention = FeatureAttention(1536)

        self.shared_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.emotion_head = nn.Linear(512, num_classes)
        self.intensity_head = nn.Linear(512, 3)
        self.valence_head = nn.Linear(512, 3)
        self.arousal_head = nn.Linear(512, 2)

    def forward(self, face, eyes, mouth, cheek, forehead):
        face_feat = self.face_stream(face)
        eye_feat = self.eye_stream(eyes)
        mouth_feat = self.mouth_stream(mouth)
        cheek_feat = self.cheek_stream(cheek)
        forehead_feat = self.forehead_stream(forehead)

        combined = torch.cat([
            face_feat, eye_feat, mouth_feat,
            cheek_feat, forehead_feat
        ], dim=1)

        attended = self.attention(combined)
        shared = self.shared_fc(attended)

        return {
            "emotion": self.emotion_head(shared),
            "intensity": self.intensity_head(shared),
            "valence": self.valence_head(shared),
            "arousal": self.arousal_head(shared),
        }