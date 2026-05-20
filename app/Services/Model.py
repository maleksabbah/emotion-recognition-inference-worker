import torch
import torch.nn as nn
from torchvision import models


class RegionCNN(nn.Module):
    """Simple 3-layer RegionCNN (V2 baseline)."""
    def __init__(self, out_features=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


class RegionTransformerFusion(nn.Module):
    """
    Transformer-based fusion for multi-stream face region features.
    Takes 5 region feature vectors, projects to common dimension,
    adds positional embeddings, runs self-attention, and concatenates
    all enriched tokens for classification.
    """
    def __init__(self, face_dim=512, region_dim=256, embed_dim=256,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_regions = 5

        self.face_proj = nn.Linear(face_dim, embed_dim)
        self.eyes_proj = nn.Linear(region_dim, embed_dim)
        self.mouth_proj = nn.Linear(region_dim, embed_dim)
        self.cheeks_proj = nn.Linear(region_dim, embed_dim)
        self.forehead_proj = nn.Linear(region_dim, embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_regions, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, face_feat, eye_feat, mouth_feat, cheek_feat, forehead_feat):
        face_tok = self.face_proj(face_feat)
        eyes_tok = self.eyes_proj(eye_feat)
        mouth_tok = self.mouth_proj(mouth_feat)
        cheeks_tok = self.cheeks_proj(cheek_feat)
        forehead_tok = self.forehead_proj(forehead_feat)

        tokens = torch.stack([face_tok, eyes_tok, mouth_tok, cheeks_tok, forehead_tok], dim=1)
        tokens = tokens + self.pos_embedding
        tokens = self.transformer(tokens)
        tokens = self.output_norm(tokens)
        fused = tokens.reshape(tokens.size(0), -1)

        return fused


class MultiStreamEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Face stream: ResNet18 pretrained
        resnet = models.resnet18(weights='DEFAULT')
        resnet.fc = nn.Identity()
        self.face_stream = resnet  # outputs 512d

        # Region streams (simple 3-layer)
        self.eye_stream = RegionCNN(256)
        self.mouth_stream = RegionCNN(256)
        self.cheek_stream = RegionCNN(256)
        self.forehead_stream = RegionCNN(256)

        # Transformer fusion
        self.fusion = RegionTransformerFusion(
            face_dim=512,
            region_dim=256,
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )

        # Shared FC: 1280 → 512 (all heads read from this)
        self.shared_fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # All heads branch from the shared 512d
        self.emotion_head = nn.Linear(512, num_classes)        # 7 emotions
        self.valence_head = nn.Linear(512, 3)                  # negative, neutral, positive
        self.arousal_head = nn.Linear(512, 2)                  # low, high
        self.intensity_head = nn.Linear(512, 3)                # low, medium, high

        # Aliases so UNetTraining.py optimizer param groups work unchanged
        # (it references emotion_fc, valence_fc, arousal_fc, intensity_fc)
        self.emotion_fc = self.shared_fc
        self.valence_fc = self.shared_fc
        self.arousal_fc = self.shared_fc
        self.intensity_fc = self.shared_fc

    def forward(self, face, eyes, mouth, cheek, forehead):
        face_feat = self.face_stream(face)
        eye_feat = self.eye_stream(eyes)
        mouth_feat = self.mouth_stream(mouth)
        cheek_feat = self.cheek_stream(cheek)
        forehead_feat = self.forehead_stream(forehead)

        fused = self.fusion(face_feat, eye_feat, mouth_feat, cheek_feat, forehead_feat)

        # Shared projection — one forward pass, all heads read from it
        shared = self.shared_fc(fused)

        return {
            "emotion": self.emotion_head(shared),
            "valence": self.valence_head(shared),
            "arousal": self.arousal_head(shared),
            "intensity": self.intensity_head(shared),
        }


if __name__ == "__main__":
    model = MultiStreamEmotionNet(num_classes=7)

    face = torch.randn(4, 3, 224, 224)
    eyes = torch.randn(4, 3, 64, 64)
    mouth = torch.randn(4, 3, 64, 64)
    cheeks = torch.randn(4, 3, 64, 64)
    forehead = torch.randn(4, 3, 64, 64)

    outputs = model(face, eyes, mouth, cheeks, forehead)
    for key, val in outputs.items():
        print(f"{key}: {val.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"Fusion module params: {fusion_params:,}")
