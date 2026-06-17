"""
UNetModel_emotion.py — matches emotion_model.pth (118.9MB, Apr 26)

Architecture from checkpoint keys:
  - Face: InceptionResnetV1 (VGGFace2), 512-d
  - Regions: BlurPool RegionCNN (3 conv blocks, BlurPool at indices 4,9, conv at 0,5,10), 256-d
  - Fusion: RegionTransformerFusion (5 tokens, 256-d embed, 4 heads, 2 layers, 1024 FFN)
  - shared_fc: 1280->512
  - All FCs are aliases to shared_fc (all [512, 1280])
  - Heads: emotion [7,512], valence [3,512], arousal [2,512], intensity [3,512]
"""
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class BlurPool(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride
        self.channels = channels
        filt = torch.tensor([1., 3., 3., 1.])
        filt = filt[:, None] * filt[None, :]
        filt = filt / filt.sum()
        filt = filt[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer('filt', filt)

    def forward(self, x):
        return nn.functional.conv2d(
            x, self.filt, stride=self.stride,
            padding=1, groups=self.channels
        )


class RegionCNN(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),        # 0
            nn.BatchNorm2d(32),                      # 1
            nn.ReLU(),                               # 2
            nn.Identity(),                           # 3 (padding to keep indices)
            BlurPool(32, stride=2),                  # 4 (.filt)

            nn.Conv2d(32, 64, 3, padding=1),        # 5
            nn.BatchNorm2d(64),                      # 6
            nn.ReLU(),                               # 7
            nn.Identity(),                           # 8
            BlurPool(64, stride=2),                  # 9 (.filt)

            nn.Conv2d(64, 128, 3, padding=1),       # 10
            nn.BatchNorm2d(128),                     # 11
            nn.ReLU(),                               # 12
            nn.AdaptiveAvgPool2d(1),                 # 13
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


class RegionTransformerFusion(nn.Module):
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
        tokens = torch.stack([
            self.face_proj(face_feat),
            self.eyes_proj(eye_feat),
            self.mouth_proj(mouth_feat),
            self.cheeks_proj(cheek_feat),
            self.forehead_proj(forehead_feat),
        ], dim=1)
        tokens = tokens + self.pos_embedding
        tokens = self.transformer(tokens)
        tokens = self.output_norm(tokens)
        return tokens.reshape(tokens.size(0), -1)


class MultiStreamEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.face_stream = InceptionResnetV1(pretrained='vggface2')

        self.eye_stream = RegionCNN(256)
        self.mouth_stream = RegionCNN(256)
        self.cheek_stream = RegionCNN(256)
        self.forehead_stream = RegionCNN(256)

        self.fusion = RegionTransformerFusion(
            face_dim=512, region_dim=256, embed_dim=256,
            num_heads=4, num_layers=2, dropout=0.1
        )

        # Shared FC
        self.shared_fc = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.5),
        )

        # Aliases — all point to shared_fc
        self.emotion_fc = self.shared_fc
        self.valence_fc = self.shared_fc
        self.arousal_fc = self.shared_fc
        self.intensity_fc = self.shared_fc

        self.emotion_head = nn.Linear(512, num_classes)
        self.valence_head = nn.Linear(512, 3)
        self.arousal_head = nn.Linear(512, 2)
        self.intensity_head = nn.Linear(512, 3)

    def forward(self, face, eyes, mouth, cheek, forehead):
        face_feat = self.face_stream(face)
        eye_feat = self.eye_stream(eyes)
        mouth_feat = self.mouth_stream(mouth)
        cheek_feat = self.cheek_stream(cheek)
        forehead_feat = self.forehead_stream(forehead)

        fused = self.fusion(face_feat, eye_feat, mouth_feat, cheek_feat, forehead_feat)
        shared = self.shared_fc(fused)

        return {
            "emotion": self.emotion_head(shared),
            "valence": self.valence_head(shared),
            "arousal": self.arousal_head(shared),
            "intensity": self.intensity_head(shared),
        }