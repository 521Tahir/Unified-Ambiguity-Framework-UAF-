"""CTNet for CMU-MOSEI (pre-extracted features, multi-label)."""
import torch
import torch.nn as nn

from baselines.ctnet.ctnet_model import BidirCrossAttn, SentimentGate
from baselines.mosei_feature_encoders import MOSEIFeatureEncoders


class CTNetMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        text_model: str = "roberta-base",
        use_audio: bool = True,
        use_video: bool = True,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.modalities = ["text"] + (["audio"] if use_audio else []) + (["video"] if use_video else [])
        M = len(self.modalities)

        self.encoders = MOSEIFeatureEncoders(
            text_model=text_model, d_model=d_model,
            use_audio=use_audio, use_video=use_video, dropout=dropout,
        )
        self.intra = nn.ModuleDict({
            m: nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            ) for m in self.modalities
        })
        mods = self.modalities
        self.cross_pairs = nn.ModuleDict({
            f"{mods[i]}_{mods[j]}": BidirCrossAttn(d_model, n_heads, dropout)
            for i in range(len(mods)) for j in range(i + 1, len(mods))
        })
        self.sent_gate = SentimentGate(d_model, M)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * M),
            nn.Linear(d_model * M, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, batch):
        _, seqs = self.encoders(batch["text"], batch.get("audio"), batch.get("video"))
        encoded = {m: self.intra[m](seqs[m]) for m in self.modalities}
        mods = self.modalities
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a_new, b_new = self.cross_pairs[f"{mods[i]}_{mods[j]}"](encoded[mods[i]], encoded[mods[j]])
                encoded[mods[i]], encoded[mods[j]] = a_new, b_new
        node_feats = torch.stack([encoded[m].mean(dim=1) for m in self.modalities], dim=1)
        node_feats = self.sent_gate(node_feats)
        B, M, D = node_feats.shape
        z = node_feats.reshape(B, M * D)
        return self.classifier(z), z
