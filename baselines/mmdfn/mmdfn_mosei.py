"""MM-DFN for CMU-MOSEI (pre-extracted features, multi-label)."""
import torch
import torch.nn as nn

from baselines.mmdfn.mmdfn_model import IntraModalEncoder, DynamicGraphFusion
from baselines.mosei_feature_encoders import MOSEIFeatureEncoders


class MMDFNMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        text_model: str = "roberta-base",
        use_audio: bool = True,
        use_video: bool = True,
        n_heads: int = 4,
        n_graph_layers: int = 2,
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
            m: IntraModalEncoder(d_model, n_heads, dropout) for m in self.modalities
        })
        self.graph_layers = nn.ModuleList([
            DynamicGraphFusion(d_model, M, dropout) for _ in range(n_graph_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * M),
            nn.Linear(d_model * M, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, batch):
        _, seqs = self.encoders(batch["text"], batch.get("audio"), batch.get("video"))
        node_feats = torch.stack([self.intra[m](seqs[m]) for m in self.modalities], dim=1)
        for gfn in self.graph_layers:
            node_feats = gfn(node_feats)
        B, M, D = node_feats.shape
        z = node_feats.reshape(B, M * D)
        return self.classifier(z), z
