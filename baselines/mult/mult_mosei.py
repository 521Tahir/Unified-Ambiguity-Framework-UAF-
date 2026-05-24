"""
MulT for CMU-MOSEI (pre-extracted features, multi-label).
Same cross-modal attention architecture as MulTModel,
but uses MOSEIFeatureEncoders instead of TrimodalEncoders.
"""
from typing import Optional

import torch
import torch.nn as nn

from baselines.mult.mult_model import CrossModalStream
from baselines.mosei_feature_encoders import MOSEIFeatureEncoders


class MulTMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        text_model: str = "roberta-base",
        use_audio: bool = True,
        use_video: bool = True,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.modalities = ["text"] + (["audio"] if use_audio else []) + (["video"] if use_video else [])

        self.encoders = MOSEIFeatureEncoders(
            text_model=text_model, d_model=d_model,
            use_audio=use_audio, use_video=use_video, dropout=dropout,
        )

        self.cross_streams = nn.ModuleDict()
        for src in self.modalities:
            for tgt in self.modalities:
                if src != tgt:
                    self.cross_streams[f"{src}_{tgt}"] = CrossModalStream(d_model, n_layers, n_heads, dropout)

        self.self_streams = nn.ModuleDict({
            m: CrossModalStream(d_model, 1, n_heads, dropout) for m in self.modalities
        })

        n_streams = len(self.modalities) ** 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_streams * d_model),
            nn.Linear(n_streams * d_model, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, batch):
        texts = batch["text"]
        audio = batch.get("audio", None)
        video = batch.get("video", None)

        _, seqs = self.encoders(texts, audio, video)

        parts = []
        for tgt in self.modalities:
            parts.append(self.self_streams[tgt](seqs[tgt], seqs[tgt]))
            for src in self.modalities:
                if src != tgt:
                    parts.append(self.cross_streams[f"{src}_{tgt}"](seqs[tgt], seqs[src]))

        z = torch.cat(parts, dim=-1)
        return self.classifier(z), z
