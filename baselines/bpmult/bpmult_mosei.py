"""
BPMulT for CMU-MOSEI.
Ref: Based on MulT with Gated Multimodal Unit (GMU) for dynamic modal tradeoff.
Adds a learnable gating mechanism that weights each modality's contribution
based on content relevance — following BPMulT (Bipolar MulT with gating).
"""
import torch
import torch.nn as nn

from baselines.mult.mult_model import CrossModalStream
from baselines.mosei_feature_encoders import MOSEIFeatureEncoders


class GatedModalUnit(nn.Module):
    """Gated Multimodal Unit: learns dynamic per-sample modality weights."""
    def __init__(self, d_model: int, n_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_modalities, n_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, modal_feats):
        # modal_feats: list of [B, d_model]
        cat = torch.cat(modal_feats, dim=-1)          # [B, n_mod * d_model]
        weights = self.gate(cat).unsqueeze(-1)         # [B, n_mod, 1]
        stacked = torch.stack(modal_feats, dim=1)      # [B, n_mod, d_model]
        return (stacked * weights).sum(dim=1)          # [B, d_model]


class BPMulTMosei(nn.Module):
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
        n_mod = len(self.modalities)

        self.encoders = MOSEIFeatureEncoders(
            text_model=text_model, d_model=d_model,
            use_audio=use_audio, use_video=use_video, dropout=dropout,
        )

        # Cross-modal streams (same as MulT)
        self.cross_streams = nn.ModuleDict()
        for src in self.modalities:
            for tgt in self.modalities:
                if src != tgt:
                    self.cross_streams[f"{src}_{tgt}"] = CrossModalStream(d_model, n_layers, n_heads, dropout)

        self.self_streams = nn.ModuleDict({
            m: CrossModalStream(d_model, 1, n_heads, dropout) for m in self.modalities
        })

        # Gated Multimodal Unit per target modality
        self.gmu = nn.ModuleDict({
            tgt: GatedModalUnit(d_model, n_mod) for tgt in self.modalities
        })

        # Classifier over GMU outputs
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * n_mod),
            nn.Linear(d_model * n_mod, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, batch):
        texts = batch["text"]
        audio = batch.get("audio", None)
        video = batch.get("video", None)

        _, seqs = self.encoders(texts, audio, video)

        gmu_outs = []
        for tgt in self.modalities:
            modal_feats = [self.self_streams[tgt](seqs[tgt], seqs[tgt])]
            for src in self.modalities:
                if src != tgt:
                    modal_feats.append(self.cross_streams[f"{src}_{tgt}"](seqs[tgt], seqs[src]))
            gated = self.gmu[tgt](modal_feats)
            gmu_outs.append(gated)

        z = torch.cat(gmu_outs, dim=-1)
        return self.classifier(z), z
