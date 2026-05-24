"""OV-MER for CMU-MOSEI (multilabel, pre-extracted features)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion.mosei_fusion import BaselineFusion
from baselines.ovmer.ovmer_model import SemanticPrototype

MOSEI_CLASS_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


class OVMERMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        use_audio: bool = True,
        use_video: bool = True,
        text_model: str = "roberta-base",
        modality_dropout: float = 0.1,
        dropout: float = 0.2,
        clip_embed_dim: int = 512,
        w_proto: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.w_proto = w_proto
        self.base = BaselineFusion(
            num_classes=num_classes, d_model=d_model,
            use_audio=use_audio, use_video=use_video,
            text_model=text_model, modality_dropout=modality_dropout,
        )
        # Prototype head (operates on d_model space)
        self.proto_proj = nn.Linear(d_model, num_classes)
        self.refine = nn.Sequential(
            nn.Linear(d_model + num_classes, d_model),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.register_buffer("pos_weight", torch.ones(num_classes))

    def forward(self, batch):
        logits_base, z = self.base(batch)
        proto_sim = self.proto_proj(z)
        logits = self.refine(torch.cat([proto_sim, z], dim=-1))
        return logits, z, {"proto_sim": proto_sim}

    def loss(self, logits, y, z, aux):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        if "proto_sim" in aux:
            proto_bce = F.binary_cross_entropy_with_logits(
                aux["proto_sim"], y, pos_weight=self.pos_weight
            )
            total = (1 - self.w_proto) * bce + self.w_proto * proto_bce
        else:
            total = bce
        return total, bce
