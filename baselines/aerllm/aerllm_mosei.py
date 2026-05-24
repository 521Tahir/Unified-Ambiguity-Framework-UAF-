"""
AER-LLM for CMU-MOSEI (multilabel, pre-extracted features).
Same AER architecture adapted for BaselineFusion feature-based pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion.mosei_fusion import BaselineFusion
from baselines.aerllm.aerllm_model import AmbiguityContrastiveHead


class AERLLMMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        use_audio: bool = True,
        use_video: bool = True,
        text_model: str = "roberta-base",
        modality_dropout: float = 0.1,
        dropout: float = 0.2,
        contrastive_dim: int = 128,
        w_contrastive: float = 0.1,
        w_entropy: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.w_contrastive = w_contrastive
        self.w_entropy = w_entropy

        self.base = BaselineFusion(
            num_classes=num_classes,
            d_model=d_model,
            use_audio=use_audio,
            use_video=use_video,
            text_model=text_model,
            modality_dropout=modality_dropout,
        )
        self.aer_head = AmbiguityContrastiveHead(d_model, contrastive_dim)
        self.register_buffer("pos_weight", torch.ones(num_classes))

    def forward(self, batch):
        logits, z = self.base(batch)
        z_proj = self.aer_head(z)
        return logits, z, z_proj, {}

    def loss(self, logits, y, z, aux, z_proj=None):
        base_bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

        with torch.no_grad():
            probs = torch.sigmoid(logits.detach())
            # Per-sample entropy for multilabel
            entropy = -(probs * (probs + 1e-8).log()
                        + (1 - probs) * (1 - probs + 1e-8).log()).mean(dim=-1)

        # Entropy-weighted BCE
        w = 1.0 + self.w_entropy * entropy.detach()
        w = w / w.mean()
        per_sample = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(1)
        entropy_bce = (w * per_sample).mean()

        contrastive = 0.0
        if z_proj is not None:
            contrastive = AmbiguityContrastiveHead.ambiguity_contrastive_loss(z_proj, entropy)

        total = (1 - self.w_contrastive) * entropy_bce + self.w_contrastive * contrastive
        return total, base_bce
