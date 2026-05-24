"""UniMSE for CMU-MOSEI (multilabel, pre-extracted features)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion.mosei_fusion import BaselineFusion


class UniMSEMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        use_audio: bool = True,
        use_video: bool = True,
        text_model: str = "roberta-base",
        modality_dropout: float = 0.1,
        dropout: float = 0.2,
        w_contrastive: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.w_contrastive = w_contrastive
        self.base = BaselineFusion(
            num_classes=num_classes, d_model=d_model,
            use_audio=use_audio, use_video=use_video,
            text_model=text_model, modality_dropout=modality_dropout,
        )
        self.align_proj = nn.Linear(d_model, d_model)
        self.register_buffer("pos_weight", torch.ones(num_classes))

    def forward(self, batch):
        logits, z = self.base(batch)
        return logits, z, {}

    def loss(self, logits, y, z, aux):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        # Contrastive: pull same-label pattern samples together
        z_norm = F.normalize(self.align_proj(z), dim=-1)
        B = z_norm.size(0)
        if B > 1:
            sim = torch.matmul(z_norm, z_norm.T) / 0.07
            # Use Hamming similarity as soft positive labels
            label_sim = 1.0 - (y.unsqueeze(0) - y.unsqueeze(1)).abs().mean(-1)
            mask = 1 - torch.eye(B, device=z.device)
            log_sm = F.log_softmax(sim * mask - 1e4 * (1 - mask), dim=1)
            pos = label_sim * mask
            pos = pos / (pos.sum(1, keepdim=True) + 1e-8)
            contrastive = -(pos * log_sm).sum(1).mean()
            if not torch.isfinite(contrastive):
                contrastive = z.sum() * 0.0
        else:
            contrastive = z.sum() * 0.0
        total = bce + self.w_contrastive * contrastive
        return total, bce
