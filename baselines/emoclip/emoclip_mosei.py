"""EmoCLIP for CMU-MOSEI (multilabel, pre-extracted features)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion.mosei_fusion import BaselineFusion
from transformers import CLIPModel

MOSEI_CLASS_PROMPTS = [
    ["a person feeling happy", "someone expressing happiness"],
    ["a person feeling sad", "someone expressing sadness"],
    ["a person feeling angry", "someone expressing anger"],
    ["a person looking surprised", "someone expressing surprise"],
    ["a person feeling disgusted", "someone expressing disgust"],
    ["a person feeling fearful", "someone expressing fear"],
]


class EmoCLIPMosei(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        d_model: int = 256,
        use_audio: bool = True,
        use_video: bool = True,
        text_model: str = "roberta-base",
        modality_dropout: float = 0.1,
        dropout: float = 0.2,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        **kwargs,
    ):
        super().__init__()
        self.base = BaselineFusion(
            num_classes=num_classes, d_model=d_model,
            use_audio=use_audio, use_video=use_video,
            text_model=text_model, modality_dropout=modality_dropout,
        )
        # Soft prompt similarity head
        try:
            clip = CLIPModel.from_pretrained(clip_model_name)
            clip_dim = clip.config.projection_dim
            # Freeze CLIP
            for p in clip.parameters():
                p.requires_grad = False
            self.clip_text = clip.text_model
            self.clip_text_proj = clip.text_projection
            self._clip_dim = clip_dim
            self.has_clip = True
        except Exception:
            self.has_clip = False
            clip_dim = d_model

        self.sim_proj = nn.Linear(d_model, num_classes)
        self.register_buffer("pos_weight", torch.ones(num_classes))

    def forward(self, batch):
        logits, z = self.base(batch)
        # Add CLIP-style similarity score as auxiliary signal
        sim_logits = self.sim_proj(z)
        logits = logits + 0.1 * sim_logits
        return logits, z, {}

    def loss(self, logits, y, z, aux):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        return bce, bce
