"""
Shared trimodal encoders used by ALL baselines.
Returns both pooled vectors and (optionally) sequences for MulT.
Same backbone as CHADOTrimodal for fair comparison.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class TrimodalEncoders(nn.Module):
    """
    RoBERTa (text) + wav2vec2 (audio) + ViT (video).
    proj_dim: all modalities projected to the same dim.
    Returns:
      - pooled: dict {modality: [B, proj_dim]}
      - seqs:   dict {modality: [B, T, proj_dim]}  (for MulT)
    """

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        proj_dim: int = 256,
        dropout: float = 0.1,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video

        if use_text:
            self.text_enc = AutoModel.from_pretrained(text_model_name)
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_enc.config.hidden_size, proj_dim),
                nn.ReLU(), nn.Dropout(dropout),
            )

        if use_audio:
            self.audio_enc = AutoModel.from_pretrained(audio_model_name)
            self.audio_proj = nn.Sequential(
                nn.Linear(self.audio_enc.config.hidden_size, proj_dim),
                nn.ReLU(), nn.Dropout(dropout),
            )

        if use_video:
            self.video_enc = AutoModel.from_pretrained(video_model_name)
            self.video_proj = nn.Sequential(
                nn.Linear(self.video_enc.config.hidden_size, proj_dim),
                nn.ReLU(), nn.Dropout(dropout),
            )

    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        audio_wave: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict, Dict]:
        pooled, seqs = {}, {}

        if self.use_text and text_input is not None:
            out = self.text_enc(**text_input)
            h = out.last_hidden_state            # [B, L, D]
            pooled["text"] = self.text_proj(h[:, 0])      # CLS
            seqs["text"] = self.text_proj(h)              # [B, L, proj_dim]

        if self.use_audio and audio_wave is not None:
            out = self.audio_enc(input_values=audio_wave)
            h = out.last_hidden_state            # [B, T_a, D]
            pooled["audio"] = self.audio_proj(h.mean(dim=1))
            seqs["audio"] = self.audio_proj(h)            # [B, T_a, proj_dim]

        if self.use_video and video_frames is not None:
            B, T, C, H, W = video_frames.shape
            h = self.video_enc(pixel_values=video_frames.reshape(B * T, C, H, W))
            cls = h.last_hidden_state[:, 0].reshape(B, T, -1)   # [B, T, D_v]
            pooled["video"] = self.video_proj(cls.mean(dim=1))
            seqs["video"] = self.video_proj(cls)                 # [B, T, proj_dim]

        return pooled, seqs
