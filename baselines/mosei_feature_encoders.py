"""
Feature-based encoders for CMU-MOSEI baselines.
Takes pre-extracted features (text strings + audio [B,T,74] + video [B,T,35]).
Returns both pooled [B,D] and sequences [B,T,D] per modality.
Used by MulT-MOSEI, MM-DFN-MOSEI, CTNet-MOSEI for fair comparison.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class MOSEIFeatureEncoders(nn.Module):
    """
    MOSEI-specific encoder that mirrors TrimodalEncoders interface
    but works with pre-extracted COVAREP (74-dim) and Facet42 (35-dim) features.
    Text uses RoBERTa (raw strings).
    """

    def __init__(
        self,
        text_model: str = "roberta-base",
        d_model: int = 256,
        audio_in_dim: int = 74,
        video_in_dim: int = 35,
        max_text_len: int = 96,
        dropout: float = 0.1,
        use_audio: bool = True,
        use_video: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.max_text_len = max_text_len
        self.use_audio = use_audio
        self.use_video = use_video

        # Text: RoBERTa → CLS token + full sequence
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        self.text_model = AutoModel.from_pretrained(text_model)
        t_dim = self.text_model.config.hidden_size
        self.text_proj = nn.Sequential(nn.Linear(t_dim, d_model), nn.ReLU(), nn.Dropout(dropout))

        # Audio: 74-dim COVAREP → projected sequence
        if use_audio:
            self.audio_norm = nn.LayerNorm(audio_in_dim)
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_in_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
            )

        # Video: 35-dim Facet42 → projected sequence
        if use_video:
            self.video_norm = nn.LayerNorm(video_in_dim)
            self.video_proj = nn.Sequential(
                nn.Linear(video_in_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
            )

    def forward(
        self,
        texts: List[str],
        audio: Optional[torch.Tensor] = None,   # [B, T_a, 74]
        video: Optional[torch.Tensor] = None,   # [B, T_v, 35]
    ) -> Tuple[Dict, Dict]:
        pooled, seqs = {}, {}
        device = next(self.parameters()).device

        # Text
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.text_model(**enc)
        h = out.last_hidden_state                          # [B, L, t_dim]
        text_seq = self.text_proj(h)                       # [B, L, d_model]
        pooled["text"] = text_seq[:, 0]                    # CLS token
        seqs["text"] = text_seq

        # Audio
        if self.use_audio and audio is not None:
            a = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            a_seq = self.audio_proj(self.audio_norm(a))    # [B, T_a, d_model]
            pooled["audio"] = a_seq.mean(dim=1)
            seqs["audio"] = a_seq

        # Video
        if self.use_video and video is not None:
            v = torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
            v_seq = self.video_proj(self.video_norm(v))    # [B, T_v, d_model]
            pooled["video"] = v_seq.mean(dim=1)
            seqs["video"] = v_seq

        return pooled, seqs
