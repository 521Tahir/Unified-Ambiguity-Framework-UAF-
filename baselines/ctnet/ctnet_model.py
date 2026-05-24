"""
CTNet: Conversational Transformer Network (Lian et al., 2021)
"CTNet: Conversational Transformer Network for Emotion Recognition"
https://arxiv.org/abs/2109.06820

Key ideas adapted here:
  - Modality-specific transformer encoders (intra-modal)
  - Cross-modal interaction via bidirectional cross-attention
  - Sentiment-aware gating for final fusion
  - Utterance-level classification (dialog context removed for non-conversation datasets)
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.shared_encoders import TrimodalEncoders


class SentimentGate(nn.Module):
    """
    Sentiment-aware gate: learns how much of each modality to pass
    based on inter-modal agreement signal.
    """

    def __init__(self, d_model: int, num_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_modalities, num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, M, D]
        B, M, D = feats.shape
        g = self.gate(feats.reshape(B, M * D))  # [B, M]
        return feats * g.unsqueeze(-1)           # [B, M, D]


class BidirCrossAttn(nn.Module):
    """
    Bidirectional cross-modal attention between two modalities.
    Returns updated representations for both.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_ab = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a, b: [B, T, D] sequences
        a_new, _ = self.attn_ab(a, b, b)
        b_new, _ = self.attn_ba(b, a, a)
        return self.norm_a(a + a_new), self.norm_b(b + b_new)


class CTNetModel(nn.Module):
    """
    Full CTNet model adapted for utterance-level multimodal emotion recognition.
    Uses same encoder backbone as CHADOTrimodal for fair comparison.
    """

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,
        proj_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
    ):
        super().__init__()
        self.modalities = [m for m, u in [("text", use_text), ("audio", use_audio), ("video", use_video)] if u]
        M = len(self.modalities)

        self.encoders = TrimodalEncoders(
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            video_model_name=video_model_name,
            proj_dim=proj_dim,
            dropout=dropout,
            use_text=use_text,
            use_audio=use_audio,
            use_video=use_video,
        )

        # Intra-modal self-attention
        self.intra = nn.ModuleDict({
            m: nn.TransformerEncoderLayer(
                d_model=proj_dim, nhead=n_heads, dim_feedforward=proj_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for m in self.modalities
        })

        # Bidirectional cross-modal attention for all pairs
        self.cross_pairs = nn.ModuleDict()
        mods = self.modalities
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                key = f"{mods[i]}_{mods[j]}"
                self.cross_pairs[key] = BidirCrossAttn(proj_dim, n_heads, dropout)

        # Sentiment-aware gate
        self.sent_gate = SentimentGate(proj_dim, M)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim * M),
            nn.Linear(proj_dim * M, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        audio_wave: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        modality_mask=None,
    ):
        _, seqs = self.encoders(text_input, audio_wave, video_frames)

        # Intra-modal encoding
        encoded = {m: self.intra[m](seqs[m]) for m in self.modalities}

        # Bidirectional cross-modal interaction
        mods = self.modalities
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                key = f"{mods[i]}_{mods[j]}"
                a_new, b_new = self.cross_pairs[key](encoded[mods[i]], encoded[mods[j]])
                encoded[mods[i]] = a_new
                encoded[mods[j]] = b_new

        # Pool each modality sequence → [B, D]
        pooled = {m: encoded[m].mean(dim=1) for m in self.modalities}

        # Stack → [B, M, D]
        node_feats = torch.stack([pooled[m] for m in self.modalities], dim=1)

        if modality_mask is not None:
            node_feats = node_feats * modality_mask.unsqueeze(-1)

        # Sentiment-aware gating
        node_feats = self.sent_gate(node_feats)

        B, M, D = node_feats.shape
        z = node_feats.reshape(B, M * D)
        logits = self.classifier(z)
        return logits, z, None
