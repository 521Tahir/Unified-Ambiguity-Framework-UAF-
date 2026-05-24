"""
MM-DFN: Multimodal Dynamic Fusion Network (Hu et al., 2021)
"MM-DFN: Multimodal Dynamic Fusion Network for Emotion Recognition in Conversations"
https://arxiv.org/abs/2203.02385

Architecture:
  - Intra-modal encoding (self-attention per modality)
  - Inter-modal graph-based dynamic fusion
    * Dynamic adjacency: learned attention between modality feature pairs
    * Graph convolution step: each node aggregates from its neighbors
  - Fusion: weighted sum of inter-modal outputs + classifier
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.shared_encoders import TrimodalEncoders


class IntraModalEncoder(nn.Module):
    """Self-attention encoder per modality."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h, _ = self.attn(x, x, x)
        x = self.norm(x + h)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)   # [B, D]


class DynamicGraphFusion(nn.Module):
    """
    Dynamic graph-based inter-modal fusion.
    Nodes = modalities. Adjacency = learned from feature pairs.
    Each node aggregates from all neighbors via softmax-weighted sum.
    """

    def __init__(self, d_model: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.M = num_modalities
        # Pairwise edge weight: (h_i, h_j) -> scalar
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        # Node update MLP
        self.node_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        # node_feats: [B, M, D]
        B, M, D = node_feats.shape
        updated = []

        for i in range(M):
            # Compute edge weights from node i to all others
            hi = node_feats[:, i:i+1].expand(B, M, D)   # [B, M, D]
            pairs = torch.cat([hi, node_feats], dim=-1)  # [B, M, 2D]
            weights = self.edge_mlp(pairs).squeeze(-1)   # [B, M]
            weights = F.softmax(weights, dim=-1)         # [B, M]

            # Weighted neighbor aggregation
            agg = (node_feats * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

            # Node update
            new_hi = self.node_update(torch.cat([node_feats[:, i], agg], dim=-1))
            new_hi = self.norm(new_hi + node_feats[:, i])
            updated.append(new_hi)

        return torch.stack(updated, dim=1)   # [B, M, D]


class MMDFNModel(nn.Module):
    """
    Full MM-DFN model.
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
        n_graph_layers: int = 2,
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

        self.intra = nn.ModuleDict({
            m: IntraModalEncoder(proj_dim, n_heads, dropout)
            for m in self.modalities
        })

        self.graph_layers = nn.ModuleList([
            DynamicGraphFusion(proj_dim, M, dropout)
            for _ in range(n_graph_layers)
        ])

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
        node_feats = torch.stack(
            [self.intra[m](seqs[m]) for m in self.modalities], dim=1
        )   # [B, M, D]

        if modality_mask is not None:
            node_feats = node_feats * modality_mask.unsqueeze(-1)

        # Dynamic graph fusion layers
        for gfn in self.graph_layers:
            node_feats = gfn(node_feats)

        B, M, D = node_feats.shape
        z = node_feats.reshape(B, M * D)
        logits = self.classifier(z)
        return logits, z, None
