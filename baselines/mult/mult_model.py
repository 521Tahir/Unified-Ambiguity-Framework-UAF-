"""
MulT: Multimodal Transformer (Tsai et al., 2019)
"Multimodal Transformer for Unaligned Multimodal Language Sequences"
https://arxiv.org/abs/1906.00295

Architecture:
  - For each directional pair (src → tgt): cross-modal attention transformer
    Q from target, K/V from source
  - 6 directional streams for trimodal: T→A, T→V, A→T, A→V, V→T, V→A
  - Final: concat all streams → MLP → classify
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.shared_encoders import TrimodalEncoders


class CrossModalTransformerLayer(nn.Module):
    """
    One cross-modal transformer layer.
    Q from tgt_seq, K/V from src_seq.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        # tgt: [B, T_tgt, D], src: [B, T_src, D]
        attn_out, _ = self.attn(tgt, src, src)
        tgt = self.norm1(tgt + self.drop(attn_out))
        tgt = self.norm2(tgt + self.drop(self.ffn(tgt)))
        return tgt


class CrossModalStream(nn.Module):
    """Stack of n_layers cross-modal transformer layers."""

    def __init__(self, d_model: int, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, src)
        return tgt.mean(dim=1)   # pool to [B, D]


class MulTModel(nn.Module):
    """
    Full MulT model.
    Uses same encoder backbone as CHADOTrimodal for fair comparison.
    Supports text-only (T), text+audio (TA), text+video (TV), trimodal (TAV).
    """

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,
        proj_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video
        self.modalities = [m for m, u in [("text", use_text), ("audio", use_audio), ("video", use_video)] if u]

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

        # Cross-modal streams: all ordered pairs (src -> tgt)
        self.cross_streams = nn.ModuleDict()
        for src in self.modalities:
            for tgt in self.modalities:
                if src != tgt:
                    self.cross_streams[f"{src}_{tgt}"] = CrossModalStream(
                        proj_dim, n_layers, n_heads, dropout
                    )

        # Self-attention on each modality
        self.self_streams = nn.ModuleDict({
            m: CrossModalStream(proj_dim, 1, n_heads, dropout)
            for m in self.modalities
        })

        # Number of cross-modal + self output vectors
        n_streams = len(self.modalities) * len(self.modalities)  # M*M (incl. self)
        classifier_in = n_streams * proj_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, 512),
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

        parts = []

        # Self + cross-modal streams
        for tgt in self.modalities:
            # Self-attention
            tgt_seq = seqs[tgt]
            self_out = self.self_streams[tgt](tgt_seq, tgt_seq)
            parts.append(self_out)

            # Cross-modal attention (src → tgt)
            for src in self.modalities:
                if src != tgt:
                    out = self.cross_streams[f"{src}_{tgt}"](tgt_seq, seqs[src])
                    parts.append(out)

        z = torch.cat(parts, dim=-1)
        logits = self.classifier(z)
        return logits, z, None

def main():
    model = MulTModel(
        text_model_name="roberta-base",
        audio_model_name="facebook/wav2vec2-base",
        video_model_name="google/vit-base-patch16-224",
        num_classes=6,
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")


if __name__ == "__main__":
    main()