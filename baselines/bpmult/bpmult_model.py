
"""
Optimized BPMulT for Maximum GPU Utilization
DDP + AMP + torch.compile optimized
"""

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")

from baselines.shared_encoders import TrimodalEncoders
from baselines.mult.mult_model import CrossModalStream


class GatedModalUnit(nn.Module):
    def __init__(self, d_model: int, n_modalities: int):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model * n_modalities, 512),
            nn.GELU(),
            nn.Linear(512, n_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, modal_feats):
        """
        modal_feats:
            List[(B, D)]
        """

        # (B, D * M)
        cat = torch.cat(modal_feats, dim=-1)

        # (B, M)
        weights = self.gate(cat)

        # (B, M, D)
        stacked = torch.stack(modal_feats, dim=1)

        # (B, M, 1)
        weights = weights.unsqueeze(-1)

        # Weighted fusion
        fused = (stacked * weights).sum(dim=1)

        return fused


class BPMulTModel(nn.Module):

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,

        # Larger dimensions for tensor-core saturation
        proj_dim: int = 768,

        # Deeper transformer
        n_layers: int = 6,

        # Better GPU occupancy
        n_heads: int = 12,

        dropout: float = 0.1,

        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
    ):
        super().__init__()

        self.modalities = (
            (["text"] if use_text else []) +
            (["audio"] if use_audio else []) +
            (["video"] if use_video else [])
        )

        n_mod = len(self.modalities)

        # ---------------------------------------------------
        # Shared Encoders
        # ---------------------------------------------------
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

        # ---------------------------------------------------
        # Cross-modal streams
        # ---------------------------------------------------
        self.cross_streams = nn.ModuleDict()

        for src in self.modalities:
            for tgt in self.modalities:

                if src == tgt:
                    continue

                self.cross_streams[f"{src}_{tgt}"] = CrossModalStream(
                    proj_dim,
                    n_layers,
                    n_heads,
                    dropout
                )

        # ---------------------------------------------------
        # Self-attention streams
        # ---------------------------------------------------
        self.self_streams = nn.ModuleDict({
            m: CrossModalStream(
                proj_dim,
                2,
                n_heads,
                dropout
            )
            for m in self.modalities
        })

        # ---------------------------------------------------
        # GMU fusion
        # ---------------------------------------------------
        self.gmu = nn.ModuleDict({
            tgt: GatedModalUnit(proj_dim, n_mod)
            for tgt in self.modalities
        })

        # ---------------------------------------------------
        # Large classifier head
        # ---------------------------------------------------
        self.classifier = nn.Sequential(

            nn.LayerNorm(proj_dim * n_mod),

            nn.Linear(proj_dim * n_mod, 2048),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024, num_classes),
        )

    def forward(
        self,
        text_input=None,
        audio_wave=None,
        video_frames=None
    ):

        # ---------------------------------------------------
        # Encoder forward
        # ---------------------------------------------------
        _, seqs = self.encoders(
            text_input,
            audio_wave,
            video_frames
        )

        outputs = []

        # ---------------------------------------------------
        # Parallel multimodal fusion
        # ---------------------------------------------------
        for tgt in self.modalities:

            tgt_seq = seqs[tgt]

            modal_feats = []

            # Self-stream
            self_feat = self.self_streams[tgt](
                tgt_seq,
                tgt_seq
            )

            modal_feats.append(self_feat)

            # Cross-streams
            for src in self.modalities:

                if src == tgt:
                    continue

                cross_feat = self.cross_streams[
                    f"{src}_{tgt}"
                ](
                    tgt_seq,
                    seqs[src]
                )

                modal_feats.append(cross_feat)

            # GMU fusion
            fused = self.gmu[tgt](modal_feats)

            outputs.append(fused)

        # ---------------------------------------------------
        # Final fusion
        # ---------------------------------------------------
        z = torch.cat(outputs, dim=-1)

        logits = self.classifier(z)

        return logits, z