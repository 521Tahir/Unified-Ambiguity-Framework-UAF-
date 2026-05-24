"""
MESCA: Modality Emotion Semantic Correlation Analysis
Zhang et al., Computers and Electrical Engineering, 2025

Architecture:
  1. Single-modal feature extraction (shared encoders: RoBERTa + wav2vec2 + ViT)
  2. RepVGG-style reparameterizable projection blocks per modality
  3. Deep CCA correlation module: T(Hi, Hj) = Σ11^(-1/2) Σ12 Σ22^(-1/2)
     Each modality Ui = concat(Hi, Corr(i→j), Corr(i→k))
  4. MLP classifier on concat(Ut, Ua, Uv)

Adaptation: uses same encoders as other baselines for fair comparison.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.shared_encoders import TrimodalEncoders


class RepVGGBlock(nn.Module):
    """1D RepVGG block: multi-branch training → single-branch inference."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.main   = nn.Linear(dim, dim)   # 3×3 branch equivalent
        self.skip1  = nn.Linear(dim, dim)   # 1×1 branch equivalent
        self.bn     = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)
        self._deploy = False
        self._fused: Optional[nn.Linear] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._deploy and self._fused is not None:
            return F.relu(self.bn(self._fused(x)))
        out = self.main(x) + self.skip1(x) + x   # identity + two branches
        return F.relu(self.bn(self.drop(out)))

    def switch_to_deploy(self):
        """Fuse all branches into a single linear layer (inference mode)."""
        w = self.main.weight.data + self.skip1.weight.data
        b = self.main.bias.data   + self.skip1.bias.data
        w = w + torch.eye(w.shape[0], device=w.device)   # identity branch
        self._fused = nn.Linear(w.shape[1], w.shape[0], bias=True, device=w.device)
        self._fused.weight = nn.Parameter(w)
        self._fused.bias   = nn.Parameter(b)
        self._deploy = True


class DeepCCACorr(nn.Module):
    """
    Batch Deep-CCA correlation feature.
    T(H1, H2) ≈ Σ11^{-1/2} Σ12 Σ22^{-1/2}  applied to H2 to produce H1-space corr.
    Uses learnable linear projection (Deep CCA) for trainability with SGD.
    """

    def __init__(self, d1: int, d2: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.proj1 = nn.Linear(d1, d_out)
        self.proj2 = nn.Linear(d2, d_out)
        self.gate  = nn.Sequential(nn.Linear(d_out * 2, d_out), nn.Sigmoid())
        self.drop  = nn.Dropout(dropout)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """h1: [B, d1], h2: [B, d2] → corr: [B, d_out]"""
        p1 = self.proj1(h1)
        p2 = self.proj2(h2)
        g  = self.gate(torch.cat([p1, p2], dim=-1))
        return self.drop(g * p1 + (1 - g) * p2)


class MESCAModel(nn.Module):
    """
    MESCA for IEMOCAP / MELD (raw trimodal inputs).
    """

    def __init__(
        self,
        text_model_name:  str = "roberta-base",
        audio_model_name: str = "facebook/wav2vec2-base",
        video_model_name: str = "google/vit-base-patch16-224",
        num_classes: int  = 4,
        proj_dim:    int  = 256,
        dropout:     float = 0.2,
        use_text:    bool = True,
        use_audio:   bool = True,
        use_video:   bool = True,
        n_repvgg:    int  = 3,
    ):
        super().__init__()
        self.use_text  = use_text
        self.use_audio = use_audio
        self.use_video = use_video
        self.proj_dim  = proj_dim

        self.encoders = TrimodalEncoders(
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            video_model_name=video_model_name,
            proj_dim=proj_dim,
            use_text=use_text,
            use_audio=use_audio,
            use_video=use_video,
        )

        # RepVGG blocks per active modality
        def _repvgg_stack(n):
            return nn.Sequential(*[RepVGGBlock(proj_dim, dropout) for _ in range(n)])

        self.text_repvgg  = _repvgg_stack(n_repvgg) if use_text  else None
        self.audio_repvgg = _repvgg_stack(n_repvgg) if use_audio else None
        self.video_repvgg = _repvgg_stack(n_repvgg) if use_video else None

        # Deep-CCA correlation modules (one per ordered pair)
        active = sum([use_text, use_audio, use_video])
        self.cca_pairs: nn.ModuleDict = nn.ModuleDict()
        mods = []
        if use_text:  mods.append("text")
        if use_audio: mods.append("audio")
        if use_video: mods.append("video")
        self._active_mods = mods

        for i, m1 in enumerate(mods):
            for j, m2 in enumerate(mods):
                if i != j:
                    self.cca_pairs[f"{m1}_{m2}"] = DeepCCACorr(proj_dim, proj_dim, proj_dim, dropout)

        # Final classifier: each modality enriched with (n_active-1) corr features
        # U_i = concat(H_i, corr_{i,j}, corr_{i,k}) → 3 * proj_dim per modality
        n_corr = active - 1   # each modality gets this many corr features concatenated
        in_dim = active * (1 + n_corr) * proj_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(
        self,
        text_input:   Optional[dict]         = None,
        audio_wave:   Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
    ):
        pooled, _ = self.encoders(
            text_input=text_input,
            audio_wave=audio_wave,
            video_frames=video_frames,
        )  # pooled: {modality: [B, proj_dim]}

        # Apply RepVGG blocks
        H = {}
        if self.use_text  and "text"  in pooled: H["text"]  = self.text_repvgg(pooled["text"])
        if self.use_audio and "audio" in pooled: H["audio"] = self.audio_repvgg(pooled["audio"])
        if self.use_video and "video" in pooled: H["video"] = self.video_repvgg(pooled["video"])

        # Compute CCA correlation features
        U_parts = []
        for m1 in self._active_mods:
            parts = [H[m1]]
            for m2 in self._active_mods:
                if m1 != m2:
                    corr = self.cca_pairs[f"{m1}_{m2}"](H[m1], H[m2])
                    parts.append(corr)
            U_parts.append(torch.cat(parts, dim=-1))

        U = torch.cat(U_parts, dim=-1)
        logits = self.classifier(U)
        return logits, U, {}


class MESCAMosei(nn.Module):
    """
    MESCA for CMU-MOSEI (pre-extracted feature dict input).
    """

    def __init__(
        self,
        num_classes: int   = 6,
        d_model:     int   = 256,
        text_model:  str   = "roberta-base",
        use_audio:   bool  = True,
        use_video:   bool  = True,
        dropout:     float = 0.2,
        n_repvgg:    int   = 3,
        **kwargs,
    ):
        super().__init__()
        from baselines.mosei_feature_encoders import MOSEIFeatureEncoders
        self.encoders = MOSEIFeatureEncoders(
            d_model=d_model,
            text_model=text_model,
            use_audio=use_audio,
            use_video=use_video,
            dropout=dropout,
        )
        self.use_audio = use_audio
        self.use_video = use_video
        self.d_model   = d_model

        def _repvgg_stack(n):
            return nn.Sequential(*[RepVGGBlock(d_model, dropout) for _ in range(n)])

        self.text_repvgg  = _repvgg_stack(n_repvgg)
        self.audio_repvgg = _repvgg_stack(n_repvgg) if use_audio else None
        self.video_repvgg = _repvgg_stack(n_repvgg) if use_video else None

        mods = ["text"]
        if use_audio: mods.append("audio")
        if use_video: mods.append("video")
        self._active_mods = mods

        self.cca_pairs: nn.ModuleDict = nn.ModuleDict()
        for m1 in mods:
            for m2 in mods:
                if m1 != m2:
                    self.cca_pairs[f"{m1}_{m2}"] = DeepCCACorr(d_model, d_model, d_model, dropout)

        active = len(mods)
        n_corr = active - 1
        in_dim = active * (1 + n_corr) * d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, batch: dict):
        texts = batch["text"]
        audio = batch.get("audio", None)
        video = batch.get("video", None)

        pooled, _ = self.encoders(texts, audio, video)  # {text, audio?, video?}

        H = {}
        H["text"] = self.text_repvgg(pooled["text"])
        if self.use_audio and "audio" in pooled:
            H["audio"] = self.audio_repvgg(pooled["audio"])
        if self.use_video and "video" in pooled:
            H["video"] = self.video_repvgg(pooled["video"])

        U_parts = []
        for m1 in self._active_mods:
            if m1 not in H:
                continue
            parts = [H[m1]]
            for m2 in self._active_mods:
                if m1 != m2 and m2 in H:
                    corr = self.cca_pairs[f"{m1}_{m2}"](H[m1], H[m2])
                    parts.append(corr)
            U_parts.append(torch.cat(parts, dim=-1))

        U = torch.cat(U_parts, dim=-1)
        logits = self.classifier(U)
        return logits, U, {}
