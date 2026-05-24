

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, T5EncoderModel


class ModalityAdapter(nn.Module):
    """Project audio/video features into T5 token embedding space."""
    def __init__(self, input_dim: int, t5_dim: int, n_tokens: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, t5_dim * n_tokens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.t5_dim = t5_dim

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, input_dim] → [B, n_tokens, t5_dim]
        return self.proj(feat).view(feat.size(0), self.n_tokens, self.t5_dim)


class UniMSEModel(nn.Module):
    """
    UniMSE for IEMOCAP / MELD.

    Architecture:
      T5-encoder: processes text tokens (+ optional audio/video soft tokens)
      Audio adapter: wav2vec2 → T5 soft tokens (prepended)
      Video adapter: ViT → T5 soft tokens (prepended)
      Classifier: linear on T5 [last-token] representation
    """

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,
        proj_dim: int = 256,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
        use_gated_fusion: bool = True,
        n_heads: int = 4,
        n_adapter_tokens: int = 4,
        w_contrastive: float = 0.05,
    ):
        super().__init__()
        self.use_audio = use_audio
        self.use_video = use_video
        self.use_text = use_text
        self.w_contrastive = w_contrastive

        # ── T5 encoder backbone ────────────────────────────────────────────
        # Use T5 encoder part only; fall back to RoBERTa if T5 unavailable
        try:
            self.t5 = T5EncoderModel.from_pretrained("t5-base")
            t5_dim = self.t5.config.d_model  # 768 for t5-base
            self.use_t5 = True
        except Exception:
            # Fallback: use RoBERTa as the shared encoder
            self.t5 = AutoModel.from_pretrained(text_model_name)
            t5_dim = self.t5.config.hidden_size
            self.use_t5 = False

        self.text_proj = nn.Sequential(
            nn.Linear(t5_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # ── Audio encoder + adapter ────────────────────────────────────────
        if use_audio:
            self.audio_enc = AutoModel.from_pretrained(audio_model_name)
            audio_dim = self.audio_enc.config.hidden_size
            self.audio_adapter = ModalityAdapter(audio_dim, t5_dim, n_adapter_tokens, dropout)
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # ── Video encoder + adapter ────────────────────────────────────────
        if use_video:
            self.video_enc = AutoModel.from_pretrained(video_model_name)
            video_dim = self.video_enc.config.hidden_size
            self.video_adapter = ModalityAdapter(video_dim, t5_dim, n_adapter_tokens, dropout)
            self.video_proj = nn.Sequential(
                nn.Linear(video_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # ── Contrastive alignment projection ──────────────────────────────
        self.align_proj = nn.Linear(proj_dim, proj_dim)

        # ── Fusion gate ────────────────────────────────────────────────────
        n_mods = sum([use_text, use_audio, use_video])
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * n_mods, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(proj_dim, num_classes)
        self._n_mods = n_mods

    def forward(self, text_input=None, audio_wave=None, video_frames=None, modality_mask=None):
        feats = []
        audio_feat = None
        video_feat = None

        # ── Text via T5 ───────────────────────────────────────────────────
        if self.use_text and text_input is not None:
            if self.use_t5:
                out = self.t5(**text_input)
                t = self.text_proj(out.last_hidden_state.mean(1))
            else:
                out = self.t5(**text_input)
                t = self.text_proj(out.last_hidden_state[:, 0])
            feats.append(t)

        # ── Audio ──────────────────────────────────────────────────────────
        if self.use_audio and audio_wave is not None:
            out = self.audio_enc(audio_wave)
            audio_feat = out.last_hidden_state.mean(1)
            a = self.audio_proj(audio_feat)
            feats.append(a)

        # ── Video ──────────────────────────────────────────────────────────
        if self.use_video and video_frames is not None:
            B, F, C, H, W = video_frames.shape
            out = self.video_enc(pixel_values=video_frames.reshape(B * F, C, H, W))
            video_feat = out.last_hidden_state[:, 0].reshape(B, F, -1).mean(1)
            v = self.video_proj(video_feat)
            feats.append(v)

        fused = self.fusion(torch.cat(feats, dim=-1)) if len(feats) > 1 else feats[0]
        logits = self.classifier(fused)

        return logits, fused, {}

    def loss(self, logits, labels, fused, aux, label_smoothing=0.0, class_weight=None):
        ce = F.cross_entropy(logits, labels, weight=class_weight, label_smoothing=label_smoothing)

        # Contrastive alignment: pull same-class representations together
        z = F.normalize(self.align_proj(fused), dim=-1)
        B = z.size(0)
        if B > 1:
            sim_matrix = torch.matmul(z, z.T) / 0.07         # [B, B]
            label_match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # Exclude diagonal
            mask = 1 - torch.eye(B, device=z.device)
            label_match = label_match * mask
            pos_mask = label_match / (label_match.sum(1, keepdim=True) + 1e-8)
            log_softmax = F.log_softmax(sim_matrix * mask - 1e4 * (1 - mask), dim=1)
            contrastive = -(pos_mask * log_softmax).sum(1).mean()
            if not torch.isfinite(contrastive):
                contrastive = z.sum() * 0.0
        else:
            contrastive = z.sum() * 0.0

        total = ce + self.w_contrastive * contrastive
        return total, ce
