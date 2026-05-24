"""
AER-LLM: Ambiguity-Enhanced Representation for LLM-based Emotion Recognition
Hong et al., 2025 — "Ambiguity-Aware Emotion Recognition with Large Language Models"

Key contributions reproduced:
  1. RoBERTa-large as the LLM backbone (instruction-tuned style fine-tuning)
  2. Ambiguity-Enhanced Representation (AER): dual-branch entropy-aware contrastive learning
     - Clear branch: low-entropy samples (confident predictions)
     - Ambiguous branch: high-entropy samples weighted up
  3. Multimodal extension: cross-attention fusion of audio/video over LLM text features
  4. Ambiguity-Calibrated Loss: entropy-scaled cross-entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from baselines.shared_encoders import TrimodalEncoders


class AmbiguityContrastiveHead(nn.Module):
    """
    AER contrastive head: separates clear vs ambiguous representations.
    Pushes apart embeddings of ambiguous and unambiguous samples in feature space.
    """
    def __init__(self, d_model: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), dim=-1)

    @staticmethod
    def ambiguity_contrastive_loss(
        z: torch.Tensor,
        entropy: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Contrastive loss: clear samples (low entropy) should be close together,
        far from ambiguous samples (high entropy).
        entropy: [B] per-sample entropy of predicted distribution.
        """
        B = z.size(0)
        if B < 2:
            return z.sum() * 0.0

        # Split into clear vs ambiguous halves by median entropy
        median_ent = entropy.median()
        clear_mask = entropy < median_ent
        ambig_mask = ~clear_mask

        if clear_mask.sum() < 1 or ambig_mask.sum() < 1:
            return z.sum() * 0.0

        z_clear = z[clear_mask].mean(0, keepdim=True)   # [1, D]
        z_ambig = z[ambig_mask].mean(0, keepdim=True)   # [1, D]

        # Simple push-apart loss
        sim = F.cosine_similarity(z_clear, z_ambig)     # scalar
        loss = F.relu(sim + 0.5)                         # push similarity < -0.5
        return loss


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention: audio/video attend over LLM text features."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_av = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_feat: torch.Tensor,   # [B, D] — LLM representation
        av_feat: torch.Tensor,     # [B, D] — audio/video pooled
    ) -> torch.Tensor:
        # Treat pooled vectors as 1-token sequences
        t = text_feat.unsqueeze(1)   # [B, 1, D]
        a = av_feat.unsqueeze(1)     # [B, 1, D]
        fused, _ = self.attn_av(query=t, key=a, value=a)
        return self.norm(t + self.dropout(fused)).squeeze(1)  # [B, D]


class AERLLMModel(nn.Module):
    """
    AER-LLM for IEMOCAP / MELD (single-label classification).

    Architecture:
      RoBERTa-large (LLM backbone)
        └─ AER: entropy-weighted ambiguity contrastive head
      wav2vec2 (audio) + ViT (video)
        └─ cross-modal attention over LLM features
      concat [text_fused, audio_pooled, video_pooled] → classifier
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
        contrastive_dim: int = 128,
        w_contrastive: float = 0.1,
        w_entropy: float = 0.05,
    ):
        super().__init__()
        self.use_audio = use_audio
        self.use_video = use_video
        self.use_text = use_text
        self.w_contrastive = w_contrastive
        self.w_entropy = w_entropy

        # ── LLM backbone (RoBERTa-base) ──────────────────────────────────
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        text_hidden = self.text_enc.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # ── Audio encoder ──────────────────────────────────────────────────
        if use_audio:
            self.audio_enc = AutoModel.from_pretrained(audio_model_name)
            audio_hidden = self.audio_enc.config.hidden_size
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_hidden, proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # ── Video encoder ──────────────────────────────────────────────────
        if use_video:
            self.video_enc = AutoModel.from_pretrained(video_model_name)
            video_hidden = self.video_enc.config.hidden_size
            self.video_proj = nn.Sequential(
                nn.Linear(video_hidden, proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # ── Cross-modal attention fusion ───────────────────────────────────
        n_mods = sum([use_text, use_audio, use_video])
        if use_text and (use_audio or use_video):
            self.cross_attn = CrossModalAttentionFusion(proj_dim, n_heads, dropout)

        # ── Ambiguity-Enhanced Representation head ─────────────────────────
        self.aer_head = AmbiguityContrastiveHead(proj_dim, contrastive_dim)

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * n_mods, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )
        self._n_mods = n_mods

    def encode(self, text_input=None, audio_wave=None, video_frames=None):
        feats = []

        if self.use_text and text_input is not None:
            out = self.text_enc(**text_input)
            t = self.text_proj(out.last_hidden_state[:, 0])  # [CLS] token
            feats.append(t)
        else:
            t = None

        if self.use_audio and audio_wave is not None:
            out = self.audio_enc(audio_wave)
            a = self.audio_proj(out.last_hidden_state.mean(1))
            feats.append(a)
        else:
            a = None

        if self.use_video and video_frames is not None:
            B, F, C, H, W = video_frames.shape
            out = self.video_enc(pixel_values=video_frames.reshape(B * F, C, H, W))
            v = self.video_proj(out.last_hidden_state[:, 0].reshape(B, F, -1).mean(1))
            feats.append(v)
        else:
            v = None

        return feats, t, a, v

    def forward(self, text_input=None, audio_wave=None, video_frames=None, modality_mask=None):
        feats, t, a, v = self.encode(text_input, audio_wave, video_frames)

        # Cross-modal: fuse audio/video into text space
        if t is not None and (a is not None or v is not None):
            av = a if v is None else (v if a is None else (a + v) / 2)
            t_fused = self.cross_attn(t, av)
            # Replace text feature with fused version
            feats[0] = t_fused
            main_feat = t_fused
        else:
            main_feat = feats[0] if feats else t

        fused = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        logits = self.classifier(fused)

        # AER contrastive projection for aux loss
        z_proj = self.aer_head(main_feat)

        return logits, z_proj, {}

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z_proj: torch.Tensor,
        aux=None,
        label_smoothing: float = 0.0,
        class_weight=None,
    ) -> tuple:
        ce = F.cross_entropy(logits, labels, weight=class_weight, label_smoothing=label_smoothing)

        # Entropy-calibrated loss: up-weight ambiguous samples
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B]

        # AER contrastive loss
        contrastive = AmbiguityContrastiveHead.ambiguity_contrastive_loss(z_proj, entropy)

        # Entropy-weighted CE: ambiguous samples get higher loss weight
        w = 1.0 + self.w_entropy * entropy.detach()
        w = w / w.mean()
        per_sample = F.cross_entropy(logits, labels, weight=class_weight,
                                     label_smoothing=label_smoothing, reduction="none")
        entropy_ce = (w * per_sample).mean()

        total = (1 - self.w_contrastive) * entropy_ce + self.w_contrastive * contrastive
        return total, ce
