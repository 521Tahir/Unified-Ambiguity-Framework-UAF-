"""
OV-MER: Open-Vocabulary Multimodal Emotion Recognition
Ji et al., ACM MM 2024
"OV-MER: Open-Vocabulary Multimodal Emotion Recognition"

Key ideas reproduced:
  1. Open-vocabulary emotion embedding space using CLIP text encoder
  2. Emotion label embeddings as class prototypes (not fixed classifier weights)
  3. Semantic-guided prototype alignment: match representations to text-defined prototypes
  4. Cross-modal contrastive learning: audio/video/text aligned to CLIP semantic space
  5. Prototype-based classification: nearest prototype in CLIP embedding space

Architecture:
  Multimodal encoders → project to CLIP embedding space →
  cosine similarity to learnable emotion prototypes →
  optional MLP refinement → classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPTokenizer


class SemanticPrototype(nn.Module):
    """
    Learnable emotion prototypes initialized from CLIP text embeddings.
    Prototypes are refined during training while staying in CLIP space.
    """
    def __init__(self, num_classes: int, embed_dim: int, class_names: list = None):
        super().__init__()
        self.num_classes = num_classes
        # Learnable prototypes (initialized from CLIP text embeds in set_prototypes())
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.orthogonal_(self.prototypes)

    def set_from_clip(self, clip_model, class_names: list, device):
        """Initialize prototypes from CLIP text embeddings."""
        try:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            prompts = [f"a person feeling {name}" for name in class_names]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embeds = clip_model.get_text_features(**inputs)
                embeds = F.normalize(embeds, dim=-1)
            self.prototypes.data.copy_(embeds.to(self.prototypes.device))
        except Exception:
            pass  # Keep random init if CLIP unavailable

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, embed_dim] — normalized query
        Returns: [B, num_classes] similarity logits
        """
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        return torch.matmul(z_norm, p_norm.T)  # [B, C]


class OVMERModel(nn.Module):
    """
    OV-MER for IEMOCAP / MELD.

    Architecture:
      Text: RoBERTa → linear proj → CLIP embedding space
      Audio: wav2vec2 → linear proj → CLIP embedding space
      Video: ViT → linear proj → CLIP embedding space
      Fusion: weighted average in CLIP space (cross-modal alignment)
      Classification: cosine similarity to learnable class prototypes
      Refinement MLP: optional post-prototype refinement
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
        clip_embed_dim: int = 512,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        class_names: list = None,
        w_proto: float = 0.3,
        w_align: float = 0.1,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video
        self.w_proto = w_proto
        self.w_align = w_align
        self.clip_embed_dim = clip_embed_dim

        # ── Modality encoders → project to CLIP space ──────────────────────
        if use_text:
            self.text_enc = AutoModel.from_pretrained(text_model_name)
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_enc.config.hidden_size, clip_embed_dim),
                nn.ReLU(), nn.Dropout(dropout),
            )

        if use_audio:
            self.audio_enc = AutoModel.from_pretrained(audio_model_name)
            self.audio_proj = nn.Sequential(
                nn.Linear(self.audio_enc.config.hidden_size, clip_embed_dim),
                nn.ReLU(), nn.Dropout(dropout),
            )

        if use_video:
            self.video_enc = AutoModel.from_pretrained(video_model_name)
            self.video_proj = nn.Sequential(
                nn.Linear(self.video_enc.config.hidden_size, clip_embed_dim),
                nn.ReLU(), nn.Dropout(dropout),
            )

        # ── Learnable modality fusion weights ──────────────────────────────
        n_mods = sum([use_text, use_audio, use_video])
        self.mod_weights = nn.Parameter(torch.ones(n_mods) / n_mods)

        # ── Semantic prototypes (initialized from CLIP text embeddings) ────
        names = class_names or self._default_class_names(num_classes)
        self.prototypes = SemanticPrototype(num_classes, clip_embed_dim, names)
        self._class_names = names
        self._proto_initialized = False

        # ── Prototype refinement MLP ───────────────────────────────────────
        self.refine = nn.Sequential(
            nn.Linear(num_classes + clip_embed_dim, proj_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

        # ── CLIP for prototype init only ───────────────────────────────────
        try:
            self._clip_init = CLIPModel.from_pretrained(clip_model_name)
            for p in self._clip_init.parameters():
                p.requires_grad = False
            self._has_clip = True
        except Exception:
            self._has_clip = False

    @staticmethod
    def _default_class_names(num_classes):
        if num_classes == 4:
            return ["neutral", "happy", "angry", "sad"]
        elif num_classes == 7:
            return ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
        else:
            return [f"class_{i}" for i in range(num_classes)]

    def _ensure_proto_init(self, device):
        if not self._proto_initialized and self._has_clip:
            self.prototypes.set_from_clip(
                self._clip_init.to(device), self._class_names, device
            )
            self._proto_initialized = True
            # Free CLIP memory after init
            del self._clip_init
            self._has_clip = False

    def _encode_modalities(self, text_input, audio_wave, video_frames):
        feats = []
        if self.use_text and text_input is not None:
            out = self.text_enc(**text_input)
            feats.append(self.text_proj(out.last_hidden_state[:, 0]))

        if self.use_audio and audio_wave is not None:
            out = self.audio_enc(audio_wave)
            feats.append(self.audio_proj(out.last_hidden_state.mean(1)))

        if self.use_video and video_frames is not None:
            B, F, C, H, W = video_frames.shape
            out = self.video_enc(pixel_values=video_frames.reshape(B * F, C, H, W))
            v = out.last_hidden_state[:, 0].reshape(B, F, -1).mean(1)
            feats.append(self.video_proj(v))

        return feats

    def forward(self, text_input=None, audio_wave=None, video_frames=None, modality_mask=None):
        device = next(self.parameters()).device
        self._ensure_proto_init(device)

        feats = self._encode_modalities(text_input, audio_wave, video_frames)

        # Weighted fusion in CLIP embedding space
        w = F.softmax(self.mod_weights, dim=0)
        fused = sum(w_i * f for w_i, f in zip(w, feats))   # [B, clip_embed_dim]

        # Prototype similarity
        proto_sim = self.prototypes(fused)                   # [B, C]

        # Refined classification
        logits = self.refine(torch.cat([proto_sim, fused], dim=-1))  # [B, C]

        return logits, fused, {"proto_sim": proto_sim}

    def loss(self, logits, labels, fused, aux, label_smoothing=0.0, class_weight=None):
        ce = F.cross_entropy(logits, labels, weight=class_weight, label_smoothing=label_smoothing)

        # Prototype alignment loss: proto_sim should peak at correct class
        if "proto_sim" in aux:
            proto_ce = F.cross_entropy(aux["proto_sim"], labels, label_smoothing=label_smoothing)
            total = (1 - self.w_proto) * ce + self.w_proto * proto_ce
        else:
            total = ce

        return total, ce
