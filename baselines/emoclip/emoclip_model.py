"""
EmoCLIP: Vision-Language Method for Emotion Recognition
Foteinopoulou & Patras, TPAMI 2023 / BMVC 2023
"EmoCLIP: A Vision-Language Method for Zero-Shot Video Facial Expression Recognition"

Key ideas reproduced:
  1. CLIP ViT visual encoder for video frames
  2. CLIP text encoder with emotion-specific prompts
  3. Prompt ensemble: "a photo of a person feeling [emotion]",
     "a video of someone expressing [emotion]", etc.
  4. Fine-tuned: soft prompt tuning (prompt parameters learnable)
  5. Multimodal extension: audio features fused with CLIP visual embedding

Implementation note:
  Uses openai/clip-vit-base-patch32 via HuggingFace CLIPModel.
  Soft prompt tokens are trained; CLIP weights partially frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoModel


# Emotion class → descriptive text prompts (averaged for prompt ensemble)
IEMOCAP_PROMPTS = {
    "neutral": [
        "a person with a neutral expression",
        "someone feeling neutral",
        "a face showing no strong emotion",
        "a person with a calm, neutral demeanor",
    ],
    "happy": [
        "a person feeling happy",
        "someone expressing happiness and joy",
        "a smiling, joyful person",
        "a face showing happiness",
    ],
    "angry": [
        "a person feeling angry",
        "someone expressing anger and frustration",
        "an angry, upset person",
        "a face showing anger",
    ],
    "sad": [
        "a person feeling sad",
        "someone expressing sadness",
        "a sad, unhappy person",
        "a face showing sadness and sorrow",
    ],
}

MELD_PROMPTS = {
    "neutral": ["a person with a neutral expression", "someone feeling calm and neutral"],
    "joy": ["a person feeling joyful and happy", "someone expressing joy and excitement"],
    "surprise": ["a person looking surprised", "someone expressing surprise and shock"],
    "anger": ["a person feeling angry", "someone expressing anger"],
    "sadness": ["a person feeling sad", "someone expressing sadness"],
    "disgust": ["a person feeling disgusted", "someone expressing disgust"],
    "fear": ["a person feeling fearful", "someone expressing fear and worry"],
}


class SoftPromptLayer(nn.Module):
    """Learnable soft prompt tokens prepended to CLIP text features."""
    def __init__(self, n_tokens: int, clip_dim: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, clip_dim) * 0.02)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        # text_features: [N_classes, D] — add soft prompt info via residual
        prompt_summary = self.tokens.mean(0, keepdim=True)  # [1, D]
        return text_features + prompt_summary


class EmoCLIPModel(nn.Module):
    """
    EmoCLIP for IEMOCAP / MELD.

    Architecture:
      CLIP ViT-B/32: visual encoder for video frames
      CLIP text encoder: emotion prompt embeddings (prompt ensemble)
      Soft prompt tuning: learnable prompt tokens
      Classification: cosine similarity between visual feature and prompt embeddings
      Audio fusion: wav2vec2 features fused with CLIP visual output
    """

    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,    # unused — overridden by CLIP
        num_classes: int,
        proj_dim: int = 256,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
        use_gated_fusion: bool = True,
        n_heads: int = 4,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        n_soft_tokens: int = 8,
        class_names: list = None,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.use_audio = use_audio
        self.use_video = use_video
        self.use_text = use_text
        self.num_classes = num_classes

        # ── CLIP backbone ──────────────────────────────────────────────────
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        clip_dim = self.clip.config.projection_dim  # 512 for ViT-B/32

        if freeze_clip:
            # Freeze most of CLIP, only train the last 2 transformer blocks
            for name, p in self.clip.named_parameters():
                p.requires_grad = False
            # Unfreeze last 2 vision transformer blocks
            for name, p in self.clip.vision_model.encoder.layers[-2:].named_parameters():
                p.requires_grad = True
            for name, p in self.clip.visual_projection.named_parameters():
                p.requires_grad = True

        # ── Soft prompt layer ──────────────────────────────────────────────
        self.soft_prompt = SoftPromptLayer(n_soft_tokens, clip_dim)

        # ── Text encoder for utterances (RoBERTa) ─────────────────────────
        if use_text:
            self.text_enc = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_enc.config.hidden_size
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, clip_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # ── Audio encoder ──────────────────────────────────────────────────
        if use_audio:
            self.audio_enc = AutoModel.from_pretrained(audio_model_name)
            audio_dim = self.audio_enc.config.hidden_size
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, clip_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # ── Fusion: combine CLIP visual + text + audio ─────────────────────
        n_mods = sum([use_text, use_audio, use_video])
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * n_mods, proj_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

        # ── Class prompt embeddings (computed once, updated via soft prompts)
        # Will be set per-dataset via set_class_names()
        self.class_names = class_names or []
        self._prompt_embeds = None  # [num_classes, clip_dim]

        # ── Final classifier (on fused representation) ─────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim + num_classes, proj_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

        # Store for prompt building
        self._clip_dim = clip_dim
        self._proj_dim = proj_dim

    def _build_prompt_embeds(self, class_names: list, prompts_dict: dict, device):
        """Build averaged prompt embeddings for each class."""
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        embeds = []
        for cname in class_names:
            templates = prompts_dict.get(cname, [f"a person feeling {cname}"])
            inputs = tokenizer(templates, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_feats = self.clip.get_text_features(**inputs)   # [n_templates, clip_dim]
                avg = F.normalize(text_feats.mean(0), dim=-1)        # [clip_dim]
            embeds.append(avg)
        return torch.stack(embeds, dim=0)  # [num_classes, clip_dim]

    def _get_prompt_embeds(self, device):
        if self._prompt_embeds is None or self._prompt_embeds.device != device:
            # Detect dataset from num_classes
            if self.num_classes == 4:
                names = ["neutral", "happy", "angry", "sad"]
                prompts = IEMOCAP_PROMPTS
            else:
                names = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
                prompts = MELD_PROMPTS
            self._prompt_embeds = self._build_prompt_embeds(names, prompts, device)
        return self._prompt_embeds.to(device)

    def forward(self, text_input=None, audio_wave=None, video_frames=None, modality_mask=None):
        device = next(self.parameters()).device
        feats = []

        # ── CLIP visual encoding ───────────────────────────────────────────
        if self.use_video and video_frames is not None:
            B, nF, C, H, W = video_frames.shape
            # CLIP expects standard ImageNet normalisation — already done in dataset
            pixel_values = video_frames.reshape(B * nF, C, H, W)
            vis_feats = self.clip.get_image_features(pixel_values=pixel_values)  # [B*nF, clip_dim]
            vis_feats = F.normalize(vis_feats, dim=-1).reshape(B, nF, -1).mean(1)  # [B, clip_dim]
            feats.append(vis_feats)
        else:
            vis_feats = None

        # ── Text utterance encoding (RoBERTa) ─────────────────────────────
        if self.use_text and text_input is not None:
            out = self.text_enc(**text_input)
            t = self.text_proj(out.last_hidden_state[:, 0])
            feats.append(t)

        # ── Audio encoding ─────────────────────────────────────────────────
        if self.use_audio and audio_wave is not None:
            out = self.audio_enc(audio_wave)
            a = self.audio_proj(out.last_hidden_state.mean(1))
            feats.append(a)

        # ── Fuse modalities ────────────────────────────────────────────────
        fused_mm = self.fusion(torch.cat(feats, dim=-1)) if len(feats) > 1 else feats[0]

        # ── Prompt-based similarity scores ────────────────────────────────
        prompt_embeds = self._get_prompt_embeds(device)          # [C, clip_dim]
        prompt_embeds = self.soft_prompt(prompt_embeds)          # apply soft prompts
        if vis_feats is not None:
            sim_scores = torch.matmul(
                F.normalize(vis_feats, dim=-1),
                F.normalize(prompt_embeds, dim=-1).T
            )  # [B, C]
        else:
            # Fall back to uniform similarity if no video
            sim_scores = torch.zeros(fused_mm.size(0), self.num_classes, device=device)

        # ── Final classification: concat fused + sim scores ────────────────
        logits = self.classifier(torch.cat([fused_mm, sim_scores], dim=-1))

        return logits, fused_mm, {}

    def loss(self, logits, labels, fused, aux, label_smoothing=0.0, class_weight=None):
        ce = F.cross_entropy(logits, labels, weight=class_weight, label_smoothing=label_smoothing)
        return ce, ce
