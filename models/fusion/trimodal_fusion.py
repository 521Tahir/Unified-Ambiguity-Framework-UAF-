from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class BaselineOutputs:
    logits: torch.Tensor
    fused: torch.Tensor
    gate: Optional[torch.Tensor]  # [B, M]


class GatedFusion(nn.Module):
    def __init__(self, dim: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate = nn.Linear(dim * num_modalities, num_modalities)

    def forward(self, embs: torch.Tensor):
        # embs: [B, M, D]
        B, M, D = embs.shape
        x = embs.reshape(B, M * D)
        w = self.gate(x)               # [B,M]
        g = torch.softmax(w, dim=-1)   # [B,M]
        fused = (embs * g.unsqueeze(-1)).sum(dim=1)  # [B,D]
        return fused, g


class TriModalBaseline(nn.Module):
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
    ):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video

        self.modalities = []

        if use_text:
            self.modalities.append("text")
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            t_dim = self.text_encoder.config.hidden_size
            self.text_proj = nn.Sequential(nn.Linear(t_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        else:
            self.text_encoder = None
            self.text_proj = None

        if use_audio:
            self.modalities.append("audio")
            self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
            a_dim = self.audio_encoder.config.hidden_size
            self.audio_proj = nn.Sequential(nn.Linear(a_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        else:
            self.audio_encoder = None
            self.audio_proj = None

        if use_video:
            self.modalities.append("video")
            self.video_encoder = AutoModel.from_pretrained(video_model_name)
            v_dim = self.video_encoder.config.hidden_size
            self.video_proj = nn.Sequential(nn.Linear(v_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        else:
            self.video_encoder = None
            self.video_proj = None

        self.num_modalities = len(self.modalities)
        assert self.num_modalities >= 1, "At least one modality must be enabled."

        self.fusion = GatedFusion(proj_dim, self.num_modalities) if (use_gated_fusion and self.num_modalities > 1) else None
        self.classifier = nn.Linear(proj_dim, num_classes)

    def freeze_encoders(self, freeze_text: bool, freeze_audio: bool, freeze_video: bool):
        def set_req(m, req: bool):
            if m is None:
                return
            for p in m.parameters():
                p.requires_grad = req

        if self.text_encoder is not None:
            set_req(self.text_encoder, not freeze_text)
        if self.audio_encoder is not None:
            set_req(self.audio_encoder, not freeze_audio)
        if self.video_encoder is not None:
            set_req(self.video_encoder, not freeze_video)

    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        audio_wave: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,  # [B,M]
    ):
        embs = []

        for m in self.modalities:
            if m == "text":
                out = self.text_encoder(**text_input)
                pooled = out.last_hidden_state[:, 0]  # CLS
                embs.append(self.text_proj(pooled))

            elif m == "audio":
                out = self.audio_encoder(input_values=audio_wave)
                pooled = out.last_hidden_state.mean(dim=1)
                embs.append(self.audio_proj(pooled))

            elif m == "video":
                # video_frames: [B,T,3,H,W] -> ViT expects [B,3,H,W]
                B, T, C, H, W = video_frames.shape
                frames = video_frames.reshape(B * T, C, H, W)
                out = self.video_encoder(pixel_values=frames)
                pooled = out.last_hidden_state[:, 0]        # [B*T, D]
                pooled = pooled.reshape(B, T, -1).mean(dim=1)
                embs.append(self.video_proj(pooled))

        embs = torch.stack(embs, dim=1)  # [B,M,D]

        if modality_mask is not None:
            embs = embs * modality_mask.unsqueeze(-1)

        gate = None
        if self.fusion is not None:
            fused, gate = self.fusion(embs)
        else:
            fused = embs.mean(dim=1)

        logits = self.classifier(fused)
        return logits, fused, gate
