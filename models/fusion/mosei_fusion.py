

import torch
import torch.nn as nn


class RobertaTextEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", d_model=256, max_len=96):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_len = max_len

        hidden = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        enc = {k: v.to(next(self.model.parameters()).device) for k, v in enc.items()}
        out = self.model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)


class TemporalPool(nn.Module):
    def __init__(self, in_dim, d_model=256):
        super().__init__()
        self.in_dim = in_dim
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B,T,D]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.in_norm(x)
        h = self.proj(x)                     # [B,T,d_model]
        w = self.attn(h).squeeze(-1)         # [B,T]
        w = torch.softmax(w, dim=-1)
        return torch.sum(h * w.unsqueeze(-1), dim=1)  # [B,d_model]


class BaselineFusion(nn.Module):
    """
    CHADO-aligned reliability gating:
      - Per-modality LayerNorm (stabilize scales)
      - Reliability gate per sample computed from feature energy
      - Constrained global gates (sigmoid) so they cannot flip sign
      - Modality dropout

    Cross-dataset robustness:
      - If audio/video features are missing (None), gate is forced to 0 and modality is skipped safely.
    """
    def __init__(
        self,
        num_classes=6,
        d_model=256,
        use_audio=True,
        use_video=True,
        text_model="roberta-base",
        max_text_len=96,
        modality_dropout=0.1,
    ):
        super().__init__()
        self.use_audio = use_audio
        self.use_video = use_video
        self.modality_dropout = float(modality_dropout)

        self.text_enc = RobertaTextEncoder(model_name=text_model, d_model=d_model, max_len=max_text_len)

        # NOTE: keep dims consistent with your MOSEI features
        self.audio_pool = TemporalPool(in_dim=74, d_model=d_model) if use_audio else None
        self.video_pool = TemporalPool(in_dim=35, d_model=d_model) if use_video else None

        # Constrained global gates (0..1), prevents negative scaling collapse
        self.g_text_raw = nn.Parameter(torch.tensor(2.0))  # sigmoid(2)=0.88
        self.g_audio_raw = nn.Parameter(torch.tensor(0.0)) if use_audio else None
        self.g_video_raw = nn.Parameter(torch.tensor(0.0)) if use_video else None

        # Reliability gating network: maps energy -> gate in (0..1)
        self.rel_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        in_fuse = d_model + (d_model if use_audio else 0) + (d_model if use_video else 0)
        self.fuse = nn.Sequential(
            nn.LayerNorm(in_fuse),
            nn.Linear(in_fuse, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, d_model),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def _mod_drop(self, z, p):
        if (not self.training) or p <= 0:
            return z
        keep = (torch.rand(z.size(0), 1, device=z.device) > p).float()
        return z * keep

    def _global_gate(self, raw):
        return torch.sigmoid(raw)

    def _reliability_gate(self, x):
        """
        x: [B,T,D] -> gate: [B,1] in (0..1)
        If x is None (cross-dataset missing modality), return None and caller will replace with zeros.
        """
        if x is None:
            return None
        energy = x.abs().mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        energy = energy.view(-1, 1)                      # [B,1]
        return self.rel_mlp(energy)                      # [B,1]

    def forward(self, batch):
        # Text
        zt = self.text_enc(batch["text"])
        zt = zt * self._global_gate(self.g_text_raw)
        zt = self._mod_drop(zt, self.modality_dropout)
        zs = [zt]

        B = zt.size(0)

        # Audio
        if self.use_audio:
            xa = batch.get("audio", None)  # [B,T,74] or None in cross-dataset
            ra = self._reliability_gate(xa)
            if ra is None:
                # modality missing => force off
                ra = torch.zeros((B, 1), device=zt.device, dtype=zt.dtype)
                za = torch.zeros((B, zt.size(1)), device=zt.device, dtype=zt.dtype)
            else:
                za = self.audio_pool(xa)
                za = za * self._global_gate(self.g_audio_raw) * ra
                za = self._mod_drop(za, self.modality_dropout)
            zs.append(za)

        # Video
        if self.use_video:
            xv = batch.get("video", None)  # [B,T,35] or None in cross-dataset
            rv = self._reliability_gate(xv)
            if rv is None:
                rv = torch.zeros((B, 1), device=zt.device, dtype=zt.dtype)
                zv = torch.zeros((B, zt.size(1)), device=zt.device, dtype=zt.dtype)
            else:
                zv = self.video_pool(xv)
                zv = zv * self._global_gate(self.g_video_raw) * rv
                zv = self._mod_drop(zv, self.modality_dropout)
            zs.append(zv)

        z = torch.cat(zs, dim=-1)
        z = self.fuse(z)
        logits = self.classifier(z)
        return logits, z
